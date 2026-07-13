"""SFT v5 训练入口：RS/EVENT 两问 OPSD + torchrun sequence padding。

核心训练流程：

1. DataLoader 每次读取若干条 route sequence；
2. collate 阶段只做本 rank local padding，主训练进程再 all-reduce 得到 global T；
3. 每个有效 frame 先让 student 自由回答 Q1；
4. Q1 RS 正确才进入 Q2，否则本帧结束，下一有效帧恢复 GT RS + RE；
5. teacher 关闭 LoRA，读取 privileged prompt，在同一批 student rollout token 上给
   full-vocabulary logits；
6. student/teacher logits 做 forward-KL，梯度只回到 LoRA student。

注意：v5 使用 torchrun 多进程 + 手动梯度 all-reduce，不把模型包进
DistributedDataParallel wrapper。原因是 OPSD 的 Q2 是否触发取决于每个 rank 的
student Q1 输出，rank 之间 forward 次数天然不一致；DDP wrapper 的 forward hook
会在这种分支生成里产生 unmatched collective，导致 NCCL watchdog 卡死。

`--check` 模式不加载模型，只检查 dataset / DDP padding / prompt / memory 状态机。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_AVAILABLE = True
except Exception:
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False

from qwen3vl_local.sft_v2.train import (  # noqa: E402
    _is_vision_module_name,
    load_model_with_lora,
    make_scheduler,
)
from qwen3vl_local.sft_v3.train import (  # noqa: E402
    KVState,
    _append_token_ids,
    _append_token_ids_with_logits,
    _append_user_turn,
    _collect_images_from_messages,
    _clone_kv_state,
    _kv_start_state,
    _student_generate_kv,
    _teacher_eval_context,
    _teacher_start_state,
)
from qwen3vl_local.sft_v5 import DATASET_VERSION  # noqa: E402
from qwen3vl_local.sft_v5.labels import (  # noqa: E402
    RS_LABEL_TO_OPTION,
    RS_OPTION_DESCRIPTIONS,
    EventTarget,
    RSTarget,
    option_for_event,
    resolve_event_target,
)
from qwen3vl_local.sft_v5.prompts import (  # noqa: E402
    SYSTEM_PROMPT_V5,
    Memory,
    build_q1_student_prompt,
    build_q1_teacher_prompt,
    build_q2_student_prompt,
    build_q2_teacher_prompt,
    loss_weights_q1,
    loss_weights_q2,
    parse_q1_output,
    parse_q2_output,
    reset_memory_for_frame,
    target_spans_q1,
    target_spans_q2,
    update_memory_after_q1,
    update_memory_after_q2,
)


@dataclass
class FrameRow:
    """训练时使用的单帧轻量对象。"""

    frame_id: int
    history_rgb_paths: List[str]
    weather_text: str
    ego_to_goal_xy: Optional[Tuple[float, float]]
    rs_label: str
    rs_option: str
    event_label: str
    event_code: str
    abnormal: bool
    event_option_map: Dict[str, str]
    regular_event_codes: List[str]
    raw: Dict[str, Any]


@dataclass
class SequenceRow:
    """一条 route sequence。"""

    scenario: str
    route_id: str
    run_dir: str
    split: str
    frames: List[FrameRow]


@dataclass
class Q1GroupedRolloutResult:
    """Q1 grouped rollout 的输出和真实并行统计。

    这里把 `grouped` 和 `batched` 明确拆开：
    - grouped：进入了同一 timestep 多 frame 的 rollout 优化路径；
    - batched：某个分组 size>=2，真的让 Qwen 在 batch 维同时 forward。
    这两个概念不能混用，否则 TensorBoard 会高估 `QWEN_BATCH_SIZE>1` 的收益。
    """

    rollouts: List[Tuple[Optional[KVState], str, Optional[KVState], torch.Tensor]]
    input_lengths: List[int]
    group_sizes: List[int]
    batched_group_sizes: List[int]
    singleton_groups: int
    batched_groups: int
    batched_frames: int
    length_histogram: Dict[int, int]
    length_seconds: float
    total_seconds: float


@dataclass
class Q2GroupedRolloutResult:
    """Q2 grouped rollout 的输出和真实并行统计。

    Q2 与 Q1 一样，padded batched KV 只用于 no_grad student rollout 采样；
    真正计算 KL 时会在 `_run_frame` 里按单样本重建精确 KV，避免把 padding
    位置写进训练语义。`input_lengths` / `length_histogram` 只用于审计 padding 压力。
    """

    rollouts: List[Optional[Tuple[Optional[KVState], str, torch.Tensor]]]
    input_lengths: List[int]
    group_sizes: List[int]
    batched_group_sizes: List[int]
    singleton_groups: int
    batched_groups: int
    batched_frames: int
    length_histogram: Dict[int, int]
    length_seconds: float
    total_seconds: float


class RouteSequenceDataset(Dataset):
    """读取 build_dataset.py 生成的 sequence_index.jsonl。"""

    def __init__(self, path: pathlib.Path, *, max_routes: int = 0, max_frames_per_route: int = 0):
        self.path = pathlib.Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"sequence index not found: {self.path}")
        rows: List[SequenceRow] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                frames: List[FrameRow] = []
                for fr in obj.get("frames", []):
                    ego_to_goal_xy = _parse_goal_xy(fr.get("ego_to_goal_xy"))
                    if ego_to_goal_xy is None:
                        # v5 的导航输入合同要求每个训练/probe frame 都有当前帧
                        # EGO_TO_GOAL_XY。旧 sequence_index 可能没有该字段；这里直接
                        # 跳过旧 frame，避免 prompt 中继续出现 UNKNOWN。
                        continue
                    # build_dataset 已经把原始 annotation 压成训练需要的最小字段；
                    # raw 仍完整保留 frame row，供多标签 EVENT 动态真值和 probe 审计回查。
                    frames.append(
                        FrameRow(
                            frame_id=int(fr["frame_id"]),
                            history_rgb_paths=[str(x) for x in fr.get("history_rgb_paths", [])],
                            weather_text=str(fr.get("weather_text", "")),
                            ego_to_goal_xy=ego_to_goal_xy,
                            rs_label=str(fr.get("rs_label", "R1")),
                            rs_option=str(fr.get("rs_option", "A")),
                            event_label=str(fr.get("event_label", "RE")),
                            event_code=str(fr.get("event_code", "R-E1")),
                            abnormal=bool(fr.get("abnormal", False)),
                            event_option_map={str(k): str(v) for k, v in (fr.get("event_option_map") or {}).items()},
                            regular_event_codes=[str(x) for x in fr.get("regular_event_codes", [])],
                            raw=fr,
                        )
                    )
                if max_frames_per_route > 0:
                    frames = frames[:max_frames_per_route]
                if not frames:
                    continue
                rows.append(
                    SequenceRow(
                        scenario=str(obj.get("scenario", "")),
                        route_id=str(obj.get("route_id", "")),
                        run_dir=str(obj.get("run_dir", "")),
                        split=str(obj.get("split", "")),
                        frames=frames,
                    )
                )
                if max_routes > 0 and len(rows) >= max_routes:
                    break
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> SequenceRow:
        return self.rows[idx]


class LengthBalancedDistributedSampler(Sampler[int]):
    """按 route frame 数均衡各 rank 负载的 distributed sampler。

    普通 `DistributedSampler` 只保证每个 rank 拿到的 route 数接近一致；但 v5 的
    训练耗时更接近“route 内有效 frame 数 × Q1/Q2 生成长度”。如果某个 rank 恰好
    抽到一批长 route，其它 rank 会在 batch 末尾的 `_ddp_sum_int` 或 optimizer step
    等 collective 前空等。这个 sampler 在**每个 rank 样本数一致**的前提下，按 route
    长度贪心分配，尽量让每个 rank 的总 frame 数接近。

    保持每个 rank 样本数一致非常重要：DataLoader batch 数如果不一致，训练 loop 里的
    all-reduce 调用次数就会不一致，最终仍可能卡住。
    """

    def __init__(
        self,
        dataset: RouteSequenceDataset,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) / max(1, self.num_replicas))) if len(self.dataset) else 0
        self.total_size = self.num_samples * max(1, self.num_replicas)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """与 PyTorch DistributedSampler 同款接口，保证每个 epoch 排布不同。"""

        self.epoch = int(epoch)

    def __iter__(self) -> Iterable[int]:
        n = len(self.dataset)
        if n == 0:
            return iter([])
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            # 先随机打散，再按长度降序贪心。这样长 route 的相对顺序每个 epoch 会变，
            # 但仍能把长短 route 均匀摊到各 rank。
            perm = torch.randperm(n, generator=generator).tolist()
        else:
            perm = list(range(n))
        lengths = {idx: len(self.dataset.rows[idx].frames) for idx in range(n)}
        # sort 是稳定排序；前面的随机 perm 会成为同长度 route 的 tie-breaker。
        ordered = sorted(perm, key=lambda idx: lengths[idx], reverse=True)
        rank_indices: List[List[int]] = [[] for _ in range(self.num_replicas)]
        rank_loads = [0 for _ in range(self.num_replicas)]
        for idx in ordered:
            # 只在还有样本名额的 rank 中选择当前总 frame 数最小者，确保最终每个
            # rank 的样本数都不超过 num_samples。
            candidates = [r for r in range(self.num_replicas) if len(rank_indices[r]) < self.num_samples]
            if not candidates:
                break
            target = min(candidates, key=lambda r: (rank_loads[r], len(rank_indices[r]), r))
            rank_indices[target].append(idx)
            rank_loads[target] += lengths[idx]
        # 如果数据量不能整除 world_size，尾部 rank 需要补样本来保证 batch 数一致。
        # 补样本优先使用全局最短 route，而不是复制本 rank 自己的 route；否则某个
        # rank 只拿到一条超长 route 时，padding 会把这条长 route 重复多次，反而让
        # 负载更不均衡。
        global_short_first = sorted(range(n), key=lambda idx: (lengths[idx], idx))
        for r in range(self.num_replicas):
            if not rank_indices[r]:
                rank_indices[r].append(global_short_first[r % len(global_short_first)])
            fill_pos = 0
            while len(rank_indices[r]) < self.num_samples:
                rank_indices[r].append(global_short_first[fill_pos % len(global_short_first)])
                fill_pos += 1
        return iter(rank_indices[self.rank][: self.num_samples])

    def local_epoch_frame_count(self) -> int:
        """返回当前 epoch 本 rank sampler 分到的 frame 数，用于启动日志审计。"""

        return sum(len(self.dataset.rows[idx].frames) for idx in list(iter(self)))


def _global_max_int(value: int) -> int:
    """torchrun 多进程下对一个 int 做 all_reduce max；单进程直接返回。"""

    if dist.is_available() and dist.is_initialized():
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return int(tensor.item())
    return int(value)


def _global_min_int(value: int) -> int:
    """torchrun 多进程下对一个 int 做 all_reduce min；单进程直接返回。"""

    if dist.is_available() and dist.is_initialized():
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return int(tensor.item())
    return int(value)


def collate_route_sequences(batch: List[SequenceRow]) -> Dict[str, Any]:
    """按本 rank 的最长 sequence 做 local padding。

    这里不读图片，只把 frame 对象 pad 成 None。真正的图像读取放在训练 step 中，
    避免 DataLoader worker 触碰 PIL/CUDA 状态。分布式 global max 在主训练进程里
    计算，避免 `num_workers>0` 时在 worker 进程触碰 distributed runtime。
    """

    local_max = max((len(row.frames) for row in batch), default=0)
    padded: List[List[Optional[FrameRow]]] = []
    valid: List[List[bool]] = []
    for row in batch:
        frames: List[Optional[FrameRow]] = list(row.frames)
        mask = [True] * len(frames)
        while len(frames) < local_max:
            frames.append(None)
            mask.append(False)
        padded.append(frames)
        valid.append(mask)
    return {
        "routes": batch,
        "frame_rows": padded,
        "valid_mask": torch.tensor(valid, dtype=torch.bool) if valid else torch.zeros((0, local_max), dtype=torch.bool),
        "max_T_local": local_max,
    }


def pad_batch_to_global_length(batch: Dict[str, Any]) -> Dict[str, Any]:
    """主训练进程把 local padded batch 右侧补齐到所有 rank 的 global max_T。"""

    local_max = int(batch.get("max_T_local", 0))
    global_max = _global_max_int(local_max)
    if global_max == local_max:
        batch["max_T_global"] = global_max
        return batch
    for frames in batch["frame_rows"]:
        # 只在主训练进程里补到 global max_T；padding frame 是 None，后续 loop 会跳过，
        # 不读图、不进 Qwen、不产生 loss。这样不同 rank 的 sequence 长度能对齐 collective，
        # 但不会把 padding 当成训练样本。
        while len(frames) < global_max:
            frames.append(None)
    valid = batch["valid_mask"]
    if valid.shape[1] < global_max:
        pad = torch.zeros((valid.shape[0], global_max - valid.shape[1]), dtype=torch.bool)
        batch["valid_mask"] = torch.cat([valid, pad], dim=1)
    batch["max_T_global"] = global_max
    return batch


def setup_distributed() -> Tuple[int, int, int]:
    """初始化 torch.distributed。"""

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        kwargs: Dict[str, Any] = {"backend": backend, "timeout": timedelta(hours=6)}
        if torch.cuda.is_available():
            # PyTorch 2.6+ 如果不显式传 device_id，会在 barrier/collective 前警告
            # "device used by this process is currently unknown"。老版本没有该参数，
            # 因此保留 TypeError fallback。
            kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        try:
            dist.init_process_group(**kwargs)
        except TypeError:
            kwargs.pop("device_id", None)
            dist.init_process_group(**kwargs)
    return rank, world_size, local_rank


def distributed_barrier() -> None:
    """DDP/torchrun 下带 device_id 的 barrier，避免 NCCL 设备映射警告。"""

    if not (dist.is_available() and dist.is_initialized()):
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def cleanup_distributed() -> None:
    """清理 DDP process group。"""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _load_images(paths: List[str]) -> List[Image.Image]:
    """读取 4 帧 RGB history；缺文件时抛错，由外层 frame skip 记录。"""

    return [Image.open(path).convert("RGB") for path in paths]


def _frame_images_exist(frame: FrameRow) -> bool:
    """只做轻量路径存在性检查，用于 loss 归一化和 batch 前跳过坏帧。"""

    return all(pathlib.Path(path).exists() for path in frame.history_rgb_paths)


def _parse_goal_xy(value: Any) -> Optional[Tuple[float, float]]:
    """把 dataset 里的 `ego_to_goal_xy` 容错解析成二元组。

    新数据由 build_dataset.py 从 meta `next_target_points[-1]` 生成；旧 index
    可能没有这个字段，解析失败时返回 None，并由 RouteSequenceDataset 跳过该
    frame，避免 prompt 中继续出现 UNKNOWN。
    """

    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except Exception:
        return None


def _messages(images: List[Image.Image], user_prompt: str) -> List[Dict[str, Any]]:
    """构造 Qwen structured chat messages。"""

    # Qwen3-VL processor 需要 structured message：4 张历史图先放，再放同一个 user prompt。
    # system prompt 固定为 v5 协议，确保 train/eval/probe 的图文输入完全一致。
    content: List[Dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": SYSTEM_PROMPT_V5},
        {"role": "user", "content": content},
    ]


def _q2_full_messages(images: List[Image.Image], q1_prompt: str, q1_text: str, q2_prompt: str) -> List[Dict[str, Any]]:
    """构造“图像 + Q1 user + Q1 assistant + Q2 user”的完整 Q2 对话。

    这个 helper 只用于 Q2 student rollout 采样：为了让多个 route 的 Q2 能合成 padded
    batch，先把 Q1 assistant 文本放回完整对话再 prefill/generate。正式 KL scoring
    不走这里，而是用精确的 q1_ids 追加到 Q1 KV 后再追加 Q2 user turn，避免
    `q1_ids -> text -> tokenizer` 往返造成 token 边界漂移。
    """

    content: List[Dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": q1_prompt})
    return [
        {"role": "system", "content": SYSTEM_PROMPT_V5},
        {"role": "user", "content": content},
        {"role": "assistant", "content": q1_text},
        {"role": "user", "content": q2_prompt},
    ]


def _select_batch_tensor(value: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """按 batch 维选择 tensor 行；rows 必须已经在 value 所在设备或可搬过去。"""

    # 所有 KVState 成员都约定 batch 维在第 0 维；index_select 比高级索引更稳定，
    # 也更容易保持 dtype/device 不被隐式改变。
    return value.index_select(0, rows.to(value.device))


def _slice_cache_batch(cache: Any, rows: torch.Tensor) -> Any:
    """从 batched past_key_values 中切出若干样本。

    Transformers 新版 Qwen3-VL 使用 DynamicCache，只有 `reorder_cache` 没有公开
    batch slice API；这里 clone 后用 reorder_cache(index) 等价完成子 batch 选择。
    legacy tuple cache 则按每层 K/V 的 batch 维 index_select。
    """

    if hasattr(cache, "reorder_cache"):
        # DynamicCache.reorder_cache 是 in-place 操作，所以先走 _clone_kv_state 的 cache
        # clone 逻辑，避免污染原 batched state。
        import copy

        cloned_state = copy.deepcopy(cache)
        # reorder_cache 原本用于 beam search，这里借它实现“保留指定 batch 行”。
        # 传入的 rows 可能只有一个元素，也可能是同长子 batch 的若干元素。
        cloned_state.reorder_cache(rows)
        return cloned_state
    if isinstance(cache, tuple):
        sliced_layers = []
        for layer in cache:
            if isinstance(layer, tuple):
                # legacy tuple cache 通常是每层 (key, value, ...)，只切 tensor，
                # 非 tensor 元数据保持原样。
                sliced_layers.append(tuple(_select_batch_tensor(x, rows) if isinstance(x, torch.Tensor) else x for x in layer))
            elif isinstance(layer, torch.Tensor):
                sliced_layers.append(_select_batch_tensor(layer, rows))
            else:
                sliced_layers.append(layer)
        return tuple(sliced_layers)
    return cache


def _normalize_rope_deltas_batch(rope: Any, batch_size: int) -> Any:
    """把 Qwen 返回的 `rope_deltas` 统一整理成 `(batch, 1)`。

    Qwen3-VL 的不同版本可能把 `rope_deltas` 存成 `(batch, 1)`，也可能存成
    `(1, batch)`。v5 的 batched Q1 会在生成中不断缩小 active batch，如果内部状态
    继续保存横向 `(1, batch)`，下一步 decode 很容易把 batch 维误当成 token 维，
    触发 `Target sizes: [1, -1]. Tensor sizes: [2, 1]` 这类 expand 报错。因此这里在
    KVState 边界统一成“每个样本一行”的形状。
    """

    if not hasattr(rope, "detach"):
        return rope
    rd = rope.detach().clone()
    if rd.ndim == 0:
        return rd
    # 一维向量通常就是每个样本一个 delta，补一列维度即可。
    if rd.ndim == 1 and rd.numel() == batch_size:
        return rd.reshape(batch_size, 1).contiguous()
    # 标准方向 `(batch, anything)`：如果第二维不止 1，也只保留每个样本的第一个 delta。
    # Qwen3-VL decode 只需要每条样本自己的 M-RoPE offset。
    if rd.shape[0] == batch_size:
        return rd.reshape(batch_size, -1)[:, :1].contiguous()
    # 某些 transformers 版本会返回 `(1, batch)`，转置成 `(batch, 1)`。
    if rd.ndim >= 2 and rd.shape[0] == 1 and rd.numel() == batch_size:
        return rd.reshape(1, batch_size).transpose(0, 1).contiguous()
    # 兜底：元素个数刚好等于 batch_size 时，也按每样本一个 delta 解释。
    if rd.numel() == batch_size:
        return rd.reshape(batch_size, 1).contiguous()
    # 无法判断 batch 维时保守 clone，不做错误 reshape；这种情况 smoke 会暴露。
    return rd


def _slice_rope_deltas_batch(rope: Any, rows: torch.Tensor, batch_size: int) -> Any:
    """按 batch 行切 `rope_deltas`，并保持 `(new_batch, 1)` 内部契约。"""

    rd = _normalize_rope_deltas_batch(rope, batch_size)
    if not hasattr(rd, "detach") or getattr(rd, "ndim", 0) == 0:
        return rd
    if rd.shape[0] == batch_size:
        return _select_batch_tensor(rd, rows)
    return rd


def _slice_kv_state_batch(state: KVState, rows: Sequence[int]) -> KVState:
    """从 batched KVState 中切出一个子 batch，保持 Cache 类型不退化。"""

    device = state.cache_input_ids.device
    row_tensor = torch.tensor(list(rows), device=device, dtype=torch.long)
    rope = _slice_rope_deltas_batch(state.rope_deltas, row_tensor, int(state.cache_input_ids.shape[0]))
    return KVState(
        decoded_input_ids=_select_batch_tensor(state.decoded_input_ids, row_tensor).detach().clone(),
        cache_input_ids=_select_batch_tensor(state.cache_input_ids, row_tensor).detach().clone(),
        attention_mask=_select_batch_tensor(state.attention_mask, row_tensor).detach().clone(),
        past_key_values=_slice_cache_batch(state.past_key_values, row_tensor),
        rope_deltas=rope,
        next_logits=_select_batch_tensor(state.next_logits, row_tensor).detach().clone(),
    )


def _last_valid_next_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """按 attention_mask 取每条样本最后一个真实 token 的 next-token logits。

    这个小函数故意不依赖 tokenizer padding_side：left/right padding 都只相信
    attention_mask，避免 batched prefill 在短样本上取到 pad 位置 logits。
    """

    valid = attention_mask.to(torch.bool)
    # 通过 mask 反推每条样本最后一个真实 token 的位置。右 padding 时它在中间，
    # 左 padding 时它通常是最后一列；这个写法同时覆盖两种 tokenizer padding_side。
    pos = torch.arange(attention_mask.shape[1], device=attention_mask.device).view(1, -1)
    last_valid_idx = pos.masked_fill(~valid, -1).max(dim=1).values.clamp(min=0)
    row_idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[row_idx, last_valid_idx.to(logits.device), :]


def _qwen_message_input_length(bundle: Any, messages: List[Dict[str, Any]]) -> int:
    """计算单条 Qwen message 经 processor 后的真实 input length。

    v5 的 batched KV 只有在多样本 processor length 完全一致时才安全复用；否则
    past_key_values 里会含 padding token，后续增量 decode 的 `prefix_len` / M-RoPE
    位置会偏离单样本路径。
    """

    text = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = _collect_images_from_messages(messages)
    inputs = bundle.processor(
        text=[text],
        images=images or None,
        return_tensors="pt",
        padding=False,
    )
    return int(inputs["input_ids"].shape[1])


def _kv_start_state_batch(bundle: Any, messages_list: List[List[Dict[str, Any]]]) -> KVState:
    """对多个 Qwen chat 一起做 prefill，得到 batched KVState。

    这是 v5 真正并行 Qwen 的基础入口：processor 一次接收多条 text 和所有图片，
    Qwen forward 的 batch 维即为 frame batch。若上层只给 1 条消息，行为等价于
    v3 的 `_kv_start_state`。
    """

    if len(messages_list) == 1:
        # 单条消息直接复用 v3 的成熟路径，避免在 size=1 时引入 batch-only 行为差异。
        return _kv_start_state(bundle, messages_list[0])
    texts = [
        bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]
    images: List[Image.Image] = []
    for messages in messages_list:
        imgs = _collect_images_from_messages(messages)
        if imgs:
            # Qwen processor 会按 message 顺序消费图片。这里把每条样本的 4 帧 history
            # 顺序拼到一个 images list 中，和 text batch 对齐。
            images.extend(imgs)
    inputs = bundle.processor(
        text=texts,
        images=images or None,
        return_tensors="pt",
        padding=True,
    )
    kwargs: Dict[str, Any] = {
        "input_ids": inputs["input_ids"].to(bundle.device),
        "attention_mask": inputs["attention_mask"].to(bundle.device),
    }
    for key, value in inputs.items():
        if key in ("input_ids", "attention_mask"):
            continue
        kwargs[key] = value.to(bundle.device) if isinstance(value, torch.Tensor) else value
    outputs = bundle.model(**kwargs, use_cache=True, return_dict=True)
    input_ids = kwargs["input_ids"]
    attention_mask = kwargs["attention_mask"]
    valid_lengths = attention_mask.to(torch.long).sum(dim=1)
    seq_len = int(input_ids.shape[1])
    if bool((valid_lengths != seq_len).any().item()):
        # 首 token logits 可以按 attention_mask 修正，但 past_key_values 仍会保留 pad
        # 位置；后续 _append_token_ids_with_logits 的 prefix_len 会按 padded length 走，
        # M-RoPE 位置不再等价于单样本路径。因此真正进入 batched KV 的样本必须零 padding。
        raise ValueError(f"batched Qwen KV requires equal input lengths, got valid={valid_lengths.detach().cpu().tolist()} padded_seq={seq_len}")
    # 这里仍然使用 last-valid logits，而不是 outputs.logits[:, -1, :]。
    # 当前函数会拒绝 padding，但保留这个写法能让 helper 自身更健壮，也和 smoke
    # 里对 padding 压力的检查逻辑一致。
    next_logits = _last_valid_next_logits(outputs.logits, attention_mask)
    return KVState(
        decoded_input_ids=input_ids,
        cache_input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=outputs.past_key_values,
        # 在 batched prefill 的出口就统一 rope_deltas 方向，后续所有 active-batch
        # 切片和增量 decode 都只处理 `(batch, 1)`，避免 Qwen 内部 `(1, batch)` 形状
        # 在生成循环里重新触发 fallback。
        rope_deltas=_normalize_rope_deltas_batch(getattr(outputs, "rope_deltas", None), int(input_ids.shape[0])),
        next_logits=next_logits,
    )


def _kv_start_state_batch_padded(bundle: Any, messages_list: List[List[Dict[str, Any]]]) -> KVState:
    """允许 padding 的 batched prefill，仅用于 Q1/Q2 student rollout 采样。

    这个 state 不会进入 KL 训练路径，只用于并行生成 student 文本/token。
    KL 和 Q2 续接仍由 `_run_frame` 重新构造单样本精确 state，避免 padded KV
    影响 teacher/student 分布比较。
    """

    if len(messages_list) == 1:
        return _kv_start_state(bundle, messages_list[0])
    texts = [
        bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]
    images: List[Image.Image] = []
    for messages in messages_list:
        imgs = _collect_images_from_messages(messages)
        if imgs:
            images.extend(imgs)
    inputs = bundle.processor(
        text=texts,
        images=images or None,
        return_tensors="pt",
        padding=True,
    )
    kwargs: Dict[str, Any] = {
        "input_ids": inputs["input_ids"].to(bundle.device),
        "attention_mask": inputs["attention_mask"].to(bundle.device),
    }
    for key, value in inputs.items():
        if key in ("input_ids", "attention_mask"):
            continue
        kwargs[key] = value.to(bundle.device) if isinstance(value, torch.Tensor) else value
    outputs = bundle.model(**kwargs, use_cache=True, return_dict=True)
    input_ids = kwargs["input_ids"]
    attention_mask = kwargs["attention_mask"]
    return KVState(
        decoded_input_ids=input_ids,
        cache_input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=outputs.past_key_values,
        rope_deltas=_normalize_rope_deltas_batch(getattr(outputs, "rope_deltas", None), int(input_ids.shape[0])),
        next_logits=_last_valid_next_logits(outputs.logits, attention_mask),
    )


def _decode_position_ids_varlen(
    rope_deltas: Any,
    valid_prefix_lengths: torch.Tensor,
    feed_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """为 padded batched decode 计算每样本自己的 M-RoPE position_ids。

    cache 本身按 padded seq_len 存储，但 pad key 被 attention_mask 屏蔽；位置编码应跟
    每条样本的真实有效长度走，而不是跟 padded 长度走。
    """

    base = torch.arange(feed_len, device=device, dtype=torch.long).view(1, -1).expand(batch_size, -1)
    if rope_deltas is None:
        rd = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
    else:
        rd = _normalize_rope_deltas_batch(rope_deltas, batch_size)
        rd = rd.to(device) if hasattr(rd, "to") else torch.as_tensor(rd, device=device)
        if rd.ndim == 0:
            rd = rd.view(1, 1)
        if rd.shape[0] != batch_size:
            rd = rd[:batch_size] if rd.shape[0] > batch_size else rd.expand(batch_size, -1)
        rd = rd.reshape(batch_size, -1)[:, :1]
    delta = valid_prefix_lengths.to(device=device, dtype=torch.long).view(batch_size, 1) + rd.to(torch.long)
    return (base + delta).unsqueeze(0).expand(3, -1, -1).contiguous()


def _append_token_ids_padded_rollout(bundle: Any, state: KVState, suffix_ids: torch.Tensor) -> KVState:
    """向允许 padding 的 batched rollout state 追加 token。

    只用于 no_grad Q1/Q2 rollout 采样。因为 cache 长度包含 padding，
    `cache_position` 仍使用 padded cache length；但 M-RoPE `position_ids`
    使用每样本真实有效长度。返回的 padded state 不进入 KL/Q2 训练路径。
    """

    if suffix_ids.ndim == 1:
        suffix_ids = suffix_ids.unsqueeze(0)
    suffix_ids = suffix_ids.to(state.cache_input_ids.device)
    old_attention = state.attention_mask.to(state.cache_input_ids.device)
    feed_len = int(suffix_ids.shape[1])
    batch_size = int(suffix_ids.shape[0])
    prefix_cache_len = int(state.cache_input_ids.shape[1])
    valid_prefix_lengths = old_attention.to(torch.long).sum(dim=1)
    attention_mask = torch.cat([old_attention, torch.ones_like(suffix_ids, device=old_attention.device)], dim=1)
    cache_position = torch.arange(prefix_cache_len, prefix_cache_len + feed_len, device=suffix_ids.device)
    position_ids = _decode_position_ids_varlen(
        state.rope_deltas,
        valid_prefix_lengths,
        feed_len,
        batch_size,
        suffix_ids.device,
    )
    outputs = bundle.model(
        input_ids=suffix_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=state.past_key_values,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )
    decoded_input_ids = torch.cat([state.cache_input_ids, suffix_ids], dim=1)
    return KVState(
        decoded_input_ids=decoded_input_ids,
        cache_input_ids=decoded_input_ids,
        attention_mask=attention_mask,
        past_key_values=outputs.past_key_values,
        rope_deltas=state.rope_deltas,
        next_logits=outputs.logits[:, -1, :],
    )


def _apply_repetition_penalty_batch(bundle: Any, logits: torch.Tensor, seen_ids: List[torch.Tensor], penalty: float) -> torch.Tensor:
    """逐样本施加 repetition penalty，避免把不同 frame 的历史 token 混在一起惩罚。"""

    if penalty == 1.0:
        return logits
    out = logits.clone()
    for row, ids in enumerate(seen_ids):
        if ids.numel() == 0:
            continue
        # 只惩罚该样本自己见过的 token。OPSD 训练关注 student rollout 的真实分布，
        # 如果把另一个样本或 padding 的 token 混进来，会造成 batch-vs-single 漂移。
        row_logits = out[row:row + 1]
        scores = row_logits.index_select(-1, ids.to(out.device))
        scores = torch.where(scores < 0, scores * penalty, scores / penalty)
        row_logits.index_copy_(-1, ids.to(out.device), scores)
        out[row:row + 1] = row_logits
    return out


def _student_generate_kv_batch(
    bundle: Any,
    state: KVState,
    max_new_tokens: int,
    *,
    allow_padded_cache: bool = False,
) -> Tuple[List[str], List[KVState], List[torch.Tensor]]:
    """batched greedy rollout，返回每个样本自己的文本、干净 KVState 和 token ids。

    关键点是 EOS 处理：某个样本预测到 EOS 时，保存它**追加 EOS 前**的 KVState，
    然后从 active batch 中移除；剩余样本继续并行生成。这样不会用重复 EOS/pad 污染
    已结束样本的 Q1 KV，后续 Q2 仍能接在干净的 Q1 assistant 输出后。
    """

    batch_size = int(state.next_logits.shape[0])
    if batch_size == 1:
        text, after, ids = _student_generate_kv(bundle, state, max_new_tokens)
        return [text], [after], [ids]

    eos_ids = set()
    eos = getattr(bundle.tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, (list, tuple, set)):
            eos_ids.update(int(x) for x in eos)
        else:
            eos_ids.add(int(eos))
    im_end = bundle.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        eos_ids.add(int(im_end))

    cur = state
    active = list(range(batch_size))
    generated: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
    final_states: List[Optional[KVState]] = [None for _ in range(batch_size)]
    seen_unique: List[torch.Tensor] = []
    for i in range(batch_size):
        # repetition penalty 只应该看到真实 prompt token。padding token 如果进入
        # seen 集合，会让 batch 路径和 single 路径产生轻微但系统性的差异。
        mask = state.attention_mask[i].to(torch.bool)
        real_ids = state.decoded_input_ids[i][mask]
        seen_unique.append(torch.unique(real_ids.reshape(-1).to(state.next_logits.device)))
    penalty = 1.05

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if not active:
                break
            # cur 只包含仍未 EOS 的 active 样本；seen_unique 仍按原始 batch 编号保存，
            # 因此这里要用 active 映射回原始样本。
            logits = _apply_repetition_penalty_batch(bundle, cur.next_logits, [seen_unique[i] for i in active], penalty)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_ids = [int(x) for x in next_token.reshape(-1).tolist()]
            keep_rows: List[int] = []
            keep_tokens: List[torch.Tensor] = []
            for row_idx, token_id in enumerate(token_ids):
                orig_idx = active[row_idx]
                if token_id in eos_ids:
                    # EOS 本身不追加到 KV。这样 final_states[orig_idx] 仍停在最后一个
                    # 非 EOS assistant token，后续 Q2 user turn 可以接在干净的 Q1 后面。
                    final_states[orig_idx] = _slice_kv_state_batch(cur, [row_idx])
                    continue
                tok = next_token[row_idx:row_idx + 1]
                generated[orig_idx].append(tok.detach().clone())
                seen_unique[orig_idx] = torch.unique(torch.cat([seen_unique[orig_idx], tok.reshape(-1).to(seen_unique[orig_idx].device)], dim=0))
                keep_rows.append(row_idx)
                keep_tokens.append(tok)
            if not keep_rows:
                active = []
                break
            # 把已经 EOS 的行从 cur 中移除，只对仍活跃样本追加本轮 token。
            # 这一步是 batched generate 的关键：不让 finished 样本继续吃 pad/EOS。
            cur = _slice_kv_state_batch(cur, keep_rows)
            suffix = torch.cat(keep_tokens, dim=0).to(cur.cache_input_ids.device)
            if allow_padded_cache:
                cur = _append_token_ids_padded_rollout(bundle, cur, suffix)
            else:
                cur, _ = _append_token_ids(bundle, cur, suffix)
            # `_append_token_ids` 会根据当前 active batch 继续 decode；这里再次规范
            # rope_deltas，防止底层 forward 返回的新 Cache/输出把形状恢复成 `(1, batch)`。
            cur.rope_deltas = _normalize_rope_deltas_batch(cur.rope_deltas, int(cur.cache_input_ids.shape[0]))
            active = [active[i] for i in keep_rows]
        for row_idx, orig_idx in enumerate(active):
            # 达到 max_new_tokens 仍没 EOS 的样本，最终 KV 就是当前 active state。
            final_states[orig_idx] = _slice_kv_state_batch(cur, [row_idx])

    texts: List[str] = []
    ids_out: List[torch.Tensor] = []
    states_out: List[KVState] = []
    for idx in range(batch_size):
        if generated[idx]:
            ids = torch.cat(generated[idx], dim=1).to(state.cache_input_ids.device)
            text = bundle.processor.batch_decode(ids, skip_special_tokens=True)[0]
        else:
            # 空生成也保留 shape=(1,0)，方便后续 _append_token_ids_with_logits
            # 直接用 numel 判断，无需额外处理 None。
            ids = state.cache_input_ids.new_zeros((1, 0))
            text = ""
        texts.append(text)
        ids_out.append(ids)
        assert final_states[idx] is not None
        states_out.append(final_states[idx])  # type: ignore[arg-type]
    return texts, states_out, ids_out


def _loss_positions(bundle: Any, text: str, span_fn: Any, weights: Mapping[str, float]) -> Dict[str, List[int]]:
    """把字符 span 映射到 token 下标。"""

    enc = bundle.tokenizer(text or "", return_offsets_mapping=True, add_special_tokens=False)
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    spans = span_fn(text or "")
    out: Dict[str, List[int]] = {key: [] for key in weights}
    for key, (lo, hi) in spans.items():
        if key not in out:
            continue
        # 字符 span 与 token offset 相交即可纳入 loss。这样即使 tokenizer 把 "A - ..."
        # 切成多个 token，也能完整监督离散值所在整段。
        out[key] = [i for i, (a, b) in enumerate(offsets) if a < hi and b > lo]
    return out


def _select_logits_at_positions(logits: torch.Tensor, positions: List[int]) -> Optional[torch.Tensor]:
    """只保留需要监督的 token 位置，尽早释放完整 vocab logits。"""

    if not positions:
        return None
    idx = torch.tensor(positions, device=logits.device, dtype=torch.long)
    return logits[:, idx, :].reshape(-1, logits.shape[-1])


def _kl_selected_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
    """对已经裁剪到监督位置的 logits 计算 forward-KL。"""

    temp = max(float(temperature), 1e-6)
    return F.kl_div(
        F.log_softmax(student_logits / temp, dim=-1),
        F.softmax(teacher_logits.detach() / temp, dim=-1),
        reduction="batchmean",
    ) * (temp * temp)


def _trainable_graph_zero(bundle: Any, fallback: torch.Tensor) -> torch.Tensor:
    """返回一个带 autograd graph 的 0，供“无监督 span”帧安全 backward。

    某些 student rollout 可能没有输出 `RS:` / `ABNORMAL:` / `EVENT:` 等可监督字段。
    这种帧的 OPSD loss 应该是 0，但仍要允许外层执行 backward，并保持所有 rank 的
    optimizer/all-reduce 节奏一致。直接用 no-grad KV logits 构造的 0 没有 grad_fn，
    会触发 `element 0 of tensors does not require grad`；这里改为从第一个可训练 LoRA
    参数构造 `param.sum() * 0.0`。
    """

    model = bundle.model if hasattr(bundle, "model") else bundle
    for param in model.parameters():
        if param.requires_grad:
            return param.reshape(-1)[0] * 0.0
    return fallback.sum() * 0.0


def _pad_rollout_id_list(bundle: Any, ids_list: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """把多条 rollout ids padding 成 `(B, L)`，并返回真实 token mask。

    这个 padding 只用于 batched teacher/student scoring。loss 位置来自每个样本自己的
    span positions，padding token 永远不会进入 KL。
    """

    if not ids_list:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        return torch.zeros((0, 0), device=device, dtype=torch.long), torch.zeros((0, 0), device=device, dtype=torch.bool)
    pad_id = getattr(bundle.tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(bundle.tokenizer, "eos_token_id", 0)
        if isinstance(pad_id, (list, tuple, set)):
            pad_id = next(iter(pad_id), 0)
    device = ids_list[0].device
    max_len = max(int(ids.reshape(1, -1).shape[1]) for ids in ids_list)
    padded = torch.full((len(ids_list), max_len), int(pad_id or 0), device=device, dtype=torch.long)
    mask = torch.zeros((len(ids_list), max_len), device=device, dtype=torch.bool)
    for row, ids in enumerate(ids_list):
        flat = ids.reshape(1, -1).to(device=device, dtype=torch.long)
        length = int(flat.shape[1])
        if length <= 0:
            continue
        padded[row, :length] = flat[0]
        mask[row, :length] = True
    return padded, mask


def _append_token_ids_with_logits_padded_scoring(
    bundle: Any,
    state: KVState,
    suffix_ids: torch.Tensor,
    suffix_mask: torch.Tensor,
) -> Tuple[KVState, torch.Tensor]:
    """在 padded batched state 后追加一批 rollout ids，并返回逐 token 预测 logits。

    与 `_append_token_ids_with_logits` 的语义相同：`pred_logits[:, j, :]` 对齐
    `suffix_ids[:, j]` 这个 token 被预测时的分布。不同点是这里允许 prefix 和 suffix
    都带 padding；attention mask 会屏蔽 padding，M-RoPE position_ids 按每个样本自己的
    真实 prefix length 计算。该函数用于并行 teacher/student KL scoring。
    """

    if suffix_ids.ndim == 1:
        suffix_ids = suffix_ids.unsqueeze(0)
    if suffix_mask.ndim == 1:
        suffix_mask = suffix_mask.unsqueeze(0)
    suffix_ids = suffix_ids.to(state.cache_input_ids.device)
    suffix_mask = suffix_mask.to(state.cache_input_ids.device, dtype=torch.bool)
    if suffix_ids.shape[1] == 0:
        empty = state.next_logits.new_zeros((suffix_ids.shape[0], 0, state.next_logits.shape[-1]))
        return state, empty
    old_attention = state.attention_mask.to(state.cache_input_ids.device)
    attention_mask = torch.cat([old_attention, suffix_mask.to(old_attention.dtype)], dim=1)
    batch_size = int(suffix_ids.shape[0])
    feed_len = int(suffix_ids.shape[1])
    prefix_cache_len = int(state.cache_input_ids.shape[1])
    valid_prefix_lengths = old_attention.to(torch.long).sum(dim=1)
    cache_position = torch.arange(prefix_cache_len, prefix_cache_len + feed_len, device=suffix_ids.device)
    position_ids = _decode_position_ids_varlen(
        state.rope_deltas,
        valid_prefix_lengths,
        feed_len,
        batch_size,
        suffix_ids.device,
    )
    outputs = bundle.model(
        input_ids=suffix_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=state.past_key_values,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )
    pred_logits = torch.cat([state.next_logits.unsqueeze(1), outputs.logits[:, :-1, :]], dim=1)
    decoded_input_ids = torch.cat([state.cache_input_ids, suffix_ids], dim=1)
    new_state = KVState(
        decoded_input_ids=decoded_input_ids,
        cache_input_ids=decoded_input_ids,
        attention_mask=attention_mask,
        past_key_values=outputs.past_key_values,
        rope_deltas=state.rope_deltas,
        next_logits=outputs.logits[:, -1, :],
    )
    return new_state, pred_logits


def _append_token_ids_padded_no_logits(
    bundle: Any,
    state: KVState,
    ids_list: Sequence[torch.Tensor],
) -> KVState:
    """把多条变长 token ids 追加到 padded KV，但不保留逐 token vocab logits。

    该函数用于构造 Q2 parallel KL 的上下文：先把每个样本自己的精确 `q1_ids`
    追加到 Q1 prompt KV，再追加 Q2 user turn。这里不会产生训练 loss，所以外层会在
    `torch.no_grad()` / teacher eval context 下调用，避免 prompt/context 构造占用
    autograd 显存。真正需要梯度的 logits 只在后续 q2 rollout token scoring 时保留。
    """

    if len(ids_list) == 1:
        new_state, _ = _append_token_ids(bundle, state, ids_list[0])
        return new_state
    suffix_ids, suffix_mask = _pad_rollout_id_list(bundle, ids_list)
    if suffix_ids.numel() == 0 or int(suffix_mask.sum().item()) == 0:
        return state
    suffix_ids = suffix_ids.to(state.cache_input_ids.device)
    suffix_mask = suffix_mask.to(state.cache_input_ids.device, dtype=torch.bool)
    if int(suffix_ids.shape[0]) != int(state.cache_input_ids.shape[0]):
        raise ValueError(
            f"padded append batch mismatch: ids={int(suffix_ids.shape[0])} "
            f"state={int(state.cache_input_ids.shape[0])}"
        )
    old_attention = state.attention_mask.to(state.cache_input_ids.device)
    attention_mask = torch.cat([old_attention, suffix_mask.to(old_attention.dtype)], dim=1)
    batch_size = int(suffix_ids.shape[0])
    feed_len = int(suffix_ids.shape[1])
    prefix_cache_len = int(state.cache_input_ids.shape[1])
    valid_prefix_lengths = old_attention.to(torch.long).sum(dim=1)
    cache_position = torch.arange(prefix_cache_len, prefix_cache_len + feed_len, device=suffix_ids.device)
    position_ids = _decode_position_ids_varlen(
        state.rope_deltas,
        valid_prefix_lengths,
        feed_len,
        batch_size,
        suffix_ids.device,
    )
    outputs = bundle.model(
        input_ids=suffix_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=state.past_key_values,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )
    decoded_input_ids = torch.cat([state.cache_input_ids, suffix_ids], dim=1)
    return KVState(
        decoded_input_ids=decoded_input_ids,
        cache_input_ids=decoded_input_ids,
        attention_mask=attention_mask,
        past_key_values=outputs.past_key_values,
        rope_deltas=state.rope_deltas,
        next_logits=_last_valid_next_logits(outputs.logits, suffix_mask.to(outputs.logits.device)),
    )


def _render_user_turn_ids(bundle: Any, user_text: str) -> torch.Tensor:
    """把“关闭上一轮 assistant + 新 user turn + generation prompt”渲染成 token ids。"""

    suffix = bundle.processor.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not suffix.startswith("\n"):
        suffix = "\n" + suffix
    enc = bundle.tokenizer("<|im_end|>" + suffix, add_special_tokens=False, return_tensors="pt")
    return enc["input_ids"]


def _append_user_turn_batch_padded(bundle: Any, state: KVState, user_texts: Sequence[str]) -> KVState:
    """对 batched Q1-after KV 追加每个样本自己的 Q2 user turn。

    单样本时直接复用旧 `_append_user_turn`，方便 smoke 和旧逐帧路径做 bit-level 对照；
    多样本时走 padded token ids，attention mask 会屏蔽不同长度 user turn 的 padding。
    """

    if len(user_texts) == 1:
        return _append_user_turn(bundle, state, user_texts[0])
    ids_list = [_render_user_turn_ids(bundle, text) for text in user_texts]
    return _append_token_ids_padded_no_logits(bundle, state, ids_list)


def _opsd_loss_batch_states(
    bundle: Any,
    *,
    student_state: KVState,
    teacher_state: KVState,
    rollout_texts: Sequence[str],
    rollout_ids_list: Sequence[torch.Tensor],
    span_fn: Any,
    weights: Mapping[str, float],
    temperature: float,
) -> Tuple[torch.Tensor, List[Dict[str, float]], List[torch.Tensor]]:
    """在已经构造好的 batched student/teacher KV state 上计算 OPSD KL。

    与 `_opsd_loss_batch_messages` 的区别是：本函数不重新 prefill prompt，而是直接接
    收外层按精确 token ids 构造好的 state。Q2 parallel KL 用它来避免把 Q1 rollout
    先 decode 成文本再重新套 chat template。
    """

    if not rollout_ids_list:
        raise ValueError("_opsd_loss_batch_states requires at least one rollout.")
    suffix_ids, suffix_mask = _pad_rollout_id_list(bundle, rollout_ids_list)
    positions_by_sample: List[Dict[str, List[int]]] = []
    for row, text in enumerate(rollout_texts):
        raw_positions = _loss_positions(bundle, text, span_fn, weights)
        valid_len = int(suffix_mask[row].sum().item())
        positions_by_sample.append(
            {key: [p for p in raw_positions.get(key, []) if p < valid_len] for key in weights}
        )
    teacher_selected: Dict[Tuple[int, str], torch.Tensor] = {}
    with _teacher_eval_context(bundle):
        _, teacher_logits = _append_token_ids_with_logits_padded_scoring(bundle, teacher_state, suffix_ids, suffix_mask)
        # 立即把 teacher full-vocab logits 裁剪到监督 span，再释放完整 B x L x vocab
        # 张量。parallel KL 的显存峰值主要就在 logits；不要让 teacher/student 两份大
        # logits 同时常驻超过必要时间。
        for row, positions in enumerate(positions_by_sample):
            for key, pos in positions.items():
                if not pos:
                    continue
                idx = torch.tensor(pos, device=teacher_logits.device, dtype=torch.long)
                teacher_selected[(row, key)] = teacher_logits[row:row + 1, idx, :].reshape(-1, teacher_logits.shape[-1]).detach()
        del teacher_logits
    _, student_logits = _append_token_ids_with_logits_padded_scoring(bundle, student_state, suffix_ids, suffix_mask)
    zero = student_logits.sum() * 0.0
    per_sample_losses: List[torch.Tensor] = []
    per_sample_parts: List[Dict[str, float]] = []
    for row, positions in enumerate(positions_by_sample):
        sample_total = zero
        sample_parts: Dict[str, float] = {}
        for key, weight in weights.items():
            pos = positions.get(key, [])
            t = teacher_selected.get((row, key))
            if pos and t is not None:
                idx = torch.tensor(pos, device=student_logits.device, dtype=torch.long)
                s = student_logits[row:row + 1, idx, :].reshape(-1, student_logits.shape[-1])
                part_loss = _kl_selected_logits(s, t, temperature=temperature)
            else:
                part_loss = zero
            sample_total = sample_total + float(weight) * part_loss
            sample_parts[key] = float(part_loss.detach().item()) if part_loss.numel() else 0.0
        per_sample_losses.append(sample_total)
        per_sample_parts.append(sample_parts)
    return torch.stack(per_sample_losses).sum(), per_sample_parts, per_sample_losses


def _opsd_loss_batch_messages(
    bundle: Any,
    *,
    student_messages: Sequence[List[Dict[str, Any]]],
    teacher_messages: Sequence[List[Dict[str, Any]]],
    rollout_texts: Sequence[str],
    rollout_ids_list: Sequence[torch.Tensor],
    span_fn: Any,
    weights: Mapping[str, float],
    temperature: float,
) -> Tuple[torch.Tensor, List[Dict[str, float]], List[torch.Tensor]]:
    """对一组样本并行计算 OPSD teacher/student KL。

    这是 v5 全流程并行化的核心 scoring helper：同一个 chunk 内的 student prompt
    prefill、teacher privileged prompt prefill、同一批 student rollout token scoring
    都走 batched Qwen。函数仍按样本分别计算 span KL，最后返回每个样本自己的 loss，
    因此外层 loss 归一化语义与旧逐帧 `_opsd_loss` 保持一致。
    """

    if not rollout_ids_list:
        raise ValueError("_opsd_loss_batch_messages requires at least one rollout.")
    with torch.no_grad():
        student_state = _kv_start_state_batch_padded(bundle, list(student_messages))
    with _teacher_eval_context(bundle):
        teacher_state = _kv_start_state_batch_padded(bundle, list(teacher_messages))
    return _opsd_loss_batch_states(
        bundle,
        student_state=student_state,
        teacher_state=teacher_state,
        rollout_texts=rollout_texts,
        rollout_ids_list=rollout_ids_list,
        span_fn=span_fn,
        weights=weights,
        temperature=temperature,
    )


def _opsd_loss(
    bundle: Any,
    *,
    student_state: Any,
    teacher_state: Any,
    rollout_text: str,
    rollout_ids: torch.Tensor,
    span_fn: Any,
    weights: Mapping[str, float],
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """在同一批 student rollout token 上比较 teacher/student logits。"""

    zero = _trainable_graph_zero(bundle, student_state.next_logits)
    if rollout_ids.numel() == 0:
        # student 没生成任何可监督 token 时，返回带 graph/device 的 0，保证外层
        # backward 和 loss 归一化逻辑不用特判。
        return zero, {key: 0.0 for key in weights}
    positions = _loss_positions(bundle, rollout_text, span_fn, weights)
    active_positions = {key: pos for key, pos in positions.items() if pos}
    if not active_positions:
        # 生成文本缺少目标字段时不强行对全段做 KL，避免把无关分析 token 当成离散标签。
        return zero, {key: 0.0 for key in weights}

    teacher_selected: Dict[str, torch.Tensor] = {}
    with _teacher_eval_context(bundle):
        # teacher 使用同一个 base Qwen，但 adapter 被临时禁用，并吃 privileged prompt。
        # logits 只 detach 作目标分布；反向梯度只流向当前启用 LoRA 的 student。
        _, teacher_logits, _ = _append_token_ids_with_logits(bundle, _clone_kv_state(teacher_state), rollout_ids)
        for key, pos in active_positions.items():
            selected = _select_logits_at_positions(teacher_logits, pos)
            if selected is not None:
                teacher_selected[key] = selected.detach()
        del teacher_logits

    _, student_logits, _ = _append_token_ids_with_logits(bundle, _clone_kv_state(student_state), rollout_ids)
    student_selected: Dict[str, torch.Tensor] = {}
    for key, pos in active_positions.items():
        # student logits 保留梯度，teacher logits 已 detach；两边只在同一 token 位置比较。
        selected = _select_logits_at_positions(student_logits, pos)
        if selected is not None:
            student_selected[key] = selected
    del student_logits

    total = zero
    parts: Dict[str, float] = {}
    for key, weight in weights.items():
        if key not in student_selected or key not in teacher_selected:
            loss = zero
        else:
            loss = _kl_selected_logits(student_selected[key], teacher_selected[key], temperature=temperature)
        total = total + float(weight) * loss
        parts[key] = float(loss.detach().item()) if loss.numel() else 0.0
    return total, parts


def _rs_target_from_frame(frame: FrameRow) -> RSTarget:
    """把 FrameRow 还原成 RSTarget。"""

    return RSTarget(
        label=frame.rs_label,
        option=RS_LABEL_TO_OPTION.get(frame.rs_label, frame.rs_option),
        description=RS_OPTION_DESCRIPTIONS.get(RS_LABEL_TO_OPTION.get(frame.rs_label, frame.rs_option), ""),
        confidence=float(frame.raw.get("rs_confidence", 0.0) or 0.0),
        secondary=tuple(frame.raw.get("rs_secondary") or []),
        candidates={str(k): float(v) for k, v in (frame.raw.get("rs_candidates") or {}).items()},
    )


def _event_target_from_frame(frame: FrameRow, student_event: Optional[str] = None) -> EventTarget:
    """从压缩 frame row 解析 EventTarget。

    frame.raw 已保留 event_labels_raw，resolve_event_target 会按 v5 的多标签规则处理。
    """

    raw = dict(frame.raw)
    if "events" not in raw:
        raw["events"] = raw.get("event_labels_raw") or [frame.event_code]
    return resolve_event_target(raw, student_event=student_event)


def _reset_memory_for_frame_row(frame: FrameRow) -> Memory:
    """按当前帧 GT RS + 当前帧目的地坐标重置 v5 memory。"""

    return reset_memory_for_frame(_rs_target_from_frame(frame), ego_to_goal_xy=frame.ego_to_goal_xy)


def _run_frame(
    bundle: Any,
    memory: Memory,
    frame: FrameRow,
    *,
    max_new_tokens_q1: int,
    max_new_tokens_q2: int,
    temperature: float,
    q1_student_state: Optional[KVState] = None,
    q1_text: Optional[str] = None,
    q1_after: Optional[KVState] = None,
    q1_ids: Optional[torch.Tensor] = None,
    q2_student_state: Optional[KVState] = None,
    q2_text: Optional[str] = None,
    q2_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any], Memory, bool]:
    """运行单帧 Q1/Q2，返回 loss、统计、更新后的 memory、是否需要下一帧 reset。"""

    timings: Dict[str, float] = {
        "q1_student_seconds": 0.0,
        "q1_teacher_seconds": 0.0,
        "q1_loss_seconds": 0.0,
        "q2_rollout_seconds": 0.0,
        "q2_teacher_seconds": 0.0,
        "q2_loss_seconds": 0.0,
    }
    rs_target = _rs_target_from_frame(frame)
    event_target_static = _event_target_from_frame(frame)

    # ---- Q1: student rollout ----
    images: Optional[List[Image.Image]] = None
    if q1_text is not None and q1_ids is not None and (q1_student_state is None or q1_after is None):
        q1_student_start = time.time()
        images = _load_images(frame.history_rgb_paths)
        q1_prompt = build_q1_student_prompt(memory)
        with torch.no_grad():
            # Q1 文本/token 已由 padded batched rollout 得到；这里只重建单样本精确
            # Q1 prompt KV 和 Q1-after KV 给 KL/Q2 使用。padded KV 不进入训练图，
            # 因而不会把 padding prefix length 或 M-RoPE 位置带进 OPSD loss。
            q1_student_state = _kv_start_state(bundle, _messages(images, q1_prompt))
            q1_after, _ = _append_token_ids(bundle, q1_student_state, q1_ids)
        timings["q1_student_seconds"] = time.time() - q1_student_start
    elif q1_student_state is None or q1_text is None or q1_after is None or q1_ids is None:
        q1_student_start = time.time()
        images = _load_images(frame.history_rgb_paths)
        q1_prompt = build_q1_student_prompt(memory)
        with torch.no_grad():
            # student 先自由生成 Q1；OPSD 的监督不是 teacher-forced token，而是随后在
            # 这批 student 自己采样出的 token 上比较 teacher/student 分布。
            q1_student_state = _kv_start_state(bundle, _messages(images, q1_prompt))
            q1_text, q1_after, q1_ids = _student_generate_kv(bundle, q1_student_state, max_new_tokens_q1)
        timings["q1_student_seconds"] = time.time() - q1_student_start
    q1_parsed = parse_q1_output(q1_text)
    q1_teacher_prompt = build_q1_teacher_prompt(
        memory,
        rs_target=rs_target,
        event_target=event_target_static,
        weather_text=frame.weather_text,
    )
    if images is None:
        images = _load_images(frame.history_rgb_paths)
    q1_teacher_start = time.time()
    q1_teacher_state = _teacher_start_state(bundle, _messages(images, q1_teacher_prompt))
    timings["q1_teacher_seconds"] = time.time() - q1_teacher_start
    q1_loss_start = time.time()
    q1_loss, q1_parts = _opsd_loss(
        bundle,
        student_state=q1_student_state,
        teacher_state=q1_teacher_state,
        rollout_text=q1_text,
        rollout_ids=q1_ids,
        span_fn=target_spans_q1,
        weights=loss_weights_q1(),
        temperature=temperature,
    )
    timings["q1_loss_seconds"] = time.time() - q1_loss_start

    student_rs = q1_parsed.get("rs_label")
    q1_rs_correct = student_rs == frame.rs_label
    q1_abnormal = q1_parsed.get("abnormal") == "YES" if q1_parsed.get("abnormal") else None
    memory_after_q1 = update_memory_after_q1(memory, student_rs_label=student_rs, student_abnormal=q1_abnormal)
    stats: Dict[str, Any] = {
        "q1_rs_correct": q1_rs_correct,
        "q1_abnormal_correct": q1_abnormal == frame.abnormal if q1_abnormal is not None else False,
        "q2_triggered": False,
        "candidate_mismatch": False,
        "q1_rollout_tokens": int(q1_ids.numel()),
        "q2_rollout_tokens": 0,
        "q1_parts": q1_parts,
        "timings": timings,
    }
    if not q1_rs_correct:
        # Q1 的 RS 是 Q2 候选池的上层条件。RS 错时继续问 Q2 会把错误道路结构传下去，
        # 所以本帧立即截断；下一帧由外层恢复 GT RS + RE。
        return q1_loss, stats, memory_after_q1, True

    # ---- Q2: 只有 RS 正确才进入 ----
    q2_prompt = build_q2_student_prompt(
        memory_after_q1,
        option_map=frame.event_option_map,
        q1_abnormal=bool(q1_abnormal),
        regular_event_codes=frame.regular_event_codes,
    )
    if q2_text is not None and q2_ids is not None and q2_student_state is None:
        q2_rollout_start = time.time()
        with torch.no_grad():
            # Q2 文本/token 已由 padded batched rollout 得到；这里只重建单样本精确
            # Q2 prompt state 给 KL 使用，不再重复生成。
            q2_student_state = _append_user_turn(bundle, q1_after, q2_prompt)
        timings["q2_rollout_seconds"] = time.time() - q2_rollout_start
    elif q2_student_state is None or q2_text is None or q2_ids is None:
        q2_rollout_start = time.time()
        with torch.no_grad():
            q2_student_state = _append_user_turn(bundle, q1_after, q2_prompt)
            q2_text, _q2_after, q2_ids = _student_generate_kv(bundle, q2_student_state, max_new_tokens_q2)
        timings["q2_rollout_seconds"] = time.time() - q2_rollout_start
    else:
        # Q2 student rollout 已在当前 timestep 的 grouped/batched 路径中完成；这里继续
        # 使用同一段 q2_text/q2_ids 做 teacher/student KL，不再重复生成。
        timings["q2_rollout_seconds"] = 0.0
    q2_parsed = parse_q2_output(q2_text, frame.event_option_map)
    event_target = _event_target_from_frame(frame, student_event=q2_parsed.get("event_label"))
    # EVENT 支持“单标签训练、双标签容错”：如果 raw label 有多个 UE/RE，而 student
    # 选中了其中之一，teacher target 会接受这个 student_event 作为解释目标。
    target_option = option_for_event(event_target.label, frame.event_option_map)
    stats["candidate_mismatch"] = target_option is None
    q2_teacher_prompt = build_q2_teacher_prompt(
        memory_after_q1,
        option_map=frame.event_option_map,
        q1_abnormal=bool(q1_abnormal),
        event_target=event_target,
        regular_event_codes=frame.regular_event_codes,
    )
    # teacher 的 Q2 也保持串行对话：先用 privileged Q1 prompt 吃掉同一段 student
    # Q1 rollout token，再追加 Q2 teacher user turn。这样 Q2 KL 目标分布和 student
    # 一样是“基于 Q1 KV cache 继续问”，不是 fresh dialog。
    q2_teacher_start = time.time()
    with _teacher_eval_context(bundle):
        q1_teacher_after, _, _ = _append_token_ids_with_logits(bundle, _clone_kv_state(q1_teacher_state), q1_ids)
        q2_teacher_state = _append_user_turn(bundle, q1_teacher_after, q2_teacher_prompt)
    timings["q2_teacher_seconds"] = time.time() - q2_teacher_start
    q2_loss_start = time.time()
    q2_loss, q2_parts = _opsd_loss(
        bundle,
        student_state=q2_student_state,
        teacher_state=q2_teacher_state,
        rollout_text=q2_text,
        rollout_ids=q2_ids,
        span_fn=target_spans_q2,
        weights=loss_weights_q2(),
        temperature=temperature,
    )
    timings["q2_loss_seconds"] = time.time() - q2_loss_start
    student_event = q2_parsed.get("event_label")
    memory_after_q2 = update_memory_after_q2(memory_after_q1, student_event_label=student_event)
    q2_invalid = student_event is None
    stats.update(
        {
            "q2_triggered": True,
            "q2_event_correct": student_event == event_target.label,
            "q2_invalid_output": q2_invalid,
            "q2_rollout_tokens": int(q2_ids.numel()),
            "q2_parts": q2_parts,
        }
    )
    return q1_loss + q2_loss, stats, memory_after_q2, q2_invalid


def _run_chunk_parallel_kl(
    bundle: Any,
    chunk: Sequence[Tuple[int, SequenceRow, FrameRow, Memory]],
    *,
    q1_rollouts: Sequence[Optional[Tuple[Optional[KVState], str, Optional[KVState], torch.Tensor]]],
    q2_rollouts: Sequence[Optional[Tuple[Optional[KVState], str, torch.Tensor]]],
    temperature: float,
) -> Tuple[torch.Tensor, List[Tuple[int, Dict[str, Any], Memory, bool, float]]]:
    """对同一 timestep 的 chunk 并行计算 Q1/Q2 teacher/student KL。

    返回 `(total_loss, per_frame_results)`，其中 `per_frame_results` 的元素是
    `(route_batch_index, stats, next_memory, need_reset, detached_loss_value)`。

    约束：
    - Q1 rollout 必须已由 grouped/batched 路径生成；否则无法避免重复采样。
    - 对 Q1 RS 正确的样本，Q2 rollout 也必须已准备好；否则回退旧逐帧路径。
    - 使用 batched full/padded prompt state 只做 KL scoring，不把 padded KV 写回 memory。
    """

    if len(chunk) == 0:
        raise ValueError("empty chunk")
    if len(q1_rollouts) != len(chunk) or len(q2_rollouts) != len(chunk):
        raise ValueError("chunk rollout length mismatch")
    if any(item is None for item in q1_rollouts):
        raise ValueError("parallel KL requires all Q1 rollouts to be ready")

    images_per_frame = [_load_images(item[2].history_rgb_paths) for item in chunk]
    q1_texts: List[str] = []
    q1_ids_list: List[torch.Tensor] = []
    q1_student_messages: List[List[Dict[str, Any]]] = []
    q1_teacher_messages: List[List[Dict[str, Any]]] = []
    q1_parsed_list: List[Dict[str, str]] = []
    q1_abnormal_flags: List[Optional[bool]] = []
    for (images, (_b, _route, frame, memory), rollout) in zip(images_per_frame, chunk, q1_rollouts):
        assert rollout is not None
        _q1_state, q1_text, _q1_after, q1_ids = rollout
        q1_texts.append(q1_text)
        q1_ids_list.append(q1_ids)
        q1_parsed = parse_q1_output(q1_text)
        q1_parsed_list.append(q1_parsed)
        q1_abnormal = q1_parsed.get("abnormal") == "YES" if q1_parsed.get("abnormal") else None
        q1_abnormal_flags.append(q1_abnormal)
        rs_target = _rs_target_from_frame(frame)
        event_target_static = _event_target_from_frame(frame)
        q1_student_messages.append(_messages(images, build_q1_student_prompt(memory)))
        q1_teacher_messages.append(
            _messages(
                images,
                build_q1_teacher_prompt(
                    memory,
                    rs_target=rs_target,
                    event_target=event_target_static,
                    weather_text=frame.weather_text,
                ),
            )
        )

    q1_total_loss, q1_parts_list, q1_loss_list = _opsd_loss_batch_messages(
        bundle,
        student_messages=q1_student_messages,
        teacher_messages=q1_teacher_messages,
        rollout_texts=q1_texts,
        rollout_ids_list=q1_ids_list,
        span_fn=target_spans_q1,
        weights=loss_weights_q1(),
        temperature=temperature,
    )
    total_loss = q1_total_loss
    q2_parts_by_local: Dict[int, Dict[str, float]] = {}
    q2_loss_by_local: Dict[int, torch.Tensor] = {}

    q2_local_indices: List[int] = []
    q2_q1_ids_list: List[torch.Tensor] = []
    q2_student_prompts: List[str] = []
    q2_teacher_prompts: List[str] = []
    q2_texts: List[str] = []
    q2_ids_list: List[torch.Tensor] = []
    q2_event_targets: Dict[int, EventTarget] = {}
    for local_idx, ((images, (_b, _route, frame, memory)), q1_text, q1_parsed, q1_abnormal) in enumerate(
        zip(zip(images_per_frame, chunk), q1_texts, q1_parsed_list, q1_abnormal_flags)
    ):
        if q1_parsed.get("rs_label") != frame.rs_label:
            continue
        q2_rollout = q2_rollouts[local_idx]
        if q2_rollout is None:
            raise ValueError("parallel KL requires Q2 rollout for every Q1-correct frame")
        _q2_state, q2_text, q2_ids = q2_rollout
        q2_parsed = parse_q2_output(q2_text, frame.event_option_map)
        event_target = _event_target_from_frame(frame, student_event=q2_parsed.get("event_label"))
        q2_event_targets[local_idx] = event_target
        memory_after_q1 = update_memory_after_q1(
            memory,
            student_rs_label=q1_parsed.get("rs_label"),
            student_abnormal=q1_abnormal,
        )
        q1_student_prompt = build_q1_student_prompt(memory)
        q1_teacher_prompt = build_q1_teacher_prompt(
            memory,
            rs_target=_rs_target_from_frame(frame),
            event_target=_event_target_from_frame(frame),
            weather_text=frame.weather_text,
        )
        q2_student_prompt = build_q2_student_prompt(
            memory_after_q1,
            option_map=frame.event_option_map,
            q1_abnormal=bool(q1_abnormal),
            regular_event_codes=frame.regular_event_codes,
        )
        q2_teacher_prompt = build_q2_teacher_prompt(
            memory_after_q1,
            option_map=frame.event_option_map,
            q1_abnormal=bool(q1_abnormal),
            event_target=event_target,
            regular_event_codes=frame.regular_event_codes,
        )
        q2_local_indices.append(local_idx)
        q2_texts.append(q2_text)
        q2_ids_list.append(q2_ids)
        q2_q1_ids_list.append(q1_ids_list[local_idx])
        q2_student_prompts.append(q2_student_prompt)
        q2_teacher_prompts.append(q2_teacher_prompt)

    if q2_local_indices:
        q2_student_q1_messages = [q1_student_messages[i] for i in q2_local_indices]
        q2_teacher_q1_messages = [q1_teacher_messages[i] for i in q2_local_indices]
        with torch.no_grad():
            # Q2 parallel KL 不能把 q1_ids decode 成 q1_text 再重新 tokenize；那会和旧逐帧
            # 路径的“精确 q1_ids 追加到 Q1 KV”产生 token 边界漂移。这里按旧路径同款：
            # Q1 prompt prefill -> 追加每条样本自己的 q1_ids -> 追加 Q2 user turn。
            q2_student_prompt_state = _kv_start_state_batch_padded(bundle, q2_student_q1_messages)
            q2_student_after_q1 = _append_token_ids_padded_no_logits(bundle, q2_student_prompt_state, q2_q1_ids_list)
            q2_student_state = _append_user_turn_batch_padded(bundle, q2_student_after_q1, q2_student_prompts)
        with _teacher_eval_context(bundle):
            q2_teacher_prompt_state = _kv_start_state_batch_padded(bundle, q2_teacher_q1_messages)
            q2_teacher_after_q1 = _append_token_ids_padded_no_logits(bundle, q2_teacher_prompt_state, q2_q1_ids_list)
            q2_teacher_state = _append_user_turn_batch_padded(bundle, q2_teacher_after_q1, q2_teacher_prompts)
        q2_total_loss, q2_parts_list, q2_loss_list = _opsd_loss_batch_states(
            bundle,
            student_state=q2_student_state,
            teacher_state=q2_teacher_state,
            rollout_texts=q2_texts,
            rollout_ids_list=q2_ids_list,
            span_fn=target_spans_q2,
            weights=loss_weights_q2(),
            temperature=temperature,
        )
        total_loss = total_loss + q2_total_loss
        for local_idx, parts, loss_tensor in zip(q2_local_indices, q2_parts_list, q2_loss_list):
            q2_parts_by_local[local_idx] = parts
            q2_loss_by_local[local_idx] = loss_tensor

    per_frame_results: List[Tuple[int, Dict[str, Any], Memory, bool, float]] = []
    for local_idx, (b, _route, frame, memory) in enumerate(chunk):
        q1_parsed = q1_parsed_list[local_idx]
        q1_abnormal = q1_abnormal_flags[local_idx]
        q1_rs_correct = q1_parsed.get("rs_label") == frame.rs_label
        memory_after_q1 = update_memory_after_q1(
            memory,
            student_rs_label=q1_parsed.get("rs_label"),
            student_abnormal=q1_abnormal,
        )
        stats: Dict[str, Any] = {
            "q1_rs_correct": q1_rs_correct,
            "q1_abnormal_correct": q1_abnormal == frame.abnormal if q1_abnormal is not None else False,
            "q2_triggered": False,
            "candidate_mismatch": False,
            "q1_rollout_tokens": int(q1_ids_list[local_idx].numel()),
            "q2_rollout_tokens": 0,
            "q1_parts": q1_parts_list[local_idx],
            "timings": {
                # chunk 级 batch forward 已经合并计时；这里不把总耗时硬摊到每帧，
                # 避免和旧逐帧计时混淆。吞吐主要看 q1/q2 grouped seconds。
                "q1_student_seconds": 0.0,
                "q1_teacher_seconds": 0.0,
                "q1_loss_seconds": 0.0,
                "q2_rollout_seconds": 0.0,
                "q2_teacher_seconds": 0.0,
                "q2_loss_seconds": 0.0,
            },
        }
        frame_loss = q1_loss_list[local_idx]
        if not q1_rs_correct:
            per_frame_results.append((b, stats, memory_after_q1, True, float(frame_loss.detach().item())))
            continue
        q2_rollout = q2_rollouts[local_idx]
        assert q2_rollout is not None
        _q2_state, q2_text, q2_ids = q2_rollout
        q2_parsed = parse_q2_output(q2_text, frame.event_option_map)
        event_target = q2_event_targets[local_idx]
        target_option = option_for_event(event_target.label, frame.event_option_map)
        stats["candidate_mismatch"] = target_option is None
        student_event = q2_parsed.get("event_label")
        memory_after_q2 = update_memory_after_q2(memory_after_q1, student_event_label=student_event)
        q2_invalid = student_event is None
        q2_loss_tensor = q2_loss_by_local.get(local_idx, frame_loss * 0.0)
        frame_loss = frame_loss + q2_loss_tensor
        stats.update(
            {
                "q2_triggered": True,
                "q2_event_correct": student_event == event_target.label,
                "q2_invalid_output": q2_invalid,
                "q2_rollout_tokens": int(q2_ids.numel()),
                "q2_parts": q2_parts_by_local.get(local_idx, {key: 0.0 for key in loss_weights_q2()}),
            }
        )
        per_frame_results.append((b, stats, memory_after_q2, q2_invalid, float(frame_loss.detach().item())))
    return total_loss, per_frame_results


def _run_single_q1_rollout_from_images(
    bundle: Any,
    images: List[Image.Image],
    memory: Memory,
    *,
    max_new_tokens_q1: int,
) -> Tuple[KVState, str, KVState, torch.Tensor]:
    """单样本 Q1 rollout；供 grouped 路径中的 singleton group 复用。"""

    with torch.no_grad():
        state = _kv_start_state(bundle, _messages(images, build_q1_student_prompt(memory)))
        text, after, ids = _student_generate_kv(bundle, state, max_new_tokens_q1)
    return state, text, after, ids


def _run_q1_rollout_grouped(
    bundle: Any,
    memories: Sequence[Memory],
    frames: Sequence[FrameRow],
    *,
    max_new_tokens_q1: int,
) -> Q1GroupedRolloutResult:
    """运行 Q1 student rollout，并统计真实 batched 情况。

    返回值逐样本包含：
    `q1_student_state`（Q1 prompt prefill 后）、`q1_text`、`q1_after`（Q1 生成后干净 KV）、
    `q1_ids`。后续 `_run_frame` 会用这些 token 做同款 OPSD loss 和 Q2 状态机。

    注意：多样本 Q1 现在和 Q2 一样，padded batched KV 只用于 student 采样 token；
    返回的 state 置为 None，后续 `_run_frame` 会按单样本精确 prompt 重新构造
    q1_student_state/q1_after，再做 KL 和 Q2 续接。这样 mixed-length prompt 也能
    真正 4 路采样，同时不把 padded past_key_values 带入训练语义。
    """

    total_start = time.time()
    if not frames:
        return Q1GroupedRolloutResult(
            rollouts=[],
            input_lengths=[],
            group_sizes=[],
            batched_group_sizes=[],
            singleton_groups=0,
            batched_groups=0,
            batched_frames=0,
            length_histogram={},
            length_seconds=0.0,
            total_seconds=0.0,
        )
    # 先统一读取图片，后面 length 计算和真正 prefill 共用同一组 PIL 对象；
    # 这样不会因为重复打开文件导致随机 IO 波动太大。
    images_per_frame = [_load_images(frame.history_rgb_paths) for frame in frames]
    messages_list = [
        _messages(images, build_q1_student_prompt(memory))
        for images, memory in zip(images_per_frame, memories)
    ]
    length_start = time.time()
    # 逐条记录 processor input length 只用于日志里的 padding pressure 审计。
    # Q1 rollout 本身允许 mixed-length padded batch，训练用 KV 会在 _run_frame
    # 中按单样本精确重建。
    input_lengths = [_qwen_message_input_length(bundle, messages) for messages in messages_list]
    length_seconds = time.time() - length_start
    groups: Dict[int, List[int]] = {}
    for idx, length in enumerate(input_lengths):
        groups.setdefault(int(length), []).append(idx)
    outputs: List[Optional[Tuple[Optional[KVState], str, Optional[KVState], torch.Tensor]]] = [None for _ in frames]
    group_sizes: List[int] = [len(indices) for indices in groups.values()]
    batched_group_sizes: List[int] = []
    singleton_groups = 0
    if len(frames) == 1:
        singleton_groups = 1
        outputs[0] = _run_single_q1_rollout_from_images(
            bundle,
            images_per_frame[0],
            memories[0],
            max_new_tokens_q1=max_new_tokens_q1,
        )
    else:
        # mixed-length prompt 也放在同一个 padded batch 中采样；padded KV 不向外返回。
        # length_histogram 仍保留真实长度分布，方便从日志判断 prompt 长度差异。
        batched_group_sizes.append(len(frames))
        with torch.no_grad():
            state_batch = _kv_start_state_batch_padded(bundle, messages_list)
            texts, _after_states, ids_list = _student_generate_kv_batch(
                bundle,
                state_batch,
                max_new_tokens_q1,
                allow_padded_cache=True,
            )
        for idx in range(len(frames)):
            outputs[idx] = (None, texts[idx], None, ids_list[idx])
    assert all(item is not None for item in outputs)
    return Q1GroupedRolloutResult(
        rollouts=[item for item in outputs if item is not None],
        input_lengths=input_lengths,
        group_sizes=group_sizes,
        batched_group_sizes=batched_group_sizes,
        singleton_groups=singleton_groups,
        batched_groups=len(batched_group_sizes),
        batched_frames=sum(batched_group_sizes),
        length_histogram={int(length): len(indices) for length, indices in groups.items()},
        length_seconds=length_seconds,
        total_seconds=time.time() - total_start,
    )


def _run_q1_rollout_batch(
    bundle: Any,
    memories: Sequence[Memory],
    frames: Sequence[FrameRow],
    *,
    max_new_tokens_q1: int,
) -> List[Tuple[KVState, str, KVState, torch.Tensor]]:
    """兼容旧调用：只返回 rollouts，不返回 grouped 统计。"""

    return _run_q1_rollout_grouped(
        bundle,
        memories,
        frames,
        max_new_tokens_q1=max_new_tokens_q1,
    ).rollouts


def _run_q2_rollout_grouped(
    bundle: Any,
    *,
    frames: Sequence[FrameRow],
    memories: Sequence[Memory],
    q1_texts: Sequence[str],
    q1_abnormal_flags: Sequence[bool],
    max_new_tokens_q2: int,
) -> Q2GroupedRolloutResult:
    """用 mixed-length padded batch 运行 Q2 student rollout。

    返回的 `rollouts` 与输入顺序一一对应。Q2 rollout 是 no_grad 采样路径；n>1 时不再
    按 full-dialog length 拆组，而是整组走 padded prefill/generate。`input_lengths` /
    `length_histogram` 仍会记录 padding pressure，方便 smoke 和日志审计。返回的 state
    置为 None，外层会为 KL 重新构造精确 `q1_ids -> Q1 KV -> Q2 user turn` 状态。
    """

    total_start = time.time()
    n = len(frames)
    if n == 0:
        return Q2GroupedRolloutResult(
            rollouts=[],
            input_lengths=[],
            group_sizes=[],
            batched_group_sizes=[],
            singleton_groups=0,
            batched_groups=0,
            batched_frames=0,
            length_histogram={},
            length_seconds=0.0,
            total_seconds=0.0,
        )
    images_per_frame = [_load_images(frame.history_rgb_paths) for frame in frames]
    messages_list: List[List[Dict[str, Any]]] = []
    for images, memory, frame, q1_text, q1_abnormal in zip(images_per_frame, memories, frames, q1_texts, q1_abnormal_flags):
        q1_prompt = build_q1_student_prompt(memory)
        memory_after_q1 = update_memory_after_q1(
            memory,
            student_rs_label=parse_q1_output(q1_text).get("rs_label"),
            student_abnormal=q1_abnormal,
        )
        q2_prompt = build_q2_student_prompt(
            memory_after_q1,
            option_map=frame.event_option_map,
            q1_abnormal=bool(q1_abnormal),
            regular_event_codes=frame.regular_event_codes,
        )
        messages_list.append(_q2_full_messages(images, q1_prompt, q1_text, q2_prompt))

    length_start = time.time()
    input_lengths = [_qwen_message_input_length(bundle, messages) for messages in messages_list]
    length_seconds = time.time() - length_start
    groups: Dict[int, List[int]] = {}
    for idx, length in enumerate(input_lengths):
        groups.setdefault(int(length), []).append(idx)

    outputs: List[Optional[Tuple[Optional[KVState], str, torch.Tensor]]] = [None for _ in range(n)]
    group_sizes = [len(indices) for indices in groups.values()]
    batched_group_sizes: List[int] = []
    singleton_groups = 0
    if n == 1:
        singleton_groups = 1
        with torch.no_grad():
            state = _kv_start_state(bundle, messages_list[0])
            text, _after, ids = _student_generate_kv(bundle, state, max_new_tokens_q2)
        outputs[0] = (state, text, ids)
    else:
        batched_group_sizes.append(n)
        with torch.no_grad():
            state_batch = _kv_start_state_batch_padded(bundle, messages_list)
            texts, _after_states, ids_list = _student_generate_kv_batch(
                bundle,
                state_batch,
                max_new_tokens_q2,
                allow_padded_cache=True,
            )
        for idx in range(n):
            # state=None 是刻意的：padded rollout 只提供 student 采样文本/token；
            # KL 训练 state 在 _run_frame 中按单样本精确 KV 重新构造。
            outputs[idx] = (None, texts[idx], ids_list[idx])

    return Q2GroupedRolloutResult(
        rollouts=outputs,
        input_lengths=input_lengths,
        group_sizes=group_sizes,
        batched_group_sizes=batched_group_sizes,
        singleton_groups=singleton_groups,
        batched_groups=len(batched_group_sizes),
        batched_frames=sum(batched_group_sizes),
        length_histogram={int(length): len(indices) for length, indices in groups.items()},
        length_seconds=length_seconds,
        total_seconds=time.time() - total_start,
    )


def _trainable_param_groups(bundle: Any, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[nn.Parameter], List[nn.Parameter]]:
    """按语言/视觉 LoRA 分组，复用 v2/v3 的视觉保险口径。"""

    language: List[nn.Parameter] = []
    vision: List[nn.Parameter] = []
    for name, param in bundle.model.named_parameters():
        if not param.requires_grad:
            continue
        # 视觉 LoRA 默认关闭；如果用户显式开启，视觉参数单独用较小 LR 和较低 clip norm，
        # 防止少量 RS/EVENT 监督把 Qwen 视觉表征冲坏。
        if _is_vision_module_name(name):
            vision.append(param)
        else:
            language.append(param)
    groups: List[Dict[str, Any]] = []
    if language:
        groups.append({"params": language, "lr": float(args.learning_rate), "name": "language"})
    if vision:
        groups.append({"params": vision, "lr": float(args.learning_rate) * float(args.vision_lr_scale), "name": "vision"})
    return groups, language, vision


def _all_trainable_params(language_params: List[nn.Parameter], vision_params: List[nn.Parameter]) -> List[nn.Parameter]:
    """返回当前需要手动同步的 LoRA 参数列表。"""

    return [p for p in [*language_params, *vision_params] if p.requires_grad]


def _broadcast_trainable_params(params: List[nn.Parameter], *, src: int = 0) -> None:
    """启动时从 rank0 广播 LoRA 初始参数，确保各 rank 起点完全一致。"""

    if not (dist.is_available() and dist.is_initialized()):
        return
    for param in params:
        dist.broadcast(param.data, src=src)


def _sync_trainable_grads(params: List[nn.Parameter]) -> None:
    """手动 all-reduce LoRA 梯度。

    这里替代 DDP wrapper 的 reducer。v5 的 forward 轮数会因 Q2 触发门而在 rank
    之间不同，但每个 optimizer step 都会在所有 rank 上调用本函数，因此 collective
    次序固定，不会出现 DDP forward hook 的不匹配问题。
    """

    if not (dist.is_available() and dist.is_initialized()):
        return
    world = float(dist.get_world_size())
    for param in params:
        if param.grad is None:
            param.grad = torch.zeros_like(param.data)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world)


def _ddp_sum_int(value: int) -> int:
    """把一个 int 在所有 rank 上求和；单进程直接返回。"""

    if not (dist.is_available() and dist.is_initialized()):
        return int(value)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _is_cuda_oom(exc: BaseException) -> bool:
    """识别 CUDA OOM，避免 OOM 后静默 fallback 到不稳定的单样本路径。"""

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "cuda out of memory" in text or "out of memory" in text


def _cuda_memory_text() -> str:
    """返回当前 rank 的 CUDA 显存摘要，供长时间训练心跳日志使用。"""

    if not torch.cuda.is_available():
        return "cuda_mem=NA"
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return f"cuda_mem={allocated:.1f}G reserved={reserved:.1f}G peak={peak:.1f}G"


def _save_adapter(bundle: Any, output_dir: pathlib.Path, args: argparse.Namespace) -> None:
    """保存 LoRA adapter 与 v5 元数据。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    model = bundle.unwrap() if hasattr(bundle, "unwrap") else bundle.model
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(output_dir))
    meta = {
        "dataset_version": DATASET_VERSION,
        "lora_vision_scope": getattr(bundle, "lora_vision_scope", "off"),
        "lora_target_modules": getattr(bundle, "lora_target_modules", []),
        "max_new_tokens_q1": int(args.max_new_tokens_q1),
        "max_new_tokens_q2": int(args.max_new_tokens_q2),
        "temperature": float(args.temperature),
    }
    with open(output_dir / "sft_v5_adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def run_check(loader: DataLoader, *, max_batches: int = 2) -> None:
    """轻量检查 dataset 和 DDP padding，不加载模型。"""

    for batch_idx, batch in enumerate(loader):
        batch = pad_batch_to_global_length(batch)
        routes = batch["routes"]
        max_t = int(batch["max_T_global"])
        valid = batch["valid_mask"]
        print(f"[check] batch={batch_idx} routes={len(routes)} max_T_global={max_t} valid={int(valid.sum().item())}")
        for row in routes[:2]:
            first = row.frames[0]
            mem = _reset_memory_for_frame_row(first)
            print(
                f"  route={row.scenario}/{row.route_id} frames={len(row.frames)} "
                f"first_rs={first.rs_label} q2_options={first.event_option_map} mem={mem.rs_label}/{mem.event_label}"
            )
        if batch_idx + 1 >= max_batches:
            break


_TRAIN_WINDOW_KEYS = (
    "batches",
    "frames",
    "loss_sum",
    "q1_loss_analysis_sum",
    "q1_loss_rs_sum",
    "q1_loss_abnormal_sum",
    "q2_loss_analysis_sum",
    "q2_loss_event_sum",
    "q1_rs_correct",
    "q1_abnormal_correct",
    "q2_triggered",
    "q2_event_correct",
    "q2_invalid_output",
    "candidate_mismatch",
    "reset_next",
    "q1_rollout_tokens",
    "q2_rollout_tokens",
    "q1_token_cap_hits",
    "q2_token_cap_hits",
    "q1_grouped_chunks",
    "q1_grouped_frames",
    "q1_batched_groups",
    "q1_singleton_groups",
    "q1_batched_frames",
    "q1_length_seconds",
    "q1_grouped_seconds",
    "q2_grouped_chunks",
    "q2_grouped_frames",
    "q2_batched_groups",
    "q2_singleton_groups",
    "q2_batched_frames",
    "q2_length_seconds",
    "q2_grouped_seconds",
    "time_q1_student_seconds",
    "time_q1_teacher_seconds",
    "time_q1_loss_seconds",
    "time_q2_rollout_seconds",
    "time_q2_teacher_seconds",
    "time_q2_loss_seconds",
    "parallel_kl_chunks",
    "parallel_kl_frames",
    "parallel_kl_seconds",
    "parallel_kl_fallbacks",
    "valid_slots",
    "padding_slots",
    "max_T_local_sum",
    "max_T_global_sum",
)


def _new_train_window_stats() -> Dict[str, float]:
    """创建一个 logging window 内的 on-policy 采样/训练统计容器。"""

    return {key: 0.0 for key in _TRAIN_WINDOW_KEYS}


def _add_batch_shape_stats(stats: Dict[str, float], batch: Mapping[str, Any]) -> None:
    """记录本 rank 当前 batch 的 padding 形状。

    这些数字能直接审计 DDP padding 是否按预期运行：valid_slots 是真实 frame 数，
    padding_slots 是补齐到 global max_T 后额外占位的 None frame 数。
    """

    routes = batch.get("routes") or []
    valid = batch.get("valid_mask")
    valid_count = int(valid.sum().item()) if isinstance(valid, torch.Tensor) else 0
    max_t_global = int(batch.get("max_T_global", 0))
    max_t_local = int(batch.get("max_T_local", 0))
    total_slots = len(routes) * max_t_global
    stats["batches"] += 1.0
    stats["valid_slots"] += float(valid_count)
    stats["padding_slots"] += float(max(0, total_slots - valid_count))
    stats["max_T_local_sum"] += float(max_t_local)
    stats["max_T_global_sum"] += float(max_t_global)


def _add_frame_rollout_stats(
    stats: Dict[str, float],
    frame_stats: Mapping[str, Any],
    *,
    need_reset: bool,
    max_new_tokens_q1: int,
    max_new_tokens_q2: int,
) -> None:
    """累计一个有效 frame 的 on-policy rollout 统计。"""

    stats["frames"] += 1.0
    q1_parts = frame_stats.get("q1_parts") or {}
    if isinstance(q1_parts, Mapping):
        stats["q1_loss_analysis_sum"] += float(q1_parts.get("analysis", 0.0) or 0.0)
        stats["q1_loss_rs_sum"] += float(q1_parts.get("rs", 0.0) or 0.0)
        stats["q1_loss_abnormal_sum"] += float(q1_parts.get("abnormal", 0.0) or 0.0)
    q2_parts = frame_stats.get("q2_parts") or {}
    if isinstance(q2_parts, Mapping):
        stats["q2_loss_analysis_sum"] += float(q2_parts.get("analysis", 0.0) or 0.0)
        stats["q2_loss_event_sum"] += float(q2_parts.get("event", 0.0) or 0.0)
    stats["q1_rs_correct"] += float(bool(frame_stats.get("q1_rs_correct", False)))
    stats["q1_abnormal_correct"] += float(bool(frame_stats.get("q1_abnormal_correct", False)))
    stats["q2_triggered"] += float(bool(frame_stats.get("q2_triggered", False)))
    stats["q2_event_correct"] += float(bool(frame_stats.get("q2_event_correct", False)))
    stats["q2_invalid_output"] += float(bool(frame_stats.get("q2_invalid_output", False)))
    stats["candidate_mismatch"] += float(bool(frame_stats.get("candidate_mismatch", False)))
    stats["reset_next"] += float(bool(need_reset))
    q1_tokens = int(frame_stats.get("q1_rollout_tokens", 0) or 0)
    q2_tokens = int(frame_stats.get("q2_rollout_tokens", 0) or 0)
    stats["q1_rollout_tokens"] += float(q1_tokens)
    stats["q2_rollout_tokens"] += float(q2_tokens)
    # 1024 是防无限生成的安全上限，不是结构字段早停。这个指标专门用来发现
    # 模型是否经常没出 EOS / <|im_end|> 而打满上限；如果命中率高，远端日志里
    # q1_tokens/q2_tokens 会接近 max_new_tokens，parallel KL 也会显著更吃显存。
    stats["q1_token_cap_hits"] += float(q1_tokens >= int(max_new_tokens_q1) and int(max_new_tokens_q1) > 0)
    stats["q2_token_cap_hits"] += float(q2_tokens >= int(max_new_tokens_q2) and int(max_new_tokens_q2) > 0)
    timings = frame_stats.get("timings") or {}
    if isinstance(timings, Mapping):
        stats["time_q1_student_seconds"] += float(timings.get("q1_student_seconds", 0.0) or 0.0)
        stats["time_q1_teacher_seconds"] += float(timings.get("q1_teacher_seconds", 0.0) or 0.0)
        stats["time_q1_loss_seconds"] += float(timings.get("q1_loss_seconds", 0.0) or 0.0)
        stats["time_q2_rollout_seconds"] += float(timings.get("q2_rollout_seconds", 0.0) or 0.0)
        stats["time_q2_teacher_seconds"] += float(timings.get("q2_teacher_seconds", 0.0) or 0.0)
        stats["time_q2_loss_seconds"] += float(timings.get("q2_loss_seconds", 0.0) or 0.0)


def _add_q1_grouped_stats(stats: Dict[str, float], grouped: Q1GroupedRolloutResult) -> None:
    """累计 Q1 grouped rollout 的真实并行程度。"""

    # q1_grouped_frames 是“尝试过 grouped 路径”的 frame 数；
    # q1_batched_frames 是真正 size>=2 batched rollout 的 frame 数。二者的比值只代表
    # grouped 路径内部效率，不能当作全训练 frame 的 batch 生效率。
    stats["q1_grouped_chunks"] += 1.0
    stats["q1_grouped_frames"] += float(len(grouped.rollouts))
    stats["q1_batched_groups"] += float(grouped.batched_groups)
    stats["q1_singleton_groups"] += float(grouped.singleton_groups)
    stats["q1_batched_frames"] += float(grouped.batched_frames)
    stats["q1_length_seconds"] += float(grouped.length_seconds)
    stats["q1_grouped_seconds"] += float(grouped.total_seconds)


def _add_q2_grouped_stats(stats: Dict[str, float], grouped: Q2GroupedRolloutResult) -> None:
    """累计 Q2 grouped rollout 的真实并行程度。"""

    stats["q2_grouped_chunks"] += 1.0
    stats["q2_grouped_frames"] += float(len(grouped.rollouts))
    stats["q2_batched_groups"] += float(grouped.batched_groups)
    stats["q2_singleton_groups"] += float(grouped.singleton_groups)
    stats["q2_batched_frames"] += float(grouped.batched_frames)
    stats["q2_length_seconds"] += float(grouped.length_seconds)
    stats["q2_grouped_seconds"] += float(grouped.total_seconds)


def _ddp_sum_train_stats(stats: Mapping[str, float]) -> Dict[str, float]:
    """把各 rank 的 logging window 统计按 sum 聚合到所有 rank。"""

    values = [float(stats.get(key, 0.0)) for key in _TRAIN_WINDOW_KEYS]
    if not (dist.is_available() and dist.is_initialized()):
        return dict(zip(_TRAIN_WINDOW_KEYS, values))
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(value) for key, value in zip(_TRAIN_WINDOW_KEYS, tensor.cpu().tolist())}


def _format_train_window(stats: Mapping[str, float]) -> str:
    """把聚合后的 on-policy 统计格式化成一行日志。"""

    frames = max(1.0, float(stats.get("frames", 0.0)))
    q2 = max(1.0, float(stats.get("q2_triggered", 0.0)))
    batches = max(1.0, float(stats.get("batches", 0.0)))
    valid_slots = max(1.0, float(stats.get("valid_slots", 0.0)))
    rollout_tokens = float(stats.get("q1_rollout_tokens", 0.0)) + float(stats.get("q2_rollout_tokens", 0.0))
    grouped_frames = max(1.0, float(stats.get("q1_grouped_frames", 0.0)))
    batched_frames = float(stats.get("q1_batched_frames", 0.0))
    grouped_chunks = max(1.0, float(stats.get("q1_grouped_chunks", 0.0)))
    q2_grouped_frames = max(1.0, float(stats.get("q2_grouped_frames", 0.0)))
    q2_batched_frames = float(stats.get("q2_batched_frames", 0.0))
    q2_grouped_chunks = max(1.0, float(stats.get("q2_grouped_chunks", 0.0)))
    return (
        f"loss/frame={float(stats.get('loss_sum', 0.0)) / frames:.4f} "
        f"q1_loss={{a:{float(stats.get('q1_loss_analysis_sum', 0.0)) / frames:.3f},"
        f"rs:{float(stats.get('q1_loss_rs_sum', 0.0)) / frames:.3f},"
        f"abn:{float(stats.get('q1_loss_abnormal_sum', 0.0)) / frames:.3f}}} "
        f"q2_loss={{a:{float(stats.get('q2_loss_analysis_sum', 0.0)) / q2:.3f},"
        f"event:{float(stats.get('q2_loss_event_sum', 0.0)) / q2:.3f}}} "
        f"frames={int(stats.get('frames', 0.0))} "
        f"q2_rate={float(stats.get('q2_triggered', 0.0)) / frames:.3f} "
        f"rs_acc={float(stats.get('q1_rs_correct', 0.0)) / frames:.3f} "
        f"abn_acc={float(stats.get('q1_abnormal_correct', 0.0)) / frames:.3f} "
        f"event_acc={float(stats.get('q2_event_correct', 0.0)) / q2:.3f} "
        f"invalid={int(stats.get('q2_invalid_output', 0.0))} "
        f"reset={int(stats.get('reset_next', 0.0))} "
        f"tok/frame={rollout_tokens / frames:.1f} "
        f"cap_hit={{q1:{float(stats.get('q1_token_cap_hits', 0.0)) / frames:.3f},"
        f"q2:{float(stats.get('q2_token_cap_hits', 0.0)) / q2:.3f}}} "
        f"time/frame={{q1stu:{float(stats.get('time_q1_student_seconds', 0.0)) / frames:.2f},"
        f"q1teach:{float(stats.get('time_q1_teacher_seconds', 0.0)) / frames:.2f},"
        f"q1loss:{float(stats.get('time_q1_loss_seconds', 0.0)) / frames:.2f},"
        f"q2roll:{float(stats.get('time_q2_rollout_seconds', 0.0)) / frames:.2f},"
        f"q2teach:{float(stats.get('time_q2_teacher_seconds', 0.0)) / frames:.2f},"
        f"q2loss:{float(stats.get('time_q2_loss_seconds', 0.0)) / frames:.2f}}} "
        f"q1group/chunk={float(stats.get('q1_grouped_seconds', 0.0)) / grouped_chunks:.2f}s "
        f"q2group/chunk={float(stats.get('q2_grouped_seconds', 0.0)) / q2_grouped_chunks:.2f}s "
        f"q1_batched_all={batched_frames / frames:.3f} "
        f"q1_batched_grouped={batched_frames / grouped_frames:.3f} "
        f"q2_batched_all={q2_batched_frames / frames:.3f} "
        f"q2_batched_grouped={q2_batched_frames / q2_grouped_frames:.3f} "
        f"parallel_kl={float(stats.get('parallel_kl_frames', 0.0)) / frames:.3f} "
        f"parallel_kl/chunk={float(stats.get('parallel_kl_seconds', 0.0)) / max(1.0, float(stats.get('parallel_kl_chunks', 0.0))):.2f}s "
        f"parallel_fallbacks={int(stats.get('parallel_kl_fallbacks', 0.0))} "
        f"pad_rate={float(stats.get('padding_slots', 0.0)) / (valid_slots + float(stats.get('padding_slots', 0.0))):.3f} "
        f"maxT={float(stats.get('max_T_global_sum', 0.0)) / batches:.1f}"
    )


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""

    p = argparse.ArgumentParser(description="Train SFT v5 RS/EVENT OPSD")
    p.add_argument("--train-index", type=str, required=True)
    p.add_argument("--val-index", type=str, default=None)
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-vision-scope", type=str, default="off", choices=["off", "merger", "last4", "all"])
    p.add_argument("--vision-lr-scale", type=float, default=0.1)
    p.add_argument("--strict-vision-scope", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--language-clip-norm", type=float, default=1.0)
    p.add_argument("--vision-clip-norm", type=float, default=0.3)
    p.add_argument("--max-new-tokens-q1", type=int, default=1024)
    p.add_argument("--max-new-tokens-q2", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--qwen-batch-size", type=int, default=1, help="每个 rank/timestep 内尝试并行 Q1/Q2 student rollout 的 frame 数；Q1/Q2 允许 mixed-length padded rollout，KL 会重建单样本精确 KV")
    p.add_argument("--sampler-mode", type=str, default="length_balanced", choices=["length_balanced", "distributed"], help="多卡 route 分片方式；length_balanced 按 route frame 数均衡各 rank，distributed 为 PyTorch 默认 DistributedSampler")
    p.add_argument("--parallel-kl", action=argparse.BooleanOptionalAction, default=True, help="同一 timestep/chunk 内并行 Q1/Q2 teacher/student KL；失败时回退逐帧路径，CUDA OOM 直接中止")
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--progress-frames", type=int, default=5, help="rank0 每处理多少个本地有效 frame 打一次进度；0 表示关闭逐帧进度")
    p.add_argument("--heartbeat-seconds", type=float, default=120.0, help="rank0 长操作超过多少秒补一条心跳；0 表示关闭按时间心跳")
    p.add_argument("--check", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true", help="experimental; KV-cache OPSD normally keeps this off")
    p.add_argument("--no-grad-checkpoint", action="store_true", help="legacy compatibility flag; v5 keeps grad checkpoint off by default")
    p.add_argument("--seed", type=int, default=20260711)
    return p.parse_args()


def main() -> None:
    """训练主入口。"""

    args = parse_args()
    torch.manual_seed(int(args.seed))
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train_ds = RouteSequenceDataset(
        pathlib.Path(args.train_index),
        max_routes=int(args.max_routes),
        max_frames_per_route=int(args.max_frames_per_route),
    )
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        if str(args.sampler_mode) == "length_balanced":
            # 默认使用长度均衡 sampler：仍然保证每个 rank 的 route 数 / batch 数一致，
            # 但尽量让每个 rank 的总 frame 数接近，减少长 route rank 拖住其它 rank。
            sampler = LengthBalancedDistributedSampler(
                train_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=int(args.seed),
            )
        else:
            # PyTorch 默认 DistributedSampler 只均衡 route 数，不均衡 route frame 数。
            # 保留该模式方便和旧 run 做对照。
            sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=int(args.seed))
    else:
        sampler = None
    loader = DataLoader(
        train_ds,
        batch_size=int(args.per_device_batch_size),
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=int(args.num_workers),
        collate_fn=collate_route_sequences,
    )
    if args.check:
        run_check(loader)
        cleanup_distributed()
        return

    output_dir = pathlib.Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier()

    bundle = load_model_with_lora(
        pathlib.Path(args.model_dir),
        device=device,
        lora_rank=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        lora_vision_scope=str(args.lora_vision_scope),
        strict_vision_scope=bool(args.strict_vision_scope),
        gradient_checkpointing=bool(args.grad_checkpoint),
    )
    groups, language_params, vision_params = _trainable_param_groups(bundle, args)
    trainable_params = _all_trainable_params(language_params, vision_params)
    _broadcast_trainable_params(trainable_params, src=0)
    if rank == 0 and bool(args.grad_checkpoint):
        print("[warn] --grad-checkpoint is experimental for v5: Qwen KV-cache generation requires use_cache=True.")
    optimizer = torch.optim.AdamW(groups, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    total_steps = max(1, math.ceil(len(loader) * int(args.num_epochs) / max(1, int(args.grad_accum))))
    scheduler = make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=int(total_steps * float(args.warmup_ratio)),
    )
    tb = SummaryWriter(str(output_dir / "tb")) if rank == 0 and _TB_AVAILABLE else None

    global_step = 0
    micro_step = 0
    processed_local_frames = 0
    window_stats = _new_train_window_stats()
    start = time.time()
    last_heartbeat = start
    bundle.model.train()

    def rank0_log(message: str) -> None:
        """rank0 统一日志出口，flush=True 保证 tee/log.txt 里能实时看到。"""

        if rank == 0:
            print(message, flush=True)

    def should_log_frame() -> bool:
        """判断本地 frame 心跳是否应该输出。"""

        interval = int(args.progress_frames)
        if interval <= 0:
            return False
        # 前几个 frame 总是打印，方便判断训练是否已经进入真实 Qwen OPSD。
        return processed_local_frames < 3 or processed_local_frames % interval == 0

    def should_time_heartbeat(now: Optional[float] = None) -> bool:
        """判断距离上次日志是否已经超过 heartbeat 秒。"""

        interval = float(args.heartbeat_seconds)
        if interval <= 0:
            return False
        return ((time.time() if now is None else now) - last_heartbeat) >= interval

    def complete_optimizer_step(epoch_idx: int) -> None:
        """完成一次全 rank 对齐的 LoRA optimizer step。"""

        nonlocal global_step, window_stats, last_heartbeat
        sync_start = time.time()
        rank0_log(
            f"[sync-start] step_next={global_step + 1} epoch={epoch_idx} "
            f"local_window_frames={int(window_stats.get('frames', 0.0))} {_cuda_memory_text()}"
        )
        _sync_trainable_grads(trainable_params)
        rank0_log(f"[sync-done] step_next={global_step + 1} grad_all_reduce={time.time() - sync_start:.1f}s {_cuda_memory_text()}")
        if language_params:
            torch.nn.utils.clip_grad_norm_(language_params, float(args.language_clip_norm))
        if vision_params:
            torch.nn.utils.clip_grad_norm_(vision_params, float(args.vision_clip_norm))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        last_heartbeat = time.time()
        log_every = max(1, int(args.logging_steps))
        if global_step % log_every == 0:
            reduced_stats = _ddp_sum_train_stats(window_stats)
            # 所有 rank 都必须在 collective 后清零窗口；只有 rank0 负责输出人可读日志。
            window_stats = _new_train_window_stats()
            elapsed = time.time() - start
            if rank == 0:
                log_line = _format_train_window(reduced_stats)
                print(f"[train] step={global_step} epoch={epoch_idx} {log_line} elapsed={elapsed:.1f}s {_cuda_memory_text()}", flush=True)
                if tb is not None:
                    frames = max(1.0, reduced_stats["frames"])
                    q2 = max(1.0, reduced_stats["q2_triggered"])
                    valid = max(1.0, reduced_stats["valid_slots"])
                    pad_total = valid + reduced_stats["padding_slots"]
                    tb.add_scalar("train/loss_frame", reduced_stats["loss_sum"] / frames, global_step)
                    tb.add_scalar("train/loss/q1_analysis", reduced_stats["q1_loss_analysis_sum"] / frames, global_step)
                    tb.add_scalar("train/loss/q1_rs", reduced_stats["q1_loss_rs_sum"] / frames, global_step)
                    tb.add_scalar("train/loss/q1_abnormal", reduced_stats["q1_loss_abnormal_sum"] / frames, global_step)
                    tb.add_scalar("train/loss/q2_analysis", reduced_stats["q2_loss_analysis_sum"] / q2, global_step)
                    tb.add_scalar("train/loss/q2_event", reduced_stats["q2_loss_event_sum"] / q2, global_step)
                    tb.add_scalar("train/q2_trigger_rate", reduced_stats["q2_triggered"] / frames, global_step)
                    tb.add_scalar("train/q1_rs_acc_window", reduced_stats["q1_rs_correct"] / frames, global_step)
                    tb.add_scalar("train/q1_abnormal_acc_window", reduced_stats["q1_abnormal_correct"] / frames, global_step)
                    tb.add_scalar("train/q2_event_acc_window", reduced_stats["q2_event_correct"] / q2, global_step)
                    tb.add_scalar("train/q2_invalid_output", reduced_stats["q2_invalid_output"], global_step)
                    tb.add_scalar("train/reset_next", reduced_stats["reset_next"], global_step)
                    tb.add_scalar("train/rollout_tokens_per_frame", (reduced_stats["q1_rollout_tokens"] + reduced_stats["q2_rollout_tokens"]) / frames, global_step)
                    tb.add_scalar("train/q1_token_cap_hit_rate", reduced_stats["q1_token_cap_hits"] / frames, global_step)
                    tb.add_scalar("train/q2_token_cap_hit_rate", reduced_stats["q2_token_cap_hits"] / q2, global_step)
                    q1_grouped_frames = max(1.0, reduced_stats["q1_grouped_frames"])
                    q1_grouped_chunks = max(1.0, reduced_stats["q1_grouped_chunks"])
                    # 这三个 qwen 指标的分母刻意不同：
                    # - q1_batched_frame_rate：全训练 frame 口径，判断 QWEN_BATCH_SIZE 是否真有收益；
                    # - q1_grouped_frame_rate：有多少 frame 进入了 grouped 尝试路径；
                    # - q1_batched_frame_rate_grouped：只看 grouped 路径内部，排查 fallback/有效 batch 率。
                    tb.add_scalar("qwen/q1_batched_frame_rate", reduced_stats["q1_batched_frames"] / frames, global_step)
                    tb.add_scalar("qwen/q1_grouped_frame_rate", reduced_stats["q1_grouped_frames"] / frames, global_step)
                    tb.add_scalar("qwen/q1_batched_frame_rate_grouped", reduced_stats["q1_batched_frames"] / q1_grouped_frames, global_step)
                    tb.add_scalar("qwen/q1_batched_groups", reduced_stats["q1_batched_groups"], global_step)
                    tb.add_scalar("qwen/q1_singleton_groups", reduced_stats["q1_singleton_groups"], global_step)
                    q2_grouped_frames = max(1.0, reduced_stats["q2_grouped_frames"])
                    q2_grouped_chunks = max(1.0, reduced_stats["q2_grouped_chunks"])
                    tb.add_scalar("qwen/q2_batched_frame_rate", reduced_stats["q2_batched_frames"] / frames, global_step)
                    tb.add_scalar("qwen/q2_grouped_frame_rate", reduced_stats["q2_grouped_frames"] / frames, global_step)
                    tb.add_scalar("qwen/q2_batched_frame_rate_grouped", reduced_stats["q2_batched_frames"] / q2_grouped_frames, global_step)
                    tb.add_scalar("qwen/q2_batched_groups", reduced_stats["q2_batched_groups"], global_step)
                    tb.add_scalar("qwen/q2_singleton_groups", reduced_stats["q2_singleton_groups"], global_step)
                    # 如果这个耗时很高而 q1_batched_frame_rate 很低，说明逐帧 processor
                    # 长度预计算比真实 batch 省下的时间还贵，应先退回 QWEN_BATCH_SIZE=1。
                    tb.add_scalar("qwen/q1_length_seconds_per_chunk", reduced_stats["q1_length_seconds"] / q1_grouped_chunks, global_step)
                    tb.add_scalar("qwen/q1_grouped_seconds_per_chunk", reduced_stats["q1_grouped_seconds"] / q1_grouped_chunks, global_step)
                    tb.add_scalar("qwen/q2_length_seconds_per_chunk", reduced_stats["q2_length_seconds"] / q2_grouped_chunks, global_step)
                    tb.add_scalar("qwen/q2_grouped_seconds_per_chunk", reduced_stats["q2_grouped_seconds"] / q2_grouped_chunks, global_step)
                    # 这些 time/* 指标用于判断 GPU 利用率低到底卡在 Q1 batched rollout、
                    # Q1/Q2 teacher prefill，还是 Q2 rollout / KL 单样本路径。
                    tb.add_scalar("time/frame_q1_student_seconds", reduced_stats["time_q1_student_seconds"] / frames, global_step)
                    tb.add_scalar("time/frame_q1_teacher_seconds", reduced_stats["time_q1_teacher_seconds"] / frames, global_step)
                    tb.add_scalar("time/frame_q1_loss_seconds", reduced_stats["time_q1_loss_seconds"] / frames, global_step)
                    tb.add_scalar("time/frame_q2_rollout_seconds", reduced_stats["time_q2_rollout_seconds"] / frames, global_step)
                    tb.add_scalar("time/frame_q2_teacher_seconds", reduced_stats["time_q2_teacher_seconds"] / frames, global_step)
                    tb.add_scalar("time/frame_q2_loss_seconds", reduced_stats["time_q2_loss_seconds"] / frames, global_step)
                    tb.add_scalar("parallel_kl/frame_rate", reduced_stats["parallel_kl_frames"] / frames, global_step)
                    tb.add_scalar("parallel_kl/seconds_per_chunk", reduced_stats["parallel_kl_seconds"] / max(1.0, reduced_stats["parallel_kl_chunks"]), global_step)
                    tb.add_scalar("parallel_kl/fallbacks", reduced_stats["parallel_kl_fallbacks"], global_step)
                    tb.add_scalar("ddp/padding_rate", reduced_stats["padding_slots"] / max(1.0, pad_total), global_step)
                    tb.add_scalar("ddp/max_T_global_avg", reduced_stats["max_T_global_sum"] / max(1.0, reduced_stats["batches"]), global_step)
        if rank == 0 and int(args.save_steps) > 0 and global_step % int(args.save_steps) == 0:
            _save_adapter(bundle, output_dir / f"checkpoint-{global_step}", args)

    for epoch in range(int(args.num_epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)
            if isinstance(sampler, LengthBalancedDistributedSampler):
                # 所有 rank 都进入这三个 all-reduce，rank0 打印本 epoch 的 sampler
                # 均衡效果。若 max/min 仍差很多，说明 route 长度极端或 batch size 太小。
                local_sampler_frames = sampler.local_epoch_frame_count()
                global_sampler_frames = _ddp_sum_int(local_sampler_frames)
                max_sampler_frames = _global_max_int(local_sampler_frames)
                min_sampler_frames = _global_min_int(local_sampler_frames)
                rank0_log(
                    f"[sampler] epoch={epoch} mode=length_balanced "
                    f"local_rank0_frames={local_sampler_frames} "
                    f"global_frames={global_sampler_frames} "
                    f"avg_per_rank={global_sampler_frames / max(1, world_size):.1f} "
                    f"min_rank_frames={min_sampler_frames} max_rank_frames={max_sampler_frames}"
                )
        for batch_idx, batch in enumerate(loader):
            batch = pad_batch_to_global_length(batch)
            _add_batch_shape_stats(window_stats, batch)
            frame_count = 0
            batch_processed_frames = 0
            routes: List[SequenceRow] = batch["routes"]
            frame_rows: List[List[Optional[FrameRow]]] = batch["frame_rows"]
            local_frame_slots = sum(1 for frames in frame_rows for frame in frames if frame is not None)
            local_loss_slots = sum(1 for frames in frame_rows for frame in frames if frame is not None and _frame_images_exist(frame))
            # slots 统计分两层：
            # - frame_slots：DataLoader 给出的非 padding frame，用来观察 sequence/global padding；
            # - loss_slots：RGB 路径也存在的 frame，用作 loss 归一化分母。
            # 正常数据两者一致；如果个别 frame 缺图，训练会 skip 且不稀释 loss。
            global_frame_slots = _ddp_sum_int(local_frame_slots)
            global_loss_slots = _ddp_sum_int(local_loss_slots)
            qwen_batch_size = max(1, int(args.qwen_batch_size))
            rank0_log(
                f"[batch-start] epoch={epoch} batch={batch_idx} routes={len(routes)} "
                f"maxT_local={int(batch['max_T_local'])} maxT_global={int(batch['max_T_global'])} "
                f"valid_local={local_frame_slots} valid_global={global_frame_slots} "
                f"loss_local={local_loss_slots} loss_global={global_loss_slots} qwen_batch={qwen_batch_size} "
                f"micro_step_next={micro_step + 1} {_cuda_memory_text()}"
            )
            # 梯度同步函数会 all_reduce(SUM) 后再除以 world_size；这里预乘 world_size，
            # 使最终等价于按 global 有效 frame 数平均，而不是每个 rank 等权平均。
            # 分母使用实际可训练 frame（RGB 路径存在）数量，避免少量缺图 skip 把 loss
            # 额外缩小；正常 build_dataset 已过滤缺图时它与 valid_global 相同。
            loss_scale = float(world_size) / float(max(1, global_loss_slots) * max(1, int(args.grad_accum)))
            # 每条 route 各自维护 memory/reset；batch 内 route 之间互不影响。
            # reset_next=True 表示上一个有效帧 RS 错或 Q2 非法，下一帧开头恢复 GT RS + RE。
            reset_next = [False for _ in routes]
            memories: List[Optional[Memory]] = [None for _ in routes]
            for t in range(int(batch["max_T_global"])):
                active_items: List[Tuple[int, SequenceRow, FrameRow, Memory]] = []
                for b, route in enumerate(routes):
                    frame = frame_rows[b][t]
                    if frame is None:
                        # global padding 位置不参与任何计算，确保 DDP 对齐只影响 loop 长度，
                        # 不影响训练样本数量和 loss。
                        continue
                    if not _frame_images_exist(frame):
                        if rank == 0:
                            rank0_log(f"[warn] skip missing image paths route={route.route_id} frame={frame.frame_id}")
                        continue
                    rs_target = _rs_target_from_frame(frame)
                    if memories[b] is None or reset_next[b]:
                        memories[b] = reset_memory_for_frame(rs_target, ego_to_goal_xy=frame.ego_to_goal_xy)
                        reset_next[b] = False
                    assert memories[b] is not None
                    active_items.append((b, route, frame, memories[b]))
                for chunk_start in range(0, len(active_items), qwen_batch_size):
                    chunk = active_items[chunk_start:chunk_start + qwen_batch_size]
                    q1_rollouts: List[Optional[Tuple[Optional[KVState], str, Optional[KVState], torch.Tensor]]] = [None] * len(chunk)
                    q2_rollouts: List[Optional[Tuple[Optional[KVState], str, torch.Tensor]]] = [None] * len(chunk)
                    if qwen_batch_size > 1 and len(chunk) > 1:
                        # 只把同一 timestep 的多条 route 合成 Q1 rollout chunk。Q1/Q2 的
                        # padded batched KV 只用于采样文本/token；KL 和 Q2 续接会在 _run_frame
                        # 里重建单样本精确 KV，避免 padding prefix length 污染训练语义。
                        q1_batch_start = time.time()
                        try:
                            q1_grouped = _run_q1_rollout_grouped(
                                bundle,
                                [item[3] for item in chunk],
                                [item[2] for item in chunk],
                                max_new_tokens_q1=int(args.max_new_tokens_q1),
                            )
                            q1_rollouts = list(q1_grouped.rollouts)
                            _add_q1_grouped_stats(window_stats, q1_grouped)
                            if rank == 0:
                                rank0_log(
                                    f"[q1-grouped] epoch={epoch} batch={batch_idx} t={t} "
                                    f"size={len(chunk)} group_sizes={q1_grouped.group_sizes} "
                                    f"batched_groups={q1_grouped.batched_groups} "
                                    f"singleton_groups={q1_grouped.singleton_groups} "
                                    f"batched_frames={q1_grouped.batched_frames} "
                                    f"length_hist={q1_grouped.length_histogram} "
                                    f"length_s={q1_grouped.length_seconds:.3f} "
                                    f"dt={time.time() - q1_batch_start:.1f}s {_cuda_memory_text()}"
                                )
                        except Exception as exc:
                            if _is_cuda_oom(exc):
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                rank0_log(
                                    f"[error] q1 batch OOM epoch={epoch} batch={batch_idx} t={t} "
                                    f"size={len(chunk)}; abort instead of unsafe fallback. {_cuda_memory_text()}"
                                )
                                # OOM 后 CUDA allocator / NCCL 状态可能已经不干净。这里直接
                                # 抛出让 torchrun 重启/停止，比静默 fallback 单样本更安全。
                                raise
                            # batched Qwen 是新增优化路径；任何 processor/cache 兼容问题都回退
                            # 到旧单帧路径，保证训练语义优先于吞吐。
                            if rank == 0:
                                rank0_log(
                                    f"[warn] q1 batch fallback epoch={epoch} batch={batch_idx} t={t} "
                                    f"size={len(chunk)} reason={type(exc).__name__}: {exc}"
                                )
                                if os.environ.get("Q1_BATCH_TRACEBACK", "0") == "1":
                                    # 默认只打一行 fallback，避免训练日志爆炸；需要追新兼容问题时
                                    # 再显式打开 traceback，能定位是 processor、prefill 还是增量
                                    # decode 触发了回退。
                                    rank0_log(traceback.format_exc(limit=12).rstrip())
                            q1_rollouts = [None] * len(chunk)
                            # Q2 student rollout 也要为 singleton candidate 生成；否则某个
                            # chunk 里只有 1 个 Q1-correct frame 时，parallel KL 会因为缺
                            # q2_rollout 整块 fallback 到旧逐帧路径。
                        q2_candidate_local_indices: List[int] = []
                        q2_candidate_frames: List[FrameRow] = []
                        q2_candidate_memories: List[Memory] = []
                        q2_candidate_q1_texts: List[str] = []
                        q2_candidate_abnormal: List[bool] = []
                        for local_idx, rollout in enumerate(q1_rollouts):
                            if rollout is None:
                                continue
                            _b, _route, frame_for_q2, memory_for_q2 = chunk[local_idx]
                            _q1_state, q1_text_for_q2, _q1_after, _q1_ids = rollout
                            q1_parsed_for_q2 = parse_q1_output(q1_text_for_q2)
                            if q1_parsed_for_q2.get("rs_label") != frame_for_q2.rs_label:
                                continue
                            q2_candidate_local_indices.append(local_idx)
                            q2_candidate_frames.append(frame_for_q2)
                            q2_candidate_memories.append(memory_for_q2)
                            q2_candidate_q1_texts.append(q1_text_for_q2)
                            q2_candidate_abnormal.append(bool(q1_parsed_for_q2.get("abnormal") == "YES"))
                        if len(q2_candidate_frames) >= 1:
                            q2_batch_start = time.time()
                            try:
                                q2_grouped = _run_q2_rollout_grouped(
                                    bundle,
                                    frames=q2_candidate_frames,
                                    memories=q2_candidate_memories,
                                    q1_texts=q2_candidate_q1_texts,
                                    q1_abnormal_flags=q2_candidate_abnormal,
                                    max_new_tokens_q2=int(args.max_new_tokens_q2),
                                )
                                _add_q2_grouped_stats(window_stats, q2_grouped)
                                for candidate_idx, rollout in zip(q2_candidate_local_indices, q2_grouped.rollouts):
                                    q2_rollouts[candidate_idx] = rollout
                                if rank == 0:
                                    rank0_log(
                                        f"[q2-grouped] epoch={epoch} batch={batch_idx} t={t} "
                                        f"size={len(q2_candidate_frames)} group_sizes={q2_grouped.group_sizes} "
                                        f"batched_groups={q2_grouped.batched_groups} "
                                        f"singleton_groups={q2_grouped.singleton_groups} "
                                        f"batched_frames={q2_grouped.batched_frames} "
                                        f"length_hist={q2_grouped.length_histogram} "
                                        f"length_s={q2_grouped.length_seconds:.3f} "
                                        f"dt={time.time() - q2_batch_start:.1f}s {_cuda_memory_text()}"
                                    )
                            except Exception as exc:
                                if _is_cuda_oom(exc):
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    rank0_log(
                                        f"[error] q2 batch OOM epoch={epoch} batch={batch_idx} t={t} "
                                        f"size={len(q2_candidate_frames)}; abort instead of unsafe fallback. {_cuda_memory_text()}"
                                    )
                                    raise
                                if rank == 0:
                                    rank0_log(
                                        f"[warn] q2 batch fallback epoch={epoch} batch={batch_idx} t={t} "
                                        f"size={len(q2_candidate_frames)} reason={type(exc).__name__}: {exc}"
                                    )
                                    if os.environ.get("Q2_BATCH_TRACEBACK", "0") == "1":
                                        rank0_log(traceback.format_exc(limit=12).rstrip())
                                q2_rollouts = [None] * len(chunk)
                    parallel_chunk_done = False
                    if bool(args.parallel_kl) and len(chunk) > 1 and all(rollout is not None for rollout in q1_rollouts):
                        parallel_start = time.time()
                        try:
                            chunk_loss, chunk_results = _run_chunk_parallel_kl(
                                bundle,
                                chunk,
                                q1_rollouts=q1_rollouts,
                                q2_rollouts=q2_rollouts,
                                temperature=float(args.temperature),
                            )
                            # chunk_loss 是当前 chunk 内各 frame loss 的 sum；乘同一个
                            # global loss_scale 后一次 backward，减少逐帧小 backward 和
                            # teacher/student KL 小 forward 带来的 GPU 碎片化。
                            if not bool(chunk_loss.requires_grad):
                                # 极端情况下，chunk 内所有样本的 student 输出都没有命中
                                # RS/ABNORMAL/EVENT 等监督字段，batched KL 会得到一个数值为
                                # 0 的 loss。这里仍然需要让 backward 走完，保证本 rank 与
                                # 其它 rank 的梯度同步节奏一致。
                                if rank == 0:
                                    rank0_log(
                                        f"[warn] chunk loss has no grad; replace with graph-zero "
                                        f"epoch={epoch} batch={batch_idx} t={t} size={len(chunk)}"
                                    )
                                chunk_loss = _trainable_graph_zero(bundle, chunk_loss)
                            (chunk_loss * loss_scale).backward()
                            chunk_loss_value = float(chunk_loss.detach().item())
                            frame_count += len(chunk_results)
                            batch_processed_frames += len(chunk_results)
                            processed_local_frames += len(chunk_results)
                            q1_tok = 0
                            q2_tok = 0
                            reset_count = 0
                            for b_result, stats, next_mem, need_reset, frame_loss_value in chunk_results:
                                q1_tok += int(stats.get("q1_rollout_tokens", 0) or 0)
                                q2_tok += int(stats.get("q2_rollout_tokens", 0) or 0)
                                reset_count += int(bool(need_reset))
                                window_stats["loss_sum"] += float(frame_loss_value)
                                _add_frame_rollout_stats(
                                    window_stats,
                                    stats,
                                    need_reset=bool(need_reset),
                                    max_new_tokens_q1=int(args.max_new_tokens_q1),
                                    max_new_tokens_q2=int(args.max_new_tokens_q2),
                                )
                                memories[b_result] = next_mem
                                reset_next[b_result] = bool(need_reset)
                            if rank == 0 and (should_log_frame() or should_time_heartbeat()):
                                rank0_log(
                                    f"[chunk-train] epoch={epoch} batch={batch_idx} t={t} "
                                    f"size={len(chunk_results)} dt={time.time() - parallel_start:.1f}s "
                                    f"loss_sum={chunk_loss_value:.4f} "
                                    f"q1_tokens={q1_tok} q2_tokens={q2_tok} resets={reset_count} "
                                    f"parallel_kl=1 {_cuda_memory_text()}"
                                )
                                last_heartbeat = time.time()
                            window_stats["parallel_kl_chunks"] += 1.0
                            window_stats["parallel_kl_frames"] += float(len(chunk_results))
                            window_stats["parallel_kl_seconds"] += float(time.time() - parallel_start)
                            parallel_chunk_done = True
                        except Exception as exc:
                            if _is_cuda_oom(exc):
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                rank0_log(
                                    f"[error] parallel KL OOM epoch={epoch} batch={batch_idx} t={t} "
                                    f"size={len(chunk)}; abort instead of unsafe fallback. {_cuda_memory_text()}"
                                )
                                raise
                            if rank == 0:
                                rank0_log(
                                    f"[warn] parallel KL fallback epoch={epoch} batch={batch_idx} t={t} "
                                    f"size={len(chunk)} reason={type(exc).__name__}: {exc}"
                                )
                                if os.environ.get("PARALLEL_KL_TRACEBACK", "0") == "1":
                                    rank0_log(traceback.format_exc(limit=12).rstrip())
                            window_stats["parallel_kl_fallbacks"] += 1.0
                    if parallel_chunk_done:
                        continue
                    for chunk_idx, (b, route, frame, memory_for_frame) in enumerate(chunk):
                        frame_start = time.time()
                        log_this_frame = should_log_frame()
                        if log_this_frame:
                            rank0_log(
                                f"[frame-start] epoch={epoch} batch={batch_idx} t={t} route_idx={b} "
                                f"route={route.scenario}/{route.route_id} frame={frame.frame_id} "
                                f"batch_frame={batch_processed_frames + 1}/{local_frame_slots} "
                                f"mem={memories[b].rs_label}/{memories[b].event_label} {_cuda_memory_text()}"
                            )
                        try:
                            rollout = q1_rollouts[chunk_idx]
                            rollout_kwargs: Dict[str, Any] = {}
                            if rollout is not None:
                                # grouped/batched 路径已经完成 Q1 student rollout；如果其中
                                # state/after 为 None，_run_frame 会复用这些 token 重建单样本
                                # 精确 KV，只是不再重复生成 Q1 文本。
                                q1_student_state, q1_text, q1_after, q1_ids = rollout
                                rollout_kwargs = {
                                    "q1_student_state": q1_student_state,
                                    "q1_text": q1_text,
                                    "q1_after": q1_after,
                                    "q1_ids": q1_ids,
                                }
                            q2_rollout = q2_rollouts[chunk_idx]
                            if q2_rollout is not None:
                                q2_student_state, q2_text, q2_ids = q2_rollout
                                rollout_kwargs.update(
                                    {
                                        "q2_student_state": q2_student_state,
                                        "q2_text": q2_text,
                                        "q2_ids": q2_ids,
                                    }
                                )
                            loss, stats, next_mem, need_reset = _run_frame(
                                bundle,
                                memory_for_frame,
                                frame,
                                max_new_tokens_q1=int(args.max_new_tokens_q1),
                                max_new_tokens_q2=int(args.max_new_tokens_q2),
                                temperature=float(args.temperature),
                                **rollout_kwargs,
                            )
                        except FileNotFoundError as exc:
                            if rank == 0:
                                print(f"[warn] skip missing image route={route.route_id} frame={frame.frame_id}: {exc}", flush=True)
                            continue
                        # 关键显存控制：每帧 OPSD loss 算完立刻 backward，只把 LoRA
                        # 梯度累积在参数上，不把整条 route sequence 的 Qwen 计算图留到
                        # batch 末尾。这样长序列不会把 H20 95GB 显存吃满。
                        loss_value = float(loss.detach().item())
                        window_stats["loss_sum"] += loss_value
                        if not bool(loss.requires_grad):
                            if rank == 0:
                                rank0_log(
                                    f"[warn] loss has no grad; replace with graph-zero "
                                    f"route={route.scenario}/{route.route_id} frame={frame.frame_id}"
                                )
                            loss = _trainable_graph_zero(bundle, loss)
                        (loss * loss_scale).backward()
                        del loss
                        frame_count += 1
                        batch_processed_frames += 1
                        processed_local_frames += 1
                        _add_frame_rollout_stats(
                            window_stats,
                            stats,
                            need_reset=bool(need_reset),
                            max_new_tokens_q1=int(args.max_new_tokens_q1),
                            max_new_tokens_q2=int(args.max_new_tokens_q2),
                        )
                        memories[b] = next_mem
                        reset_next[b] = bool(need_reset)
                        frame_elapsed = time.time() - frame_start
                        now = time.time()
                        if log_this_frame or should_time_heartbeat(now):
                            timings = stats.get("timings") or {}
                            rank0_log(
                                f"[frame-done] epoch={epoch} batch={batch_idx} t={t} route_idx={b} "
                                f"frame={frame.frame_id} dt={frame_elapsed:.1f}s "
                                f"loss={loss_value:.4f} "
                                f"q1_tokens={int(stats.get('q1_rollout_tokens', 0))} "
                                f"q2_tokens={int(stats.get('q2_rollout_tokens', 0))} "
                                f"time={{q1stu:{float(timings.get('q1_student_seconds', 0.0)):.1f},"
                                f"q1teach:{float(timings.get('q1_teacher_seconds', 0.0)):.1f},"
                                f"q1loss:{float(timings.get('q1_loss_seconds', 0.0)):.1f},"
                                f"q2roll:{float(timings.get('q2_rollout_seconds', 0.0)):.1f},"
                                f"q2teach:{float(timings.get('q2_teacher_seconds', 0.0)):.1f},"
                                f"q2loss:{float(timings.get('q2_loss_seconds', 0.0)):.1f}}} "
                                f"q2={int(bool(stats.get('q2_triggered', False)))} "
                                f"reset={int(bool(need_reset))} {_cuda_memory_text()}"
                            )
                            last_heartbeat = now
            rank0_log(
                f"[batch-local-done] epoch={epoch} batch={batch_idx} local_frames={frame_count} "
                f"calling_global_frame_reduce=1 {_cuda_memory_text()}"
            )
            global_frame_count = _ddp_sum_int(frame_count)
            rank0_log(
                f"[batch-global-done] epoch={epoch} batch={batch_idx} "
                f"global_frames={global_frame_count} local_frames={frame_count}"
            )
            if global_frame_count == 0:
                continue
            micro_step += 1
            if micro_step % max(1, int(args.grad_accum)) == 0:
                complete_optimizer_step(epoch)
                if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
                    break
        if micro_step % max(1, int(args.grad_accum)) != 0 and not (int(args.max_steps) > 0 and global_step >= int(args.max_steps)):
            # epoch 末尾如果还有未满 grad_accum 的梯度，也要同步并更新一次；
            # 否则最后几个有效 batch 的 OPSD 信号会被静默丢掉。
            complete_optimizer_step(epoch)
            micro_step = 0
        if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
            break
    if rank == 0:
        _save_adapter(bundle, output_dir / "final", args)
    if tb is not None:
        tb.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
