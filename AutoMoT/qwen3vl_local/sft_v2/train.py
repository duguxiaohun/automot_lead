"""训练 SFT v2 LoRA：串行 ``SCENE -> STATUS/SUBGOAL`` 选择题监督。

训练目标很窄：冻结 Qwen3-VL-Instruct base，只训练 LoRA adapter，让模型学会在两段
对话中复制合法选择值。loss 只打在 ``SCENE``、``STATUS``、``SUBGOAL`` 的值 token
上，prompt、字段名、冒号、换行和图像 token 全部不参与监督。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except Exception:
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False

from qwen3vl_local.sft_v2.prompts import extract_gt, target_spans


def strip_image_placeholders(text: str) -> str:
    """移除 jsonl 中给人看的 ``<image>`` 占位符。

    训练时图片不是靠字符串占位传入，而是通过 processor 的 structured image message
    生成 vision token；因此这里要把文本里的占位符剥掉。
    """

    s = text.lstrip()
    while s.startswith("<image>"):
        s = s[len("<image>"):]
    return s.lstrip("\n")


class SerialChoiceDataset(Dataset):
    """加载 SFT v2 jsonl，并整理成训练需要的扁平字段。

    每条 row 必须包含 ``stage_messages.scene`` 和 ``stage_messages.status``。dataset
    只做轻量解析，不读图片；图片在训练 loop 里按需加载，避免 DataLoader 初始化阶段
    持有大量 PIL 对象。
    """

    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"jsonl not found: {self.path}")
        self.rows: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    parsed_gt = extract_gt(row["messages"][2]["content"])
                    choice_meta = row.get("choice_meta") or {}
                    # choice_meta 是新格式的权威标签来源；parsed_gt 只是兼容旧 row 的兜底。
                    gt = {
                        "scene": choice_meta.get("target_scene") or parsed_gt["scene"],
                        "status": choice_meta.get("target_status") or parsed_gt["status"],
                        "subgoal": choice_meta.get("target_subgoal") or parsed_gt["subgoal"],
                    }
                    stages = row.get("stage_messages")
                    if not stages or "scene" not in stages or "status" not in stages:
                        raise ValueError(f"SFT v2 row missing stage_messages.scene/status in {self.path}")
                    scene_msgs = stages["scene"]
                    status_msgs = stages["status"]
                    scene_assistant = scene_msgs[2]["content"]
                    self.rows.append({
                        "scenario": row.get("scenario", gt["scene"]),
                        "run_id": row.get("run_id", ""),
                        "anchor": int(row.get("anchor", -1)),
                        "images": list(row.get("images", [])),
                        "scene_system_prompt": scene_msgs[0]["content"],
                        "scene_user_prompt": strip_image_placeholders(scene_msgs[1]["content"]),
                        "scene_assistant": scene_assistant,
                        "status_system_prompt": status_msgs[0]["content"],
                        "status_user_prompt": strip_image_placeholders(status_msgs[1]["content"]),
                        "status_assistant": status_msgs[2]["content"],
                        "gt_scene": gt["scene"],
                        "gt_status": gt["status"],
                        "gt_subgoal": gt["subgoal"],
                        "is_transition_sample": bool(row.get("is_transition_sample", False)),
                        # wrong_scene_augmented 表示第二阶段 prompt/assistant 已经按所选
                        # 场景的同相位事件重写，因此仍然可以正常监督 STATUS/SUBGOAL。
                        "wrong_scene_augmented": bool(choice_meta.get("wrong_scene_augmented", False)),
                    })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def collate_passthrough(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """DataLoader collate：保持 list[dict] 原样返回。

    当前 batch 内每个样本的图像数量、token 长度和 vision tensor 形状都可能不同，直接
    padding 成大 batch 反而容易引入 mask 错位；这里逐样本 forward/backward。
    """
    return batch


@dataclass
class ModelBundle:
    """训练中一起传递的模型、processor、tokenizer 与设备信息。"""

    model: Any
    processor: Any
    tokenizer: Any
    device: torch.device

    def unwrap(self):
        """如果外层套了 DDP，则取回真实 PEFT/Qwen 模型对象。"""

        return getattr(self.model, "module", self.model)


def load_model_with_lora(
    model_dir: pathlib.Path,
    *,
    device: torch.device,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    gradient_checkpointing: bool,
) -> ModelBundle:
    """加载本地 Qwen3-VL-Instruct，并注入 PEFT LoRA。

    base model 全部冻结，只让 LoRA 参数更新。这里要求 fast tokenizer，是因为后续需要
    offset_mapping 把字符级 value span 映射到 token 级 loss mask。
    """

    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError:
        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass

    print(f"[load] base model from {model_dir}")
    model = ModelClass.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("SFT v2 needs a fast tokenizer for offset_mapping.")

    for p in model.parameters():
        # 先冻结 base，再通过 get_peft_model 注入可训练 LoRA 参数。
        p.requires_grad = False

    if gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return ModelBundle(model=model, processor=processor, tokenizer=tokenizer, device=device)


def load_images(paths: List[str]) -> List[Image.Image]:
    """读取一个样本的 RGB 历史帧。"""

    return [Image.open(p).convert("RGB") for p in paths]


def _overlap(off: Tuple[int, int], lo: int, hi: int) -> bool:
    """判断 tokenizer token 的字符区间是否与监督值区间重叠。"""

    return off[0] < hi and off[1] > lo


def _assistant_token_mask(bundle: ModelBundle, assistant_text: str) -> Tuple[List[int], List[bool]]:
    """对单个 assistant turn 生成 token 级 value mask。

    ``target_spans`` 返回字符区间；fast tokenizer 返回每个 token 对应的字符 offset。
    两者相交的 token 才会被标为 True，后续 loss 权重为 ``label_weight``。
    """

    enc = bundle.tokenizer(
        assistant_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    token_ids = list(enc["input_ids"])
    offsets = [tuple(x) for x in enc["offset_mapping"]]
    spans = list(target_spans(assistant_text).values())
    value_mask = []
    for off in offsets:
        off_i = (int(off[0]), int(off[1]))
        value_mask.append(any(_overlap(off_i, lo, hi) for lo, hi in spans))
    return token_ids, value_mask


def _find_subsequence(haystack: List[int], needle: List[int], start: int, *, last: bool = False) -> int:
    """在完整 input_ids 中查找某个 assistant target 的 token 子序列。

    第二阶段的 STATUS/SUBGOAL 可能和 user prompt 里的 PREVIOUS_STATUS_HINT 完全相同，
    所以 turn_idx=1 时会从后往前找，避免误把 prompt 侧文本当成监督目标。
    """

    if not needle:
        raise ValueError("empty assistant token sequence")
    end = len(haystack) - len(needle)
    indices = range(end, max(start, 0) - 1, -1) if last else range(max(start, 0), end + 1)
    for i in indices:
        if haystack[i:i + len(needle)] == needle:
            return i
    raise ValueError("assistant token sequence not found in expanded input_ids")


def _assert_inside_assistant_turn(
    expanded_ids: List[int],
    pos: int,
    asst_header_ids: List[int],
    turn_idx: int,
) -> None:
    """确认匹配到的监督子串确实位于 assistant turn 内。

    防止将来 prompt 文本里出现和 assistant target 完全一样的 token 子串时静默错配
    （比如示例里写 "SCENE: Accident"），把 prompt 段当成监督目标。
    """

    H = len(asst_header_ids)
    if pos < H or expanded_ids[pos - H:pos] != asst_header_ids:
        raise ValueError(
            f"assistant value subsequence (turn={turn_idx}, pos={pos}) is not "
            "preceded by <|im_start|>assistant\\n header; likely matched a "
            "prompt-side substring. Check prompt text or tokenizer behavior."
        )


def build_student_inputs(
    bundle: ModelBundle,
    sample: Dict[str, Any],
    images: List[Image.Image],
    *,
    max_length: int,
    label_weight: float,
) -> Optional[Dict[str, Any]]:
    """构造单样本 multi-turn 输入和 token 级 loss 权重。

    训练样本被渲染为一条完整对话：

    ``system -> user(images+scene prompt) -> assistant(scene) -> user(status prompt) -> assistant(status/subgoal)``

    其中只有两个 assistant turn 里的值 token 有 loss。wrong-scene 增强样本的第二阶段
    prompt 和 target 已经在 build_dataset.py 中映射到 selected scene，因此这里统一打
    STATUS/SUBGOAL loss。
    """

    messages = [
        {"role": "system", "content": sample["scene_system_prompt"]},
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": sample["scene_user_prompt"]}]
            ),
        },
        {"role": "assistant", "content": sample["scene_assistant"]},
        {"role": "user", "content": sample["status_user_prompt"]},
        {"role": "assistant", "content": sample["status_assistant"]},
    ]
    chat_text = bundle.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_inputs = bundle.processor(
        text=[chat_text],
        images=images if images else None,
        return_tensors="pt",
        padding=True,
    )
    input_ids = prompt_inputs["input_ids"][0]
    if int(input_ids.shape[0]) > max_length:
        # 超长样本直接跳过；DDP 下 run_batch 会同步所有 rank 的跳过决策。
        return None

    labels = input_ids.clone()
    # 权重初始全 0，只有 assistant 的值 token 会被置为 label_weight。
    weights = torch.zeros_like(input_ids, dtype=torch.float32)
    expanded_ids = [int(x) for x in input_ids.tolist()]
    asst_header_ids = list(bundle.tokenizer(
        "<|im_start|>assistant\n",
        add_special_tokens=False,
    )["input_ids"])
    cursor = 0
    for turn_idx, assistant_text in enumerate((sample["scene_assistant"], sample["status_assistant"])):
        assistant_ids, value_mask = _assistant_token_mask(bundle, assistant_text)
        # keep 样本里，第二阶段 target 可能和 user prompt 的 PREVIOUS_STATUS_HINT 完全一致。
        # 因此第二个 assistant turn 从渲染后序列末尾反向匹配，避免误选 prompt 侧文本。
        pos = _find_subsequence(expanded_ids, assistant_ids, cursor, last=(turn_idx == 1))
        _assert_inside_assistant_turn(expanded_ids, pos, asst_header_ids, turn_idx)
        for j, is_value in enumerate(value_mask):
            if is_value:
                # 权重写在原始 token 位置；真正算 CE 时会整体右移一位对齐 next-token label。
                weights[pos + j] = label_weight
        cursor = pos + len(assistant_ids)

    extra = {k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask")}
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "loss_weights": weights,
        "vision": extra,
        "chat_text": chat_text,
    }


def loss_parts_one_sample(bundle: ModelBundle, packed: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """单样本前向，返回加权 loss 的分子和分母。

    返回分子/分母而不是直接平均，是为了将来扩展 batch 汇总时仍能按 token 权重精确
    归一化。
    """

    kwargs: Dict[str, Any] = {
        "input_ids": packed["input_ids"].unsqueeze(0).to(bundle.device),
        "attention_mask": packed["attention_mask"].unsqueeze(0).to(bundle.device),
    }
    labels = packed["labels"].unsqueeze(0).to(bundle.device)
    weights = packed["loss_weights"].unsqueeze(0).to(bundle.device)
    for k, v in packed["vision"].items():
        kwargs[k] = v.to(bundle.device) if isinstance(v, torch.Tensor) else v

    out = bundle.model(**kwargs, use_cache=False, return_dict=True)
    logits = out.logits
    # causal LM 标准右移：第 t 个 logits 预测第 t+1 个 token。
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].contiguous()
    active = shift_labels.ne(-100) & shift_weights.gt(0)
    if not bool(active.any()):
        # 极端情况下如果没有任何监督 token，返回可反传的 0，避免图断掉。
        zero = shift_logits.sum() * 0.0
        return zero, zero.detach()
    per_tok = F.cross_entropy(shift_logits[active], shift_labels[active], reduction="none")
    active_weights = shift_weights[active]
    return (per_tok * active_weights).sum(), active_weights.sum()


def loss_one_sample(bundle: ModelBundle, packed: Dict[str, Any]) -> torch.Tensor:
    """计算单样本标量 loss。"""

    num, den = loss_parts_one_sample(bundle, packed)
    return num / den.clamp_min(1e-6)


@dataclass
class StepStats:
    """一个 micro-batch 的训练统计。"""

    loss_sum: float = 0.0
    n_samples: int = 0
    n_skipped: int = 0


def _ddp_all_ranks_valid(local_valid: bool, device: torch.device) -> bool:
    """DDP 下同步“当前样本是否可用”的布尔结果。

    如果某个 rank 图片缺失或样本超长，所有 rank 都跳过本轮，避免 collective 操作数量
    不一致导致训练挂起。
    """

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return local_valid
    flag = torch.tensor([1.0 if local_valid else 0.0], device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item() > 0.5)


def run_batch(
    bundle: ModelBundle,
    batch: List[Dict[str, Any]],
    *,
    max_length: int,
    label_weight: float,
    loss_scale: float,
    sync_grads: bool,
) -> StepStats:
    """逐样本跑一个 micro-batch，并累积梯度。

    由于每个样本的图文 token 长度不同，这里不做张量级 batch padding，而是在同一个
    DataLoader batch 内顺序 forward/backward。``sync_grads=False`` 时使用 DDP no_sync
    实现梯度累积。
    """

    from contextlib import nullcontext

    stats = StepStats()
    sync_ctx = bundle.model.no_sync() if (not sync_grads and hasattr(bundle.model, "no_sync")) else nullcontext()
    with sync_ctx:
        for sample in batch:
            try:
                images = load_images(sample["images"])
                local_ok = True
            except (FileNotFoundError, OSError) as exc:
                print(f"[warn] image load failed {sample.get('run_id')} anchor={sample.get('anchor')}: {exc}")
                images = []
                local_ok = False
            if not _ddp_all_ranks_valid(local_ok, bundle.device):
                # 任意 rank 图片读取失败时，全体跳过该样本。
                stats.n_skipped += 1
                continue
            packed = build_student_inputs(
                bundle,
                sample,
                images,
                max_length=max_length,
                label_weight=label_weight,
            )
            if not _ddp_all_ranks_valid(packed is not None, bundle.device):
                # 任意 rank 因 max_length 等原因无法构造输入时，全体跳过。
                stats.n_skipped += 1
                continue
            bundle.model.train()
            loss = loss_one_sample(bundle, packed)
            (loss / max(loss_scale, 1.0)).backward()
            stats.loss_sum += float(loss.detach().item())
            stats.n_samples += 1
    return stats


@torch.no_grad()
def evaluate_loss(
    bundle: ModelBundle,
    loader: DataLoader,
    *,
    max_length: int,
    label_weight: float,
    max_samples: int,
) -> Dict[str, float]:
    """计算强制喂入真实答案的验证 loss，支持 DDP 汇总。"""

    bundle.model.eval()
    losses: List[float] = []
    skipped = 0
    for batch in loader:
        for sample in batch:
            if max_samples > 0 and len(losses) >= max_samples:
                break
            try:
                images = load_images(sample["images"])
                packed = build_student_inputs(
                    bundle,
                    sample,
                    images,
                    max_length=max_length,
                    label_weight=label_weight,
                )
                if packed is None:
                    skipped += 1
                    continue
                losses.append(float(loss_one_sample(bundle, packed).item()))
            except (FileNotFoundError, OSError):
                skipped += 1
        if max_samples > 0 and len(losses) >= max_samples:
            break
    local_sum = float(sum(losses))
    local_n = float(len(losses))
    local_skipped = float(skipped)
    # DDP 下每个 rank 只看到自己切片的 val 样本；不汇总会让 rank0 打印的 val loss 仅来自
    # rank0 的子集。这里把 (sum, n, skipped) 全部 all_reduce SUM，再算全局均值。
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t = torch.tensor([local_sum, local_n, local_skipped], device=bundle.device, dtype=torch.float64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        local_sum, local_n, local_skipped = float(t[0].item()), float(t[1].item()), float(t[2].item())
    return {
        "loss": local_sum / max(local_n, 1.0),
        "samples": local_n,
        "skipped": local_skipped,
    }


def setup_distributed() -> Tuple[int, int, int]:
    """初始化 torch.distributed，并返回 rank/world/local_rank。"""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """清理 DDP process group。"""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_rank0(rank: int) -> bool:
    """判断是否为主进程。"""

    return rank == 0


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """创建 warmup + cosine decay 学习率调度器。"""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args() -> argparse.Namespace:
    """解析训练命令行参数。"""

    p = argparse.ArgumentParser(description="Train SFT v2 serial-choice LoRA")
    p.add_argument("--train-jsonl", type=str, required=True)
    p.add_argument("--val-jsonl", type=str, default=None)
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--label-weight", type=float, default=1.0)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=10000)
    p.add_argument("--eval-steps", type=int, default=10000)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--check", action="store_true")
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--seed", type=int, default=20260617)
    return p.parse_args()


def main() -> None:
    """训练入口：加载数据、模型、LoRA，执行训练/验证/保存。"""

    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    output_dir = pathlib.Path(args.output_dir)
    if is_rank0(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[init] output_dir={output_dir} rank={rank} world={world_size} device={device}")

    bundle = load_model_with_lora(
        pathlib.Path(args.model_dir),
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=not args.no_grad_checkpoint,
    )
    if world_size > 1:
        bundle.model = torch.nn.parallel.DistributedDataParallel(
            bundle.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    train_ds = SerialChoiceDataset(pathlib.Path(args.train_jsonl))
    val_ds = SerialChoiceDataset(pathlib.Path(args.val_jsonl)) if args.val_jsonl else None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        # DistributedSampler 会 padding 到 world_size 整除；当前训练能接受这点。
        # val loss 后面会 all_reduce 汇总各 rank 看到的样本。
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if val_ds else None
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_passthrough,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.per_device_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_passthrough,
    ) if val_ds else None

    if args.check:
        args.max_steps = 2 if args.max_steps == 0 else min(args.max_steps, 2)
        if is_rank0(rank):
            print("[check] max_steps=2, no final save")

    # 只把 LoRA 参数交给优化器；base Qwen 参数在 load_model_with_lora 里已经冻结。
    params = [p for p in bundle.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    # len(train_loader) 已经是当前 rank 看到的 micro-batch 数；再除以 grad_accum 得到优化步数。
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(args.grad_accum, 1)))
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.num_epochs
    scheduler = make_scheduler(optimizer, total_steps, int(total_steps * args.warmup_ratio))

    tb = SummaryWriter(log_dir=str(output_dir / "tb")) if (is_rank0(rank) and _TB_AVAILABLE) else None
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    saved: List[pathlib.Path] = []
    stop = False

    def finish_step(epoch: int, loss_sum: float, n_samples: int, reason: str) -> None:
        """完成一次 optimizer step，并按需 eval/save。

        tail step 在 DDP no_sync 累积未满时需要手动 all_reduce 梯度，保证最后一小段
        样本不会只更新各自 rank 的局部梯度。
        """

        nonlocal global_step, saved
        if n_samples <= 0:
            optimizer.zero_grad(set_to_none=True)
            return
        trainable = [p for p in bundle.unwrap().parameters() if p.requires_grad]
        if reason == "tail" and world_size > 1 and torch.distributed.is_initialized():
            for p in trainable:
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                    p.grad.div_(float(world_size))
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        loss = loss_sum / max(n_samples, 1)
        if is_rank0(rank) and (global_step == 1 or global_step % args.logging_steps == 0 or reason == "tail"):
            lr = scheduler.get_last_lr()[0]
            print(f"[train] epoch={epoch} step={global_step}/{total_steps} loss={loss:.4f} lr={lr:.2e} samples={n_samples} elapsed={(time.time()-t0)/60:.1f}m")
            if tb:
                tb.add_scalar("train/loss", loss, global_step)
                tb.add_scalar("train/lr", lr, global_step)
        if (not args.check) and val_loader is not None and args.eval_steps > 0 and global_step % args.eval_steps == 0:
            metrics = evaluate_loss(
                bundle,
                val_loader,
                max_length=args.max_length,
                label_weight=args.label_weight,
                max_samples=args.max_eval_samples,
            )
            if is_rank0(rank):
                print(f"[eval@{global_step}] {metrics}")
                if tb:
                    for k, v in metrics.items():
                        tb.add_scalar(f"val/{k}", v, global_step)
        if (not args.check) and args.save_steps > 0 and global_step % args.save_steps == 0 and is_rank0(rank):
            ckpt = output_dir / f"checkpoint-{global_step}"
            bundle.unwrap().save_pretrained(str(ckpt))
            saved.append(ckpt)
            if args.save_total_limit > 0 and len(saved) > args.save_total_limit:
                old = saved.pop(0)
                import shutil
                shutil.rmtree(old, ignore_errors=True)
            print(f"[save] {ckpt}")

    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            # 每个 epoch 更新 sampler seed，保证 DDP shuffle 可复现但不固定同一顺序。
            train_sampler.set_epoch(epoch)
        accum_loss = 0.0
        accum_samples = 0
        accum_count = 0
        for batch in train_loader:
            # 只有梯度累积的最后一个 micro-batch 才触发 DDP 梯度同步；前面的 micro-batch 用 no_sync
            # 减少通信开销。单卡时 sync_grads 恒为 True。
            is_last_micro = accum_count + 1 >= args.grad_accum
            stats = run_batch(
                bundle,
                batch,
                max_length=args.max_length,
                label_weight=args.label_weight,
                loss_scale=float(max(args.grad_accum, 1) * max(args.per_device_batch_size, 1)),
                sync_grads=(world_size <= 1) or is_last_micro,
            )
            if stats.n_samples <= 0:
                continue
            accum_loss += stats.loss_sum
            accum_samples += stats.n_samples
            accum_count += 1
            if accum_count >= args.grad_accum:
                # 满一个梯度累积窗口后执行一次 optimizer step。
                finish_step(epoch, accum_loss, accum_samples, "grad_accum")
                accum_loss = 0.0
                accum_samples = 0
                accum_count = 0
                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop = True
                    break
        if accum_count > 0 and not stop:
            # epoch 末尾不足 grad_accum 的残余 micro-batch 也要更新一次；DDP 下 finish_step 会手动
            # all_reduce 这段 tail 梯度，保证所有 rank 参数仍然一致。
            finish_step(epoch, accum_loss, accum_samples, "tail")
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
        if stop:
            break

    if is_rank0(rank) and not args.check:
        # 训练完成后只保存 adapter/processor 到 final；base 模型仍然从本地 MODEL_DIR 读取。
        final_dir = output_dir / "final"
        bundle.unwrap().save_pretrained(str(final_dir))
        try:
            bundle.processor.save_pretrained(str(final_dir))
        except Exception as exc:
            print(f"[warn] save processor skipped: {exc}")
        print(f"[done] final adapter -> {final_dir}")
    if tb:
        tb.flush()
        tb.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
