"""SFT base 自由生成评估入口。

评估按真实串行协议执行：Q1 直接生成 RS，随后总是沿 Q1 KV 继续问 Q2 EVENT。
Q2 候选按学生预测/维护的 RS 生成，不再用 GT RS gate 美化结果。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TextIO

from PIL import Image

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _maybe_apply_gpu_ids() -> None:
    """在 torch 初始化前应用 GPU_IDS pin 卡约定。

    训练 launcher 会处理 GPU_IDS；eval.py 是直接 `python` 调用，所以必须在 import
    torch 前把 GPU_IDS 翻译成 CUDA_VISIBLE_DEVICES，否则文档里的 `GPU_IDS=0 python ...`
    会被忽略。
    """

    pinned = ",".join(part.strip() for part in os.environ.get("GPU_IDS", "").split(",") if part.strip())
    if pinned:
        os.environ["CUDA_VISIBLE_DEVICES"] = pinned
        print(f"[gpu] using GPU_IDS={pinned}")


_maybe_apply_gpu_ids()

import torch
import torch.distributed as dist

from qwen3vl_local.sft_base import DATASET_VERSION  # noqa: E402
from qwen3vl_local.sft_base.eval_candidates import q2_candidates_for_student_rs  # noqa: E402
from qwen3vl_local.sft_base.labels import (  # noqa: E402
    EVENT_LABELS,
    EVENT_LABEL_TO_TOKEN,
    REGULAR_MAJORITY_EVENT_BY_RS,
    REGULAR_EVENT_LABELS,
    REGULAR_ZERO_INFO_BASELINE_BY_RS,
    REGULAR_ZERO_INFO_BASELINE_END_TO_END,
    RS_LABELS,
    event_in_candidates,
    is_regular_event,
    is_unusual,
)
from qwen3vl_local.sft_base.prompts import (  # noqa: E402
    Memory,
    build_q1_prompt,
    build_q2_prompt,
    parse_q1_output,
    parse_q2_output,
    refresh_memory_goal,
    reset_memory_for_frame,
    update_memory_after_q1,
    update_memory_after_q2,
)
from qwen3vl_local.sft_base.train import (  # noqa: E402
    RouteSequenceDataset,
    _event_target_from_frame,
    _load_images,
    _messages,
    _rs_target_from_frame,
)
from qwen3vl_local.sft_v3.train import _append_user_turn, _kv_start_state, _student_generate_kv  # noqa: E402


_VISION_SCOPE_CHOICES = {"off", "merger", "last4", "all"}
_EVAL_TASK_TO_MODE = {
    "full": "full_route",
    "full_route": "full_route",
    "rs": "rs_transition",
    "rs_transition": "rs_transition",
    "event": "event_transition",
    "event_transition": "event_transition",
}
_EVENT_LABELS = EVENT_LABELS
_RS_CM_LABELS = (*RS_LABELS, "INVALID")
_EVENT_CM_LABELS = (*_EVENT_LABELS, "INVALID", "UNREACHABLE")
_REGULAR_CM_LABELS = (*REGULAR_EVENT_LABELS, "UE", "INVALID", "UNREACHABLE")
_GLOBAL_EVENT_MAJORITY_LABEL = "R-E1"
_GLOBAL_EVENT_MAJORITY_TOKEN = EVENT_LABEL_TO_TOKEN[_GLOBAL_EVENT_MAJORITY_LABEL]
_RS_TRANSITION_LABELS = tuple(f"{src}->{dst}" for src in RS_LABELS for dst in RS_LABELS if src != dst) + ("NO_CHANGE", "INVALID")
_EVENT_TRANSITION_LABELS = tuple(f"{src}->{dst}" for src in _EVENT_LABELS for dst in _EVENT_LABELS if src != dst) + ("NO_CHANGE", "INVALID")


def setup_distributed() -> tuple[int, int, int]:
    """初始化可选 torchrun 多卡评测环境。

    单进程评测时没有 `WORLD_SIZE`，函数只返回 `(0, 0, 1)`，保持旧行为。
    多卡评测时由 torchrun 注入 `RANK/WORLD_SIZE/LOCAL_RANK`；每个进程只加载
    一份模型到自己的 `cuda:LOCAL_RANK`，后续按 case index 分片评估。
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("sft_base multi-GPU eval requires CUDA.")
        visible = [
            part.strip()
            for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if part.strip()
        ]
        if visible and world_size > len(visible):
            raise ValueError(
                f"WORLD_SIZE={world_size} but CUDA_VISIBLE_DEVICES only exposes "
                f"{len(visible)} GPU(s): {os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """关闭 torch.distributed，避免 torchrun 退出时残留通信组。"""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    """rank0 负责打印汇总、写最终 metrics/jsonl。"""

    return rank == 0


@dataclass
class EvalBundle:
    """评估时传递给 v3 KV helper 的轻量 bundle。"""

    model: Any
    processor: Any
    tokenizer: Any
    device: torch.device

    def unwrap(self) -> Any:
        return self.model


@dataclass
class EvalCase:
    """一次自由生成评估片段。

    full_route 模式下 frames 是完整 route；RS/EVENT 专项模式下 frames 是
    对应转折点前后窗口。评估只在片段第一帧初始化 memory，后续不做脚本纠偏。
    """

    case_id: str
    route: Any
    route_index: int
    frames: List[Any]
    start_index: int
    transition_index: Optional[int] = None
    transition_kind: Optional[str] = None
    source_rs: Optional[str] = None
    target_rs: Optional[str] = None
    source_event: Optional[str] = None
    target_event: Optional[str] = None
    source_abnormal: Optional[bool] = None
    target_abnormal: Optional[bool] = None


def _resolve_model_path(path: pathlib.Path) -> pathlib.Path:
    """按 AutoMoT 运行目录口径规范化模型路径。"""

    path = pathlib.Path(path)
    if not path.is_absolute():
        path = _AUTOMOT_ROOT / path
    return path.resolve()


def _validate_adapter_config(adapter_dir: pathlib.Path, model_dir: pathlib.Path) -> Dict[str, Any]:
    """读取并校验 sft_base adapter 自描述配置。

    PEFT 只看权重形状，传错 v2/v5/base adapter 时不一定会报错，但指标会没有意义。
    因此 eval 入口先强制校验 route、dataset version、base model path 和 vision scope，
    让 adapter 语义不匹配尽早失败。
    """

    cfg_path = pathlib.Path(adapter_dir) / "sft_base_adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"sft_base adapter config not found: {cfg_path}. "
            "Refusing to evaluate without route/dataset/base-model validation."
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("route") != "sft_base_token_choice":
        raise ValueError(
            "adapter route mismatch: current eval is token-only and no longer supports "
            f"ABC/direct-choice adapters. expected sft_base_token_choice, got {cfg.get('route')!r}. "
            "Please evaluate a token-choice adapter or retrain with the current sft_base code."
        )
    if cfg.get("dataset_version") != DATASET_VERSION:
        raise ValueError(f"adapter dataset_version mismatch: expected {DATASET_VERSION}, got {cfg.get('dataset_version')!r}")
    saved_model_dir = cfg.get("base_model_dir")
    if not saved_model_dir:
        raise ValueError("adapter config missing base_model_dir")
    if _resolve_model_path(pathlib.Path(saved_model_dir)) != _resolve_model_path(model_dir):
        raise ValueError(
            "adapter base_model_dir mismatch: "
            f"adapter={_resolve_model_path(pathlib.Path(saved_model_dir))} "
            f"eval={_resolve_model_path(model_dir)}"
        )
    scope = str(cfg.get("lora_vision_scope", ""))
    if scope not in _VISION_SCOPE_CHOICES:
        raise ValueError(f"adapter lora_vision_scope invalid: {scope!r}")
    lora_vision = bool(cfg.get("lora_vision", False))
    if (scope == "off") != (not lora_vision):
        raise ValueError(f"adapter vision metadata inconsistent: lora_vision={lora_vision}, scope={scope!r}")
    return cfg


def load_eval_bundle(model_dir: pathlib.Path, adapter_dir: Optional[pathlib.Path], device: torch.device, *, merge_lora: bool) -> EvalBundle:
    """加载 base Qwen 和可选 LoRA adapter。"""

    from transformers import AutoProcessor

    # 先校验 adapter 再加载大模型。这样用户误传旧 ABC/direct-choice checkpoint 时，
    # 多卡评测不会先把 4 份 Qwen 都加载到显存里才报错，失败会更快也更清楚。
    adapter_cfg: Optional[Dict[str, Any]] = None
    if adapter_dir is not None:
        adapter_cfg = _validate_adapter_config(adapter_dir, model_dir)

    try:
        from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError:
        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass

    kwargs = {"local_files_only": True, "trust_remote_code": True}
    try:
        model = ModelClass.from_pretrained(str(model_dir), dtype=torch.bfloat16, **kwargs)
    except TypeError:
        model = ModelClass.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16, **kwargs)
    if adapter_dir is not None:
        from peft import PeftModel

        cfg = adapter_cfg or _validate_adapter_config(adapter_dir, model_dir)
        print(f"[adapter] validated sft_base adapter scope={cfg.get('lora_vision_scope')}")
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        if merge_lora and hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    return EvalBundle(model=model, processor=processor, tokenizer=processor.tokenizer, device=device)


def _q1_messages(images: List[Any], prompt: str) -> List[Dict[str, Any]]:
    """复用 train._messages 的 system/image/user 结构，但不带 assistant target。"""

    return _messages(images, prompt, "", None, None)[:2]


def _generate_start(bundle: EvalBundle, images: List[Any], prompt: str, max_new_tokens: int) -> tuple[str, Any]:
    """Q1 fresh prefill + decode。"""

    with torch.inference_mode():
        state = _kv_start_state(bundle, _q1_messages(images, prompt))
        text, after, _ = _student_generate_kv(bundle, state, max_new_tokens)
    return text, after


def _generate_next(bundle: EvalBundle, previous_state: Any, prompt: str, max_new_tokens: int) -> tuple[str, Any]:
    """在 Q1 assistant 输出后的 KV 上追加 Q2 user turn。

    这里模拟真实推理：Q2 复用已经吃过图像和 Q1 对话的 KV cache，只追加一轮文本
    user prompt。训练里的多轮 chat 展开与这条路径保持同样的消息顺序。
    """

    with torch.inference_mode():
        state = _append_user_turn(bundle, previous_state, prompt)
        text, after, _ = _student_generate_kv(bundle, state, max_new_tokens)
    return text, after


def _route_name(route: Any) -> str:
    """生成稳定 route 名称，优先使用数据里的 route_id。"""

    return str(getattr(route, "route_id", "") or "route")


def _select_full_route_cases(ds: RouteSequenceDataset, args: argparse.Namespace) -> List[EvalCase]:
    """选择完整路径评估 case。

    sample_routes>0 时按 seed 从整份 index 随机抽 route；否则沿用 max_routes 的
    顺序截断语义，兼容老命令。
    """

    items = list(enumerate(ds.rows))
    if int(args.sample_routes) > 0:
        rng = random.Random(int(args.seed))
        k = min(int(args.sample_routes), len(items))
        items = rng.sample(items, k)
    elif int(args.max_routes) > 0:
        items = items[: int(args.max_routes)]
    cases: List[EvalCase] = []
    for route_index, route in items:
        cases.append(
            EvalCase(
                case_id=f"route:{route_index}:{_route_name(route)}",
                route=route,
                route_index=route_index,
                frames=list(route.frames),
                start_index=0,
            )
        )
    return cases


def _transition_kinds(prev_frame: Any, frame: Any, wanted: str) -> List[str]:
    """判断相邻两帧是否是 RS/EVENT 转折点。"""

    kinds: List[str] = []
    if wanted in {"rs", "both"} and prev_frame.rs_label != frame.rs_label:
        kinds.append("rs")
    event_changed = prev_frame.event_label != frame.event_label or bool(prev_frame.abnormal) != bool(frame.abnormal)
    if wanted in {"event", "both"} and event_changed:
        kinds.append("event")
    return kinds


def _select_transition_cases(ds: RouteSequenceDataset, args: argparse.Namespace) -> List[EvalCase]:
    """选择 RS/EVENT 转折点前后窗口评估 case。

    片段从转折点前 window 帧开始，memory 只在片段第一帧按该帧真值初始化。
    如果模型在转折后没有自己更新 RS/EVENT，后续帧会继续暴露这个错误。
    """

    window = max(1, int(args.transition_window))
    cases: List[EvalCase] = []
    items = list(enumerate(ds.rows))
    if int(args.max_routes) > 0:
        items = items[: int(args.max_routes)]
    wanted = "rs" if args.eval_mode == "rs_transition" else "event"
    for route_index, route in items:
        frames = list(route.frames)
        for idx in range(1, len(frames)):
            kinds = _transition_kinds(frames[idx - 1], frames[idx], wanted)
            if not kinds:
                continue
            start = max(0, idx - window)
            end = min(len(frames), idx + window + 1)
            kind = "+".join(kinds)
            cases.append(
                EvalCase(
                    case_id=f"transition:{kind}:{route_index}:{_route_name(route)}:{idx}",
                    route=route,
                    route_index=route_index,
                    frames=frames[start:end],
                    start_index=start,
                    transition_index=idx,
                    transition_kind=kind,
                    source_rs=frames[idx - 1].rs_label,
                    target_rs=frames[idx].rs_label,
                    source_event=frames[idx - 1].event_label,
                    target_event=frames[idx].event_label,
                    source_abnormal=bool(frames[idx - 1].abnormal),
                    target_abnormal=bool(frames[idx].abnormal),
                )
            )
    if int(args.max_transition_cases) > 0 and len(cases) > int(args.max_transition_cases):
        rng = random.Random(int(args.seed))
        cases = rng.sample(cases, int(args.max_transition_cases))
    return cases


def _select_eval_cases(ds: RouteSequenceDataset, args: argparse.Namespace) -> List[EvalCase]:
    """按 eval_mode 生成评估 case 列表。"""

    if args.eval_mode in {"rs_transition", "event_transition"}:
        return _select_transition_cases(ds, args)
    return _select_full_route_cases(ds, args)


def _stable_case_seed(args: argparse.Namespace, case: EvalCase) -> int:
    """生成稳定 case 随机种子，避免 Python hash 随进程漂移。"""

    base = int(args.seed) + case.route_index * 100003 + case.start_index * 9176
    if case.transition_index is not None:
        base += case.transition_index * 137
    return base


def _pick_different(rng: random.Random, current: str, candidates: List[str]) -> str:
    """从候选中选一个不同于 current 的值。"""

    pool = [item for item in candidates if item and item != current]
    if not pool:
        return current
    return rng.choice(sorted(set(pool)))


def _apply_initial_memory_noise(memory: Memory, frame: Any, args: argparse.Namespace, case: EvalCase) -> Memory:
    """按配置只在评估片段第一帧注入 memory 噪声。"""

    mode = str(args.initial_memory_noise)
    if mode in {"none", "unknown"}:
        return memory
    rng = random.Random(_stable_case_seed(args, case))
    mem = memory.copy()
    if mode in {"rs", "both", "random"} and (mode != "random" or rng.random() < 0.5):
        mem.rs_label = _pick_different(rng, mem.rs_label, list(RS_LABELS))
    if mode in {"event", "both", "random"}:
        event_candidates = [str(v) for v in frame.event_candidates]
        mem.event_label = _pick_different(rng, mem.event_label, event_candidates)
    return mem


def _eval_goal_xy(frame: Any, args: argparse.Namespace) -> Optional[List[float]]:
    """按 eval 消融设置返回学生可见的导航目标。"""

    if bool(getattr(args, "ablate_goal", False)):
        return None
    return frame.ego_to_goal_xy


def _initial_memory_for_frame(frame: Any, args: argparse.Namespace, case: EvalCase) -> Memory:
    """创建评估片段首帧 memory，默认 UNKNOWN 冷启动。"""

    goal_xy = _eval_goal_xy(frame, args)
    if str(args.initial_memory_noise) == "unknown":
        return refresh_memory_goal(Memory(rs_label="UNKNOWN", event_label="UNKNOWN"), goal_xy)
    rs_target = _rs_target_from_frame(frame)
    memory = reset_memory_for_frame(rs_target, ego_to_goal_xy=goal_xy)
    return _apply_initial_memory_noise(memory, frame, args, case)


def _image_ablation_seed(args: argparse.Namespace, case: EvalCase, frame: Any, image_idx: int) -> int:
    """为随机图 ablation 生成稳定种子。"""

    return int(args.seed) + case.route_index * 100003 + int(frame.frame_id) * 9176 + image_idx * 137


def _apply_image_ablation(images: List[Any], args: argparse.Namespace, case: EvalCase, frame: Any) -> List[Any]:
    """把输入 RGB history 替换成黑图或随机噪声图。

    这个诊断只改变图像，不改变 prompt、memory、候选和输出解析。如果黑图/随机图下
    RS/EVENT 指标几乎不变，说明模型主要在走语言 memory 捷径；如果明显变差或变乱，
    才能说明视觉通路至少参与了当前决策。
    """

    mode = str(getattr(args, "image_ablation", "none"))
    if mode == "none":
        return images
    out: List[Any] = []
    for idx, image in enumerate(images):
        size = getattr(image, "size", None) or (1152, 384)
        if mode == "black":
            out.append(Image.new("RGB", size, (0, 0, 0)))
            continue
        if mode == "random":
            width, height = int(size[0]), int(size[1])
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_image_ablation_seed(args, case, frame, idx))
            payload = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8, generator=generator).numpy().tobytes()
            out.append(Image.frombytes("RGB", (width, height), payload))
            continue
        raise ValueError(f"unknown image_ablation mode: {mode}")
    return out


def _transition_window_bounds(case: EvalCase, args: argparse.Namespace) -> tuple[int, int]:
    """返回转折命中容忍窗口的绝对帧下标范围。"""

    assert case.transition_index is not None
    tol = max(0, int(args.transition_tolerance))
    return case.transition_index - tol, case.transition_index + tol


def _first_hit(records: List[Dict[str, Any]], key: str, target: Optional[str], lo: int, hi: int) -> Optional[int]:
    """查找容忍窗口内首次预测到目标标签的帧下标。"""

    if target is None:
        return None
    for rec in records:
        idx = int(rec["frame_index"])
        if lo <= idx <= hi and rec.get(key) == target:
            return idx
    return None


def _first_bool_hit(records: List[Dict[str, Any]], key: str, target: Optional[bool], lo: int, hi: int) -> Optional[int]:
    """查找容忍窗口内首次预测到目标布尔状态的帧下标。"""

    if target is None:
        return None
    for rec in records:
        idx = int(rec["frame_index"])
        if lo <= idx <= hi and rec.get(key) == target:
            return idx
    return None


def _record_at_or_after(records: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
    """返回 index 处或之后的第一条记录，用于判断窗口左边界是否已在目标值。"""

    best: Optional[Dict[str, Any]] = None
    for rec in records:
        frame_index = int(rec["frame_index"])
        if frame_index < index:
            continue
        if best is None or frame_index < int(best["frame_index"]):
            best = rec
    return best


def _classify_hit_timing(hit_index: int, transition_index: int) -> str:
    """把命中帧分类为提前、准点或滞后。"""

    if hit_index < transition_index:
        return "early"
    if hit_index > transition_index:
        return "late"
    return "on_time"


def _event_is_ue(label: Optional[str]) -> Optional[bool]:
    """把 EVENT label 折叠成 UE/RE；非法或不可达返回 None。"""

    if label not in _EVENT_LABELS:
        return None
    return is_unusual(str(label))


def _update_change_confusion(counters: Dict[str, int], prefix: str, *, gt_changed: bool, pred_changed: Optional[bool]) -> None:
    """更新相邻帧变化检测的 TP/FP/FN/TN/invalid。"""

    if pred_changed is None:
        counters[f"{prefix}_invalid"] += 1
        if gt_changed:
            counters[f"{prefix}_fn"] += 1
        return
    if gt_changed and pred_changed:
        counters[f"{prefix}_tp"] += 1
    elif (not gt_changed) and pred_changed:
        counters[f"{prefix}_fp"] += 1
    elif gt_changed and not pred_changed:
        counters[f"{prefix}_fn"] += 1
    else:
        counters[f"{prefix}_tn"] += 1


def _transition_direction(prev_label: Optional[str], label: Optional[str], valid_labels: tuple[str, ...]) -> str:
    """把相邻两帧标签压成 source->target / NO_CHANGE / INVALID。"""

    if prev_label not in valid_labels or label not in valid_labels:
        return "INVALID"
    if prev_label == label:
        return "NO_CHANGE"
    return f"{prev_label}->{label}"


def _score_adjacent_changes(counters: Dict[str, int], records: List[Dict[str, Any]]) -> None:
    """在完整相邻帧上统计变化检测 precision/recall/F1 所需的混淆计数。"""

    for prev, cur in zip(records, records[1:]):
        gt_rs_dir = _transition_direction(prev.get("gt_rs"), cur.get("gt_rs"), RS_LABELS)
        pred_rs_dir = _transition_direction(prev.get("pred_rs"), cur.get("pred_rs"), RS_LABELS)
        counters[f"rs_transition_dir_cm_{gt_rs_dir}_{pred_rs_dir}"] += 1
        _update_change_confusion(
            counters,
            "rs_change",
            gt_changed=gt_rs_dir != "NO_CHANGE",
            pred_changed=None if pred_rs_dir == "INVALID" else pred_rs_dir != "NO_CHANGE",
        )

        if not bool(prev.get("event_score_valid")) or not bool(cur.get("event_score_valid")):
            continue
        gt_event_dir = _transition_direction(prev.get("gt_event"), cur.get("gt_event"), _EVENT_LABELS)
        pred_event_dir = _transition_direction(prev.get("pred_event"), cur.get("pred_event"), _EVENT_LABELS)
        counters[f"event_transition_dir_cm_{gt_event_dir}_{pred_event_dir}"] += 1
        prev_gt_ue = bool(prev.get("gt_abnormal"))
        cur_gt_ue = bool(cur.get("gt_abnormal"))
        prev_pred_ue = _event_is_ue(prev.get("pred_event"))
        cur_pred_ue = _event_is_ue(cur.get("pred_event"))
        pred_re_to_ue = None if prev_pred_ue is None or cur_pred_ue is None else ((not prev_pred_ue) and cur_pred_ue)
        pred_ue_to_re = None if prev_pred_ue is None or cur_pred_ue is None else (prev_pred_ue and (not cur_pred_ue))
        _update_change_confusion(
            counters,
            "re_to_ue",
            gt_changed=(not prev_gt_ue) and cur_gt_ue,
            pred_changed=pred_re_to_ue,
        )
        _update_change_confusion(
            counters,
            "ue_to_re",
            gt_changed=prev_gt_ue and (not cur_gt_ue),
            pred_changed=pred_ue_to_re,
        )


def _score_transition_case(
    case: EvalCase,
    args: argparse.Namespace,
    counters: Dict[str, int],
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """用容忍窗口评估一次 RS 或 EVENT 转折是否真的变对。"""

    if case.transition_index is None:
        return {}
    lo, hi = _transition_window_bounds(case, args)
    counters["transition_cases"] += 1
    result: Dict[str, Any] = {
        "transition_tolerance": int(args.transition_tolerance),
        "transition_hit_window": [lo, hi],
        "transition_case_hit": False,
        "transition_hit_frame": None,
        "transition_hit_timing": None,
    }
    if case.transition_kind == "rs":
        counters["rs_transition_cases"] += 1
        hit_index = _first_hit(records, "pred_rs", case.target_rs, lo, hi)
        left_edge = _record_at_or_after(records, lo)
        result.update(
            {
                "transition_source": case.source_rs,
                "transition_target": case.target_rs,
                "transition_test_field": "rs",
                "transition_already_at_target": bool(left_edge is not None and left_edge.get("pred_rs") == case.target_rs),
            }
        )
        if hit_index is not None:
            counters["rs_transition_hit_cases"] += 1
            if left_edge is not None and left_edge.get("pred_rs") == case.target_rs:
                counters["rs_transition_already_at_target_hits"] += 1
            timing = _classify_hit_timing(hit_index, int(case.transition_index))
            offset = hit_index - int(case.transition_index)
            counters[f"rs_transition_{timing}_hits"] += 1
            counters["rs_transition_hit_offset_sum"] += offset
            counters["rs_transition_abs_hit_offset_sum"] += abs(offset)
            if offset < 0:
                counters["rs_transition_max_early_lead"] = max(counters["rs_transition_max_early_lead"], abs(offset))
            if offset > 0:
                counters["rs_transition_max_late_lag"] = max(counters["rs_transition_max_late_lag"], offset)
            result.update(
                {
                    "transition_case_hit": True,
                    "transition_hit_frame": hit_index,
                    "transition_hit_offset": offset,
                    "transition_hit_timing": timing,
                }
            )
        return result

    event_hit = _first_hit(records, "pred_event", case.target_event, lo, hi)
    abnormal_hit = _first_bool_hit(records, "pred_abnormal_bool", case.target_abnormal, lo, hi)
    left_edge = _record_at_or_after(records, lo)
    target_edge = _record_at_or_after(records, int(case.transition_index))
    result.update(
        {
            "transition_source": case.source_event,
            "transition_target": case.target_event,
            "transition_source_abnormal": case.source_abnormal,
            "transition_target_abnormal": case.target_abnormal,
            "transition_test_field": "event",
            "abnormal_hit_frame": abnormal_hit,
            "transition_already_at_target": bool(left_edge is not None and left_edge.get("pred_event") == case.target_event),
            "transition_score_valid": not bool(target_edge is not None and not target_edge.get("event_score_valid", True)),
        }
    )
    if target_edge is not None and not target_edge.get("event_score_valid", True):
        counters["event_transition_dataset_candidate_mismatch_cases"] += 1
        return result
    counters["event_transition_cases"] += 1
    if abnormal_hit is not None:
        counters["event_transition_abnormal_hit_cases"] += 1
    if event_hit is not None:
        counters["event_transition_hit_cases"] += 1
        if left_edge is not None and left_edge.get("pred_event") == case.target_event:
            counters["event_transition_already_at_target_hits"] += 1
        timing = _classify_hit_timing(event_hit, int(case.transition_index))
        offset = event_hit - int(case.transition_index)
        counters[f"event_transition_{timing}_hits"] += 1
        counters["event_transition_hit_offset_sum"] += offset
        counters["event_transition_abs_hit_offset_sum"] += abs(offset)
        if offset < 0:
            counters["event_transition_max_early_lead"] = max(counters["event_transition_max_early_lead"], abs(offset))
        if offset > 0:
            counters["event_transition_max_late_lag"] = max(counters["event_transition_max_late_lag"], offset)
        result.update(
            {
                "transition_case_hit": True,
                "transition_hit_frame": event_hit,
                "transition_hit_offset": offset,
                "transition_hit_timing": timing,
            }
        )
    return result


def _write_frame_record(
    fp: Optional[TextIO],
    case: EvalCase,
    frame: Any,
    abs_index: int,
    parsed_q1: Dict[str, Any],
    parsed_q2: Optional[Dict[str, Any]],
    q1_text: str,
    q2_text: Optional[str],
    q1_rs_ok: bool,
    event_ok: Optional[bool],
    q2_candidate_source: str,
    q2_candidates: Optional[List[str]],
    event_reachable_under_pred_rs: bool,
    dataset_candidate_mismatch: bool,
    gt_event_code_raw: Optional[str],
    gt_regular_remapped: bool,
    args: argparse.Namespace,
) -> None:
    """把每帧自由生成结果写成 jsonl，便于定位转折处是否自行纠正。"""

    if fp is None:
        return
    hit_lo = hit_hi = None
    if case.transition_index is not None:
        hit_lo, hit_hi = _transition_window_bounds(case, args)
    rec = {
        "case_id": case.case_id,
        "scenario": getattr(case.route, "scenario", None),
        "route_id": getattr(case.route, "route_id", None),
        "route_index": case.route_index,
        "frame_id": frame.frame_id,
        "frame_index": abs_index,
        "transition_kind": case.transition_kind,
        "transition_index": case.transition_index,
        "transition_tolerance": int(args.transition_tolerance),
        "transition_hit_window": [hit_lo, hit_hi] if hit_lo is not None else None,
        "post_transition": case.transition_index is not None and abs_index >= case.transition_index,
        "in_transition_tolerance": hit_lo is not None and hit_lo <= abs_index <= hit_hi,
        "transition_source_rs": case.source_rs,
        "transition_target_rs": case.target_rs,
        "transition_source_event": case.source_event,
        "transition_target_event": case.target_event,
        "transition_source_abnormal": case.source_abnormal,
        "transition_target_abnormal": case.target_abnormal,
        "initial_memory_noise": args.initial_memory_noise,
        "image_ablation": getattr(args, "image_ablation", "none"),
        "goal_ablation": bool(getattr(args, "ablate_goal", False)),
        "gt_rs": frame.rs_label,
        "pred_rs": parsed_q1.get("rs_label"),
        "pred_rs_token": parsed_q1.get("rs_token"),
        "rs_ok": q1_rs_ok,
        "gt_abnormal": bool(frame.abnormal),
        "gt_event": frame.event_label,
        "gt_event_code_raw": gt_event_code_raw,
        "gt_regular_remapped": bool(gt_regular_remapped),
        "pred_event": parsed_q2.get("event_label") if parsed_q2 else None,
        "pred_event_token": parsed_q2.get("event_token") if parsed_q2 else None,
        "event_ok": event_ok,
        "pred_abnormal_bool": (is_unusual(str(parsed_q2.get("event_label"))) if parsed_q2 and parsed_q2.get("event_label") else None),
        "q2_candidate_source": q2_candidate_source,
        "q2_candidates": q2_candidates,
        "event_reachable_under_pred_rs": bool(event_reachable_under_pred_rs),
        "dataset_candidate_mismatch": bool(dataset_candidate_mismatch),
        "event_score_valid": not bool(dataset_candidate_mismatch),
        "q1_text": q1_text,
        "q2_text": q2_text,
    }
    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _evaluate_case(
    bundle: EvalBundle,
    case: EvalCase,
    args: argparse.Namespace,
    counters: Dict[str, int],
    jsonl_fp: Optional[TextIO],
) -> None:
    """执行单个评估片段。

    no-correction：Q1 错也继续问 Q2，但 Q2 候选必须按学生预测/维护的 RS 生成；
    Q2 非法只不更新 EVENT，下一帧继续沿用学生输出维护出的 memory。
    """

    memory: Optional[Memory] = None
    case_records: List[Dict[str, Any]] = []
    counters["cases"] += 1
    counters["case_frames"] += len(case.frames)
    counters["evaluated_routes"] += int(case.start_index == 0 and case.transition_index is None)
    first_gt_rs = case.frames[0].rs_label if case.frames else None
    first_pred_rs: Optional[str] = None
    prev_gt_rs: Optional[str] = None
    prev_pred_rs: Optional[str] = None
    for pos, frame in enumerate(case.frames):
        abs_index = case.start_index + pos
        if memory is None:
            memory = _initial_memory_for_frame(frame, args, case)
            counters["initial_noise_cases"] += int(args.initial_memory_noise != "none")
        else:
            memory = refresh_memory_goal(memory, _eval_goal_xy(frame, args))

        images = _apply_image_ablation(_load_images(frame.history_rgb_paths), args, case, frame)
        q1_text, q1_after = _generate_start(
            bundle,
            images,
            build_q1_prompt(memory, choice_seed=f"rs::{frame.frame_id}"),
            int(args.max_new_tokens_q1),
        )
        parsed_q1 = parse_q1_output(q1_text)
        if first_pred_rs is None:
            first_pred_rs = parsed_q1.get("rs_label")
        q1_rs_ok = parsed_q1.get("rs_label") == frame.rs_label

        counters["frames"] += 1
        counters["q1_rs_correct"] += int(q1_rs_ok)
        counters["q1_rs_wrong"] += int(not q1_rs_ok)
        pred_rs_label = parsed_q1.get("rs_label") if parsed_q1.get("rs_label") in RS_LABELS else "INVALID"
        counters[f"rs_cm_{frame.rs_label}_{pred_rs_label}"] += 1
        counters["rs_first_gt_baseline_correct"] += int(first_gt_rs is not None and frame.rs_label == first_gt_rs)
        counters["rs_first_pred_baseline_correct"] += int(first_pred_rs is not None and frame.rs_label == first_pred_rs)
        if prev_gt_rs is not None:
            counters["rs_change_denominator"] += 1
            counters["rs_gt_change_count"] += int(frame.rs_label != prev_gt_rs)
            counters["rs_pred_change_count"] += int(parsed_q1.get("rs_label") != prev_pred_rs)
        prev_gt_rs = frame.rs_label
        prev_pred_rs = parsed_q1.get("rs_label")
        if case.transition_index is not None and abs_index >= case.transition_index:
            counters["transition_post_frames"] += 1
            counters["transition_post_rs_correct"] += int(q1_rs_ok)

        memory = update_memory_after_q1(memory, student_rs_label=parsed_q1.get("rs_label"))
        q2_candidates, regular_codes, candidate_source, event_reachable = q2_candidates_for_student_rs(
            frame,
            memory.rs_label if memory.rs_label in RS_LABELS else parsed_q1.get("rs_label"),
            seed=int(args.seed),
        )
        counters[f"q2_candidate_source_{candidate_source}"] += 1
        counters["q2_candidates_from_pred_rs"] += int(candidate_source == "pred_rs_static_candidates")
        is_multi_candidate = len(q2_candidates) > 1
        counters["q2_single_candidate"] += int(not is_multi_candidate)
        counters["q2_multi_candidate"] += int(is_multi_candidate)
        dataset_candidate_mismatch = not event_in_candidates(frame.event_label, frame.event_candidates)
        counters["dataset_candidate_mismatch"] += int(dataset_candidate_mismatch)
        counters["dataset_candidate_mismatch_ue"] += int(dataset_candidate_mismatch and bool(frame.abnormal))
        counters["dataset_candidate_mismatch_re"] += int(dataset_candidate_mismatch and not bool(frame.abnormal))
        counters["event_gt_ue_frames"] += int(bool(frame.abnormal))
        counters["event_gt_re_frames"] += int(not bool(frame.abnormal))
        event_score_valid = not dataset_candidate_mismatch
        counters["event_score_valid_frames"] += int(event_score_valid)
        counters["q2_multi_candidate_scored"] += int(event_score_valid and is_multi_candidate)
        counters["event_unreachable_due_to_rs"] += int(event_score_valid and not event_reachable)

        q2_prompt = build_q2_prompt(
            memory,
            candidates=q2_candidates,
            regular_event_codes=regular_codes,
        )
        q2_text, _ = _generate_next(bundle, q1_after, q2_prompt, int(args.max_new_tokens_q2))
        parsed_q2 = parse_q2_output(q2_text, q2_candidates)
        baseline_target = _event_target_from_frame(frame)
        target = _event_target_from_frame(frame, student_event=parsed_q2.get("event_label"))
        gt_regular_remapped = (not bool(frame.abnormal)) and str(baseline_target.event_code) != str(baseline_target.label)
        event_ok = (bool(event_reachable) and parsed_q2.get("event_label") == target.label) if event_score_valid else None
        pred_event_label = parsed_q2.get("event_label") or "INVALID"
        pred_ue = is_unusual(str(pred_event_label))
        gt_ue = bool(frame.abnormal)
        if event_score_valid:
            counters["q2_joint_correct"] += int(q1_rs_ok and bool(event_ok))
            counters["q2_event_correct_when_rs_wrong"] += int((not q1_rs_ok) and bool(event_ok))
            candidate_count_bucket = min(10, max(1, len(q2_candidates)))
            bucket_prefix = f"q2_candidate_count_{candidate_count_bucket}"
            counters[f"{bucket_prefix}_total"] += 1
            counters[f"{bucket_prefix}_correct"] += int(bool(event_ok))
            if pred_ue and gt_ue:
                counters[f"{bucket_prefix}_ue_tp"] += 1
            elif pred_ue and not gt_ue:
                counters[f"{bucket_prefix}_ue_fp"] += 1
            elif (not pred_ue) and gt_ue:
                counters[f"{bucket_prefix}_ue_fn"] += 1
            else:
                counters[f"{bucket_prefix}_ue_tn"] += 1
            rs_bucket_prefix = f"q2_rs_{frame.rs_label}_candidate_count_{candidate_count_bucket}"
            counters[f"{rs_bucket_prefix}_total"] += 1
            counters[f"{rs_bucket_prefix}_correct"] += int(bool(event_ok))
            if pred_ue and gt_ue:
                counters[f"{rs_bucket_prefix}_ue_tp"] += 1
            elif pred_ue and not gt_ue:
                counters[f"{rs_bucket_prefix}_ue_fp"] += 1
            elif (not pred_ue) and gt_ue:
                counters[f"{rs_bucket_prefix}_ue_fn"] += 1
            else:
                counters[f"{rs_bucket_prefix}_ue_tn"] += 1
            counters["event_global_majority_correct"] += int(target.label == _GLOBAL_EVENT_MAJORITY_LABEL)
            cm_pred_event = pred_event_label if event_reachable else "UNREACHABLE"
            counters[f"event_cm_{target.label}_{cm_pred_event}"] += 1
            if not gt_ue:
                remap_group = "remapped" if gt_regular_remapped else "unchanged"
                counters[f"event_raw_regular_{remap_group}_total"] += 1
                counters[f"event_raw_regular_{remap_group}_correct"] += int(bool(event_ok))
                counters[f"event_raw_regular_{remap_group}_ue_fp"] += int(pred_ue)
                counters[f"event_raw_regular_by_rs_{frame.rs_label}_{remap_group}_total"] += 1
                counters[f"event_raw_regular_by_rs_{frame.rs_label}_{remap_group}_correct"] += int(bool(event_ok))
                counters[f"event_raw_regular_by_rs_{frame.rs_label}_{remap_group}_ue_fp"] += int(pred_ue)
                counters[f"event_raw_regular_combo_{frame.rs_label}_{baseline_target.event_code}_{baseline_target.label}_total"] += 1
                counters[f"event_raw_regular_combo_{frame.rs_label}_{baseline_target.event_code}_{baseline_target.label}_correct"] += int(bool(event_ok))
                counters[f"event_raw_regular_combo_{frame.rs_label}_{baseline_target.event_code}_{baseline_target.label}_ue_fp"] += int(pred_ue)
                baseline_label = REGULAR_MAJORITY_EVENT_BY_RS.get(frame.rs_label)
                counters["regular_majority_static_correct"] += int(baseline_target.label == baseline_label)
                counters[f"regular_majority_static_by_rs_{frame.rs_label}_total"] += 1
                counters[f"regular_majority_static_by_rs_{frame.rs_label}_correct"] += int(baseline_target.label == baseline_label)
                if not event_reachable:
                    regular_pred_bucket = "UNREACHABLE"
                elif parsed_q2.get("event_label") is None:
                    regular_pred_bucket = "INVALID"
                elif is_regular_event(str(parsed_q2.get("event_label"))):
                    regular_pred_bucket = str(parsed_q2.get("event_label"))
                else:
                    regular_pred_bucket = "UE"
                counters[f"regular_cm_{target.label}_{regular_pred_bucket}"] += 1
                counters[f"regular_gt_by_rs_{frame.rs_label}_{baseline_target.label}"] += 1
                if q1_rs_ok:
                    counters[f"regular_gt_by_rs_when_rs_correct_{frame.rs_label}_{baseline_target.label}"] += 1
            cell_prefix = "q2_multi" if is_multi_candidate else "q2_single"
            gt_prefix = "ue" if gt_ue else "re"
            counters[f"{cell_prefix}_{gt_prefix}_total"] += 1
            counters[f"{cell_prefix}_{gt_prefix}_correct"] += int(bool(event_ok))
            if (not is_multi_candidate) and parsed_q2.get("event_label") is None:
                counters["q2_single_candidate_invalid"] += 1
            if is_multi_candidate and (not gt_ue) and pred_ue:
                counters["ue_fp_on_multi_candidate_re"] += 1
            if is_multi_candidate and not gt_ue:
                counters[f"q2_rs_{frame.rs_label}_multi_re_total"] += 1
                counters[f"q2_rs_{frame.rs_label}_multi_re_correct"] += int(bool(event_ok))
                counters[f"q2_rs_{frame.rs_label}_multi_re_ue_fp"] += int(pred_ue)
                counters[f"q2_rs_{frame.rs_label}_multi_re_regular_tn"] += int(bool(parsed_q2.get("event_label") and is_regular_event(str(parsed_q2.get("event_label")))))
                counters[f"q2_rs_{frame.rs_label}_multi_re_invalid"] += int(parsed_q2.get("event_label") is None)
            if is_multi_candidate and gt_ue:
                counters[f"q2_rs_{frame.rs_label}_multi_ue_total"] += 1
                counters[f"q2_rs_{frame.rs_label}_multi_ue_correct"] += int(bool(event_ok))
                counters[f"q2_rs_{frame.rs_label}_multi_ue_pred_regular"] += int(bool(parsed_q2.get("event_label") and is_regular_event(str(parsed_q2.get("event_label")))))
                counters[f"q2_rs_{frame.rs_label}_multi_ue_pred_invalid"] += int(parsed_q2.get("event_label") is None)
            if pred_ue and gt_ue:
                counters["ue_binary_tp"] += 1
            elif pred_ue and not gt_ue:
                counters["ue_binary_fp"] += 1
            elif (not pred_ue) and gt_ue:
                counters["ue_binary_fn"] += 1
            else:
                counters["ue_binary_tn"] += 1
            if is_multi_candidate:
                counters["q2_event_correct_multi_candidate"] += int(bool(event_ok))
                if parsed_q2.get("event_label") and is_regular_event(str(parsed_q2.get("event_label"))):
                    counters["q2_pred_re_multi_candidate"] += 1
                elif parsed_q2.get("event_label") is None:
                    counters["q2_pred_invalid_multi_candidate"] += 1
                else:
                    counters["q2_pred_ue_multi_candidate"] += 1
                if pred_ue and gt_ue:
                    counters["ue_binary_tp_multi_candidate"] += 1
                elif pred_ue and not gt_ue:
                    counters["ue_binary_fp_multi_candidate"] += 1
                elif (not pred_ue) and gt_ue:
                    counters["ue_binary_fn_multi_candidate"] += 1
                else:
                    counters["ue_binary_tn_multi_candidate"] += 1

        counters["q2_triggered"] += 1
        counters["q2_candidate_mismatch"] += int(event_score_valid and not event_reachable)
        if event_score_valid:
            counters["q2_event_correct"] += int(bool(event_ok))
            counters["q2_when_rs_correct"] += int(q1_rs_ok)
            counters["q2_event_correct_when_rs_correct"] += int(q1_rs_ok and bool(event_ok))
            counters["q2_gt_re_when_rs_correct"] += int(q1_rs_ok and not target.abnormal)
            counters["q2_gt_re_total"] += int(not target.abnormal)
            if parsed_q2.get("event_label") and is_regular_event(str(parsed_q2.get("event_label"))):
                counters["q2_pred_re"] += 1
            elif parsed_q2.get("event_label") is None:
                counters["q2_pred_invalid"] += 1
            else:
                counters["q2_pred_ue"] += 1
            if case.transition_index is not None and abs_index >= case.transition_index:
                counters["transition_post_q2_triggered"] += 1
                counters["transition_post_event_correct"] += int(bool(event_ok))
            if frame.abnormal:
                counters["q2_ue_total"] += 1
                counters["q2_ue_correct"] += int(bool(event_ok))
                counters["q2_ue_pred_regular"] += int(bool(parsed_q2.get("event_label") and is_regular_event(str(parsed_q2.get("event_label")))))
            else:
                counters["q2_re_total"] += 1
                counters["q2_re_correct"] += int(bool(event_ok))

        memory = update_memory_after_q2(memory, student_event_label=parsed_q2.get("event_label"))
        if parsed_q2.get("event_label") is None:
            counters["q2_invalid_output"] += 1
        case_records.append(
            {
                "frame_index": abs_index,
                "gt_rs": frame.rs_label,
                "gt_event": target.label,
                "gt_event_code_raw": baseline_target.event_code,
                "gt_regular_remapped": gt_regular_remapped,
                "gt_abnormal": bool(frame.abnormal),
                "event_score_valid": event_score_valid,
                "pred_rs": parsed_q1.get("rs_label"),
                "pred_abnormal_bool": pred_ue,
                "pred_event": parsed_q2.get("event_label") if event_score_valid and event_reachable else None,
            }
        )
        _write_frame_record(
            jsonl_fp,
            case,
            frame,
            abs_index,
            parsed_q1,
            parsed_q2,
            q1_text,
            q2_text,
            q1_rs_ok,
            event_ok,
            candidate_source,
            q2_candidates,
            event_reachable,
            dataset_candidate_mismatch,
            baseline_target.event_code,
            gt_regular_remapped,
            args,
        )
    _score_adjacent_changes(counters, case_records)
    transition_result = _score_transition_case(case, args, counters, case_records)
    pred_rs_values = [rec.get("pred_rs") for rec in case_records if rec.get("pred_rs") is not None]
    if pred_rs_values and len(set(pred_rs_values)) == 1:
        counters["rs_locked_cases"] += 1
        counters["rs_locked_eq_first_gt_cases"] += int(first_gt_rs is not None and pred_rs_values[0] == first_gt_rs)
    if jsonl_fp is not None and transition_result:
        jsonl_fp.write(
            json.dumps(
                {
                    "record_type": "transition_case_summary",
                    "case_id": case.case_id,
                    "scenario": getattr(case.route, "scenario", None),
                    "route_id": getattr(case.route, "route_id", None),
                    "route_index": case.route_index,
                    "transition_index": case.transition_index,
                    "transition_kind": case.transition_kind,
                    **transition_result,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _new_counters() -> Dict[str, int]:
    """创建评估计数器。

    所有指标都先累积成整数计数；多卡时每个 rank 只更新自己的分片，最后用
    `dist.all_reduce(SUM)` 合并，再由 rank0 统一计算比例指标。
    """

    counters: Dict[str, int] = {
        "cases": 0,
        "case_frames": 0,
        "evaluated_routes": 0,
        "frames": 0,
        "q1_rs_correct": 0,
        "q1_rs_wrong": 0,
        "rs_first_gt_baseline_correct": 0,
        "rs_first_pred_baseline_correct": 0,
        "rs_pred_change_count": 0,
        "rs_gt_change_count": 0,
        "rs_change_denominator": 0,
        "rs_change_tp": 0,
        "rs_change_fp": 0,
        "rs_change_fn": 0,
        "rs_change_tn": 0,
        "rs_change_invalid": 0,
        "rs_locked_cases": 0,
        "rs_locked_eq_first_gt_cases": 0,
        "q2_triggered": 0,
        "event_score_valid_frames": 0,
        "event_gt_ue_frames": 0,
        "event_gt_re_frames": 0,
        "dataset_candidate_mismatch": 0,
        "dataset_candidate_mismatch_ue": 0,
        "dataset_candidate_mismatch_re": 0,
        "q2_event_correct": 0,
        "q2_joint_correct": 0,
        "q2_when_rs_correct": 0,
        "q2_event_correct_when_rs_correct": 0,
        "q2_event_correct_when_rs_wrong": 0,
        "q2_gt_re_when_rs_correct": 0,
        "event_global_majority_correct": 0,
        "regular_majority_static_correct": 0,
        "event_raw_regular_remapped_total": 0,
        "event_raw_regular_remapped_correct": 0,
        "event_raw_regular_remapped_ue_fp": 0,
        "event_raw_regular_unchanged_total": 0,
        "event_raw_regular_unchanged_correct": 0,
        "event_raw_regular_unchanged_ue_fp": 0,
        "q2_candidate_mismatch": 0,
        "q2_invalid_output": 0,
        "q2_ue_total": 0,
        "q2_ue_correct": 0,
        "q2_ue_pred_regular": 0,
        "q2_re_total": 0,
        "q2_re_correct": 0,
        "q2_gt_re_total": 0,
        "q2_pred_re": 0,
        "q2_pred_ue": 0,
        "q2_pred_invalid": 0,
        "q2_candidates_from_pred_rs": 0,
        "q2_candidate_source_pred_rs_static_candidates": 0,
        "q2_candidate_source_invalid_rs_fallback": 0,
        "q2_single_candidate": 0,
        "q2_multi_candidate": 0,
        "q2_multi_candidate_scored": 0,
        "q2_event_correct_multi_candidate": 0,
        "q2_pred_re_multi_candidate": 0,
        "q2_pred_ue_multi_candidate": 0,
        "q2_pred_invalid_multi_candidate": 0,
        "q2_single_re_total": 0,
        "q2_single_re_correct": 0,
        "q2_single_ue_total": 0,
        "q2_single_ue_correct": 0,
        "q2_multi_re_total": 0,
        "q2_multi_re_correct": 0,
        "q2_multi_ue_total": 0,
        "q2_multi_ue_correct": 0,
        "q2_single_candidate_invalid": 0,
        "ue_fp_on_multi_candidate_re": 0,
        "event_unreachable_due_to_rs": 0,
        "ue_binary_tp": 0,
        "ue_binary_fp": 0,
        "ue_binary_fn": 0,
        "ue_binary_tn": 0,
        "ue_binary_tp_multi_candidate": 0,
        "ue_binary_fp_multi_candidate": 0,
        "ue_binary_fn_multi_candidate": 0,
        "ue_binary_tn_multi_candidate": 0,
        "re_to_ue_tp": 0,
        "re_to_ue_fp": 0,
        "re_to_ue_fn": 0,
        "re_to_ue_tn": 0,
        "re_to_ue_invalid": 0,
        "ue_to_re_tp": 0,
        "ue_to_re_fp": 0,
        "ue_to_re_fn": 0,
        "ue_to_re_tn": 0,
        "ue_to_re_invalid": 0,
        "script_resets": 0,
        "rs_wrong_resets": 0,
        "initial_noise_cases": 0,
        "transition_cases": 0,
        "rs_transition_cases": 0,
        "rs_transition_hit_cases": 0,
        "rs_transition_already_at_target_hits": 0,
        "rs_transition_early_hits": 0,
        "rs_transition_on_time_hits": 0,
        "rs_transition_late_hits": 0,
        "rs_transition_hit_offset_sum": 0,
        "rs_transition_abs_hit_offset_sum": 0,
        "rs_transition_max_early_lead": 0,
        "rs_transition_max_late_lag": 0,
        "event_transition_cases": 0,
        "event_transition_dataset_candidate_mismatch_cases": 0,
        "event_transition_hit_cases": 0,
        "event_transition_abnormal_hit_cases": 0,
        "event_transition_already_at_target_hits": 0,
        "event_transition_early_hits": 0,
        "event_transition_on_time_hits": 0,
        "event_transition_late_hits": 0,
        "event_transition_hit_offset_sum": 0,
        "event_transition_abs_hit_offset_sum": 0,
        "event_transition_max_early_lead": 0,
        "event_transition_max_late_lag": 0,
        "transition_post_frames": 0,
        "transition_post_rs_correct": 0,
        "transition_post_q2_triggered": 0,
        "transition_post_event_correct": 0,
    }
    for gt in _RS_CM_LABELS:
        for pred in _RS_CM_LABELS:
            counters[f"rs_cm_{gt}_{pred}"] = 0
    for gt in _EVENT_LABELS:
        for pred in _EVENT_CM_LABELS:
            counters[f"event_cm_{gt}_{pred}"] = 0
    for gt in REGULAR_EVENT_LABELS:
        for pred in _REGULAR_CM_LABELS:
            counters[f"regular_cm_{gt}_{pred}"] = 0
    for rs in RS_LABELS:
        counters[f"regular_majority_static_by_rs_{rs}_total"] = 0
        counters[f"regular_majority_static_by_rs_{rs}_correct"] = 0
        counters[f"q2_rs_{rs}_multi_re_total"] = 0
        counters[f"q2_rs_{rs}_multi_re_correct"] = 0
        counters[f"q2_rs_{rs}_multi_re_ue_fp"] = 0
        counters[f"q2_rs_{rs}_multi_re_regular_tn"] = 0
        counters[f"q2_rs_{rs}_multi_re_invalid"] = 0
        counters[f"q2_rs_{rs}_multi_ue_total"] = 0
        counters[f"q2_rs_{rs}_multi_ue_correct"] = 0
        counters[f"q2_rs_{rs}_multi_ue_pred_regular"] = 0
        counters[f"q2_rs_{rs}_multi_ue_pred_invalid"] = 0
        for remap_group in ("remapped", "unchanged"):
            counters[f"event_raw_regular_by_rs_{rs}_{remap_group}_total"] = 0
            counters[f"event_raw_regular_by_rs_{rs}_{remap_group}_correct"] = 0
            counters[f"event_raw_regular_by_rs_{rs}_{remap_group}_ue_fp"] = 0
        for label in REGULAR_EVENT_LABELS:
            counters[f"regular_gt_by_rs_{rs}_{label}"] = 0
            counters[f"regular_gt_by_rs_when_rs_correct_{rs}_{label}"] = 0
        for raw in REGULAR_EVENT_LABELS:
            for mapped in REGULAR_EVENT_LABELS:
                counters[f"event_raw_regular_combo_{rs}_{raw}_{mapped}_total"] = 0
                counters[f"event_raw_regular_combo_{rs}_{raw}_{mapped}_correct"] = 0
                counters[f"event_raw_regular_combo_{rs}_{raw}_{mapped}_ue_fp"] = 0
    for gt in _RS_TRANSITION_LABELS:
        for pred in _RS_TRANSITION_LABELS:
            counters[f"rs_transition_dir_cm_{gt}_{pred}"] = 0
    for gt in _EVENT_TRANSITION_LABELS:
        for pred in _EVENT_TRANSITION_LABELS:
            counters[f"event_transition_dir_cm_{gt}_{pred}"] = 0
    for n in range(1, 11):
        prefix = f"q2_candidate_count_{n}"
        counters[f"{prefix}_total"] = 0
        counters[f"{prefix}_correct"] = 0
        counters[f"{prefix}_ue_tp"] = 0
        counters[f"{prefix}_ue_fp"] = 0
        counters[f"{prefix}_ue_fn"] = 0
        counters[f"{prefix}_ue_tn"] = 0
        for rs in RS_LABELS:
            rs_prefix = f"q2_rs_{rs}_candidate_count_{n}"
            counters[f"{rs_prefix}_total"] = 0
            counters[f"{rs_prefix}_correct"] = 0
            counters[f"{rs_prefix}_ue_tp"] = 0
            counters[f"{rs_prefix}_ue_fp"] = 0
            counters[f"{rs_prefix}_ue_fn"] = 0
            counters[f"{rs_prefix}_ue_tn"] = 0
    return counters


def _sync_counters(counters: Dict[str, int], device: torch.device) -> Dict[str, int]:
    """跨 rank 汇总整数 counters。"""

    if not dist.is_available() or not dist.is_initialized():
        return counters
    sum_keys = [key for key in counters if "_max_" not in key]
    max_keys = [key for key in counters if "_max_" in key]
    synced = dict(counters)
    if sum_keys:
        # 大部分字段是计数或 offset sum，跨 rank 应该求和。
        sum_tensor = torch.tensor(
            [int(counters[k]) for k in sum_keys],
            dtype=torch.long,
            device=device,
        )
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        synced.update({key: int(value) for key, value in zip(sum_keys, sum_tensor.cpu().tolist())})
    if max_keys:
        # `*_max_early_lead` / `*_max_late_lag` 是极值，不能被 SUM 放大。
        max_tensor = torch.tensor(
            [int(counters[k]) for k in max_keys],
            dtype=torch.long,
            device=device,
        )
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        synced.update({key: int(value) for key, value in zip(max_keys, max_tensor.cpu().tolist())})
    return synced


def _rank_jsonl_path(output_jsonl: str, rank: int, world_size: int) -> pathlib.Path:
    """多卡时每个 rank 先写自己的 jsonl 分片，rank0 再合并。"""

    out_path = pathlib.Path(output_jsonl)
    if world_size <= 1:
        return out_path
    return out_path.with_suffix(out_path.suffix + f".rank{rank}")


def _merge_rank_jsonl(output_jsonl: str, world_size: int) -> None:
    """rank0 合并各 rank 的 jsonl 分片到用户指定的最终 output-jsonl。"""

    if world_size <= 1:
        return
    out_path = pathlib.Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as dst:
        for rank in range(world_size):
            shard_path = _rank_jsonl_path(output_jsonl, rank, world_size)
            if not shard_path.exists():
                continue
            with open(shard_path, "r", encoding="utf-8") as src:
                for line in src:
                    dst.write(line)
            shard_path.unlink(missing_ok=True)


def _eval_stem(eval_mode: str) -> str:
    """把 eval_mode 统一映射成文件/目录名。

    日常命令只传 `--task rs/event/full`，内部会展开成 eval_mode；所有自动输出
    都复用这个 stem，保证不同任务的结果不会互相覆盖。
    """

    return {
        "full_route": "full_route",
        "rs_transition": "rs_transition",
        "event_transition": "event_transition",
    }[eval_mode]


def _eval_result_group(args: argparse.Namespace) -> str:
    """生成自动输出的任务目录名，图像消融会单独分目录。"""

    stem = _eval_stem(args.eval_mode)
    suffixes: List[str] = []
    ablation = str(getattr(args, "image_ablation", "none"))
    if ablation != "none":
        suffixes.append(ablation)
    if bool(getattr(args, "ablate_goal", False)):
        suffixes.append("no_goal")
    if suffixes:
        return f"{stem}_{'_'.join(suffixes)}"
    return stem


def _dist_broadcast_text(value: Optional[str], *, rank: int, world_size: int) -> Optional[str]:
    """把 rank0 生成的字符串广播到所有 rank。

    自动输出目录里带时间戳。多卡 torchrun 时每个进程都会独立 parse args，如果各自
    `datetime.now()`，就可能因为秒级差异写到不同目录。这里由 rank0 生成一次目录，
    再广播给其它 rank，确保 jsonl 分片和最终合并落在同一个地方。
    """

    if world_size <= 1:
        return value
    values: List[Optional[str]] = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _prepare_eval_outputs(args: argparse.Namespace, *, rank: int, world_size: int) -> None:
    """补齐本次评测的输出路径。

    默认目录结构：

    `adapter_run/eval_results/<task>/<YYYYMMDD_HHMMSS>/`

    目录内固定写三类文件：
    - `metrics.json`：汇总指标，适合脚本读。
    - `frames.jsonl`：逐帧/逐 case 复盘，适合排查错误样本。
    - `summary.md`：中文说明和关键指标，适合人工快速看。
    """

    explicit_output_dir = getattr(args, "output_dir", None)
    if explicit_output_dir:
        output_dir = pathlib.Path(explicit_output_dir)
    elif not args.output_json and not args.output_jsonl:
        if rank == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = _default_eval_output_root(args.adapter_dir) / "eval_results" / _eval_result_group(args) / timestamp
            output_dir_text: Optional[str] = str(output_dir)
        else:
            output_dir_text = None
        output_dir = pathlib.Path(str(_dist_broadcast_text(output_dir_text, rank=rank, world_size=world_size)))
    elif args.output_json:
        output_dir = pathlib.Path(args.output_json).parent
    else:
        output_dir = pathlib.Path(args.output_jsonl).parent

    args.output_dir = str(output_dir)
    if not args.output_json:
        args.output_json = str(output_dir / "metrics.json")
    if not args.output_jsonl:
        args.output_jsonl = str(output_dir / "frames.jsonl")
    if not getattr(args, "output_summary", None):
        args.output_summary = str(output_dir / "summary.md")
    pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """由 TP/FP/FN 计算 P/R/F1。"""

    precision = float(tp) / max(float(tp + fp), 1.0)
    recall = float(tp) / max(float(tp + fn), 1.0)
    f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _multiclass_report(counters: Dict[str, int], *, prefix: str, labels: tuple[str, ...], pred_labels: tuple[str, ...]) -> Dict[str, Any]:
    """从 counters 中的混淆矩阵字段生成 per-class 与 macro 指标。"""

    matrix: Dict[str, Dict[str, int]] = {
        gt: {pred: int(counters.get(f"{prefix}_cm_{gt}_{pred}", 0)) for pred in pred_labels}
        for gt in labels
    }
    per_class: Dict[str, Dict[str, float]] = {}
    support_total = 0
    correct_total = 0
    for label in labels:
        tp = matrix[label].get(label, 0)
        fp = sum(matrix[gt].get(label, 0) for gt in labels if gt != label)
        fn = sum(count for pred, count in matrix[label].items() if pred != label)
        support = sum(matrix[label].values())
        support_total += support
        correct_total += tp
        per_class[label] = {
            **_prf(tp, fp, fn),
            "support": float(support),
            "predicted": float(sum(matrix[gt].get(label, 0) for gt in labels)),
        }
    macro_f1 = sum(v["f1"] for v in per_class.values()) / max(len(per_class), 1)
    return {
        "confusion_matrix": matrix,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "micro_acc": float(correct_total) / max(float(support_total), 1.0),
    }


def _change_report(counters: Dict[str, int], prefix: str) -> Dict[str, float]:
    """生成相邻帧变化检测报告，FP/TN 侧对应不该切时是否乱切。"""

    tp = int(counters.get(f"{prefix}_tp", 0))
    fp = int(counters.get(f"{prefix}_fp", 0))
    fn = int(counters.get(f"{prefix}_fn", 0))
    tn = int(counters.get(f"{prefix}_tn", 0))
    invalid = int(counters.get(f"{prefix}_invalid", 0))
    prf = _prf(tp, fp, fn)
    return {
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "false_transition_rate_when_gt_stable": float(fp) / max(float(fp + tn), 1.0),
        "invalid_rate": float(invalid) / max(float(tp + fp + fn + tn + invalid), 1.0),
    }


def _sparse_transition_direction_report(
    counters: Dict[str, int],
    *,
    prefix: str,
    labels: tuple[str, ...],
) -> Dict[str, Dict[str, int]]:
    """输出非零的转折方向混淆矩阵，避免 metrics.json 被大量 0 占满。"""

    report: Dict[str, Dict[str, int]] = {}
    for gt in labels:
        row: Dict[str, int] = {}
        for pred in labels:
            count = int(counters.get(f"{prefix}_transition_dir_cm_{gt}_{pred}", 0))
            if count:
                row[pred] = count
        if row:
            report[gt] = row
    return report


def _build_metrics(
    *,
    args: argparse.Namespace,
    counters: Dict[str, int],
    route_count: int,
    selected_case_count: int,
    world_size: int,
) -> Dict[str, Any]:
    """由合并后的 counters 计算最终 metrics JSON。"""

    frames = max(1, counters["frames"])
    q2_total = max(1, counters["event_score_valid_frames"])
    q2_multi_total = max(1, counters["q2_multi_candidate_scored"])
    q2_rs_correct_total = max(1, counters["q2_when_rs_correct"])
    transition_post_frames = max(1, counters["transition_post_frames"])
    transition_post_q2 = max(1, counters["transition_post_q2_triggered"])
    rs_first_gt_baseline = counters["rs_first_gt_baseline_correct"] / frames
    rs_first_pred_baseline = counters["rs_first_pred_baseline_correct"] / frames
    rs_acc = counters["q1_rs_correct"] / frames
    regular_majority_correct = 0
    regular_majority_correct_rs = 0
    regular_majority_by_rs: Dict[str, Dict[str, Any]] = {}
    regular_majority_static_by_rs: Dict[str, Dict[str, Any]] = {}
    for rs in RS_LABELS:
        counts = {label: int(counters.get(f"regular_gt_by_rs_{rs}_{label}", 0)) for label in REGULAR_EVENT_LABELS}
        correct = max(counts.values()) if counts else 0
        label = max(counts.items(), key=lambda item: item[1])[0] if counts else None
        regular_majority_correct += correct
        counts_rs = {label: int(counters.get(f"regular_gt_by_rs_when_rs_correct_{rs}_{label}", 0)) for label in REGULAR_EVENT_LABELS}
        regular_majority_correct_rs += max(counts_rs.values()) if counts_rs else 0
        regular_majority_by_rs[rs] = {
            "majority_regular_label": label,
            "majority_count": correct,
            "regular_counts": counts,
        }
        static_total = int(counters.get(f"regular_majority_static_by_rs_{rs}_total", 0))
        static_correct = int(counters.get(f"regular_majority_static_by_rs_{rs}_correct", 0))
        regular_majority_static_by_rs[rs] = {
            "majority_regular_label": REGULAR_MAJORITY_EVENT_BY_RS.get(rs),
            "regular_correct": static_correct,
            "regular_total": static_total,
            "regular_accuracy": static_correct / max(1, static_total),
            "full_data_regular_accuracy_reference": REGULAR_ZERO_INFO_BASELINE_BY_RS.get(rs),
        }
    event_regular_baseline_oracle_majority_q2 = regular_majority_correct / q2_total
    event_regular_baseline_given_gt_rs = counters["regular_majority_static_correct"] / q2_total
    event_global_majority_baseline = counters["event_global_majority_correct"] / q2_total
    event_acc_q2 = counters["q2_event_correct"] / q2_total
    event_regular_baseline_rs_correct = regular_majority_correct_rs / q2_rs_correct_total
    event_acc_rs_correct = counters["q2_event_correct_when_rs_correct"] / q2_rs_correct_total
    change_den = max(1, counters["rs_change_denominator"])
    ue_binary = _prf(counters["ue_binary_tp"], counters["ue_binary_fp"], counters["ue_binary_fn"])
    ue_binary_multi = _prf(
        counters["ue_binary_tp_multi_candidate"],
        counters["ue_binary_fp_multi_candidate"],
        counters["ue_binary_fn_multi_candidate"],
    )
    rs_report = _multiclass_report(counters, prefix="rs", labels=RS_LABELS, pred_labels=_RS_CM_LABELS)
    event_report = _multiclass_report(counters, prefix="event", labels=_EVENT_LABELS, pred_labels=_EVENT_CM_LABELS)
    regular_report = _multiclass_report(counters, prefix="regular", labels=REGULAR_EVENT_LABELS, pred_labels=_REGULAR_CM_LABELS)
    rs_gt_r3_total = sum(counters.get(f"rs_cm_R3_{pred}", 0) for pred in _RS_CM_LABELS)
    rs_pred_r3_total = sum(counters.get(f"rs_cm_{gt}_R3", 0) for gt in _RS_CM_LABELS)
    rs_r3_tp = counters.get("rs_cm_R3_R3", 0)
    candidate_count_report: Dict[str, Dict[str, Any]] = {}
    rs_candidate_count_report: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for n in range(1, 11):
        prefix = f"q2_candidate_count_{n}"
        total = counters.get(f"{prefix}_total", 0)
        if total <= 0:
            continue
        prf = _prf(
            counters.get(f"{prefix}_ue_tp", 0),
            counters.get(f"{prefix}_ue_fp", 0),
            counters.get(f"{prefix}_ue_fn", 0),
        )
        candidate_count_report[str(n)] = {
            "total": total,
            "accuracy": counters.get(f"{prefix}_correct", 0) / max(1, total),
            "ue_vs_re_tp": counters.get(f"{prefix}_ue_tp", 0),
            "ue_vs_re_fp": counters.get(f"{prefix}_ue_fp", 0),
            "ue_vs_re_fn": counters.get(f"{prefix}_ue_fn", 0),
            "ue_vs_re_tn": counters.get(f"{prefix}_ue_tn", 0),
            "ue_vs_re_precision": prf["precision"],
            "ue_vs_re_recall": prf["recall"],
            "ue_vs_re_f1": prf["f1"],
        }
    for rs in RS_LABELS:
        rs_report_by_count: Dict[str, Dict[str, Any]] = {}
        for n in range(1, 11):
            prefix = f"q2_rs_{rs}_candidate_count_{n}"
            total = counters.get(f"{prefix}_total", 0)
            if total <= 0:
                continue
            prf = _prf(
                counters.get(f"{prefix}_ue_tp", 0),
                counters.get(f"{prefix}_ue_fp", 0),
                counters.get(f"{prefix}_ue_fn", 0),
            )
            rs_report_by_count[str(n)] = {
                "total": total,
                "accuracy": counters.get(f"{prefix}_correct", 0) / max(1, total),
                "ue_vs_re_tp": counters.get(f"{prefix}_ue_tp", 0),
                "ue_vs_re_fp": counters.get(f"{prefix}_ue_fp", 0),
                "ue_vs_re_fn": counters.get(f"{prefix}_ue_fn", 0),
                "ue_vs_re_tn": counters.get(f"{prefix}_ue_tn", 0),
                "ue_vs_re_precision": prf["precision"],
                "ue_vs_re_recall": prf["recall"],
                "ue_vs_re_f1": prf["f1"],
            }
        if rs_report_by_count:
            rs_candidate_count_report[rs] = rs_report_by_count
    ue_fp_on_multi_candidate_re_by_rs: Dict[str, Dict[str, Any]] = {}
    q2_multi_re_by_rs_report: Dict[str, Dict[str, Any]] = {}
    q2_multi_ue_by_rs_report: Dict[str, Dict[str, Any]] = {}
    for rs in RS_LABELS:
        total_re = int(counters.get(f"q2_rs_{rs}_multi_re_total", 0))
        correct_re = int(counters.get(f"q2_rs_{rs}_multi_re_correct", 0))
        fp = int(counters.get(f"q2_rs_{rs}_multi_re_ue_fp", 0))
        tn = int(counters.get(f"q2_rs_{rs}_multi_re_regular_tn", 0))
        invalid_re = int(counters.get(f"q2_rs_{rs}_multi_re_invalid", 0))
        ue_fp_on_multi_candidate_re_by_rs[rs] = {
            "regular_total": total_re,
            "ue_fp": fp,
            "regular_tn": tn,
            "invalid": invalid_re,
            "rate": float(fp) / max(float(total_re), 1.0),
        }
        q2_multi_re_by_rs_report[rs] = {
            "total": total_re,
            "accuracy": correct_re / max(1, total_re),
            "ue_fp": fp,
            "pred_regular": tn,
            "pred_invalid": invalid_re,
            "ue_fp_rate": float(fp) / max(float(total_re), 1.0),
        }
        total_ue = int(counters.get(f"q2_rs_{rs}_multi_ue_total", 0))
        correct_ue = int(counters.get(f"q2_rs_{rs}_multi_ue_correct", 0))
        pred_regular = int(counters.get(f"q2_rs_{rs}_multi_ue_pred_regular", 0))
        pred_invalid = int(counters.get(f"q2_rs_{rs}_multi_ue_pred_invalid", 0))
        q2_multi_ue_by_rs_report[rs] = {
            "total": total_ue,
            "accuracy": correct_ue / max(1, total_ue),
            "pred_regular": pred_regular,
            "pred_invalid": pred_invalid,
            "pred_regular_rate": float(pred_regular) / max(float(total_ue), 1.0),
            "pred_invalid_rate": float(pred_invalid) / max(float(total_ue), 1.0),
        }
    raw_regular_remap_groups: Dict[str, Dict[str, Any]] = {}
    for group in ("unchanged", "remapped"):
        total = int(counters.get(f"event_raw_regular_{group}_total", 0))
        correct = int(counters.get(f"event_raw_regular_{group}_correct", 0))
        ue_fp = int(counters.get(f"event_raw_regular_{group}_ue_fp", 0))
        raw_regular_remap_groups[group] = {
            "total": total,
            "accuracy": correct / max(1, total),
            "ue_fp": ue_fp,
            "ue_fp_rate": float(ue_fp) / max(float(total), 1.0),
        }
    raw_regular_remap_by_rs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for rs in RS_LABELS:
        raw_regular_remap_by_rs[rs] = {}
        for group in ("unchanged", "remapped"):
            total = int(counters.get(f"event_raw_regular_by_rs_{rs}_{group}_total", 0))
            correct = int(counters.get(f"event_raw_regular_by_rs_{rs}_{group}_correct", 0))
            ue_fp = int(counters.get(f"event_raw_regular_by_rs_{rs}_{group}_ue_fp", 0))
            raw_regular_remap_by_rs[rs][group] = {
                "total": total,
                "accuracy": correct / max(1, total),
                "ue_fp": ue_fp,
                "ue_fp_rate": float(ue_fp) / max(float(total), 1.0),
            }
    raw_regular_remap_combo_report: list[Dict[str, Any]] = []
    for rs in RS_LABELS:
        for raw in REGULAR_EVENT_LABELS:
            for mapped in REGULAR_EVENT_LABELS:
                total = int(counters.get(f"event_raw_regular_combo_{rs}_{raw}_{mapped}_total", 0))
                if total <= 0:
                    continue
                correct = int(counters.get(f"event_raw_regular_combo_{rs}_{raw}_{mapped}_correct", 0))
                ue_fp = int(counters.get(f"event_raw_regular_combo_{rs}_{raw}_{mapped}_ue_fp", 0))
                raw_regular_remap_combo_report.append(
                    {
                        "rs": rs,
                        "event_code_raw": raw,
                        "event_label": mapped,
                        "remapped": raw != mapped,
                        "total": total,
                        "accuracy": correct / max(1, total),
                        "ue_fp": ue_fp,
                        "ue_fp_rate": float(ue_fp) / max(float(total), 1.0),
                    }
                )
    raw_regular_remap_combo_report.sort(key=lambda row: (-int(row["total"]), str(row["rs"]), str(row["event_code_raw"]), str(row["event_label"])))
    rs_change = _change_report(counters, "rs_change")
    re_to_ue = _change_report(counters, "re_to_ue")
    ue_to_re = _change_report(counters, "ue_to_re")
    false_fp = counters["rs_change_fp"] + counters["re_to_ue_fp"] + counters["ue_to_re_fp"]
    false_tn = counters["rs_change_tn"] + counters["re_to_ue_tn"] + counters["ue_to_re_tn"]
    public_counters = {key: value for key, value in counters.items() if "_transition_dir_cm_" not in key}
    return {
        "eval_mode": args.eval_mode,
        "image_ablation": str(getattr(args, "image_ablation", "none")),
        "goal_ablation": bool(getattr(args, "ablate_goal", False)),
        "script_correction": "none",
        "initial_memory": str(args.initial_memory_noise),
        "event_gate_uses_ground_truth": False,
        "q2_candidates_use_predicted_rs": True,
        "world_size": int(world_size),
        "route_count": route_count,
        "selected_case_count": selected_case_count,
        **public_counters,
        "rs_acc": rs_acc,
        "rs_first_gt_lock_baseline_acc": rs_first_gt_baseline,
        "rs_first_pred_lock_baseline_acc": rs_first_pred_baseline,
        "rs_visual_gain_over_first_gt_lock": rs_acc - rs_first_gt_baseline,
        "rs_visual_gain_over_first_pred_lock": rs_acc - rs_first_pred_baseline,
        "rs_pred_change_rate": counters["rs_pred_change_count"] / change_den,
        "rs_gt_change_rate": counters["rs_gt_change_count"] / change_den,
        "rs_gt_r3_rate": rs_gt_r3_total / frames,
        "rs_pred_r3_rate": rs_pred_r3_total / frames,
        "rs_r3_precision": rs_r3_tp / max(1, rs_pred_r3_total),
        "rs_change_precision": rs_change["precision"],
        "rs_change_recall": rs_change["recall"],
        "rs_change_f1": rs_change["f1"],
        "rs_false_transition_rate_when_gt_stable": rs_change["false_transition_rate_when_gt_stable"],
        "rs_change_invalid_rate": rs_change["invalid_rate"],
        "rs_locked_case_rate": counters["rs_locked_cases"] / max(1, counters["cases"]),
        "rs_locked_eq_first_gt_rate": counters["rs_locked_eq_first_gt_cases"] / max(1, counters["rs_locked_cases"]),
        "rs_confusion_report": rs_report,
        "rs_transition_direction_confusion": _sparse_transition_direction_report(counters, prefix="rs", labels=_RS_TRANSITION_LABELS),
        "event_acc_end_to_end": event_acc_q2,
        "joint_acc": counters["q2_joint_correct"] / q2_total,
        "event_acc_when_rs_wrong": counters["q2_event_correct_when_rs_wrong"] / max(1, counters["event_score_valid_frames"] - counters["q2_when_rs_correct"]),
        "event_acc_when_rs_correct": event_acc_rs_correct,
        "event_global_majority_baseline": event_global_majority_baseline,
        "event_global_majority_baseline_label": _GLOBAL_EVENT_MAJORITY_LABEL,
        "event_global_majority_baseline_token": _GLOBAL_EVENT_MAJORITY_TOKEN,
        "event_regular_baseline_given_gt_rs": event_regular_baseline_given_gt_rs,
        "event_regular_baseline_given_gt_rs_expected_full_data": REGULAR_ZERO_INFO_BASELINE_END_TO_END,
        "event_regular_baseline_end_to_end": event_regular_baseline_given_gt_rs,
        "event_regular_baseline_expected_full_data": REGULAR_ZERO_INFO_BASELINE_END_TO_END,
        "event_any_regular_rate_end_to_end": counters["q2_gt_re_total"] / q2_total,
        "event_majority_regular_baseline_end_to_end": event_regular_baseline_given_gt_rs,
        "event_oracle_majority_regular_baseline_end_to_end": event_regular_baseline_oracle_majority_q2,
        "event_regular_baseline_when_rs_correct": event_regular_baseline_rs_correct,
        "event_majority_regular_baseline_by_rs": regular_majority_by_rs,
        "event_static_regular_baseline_by_rs": regular_majority_static_by_rs,
        "event_visual_gain_over_global_majority_baseline": event_acc_q2 - event_global_majority_baseline,
        "event_visual_gain_over_regular_baseline": event_acc_q2 - event_global_majority_baseline,
        "event_gap_to_given_gt_rs_regular_baseline": event_acc_q2 - event_regular_baseline_given_gt_rs,
        "event_visual_gain_over_given_gt_rs_regular_baseline": event_acc_q2 - event_regular_baseline_given_gt_rs,
        "event_pred_re_rate": counters["q2_pred_re"] / q2_total,
        "event_pred_ue_rate": counters["q2_pred_ue"] / q2_total,
        "event_pred_invalid_rate": counters["q2_pred_invalid"] / q2_total,
        "event_acc_multi_candidate": counters["q2_event_correct_multi_candidate"] / q2_multi_total,
        "event_pred_re_rate_multi_candidate": counters["q2_pred_re_multi_candidate"] / q2_multi_total,
        "event_pred_ue_rate_multi_candidate": counters["q2_pred_ue_multi_candidate"] / q2_multi_total,
        "event_pred_invalid_rate_multi_candidate": counters["q2_pred_invalid_multi_candidate"] / q2_multi_total,
        "event_acc_single_re": counters["q2_single_re_correct"] / max(1, counters["q2_single_re_total"]),
        "event_acc_single_ue": counters["q2_single_ue_correct"] / max(1, counters["q2_single_ue_total"]),
        "event_acc_multi_re": counters["q2_multi_re_correct"] / max(1, counters["q2_multi_re_total"]),
        "event_acc_multi_ue": counters["q2_multi_ue_correct"] / max(1, counters["q2_multi_ue_total"]),
        "single_candidate_invalid_rate": counters["q2_single_candidate_invalid"] / max(1, counters["q2_single_re_total"] + counters["q2_single_ue_total"]),
        "ue_fp_on_multi_candidate_re_rate": counters["ue_fp_on_multi_candidate_re"] / max(1, counters["q2_multi_re_total"]),
        "dataset_candidate_mismatch_rate": counters["dataset_candidate_mismatch"] / frames,
        "dataset_candidate_mismatch_ue_rate": counters["dataset_candidate_mismatch_ue"] / max(1, counters["event_gt_ue_frames"]),
        "dataset_candidate_mismatch_re_rate": counters["dataset_candidate_mismatch_re"] / max(1, counters["event_gt_re_frames"]),
        "event_score_valid_rate": counters["event_score_valid_frames"] / frames,
        "event_unreachable_due_to_rs_rate": counters["event_unreachable_due_to_rs"] / q2_total,
        "q2_candidates_from_pred_rs_rate": counters["q2_candidates_from_pred_rs"] / frames,
        "q2_single_candidate_rate": counters["q2_single_candidate"] / frames,
        "q2_multi_candidate_rate": counters["q2_multi_candidate"] / frames,
        "q2_multi_candidate_scored_rate": counters["q2_multi_candidate_scored"] / frames,
        "q2_candidate_count_report": candidate_count_report,
        "q2_rs_candidate_count_report": rs_candidate_count_report,
        "q2_multi_re_by_rs_report": q2_multi_re_by_rs_report,
        "q2_multi_ue_by_rs_report": q2_multi_ue_by_rs_report,
        "event_confusion_report": event_report,
        "regular_internal_confusion_report": regular_report,
        "ue_fp_on_multi_candidate_re_by_rs": ue_fp_on_multi_candidate_re_by_rs,
        "event_raw_regular_remap_report": {
            "groups": raw_regular_remap_groups,
            "by_rs": raw_regular_remap_by_rs,
            "combos": raw_regular_remap_combo_report,
        },
        "ue_acc": counters["q2_ue_correct"] / max(1, counters["q2_ue_total"]),
        "re_acc": counters["q2_re_correct"] / max(1, counters["q2_re_total"]),
        "ue_pred_regular_rate": counters["q2_ue_pred_regular"] / max(1, counters["q2_ue_total"]),
        "ue_vs_re_tp": counters["ue_binary_tp"],
        "ue_vs_re_fp": counters["ue_binary_fp"],
        "ue_vs_re_fn": counters["ue_binary_fn"],
        "ue_vs_re_tn": counters["ue_binary_tn"],
        "ue_vs_re_precision": ue_binary["precision"],
        "ue_vs_re_recall": ue_binary["recall"],
        "ue_vs_re_f1": ue_binary["f1"],
        "ue_vs_re_tp_multi_candidate": counters["ue_binary_tp_multi_candidate"],
        "ue_vs_re_fp_multi_candidate": counters["ue_binary_fp_multi_candidate"],
        "ue_vs_re_fn_multi_candidate": counters["ue_binary_fn_multi_candidate"],
        "ue_vs_re_tn_multi_candidate": counters["ue_binary_tn_multi_candidate"],
        "ue_vs_re_precision_multi_candidate": ue_binary_multi["precision"],
        "ue_vs_re_recall_multi_candidate": ue_binary_multi["recall"],
        "ue_vs_re_f1_multi_candidate": ue_binary_multi["f1"],
        "re_to_ue_precision": re_to_ue["precision"],
        "re_to_ue_recall": re_to_ue["recall"],
        "re_to_ue_f1": re_to_ue["f1"],
        "re_to_ue_false_transition_rate_when_gt_stable": re_to_ue["false_transition_rate_when_gt_stable"],
        "re_to_ue_invalid_rate": re_to_ue["invalid_rate"],
        "ue_to_re_precision": ue_to_re["precision"],
        "ue_to_re_recall": ue_to_re["recall"],
        "ue_to_re_f1": ue_to_re["f1"],
        "ue_to_re_false_transition_rate_when_gt_stable": ue_to_re["false_transition_rate_when_gt_stable"],
        "ue_to_re_invalid_rate": ue_to_re["invalid_rate"],
        "false_transition_rate_when_gt_stable": float(false_fp) / max(float(false_fp + false_tn), 1.0),
        "event_transition_direction_confusion": _sparse_transition_direction_report(counters, prefix="event", labels=_EVENT_TRANSITION_LABELS),
        "q2_trigger_rate": counters["q2_triggered"] / frames,
        "transition_post_rs_acc": counters["transition_post_rs_correct"] / transition_post_frames,
        "transition_post_event_acc": counters["transition_post_event_correct"] / transition_post_q2,
        "rs_transition_hit_rate": counters["rs_transition_hit_cases"] / max(1, counters["rs_transition_cases"]),
        "rs_transition_already_at_target_rate": counters["rs_transition_already_at_target_hits"] / max(1, counters["rs_transition_hit_cases"]),
        "rs_transition_hit_offset_avg": counters["rs_transition_hit_offset_sum"] / max(1, counters["rs_transition_hit_cases"]),
        "rs_transition_abs_hit_offset_avg": counters["rs_transition_abs_hit_offset_sum"] / max(1, counters["rs_transition_hit_cases"]),
        "event_transition_hit_rate": counters["event_transition_hit_cases"] / max(1, counters["event_transition_cases"]),
        "event_transition_already_at_target_rate": counters["event_transition_already_at_target_hits"] / max(1, counters["event_transition_hit_cases"]),
        "event_transition_hit_offset_avg": counters["event_transition_hit_offset_sum"] / max(1, counters["event_transition_hit_cases"]),
        "event_transition_abs_hit_offset_avg": counters["event_transition_abs_hit_offset_sum"] / max(1, counters["event_transition_hit_cases"]),
        "event_transition_abnormal_hit_rate": counters["event_transition_abnormal_hit_cases"] / max(1, counters["event_transition_cases"]),
    }


def evaluate(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """执行自由生成评估。

    离散 memory 在 eval 中由学生输出维护：Q1 更新 RS，Q2 更新 EVENT。
    但 EGO_TO_GOAL_XY 是每帧连续量，所以即使没有 reset 也会在提问前刷新为当前帧。
    默认不做脚本纠偏；如果 Q1 RS 错，只跳过本帧 Q2，让后续帧继续暴露漂移。
    """

    rank, local_rank, world_size = setup_distributed()
    try:
        if world_size > 1 and torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ds = RouteSequenceDataset(
            pathlib.Path(args.index),
            max_routes=0,
            max_frames_per_route=int(args.max_frames_per_route),
        )
        if args.check:
            metrics = {
                "route_count": len(ds),
                "check_only": True,
                "world_size": int(world_size),
                "rank": int(rank),
            }
            return metrics if is_rank0(rank) else None

        _prepare_eval_outputs(args, rank=rank, world_size=world_size)
        bundle = load_eval_bundle(
            pathlib.Path(args.model_dir),
            pathlib.Path(args.adapter_dir) if args.adapter_dir else None,
            device,
            merge_lora=bool(args.merge_lora),
        )
        cases = _select_eval_cases(ds, args)
        if is_rank0(rank):
            print(
                f"[eval] mode={args.eval_mode} world_size={world_size} "
                f"selected_cases={len(cases)} output_dir={args.output_dir}"
            )

        counters = _new_counters()
        jsonl_fp: Optional[TextIO] = None
        jsonl_path: Optional[pathlib.Path] = None
        if args.output_jsonl:
            jsonl_path = _rank_jsonl_path(args.output_jsonl, rank, world_size)
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_fp = open(jsonl_path, "w", encoding="utf-8")
        try:
            for case_idx, case in enumerate(cases):
                # 多卡评测按 case 分片。full_route 模式下一条 case 是完整 route；
                # transition 模式下一条 case 是一个转折窗口。这样每个片段内部的
                # memory 仍按真实串行协议推进，不会被跨 rank 打断。
                if world_size > 1 and (case_idx % world_size) != rank:
                    continue
                _evaluate_case(bundle, case, args, counters, jsonl_fp)
        finally:
            if jsonl_fp is not None:
                jsonl_fp.close()

        counters = _sync_counters(counters, device)
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
            if args.output_jsonl and is_rank0(rank):
                _merge_rank_jsonl(args.output_jsonl, world_size)
            dist.barrier()

        if not is_rank0(rank):
            return None
        return _build_metrics(
            args=args,
            counters=counters,
            route_count=len(ds),
            selected_case_count=len(cases),
            world_size=world_size,
        )
    finally:
        cleanup_distributed()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SFT base token-choice adapter")
    p.add_argument("--index", type=str, default="checkpoints/sft_base_data/val_sequence_index.jsonl")
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--adapter-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--output-jsonl", type=str, default=None)
    p.add_argument("--output-summary", type=str, default=None)
    p.add_argument("--task", choices=sorted(_EVAL_TASK_TO_MODE), required=True)
    p.add_argument("--eval-mode", choices=["full_route", "rs_transition", "event_transition"], default="full_route")
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--sample-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
    p.add_argument("--transition-window", type=int, default=8)
    p.add_argument("--transition-tolerance", type=int, default=3)
    p.add_argument("--max-transition-cases", type=int, default=0)
    p.add_argument("--initial-memory-noise", choices=["unknown", "none", "rs", "event", "both", "random"], default="unknown")
    p.add_argument("--image-ablation", choices=["none", "black", "random"], default="none")
    p.add_argument("--ablate-goal", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=20260724)
    p.add_argument("--max-new-tokens-q1", type=int, default=32)
    p.add_argument("--max-new-tokens-q2", type=int, default=24)
    p.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.task:
        args.eval_mode = _EVAL_TASK_TO_MODE[args.task]
    _apply_eval_defaults(args)
    return args


def _default_eval_output_root(adapter_dir: Optional[str]) -> pathlib.Path:
    """根据 adapter 路径推断评估输出目录。"""

    if adapter_dir:
        path = pathlib.Path(adapter_dir)
        # 常规 adapter 目录是 checkpoints/sft_base_runs/latest/final；
        # 输出应该落到 latest/ 下，方便和训练日志放在一起。
        if path.name == "final":
            return path.parent
        return path
    return pathlib.Path("checkpoints/sft_base_runs/latest")


def _apply_eval_defaults(args: argparse.Namespace) -> None:
    """补齐日常评估默认值，让命令只需要改 GPU / 模型 / 任务。

    - RS/EVENT transition 默认抽 128 个 case。
    - full_route 默认随机抽 16 条 route。
    - 输出路径在 evaluate() 里根据 rank0 时间戳统一补齐，避免多卡进程各写各的。
    """

    if args.eval_mode == "full_route" and int(args.sample_routes) <= 0 and int(args.max_routes) <= 0:
        args.sample_routes = 16
    if args.eval_mode in {"rs_transition", "event_transition"} and int(args.max_transition_cases) <= 0:
        args.max_transition_cases = 128


def _metric_value(metrics: Dict[str, Any], key: str) -> str:
    """把指标格式化成人看的字符串，比例指标按百分比显示。"""

    value = metrics.get(key)
    if value is None:
        return "NA"
    if isinstance(value, float):
        if "_offset" in key:
            return f"{value:.2f} frames"
        if key.endswith("_per_100_frames"):
            return f"{value:.2f}"
        return f"{value * 100:.2f}%"
    return str(value)


def _summary_metric_rows(eval_mode: str) -> List[tuple[str, str]]:
    """按任务选择 summary.md 中优先展示的指标。"""

    common = [
        ("rs_acc", "全部评估帧上的 RS token 准确率"),
        ("rs_first_pred_lock_baseline_acc", "零视觉基线：模型首帧预测 RS 锁死不变的准确率"),
        ("rs_visual_gain_over_first_pred_lock", "RS 相对锁死首帧预测基线的净增益"),
        ("rs_pred_change_rate", "预测 RS 帧间变化率"),
        ("rs_gt_change_rate", "GT RS 帧间变化率"),
        ("rs_pred_r3_rate", "预测为 R3 的帧比例，用于监控高速/匝道 shortcut"),
        ("rs_gt_r3_rate", "GT 为 R3 的帧比例"),
        ("rs_r3_precision", "预测 R3 时真正为 R3 的比例，越低越说明 R3 假阳性严重"),
        ("rs_change_f1", "相邻帧 RS 变化检测 F1，必须和预测/GT 变化率一起读"),
        ("rs_false_transition_rate_when_gt_stable", "GT RS 未变化时预测假转折的比例，越低越好"),
        ("rs_locked_case_rate", "整段预测 RS 完全不变的 case 比例"),
        ("joint_acc", "真实串行主指标：RS 正确且 EVENT 正确的比例"),
        ("event_acc_end_to_end", "端到端 Q2 EVENT token 准确率，每帧都问 Q2"),
        ("event_global_majority_baseline", "端到端零信息下界：永远答全局最高频 regular token 的准确率"),
        ("event_visual_gain_over_global_majority_baseline", "EVENT 相对端到端全局多数类下界的净增益"),
        ("event_regular_baseline_given_gt_rs", "GT-RS oracle 参照：已知正确 RS 时永远答该 RS 多数 regular 的准确率"),
        ("event_gap_to_given_gt_rs_regular_baseline", "EVENT 相对 GT-RS oracle 参照的差值，通常不是端到端门槛"),
        ("event_pred_ue_rate", "Q2 输出 UE token 的比例"),
        ("event_acc_multi_candidate", "排除单候选送分题后的 EVENT 准确率"),
        ("event_pred_ue_rate_multi_candidate", "排除单候选送分题后的 UE 输出比例"),
        ("event_acc_single_re", "单候选 regular 帧 EVENT 准确率，正常应接近 0 分母"),
        ("event_acc_single_ue", "单候选 UE 帧 EVENT 准确率，检查纯 UE 正样本是否被学到"),
        ("event_acc_multi_re", "多候选 regular 硬负样本准确率，越低越说明 UE 假阳性严重"),
        ("event_acc_multi_ue", "多候选 UE 硬正样本准确率"),
        ("ue_fp_on_multi_candidate_re_rate", "多候选 regular 帧被误报成 UE 的比例，越低越好"),
        ("single_candidate_invalid_rate", "单候选题输出候选外 token 的比例"),
        ("ue_vs_re_f1", "由 EVENT 折叠出的 UE-vs-regular 二分类 F1"),
        ("ue_vs_re_f1_multi_candidate", "排除单候选送分题后的 UE-vs-regular 二分类 F1"),
        ("dataset_candidate_mismatch_rate", "GT EVENT 不在 dataset 自己候选表中的帧比例；这些帧不进 EVENT 评分"),
        ("dataset_candidate_mismatch_ue_rate", "UE 帧中 GT EVENT 不在 dataset 候选表中的比例"),
        ("re_to_ue_f1", "相邻帧 regular->UE 起始检测 F1，重点看异常起始漏检/延迟"),
        ("re_to_ue_false_transition_rate_when_gt_stable", "GT 未发生 regular->UE 时预测假异常起始的比例"),
        ("ue_to_re_f1", "相邻帧 UE->regular 结束检测 F1，衡量异常解除时机"),
        ("event_unreachable_due_to_rs_rate", "GT EVENT 在学生 RS 候选下不可达的比例"),
        ("q2_single_candidate_rate", "Q2 只有一个候选的帧比例"),
        ("q2_multi_candidate_rate", "Q2 有多个候选、真正需要判别的帧比例"),
        ("q2_candidates_from_pred_rs_rate", "按有效学生 RS 生成静态 Q2 候选的比例"),
        ("q2_trigger_rate", "进入 Q2 的比例；新协议应接近 100%"),
        ("ue_pred_regular_rate", "UE 帧进入 Q2 后仍被预测为 regular 子类的比例，越低越好"),
    ]
    if eval_mode == "rs_transition":
        return [
            ("rs_transition_hit_rate", "RS 转折 case 在容忍窗口内切到目标 RS 的比例"),
            ("rs_transition_already_at_target_rate", "RS 命中 case 中窗口左边界已经等于目标 RS 的比例，越高越像锁死撞上"),
            ("rs_transition_hit_offset_avg", "RS 命中帧相对标注转折帧的平均偏移，负数表示提前"),
            ("rs_transition_abs_hit_offset_avg", "RS 命中帧相对标注转折帧的平均绝对偏移"),
            ("transition_post_rs_acc", "转折点后所有帧的 RS 准确率"),
            *common,
        ]
    if eval_mode == "event_transition":
        return [
            ("event_transition_hit_rate", "EVENT 转换 case 在容忍窗口内切到目标 EVENT 的比例"),
            ("event_transition_abnormal_hit_rate", "EVENT 转换 case 在容忍窗口内由 Q2 折叠出的 UE/RE 是否切对"),
            ("event_transition_already_at_target_rate", "EVENT 命中 case 中窗口左边界已经等于目标 EVENT 的比例，越高越像锁死撞上"),
            ("event_transition_hit_offset_avg", "EVENT 命中帧相对标注转折帧的平均偏移，负数表示提前"),
            ("event_transition_abs_hit_offset_avg", "EVENT 命中帧相对标注转折帧的平均绝对偏移"),
            ("transition_post_event_acc", "转折点后、且进入 Q2 的 EVENT 准确率"),
            *common,
        ]
    return [
        ("rs_acc", "随机完整路线中所有帧的 RS token 准确率"),
        ("joint_acc", "随机完整路线真实串行主指标：RS 正确且 EVENT 正确"),
        ("event_acc_end_to_end", "随机完整路线中所有帧的端到端 EVENT token 准确率"),
        ("q2_trigger_rate", "完整路线中进入 Q2 的比例；新协议应接近 100%"),
        ("rs_pred_change_rate", "预测 RS 帧间变化率，需接近 GT 而不是只看 change F1"),
        ("rs_gt_change_rate", "GT RS 帧间变化率"),
        ("rs_change_f1", "完整路线相邻帧 RS 变化检测 F1，低切换率下可能虚高"),
        ("event_global_majority_baseline", "端到端零信息下界：永远答全局最高频 regular token"),
        ("event_visual_gain_over_global_majority_baseline", "EVENT 相对端到端全局多数类下界的净增益"),
        ("event_regular_baseline_given_gt_rs", "GT-RS oracle 参照：已知正确 RS 时的多数 regular 准确率"),
        ("re_to_ue_f1", "完整路线相邻帧 regular->UE 起始检测 F1"),
        ("false_transition_rate_when_gt_stable", "RS/regular->UE/UE->regular 合并后的 GT 稳定帧假转折比例，越低越好"),
        ("event_acc_multi_candidate", "排除单候选送分题后的 EVENT 准确率"),
        ("event_pred_ue_rate_multi_candidate", "排除单候选送分题后的 UE 输出比例"),
        ("event_acc_multi_re", "多候选 regular 硬负样本准确率"),
        ("event_acc_multi_ue", "多候选 UE 硬正样本准确率"),
        ("ue_fp_on_multi_candidate_re_rate", "多候选 regular 帧被误报成 UE 的比例"),
        ("ue_vs_re_f1_multi_candidate", "排除单候选送分题后的 UE-vs-regular F1"),
        ("ue_acc", "UE 帧进入 Q2 后的 EVENT 准确率"),
        ("re_acc", "regular 帧进入 Q2 后的 EVENT 准确率"),
    ]


def _write_summary(path: pathlib.Path, metrics: Dict[str, Any], args: argparse.Namespace) -> None:
    """写中文 Markdown 总结，方便不用打开 JSON 也能快速判断结果。"""

    rows = _summary_metric_rows(str(metrics.get("eval_mode", args.eval_mode)))
    lines = [
        "# SFT Base Eval Summary",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 任务：`{getattr(args, 'task', '')}` / `{metrics.get('eval_mode')}`",
        f"- Adapter：`{args.adapter_dir}`",
        f"- Base model：`{args.model_dir}`",
        f"- Index：`{args.index}`",
        f"- 多卡数：`{metrics.get('world_size')}`",
        f"- 图像消融：`{metrics.get('image_ablation')}`",
        f"- 目标坐标消融：`{metrics.get('goal_ablation')}`",
        f"- 首帧 memory：`{metrics.get('initial_memory')}`",
        f"- Q2 GT gate：`{metrics.get('event_gate_uses_ground_truth')}`",
        f"- Q2 候选按学生 RS：`{metrics.get('q2_candidates_use_predicted_rs')}`",
        f"- 选中 case 数：`{metrics.get('selected_case_count')}`",
        f"- 实际评估帧数：`{metrics.get('frames')}`",
        "",
        "## 保存文件",
        "",
        f"- 汇总指标 JSON：`{args.output_json}`",
        f"- 逐帧复盘 JSONL：`{args.output_jsonl}`",
        f"- 中文摘要：`{args.output_summary}`",
        "",
        "## 关键指标",
        "",
        "| 指标 | 数值 | 含义 |",
        "|---|---:|---|",
    ]
    for key, meaning in rows:
        lines.append(f"| `{key}` | {_metric_value(metrics, key)} | {meaning} |")
    lines.extend(
        [
            "",
            "## 读数提醒",
            "",
            "- `frames.jsonl` 是最重要的排查文件：每一行包含 GT/PRED RS、GT/PRED EVENT、原始生成文本和转折窗口信息。",
            "- 新协议里 Q2 候选由学生 RS 的静态全集决定，并把 R-E1..R-E5 regular 子类展开；逐帧 allowed_events 只用于 GT/审计。",
            "- `joint_acc` 是更真实的串行主指标；Q1 错但 Q2 蒙对的帧不会被算作 joint 成功。",
            "- `event_global_majority_baseline` 才是端到端零信息下界；`event_regular_baseline_given_gt_rs` 假设 GT RS 已知，只能当 oracle 参照，不应作为 `event_acc_end_to_end` 门槛。",
            "- `rs_change_f1` 必须和 `rs_pred_change_rate` / `rs_gt_change_rate` 一起读；预测几乎不切换时 F1 可能偶然虚高。",
            "- 黑图/随机图/no-goal 测试下若 `rs_visual_gain_over_first_pred_lock` 和 `event_visual_gain_over_global_majority_baseline` 仍接近原图，说明模型主要依赖 memory/语言捷径。",
            "- `event_pred_ue_rate` 长期接近 0，是 EVENT 坍缩到 regular 子类的直接信号；ABNORMAL 已从 Q1 协议删除。",
            "- `*_hit_offset_avg` 为负表示模型提前切换，为正表示滞后切换。",
            "- `script_correction` 应为 `none`，表示评测没有脚本纠偏。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    if metrics is None:
        return
    metrics = {
        **metrics,
        "output_dir": args.output_dir,
        "output_json": args.output_json,
        "output_jsonl": args.output_jsonl,
        "output_summary": args.output_summary,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_json:
        path = pathlib.Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    if args.output_summary:
        _write_summary(pathlib.Path(args.output_summary), metrics, args)
        print(f"[eval] saved metrics={args.output_json}")
        print(f"[eval] saved frames={args.output_jsonl}")
        print(f"[eval] saved summary={args.output_summary}")


if __name__ == "__main__":
    main()
