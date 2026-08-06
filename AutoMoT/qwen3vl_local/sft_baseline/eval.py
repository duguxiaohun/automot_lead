"""SFT baseline 自由生成评估入口。

当前 baseline 是单问协议：每帧只做一次多模态生成，输出两行：

```
ROAD: HIGHWAY|NON_HIGHWAY
EVENT: RE|UE
```

评估保留 closed-loop memory：下一帧 prompt 看到的 `PREVIOUS_ROAD` /
`PREVIOUS_EVENT` 来自学生上一帧输出；但不做任何脚本纠偏。
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import pathlib
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TextIO, Tuple

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
    """在 torch 初始化前应用 GPU_IDS pin 卡约定。"""

    pinned = ",".join(part.strip() for part in os.environ.get("GPU_IDS", "").split(",") if part.strip())
    if pinned:
        os.environ["CUDA_VISIBLE_DEVICES"] = pinned
        print(f"[gpu] using GPU_IDS={pinned}")


_maybe_apply_gpu_ids()

import torch
import torch.distributed as dist
try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_AVAILABLE = True
except Exception:
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False

from qwen3vl_local.sft_baseline import DATASET_VERSION  # noqa: E402
from qwen3vl_local.sft_baseline.labels import (  # noqa: E402
    EVENT_FAMILY_LABELS,
    ROAD_LABELS,
    event_family_from_label,
    road_label_from_rs,
)
from qwen3vl_local.sft_baseline.prompts import (  # noqa: E402
    Memory,
    build_q1_prompt,
    parse_q1_output,
    refresh_memory_goal,
    update_memory_after_q1,
    update_memory_after_q2,
)
from qwen3vl_local.sft_baseline.train import RouteSequenceDataset, _build_inputs, _load_images, _messages  # noqa: E402
from qwen3vl_local.sft_v3.train import _kv_start_state, _student_generate_kv  # noqa: E402


_VISION_SCOPE_CHOICES = {"off", "merger", "last4", "all"}
_EVAL_TASK_TO_MODE = {
    "full": "full_route",
    "full_route": "full_route",
    "road": "road_transition",
    "highway": "road_transition",
    "rs": "road_transition",
    "road_transition": "road_transition",
    "event": "event_transition",
    "event_transition": "event_transition",
}


@dataclass
class EvalBundle:
    """评估时传递给 KV helper 的轻量 bundle。"""

    model: Any
    processor: Any
    tokenizer: Any
    device: torch.device

    def unwrap(self) -> Any:
        """返回未包装模型。"""

        return self.model


@dataclass
class EvalCase:
    """一次评估片段。"""

    case_id: str
    route: Any
    route_index: int
    frames: List[Any]
    start_index: int
    transition_index: Optional[int] = None
    transition_kind: Optional[str] = None
    transition_source: Optional[str] = None
    transition_target: Optional[str] = None


def setup_distributed() -> tuple[int, int, int]:
    """初始化可选 torchrun 多卡评测环境。"""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("sft_baseline multi-GPU eval requires CUDA.")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """关闭 torch.distributed。"""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _resolve_model_path(path: pathlib.Path) -> pathlib.Path:
    """按 AutoMoT 运行目录口径规范化模型路径。"""

    path = pathlib.Path(path)
    if not path.is_absolute():
        path = _AUTOMOT_ROOT / path
    return path.resolve()


def _validate_adapter_config(adapter_dir: pathlib.Path, model_dir: pathlib.Path, *, prompt_memory_mode: str = "memory") -> Dict[str, Any]:
    """读取并校验 baseline adapter 自描述配置。"""

    cfg_path = pathlib.Path(adapter_dir) / "sft_baseline_adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"sft_baseline adapter config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("route") != "sft_baseline_highway_reue_joint":
        raise ValueError(f"adapter route mismatch: {cfg.get('route')!r}")
    if cfg.get("dataset_version") != DATASET_VERSION:
        raise ValueError(f"adapter dataset_version mismatch: expected {DATASET_VERSION}, got {cfg.get('dataset_version')!r}")
    saved_model_dir = cfg.get("base_model_dir")
    if not saved_model_dir:
        raise ValueError("adapter config missing base_model_dir")
    if _resolve_model_path(pathlib.Path(saved_model_dir)) != _resolve_model_path(model_dir):
        raise ValueError("adapter base_model_dir mismatch")
    scope = str(cfg.get("lora_vision_scope", ""))
    if scope not in _VISION_SCOPE_CHOICES:
        raise ValueError(f"adapter lora_vision_scope invalid: {scope!r}")
    saved_prompt_memory_mode = str(cfg.get("prompt_memory_mode", "memory"))
    if saved_prompt_memory_mode != str(prompt_memory_mode):
        raise ValueError(
            "adapter prompt_memory_mode mismatch: "
            f"checkpoint={saved_prompt_memory_mode!r}, eval={str(prompt_memory_mode)!r}. "
            "Use --prompt-memory-mode to match the training prompt."
        )
    return cfg


def load_eval_bundle(
    model_dir: pathlib.Path,
    adapter_dir: Optional[pathlib.Path],
    device: torch.device,
    *,
    merge_lora: bool,
    prompt_memory_mode: str = "memory",
) -> EvalBundle:
    """加载 base Qwen 和可选 LoRA adapter。"""

    from transformers import AutoProcessor

    adapter_cfg: Optional[Dict[str, Any]] = None
    if adapter_dir is not None:
        adapter_cfg = _validate_adapter_config(adapter_dir, model_dir, prompt_memory_mode=prompt_memory_mode)
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

        cfg = adapter_cfg or _validate_adapter_config(adapter_dir, model_dir, prompt_memory_mode=prompt_memory_mode)
        print(f"[adapter] validated sft_baseline adapter scope={cfg.get('lora_vision_scope')}")
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        if merge_lora and hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    return EvalBundle(model=model, processor=processor, tokenizer=processor.tokenizer, device=device)


def _q_messages(images: List[Any], prompt: str) -> List[Dict[str, Any]]:
    """复用 train._messages 的 system/image/user 结构，但不带 assistant target。"""

    return _messages(images, prompt, "", None, None)[:2]


def _generate_once(bundle: EvalBundle, images: List[Any], prompt: str, max_new_tokens: int) -> str:
    """单问 fresh prefill + decode。"""

    with torch.inference_mode():
        state = _kv_start_state(bundle, _q_messages(images, prompt))
        text, _, _ = _student_generate_kv(bundle, state, max_new_tokens)
    return text


def _score_q1_target(bundle: EvalBundle, images: List[Any], prompt: str, target: str, max_length: int) -> Optional[Tuple[float, float]]:
    """teacher-forced 计算一个 ROAD/EVENT 组合的值 token 平均 log-prob。

    这个分数不用于训练，只用于阈值/PR 诊断：它回答“在同一个 prompt 下，
    模型更愿意补哪个 ROAD/EVENT 值”。相比只看自由生成文本，分数可以通过
    `--road-logit-bias` / `--event-logit-bias` 做闭环阈值扫描。
    """

    packed = _build_inputs(
        bundle,
        images=images,
        q1_prompt=prompt,
        q1_target=target,
        q1_loss_weights={"road": 1.0, "event": 1.0},
        q2_prompt=None,
        q2_target=None,
        q2_loss_weights=None,
        max_length=int(max_length),
    )
    if packed is None:
        return None
    kwargs: Dict[str, Any] = {
        "input_ids": packed["input_ids"].unsqueeze(0).to(bundle.device),
        "attention_mask": packed["attention_mask"].unsqueeze(0).to(bundle.device),
    }
    labels = packed["labels"].unsqueeze(0).to(bundle.device)
    weights = packed["loss_weights"].unsqueeze(0).to(bundle.device)
    comp_ids = packed["loss_component_ids"].unsqueeze(0).to(bundle.device)
    for key, value in packed["vision"].items():
        kwargs[key] = value.to(bundle.device) if isinstance(value, torch.Tensor) else value
    with torch.inference_mode():
        out = bundle.model(**kwargs, use_cache=False, return_dict=True)
        log_probs = torch.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
        shift_labels = labels[:, 1:]
        shift_weights = weights[:, 1:]
        shift_comp_ids = comp_ids[:, 1:]
        token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        road_mask = shift_weights.gt(0) & shift_comp_ids.eq(1)
        event_mask = shift_weights.gt(0) & shift_comp_ids.eq(2)
        if not bool(road_mask.any() and event_mask.any()):
            return None
        road_score = float(token_logp[road_mask].mean().item())
        event_score = float(token_logp[event_mask].mean().item())
    return road_score, event_score


def _score_q1_options(bundle: EvalBundle, images: List[Any], prompt: str, args: argparse.Namespace) -> Dict[str, Any]:
    """对四个 ROAD/EVENT 二分类组合打分并返回带 bias 的预测。"""

    joint_scores: Dict[Tuple[str, str], float] = {}
    for road in ROAD_LABELS:
        for event in EVENT_FAMILY_LABELS:
            target = f"ROAD: {road}\nEVENT: {event}"
            scores = _score_q1_target(bundle, images, prompt, target, int(args.max_length))
            if scores is None:
                joint_scores[(road, event)] = float("-inf")
                continue
            road_score, event_score = scores
            joint_scores[(road, event)] = road_score + event_score
    highway_score = max(joint_scores[("HIGHWAY", event)] for event in EVENT_FAMILY_LABELS)
    non_highway_score = max(joint_scores[("NON_HIGHWAY", event)] for event in EVENT_FAMILY_LABELS)
    ue_score = max(joint_scores[(road, "UE")] for road in ROAD_LABELS)
    re_score = max(joint_scores[(road, "RE")] for road in ROAD_LABELS)
    road_delta = highway_score - non_highway_score
    event_delta = ue_score - re_score
    pred_road = "HIGHWAY" if road_delta + float(args.road_logit_bias) >= 0.0 else "NON_HIGHWAY"
    pred_event = "UE" if event_delta + float(args.event_logit_bias) >= 0.0 else "RE"
    return {
        "pred_road": pred_road,
        "pred_event": pred_event,
        "road_score_delta": road_delta,
        "event_score_delta": event_delta,
        "road_logit_bias": float(args.road_logit_bias),
        "event_logit_bias": float(args.event_logit_bias),
        "joint_scores": {f"{road}/{event}": score for (road, event), score in joint_scores.items()},
        "raw_text": f"ROAD: {pred_road}\nEVENT: {pred_event}",
    }


def _route_name(route: Any) -> str:
    """生成稳定 route 名称。"""

    return str(getattr(route, "route_id", "") or "route")


def _frame_road(frame: Any) -> str:
    """返回当前帧 ROAD 二分类 GT。"""

    return road_label_from_rs(str(frame.rs_label))


def _frame_event(frame: Any) -> str:
    """返回当前帧 EVENT family GT。"""

    return event_family_from_label(str(frame.event_label))


def _select_full_route_cases(ds: RouteSequenceDataset, args: argparse.Namespace) -> List[EvalCase]:
    """选择完整 route case。"""

    items = list(enumerate(ds.rows))
    if int(args.sample_routes) > 0:
        rng = random.Random(int(args.seed))
        items = rng.sample(items, min(int(args.sample_routes), len(items)))
    elif int(args.max_routes) > 0:
        items = items[: int(args.max_routes)]
    return [
        EvalCase(case_id=f"route:{idx}:{_route_name(route)}", route=route, route_index=idx, frames=list(route.frames), start_index=0)
        for idx, route in items
    ]


def _select_transition_cases(ds: RouteSequenceDataset, args: argparse.Namespace) -> List[EvalCase]:
    """选择 ROAD 或 EVENT family 转折窗口。"""

    wanted = "road" if args.eval_mode == "road_transition" else "event"
    window = max(1, int(args.transition_window))
    cases: List[EvalCase] = []
    items = list(enumerate(ds.rows))
    if int(args.max_routes) > 0:
        items = items[: int(args.max_routes)]
    for route_index, route in items:
        frames = list(route.frames)
        for idx in range(1, len(frames)):
            prev = _frame_road(frames[idx - 1]) if wanted == "road" else _frame_event(frames[idx - 1])
            cur = _frame_road(frames[idx]) if wanted == "road" else _frame_event(frames[idx])
            if prev == cur:
                continue
            start = max(0, idx - window)
            end = min(len(frames), idx + window + 1)
            cases.append(
                EvalCase(
                    case_id=f"transition:{wanted}:{route_index}:{_route_name(route)}:{idx}",
                    route=route,
                    route_index=route_index,
                    frames=frames[start:end],
                    start_index=start,
                    transition_index=idx,
                    transition_kind=wanted,
                    transition_source=prev,
                    transition_target=cur,
                )
            )
    if int(args.max_transition_cases) > 0 and len(cases) > int(args.max_transition_cases):
        rng = random.Random(int(args.seed))
        cases = rng.sample(cases, int(args.max_transition_cases))
    return cases


def _select_eval_cases(ds: RouteSequenceDataset, args: argparse.Namespace) -> List[EvalCase]:
    """按 eval_mode 生成 case 列表。"""

    if args.eval_mode in {"road_transition", "event_transition"}:
        return _select_transition_cases(ds, args)
    return _select_full_route_cases(ds, args)


def _initial_memory(frame: Any, args: argparse.Namespace) -> Memory:
    """创建评估片段首帧 memory，默认 UNKNOWN 冷启动。"""

    if str(args.initial_memory_noise) == "unknown":
        return refresh_memory_goal(Memory(rs_label="UNKNOWN", event_label="UNKNOWN"), _eval_goal_xy(frame, args))
    return refresh_memory_goal(Memory(rs_label=frame.rs_label, event_label=frame.event_label), _eval_goal_xy(frame, args))


def _prompt_memory_for_mode(memory: Memory, args: argparse.Namespace) -> Memory:
    """按 eval 开关决定 prompt 是否展示离散 memory。"""

    out = memory.copy()
    mode = str(getattr(args, "prompt_memory_mode", "memory")).lower()
    if mode == "memory":
        return out
    if mode == "unknown":
        out.rs_label = "UNKNOWN"
        out.event_label = "UNKNOWN"
        return out
    if mode == "hidden":
        out.rs_label = "UNKNOWN"
        out.event_label = "UNKNOWN"
        out.hide_priors = True
        return out
    raise ValueError(f"unknown prompt_memory_mode: {mode}")


def _eval_goal_xy(frame: Any, args: argparse.Namespace) -> Optional[List[float]]:
    """按 eval 消融设置返回学生可见的导航目标。"""

    if bool(getattr(args, "ablate_goal", False)):
        return None
    return frame.ego_to_goal_xy


def _apply_image_ablation(images: List[Any], args: argparse.Namespace, case: EvalCase, frame: Any) -> List[Any]:
    """把输入 RGB history 替换成黑图或随机噪声图。"""

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
            generator.manual_seed(int(args.seed) + case.route_index * 100003 + int(frame.frame_id) * 9176 + idx * 137)
            payload = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8, generator=generator).numpy().tobytes()
            out.append(Image.frombytes("RGB", (width, height), payload))
            continue
        raise ValueError(f"unknown image_ablation mode: {mode}")
    return out


def _new_counters() -> Dict[str, int]:
    """创建评估计数器。"""

    counters: Dict[str, int] = {
        "cases": 0,
        "frames": 0,
        "road_correct": 0,
        "event_correct": 0,
        "joint_correct": 0,
        "road_invalid": 0,
        "event_invalid": 0,
        "road_change_tp": 0,
        "road_change_fp": 0,
        "road_change_fn": 0,
        "road_change_tn": 0,
        "event_change_tp": 0,
        "event_change_fp": 0,
        "event_change_fn": 0,
        "event_change_tn": 0,
        "transition_cases": 0,
        "transition_hit_cases": 0,
        "transition_hit_offset_sum": 0,
        "transition_abs_hit_offset_sum": 0,
        "transition_post_frames": 0,
        "transition_post_correct": 0,
    }
    for gt in (*ROAD_LABELS, "INVALID"):
        for pred in (*ROAD_LABELS, "INVALID"):
            counters[f"road_cm_{gt}_{pred}"] = 0
    for gt in (*EVENT_FAMILY_LABELS, "INVALID"):
        for pred in (*EVENT_FAMILY_LABELS, "INVALID"):
            counters[f"event_cm_{gt}_{pred}"] = 0
    return counters


def _update_binary_change(counters: Dict[str, int], prefix: str, gt_prev: str, gt_cur: str, pred_prev: Optional[str], pred_cur: Optional[str]) -> None:
    """更新相邻帧变化检测计数。"""

    gt_changed = gt_prev != gt_cur
    if pred_prev not in {"HIGHWAY", "NON_HIGHWAY", "RE", "UE"} or pred_cur not in {"HIGHWAY", "NON_HIGHWAY", "RE", "UE"}:
        pred_changed = False
    else:
        pred_changed = pred_prev != pred_cur
    if gt_changed and pred_changed:
        counters[f"{prefix}_tp"] += 1
    elif (not gt_changed) and pred_changed:
        counters[f"{prefix}_fp"] += 1
    elif gt_changed and not pred_changed:
        counters[f"{prefix}_fn"] += 1
    else:
        counters[f"{prefix}_tn"] += 1


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """由 TP/FP/FN 计算 P/R/F1。"""

    precision = float(tp) / max(float(tp + fp), 1.0)
    recall = float(tp) / max(float(tp + fn), 1.0)
    f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _binary_label_report(counters: Dict[str, int], prefix: str, positive: str, negative: str) -> Dict[str, Any]:
    """生成二分类混淆矩阵与正类 P/R/F1。"""

    tp = counters[f"{prefix}_cm_{positive}_{positive}"]
    fp = counters[f"{prefix}_cm_{negative}_{positive}"]
    fn = counters[f"{prefix}_cm_{positive}_{negative}"] + counters[f"{prefix}_cm_{positive}_INVALID"]
    tn = counters[f"{prefix}_cm_{negative}_{negative}"]
    return {
        "positive": positive,
        "negative": negative,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        **_prf(tp, fp, fn),
        "confusion_matrix": {
            positive: {
                positive: counters[f"{prefix}_cm_{positive}_{positive}"],
                negative: counters[f"{prefix}_cm_{positive}_{negative}"],
                "INVALID": counters[f"{prefix}_cm_{positive}_INVALID"],
            },
            negative: {
                positive: counters[f"{prefix}_cm_{negative}_{positive}"],
                negative: counters[f"{prefix}_cm_{negative}_{negative}"],
                "INVALID": counters[f"{prefix}_cm_{negative}_INVALID"],
            },
        },
    }


def _write_record(fp: Optional[TextIO], rec: Dict[str, Any]) -> None:
    """写逐帧 JSONL。"""

    if fp is not None:
        fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _evaluate_case(bundle: EvalBundle, case: EvalCase, args: argparse.Namespace, counters: Dict[str, int], jsonl_fp: Optional[TextIO]) -> None:
    """执行单个评估片段。"""

    memory: Optional[Memory] = None
    records: List[Dict[str, Any]] = []
    counters["cases"] += 1
    for pos, frame in enumerate(case.frames):
        abs_index = case.start_index + pos
        if memory is None:
            memory = _initial_memory(frame, args)
        else:
            memory = refresh_memory_goal(memory, _eval_goal_xy(frame, args))

        images = _apply_image_ablation(_load_images(frame.history_rgb_paths), args, case, frame)
        prompt_memory = _prompt_memory_for_mode(memory, args)
        prompt = build_q1_prompt(prompt_memory, choice_seed=f"joint::{frame.frame_id}")
        score_payload: Optional[Dict[str, Any]] = None
        if str(args.prediction_mode) == "score":
            score_payload = _score_q1_options(bundle, images, prompt, args)
            text = str(score_payload["raw_text"])
            pred_road = score_payload["pred_road"]
            pred_event = score_payload["pred_event"]
        else:
            text = _generate_once(bundle, images, prompt, int(args.max_new_tokens))
            parsed = parse_q1_output(text)
            pred_road = parsed.get("road")
            pred_event = parsed.get("event")
        gt_road = _frame_road(frame)
        gt_event = _frame_event(frame)
        road_ok = pred_road == gt_road
        event_ok = pred_event == gt_event

        counters["frames"] += 1
        counters["road_correct"] += int(road_ok)
        counters["event_correct"] += int(event_ok)
        counters["joint_correct"] += int(road_ok and event_ok)
        counters["road_invalid"] += int(pred_road not in ROAD_LABELS)
        counters["event_invalid"] += int(pred_event not in EVENT_FAMILY_LABELS)
        road_bucket = pred_road if pred_road in ROAD_LABELS else "INVALID"
        event_bucket = pred_event if pred_event in EVENT_FAMILY_LABELS else "INVALID"
        counters[f"road_cm_{gt_road}_{road_bucket}"] += 1
        counters[f"event_cm_{gt_event}_{event_bucket}"] += 1
        if case.transition_index is not None and abs_index >= case.transition_index:
            counters["transition_post_frames"] += 1
            key = "road" if case.transition_kind == "road" else "event"
            counters["transition_post_correct"] += int((pred_road == gt_road) if key == "road" else (pred_event == gt_event))

        memory = update_memory_after_q1(memory, student_rs_label=pred_road)
        memory = update_memory_after_q2(memory, student_event_label=pred_event)
        rec = {
            "case_id": case.case_id,
            "scenario": getattr(case.route, "scenario", None),
            "route_id": getattr(case.route, "route_id", None),
            "route_index": case.route_index,
            "frame_id": frame.frame_id,
            "frame_index": abs_index,
            "transition_kind": case.transition_kind,
            "transition_index": case.transition_index,
            "post_transition": case.transition_index is not None and abs_index >= case.transition_index,
            "gt_rs": frame.rs_label,
            "gt_road": gt_road,
            "pred_road": pred_road,
            "road_ok": road_ok,
            "gt_event_raw": frame.event_label,
            "gt_event": gt_event,
            "pred_event": pred_event,
            "event_ok": event_ok,
            "joint_ok": road_ok and event_ok,
            "raw_text": text,
            "prediction_mode": str(args.prediction_mode),
            "score": score_payload,
            "prompt": prompt if bool(args.save_prompts) else None,
        }
        records.append(rec)
        _write_record(jsonl_fp, rec)

    for prev, cur in zip(records, records[1:]):
        _update_binary_change(counters, "road_change", prev["gt_road"], cur["gt_road"], prev.get("pred_road"), cur.get("pred_road"))
        _update_binary_change(counters, "event_change", prev["gt_event"], cur["gt_event"], prev.get("pred_event"), cur.get("pred_event"))
    if case.transition_index is not None:
        counters["transition_cases"] += 1
        target = case.transition_target
        key = "pred_road" if case.transition_kind == "road" else "pred_event"
        hit = None
        for rec in records:
            if rec["frame_index"] >= int(case.transition_index) and rec.get(key) == target:
                hit = int(rec["frame_index"])
                break
        if hit is not None:
            counters["transition_hit_cases"] += 1
            offset = hit - int(case.transition_index)
            counters["transition_hit_offset_sum"] += offset
            counters["transition_abs_hit_offset_sum"] += abs(offset)


def _sync_counters(counters: Dict[str, int], device: torch.device) -> Dict[str, int]:
    """跨 rank 汇总整数 counters。"""

    if not dist.is_available() or not dist.is_initialized():
        return counters
    keys = list(counters)
    tensor = torch.tensor([int(counters[k]) for k in keys], dtype=torch.long, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: int(value) for key, value in zip(keys, tensor.cpu().tolist())}


def _rank_jsonl_path(output_jsonl: str, rank: int, world_size: int) -> pathlib.Path:
    """多卡时每个 rank 先写自己的 jsonl 分片。"""

    out_path = pathlib.Path(output_jsonl)
    return out_path if world_size <= 1 else out_path.with_suffix(out_path.suffix + f".rank{rank}")


def _merge_rank_jsonl(output_jsonl: str, world_size: int) -> None:
    """rank0 合并各 rank 的 jsonl 分片。"""

    if world_size <= 1:
        return
    out_path = pathlib.Path(output_jsonl)
    with open(out_path, "w", encoding="utf-8") as dst:
        for rank in range(world_size):
            shard = _rank_jsonl_path(output_jsonl, rank, world_size)
            if not shard.exists():
                continue
            with open(shard, "r", encoding="utf-8") as src:
                for line in src:
                    dst.write(line)
            shard.unlink(missing_ok=True)


def _dist_broadcast_text(value: Optional[str], *, rank: int, world_size: int) -> Optional[str]:
    """把 rank0 生成的字符串广播到所有 rank。"""

    if world_size <= 1:
        return value
    values: List[Optional[str]] = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _default_eval_output_root(adapter_dir: Optional[str]) -> pathlib.Path:
    """根据 adapter 路径推断评估输出目录。"""

    if adapter_dir:
        path = pathlib.Path(adapter_dir)
        return path.parent if path.name == "final" else path
    return pathlib.Path("checkpoints/sft_baseline_runs/latest")


def _prepare_eval_outputs(args: argparse.Namespace, *, rank: int, world_size: int) -> None:
    """补齐本次评测的输出路径。"""

    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir)
    elif not args.output_json and not args.output_jsonl:
        if rank == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_text: Optional[str] = str(_default_eval_output_root(args.adapter_dir) / "eval_results" / args.eval_mode / timestamp)
        else:
            output_text = None
        output_dir = pathlib.Path(str(_dist_broadcast_text(output_text, rank=rank, world_size=world_size)))
    elif args.output_json:
        output_dir = pathlib.Path(args.output_json).parent
    else:
        output_dir = pathlib.Path(args.output_jsonl).parent
    args.output_dir = str(output_dir)
    args.output_json = args.output_json or str(output_dir / "metrics.json")
    if bool(args.write_frames):
        args.output_jsonl = args.output_jsonl or str(output_dir / "frames.jsonl")
    else:
        args.output_jsonl = None
    args.output_summary = args.output_summary or str(output_dir / "summary.md")
    args.output_html = args.output_html or str(output_dir / "report.html")
    args.output_tb = args.output_tb or (str(output_dir / "tb") if bool(args.write_tb) else None)
    pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl:
        pathlib.Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output_html).parent.mkdir(parents=True, exist_ok=True)
    if args.output_tb:
        pathlib.Path(args.output_tb).mkdir(parents=True, exist_ok=True)


def _build_metrics(args: argparse.Namespace, counters: Dict[str, int], *, route_count: int, selected_case_count: int, world_size: int) -> Dict[str, Any]:
    """由 counters 计算最终 metrics。"""

    frames = max(1, int(counters["frames"]))
    road_report = _binary_label_report(counters, "road", positive="HIGHWAY", negative="NON_HIGHWAY")
    event_report = _binary_label_report(counters, "event", positive="UE", negative="RE")
    road_change = _prf(counters["road_change_tp"], counters["road_change_fp"], counters["road_change_fn"])
    event_change = _prf(counters["event_change_tp"], counters["event_change_fp"], counters["event_change_fn"])
    return {
        "eval_mode": args.eval_mode,
        "dataset_version": DATASET_VERSION,
        "world_size": int(world_size),
        "route_count": int(route_count),
        "selected_case_count": int(selected_case_count),
        "image_ablation": str(args.image_ablation),
        "goal_ablation": bool(args.ablate_goal),
        "initial_memory": str(args.initial_memory_noise),
        "prompt_memory_mode": str(args.prompt_memory_mode),
        "prediction_mode": str(args.prediction_mode),
        "road_logit_bias": float(args.road_logit_bias),
        "event_logit_bias": float(args.event_logit_bias),
        "script_correction": "none",
        **counters,
        "road_acc": counters["road_correct"] / frames,
        "event_acc": counters["event_correct"] / frames,
        "joint_acc": counters["joint_correct"] / frames,
        "road_invalid_rate": counters["road_invalid"] / frames,
        "event_invalid_rate": counters["event_invalid"] / frames,
        "highway_precision": road_report["precision"],
        "highway_recall": road_report["recall"],
        "highway_f1": road_report["f1"],
        "ue_precision": event_report["precision"],
        "ue_recall": event_report["recall"],
        "ue_f1": event_report["f1"],
        "road_report": road_report,
        "event_report": event_report,
        "road_confusion_report": road_report,
        "event_confusion_report": event_report,
        "road_change_precision": road_change["precision"],
        "road_change_recall": road_change["recall"],
        "road_change_f1": road_change["f1"],
        "event_change_precision": event_change["precision"],
        "event_change_recall": event_change["recall"],
        "event_change_f1": event_change["f1"],
        "transition_hit_rate": counters["transition_hit_cases"] / max(1, counters["transition_cases"]),
        "transition_hit_offset_avg": counters["transition_hit_offset_sum"] / max(1, counters["transition_hit_cases"]),
        "transition_abs_hit_offset_avg": counters["transition_abs_hit_offset_sum"] / max(1, counters["transition_hit_cases"]),
        "transition_post_acc": counters["transition_post_correct"] / max(1, counters["transition_post_frames"]),
    }


def evaluate(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """执行自由生成评估。"""

    rank, local_rank, world_size = setup_distributed()
    try:
        device = torch.device(f"cuda:{local_rank}" if world_size > 1 and torch.cuda.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        ds = RouteSequenceDataset(pathlib.Path(args.index), max_routes=0, max_frames_per_route=int(args.max_frames_per_route))
        if args.check:
            return {"route_count": len(ds), "check_only": True, "world_size": int(world_size), "rank": int(rank)} if rank == 0 else None
        _prepare_eval_outputs(args, rank=rank, world_size=world_size)
        bundle = load_eval_bundle(
            pathlib.Path(args.model_dir),
            pathlib.Path(args.adapter_dir) if args.adapter_dir else None,
            device,
            merge_lora=bool(args.merge_lora),
            prompt_memory_mode=str(args.prompt_memory_mode),
        )
        cases = _select_eval_cases(ds, args)
        if rank == 0:
            print(f"[eval] mode={args.eval_mode} world_size={world_size} selected_cases={len(cases)} output_dir={args.output_dir}")
        counters = _new_counters()
        jsonl_fp: Optional[TextIO] = None
        if args.output_jsonl:
            jsonl_path = _rank_jsonl_path(args.output_jsonl, rank, world_size)
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_fp = open(jsonl_path, "w", encoding="utf-8")
        try:
            for case_idx, case in enumerate(cases):
                if world_size > 1 and (case_idx % world_size) != rank:
                    continue
                _evaluate_case(bundle, case, args, counters, jsonl_fp)
        finally:
            if jsonl_fp is not None:
                jsonl_fp.close()
        counters = _sync_counters(counters, device)
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
            if args.output_jsonl and rank == 0:
                _merge_rank_jsonl(args.output_jsonl, world_size)
            dist.barrier()
        if rank != 0:
            return None
        return _build_metrics(args, counters, route_count=len(ds), selected_case_count=len(cases), world_size=world_size)
    finally:
        cleanup_distributed()


def _apply_eval_defaults(args: argparse.Namespace) -> None:
    """补齐日常评估默认值。"""

    if args.eval_mode == "full_route" and int(args.sample_routes) <= 0 and int(args.max_routes) <= 0:
        args.sample_routes = 16
    if args.eval_mode in {"road_transition", "event_transition"} and int(args.max_transition_cases) <= 0:
        args.max_transition_cases = 128


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""

    p = argparse.ArgumentParser(description="Evaluate SFT baseline highway/RE-UE adapter")
    p.add_argument("--index", type=str, default="checkpoints/sft_baseline_data/val_sequence_index.jsonl")
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--adapter-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--output-jsonl", type=str, default=None)
    p.add_argument("--output-summary", type=str, default=None)
    p.add_argument("--output-html", type=str, default=None)
    p.add_argument("--output-tb", type=str, default=None)
    p.add_argument("--write-frames", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--write-tb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--task", choices=sorted(_EVAL_TASK_TO_MODE), required=True)
    p.add_argument("--eval-mode", choices=["full_route", "road_transition", "event_transition"], default="full_route")
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--sample-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
    p.add_argument("--transition-window", type=int, default=8)
    p.add_argument("--max-transition-cases", type=int, default=0)
    p.add_argument("--initial-memory-noise", choices=["unknown", "none"], default="unknown")
    p.add_argument("--prompt-memory-mode", choices=["memory", "hidden", "unknown"], default="memory")
    p.add_argument("--prediction-mode", choices=["generate", "score"], default="generate")
    p.add_argument("--road-logit-bias", type=float, default=0.0)
    p.add_argument("--event-logit-bias", type=float, default=0.0)
    p.add_argument("--image-ablation", choices=["none", "black", "random"], default="none")
    p.add_argument("--ablate-goal", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save-prompts", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=20260724)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.task:
        args.eval_mode = _EVAL_TASK_TO_MODE[args.task]
    _apply_eval_defaults(args)
    return args


def _write_summary(path: pathlib.Path, metrics: Dict[str, Any], args: argparse.Namespace) -> None:
    """写中文 Markdown 总结。"""

    lines = [
        "# SFT Baseline Eval Summary",
        "",
        f"- 任务：`{args.task}` / `{metrics.get('eval_mode')}`",
        f"- Adapter：`{args.adapter_dir}`",
        f"- Index：`{args.index}`",
        f"- 评估帧数：`{metrics.get('frames')}`",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| road_acc | {metrics.get('road_acc', 0.0) * 100:.2f}% |",
        f"| highway_f1 | {metrics.get('highway_f1', 0.0) * 100:.2f}% |",
        f"| event_acc | {metrics.get('event_acc', 0.0) * 100:.2f}% |",
        f"| ue_f1 | {metrics.get('ue_f1', 0.0) * 100:.2f}% |",
        f"| joint_acc | {metrics.get('joint_acc', 0.0) * 100:.2f}% |",
        f"| road_change_f1 | {metrics.get('road_change_f1', 0.0) * 100:.2f}% |",
        f"| event_change_f1 | {metrics.get('event_change_f1', 0.0) * 100:.2f}% |",
        "",
        "逐帧输出见 `frames.jsonl`，其中包含 GT/PRED ROAD、GT/PRED EVENT 和原始生成文本。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _confusion_matrix_markdown(title: str, report: Dict[str, Any]) -> str:
    """把二分类 confusion matrix 渲染成 TensorBoard text 可读 Markdown。"""

    matrix = report.get("confusion_matrix") or {}
    labels = [str(report.get("positive", "POS")), str(report.get("negative", "NEG"))]
    pred_labels = [*labels, "INVALID"]
    lines = [
        f"### {title}",
        "",
        "| GT \\ Pred | " + " | ".join(pred_labels) + " |",
        "|---|" + "|".join("---:" for _ in pred_labels) + "|",
    ]
    for gt in labels:
        row = matrix.get(gt, {})
        values = [str(int(row.get(pred, 0) or 0)) for pred in pred_labels]
        lines.append(f"| {gt} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_tensorboard(path: pathlib.Path, metrics: Dict[str, Any], args: argparse.Namespace) -> None:
    """写一个轻量 eval TensorBoard：核心 scalar + 两个二分类混淆矩阵文本。"""

    if not _TB_AVAILABLE:
        print("[tb][warn] tensorboard is not available; skip eval TensorBoard")
        return
    path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(path))
    try:
        scalar_keys = [
            "road_acc",
            "event_acc",
            "joint_acc",
            "highway_precision",
            "highway_recall",
            "highway_f1",
            "ue_precision",
            "ue_recall",
            "ue_f1",
            "road_invalid_rate",
            "event_invalid_rate",
            "road_change_precision",
            "road_change_recall",
            "road_change_f1",
            "event_change_precision",
            "event_change_recall",
            "event_change_f1",
            "transition_hit_rate",
            "transition_post_acc",
        ]
        for key in scalar_keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                writer.add_scalar(f"eval/{key}", float(value), 0)
        writer.add_scalar("eval/frames", float(metrics.get("frames", 0) or 0), 0)
        writer.add_scalar("eval/cases", float(metrics.get("cases", 0) or 0), 0)
        writer.add_text("eval/road_confusion_matrix", _confusion_matrix_markdown("ROAD Confusion Matrix", metrics.get("road_report") or {}), 0)
        writer.add_text("eval/event_confusion_matrix", _confusion_matrix_markdown("EVENT Confusion Matrix", metrics.get("event_report") or {}), 0)
        writer.add_text(
            "eval/run",
            "\n".join(
                [
                    f"- task: `{args.task}` / `{metrics.get('eval_mode')}`",
                    f"- adapter: `{args.adapter_dir}`",
                    f"- index: `{args.index}`",
                    f"- frames: `{metrics.get('frames')}`",
                    f"- output_jsonl: `{args.output_jsonl}`",
                ]
            ),
            0,
        )
    finally:
        writer.flush()
        writer.close()


def _write_html_report(path: pathlib.Path, metrics: Dict[str, Any], args: argparse.Namespace) -> None:
    """写单文件 HTML 可视化报告，按 sft_base 风格直接展示 confusion matrix。"""

    metrics_payload = json.dumps(metrics, ensure_ascii=False).replace("</", "<\\/")
    title = f"SFT Baseline Eval Report - {metrics.get('eval_mode', args.eval_mode)}"
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_lib.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --text: #182033;
      --muted: #667085;
      --line: #d9dee8;
      --accent: #1d4ed8;
      --good: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 22px 28px 16px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    h3 {{ margin: 16px 0 10px; font-size: 15px; }}
    main {{ padding: 22px 28px 40px; max-width: 1500px; margin: 0 auto; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 18px;
    }}
    .meta {{ color: var(--muted); font-size: 13px; display: flex; flex-wrap: wrap; gap: 8px 18px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
    .card {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfcff; }}
    .card .label {{ color: var(--muted); font-size: 12px; margin-bottom: 6px; }}
    .card .value {{ font-size: 22px; font-weight: 700; }}
    .grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; align-items: start; }}
    .matrix-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid var(--line); padding: 7px 9px; text-align: right; vertical-align: middle; }}
    th {{ background: #f1f4f9; color: #344054; font-weight: 650; white-space: nowrap; }}
    td.row-label, th.row-label {{ text-align: left; background: #f8fafc; }}
    td.cell {{ min-width: 78px; font-variant-numeric: tabular-nums; }}
    .cell .count {{ font-weight: 700; }}
    .cell .pct {{ color: #475467; font-size: 11px; }}
    .diag {{ outline: 2px solid rgba(15, 118, 110, 0.35); outline-offset: -2px; }}
    .hint {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
    .pill {{ display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 2px 8px; background: #fff; color: #475467; }}
  </style>
</head>
<body>
  <header>
    <h1>SFT Baseline Eval Report</h1>
    <div id="meta" class="meta"></div>
  </header>
  <main>
    <section>
      <h2>关键指标</h2>
      <div id="kpis" class="cards"></div>
      <div class="hint">ROAD 是 HIGHWAY/NON_HIGHWAY 二分类；EVENT 是 UE/RE 二分类。</div>
    </section>
    <section>
      <h2>Confusion Matrix</h2>
      <div class="grid2">
        <div id="roadMatrix"></div>
        <div id="eventMatrix"></div>
      </div>
    </section>
    <section>
      <h2>Change Detection</h2>
      <div id="changeMatrix" class="grid2"></div>
    </section>
  </main>
  <script>
    const DATA = {metrics_payload};

    function htmlEscape(value) {{
      const map = {{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}};
      return String(value).replace(/[&<>"']/g, ch => map[ch]);
    }}
    function intFmt(value) {{ return Number(value || 0).toLocaleString("en-US"); }}
    function pct(value) {{ return (Number(value || 0) * 100).toFixed(2) + "%"; }}
    function fmt(value) {{
      if (value === null || value === undefined) return "NA";
      if (typeof value === "number") return Math.abs(value) <= 1 ? pct(value) : intFmt(value);
      return String(value);
    }}
    function get(path) {{ return path.split(".").reduce((obj, key) => obj && obj[key], DATA); }}
    function matrixLabels(matrix) {{
      const rows = Object.keys(matrix || {{}});
      const cols = [];
      for (const row of rows) {{
        for (const col of Object.keys(matrix[row] || {{}})) {{
          if (!cols.includes(col)) cols.push(col);
        }}
      }}
      return [rows, cols];
    }}
    function cellColor(count, maxCount, isDiag) {{
      if (!maxCount) return "#fff";
      const t = Math.min(1, Number(count || 0) / maxCount);
      const hue = isDiag ? 173 : 217;
      const sat = isDiag ? 46 : 76;
      const light = 96 - t * 38;
      return `hsl(${{hue}} ${{sat}}% ${{light}}%)`;
    }}
    function renderMatrix(id, title, reportOrMatrix) {{
      const target = document.getElementById(id);
      const matrix = reportOrMatrix.confusion_matrix || reportOrMatrix || {{}};
      const [rows, cols] = matrixLabels(matrix);
      const maxCount = Math.max(0, ...rows.flatMap(r => cols.map(c => Number((matrix[r] || {{}})[c] || 0))));
      let out = `<h3>${{htmlEscape(title)}}</h3><div class="matrix-wrap"><table><thead><tr><th class="row-label">GT \\\\ Pred</th>`;
      for (const col of cols) out += `<th>${{htmlEscape(col)}}</th>`;
      out += `</tr></thead><tbody>`;
      for (const row of rows) {{
        const rowTotal = cols.reduce((s, c) => s + Number((matrix[row] || {{}})[c] || 0), 0);
        out += `<tr><td class="row-label">${{htmlEscape(row)}} <span class="pill">n=${{intFmt(rowTotal)}}</span></td>`;
        for (const col of cols) {{
          const count = Number((matrix[row] || {{}})[col] || 0);
          const diag = row === col;
          const rowPct = rowTotal ? count / rowTotal : 0;
          out += `<td class="cell ${{diag ? "diag" : ""}}" style="background:${{cellColor(count, maxCount, diag)}}"><div class="count">${{intFmt(count)}}</div><div class="pct">${{pct(rowPct)}}</div></td>`;
        }}
        out += `</tr>`;
      }}
      out += `</tbody></table></div>`;
      target.innerHTML = out;
    }}
    function renderKpis() {{
      const items = [
        ["ROAD acc", "road_acc"],
        ["HIGHWAY F1", "highway_f1"],
        ["EVENT acc", "event_acc"],
        ["UE F1", "ue_f1"],
        ["Joint acc", "joint_acc"],
        ["ROAD invalid", "road_invalid_rate"],
        ["EVENT invalid", "event_invalid_rate"],
        ["ROAD change F1", "road_change_f1"],
        ["EVENT change F1", "event_change_f1"],
      ];
      document.getElementById("kpis").innerHTML = items.map(([label, key]) => `<div class="card"><div class="label">${{htmlEscape(label)}}</div><div class="value">${{fmt(get(key))}}</div></div>`).join("");
    }}
    function changeMatrix(prefix) {{
      return {{
        "CHANGE": {{"CHANGE": Number(DATA[`${{prefix}}_tp`] || 0), "STABLE": Number(DATA[`${{prefix}}_fn`] || 0)}},
        "STABLE": {{"CHANGE": Number(DATA[`${{prefix}}_fp`] || 0), "STABLE": Number(DATA[`${{prefix}}_tn`] || 0)}},
      }};
    }}
    function renderMeta() {{
      const fields = [
        ["task", `${{DATA.task || ""}} / ${{DATA.eval_mode || ""}}`],
        ["adapter", DATA.adapter_dir || "base"],
        ["frames", intFmt(DATA.frames)],
        ["routes", intFmt(DATA.route_count)],
        ["image", DATA.image_ablation],
        ["goal_ablation", DATA.goal_ablation],
      ];
      document.getElementById("meta").innerHTML = fields.map(([k, v]) => `<span><b>${{htmlEscape(k)}}:</b> ${{htmlEscape(v)}}</span>`).join("");
    }}
    function main() {{
      renderMeta();
      renderKpis();
      renderMatrix("roadMatrix", "GT ROAD × Pred ROAD", DATA.road_confusion_report || {{}});
      renderMatrix("eventMatrix", "GT EVENT × Pred EVENT", DATA.event_confusion_report || {{}});
      document.getElementById("changeMatrix").innerHTML = `<div id="roadChange"></div><div id="eventChange"></div>`;
      renderMatrix("roadChange", "ROAD change", changeMatrix("road_change"));
      renderMatrix("eventChange", "EVENT change", changeMatrix("event_change"));
    }}
    main();
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main() -> None:
    """CLI 入口。"""

    args = parse_args()
    metrics = evaluate(args)
    if metrics is None:
        return
    metrics = {
        **metrics,
        "task": args.task,
        "adapter_dir": args.adapter_dir,
        "model_dir": args.model_dir,
        "index": args.index,
        "output_dir": args.output_dir,
        "output_json": args.output_json,
        "output_jsonl": args.output_jsonl,
        "output_summary": args.output_summary,
        "output_html": args.output_html,
        "output_tb": args.output_tb,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_json:
        path = pathlib.Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    if args.output_summary:
        _write_summary(pathlib.Path(args.output_summary), metrics, args)
    if args.output_html:
        _write_html_report(pathlib.Path(args.output_html), metrics, args)
    if args.output_tb:
        _write_tensorboard(pathlib.Path(args.output_tb), metrics, args)


if __name__ == "__main__":
    main()
