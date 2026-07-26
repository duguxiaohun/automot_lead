"""SFT base 自由生成评估入口。

评估按真实串行协议执行：Q1 直接生成 RS/ABNORMAL，RS 正确才沿 Q1 KV
继续问 Q2 EVENT。这里不使用 teacher，也不做 CoT。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TextIO

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

from qwen3vl_local.sft_base import DATASET_VERSION  # noqa: E402
from qwen3vl_local.sft_base.labels import RS_LABEL_TO_OPTION, option_for_event  # noqa: E402
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
    if cfg.get("route") != "sft_base_direct_choice":
        raise ValueError(f"adapter route mismatch: expected sft_base_direct_choice, got {cfg.get('route')!r}")
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

        cfg = _validate_adapter_config(adapter_dir, model_dir)
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
    if mode == "none":
        return memory
    rng = random.Random(_stable_case_seed(args, case))
    mem = memory.copy()
    if mode in {"rs", "both", "random"} and (mode != "random" or rng.random() < 0.5):
        mem.rs_label = _pick_different(rng, mem.rs_label, list(RS_LABEL_TO_OPTION.keys()))
    if mode in {"event", "both", "random"}:
        event_candidates = ["RE"] + [str(v) for v in frame.event_option_map.values()]
        mem.event_label = _pick_different(rng, mem.event_label, event_candidates)
    return mem


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


def _classify_hit_timing(hit_index: int, transition_index: int) -> str:
    """把命中帧分类为提前、准点或滞后。"""

    if hit_index < transition_index:
        return "early"
    if hit_index > transition_index:
        return "late"
    return "on_time"


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
        result.update(
            {
                "transition_source": case.source_rs,
                "transition_target": case.target_rs,
                "transition_test_field": "rs",
            }
        )
        if hit_index is not None:
            counters["rs_transition_hit_cases"] += 1
            timing = _classify_hit_timing(hit_index, int(case.transition_index))
            counters[f"rs_transition_{timing}_hits"] += 1
            result.update(
                {
                    "transition_case_hit": True,
                    "transition_hit_frame": hit_index,
                    "transition_hit_timing": timing,
                }
            )
        return result

    counters["event_transition_cases"] += 1
    event_hit = _first_hit(records, "pred_event", case.target_event, lo, hi)
    abnormal_hit = _first_bool_hit(records, "pred_abnormal_bool", case.target_abnormal, lo, hi)
    result.update(
        {
            "transition_source": case.source_event,
            "transition_target": case.target_event,
            "transition_source_abnormal": case.source_abnormal,
            "transition_target_abnormal": case.target_abnormal,
            "transition_test_field": "event",
            "abnormal_hit_frame": abnormal_hit,
        }
    )
    if abnormal_hit is not None:
        counters["event_transition_abnormal_hit_cases"] += 1
    if event_hit is not None:
        counters["event_transition_hit_cases"] += 1
        timing = _classify_hit_timing(event_hit, int(case.transition_index))
        counters[f"event_transition_{timing}_hits"] += 1
        result.update(
            {
                "transition_case_hit": True,
                "transition_hit_frame": event_hit,
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
    q1_abnormal_ok: bool,
    event_ok: Optional[bool],
    args: argparse.Namespace,
) -> None:
    """把每帧自由生成结果写成 jsonl，便于定位转折处是否自行纠正。"""

    if fp is None:
        return
    pred_abnormal_bool = parsed_q1.get("abnormal") == "YES" if parsed_q1.get("abnormal") else None
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
        "gt_rs": frame.rs_label,
        "pred_rs": parsed_q1.get("rs_label"),
        "rs_ok": q1_rs_ok,
        "gt_abnormal": bool(frame.abnormal),
        "pred_abnormal": parsed_q1.get("abnormal"),
        "pred_abnormal_bool": pred_abnormal_bool,
        "abnormal_ok": q1_abnormal_ok,
        "gt_event": frame.event_label,
        "pred_event": parsed_q2.get("event_label") if parsed_q2 else None,
        "event_ok": event_ok,
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

    no-correction：Q1 错只跳过当前帧 Q2，Q2 非法只不更新 EVENT，下一帧
    继续沿用学生输出维护出的 memory；评测阶段不允许脚本纠偏。
    """

    memory: Optional[Memory] = None
    case_records: List[Dict[str, Any]] = []
    counters["cases"] += 1
    counters["case_frames"] += len(case.frames)
    counters["evaluated_routes"] += int(case.start_index == 0 and case.transition_index is None)
    for pos, frame in enumerate(case.frames):
        abs_index = case.start_index + pos
        rs_target = _rs_target_from_frame(frame)
        if memory is None:
            memory = reset_memory_for_frame(rs_target, ego_to_goal_xy=frame.ego_to_goal_xy)
            memory = _apply_initial_memory_noise(memory, frame, args, case)
            counters["initial_noise_cases"] += int(args.initial_memory_noise != "none")
        else:
            memory = refresh_memory_goal(memory, frame.ego_to_goal_xy)

        images = _load_images(frame.history_rgb_paths)
        q1_text, q1_after = _generate_start(bundle, images, build_q1_prompt(memory), int(args.max_new_tokens_q1))
        parsed_q1 = parse_q1_output(q1_text)
        q1_rs_ok = parsed_q1.get("rs_label") == frame.rs_label
        q1_abnormal = parsed_q1.get("abnormal") == "YES" if parsed_q1.get("abnormal") else None
        q1_abnormal_ok = q1_abnormal == frame.abnormal if q1_abnormal is not None else False

        counters["frames"] += 1
        counters["q1_rs_correct"] += int(q1_rs_ok)
        counters["q1_rs_wrong"] += int(not q1_rs_ok)
        counters["q1_abnormal_correct"] += int(q1_abnormal_ok)
        if case.transition_index is not None and abs_index >= case.transition_index:
            counters["transition_post_frames"] += 1
            counters["transition_post_rs_correct"] += int(q1_rs_ok)
            counters["transition_post_abnormal_correct"] += int(q1_abnormal_ok)

        memory = update_memory_after_q1(memory, student_rs_label=parsed_q1.get("rs_label"), student_abnormal=q1_abnormal)
        if not q1_rs_ok:
            case_records.append(
                {
                    "frame_index": abs_index,
                    "pred_rs": parsed_q1.get("rs_label"),
                    "pred_abnormal_bool": q1_abnormal,
                    "pred_event": None,
                }
            )
            _write_frame_record(
                jsonl_fp,
                case,
                frame,
                abs_index,
                parsed_q1,
                None,
                q1_text,
                None,
                q1_rs_ok,
                q1_abnormal_ok,
                None,
                args,
            )
            continue

        q2_prompt = build_q2_prompt(
            memory,
            option_map=frame.event_option_map,
            q1_abnormal=bool(q1_abnormal),
            regular_event_codes=frame.regular_event_codes,
        )
        q2_text, _ = _generate_next(bundle, q1_after, q2_prompt, int(args.max_new_tokens_q2))
        parsed_q2 = parse_q2_output(q2_text, frame.event_option_map)
        target = _event_target_from_frame(frame, student_event=parsed_q2.get("event_label"))
        event_ok = parsed_q2.get("event_label") == target.label

        counters["q2_triggered"] += 1
        counters["q2_candidate_mismatch"] += int(option_for_event(target.label, frame.event_option_map) is None)
        counters["q2_event_correct"] += int(event_ok)
        if case.transition_index is not None and abs_index >= case.transition_index:
            counters["transition_post_q2_triggered"] += 1
            counters["transition_post_event_correct"] += int(event_ok)
        if frame.abnormal:
            counters["q2_ue_total"] += 1
            counters["q2_ue_correct"] += int(event_ok)
        else:
            counters["q2_re_total"] += 1
            counters["q2_re_correct"] += int(event_ok)

        memory = update_memory_after_q2(memory, student_event_label=parsed_q2.get("event_label"))
        if parsed_q2.get("event_label") is None:
            counters["q2_invalid_output"] += 1
        case_records.append(
            {
                "frame_index": abs_index,
                "pred_rs": parsed_q1.get("rs_label"),
                "pred_abnormal_bool": q1_abnormal,
                "pred_event": parsed_q2.get("event_label"),
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
            q1_abnormal_ok,
            event_ok,
            args,
        )
    transition_result = _score_transition_case(case, args, counters, case_records)
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


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """执行自由生成评估。

    离散 memory 在 eval 中由学生输出维护：Q1 更新 RS/ABNORMAL，Q2 更新 EVENT。
    但 EGO_TO_GOAL_XY 是每帧连续量，所以即使没有 reset 也会在提问前刷新为当前帧。
    默认不做脚本纠偏；如果 Q1 RS 错，只跳过本帧 Q2，让后续帧继续暴露漂移。
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = RouteSequenceDataset(
        pathlib.Path(args.index),
        max_routes=0,
        max_frames_per_route=int(args.max_frames_per_route),
    )
    if args.check:
        return {"route_count": len(ds), "check_only": True}
    bundle = load_eval_bundle(
        pathlib.Path(args.model_dir),
        pathlib.Path(args.adapter_dir) if args.adapter_dir else None,
        device,
        merge_lora=bool(args.merge_lora),
    )
    cases = _select_eval_cases(ds, args)
    counters: Dict[str, int] = {
        "cases": 0,
        "case_frames": 0,
        "evaluated_routes": 0,
        "frames": 0,
        "q1_rs_correct": 0,
        "q1_rs_wrong": 0,
        "q1_abnormal_correct": 0,
        "q2_triggered": 0,
        "q2_event_correct": 0,
        "q2_candidate_mismatch": 0,
        "q2_invalid_output": 0,
        "q2_ue_total": 0,
        "q2_ue_correct": 0,
        "q2_re_total": 0,
        "q2_re_correct": 0,
        "script_resets": 0,
        "rs_wrong_resets": 0,
        "initial_noise_cases": 0,
        "transition_cases": 0,
        "rs_transition_cases": 0,
        "rs_transition_hit_cases": 0,
        "rs_transition_early_hits": 0,
        "rs_transition_on_time_hits": 0,
        "rs_transition_late_hits": 0,
        "event_transition_cases": 0,
        "event_transition_hit_cases": 0,
        "event_transition_abnormal_hit_cases": 0,
        "event_transition_early_hits": 0,
        "event_transition_on_time_hits": 0,
        "event_transition_late_hits": 0,
        "transition_post_frames": 0,
        "transition_post_rs_correct": 0,
        "transition_post_abnormal_correct": 0,
        "transition_post_q2_triggered": 0,
        "transition_post_event_correct": 0,
    }
    jsonl_fp: Optional[TextIO] = None
    if args.output_jsonl:
        out_path = pathlib.Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_fp = open(out_path, "w", encoding="utf-8")
    try:
        for case in cases:
            _evaluate_case(bundle, case, args, counters, jsonl_fp)
    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()

    frames = max(1, counters["frames"])
    q2_total = max(1, counters["q2_triggered"])
    transition_post_frames = max(1, counters["transition_post_frames"])
    transition_post_q2 = max(1, counters["transition_post_q2_triggered"])
    return {
        "eval_mode": args.eval_mode,
        "script_correction": "none",
        "route_count": len(ds),
        "selected_case_count": len(cases),
        **counters,
        "rs_acc": counters["q1_rs_correct"] / frames,
        "abnormal_acc": counters["q1_abnormal_correct"] / frames,
        "event_acc_when_rs_correct": counters["q2_event_correct"] / q2_total,
        "ue_acc": counters["q2_ue_correct"] / max(1, counters["q2_ue_total"]),
        "re_acc": counters["q2_re_correct"] / max(1, counters["q2_re_total"]),
        "q2_trigger_rate": counters["q2_triggered"] / frames,
        "transition_post_rs_acc": counters["transition_post_rs_correct"] / transition_post_frames,
        "transition_post_abnormal_acc": counters["transition_post_abnormal_correct"] / transition_post_frames,
        "transition_post_event_acc": counters["transition_post_event_correct"] / transition_post_q2,
        "rs_transition_hit_rate": counters["rs_transition_hit_cases"] / max(1, counters["rs_transition_cases"]),
        "event_transition_hit_rate": counters["event_transition_hit_cases"] / max(1, counters["event_transition_cases"]),
        "event_transition_abnormal_hit_rate": counters["event_transition_abnormal_hit_cases"] / max(1, counters["event_transition_cases"]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SFT base direct-choice adapter")
    p.add_argument("--index", type=str, required=True)
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--adapter-dir", type=str, default=None)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--output-jsonl", type=str, default=None)
    p.add_argument("--eval-mode", choices=["full_route", "rs_transition", "event_transition"], default="full_route")
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--sample-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
    p.add_argument("--transition-window", type=int, default=8)
    p.add_argument("--transition-tolerance", type=int, default=3)
    p.add_argument("--max-transition-cases", type=int, default=0)
    p.add_argument("--initial-memory-noise", choices=["none", "rs", "event", "both", "random"], default="none")
    p.add_argument("--seed", type=int, default=20260724)
    p.add_argument("--max-new-tokens-q1", type=int, default=32)
    p.add_argument("--max-new-tokens-q2", type=int, default=24)
    p.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--check", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_json:
        path = pathlib.Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
