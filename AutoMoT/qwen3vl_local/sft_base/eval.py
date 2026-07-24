"""SFT base 自由生成评估入口。

评估按真实串行协议执行：Q1 直接生成 RS/ABNORMAL，RS 正确才沿 Q1 KV
继续问 Q2 EVENT。这里不使用 teacher，也不做 CoT。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
from qwen3vl_local.sft_base.labels import option_for_event  # noqa: E402
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


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """执行自由生成评估。

    离散 memory 在 eval 中由学生输出维护：Q1 更新 RS/ABNORMAL，Q2 更新 EVENT。
    但 EGO_TO_GOAL_XY 是每帧连续量，所以即使没有 reset 也会在提问前刷新为当前帧。
    如果 Q1 RS 错，按 v5 串行口径不继续问本帧 Q2，并让下一帧 reset memory。
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = RouteSequenceDataset(
        pathlib.Path(args.index),
        max_routes=int(args.max_routes),
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
    counters = {
        "frames": 0,
        "q1_rs_correct": 0,
        "q1_abnormal_correct": 0,
        "q2_triggered": 0,
        "q2_event_correct": 0,
        "q2_candidate_mismatch": 0,
        "q2_invalid_output": 0,
        "q2_ue_total": 0,
        "q2_ue_correct": 0,
        "q2_re_total": 0,
        "q2_re_correct": 0,
        "rs_wrong_resets": 0,
    }
    for route in ds.rows:
        memory: Optional[Memory] = None
        reset_next = False
        for frame in route.frames:
            rs_target = _rs_target_from_frame(frame)
            if memory is None or reset_next:
                memory = reset_memory_for_frame(rs_target, ego_to_goal_xy=frame.ego_to_goal_xy)
                reset_next = False
            else:
                memory = refresh_memory_goal(memory, frame.ego_to_goal_xy)
            images = _load_images(frame.history_rgb_paths)
            q1_text, q1_after = _generate_start(bundle, images, build_q1_prompt(memory), int(args.max_new_tokens_q1))
            parsed_q1 = parse_q1_output(q1_text)
            q1_rs_ok = parsed_q1.get("rs_label") == frame.rs_label
            q1_abnormal = parsed_q1.get("abnormal") == "YES" if parsed_q1.get("abnormal") else None
            counters["frames"] += 1
            counters["q1_rs_correct"] += int(q1_rs_ok)
            counters["q1_abnormal_correct"] += int(q1_abnormal == frame.abnormal if q1_abnormal is not None else False)
            memory = update_memory_after_q1(memory, student_rs_label=parsed_q1.get("rs_label"), student_abnormal=q1_abnormal)
            if not q1_rs_ok:
                counters["rs_wrong_resets"] += 1
                reset_next = True
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
            counters["q2_triggered"] += 1
            counters["q2_candidate_mismatch"] += int(option_for_event(target.label, frame.event_option_map) is None)
            event_ok = parsed_q2.get("event_label") == target.label
            counters["q2_event_correct"] += int(event_ok)
            if frame.abnormal:
                counters["q2_ue_total"] += 1
                counters["q2_ue_correct"] += int(event_ok)
            else:
                counters["q2_re_total"] += 1
                counters["q2_re_correct"] += int(event_ok)
            memory = update_memory_after_q2(memory, student_event_label=parsed_q2.get("event_label"))
            if parsed_q2.get("event_label") is None:
                counters["q2_invalid_output"] += 1
                reset_next = True

    frames = max(1, counters["frames"])
    q2_total = max(1, counters["q2_triggered"])
    return {
        **counters,
        "rs_acc": counters["q1_rs_correct"] / frames,
        "abnormal_acc": counters["q1_abnormal_correct"] / frames,
        "event_acc_when_rs_correct": counters["q2_event_correct"] / q2_total,
        "ue_acc": counters["q2_ue_correct"] / max(1, counters["q2_ue_total"]),
        "re_acc": counters["q2_re_correct"] / max(1, counters["q2_re_total"]),
        "q2_trigger_rate": counters["q2_triggered"] / frames,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SFT base direct-choice adapter")
    p.add_argument("--index", type=str, required=True)
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--adapter-dir", type=str, default=None)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
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
