"""SFT v5 自由生成评估入口。

评估不使用 teacher，也不做 Phase B 纠偏；它按真实推理方式让 student 自己维护
RS/EVENT memory。Q1 RS 错时跳过本帧 Q2，下一有效帧恢复 GT RS + RE。
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

import torch

from qwen3vl_local.sft_v3.train import _kv_start_state, _student_generate_kv  # noqa: E402
from qwen3vl_local.sft_v5.labels import option_for_event  # noqa: E402
from qwen3vl_local.sft_v5.prompts import (  # noqa: E402
    Memory,
    build_q1_student_prompt,
    build_q2_student_prompt,
    parse_q1_output,
    parse_q2_output,
    reset_memory_for_frame,
    update_memory_after_q1,
    update_memory_after_q2,
)
from qwen3vl_local.sft_v5.train import (  # noqa: E402
    RouteSequenceDataset,
    _load_images,
    _messages,
    _rs_target_from_frame,
    _event_target_from_frame,
)


@dataclass
class EvalBundle:
    """评估时传递给 v3 KV helper 的轻量 bundle。"""

    model: Any
    processor: Any
    tokenizer: Any
    device: torch.device

    def unwrap(self) -> Any:
        return self.model


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

        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        if merge_lora and hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    return EvalBundle(model=model, processor=processor, tokenizer=processor.tokenizer, device=device)


def _generate(bundle: EvalBundle, images: List[Any], prompt: str, max_new_tokens: int) -> str:
    """对单个 prompt 做 KV prefill + greedy decode。"""

    with torch.inference_mode():
        state = _kv_start_state(bundle, _messages(images, prompt))
        text, _, _ = _student_generate_kv(bundle, state, max_new_tokens)
    return text


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """执行自由生成评估。"""

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
                memory = reset_memory_for_frame(rs_target)
                reset_next = False
            images = _load_images(frame.history_rgb_paths)
            q1_text = _generate(bundle, images, build_q1_student_prompt(memory), int(args.max_new_tokens_q1))
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
            q2_text = _generate(
                bundle,
                images,
                build_q2_student_prompt(
                    memory,
                    option_map=frame.event_option_map,
                    q1_abnormal=bool(q1_abnormal),
                    regular_event_codes=frame.regular_event_codes,
                ),
                int(args.max_new_tokens_q2),
            )
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
    p = argparse.ArgumentParser(description="Evaluate SFT v5 adapter")
    p.add_argument("--index", type=str, required=True)
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--adapter-dir", type=str, default=None)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
    p.add_argument("--max-new-tokens-q1", type=int, default=256)
    p.add_argument("--max-new-tokens-q2", type=int, default=192)
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
