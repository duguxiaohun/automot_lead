"""Standalone Qwen3-VL-4B-Instruct paradigm-A runner.

This entrypoint is intentionally Qwen-only. It does not import
``vlm_paradigm_a_runner.py`` and does not call AutoMoT inference code.

Default checkpoint:
    AutoMoT/checkpoints/Qwen3-VL-4B-Instruct

The actual inference steps live in ``AutoMoT/qwen3vl_local`` so they can be
modified locally:
    - prompt_pipeline.py: memory / prompt / output parsing
    - image_io.py: synthetic and LEAD RGB loading
    - engine.py: local model loading, prefill, token-by-token decode, KV cache
    - cache_utils.py: KV cache summaries and optional tensor persistence
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
_CHECKPOINT_DIR = _AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B-Instruct"
_DEFAULT_OUTPUT_ROOT = _AUTOMOT_ROOT / "eval_json" / "qwen3vl_instruct_paradigm_a"

for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from qwen3vl_local.engine import LocalQwen3VLInstructEngine, dump_trace  # noqa: E402
from qwen3vl_local.image_io import (  # noqa: E402
    auto_detect_scenario_from_route,
    build_synthetic_raw_and_model_input,
    load_lead_rgb_clip,
)
from qwen3vl_local.prompt_pipeline import (  # noqa: E402
    DrivingMemory,
    build_system_prompt,
    build_user_prompt,
    parse_vlm_output,
    update_memory,
)


@dataclass
class ParadigmAStepRecord:
    step_idx: int
    timestamp: str
    scenario: str
    checkpoint_dir: str
    num_images: int
    num_raw_images: int
    memory_before: dict
    system_prompt: str
    user_prompt: str
    raw_vlm_text: str
    parsed: Dict[str, Optional[str]]
    memory_after: dict
    save_dir: Optional[str] = None
    image_files: List[str] = field(default_factory=list)
    raw_image_files: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step_idx": self.step_idx,
            "timestamp": self.timestamp,
            "scenario": self.scenario,
            "checkpoint_dir": self.checkpoint_dir,
            "num_images": self.num_images,
            "num_raw_images": self.num_raw_images,
            "memory_before": self.memory_before,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "raw_vlm_text": self.raw_vlm_text,
            "parsed": self.parsed,
            "memory_after": self.memory_after,
            "save_dir": self.save_dir,
            "image_files": self.image_files,
            "raw_image_files": self.raw_image_files,
        }


def _save_pil_list(imgs: List[Any], out_dir: pathlib.Path, prefix: str) -> List[str]:
    saved: List[str] = []
    for i, img in enumerate(imgs):
        fname = f"{prefix}{i:03d}.png"
        fpath = out_dir / fname
        if hasattr(img, "save"):
            img.save(str(fpath))
            saved.append(str(fpath.relative_to(out_dir.parent)))
    return saved


def dump_record(
    record: ParadigmAStepRecord,
    trace_dict: dict,
    images: List[Any],
    target_dir: pathlib.Path,
    raw_images: Optional[List[Any]] = None,
) -> None:
    target_dir = pathlib.Path(target_dir)
    inputs_dir = target_dir / "inputs"
    outputs_dir = target_dir / "outputs"
    trace_dir = target_dir / "trace"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    (inputs_dir / "system_prompt.txt").write_text(record.system_prompt, encoding="utf-8")
    (inputs_dir / "user_prompt.txt").write_text(record.user_prompt, encoding="utf-8")
    (inputs_dir / "memory_before.json").write_text(
        json.dumps(record.memory_before, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    record.image_files = _save_pil_list(images, inputs_dir, "image_")
    if raw_images:
        record.raw_image_files = _save_pil_list(raw_images, inputs_dir, "raw_image_")
    record.save_dir = str(target_dir)

    (outputs_dir / "raw_vlm_text.txt").write_text(record.raw_vlm_text, encoding="utf-8")
    (outputs_dir / "parsed.json").write_text(
        json.dumps(record.parsed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outputs_dir / "memory_after.json").write_text(
        json.dumps(record.memory_after, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (trace_dir / "generation_trace.json").write_text(
        json.dumps(trace_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target_dir / "step.json").write_text(
        json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare_images(args: argparse.Namespace) -> Tuple[str, List[Any], List[Any]]:
    scenario = args.scenario
    if args.route_dir:
        raw_images, model_input_images = load_lead_rgb_clip(
            route_dir=args.route_dir,
            anchor=args.anchor,
            rgb_frame_step=args.rgb_frame_step,
            rgb_frame_count=args.num_frames,
        )
        auto_scenario = auto_detect_scenario_from_route(args.route_dir)
        if auto_scenario and auto_scenario != scenario:
            print(f"[scenario] auto-detected '{auto_scenario}', overriding '{scenario}'")
            scenario = auto_scenario
    else:
        raw_images, model_input_images = build_synthetic_raw_and_model_input(
            num_frames=args.num_frames,
            raw_size=(1920, 1080),
            model_input_size=(1152, 384),
        )
    return scenario, raw_images, model_input_images


def run_once(args: argparse.Namespace) -> None:
    checkpoint_dir = pathlib.Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Local Qwen3-VL-4B-Instruct checkpoint not found: {checkpoint_dir}"
        )

    save_root = pathlib.Path(args.save_root).resolve() if args.save_root else _DEFAULT_OUTPUT_ROOT
    step_dir = save_root / "step_000000"
    cache_dir = step_dir / "kv_cache"

    scenario, raw_images, model_input_images = prepare_images(args)
    memory = DrivingMemory.from_scenario(scenario)
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(memory, image_description="<image>")

    engine = LocalQwen3VLInstructEngine(
        checkpoint_dir=checkpoint_dir,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        save_cache=args.save_cache,
    )
    raw_text, trace = engine.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=model_input_images,
        cache_dir=cache_dir,
    )
    parsed = parse_vlm_output(raw_text)
    new_memory = update_memory(memory, parsed)

    record = ParadigmAStepRecord(
        step_idx=0,
        timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        scenario=scenario,
        checkpoint_dir=str(checkpoint_dir),
        num_images=len(model_input_images),
        num_raw_images=len(raw_images),
        memory_before=memory.to_dict(),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_vlm_text=raw_text,
        parsed=parsed,
        memory_after=new_memory.to_dict(),
    )

    dump_record(
        record=record,
        trace_dict=trace.to_dict(),
        images=model_input_images,
        target_dir=step_dir,
        raw_images=raw_images,
    )
    dump_trace(trace, step_dir / "trace")

    print("-" * 72)
    print(f"[checkpoint] {checkpoint_dir}")
    print(f"[output] {step_dir}")
    print(f"[raw text len] {len(raw_text)}")
    print(raw_text if raw_text else "<EMPTY STRING>")
    print(f"[parsed] {parsed}")
    print(f"[memory] {new_memory.to_dict()}")
    print(f"[cache summary] {step_dir / 'trace' / 'generation_trace.json'}")
    if args.save_cache:
        print(f"[cache tensors] {cache_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Qwen3-VL-4B-Instruct paradigm-A runner")
    parser.add_argument("--checkpoint-dir", type=str, default=str(_CHECKPOINT_DIR),
                        help="Local checkpoint directory. Default: AutoMoT/checkpoints/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default="auto",
                        help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32", "auto"],
                        default="bfloat16")
    parser.add_argument("--max-gen-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--save-cache", action="store_true",
                        help="Save prefill/final past_key_values tensors with torch.save. Can be large.")
    parser.add_argument("--save-root", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="MergerIntoSlowTraffic")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--route-dir", type=str,
                        default="/data/lead_data/data/Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46",
                        help="真实 LEAD 路由目录,目录下需有 rgb/*.jpg 子目录。"
                             "示例: /data/lead_data/data/Accident/Town03_Rep0_route_001783_...。"
                             "不传或设为空字符串 → 用合成图(仅验证通路,语义不可信)。")
    parser.add_argument("--anchor", type=int, default=12)
    parser.add_argument("--rgb-frame-step", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_once(args)


if __name__ == "__main__":
    main()
