"""Static sanity check for SFT v2 target-token loss spans."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_v2.prompts import format_scene_assistant, format_status_assistant, target_spans
from qwen3vl_local.sft_v2.train import _assistant_token_mask


class _TokenizerBundle:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def _check_token_weights(model_dir: pathlib.Path, scene_text: str, status_text: str) -> dict:
    """Optionally verify tokenizer-level value masks when local tokenizer exists."""

    if not model_dir.exists():
        return {"skipped": True, "reason": f"model_dir not found: {model_dir}"}
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    bundle = _TokenizerBundle(processor.tokenizer)
    checks = {}
    for name, text in (("scene", scene_text), ("status", status_text)):
        token_ids, value_mask = _assistant_token_mask(bundle, text)
        checks[name] = {
            "tokens": len(token_ids),
            "value_tokens": sum(1 for x in value_mask if x),
            "format_tokens": sum(1 for x in value_mask if not x),
            "mask_values": sorted({int(x) for x in value_mask}),
        }
    ok = (
        checks["scene"]["value_tokens"] > 0
        and checks["status"]["value_tokens"] > 0
        and checks["scene"]["format_tokens"] > 0
        and checks["status"]["format_tokens"] > 0
        and checks["scene"]["mask_values"] in ([0, 1], [1])
        and checks["status"]["mask_values"] == [0, 1]
    )
    return {"skipped": False, "ok": ok, "details": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SFT v2 loss mask spans")
    parser.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    args = parser.parse_args()

    scene_text = format_scene_assistant("Accident")
    status_text = format_status_assistant("hazard_detect", "max_brake_or_min_gap")
    scene_spans = target_spans(scene_text)
    status_spans = target_spans(status_text)
    scene_recovered = {k: scene_text[a:b] for k, (a, b) in scene_spans.items()}
    status_recovered = {k: status_text[a:b] for k, (a, b) in status_spans.items()}
    span_ok = scene_recovered == {"scene": "Accident"} and status_recovered == {
        "status": "hazard_detect",
        "subgoal": "max_brake_or_min_gap",
    }
    token_check = _check_token_weights(pathlib.Path(args.model_dir), scene_text, status_text)
    ok = span_ok and token_check.get("ok", True)
    print(json.dumps({
        "scene_assistant": scene_text,
        "status_assistant": status_text,
        "scene_spans": scene_spans,
        "status_spans": status_spans,
        "scene_recovered": scene_recovered,
        "status_recovered": status_recovered,
        "span_ok": span_ok,
        "token_check": token_check,
        "ok": ok,
    }, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
