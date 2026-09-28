"""Read-only local installation check; never downloads or starts training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .backend import local_model_dir, require_runtime, backend_contract


def check(model_dir, *, action=False):
    errors = []
    report = {"model_dir": str(Path(model_dir).expanduser().resolve()),
              "backend": backend_contract(), "errors": errors}
    vendor = Path(__file__).parent / "vendor"
    manifest = json.loads((vendor / "UPSTREAM.json").read_text())
    for name, record in manifest["files"].items():
        path = vendor / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["local_sha256"]:
            errors.append(f"Local source SHA256 mismatch: {path}")
    try:
        require_runtime()
    except (ImportError, RuntimeError) as exc:
        errors.append(str(exc))
    try:
        root = local_model_dir(model_dir)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        root = None
    if root is not None:
        config = json.loads((root / "config.json").read_text())
        text = config.get("text_config", {})
        expected = dict(hidden_size=2560, num_hidden_layers=32, num_key_value_heads=4, head_dim=256)
        for key, value in expected.items():
            if text.get(key) != value:
                errors.append(f"Expected Qwen3.5-4B {key}={value}, got {text.get(key)}")
        for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
                     "video_preprocessor_config.json", "chat_template.jinja", "generation_config.json"):
            if not (root / name).is_file():
                errors.append(f"Missing local model asset: {name}")
        index = root / "model.safetensors.index.json"
        if index.is_file():
            shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            shards = ["model.safetensors"]
        for name in shards:
            path = (root / name).resolve()
            if root not in path.parents or not path.is_file() or path.stat().st_size == 0:
                errors.append(f"Missing/invalid local weight shard: {name}")
        report["weight_shards"] = shards
    if action:
        runner = Path(__file__).resolve().parents[2] / "leaderboard/team_code/mot_lead_offline_runner.py"
        if not runner.is_file():
            errors.append(f"Missing Action runner: {runner}")
    report["ready"] = not errors
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(Path(__file__).resolve().parents[2] / "checkpoints/Qwen3.5-4B"))
    parser.add_argument("--action", action="store_true", help="also check external Action runner")
    args = parser.parse_args()
    report = check(args.model_dir, action=args.action)
    # Compact output: provenance is available on disk; diagnostics remain readable.
    report["backend"] = {k:v for k,v in report["backend"].items() if k != "source_sha256"}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
