"""Persist and validate the local Qwen3.5 adapter contract."""
import json
from pathlib import Path
from .backend import backend_contract
from .identity import require_weight_identity


def save_adapter(model, path, **kwargs):
    assets = getattr(model.config, "local_asset_sha256", None)
    require_weight_identity(assets)
    model.save_pretrained(str(path), **kwargs)
    (Path(path) / "qwen35_backend.json").write_text(json.dumps(backend_contract(), indent=2) + "\n")
    assets = getattr(model.config, "local_asset_sha256", None)
    if assets:
        (Path(path) / "qwen35_base_assets.json").write_text(json.dumps(assets, indent=2) + "\n")


def validate_adapter(path, model=None, *, base_assets=None):
    root = Path(path).expanduser().resolve()
    marker = root / "qwen35_backend.json"
    if not marker.is_file():
        raise ValueError(f"Adapter lacks Qwen3.5 local backend contract: {root}; Qwen3-VL adapters require original source/base")
    if json.loads(marker.read_text()) != backend_contract():
        raise ValueError(f"Adapter Qwen3.5 source/template contract mismatch: {root}; use its original source")
    asset_marker = root / "qwen35_base_assets.json"
    if not asset_marker.is_file():
        raise ValueError(f"Adapter lacks base assets contract: {root}")
    recorded_assets = json.loads(asset_marker.read_text())
    require_weight_identity(recorded_assets)
    if model is not None:
        base_assets = getattr(getattr(model, "config", None), "local_asset_sha256", None)
        require_weight_identity(base_assets)
    if base_assets is not None:
        require_weight_identity(base_assets)
        if recorded_assets != base_assets:
            raise ValueError("Adapter/base weights, tokenizer, config or processor assets differ")
    return root


class LocalPeftModel:
    @classmethod
    def from_pretrained(cls, model, path, *args, **kwargs):
        from peft import PeftModel
        path = validate_adapter(path, model)
        if getattr(model.config, "model_type", None) != "qwen3_5":
            raise ValueError("Qwen3.5 adapter requires Qwen3.5 base model")
        kwargs["local_files_only"] = True
        return PeftModel.from_pretrained(model, str(path), *args, **kwargs)
