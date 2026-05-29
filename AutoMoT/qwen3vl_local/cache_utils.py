"""KV cache inspection and persistence helpers."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List


def _tensor_summary(x: Any) -> Dict[str, Any]:
    return {
        "shape": list(x.shape) if hasattr(x, "shape") else None,
        "dtype": str(getattr(x, "dtype", None)),
        "device": str(getattr(x, "device", None)),
    }


def summarize_kv_cache(cache: Any) -> Dict[str, Any]:
    """Return JSON-safe cache structure metadata without copying tensor values."""
    summary: Dict[str, Any] = {"type": type(cache).__name__, "layers": []}

    legacy = cache
    if hasattr(cache, "to_legacy_cache"):
        try:
            legacy = cache.to_legacy_cache()
            summary["legacy_type"] = type(legacy).__name__
        except Exception as e:
            summary["to_legacy_cache_error"] = repr(e)
            legacy = cache

    if isinstance(legacy, (list, tuple)):
        layers: List[Dict[str, Any]] = []
        for i, layer in enumerate(legacy):
            layer_info: Dict[str, Any] = {"layer": i, "type": type(layer).__name__}
            if isinstance(layer, (list, tuple)) and len(layer) >= 2:
                layer_info["key"] = _tensor_summary(layer[0])
                layer_info["value"] = _tensor_summary(layer[1])
            layers.append(layer_info)
        summary["layers"] = layers
        summary["num_layers"] = len(layers)
        return summary

    for attr in ("key_cache", "value_cache"):
        val = getattr(cache, attr, None)
        if isinstance(val, (list, tuple)):
            summary[attr] = [_tensor_summary(x) for x in val]

    return summary


def save_kv_cache(cache: Any, path: pathlib.Path) -> str:
    """Save a cache object or its legacy tuple form with torch.save."""
    import torch

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = cache
    if hasattr(cache, "to_legacy_cache"):
        payload = cache.to_legacy_cache()
    torch.save(payload, str(path))
    return str(path)

