"""KV cache inspection and persistence helpers.

这个文件只做两件轻量工作：
1. 把 transformers 返回的 KV cache 结构摘要成 JSON 友好的 shape/dtype/device。
2. 在显式打开 --save-cache 时，把 cache 张量保存成 .pt 文件。

注意：summary 不复制 tensor 值，适合常规日志；save_kv_cache 会保存真实张量，
体积可能很大，只用于需要离线分析 KV cache 时。
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List


def _tensor_summary(x: Any) -> Dict[str, Any]:
    """提取单个张量的元信息。

    这里不依赖 torch.Tensor 类型判断，只用 duck typing 读 shape/dtype/device，
    这样兼容不同 transformers cache 实现里的 tensor-like 对象。
    """

    return {
        "shape": list(x.shape) if hasattr(x, "shape") else None,
        "dtype": str(getattr(x, "dtype", None)),
        "device": str(getattr(x, "device", None)),
    }


def summarize_kv_cache(cache: Any) -> Dict[str, Any]:
    """返回 KV cache 的结构摘要，不复制真实张量值。

    新版 transformers 可能返回 DynamicCache；旧版或部分模型返回 legacy tuple。
    为了让日志统一，这里优先调用 to_legacy_cache()，再按 layer 提取 key/value
    的形状。若转换失败，则退回读取 key_cache/value_cache 属性。
    """

    summary: Dict[str, Any] = {"type": type(cache).__name__, "layers": []}

    # DynamicCache -> tuple 只是结构视图转换，不会把大张量复制进 JSON。
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
                # 标准 legacy cache 每层至少包含 key/value 两个张量。
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
    """用 torch.save 保存 KV cache，并返回保存路径。

    保存前尽量转换成 legacy tuple，方便之后在没有原 DynamicCache 类上下文时
    也能用 torch.load 读取结构。这个函数只在 --save-cache 打开时调用。
    """

    import torch

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = cache
    if hasattr(cache, "to_legacy_cache"):
        payload = cache.to_legacy_cache()
    torch.save(payload, str(path))
    return str(path)

