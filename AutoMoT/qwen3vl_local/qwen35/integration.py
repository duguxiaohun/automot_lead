"""Hybrid-cache bridge to the project's token-level planning decoder."""
from qwen3vl_local.goalgen.qwen_kv import segment_kv_for_dit


def install_runner_bridge(module):
    """Bind the external runner to the reviewed local hybrid-cache bridge."""
    if not hasattr(module, "_segment_qwen_cache_for_leadmot"):
        raise RuntimeError("Unsupported external runner: missing Qwen cache segmentation interface")
    module._segment_qwen_cache_for_leadmot = segment_for_decoder


def segment_for_decoder(cache, config):
    expected = tuple(config.qwen_full_attention_layers)
    if hasattr(cache, "layer_types"):
        actual = tuple(i for i, t in enumerate(cache.layer_types) if t == "full_attention")
        if actual != expected or len(cache.layer_types) != config.num_qwen_layers:
            raise ValueError(f"Qwen3.5 decoder/cache layer mismatch: {actual} != {expected}")
    else:
        raise ValueError("Qwen3.5 planning requires the complete hybrid cache")
    segments = segment_kv_for_dit(cache, config.num_layers, config.kv_segment_mode)
    for k, v in segments:
        if k.shape[1] != config.num_kv_heads or k.shape[-1] != config.head_dim:
            raise ValueError("Qwen3.5 planning K/V head layout mismatch")
    return segments
