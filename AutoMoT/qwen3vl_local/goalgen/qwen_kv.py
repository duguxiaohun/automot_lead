"""Qwen teacher-forced prefill helpers for GoalGen.

This module reuses ``LocalQwen3VLInstructEngine`` only for prefill. Qwen stays
frozen; the returned K/V tensors are detached and used as language memory by
DiT-MoT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from ..engine import LocalQwen3VLInstructEngine
from ..prompt_pipeline import DrivingMemory
from .prompt import build_teacher_system_prompt, build_teacher_user_prompt, describe_image_inputs


@dataclass
class PrefillResult:
    """Teacher-forced prefill output consumed by DiT.

    ``pooled_kv`` keeps the historical field name for compatibility. With the
    default ``select_last`` mode each DiT segment is the **last** Qwen layer of
    its 3-layer group (token-level K/V, shape ``[B, n_kv, S, head_dim]``);
    ``concat_layers`` is the heavier "concat all 3 layers along token axis"
    variant kept available for ablation.
    """

    pooled_kv: List[Tuple[torch.Tensor, torch.Tensor]]
    seq_len: int
    n_kv_heads: int
    head_dim: int
    num_qwen_layers: int
    chat_text: str
    kv_segment_mode: str = "select_last"


def _to_layer_list(past_key_values: Any) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Normalize DynamicCache / legacy tuple into ``[(K, V), ...]``."""

    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    if not isinstance(past_key_values, (list, tuple)):
        raise TypeError(f"unexpected past_key_values type: {type(past_key_values)}")

    layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for layer in past_key_values:
        if not isinstance(layer, (list, tuple)) or len(layer) != 2:
            raise TypeError("each layer should be a (K, V) pair")
        k, v = layer
        layers.append((k.detach(), v.detach()))
    return layers


def segment_kv_for_dit(
    past_key_values: Any,
    num_segments: int = 12,
    mode: str = "select_last",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Split Qwen KV cache into DiT-layer memories.

    Default ``select_last`` keeps only the last Qwen layer of each group as
    token-level K/V: for a 36-layer Qwen and a 12-layer DiT, block i receives
    Qwen layer ``3i + 2`` directly, shape ``[B, n_kv, S, D]``.

    ``concat_layers`` is the heavier variant: it keeps all 3 layers in the group
    by concatenating along the token axis -> ``[B, n_kv, 3*S, D]``. Use only for
    ablation; default is the memory-friendly ``select_last``.
    ``mean`` preserves the old layer-mean behavior.
    """

    layers = _to_layer_list(past_key_values)
    total = len(layers)
    if num_segments <= 0:
        raise ValueError("num_segments must be > 0")
    if total < num_segments:
        raise ValueError(f"qwen layers {total} < num_segments {num_segments}")

    mode = mode.lower()
    if mode not in {"concat_layers", "select_last", "mean"}:
        raise ValueError(f"unsupported qwen KV segment mode: {mode}")

    segments: List[Tuple[torch.Tensor, torch.Tensor]] = []
    base = total // num_segments
    extra = total - base * num_segments
    cursor = 0
    for seg in range(num_segments):
        seg_len = base + (extra if seg == num_segments - 1 else 0)
        seg_layers = layers[cursor: cursor + seg_len]

        if mode == "select_last":
            segments.append(seg_layers[-1])
        elif mode == "mean":
            ks = torch.stack([kv[0] for kv in seg_layers], dim=0)
            vs = torch.stack([kv[1] for kv in seg_layers], dim=0)
            segments.append((ks.mean(dim=0), vs.mean(dim=0)))
        else:
            k_cat = torch.cat([kv[0] for kv in seg_layers], dim=2)
            v_cat = torch.cat([kv[1] for kv in seg_layers], dim=2)
            segments.append((k_cat, v_cat))

        cursor += seg_len
    return segments


def pool_kv_for_dit(
    past_key_values: Any,
    num_segments: int = 12,
    mode: str = "select_last",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Backward-compatible alias; default no longer layer-pools."""

    return segment_kv_for_dit(
        past_key_values,
        num_segments=num_segments,
        mode=mode,
    )


def teacher_forced_prefill(
    engine: LocalQwen3VLInstructEngine,
    memory: DrivingMemory,
    images: List[Any],
    num_segments: int = 12,
    kv_segment_mode: str = "select_last",
) -> PrefillResult:
    """Run teacher-forced Qwen prefill and return DiT-ready K/V memories."""

    engine.load()

    system_prompt = build_teacher_system_prompt()
    user_prompt = build_teacher_user_prompt(
        memory,
        image_description=describe_image_inputs(len(images)),
    )

    messages = engine.build_messages(system_prompt, user_prompt, images)
    chat_text = engine.apply_chat_template(messages)
    inputs = engine.prepare_inputs(chat_text, images)

    with torch.no_grad():
        outputs = engine.prefill(inputs)

    segmented = segment_kv_for_dit(
        outputs.past_key_values,
        num_segments=num_segments,
        mode=kv_segment_mode,
    )
    k0, _ = segmented[0]

    return PrefillResult(
        pooled_kv=segmented,
        seq_len=int(k0.shape[2]),
        n_kv_heads=int(k0.shape[1]),
        head_dim=int(k0.shape[3]),
        num_qwen_layers=len(_to_layer_list(outputs.past_key_values)),
        chat_text=chat_text,
        kv_segment_mode=kv_segment_mode,
    )


def summarize_pooled_kv(pooled: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
    """Return a compact JSON-friendly summary without tensor payloads."""

    if not pooled:
        return {"num_segments": 0}
    k0, v0 = pooled[0]
    return {
        "num_segments": len(pooled),
        "kv_shape": list(k0.shape),
        "k_dtype": str(k0.dtype),
        "v_dtype": str(v0.dtype),
        "device": str(k0.device),
    }
