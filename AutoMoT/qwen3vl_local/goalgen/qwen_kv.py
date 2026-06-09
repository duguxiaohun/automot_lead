"""GoalGen 使用的 Qwen teacher-forced prefill helper。

本模块只复用 ``LocalQwen3VLInstructEngine`` 跑 prefill。Qwen 全程冻结；返回的
K/V 张量都已 detach，作为 DiT-MoT 的语言 memory 使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.util import DEBUG
from typing import Any, Dict, List, Tuple

import torch

from ..engine import LocalQwen3VLInstructEngine
from ..prompt_pipeline import DrivingMemory
from .prompt import build_teacher_system_prompt, build_teacher_user_prompt, describe_image_inputs


@dataclass
class PrefillResult:
    """供 DiT 消费的 teacher-forced 预填充结果。

    ``pooled_kv`` 保留历史字段名只是为了兼容旧调用方。默认 ``select_last`` 模式下，
    每个 DiT 段拿到的是对应 3 层 Qwen 小组里的**最后一层** token-level K/V，
    形状为 ``[B, n_kv, S, head_dim]``；``concat_layers`` 是更重的变体，会把
    3 层沿 token 维拼接，只用于消融实验。
    """

    pooled_kv: List[Tuple[torch.Tensor, torch.Tensor]]
    seq_len: int
    n_kv_heads: int
    head_dim: int
    num_qwen_layers: int
    chat_text: str
    kv_segment_mode: str = "select_last"


def _to_layer_list(past_key_values: Any) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """把 DynamicCache 或旧式 tuple 统一规整成 ``[(K, V), ...]``。"""

    # transformers 4.42+ 默认返回 DynamicCache 对象（不是 tuple）；用 to_legacy_cache()
    # 把它一致转成老式 [(K, V), ...] 结构，下游切分代码不用再对两种 cache 类型各写一套。
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    if not isinstance(past_key_values, (list, tuple)):
        raise TypeError(f"不支持的 past_key_values 类型：{type(past_key_values)}")

    layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for layer in past_key_values:
        if not isinstance(layer, (list, tuple)) or len(layer) != 2:
            raise TypeError("每一层都应该是 (K, V) 二元组")
        k, v = layer
        # detach 切断对 Qwen 计算图的引用：上游 prefill 在 no_grad 里跑本身无 grad，
        # detach 是道保险——防止未来有人忘了 no_grad 时 DiT 训练的反传无意间穿回 Qwen，
        # 把"Qwen 全程冻结"的约定打破。
        layers.append((k.detach(), v.detach()))
    return layers


def segment_kv_for_dit(
    past_key_values: Any,
    num_segments: int = 12,
    mode: str = "select_last",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """把 Qwen KV cache 切成按 DiT 层使用的语言记忆。

    默认 ``select_last`` 只保留每个小组的最后一层 Qwen，直接作为 token-level K/V：
    对 36 层 Qwen 与 12 层 DiT 来说，第 i 个 DiT 层拿到 Qwen 的第 ``3i + 2``
    层，形状是 ``[B, n_kv, S, D]``。

    ``concat_layers`` 是更重的变体：保留组内全部 3 层，并沿 token 轴拼成
    ``[B, n_kv, 3*S, D]``。它只用于消融实验；默认应使用更省显存的
    ``select_last``。``mean`` 保留旧版层平均行为，方便对照。
    """

    layers = _to_layer_list(past_key_values)
    total = len(layers)
    if num_segments <= 0:
        raise ValueError("num_segments 必须 > 0")
    if total < num_segments:
        raise ValueError(f"Qwen 层数 {total} 小于 num_segments {num_segments}")

    mode = mode.lower()
    if mode not in {"concat_layers", "select_last", "mean"}:
        raise ValueError(f"不支持的 Qwen KV 分段模式：{mode}")

    segments: List[Tuple[torch.Tensor, torch.Tensor]] = []
    # 36 / 12 = 3，base=3，extra=0 是常规情况；如果总层数不整除（例如 37 层 / 12 段），
    # 余数 extra 全塞给最后一段，让前 11 段都是 base，最后一段吃残余。
    # 不平均分配是为了保持"前段对前层 Qwen"的语义稳定——浅层 → DiT 浅 block；
    # 余数堆在末段对生成头尾的 KV 影响较小。
    base = total // num_segments
    extra = total - base * num_segments
    cursor = 0
    for seg in range(num_segments):
        seg_len = base + (extra if seg == num_segments - 1 else 0)
        seg_layers = layers[cursor: cursor + seg_len]

        if mode == "select_last":
            # 取 group 内最后一层 Qwen 的 K/V。语言侧 token 数保持 S（≈2300），
            # 显存最省；最后一层通常承载语义最丰富的 hidden，比第一层更适合喂下游。
            segments.append(seg_layers[-1])
        elif mode == "mean":
            # 旧版层平均：把 3 层的 K/V 在 layer 维 stack 后求均值。
            # 缺点是把不同层语义混在一起，方向性会被冲淡，留作消融对照。
            ks = torch.stack([kv[0] for kv in seg_layers], dim=0)
            vs = torch.stack([kv[1] for kv in seg_layers], dim=0)
            segments.append((ks.mean(dim=0), vs.mean(dim=0)))
        else:
            # concat_layers：3 层 K/V 沿 token 轴 (dim=2) 拼接，单段 token 数 = 3*S。
            # 信息保留最完整但语言侧每个 DiT block 的 attention 成本翻 3 倍，
            # 在 96GB H20 + bf16 上 4 帧历史 + 12 层 DiT 接近显存溢出临界。
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
    """向后兼容别名；默认行为已经不再做层平均。"""

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
    """运行 teacher-forced Qwen 预填充，并返回 DiT 可直接使用的 K/V 记忆。"""

    # engine.load() 内部做"已加载就跳过"的幂等检查；每一步都喊一次是为了让训练器
    # 重启后第一个 step 也能自动唤醒模型，避免 runner 处理 lazy load 状态分支。
    engine.load()

    system_prompt = build_teacher_system_prompt()
    # describe_image_inputs(len(images)) 让 prompt 文字与实际传入图像数一致；
    # Qwen processor 不会校验"prompt 里说了几张图 vs 真传几张图"，文字和实物对齐有助于
    # KV cache 里"图像和语言之间的对应关系"质量。
    user_prompt = build_teacher_user_prompt(
        memory,
        image_description=describe_image_inputs(len(images)),
    )

    messages = engine.build_messages(system_prompt, user_prompt, images)
    chat_text = engine.apply_chat_template(messages)
    inputs = engine.prepare_inputs(chat_text, images)

    # no_grad 是性能 + 内存的硬性需求：Qwen ~4B 参数，prefill 一旦带 autograd state
    # 会瞬间多吃几个 GB；且我们不会回传梯度到 Qwen，开 grad 完全是浪费。
    with torch.no_grad():
        outputs = engine.prefill(inputs)

    segmented = segment_kv_for_dit(
        outputs.past_key_values,
        num_segments=num_segments,
        mode=kv_segment_mode,
    )
    # 从第 0 段读形状元信息：所有段的 (B, n_kv_heads, S, head_dim) 一致（除 concat_layers
    # 模式下 S 维三倍以外）。当前共享架构下 DiT 直接以 (n_heads=8, head_dim=128) 接 Qwen K/V，
    # 不再需要 language_kv_input_dim probe；这些字段保留只是给 runner/eval 做形状摘要。
    k0, _ = segmented[0]

    # -> seq_len: 2255
    # -> n_kv_heads: 8
    # -> head_dim: 128
    # -> num_qwen_layers (原始 KV Cache 层数): 36
    # -> kv_segment_mode: 'select_last'
    # -> chat_text (length: 2701): '<|im_start|>system\nYou are an autonomous driving a...'
    # -> pooled_kv (共 12 段):
    #     段 00 | K shape: torch.Size([1, 8, 2255, 128]), V shape: torch.Size([1, 8, 2255, 128])

    return PrefillResult(
        pooled_kv=segmented,
        seq_len=int(k0.shape[2]),
        n_kv_heads=int(k0.shape[1]),
        head_dim=int(k0.shape[3]),
        # 这里再调一次 _to_layer_list 不会重新 detach（已经 detach 过），只是为了拿到层数；
        # 比缓存 len(layers) 多一次 O(layers) 遍历，但代码更线性、不依赖局部状态。
        num_qwen_layers=len(_to_layer_list(outputs.past_key_values)),
        chat_text=chat_text,
        kv_segment_mode=kv_segment_mode,
    )


def summarize_pooled_kv(pooled: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
    """返回适合写入 JSON 的紧凑摘要，不包含真实张量内容。"""

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


def pad_pooled_kv_batch(
    prefills: List[PrefillResult],
    *,
    target_device: torch.device | None = None,
    target_dtype: torch.dtype | None = None,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, List[int]]:
    """把 B 条独立 PrefillResult 沿 batch 维拼成 batched K/V + key padding mask。

    背景：每条样本独立跑 ``teacher_forced_prefill`` 拿到的 ``pooled_kv`` seq_len 不同
    （prompt 长度 + 图像 token 数随样本变化）。batched DiT 训练时必须把它们 pad 到
    本 batch 内最长 seq_len，并提供 padding mask 告诉 JointAttention 哪些 token 是
    real / pad，否则 vision Q 在 SDPA 内部会无差别地 attend 到 padding 位置，等于
    安静地把无意义的 zero K/V 引入 softmax 分母，**训练表现会缓慢劣化但不报错**。

    输入：
    - prefills：长度 B 的 PrefillResult 列表；要求所有元素的 ``len(pooled_kv)``、
      ``n_kv_heads``、``head_dim`` 必须一致（否则属于 dataset 构建错误，立即抛错）。
    - target_device / target_dtype：可选 cast 目标。一般传 DiT device + dit_dtype，
      让 batched K/V 直接落到 DiT 用得到的位置，避免 train loop 里再 .to() 一次。

    返回：
    - batched_pooled_kv：长度 == num_segments 的列表，每段 ``(K, V)`` 形状
      ``[B, n_kv_heads, S_max, head_dim]``；padding 位置填 0。
    - lang_key_padding_mask：bool tensor ``[B, S_max]``，True 表示真实 token，
      False 表示 padding。可直接传给 DiTMoT.forward 的 ``lang_key_padding_mask``。
    - seq_lens：长度 B 的 int 列表，原始 prefill seq_len，用于日志统计 / sanity 检查。
    """

    if len(prefills) == 0:
        raise ValueError("pad_pooled_kv_batch 收到空 prefills 列表")
    num_segments = len(prefills[0].pooled_kv)
    n_kv_heads = prefills[0].n_kv_heads
    head_dim = prefills[0].head_dim
    for i, pf in enumerate(prefills):
        if len(pf.pooled_kv) != num_segments:
            raise ValueError(
                f"prefill[{i}] 段数 {len(pf.pooled_kv)} != prefill[0] 段数 {num_segments}"
            )
        if pf.n_kv_heads != n_kv_heads or pf.head_dim != head_dim:
            raise ValueError(
                f"prefill[{i}] (n_kv={pf.n_kv_heads}, head_dim={pf.head_dim}) "
                f"与 prefill[0] (n_kv={n_kv_heads}, head_dim={head_dim}) 不一致"
            )

    seq_lens = [int(pf.seq_len) for pf in prefills]
    s_max = max(seq_lens)
    b = len(prefills)

    # device / dtype：默认沿用首条 prefill 的 K 张量参数，调用方可显式 override。
    ref_k = prefills[0].pooled_kv[0][0]
    out_device = target_device if target_device is not None else ref_k.device
    out_dtype = target_dtype if target_dtype is not None else ref_k.dtype

    batched_pooled_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for seg_idx in range(num_segments):
        # 每段独立 allocate：bf16 下 [B=8, 8, 2300, 128] 占 ~37MB，B=4 / 12 段总量在 ~1.7GB；
        # 训练时 grad checkpoint 会把这部分作为 detach 输入，不进激活回传图，显存可控。
        k_batch = torch.zeros(b, n_kv_heads, s_max, head_dim, device=out_device, dtype=out_dtype)
        v_batch = torch.zeros_like(k_batch)
        for sample_idx, pf in enumerate(prefills):
            k_seg, v_seg = pf.pooled_kv[seg_idx]
            # k_seg 原本是 [1, n_kv, S_i, head_dim]；按 sample_idx 写入 [B, n_kv, S_i, head_dim] 切片。
            # .to() 显式对齐 dtype/device，避免后续 SDPA 内部混合精度路径触发错误。
            s_i = k_seg.shape[2]
            k_batch[sample_idx, :, :s_i, :] = k_seg[0].to(device=out_device, dtype=out_dtype)
            v_batch[sample_idx, :, :s_i, :] = v_seg[0].to(device=out_device, dtype=out_dtype)
        batched_pooled_kv.append((k_batch, v_batch))

    # mask：True=有效，False=padding。调用方会在批内等长时改传 None，让 DiT 保持无 mask fast path。
    mask = torch.zeros(b, s_max, dtype=torch.bool, device=out_device)
    for sample_idx, s_i in enumerate(seq_lens):
        mask[sample_idx, :s_i] = True

    return batched_pooled_kv, mask, seq_lens
