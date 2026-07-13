"""Qwen3-VL M-RoPE 增量解码的本地实现。

为什么需要这个模块
==================
Qwen3-VL 的 KV-cache 增量解码依赖一套 multimodal RoPE（M-RoPE）位置编码：因为视觉
token 在序列里压缩了文本位置，decode 阶段每个新 token 的位置不是简单的
``cache_len``，而是 ``cache_position + rope_deltas``。

transformers 的 ``Qwen3VLModel.forward`` 在 decode 分支里是这样算的（4.57.x）::

    batch_size, seq_length, _ = inputs_embeds.shape
    delta = cache_position[0] + self.rope_deltas
    position_ids = torch.arange(seq_length)
    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
    position_ids = position_ids.add(delta)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

原本各处（sft_v4/train、sft_v4/eval、engine）是通过
``model.prepare_inputs_for_generation(...)`` 让库自己拼这套输入。但一旦 base model 被
``peft`` 包了一层，``PeftModelForCausalLM.prepare_inputs_for_generation`` 会把
``cache_position`` 裁掉，导致 Qwen3-VL 把每个续写 token 的 M-RoPE 位置算成 0，RoPE
彻底错位、logits 崩坏——表现为老师/学生生成退化成 "no, no, no" / "right right right"
之类的复读，teacher-forced loss 也跟着被污染。

因此这里把"算 decode 位置 + 跑增量 forward"这一小段 Qwen 专有逻辑**搬到本地**，
不再经过 transformers / peft 的 ``prepare_inputs_for_generation`` 黑盒：

1. 位置用**本条 KV 状态自己记下的** ``rope_deltas`` 复算，而不是读
   ``model.rope_deltas`` 属性（后者会被跨分支 prefill 覆盖，teacher/student 共用一个
   base 时尤其危险）；
2. decode 阶段不再重传 ``pixel_values``（图像 token 已在 cache 里），与原生
   ``prepare_inputs_for_generation`` 在 ``cache_position[0] != 0`` 时把图像置 None 的
   行为一致。

只要把同一前缀喂给"本模块的增量 forward"和"从头全量无 cache forward"，二者的
next-token logits 应当在 bf16 数值噪声（~0.2-0.4）内一致——这正是
``sft_v4/test_kv_vs_native.py`` 守护的契约。
"""

from __future__ import annotations

from typing import Any

import torch


def qwen3vl_decode_position_ids(
    rope_deltas: Any,
    prefix_len: int,
    feed_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """复算 Qwen3-VL decode 阶段的 M-RoPE ``position_ids``（形状 ``(3, batch, feed_len)``）。

    完全照搬 ``Qwen3VLModel.forward`` 的 decode 分支公式：
    ``position_ids[:, j] = j + cache_position[0] + rope_deltas``，其中
    ``cache_position[0] == prefix_len``（已经在 cache 里的 token 数）。

    参数
    ----
    rope_deltas:
        本条 KV 状态在 prefill 时拿到的 ``rope_deltas``（标量 / ``(batch,)`` / ``(batch,1)``
        皆可）。为 ``None`` 时退化成纯文本位置（``prefix_len``），仅作兜底。
    prefix_len:
        已进入 ``past_key_values`` 的前缀长度，等价于 ``cache_position[0]``。
    feed_len:
        本次要喂进去的新 token 数（decode 单步时为 1，teacher-forced 续写时可 >1）。
    batch_size:
        batch 维大小。
    device:
        输出张量所在设备。
    """

    base = (
        torch.arange(feed_len, device=device, dtype=torch.long)
        .view(1, -1)
        .expand(batch_size, -1)
    )
    if rope_deltas is None:
        delta = torch.full((batch_size, 1), int(prefix_len), device=device, dtype=torch.long)
    else:
        rd = rope_deltas.to(device) if hasattr(rope_deltas, "to") else torch.as_tensor(rope_deltas, device=device)
        if rd.ndim == 0:
            rd = rd.view(1, 1)
        elif rd.ndim == 1:
            rd = rd.view(-1, 1)
        elif rd.ndim >= 2:
            # 不同 transformers / Qwen3-VL 版本返回的 rope_deltas 方向不完全一致：
            # 常见形状既可能是 (batch, 1)，也可能是 (1, batch)。decode 阶段只需要
            # “每个样本一个 delta”，所以这里统一整理成 (batch, 1)。如果不做这一步，
            # batched rollout 在 active batch 从 2 缩到 1 时会出现
            # "Target sizes: [1, -1]. Tensor sizes: [2, 1]" 这类 expand 报错。
            if rd.shape[0] == 1 and rd.numel() == batch_size:
                rd = rd.reshape(1, batch_size).transpose(0, 1).contiguous()
            elif rd.shape[0] == batch_size:
                rd = rd.reshape(batch_size, -1)[:, :1]
            elif rd.numel() == batch_size:
                rd = rd.reshape(batch_size, 1)
        if rd.shape[0] != batch_size:
            # 正常情况下调用方会传入与 feed_ids batch 一致的 delta；但 Qwen 的
            # incremental output 有时会暴露模型对象上一次 batched prefill 的 stale
            # rope_deltas。若 active batch 已缩小到 1，而 delta 仍是 (2, 1)，直接
            # expand 会报 shape 错。这里做最后一道防御：多出来的行按当前 batch 裁掉；
            # 行数不够时才使用 expand 复制。
            if rd.shape[0] > batch_size:
                rd = rd[:batch_size].contiguous()
        delta = (rd + int(prefix_len)).to(torch.long)
        if delta.shape[0] != batch_size:
            delta = delta.expand(batch_size, -1)
    position_ids = (base + delta).unsqueeze(0).expand(3, -1, -1).contiguous()
    return position_ids


def qwen3vl_incremental_forward(
    model: Any,
    *,
    feed_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Any,
    prefix_len: int,
    rope_deltas: Any,
    return_dict: bool = True,
) -> Any:
    """用本地复算的 M-RoPE ``position_ids`` 跑一次增量 forward（绕开 prepare_inputs）。

    这与原生 ``prepare_inputs_for_generation`` + ``model(**model_inputs)`` 行为等价，
    但：

    - 不经过 peft 的 kwargs 裁剪，``cache_position`` / ``position_ids`` 不会丢；
    - 位置来自传入的 ``rope_deltas``（本条 KV 状态自带），不依赖会被覆盖的
      ``model.rope_deltas`` 属性；
    - decode 阶段不传 ``pixel_values``（图像 token 已在 cache）。

    ``feed_ids`` 形状 ``(batch, feed_len)``；``attention_mask`` 形状
    ``(batch, prefix_len + feed_len)``。返回模型 forward 的 outputs。
    """

    if feed_ids.ndim == 1:
        feed_ids = feed_ids.unsqueeze(0)
    device = feed_ids.device
    feed_len = int(feed_ids.shape[1])
    batch_size = int(feed_ids.shape[0])
    cache_position = torch.arange(prefix_len, prefix_len + feed_len, device=device)
    position_ids = qwen3vl_decode_position_ids(
        rope_deltas, prefix_len, feed_len, batch_size, device
    )
    return model(
        input_ids=feed_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        cache_position=cache_position,
        use_cache=True,
        return_dict=return_dict,
    )
