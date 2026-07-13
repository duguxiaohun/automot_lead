"""SFT v5 batched KV helper 的轻量测试。

这些测试不加载 Qwen，只验证 batch state 切片这类纯 tensor 逻辑。真正的
Qwen batched rollout 需要 GPU/model，在训练 smoke 中验证。
"""

from __future__ import annotations

import pathlib
import sys

import torch

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_v3.train import KVState
from qwen3vl_local.mrope_utils import qwen3vl_decode_position_ids
from qwen3vl_local.sft_v5.train import _last_valid_next_logits, _normalize_rope_deltas_batch, _slice_kv_state_batch


def main() -> None:
    # 构造一个最小 KVState：batch=3、seq=5。这里不用真实 Qwen cache，
    # 只要保证 tensor 的第 0 维是 batch 维，就能覆盖 slice helper 的核心约定。
    batch = 3
    seq = 5
    hidden = 4
    decoded = torch.arange(batch * seq).view(batch, seq)
    attention = torch.ones(batch, seq, dtype=torch.long)
    # legacy tuple cache 形状近似 (batch, heads, seq, head_dim)，足够验证 batch 维切片。
    key = torch.arange(batch * 2 * seq * hidden).view(batch, 2, seq, hidden)
    value = key + 1000
    state = KVState(
        decoded_input_ids=decoded,
        cache_input_ids=decoded.clone(),
        attention_mask=attention,
        past_key_values=((key, value),),
        rope_deltas=torch.tensor([[10], [20], [30]]),
        next_logits=torch.arange(batch * hidden).view(batch, hidden).float(),
    )
    sliced = _slice_kv_state_batch(state, [2, 0])
    # 切片顺序故意用 [2, 0]，验证 helper 不是简单取前 N 个，而是按指定 row 重排。
    assert sliced.decoded_input_ids.tolist() == [decoded[2].tolist(), decoded[0].tolist()]
    assert sliced.attention_mask.shape == (2, seq)
    assert sliced.rope_deltas.tolist() == [[30], [10]]
    skey, svalue = sliced.past_key_values[0]
    assert torch.equal(skey[0], key[2])
    assert torch.equal(skey[1], key[0])
    assert torch.equal(svalue[0], value[2])
    assert torch.equal(sliced.next_logits[0], state.next_logits[2])

    # rope_deltas 也可能是 (1, batch) 方向；这正是 batched Qwen fallback 报错的来源。
    # v5 内部现在统一要求切片后是 (new_batch, 1)，后续增量 decode 才不会把 batch
    # 维误当成 token/feed 维。
    state_transposed_rope = KVState(
        decoded_input_ids=decoded,
        cache_input_ids=decoded.clone(),
        attention_mask=attention,
        past_key_values=((key, value),),
        rope_deltas=torch.tensor([[10, 20, 30]]),
        next_logits=torch.arange(batch * hidden).view(batch, hidden).float(),
    )
    sliced_transposed = _slice_kv_state_batch(state_transposed_rope, [2, 0])
    assert sliced_transposed.rope_deltas.tolist() == [[30], [10]]

    # prefill 出口也要直接归一化，避免只有 slice helper 修好、但 active batch 生成循环
    # 内部仍拿到横向 (1, batch) 的 rope_deltas。
    normalized = _normalize_rope_deltas_batch(torch.tensor([[10, 20, 30]]), batch)
    assert normalized.shape == (batch, 1)
    assert normalized.tolist() == [[10], [20], [30]]

    # M-RoPE helper 自身也要能把 (1, batch) 归一化成每样本一个 delta，
    # 否则 batch_size=2/feed_len=1 时会错误广播成 feed_len=2。
    pos = qwen3vl_decode_position_ids(torch.tensor([[10, 20]]), prefix_len=5, feed_len=1, batch_size=2, device=torch.device("cpu"))
    assert pos.shape == (3, 2, 1)
    assert pos[0, :, 0].tolist() == [15, 25]

    # active batch 缩到 1 时也不能再拿着两个 delta expand。
    pos_one = qwen3vl_decode_position_ids(torch.tensor([[30]]), prefix_len=5, feed_len=1, batch_size=1, device=torch.device("cpu"))
    assert pos_one.shape == (3, 1, 1)
    assert pos_one[0, 0, 0].item() == 35

    logits = torch.arange(2 * 5 * 3).view(2, 5, 3).float()
    # 第一条是 right padding，最后真实位置为 2；第二条是 left padding，最后真实位置为 4。
    # 这个用例防止以后有人重新改回 logits[:, -1, :]，那会在 right padding 上取错。
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]], dtype=torch.long)
    picked = _last_valid_next_logits(logits, mask)
    assert torch.equal(picked[0], logits[0, 2])
    assert torch.equal(picked[1], logits[1, 4])
    print("[test_batched_kv_helpers] ok")


if __name__ == "__main__":
    main()
