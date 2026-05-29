"""SFT v1 loss_scale sanity check — 在 token 级别可视化 ANALYSIS 段 mask。

不依赖 ms-swift，也不依赖 GPU。只用 HuggingFace tokenizer + 你已经生成的
jsonl 样本，把 assistant content 按字符 → token 映射展开，标出三段：

    [MASK]  对应 ANALYSIS: Observations recorded.\n   （swift loss_scale 应权重 0）
    [LOSS]  对应 STATUS: <event_name>\n               （应算 loss）
    [LOSS]  对应 SUBGOAL: <event_name>                 （应算 loss）

目的不是模拟 swift 内部 loss_scale 算法，而是给你一份"如果 swift 做对了，
这些 token 应该是 mask、那些 token 应该算 loss"的人工对照表。

典型用法（远程或本地都可）：

```bash
# 默认看 train.jsonl 第一条
python AutoMoT/tools/check_loss_mask.py

# 看第 N 条
python AutoMoT/tools/check_loss_mask.py --sample-idx 7

# 指定 tokenizer 目录
python AutoMoT/tools/check_loss_mask.py \
    --tokenizer-dir AutoMoT/checkpoints/Qwen3-VL-4B-Instruct
```

观察要点：
- ANALYSIS 段每个 token 都应被列为 [MASK]，token 数应在 5-10 之间
- STATUS / SUBGOAL 的字面前缀和 event_name token 应被列为 [LOSS]
- 如果 STATUS event_name 只有 1 个 token，说明 BPE 把它当成完整词；
  这是 v1 监督信号最稠密的位置，绝对不能被 mask
- 如果发现 ANALYSIS 段有 token 被标 [LOSS]（或反过来），说明
  PLACEHOLDER_ANALYSIS 跟训练脚本里的 LOSS_SCALE regex 不匹配，必须修
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from typing import Dict, List, Optional, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[1]
_PROJECT_ROOT = _THIS_FILE.parents[2]

# HF 离线开关：tokenizer 也应该只读本地缓存。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# 与 sft_v1_train.sh 中 LOSS_SCALE 完全一致的 regex。一旦其中一边改了，
# 另一边必须同步，否则 check 与训练会失配。
LOSS_SCALE_REGEX = r"ANALYSIS:.*?(?=\nSTATUS:)"


def load_first_sample(jsonl_path: pathlib.Path, idx: int) -> Dict:
    """读 jsonl 第 idx 条样本。"""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == idx:
                return json.loads(line)
    raise IndexError(f"jsonl {jsonl_path} has fewer than {idx+1} samples")


def find_mask_char_range(text: str) -> Optional[Tuple[int, int]]:
    """在 assistant text 上找 ANALYSIS 段的字符 range（与 swift regex 同义）。

    返回 [start, end)，end 是 \nSTATUS: 之前的位置。匹配不到返回 None。
    """
    m = re.search(LOSS_SCALE_REGEX, text, flags=re.DOTALL)
    if m is None:
        return None
    return m.start(), m.end()


def tokenize_with_offsets(tokenizer, text: str) -> Tuple[List[int], List[Tuple[int, int]], List[str]]:
    """对纯文本 tokenize，返回 (ids, offsets, decoded_tokens)。

    offsets[i] = (char_start, char_end)，半开区间。
    decoded_tokens[i] 是每个 token id 单独 decode 出来的字符串（包含前导空格等）。
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    ids = list(enc["input_ids"])
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    decoded = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in ids]
    return ids, offsets, decoded


def classify_tokens(
    offsets: List[Tuple[int, int]],
    mask_range: Optional[Tuple[int, int]],
) -> List[str]:
    """每个 token 标 [MASK] 或 [LOSS]。

    判定规则：token 的字符区间与 mask_range 有非空交集即视为 [MASK]。
    其它都是 [LOSS]。mask_range 为 None 时全部标 [LOSS]（会触发顶部警告）。
    """
    tags: List[str] = []
    if mask_range is None:
        return ["[LOSS]"] * len(offsets)
    ms, me = mask_range
    for (s, e) in offsets:
        if s < me and e > ms:
            tags.append("[MASK]")
        else:
            tags.append("[LOSS]")
    return tags


def print_token_table(
    ids: List[int],
    offsets: List[Tuple[int, int]],
    decoded: List[str],
    tags: List[str],
) -> None:
    """打印 (idx, tag, id, char_range, decoded_repr) 表。"""
    print(f"{'idx':>4} {'tag':<6} {'id':>7} {'char_range':>14}  decoded")
    print("-" * 80)
    for i, (tid, off, tok, tag) in enumerate(zip(ids, offsets, decoded, tags)):
        repr_tok = tok.replace("\n", "\\n").replace("\r", "\\r")
        print(f"{i:>4} {tag:<6} {tid:>7} {f'[{off[0]},{off[1]})':>14}  {repr_tok!r}")


def summarize(tags: List[str], decoded: List[str]) -> Dict[str, int]:
    """汇总 mask / loss token 数；找出 STATUS 与 SUBGOAL event_name 的 token 数。"""
    n_mask = sum(1 for t in tags if t == "[MASK]")
    n_loss = sum(1 for t in tags if t == "[LOSS]")
    # 找 "STATUS:" 在 decoded 列里的位置，event_name 紧跟其后到 \n。
    text_joined = "".join(decoded)
    return {
        "n_mask": n_mask,
        "n_loss": n_loss,
        "joined_length": len(text_joined),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str,
                        default=str(_AUTOMOT_ROOT / "checkpoints" / "sft_v1_data" / "train.jsonl"))
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--tokenizer-dir", type=str,
                        default=str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B-Instruct"))
    args = parser.parse_args()

    jsonl_path = pathlib.Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"[err] jsonl not found: {jsonl_path}", file=sys.stderr)
        sys.exit(2)

    sample = load_first_sample(jsonl_path, args.sample_idx)
    assistant_text = sample["messages"][-1]["content"]
    print(f"[load] jsonl={jsonl_path} sample_idx={args.sample_idx}")
    print(f"[load] scenario={sample.get('scenario')} run_id={sample.get('run_id')}"
          f" anchor={sample.get('anchor')}")
    print()
    print("===== assistant text =====")
    print(assistant_text)
    print("==========================")
    print()

    mask_range = find_mask_char_range(assistant_text)
    if mask_range is None:
        print("[WARN] LOSS_SCALE regex 在 assistant text 上匹配不到。"
              "ANALYSIS 段不会被 mask，训练时整段都会算 loss！")
    else:
        masked_str = assistant_text[mask_range[0]:mask_range[1]]
        print(f"[mask] regex matched chars [{mask_range[0]},{mask_range[1]})  "
              f"-> {masked_str!r}")

    print()
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError:
        print("[err] transformers 未安装。请在远程或装好 transformers 的环境运行：\n"
              "      pip install transformers", file=sys.stderr)
        sys.exit(3)

    tokenizer_dir = pathlib.Path(args.tokenizer_dir)
    if not tokenizer_dir.exists():
        print(f"[err] tokenizer_dir 不存在: {tokenizer_dir}\n"
              f"      可用 --tokenizer-dir 指向有 Qwen3-VL tokenizer 文件的目录。",
              file=sys.stderr)
        sys.exit(4)

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
        trust_remote_code=True,
    )

    ids, offsets, decoded = tokenize_with_offsets(tokenizer, assistant_text)
    tags = classify_tokens(offsets, mask_range)
    print_token_table(ids, offsets, decoded, tags)

    summary = summarize(tags, decoded)
    print()
    print(f"[summary] total tokens = {len(ids)}, "
          f"mask = {summary['n_mask']}, loss = {summary['n_loss']}")
    if summary["n_loss"] < 5:
        print("[WARN] 算 loss 的 token 太少（<5），STATUS/SUBGOAL 监督信号可能稀薄。"
              "确认 PLACEHOLDER_ANALYSIS 是不是把 STATUS 行也吞掉了。")
    if summary["n_mask"] == 0:
        print("[WARN] 没有任何 token 被 mask。检查 LOSS_SCALE_REGEX 与 PLACEHOLDER_ANALYSIS。")


if __name__ == "__main__":
    main()
