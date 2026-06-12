"""SFT loss-mask 静态自检（无 GPU）。

可视化 train.py 内置 per-token 权重切法是否符合预期：

1) ANALYSIS body token 权重应为 SFT_ANALYSIS_WEIGHT（默认 0.3）
2) STATUS / SUBGOAL 事件名 token 权重应为 1.0
3) 起手 "ANALYSIS:" 字面、段切换 "\\nSTATUS:" / "\\nSUBGOAL:" 字面、末尾 tail / EOS
   全部权重 1.0（与 train.py 同口径，避免段切换无监督导致循环复读）

与历史 ms-swift check 脚本的差别：
- 不再调用 loss_scale plugin（train.py 不再走 swift）。
- 直接复用 train.py 的 _FULL_ASSIST_RE + 权重表语义，运行环境只需要 tokenizer。

典型用法（从 AutoMoT/ 目录运行）：

```bash
python qwen3vl_local/sft/check_loss_mask.py \
  --jsonl checkpoints/sft_data_pending/train.jsonl \
  --sample-idx 0

SFT_ANALYSIS_WEIGHT=0.5 python qwen3vl_local/sft/check_loss_mask.py \
  --jsonl checkpoints/sft_data_pending/train.jsonl \
  --sample-idx 12
```
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from typing import Dict, List, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


ANALYSIS_WEIGHT = float(os.environ.get("SFT_ANALYSIS_WEIGHT", "0.3"))

# 与 train.py::_FULL_ASSIST_RE 严格保持一致。
FULL_PATTERN = re.compile(
    r"ANALYSIS:[ \t]*"
    r"(?P<analysis>[^\n]*?)"
    r"\s*\nSTATUS:[ \t]*"
    r"(?P<status>\S[^\n]*?)"
    r"\s*\nSUBGOAL:[ \t]*"
    r"(?P<subgoal>\S[^\n]*)",
    flags=re.DOTALL,
)


def load_one_sample(jsonl_path: pathlib.Path, idx: int) -> Dict:
    """从 jsonl 中读取第 idx 条样本；idx 越界时给出清晰错误。"""

    with open(jsonl_path, "r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]
    if idx < 0 or idx >= len(rows):
        raise IndexError(f"sample-idx 越界: idx={idx}, total={len(rows)}")
    return json.loads(rows[idx])


def find_weight_ranges(text: str) -> Dict[str, Tuple[int, int]]:
    """按 ANALYSIS / STATUS / SUBGOAL 三段结构返回 char 范围。"""

    m = FULL_PATTERN.search(text)
    if m is None:
        return {}
    a_start, a_end = m.span("analysis")
    s_start, s_end = m.span("status")
    g_start, g_end = m.span("subgoal")
    return {
        "prefix":   (0, a_start),
        "analysis": (a_start, a_end),
        "mid1":     (a_end, s_start),
        "status":   (s_start, s_end),
        "mid2":     (s_end, g_start),
        "subgoal":  (g_start, g_end),
        "tail":     (g_end, len(text)),
    }


def tokenize_with_offsets(tokenizer, text: str):
    """对 assistant 文本做 tokenize，同时返回 token id、字符 offset 和可读 token。"""

    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = list(enc["input_ids"])
    offsets = [tuple(x) for x in enc["offset_mapping"]]
    decoded = [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in ids]
    return ids, offsets, decoded


def _overlap(off: Tuple[int, int], r: Tuple[int, int]) -> bool:
    """判断 token 的字符 offset 是否和某个范围相交。"""

    return off[0] < r[1] and off[1] > r[0]


def classify_weights(offsets, ranges: Dict[str, Tuple[int, int]]) -> List[float]:
    """与 train.py.build_student_inputs 的权重表完全一致：

    ANALYSIS body = ANALYSIS_WEIGHT；其余 assistant 段（含 prefix / mid /
    status / subgoal / tail）= 1.0。
    """

    if not ranges:
        return [0.0] * len(offsets)
    out: List[float] = []
    for off in offsets:
        if _overlap(off, ranges["analysis"]):
            out.append(ANALYSIS_WEIGHT)
        elif (_overlap(off, ranges["prefix"]) or _overlap(off, ranges["mid1"])
              or _overlap(off, ranges["status"]) or _overlap(off, ranges["mid2"])
              or _overlap(off, ranges["subgoal"]) or _overlap(off, ranges["tail"])):
            out.append(1.0)
        else:
            out.append(0.0)
    return out


def tag_from_weight(w: float) -> str:
    """把数值权重转成表格里易扫的短标签。"""

    if w >= 0.99:
        return "[W1.0]"
    if abs(w - ANALYSIS_WEIGHT) < 1e-3:
        return f"[W{ANALYSIS_WEIGHT:.1f}]"
    return "[W0.0]"


def print_token_table(ids, offsets, decoded, weights) -> None:
    """打印 token 级权重表，人工检查 mask 是否切到预期片段。"""

    print(f"{'idx':>4} {'tag':<7} {'w':>5} {'id':>7} {'char_range':>14}  decoded")
    print("-" * 92)
    for i, (tid, off, tok, w) in enumerate(zip(ids, offsets, decoded, weights)):
        repr_tok = tok.replace("\n", "\\n").replace("\r", "\\r")
        print(f"{i:>4} {tag_from_weight(w):<7} {w:>5.1f} {tid:>7} {f'[{off[0]},{off[1]})':>14}  {repr_tok!r}")


def main() -> None:
    """命令行入口：加载样本和 tokenizer，打印三段范围与 token 权重表。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        type=str,
        default=str(_AUTOMOT_ROOT / "checkpoints" / "sft_data_pending" / "train.jsonl"),
        help="可以是 pending jsonl（assistant 含 __TEACHER_PENDING__）或 build_teacher 物化后的 materialized jsonl。",
    )
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B-Instruct"),
    )
    args = parser.parse_args()

    jsonl_path = pathlib.Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"[err] jsonl 不存在: {jsonl_path}", file=sys.stderr)
        sys.exit(2)

    sample = load_one_sample(jsonl_path, args.sample_idx)
    assistant = sample["messages"][-1]["content"]
    print(f"[load] jsonl={jsonl_path} sample_idx={args.sample_idx}")
    print(f"[load] dataset_version={sample.get('dataset_version')} scenario={sample.get('scenario')} "
          f"run_id={sample.get('run_id')} anchor={sample.get('anchor')}")
    print()
    print("===== assistant text =====")
    print(assistant)
    print("==========================")

    ranges = find_weight_ranges(assistant)
    if not ranges:
        print("[WARN] FULL_PATTERN 匹配不到三段；assistant 文本可能损坏。")
    else:
        for name in ("prefix", "analysis", "mid1", "status", "mid2", "subgoal", "tail"):
            s, e = ranges[name]
            print(f"[range] {name:8s} chars [{s},{e}) -> {assistant[s:e]!r}")

    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError:
        print("[err] 缺少 transformers，请先安装", file=sys.stderr)
        sys.exit(3)

    tokenizer_dir = pathlib.Path(args.tokenizer_dir)
    if not tokenizer_dir.exists():
        print(f"[err] tokenizer_dir 不存在: {tokenizer_dir}", file=sys.stderr)
        sys.exit(4)

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    # return_offsets_mapping=True 是 Fast tokenizer 独占功能。
    # train.py 也加了同样 assert，这里再守一遍，避免静态自检跑下来误以为通过。
    assert getattr(tokenizer, "is_fast", False), (
        "check_loss_mask 需要 Fast tokenizer (PreTrainedTokenizerFast)；"
        "当前 tokenizer 不是 Fast 版本，return_offsets_mapping 无法工作。"
    )

    ids, offsets, decoded = tokenize_with_offsets(tokenizer, assistant)
    weights = classify_weights(offsets, ranges)
    print()
    print_token_table(ids, offsets, decoded, weights)

    n_w1 = sum(1 for x in weights if x >= 0.99)
    n_wa = sum(1 for x in weights if abs(x - ANALYSIS_WEIGHT) < 1e-3)
    n_w0 = sum(1 for x in weights if x < min(0.99, ANALYSIS_WEIGHT) - 1e-3)
    print()
    print(f"[summary] total={len(weights)} w1={n_w1} w{ANALYSIS_WEIGHT:.1f}={n_wa} w0={n_w0}")
    if n_w1 < 2:
        print("[WARN] 权重 1.0 token 太少（<2），事件名可能被切错")
    if n_wa < 2:
        print("[WARN] 权重 ANALYSIS_WEIGHT token 太少，ANALYSIS body 可能没被切出来")
    if n_w0 > 0:
        print("[WARN] 出现 W0.0 token；assistant 全段应在 loss 内，请检查 prompt / FULL_PATTERN")


if __name__ == "__main__":
    main()
