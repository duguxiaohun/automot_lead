"""SFT v2 loss_scale 静态 sanity。

目标：在不启 GPU 的前提下，直接可视化 v2 插件的 token 级权重是否符合预期：

1) ANALYSIS 段正文 token 权重应为 0.3
2) STATUS / SUBGOAL 的事件名 token 权重应为 1.0
3) 其余字面（ANALYSIS:/STATUS:/SUBGOAL:）与空白应为 0.0

并且会调用 tools/sft_v2_loss_scale_plugin.py 的真实插件类做一次切片校验，
确保训练侧不会出现“本地正则看起来对，swift 实际走了 fallback”这种分歧。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[1]

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# 与 sft_v2_loss_scale_plugin.py 主正则语义保持一致。
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
    with open(jsonl_path, "r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]
    if idx < 0 or idx >= len(rows):
        raise IndexError(f"sample-idx 越界: idx={idx}, total={len(rows)}")
    return json.loads(rows[idx])


def parse_assistant(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    a = re.search(r"^ANALYSIS:\s*(.*)$", text, flags=re.MULTILINE)
    s = re.search(r"^STATUS:\s*(.+)$", text, flags=re.MULTILINE)
    g = re.search(r"^SUBGOAL:\s*(.+)$", text, flags=re.MULTILINE)
    analysis = a.group(1).strip() if a else None
    status = s.group(1).strip() if s else None
    subgoal = g.group(1).strip() if g else None
    return analysis, status, subgoal


def find_weight_ranges(text: str) -> Dict[str, Tuple[int, int]]:
    m = FULL_PATTERN.search(text)
    if m is None:
        return {}
    return {
        "analysis": m.span("analysis"),
        "status": m.span("status"),
        "subgoal": m.span("subgoal"),
    }


def tokenize_with_offsets(tokenizer, text: str) -> Tuple[List[int], List[Tuple[int, int]], List[str]]:
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    ids = list(enc["input_ids"])
    offsets = [tuple(x) for x in enc["offset_mapping"]]
    decoded = [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in ids]
    return ids, offsets, decoded


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and a[1] > b[0]


def classify_weights(
    offsets: List[Tuple[int, int]],
    ranges: Dict[str, Tuple[int, int]],
) -> List[float]:
    if not ranges:
        return [0.0] * len(offsets)

    out: List[float] = []
    for off in offsets:
        # 优先级：1.0 > 0.3 > 0.0
        if overlap(off, ranges["status"]) or overlap(off, ranges["subgoal"]):
            out.append(1.0)
        elif overlap(off, ranges["analysis"]):
            out.append(0.3)
        else:
            out.append(0.0)
    return out


def tag_from_weight(w: float) -> str:
    if w >= 0.99:
        return "[W1.0]"
    if 0.29 <= w <= 0.31:
        return "[W0.3]"
    return "[W0.0]"


def print_token_table(
    ids: List[int],
    offsets: List[Tuple[int, int]],
    decoded: List[str],
    weights: List[float],
) -> None:
    print(f"{'idx':>4} {'tag':<7} {'w':>5} {'id':>7} {'char_range':>14}  decoded")
    print("-" * 92)
    for i, (tid, off, tok, w) in enumerate(zip(ids, offsets, decoded, weights)):
        repr_tok = tok.replace("\n", "\\n").replace("\r", "\\r")
        tag = tag_from_weight(w)
        print(f"{i:>4} {tag:<7} {w:>5.1f} {tid:>7} {f'[{off[0]},{off[1]})':>14}  {repr_tok!r}")


def summarize(weights: List[float]) -> Dict[str, int]:
    n_w1 = sum(1 for x in weights if x >= 0.99)
    n_w03 = sum(1 for x in weights if 0.29 <= x <= 0.31)
    n_w0 = sum(1 for x in weights if x < 0.29)
    return {"n_w1": n_w1, "n_w03": n_w03, "n_w0": n_w0}


def print_plugin_check(text: str) -> None:
    print()
    print("===== plugin sanity (v2) =====")
    plugin_path = _THIS_FILE.with_name("sft_v2_loss_scale_plugin.py")
    try:
        spec = importlib.util.spec_from_file_location("sft_v2_loss_scale_plugin", plugin_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load plugin spec: {plugin_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        loss_scale = module.SftV2AnalysisSupervisedLossScale()
        parts, scales = loss_scale.get_loss_scale(text)
        analysis_w = float(module.ANALYSIS_WEIGHT)
        status_w = float(module.STATUS_WEIGHT)
        subgoal_w = float(module.SUBGOAL_WEIGHT)
    except Exception as exc:
        print(f"[WARN] 无法加载/调用 v2 插件: {exc!r}")
        print("================================")
        return

    joined = "".join(part for part in parts if isinstance(part, str))
    if joined != text:
        print("[WARN] ''.join(parts) != 原始 context，切片有漂移")

    cursor = 0
    for i, (part, scale) in enumerate(zip(parts, scales)):
        start = text.find(part, cursor) if isinstance(part, str) else -1
        end = start + len(part) if start >= 0 and isinstance(part, str) else -1
        if end >= 0:
            cursor = end
        preview = part.replace("\n", "\\n") if isinstance(part, str) else repr(part)
        print(f"[plugin] seg={i} w={float(scale):.2f} chars=[{start},{end}) text={preview!r}")

    loss_text = "".join(part for part, scale in zip(parts, scales)
                        if isinstance(part, str) and float(scale) > 0)
    masked_text = "".join(part for part, scale in zip(parts, scales)
                          if isinstance(part, str) and abs(float(scale)) < 1e-8)

    analysis, status, subgoal = parse_assistant(text)
    checks: Sequence[Tuple[str, Optional[str], float]] = (
        ("ANALYSIS body", analysis, analysis_w),
        ("STATUS event_name", status, status_w),
        ("SUBGOAL event_name", subgoal, subgoal_w),
    )
    for label, value, target_w in checks:
        if not value:
            print(f"[WARN] {label} 解析失败")
            continue
        in_loss = value in loss_text
        in_mask = value in masked_text
        print(f"[plugin] {label}={value!r} target_w={target_w:.2f} in_loss={in_loss} in_mask={in_mask}")
        if target_w > 0 and (not in_loss or in_mask):
            print(f"[WARN] {label} 没有按预期进入 loss 区域")

    for literal in ("ANALYSIS:", "STATUS:", "SUBGOAL:"):
        in_loss = literal in loss_text
        in_mask = literal in masked_text
        print(f"[plugin] literal={literal!r} in_loss={in_loss} in_mask={in_mask}")
        if in_loss:
            print(f"[WARN] 字面 {literal!r} 不应进入 loss")

    # 目标切片通常是 6 段（无 tail）或 7 段（有 tail）。
    if len(parts) < 6 or len(parts) > 7:
        print(f"[WARN] 分段数量异常: len(parts)={len(parts)}，期望 6 或 7")
    print("================================")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        type=str,
        default=str(_AUTOMOT_ROOT / "checkpoints" / "sft_v2_data" / "train.jsonl"),
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
    print(f"[load] dataset_version={sample.get('dataset_version')} scenario={sample.get('scenario')} run_id={sample.get('run_id')} anchor={sample.get('anchor')}")
    print()
    print("===== assistant text =====")
    print(assistant)
    print("==========================")

    ranges = find_weight_ranges(assistant)
    if not ranges:
        print("[WARN] FULL_PATTERN 匹配不到，token 会全部按 0.0 处理")
    else:
        for name in ("analysis", "status", "subgoal"):
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

    ids, offsets, decoded = tokenize_with_offsets(tokenizer, assistant)
    weights = classify_weights(offsets, ranges)
    print()
    print_token_table(ids, offsets, decoded, weights)

    stat = summarize(weights)
    print()
    print(f"[summary] total={len(weights)} w1={stat['n_w1']} w03={stat['n_w03']} w0={stat['n_w0']}")
    if stat["n_w1"] < 2:
        print("[WARN] 权重 1.0 token 太少（<2），事件名可能被切错")
    if stat["n_w03"] < 3:
        print("[WARN] 权重 0.3 token 太少（<3），ANALYSIS 正文可能没被纳入监督")
    if stat["n_w0"] < 5:
        print("[WARN] 权重 0.0 token 太少，字面/空白可能误入监督")

    print_plugin_check(assistant)


if __name__ == "__main__":
    main()
