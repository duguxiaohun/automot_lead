"""SFT v2 loss_scale 静态 sanity。

目标：在不启 GPU 的前提下，直接可视化 v2 插件的 token 级权重是否符合预期：

1) ANALYSIS 段正文 token 权重应为 0.3
2) STATUS / SUBGOAL 的事件名 token 权重应为 1.0
3) 起手 ANALYSIS: 字面应为 0.0；段切换字面（\nSTATUS: / \nSUBGOAL:）
   应为 1.0；jsonl 若没有 tail/EOS，脚本会额外合成一个 "\n<eos>" tail
   检查 plugin 对 tail/EOS 的权重是否为 1.0。注意：synthetic tail 只验证
   plugin 行为，不证明 ms-swift runtime context 一定包含 EOS。

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
    a_start, a_end = m.span("analysis")
    s_start, s_end = m.span("status")
    g_start, g_end = m.span("subgoal")
    return {
        "prefix": (0, a_start),
        "analysis": (a_start, a_end),
        "mid1": (a_end, s_start),
        "status": (s_start, s_end),
        "mid2": (s_end, g_start),
        "subgoal": (g_start, g_end),
        "tail": (g_end, len(text)),
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
        # 对齐 sft_v2_loss_scale_plugin._split_full:
        # [prefix, analysis, mid1, status, mid2, subgoal, tail]
        # -> [0.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0]
        if (
            overlap(off, ranges["mid1"])
            or overlap(off, ranges["status"])
            or overlap(off, ranges["mid2"])
            or overlap(off, ranges["subgoal"])
            or overlap(off, ranges["tail"])
        ):
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

    # 字面检查（2026-06-02 修订，对齐 plugin 段切换权重升级）：
    #
    #   "ANALYSIS:" —— assistant prefix 起手字面，仍 mask（weight=0）；in_loss=True 才报警。
    #   "STATUS:" / "SUBGOAL:" —— 段切换字面前面带 "\n"，现在 weight=1.0；预期 in_loss=True，
    #                           只有当它们落到 mask 段（in_mask=True）才报警。
    #
    # 历史踩坑：v2.0 plugin 把段切换字面 mask=0，自由生成陷入 "ANALYSIS×N 循环复读"，
    # 详见 PROJECT_CONTEXT.md §18.5。
    LITERAL_CHECKS = (
        # (字面, 预期 in_loss, 不符合时的 WARN 消息)
        ("ANALYSIS:", False, "字面 'ANALYSIS:' 不应进入 loss（assistant prefix 起手，weight=0）"),
        ("STATUS:",   True,  "字面 'STATUS:' 应该进入 loss（段切换字面 weight=1.0，2026-06-02 修订）"),
        ("SUBGOAL:",  True,  "字面 'SUBGOAL:' 应该进入 loss（段切换字面 weight=1.0，2026-06-02 修订）"),
    )
    for literal, expect_in_loss, warn_msg in LITERAL_CHECKS:
        in_loss = literal in loss_text
        in_mask = literal in masked_text
        print(f"[plugin] literal={literal!r} in_loss={in_loss} in_mask={in_mask} expect_in_loss={expect_in_loss}")
        if in_loss != expect_in_loss:
            print(f"[WARN] {warn_msg}")

    # 目标切片通常是 6 段（无 tail）或 7 段（有 tail）。
    if len(parts) < 6 or len(parts) > 7:
        print(f"[WARN] 分段数量异常: len(parts)={len(parts)}，期望 6 或 7")
    print("================================")


def print_synthetic_eos_check(text: str, tokenizer) -> None:
    """jsonl assistant content 通常不含 chat template 的 EOS。

    为了验证 plugin 的 tail/EOS 权重，不把 EOS 硬塞进 SUBGOAL 同一行，而是在
    assistant 后追加一行 tokenizer.eos_token（通常是 ``<|im_end|>``）。这样
    ``_FULL_PATTERN`` 的 subgoal 捕获仍停在原事件名，追加部分会落入 tail。
    这个检查只验证 plugin 对 tail 的权重规则；真实训练时 EOS 是否进入 context
    由 ms-swift 的 chat template / runtime 拼接决定。若 runtime 把 EOS 贴在
    SUBGOAL 同一行，它会落入 subgoal 段，权重同样是 1.0；这里专门覆盖换行 tail。
    """
    ranges = find_weight_ranges(text)
    if not ranges:
        return
    tail_start, tail_end = ranges["tail"]
    if tail_end > tail_start:
        print("[tail/eos] 原始 assistant 已含 tail；上面的 token/plugin 表已覆盖 tail 权重。")
        return

    eos_token = getattr(tokenizer, "eos_token", None) or "<|im_end|>"
    synthetic = text.rstrip() + "\n" + eos_token
    synthetic_ranges = find_weight_ranges(synthetic)
    if not synthetic_ranges:
        print("[tail/eos][WARN] 合成 EOS tail 后 FULL_PATTERN 反而匹配失败")
        return
    s, e = synthetic_ranges["tail"]
    tail_text = synthetic[s:e]
    print()
    print("===== synthetic tail/EOS sanity =====")
    print("[tail/eos] synthetic-only: 验证 plugin tail 权重，不等价于断言 ms-swift runtime 一定传入 EOS。")
    print("[tail/eos] note: 若 runtime 把 EOS 贴在 SUBGOAL 同一行，它会落入 subgoal 段，仍是 W1.0。")
    print(f"[tail/eos] appended={eos_token!r} tail chars [{s},{e}) -> {tail_text!r}")

    ids, offsets, decoded = tokenize_with_offsets(tokenizer, synthetic)
    weights = classify_weights(offsets, synthetic_ranges)
    tail_rows = [
        (i, tid, off, tok, w)
        for i, (tid, off, tok, w) in enumerate(zip(ids, offsets, decoded, weights))
        if overlap(off, synthetic_ranges["tail"])
    ]
    if not tail_rows:
        print("[tail/eos][WARN] tokenizer offset 中没有落入 tail 的 token")
    for i, tid, off, tok, w in tail_rows:
        repr_tok = tok.replace("\n", "\\n").replace("\r", "\\r")
        print(f"[tail/eos] idx={i} {tag_from_weight(w)} w={w:.1f} id={tid} chars=[{off[0]},{off[1]}) decoded={repr_tok!r}")
    if any(w < 0.99 for _, _, _, _, w in tail_rows):
        print("[tail/eos][WARN] tail/EOS token 未全部按 W1.0 标注")
    print_plugin_check(synthetic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        type=str,
        default=str(_AUTOMOT_ROOT / "checkpoints" / "sft_v2_lora" / "runtime_teacher_data" / "train.jsonl"),
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
    if sample.get("dataset_version") == "v2_pending":
        print("[err] check_loss_mask_v2.py 需要已物化 teacher ANALYSIS 的 v2 jsonl，不能直接检查 v2_pending 占位数据。", file=sys.stderr)
        print("[hint] 先跑 bash tools/sft_v2_train.sh check，或手动运行 tools/build_sft_dataset_v2_teacher.py 生成 runtime jsonl。", file=sys.stderr)
        sys.exit(2)
    assistant = sample["messages"][-1]["content"]
    print(f"[load] jsonl={jsonl_path} sample_idx={args.sample_idx}")
    print(f"[load] dataset_version={sample.get('dataset_version')} scenario={sample.get('scenario')} run_id={sample.get('run_id')} anchor={sample.get('anchor')}")
    print()
    print("===== assistant text =====")
    print(assistant)
    print("==========================")

    ranges = find_weight_ranges(assistant)
    if not ranges:
        print("[WARN] FULL_PATTERN 匹配不到，静态 token 表无法按 full pattern 分类；请以下方 plugin sanity 的真实切片为准。")
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
    if stat["n_w0"] < 1:
        print("[WARN] 权重 0.0 token 太少，起手 ANALYSIS: prefix 可能误入监督")

    print_plugin_check(assistant)
    print_synthetic_eos_check(assistant, tokenizer)


if __name__ == "__main__":
    main()
