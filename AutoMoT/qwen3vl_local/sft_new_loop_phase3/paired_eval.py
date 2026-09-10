"""配对评测已有生成结果；拒绝样本/真值/输入错配，不把重标注收益算模型提升。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.sft_new_loop_phase3.audit_temporal_slices import slices


def temporal_slices(row):
    """将生成结果适配到同一诊断口径，不能把字符串 NO 当作布尔 True。"""
    return slices(dict(history_rgb_paths=row["history_rgb_paths_all4"],
        invalid_action_context=row["gt"].get("INVALID_ACTION_CONTEXT") == "YES",
        action_evidence=row["action_evidence"],
        answers={key: value == "YES" for key,value in row["gt"].items()}))


def summary(counts):
    """分母包括两侧都错的样本；空切片不制造 0% 准确率。"""
    n = sum(counts.values())
    return dict(cases=n, transitions=dict(counts),
        left_exact=(counts["left_1_right_0"]+counts["left_1_right_1"])/n if n else None,
        right_exact=(counts["left_0_right_1"]+counts["left_1_right_1"])/n if n else None,
        exact_delta=(counts["left_0_right_1"]-counts["left_1_right_0"])/n if n else None)


def identity(row):
    return tuple(row[k] for k in ("scenario", "route_id", "frame_id", "context_id",
        "prompt_road_structure", "augment_variant")) + (tuple(row["gt"]),)


def load_cases(path):
    path = Path(path)
    files = sorted(path.glob("cases*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"no case files: {path}")
    rows = {}
    hashes = {}
    for file in files:
        hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        for line in file.read_text().splitlines():
            row = json.loads(line)
            key = identity(row)
            if key in rows:
                raise ValueError(f"duplicate paired case: {key}")
            rows[key] = row
    return rows, hashes


def compare(left, right):
    """全量配对，输入和答案不一致时立即失败；不只保留容易命中的交集。"""
    if not left or left.keys() != right.keys():
        raise ValueError(f"case set mismatch: left_only={len(left.keys()-right.keys())}, "
                         f"right_only={len(right.keys()-left.keys())}; empty={not left}")
    counts = Counter()
    contexts = {}
    temporal = {}
    groups = {}
    signatures = {}
    hash_verified = 0
    for key, a in left.items():
        b = right[key]
        for field in ("gt", "true_rs", "goal_ego_xy", "context_detail", "history_rgb_mode",
                      "history_rgb_selected_indices"):
            if a.get(field) != b.get(field):
                raise ValueError(f"paired {field} mismatch: {key}")
        # 支持数据根目录搬迁，保留 scenario/run/rgb/frame 身份。
        paths = lambda row: [tuple(Path(p).parts[-4:]) for p in row["history_rgb_paths_used"]]
        if a.get("history_rgb_sha256") is not None or b.get("history_rgb_sha256") is not None:
            if (not a.get("history_rgb_sha256") or not b.get("history_rgb_sha256")
                    or a["history_rgb_sha256"] != b["history_rgb_sha256"]):
                raise ValueError(f"paired RGB bytes mismatch or missing fingerprint: {key}")
            hash_verified += 1
        if paths(a) != paths(b):
            raise ValueError(f"paired RGB mismatch: {key}")
        if a["action_evidence"].get("future_speeds_exact_mps") != b["action_evidence"].get("future_speeds_exact_mps"):
            raise ValueError(f"paired source trajectory mismatch: {key}")
        # 严格 parser 的失败必须保留；不使用 answer-only 的宽松匹配。
        ok_a = bool(a["all_ok"])
        ok_b = bool(b["all_ok"])
        category = f"left_{int(ok_a)}_right_{int(ok_b)}"
        counts[category] += 1
        contexts.setdefault(a["context_id"], Counter())[category] += 1
        # 旧包也能用原始精度证据切片；缺字段时明确列为 unavailable。
        flags = temporal_slices(a) if "history_rgb_paths_all4" in a else ["unavailable"]
        for flag in flags or ["no_diagnostic_flag"]:
            temporal.setdefault(flag, Counter())[category] += 1
        group = "invalid" if a["gt"].get("INVALID_ACTION_CONTEXT") == "YES" else "valid"
        groups.setdefault(group, Counter())[category] += 1
        signature = a.get("action_signature", "unavailable")
        signatures.setdefault(signature, Counter())[category] += 1
    n = len(left)
    return dict(paired_cases=n, left_exact=(counts["left_1_right_0"]+counts["left_1_right_1"])/n,
        right_exact=(counts["left_0_right_1"]+counts["left_1_right_1"])/n,
        exact_delta=(counts["left_0_right_1"]-counts["left_1_right_0"])/n,
        transitions=dict(counts), by_context={k:dict(v) for k,v in contexts.items()},
        context_metrics={k:summary(v) for k,v in contexts.items()},
        temporal_slice_metrics={k:summary(v) for k,v in temporal.items()},
        validity_metrics={k:summary(v) for k,v in groups.items()},
        action_signature_metrics={k:summary(v) for k,v in signatures.items()},
        temporal_slices_overlap=True,
        excluded_cases=0, rgb_hash_verified_cases=hash_verified,
        rgb_check="sha256" if hash_verified == n else "legacy path identity only; bytes not verified",
        interpretation="Paired inference comparison only; this tool does not verify equal training budgets.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", required=True, type=Path)
    p.add_argument("--right", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    left, lh = load_cases(args.left)
    right, rh = load_cases(args.right)
    report = compare(left, right)
    report.update(left=str(args.left), right=str(args.right), source_sha256={**lh, **rh})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in (
        "source_sha256", "by_context", "context_metrics", "temporal_slice_metrics",
        "validity_metrics", "action_signature_metrics")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
