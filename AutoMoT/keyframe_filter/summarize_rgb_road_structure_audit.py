#!/usr/bin/env python3
"""Summarize full-frame RGB blind ROAD_STRUCTURE audits by scenario.

Input is the output directory from ``rgb_blind_rs_event_audit.py``. The summary
compares three things:

1. Scenario candidate pool from ``collector.SCENARIO_TO_ROAD_STRUCTURE``.
2. Current per-frame rule labels produced during the audit.
3. RGB-blind R4/R5 cues from the audit.

The RGB blind pass is intentionally conservative. It is useful for finding
missing/excess junction labels, while R2/R3 still require scenario-aware RGB
review and topology/meta evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

KEYFRAME_DIR = Path(__file__).resolve().parent
AUTOMOT_ROOT = KEYFRAME_DIR.parent
if str(KEYFRAME_DIR) not in sys.path:
    sys.path.insert(0, str(KEYFRAME_DIR))
if str(AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOMOT_ROOT))

from collector import SCENARIO_TO_ROAD_STRUCTURE  # noqa: E402


RS_ORDER = ["R1", "R2", "R3", "R4", "R5"]
STRONG_MISMATCH_PREFIXES = (
    "blind_R4_label_",
    "blind_R5_label_",
    "label_R4_without_rgb_junction_signal",
    "label_R5_without_rgb_junction_signal",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _counter_from_dict(value: dict[str, Any] | None) -> Counter:
    out: Counter = Counter()
    for key, count in (value or {}).items():
        try:
            out[str(key)] += int(count)
        except Exception:
            continue
    return out


def _pct(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count * 100.0 / total, 3)


def _dist_with_pct(counter: Counter, total: int) -> dict[str, dict[str, float | int]]:
    return {
        label: {"frames": int(counter.get(label, 0)), "pct": _pct(int(counter.get(label, 0)), total)}
        for label in RS_ORDER
        if int(counter.get(label, 0)) > 0
    }


def _labels_with_min_pct(counter: Counter, total: int, min_pct: float) -> set[str]:
    return {label for label, count in counter.items() if _pct(int(count), total) >= min_pct}


def _candidate_notes(
    scenario: str,
    allowed: set[str],
    label_rs: Counter,
    blind_rs: Counter,
    mismatch: Counter,
    total: int,
    min_pct: float,
) -> tuple[list[str], list[str], list[str]]:
    observed = _labels_with_min_pct(label_rs, total, min_pct)
    blind_observed = _labels_with_min_pct(blind_rs, total, min_pct)
    missing: list[str] = []
    extra: list[str] = []
    review: list[str] = []

    for label in sorted(observed - allowed, key=RS_ORDER.index):
        missing.append(f"{label}: current labels use it but scenario candidate pool excludes it")

    for label in sorted(allowed - observed, key=RS_ORDER.index):
        # Do not overreact to rare routes absent from the current data subset.
        extra.append(f"{label}: allowed but not used above {min_pct}% in current full-frame labels")

    for label in ("R4", "R5"):
        if label in blind_observed and label not in allowed:
            missing.append(f"{label}: RGB blind sees >= {min_pct}% but candidate pool excludes it")

    for key, count in mismatch.most_common():
        if count <= 0:
            continue
        if not key.startswith(STRONG_MISMATCH_PREFIXES):
            continue
        review.append(f"{key}={count}")

    # Scenario-family sanity checks that RGB blind cannot infer directly.
    if scenario.endswith("TwoWays"):
        if "R1" in allowed:
            review.append("TwoWays still allows R1; check whether effective drivable road should be R2/R4/R5 only")
        if scenario == "VehicleOpensDoorTwoWays" and "R1" in allowed:
            review.append("VehicleOpensDoorTwoWays still allows R1 primary; current taxonomy wants R2 when parking/door blocks side lanes")
    if scenario in {"EnterActorFlow", "EnterActorFlowV2", "HighwayExit", "MergerIntoSlowTrafficV2"}:
        if allowed != {"R3"}:
            review.append("highway-only scenario should be checked before allowing non-R3 labels")

    return missing, extra, review


def summarize(input_dir: Path, output_dir: Path, min_pct: float) -> dict[str, Any]:
    route_rows = _load_json(input_dir / "route_blind_rs_event_audit.json")
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in route_rows:
        by_scenario[str(row.get("scenario", ""))].append(row)

    scenario_rows: list[dict[str, Any]] = []
    for scenario in sorted(by_scenario):
        rows = by_scenario[scenario]
        status = Counter(str(row.get("status", "")) for row in rows)
        blind_rs: Counter = Counter()
        label_rs: Counter = Counter()
        mismatch: Counter = Counter()
        frames = 0
        label_frames = 0
        for row in rows:
            frames += int(row.get("frame_count") or 0)
            label_frames += int(row.get("label_frame_count") or 0)
            blind_rs.update(_counter_from_dict(row.get("blind_rs_distribution")))
            label_rs.update(_counter_from_dict(row.get("label_rs_distribution")))
            mismatch.update(_counter_from_dict(row.get("mismatch_counts")))
        allowed = {rs.value for rs in SCENARIO_TO_ROAD_STRUCTURE.get(scenario, [])}
        missing, extra, review = _candidate_notes(
            scenario,
            allowed,
            label_rs,
            blind_rs,
            mismatch,
            max(label_frames, 1),
            min_pct,
        )
        scenario_rows.append(
            {
                "scenario": scenario,
                "routes": len(rows),
                "status_counts": dict(sorted(status.items())),
                "frames_rgb_read": frames,
                "frames_labeled": label_frames,
                "allowed_rs": sorted(allowed, key=RS_ORDER.index),
                "label_rs_distribution": _dist_with_pct(label_rs, max(label_frames, 1)),
                "blind_rgb_rs_distribution": _dist_with_pct(blind_rs, max(frames, 1)),
                "candidate_missing_or_too_narrow": missing,
                "candidate_extra_or_unused": extra,
                "review_flags": review[:12],
                "top_mismatch_counts": dict(mismatch.most_common(12)),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "scenario_road_structure_candidate_audit.json"
    json_path.write_text(json.dumps(scenario_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "scenario_road_structure_candidate_audit.csv"
    fields = [
        "scenario",
        "routes",
        "frames_labeled",
        "allowed_rs",
        "label_rs_distribution",
        "blind_rgb_rs_distribution",
        "candidate_missing_or_too_narrow",
        "candidate_extra_or_unused",
        "review_flags",
        "top_mismatch_counts",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in scenario_rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(field), (dict, list))
                    else row.get(field)
                    for field in fields
                }
            )

    md_path = output_dir / "scenario_road_structure_candidate_audit.md"
    lines = [
        "# Scenario ROAD_STRUCTURE Candidate Audit",
        "",
        f"Source: `{input_dir}`",
        "",
        "RGB blind cues are conservative and mainly validate R4/R5. Use label distributions plus RGB sheets for R2/R3.",
        "",
        "| Scenario | Allowed | Current label distribution | RGB blind distribution | Candidate notes | Review flags |",
        "|---|---|---|---|---|---|",
    ]
    for row in scenario_rows:
        notes = row["candidate_missing_or_too_narrow"] or row["candidate_extra_or_unused"]
        lines.append(
            "| {scenario} | {allowed} | {label} | {blind} | {notes} | {review} |".format(
                scenario=row["scenario"],
                allowed=", ".join(row["allowed_rs"]),
                label=json.dumps(row["label_rs_distribution"], ensure_ascii=False),
                blind=json.dumps(row["blind_rgb_rs_distribution"], ensure_ascii=False),
                notes="<br>".join(notes[:8]) if notes else "",
                review="<br>".join(row["review_flags"][:8]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "md": str(md_path), "scenarios": len(scenario_rows)}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-pct", type=float, default=0.05, help="Minimum percentage to treat a label as observed.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = args.output_dir or args.input_dir
    result = summarize(args.input_dir, output_dir, args.min_pct)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
