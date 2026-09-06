"""生成不超过 30,000,000 字节的审计 ZIP；不打包权重、缓存和完整视频。"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
import os

LIMIT = 30_000_000


def pack(root, output=None, max_bytes=LIMIT):
    """核心指标必须完整；可选案例/历史按预算选入，遗漏写清单，原文件不变。"""
    root = Path(root).resolve()
    output = Path(output or root / "audit.zip").resolve()
    if not 4096 <= max_bytes <= LIMIT:
        raise ValueError("audit cap must be 4096..30,000,000 bytes")
    core_names = (
        "model_contract.json",
        "metrics.json",
        "benchmark_report.json",
        "run_manifest.json",
        "config.json",
        "selected_priors.json",
        "lora/bundle_manifest.json",
        "training_plan.json",
        "route_results.csv",
        "scenario_results.csv",
        "ability_results.csv",
        "paper_table.md",
        "test/metrics.json",
        "probe/metrics.json",
    )
    core = [root / name for name in core_names if (root / name).is_file()]
    patterns = (
        "validation/*.json",
        "epoch_audit/*.json",
        "audit/*.json",
        "test/cases/*.json",
        "probe/cases/*.json",
        "probe/cases/*.png",
        "eval_per_route/*.json",
        "logs/*.log",
        "log.txt",
        "rollouts/*/*/prior_*.json",
        "rollouts/*/*/prior_counts.json",
        "rollouts/*/*/latency.json",
    )
    optional = sorted(
        {
            p
            for pattern in patterns
            for p in root.glob(pattern)
            if p.is_file() and not p.is_symlink()
        }
        - set(core)
    )
    included, skipped = [], []
    # 不可压缩内容也满足上限；预留目录和清单空间，最后另验实际 ZIP 大小。
    budget = int(max_bytes * 0.88)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output.parent, suffix=".zip.tmp")
    os.close(fd)
    try:
        with zipfile.ZipFile(
            tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as z:
            for path in core + optional:
                name = str(path.relative_to(root))
                size = path.stat().st_size
                if path.is_symlink() or size + 512 > budget:
                    if path in core:
                        raise ValueError(
                            f"core audit file exceeds budget: {name}; no incomplete core archive published"
                        )
                    skipped.append(name)
                    continue
                data = path.read_bytes()
                if len(data) + 512 > budget:
                    raise RuntimeError(f"audit input changed while packing: {path}")
                z.writestr(name, data)
                budget -= len(data) + 512
                included.append(
                    dict(
                        path=name,
                        bytes=len(data),
                        sha256=hashlib.sha256(data).hexdigest(),
                    )
                )
            manifest = dict(
                schema="action_audit_zip_v1",
                cap_bytes=max_bytes,
                included=included,
                omitted_optional_count=len(skipped),
                omitted_optional_first_100=skipped[:100],
                excluded_categories=[
                    "weights",
                    "text_cache",
                    "full RGB/video",
                    "raw TensorBoard events (validation JSON preserves metrics)",
                    "full motion telemetry",
                ],
                note="Metrics are complete; cases/history may be sampled by size. Original outputs remain on disk.",
            )
            z.writestr(
                "AUDIT_MANIFEST.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )
        if Path(tmp).stat().st_size > max_bytes:
            raise ValueError("compressed archive exceeds hard cap; nothing published")
        os.replace(tmp, output)
    finally:
        Path(tmp).unlink(missing_ok=True)
    print(
        f"[audit] {output} ({output.stat().st_size} bytes; cap {max_bytes})", flush=True
    )
    return output


def main():
    """可独立重打包已有训练或评测结果。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output")
    args = p.parse_args()
    pack(args.root, args.output)


if __name__ == "__main__":
    main()
