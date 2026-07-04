# RGB R4/R5 Audit Results

This directory preserves the key pushable artifacts from the full RGB R4/R5 audit described in
`../RGB_R4_R5_AUDIT_SUMMARY.md`.

Source run:

```bash
python AutoMoT/keyframe_filter/rgb_r4_r5_audit.py \
  --output-dir /tmp/automot_rgb_r4_r5_full_audit \
  --workers 16 \
  --progress-interval 100
```

Scope:

- 43 scenarios discovered.
- 9715 routes discovered.
- 8752 valid routes analyzed after abnormal-duration filtering.
- 1102886 stitched RGB frames read and analyzed.
- 43 scenario evidence sheets preserved.

Files:

- `scenario_rgb_r4_r5_summary.csv`: compact per-scenario RGB class, ratios, and route class counts.
- `scenario_rgb_r4_r5_summary.json`: JSON version of the per-scenario summary.
- `route_rgb_r4_r5_audit.csv`: route-level class and aggregate frame counters.
- `rule_update_decisions.csv`: hand-curated rule-update decisions derived from the full audit.
- `evidence_sheets/*.jpg`: per-scenario visual evidence sheets for quick manual review.
- `manifest.json`: file sizes and hashes for integrity checks.

Not preserved:

- `route_rgb_r4_r5_audit.json` from the temporary run was about 62 MB and mostly duplicates the route CSV plus example details.
  It is intentionally omitted from git to avoid repository bloat.

The temporary source directory `/tmp/automot_rgb_r4_r5_full_audit` can be deleted after these files are present.
