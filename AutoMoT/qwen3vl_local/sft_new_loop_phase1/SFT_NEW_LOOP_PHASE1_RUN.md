# SFT New Loop Phase1

Fuses `sft_loop_phase1` and `sft_loop_phase2_augment` into one YES/NO turn.
Each sample always asks Phase1 visible-fact questions, and embeds them inside
one of the Phase2 augment variants:

- Phase1: `HIGHWAY`, `STATIC_OBSTACLE`, `VULNERABLE`, `TRAFFIC_LIGHT_ABNORMAL`
- Phase2 `all_random_order`: all `RS1`, `RS2`, `RS4`, `RS5` in random order
- Phase2 `subset_random`: a random 1/2/3-question RS subset
- Phase2 `hierarchical_probe`: `RS_HIGHWAY`, `GROUP`, and one concrete RS detail

Training uses the Phase2 augment train ratio `4:1:1`; eval/generation uses the
Phase2 eval ratio `2:1:1`. Generation validation defaults to
`GENERATION_EVAL_BALANCE_COUNT=16` so subset/hierarchical diagnostics have
enough samples to expose imbalance. `RS_HIGHWAY` is the Phase2 hierarchical
highway/R3 question and is intentionally separate from the audited Phase1
`HIGHWAY` line. Full training defaults to `FOCUS_BALANCE_COUNT=9216`, which
gives 147,456 sampled cases per epoch across the eight focus keys and matches
the old Phase2 augment epoch case count; `train_balance.json` records repeat
audits so rare positives are not silently overused. The default validation and
checkpoint cadence is sized for the larger epoch: teacher-forced eval every
2,000 steps, generation eval every 2,000 steps, and checkpoint save every
20,000 steps.

The dataset builder follows the latest Phase2 filtering: abnormal LEAD routes
are removed, full-frame RGB review coverage is checked, and visual-risk frames
are excluded unless `--include-visual-risk` is set. Phase1 labels come from the
audited answer table; Phase2 labels come from per-frame RS annotations. Visual
subgroup overrides only apply from structured RGB audit notes/annotations, not
from free-text `audit_evidence`.

Run from `AutoMoT/`:

```bash
python qwen3vl_local/sft_new_loop_phase1/build_dataset.py
GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase1/train.sh check
GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase1/train.sh single
GPU_IDS=0 python qwen3vl_local/sft_new_loop_phase1/eval.py \
  --adapter-dir checkpoints/sft_new_loop_phase1_runs/latest/final
```

Balance artifacts:

- `checkpoints/sft_new_loop_phase1_data/manifest.json`
- `checkpoints/sft_new_loop_phase1_runs/.../train_balance.json`
- `checkpoints/sft_new_loop_phase1_runs/.../balance/epoch_*.json`
- `checkpoints/sft_new_loop_phase1_runs/.../balance/epochs.jsonl`
- `checkpoints/sft_new_loop_phase1_runs/.../train_run_manifest.json`
- `checkpoints/sft_new_loop_phase1_runs/.../train_metrics.jsonl`
- `checkpoints/sft_new_loop_phase1_runs/.../train_eval_metrics.jsonl`
- eval `metrics.json`

The sampling contract has two layers. The Phase1 half is exact over the four
Phase1 focus keys with YES:NO = 1:1. The Phase2 half is also exact over
`RS1/RS2/RS4/RS5` with YES:NO = 1:1, then assigns all/subset/hierarchical
augment specs under the Phase2 train/eval ratios. Focus balance and the three
variant totals are hard constraints; concrete `augment_balance_key` buckets are
target-driven and each epoch records exact deviation reports. It records
`augment_balance_key` counts, variant reports, answer-pattern diagnostics,
subset unasked-line leakage, `RS_HIGHWAY`, and all `GROUP:<id>` metrics. The
combined work keeps Phase1-focus and Phase2-focus cases 1:1.
`train_metrics.jsonl` includes per-window `augment_counts`.
