# SFT New Loop Phase1

Fuses `sft_loop_phase1` and `sft_loop_phase2_augment` into one YES/NO turn.
Each sample always asks Phase1 visible-fact questions, and embeds them inside
one of the Phase2 augment variants:

- Phase1: `HIGHWAY`, `STATIC_OBSTACLE`, `VULNERABLE`, `TRAFFIC_LIGHT_ABNORMAL`
- Phase2 `all_random_order`: all `RS1`, `RS2`, `RS4`, `RS5` in random order
- Phase2 `subset_random`: a random 1/2/3-question RS subset
- Phase2 `hierarchical_probe`: `RS_HIGHWAY`, `GROUP`, and one concrete RS detail

Training uses the Phase2 augment train ratio `4:1:1`; eval/generation uses the
Phase2 eval ratio `2:1:1`. `RS_HIGHWAY` is the Phase2 hierarchical highway/R3
question and is intentionally separate from the audited Phase1 `HIGHWAY` line.

The dataset builder follows the latest Phase2 filtering: abnormal LEAD routes
are removed, full-frame RGB review coverage is checked, and visual-risk frames
are excluded unless `--include-visual-risk` is set. Phase1 labels come from the
audited answer table; Phase2 labels come from per-frame RS annotations.

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
- eval `metrics.json`

The sampling contract is exact per split: every focus answer key has YES:NO =
1:1 in the focused work list. Since there are four Phase1 focus keys and four
Phase2 focus keys, Phase1-focus and Phase2-focus cases are also 1:1.
