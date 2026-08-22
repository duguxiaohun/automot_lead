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
audits so rare positives are not silently overused. Training also defaults to
`MAX_TRAIN_FRAME_REPEAT=10`: sampling aborts before the model is loaded if any
one frame exceeds that per-epoch reuse limit. The default validation and
checkpoint cadence is sized for the larger epoch: teacher-forced eval every
2,000 steps, generation eval every 2,000 steps, and checkpoint save every
20,000 steps.

The dataset builder follows the latest Phase2 filtering: abnormal LEAD routes
are removed, full-frame RGB review coverage is checked, and visual-risk frames
are excluded unless `--include-visual-risk` is set. Phase1 labels come from the
audited answer table; Phase2 labels come from per-frame RS annotations. Visual
subgroup overrides only apply from structured RGB audit notes/annotations, not
from free-text `audit_evidence`.

## 0. Run Layout

Run every command below from `AutoMoT/`. The directory intentionally mirrors
`sft_loop_phase2_augment`: dataset build, base eval, LoRA training, LoRA eval,
error RGB sampling, TensorBoard, and RGB-mode matrix all have local entries.

Default important paths:

- dataset: `checkpoints/sft_new_loop_phase1_data/frame_index.jsonl`
- train runs: `checkpoints/sft_new_loop_phase1_runs/run_<RUN_TAG>_combined_phase1_phase2_<rgb_mode>/`
- eval review runs: `checkpoints/sft_new_loop_phase1_eval_review/<timestamp>/`
- matrix runs: `checkpoints/sft_new_loop_phase1_eval_matrix/<timestamp>/`
- audit samples: `checkpoints/sft_new_loop_phase1_audit_samples/`
- bundled RGB coverage proof: `qwen3vl_local/sft_new_loop_phase1/phase2_rgb_audit_coverage.json`

## 1. Build Dataset

```bash
python qwen3vl_local/sft_new_loop_phase1/visual_audit.py
python qwen3vl_local/sft_new_loop_phase1/build_dataset.py
```

`build_dataset.py` stores `history_rgb_paths` relative to `--data-root`
(`lead_data` by default). Train/eval resolve those relative paths with
`--data-root`, and also remap legacy absolute paths containing `lead_data`.
On a remote machine with a different checkout or data mount, set only:

```bash
DATA_ROOT=/path/to/lead_data GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase1/train.sh check
```

If the dataset was built on another machine before relative paths were added,
rebuild it on the target machine instead of editing the 1GB JSONL by hand.

Acceptance checks after build:

- `visual_audit_manifest.json` is written under `checkpoints/sft_new_loop_phase1_data/`.
- `manifest.json` has non-empty train/val/test splits.
- All 16 focus buckets exist for train/val/test.
- `visual_risk` filtered count matches expectation.
- Town12 free-text HIGHWAY override count remains 0.
- Random RGB path probes resolve through `--data-root`.

## 2. Base Qwen Eval

Production prompt:

```bash
GPU_IDS=0 python qwen3vl_local/sft_new_loop_phase1/eval.py \
  --history-rgb-mode 4rgb \
  --split test \
  --cases-per-bin 64
```

Audit prompt:

```bash
GPU_IDS=0 python qwen3vl_local/sft_new_loop_phase1/eval.py \
  --history-rgb-mode 4rgb \
  --audit-prompt \
  --split test \
  --cases-per-bin 64
```

For a four-GPU eval:

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_new_loop_phase1/eval.py \
  --history-rgb-mode 4rgb \
  --split test \
  --cases-per-bin 64
```

## 3. LoRA Training

Smoke check:

```bash
GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase1/train.sh check
```

Single GPU:

```bash
GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase1/train.sh single
```

Four-GPU DDP:

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_new_loop_phase1/train.sh ddp
```

Important defaults:

- `FOCUS_BALANCE_COUNT=9216`
- `MAX_TRAIN_FRAME_REPEAT=10`
- `GENERATION_EVAL_BALANCE_COUNT=16`
- `EVAL_STEPS=2000`
- `GENERATION_EVAL_STEPS=2000`
- `SAVE_STEPS=20000`
- `HISTORY_RGB_MODE=4rgb`

Use `HISTORY_RGB_MODE=2rgb_endpoints` for the two-frame endpoint input.

## 4. LoRA Eval

`eval.sh` resolves a run directory or adapter directory, then runs base
production/audit and LoRA production/audit, and writes a compact audit bundle.

```bash
ADAPTER_DIR=checkpoints/sft_new_loop_phase1_runs/latest/final \
  bash qwen3vl_local/sft_new_loop_phase1/eval.sh
```

Passing the run root is also supported:

```bash
bash qwen3vl_local/sft_new_loop_phase1/eval.sh \
  checkpoints/sft_new_loop_phase1_runs/latest
```

For direct eval without the review bundle:

```bash
GPU_IDS=0 python qwen3vl_local/sft_new_loop_phase1/eval.py \
  --adapter-dir checkpoints/sft_new_loop_phase1_runs/latest/final \
  --split test \
  --cases-per-bin 64
```

## 5. Error RGB Audit Samples

After an eval, sample common fused error types and copy the RGB history used by
the model:

```bash
python qwen3vl_local/sft_new_loop_phase1/audit_eval_cases.py \
  --eval-dir checkpoints/sft_new_loop_phase1_eval_review/<timestamp>/lora_production \
  --output-dir checkpoints/sft_new_loop_phase1_audit_samples/<tag> \
  --data-root lead_data \
  --per-target 12 \
  --overwrite
```

The sampler covers Phase1 false positives/false negatives, Phase2 RS errors,
hierarchical `RS_HIGHWAY` / `GROUP` errors, invalid answers, multi-YES Phase2
outputs, and subset unasked-line leakage.

## 6. TensorBoard

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_new_loop_phase1_runs/latest
```

Useful streams:

- `train/loss`
- `eval/exact_match_accuracy`
- `generation_eval/exact_match_accuracy`
- `focus/*`
- `variant/*`
- `augment/*`

## 7. RGB Mode Matrix

Run the phase2-style 10-step comparison for fused prompts:

```bash
bash qwen3vl_local/sft_new_loop_phase1/run_rgb_mode_matrix.sh
```

It runs:

1. base 4rgb production
2. base 4rgb audit
3. base 2rgb_endpoints production
4. base 2rgb_endpoints audit
5. train 4rgb LoRA
6. LoRA 4rgb production
7. LoRA 4rgb audit
8. train 2rgb_endpoints LoRA
9. LoRA 2rgb_endpoints production
10. LoRA 2rgb_endpoints audit

Common overrides:

```bash
GPU_IDS=0 DATA_ROOT=/path/to/lead_data CASES_PER_BIN=64 \
  bash qwen3vl_local/sft_new_loop_phase1/run_rgb_mode_matrix.sh
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
augment specs under the Phase2 train/eval ratios. Focus balance, the three
variant totals, Phase2 `(focus_bucket, variant)` quotas, and all
`all_random_order/RS*:YES|NO` buckets are hard constraints. Phase1 focus buckets
are sampled naturally first; only missing global `R1/R2/R4/R5` capacity needed
by exact all-random quotas is repaired from unused rows in a compatible focus
bucket. The sampler never cycles a rare per-focus RS subgroup to make a
secondary distribution look uniform. All-random YES slots are reserved first,
and its NO slots are assigned by an exact capacity matching step. Subset/hierarchical
`augment_balance_key` buckets are target-driven and each epoch records exact
deviation reports. It records `augment_balance_key` counts, focus-variant
counts, all-random deviation, Phase2 focus-variant deviation, variant reports,
answer-pattern diagnostics, subset unasked-line leakage, `RS_HIGHWAY`, and all
`GROUP:<id>` metrics. The combined work keeps Phase1-focus and Phase2-focus cases 1:1.
`train_metrics.jsonl` includes per-window `augment_counts`.
