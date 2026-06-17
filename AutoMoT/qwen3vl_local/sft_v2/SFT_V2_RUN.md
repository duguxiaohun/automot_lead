# SFT v2 Runbook

Run commands from the remote `AutoMoT/` directory.

## 1. Build Data

Default build keeps all valid candidates per scenario:

```bash
python qwen3vl_local/sft_v2/build_dataset.py \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --output-dir checkpoints/sft_v2_data
```

By default, train rows apply `--wrong-scene-ratio 0.15`: a subset of stage-2
prompts lists a wrong selected scene. The previous hint and supervised
status/subgoal are phase-mapped into that selected scene's own event sequence.
Set `--wrong-scene-ratio 0` to disable it.

Optional downsampled build:

```bash
python qwen3vl_local/sft_v2/build_dataset.py \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --samples-per-scenario 800 \
  --output-dir checkpoints/sft_v2_data_800
```

Quick check:

```bash
python qwen3vl_local/sft_v2/build_dataset.py \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --dry-run \
  --output-dir checkpoints/sft_v2_data_dry
```

Artifacts:

- `train.jsonl`
- `val.jsonl`
- `stats.json`

Each row stores two serial stages under `stage_messages`:

```text
stage_messages.scene   -> image turn + SCENE
stage_messages.status  -> text follow-up turn + STATUS / SUBGOAL
```

## 2. Loss Mask Check

```bash
python qwen3vl_local/sft_v2/check_loss_mask.py
```

Expected: `ok: true`. This means the value spans for `SCENE`, `STATUS`, and
`SUBGOAL` are located correctly. If `--model-dir` exists locally, the script
also verifies tokenizer-level 0/1 value-token masks. Format tokens are not
trained.

## 3. Train

Single GPU:

```bash
GPU_IDS=0 bash qwen3vl_local/sft_v2/train.sh single
```

4-GPU DDP with auto GPU selection:

```bash
DDP_GPU_COUNT=4 bash qwen3vl_local/sft_v2/train.sh ddp
```

Explicit 4-GPU pin:

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v2/train.sh ddp
```

Sanity check:

```bash
GPU_IDS=0 bash qwen3vl_local/sft_v2/train.sh check
```

Common env:

| env | default | note |
|---|---:|---|
| `MODEL_DIR` | `checkpoints/Qwen3-VL-4B-Instruct` | local model dir |
| `TRAIN_JSONL` | `checkpoints/sft_v2_data/train.jsonl` | train jsonl |
| `VAL_JSONL` | `checkpoints/sft_v2_data/val.jsonl` | val jsonl |
| `OUTPUT_DIR` | `checkpoints/sft_v2_lora` | base output dir |
| `MAX_LENGTH` | `8192` | prompt contains all scenes or one event sequence |
| `RUN_TAG` | timestamp | writes to `OUTPUT_DIR/run_<tag>` |
| `NO_RUN_SUBDIR` | `0` | set `1` for old overwrite behavior |
| `GPU_IDS` | empty | explicit GPU pin |
| `DDP_GPU_COUNT` | `8` | GPU count for auto DDP selection |
| `LABEL_WEIGHT` | `1.0` | value-token loss weight |

Training runs one multi-turn forward per sample:

1. image + scene prompt -> supervise `SCENE` value token only.
2. append status prompt with selected scene -> supervise `STATUS/SUBGOAL` value tokens only.

For wrong-scene augmented rows, the selected scene is intentionally not the GT
scene, and the supervised status/subgoal are legal same-phase events from that
selected scene.

## 4. Eval

```bash
GPU_IDS=0 python qwen3vl_local/sft_v2/eval.py \
  --jsonl checkpoints/sft_v2_data/val.jsonl \
  --lora-dir checkpoints/sft_v2_lora/latest/final \
  --save-root checkpoints/sft_v2_lora/latest \
  --max-samples 100
```

Full val:

```bash
GPU_IDS=0 python qwen3vl_local/sft_v2/eval.py \
  --jsonl checkpoints/sft_v2_data/val.jsonl \
  --lora-dir checkpoints/sft_v2_lora/latest/final \
  --save-root checkpoints/sft_v2_lora/latest
```

Eval uses the true two-stage protocol:

1. Generate `SCENE`.
2. If `SCENE` is invalid, stop the sample.
3. If `SCENE` is valid, append a new prompt from the predicted scene's event
   sequence and continue from the scene-step KV cache to generate
   `STATUS/SUBGOAL`.
   The previous-status hint is phase-mapped into the predicted scene so the
   prompt stays internally consistent even when the predicted scene is wrong.

Outputs under `checkpoints/sft_v2_lora/latest/eval_v2/`:

- `metrics.json`
- `scenario_metrics.json`
- `predictions.jsonl`
- `predictions_diff.jsonl`
- `cases/`

`status_accuracy` / `subgoal_accuracy` are serial metrics: scene must also be
correct. `status_raw_accuracy` / `subgoal_raw_accuracy` are diagnostics only.
`valid_total` and `*_valid_scene` metrics report the same task after excluding
invalid-scene rows from the denominator. `status_kv_reuse_rate` should stay near
1.0; fallback means the second stage had to rebuild the full multi-turn context.

Base comparison:

```bash
GPU_IDS=0 python qwen3vl_local/sft_v2/eval.py \
  --lora-dir '' \
  --save-root checkpoints/sft_v2_base_eval \
  --max-samples 100
```

## 5. Case Probe

```bash
GPU_IDS=0 python qwen3vl_local/sft_v2/probe.py \
  --lora-dir checkpoints/sft_v2_lora/latest/final \
  --save-root checkpoints/sft_v2_lora/latest \
  --num-per-scenario 4 --seed 0
```

Specific scenarios:

```bash
GPU_IDS=0 python qwen3vl_local/sft_v2/probe.py \
  --lora-dir checkpoints/sft_v2_lora/latest/final \
  --save-root checkpoints/sft_v2_lora/latest \
  --scenarios Accident,ConstructionObstacle \
  --num-per-scenario 6 --seed 7
```

Outputs under `checkpoints/sft_v2_lora/latest/eval_cases_v2/`.
