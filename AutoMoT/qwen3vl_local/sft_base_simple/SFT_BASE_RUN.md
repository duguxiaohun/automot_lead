# SFT Base Simple Run

从远端 `AutoMoT/` 目录运行。数据根默认是 `lead_data`，输出默认写到 `checkpoints/sft_base_simple_*`。

## Build Dataset

```bash
python qwen3vl_local/sft_base_simple/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_base_simple_data
```

Smoke：

```bash
python qwen3vl_local/sft_base_simple/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_base_simple_data_smoke \
  --max-routes 4 \
  --max-frames-per-route 16
```

## Checks

```bash
python qwen3vl_local/sft_base_simple/check_loss_mask.py
python qwen3vl_local/sft_base_simple/test_memory_curriculum.py
python qwen3vl_local/sft_base_simple/test_prompt_snapshots.py
python qwen3vl_local/sft_base_simple/test_train_resume.py
```

## Train

单卡：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_base_simple/train.sh single
```

4 卡：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_base_simple/train.sh ddp
```

默认训练口径：

- `HIGHWAY_ROUTE_SAMPLE_TARGET=0.5`
- `FOURBIN_ROUTES_PER_BATCH=16`
- `JOINT_BALANCE_REPEAT_MODE=none`
- `JOINT_TARGET_BALANCE_MODE=exact`
- `JOINT_TARGET_BALANCE_COUNT=8`
- `UE_FRAME_REPEAT=1`
- `UE_EVENT_LOSS_WEIGHT=1.0`
- `UE_REPEAT_MODE=none`
- `REGULAR_REPEAT_MODE=none`
- `MEMORY_EARLY_UE_FRAMES=4`

常用覆盖：

```bash
OUTPUT_DIR=checkpoints/sft_base_simple_runs \
HIGHWAY_ROUTE_SAMPLE_TARGET=0.5 \
FOURBIN_ROUTES_PER_BATCH=16 \
JOINT_TARGET_BALANCE_MODE=exact \
JOINT_TARGET_BALANCE_COUNT=8 \
UE_EVENT_LOSS_WEIGHT=1.0 \
RE_EVENT_LOSS_WEIGHT=1.0 \
MEMORY_RS_WRONG_PROB=0.30 \
MEMORY_RS_UNKNOWN_PROB=0.40 \
MEMORY_EVENT_WRONG_PROB=0.35 \
MEMORY_EVENT_UNKNOWN_PROB=0.35 \
MEMORY_EARLY_UE_FRAMES=4 \
MEMORY_EARLY_UE_WRONG_SCALE=1.75 \
MEMORY_EARLY_UE_UNKNOWN_SCALE=1.35 \
MEMORY_EARLY_UE_DROPOUT_SCALE=1.50 \
MEMORY_EARLY_UE_RESAMPLE_PROB=0.70 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_base_simple/train.sh ddp
```

纯视觉 no-memory 基线：

```bash
PROMPT_MEMORY_MODE=hidden \
LORA_VISION_SCOPE=merger \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_base_simple/train.sh ddp
```

closed-loop probe：

```bash
CLOSED_LOOP_PROBE_STEPS=50 \
CLOSED_LOOP_PROBE_FOURBIN_CASES=128 \
CLOSED_LOOP_PROBE_WRITE_FRAMES=0 \
CLOSED_LOOP_PROBE_GPU_IDS=0 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_base_simple/train.sh ddp
```

`CLOSED_LOOP_PROBE_FOURBIN_CASES` 控制 closed-loop probe 的四格 balanced eval 总预算；脚本会折成每格 case 数。

默认防覆盖目录：

```text
checkpoints/sft_base_simple_runs/run_<RUN_TAG>/
checkpoints/sft_base_simple_runs/latest -> run_<RUN_TAG>
```

## Eval

四格均衡 full eval 是默认测试口径：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base_simple/eval.py \
  --adapter-dir checkpoints/sft_base_simple_runs/latest/final \
  --task full
```

默认 joint eval 会先按当前帧 GT 四格抽受评帧，再按 route 顺序从首帧闭环推进 memory 到最远受评帧；`metrics.json` 里的 `frames` 是四格受评帧数，`rollout_frames` 是为得到真实 previous memory 实际跑过的帧数。`--initial-memory-noise none` 与 joint eval 组合会被拒绝，避免把当前帧 GT 写进 prompt。

补充整 route 闭环评估，用来看完整相邻帧 change matrix：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base_simple/eval.py \
  --adapter-dir checkpoints/sft_base_simple_runs/latest/final \
  --task full \
  --full-balance-mode none \
  --sample-routes 64
```

扩大每格 case 数：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base_simple/eval.py \
  --adapter-dir checkpoints/sft_base_simple_runs/latest/final \
  --task full \
  --full-balance-cases-per-bin 128 \
  --no-write-frames
```

黑图消融：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base_simple/eval.py \
  --adapter-dir checkpoints/sft_base_simple_runs/latest/final \
  --task full \
  --image-ablation black \
  --ablate-goal
```

阈值/PR 诊断：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base_simple/eval.py \
  --adapter-dir checkpoints/sft_base_simple_runs/latest/final \
  --task full \
  --prediction-mode score \
  --event-logit-bias 0.0 \
  --full-balance-cases-per-bin 128 \
  --no-write-frames \
  --output-dir checkpoints/sft_base_simple_runs/latest/eval_results/event_score_bias_0
```

一键诊断：

```bash
CKPT=checkpoints/sft_base_simple_runs/latest/final \
TRIAGE_PROFILE=fast \
BALANCED_CASES_PER_BIN=128 \
GPU_IDS=0 bash qwen3vl_local/sft_base_simple/run_triage_eval.sh
```

4 卡评估：

```bash
CKPT=checkpoints/sft_base_simple_runs/latest/final \
TRIAGE_PROFILE=fast \
BALANCED_CASES_PER_BIN=128 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_base_simple/run_triage_eval.sh
```

默认输出：

```text
checkpoints/sft_base_simple_runs/latest/eval_results/full_route/<timestamp>/
  metrics.json
  frames.jsonl
  summary.md
  report.html
  tb/
```

TensorBoard：

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_base_simple_runs/latest
```

重点看：

- `road_acc`
- `highway_f1`
- `event_acc`
- `ue_f1`
- `joint_acc`
- `train/fourbin_highway_ue_last_batch`
- `train/fourbin_highway_re_last_batch`
- `train/fourbin_non_highway_ue_last_batch`
- `train/fourbin_non_highway_re_last_batch`
- `train/road_highway_rate_last_batch`
- `train/event_ue_rate_last_batch`
- `memory/early_ue_event_re_rate_last_batch`
- `memory/early_ue_event_ue_rate_last_batch`
- `memory/early_ue_event_unknown_rate_last_batch`
- `memory/early_ue_event_hidden_rate_last_batch`

