# SFT Baseline Run

从远端 `AutoMoT/` 目录运行。数据根默认是 `lead_data`，输出默认写到 `checkpoints/sft_baseline_*`。

## Build Dataset

```bash
python qwen3vl_local/sft_baseline/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_baseline_data
```

Smoke：

```bash
python qwen3vl_local/sft_baseline/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_baseline_data_smoke \
  --max-routes 4 \
  --max-frames-per-route 16
```

## Static Checks

```bash
python qwen3vl_local/sft_baseline/check_loss_mask.py
python qwen3vl_local/sft_baseline/test_memory_curriculum.py
python qwen3vl_local/sft_baseline/test_prompt_snapshots.py
python qwen3vl_local/sft_baseline/test_train_resume.py
```

## Train

单卡：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_baseline/train.sh single
```

4 卡：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_baseline/train.sh ddp
```

常用覆盖：

```bash
OUTPUT_DIR=checkpoints/sft_baseline_runs \
UE_EVENT_LOSS_WEIGHT=4.0 \
RE_EVENT_LOSS_WEIGHT=1.0 \
MEMORY_RS_WRONG_PROB=0.30 \
MEMORY_RS_UNKNOWN_PROB=0.40 \
MEMORY_EVENT_WRONG_PROB=0.35 \
MEMORY_EVENT_UNKNOWN_PROB=0.35 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_baseline/train.sh ddp
```

默认 `LORA_VISION_SCOPE=off`，只训练语言侧 LoRA，不微调视觉塔。需要做视觉 LoRA
消融时再显式加 `LORA_VISION_SCOPE=merger|last4|all`。

默认仍使用防覆盖目录：

```text
checkpoints/sft_baseline_runs/run_<RUN_TAG>/
checkpoints/sft_baseline_runs/latest -> run_<RUN_TAG>
```

## Eval

完整 route 抽样：

```bash
GPU_IDS=0 python qwen3vl_local/sft_baseline/eval.py \
  --adapter-dir checkpoints/sft_baseline_runs/latest/final \
  --task full
```

高速/非高速转折：

```bash
GPU_IDS=0 python qwen3vl_local/sft_baseline/eval.py \
  --adapter-dir checkpoints/sft_baseline_runs/latest/final \
  --task road
```

RE/UE 转折：

```bash
GPU_IDS=0 python qwen3vl_local/sft_baseline/eval.py \
  --adapter-dir checkpoints/sft_baseline_runs/latest/final \
  --task event
```

黑图消融：

```bash
GPU_IDS=0 python qwen3vl_local/sft_baseline/eval.py \
  --adapter-dir checkpoints/sft_baseline_runs/latest/final \
  --task full \
  --image-ablation black \
  --ablate-goal
```

默认输出：

```text
checkpoints/sft_baseline_runs/latest/eval_results/<full_route|road_transition|event_transition>/<timestamp>/
  metrics.json
  frames.jsonl
  summary.md
```

关键指标看：

- `road_acc`
- `highway_f1`
- `event_acc`
- `ue_f1`
- `joint_acc`
- `road_change_f1`
- `event_change_f1`
