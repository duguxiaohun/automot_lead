# SFT Loop Phase1 Run

这个子包训练/测试第一轮无 memory 的四个视觉事实问题：

```text
HIGHWAY: YES|NO
OBSTACLE: YES|NO
VULNERABLE: YES|NO
TRAFFIC_LIGHT_ABNORMAL: YES|NO
```

所有命令都从 `AutoMoT/` 目录运行。默认只读本地
`checkpoints/Qwen3-VL-4B-Instruct`，不会联网下载。

## 1. 构建数据

```bash
python qwen3vl_local/sft_loop_phase1/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --answer-table keyframe_filter/collection_output/phase1_four_question_audit/phase1_four_question_answer_table.json \
  --output-dir checkpoints/sft_loop_phase1_data
```

这基本是一次性命令：同一个 `frame_index.jsonl` 可以反复用于 base Qwen 测试、LoRA 训练和
LoRA 复测。只有下面情况需要重建：

- `phase1_four_question_answer_table.json` 更新了；
- `collection_output/*_result.json` 的 RS/EVENT 标注更新了；
- 异常 route 过滤逻辑或 `lead_data` 内容变了；
- 想改变 `--split-seed` / `--test-ratio` / `--val-ratio`；
- `build_dataset.py` 的字段 schema 改了。

输出：

- `checkpoints/sft_loop_phase1_data/frame_index.jsonl`
- `checkpoints/sft_loop_phase1_data/manifest.json`

脚本按 route 稳定 split，默认 `test_ratio=0.10`，并剔除 `noScenarios`、异常时长 route
和 data-missing route。

## 2. 先测原始 Qwen

这是 prompt 迭代最重要的一步。默认使用 `audit=True` prompt，因此模型会输出
`EVIDENCE_*` 和四个答案；parser 仍只解析四个 YES/NO。

1 卡测原始 Qwen：

```bash
GPU_IDS=0 \
python qwen3vl_local/sft_loop_phase1/eval.py \
  --index checkpoints/sft_loop_phase1_data/frame_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_loop_phase1_eval/base_prompt_v3 \
  --cases-per-bin 64 \
  --audit-prompt
```

评估集按 8 桶 exact balance：

```text
HIGHWAY:YES / HIGHWAY:NO
OBSTACLE:YES / OBSTACLE:NO
VULNERABLE:YES / VULNERABLE:NO
TRAFFIC_LIGHT_ABNORMAL:YES / TRAFFIC_LIGHT_ABNORMAL:NO
```

每个样本仍回答全部四个问题；`focus_question` 只用于采样和统计，不进入 prompt。

输出：

- `metrics.json`：总体 exact match、每个问题 accuracy、8 桶计数。
- `cases.jsonl`：每个 case 的 prompt、RGB 路径、GT、parsed、raw output、ok_by_key。
- `summary.md`：简短结果。
- `error_cases/`：错例的 `case.json` 和 4 帧 RGB history。把这个目录或其中若干错例给我看，
  就可以继续分析到底是高速判据、障碍路径、弱势参与者，还是信号灯异常描述没教明白。

如果只想跑很小 smoke：

```bash
GPU_IDS=0 \
python qwen3vl_local/sft_loop_phase1/eval.py \
  --index checkpoints/sft_loop_phase1_data/frame_index.jsonl \
  --output-dir checkpoints/sft_loop_phase1_eval/smoke_base \
  --cases-per-bin 2 \
  --max-frames 200 \
  --overwrite
```

2 卡测原始 Qwen：

```bash
GPU_IDS=0,1 torchrun --nproc_per_node=2 \
  qwen3vl_local/sft_loop_phase1/eval.py \
  --index checkpoints/sft_loop_phase1_data/frame_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_loop_phase1_eval/base_prompt_v3_2gpu \
  --cases-per-bin 64 \
  --audit-prompt
```

4 卡测原始 Qwen：

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase1/eval.py \
  --index checkpoints/sft_loop_phase1_data/frame_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_loop_phase1_eval/base_prompt_v3_4gpu \
  --cases-per-bin 64 \
  --audit-prompt
```

多卡 eval 会写 `cases_rank0.jsonl`、`cases_rank1.jsonl` ...，rank0 汇总
`metrics.json` / `summary.md`。

## 3. 训练 LoRA

默认 LoRA 只挂语言侧，视觉侧保持 frozen。单卡：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_loop_phase1/train.sh single
```

2 卡 DDP：

```bash
GPU_IDS=0,1 bash qwen3vl_local/sft_loop_phase1/train.sh ddp
```

4 卡 DDP：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase1/train.sh ddp
```

快速 check / smoke：

1 卡：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_loop_phase1/train.sh check
```

2 卡：

```bash
MAX_STEPS=2 \
FOCUS_BALANCE_COUNT=2 \
GPU_IDS=0,1 bash qwen3vl_local/sft_loop_phase1/train.sh ddp
```

4 卡：

```bash
MAX_STEPS=2 \
FOCUS_BALANCE_COUNT=2 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase1/train.sh ddp
```

常用覆盖：

1 卡：

```bash
MAX_STEPS=3000 \
FOCUS_BALANCE_COUNT=512 \
LR=1e-5 \
LORA_RANK=16 \
LORA_VISION_SCOPE=off \
GPU_IDS=0 bash qwen3vl_local/sft_loop_phase1/train.sh single
```

2 卡：

```bash
MAX_STEPS=3000 \
FOCUS_BALANCE_COUNT=512 \
LR=1e-5 \
LORA_RANK=16 \
LORA_VISION_SCOPE=off \
GPU_IDS=0,1 bash qwen3vl_local/sft_loop_phase1/train.sh ddp
```

4 卡：

```bash
MAX_STEPS=3000 \
FOCUS_BALANCE_COUNT=512 \
LR=1e-5 \
LORA_RANK=16 \
LORA_VISION_SCOPE=off \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase1/train.sh ddp
```

训练采样同样按 8 桶 exact balance。每个 work item 有一个不可见
`focus_question`，但 loss 默认监督同一 assistant target 的四个 YES/NO 值 token。
训练产物：

- `checkpoints/sft_loop_phase1_runs/latest/final/adapter_model.safetensors`
- `checkpoints/sft_loop_phase1_runs/latest/final/adapter_config.json`
- `checkpoints/sft_loop_phase1_runs/latest/final/sft_loop_phase1_adapter_config.json`
- `checkpoints/sft_loop_phase1_runs/latest/tb/`
- `checkpoints/sft_loop_phase1_runs/latest/train_balance.json`

默认防覆盖目录与 `sft_base_simple` 类似：

```text
checkpoints/sft_loop_phase1_runs/run_<RUN_TAG>/
checkpoints/sft_loop_phase1_runs/latest -> run_<RUN_TAG>
```

## 4. 测试训练后的 LoRA

单卡：

```bash
GPU_IDS=0 \
python qwen3vl_local/sft_loop_phase1/eval.py \
  --index checkpoints/sft_loop_phase1_data/frame_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_loop_phase1_runs/latest/final \
  --output-dir checkpoints/sft_loop_phase1_eval/lora_final_prompt_v3 \
  --cases-per-bin 64 \
  --audit-prompt
```

2 卡：

```bash
GPU_IDS=0,1 torchrun --nproc_per_node=2 \
  qwen3vl_local/sft_loop_phase1/eval.py \
  --index checkpoints/sft_loop_phase1_data/frame_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_loop_phase1_runs/latest/final \
  --output-dir checkpoints/sft_loop_phase1_eval/lora_final_prompt_v3_2gpu \
  --cases-per-bin 64 \
  --audit-prompt
```

4 卡：

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase1/eval.py \
  --index checkpoints/sft_loop_phase1_data/frame_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_loop_phase1_runs/latest/final \
  --output-dir checkpoints/sft_loop_phase1_eval/lora_final_prompt_v3_4gpu \
  --cases-per-bin 64 \
  --audit-prompt
```

建议保留 base 和 LoRA 两份结果：

- `checkpoints/sft_loop_phase1_eval/base_prompt_v3/`
- `checkpoints/sft_loop_phase1_eval/lora_final_prompt_v3/`

这样可以直接比较：

- base Qwen 是不是已经会答某些问题；
- LoRA 是否只修正了答案格式，还是确实改善了视觉判据；
- 错例 evidence 是否暴露 prompt 仍然含糊，例如把直路/护栏误判高速，或把他车闯红灯误判灯异常。
