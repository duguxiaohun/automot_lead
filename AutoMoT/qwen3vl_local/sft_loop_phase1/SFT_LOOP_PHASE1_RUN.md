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

正式输出约定：

- 数据索引写到 `checkpoints/sft_loop_phase1_data/`。
- base/LoRA eval 结果写到 `checkpoints/sft_loop_phase1_eval/<run_name>/`。
- 训练结果写到 `checkpoints/sft_loop_phase1_runs/`。
- 不建议把正式 eval 写到仓库根目录的 `sft_loop_phase1_eval/`；如果命令显式传
  `--output-dir sft_loop_phase1_eval/...`，脚本会按你给的路径写到根目录，这只适合临时调试。

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
  --output-dir checkpoints/sft_loop_phase1_eval/base_zero_shot_prompt \
  --cases-per-bin 64 \
  --audit-prompt
```

评估集按四个主任务分别做 YES/NO 1:1。也就是说 HIGHWAY 模块只保证
`HIGHWAY:YES / HIGHWAY:NO` 均衡，OBSTACLE 模块只保证
`OBSTACLE:YES / OBSTACLE:NO` 均衡，以此类推：

```text
HIGHWAY:YES / HIGHWAY:NO
OBSTACLE:YES / OBSTACLE:NO
VULNERABLE:YES / VULNERABLE:NO
TRAFFIC_LIGHT_ABNORMAL:YES / TRAFFIC_LIGHT_ABNORMAL:NO
```

每个样本仍回答全部四个问题；`focus_question` / `task` 只用于采样和统计，不进入 prompt。
例如 HIGHWAY 模块的主问题是“是否高速”，该模块会记录 HIGHWAY 的
TP/FP/FN/TN、precision/recall/F1，同时顺带记录这批 HIGHWAY 1:1 样本上
OBSTACLE / VULNERABLE / TRAFFIC_LIGHT_ABNORMAL 的结果；这些副问题在该模块里不要求
YES/NO 均衡。其它三个模块同理。

输出：

- `metrics.json`：总结果 + `task_reports.{HIGHWAY,OBSTACLE,VULNERABLE,TRAFFIC_LIGHT_ABNORMAL}`。
  每个 task report 都有主问题 1:1 balance、TP/FP/FN/TN、precision/recall/F1 和副问题统计。
- `cases.jsonl`：所有 case 放在一起，含 prompt、RGB 路径、GT、parsed、raw output、ok_by_key。
- `task_cases/<TASK>/cases.jsonl`：按主任务拆开的 case 记录，方便只看某一类问题。
- `summary.md`：四个主任务模块的简短 Markdown 报告。
- `error_cases/<TASK>/case_*/rgb/`：主问题答错的 4 帧 RGB history，父目录明确区分来自哪个主任务。
- `rgb_cases/<TASK>/case_*/rgb/`：只有显式加 `--save-all-rgb` 时才复制所有受评 RGB；默认不复制全量 RGB，
  避免输出太大。

你后续给我分析时，优先打包这些轻量文件即可，不需要传全量 RGB：

```bash
tar -czf /tmp/base_zero_shot_prompt_records.tgz \
  checkpoints/sft_loop_phase1_eval/base_zero_shot_prompt/metrics.json \
  checkpoints/sft_loop_phase1_eval/base_zero_shot_prompt/summary.md \
  checkpoints/sft_loop_phase1_eval/base_zero_shot_prompt/cases.jsonl \
  checkpoints/sft_loop_phase1_eval/base_zero_shot_prompt/task_cases
```

如果某个主任务表现很差，再只补充该主任务的少量 RGB 错例，例如：

```bash
tar -czf /tmp/base_highway_error_rgb_sample.tgz \
  checkpoints/sft_loop_phase1_eval/base_zero_shot_prompt/error_cases/HIGHWAY
```

不建议第一次就传 `rgb_cases/`，除非你专门加了 `--save-all-rgb` 并且只想给我一个很小 smoke。

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
  --output-dir checkpoints/sft_loop_phase1_eval/base_zero_shot_prompt_2gpu \
  --cases-per-bin 64 \
  --audit-prompt
```

4 卡测原始 Qwen：

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase1/eval.py \
  --index checkpoints/sft_loop_phase1_data/frame_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_loop_phase1_eval/base_zero_shot_prompt_4gpu \
  --cases-per-bin 64 \
  --audit-prompt
```

根据 base 错例修过 prompt 后的复测，也放在 `checkpoints/` 下另起目录：

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase1/eval.py \
  --index checkpoints/sft_loop_phase1_data/frame_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_loop_phase1_eval/base_zero_shot_prompt_after_feedback_4gpu \
  --cases-per-bin 64 \
  --audit-prompt \
  --overwrite
```

多卡 eval 会写 `cases_rank0.jsonl`、`cases_rank1.jsonl` ...，rank0 汇总
`metrics.json` / `summary.md`；同时 `task_cases/<TASK>/` 下也会按 rank 拆分。

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

训练采样同样按四个主任务各自 YES/NO 1:1。每个 work item 有一个不可见
`focus_question`，但 loss 默认监督同一 assistant target 的四个 YES/NO 值 token；
也就是训练和测试都保持“主问题均衡，副问题顺带记录/监督”的口径。
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
  --output-dir checkpoints/sft_loop_phase1_eval/lora_zero_shot_prompt \
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
  --output-dir checkpoints/sft_loop_phase1_eval/lora_zero_shot_prompt_2gpu \
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
  --output-dir checkpoints/sft_loop_phase1_eval/lora_zero_shot_prompt_4gpu \
  --cases-per-bin 64 \
  --audit-prompt
```

建议保留 base 和 LoRA 两份结果：

- `checkpoints/sft_loop_phase1_eval/base_zero_shot_prompt/`
- `checkpoints/sft_loop_phase1_eval/lora_zero_shot_prompt/`

这样可以直接比较：

- base Qwen 是不是已经会答某些问题；
- LoRA 是否只修正了答案格式，还是确实改善了视觉判据；
- 错例 evidence 是否暴露 prompt 仍然含糊，例如把直路/护栏误判高速，或把他车闯红灯误判灯异常。
