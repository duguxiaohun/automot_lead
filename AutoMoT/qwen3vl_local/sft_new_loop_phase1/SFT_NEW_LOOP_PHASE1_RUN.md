# SFT New Loop Phase1 运行手册

本目录把 `sft_loop_phase1` 的四个可见事实问题和
`sft_loop_phase2_augment` 的 ROAD_STRUCTURE 三类增强问法融合到同一轮
YES/NO 问答里。每个样本固定包含 Phase1 四问：

- `HIGHWAY`
- `STATIC_OBSTACLE`
- `VULNERABLE`
- `TRAFFIC_LIGHT_ABNORMAL`

同时按 Phase2 最新增强方式加入 ROAD_STRUCTURE 问题：

- `all_random_order`：随机顺序回答 `RS1/RS2/RS4/RS5` 全部四问。
- `subset_random`：随机回答 1/2/3 个 RS 子问题。
- `hierarchical_probe`：先问 `RS_HIGHWAY` 和 `GROUP`，再问一个具体 RS 细节。

训练使用 Phase2 augment 的 `4:1:1` 比例，eval / generation eval 使用 `2:1:1`。
`RS_HIGHWAY` 是 Phase2 hierarchical 里的高速/R3 判断，和 Phase1 审计过的
`HIGHWAY` 是两个不同输出行，不要混淆。

正式训练默认 `FOCUS_BALANCE_COUNT=9216`，即每轮 147,456 个 sampled case，
和旧 `sft_loop_phase2_augment` 的每轮训练量对齐。八个主 focus 问题全部保持
YES:NO = 1:1，Phase1 四问与 Phase2 四问总量也是 1:1。默认还会检查
`MAX_TRAIN_FRAME_REPEAT=10`，任一帧在单轮采样里复用超过上限会在加载模型前中止。

数据构建沿用 Phase2 最新过滤：剔除异常时长 route、检查 full-frame RGB review 覆盖，
默认排除 visual-risk 帧。Phase1 标签来自已审计四问答案表；Phase2 标签来自逐帧 RS
标注。视觉子组覆盖只允许来自结构化 RGB audit notes / annotations，不能从自由文本
`audit_evidence` 推断 route 标签。

## 0. 目录与产物

以下命令都从 `AutoMoT/` 目录运行，路径不要再加 `AutoMoT/` 前缀。

常用产物路径：

- 数据集：`checkpoints/sft_new_loop_phase1_data/frame_index.jsonl`
- 训练 run：`checkpoints/sft_new_loop_phase1_runs/run_<RUN_TAG>_combined_phase1_phase2_<rgb_mode>/`
- 一键 eval：`checkpoints/sft_new_loop_phase1_eval_review/<timestamp>/`
- 一键 pipeline：`checkpoints/sft_new_loop_phase1_pipeline/<timestamp>/`
- RGB 模式矩阵：`checkpoints/sft_new_loop_phase1_eval_matrix/<timestamp>/`
- 错例抽样：`checkpoints/sft_new_loop_phase1_audit_samples/`
- RGB 覆盖证明：`qwen3vl_local/sft_new_loop_phase1/phase2_rgb_audit_coverage.json`

## 1. 构建数据集

先生成视觉审计 manifest，再构建 fused index：

```bash
python qwen3vl_local/sft_new_loop_phase1/visual_audit.py
python qwen3vl_local/sft_new_loop_phase1/build_dataset.py
```

如果远程机器的数据根目录不同，只设置 `DATA_ROOT` 或显式传 `--data-root`：

```bash
python qwen3vl_local/sft_new_loop_phase1/build_dataset.py \
  --data-root /path/to/lead_data
```

`build_dataset.py` 默认把 `history_rgb_paths` 保存为相对 `--data-root` 的路径。
训练和评测会用 `--data-root` 解析相对路径，也能重映射旧 JSONL 里包含 `lead_data`
的绝对路径。若旧数据是在别的机器上构建的，长期方案是重新运行 `build_dataset.py`，
不要手工替换接近 1GB 的 JSONL。

构建后重点检查：

- `checkpoints/sft_new_loop_phase1_data/visual_audit_manifest.json` 已生成。
- `manifest.json` 的 train / val / test 都非空。
- train / val / test 都有 16 个非空 focus 桶。
- visual-risk 过滤数量符合预期。
- Town12 自由文本 HIGHWAY override 计数为 0。
- 随机抽查 RGB 路径能通过 `--data-root` 找到真实文件。

## 2. 快速 check

先跑小规模链路，确认数据、模型加载、loss、DDP 同步分支没有明显错误：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase1/train.sh check
```

常用远程数据根：

```bash
DATA_ROOT=/path/to/lead_data GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase1/train.sh check
```

check 模式默认只跑很少 step，但仍从完整 index 里构造均衡桶，不靠文件前几行取样。

## 3. 正式训练

单卡：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase1/train.sh single
```

四卡 DDP：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_new_loop_phase1/train.sh ddp
```

两帧端点输入对照：

```bash
HISTORY_RGB_MODE=2rgb_endpoints GPU_IDS=0,1,2,3 \
  bash qwen3vl_local/sft_new_loop_phase1/train.sh ddp
```

关键默认值：

- `FOCUS_BALANCE_COUNT=9216`
- `MAX_TRAIN_FRAME_REPEAT=10`
- `NUM_EPOCHS=3`
- `EVAL_STEPS=2000`
- `GENERATION_EVAL_STEPS=2000`
- `GENERATION_EVAL_BALANCE_COUNT=16`
- `SAVE_STEPS=20000`
- `HISTORY_RGB_MODE=4rgb`

训练产物会写入 `checkpoints/sft_new_loop_phase1_runs/`，非 check run 会更新
`checkpoints/sft_new_loop_phase1_runs/latest` 软链接。

## 4. base Qwen 评测

production prompt：

```bash
GPU_IDS=0 python qwen3vl_local/sft_new_loop_phase1/eval.py \
  --history-rgb-mode 4rgb \
  --split test \
  --cases-per-bin 64
```

audit prompt：

```bash
GPU_IDS=0 python qwen3vl_local/sft_new_loop_phase1/eval.py \
  --history-rgb-mode 4rgb \
  --audit-prompt \
  --split test \
  --cases-per-bin 64
```

四卡评测：

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_new_loop_phase1/eval.py \
  --history-rgb-mode 4rgb \
  --split test \
  --cases-per-bin 64
```

正式 eval 的采样合同是八个主问题各自 YES:NO = 1:1，eval variant 比例为
`all_random_order/subset_random/hierarchical_probe = 2:1:1`。

## 5. LoRA 一键评测

`eval.sh` 会解析 run 目录或 adapter 目录，然后依次运行：

1. base production
2. base audit-prompt
3. LoRA production
4. LoRA audit-prompt
5. production 错例 RGB 抽样
6. 小型审计 tar 包

推荐传 run 根目录：

```bash
bash qwen3vl_local/sft_new_loop_phase1/eval.sh \
  checkpoints/sft_new_loop_phase1_runs/latest
```

也可以直接传 adapter：

```bash
ADAPTER_DIR=checkpoints/sft_new_loop_phase1_runs/latest/final \
  bash qwen3vl_local/sft_new_loop_phase1/eval.sh
```

常用覆盖：

```bash
GPU_IDS=0,1,2,3 CASES_PER_BIN=64 AUDIT_PER_TARGET=8 \
  bash qwen3vl_local/sft_new_loop_phase1/eval.sh \
  checkpoints/sft_new_loop_phase1_runs/latest
```

如果只想直接跑 `eval.py`，不打包：

```bash
GPU_IDS=0 python qwen3vl_local/sft_new_loop_phase1/eval.py \
  --adapter-dir checkpoints/sft_new_loop_phase1_runs/latest/final \
  --split test \
  --cases-per-bin 64
```

## 6. 错例 RGB 抽样

`eval.sh` 默认已经调用错例抽样。若要单独从某个 eval 目录抽样：

```bash
python qwen3vl_local/sft_new_loop_phase1/audit_eval_cases.py \
  --eval-dir checkpoints/sft_new_loop_phase1_eval_review/<timestamp>/lora_production \
  --output-dir checkpoints/sft_new_loop_phase1_audit_samples/<tag> \
  --data-root lead_data \
  --per-target 12 \
  --overwrite
```

脚本会覆盖以下 fused 错误类型：

- Phase1 四问的 false positive / false negative。
- Phase2 `RS1/RS2/RS4/RS5` 的 false positive / false negative。
- hierarchical 的 `RS_HIGHWAY` 和 `GROUP` 错误。
- 输出格式非法。
- Phase2 多个 RS 同时 YES。
- subset 模式下额外输出未要求的问题行。

## 7. TensorBoard

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_new_loop_phase1_runs/latest
```

重点看：

- `train/loss`
- `eval/exact_match_accuracy`
- `generation_eval/exact_match_accuracy`
- `focus/*`
- `variant/*`
- `augment/*`

## 8. RGB 模式矩阵

`run_rgb_mode_matrix.sh` 用于做 phase2 风格的 4rgb / 2rgb_endpoints 对照：

```bash
bash qwen3vl_local/sft_new_loop_phase1/run_rgb_mode_matrix.sh
```

它会顺序执行 10 步：

1. base 4rgb production
2. base 4rgb audit-prompt
3. base 2rgb_endpoints production
4. base 2rgb_endpoints audit-prompt
5. 训练 4rgb LoRA
6. LoRA 4rgb production
7. LoRA 4rgb audit-prompt
8. 训练 2rgb_endpoints LoRA
9. LoRA 2rgb_endpoints production
10. LoRA 2rgb_endpoints audit-prompt

常用覆盖：

```bash
GPU_IDS=0 DATA_ROOT=/path/to/lead_data CASES_PER_BIN=64 \
  bash qwen3vl_local/sft_new_loop_phase1/run_rgb_mode_matrix.sh
```

## 9. 一键全流程

`run_full_pipeline.sh` 是按 phase3 的一键流程形式适配到 fused Phase1+Phase2 的入口。
默认流程为：

1. 生成 `visual_audit_manifest.json`
2. 构建 fused 数据集
3. base production / audit-prompt eval
4. 训练 LoRA
5. LoRA production / audit-prompt eval
6. 对 base 和 LoRA production 错例做 RGB 抽样

默认只跑 `4rgb`：

```bash
bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
```

同时跑 4rgb 和 2rgb_endpoints：

```bash
GPU_IDS=0,1,2,3 HISTORY_RGB_MODES="4rgb 2rgb_endpoints" \
  bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
```

跳过训练，评测已有 adapter：

```bash
RUN_BUILD=0 RUN_TRAIN=0 \
ADAPTER_DIR=checkpoints/sft_new_loop_phase1_runs/latest/best_generation \
GPU_IDS=0,1,2,3 \
  bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
```

只构建数据，不训练不评测：

```bash
RUN_BASE_EVAL=0 RUN_TRAIN=0 RUN_LORA_EVAL=0 RUN_AUDIT_CASES=0 \
  bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
```

常用变量：

- `PIPELINE_TIMESTAMP`：控制 pipeline 输出目录名。
- `PIPELINE_ROOT`：覆盖 pipeline 输出根目录。
- `HISTORY_RGB_MODES`：空格分隔的 RGB 模式列表。
- `RUN_VISUAL_AUDIT/RUN_BUILD/RUN_BASE_EVAL/RUN_TRAIN/RUN_LORA_EVAL/RUN_AUDIT_CASES`
  控制各阶段开关。
- `TRAIN_FOCUS_BALANCE_COUNT`：默认继承 `FOCUS_BALANCE_COUNT=9216`。
- `TRAIN_MAX_FRAME_REPEAT`：默认继承 `MAX_TRAIN_FRAME_REPEAT=10`。
- `BASE_CASES_PER_BIN/LORA_CASES_PER_BIN`：默认 64。
- `AUDIT_PER_TARGET`：默认 12。

## 10. 均衡与审计合同

训练采样分两层：

- Phase1 四个 focus 问题各自 YES:NO = 1:1。
- Phase2 四个 focus 问题 `RS1/RS2/RS4/RS5` 也各自 YES:NO = 1:1。

在这个前提下，训练保持：

- Phase1 focus 总量 = Phase2 focus 总量。
- 三种 Phase2 variant 总量为 `4:1:1`。
- `all_random_order/RS*:YES|NO` 桶为硬约束。
- Phase2 `(focus_bucket, variant)` 配额为硬约束。
- subset / hierarchical 具体 `augment_balance_key` 使用目标驱动近似均衡，并写入偏差报告。

训练和评测会记录：

- `manifest.json`
- `train_balance.json`
- `balance/epoch_*.json`
- `balance/epochs.jsonl`
- `train_run_manifest.json`
- `train_metrics.jsonl`
- `train_eval_metrics.jsonl`
- eval `metrics.json`
- `augment_target_deviation`
- `all_random_order_target_deviation`
- `phase2_focus_variant_*`
- `repeat_audit`
- per-question / per-variant / pattern diagnostics

正式训练前建议看：

- `augment_target_deviation.max_abs_delta`
- `augment_target_deviation.total_abs_delta`
- `repeat_audit.max_repeat`
- `repeat_audit.mean_repeat`
- 八个 focus 是否全部 exact。
- 三种 variant 是否 exact。
