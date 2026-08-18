# SFT Loop Phase2 Augment 运行说明

`sft_loop_phase2_augment` 是 `sft_loop_phase2` 的随机问法增强版。输入仍是 LEAD 四帧
RGB history，输出只回答当前 prompt 中被问到的答案行。

三类训练 / 评测问法：

- `all_random_order`：`RS1/RS2/RS4/RS5` 四题全问，顺序随机。
- `subset_random`：只问 `1/2/3` 个 RS 细问题；全 `NO` 只表示被问到的题都不是，不代表高速。
- `hierarchical_probe`：依次问 `HIGHWAY`、`GROUP`、`DETAIL`。

示例输出：

```text
RS4: YES
RS2: NO
```

或：

```text
HIGHWAY: NO
GROUP: YES
DETAIL: YES
```

以下命令均从 `AutoMoT/` 目录运行，只读取本地
`checkpoints/Qwen3-VL-4B-Instruct`。

## 1. 构建索引

```bash
python qwen3vl_local/sft_loop_phase2_augment/visual_audit.py \
  --collection-dir keyframe_filter/collection_output \
  --output checkpoints/sft_loop_phase2_augment_data/visual_audit_manifest.json

python qwen3vl_local/sft_loop_phase2_augment/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_loop_phase2_augment_data \
  --test-ratio 0.10 \
  --val-ratio 0.05
```

构建器会剔除 `noScenarios`、异常时长 route、缺数据 route，以及默认剔除
`visual_label_risk` 风险帧；不会改原始 RS 标签。

## 2. Base Qwen 测试

先测 production，再测 audit。audit 只用于看证据输出，不能和 production 分数直接横比。

```bash
# production
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase2_augment/eval.py \
  --history-rgb-mode 4rgb

# audit
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase2_augment/eval.py \
  --history-rgb-mode 4rgb \
  --audit-prompt
```

结果写入 `checkpoints/sft_loop_phase2_augment_eval/`。重点看：

- `metrics.json`：总体 exact、variant exact/valid、逐问题 precision/recall/F1。
- `summary.md`：人读摘要。
- `cases_rank*.jsonl` / `variant_cases/`：逐 case 的 prompt/spec/raw output。
- `error_cases/`：错误样本 RGB。
- `answer_pattern_diagnostics`：全 NO、多 YES、invalid、subset 未问 RS 泄漏。

## 3. LoRA 训练

默认输出到：

```text
checkpoints/sft_loop_phase2_augment_runs/<run_name>/<YYYYmmdd_HHMMSS>/
```

`train.log`、`tb/`、`train_run_manifest.json` 都在同一个时间目录下。launcher 同时维护：

```text
checkpoints/sft_loop_phase2_augment_runs/latest
checkpoints/sft_loop_phase2_augment_runs/latest_check
```

链路 smoke 只确认加载、DDP、loss、保存是否正常：

```bash
GPU_IDS=0 \
CHECK_MAX_FRAMES=2000 \
CHECK_FOCUS_BALANCE_COUNT=2 \
CHECK_MAX_STEPS=2 \
bash qwen3vl_local/sft_loop_phase2_augment/train.sh check
```

正式四卡训练保留 val 与 free-generation probe：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase2_augment/train.sh ddp
```

训练目录关键产物：

- `tb/`：TensorBoard。
- `train_run_manifest.json`：本次 run 的路径、数据、prompt、eval 配置。
- `train_balance.json`：训练 / val / generation probe 采样分布。
- `train_metrics.jsonl`：每个 log window 的 loss、LR、variant 比例、训练 token acc。
- `train_eval_metrics.jsonl`：训练中 teacher-forced val 与 free-generation val 指标。
- `generation_val_cases.jsonl`：训练中 free-generation raw output 与解析结果。
- `best_val/`：通过 generation 格式闸门后的最低 val loss adapter。
- `checkpoint-*` / `final/`：周期保存与最终 adapter。

默认 `FOCUS_BALANCE_COUNT=1024`，`EVAL_STEPS=2000`，`GENERATION_EVAL_STEPS=2000`，
`SAVE_BEST_VAL=1`。正式训练不要关闭 eval；只在定位吞吐或显存问题时临时覆盖这些变量。

## 4. LoRA 测试

LoRA eval 会从 adapter 配置自动读取 `history_rgb_mode`，不要再传 `--history-rgb-mode`。

```bash
# production
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase2_augment/eval.py \
  --adapter-dir checkpoints/sft_loop_phase2_augment_runs/latest/best_val

# audit
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase2_augment/eval.py \
  --adapter-dir checkpoints/sft_loop_phase2_augment_runs/latest/best_val \
  --audit-prompt
```

如果训练没有生成 `best_val/`，再评测 `final/`：

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase2_augment/eval.py \
  --adapter-dir checkpoints/sft_loop_phase2_augment_runs/latest/final
```

## 5. 错例抽样 Audit

全量 production / audit eval 后，从 `error_cases/` 抽小样本并从 `lead_data` 补真实 RGB：

```bash
python qwen3vl_local/sft_loop_phase2_augment/audit_eval_cases.py \
  --eval-dir checkpoints/sft_loop_phase2_augment_eval/<eval_run>/<timestamp> \
  --output-dir checkpoints/sft_loop_phase2_augment_audit_samples/<name> \
  --per-target 12 \
  --overwrite
```

常用 `<eval_run>`：`base_rs_augmented_final_4rgb`、
`base_rs_augmented_final_4rgb_audit`、`lora_rs_augmented_final_4rgb`、
`lora_rs_augmented_final_4rgb_audit`。

默认抽 `rs2_fn/highway_fn/rs1_fp/rs4_fn/rs5_fn/multi_yes`。每个 case 目录包含
`case.json`、`audit_note.md`、`rgb/`。这个脚本不运行模型。

## 6. TensorBoard

```bash
bash qwen3vl_local/tb_serve.sh \
  checkpoints/sft_loop_phase2_augment_runs/latest
```

重点看：

- `train_window/loss_mean`
- `train_window/answer_acc/*`
- `val/loss`
- `val/value_token_acc`
- `val/format_token_acc`
- `val/variant/*_exact`
- `val_generation/exact_accuracy`
- `val_generation/format_valid_rate`
- `val_generation/metric/*_acc`
- `val_generation/pattern/*`

训练中优先使用 `best_val/`；test split 只用于最终泛化确认。

## 7. RGB 模式矩阵

如需一次跑完 4rgb / 2rgb_endpoints 的 base、训练、LoRA production、LoRA audit：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase2_augment/run_rgb_mode_matrix.sh
```
