# SFT Loop Phase2 Augment 运行说明

`sft_loop_phase2_augment` 从 `sft_loop_phase2` 复制而来，但训练/评测不再固定四问顺序。
每个 case 仍然只做一次 RGB prefill + 一次 assistant 生成；如果同一 prompt 里有多题，
模型在自回归生成后续行时天然能看到前面行的 KV cache。

三类问法按目标比例 `2:1:1` 采样：

- `all_random_order`：问 `RS1/RS2/RS4/RS5` 四题，但输出顺序可复现随机；定义沿用原 Phase2 prompt。
- `subset_random`：随机问 `1/2/3` 个 RS 细问题，三种题数均衡；实际输出行里的 RS×YES/NO 按可达边际均衡。由于 RS 标签是 one-hot，q3 每个 case 最多只有一个 YES，所以 q3 的可达目标是每个 RS 约 YES:16 / NO:32，而不是机械 1:1。允许全 `NO`，全 `NO` 只表示“被问到的题都不是”，不代表高速。
- `hierarchical_probe`：固定先问 `HIGHWAY`，再问一个专门设计的组级几何问题，最后问一个 RS 细问题；组级问题不是简单拼接两个 RS 定义，也避免把组题做成 `not HIGHWAY` 的反面题。组题包含 ordinary lane-following、open surface path、junction control、local right-of-way rule、shared/conflicting space 等视角，其中 `LOCAL_RIGHT_OF_WAY_RULE` 覆盖信号/stop/yield 标线标志，也覆盖无灯优先权、几何和 gap acceptance。采样器同时约束 HIGHWAY、GROUP、DETAIL 三组 YES/NO 边际分布。

输出格式随问法变化，例如：

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

训练只监督当前 prompt 中出现的答案行；未被问到的 RS 不施加隐藏 loss。

## 1. 构建索引

从 `AutoMoT/` 目录运行：

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

构建逻辑仍剔除 `noScenarios`、异常时长 route、缺数据 route，以及默认剔除 `visual_label_risk`
风险帧；不会改原始 RS 标签。

## 2. Base 评测

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase2_augment/eval.py \
  --history-rgb-mode 4rgb

GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase2_augment/eval.py \
  --history-rgb-mode 4rgb \
  --audit-prompt
```

结果写到 `checkpoints/sft_loop_phase2_augment_eval/`。`metrics.json` 会统计：
总体 exact、三类增强问法的 valid/exact、`RS1/RS2/RS4/RS5/HIGHWAY/GROUP:*` 的 YES/NO
二分类指标、每个 balance key 的采样数量，以及逐 case 的 prompt/spec/raw output。
`answer_pattern_diagnostics` 会单独报告 subset 的全 NO 是否发生在高速/非高速 GT、
多 YES、invalid 和未被问到的 RS 行泄漏，防止把 subset 全 NO 重新误读成高速。

### 2.1 只抽小样本错例审计

如果全量 `error_cases/` 搬运后 RGB 损坏或太大，不需要重跑 Qwen。可以直接读取
eval 目录里的 `case.json`，再从本机 `lead_data` 补真实 RGB，按错误类型抽一个小目录：

```bash
python qwen3vl_local/sft_loop_phase2_augment/audit_eval_cases.py \
  --eval-dir checkpoints/sft_loop_phase2_augment_eval/base_rs_augmented_final_4rgb/20260817_172821 \
  --output-dir checkpoints/sft_loop_phase2_augment_audit_samples/base_4rgb_20260817 \
  --per-target 12 \
  --overwrite
```

默认抽 `rs2_fn/highway_fn/rs1_fp/rs4_fn/rs5_fn/multi_yes` 六类，每个 case 目录包含
`case.json`、`audit_note.md` 和实际输入的 RGB。这个脚本只做文件整理，不运行模型。

## 3. 训练与 LoRA 评测

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase2_augment/train.sh ddp

GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase2_augment/eval.py \
  --adapter-dir checkpoints/sft_loop_phase2_augment_runs/run_rs_augmented_format_supervised_4rgb/final
```

训练启动时会先读取 `frame_index.jsonl` 并构造增强采样 work list，然后才加载 Qwen。
为了避免默认全量均衡在 CPU 阶段等待太久，先做一个只读小样本的链路 smoke：

```bash
mkdir -p checkpoints/sft_loop_phase2_augment_runs

GPU_IDS=0 \
CHECK_MAX_FRAMES=2000 \
CHECK_FOCUS_BALANCE_COUNT=2 \
CHECK_MAX_STEPS=2 \
RUN_LOG=checkpoints/sft_loop_phase2_augment_runs/check_rs_augmented_4rgb.log \
bash qwen3vl_local/sft_loop_phase2_augment/train.sh check
```

确认能看到 `[startup] loading Qwen + LoRA...` 且 loss 正常打印后，再启动正式训练。
`FOCUS_BALANCE_COUNT>0` 会走目标桶快速采样；`FOCUS_BALANCE_COUNT=0` 表示按训练 split 中
最小原始桶做全量均衡，适合最终审计型长训，不适合第一次 smoke。

常规 4 卡训练建议先关闭周期验证，训练完单独 eval：

```bash
mkdir -p checkpoints/sft_loop_phase2_augment_runs

GPU_IDS=0,1,2,3 \
FOCUS_BALANCE_COUNT=1024 \
EVAL_STEPS=0 \
GENERATION_EVAL_STEPS=0 \
OUTPUT_DIR=checkpoints/sft_loop_phase2_augment_runs/run_rs_augmented_4rgb \
RUN_LOG=checkpoints/sft_loop_phase2_augment_runs/run_rs_augmented_4rgb.log \
bash qwen3vl_local/sft_loop_phase2_augment/train.sh ddp
```

adapter 保存 `sft_loop_phase2_augment_adapter_config.json`，LoRA eval 会从该配置读取
`history_rgb_mode`，因此评测 LoRA 时不要再传 `--history-rgb-mode`。
配置里的 `production_prompt_sha256` 是完整增强 prompt contract 指纹，会覆盖
all-random、subset、hierarchical、所有 group/detail 模板，而不是单个默认 prompt。
该指纹使用稳定 JSON 规范化，`GROUP_DEFINITIONS` 内部的 set 会排序后写入 hash，
不会受 `PYTHONHASHSEED` 影响。

## 4. RGB 模式矩阵

```bash
bash qwen3vl_local/sft_loop_phase2_augment/run_rgb_mode_matrix.sh
```

脚本会顺序跑 4rgb / 2rgb_endpoints 的 base production/audit、LoRA 训练、LoRA
production/audit。需要换卡时覆盖 `GPU_IDS=4,5`。

## 5. TensorBoard

```bash
bash qwen3vl_local/tb_serve.sh \
  checkpoints/sft_loop_phase2_augment_runs/run_rs_augmented_format_supervised_4rgb
```

重点看 `train/loss`、`val/loss`、`val/value_token_acc`、`val/format_token_acc`、
`val/variant/*_exact` 和 `val_generation/variant/*_valid`。如果某个增强问法格式坍塌，
先看 `generation_val_cases.jsonl` 里该 variant 的 raw output 和 `augment_spec`；
完整的 generation probe pattern 分布会追加写入 `generation_val_cases_pattern_reports.jsonl`，
字段和 eval 的 `answer_pattern_diagnostics` 对齐。
