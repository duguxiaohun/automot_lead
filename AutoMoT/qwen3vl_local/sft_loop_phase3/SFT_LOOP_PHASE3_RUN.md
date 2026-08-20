# SFT Loop Phase3 运行说明

`sft_loop_phase3` 是 EVENT 级 loop：先把上层 Phase2 的 RS 结果构造成上一轮
assistant answer block，再在后一轮 user turn 里基于当前 RS gate 问事件；训练和
eval 都走这个多轮 chat 形态，更接近真实 KV 续接分布。

- `RS1/R2`：问 `UE1/UE3/UE5`；三项全 `NO` 表示本阶段折叠为 `RE`。
- `RS4/R5`：只问 `UE6`；`NO` 表示本阶段折叠为 `RE`。
- `INVALID_RS_CONTEXT`：只在注入 wrong-RS 上下文时为 `YES`，且所有 UE 必须为 `NO`。

本阶段暂不训练 `UE2/UE4/UE7`；`UE8` 也先折成 `RE`，避免把普通路口停车/排队误学成独立异常。

以下命令均从 `AutoMoT/` 目录运行，只读取本地 `checkpoints/Qwen3-VL-4B-Instruct`。

## 1. 构建索引

```bash
python qwen3vl_local/sft_loop_phase3/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_loop_phase3_data \
  --test-ratio 0.10 \
  --val-ratio 0.05
```

构建器会剔除 `noScenarios`、异常时长 route、缺数据 route，以及默认剔除
`visual_label_risk` 风险帧。默认每个 split 中 `UE1/UE3/UE5/UE6` 正类
按 `1:1:1:1` 采样，`RE` 等于单个 UE 桶，wrong-RS invalid 为主数据的 `20%`。
invalid 会先按 `source_class / true_rs / fake_rs` 展开，再按 source class 与
true/fake RS signature 轮转抽样，避免某个大场景或某类 RS 错误支配 invalid 增强。
R3/highway invalid 会同时展开到 `RS1/RS2/RS4/RS5` 四种 fake gate。

可调参数：

```bash
python qwen3vl_local/sft_loop_phase3/build_dataset.py \
  --target-per-ue 4096 \
  --regular-multiplier 1.0 \
  --invalid-ratio 0.20
```

## 2. Base Qwen 测试

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase3/eval.py \
  --history-rgb-mode 4rgb
```

结果写入 `checkpoints/sft_loop_phase3_eval/`。重点看：

- `metrics.json`：UE1/UE3/UE5/UE6/INVALID 的 precision/recall/F1，以及
  `invalid_contract` 下的 `invalid_joint_ok_rate`、`invalid_ue_all_no_rate`。
- `cases_rank*.jsonl` / `cases.jsonl`：逐 case 的 prompt/spec/raw output。
  `prompt` / `phase3_user_prompt` 是实际后一轮 user turn；`phase2_user_prompt`、
  `phase2_assistant` 和 `actual_chat_messages` 记录真实多轮输入结构。
- `error_cases/`：错误样本 RGB。

## 3. LoRA 训练

默认训练节奏与 `sft_loop_phase2_augment` 对齐：`NUM_EPOCHS=3`、
`FOCUS_BALANCE_COUNT=1024`、`EVAL_STEPS=2000`、`GENERATION_EVAL_STEPS=2000`、
`SAVE_STEPS=20000`、`GRAD_ACCUM=1`。phase3 只学 RS gate 下的
`UE1/UE3/UE5/UE6/INVALID` 二值事件判断，默认强度先按 augment 路线走；
UE1/UE3/UE5/UE6 正类保持 `1:1:1:1`，但训练默认把 `RE` all-NO 桶放大为单个 UE 桶
的 `2.0x`（`REGULAR_FOCUS_MULTIPLIER=2.0`），用于压低复杂普通交通被误报成 UE 的
false positive。2026-08-20 对 `audit_bundle/audit_lora_production` 的逐帧 RGB 审计显示
UE6 FN 是最大剩余错误、invalid FN 高于 FP，因此默认另外使用
`UE6_FOCUS_MULTIPLIER=1.5` 和 `INVALID_FOCUS_MULTIPLIER=1.25`；UE1/UE3/UE5 仍默认
`1.0`，需要消融时可分别设置 `UE1_FOCUS_MULTIPLIER` / `UE3_FOCUS_MULTIPLIER` /
`UE5_FOCUS_MULTIPLIER`。eval/generation 仍保持均衡口径，不受这些训练倍率影响。

想进一步充分训练可显式加大 `FOCUS_BALANCE_COUNT` 或设置 `MAX_STEPS` 固定总步数。
若 UE recall 明显不足，可临时调低 `REGULAR_FOCUS_MULTIPLIER=1.0`；若 RE 仍大量误报
为 UE3/UE6，可提高到 `2.5` 或 `3.0` 后对比 validation FP/FN。

Smoke：

```bash
GPU_IDS=0 \
CHECK_MAX_STEPS=2 \
CHECK_FOCUS_BALANCE_COUNT=2 \
bash qwen3vl_local/sft_loop_phase3/train.sh check
```

正式四卡：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase3/train.sh ddp
```

更充分训练示例：

```bash
GPU_IDS=0,1,2,3 \
FOCUS_BALANCE_COUNT=2048 \
REGULAR_FOCUS_MULTIPLIER=2.0 \
INVALID_FOCUS_MULTIPLIER=1.25 \
UE6_FOCUS_MULTIPLIER=1.5 \
bash qwen3vl_local/sft_loop_phase3/train.sh ddp
```

默认输出到：

```text
checkpoints/sft_loop_phase3_runs/run_event_gate_format_supervised_4rgb/<YYYYmmdd_HHMMSS>/
```

训练目录包含 `tb/`、`train_run_manifest.json`、`train_balance.json`、
`train_metrics.jsonl`、`train_eval_metrics.jsonl`、`generation_val_cases.jsonl`、
`best_generation/`、`best_val/`、`checkpoint-*` 和 `final/`。

DDP 训练按 global step 对齐所有 rank：每步用 `step_in_epoch * world_size + rank`
取样，尾部不整除时 wrap padding；样本过长被跳过时会跑一次短图文 DDP forward，
再用 `logits.sum() * 0` 做 zero-loss backward，避免某些 rank 正常 forward、
某些 rank 只做参数零和导致 reducer 边界不一致。
`GRAD_ACCUM>1` 且最后不足一个 accumulation window 时，训练结束前会 flush
这段残余梯度并同步 scheduler；若 `SAVE_STEPS` 落在 accumulation window 中间，
`checkpoint-*` 会延迟到下一次 optimizer step 后保存。

## 3.1 一键完整流程

从数据集构建、base 测试、训练、LoRA 测试到错例 audit，可直接跑：

```bash
bash qwen3vl_local/sft_loop_phase3/run_full_pipeline.sh
```

默认跑 `4rgb`。同时跑四帧和首尾两帧对照：

```bash
GPU_IDS=0,1,2,3 \
HISTORY_RGB_MODES="4rgb 2rgb_endpoints" \
bash qwen3vl_local/sft_loop_phase3/run_full_pipeline.sh
```

常用开关：

```bash
# 只重跑 eval/audit，不重新训练；adapter 可显式指定。
RUN_BUILD=0 RUN_TRAIN=0 \
ADAPTER_DIR=checkpoints/sft_loop_phase3_runs/latest/best_generation \
bash qwen3vl_local/sft_loop_phase3/run_full_pipeline.sh

# 跳过 audit-prompt 额外评测，只保留 production eval 和错例抽样。
RUN_AUDIT_PROMPT_EVAL=0 bash qwen3vl_local/sft_loop_phase3/run_full_pipeline.sh
```

pipeline 产物写到 `checkpoints/sft_loop_phase3_pipeline/<timestamp>/`；训练产物仍写到
`checkpoints/sft_loop_phase3_runs/run_event_gate_format_supervised_<mode>/<timestamp>/`，
并更新 `checkpoints/sft_loop_phase3_runs/latest`。

## 4. LoRA 测试

LoRA eval 从 adapter config 读取 `history_rgb_mode`：

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase3/eval.py \
  --adapter-dir checkpoints/sft_loop_phase3_runs/latest
```

`eval.py` 会优先使用 `latest/best_generation`，再回退 `best_val`、`final`。

## 5. 错例 Audit

```bash
python qwen3vl_local/sft_loop_phase3/audit_eval_cases.py \
  --eval-dir checkpoints/sft_loop_phase3_eval/<eval_run>/<timestamp> \
  --output-dir checkpoints/sft_loop_phase3_audit_samples/<name> \
  --per-target 12 \
  --overwrite
```

默认抽 `UE1/UE3/UE5/UE6` 的 FP/FN、`INVALID_RS_CONTEXT` 的 FP/FN、
`invalid_context_not_all_no`、`multi_ue_yes` 和格式非法样本。

## 5.1 独立评测打包

给定一个 LoRA adapter/run 目录，直接跑 base + LoRA production/audit-prompt eval、
错例 RGB audit、visual audit manifest，并生成不超过 `30MB` 的审计压缩包：

```bash
ADAPTER_DIR=checkpoints/sft_loop_phase3_runs/latest/best_generation \
bash qwen3vl_local/sft_loop_phase3/eval.sh
```

默认四卡 `GPU_IDS=0,1,2,3`。base eval 默认从 adapter config 读取同一个
`history_rgb_mode`，确保 base/LoRA 输入合同一致；显式设置 `HISTORY_RGB_MODE=...`
时才覆盖。输出到 `checkpoints/sft_loop_phase3_eval_review/<timestamp>/`，压缩包为
`sft_loop_phase3_audit_bundle.tar.gz`。包内包含 metrics/report/case JSONL、
adapter/run-root 小型元信息（不含权重），以及按错误 variant/target 分层抽样的
降采样 error RGB，供代码与 prompt 审计。

## 6. TensorBoard

```bash
bash qwen3vl_local/tb_serve.sh \
  checkpoints/sft_loop_phase3_runs/latest
```

重点看 `train_window/answer_acc/*`、`val/per_question` 对应 scalar、
`val/invalid_joint_token_ok_rate`、`val_generation/exact_accuracy`、
`val_generation/format_valid_rate` 和 `val_generation/invalid/*joint_ok_rate`。

## 7. 当前 pipeline 结果解读口径

`checkpoints/sft_loop_phase3_pipeline/20260820_003925` 显示 base Qwen 基本是 all-NO
基线，LoRA 后 production exact 约 `0.77`，格式合法率为 `1.0`，说明输出合同和训练链路
已经正常。`checkpoints/audit_bundle` 的 v2 复训结果 production exact 约 `0.7786`，
相对旧 v1 小幅提升，并把 `INVALID_RS_CONTEXT` FP 从 14 降到 4、`UE3` FP 从 13 降到 6，
但 `UE6` FN 从 12 升到 20、invalid FN 从 10 升到 15。逐帧 RGB 审计后的主要剩余错误是：

- `RE` 被误报成 `UE3/UE6`：普通路口转弯/横穿车辆、事故/夜间复杂交通容易被当成动态占道或违规冲突。
- 弱证据正样本漏报：早期帧、left-pad 历史、夜间/雾天中 UE1 hard-brake、UE3 cut-in、UE5 oncoming invasion、UE6 red-light/right-of-way violation 的视觉证据不足。
- `INVALID_RS_CONTEXT` 与低能见度/拥堵场景混淆：invalid 应只表示 RS gate 明显不适用，不能因为 fog/night/blocked traffic 或没看到 UE 就报 YES；但当道路拓扑本身清楚显示 highway/plain road/local junction mismatch 时，fog/night 也不应阻止 invalid。
- 一部分 invalid FP 在 RGB 上反而像上游 RS/GT 可疑样本，例如 GT 为 R1 但画面清楚显示信号灯/斑马线路口；这类需要回流到 Phase2/标签复核，不要单靠 phase3 prompt 压掉。

因此当前默认使用 prompt `sft_loop_phase3_event_gate_visual_v3`：输出格式不变，在 v2
“证据弱则 RE/all-NO”的基础上，补强了三点：UE6 对已进入冲突区的横向车辆不因夜雾直接
降为 NO；UE1/UE3 通过“同车道急减速 vs 侧向进入 ego corridor”拆开；invalid 通过道路拓扑
而不是事件难度判定。旧 v1/v2 adapter 建议按本文件重新训练后再比较。
