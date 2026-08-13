# SFT Loop Phase2 运行说明

Phase2 是不带 memory 的第二轮 loop，针对当前最新帧独立回答四个
`ROAD_STRUCTURE` 二元问题：

```text
RS1: YES|NO
RS2: YES|NO
RS4: YES|NO
RS5: YES|NO
```

R3 不在本轮单独提问，因为 Phase1 已经负责 `HIGHWAY`。可见的 R3
高速主路、匝道、合流、分流或驶出帧仍会保留为鲁棒性负例：四项都应为 `NO`，不会被丢弃或改标为 R1。

以下所有命令均从 `AutoMoT/` 目录运行，只读取本地
`checkpoints/Qwen3-VL-4B-Instruct`。

## 1. 审计并构建索引

仓库内随 Phase2 代码提交了轻量覆盖证明 `phase2_rgb_audit_coverage.json`，其中只有
42 个场景、197 个场景-Town、582 条已完成全帧 RGB 审计 route 的计数，不含 RGB、sheet 或大审计产物。因此远程没有 `collection_output` 审计目录也能直接运行。

以下命令会写本次运行的轻量审计清单：本地若有完整 Phase1 审计目录，会优先核对该目录；远程没有时自动使用随代码提交的覆盖证明。它不会重新生成或复制 RGB：

```bash
python qwen3vl_local/sft_loop_phase2/visual_audit.py \
  --collection-dir keyframe_filter/collection_output \
  --output checkpoints/sft_loop_phase2_data/visual_audit_manifest.json
```

然后在含有 LEAD 数据的服务器上构建干净索引：

```bash
python qwen3vl_local/sft_loop_phase2/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_loop_phase2_data \
  --test-ratio 0.10 \
  --val-ratio 0.05
```

构建器也会自动使用该覆盖证明；本地完整审计目录只作为可选的额外核对来源。它会剔除 `noScenarios`、异常时长 route、缺失数据 route，以及少量已明确标为
`visual_label_risk` 的视觉标签风险帧；它绝不会修改原始 RS 标签。要构建单独的鲁棒性/带噪声标签索引，请使用不同输出目录并显式保留风险帧：

```bash
python qwen3vl_local/sft_loop_phase2/build_dataset.py \
  --output-dir checkpoints/sft_loop_phase2_data_with_visual_risk \
  --include-visual-risk
```

第一次实验不要使用带风险帧的索引作为干净基线；它只用于已经得到干净结果后的具名鲁棒性实验。

`manifest.json.focus_bin_availability` 中 train / val / test 三个 split 都必须有以下八个桶：

```text
RS1:YES / RS1:NO
RS2:YES / RS2:NO
RS4:YES / RS4:NO
RS5:YES / RS5:NO
```

最终训练和评测会将这八个桶采样到严格相等。一个 focus case 仍然要求模型回答全部四问；只是在该模块里 focus 主问题的 `YES:NO` 保持 1:1。
`manifest.json.highway_negative_counts` 会单独统计被保留的 R3 四项全 `NO` 样本。

正式训练默认 `FOCUS_BALANCE_COUNT=0`：每个 epoch 自动取最少原始桶的全部样本作为每桶目标，因此当前索引中每桶为 `62,208`，全局每 epoch 为 `497,664` 个 case。默认训练 3 个 epoch，合计约 `1,492,992` 个全局 case。富余桶会在每个 epoch 使用不同稳定随机种子重新抽样；最少桶会被完整使用。若显式将 `FOCUS_BALANCE_COUNT` 设得高于某桶的原始量，该稀缺桶会自动重复采样，仍保持八桶严格相等。

这是正式训练规模，不再沿用小样本 smoke 的频率：默认每 `10,000` 个 rank-local step 做一次固定均衡 val、每 `20,000` step 保存一个 checkpoint，并用前 `2,000` step warmup。需要试跑时使用 `train.sh check`；需要改变正式频率时可覆盖 `EVAL_STEPS`、`SAVE_STEPS`、`WARMUP_STEPS`。

## 2. 先测 Base Qwen 的提示词

正式 production 指标只要求模型输出四行 `YES/NO`。`--audit-prompt` 是单独的诊断提示词，会额外输出简短、可见的 `EVIDENCE_RS*` 证据。因为两者输入提示词不同，audit 指标不能与 production 指标直接横向比较。

```bash
# 4 卡 base Qwen 正式 production 评测
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase2/eval.py \
  --history-rgb-mode 4rgb

# 同一固定测试集上的可见证据诊断
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase2/eval.py \
  --history-rgb-mode 4rgb \
  --audit-prompt
```

Base Qwen 评测可以显式选择 `2rgb_endpoints`；它严格表示原始历史序列 `[0,3]`，即第一张和第四张 RGB：

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase2/eval.py \
  --history-rgb-mode 2rgb_endpoints
```

结果会按时间戳写入 `checkpoints/sft_loop_phase2_eval/` 下，不会覆盖旧结果。`metrics.json` 包含全局逐问题指标，以及四个主任务模块的均衡主问题 `TP/FP/FN/TN`。每个模块的 case 记录在 `task_cases/<RS>/`；只有主问题错误时才会把模型实际看到的 RGB 复制到 `error_cases/<RS>/`。

## 3. 训练并复测 LoRA

```bash
# 输入四张 RGB；训练过程中会定期评测 route-disjoint 的 val split。
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase2/train.sh ddp

# 只输入第一张和第四张 RGB。这会训练另一份 adapter，不能当作 4rgb adapter 的运行时切换。
HISTORY_RGB_MODE=2rgb_endpoints \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase2/train.sh ddp
```

如需让最少桶重复、进一步扩大每个 epoch 的均衡规模，例如每桶 `131072` 个 case：

```bash
FOCUS_BALANCE_COUNT=131072 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase2/train.sh ddp
```

adapter 会保存 `sft_loop_phase2_adapter_config.json`，其中包含提示词 SHA 和 RGB 模式。LoRA 评测会从 adapter 自动读取模式，因此评测 LoRA 时不要传 `--history-rgb-mode`。

```bash
# 4rgb LoRA 的正式 production 评测，再做 audit 诊断
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase2/eval.py \
  --adapter-dir checkpoints/sft_loop_phase2_runs/run_rs_four_binary_final_4rgb/final

GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase2/eval.py \
  --adapter-dir checkpoints/sft_loop_phase2_runs/run_rs_four_binary_final_4rgb/final \
  --audit-prompt
```

索引构建完成后，如需一次跑完四卡 4rgb / 2rgb 的完整对照矩阵：

```bash
bash qwen3vl_local/sft_loop_phase2/run_rgb_mode_matrix.sh
```

该脚本顺序执行 base production/audit、两种 RGB 模式的 LoRA 训练，以及对应的 LoRA production/audit。仅在需要换卡时覆盖，例如：`GPU_IDS=4,5 bash qwen3vl_local/sft_loop_phase2/run_rgb_mode_matrix.sh`。
