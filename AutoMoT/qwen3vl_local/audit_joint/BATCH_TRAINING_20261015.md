# Phase3 实测配置与 Phase4 独立对照

> 最新复审修正见 [连续batch复审](BATCH_AUDIT_FIXES_20261015.md)：通用Phase3入口默认按实际更新数预热10%，本页B2专用入口仍显式500步；同锚点action补偿升级v2，probe区分容量与可启动候选。最终同步后生成新学生view。

> 发布更新：本轮已获用户授权按白名单发布，统一 v60_batch_profiles_20261015（523源），发布前209项相关回归通过。同步最终发布源码后执行本文入口；历史段落中的“未push”描述保留当时状态。

Phase3 采用四卡、每卡 B2、gradient checkpointing 开、累积 2 次。一次前向全局 8 片段，每次更新 16 片段。新增 `train_phase3_batch.sh` 作为该配置的明确入口；直接使用旧 `train.sh` 不会自动切换配置。两图 `[t-2,t]`／步长 2，四图 `[t-3,t-2,t-1,t]`／步长 1。

## 已收到的服务器证据

用户交回 `p3_batch_e2e_20261015_204857` 的八条结果，都是四 H20、同 160 个呈现、每更新 16 样本、两次重复；八条均 measured 且满足 15% 显存余量。以下是两个重复的吞吐中位数和最大 reserved，并非模型质量验收，也未独立读取服务器逐 rank 文件。

| 模式 | B1，累积4 | B2，累积2 | 吞吐提升 | B2 最大预留显存 |
|---|---:|---:|---:|---:|
| 两图 | 1.832 片段/s | 2.006 片段/s | 9.53% | 29.47 GiB |
| 四图 | 1.236 片段/s | 1.342 片段/s | 8.60% | 40.55 GiB |

之前 detailed 对照也支持 B2，B4 未更快。无需重复 Phase3 容量／速度探测。这里只与开启 checkpointing 的固定样本 B1 比较，不把这个百分比套到历史全部训练。

## Phase3 优化训练入口

在服务器 AutoMoT 目录及既有 automot 环境，同步本轮代码后运行。沿用已准备的正式索引，不重建 Phase3 数据。两模式顺序运行，输出目录分开。

```bash
export INDEX=checkpoints/phase3_prepare_20261012_231356_193369568/data/frame_index.jsonl
export DATA_ROOT=lead_data MODEL_DIR=checkpoints/Qwen3.5-4B
export PROMPT_VARIANT=v23_rgb_stage_candidate_20261006
export GPU_IDS=0,1,2,3
export BATCH_SIZE=2 GRAD_ACCUM=2 GRADIENT_CHECKPOINTING=1
export NUM_EPOCHS=3 FOCUS_BALANCE_COUNT=1024
export OUTPUT_DIR="checkpoints/phase3_batch2_$(date +%Y%m%d_%H%M%S_%N)"
for mode in 2rgb_short 4rgb; do
  HISTORY_RGB_MODE="$mode" bash qwen3vl_local/audit_joint/train_phase3_batch.sh ddp || break
done
```

入口使用 `$OUTPUT_DIR/run_<RUN_TAG>_<mode>`，已存在则拒绝；不指定 GPU_IDS 时自动选择四卡。可用同一入口的 `sampling` 模式只检查采样计划。遗留 BATCH_SIZE、GRAD_ACCUM、GRADIENT_CHECKPOINTING、ACTION_OUTPUT_MODE 或 ANCHOR_STRIDE 与该配置冲突时会明确报错，不静默覆盖。

warmup 默认 500 次 optimizer 更新，保留旧四卡 B1／累积1 默认的 `2000×4=8000` 个呈现；评估和生成评估默认每 1000 个 microbatch，即 `1000×8=8000` 个呈现，保存间隔同理改为 10000。三轮各10240呈现共1920次更新，避免沿用 warmup2000 导致整场都在预热。以上是按样本量换算，未证明学习率／收敛最优；显式 WARMUP_STEPS、EVAL_STEPS 等仍可覆盖，改变训练预算时应一起核对。训练输出保留解析后的实际参数和采样／曝光记录。此入口从基座开始新实验，不声称旧 Phase3 adapter 可恢复优化器。

## Phase4 固定更新批量对照

从现有data4生成视图、源码检查和四卡实测的完整可执行命令见 [Phase4实测](PHASE4_BENCHMARK_20261015.md)。当前默认独立比较B1／B2／B4。

复用已完成的 `checkpoints/phase4_v57_20261014_232157_762998141/data4`。先按 [正式包验收说明](PHASE4_V57_ACCEPTANCE_20261015.md) 在最终同步后的源码上生成当前学生视图。此轮 `run_automatic.sh` 变更属于学生合同，已用此前源码生成的 view 需重新生成；原生生产库无需重放或重建。

```bash
# DATASET 指向这次生成的 student/view；不是原生 data4。
GPU_IDS=0,1,2,3 DATASET="${DATASET:?设置当前学生view}" \
OUTPUT_DIR="checkpoints/p4_batch_e2e_$(date +%Y%m%d_%H%M%S_%N)" \
bash qwen3vl_local/audit_joint/benchmark_phase4_batch.sh
```

默认两／四图、B1／B2／B4、两个重复、end_to_end；每 event 32 个计时呈现，320 个全局呈现、10 次更新；每 event 另预热一次更新。有效更新量固定 32，与 Phase4 四卡 B1／累积8一致，因此 B2 累积4、B4累积2。入口末尾可加工具参数进行明确的独立对照。

Phase4 benchmark 现在直接调用正式训练的 `encode_batch`：逐样本传输后在设备上组批。此前工具的 CPU 组批后统一传输路径不再用于 Phase4。detailed 模式把该整段记为 `encode_batch_device`，不误标为纯 CPU；end_to_end 的异步阶段耗时仍仅叫 host_submission_seconds。Phase3 编码路径保持原样。benchmark 不保存 checkpoint，不替代正式准入，不证明收敛。

交回 `summary.json` 和 `request.json` 后，根据各模式结果选择 B。Phase4 目前没有四卡实测结论，默认正式入口仍 B1。若独立实测选定 B2，正式 `run_automatic.sh` 设 `BATCH_SIZE=2` 即自动累积4；默认 `UPDATE_SAMPLES=32`，B1／2／4／8对应累积8／4／2／1。显式 ACCUMULATION 与该关系冲突会在创建输出前拒绝；需要不同有效更新量时明确设置 UPDATE_SAMPLES。正式入口仍执行当前视图准入、首轮训练、重载和固定 test。

本轮保留十事件等量、连续段优先、跨 epoch 少见题优先和 action 软补偿；未改这些采样算法、标签、提示词或原生老师。本轮联合回归与发布核验见 `v60_release_verification_20261015.json`。
