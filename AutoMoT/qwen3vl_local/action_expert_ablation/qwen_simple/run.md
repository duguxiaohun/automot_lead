# Qwen Simple Action Expert

2026-09-23：已同步v23的anchor≥4有效历史与Action物理路线支持补齐；需新full map、新run，详见 [共享说明](../run.md)。

当前 action-balanced 按全局六种语义动作等量、背景1/6采样，小事件权重最多2倍，重复上限8；与event同源同参数时预算一致，详见 [共享说明](../run.md)。

## action-balanced 全局动作均衡（2026-09-23）

`--action-balanced` 全局均衡主要动作（含 KEEP），小事件温和加权，不再要求事件等量；普通背景 UNCOND 保留1/6。
与 `--event-balanced` 二选一，不自动开启 token；完整预算/重复上限/恢复说明见 [共用说明](../run.md)。

```bash
# AutoMoT/ 下；单图 + token + 全局动作均衡，bev_only 的 BEV 仍为单帧
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --action-balanced --high-level-action-token --rgb-frame-count 1
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --action-balanced --high-level-action-token --rgb-frame-count 1
```

## 动作 token 防相似坍塌（2026-09-22）

新训练开启 `--high-level-action-token` 即默认启用弱分离正则（weight=0.01，cosine margin=0.5），
七个 token（含 UNCOND）仍可学习，维度与 concat 不变。用 `--action-token-separation-weight 0` 做独立新 run 对照。
完整公式、日志和恢复约束见 [共用说明](../run.md#2026-09-22-动作-token-弱分离正则)。

```bash
# AutoMoT/ 下：单当前图 + token + 默认弱分离；bev_only 本身仍是单帧 BEV
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --high-level-action-token --rgb-frame-count 1
# 无正则对照（新 run）
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --high-level-action-token --action-token-separation-weight 0
```

v21 标定通过共享 candidate/full map 和可选动作 token 接入；保持本消融原有输入定义。代码同步不代表旧生产索引已重建，详见 [v21 同步范围](../run.md)。

## 动作 token 与单图（2026-09-21）

默认仍为四图、动作 token 关闭。KEEP 统一一类，普通 RE/不可用动作输入独立 UNCOND；七类 embedding 与 BEV 等宽，在 BEV token 后追加一个 token。

```bash
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced --high-level-action-token
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced --high-level-action-token --rgb-frame-count 1
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced --high-level-action-token --rgb-frame-count 1
# 关闭 token、恢复四图，另开基线 run
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced --no-high-level-action-token --rgb-frame-count 4
```

也支持 `HIGH_LEVEL_ACTION_TOKEN=1/0`、`RGB_FRAME_COUNT=1/4`，CLI 优先。单图只取当前 anchor 的完整三视角拼接图，Qwen 提示词明确只有当前图。来源为 Phase3 离线标注，图数/动作条件绑定恢复合同，需分别新训；详细公平对照、来源/审计边界见 [共用运行说明](../run.md)。


优化细节已统一为默认值，无需追加新开关。每个 epoch 训练结束及完整验证后，自动更新当前 run 的 **`training_audit.zip`**；训练被中途终止时可直接带走此文件审计。包内有进度、各轮训练/验证指标、loss/LR/更新幅度和实际配置，详见 [默认训练与中途审计](../../action_prior/OPTIMIZATION.md)。

新训练默认 Muon＋辅助 AdamW、默认7轮，首轮5%更新warmup＋1/2/4轮cosine（warmup占用首周期），与主线及 bev_only 共用代码。开启/基线对照与恢复说明见 [共享优化说明](../../action_prior/OPTIMIZATION.md)。

这个消融使用 4 张 LEAD stitched RGB 和 LeadMoT 原本的简短导航 prompt 跑 base
Qwen prefill，然后把 Qwen KV、frozen BEV、speed/target/final goal 送入与
`action_prior` 相同的联合轨迹 Flow Matching decoder。

```bash
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh
bash qwen3vl_local/action_expert_ablation/qwen_simple/smoke.sh
bash qwen3vl_local/action_expert_ablation/qwen_simple/eval.sh \
  --checkpoint checkpoints/action_expert_ablation/qwen_simple/latest/best.pt
```

TB 看 `train/loss`、`train/route_fm_mse`、`train/waypoint_fm_mse`、`val/route_ade_m`、
`val/waypoint_ade_m`。event/action 两种模式均追加共享 full-map 事件桶，不产生 prior/复核分桶。

训练与验证由 [`action_prior/training_core.py`](../../action_prior/training_core.py) 统一执行，
与主线共用梯度累积、Muon/AdamW、EMA、验证选优和恢复流程。
对齐 step 时同时核对 `train/samples_seen`（累计呈现数）和 `train/step_samples`（本次更新样本数）；
默认四卡 × 16 累积为 64 case/完整 step，epoch 尾部可能不足。
FM loss 用于看训练趋势，效果看采样 ADE/FDE 和独立 test。
完整对比条件、续训参数及代码版本限制见[总运行文档](../run.md)。

## 事件均衡开关

```bash
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced
# 默认即事件均衡；切换动作均衡：--action-balanced；不再提供随机采样。
```

自动准备、UE1–UE7/RE2/RE3/RE5 各 1 份与确认常规背景 2 份、重复上限、DDP 采样和事件桶评测
均直接引用 `action_prior`。不加 `--dataset-priors`，事件标签只用于采样与指标。
先验文本开关 `--event-balanced-scene-priors` 不适用于本消融。
共享实现位置、比例/预算调整、显式索引、续训与搬迁 demo 见[总文档](../run.md#与主线完全共用-event-balanced)。
