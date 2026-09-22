# BEV-Only Action Expert

## 动作 token 防相似坍塌（2026-09-22）

新训练开启 `--high-level-action-token` 即默认启用弱分离正则（weight=0.01，cosine margin=0.5），
七个 token（含 UNCOND）仍可学习，维度与 concat 不变。用 `--action-token-separation-weight 0` 做独立新 run 对照。
完整公式、日志和恢复约束见 [共用说明](../run.md#2026-09-22-动作-token-弱分离正则)。

```bash
# AutoMoT/ 下：单当前图 + token + 默认弱分离；bev_only 本身仍是单帧 BEV
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --high-level-action-token --rgb-frame-count 1
# 无正则对照（新 run）
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --high-level-action-token --action-token-separation-weight 0
```

v21 标定通过共享 candidate/full map 和可选动作 token 接入；保持本消融原有输入定义。代码同步不代表旧生产索引已重建，详见 [v21 同步范围](../run.md)。

## 动作 token 与单图（2026-09-21）

默认仍为四图、动作 token 关闭。KEEP 统一一类，普通 RE/不可用动作输入独立 UNCOND；七类 embedding 与 BEV 等宽，在 BEV token 后追加一个 token。

```bash
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced --high-level-action-token
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced --high-level-action-token --rgb-frame-count 1
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced --high-level-action-token --rgb-frame-count 1
# 关闭 token、恢复四图，另开基线 run
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced --no-high-level-action-token --rgb-frame-count 4
```

也支持 `HIGH_LEVEL_ACTION_TOKEN=1/0`、`RGB_FRAME_COUNT=1/4`，CLI 优先。本分支没有 Qwen，BEV 原本就是单帧 RGB＋LiDAR，因此图数选项不改变有效模型条件。来源为 Phase3 离线标注，图数/动作条件绑定恢复合同，需分别新训；详细公平对照、来源/审计边界见 [共用运行说明](../run.md)。


优化细节已统一为默认值，无需追加新开关。每个 epoch 训练结束及完整验证后，自动更新当前 run 的 **`training_audit.zip`**；训练被中途终止时可直接带走此文件审计。包内有进度、各轮训练/验证指标、loss/LR/更新幅度和实际配置，详见 [默认训练与中途审计](../../action_prior/OPTIMIZATION.md)。

新训练默认 Muon＋辅助 AdamW、默认7轮，首轮5%更新warmup＋1/2/4轮cosine（warmup占用首周期），与主线及 qwen_simple 共用代码。开启/基线对照与恢复说明见 [共享优化说明](../../action_prior/OPTIMIZATION.md)。

这个消融不初始化 Qwen，不读取图文 KV。每层 Prefix-KV attention 收到长度为 0 的
prefix K/V，因此 decoder 只依赖 frozen LEAD BEV、speed、target point、next target point、
final goal 和 route/waypoint query 自身。注意这里的 LEAD BEV 仍是当前 stitched RGB
与 LiDAR BEV 融合后的特征；本实验测的是移除 Qwen 图文分支，不是纯 LiDAR 或完全无视觉。

```bash
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
bash qwen3vl_local/action_expert_ablation/bev_only/smoke.sh
bash qwen3vl_local/action_expert_ablation/bev_only/eval.sh \
  --checkpoint checkpoints/action_expert_ablation/bev_only/latest/best.pt
```

TB tag 与 `qwen_simple` 相同；由于没有 Qwen prefill，吞吐可用
`train/samples_per_second` 直接观察。

训练与验证由 [`action_prior/training_core.py`](../../action_prior/training_core.py) 统一执行，
与主线共用梯度累积、Muon/AdamW、EMA、验证选优和恢复流程。
对齐 step 时同时核对 `train/samples_seen`（累计呈现数）和 `train/step_samples`（本次更新样本数）；
默认四卡 × 16 累积为 64 case/完整 step，epoch 尾部可能不足。
FM loss 用于看训练趋势，效果看采样 ADE/FDE 和独立 test。
完整对比条件、续训参数及代码版本限制见[总运行文档](../run.md)。

## 事件均衡开关

```bash
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced
# 关闭：--no-event-balanced；环境变量写法：EVENT_BALANCED=1。
```

自动准备、UE1–UE7/RE2/RE3/RE5 各 1 份与确认常规背景 2 份、重复上限、DDP 采样和事件桶评测
均直接引用 `action_prior`。不加 `--dataset-priors`，事件标签只用于采样与指标。
先验文本开关 `--event-balanced-scene-priors` 不适用于本消融。
共享实现位置、比例/预算调整、显式索引、续训与搬迁 demo 见[总文档](../run.md#与主线完全共用-event-balanced)。
