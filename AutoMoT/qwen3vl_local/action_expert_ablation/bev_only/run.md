# BEV-Only Action Expert

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
与主线共用梯度累积、AdamW/EMA、验证选优和恢复流程。
对齐 step 时同时核对 `train/samples_seen`（累计呈现数）和 `train/step_samples`（本次更新样本数）；
默认四卡 × 16 累积为 64 case/完整 step，epoch 尾部可能不足。
FM loss 用于看训练趋势，效果看采样 ADE/FDE 和独立 test。
完整对比条件、续训参数及代码版本限制见[总运行文档](../run.md)。
