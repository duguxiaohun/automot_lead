# Qwen Simple Action Expert

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
`val/waypoint_ade_m`。这里没有 RS/EVENT/prior 分桶。


训练与验证由 [`action_prior/training_core.py`](../../action_prior/training_core.py) 统一执行，
与主线共用梯度累积、AdamW/EMA、验证选优和恢复流程。
对齐 step 时同时核对 `train/samples_seen`（累计呈现数）和 `train/step_samples`（本次更新样本数）；
默认四卡 × 16 累积为 64 case/完整 step，epoch 尾部可能不足。
FM loss 用于看训练趋势，效果看采样 ADE/FDE 和独立 test。
完整对比条件、续训参数及代码版本限制见[总运行文档](../run.md)。
