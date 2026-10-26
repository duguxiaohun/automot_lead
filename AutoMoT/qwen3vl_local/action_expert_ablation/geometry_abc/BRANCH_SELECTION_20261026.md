# 单分支 ABC 训练、测试与交包

统一入口增加 `--branch bev_only|qwen_simple|both`。新运行省略参数仍为六组；指定 `bev_only` 时仅 a_bev_only、b_bev_only、c_bev_only，三组探针全部通过后依次训练、测试、审计并自动交包。指定 `qwen_simple` 同理。单组七轮、12719 更新、813792 呈现及原标定/提示词/数据划分/采样不变。

分组范围写入 admission.json 的 selected_groups，探针绑定 admission SHA。worker 在加载 runner 或初始化 GPU 前拒绝范围外分组，训练只要求当前范围的全部探针。正式 audit/ZIP 也只验收并列出选定组，不把省略组当作完成。旧无 selected_groups 的回执明确视为原六组。已有 --run-dir 不能换分支范围；须新建 output。现有任务继续使用启动时冻结快照，不热更新。

共享数据、资源、processor 准入保持现状，命令仍提供 Qwen3.5 基座目录。BEV 单分支不运行 Qwen 探针或训练，但共享 processor 检查在没有兼容回执时仍可能执行；--recover-from 可在精确代码兼容、模型与数据一致且原始来源复核后复用该回执。不会加载旧 Phase3 LoRA 或旧 Action decoder。此次组选择不改变观测或 packing，兼容表仅扩展这四份已审查调度源码。

服务器在 automot 环境执行：

```bash
cd /mnt/workspace/cee3sgh/AutoMoT && \
bash ./sync_automot.sh && \
ABC_ACTION_MAPS=$(python -c 'from pathlib import Path; from qwen3vl_local.audit_joint.phase3_maps import resolve; print(resolve(Path.cwd(), Path("checkpoints/abc_action_maps_20261025"))[0])') && \
bash qwen3vl_local/action_expert_ablation/geometry_abc/run.sh all \
  --branch bev_only \
  --recover-from checkpoints/action_expert_ablation/geometry_abc_checkpoint_fixed/run_20261025_203923 \
  --maps "$ABC_ACTION_MAPS" \
  --data-root lead_data \
  --data-dir checkpoints/abc_trajectory_v1 \
  --model-dir checkpoints/Qwen3.5-4B \
  --lead-bev-ckpt checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth \
  --output checkpoints/action_expert_ablation/geometry_abc_bev_only
```

原仅探针失败目录保留，新建独立实验；不是从旧探针或失败训练权重续训。如果已有另一套正式任务在运行，不同时启动这条四卡命令。

本地验证覆盖三种选择的流程、默认六组保留、已有运行范围变更拒绝、范围外 worker 提前拒绝，以及真实 audit.build 指标重算和 ZIP 写入的三组/六组 fixture 集成。模型推理、权重和来源证明在该打包测试中明确使用替身；不称真实 GPU 验收。169 份数据构建来源、核心数值训练及采样保持原字节，不重建全池。测试与源码证据在 checkpoints/action_branch_selection_20261026。
