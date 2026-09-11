# Action prior 可选审计

日常训练和 TensorBoard 用法见 [run.md](run.md)。以下命令均从 `AutoMoT/` 执行。

## 常用命令

```bash
# CPU 合同与采样测试，不加载模型或 CARLA。
bash qwen3vl_local/action_prior/test.sh

# 对比本地 Phase1/2 LoRA 的已有验证成绩，不重新训练。
bash qwen3vl_local/action_prior/rank_loras.sh

# 复核先验（真实模型，使用已有 action 索引；不支持 dataset-priors）。
DATA_DIR=checkpoints/action_prior_data/已有索引目录 bash qwen3vl_local/action_prior/audit_priors.sh
GPU_IDS=0 DATA_DIR=checkpoints/action_prior_data/已有索引目录 bash qwen3vl_local/action_prior/audit_priors.sh

# 已有 run 重新打包轻量审计结果；full pipeline 默认自动执行。
python qwen3vl_local/action_prior/audit_bundle.py --root checkpoints/action_prior/latest
```

`audit.zip` 上限 30 MB，保留指标和选取的案例，列出未打包项目；权重、缓存、完整视频及 TB 不在包内。

## 如何解释结果

- `sampling/epoch_*.json`：精确配额、最优/实际唯一帧数、重复直方图和各桶路线数；`repeat_presentations` 是额外重复次数。
- `validation/`：自然分布与各桶 ADE/FDE。`--best-selection-metric event_balanced_ade` 按 1:…:1:2 选 best，要求 val 全桶覆盖；默认 `natural_ade`。参与 best 选优的验证始终完整遍历 val。
- `epoch_audit/`：实际呈现计数。`audit/`：先验接受/无效和摘要 fallback 的少量案例。
- `domain_inapplicable` 是正常域外；判断复核失败看 `unconfirmed`，不能把全部 invalid 都当错误。
- `history/independent/compare` 比较复核方式；compare 仍按 history 接受，一致率不代表准确率。
- 上游候选池重叠只是可能见过的上界，不能当作实际训练命中；来源缺失记 unknown。

## 来源搬迁与条件对照

```bash
# 标签/full map 搬迁：内容必须相同，续训和最终评测使用新路径。
RESUME=checkpoints/action_prior/latest/latest.pt bash qwen3vl_local/action_prior/run_full_pipeline.sh \
  --prior-labels /新路径/prior_labels.jsonl --event-balance-index /新路径/full_event_mapping.jsonl
GPU_IDS=0,1,2,3 RESUME=checkpoints/action_prior/latest/latest.pt bash qwen3vl_local/action_prior/run_full_pipeline.sh \
  --prior-labels /新路径/prior_labels.jsonl --event-balance-index /新路径/full_event_mapping.jsonl

# 同一 checkpoint 的干净先验对照。
bash qwen3vl_local/action_prior/eval.sh --checkpoint checkpoints/action_prior/latest/best.pt --prior-noise 0
GPU_IDS=0 bash qwen3vl_local/action_prior/eval.sh --checkpoint checkpoints/action_prior/latest/best.pt --prior-noise 0
```

显式 `--prior-labels` / `--event-balance-index` 是已有文件覆盖，缺失会报错，不覆盖人工指定文件。
自动均衡候选/full map 位于 `checkpoints/action_prior_prepared/`，按来源、规则和 action split 内容区分，构建成功后才发布；续训不会重新生成改变合同的索引。

base/prior、Qwen-simple/BEV-only 等消融应分别训练，核对 seed、实际 split 和采样预算；不能临时切换同一 decoder 的条件。
特别是 balanced 将 Phase3 开发路线移入 train，和 uniform 比较时须核对实际 holdout。消融用法见 [消融运行说明](../action_expert_ablation/run.md)。

## CARLA / Bench2Drive

```bash
# 仅在已配 CARLA 的训练机运行。dataset-prior 模型需显式切回 LoRA，并披露条件迁移。
ACTION_DATASET_PRIORS=0 bash qwen3vl_local/action_prior/eval.sh --bench2drive --checkpoint checkpoints/action_prior/latest/best.pt
GPU_IDS=0 ACTION_DATASET_PRIORS=0 bash qwen3vl_local/action_prior/eval.sh --bench2drive --checkpoint checkpoints/action_prior/latest/best.pt
```

纯均衡采样模型在闭环不需要 full map；`event-balanced-scene-priors` 模型不支持闭环。
正式 220 路线只用于最终评测，不能参与训练选优。离线 ADE、CPU 测试和合成验证均不代表真实闭环成绩。
