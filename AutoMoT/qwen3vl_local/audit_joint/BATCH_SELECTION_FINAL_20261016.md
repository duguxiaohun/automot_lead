# Phase3／Phase4 两／四图最终运行配置

用户已明确接受Phase4两图和四图每卡B2，不再补充B4、正式32/update短对照或新的容量实验。本次batch选型关闭；路线校验、学生视图迁移、整轮训练及模型质量评估是其它任务，不再作为batch选型的前置条件。

| 阶段 | RGB | 每卡batch | 卡数 | 一次前向全局片段 | 梯度累积 | 每次更新样本 | 输入帧／锚点步长 |
|---|---|---:|---:|---:|---:|---:|---|
| Phase3 | 2 | 2 | 4 | 8 | 2 | 16 | [t-2,t]／2 |
| Phase3 | 4 | 2 | 4 | 8 | 2 | 16 | [t-3,t-2,t-1,t]／1 |
| Phase4 | 2 | 2 | 4 | 8 | 4 | 32 | [t-2,t]／2 |
| Phase4 | 4 | 2 | 4 | 8 | 4 | 32 | [t-3,t-2,t-1,t]／1 |

均启用gradient checkpointing，沿既有LoRA16/32、vision off配置。每卡两个独立片段经过一次batch forward；梯度累积另计。不要把B2解释为每卡两个batch或把累积4解释为一次前向4题。

选择依据：服务器Phase3四H20两次重复B2较B1吞吐+9.53%/+8.60%，最大reserved29.47/40.55GiB；Phase4四H20-3e小池单次B2较B1+7.89%/+6.20%，reserved29.77/41.38GiB，均满足当次余量。Phase4小池比较为8/update，正式32/update是用户接受并选定的运行配置，不伪称已经完成对应新实测或证明全局最优。

## 已落实的训练和采样要求

- Phase3和Phase4当前rgb_short训练器均真实组batch，一次前向多题。
- 训练、val/test及Action共享评估默认十事件等量；Phase4自动参考评估使用独立references4，原生data4本身可以只有train。
- 优先保持同event内的连续锚点；短段拼接独立样本，不跨路线伪造一题的历史。
- transition之间允许合法历史重叠，仍保留各题身份和split隔离。
- 曝光次数／未见优先、连续段优先、同锚点选题及action软补偿保留；不承诺每个batch内action严格均匀。
- 跨epoch维护曝光记录，优先尚未见过的片段。Phase3是同一次训练的多epoch延续；Phase4支持绑定历史的epoch边界恢复，不能据此声称Phase3已有优化器断点恢复。

## 正式入口

Phase3继续使用 `audit_joint/train_phase3_batch.sh`，已固定B2／累积2。现有命令见 [Phase3训练配置](BATCH_TRAINING_20261015.md)。

Phase4的 `audit_joint.phase4_first_epoch` 函数与CLI默认均改为B2，实际把B2／累积4／32update传入两模式准入和训练，配置进入workflow及交接回执。旧shell环境中的BATCH_SIZE不会覆盖该编排计算出的配置；显式CLI覆盖仍可用于将来的其它实验。

新增 `audit_joint/train_phase4_batch.sh` 是本次选定配置的固定入口。要求已有有效学生view，拒绝冲突的batch／累积／update／checkpointing环境值，按两图→四图执行既有首轮、重载、固定test流程；不会先跑benchmark、生成view或迁移旧view。显式GPU_IDS沿用用户指定四卡，未指定时由原入口选择可见空闲四卡。输出新建，既有目录拒绝覆盖。

以下是用户决定开始正式训练时的入口，不是要求现在再跑一次验收：

```bash
export PREPARED=checkpoints/phase4_v57_20261014_232157_762998141
export DATASET=/实际已有效的学生view目录
export DATA_ROOT=lead_data MODEL_DIR=checkpoints/Qwen3.5-4B
export GPU_IDS=0,1,2,3
export BATCH_SIZE=2 ACCUMULATION=4 UPDATE_SAMPLES=32 GRADIENT_CHECKPOINTING=1
export OUTPUT_DIR="checkpoints/p4_batch2_$(date +%Y%m%d_%H%M%S_%N)"
bash qwen3vl_local/audit_joint/train_phase4_batch.sh
```

DATASET必须替换为真实有效目录；没有收到服务器迁移成功结果，不编造一个已存在的新view。入口保留原生/学生合同、独立references4、正式准入检查，不能把此前native_validated等同训练准入。运行正式训练的耗时不属于短benchmark预算。

底层通用 `rgb_short/train.py`、`run.sh`、`run_automatic.sh` 的历史默认值保持兼容；本次选定配置由上述正式编排或固定入口明确传入，不应绕开它们仅凭底层历史默认启动。没有修改这些合同绑定源码，也没有修改phase4_load/production_cache，因此本次配置发布不会额外使学生view或v69路线缓存失效；原本过期的view仍需通过原有合同检查。

本次仅本地CPU接线与采样回归，不执行新的服务器GPU测试、全库校验、视图迁移或正式训练。已有诊断和benchmark工具保留为手动工具，不再作为本轮待办。
