# SFT Base Plan

`sft_base` 是 `sft_v5` 的直接监督对照组，用来测试“不做 OPSD、不写 CoT、不跑 teacher，只让 Qwen3-VL 直接输出固定语义 token”能否训好 RS/EVENT。

## 核心差异

- 复用 `sft_v5` 的数据构建、异常 route 剔除、4 帧 stitched RGB history、`EGO_TO_GOAL_XY`、RS/EVENT 候选池与随机候选顺序。
- Q2 候选集合和展示顺序沿用 `sft_v5` 的 seed namespace，但本路线**已完全没有 A/B/C 选项字母**：候选在数据里就是有序 list，学生只输出固定 EVENT token，避免学习位置捷径。
- Q1 仍判 `RS` 与 `ABNORMAL`，Q2 仍在串行上下文中判 `EVENT`。
- 训练目标是普通 teacher-forced CE，不让 student 先 rollout，也不使用 privileged teacher logits。
- 训练时 memory 由 GT answer teacher-forced 更新，用来提供干净直接监督；它不是 v5 的 on-policy student 自维护 memory 分布。`EGO_TO_GOAL_XY` 每帧刷新为当前帧 ego-frame goal；eval 仍按学生自己的 Q1/Q2 输出维护离散 memory。
- Prompt 禁止 CoT；协议只输出固定语义 token：

```text
RS: SIGNAL_INTERSECTION
ABNORMAL: NO
```

和：

```text
EVENT: RULE_VIOLATION
```

RS/EVENT prompt 中的候选展示顺序可以稳定打乱；训练和评估只用 token 字典还原标签，代码里已经不存在任何选项字母。Q2 输出的 token 必须出现在本帧候选里，否则按非法处理：不更新 memory、不计正确。

固定 token 字典：

| 内部标签 | 输出 token |
|---|---|
| `R1` | `ORDINARY_ROAD` |
| `R2` | `BIDIRECTIONAL_NARROW` |
| `R3` | `HIGHWAY_MERGE_EXIT` |
| `R4` | `SIGNAL_INTERSECTION` |
| `R5` | `PRIORITY_INTERSECTION` |
| `RE` | `REGULAR` |
| `U-E1` | `LEAD_BRAKE` |
| `U-E2` | `STATIC_OBSTACLE` |
| `U-E3` | `MOVING_CUT_IN` |
| `U-E4` | `VULNERABLE_CROSSING` |
| `U-E5` | `ONCOMING_INVASION` |
| `U-E6` | `RULE_VIOLATION` |
| `U-E7` | `RULE_UNCERTAIN` |
| `U-E8` | `BLOCKED_SPACE` |

## LoRA

默认只保存 adapter delta，base Qwen checkpoint 只读。和 v2/v5 一样支持 `off/merger/last4/all` 四档视觉 LoRA；本路线默认 `LORA_VISION_SCOPE=merger`，也就是默认微调视觉桥接部分。视觉 LoRA 默认启用 fuse guard，连续异常时保存 `fuse_stop_step_<N>/` 并跳过正常 `final/`。

eval 加载 adapter 前必须读 `sft_base_adapter_config.json`，校验 route、dataset version、base model path 与 vision scope，避免误拿 v2/v5/base adapter 产出无意义指标。

## 评测口径

- `eval.py` 固定分三类测试：`full_route` 随机完整路径闭环测试、`rs_transition` RS 转折专项测试、`event_transition` UE/RE/EVENT 转换专项测试。
- 评测阶段完全不做脚本纠正：Q1 RS 错只跳过当前帧 Q2，下一帧继续沿用学生 memory；Q2 EVENT 非法只不更新 EVENT，也不能 reset。`script_resets` 只是审计字段，正常必须恒为 0。
- 转折专项不要求预测和数据标注逐帧完全同拍；用 `--transition-tolerance` 设置容忍窗口，只要模型在转折点前后若干帧内切到目标 RS 或 EVENT，就算该 case 命中，并记录 early/on_time/late。
- 起始 memory 噪声只允许注入在每条完整 route 或每个转折窗口的第一帧，用来测模型是否能自恢复；后续帧仍然不能纠正。
- 针对 checkpoint-600 暴露的 `ue_acc=0`，训练默认提高 UE 的 Q2 EVENT token 权重，并对 UE 帧做轻量重复采样；这只强化异常 EVENT 监督，不改 Q1 RS/ABNORMAL 的损失定义，也不改变 route 内 teacher-forced memory 推进。
- 评估额外返回 `ue_q1_abnormal_acc`、`ue_pred_regular_rate`、`*_hit_offset_avg`、`*_abs_hit_offset_avg`、`*_max_early_lead` 和 `*_max_late_lag`，用于拆分 UE 是卡在 Q1 还是 Q2，以及 RS early 是合理提前还是过早抢跑。

## 实现约束

- `labels.py` 中 `DATASET_VERSION` 标识 sft_base 自己的训练协议；`CHOICE_ORDER_DATASET_VERSION` 固定为 v5 的 namespace，用来保证 Q2 候选顺序扰动与 v5 逐样本对齐。`build_dataset.py --choice-seed` 的默认值必须与 `sft_v5/build_dataset.py --option-seed` 相同，否则相位会错开。
- 当前 `DATASET_VERSION=sft_base_rs_event_token_choice`，adapter config 的 `route=sft_base_token_choice`；旧的 A/B/C adapter 已完全废弃，会被 eval adapter config 校验拒绝。不要通过手动改 config 混跑旧权重。
- 数据 schema 里**没有** `rs_option` / `event_option_map`：Q2 候选是 `event_candidates_ordered`（有序 list，顺序即展示顺序）。`RouteSequenceDataset` 读到缺该字段的旧 index 会直接报错要求重建，不做兼容降级——静默兼容会让候选顺序或集合悄悄变化，指标看起来正常但训练分布已经错了。
- `train.py` 默认 `--ue-event-loss-weight 3.0 --re-event-loss-weight 1.0 --ue-frame-repeat 2`；若 UE 仍然学不动，可继续提高 UE 权重或重复次数，但要同步观察 `train/q2_ue_rate_last_batch` 和训练时长。
- `prompts.py` 中 memory 的 RS/EVENT 可以跨帧延续，但 `EGO_TO_GOAL_XY` 必须在每帧 prompt 前刷新，不能沿用首帧坐标。
- `train.py` 的 DDP 累积使用手动 `_sync_trainable_grads()`：每帧 forward/backward 都放在 `no_sync()` 内，本 micro-batch 末尾对所有 trainable LoRA 参数补零并 all-reduce 一次。
- `train.py` 里的 memory 更新是 teacher-forced baseline 逻辑；如果要测 v5 on-policy student memory 分布，应回到 `sft_v5` 或另开路线，不在 base 内混用。
