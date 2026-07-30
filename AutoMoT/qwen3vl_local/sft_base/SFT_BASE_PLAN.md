# SFT Base Plan

`sft_base` 是 `sft_v5` 的直接监督对照组，用来测试“不做 OPSD、不写 CoT、不跑 teacher，只让 Qwen3-VL 直接输出固定语义 token”能否训好 RS/EVENT。

## 核心差异

- 复用 `sft_v5` 的数据构建、异常 route 剔除、4 帧 stitched RGB history、`EGO_TO_GOAL_XY`、RS/EVENT 候选池与随机候选顺序。
- Q2 候选集合和展示顺序沿用 `sft_v5` 的 seed namespace，但本路线**已完全没有 A/B/C 选项字母**：候选在数据里就是有序 list，学生只输出固定 EVENT token，避免学习位置捷径。
- Q1 仍判 `RS` 与 `ABNORMAL`，Q2 仍在串行上下文中判 `EVENT`。
- 训练目标仍是普通 teacher-forced CE，不让 student 先 rollout，也不使用 privileged teacher logits。
- 训练时 route 内的离散 memory 仍按 GT answer 推进，但进入 prompt 前会做 anti-copy curriculum：首帧默认 `UNKNOWN`，非首帧按概率把 `BELIEVED_RS/BELIEVED_EVENT` 置错或置为 `UNKNOWN`。这样 memory 是不可靠先验，不再是可直接抄的答案。`EGO_TO_GOAL_XY` 每帧刷新为当前帧 ego-frame goal；eval 仍按学生自己的 Q1/Q2 输出维护离散 memory。
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
- 针对 checkpoint-600 暴露的 `ue_acc=0` 和 memory-copy shortcut，训练默认同时使用首帧 UNKNOWN、memory wrong/UNKNOWN 扰动、转折邻域重复、Q1 ABNORMAL=YES 加权、Q2 UE 加权和 RE 降权。RE loss 不置零，避免模型从全 REGULAR 坍缩到全 UE。
- `eval.py` 支持 `--image-ablation black/random`，用于黑图/随机图诊断；如果消融后 `rs_visual_gain_over_first_gt_lock` 或 `event_visual_gain_over_regular_baseline` 基本不变，说明模型主要依赖 memory/语言捷径而不是视觉。
- 评估额外返回零视觉基线、净视觉增益、预测变化率、锁死 case 比例、ABNORMAL YES 率、UE 输出率、`*_hit_offset_avg`、`*_abs_hit_offset_avg`、`*_max_early_lead` 和 `*_max_late_lag`，避免 `re_acc` 或表面 transition hit rate 继续误导。

## 实现约束

- `labels.py` 中 `DATASET_VERSION` 标识 sft_base 自己的训练协议；`CHOICE_ORDER_DATASET_VERSION` 固定为 v5 的 namespace，用来保证 Q2 候选顺序扰动与 v5 逐样本对齐。`build_dataset.py --choice-seed` 的默认值必须与 `sft_v5/build_dataset.py --option-seed` 相同，否则相位会错开。
- 当前 `DATASET_VERSION=sft_base_rs_event_token_choice`，adapter config 的 `route=sft_base_token_choice`；旧的 A/B/C adapter 已完全废弃，会被 eval adapter config 校验拒绝。不要通过手动改 config 混跑旧权重。
- 数据 schema 里**没有** `rs_option` / `event_option_map`：Q2 候选是 `event_candidates_ordered`（有序 list，顺序即展示顺序）。`RouteSequenceDataset` 读到缺该字段的旧 index 会直接报错要求重建，不做兼容降级——静默兼容会让候选顺序或集合悄悄变化，指标看起来正常但训练分布已经错了。
- `train.py` 默认 `--first-frame-memory-unknown --memory-rs-wrong-prob 0.15 --memory-rs-unknown-prob 0.10 --memory-event-wrong-prob 0.20 --memory-event-unknown-prob 0.10 --transition-frame-repeat 12 --transition-frame-window 1 --ue-event-loss-weight 4.0 --re-event-loss-weight 0.5 --ue-frame-repeat 2 --abnormal-yes-loss-weight 4.0`；若 UE 仍然学不动，优先调高转折邻域重复和 memory 扰动，不建议把 RE loss 直接设为 0。
- `prompts.py` 中 memory 的 RS/EVENT 可以跨帧延续，但 `EGO_TO_GOAL_XY` 必须在每帧 prompt 前刷新，不能沿用首帧坐标。
- `train.py` 的 DDP 累积使用手动 `_sync_trainable_grads()`：每帧 forward/backward 都放在 `no_sync()` 内，本 micro-batch 末尾对所有 trainable LoRA 参数补零并 all-reduce 一次。
- `train.py` 里的 memory 主轨迹仍是 teacher-forced，但 prompt memory 已加入 anti-copy curriculum；如果要测 v5 on-policy student memory 分布，应回到 `sft_v5` 或另开路线，不在 base 内混用。
