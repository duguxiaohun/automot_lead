# SFT Base Plan

`sft_base` 是 `sft_v5` 的直接监督对照组，用来测试“不做 OPSD、不写 CoT、不跑 teacher，只让 Qwen3-VL 直接输出固定语义 token”能否训好 RS/EVENT。

## 核心差异

- 复用 `sft_v5` 的数据构建、异常 route 剔除、4 帧 stitched RGB history、`EGO_TO_GOAL_XY`、RS/EVENT 候选池与随机候选顺序。
- Q2 候选集合和展示顺序沿用 `sft_v5` 的 seed namespace，但本路线**已完全没有 A/B/C 选项字母**：候选在数据里就是有序 list，学生只输出固定 EVENT token，避免学习位置捷径。
- Q1 只判 `RS`，Q2 在串行上下文中判 `EVENT`；`ABNORMAL` 已删除，UE/RE 只从 Q2 EVENT 折叠。
- 训练目标仍是普通 teacher-forced CE，不让 student 先 rollout，也不使用 privileged teacher logits。
- 训练时 route 内的离散 memory 仍按 GT answer 推进，但进入 prompt 前会做 anti-copy curriculum：首帧默认 `UNKNOWN`，非首帧按概率把 `BELIEVED_RS/BELIEVED_EVENT` 置错或置为 `UNKNOWN`。验证 loss 使用同一套 prompt-memory 扰动，seed 固定且不随 rank 漂移，避免 clean memory 低估 val loss。Q1 prompt 只显示 `BELIEVED_RS` 和 `EGO_TO_GOAL_XY`，不显示 `BELIEVED_EVENT`，避免 EVENT 反向泄漏 RS；Q1 更新 RS 后，Q2 会在当前 RS 语境下重新采样 EVENT memory，其中 keep 分支沿用进入本帧前的干净 EVENT memory（上一帧 GT），不是本帧 target，让 Q2 的 EVENT wrong/UNKNOWN/keep 边际恢复且不泄漏答案。`EGO_TO_GOAL_XY` 每帧刷新为当前帧 ego-frame goal；eval 仍按学生自己的 Q1/Q2 输出维护离散 memory。
- Prompt 禁止 CoT；协议只输出固定语义 token：

```text
RS: SIGNAL_INTERSECTION
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

默认只保存 adapter delta，base Qwen checkpoint 只读。和 v2/v5 一样支持 `off/merger/last4/all` 四档视觉 LoRA；本路线默认 `LORA_VISION_SCOPE=last4`、`LORA_RANK=32`、`LORA_ALPHA=64`，保持 alpha/rank scaling=2，用来回应黑图/随机图消融暴露的视觉零增益。视觉 LoRA 默认启用 fuse guard，连续异常时保存 `fuse_stop_step_<N>/` 并跳过正常 `final/`。

eval 加载 adapter 前必须读 `sft_base_adapter_config.json`，校验 route、dataset version、base model path 与 vision scope，避免误拿 v2/v5/base adapter 产出无意义指标。

## 评测口径

- `eval.py` 固定分三类测试：`full_route` 随机完整路径闭环测试、`rs_transition` RS 转折专项测试、`event_transition` UE/RE/EVENT 转换专项测试。
- 评测阶段完全不做脚本纠正：Q1 RS 错也继续问 Q2，但 Q2 候选按学生预测/维护的 RS 生成；Q2 EVENT 非法只不更新 EVENT，也不能 reset。`script_resets` 只是审计字段，正常必须恒为 0。
- 转折专项不要求预测和数据标注逐帧完全同拍；用 `--transition-tolerance` 设置容忍窗口，只要模型在转折点前后若干帧内切到目标 RS 或 EVENT，就算该 case 命中，并记录 early/on_time/late。
- 起始 memory 噪声只允许注入在每条完整 route 或每个转折窗口的第一帧，用来测模型是否能自恢复；后续帧仍然不能纠正。
- 针对 checkpoint 消融暴露的视觉零增益、`ue_acc=0` 和 memory-copy shortcut，训练默认同时使用首帧 UNKNOWN、强 memory wrong/UNKNOWN 扰动、memory dropout、转折邻域重复、Q2 UE 加权和 RE 降权。RE loss 不置零，避免模型从全 REGULAR 坍缩到全 UE。
- `eval.py` 支持 `--image-ablation black/random`，用于黑图/随机图诊断；如果消融后 `rs_visual_gain_over_first_pred_lock` 或 `event_visual_gain_over_regular_baseline` 基本不变，说明模型主要依赖 memory/语言捷径而不是视觉。
- 评估额外返回零视觉基线、净视觉增益、预测变化率、锁死 case 比例、UE 输出率、UE-vs-RE TP/FP/FN/TN 与 F1、单/多候选 × RE/UE 四格 EVENT 指标、`ue_fp_on_multi_candidate_re_rate`、`single_candidate_invalid_rate`、dataset candidate mismatch 在全量/UE/RE 中的占比、相邻帧 RS change / RE->UE / UE->RE 的 TP/FP/FN/TN/F1/invalid、GT 稳定帧假转折率、转折命中时窗口左边界已在目标值的比例、RS/EVENT 转折方向混淆矩阵、RS/EVENT 混淆矩阵、GT EVENT 在学生 RS 候选下不可达比例、`*_hit_offset_avg`、`*_abs_hit_offset_avg`、`*_max_early_lead` 和 `*_max_late_lag`。GT EVENT 不在 dataset 自己候选表里的帧与训练侧 Q2 skip 对齐，不进入 EVENT 分母；GT EVENT 在 dataset 候选里但不在学生 RS 候选里，仍按模型错误计分。这样避免 `re_acc`、单候选送分题、数据缺陷或表面 transition hit rate 继续误导。

## 实现约束

- `labels.py` 中 `DATASET_VERSION` 标识 sft_base 自己的训练协议；`CHOICE_ORDER_DATASET_VERSION` 固定为 v5 的 namespace，用来保证 Q2 候选顺序扰动与 v5 逐样本对齐。`build_dataset.py --choice-seed` 的默认值必须与 `sft_v5/build_dataset.py --option-seed` 相同，否则相位会错开。
- 当前 `DATASET_VERSION=sft_base_rs_event_token_choice`，adapter config 的 `route=sft_base_token_choice`；旧的 A/B/C adapter 已完全废弃，会被 eval adapter config 校验拒绝。不要通过手动改 config 混跑旧权重。
- 数据 schema 里**没有** `rs_option` / `event_option_map`：Q2 候选是 `event_candidates_ordered`（有序 list，顺序即展示顺序）。`RouteSequenceDataset` 读到缺该字段的旧 index 会直接报错要求重建，不做兼容降级——静默兼容会让候选顺序或集合悄悄变化，指标看起来正常但训练分布已经错了。
- `train.py` 默认 `--first-frame-memory-unknown --memory-rs-wrong-prob 0.30 --memory-rs-unknown-prob 0.40 --memory-event-wrong-prob 0.35 --memory-event-unknown-prob 0.35 --rs-wrong-event-unknown-prob 0.25 --memory-dropout-prob 0.15 --memory-perturbation-mode route_state --memory-perturb-duration-min 3 --memory-perturb-duration-max 5 --transition-frame-repeat 4 --transition-frame-window 3 --ue-event-loss-weight 4.0 --re-event-loss-weight 1.0 --single-candidate-re-scale 0.1 --ue-frame-repeat 2 --ue-repeat-mode inverse_sqrt --ue-repeat-max 8`；route 首帧固定 UNKNOWN/UNKNOWN 且直接返回；非首帧 dropout 是独立第一层，命中就隐藏 RS/EVENT 先验并跳过其它扰动；默认 route_state 会让 wrong/UNKNOWN/keep 在同一 route 内连续保持 3-5 个真实帧，模拟“错误持续一段时间后纠偏”，`frame` 模式只作旧逐帧独立扰动消融。RS unknown 时 EVENT 仍独立扰动，RS wrong 时 EVENT 在 UNKNOWN 与新 RS 自洽错误候选间分流。Q1 后 RS hypothesis 真正改变时旧 EVENT 立即失效为 UNKNOWN；训练侧随后为 Q2 单独按当前 RS 池重采 EVENT，keep 分支使用上一帧干净 EVENT memory，避免 GT RS 纠正把 Q2 EVENT memory 大量压成 UNKNOWN，也避免在 EVENT 转折帧泄漏本帧答案；GT EVENT 在候选表内的单/多候选 Q2 都参与训练，其中单候选 RE 低权重保温、多候选 RE 作为硬负样本保留 loss、UE 保持高权重。若 UE 仍然学不动，优先看黑图/no-goal 消融和按候选数量分层后的 `event_pred_ue_rate_multi_candidate`，不建议把 RE loss 直接设为 0。
- `eval_candidates.py` 和 `build_dataset.py` 的 Q2 候选都固定来自当前/学生 RS 的静态候选全集；逐帧 `allowed_events` 只用于 GT 解析与审计，不参与 prompt 候选构造，避免候选长度泄漏当帧异常事实。当 eval 算出的候选集合与 dataset `event_candidates_ordered` 相同时，直接复用 dataset 顺序，避免 eval seed 造成 order drift。
- `EVENT_CANDIDATES_BY_RS` 按严格方案 A 维护：复用 build_dataset 的异常/缺失/失败 route 过滤后，只保留 `count >= 20` 且 `rs_frame_rate >= 0.1%` 的 RS×UE 组合；当前 prompt 候选数为 R1=6、R2=4、R3=1、R4=5、R5=5，R3 仍保持纯 RE。
- `audit_rs_event_cooccurrence.py` 用来按 build_dataset 同款异常/缺失 route 过滤后统计 RS x UE 共现，检查静态表缺口、低频拒绝项、spurious 静态项和 R3 是否真是纯 RE；`audit_eval_candidate_drift.py` 用来扫 val index，量化 `pred_rs == gt_rs` 时 eval 静态候选相对训练候选 `event_candidates_ordered` 的 set/order 偏差，并分开输出 set mismatch、scoreable unreachable、dataset candidate mismatch 与 GT static candidate mismatch examples；当前静态协议下 set/order mismatch 与 `unreachable_when_pred_eq_gt_scoreable_rate` 都应接近 0，`--expect-mismatch-combinations` 只断言 GT static mismatch 的组合类型落在已知低频拒绝集合内，不按全量 163 帧硬套 val split。
- `prompts.py` 中 memory 的 RS/EVENT 可以跨帧延续，但 `EGO_TO_GOAL_XY` 必须在每帧 prompt 前刷新，不能沿用首帧坐标。
- `train.py` 的 DDP 累积使用手动 `_sync_trainable_grads()`：每帧 forward/backward 都放在 `no_sync()` 内，本 micro-batch 末尾对所有 trainable LoRA 参数补零并 all-reduce 一次。
- `train.py` 里的 memory 主轨迹仍是 teacher-forced，但 train/val prompt memory 都已加入 anti-copy curriculum；如果要测 v5 on-policy student memory 分布，应回到 `sft_v5` 或另开路线，不在 base 内混用。
