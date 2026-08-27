# New Loop Phase2 融合版 2RGB Endpoints 审计（2026-08-27）

## 1. 审计对象与方法

本次比较三组结果：

- 当前融合模型：`checkpoints/sft_new_loop_phase2_20260825_150826_2rgb_endpoints_audit_bundle`；
- 上一版 direct EVENT：`checkpoints/sft_new_loop_phase2_20260825_133640_audit_bundle`；
- 旧 RS-gated Phase3 prompt v2：`sft_loop_phase3/EVAL_PROMPT_V2_V3_20260821.md` 中的
  `20260821_105049_audit_bundle`。

融合模型只看原四帧 history 的第 0、3 帧，不包含 synthetic ROAD_STRUCTURE
user/assistant 或 RS KV 前缀。指标比较后，又逐一读取融合模型 production 的全部 69 个
`error_cases/**/case.json`，并从每例 `history_rgb_paths_used` 打开模型实际使用的 oldest/newest
两张 stitched RGB；随后再打开每例 `history_rgb_paths_all4` 中训练未输入的第 1、2 帧，检查
2RGB 是否遗漏关键中间运动。本轮共检查 69×4=276 张图，其中 138 张是模型实际输入，另
138 张只用于离线归因，不把后者假装成模型可见证据。没有根据 scenario 名猜事件，也没有为
单个 frame/route 建黑名单。只有跨多个场景重复出现、且 RGB 能直接支持的错误模式才进入 prompt。

## 2. 总体指标

| 模型 | production exact | errors | UE1 | UE3 | UE5 | UE6 | RE | INVALID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧 Phase3 prompt v2 / 4RGB + RS prefix | 299/384, 77.86% | 85 | 49 | 51 | 55 | 44 | 51 | 49 |
| 旧 direct EVENT v1 / 4RGB | 298/384, 77.60% | 86 | 49 | 42 | 54 | 55 | 45 | 53 |
| 融合 direct EVENT v2 / 2RGB endpoints | **315/384, 82.03%** | **69** | **52** | **52** | **57** | 54 | 46 | **54** |

融合模型相对旧 direct EVENT 提升 4.43 个百分点，错误减少 17 个；相对旧 Phase3 v2
提升 4.17 个百分点，错误减少 16 个。在新旧 direct EVENT 的 284 个共同 case 上，融合模型
为 240/284，旧模型为 224/284；融合模型独占正确 24 个，旧模型独占正确 8 个。因此提升不是
单纯由 eval case 换样造成，但由于 prompt、RGB mode、数据过滤同时变化，不能把提升单独归因
给 2RGB 或 prompt。

融合模型的 per-question F1 相对旧 direct EVENT：UE1 0.8235→0.8548、UE3
0.7568→0.8189、UE5 0.9000→0.9344、UE6 0.8871→0.8926、INVALID
0.8689→0.8852。UE3 recall 由 0.6562 提高到 0.8125，但 precision 由 0.8936
降到 0.8254，说明 v2 成功追回早期 cut-in，同时带来新的静态/视差 FP。

## 3. 全量错例 RGB 复核

### 3.1 错误转移统计

69 个 production 错例的 GT→预测如下：

| GT→预测 | 数量 | 主要场景 |
|---|---:|---|
| INVALID→RE | 9 | HardBreakRoute 5，其余为雨雾/夜间边界 |
| INVALID→UE1 | 1 | VehicleTurningRoute |
| RE→INVALID | 3 | BlockedIntersection、NonSignalizedJunctionLeftTurn、VehicleTurningRoute |
| RE→UE1 | 3 | 普通跟车/队列或侧车进入视野 |
| RE→UE3 | 8 | Accident/ControlLoss/ConstructionObstacle/StaticCutIn/普通交通 |
| RE→UE5 | 1 | InvadingTurn 低能见度边界 |
| RE→UE6 | 3 | BlockedIntersection 2、DynamicObjectCrossing 1 |
| UE1→RE | 9 | HardBreakRoute |
| UE1→UE3 | 2 | 同一路线密集路边停车/拥堵 |
| UE1 但其它字段错 | 1 | HardBreakRoute |
| UE3→RE | 8 | DynamicObjectCrossing 6、ParkingCutIn 2 |
| UE3→UE1 | 3 | DynamicObjectCrossing |
| UE3→INVALID | 1 | DynamicObjectCrossing |
| UE5→RE | 7 | InvadingTurn |
| UE6→RE | 10 | OppositeVehicleRunningRedLight 9、CrossJunctionDefectTrafficLight 1 |

### 3.2 可以归因给模型、适合进入 prompt 的重复证据

`RE→UE3` 的 8 例以及 `UE1→UE3` 的 2 例形成了唯一清晰且跨场景重复的 prompt 问题：

- `case 050 Accident f123`、`case 084 AccidentTwoWays f34`：事故/拥堵车辆位置杂乱，
  但两个端点没有同一车辆相对车道线持续横向进入；
- `case 128/153 ControlLoss`：相邻车辆随 ego 前进产生明显图像位移，缺少跨车道线证据；
- `case 342 ConstructionObstacle f52`：静态施工车辆和路障被当成动态占用；
- `case 004 DynamicObjectCrossing f67`、`case 063 StaticCutIn f77`：邻车靠近或斜置，
  但 newest frame 未形成可确认的 future-corridor 侵入；
- `case 178/308 HardBreakRoute`：密集路边车辆与同车道 lead 同时出现，模型把静态侧车
  当成 UE3，反而漏掉 GT UE1。

共同视觉原因不是“UE3 的 about-to-occupy 太宽”本身，而是 2RGB 端点下把 ego 前进视差、
新进入视野、静态事故/施工和路边停车误当成他车横向运动。因此 prompt v3 只补充：actor
图像位置/尺度变化不等于 lateral entry；还必须看到其相对车道边界/ego corridor 的关系改变，
或 newest frame 已经侵入。继续保留早期 cut-in，不要求完整入道或多 cue，避免重新造成 v1
的 UE3 recall 崩塌。

### 3.3 不应靠放宽/收紧 prompt 追标签的错例

- **UE1→RE 9 例**：多为夜间、雾雨或远距离 lead；端点中车辆尺度变化很小，部分只有普通
  尾灯，无法从 RGB 稳定证明“突然减速”。放宽到“看到前车/刹车灯即 UE1”会直接伤害 RE。
- **UE3→RE 8 例**：部分 highway/雨雾画面能看到邻车，但端点乃至补看的中间帧仍缺少清晰
  横向跨线轨迹；另有 ParkingCutIn 最新帧只剩停放车辆。现有 `about to occupy` 已足够，
  不再凭标签扩写，也没有证据支持把问题简单归咎于删掉中间两帧。
- **UE5→RE 7 例**：多数 newest frame 看不到仍在侵入的对向 actor，若有车辆也常只出现在
  oldest frame；这是事件 span/newest-state 标签边界，不应删除 v2 的最新帧合同。
- **UE6→RE 10 例**：四帧 RGB 常能看到横穿车辆，部分 actor 在中间帧最清楚，但信号状态、
  路权或违规 cue 仍不可见；同时三个
  `RE→UE6` 例证明“路口有横穿车辆即 UE6”会误报 BlockedIntersection/普通 crossing。
  因而保留“冲突 actor + 可见违规/优先权 cue”双条件。
- **INVALID 双向错误**：五个 HardBreakRoute INVALID→RE 来自同一路线接近乡村路口的边界帧；
  三个 RE→INVALID 中 newest frame 又常像连续道路。它们更像 RS/RGB 时序或道路域可见性边界，
  不为单一路线改 invalid 定义。

## 4. Audit prompt 的独立格式问题

融合模型 audit prompt 的严格 exact 为 263/384=68.49%，format-valid 为
327/384=85.16%。但只解析开头有序 YES/NO 行时，316/384=82.29% 的事件答案正确。
57 个严格格式失败全部存在空 `EVIDENCE_*:` 行，其中 53 个 case 的 YES/NO 原本完全正确。

这说明 production EVENT 判断没有在 audit 下整体崩掉，退化主要是未训练过的 evidence schema
指令遵循。正式 strict parser 保持不变，不接受空 evidence；prompt v3 明确要求每一题（包括
NO）都写非空可见线索。eval 新增非评分 `answer_only_diagnostics`，将事件答案正确率与完整
answer+evidence 合同分开报告，避免继续把两类问题混成一个数字。

## 5. 本轮落地与明确未做事项

已落地：

1. `prompts.py` 升级为 `sft_new_loop_phase2_direct_event_visual_v3`，只新增经过上述 RGB
   重复证据支持的 UE3 静态/视差边界；
2. audit prompt 明确所有 `EVIDENCE_*` 必须非空，NO 也要写可见 absence/boundary；
3. 新增 `parse_event_answer_lines` 和 eval `answer_only_diagnostics`，但不改变 strict 正式评分；
4. 回归测试锁定 v3 RGB 边界、严格 evidence parser 和 answer-only 诊断合同。

明确未做：

- 未按场景名、车辆颜色、route/frame id 写规则；
- 未把 UE3 收紧成“必须已经完全入道”；
- 未为 UE5/UE6 的不可观察标签放宽视觉证据；
- 未放宽 strict parser 接受空 evidence；
- 未声称 v3 已提升指标。prompt hash 已变化，必须重新训练并用固定 384-case 或固定 case-id
  manifest 复测后，才能确认修改效果。

## 6. 下一轮验收重点

1. production exact 不低于 v2 的 315/384；
2. UE3 recall 尽量保持 0.8125，precision 高于 0.8254；
3. `RE→UE3` 与 `UE1→UE3` 合计从 10 例下降，同时不增加 `UE3→RE`；
4. UE5、UE6、INVALID 不因本轮小改发生显著回退；
5. audit strict format-valid 高于 0.8516，且同时报告 strict exact 与 answer-only exact；
6. 最好固定同一 384-case manifest，另做 v2/v3 同 RGB mode 对照，避免数据换样混淆。
