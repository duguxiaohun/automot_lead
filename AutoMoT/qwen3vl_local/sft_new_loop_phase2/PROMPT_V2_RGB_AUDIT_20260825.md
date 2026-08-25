# New Loop Phase2 Prompt v2 RGB Audit（2026-08-25）

## 审计输入

- 新模型：`checkpoints/sft_new_loop_phase2_20260825_133640_audit_bundle`
- 旧 Phase3 对照：`checkpoints/sft_loop_ohase3_eval/20260821_105049_audit_bundle`
- 旧 Phase3 prompt v3 对照：`checkpoints/sft_loop_ohase3_eval/20260821_100309_audit_bundle`
- 标签定义与既有逐帧结论：`keyframe_filter/ROAD_EVENT_CLASSIFICATION_PLAN.md`、
  `keyframe_filter/ROAD_EVENT_RGB_AUDIT_ARCHIVE_202607.md`、
  `qwen3vl_local/sft_loop_phase3/EVAL_PROMPT_V2_V3_20260821.md`

本轮没有按场景名凭空扩写规则，也没有根据 test 错例建立 frame/route 黑名单。先比较固定均衡
test 指标，再逐帧查看 bundle 中实际喂给模型的 oldest→newest RGB，最后只提炼可跨场景复用的
视觉判据。

## 指标定位

固定 384 条均衡测试上，新模型 production exact 为 `298/384 = 77.60%`；旧 Phase3
prompt v2 为 `299/384 = 77.86%`，基本持平但没有整体提升；旧 prompt v3 为
`244/384 = 63.54%`。相对旧 prompt v2，新模型 UE6 从 `44/64` 提升到 `55/64`，invalid
从 `49/64` 提升到 `53/64`；UE3 从 `51/64` 降到 `42/64`，RE 从 `51/64` 降到
`45/64`，UE1 持平 `49/64`，UE5 从 `55/64` 微降到 `54/64`。

新模型 production 与 audit-prompt exact 都是 `77.60%`，说明直接 EVENT 合同对“是否要求
evidence”较稳定；但训练时 best generation 只用 96 条 val case 选出，step 4000 为 `79.17%`，
step 10000 已回落到 `70.83%`，因此仍需保留频繁 generation eval 和 best checkpoint，而不能
默认 final 最好。

## 逐帧错例证据与规则落点

1. `UE3 FN / ParkingCutIn / f23`：可见停车侧车辆的斜置姿态，但跨帧侧向进入证据弱。
   Prompt v2 因此保留“即将占用”正例，并允许跨帧侧移、压线、车头/车身已经侵入或逐步进入
   中的任一证据，不要求完整入道或多个 cue；仅在路径外的斜置姿态不能单独判 UE3。
2. `UE1 FP / DynamicObjectCrossing / f122`：同屏有前车刹车灯与侧向横穿车，UE1/UE3 证据
   容易叠加。v2 明确同一主交互按运动类型分界：同车道纵向突然减速为 UE1，侧向进入为 UE3，
   单帧接近或同一刹车灯不能同时支撑两者。
3. `UE3 FP / StaticCutIn / f76`：红车在 RGB 中确有明显横向接近，RE 标签本身存在边界疑问。
   因此不把该帧反向写成“红车/StaticCutIn 必为 NO”，而是在 audit note 中要求记录
   `MODEL / LABEL_OR_BOUNDARY / BOTH` 责任。
4. `UE5 FN / InvadingTurn / f43、f91`：最后一帧主要是空路、锥桶或雾，未见仍在侵入的
   对向车辆。v2 要求 newest moment 仍有对向侵入实体或其即时避让影响；只在旧帧出现、交互
   已结束或空锥桶通道不足以支撑 UE5。
5. `UE6 FN / OppositeVehicleRunningRedLight / f33` 以及同类 f3/f7/f11：可见车辆横穿或已驶离，
   但信号/路权违规证据不可见。v2 要求“冲突占用 + 可见违规/优先权 cue”同时成立；刚驶离但
   ego 仍明显避让可保持 YES，不能仅由路口里有车或车辆类型推断闯灯。
6. `UE1 FN / HardBreakRoute / f24` 与 `invalid FN / HardBreakRoute / f141`：夜间画面弱，前者
   看不出突然减速，后者仍可看出连续道路。v2 将低能见度拆成两步：UE 证据弱时保守 NO，
   但道路问题域仍相容时不得自动改成 invalid。
7. `invalid FP / CrossingBicycleFlow / f2`：标签的 true RS 是局部路口，但 newest RGB 更像连续
   多车道路段。该例提示 invalid 误差中可能含 RS/RGB 时序边界问题；本轮不后验改标签，只在
   错例模板中新增“question-domain visually applicable”人工字段。

## 落地改动

- `prompts.py` 升级为 `sft_new_loop_phase2_direct_event_visual_v2`，加入 newest-state、可观察性、
  UE1/UE3/UE5 互斥和 UE6 可见违规合同；prompt hash 随之变化，旧 adapter 必须重训。
- `visual_audit.py` 不再只读取 RS review reason；显式 EVENT review request 也进入
  `visual_label_risk`。它只过滤默认 clean pool，不改写原始 GT。
- `audit_eval_cases.py` 的每例 note 新增标签可见性、道路域适用性和模型/标签责任检查项。
- `eval.sh` 默认执行全量 frame-risk 扫描；`run_full_pipeline.sh` 默认调用 `eval.sh`，最终生成
  硬上限 30MB 的审计包。

这些改动是下一次训练的实验假设，不把 prompt v2 的提升当成既成事实。重训后应继续用相同
384 条均衡口径与旧 Phase3 对比，重点看 UE3、RE 是否回升，同时确认 UE6/invalid 不回退。
