# ROAD_STRUCTURE Full RGB Audit 2026-07-06

本轮按 `rgb_blind_rs_event_audit.py` 全量扫描 LEAD RGB，并在同一批 route 上跑当前
`collector.py` 帧级 ROAD_STRUCTURE 规则。

- 输入：`AutoMoT/lead_data`
- 输出目录：`/tmp/automot_all_scenarios_rgb_blind_audit_20260706`
- 汇总目录：`/tmp/automot_all_scenarios_rgb_blind_audit_20260706_summary`
- 路由覆盖：9715 routes / 43 scenarios
- 成功标注：8614 routes，`skipped_abnormal_duration=963`，`data_missing_skip=138`
- RGB 帧：1102886；成功标注帧：1062401

注意：RGB blind 自动检测主要用于发现 R4/R5 灯控、STOP、无灯路口线索；它不能可靠自动区分
R2/R3。因此 R2/R3 结论以当前规则输出 + 逐场景 RGB sheet/历史人工复核为主，
blind RGB 只作为冲突提示。

## 2026-07-08 R2 回退专项复核

用户指出上一版把部分普通 R1/R-E1 场景误升为 R2。当前修正后，runtime 不再根据
非 TwoWays 场景级 XODR sparse scan 批量开放 R2；`LAYOUT_R2_ROUTE_IDS` 为空。
R2 只保留在 `*_TwoWays`、`InvadingTurn` 这类候选本身允许 R2 的场景，或后续经
逐 route / 逐帧 RGB 确认后手工写入白名单。

专项重跑命令：

```bash
python AutoMoT/keyframe_filter/quick_start.py annotate-rs \
  --scenario all \
  --max-frames-per-route 0 \
  --lead-data-root AutoMoT/lead_data \
  --output-dir AutoMoT/keyframe_filter/collection_output/current_r2_reaudit_20260708
```

该 all-run 为单进程长跑，先完成若干风险场景后在 `DynamicObjectCrossing` 中途手动
中断以避免继续占用机器；随后对所有当前允许 R2 的候选场景做了逐场景全量补跑。
未完整完成的 `DynamicObjectCrossing` 不计入本表。

| Scenario | routes | frames | R2 routes | 当前规则输出占比 |
|---|---:|---:|---:|---|
| Accident | 172 | 23553 | 0 | R1 87.0%, R4 13.0% |
| AccidentTwoWays | 457 | 71237 | 457 | R2 80.0%, R4 16.0%, R5 4.0% |
| BlockedIntersection | 152 | 22521 | 0 | R1 5.7%, R4 64.6%, R5 29.6% |
| ConstructionObstacle | 170 | 22231 | 0 | R1 85.2%, R4 14.8% |
| ConstructionObstacleTwoWays | 454 | 70530 | 454 | R2 81.8%, R4 14.2%, R5 4.0% |
| ControlLoss | 306 | 39612 | 0 | R1 78.5%, R4 15.0%, R5 6.6% |
| CrossJunctionDefectTrafficLight | 123 | 6723 | 0 | R1 7.4%, R4 92.6% |
| CrossingBicycleFlow | 48 | 5005 | 0 | R1 37.0%, R4 63.0% |
| HazardAtSideLaneTwoWays | 88 | 14923 | 88 | R2 87.2%, R4 6.3%, R5 6.5% |
| InvadingTurn | 98 | 11883 | 59 | R1 72.4%, R2 12.7%, R4 1.0%, R5 13.9% |
| ParkedObstacleTwoWays | 96 | 14030 | 96 | R2 81.1%, R4 3.0%, R5 15.9% |
| VehicleOpensDoorTwoWays | 104 | 14157 | 104 | R2 44.8%, R4 49.9%, R5 5.3% |

结论：

- 普通非 TwoWays 风险场景 `Accident`、`ConstructionObstacle`、`ControlLoss`、
  `BlockedIntersection`、`CrossJunctionDefectTrafficLight`、`CrossingBicycleFlow`
  均未再产生 R2。
- `AccidentTwoWays` 与 `ConstructionObstacleTwoWays` 每条有效 route 均保留 R2，
  且 R2 占约 80% / 81.8%，说明 TwoWays 的对向单车道主体没有被压回 R1。
- `HazardAtSideLaneTwoWays`、`ParkedObstacleTwoWays`、`VehicleOpensDoorTwoWays`
  也均保留 R2；其中 `VehicleOpensDoorTwoWays` 的 R4 比例约 49.9%，必须继续结合
  RGB sheet 核查是否真实灯控，不能只凭场景名或远灯态覆盖 R2。
- `InvadingTurn` 只在 59/98 条有效 route 产生 primary R2，更多帧为 R1/R5/R4 + U-E5；
  这符合“对向侵入是 EVENT，R2 只在对向单车道/侵占规则空间成立时出现”的方向，
  但 review ratio 高，需要逐帧看图确认 U-E5 是否过早。
- 后续若发现非 TwoWays route 确实是对向单车道，必须先逐帧 RGB 复核，再加入
  `LAYOUT_R2_ROUTE_IDS`；不能按 scenario 批量打开。

逐帧 RGB evidence：

- `AutoMoT/keyframe_filter/collection_output/r2_rgb_route_sheets_20260708/route_sheet_manifest.csv`
  包含 1577 条 route 的 contact sheet 路径、异常时长标记、当前 RS/EVENT 分布和 R2 帧数。
- `AutoMoT/keyframe_filter/R2_ROUTE_RGB_REVIEW_INDEX_20260708.csv` 是可追踪的 R2 人工复核索引，
  按 route 关联 `first_sheet` / `sheet_dir`、`r2_ratio`、`review_required_ratio`。
  该索引只用于人工逐帧确认；不作为自动真值。

## 场景占比

| Scenario | 候选 RS | 当前规则输出占比 |
|---|---|---|
| Accident | R1,R4,R5 | R1 77.9%, R4 15.3%, R5 6.8% |
| AccidentTwoWays | R2,R4,R5 | R2 80.8%, R4 15.5%, R5 3.7% |
| BlockedIntersection | R1,R4,R5 | R1 5.7%, R4 53.0%, R5 41.3% |
| ConstructionObstacle | R1,R4,R5 | R1 74.8%, R4 16.6%, R5 8.5% |
| ConstructionObstacleTwoWays | R2,R4,R5 | R2 81.4%, R4 14.3%, R5 4.3% |
| ControlLoss | R1,R4,R5 | R1 51.0%, R4 28.5%, R5 20.5% |
| CrossJunctionDefectTrafficLight | R1,R4 | 旧审计 R1 4.7%, R5 95.3%；当前口径应改为 R4+U-E7 |
| CrossingBicycleFlow | R1,R4 | R1 37.6%, R4 62.4% |
| DynamicObjectCrossing | R1,R4,R5 | R1 61.3%, R4 23.4%, R5 15.3% |
| EnterActorFlow | R3 | R3 100.0% |
| EnterActorFlowV2 | R3 | R3 100.0% |
| HardBreakRoute | R1,R3,R4,R5 | R1 52.1%, R3 14.0%, R4 20.9%, R5 13.0% |
| HazardAtSideLane | R1,R4,R5 | R1 58.4%, R4 35.6%, R5 5.9% |
| HazardAtSideLaneTwoWays | R2,R4,R5 | R2 86.5%, R4 6.3%, R5 7.2% |
| HighwayCutIn | R3,R4 | R3 99.1%, R4 0.9% |
| HighwayExit | R3 | R3 100.0% |
| InterurbanActorFlow | R1,R3,R5 | R1 47.8%, R3 0.5%, R5 51.7% |
| InterurbanAdvancedActorFlow | R1,R5 | R1 61.4%, R5 38.6% |
| InvadingTurn | R1,R2,R4,R5 | R1 45.7%, R2 10.5%, R4 1.0%, R5 42.8% |
| MergerIntoSlowTraffic | R3,R4 | R3 99.9%, R4 0.1% |
| MergerIntoSlowTrafficV2 | R3 | R3 100.0% |
| NonSignalizedJunctionLeftTurn | R1,R5 | R1 3.5%, R5 96.5% |
| NonSignalizedJunctionLeftTurnEnterFlow | R1,R5 | R1 2.0%, R5 98.0% |
| NonSignalizedJunctionRightTurn | R1,R4,R5 | R1 3.9%, R4 11.6%, R5 84.5% |
| OppositeVehicleRunningRedLight | R1,R4 | R1 10.0%, R4 90.0% |
| OppositeVehicleTakingPriority | R1,R4,R5 | R1 4.0%, R4 16.4%, R5 79.6% |
| ParkedObstacle | R1,R4,R5 | R1 76.1%, R4 14.5%, R5 9.4% |
| ParkedObstacleTwoWays | R2,R4,R5 | R2 79.8%, R4 3.0%, R5 17.1% |
| ParkingCrossingPedestrian | R1,R4,R5 | R1 41.8%, R4 49.0%, R5 9.2% |
| ParkingCutIn | R1,R4,R5 | R1 31.8%, R4 60.0%, R5 8.2% |
| ParkingExit | R1,R4 | R1 63.9%, R4 36.1% |
| PedestrianCrossing | R1,R4,R5 | R1 15.2%, R4 68.8%, R5 16.0% |
| PriorityAtJunction | R1,R4,R5 | R1 21.5%, R4 61.4%, R5 17.1% |
| RedLightWithoutLeadVehicle | R1,R4 | R1 6.2%, R4 93.8% |
| SignalizedJunctionLeftTurn | R1,R4 | R1 17.5%, R4 82.5% |
| SignalizedJunctionLeftTurnEnterFlow | R1,R4 | R1 18.3%, R4 81.7% |
| SignalizedJunctionRightTurn | R1,R4 | R1 14.2%, R4 85.8% |
| StaticCutIn | R1,R3,R4,R5 | R1 46.8%, R3 42.6%, R4 7.1%, R5 3.5% |
| T_Junction | R1,R4,R5 | R1 1.3%, R4 97.3%, R5 1.4% |
| VehicleOpensDoorTwoWays | R2,R4,R5 | R2 44.8%, R4 49.9%, R5 5.3% |
| VehicleTurningRoute | R1,R4,R5 | R1 17.3%, R4 62.4%, R5 20.3% |
| VehicleTurningRoutePedestrian | R1,R4,R5 | 场景级候选仍为 R1/R4/R5；非 TwoWays R2 暂停动态加入，待所有 id 逐帧 RGB 复核后再按 route 白名单开放 |
| noScenarios | R1,R4,R5 | R1 69.7%, R4 12.6%, R5 17.7% |

## 候选池结论

### 已修正并通过全量验证

- `*_TwoWays` 当前不再输出 R1 或独立停车主标签：
  - `AccidentTwoWays`: R2/R4/R5 = 80.8/15.5/3.7
  - `ConstructionObstacleTwoWays`: 81.4/14.3/4.3
  - `HazardAtSideLaneTwoWays`: 86.5/6.3/7.2
  - `ParkedObstacleTwoWays`: 79.8/3.0/17.1
  - `VehicleOpensDoorTwoWays`: 44.8/49.9/5.3
- 这符合“有效可行驶通道为对向单车道”的口径：两侧停车/障碍/开门风险使侧向 lane 不可行驶时，也按 R2。

### 暂不建议加入候选的 RGB blind 提示

- `CrossJunctionDefectTrafficLight` 的 RGB blind R4 是看见红绿灯硬件；当前口径保留 R4，故障/失效语义由 U-E7 表达。
- `EnterActorFlow*`、`HighwayExit`、`MergerIntoSlowTrafficV2` 的 RGB blind R5 多来自匝道/导流线/路牌/路面线误报；候选保持 R3。
- `NonSignalizedJunctionLeftTurn*` 的 RGB blind R4 需要人工 sheet 复核；当前 strict no-R4 规则暂不改。
- Signalized 场景的 RGB blind R5 多来自 stopline/crosswalk；不应因此加入 R5。

### 本轮根据 RGB 收紧的候选池

- `ParkingCutIn`: 全量逐帧规则输出为 R1/R4/R5，额外查看 R4/R5 evidence sheet 与 route RGB sheet 后未见稳定停车空间主导段；切入由 U-E3 表达。targeted rerun 后 RS 为 R1/R4/R5=31.8/60.0/8.2，EVENT 为 R-E1/R-E4/R-E5/U-E3，无候选外 R-E2。
- `StaticCutIn`: 全量逐帧规则输出为 R1/R3/R4/R5，额外查看 R4/R5 evidence sheet 与 route RGB sheet 后未见稳定停车空间主导段；merge/highway 桶保留 R3，普通城市桶保留 R1/R4/R5。targeted rerun 后 EVENT 为 R-E1/R-E3/R-E4/R-E5/U-E3 + 少量允许的 R-E2，无候选外事件。

### 需要继续复核的候选池

- `T_Junction`: R4 仍占 97.3%，R5 仅 1.4%。虽然候选正确包含 R1/R4/R5，但需要继续看 RGB sheet，确认无灯/STOP T 路口是否仍被 R4 吃掉。
- `VehicleOpensDoorTwoWays`: R4 占 49.9% 偏高；候选池本身正确，但需继续检查真实灯控证据，避免开门/停车路段被 R4 过度覆盖。
- `VehicleTurningRoutePedestrian`: 曾抽查 `Town12_393_0/1`、`Town12_398_0/1`、`Town13_84_0` 后发现疑似非 TwoWays R2；
  但由于需要按“对向单车道”严格逐帧复核全部 id，runtime 当前不开放非 TwoWays R2，避免普通 R1/R-E1 被误升。
- `noScenarios`: R5 已有 17.7%，说明保守候选 R1/R4/R5 是必要的；但 blind_R5_label_R1 仍多，后续可专门审计 noScenarios 的 STOP/无灯召回。

## 产物

- route 级盲审：`/tmp/automot_all_scenarios_rgb_blind_audit_20260706/route_blind_rs_event_audit.json`
- route 级 span/details：`/tmp/automot_all_scenarios_rgb_blind_audit_20260706/route_blind_rs_event_details.json`
- 场景级候选池汇总：
  `/tmp/automot_all_scenarios_rgb_blind_audit_20260706_summary/scenario_road_structure_candidate_audit.{json,csv,md}`
