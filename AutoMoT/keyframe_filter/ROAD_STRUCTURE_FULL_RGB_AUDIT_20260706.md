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
| InterurbanAdvancedActorFlow | R1,R5 | R1 55.9%, R5 44.1% |
| InvadingTurn | R1,R2,R4,R5 | R1 45.7%, R2 10.5%, R4 1.0%, R5 42.8% |
| MergerIntoSlowTraffic | R3,R4 | R3 99.9%, R4 0.1% |
| MergerIntoSlowTrafficV2 | R3 | R3 100.0% |
| NonSignalizedJunctionLeftTurn | R1,R5 | R1 3.5%, R5 96.5% |
| NonSignalizedJunctionLeftTurnEnterFlow | R1,R5 | R1 2.0%, R5 98.0% |
| NonSignalizedJunctionRightTurn | R1,R4,R5 | R1 51.6%, R4 7.4%, R5 41.0% |
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

### 2026-07-09 HazardAtSideLane EVENT 纠正

- XML 的 `bicycle_drive_distance/bicycle_speed` 与逐帧 RGB 均确认
  `HazardAtSideLane*` 核心是自行车/行人从侧边进入或横穿自车路径，不是静态障碍。
- EVENT 改为 `R-E1 -> U-E4 -> R-E2 -> R-E1`：`dist_to_biker<=30m` 的可交互段连续
  U-E4，对象离开后仍在回目标车道时才 R-E2；TwoWays 的 R2 仅表示道路结构。
- 全量复验覆盖 182 条 route / 34117 帧：`U-E4=7059, R-E2=1423, U-E2=0`；
  HazardAtSideLane / TwoWays 的 U-E4 最短分别为 14 / 28 帧，均没有多段碎片。
  其中 U-E4 结束后 8 帧内仍有回目标车道/横向回正证据的残留漏接数为 0，直接进入 R-E2，
  不再夹 `R-E1` 空洞。
- 17 条 `Town13 route30_*` 的 meta 全程 `scenario_active=None, dist_to_biker=inf`，
  完整 RGB 也没有自行车/行人，保持普通道路/路口事件，不强制制造 U-E4。

### 2026-07-09 高速边界、InvadingTurn 与无信号路口复核

- HighwayExit 旧版全量 85 route / 10416 帧在“只向后补最多 4 帧”口径下令 R-E2 从
  499 帧增至 884 帧；2026-07-09 在该基础上继续前补 2 帧、后补 2 帧，累计边界为
  前补最多 2 帧、后补最多 6 帧。新口径 30 route / 3627 帧 smoke 输出 R-E2=422，
  其中轨迹扩边 298 帧且全部保持 R3；同批误设“前后各 2 帧”时仅 R-E2=297。
- MergerIntoSlowTraffic / MergerIntoSlowTrafficV2 参考 HighwayCutIn / HighwayExit 重新收口：
  起始/中间普通主线跟车保持 R-E1，trigger-only 圆窗不再单独制造 R-E3，靠近 merge/actor-flow
  切 R-E3，真实目标变道切 R-E2；R-E2 核心按轨迹前后各最多补 5 帧，R-E2 后仍在
  actor-flow/merge 近邻的 tail 最多 64 帧保持 R-E3，3 帧以内夹在 R-E1 中的孤立 R-E3
  小岛平滑回 R-E1。2026-07-09 全量 188 route / 31855 帧：
  R-E1=18407、R-E2=2487、R-E3=10949、R-E4=12；
  其中 MergerIntoSlowTraffic 为 R-E1=8122、R-E2=1076、R-E3=4129、R-E4=12，
  V2 为 R-E1=10285、R-E2=1411、R-E3=6820。代表 RGB 边界：
  `Town12_968_0` 为 `R-E1 f0-111 -> R-E3 f112-138 -> R-E2 f139-158 -> R-E3 f159-205`；
  `Town06_route_000832` 为 `R-E3 f0-7 -> R-E2 f8-30 -> R-E3 f31-58`。
- InterurbanActorFlow 全量 90 route / 11007 帧：删除 active+close-trigger 的无控制源 R5，
  R-E1 从 4622 增至 6091，R-E5 从 6023 降至 4550。新增 R-E2 前 3 / 后 4 轨迹扩边后，
  30 route / 3629 帧 smoke 输出 R-E2=350，其中扩边 226 帧且全部位于 R1，没有跨入 R5/R-E5。
- InterurbanAdvancedActorFlow 全量有效 72 route / 8453 帧：junction 配置从 `55/25m`
  放宽到 `72/33m` 后，RS 为 R1/R5=4728/3725，未打开 R4；EVENT 为
  R-E1/R-E2/R-E5=4390/915/3148。R-E2 覆盖 55 条 route 的 107 段，其中
  R5 内 577 帧、相邻 R1 过渡 338 帧；新 advanced junction 轨迹补偿直接标记 595 帧，
  event review 为 0。
- InvadingTurn 全量 98 route / 11883 帧：最终 U-E5 按 RGB 可见锥桶/对向占道长度保持。
  核心仍优先看 R2 对向侵占规则与本地轨迹响应；final pass 对连续
  `passive_oncoming_invasion`、trigger>=35m 且有 R2 或 R1 响应证据的长 cluster 输出 U-E5，
  单段最多补 48 帧，避免锥桶还在画面中就退回 R-E1，也避免全程异常。2026-07-10
  复跑 U-E5=4744，含 R2 的有效 route 均不再缺 U-E5。代表 RGB 边界：
  `Town12_1229_0` 为 `U-E5 f29-78`，`Town12_1826_0` 为 `U-E5 f42-81`，
  `Town13_1378_0` 为 `U-E5 f28-77`，`Town13_1469_0` 为 `U-E5 f76-123`。
- 三类 NonSignalizedJunction 全量 426 route / 37512 帧：最终
  `R1=14342, R4=709, R5=22461`，不再近乎全程 R5。代表 `Town12_1375_0`
  为 `R1 f0-33 -> R5 f34-67 -> R1 f68-72`，与 RGB 的直道、STOP 右转、驶离一致。
- 2026-07-09 追加复核 `NonSignalizedJunctionRightTurn` 全量 93 route / 8074 帧：
  修复前 `R1=3446, R4=709, R5=3919`，修复后 `R1=4170, R4=598, R5=3306`。
  主要问题不是通用窗口本身，而是 `scenario_active`、远处 `traffic.stop` 和 meta/bbox 灯控在局部路口核心外
  仍能直接制造 R4/R5。现使用 `distance_to_intersection_index_ego`、`dist_to_junction` 与近距离
  trigger 形成 RightTurn 专用核心门控；代表 `Town12_1210_0` 从
  `R5 f0-5 -> R1 f6-34 -> R5 f35-75 -> R1 f76-78 -> R4 f79-82`
  收敛为 `R1 f0-34 -> R5 f35-74 -> R1 f75-82`，RGB 中 f0/f5 是直道、f35/f50 是 STOP 核心、f75/f80 已驶离。
  灯控子集未被删除，`Town13_75_0` 为 `R1 f0-25 -> R4 f26-183 -> R5 f184-189 -> R1 f190-200`。
- 2026-07-10 追加复核 `NonSignalizedJunctionLeftTurnEnterFlow + NonSignalizedJunctionRightTurn`
  全量 268 route / 24005 帧：LeftTurnEnterFlow 进入窗口放宽到 `junction_pre_m=84`，
  RightTurn 放宽到 `junction_pre_m=63` 且保留局部核心门控。路口短缝合并同步改 RS+EVENT，
  同一 R4/R5 路口段中夹入 <=12 帧 R1/R2 或 R-E1/R-E2 的残留均为 0。
  `Town03 route001042` 为 `R5/R-E5 f0-45 -> R1/R-E1 f46-63`，
  `Town13 route001061` 为 `R5/R-E5 f0-305 -> R1/R-E1 f306-312`。

### 2026-07-09 VehicleOpenDoor / Hazard 恢复边界复核

- VehicleOpensDoorTwoWays 全量 104 route / 14157 帧：只平移紧接 U-E2 的恢复 R-E2。
  代表 `Town12_1544_0` 从 `U-E2 f40-57 -> R-E2 f58-74` 调整为
  `U-E2 f40-54 -> R-E2 f55-70`，即起点提前 3 帧、终点提前 4 帧。
- HazardAtSideLane* 全量 182 route / 34117 帧：U-E4 保持 7059 帧且仍为单段；
  U-E4 后恢复 R-E2 补回 819 帧，最终 R-E2=1423；若恢复段被 R4/R5 接管，
  仍保留 regular+R-E2 overlay。代表 `Town12_2297_0` 为
  `U-E4 f23-37 -> R-E2 f38-46`，`Town12_1497_0` 为
  `U-E4 f77-130 -> R-E2 f131-136`。
- HazardAtSideLane 非 TwoWays 前 30 帧只有 bbox-only STOP、close-trigger 或 untrusted XODR 时不再升 R5：
  初始弱路口降级 393 帧，`Town13_1619_10` 从开头暗光直道 R5 恢复为全程 R1，
  `Town12_2193_0` 开头 f7-20 从 R5 回 R1，后面看到真实红绿灯后才进入 R4。
- HazardAtSideLaneTwoWays 的 bbox-only 初始伪 R5 清除 336 帧：
  `R2 13013 -> 13349`，`R5 973 -> 637`。代表 `Town12_1364_0` 的 f7-21
  从 R5 恢复 R2；真实初始信号路口 `Town12_1067_0` 仍保持 R4。
- 两个 Hazard 场景最终 `primary_event` 均位于当前 RS allowed events，违规数 0。

### 2026-07-09 路口控制源与停车事件边界复核

- 全量覆盖 9 场景、1429 route、166928 帧，最终 RS/EVENT 候选违规 0。
- T_Junction：R4/R5 混合 route `50 -> 0`，R1 `397 -> 6054`，退出直行段明显缩短。
- PedestrianCrossing：混合 route `79 -> 0`，R1 `2758 -> 4390`；进入窗口收紧。
- PriorityAtJunction：99/99 route 有有效灯态，锁定 R4；混合 route `73 -> 0`，
  R5 清零，R1 `7156 -> 8109`。
- OppositeVehicleTakingPriority：确有 12 条有效灯态 route，因此保留 R4/U-E7；
  有灯 route 锁 R4、无灯 route 锁 R5，混合 route `10 -> 0`，R1 `5662 -> 6668`。
- OppositeVehicleRunningRedLight：U-E6 从 21162 帧压缩到主冲突上下文 1075 帧、163 route；
  所有 U-E6 帧均保留 `R-E4 + U-E6` 同帧叠加。主冲突 span 前补 1 帧、后补 2 帧上下文，
  避免 RGB 复核时只看到 2 帧异常闪现。
- ParkingExit：6 条无有效灯态却输出 R4 的 route 清零；R-E2 从 6454 增至 8698，
  继续到横向动作/回正完成。
- ParkingCutIn：U-E3 从 1955 增至 3059，避免 RGB 刚进入切入核心就提前退出。
- ParkedObstacle / TwoWays 的 U-E2 后恢复 R-E2 均按“开始提前最多 3 帧、结束提前最多
  4 帧”平移；独立变道不动。

### 2026-07-10 Priority/Parked/ParkingCutIn 边界复核

- OppositeVehicleTakingPriority：按 RGB 接近路口边界把进入侧从 50m 放宽到
  `junction_pre_m=75`。全量复验 97 route / 13238 帧，RS 为
  R1=6668、R4=2078、R5=4492；EVENT 为 R-E1=6668、R-E4=2078、R-E5=5、U-E7=4487。
- ParkedObstacle：按 RGB 把进入侧从 60m 放宽到 `junction_pre_m=72`。全量复验
  168 route / 20676 帧，RS 为 R1=18330、R4=2346；EVENT 为
  R-E1=12734、U-E2=3656、R-E2=1963、R-E4=2323。
- ParkedObstacleTwoWays：全量复验 96 route / 14030 帧，RS 为
  R2=11805、R4=454、R5=1771；EVENT 为 R-E1=5863、U-E2=4534、R-E2=1413、R-E4=449、R-E5=1771。
  当前未发现残留短 `R-E2 -> R-E4/R-E5 -> R-E2` 插缝；代码保留 8 帧以内短路口插缝合并兜底。
- ParkingCutIn：U-E3 改为近距离 cut-in + 动态响应/横向证据触发，R4/R5 overlay
  不再按 distance-only 拖尾。全量复验 97 route / 13892 帧，U-E3=330 帧 / 80 route，
  最长连续 span 7 帧。`Town12_Rep0_1757_0` 从旧 f48-74 收为 f48-51，
  `Town12_Rep0_815_0` 从旧 f211-236 收为 f211-215；f52/f216 之后虽然 cut-in 车仍近，
  但无 brake/hazard 且自车轨迹已稳定，回 R-E4。

### 2026-07-09 R4/R5 单控制源锁

- 全局规则：所有同时允许 R4/R5 的场景，同一条 route 不允许 `R4 -> R5` 或 `R5 -> R4`
  互跳。若 route 内同时出现 R4/R5，按有效 `traffic_light_state` 或连续可见 traffic light
  判断控制源；有灯锁 R4，无灯锁 R5。
- EVENT 不改非常规触发逻辑：只同步当前 RS 的 regular event，即 R4 对 R-E4、R5 对 R-E5；
  已存在的 U-E4/U-E6/U-E7/U-E8 继续保留为同帧叠加事件。
- `VehicleTurningRoutePedestrian` 全量 91 route / 20399 帧验证：修复前 mixed R4/R5 route=50、
  直接相邻 `R4 <-> R5` 跳变=47；修复后 mixed route=0、直接跳变=0。
  计数从 `R1/R4/R5=8610/5629/6160` 调整为 `8610/8066/3723`；
  `U-E4` 保持 3454 帧不变，说明只修控制源抖动，没有重写行人事件逻辑。
- 全部 24 个同时允许 R4/R5 的场景在历史全量 annotation 上复算：旧结果 mixed route=759、
  direct R4/R5 jump route=698；按新 route 级控制源锁复算后 mixed=0、direct=0。
  实际重跑 Accident / AccidentTwoWays / BlockedIntersection / ConstructionObstacle /
  ConstructionObstacleTwoWays 共 1405 route / 210072 frame，输出 mixed=0、direct=0，
  `route_junction_control_lock` 共改写 1834 帧。

### 2026-07-10 ParkingExit / PedestrianCrossing RGB 边界复核

- ParkingExit：全量 241 route / 17586 帧复验，RS 不变为 R1=11302、R4=6284；
  EVENT 从 `R-E2=8698,R-E1=2604,R-E4=6284` 调整为
  `R-E2=7903,R-E1=3399,R-E4=6284`。210 条初始 R-E2 route 中 158 条收尾提前 5 帧，
  52 条保持不变；RGB 抽查 `Town03_route_001610` f34-f44 已是主路正常跟车，不应继续拖 R-E2。
- PedestrianCrossing：全量 98 route / 18141 帧复验，入口没有任何 route 提前，
  首个 R4/R5 起点变化为 0 或向后 1-11 帧；RS 从
  `R1=4074,R4=12722,R5=1345` 调整为 `R1=4367,R4=12621,R5=1153`。
  `Town13_1106_0`、`Town12_827_0` 这类 1-3 帧 R1 短跳按 RGB 路口/横穿上下文同步回填
  RS+EVENT；`Town13_1239_*` 的 16-20 帧普通直行段不按短缝强行并入路口。

## 产物

- route 级盲审：`/tmp/automot_all_scenarios_rgb_blind_audit_20260706/route_blind_rs_event_audit.json`
- route 级 span/details：`/tmp/automot_all_scenarios_rgb_blind_audit_20260706/route_blind_rs_event_details.json`
- 场景级候选池汇总：
  `/tmp/automot_all_scenarios_rgb_blind_audit_20260706_summary/scenario_road_structure_candidate_audit.{json,csv,md}`
