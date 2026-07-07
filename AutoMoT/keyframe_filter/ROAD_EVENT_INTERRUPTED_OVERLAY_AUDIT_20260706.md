# Interrupted EVENT Overlay Audit 2026-07-06

本轮目的：处理少量“R1/R2 视角下的突发事件尚未自然结束，但 ROAD_STRUCTURE 已进入 R4/R5
路口控制区”的边界。典型例子是 Accident / Construction / ParkedObstacle 绕障过程中突然进入
信号灯或无灯路口窗口，旧逻辑会把 `U-E2 -> R-E4/R-E5` 硬切断，导致障碍动作和后续 `R-E2`
恢复段丢失。

## Rule

不把 ROAD_STRUCTURE 强行改回 R1。当前帧如果已经有稳定 R4/R5 证据，primary RS 仍保持 R4/R5；
但 EVENT 层允许短时叠加：

```text
primary_road_structure = R4/R5
events = {R-E4/R-E5, U-E1/U-E2/U-E3/U-E4 or R-E2}
primary_event = safety/recovery event
event_evidence.interrupted_event_overlay = {...}
```

触发条件：

- 前一帧是非 R4/R5 的 `U-E1/U-E2/U-E3/U-E4`；
- 当前帧 primary RS 被 R4/R5 接管；
- 当前帧仍有对应突发事件证据，或有明确回目标/原车道的 `R-E2` 恢复证据。

退出条件：

- 对象距离 / hazard / route 横向回正证据自然结束；
- 或总叠加达到 `24` 帧（约 6s）；
- 或 `R-E2` 恢复子阶段达到 `12` 帧（约 3s）。

这样可以保留“R4/R5 控制源已经到来”和“R1/R2 视角突发动作尚未完成”两套信息，同时避免叠加状态污染后续普通路口。

## Code Path

- `collector.py::_apply_event_route_postprocess`
  - 检测 `U-E1/U-E2/U-E3/U-E4 -> R4/R5 regular` 的硬切断；
  - 写入 `event_evidence.interrupted_event_overlay`；
  - 同帧 `events` 同时保留路口 regular event 与 overlay event。
- `collector.py::_apply_event_candidate_clamp`
  - 只有 `interrupted_event_overlay.active=true` 的帧才临时放行 overlay event；
  - 其余 R4/R5 帧仍按当前 RS allowed events 收紧。

## Validation

清理 `/tmp/automot*` 后，根分区可用空间从约 36G 回到约 105G。随后重跑：

```bash
python AutoMoT/keyframe_filter/quick_start.py annotate-rs \
  --scenario Accident \
  --max-frames-per-route 0 \
  --output-dir /tmp/automot_interrupted_overlay_accident_20260706
```

以及同向障碍高风险场景的全量前两类：

```bash
python AutoMoT/keyframe_filter/quick_start.py annotate-rs \
  --scenario ConstructionObstacle,ParkedObstacle,... \
  --max-frames-per-route 0 \
  --output-dir /tmp/automot_interrupted_overlay_highrisk_20260706
```

完成并统计的覆盖：

| Scenario | Routes | Frames |
|---|---:|---:|
| Accident | 172 | 23553 |
| ConstructionObstacle | 170 | 22231 |
| ParkedObstacle | 168 | 20676 |
| Total | 510 | 66460 |

结果：

- `interrupted_event_overlay`：20 条 route / 315 帧，约占 0.47%。
- 最大 overlay age：24 帧，命中上限后退出为 R4/R5 regular。
- 当前 `event_evidence.allowed_events` 违规数：0。

RGB 复核样例：

- `Accident/Town05_Rep0_route_001775...`：f101-f124 为 R5 控制区内的事故绕行/回正 overlay；
  f125 起回 R5/R-E5。
- `Accident/Town12_Rep0_645_0...`：f155-f178 为 R4 控制区内仍可见事故车辆/警车和回正动作；
  f179 起回 R4/R-E4。
- `ParkedObstacle/Town06_Rep0_route_001930...`：夜间停放障碍在 R4 控制区附近延续到 f91；
  f92 起回 R4/R-E4。
- `ConstructionObstacle/Town12_Rep0_653_0...`：施工/障碍核心接近 R4，f84-f91 保留 overlay；
  f92 起回 R4/R-E4。

结论：叠加状态确实存在，但比例很低；用 evidence + 24 帧总上限可以覆盖自然退出，同时不影响后续普通路口。

## Full R1 Sudden-Event Audit 2026-07-07

按用户要求，继续覆盖所有包含 R1 突发事件的场景逐 route / 逐帧重跑，输入为
`AutoMoT/lead_data`，并沿用 abnormal duration route 剔除规则。覆盖场景：

`HardBreakRoute, BlockedIntersection, Accident, ConstructionObstacle, ParkedObstacle,
HazardAtSideLane, AccidentTwoWays, ConstructionObstacleTwoWays, ParkedObstacleTwoWays,
HazardAtSideLaneTwoWays, VehicleOpensDoorTwoWays, ParkingCutIn, StaticCutIn,
DynamicObjectCrossing, ParkingCrossingPedestrian, PedestrianCrossing, CrossingBicycleFlow,
VehicleTurningRoute, VehicleTurningRoutePedestrian`。

最终统计：

| Metric | Value |
|---|---:|
| Eligible routes | 3552 |
| Frames | 526001 |
| Overlay frames | 1472 |
| Overlay routes | 99 |
| Max overlay age | 24 frames |
| Raw U-E -> R4/R5 regular boundaries | 675 |
| Direct boundaries whose previous frame was not already overlay | 623 |

按场景 overlay 帧数：

| Scenario | Overlay frames |
|---|---:|
| HardBreakRoute | 18 |
| Accident | 24 |
| ConstructionObstacle | 43 |
| ParkedObstacle | 24 |
| AccidentTwoWays | 1 |
| ConstructionObstacleTwoWays | 177 |
| VehicleOpensDoorTwoWays | 168 |
| ParkingCutIn | 434 |
| DynamicObjectCrossing | 12 |
| ParkingCrossingPedestrian | 139 |
| PedestrianCrossing | 48 |
| CrossingBicycleFlow | 333 |
| VehicleTurningRoute | 51 |

关键修正：

- `DynamicObjectCrossing/Town12_Rep0_4346_0...` RGB 显示 f99-f108 内横穿车辆仍在 R4 控制区内，
  旧规则只看 pedestrian/biker 距离，导致 `U-E4 -> R-E4` 被切断；现改为 U-E4 在
  动态横穿 / 行人 / 自行车 / 转弯冲突场景中允许中距离短续，最多 10 帧。
- U-E4 近距离或 emergency/hazard 证据仍可按 24 帧总上限继续；中距离短续只用于防止
  路口 RS 抢占的前几帧，不把普通路口长期污染为突发事件。
- `Accident/Construction/Parked/VehicleOpensDoor/ParkingCutIn` 等 R1 绕障、cut-in、
  急刹类被 R4/R5 夹断的核心问题已缓解：RGB 样例中同帧保留 `R-E4/R-E5 + U-E*`
  或恢复 `R-E2`，直到对象/回正证据结束或上限退出。
- 复查 `Accident/Town03_Rep0_route_001792...` 后发现 RGB 是沿墙直道，不是路口；
  旧规则把 bbox stop-sign / traffic-light 与静态 XODR junction hint 组合成 R5/R4。
  现对 `Accident/ConstructionObstacle/ParkedObstacle/ControlLoss` 的 outside-XML
  R4/R5 入口增加 local-junction guard：stop/yield 不能单独创建 R5；traffic-light
  outside-XML 需要 bbox/meta/XODR/trigger 之一提供本地路口上下文。后续复查
  `Accident/Town03_route_001793` 发现 XML trigger 起点附近普通前车减速被误触发为
  U-E2，且环岛/弯曲多连接 junction 中 `light_hazard` 会短暂闪成 R4；现要求同向障碍
  U-E2 必须有具体障碍距离、active scenario、scenario obstacle、vehicle hazard 或真实偏离/回正证据，
  并把 roundabout-like static junction loop 压回 R1。再复查 `Town03_route_001794`
  发现 route 快结束的持续红绿灯路口应恢复 R4；现用 `traffic_light_state + bbox traffic_light +
  near static signal` 窄召回末尾稳定灯控段。回归后 1792 无伪 U-E2/R4，
  1793/1794 的中段环岛/弯道伪 R4/R-E2 被清掉，末尾稳定灯控段保留 R4/R-E4；overlay id 清单同步重生为
  99 route / 1472 帧。

RGB 抽样结论：

- 障碍 / cut-in / accident 类 overlay 与画面吻合，蓝色 overlay 段内仍有障碍、事故车辆、
  开门车辆、cut-in 车辆或回正动作；退出后画面进入普通路口/正常跟车。
- HardBreakRoute 剩余 4 个 direct 边界多为低光或前车已转离/不再急刹，未强续 U-E1 是合理保守策略。
- U-E4 场景 direct 边界数量仍多，集中在 `VehicleTurningRoute`,
  `VehicleTurningRoutePedestrian`, `PedestrianCrossing`, `DynamicObjectCrossing`。
  RGB 抽样显示其中大量是行人/骑行者/转弯车辆已经离开冲突点、或对象只在远处/侧方可见；
  因此不应把所有 `U-E4 -> R4/R5` 边界都强行变成双状态。当前策略只在近距离、
  hazard 或中距离短窗证据存在时叠加。

结论：这轮修正缓解了真正的“R1 突发事件被路口 RS 硬夹断”的问题，但不会承诺所有
`U-E4 -> R4/R5` 边界都必须双状态；是否叠加以 RGB 可见冲突对象、meta 距离/hazard
和 10/24 帧上限共同决定。
