# ROAD / EVENT 分类重标注方案

本文整理 `classifier_logic.txt` 中的人工调研结论，并结合旧版
`rule_based_keyframe_filter.py` 的生成方式，形成一套更清晰的两层语义体系：

- 上层：`ROAD_STRUCTURE`，表示当前帧处于哪一种驾驶决策规则空间。
- 下层：`EVENT`，表示当前帧在这个规则空间中触发了哪些常规或突发驾驶事件。

核心口径：`ROAD_STRUCTURE` 不是纯物理几何分类，而是驾驶决策规则空间分类。不同道路结构之所以要分开，
是因为它们下层可触发的事件、通行优先级和动作约束不同。

## 0. 数据 / XML 命名规则

本方案中凡是需要从 LEAD run 找 route XML，都必须使用用户整理后的固定命名：

```text
lead_data/<Scenario>/<run_id>
  -> parse (Scenario, Town, route_key)
  -> data/lead/<Scenario>/<Town>_<route_key>.xml
```

解析规则：

- `Scenario` 必须直接取 run 的父目录，不能从 XML 内 scenario type 或 `data_routes` 源目录反推。
- `run_id` 先剥末尾 `MM_DD_HH_MM_SS` 时间戳，再只在存在时剥尾部采集后缀 `_route0`。
- 剩余部分就是 `route_key`；`Town12_route15` 这类 legacy key 本体里的 `route15` 不能剥，也不能要求它带 `_route0`。
- 文件名公式固定为：`route_key` 以 `route_` 开头时用 `<Town>_<route_key>.xml`，否则用 `<Town>_route_<route_key>.xml`。

示例：

```text
Town03_Rep0_route_001783_route0_... -> data/lead/<Scenario>/Town03_route_001783.xml
Town12_Rep0_1054_0_route0_...       -> data/lead/<Scenario>/Town12_route_1054_0.xml
Town06_Rep0_Town06_13_route0_...    -> data/lead/<Scenario>/Town06_route_Town06_13.xml
Town12_Rep0_Town12_route15_...      -> data/lead/<Scenario>/Town12_route_Town12_route15.xml
```

2026-07-03 全量核对：`lead_data` 9715 个 run 去重后 9294 个 `(Scenario,Town,route_key)`；
`data/lead` 正好 9294 个 XML，缺失 0、冗余 0、命名不规范 0、XML 解析失败 0、内容结构异常 0。
40 个 XML 的 `data_routes` 源在其它 scenario 目录，不是缺失；现有
`ParkedObstacle/Town12_route_Town12_route15.xml` 覆盖有效并与
`lead_data/ParkedObstacle/Town12_Rep0_Town12_route15_*` 对应，不能当作 `xml_available=false`。

## 1. 为什么要替换旧 status/subgoal

旧 `keyframes_all_scenarios.json` 来自 `rule_based_keyframe_filter.py`，主要逻辑是：

- 每个 CARLA scenario 固定选择 `initial + 3 middle + final`。
- middle event 由 scenario 绑定的 `dist_to_*`、speed、accel、brake 等信号粗略定位。
- 缺少 meta 信号时退到 bbox 最近距离，再不行用 RGB 文件大小变化做 fallback。
- 一个 scenario 只输出少量关键点，不表达整段 route 内道路结构和事件的多次切换。

这个方法可以作为候选 keyframe / 候选 span 提议器，但不适合作最终帧级语义监督。
问题主要在于：

- scenario 名称同时混合道路结构、异常对象、路线意图和参与者行为。
- 同一 scenario 中常常同时存在 R1、R4、R5 等多个驾驶决策空间。
- 旧 middle event 只有 3 个点，无法覆盖“正常行驶 -> 突发事件 -> 恢复 -> 进入路口”的连续过程。
- bbox / RGB fallback 语义弱，只能用于人工抽检或低置信度候选。

因此后续应把 scenario 降级为候选集先验，而不是直接当作状态真值。

## 2. 总体层级

推荐使用两层主标签：

```text
ROAD_STRUCTURE: R1-R6 单选
EVENTS: R-E / U-E 多选
```

这里 `ROAD_STRUCTURE` 建议单选，因为它代表当前帧主要按照哪套规则空间做决策。
`EVENTS` 可以多选，因为一帧里可能同时存在背景常规事件和安全关键突发事件。例如：

```text
ROAD_STRUCTURE = R4 信号灯路口
EVENTS = [R-E4 信号灯路口通行, U-E6 违规车辆冲突]
```

再比如 Accident 的完整过程可以理解为：

```text
R1 常规道路 / 同向可行驶道路
  -> EVENTS = [R-E1 跟车 / 车道保持]
  -> EVENTS = [R-E1 跟车 / 车道保持, U-E2 静态障碍物占道]
  -> EVENTS = [U-E2 静态障碍物占道]
  -> EVENTS = [R-E2 目标导向型变道] 或 [R-E1 跟车 / 车道保持]
  -> 如果进入路口，ROAD_STRUCTURE 切到 R4/R5
```

## 3. ROAD_STRUCTURE：驾驶决策规则空间

R1-R6 当前覆盖面足够，不需要继续细分。尤其 R1 可以明确作为“默认 / 其它全部”桶，
因为 LEAD 数据中大量片段本质就是车道保持、跟车和普通同向道路行驶。

### R1. 常规道路 / 同向可行驶道路

定义：当前帧仍处于同向道路或普通可行驶区域，决策规则主要是车道保持、跟车、安全距离、
同向变道、同向绕障前后恢复。

它可以作为默认桶：

- 没看到明确路口、红绿灯、停止线、横向车流时，默认 R1。
- 没看到主辅路、匝道、合流、驶出结构时，默认 R1。
- 没看到明显停车带 / 路边停车占道空间时，默认 R1。
- 普通跟车、普通车道保持、非结构化道路正常前进、环岛内可行驶路径都可归 R1。

典型场景或片段：

- ControlLoss
- noScenarios
- DynamicObjectCrossing 的多数正常行驶片段
- HardBreakRoute 急刹前后的普通跟车片段
- Accident / ConstructionObstacle / ParkedObstacle 的障碍前后普通片段
- StaticCutIn 起步时不在期望车道的部分片段，如果不是匝道或停车区

### R2. 双向单车道 / 借对向车道道路

定义：自车方向可用空间接近单车道，前方阻塞时可能需要短暂使用对向车道；或者对向车道参与当前决策。

这里暂不增加子属性，不拆成主动借道和被动让行。二者都属于 R2 规则空间，下层事件区分行为：

- U-E2：自车因静态障碍被迫借对向绕行。
- U-E5：对向车辆异常侵占自车道，自车被动让行。

典型场景：

- AccidentTwoWays
- ConstructionObstacleTwoWays
- ParkedObstacleTwoWays
- HazardAtSideLaneTwoWays
- VehicleOpensDoorTwoWays
- InvadingTurn

### R3. 高速 / 匝道 / 合流 / 驶出道路

定义：当前帧处于主辅路、匝道、合流、分流、驶出、高速切入等规则空间。
核心决策是速度匹配、侧后方间隙、目标车道和主路车流关系。

典型场景：

- EnterActorFlow / EnterActorFlowV2
- MergerIntoSlowTraffic / MergerIntoSlowTrafficV2
- HighwayCutIn
- HighwayExit
- InterurbanActorFlow 的合流/变道片段
- StaticCutIn 中如果位置确实在匝道/汇入口附近，也可归 R3

### R4. 信号灯路口

定义：当前帧处于路口决策区，且红绿灯是主要通行规则。

典型规则：

- 红灯停。
- 绿灯行。
- 绿灯左转 / 右转时仍需观察冲突对象。
- 如果信号灯正常，但对方闯红灯，仍然是 R4，突发事件由 U-E6 表达。

典型场景：

- RedLightWithoutLeadVehicle
- SignalizedJunctionLeftTurn
- SignalizedJunctionLeftTurnEnterFlow
- SignalizedJunctionRightTurn
- OppositeVehicleRunningRedLight
- T_Junction 中有灯控的片段
- CrossingBicycleFlow / PedestrianCrossing / VehicleTurningRoute 系列中有灯控的片段

### R5. 无信号灯 / 信号灯失效路口

定义：当前帧处于路口决策区，但没有正常可用的红绿灯规则，主要依赖路权、安全间隙、
横向车流、对向车流和行人/自行车让行关系。

典型场景：

- NonSignalizedJunctionLeftTurn
- NonSignalizedJunctionLeftTurnEnterFlow
- NonSignalizedJunctionRightTurn
- OppositeVehicleTakingPriority
- PriorityAtJunction
- CrossJunctionDefectTrafficLight
- T_Junction 中无灯控的片段
- PedestrianCrossing / VehicleTurningRoute 系列中无灯控的片段

CrossJunctionDefectTrafficLight 建议归 R5，但同时触发 U-E7，因为它不是普通无灯路口，
而是“规则源失效”导致四向冲突不确定性更高。

### R6. 路边停车 / 停车占道道路

定义：当前帧的有效通行空间和风险模型由路边停车、停车带、停车位汇入、开门、遮挡视线等因素主导。

典型场景：

- ParkingCutIn
- ParkingExit
- ParkingCrossingPedestrian
- VehicleOpensDoorTwoWays 中停车/开门风险明显的片段
- StaticCutIn 如果确实发生在路边停车区域附近，也可归 R6

R6 和 R2/R1 在视觉上可能有重叠，但如果单选 ROAD_STRUCTURE，建议以当前最主要决策规则为准：

- 需要处理停车位汇入、路边车启动、停车遮挡、开门风险时，优先 R6。
- 需要处理借对向车道或对向车侵占时，优先 R2。
- 停车风险不主导时，回到 R1。

## 4. EVENT：常规事件与突发事件

事件允许多选。常规事件表示当前背景驾驶任务，突发事件表示打断或覆盖背景任务的安全关键因素。

### 常规事件

#### R-E1. 跟车 / 车道保持

最基础的背景事件。适用于：

- 普通车道保持。
- 普通跟车。
- 保持安全距离。
- 障碍、急刹、路口、合流事件前后的正常行驶片段。
- R1 下的大多数帧。

#### R-E2. 目标导向型变道

定义：当前车道通常仍可行，但路线目标要求自车主动换到目标车道。

包括：

- 为高速出口提前变到右侧车道。
- 为左转提前变到左侧车道。
- 自车初始不在期望车道，需要修正到目标车道。
- 从停车位或停车侧汇入主行驶车道。

注意和 U-E2 区分：

- R-E2 是路线 / 目标导致的主动变道。
- U-E2 是当前路径被静态障碍阻断后的被迫绕行。

#### R-E3. 常规匝道合流 / 并线 / 驶出

定义：道路结构本身要求自车完成合流、并线、汇入或驶出。

适用于：

- EnterActorFlow / MergerIntoSlowTraffic 的自车主动合流。
- HighwayExit 的驶出过程。
- R3 下的常规速度匹配、找间隙、进入目标车流。

HighwayCutIn 不属于 R-E3，因为它是他车切入自车路径，应由 U-E3 表达。

#### R-E4. 信号灯路口通行

定义：自车依据正常信号灯规则完成停车、等待、直行、左转、右转或启动。

适用于 R4。

#### R-E5. 无信号灯路口通行

定义：自车依据路权、安全间隙、横向/对向车流完成直行、左转或右转。

适用于 R5。

#### 关于 R-E6

取消 `R-E6 常规停车等待 / 恢复启动`，不再作为独立事件。

原因：

- 它和 R-E3/R-E4/R-E5 大量重叠。
- 等待和恢复更像事件内部阶段，而不是新的事件类别。
- 如果后续需要表达停车等待，可在数据里用辅助字段，例如 `phase=wait_red`、
  `phase=wait_gap`、`phase=resume_green`、`phase=resume_after_clear`，但主 EVENT 不保留 R-E6。

### 突发事件

#### U-E1. 前车急刹 / 突然减速

定义：前车突然急刹或快速减速，自车必须紧急减速以避免追尾。

典型场景：

- HardBreakRoute

#### U-E2. 静态障碍物占道

定义：自车当前行驶路径被静态障碍物阻挡，原本车道保持或跟车任务被打断。

标注时机约束（重要）：

- route xml 里的 scenario `trigger_point` 不是 U-E2 的直接起标点，它只表示 scenario 机制触发。
- U-E2 应在“可见/可感知”后注入：例如相关 `dist_to_*` 距离进入阈值、或障碍进入前向可视区域。
- 不要在 trigger 刚发生就打 U-E2，否则会出现“自车尚未看到障碍但标签已触发”的噪声监督。
- 建议使用 on/off 双阈值和最短持续帧数，减少边界抖动。

包括：

- 事故车辆。
- 施工障碍。
- 停放车辆。
- 静止车辆。
- 路侧危险物。
- 开门车辆造成的静态占道风险。

典型场景：

- Accident
- ConstructionObstacle
- ParkedObstacle
- AccidentTwoWays
- ConstructionObstacleTwoWays
- ParkedObstacleTwoWays
- HazardAtSideLaneTwoWays
- VehicleOpensDoorTwoWays

同向绕行还是借对向绕行不需要拆成两个事件，由 ROAD_STRUCTURE 决定：

- R1/R6 + U-E2：多为同向绕行或停车等待。
- R2 + U-E2：多为借对向车道绕行。

#### U-E3. 动态车辆切入 / 动态占道

定义：其他车辆突然进入或即将进入自车未来路径，自车被动减速、让行或停车。

典型场景：

- HighwayCutIn
- ParkingCutIn
- StaticCutIn 后半段
- DynamicObjectCrossing 中存在动态对象干扰的片段

#### U-E4. 行人 / 自行车横穿

定义：行人、自行车或小型动态交通参与者进入自车预期路径，自车必须让行或停车。

它可以发生在 R1/R4/R5/R6，不绑定单一道路结构。

典型场景：

- CrossingBicycleFlow
- PedestrianCrossing
- ParkingCrossingPedestrian
- VehicleTurningRoute
- VehicleTurningRoutePedestrian

#### U-E5. 对向车辆异常侵占自车道

定义：对向车辆侵入自车车道，自车被迫减速或停车等待。

典型场景：

- InvadingTurn

它和 R2 + U-E2 的区别：

- R2 + U-E2：自车主动借对向车道绕过静态障碍。
- R2 + U-E5：对向车进入自车道，自车被动让行。

#### U-E6. 违规车辆冲突

定义：自车本来按规则可以通行，但其他车辆违反规则，导致自车必须避让。

典型场景：

- OppositeVehicleRunningRedLight

它和 U-E7 的区别：

- U-E6：规则正常，但参与者违规。
- U-E7：规则源本身失效或不可用。

#### U-E7. 信号灯故障 / 路口规则失效

定义：路口正常灯控规则失效，自车不能按红灯停、绿灯行通行，必须全程观察四向车流。

典型场景：

- CrossJunctionDefectTrafficLight

它通常覆盖整个路口通行阶段，不是短暂插入事件。

#### U-E8. 前方道路暂时阻塞 / 阻塞解除

定义：前方道路暂时无可通行空间，自车需要长时间等待；之后阻塞解除，自车恢复行驶。

典型场景：

- BlockedIntersection

不要混入 U-E2。BlockedIntersection 的重点不是绕障，而是等待阻塞解除后重新判断道路可通行。

## 5. ROAD_STRUCTURE 到 EVENT 候选裁剪

这一步用于降低 Qwen 判断难度。候选表不是硬互斥表，事件可多选。

| ROAD_STRUCTURE | 常规事件候选 | 突发事件候选 |
|---|---|---|
| R1 常规道路 / 同向可行驶道路 | R-E1, R-E2 | U-E1, U-E2, U-E3, U-E4 |
| R2 双向单车道 / 借对向车道道路 | R-E1 | U-E2, U-E5 |
| R3 高速 / 匝道 / 合流 / 驶出道路 | R-E1, R-E2, R-E3 | U-E3 |
| R4 信号灯路口 | R-E4 | U-E4, U-E6, U-E8 |
| R5 无信号灯 / 信号灯失效路口 | R-E5 | U-E4, U-E7, U-E8 |
| R6 路边停车 / 停车占道道路 | R-E1, R-E2 | U-E2, U-E3, U-E4 |

多选示例：

```text
R1 + [R-E1, U-E2]
常规车道保持背景下看到静态障碍占道。

R4 + [R-E4, U-E6]
绿灯路口通行中遇到违规车辆冲突。

R6 + [R-E1, U-E4]
路边停车区域正常前进中遇到行人横穿。
```

## 6. Scenario / ROAD_STRUCTURE / EVENT 候选映射

具体映射不再在本文维护，统一放在 `ROAD_EVENT_CANDIDATE_MAPPING.md`：

- 第 3 节：每个 scenario 的 `ROAD_STRUCTURE` 候选。
- 第 4 节：每个 `ROAD_STRUCTURE` 下允许的 `EVENT` 候选。
- 第 5 节：每个 scenario 进一步排除后的精细 `EVENTS` 候选。

这样避免本文的总原则和落地候选表出现两套不同版本。后续修改 scenario 候选或事件候选时，
优先修改 `ROAD_EVENT_CANDIDATE_MAPPING.md`。

## 7. 标注与推理流程

### Step 1：先判断 ROAD_STRUCTURE

Qwen 先在 scenario 给出的候选 ROAD_STRUCTURE 中选择当前帧主规则空间。

建议默认策略：

- 没有明确证据时保持 R1。
- 看到红绿灯、停止线、路口几何、横向车流时，切到 R4/R5。
- 看到主辅路、匝道、合流、出口时，切到 R3。
- 看到停车带、路边停车、停车位汇入、开门风险时，切到 R6。
- 看到双向单车道并且对向车道参与决策时，切到 R2。

### Step 2：在 ROAD_STRUCTURE 下选择 EVENTS

Qwen 只在当前 ROAD_STRUCTURE 对应的事件候选中选择，可多选。

建议输出格式：

```text
ROAD_STRUCTURE: R1
EVENTS: R-E1, U-E2
```

如果没有突发事件，输出一个常规背景事件即可：

```text
ROAD_STRUCTURE: R1
EVENTS: R-E1
```

### Step 3：由 EVENTS 推导 STATUS / SUBGOAL

如果后续仍需要 STATUS / SUBGOAL，可以从 EVENTS 推导：

```text
STATUS = 当前最主要、最安全关键的事件
SUBGOAL = 下一步应完成的事件或阶段目标
```

例子：

```text
ROAD_STRUCTURE: R1
EVENTS: R-E1, U-E2
STATUS: U-E2
SUBGOAL: R-E2
```

含义：当前主要问题是静态障碍物占道，下一步目标是完成安全绕行/回到目标车道。

## 8. 新数据重生成思路

建议先找突发事件 span，再做 ROAD_STRUCTURE / EVENTS 重标注。

### A. 突发事件 span 提议器

旧 `rule_based_keyframe_filter.py` 可改造成弱 span 提议器：

- U-E1：HardBreakRoute 中 `brake` 峰值、`accel_x` 最小值、前车距离快速缩小。
- U-E2：障碍距离字段进入阈值、速度下降、通过后距离增大或速度恢复。
- U-E3：cut-in 距离字段下降、相对对象进入前方路径、速度/刹车响应。
- U-E4：pedestrian/biker 距离进入阈值，速度下降，目标离开路径后恢复。
- U-E5：InvadingTurn 中 cut-in/oncoming 距离、自车停车等待、对向车离开。
- U-E6：OppositeVehicleRunningRedLight 中路口距离、威胁车接近、自车让行。
- U-E7：CrossJunctionDefectTrafficLight 的整个路口接近到通过阶段。
- U-E8：BlockedIntersection 中长时间低速/停车、前方阻塞、解除后恢复。

建议输出 span，而不是单帧：

```json
{
  "event": "U-E2",
  "start_frame": 32,
  "peak_frame": 58,
  "end_frame": 76,
  "source": "meta_dist+speed",
  "confidence": 0.82
}
```

### B. ROAD_STRUCTURE 弱判断

道路结构更依赖视觉，旧 meta 很难直接判断 R1/R4/R5 的切换。

初期策略：

- 用 scenario 候选表降低选择难度。
- Qwen 根据当前帧视觉判断 ROAD_STRUCTURE。
- 对没有明确证据的帧默认 R1。
- R4/R5、StaticCutIn、HazardAtSideLane、T_Junction、PedestrianCrossing 等需要重点人工抽检。

### C. EVENTS 合成

每帧的 EVENTS 可以由两部分合成：

```text
EVENTS = 常规背景事件 + 命中的突发事件 span
```

例子：

- R1 下没有 span：`EVENTS=[R-E1]`
- R1 下命中 U-E2 span：`EVENTS=[R-E1, U-E2]`
- R4 下命中 U-E6 span：`EVENTS=[R-E4, U-E6]`
- R5 下命中 U-E7 span：`EVENTS=[R-E5, U-E7]`

## 9. ROAD_STRUCTURE 调研与实现协议

本节合并原先分散在 ROAD_STRUCTURE 调研协议和地图/XML 标注方案里的全局执行口径。
逐场景 ROAD_STRUCTURE 规则、证据需求、阈值风险和未完善点单独维护在
`ROAD_STRUCTURE_PER_SCENARIO_RULES.md`；`collection_output/rs_research/<Scenario>/`
里的本地调研结论更新后，必须同步沉淀到该逐场景文档。

### 9.1 三源输入与边界

每个 frame 的 `primary_road_structure` 必须由三类证据共同约束：

- `lead_data/<Scenario>/<run_id>/metas/*.pkl`：帧级事实来源，包括 `pos_global`、`theta`、
  speed、junction、灯态、active scenario、`dist_to_*`、bbox/RGB 等。
- `data/lead/<Scenario>/<Town>_<route_key>.xml`：route、trigger、scenario 参数和地理窗口。
- `CARLA_0915/.../<Town>.xodr`：静态拓扑来源，包括 road/lane/junction/signal/controller、
  stop/yield、parking/shoulder、merge/split 等。

XML trigger 只是 scenario 机制锚点，不是事件可见起点，也不是结构标签唯一依据。
XODR 只证明拓扑可能性，不提供实时灯色；实时灯色只用 meta 的
`traffic_light_state` / `light_hazard`。`dist_to_*` 主要服务 EVENT/span，可给 RS 加置信，
但不能单独决定 R2/R3/R6。

所有 LEAD run 在进入调研、标注、SFT/GoalGen/LeadMoT 数据集或 probe 前，必须先剔除异常时长 route：
4Hz 下 `rgb/*.jpg >= 361` 且不在 `BlockedIntersection` / `ControlLoss` 白名单内的 run
都视为异常采集。代码统一复用 `lead_video_tools.abnormal_duration_filter.is_abnormal_lead_route`。

### 9.1.1 帧级标注输出与页面验收口径

当前可执行标注 JSON 必须同时保留“候选全集”和“本帧结果”，两者不能混用：

- `road_structures`：该 scenario 允许的 ROAD_STRUCTURE 候选全集，用于约束搜索空间和兼容旧工具。
- `primary_road_structure`：当前 frame 的单选主 RS 标签。
- `secondary_road_structures`：当前 frame 的次要 RS，用于 R2/R6、R4/R3 等冲突或共存结构。
- `frame_rs_annotation`：当前 frame 的可解释标注块，至少包含
  `label/secondary/confidence/comment/rule_kind/rules_fired/decision_source/review_required/review_reasons/metrics/xodr_summary`。

`confidence` 的语义固定为“本帧 `primary_road_structure` / `frame_rs_annotation.label` 的置信度”，
不是 `road_structures` 候选全集的置信度。页面、probe、人工复核表和后续训练数据都必须按这个口径读取。

Web 验收页面必须展示三层信息：

- 本帧最终标签：绿色主标签读取 `frame_rs_annotation.label`，旁边展示 secondary、confidence、comment 和 review 原因。
- 场景候选全集：单独展示 `road_structures`，只说明该 scenario 可选哪些 RS。
- 证据归因：展示 XML/route 投影、trigger 距离、LEAD meta 灯态/active scenario、XODR
  source/trusted/road/lane/junction/opposite/parking/merge 等摘要。若 XODR 不可信、XML 投影误差过大或证据冲突，
  必须在 review 状态里能看出原因。

这也是后续“代码 → 小样本可视化 → 查错帧 → 修正规则/阈值 → 再跑 smoke”的闭环入口；
不能只看候选 RS 分布或平均置信度来判断规则正确。

### 9.2 调研包完成标准

每个 scenario 都必须先有独立调研包，默认输出：

```text
keyframe_filter/collection_output/rs_research/<Scenario>/
```

最小要求：

- 覆盖该 scenario 涉及的全部 town；每个 town 至少抽 5 条分散 run
  （不足 5 条可读 run 则全读）。
- 对每条 sampled run 核对 XML route/trigger、XODR 局部拓扑、ego global trace、
  meta frame features、RGB contact sheet 和关键边界帧。
- 边界增补必须覆盖 trigger 前后、active scenario 变化、`traffic_light_state` 变化、
  `junction_id/is_junction` 变化，以及候选 RS 分数接近帧。
- `rules/thresholds.json` 的阈值必须写明 `source`、`supporting_runs`、
  `reviewed_artifacts` 和 `reason`；匿名 magic number 只能作为临时默认值。

`collection_output/` 是本地自动调研输出目录，不入库、不 push。需要共享的全局结论应沉淀进本文，
逐场景结论应沉淀进 `ROAD_STRUCTURE_PER_SCENARIO_RULES.md`，运行入口说明沉淀进
`README.md`，可执行阈值再沉淀进小型规则配置或后续代码。

完成 `rs_research.py --samples-per-town 5` 只表示 `auto_artifacts_ready`，不代表规则可标
`final_complete=true`。只有当 `manual_map_rgb_checked=true`、
`thresholds_have_provenance=true`、`runtime_rule_ready=true` 同时成立时，runtime 才能把
`complete_investigation_status.is_complete` 置为 true。

当前 5-id/town 自动调研结论：

- 43 个 scenario 均已生成自动证据包。
- 只有 `NonSignalizedJunctionLeftTurn/Town10HD` 本地缺可读 meta；其它 scenario 自动输入完整。
- 43/43 个 scenario 的 `map_rgb_alignment_status=not_checked`、
  `manual_final_complete=false`。
- 当前阈值仍是 `temporary_default_rule_config`，`reviewed_artifacts=[]`；
  因此只能作为临时规则设计和 smoke test 依据。

### 9.3 Runtime 统一门控

代码应先按 R1-R6 统一证据门控生成候选，再套 scenario config，不要写 43 个互相独立的
magic-number 分支。

| RS | High-confidence 门控 | 常见否决 |
|---|---|---|
| R1 | 默认桶；同向障碍、急刹、动态对象、control loss 本身不改变 RS | brake/accel/vehicle_hazard/walker_hazard 不参与 RS 升级 |
| R2 | scenario prior + trigger/active 窗口 + 局部 XODR opposite driving lane + 同向车道不足 | 只有 TwoWays 名称或 `dist_to_*` 最高 medium；灯态/路口主导时 R4/R5 primary |
| R3 | 高速/合流 scenario prior + actor-flow/merge/exit 窗口 + ramp/merge/split/lane-count-change 证据 | `start_actor_flow/end_actor_flow` 字段名不够；自行车/路口横向流 veto R3 |
| R4 | 有效 `traffic_light_state` / `light_hazard`，或同源受控 junction/controller 支撑 | `CrossJunctionDefectTrafficLight` 由 R5 override；阻塞/违规只是 EVENT |
| R5 | nonsignalized/priority/defect prior + route/trigger/junction 窗口 + 无有效正常灯态或 defect override | 连续有效灯态且非 defect scenario 时不 high R5 |
| R6 | Parking* / parking 子型 prior + parking trigger/active 窗口 + parking/shoulder/curbside 或停车汇入证据 | `ParkedObstacle` 不是 R6；灯控路口主导时 R4 primary |

仲裁优先级默认：

```text
R4/R5 > R3 > R2/R6 > R1
```

例外：

- `CrossJunctionDefectTrafficLight`：R5 覆盖 R4，并写 `defect_signal_overrides_R4`。
- `VehicleOpensDoorTwoWays`：R2/R6 可同时成立，primary 取决于是否必须占用/等待对向车道。
- `noScenarios`：没有 scenario prior 时，只允许强灯态 + 同源受控 junction 升级 R4，否则 R1。

### 9.4 规则族结论

- `same_direction_obstacle`：`Accident`、`ConstructionObstacle`、`ParkedObstacle`。
  静态同向障碍是 EVENT 证据，不把整段升级成 R2/R6；只在受控路口窗口进入 R4。
- `twoways_obstacle` / `invading_turn` / `vehicle_opens_door_twoways`：
  只有 XML trigger、XODR 对向/双向单车道拓扑、meta active 或距离字段共同成立时进入 R2；
  TwoWays 名称本身不能全程给 R2。
- `highway_merge` / `interurban`：只有 ramp/merge/split/highway 拓扑支持时进入 R3；
  EnterFlow/Merger/HighwayExit 的行驶事件不能替代 XODR 拓扑证据。
- `signalized_junction`：灯态有效、受控 junction 或 controller/traffic light 近邻成立时进入 R4；
  `BlockedIntersection` 和 `OppositeVehicleRunningRedLight` 的阻塞/违规只是 EVENT，不改成 R5。
- `nonsignalized_junction` / `defect_junction`：无有效灯态、stop/yield/priority 或灯故障机制成立时进入 R5；
  `CrossJunctionDefectTrafficLight` 强制 R5 覆盖 R4。
- `parking` / `parking_exit` / `static_cutin`：R6 只给 parking/shoulder/curb/parking-exit 结构窗口；
  停车相关 scenario 在信号灯路口段仍优先 R4/R5。
- `default_meta_map` / `noscenario`：默认 R1；ControlLoss、HardBreak、DynamicObjectCrossing、
  HazardAtSideLane 等行为/突发事件本身不改变 RS，只能通过灯态或 junction 证据临时进入 R4。

初始阈值只作为调研起点：`junction_pre_m=40~60`、`junction_post_m=20~40`、
`two_way_min_pre_m=45~80`、`merge_pre_m=30~50`、`merge_post_m=40~50`、
`parking_pre_m=20~35`、`parking_post_m=50~60`。正式阈值必须在 scenario 调研包中补齐
provenance 后才能进入 high-confidence runtime。

### 9.5 Smoke test 修正

已有小样本 smoke test 证明 `rs_research.py` 和 `ScenarioCollector` 能跑通，但暴露了需要收紧的规则：

- 需要 per-frame XODR 时必须用能 `import carla` 的 Python；否则 XODR 证据为空，
  R2/R3/R6 只能 medium/low + review。
- R2 primary 不能只靠 scenario + trigger；若没有 `has_opposite_driving_lane=true` 且
  `same_direction_lane_count<=1`，不应 high。
- R3 high 必须要求 ramp/merge/split/lane-count-change；只有 active/trigger 时最多 medium，
  且 `review_required=true`。
- R6 不能只靠附近 shoulder/parking hint；必须结合 Parking* prior、parking window、方向、
  bbox/RGB 路边车列。
- `route_projection_error_m > 5m` 时，无论候选分数多高，都必须 `review_required=true`。
- `complete=true` 只能解释为 auto input complete；文档和代码字段应区分
  `auto_input_complete` / `manual_final_complete`。

扩展 smoke test 又抽了 9 个高风险场景，每个 scenario 先跑 1 条 route：

```text
ParkingCrossingPedestrian, VehicleOpensDoorTwoWays, StaticCutIn,
InterurbanActorFlow, InterurbanAdvancedActorFlow,
NonSignalizedJunctionLeftTurnEnterFlow, T_Junction,
PedestrianCrossing, SignalizedJunctionRightTurn
```

观察结果：

| Scenario | 小样本分布 | 暴露的问题 |
|---|---|---|
| `ParkingCrossingPedestrian` | 276 帧：R4=227、R6=39、R1=10；review=202 | R4/R6 仲裁方向对，但大量 `xml_route_projection_error_high`，说明 XML sparse route 不能作为边界主依据。 |
| `VehicleOpensDoorTwoWays` | 76 帧：R6=52、R4=21、R1=3；15 次切换 | R2/R6 同时触发但 primary 多为 R6；切换偏碎，需要 hysteresis 和“是否必须借/等对向车道”的动作主因判断。 |
| `StaticCutIn` | 105 帧：R6=97、R1=8 | 该样本像 parking-side cut-in；但全程有 `r3_lacks_xodr_merge_split_confirmation`，说明 StaticCutIn 必须逐 run 分 R3/R6/R1，不能统一。 |
| `InterurbanActorFlow` | 122 帧：R3=102、R5=4、R1=16；review=109 | 前段 R3/后段 R4-R5 的思路对，但 `interurban_junction_r5_medium` 大量触发且 projection error 高，边界不能只靠 XML route_s。 |
| `InterurbanAdvancedActorFlow` | 120 帧：R5=80、R1=40；review=111 | junction primary 方向可用；R3 只应在明确 topology 时开放。 |
| `NonSignalizedJunctionLeftTurnEnterFlow` | 59 帧全 R5；review=37 | `enter_flow_not_r3` 正确；R5 窗口需要 meta junction/stop/yield + XODR 同源支撑。 |
| `T_Junction` | 163 帧全 R4；140 帧有灯态 | 大体合理；但 23 帧无有效灯态仍未 review，`review_if_no_tl=true` 需要落实为“无灯态且无同源 controller 时 review”。 |
| `PedestrianCrossing` | 154 帧：R4=114、R5=40；review=142 | 行人不决定 RS 的 veto 正确；R4/R5 边界被 projection error 污染，需要靠灯态和局部 junction。 |
| `SignalizedJunctionRightTurn` | 44 帧全 R4；26 帧有灯态；review=35 | 右转场景 R4 方向正确；right-turn channel 不能按 XML sparse line 判 exit。 |

由此把 runtime 思路进一步修正为“两级窗口”：

```text
candidate recall window:
  scenario prior + XML trigger/active/meta distance
  分数只给 0.55~0.70，用于把候选拉进来

topology/meta confirmation window:
  局部 XODR + meta 时序 + bbox/RGB 人工证据
  命中后才升到 0.85~0.95，并允许 high-confidence primary
```

具体含义：

- R2：召回可用 TwoWays prior + trigger/active；确认必须有 opposite lane、同向车道不足、
  或 RGB/bbox/动作主因证明对向参与。否则 primary R2 要 review，分数不超过 0.70。
- R3：召回可用 Highway/Merger/EnterFlow prior + actor-flow window；确认必须有 merge/split/ramp、
  lane-count change 或目标出口车道证据。`xodr_available=true` 不等于 R3 topology。
- R6：召回可用 Parking* prior + parking window；确认必须有 parking/shoulder/curbside、
  parking->Driving 转换或路边静态车列。普通 shoulder hint 不够。
- R4/R5：召回可用 junction/trigger window；确认必须看灯态、light_hazard、stop/yield/controller
  和 route 同源。`is_junction=false` 不代表不是 stopline approach。
- XML route projection error >5m 时，不能用 route_s 做边界；应改用 meta 时序和 XODR map waypoint
  吸附，并强制 `review_required=true`。

下一步代码完善优先级：

1. `rs_research.py` 增加 boundary frame 采样和 `projection_metrics.json`，把 projection error
   系统性记录到调研包。
2. `collector.py` 将当前特种 RS 的打分拆成 recall score 和 confirmation score。
3. `collector.py` 对 `r2_scenario_trigger_medium`、`r3_lacks_xodr_merge_split_confirmation`、
   `r6_lacks_xodr_parking_or_shoulder_confirmation` 强制 review，并限制分数上限。
4. 加最短持续帧与 transition margin：R2/R3/R6 少于 4 帧的孤立片段合并或 review，
   R4 有有效灯态时不可被平滑覆盖。

### 9.6 错帧回查流程

用户指出某一帧错标时，先查该 scenario 的调研包，而不是直接调阈值：

```text
scenario README
  -> maps/*route_trigger_ego_trace.png
  -> XML/XODR local evidence
  -> meta/*__frame_features.jsonl
  -> RGB contact sheet / boundary frames
  -> thresholds.json / runtime rule
```

归因口径：

- R4/R5 错：优先查灯态、controller/stop/yield、route/junction 同源，不先改 scenario 名映射。
- R2 错：优先查局部 opposite lane、same-dir lane count、trigger/active 窗口和障碍窗口。
- R3 错：优先查 ramp/merge/split/topology，不把 EnterFlow 名称直接当 R3。
- R6 错：优先查 parking/shoulder/curb/停车起步空间，不查 parked obstacle 字段本身。
- R1 错：检查是否有被 veto 的事件误当结构，或 junction/merge/parking 窗口过窄。

## 10. 待核实问题

需要后续结合 `collection_output/rs_research/<Scenario>/maps/*route_trigger_ego_trace.png`、
`rgb/*sample_contact_sheet.jpg` 和 `meta/*__frame_features.jsonl` 核实：

- `HazardAtSideLane` 当前 RS 口径保持 R1/R4；只有 map/RGB 明确存在对向参与时才允许加入 R2。
- `StaticCutIn` 当前保留 R1/R3/R4/R6 混合候选，必须按每个 run 的 map/RGB 拆分。
- `T_Junction` 当前为 signalized_junction 规则族，若 meta 灯态缺失则 review，不自动转 R5。
- `PedestrianCrossing`、`VehicleTurningRoute*` 保留 R4/R5 候选；最终按灯态、signal/controller、
  stop/yield 和 junction 证据决定。
- `PriorityAtJunction`、`OppositeVehicleTakingPriority` 当前稳定按 R5 规则族处理。
- `InterurbanAdvancedActorFlow` 当前不默认 R3；只有 XODR merge/highway/ramp 证据成立时才打开 R3。
- `NonSignalizedJunctionLeftTurn/Town10HD` 缺可读 meta，后续 EVENT/RS 评估必须标记该 town 的
  confidence 不高于 medium，直到补齐 meta 或人工确认。

## 11. 当前结论

最终推荐口径：

- R1-R6 保持 6 类，不继续细分。
- R1 明确作为默认 / 其它全部规则空间。
- R2 不增加子属性，主动借道和被动让行交给事件区分。
- R-E6 取消，不作为独立事件。
- EVENTS 支持多选，用常规事件描述背景任务，用突发事件描述安全关键打断。
- 旧 keyframe 逻辑只作为 span 提议器和人工抽检入口，不作为最终帧级 STATUS/SUBGOAL 真值。
- EVENT 规则必须消费新的逐场景 RS 结果：先定 `primary_road_structure`，再按
  `ROAD_EVENT_CANDIDATE_MAPPING.md` 求 scenario/event 候选交集。不能用 scenario 名直接决定 R2/R3/R4/R5/R6。

这套结构更贴近你的原始意图：先判断当前处于哪套驾驶决策规则空间，再在该空间下判断可能发生的事件。
