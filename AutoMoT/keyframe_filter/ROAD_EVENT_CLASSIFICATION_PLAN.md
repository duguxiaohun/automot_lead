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

运行时若某条 run 缺少 `metas/*.pkl` 或无法匹配 route XML，直接视为数据质量问题并
`data_missing_skip`，不进入 RS/EVENT 标注、调参或 probe。可视化/summary 必须写明
`missing_meta` / `missing_route_xml` 等原因；不要把这类样本作为规则失败，也不要用
RGB-only、bbox-only 或错误 XML/XODR 硬补逐帧标签。

## 1. 为什么要替换旧 status/subgoal

旧 `keyframes_all_scenarios.json` 来自 `rule_based_keyframe_filter.py`，主要逻辑是：

- 每个 CARLA scenario 固定选择 `initial + 3 middle + final`。
- middle event 由 scenario 绑定的 `dist_to_*`、speed、accel、brake 等信号粗略定位。
- 旧逻辑缺少 meta 信号时会退到 bbox 最近距离或 RGB 文件大小变化；新 RS/EVENT 监督不再这样做，
  缺 meta/XML 的 run 直接 `data_missing_skip`。
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
`EVENTS` 在数据结构上仍保留多选，但标注口径改成“默认单主事件，路口双触发例外”：

- 非路口/非十字路口核心段默认只输出一个主事件。障碍、cut-in、急刹、借道等非常规 span 命中时，
  主事件就是对应 U-E；退出非常规 span 后再回到 R-E1/R-E2/R-E3 等常规事件。
- 只有 R4/R5 路口规则空间允许常规路口事件和安全关键突发事件同帧共存，例如
  `R-E4 + U-E4`、`R-E4 + U-E6`、`R-E5 + U-E7`、`R-E4 + U-E8`。
- 即使在 R4/R5，也不能因为 scenario 名称就全程双触发；必须由 XML trigger/active window、
  meta 距离/速度/刹车、轨迹减速点和 RGB 可见对象共同限定触发范围。
- 场景级 EVENT 表只是上限；最终候选必须再和当前 ROAD_STRUCTURE 的候选池取交集，
  并始终保留当前 RS 的 regular event。R4/R5 路口候选池直接删除 `U-E2/U-E3`，
  所以红灯等待、路口排队、路口起步只能是 `R-E4/R-E5` 或路口专属 U-E。
- `R-E6` 已取消。停车等待、恢复启动、等待间隙都只是事件内部 phase，不单独作为主事件。

例如：

```text
ROAD_STRUCTURE = R4 信号灯路口
EVENTS = [R-E4 信号灯路口通行, U-E6 违规车辆冲突]
```

再比如 Accident 的完整过程应理解为单主事件切换：

```text
R1 常规道路 / 同向可行驶道路
  -> EVENTS = [R-E1 跟车 / 车道保持]
  -> EVENTS = [U-E2 静态障碍物占道]
  -> EVENTS = [R-E2 目标导向型变道]  # 轨迹开始回到目标/原车道时
  -> EVENTS = [R-E1 跟车 / 车道保持]
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
- 没看到高速/快速路、主辅路、匝道、合流、驶出结构时，默认 R1；但明确高速/merge
  scenario 的候选池不开放 R1，非路口段默认 R3。
- 没看到明显停车带 / 路边停车占道空间时，默认 R1。
- 普通跟车、普通车道保持、非结构化道路正常前进、环岛内可行驶路径都可归 R1。
- 环岛 / roundabout 明确归 R1：即使 XODR 把它编码成 junction road，也不能仅凭
  `is_junction`、junction connection 或 route trigger 升为 R4/R5。

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

### R3. 高速合流 / 匝道 / 分流 / 驶出决策结构

定义：当前帧处于高速/快速路、主辅路、匝道、合流、分流、驶出等会改变目标车道、
速度匹配或主辅路关系的规则空间。对于 `EnterActorFlow*`、`HighwayExit`、`MergerIntoSlowTrafficV2`
这类全量逐帧 RGB 已验证为稳定无灯控高速/快速路背景的 scenario，R3 是默认道路空间，
不再开放 R1/R4。`HighwayCutIn` 与 `MergerIntoSlowTraffic` 主体仍是 R3，但全量 RGB 发现少量真实灯控子集，
因此保留 R4 候选；R4 必须由逐帧 RGB/meta/bbox 灯控同源证据触发。对 `HardBreakRoute`、`PriorityAtJunction`
这类同样出现在 Town12/13 但 RGB 可能是城市/乡村/路口的场景，仍需逐 run 由 RGB/XODR 区分 R1/R3/R4/R5。
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
- 同向/借对向绕障完成后，轨迹开始回到原本目标车道或导航期望车道。

注意和 U-E2 区分：

- R-E2 是路线 / 目标导致的主动变道。
- U-E2 是当前路径被静态障碍阻断后的被迫绕行。
- 同一个非路口核心片段不要同时输出 R-E2 和 U-E2：接近/绕过障碍时是 U-E2，
  开始回归目标车道后切成 R-E2。
- 障碍恢复段不能假设道路是直线。代码必须结合 `meta["route"]` 的 ego-frame 近前方
  局部中心线、`signed_dist_to_lane_change`、自车速度/位置与 XODR/XML 触发位置判断：
  当轨迹已经过障碍核心，并出现负向 `signed_dist_to_lane_change <= -0.45m`、局部中心线
  偏移峰值后下降或紧邻 R-E2 边界时，可提前 1-2 帧切 R-E2；
  若 `signed_dist_to_lane_change` 仍为正且还没到障碍最近点，哪怕数值在下降，也只是正在进入避让车道，
  仍归 U-E2，不能提前切 R-E2；
  一旦 route 横向偏移回到中心线容差内且未来 3 帧稳定，R-E2 必须退出为 R-E1/R-E4/R-E5。
- 红绿灯路口场景内不要为了排队、起步或转向轨迹偏移加入 R-E2；R4 内常规任务统一归 R-E4。

#### R-E3. 常规匝道合流 / 并线 / 驶出

定义：道路结构本身要求自车完成合流、并线、汇入或驶出。

适用于：

- EnterActorFlow / MergerIntoSlowTraffic 的自车主动合流。
- HighwayExit 的驶出过程。
- R3 下的常规速度匹配、找间隙、进入目标车流。

HighwayCutIn 当前不默认视为 U-E3；它多是自车在主路保持/减速跟车或少量 R-E2 目标导向变道。
只有后续 RGB/轨迹复核确认“他车突然进入自车未来路径并迫使避让”时，才单独回灌 U-E3。

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
- BlockedIntersection 中由前车/前方阻塞突然停止导致的急减速片段。

标注时机约束：

- XML/active scenario 只能给候选窗口；真正起点看自车开始明显减速、brake/accel_x 峰值、
  前车距离快速缩小或 RGB 可见前车突然停止。
- 红灯等待不能误打 U-E1。若停车原因是红灯或正常路口队列，保持 R-E4/R-E5 或 U-E8。

#### U-E2. 静态障碍物占道

定义：自车当前行驶路径被静态障碍物阻挡，原本车道保持或跟车任务被打断。

标注时机约束（重要）：

- route xml 里的 scenario `trigger_point` 不是 U-E2 的直接起标点，它只表示 scenario 机制触发。
- U-E2 应在“可见/可感知”后注入：例如相关 `dist_to_*` 距离进入阈值、或障碍进入前向可视区域。
- 不要在 trigger 刚发生就打 U-E2，否则会出现“自车尚未看到障碍但标签已触发”的噪声监督。
- 2026-07-05 复核后，Accident / Construction / Parked / Hazard / VehicleOpensDoorTwoWays
  的距离触发整体后移约 2-3m；`speed_reduced_by_obj_distance` 只作为更保守的近 trigger
  辅助信号，避免对象距离弱信号过早吞掉 R-E2 恢复段。
- 建议使用 on/off 双阈值和最短持续帧数，减少边界抖动。
- 绕障的第一次离开原车道也属于 U-E2，不要在 U-E2 前先打一小段 R-E2；R-E2 只给
  绕过障碍后回原车道/目标车道的恢复变道。
- U-E2 持续到自车完成绕障核心动作或不再被障碍约束；如果轨迹开始回到原车道/目标车道，
  后续切成 R-E2；R-E2 结束后必须回到 R-E1/R-E4/R-E5 等当前道路常规事件，不能继续粘住。
- 如果在 trigger 附近自车没有产生绕行、借道、减速避让等轨迹/速度响应，视为该 run 未触发，
  不要硬写 U-E2。
- 检验准则：若 U-E2 已经远离 XML trigger 和具体障碍距离，且没有 route-change 轨迹，必须释放为常规事件；
  若路线末尾仍保持 U-E2，必须写入 event review，除非 RGB/距离证据能解释“终点仍被障碍约束”。

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

注意：

- `HighwayCutIn` 当前按用户先验先不把“侧方车辆进入主路”直接作为 U-E3 训练候选；
  它更常作为 R3 背景下的跟车/速度匹配，只有 RGB/轨迹明确显示自车被迫避让突发切入时再人工回灌。
- `ParkingCutIn` / `StaticCutIn` 仍保留 U-E3，因为停车车辆突然启动或动态占道会直接进入自车路径。
  但如果自车已经进入 R-E2 恢复/目标变道，随后只短暂回到 U-E3，则不重新开启 U-E3，
  而是把该短反跳并入 R-E2，避免 `R-E2 -> U-E3` 的状态震荡。
- `DynamicObjectCrossing` 经 2026-07-05 RGB top mismatch 复核后恢复 U-E3 候选：
  动态车辆/运动物体确实进入自车未来路径时不应全程压成 R-E1；但 U-E3 只覆盖交互短 span，
  不能把远处侧向车流、普通路口车辆或整段 R5 路口当成非常规。
- 和 U-E2 一样，U-E3 不能只靠 `dist_to_cutin_vehicle` 阈值粘住整段。距离最近点之后若对象开始远离、
  且没有 `brake_cutin` / `vehicle_hazard` 等持续强响应，应释放为当前道路 regular event；
  若自车仍处于目标/恢复变道，则只保留短 R-E2。

#### U-E4. 行人 / 自行车横穿

定义：行人、自行车或小型动态交通参与者进入自车预期路径，自车必须让行或停车。

它可以发生在 R1/R4/R5/R6，不绑定单一道路结构。

典型场景：

- CrossingBicycleFlow
- PedestrianCrossing
- ParkingCrossingPedestrian
- VehicleTurningRoute
- VehicleTurningRoutePedestrian

U-E4 必须分场景做时序分型，不能共用一个“路口 trigger 起点”：

- `ParkingCrossingPedestrian`：横穿发生在进入路口前或非路口停车遮挡区域。U-E4 只覆盖对象进入
  自车前向路径到让行/恢复的短 span；通过后即回 R-E1，不随之后可能出现的路口持续。
- `CrossingBicycleFlow`：通常是信号灯路口自行车横穿，可允许 `R-E4 + U-E4`，但 U-E4
  起点必须晚于自行车进入可见/可交互范围，不能从路口 approach 全程开始。
- `VehicleTurningRoute` / `VehicleTurningRoutePedestrian`：横穿多发生在自车已转弯或驶出路口过程中。
  U-E4 不能在转弯前过早触发；应等 RGB/轨迹显示对象将进入转弯后路径、或 meta 距离进入危险阈值。
- `PedestrianCrossing`：按 RGB/灯态分成 R4 或 R5；U-E4 只覆盖行人实际需要让行的交互 span。
- `DynamicObjectCrossing`：若 `dist_to_pedestrian` / pedestrian-bike 类 hazard 显示行人、骑行者或小动态对象横穿，
  可开放 U-E4；2026-07-05 子集审计中恢复 U-E3/U-E4 后，
  `event_regular_during_rgb_object_or_motion_activity` 从 2363 降到 2030 帧。

这些分型结论必须写入 per-scenario 调研记录或后续规则配置。RGB 未复核前，行人/自行车边界帧保持 review。

#### U-E5. 对向车辆异常侵占自车道

定义：对向车辆侵入自车车道，自车被迫减速或停车等待。

典型场景：

- InvadingTurn

它和 R2 + U-E2 的区别：

- R2 + U-E2：自车主动借对向车道绕过静态障碍。
- R2 + U-E5：双向窄路/对向占道空间中，对向车进入自车道，自车被动让行。
- R5 + U-E5：InvadingTurn 这类无信号/STOP 路口中，对向车异常侵占导致自车让行。

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

`OppositeVehicleTakingPriority` 按用户先验也纳入 U-E7 候选：其语义不是“对方违规”，而是无正常信号灯/
优先权规则下需要额外观察和让行。它仍由 R5 承载，不能回到 R4。

#### U-E8. 前方道路暂时阻塞 / 阻塞解除

定义：前方道路暂时无可通行空间，自车需要长时间等待；之后阻塞解除，自车恢复行驶。

典型场景：

- BlockedIntersection

不要混入 U-E2。BlockedIntersection 的重点不是绕障，而是等待阻塞解除后重新判断道路可通行。

BlockedIntersection 需要区分三类停车原因：

- 红灯/正常队列停车：只输出 R-E4，不输出 U-E1/U-E8。
- 前车突然刹停导致自车急减速：短 span 输出 U-E1。
- 前方路口空间被堵住且一段时间后解除：核心等待/解除 span 输出 U-E8；若仍在信号灯路口规则空间，
  可输出 `R-E4 + U-E8`，但不能让整条 route 都双触发。

## 4.1 事件触发门控总规则

所有非常规事件必须同时满足“scenario 白名单 + XML/active 候选窗口 + 行为/视觉证据”三层约束：

1. **scenario 白名单**：只允许本文和 `ROAD_EVENT_CANDIDATE_MAPPING.md` 中列出的非常规事件。
   如果某 scenario 没有列出某个 U-E，不允许由通用距离字段或 RGB 猜测临时加入。
2. **XML/active 窗口**：非常规事件只能在 XML trigger/route 近邻、`current_active_scenario_type`
   或已有 scenario 距离字段给出的候选范围内触发；窗口外回到常规事件。
3. **行为/视觉证据**：起点不得只取 XML trigger。必须看到速度/刹车/轨迹响应、目标距离进入阈值、
   或 RGB 中对象进入可交互范围。退出时也要看对象离开安全距离、轨迹恢复或速度恢复。

常规事件默认由 ROAD_STRUCTURE 决定：

- R1 无非常规：R-E1；若轨迹显示主动变道/回归目标车道，则 R-E2。
- R2/障碍核心借道绕障：U-E2；为绕障离开原车道仍是 U-E2；绕障后回目标车道：R-E2；
  R-E2 完成后非核心普通段回 R-E1/R-E4/R-E5。
- R3 高速/合流：R-E1 或 R-E3；若自车为出口/目标车道主动变道，则 R-E2。
- R4：R-E4。只有 U-E4/U-E6/U-E8 等路口安全事件进入有效 span 时才双触发。
- R5：R-E5。CrossJunctionDefectTrafficLight/OppositeVehicleTakingPriority 等有效 span 可叠 U-E7。
- R6 删除独立常规事件；停车区域正常跟车为 R-E1，自车从停车侧/路边并入主路为 R-E2。

当前可执行实现位于 `collector.py::RoadEventRuleEngine`，输出 `events`、
`primary_event`、`event_evidence`、`frame_event_annotation`。实现补充口径：

- `route` 局部中心线复核是通用的 R-E2 边界工具。实现上不直接拿前方 waypoint 的
  y 值当偏离，因为弯道会天然产生 y；而是用 `meta["route"]` ego-frame 近前方 1-8m
  点拟合局部切线，再把 ego 原点投影到这条局部中心线，得到 `route_lateral_offset_m`。
  `target_lane_change_active` 要同时满足局部中心线确实偏离，且有 `changed_route`、
  `signed_dist_to_lane_change` 或足够 offset 支撑；障碍恢复专用逻辑还允许过障碍核心后的
  负向 lane-change 距离作为“准备回正”信号，避免等车身已经回正后才补 R-E2。
  Town06 这类弯道路段不能只看 `route_lateral_abs_m`：局部 route 中心线可能贴着车身，
  造成 `route_centered=true` 但车辆仍在换道/回正。障碍恢复段若 `changed_route` 与
  `signed_dist_to_lane_change` 连续有效，R-E2 不能被中心线提前释放；必须等 signed
  lane-change 也结束后，才允许按中心线稳定回 R-E1/R-E4/R-E5。
  该思路可迁移到 `HighwayCutIn`、`HighwayExit`、`InterurbanActorFlow`、`ParkingExit`、
  `StaticCutIn` 等目标导向变道/汇入场景，用于避免弯路、出口曲线、路口转向或旧
  `changed_route` flag 把常规 R-E3/R-E4/R-E5 后段误标成 R-E2。它不用于决定 R4/R5
  路口控制源，也不单独触发 U-E3/U-E4/U-E5/U-E6/U-E7/U-E8。
  全局 route 后处理还会把夹在同一个常规 R-E1/R-E3/R-E4/R-E5 前后之间、长度不超过
  2 帧的孤立 R-E2 平滑回常规事件；ParkingExit/StaticCutIn 这类持续多帧并入/修正不会被该规则压掉。
- 障碍 / TwoWays / 开门 / Hazard 类：U-E2 只在障碍距离、对象减速距离、door/open 或 TwoWays
  core evidence 命中时触发；仅由 route 开头 XML trigger + `speed_reduced_by_obj_distance`
  触发、且短窗口内没有具体障碍距离/变道轨迹的初始 U-E2 会压回常规事件。
  U-E2 前紧邻的短 R-E2 若由避障离开原车道产生，会吸收进 U-E2；U-E2 结束后的 R-E2
  必须由 `changed_route`、负向 `signed_dist_to_lane_change <= -0.45m` 或 ego-frame route
  局部中心线收敛趋势支持，代表回原/目标车道。R-E2 到达中心线容差并稳定 3 帧后回当前道路常规事件。
  已进入恢复 R-E2 后，不允许再短暂回跳 U-E2；4 帧以内的 U-E2 反跳按距离/XODR 投影抖动合入 R-E2。
  2026-07-04 RGB 审计后，障碍类 U-E2 route 后处理允许合并
  6 帧以内的短 R-E1/R-E4/R-E5 gap，避免 XML/XODR 路口边界把仍在绕障的片段切碎；
  2026-07-05 中心线复核后，夹在两个 U-E2 span 中间的 1-2 帧 R-E2 视为抖动并合回 U-E2；
  远离 trigger 且无障碍/轨迹证据的 U-E2 尾段会释放为常规事件，路线末尾残留 U-E2 必须 event review。
  若 U-E2 刚被释放但随后 1-3 帧仍有 `changed_route` / `signed_dist_to_lane_change`
  与 ego-frame route 中心线横向偏离，且没有负向回正趋势，则这几帧仍属于避障换道未彻底完成，
  会补回 U-E2；一旦出现负向 `signed_dist_to_lane_change`、局部中心线收敛趋势或进入 R4/R5，
  就不再延长 U-E2，避免吞掉真正的恢复 R-E2。
  只要当前 RS 是 R4/R5 或存在有效灯控等待上下文，U-E2 会从候选中删除并释放为 R-E4/R-E5。
  EVENT route postprocess 末尾会再次执行候选池一致性兜底：若桥接/单核心等后处理又产生
  `R4/R5 + U-E2/U-E3`，强制回 `R-E4/R-E5` 并记录 `*_candidate_pool_excludes_u2_u3`。
  同一 route 内若出现多个 U-E2 span，不能按时长保留；必须优先保留具体事故/施工/停放障碍距离、
  绕障/回正轨迹和 route 中心线偏离证据最强的 span。`*_TwoWays` 的 R2 借对向车道绕障核心
  本身就是强 U-E2 证据，应在进入/准备进入对向借道核心时开始 U-E2，而不是等绕完后才补标。
  选择唯一 U-E2 span 时，TwoWays 优先看最终 R2 重叠帧数，再看具体障碍距离和中心线偏离；
  后处理若先切出短 R-E2/R-E1 但两侧仍属于同一 R2/障碍核心，必须二次合并回 U-E2。
  这样可以避免运动前车跟车距离误触发的早期 U-E2 或后段普通恢复片段抢占真正静态障碍核心。
  如果 U-E2 后面马上进入恢复/目标变道 R-E2，中间 4 帧以内的短 R-E1 必须并入 R-E2，
  不允许出现 `U-E2 -> R-E1 -> R-E2` 的状态断裂。
  对事故/施工/停放/side hazard/开门等障碍恢复类，该桥接窗口放宽到 8 帧；但 R-E2
  优先贴近最近 U-E2 恢复窗口；但 R-E2 不是只能出现在 U-E2/U-E3 之后。若没有近期 U-E2，
  仍可由真实自车换道证据独立保留：`changed_route`、`signed_dist_to_lane_change`、
  ego-frame route 局部中心线横向偏离/回正趋势必须至少命中一类，且当前 RS 不能是 R4/R5。
  没有这些 XODR/meta 中心线证据的孤立 R-E2，才视为后段弯道、普通跟车或投影扰动并释放回当前道路 regular event。
  post-U2 恢复窗口只允许由长度足够的 U-E2 核心 span 开启；1-3 帧的 early trigger
  抖动不能触发后续 R-E2，否则会把普通弯道或运动前车跟车误标成目标变道。
  对非 TwoWays 同向障碍，`dist_to_*` 不能按对称距离阈值一路粘到障碍点身后：最近点之后若
  距离回升且自车已经回到局部 route 中心线，U-E2 立即释放；若还在回正轨迹中，只给短 R-E2。
  最近点之前的正向 signed-distance 下降不能作为 R-E2 起点；它表示自车还在完成避障换道，
  必须继续保持 U-E2，直到贴近/越过障碍核心或出现负向回原/目标车道趋势。
- R4 primary 必须足够严格，因为 R4/R5 会从候选池删除 `U-E2/U-E3`。对同向障碍、默认和 noScenarios 场景，
  仅有 meta `traffic_light_state`、`light_hazard`、bbox `traffic_light` 或远处 static signal
  不足以升 R4；必须同时有 strong control context（路口/stopline/signal-junction 同源证据）。
  缺少上下文时保持 R1，让 U-E2/U-E3 仍能按对象证据触发。
  2026-07-05 RGB top mismatch 复核后，同向障碍类补充可见控制源恢复：
  `meta traffic_light_state + bbox traffic_light` 可在稀疏 XML window 外恢复 R4，
  `bbox/meta STOP/yield` 可恢复 R5；子集回归中 Accident 的 `blind_R5_label_R1`
  从 289 降到 7、`blind_R4_label_R1` 从 132 降到 21，ConstructionObstacle 的
  `blind_R5_label_R1` 从 231 降到 14，ParkedObstacle 从 304 降到 77。
  noScenarios 另有一个窄召回：若 meta `traffic_light_state` 有效、bbox 连续看到
  `traffic_light`，并处于 junction window，且无 STOP/yield 或 roundabout 证据，则按
  `r4_noscenario_stable_tl_bbox_approach` 给 R4/R-E4。它只修正常规无事件 route 的稳定灯控
  approach，不能外推到 Accident/UE2/UE3 等容易被路口候选干扰的场景。
- `HardBreakRoute`：U-E1 必须贴近 ego 实际 hard decel / brake / target speed drop；
  仅有近距离 lead 或 XML active window 不足以触发。低速近距离 vehicle hazard 可作为
  兜底，但不能把整段 active window 全部打成 U-E1。
  2026-07-05 RGB 复核发现该类会经过 STOP/无灯路口，因此候选池加入 R5/R-E5；
  子集审计 `blind_R5_label_R1` 从 1001 降到 104 帧。剩余 top case 多为雾天远处灯点/路面线导致 blind 过宽，
  不继续靠单帧 stopline/turn-marking hint 扩大 R5。
- `ParkingCutIn` / `StaticCutIn`：U-E3 必须有 cut-in 距离、`brake_cutin` 或
  `vehicle_hazard`，不能仅凭普通急减速触发。进入 R-E2 目标/恢复变道后，4 帧以内
  的 U-E3 反跳，或中间只夹 1-2 帧常规事件再跳 U-E3，视作 cut-in 证据抖动并合入 R-E2。
  U-E3 同样不进入 R4/R5 候选池；路口等待/起步阶段不能继续保持 U-E3。
  同一 route 内若出现多个 U-E3 span，也必须保留 cut-in 证据最强的 span：优先看
  `dist_to_cutin_vehicle`、`brake_cutin`、`vehicle_hazard` 和对象是否进入自车未来路径，
  普通跟车减速、红灯等待或无对象支撑的短扰动不能抢占真正动态占道核心。
  经过 cut-in 最近点后若距离回升且无持续强响应，后段不再保持 U-E3；若仍有自车回正/目标变道，
  只转短 R-E2，否则释放为 regular event。
  如果 U-E3 后面马上进入 R-E2，中间 4 帧以内的短 R-E1 同样并入 R-E2。
- `InvadingTurn`：U-E5 必须有对向车辆 hazard 或近距离对象证据，不能仅凭自车减速触发；
  同一对向侵占过程内 5 帧以内的短断点会被合并。
- `OppositeVehicleRunningRedLight`：U-E6 必须在 R4 路口窗口内由冲突车辆、近距离对象或
  ego 让行响应触发；同一冲突车过程内 5 帧以内的短断点会被合并。
- `CrossJunctionDefectTrafficLight`：U-E7 是主事件；有冲突车辆时 U-E6 作为同帧 secondary
  `events` 保留，避免 U-E7/U-E6 primary 来回切换。
- 行人/自行车：`ParkingCrossingPedestrian` 可在进入路口前由近距离行人/刹车触发；
  `VehicleTurningRoute*` 只在转弯/驶出后对象进入交互范围时触发，不能沿用同一个早触发模板。
- `quick_start.py annotate-rs` 虽沿用历史命令名，但当前正式输出是 RS + EVENT：
  逐帧写 `primary_road_structure/frame_rs_annotation` 与
  `primary_event/frame_event_annotation`，全局摘要写 `road_structure_labels/event_labels`。
  Web 页面左侧展示当前 scenario 候选 RS/EVENT 的中文含义，右侧分开展示本帧主 RS
  与本帧主 EVENT，避免把候选全集误读为单帧标签。

## 5. ROAD_STRUCTURE 到 EVENT 候选裁剪

这一步用于降低 Qwen 判断难度。候选表不是硬互斥表，数据结构仍支持多选；
但当前标注默认单主事件，只有 R4/R5 路口安全关键 span 允许双触发。

| ROAD_STRUCTURE | 常规事件候选 | 突发事件候选 |
|---|---|---|
| R1 常规道路 / 同向可行驶道路 | R-E1, R-E2 | U-E1, U-E2, U-E3, U-E4 |
| R2 双向单车道 / 借对向车道道路 | R-E1, R-E2 | U-E2, U-E5 |
| R3 高速合流 / 匝道 / 分流 / 驶出决策结构 | R-E1, R-E2, R-E3 | 仅按 scenario 白名单加入，默认无非常规 |
| R4 信号灯路口 | R-E4 | U-E4, U-E6, U-E8 |
| R5 无信号灯 / 信号灯失效路口 | R-E5 | U-E4, U-E6, U-E7, U-E8, U-E5 |
| R6 路边停车 / 停车占道道路 | R-E1, R-E2 | U-E2, U-E3, U-E4 |

多选示例：

```text
R1 + [U-E2]
常规车道保持被静态障碍占道打断，非路口段只保留安全关键主事件。

R4 + [R-E4, U-E6]
绿灯路口通行中遇到违规车辆冲突。

R6 + [U-E4]
路边停车区域正常前进中遇到行人横穿，非路口段只保留横穿主事件。
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

Qwen 只在当前 ROAD_STRUCTURE 对应的事件候选中选择。默认输出单主事件；
只有 R4/R5 路口中确有可见安全关键 span 时，才允许常规路口事件与 U-E 双触发。

建议输出格式：

```text
ROAD_STRUCTURE: R1
EVENTS: U-E2
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
EVENTS: U-E2
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

每帧的 EVENTS 先由“常规背景事件 + 命中的突发事件 span”合成，再按门控裁剪：

```text
EVENTS = 常规背景事件 + 命中的突发事件 span
```

例子：

- R1 下没有 span：`EVENTS=[R-E1]`
- R1 下命中 U-E2 span：`EVENTS=[U-E2]`
- R4 下命中 U-E6 span：`EVENTS=[R-E4, U-E6]`
- R5 下命中 U-E7 span：`EVENTS=[R-E5, U-E7]`
- CrossJunctionDefectTrafficLight 有冲突车：`primary_event=U-E7`，`EVENTS=[R-E5, U-E7, U-E6]`

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
| R1 | 默认桶；同向障碍、急刹、动态对象、control loss、roundabout 本身不改变 RS | brake/accel/vehicle_hazard/walker_hazard 不参与 RS 升级 |
| R2 | 核心借道/障碍帧需要 trigger/active + opposite lane / 同向车道不足 / meta obstruction；meta 可用时允许 XML trigger 极近或 trigger-close + XML 场景障碍近距离召回短核心 R2；非核心 TwoWays road-layout 只能作为弱候选 | TwoWays 名称不代表全程 R2；绕过静态/动态障碍后重新按 XODR/meta 判 R1/R4；灯态/路口主导时 R4/R5 primary；缺 meta 或 XML 的 run 直接 data_missing_skip |
| R3 | 高速/快速路/合流/驶出 scenario prior；明确高速/merge 场景不开放 R1，非路口默认 R3；merge/exit 窗口和 actor-flow 提供更强证据 | `PriorityAtJunction`、部分 `HardBreakRoute`、Interurban rural/junction 不能只因 Town12/13 判 R3 |
| R4 | 有效 `traffic_light_state` / `light_hazard`，或同源受控 junction/controller 支撑，且 XODR 未判为 roundabout；有效 meta 灯态可动态开放 R4 候选；noScenarios 可用“有效灯态 + bbox 灯 + junction window”窄召回稳定灯控 approach | `CrossJunctionDefectTrafficLight` 由 R5 override；`NonSignalizedJunctionLeftTurn*` 是 strict no-R4；阻塞/违规只是 EVENT；roundabout 强制回 R1；同向障碍/默认场景有 meta/bbox 灯态但缺 strong context 时必须降回 R1，避免 R4 删除 U-E2/U-E3；只有弱静态 signal/bbox traffic_light 且无 meta 灯态时不能自动 R4，尤其同帧已有 STOP/yield 证据时必须优先 R5 |
| R5 | nonsignalized/priority/defect prior，或已开放 R5 的 default/noScenarios/ControlLoss/ParkingCutIn/障碍族场景 + route/trigger/junction/STOP/yield 同源证据，且 XODR 未判为 roundabout | 连续有效灯态且非 defect scenario 时不 high R5；roundabout 强制回 R1；雾天/弯道 blind R5 不足以单独修改 collector；障碍族 R5 只表达 route 前后 STOP/无灯 regular，不表达障碍核心 |
| R6 | Parking* / parking 子型 prior + parking trigger/active 窗口 + parking/shoulder/curbside 或停车汇入证据 | `ParkedObstacle` 不是 R6；灯控路口主导时 R4 primary |

仲裁优先级默认：

```text
R4/R5 > R3 > R2/R6 > R1
```

例外：

- `CrossJunctionDefectTrafficLight`：R5 覆盖 R4，并写 `defect_signal_overrides_R4`。
- `VehicleOpensDoorTwoWays`：R2/R6 可同时成立，primary 取决于是否必须占用/等待对向车道。
- `noScenarios`：没有 scenario prior 时默认 R1；强灯态 + 同源受控 junction 升级 R4；
  稳定 meta 灯态 + bbox 灯 + junction window 可作为 R4 approach 窄召回；STOP/无灯控制证据可给 R5。
- `CrossJunctionDefectTrafficLight`：RGB 看到红绿灯硬件不代表 R4；该场景的控制规则失效语义由 R5+U-E7 表达。
  审计比较时 `blind_R4_label_R5` 应作为 defect R5 解释，不算规则错配。
- R3/高速/合流类：导流线、让行牌、宽路面和分流/合流曲线容易被 RGB blind detector 误判 R5。
  若 collector 当前帧有 `rule_kind=highway_merge/highway_exit`、`r3_*` 规则或 route 级 highway RGB bucket，
  `blind_R5_label_R3` 应作为审计器误报解释；不要为了降低该项把 Enter/Merger/HighwayExit 改成 R5。
- 障碍族：Accident / ConstructionObstacle / ParkedObstacle / HazardAtSideLane 及 TwoWays 版本、
  VehicleOpensDoorTwoWays 都可在 route 前后真实 STOP/无灯路口输出 R5/R-E5；但障碍核心仍按
  U-E2/R2/R-E2 处理，不能因开了 R5 候选就把核心绕障改成路口 regular。
- `roundabout`：XODR 若显示局部为 roundabout，则 R4/R5 junction 分支全部失效，primary 回 R1；
  页面必须展示 `map_is_roundabout=true`，用于区分环岛和十字/丁字路口。
- 动态 R4 候选：除 `CrossJunctionDefectTrafficLight`、`NonSignalizedJunctionLeftTurn*` 和 RGB 已确认无稳定灯控的场景外，
  只要单帧有有效 `traffic_light_state` 或 `light_hazard`，就临时把 R4 加入候选池；但同向障碍、默认和 noScenarios
  缺少 strong control context 时不能以 R4 做 primary，只写证据/review 并保持 R1。bbox/static signal 只是辅助证据；
  在无有效 meta 灯态且同帧存在 STOP/yield/无灯控制证据时，不允许靠 bbox/static signal 把 R5 抬成 R4。
  这修复了 Accident / Hazard / Parking / Priority 等场景“明明有灯态却被候选池或弱分数压回 R1”的问题。
- 反向恢复必须更谨慎：若保守降级导致真灯控路口被压成 R1，只允许 route 级
  `r4_context_recovery` 恢复连续稳定片段。恢复条件是灯态/bbox traffic_light 连续不少于 4 帧，
  且有 `strong_control_context`、`close_trigger_for_junction` 或 bbox junction hint。弱
  `near_junction`、宽 `junction_window`、远处/瞬时灯光只能 review，不能自动恢复 R4。

时序稳定统一后处理：

- 所有 RS 都要经过 route 级短片段去抖，不只 R1/R4。
- 4Hz 数据下，R2/R3/R4/R5/R6 的候选片段至少连续 4 帧（约 1 秒）才作为真实结构切换；
  短于 4 帧的片段并回前后邻居，若前后标签相同则直接填平。
- R1 作为默认桶至少连续 2 帧；短 R1 缝隙夹在同一特殊 RS 中间时，也会被填回该特殊 RS。
- 去抖后的帧必须写 `evidence.temporal_smoothing` 与 `temporal_smoothing_applied`，
  让人工复核能看到原始标签和替换原因。

### 9.4 规则族结论

- `same_direction_obstacle`：`Accident`、`ConstructionObstacle`、`ParkedObstacle`。
  静态同向障碍是 EVENT 证据，不把整段升级成 R2/R6；只在受控路口窗口进入 R4。
- `twoways_obstacle` / `invading_turn` / `vehicle_opens_door_twoways`：
  R2 只覆盖必须借/等对向车道的核心片段。核心帧需要 XML trigger、XODR 对向/双向单车道拓扑、
  meta active、近距离障碍、stuck、vehicle_hazard 或 lane-change 证据共同支撑。
  对 `*_TwoWays`，如果 meta 核心证据缺失但 XML trigger 已极近，或 trigger-close 且 XML 场景障碍近距离成立，
  允许短核心 R2 召回，并写 `r2_xml_trigger_core_confirmed`。
  `two_way_layout_prior` 只能保留弱 R2 候选，不能把障碍前后或绕过障碍后的普通双向道路维持为 primary R2。
  过最近障碍点后，若连续约 0.75s 远离超过 2m 且没有 `stuck` / `vehicle_hazard`，
  route 级 `twoways_core_span_clipping` 必须把后段按证据回 R1/R4；若同一条 TwoWays route
  出现多个 R2 片段，`twoways_longest_r2_filter` 只保留最长连续 R2 核心段，其它 R2 短扰动回 R1/R4。
  当前
  `AccidentTwoWays/Town01` 的代码参考边界是
  `R1 f0-f55 -> R2 f56-f108 -> R1 f109-f149 -> R4 f150-end`。
  后段回 R4 也必须有同源路口/灯控证据；静态 XODR 远处 signal 或无结构 junction hint
  与 RGB 普通山路冲突时，后段回 R1。
- `highway_merge`：`EnterActorFlow*`、`HighwayExit`、`MergerIntoSlowTrafficV2`
  已由全量逐帧 RGB 确认是稳定无灯控高速/快速路背景，候选池删除 R1/R4；非路口段默认 R3。
  `HighwayCutIn` 与 `MergerIntoSlowTraffic` 删除 R1 但保留少量 R4 子集；R4 只能由逐帧
  RGB/meta/bbox 灯控同源证据触发，merge/exit/actor-flow 窗口只提高 R3 置信度。
- 混合场景 route 分桶：`HardBreakRoute`、`InterurbanActorFlow`、`InterurbanAdvancedActorFlow`、
  `StaticCutIn`、`ParkingCutIn` 不能只按 scenario 或 Town12/13 判 RS。必须先用 RGB sheet
  把 route 分为高速/快速路桶、普通城市/乡村桶、停车/路边桶等；高速桶候选收敛为 R3/R4，
  非高速桶保留 R1。当前已完成逐 id 均匀 5 帧 RGB 复核：HardBreakRoute 97 个 route 中 16 个进高速桶；
  StaticCutIn 100 个 route 中 44 个进高速桶；InterurbanActorFlow 91 个、InterurbanAdvancedActorFlow 78 个、
  ParkingCutIn 99 个未发现高速桶。`Town12_Rep0_258_0_route0_01_08_09_35_42`
  这类乡村普通路必须保持 R1；精确高速 id 清单以 `collector.py` 的
  `MIXED_SCENARIO_HIGHWAY_ROUTE_IDS` 为准。
- `interurban`：`InterurbanActorFlow` 保留 R1/R3/R5，已按全量 RGB 删除 R4；Town12/13 是提示但不是充分条件，乡村道路、STOP/priority/junction 仍需按 RGB/XODR 分段。
- `blocked_intersection`：`BlockedIntersection` 的阻塞只是 EVENT；RS 由控制源决定。
  灯态/信号灯同源证据成立时进入 R4；STOP/yield/priority/无灯路口证据成立时进入 R5；
  两类控制源都缺失时回 R1 + RGB review。
- `nonsignalized_junction`：无灯/STOP/yield/priority 路口进入 R5；只有全量 RGB 已确认存在灯控子集的
  `NonSignalizedJunctionRightTurn`、`OppositeVehicleTakingPriority`、`PriorityAtJunction` 才允许有效灯态或强灯控同源证据打开 R4。
  `NonSignalizedJunctionLeftTurn` 与 `NonSignalizedJunctionLeftTurnEnterFlow` 是 strict no-R4，
  bbox/static signal 弱提示不能覆盖 STOP/yield/无灯控制源。
- `signalized_junction`：灯态有效、受控 junction 或 controller/traffic light 近邻成立时进入 R4；
  `OppositeVehicleRunningRedLight` 的违规只是 EVENT，不改成 R5。
  若 primary R4 没有有效 `traffic_light_state`，必须写
  `signalized_r4_without_meta_tl_requires_rgb_confirmation`，人工逐帧确认 RGB 里仍有 stopline /
  crosswalk / cross traffic / blocked pocket 等路口证据；XODR/XML 只能辅助低能见度判断。
  若只有 static signal 近邻 + 灯态而缺少 `is_junction`/XODR junction，strong context 距离阈值为
  25m；25-35m 只保留弱 R4 候选，避免在雾中普通路段过早覆盖 R1。
- `nonsignalized_junction` / `defect_junction`：无有效灯态、stop/yield/priority 或灯故障机制成立时进入 R5；
  `CrossJunctionDefectTrafficLight` 强制 R5 覆盖 R4。
- `parking` / `parking_exit` / `static_cutin`：R6 只给 parking/shoulder/curb/parking-exit 结构窗口；
  停车相关 scenario 在信号灯路口段仍优先 R4/R5。
- `default_meta_map` / `noscenario`：默认 R1；ControlLoss、HardBreak、DynamicObjectCrossing、
  HazardAtSideLane 等行为/突发事件本身不改变 RS。稳定灯控证据可临时进入 R4；
  只有场景候选显式开放 R5 时，STOP/无灯路口强证据才可进入 R5。
  2026-07-05 全量 RGB 盲审后，`DynamicObjectCrossing` 恢复 R5 候选：
  XODR 拓扑不可信时，连续 bbox/meta STOP + RGB 路口证据仍可召回 R5/R-E5。
  同日 RGB top mismatch 复核后，`HardBreakRoute` 与 `ParkingCrossingPedestrian`
  也恢复 R5 候选；前者从全漏 R5 变为少量边界漏检，后者总 mismatch 从 955 降到 220 帧。
  这两个场景的剩余 top case 多为雾天、停车/行人交互、路面箭头/线造成的 blind 审计宽判，
  不应为了追平 blind detector 再放宽到“看到一两帧线/远灯就 R5”。
  `StaticCutIn` 同日恢复 R5/R-E5：top RGB case 有连续 stopline/横向路口形态，
  子集 `blind_R5_label_R1` 从 321 降到 27 帧；剩余 activity mismatch 多为邻道/前方车辆普通运动，
  不据此扩大 U-E3。
  `ControlLoss` 同日加入可见控制源恢复规则：`meta traffic_light_state + bbox traffic_light`
  可在稀疏 XML window 外恢复 R4，`bbox/meta STOP/yield` 可恢复 R5；子集总 mismatch 从 2441 降到 64。
  RGB 盲审脚本也同步收紧 TwoWays 解释口径：R2/U-E2 核心段不会因车尾颜色、中心线或弱 stopline
  被误计为 R5 mismatch；完整 `U-E2 -> R-E2` 后回到 R-E1 的普通车流 motion 不再计作漏事件。

初始阈值只作为调研起点：`junction_pre_m=40~60`、`junction_post_m=20~40`；
运行时路口窗口会轻量收缩为 `0.85 * junction_pre/post`（pre 最小 30m、post 最小 15m），
`dist_to_junction_near=45m`，strong junction 上限 30m，static signal near 上限 45m，
让十字路口/丁字路口只覆盖接近、进入、刚离开的局部片段。
`BlockedIntersection` 的十字路口窗口在该基础前额外压缩 20%，基准为
`junction_pre_m=48`、`junction_post_m=20`；阻塞是 EVENT，不应把 R4/R5 范围拖长。
窗口内有灯控同源证据才 R4；STOP/yield/无灯控制源优先 R5，不能再按场景名默认 R4。
静态 XODR 的 junction/signal 不能单独作为 R4 strong context：若 `map_junction_id=-1`
或 `junction_connection_count=0`，只能写弱证据/review；必须由有效 meta 灯态、
stop/light hazard、结构化 junction/controller 或 RGB 可见路口/信号灯同源后才升 R4。
`two_way_min_pre_m=50~80`（`*_TwoWays` 入口窗口较前一版统一前扩约 5m，核心证据和后段裁剪不变）、
`merge_pre_m=30~50`、`merge_post_m=40~50`、
`parking_pre_m=20~35`、`parking_post_m=50~60`。正式阈值必须在 scenario 调研包中补齐
provenance 后才能进入 high-confidence runtime。

### 9.5 Smoke test 修正

已有小样本 smoke test 证明 `rs_research.py` 和 `ScenarioCollector` 能跑通，但暴露了需要收紧的规则：

- 需要 per-frame XODR 时必须用能 `import carla` 的 Python；否则 XODR 证据为空，
  R2/R3/R6 只能 medium/low + review。
- R2 primary 不能只靠 scenario 名称或 layout-prior；核心借道/障碍帧若没有
  `has_opposite_driving_lane=true` 且 `same_direction_lane_count<=1`，需要近距离障碍、
  stuck、vehicle_hazard、lane-change 证据，或 XML trigger 极近 / trigger-close + XML 场景障碍近距离召回。
  另有一类 `two_way_layout_prior`
  是非核心道路布局弱候选：它不能把清晰 TwoWays road span 保持为 primary R2；
  绕过静态/动态障碍后必须重新按 XODR/meta 判 R1/R4，不能默认一直 R1，也不能默认一直 R2。
  `twoways_core_span_clipping` 会检查整条 route 的最近障碍距离曲线，过最近点后持续远离且无
  stuck / vehicle_hazard 时裁掉 R2 尾巴。
- R3 对明确高速/merge scenario 是默认道路空间；`EnterActorFlow*`、`HighwayExit`、
  `MergerIntoSlowTrafficV2` 不再开放 R1/R4。`HighwayCutIn`、`MergerIntoSlowTraffic`
  不开放 R1，但保留少量 R4 子集，且只能由逐帧灯控同源证据触发。对 HardBreak/Interurban/Priority
  仍必须结合 RGB/XODR，不能只按 Town12/13，也不能只靠 meta 灯态。
- `InterurbanActorFlow` 2026-07-04 全量逐帧 RGB 审计未见稳定信号灯路口，候选删除 R4。
  `InvadingTurn` 在 2026-07-05 RGB 盲审复核中发现 Town12 稳定信号灯子集
  （例如 `Town12_Rep0_1150_0_route0_01_09_20_35_02`），因此恢复 R4 候选；
  无灯/STOP 十字路口窗口仍输出 R5，投影误差高时保留 review 但不能把 STOP 证据压回 R1。
  `InvadingTurn` 的 static XODR 多数帧 `map_junction_id=-1` / `xodr_topology_trusted=false`，
  所以 R5 召回应优先参考 RGB STOP/无灯路口、meta active/is_junction/stop 和 XML trigger。
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
| `InterurbanActorFlow` | 122 帧：R3=102、R5=4、R1=16；review=109 | 旧 smoke 暴露 projection error 会把无灯/STOP 十字路口压回 R1；2026-07-04 全量逐帧 RGB 审计后删除 R4，STOP/active close-trigger 窗口允许 R5 主标签。 |
| `InterurbanAdvancedActorFlow` | 120 帧：R5=80、R1=40；review=111 | junction primary 方向可用；R3 只应在明确 topology 时开放。 |
| `NonSignalizedJunctionLeftTurnEnterFlow` | 59 帧全 R5；review=37 | `enter_flow_not_r3` 正确；R5 窗口需要 meta junction/stop/yield + XODR 同源支撑。 |
| `T_Junction` | 旧 smoke 为 163 帧全 R4；全量 RGB 审计更新为 R1/R4/R5 共有 | 旧结果已过时；灯控 T 走 R4/R-E4，无灯/STOP/yield T 走 R5/R-E5，不能只因“没看见灯”自动转 R5。 |
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
  或 RGB/bbox/动作主因证明对向参与。否则只能作为弱候选；障碍前和绕过障碍后回 R1/R4。
- R3：明确高速/merge/exit/enter-flow 场景直接以 R3 作为非路口默认空间；actor-flow / merge / exit
  window、merge/split/ramp/lane-count-change 或目标出口车道证据用于提高置信度和边界定位。
  `xodr_available=true` 不等于 R3 topology，但坏 XODR 不能把高速 RGB 压回 R1。
- `MergerIntoSlowTraffic*` 的 XML `start_actor_flow/end_actor_flow` 是合流慢车流证据；
  当 RGB 显示明显 merge 口、而 route/XODR 投影误差导致 topology 不可信时，可用 actor-flow
  强近邻或 trigger 距离作为 R3 fallback，并在 evidence 中保留投影误差 review。
  active scenario 不能单独延长“核心合流事件窗口”；逐帧 RGB 显示 merge 完成后，
  `current_active_scenario_type` 即使仍为 MergerIntoSlowTraffic，主线高速/快速路背景仍按 R3，
  不能退回 R1。
- R6：召回可用 Parking* prior + parking window；确认必须有 parking/shoulder/curbside、
  parking->Driving 转换或路边静态车列。普通 shoulder hint 不够。
- R4/R5：召回可用 junction/trigger window；确认必须看灯态、light_hazard、stop/yield/controller
  和 route 同源。`is_junction=false` 不代表不是 stopline approach。
  真实十字路口/T 形路口经过必须持续一段时间；4Hz 下 R4/R5 少于 4 帧时，即使有瞬时
  traffic_light、bbox traffic_light 或 XODR junction hint，也按全局短片段去抖并回邻近稳定 RS。
- `light_hazard` 不是独立 ROAD_STRUCTURE 证据。图像复核发现同向事故/施工/急刹/拥堵路段也可能出现
  hazard 相关 meta 信号；在 `same_direction_obstacle` / `default_meta_map` 这类场景里，
  只有 `light_hazard` 同时具备 meta junction 或 stop hazard 等强控制上下文时才升 R4，
  否则保持 R1，把具体风险交给 EVENT。
- XML route projection error >5m 时，不能用 route_s 做边界；应改用 meta 时序和 XODR map waypoint
  吸附，并强制 `review_required=true`。
- XML 匹配必须 town-aware：先按 `(Scenario,Town,route_key)` 和别名精确匹配，再按
  `(Scenario,Town,route_num)` 兜底。跨 town 纯数字冲突不能随便选一个 XML；例如
  `Town07_route_001456` 不得匹配到 `Town12_route_1456_0` 后再用错误 XODR 判 R4。
- 缺少 topology confirmation 的 R6/R2 弱候选必须低于稳定 R1；R3 只有在明确高速/merge
  场景中可作为默认道路空间。`ParkingCutIn`、`StaticCutIn` 等 RGB sheet 显示，事件名和
  trigger window 很容易覆盖到普通直行/普通切入片段；没有 parking/opposite-lane 或可见
  road-structure 证据时，不应制造低置信特殊 RS 或大面积候选分差 review。

下一步代码完善优先级：

1. `rs_research.py` 增加 boundary frame 采样和 `projection_metrics.json`，把 projection error
   系统性记录到调研包。
2. `collector.py` 将当前特种 RS 的打分拆成 recall score 和 confirmation score。
3. `collector.py` 对 `r2_scenario_trigger_medium`、`r3_lacks_xodr_merge_split_confirmation`、
   `r6_lacks_xodr_parking_or_shoulder_confirmation` 强制 review，并限制分数上限。
4. route 级最短持续帧已落地：R2/R3/R4/R5/R6 少于 4 帧、R1 少于 2 帧的孤立片段会合并到邻近稳定段；
   但 smoothing 不允许把只有弱候选的 R1 帧提升成特殊 RS。TwoWays 弱 layout-prior 帧必须自身命中
   `r2_core_obstruction_confirmed`，才可被邻近 R2 核心 span 吸收；TwoWays route 还会在 smoothing 前做
   `twoways_core_span_clipping`，把已绕过障碍的 R2 后尾裁回 R1/R4；随后
   `twoways_longest_r2_filter` 只保留最长连续 R2 段，清掉非最长 R2 碎片。后续若仍有边界抖动，再增加
   transition margin / hysteresis。

当前 RGB-first 全帧复核基线：

- 覆盖 43 个 scenario、204 条 scenario-town route、24387 帧；每个 town 1 条 route，全帧标注并生成
  `all_frames_*.jpg`。`candidate_anomalies=15788` 是逐帧看图索引，不是错帧数。
- 全部历史迭代可以归结为六类错配根因：
  1. XML route / trigger 投影误差高，却仍把 route_s 当 hard boundary，导致普通路段被过早升为 R4/R5/R3。
  2. 静态 XODR signal/opposite/parking/merge/junction hint 与 RGB 不同源，弱 topology 被当成 high confirmation。
  3. scenario 名称、active scenario、事件距离被当成 ROAD_STRUCTURE 真值，导致 EVENT 覆盖 RS。
  4. 坏 XODR 被当作否定证据，导致明显高速/merge 或 TwoWays 核心借道/绕障片段被压回 R1。
  5. 低能见度、夜间、雾天只看 summary/confidence，没有逐帧确认 RGB，容易把 review index 当错帧或把高置信当正确。
  6. 转弯/行人/事故/施工/急刹/切入/开门等事件和道路控制源混在一起，导致 R4/R5/R6/R3 过宽。
- 已回灌的通用修正：
  无有效 `traffic_light_state` 的 signalized R4 必须写 RGB confirmation review；
  static-signal-only strong control context 收紧到 25m；
  Interurban no-light R5 在 `route_projection_error_high` 或弱可见控制证据下必须降为弱候选；
  VehicleTurningRoute* 的无灯 R5 在 `route_projection_error_high` 下必须有 stop / is_junction /
  非静态可信 XODR 近路口证据；
  `MergerIntoSlowTraffic*` 允许 XML actor-flow/trigger 强近邻在 XODR 投影失败时召回 R3；
  active scenario 不能单独延长核心合流窗口，但高速主线仍保持 R3；
  TwoWays road-layout 与核心借道/障碍分层，R2 只覆盖必须借/等对向的核心 span，障碍前后回 R1/R4。
- 逐场景定位、通病抽象复用是固定调参原则：先从每个场景的 RGB sheet 找具体错配，再判断是场景私有阈值、
  全局证据门控、XML/XODR 投影问题，还是低能见度证据不足。多个场景共同出现的稳定普通路段低置信问题已抽象为
  `r1_stable_no_special_structure_confirmed`：没有任何特殊 RS 达到有效候选阈值时，R1 是稳定普通道路结构，
  不是低置信兜底。

RGB-first full-frame review protocol:

- 每个 scenario 的每个 town 抽 1 条 readable route，生成全量全帧 RGB contact sheet。
- 人工复核必须从第一帧看到最后一帧，不允许只看 summary、confidence、标签分布或前段代表帧。
- 稳定 R1 也必须逐帧看，避免漏掉后段 merge/parking/junction/TwoWays 结构。
- 判定顺序固定为 RGB 可见道路结构优先；低能见度或遮挡时才用 XODR/XML/meta 补证。
- 异常记录必须写清帧范围、RGB 观察、标签冲突、触发的 evidence/rules 和原因归类：
  规则思路问题、阈值参数问题、XML/XODR 投影问题或低能见度证据不足。

Blind RGB audit protocol:

- 2026-07-05 新增 `rgb_blind_rs_event_audit.py` 作为固定入口。该工具先逐帧读取
  `rgb/*.jpg` 生成 blind sheet 和 `manual_blind_answer_template.csv`，此阶段不读取当前
  `primary_road_structure/primary_event` 作为判断依据；之后才调用 collector 生成当前标签并写
  compare sheet / mismatch CSV。
- 真正的人工结论必须先写在 `manual_blind_answer_template.csv` 的
  `manual_rs_spans/manual_event_spans` 中，格式建议为 `R1:0-23;U-E2:24-39;R-E2:40-67`。
  填完 blind answer 后再打开 `route_blind_rs_event_audit.csv` /
  `route_blind_rs_event_details.json` 对账，避免被当前标签“带答案”。
- 自动 blind guess 只做风险排序，不当作真值。R4 可用红绿灯颜色块做高置信提示；
  R5/无灯路口在纯 RGB CV 下容易被普通车道线误报，因此默认只保留
  `rgb_stopline_candidate/rgb_turn_marking_candidate` 证据，不自动把整段升为 R5。
- 全量闭环顺序固定为：
  `rgb_blind_rs_event_audit.py --write-sheets` 全量/按 town 生成盲审包 →
  逐 route 填 manual blind spans →
  对比当前 labels 找 `missing_core_event`、`wrong_R4/R5`、`R-E1 eats U/R-E2` 等错因 →
  修改 `ROAD_EVENT_*` 思路和 `collector.py` →
  重跑同一批 audit 与 `quick_start.py annotate-rs` smoke。

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
- `StaticCutIn` 当前保留 R1/R3/R4/R6 混合候选，必须按每个 run 的 map/RGB 拆分；若筛成高速/merge 桶则
  候选收敛为 R3/R4；若筛成停车/路边桶则以 R6/R4 为主，普通切入桶保留 R1。
- `T_Junction` 按全量 RGB 审计改为 R1/R4/R5 共有候选；灯控 T 走 R4/R-E4，无灯/STOP/yield T
  走 R5/R-E5。缺灯态不能自动转 R5，必须有 stop/yield/priority 或 RGB 无灯控制源证据。
- `PedestrianCrossing`、`VehicleTurningRoute*` 保留 R4/R5 候选；最终按灯态、signal/controller、
  stop/yield 和 junction 证据决定。
- `OppositeVehicleTakingPriority` 以全量 RGB 无灯 priority/R5 为主，但少量灯控子集保留 R4/R-E4；
  U-E7 只在 primary RS 为 R5 的无灯/路权失效帧触发，不能覆盖正常灯控 R4 帧。
- `PriorityAtJunction` 是混合场景：全量 RGB 同时看到真实灯控城市十字路口和无灯/让行段，候选保持 R1/R4/R5。
- `InterurbanAdvancedActorFlow` 当前不默认 R3，且全量 RGB 未见稳定 R4；只有 RGB route 高速桶或 XODR merge/highway/ramp 证据成立时才打开 R3。
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
- 2026-07-04 已用 `annotate-rs --scenario all --samples-per-town 5 --max-frames-per-route 120`
  覆盖 43 个 scenario、993 条 route、91320 帧做 EVENT 回归；常规场景未漏出 U-E，
  异常场景命中各自白名单 U-E。高切换场景又用
  `InvadingTurn,CrossJunctionDefectTrafficLight,Accident,ConstructionObstacle,HazardAtSideLane`
  共 130 条 route、12102 帧 targeted smoke 回归，确认缺陷路口 U-E7 主事件稳定、U-E6 为
  secondary，动态切入不再靠普通急减速误触发。
- 2026-07-04 最终收口又跑全 43 场景轻量 smoke
  `annotate-rs --scenario all --max-routes 1 --max-frames-per-route 10`；一致性检查确认
  `primary_event` 均在 scenario 白名单内，`frame_event_annotation.label` 与
  `primary_event` 一致，`frame_rs_annotation.label` 与 `primary_road_structure` 一致。

这套结构更贴近你的原始意图：先判断当前处于哪套驾驶决策规则空间，再在该空间下判断可能发生的事件。
