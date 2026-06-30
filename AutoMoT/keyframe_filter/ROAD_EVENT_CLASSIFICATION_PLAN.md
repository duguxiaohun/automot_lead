# ROAD / EVENT 分类重标注方案

本文整理 `classifier_logic.txt` 中的人工调研结论，并结合旧版
`rule_based_keyframe_filter.py` 的生成方式，形成一套更清晰的两层语义体系：

- 上层：`ROAD_STRUCTURE`，表示当前帧处于哪一种驾驶决策规则空间。
- 下层：`EVENT`，表示当前帧在这个规则空间中触发了哪些常规或突发驾驶事件。

核心口径：`ROAD_STRUCTURE` 不是纯物理几何分类，而是驾驶决策规则空间分类。不同道路结构之所以要分开，
是因为它们下层可触发的事件、通行优先级和动作约束不同。

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

## 9. 待核实问题

需要后续结合视频或数据核实：

- HazardAtSideLane 到底主要是 R1 同向多车道避让，还是存在 R2 双向单车道片段。
- StaticCutIn 是否混合 R1、R3、R6，需要按视频位置拆分。
- T_Junction、PedestrianCrossing、VehicleTurningRoute 系列是否有灯，决定 R4/R5。
- PriorityAtJunction 是否稳定归 R5。
- InterurbanAdvancedActorFlow 是否存在前置 R1 变道片段。

## 10. 当前结论

最终推荐口径：

- R1-R6 保持 6 类，不继续细分。
- R1 明确作为默认 / 其它全部规则空间。
- R2 不增加子属性，主动借道和被动让行交给事件区分。
- R-E6 取消，不作为独立事件。
- EVENTS 支持多选，用常规事件描述背景任务，用突发事件描述安全关键打断。
- 旧 keyframe 逻辑只作为 span 提议器和人工抽检入口，不作为最终帧级 STATUS/SUBGOAL 真值。

这套结构更贴近你的原始意图：先判断当前处于哪套驾驶决策规则空间，再在该空间下判断可能发生的事件。
