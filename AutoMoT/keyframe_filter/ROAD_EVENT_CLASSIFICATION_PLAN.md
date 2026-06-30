# ROAD / EVENT 语义重标注方案

本文基于 `classifier_logic.txt` 的人工调研结论，以及旧版
`rule_based_keyframe_filter.py` / `keyframes_all_scenarios.json` 的生成逻辑，整理新的
道路结构 + 事件语义空间设计。目标不是立刻替换实现，而是先明确分类边界、监督信号可获得性、
迁移路径和待验证问题。

## 1. 总体判断

你的方向是合理的：旧方案按 CARLA scenario 固定抽 `initial + 3 middle + final`，
只能得到粗关键帧，不适合表达同一条 route 内不断切换的道路结构和事件状态。

新的思路应该拆成两个近似正交的轴：

- `ROAD_STRUCTURE`：当前画面/局部道路属于什么通行规则空间。
- `EVENT`：在这个道路空间下，当前帧正在处理什么驾驶事件。

这样能解释 Accident 例子：

```text
R1 常规道路
  -> R-E1 跟车 / 车道保持
  -> U-E2 静态障碍物占道
  -> 被迫绕行
  -> R-E2 或 R-E1 回归目标车道 / 车道保持
  -> 如果进入路口，ROAD_STRUCTURE 切到 R4/R5
  -> 使用路口事件集合继续判断
```

关键点是：`scenario` 只能作为候选集先验，不能再作为唯一标签。一个 scenario 可以包含多个
道路结构，一个道路结构也会出现在多个 scenario 中。

## 2. 旧 keyframe 逻辑的可复用与噪声源

旧脚本的核心行为：

- 每个 scenario 在 `SCENARIO_CONFIG` 中绑定一个距离字段、阈值和三个 middle event。
- 主要信号来自 `metas/*.pkl` 的 `dist_to_*`、`speed`、`accel_x`、`brake`。
- 缺 meta 信号时退到 bbox 最近目标距离，再不行退到 RGB 文件大小变化峰值。
- 输出 JSON 每条 run 只有 `initial`、3 个 `middle`、`final`，并强制 middle 顺序递增。

可复用部分：

- `dist_to_accident_site` / `dist_to_construction_site` / `dist_to_parked_obstacle`
  适合作为 U-E2 静态障碍物占道的弱 span 信号。
- `dist_to_pedestrian` / `dist_to_biker` 适合作为 U-E4 行人/自行车横穿的弱 span 信号。
- `dist_to_cutin_vehicle` 适合作为 U-E3 动态车辆切入 / U-E5 对向侵占的弱 span 信号。
- `speed` / `accel_x` / `brake` 适合作为 U-E1 急刹、等待、恢复启动的辅助边界。
- `signal_source` 和 `confidence` 思路可以保留，用来标记标签来源可靠性。

主要噪声源：

- 固定每个 scenario 只有 3 个 middle event，会丢掉长 route 中多次结构切换。
- scenario 名称把道路结构、对象类型、任务意图混在一起，导致标签不对齐。
- RGB fallback 基于文件大小峰值，语义非常弱，只能用于人工抽检，不适合直接监督。
- bbox fallback 只能知道附近有对象，不能可靠区分“对象是否真的占据自车未来路径”。
- `enforce_event_order` 会把帧位置强行拉开，适合展示，不适合精确 span 标注。

结论：旧逻辑不要直接生成最终 STATUS/SUBGOAL；更适合当作候选 span 提议器和人工/Qwen 复核的先验。

## 3. 道路结构划分审阅

你当前的 R1-R6 覆盖面基本够用，但建议明确它们不是完全物理几何分类，而是“驾驶决策规则空间”。
这样才能接受同一个物理道路在不同片段切换标签。

### R1. 常规道路 / 同向可行驶道路

保留。它是最重要的背景类，覆盖普通跟车、车道保持、同向变道、同向绕障前后。

建议收紧定义：

- R1 表示“当前局部通行规则仍是同向道路规则”。
- 行人/自行车、切入、急刹、静态障碍都不改变 R1 本身，只改变 EVENT。
- 如果画面进入明确路口控制区，应切到 R4/R5，而不是继续 R1。

风险：R1 现在有点大，容易变成“其它全部”。后续 prompt 里要把 R1 写成默认背景类，
并要求看到明确路口/合流/停车区证据才离开 R1。

### R2. 双向单车道 / 借对向车道道路

保留。这个类很有必要，因为“主动借对向车道绕障”与同向变道绕障的策略完全不同。

建议增加一个子属性而不是拆类：

- `R2_ACTIVE_BORROW`：自车主动借对向车道绕过障碍。
- `R2_PASSIVE_YIELD`：对向车侵占自车道，自车被动让行，例如 InvadingTurn。

这样 R2 仍是道路结构，U-E2/U-E5 决定行为方向。

### R3. 高速 / 匝道 / 合流道路

保留。它比普通 R1 更依赖侧后方间隙、速度匹配、主辅路关系。

需要注意 R-E2 与 R-E3 的边界：

- HighwayExit 同时是“目标导向型变道”和“高速驶出结构”。
- 建议道路结构为 R3，主事件优先标 R-E3；如果需要更细，可以给事件加原因字段
  `reason=route_exit`。

### R4. 信号灯路口

保留，但它是“规则控制结构”，不只是几何结构。只要红绿灯是当前主要决策依据，就应标 R4。

建议 R4 与 R5 互斥：

- R4：信号灯正常可用，红绿灯是主规则。
- R5：无灯、灯不可用、或灯控不应作为主规则。

OppositeVehicleRunningRedLight 仍属于 R4，因为灯正常，是对方违规。

### R5. 无信号灯路口

保留。CrossJunctionDefectTrafficLight 放 R5 是合理的，但要额外保留 U-E7，
因为“灯坏导致四向规则失效”比普通无灯路口更难。

PriorityAtJunction 是否必然无灯需要核实；如果无法从 meta 获取，只能先由 scenario 先验 + Qwen 视觉判断。

### R6. 路边停车 / 停车占道道路

保留，但建议把它定义成“停车带/路边遮挡导致有效通行空间和风险模型改变”，而不是单纯停车事件。

R6 与 R2 允许重叠，例如 VehicleOpensDoorTwoWays：

- 道路结构上可能是 R2 双向单车道。
- 风险上下文上又是 R6 路边停车/开门。

如果模型只能单选 ROAD_STRUCTURE，建议优先选更影响规划动作的结构：

```text
R2 借对向车道风险 > R6 停车遮挡上下文
```

如果后续允许多标签，可把 R6 作为 `ROAD_CONTEXT` 辅助标签。

## 4. 是否需要改分类

建议小改，不建议推翻。

推荐保留 6 类道路结构，但把名字和边界写得更严格：

| 建议 token | 对应你的分类 | 主判据 |
|---|---|---|
| `R1_SAME_DIRECTION_ROAD` | R1 | 同向道路/可行驶区，非路口、非合流、非停车带主导 |
| `R2_TWO_WAY_BORROW_ROAD` | R2 | 双向单车道，是否涉及借对向/对向侵占 |
| `R3_MERGE_EXIT_ROAD` | R3 | 主辅路、匝道、合流、驶出、高速切入 |
| `R4_SIGNALIZED_JUNCTION` | R4 | 信号灯可用且是主要通行规则 |
| `R5_UNSIGNALIZED_OR_DEFECTIVE_JUNCTION` | R5 | 无灯/灯失效/按路权和安全间隙通行 |
| `R6_PARKING_ROADSIDE_ZONE` | R6 | 路边停车、停车位汇入、开门、遮挡风险主导 |

与当前 `sft_v4/prompts.py` 的 6 桶相比，你的新 R1-R6 更细地拆开了：

- 旧 `JUNCTION` -> 新 R4/R5。
- 旧 `ROADSIDE_HAZARD` -> 新 R1/R2/R6 + U-E2。
- 旧 `VRU_CROSSING` 更像事件 U-E4，不应作为道路结构主类。
- 旧 `OPEN_ROAD_DYNAMICS` 更像 R1 + U-E1/U-E5。

因此，如果后续重构 v4 prompt，你的新分类更贴近驾驶策略；但需要重新生成/验证数据，不能只改 prompt 名字。

## 5. 事件分类审阅

你的事件集合整体合理，建议保持“常规事件 R-E* + 突发事件 U-E*”双层。

### 常规事件

`R-E1 跟车 / 车道保持` 应作为所有道路结构下的默认背景事件。但在 R4/R5 中，
如果车辆已经进入路口决策区，优先使用 R-E4/R-E5，而不是继续 R-E1。

`R-E2 目标导向型变道` 合理，关键是和 U-E2 被迫变道分开：

- R-E2：路线/目标要求，当前车道通常仍可行。
- U-E2：当前路径被障碍阻断，变道是被迫绕行。

`R-E3 常规匝道合流 / 并线 / 驶出` 合理，但 HighwayExit 与 R-E2 有重叠。
建议主事件标 R-E3，辅助原因标 `route_required_exit`。

`R-E4 信号灯路口通行` 与 `R-E5 无信号灯路口通行` 合理，应该作为 R4/R5 的默认事件。

`R-E6 常规停车等待 / 恢复启动` 建议不要作为主事件默认输出，而作为 phase/substate：

```text
event = R-E4
phase = wait_red / resume_green

event = R-E5
phase = wait_gap / pass_gap

event = R-E3
phase = wait_merge_gap / merge
```

如果模型输出只能有一个 EVENT，R-E6 会和 R-E3/R-E4/R-E5 重叠太多。

### 突发事件

`U-E1 前车急刹 / 突然减速` 保留，主要对应 HardBreakRoute。

`U-E2 静态障碍物占道` 保留，是 Accident / Construction / ParkedObstacle 的核心事件。
建议把“同向绕行”和“借对向绕行”交给 ROAD_STRUCTURE 或 action subtype，不要再拆两个事件。

`U-E3 动态车辆切入 / 动态占道` 保留。它要和 R-E3 常规合流严格区分：

- 自车主动找间隙汇入：R-E3。
- 他车进入自车路径，自车被动让行：U-E3。

`U-E4 行人 / 自行车横穿` 保留。它可以发生在 R1/R4/R5/R6，不应单独变成道路结构。

`U-E5 对向车辆异常侵占自车道` 保留。它是 R2 下的被动让行，与主动借道绕障不同。

`U-E6 违规车辆冲突` 保留，适合 OppositeVehicleRunningRedLight。
它和 U-E7 的区别是：U-E6 是规则正常但参与者违规；U-E7 是规则源本身失效。

`U-E7 信号灯故障 / 路口规则失效` 保留，但它不是短插入事件，而是整段路口通行模式。
可以看成 R5 下的异常主事件。

`U-E8 前方道路暂时阻塞 / 阻塞解除` 保留。BlockedIntersection 不适合混入 U-E2，
因为核心不是绕障，而是等待长期阻塞解除后重新启动。

## 6. 建议的优先级规则

一帧可能同时满足多个线索，需要固定仲裁规则。

道路结构优先级建议：

```text
明确路口区域:
  有正常可用信号灯 -> R4
  无灯或灯失效 -> R5
否则如果有主辅路/匝道/高速出口/合流结构 -> R3
否则如果有效通行空间由路边停车/停车位主导 -> R6
否则如果双向单车道且对向车道参与决策 -> R2
否则 -> R1
```

如果只能单选，R4/R5 应优先于 R1/R2/R6，因为进入路口后通行规则变了。
如果未来允许多标签，建议拆成：

```text
ROAD_STRUCTURE: R1/R2/R3/R4/R5
ROAD_CONTEXT: parking_zone / roadside_occlusion / none
```

事件优先级建议：

```text
U-E6 违规冲突 / U-E7 规则失效
> U-E4 行人/自行车横穿
> U-E5 对向侵占
> U-E3 动态切入
> U-E2 静态障碍占道
> U-E1 前车急刹
> R-E4/R-E5 路口常规通行
> R-E3 合流/驶出
> R-E2 目标导向变道
> R-E1 跟车/车道保持
```

这不是说高优先级一定更重要，而是为了让单标签时不把安全关键事件淹没在背景事件里。

## 7. Scenario 到道路结构候选的初始映射

这个映射只作为 Qwen 选择范围，不代表帧级真值。

| Scenario | 候选 ROAD_STRUCTURE |
|---|---|
| Accident | R1, 条件性 R4/R5 |
| AccidentTwoWays | R2, 条件性 R4/R5 |
| BlockedIntersection | R4/R5, 条件性 R1 |
| ConstructionObstacle | R1, 条件性 R4/R5 |
| ConstructionObstacleTwoWays | R2, 条件性 R4/R5 |
| ControlLoss | R1 |
| CrossingBicycleFlow | R4/R5, 条件性 R1 |
| CrossJunctionDefectTrafficLight | R5 |
| DynamicObjectCrossing | R1, 条件性 R4/R5 |
| EnterActorFlow / EnterActorFlowV2 | R3, 条件性 R1 |
| HardBreakRoute | R1, 条件性 R4/R5 |
| HazardAtSideLane | R1 或 R2，需核实 |
| HazardAtSideLaneTwoWays | R2 |
| HighwayCutIn | R3 |
| HighwayExit | R3 |
| InterurbanActorFlow | R1, R4/R5 |
| InterurbanAdvancedActorFlow | R4/R5，条件性 R1 |
| InvadingTurn | R2 |
| MergerIntoSlowTraffic / V2 | R3 |
| NonSignalizedJunctionLeftTurn / EnterFlow | R5 |
| NonSignalizedJunctionRightTurn | R5 |
| noScenarios | R1 |
| OppositeVehicleRunningRedLight | R4 |
| OppositeVehicleTakingPriority | R5 |
| ParkedObstacle | R1, 条件性 R4/R5 |
| ParkedObstacleTwoWays | R2, 条件性 R4/R5 |
| ParkingCrossingPedestrian | R6, 条件性 R1/R5 |
| ParkingCutIn | R6 |
| ParkingExit | R6 |
| PedestrianCrossing | R5 或 R4，需按是否有灯判断 |
| PriorityAtJunction | R5，是否有灯需核实 |
| RedLightWithoutLeadVehicle | R4 |
| SignalizedJunctionLeftTurn / EnterFlow | R4 |
| SignalizedJunctionRightTurn | R4 |
| StaticCutIn | R1/R3/R6，需按数据位置拆分 |
| T_Junction | R4 或 R5，需按信号灯判断 |
| VehicleOpensDoorTwoWays | R2 + R6，上层单选时优先 R2 |
| VehicleTurningRoute | R4/R5 |
| VehicleTurningRoutePedestrian | R4/R5 |

## 8. ROAD_STRUCTURE 到事件候选的裁剪

这一步能显著降低 Qwen 难度。

| ROAD_STRUCTURE | 常规事件候选 | 突发事件候选 |
|---|---|---|
| R1 | R-E1, R-E2 | U-E1, U-E2, U-E3, U-E4 |
| R2 | R-E1 | U-E2, U-E5 |
| R3 | R-E1, R-E2, R-E3 | U-E3 |
| R4 | R-E4 | U-E4, U-E6, U-E8 |
| R5 | R-E5 | U-E4, U-E7, U-E8 |
| R6 | R-E1, R-E2 | U-E2, U-E3, U-E4 |

注意：

- U-E4 可以跨 R1/R4/R5/R6。
- U-E2 可以跨 R1/R2/R6，但行为不同：R1/R6 多为同向绕行，R2 多为借对向。
- U-E8 主要是 BlockedIntersection，不应开放给所有道路。
- R-E6 建议作为 phase，不放入主事件候选表。

## 9. 新数据重生成思路

建议分两阶段，先突发事件 span，再道路结构。

### 阶段 A：先找突发事件 span

原因：突发事件更容易由 meta/bbox/速度信号定位，且对规划最关键。

可用弱标签规则：

- U-E1：HardBreakRoute 中 `brake` 峰值、`accel_x` 最小值、前车距离快速缩小。
- U-E2：障碍距离字段进入阈值、速度下降、通过后距离增大或速度恢复。
- U-E3：cut-in 距离字段下降、相对对象进入前方路径、速度/刹车响应。
- U-E4：pedestrian/biker 距离进入阈值，速度下降，目标离开路径后恢复。
- U-E5：InvadingTurn 中 cut-in/oncoming 距离 + 自车停止等待 + 对向车离开。
- U-E6：OppositeVehicleRunningRedLight 中路口距离 + 威胁车接近 + 自车让行。
- U-E7：CrossJunctionDefectTrafficLight 的整个路口接近到通过阶段。
- U-E8：BlockedIntersection 中长时间低速/停车 + path blocked + 解除后恢复。

输出不要只是一帧，建议是：

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

### 阶段 B：道路结构先用候选约束 + Qwen 判断

道路结构更依赖图像，不看图片很难从旧 meta 精确判别 R1/R4/R5。

初期可采用：

- 由 scenario 提供候选道路结构集合。
- Qwen step1 只在候选集合里选当前 `ROAD_STRUCTURE`。
- 如果没有看到红绿灯、路口停止线、横向车流、转弯区域等明确证据，默认保持 R1。
- 如果看到红绿灯/停止线/路口几何，再从 R4/R5 中选择。
- 如果该 scenario 本身不可能出现某类道路，就不提供该候选。

这部分暂时没有强监督真值，建议先作为弱监督/自训练目标，而不是硬标签。

### 阶段 C：事件标签由 ROAD_STRUCTURE 裁剪

每帧先确定候选 ROAD_STRUCTURE，再列出该道路结构允许的事件候选。

如果 frame 落在阶段 A 的突发 span 内，优先给突发事件。
否则按道路结构给常规背景事件：

- R1/R2/R6 默认 R-E1，除非 route/lane-change 信号明显支持 R-E2。
- R3 默认 R-E3，合流前后可落回 R-E1。
- R4 默认 R-E4。
- R5 默认 R-E5。

### 阶段 D：subgoal 生成

旧 keyframe 的 subgoal 可以迁移为“事件阶段目标”，但不要直接复用旧 event 名。

建议新 subgoal 表达：

```text
STATUS = 当前正在发生的事件
SUBGOAL = 下一步应完成的可见驾驶阶段
```

例子：

```text
STATUS: R-E1
SUBGOAL: U-E2   # 前方障碍逐渐可见，下一目标是准备绕行

STATUS: U-E2
SUBGOAL: R-E2   # 已经绕过障碍，下一目标是回到目标车道

STATUS: R-E4
SUBGOAL: U-E6   # 绿灯路口中看到违规车，下一目标转为避让冲突
```

## 10. 待办与风险

必须人工/视频核实的问题：

- HazardAtSideLane 到底是 R1 同向多车道避让，还是部分样本应归 R2。
- StaticCutIn 是否混合了 R1、R3、R6，需要按视频位置拆分。
- T_Junction、PedestrianCrossing、VehicleTurningRoute 系列是否有灯，决定 R4/R5。
- PriorityAtJunction 是否应稳定归 R5。
- InterurbanAdvancedActorFlow 是否有前置 R1 变道片段。

实现风险：

- 只靠 scenario 监督 ROAD_STRUCTURE 会把同一 route 内的切换抹掉。
- Qwen 弱标签会受 prompt 偏置影响，需要人工抽检闭环。
- 突发 span 的 start/end 边界不会很准，训练时应容忍模糊边界。
- 如果 EVENT 单标签，R-E6 不适合作主事件；否则会与路口/合流事件大量重叠。
- 如果 ROAD_STRUCTURE 单标签，R6 与 R2/R1 可能冲突；后续最好引入 `ROAD_CONTEXT`。

## 11. 推荐的下一步

1. 固化 R1-R6 和 R-E/U-E 定义，先不要急着改代码训练。
2. 给每个 scenario 写候选 ROAD_STRUCTURE 集合和候选 EVENT 集合。
3. 从旧 `rule_based_keyframe_filter.py` 抽出 meta span 提议器，只生成突发事件 span，不再输出 3 个 middle 点。
4. 用 verification viewer 或新工具抽检每个事件 20-50 条视频，先修阈值和边界。
5. 设计 Qwen prompt：step1 选 ROAD_STRUCTURE，step2 在该结构下选 EVENT，step3 输出 STATUS/SUBGOAL。
6. 先在不训练的 frozen Qwen 上跑小样本，检查它是否能稳定区分 R1/R4/R5、R-E2/U-E2、R-E3/U-E3。
7. 通过抽检后，再生成新的 dataset jsonl，替代旧 `keyframes_all_scenarios.json` 的 status/subgoal 监督。

## 12. 当前结论

你的分类体系总体有理有据，覆盖了 LEAD 这些 scenario 的主要决策情况。最需要调整的不是增加很多类，
而是明确三条边界：

- 道路结构和事件分离：VRU、急刹、切入、障碍都更像事件，不要再让 scenario 名直接决定道路结构。
- 常规事件和突发事件分离：目标导向变道、常规合流、路口通行不要和被迫避让混在一起。
- 单标签与多标签分离：R6 这类停车上下文、R-E6 这类等待/恢复，更适合作辅助属性或 phase。

按这个方案走，后续数据标签会比旧 keyframe 方法更贴近实际驾驶过程，也更适合让 Qwen 做分层选择，
而不是在几十个混杂 status/subgoal 里直接猜。
