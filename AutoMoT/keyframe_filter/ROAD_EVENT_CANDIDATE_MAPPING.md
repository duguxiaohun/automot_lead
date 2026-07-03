# ROAD_STRUCTURE / EVENT 候选映射表

本文是 `ROAD_EVENT_CLASSIFICATION_PLAN.md` 的落地候选表，目标是给后续 Qwen 标注/训练提供
三个层级的候选空间：

1. 每个 CARLA scenario 可能出现哪些 `ROAD_STRUCTURE`。
2. 每个 `ROAD_STRUCTURE` 下允许哪些 `EVENT`。
3. 每个 scenario 在自己的 `ROAD_STRUCTURE` 并集下，进一步排除明显不可能事件后的精细 `EVENTS`。

## 0. 标注口径

`ROAD_STRUCTURE` 表示驾驶决策规则空间，不是纯物理几何。它的作用是先限定当前帧应使用哪套通行规则，
再在这套规则下选择事件。

XML / 数据命名必须沿用本项目 `lead_data` 到 `data/lead` 的固定映射，不能只提取数字 route：

```text
lead_data/<Scenario>/<run_id>
  -> parse (Scenario, Town, route_key)
  -> data/lead/<Scenario>/<Town>_<route_key>.xml
```

解析 `run_id` 时，`Scenario` 必须取父目录；先剥末尾 `MM_DD_HH_MM_SS` 时间戳，再只在存在时剥尾部采集后缀
`_route0`，剩余部分就是 `route_key`。`Town12_route15` 里的 `route15` 是 key 本体，不能剥，也不能要求它带
`_route0`。文件名公式固定为：`route_key` 以 `route_` 开头时用 `<Town>_<route_key>.xml`，否则用
`<Town>_route_<route_key>.xml`。示例：

```text
Town03_Rep0_route_001783_route0_... -> data/lead/<Scenario>/Town03_route_001783.xml
Town12_Rep0_1054_0_route0_...       -> data/lead/<Scenario>/Town12_route_1054_0.xml
Town06_Rep0_Town06_13_route0_...    -> data/lead/<Scenario>/Town06_route_Town06_13.xml
Town12_Rep0_Town12_route15_...      -> data/lead/<Scenario>/Town12_route_Town12_route15.xml
```

2026-07-03 全量核对结果：`lead_data` 9715 个 run 去重后 9294 个 `(Scenario,Town,route_key)`，
`data/lead` 正好 9294 个 XML，缺失 0、冗余 0、命名不规范 0、XML 解析失败 0、内容结构异常 0。
40 个 XML 的 `data_routes` 源在其它 scenario 目录，不是缺失；现有
`ParkedObstacle/Town12_route_Town12_route15.xml` 覆盖有效，不能当作 `xml_available=false`。

`EVENTS` 允许多选。常规事件表示背景驾驶任务，突发事件表示安全关键打断。例如：

```text
ROAD_STRUCTURE: R1
EVENTS: R-E1, U-E2
```

表示当前仍在常规道路规则空间内，背景任务是跟车/车道保持，同时前方出现静态障碍物占道。

本文使用保守覆盖策略：

- 尽量保证候选覆盖率，不把可能出现的合理事件漏掉。
- 用排除法删除肯定不属于该规则空间或该 scenario 的事件。
- 不确定但可能出现的事件，会保留并在备注中说明原因。

## 1. 全局道路结构默认假设

LEAD route 通常不是只有 scenario 核心片段；很多 route 在进入/离开核心事件前后会有直道、
普通车道保持、跟车、以及普通路口片段。因此本文采用以下默认假设：

- 除非明显不成立，每个 scenario 至少包含 `R1 常规道路 / 同向可行驶道路`。
- 除无信号灯/信号灯失效/路权类 scenario 外，每个 scenario 默认可包含 `R4 信号灯路口`。
- 明确无信号灯、信号灯失效或按路权通过的 scenario，用 `R5 无信号灯 / 信号灯失效路口`
  替代默认 `R4`。
- R2/R3/R6 只在 scenario 特征明确支持时额外加入。

这意味着：Accident 这种描述中没有专门写十字路口的场景，也仍然给 `R1 + R4`，
因为实际 route 可能包含普通直道和有信号灯路口；然后在第三张表中再排除 Accident
肯定不会有的路口异常事件。

## 2. ROAD_STRUCTURE 定义

| ID | 名称 | 判定口径 |
|---|---|---|
| R1 | 常规道路 / 同向可行驶道路 | 默认规则空间；普通直道、跟车、车道保持、同向变道、普通可行驶区域 |
| R2 | 双向单车道 / 借对向车道道路 | 对向车道参与决策；包括自车借对向绕障或对向车侵占自车道 |
| R3 | 高速合流 / 匝道 / 分流 / 驶出决策结构 | 主辅路、匝道、合流、并线、驶出；普通高速直行或同向 cut-in 若无 merge/split/ramp/exit 结构，仍回 R1 + EVENT |
| R4 | 信号灯路口 | 红绿灯正常可用，红绿灯是主通行规则 |
| R5 | 无信号灯 / 信号灯失效路口 | 无灯、灯失效、或主要按路权/安全间隙通行 |
| R6 | 路边停车 / 停车占道道路 | 停车带、路边停车、停车位汇入、开门、停车遮挡主导决策 |

## 3. 每个 scenario 的 ROAD_STRUCTURE 候选

| Scenario | ROAD_STRUCTURE 候选 | 说明 |
|---|---|---|
| Accident | R1, R4 | 默认直道 + 信号灯路口；核心为同向静态障碍绕行 |
| AccidentTwoWays | R1, R2, R4 | 默认直道 + 信号灯路口；核心为双向单车道借对向绕障 |
| BlockedIntersection | R1, R4 | 跟车背景 + 看灯路口；核心为前方道路阻塞/解除 |
| ConstructionObstacle | R1, R4 | 默认直道 + 信号灯路口；核心为同向施工障碍绕行 |
| ConstructionObstacleTwoWays | R1, R2, R4 | 默认直道 + 信号灯路口；核心为双向单车道借对向绕施工障碍 |
| ControlLoss | R1, R4 | 用户调研中近似跟车；保留默认信号灯路口片段 |
| CrossingBicycleFlow | R1, R4 | 默认直道 + 信号灯路口；核心为自行车横穿 |
| CrossJunctionDefectTrafficLight | R1, R5 | 默认直道 + 信号灯失效路口；不放 R4 |
| DynamicObjectCrossing | R1, R4 | 默认直道 + 信号灯路口；可能存在动态对象横穿/干扰 |
| EnterActorFlow | R1, R3, R4 | 默认直道 + 信号灯路口；核心为合流/进入车流 |
| EnterActorFlowV2 | R1, R3, R4 | 与 EnterActorFlow 同候选 |
| HardBreakRoute | R1, R4 | 默认直道 + 信号灯路口；核心为前车急刹 |
| HazardAtSideLane | R1, R4 | 当前按同向可行驶道路处理；是否有 R2 需视频核实 |
| HazardAtSideLaneTwoWays | R1, R2, R4 | 默认直道 + 信号灯路口；核心为双向单车道侧向危险绕行 |
| HighwayCutIn | R1, R3, R4 | 默认直道 + 信号灯路口；核心为高速/匝道他车切入 |
| HighwayExit | R1, R3, R4 | 默认直道 + 信号灯路口；核心为高速驶出 |
| InterurbanActorFlow | R1, R3, R4, R5 | 左变道/进入车流 + 路口寻找时机左转；路口信号不确定时保留 R4/R5 |
| InterurbanAdvancedActorFlow | R1, R4, R5 | 主要按直道 + 路口寻找时机左转；若视频显示合流再加 R3 |
| InvadingTurn | R1, R2, R4 | 默认直道 + 信号灯路口；核心为对向车侵占自车道 |
| MergerIntoSlowTraffic | R1, R3, R4 | 默认直道 + 信号灯路口；核心为合流进入慢速车流 |
| MergerIntoSlowTrafficV2 | R1, R3, R4 | 与 MergerIntoSlowTraffic 同候选 |
| NonSignalizedJunctionLeftTurn | R1, R5 | 明确无信号灯左转；不放 R4 |
| NonSignalizedJunctionLeftTurnEnterFlow | R1, R5 | 明确无信号灯左转进入车流；不放 R4 |
| NonSignalizedJunctionRightTurn | R1, R5 | 明确无信号灯右转；不放 R4 |
| noScenarios | R1, R4 | 默认直道 + 信号灯路口；无核心异常 |
| OppositeVehicleRunningRedLight | R1, R4 | 信号灯正常但对方违规 |
| OppositeVehicleTakingPriority | R1, R5 | 无信号灯/路权类对向车优先；不放 R4 |
| ParkedObstacle | R1, R4 | 默认直道 + 信号灯路口；核心为同向停放障碍绕行 |
| ParkedObstacleTwoWays | R1, R2, R4 | 默认直道 + 信号灯路口；核心为双向单车道借对向绕停放障碍 |
| ParkingCrossingPedestrian | R1, R4, R6 | 默认直道 + 信号灯路口；核心为停车区域行人横穿 |
| ParkingCutIn | R1, R4, R6 | 默认直道 + 信号灯路口；核心为停车车辆动态切入 |
| ParkingExit | R1, R4, R6 | 默认直道 + 信号灯路口；核心为从停车区域并入主路 |
| PedestrianCrossing | R1, R4, R5 | 用户调研写“信号灯看情况有无”，保留 R4/R5 |
| PriorityAtJunction | R1, R5 | 路权类路口；不放 R4 |
| RedLightWithoutLeadVehicle | R1, R4 | 明确信号灯路口 |
| SignalizedJunctionLeftTurn | R1, R4 | 明确信号灯左转 |
| SignalizedJunctionLeftTurnEnterFlow | R1, R4 | 明确信号灯左转进入车流 |
| SignalizedJunctionRightTurn | R1, R4 | 明确信号灯右转 |
| StaticCutIn | R1, R3, R4, R6 | 可能混合初始目标变道、匝道/合流、停车区切入；需视频拆分 |
| T_Junction | R1, R4 | 用户调研为等绿灯通过，按信号灯路口 |
| VehicleOpensDoorTwoWays | R1, R2, R4, R6 | 双向单车道 + 路边停车/开门风险 |
| VehicleTurningRoute | R1, R4, R5 | 和 PedestrianCrossing 类似，转弯后横穿对象/冲突，保留 R4/R5 |
| VehicleTurningRoutePedestrian | R1, R4, R5 | 和 PedestrianCrossing 类似，转弯后行人/自行车横穿，保留 R4/R5 |

## 4. 每个 ROAD_STRUCTURE 下的 EVENT 候选

这里先从道路规则空间出发，用排除法去掉肯定不属于该空间的事件。

| ROAD_STRUCTURE | 保留 EVENTS | 排除逻辑 |
|---|---|---|
| R1 常规道路 / 同向可行驶道路 | R-E1, R-E2, U-E1, U-E2, U-E3, U-E4 | 不放 R-E3/R-E4/R-E5；不放 U-E5，因为对向侵占属于 R2；不放 U-E6/U-E7/U-E8，因为它们是路口/阻塞类 |
| R2 双向单车道 / 借对向车道道路 | R-E1, U-E2, U-E5 | 不放 U-E3，普通动态切入不是 R2 的核心；不放 U-E6/U-E7/U-E8；不放 R-E2，借对向绕障由 U-E2 表达 |
| R3 高速合流 / 匝道 / 分流 / 驶出决策结构 | R-E1, R-E2, R-E3, U-E3 | 不放 U-E2，静态障碍绕行不属于 R3 核心；不放 U-E5/U-E6/U-E7/U-E8；不放 R-E4/R-E5；物理高速直行不能单独触发 R3 |
| R4 信号灯路口 | R-E4, U-E4, U-E6, U-E8 | 不放 R-E5/U-E7，因为信号灯正常；不放 U-E5，因为对向侵占属于 R2；不放 U-E2，普通静态绕障不作为 R4 核心事件 |
| R5 无信号灯 / 信号灯失效路口 | R-E5, U-E4, U-E7, U-E8 | 不放 R-E4/U-E6，前者是信号灯正常，后者主要是闯红灯违规；不放 U-E5；不放 U-E2 |
| R6 路边停车 / 停车占道道路 | R-E1, R-E2, U-E2, U-E3, U-E4 | 不放 R-E3/R-E4/R-E5；不放 U-E5/U-E6/U-E7/U-E8 |

说明：

- R-E1 是大多数非路口/非合流结构的背景事件。
- R-E2 只保留在目标导向变道、停车区汇入、同向道路目标车道调整等空间。
- U-E4 可以跨 R1/R4/R5/R6，因为行人/自行车横穿可发生在直道、路口和停车遮挡区域。
- U-E8 只放 R4/R5，因为它描述前方道路/路口通行空间阻塞，而不是普通静态障碍绕行。

## 5. 每个 scenario 的精细 EVENTS 候选

本表先取该 scenario 的 ROAD_STRUCTURE 事件并集，再按 scenario 语义排除肯定不存在的事件。
因此它比第 4 节更窄，更适合给 Qwen 做第二层候选。

| Scenario | 精细 EVENTS 候选 | 排除/保留说明 |
|---|---|---|
| Accident | R-E1, R-E2, R-E4, U-E2 | 保留普通行驶、目标回归/变道、信号灯通行、静态事故障碍；排除对向侵占/行人/切入 |
| AccidentTwoWays | R-E1, R-E2, R-E4, U-E2 | 核心是借对向绕静态事故障碍；正常对向来车等待不等于 U-E5 |
| BlockedIntersection | R-E1, R-E4, U-E8 | 核心是信号灯路口前方阻塞和解除；排除 U-E2 绕障 |
| ConstructionObstacle | R-E1, R-E2, R-E4, U-E2 | 施工静态障碍同向绕行 |
| ConstructionObstacleTwoWays | R-E1, R-E2, R-E4, U-E2 | 双向单车道借对向绕施工障碍 |
| ControlLoss | R-E1, R-E4 | 用户调研为跟车；暂不保留急刹/障碍等突发 |
| CrossingBicycleFlow | R-E1, R-E4, U-E4 | 自行车横穿 + 信号灯路口常规通行 |
| CrossJunctionDefectTrafficLight | R-E1, R-E5, U-E7 | 信号灯失效，按无信号路口处理 |
| DynamicObjectCrossing | R-E1, R-E4, U-E3, U-E4 | 动态对象可能表现为车辆/小车切入，也可能为行人/自行车横穿，保留两类 |
| EnterActorFlow | R-E1, R-E3, R-E4 | 自车主动进入车流，非他车切入 |
| EnterActorFlowV2 | R-E1, R-E3, R-E4 | 同 EnterActorFlow |
| HardBreakRoute | R-E1, R-E4, U-E1 | 前车急刹；信号灯片段按 R-E4 |
| HazardAtSideLane | R-E1, R-E2, R-E4, U-E2 | 侧向静态危险物/占道绕行；暂按 U-E2 |
| HazardAtSideLaneTwoWays | R-E1, R-E2, R-E4, U-E2 | 双向单车道侧向危险绕行 |
| HighwayCutIn | R-E1, R-E3, R-E4, U-E3 | 高速/匝道空间下他车切入；保留 R-E3 背景但核心突发是 U-E3 |
| HighwayExit | R-E1, R-E2, R-E3, R-E4 | 高速驶出与目标导向变道；无核心突发 |
| InterurbanActorFlow | R-E1, R-E2, R-E3, R-E4, R-E5 | 左变道/进入车流/路口寻找时机左转均可能出现 |
| InterurbanAdvancedActorFlow | R-E1, R-E4, R-E5 | 主要为路口寻找时机左转；若视频显示合流再加 R-E3 |
| InvadingTurn | R-E1, R-E4, U-E5 | 对向车辆异常侵占自车道；不加入 U-E2 |
| MergerIntoSlowTraffic | R-E1, R-E3, R-E4 | 自车主动合流进入慢速车流 |
| MergerIntoSlowTrafficV2 | R-E1, R-E3, R-E4 | 同 MergerIntoSlowTraffic |
| NonSignalizedJunctionLeftTurn | R-E1, R-E5 | 无信号灯左转；不放 R-E4 |
| NonSignalizedJunctionLeftTurnEnterFlow | R-E1, R-E5 | 无信号灯左转进入车流，仍按 R-E5 主导 |
| NonSignalizedJunctionRightTurn | R-E1, R-E5 | 无信号灯右转 |
| noScenarios | R-E1, R-E4 | 默认正常行驶 + 信号灯通行；无核心突发 |
| OppositeVehicleRunningRedLight | R-E1, R-E4, U-E6 | 信号灯正常但对向车闯红灯 |
| OppositeVehicleTakingPriority | R-E1, R-E5 | 无信号灯/路权场景，对方优先不是违规突发 |
| ParkedObstacle | R-E1, R-E2, R-E4, U-E2 | 静态停放障碍绕行 |
| ParkedObstacleTwoWays | R-E1, R-E2, R-E4, U-E2 | 双向单车道借对向绕停放障碍 |
| ParkingCrossingPedestrian | R-E1, R-E4, U-E4 | 停车区行人横穿；不放 U-E3 |
| ParkingCutIn | R-E1, R-E4, U-E3 | 停车车辆启动/切入当前路径 |
| ParkingExit | R-E1, R-E2, R-E4 | 自车从停车区域汇入主路，目标导向变道/汇入 |
| PedestrianCrossing | R-E1, R-E4, R-E5, U-E4 | 信号灯有无不稳定，保留 R-E4/R-E5 + 行人横穿 |
| PriorityAtJunction | R-E1, R-E5 | 路权类无信号路口 |
| RedLightWithoutLeadVehicle | R-E1, R-E4 | 正常红灯等待/绿灯通过并入 R-E4 |
| SignalizedJunctionLeftTurn | R-E1, R-E4 | 信号灯左转 |
| SignalizedJunctionLeftTurnEnterFlow | R-E1, R-E4 | 信号灯左转进入车流；仍按 R-E4 主导 |
| SignalizedJunctionRightTurn | R-E1, R-E4 | 信号灯右转 |
| StaticCutIn | R-E1, R-E2, R-E3, R-E4, U-E3 | 起步目标变道 + 可能合流/停车区 + 后半段动态切入 |
| T_Junction | R-E1, R-E4 | 用户调研为等绿灯通过 |
| VehicleOpensDoorTwoWays | R-E1, R-E2, R-E4, U-E2 | 开门/停车静态占道风险 + 双向单车道绕行 |
| VehicleTurningRoute | R-E1, R-E4, R-E5, U-E4 | 转弯后自行车/横穿对象风险，信号灯不确定时保留 R-E4/R-E5 |
| VehicleTurningRoutePedestrian | R-E1, R-E4, R-E5, U-E4 | 转弯后行人/自行车横穿，信号灯不确定时保留 R-E4/R-E5 |

## 6. 关键排除规则

这些规则用于后续写脚本或 prompt 时保持一致：

- `U-E5 对向车辆异常侵占自车道` 只在 R2 规则空间开放，当前只给 InvadingTurn。
- `U-E7 信号灯故障 / 路口规则失效` 只给 R5，当前只给 CrossJunctionDefectTrafficLight。
- `U-E6 违规车辆冲突` 主要是信号灯正常但对方违规，当前只给 OppositeVehicleRunningRedLight。
- `U-E8 前方道路暂时阻塞 / 阻塞解除` 当前只给 BlockedIntersection。
- `R-E5 无信号灯路口通行` 只给无信号/路权/灯故障类 scenario。
- `R-E4 信号灯路口通行` 给默认信号灯配置和明确信号灯 scenario，不给无信号/灯故障 scenario。
- `R-E6` 已取消，不出现在候选表中。
- TwoWays 场景中的正常对向来车等待不等于 U-E5；只有对向车异常侵占自车道才是 U-E5。

## 7. 后续实现建议

后续脚本可以按三段式构造候选：

1. 读取 scenario，查第 3 节得到 ROAD_STRUCTURE 候选。
2. Qwen step1 从 ROAD_STRUCTURE 候选中选当前帧规则空间。
3. 根据第 4 节取该 ROAD_STRUCTURE 的事件候选，再与第 5 节的 scenario 精细事件候选求交集。

本轮 ROAD_STRUCTURE 5-id/town 调研后，EVENT 候选还必须遵守以下依赖：

- EVENT 不反推 RS。`HardBreakRoute`、`ControlLoss`、`DynamicObjectCrossing`、
  `BlockedIntersection`、`OppositeVehicleRunningRedLight` 的异常只进入 EVENT/span，
  primary RS 仍由 XML/XODR/meta 的道路结构证据决定。
- `TwoWays` 只让候选池包含 R2/U-E2，不代表全程 R2；只有 RS 输出为 R2 时才开放对向绕行相关事件。
- `Parking*` 只让候选池包含 R6/停车区事件，不代表全程 R6；灯控路口段仍优先 R4。
- `EnterFlow` 名称不等于 R3；只有 `highway_merge/interurban` 规则族且 XODR merge/highway 证据
  或 `MergerIntoSlowTraffic*` 的 XML actor-flow fallback 成立时，EVENT 才按 R3 的高速/合流规则空间收窄。
- `CrossJunctionDefectTrafficLight` 的 RS 由 defect 机制强制 R5；EVENT 选择 U-E7/R-E5 时不要再被
  XODR signal 存在性拉回 R4。

伪代码：

```text
road_candidates = SCENARIO_TO_ROAD_STRUCTURE[scenario]
road = qwen_select_road_structure(frame, road_candidates)

event_candidates = ROAD_STRUCTURE_TO_EVENTS[road] ∩ SCENARIO_TO_EVENTS[scenario]
events = qwen_select_events(frame, event_candidates, multi_select=True)
```

这样可以最大程度保证覆盖率，同时通过 ROAD_STRUCTURE 与 scenario 双重排除，把肯定不存在的事件挡在候选表外。
