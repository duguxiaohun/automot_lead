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
单条 run 若缺少 `metas/*.pkl` 或匹配 XML，则属于数据质量问题，必须 `data_missing_skip`
并在 summary/可视化里写明 `missing_meta` / `missing_route_xml`，不进入下面的 RS/EVENT 候选判定。

`EVENTS` 数据结构允许多选，但当前标注口径是“默认单主事件，路口双触发例外”。
非路口/非十字路口核心段命中 U-E 时，只输出 U-E 主事件；退出非常规 span 后再切回 R-E1/R-E2/R-E3。
只有 R4/R5 路口允许 `R-E4/R-E5 + U-E*` 同帧共存，且必须由 XML/active window、meta/轨迹和 RGB 可见对象限定。
场景级 EVENT 表只是该 scenario 的上限；最终候选还必须再和当前 ROAD_STRUCTURE 的候选池取交集，
并始终保留当前 RS 的 regular event。也就是说 R4/R5 路口帧会直接删除 `U-E2/U-E3`，
红灯等待、路口排队、路口起步只能走 `R-E4/R-E5` 或路口专属 U-E。
例如：

```text
ROAD_STRUCTURE: R1
EVENTS: U-E2
```

表示当前仍在常规道路规则空间内，但跟车/车道保持被静态障碍物占道打断。

本文使用白名单优先策略：

- 常规事件按 ROAD_STRUCTURE 和自车轨迹意图开放。
- 非常规事件只按 scenario 白名单开放；用户没有明确给出的 U-E 不自动加入候选。
- 不确定但可能出现的事件先标 review / 待 RGB 复核，不直接进入训练候选。

## 1. 全局道路结构默认假设

LEAD route 通常不是只有 scenario 核心片段；很多 route 在进入/离开核心事件前后会有直道、
普通车道保持、跟车、以及普通路口片段。因此本文采用以下默认假设：

- 默认保留 `R1 常规道路 / 同向可行驶道路`；但已由 RGB/XODR 复核确认的高速/合流主类
  可以显式删除 R1。`EnterActorFlow*`、`HighwayExit`、`MergerIntoSlowTrafficV2`
  全量逐帧 RGB 未见稳定真实灯控路口，背景默认 R3 且不开放 R4；`HighwayCutIn` 与
  `MergerIntoSlowTraffic` 主体仍是 R3，但存在少量真实灯控子集，因此保留 R4 候选，
  R4 只由逐帧 RGB/meta/bbox 灯控同源证据触发。
- 除无信号灯/信号灯失效/路权类 scenario 外，每个 scenario 默认可包含 `R4 信号灯路口`。
  但若全量 RGB 已确认没有稳定信号灯路口，帧级 meta 的 `traffic_light_state` /
  `light_hazard` 不再动态加入 R4。
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
| R3 | 高速 / 合流 / 匝道 / 分流 / 驶出决策结构 | 高速或快速路主路、主辅路、匝道、合流、并线、驶出；在明确高速/merge scenario 中 R3 是默认道路空间，是否允许 R4 取决于 RGB 是否存在真实灯控段 |
| R4 | 信号灯路口 | 红绿灯正常可用，红绿灯是主通行规则 |
| R5 | 无信号灯 / 信号灯失效路口 | 无灯、灯失效、或主要按路权/安全间隙通行 |
| R6 | 路边停车 / 停车占道道路 | 停车带、路边停车、停车位汇入、开门、停车遮挡主导决策 |

## 3. 每个 scenario 的 ROAD_STRUCTURE 候选

| Scenario | ROAD_STRUCTURE 候选 | 说明 |
|---|---|---|
| Accident | R1, R4, R5 | 同向静态障碍只进 EVENT；route 前后若 RGB/meta 显示 STOP/无灯路口则允许 R5 |
| AccidentTwoWays | R1, R2, R4, R5 | 核心为双向单车道借对向绕障；route 前后 STOP/无灯 T/十字路口由 R5/R-E5 表达 |
| BlockedIntersection | R1, R4, R5 | 跟车背景 + 灯控/无灯阻塞路口；阻塞只进 EVENT，RS 由信号灯 vs STOP/无灯控制源决定 |
| ConstructionObstacle | R1, R4, R5 | 施工障碍只进 EVENT；真实 STOP/无灯路口段允许 R5 |
| ConstructionObstacleTwoWays | R1, R2, R4, R5 | 核心为双向单车道借对向绕施工障碍；真实 STOP/无灯路口段允许 R5 |
| ControlLoss | R1, R4, R5 | 失控/跟车本身不改 RS；但全量 RGB 复核发现 STOP/无灯路口片段，R5 只由 STOP/yield/meta junction/XODR 同源证据触发 |
| CrossingBicycleFlow | R1, R4 | 默认直道 + 信号灯路口；核心为自行车横穿 |
| CrossJunctionDefectTrafficLight | R1, R5 | 默认直道 + 信号灯失效路口；不放 R4 |
| DynamicObjectCrossing | R1, R4, R5 | 动态对象横穿不直接定义 RS；但 RGB/meta/bbox 显示 STOP/无灯路口时必须允许 R5，稳定灯控时 R4 |
| EnterActorFlow | R3 | 高速/快速路进入车流；全量逐帧 RGB 未见稳定真实灯控路口，不开放 R1/R4 |
| EnterActorFlowV2 | R3 | 与 EnterActorFlow 同候选；不开放 R1/R4 |
| HardBreakRoute | R1, R3, R4, R5 | 急刹是 EVENT；道路可能是城市/乡村 R1、高速/快速路 R3，也会经过 STOP/无灯 T/十字路口 R5 |
| HazardAtSideLane | R1, R4, R5 | 侧向危险只进 EVENT；真实 STOP/无灯路口段允许 R5 |
| HazardAtSideLaneTwoWays | R1, R2, R4, R5 | 核心为双向单车道侧向危险绕行；真实 STOP/无灯路口段允许 R5 |
| HighwayCutIn | R3, R4 | 主体仍是高速/快速路切入；9715-route 全量 RGB 发现少量真实灯控子集，R4 只由逐帧 RGB/meta/bbox 灯控证据触发 |
| HighwayExit | R3 | 高速驶出/分流场景；RGB 为高速/快速路/分流背景，不开放 R1/R4 |
| InterurbanActorFlow | R1, R3, R5 | 左变道/进入车流 + 无信号/STOP 路口寻找时机；2026-07-04 全量逐帧 RGB 审计未见稳定信号灯路口，删除 R4 |
| InterurbanAdvancedActorFlow | R1, R5 | RGB 为 STOP/让行/无灯城际路口，未见稳定信号灯路口；若视频显示合流再加 R3 |
| InvadingTurn | R1, R2, R4, R5 | 默认直道 + 双向窄路/对向侵入 + 路口；2026-07-05 RGB 复核发现 Town12 稳定信号灯子集，R4 只由稳定 meta/bbox/RGB 灯控证据触发 |
| MergerIntoSlowTraffic | R3, R4 | 主体是高速/快速路合流进入慢速车流；全量 RGB 发现少量真实灯控子集，R4 只在灯控同源证据成立时打开 |
| MergerIntoSlowTrafficV2 | R3 | 同属高速合流，但全量 RGB 未发现稳定灯控子集；不开放 R1/R4，不继承 MergerIntoSlowTraffic 的少量 R4 |
| NonSignalizedJunctionLeftTurn | R1, R5 | 明确无信号灯左转；不放 R4 |
| NonSignalizedJunctionLeftTurnEnterFlow | R1, R5 | 明确无信号灯左转进入车流；不放 R4 |
| NonSignalizedJunctionRightTurn | R1, R4, R5 | 大多数是 STOP/无灯右转，但全量 RGB 发现少量灯控右转子集；R4/R5 必须按逐帧 RGB + meta/bbox 控制源区分 |
| noScenarios | R1, R4, R5 | 默认普通道路；稳定灯态+bbox 灯+路口窗口可召回 R4，STOP/无灯控制证据可召回 R5；弱 XODR hint 仍保守 R1 |
| OppositeVehicleRunningRedLight | R1, R4 | 信号灯正常但对方违规 |
| OppositeVehicleTakingPriority | R1, R4, R5 | 以 STOP/让行/无灯 priority 路口为主，但全量 RGB 有少量灯控子集；R4 需要有效灯态或 RGB/bbox 灯控确认 |
| ParkedObstacle | R1, R4, R5 | 停放障碍只进 EVENT；真实 STOP/无灯路口段允许 R5，parked 本身不等于 R6 |
| ParkedObstacleTwoWays | R1, R2, R4, R5 | 核心为双向单车道借对向绕停放障碍；真实 STOP/无灯路口段允许 R5，parked 本身不等于 R6 |
| ParkingCrossingPedestrian | R1, R4, R5, R6 | 停车区域/路边行人横穿进 EVENT；真实灯控路口 R4，STOP/无灯路口 R5，停车/路边空间 R6 |
| ParkingCutIn | R1, R4, R5, R6 | 停车车辆动态切入进 EVENT；普通路段 R1，灯控路口 R4，STOP/无灯路口 R5，停车带/路边停车空间 R6 |
| ParkingExit | R1, R4, R6 | 默认直道 + 信号灯路口；核心为从停车区域并入主路 |
| PedestrianCrossing | R1, R4, R5 | 用户调研写“信号灯看情况有无”，保留 R4/R5 |
| PriorityAtJunction | R1, R4, R5 | 全量逐帧 RGB 同时存在灯控城市十字路口与无灯/让行段；保留 R4/R5 |
| RedLightWithoutLeadVehicle | R1, R4 | 明确信号灯路口 |
| SignalizedJunctionLeftTurn | R1, R4 | 明确信号灯左转 |
| SignalizedJunctionLeftTurnEnterFlow | R1, R4 | 明确信号灯左转进入车流 |
| SignalizedJunctionRightTurn | R1, R4 | 明确信号灯右转 |
| StaticCutIn | R1, R3, R4, R5, R6 | 可能混合初始目标变道、匝道/合流、停车区切入；RGB 复核发现连续 STOP/无灯路口子段，允许 R5 |
| T_Junction | R1, R4, R5 | T 形路口可为灯控或无灯/STOP；R4/R5 按逐帧 RGB + meta/bbox 控制源区分 |
| VehicleOpensDoorTwoWays | R1, R2, R4, R5, R6 | 双向单车道 + 路边停车/开门风险；route 前后真实 STOP/无灯路口段允许 R5 |
| VehicleTurningRoute | R1, R4, R5 | 和 PedestrianCrossing 类似，转弯后横穿对象/冲突，保留 R4/R5 |
| VehicleTurningRoutePedestrian | R1, R4, R5 | 和 PedestrianCrossing 类似，转弯后行人/自行车横穿，保留 R4/R5 |

## 4. 每个 ROAD_STRUCTURE 下的 EVENT 候选

这里先从道路规则空间出发，用排除法去掉肯定不属于该空间的事件。

| ROAD_STRUCTURE | 保留 EVENTS | 排除逻辑 |
|---|---|---|
| R1 常规道路 / 同向可行驶道路 | R-E1, R-E2, U-E1, U-E2, U-E3, U-E4 | 不放 R-E3/R-E4/R-E5；不放 U-E5，因为对向侵占属于 R2；不放 U-E6/U-E7/U-E8，因为它们是路口/阻塞类；非路口命中 U-E 时默认不叠 R-E1 |
| R2 双向单车道 / 借对向车道道路 | R-E1, R-E2, U-E2, U-E5 | U-E2 表达借对向绕障核心；TwoWays 中 R2 借道核心本身就是强 U-E2 证据，不等绕完后才补标；R-E2 只用于绕障后回原/目标车道；不放 U-E3/U-E6/U-E7/U-E8 |
| R3 高速合流 / 匝道 / 分流 / 驶出决策结构 | R-E1, R-E2, R-E3 | 默认无非常规；HighwayCutIn 先按常规速度匹配/跟车/自车目标变道处理，只有人工回灌明确突发切入时再加入 U-E3 |
| R4 信号灯路口 | R-E4, U-E4, U-E6, U-E8 | 不放 R-E5/U-E7，因为信号灯正常；不放 U-E5，因为对向侵占属于 R2；不放 U-E2/U-E3，普通静态绕障或动态切入不作为 R4 核心事件 |
| R5 无信号灯 / 信号灯失效路口 | R-E5, U-E4, U-E6, U-E7, U-E8, U-E5 | 不放 R-E4；U-E6 只给 CrossJunctionDefectTrafficLight 这类四向车辆冲突补充，不作为普通无灯路口默认候选；U-E5 仅允许 InvadingTurn；不放 U-E2/U-E3 |
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
| Accident | R-E1, R-E2, R-E5, U-E2 | U-E2 覆盖静态事故障碍占道核心；为绕障离开原车道也归 U-E2；若释放后仍有 route 中心线横向偏移且未开始回正，可延长 1-3 帧；绕过后结合 `signed_dist_to_lane_change` 与 ego-frame route 中心线收敛切 R-E2，到达原/目标中心线后回常规事件；STOP/无灯路口 regular 为 R-E5 |
| AccidentTwoWays | R-E1, R-E2, R-E5, U-E2 | 先 U-E2 借对向绕障，R2 借道核心本身可触发并保留 U-E2；再按轨迹和局部中心线回归切 R-E2；R-E2 到达中心线后回 R-E1/R-E4/R-E5；STOP/无灯路口 regular 为 R-E5；正常对向来车等待不等于 U-E5 |
| BlockedIntersection | R-E1, R-E4, U-E1, U-E8 | 红灯/正常队列只 R-E4；前车突然刹停短 span 可 U-E1；路口空间阻塞/解除核心可 R-E4+U-E8 |
| ConstructionObstacle | R-E1, R-E2, R-E5, U-E2 | 同 Accident，施工静态障碍核心与避障离道为 U-E2，释放过早时按 route 中心线横向偏移短补 U-E2，恢复目标车道 R-E2；STOP/无灯路口 regular 为 R-E5；R-E2 不拖过中心线回正点 |
| ConstructionObstacleTwoWays | R-E1, R-E2, R-E5, U-E2 | 同 AccidentTwoWays，先 U-E2 再 R-E2，之后按中心线回正释放为常规事件；STOP/无灯路口 regular 为 R-E5 |
| ControlLoss | R-E1, R-E4, R-E5 | 失控/低速/急刹不直接改变 EVENT taxonomy；当前 RS 为灯控路口走 R-E4，无灯/STOP 路口走 R-E5，其余 R-E1 |
| CrossingBicycleFlow | R-E1, R-E4, U-E4 | 信号灯路口自行车横穿；U-E4 起点必须等自行车进入可见/可交互范围，可与 R-E4 双触发但不能全路口持续 |
| CrossJunctionDefectTrafficLight | R-E1, R-E5, U-E6, U-E7 | 信号灯/路口规则失效按 R5；有效路口阶段可 R-E5+U-E7，四向车辆冲突明显时可叠 U-E6 |
| DynamicObjectCrossing | R-E1, R-E4, R-E5, U-E3, U-E4 | 动态对象横穿不直接改变 RS；STOP/无灯路口为 R-E5，灯控路口为 R-E4；车辆/动态切入短 span 可 U-E3，行人/骑行者/小动态对象横穿交互短 span 可 U-E4 |
| EnterActorFlow | R-E1, R-E3 | 自车主动进入车流，非他车切入；RGB 未见真实 R4 |
| EnterActorFlowV2 | R-E1, R-E3 | 同 EnterActorFlow |
| HardBreakRoute | R-E1, R-E4, R-E5, U-E1 | 前车急刹；信号灯片段按 R-E4，STOP/无灯路口按 R-E5；U-E1 只贴近真实急刹窗口 |
| HazardAtSideLane | R-E1, R-E2, R-E5, U-E2 | 行人/障碍视作静态占道；若起点附近未发生变道/避让则不硬触发 U-E2；避障离道为 U-E2，回归目标车道才 R-E2，中心线回正后释放；STOP/无灯路口 regular 为 R-E5 |
| HazardAtSideLaneTwoWays | R-E1, R-E2, R-E5, U-E2 | 双向单车道侧向危险绕行，参考 AccidentTwoWays；R-E2 边界按局部 route 中心线复核；STOP/无灯路口 regular 为 R-E5 |
| HighwayCutIn | R-E1, R-E2, R-E3, R-E4 | 侧方车进入主路先按常规跟车/速度匹配处理；自车若轨迹主动换道可 R-E2；少量灯控子集可走 R-E4；默认不开放 U-E3 |
| HighwayExit | R-E1, R-E2, R-E3 | 高速驶出与目标导向变道；无核心突发 |
| InterurbanActorFlow | R-E1, R-E2, R-E5 | 先左侧目标变道，再无信号/路权路口寻找时机左转；无非常规 |
| InterurbanAdvancedActorFlow | R-E1, R-E5 | 主要为无灯/STOP 路口寻找时机左转；RGB 未见稳定 R4 |
| InvadingTurn | R-E1, R-E4, R-E5, U-E5 | 对向车辆在 R2/R4/R5 空间异常侵占自车道；有稳定灯控证据时 RS 可为 R4，但 U-E5 仍表达对向侵入；不加入 U-E2 |
| MergerIntoSlowTraffic | R-E1, R-E3, R-E4 | 自车主动合流进入慢速车流；少量灯控子集可走 R-E4，但必须先由逐帧证据判成 R4 |
| MergerIntoSlowTrafficV2 | R-E1, R-E3 | 同属高速合流但无稳定灯控子集；不开放 R-E4 |
| NonSignalizedJunctionLeftTurn | R-E1, R-E5 | 无信号灯左转；不放 R-E4 |
| NonSignalizedJunctionLeftTurnEnterFlow | R-E1, R-E5 | 无信号灯左转进入车流，仍按 R-E5 主导 |
| NonSignalizedJunctionRightTurn | R-E1, R-E4, R-E5 | 大多数无信号灯右转走 R-E5；少量灯控右转子集按 R-E4 |
| noScenarios | R-E1, R-E4, R-E5 | 默认正常行驶；稳定灯控路口 R-E4，STOP/无灯路口 R-E5；无核心突发 U-E |
| OppositeVehicleRunningRedLight | R-E1, R-E4, U-E6 | 信号灯正常但对向车闯红灯 |
| OppositeVehicleTakingPriority | R-E1, R-E4, R-E5, U-E7 | 以无信号灯/路权失效式让行场景 R-E5+U-E7 为主；少量灯控子集可走 R-E4；不是 U-E6 违规闯灯 |
| ParkedObstacle | R-E1, R-E2, R-E5, U-E2 | 静态停放障碍绕行；避障离道与绕过核心为 U-E2，同向绕过后回目标车道切 R-E2，中心线回正后退出；STOP/无灯路口 regular 为 R-E5 |
| ParkedObstacleTwoWays | R-E1, R-E2, R-E5, U-E2 | 双向单车道借对向绕停放障碍；先 U-E2 后 R-E2，之后按中心线回正释放为常规事件；STOP/无灯路口 regular 为 R-E5 |
| ParkingCrossingPedestrian | R-E1, R-E4, R-E5, U-E4 | 进入路口前/停车遮挡区域行人横穿；U-E4 只覆盖横穿交互短 span，通过后回当前 RS 的 regular event，STOP/无灯路口为 R-E5 |
| ParkingCutIn | R-E1, R-E4, R-E5, U-E3 | 停车车辆启动/切入当前路径为 U-E3；若当前帧已进入灯控/无灯路口，则 regular 事件随 RS 切为 R-E4/R-E5，U-E3 不在 R4/R5 候选池持续 |
| ParkingExit | R-E1, R-E2, R-E4 | 自车从停车区域汇入主路，目标导向变道/汇入 |
| PedestrianCrossing | R-E1, R-E4, R-E5, U-E4 | 信号灯看 RGB/meta 分 R4/R5；U-E4 只覆盖实际让行行人 span |
| PriorityAtJunction | R-E1, R-E4, R-E5 | 混合灯控/无灯路权路口，按 RGB/meta 分帧 |
| RedLightWithoutLeadVehicle | R-E1, R-E4 | 正常红灯等待/绿灯通过并入 R-E4 |
| SignalizedJunctionLeftTurn | R-E1, R-E4 | 信号灯左转 |
| SignalizedJunctionLeftTurnEnterFlow | R-E1, R-E4 | 信号灯左转进入车流；仍按 R-E4 主导 |
| SignalizedJunctionRightTurn | R-E1, R-E4 | 信号灯右转 |
| StaticCutIn | R-E1, R-E2, R-E3, R-E4, R-E5, U-E3 | 起步目标变道 + 可能合流/停车区 + 后半段动态切入；STOP/无灯路口 regular 为 R-E5 |
| T_Junction | R-E1, R-E4, R-E5 | 灯控 T 路口走 R-E4，无灯/STOP/yield T 路口走 R-E5 |
| VehicleOpensDoorTwoWays | R-E1, R-E2, R-E5, U-E2 | 开门/停车静态占道风险 + 双向单车道绕行；参考 AccidentTwoWays；STOP/无灯路口 regular 为 R-E5 |
| VehicleTurningRoute | R-E1, R-E4, R-E5, U-E4 | 转弯/驶出路口后自行车或横穿对象进入路径；U-E4 不能在转弯前过早触发 |
| VehicleTurningRoutePedestrian | R-E1, R-E4, R-E5, U-E4 | 转弯/驶出路口后行人横穿；U-E4 不能在对象可见前过早触发 |

## 6. 关键排除规则

这些规则用于后续写脚本或 prompt 时保持一致：

- `U-E5 对向车辆异常侵占自车道` 只给 InvadingTurn；primary RS 可是 R2，也可是在 R4/R5 路口窗口内。
- `U-E7 信号灯故障 / 路口规则失效` 只给 R5，当前给 CrossJunctionDefectTrafficLight 和 OppositeVehicleTakingPriority。
- `U-E6 违规车辆冲突` 主要是信号灯正常但对方违规，当前只给 OppositeVehicleRunningRedLight。
- `CrossJunctionDefectTrafficLight` 是例外：灯/规则失效的路口可在 R5+U-E7 基础上叠 U-E6 表达四向车辆冲突。
- `U-E8 前方道路暂时阻塞 / 阻塞解除` 当前只给 BlockedIntersection。
- `R-E5 无信号灯路口通行` 只给无信号/路权/灯故障类 scenario 或混合场景中的无灯控制帧。
- `R-E4 信号灯路口通行` 给默认信号灯配置、明确信号灯 scenario，以及全量 RGB 证明存在灯控子集的混合 scenario；
  对无灯/灯故障帧本身不输出 R-E4。
- `NonSignalizedJunctionLeftTurn` 与 `NonSignalizedJunctionLeftTurnEnterFlow` 是 strict no-R4/no-R-E4。
  即使 bbox 或静态 XODR 弱提示报 `traffic_light`，只要没有有效 `traffic_light_state` 且同帧有
  STOP/yield/无灯路口证据，就保持 R5/R-E5 并写 review，不动态打开 R4。
- `R-E6` 已取消，不出现在候选表中。
- TwoWays 场景中的正常对向来车等待不等于 U-E5；只有对向车异常侵占自车道才是 U-E5。
- 障碍/TwoWays 场景不要输出 `R-E1+U-E2` 这类非路口叠加；为绕障离开原车道与核心绕行都用 U-E2，
  回原/目标车道用 R-E2，R-E2 完成后回常规事件。
- `U-E2/U-E3` 只允许在非路口候选池中触发；一旦 RS 切到 R4/R5，EVENT 必须随场景变成
  `R-E4/R-E5` 或路口专属 U-E，不能把等红灯/路口排队继续保持为 `U-E2/U-E3`。
  EVENT 后处理末尾强制执行这个候选池约束，防止桥接/单核心规则又把路口帧改回 U-E2/U-E3。
  所以 R4/R5 primary 必须严格：同向障碍/默认/noScenarios 场景只有 meta/bbox 灯态与 strong control context
  同源时才升 R4；远处/瞬时 `traffic_light` 不能压掉 R1/U-E2。
  若需要从过度保守的 R1 恢复 R4，必须走 route 级 `r4_context_recovery`：
  连续灯态/bbox traffic_light 不少于 4 帧，并且有 strong control context、close trigger
  或 bbox junction hint；弱 `near_junction` / 宽 `junction_window` 只能 review。
- 同一 route 的 `U-E2/U-E3` 是一次性核心 span，但保留时不能只选最长段。
  `U-E2` 必须优先保留具体静态障碍距离、绕障/回正轨迹和 route 中心线偏离证据最强的段；
  前方运动车辆跟车距离、普通减速或红灯等待不能抢占唯一 U-E2 名额。`U-E3` 必须优先保留
  `dist_to_cutin_vehicle`、`brake_cutin`、`vehicle_hazard` 与对象进入自车未来路径最强的段；
  普通跟车减速或路口等待不能抢占唯一 U-E3 名额。
- `U-E2/U-E3` 后若接入恢复/目标变道 `R-E2`，中间 4 帧以内的短 `R-E1` 统一改成
  `R-E2`，避免 `U-E2/U-E3 -> R-E1 -> R-E2` 的断裂监督。
  障碍恢复类场景的 `U-E2 -> R-E1 -> R-E2` 桥接可放宽到 8 帧；但 R-E2 必须贴近最近 U-E2，
  离最近 U-E2 超过约 24 帧的后段 R-E2 视为弯道/跟车/投影扰动并回 regular event。
  `dist_to_*` 阈值只能定位进入核心，不能在最近点之后继续粘住 U 类事件：非 TwoWays U-E2
  过障碍最近点后按距离回升 + 中心线回正释放；U-E3 过 cut-in 最近点后按距离回升 + 无持续强响应释放。
- R-E2 的触发/退出统一借鉴局部中心线复核：用 `meta["route"]` ego-frame 近前方 1-8m
  点拟合局部切线，计算 ego 到目标中心线的横向误差。只有中心线偏离且
  `changed_route` / `signed_dist_to_lane_change` / 足够 offset 支撑时才认为目标导向变道 active；
  障碍恢复段可在过障碍核心后用 `signed_dist_to_lane_change <= -0.45m` 和中心线收敛趋势
  提前 1-2 帧触发。Town06 等弯道中 `route_lateral_abs_m` 可能因为局部 route 贴车而过小；
  若 `changed_route` 与 `signed_dist_to_lane_change` 仍连续有效，不允许仅凭
  `route_centered=true` 把恢复段释放成 R-E1。post-U2 R-E2 只能由长度足够的 U-E2
  核心 span 开启，1-3 帧 trigger 抖动不能触发恢复窗口；进入恢复 R-E2 后 4 帧以内的
  U-E2 反跳合入 R-E2，回到中心线容差且 signed lane-change 结束并稳定后退出。
  该规则用于障碍恢复、HighwayCutIn/HighwayExit、
  InterurbanActorFlow、ParkingExit、StaticCutIn 等 R-E2 候选场景；不用于判定 R4/R5 控制源或
  单独触发其它 U-E。同一个常规 R-E 前后夹住的 1-2 帧孤立 R-E2 视为中心线/flag 抖动，
  平滑回前后常规事件。
- `HighwayCutIn` 默认不开放 U-E3；若后续 RGB/轨迹全量复核确认有真实突发切入，再单独回灌白名单。
- `ParkingCutIn` / `StaticCutIn` 的 U-E3 仍保留给真实动态占道；但如果已进入 R-E2 目标/恢复变道，
  之后 4 帧以内短暂回 U-E3，或中间只夹 1-2 帧常规事件再回 U-E3，统一合入 R-E2。
  cut-in 最近点之后对象已远离且没有 `brake_cutin` / `vehicle_hazard` 时，不再保持 U-E3。

## 7. 后续实现建议

后续脚本可以按三段式构造候选：

1. 读取 scenario，查第 3 节得到 ROAD_STRUCTURE 候选。
2. Qwen step1 从 ROAD_STRUCTURE 候选中选当前帧规则空间。
3. 根据第 4 节取该 ROAD_STRUCTURE 的事件候选，再与第 5 节的 scenario 精细事件候选求交集。

本轮 ROAD_STRUCTURE 5-id/town 调研后，EVENT 候选还必须遵守以下依赖：

- EVENT 不反推 RS。`HardBreakRoute`、`ControlLoss`、`DynamicObjectCrossing`、
  `BlockedIntersection`、`OppositeVehicleRunningRedLight` 的异常只进入 EVENT/span，
  primary RS 仍由 XML/XODR/meta 的道路结构证据决定。
- `TwoWays` 只让候选池包含 R2/U-E2，不代表全程 R2；R2 只覆盖必须借/等对向的核心障碍 span。
  meta 可用时允许 XML trigger 极近或 trigger-close + XML 场景障碍近距离辅助召回短核心 R2；
  缺 meta 或 XML 的 run 直接 `data_missing_skip`，不进入候选判定；
  绕过障碍后重新按 XODR/meta 判 R1/R4。
- `Parking*` 只让候选池包含 R6/停车区事件，不代表全程 R6；灯控路口段仍优先 R4。
- `R4/R5` 路口/T 形路口必须是一段连续窗口。4Hz 下少于 4 帧的 R4/R5 即使有瞬时灯态、
  bbox traffic_light 或 XODR junction hint，也按扰动并回邻近稳定 RS。
- `EnterActorFlow*`、`HighwayExit`、`MergerIntoSlowTrafficV2` 是稳定无灯控的高速/快速路背景场景，
  候选池不开放 R1/R4；EVENT 按 R3 高速/合流规则空间收窄。`HighwayCutIn` 与
  `MergerIntoSlowTraffic` 在 2026-07-04 全量逐帧 RGB 审计中发现少量真实灯控子集，因此恢复 R4
  候选，但 R4 必须由逐帧 RGB/meta/bbox 灯控证据触发，匝道/导流线/停车线不能单独制造 R4/R5。
- 混合场景不能只按 Town12/13 判高速。`HardBreakRoute`、`InterurbanActorFlow*`、
  `StaticCutIn`、`ParkingCutIn` 必须先按 route RGB 分桶；高速/快速路桶候选收敛为 R3/R4，
  非高速桶保留 R1。当前逐 id 均匀 5 帧 RGB 复核结果：HardBreakRoute 16/97 与 StaticCutIn 44/100
  进入高速桶；InterurbanActorFlow、InterurbanAdvancedActorFlow、ParkingCutIn 高速桶为空。
  `Town12_Rep0_258_0_route0_01_08_09_35_42` 是 HardBreakRoute 非高速反例。
- `CrossJunctionDefectTrafficLight` 的 RS 由 defect 机制强制 R5；EVENT 选择 U-E7/R-E5 时不要再被
  XODR signal 存在性拉回 R4。执行时 `primary_event` 保持 U-E7；四向车辆冲突明显时把
  U-E6 作为同帧 secondary event 叠加，不让 U-E6/U-E7 在 primary 上来回抢占。
- 当前生成入口 `quick_start.py annotate-rs` 会同时输出 RS 与 EVENT。人工/训练读取时应使用
  `frame_rs_annotation.label` 和 `frame_event_annotation.label` 作为本帧最终标签；
  `road_structures/events` 是候选/集合字段，不应当作“单帧唯一标签”直接显示给人看。
- Web 页面和 summary 使用 `road_structure_labels/event_labels` 解释代号含义；新增或修改候选时，
  必须同步维护这两个词典，避免可视化验收时只看到 R1/U-E2 等裸代号。

伪代码：

```text
road_candidates = SCENARIO_TO_ROAD_STRUCTURE[scenario]
road = qwen_select_road_structure(frame, road_candidates)

event_candidates = ROAD_STRUCTURE_TO_EVENTS[road] ∩ SCENARIO_TO_EVENTS[scenario]
events = qwen_select_primary_event(frame, event_candidates)
if road in {R4, R5} and junction_unusual_span_visible(frame):
    events = merge_regular_junction_event_with_unusual_event(events)
```

这样可以最大程度保证白名单约束，同时通过 ROAD_STRUCTURE 与 scenario 双重排除，把肯定不存在的事件挡在候选表外。
