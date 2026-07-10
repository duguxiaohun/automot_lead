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
R4/R5 本身是 route 级单控制源：同一条 route 不允许 `R4 -> R5` 或 `R5 -> R4` 互跳；
若同时出现，按有效灯态或稳定可见红绿灯锁为 R4，否则锁为 R5。该锁只同步 regular event
（R4->R-E4、R5->R-E5），不改变 U-E4/U-E6/U-E7/U-E8 等非常规事件触发逻辑。
2026-07-09 复算 24 个 R4/R5 候选场景：旧结果 mixed route=759、direct jump route=698；
锁定后 mixed=0、direct=0。实际重跑 5 个高风险大场景 1405 route / 210072 frame 也为 0 / 0。
场景级 EVENT 表只是该 scenario 的上限；最终候选还必须再和当前 ROAD_STRUCTURE 的候选池取交集，
并始终保留当前 RS 的 regular event。也就是说 R4/R5 路口帧会直接删除 `U-E2/U-E3`，
红灯等待、路口排队、路口起步只能走 `R-E4/R-E5` 或路口专属 U-E。
当前可执行实现还在 route 级 EVENT 后处理之后再次执行最终 clamp：
`final_events = scenario_fine_events ∩ current_primary_rs_events ∪ current_rs_regular_event`。
如果桥接、单核心选择或恢复段规则写出了当前 RS 候选池外的事件，会回退到当前 RS 的 regular event
并在 `event_evidence.allowed_events` / `event_candidate_clamp` 中记录。例外只有两类：
`AccidentTwoWays` 的 R2 overlay 可在 R4/R5 边界保留 `U-E2/R-E2`；
`InvadingTurn` 的对向侵占可在 R4/R5 边界保留 `U-E5`。
新增 interrupted overlay 是第三类受控例外：只有当非路口 `U-E1/U-E2/U-E3/U-E4` 或静态障碍 `U-E2 -> R-E2` 恢复链被 R4/R5
突然接管且 evidence 显示突发动作或回正尚未自然结束时，才允许同帧保留
`R-E4/R-E5 + U-E*` 或 `R-E4/R-E5 + R-E2`；总时长上限 24 帧，恢复 `R-E2` 子阶段上限 12 帧。
若 R4/R5 接管首帧恢复证据不足，最终候选池收口可用最近 8 帧 U-E2/U-E3 source + 当前回正/回车道证据补回 `R-E2` overlay。
U-E4 的中距离横穿/转弯冲突只短续 10 帧，避免把普通 R4/R5 长期污染为突发事件。
这类 overlay 不能只靠 `secondary_road_structures` 表达，因为普通 secondary 也可能来自候选冲突或不确定性。
程序化判断必须读取专用 `road_structure_overlay.active=true` / `frame_rs_annotation.overlay.active=true`：
`base_road_structure` 表示被截断突发事件原本所属的 R1/R2 视角，
`intersection_road_structure` 表示当前 primary R4/R5 控制源；Web 可视化单独显示
`RS-Overlay base→intersection`。
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
- 除无信号灯/路权类 scenario 外，每个 scenario 默认可包含 `R4 信号灯路口`。
  但若全量 RGB 已确认没有稳定信号灯路口，帧级 meta 的 `traffic_light_state` /
  `light_hazard` 不再动态加入 R4。
- 明确无信号灯或按路权通过的 scenario，用 `R5 无信号灯 / 路权路口`
  替代默认 `R4`。
- R2/R3 只在 scenario 特征明确支持时额外加入；停车、开门、遮挡不再单独形成 ROAD_STRUCTURE，
  而是并入 R1/R2 后由 EVENT 表达。非 TwoWays 的 R2 不能只靠场景级 XODR sparse scan 批量开放；
  只有逐 route RGB 已确认且进入白名单时才可动态加入，避免普通 R1/R-E1 场景被误升 R2。

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
| R5 | 无信号灯 / 路权路口 | 无灯、STOP/yield、或主要按路权/安全间隙通行 |

## 3. 每个 scenario 的 ROAD_STRUCTURE 候选

| Scenario | ROAD_STRUCTURE 候选 | 说明 |
|---|---|---|
| Accident | R1, R4, R5 | 同向静态障碍只进 EVENT；route 前后若 RGB/meta 显示 STOP/无灯路口则允许 R5 |
| AccidentTwoWays | R2, R4, R5 | 全量 RGB 复核后按有效可行驶通道口径处理：非路口默认 R2；有灯路口 R4；STOP/无灯/路权路口 R5；前 30 帧 bbox-only stop/sign 且无真实 junction/control 时回 R2，XML 事故 trigger 不单独制造 R5 |
| BlockedIntersection | R1, R4, R5 | 跟车背景 + 灯控/无灯阻塞路口；阻塞只进 EVENT，RS 由信号灯 vs STOP/无灯控制源决定；进入侧收紧为 `junction_pre_m=32`；稳定灯控 R4 尾段若仅因视角丢失灯态且无 STOP/yield，继续保持 R4 |
| ConstructionObstacle | R1, R4, R5 | 施工障碍只进 EVENT；真实 STOP/无灯路口段允许 R5；进入侧收为 `junction_pre_m=42` |
| ConstructionObstacleTwoWays | R2, R4, R5 | 非路口默认 R2；施工核心由 U-E2/R-E2 表达；真实 STOP/无灯路口段允许 R5；进入侧收为 `junction_pre_m=42` 且 factor 0.70（有效约 29.4m），施工 XML trigger 不单独制造 R5 |
| ControlLoss | R1, R4, R5 | 失控/跟车本身不改 RS；STOP/无灯路口片段由可见 junction/control 同源证据触发 R5；弱 junction/远灯召回再压约 40%；Town01-04 起始 30 帧 meta-only/XODR-only/stop-only 伪路口回 R1 |
| CrossingBicycleFlow | R1, R4 | 默认直道 + 信号灯路口；核心为自行车横穿；进入侧收为 `junction_pre_m=35` 且 factor 0.90（有效约 31.5m） |
| CrossJunctionDefectTrafficLight | R1, R4 | 默认直道 + 信号灯路口；信号灯失效是 U-E7，不改变 R4；远距离 meta/bbox 灯态不能单独提前进入 R4，必须有近路口/本地灯控证据 |
| DynamicObjectCrossing | R1, R4, R5 | 动态对象横穿不直接定义 RS；但 RGB/meta/bbox 显示 STOP/无灯路口时必须允许 R5，稳定灯控时 R4；R4/R5 触发比普通 default 场景更紧，远灯或 route 起始弱 STOP 不直接升路口 |
| EnterActorFlow | R1, R3 | 高速/快速路进入车流；远离 actor-flow/merge 起点的普通直道可为 R1，靠近汇入控制区后进入 R3；全量逐帧 RGB 未见稳定真实灯控路口，不开放 R4 |
| EnterActorFlowV2 | R1, R3 | 与 EnterActorFlow 同候选；远端直道 R1、近汇入 R3，不开放 R4 |
| HardBreakRoute | R1, R3, R4, R5 | 急刹是 EVENT；道路可能是城市/乡村 R1、高速/快速路 R3，也会经过 STOP/无灯 T/十字路口 R5 |
| HazardAtSideLane | R1, R4, R5 | 侧向危险只进 EVENT；真实灯控/STOP/无灯路口段允许 R4/R5，前 30 帧仅 bbox-only STOP、close-trigger 或 untrusted XODR 时保持 R1 |
| HazardAtSideLaneTwoWays | R2, R4, R5 | 乡路/窄路/等效对向单车道默认 R2；真实灯控/无灯路口分别 R4/R5 |
| HighwayCutIn | R3, R4 | 主体仍是高速/快速路切入；9715-route 全量 RGB 发现少量真实灯控子集，R4 只由逐帧 RGB/meta/bbox 灯控证据触发 |
| HighwayExit | R3 | 高速驶出/分流场景；RGB 为高速/快速路/分流背景，不开放 R1/R4 |
| InterurbanActorFlow | R1, R3, R5 | 左变道/进入车流 + 无信号/STOP 路口寻找时机；2026-07-04 全量逐帧 RGB 审计未见稳定信号灯路口，删除 R4 |
| InterurbanAdvancedActorFlow | R1, R5 | RGB 为 STOP/让行/无灯城际路口，未见稳定信号灯路口；junction 进入/退出窗口较此前放宽约 30%，若视频显示合流再加 R3 |
| InvadingTurn | R1, R2, R4, R5 | 默认直道 + 双向窄路/对向侵入 + 路口；2026-07-05 RGB 复核发现 Town12 稳定信号灯子集，R4 只由稳定 meta/bbox/RGB 灯控证据触发 |
| MergerIntoSlowTraffic | R3, R4 | 主体是高速/快速路合流进入慢速车流；全量 RGB 发现少量真实灯控子集，R4 只在灯控同源证据成立时打开 |
| MergerIntoSlowTrafficV2 | R3 | 同属高速合流，但全量 RGB 未发现稳定灯控子集；不开放 R1/R4，不继承 MergerIntoSlowTraffic 的少量 R4 |
| NonSignalizedJunctionLeftTurn | R1, R5 | 明确无信号灯左转；不放 R4 |
| NonSignalizedJunctionLeftTurnEnterFlow | R1, R5 | 明确无信号灯左转进入车流；不放 R4 |
| NonSignalizedJunctionRightTurn | R1, R4, R5 | 大多数是 STOP/无灯右转，但全量 RGB 发现少量灯控右转子集；R4/R5 必须按逐帧 RGB + meta/bbox 控制源区分；远处直道和驶离后直道由 RightTurn 局部核心门控恢复 R1 |
| noScenarios | R1, R3, R4, R5 | 默认普通道路；只有本地 ego-affecting/overhead/近距离 bbox 灯 + 有效灯态/路口窗口才召回 R4，STOP/yield/无灯控制源召回 R5 且压制弱灯；可信 XODR ramp/merge/split 或人工 RGB highway bucket 才开放 R3；远处单灯框、bbox-only 弱灯和弱 XODR hint 仍保守 R1 |
| OppositeVehicleRunningRedLight | R1, R4 | 信号灯正常但对方违规 |
| OppositeVehicleTakingPriority | R1, R4, R5 | 以 STOP/让行/无灯 priority 路口为主，但全量 RGB 有少量灯控子集；R4 需要有效灯态或 RGB/bbox 灯控确认；2026-07-10 RGB 边界回灌后进入侧 `junction_pre_m=75` |
| ParkedObstacle | R1, R4, R5 | 停放障碍只进 EVENT；真实 STOP/无灯路口段允许 R5；parked 本身不改变 RS；2026-07-10 RGB 边界回灌后进入侧 `junction_pre_m=72` |
| ParkedObstacleTwoWays | R2, R4, R5 | 停车/障碍占掉侧向 lane 后按有效可行驶对向单车道 R2；真实 STOP/无灯路口段允许 R5；短 `R-E2 -> R-E4/R-E5 -> R-E2` 插缝按噪音并回 R2/R-E2 |
| ParkingCrossingPedestrian | R1, R4, R5 | 停车区域/路边行人横穿进 EVENT；真实灯控路口 R4，STOP/无灯路口 R5；停车/遮挡本身并入 R1 |
| ParkingCutIn | R1, R4, R5 | 停车车辆动态切入进 EVENT；普通路段 R1，灯控路口 R4，STOP/无灯路口 R5 |
| ParkingExit | R1, R4 | 从停车区域并入主路由 R-E2 表达；道路结构仍是 R1，若进入灯控路口则 R4；初始驶出 R-E2 收尾按 RGB 提前约 5 帧释放 |
| PedestrianCrossing | R1, R4, R5 | 用户调研写“信号灯看情况有无”，保留 R4/R5；入口收紧、出口 tail 延长，同一 R4/R5 路口内 1-8 帧短 R1/R-E1 缝同步缝合 RS+EVENT |
| PriorityAtJunction | R1, R4, R5 | 混合 priority 场景：真实灯控城市路口为 R4，无灯/让行控制为 R5，前后直行段 R1；2026-07-10 保护局部有效灯控帧，route lock 不能把仍在本地灯控/stopline 区内的 R4 错压回 R1；Town13 晚触发灯控 approach 对第一段稳定 R4 最多前补 4 帧 |
| RedLightWithoutLeadVehicle | R1, R4 | 明确信号灯路口；驶离灯控区后若 trigger>52m 且无本地 junction/control，则释放回 R1 |
| SignalizedJunctionLeftTurn | R1, R4 | 明确信号灯左转 |
| SignalizedJunctionLeftTurnEnterFlow | R1, R4 | 明确信号灯左转进入车流；Town01/02 前 30 帧远灯/弱 trigger 且无本地 junction core 时回 R1 |
| SignalizedJunctionRightTurn | R1, R4 | 明确信号灯右转 |
| StaticCutIn | R1, R3, R4, R5 | 可能混合初始目标变道、匝道/合流、普通道路切入；停车侧切入由 U-E3 表达；连续 STOP/无灯路口子段允许 R5 |
| T_Junction | R1, R4, R5 | T 形路口可为灯控或无灯/STOP；R4/R5 按逐帧 RGB + meta/bbox 控制源区分；退出侧稍延迟但禁止 R4/R5 互跳 |
| VehicleOpensDoorTwoWays | R2, R4, R5 | 两侧停车/开门风险占用侧向 lane 时按有效可行驶对向单车道 R2；route 前后真实 STOP/无灯路口段允许 R5 |
| VehicleTurningRoute | R1, R4, R5 | 和 PedestrianCrossing 类似，转弯后横穿对象/冲突，保留 R4/R5；RGB 复核后收紧十字路口窗口，远灯态/trigger-only STOP hint 不能提前覆盖普通道路 |
| VehicleTurningRoutePedestrian | R1, R4, R5 | 和 PedestrianCrossing 类似，转弯后行人/自行车横穿，保留 R4/R5；非 TwoWays R2 暂停动态加入，待所有 id 逐帧 RGB 复核后再按 route 白名单开放；稳定 R4/R5 路口段夹住的短 R1 空洞按本地路口/stop-yield 证据回填，最终输出层再兜底同步 RS+EVENT 短缝 |

## 4. 每个 ROAD_STRUCTURE 下的 EVENT 候选

这里先从道路规则空间出发，用排除法去掉肯定不属于该空间的事件。

| ROAD_STRUCTURE | 保留 EVENTS | 排除逻辑 |
|---|---|---|
| R1 常规道路 / 同向可行驶道路 | R-E1, R-E2, U-E1, U-E2, U-E3, U-E4 | 不放 R-E3/R-E4/R-E5；不放 U-E5，因为对向侵占属于 R2；不放 U-E6/U-E7/U-E8，因为它们是路口/阻塞类；非路口命中 U-E 时默认不叠 R-E1 |
| R2 双向单车道 / 借对向车道道路 | R-E1, R-E2, U-E2, U-E5 | R2 只是道路/可行驶空间标签，不能单独触发或保留 U-E2/R-E2；正常沿 R2 车道保持仍是 R-E1。U-E2 必须由 XML trigger、具体障碍距离、对向绕行核心、开门/强制制动等独立事件证据触发；R-E2 只用于真实回原/目标车道过程；不放 U-E3/U-E6/U-E7/U-E8 |
| R3 高速合流 / 匝道 / 分流 / 驶出决策结构 | R-E1, R-E2, R-E3 | R3 是道路空间，不等于机械全程 R-E3；主线内正常车道保持/速度匹配为 R-E1；进入匝道、actor-flow/merge approach、分流/驶出匝道过渡保持 R-E3，不能在 R-E3 与真实变道 R-E2 之间插入 R-E1；R-E2 必须由局部 route 中心线偏离 + `changed_route` / signed lane-change 等组合证据确认，不能只凭 `signed_dist_to_lane_change` 单独触发；HighwayCutIn 仍先按常规速度匹配/跟车/自车目标变道处理，只有人工回灌明确突发切入时再加入 U-E3 |
| R4 信号灯路口 | R-E4, U-E4, U-E6, U-E8；CrossJunctionDefectTrafficLight 额外允许 U-E7 | 不放 R-E5；正常信号灯不放 U-E7，但故障信号灯场景保持 R4+U-E7；不放 U-E5，因为对向侵占属于 R2；不放 U-E2/U-E3，普通静态绕障或动态切入不作为 R4 核心事件 |
| R5 无信号灯 / 路权路口 | R-E5, U-E4, U-E6, U-E7, U-E8, U-E5 | 不放 R-E4；U-E7 主要给 OppositeVehicleTakingPriority 等无灯/路权失效；U-E5 仅允许 InvadingTurn；不放 U-E2/U-E3 |

说明：

- R-E1 是大多数非路口/非合流结构的背景事件。
- R-E2 只保留在目标导向变道、停车区汇入、同向道路目标车道调整、借道后回正等空间。
- U-E4 可以跨 R1/R4/R5，因为行人/自行车横穿可发生在直道、路口和停车遮挡区域。
- U-E8 只放 R4/R5，因为它描述前方道路/路口通行空间阻塞，而不是普通静态障碍绕行。

## 5. 每个 scenario 的精细 EVENTS 候选

本表先取该 scenario 的 ROAD_STRUCTURE 事件并集，再按 scenario 语义排除肯定不存在的事件。
因此它比第 4 节更窄，更适合给 Qwen 做第二层候选。

| Scenario | 精细 EVENTS 候选 | 排除/保留说明 |
|---|---|---|
| Accident | R-E1, R-E2, R-E5, U-E2 | U-E2 覆盖静态事故障碍占道核心；为绕障离开原车道也归 U-E2；若释放后仍有 route 中心线横向偏移且未开始回正，可延长 1-3 帧；绕过后结合 `signed_dist_to_lane_change` 与 ego-frame route 中心线收敛切 R-E2，到达原/目标中心线后回常规事件；STOP/无灯路口 regular 为 R-E5 |
| AccidentTwoWays | R-E1, R-E2, R-E5, U-E2 | 先 U-E2 借对向绕障，R2 借道核心本身可触发并保留 U-E2；再按轨迹和局部中心线回归切 R-E2；UE2 后若紧接 RE2，不允许中间夹短 R-E1，16 帧以内空洞按前段 U-E2 / 后段 R-E2 分摊；R-E2 到达中心线后回 R-E1/R-E4/R-E5；STOP/无灯路口 regular 为 R-E5；正常对向来车等待不等于 U-E5 |
| BlockedIntersection | R-E1, R-E4, U-E1, U-E8 | 红灯/正常队列只 R-E4；前车突然刹停短 span 可 U-E1；路口空间阻塞/解除核心可 R-E4+U-E8；U-E8 near-trigger 小幅放宽到约 52m，低速等待阈值为 speed<=0.9m/s+brake |
| ConstructionObstacle | R-E1, R-E2, R-E5, U-E2 | 同 Accident，施工静态障碍核心与避障离道为 U-E2，释放过早时按 route 中心线横向偏移短补 U-E2，恢复目标车道 R-E2；若 U-E2/R-E2 恢复链被 R4/R5 截断，使用 interrupted overlay 同帧保留 R-E4/R-E5 regular，最终 clamp 仍可根据最近 8 帧 U-E2 source 补回恢复 R-E2 overlay；STOP/无灯路口 regular 为 R-E5；R-E2 不拖过中心线回正点 |
| ConstructionObstacleTwoWays | R-E1, R-E2, R-E5, U-E2 | 同 AccidentTwoWays，先 U-E2 再 R-E2，之后按中心线回正释放为常规事件；STOP/无灯路口 regular 为 R-E5 |
| ControlLoss | R-E1, R-E4, R-E5 | 失控/低速/急刹不直接改变 EVENT taxonomy；当前 RS 为灯控路口走 R-E4，无灯/STOP 路口走 R-E5，其余 R-E1 |
| CrossingBicycleFlow | R-E1, R-E4, U-E4 | 信号灯路口自行车横穿；U-E4 起点必须等自行车进入可见/可交互范围，可与 R-E4 双触发但不能全路口持续；route 级只保留一段连续 U-E4，内部短 R-E4/R-E1 gap 合并；自行车流允许 14 帧以内 meta 距离短空窗 |
| CrossJunctionDefectTrafficLight | R-E1, R-E4, U-E6, U-E7 | 有红绿灯硬件/控制路口，RS 为 R4；信号失效/规则源失效用 U-E7 表达，有效路口阶段可 R-E4+U-E7，四向车辆冲突明显时可叠 U-E6 |
| DynamicObjectCrossing | R-E1, R-E4, R-E5, U-E3, U-E4 | 动态对象横穿不直接改变 RS；直道/隧道里的 U-E4 仍保持 R1/R-E1+U-E4；STOP/无灯路口为 R-E5，灯控路口为 R-E4；短弱 bbox 灯/close-trigger 不能单独升 R4，长稳定 meta+bbox 灯控段可 route 级恢复 R4；车辆/动态切入短 span 可 U-E3，行人/骑行者/小动态对象横穿交互短 span 可 U-E4；U-E4 route 级单段连续化 |
| EnterActorFlow | R-E1, R-E2, R-E3 | 自车主动进入车流，非他车切入；RGB 未见真实 R4；远离 actor-flow/merge 起点的直道保持 R-E1，靠近汇入控制区后保持 R-E3，R-E3 回填上限约 36 帧且至少保留约 16 帧准备汇入段；真实目标变道由 R-E2 表达，不能被 R-E3 吞掉；围绕已有 R-E2 核心按 `changed_route + signed_dist_to_lane_change` 补完整变道轨迹，避免只标横向峰值一两帧；merge 完成后的主线正常跟车才回 R-E1 |
| EnterActorFlowV2 | R-E1, R-E2, R-E3 | 同 EnterActorFlow |
| HardBreakRoute | R-E1, R-E4, R-E5, U-E1 | 前车急刹；信号灯片段按 R-E4，STOP/无灯路口按 R-E5；U-E1 只贴近真实急刹窗口 |
| HazardAtSideLane | R-E1, R-E2, R-E4, R-E5, U-E4 | XML 与逐帧 RGB 均确认核心对象为自行车/行人横穿或侧向进入自车路径，不按静态障碍输出 U-E2；`dist_to_biker<=30m` 的可交互段连续标 U-E4，对象离开后 8 帧内若仍有目标车道/横向回正证据，必须从 U-E4 后第一帧直接切 R-E2，不能夹 R-E1；若此时 RS 已被 R4/R5 接管，保留 `R-E4/R-E5 + R-E2` interrupted overlay，中心线回正后释放 |
| HazardAtSideLaneTwoWays | R-E1, R-E2, R-E4, R-E5, U-E4 | R2 只表达有效可行驶对向单车道路结构；侧向自行车/行人进入路径时仍输出 U-E4，离开对象后借对向/横移回正阶段直接切 R-E2；R4/R5 到来不能吞掉该恢复动作，使用 regular+R-E2 overlay；起始 30 帧仅 bbox STOP、无真实 junction/control 时保持 R2 |
| HighwayCutIn | R-E1, R-E2, R-E3, R-E4 | 侧方车进入主路先按常规跟车/速度匹配处理，默认 R-E1；自车若由局部 route 中心线偏离 + lane-change 组合证据证明主动换道才 R-E2；已有 R-E2 核心可按轨迹支撑前补最多 3 帧、后补最多 4 帧；少量灯控子集可走 R-E4；默认不开放 U-E3，且不因 HighwayCutIn 场景名全程输出 R-E3 |
| HighwayExit | R-E1, R-E2, R-E3 | 高速驶出不是 trigger 圆窗事件：出口前仍在主线内正常跟车为 R-E1，真实目标导向变道为 R-E2；在原有后补 4 帧基础上累计为前补最多 2 帧、后补最多 6 帧，完成出口换道后再切 R-E3；进入驶出匝道后保持 R-E3，不能回落 R-E1；`next_commands[0]==3`、exit actor-flow 近邻和 route 级 exit transition 都可支持 R-E3 |
| InterurbanActorFlow | R-E1, R-E2, R-E5 | 先左侧目标变道，再无信号/路权路口寻找时机左转；已有 R-E2 核心按轨迹前补最多 3 帧、后补最多 4 帧，但不能跨入 R5/R-E5 路口段；仅 active+close-trigger 而没有 stop/junction 控制证据的开头直道保持 R-E1 |
| InterurbanAdvancedActorFlow | R-E1, R-E2, R-E5 | 主要为无灯/STOP 路口寻找时机左转；RGB 未见稳定 R4；若通过路口时存在 `changed_route` + 横向偏移/换道符号证据，R5 段内也允许 R-E2，并按轨迹前补最多 3 帧、后补最多 4 帧 |
| InvadingTurn | R-E1, R-E4, R-E5, U-E5 | 对向车辆异常侵占可发生在 R1/R2/R4/R5；U-E5 按 RGB 可见锥桶/对向占道长度保持：核心看 R2 侵占规则与轨迹响应，final pass 对连续 `passive_oncoming_invasion`、trigger>=35m 且有 R2 或 R1 响应证据的长 cluster 输出 U-E5，单段最多补 48 帧；R5 仍须有 STOP/yield/真实 junction 同源证据；不加入 U-E2 |
| MergerIntoSlowTraffic | R-E1, R-E2, R-E3, R-E4 | 自车主动合流进入慢速车流；刚开始/中间普通主线跟车保持 R-E1，禁用 trigger-only 圆窗直接制造 R-E3，靠近 merge/actor-flow approach 才切 R-E3，真实目标变道才 R-E2；已有 R-E2 核心按轨迹前后各最多补 5 帧，R-E2 后若 actor-flow/merge 仍近则最多 64 帧 tail 保持 R-E3；3 帧以内夹在 R-E1 中的孤立 R-E3 小岛平滑回 R-E1；少量灯控子集可走 R-E4 |
| MergerIntoSlowTrafficV2 | R-E1, R-E2, R-E3 | 同属高速合流但无稳定灯控子集；刚开始/中间普通主线跟车保持 R-E1，禁用 trigger-only 圆窗直接制造 R-E3，靠近 merge/actor-flow approach 才切 R-E3；已有 R-E2 核心前后各最多补 5 帧；R-E2 后仍处于分离汇入/匝道空间时保持 R-E3，不误回 R-E1 |
| NonSignalizedJunctionLeftTurn | R-E1, R-E5 | 无信号灯左转；R-E5 仅覆盖局部路口核心，驶离后回 R-E1；不放 R-E4；同一 R-E5 中间不超过 12 帧的 R-E1/R-E2 短缝会同步把 RS+EVENT 合并回 R5/R-E5 |
| NonSignalizedJunctionLeftTurnEnterFlow | R-E1, R-E5 | 无信号灯左转进入车流；进入窗口按 RGB 放宽到 `junction_pre_m=84`；局部核心 R-E5，进入后的直行段回 R-E1；同一 R-E5 中间不超过 12 帧的 R-E1/R-E2 短缝同步合并 RS+EVENT，`Town03 route001042` / `Town13 route001061` 开头连续保持 R5/R-E5 |
| NonSignalizedJunctionRightTurn | R-E1, R-E4, R-E5 | 大多数无信号灯右转核心走 R-E5，驶离后回 R-E1；进入窗口放宽到 `junction_pre_m=63`，但仍用局部核心门控压掉远处直道；少量灯控右转子集按 R-E4；同一 R-E4/R-E5 中间不超过 12 帧的 R-E1/R-E2 短缝同步合并 RS+EVENT |
| noScenarios | R-E1, R-E2, R-E3, R-E4, R-E5 | 默认正常行驶；局部目标车道变化由 `changed_route` / `signed_dist_to_lane_change` + route 横向偏移输出 R-E2，并做 1 帧短缝合并和少量轨迹补边；可信 R3 merge/split 核心可输出 R-E3；本地稳定灯控路口 R-E4，STOP/yield/无灯路口 R-E5；无核心突发 U-E |
| OppositeVehicleRunningRedLight | R-E1, R-E4, U-E6 | U-E6 在 R4 窗口内表达违规车辆冲突；按冲突车、近距离对象、bbox/RGB 横穿或对向动态车辆、自车停车/让行响应保留等待上下文，输出必须是 `R-E4 + U-E6` 同帧叠加；同 route 多段横向车候选优先保留导致自车停车/等待的 span，再按 bbox 冲突帧数和长度排序；2026-07-10 全量验证 U-E6 从上一版 3515 扩为 5744 帧，R4 route 完全无 U-E6 从 126 条降到 9 条，冲突解除后回 R-E4 |
| OppositeVehicleTakingPriority | R-E1, R-E4, R-E5, U-E7 | 确有 12 条有效灯态 route；有灯 route 锁 R4，无灯 route 锁 R5，禁止互切；不是 U-E6；进入侧 `junction_pre_m=75` |
| ParkedObstacle | R-E1, R-E2, R-E4, R-E5, U-E2 | 静态停放障碍绕行；U-E2 后恢复 R-E2 起点提前最多 3 帧、终点提前最多 4 帧；灯控 regular 为 R-E4，STOP/无灯 regular 为 R-E5 |
| ParkedObstacleTwoWays | R-E1, R-E2, R-E4, R-E5, U-E2 | 双向单车道借对向绕障；U-E2 后恢复 R-E2 起点提前最多 3 帧、终点提前最多 4 帧；短路口插缝不打断连续 R-E2 |
| ParkingCrossingPedestrian | R-E1, R-E4, R-E5, U-E4 | 进入路口前/停车遮挡区域行人横穿；U-E4 只覆盖横穿交互短 span，通过后回当前 RS 的 regular event，STOP/无灯路口为 R-E5 |
| ParkingCutIn | R-E1, R-E4, R-E5, U-E3 | 停车车辆启动/切入当前路径为 U-E3；起点必须有近距离 cut-in + `brake_cutin`/`vehicle_hazard`/目标变道/横向轨迹证据；R4/R5 overlay 只在响应持续或未回正时短续，不再按 distance-only 拖到前车消失 |
| ParkingExit | R-E1, R-E2, R-E4 | 无有效灯态禁止 R4；R-E2 持续到横向变道与回正完成，但初始驶出收尾在当前基础上提前约 5 帧回常规事件 |
| PedestrianCrossing | R-E1, R-E4, R-E5, U-E4 | 信号灯看 RGB/meta 分 R4/R5；U-E4 只覆盖实际让行行人 span；路口退出允许约 6 帧 RGB/灯控/行人证据尾段，1-8 帧短 regular gap 回填同类路口事件 |
| PriorityAtJunction | R-E1, R-E4, R-E5 | 混合灯控/无灯路权路口，按 RGB/meta 分帧 |
| RedLightWithoutLeadVehicle | R-E1, R-E4 | 正常红灯等待/绿灯通过并入 R-E4；驶离灯控区后同步释放 R1/R-E1 |
| SignalizedJunctionLeftTurn | R-E1, R-E4 | 信号灯左转 |
| SignalizedJunctionLeftTurnEnterFlow | R-E1, R-E4 | 信号灯左转进入车流；仍按 R-E4 主导；Town01/02 起始弱 R4 过滤后回 R-E1 |
| SignalizedJunctionRightTurn | R-E1, R-E4 | 信号灯右转 |
| StaticCutIn | R-E1, R-E2, R-E3, R-E4, R-E5, U-E3 | 起步目标变道 + 可能合流/停车区 + 后半段动态切入；STOP/无灯路口 regular 为 R-E5 |
| T_Junction | R-E1, R-E4, R-E5 | 灯控 T 路口走 R-E4，无灯/STOP/yield T 路口走 R-E5；退出侧适当延迟但控制源不互跳 |
| VehicleOpensDoorTwoWays | R-E1, R-E2, R-E5, U-E2 | 开门/停车静态占道风险 + 双向单车道绕行；紧接 U-E2 的恢复 R-E2 向前最多提前 3 帧、结束向前最多提前 4 帧；只调整恢复段，不动独立变道碎片；STOP/无灯路口 regular 为 R-E5 |
| VehicleTurningRoute | R-E1, R-E4, R-E5, U-E4 | 转弯/驶出路口后自行车或横穿对象进入路径；U-E4 不能在转弯前过早触发，VehicleTurningRoute 使用更近的 biker 交互阈值约 16m，single-span 只给 2m support padding |
| VehicleTurningRoutePedestrian | R-E1, R-E4, R-E5, U-E4 | 转弯/驶出路口后行人横穿；U-E4 不能在对象可见前过早触发；STOP/无灯路口内部短 R1 gap 回填后 regular event 同步为 R-E5；最终 6 帧以内同类 R4/R5 regular 短缝同步缝合 RS+EVENT |

## 6. 关键排除规则

这些规则用于后续写脚本或 prompt 时保持一致：

- `U-E5 对向车辆异常侵占自车道` 只给 InvadingTurn；primary RS 可是 R2，也可是在 R4/R5 路口窗口内。
- `U-E7 信号灯故障 / 路口规则失效` 可给 CrossJunctionDefectTrafficLight 的 R4，也可给 OppositeVehicleTakingPriority 这类 R5 路权失效。
- `U-E6 违规车辆冲突` 主要是信号灯正常但对方违规，当前只给 OppositeVehicleRunningRedLight。
- `CrossJunctionDefectTrafficLight` 是例外：它有红绿灯所以 RS 保持 R4，在 R4+U-E7 基础上叠 U-E6 表达四向车辆冲突。
- `U-E8 前方道路暂时阻塞 / 阻塞解除` 当前只给 BlockedIntersection；事件触发范围可比 RS 路口窗口略宽，避免阻塞等待边界漏标。
- `R-E5 无信号灯路口通行` 只给无信号/路权类 scenario 或混合场景中的无灯控制帧。
- `R-E4 信号灯路口通行` 给默认信号灯配置、明确信号灯 scenario，以及全量 RGB 证明存在灯控子集的混合 scenario；
  对无灯/灯故障帧本身不输出 R-E4。
- `NonSignalizedJunctionLeftTurn` 与 `NonSignalizedJunctionLeftTurnEnterFlow` 是 strict no-R4/no-R-E4。
  即使 bbox 或静态 XODR 弱提示报 `traffic_light`，只要没有有效 `traffic_light_state` 且同帧有
  STOP/yield/无灯路口证据，就保持 R5/R-E5 并写 review，不动态打开 R4。
- TwoWays 场景中的正常对向来车等待不等于 U-E5；只有对向车异常侵占自车道才是 U-E5。
- 障碍/TwoWays 场景不要输出 `R-E1+U-E2` 这类非路口叠加；为绕障离开原车道与核心绕行都用 U-E2，
  回原/目标车道用 R-E2，R-E2 完成后回常规事件。
- `U-E2/U-E3` 只允许在非路口候选池中触发；一旦 RS 切到 R4/R5，EVENT 必须随场景变成
  `R-E4/R-E5` 或路口专属 U-E，不能把等红灯/路口排队继续保持为 `U-E2/U-E3`。
  EVENT 后处理末尾强制执行这个候选池约束，防止桥接/单核心规则又把路口帧改回 U-E2/U-E3。
  所以 R4/R5 primary 必须严格：同向障碍/默认/noScenarios 场景只有 meta/bbox 灯态与 strong control context
  同源时才升 R4；noScenarios 还要检查 bbox 灯的 forward/distance/physical distance/affects_ego/overhead，
  远处/瞬时 `traffic_light` 不能压掉 R1/U-E2，近 STOP/yield 会优先解释为 R5。
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
- `U-E2/U-E3 -> 1-3 帧 R-E2 -> 同一个 U-E2/U-E3` 且邻域仍有核心障碍/切入证据时，
  短 R-E2 合回对应 unusual event；meta distance 缺帧不能制造假恢复。
  `dist_to_*` 阈值只能定位进入核心，不能在最近点之后继续粘住 U 类事件：非 TwoWays U-E2
  过障碍最近点后按距离回升 + 中心线回正释放；U-E3 过 cut-in 最近点后按距离回升 + 无持续强响应释放。
  对 Accident / ConstructionObstacle / ParkedObstacle，还要按整条 route
  的同一 `U-E2/R-E2` 簇做最终收口：最近障碍点之前，或距离最近点回升不足约 4.5m 且尚未越过
  核心最近点与横向避让峰值/没有回正趋势的 R-E2 合回 U-E2；若已过核心最近点与横向峰值并出现 `route_lateral_abs_m`
  回落、负向 `signed_dist_to_lane_change` 或局部中心线收敛，则 R-E2 从开始回目标/原车道处起标。
  进入恢复 R-E2 后夹在前后 R-E2 中间的 1-2 帧 U-E2 反跳，只有在横向回正证据连续时才吸收入 R-E2。
- R-E2 的触发/退出统一借鉴局部中心线复核：用 `meta["route"]` ego-frame 近前方 1-8m
  点拟合局部切线，计算 ego 到目标中心线的横向误差。只有中心线偏离且
  `changed_route` / `signed_dist_to_lane_change` / 足够 offset 支撑时才认为目标导向变道 active；
  障碍恢复段可在过障碍核心后用 `signed_dist_to_lane_change <= -0.45m` 和中心线收敛趋势
  提前 1-2 帧触发。Town06 等弯道中 `route_lateral_abs_m` 可能因为局部 route 贴车而过小；
  若 `changed_route` 与 `signed_dist_to_lane_change` 仍连续有效，不允许仅凭
  `route_centered=true` 把恢复段释放成 R-E1。post-U2 R-E2 只能由长度足够的 U-E2
  核心 span 开启，1-3 帧 trigger 抖动不能触发恢复窗口；进入恢复 R-E2 后 4 帧以内的
  U-E2 反跳合入 R-E2，回到 `1.10 * route_center_tolerance` 且 signed lane-change 结束、
  未来 2 帧稳定后退出。
  非 TwoWays 静态障碍场景不能因为障碍距离仍小于 21.5m 就压掉已经有回正/换道证据的 R-E2；
  `_lane_change_re2_supported`、回正趋势或中心线附近优先于 static-obstacle-clear 延迟保护。
  该规则用于障碍恢复、HighwayCutIn/HighwayExit、
  InterurbanActorFlow、ParkingExit、StaticCutIn 等 R-E2 候选场景；不用于判定 R4/R5 控制源或
  单独触发其它 U-E。同一个常规 R-E 前后夹住的 1-2 帧孤立 R-E2 视为中心线/flag 抖动，
  平滑回前后常规事件。
- `HighwayCutIn` 默认不开放 U-E3；若后续 RGB/轨迹全量复核确认有真实突发切入，再单独回灌白名单。
- `ParkingCutIn` / `StaticCutIn` 的 U-E3 仍保留给真实动态占道；ParkingCutIn 不把切入解释成 R-E2 或新的 RS 切换。
  如果 StaticCutIn 已进入 R-E2 目标/恢复变道，
  之后 4 帧以内短暂回 U-E3，或中间只夹 1-2 帧常规事件再回 U-E3，统一合入 R-E2。
  cut-in 最近点之后对象已远离且没有 `brake_cutin` / `vehicle_hazard` 时，不再保持 U-E3；cut-in active 保持距离约 28m。

## 7. 后续实现建议

后续脚本可以按三段式构造候选：

1. 读取 scenario，查第 3 节得到 ROAD_STRUCTURE 候选。
2. Qwen step1 从 ROAD_STRUCTURE 候选中选当前帧规则空间。
3. 根据第 4 节取该 ROAD_STRUCTURE 的事件候选，再与第 5 节的 scenario 精细事件候选求交集。

本轮 ROAD_STRUCTURE 5-id/town 调研后，EVENT 候选还必须遵守以下依赖：

- EVENT 不反推 RS。`HardBreakRoute`、`ControlLoss`、`DynamicObjectCrossing`、
  `BlockedIntersection`、`OppositeVehicleRunningRedLight` 的异常只进入 EVENT/span，
  primary RS 仍由 XML/XODR/meta 的道路结构证据决定。
- `*_TwoWays` 的 ROAD_STRUCTURE 候选删除 R1。R2 表示有效可行驶通道为对向单车道：
  包括黄中心线双向窄路、乡路，以及四车道但两侧停车/障碍/开门风险导致侧向 lane 不可行驶的等效双向单车道。
  R2 本身不改变 event：正常通行仍是 R-E1；`U-E2/U-E3` 才表示必须借/等对向的核心异常；
  绕障结束后非路口仍回 R2/R-E1 或短 R-E2，
  有灯路口覆盖为 R4/R-E4，STOP/无灯/路权路口覆盖为 R5/R-E5。
- `Parking*` 不再保留独立停车 RS。停车空间、开门、遮挡和停车位驶出并入 R1/R2，
  切入/行人/汇入分别由 U-E3/U-E4/R-E2 表达，灯控/无灯路口段仍分别优先 R4/R5。
- `R4/R5` 路口/T 形路口必须是一段连续窗口。4Hz 下少于 4 帧的 R4/R5 即使有瞬时灯态、
  bbox traffic_light 或 XODR junction hint，也按扰动并回邻近稳定 RS。
- `EnterActorFlow*`、`HighwayExit`、`MergerIntoSlowTrafficV2` 是稳定无灯控的高速/快速路背景场景，
  候选池不开放 R1/R4；EVENT 按 R3 高速/合流规则空间收窄。`HighwayCutIn` 与
  `MergerIntoSlowTraffic` 在 2026-07-04 全量逐帧 RGB 审计中发现少量真实灯控子集，因此恢复 R4
  候选，但 R4 必须由逐帧 RGB/meta/bbox 灯控证据触发，匝道/导流线/停车线不能单独制造 R4/R5。
- 混合场景不能只按 Town12/13 判高速。`HardBreakRoute`、`InterurbanActorFlow*`、
  `StaticCutIn`、`ParkingCutIn` 必须先按 route RGB 分桶；高速/快速路桶候选收敛为 R3/R4，
  非高速桶保留 R1/R4/R5。当前逐 id 均匀 5 帧 RGB 复核结果：HardBreakRoute 16/97 与 StaticCutIn 44/100
  进入高速桶；InterurbanActorFlow、InterurbanAdvancedActorFlow、ParkingCutIn 高速桶为空。
  `Town12_Rep0_258_0_route0_01_08_09_35_42` 是 HardBreakRoute 非高速反例。
- route 级 EVENT 后处理之后必须再次执行 scenario event 候选白名单与当前 primary RS event 候选池的交集 clamp；
  若后处理把非候选事件写入 primary event，则回退到当前 RS 的 regular event 或 R-E1。
  R3 的 regular event 是 R-E1，不是 R-E3；R-E3 只用于合流/驶出/actor-flow 核心。
  当前 `annotate-rs --scenario all` 全量回归覆盖 43 个 scenario、8614 条 route、1062401 帧，
  `primary_event` 越过当前 RS allowed events 的违规数为 0。
- `CrossJunctionDefectTrafficLight` 的 RS 由 signal/controller/defect 机制保持 R4；EVENT 选择 U-E7/R-E4。
  执行时 `primary_event` 保持 U-E7；四向车辆冲突明显时把
  U-E6 作为同帧 secondary event 叠加，不让 U-E6/U-E7 在 primary 上来回抢占。
- 当前生成入口 `quick_start.py annotate-rs` 会同时输出 RS 与 EVENT。人工/训练读取时应使用
  `frame_rs_annotation.label` 和 `frame_event_annotation.label` 作为本帧最终标签；
  `road_structures/events` 是候选/集合字段，不应当作“单帧唯一标签”直接显示给人看。
  如果要识别 R4/R5 路口内被截断的 R1/R2 突发状态，读取 `road_structure_overlay` 或
  `frame_rs_annotation.overlay`，不要只用普通 `secondary_road_structures` 推断。
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
