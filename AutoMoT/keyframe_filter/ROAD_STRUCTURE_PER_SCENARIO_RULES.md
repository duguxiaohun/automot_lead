# ROAD_STRUCTURE 逐场景规则设计

本文是 `collection_output/rs_research/<Scenario>/rules/scenario_rule_design.md` 的可追踪整理版。
`collection_output/` 仍只作为本机调研证据目录，不 push；凡是本地调研结论、阈值或失败模式发生变化，
必须同步更新本文。

本文只讨论 `ROAD_STRUCTURE` 的帧级划分。ROAD / EVENT 总口径仍以
`ROAD_EVENT_CLASSIFICATION_PLAN.md` 为准。
实现时 EVENT 不能只按 scenario 名称决定：scenario 级 EVENT 表只是上限，最终必须再和
当前 `ROAD_STRUCTURE` 的候选池取交集，并始终保留当前 RS 的 regular event。尤其是
R4/R5 十字路口或 T 形路口候选池不包含 `U-E2/U-E3`；红灯等待、路口排队、路口起步
必须回到 `R-E4/R-E5` 或路口专属 U-E，不能继续保持绕障/切入标签。
反过来，R4 也不能由远处/瞬时灯态单独触发：同向障碍/默认/noScenarios 场景必须同时具备
meta/bbox 灯态和 strong control context（路口/stopline/signal-junction 同源证据）才允许 R4 primary；
缺少上下文时保持 R1，让 U-E2/U-E3 候选池仍可工作。
`U-E2/U-E3` 后紧接恢复/目标变道 `R-E2`，中间 4 帧以内的短 `R-E1` 也应合入
`R-E2`，避免异常核心到恢复变道之间出现断裂监督。
同一 route 只保留一次 `U-E2/U-E3` 核心时，保留依据必须是证据强度而不是 span 时长；
`U-E2` 尤其要优先看具体静态障碍距离、绕障/回正轨迹和 route 中心线偏离，不能让运动前车
跟车距离或普通减速误触发的早期片段占掉真正事故/施工/停放障碍核心。`U-E3` 要优先看
`dist_to_cutin_vehicle`、`brake_cutin`、`vehicle_hazard` 和对象进入未来路径，不能让普通跟车
减速或路口等待占掉真正动态切入/动态占道核心。
障碍恢复类场景中，`U-E2 -> R-E1 -> R-E2` 的短桥接可放宽到 8 帧；但 R-E2 必须贴近
最近 U-E2 恢复窗口，离最近 U-E2 超过约 24 帧的后段 R-E2 应释放回当前道路 regular event。
所有 `dist_to_*` 都只作为“进入核心”的证据，不允许在最近点之后按对称距离阈值继续粘住 U 类事件：
非 TwoWays U-E2 经过障碍最近点后按距离回升 + 中心线回正截尾；U-E3 经过 cut-in 最近点后按距离回升
且无持续 `brake_cutin` / `vehicle_hazard` 截尾。
同理，真实 R4/R5 路口/T 形路口经过不应只有 2-3 帧；少于 4 帧的 R4/R5 即使有瞬时灯态、
bbox traffic_light 或 XODR junction hint，也按时序扰动并回邻近稳定 RS。
2026-07-06 在 RS 候选收紧后已重新跑 EVENT 同步审计，详见
[`ROAD_EVENT_RGB_AUDIT_ARCHIVE_202607.md`](ROAD_EVENT_RGB_AUDIT_ARCHIVE_202607.md)。全量覆盖 43 个 scenario、8614 条 route、
1062401 帧，当前 primary RS allowed events 违规数为 0；高速 R3 普通段回 R-E1，
TwoWays 删除 R1 后仍保留 U-E2/R-E2，不再被 R4/R5 regular 全部吞掉。
少量非路口 `U-E1/U-E2/U-E3/U-E4` 或静态障碍 `U-E2 -> R-E2` 恢复链被 R4/R5 控制源接管的边界使用 interrupted overlay：
primary RS 仍为 R4/R5，但 EVENT 可短时叠加 `R-E4/R-E5 + U-E*` 或
`R-E4/R-E5 + R-E2`，总上限 24 帧，恢复 `R-E2` 子阶段上限 12 帧；
U-E4 中距离横穿/转弯冲突只短续 10 帧。overlay 段必须写入专用
`road_structure_overlay` / `frame_rs_annotation.overlay`，其中 `base_road_structure`
表示被截断突发事件原本所属 R1/R2，`intersection_road_structure` 表示当前 primary R4/R5；
普通 `secondary_road_structures` 仍兼容写入 base RS，但不能单独作为 overlay 判据。详见
[`ROAD_EVENT_RGB_AUDIT_ARCHIVE_202607.md`](ROAD_EVENT_RGB_AUDIT_ARCHIVE_202607.md)。

## 0. 当前调研结论是否可直接当最终规则

不能。当前自动调研包已经能证明 XML 命名、route 匹配、meta 读取和部分 XODR 探针链路可跑通，
但还没有达到“最终规则”标准：

- 43 个 scenario 都已有本地规则草案，但 `map_rgb_alignment_status` 仍是 `not_checked`。
- `thresholds.json` 仍是 `temporary_default_rule_config`，缺少逐阈值的 supporting runs、reviewed artifacts 和来源说明。
- 自动生成的 map 图主要是 XML route / ego trace / trigger，对 lane、junction、signal、parking、merge/split 的局部 XODR 证据展示不足。
- RGB contact sheet 只适合作人工入口，尚未系统生成边界帧、错帧帧号和规则冲突样例。
- 默认 Python 环境下 CARLA API 不一定可用；需要 per-frame XODR 时应使用
  `/home/codon/anaconda3/envs/carla/bin/python`，否则 R2/R3 只能给 medium/low + review。

因此本文里的规则分为两层：

- `已确定口径`：可以进入代码的方向性规则，例如 defect traffic light 场景保持 R4+U-E7、正常信号灯路口优先 R4。
- `待验证阈值`：数值窗口和 high confidence 条件，必须继续用 `collection_output` 的地图/RGB/meta/XODR 证据校正。

## 1. 全局判定门控

### 1.1 数据源职责

- XML：只负责 route 粗投影、trigger / scenario parameter 窗口、数据追溯；不能单独作为帧级 RS 真值。
- XODR：负责局部拓扑确认，包括 lane direction、opposite lane、junction、signal/controller、merge/split、parking/shoulder/curbside。
- Meta/bbox：负责帧级 ego pose、speed、traffic light state、junction flag、active scenario、finite `dist_to_*` 信号；
  runtime 还会读取同帧 `bboxes/*.pkl` 的轻量类别摘要，`traffic_light` 辅助 R4，
  `stop_sign/yield/junction/crosswalk` 辅助 R5。
- RGB：负责人工确认地图和语义是否一致，尤其是边界帧、夜间/遮挡、停车侧、对向占道、路口灯态。

### 1.2 置信度

- `high`：scenario prior + XML window + XODR topology + meta/RGB 中至少三源一致，建议分数 `>= 0.85`。
- `medium`：两源一致，或强 meta 信号成立但 XODR/RGB 不完整，建议分数 `0.65-0.84`。
- `low`：只有 scenario prior、只有 XML trigger、或只有弱 XODR hint，建议分数 `< 0.65`，必须 review。

运行时必须记录 `diagnostic_attribution`，至少写出主标签来自 XML / XODR / meta / RGB / arbitration / threshold 中哪几类证据。

缺少 `metas/*.pkl` 或无法匹配 route XML 的 run 不参与 ROAD_STRUCTURE 判定。
这类样本属于数据质量缺口，必须以 `data_missing_skip` 写入采集/可视化 summary，
列出 `missing_meta` / `missing_route_xml` 等原因；不要用 RGB-only、bbox-only 或错误 XML/XODR
继续生成逐帧 RS 标签。

### 1.3 几何质量阈值

用于调阈值的 run 需要先满足：

- route projection median error `<= 3m`
- route projection p90 error `<= 5m`
- trigger 到 ego trace 最近距离 `<= 20m`
- 单帧 XML route projection error `> 5m` 时，不允许用 route_s 作为边界 hard truth，必须降置信并 review

这些阈值是当前工作阈值，不是最终统计阈值；后续要用 boundary frames 重新估计。

### 1.4 时间平滑与冲突

- R2/R3/R4/R5 结构片段最短持续 `4` 帧，即约 `1s`；R1 最短持续 `2` 帧。
  更短片段默认视为时序噪音，由 route 级 smoothing 并回邻近稳定 RS，并写入 evidence。
- R4 即使有红绿灯或 stopline 证据，也必须形成连续稳定片段；单帧 R4 仍按噪音处理。
- 有效 `traffic_light_state` 是强 R4 召回证据：包括 `CrossJunctionDefectTrafficLight`；
  即使当前 scenario 初始候选池没有 R4，也要临时开放 R4，并写
  `r4_meta_tl_without_strong_context_review` / RGB review；没有 meta 灯态、只有弱静态 XODR signal
  时仍保持保守 R1/review，尤其是 `noScenarios`。
  `CrossJunctionDefectTrafficLight` 的失效语义进入 U-E7，不改变其有红绿灯的 R4 结构。
- 对过度保守降级的 R1 片段，route 级 `r4_context_recovery` 只能恢复“稳定灯态 + 强路口上下文”的段：
  连续灯态/bbox traffic_light 至少 4 帧，并且有 `strong_control_context`、`close_trigger_for_junction`
  或 bbox junction hint。弱 `near_junction` 或宽 `junction_window` 不足以恢复 R4，避免把城市直路
  长时间远处灯光误当十字路口。
- 静态 XODR 的 junction/signal 只能辅助 R4/R5：`map_junction_id=-1` 或
  `junction_connection_count=0` 的 hint 不能单独作为 strong control context。若 RGB 看不到
  stopline、traffic light、横向车流或路口几何，应保守回 R1/review。
- 高速/merge 场景里的 `traffic_light_state` 或 bbox `traffic_light` 若缺少同源 junction / stopline /
  controller / strong control context，只保留弱 R4 候选和 review，不允许压过默认 R3。
- XML 匹配必须按 `(Scenario,Town,route_key)` / `(Scenario,Town,route_num)` 优先；
  跨 town 纯数字冲突不能随便选择一个 XML，否则会把错误 XODR 拓扑带入 R4/R5 判断。
- 主候选与次候选分差 `< 0.15` 时，保留 primary 但必须 review。
- 同一帧若 R4/R5 与 R2/R3 冲突，优先判断是否处于真实路口控制区；路口控制证据同源时 R4/R5 做 primary，结构风险做 secondary。

### 1.5 本轮自动调研暴露的不完善点

`collection_output/rs_research/*/rules/scenario_rule_design.md` 已经统一生成 43 个场景的规则草案、抽样 run、
XODR 摘要和 `thresholds.json`，但这些产物仍偏“可运行模板”，不是最终人工规则。需要补强的点如下：

- `map_rgb_alignment_status` 全部仍是 `not_checked`，所以地图 route / trigger / ego trace 与 RGB 三视角是否贴合还没有闭环验收。
- `thresholds.json` 虽然已列出 `supporting_runs`，但 `reviewed_artifacts` 仍为空，`source` 仍是 `temporary_default_rule_config`；这些数值只能作为初始搜索窗口。
- 每个场景的自动 `scenario_rule_design.md` 都提示“当前自动审计未发现 town 级输入缺口”，这只说明 XML/XODR/meta 可读，不等价于边界帧正确。
- 当前 XODR 探针以 route/trigger 近邻和 town 级 signal/junction 摘要为主，仍缺少 per-frame lane successor、opposite-lane、parking/shoulder、merge/split 的局部拓扑证据。
- RGB contact sheet 是人工入口，不是判定证据；后续应为每个结构切换点生成 `boundary_before/current/after` 帧，并记录错帧归因。
- 本轮跨场景 R4 漏检审计发现，`Accident`、`HazardAtSideLane`、`ControlLoss`、`Parking*`
  等大量帧已有有效 meta 灯态，却因缺少 `strong_control_context` 被压成 `R1`；另有
  `PriorityAtJunction` / `NonSignalizedJunction*` 等场景因初始候选池没有 R4，把有效灯态 R4 过滤掉。
  代码已改为“有效 meta 灯态动态开放 R4 并做 primary + review”，但 2026-07-04 全量逐帧 RGB 审计确认无稳定灯控的场景
  会显式屏蔽这条动态 R4 兜底，避免伪灯态污染无灯路口/高速背景；复审后剩余 R1+灯控证据主要是
  `noScenarios` 弱静态 XODR signal、已屏蔽的 RGB no-R4 场景或短 R4 片段 smoothing。
- `noScenarios` 已显示部分 run 存在 Red 灯态或 junction meta，但候选仍保守为 R1/R4；不要从弱 XODR hint 自动挖 R3/R5，除非单独建立 topology-only 人工样本集。
- `T_Junction` 经全量逐帧 RGB 审计确认存在灯控与无灯/STOP T 形路口子集，候选为 R1/R4/R5；R4/R5 必须由逐帧 RGB + meta/bbox 控制源区分。
- 普通 Python 下静态 XODR planView 近邻可能出现几十米投影误差；只有 `map_projection_error_m <= 20m`
  的静态 XODR 拓扑才允许作为 R2/R3 high 证据，否则必须写 `xodr_topology_untrusted` 并降级 review。
- `light_hazard=True` 不能单独把非路口帧升为 R4；必须同时有 `near_junction`、有效灯态或可信静态 signal 近邻。
  本轮 smoke test 中该门控把 `AccidentTwoWays` 小样本的 R2/R4 抖动从 38 次切换降到 2 次核心切换。
- 环岛 / roundabout 归 R1，不归 R4/R5。XODR 即使把环岛编码成 junction road，也必须进一步检查
  roundabout 几何/连接特征；`map_is_roundabout=true` 时，R4/R5 的 junction/window 分支都要失效。
- 所有 `*_TwoWays` / `InvadingTurn` / `VehicleOpensDoorTwoWays` 的 R2 high 按有效可行驶通道判断：同向可行驶 lane 不足、对向交互主导，或四车道但两侧停车/障碍/开门风险让侧向 lane 不可行驶，均可支撑 R2。
- 所有 `highway_merge` / `interurban` / `static_cutin` 的 R3 high 必须补 merge/split/ramp/lane-count-change 证据；切入、EnterFlow、低速车流等事件名不能单独触发 R3。
- `parking` / `parking_exit` / `vehicle_opens_door_twoways` 不再生成独立停车 RS；parking lane、shoulder、curbside 或 RGB 路边停车空间只能辅助 R1/R2 与事件判定。

### 1.6 当前 `thresholds.json` 初始规则族

下表只沉淀本轮自动调研输出里的初始窗口，目的是让后续实现和人工复核有统一起点。
所有值在 `reviewed_artifacts` 补齐前都必须视为 `temporary_default`，不能作为 hard truth。

| 规则族 | 适用场景 | 当前初始窗口 / 关键开关 | 必须补的证据 |
|---|---|---|---|
| `same_direction_obstacle` | Accident, ConstructionObstacle, ParkedObstacle | Accident `junction_pre_m=54`, `junction_post_m=22`; Accident 前 30 帧只有缺少真实控制源的弱 R4/R5 hint 才强制回 R1，Town13 除外；ConstructionObstacle 进入侧收为 `junction_pre_m=42`、post 仍 25；ParkedObstacle 按 RGB 边界放宽为 `junction_pre_m=72`, `junction_post_m=25`; veto R2 | 障碍只进 EVENT；Accident 起始段默认不应被弱十字路口 hint 抢走，但若初始帧已有有效灯态、bbox traffic_light、STOP/yield 或明确 junction 控制源，则保留 R4/R5；Town13 按原始 XODR/meta/RGB 证据保留 R2/R4/R5；Accident/Construction 路口窗口收缩以减少普通绕障路段误升 R4/R5，ParkedObstacle 则按 RGB 召回已进入真实路口控制区的帧；R4 需灯控/stopline；STOP/无灯路口 regular 允许 R5；同向绕障不得升 R2；可见 `meta TL + bbox traffic_light` / `bbox/meta STOP/yield` 可越过稀疏 XML window 恢复 R4/R5 |
| `twoways_obstacle` | AccidentTwoWays, ConstructionObstacleTwoWays, HazardAtSideLaneTwoWays, ParkedObstacleTwoWays | `two_way_min_pre_m=50-75`, `two_way_post_pad_m=20`, `trigger_close_m=70-75`, `two_way_xml_core_close_m=8`, `two_way_obstacle_core_m=18-20`, `two_way_approach_obstacle_m=28-30`, `two_way_exit_delta_m=2`, `two_way_exit_hold_frames=3`, `two_way_layout_prior=true`; R4/R5 弱召回按场景收紧：AccidentTwoWays factor 0.85，ConstructionObstacleTwoWays `junction_pre_m=42` 且 factor 0.70（有效约 29.4m），HazardAtSideLaneTwoWays / ParkedObstacleTwoWays factor 0.75；AccidentTwoWays 前 30 帧只有缺少真实控制源的弱 R4/R5 hint 才强制回 R2，Town13 除外 | `*_TwoWays` 候选删除 R1；双向窄路和两侧停车/障碍导致侧向 lane 不可行驶的四车道路段都按有效可行驶对向单车道 R2。Accident/Construction/Parked 的 EVENT 仍按 U-E2 核心后接 R-E2；HazardAtSideLaneTwoWays 是明确例外，自行车/行人进入路径时按 U-E4，对象离开后回正才 R-E2。只有真实灯控/STOP/无灯控制源覆盖为 R4/R5；twoways 的事故/施工 XML trigger 不能单独制造 R5 |
| `default_meta_map` | ControlLoss, CrossingBicycleFlow, DynamicObjectCrossing, HazardAtSideLane | 默认 `junction_pre_m=50`, `junction_post_m=25`; CrossingBicycleFlow 进入侧收为 `junction_pre_m=35` 且 factor 0.90（有效约 31.5m）；场景动作多数 veto RS 升级；R4/R5 弱召回按 RGB 复核分层收紧：ControlLoss factor 0.60，DynamicObjectCrossing factor 0.65，HazardAtSideLane factor 0.90 | 横穿、失控、side-lane hazard 只进 EVENT；RS 由路网/灯控/STOP 无灯控制源决定；场景级 factor 同时缩短 junction window、meta near、strong context、static signal near 和 close-trigger，避免远处/瞬时弱路口 hint 抢占 R1/R2；ControlLoss Town01-04 前 30 帧只信有效灯态+灯框+本地上下文或视觉 junction hint，meta-only/XODR-only/stop-only 伪路口回 R1；DynamicObjectCrossing 额外要求 R4/R5 近控制源，远灯/起始弱 STOP 不直接升路口；HazardAtSideLane 前 30 帧仅 bbox-only STOP、close-trigger 或 untrusted XODR 时回 R1 |
| `blocked_intersection` | BlockedIntersection | `junction_pre_m=32`, `junction_post_m=18`; 在通用十字路口进入侧上额外收紧 | 阻塞只进 EVENT；有有效灯态/信号灯同源证据时 R4，RGB/meta/bbox 显示 STOP/yield/无灯路口时 R5，二者都缺失则 R1 + review |
| `signalized_junction` | OppositeVehicleRunningRedLight, RedLightWithoutLeadVehicle, Signalized*Turn*, T_Junction | `junction_pre_m=50-60`, `junction_post_m=20-32`; runtime effective pre = `0.36 * pre`、post = `0.28 * post`，pre 最小 16m、post 最小 5m；RedLightWithoutLeadVehicle 离开侧额外用 `scenario_active_signal_max_m=52` 释放尾段；SignalizedJunctionLeftTurnEnterFlow Town01/02 前 30 帧弱 R4 过滤；T_Junction `junction_post_m=32`, `review_if_no_tl=True`; static signal near <=35m 且 strong context <=22m；close-trigger 上限 25m | 有效灯态、light_hazard、signal/controller、stopline approach 至少多源一致；无有效 `traffic_light_state` 的 R4 必须写 RGB confirmation review；T_Junction 若 RGB/stop/yield 显示无灯控制则允许 R5；起始远灯/弱 trigger 不能覆盖普通直道，离开灯控区要回 R1，但仍允许 T_Junction 退出侧保留足够尾段防止 RGB 仍在路口内时过早释放 |
| `defect_junction` | CrossJunctionDefectTrafficLight | `junction_pre_m=60`, `junction_post_m=20`, `junction_tighten_factor=0.65`, `junction_min_pre_m=10`, `junction_min_post_m=3`, `rule_note=signalized_rs_with_defect_event` | defect 场景仍是有红绿灯的 R4；信号灯失效/规则源失效进入 U-E7；远距离 `traffic_light_state` / bbox traffic_light 不能单独把起始直道升 R4，必须有近 `dist_to_junction`、`is_junction`、bbox junction、可信 XODR signal/junction 等本地控制源 |
| `nonsignalized_junction` | NonSignalizedJunction*, OppositeVehicleTakingPriority, PriorityAtJunction | `junction_pre_m=50-84`, `junction_post_m=20`; `NonSignalizedJunctionLeftTurnEnterFlow` 放宽为 `junction_pre_m=84`；`NonSignalizedJunctionRightTurn` 放宽为 `junction_pre_m=63` 且保留 `junction_tighten_factor=0.75`；`OppositeVehicleTakingPriority` 按 RGB 边界放宽为 `junction_pre_m=75`；RightTurn 用 `distance_to_intersection_index_ego` 前 45m / 后 5m + `dist_to_junction<=18m` + 近 trigger 形成局部核心门控 | no-light / priority / stop / yield 证据；NonSignalizedJunctionLeftTurn 与 NonSignalizedJunctionLeftTurnEnterFlow 是 strict no-R4，bbox/static signal 弱提示不能动态打开 R4；NonSignalizedJunctionRightTurn 与 OppositeVehicleTakingPriority 以 R5 为主但全量 RGB 有少量灯控子集，R4 仅在有效灯态或强灯控同源证据成立时开放；RightTurn 的远处 `scenario_active`、残留 stop_hazard 或 meta/bbox 灯控不能单独覆盖起始/驶离直道；同一 R-E4/R-E5 中间夹不超过 12 帧 R-E1/R-E2 时按路口短缝噪音同步合并 RS+EVENT 回前后相同 R4/R5 路口段；PriorityAtJunction 是 R4/R5 混合 |
| `pedestrian_crossing` | PedestrianCrossing | `junction_pre_m=36`, `junction_post_m=60`, `junction_tighten_factor=0.70`, `junction_min_pre_m=12`, `junction_min_post_m=5`, `pedestrian_exit_tail_frames=6`; `pedestrian_not_rs` | 行人只进 EVENT；R4/R5 取决于 crossing 是否与路口控制源同源；入口按 RGB 稍收紧，退出保留短尾段；同一 R4/R5 路口内 1-8 帧普通事件短缝同步缝合 RS+EVENT |
| `highway_merge` | EnterActorFlow*, HighwayCutIn, HighwayExit, MergerIntoSlowTraffic* | 默认 RS=R3；EnterActorFlow* 保留 R1/R3 且删除 R4，远端直道可 R1、近汇入控制区切 R3；HighwayExit、MergerIntoSlowTrafficV2 候选删除 R1/R4；HighwayCutIn 与 MergerIntoSlowTraffic 保留少量 R4 子集；`merge_pre_m=30-50`, `merge_post_m=40-50`, `trigger_close_m=90`; EVENT 二次窗口：EnterActorFlow* 用 trigger、actor-flow、ramp/merge hint 与 route 级 postprocess 维持准备汇入段 R-E3；MergerIntoSlowTraffic* 禁用 trigger-only 圆窗直接制造 R-E3，已有 R-E2 核心前后各最多补 5 帧，R-E2 后近 actor-flow/merge tail 最多 64 帧保持 R-E3；HighwayExit 从出口变道/分流/`next_commands[0]==3` 起进入 R-E2/R-E3，驶出匝道段保持 R-E3 | 全量逐帧 RGB 显示主体为高速/快速路/匝道背景；RS=R3 不能机械同步成全程 R-E3，四问 `HIGHWAY` 也不能机械等同 RS：2026-08-09 四问复核确认 `EnterActorFlow*/R1/R-E1` 因真实高速/快速路拓扑仍答 `HIGHWAY=YES`。HighwayCutIn 默认 R-E1，局部 route 中心线偏离 + lane-change 组合证据才 R-E2；Enter/Merger 的匝道/准备汇入段不能在 R-E3 与 R-E2 之间插入 R-E1；Merger 刚开始/中间普通主线跟车仍可 R-E1，R-E2 后若仍处于分离汇入/匝道空间才保持 R-E3，3 帧以内夹在 R-E1 中的孤立 R-E3 小岛平滑回 R-E1；EnterActorFlow* 的 R-E2 围绕已有变道核心按 `changed_route + signed_dist_to_lane_change` 补完整轨迹；HighwayExit 出口前主线正常跟车为 R-E1，真实目标换道为 R-E2，驶出匝道为 R-E3。HighwayCutIn 与 MergerIntoSlowTraffic 的 R4 必须有 RGB/meta/bbox 灯控同源证据；匝道/导流线/停车线不能单独制造 R5 |
| `hardbreak_route` | HardBreakRoute | `junction_pre_m=50`, `junction_post_m=25`; route 级 RGB 高速桶候选收敛为 R3/R4 | 97 个 route 已逐 id 均匀 5 帧 RGB 复核；16 个高速/快速路桶给 R3/R4，其余城市/乡村 route 保留 R1/R4 |
| `interurban` | InterurbanActorFlow | `merge_pre_m=50`, `merge_post_m=45`, `junction_pre_m=55`, `junction_post_m=25`; route 级 RGB 高速桶为空 | 2026-07-04 全量逐帧 RGB 审计未见稳定信号灯路口，删除 R4；保留 R1/R3/R5，STOP/active close-trigger 无灯路口可给 R5。2026-08-09 四问复核确认 `InterurbanActorFlow/R3/R-E1` 仍非高速，`HIGHWAY=NO`；不能凭 R3、直道、宽路或空旷路面答 highway |
| `interurban_advanced` | InterurbanAdvancedActorFlow | `junction_pre_m=72`, `junction_post_m=33`, `r3_requires_topology=True`; route 级 RGB 高速桶为空 | 2026-07-04 全量逐帧 RGB 审计未见稳定 R4，默认 R1/R5，只有明确 RGB/XODR merge 才临时打开 R3；junction 进入/退出窗口较旧 `55/25m` 放宽约 30% |
| `invading_turn` | InvadingTurn | `two_way_min_pre_m=80`, `two_way_post_pad_m=20`, `trigger_close_m=75` | 2026-07-05 RGB 复核发现 Town12 稳定信号灯子集，恢复 R4；无灯/STOP 路口仍为 R5，对向车侵入/heading conflict 仍由 U-E5 表达 |
| `parking` / `parking_exit` | ParkingCrossingPedestrian, ParkingCutIn, ParkingExit | `parking_pre_m=20-35`, `parking_post_m=50-60`; ParkingCutIn route 级高速桶为空且全量逐帧 RGB 未见稳定独立停车结构 | parking/shoulder/curbside 或 RGB 停车空间不再保留独立 RS；ParkingExit 用 R1 + R-E2 表达驶出并入，ParkingCutIn 的切入归 U-E3，ParkingCrossingPedestrian 的行人归 U-E4；若 STOP/无灯路口证据同源则允许 R5 |
| `static_cutin` | StaticCutIn | `merge_pre_m=35`, `merge_post_m=55`; route 级 RGB 高速桶候选收敛为 R3/R4 | 100 个 route 已逐 id 均匀 5 帧 RGB 复核；44 个高速/快速路桶按 R3/R4，其余按 R1/R4/R5；R3 必须有高速/合流证据 |
| `vehicle_opens_door_twoways` | VehicleOpensDoorTwoWays | `two_way_min_pre_m=55`, `two_way_post_pad_m=20`, `parking_pre_m=35`, `parking_post_m=55` | 候选为 R2/R4/R5；两侧停车/开门风险占用侧向 lane 时主 RS 为 R2，不再生成独立停车 RS |
| `vehicle_turning` | VehicleTurningRoute, VehicleTurningRoutePedestrian | VehicleTurningRoute `junction_pre_m=50`, `junction_post_m=20`, `junction_tighten_factor=0.65`, `junction_min_pre_m=8`, `junction_min_post_m=3`, `turning_trigger_core_m=8`, `turning_nosignal_trigger_core_m=5`, `turning_tl_requires_local_junction=true`, `multi_trigger=True`; VehicleTurningRoutePedestrian `junction_pre_m=50`, `junction_post_m=40`, `junction_tighten_factor=0.60`, `junction_min_pre_m=8`, `junction_min_post_m=3`, `turning_trigger_core_m=8`, `turning_nosignal_trigger_core_m=5`, `turning_tl_requires_local_junction=true`, `turning_final_regular_gap_max_frames=6`; projection-error R5 demotion | 多 trigger 分段；行人/横穿不改变 RS，控制源决定 R4/R5；VehicleTurningRoute / Pedestrian 的远灯态必须落在本地路口/收紧窗口内，只有 XML trigger/STOP 而缺少本地 junction 证据的无灯 R5 必须进入 5m 核心区才触发；高投影误差下无灯 R5 必须有 stop / is_junction / 非静态可信 XODR 近路口证据；最终输出层对同类 R4/R5 段中的短 regular gap 同步缝合 RS+EVENT |
| `noscenario` | noScenarios | `junction_pre_m=50`, `junction_post_m=25`, `conservative=True` | 默认保守 R1；稳定 meta 灯态 + bbox traffic_light + junction window 召回 R4，STOP/无灯控制证据召回 R5；弱 topology hint 只写 evidence/review |

### 1.7 时序稳定与环岛仲裁

帧级规则输出后必须再经过 route 级时序稳定，避免 `R1 -> R4 -> R1` 或任意
`R* -> Rk -> R*` 的单帧/短片段扰动被当成真实道路结构切换：

- R2/R3/R4/R5 最短有效持续为 4 帧（4Hz 下约 1 秒）。
- R1 最短有效持续为 2 帧；短 R1 夹在同一特殊 RS 中间时，视为噪音缝隙并填平。
- 短片段前后标签一致时直接改为该标签；前后不一致时并入更长邻接片段，并写
  `evidence.temporal_smoothing`。
- 去抖是所有 RS 的统一后处理，不是 R4 特例；`frame_rs_annotation.label` 必须反映去抖后的最终标签。
- TwoWays 静态/侧向障碍在去抖前先做事件型 R2 裁剪：有有效可行驶对向单车道证据的正常直道保持 R2；
  无拓扑支撑、仅靠核心障碍召回的临时 R2，过最近障碍点后若距离连续约 0.75s 至少远离 2m，
  且没有 `stuck` / `vehicle_hazard`，后续帧按当前控制源回 R2/R4/R5，并把原因写入 route 摘要。
- EVENT 与核心 span 对齐：TwoWays 的 U-E2/U-E3 不能早于障碍核心证据；核心前是 R2 + R-E1，
  进入核心时是 U-E2/U-E3，离开核心并回目标/原车道时才是 R-E2。
- 如果同一条 TwoWays route 中出现多个无拓扑支撑的 R2 片段，只保留最长连续事件型 R2 段；
  有双向单车道拓扑支撑的 R2 不参与该过滤。
- RGB 盲审统计不把 R2/U-E2 核心段的弱 stopline/turn-marking、车尾颜色或中心线误计为
  `blind_R5_label_R2`；完整 `U-E2 -> R-E2` 后已经回到 R-E1 的普通交通流 motion，
  也不再计作 TwoWays 漏事件。否则 top mismatch 会被审计器假阳性主导。

环岛判断优先级高于 R4/R5：

- 静态 XODR 会结合 junction id、连接 road 数、局部曲率、geometry 长度和 signal 距离输出
  `map_is_roundabout`。
- CARLA API probe 可用时仍合并静态 XODR 的 roundabout hint，避免不同 Python 环境下规则漂移。
- `map_is_roundabout=true` 时，R4/R5 分数被移除，R1 作为 primary，并记录
  `roundabout_xodr_forces_r1` / `roundabout_removed_junction_rs_scores`。

### 1.8 已落地的可执行标注与可视化入口

本轮已把本文思路落到 `collector.py` / `quick_start.py` / `web_app.py`：

- `SCENARIO_RULE_CONFIG` 为 43 个 scenario 提供独立规则族和阈值；`annotate-rs --rule-config-json`
  可加载每场景阈值覆盖文件，便于在不改代码的情况下调参。
- `quick_start.py annotate-rs` 会逐帧调用 `RoadStructureRuleEngine` 与 `RoadEventRuleEngine`，
  命令名沿用历史 RS 入口，但当前正式输出是 RS + EVENT。保留旧字段
  `road_structures` 候选全集，同时新增 `primary_road_structure` 与显式单帧结果
  `frame_rs_annotation`；EVENT 侧新增 `primary_event`、`events`、`event_evidence` 与
  `frame_event_annotation`。
- `frame_rs_annotation` 包含 `label/secondary/overlay/confidence/comment/rule_kind/rules_fired/decision_source/review_required/review_reasons/metrics/xodr_summary`，
  可直接作为人工验收和后续训练输入的帧级解释结果。
  其中 `overlay` 是 R4/R5 路口截断 R1/R2 突发事件时的专用结构化字段，普通 secondary
  可能表示候选冲突或不确定性，不能单独用于判断 overlay。
- `frame_event_annotation` 包含 `label/events/regular_event/unusual_event/allowed_events/rules_fired/metrics/review_required/review_reasons/comment`，
  用于解释当前 EVENT 是否来自常规 RS、异常白名单、XML/active 窗口还是 meta/轨迹证据。
- `web_app.py` 已把候选全集和本帧最终标签拆开展示：顶部绿色标签读取
  `frame_rs_annotation.label` / `primary_road_structure`，置信度只表示该帧 primary RS 的置信度；
  RS 卡片中的 `RS Overlay` 单独读取 `frame_rs_annotation.overlay` / `road_structure_overlay`，
  显示为 `RS-Overlay base→intersection`；
  红色 EVENT 主标签读取 `frame_event_annotation.label` / `primary_event`；
  `road_structures` 与 `events` 只作为候选/同帧集合展示，不再和本帧主标签混用。
- Web 与 `frame_rs_annotation_summary.json` 均暴露 `road_structure_labels/event_labels`，
  用中文解释 R1-R5、R-E*/U-E* 代号，避免人工验收时只看到裸代号。
- Web 证据面板会展示 XML/route 投影、trigger 距离、LEAD meta 灯态/active scenario、XODR
  source/trusted/road/lane/junction/opposite/parking/merge 等摘要；若这些证据不足或冲突，
  页面会显示 review 状态和原因，供下一轮规则修正。
- `web_app.py` 默认路径已改为当前仓库相对路径：
  `AutoMoT/lead_data`、`AutoMoT/lead_video`、`AutoMoT/keyframe_filter/collection_output`；
  仍可用 `LEAD_DATA_ROOT`、`LEAD_VIDEO_ROOT`、`KEYFRAME_COLLECTION_OUTPUT` 覆盖。
- route 投影误差 `>5m` 时，代码会禁用 `route_s` hard window，只允许 trigger distance / meta active / junction/light 等证据参与，并写
  `route_s_window_disabled_projection_error_gt_5m`。
- 2026-07-04 最终 smoke 已覆盖全 43 场景 `--max-routes 1 --max-frames-per-route 10`；
  程序化检查确认 `frame_rs_annotation.label == primary_road_structure`、
  `frame_event_annotation.label == primary_event`，且所有 `primary_event` 均落在当前 scenario
  EVENT 白名单中。
- `noScenarios` 调整为无 meta 有效灯态或 light hazard 时强制保守 R1；静态 XODR signal/junction hint 只进 evidence/review，不再把普通无场景帧自动推成 R4。
- `StaticCutIn` 调整为 cut-in 窗口内若没有 merge/highway 拓扑证据，则回 R1 中置信；2026-07-06 全量逐帧 RGB 未见稳定独立停车结构，因此候选收紧为 R1/R3/R4/R5。
- RGB-first 全量复核覆盖 43 个 scenario、204 条 scenario-town route、24387 帧；
  `candidate_anomalies=15788` 只是逐帧看图索引，不是错帧数。summary、confidence、
  `candidate_anomalies` 和标签分布只能帮忙定位，最终异常必须人工读取每条 route 的
  `all_frames_*.jpg` 后确认。
- 标定与 RGB 不匹配主要来自六类根因：
  1. XML/XODR route 投影误差高，却仍把 route_s / trigger window 当 hard boundary。
  2. 静态 XODR signal/opposite/parking/merge/junction hint 与 RGB 不同源，被误当 high confirmation。
  3. scenario 名称、active scenario 或事件距离被当作 ROAD_STRUCTURE 真值，导致 EVENT 覆盖 RS。
  4. 坏 XODR 被当作否定证据，导致明显高速/merge 或 TwoWays 核心借道/绕障片段被压回 R1。
  5. 只看 summary/confidence，没有逐帧看稳定 R1 和低能见度帧。
  6. 行人、事故、施工、急刹、切入、开门、闯红灯等事件与道路控制源混淆。
- 已回灌的通用修正：
  R1 稳定兜底；弱特殊 RS 不再低分压过 R1；`route_projection_error_m > 5m` 时
  `scenario_active` 和普通 `trigger_close_m` 只写 review/evidence；静态 XODR hint 在高投影误差帧降级为
  `*_demoted_projection_error`；无有效 `traffic_light_state` 的 R4 必须 RGB confirmation；
  static-signal-only strong context 距离收紧到 25m；Interurban / vehicle_turning 的 R5
  高投影误差降级；Merger actor-flow/trigger fallback；TwoWays layout-prior 与核心借道/障碍分层。
- 场景级回灌原则：
  同向事故/施工/急刹/动态障碍只进 EVENT，不能靠 `light_hazard` 或障碍距离升 RS；
  TwoWays 按有效可行驶通道而不是原始 lane 数决定 R2；确认双向窄路或两侧停车/障碍压缩 lane 时，道路层保持 R2；
  若出现多个无拓扑支撑的临时 R2 片段，最长连续事件型 R2 段优先作为真正核心，其它碎片按控制源回 R2/R4/R5；
  高速/merge 场景不能再被默认 R1 吃掉，已对 `EnterActorFlow*`、`HighwayExit`、
  `MergerIntoSlowTrafficV2` 删除 R1/R4 候选；`HighwayCutIn`、`MergerIntoSlowTraffic` 全量 RGB
  发现少量真实灯控子集，因此恢复 R4 候选但不恢复 R1；`PriorityAtJunction` 和 `HardBreakRoute` 不能只因 Town12/13 判高速。
  R3 只表示 merge/ramp/split/exit 特殊结构，不表示物理高速路本身；
  独立停车结构已移除，停车带/路边停车空间不表示 parked obstacle 事件；
  signalized / nonsignalized / defect junction 由控制源决定，低能见度时 XODR/XML/meta 只作补证。
- 逐帧 RGB 复核固定流程：
  每个 scenario 的每个 town 抽 1 条 readable route，生成全量全帧 contact sheet；
  从第一帧看到最后一帧，稳定高置信 R1 也不能跳过；
  异常记录必须写明 frame 范围、RGB 观察、当前标签、冲突原因、关键 evidence/rules，
  并归因到规则思路、阈值参数、XML/XODR 投影或低能见度证据不足。

已执行 smoke：

```bash
python -m py_compile AutoMoT/keyframe_filter/collector.py AutoMoT/keyframe_filter/quick_start.py
python AutoMoT/keyframe_filter/quick_start.py annotate-rs \
  --scenario T_Junction,AccidentTwoWays,ParkingExit,HighwayExit,noScenarios \
  --max-routes 1 --max-frames-per-route 40 \
  --output-dir /tmp/automot_rs_annotation_smoke
python AutoMoT/keyframe_filter/quick_start.py annotate-rs \
  --scenario all --max-routes 1 --max-frames-per-route 10 \
  --output-dir /tmp/automot_rs_annotation_all_smoke
```

结果摘要：43 场景小样本均可执行生成；全场景 430 帧 smoke 中主标签分布为
`R1=174, R2=10, R3=70, R4=118, R5=58`，confidence 为
`min=0.70/avg=0.8360/max=0.98`，review frame ratio 为 `0.5256`。
review 增加主要来自图像优先复核后新增的投影/静态拓扑降级归因：
`route_projection_error_high`、`static_xodr_topology_demoted_by_projection_error`、
`structure_window_demoted_by_projection_error`、`weaker_special_rs_kept_as_candidate_not_primary`、
`candidate_score_gap_lt_0.15`。
这表示可疑 R2/R3/R4/R5 不再被静态拓扑或 scenario 窗口直接当主标签；
后续若用 `/home/codon/anaconda3/envs/carla/bin/python` 跑 CARLA API XODR probe，应优先比较这些 review 是否下降。

## 2. 逐场景规则

### 2.0 每场景 RGB 错配归因总表

| 场景 | 最容易导致标定与 RGB 不匹配的原因 | 当前修正口径 |
|---|---|---|
| Accident | 事故事件或 light_hazard 被误当道路结构，普通同向路段提前升 R4/R2；route 前后 STOP/无灯路口被压成 R1 | 障碍只进 EVENT；非 TwoWays 默认不自动升 R2，只有强路口/灯控上下文才 R4；STOP/无灯路口 regular 可 R5 |
| AccidentTwoWays | STOP/无灯 T/十字路口被压成 R1，或旧规则把非路口 TwoWays 直道误回 R1 | 候选删除 R1；非路口按有效可行驶对向单车道 R2，灯控/STOP/无灯控制源覆盖 R4/R5；U-E2/R-E2 只表达绕障/回正动作 |
| BlockedIntersection | 低能见度下无 meta 灯态的 R4 容易被误判为置信度问题 | RGB 可见 stopline/crosswalk/cross traffic/blocked pocket 时保留 R4 + review |
| ConstructionObstacle | 施工物事件或静态 signal 近邻把普通路段提前升 R4/R2；STOP/无灯路口被压成 R1 | 施工只进 EVENT；非 TwoWays 默认不自动升 R2；static-signal-only strong context 限 25m；STOP/无灯路口可 R5；R-E2 只保留中心线回正短段 |
| ConstructionObstacleTwoWays | 施工锥桶与 two-way road-layout 混淆；STOP/无灯路口被压成 R1/R2 | 非路口按有效可行驶对向单车道 R2；施工占道看 U-E2/R-E2；STOP/无灯路口可 R5；R-E2 用局部中心线截尾 |
| ControlLoss | 失控/低速/急刹被误当停车或路口结构，或 STOP/无灯路口被全程压成 R1 | 失控只进 EVENT；RS 由道路与控制源决定，灯控为 R4，STOP/无灯为 R5，其余 R1；非 TwoWays 默认不自动升 R2 |
| CrossJunctionDefectTrafficLight | 旧规则看到 defect 就误转 R5 | 红绿灯硬件/受控路口仍是 R4；defect 只改变 EVENT U-E7 |
| CrossingBicycleFlow | actor flow 字样误触发 R3 | 自行车横穿只进 EVENT；只有真实灯控路口才 R4 |
| DynamicObjectCrossing | 动态对象横穿被当成无灯路口/停车结构 | 动态对象只进 EVENT；RS 只看路网/灯控；非 TwoWays 默认不自动升 R2 |
| EnterActorFlow | 高速/快速路 enter flow 被 R1 默认桶吃掉，或被伪灯态误升 R4 | 候选保留 R1/R3 并删除 R4；远端直道可 R1，靠近 enter/merge 控制区切 R3；EVENT 用轨迹补全 R-E2 |
| EnterActorFlowV2 | 短 route / 投影抖动导致 R3 边界过宽或被压回 R1 | 同 EnterActorFlow，短 route 必须靠 RGB 和强拓扑确认；不接收动态 R4 |
| HardBreakRoute | 急刹被误当红灯/停车结构 | 急刹只进 EVENT；红灯停车必须有灯控/stopline 同源 |
| HazardAtSideLane | side-lane hazard 被误当 R2 或静态障碍 U-E2；bbox-only STOP / close-trigger 把开头直道误升 R5；U-E4 后恢复变道被短 R-E1 隔断 | 非 TwoWays 默认 R1 + EVENT，真实灯控/STOP/无灯源才 R4/R5；XML/RGB 确认自行车/行人进入路径时为 U-E4，对象离开后 8 帧内有回目标车道/横向回正证据时从 U-E4 后第一帧直接 R-E2；只有对向参与才 review R2 |
| HazardAtSideLaneTwoWays | two-way hazard 与普通双向道路/动态自行车核心混在一起；bbox-only STOP 把开头直道误升 R5；U-E4 后 R-E2 被路口 regular 吃掉 | 有效可行驶对向单车道时 ROAD_STRUCTURE 保持 R2；自行车/行人进入路径由 U-E4 表达，离开对象后的回正直接 R-E2；R4/R5 接管时允许 regular+R-E2 overlay；前 30 帧无真实 junction/control 的 bbox-only STOP 回 R2 |
| HighwayCutIn | 高速/快速路背景被 R1 默认桶吃掉，或把切入事件和道路结构混淆；少量灯控子集会被场景级 no-R4 误删 | 候选删除 R1、保留 R3/R4；非路口高速默认 R3，R4 只靠逐帧灯控同源证据触发，切入只进 EVENT；自车真实目标变道才由中心线偏离给 R-E2 |
| HighwayExit | 高速出口场景被 R1 默认桶吃掉，或 exit 名称把 R4/R3 边界拖宽 | 候选删除 R1/R4；高速/分流背景默认 R3；出口曲线/匝道弯道不再仅凭 changed_route 产生 R-E2 |
| InterurbanActorFlow | 高投影误差下 stop/junction hint 过弱或被压回 R1 | 全量 RGB 显示大量 STOP/无灯十字路口且无稳定 R4；stop/junction/active close-trigger 可给 R5，仍保留 projection review；R-E2 只给中心线偏离明确的目标变道 |
| InterurbanAdvancedActorFlow | advanced/actor-flow 先验过强，普通城际道路误升 R3，或被伪灯态误升 R4 | 默认 R1/R5；R3 必须有可见或可信 merge/topology，R4 已按 RGB 删除 |
| InvadingTurn | 对向车事件与无灯十字路口结构混淆，且旧候选缺 R5 | R2 表示双向窄路/对向占道主导；R5 表示无信号/STOP 路口主导；EVENT 用 U-E5 表达被动让行/侵入 |
| MergerIntoSlowTraffic | 坏 XODR 把明显高速/merge 压回 R1，或 active scenario 把核心合流窗口拖宽；少量灯控子集会被场景级 no-R4 误删 | 候选删除 R1、保留 R3/R4；高速/合流背景默认 R3，R4 只靠逐帧灯控同源证据触发，actor-flow/trigger 只用于定位核心窗口 |
| MergerIntoSlowTrafficV2 | 与 Merger 同属高速合流，但全量 RGB 未发现稳定灯控子集 | 候选删除 R1/R4；主线高速/快速路保持 R3，不继承 MergerIntoSlowTraffic 的少量 R4 子集 |
| NonSignalizedJunctionLeftTurn | bbox traffic_light 与 stop_sign 同时报出，弱灯控提示把无灯/STOP 左转误升 R4 | strict no-R4；无灯/priority/stop 证据给 R5；bbox/static signal 弱提示只写 review，不动态打开 R4 |
| NonSignalizedJunctionLeftTurnEnterFlow | EnterFlow 名称误触发 R3，且 bbox traffic_light 与 stop/yield 冲突时误升 R4 | 这是无灯路口进入车流，主口径 R5，不是匝道 R3；同样 strict no-R4 |
| NonSignalizedJunctionRightTurn | slip lane / 右转动作被误当 R3，或少量灯控子集被场景名压成 R5 | 大多数无灯 junction 给 R5；少量灯控右转可给 R4，但必须有逐帧 RGB/meta/bbox 灯控证据；右转动作本身不是 R3 |
| OppositeVehicleRunningRedLight | 闯红灯事件被误当 R5 | 信号规则仍有效，主 RS 是 R4，违规进 EVENT |
| OppositeVehicleTakingPriority | priority/no-light 与正常灯控冲突 | 以 priority/no-light R5 为主；全量 RGB 有少量灯控子集，R4 只在灯控同源证据成立时开放 |
| ParkedObstacle | parked 字样误触发独立停车结构；STOP/无灯路口被压成 R1 | ParkedObstacle 是障碍 EVENT；不是 ROAD_STRUCTURE；非 TwoWays 默认不自动升 R2；STOP/无灯路口可 R5；绕行恢复用中心线回正截断 R-E2 |
| ParkedObstacleTwoWays | parked / TwoWays / 停车结构混淆；STOP/无灯路口被压成 R1/R2 | 停车/障碍占掉侧向 lane 后按有效可行驶对向单车道 R2；parked 本身不是 RS；真实 STOP/无灯路口可 R5；R-E2 到中心线后释放 |
| ParkingCrossingPedestrian | 行人横穿与停车结构混淆 | 停车空间不单独成 RS；行人进 U-E4；普通段 R1，灯控/无灯路口优先 R4/R5 |
| ParkingCutIn | 切入事件或 shoulder hint 误触发独立停车结构；STOP/无灯路口被候选缺失压成 R1 | 全量逐帧 RGB 未见稳定独立停车结构，候选收紧为 R1/R4/R5；真实 STOP/无灯路口走 R5，切入本身留给 U-E3 |
| ParkingExit | 停车驶出完成后仍保持停车结构 | parking-to-driving transition 用 R1 + R-E2 表达，完成后回 R1/R-E1；若两侧停车压缩有效通道则 R2 |
| PedestrianCrossing | 行人/斑马线被直接当 R4/R5 | RS 由 crossing 是否同源于路口控制决定，行人进 EVENT |
| PriorityAtJunction | 夜间/低能见度下把 R5 当置信度异常，或把真实灯控城市路口漏成 R5 | 可见 traffic light/stopline/controller 时 R4；priority/stop/yield 无灯段给 R5；普通离开段回 R1 |
| RedLightWithoutLeadVehicle | is_junction=false 时漏掉 stopline approach | 有效灯态/stopline approach 同源即可 R4；非 TwoWays 默认不自动升 R2 |
| SignalizedJunctionLeftTurn | 左转让行被误当 R5 | 信号灯左转仍是 R4，让行/冲突进 EVENT |
| SignalizedJunctionLeftTurnEnterFlow | EnterFlow 字样误触发 R3 | 信号灯路口内仍 R4，enter-flow 只进 EVENT |
| SignalizedJunctionRightTurn | 右转专用道或稀疏 route 误触发 R3/R1 | 有效 signal/controller/stopline 给 R4，离开后 R1 |
| StaticCutIn | 静态车切入来源不明，R3/R1/R4/R5 混淆 | merge/highway 来源 R3，都缺证据则 R1；R-E2 只给初始目标车道修正，U-E3 仍靠 cut-in 对象证据 |
| T_Junction | 默认 T 路口被误扩展成单一 R4 或单一 R5 | 保留 R1/R4/R5；灯控 T 给 R4，无灯/STOP/yield T 给 R5，必须逐帧看 RGB 控制源；非 TwoWays 默认不自动升 R2 |
| VehicleOpensDoorTwoWays | 开门停车侧与双向借道冲突；STOP/无灯路口被压成 R1/R2 | 两侧停车/开门风险占用侧向 lane 时主 RS 为 R2；不再输出独立停车主标签；紧接 U-E2 的恢复 R-E2 起点提前最多 3 帧、终点提前最多 4 帧；STOP/无灯路口 regular 可 R5 |
| VehicleTurningRoute | 投影误差高时普通转弯前路段被过宽 junction window 标 R4/R5 | 高误差无灯 R5 必须有 stop/is_junction/可信 XODR 近路口证据；非白名单 route 默认不自动升 R2 |
| VehicleTurningRoutePedestrian | 行人事件和转弯路口窗口导致普通住宅道路误标 R5；非 TwoWays R2 需要重新逐帧审 | 行人进 EVENT；普通同向路段 R1，非 TwoWays R2 暂停动态开放，待全量 id 逐帧 RGB 确认后再加 route 白名单；真实 stop/priority/路口段 R5，灯控路口 R4 |
| noScenarios | 弱静态 topology hint 或瞬时灯态在无事件场景中制造特殊 RS；稳定灯控 approach 被 strong context 过严漏成 R1 | 保守 R1/R4/R5；非 TwoWays 默认不自动升 R2；R4 需要稳定 meta 灯态 + bbox 灯 + junction window，R5 需要 STOP/无灯控制证据 |

### Accident

- 候选 RS：R1, R4, R5。
- 已确定口径：事故障碍本身是 EVENT，不是 ROAD_STRUCTURE；默认 R1，进入真实信号灯路口才切 R4。
- 分段逻辑：finite `dist_to_accident` / active scenario 只用于标记事故事件窗口；窗口内仍按道路结构判断 R1/R4。
  Accident route 前 30 帧如果只是被弱静态 junction / signal hint 误标成 R4/R5，route 级后处理强制回 R1；
  但若同帧存在有效灯态、bbox traffic_light、STOP/yield 或明确 junction 控制源，说明初始确实在路口控制区，
  保留原始 R4/R5。被压回的帧会把对应 R-E4/R-E5 常规事件重算为 R-E1；Town13 除外，按原始证据保留。
  其它事故起始接近段不允许凭十字路口窗口覆盖正常跟车。
  EVENT 层按轨迹分三段：开头只有 XML trigger/对象减速距离、但无具体障碍距离和变道轨迹时回 R-E1；
  为绕障离开原车道 + 绕过障碍核心为 U-E2；绕过后负向/收敛的 lane-change 轨迹切 R-E2；
  回正结束后释放为当前道路常规事件。2026-07-05 起，U-E2 距离触发后移 3m
  （32m -> 29m），`speed_reduced_by_obj_distance` 辅助触发再减 2m；R-E2 恢复段使用
  `meta["route"]` ego-frame 1-8m 近前方局部切线横向误差，确认自车在弯路上也确实正向原/目标
  中心线收敛，而不是拿全局直线、前方 waypoint y 值或 late `changed_route` 硬补。
  2026-07-05 二次修正后，过障碍核心后的 `signed_dist_to_lane_change <= -0.45m` 可作为
  “准备回原/目标车道”早期信号，R-E2 起点可提前 1-2 帧；退出要求未来 3 帧稳定在中心线容差内。
  2026-07-05 三次修正后，Accident 的 junction window 从 `60/25m` 小幅收为 `54/22m`，
  避免普通绕障片段被过早 R4/R5 化；但 `meta TL + bbox traffic_light` 与 `STOP/yield`
  可见控制源仍可在稀疏 XML window 外恢复真正 R4/R5。
  U-E2 若刚被释放但后续 1-3 帧仍显示 `changed_route` / `signed_dist_to_lane_change`
  与局部 route 中心线横向偏移、且没有回正趋势，则补回 U-E2，表示避障换道尚未彻底完成。
  障碍最近点之前，正向 `signed_dist_to_lane_change` 下降只能说明自车正在完成避障换道，
  仍保持 U-E2；R-E2 起点必须贴近/越过障碍最近点，或已经出现负向回原/目标车道趋势。
  R-E2 不再强制要求前 24 帧内出现 U-E2；只要 XODR/meta 中心线证据显示自车正在独立换道或回正，
  即可保留为 R-E2，否则才压回 regular。
  非 TwoWays Accident 额外按 `dist_to_accident_site` 最近点截尾：最近点之后距离开始回升且
  ego 已回中心线时，即使仍在 29m 阈值内也不再保持 U-E2；若仍处在回正轨迹，只给短 R-E2。
- 证据需求：XML accident trigger + meta distance 确定事件窗口；R4 必须有有效灯态且具备强路口/stopline/signal-junction 上下文，
  或可信 XODR/meta junction 与静态 signal 同源。仅有 static signal near / distance-to-junction 不足以把普通路段升 R4。
- 待完善点：不要把同向绕障误升为 R2；若 projection error 高，事故窗口只给事件候选，不改 RS。
  若 U-E2 远离 XML trigger/障碍距离且无 route-change 轨迹，必须释放；路线末尾仍 U-E2 必须 event review。

### AccidentTwoWays

- 候选 RS：R2, R4, R5。
- 已确定口径：two-way 障碍核心窗口内，优先用可信 XODR 证明对向 lane 参与；若 XODR 不可信，近距离障碍、
  `accident_two_ways_stuck`、`vehicle_hazard` 或明显 lane-change 核心证据可给 R2=0.90，并强制 review。
  route 前 30 帧如果被静态 junction / signal hint 误标成 R4/R5，route 级后处理强制回 R2，
  并把对应 R-E4/R-E5 常规事件重算为 R-E1；Town13 除外，按原始证据保留；其它事故双向起始接近段不允许凭十字路口窗口覆盖正常道路。
  如果 R2 核心与 R4/R5 路口控制区重叠，RS primary 可以仍是 R4/R5，但 EVENT 层必须启用
  R2 overlay：U-E2/R-E2 优先于 R-E4/R-E5，不能让路口常规事件覆盖借对向绕障/回正事件。
  `Town07_route_001454` 这类 XML 明确给出 `distance=47`、`frequency=33-160`、trigger 在 route 起点，
  且 meta 长时间有近距离事故障碍 / `vehicle_hazard` / `scenario_obstacles_ids` 的 case，
  即使 route projection error 导致 `two_way_window` 失效，也必须由 meta 核心证据召回 R2；
  强 R2 核心分数和优先级高于 bbox traffic_light/R4，不能只把 R2 放 secondary。
  若 meta 核心证据缺失但 XML trigger 已极近，或 trigger-close 且 XML 场景障碍进入近距离窗口，
  给短核心 R2=0.88，用于避免“明明 XML 已到 R2 核心附近却仍全 R1”。
  2026-07-09 复核 `Town12 route1218_0`：f7-f24 RGB 为雾天直路/双向单车道，路边 warning sign
  被 bbox 误报为 stop_sign，meta/XODR 距路口仍三百米以上；因此 bbox-only stop/sign 且无
  meta junction、bbox junction hint、可信 XODR junction 或有效灯控时，前 30 帧强制回 R2，
  XML 事故 trigger 不能单独制造 R5/R-E5。
  RGB-first 复核进一步确认：`*_TwoWays` 的 R2 按有效可行驶通道定义；黄中心线窄路和两侧停车/障碍压缩 lane 的四车道路段都应持续 R2。
  自车必须借/等对向的那一小段由 U-E2/R-E2 表达。
  灯控路口仍优先 R4；STOP/无灯/路权路口走 R5。
- 分段逻辑：先判 R4；再判 R2；R2 结束后再次按 meta/XODR 灯控判 R4。核心障碍/借道帧由 XML trigger / distance / active window + opposite driving lane /
  同向可用 lane 不足 / meta obstruction 支撑；确认双向单车道的非核心帧保持 R2 + R-E1，
  纯场景名 layout-prior 已改为有效可行驶 lane 判定；非路口不回 R1。若后段有 valid traffic light 或 XODR 近信号 + junction context，则切 R4；STOP/无灯控制源切 R5。
  旧 `R1 f0-f55 -> R2 ... -> R1 ...` 边界作废；非路口 TwoWays 直道应保持 R2。
- 证据需求：XODR lane_id 符号反转、lane direction、lane count、meta active scenario、近距离障碍距离、
  `scenario_obstacles_ids`、`signed_dist_to_lane_change`、RGB 对向借道边界。
- EVENT 复核补充：AccidentTwoWays 与其它 TwoWays 障碍共享 obstacle-recovery 后处理；
  U-E2 只覆盖借/等对向和绕过障碍核心。TwoWays 与同向障碍不同，R2 借对向车道绕障段本身
  就是强 U-E2 证据，必须从进入/准备进入借道核心开始标，而不能等绕完后由后段距离/减速补标。
  若同一 route 内出现多个 U-E2 片段，优先保留和最终 R2 核心重叠最多的片段；恢复规则造成的
  6 帧以内短 `U-E2 -> R-E2/R-E1 -> U-E2` 断裂应合回 U-E2。
  `U-E2 -> R-E2` 后如果只剩一小段非路口 R-E1，且马上进入 R4/R5 或 route 结束，
  应继续并入 R-E2，表示借对向绕障后的回目标/原车道尾段，不能过早释放成普通跟车。
  2026-07-06 修正后，TwoWays 核心结束后 24 帧内只要仍有回目标/原车道变道证据，
  `U-E2/U-E3` 必须接入 R-E2；最终一致性检查后还会桥接短非路口
  `U-E2/U-E3 -> R-E1 -> R-E2` gap。`AccidentTwoWays` 对 16 帧以内的
  `U-E2 -> R-E1 -> R-E2` 空洞按证据分摊：仍在借对向/障碍核心的前段并入 U-E2，
  出现回目标/原车道轨迹的后段并入 R-E2；更长的 R-E1 段保留给 RGB 复核，不强吞。
  核心后 16 帧内的弱 U-E4 若没有近距离行人/骑行者
  或 walker/emergency hazard 证据，也按恢复变道改回 R-E2，避免 `U-E2 -> U-E4` 假跳转。
  但 TwoWays 核心障碍仍 close / stuck / hazard 时，不能因为 `signed_dist_to_lane_change`
  或路口控制源提前释放成 R-E2/R-E4；此时必须保持 U-E2，直到核心障碍解除。
  R-E2 可以在回正趋势开始前 1-2 帧起标，但一旦 ego-frame route 横向偏移回到中心线容差内
  就释放为 R-E1/R-E4/R-E5。
- RGB 复核结论：本轮 TwoWays smoke 覆盖 Accident/Construction/Hazard/Parked 四类各 5 条 route，
  均能召回核心 R2；旧 `AccidentTwoWays/Town01` 的 `R1 -> R2 -> R1 -> R4` 边界已作废，
  非路口 TwoWays 直道保持 R2，再由真实红绿灯/STOP/无灯控制源切 R4/R5。
- 待完善点：`Town15` 有强停车/路边车列视觉，当前仍归 R2 + review；后续复核时要确认它是纯 TwoWays road-layout，
  还是应在 EVENT/secondary 里记录 parking semantics。

### BlockedIntersection

- 候选 RS：R1, R4, R5。
- 已确定口径：blocked intersection 是路口内阻塞事件；阻塞本身只进入 EVENT，不决定 R4/R5。
  有正常灯控/信号灯同源证据时 primary R4；RGB/meta/bbox 显示 STOP、yield、priority 或无灯路口控制源时 primary R5。
- 分段逻辑：接近/进入 junction 时先判断控制源；该场景十字路口窗口专项压缩 20%
  后又按 RGB 再收紧进入侧（`junction_pre_m=32`, `junction_post_m=18`），离开 junction 后回 R1；阻塞对象作为 EVENT。
  如果 R4 不是由有效 `traffic_light_state` 支撑，必须写
  `blocked_r4_without_meta_tl_requires_rgb_confirmation` 并逐帧看 RGB；低能见度 contact sheet
  只能用于定位，最终以单帧 RGB 的 traffic light/stopline/STOP/yield/cross traffic/blocked pocket 为准。
  若同一路口前段已稳定 R4 且后段没有 STOP/yield 证据，尾段灯态/bbox 灯因视角丢失时仍保持 R4；
  `Town12_route_2399_0` 这类灯控路口出口不能仅因 `is_junction=True` + 灯态缺失切成 R5。
- 证据需求：RGB 控制源、meta traffic_light_state / stop_sign_close / is_junction、bbox traffic_light / stop_sign / yield、
  XODR junction + signal/controller、XML trigger。
- RGB 复核结论：2026-07-04 全场景逐帧审查发现 `Town12_Rep0_1881_0_route0_01_10_00_50_44`
  是 STOP/无灯阻塞路口，旧逻辑误标 R4；修复后该 route 为 `R5=106, R1=5`，EVENT 仍覆盖 U-E1/U-E8。
  其它 BlockedIntersection route 仍可存在真实灯控 R4，不能场景级禁 R4。
- 待完善点：低能见度下若只有 static signal 近邻且没有 meta/RGB/bbox 灯控同源证据，应保持 R4 review；
  若人工 RGB 看到 STOP/无灯控制源，应优先 R5，即使 XML/XODR 投影误差较高。

### ConstructionObstacle

- 候选 RS：R1, R4。
- 已确定口径：施工障碍是 EVENT；非 TwoWays 场景不应自动进入 R2。
- 分段逻辑：施工窗口内仍按普通同向道路 R1，真实灯控路口覆盖为 R4。
  EVENT 层同 Accident：避障离道和绕过施工障碍为 U-E2，回目标车道为 R-E2，完成后回 R-E1/R-E4。
  2026-07-05 起，施工障碍 U-E2 距离触发后移 3m（35m -> 32m），恢复 R-E2 只保留
  ego-frame route 中心线收敛到回正的短段，防止 U-E2 后半段或 R4 后段被 late R-E2 覆盖。
- 2026-07-08 复核 `Town06 route_002028`：施工 U-E2/R-E2 恢复链正好进入信号灯路口时，
  不允许 R4 的 R-E4 直接覆盖事件；仍在施工车/锥桶核心旁边保持 `R-E4 + U-E2`，
  过最近点后切 `R-E4 + R-E2`，直到回正链自然结束或 overlay 上限触发。
- 证据需求：XML construction trigger + meta distance 定位事件，XODR/灯态确认 R4。
- 2026-07-09 RGB 抽样后进一步收紧进入十字路口范围：`junction_pre_m` 从 60m 改为 42m，
  post 仍为 25m，避免普通施工/跟车接近段被提前标为 R4/R5。
- RGB 复核结论：`Town04` 曾在 f203 过早升 R4，单帧 RGB 仍是雾中普通弯道/跟车；
  根因是 static signal + 有效灯态在 35m 内就触发 strong_control_context。代码已把 static-signal-only
  strong context 收紧到 25m，R4 起点后移到 f209；施工主体仍全段 R1。
- 待完善点：需要 RGB 边界帧确认施工物是否实际导致 lane closure；没有 two-way/opposite 证据时不升 R2。
  trigger-only 初始 U-E2 和路线尾端粘滞 U-E2 均视为事件逻辑问题，不作为训练标签保留。

### ConstructionObstacleTwoWays

- 候选 RS：R2, R4, R5。
- 已确定口径：two-way 施工障碍所在道路按有效可行驶对向单车道 R2；施工核心由 U-E2/R-E2 表达。
- 分段逻辑：R4 优先；two-way construction 窗口内满足对向参与条件给 R2；
  施工物通过后若仍被 XODR/meta 确认为双向单车道，ROAD_STRUCTURE 保持 R2，EVENT 回 R-E1/R-E2；
  若没有路口控制源，非核心段仍保持 R2；真实灯控/STOP/无灯路口才切 R4/R5。
- 2026-07-09 RGB 抽样后进一步收紧进入十字路口范围：`junction_pre_m=42` 且
  `junction_tighten_factor=0.70`，有效进入侧约 29.4m；twoways 的 XML 障碍 trigger
  不能单独作为 R5/junction trigger。
- 证据需求：XML distance/offset、XODR opposite lane、meta active scenario、RGB 施工占道。
- RGB 复核结论：静态施工/障碍不是持续道路结构；后续按 `AccidentTwoWays` 同款核心 span 口径复跑。
- 2026-07-08 十字路口灵敏度复核：RGB sheet 显示 465 个有效 route 中约 47% 无可见 R4/R5，
  真实路口帧占比约 14.7%，因此在全局窗口后追加 `junction_tighten_factor=0.70`
  并降低场景级 pre/post 保底到 10m/3m；smoke 6 route 后有效窗口样例为 15.12m/4.9m，
  分布为 R2 84.3%、R4 15.7%，没有把 two-way/施工核心压成 R4/R5。
- 待完善点：施工锥桶/障碍只在路边但不压占自车路径时，应保持 R1 + EVENT。

### ControlLoss

- 候选 RS：R1, R4, R5。
- 已确定口径：失控/急刹是 EVENT，不改变 RS；默认 R1，正常灯控路口为 R4，STOP/yield/无灯路口为 R5。
- 分段逻辑：speed、brake、accel、yaw 异常只进入事件层；RS 由道路结构和控制源证据决定。
  ControlLoss 的 XML route_s 常漏掉长时间可见控制源，因此 `meta traffic_light_state + bbox traffic_light`
  可在 XML window 外恢复 R4，`bbox/meta STOP/yield` 可在 XML window 外恢复 R5。
- 证据需求：meta speed/brake/accel 作为事件证据；meta/bbox traffic_light 给 R4；
  STOP/yield/meta junction/XODR 近路口或可见 STOP/yield 控制源给 R5。
- RGB 回灌：2026-07-05 三场景回归中，开放 R5 后 `blind_R5_label_R1` 从 4755 帧降到 2094 帧。
  同日晚间全量 top mismatch 复核后，加入“可见控制源可越过稀疏 XML window”规则；
  ControlLoss 子集总 mismatch 从 2441 降到 64，`blind_R5_label_R1` 从 2094 降到 1，
  `blind_R4_label_R1` 从 339 降到 56，副作用仅 `label_R4_without_rgb_junction_signal=6`。
- 2026-07-08 十字路口灵敏度复核：RGB sheet 显示可见路口真实存在，但普通失控/跟车帧里
  possible turn marking、远灯、雾天弱 hint 也多，故追加 `junction_tighten_factor=0.60`
  并把场景级 pre/post 保底降到 8m/3m；smoke 6 route 后有效窗口样例为 10.8m/4.2m，
  R1 68.7%、R4 26.2%、R5 5.1%，真实灯控仍能保留。
- 2026-07-09 针对 Town01-04 起始段误路口加入 route 级覆盖：前 30 帧只有有效灯态+灯框+本地上下文、
  或 RGB/bbox junction hint 成立时才保留 R4/R5；meta-only、XODR-only、stop-only 的起始伪路口回 R1/R-E1。
- 待完善点：长时低速不能误判为停车结构；雾天只有远处红点或单帧 lane marking 时不扩大 R4/R5，必须持续成段并有 bbox/meta 控制源。

### CrossJunctionDefectTrafficLight

- 候选 RS：R1, R4。
- 已确定口径：defect traffic light 场景中，红绿灯硬件/受控路口仍归 R4；信号失效、规则源失效、异常灯控语义由 EVENT 的 U-E7 表达，不再把 RS 改成 R5。
- 分段逻辑：trigger 对应 junction/stopline 前后窗口仍是基础，但必须有本地路口/近灯控证据才设 R4；远距离 `traffic_light_state`、`light_hazard` 或 bbox traffic_light 只作为候选/复核线索，不能单独覆盖起始直道；窗口外 R1。
- EVENT 耦合：有效路口阶段 regular 为 R-E4，并叠加 U-E7；若四向车辆冲突明显，同帧可叠 U-E6。
- 证据需求：XML trigger / traffic_direction / source_dist_interval、XODR junction/signal/controller、meta active scenario、RGB 确认红绿灯硬件与路口。
- 2026-07-08 `Town12 route_002107` 与同场景均匀采样 RGB 复核：原规则因 XML trigger 在 route 起点、meta/bbox 远灯态有效而从 f0 起全程 R4；现按 `junction_tighten_factor=0.65` 把 R4 进入压到约 20-23m 本地路口证据段。2107 从 `R4 f0-45` 修正为 `R1 f0-18 -> R4 f19-45 -> R1 f46-53`，与 RGB 中 f0-18 直道、f19 后接近/通过红绿灯路口相符。
- 待完善点：实测 map `is_junction=false` 仍可能需要 R4，说明 stopline approach、signal/controller 和 junction polygon 需要分开建证据；但远灯态必须受近路口距离或视觉路口几何约束。

### CrossingBicycleFlow

- 候选 RS：R1, R4。
- 已确定口径：自行车横穿/流量是 EVENT；有正常信号灯路口才 R4，否则保持 R1。
- 分段逻辑：bicycle flow 窗口只改变事件；traffic light / controlled junction 同源时切 R4。
- 2026-07-09 RGB 抽样后进一步收紧进入十字路口范围：`junction_pre_m` 从 50m 改为 35m，
  并继续乘 0.90，等效进入侧约 31.5m；自行车横穿本身不反推 R4。
- 证据需求：XML bicycle trigger、meta active scenario、traffic_light_state、XODR junction/signal。
- EVENT 复核：2026-07-08 `Town12 route823_{0,1,2,3}` 曾出现
  `U-E4 -> R-E4 -> U-E4` 假断裂；现按 `dist_to_biker/dist_to_pedestrian`、
  `nearest_ped_bike_m`、walker/emergency hazard 与 `event_crossing_distance`
  构成连续证据带，每条 route 最多保留一段连续 U-E4，内部短 regular gap 合并。
  2026-07-09 复核 `Town12 route2385_{0,1,2,3}`：持续 bicycle flow 中
  `dist_to_biker` 会在遮挡/夜间/横穿队列中短空，CrossingBicycleFlow 的内部 gap 合并上限
  单独放宽到 14 帧，主 span 内不保留单帧 `R-E4/R-E1` 洞；全量 48 route smoke 后仍为
  每 route 仅一段 U-E4。
- 待完善点：不要因为 actor flow 字样误归 R3；R3 只给 merge/split/ramp 拓扑。

### DynamicObjectCrossing

- 候选 RS：R1, R4, R5。
- 已确定口径：动态对象横穿是 EVENT，不直接定义道路结构；但道路控制源必须如实表达。
  2026-07-05 全量 RGB 盲审发现多条 Town13 route 明确经过 STOP/无灯路口，
  若候选缺 R5 会被错误压成全程 R1。
- 分段逻辑：多数普通道路帧 R1；稳定灯控路口为 R4；STOP/无灯路口为 R5；
  车辆/动态切入类横向干扰可触发短 span U-E3，行人/骑行者/小动态对象横穿交互可触发短 span U-E4，
  但必须由对象距离/类型/自车响应限定，不能把整个 R4/R5 路口都当非常规。
- 2026-07-08 RGB 抽查 `Town03/Town04/Town05/Town12/Town13` 多条 route 后，进一步压缩十字路口：
  `junction_tighten_factor=0.65`，`junction_min_pre_m=8m`，`junction_min_post_m=2.5m`。
  R4 直接触发需要近 junction / close-trigger-for-junction / bbox junction；有效 meta 灯态 + bbox traffic_light
  也必须靠近控制源。R4 context recovery 与 temporal smoothing 在本场景不能只凭远灯或 weak bbox 续命。
  route 起始 8m 内若没有近 junction / bbox junction，弱 STOP/R5 证据保持 R1。
- 2026-07-09 逐帧 RGB 复核发现两类相反错误需要同时处理：
  `Town04_Rep0_Town04_Scenario3_27...` 是直道/隧道段，U-E4 横穿事件不应反推 RS 为 R4；
  `Town10HD_Rep0_Town10HD_Scenario3_{2,10}...` 是真实信号灯路口，过度依赖 XODR near-junction 又会漏成 R1。
  因此 DynamicObjectCrossing 的 R4 gate 改为“短弱 bbox 灯/close-trigger 不够，长稳定灯控证据必须恢复”：
  少于 6 帧且只有 `r4_dynamic_crossing_meta_bbox_light_near_control` 的弱 R4 片段回 R1；
  连续稳定灯控段可在 `light_count>=24` 且 `meta_light_count>=12` 时整段恢复 R4，
  已进入稳定 R4 前最多 4 帧的 close-light tail 也可补回 R4。
  全量 smoke：294 route / 42608 frame，R1/R4/R5 为 27477/10169/4962，U-E4 仍为 5520 帧。
- 证据需求：RGB stopline/crosswalk/STOP、bbox stop/yield、meta stop/is_junction/active scenario、
  XODR/traffic light。XODR 拓扑不可信时，连续 bbox/meta STOP + RGB 路口证据可召回 R5。
- RGB 回灌：2026-07-05 对 `Town03_Rep0_Town03_Scenario3_6...` 等 top mismatch 逐帧复核后，
  恢复 U-E3/U-E4 候选；DynamicObjectCrossing 子集审计中
  `event_regular_during_rgb_object_or_motion_activity` 从 2363 降到 2030 帧，剩余主要是 blind motion detector 对侧向车/远处对象的宽判。
- 待完善点：动态对象类型仍混合，后续应继续按 actor 类型和交互距离细分 U-E3/U-E4 的边界。

### EnterActorFlow

- 候选 RS：R1, R3。
- 已确定口径：2026-07-04 全量逐帧 RGB 审计未见稳定真实灯控路口；该类基本是高速/快速路进入车流，不开放 R4。
  但距离 actor-flow/merge 起点较远的普通直道不应强行全程 R3，可保持 R1/R-E1，靠近汇入控制区后再切 R3。
- 分段逻辑：actor-flow start/end / trigger 窗口 + XODR merge 拓扑提高 R3 置信；`actor_flow_distance<=40m`
  或近 trigger 的局部窗口进入 R3，窗口外普通直道保持 R1，meta 伪灯态不动态加入 R4。
  EVENT 层中匝道进入/准备汇入段保持 R-E3，真实目标变道优先切 R-E2，merge 完成后的主线正常跟车才回 R-E1。
  2026-07-09 修正后，EnterActorFlow* 的 R-E3 回填上限为 36 帧，至少保留约 16 帧准备汇入段；
  `changed_route/signed_dist_to_lane_change/route_lateral_abs` 与 actor-flow 近邻共同确认 R-E2，避免变道被 R-E3 吃掉。
  后续又加入围绕已有 R-E2 核心的轨迹补全：即使 `route_lateral_abs` 峰值只在一两帧出现，只要
  `changed_route + signed_dist_to_lane_change` 连续显示目标变道过程，前后帧也补入同一段 R-E2。
- 证据需求：XML start/end actor flow、other_actor_location、XODR merge/split、meta active scenario、RGB 车流关系。
- RGB 复核结论：2026-07-04 全量审计 80 个有效 route，R4 比例为 0；2026-07-09 EnterActorFlow*
  全量回归 123 条 route / 13490 帧输出 R1=3866、R3=9624，EVENT 为 R-E1=6214、R-E2=1613、R-E3=5663；
  R-E2 连续段最短长度 EnterActorFlow=7 帧、EnterActorFlowV2=11 帧，1-3 帧短碎片为 0。
  2026-08-09 四问 RGB 复核进一步确认：`EnterActorFlow/R1/R-E1` 虽然 RS 为 R1，但画面仍是高速/快速路
  actor-flow/merge 拓扑，四问 `HIGHWAY=YES`；提示词不能把 RS=R1 机械翻译成非高速。
- 待完善点：如果未来发现城区 EnterActorFlow 子集，需要按 run 级 RGB/XODR 分流，而不是恢复全局 R1 候选。

### EnterActorFlowV2

- 候选 RS：R1, R3。
- 已确定口径：与 EnterActorFlow 同口径；远离 actor-flow/merge 起点的普通直道可保持 R1，不开放 R4。
- 分段逻辑：actor-flow 核心窗口提高 R3 置信；窗口外普通直道可保持 R1，meta 伪灯态不动态加入 R4。
  EVENT 层同 EnterActorFlow：准备汇入段保持 R-E3，真实目标变道切 R-E2，完成后才释放 R-E1。
- 证据需求：XML start/end、meta active scenario、XODR ramp/merge、RGB 侧向汇入。
- RGB 复核结论：`rs_full_frame_review_highway_twoways_user_fix` 一条 route 共 133 帧，输出 `R3=133`。
  2026-08-09 四问 RGB 复核进一步确认：`EnterActorFlowV2/R1/R-E1` 同样应答 `HIGHWAY=YES`，
  因为可见拓扑是高速/快速路并入/actor-flow 背景，不是普通直宽道路。
- 待完善点：短 route 下 projection 抖动更容易污染边界，必须用 RGB + meta/XODR waypoint 吸附辅助。

### HardBreakRoute

- 候选 RS：R1, R3, R4, R5。
- 已确定口径：急刹是 EVENT；道路背景可能是城市/乡村 R1，也可能是高速/快速路 R3；
  RGB 逐帧复核发现多条 route 会经过 STOP/无灯 T/十字路口，必须允许 R5/R-E5。
- 分段逻辑：brake/accel/speed 只决定事件窗口；先按整条 route 的 RGB 分桶。
  高速/快速路桶候选收敛为 R3/R4；非高速桶保留 R1，城市街区/乡村普通道路保持 R1，
  灯控/stopline 段 R4；STOP/无灯 T/十字路口段 R5。雾天远处红点、单帧 stopline/turn-marking hint
  不足以扩大成 R5，必须是一段连续路口 traversal。
- 证据需求：meta brake/speed、XML route、XODR signal/junction/ramp/lane count、RGB 道路形态。
- RGB 复核结论：HardBreakRoute 97 个 route 已逐 id 抽取均匀 5 帧 stitched RGB 复核，不能只按 Town12/13 判高速；例如
  `Town12_Rep0_258_0_route0_01_08_09_35_42` 是乡村普通路，保留 R1。
  当前筛入高速/快速路桶 16 个 route：
  `Town12_Rep0_1428_0_route0_01_10_00_18_08`,
  `Town12_Rep0_1439_0_route0_01_10_04_51_04`,
  `Town12_Rep0_2452_0_route0_01_11_03_14_24`,
  `Town12_Rep0_2510_0_route0_01_09_22_05_04`,
  `Town12_Rep0_2585_0_route0_01_09_13_23_09`,
  `Town12_Rep0_4115_0_route0_01_09_08_38_49`,
  `Town12_Rep0_4118_0_route0_01_09_11_51_26`,
  `Town12_Rep0_4139_0_route0_01_11_01_24_29`,
  `Town12_Rep0_947_0_route0_01_10_04_45_34`,
  `Town12_Rep0_954_0_route0_01_10_01_42_01`,
  `Town13_Rep0_1258_0_route0_01_09_16_21_37`,
  `Town13_Rep0_1269_0_route0_01_07_23_43_15`,
  `Town13_Rep0_1275_0_route0_01_09_14_40_40`,
  `Town13_Rep0_1387_0_route0_01_08_06_02_20`,
  `Town13_Rep0_1663_0_route0_01_09_13_45_51`,
  `Town13_Rep0_1666_0_route0_01_10_04_27_25`。
- RGB 回灌：2026-07-05 子集审计开放 R5 后，HardBreakRoute 的 `blind_R5_label_R1`
  从 1001 帧降到 104 帧；top 剩余如 `Town13_Rep0_1493_0...` 为雾天/远处灯点/路面线导致 blind 过宽，
  人眼未见稳定路口，不继续扩大。
- 待完善点：急刹靠近路口时要区分“红灯停车 R4”“STOP/无灯路口 R5”“前车急刹 R1 + EVENT”和“高速急刹 R3 + EVENT”。

### HazardAtSideLane

- 候选 RS：R1, R4, R5。
- 已确定口径：非 TwoWays 侧向 hazard 不直接进入 R2；默认 R1，只有真实灯控/STOP/无灯源才进入 R4/R5。
- 分段逻辑：XML `bicycle_drive_distance/bicycle_speed` 和逐帧 RGB 确认核心对象为自行车/行人；
  `dist_to_biker<=30m` 的可交互段连续输出 U-E4，对象离开后若 8 帧内仍在回目标车道/横向回正，
  从 U-E4 后第一帧直接切 R-E2，不允许夹 R-E1；有灯控/STOP/无灯路口时 RS 可切 R4/R5，
  但 U-E4 或 R-E2 恢复仍作为同帧事件/overlay 保留。
- 证据需求：XML bicycle 参数、meta `dist_to_biker/changed_route`、逐帧 RGB、XODR/traffic light。
- RGB 复核结论：2026-07-09 全量 94 条有效 route 中，77 条实际触发自行车事件且每条只有
  一段连续 U-E4，最短 14 帧；17 条 `Town13 route30_*` 的 meta 全程
  `scenario_active=None, dist_to_biker=inf`，完整 RGB 也未出现自行车/行人，因此不硬造 U-E4。
- 初始保护：前 30 帧仅 bbox-only STOP、close-trigger 或 untrusted XODR 时回 R1；本轮全量清除
  393 帧初始弱路口，代表 `Town13_1619_10` 开头暗光直道恢复 R1。
- 待完善点：若局部 XODR 证明实际需要借对向，应作为数据异常或 review，而非直接改规则。

### HazardAtSideLaneTwoWays

- 候选 RS：R2, R4, R5。
- 已确定口径：two-way side-lane 场景按有效可行驶对向单车道 R2；侧向自行车/行人是否正在影响当前决策由 U-E4 表达。
- 分段逻辑：有效可行驶对向单车道保持 R2 + regular event；灯控路口 R4；STOP/无灯/路权路口 R5；
  `dist_to_biker<=30m` 且自行车/行人进入路径时进入 U-E4，对象离开后仍在借对向/横移回正时直接进入 R-E2。
- 证据需求：XML bicycle 参数、XODR lane direction、meta `dist_to_biker/changed_route`、RGB 对向/侧向动态对象。
- RGB 复核结论：TwoWays 家族边界复核中 `HazardAtSideLaneTwoWays` 主要是乡路/窄路/等效对向单车道；
  旧版把核心前后压回 R1 的口径已作废。2026-07-09 全量 88 条有效 route 均召回单段连续 U-E4，
  最短 28 帧；代表路线 `Town12 route467_0` 为 f94-f128 U-E4、f129-f132 R-E2，
  `Town13 route1302_2` 为 f48-f79 U-E4、f80-f81 R-E2。
- 初始保护：前 30 帧无真实 junction/control 的 bbox-only STOP 回 R2，本轮清除 336 帧初始伪 R5；
  真实初始信号路口 `Town12_1067_0` 仍保持 R4。
- 待完善点：继续用 RGB 复核 U-E4/R-E2 边界，避免远处自行车过早触发或回正阶段仍粘在 U-E4。

### HighwayCutIn

- 候选 RS：R3, R4。
- 已确定口径：全量逐帧 RGB 确认主体仍是高速/快速路 cut-in 背景；他车切入是 EVENT，底层道路空间默认 R3。
  但存在少量真实灯控子集，不能再把整个 scenario 放入 no-R4。
- 分段逻辑：merge/ramp/split/exit 或 lane-count-change topology 提高 R3 置信；无这些强 topology 时仍保持 R3 默认；
  R4 必须由逐帧 RGB 可见灯控 + meta/bbox 灯控同源证据触发；匝道/导流线/停车线不能单独制造 R5。
- EVENT 口径：默认 R-E1；只有自车目标变道由局部 route 中心线偏离 + `changed_route` / signed lane-change
  组合证据确认时才 R-E2；已有 R-E2 核心按轨迹支撑前补最多 3 帧、后补最多 4 帧，避免只标横向峰值短段。
- 证据需求：XODR road/lane topology、XML actor flow、meta active scenario、RGB 车道线/侧向车流/灯控硬件。
- RGB 复核结论：2026-07-04 全量审计 75 个有效 route，其中 9 个 route 有确认灯控帧；
  其余仍按 R3 背景处理，R5 检测项多来自匝道/导流线/路口近邻弱证据，不进入候选。
- 待完善点：若未来发现城区 cut-in 子集，应单独按 run 级分流，不能恢复全局 R1。

### HighwayExit

- 候选 RS：R3。
- 已确定口径：驶出/分流是 R3 典型结构，且该 scenario 背景为高速/快速路，不开放 R1/R4。
- 分段逻辑：exit/actor-flow 窗口 + XODR split/ramp 提高 R3 置信；驶出完成后仍处于高速/快速路背景时保持 R3；
  meta 伪灯态不动态加入 R4。
  EVENT 层中出口前主线正常跟车可 R-E1，出口目标变道为 R-E2；在原有后补 4 帧基础上累计为
  前补最多 2 帧、后补最多 6 帧；
  进入分流/驶出匝道后保持 R-E3，不再回落 R-E1。
- 证据需求：XML route/trigger/other_actor_location、XODR ramp/split、meta active scenario、RGB 出口边界。
- RGB 复核结论：`rs_full_frame_review_highway_twoways_user_fix` 两条 route 共 232 帧，输出 `R3=232`。
- 待完善点：若出口后进入城市路网，需要用 RGB/XODR route 分段切 R4/R1-like 后续类别；当前样本未显示该类。

### InterurbanActorFlow

- 候选 RS：R1, R3, R5；R4 已删除。
- 已确定口径：全量 `AutoMoT/lead_video/InterurbanActorFlow` 前视与三视角总览未见稳定信号灯路口，
  大量 route 是 STOP 标线/无灯十字路口/priority gap selection；不能再把候选写成 R4/R5 不确定。
- 分段逻辑：merge/actor-flow 拓扑窗口仅在强 merge/topology 时给 R3；STOP、无灯 junction、
  is_junction 或可信局部 XODR 路口窗口给 R5；仅 `active scenario + close trigger`
  而没有控制源时仍为 R1。R-E2 已有核心按轨迹前补最多 3 帧、后补最多 4 帧，只跨 R1/R3 的 R-E1，
  不跨 R5/R-E5 路口段。`route_projection_error_high` 继续写 review。
- 证据需求：RGB STOP/路口、XML actor-flow + route、XODR merge/junction、meta stop/is_junction/active scenario。
- RGB 复核结论：2026-07-04 覆盖 90 个有效 lead_video route 的前视 3 帧总览和三视角中帧总览；
  未见稳定红绿灯路口，看到大量 STOP/无灯交叉口。旧 91-route 高速桶复核仍成立：未发现高速/快速路桶。
  2026-08-09 四问 RGB 复核进一步确认：`InterurbanActorFlow/R3/R-E1` 只有 R3 标签/短段形态，
  RGB 未见封闭高速、匝道、连续分隔主线或出入口控制，因此四问 `HIGHWAY=NO`。
- 待完善点：实测 projection error 多，边界必须优先使用 RGB + XODR/meta waypoint 吸附，不能只看 sparse route_s。

### InterurbanAdvancedActorFlow

- 候选 RS：R1, R5，R3 仅 medium/review；候选 EVENT：R-E1, R-E2, R-E5。
- 已确定口径：advanced actor flow 不自动 R3/R5；R3 只有出现明确 merge/split/ramp 时才成立，
  R5 也必须有可见或低投影误差下的无信号路口/priority 证据；全量 RGB 未见稳定 R4。
- 分段逻辑：junction/priority/no-light 给 R5，当前 `junction_pre/post=72/33m`，相对旧 `55/25m` 约放宽 30%；明确合流拓扑才临时 R3；其余 R1；meta 伪灯态不动态加入 R4。若通过 R5 路口时存在 `changed_route` + 横向偏移/换道符号证据，R5/R-E5 段内可保留 R-E2，并按轨迹前补最多 3 帧、后补最多 4 帧。
- 证据需求：XODR junction/signal/priority、XML actor-flow、meta light/junction、RGB 路口和侧向车流。
- RGB 复核结论：78 个 route 已逐 id 抽取均匀 5 帧 stitched RGB 复核，未发现高速/快速路桶。
  `Town12/Town13` 旧 R5 多由 scenario/stop hint 触发，RGB 更像普通城际道路，已回到 R1 + review。
- 待完善点：实测大量 R5/R1 且 review 高，需补更细的路口窗口和投影修正。

### InvadingTurn

- 候选 RS：R1, R2, R4, R5。
- 已确定口径：`InvadingTurn` 是混合场景。2026-07-05 盲审复核发现
  `Town12_Rep0_1150_0_route0_01_09_20_35_02` 存在稳定信号灯与斑马线路口，
  因此不能整类删除 R4；但 Town13 等 route 仍是无灯/STOP/priority 路口和对向车侵占冲突。
- 分段逻辑：无灯/STOP junction 窗口给 R5，且 R5 必须有 STOP/yield、真实 junction 或 bbox stop/yield 与 junction 几何同源；
  close-trigger / active scenario / 孤立 bbox stop-sign 只能保留 review，不能单独把直道起始段提升成 R5。对向车侵入但道路结构主导为双向窄路/对向占道时给 R2；
  稳定 meta/bbox/RGB 灯控证据给 R4；窗口外 R1。EVENT 用 U-E5 表达被动让行/对向侵入，
  U-E5 可落在 R1/R2/R4/R5，但必须同时满足 `vehicle_hazard=true` 和对象距离不大于 35m；
  普通近车距离、scenario active 或停车等待不能单独触发。普通信号灯通过段为 R-E4。
- 证据需求：RGB STOP/无灯路口或稳定信号灯、对向车轨迹、XML trigger/turn direction、
  XODR lane direction、meta active scenario/is_junction/stop/traffic_light_state。
- XODR/route 复核结论：2026-07-04 全量标注中 11883 帧均走 static XODR probe，
  11083 帧为 `xodr_topology_trusted=false, map_is_junction=false, map_junction_id=-1`，
  且 8040 帧 `route_projection_error_m > 5m`；因此 InvadingTurn 的十字路口不能主要依赖
  XODR junction，要优先用 RGB STOP/无灯路口、meta active/is_junction/stop 与 XML trigger 召回 R5。
- 去抖结论：`InvadingTurn` 的 R4/R5 必须是连续稳定片段；1-3 帧 R5/R1 互跳要迭代平滑并同步 EVENT。
  夹在 R5 之间的短 R1 gap 只有仍有 junction/STOP/yield 同源控制证据时才允许桥接为 R5，不能再只凭 trigger<45m 桥接。
- 待完善点：R2/R5 仲裁需要继续看主导决策空间：窄路会车/侵占为 R2，路口找 gap/让行优先为 R5。

### MergerIntoSlowTraffic

- 候选 RS：R3, R4。
- 已确定口径：慢车流合流主体发生在高速/快速路背景，非路口默认 R3；低速不改变 RS。
  全量逐帧 RGB 发现少量真实灯控子集，因此不再使用场景级 no-R4。
- 分段逻辑：merge/actor-flow 窗口 + topology 给 R3；若 XODR/route 投影失效但 XML actor-flow
  强近邻或 trigger 距离仍支持 merge，则走
  `r3_merger_actor_flow_or_trigger_fallback` 给 R3 并保留 review；窗口外仍保持高速 R3 背景；
  R4 必须由逐帧 RGB 可见灯控 + meta/bbox 灯控同源证据触发。
  EVENT 层中起始/中间普通主线跟车保持 R-E1，trigger-only 圆窗不单独制造 R-E3，准备汇入段保持 R-E3，真实目标变道才切 R-E2；
  已有 R-E2 核心按轨迹前后各最多补 5 帧，R-E2 后仍贴近 actor-flow/merge 的 tail 保持 R-E3，
  merge 完成且 RGB/route 显示已经回到主线正常跟车后才回 R-E1。
- RGB 复核结论：2026-07-04 全量审计 88 个有效 route，其中 9 个 route 有确认灯控帧；
  弱 R5 证据多为 merge/ramp/导流线，不作为无灯路口候选。
- 证据需求：XML flow window、`start_actor_flow/end_actor_flow`、XODR merge/lane count、meta speed/active scenario、RGB 慢车流间隙/灯控硬件。
- 待完善点：不要把低速误归停车结构或 R5；低速只是事件/动作状态。复核时优先看 RGB 与
  `actor_flow_distance_m`，不要因为 `route_projection_error_high` 自动压回 R1。
  active scenario 只能辅助解释，不能单独制造 R4/R5。

### MergerIntoSlowTrafficV2

- 候选 RS：R3。
- 已确定口径：同属高速/快速路合流语义，但全量逐帧 RGB 未发现稳定真实灯控子集；V2 保持纯 R3/no-R4，不继承 MergerIntoSlowTraffic 的少量 R4 子集。
- 分段逻辑：非路口默认 R3；合流核心窗口提高置信；meta 伪灯态不动态加入 R4。
  EVENT 层中起始/中间普通主线跟车保持 R-E1，trigger-only 圆窗不单独制造 R-E3，准备汇入段保持 R-E3，真实目标变道切 R-E2，
  已有核心前后各最多补 5 帧；R-E2 后若 RGB 仍处于分离汇入/匝道空间且 actor-flow/merge
  近邻仍成立，则继续保持 R-E3，不因 active span 结束回落 R-E1。
- RGB 复核结论：2026-07-04 全量审计 103 个有效 route，R4 比例为 0；保持 R3/no-R4。
- 证据需求：XML actor-flow、XODR merge、meta active scenario、RGB 侧向车流。
- 待完善点：V2 需要和 canonical scene 合并时保留 raw scenario，用于回查失败样例。

### NonSignalizedJunctionLeftTurn

- 候选 RS：R1, R5。
- 已确定口径：无信号灯左转是 R5；这是 strict no-R4 场景。若 bbox 或静态 XODR 同时给出
  `traffic_light` 弱提示和 `stop_sign/yield` 证据，以 STOP/yield/无灯控制源为准，不动态打开 R4。
- 分段逻辑：junction/priority/stop/yield 局部核心窗口 R5；scenario 退出、meta 非 junction
  且超出 route/trigger 窗口后恢复 R1，残留 stop_hazard 或静态 XODR 弱 junction hint 不得锁死 R5。
- 证据需求：XODR junction/priority/sign、XML route trigger、meta is_junction、RGB 路口入口。
- RGB 复核回灌：2026-07-04 5-id/town 复测中，修复前样本分布为 `R1:23, R4:571, R5:1352`；
  代表 route `Town04_route_000930` 的 RGB 是无灯/STOP 小路口，但 bbox 同时报
  `traffic_light=True, stop_sign=True`，导致旧代码误判 R4。修复后目标复测分布为 `R5:756`。
- 待完善点：Town10HD 样本曾出现 meta 缺口；缺 meta/XML 的 run 直接 `data_missing_skip`，
  不再用 XODR/XML 给临时 medium 标签。

### NonSignalizedJunctionLeftTurnEnterFlow

- 候选 RS：R1, R5。
- 已确定口径：名字里有 EnterFlow，但这是无信号灯路口进入车流，不是匝道 R3；同样 strict no-R4。
- 分段逻辑：junction/priority/no-light 局部核心窗口 R5；进入车流后的直行段恢复 R1；
  仅明确 ramp/merge 才 review R3。
- 证据需求：XODR junction/priority、XML enter-flow trigger、meta active scenario、RGB 路口车流。
- RGB 复核回灌：修复前样本分布为 `R1:24, R4:542, R5:1798`，主要是 bbox/static signal 弱提示在
  stop/yield 口抢占 R4；修复后目标复测分布为 `R1:14, R5:942`，不再输出 R4。
- 待完善点：实测 `enter_flow_not_r3` 方向正确，但 projection error 高，需要补边界帧。

### NonSignalizedJunctionRightTurn

- 候选 RS：R1, R4, R5。
- 已确定口径：大多数 RGB 显示 STOP/无灯右转路口；无灯右转路口归 R5，右转动作本身不是 R3。
  但全量逐帧 RGB 发现少量真实灯控右转子集，不能再全局禁 R4。
- 分段逻辑：no-light junction 核心窗口 R5；有效灯控/灯态同源窗口 R4；移动速度大于 5m/s、
  route 横向偏移小于 0.04m 且 meta 非 junction 的前后直行段恢复 R1。
- 证据需求：XODR junction/priority/yield/signal、XML route turn、meta junction/light、RGB 入口/横向车流/灯控硬件。
- RGB 复核结论：2026-07-04 全量审计 93 个有效 route，R5 为主（80/93），但 12 个 route 有确认灯控帧。
- 待完善点：若右转专用 slip lane 有 yield/merge 属性，需人工决定 R5 vs R3 的优先级，目前保守 review。

### OppositeVehicleRunningRedLight

- 候选 RS：R1, R4。
- 已确定口径：对向车闯红灯是 R4 下的突发事件，不是 R5；信号规则仍有效。
- 分段逻辑：受控路口/红绿灯窗口 R4；窗口外 R1；违规对象进入 EVENT。
- 证据需求：meta traffic_light_state、XODR signal/controller、XML trigger、RGB 对向车与灯态。
- EVENT 边界：U-E6 由冲突车辆、近距离对象、bbox/RGB 横穿或对向动态车辆、ego 停车/让行响应共同触发；
  允许主冲突前约 6 帧、冲突后最多约 32 帧等待上下文，冲突解除后回 R-E4。
  多段横向车候选时先选导致 ego 停车/等待的 span，再比较 bbox 冲突帧数、span 长度和最近对象距离；
  `Town05_Scenario8_123` 这类自车已进路口后停车的 case 应保留早期等待段，而不是后续正常通行时更长的横向车段。
- 2026-07-10 复验：287 route / 28165 帧中，U-E6 从上一版 3515 帧扩为 5744 帧；
  R4 route 完全无 U-E6 从 126 条降到 9 条，1-3 帧短 U-E6 降为 0，超过 80 帧的长段仅剩 2 条且 RGB 为持续横向冲突车流。
- 待完善点：若灯态缺失但场景名强提示，最多 R4 medium + review。

### OppositeVehicleTakingPriority

- 候选 RS：R1, R4, R5。
- 已确定口径：全量 RGB 以 STOP/让行/无灯 priority 路口为主；对向车抢优先权主要归 R5。
  少量 route 可见真实灯控子集，因此 R4 作为候选恢复，但必须逐帧强证据触发。
- 分段逻辑：priority/no-light junction 窗口 R5；有效灯控/灯态同源窗口 R4；窗口外 R1。
  2026-07-10 按 RGB 边界把进入侧放宽到 `junction_pre_m=75`（约 +50%），避免已经清楚接近 priority 路口的帧仍保持 R1。
- 证据需求：XODR priority/yield/junction/signal、XML trigger、meta active scenario/light、RGB 对向车流/灯控硬件。
- RGB 复核结论：2026-07-04 全量审计 97 个有效 route，79 个 route 有无灯/priority 窗口，7 个 route 有确认灯控帧。
- 待完善点：R4 比例小，后续最好生成 route 级灯控白名单或边界帧，避免把无灯 priority 窗口误升 R4。

### ParkedObstacle

- 候选 RS：R1, R4, R5。
- 已确定口径：ParkedObstacle 不是 Parking*；停放障碍是 EVENT，默认 R1，不自动生成停车 RS。
- 分段逻辑：障碍窗口只标事件；灯控路口 R4；STOP/无灯路口 R5；其余 R1。
  2026-07-10 按 RGB 边界把进入侧放宽到 `junction_pre_m=72`（约 +20%），召回停放障碍前后真实路口控制区。
- 证据需求：XML obstacle trigger、meta distance/active scenario、XODR/traffic light。
- RGB 复核结论：2026-08-09 四问复核按组合级保持 `ParkedObstacle × U-E2` 的 `OBSTACLE=YES`，
  即便个别当前帧因为时间分散不一定清楚看到占道物。另发现 Town12 有 highway-like fast-road 子组；
  当前 `scenario × RS × EVENT` 聚合表不全局翻转 `HIGHWAY`，若后续拆 `Town/route topology subgroup`，
  该子组应单独给 `HIGHWAY=YES`。
- 待完善点：只有确认障碍来自停车带/路边停车空间且压缩有效通道时，才考虑 R2 或 EVENT review。

### ParkedObstacleTwoWays

- 候选 RS：R2, R4, R5。
- 已确定口径：停放障碍/两侧停车压缩可行驶 lane 后按 R2；“parked” 不是 ROAD_STRUCTURE 的充分条件。
- 分段逻辑：two-way obstruction 窗口 + opposite lane 参与给 R2；灯控路口 R4；
  障碍未进入决策区或自车已经绕过障碍后仍保持 R2；只有真实灯控/STOP/无灯控制源切 R4/R5。
- 证据需求：XML distance/offset、XODR lane direction/count、meta active scenario、RGB 借道。
- RGB 复核结论：TwoWays 家族边界复核中 `ParkedObstacleTwoWays` 2 条 route 共 160 帧，
  旧版会让 core 前后 layout-prior 过强；新版只在障碍压占路径/借对向的核心段给 R2。
- 2026-07-10 全量复验 96 route / 14030 帧，未发现残留短
  `R-E2 -> R-E4/R-E5 -> R-E2` 插缝；代码仍保留 8 帧以内短路口插缝合并，防止后续阈值变化重新打断恢复变道。
- 待完善点：路边停放但不需要借对向时，保持 R2/R-E1 + EVENT；若 route 证据证明并非对向单车道，再单独 review。

### ParkingCrossingPedestrian

- 候选 RS：R1, R4, R5。
- 已确定口径：停车区/路边停车空间不单独成 RS；行人横穿是 U-E4；灯控路口优先 R4；
  STOP/无灯 T/十字路口为 R5。
- 分段逻辑：parking/curbside/shoulder 窗口只辅助行人/遮挡事件；灯控路口 R4；
  STOP/yield/无灯路口连续 traversal 给 R5；其余 R1。
- 证据需求：XML direction/crossing_angle、XODR parking/shoulder/curbside、meta active scenario、RGB 行人和停车侧。
- RGB 回灌：2026-07-05 子集审计开放 R5 后，ParkingCrossingPedestrian 总 mismatch 从旧 full audit
  的 955 帧降到 220 帧；top 剩余如 `Town13_Rep0_1085_0...` 是雾天停车/行人交互与路面箭头/线被 blind detector
  扩成 R5，人眼看并非稳定完整路口窗口，当前保守 R1/U-E4 更合理。
- 待完善点：实测 projection error 多，R4/R5 边界需要 RGB boundary frames 核验；不能只靠 crosswalk/turn-marking 单证据扩张。

### ParkingCutIn

- 候选 RS：R1, R4, R5。
- 已确定口径：全量逐帧 RGB 未看到稳定独立停车结构；灯控路口优先 R4；STOP/无灯路口为 R5；切入行为本身进入 EVENT。
- 分段逻辑：STOP/yield/meta junction/XODR 近路口给 R5；灯控路口给 R4；窗口外 R1；停车车启动/切入只改变 EVENT 为 U-E3。
- 证据需求：XML direction/front/behind、XODR parking lane/shoulder、meta active scenario、RGB 停车车启动；STOP/无灯路口必须有 STOP/yield/meta junction 或可信近路口证据。
- EVENT 边界：U-E3 起点必须是 `dist_to_cutin_vehicle<=26m` 且有 `brake_cutin`、`vehicle_hazard`、
  目标变道或横向轨迹之一；进入 R4/R5 后只在响应持续或自车尚未回正时短续，不再用 distance-only 等到前车消失。
- RGB 复核结论：99 个 route 已逐 id 抽取均匀 5 帧 stitched RGB 复核，未发现高速/快速路桶；
  2026-07-06 全量逐帧 RGB audit 也未输出稳定独立停车结构，因此保留 R1/R4/R5。
- RGB 回灌：2026-07-05 三场景回归中，开放 R5 后 `blind_R5_label_R1` 从 802 帧降到 86 帧；2026-07-10
  全量复验 97 route / 13892 帧，U-E3=330 帧 / 80 route，最长连续 span 7 帧；
  `Town12_Rep0_1757_0` 从旧 f48-74 收为 f48-51，`Town12_Rep0_815_0` 从旧 f211-236 收为 f211-215。
- 待完善点：继续用 RGB boundary frames 核查少数 zero-U3 route 是否确实没有停车车侵入当前路径。

### ParkingExit

- 候选 RS：R1, R4。
- 已确定口径：从停车位/路边驶出并汇入主路的过程仍是 R1，由 R-E2 表达；汇入完成后回 R-E1。
- 分段逻辑：parking exit 窗口 + parking-to-driving transition 给 R1/R-E2；灯控路口 R4；完成后 R1/R-E1。2026-07-10 RGB 复核显示部分初始驶出段收尾拖长，初始 R-E2 在当前基础上提前约 5 帧释放。
- 证据需求：XML parking trigger、XODR parking/curbside、meta pose/speed/active scenario、RGB 车位出口。
- 待完善点：与 R3 merge 的区别是停车空间 vs 匝道/主辅路拓扑，二者接近时必须 review。

### PedestrianCrossing

- 候选 RS：R1, R4, R5。
- 已确定口径：行人横穿是 EVENT；RS 由是否在路口、是否有有效灯控决定。
- 分段逻辑：有效灯控 junction/stopline 给 R4；无灯/priority crossing junction 给 R5；普通路段 crossing 保持 R1。2026-07-10 将进入侧从 40m 收为 36m、退出侧放到 60m，并允许退出后最多 6 帧 RGB/灯控/行人证据尾段保留 R4/R5；同一 R4/R5 路口内 1-8 帧 R1/R-E1 短缝同步回填 RS+EVENT。
- 证据需求：XODR signal/junction/crosswalk、meta traffic_light_state/is_junction、XML pedestrian trigger、RGB 行人/斑马线。
- 待完善点：实测 R4/R5 review 很高，crosswalk 与 junction/source 不同源时要降置信。

### PriorityAtJunction

- 候选 RS：R1, R4, R5。
- 已确定口径：当前 runtime 候选保持 R1/R4/R5，重点修复“已经在本地灯控/stopline 区内却被 route lock
  错压成 R1”的问题；不能按 Town12/13 自动高速。
- 分段逻辑：traffic light / stopline / controller 与 junction 同源时给 R4；无灯/让行控制源给 R5；窗口外 R1。
- 证据需求：XODR priority/yield/sign/junction/signal、XML trigger、meta active scenario/light、RGB 横向车流与灯控硬件。
- 2026-07-10 复验：99 route / 9702 帧，route lock 改动从旧 68 帧降到 3 帧，R4 从
  1333 帧增到 1362 帧；`Town12_Rep0_4022_0` 的 f18/f20/f25/f35/f45 保持 R4/R-E4，f60 回 R1/R-E1。
- 2026-07-10 Town13 晚触发回灌：`Town13_Rep0_1105_0` 与 `Town13_Rep0_1099_0` RGB 在 f43-f44 已明显接近灯控路口，
  旧规则到 f47 才 R4；route lock 现在以第一段稳定 R4 为锚点，在锚点 >= f30 时最多前补 4 帧，
  两条均变为 `R1 f0-42 -> R4 f43-69 -> R1 tail`，且 Town13 全量未产生短 R4 小岛。
- 待完善点：低能见度时要显式区分 priority sign 与 traffic light controller 的冲突。

### RedLightWithoutLeadVehicle

- 候选 RS：R1, R4。
- 已确定口径：红灯停车是 R4；即使 `is_junction=false`，只要 stopline/traffic light approach 同源，也应 R4。
- 分段逻辑：traffic light / stopline approach 窗口 R4；离开灯控区 R1。trigger 距离超过约 52m、
  自车已恢复行驶且没有本地 junction/window/control 时，尾段强制释放为 R1/R-E1。
- 证据需求：meta traffic_light_state、XODR signal/stopline/controller、XML route trigger、RGB 灯态。
- 2026-07-10 复验：355 route / 48310 帧，R1/R4=4021/44289；`Town01_Rep0_Town01_Scenario7_16`
  中 f140-f160 仍 R4/R-E4，f165/f170/f172 释放为 R1/R-E1。
- 待完善点：需要把 stopline approach 纳入 R4，不只依赖 junction polygon。

### SignalizedJunctionLeftTurn

- 候选 RS：R1, R4。
- 已确定口径：信号灯左转全程属于 R4，包括等待 gap 和转弯中观察冲突对象。
- 分段逻辑：有效 signal/controller + junction/stopline 窗口 R4；窗口外 R1。
- 证据需求：meta traffic_light_state、XODR signal/controller/junction、XML turn route、RGB 灯态。
- 待完善点：左转等待对向车不是 R5；事件层表达让行/冲突。

### SignalizedJunctionLeftTurnEnterFlow

- 候选 RS：R1, R4。
- 已确定口径：EnterFlow 发生在信号灯路口内，仍归 R4，不归 R3。
- 分段逻辑：signalized junction 窗口 R4；窗口外 R1；enter-flow 只影响事件。Town01/Town02 前
  30 帧若只有远灯/弱 trigger 且无本地 junction core，则回 R1/R-E1。
- 证据需求：traffic_light_state、XODR controller/junction、XML enter-flow trigger、RGB 对向/横向车流。
- 2026-07-10 复验：173 route / 12884 帧，起始弱 R4 过滤 210 帧；`Town01_route_002329`
  f0-f12 回 R1/R-E1，f16 起真实接近路口保持 R4/R-E4。
- 待完善点：名字中的 EnterFlow 不能作为 R3 先验。

### SignalizedJunctionRightTurn

- 候选 RS：R1, R4。
- 已确定口径：信号灯右转是 R4；右转专用道/短 slip lane 不应因 sparse route 被误判 R3。
- 分段逻辑：signal/controller/stopline 窗口 R4；离开后 R1。
- 证据需求：meta traffic_light_state、XODR signal/controller、XML turn route、RGB 灯态和右转车道。
- 待完善点：实测右转场景不应依赖稀疏 XML route 线；要用灯态和 stopline source 做主证据。

### StaticCutIn

- 候选 RS：R1, R3, R4, R5。
- 已确定口径：StaticCutIn 不是单一结构；每个 run 要按局部拓扑拆成合流侧 R3、灯控/无灯路口 R4/R5 或普通 R1。
- 分段逻辑：先按 route 级 RGB 分高速/快速路桶；高速桶候选收敛为 R3/R4。
  非高速桶保留 R1/R4/R5；merge/ramp/split evidence 给 R3；
  灯控路口 R4；STOP/无灯路口连续 traversal 给 R5；否则 R1。
- 证据需求：XODR parking/shoulder/merge/split、XML trigger、meta active scenario、RGB 静态车切入来源。
- RGB 复核结论：100 个 route 已逐 id 抽取均匀 5 帧 stitched RGB 复核；44 个 route 是多车道护栏/高架/快速路形态，
  已写入 route 级高速桶并收敛为 R3/R4。其余 `Town13_Rep0_1704/1705/1714/1715/1717/1718` 等城市或普通宽路仍保留 R1，
  不能因 Town13 或 StaticCutIn 名称统一判 R3。
- RGB 回灌：2026-07-05 top mismatch 逐帧复核发现 `Town13_Rep0_1212_0...` 等路线存在连续 stopline/
  横向路口形态，候选缺 R5 会压成 R1；开放 R5 后 StaticCutIn 子集
  `blind_R5_label_R1` 从 321 降到 27 帧，未出现 R5 可疑暴涨。剩余
  `event_regular_during_rgb_object_or_motion_activity` 多为邻道/前方车辆普通运动，不扩大 U-E3。
- 2026-07-06 全量逐帧 RGB audit 输出为 R1/R3/R4/R5，没有稳定独立停车结构。
- 待完善点：R3 必须有高速/合流证据；若 R3 证据弱于普通路段，必须回 R1。

### T_Junction

- 候选 RS：R1, R4, R5。
- 已确定口径：T 形路口不是天然等于灯控；全量逐帧 RGB 同时看到灯控 T 路口和无灯/STOP/yield T 路口。
- 分段逻辑：junction/stopline 窗口内有有效灯态或强 signal/controller 证据给 R4；RGB/stop/yield/priority 显示无灯控制给 R5；窗口外 R1。退出侧允许比通用 signalized 稍长，避免仍在 T 路口内时过早释放。
- 证据需求：XODR junction/signal/priority、meta traffic_light_state/stop/is_junction、XML route turn、RGB 路口控制源。
- RGB 复核结论：2026-07-04 全量审计 247 个有效 route，216 个 route 有确认灯控帧，169 个 route 有无灯/STOP/yield 型 junction 证据；
  该场景按 R4/R5 共有处理，但单帧不能只因“没看到灯”就判 R5。
- 2026-07-10 复验：246 route / 31498 帧，R1/R4/R5=2048/29374/76；
  route lock 改动从旧 5723 帧降到 1928 帧，`Town01_Scenario7_68` f44-f63 保持 R4/R-E4。
- 待完善点：需要 route 级边界帧标注区分灯控 T 和无灯 T，避免同一 junction approach 上的临时遮挡导致 R4/R5 抖动。

### VehicleOpensDoorTwoWays

- 候选 RS：R2, R4, R5。
- 已确定口径：开门风险和两侧停车占用侧向 lane，使有效可行驶通道成为对向单车道时，主 RS 为 R2；不再生成独立停车主标签。
- 分段逻辑：two-way/open-door context 给 R2；灯控路口 R4；STOP/无灯/路权路口 R5。
- 证据需求：XML trigger/direction、XODR parking/opposite lane、meta active scenario、RGB 开门车辆和对向车。
- 待完善点：继续用 RGB 复核 U-E2/R-E2 边界，避免开门核心过早或过晚。

### VehicleTurningRoute

- 候选 RS：R1, R4, R5。
- 已确定口径：转弯路线本身不是 RS；是否 R4/R5 取决于路口控制源。
- 分段逻辑：signalized junction 给 R4；non-signalized/priority junction 给 R5；普通弯道/车道跟随 R1。
- 2026-07-08 RGB 回灌：20-route smoke 中旧规则 `R1/R4/R5=522/2010/228`，
  `U-E4=592`。Town02/Town03/Town05/Town10HD 的 U-E4 起点普遍在
  `dist_to_biker ~= 20-22m`，RGB 仍是远端路口/远处骑行者，实际横穿冲突更接近
  `dist_to_biker <= 16m`。因此 VehicleTurningRoute 的 junction window 收为
  `factor=0.65`，并启用远灯态本地路口证据约束；U-E4 自行车阈值收为 16m，
  route 级 single-span support padding 从通用 6m 降为 2m。复跑同样样本后
  `R1/R4/R5=873/1661/226`，`U-E4=435`；Town02 的 U-E4 从 f18 推迟到 f70，
  Town03 从 f35 推迟到 f68，仍覆盖真实横穿冲突。
- RGB 回灌：投影误差高时，不能只靠 `junction_window` 把无灯车转弯场景整段拉成 R5；
  必须有 `stop_hazard`、`is_junction` 或非静态可信 XODR 近路口证据，否则给 R1 并保留
  `vehicle_turning_r5_demoted_projection_error_rgb_required` 复核线索。
- 证据需求：XML turn route、XODR junction/signal/priority、meta light/junction、RGB 路口。
- 待完善点：多 trigger 场景要按每个路口窗口分段，不能整段一个 RS。

### VehicleTurningRoutePedestrian

- 候选 RS：场景级 R1, R4, R5；非 TwoWays R2 暂停动态加入。
- 已确定口径：行人是 EVENT；RS 仍由道路布局与 turn route 所在控制源决定。非白名单 route
  即使 XODR sparse scan 有 opposite/parking hint，也先保持原候选，避免普通 R1/R-E1 被误升 R2。
- 分段逻辑：有效灯控路口 R4；无灯/priority 路口 R5；其它普通同向路段 R1。只有全量逐帧 RGB
  审完并写入 route 白名单后，才允许非 TwoWays 片段动态 R2。
- RGB 回灌：`Town12` 旧结果把 0-144 基本整段标 R5，但 RGB 中 9-82 是普通住宅道路跟车；
  新规则将高投影误差且缺少真实路口/stop/可信 XODR 的帧降回 R1，只在 83-144 的 stop/路口/转弯区域保留 R5。
- 2026-07-08 追加 RGB 抽样压缩：按 Town12/Town13 均匀采样 12 条 route 看边界帧，发现远灯态和 XML trigger-only STOP hint 仍会让住宅/乡村普通道路过早进入 R4/R5。现对本场景额外使用
  `junction_tighten_factor=0.60`，有灯触发核心为 8m，无灯/STOP trigger-only 核心为 5m；
  若缺少 `dist_to_junction` 近区、`is_junction`、bbox junction 或可信 XODR 本地路口证据，则远处 meta/bbox 灯态不能单独升 R4。
  12-route smoke 从 `R1/R4/R5=444/1083/882` 调整为 `922/874/613`，`Town12_1645_0`、`Town13_1615_1`、`Town13_1741_1` 的起始普通道路段回到 R1，主路口/行人 U-E4 段仍保留 R4/R5。
- 2026-07-08 追加 gap 修复：`Town12_1754_0/1` 的 RGB 显示短 R1 gap 仍处在 STOP 无灯路口过程中，
  原因是 `close_trigger_for_junction` 已退出、`near_junction` 尚未接上。新增
  `vehicle_turning_junction_gap_recovery`：只在 R1 gap 不超过 16 帧、前后同为稳定 R4/R5 且两侧各至少
  8 帧、gap 内仍有 turning local junction / strong control / stop-yield 证据时回填。该规则修复
  1754 的 R5-R1-R5 抖动，但不会合并 `Town12_1645_0` 这类前后仅数帧的弱 R5 小片段。
- 2026-07-09 追加 R4/R5 单控制源锁：用户确认同一 route 不允许 R4/R5 互跳，不能因为灯态遮挡或
  STOP hint 把同一个十字路口从 R4 突然改成 R5。全量 91 route / 20399 帧修复前有 50 条 route
  同时出现 R4/R5、47 次直接相邻 `R4 <-> R5` 跳变；修复后 mixed route=0、直接跳变=0。
  EVENT 逻辑不重写，只把 regular event 随 RS 控制源同步为 R-E4/R-E5，U-E4 保持 3454 帧不变。
  同日扩展到所有同时允许 R4/R5 的场景：历史全量结果 24 场景复算中，旧结果 mixed route=759、
  direct jump route=698；按新锁复算后均为 0。实际重跑 5 个高风险大场景 1405 route /
  210072 frame，也没有 R4/R5 共存或相邻跳变。
- 2026-07-10 追加最终短缝兜底：同类稳定 R4/R5 路口段中间若只夹 6 帧以内 R1/R2 regular
  gap，最终输出层同步回填 RS 与 EVENT，防止只修 event 不修 rs。全量 91 route / 20399 帧复验：
  `vehicle_turning_junction_gap_recovery` 修正 22 帧，temporal smoothing 修正 6 帧，
  `final_junction_regular_gap_merge` 未再需要改动；最终 `R-E4/R-E5 -> R-E1/R-E2 <=6帧 -> 同类 R-E4/R-E5`
  和对应 `R4/R5 -> R1/R2 <=6帧 -> 同类 R4/R5` 残留均为 0。
- 2026-07-08 追加非 TwoWays R2 修复后又回退：曾抽查 `Town12_393_0/1`、`Town12_398_0/1`、
  `Town13_84_0` 的 RGB，发现疑似黄中心线乡路或两侧停车压缩通道；但用户指出普通
  R1/R-E1 被误升风险后，runtime 先清空 `LAYOUT_R2_ROUTE_IDS`。后续必须按所有 id 逐帧 RGB
  审完，确认确实是对向单车道，再逐 route 写入白名单。
- 证据需求：XML turn/pedestrian trigger、XODR signal/junction/crosswalk、meta light/junction、RGB 行人边界。
- 待完善点：行人 crossing 与路口控制源不同步时，保留 primary RS 但 review。

### noScenarios

- 候选 RS：R1, R3, R4, R5。
- 已确定口径：没有 scenario 事件先验时保守 R1；R4 只接受本地受控灯控证据，不能由远处/单个 bbox 灯框或弱 XODR signal 把直道整段抬成路口；STOP/yield/无灯控制证据可召回 R5，并会压制同帧弱灯控提示。
- 分段逻辑：普通路段 R1；正常灯控路口 R4；STOP/yield/无灯路口 R5；可信 XODR ramp/merge/split 或人工 RGB highway bucket 才允许 R3；不从弱 XODR hint、Town 名、普通宽路或 route 名自动产生 R2/R3。
- 证据需求：R4 需要有效灯态/light hazard 与本地距离证据同源：overhead light、affects_ego 且 forward/distance 足够近、近 physical traffic light，或近 junction 内的连续灯控；R5 需要 STOP/yield/meta junction 或可信 XODR 近路口证据。bbox semantic metrics 必须写入 evidence，包含 traffic light count/min distance/forward x/physical distance/affects_ego/same_lane/overhead 以及 STOP/YIELD 最近距离；弱灯态或弱 topology 只进 evidence/review。
- RGB 回灌：2026-07-05 三场景回归中，开放 R5 后 `blind_R5_label_R1` 从 18879 帧降到 3453 帧；再加入 `r4_noscenario_stable_tl_bbox_approach` 后，`blind_R4_label_R1` 从 759 降到 315，R4 过标仅从 12 增到 21。`Town07_Rep0_route_000276...` 人工 RGB 显示 32-176 帧为稳定灯控路口，旧 strong context 过严仅标 183-188，修复后召回为连续 R4。
- 2026-07-10 noScenarios 专项 RGB/XODR 复核：`Town07_Rep0_Town07_100...` 与 `Town07_Rep0_Town07_99...` 大部分 RGB 是农场/乡村直道，旧规则因单个远处 traffic_light/bbox 弱提示几乎整段 R4，修复后分别为 `R1 0-189 -> R4 190-212 -> R1 213-223`、`R1 0-187 -> R4 188-207`；`Town07_Rep0_Town07_51...` STOP/无灯控制全段改为 R5；`Town15_Rep0_route_000616...` 与 `Town06_Rep0_route_002205...` 的真实灯控路口仍保持 R4。
- 2026-07-10 RE2/R3 专项验证：noScenarios XML 通过 route 几何精确匹配可关联到既有具体 scenario 的只有 96 条，且均为 `VehicleTurningRoute`，不能靠“原场景名”批量恢复高速/合流 R3；抽查 Town15/Town06 RGB 多为普通城市/乡村宽路或路口，不应按 Town 自动升 R3。全量直接标注 1436 route，R-E2 从候选缺失修复为 13850 帧，R1/R4/R5=122686/7617/26549，R3 仍为 0，表示当前未发现可信 ramp/merge/split 证据；后续若逐帧 RGB 确认具体 noScenarios route 是匝道/合流，应写入人工 highway bucket 或补充几何匹配表。
- 旧版全量验证：`annotate-rs --scenario noScenarios --max-frames-per-route 0` 得到 1373 route / 147935 frame，`R1/R4/R5=114354/7364/26217`；相对旧分布 `111104/11233/25598`，主要把过宽 R4 收回 R1，同时保留 STOP/yield R5。
- 待完善点：blind audit 在雾天/弯道/横向车流上仍会产生 R5 误报；collector 不应仅因 blind R5 就放大 R5，必须继续要求控制源同源证据。

## 4.1 2026-07-05 全量 RGB 审计回灌

最新全量审计输出：

- `/tmp/automot_rgb_blind_full_after_obstacle_r3_comparefix`：9715 route，耗时 3505.8s，包含
  noScenarios R4 approach、ControlLoss/ParkingCutIn/noScenarios R5、CrossJunction defect 解释、
  R3 合流解释和障碍族 R5 候选。
- 上一轮对照 `/tmp/automot_rgb_blind_full_after_r4approach`：9715 route，耗时 3546.6s。
- `CrossJunctionDefectTrafficLight`：RGB 能看到红绿灯且 scenario 语义是信号失效/规则失效，RS 应保持 R4，EVENT 用 R-E4+U-E7 表达。
  旧 `blind_R4_label_R5` 解释作废；若审计里仍出现旧 R5，应按规则污染回查。
- R3/高速/合流类：`EnterActorFlow*`、`MergerIntoSlowTraffic*`、`HighwayExit` 中导流线、让行牌和宽路面容易被 blind detector 误判 R5。
  若当前标签有 `rule_kind=highway_merge/highway_exit`、`r3_*` 规则或 highway RGB bucket，应解释 `blind_R5_label_R3`；
  子集回归后 Enter/Merger/HighwayExit 假错配归零，HighwayCutIn 只剩 37 帧灯控边界争议。
- 障碍/侧向风险场景的 route 前后经常包含真实 STOP/无灯 T/十字路口。已给
  Accident / ConstructionObstacle / ParkedObstacle / HazardAtSideLane 及其 TwoWays 版本、
  VehicleOpensDoorTwoWays 开放 R5/R-E5，但仍要求 STOP/yield/meta junction/XODR 近路口同源证据。
  全局 junction effective window 当前为 pre `0.36 * junction_pre` 且最小 16m、
  post `0.28 * junction_post` 且最小 5m；仅靠宽 junction window 或弱静态 signal 不能提前吞掉
  U-E2/R-E2 恢复链。
  XML `<weather>` 显示夜间、低太阳高度或大雾时，collector 会对所有场景再乘
  `low_visibility_factor=0.65~0.95`，同步收缩 junction window、meta near、strong context、
  static signal near 和 close-trigger；单纯下雨不再压缩，雨只在叠加雾/夜/低太阳时轻微增强收缩。
  低能见度下必须更接近路口控制源才允许 R4/R5 覆盖。
  子集回归显示：
  `Accident 2868->1771`、`AccidentTwoWays 7725->4585`、
  `ConstructionObstacle 2191->1325`、`ConstructionObstacleTwoWays 7309->5058`、
  `HazardAtSideLane 3283->2180`、`HazardAtSideLaneTwoWays 1351->639`、
  `ParkedObstacle 2533->1437`、`ParkedObstacleTwoWays 2475->841`、
  `VehicleOpensDoorTwoWays 1459->843`。
  新增 `label_R5_without_rgb_junction_signal` 仅个位数，说明 R5 没有明显过宽。
- 当前最终 top mismatch 已不再是 R3/CrossJunction 假错配，主要集中到：
  `ConstructionObstacleTwoWays`、`AccidentTwoWays`、`DynamicObjectCrossing`、`noScenarios`、
  `VehicleTurningRoute`、`ControlLoss` 等。绝对帧数最高的前几项大多由
  `event_regular_during_rgb_object_or_motion_activity`、残余 `blind_R5_label_R1/R2`
  或 blind 雾天/弯道误报组成，下一轮应优先抽 sheet 复核 U-E2/U-E3/U-E4 span 边界，
  而不是继续盲目放宽 R4/R5。

## 3. 同步更新要求

每次更新 `collection_output/rs_research/<Scenario>/` 后，同步检查本文对应场景：

- candidate RS 是否变化。
- XML / XODR / meta / RGB 哪一类证据新增或失效。
- 是否产生新的 high / medium / low 阈值。
- 是否发现 projection、boundary、arbitration 或 RGB mismatch 失败模式。
- 是否有 smoke test 能证明规则能跑成功，以及错帧能归因。

正式写代码前，本文每个场景至少应补齐：

- 1 个可复现代表 run。
- 1 组边界帧号。
- 1 组 high/medium/low 证据例子。
- 1 条失败模式或说明“未发现”。
- 对应 `thresholds.json` 的数值来源和支持样本。
