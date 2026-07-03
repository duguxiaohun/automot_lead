# ROAD_STRUCTURE 逐场景规则设计

本文是 `collection_output/rs_research/<Scenario>/rules/scenario_rule_design.md` 的可追踪整理版。
`collection_output/` 仍只作为本机调研证据目录，不 push；凡是本地调研结论、阈值或失败模式发生变化，
必须同步更新本文。

本文只讨论 `ROAD_STRUCTURE` 的帧级划分。ROAD / EVENT 总口径仍以
`ROAD_EVENT_CLASSIFICATION_PLAN.md` 为准。

## 0. 当前调研结论是否可直接当最终规则

不能。当前自动调研包已经能证明 XML 命名、route 匹配、meta 读取和部分 XODR 探针链路可跑通，
但还没有达到“最终规则”标准：

- 43 个 scenario 都已有本地规则草案，但 `map_rgb_alignment_status` 仍是 `not_checked`。
- `thresholds.json` 仍是 `temporary_default_rule_config`，缺少逐阈值的 supporting runs、reviewed artifacts 和来源说明。
- 自动生成的 map 图主要是 XML route / ego trace / trigger，对 lane、junction、signal、parking、merge/split 的局部 XODR 证据展示不足。
- RGB contact sheet 只适合作人工入口，尚未系统生成边界帧、错帧帧号和规则冲突样例。
- 默认 Python 环境下 CARLA API 不一定可用；需要 per-frame XODR 时应使用
  `/home/codon/anaconda3/envs/carla/bin/python`，否则 R2/R3/R6 只能给 medium/low + review。

因此本文里的规则分为两层：

- `已确定口径`：可以进入代码的方向性规则，例如 defect traffic light 场景优先 R5、正常信号灯路口优先 R4。
- `待验证阈值`：数值窗口和 high confidence 条件，必须继续用 `collection_output` 的地图/RGB/meta/XODR 证据校正。

## 1. 全局判定门控

### 1.1 数据源职责

- XML：只负责 route 粗投影、trigger / scenario parameter 窗口、数据追溯；不能单独作为帧级 RS 真值。
- XODR：负责局部拓扑确认，包括 lane direction、opposite lane、junction、signal/controller、merge/split、parking/shoulder/curbside。
- Meta：负责帧级 ego pose、speed、traffic light state、junction flag、active scenario、finite `dist_to_*` 信号。
- RGB：负责人工确认地图和语义是否一致，尤其是边界帧、夜间/遮挡、停车侧、对向占道、路口灯态。

### 1.2 置信度

- `high`：scenario prior + XML window + XODR topology + meta/RGB 中至少三源一致，建议分数 `>= 0.85`。
- `medium`：两源一致，或强 meta 信号成立但 XODR/RGB 不完整，建议分数 `0.65-0.84`。
- `low`：只有 scenario prior、只有 XML trigger、或只有弱 XODR hint，建议分数 `< 0.65`，必须 review。

运行时必须记录 `diagnostic_attribution`，至少写出主标签来自 XML / XODR / meta / RGB / arbitration / threshold 中哪几类证据。

### 1.3 几何质量阈值

用于调阈值的 run 需要先满足：

- route projection median error `<= 3m`
- route projection p90 error `<= 5m`
- trigger 到 ego trace 最近距离 `<= 20m`
- 单帧 XML route projection error `> 5m` 时，不允许用 route_s 作为边界 hard truth，必须降置信并 review

这些阈值是当前工作阈值，不是最终统计阈值；后续要用 boundary frames 重新估计。

### 1.4 时间平滑与冲突

- R2/R3/R4/R5/R6 结构片段最短持续 `4` 帧，即约 `1s`；R1 最短持续 `2` 帧。
  更短片段默认视为时序噪音，由 route 级 smoothing 并回邻近稳定 RS，并写入 evidence。
- R4 即使有红绿灯或 stopline 证据，也必须形成连续稳定片段；单帧 R4 仍按噪音处理。
- 主候选与次候选分差 `< 0.15` 时，保留 primary 但必须 review。
- 同一帧若 R4/R5 与 R2/R3/R6 冲突，优先判断是否处于真实路口控制区；路口控制证据同源时 R4/R5 做 primary，结构风险做 secondary。

### 1.5 本轮自动调研暴露的不完善点

`collection_output/rs_research/*/rules/scenario_rule_design.md` 已经统一生成 43 个场景的规则草案、抽样 run、
XODR 摘要和 `thresholds.json`，但这些产物仍偏“可运行模板”，不是最终人工规则。需要补强的点如下：

- `map_rgb_alignment_status` 全部仍是 `not_checked`，所以地图 route / trigger / ego trace 与 RGB 三视角是否贴合还没有闭环验收。
- `thresholds.json` 虽然已列出 `supporting_runs`，但 `reviewed_artifacts` 仍为空，`source` 仍是 `temporary_default_rule_config`；这些数值只能作为初始搜索窗口。
- 每个场景的自动 `scenario_rule_design.md` 都提示“当前自动审计未发现 town 级输入缺口”，这只说明 XML/XODR/meta 可读，不等价于边界帧正确。
- 当前 XODR 探针以 route/trigger 近邻和 town 级 signal/junction 摘要为主，仍缺少 per-frame lane successor、opposite-lane、parking/shoulder、merge/split 的局部拓扑证据。
- RGB contact sheet 是人工入口，不是判定证据；后续应为每个结构切换点生成 `boundary_before/current/after` 帧，并记录错帧归因。
- `noScenarios` 已显示部分 run 存在 Red 灯态或 junction meta，但候选仍保守为 R1/R4；不要从弱 XODR hint 自动挖 R3/R5/R6，除非单独建立 topology-only 人工样本集。
- `T_Junction` 自动规则和候选映射都把该场景归为 `signalized_junction`，候选为 R1/R4；若人工 RGB 发现无灯 T 路口，再另行加 R5，不应预先放入常规候选。
- 普通 Python 下静态 XODR planView 近邻可能出现几十米投影误差；只有 `map_projection_error_m <= 20m`
  的静态 XODR 拓扑才允许作为 R2/R3/R6 high 证据，否则必须写 `xodr_topology_untrusted` 并降级 review。
- `light_hazard=True` 不能单独把非路口帧升为 R4；必须同时有 `near_junction`、有效灯态或可信静态 signal 近邻。
  本轮 smoke test 中该门控把 `AccidentTwoWays` 小样本的 R2/R4 抖动从 38 次切换降到 2 次核心切换。
- 环岛 / roundabout 归 R1，不归 R4/R5。XODR 即使把环岛编码成 junction road，也必须进一步检查
  roundabout 几何/连接特征；`map_is_roundabout=true` 时，R4/R5 的 junction/window 分支都要失效。
- 所有 `*_TwoWays` / `InvadingTurn` / `VehicleOpensDoorTwoWays` 的 R2 high 必须补“同向 lane 不足或对向交互主导”的证据，不能只靠场景名和 trigger close。
- 所有 `highway_merge` / `interurban` / `static_cutin` 的 R3 high 必须补 merge/split/ramp/lane-count-change 证据；切入、EnterFlow、低速车流等事件名不能单独触发 R3。
- 所有 `parking` / `parking_exit` / `vehicle_opens_door_twoways` 的 R6 high 必须补 parking lane、shoulder、curbside 或 RGB 路边停车空间证据；“parked obstacle” 不等于 R6。

### 1.6 当前 `thresholds.json` 初始规则族

下表只沉淀本轮自动调研输出里的初始窗口，目的是让后续实现和人工复核有统一起点。
所有值在 `reviewed_artifacts` 补齐前都必须视为 `temporary_default`，不能作为 hard truth。

| 规则族 | 适用场景 | 当前初始窗口 / 关键开关 | 必须补的证据 |
|---|---|---|---|
| `same_direction_obstacle` | Accident, ConstructionObstacle, ParkedObstacle | `junction_pre_m=60`, `junction_post_m=25`; veto R2/R6 | 障碍只进 EVENT；R4 需灯控/stopline，同向绕障不得升 R2/R6 |
| `twoways_obstacle` | AccidentTwoWays, ConstructionObstacleTwoWays, HazardAtSideLaneTwoWays, ParkedObstacleTwoWays | `two_way_min_pre_m=45-70`, `two_way_post_pad_m=20`, `trigger_close_m=70-75`, `two_way_layout_prior=true`, route 级 pre-core demotion | 核心帧看 opposite lane + 同向可用 lane 不足 / 借对向必要性；有 core 的 route 在 first core 前保持 R1/weak R2，core 后清晰 TwoWays layout 保留 R2 + review；无 core 但 RGB 清楚双向布局可召回 R2 + review |
| `default_meta_map` | ControlLoss, CrossingBicycleFlow, DynamicObjectCrossing, HardBreakRoute, HazardAtSideLane | `junction_pre_m=50`, `junction_post_m=25`; 场景动作多数 veto RS 升级 | 急刹、横穿、失控、side-lane hazard 只进 EVENT；RS 由路网/灯控决定 |
| `signalized_junction` | BlockedIntersection, OppositeVehicleRunningRedLight, RedLightWithoutLeadVehicle, Signalized*Turn*, T_Junction | `junction_pre_m=50-60`, `junction_post_m=20-25`; T_Junction `review_if_no_tl=True`; static-signal-only strong context <=25m | 有效灯态、light_hazard、signal/controller、stopline approach 至少多源一致；无有效 `traffic_light_state` 的 R4 必须写 RGB confirmation review |
| `defect_junction` | CrossJunctionDefectTrafficLight | `junction_pre_m=60`, `junction_post_m=20`, `override=r5_over_r4` | defect 场景即使有 signal/controller 也优先 R5；找不到路口只能 medium + review |
| `nonsignalized_junction` | NonSignalizedJunction*, OppositeVehicleTakingPriority, PriorityAtJunction | `junction_pre_m=45-60`, `junction_post_m=20` | no-light / priority / stop / yield 证据；连续有效灯态必须 conflict review |
| `pedestrian_crossing` | PedestrianCrossing | `junction_pre_m=40`, `junction_post_m=40`; `pedestrian_not_rs` | 行人只进 EVENT；R4/R5 取决于 crossing 是否与路口控制源同源 |
| `highway_merge` | EnterActorFlow*, HighwayCutIn, HighwayExit, MergerIntoSlowTraffic* | `merge_pre_m=30-50`, `merge_post_m=40-50`, `trigger_close_m=90`; slow traffic 保持 R3 | merge/split/ramp/lane-count-change；cut-in/slow/EnterFlow 名称不能单独 high |
| `interurban` | InterurbanActorFlow | `merge_pre_m=50`, `merge_post_m=45`, `junction_pre_m=55`, `junction_post_m=25`; projection-error R5 demotion | 先拆 R3 合流窗口，再拆 R4/R5 路口窗口；projection error 高时禁用 route_s hard boundary，R5 也必须有 RGB/低误差 junction/control 支撑 |
| `interurban_advanced` | InterurbanAdvancedActorFlow | `junction_pre_m=55`, `junction_post_m=25`, `r3_requires_topology=True`; projection-error R5 demotion | 默认 R1；只有明确 merge/split/ramp 才给 R3 medium/review，只有可见或低误差无灯路口/priority 才给 R5 |
| `invading_turn` | InvadingTurn | `two_way_min_pre_m=80`, `two_way_post_pad_m=20`, `trigger_close_m=75` | 对向车侵入/heading conflict；EVENT 区分被动让行与自车借道 |
| `parking` / `parking_exit` | ParkingCrossingPedestrian, ParkingCutIn, ParkingExit | `parking_pre_m=20-35`, `parking_post_m=50-60` | parking/shoulder/curbside 或 RGB 停车空间；行人/切入仍归 EVENT |
| `static_cutin` | StaticCutIn | `parking_pre_m=35`, `parking_post_m=55`, `merge_pre_m=35`, `merge_post_m=55` | R3/R6 二选一主导；分差小于 0.15 必须 review |
| `vehicle_opens_door_twoways` | VehicleOpensDoorTwoWays | `two_way_min_pre_m=50`, `two_way_post_pad_m=20`, `parking_pre_m=35`, `parking_post_m=55` | R2/R6 可能共存，primary 看当前主导决策，secondary 记录另一项 |
| `vehicle_turning` | VehicleTurningRoute, VehicleTurningRoutePedestrian | `junction_pre_m=50`, `junction_post_m=20-40`, `multi_trigger=True`; projection-error R5 demotion | 多 trigger 分段；行人/横穿不改变 RS，控制源决定 R4/R5；高投影误差下无灯 R5 必须有 stop / is_junction / 非静态可信 XODR 近路口证据 |
| `noscenario` | noScenarios | `junction_pre_m=50`, `junction_post_m=25`, `conservative=True` | 只允许 R1/R4；弱 topology hint 只写 evidence/review |

### 1.7 时序稳定与环岛仲裁

帧级规则输出后必须再经过 route 级时序稳定，避免 `R1 -> R4 -> R1` 或任意
`R* -> Rk -> R*` 的单帧/短片段扰动被当成真实道路结构切换：

- R2/R3/R4/R5/R6 最短有效持续为 4 帧（4Hz 下约 1 秒）。
- R1 最短有效持续为 2 帧；短 R1 夹在同一特殊 RS 中间时，视为噪音缝隙并填平。
- 短片段前后标签一致时直接改为该标签；前后不一致时并入更长邻接片段，并写
  `evidence.temporal_smoothing`。
- 去抖是所有 RS 的统一后处理，不是 R4 特例；`frame_rs_annotation.label` 必须反映去抖后的最终标签。

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
- `quick_start.py annotate-rs` 会逐帧调用 `RoadStructureRuleEngine`，保留旧字段
  `road_structures` 候选全集，同时新增 `primary_road_structure` 与显式单帧结果
  `frame_rs_annotation`。
- `frame_rs_annotation` 包含 `label/secondary/confidence/comment/rule_kind/rules_fired/decision_source/review_required/review_reasons/metrics/xodr_summary`，
  可直接作为人工验收和后续训练输入的帧级解释结果。
- `web_app.py` 已把候选全集和本帧最终标签拆开展示：顶部绿色标签读取
  `frame_rs_annotation.label` / `primary_road_structure`，置信度只表示该帧 primary RS 的置信度；
  `road_structures` 只作为“该 scenario 可选 RS 候选全集”展示，不再和本帧标注混用。
- Web 证据面板会展示 XML/route 投影、trigger 距离、LEAD meta 灯态/active scenario、XODR
  source/trusted/road/lane/junction/opposite/parking/merge 等摘要；若这些证据不足或冲突，
  页面会显示 review 状态和原因，供下一轮规则修正。
- `web_app.py` 默认路径已改为当前仓库相对路径：
  `AutoMoT/lead_data`、`AutoMoT/lead_video`、`AutoMoT/keyframe_filter/collection_output`；
  仍可用 `LEAD_DATA_ROOT`、`LEAD_VIDEO_ROOT`、`KEYFRAME_COLLECTION_OUTPUT` 覆盖。
- route 投影误差 `>5m` 时，代码会禁用 `route_s` hard window，只允许 trigger distance / meta active / junction/light 等证据参与，并写
  `route_s_window_disabled_projection_error_gt_5m`。
- `noScenarios` 调整为无 meta 有效灯态或 light hazard 时强制保守 R1；静态 XODR signal/junction hint 只进 evidence/review，不再把普通无场景帧自动推成 R4。
- `StaticCutIn` 调整为 cut-in 窗口内若没有 parking/merge 拓扑证据，则回 R1 中置信；R3/R6 仍必须有对应 XODR/RGB 证据才 high。
- RGB-first 全量复核覆盖 43 个 scenario、204 条 scenario-town route、24387 帧；
  `candidate_anomalies=15788` 只是逐帧看图索引，不是错帧数。summary、confidence、
  `candidate_anomalies` 和标签分布只能帮忙定位，最终异常必须人工读取每条 route 的
  `all_frames_*.jpg` 后确认。
- 标定与 RGB 不匹配主要来自六类根因：
  1. XML/XODR route 投影误差高，却仍把 route_s / trigger window 当 hard boundary。
  2. 静态 XODR signal/opposite/parking/merge/junction hint 与 RGB 不同源，被误当 high confirmation。
  3. scenario 名称、active scenario 或事件距离被当作 ROAD_STRUCTURE 真值，导致 EVENT 覆盖 RS。
  4. 坏 XODR 被当作否定证据，导致明显 merge 或 TwoWays road-layout 被压回 R1。
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
  TwoWays 既要避免只按场景名或 layout-prior 全段 R2，也要避免清晰双向 road-layout 被坏 XODR 压回 R1；
  具体执行为“core 前不铺 R2、core 后可延续、无 core 但 RGB 清晰则 review 召回”；
  R3 只表示 merge/ramp/split/exit 特殊结构，不表示物理高速路本身；
  R6 只表示停车带/路边停车空间主导，不表示 parked obstacle 事件；
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
`R1=242, R2=0, R3=20, R4=88, R5=70, R6=10`，confidence 为
`min=0.70/avg=0.8120/max=0.98`，review frame ratio 为 `0.4140`。
review 增加主要来自图像优先复核后新增的投影/静态拓扑降级归因：
`route_projection_error_high`、`static_xodr_topology_demoted_by_projection_error`、
`structure_window_demoted_by_projection_error`、`weaker_special_rs_kept_as_candidate_not_primary`、
`candidate_score_gap_lt_0.15`。
这表示可疑 R2/R3/R6/R4/R5 不再被静态拓扑或 scenario 窗口直接当主标签；
后续若用 `/home/codon/anaconda3/envs/carla/bin/python` 跑 CARLA API XODR probe，应优先比较这些 review 是否下降。

## 2. 逐场景规则

### 2.0 每场景 RGB 错配归因总表

| 场景 | 最容易导致标定与 RGB 不匹配的原因 | 当前修正口径 |
|---|---|---|
| Accident | 事故事件或 light_hazard 被误当道路结构，普通同向路段提前升 R4/R2 | 障碍只进 EVENT；只有强路口/灯控上下文才 R4 |
| AccidentTwoWays | trigger 前后清晰双向 road-layout 被坏 XODR/窗口结束压回 R1 | R2 拆成核心借道层和 TwoWays layout-prior 层 |
| BlockedIntersection | 低能见度下无 meta 灯态的 R4 容易被误判为置信度问题 | RGB 可见 stopline/crosswalk/cross traffic/blocked pocket 时保留 R4 + review |
| ConstructionObstacle | 施工物事件或静态 signal 近邻把普通路段提前升 R4/R2 | 施工只进 EVENT；static-signal-only strong context 限 25m |
| ConstructionObstacleTwoWays | 施工锥桶与 two-way road-layout 混淆 | 核心占道看 R2；非核心双向道路看 layout-prior；未占道则 EVENT |
| ControlLoss | 失控/低速/急刹被误当停车或路口结构 | 失控只进 EVENT；RS 由道路与灯控决定 |
| CrossJunctionDefectTrafficLight | 看到信号灯硬件就误转 R4 | defect/异常信号控制场景优先 R5，硬件可见不等于正常灯控 |
| CrossingBicycleFlow | actor flow 字样误触发 R3 | 自行车横穿只进 EVENT；只有真实灯控路口才 R4 |
| DynamicObjectCrossing | 动态对象横穿被当成无灯路口/停车结构 | 动态对象只进 EVENT；RS 只看路网/灯控 |
| EnterActorFlow | enter flow 与 merge/ramp 拓扑混淆 | 只有可见或可信 merge/split/ramp 才 R3，完成后 R1 |
| EnterActorFlowV2 | 短 route / 投影抖动导致 R3 边界过宽 | 同 EnterActorFlow，短 route 必须靠 RGB 和强拓扑确认 |
| HardBreakRoute | 急刹被误当红灯/停车结构 | 急刹只进 EVENT；红灯停车必须有灯控/stopline 同源 |
| HazardAtSideLane | side-lane hazard 被误当 R2 | 非 TwoWays 默认 R1 + EVENT；只有对向参与才 review R2 |
| HazardAtSideLaneTwoWays | two-way hazard 与普通双向道路/占道核心混在一起 | core 前保持 R1；core/后段双向布局可 R2 + review，核心借道需障碍/对向证据 |
| HighwayCutIn | 物理高速路或切入事件被误标为 R3 | R3 只表示 merge/ramp/split/exit 特殊结构；普通高速直行 R1 |
| HighwayExit | exit 名称导致整段 R3 | 只在出口/分流拓扑窗口 R3，驶出完成回 R1 |
| InterurbanActorFlow | 高投影误差下 stop/junction hint 把普通城际路升 R5 | R5 必须有 RGB 可见或低误差 junction/control 支撑 |
| InterurbanAdvancedActorFlow | advanced/actor-flow 先验过强，普通城际道路误升 R5/R3 | 默认 R1；R3/R5 都必须有可见或可信拓扑 |
| InvadingTurn | 对向车事件与自车借道结构混淆 | R2 表示对向参与决策；EVENT 区分被动让行/侵入 |
| MergerIntoSlowTraffic | 坏 XODR 把明显 merge 压回 R1，或 active scenario 把 merge 完成后拖成 R3 | actor-flow/trigger 强近邻可召回 R3；active scenario 不能单独延长 |
| MergerIntoSlowTrafficV2 | 与 Merger 相同，且短路线更容易出现 R3/R1 边界误差 | 按 RGB 分段；明显 merge 保 R3，宽直主路回 R1 |
| NonSignalizedJunctionLeftTurn | 静态 signal hint 与无灯路口语义冲突 | 无灯/priority/stop 证据给 R5；正常灯态进入 conflict review |
| NonSignalizedJunctionLeftTurnEnterFlow | EnterFlow 名称误触发 R3 | 这是无灯路口进入车流，主口径 R5，不是匝道 R3 |
| NonSignalizedJunctionRightTurn | slip lane / 右转动作被误当 R3 | 无灯 junction 给 R5；右转动作本身不是 R3 |
| OppositeVehicleRunningRedLight | 闯红灯事件被误当 R5 | 信号规则仍有效，主 RS 是 R4，违规进 EVENT |
| OppositeVehicleTakingPriority | priority/no-light 与正常灯控冲突 | priority/no-light 给 R5；有效灯态必须 conflict review |
| ParkedObstacle | parked 字样误触发 R6 | ParkedObstacle 是障碍 EVENT；不是 parking-space RS |
| ParkedObstacleTwoWays | parked / TwoWays / R6 混淆 | core 前保持 R1；需要借对向时 R2；parked 本身不等于 R6 |
| ParkingCrossingPedestrian | 行人横穿与停车结构混淆 | 停车空间主导才 R6；行人进 EVENT；灯控优先 R4 |
| ParkingCutIn | 切入事件或 shoulder hint 误触发 R6 | 必须有停车带/路边车/curbside 证据才 R6 |
| ParkingExit | 停车驶出完成后仍保持 R6 | parking-to-driving transition 时 R6，完成后回 R1 |
| PedestrianCrossing | 行人/斑马线被直接当 R4/R5 | RS 由 crossing 是否同源于路口控制决定，行人进 EVENT |
| PriorityAtJunction | 夜间/低能见度下把 R5 当置信度异常 | 可见 priority/junction/cross traffic 时 R5；普通离开段回 R1 |
| RedLightWithoutLeadVehicle | is_junction=false 时漏掉 stopline approach | 有效灯态/stopline approach 同源即可 R4 |
| SignalizedJunctionLeftTurn | 左转让行被误当 R5 | 信号灯左转仍是 R4，让行/冲突进 EVENT |
| SignalizedJunctionLeftTurnEnterFlow | EnterFlow 字样误触发 R3 | 信号灯路口内仍 R4，enter-flow 只进 EVENT |
| SignalizedJunctionRightTurn | 右转专用道或稀疏 route 误触发 R3/R1 | 有效 signal/controller/stopline 给 R4，离开后 R1 |
| StaticCutIn | 静态车切入来源不明，R3/R6/R1 混淆 | parking 来源 R6，merge 来源 R3，都缺证据则 R1 |
| T_Junction | 默认 T 路口被误扩展成 R5 | 当前按 signalized_junction R1/R4；无灯 T 需 RGB 证据后再扩候选 |
| VehicleOpensDoorTwoWays | 开门停车侧与双向借道冲突 | R6/R2 可共存，primary 看当前主导决策，另一项进 secondary/review |
| VehicleTurningRoute | 投影误差高时普通转弯前路段被过宽 junction window 标 R4/R5 | 高误差无灯 R5 必须有 stop/is_junction/可信 XODR 近路口证据 |
| VehicleTurningRoutePedestrian | 行人事件和转弯路口窗口导致普通住宅道路误标 R5 | 行人进 EVENT；普通路段 R1，真实 stop/priority/路口段 R5 |
| noScenarios | 弱静态 topology hint 在无事件场景中制造特殊 RS | 保守 R1/R4；不从弱 hint 自动挖 R3/R5/R6 |

### Accident

- 候选 RS：R1, R4。
- 已确定口径：事故障碍本身是 EVENT，不是 ROAD_STRUCTURE；默认 R1，进入真实信号灯路口才切 R4。
- 分段逻辑：finite `dist_to_accident` / active scenario 只用于标记事故事件窗口；窗口内仍按道路结构判断 R1/R4。
- 证据需求：XML accident trigger + meta distance 确定事件窗口；R4 必须有有效灯态且具备强路口/stopline/signal-junction 上下文，
  或可信 XODR/meta junction 与静态 signal 同源。仅有 static signal near / distance-to-junction 不足以把普通路段升 R4。
- 待完善点：不要把同向绕障误升为 R2；若 projection error 高，事故窗口只给事件候选，不改 RS。

### AccidentTwoWays

- 候选 RS：R1, R2, R4。
- 已确定口径：two-way 障碍核心窗口内，优先用可信 XODR 证明对向 lane 参与；若 XODR 不可信，近距离障碍、
  `accident_two_ways_stuck`、`vehicle_hazard` 或明显 lane-change 核心证据可给 R2=0.90，并强制 review。
  但 RGB-first 复核确认不能把 `two_way_layout_prior` 当全 route 先验：
  对存在 core obstruction 的 route，first core 前仍按 R1/weak R2；core 后非核心帧若 RGB 仍是清晰双向布局，
  使用 `two_way_layout_prior` 保留 R2=0.82 + review。
  对没有 core obstruction 的 route，例如本轮 `Town02` 样本，若 RGB 清楚显示中心黄线/双向单车道，
  可作为纯 layout 召回保留 R2 + review。
  灯控路口仍优先 R4，但有效灯态缺强路口上下文时只保留 weak R4 candidate。
- 分段逻辑：先判 R4；再判 R2。核心障碍/借道帧由 XML trigger / distance / active window + opposite driving lane /
  同向可用 lane 不足 / meta obstruction 支撑；route 级后处理会把 first core 前仅由 layout-prior 给出的 R2
  通过 `twoways_layout_prior_pre_core_demoted_to_r1` 收回 R1，避免 `Town01/Town12/Town13` 这类接近段被全程铺 R2。
  core 后或无 core 但 RGB 清晰的双向 layout，才保持 R2 + review。
- 证据需求：XODR lane_id 符号反转、lane direction、lane count、meta active scenario、近距离障碍距离、
  `scenario_obstacles_ids`、`signed_dist_to_lane_change`、RGB 对向借道边界。
- RGB 复核结论：`rs_full_frame_review_twoways_boundary_recheck` 中 7 个 town 共 1119 帧，
  从旧全 R2 收紧为 `R1=277/R2=842`；`Town01/Town07/Town12/Town13` 回到 core 前 R1、core/后段 R2，
  `Town02` 因无 core 但 RGB 黄线双向布局清楚而保留全程 R2 + review。
- 待完善点：`Town15` 有强停车/路边车列视觉，当前仍归 R2 + review；后续做 Parking/R6 复核时要确认它是纯 TwoWays road-layout，
  还是应在 EVENT/secondary 里记录 parking semantics。

### BlockedIntersection

- 候选 RS：R1, R4。
- 已确定口径：blocked intersection 是路口内阻塞事件，不等于 R5；有正常灯控时 primary R4。
- 分段逻辑：接近/进入受控 junction 设 R4；离开 junction 后回 R1；阻塞对象作为 EVENT。
  如果 R4 不是由有效 `traffic_light_state` 支撑，而是由 junction/window/static signal 支撑，
  必须写 `signalized_r4_without_meta_tl_requires_rgb_confirmation` 并逐帧看 RGB；低能见度
  contact sheet 只能用于定位，最终以单帧 RGB 的 stopline/crosswalk/cross traffic/blocked pocket 为准。
- 证据需求：XODR junction + signal/controller、meta traffic_light_state、XML trigger。
- RGB 复核结论：`Town06` 雨雾 f100/f130 单帧 RGB 仍可见左侧 blocked intersection pocket，
  因此保留 R4；`Town07` 后段低能见度且 meta 灯态缺失，保留 R4 但强制 RGB review；
  `Town12` 夜间投影误差高，但 RGB 可见斑马线/横向车流，保留 R4 + projection review；
  `Town13` 是参考分段，f20-f85 R4、f86 后退出到 R1，说明规则应按可见路口范围退出。
- 待完善点：若 meta `is_junction=false` 但 stopline / traffic light 有效，应按 stopline approach 保留 R4 并 review；
  若人工 RGB 发现无 stopline/crosswalk/cross traffic/blocked pocket 且只剩静态 signal 近邻，则应把 R4 降为候选并让 R1 做主标签。

### ConstructionObstacle

- 候选 RS：R1, R4。
- 已确定口径：施工障碍是 EVENT；非 TwoWays 场景不应自动进入 R2。
- 分段逻辑：施工窗口内仍按普通同向道路 R1，真实灯控路口覆盖为 R4。
- 证据需求：XML construction trigger + meta distance 定位事件，XODR/灯态确认 R4。
- RGB 复核结论：`Town04` 曾在 f203 过早升 R4，单帧 RGB 仍是雾中普通弯道/跟车；
  根因是 static signal + 有效灯态在 35m 内就触发 strong_control_context。代码已把 static-signal-only
  strong context 收紧到 25m，R4 起点后移到 f209；施工主体仍全段 R1。
- 待完善点：需要 RGB 边界帧确认施工物是否实际导致 lane closure；没有 two-way/opposite 证据时不升 R2。

### ConstructionObstacleTwoWays

- 候选 RS：R1, R2, R4。
- 已确定口径：two-way 施工障碍可进入 R2，但 high 需要 XODR 对向 lane + 同向 lane 不足。
- 分段逻辑：R4 优先；two-way construction 窗口内满足对向参与条件给 R2；有 core 的 route 在 first core 前保持 R1，
  core 后清晰 TwoWays layout 可用 `two_way_layout_prior` 延续 R2 + review。
- 证据需求：XML distance/offset、XODR opposite lane、meta active scenario、RGB 施工占道。
- RGB 复核结论：TwoWays 家族边界复核中 `ConstructionObstacleTwoWays` 7 条 route 共 1042 帧，
  `twoways_layout_prior_pre_core_demoted_to_r1` 收回 183 帧 core 前 R2，剩余 `R1=183/R2=859`。
- 待完善点：施工锥桶/障碍只在路边但不压占自车路径时，应保持 R1 + EVENT。

### ControlLoss

- 候选 RS：R1, R4。
- 已确定口径：失控/急刹是 EVENT，不改变 RS；默认 R1，路口灯控才 R4。
- 分段逻辑：speed、brake、accel、yaw 异常只进入事件层；RS 由道路结构证据决定。
- 证据需求：meta speed/brake/accel 作为事件证据，XODR/traffic light 作为 RS 证据。
- 待完善点：长时低速不能误判为 parking R6；没有 parking topology 时仍是 R1。

### CrossJunctionDefectTrafficLight

- 候选 RS：R1, R5。
- 已确定口径：defect traffic light 场景中，即使 XODR/meta 有 signal/controller，也按“信号失效路口”归 R5，不归 R4。
- 分段逻辑：trigger 对应 junction/stopline 前后窗口设 R5；找不到 junction 时给 R5 medium + review；窗口外 R1。
- 证据需求：XML trigger / traffic_direction / source_dist_interval、XODR junction/signal、meta active scenario、RGB 确认路口。
- 待完善点：实测 map `is_junction=false` 仍可能全段 R5，说明 stopline approach 和 junction polygon 需要分开建证据。

### CrossingBicycleFlow

- 候选 RS：R1, R4。
- 已确定口径：自行车横穿/流量是 EVENT；有正常信号灯路口才 R4，否则保持 R1。
- 分段逻辑：bicycle flow 窗口只改变事件；traffic light / controlled junction 同源时切 R4。
- 证据需求：XML bicycle trigger、meta active scenario、traffic_light_state、XODR junction/signal。
- 待完善点：不要因为 actor flow 字样误归 R3；R3 只给 merge/split/ramp 拓扑。

### DynamicObjectCrossing

- 候选 RS：R1, R4。
- 已确定口径：动态对象横穿是 EVENT，不直接定义道路结构。
- 分段逻辑：多数帧 R1；进入有效灯控路口时 R4；对象横穿窗口叠加 EVENT。
- 证据需求：XML trigger、meta distance/active scenario、XODR/traffic light。
- 待完善点：若对象出现在无灯路口，需要补充规则是否允许 R5；目前只在有 junction/priority 证据时 review。

### EnterActorFlow

- 候选 RS：R1, R3, R4。
- 已确定口径：actor flow 的合流/汇入窗口可归 R3，但 high 必须有 ramp/merge/lane-count-change 或主辅路证据。
- 分段逻辑：actor-flow start/end / trigger 窗口 + XODR merge 拓扑给 R3；灯控路口覆盖 R4；完成合流后 R1。
- 证据需求：XML start/end actor flow、other_actor_location、XODR merge/split、meta active scenario、RGB 车流关系。
- 待完善点：只靠 XML actor-flow 不足以 high；需要把 candidate recall window 与 topology confirmation window 分开。

### EnterActorFlowV2

- 候选 RS：R1, R3, R4。
- 已确定口径：与 EnterActorFlow 同口径；V2 的 route/trigger 可能更短，不能用整段 sparse route 当 R3。
- 分段逻辑：只在 actor-flow 核心窗口内尝试 R3；窗口外回 R1；灯控路口 R4。
- 证据需求：XML start/end、meta active scenario、XODR ramp/merge、RGB 侧向汇入。
- 待完善点：短 route 下 projection 抖动更容易污染边界，必须用 meta/XODR waypoint 吸附辅助。

### HardBreakRoute

- 候选 RS：R1, R4。
- 已确定口径：急刹是 EVENT；RS 默认 R1，灯控路口才 R4。
- 分段逻辑：brake/accel/speed 只决定事件窗口；道路结构由 XODR/meta junction 和灯态决定。
- 证据需求：meta brake/speed、XML route、XODR signal/junction。
- 待完善点：急刹靠近路口时要区分“红灯停车 R4”与“前车急刹 R1 + EVENT”。

### HazardAtSideLane

- 候选 RS：R1, R4。
- 已确定口径：非 TwoWays 侧向 hazard 不直接进入 R2；默认 R1。
- 分段逻辑：hazard 事件窗口叠加 EVENT；有灯控路口切 R4。
- 证据需求：XML hazard trigger、meta active scenario、XODR/traffic light。
- 待完善点：若局部 XODR 证明实际需要借对向，应作为数据异常或 review，而非直接改规则。

### HazardAtSideLaneTwoWays

- 候选 RS：R1, R2, R4。
- 已确定口径：two-way side-lane hazard 可进入 R2，但必须证明对向 lane 影响当前决策。
- 分段逻辑：灯控路口 R4；two-way hazard 窗口 + opposite lane + 同向 lane 不足给 R2；
  若 route 有 core 证据，first core 前的 layout-prior 仍回 R1，避免只靠 TwoWays 名称全段 R2。
- 证据需求：XML trigger/distance、XODR lane direction、meta active scenario、RGB 对向/路边风险。
- RGB 复核结论：TwoWays 家族边界复核中 `HazardAtSideLaneTwoWays` 2 条 route 共 413 帧，
  core 前 demotion 251 帧，剩余 `R1=251/R2=162`；后续重点看 side-lane hazard 是否真正压占自车路径。
- 待完善点：侧向风险未压占自车路径时，保持 R1 + EVENT。

### HighwayCutIn

- 候选 RS：R1, R3, R4。
- 已确定口径：他车切入是 EVENT；物理上像高速/快速路不等于 R3，只有道路本身处于主辅路、匝道、合流、分流或驶出拓扑时才 R3。
- 分段逻辑：merge/ramp/split/exit 或 lane-count-change topology 成立时 R3；普通高速多车道同向切入仍 R1；灯控路口 R4。
- 证据需求：XODR road/lane topology、XML actor flow、meta active scenario、RGB 车道线和侧向车流。
- 待完善点：不能把所有 cut-in 都当 R3；R3 high 需要 topology，不是 action 名称。

### HighwayExit

- 候选 RS：R1, R3, R4。
- 已确定口径：驶出/分流是 R3 典型结构，但 high 需要 exit/split/lane-count-change 证据。
- 分段逻辑：exit/actor-flow 窗口 + XODR split/ramp 给 R3；驶出完成后 R1；灯控路口 R4。
- 证据需求：XML route/trigger/other_actor_location、XODR ramp/split、meta active scenario、RGB 出口边界。
- 待完善点：实测仅有 `r3_lacks_xodr_merge_split_confirmation` 时仍可能高分，代码应把这类帧降为 medium + review。

### InterurbanActorFlow

- 候选 RS：R1, R3, R4, R5。
- 已确定口径：前段可能是 R3 合流/主辅路，后段可能进入 R4/R5 路口；不能整段同一个 RS。
- 分段逻辑：merge/actor-flow 拓扑窗口给 R3；真实灯控路口给 R4；无灯/priority 路口给 R5；其余 R1。
  R5 必须有 RGB 可见或低投影误差下的无信号 junction/stop/priority 证据；`route_projection_error_high`
  时的 stop/junction hint 只能降为弱候选 + review，不能压过 R1。
- 证据需求：XML actor-flow + route、XODR merge/junction/signal、meta traffic_light_state/is_junction、RGB 边界帧。
- RGB 复核结论：`Town12` 大面积 R5 与 RGB 不符，已回到 R1；
  `Town13` 有可见无信号 fork/junction 和侧向/对向车流，保留分段 R5。
- 待完善点：实测 projection error 多，边界必须优先使用 RGB + XODR/meta waypoint 吸附，不能只看 sparse route_s。

### InterurbanAdvancedActorFlow

- 候选 RS：R1, R4, R5，R3 仅 medium/review。
- 已确定口径：advanced actor flow 不自动 R3/R5；R3 只有出现明确 merge/split/ramp 时才成立，
  R5 也必须有可见或低投影误差下的无信号路口/priority 证据。
- 分段逻辑：junction + light 给 R4；junction/priority/no-light 给 R5；明确合流拓扑才临时 R3；其余 R1。
- 证据需求：XODR junction/signal/priority、XML actor-flow、meta light/junction、RGB 路口和侧向车流。
- RGB 复核结论：`Town12/Town13` 旧 R5 多由 scenario/stop hint 触发，RGB 更像普通城际道路，
  已回到 R1 + review。
- 待完善点：实测大量 R5/R1 且 review 高，需补更细的路口窗口和投影修正。

### InvadingTurn

- 候选 RS：R1, R2, R4。
- 已确定口径：对向车侵入/转弯占道属于 R2 决策空间，事件层区分被动让行 U-E5，不是自车主动借道。
- 分段逻辑：invading trigger 窗口 + opposite lane / heading conflict 给 R2；灯控路口 R4；窗口外 R1。
- 证据需求：XML trigger/turn direction、XODR lane direction、meta active scenario、RGB 对向车轨迹。
- 待完善点：需要在 EVENT 中区分“自车借对向绕障”和“对向车侵入自车道”，RS 可同为 R2。

### MergerIntoSlowTraffic

- 候选 RS：R1, R3, R4。
- 已确定口径：慢车流合流是 R3，只要局部 merge/ramp 拓扑仍成立，低速不改变 RS。
- 分段逻辑：merge/actor-flow 窗口 + topology 给 R3；若 XODR/route 投影失效但 XML actor-flow
  强近邻或 trigger 距离仍支持 merge，则走
  `r3_merger_actor_flow_or_trigger_fallback` 给 R3 并保留 review；合流完成回 R1；灯控路口 R4。
- RGB 复核结论：`Town12/Town13` 不再出现明显 merge 被 all/mostly R1 吃掉的问题。
  可见 merge/actor-flow 窗口为 R3；中间普通主路段回 R1；`Town13` 二次 split/merge 边界
  再回 R3。这是正确的分段模式，不应为了 scenario 名称把全段强行 R3。
- 证据需求：XML flow window、`start_actor_flow/end_actor_flow`、XODR merge/lane count、meta speed/active scenario、RGB 慢车流间隙。
- 待完善点：不要把低速误归 R6 或 R5；低速只是事件/动作状态。复核时优先看 RGB 与
  `actor_flow_distance_m`，不要因为 `route_projection_error_high` 自动压回 R1。
  active scenario 只能辅助解释，不能单独延长 R3；如果 RGB 显示 merge 已完成，应回 R1。

### MergerIntoSlowTrafficV2

- 候选 RS：R1, R3, R4。
- 已确定口径：同 MergerIntoSlowTraffic；V2 只影响样本/route 形态，不改变可见 RS 语义。
- 分段逻辑：只在合流核心窗口内 R3；完成后 R1；灯控路口 R4。
- RGB 复核结论：`Town06` 短 route 全段处于 merge context；`Town12/Town13` 是分段 R3/R1，
  其中 `Town12` 末尾可见 split/merge boundary 后恢复 R3。不能再出现“明显 merge 路口基本都标 R1”，
  但也不能把宽直主路跟车段全刷成 R3。
- 证据需求：XML actor-flow、XODR merge、meta active scenario、RGB 侧向车流。
- 待完善点：V2 需要和 canonical scene 合并时保留 raw scenario，用于回查失败样例。

### NonSignalizedJunctionLeftTurn

- 候选 RS：R1, R5。
- 已确定口径：无信号灯左转是 R5；若出现正常灯态，只能 medium/review，不应静默改 R4。
- 分段逻辑：junction/priority/stop/yield 窗口 R5；窗口外 R1。
- 证据需求：XODR junction/priority/sign、XML route trigger、meta is_junction、RGB 路口入口。
- 待完善点：Town10HD 样本曾出现 meta 缺口；缺 meta 时只能用 XODR/XML 给 medium 并 review。

### NonSignalizedJunctionLeftTurnEnterFlow

- 候选 RS：R1, R5。
- 已确定口径：名字里有 EnterFlow，但这是无信号灯路口进入车流，不是匝道 R3。
- 分段逻辑：junction/priority/no-light 窗口 R5；窗口外 R1；仅明确 ramp/merge 才 review R3。
- 证据需求：XODR junction/priority、XML enter-flow trigger、meta active scenario、RGB 路口车流。
- 待完善点：实测 `enter_flow_not_r3` 方向正确，但 projection error 高，需要补边界帧。

### NonSignalizedJunctionRightTurn

- 候选 RS：R1, R5。
- 已确定口径：无灯右转路口归 R5；右转动作本身不是 R3。
- 分段逻辑：no-light junction 窗口 R5；离开后 R1。
- 证据需求：XODR junction/priority/yield、XML route turn、meta junction、RGB 入口/横向车流。
- 待完善点：若右转专用 slip lane 有 yield/merge 属性，需人工决定 R5 vs R3 的优先级，目前保守 review。

### OppositeVehicleRunningRedLight

- 候选 RS：R1, R4。
- 已确定口径：对向车闯红灯是 R4 下的突发事件，不是 R5；信号规则仍有效。
- 分段逻辑：受控路口/红绿灯窗口 R4；窗口外 R1；违规对象进入 EVENT。
- 证据需求：meta traffic_light_state、XODR signal/controller、XML trigger、RGB 对向车与灯态。
- 待完善点：若灯态缺失但场景名强提示，最多 R4 medium + review。

### OppositeVehicleTakingPriority

- 候选 RS：R1, R5。
- 已确定口径：对向车抢优先权发生在无信号/优先级路口，归 R5。
- 分段逻辑：priority/no-light junction 窗口 R5；窗口外 R1。
- 证据需求：XODR priority/yield/junction、XML trigger、meta active scenario、RGB 对向车流。
- 待完善点：若存在有效灯态，应进入 conflict review，不能自动 R4 或 R5。

### ParkedObstacle

- 候选 RS：R1, R4。
- 已确定口径：ParkedObstacle 不是 Parking*；停放障碍是 EVENT，默认 R1，不自动 R6。
- 分段逻辑：障碍窗口只标事件；灯控路口 R4；其余 R1。
- 证据需求：XML obstacle trigger、meta distance/active scenario、XODR/traffic light。
- 待完善点：只有确认障碍来自停车带/路边停车空间且主导决策时，才考虑 review R6。

### ParkedObstacleTwoWays

- 候选 RS：R1, R2, R4。
- 已确定口径：停放障碍 + two-way 可进入 R2；“parked” 不是 R6 的充分条件。
- 分段逻辑：two-way obstruction 窗口 + opposite lane 参与给 R2；灯控路口 R4；
  route 有 core 证据时 first core 前保持 R1，core 后清晰双向 layout 可延续 R2 + review。
- 证据需求：XML distance/offset、XODR lane direction/count、meta active scenario、RGB 借道。
- RGB 复核结论：TwoWays 家族边界复核中 `ParkedObstacleTwoWays` 2 条 route 共 160 帧，
  core 前 demotion 64 帧，剩余 `R1=64/R2=96`；parking/R6 只能在停车空间主导时进入 secondary 或后续复核。
- 待完善点：路边停放但不需要借对向时，保持 R1 + EVENT。

### ParkingCrossingPedestrian

- 候选 RS：R1, R4, R6。
- 已确定口径：停车区/路边停车空间主导时 R6；行人横穿是 EVENT；灯控路口优先 R4。
- 分段逻辑：parking/curbside/shoulder 窗口 + 停车空间证据给 R6；灯控路口 R4；其余 R1。
- 证据需求：XML direction/crossing_angle、XODR parking/shoulder/curbside、meta active scenario、RGB 行人和停车侧。
- 待完善点：实测 projection error 多，R4/R6 边界需要 RGB boundary frames 核验。

### ParkingCutIn

- 候选 RS：R1, R4, R6。
- 已确定口径：路边停车车切入主路时 R6；灯控路口优先 R4；切入行为本身进入 EVENT。
- 分段逻辑：parking trigger/distance 窗口 + parking/shoulder/curbside 或 parked car side evidence 给 R6；窗口外 R1。
- 证据需求：XML direction/front/behind、XODR parking lane/shoulder、meta active scenario、RGB 停车车启动。
- 待完善点：R6 high 必须有停车空间或路边车证据；不能只靠 scenario 名称。

### ParkingExit

- 候选 RS：R1, R4, R6。
- 已确定口径：从停车位/路边驶出并汇入主路的过程归 R6，汇入完成后回 R1。
- 分段逻辑：parking exit 窗口 + parking-to-driving transition 给 R6；灯控路口 R4；完成后 R1。
- 证据需求：XML parking trigger、XODR parking/curbside、meta pose/speed/active scenario、RGB 车位出口。
- 待完善点：与 R3 merge 的区别是停车空间 vs 匝道/主辅路拓扑，二者接近时必须 review。

### PedestrianCrossing

- 候选 RS：R1, R4, R5。
- 已确定口径：行人横穿是 EVENT；RS 由是否在路口、是否有有效灯控决定。
- 分段逻辑：有效灯控 junction/stopline 给 R4；无灯/priority crossing junction 给 R5；普通路段 crossing 保持 R1。
- 证据需求：XODR signal/junction/crosswalk、meta traffic_light_state/is_junction、XML pedestrian trigger、RGB 行人/斑马线。
- 待完善点：实测 R4/R5 review 很高，crosswalk 与 junction/source 不同源时要降置信。

### PriorityAtJunction

- 候选 RS：R1, R5。
- 已确定口径：优先权路口属于 R5；有正常 traffic light 时必须 conflict review。
- 分段逻辑：priority/yield/stop controlled junction 窗口 R5；窗口外 R1。
- 证据需求：XODR priority/yield/sign/junction、XML trigger、meta active scenario、RGB 横向车流。
- 待完善点：需要显式区分 priority sign 与 traffic light controller 的冲突。

### RedLightWithoutLeadVehicle

- 候选 RS：R1, R4。
- 已确定口径：红灯停车是 R4；即使 `is_junction=false`，只要 stopline/traffic light approach 同源，也应 R4。
- 分段逻辑：traffic light / stopline approach 窗口 R4；离开灯控区 R1。
- 证据需求：meta traffic_light_state、XODR signal/stopline/controller、XML route trigger、RGB 灯态。
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
- 分段逻辑：signalized junction 窗口 R4；窗口外 R1；enter-flow 只影响事件。
- 证据需求：traffic_light_state、XODR controller/junction、XML enter-flow trigger、RGB 对向/横向车流。
- 待完善点：名字中的 EnterFlow 不能作为 R3 先验。

### SignalizedJunctionRightTurn

- 候选 RS：R1, R4。
- 已确定口径：信号灯右转是 R4；右转专用道/短 slip lane 不应因 sparse route 被误判 R3。
- 分段逻辑：signal/controller/stopline 窗口 R4；离开后 R1。
- 证据需求：meta traffic_light_state、XODR signal/controller、XML turn route、RGB 灯态和右转车道。
- 待完善点：实测右转场景不应依赖稀疏 XML route 线；要用灯态和 stopline source 做主证据。

### StaticCutIn

- 候选 RS：R1, R3, R4, R6。
- 已确定口径：StaticCutIn 不是单一结构；每个 run 要按局部拓扑拆成停车侧 R6、合流侧 R3 或普通 R1。
- 分段逻辑：parking/curbside evidence 给 R6；merge/ramp/split evidence 给 R3；灯控路口 R4；否则 R1。
- 证据需求：XODR parking/shoulder/merge/split、XML trigger、meta active scenario、RGB 静态车切入来源。
- 待完善点：R3/R6 不能同时 high；若二者证据分差小于 0.15，必须 review。

### T_Junction

- 候选 RS：R1, R4。
- 已确定口径：本轮自动草案与 `ROAD_EVENT_CANDIDATE_MAPPING.md` 均把 T_Junction 归为 signalized junction；当前默认按等绿灯通过处理，不预先加入 R5。
- 分段逻辑：junction/stopline 窗口内先查 traffic light/controller；有效灯态 R4；无有效灯态或 priority/no-light 证据只给 review，不直接 high R5；窗口外 R1。
- 证据需求：XODR junction/signal/priority、meta traffic_light_state、XML route turn、RGB 路口。
- 待完善点：`thresholds.json` 已有 `review_if_no_tl=True`；若 RGB 边界帧证明存在稳定无灯 T 路口，再同步扩展候选表与本文为 R1/R4/R5。

### VehicleOpensDoorTwoWays

- 候选 RS：R1, R2, R4, R6。
- 已确定口径：开门风险来自停车/路边空间时可 R6；若必须借对向/等待对向则 R2；灯控路口 R4。
- 分段逻辑：parking/open-door context 给 R6；two-way opposite-lane interaction 给 R2；二者冲突时看当前主导决策并 secondary 记录另一项。
- 证据需求：XML trigger/direction、XODR parking/opposite lane、meta active scenario、RGB 开门车辆和对向车。
- 待完善点：实测 R2/R6 抖动明显，需要 hysteresis、动作因果和分差 review。

### VehicleTurningRoute

- 候选 RS：R1, R4, R5。
- 已确定口径：转弯路线本身不是 RS；是否 R4/R5 取决于路口控制源。
- 分段逻辑：signalized junction 给 R4；non-signalized/priority junction 给 R5；普通弯道/车道跟随 R1。
- RGB 回灌：投影误差高时，不能只靠 `junction_window` 把无灯车转弯场景整段拉成 R5；
  必须有 `stop_hazard`、`is_junction` 或非静态可信 XODR 近路口证据，否则给 R1 并保留
  `vehicle_turning_r5_demoted_projection_error_rgb_required` 复核线索。
- 证据需求：XML turn route、XODR junction/signal/priority、meta light/junction、RGB 路口。
- 待完善点：多 trigger 场景要按每个路口窗口分段，不能整段一个 RS。

### VehicleTurningRoutePedestrian

- 候选 RS：R1, R4, R5。
- 已确定口径：行人是 EVENT；RS 仍由 turn route 所在控制源决定。
- 分段逻辑：有效灯控路口 R4；无灯/priority 路口 R5；普通路段 R1。
- RGB 回灌：`Town12` 旧结果把 0-144 基本整段标 R5，但 RGB 中 9-82 是普通住宅道路跟车；
  新规则将高投影误差且缺少真实路口/stop/可信 XODR 的帧降回 R1，只在 83-144 的 stop/路口/转弯区域保留 R5。
- 证据需求：XML turn/pedestrian trigger、XODR signal/junction/crosswalk、meta light/junction、RGB 行人边界。
- 待完善点：行人 crossing 与路口控制源不同步时，保留 primary RS 但 review。

### noScenarios

- 候选 RS：R1, R4。
- 已确定口径：没有 scenario 事件先验时保守 R1；只有明确有效 signal/controller/stopline/junction 才 R4。
- 分段逻辑：普通路段 R1；正常灯控路口 R4；不从弱 XODR hint 自动产生 R2/R3/R5/R6。
- 证据需求：XODR signal/junction、meta traffic_light_state/is_junction、XML route。
- 待完善点：若后续要从 noScenarios 中挖 R5/R3/R6，必须先独立建立 topology-only 高置信规则和人工样本集。

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
