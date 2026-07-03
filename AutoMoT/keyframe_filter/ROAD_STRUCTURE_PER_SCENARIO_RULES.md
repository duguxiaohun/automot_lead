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
| `twoways_obstacle` | AccidentTwoWays, ConstructionObstacleTwoWays, HazardAtSideLaneTwoWays, ParkedObstacleTwoWays | `two_way_min_pre_m=45-70`, `two_way_post_pad_m=20`, `trigger_close_m=70-75` | opposite lane + 同向可用 lane 不足 / 借对向必要性 |
| `default_meta_map` | ControlLoss, CrossingBicycleFlow, DynamicObjectCrossing, HardBreakRoute, HazardAtSideLane | `junction_pre_m=50`, `junction_post_m=25`; 场景动作多数 veto RS 升级 | 急刹、横穿、失控、side-lane hazard 只进 EVENT；RS 由路网/灯控决定 |
| `signalized_junction` | BlockedIntersection, OppositeVehicleRunningRedLight, RedLightWithoutLeadVehicle, Signalized*Turn*, T_Junction | `junction_pre_m=50-60`, `junction_post_m=20-25`; T_Junction `review_if_no_tl=True` | 有效灯态、light_hazard、signal/controller、stopline approach 至少多源一致 |
| `defect_junction` | CrossJunctionDefectTrafficLight | `junction_pre_m=60`, `junction_post_m=20`, `override=r5_over_r4` | defect 场景即使有 signal/controller 也优先 R5；找不到路口只能 medium + review |
| `nonsignalized_junction` | NonSignalizedJunction*, OppositeVehicleTakingPriority, PriorityAtJunction | `junction_pre_m=45-60`, `junction_post_m=20` | no-light / priority / stop / yield 证据；连续有效灯态必须 conflict review |
| `pedestrian_crossing` | PedestrianCrossing | `junction_pre_m=40`, `junction_post_m=40`; `pedestrian_not_rs` | 行人只进 EVENT；R4/R5 取决于 crossing 是否与路口控制源同源 |
| `highway_merge` | EnterActorFlow*, HighwayCutIn, HighwayExit, MergerIntoSlowTraffic* | `merge_pre_m=30-50`, `merge_post_m=40-50`, `trigger_close_m=90`; slow traffic 保持 R3 | merge/split/ramp/lane-count-change；cut-in/slow/EnterFlow 名称不能单独 high |
| `interurban` | InterurbanActorFlow | `merge_pre_m=50`, `merge_post_m=45`, `junction_pre_m=55`, `junction_post_m=25` | 先拆 R3 合流窗口，再拆 R4/R5 路口窗口；projection error 高时禁用 route_s hard boundary |
| `interurban_advanced` | InterurbanAdvancedActorFlow | `junction_pre_m=55`, `junction_post_m=25`, `r3_requires_topology=True` | 默认 R1/R4/R5；只有明确 merge/split/ramp 才给 R3 medium/review |
| `invading_turn` | InvadingTurn | `two_way_min_pre_m=80`, `two_way_post_pad_m=20`, `trigger_close_m=75` | 对向车侵入/heading conflict；EVENT 区分被动让行与自车借道 |
| `parking` / `parking_exit` | ParkingCrossingPedestrian, ParkingCutIn, ParkingExit | `parking_pre_m=20-35`, `parking_post_m=50-60` | parking/shoulder/curbside 或 RGB 停车空间；行人/切入仍归 EVENT |
| `static_cutin` | StaticCutIn | `parking_pre_m=35`, `parking_post_m=55`, `merge_pre_m=35`, `merge_post_m=55` | R3/R6 二选一主导；分差小于 0.15 必须 review |
| `vehicle_opens_door_twoways` | VehicleOpensDoorTwoWays | `two_way_min_pre_m=50`, `two_way_post_pad_m=20`, `parking_pre_m=35`, `parking_post_m=55` | R2/R6 可能共存，primary 看当前主导决策，secondary 记录另一项 |
| `vehicle_turning` | VehicleTurningRoute, VehicleTurningRoutePedestrian | `junction_pre_m=50`, `junction_post_m=20-40`, `multi_trigger=True` | 多 trigger 分段；行人/横穿不改变 RS，控制源决定 R4/R5 |
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
- 第一轮全帧 RGB 视觉复核已覆盖 `rs_full_frame_review` 下 43/43 个场景的
  `scenario_visual_review_summary.json`。复核结论已回灌到运行时门控：
  R2/R3/R6 缺可信 XODR 拓扑或可见占道/合流/停车空间证据时，不再仅凭 scenario/trigger
  窗口压过 R1，而是降为 secondary + review；R4/R5 缺 meta 灯态、junction hint
  或可信 XODR signal/junction 时，也不再给 high confidence 的窗口标签。
  后续若人工 RGB 证明某场景确实需要更长窗口，应通过逐场景 `--rule-config-json`
  调 pre/post 参数，而不是放宽全局逻辑门控。
- 第二轮图像优先全量复核已重跑到
  `collection_output/rs_full_frame_review_after_visual_gate/`：43 个 scenario、204 个
  scenario-town route、24387 帧，每个 town 1 条 route 全帧标注并查看对应 RGB overview。
  复核产物为各场景的 `scenario_visual_review_summary.json` 和
  `global_visual_review_summary.json`。本轮明确不以 confidence 作为正确性证明；
  confidence/review 只用于定位候选 span，最终异常以 RGB 逐帧可见结构为主，低能见度再参考 XODR/XML/meta。
  已记录 63 条高优先级发现，主要集中在两类：
  R2/R3/R6 缺可见占道/合流/停车空间时属于规则思路门控问题；
  R4/R5 从 frame0 开始且 route projection error 高时属于 pre-window / projection 参数问题。
- 第二轮错配回灌后的代码原则：
  弱特殊 RS 候选不能再通过全局 priority 低分压过 R1；
  `route_projection_error_m > 5m` 时，`scenario_active` 和普通 `trigger_close_m`
  只能写 review/evidence，不能单独撑起 two-way / merge / parking / junction 结构窗口；
  静态 XODR 的 signal/opposite/parking/merge/junction hint 在 route 投影高误差帧降级为
  `*_demoted_projection_error` 证据，不再作为 R2/R3/R6/R4/R5 high 证据。
  对 nonsignalized 场景，如果静态 XODR 仍提示 signal/controller，需要写
  `nonsignalized_with_signal_topology_conflict`，由人工结合 RGB 判定是真实地图信号、
  XODR 误匹配，还是场景命名与 road structure 口径冲突。
- 逐帧 RGB 复核的定点回灌原则：
  `Accident/Town03` 这类“XODR/static signal 近邻但画面没有清晰路口、stopline 或可见信号控制”的帧，
  不允许静态 signal + 距离字段单独把 R1 升成 R4；有效灯态若也缺强路口上下文，只给 weak R4 candidate
  并保持 R1 primary + review。
  `Accident/Town05` 本轮全帧 RGB 复核进一步发现：同向事故/拥堵路段里 `light_hazard`
  会在没有稳定可见受控路口时把 R1 误升为 R4，并造成 R1/R4 抖动。代码已收紧：
  `same_direction_obstacle` / `default_meta_map` 场景中，`light_hazard` 只有同时具备
  meta junction 或 stop hazard 这类强控制上下文时才允许升 R4；仅有静态 signal 近邻、
  弱 distance-to-junction 或普通 hazard 时保持 R1，把事故/施工/急刹等交给 EVENT。
  `AccidentTwoWays/Town01` 这类普通双向道路早段不能只因 scenario active / trigger window 给 R2；
  R2 high 必须额外满足近距离障碍、`*_two_ways_stuck`、`vehicle_hazard`、近距离
  `scenario_obstacles_ids` 或 `signed_dist_to_lane_change` 核心证据。若缺可信 XODR opposite lane，
  核心帧可用 meta obstruction 保留 R2，但必须带 `special_rs_lacks_full_topology_confirmation` review。
  `HighwayCutIn` / `EnterActorFlow*` / `MergerIntoSlowTraffic*` 和 `ParkingCutIn` /
  `StaticCutIn` / `VehicleOpensDoorTwoWays` 的全帧 RGB sheet 显示：很多帧只是高速直行、
  普通 cut-in 或停车侧事件，并没有可见 merge/split/parking-space road structure。
  缺 XODR/RGB topology confirmation 时，R3/R6/R2 现在只作为弱候选，分数低于稳定 R1，
  不再把 R1 置信压到低置信或触发大面积 `candidate_score_gap_lt_0.15`。
  代码层面这些判断已拆成 `strong_control_context` 与 `twoway_obstruction_evidence` 两个证据字段，
  逐帧 review 时优先看这两个字段，再决定是阈值问题还是道路结构口径问题。
- 2026-07-03 在用户指定目录 `collection_output/rs_full_frame_review/` 重新跑 43 个场景：
  每个 town 1 条 route，全帧 204 route / 24387 帧，生成 `scenario_visual_review_summary.json`
  和 `global_visual_review_summary.json`。异常桶以 `xml_projection_or_boundary_parameter`
  与 `arbitration_or_threshold_margin` 为主。代码已把稳定高置信标签上的
  XML/XODR 投影质量告警、候选分差和 weaker-special 这类审计提示从
  “视觉错配候选”中拆出：这些提示仍保留 route review reason，但不再塞进
  `candidate_anomalies.jsonl` / anomaly sheet；真正需要人工逐帧看的仍是标签切换、
  低置信、拓扑确认缺失和 RGB 可见语义冲突。
- 本轮修正规则遵循“逐场景定位、通病抽象复用”：先用每个 scenario/town 的 RGB sheet
  找到具体错配帧，再判断它是场景私有问题还是全局门控问题。`Accident/Town05`
  触发了 `light_hazard` 全局收紧；`HighwayCutIn` / `ParkingCutIn` /
  `StaticCutIn` 触发了 R3/R6 弱拓扑候选降权；多个 TwoWays 场景触发了 R2
  confirmation 复核保留；多个场景共同暴露“没有特殊结构确认时 R1 仍为 0.35 低置信”，
  因此代码新增 `r1_stable_no_special_structure_confirmed` 兜底：当没有任何特殊 RS
  达到有效候选阈值时，把 R1 视为稳定普通道路结构，而不是继续输出低置信。

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

### Accident

- 候选 RS：R1, R4。
- 已确定口径：事故障碍本身是 EVENT，不是 ROAD_STRUCTURE；默认 R1，进入真实信号灯路口才切 R4。
- 分段逻辑：finite `dist_to_accident` / active scenario 只用于标记事故事件窗口；窗口内仍按道路结构判断 R1/R4。
- 证据需求：XML accident trigger + meta distance 确定事件窗口；R4 必须有有效灯态且具备强路口/stopline/signal-junction 上下文，
  或可信 XODR/meta junction 与静态 signal 同源。仅有 static signal near / distance-to-junction 不足以把普通路段升 R4。
- 待完善点：不要把同向绕障误升为 R2；若 projection error 高，事故窗口只给事件候选，不改 RS。

### AccidentTwoWays

- 候选 RS：R1, R2, R4。
- 已确定口径：two-way 障碍窗口内，优先用可信 XODR 证明对向 lane 参与；若 XODR 不可信，只有近距离障碍、
  `accident_two_ways_stuck`、`vehicle_hazard` 或明显 lane-change 核心证据同时成立时才给 R2 high，并强制 review。
  灯控路口仍优先 R4，但有效灯态缺强路口上下文时只保留 weak R4 candidate。
- 分段逻辑：窗口外 R1；XML trigger / distance / active window 内检查 opposite driving lane、同向可用 lane 是否不足、障碍是否压占自车通行空间。
- 证据需求：XODR lane_id 符号反转、lane direction、lane count、meta active scenario、近距离障碍距离、
  `scenario_obstacles_ids`、`signed_dist_to_lane_change`、RGB 对向借道边界。
- 待完善点：实测发现只靠 scenario trigger 会把普通双向路早段误打 R2；R2 high 必须要求“同向 lane 不足或视觉/对象证明必须借对向”。

### BlockedIntersection

- 候选 RS：R1, R4。
- 已确定口径：blocked intersection 是路口内阻塞事件，不等于 R5；有正常灯控时 primary R4。
- 分段逻辑：接近/进入受控 junction 设 R4；离开 junction 后回 R1；阻塞对象作为 EVENT。
- 证据需求：XODR junction + signal/controller、meta traffic_light_state、XML trigger。
- 待完善点：若 meta `is_junction=false` 但 stopline / traffic light 有效，应按 stopline approach 保留 R4 并 review。

### ConstructionObstacle

- 候选 RS：R1, R4。
- 已确定口径：施工障碍是 EVENT；非 TwoWays 场景不应自动进入 R2。
- 分段逻辑：施工窗口内仍按普通同向道路 R1，真实灯控路口覆盖为 R4。
- 证据需求：XML construction trigger + meta distance 定位事件，XODR/灯态确认 R4。
- 待完善点：需要 RGB 边界帧确认施工物是否实际导致 lane closure；没有 two-way/opposite 证据时不升 R2。

### ConstructionObstacleTwoWays

- 候选 RS：R1, R2, R4。
- 已确定口径：two-way 施工障碍可进入 R2，但 high 需要 XODR 对向 lane + 同向 lane 不足。
- 分段逻辑：R4 优先；two-way construction 窗口内满足对向参与条件给 R2；其余 R1。
- 证据需求：XML distance/offset、XODR opposite lane、meta active scenario、RGB 施工占道。
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
- 分段逻辑：灯控路口 R4；two-way hazard 窗口 + opposite lane + 同向 lane 不足给 R2；否则 R1。
- 证据需求：XML trigger/distance、XODR lane direction、meta active scenario、RGB 对向/路边风险。
- 待完善点：侧向风险未压占自车路径时，保持 R1 + EVENT。

### HighwayCutIn

- 候选 RS：R1, R3, R4。
- 已确定口径：他车切入是 EVENT；只有道路本身处于高速/主辅路/匝道/合流拓扑时才 R3。
- 分段逻辑：merge/ramp/highway topology 成立时 R3；普通多车道同向切入仍 R1；灯控路口 R4。
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
- 证据需求：XML actor-flow + route、XODR merge/junction/signal、meta traffic_light_state/is_junction、RGB 边界帧。
- 待完善点：实测 projection error 多，边界必须优先使用 XODR/meta waypoint 吸附，不能只看 sparse route_s。

### InterurbanAdvancedActorFlow

- 候选 RS：R1, R4, R5，R3 仅 medium/review。
- 已确定口径：当前证据更支持路口/优先级场景，R3 只有出现明确 merge/split/ramp 时才成立。
- 分段逻辑：junction + light 给 R4；junction/priority/no-light 给 R5；明确合流拓扑才临时 R3；其余 R1。
- 证据需求：XODR junction/signal/priority、XML actor-flow、meta light/junction、RGB 路口和侧向车流。
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
- 分段逻辑：merge/actor-flow 窗口 + topology 给 R3；合流完成回 R1；灯控路口 R4。
- 证据需求：XML flow window、XODR merge/lane count、meta speed/active scenario、RGB 慢车流间隙。
- 待完善点：不要把低速误归 R6 或 R5；低速只是事件/动作状态。

### MergerIntoSlowTrafficV2

- 候选 RS：R1, R3, R4。
- 已确定口径：同 MergerIntoSlowTraffic；V2 只影响样本/route 形态，不改变可见 RS 语义。
- 分段逻辑：只在合流核心窗口内 R3；完成后 R1；灯控路口 R4。
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
- 分段逻辑：two-way obstruction 窗口 + opposite lane 参与给 R2；灯控路口 R4；其余 R1。
- 证据需求：XML distance/offset、XODR lane direction/count、meta active scenario、RGB 借道。
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
- 证据需求：XML turn route、XODR junction/signal/priority、meta light/junction、RGB 路口。
- 待完善点：多 trigger 场景要按每个路口窗口分段，不能整段一个 RS。

### VehicleTurningRoutePedestrian

- 候选 RS：R1, R4, R5。
- 已确定口径：行人是 EVENT；RS 仍由 turn route 所在控制源决定。
- 分段逻辑：有效灯控路口 R4；无灯/priority 路口 R5；普通路段 R1。
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
