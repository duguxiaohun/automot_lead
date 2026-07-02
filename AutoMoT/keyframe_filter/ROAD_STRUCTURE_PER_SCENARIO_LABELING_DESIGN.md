# ROAD_STRUCTURE 逐场景标定实现设计

本文只设计 `ROAD_STRUCTURE`，不做 EVENT 细分。目标是把
`AutoMoT/data/lead` 的 route XML、真实 LEAD `metas/*.pkl`、可选
`bboxes/*.pkl`、以及 `AutoMoT/CARLA_0915` 的 XODR 地图，组合成可落代码的
帧级 `primary_road_structure` 标注规则。

核心原则：

- `ROAD_STRUCTURE` 是当前驾驶决策规则空间，不是纯视觉外观，也不是事件类型。
- XML 的 `trigger_point` 是 scenario 机制锚点，不是事件可见起点，也不是结构标签的唯一依据。
- XODR 给“道路拓扑是否支持该结构”；meta 给“当前帧是否处在该结构/灯态/活跃窗口”；XML 给“该 run 的场景先验和地理窗口”。
- 每帧输出一个训练主标签 `primary_road_structure`，可附带少量 `secondary_road_structures` 用于审计。

---

## 0. 可用输入与实现边界

### 0.1 XML

XML 根目录：

```text
AutoMoT/data/lead/<Scenario>/*.xml
```

本地 XML 覆盖 43 类 scenario。文件命名不完全统一，必须建多键索引：

```text
by_route_num: route_001783 / 001783
by_route_id_raw: Town01_Scenario7_0 / Town01_route_Town01_Scenario7_0
by_town_and_id: (Town, id)
```

XML 统计显示很多 scenario 的 `waypoints` 极短：

- 路口/停车/动态横穿类经常只有 2-4 个 waypoint。
- Accident / Construction / ParkedObstacle 这类长路线可到 80+ waypoint。
- `VehicleTurningRoute` 一个 XML 里可含多个 `<scenario>`，不能假设 route 只有一个 trigger。

所以 XML waypoint 只能做 route 粗投影；正式边界要优先用 meta 位姿 + XODR 吸附 + trigger 窗口。

### 0.2 XML 缺失清单

`AutoMoT/data/lead/cache_lead_recheck_summary.json` 记录真实数据有但 XML 缺失的 run：

- `ConstructionObstacleTwoWays`: 4 条，Town12，route id `001439/001485/001542/001554`。
- `noScenarios`: 36 条，Town04/Town05/Town06/Town07/Town10HD。

这些 run 走 `meta + XODR` 降级规则：

```text
xml_available=false
xml_missing_reason=listed_in_cache_lead_recheck_summary
confidence_level<=medium
review_required=true for R2/R3/R5/R6 and boundary-sensitive frames
```

### 0.3 XODR

XODR 搜索路径：

```text
AutoMoT/CARLA_0915/CarlaUE4/Content/Carla/Maps/OpenDrive/<Town>.xodr
AutoMoT/CARLA_0915/CarlaUE4/Content/Carla/Maps/<Town>/OpenDrive/<Town>.xodr
AutoMoT/CARLA_0915/AdditionalMaps_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/<Town>.xodr
AutoMoT/CARLA_0915/AdditionalMaps_0.9.15/CarlaUE4/Content/Carla/Maps/<Town>/OpenDrive/<Town>.xodr
```

XODR 用途：

- `carla.Map.get_waypoint(location)`：当前位置 road/lane/junction。
- 相邻 lane 遍历：同向车道数、是否有相邻反向 driving lane。
- `Junction.get_waypoints(Driving)`：路口出入口对，用于 T 字/十字/匝道节点粗分。
- 直接解析 `<signal>` / `<controller>`：信号灯/Stop Sign/受控路口索引。
- route densify：从 XML 稀疏 waypoint 过渡到 1-2m 路网点。

XODR 不提供实时灯色。实时灯色只用 meta `traffic_light_state`。

### 0.4 Meta

每帧从 `metas/*.pkl` 抽：

```text
ego: pos_global, theta, ego_matrix, speed
route/topology: is_junction, is_intersection, junction_id,
                dist_to_junction, distance_to_next_junction,
                lane_id, lane_type_str, lane_change_str, ego_lane_width
signal: traffic_light_state, light_hazard, stop_sign_close, stop_sign_hazard
scenario: current_active_scenario_type
distance: dist_to_accident_site, dist_to_construction_site,
          dist_to_parked_obstacle, dist_to_vehicle_opens_door,
          dist_to_cutin_vehicle, dist_to_pedestrian, dist_to_biker
hazard: vehicle_hazard, walker_hazard
```

meta 的 `dist_to_*` 主要服务 EVENT 可见性；对 RS 只能加置信，不能单独决定结构。

---

## 1. 统一数据结构

建议先把每帧整理成 `FrameRSFeatures`：

```text
scenario, run_id, frame_id
town, xml_available, xodr_available
ego_xy, yaw, speed
route_s, route_projection_error_m
trigger_s_list, nearest_trigger_s, trigger_distance_xy
xml_tags: direction, distance, frequency, speed,
          start_actor_flow, end_actor_flow, other_actor_location,
          front_vehicle_distance, behind_vehicle_distance,
          source_dist_interval, flow_speed, traffic_direction,
          crossing_angle, offset, blocker_model
meta_flags: traffic_light_state, light_hazard, stop_sign_hazard,
            is_junction, junction_id, dist_to_junction,
            current_active_scenario_type
meta_distances: all finite dist_to_* fields
xodr: map_road_id, map_lane_id, map_lane_type, map_is_junction,
      map_junction_id, lane_count_same_dir, has_opposite_driving_lane,
      has_parking_or_shoulder_nearby, has_signal_controller_nearby,
      has_stop_or_yield_nearby, ramp_merge_split_hint,
      route_heading_change_deg, route_lane_count_change
```

每帧候选分数结构：

```text
candidates = {
  R1: {score, rules, confidence_hint},
  R2: {score, rules, confidence_hint},
  R3: {score, rules, confidence_hint},
  R4: {score, rules, confidence_hint},
  R5: {score, rules, confidence_hint},
  R6: {score, rules, confidence_hint},
}
```

仲裁优先级：

```text
R4/R5 > R3 > R2/R6 > R1
```

例外：

- `CrossJunctionDefectTrafficLight`：R5 覆盖 R4，写 `defect_signal_overrides_R4`。
- `VehicleOpensDoorTwoWays`：R2/R6 都可能成立，按“是否必须占用/等待对向车道”决定 primary。
- `noScenarios`：没有 scenario 先验时，禁止单靠 XODR hint 升级 R2/R3/R5/R6；只允许高置信 R4，否则 R1。

---

## 2. 全局原子判据

### 2.1 R4 信号灯路口

强证据任一成立：

- `traffic_light_state in {Red, Yellow, Green}` 且不是 `None/Off/Unknown`。
- `light_hazard=True`。
- scenario 属于明确信号灯类，并且 trigger/junction 同源，XODR 有 signal controller。

同源校验：

- 当前帧投影到 route 后，前方/当前位置 junction 与 XML trigger 最近 junction 相同，或距离小于 50-60m。
- 如果 `is_junction=False` 但处于红灯 stopline 前等待，仍保持 R4；不要因尚未进入 junction polygon 提前退 R1。

降级：

- 只有 XODR signal controller、没有有效灯态、且 scenario 不是明确信号灯类时，只给 medium R4 或 review。
- nonsignalized scenario 中短暂出现有效灯态：可标 R4 medium，但必须 `review_required=true`，除非连续灯态和 route 同源都成立。

### 2.2 R5 无信号/失效/路权路口

强证据：

- scenario 属于无灯/路权/灯故障类。
- 当前帧在 trigger 对应 junction 接近、等待、穿越窗口。
- 无有效正常灯态，或 `CrossJunctionDefectTrafficLight` 明确声明灯故障。

XODR 辅助：

- junction 附近无 signal controller。
- 有 stop/yield/sign type hint。
- junction lane pairs 支持 T 字/十字路口。

否决：

- 有连续有效正常灯态且不是 defect scenario 时，不给 high R5；改 R4 medium/review。
- 普通弯道、普通岔路、环岛，不因为 heading 变化打 R5。

### 2.3 R3 高速/匝道/合流/驶出

强证据：

- scenario 属于 `{EnterActorFlow, EnterActorFlowV2, HighwayCutIn, HighwayExit, MergerIntoSlowTraffic, MergerIntoSlowTrafficV2, InterurbanActorFlow}`。
- XML `start_actor_flow/end_actor_flow/other_actor_location` 与 trigger 构成的窗口命中。
- XODR 显示 ramp/merge/split、lane count 变化、主辅路/匝道邻接，或 `is_junction=True` 但无 signal controller 且形态是合流节点。

否决：

- 仅有 XML 字段名 `start_actor_flow` 不够；自行车横穿、路口横向流的 actor flow 不能打 R3。
- 进入真实信号灯/无灯路口决策区后，R4/R5 覆盖 R3。

### 2.4 R2 双向单车道/对向参与

强证据：

- scenario 属于 TwoWays 或 InvadingTurn 类。
- XODR 相邻 driving lane 存在 lane_id 符号反转，且同向车道数不足以在同向空间绕行。
- 当前帧在 trigger/active/距离窗口内。

中证据：

- XML 明确 TwoWays/InvadingTurn，但 XODR 不可用；scenario+trigger 给 medium R2。
- `current_active_scenario_type` 命中，且对应 `dist_to_*` finite/接近。

否决：

- TwoWays 不是全程 R2；窗口外回 R1/R4。
- 灯态/路口主导时 primary=R4/R5，R2 只能 secondary。
- `HazardAtSideLane` 没有 TwoWays 后缀，不自动 R2。

### 2.5 R6 路边停车/停车占道

强证据：

- scenario 属于 Parking*、ParkingExit、StaticCutIn 的停车子型，或 VehicleOpensDoorTwoWays 的停车风险片段。
- XODR 当前/相邻 lane 为 Parking/Shoulder，或 route 旁侧存在停车 lane/curbside。
- bbox 中旁侧静态车辆密集，且横向分布在 parking/shoulder/curbside 侧。
- 当前帧在 trigger/active/停车汇入窗口内。

否决：

- `ParkedObstacle` / `ParkedObstacleTwoWays` 不是 R6；前者 R1，后者 R2。
- 只有 `dist_to_parked_obstacle` 或 `dist_to_cutin_vehicle` 不足以 high R6。
- 灯控路口主导时 primary=R4，R6 secondary。

### 2.6 R1 默认

所有特殊候选未确认，或特殊候选只剩低置信 hint 时，回 R1。

强制 R1：

- ControlLoss / HardBreakRoute 的急刹、低速、失控现象。
- DynamicObjectCrossing 的普通道路横穿。
- noScenarios 无有效灯态。
- TwoWays/Parking/R3 窗口结束后的普通跟车/直行段。

---

## 3. 通用窗口函数

代码里不要把阈值散落在 43 个 policy 中，统一写 helper：

```text
window_from_trigger(route_s, trigger_s, pre_m, post_m)
window_from_xml_distance(route_s, trigger_s, distance, min_pre, min_post)
window_from_actor_flow(route_s, trigger_s, start_actor_flow_s, end_actor_flow_s, pre_m, post_m)
window_from_junction(route_s, nearest_junction_s, approach_m, exit_m)
window_from_heading_change(route_s, turn_start_s, turn_end_s, pre_m, post_m)
```

初始阈值：

- 灯控/无灯路口：trigger/junction 前 50-60m，离开后 20-25m。
- TwoWays 绕障：`trigger_s ± max(xml.distance, 50m)`，后向保持 20m。
- R3 合流/驶出：actor-flow min/max 前 40-50m，后 40-50m。
- ParkingCutIn：trigger 前 30m 后 50m。
- ParkingExit：trigger 前 20m 后 60m。
- Pedestrian/Dynamic crossing：trigger 前后 40m，但只影响 R4/R5/R1，不单独定义特殊 RS。

所有窗口都必须带 hysteresis：

- 进入：强证据单帧或中证据连续 2 帧。
- 保持：证据消失后 3-5 帧。
- 最短片段：少于 4 帧的 R2/R3/R5/R6 合并到邻居；R4 有有效灯态时不合并。
- 切换前后 3 帧标 `transition_margin=true`。

---

## 4. 逐场景规则

每节格式：

```text
XML 事实：本地 XML 静态统计。
目标 RS：允许的 primary 候选。
主规则：如何用 XML/XODR/meta 判。
否决/降级：避免误标。
输出 evidence：rules_fired 建议。
```

### 4.1 Accident

XML 事实：187 条；Town03/04/05/06/10HD/12/13；waypoints 2-82，均值 11.1；tags=`trigger_point/direction/distance/speed`。

目标 RS：R1/R4。

主规则：

- R4：有效 `traffic_light_state`，或 route 当前进入同源受控路口窗口。
- R1：其它全部。同向事故障碍不改变道路结构。
- XML `distance/direction/speed` 只写入 obstacle evidence，不提升 R2/R6。

否决/降级：

- 禁止因为 `dist_to_accident_site` 打 R2；Accident 不是 TwoWays。
- XODR 有 parking lane 也不打 R6。

输出 evidence：`r1_same_direction_accident_context`、`r4_tl_confirmed`、`obstacle_event_ignored_for_rs`。

### 4.2 AccidentTwoWays

XML 事实：542 条；Town01/02/05/07/12/13/15；waypoints 2-68，均值 3.5；tags=`trigger_point/frequency/distance/front_vehicle_distance/behind_vehicle_distance/direction`。

目标 RS：R1/R2/R4。

主规则：

- R4 优先：有效灯态或受控路口同源窗口。
- R2：TwoWays scenario + trigger/distance 窗口 + XODR opposite driving lane + 同向车道不足。
- XODR 缺失时：scenario+trigger/active 可给 medium R2。
- R1：trigger 窗口外、障碍已结束、普通接近/离开段。

窗口：

- `trigger_s ± max(xml.distance, 45m)`，后向 pad 20m。
- route_s 缺失时用 trigger 欧氏距离 `<70m` 弱召回。

否决/降级：

- TwoWays 不是全程 R2。
- R4 与 R2 冲突时 primary=R4，secondary=[R2]。

输出 evidence：`r2_twoways_opposite_lane_confirmed`、`r2_twoways_xml_only_medium`、`r4_overrides_r2`。

### 4.3 BlockedIntersection

XML 事实：155 条；Town06/07/12/13；waypoints 2-5，均值 2.2；tags 主要 `trigger_point`，且一个 route 可含多个 blocked trigger。

目标 RS：R1/R4。

主规则：

- R4：trigger 对应受控 junction，或有效灯态/`light_hazard`。
- R1：路口外跟车、接近和离开背景。

否决/降级：

- 前方阻塞是事件 U-E8，不是 R5。
- 如果 trigger 附近无 controller 且无灯态，输出 R1/R4 medium + review，不自动 R5。

输出 evidence：`r4_blocked_signalized_intersection`、`blocked_event_ignored_for_rs`。

### 4.4 ConstructionObstacle

XML 事实：187 条；Town03/04/05/06/10HD/12/13；waypoints 2-82，均值 10.9；tags=`trigger_point/direction/distance/speed`。

目标 RS：R1/R4。

主规则同 Accident；`dist_to_construction_site` 只用于事件可见性。

否决/降级：

- 禁止把同向施工障碍打 R2。
- 禁止因施工/低速打 R6。

输出 evidence：`r1_same_direction_construction_context`、`r4_tl_confirmed`。

### 4.5 ConstructionObstacleTwoWays

XML 事实：521 条；Town01/02/05/07/12/13/15；waypoints 2-68，均值 4.0；tags=`trigger_point/frequency/distance/front_vehicle_distance/behind_vehicle_distance/direction`。

目标 RS：R1/R2/R4。

主规则同 AccidentTwoWays；距离字段换成 `dist_to_construction_site`。

缺 XML 特例：

- 4 条 Town12 run 没 XML。
- 用 `current_active_scenario_type`、`dist_to_construction_site`、XODR opposite lane 判 medium R2。
- 灯态仍可 high R4。

否决/降级：

- XML 缺失时 R2 最高 medium，边界帧 review。

输出 evidence：`r2_construction_twoways_window`、`xml_missing_r2_medium`。

### 4.6 ControlLoss

XML 事实：309 条；多 town；waypoints 2-4，均值 2.1；tags 基本只有 `trigger_point`。

目标 RS：R1/R4。

主规则：

- R4：有效灯态或同源受控路口。
- R1：其它。

否决/降级：

- 不用 brake/accel/speed/vehicle_hazard 判 RS。
- 不因失控、低速、停车输出 R2/R3/R6。

输出 evidence：`control_loss_behavior_ignored_for_rs`。

### 4.7 CrossingBicycleFlow

XML 事实：49 条；Town12；waypoints 2-4，均值 2.8；tags=`trigger_point/start_actor_flow/flow_speed/source_dist_interval`。

目标 RS：R1/R4。

主规则：

- R4：trigger 附近为受控路口，或 meta 有效灯态。
- R1：自行车横穿发生在普通路段或路口外时。

否决/降级：

- `start_actor_flow` 是自行车横穿流，不是匝道合流，禁止 R3。
- `dist_to_biker` 只影响事件，不改 RS。

输出 evidence：`bicycle_flow_not_r3`、`r4_crossing_bicycle_signal_window`。

### 4.8 CrossJunctionDefectTrafficLight

XML 事实：142 条；Town03/04/05/07/10HD/12/13/15；waypoints 固定 2；tags=`trigger_point/flow_speed/traffic_direction/source_dist_interval`。

目标 RS：R1/R5。

主规则：

- R5：trigger 对应 junction 前 60m、junction 内、离开后 20m。
- XODR 有 signal/controller 反而增强“灯故障路口”证据。
- 有效灯态不改 R4，只记录 defective signal context。

否决/降级：

- 本场景全局 R4 优先级失效；必须用 `defect_signal_overrides_R4`。
- 若 XODR 也找不到 junction，R5 medium + review。

输出 evidence：`defect_signal_overrides_R4`、`r5_defect_junction_window`。

### 4.9 DynamicObjectCrossing

XML 事实：308 条；多 town；waypoints 2-4，均值 2.2；tags=`trigger_point/distance/blocker_model/crossing_angle/direction`。

目标 RS：R1/R4。

主规则：

- R4：真实灯态或受控路口窗口。
- R1：普通路段动态对象横穿。

否决/降级：

- `crossing_angle/blocker_model` 只描述事件，不打 R5/R6。
- `vehicle_hazard/walker_hazard` 不参与 RS。

输出 evidence：`dynamic_crossing_event_ignored_for_rs`。

### 4.10 EnterActorFlow

XML 事实：88 条；Town12/13；waypoints 2-11，均值 4.2；tags=`trigger_point/start_actor_flow/end_actor_flow/flow_speed/source_dist_interval`。

目标 RS：R1/R3/R4。

主规则：

- R3：`trigger/start_actor_flow/end_actor_flow` 投影 min/max，窗口 `[min_s-30m, max_s+40m]`，并由 XODR ramp/merge/lane-count-change 补强。
- R4：若出现有效灯态且同源受控路口明确，R4 优先。
- R1：完成合流、lane topology 稳定后。

否决/降级：

- `is_junction=True` 在这里默认可能是 merge node，不能打 R5。
- 没有 actor-flow/merge/topology 证据时回 R1。

输出 evidence：`r3_enter_actor_flow_window`、`merge_node_not_r5`。

### 4.11 EnterActorFlowV2

XML 事实：43 条；Town12；waypoints 固定 4；tags 同 EnterActorFlow。

目标 RS：R1/R3/R4。

主规则：

- 与 EnterActorFlow 同口径。
- route 很短时优先使用 trigger/start/end actor-flow 三点，不用稀疏 route 全程覆盖。

否决/降级：

- V2 只保留 raw metadata，不单独改变 RS 类别。
- 离开三点窗口后必须回 R1。

输出 evidence：`r3_enter_actor_flow_v2_window`。

### 4.12 HardBreakRoute

XML 事实：97 条；Town12/13；waypoints 2-4，均值 2.4；tags 主要 `trigger_point`。

目标 RS：R1/R4。

主规则：

- R4：实际经过信号灯路口。
- R1：其它。

否决/降级：

- brake/accel/vehicle_hazard 不决定 RS。
- 急刹不是 R5/R6。

输出 evidence：`hard_brake_event_ignored_for_rs`。

### 4.13 HazardAtSideLane

XML 事实：97 条；Town12/13；waypoints 2-3，均值 2.5；tags=`trigger_point/distance/bicycle_drive_distance/bicycle_speed/speed`。

目标 RS：R1/R4。

主规则：

- R1：同向或侧向危险道路，默认不涉及对向借道。
- R4：真实灯态/受控路口窗口。

否决/降级：

- 没有 TwoWays 后缀，不开放 R2。
- `dist_to_biker/pedestrian` 或 XODR opposite hint 只能 review，不自动改 R2。

输出 evidence：`side_lane_not_twoways`。

### 4.14 HazardAtSideLaneTwoWays

XML 事实：96 条；Town12/13；waypoints 2-4，均值 2.4；tags=`trigger_point/frequency/distance/bicycle_speed/bicycle_drive_distance`。

目标 RS：R1/R2/R4。

主规则：

- R4：灯态/受控路口优先。
- R2：trigger/distance 窗口 + XODR opposite lane/narrow road。
- XODR 缺失时 scenario+trigger medium R2。

否决/降级：

- 自行车/侧向危险是事件证据；R2 来自 TwoWays 结构。
- 若 XODR 显示多同向车道足够绕行，R2 降 medium/review。

输出 evidence：`r2_side_lane_twoways_opposite_lane`。

### 4.15 HighwayCutIn

XML 事实：93 条；Town12/13；waypoints 2-9，均值 4.2；tags=`trigger_point/other_actor_location`。

目标 RS：R1/R3/R4。

主规则：

- R3：trigger 与 `other_actor_location` 构造切入窗口，route_s 覆盖两点 min/max 后前后 pad 40m。
- XODR 的 ramp/merge/auxiliary lane/lane-change 入口给 high R3。
- R4：进入城市信号路口且灯态同源时覆盖 R3。

否决/降级：

- 他车切入本身是事件；只有发生在高速/匝道拓扑窗口内才 R3。
- 如果 XODR 是普通城市路且无 merge/ramp，只给 R1/R4 + review。

输出 evidence：`r3_highway_cutin_topology`、`cutin_event_not_alone_rs`。

### 4.16 HighwayExit

XML 事实：94 条；Town12/13；waypoints 3-28，均值 5.6；tags=`trigger_point/start_actor_flow/end_actor_flow/flow_speed/source_dist_interval`。

目标 RS：R1/R3/R4。

主规则：

- R3：驶出/分流窗口，actor-flow 三点 min/max 前 50m 后 50m；若 XODR 有 lane split/exit lane，并入 R3。
- R4：出口后接信号路口且灯态同源时切 R4。
- R1：窗口外。

否决/降级：

- 出口目标变道仍是 R3，不是普通 R1 变道。
- 出口完成且 lane topology 稳定后回 R1。

输出 evidence：`r3_highway_exit_split_window`。

### 4.17 InterurbanActorFlow

XML 事实：91 条；Town12/13；waypoints 2-5，均值 3.3；tags=`trigger_point/start_actor_flow/end_actor_flow/flow_speed/source_dist_interval`。

目标 RS：R1/R3/R4/R5。

主规则：

- 前段 R3：actor-flow 三点 + XODR lane-change/merge/主辅路证据。
- 后段 R4/R5：route 末端或 trigger junction 进入路口决策区；有效灯态/信号控制为 R4，无灯/路权为 R5。
- R1：两段之外。

冲突仲裁：

- 已在 junction/stopline 决策区：R4/R5 优先。
- 仍在合流/变道前段：R3 优先。

输出 evidence：`interurban_r3_to_junction_sequence`、`junction_overrides_r3`。

### 4.18 InterurbanAdvancedActorFlow

XML 事实：76 条；Town12/13；waypoints 2-4，均值 2.5；tags 同 InterurbanActorFlow。

目标 RS：R1/R4/R5，必要时短 R3 medium/review。

主规则：

- 主体按路口通行：有效灯态/受控 junction 为 R4，无灯/路权窗口为 R5。
- actor-flow 只作为冲突车流证据；只有 XODR 明确存在 merge/split 才给 R3 medium。

否决/降级：

- 不因 XML 有 start/end_actor_flow 自动 R3。
- R3 与 R4/R5 冲突时路口优先。

输出 evidence：`advanced_actor_flow_junction_primary`、`r3_only_with_merge_topology`。

### 4.19 InvadingTurn

XML 事实：100 条；Town12/13；waypoints 2-3，均值 2.4；tags=`trigger_point/offset/distance`。

目标 RS：R1/R2/R4。

主规则：

- R2：trigger/distance/offset 影响区 + XODR opposite lane/narrow road。
- R4：灯态有效时优先。
- R1：窗口外。

否决/降级：

- 这是被动等待对向车侵占，不是自车主动借道；RS 同为 R2，但 evidence 必须区分。
- 若路网不支持对向交会，R2 review。

输出 evidence：`r2_passive_invading_turn`、`offset_conflict_window`。

### 4.20 MergerIntoSlowTraffic

XML 事实：94 条；Town12/13；waypoints 2-28，均值 5.4；tags=`trigger_point/start_actor_flow/end_actor_flow/flow_speed/source_dist_interval`。

目标 RS：R1/R3/R4。

主规则：

- R3：actor-flow + ramp/merge topology 窗口。
- 慢速跟车仍在合流窗口内时保持 R3。
- R4：真实灯态同源时优先。
- R1：合流完成、lane topology 稳定超过 10 帧后。

否决/降级：

- 不能因速度低退出 R3。

输出 evidence：`r3_merger_slow_traffic_window`。

### 4.21 MergerIntoSlowTrafficV2

XML 事实：103 条；Town06/12/13；waypoints 3-27，均值 6.4；tags 同 MergerIntoSlowTraffic。

目标 RS：R1/R3/R4。

主规则同 MergerIntoSlowTraffic。

实现注意：

- Town06 用常规 OpenDrive 路径。
- Town12/13 用大地图路径。
- V2 后缀只保留 raw metadata，不改变 RS 口径。

输出 evidence：`r3_merger_v2_same_policy`。

### 4.22 NonSignalizedJunctionLeftTurn

XML 事实：209 条；Town03/04/05/07/10HD/12/13；waypoints 2-4，均值 2.5；tags=`trigger_point/flow_speed/source_dist_interval`。

目标 RS：R1/R5。

主规则：

- R5：trigger 对应 junction 前 50m 到离开后 20m，无有效正常灯态。
- XODR 无 signal controller 或 stop/yield/priority hint 给 high R5。
- R1：路口外。

否决/降级：

- 有效灯态同源时 R4 medium + review，不直接 high R5。
- `flow_speed/source_dist_interval` 是横向流，不是 R3。

输出 evidence：`r5_nonsignalized_left_turn`、`nonsig_with_tl_conflict_review`。

### 4.23 NonSignalizedJunctionLeftTurnEnterFlow

XML 事实：199 条；Town03/04/05/07/10HD/12/13；waypoints 2-4，均值 2.5；tags 同无灯左转。

目标 RS：R1/R5。

主规则：

- 与 NonSignalizedJunctionLeftTurn 同口径。
- `EnterFlow` 只扩大等待/横向流证据窗口到 trigger 前 60m，不改 R3。

否决/降级：

- 字段名 EnterFlow 不允许打 R3。

输出 evidence：`r5_nonsig_left_turn_enter_flow_not_r3`。

### 4.24 NonSignalizedJunctionRightTurn

XML 事实：94 条；Town12/13；waypoints 2-4，均值 2.4；tags=`trigger_point/flow_speed/source_dist_interval`。

目标 RS：R1/R5。

主规则：

- R5：trigger 前 45m 到通过 junction 后 20m，无灯/路权右转。
- R1：路口外。

否决/降级：

- 连续有效灯态时输出 R4 medium + review。
- 普通右弯不在 junction 时仍 R1。

输出 evidence：`r5_nonsignalized_right_turn`。

### 4.25 noScenarios

XML 事实：1381 条；Town03/04/05/06/07/10HD/15；waypoints 2-9，均值 2.3；无 scenario tags。

目标 RS：R1/R4。

主规则：

- R4：只有真实有效灯态 + XODR 受控路口同源，或 `light_hazard` 强证据。
- R1：其它全部。

否决/降级：

- 没有 scenario 先验时，XODR opposite/parking/merge 只写 hint，不输出 R2/R3/R5/R6。
- 缺 XML 的 36 条最高 medium，边界 review。

输出 evidence：`noscenario_conservative_r1`、`noscenario_tl_only_r4`。

### 4.26 OppositeVehicleRunningRedLight

XML 事实：294 条；多 town；waypoints 2-9，均值 2.3；tags=`trigger_point/direction`。

目标 RS：R1/R4。

主规则：

- R4：信号灯正常可用的路口窗口，对方违规是事件。
- R1：路口外。

否决/降级：

- 不因为对向车闯红灯把 R4 改 R5；规则并未失效。

输出 evidence：`r4_opposite_vehicle_red_light_violation_context`。

### 4.27 OppositeVehicleTakingPriority

XML 事实：97 条；Town12/13；waypoints 2-4，均值 2.5；tags=`trigger_point/direction`。

目标 RS：R1/R5。

主规则：

- R5：无灯/路权路口窗口，需要等待对向优先车。
- R1：路口外。

否决/降级：

- 有效灯态同源时 review；默认不开放 high R4。
- 对向车优先不是违规，不属于 R4 违规语义。

输出 evidence：`r5_opposite_priority_junction`。

### 4.28 ParkedObstacle

XML 事实：195 条；Town03/04/05/06/10HD/12/13；waypoints 2-124，均值 11.7；tags=`trigger_point/direction/distance/speed`。

目标 RS：R1/R4。

主规则：

- R1：同向停放障碍绕行。
- R4：实际灯态/受控路口窗口。

否决/降级：

- `ParkedObstacle` 不是 Parking*，不打 R6。
- `dist_to_parked_obstacle` 是事件可见性，不改变 RS。

输出 evidence：`parked_obstacle_not_parking_rs`。

### 4.29 ParkedObstacleTwoWays

XML 事实：99 条；Town12/13；waypoints 2-3，均值 2.4；tags=`trigger_point/frequency/distance`。

目标 RS：R1/R2/R4。

主规则：

- R2：TwoWays trigger/distance 窗口 + opposite lane/narrow road。
- R4：有效灯态优先。
- R1：窗口外。

否决/降级：

- 停放障碍不等于 R6；核心是借对向。

输出 evidence：`r2_parked_obstacle_twoways`、`parked_not_r6`。

### 4.30 ParkingCrossingPedestrian

XML 事实：98 条；Town12/13；waypoints 2-3，均值 2.3；tags=`trigger_point/distance/direction/crossing_angle`。

目标 RS：R1/R4/R6。

主规则：

- R4：灯态有效或受控路口窗口，primary=R4。
- R6：trigger/distance 窗口内，且 XODR parking/shoulder 或 bbox 路边停车证据成立。
- R1：窗口外。

否决/降级：

- 行人横穿是事件，不决定 RS。
- R4/R6 同时成立时 primary=R4，secondary=[R6]。

输出 evidence：`r6_parking_pedestrian_context`、`r4_overrides_r6`。

### 4.31 ParkingCutIn

XML 事实：99 条；Town12/13；waypoints 2-3，均值 2.4；tags=`trigger_point/direction`。

目标 RS：R1/R4/R6。

主规则：

- R6：trigger 前 30m 后 50m，direction 指示停车侧；XODR parking/shoulder/curbside 或 bbox 静态车列给 high。
- R4：灯控路口优先。
- R1：窗口外。

否决/降级：

- 没有任何停车空间证据时，scenario+trigger 只能 R6 medium + review。
- `dist_to_cutin_vehicle` 是事件切入，不单独决定 RS。

输出 evidence：`r6_parking_cutin_space_confirmed`。

### 4.32 ParkingExit

XML 事实：248 条；Town03/10HD/12/13/15；waypoints 2-68，均值 3.3；tags=`trigger_point/front_vehicle_distance/behind_vehicle_distance/direction`。

目标 RS：R1/R6/R4。

主规则：

- R6：停车位/停车带汇入主路窗口，trigger 前 20m 后 60m；`front/behind_vehicle_distance` 定义停车空隙，direction 定位侧向。
- XODR lane_type Parking/Shoulder -> Driving 切换前后 high R6。
- R4：若有灯态，primary=R4，secondary=[R6]。
- R1：完成汇入，Driving lane 稳定且离 trigger >60m。

否决/降级：

- ParkingExit 是目标导向汇入，不是同向静态障碍。

输出 evidence：`r6_parking_exit_merge_to_driving`。

### 4.33 PedestrianCrossing

XML 事实：100 条；Town12/13；waypoints 2-3，均值 2.2；tags 主要 `trigger_point`。

目标 RS：R1/R4/R5。

主规则：

- R4：trigger 或当前帧在 signalized junction，或有效灯态。
- R5：无有效灯态，但 trigger 位于无灯/stop/yield junction。
- R1：普通路段行人横穿。

否决/降级：

- 行人本身不决定 R5；没有 junction 证据时保持 R1。

输出 evidence：`pedestrian_crossing_space_r1_r4_r5`。

### 4.34 PriorityAtJunction

XML 事实：99 条；Town12/13；waypoints 2-3，均值 2.2；tags 主要 `trigger_point`。

目标 RS：R1/R5。

主规则：

- R5：路权类 junction 窗口，trigger 前 50m 后 20m。
- XODR 无 signal controller 或 stop/yield/priority hint 给 high。
- R1：路口外。

否决/降级：

- 有效灯态连续出现时 R4 medium + review，不能 high R5。

输出 evidence：`r5_priority_junction`。

### 4.35 RedLightWithoutLeadVehicle

XML 事实：359 条；多 town；waypoints 2-4，均值 2.1；tags=`trigger_point/flow_speed/source_dist_interval`。

目标 RS：R1/R4。

主规则：

- R4：trigger stopline/junction 前 60m 到离开后 20m；有效灯态强证据。
- `is_junction=False` 但处于红灯等待区时仍 R4。
- R1：路口外。

否决/降级：

- 没有灯态但 scenario 强先验 + XODR signalized 时 R4 medium。
- 两者都缺则 review，不强 high R4。

输出 evidence：`r4_red_light_without_lead_vehicle`。

### 4.36 SignalizedJunctionLeftTurn

XML 事实：345 条；多 town；waypoints 2-4，均值 2.5；tags=`trigger_point/flow_speed/source_dist_interval`。

目标 RS：R1/R4。

主规则：

- R4：trigger 前 60m 到转弯后 25m；有效灯态或 XODR signalized left-turn path。
- route heading 变化可确认左转结束。
- R1：路口外。

否决/降级：

- 不因等待对向车或 gap 接受把 R4 改 R5。

输出 evidence：`r4_signalized_left_turn`。

### 4.37 SignalizedJunctionLeftTurnEnterFlow

XML 事实：212 条；多 town；waypoints 2-4，均值 2.6；tags=`trigger_point/flow_speed/source_dist_interval`。

目标 RS：R1/R4。

主规则同 SignalizedJunctionLeftTurn；`EnterFlow` 只是冲突流/等待证据，不改 R3/R5。

否决/降级：

- 字段名 EnterFlow 不打 R3。

输出 evidence：`r4_signalized_left_turn_enter_flow_not_r3`。

### 4.38 SignalizedJunctionRightTurn

XML 事实：336 条；多 town；waypoints 2-4，均值 2.2；tags=`trigger_point/flow_speed/source_dist_interval`。

目标 RS：R1/R4。

主规则：

- R4：trigger 前 50m 到右转 heading 稳定后 20m。
- 有效灯态强 R4；右转专用 channel 即使 `map_is_junction=False`，只要灯态/controller 同源仍保持 R4。
- R1：路口外。

否决/降级：

- 右转 channel 的 junction polygon 边界不能让 R4 提前退出。

输出 evidence：`r4_signalized_right_turn`。

### 4.39 StaticCutIn

XML 事实：99 条；Town12/13；waypoints 2-4，均值 2.5；tags=`trigger_point/distance/direction/speed`。

目标 RS：R1/R3/R4/R6。

主规则：

- R4：有效灯态/受控路口优先。
- R6：cut-in 侧为 parking/shoulder/curbside，或 bbox 显示路边停车车列。
- R3：cut-in 侧为 ramp/merge/auxiliary lane，且 XODR 有合流/主辅路证据。
- R1：普通同向 lane change/cut-in。

否决/降级：

- R3 与 R6 不能同时 high；若地图证据冲突，primary 用 R1/R4，R3/R6 review。
- `dist_to_cutin_vehicle` 只加对应窗口置信。

输出 evidence：`static_cutin_r6_parking_side`、`static_cutin_r3_merge_side`、`static_cutin_r1_same_direction`。

### 4.40 T_Junction

XML 事实：247 条；Town01/02/03/04/05/10HD；waypoints 固定 2 左右；tags 主要 `trigger_point`。

目标 RS：R1/R4。

主规则：

- R4：T junction 有 signal controller 或有效灯态。
- R1：路口外。

否决/降级：

- 若无灯态且无 controller，临时 R5 medium + review，不直接 high R4。
- T 字形来自 junction lane pairs/route heading，不单靠 scenario 名。

输出 evidence：`r4_signalized_t_junction`、`t_junction_unsignalized_review`。

### 4.41 VehicleOpensDoorTwoWays

XML 事实：112 条；Town12/13；waypoints 2-68，均值 9.4；tags=`trigger_point/distance/frequency/direction`。

目标 RS：R1/R2/R4/R6。

主规则：

- R4：灯态/受控路口优先。
- R2：opposite lane+narrow road，且绕开开门车辆必须占用或等待对向车道。
- R6：parking/shoulder/curbside stopped vehicles 证据成立，且主决策是路边开门风险。
- R1：窗口外。

仲裁：

```text
R4 > 必须借/等待对向车道的 R2 > 停车开门空间 R6 > R1
```

否决/降级：

- R2/R6 分差 <0.15 时 review。
- `dist_to_vehicle_opens_door` 强化 R6，但不单独 high R6。

输出 evidence：`vehicle_open_door_r2_required_opposite_lane`、`vehicle_open_door_r6_parking_context`。

### 4.42 VehicleTurningRoute

XML 事实：764 条；多 town；waypoints 2-5，均值 2.1；tags 主要 `trigger_point`；scenario tag 总数 1062，说明单 route 可能多 trigger。

目标 RS：R1/R4/R5。

主规则：

- 对每个 trigger 建转弯窗口：trigger 前 50m 到 route heading 完成转弯后 20m。
- R4：受控 junction 或有效灯态。
- R5：无控制/stop/yield/priority junction。
- R1：普通弯道或路口外。

否决/降级：

- 转弯动作本身不等于路口；XODR 不在 junction 且无灯态时保持 R1。
- 多 trigger 要分别建窗口，不能只取第一个。

输出 evidence：`vehicle_turning_route_signalized_or_unsignalized`。

### 4.43 VehicleTurningRoutePedestrian

XML 事实：96 条；Town12/13；waypoints 2-4，均值 2.5；tags 主要 `trigger_point`。

目标 RS：R1/R4/R5。

主规则：

- 与 VehicleTurningRoute 同口径。
- 行人/自行车横穿只影响 EVENT，不改 RS。

否决/降级：

- 不因为 pedestrian 字样把普通道路横穿强打 R5。

输出 evidence：`vehicle_turning_pedestrian_event_ignored_for_rs`。

---

## 5. Policy 组织建议

不要写 43 个完全独立函数。建议用 10 个 policy + scenario config：

```text
SameDirectionObstaclePolicy:
  Accident, ConstructionObstacle, ParkedObstacle

TwoWaysObstaclePolicy:
  AccidentTwoWays, ConstructionObstacleTwoWays, ParkedObstacleTwoWays,
  HazardAtSideLaneTwoWays

SignalizedJunctionPolicy:
  RedLightWithoutLeadVehicle, SignalizedJunctionLeftTurn,
  SignalizedJunctionLeftTurnEnterFlow, SignalizedJunctionRightTurn,
  T_Junction, OppositeVehicleRunningRedLight, BlockedIntersection

NonSignalizedJunctionPolicy:
  NonSignalizedJunctionLeftTurn, NonSignalizedJunctionLeftTurnEnterFlow,
  NonSignalizedJunctionRightTurn, OppositeVehicleTakingPriority,
  PriorityAtJunction

DefectJunctionPolicy:
  CrossJunctionDefectTrafficLight

HighwayMergePolicy:
  EnterActorFlow, EnterActorFlowV2, HighwayCutIn, HighwayExit,
  MergerIntoSlowTraffic, MergerIntoSlowTrafficV2

InterurbanPolicy:
  InterurbanActorFlow, InterurbanAdvancedActorFlow

ParkingPolicy:
  ParkingCutIn, ParkingCrossingPedestrian, ParkingExit

HybridPolicy:
  StaticCutIn, VehicleOpensDoorTwoWays, PedestrianCrossing,
  VehicleTurningRoute, VehicleTurningRoutePedestrian

DefaultMetaMapPolicy:
  ControlLoss, DynamicObjectCrossing, HardBreakRoute, HazardAtSideLane,
  CrossingBicycleFlow, noScenarios
```

各 policy 输出候选分数，统一交给仲裁器，避免每个场景各自发明优先级。

---

## 6. 输出格式

每帧建议输出：

```json
{
  "frame": 87,
  "primary_road_structure": "R4",
  "secondary_road_structures": ["R6"],
  "physical_road_structure_hint": ["signalized_junction", "parking_context"],
  "confidence_level": "high",
  "confidence": 0.91,
  "review_required": false,
  "transition_margin": false,
  "route_progress_m": 45.12,
  "route_projection_error_m": 0.42,
  "road_structure_candidates": {
    "R1": 0.12,
    "R4": 0.91,
    "R6": 0.64
  },
  "arbitration_reason": "traffic_light_priority_over_parking_context",
  "evidence": {
    "traffic_light_state": "Green",
    "current_active_scenario_type": "ParkingCutIn",
    "map_junction_id": 17,
    "has_signal_controller_nearby": true,
    "has_parking_or_shoulder_nearby": true,
    "rules_fired": ["r4_tl_confirmed", "r6_parking_cutin_space_confirmed", "r4_overrides_r6"]
  }
}
```

run 级额外输出：

- `review_spans`: XML 缺失、town 不一致、projection error 大、候选冲突、平滑前后变化大。
- `label_distribution`: 每个 R 的帧数/比例。
- `confidence_distribution`: high/medium/low 比例。
- `rules_fired_counter`: 便于看某条规则是否异常泛滥。

---

## 7. 第一版落地顺序

1. 建 XML 索引，支持 route 编号、多 trigger、Town*_Scenario*。
2. 读缺 XML summary，给 run 降级标记。
3. 读 meta，抽灯态、junction、active scenario、dist_to_*。
4. Phase A：不用 CARLA API，只用 scenario+xml+meta 跑通 R1/R4/R5/R2/R3/R6 候选。
5. Phase B：接 XODR，补 junction/controller/opposite lane/parking/merge/split。
6. 加 hysteresis 与 transition_margin。
7. 输出 framewise JSON、review spans、每 scenario 分布统计。

### 7.1 `quick_start.py` / `collector.py` 当前落地口径

当前代码层先按“保守增强”落地，不删除旧的强行填充候选：

- `collector.py` 的 `SCENARIO_TO_ROAD_STRUCTURE` 仍作为每帧 `road_structures` 候选全集输出；
- 新增 `primary_road_structure` / `secondary_road_structures` / `road_structure_candidates` /
  `evidence`，用于承载本文规则生成的主标签、备选标签、分数和证据；
- `RouteXmlIndex` 从 `AutoMoT/data/lead/<Scenario>/*.xml` 建索引，支持 `route_001783`
  与 `Town*_Scenario*` 等命名；
- `XodrTopologyProbe` 在远端环境有 `carla` Python API 时，直接用 XODR 构造 `carla.Map`
  查询 lane/junction/opposite/parking/merge 证据；本地缺 API 时自动降级，不影响 XML+meta 规则；
- `quick_start.py` 新增 `ROAD_STRUCTURE XML/XODR画像` 菜单项：逐 scenario 遍历所有 town，
  每个 town 至少抽 3 个 XML（若该 town 不足 3 个则全部抽取），同时记录 XODR 是否存在、
  junction/signal/controller 粗统计、waypoint 数与 scenario tag，用于检查“每类场景在每个 town
  实际面对的地图结构”。

后续如果有完整 LEAD meta，可继续把 hysteresis、transition_margin 和 review_spans 从设计稿补成
跨帧后处理；当前第一版已经能在单帧 annotation 中保存足够的 route/XML/XODR/meta evidence。

---

## 8. 验收 sanity

必须满足：

- `noScenarios` 不应产生大量 R2/R3/R5/R6。
- `CrossJunctionDefectTrafficLight` 不应产生 R4 primary。
- `Signalized*`、`RedLightWithoutLeadVehicle` 核心窗口以 R4 为主。
- `NonSignalized*`、`PriorityAtJunction`、`OppositeVehicleTakingPriority` 核心窗口以 R5 为主。
- TwoWays 不是全程 R2；只有 trigger/active/opposite-lane 窗口 R2。
- Parking* 不是全程 R6；灯控路口段 R4 优先。
- `ParkedObstacle` 不应打 R6；`ParkedObstacleTwoWays` 应在核心窗口打 R2。
- R3 只在高速/匝道/合流/驶出 policy 中出现；自行车横穿/无灯 EnterFlow 不应打 R3。
- R4 在红灯等待 stopline 前也能保持，不因 `is_junction=False` 提前变 R1。

建议抽检：

- 每类 scenario 随机 5 个 run。
- 每个 run 抽 label 切换前后各 5 帧。
- 每条 `review_required=true` 的 span 至少抽前 20 个。
- 对所有 rules_fired 计数 top10 做人工看图，避免某个弱规则泛滥。
