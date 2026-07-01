# 基于 CARLA 地图 + Route XML + LEAD Meta 的 ROAD_STRUCTURE 帧级标注方案

## 1. 背景与目标

你当前的问题很明确：仅靠 VLM（Qwen）看图判断每帧属于哪个场景/道路结构，稳定性不足，尤其在场景切换边界和视觉歧义帧上容易漂移。

本方案目标是把 `ROAD_STRUCTURE` 的主判断从“视觉语义”切到“几何与规则语义”，即：

1. 用每帧自车全局坐标（`metas/*.pkl`）确定自车在地图中的位置与拓扑关系。
2. 用 `data/data_routes/lead/<Scenario>/route_xxxxxx.xml` 提供 route 先验和 scenario 触发信息。
3. 用 meta 中的 scenario 距离字段与激活字段，提供“事件区间”证据（尤其是 R2/R6 这类局部触发结构）。
4. 产出每帧 `ROAD_STRUCTURE`（R1~R6）和可解释的置信来源，作为后续 EVENT 标注与 Qwen 候选裁剪的上游输入。

一句话：把 Qwen 从“主裁判”降级为“补充裁判”，先用地图与轨迹给出强规则主标签。

---

## 2. 你当前数据里已经可用的关键信号

### 2.1 路线 XML（强先验）

示例：`data/data_routes/lead/Accident/route_001761.xml`

- `town`：例如 `Town05`
- `waypoints/position(x,y,z)`：route 轨迹稀疏采样点
- `scenarios/scenario`：包含 `type`、`trigger_point` 等

这些信息可用于：

- run 与 route 的确定性匹配
- route 进度估计（沿 XML polyline 的弧长）
- scenario 触发点的几何锚定

### 2.2 每帧 meta（强观测）

`metas/*.pkl` 实际是 xz 压缩 pickle（需 `lzma.open` 读取）。关键字段（实测存在）：

- 位姿：`ego_matrix`（4x4），`theta`
- 路口：`is_junction`，`is_intersection`，`dist_to_junction`，`distance_to_next_junction`
- 信号灯：`traffic_light_state`
- 车道：`lane_id`，`ego_lane_id`，`lane_type_str`，`lane_change_str`
- 场景激活：`current_active_scenario_type`
- 场景距离：`dist_to_accident_site`、`dist_to_construction_site`、`dist_to_vehicle_opens_door`、`dist_to_cutin_vehicle`、`dist_to_pedestrian` 等

这些字段足以先做一个不依赖 VLM 的强规则版本。

### 2.3 结果文件（run 级辅助）

`results.json` 含 `route_id`（如 `RouteScenario_route_001761_rep0`）与时长，用于 run→xml 对齐校验。

---

## 3. run 与 route XML 的对齐规则

## 3.1 目标

对每个 run 建立唯一映射：

- `scenario_name`
- `route_num`（如 001761）
- `town`
- `route_xml_path`

## 3.2 推荐实现

1. 从 run 文件夹名提取 route 编号：匹配 `route_(\d+)`。
2. 结合 scenario 目录拼接 XML：
   - `data/data_routes/lead/<scenario_name>/route_<route_num>.xml`
3. 若存在多个候选（极少数命名异常），用 `results.json.route_id` 二次确认。
4. 校验 `xml.route.town == meta.town`（抽样帧）防止错配。

## 3.3 对齐失败兜底

- 若 XML 缺失：该 run 仅用 meta 做弱规则标注，置信度降级。
- 若 town 不一致：打硬错误并进入人工复核队列。

---

## 4. 帧级几何定位与拓扑特征构建

## 4.1 从 meta 取每帧全局坐标

对每帧：

- `x = ego_matrix[0][3]`
- `y = ego_matrix[1][3]`
- `z = ego_matrix[2][3]`
- `yaw = theta`

## 4.2 route 进度 s（沿线弧长）

把 XML waypoints 组成 polyline，计算每帧点到 polyline 的最近投影，得到：

- `s_frame`：沿路线累计距离（米）
- `d_route`：到路线中心线距离（米）

用途：

- 同一 run 内时序对齐稳定
- 可把 trigger_point 投影成 `s_trigger`，构造“触发窗口”

## 4.3 地图拓扑特征（理想路径）

若可连接 CARLA map（或离线 OpenDRIVE），建议为每帧补充：

- `is_junction_map`
- `junction_id_map`
- `lane_count_same_dir`
- `has_opposite_driving_lane_nearby`
- `is_ramp_or_merge_zone`
- `has_traffic_light_actor_near_junction`

说明：仓库内未发现 `.xodr`，因此实现时可采用两级策略：

- A 档（优先）：CARLA API 实时查询地图拓扑
- B 档（离线）：先用 meta 字段近似替代（`is_junction`/`dist_to_junction`/`traffic_light_state` 等）

---

## 5. ROAD_STRUCTURE 判定主逻辑（核心）

建议做成“候选 + 打分 + 优先级覆盖 + 时序平滑”四段式。

## 5.1 每帧候选打分（score）

对每帧计算 `score[R1..R6]`，初值 0。

### R4（信号灯路口）加分

- `is_junction` 或 `distance_to_next_junction < T_junc_near`：+a
- `traffic_light_state in {Red,Yellow,Green}`：+b（强证据）
- scenario 先验属于 signalized 类：+c

### R5（无信号灯/灯失效路口）加分

- `is_junction == True`：+a
- 且 `traffic_light_state in {None,Unknown}` 持续 K 帧：+b
- scenario 先验属于 non-signalized / priority / defect 类：+c

### R3（高速/匝道/合流/驶出）加分

- scenario in {HighwayExit, HighwayCutIn, EnterActorFlow, MergerIntoSlowTraffic, ...}：+a
- 地图特征 `is_ramp_or_merge_zone == True`：+b（强证据）
- route 命令/曲率模式符合并线或驶出：+c

### R2（双向单车道/借对向）加分

- scenario in {AccidentTwoWays, ConstructionObstacleTwoWays, ParkedObstacleTwoWays, HazardAtSideLaneTwoWays, InvadingTurn, VehicleOpensDoorTwoWays}：+a
- 地图特征存在紧邻对向车道且当前道路窄：+b（强证据）
- 对向相关风险距离字段进入阈值窗口：+c

### R6（停车带/路边停车主导）加分

- scenario in {ParkingCutIn, ParkingExit, ParkingCrossingPedestrian, VehicleOpensDoorTwoWays, StaticCutIn(部分)}：+a
- `dist_to_vehicle_opens_door` / `dist_to_parked_obstacle` / `dist_to_cutin_vehicle` 接近：+b（强证据）
- 地图特征 `lane_type=Parking/Shoulder` 或路侧停车密集区域：+c

### R1（常规道路）加分

- 默认保底分 `base_r1`
- 不在路口、不在触发窗口、无 R2/R3/R6 强证据时持续占优

## 5.2 决策优先级（避免冲突）

当多类分数接近时，用优先级裁决：

1. R6（局部强触发）
2. R5 / R4（路口规则主导）
3. R3（高速/匝道结构）
4. R2（对向参与）
5. R1（默认）

关键原则：

- 路口中优先判 R4/R5，不要被 R1 吃掉。
- R6 只在“触发区间”内生效，离开后应自动回落到 R1 或 R4/R5。
- R2/R3 不应整段常驻，除非拓扑证据持续存在。

## 5.3 R4 vs R5 的判别准则

建议组合判断，避免单信号误判：

1. 先看是否在路口区域（`is_junction` 或 `dist_to_junction < T`）。
2. 在路口区域内：
   - 若存在有效灯态（Red/Yellow/Green）连续 >= K 帧，优先 R4。
   - 若灯态长期 None 且 scenario 属于非信号/失效类，优先 R5。
3. `CrossJunctionDefectTrafficLight` 强制偏置到 R5。

## 5.4 R6 的区间化触发（你最关心）

R6 不做全程标签，只做局部片段标签。

构建方式：

1. 取关键距离字段（如 `dist_to_vehicle_opens_door`、`dist_to_cutin_vehicle`）。
2. 找到低于阈值 `T_r6_on` 的连续帧段，作为激活核心段。
3. 对核心段前后各扩张 `pad_in/pad_out` 帧，形成 R6 活跃窗口。
4. 活跃窗口之外，按 R1 或路口规则（R4/R5）回退。

这样就满足你说的：

- 在关键触发区域才是 R6
- 驶离后自动变回 R1 或 R4/R5

---

## 6. 场景触发窗口（Scenario Window）融合

你现有数据里 `current_active_scenario_type` 已经非常有价值，可直接并入规则层。

推荐策略：

1. 由 `current_active_scenario_type == scenario_name` 形成原始激活掩码。
2. 再与距离字段阈值交集，得到高置信触发窗口。
3. 窗口内允许 R2/R3/R6 获得更高先验分；窗口外大幅衰减。

这样可以抑制“某些场景名在视觉上像全程生效”的误判。

## 6.1 重要约束：XML trigger 不是事件“可见起点”

这个约束必须单独强调：

- XML 里的 `trigger_point` 代表 scenario 机制开始被仿真系统激活的地理锚点。
- 它不等于人类或模型在当前相机帧里“已经看见事件对象”。

因此，`trigger_point` 只能作为“候选时间窗起点”，不能直接作为 EVENT 注入起点。

否则会出现你说的问题：自车还没看见障碍，标签却已经打成 `U-E2`，导致学习被污染。

## 6.2 U-E2 注入时机（防过早标注）

对 `U-E2 静态障碍物占道`，推荐三段式门控：

1. 激活门（scenario-level）：
  - 满足 `current_active_scenario_type == scenario_name` 或 `s_frame >= s_trigger - pad_pre`。
  - 只说明“事件可能发生”，不直接打 U-E2。
2. 可见/可感知门（per-frame）：
  - 仅当 `dist_to_accident_site` / `dist_to_construction_site` / `dist_to_parked_obstacle` / `dist_to_vehicle_opens_door`
    之一进入阈值 `T_visible_on` 时，才允许注入 `U-E2`。
  - 若有 bbox/前向视锥信息，可加条件“障碍位于前向可视区域”。
3. 持续门（hysteresis）：
  - 注入后直到距离回升到 `T_visible_off`（`T_visible_off > T_visible_on`）并持续 K 帧才退出。
  - 防止边界抖动。

推荐阈值初始化：

- `T_visible_on = 25m`（远距先保守）
- `T_visible_off = 32m`
- `K = 3~5` 帧

最终以你抽检结果再调参。

## 6.3 与 ROAD_STRUCTURE 的关系

- ROAD_STRUCTURE 帧级标注本身不依赖“是否看见障碍”，它主要由道路规则空间决定。
- 上述“可见性门控”主要作用于 EVENT 注入时机（尤其 U-E2/U-E3/U-E4）。

也就是：

- 你可以先稳定输出 ROAD_STRUCTURE。
- 再在对应 ROAD_STRUCTURE 下，按可见门控注入突发事件标签。

---

## 7. 时序平滑（避免帧抖动）

建议最少做两层平滑：

1. 中值/众数滑窗（窗口 5~9 帧）
2. 最短持续约束（例如少于 4 帧的孤立标签回并到邻居）

可选：HMM/Viterbi，转移代价建议：

- 低代价：R1 <-> R4、R1 <-> R5、R1 <-> R6
- 中代价：R1 <-> R2、R1 <-> R3
- 高代价：R2 <-> R5、R3 <-> R5（通常不直接跳）

---

## 8. 产出格式建议

建议新增每 run 一个帧级标注文件，例如：

- `keyframe_filter/outputs/road_structure_labels/<Scenario>/<run_id>.json`

单帧结构示例：

```json
{
  "frame": 87,
  "timestamp_sec": 21.75,
  "ego": {"x": 210.74, "y": -11.27, "z": -0.02, "yaw": -1.58},
  "route_progress_m": 45.12,
  "road_structure": "R6",
  "confidence": 0.91,
  "evidence": {
    "is_junction": false,
    "traffic_light_state": "None",
    "current_active_scenario_type": "ParkingCutIn",
    "dist_to_cutin_vehicle": 9.8,
    "rules_fired": ["scenario_window", "r6_distance_trigger", "priority_override"]
  }
}
```

注意保留 `evidence.rules_fired`，便于后续人工抽检和阈值调参。

---

## 9. 与你现有 ROAD/EVENT 体系的衔接

你已经有两份关键文档：

- `ROAD_EVENT_CLASSIFICATION_PLAN.md`
- `ROAD_EVENT_CANDIDATE_MAPPING.md`

衔接方式：

1. 本方案先生成每帧 `ROAD_STRUCTURE` 单标签。
2. 再按映射表第 4/5 节，得到每帧 `EVENT` 候选集合。
3. 最后再决定是否让 Qwen 只在“候选集合内”做细粒度判别。

这样 Qwen 不再需要先猜你在哪种道路结构，错误会明显减少。

---

## 10. 分阶段落地计划（建议）

## Phase A（1~2 天）：不依赖 CARLA 地图，先跑通

输入：`metas + route_xml + candidate_mapping`

- 完成 run->xml 对齐
- 完成帧级 `R1~R6` 规则初版（主要依赖 meta 字段）
- 输出每帧标签与证据

目标：先把 Qwen 的结构判断替换掉 70% 以上。

## Phase B（2~4 天）：引入 CARLA 地图拓扑增强

- 用 `carla.Map.get_waypoint` / junction / lane topology 增强 R2/R3/R4/R5 边界
- 增加 ramp/merge 检测与对向车道参与检测

目标：减少 R2/R3 与 R4/R5 混淆。

## Phase C（持续）：阈值自动校准与评估

- 抽样人工标注 200~500 帧作验证集
- 调整阈值与优先级
- 增加失败案例回放清单

---

## 11. 关键实现细节与风险提示

1. `metas/*.pkl` 是 xz 压缩，读取必须 `lzma.open`。
2. 很多距离字段会出现 `inf`，计算阈值前要先过滤。
3. `current_active_scenario_type` 不是全程有效，常出现 `None`，必须与距离字段联合。
4. `traffic_light_state` 可能短时抖动，不可单帧硬判 R4/R5。
5. 若拿不到 CARLA 地图，不影响第一版上线；先做 meta+xml 版即可。

---

## 12. 伪代码（可直接转脚本）

```text
for run in all_runs:
    scenario = parse_scenario_from_path(run)
    route_num = parse_route_num(run)
    xml = load_route_xml(scenario, route_num)
    route_polyline = xml.waypoints
    trigger_points = xml.scenarios.trigger_point

    frames = load_all_metas_lzma(run)

    for t, meta in enumerate(frames):
        ego = extract_ego_xyz_yaw(meta)
        s, d = project_to_route_polyline(ego.xy, route_polyline)

        feat = build_features(meta, ego, s, d)
        # feat: is_junction, traffic_light_state, dist_to_*, active_scenario, ...

        score = init_scores(R1..R6)
        score += apply_r4_rules(feat, scenario)
        score += apply_r5_rules(feat, scenario)
        score += apply_r3_rules(feat, scenario, map_feat)
        score += apply_r2_rules(feat, scenario, map_feat)
        score += apply_r6_rules(feat, scenario)
        score += apply_r1_default(feat)

        label[t] = argmax_with_priority(score)
        evidence[t] = collect_fired_rules()

        # 事件注入要做“可见性门控”，不要把 xml trigger 直接当作 U-E2 起点
        event_gate = build_event_visibility_gate(feat, s, trigger_points)
        events[t] = inject_events_with_hysteresis(label[t], feat, event_gate)

    label = temporal_smoothing(label)
      events = temporal_smoothing_events(events)
      dump_framewise_road_and_events(run, label, events, evidence)
```

---

## 13. 结论

你的思路是对的，而且从现有数据字段看已经具备可实施条件。

最务实路径是：

1. 先用 `meta + route xml` 做强规则帧级 ROAD_STRUCTURE；
2. R6 做区间触发，不做全程标签；
3. 后续再接 CARLA 地图拓扑增强边界；
4. Qwen 只在候选空间内做细分，不再承担主结构判断。

这条线可以显著提升一致性，也更容易解释和调参。