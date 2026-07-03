# ROAD_STRUCTURE 逐场景调研协议

本文规定后续为每个 CARLA scenario 设计 ROAD_STRUCTURE(RS) 帧级筛选逻辑前，必须完成的调研流程、证据产物和写代码约束。

核心目标不是给 43 类 scenario 套一组泛化模板，而是为每个 scenario 建立可追责的证据链：
如果后续某一帧被指出错标，能定位是 XML 没用好、XODR 拓扑解释错、meta 字段缺失/阈值不准、
RGB 可见性误判、轨迹投影不可信，还是规则仲裁优先级有问题。

本协议是后续写 `collector.py` / `quick_start.py` / `frame_annotation_logic.py` 的前置门槛。
没有完成本协议要求的 scenario 调研包时，只能写临时规则，并且帧级 evidence 必须标出
`complete_investigation_status.is_complete=false`。

## 0. 调研原则：每个场景单独建立机制假设

RS 筛选不能从“场景名 -> 固定 RS”开始，而要从“该 scenario 在哪些 town、哪些 route
阶段、哪些 XODR 拓扑和哪些帧级 meta/RGB 证据下，应该进入或退出某个 RS”开始。
每个 scenario 都必须先写自己的机制假设，再用数据验证它：

```text
scenario mechanism hypothesis
  -> town/run coverage from lead_data
  -> XML route + trigger + scenario parameter audit
  -> XODR town topology audit
  -> ego global trace projected onto XML/XODR
  -> RGB contact sheet and boundary frame inspection
  -> RS phase segmentation and threshold fitting
  -> confidence/review policy
  -> runtime rule_config / code implementation
```

机制假设必须回答：

- 该 scenario 的核心道路结构变化是什么：接近区、触发区、核心区、退出区分别可能是什么 RS。
- 哪些 town/route 会让同一 scenario 的 RS 逻辑不同，例如 Town12/Town13 大图高速/匝道、
  Town03/Town05 紧凑路口、Town15 新图等。
- XML 的 trigger 点只是机制锚点还是实际结构切换边界；如果不是边界，偏移多少米、理由是什么。
- XODR 里哪些 road/lane/junction/signal/controller/parking/shoulder/merge-split
  支撑该 RS；哪些只是弱 hint。
- meta 哪些字段是强证据，哪些只能做事件 span 或可见性辅助。
- RGB 只能做人工可见性核验和冲突审计，不能绕过 XML/XODR/meta 单独决定 RS。

没有上述机制假设的 scenario，即使 `rs_research.py` 自动生成了文件夹，也只能标记为
`auto_artifacts_ready=true`、`final_complete=false`。

## 1. 数据源硬约束

每个 scenario 的 RS 逻辑必须同时调研以下三类输入：

```text
AutoMoT/lead_data/<Scenario>/<run_id>/...
AutoMoT/data/lead/<Scenario>/<Town>_<route_key>.xml
AutoMoT/CARLA_0915/.../<Town>.xodr
```

这三类输入的角色固定如下，写规则时不能互相替代：

- `lead_data` 是帧级事实来源：自车全局坐标历史轨迹、速度、灯态、junction、active scenario、
  `dist_to_*`、bbox/RGB 都从这里来。
- `data/lead` XML 是 route 与 scenario 机制来源：route waypoints、trigger、scenario tag 和
  scenario 参数只负责给地理窗口、机制锚点和参数先验，不是单帧真值。
- `CARLA_0915` XODR 是静态拓扑来源：road/lane/junction/signal/controller/stop/yield/parking/
  shoulder/merge-split 只负责证明某类 RS 在该位置有拓扑基础，不能提供实时灯色或事件可见性。

LEAD run 输入还必须先经过异常时长过滤：

- LEAD 为 4Hz，`rgb/*.jpg` 数量 `>=361` 表示严格大于 1 分 30 秒。
- 除 `BlockedIntersection`、`ControlLoss` 白名单外，所有 `duration_s > 90` 的 run 都是异常采集，
  不允许进入 RS 调研、阈值拟合、SFT/GoalGen/LeadMoT 数据集或 probe。
- `Accident`、`park*`、`dynamic*` 不再有 90-100 秒存疑段豁免。
- 代码统一复用 `lead_video_tools.abnormal_duration_filter.is_abnormal_lead_route`；
  不允许各入口自行写一套时长判断。

当前数据状态：

- `AutoMoT/lead_data` 有 9715 个 run id，覆盖 43 个 scenario。
- `AutoMoT/data/lead` 与 `lead_data` 去重后的 9294 个 `(Scenario,Town,route_key)` 一一对应。
- 从 `lead_data/<Scenario>/<run_id>` 查 XML 时，`Scenario` 必须取 run 的父目录；
  run_id 先剥末尾 `MM_DD_HH_MM_SS` 时间戳，再只在存在时剥尾部采集后缀
  `_route0`；`Town12_route15` 这类 legacy key 本体里的 `route15` 不能剥，
  也不能要求它带 `_route0`。XML 文件名公式：`route_key` 以 `route_`
  开头时用 `<Town>_<route_key>.xml`，否则用 `<Town>_route_<route_key>.xml`。
- 2026-07-03 核对结果为缺失 0、冗余 0、命名不规范 0、XML 解析失败 0、内容结构异常 0；`data/lead` 内没有 `<weathis_juncer>` 残留。40 个 XML 的 `data_routes` 源文件位于不同 scenario 目录（36 个 `noScenarios`、4 个 `ConstructionObstacleTwoWays`），不是缺失；`ParkedObstacle/Town12_route_Town12_route15.xml` 覆盖有效并与 `lead_data/ParkedObstacle/Town12_Rep0_Town12_route15_*` 对应，但未在 `AutoMoT/data/data_routes` 找到直接源文件。
- XML 如果后续版本出现缺失，必须先从 `AutoMoT/data/data_routes` 按 `(town, route_key)` 全局反查补齐；只有确实找不到唯一源且 `data/lead` 也没有有效 XML 时，才允许进入 `xml_available=false` 降级逻辑。不要把现有 `ParkedObstacle/Town12_route_Town12_route15.xml` 当作缺失。
- `AutoMoT/CARLA_0915` 是 XODR 拓扑来源；没有 XODR 或无法加载 CARLA API 时，该 town 的调研状态不能标为 complete。

## 2. 每个 Scenario 的最小调研单元

对每个 scenario，先从 `lead_data` 和 `data/lead` 得到它涉及的全部 town。每个 town 至少抽 5 条 run id：

```text
first / q1 / middle / q3 / last
```

如果某个 town 不足 5 条，则读取全部可读 run。抽样不能只取连续前五条，因为 route 分布容易集中在同一小片区域。
如果某个 town 在异常时长过滤后没有任何可读 meta run，例如
`NonSignalizedJunctionLeftTurn/Town10HD` 当前本地两个 run 都没有可读 meta，则该 town
记录为 `meta_sample_available=false` 并跳过样本调研；其它有可读 meta 的 town 仍必须保证
5 个分散 id 或全量可读 run。

每个抽样 run 必须读取：

- XML：完整 route attr、weather、waypoints、所有 scenario trigger、scenario tag 和参数。
- XODR：该 town 的 road/lane/junction/signal/controller/stop/yield/parking/shoulder/merge-split 画像；
  对每个抽样 run，还要抽取 route/trigger/ego trace 附近的局部拓扑，而不是只写全 town 粗统计。
- LEAD meta：每个抽样 id 至少 first/middle/last 三帧；如果 `metas/` 为空，记录为空原因并降低完整性。
- RGB：每个抽样 id 至少 first/middle/last 三帧 stitched RGB；关键边界帧还要读取 trigger 附近、active scenario 附近和 RS 切换前后帧。
- 轨迹：自车 `pos_global/theta/speed` 历史轨迹，投影到 XML route 和 XODR road/lane。

抽样后必须再做一次边界增补，不允许只看每个 id 的 first/middle/last：

- XML trigger 前后：`s_trigger - 60m` 到 `s_trigger + 80m`，每 10m 至少抽 1 帧。
- `current_active_scenario_type` 变化点前后：前 8 帧、当前帧、后 8 帧。
- `traffic_light_state` 从 `None` 变为有效灯态或反向变化的前后：前 8 帧、当前帧、后 8 帧。
- `is_junction/junction_id` 变化点前后：前 8 帧、当前帧、后 8 帧。
- 候选 RS 分数接近的边界：top1-top2 分差 < 0.08 时抽前后窗口。

边界增补帧必须进入 `meta/*__frame_features.jsonl`、`rgb/*__boundary_frames/` 或等价摘要中，
并在 `rules/scenario_rule_design.md` 中说明它们如何影响阈值。

## 2.1 调研后的分析总结硬要求

完成 `rs_research.py --samples-per-town 5` 只是拿到证据包，不等于规则设计完成。
每个 scenario 调研后必须把结果回写为三层总结：

1. `rules/scenario_rule_design.md`：写清该 scenario 独有的 approach/core/exit 切换逻辑、
   候选 RS、rule_kind、阈值、置信度和 review 条件。
2. `ROAD_STRUCTURE_PER_SCENARIO_LABELING_DESIGN.md`：写入该 scenario 的最终设计口径，
   包括它为什么不是同一规则族中其它 scenario 的简单复制。
3. `ROAD_STRUCTURE_MAP_XML_LABELING_PLAN.md`：写入跨场景分析矩阵，记录本轮 sample 覆盖、
   自动完整性、阈值初值、未完成输入缺口和下一步人工复核入口。

每个 scenario 的分析总结至少回答：

- 该场景的 RS 变化由道路结构触发，还是由事件触发；如果是事件触发，为什么不能改变 RS。
- XML trigger 是前置窗口中心、核心窗口边界，还是只作为机制锚点。
- XODR 需要证明哪些结构：signal/controller、junction、opposite lane、merge/split、
  highway/ramp、parking/shoulder/curb。
- LEAD meta 中哪些字段是强证据，哪些只作为 EVENT/span 辅助。
- RGB/contact sheet 是否支持地图判断；若未人工检查，规则必须保留 review 通道。
- 阈值是全局规则族默认、scenario 特化，还是 town/run 特化；必须写出原因。

本轮 5-id/town 自动调研后的统一判断：

- 43 个 scenario 均已生成 `auto_artifacts_ready` 产物。
- 只有 `NonSignalizedJunctionLeftTurn/Town10HD` 缺可读 meta；该 town 在本地只有 2 个 run，
  且 `readable_run_count=0`，因此不能靠换 id 自动补齐。
- `SignalizedJunctionRightTurn/Town07` 已切换到唯一可读 meta 的
  `Town07_Rep0_route_002583_route0_01_09_12_58_34`，按“可读 run 全量读取”记为 auto-ok。
- 所有 scenario 的 `manual_map_rgb_checked=false`，因此当前阈值仍是
  `temporary_default_rule_config` 级别；正式写代码前必须完成人工 map/RGB 对齐验收。

## 3. 每个 Scenario 的调研文件夹

每个 scenario 必须有独立调研文件夹，建议输出到：

```text
AutoMoT/keyframe_filter/collection_output/rs_research/<Scenario>/
```

该目录默认作为本地调研产物，不入库、不 push；文档和代码只记录生成规范。需要共享给
远端的内容，必须先从自动产物中提炼为方案文档、阈值配置或小型审计摘要。目录结构：

```text
rs_research/<Scenario>/
  README.md
  town_index.json
  sampled_runs.json
  research_notes.md
  xml/
    <Town>/<run_id>__xml_summary.json
    <Town>/<run_id>__route.xml
  xodr/
    <Town>__xodr_summary.json
    <Town>__junction_signal_index.json
  meta/
    <Town>/<run_id>__meta_probe.json
    <Town>/<run_id>__frame_features.jsonl
  rgb/
    <Town>/<run_id>__sample_contact_sheet.jpg
    <Town>/<run_id>__boundary_frames/
  maps/
    <Town>/<run_id>__route_trigger_ego_trace.png
    <Town>/<run_id>__xodr_overlay.html
    <Town>/<run_id>__projection_metrics.json
  rules/
    scenario_rule_design.md
    thresholds.json
    confidence_policy.json
    failure_modes.md
    boundary_cases.jsonl
    validation_replay.json
```

`README.md` 必须列出：

- 本 scenario 涉及哪些 town。
- 每个 town 读取了哪些 run id。
- 每个 run 对应哪个 XML。
- 每个 town 用了哪个 XODR。
- 哪些 run 缺 meta/RGB/waypoint 或投影误差过大。
- 当前 RS 规则是否已 complete；如果不 complete，缺口是什么。
- `maps/*route_trigger_ego_trace.png` 与 `rgb/*sample_contact_sheet.jpg` 是否已人工检查；
  不允许只写“待检查”后把规则标 complete。

`research_notes.md` 必须记录本 scenario 的完整调研日志，至少包含：

- 读取顺序：哪些 town、哪些 run、哪些 XML、哪些 XODR。
- 对每个 sampled run 的人工观察：地图对齐是否可信，RGB 是否支持地图判断，哪些帧需要边界复核。
- 机制假设如何被修改：例如原先认为 trigger 前 60m 进入 R2，但 RGB/ego trace 显示必须到
  对向车道可见或 lane narrowing 后才进入。
- 仍不确定的地方：需要更多 run、更多 town、CARLA API waypoint 吸附或手工标注。

`maps/*route_trigger_ego_trace.png` 和 `rgb/*sample_contact_sheet.jpg` 是人工验收的主入口：

- 每个 scenario 的 `README.md` 必须显式写 `map_rgb_alignment_status`，取值为
  `checked_aligned` / `checked_conflict` / `not_checked`。
- 如果 `maps` 中 ego trace、XML route、trigger 明显不贴合，必须在该 run 写
  `projection_untrusted=true`，并且该 run 不可用于调阈值。
- 如果 `rgb` contact sheet 显示道路结构与 XODR/XML 判定冲突，必须在
  `rules/failure_modes.md` 写入冲突帧和冲突原因，帧级输出降到 `medium/low`。

## 4. 地图对齐与可视化

每个抽样 run 必须生成地图可视化，而不是只读字段：

- XML sparse waypoints。
- XML trigger point 和 scenario tag。
- 自车历史轨迹 `pos_global`。
- 每个抽样 id 的 first/middle/last 帧的位置和 yaw。
- meta active scenario / dist_to_* 触发区间。
- XODR road/lane/junction/signal/controller/stop/yield 的近邻摘要。

PNG 可用于快速检查，HTML/SVG 可用于细看。可视化至少要回答：

- 自车轨迹是否贴合 XML route。
- trigger 是否落在 route 附近。
- RS 切换边界是否和 junction/merge/opposite-lane/parking/signal 拓扑一致。
- RGB 中可见的道路结构和地图拓扑是否冲突。

如果 XML route 投影误差长期大于阈值，应把该 run 标为 `projection_untrusted`，不能用它调阈值。

地图对齐的最低数值验收：

- route projection median error <= 3.0m 才能作为高可信 route_s 依据。
- route projection p90 error <= 5.0m 才能用于阈值拟合。
- trigger 到 ego trace 最近距离 <= 20.0m 才能认为 XML trigger 与该 run 对齐。
- XODR waypoint 吸附结果的 road/lane 变化必须与 ego trace 方向连续；出现 lane sign 跳变时，
  必须检查是否真实对向/路口/地图边界，而不是直接把它当成 R2。

超过上述阈值不代表数据废弃，但只能作为低置信审计样本，不能用于提升规则分数。

## 4.1 XODR 局部证据要求

XODR 不能只做 town 级 `junctions/signals/controllers` 计数。每个 sampled run 必须在
route 和 ego trace 周围生成局部证据：

- 每个 XML waypoint 与 ego trace sample 的最近 `road_id/lane_id/s/lane_type/is_junction/junction_id`。
- trigger 点前后 80m route 窗口内的 junction、signal/controller、stop/yield、lane 数变化。
- ego trace 是否经过对向 driving lane、parking/shoulder lane、merge/split 或 highway/ramp road。
- 对 R2/R3/R6 候选，必须列出“拓扑存在”和“当前帧真的需要该结构”的区别。

局部 XODR 证据建议写入：

```text
xodr/<Town>__junction_signal_index.json
maps/<Town>/<run_id>__projection_metrics.json
meta/<Town>/<run_id>__frame_features.jsonl
```

`frame_features.jsonl` 中每帧至少应能回查：

```text
frame_id, ego_xy, speed, route_s, route_projection_error_m,
xodr_road_id, xodr_lane_id, xodr_junction_id,
xodr_flags, xml_trigger_delta_s, meta_flags,
candidate_scores, top_gap, review_required_reason
```

## 5. 每场景专属规则设计

每个 scenario 的 `rules/scenario_rule_design.md` 必须包含：

```text
1. Scenario 语义与候选 RS
2. 涉及 town 与样本覆盖
3. XML 字段如何使用
4. XODR 拓扑如何使用
5. Meta 字段如何使用
6. RGB 人工观察结论
7. 自车轨迹与 trigger/route 对齐结论
8. 帧级 RS 分段逻辑
9. 置信度规则
10. 低置信/冲突/缺字段时的 review 规则
11. 已知失败模式
12. 对应代码配置与阈值
13. 地图/RGB 对齐验收结论
14. 错帧回查路径
```

禁止只写“该 scenario 属于 R2/R4/R5”。必须说明这个 scenario 在不同 town、不同 route 阶段如何从 R1 切到目标 RS，以及何时切回 R1。

第 8 节必须用阶段化规则写，而不是散文概括。推荐格式：

```text
phase A: approach
  condition: route_s < trigger_s - X and no strong topology/meta evidence
  primary: R1
  confidence: high/medium/low rule

phase B: pre-trigger structure window
  condition: trigger_s - X <= route_s <= trigger_s + Y and XODR flags ...
  primary: R2/R3/R4/R5/R6
  boundary checks: ...

phase C: core scenario span
  condition: active_scenario or finite dist_to_* or RGB-visible object ...
  primary: ...

phase D: exit / recovery
  condition: route_s > trigger_s + Z or junction/merge/parking topology exited
  primary: R1 or route-topology-specific RS
```

对于多 trigger XML（例如一个 route 内多个 `<scenario>`），必须逐 trigger 建 window，
不能把全 route 合并成单一窗口。

第 12 节的阈值必须包含来源字段：

```json
{
  "threshold_name": {
    "value": 35.0,
    "unit": "m",
    "source": "sampled_meta_route_s_and_trigger_window",
    "supporting_runs": ["Town12_...", "Town13_..."],
    "reviewed_artifacts": [
      "maps/Town12/...__route_trigger_ego_trace.png",
      "rgb/Town12/...__sample_contact_sheet.jpg"
    ],
    "reason": "covers visible parking-exit merge start without tagging pre-approach frames"
  }
}
```

没有 `source/supporting_runs/reviewed_artifacts/reason` 的阈值只能作为默认初始值，不能声称
scenario 调研 complete。

## 6. 置信度与 Review

每帧输出必须包含：

```text
primary_road_structure
secondary_road_structures
confidence_score
confidence_level: high | medium | low
review_required
rules_fired
diagnostic_attribution
```

置信度建议：

- `high`：scenario prior、XML window、XODR topology、meta signal/junction/active 字段至少三源一致。
- `medium`：两源一致，或强 meta 信号成立但 XODR/RGB 支持不足。
- `low`：只有 scenario prior 或只有弱 XODR hint；必须 `review_required=true`。

必须记录降置信原因：

- `xml_missing_or_unmatched`
- `xodr_missing_or_unloaded`
- `meta_missing_or_empty`
- `route_projection_error_high`
- `rgb_conflicts_with_map`
- `candidate_scores_close`
- `scenario_prior_conflicts_with_meta`
- `boundary_hysteresis_uncertain`

`diagnostic_attribution` 必须按来源分桶，便于错帧回查：

```text
xml: matched/missing/untrusted, trigger_s, route_s, projection_error
xodr: town_loaded, road_id, lane_id, junction_id, topology_flags
meta: frame_id, ego_xy, speed, tl_state, active_scenario, finite_dist_to
rgb: checked/not_checked/conflict, contact_sheet_path, boundary_frame_path
arbitration: candidate_scores, top_gap, priority_override, hysteresis_state
```

## 7. 写代码规则

后续改 `collector.py` / `quick_start.py` / `frame_annotation_logic.py` 时必须遵守：

- 先更新或生成对应 scenario 的调研文件夹，再改规则阈值。
- 每个 scenario 必须有独立 `rule_config`，即使共享 policy 模板，也要显式记录自己的 town、XML 字段、meta 字段、阈值和失败模式。
- 不允许只靠 scenario 名全程强行填一个 RS；scenario 名只能给候选池和先验。
- 不允许只靠 RGB 判断 RS；RGB 只作为可见性和冲突审计证据。
- R4/R5 必须使用 meta 灯态、XODR signal/controller/stop/yield、junction 关系和 XML trigger 联合判定。
- R2 必须检查 XODR 对向车道/窄路/TwoWays trigger 窗口，不能 TwoWays 全程 R2。
- R3 必须检查 ramp/merge/split/highway/interurban 的 XODR 或 XML 证据，不能把任意 actor flow 当合流。
- R6 必须检查 parking/shoulder/curb/parking-exit 结构，`ParkedObstacle` 默认不是 R6。
- `noScenarios` 禁止弱证据升级到 R2/R3/R5/R6；除非 meta 灯态与 XODR signal 强一致，否则默认 R1。
- 所有阈值必须写入 `thresholds.json` 或 scenario config，不能散落在代码分支里。
- 每个规则分支都要写 `rules_fired` 和 `diagnostic_attribution`，方便逐帧回查。
- `rs_research.py` 生成的是证据包，不是最终标注器；运行时代码只能消费已经写入
  `rules/thresholds.json`、`rules/confidence_policy.json` 或 scenario config 的规则。
- 新增 scenario 阈值时，必须先在该 scenario 的调研包中更新
  `boundary_cases.jsonl` 和 `thresholds.json`，再改代码。
- 任何代码分支都不能依赖“只看 scenario 名”的 hard override；scenario 名只能决定候选池、
  默认先验和需要读取哪些字段。
- 代码输出的每一帧必须可以反向定位到调研包路径：
  `research_artifacts.scenario_dir`、`map_artifact`、`rgb_artifact`、`threshold_source`。
- 如果 `maps/*route_trigger_ego_trace.png` 或 `rgb/*sample_contact_sheet.jpg`
  没有 `checked_aligned` 记录，边界敏感帧必须 `review_required=true`。

更严格的代码顺序：

1. 先运行或更新 `rs_research.py`，生成/刷新对应 scenario 调研包。
2. 人工检查该 scenario 的 `maps/*route_trigger_ego_trace.png` 与
   `rgb/*sample_contact_sheet.jpg` 是否对齐，并在 README/rules 中记录结论。
3. 将阈值写入 `rules/thresholds.json` 或 scenario config，并写明来源。
4. 再改运行时代码分支。
5. 用同一批 sampled runs 回放帧级输出，确认每个边界帧都有 `rules_fired`、
   `diagnostic_attribution`、`review_required`。

禁止在运行时代码里新增匿名 magic number；如果必须临时加默认值，命名必须带
`temporary_default`，并让 evidence 写 `threshold_source=temporary_default`。

## 7.1 `rs_research.py` 更新方向

后续更新 `rs_research.py` 时，优先补齐这些能力：

1. 从 `lead_data/<Scenario>/` 自动枚举全部 town，按 first/q1/middle/q3/last 抽样且优先有 meta/RGB 的 run。
2. 对每个 sampled run 复制/摘要匹配 XML，并记录 XML 匹配公式和失败原因。
3. 对每个 town 加载 XODR，生成 town 级摘要和 route/trigger/ego trace 局部拓扑摘要。
4. 用 meta 全轨迹计算 route projection error、trigger 最近距离、junction/signal/active scenario 变化点。
5. 生成 `maps/*route_trigger_ego_trace.png`，图上必须同时显示 XML route、trigger、ego trace、
   每个 id 的 first/middle/last、边界帧和 projection 异常点。
6. 生成 `rgb/*sample_contact_sheet.jpg`，每个 id 至少含 first/middle/last；边界帧另存到
   `rgb/*__boundary_frames/`。
7. 写 `rules/scenario_rule_design.md` 的完整模板，并预填当前自动观察，但保留人工确认字段。
8. 写 `rules/boundary_cases.jsonl`：每条边界帧包含 frame_id、触发来源、候选分数、推荐 review 原因。
9. 写 `rules/validation_replay.json`：用 sampled runs 回放当前规则后统计每个 RS 的帧数、
   low confidence 数量、review_required 数量和主要失败原因。

脚本的输出状态必须分层：

```text
data_discovered: lead_data/xml/xodr 是否找到
auto_artifacts_ready: 自动摘要/图像是否生成
manual_map_rgb_checked: 人工是否检查 maps/rgb
thresholds_have_provenance: 阈值是否有来源
runtime_rule_ready: 是否允许代码标 complete
```

只有全部为 true，才能把 scenario 写成 `final_complete=true`。

## 8. 错帧回查流程

当用户指出某个 scenario/run/frame 的 RS 有问题时，按下面顺序查，不要直接改阈值：

1. 打开该 scenario 的 `README.md`，确认 town/run 是否在调研样本内，`map_rgb_alignment_status`
   是否已经人工检查。
2. 打开 `maps/<Town>/<run_id>__route_trigger_ego_trace.png`，确认 ego trace、XML route、
   trigger 是否对齐；若不对齐，优先修 XML/run 匹配或 route_s 投影。
3. 打开 `xml/<Town>/<run_id>__xml_summary.json` 和 `xodr/<Town>__xodr_summary.json`，
   确认 route trigger、scenario tag、junction/signal/parking/merge 拓扑是否被规则使用。
4. 打开 `meta/<Town>/<run_id>__frame_features.jsonl`，检查该 frame 附近的 speed、灯态、
   junction、active scenario、`dist_to_*` 和 route projection error。
5. 打开 `rgb/<Town>/<run_id>__sample_contact_sheet.jpg` 或 boundary frame，确认可见道路结构
   是否支持地图判断。
6. 最后才看 `rules/thresholds.json` 和运行时代码：判断是阈值边界、仲裁优先级、证据缺失，
   还是数据本身不可置信。

## 9. 逐场景覆盖清单

当前 `lead_data` 中各 scenario 的 town/run 覆盖如下。逐场景调研必须覆盖表中所有 town：

```text
Accident: Town03(13), Town04(18), Town05(23), Town06(34), Town10HD(6), Town12(51), Town13(47)
AccidentTwoWays: Town01(12), Town02(3), Town05(1), Town07(21), Town12(216), Town13(198), Town15(145)
BlockedIntersection: Town06(47), Town07(10), Town12(49), Town13(50)
ConstructionObstacle: Town03(13), Town04(18), Town05(24), Town06(31), Town10HD(6), Town12(51), Town13(50)
ConstructionObstacleTwoWays: Town01(11), Town02(3), Town05(1), Town07(18), Town12(209), Town13(188), Town15(129)
ControlLoss: Town01(5), Town02(6), Town03(23), Town04(59), Town05(60), Town06(31), Town07(12), Town10HD(16), Town12(50), Town13(48)
CrossJunctionDefectTrafficLight: Town03(27), Town04(17), Town05(40), Town07(7), Town10HD(7), Town12(16), Town13(14), Town15(27)
CrossingBicycleFlow: Town12(49)
DynamicObjectCrossing: Town01(6), Town02(6), Town03(23), Town04(59), Town05(63), Town06(30), Town07(13), Town10HD(15), Town12(50), Town13(49)
EnterActorFlow: Town12(48), Town13(45)
EnterActorFlowV2: Town12(43)
HardBreakRoute: Town12(49), Town13(48)
HazardAtSideLane: Town12(54), Town13(50)
HazardAtSideLaneTwoWays: Town12(49), Town13(52)
HighwayCutIn: Town12(50), Town13(56)
HighwayExit: Town12(58), Town13(40)
InterurbanActorFlow: Town12(42), Town13(49)
InterurbanAdvancedActorFlow: Town12(39), Town13(39)
InvadingTurn: Town12(51), Town13(51)
MergerIntoSlowTraffic: Town12(49), Town13(47)
MergerIntoSlowTrafficV2: Town06(2), Town12(50), Town13(53)
NonSignalizedJunctionLeftTurn: Town03(5), Town04(10), Town05(19), Town07(17), Town10HD(2), Town12(114), Town13(82)
NonSignalizedJunctionLeftTurnEnterFlow: Town03(4), Town04(8), Town05(18), Town07(15), Town10HD(1), Town12(104), Town13(65)
NonSignalizedJunctionRightTurn: Town12(47), Town13(48)
OppositeVehicleRunningRedLight: Town01(24), Town02(16), Town03(29), Town04(35), Town05(47), Town06(14), Town07(13), Town10HD(17), Town12(49), Town13(52)
OppositeVehicleTakingPriority: Town12(50), Town13(47)
ParkedObstacle: Town03(12), Town04(21), Town05(24), Town06(35), Town10HD(7), Town12(52), Town13(57)
ParkedObstacleTwoWays: Town12(50), Town13(52)
ParkingCrossingPedestrian: Town12(53), Town13(48)
ParkingCutIn: Town12(49), Town13(50)
ParkingExit: Town03(41), Town10HD(95), Town12(52), Town13(54), Town15(14)
PedestrianCrossing: Town12(51), Town13(50)
PriorityAtJunction: Town12(49), Town13(50)
RedLightWithoutLeadVehicle: Town01(23), Town02(15), Town03(46), Town04(36), Town05(102), Town06(17), Town07(13), Town10HD(17), Town12(47), Town13(43)
SignalizedJunctionLeftTurn: Town01(12), Town02(9), Town03(23), Town04(32), Town05(47), Town06(50), Town07(18), Town10HD(8), Town12(52), Town13(52), Town15(72)
SignalizedJunctionLeftTurnEnterFlow: Town01(10), Town02(10), Town03(23), Town04(27), Town05(51), Town07(6), Town10HD(8), Town12(48), Town13(46), Town15(13)
SignalizedJunctionRightTurn: Town01(30), Town02(23), Town03(34), Town04(36), Town05(53), Town07(5), Town10HD(20), Town12(48), Town13(50), Town15(63)
StaticCutIn: Town12(49), Town13(51)
T_Junction: Town01(24), Town02(16), Town03(45), Town04(41), Town05(102), Town10HD(19)
VehicleOpensDoorTwoWays: Town12(51), Town13(66)
VehicleTurningRoute: Town01(46), Town02(32), Town03(158), Town04(125), Town05(134), Town06(54), Town07(129), Town10HD(45), Town12(47), Town13(50)
VehicleTurningRoutePedestrian: Town12(50), Town13(51)
noScenarios: Town03(34), Town04(217), Town05(119), Town06(320), Town07(165), Town10HD(27), Town15(554)
```

## 10. 完成标准

某个 scenario 的 RS 逻辑只有满足以下条件，才能标记为调研完成：

- 所有涉及 town 都有 XODR 摘要和地图可视化。
- 每个 town 至少 5 条 run 或全部可读 run 被读取。
- 每个抽样 run 的 XML、meta、RGB、自车轨迹都完成摘要。
- 已生成 route/trigger/ego trace 地图 overlay。
- `scenario_rule_design.md` 写明帧级切换边界、置信度和失败模式。
- 至少 10 个边界帧完成 RGB 人工抽检；不足 10 个边界帧则全量抽检。
- 代码规则中的每个阈值都能追溯到该 scenario 的调研文件。

不满足以上条件时，代码可以输出临时 RS，但必须在 evidence 中写：

```text
complete_investigation_status.is_complete=false
```

完成标准还要求：

- `rules/thresholds.json` 中每个阈值有来源、单位、支持 run 和对应可视化路径。
- `rules/confidence_policy.json` 写明 high/medium/low 的源一致性条件。
- `rules/failure_modes.md` 至少列出 XML/XODR/meta/RGB/仲裁五类潜在失败模式。
- `scenario_rule_design.md` 能回答“为什么这一帧从 R1 切到目标 RS、为什么又切回 R1”。
