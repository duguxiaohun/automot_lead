# 场景事件采集系统

本目录用于从 LEAD 离线数据中采集帧级 ROAD_STRUCTURE / EVENT 候选，并用
XML + XODR + meta 生成更精细的 `primary_road_structure`。

当前重点是 ROAD_STRUCTURE + EVENT：

- 保留每个 scenario 的候选全集 `road_structures`，避免破坏旧 Web/分析逻辑。
- 额外输出 `primary_road_structure`、`secondary_road_structures`、
  `road_structure_candidates`、`evidence`。
- 逐帧 EVENT 由 `RoadEventRuleEngine` 在已确定的 `primary_road_structure` 上再融合
  scenario 白名单、XML trigger/active window、LEAD meta 距离/刹车/速度/轨迹字段和
  route 级去抖得到；输出 `events`、`primary_event`、`event_evidence` 和
  `frame_event_annotation`。
- `AutoMoT/data/lead` 只提供 route XML；真实帧数据必须来自 `lead_data`。
- `metas/*.pkl` 或匹配 route XML 缺失时，该 run 是数据质量问题，不进入 RS/EVENT 标注、
  调参或 probe。采集和可视化只记录 `data_missing_skip` / `skip_reasons`，明确说明
  `missing_meta` 或 `missing_xml`，不要把这类样本当作规则漏检。
- EVENT 口径以 `ROAD_EVENT_CLASSIFICATION_PLAN.md` / `ROAD_EVENT_CANDIDATE_MAPPING.md`
  为准：默认单主事件，只有 R4/R5 路口允许常规路口事件与路口专属 U-E 双触发；
  `U-E2/U-E3` 不进入十字路口/T 形路口候选池。非常规事件必须在
  scenario 白名单、当前 RS 候选池和 XML/active 候选窗口内，再由 meta/轨迹/RGB 证据触发。旧
  `keyframes_all_scenarios.json` 和 `frame_annotation_logic.py` 只能作 legacy/span 提议参考。
- XML 命名规范固定为 `data/lead/<Scenario>/<Town>_<route_key>.xml`。从
  `lead_data/<Scenario>/<run_id>` 查 XML 时，`Scenario` 必须直接取 run 的父目录；
  run_id 先剥末尾 `MM_DD_HH_MM_SS` 时间戳，再只在存在时剥尾部采集后缀 `_route0`，
  剩余部分就是 route_key。`Town12_route15` 里的 `route15` 是 key 本体，不能剥，
  也不能要求它带 `_route0`。
- XML 文件名只按 route_key 公式生成：`route_key` 以 `route_` 开头时用
  `<Town>_<route_key>.xml`，否则用 `<Town>_route_<route_key>.xml`。示例：
  旧数字 route 用 `Town03_route_001783.xml`，新版子编号用
  `Town12_route_1054_0.xml`，命名本身带 Town 的 legacy key 用
  `Town06_route_Town06_13.xml`，legacy key 内部带 route 编号时保留完整 key，
  如 `Town12_route_Town12_route15.xml`。
- XML 索引匹配必须优先使用 `(Scenario,Town,route_key)` / 别名精确匹配，再用
  `(Scenario,Town,route_num)` 数字兜底；不能让 `Town07_route_001456` 被纯数字
  `1456` 撞到 `Town12_route_1456_0.xml`。若出现跨 town 数字歧义，宁可降级 review，
  不要拿错误 XML/XODR 解释 RGB。
- 2026-07-03 全量核对确认 `lead_data` 去重后的 9294 个
  `(Scenario,Town,route_key)` 均有对应 XML，命名不规范 0、XML 解析失败 0、
  内容结构异常 0。其中 40 个 XML 的 `data_routes` 源文件位于不同 scenario 目录，
  但不是缺失；另有 `ParkedObstacle/Town12_route_Town12_route15.xml` 覆盖有效并与
  `lead_data/ParkedObstacle/Town12_Rep0_Town12_route15_*` 对应，但未在
  `AutoMoT/data/data_routes` 找到直接源文件。使用时以 `lead_data` / `data/lead`
  的 scenario 目录为准，不能把该项当作 XML 缺失。

---

## 快速开始

默认从远端 `AutoMoT/` 当前目录下读取：

```text
lead_data/<Scenario>/<run_id>/metas/*.pkl
data/lead/<Scenario>/*.xml
CARLA_0915/.../*.xodr
```

在 `AutoMoT/` 目录运行：

```bash
python keyframe_filter/quick_start.py
```

帧级规则会优先使用能 `import carla` 的 Python 环境读取精确 XODR 拓扑
（`map_road_id/lane_id/lane_type`、`has_opposite_driving_lane`、
`has_parking_or_shoulder_nearby` 等），例如：

```bash
/home/codon/anaconda3/envs/carla/bin/python keyframe_filter/quick_start.py
```

默认 Python 若不能 import `carla`，采集不会中断，会自动降级为静态 XODR planView/lane/signal
近邻解析。静态解析只有在 `map_projection_error_m <= 20m` 时设置
`xodr_topology_trusted=true` 并允许 R2/R3 使用 topology high 证据；超过该误差时只保留
`xodr_topology_untrusted` 诊断，特殊 RS 会降为 medium/low + review。
静态 XODR 的 junction/signal 只能作为 R4/R5 辅助证据：`map_junction_id=-1` 或
`junction_connection_count=0` 的 junction hint 不能单独制造强路口上下文；若 RGB 看不到
stopline、traffic light、横向车流或路口几何，应回 R1/review。
环岛 / roundabout 明确按 R1 处理：XODR 若输出 `map_is_roundabout=true`，即使附近有
junction road，也会压住 R4/R5 并在 evidence 中写入 roundabout 规则命中。

菜单里的 1/2/3/4 都是正式“采集 + 逐帧 RS/EVENT 标注”入口：每帧都会同时写
`road_structures` 候选全集与该帧独属的 `primary_road_structure` /
`frame_rs_annotation.label`，并在已确定 RS 的基础上写 `primary_event` /
`frame_event_annotation.label`。第 9 项只作为小范围 smoke / 参数闭环调试入口保留。

非交互生成逐帧标注（等价于走采集器的逐帧标注链路）：

```bash
python keyframe_filter/quick_start.py annotate-rs \
  --scenario all \
  --output-dir keyframe_filter/collection_output/rs_annotation_full
```

`annotate-rs` 默认全量：不传 `--max-routes` 时，会处理所选 scenario 下全部合法
routes；不传 `--max-frames-per-route` 或传 `0` 时，会处理每条 route 的全部帧。
小范围 smoke / 参数闭环调试才显式传 `--max-routes`：

```bash
python keyframe_filter/quick_start.py annotate-rs \
  --scenario T_Junction,AccidentTwoWays,ParkingExit \
  --max-routes 1 \
  --max-frames-per-route 80 \
  --output-dir /tmp/automot_rs_annotation_test
```

按每个 town 抽样时使用 `--samples-per-town`，它优先于 `--max-routes`，适合做场景/town
覆盖回归：

```bash
python keyframe_filter/quick_start.py annotate-rs \
  --scenario all \
  --samples-per-town 5 \
  --max-frames-per-route 120 \
  --output-dir /tmp/automot_event_rules_all_town5_120
```

该入口会按每个 scenario 的 `SCENARIO_RULE_CONFIG` 独立规则逐帧输出
`primary_road_structure`、`secondary_road_structures`、`primary_event`、`events`、
`annotation_comment`、`event_evidence`、`evidence.review_reasons` 和 route 级分布/切换摘要。
全局摘要 `frame_rs_annotation_summary.json` 还会写入 `road_structure_labels` 与
`event_labels`，Web 页面和人工检查都应使用这两个词典解释代号。
`road_structures` 仍保留旧候选全集；真正的单帧道路结构结果请读新增的
`frame_rs_annotation`，真正的单帧事件结果请读 `frame_event_annotation`：

```json
{
  "frame_id": 0,
  "frame_time_s": 0.0,
  "road_structures": ["R1", "R4"],
  "primary_road_structure": "R4",
  "frame_rs_annotation": {
    "label": "R4",
    "secondary": [],
    "confidence": 0.96,
    "comment": "R4：规则族=signalized_junction...",
    "rule_kind": "signalized_junction",
    "rules_fired": ["r1_default_candidate", "r4_tl_confirmed"],
    "decision_source": "meta_traffic_light",
    "review_required": false,
    "review_reasons": [],
    "metrics": {
      "route_progress_m": 0.4,
      "route_projection_error_m": 0.0,
      "trigger_distance_m": 2.0,
      "traffic_light_state": "Red"
    },
    "xodr_summary": {
      "available": true,
      "source": "static_xodr",
      "trusted": false,
      "opposite_lane": false,
      "parking_or_shoulder": false,
      "merge_split_hint": false
    }
  }
}
```

EVENT 后处理会按 scenario 家族执行 route 级去抖：

- 障碍 / TwoWays / 开门 / Hazard 类：U-E2 短间隙会合并；仅靠 route 开头 XML trigger
  和 `speed_reduced_by_obj_distance` 的初始 U-E2，如果短窗口内没有具体障碍距离或变道轨迹，
  会压回常规事件。为绕障离开原车道的短 R-E2 会吸收进 U-E2；U-E2 结束后只有在
  过障碍核心后的负向 `signed_dist_to_lane_change <= -0.45m`、局部中心线偏移峰值/下降
  或紧邻 R-E2 边界支持“准备回目标/原车道”时才写 R-E2；起点可比明显回正早 1-2 帧。
  一旦进入恢复 R-E2，后面 4 帧以内的短 U-E2 反跳视为距离/投影抖动并合入 R-E2，
  不允许出现 `R-E2 -> U-E2` 的恢复段回跳。
  R-E2 退出稍微提前：回到 `1.10 * route_center_tolerance` 且未来 2 帧稳定、无
  signed lane-change active 时释放为当前道路 regular event，避免恢复段尾部粘滞。
  2026-07-04 RGB 审计后，U-E2 短 gap 合并放宽到 6 帧，并允许障碍绕行过程中短暂
  R-E4/R-E5 投影边界被合并回 U-E2。
  若 U-E2 远离 XML trigger/具体障碍且无 route-change 轨迹，会释放为常规事件；路线末尾仍为
  U-E2 时写入 event review。
- `U-E2` / `U-E3` 是非路口直道核心异常，每条 route 最多保留一次证据最强的可信 span；
  `U-E2` 的保留优先看具体事故/施工/停放障碍距离、绕障/回正轨迹和 route 中心线偏离，
  `*_TwoWays` 还必须把 R2 借对向车道绕障核心本身视为强 U-E2 证据，不能只等同向障碍距离
  触发后才开始 U-E2；不能让前方运动车辆跟车距离或普通减速抢占唯一 U-E2 名额；`U-E3` 的保留优先看
  `dist_to_cutin_vehicle`、`brake_cutin`、`vehicle_hazard` 和对象进入自车未来路径，
  不能让普通跟车减速或红灯等待抢占唯一 U-E3 名额。
  反过来，`*_TwoWays` 的 `U-E2/U-E3` 也不能只靠 XML trigger 或旧 R2 候选提前开始：
  当前帧必须仍有最终 R2、具体障碍核心距离、TwoWays core/stuck/hazard 或强 R2 core rule。
  真正借/占对向车道之前保持当前道路 regular；越过核心后回目标/原车道才进入 R-E2。
  若 `U-E2/U-E3` 核心已经结束，且后续 24 帧内仍有回目标/原车道变道证据，后处理必须接入
  `R-E2`，不能继续粘在 `U-E2/U-E3`。核心后 16 帧内出现的弱 `U-E4` 只有在近距离行人/骑行者
  或 walker/emergency hazard 证据成立时才保留；否则按恢复变道改回 `R-E2`。
  TwoWays 单次 U-E2 span 评分以最终 R2 重叠为硬优先；后处理若先切出短 R-E2/R-E1 再发现两侧仍是
  同一借道绕障核心，会二次合并回 U-E2，避免 `U-E2 -> R-E2/R-E1 -> U-E2` 的假断裂。
  R4/R5 路口、有效红绿灯或红灯等待帧不允许继续保持 `U-E2/U-E3`；这类帧会释放为
  `R-E4/R-E5`（或当前道路常规事件），避免把等红灯、路口排队、路口起步误当成绕障/切入。
  EVENT route postprocess 最后会再次检查候选池一致性：任何后续规则若重新写出
  `R4/R5 + U-E2/U-E3`，都会被强制回 `R-E4/R-E5` 并记录原因。
  反向也必须严格：同向障碍/默认/noScenarios 场景只有 meta/bbox 灯态和 strong control context 同时成立才升 R4；
  缺少路口/stopline/signal-junction 同源证据时，瞬时 `traffic_light` 只作弱证据，primary 保持 R1，
  避免 R4 候选池误删 U-E2/U-E3。
  同向障碍的 `U-E2` 不能只靠 XML trigger 附近普通前车减速或刹车响应触发；若
  `dist_to_accident/construction/parked_obstacle` 为 inf、`scenario_obstacles_ids` 为空、
  active scenario 未触发，且 ego 仍稳定在 route 中心线，则保持当前道路 regular event。
  若前一轮过度保守把稳定灯控路口压成 R1，只允许 route 级 `r4_context_recovery` 恢复：
  必须是连续不少于 4 帧的灯态/bbox traffic_light 片段，并且片段内有
  `strong_control_context`、`close_trigger_for_junction` 或 bbox junction hint；弱
  `near_junction` / 宽 `junction_window` 只能保留 review，不能把长直路整段恢复成 R4。
  同向障碍 route 末尾若持续存在有效 `traffic_light_state + bbox traffic_light` 且 static signal
  已进入近距离范围，可窄召回为 R4/R-E4；这只用于真实稳定灯控路口，不用于环岛/弯道内的瞬时 light_hazard。
  若 `U-E2/U-E3` 后即将进入恢复/目标变道 `R-E2`，中间 4 帧以内的短 `R-E1`
  视为边界抖动并合入 `R-E2`，不允许出现 `U-E2/U-E3 -> R-E1 -> R-E2` 的断裂。
  对事故/施工/停放/side hazard/开门等障碍恢复类，`U-E2 -> R-E1 -> R-E2`
  的桥接窗口放宽到 8 帧；反过来，如果 R-E2 离最近 U-E2 超过约 6 秒（24 帧），
  视为后段弯道/跟车/投影扰动，释放回当前道路 regular event。
  `AccidentTwoWays` 对 `U-E2 -> R-E2 -> 短 R-E1 -> R-E4/R-E5/route-end` 还会做尾段桥接：
  这段短 R-E1 仍按借对向绕障后的回目标/原车道处理为 R-E2，避免恢复尾段过早变成普通跟车。
  EVENT 最终一致性检查之后还会再执行一次最多 2 帧的 `U-E2/U-E3 -> R-E1 -> R-E2`
  桥接，但跳过真实 R4/R5 帧，防止 RS 平滑把已完成恢复变道的边界重新打断。
  如果 `AccidentTwoWays` 的 R2 核心刚好叠在 R4/R5 路口跟前，ROAD_STRUCTURE 可以保持
  R4/R5 主标签，但 EVENT 层按 R2 overlay 处理：U-E2/R-E2 优先于 R-E4/R-E5，不能让路口常规事件吃掉借道绕障事件。
  对 `Town07_route_001454` 这类 XML/meta 强核心，R2 primary 还会反过来压过 R4/R5：
  即使 route projection error 让 two-way window 失效，只要近距离事故障碍、stuck 或 vehicle_hazard 仍在，
  就保持 R2/U-E2，不能因 bbox traffic_light 或路口控制源提前释放。
  非 TwoWays 同向障碍还会检查 `dist_to_*` 最近点：经过最近障碍点后若距离开始回升且
  ego-frame route 已稳定回中心线，后段即使仍在距离阈值内也释放为常规；若仍在回正轨迹中，
  只保留很短的 R-E2，避免“绕完以后才开始 U-E2”或 U-E2 尾段粘滞。
- `HardBreakRoute`：U-E1 不再仅凭近距离 lead/active window 触发，必须有 ego hard decel，
  或近距离低速 vehicle hazard，避免 XML 窗口覆盖过长。
- `ParkingCutIn` / `StaticCutIn`：U-E3 必须有 cut-in 距离、`brake_cutin` 或
  `vehicle_hazard`，不再仅凭普通急减速触发。若已经进入 R-E2 目标/恢复变道，
  后面 4 帧以内的 U-E3 反跳，或中间只夹 1-2 帧常规事件再跳 U-E3，都视作 cut-in
  证据抖动并合入 R-E2。
  U-E3 也按最近点截尾：`dist_to_cutin_vehicle` 过最近点后若对象开始远离且没有持续
  `brake_cutin` / `vehicle_hazard`，后段释放为 regular；若自车还在回正/目标变道，只保留短 R-E2。
- `InvadingTurn`：U-E5 必须有对向车辆 hazard 或近距离对象证据，不能仅凭自车减速触发；
  route 级后处理会合并 5 帧以内的 U-E5 短断点。
- `OppositeVehicleRunningRedLight`：U-E6 必须发生在 R4 路口窗口且有冲突车/近距离对象/自车响应；
  route 级后处理会合并 5 帧以内的 U-E6 短断点。
- `CrossJunctionDefectTrafficLight`：U-E7 是主事件；有冲突车辆时把 U-E6 放入同帧
  secondary `events`，避免 U-E7/U-E6 primary 来回抢占。
- 行人/自行车横穿按场景区分时机：`ParkingCrossingPedestrian` 可在进入路口前由近距离
  pedestrian/walker hazard 触发；`VehicleTurningRoute*` 只在转弯/驶出后对象进入交互范围时触发。

每条 route 还会先执行 TwoWays 事件型 R2 裁剪和 R2 碎片过滤，再执行统一时序去抖：`*_TwoWays` 的
ROAD_STRUCTURE 候选不再包含 R1。R2 表示有效可行驶通道为对向单车道：黄中心线窄路、乡路，
以及四车道但两侧停车/障碍/开门风险导致侧向 lane 不可行驶的等效双向单车道，都属于 R2。
障碍核心由 U-E2/U-E3 表达，核心后回目标/原车道由 R-E2 表达；非路口常规行驶仍保持 R2/R-E1。
真实灯控路口覆盖为 R4/R-E4，STOP/无灯/路权路口覆盖为 R5/R-E5，并在 route 摘要写
`twoways_core_span_clipping.changes`。若同一条 TwoWays route 中出现多个无拓扑支撑的临时 R2 片段，只保留最长连续
事件型 R2 段，其它偶发短 R2 扰动按当前控制源回 R2/R4/R5，并写入
`twoways_longest_r2_filter.changes`。随后 R2/R3/R4/R5 短于 4 帧、R1 短于 2 帧的孤立片段
会并回邻近稳定片段，去抖原因写入 `evidence.temporal_smoothing`，route 摘要写入
`temporal_smoothing.changes`。这条规则适用于全部 RS，不只是 R1/R4；其中 R4/R5
路口/T 形路口如果只出现 2-3 帧，即使有瞬时灯态、bbox traffic_light 或 XODR junction
命中，也视为扰动而不是一次真实路口经过。

Web 可视化页面也按这个口径展示：顶部绿色标签是
`frame_rs_annotation.label` / `primary_road_structure`，置信度对应这个“本帧最终 RS
标签”，不是候选全集的置信度；候选全集单独显示为“该场景全部候选 RS”。页面下方的证据归因
会同时列出 XML/route 进度、LEAD meta 动态字段和 XODR topology 摘要，用来判断该帧标注
到底由哪类证据触发、是否需要人工复核。新版 Web 左侧会显示该 scenario 下候选
RS/EVENT 的中文含义；右侧会分别展示本帧主 RS 和本帧主 EVENT，`events` 是同帧事件集合，
`primary_event` / `frame_event_annotation.label` 才是训练和验收时的主事件。
如果某条 route 因 `metas/*.pkl` 或 XML 缺失被跳过，可视化 summary 会显示
`status=data_missing_skip`、`manual_rgb_review_status=skipped_data_missing` 和具体
`skip_reasons`；这类 route 不生成逐帧 sheet，不参与错帧率统计。

调参时可传入规则覆盖文件，不需要直接改代码：

```bash
python keyframe_filter/quick_start.py annotate-rs \
  --scenario HighwayExit \
  --max-routes 3 \
  --max-frames-per-route 120 \
  --rule-config-json /tmp/rs_rule_overrides.json
```

覆盖 JSON 格式：

```json
{
  "scenarios": {
    "HighwayExit": {
      "merge_pre_m": 45,
      "merge_post_m": 55,
      "trigger_close_m": 85
    }
  }
}
```

Smoke test 口径：

- `python -m py_compile keyframe_filter/collector.py keyframe_filter/quick_start.py`
- `python keyframe_filter/quick_start.py annotate-rs --scenario T_Junction,AccidentTwoWays,ParkingExit,HighwayExit,noScenarios --max-routes 1 --max-frames-per-route 40 --output-dir /tmp/automot_rs_annotation_smoke`
- `python keyframe_filter/quick_start.py annotate-rs --scenario all --max-routes 1 --max-frames-per-route 10 --output-dir /tmp/automot_rs_annotation_all_smoke`
- `python keyframe_filter/quick_start.py annotate-rs --scenario all --samples-per-town 5 --max-frames-per-route 120 --output-dir /tmp/automot_event_rules_all_town5_120`

2026-07-04 最终收口 smoke：

- 语法检查：`python -m py_compile AutoMoT/keyframe_filter/collector.py AutoMoT/keyframe_filter/quick_start.py AutoMoT/keyframe_filter/web_app.py AutoMoT/keyframe_filter/rs_full_frame_review.py AutoMoT/keyframe_filter/frame_annotation_logic.py`
- 全 43 场景轻量回归：`python AutoMoT/keyframe_filter/quick_start.py annotate-rs --scenario all --max-routes 1 --max-frames-per-route 10 --output-dir /tmp/automot_rs_event_all10_final_smoke`
- 程序化一致性检查通过：每帧 `primary_event` 都在当前 scenario 的 `SCENARIO_TO_FINE_EVENTS`
  白名单内，`frame_event_annotation.label == primary_event`，`frame_rs_annotation.label == primary_road_structure`。
- Web smoke：Flask test client 访问 `/` 和 `/api/scenarios` 均返回 200，`/api/scenarios`
  会返回 `road_structure_labels` 与 `event_labels`，页面可解释 RS/EVENT 代号含义。

2026-07-04 EVENT 规则回归：`--scenario all --samples-per-town 5 --max-frames-per-route 120`
覆盖 43 个 scenario、993 条 route、91320 帧。常规场景
`ControlLoss/DynamicObjectCrossing/EnterActorFlow*/HighwayCutIn/HighwayExit/MergerIntoSlowTraffic*/RedLightWithoutLeadVehicle/SignalizedJunction*/T_Junction/noScenarios`
未产生 U-E；障碍类、急刹、行人/自行车、缺陷灯、闯红灯、InvadingTurn 均命中各自白名单
U-E。针对高切换 route 又回归
`InvadingTurn,CrossJunctionDefectTrafficLight,Accident,ConstructionObstacle,HazardAtSideLane`
共 130 条 route、12102 帧，确认 `CrossJunctionDefectTrafficLight` 的 primary 稳定为 U-E7，
U-E6 仅作为 secondary 事件；`ParkingCutIn/StaticCutIn` 不再因普通减速误触发 U-E3。

2026-07-06 RS 更新后的 EVENT 同步回归：

- 全量：`python AutoMoT/keyframe_filter/quick_start.py annotate-rs --scenario all --output-dir /tmp/automot_event_rs_sync_all_20260706`
  覆盖 43 个 scenario、8614 条 route、1062401 帧，`primary_event` 越过当前
  `primary_road_structure` allowed events 的违规数为 0。
- 当前代码 smoke：
  `python -m py_compile AutoMoT/keyframe_filter/collector.py AutoMoT/keyframe_filter/quick_start.py`
  以及高风险场景
  `EnterActorFlow,EnterActorFlowV2,HighwayExit,HighwayCutIn,MergerIntoSlowTraffic,MergerIntoSlowTrafficV2,AccidentTwoWays,ParkedObstacleTwoWays,ParkingCutIn,StaticCutIn,VehicleOpensDoorTwoWays,InvadingTurn`
  的 36 route / 3760 帧小样本，allowed events 违规数同样为 0。
- R3 regular event 明确为 R-E1，R-E3 只贴 merge/exit/actor-flow core；
  全量中 `HighwayCutIn` 为 R-E1 98.4%，`HighwayExit` 为 R-E1 79.3% / R-E3 17.0%，
  `EnterActorFlow*` 与 `MergerIntoSlowTraffic*` 均不再被压成全程 R-E3。
- `*_TwoWays` 候选 RS 删除 R1；`AccidentTwoWays` 全量为 R2 80.8%，EVENT 保留
  U-E2 29.0% 与 R-E2 17.4%。如果 R2 core 与 R4/R5 路口边界重叠，EVENT 层使用
  R2 overlay，U-E2/R-E2 优先于 R-E4/R-E5。
- 详细场景分布与命令见 `ROAD_EVENT_RS_SYNC_AUDIT_20260706.md`。
- 旧 R6 删除后的高风险停车/开门/停放障碍场景已重跑全量逐帧审计，详见
  `ROAD_EVENT_NO_R6_RGB_AUDIT_20260706.md`：7 个场景、895 条可标注 route、106471 帧；
  runtime 枚举、候选池和本轮输出中均无 R6 类别，RS 候选越界 0，当前 RS allowed events 越界 0。
- 少量 `U-E1/U-E2/U-E3/U-E4` 被 R4/R5 突然接管的边界现在走 interrupted overlay，详见
  `ROAD_EVENT_INTERRUPTED_OVERLAY_AUDIT_20260706.md`。primary RS 保持 R4/R5，但 EVENT 同帧保留
  `R-E4/R-E5 + U-E*` 或 `R-E4/R-E5 + R-E2`；总叠加最长 24 帧，恢复 `R-E2` 子阶段最长 12 帧。
  U-E4 中距离横穿/转弯冲突只给 10 帧短续，防止普通路口被长期污染。2026-07-07 全量 R1
  突发事件场景审计覆盖 3552 route / 526001 帧；在修复同向障碍直道 stop/yield 伪 R5 后，
  触发 99 route / 1472 帧。

2026-07-03 第一轮全帧 RGB 复核回灌后 smoke 结论：43 场景小样本均可生成逐帧标注；
静态 XODR 模式下 R2/R3 若缺少 opposite/merge 局部拓扑，不再仅凭
scenario/trigger 窗口压过 R1，而是降为 secondary/review；
`noScenarios` 已调成只有 meta 有效灯态或 light hazard 时才允许 R4，否则保守 R1；
`StaticCutIn` 无 R3 merge/highway 证据时回 R1 中置信；2026-07-06 全量逐帧 RGB 未见稳定独立停车结构，候选已收紧为 R1/R3/R4/R5。
当前 43 场景 × 10 帧 smoke 分布为
`R1=192, R2=10, R3=20, R4=128, R5=80`，
`confidence min/avg/max = 0.70/0.8302/0.98`，`review_ratio=0.3093`。

全帧复核必须按 RGB-first 执行：每个 scenario 的每个 town 抽 1 条 route，完整生成
`all_frames_*.jpg`，人工从 f0 到最后一帧逐帧看图。summary、置信度、标签分布和
`candidate_anomalies` 都只是索引；稳定高置信 R1 也不能跳过。若 RGB 清晰显示
merge/parking/junction/TwoWays 等特殊结构而标签仍为 R1，必须记录为视觉冲突，再查
rules/evidence 判断是规则思路错误、参数阈值错误还是 XML/XODR 投影错误。只有低能见度
或遮挡严重时才把 XODR/XML/meta 作为主要补充证据。

RGB-first 全量复核入口：

```bash
python keyframe_filter/rs_full_frame_review.py \
  --scenario all \
  --samples-per-town 1 \
  --max-routes-per-town 1 \
  --max-frames-per-route 0 \
  --frames-per-sheet 40 \
  --sheet-cols 5 \
  --output-dir keyframe_filter/collection_output/rs_full_frame_review_rgb_first_current
```

当前 RGB-first 闭环覆盖 43 个 scenario、204 个 scenario-town route、24387 帧；
`candidate_anomalies=15788` 只是逐帧看图索引，不是错帧数。最终异常必须读取每个 route 的
`all_frames_*.jpg` 后人工确认，并按下面的根因归类回灌规则。

已确认导致标定与真实 RGB 不匹配的主要原因：

- `route_projection_error_high` 时仍把 XML route_s / trigger window 当 hard boundary，导致普通路段被过早标成 R4/R5/R3。
- 静态 XODR signal/opposite/parking/merge/junction hint 与 RGB 不同源，尤其在雾、夜间、稀疏 route 或投影偏移时会误导规则。
- 只按 scenario 名称或 active scenario 延长特殊 RS，导致事件已经结束或 merge/路口已经离开后仍保持 R2/R3/R4/R5。
- 反向问题也存在：明显高速/merge 或双向单车道因 XODR 投影失败被压回 R1；坏 XODR 不能当作否定证据。
- `two_way_layout_prior` 不能只靠场景名把整条 route 升成 R2；但一旦 XODR/meta 确认是双向单车道，
  正常直道也应为 R2。是否处在借对向绕障核心由 EVENT 的 `U-E2/U-E3` 决定，而不是靠 RS 从 R1/R2 切换表达。
- 高速/merge/exit/enter-flow 场景此前被 R1 默认桶吃掉；全量逐帧 RGB 确认
  `HighwayExit`、`EnterActorFlow*`、`MergerIntoSlowTrafficV2` 稳定按 R3 收敛且不开放 R1/R4。
  `HighwayCutIn` 与 `MergerIntoSlowTraffic` 主体仍是 R3，但存在少量真实灯控子集，因此恢复 R4 候选；
  R4 必须由逐帧 RGB/meta/bbox 灯控证据触发，匝道/导流线/停车线不能单独制造 R5。
- 行人、事故、施工、急刹、切入、开门、闯红灯等多数是 EVENT，不应直接改变 ROAD_STRUCTURE；RS 必须由道路几何和控制源决定。
- 置信度和 summary 只能作为索引。高置信不等于 RGB 正确，稳定 R1 也必须逐帧看，避免漏掉后段 merge、停车带、路口或双向路结构。

第二轮错配回灌后，`collector.py` 已按图像优先结论收紧：

- 弱 R2/R3/R4/R5 候选不能通过 priority tie-break 低分压过 R1。
- `route_projection_error_m > 5m` 时，普通 `scenario_active` / `trigger_close_m`
  只作为 review 线索，不再单独撑起 two-way / merge / parking / junction 窗口。
- 静态 XODR 的 signal/opposite/parking/merge/junction hint 在高投影误差帧降级为
  `*_demoted_projection_error` 证据；nonsignalized 场景遇到静态 signal 会写
  `nonsignalized_with_signal_topology_conflict`，需要人工结合 RGB 确认。
- `MergerIntoSlowTraffic*` 的明显 merge 口不能被坏 XODR 自动压回 R1；当 RGB/LEAD XML
  已支持合流，但 route/XODR 投影误差导致 topology 不可信时，规则会使用 XML
  `start_actor_flow/end_actor_flow` 强近邻和 trigger 距离作为 fallback，
  写入 `r3_merger_actor_flow_or_trigger_fallback`，主标签可为 R3，同时保留 review 原因。
  `EnterActorFlow*`、`HighwayExit`、`MergerIntoSlowTrafficV2`
  已从候选池删除 R1/R4；`HighwayCutIn`、`MergerIntoSlowTraffic` 删除 R1 但保留少量 R4 子集，
  且 R4 只由逐帧灯控同源证据触发；active scenario 只作为审计证据，不单独制造 R4/R5。
- 人工逐帧看图后又补了两条更强的图像优先门控：
  静态 signal 或灯态只有在 `is_junction` / 可信 XODR junction / stopline / 近距离 signal-junction
  上下文成立时才给 R4 high；否则回 R1 + review。
  `collector.py` 会读取同帧 `bboxes/*.pkl` 的轻量语义摘要：`traffic_light` 可辅助 R4，
  `stop_sign/yield/junction/crosswalk` 可辅助 R5；但高速/merge 场景中缺少同源控制上下文的灯控 hint
  会被降级为 review 弱候选，不压过默认 R3。
  TwoWays 的 R2 high 分两类：XODR/meta 确认双向单车道时，正常直道也保持 R2；
  若拓扑不确认，则只能由近距离障碍、stuck、vehicle_hazard、lane-change 核心证据，
  或 XML trigger 极近 / trigger-close + XML 场景障碍近距离临时召回事件型 R2。
  `twoways_obstacle` 的纯场景名 layout-prior 仍降为弱候选；是否处在借对向核心由 U-E2/U-E3 表达。
  输出 evidence 里同步写入 `strong_control_context` 和 TwoWays 专用
  `twoway_obstruction_evidence`，用于区分规则思路问题、参数窗口问题和底层证据缺失。

修正后 smoke：

```bash
python -m py_compile keyframe_filter/collector.py keyframe_filter/quick_start.py keyframe_filter/rs_full_frame_review.py
python keyframe_filter/quick_start.py annotate-rs \
  --scenario AccidentTwoWays,InterurbanActorFlow,MergerIntoSlowTrafficV2,NonSignalizedJunctionLeftTurn,ControlLoss,ParkingExit,VehicleTurningRoute \
  --max-routes 1 --max-frames-per-route 80 \
  --output-dir /tmp/automot_rs_annotation_visual_fix_smoke3
python keyframe_filter/quick_start.py annotate-rs \
  --scenario all --max-routes 1 --max-frames-per-route 10 \
  --output-dir /tmp/automot_rs_annotation_all_visual_fix_smoke
```

全 43 场景 × 10 帧 smoke 分布为
`R1=174, R2=10, R3=70, R4=118, R5=58`，
`confidence min/avg/max = 0.70/0.8360/0.98`，`review_ratio=0.5256`。
代表性错配路线里，`Accident/Town03` 无清晰路口/灯控画面时尾段 R4 回 R1；
`AccidentTwoWays` 当前按有效可行驶通道判 R2，旧 `R1 -> R2 -> R1 -> R4` 边界已作废；
非路口 TwoWays 直道保持 R2，只有真实灯控/STOP/无灯控制源才覆盖为 R4/R5；
本轮 TwoWays smoke 覆盖 Accident/Construction/Hazard/Parked 四类各 5 条 route，均能召回核心 R2；
`HighwayExit/EnterActorFlow*/MergerIntoSlowTrafficV2` 不再被 R1/R4 吃掉，局部复核输出均为 R3；
`HighwayCutIn/MergerIntoSlowTraffic` 主体仍是 R3，但保留逐帧灯控子集 R4；
`HardBreakRoute` 抽样显示城市/乡村/快速路混合，已改成 route 级分桶：
高速 route 候选收敛为 R3/R4，非高速 route 保留 R1；
`PriorityAtJunction` 虽在 Town12/13，但不按高速处理；全量 RGB 显示其同时包含真实灯控城市十字路口和无灯/让行段，
因此保持 R1/R4/R5 混合候选。

逐场景 RS 调研产物生成：

```bash
/home/codon/anaconda3/envs/carla/bin/python keyframe_filter/rs_research.py --samples-per-town 5
```

输出到：

```text
keyframe_filter/collection_output/rs_research/<Scenario>/
```

该入口会为每个 scenario 覆盖全部 town，每个 town 优先抽 5 条有 meta 的 run，生成
XML/XODR/meta/RGB/map/rules 证据链。若某个 town 的真实 run 缺 meta，会在
`scenario_audit.json` 与 `rules/scenario_rule_design.md` 中保留 incomplete 原因。
所有 LEAD run 会先经过异常时长硬过滤：4Hz 下 `rgb/*.jpg >= 361`
（严格大于 1 分 30 秒）且不在 `BlockedIntersection/ControlLoss` 白名单内的 route
不得进入调研、标注或 probe。若某个 town 在过滤后没有任何可读 meta run，就记录缺口并跳过；
其它 town 仍必须抽 5 个分散 id 或全部可读 run。

这一步只表示自动证据包生成完成，不表示 RS 规则已经可直接标 complete。每个 scenario
还必须有自己的机制假设和人工复核记录：

- 先列出该 scenario 涉及的全部 town，并确认每个 town 至少抽到 5 条分散 run
  （不足 5 条则全读可读 run）。
- 对每条 sampled run，核对 XML route/trigger、XODR 局部拓扑、自车全局坐标历史轨迹、
  meta 边界字段和 RGB contact sheet。
- 重点看 `maps/*route_trigger_ego_trace.png` 与 `rgb/*sample_contact_sheet.jpg`
  是否对齐；不对齐时先记录 `projection_untrusted` / `rgb_conflicts_with_map`，
  不要直接调阈值。
- 把每个 scenario 的 approach / pre-trigger / core / exit 分段逻辑写进
  `rules/scenario_rule_design.md`，把阈值来源写进 `rules/thresholds.json`。
- 只有 `manual_map_rgb_checked=true`、`thresholds_have_provenance=true`、
  `runtime_rule_ready=true` 后，代码才能把该 scenario 的
  `complete_investigation_status.is_complete` 标为 true。

生成后不要直接把 `complete=True` 当成最终规则可用。每个 scenario 还必须人工检查：

- `maps/*route_trigger_ego_trace.png`：XML route、trigger、自车历史轨迹是否贴合。
- `rgb/*sample_contact_sheet.jpg`：RGB 中可见道路结构是否和 XODR/XML 判断一致。
- `rules/thresholds.json`：每个阈值是否写明来源、支持 run 和 reviewed artifacts。
- `rules/failure_modes.md`：是否列出 XML/XODR/meta/RGB/仲裁五类失败模式。

只有这些检查被写回 scenario README/rules 后，后续 RS 代码才可以把该 scenario 标成
`complete_investigation_status.is_complete=true`。

错帧回查也必须从调研包开始：先看 scenario README 和 map/RGB 对齐记录，再看
XML/XODR/meta frame features，最后才看阈值和运行时代码。这样用户指出某一帧有问题时，
可以判断到底是 XODR 没用好、XML 匹配/投影不可信、meta 边界字段缺失、RGB 与地图冲突，
还是仲裁优先级和阈值本身需要改。

如果真实 LEAD 数据不在默认 `lead_data`，用环境变量指定：

```bash
LEAD_DATA_ROOT=/path/to/lead_data python keyframe_filter/quick_start.py
```

Web 视频目录和输出目录也可覆盖：

```bash
LEAD_VIDEO_ROOT=/path/to/lead_video \
KEYFRAME_COLLECTION_OUTPUT=/path/to/output \
python keyframe_filter/quick_start.py
```

启动 Web 后，页面右侧会展示：本帧最终 RS 标注、本帧 EVENT 标注、该 scenario 的候选
RS/EVENT 全集，以及 XML / LEAD meta / XODR 证据归因。绿色主标签才是当前 frame 的最终
`frame_rs_annotation.label`；红色 EVENT 主标签才是 `frame_event_annotation.label`；
候选全集不是标注结果。

Web 不需要为了同一输出目录里的新结果而重启：后端每个 API 请求都会重新读取
`KEYFRAME_COLLECTION_OUTPUT` 下的 JSON，并给响应加 `Cache-Control: no-store`。跑完新的
`annotate-rs` 后，页面上点击“刷新标注结果”即可重读当前 scenario/route/frame 的标注；
整页浏览器刷新也可以。如果新结果写到了另一个 `--output-dir`，需要用新的
`KEYFRAME_COLLECTION_OUTPUT=/path/to/output` 重新启动 Web，或把命令输出写回当前 Web
绑定的目录。

---

## 菜单功能

采集模式：

1. 单场景全部采集 + 逐帧 RS/EVENT 标注
2. 单场景指定数采集 + 逐帧 RS/EVENT 标注
3. 多场景采集 + 逐帧 RS/EVENT 标注
4. 全部采集 + 逐帧 RS/EVENT 标注

其他功能：

5. 多角度结构分析
6. 启动 Web 应用
7. 显示所有场景
8. ROAD_STRUCTURE XML/XODR 画像
9. 逐帧 RS/EVENT 标注 smoke / 参数闭环调试入口
10. 退出

`ROAD_STRUCTURE XML/XODR 画像` 会逐 scenario 遍历所有 town，每个 town 默认抽 5 个 XML，
并记录 XODR 是否存在、junction/signal/controller 粗统计、waypoint 数和 scenario tag。
输出：

```text
keyframe_filter/collection_output/road_structure_xml_xodr_audit.json
```

---

## Python API

```python
from collector import ScenarioCollector

collector = ScenarioCollector(
    lead_data_root="lead_data",
    output_dir="keyframe_filter/collection_output",
)

result = collector.collect_one_scenario("Accident", max_routes=5)

if result["status"] != "success":
    print(result["error"])
else:
    print(result["total_frames"])
```

`ScenarioCollector()` 默认等价于：

```text
lead_data_root = AutoMoT/lead_data
output_dir = AutoMoT/keyframe_filter/collection_output
xml_root = AutoMoT/data/lead
carla_root = AutoMoT/CARLA_0915
```

也可以用环境变量覆盖：

- `LEAD_DATA_ROOT`
- `KEYFRAME_COLLECTION_OUTPUT`

---

## 输出结构

场景级结果：

```json
{
  "scenario": "Accident",
  "status": "success",
  "road_candidates": ["R1", "R4"],
  "event_candidates": ["R-E1", "R-E2", "R-E4", "U-E2"],
  "total_frames": 1234,
  "routes": []
}
```

帧级结果保留旧字段，并新增主 RS 字段：

```json
{
  "frame_id": 80,
  "road_structures": ["R1", "R4"],
  "events": ["R-E1", "R-E2", "R-E4", "U-E2"],
  "primary_road_structure": "R4",
  "secondary_road_structures": [],
  "road_structure_candidates": {"R1": 0.35, "R4": 0.95},
  "annotation_comment": "R4：规则族=signalized_junction，来源=meta_traffic_light，置信=0.96...",
  "evidence": {
    "rules_fired": ["r1_default_candidate", "r4_tl_confirmed"],
    "rule_kind": "signalized_junction",
    "xml_path": "data/lead/Accident/...",
    "route_progress_m": 42.5,
    "xodr": {
      "xodr_source": "static_xodr",
      "xodr_topology_trusted": true
    },
    "review_required": false,
    "review_reasons": []
  }
}
```

route 级结果还会写入：

- `primary_rs_distribution`：该 route 内 primary RS 计数。
- `review_required_frames` / `review_reason_distribution`：需要人工回查的帧数与原因。
- `primary_rs_transitions`：最多保留前 50 个 primary RS 切换帧，便于检查边界抖动。

如果数据目录不存在，采集器会返回明确错误，不再触发 `total_frames` 二次异常：

```json
{
  "scenario": "Accident",
  "status": "error",
  "error": "场景目录不存在: .../lead_data/Accident",
  "total_frames": 0
}
```

---

## ROAD_STRUCTURE 口径

| ID | 含义 |
|---|---|
| R1 | 常规道路 / 同向可行驶道路 |
| R2 | 双向单车道 / 对向车道参与决策 |
| R3 | 高速合流 / 匝道 / 分流 / 驶出决策结构 |
| R4 | 信号灯路口 |
| R5 | 无信号灯 / 信号灯失效 / 路权路口 |

规则实现来自：

- `ROAD_EVENT_CLASSIFICATION_PLAN.md`：ROAD/EVENT 语义、RS 调研协议、runtime 门控和错帧回查流程
- `ROAD_EVENT_CANDIDATE_MAPPING.md`：Qwen/probe 可解析的 scenario / ROAD_STRUCTURE / EVENT 候选表

核心约束：

- 每个 scenario 的 RS 规则必须先有独立调研文件夹，默认位于
  `keyframe_filter/collection_output/rs_research/<Scenario>/`；里面记录涉及 town、
  抽样 run id、XML 摘要、XODR 摘要、meta/RGB 摘要、自车轨迹与 trigger/route 地图可视化、
  置信度规则和失败模式。
- 调研时每个 scenario 必须覆盖它涉及的所有 town；每个 town 至少读取 5 条分散 run
  （不足 5 条可读 run 则全读），并把 XML route/trigger、XODR road/lane/junction/signal、
  LEAD meta 的全局轨迹和 RGB 对齐到同一张地图上。
- 后续改 RS 代码阈值前，必须先能在对应 scenario 的调研文件夹中找到证据；否则输出只能标为
  `complete_investigation_status.is_complete=false` 的临时规则。
- 运行时代码不得新增匿名 magic number；每个 scenario 阈值必须来自 `rules/thresholds.json`
  或显式 scenario config，并能回指到 maps/RGB/meta/XML/XODR 证据。
- 用户指出错帧时，先查该 scenario 的调研包：README -> map trace -> XML/XODR -> meta jsonl ->
  RGB contact sheet/boundary frame -> thresholds/code。不要直接调阈值。
- TwoWays 的双向单车道直道应输出 R2；如果该 route/片段并非双向单车道，才不能凭场景名输出 R2。
  `two_way_layout_prior` 只能作为弱候选，必须有 XODR/meta 对向单车道拓扑或核心障碍证据才可做 primary。
- Parking* 不生成独立停车 RS；`ParkingCutIn` 已按全量逐帧 RGB 收紧为 R1/R4/R5，灯控/无灯路口段分别优先 R4/R5。
- `CrossJunctionDefectTrafficLight` 强制 R5 覆盖 R4。
- `ParkedObstacle` 是障碍 EVENT，不是停车 ROAD_STRUCTURE；`ParkedObstacleTwoWays` 核心窗口才是 R2。
- `data/lead` XML 不能替代真实 `lead_data` 帧数据。

### 5-id/town 调研结论摘要

已按 `python keyframe_filter/rs_research.py --samples-per-town 5` 生成
`collection_output/rs_research/<Scenario>/`。本轮覆盖 43 个 scenario；除
`NonSignalizedJunctionLeftTurn/Town10HD` 本地没有可读 meta 外，其余 scenario 均已有
自动证据链。该状态是 `auto_artifacts_ready`，不是人工最终完成；每个 scenario 的
`maps/*route_trigger_ego_trace.png` 与 `rgb/*sample_contact_sheet.jpg` 仍需人工确认后，
才能把对应规则标成 final complete。

`collection_output/` 是本地自动调研输出目录，包含 map trace、RGB contact sheet、
meta/XML/XODR 摘要和中间 JSON。该目录默认不入库、不 push；后续需要共享的结论应整理进
本 README、ROAD_STRUCTURE/ROAD_EVENT 方案文档或小型规则配置。

规则族结论：

- `same_direction_obstacle`：`Accident`、`ConstructionObstacle`、`ParkedObstacle`。
  静态同向障碍是 EVENT 证据，不把整段升级成 R2；只在受控路口窗口进入 R4。
  `Accident` 前 30 帧如果只是被弱静态 junction/signal hint 误标成 R4/R5，会强制回 R1；
  若同帧存在有效 `traffic_light_state`、bbox traffic_light、STOP/yield 或明确 junction 控制源，
  说明 route 初始确实就在路口控制区，保留原 R4/R5。Town13 例外：按原始 XODR/meta/RGB
  证据保留 R2/R4/R5；被压回的帧会把 R-E4/R-E5 常规事件同步回 R-E1。
- `twoways_obstacle` / `invading_turn` / `vehicle_opens_door_twoways`：
  R2 表示有效可行驶通道为对向单车道；XODR/meta 确认黄中心线窄路，或确认多车道两侧停车/障碍/开门风险让侧向 lane 不可行驶时，
  正常直道也应为 R2。必须借/等对向的核心动作由 U-E2/U-E3 表达；
  `AccidentTwoWays` 前 30 帧同样只压制弱静态 junction/signal hint；如果初始帧已有真实
  灯控、STOP/yield 或 junction 控制源，保留 R4/R5。Town13 例外，应该是什么 ROAD_STRUCTURE
  就保留什么；其它 town 若被压回则强制回 R2，并把由十字路口产生的 R-E4/R-E5 常规事件同步回 R-E1。
  但在核心借对向/绕障已经发生、且同帧又靠近真实 R4/R5 控制区时，允许 R2 与 R4/R5
  在事件层叠加：道路主标签可为 R4/R5，事件仍优先输出 U-E2/R-E2。
  若 meta/XML 明确是强 R2 核心，则道路主标签也优先 R2，R4/R5 只能作为 secondary/overlay 信息。
  对 `*_TwoWays`，XML trigger 极近或 trigger-close + XML 场景障碍近距离可召回短核心 R2。
  道路布局层按有效可行驶 lane 数给 R2；R2 结束后仅在真实灯控/STOP/无灯路口证据成立时切 R4/R5。
  `InvadingTurn` 已按 2026-07-06 全量逐帧 RGB 审计保留 R1/R2/R4/R5：R2 表达对向占道/双向窄路，
  R4 只给稳定灯控子集，STOP/无灯路口窗口给 R5，对向侵占事件由 U-E5 表达。该场景的 XODR 静态 probe 多数帧 `map_junction_id=-1` 且
  `xodr_topology_trusted=false`，不能主要靠 XODR junction 召回十字路口；强 R5 短段和
  R5 间的短 R1 gap 会按 STOP/active/trigger 证据做专门去抖。
- `highway_merge`：`EnterActorFlow*`、`HighwayExit`、`MergerIntoSlowTrafficV2`
  候选删除 R1/R4，非路口默认 R3；`HighwayCutIn` 与 `MergerIntoSlowTraffic`
  删除 R1 但保留少量 R4 子集，R4 只由逐帧灯控同源证据触发；merge/split/ramp/actor-flow
  只提高 R3 置信与定位边界。
- 混合场景 route 分桶：`HardBreakRoute` / `interurban` / `StaticCutIn` / `ParkingCutIn`
  不能只按 Town12/13 判高速。先用 RGB sheet 把 route 分成高速/快速路桶和非高速桶；
  高速桶候选收敛为 R3/R4，非高速桶保留 R1。当前已逐 id 均匀 5 帧 RGB 复核：
  HardBreakRoute 97 个 route 中 16 个进高速桶，StaticCutIn 100 个 route 中 44 个进高速桶；
  InterurbanActorFlow 91 个、InterurbanAdvancedActorFlow 78 个、ParkingCutIn 99 个未发现高速桶。
  2026-07-04 又对所有 `lead_data` route 做全量逐帧 RGB 审计：
  `InterurbanActorFlow`、`InterurbanAdvancedActorFlow`、`InvadingTurn`、`NonSignalizedJunctionRightTurn`、
  `OppositeVehicleTakingPriority` 均以 STOP/无灯 junction/active close-trigger R5 为主，其中
  `NonSignalizedJunctionRightTurn` 与 `OppositeVehicleTakingPriority` 有少量灯控子集，保留 R4/R5 逐帧仲裁；
  `EnterActorFlow*`、`HighwayExit`、`MergerIntoSlowTrafficV2` 未见稳定真实灯控路口，保持纯 R3/no-R4；
  `HighwayCutIn`、`MergerIntoSlowTraffic` 保持 R3/R4，`PriorityAtJunction` 与 `T_Junction` 是灯控/无灯混合，保留 R4/R5。
  `Town12_Rep0_258_0_route0_01_08_09_35_42` 这类乡村普通路明确保留 R1。
  输出中 `evidence.route_semantic_bucket` 和 route 级 `route_semantic_bucket_distribution`
  会记录当前 route 走 `highway_rgb_route` 还是 `mixed_reviewed_non_highway`。
- `signalized_junction`：灯态有效、受控 junction 或 controller/traffic light 近邻成立时进入 R4；
  `BlockedIntersection` 和 `OppositeVehicleRunningRedLight` 的阻塞/违规只是 EVENT，不改成 R5。
  `BlockedIntersection` 已经形成稳定 R4 灯控片段后，若尾段仍在同一 junction/blocked context 内、
  但 `traffic_light_state` / bbox traffic_light 因视角丢失变空，且没有 STOP/yield 证据，继续保持 R4；
  不能仅因 `is_junction=True` 且灯态缺失把灯控路口出口误改成 R5。
  R4 不是只属于 signalized scenario：除 `CrossJunctionDefectTrafficLight`、
  已 RGB 确认无稳定信号灯路口的 no-R4 场景外，任意场景只要单帧有有效
  `traffic_light_state` / `light_hazard`，都会动态开放 R4 候选；但同向障碍/默认/noScenarios
  缺少 strong control context 时必须降回 R1，只保留证据与 review，不能让弱灯态压掉 U-E2/U-E3。
  如果 primary R4 不是由有效 `traffic_light_state` 支撑，而是由 junction/window/static signal
  支撑，必须写 `signalized_r4_without_meta_tl_requires_rgb_confirmation`，逐帧看 RGB 确认
  stopline/crosswalk/cross traffic/blocked pocket 是否仍可见，不能只按置信度放行。
  若只有 static signal 近邻 + 灯态而缺少 `is_junction`/XODR junction，strong context 距离阈值为
  25m；25-35m 只保留弱 R4 候选，避免在雾中普通路段过早覆盖 R1。
- `nonsignalized_junction` / `defect_junction`：无有效灯态、stop/yield/priority 或灯故障机制成立时进入 R5；
  `CrossJunctionDefectTrafficLight` 强制 R5 覆盖 R4。
- `parking` / `parking_exit` / `static_cutin`：parking/shoulder/curb/parking-exit 不再输出独立 RS；
  普通非路口为 R1，停车驶出/并入主路用 R-E2，停车车辆切入用 U-E3，行人横穿用 U-E4；
  两侧停车/开门压缩有效通道时可归 R2，停车相关 scenario 在信号灯/无灯路口段仍优先 R4/R5。
- `default_meta_map` / `noscenario`：默认 R1；ControlLoss、HardBreak、DynamicObjectCrossing、
  HazardAtSideLane 等行为/突发事件本身不改变 RS，只能通过灯态或 junction 证据临时进入 R4。
  本轮跨场景 R4 漏检审计覆盖 43 个 scenario 各 2 条 route：修复前大量 `Accident`、`HazardAtSideLane`、
  `ControlLoss`、`Parking*` 等帧命中 `r4_tl_seen_without_strong_junction_context` 但 primary 仍是 R1；
  修复后剩余 R1+灯控证据主要是 `noScenarios` 弱静态 XODR signal 或短 R4 片段被时序平滑。

本轮自动阈值是调研初值：`junction_pre_m=40~60`、`junction_post_m=20~40`；
运行时同时收紧十字路口进入和退出侧：effective pre = `0.36 * junction_pre`
（pre 最小 16m），effective post = `0.28 * junction_post`（post 最小 5m）。
`dist_to_junction_near=35m`，strong junction 上限 22m，static signal near 上限 35m，
close-trigger 上限 25m；进入/退出侧与辅助召回阈值都继续收窄，避免 R4/R5 过早吞掉正常接近/跟车阶段；
离开路口后也更快回普通道路。
XML `<weather>` 会进一步调节这些距离：夜间、低太阳高度、大雾等低能见度 route
在上述基准上再乘 `low_visibility_factor=0.65~0.95`，并允许 pre/post 保底同步降到
10m/3m。轻/中雾通常约 0.92，夜间约 0.78，重雾约 0.85，夜雾叠加最低 0.65。
单纯下雨不再压缩 R4/R5 范围；雨只在叠加雾、夜间或低太阳高度时轻微增强收缩，
避免 RGB 已经能看到红绿灯但 R4 进入过晚。
该收缩对所有场景生效，主要压缩 R4/R5 的 junction window、meta near、strong context、
static signal near 和 close-trigger 距离；不会单独改变 U-E2/R-E2/R3 等事件逻辑。
每帧 evidence 的 `junction_window_config` 会记录有效阈值、天气系数和触发原因。
`BlockedIntersection` 的十字路口窗口额外压缩 20%，基准为 `junction_pre_m=48`、
`junction_post_m=20`，阻塞本身只进 EVENT，不能扩大成整段 R4/R5。
但 static signal 近邻不是 R4 充分条件；只有有效 meta 灯态、stop/light hazard、
结构化 XODR junction 或 RGB 可见控制区同源时，才允许升为 R4。
`two_way_min_pre_m=50~80`（`*_TwoWays` 较前一版统一提前约 5m 开始召回，但核心证据和后段裁剪不变）、
`merge_pre_m=30~50`、`merge_post_m=40~50`、
`parking_pre_m=20~35`、`parking_post_m=50~60`。这些阈值必须在每个
`rules/thresholds.json` 中补齐 `supporting_runs/reviewed_artifacts/reason` 后才能作为正式代码依据。

---

## 故障排除

| 问题 | 处理 |
|---|---|
| `status=error` 且 `场景目录不存在` | 检查 `LEAD_DATA_ROOT` 或默认 `lead_data/<Scenario>` 是否存在 |
| `Routes数: 0` | 检查 scenario 目录下是否有 run 子目录 |
| 没有 `metas/*.pkl` | 当前 run 记为 `data_missing_skip/missing_meta` 并跳过；采集、调参、可视化不再跑规则 |
| XML 匹配不到 | 当前 run 记为 `data_missing_skip/missing_route_xml` 并跳过；先按 `(scenario,town,route_key)` 确认 `data/lead`，再按 `(town,route_key)` 全局查 `data_routes`；不能用错误 XML/XODR 降级硬跑 |
| XML 匹配到其它 Town | 先查 run id 的 Town 与 `xml_town` 是否一致；纯数字 route 兜底有跨 town 歧义时必须修索引或降级，不能继续用错误 XML/XODR 判断 R4/R5 |
| 没有 carla Python API | XODR 查询自动降级到静态 planView/lane/signal 近邻；若 `xodr_topology_untrusted` 很多，优先检查 XODR 坐标系/地图路径 |
| Web 看不到结果 | 确认 `collection_output/*_result.json` 已生成 |
| Web 只有候选没有本帧标签 | 重新打开新版 `web_app.py`；绿色“本帧最终标签”来自 `frame_rs_annotation.label`，置信度对应这个标签 |
| Web XODR 显示 `trusted=false` | 说明静态 XODR 投影或局部拓扑不足以 high confidence；优先用 CARLA Python 环境重跑并对比 review 是否下降 |
| 环岛被标成 R4/R5 | 检查 Web XODR 摘要是否有 `roundabout=true`；若没有，说明该 town 的 XODR 环岛几何特征需要补 probe 规则 |
| 单帧 R4/R5/R2/R3 抖动 | 检查 route 摘要 `temporal_smoothing.changes`；短片段默认会被并回邻近稳定 RS |
| TwoWays 出现多个 R2 碎片 | 查 `twoways_longest_r2_filter.kept` 和 `changes`；有有效可行驶对向单车道证据的 R2 不应被裁掉，只有无拓扑支撑的临时事件型 R2 碎片会按控制源回 R2/R4/R5 |
| TwoWays 核心仍全 R1 | 候选池应已删除 R1；查 `allowed_road_structures`、`twoway_layout_prior_allowed`、`twoway_obstruction_evidence` 和 `trigger_distance_m` |
| TwoWays 绕障后红绿灯段仍 R2 | 查 `rules_fired` 是否有 `twoways_post_core_meta_tl_r4` 或 `twoways_post_core_xodr_signal_r4`；没有则继续核查 meta 灯态、XODR `nearest_signal_m`、`map_is_junction` 与 RGB 灯控是否同源 |
| TwoWays 普通山路误标 R4 | 先看 RGB contact sheet；若无路口/灯控，检查 XML 是否跨 Town 误配，以及静态 XODR 是否只有 `map_junction_id=-1` / `junction_connection_count=0` 的弱 junction hint |

---

## RGB 盲审对账

新增 `rgb_blind_rs_event_audit.py` 用于“先看 RGB 自己判断，再看当前标签”的闭环：

```bash
python AutoMoT/keyframe_filter/rgb_blind_rs_event_audit.py \
  --scenarios Accident,InvadingTurn \
  --samples-per-town 1 \
  --write-sheets \
  --output-dir /tmp/automot_rgb_blind_rs_event_audit
```

输出重点：

- `manual_blind_answer_template.csv`：只给 blind sheet 路径，不给当前标签；先在这里填
  `manual_rs_spans/manual_event_spans`。
- `blind_sheets/<Scenario>/<route>/blind_page_000.jpg`：按时间顺序展示 RGB，不显示当前标签。
- `blind_sheets/<Scenario>/<route>/compare_page_000.jpg`：只在写完 blind answer 后查看，用于和当前标签对账。
- `route_blind_rs_event_audit.csv` / `scenario_blind_rs_event_summary.csv`：风险排序和 mismatch 摘要。

注意：自动 blind guess 只是保守 triage，不是真值。尤其 R5/无灯路口不能只靠车道线 CV 自动判断，
必须逐帧看 blind sheet 后手写 span；R4 颜色块也只能作为提示，最终仍以 RGB 可见控制区和时序上下文为准。

---

## 参考文件

- `collector.py`：采集器、XML 索引、XODR probe、RS 规则引擎
- `quick_start.py`：交互式入口、XML/XODR 画像和 `annotate-rs` 逐帧标注命令
- `analyzer.py`：结果统计
- `rgb_blind_rs_event_audit.py`：RGB-first 盲审 sheet、人工 span 模板与当前标签对账
- `web_app.py`：Web 可视化
- `ROAD_EVENT_CLASSIFICATION_PLAN.md`：ROAD/EVENT 总方案 + ROAD_STRUCTURE 调研/实现协议
- `ROAD_EVENT_CANDIDATE_MAPPING.md`：ROAD/EVENT 候选映射
