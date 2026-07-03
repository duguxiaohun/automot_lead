# 场景事件采集系统

本目录用于从 LEAD 离线数据中采集帧级 ROAD_STRUCTURE / EVENT 候选，并用
XML + XODR + meta 生成更精细的 `primary_road_structure`。

当前重点是 ROAD_STRUCTURE：

- 保留每个 scenario 的候选全集 `road_structures`，避免破坏旧 Web/分析逻辑。
- 额外输出 `primary_road_structure`、`secondary_road_structures`、
  `road_structure_candidates`、`evidence`。
- `AutoMoT/data/lead` 只提供 route XML；真实帧数据必须来自 `lead_data`。
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

逐场景 RS 调研产物生成：

```bash
python keyframe_filter/rs_research.py --samples-per-town 5
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

可选输出目录：

```bash
KEYFRAME_COLLECTION_OUTPUT=/path/to/output python keyframe_filter/quick_start.py
```

---

## 菜单功能

采集模式：

1. 单场景全部采集
2. 单场景指定数采集
3. 多场景采集
4. 全部采集

其他功能：

5. 多角度结构分析
6. 启动 Web 应用
7. 显示所有场景
8. ROAD_STRUCTURE XML/XODR 画像
9. 退出

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
  "evidence": {
    "rules_fired": ["r1_default_candidate", "r4_tl_confirmed"],
    "xml_path": "data/lead/Accident/...",
    "route_progress_m": 42.5,
    "review_required": false
  }
}
```

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
| R3 | 高速 / 匝道 / 合流 / 驶出 |
| R4 | 信号灯路口 |
| R5 | 无信号灯 / 信号灯失效 / 路权路口 |
| R6 | 路边停车 / 停车占道 |

规则实现来自：

- `ROAD_STRUCTURE_MAP_XML_LABELING_PLAN.md`
- `ROAD_STRUCTURE_PER_SCENARIO_LABELING_DESIGN.md`
- `ROAD_STRUCTURE_SCENARIO_RESEARCH_PROTOCOL.md`

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
- TwoWays 不全程 R2，只在 trigger/active/opposite-lane 窗口内 R2。
- Parking* 不全程 R6，灯控路口段 R4 优先。
- `CrossJunctionDefectTrafficLight` 强制 R5 覆盖 R4。
- `ParkedObstacle` 不是 R6；`ParkedObstacleTwoWays` 核心窗口才是 R2。
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
  静态同向障碍是 EVENT 证据，不把整段升级成 R2/R6；只在受控路口窗口进入 R4。
- `twoways_obstacle` / `invading_turn` / `vehicle_opens_door_twoways`：
  只有 XML trigger、XODR 对向/双向单车道拓扑、meta active 或距离字段共同成立时进入 R2；
  TwoWays 名称本身不能全程给 R2。
- `highway_merge` / `interurban`：只有 ramp/merge/split/highway 拓扑支持时进入 R3；
  EnterFlow/Merger/HighwayExit 的行驶事件不能替代 XODR 拓扑证据。
- `signalized_junction`：灯态有效、受控 junction 或 controller/traffic light 近邻成立时进入 R4；
  `BlockedIntersection` 和 `OppositeVehicleRunningRedLight` 的阻塞/违规只是 EVENT，不改成 R5。
- `nonsignalized_junction` / `defect_junction`：无有效灯态、stop/yield/priority 或灯故障机制成立时进入 R5；
  `CrossJunctionDefectTrafficLight` 强制 R5 覆盖 R4。
- `parking` / `parking_exit` / `static_cutin`：R6 只给 parking/shoulder/curb/parking-exit 结构窗口；
  停车相关 scenario 在信号灯路口段仍优先 R4/R5。
- `default_meta_map` / `noscenario`：默认 R1；ControlLoss、HardBreak、DynamicObjectCrossing、
  HazardAtSideLane 等行为/突发事件本身不改变 RS，只能通过灯态或 junction 证据临时进入 R4。

本轮自动阈值是调研初值：`junction_pre_m=40~60`、`junction_post_m=20~40`、
`two_way_min_pre_m=45~80`、`merge_pre_m=30~50`、`merge_post_m=40~50`、
`parking_pre_m=20~35`、`parking_post_m=50~60`。这些阈值必须在每个
`rules/thresholds.json` 中补齐 `supporting_runs/reviewed_artifacts/reason` 后才能作为正式代码依据。

---

## 故障排除

| 问题 | 处理 |
|---|---|
| `status=error` 且 `场景目录不存在` | 检查 `LEAD_DATA_ROOT` 或默认 `lead_data/<Scenario>` 是否存在 |
| `Routes数: 0` | 检查 scenario 目录下是否有 run 子目录 |
| 没有 `metas/*.pkl` | 当前 run 会被跳过；采集需要真实 LEAD meta |
| XML 匹配不到 | 先按 `(scenario,town,route_key)` 确认 `data/lead`，再按 `(town,route_key)` 全局查 `data_routes`；只有两边都没有有效 XML 时才设 `xml_available=false` 并降级 |
| 没有 carla Python API | XODR 拓扑查询自动降级，不应中断采集 |
| Web 看不到结果 | 确认 `collection_output/*_result.json` 已生成 |

---

## 参考文件

- `collector.py`：采集器、XML 索引、XODR probe、RS 规则引擎
- `quick_start.py`：交互式入口和 XML/XODR 画像
- `analyzer.py`：结果统计
- `web_app.py`：Web 可视化
- `ROAD_EVENT_CANDIDATE_MAPPING.md`：ROAD/EVENT 候选映射
- `ROAD_STRUCTURE_PER_SCENARIO_LABELING_DESIGN.md`：逐场景 RS 标定设计
