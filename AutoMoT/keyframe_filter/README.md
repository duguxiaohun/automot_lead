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

帧级规则会优先使用能 `import carla` 的 Python 环境读取精确 XODR 拓扑
（`map_road_id/lane_id/lane_type`、`has_opposite_driving_lane`、
`has_parking_or_shoulder_nearby` 等），例如：

```bash
/home/codon/anaconda3/envs/carla/bin/python keyframe_filter/quick_start.py
```

默认 Python 若不能 import `carla`，采集不会中断，会自动降级为静态 XODR planView/lane/signal
近邻解析。静态解析只有在 `map_projection_error_m <= 20m` 时设置
`xodr_topology_trusted=true` 并允许 R2/R3/R6 使用 topology high 证据；超过该误差时只保留
`xodr_topology_untrusted` 诊断，特殊 RS 会降为 medium/low + review。
环岛 / roundabout 明确按 R1 处理：XODR 若输出 `map_is_roundabout=true`，即使附近有
junction road，也会压住 R4/R5 并在 evidence 中写入 roundabout 规则命中。

菜单里的 1/2/3/4 都是正式“采集 + 逐帧 RS 标注”入口：每帧都会同时写
`road_structures` 候选全集与该帧独属的 `primary_road_structure` /
`frame_rs_annotation.label`。第 9 项只作为小范围 smoke / 参数闭环调试入口保留。

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

该入口会按每个 scenario 的 `SCENARIO_RULE_CONFIG` 独立规则逐帧输出
`primary_road_structure`、`secondary_road_structures`、`annotation_comment`、
`evidence.review_reasons` 和 route 级分布/切换摘要。`road_structures` 仍保留旧候选全集；
真正的单帧标定结果请读新增的 `frame_rs_annotation`：

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

每条 route 还会执行统一时序去抖：R2/R3/R4/R5/R6 短于 4 帧、R1 短于 2 帧的孤立片段会并回邻近稳定片段，
去抖原因写入 `evidence.temporal_smoothing`，route 摘要写入 `temporal_smoothing.changes`。
这条规则适用于全部 RS，不只是 R1/R4。

Web 可视化页面也按这个口径展示：顶部绿色标签是
`frame_rs_annotation.label` / `primary_road_structure`，置信度对应这个“本帧最终 RS
标签”，不是候选全集的置信度；候选全集单独显示为“该场景全部候选 RS”。页面下方的证据归因
会同时列出 XML/route 进度、LEAD meta 动态字段和 XODR topology 摘要，用来判断该帧标注
到底由哪类证据触发、是否需要人工复核。

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

2026-07-03 第一轮全帧 RGB 复核回灌后 smoke 结论：43 场景小样本均可生成逐帧标注；
静态 XODR 模式下 R2/R3/R6 若缺少 opposite/merge/parking 局部拓扑，不再仅凭
scenario/trigger 窗口压过 R1，而是降为 secondary/review；
`noScenarios` 已调成只有 meta 有效灯态或 light hazard 时才允许 R4，否则保守 R1；
`StaticCutIn` 无 R3/R6 拓扑证据时回 R1 中置信，不再给 0.35 低置信。
当前 43 场景 × 10 帧 smoke 分布为
`R1=182, R2=10, R3=20, R4=128, R5=80, R6=10`，
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
- 只按 scenario 名称或 active scenario 延长特殊 RS，导致事件已经结束或 merge/路口已经离开后仍保持 R2/R3/R4/R5/R6。
- 反向问题也存在：明显高速/merge 因 XODR 投影失败被压回 R1；坏 XODR 不能当作否定证据，
  但 TwoWays road-layout 本身也不能把非核心片段升成 R2。
- `two_way_layout_prior` 若被理解成整条 route 的场景名先验，会把 `AccidentTwoWays`、
  `ConstructionObstacleTwoWays` 等非核心片段误标为 R2；它现在只能作为弱候选，R2 primary
  只覆盖必须借/等对向的核心障碍 span，绕过障碍后恢复 R1/R4。
- 高速/merge/exit/enter-flow 场景此前被 R1 默认桶吃掉；RGB 抽样确认
  `HighwayCutIn`、`HighwayExit`、`EnterActorFlow*`、`MergerIntoSlowTraffic*`
  是高速/快速路背景，候选池不再开放 R1，非路口默认 R3。
- 行人、事故、施工、急刹、切入、开门、闯红灯等多数是 EVENT，不应直接改变 ROAD_STRUCTURE；RS 必须由道路几何和控制源决定。
- 置信度和 summary 只能作为索引。高置信不等于 RGB 正确，稳定 R1 也必须逐帧看，避免漏掉后段 merge、停车带、路口或双向路结构。

第二轮错配回灌后，`collector.py` 已按图像优先结论收紧：

- 弱 R2/R3/R4/R5/R6 候选不能通过 priority tie-break 低分压过 R1。
- `route_projection_error_m > 5m` 时，普通 `scenario_active` / `trigger_close_m`
  只作为 review 线索，不再单独撑起 two-way / merge / parking / junction 窗口。
- 静态 XODR 的 signal/opposite/parking/merge/junction hint 在高投影误差帧降级为
  `*_demoted_projection_error` 证据；nonsignalized 场景遇到静态 signal 会写
  `nonsignalized_with_signal_topology_conflict`，需要人工结合 RGB 确认。
- `MergerIntoSlowTraffic*` 的明显 merge 口不能被坏 XODR 自动压回 R1；当 RGB/LEAD XML
  已支持合流，但 route/XODR 投影误差导致 topology 不可信时，规则会使用 XML
  `start_actor_flow/end_actor_flow` 强近邻和 trigger 距离作为 fallback，
  写入 `r3_merger_actor_flow_or_trigger_fallback`，主标签可为 R3，同时保留 review 原因。
  `HighwayCutIn`、`HighwayExit`、`EnterActorFlow*`、`MergerIntoSlowTraffic*`
  已从候选池删除 R1；active scenario 只作为审计证据，不单独制造 R4/R5。
- 人工逐帧看图后又补了两条更强的图像优先门控：
  静态 signal 或灯态只有在 `is_junction` / 可信 XODR junction / stopline / 近距离 signal-junction
  上下文成立时才给 R4 high；否则回 R1 + review。
  TwoWays 的 R2 high 必须有近距离障碍、stuck、vehicle_hazard 或 lane-change 核心证据；
  双向 road-layout 本身不能维持 R2 primary。`twoways_obstacle` 现在把 layout-prior
  降为 R2=0.58 弱候选，R1=0.78 做主标签；只有核心障碍/借道帧由 trigger/meta 给 R2=0.90。
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
`R1=242, R2=0, R3=20, R4=88, R5=70, R6=10`，
`confidence min/avg/max = 0.70/0.8120/0.98`，`review_ratio=0.4140`。
代表性错配路线里，`Accident/Town03` 无清晰路口/灯控画面时尾段 R4 回 R1；
`AccidentTwoWays/Town01` 逐帧 RGB 复核后确认，正确边界是
`R1 f0-f54 -> R2 f55-f135 -> R1 f136-end`，绕过障碍后不再保持 R2；
`HighwayCutIn/HighwayExit/EnterActorFlow*/MergerIntoSlowTraffic*` 不再被 R1 吃掉，
局部复核输出均为 R3；`HardBreakRoute` 抽样显示城市/乡村/快速路混合，已改成 route 级分桶：
高速 route 候选收敛为 R3/R4，非高速 route 保留 R1；
`PriorityAtJunction` 虽在 Town12/13，但仍保持 R1/R5，不按高速处理。

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

启动 Web 后，页面右侧会分三块展示：本帧最终 RS 标注、该 scenario 的候选 RS 全集、
以及 XML / LEAD meta / XODR 证据归因。绿色主标签才是当前 frame 的最终
`frame_rs_annotation.label`；候选全集不是标注结果。

---

## 菜单功能

采集模式：

1. 单场景全部采集 + 逐帧 RS 标注
2. 单场景指定数采集 + 逐帧 RS 标注
3. 多场景采集 + 逐帧 RS 标注
4. 全部采集 + 逐帧 RS 标注

其他功能：

5. 多角度结构分析
6. 启动 Web 应用
7. 显示所有场景
8. ROAD_STRUCTURE XML/XODR 画像
9. 逐帧 RS 标注 smoke / 参数闭环调试入口
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
| R6 | 路边停车 / 停车占道 |

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
- TwoWays 只在必须借/等对向的核心障碍 span 输出 R2；障碍前和绕过障碍后回 R1/R4，
  `two_way_layout_prior` 只能作为弱候选，不做 primary。
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
  R2 只覆盖核心借道/障碍层。核心层需要 XML trigger、XODR 对向/双向单车道拓扑、
  meta active、近距离障碍、stuck、vehicle_hazard 或 lane-change 证据；道路布局层只给弱候选。
- `highway_merge`：`HighwayCutIn`、`HighwayExit`、`EnterActorFlow*`、`MergerIntoSlowTraffic*`
  候选删除 R1，非路口默认 R3；merge/split/ramp/actor-flow 只提高置信与定位边界。
- 混合场景 route 分桶：`HardBreakRoute` / `interurban` / `StaticCutIn` / `ParkingCutIn`
  不能只按 Town12/13 判高速。先用 RGB sheet 把 route 分成高速/快速路桶和非高速桶；
  高速桶候选收敛为 R3/R4，非高速桶保留 R1。当前已逐 id 均匀 5 帧 RGB 复核：
  HardBreakRoute 97 个 route 中 16 个进高速桶，StaticCutIn 100 个 route 中 44 个进高速桶；
  InterurbanActorFlow 91 个、InterurbanAdvancedActorFlow 78 个、ParkingCutIn 99 个未发现高速桶。
  `Town12_Rep0_258_0_route0_01_08_09_35_42` 这类乡村普通路明确保留 R1。
  输出中 `evidence.route_semantic_bucket` 和 route 级 `route_semantic_bucket_distribution`
  会记录当前 route 走 `highway_rgb_route` 还是 `mixed_reviewed_non_highway`。
- `signalized_junction`：灯态有效、受控 junction 或 controller/traffic light 近邻成立时进入 R4；
  `BlockedIntersection` 和 `OppositeVehicleRunningRedLight` 的阻塞/违规只是 EVENT，不改成 R5。
  如果 primary R4 不是由有效 `traffic_light_state` 支撑，而是由 junction/window/static signal
  支撑，必须写 `signalized_r4_without_meta_tl_requires_rgb_confirmation`，逐帧看 RGB 确认
  stopline/crosswalk/cross traffic/blocked pocket 是否仍可见，不能只按置信度放行。
  若只有 static signal 近邻 + 灯态而缺少 `is_junction`/XODR junction，strong context 距离阈值为
  25m；25-35m 只保留弱 R4 候选，避免在雾中普通路段过早覆盖 R1。
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
| 没有 carla Python API | XODR 查询自动降级到静态 planView/lane/signal 近邻；若 `xodr_topology_untrusted` 很多，优先检查 XODR 坐标系/地图路径 |
| Web 看不到结果 | 确认 `collection_output/*_result.json` 已生成 |
| Web 只有候选没有本帧标签 | 重新打开新版 `web_app.py`；绿色“本帧最终标签”来自 `frame_rs_annotation.label`，置信度对应这个标签 |
| Web XODR 显示 `trusted=false` | 说明静态 XODR 投影或局部拓扑不足以 high confidence；优先用 CARLA Python 环境重跑并对比 review 是否下降 |
| 环岛被标成 R4/R5 | 检查 Web XODR 摘要是否有 `roundabout=true`；若没有，说明该 town 的 XODR 环岛几何特征需要补 probe 规则 |
| 单帧 R4/R5/R2/R3/R6 抖动 | 检查 route 摘要 `temporal_smoothing.changes`；短片段默认会被并回邻近稳定 RS |

---

## 参考文件

- `collector.py`：采集器、XML 索引、XODR probe、RS 规则引擎
- `quick_start.py`：交互式入口、XML/XODR 画像和 `annotate-rs` 逐帧标注命令
- `analyzer.py`：结果统计
- `web_app.py`：Web 可视化
- `ROAD_EVENT_CLASSIFICATION_PLAN.md`：ROAD/EVENT 总方案 + ROAD_STRUCTURE 调研/实现协议
- `ROAD_EVENT_CANDIDATE_MAPPING.md`：ROAD/EVENT 候选映射
