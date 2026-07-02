# 场景事件采集系统

本目录用于从 LEAD 离线数据中采集帧级 ROAD_STRUCTURE / EVENT 候选，并用
XML + XODR + meta 生成更精细的 `primary_road_structure`。

当前重点是 ROAD_STRUCTURE：

- 保留每个 scenario 的候选全集 `road_structures`，避免破坏旧 Web/分析逻辑。
- 额外输出 `primary_road_structure`、`secondary_road_structures`、
  `road_structure_candidates`、`evidence`。
- `AutoMoT/data/lead` 只提供 route XML；真实帧数据必须来自 `lead_data`。

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

`ROAD_STRUCTURE XML/XODR 画像` 会逐 scenario 遍历所有 town，每个 town 默认抽 3 个 XML，
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

核心约束：

- TwoWays 不全程 R2，只在 trigger/active/opposite-lane 窗口内 R2。
- Parking* 不全程 R6，灯控路口段 R4 优先。
- `CrossJunctionDefectTrafficLight` 强制 R5 覆盖 R4。
- `ParkedObstacle` 不是 R6；`ParkedObstacleTwoWays` 核心窗口才是 R2。
- `data/lead` XML 不能替代真实 `lead_data` 帧数据。

---

## 故障排除

| 问题 | 处理 |
|---|---|
| `status=error` 且 `场景目录不存在` | 检查 `LEAD_DATA_ROOT` 或默认 `lead_data/<Scenario>` 是否存在 |
| `Routes数: 0` | 检查 scenario 目录下是否有 run 子目录 |
| 没有 `metas/*.pkl` | 当前 run 会被跳过；采集需要真实 LEAD meta |
| XML 匹配不到 | `xml_available=false`，RS 会降级为 meta/XODR 弱规则 |
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
