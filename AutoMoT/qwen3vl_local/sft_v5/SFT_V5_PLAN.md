# SFT v5 方案说明：RS / EVENT OPSD + torchrun sequence padding

本文是 `qwen3vl_local/sft_v5/` 的实现蓝图。目标是仿照 `sft_v3`
的 offline on-policy self-distillation (OPSD) 思路，但把监督目标从
CARLA scenario / status / subgoal 改成 `AutoMoT/keyframe_filter` 标定出的
帧级 `ROAD_STRUCTURE` 与 `EVENT`。

v5 的关键变化：

- 数据源改为 `AutoMoT/keyframe_filter/collection_output/*_result.json` 的
  全帧标定结果，不再使用 `keyframes_all_scenarios.json` 的 5-anchor 子场景。
- 训练目标改成两问串行：
  1. 当前天气、道路结构 RS、是否正在发生/处在突发事件。
  2. 在第 1 问 RS 正确的前提下，判断当前 EVENT；若第 1 问认为有突发事件，
     在当前 RS 对应的 U-E 候选中选择；否则输出当前 RS 的 regular event (RE)。
- Memory 只保留 `RS` 与 `EVENT`，不再保留 scene/status/subgoal。
- 多卡训练改成 torchrun 多进程：DataLoader 每次取 route sequence，collate 只做本 rank
  的 local padding / local length；`train.py` 主进程再 all-reduce 当前 step 的全局
  最长 sequence，补齐 mask 后进入统一时间循环。默认每 512 个 global 有效 frame，
  或最迟 32 个 global timestep，在完整 timestep 边界手动 all-reduce LoRA 梯度并
  更新权重，不再等待完整超长 route batch。
- Prompt 全部使用英文，标签选项必须是自然语言描述，不训练模型只背裸标签名。

> 约定：本文里 `RS` 指 ROAD_STRUCTURE，`UE` 指 unusual event，`RE` 指 regular /
> no-unusual-event 状态。除非后续拍板修改，v5 默认把 `R-E1..R-E5` 折叠成
> “当前 RS 下没有突发事件的 RE”，详见 §4.2。

---

## 1. 文件结构

计划落地为以下文件：

```text
qwen3vl_local/sft_v5/
  __init__.py
  SFT_V5_PLAN.md
  SFT_V5_RUN.md
  labels.py
  prompts.py
  build_dataset.py
  train.py
  train.sh
  eval.py
  probe.py
  inspect_teacher.py
  check_loss_mask.py
  test_memory_update.py
  test_dataset_contract.py
```

职责划分：

- `labels.py`
  固定 RS A-E 选项、RS 到 R1-R5 的映射、EVENT 候选池、自然语言描述、
  multi-label 单标签解析规则、天气文本化规则。
- `prompts.py`
  固定 Memory 文本、两问 prompt、teacher privileged prompt、输出解析、
  target span 与 GT leak 检查。这里不读图片、不加载模型。
- `build_dataset.py`
  从 `collection_output/*_result.json` 构建 route-level sequence index。
  输出 `sequence_index.jsonl`，每行是一条 route 的元数据和 frame label 摘要。
- `train.py`
  DDP + OPSD 主入口。每个 rank 默认用 `LengthBalancedDistributedSampler` 读 sequence batch，
  collate 成 `[B, T_max]` 的 padded batch，按时间步推进 memory 与 Q/A。
- `eval.py`
  自由生成评估。使用同一套 Memory 和两问协议，不用 teacher 强制纠偏。
- `probe.py`
  case-level dump，保存 RGB 路径、prompt、student 输出、teacher prompt/target、
  memory transition 与 GT。
- `inspect_teacher.py`
  抽检 privileged teacher 的分析质量；只跑 base Qwen `disable_adapter()`，
  不训练。
- `check_loss_mask.py`
  静态检查 analysis / RS / abnormal / event span 权重，防止 prompt 改动导致
  离散标签 loss 掉线。

---

## 2. 数据来源与过滤

### 2.1 输入目录

默认从 `AutoMoT/` 当前目录运行，路径使用项目惯例：

```bash
python qwen3vl_local/sft_v5/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_v5_data
```

`build_dataset.py` 只读取：

- `keyframe_filter/collection_output/<Scenario>_result.json`
- `lead_data/<Scenario>/<run_id>/rgb/*.jpg`
- 标定结果中的 `xml_path` 或 `annotation.evidence.xml_weather`

明确不读取 / 不训练：

- `keyframe_filter/collection_output/noScenarios_result.json`
- `keyframe_filter/collection_output/multi_scenario_collection.json`
- `keyframe_filter/collection_output/frame_rs_annotation_summary.json`
- 单场景 result 里的 `abnormal_duration_skipped` routes
- 单场景 result 里的 `data_missing_skipped` routes
- 任意 `route.status != "success"` 的 route
- 任意 `xml_available=false`、缺 `metas/*.pkl`、缺 RGB、缺可解析逐帧 annotation 的 route

原因：用户明确要求 `noScenarios_result.json` 不参与训练；缺 meta/xml 的 route 在
`keyframe_filter` 规范里是数据质量问题，不是标签规则失败；异常时长 route 已经由
`lead_video_tools.abnormal_duration_filter` 规则剔除。

`review_required=true` 的帧正常参与训练。它表示标定证据需要人工关注或 RGB 复核，
不是数据结构缺失；只要 route 成功、XML/meta/RGB/逐帧 annotation 完整，就进入 v5。

### 2.2 sequence row schema

`build_dataset.py` 输出 `sequence_index.jsonl`。每行是一条 route sequence：

```json
{
  "dataset_version": "sft_v5_rs_event_sequence",
  "scenario": "VehicleTurningRoute",
  "route_id": "Town03_Rep0_...",
  "run_dir": "lead_data/VehicleTurningRoute/Town03_Rep0_...",
  "xml_path": "data/lead/VehicleTurningRoute/Town03_route_....xml",
  "split": "train",
  "num_frames": 128,
  "frames": [
    {
      "frame_id": 0,
      "frame_time_s": 0.0,
      "rgb_path": "lead_data/.../rgb/0000.jpg",
      "history_rgb_paths": [".../0000.jpg", ".../0000.jpg", ".../0000.jpg", ".../0000.jpg"],
      "weather": {
        "cloudiness": 10.0,
        "precipitation": 0.0,
        "wetness": 0.0,
        "fog_density": 3.0,
        "sun_altitude_angle": 45.0
      },
      "weather_text": "clear daytime weather with light cloudiness, dry road surface, and low fog",
      "rs_label": "R1",
      "rs_option": "A",
      "rs_confidence": 0.78,
      "rs_secondary": [],
      "rs_candidates": {"R1": 0.78, "R4": 0.36},
      "event_labels_raw": ["R-E1"],
      "event_label": "RE",
      "event_code": "R-E1",
      "abnormal": false,
      "scenario_event_candidates": ["R-E1", "R-E2", "R-E4", "R-E5", "U-E4"],
      "frame_allowed_events_raw": ["R-E4", "U-E4"],
      "regular_event_codes": ["R-E4"],
      "event_candidate_codes": ["RE", "U-E4"],
      "event_option_map": {"A": "U-E4", "B": "RE"},
      "review_required": false,
      "source": {
        "meta_path": ".../metas/0000.pkl",
        "annotation_comment": "...",
        "event_comment": "..."
      }
    }
  ]
}
```

注意：

- `history_rgb_paths` 采用 v3 同款 4 帧 left-pad。第 0/1/2 帧历史不足时复制
  frame 0。
- `weather` 优先取 annotation 的 `evidence.xml_weather`；如果老结果没有该字段，
  再按 route progress 从 XML `<weathers><weather route_percentage=...>` 选择最近天气。
- `event_labels_raw` 保存原始 `frame_event_annotation.events`，用于审计多标签；
  `event_label` 是 v5 单标签训练目标，见 §4。
- `event_code` 保存最终选中的原始 EVENT code；若折叠为 RE，`event_label="RE"`，
  `event_code` 仍保存最接近的 `R-E*` 方便回查。
- `frame_allowed_events_raw` 优先来自每帧 `frame_event_annotation.allowed_events`，
  其次来自 `event_evidence.allowed_events`；只有两者都缺失时，才 fallback 到
  `scenario_event_candidates ∩ EVENT_CANDIDATES_BY_RS[current_rs]` 静态表。
- `regular_event_codes` 保存当前帧允许的所有 `R-E*`。v5 监督仍只训练一个 `RE`
  类，但 R3 等道路结构的 RE 描述必须覆盖 `R-E1/R-E2/R-E3` 等多个 regular mode。

### 2.3 JSON 读取策略

单场景 result 可能很大。实现时按 scenario 文件逐个处理：

1. 先读取顶层 `abnormal_duration_skipped` 与 `data_missing_skipped`，形成 skip set。
2. 遍历 `routes` / route result 列表，只保留 `status="success"`。
3. 每条 route 只把训练需要字段写入 `sequence_index.jsonl`，不要把完整 evidence
   原样复制进 v5 dataset。

如果 stdlib `json.load` 内存压力过大，再引入可选 `ijson` streaming reader；
但第一版可以先用逐 scenario 加载，因为 result 文件数量有限且 build 是离线一次性过程。

---

## 3. RS 选项：ABCDE 对应 R1-R5

v5 不直接让模型输出 `R1` / `R2`。Prompt 里展示 A-E，每项是一句话语义描述；
parser 读取 option letter，再映射回 R label。

```python
RS_OPTIONS = {
  "A": "R1",
  "B": "R2",
  "C": "R3",
  "D": "R4",
  "E": "R5",
}
```

### A = R1

`A. Ordinary same-direction drivable road: the ego vehicle is mainly following,
lane-keeping, making same-direction lane adjustments, or recovering on a normal
drivable path; there is no dominant intersection rule, traffic-light control,
highway merge/exit structure, or opposing-lane borrowing requirement.`

覆盖要点：

- 常规道路 / 同向可行驶道路。
- 普通车道保持、跟车、安全距离。
- 同向变道、绕障前后恢复、停车侧/路边通行但未压缩成对向借道。
- 环岛也归入 R1，即使 XODR 把它编码成 junction road。

### B = R2

`B. Bidirectional single-lane or opposing-lane-sharing road: the usable corridor
is narrow enough that the oncoming lane affects the decision, including borrowing
the opposing lane to pass a blockage or yielding because an oncoming vehicle
invades the ego lane.`

覆盖要点：

- 双向单车道 / 借对向车道道路。
- 对向车道参与决策。
- 自车可能因静态障碍借对向绕行。
- 对向车辆异常侵占时需要被动让行。

### C = R3

`C. Highway, ramp, merge, split, or exit structure: the ego vehicle is in a
high-speed or ramp-like decision space where speed matching, gap selection,
target-lane tracking, merging, diverging, or exiting dominates the driving rule.`

覆盖要点：

- 高速 / 快速路 / 主辅路。
- 匝道、合流、分流、驶出。
- 速度匹配、侧后方间隙、目标车道和主路车流关系。

### D = R4

`D. Signalized intersection: the ego vehicle is inside or approaching an
intersection where working traffic lights are the main right-of-way rule,
including red-light waiting, green-light crossing, and protected or permissive
turning under signal control.`

覆盖要点：

- 信号灯路口。
- 红灯停、绿灯行。
- 信号灯左转 / 右转仍需观察冲突对象。
- 对方闯红灯时 RS 仍是 R4，突发由 UE 表达。

### E = R5

`E. Unsignalized or priority-controlled intersection: the ego vehicle is inside
or approaching an intersection without a reliable traffic-light rule, so it must
use stop/yield signs, priority, road geometry, cross traffic, pedestrians, or
safe-gap reasoning to proceed.`

覆盖要点：

- 无信号灯 / 路权路口。
- STOP / yield / 无灯 T 口或十字路口。
- 横向车流、对向车流、行人/自行车让行关系。
- 规则源失效由 EVENT 表达：有灯控硬件的故障路口可仍按 R4 + U-E7，
  无可靠灯控规则或路权失效的帧按 R5 + U-E7，最终以 `keyframe_filter`
  的逐帧标定为准。

---

## 4. EVENT 选项与单标签规则

### 4.1 EVENT 自然语言描述

v5 的 EVENT prompt 展示自然语言选项，内部仍保存原始 code 方便统计。

RE 不是一个原始 `keyframe_filter` code，而是 v5 为第二问引入的折叠标签：

```text
RE: No unusual event is currently interrupting the driving task; continue the
regular behavior implied by the current road structure.
```

不同 RS 下的 RE 解释。RE 是单一监督类，但它的文案要吸收当前帧
`regular_event_codes` 的细分含义；例如 R3 可同时覆盖主线跟车/速度匹配
(`R-E1`)、目标导向变道/回目标车道 (`R-E2`) 和常规匝道合流/分流/驶出
(`R-E3`)。

- R1 / option A:
  `No unusual event; continue ordinary same-direction lane keeping, following,
  safe-distance keeping, same-direction lane adjustment, or recovery after a
  completed maneuver.`
- R2 / option B:
  `No unusual event; continue along the bidirectional narrow-road space while
  keeping safe clearance from oncoming traffic, without an active blockage or
  invading oncoming vehicle.`
- R3 / option C:
  `No unusual event; continue the regular highway/ramp behavior allowed in this
  frame, such as mainline following and speed matching, target-lane change or
  recovery, normal merging, diverging, or exiting.`
- R4 / option D:
  `No unusual event; obey normal traffic-light intersection rules such as
  stopping for red, proceeding on green, or turning under signal control.`
- R5 / option E:
  `No unusual event; negotiate the unsignalized or priority intersection using
  stop/yield rules, right-of-way, and safe-gap reasoning.`

UE descriptions:

- `U-E1`:
  `A lead vehicle suddenly brakes or decelerates, requiring the ego vehicle to
  react with reduced speed or increased following distance.`
- `U-E2`:
  `A static obstacle, accident, construction object, parked vehicle, open door,
  or blocked lane occupies the ego path and forces avoidance, stopping, or
  borrowing space.`
- `U-E3`:
  `A moving vehicle cuts in, pulls out, or dynamically occupies the ego path,
  creating a short-horizon conflict.`
- `U-E4`:
  `A pedestrian, cyclist, or small vulnerable road user crosses or laterally
  enters the ego vehicle's intended path.`
- `U-E5`:
  `An oncoming vehicle abnormally invades the ego lane or priority space,
  forcing the ego vehicle to yield or adjust.`
- `U-E6`:
  `Another vehicle violates the expected intersection rule, such as running a
  red light or crossing against the ego vehicle's priority, creating conflict.`
- `U-E7`:
  `The intersection rule source is unreliable or failed, such as defective
  traffic lights or ambiguous priority, so normal right-of-way reasoning is
  broken.`
- `U-E8`:
  `The forward road or intersection space is temporarily blocked or reopening,
  requiring waiting, queue handling, or cautious release.`

### 4.2 默认 EVENT 候选池

第 2 问不显示全量 13 个 EVENT。它优先使用当前帧标定结果已经写好的
`frame_event_annotation.allowed_events` 作为候选池，因为 `collector.py` 的最终输出
已经包含 scenario 白名单、当前 primary RS、route-level postprocess、candidate clamp
和 interrupted overlay 例外。只有旧结果缺少逐帧 allowed events 时，才 fallback 到
静态表。

```python
q2_raw_candidates = (
    frame.frame_event_annotation.allowed_events
    or frame.event_evidence.allowed_events
    or (scenario_event_candidates[scenario] ∩ EVENT_CANDIDATES_BY_RS[current_rs])
)
q2_display_candidates = collapse_allowed_regular_events_to_RE(q2_raw_candidates, current_rs)
```

核心约束：

- Q2 首选每帧 `allowed_events`，不重新覆盖 collector 的最终 clamp / overlay 结果。
- fallback 静态候选才要求同时属于**当前 RS**和**当前 scenario**。
- 所有逐帧 allowed 的 `R-E*` regular 分支在 prompt 里合并显示为一个 `RE` 选项；
  这保留 interrupted overlay / final clamp 的例外口径，同时不把 `R-E*` 变成 v5
  分类目标。
- Q1 的 abnormal flag 不决定候选池，只影响 Q2 题目措辞和 teacher resolver 的优先解释。

RS-level EVENT pool：

```python
EVENT_CANDIDATES_BY_RS = {
  "R1": ["R-E1", "R-E2", "U-E1", "U-E2", "U-E3", "U-E4"],
  "R2": ["R-E1", "R-E2", "U-E2", "U-E5"],
  "R3": ["R-E1", "R-E2", "R-E3"],
  "R4": ["R-E4", "U-E4", "U-E6", "U-E7", "U-E8"],
  "R5": ["R-E5", "U-E4", "U-E5", "U-E6", "U-E7", "U-E8"],
}
```

静态表仅作 fallback 和 contract check。正常 build / train / eval 应使用逐帧
`allowed_events`。

说明：

- `scenario_event_candidates` 读单场景 result 顶层的 `event_candidates`，只用于
  fallback；prompt 不显示 scenario 名。
- R4 只有在当前 scenario 候选也包含 `U-E7` 时才显示 `U-E7`；不会因为 R4 pool
  里有 U-E7 就给所有 R4 场景开放。
- R3 不开放 UE，但 R3 的 `RE` 不是单一语义：`regular_event_codes` 可能包含
  `R-E1/R-E2/R-E3`。Q2 prompt 中仍显示一个 RE 选项，文字必须列出这些 allowed
  regular modes，训练目标仍是 `RE`。

第 2 问候选构造：

- 若 Q1 parsed `ABNORMAL=NO`：
  - 显示逐帧 allowed events 折叠后的候选池：`RE + U-E*`；如果 allowed events
    只有 regular events，则允许退化为单选 `RE`。
  - prompt 会说明“你刚才判断没有突发事件，但仍需从这些候选里确认当前 EVENT”。
  - supervised event label 按 GT 解析：若 GT 没有 UE，目标是 `RE`；若 GT 有 UE，
    teacher 会解释为什么 Q1 的 no-abnormal 判断应被 Q2 修正为某个 UE。
- 若 Q1 parsed `ABNORMAL=YES`：
  - 显示同一个逐帧候选池：`RE + U-E*`；如果 allowed events 没有 UE（例如大部分 R3
    帧），候选池就是单选 `RE`，teacher 会把 EVENT 拉回 regular。
  - prompt 会说明“你刚才判断有突发事件；若视觉证据不足，也可以选择 RE”。
  - parser 输出 option letter，训练内部通过本帧 `event_option_map` 映射到 `RE`
    或具体 `U-E*`。

选项字母随机化：

- Q2 的 `A/B/C/...` 每次都必须按可复现随机顺序重排，不能让某个字母长期固定代表
  `RE` 或某个 `U-E*`。
- 随机 seed 建议使用 `hash(dataset_version, run_id, frame_id, q2_variant_seed)`。
- dataset row / probe dump 必须保存本帧 `event_option_map`，例如
  `{"A": "U-E4", "B": "RE", "C": "U-E6"}`，parser 只输出 option letter，
  训练代码再按该 map 还原成 event label。

### 4.3 从原始 EVENT 到 v5 单标签

每帧原始 EVENT 可能是单标签，也可能是路口 regular + UE 双标签。v5 按以下规则
生成单标签：

1. 读取 raw events：
   - 优先 `frame_event_annotation.events`
   - 其次 `events`
   - 兜底 `[primary_event]`
2. 如果 raw events 中存在任意 `U-E*`：
   - v5 `abnormal=true`
   - 候选真值集合 = 所有 `U-E*`
   - 如果只有一个 UE，单标签就是它。
   - 如果有多个 UE：
     - 若 student 第 2 问输出在这些 UE 里，teacher target 采用 student 输出的 UE，
       让教师分析解释“为什么这个可接受的 UE 成立”。
     - 否则选择置信度最高的 UE；没有事件级置信度时使用 `primary_event`；
       再没有则按稳定顺序 `U-E1..U-E8` 取第一个。
3. 如果 raw events 全是 `R-E*`：
   - v5 `abnormal=false`
   - 默认折叠成 `event_label="RE"`。
   - `event_code` 保存主原始 `R-E*`，`regular_event_codes` 保存所有允许或命中的
     regular codes，用于审计和生成更细的 RE 文案。
   - 即使存在多个 `R-E*`（尤其 R3 下 `R-E1/R-E2/R-E3`），训练监督仍是一个
     `RE` 类，不让模型在 v5 中学习细分 regular label。

用户提出的“UE 优先；同类双标签时如果 student 答案在双标签里就用 student，否则用高置信”
在 v5 中落到 teacher target resolver，而不是静态 dataset row。这样 OPSD 的 teacher
仍然能基于 student rollout token 给出同一上下文下的分布偏好。

### 4.4 RS 双标签 / secondary 处理

RS 训练目标默认取高置信单标签：

1. 优先读取 `frame_rs_annotation.label` / `primary_road_structure`。
2. 如果 `frame_rs_annotation.secondary` 或 `road_structure_overlay` 表示双标签：
   - 读取 `road_structure_candidates` 分数。
   - 选择分数最高的 RS 作为 GT。
   - 分数缺失时保持 primary。
3. RS 不采用“student 若在双标签里就算对”的策略；用户已明确 RS 双标签时回答置信度更大的标签。

---

## 5. Memory 设计

v5 Memory 是跨帧纯文本状态：

```text
[MEMORY]
BELIEVED_RS: Ordinary same-direction drivable road ...
EGO_TO_GOAL_XY=(+12.3, -1.5) m
[/MEMORY]
```

Q1 使用 road-only memory，不提前暴露 event；Q2 才在同一轮 Q1 之后使用
road + event memory：

```text
[MEMORY]
BELIEVED_RS: Ordinary same-direction drivable road ...
BELIEVED_EVENT: No unusual event; continue ordinary same-direction behavior.
EGO_TO_GOAL_XY=(+12.3, -1.5) m
[/MEMORY]
```

注意：memory 渲染文本只写自然语言描述，不写 `A/B/C/D/E` 选项字母，也不写
`RE/U-E*` 事件代码。A-E 只出现在 `RS_CHOICES` 和最终 `RS:` 答案里；Q2 的动态
事件选项字母只出现在 `EVENT_CHOICES` 和最终 `EVENT:` 答案里。
`EGO_TO_GOAL_XY` 是必需学生输入：新 build_dataset 会在缺失时跳过 frame/route；
运行时 `RouteSequenceDataset` 也会跳过旧 index 中缺 `ego_to_goal_xy` 的 frame，
避免 prompt 继续显示 `UNKNOWN`。

内部 dataclass：

```python
@dataclass
class Memory:
    rs_label: str        # R1-R5
    event_label: str     # RE or U-E*, only rendered from Q2 onward
    ego_to_goal_x: float | None
    ego_to_goal_y: float | None
```

初始化：

- 每条 route sequence 的第一帧默认 `rs=GT_RS`，`event=RE under GT_RS`。
- 可选 `--p-init-rs-correct` 做扰动实验；第一版默认 1.0，因为 v5 的重点是学习
  RS/EVENT 标定本身，不先引入随机错 memory。

帧间更新：

- Q1 RS 正确：
  - `memory.rs` 更新为 student RS。
  - 若 Q1 abnormal=no，`memory.event=RE under student RS`。
  - 若 Q1 abnormal=yes，等待 Q2 更新 event。
- Q1 RS 错误：
  - 当前帧立即停止，不跑 Q2，不计算 Q2 loss。
  - 下一帧开始前恢复 `memory.rs=GT_RS(next frame)`，
    `memory.event=RE under GT_RS(next frame)`。
  - 这对应用户说的“RS 回答错误就结束采样，下一次采样默认恢复真值 RS，事件 memory
    对应 RS 真值下面的默认 RE”。
- Q2 输出合法：
  - `memory.event` 更新为 parsed event。
- Q2 输出非法：
  - 若输出不在本帧 `event_option_map` 中，下一帧恢复 GT RS + RE。
  - 若输出映射为 `RE`，memory 保持当前 RS 下的 RE。

实现时要显式记录：

```json
{
  "frame_id": 17,
  "q1_rs_correct": false,
  "q1_abnormal_correct": true,
  "q2_triggered": false,
  "reset_next_frame": true,
  "memory_before": "...",
  "memory_after": "..."
}
```

---

## 6. 两问 Prompt 协议

所有 prompt 必须英文。System prompt 只放通用角色和证据原则：

```text
You are an autonomous driving agent. Use the stitched RGB history as visual
context, ordered from oldest to newest. Focus on traffic lights/signs, nearby
vehicles/pedestrians/obstacles, lane markings and road structure, and key
factors affecting ego decisions. Keep the current memory by default and change
it only when clear visual evidence supports the change. Describe weak, distant,
foggy, or occluded evidence as uncertain. Never mention ground truth, answer
keys, hidden labels, dataset rules, or scenario names.
```

### 6.1 Q1 student prompt

输入：

- 4 张 stitched RGB history。
- 当前 road-only `MEMORY`，只包含 `BELIEVED_RS` 和 `EGO_TO_GOAL_XY`，不包含
  `BELIEVED_EVENT`。
  `EGO_TO_GOAL_XY` 必须来自当前帧 meta `next_target_points[-1]` 转 ego frame，
  和 v3/v4/LeadMoT final goal 同源。
- `RS_CHOICES` A-E。

Prompt 模板：

```text
{memory_text}

[RS_CHOICES]
A. Ordinary same-direction drivable road: ...
B. Bidirectional single-lane or opposing-lane-sharing road: ...
C. Highway, ramp, merge, split, or exit structure: ...
D. Signalized intersection: ...
E. Unsignalized or priority-controlled intersection: ...
[/RS_CHOICES]

[QUESTION_1]
Analyze the latest frame in the RGB history.
Decide:
1. the current road-structure option from RS_CHOICES;
2. whether an unusual event is currently happening or still affecting the ego vehicle.

Use visible road geometry, lane layout, traffic lights or stop/yield cues, nearby
actors, ego-path conflicts, and image-visible weather or visibility cues. Do not
use a scenario name. If the evidence is weak, keep the memory unless contradicted.

Output exactly these concise CoT lines:
Scene Description: <1-2 concise sentences about visible weather/visibility, lane markings, road layout, traffic lights/signs, surrounding motion, and goal direction>
Critical Object Description: <1-2 concise sentences naming up to 2-3 key actors or map cues, their locations/actions, likely next motion, and why they matter to ego>
Reasoning on Intent: <1-2 concise sentences using motion, signals, lanes, ego state, and EGO_TO_GOAL_XY to decide RS and abnormality>
RS: <A|B|C|D|E> - <copy the chosen option meaning in your own words>
ABNORMAL: <YES|NO>
[/QUESTION_1]
```

Q1 parser：

- `RS:` 读取第一个 `A-E`。
- `ABNORMAL:` 读取 `YES/NO`。
- 缺失或非法时该项 loss 仍可对 generated tokens 做 teacher KL，但 memory 不更新。

### 6.2 Q1 teacher prompt

Teacher 是 privileged base Qwen：同样看图，但 user prompt 额外包含 GT 和 XML weather。
Student 不看 XML weather；它只能从 RGB 中判断天气 / 能见度。XML weather 只用于
teacher 分析监督，让 teacher target 在天气描述上更稳定。若 XML weather 与 RGB
可见天气 / 能见度冲突，teacher target 必须以 RGB 可见证据为准，并可把 XML weather
当作弱先验而不是最终描述。

```text
[REFERENCE]
XML_WEATHER: {weather_text}
ANSWER_RS: {gt_rs_option} - {gt_rs_description}
ANSWER_ABNORMAL: {YES|NO}
ANSWER_EVENT_FOR_REASONING: {gt_event_description}
[/REFERENCE]
```

Teacher target 文本仍清洗成学生视角：

```text
Scene Description: The RGB history shows weather, lane markings, traffic controls, surrounding motion, and a signalized intersection layout.
Critical Object Description: The relevant signal/vehicle/pedestrian/object is ...
Reasoning on Intent: The scene does / does not show an unusual event affecting the ego path because ...
RS: D - Signalized intersection with traffic-light control.
ABNORMAL: YES
```

禁止让 target 中出现 `ANSWER_`、`GROUND_TRUTH`、`REFERENCE`、`R1` 等私有字段名。

### 6.3 Q2 student prompt

只有 Q1 parsed RS 正确时才进入 Q2。Q2 使用 Q1 输出的 RS 作为当前 RS；
训练、评估和 probe 的模型路径都把 Q2 当作 Q1 assistant 输出后的第二轮 user turn，
通过 Q1 KV cache 续接，不重新对同一帧 fresh prefill。训练时如果 Q1 RS 错，跳过
Q2 并 reset 下一帧。

Q2 的 prompt 前缀使用 road + event memory：`BELIEVED_RS` 和 `BELIEVED_EVENT`
都只写自然语言描述，仍然不写 A-E / RE / U-E* 这类局部选项或标签代码。

若 Q1 `ABNORMAL=NO`：

```text
[EVENT_CHOICES under RS={chosen_rs_option}]
B. A pedestrian, cyclist, or small vulnerable road user crosses or laterally enters the ego path.
A. No unusual event: continue the regular behavior for this road structure ...
C. Another vehicle violates the expected intersection rule and creates conflict.
[/EVENT_CHOICES]

[QUESTION_2]
Decide the current event from EVENT_CHOICES. The choices have already been
filtered to events that are possible for the current road structure and this
route type. You judged in Question 1 that no unusual event is active. Confirm
the current event from the listed choices. If the only listed choice is RE, use
the analysis to explain which regular behavior is visible under the current road
structure. Choose a U-E option only when it is listed and visibly affects the ego
vehicle.

Output exactly these concise CoT lines:
Scene Description: <one concise sentence continuing from Question 1 and the current RS>
Critical Object Description: <1-2 concise sentences naming up to 2-3 event-relevant actors or cues, or stating that no critical object is visible>
Reasoning on Intent: <1-2 concise sentences explaining why the selected event is active or why regular behavior continues>
EVENT: <option letter> - <copy the chosen event meaning in your own words>
[/QUESTION_2]
```

若 Q1 `ABNORMAL=YES`：

```text
[EVENT_CHOICES under RS={chosen_rs_option}]
A. A pedestrian, cyclist, or small vulnerable road user crosses or laterally enters the ego path.
B. Another vehicle violates the expected intersection rule and creates conflict.
C. The forward road or intersection space is temporarily blocked or reopening.
[/EVENT_CHOICES]

[QUESTION_2]
Decide the current event from EVENT_CHOICES. The choices have already been
filtered to events that are possible for the current road structure and this
route type. You judged in Question 1 that an unusual event is active. Choose the
listed unusual event that most directly affects the ego vehicle right now. If no
unusual event is listed, or if the latest frame does not support any listed
unusual event, choose the regular-event option instead. Do not invent an event
that is not listed.

Output exactly these concise CoT lines:
Scene Description: <one concise sentence continuing from Question 1 and the current RS>
Critical Object Description: <1-2 concise sentences naming up to 2-3 event-relevant actors or cues, or stating that no critical object is visible>
Reasoning on Intent: <1-2 concise sentences explaining the selected event or why regular behavior should continue>
EVENT: <option letter> - <copy the chosen event meaning in your own words>
[/QUESTION_2]
```

### 6.4 Q2 teacher prompt

Teacher receives:

- current RS after Q1
- Q1 abnormal decision
- raw event set
- resolved single-label target after applying §4.3
- previous event memory

Teacher target examples:

No abnormal:

```text
Scene Description: Continue from the current signalized intersection decision.
Critical Object Description: No pedestrian, vehicle cut-in, obstacle, or blocked
intersection space interrupts the ego path.
Reasoning on Intent: The vehicle should keep the regular traffic-light intersection
behavior under the current signalized intersection structure.
EVENT: A - No unusual event; obey normal traffic-light intersection rules.
```

Abnormal:

```text
Scene Description: Continue from the current road-structure decision.
Critical Object Description: A vulnerable road user is crossing laterally into
the ego vehicle's intended path.
Reasoning on Intent: The interruption is not merely normal lane keeping or signal
compliance. This matches the event option about a pedestrian or cyclist crossing
the ego path.
EVENT: A - A pedestrian, cyclist, or small vulnerable road user crosses or
laterally enters the ego path.
```

---

## 7. OPSD Loss

v5 默认继承 v3 的核心思想：student 先自由生成，memory 由 student rollout 推进；
teacher 不提供 hard answer text 给 student，而是在同一批 student token 上给
privileged teacher 分布，并只在监督 span token 上做 forward-KL。

### 7.1 Loss type

默认：

```text
loss_type = forward_kl_teacher_to_student
```

公式与 v3 一致：

```python
teacher_probs = softmax(teacher_logits / T)
student_log_probs = log_softmax(student_logits / T)
loss = KL(teacher_probs || student_log_probs) * T * T
```

可选 debug：

- `--loss-type ce_teacher_text`：用 teacher target 文本做 hard CE，仅用于 smoke，
  不作为主实验。

### 7.2 Token span 权重

Q1:

- structured CoT lines (`Scene Description / Critical Object Description / Reasoning on Intent`): `0.2`
- `RS` option letter + description span: `1.2`
- `ABNORMAL` value span: `0.8`
- formatting tokens / prompt tokens: `0`

Q2:

- structured CoT lines (`Scene Description / Critical Object Description / Reasoning on Intent`): `0.2`
- `EVENT` option letter + description span: `1.2`
- formatting tokens / prompt tokens: `0`

默认总 loss：

```python
loss = q1_analysis + q1_rs + q1_abnormal + q2_analysis + q2_event
```

TensorBoard 会同时记录总 loss 和未加权 KL 分项：

- `train/loss_frame`
- `train/loss/q1_analysis`
- `train/loss/q1_rs`
- `train/loss/q1_abnormal`
- `train/loss/q2_analysis`
- `train/loss/q2_event`

Q1 分项按有效 frame 平均；Q2 分项按实际进入 Q2 的 frame 平均。

若 Q1 RS 错误：

- Q1 loss 正常计算。
- Q2 不触发，Q2 loss = 0。
- 下一帧 memory reset。

若 Q1 RS 正确但 abnormal 判断与 GT 不一致：

- Q2 仍进入同一个逐帧 `allowed_events` 折叠候选池。
- Teacher resolver 按 GT 选择 RE 或 UE；如果 Q1 abnormal 判断错了，Q2 正是训练
  模型在同一帧把 EVENT 修正回正确候选的地方。
- 只有当 GT event 不在逐帧 allowed events 中时，才记录 `q2_candidate_mismatch`；
  这通常意味着标定输出或旧 fallback 候选表需要复查。

### 7.3 生成与 teacher forward

每个问答 step：

1. Student adapter enabled，自由生成 token，保存未裁剪 token ids、decoded text
   和对应 KV state。
2. Parser 从 decoded text 读 RS / ABNORMAL / EVENT。
3. Teacher `disable_adapter()`，使用 privileged prompt，在同一批 student token 上 forward，
   得到 teacher logits 后立即按 target spans 裁剪。
4. Student 在同一批 token 上 forward，logits 也立即裁剪到 target spans。
5. 对裁剪后的 teacher/student span logits 做 weighted forward-KL；完整 vocab logits
   不跨 loss 计算长期保留。

Qwen3-VL 增量 decode 必须复用 `qwen3vl_local/mrope_utils.py` 的
`qwen3vl_incremental_forward`，禁止走 PEFT wrapper 的 `generate` /
`prepare_inputs_for_generation`。

训练阶段不能做基于输出字段的结构早停：即使 student 已经生成出 `ABNORMAL:` 或
`EVENT:`，也不能立刻截断 rollout。OPSD 需要让完整分析 token 和离散答案 token
共同接受 teacher logits 监督；字段早停会系统性缩短 CoT，改变训练分布，只能作为
独立推理加速实验另行评估，不能进入 v5 训练默认路径。

### 7.4 Batched Qwen 分阶段实现

目标是让 H20 的显存和算力真正用于同一 rank 内的多 frame Qwen forward，而不是只靠
DataLoader batch 攒更多 route。实现分阶段推进：

1. **阶段 1：batched Q1/Q2 student rollout**（已落地为 `QWEN_BATCH_SIZE`）
   - 同一 rank、同一 timestep 内收集多条 route 的 frame；
   - Q1/Q2 student rollout 允许 mixed-length padded batch，用一次 Qwen prefill/generate
     采样多条 route 的 Q1/Q2 文本/token；
   - prefill 首 token logits 按 `attention_mask` 找每条样本最后一个真实 token，兼容
     left/right padding；
   - repetition penalty 只看真实 prompt token，padding token 不进入 seen set；
   - padded KV 只用于 no-grad 采样，不写回 memory；训练 scoring 由 chunk 级
     parallel KL 路径重新构造 batched student/teacher prompt state；
   - 某个样本预测 EOS 时，从 active batch 中移除并保存追加 EOS 前的干净 KV；
   - 后续 teacher 与 KL 默认走 chunk 级 batched scoring，但 rollout batch 与 KL
     autograd 微批独立；
   - 若 processor/cache 兼容失败，自动 fallback 到单帧旧路径；CUDA OOM 直接中止，
     不在 OOM 后继续降级运行。
2. **阶段 2：batched Q2 student rollout**（已落地）
   - 只对 Q1 RS 正确的子集做 batch；
   - Q2 也允许 mixed-length padded full-dialog rollout；
   - padded Q2 KV 同样只用于 no-grad 采样文本/token；
   - parallel KL 不使用 `q1_ids -> q1_text -> full-dialog tokenizer` 回环，而是按旧逐帧
     语义把精确 `q1_ids` 追加到 Q1 KV 后再追加 Q2 user turn。
3. **阶段 3：batched teacher/KL forward**（已落地为 `--parallel-kl` 默认开启）
   - 对不同长度 rollout ids 做 padding；
   - 按每个样本自己的 span positions 取 logits；
   - padding token 不进入 loss，不污染 KV。
   - teacher/student prompt state、rollout token scoring 和 span KL 在同一 rollout
     chunk 内按 `PARALLEL_KL_MICROBATCH_SIZE` 拆分；默认 8 路 rollout 对应 `4+4`
     KL 微批，每个微批立即 backward，不同时保留两份 autograd graph。
   - 显存峰值主要来自 `KL microbatch x context length` 的 attention activation 与
     student logits。1024 token 上限保持不变；若 4 路 KL forward OOM，只把当前微批
     二分为 `2+2`，不降低输出长度、不重新 rollout。
   - 自适应二分只包住尚未 backward 的 forward/scoring；backward OOM 或普通异常直接
     中止，禁止在已有部分梯度后整块 fallback 导致重复累计。
   - `PARALLEL_KL=0` / `--no-parallel-kl` 可切回旧逐帧 teacher/KL；正式开大仍建议
     用小 run 对比 loss/解析率曲线。

阶段 1/2 使用方式：

```bash
BATCH_PROFILE=max_util GPU_IDS=0,1,2,3 \
bash qwen3vl_local/sft_v5/train.sh ddp
```

`QWEN_BATCH_SIZE>1` 必须配合 `PER_DEVICE_BATCH_SIZE>1`，否则同一 timestep 没有多个
route/frame 可以并行。

阶段 1 开大前必须做真实模型对照：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/test_batched_qwen_smoke.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --num-cases 2 \
  --output-json checkpoints/sft_v5_runs/batched_qwen_smoke.json
```

该 smoke 默认优先挑 Q1 input length 不同的两帧制造 padding 压力，并对比 single Q1
与 batched/grouped Q1，包括首 token、完整生成文本、`q1_ids`、同一 `q1_ids` 上训练
KL 路径 logits 的 max/mean abs diff，以及 Q1 KV 后续接 Q2 的输出；训练前默认不传
`--adapter-dir`，只验证普通 Qwen 路径。若要强制确认真的跑到 size>=2 的
batched rollout，必须加：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/test_batched_qwen_smoke.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --num-cases 2 \
  --candidate-pool 256 \
  --require-batched-group \
  --no-prefer-different-lengths \
  --check-parallel-kl \
  --output-json checkpoints/sft_v5_runs/batched_qwen_smoke_require_batch.json
```

报告里 `actual_batched_group_sizes` / `actual_batched_frames` 才是真实 batched rollout 是否
被测到的证据；`--check-parallel-kl` 会额外比较 parallel KL 与旧逐帧 KL 的总 loss、
逐 case loss 和 Q1/Q2 loss parts，三者都在 `--parallel-loss-atol` 内才算通过。

### 7.5 OPSD 的“采样数据”定义

v5 当前实现是同步 on-policy OPSD，而不是离线 teacher 数据生成，也不是 v4 那种
collector/learner 异步 replay：

- “采样数据”指当前 student 在本 step 对 Q1/Q2 自由生成出来的 token、解析结果和
  对应 KV state。
- 这些数据只在当前 `_run_frame()` 内存中临时存在，不写 replay，不跨 step 复用。
- Teacher 只在同一批 student rollout token 的监督 span 上提供 privileged logits，
  用于 forward-KL；teacher 不提前物化 target dataset。
- torchrun 多进程下每张 H20 都同时承担 rollout 采样和训练反传角色，所以四卡默认是
  `4` 张卡同步边采样边训练。

如果后续要做“几张卡采样、几张卡训练”的真正异步 OPSD，需要新增 v5
`collector -> replay -> learner` 路线。H20 四卡的自然起点可以是 `3 collector + 1 learner`
或 `2 collector + 2 learner`，但这属于 v5 off-policy 改版，不是当前 `train.py`
已经实现的同步多进程路线。

---

## 8. torchrun + sequence padding 训练

### 8.1 Dataset / sampler

`RouteSequenceDataset` 每个 `__getitem__` 返回一条 route sequence row。

torchrun 多进程下：

```python
sampler = LengthBalancedDistributedSampler(
    train_ds,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=seed,
)
loader = DataLoader(
    train_ds,
    batch_size=per_device_batch_size,
    sampler=sampler,
    collate_fn=collate_route_sequences,
)
```

默认 sampler 是 `length_balanced`，不是 PyTorch 原生 `DistributedSampler`。它在保证
每个 rank route 数 / batch 数一致的前提下，按 route frame 数贪心分配，减少
`valid_local` 在 rank 间差异过大导致的 collective 等待。训练开始时 rank0 会打印：

```text
[sampler] epoch=0 mode=length_balanced local_rank0_frames=... global_frames=... avg_per_rank=... min_rank_frames=... max_rank_frames=...
```

如果需要和旧行为做对照，可以显式切回：

```bash
SAMPLER_MODE=distributed GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp
```

与 v3 不同，v5 不用 work-stealing + local-SGD；每个 optimizer step 前对所有
LoRA 参数手动 all-reduce 梯度。这里不能使用 `DistributedDataParallel(model)`
wrapper：Q2 是否触发由各 rank 的 Q1 student 输出决定，rank 间 forward 次数会不一致，
DDP wrapper 的 forward hook / buffer broadcast 可能产生 unmatched collective 并触发
NCCL watchdog hang。

显存策略：

- 每个有效 frame 的 Q1/Q2 OPSD loss 算完立刻 backward，只把 LoRA 梯度留在参数上；
  不把一整条 route sequence 的 Qwen 计算图累加到 batch 末尾。
- Forward-KL 只在 `target_spans_q1/q2` 标出的 token 位置上计算；teacher/student
  完整 vocab logits 会尽快裁剪成 span logits，避免两份大 logits 长时间同时常驻。
- `train.sh` 默认设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，降低
  KV/logits 张量反复申请释放时的 allocator 碎片化。

默认优化策略不是等待完整 DataLoader route batch，而是
`UPDATE_MODE=streaming_frames`：每个 global timestep 内先完成全部 Q1/Q2 rollout、
teacher/student KL 和 backward，再 all-reduce 本 timestep 的实际有效 frame 数。
累计达到 512 个 global frame，或最迟达到 32 个 global timestep，就在该完整
timestep 边界同步 LoRA 梯度并执行 optimizer step。更新后保留各 route 的离散
RS/EVENT memory，下一帧使用更新后的 student；不能在同一帧 Q1 与 Q2 之间更新。

窗口内每个 frame loss 先除以 effective target frame 数再 backward；同步时梯度跨
rank 求 SUM，再乘 `effective_target / actual_global_frames`，最终严格等价于所有 rank
实际有效 frame 的平均梯度。没有本地有效 frame 的 rank 也必须补零梯度并参加相同
collective。同步实现按 device/dtype 将 LoRA 梯度合并成约 64 MiB bucket，减少大量
小参数逐个 all-reduce 的 NCCL 启动开销。`GRAD_ACCUM` 在流式模式中作为窗口倍率：默认 1 对应 512 frame / 32
timestep，设为 2 对应 1024 / 64。epoch 尾部不足阈值的窗口必须 flush。
adapter 元数据必须同时保存原始阈值与 effective 阈值、LR 和梯度同步策略，不能只靠
启动日志恢复优化器口径。

`UPDATE_MODE=batch` 仅保留为旧实验兼容模式，此时 `GRAD_ACCUM` 才表示累计多少个
DataLoader batch；正式训练不建议使用，因为一批长 route 可能累计上万帧并让一次
optimizer update、TensorBoard loss 和 scheduler step 延迟数小时。

### 8.2 Collate padding

Collate 输入是 `List[RouteSequence]`。`collate_fn` 只在 DataLoader 所在进程做
local padding / local length，不调用 `dist.all_reduce`，避免 `num_workers>0` 时在
worker 进程里触碰 distributed runtime。主训练进程拿到 batch 后再 all-reduce
`max_T_local`。

`collate_fn` 输出：

```python
{
  "routes": [...],
  "max_T_local": int,
  "valid_mask": BoolTensor[B, max_T_local],
  "frame_rows": List[List[Optional[FrameRow]]],  # padded to max_T_local
}
```

步骤：

1. `collate_fn` 计算本 rank batch 内 `max_T_local`，padding 到 local max。
2. `train.py` 主进程对 `max_T_local` 做 `dist.all_reduce(op=MAX)` 得到
   `max_T_global`。
3. 主进程把 `valid_mask` / `frame_rows` 右侧补齐到 `max_T_global`，或在外层 loop
   中把 `t >= max_T_local` 视为 padding。
4. Padding frame 不读图、不进 Qwen、不产生 loss，只占位保证所有 rank 的外层时间
   循环长度一致。

这样每个 rank 的外层时间循环都是：

```python
for t in range(max_T_global):
    active_indices = valid_mask[:, t].nonzero()
    if not active_indices:
        continue
    run q1/q2 for active samples at time t
```

注意：DDP 不要求每个 rank 的图像 token 长度完全一致，但统一 `max_T_global`
能让日志、skip、梯度累积边界更容易审计，也满足“loader 新数据后按最长 sequence padding”
的要求。

### 8.3 梯度累积

`train.sh ddp` 默认 `BATCH_PROFILE=max_util`，即
`per_device_batch_size=8`、`qwen_batch_size=8`、`parallel_kl_microbatch_size=4`、
`grad_accum=1`；
`single/check` 模式默认仍保守使用 `1/1`。如果显存允许：

- batch 内多个 route 按时间步交错推进；
- 每个有效 frame 的 Q1/Q2 loss 按固定 target normalizer 缩放后立即 backward；
- 每个 timestep 都用轻量标量 all-reduce 汇总实际 global frame 数；
- 达到 frame/timestep 阈值后，对 LoRA 梯度做 SUM all-reduce，再按实际 global frame
  数修正为 frame 等权平均，而不是 rank 等权平均；
- optimizer step 之后保留 sequence memory，继续当前 DataLoader batch。

推荐四卡 H20 训练口径：

```text
BATCH_PROFILE=max_util
per_device_batch_size=8
qwen_batch_size=8
parallel_kl_microbatch_size=4
grad_accum=1
update_mode=streaming_frames
target_global_frames_per_step=512
max_timesteps_per_step=32
learning_rate=1e-5
outer_stride=1
max_frames_per_route=0  # 0 means full route
```

训练日志/TensorBoard 必须能审计“边采样边训练”和 DDP padding：

- `train/loss_frame`
- `train/loss/q1_analysis`
- `train/loss/q1_rs`
- `train/loss/q1_abnormal`
- `train/loss/q2_analysis`
- `train/loss/q2_event`
- `train/q2_trigger_rate`
- `train/q1_rs_acc_window`
- `train/q1_abnormal_acc_window`
- `train/q2_event_acc_window`
- `train/q2_invalid_output`
- `train/reset_next`
- `train/rollout_tokens_per_frame`
- `train/q1_token_cap_hit_rate`
- `train/q2_token_cap_hit_rate`
- `train/global_frames_per_step`
- `train/timesteps_per_step`
- `train/update_reason_code`
- `train/learning_rate`
- `time/grad_sync_seconds`
- `time/optimizer_step_seconds`
- `ddp/grad_allreduce_buckets`
- `qwen/q1_batched_frame_rate`
- `qwen/q1_grouped_frame_rate`
- `qwen/q1_batched_frame_rate_grouped`
- `qwen/q1_batched_groups`
- `qwen/q1_singleton_groups`
- `qwen/q1_length_seconds_per_chunk`
- `qwen/q1_grouped_seconds_per_chunk`
- `qwen/q2_batched_frame_rate`
- `qwen/q2_grouped_frame_rate`
- `qwen/q2_batched_frame_rate_grouped`
- `qwen/q2_batched_groups`
- `qwen/q2_singleton_groups`
- `qwen/q2_length_seconds_per_chunk`
- `qwen/q2_grouped_seconds_per_chunk`
- `time/frame_q1_student_seconds`
- `time/frame_q1_teacher_seconds`
- `time/frame_q1_loss_seconds`
- `time/frame_q2_rollout_seconds`
- `time/frame_q2_teacher_seconds`
- `time/frame_q2_loss_seconds`
- `parallel_kl/frame_rate`
- `parallel_kl/seconds_per_chunk`
- `parallel_kl/microbatches_per_chunk`
- `parallel_kl/frames_per_microbatch`
- `parallel_kl/oom_splits`
- `parallel_kl/fallbacks`
- `ddp/padding_rate`
- `ddp/max_T_global_avg`

这些指标在 logging window 内先按 rank 本地累计，再 `all_reduce(SUM)` 到全局口径，
rank0 打印一行 `[train] ...` 并写 TensorBoard。

`qwen/q1_batched_frame_rate` 是所有训练 Q1 frame 的真实 batched rollout 比例；如果它长期接近
0，说明同一 timestep 内有效 frame 不足或 batch fallback 频繁。此时先看日志里的
`[warn] q1 batch fallback`，再考虑退回 `QWEN_BATCH_SIZE=1`。

多 batch 运行 demo：

```bash
BATCH_PROFILE=max_util \
LOGGING_STEPS=1 PROGRESS_FRAMES=20 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp
```

该命令是当前推荐的四张 H20 配置：四卡各 1 个 rank，每卡 8 条 route sequence，
全局约 32 条 sequence，并在每个 rank/timestep 内尝试 8 路 Q1/Q2 student rollout batch。
现在 `GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp` 默认就是
`BATCH_PROFILE=max_util / PER_DEVICE_BATCH_SIZE=8 / QWEN_BATCH_SIZE=8`；启动时
launcher 会打印 `[batch]` 配置，第一条 `[batch-start]` 也应显示 `routes=8`
与 `qwen_batch=8`。若 8 路 OOM、频繁 fallback 或单步明显变慢，可用
traceback 判断阶段：若 OOM 位于 `_opsd_loss_batch_states`，先设
`PARALLEL_KL_MICROBATCH_SIZE=2`，保持 8 路 rollout；只有 OOM 位于 Q1/Q2 grouped
student rollout 时，才用 `BATCH_PROFILE=balanced/debug` 降低 rollout batch。若
`qwen/q1_batched_frame_rate` 长期接近 0，则先查
`[warn] q1 batch fallback`，必要时回到 `QWEN_BATCH_SIZE=1`。
若 8 路已经稳定显示 `batched_frames=8`，但 GPU util 仍只有中等水平，先确认启动日志里
`[parallel] PARALLEL_KL=1`，并观察是否出现 `[chunk-train] ... parallel_kl=1`。
`[chunk-train]` 应同时打印 `kl_microbatches=[4, 4]`；偶发
`[warn] parallel KL adaptive split ... [2, 2]` 可继续训练，长期出现则固定改成 2。

训练阶段不按 `ABNORMAL:` / `EVENT:` 字段早停；默认仅保留 `MAX_NEW_TOKENS_Q1=1024`
和 `MAX_NEW_TOKENS_Q2=1024` 作为防无限生成的安全上限。正式训练不建议使用很小的
token cap；只有 OOM 定位或极短 smoke 才临时调低。这里的语义是“无字段早停 +
EOS / `<|im_end|>` 自然停止 + 1024 token 安全上限”，不是完全无限生成。

实现注释要求：

- 代码中必须保留中文注释解释 `grouped` 与真正 `batched` 的区别，避免后续把
  `qwen/q1_batched_frame_rate_grouped` 误读成全局 batch 生效率。
- Cache 切片、last-valid logits、padding token 排除、EOS active batch 移除、KL forward
  OOM 仅在 backward 前二分、loss_slots 归一化、无监督 span 返回 graph-connected zero 这些位置都属于
  v5 correctness 关键点，修改时必须同步更新相邻注释。
- batched Qwen 的 `rope_deltas` 可能来自不同 Qwen/Transformers 版本，形状既可能是
  `(batch,1)`，也可能是 `(1,batch)`；KV 切片和 `mrope_utils.py` 的 decode position
  计算必须同时兼容两种方向，v5 内部 KVState 边界统一保存为 `(batch,1)`。若日志出现
  `Target sizes: [1, -1]. Tensor sizes: [2, 1]` 一类 `[warn] q1 batch fallback`，
  优先检查 batched prefill 出口、active batch 缩小时和 append-token 后的
  `rope_deltas` 是否仍保持每个样本一行。纯文本 incremental append 不改变图文
  M-RoPE delta，因此 append helper 必须沿用输入 state 的 `rope_deltas`，不要把
  `outputs.rope_deltas` 写回 KVState；后者可能是 Qwen 模型对象缓存的 stale batched
  delta。
- `test_batched_qwen_smoke.py` 的默认 mixed-length 模式和
  `--require-batched-group` 模式必须在代码注释和文档中保持一致：默认模式验证
  mixed-length padded rollout，强制模式要求真实 batched_frames>=2。

### 8.4 Padding 与 memory reset

每个 batch 初始化 `memory[B]`：

- 第一个有效 frame 前：`GT_RS(first frame) + RE`。
- Padding timestep：memory 不变。
- Q1 RS 错：标记 `reset_next_frame[b]=True`。
- 下一个有效 frame 开始时，若 `reset_next_frame[b]`：
  `memory = GT_RS(current frame) + RE`，然后清标记。

如果某条 route sequence 在中间出现缺 RGB / 缺 weather / 缺 label：

- build 阶段应过滤或截断。
- train 阶段遇到仍缺失的 frame，跳过该 frame 并记录 `runtime_frame_skip`，
  不让 DDP rank 崩掉。

---

## 9. Eval 指标

Frame-level 指标：

- `rs_acc`: Q1 RS option accuracy。
- `abnormal_acc`: Q1 YES/NO accuracy。
- `event_acc_when_rs_correct`: Q2 event accuracy，只统计 Q1 RS 正确且 Q2 触发帧。
- `ue_acc`: raw abnormal=true 且 Q2 触发时的 UE 准确率。
- `re_acc`: abnormal=false 且 Q2 触发时的 RE 准确率。
- `q2_trigger_rate`: Q1 RS 正确后进入 Q2 的比例。
- `rs_wrong_reset_count`: 因 Q1 RS 错导致下一帧 reset 的次数。
- `candidate_mismatch_count`: 当前 scenario + 当前 RS 交集候选池不含 GT event 的次数。

Route-level 指标：

- `route_rs_all_correct_ratio`
- `route_abnormal_f1`
- `route_ue_macro_f1`
- `mean_resets_per_100_frames`
- `mean_valid_frames_per_route`

Probe dump：

```text
probe*/
  manifest.json
  route_<idx>__<scenario>__<route_id>/
    timeline.json
    timeline.png
    frame_<frame_id>/
      rgb_00.jpg
      rgb_01.jpg
      rgb_02.jpg
      rgb_03.jpg
      rgb_paths.json
      q1_student_prompt.txt
      q1_student_output.txt
      q1_teacher_prompt.txt
      q1_teacher_target.txt
      q2_student_prompt.txt
      q2_student_output.txt
      q2_teacher_prompt.txt
      q2_teacher_target.txt
      step1_user.txt
      step1_student.txt
      step1_teacher_user.txt
      step1_teacher.txt
      step2_user.txt
      step2_student.txt
      step2_teacher_user.txt
      step2_teacher.txt
      memory_before.json
      memory_after.json
      flags.json
      labels.json
```

可视化方法单独记录在 `SFT_V5_RUN.md` 的“Probe / 可视化输入输出”章节与
`SFT_V5_VISUALIZATION_RECORD.md`。v5 probe 明确分成三类：

- 训练前 base Qwen OPSD 能力体检：`--with-model --with-teacher-model`，不传
  `--adapter-dir`，不加载任何 LoRA，让默认 Qwen 分别跑 student prompt 和
  privileged teacher prompt，用 `q*_student_output.txt` / `q*_teacher_output.txt`
  判断普通 Qwen 的基础能力与 prompt 合同是否足够支撑 OPSD。
  teacher model 的自由生成输出必须和 student 一样直接从 `Scene Description:` 开始，
  并按 `Scene Description / Critical Object Description / Reasoning on Intent /
  RS 或 EVENT` 写完整结构；如果复读 MEMORY、choices 或 REFERENCE，说明旧 demo 需要
  重跑或 prompt 合同还要继续收紧。
- 训练后 adapter 学生可视化：`--with-model --adapter-dir ...`，只加载训练后的
  student adapter，检查真实状态机下的 Q1/Q2 输出、memory 更新和 reset 行为。
- 静态 prompt / target 快检：不加载模型，只写 RGB、student prompt、teacher prompt、
  脚本化 teacher target、label、memory、flags 和 timeline。

`--with-teacher` 只保留为 v3 兼容标志；真正生成 teacher 模型文本必须显式使用
`--with-teacher-model`。

### 9.2 代码注释维护要求

`AutoMoT/qwen3vl_local/sft_v5/` 下代码采用中文注释维护：

- 函数/docstring 描述入口、输入输出和状态机职责。
- 关键逻辑块必须解释设计原因，而不只是复述代码行为；当前已覆盖
  `allowed_events` 优先级、`R-E* -> RE` 折叠、RS/EVENT 双标签单标签化、
  Q1 RS 错误截断、OPSD teacher/student logits 对齐、DDP local/global padding、
  训练前纯 base Qwen 体检不加载 LoRA、probe flags 审计字段和测试回归意图。
- 后续修改标签协议、prompt、memory、loss、probe 或 DDP 训练逻辑时，需要同步更新
  相邻代码注释和 `SFT_V5_RUN.md`，避免文档与实现脱节。

---

## 10. 训练运行草案

Build dataset：

```bash
python qwen3vl_local/sft_v5/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_v5_data \
  --val-ratio 0.1 \
  --seed 42
```

Train：

```bash
DDP_GPU_COUNT=4 bash qwen3vl_local/sft_v5/train.sh ddp
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp
```

Shell launcher：

```bash
bash qwen3vl_local/sft_v5/train.sh ddp
```

默认环境：

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
TOKENIZERS_PARALLELISM=false
```

保存：

- 只保存 LoRA adapter delta。
- 写 `sft_v5_adapter_config.json`，记录：
  - dataset version
  - RS/EVENT label version
  - RE folding policy
  - DDP world size
  - LoRA vision scope
  - max_new_tokens
  - loss weights

---

## 11. 实现顺序

1. `labels.py`
   - 写 RS A-E、EVENT descriptions、RS-level event pools、RE descriptions。
   - 写 `resolve_rs_target(frame)`、`resolve_event_target(frame, student_event=None)`。
2. `prompts.py`
   - 写 Memory dataclass。
   - 写 Q1/Q2 student prompt、teacher prompt、target builder。
   - 写 parser、span finder、GT leak checker。
3. `build_dataset.py`
   - 读取 collection result。
   - 过滤 noScenarios、异常时长、data_missing_skip。
   - 生成 route sequence index。
   - 写 summary：route/frame count、RS/EVENT 分布、skip 分布。
4. `train.py`
   - 先单卡跑通 `--max-routes 2 --max-frames-per-route 8`。
   - 再接 DDP / LengthBalancedDistributedSampler / 主进程 global sequence padding。
   - 再接 teacher forward-KL 与 TensorBoard。
5. `eval.py` / `probe.py`
   - 先自由生成评估。
   - 再加 teacher 对照 dump。
6. 测试：
   - `python qwen3vl_local/sft_v5/test_memory_update.py`
   - `python qwen3vl_local/sft_v5/test_dataset_contract.py`
   - `python qwen3vl_local/sft_v5/test_streaming_optimizer.py`
   - `python qwen3vl_local/sft_v5/test_parallel_kl_microbatch.py`
   - `python qwen3vl_local/sft_v5/check_loss_mask.py`
   - `python -m py_compile qwen3vl_local/sft_v5/*.py`

---

## 12. 已拍板规则

- RS 错误后的“结束采样”粒度与 v3 一样：只结束当前帧。Q1 RS 错则跳过本帧
  Q2，下一有效帧恢复 GT RS + 当前 GT RS 下默认 RE 后继续同一条 route sequence。
- Q2 候选优先使用每帧 `frame_event_annotation.allowed_events`；只有缺失时才 fallback
  到 `scenario_event_candidates ∩ EVENT_CANDIDATES_BY_RS[current_rs]` 静态表。Prompt
  不显示 scenario 名。
- Q2 正常监督只训练 `RE` vs `U-E*`。原始 `R-E*` 不作为 v5 分类目标，只保存在
  `event_code` / `regular_event_codes` 审计字段，并用于生成更细的 RE 解释。
- Q2 允许退化为单选 RE，并把它当作“确认无异常”的训练帧；不得为了避免单选题硬塞
  不属于逐帧 allowed events 的负例候选。
- 逐帧 allowed events 中的所有 `R-E*` regular 分支在 prompt 里显示为 `RE`。
  训练内部保留原始 `R-E*` 作为 `event_code` / `regular_event_codes` 审计字段。
- Q2 的选项字母每帧可复现随机化，不能让 `A/B/C/...` 固定绑定到某个 EVENT。
- Q1 输出字段固定为 `Scene Description / Critical Object Description /
  Reasoning on Intent / RS / ABNORMAL`。天气、道路、车道线、交通灯和周围运动
  都压缩写进 `Scene Description`，不单独做天气分类 loss。
- XML weather 只给 teacher。Student 只能从 RGB 中判断天气 / 能见度；teacher 可用 XML
  weather 生成更稳定的分析监督，但 XML 与 RGB 冲突时以 RGB 可见证据为准。
- `review_required=true` 正常参与训练；只有数据结构异常、缺 meta/XML/RGB/annotation、
  异常时长 route、`noScenarios_result.json` 这类明确排除项不训练。
- R3 不开放 UE。当前 RS=R3 时，Q2 候选通常只包含 RE，但 RE 文案必须覆盖当前
  `regular_event_codes` 对应的多个 regular mode，例如跟车/速度匹配、目标导向变道、
  常规合流/分流/驶出。
- Q2 选项随机化粒度为每个 frame 可复现随机：同一 `dataset_version + run_id + frame_id
  + seed` 永远得到同一个 `event_option_map`。
