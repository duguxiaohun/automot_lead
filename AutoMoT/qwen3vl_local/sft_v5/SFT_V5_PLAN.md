# SFT v5 方案说明：RS / EVENT OPSD + torchrun sequence padding

本文是 `qwen3vl_local/sft_v5/` 的实现蓝图。目标是仿照 `sft_v3`
的 offline on-policy self-distillation (OPSD) 思路，但把监督目标从
CARLA scenario / status / subgoal 改成 `AutoMoT/keyframe_filter` 标定出的
帧级 `ROAD_STRUCTURE` 与 `EVENT`。

v5 的关键变化：

- 数据源改为 `AutoMoT/keyframe_filter/collection_output/*_result.json` 的
  全帧标定结果，不再使用 `keyframes_all_scenarios.json` 的 5-anchor 子场景。
- 训练目标是双速率两问：
  1. `RS_SLOW`：保留三段分析，只输出道路结构 RS；稳定时以 4 帧为中心，默认在
     3/4/5 个 4Hz 帧中可复现地随机选择下一次复核间隔，错误/UNKNOWN/recovery
     时切回逐帧。
  2. `EVENT_FAST`：每个 RS gate 正确的帧运行，保留三段分析，直接在显式标注
     `[RE | REGULAR]` / `[UE | UNUSUAL]` 的混合候选中选择 EVENT。RE 就是
     regular/normal，任意 UE 就是 unusual/abnormal，不再单独询问 `ABNORMAL`。
     这相当于把旧概念中的 `EVENT_FAST_1` 和 `EVENT_FAST_2` 合并成一问。
- Memory 只保留 `RS` 与 `EVENT`，不再保留 scene/status/subgoal。
- 多卡训练改成 torchrun 多进程：DataLoader 每次取 route sequence，collate 只做本 rank
  的 local padding / local length；`train.py` 主进程再 all-reduce 当前 step 的全局
  最长 sequence，补齐 mask 后进入统一时间循环。默认每 512 个 global 有效 frame，
  或最迟 32 个 global timestep，在完整 timestep 边界手动 all-reduce LoRA 梯度并
  更新权重，不再等待完整超长 route batch。
- Prompt 全部使用英文，标签选项必须是自然语言描述，不训练模型只背裸标签名。
- v5 不改 Qwen 的视觉/文本注意力结构，不采用额外 asymmetric cross-attention；当前路线用
  aligned / omission / contradiction 数据课程和显式 memory age 近似“先验置信度”，便于
  继续复用现有 LoRA、KV 和 OPSD 训练链路。

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
  SFT_V5_VISUALIZATION_RECORD.md
  labels.py
  prompts.py
  build_dataset.py
  train.py
  train.sh
  eval.py
  metrics.py
  probe.py
  inspect_teacher.py
  check_loss_mask.py
  test_batched_kv_helpers.py
  test_batched_qwen_smoke.py
  test_memory_update.py
  test_dataset_contract.py
  test_checkpoint_probe.py
  test_parallel_kl_microbatch.py
  test_probe_selection_and_metrics.py
  test_streaming_optimizer.py
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
- `train.sh`
  `single/ddp/check` 统一 launcher；负责 GPU pin、batch profile、memory curriculum 默认值、
  run 目录防覆盖和 torchrun 参数展开。正式启动参数应先用 `DRY_RUN=1` 审计。
- `eval.py`
  大样本自由生成评估。使用同一套 Memory 和两问协议，不用 teacher 强制纠偏；
  指标按帧流式累计，可选把完整输入输出逐帧写入 JSONL。
- `metrics.py`
  `eval.py` 与 `probe.py` 共用的指标定义、混淆矩阵和流式 accumulator；每个指标必须
  同时声明中文含义与“越高越好 / 越低越好 / 仅诊断”。
- `probe.py`
  小样本定向 case-level dump。默认覆盖 UE 正例/边界/邻近 RE 硬负例、RS 变换/邻帧和
  稳定 RE；保存完整 RGB、messages、prompt、student/teacher 输出、memory transition 与 GT。
- `SFT_V5_VISUALIZATION_RECORD.md`
  规定 compact/review/full 三档 probe 产物、时间线字段、人工审查顺序，以及训练曲线中
  relation、age、随机 interval、复制与恢复指标的判读口径。
- `inspect_teacher.py`
  抽检 privileged teacher 的分析质量；只跑 base Qwen `disable_adapter()`，
  不训练。
- `check_loss_mask.py`
  静态检查 RS_SLOW analysis / RS 与 EVENT_FAST analysis / EVENT span 权重，防止 prompt 改动导致
  离散标签 loss 掉线。
- `test_*.py`
  覆盖 sequence/index 合同、memory age/扰动/延迟修复、可复现随机 RS 调度、streaming
  optimizer、parallel-KL 微批、checkpoint probe 和 batched KV helper。真实
  `test_batched_qwen_smoke.py` 会加载本地 Qwen/RGB，必须在有模型与 GPU 的服务器单独运行。

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
- `abnormal` 只是由最终 `event_label` 的 RE/UE family 派生出的数据审计布尔值；它不进入
  prompt、memory 或模型监督输出，也不是额外的 normal/abnormal 问题。
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

Prompt 短描述：
`A. Ordinary same-direction road; lane keeping/following or same-direction lane changes dominate.`

覆盖要点：

- 常规道路 / 同向可行驶道路。
- 普通车道保持、跟车、安全距离。
- 同向变道、绕障前后恢复、停车侧/路边通行但未压缩成对向借道。
- 环岛也归入 R1，即使 XODR 把它编码成 junction road。

### B = R2

Prompt 短描述：
`B. Narrow bidirectional/shared road; oncoming traffic or opposing-lane borrowing dominates.`

覆盖要点：

- 双向单车道 / 借对向车道道路。
- 对向车道参与决策。
- 自车可能因静态障碍借对向绕行。
- 对向车辆异常侵占时需要被动让行。

### C = R3

Prompt 短描述：
`C. Highway/ramp/merge/split/exit; speed matching, gaps, merging or diverging dominates.`

覆盖要点：

- 高速 / 快速路 / 主辅路。
- 匝道、合流、分流、驶出。
- 速度匹配、侧后方间隙、目标车道和主路车流关系。

### D = R4

Prompt 短描述：
`D. Signalized intersection; working traffic lights control right of way.`

覆盖要点：

- 信号灯路口。
- 红灯停、绿灯行。
- 信号灯左转 / 右转仍需观察冲突对象。
- 对方闯红灯时 RS 仍是 R4，突发由 UE 表达。

### E = R5

Prompt 短描述：
`E. Unsignalized/priority intersection; stop/yield, geometry, cross traffic or safe gaps control right of way.`

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
`labels.py` 和下列覆盖要点保留完整工程语义；实际输入 VLM 的文案使用
`prompts.py` 中的短描述，只保留行为主体、冲突类型和对 ego 的影响，避免在
memory、候选、REFERENCE 三处重复同一段长定义。

RE 不是一个原始 `keyframe_filter` code，而是 v5 为第二问引入的折叠标签。通用工程
语义如下（不直接整段复制进 prompt）：

```text
RE: No unusual event is currently interrupting the driving task; continue the
regular behavior implied by the current road structure.
```

不同 RS 下的 RE 完整解释。RE 是单一监督类，但它的语义要吸收当前帧
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
- 不再存在独立 abnormal 任务或可复用状态；EVENT 选择的 family 直接给出
  normal/abnormal。dataset 内的 `abnormal` 只保留为由 EVENT family 派生的审计字段。

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

EVENT_FAST 候选构造：

- 每帧显示逐帧 allowed events 折叠后的混合候选池：`RE + U-E*`；即使旧/损坏数据只列
  UE，也补一个 RE 负类对照；只有 regular events 时允许退化为单选 `RE`。
- 每个选项在文字前显式标记并展开 family，例如 `A. [RE | REGULAR] ...`、
  `B. [UE | UNUSUAL] ...`。模型一次
  选择就同时完成 normal/abnormal 与具体事件判断，不再先做二分类。
- parser 只输出 option letter，训练内部通过本帧 `event_option_map` 映射到 `RE`
  或具体 `U-E*`；选 RE 即 normal，选任意 UE 即 abnormal。

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

## 5. Memory 设计与错误记忆课程

v5.1 把 Memory 明确定义为“上一帧模型的未验证假设”，不是当前帧答案。RS memory
同时用于 Q2 的 EVENT 候选语义，因此不能直接删除；训练必须主动降低“复制 memory
就答对”的比例，并让学生经历连续错误状态。Q1 使用的跨帧纯文本为：

```text
[MEMORY]
PREVIOUS_RS_HYPOTHESIS: Ordinary same-direction road; lane keeping/following or same-direction lane changes dominate.
PREVIOUS_RS_HYPOTHESIS_AGE: 6 frames / 1.50 s
MEMORY_RELIABILITY: unverified; may be stale or wrong
EGO_TO_GOAL_XY=(+12.3, -1.5) m
[/MEMORY]
```

Q1 使用 road-only memory，不提前暴露 event；Q2 才在同一轮 Q1 之后使用
road + event memory：

```text
[MEMORY]
PREVIOUS_RS_HYPOTHESIS: Ordinary same-direction road; lane keeping/following or same-direction lane changes dominate.
PREVIOUS_RS_HYPOTHESIS_AGE: 6 frames / 1.50 s
PREVIOUS_EVENT_HYPOTHESIS: Regular same-direction following, lane keeping or lane adjustment; no active unusual conflict.
PREVIOUS_EVENT_HYPOTHESIS_AGE: 2 frames / 0.50 s
MEMORY_RELIABILITY: unverified; may be stale or wrong
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
    rs_age_frames: int   # RS hypothesis 连续未变化的 4Hz 帧数
    event_age_frames: int  # 当前 RS 条件下 EVENT hypothesis 连续未变化的 4Hz 帧数
```

UNKNOWN/no-prior：`Memory.rs_label/event_label` 允许内部值 `UNKNOWN`，prompt 显示
“No reliable previous ... hypothesis is available”，绝不能静默 fallback 成 R1/RE。
这里把“没有 memory”实现成固定 schema 内的 UNKNOWN/no-prior，而不是随机删除整个
`[MEMORY]` block：两者都不泄漏答案，但固定 schema 不会额外制造 train/deploy 格式漂移。

RS 与 EVENT age 在 RS 不变的普通帧中独立累加。route 首帧 age=0；进入下一个真实有效
帧时各自加 1；对应 label 真正变化时对应 age 归零，周期复核后仍输出同一 label 不重置。
但 EVENT 不是脱离道路结构的全局状态，而是条件状态 `EVENT | RS`：只要 RS hypothesis
真正改变（学生 Q1、RS corruption、UNKNOWN 注入或 delayed repair），旧 RS 下的 EVENT
立即失效为 `UNKNOWN, event_age_frames=0`；只有新 RS gate 下的 Q2 才能重新建立 EVENT。
padding、缺图 skip 不累加 age。这样 `event_age_frames=46` 只表示“在同一 RS 条件下该
EVENT 已连续 46 帧”，不会把旧道路结构的 46 帧错误带到新 RS；新 RS 同帧 Q2 重新确认
后 age=0，下一真实帧 RS/EVENT 才一起变成 1。

人工注入一个新错误 label 本身也是一次 memory 改变，因此该 contradiction 的 age 必须从
0 开始，不能随机伪造为较大的“旧时间戳”。如果学生没有依据当前 RGB 纠正它，这个错误
hypothesis 会随后续真实帧自然累加成 age=1、2、3……的 stale 样本。这样较老的错误样本
来自真实 closed-loop 失败轨迹，而不是 age 与标签互相矛盾的静态拼接。与此同时，长期稳定
且正确的 RS 也允许有较大 age，所以 age 只表示需要重新核验的先验强度，绝不能替代当前图像。

初始化与随机扰动默认值：

- route 首帧 RS/EVENT 各有 0.5 概率使用 GT，剩余为 UNKNOWN/no-prior。
- 当前 RS memory 原本正确时，以 0.05 概率替换成其它 R1-R5（contradiction/stale），
  以 0.07 概率改成 UNKNOWN/no-prior（omission）。
- 当前 EVENT memory 原本正确时，以 0.20 概率替换成其它 RE/U-E（contradiction/stale），
  以 0.12 概率额外改成 UNKNOWN/no-prior（omission），eligible 条件分布为
  `68/12/20`。旧值 0.25 已下调：RS hypothesis 变化本身现在会自然失效 EVENT 为
  UNKNOWN，继续使用 0.25 会重复制造 omission，把 Q2 omission 推到约 34%。
- 三类关系按模型真正看到的 Q1/Q2 prompt 统计：`aligned` 是 memory 与当前 GT 一致，
  `omission` 是 UNKNOWN/no-prior，`contradiction` 是已知但错误/陈旧的 label。注入概率是
  “当前 memory 原本正确且通过上游 gate”的条件概率，不等于最终训练样本比例。
- EVENT wrong 优先从本帧 `event_option_map` 里选择不属于当前多标签容错集合的可见候选，
  使 Q2 真能在“错误旧假设 vs 当前视觉证据”之间纠偏；单选题没有替代项时才退回
  全局 EVENT 表，表达一个已过期但当前候选不再允许的旧事件。双 UE 可接受集合中的
  另一个正确标签绝不能被审计成 wrong augmentation。
- EVENT repair/augmentation 只在 RS memory 经过本帧 RS 扰动后仍与 GT 对齐时执行。
  若本帧 RS hypothesis 真正变为错误/UNKNOWN，旧 EVENT 失效为 UNKNOWN，同时清空旧
  RS 语境下的 EVENT error streak/pending；若 RS 早已错误但本帧没有再次变化，则保持
  当前 UNKNOWN，不反复重置 age，也不制造学生根本没看到的 EVENT augmentation。
- 扰动只覆盖本帧输入；若学生复制错误 memory，错误会由 closed-loop 自然延续。已经
  错误/UNKNOWN 的 memory 不会每帧继续随机换标签，避免把连续纠偏任务退化成白噪声。
- 所有 draw 使用 `seed + route + frame + epoch` 的稳定 SHA-256 映射，DDP 与重跑可复现。

延迟修复：

- `RS_SLOW_INTERVAL=4`、`RS_SLOW_INTERVAL_JITTER=1`：RS 正确稳定时，每次 Q1 后
  按 `route + seed + last_query_ordinal` 从 3/4/5 帧均匀、可复现地抽取下一次
  RS_SLOW 间隔，中间帧只复用 RS memory 并运行 EVENT_FAST。jitter=0 才恢复固定 4 帧。
- route 首帧、RS memory 为 UNKNOWN、memory 与当前训练 GT 不一致、上一轮 RS
  错误或 repair pending 时，立即运行 RS_SLOW；错误期间每帧运行，直到自主恢复或
  delayed repair。RS 错误帧只停止本帧 EVENT_FAST。
- RS 默认连续 4 帧错误后只“申请修复”，每 2 个有效帧检查一次申请；学生在修复前
  自行答对会清空 streak/pending。
- EVENT 只有 EVENT_FAST 实际触发时才累计错误；默认连续 3 次错误后申请修复，并每帧检查，
  体现 EVENT 快变量的高频复核。
- 正式默认 `RS_REPAIR_MODE=EVENT_REPAIR_MODE=ground_truth`：只在上述
  patience 和 review slot 都满足后延迟写回 GT。这保证早期学生不会因为
  长期卡在 wrong/UNKNOWN 而让 EVENT 训练饿饿，也不会在错误后下一帧立刻纠正。
- `unknown` 模式保留为软擦除消融：它只去掉陈旧先验，不保证学生能退出
  UNKNOWN。在“模型只复制 memory”的压力测试中，纯软修复会让 RS 异常输入
  约 95.7%、Q2 gate 只剩约 4.3%，因此不是正式长训默认。
- 错误/陈旧 EVENT 必须由 EVENT_FAST 自己选择正确的 RE/UE 候选才算纠正，脚本不从
  其它字段推导或覆盖 EVENT。
- 强制修复仅训练期使用。修复帧答对单列为
  `recovered_after_forced_repair`，不能计入 `self_recovered_after_streak`。

### 5.1 默认阈值与异常样本量审计

本地 42 个有效场景 `collection_output` 文件（排除 `noScenarios`）共有 7241 条
`status=success` route、914466 个逐帧标注。`build_dataset.py` 默认按 route 划出 10%
validation，因此远端数据链完整时训练规模约 82.3 万帧；RGB/meta/XML 二次过滤后的
精确数字必须读取构建产物 `checkpoints/sft_v5_data/summary.json`，不能把 82.3 万当成
最终固定值。

其中原始 GT EVENT 分布为 RE 772286 帧（84.45%）、UE 142180 帧（15.55%）。GT UE
表示驾驶场景真实异常；下文的 memory anomaly 表示人为喂入错误/UNKNOWN 历史假设，
两者不能相加成一个“总异常率”。UE 的 15.55% 类别不均衡也是必须同时报告
precision、recall、F1、FP、FN，而不能只看 accuracy 的原因。

增强在 rollout 时在线发生，不会把 index 物理复制成更多行。为避免把条件注入概率误当成
最终训练分布，当前默认值先按 3000 条、每条 126 帧的恒定 GT 序列做了状态机模拟。
下面数字按约 82.3 万训练帧线性换算，只用于容量规划；真实 RS/EVENT 变化、route 长度和
学生能力都会改变结果，最终必须以 TensorBoard 为准。

- “理想当帧纠偏”模拟中，Q1 触发率约 30.5%，即约 25.1 万个 RS 训练帧；Q1 真正
  看到的 `aligned / omission / contradiction` 约为 `59.7% / 24.2% / 16.1%`，约
  `15.0 / 6.1 / 4.1` 万帧。Q1 异常 memory（后两类）约占 Q1 的 40.4%，不是简单的
  `5%+7%`，因为异常会额外触发 RS_SLOW。
- 同一理想模拟中 Q2 gate 约 100%；Q2 实际关系约为
  `59.6% / 23.0% / 17.4%`，约 `49.0 / 18.9 / 14.4` 万帧。配置中的 EVENT
  `68/12/20` 是 eligible frame 的条件分布；额外 omission 来自 RS 变化时对条件
  EVENT 的自然失效，因此最终仍接近目标 `60/23/17`。
- “学生完全复制输入 memory，直到 patience/review 延迟 GT 兜底”的压力测试中，Q1
  触发率约 55.5%（约 45.7 万帧），关系约 `35.2/39.4/25.4`；Q2 gate 约 64.0%
  （约 52.7 万帧），Q2 关系约 `38.6/43.5/17.9`。这不是期望训练终态，而是验证
  delayed repair 不会让 EVENT 永久饿死的保守压力边界。

训练必须同时观察关系、复制和门控，而不是只看单一 anomaly rate：

- 健康目标带可接受波动：Q1 relation 大致落在 `50-70 / 18-30 / 12-22`，Q2 大致落在
  `50-70 / 18-30 / 12-23`；Q1 trigger 通常约 28-35%，成熟模型的 Q2 trigger 应逐步
  高于 80%。
- 经过 warmup 后若 Q1 aligned 长期低于 45% 或 Q2 trigger 长期低于 70%，优先把 RS
  wrong/UNKNOWN 各下调 1-2 个百分点，或缩短 RS patience；不要提高 EVENT 噪声。
- 若 Q1/Q2 aligned 长期高于 80%、wrong-copy rate 仍高，说明 memory 仍过于可靠，可把
  对应 contradiction 或 omission 小幅上调 2-3 个百分点。一次只调一个维度，并继续观察
  `memory/q{1,2}_relation_*`、`memory/q{1,2}_*_age_frames_mean`、
  `train/rs_slow_trigger_rate`、`train/q2_skip_due_rs_rate` 与 recovery 指标。

实现时要显式记录：

```json
{
  "frame_id": 17,
  "q1_triggered": true,
  "rs_slow_reason": "recovery",
  "q1_rs_correct": false,
  "rs_gate_correct": false,
  "q2_triggered": false,
  "rs_error_streak": 3,
  "event_error_streak": 1,
  "rs_repair_pending": false,
  "event_repair_pending": false,
  "memory_rs_injected_wrong": true,
  "memory_rs_input_age_frames": 0,
  "memory_event_input_age_frames": 7,
  "rs_slow_interval_draw": 3,
  "memory_before": "...",
  "memory_after": "..."
}
```

---

## 6. 两问 Prompt 协议

所有 prompt 必须英文。当前协议版本是 `sft_v5_compact_prompt_v1`。System prompt 只放
所有问题共享的证据原则；Q1/Q2 的候选、任务说明和输出格式不得再在 system 中重复：

```text
You are an autonomous-driving visual reasoner. Read stitched RGB frames from
oldest to newest. Use visible road geometry, lanes, controls, and relevant actors
to choose only from the provided options. Memory is an unverified prior: check it
against the latest RGB and override conflicts; age is duration, not confidence.
State uncertainty when evidence is weak. Never mention references, hidden labels,
datasets, or scenario names.
```

Compact 合同的代表性预算（R1/RE、Q2 二选一）由 `test_memory_update.py` 固定：system
不超过 70 words，Q1 student user 不超过 160 words，Q2 student user 不超过 175 words，
代表性 teacher user 均不超过 180 words。
候选数量增加时 Q2 可相应增长，但不得重复 system 原则或展开无关标签。adapter、eval 和
probe summary 必须写 `prompt_contract_version=sft_v5_compact_prompt_v1`；旧 prompt
训练的 adapter 若要严格横向比较，应重新训练或至少明确标注协议不一致。

### 6.1 Q1 student prompt

输入：

- 4 张 stitched RGB history。
- 当前 road-only `MEMORY`，只包含 `PREVIOUS_RS_HYPOTHESIS`、可靠性声明和
  `EGO_TO_GOAL_XY`，不包含 `PREVIOUS_EVENT_HYPOTHESIS`。
  `EGO_TO_GOAL_XY` 必须来自当前帧 meta `next_target_points[-1]` 转 ego frame，
  和 v3/v4/LeadMoT final goal 同源。
- `RS_CHOICES` A-E。

Prompt 模板：

```text
{memory_text}

[RS_CHOICES]
A. Ordinary same-direction road; lane keeping/following or same-direction lane changes dominate.
B. Narrow bidirectional/shared road; oncoming traffic or opposing-lane borrowing dominates.
C. Highway/ramp/merge/split/exit; speed matching, gaps, merging or diverging dominates.
D. Signalized intersection; working traffic lights control right of way.
E. Unsignalized/priority intersection; stop/yield, geometry, cross traffic or safe gaps control right of way.
[/RS_CHOICES]

[QUESTION_1]
From the latest RGB, choose one RS_CHOICES option using road/lane geometry,
controls, goal direction and relevant actors. Verify the untrusted memory;
override conflicts.
Return exactly:
Scene Description: <current visible scene; one sentence>
Critical Object Description: <key actor/control/road cue and relevance; one sentence>
Reasoning on Intent: <why one RS option fits; one sentence>
RS: <option letter A-E>
[/QUESTION_1]
```

RS_SLOW parser：

- `RS:` 读取第一个 `A-E`。
- `RS: R4` 等非法值仍不更新 memory，但答案值的第一个生成 token 会进入高权重
  teacher-KL，让 privileged teacher 直接推动合法选项字母；整行缺失时不猜位置。

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
[/REFERENCE]
```

Teacher target 文本仍清洗成学生视角：

```text
Scene Description: The RGB history shows weather, lane markings, traffic controls, surrounding motion, and a signalized intersection layout.
Critical Object Description: The relevant lane boundary, traffic control, or map cue is ...
Reasoning on Intent: The current road geometry supports the selected RS because ...
RS: D
```

禁止让 target 中出现 `ANSWER_`、`GROUND_TRUTH`、`REFERENCE`、`R1` 等私有字段名。

### 6.3 Q2 student prompt

只有当前 RS gate 正确时才进入 EVENT_FAST。若本帧运行了 RS_SLOW，EVENT_FAST 作为
第二轮 user turn 精确续接本帧 Q1 KV；若本帧复用稳定 RS，则没有 Q1 assistant turn，
EVENT_FAST 必须用当前 RGB + memory fresh prefill，绝不能复用上一个慢帧的分析。
RS 错误时跳过本帧 EVENT_FAST，并进入逐帧 RS recovery。

Q2 的 prompt 前缀使用 road + event memory：`PREVIOUS_RS_HYPOTHESIS` 和
`PREVIOUS_EVENT_HYPOTHESIS`
都只写自然语言描述，仍然不写 A-E / RE / U-E* 这类局部选项或标签代码。

```text
[EVENT_FAMILY] [RE | REGULAR] regular/normal; [UE | UNUSUAL] unusual/abnormal. [/EVENT_FAMILY]
[EVENT_CHOICES under RS={chosen_rs_option}]
B. [UE | UNUSUAL] A pedestrian, cyclist or vulnerable road user enters ego's path.
A. [RE | REGULAR] Regular signalized-intersection behavior: obey the current traffic light.
C. [UE | UNUSUAL] A vehicle violates the expected intersection rule and conflicts with ego.
[/EVENT_CHOICES]

[QUESTION_2]
From the latest RGB, choose one EVENT_CHOICES option. RE/UE is already marked;
do not add a separate normal/abnormal decision. Verify the untrusted memory and
choose only a listed option.
Return exactly:
Scene Description: <latest visible scene; one sentence>
Critical Object Description: <key event actor/cue, or none; one sentence>
Reasoning on Intent: <why one EVENT option fits; one sentence>
EVENT: <option letter>
[/QUESTION_2]
```

Q2 parser 只接受当前 `event_option_map` 中存在的选项字母。`EVENT: RE` / `EVENT: U-E*`
仍按 invalid 进入 error streak 并保持原 EVENT memory；训练 loss 只额外选择冒号后答案
值的第一个生成 token 做高权重格式纠偏，不会把语义标签当成合法 memory。

### 6.4 Q2 teacher prompt

Teacher receives:

- current RS gate（本帧 RS_SLOW 预测或稳定 RS memory）
- raw event set
- resolved single-label target after applying §4.3
- previous event memory

慢帧 teacher 的 EVENT_FAST 精确续接 teacher 自己的 Q1 KV；快帧没有 Q1 turn，
teacher 与 student 都从本帧 RGB 做 fresh prefill。Teacher 不接收独立
normal/abnormal 答案；`ANSWER_EVENT` 选中 RE 即 normal，选中 UE 即 abnormal。

Teacher target examples:

RE:

```text
Scene Description: Continue from the current signalized intersection decision.
Critical Object Description: No pedestrian, vehicle cut-in, obstacle, or blocked
intersection space interrupts the ego path.
Reasoning on Intent: The vehicle should keep the regular traffic-light intersection
behavior under the current signalized intersection structure.
EVENT: A
```

UE:

```text
Scene Description: Continue from the current road-structure decision.
Critical Object Description: A vulnerable road user is crossing laterally into
the ego vehicle's intended path.
Reasoning on Intent: The interruption is not merely normal lane keeping or signal
compliance. This matches the event option about a pedestrian or cyclist crossing
the ego path.
EVENT: A
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
- `RS` option letter 单 token: `1.2`
- 非法但存在的 `RS:` 值：只取答案起始 token，权重 `1.2`；parser 仍判 invalid
- formatting tokens / prompt tokens: `0`

Q2:

- structured CoT lines (`Scene Description / Critical Object Description / Reasoning on Intent`): `0.2`
- `EVENT` option letter 单 token: `1.2`
- 非法但存在的 `EVENT:` 值：只取答案起始 token，权重 `1.2`；parser 仍判 invalid
- formatting tokens / prompt tokens: `0`

默认总 loss：

```python
loss = q1_analysis + q1_rs + q2_analysis + q2_event
```

TensorBoard 会同时记录总 loss 和未加权 KL 分项：

- `train/loss_frame`
- `train/loss/q1_analysis`
- `train/loss/q1_rs`
- `train/loss/q2_analysis`
- `train/loss/q2_event`

Q1 分项按实际触发 RS_SLOW 的 frame 平均；Q2 分项按实际进入
EVENT_FAST 的 frame 平均。

离散 span 只覆盖 option letter，不能再覆盖 `A - long description`。否则 teacher/student
在错误 option 已经进入自回归前缀后，会花大量高权重 token 学习“如何把错误答案解释圆”，
稀释真正把 A 改成 D 的首 token 梯度。

若 Q1 RS 错误：

- Q1 loss 正常计算。
- Q2 不触发，Q2 loss = 0。
- 错误 RS 写回后续帧，按 §5 patience/review interval 延迟修复。

若 RS gate 正确，EVENT_FAST 直接进入逐帧 `allowed_events` 折叠候选池：

- Teacher resolver 按 GT 在同一道混合选择题中选 RE 或 UE，不再依赖 Q1 的二分结果。
- EVENT memory 可能错误或过期，模型必须根据本帧 RGB 纠偏，不能直接复制。
- 只有当 GT event 不在逐帧 allowed events 中时，才记录 `q2_candidate_mismatch`；
  这通常意味着标定输出或旧 fallback 候选表需要复查。

### 7.3 生成与 teacher forward

每个问答 step：

1. Student adapter enabled，自由生成 token，保存未裁剪 token ids、decoded text
   和对应 KV state。
2. Parser 在慢帧读 RS，在每个 RS gate 正确的帧读 EVENT。
3. Teacher `disable_adapter()`，使用 privileged prompt，在同一批 student token 上 forward，
   得到 teacher logits 后立即按 target spans 裁剪。
4. Student 在同一批 token 上 forward，logits 也立即裁剪到 target spans。
5. 对裁剪后的 teacher/student span logits 做 weighted forward-KL；完整 vocab logits
   不跨 loss 计算长期保留。

Qwen3-VL 增量 decode 必须复用 `qwen3vl_local/mrope_utils.py` 的
`qwen3vl_incremental_forward`，禁止走 PEFT wrapper 的 `generate` /
`prepare_inputs_for_generation`。

训练阶段不能做基于输出字段的结构早停：即使 student 已经生成出 `RS:` 或
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
   - 某个样本预测 EOS 时直接从 active batch 中移除；纯 batched rollout 只返回文本和
   精确 token ids，不物化/返回逐样本 final KV，避免 8 份长 cache 同时常驻；
   - 后续 teacher 与 KL 默认走 chunk 级 batched scoring，但 rollout batch 与 KL
     autograd 微批独立；
   - 若 processor/cache 兼容失败，自动 fallback 到单帧旧路径；CUDA OOM 直接中止，
     不在 OOM 后继续降级运行。
2. **阶段 2：batched Q2 student rollout**（已落地）
   - 只对 Q1 RS 正确的子集做 batch；
   - Q2 也允许 mixed-length padded rollout，但必须先 prefill 当帧 Q1 图文 prompt，
     再追加 student 原始 `q1_ids` 和 Q2 user turn；rollout 本身也禁止
     `q1_ids -> q1_text -> full-dialog tokenizer` 回环；
   - padded Q2 KV 同样只用于 no-grad 采样文本/token；
   - parallel KL 使用与 rollout 完全相同的精确续接语义，保证采样与 scoring 看见同一份
     Q1 token 上下文。
3. **阶段 3：batched teacher/KL forward**（已落地为 `--parallel-kl` 默认开启）
   - 对不同长度 rollout ids 做 padding；
   - 按每个样本自己的 span positions 取 logits；
   - padding token 不进入 loss，不污染 KV。
   - teacher/student prompt state、rollout token scoring 和 span KL 在同一 rollout
     chunk 内按 `PARALLEL_KL_MICROBATCH_SIZE` 拆分；默认 8 路 rollout 对应
     `2+2+2+2` KL 微批，每个微批立即 backward，不同时保留多份 autograd graph。
   - 显存峰值主要来自 `KL microbatch x context length` 的 attention activation 与
     student logits。1024 token 上限保持不变；若 2 路 KL forward OOM，只把当前微批
     二分为两个单帧，不降低输出长度、不重新 rollout。
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
KL 路径 logits 的 max/mean abs diff，以及 direct Q1 KV 续接 Q2 与 grouped exact-KV Q2
rollout 的完整 token/text；训练前默认不传
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
- 每个 KL 微批或单帧 loss 在 backward 后立即释放计算图引用；optimizer step 使用
  `zero_grad(set_to_none=True)`。训练退出统一销毁 process group 并执行一次 GC/CUDA
  cache 清理，但正常 step 不调用 `empty_cache()`，避免吞吐下降。
- TensorBoard 同时记录逐帧 `progress/cuda_{allocated,reserved,max_allocated,max_reserved}_gb`
  和 optimizer-step 级 `memory/{allocated,reserved,max_allocated,max_reserved}_gb`。
  `allocated` 用于判断活跃 tensor/graph 是否持续累积，`reserved` 只是 allocator 的
  历史高水位，不能仅凭 `nvidia-smi` 或 reserved 增长判定显存泄漏。

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

Checkpoint / probe 策略：

- 用户实测四卡当前吞吐约 80 optimizer steps/day，正式 launcher 默认
  `SAVE_STEPS=40`，约半天保存 `checkpoint-40/80/...`；正常结束始终保存 `final/`。
- 每个 run 的 step 0 自动在 `probes/base/` 保存固定、可复现的完整 validation route。
  默认 `random` 用固定 seed 选择 1 条 route ID，从首帧测试到末帧；每个 checkpoint/final
  保存后生成同一 ID 的 LoRA student + 纯 base teacher probe，便于逐帧纵向对比。
- probe 公开选帧模式只有三种：`random` 随机完整 route ID；`rs_transition` 的同一 RS
  变化前/首帧/后帧；`ue_transition` 必须保留同一 UE span 的全部 UE 帧，再按
  context radius 补进入前和退出后邻帧。UE 不得被 `num_cases` 从中间截断；专项模式
  没有找到真实变化时不得用普通帧 fallback 凑数。
- 自动 probe 必须复用 rank0 当前训练 bundle，不能另起子进程加载第二份 Qwen。base
  student/teacher 使用 `disable_adapter()`，checkpoint student 保持 LoRA 开启；其它
  rank 在 probe 前后 barrier，防止参数变化和 collective 次序错位。
- 每个 probe 都写轻量主 `results.json`，run 级 `probes/comparison.json` 聚合 base、各
  checkpoint 和 final。默认 `--artifact-level review` 按
  `scenarios/<scenario>__<route>/frame_<id>/` 拆分连续帧，每帧只写输入 RGB、
  `input.json`、`output.json`、`memory.json`；`--artifact-level full` 才增加 legacy
  prompt/memory 文件。`compact` 只减少文件数量，不能减少
  审计字段：每帧必须保留 RGB 路径、RS_SLOW 是否触发与触发原因、
  实际 student/teacher messages、标明慢帧 Q1 KV 或快帧 fresh RGB prefill 来源的
  EVENT_FAST user turn、完整 student/base-teacher
  CoT 输出、脚本化 teacher target、
  RS/EVENT 场景真值、memory 与变化检测结果。base 和 LoRA probe 必须使用同一 schema，
  唯一区别是 LoRA probe 的 student 开启 adapter。
  `output.json` 必须直接并列学生/老师 raw output、解析结构、场景真值、teacher target 和
  逐项正确性；`memory.json` 必须并列 Q1/Q2 student memory 转换与只读 reference。
  失败写 `error.txt` 后继续训练。probe 结束必须恢复 train 模式并清理 CUDA
  cache，不能让可视化对象跨训练窗口常驻。
- 慢帧 teacher model 的 EVENT_FAST 只在其自身 Q1 RS 正确时触发，并续接 teacher
  自己的 Q1 KV 和 RS；快帧不存在 Q1，teacher 从本帧 RGB fresh prefill。不得把
  student/GT-forced EVENT prompt 接到 teacher Q1 KV 后。训练用
  `q2_teacher_training_prompt.txt` 与自主 teacher 的 `q2_teacher_model_prompt.txt` 分开保存；
  默认 `q2_teacher_prompt.txt` 必须和 `q2_teacher_output.txt` 一一配对。
- 训练中自动 probe 的默认生成上限是 Q1=256、Q2=192，只控制旁路可视化耗时；
  手工 `probe.py` 与正式 eval 默认 1024/1024，训练 rollout 也仍是 1024/1024，
  不得因为自动 probe 缩短 OPSD 训练输出或正式评估输出。

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
`per_device_batch_size=8`、`qwen_batch_size=8`、`parallel_kl_microbatch_size=2`、
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
parallel_kl_microbatch_size=2
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
- `train/loss/q2_analysis`
- `train/loss/q2_event`
- `train/rs_slow_trigger_rate`
- `train/rs_reuse_fast_rate`
- `train/q2_trigger_rate`
- `train/q1_rs_acc_window`
- `train/q2_event_acc_window`
- `train/q2_invalid_output`
- `memory/repair_pending`
- `memory/rs_wrong_copy_rate` / `memory/rs_recovery_rate`
- `memory/event_wrong_copy_rate` / `memory/event_recovery_rate`
- `memory/{rs,event}_{injected_wrong,injected_unknown,forced_repair}`
- `train/abnormal_{precision,recall,f1}`
- `train/q2_ue_{precision,recall,f1}`
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
traceback 判断阶段：正式默认已使用 `PARALLEL_KL_MICROBATCH_SIZE=2`；若 OOM 位于
`_opsd_loss_batch_states`，代码会继续把当前 KL 微批二分到单帧，保持 8 路 rollout；
只有 OOM 位于 Q1/Q2 grouped student rollout 时，才用 `BATCH_PROFILE=balanced/debug`
降低 rollout batch。若
`qwen/q1_batched_frame_rate` 长期接近 0，则先查
`[warn] q1 batch fallback`，必要时回到 `QWEN_BATCH_SIZE=1`。
若 8 路已经稳定显示 `batched_frames=8`，但 GPU util 仍只有中等水平，先确认启动日志里
`[parallel] PARALLEL_KL=1`，并观察是否出现 `[chunk-train] ... parallel_kl=1`。
`[chunk-train]` 应同时打印 `kl_microbatches=[2, 2, 2, 2]`；偶发
`[warn] parallel KL adaptive split ... [1, 1]` 可继续训练，长期出现则结合峰值决定
是否固定改成 1。

训练阶段不按 `RS:` / `EVENT:` 字段早停；默认仅保留 `MAX_NEW_TOKENS_Q1=1024`
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

### 8.4 训练 padding 与 delayed memory repair

每个 batch 初始化 `memory[B]`：

- 第一个有效 frame 前：RS/EVENT 各自按 `initial_gt_prob` 选择 GT 或 UNKNOWN。
- Padding timestep：memory 不变。
- 稳定且正确的 RS 默认在 3/4/5 个 4Hz frame 中可复现地随机选择下一次
  RS_SLOW 间隔，中间帧复用 RS memory；`UNKNOWN`、memory 与当前 RS 不一致、上次
  RS 预测错误时立即进入逐帧 recovery。
- RS_SLOW 错：增加 `rs_error_streak`，错误 student RS 原样进入下一有效帧，
  下一帧继续慢思考；若错误 RS 与输入 RS 不同，则旧条件 EVENT 同时失效为 UNKNOWN。
  只有当前 RS gate 正确才运行 EVENT_FAST。
- EVENT_FAST 每个 RS gate 正确的帧都重新读取 RGB 并预测 EVENT，不复用上帧
  normal/abnormal 或 EVENT 结果；EVENT 错时增加独立 `event_error_streak`。
- RS hypothesis 改变时先清空旧 RS 语境的 EVENT streak/pending；若同帧 Q2 仍答错，
  从新语境的 streak=1 重新累计，不能继承旧 pending 立即触发 GT repair。
- streak 达到 patience 后只设置 repair pending；RS 默认每 2 帧检查脚本修复，EVENT 默认
  每帧检查。`rs_repair_interval` 仅控制兜底修复，不控制稳定期 `rs_slow_interval`。
- 学生在 pending 真正执行前自行恢复时清空 pending，不再做多余 GT overwrite。
  已执行延迟 GT 修复的帧即使答对，也只记为干预后恢复，不冒充自主纠偏。

以上只属于训练采样协议。`eval.py` / `probe.py` 默认从 RS/EVENT=UNKNOWN 开始，
此后只刷新每帧 `EGO_TO_GOAL_XY`，RS/EVENT 由学生输出推进，不做训练期
repair。RS_SLOW 调度默认为可部署口径：UNKNOWN/非法输出会逐帧重问，合法 RS
变化后多问一帧确认，稳定时按同一 seed/key 的 3/4/5 帧随机周期复核；不使用 GT
mismatch 触发恢复。
`--rs-schedule-policy oracle` 和 `--initial-memory ground_truth` 只用于复现旧报告。

这里有一个必须明示的离线评分边界：用户要求“RS 答错则跳过 EVENT”，而
离线 eval/probe 只能通过 GT 知道一个合法 R1-R5 是否真错。因此 RS 调度默认
不看 GT，但 EVENT gate 仍是 `offline_ground_truth_rs_correctness`。摘要会写
`fully_deployable_end_to_end=false`；在上线前仍需增加 RS 置信度/几何一致性 verifier。
random 默认测试整条 ID；RS/UE 边界前后默认观察 8 帧，并报告学生首次自行
对齐 reference 的延迟。

如果某条 route sequence 在中间出现缺 RGB / 缺 weather / 缺 label：

- build 阶段应过滤或截断。
- train 阶段遇到仍缺失的 frame，跳过该 frame 并记录 `runtime_frame_skip`，
  不让 DDP rank 崩掉。

---

## 9. Eval 指标

`metrics.py` 是 `eval.py` 与 `probe.py` 的唯一指标口径。大样本评估按帧流式更新计数器，
不会把全量 prompt/output 留在内存；只有显式传 `--output-jsonl` 时才把完整逐帧证据落盘。
可单独传 `--transition-jsonl` 只落盘真实/预测 RS 变化、UE 进入/退出和 FP/FN，
  不保存大段 prompt。小样本的变化指标在 `results.json.summary`，自主 memory 恢复延迟
  在 `results.json.memory_recovery_report`；full 模式另写 `transition_report.json`。
分母没有样本的指标写 `null`，不能用 0 假装模型失败。
正式 eval 的 Q1/Q2 默认安全上限均为 1024，与训练 rollout 对齐；自动小样本 probe 的
256/192 只控制 checkpoint 旁路耗时，不用于报告正式大样本指标。

核心 frame-level 指标：

> `abnormal_*` / `route_abnormal_*` 是旧报告 schema 的兼容命名，数值全部由本帧最终
> EVENT 的 RE/UE family 派生；它们不代表模型仍回答独立 `ABNORMAL` 字段。

- `rs_acc`、`rs_transition_acc`、`rs_stable_acc`：全帧、RS 变化首帧、RS 稳定帧准确率，
  都是越高越好。
- `rs_change_detection_precision/recall/f1`：相邻帧预测 RS 是否在真实变化首帧切换，
  越高越好；`rs_change_false_positive_rate` 表示 RS 稳定时的误切换，越低越好。
- `ue_entry_detection_*` / `ue_exit_detection_*`：分别评估 RE->UE 进入帧和 UE->RE
  退出帧；precision/recall/F1 越高越好，false-positive-rate 越低越好。
- `abnormal_acc`：EVENT_FAST 选项的 RE/UE family 严格准确率，非法格式计错，越高越好。
- `abnormal_precision/recall/f1`：EVENT_FAST 对 UE 的查准率、召回率和 F1，越高越好。
- `abnormal_false_positive_rate`：真实 RE 被 EVENT_FAST 错选为 UE 的比例，越低越好。
- `abnormal_false_negative_rate`：真实 UE 未被 EVENT_FAST 正确选中的比例，非法输出也计入，
  越低越好。
- `abnormal_boundary_acc`：RE/UE 状态发生切换的首帧 EVENT_FAST 准确率，越高越好。
- `event_acc_when_rs_correct`：Q1 RS 正确并进入 Q2 后的具体 EVENT 准确率，支持既定的
  动态双标签容错，越高越好。
- `q2_ue_precision/recall/f1`：进入 Q2 后，把具体 EVENT 折为 UE/RE 的检测指标，越高越好。
- `q2_false_positive_rate` / `q2_false_negative_rate`：Q2 的 UE 假阳性/假阴性率，越低越好。
- `ue_acc` / `re_acc`：进入 Q2 后真实 UE/RE 子集的具体标签准确率，越高越好。
- `event_end_to_end_acc`、`ue_end_to_end_recall`：把 Q1 RS 门控失败也计错的端到端 EVENT
  准确率与 UE 召回率，越高越好。
- `event_end_to_end_false_positive_rate`：所有真实 RE 中最终被错误输出为 UE 的比例，
  越低越好。
- `q2_trigger_rate` / `q2_skip_due_rs_rate` 分别表示 RS 正确进入 Q2、RS 错误跳过 Q2；
  两者和非法输出率、原始 `tp/fp/tn/fn/invalid` 只用于解释门控与格式问题，不能脱离
  准确率单独判断好坏。

Route-level 指标：

- `route_rs_all_correct_ratio`：整条 route 每帧 RS 都正确的 route 比例，越高越好。
- `route_abnormal_f1_macro` / `route_ue_f1_macro`：先逐 route 算 F1 再等权平均，越高越好。
- `rs_wrong_memory_copy_rate` / `event_wrong_memory_copy_rate`：输入已知错误 memory 时仍
  原样复制错误标签的比例，越低越好；这是本轮定位 shortcut 的主指标。
- `rs_wrong_or_unknown_memory_recovery_rate` /
  `event_wrong_or_unknown_memory_recovery_rate`：错误/UNKNOWN 输入上的单帧自主恢复率，越高越好。
- `mean_resets_per_100_frames`：eval/probe 实际 GT reset 次数，closed-loop 应为 0；训练期
  delayed repair 的次数与 pending/streak 另走 `memory/*` TensorBoard。
- `mean_valid_frames_per_route`：评估规模诊断，不单独判断好坏。

Probe dump：

```text
probe*/
  results.json
  scenarios/
    <scenario>__<route_id>/
      frame_<frame_id>/
        input_rgb_00.jpg
        input_rgb_01.jpg
        input_rgb_02.jpg
        input_rgb_03.jpg
        input.json
        output.json
        memory.json
```

默认 `review` 使用以上结构。`input.json` 保存实际 messages 与 Q2 KV 续接；`output.json`
保存 student/teacher raw 与 parsed、双标签结构化真值、文本 target 和正确性；
`memory.json` 保存 Q1/Q2 输入输出、下一帧 student state 和 comparison-only reference。

只有 `--artifact-level full` 才在以上 review 文件之外增加：

```text
probe*/
  selection_plan.json
  manifest.json
  summary.json
  scenarios/
    <scenario>__<route_id>/
      timeline.json
      timeline.png
      frame_<frame_id>/
        input_rgb_00.jpg
        input_rgb_01.jpg
        input_rgb_02.jpg
        input_rgb_03.jpg
        case_record.json
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

可执行的 probe / eval 命令仅在 `SFT_V5_RUN.md` 保留快速入口；完整的输入输出
产物、目录结构和人工检查项统一记录在 `SFT_V5_VISUALIZATION_RECORD.md`。v5 检查
明确分成五类：训练中自动版本对比、训练前
base 能力、训练前 grouped/parallel 等价性、训练后 adapter 可视化、静态合同快检。

- 训练中自动版本对比：训练前 `base`、每 40 step `checkpoint-*`、训练结束 `final`；
  默认固定使用同一条 seed 可复现的 `random` validation route ID，并在每个
  `results.json` 记录选帧、输出和指标，在 `probes/comparison.json` 汇总版本对比。

- 训练前 base Qwen OPSD 能力体检：`--with-model --with-teacher-model`，不传
  `--adapter-dir`，不加载任何 LoRA，让默认 Qwen 分别跑 student prompt 和
  privileged teacher prompt，用 `q*_student_output.txt` / `q*_teacher_output.txt`
  判断普通 Qwen 的基础能力与 prompt 合同是否足够支撑 OPSD。
  teacher model 的自由生成输出必须和 student 一样直接从 `Scene Description:` 开始，
  并按 `Scene Description / Critical Object Description / Reasoning on Intent /
  RS 或 EVENT` 写完整结构；如果复读 MEMORY、choices 或 REFERENCE，说明旧 demo 需要
  重跑或 prompt 合同还要继续收紧。
- 训练后 adapter 学生可视化：`--with-model --adapter-dir ...`，只加载训练后的
  student adapter，检查真实状态机下的 Q1/Q2 输出、memory-copy 与自主恢复行为。
- 静态 prompt / target 快检：不加载模型，只写 RGB、student prompt、teacher prompt、
  脚本化 teacher target、label、memory、flags 和 timeline。

`--with-teacher` 只保留为 v3 兼容标志；真正生成 teacher 模型文本必须显式使用
`--with-teacher-model`。

### 9.2 代码注释维护要求

`AutoMoT/qwen3vl_local/sft_v5/` 下代码采用中文注释维护：

- Python 模块 docstring 需说明该文件的用法、主入口和与相邻模块的分工。
- 所有 class/function，包括 CLI 入口、嵌套 helper 和 `__len__` / `__iter__` 等魔术方法，
  都要有中文 docstring，说明输入输出、状态机职责或调用语义。
- 关键逻辑块必须解释设计原因，而不只是复述代码行为；当前已覆盖
  `allowed_events` 优先级、`R-E* -> RE` 折叠、RS/EVENT 双标签单标签化、
  Q1 RS 错误截断、OPSD teacher/student logits 对齐、DDP local/global padding、
  训练前纯 base Qwen 体检不加载 LoRA、probe flags 审计字段和测试回归意图。
- 避免“给变量赋值”这类逐行复述；对 padding、KV 续接、loss 分母、DDP collective、
  显存生命周期等非显然合同，在代码块前解释“为什么”。
- 后续修改标签协议、prompt、memory、loss、probe 或 DDP 训练逻辑时，需要同步更新
  相邻代码注释和相关文档。运行参数改动写入 `SFT_V5_RUN.md`，设计合同写入本文，
  完整可视化产物说明写入 `SFT_V5_VISUALIZATION_RECORD.md`，避免三份文档重复膨胀。

### 9.3 本轮中文注释覆盖与代码阅读路线

本轮对 `sft_v5/` 做的是**纯注释与文档增强**：补充模块级中文说明、class/function
docstring，以及函数内部关键逻辑块的设计原因。它不修改 RS/EVENT 标签协议、memory
状态机、采样概率、repair patience/interval、loss 权重、batch profile、token 上限、
optimizer 窗口或评估口径；运行结果若有变化，应先按代码改动或运行环境排查，不能把
变化归因于这次注释整理。

建议按以下顺序读代码；这样先建立标签与状态语义，再进入最长的训练主循环：

| 顺序 | 文件 / 核心入口 | 函数内部应重点理解的逻辑 |
|---:|---|---|
| 1 | `labels.py`：`resolve_rs_target`、`resolve_event_target`、`q2_raw_candidates_for_frame`、`stable_event_option_map` | 如何从原始多标签证据解析单个 RS/EVENT 监督；为什么逐帧 `allowed_events` 优先、`R-E*` 只折成一个 RE，以及选项字母如何可复现随机化。 |
| 2 | `prompts.py`：`Memory`、`prepare_training_memory`、`should_run_rs_slow`、`observe_training_memory` | Q1/Q2 为什么读取不同 memory；错误/UNKNOWN 扰动如何只改变当前输入；student 输出如何写回闭环状态；pending repair 如何在学生自主恢复之后才决定是否执行。 |
| 3 | `build_dataset.py`：`build_dataset` → `_build_route_row` → `_build_frame_row` | route 过滤、逐帧 RGB/meta/XML 对齐、`next_target_points[-1]` 到 ego 坐标、weather fallback、历史帧 left-pad，以及 frame/route 为何会被跳过。 |
| 4 | `train.py` 数据入口：`RouteSequenceDataset`、`LengthBalancedDistributedSampler`、`collate_route_sequences`、`pad_batch_to_global_length` | 为什么 worker 只做 local padding、distributed collective 必须回到主训练进程、padding frame 为什么不能读图或进入 Qwen。 |
| 5 | `train.py` Qwen/KV：`_kv_start_state_batch_padded`、`_student_generate_kv_batch`、`_run_q1_rollout_grouped`、`_run_q2_rollout_grouped` | no-grad rollout 与有梯度 scoring 的边界；last-valid logits、padding token 排除、EOS active batch 收缩、`rope_deltas` 方向兼容，以及 Q2 为什么必须用原始 `q1_ids` 续接 KV。 |
| 6 | `train.py` 单帧语义基准：`_run_frame`、`_run_event_only_frame` | 慢帧如何串行执行 Q1→RS gate→Q2，RS 错误为什么只跳过当帧 EVENT；快帧为什么不伪造 Q1，而是对当前 RGB fresh prefill 后只训练 EVENT_FAST。 |
| 7 | `train.py` KL/优化：`_opsd_loss_batch_states`、`_run_parallel_kl_microbatches`、`_sync_trainable_grads_by_global_frames`、`complete_optimizer_step` | teacher/student 如何在同一批 student token 和 span 上对齐；为什么只允许 forward OOM 在 backward 前二分；loss 分母如何按实际 global frame 校正；为什么 LoRA 梯度要按 device/dtype 分桶 all-reduce。 |
| 8 | `metrics.py` → `eval.py` → `probe.py` | closed-loop 指标怎样流式累计；RS gate 失败如何进入端到端 EVENT 指标；transition、FP/FN、memory recovery 如何从逐帧记录构造；probe 如何选择连续窗口并写审计产物。 |
| 9 | `test_*.py`、`check_loss_mask.py`、`train.sh` | 每个回归测试保护哪项合同，以及 launcher 怎样把环境变量转换成 `train.py` CLI；shell 注释重点解释模式分支、GPU 选址和默认 profile，不重复 Python 算法。 |

训练主路径可以压缩成下面这条调用链：

```text
main
  -> Dataset / sampler / local collate / global T padding
  -> 每个 global timestep 选择 RS_SLOW 帧与 EVENT_FAST-only 帧
  -> grouped no-grad student rollout
  -> RS gate 与动态 EVENT 候选解析
  -> parallel-KL 微批重建精确 student/teacher state
  -> 每个微批立即 backward
  -> 更新 student memory 与 delayed-repair 审计状态
  -> 完整 timestep 边界判断 streaming optimizer window
  -> 分桶同步 LoRA 梯度、optimizer step、checkpoint/probe
```

阅读函数内部注释时要区分三类语句：

- **correctness 合同**：padding、M-RoPE、精确 `q1_ids` 续接、loss 分母、collective
  次序和显存释放；修改这些代码前必须先跑对应测试。
- **状态机语义**：RS_SLOW 触发、RS gate、EVENT_FAST、memory 写回、delayed repair；
  修改后必须同时核对 train/eval/probe 的 closed-loop 行为。
- **审计与可视化**：日志、TensorBoard、probe schema 和 transition report；它们不参与
  loss，但不能随意删减，否则无法验证模型是否仍在复制错误 memory。

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
  - prompt contract version
  - RS/EVENT label version
  - RE folding policy
  - DDP world size
  - LoRA vision scope
  - max_new_tokens
  - loss weights
  - 完整 memory curriculum 概率、patience/review/repair mode
  - `event_conditioned_on_rs=true`、`rs_change_invalidates_event=true`、
    `rs_change_resets_event_error_context=true`
  - streaming window、gradient sync 与 checkpoint probe 配置

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
   - 先实现共用流式指标和大样本自由生成评估。
   - 再实现 UE/RS 边界定向小样本与 teacher 对照完整 dump。
6. 测试：
   - `python qwen3vl_local/sft_v5/test_memory_update.py`
   - `python qwen3vl_local/sft_v5/test_dataset_contract.py`
   - `python qwen3vl_local/sft_v5/test_streaming_optimizer.py`
   - `python qwen3vl_local/sft_v5/test_parallel_kl_microbatch.py`
   - `python qwen3vl_local/sft_v5/test_checkpoint_probe.py`
   - `python qwen3vl_local/sft_v5/test_probe_selection_and_metrics.py`
   - `python qwen3vl_local/sft_v5/check_loss_mask.py`
   - `python -m py_compile qwen3vl_local/sft_v5/*.py`

---

## 12. 已拍板规则

- 训练时 RS_SLOW 在稳定正确期默认按 3/4/5 帧可复现随机间隔运行；快帧复用 RS，
  但 EVENT_FAST
  仍必须重新分析当前 RGB。RS 错误只结束当前帧 EVENT_FAST，错误 memory 继续
  进入后续帧，下一帧恢复逐帧 RS 慢思考；超过 patience 且到 review 帧才兜底修复。
  EVENT 是 `EVENT | RS`：RS hypothesis 变化时旧 EVENT 立即变为 UNKNOWN/age=0，
  同一 RS 的周期确认则保留 EVENT 与 age；新 EVENT 只能由 gate 通过后的 Q2 重建。
  EVENT 使用更短 patience 和逐帧 review。正式
  eval/probe 不做任何 GT 纠错，而是继续学生闭环，以测量后续自主恢复能力。
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
  Reasoning on Intent / RS`；Q2 使用同样的三段分析后输出 `EVENT`。天气、道路、
  车道线、交通灯和周围运动
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
