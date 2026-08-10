# ROAD / EVENT RGB 审计归档（2026-07）

本文合并 2026-07 期间散落的 ROAD_STRUCTURE / EVENT RGB 审计记录。它不是新的规则源，而是历史审计结论索引；当前规则仍以 `collector.py`、`ROAD_EVENT_CLASSIFICATION_PLAN.md`、`ROAD_EVENT_CANDIDATE_MAPPING.md` 和 `ROAD_STRUCTURE_PER_SCENARIO_RULES.md` 为准。

原来的逐次实验 MD 已合并到这里，避免 `AutoMoT/keyframe_filter/` 顶层堆太多一次性记录。临时 RGB sheet、CSV、JSON、candidate anomalies 和 route/town/global summary 仍属于本地证据产物，不入库、不 push；如果需要复查，应按本文的命令或对应 `collection_output/` 目录重新定位证据。

## 1. 文档归并关系

| 旧记录 | 归并后的用途 |
|---|---|
| `RGB_R4_R5_AUDIT_SUMMARY.md` | R4/R5 RGB 全量审计与规则回灌摘要 |
| `RS_EVENT_RGB_REVIEW_SUMMARY.md` | town5 逐帧 RS/EVENT RGB 抽样复核摘要 |
| `ROAD_STRUCTURE_FULL_RGB_AUDIT_20260706.md` | 43 场景全量 RGB blind + RS 输出审计摘要 |
| `ROAD_EVENT_RS_SYNC_AUDIT_20260706.md` | RS 候选收紧后 EVENT 同步回归摘要 |
| `ROAD_EVENT_NO_R6_RGB_AUDIT_20260706.md` | 删除旧 R6 后的停车/开门/障碍场景复核摘要 |
| `ROAD_EVENT_INTERRUPTED_OVERLAY_AUDIT_20260706.md` | R4/R5 接管突发事件时的 interrupted overlay 审计摘要 |
| `R2_ROUTE_RGB_REVIEW_SUMMARY_20260708.md` | R2 route-level RGB 复核与 runtime gate 摘要 |

## 2. R4/R5 RGB 全量审计（2026-07-04）

目标：逐帧读取 LEAD stitched RGB，判断各 scenario / route 是否有稳定信号灯路口 R4、无灯/STOP/yield/priority/T 路口 R5，或混合控制源。

关键口径：

- R4 需要可见灯控硬件/灯色，并由 meta `traffic_light_state/light_hazard` 或 bbox `traffic_light` 等同源证据确认。
- R5 需要 junction、stopline、turn、crossing、STOP/yield/priority 等证据；不能把高速匝道、导流线、merge 线、停车线直接当无灯路口。
- `CrossJunctionDefectTrafficLight` 可见信号灯时仍是 R4；灯故障/失效进入 U-E7，不用 R5 表达。

规则回灌摘要：

- 保持 no-R4 的稳定高速/快速路背景：`EnterActorFlow`、`EnterActorFlowV2`、`HighwayExit`、`MergerIntoSlowTrafficV2`。
- `HighwayCutIn`、`MergerIntoSlowTraffic` 恢复少量 R4 子集，但主体仍是 R3；R5 弱证据不进入候选。
- `InterurbanActorFlow`、`InterurbanAdvancedActorFlow` 删除 R4，保留 R1/R3/R5 或 R1/R5。
- `NonSignalizedJunctionRightTurn`、`OppositeVehicleTakingPriority`、`PriorityAtJunction`、`T_Junction` 允许 R1/R4/R5 混合，由逐帧控制源决定。
- R5 不能从“没看见红绿灯”推出，必须有无灯/STOP/yield/priority/T-junction 证据。

## 3. town5 逐帧 RS/EVENT RGB 抽样复核

命令模板：

```bash
python AutoMoT/keyframe_filter/rs_full_frame_review.py \
  --scenario all \
  --samples-per-town 5 \
  --output-dir /tmp/automot_rs_event_rgb_review_all_town5_current \
  --frames-per-sheet 40 \
  --sheet-cols 4
```

覆盖：43 个 scenario、993 条有效 route、117122 帧。每个 scenario × town 最多 5 个 route，不足则读完全部可用 route。

关键发现：

- 许多 review 不是 RGB 硬错，而是 XML route 投影误差导致 XODR/topology 被降权；因此 `rs_full_frame_review.py` 引入 `review_severity`，区分 `hard_issue`、`boundary_review`、`soft_evidence_note`、`event_boundary`。
- `BlockedIntersection` 发现真实规则漏洞：STOP/无灯阻塞路口不能默认 R4，已改为 `blocked_intersection` 规则族，灯控同源证据给 R4，STOP/yield/无灯控制源给 R5，阻塞本身只进 EVENT。
- `NonSignalizedJunctionLeftTurn*` 发现 bbox 偶发同时报 `traffic_light` 与 `stop_sign` 导致误升 R4；已加入 strict no-R4，并让 STOP/yield 在无有效灯态时压制 bbox/static-signal 的 R4 提升。
- `HazardAtSideLane*` 核心是行人/自行车从侧边进入或横穿 ego path，应使用 U-E4，离开对象后如仍在回正才接 R-E2，不再套静态障碍 U-E2。

## 4. 43 场景全量 RGB blind + RS 输出审计（2026-07-06）

覆盖：

- 输入：`AutoMoT/lead_data`
- route 覆盖：9715 routes / 43 scenarios
- 成功标注：8614 routes
- 跳过：`skipped_abnormal_duration=963`，`data_missing_skip=138`
- RGB 帧：1102886
- 成功标注帧：1062401

RGB blind 自动检测主要用于发现 R4/R5 灯控、STOP、无灯路口线索；它不能可靠地区分 R2/R3。R2/R3 结论必须结合当前规则输出、逐场景 RGB sheet 和人工复核。

R2 专项回退结论也在本轮后确立：非 TwoWays 场景不再根据场景级 XODR sparse scan 批量开放 R2；`LAYOUT_R2_ROUTE_IDS` 默认空。普通非 TwoWays 风险场景如 `Accident`、`ConstructionObstacle`、`ControlLoss`、`BlockedIntersection`、`CrossJunctionDefectTrafficLight`、`CrossingBicycleFlow` 不应再产生 R2。

## 5. RS 候选收紧后的 EVENT 同步回归（2026-07-06）

命令模板：

```bash
python AutoMoT/keyframe_filter/quick_start.py annotate-rs \
  --scenario all \
  --output-dir /tmp/automot_event_rs_sync_all_20260706
```

覆盖 43 个 scenario、8614 条 route、1062401 帧。`primary_event` 越过当前 primary RS allowed events 的违规数为 0。

关键结论：

- `collector.py` 需要最终 EVENT 候选池兜底：每帧 EVENT 必须属于 scenario 精细白名单与当前 primary RS EVENT 候选池的交集，并始终保留当前 RS 的 regular event。
- R3 是高速/匝道/合流道路结构，不等于全程 R-E3；普通跟车 regular fallback 是 R-E1，merge/exit/actor-flow core 或真实目标变道才进入 R-E3/R-E2。
- `*_TwoWays` 候选 RS 删除 R1；有效可行驶通道为对向单车道时默认 R2，灯控/STOP/无灯控制源仍可覆盖为 R4/R5。
- R4/R5 默认删除 U-E2/U-E3，避免路口等待、红灯排队、路口起步被误标为绕障/切入；但 `AccidentTwoWays` 等 R2 core 与 R4/R5 重叠时允许 overlay，U-E2/R-E2 可优先于 R-E4/R-E5。
- `InvadingTurn` 是例外：即使 primary RS 为 R4/R5，只要对向异常侵占自车道仍可保留 U-E5。

## 6. 删除旧 R6 后的 No-R6 审计（2026-07-06）

覆盖旧 R6 高风险场景：

- `ParkingCrossingPedestrian`
- `ParkingExit`
- `ParkingCutIn`
- `StaticCutIn`
- `VehicleOpensDoorTwoWays`
- `ParkedObstacle`
- `ParkedObstacleTwoWays`

全量结果：895 条可标注 route、106471 帧，另有 16 条缺 `meta.pkl` 跳过。runtime 枚举、候选池和输出均无 R6；`primary_road_structure` 越过候选池为 0，`primary_event` 越过当前 RS allowed events 为 0。

最终语义：

- 停车环境、路边停车、开门、停放车辆不再成为独立 ROAD_STRUCTURE。
- 停车/开门/静态障碍等由 EVENT 表达：U-E2/U-E3/U-E4 或恢复 R-E2。
- 当前道路结构仍由 R1/R2/R4/R5 表达：普通道路 R1，双向窄路或有效对向单车道 R2，灯控 R4，无灯/STOP/yield/priority R5。

## 7. Interrupted EVENT overlay 审计（2026-07-06 / 2026-07-07）

问题：R1/R2 视角下的突发事件尚未自然结束，但 primary RS 已进入 R4/R5 路口控制区。旧逻辑会把 `U-E2 -> R-E4/R-E5` 硬切断，丢失障碍动作或 R-E2 恢复段。

现行口径：

```text
primary_road_structure = R4/R5
road_structure_overlay.active = true
road_structure_overlay.base_road_structure = 原 R1/R2/...
road_structure_overlay.intersection_road_structure = 当前 R4/R5
events = {R-E4/R-E5, U-E* 或 R-E2}
primary_event = safety/recovery event
```

触发：

- 前一帧是非 R4/R5 的 `U-E1/U-E2/U-E3/U-E4`；
- 当前帧 primary RS 被 R4/R5 接管；
- 当前帧仍有突发事件证据，或有明确回目标/原车道的 R-E2 恢复证据。

退出：

- 对象距离、hazard 或横向回正证据自然结束；
- 或总 overlay 达到 24 帧；
- 或 R-E2 恢复子阶段达到 12 帧。

2026-07-07 对所有包含 R1 突发事件的高风险场景全量复核：3552 route / 526001 帧，overlay 99 route / 1472 帧，最大 age 24 帧。U-E4 中距离横穿/转弯冲突只给 10 帧短续，避免普通路口长期污染。

程序化判断 overlay 时必须优先读取 `road_structure_overlay.active`；普通 `secondary_road_structures` 可能只是候选冲突或不确定性，不能单独当 overlay 判据。

## 8. R2 route-level RGB 复核（2026-07-08）

目标：确认 R2 是否真的表示“对向单车道 / 有效可行驶通道被压缩成对向单车道”，避免普通 R1/R-E1 被 XODR opposite hint 或场景名误升为 R2。

runtime gate：

- 非 TwoWays 场景不再根据场景级 XODR sparse scan 批量开放 R2。
- `LAYOUT_R2_ROUTE_IDS` 当前为空；后续只有逐 route / 逐帧 RGB 确认后，才能把非 TwoWays route 手工加入白名单。
- 当前允许 primary R2 的来源只剩 `*_TwoWays` 和 `InvadingTurn` 中确有对向侵占/双向窄路规则空间的片段。

专项结果：

| Scenario | Annotated routes | Frames | R2 routes | Primary RS 摘要 |
|---|---:|---:|---:|---|
| AccidentTwoWays | 457 | 71237 | 457 | R2 80.0%, R4 16.0%, R5 4.0% |
| ConstructionObstacleTwoWays | 454 | 70530 | 454 | R2 81.8%, R4 14.2%, R5 4.0% |
| HazardAtSideLaneTwoWays | 88 | 14923 | 88 | R2 87.2%, R4 6.3%, R5 6.5% |
| ParkedObstacleTwoWays | 96 | 14030 | 96 | R2 81.1%, R4 3.0%, R5 15.9% |
| VehicleOpensDoorTwoWays | 104 | 14157 | 104 | R2 44.8%, R4 49.9%, R5 5.3% |
| InvadingTurn | 98 | 11883 | 59 | R1 72.4%, R2 12.7%, R4 1.0%, R5 13.9% |

普通非 TwoWays 风险场景当前 R2 routes 均为 0：`Accident`、`ConstructionObstacle`、`ControlLoss`、`BlockedIntersection`、`CrossJunctionDefectTrafficLight`、`CrossingBicycleFlow`。

R2 只改 ROAD_STRUCTURE，不自动改 EVENT。R2/R-E1 是正常对向单车道通行背景；U-E2/R-E2 仍必须由 XML trigger、具体障碍距离、TwoWays core/stuck/hazard、door/open、真实借道/回正轨迹等独立证据触发。

## 9. 后续复核规则

- 需要复查旧结论时，先读本归档，再到 `ROAD_STRUCTURE_PER_SCENARIO_RULES.md` 查当前每场景规则。
- 若需要打开 RGB，优先重新生成小范围 sheet 或使用现有本地 `collection_output/` 证据；不要把 RGB/contact sheet/large JSON 加入 git。
- 若产生新的高成本人工审计结论，应新增小型摘要 MD 或更新本归档；不要再为每次实验单独散落多个日期 MD。
- 若代码改动影响 `collection_output/*_result.json` 或 SFT 数据读取路径，必须同步更新 README、PROJECT_CONTEXT 和相关训练文档；大体量 result JSON 本身仍不通过 git 携带。
