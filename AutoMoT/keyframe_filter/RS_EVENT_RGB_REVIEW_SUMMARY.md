# RS/EVENT RGB 逐帧复核总结

本轮复核按当前 RS/EVENT 逻辑重新跑：

```bash
python AutoMoT/keyframe_filter/rs_full_frame_review.py \
  --scenario all \
  --samples-per-town 5 \
  --output-dir /tmp/automot_rs_event_rgb_review_all_town5_current \
  --frames-per-sheet 40 \
  --sheet-cols 4
```

- 覆盖：43 个 scenario，993 条有效 route，117122 帧。
- 抽样：每个 scenario 下每个 town 最多 5 个 id；若合法 route 不足 5 个，则读取该 town 全部可用 id。
- RGB 证据：每条 route 的所有帧都读取并渲染到 `all_frames_*.jpg` contact sheet，sheet 上叠加当前 RS/EVENT 标签、置信度和 review 原因。
- 临时证据目录：`/tmp/automot_rs_event_rgb_review_all_town5_current`，不入库；关键结论保存在本文件与规则代码/MD 中。

## 关键发现

1. 最大的“异常”来源不是 RGB 误判，而是 XML route 投影误差导致 XODR/topology 和结构窗口被降权。旧审查脚本把这类软复核也计成 candidate anomaly，噪声过大。
2. `rs_full_frame_review.py` 已新增 `review_severity`，把复核分成 `hard_issue`、`boundary_review`、`soft_evidence_note`、`event_boundary`，并在 sheet 底部优先显示 primary RS 相关原因，避免次候选原因污染主判断。
3. `InterurbanActorFlow`、`InvadingTurn` 的代表 RGB 显示无信号/STOP/T 路口 R5 方向基本正确；问题主要是 review 原因里混入 R3/R2 次候选的弱 XODR 文案。
4. `ConstructionObstacleTwoWays` 等 TwoWays 样本中 R2 核心持续段与 RGB 窄路/障碍绕行匹配，最长 R2 段保留逻辑有效；剩余问题是 XODR 不确认应作为软提示，而不是硬错配。
5. `BlockedIntersection` 发现真实逻辑漏洞：部分 RGB 明显是 STOP/无信号阻塞路口，旧规则因场景族默认 `signalized_junction` 标成 R4。已修复为 `blocked_intersection` 规则族：灯控同源证据给 R4，STOP/yield/无灯控制源给 R5，阻塞本身只进 EVENT。
6. `NonSignalizedJunctionLeftTurn*` 发现硬错误：RGB 是无灯/STOP 左转口，但 bbox 偶发同时报 `traffic_light` 与 `stop_sign`，旧代码动态打开 R4 并把整条 route 标成 R4。已将两个 LeftTurn 场景加入 strict no-R4，并在无灯/T-junction 规则中让 STOP/yield 在无有效 meta 灯态时压制 bbox/static-signal 的 R4 提升。

## 每场景复核结论

下表是早期 town5 RGB 抽样的复核结果，部分 TwoWays / 独立停车结构结论已被 2026-07-06
全量逐帧 RGB audit 覆盖；最终候选池以 `ROAD_STRUCTURE_FULL_RGB_AUDIT_20260706.md`
和 `collector.py::SCENARIO_TO_ROAD_STRUCTURE` 为准。

| Scenario | Frames | 当前 RS 分布 | 主要问题 | 后续口径 |
|---|---:|---|---|---|
| Accident | 5005 | R1:4169, R4:836 | projection/topology review 高 | 事件不决定 RS；R4 只看灯控/路口证据 |
| AccidentTwoWays | 4087 | 旧抽样 R1:2198, R2:1109, R4:780 | R2 topology soft review 多 | 已由全量 RGB 覆盖：候选删除 R1；非路口按有效可行驶对向单车道 R2，灯控/STOP/无灯覆盖 R4/R5 |
| BlockedIntersection | 2814 | 旧逻辑 R1:229, R4:2585 | STOP/无灯路口被默认 R4 | 已改 R1/R4/R5，控制源决定 R4/R5 |
| ConstructionObstacle | 5207 | R1:4221, R4:986 | projection/topology review 高 | 施工障碍只进 EVENT；RS 仍看道路控制源 |
| ConstructionObstacleTwoWays | 4364 | 旧抽样 R1:2035, R2:1373, R4:956 | R2/XODR 确认弱 | 已由全量 RGB 覆盖：候选删除 R1；非路口默认 R2，真实控制源覆盖 R4/R5 |
| ControlLoss | 6394 | R1:4086, R4:2308 | projection review 极高 | 失控只进 EVENT；RS 不由失控动作决定 |
| CrossJunctionDefectTrafficLight | 2455 | 旧结果 R1:101, R5:2354 | 少量低置信 | 已修正口径：保留有灯控 R4，defect/失效进入 U-E7 |
| CrossingBicycleFlow | 507 | R1:201, R4:306 | projection review 高 | 自行车横穿只进 EVENT；R4 需灯控证据 |
| DynamicObjectCrossing | 7135 | R1:4852, R4:2283 | projection review 极高 | 动态横穿只进 EVENT |
| EnterActorFlow | 1050 | R3:1050 | R3 topology soft review | 保持 R3/no-R4 |
| EnterActorFlowV2 | 627 | R3:627 | R3 topology soft review | 保持 R3/no-R4 |
| HardBreakRoute | 1471 | R1:847, R3:129, R4:495 | projection review 高 | 急刹只进 EVENT；按 route RGB 分桶 |
| HazardAtSideLane | 2187 | R1:1263, R4:924 | R4 灯控上下文弱 | side-lane hazard 不制造 RS |
| HazardAtSideLaneTwoWays | 1785 | 旧抽样 R1:1461, R2:92, R4:232 | projection review 高 | 2026-07-09 纠正：候选删除 R1；R2 是有效可行驶对向单车道道路空间，自行车/行人进入路径用 U-E4，离开对象后回正才 R-E2，不再使用 U-E2 |
| HighwayCutIn | 2119 | R3:2005, R4:114 | R3 topology soft review | 主体 R3；少量 R4 必须灯控同源 |
| HighwayExit | 1212 | R3:1212 | R3 topology soft review | 保持 R3/no-R4 |
| InterurbanActorFlow | 1228 | R1:420, R5:808 | review 文案串入 R3 次候选 | R5 方向可用；删除 R4 继续成立 |
| InterurbanAdvancedActorFlow | 1171 | R1:1127, R5:44 | projection review 高 | 只有明确无灯/STOP 窗口给 R5；2026-07-09 起 junction 窗口约 +30%，通过 R5 路口且有换道轨迹证据时允许 R-E2 |
| InvadingTurn | 1278 | R1:608, R5:670 | review 文案串入 R2 次候选 | 无信号路口 R5；R2 只保留侵占主导段 |
| MergerIntoSlowTraffic | 1730 | R3:1729, R4:1 | R3 topology soft review | 主体 R3；少量 R4 要灯控同源 |
| MergerIntoSlowTrafficV2 | 1688 | R3:1688 | R3 topology soft review | 保持 R3/no-R4 |
| NonSignalizedJunctionLeftTurn | 1946 | 修复前 R1:23, R4:571, R5:1352；目标复测 R5:756 | bbox traffic_light 与 stop_sign 冲突导致误升 R4 | strict no-R4；STOP/yield/无灯左转给 R5 |
| NonSignalizedJunctionLeftTurnEnterFlow | 2364 | 修复前 R1:24, R4:542, R5:1798；目标复测 R1:14, R5:942 | bbox/static signal 弱提示在无灯口抢占 R4 | strict no-R4；enter-flow 不改 R3 |
| NonSignalizedJunctionRightTurn | 808 | R1:9, R4:77, R5:722 | projection review 高 | 保持 R1/R4/R5 混合 |
| OppositeVehicleRunningRedLight | 4854 | R1:370, R4:4484 | projection + no-meta R4 review | 保持 R4；闯红灯只进 EVENT |
| OppositeVehicleTakingPriority | 1558 | R1:52, R4:408, R5:1098 | R4/R5 仲裁边界 | R5 为主，少量灯控子集 R4 |
| ParkedObstacle | 4623 | R1:3921, R4:702 | projection review 高 | 停车障碍只进 EVENT；只按道路证据升 RS |
| ParkedObstacleTwoWays | 1340 | 旧抽样 R1:1001, R2:335, R4:4 | projection/R2 soft review | 已由全量 RGB 覆盖：候选删除 R1；非路口默认 R2，STOP/无灯可 R5 |
| ParkingCrossingPedestrian | 1211 | R1:754, R4:457 | projection review 高 | 行人只进 EVENT；停车空间不单独成 RS |
| ParkingCutIn | 2166 | R1:551, R4:1615 | R4 灯控上下文弱 | 停车切入不制造 R4，需灯控同源 |
| ParkingExit | 1569 | R1:1405, R4:164 | R4/parking-exit 仲裁边界 | parking exit 窗口给 R1/R-E2，灯控优先 R4 |
| PedestrianCrossing | 2014 | R1:143, R4:1616, R5:255 | R4/R5 控制源复核多 | 行人不决定 RS；按灯控/无灯源分 R4/R5 |
| PriorityAtJunction | 955 | R1:155, R4:566, R5:234 | R4/R5 仲裁边界 | 保持混合，不能按 Town 自动判 |
| RedLightWithoutLeadVehicle | 6822 | R1:410, R4:6412 | no-meta R4 review 多 | 保持 R4；加强 stopline/灯控同源证据 |
| SignalizedJunctionLeftTurn | 3639 | R1:584, R4:3055 | projection review 高 | 保持 R4 policy |
| SignalizedJunctionLeftTurnEnterFlow | 3480 | R1:488, R4:2992 | projection review 高 | 保持 R4；enter-flow 只进 EVENT |
| SignalizedJunctionRightTurn | 3080 | R1:328, R4:2752 | projection review 高 | 保持 R4 policy |
| StaticCutIn | 1331 | R1:636, R3:537, R4:158 | route 分桶/topology review | 高速桶 R3，城市/普通路段按 R1/R4；2026-07-06 全量逐帧 RGB 后候选收紧为 R1/R3/R4/R5 |
| T_Junction | 3785 | R1:85, R4:3690, R5:10 | no-meta R4 review 多 | 保持 R1/R4/R5；缺灯态不能自动 R5 |
| VehicleOpensDoorTwoWays | 1518 | 旧抽样 R1:731, R4:787 | R4 灯控上下文弱 | 已由全量 RGB 覆盖：候选删除 R1；两侧停车/开门压缩可行驶 lane 时主 RS 为 R2，真实控制源覆盖 R4/R5 |
| VehicleTurningRoute | 7372 | R1:1291, R4:4539, R5:1542 | projection review 极高 | 转弯路线不决定 RS；控制源分 R4/R5 |
| VehicleTurningRoutePedestrian | 2036 | R1:377, R4:938, R5:721 | projection review 极高 | 同 VehicleTurningRoute，行人只进 EVENT |
| noScenarios | 3137 | R1:2434, R4:703 | projection review 高 | 保守 R1；强灯控才 R4 |

## 已同步修改

- `collector.py`
  - `BlockedIntersection` 候选从 R1/R4 改为 R1/R4/R5。
  - 新增 `blocked_intersection` 规则族：R4 需要灯控同源证据；STOP/yield/无灯控制源给 R5。
  - 新增 `blocked_r4_without_meta_tl_requires_rgb_confirmation` review reason。
  - `NonSignalizedJunctionLeftTurn`、`NonSignalizedJunctionLeftTurnEnterFlow` 加入 no-R4 动态兜底黑名单。
  - 无灯路口/T 路口中，若无有效 `traffic_light_state` 且有 STOP/yield 证据，bbox/static signal 弱提示不再提升 R4。
- `rs_full_frame_review.py`
  - 新增 `review_severity_distribution` / `issue_bucket_distribution`。
  - RGB sheet 底部原因按 primary RS 过滤，避免次候选 R2/R3/R4/R5 弱证据污染主解释。
  - `candidate_anomalies.jsonl` 同时保留 `reasons` 和 `raw_reasons`。
- 文档
  - `ROAD_EVENT_CANDIDATE_MAPPING.md`、`ROAD_STRUCTURE_PER_SCENARIO_RULES.md`、
    `ROAD_EVENT_CLASSIFICATION_PLAN.md` 已同步 BlockedIntersection 新口径。

## 2026-07-09 边界复核补充

- HighwayExit 的出口变道 R-E2 在原有后补 4 帧基础上继续前补 2 帧、后补 2 帧，
  累计为前补最多 2 帧、后补最多 6 帧，随后再进入 R-E3。
- InterurbanActorFlow 的已有 R-E2 核心按轨迹前补最多 3 帧、后补最多 4 帧，不跨 R5/R-E5。
- InterurbanAdvancedActorFlow 的无灯/STOP junction 配置从 `55/25m` 放宽到 `72/33m`；
  通过 R5 路口时若有 `changed_route` + 横向偏移/换道符号证据，R5 段也可输出 R-E2，
  并按轨迹前补最多 3 帧、后补最多 4 帧。
- MergerIntoSlowTraffic* 参考 HighwayCutIn/HighwayExit：刚开始/中间普通主线跟车保持 R-E1，
  trigger-only 圆窗不再单独制造 R-E3，靠近 merge/actor-flow 切 R-E3，真实目标变道切 R-E2；
  R-E2 按轨迹前后各最多补 5 帧，R-E2 后近 actor-flow/merge tail 最多 64 帧保持 R-E3，
  并允许 8 帧短空窗桥接；3 帧以内夹在 R-E1 中的孤立 R-E3 小岛平滑回 R-E1。
- InterurbanActorFlow 无 stop/junction 证据的初始直道恢复 R1/R-E1。
- InvadingTurn 的 U-E5 按 RGB 可见锥桶/对向占道长度保持；final pass 对连续
  `passive_oncoming_invasion`、trigger>=35m 且有 R2 或 R1 响应证据的长 cluster 输出 U-E5，
  单段最多补 48 帧，2026-07-10 全量为 4744 帧且含 R2 route 不再缺 U-E5。
- NonSignalizedJunction* 使用“移动且贴 route 中心的直行段”门控恢复 R1，
  仅停车/转弯/路口核心保留 R4/R5。
- NonSignalizedJunctionRightTurn 追加 `distance_to_intersection_index_ego` 局部核心门控：
  全量 93 route / 8074 帧从 `R1=3446,R4=709,R5=3919` 调整为
  `R1=4170,R4=598,R5=3306`；`Town12_1210_0` 的起始直道和驶离直道均恢复 R1，
  灯控子集 `Town13_75_0` 核心 R4 仍保留。

## 2026-07-09 Hazard / VehicleOpenDoor 补充

- VehicleOpensDoorTwoWays 的 U-E2 后恢复 R-E2 起点提前最多 3 帧、终点提前最多 4 帧。
- HazardAtSideLane* 的 U-E4 结束后若仍有回正轨迹，R4/R5 下保留 regular+R-E2 overlay；
  全量补回 819 帧 U-E4 后恢复 R-E2，U-E4 总量不变。U-E4 后 8 帧内仍有
  `target_lane_change/changed_route/route_lateral_abs` 支撑却没有接 R-E2 的残留为 0。
- HazardAtSideLane 非 TwoWays 前 30 帧仅 bbox-only STOP、close-trigger 或 untrusted XODR
  不能升 R4/R5，清除 393 帧初始弱路口；`Town13_1619_10` 开头暗光直道恢复 R1。
- HazardAtSideLaneTwoWays 前 30 帧 bbox-only STOP 伪路口回 R2，清除 336 帧初始误标 R5。

## 2026-07-09 Junction / Parking 补充

- T_Junction、PedestrianCrossing、PriorityAtJunction、OppositeVehicleTakingPriority
  的 R4/R5混合 route 全部归零，控制源按 route 有效灯态锁定。
- R4/R5 单控制源锁扩展到所有同时允许 R4/R5 的场景：只防止控制源互跳，并只同步
  R4/R5 对应 regular event，不改变 U-E 触发逻辑。`VehicleTurningRoutePedestrian`
  全量验证从 50 条 mixed route / 47 次直接 R4-R5 跳变降为 0 / 0，U-E4 保持 3454 帧不变。
- 24 个同时允许 R4/R5 的场景历史全量结果复算：旧结果 759 条 route 同时出现 R4/R5、
  698 条 route 有相邻 R4/R5 跳变；按新 route-level 控制源锁复算后 mixed=0、direct=0。
  实际重跑 Accident、AccidentTwoWays、BlockedIntersection、ConstructionObstacle、
  ConstructionObstacleTwoWays 共 1405 route / 210072 frame，mixed=0、direct=0。
- OppositeVehicleRunningRedLight U-E6 二次修正：不只看违规车瞬间，也看自车停车/让行等待与轨迹响应。
  2026-07-10 全量 287 route / 28165 帧中，U-E6 从旧 1075 扩为 3515 帧；
  `Town13_1047_5` f128/f145/f155 均为 `R-E4 + U-E6`，冲突解除后的 f170 回 R-E4。
- PriorityAtJunction route lock 保护本地灯控/stopline：99 route / 9702 帧，R4 从 1333 增到 1362，
  route lock 改动从 68 降到 3，`Town12_4022_0` f18-f45 保持 R4/R-E4，f60 回 R1/R-E1。
- RedLightWithoutLeadVehicle 出口尾段收紧：355 route / 48310 帧，R1/R4=4021/44289；
  `Town01_Scenario7_16` f165/f170/f172 释放为 R1/R-E1。
- SignalizedJunctionLeftTurnEnterFlow Town01/02 起始弱 R4 过滤：173 route / 12884 帧，
  起始 210 帧回 R1/R-E1；`Town01 route_002329` f0-f12 为 R1，f16 起恢复 R4。
- T_Junction 出口侧延迟：246 route / 31498 帧，R1/R4/R5=2048/29374/76；
  `Town01_Scenario7_68` f44-f63 保持 R4/R-E4。
- VehicleTurningRoutePedestrian 短缝复验：91 route / 20399 帧，
  `vehicle_turning_junction_gap_recovery` 修正 22 帧、temporal smoothing 修正 6 帧；
  最终 `R-E4/R-E5 -> R-E1/R-E2 <=6帧 -> 同类 R-E4/R-E5` 与对应 RS 短缝残留均为 0。
- ParkingExit 无灯伪 R4 route `6 -> 0`，R-E2 延续到变道完成；ParkingCutIn U-E3
  `1955 -> 3059`，补齐提前退出尾段。
- ParkedObstacle* 的恢复 R-E2 起点提前最多 3 帧、终点提前最多 4 帧。

### 2026-07-10 Priority/Parked/ParkingCutIn 微调

- OppositeVehicleTakingPriority 进入侧按 RGB 放宽到 `junction_pre_m=75`；复验 97 route /
  13238 帧，R1/R4/R5=6668/2078/4492。
- ParkedObstacle 进入侧按 RGB 放宽到 `junction_pre_m=72`；复验 168 route /
  20676 帧，R1/R4=18330/2346，U-E2 后 R-E2 仍保持恢复边界。
- ParkedObstacleTwoWays 复验 96 route / 14030 帧，没有残留短
  `R-E2 -> R-E4/R-E5 -> R-E2` 插缝；保留 8 帧以内兜底合并。
- ParkingCutIn U-E3 改为近距离 + 动态响应/横向证据触发；R4/R5 overlay 不再按
  distance-only 拖到前车消失。复验 97 route / 13892 帧，U-E3=330 帧 / 80 route，
  最长 span 7 帧；`1757_0` 收为 f48-51，`815_0` 收为 f211-215。

### 2026-07-10 ParkingExit / PedestrianCrossing 微调

- ParkingExit 初始驶出 R-E2 收尾按 RGB 提前约 5 帧：全量 241 route / 17586 帧，
  `R-E2 8698 -> 7903`，`R-E1 2604 -> 3399`，R4/R-E4 不变；210 条初始 R-E2 route
  中 158 条实际提前，52 条因不满足长度/边界条件保持原样。
- PedestrianCrossing 入口侧收紧、退出侧延迟：全量 98 route / 18141 帧，
  首个 R4/R5 起点无提前，最多向后 11 帧；`R1/R4/R5=4367/12621/1153`。
  `R4/R-E4 -> 1-8 帧 R1/R-E1/R-E2 -> R4/R-E4` 的短缝同步缝合 RS+EVENT；
  16-20 帧普通直行段仍保留 R1/R-E1，不做过度吞并。
