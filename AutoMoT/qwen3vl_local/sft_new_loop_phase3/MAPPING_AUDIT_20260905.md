# Phase3 映射、RGB 与动作候选审计（2026-09-05）

后续扩展审计与最新产物见 [BOUNDARY_AUDIT_20260905.md](BOUNDARY_AUDIT_20260905.md)。
下面的 filtered_v5 / 41 项测试是较早的历史结果；当前为 filtered_v8、动作规则 v5。
旧 balanced_v3 轨迹缓存不兼容新规则，不能执行下文历史复现命令继续训练。

## 结论与影响范围

**普通无灯路口不应自动算 U-E7；这一修正无需改动现有 Phase1/2 模型接口。**
Phase1 本来问的是可辨认的异常信号灯硬件，不是“没有信号灯”；Phase2 不直接问 U-E7。
本次保留 Phase1/2 的源码、prompt、人工答案表；在 Phase3 独立适配历史 taxonomy。
`ROAD_EVENT_CLASSIFICATION_PLAN.md` 同步澄清历史 U7 与当前灯故障语义的差异。

已经确认的是下述接口语义和若干具体反例。**不能声称每条旧 RS/EVENT span、每个动作
边界都已精确确认**。扩大 RGB 核查后仍发现 RS 起点偏早、事件尾段延长和 lane-id
不等于跨线等问题，当前生成物是带排除规则的候选数据，不是全量人工真值。

## 1. Phase1 / Phase2 回答究竟恢复什么

### RS 映射

| 实际回答 | 恢复 | 语义与边界 |
|---|---|---|
| RS1=YES，其他结构没有矛盾 YES | R1 | 普通道路通道，车道保持/跟车规则主导；道路另一侧有对向车不必然 R2 |
| RS2=YES，其他结构没有矛盾 YES | R2 | 可用空间接近单通道，对向借道/共享约束直接参与当前决策 |
| RS4=YES，其他结构没有矛盾 YES | R4 | 当前局部路口由可辨认信号灯硬件/灯控规则主导 |
| RS5=YES，其他结构没有矛盾 YES | R5 | 当前局部无可用信号规则的路口，以 STOP/yield/路权/空隙判断为主 |
| RS1/RS2/RS4/RS5 **四问完整且全 NO** | R3 | 上游完整结构合同的剩余类；不要额外要求 Phase1 HIGHWAY=YES |
| hierarchical 的 RS_HIGHWAY=YES | R3 | 这是结构分支答案，与 Phase1 独立事实 HIGHWAY 不同 |
| 只问了部分结构、未出现可确定的 YES | UNKNOWN | 未问不能补成 NO；不能因此自动恢复 R3 |
| 多个互斥结构 YES，或 RS_HIGHWAY 与其他结构冲突 | UNKNOWN | 应重问/回查，不能任意选优先级覆盖矛盾 |

Phase1 `HIGHWAY` 是独立可见事实，不代替结构问题。完整四问全 NO 的解码规则正确，
不代表源数据中每个 R3 都正确：InterurbanActorFlow/Town12 的一个 0–4 段仍是黄色
对向中心线乡间道路，已经精确隔离该段。不能为了这条错标修改整个 R3 解码逻辑。

### 七个异常事件

| 异常 | 已有回答来源 | Phase3 context | 当前询问动作 | 视觉确认重点 |
|---|---|---|---|---|
| U-E1 | Phase2 UE1=YES | LEAD_BRAKE | 减速、停车、恢复 | 前车已有急减速影响；持续等候与释放阶段不应凭“前车近”重造急刹 |
| U-E2 | Phase1 STATIC_OBSTACLE=YES | STATIC_BLOCKAGE | 五动作 | 静态占道与实际绕行空间；目标离开哪条车道需看边界、后侧/对向空隙 |
| U-E3 | Phase2 UE3=YES | DYNAMIC_CUTIN | 减速、停车、恢复 | **他车**正在/明显准备侵入当前通道；排除稳定跟车、对向正常通过、ego 视差 |
| U-E4 | Phase1 VULNERABLE=YES | VULNERABLE_CROSSING | 五动作 | 包括横穿者和沿路骑行者；兼容 ID 中 CROSSING 不应限制成仅横穿 |
| U-E5 | Phase2 UE5=YES | ONCOMING_INVASION | 减速、停车、恢复 | 对向车侵占 ego 可用通道；锥桶本身和已清空尾段不证明侵入 |
| U-E6 | Phase2 UE6=YES | JUNCTION_RULE_CONFLICT | 减速、停车、恢复 | 他车违反当前路口规则且进入冲突路径；正常主路优先通过不是违规 |
| U-E7 | Phase1 TRAFFIC_LIGHT_ABNORMAL=YES | SIGNAL_FAILURE | 减速、停车、恢复 | 已有灯具的异常工作证据；无灯、暗夜、灯面不可读均不自动成立 |

Phase2 的 UE 输出只有在 `INVALID_EVENT_CONTEXT=NO` 时使用；没有这个有效性答案
不能默认有效。Phase1 独立事实不随 Phase2 invalid 被清除。
这里“七类能确定”指七个独立标志的来源明确，不意味着一帧只能有一个事件。
CrossJunctionDefectTrafficLight 有 U6+U7 并发；旧单一优先级会漏掉一个，应保留集合。

U1/U2/U3/U4/U5 与 R1–R5 的兼容范围是“可共存”，不从 RS 自动生成正例。
U6/U7 限局部路口 R4/R5。高速他车真实切入仍是 U3，复用 Phase2 显式 RGB YES
清单；HIGHWAY_CUTIN 仅为审计子型。普通高速并行/稳定跟车仍不能造 U3。

### 常规事件与在线前提

| 常规事件 | 可否从异常全 NO 唯一恢复 | Phase3 处理 |
|---|---|---|
| R-E1 普通路段行驶 | 否 | 本轮没有独立动作桶 |
| R-E2 目标变道/恢复车道 | 否 | POST_BYPASS_RETURN，五动作；需明确 transition/导航或恢复状态 |
| R-E3 常规匝道合流/并线/驶出 | 否 | RAMP_MERGE_EXIT，五动作；需活动过渡段证据，R3 本身不足 |
| R-E4 正常灯控路口行为 | 否 | 本轮没有独立动作桶；可作为并发道路背景 |
| R-E5 无灯路口常规路权让行 | 否 | UNSIGNALIZED_PRIORITY，三种纵向动作；仅从显式 R5/R-E5 候选进入 |

七异常加 R-E2/R-E3/R-E5，共十个 context；仍只有五种动作。
普通 STOP 或让行口属于 R5/R-E5 候选，不自动算 U6/U7。无灯口若有明确抢行/违反
优先权证据，可以是 U6；没有故障灯硬件证据不成为 U7。

`OppositeVehicleTakingPriority/R5/U-E7` 的旧 raw 标签与已审计 Phase1
`TRAFFIC_LIGHT_ABNORMAL=NO` 不一致。`source_mapping.py` 去掉此旧 U7，保留原标注
已经存在的 R-E5；不会把所有 R5 无条件生成为常规正例，也不擅自将这些片段改成 U6。
U-E8 不在七问输出内，本轮不把含 U8 的片段伪装为普通 R-E5。

R-E2 **不一定跟在 U-E2 后**：沿路骑行绕行、ParkingExit、普通目标变道，甚至某些
旧 span 的 R-E2 在 U2 之前。本轮只有真实既往 U2 才写“此前遇到静态阻塞”，还会明确
要求判断是否真正离开原车道；不向所有 R-E2 样本编造绕障历史。
两条变道 NO 仅表示未来三秒不跨线，不能清除等待中的恢复状态；移除旧 24 帧截断。
`RecoveryState` 是可供接入的状态组件，尚不能冒充已完成的 RGB 在线 transition 检测器。

## 2. 本次证据覆盖的准确口径

| 工作 | 范围 | 可以证明什么 |
|---|---|---|
| 缓存逐帧机器扫描 | 582 条正常 route，68,073 帧，42 scenarios，197 scenario×Town | 源 RS/EVENT 组合、候选映射、速度/lane 元数据覆盖 |
| 本轮实际查看连续 RGB 图组 | 197 个组合，11 Town，197 图组，共 7,826 个缩略帧格 | 更广的天气、道路、参与者特征与明显错误；不是 582 条全程人工复核 |
| 放大原始 RGB | 具体疑点连续帧，另见 raw followup 和本文示例 | 确认小目标、视差、跨线等局部事实；不能外推整条 route |
| 自动动作候选与均衡诊断集 | filtered_v5，train/val/test 各 384 条 | 10×32 有效 +64 invalid；采样、提示词、解析的数据链路 |

先执行异常时长 route 过滤，复用既有 full_route_rgb_label_review 缓存，没有重建重复图组。
本次不是整个 LEAD 数据集全量人工检查。`rgb_mapping_review_20260905.jsonl` 每条记录
均保留 `action_boundaries_confirmed=false`。机器 summary 的
`new_manual_rgb_reviews=0` 表示扫描程序没有看图，与独立人工记录并不矛盾。

本地没有发现可用 `.xodr` 原文件。本轮读取的是已有注释及 meta 的 road/lane 字段，
**没有完成原始 OpenDRIVE 车道段拓扑核验**，尤其不能把缓存的 map provenance 当真值。

## 3. 具体反例与动作边界

原始图片位于 `AutoMoT/lead_data/<scenario>/<route_id>/rgb/<四位帧号>.jpg`。
全部 197 个组合的 route_id、Town、图组路径与逐段说明均在人工 notes 文件中。

### 绕障恢复：目的地在左仍应向右

`AccidentTwoWays/Town01_Rep0_route_001543_route0_01_08_23_18_47`：

- U2 结束于 99，R-E2 从 100 开始，真正 lane +1→-1 恢复在 134。旧 24 帧截断漏掉后段。
- 122/124 的最终目标 y 约 -34.9/-36.8m，但近期恢复方向是 RIGHT；不能用目标 y 符号选侧。
- 58 的速度窗约 `[6.898,6.371,3.060,3.328,1.878,.214,.308,.316,.350]`，即时窗中
  有持续近停，应 STOP；旧逻辑误为减速。60 后极小负速度是静止数值抖动，不应当缺失。
- 104 是走停交替的释放片段，先达到加速阈值，不能把窗口中任一低速值都当持续停车。

### U1 与普通停车：时域定义必须一致

HardBreakRoute/Town13（notes 65）：88 速度从 13.454 降至 0，但随后 .996，未在
即时窗内连续两帧近停；当前规则为 DECELERATE。90 窗内已有 .426/.099，才为 STOP。
这证明“未来任何时候到零就 STOP”会与当前短窗合同冲突。
HardBreakRoute/Town12（notes 64）40 是持续等候，70 开始释放；不能仅靠前车距离
重新认定每帧都发生新急刹。

STOP 当前阈值仍是工程候选规则：两帧 ≤0.5m/s、1.5s 窗；DECELERATE/RESUME 在
2s 窗中取首次达到 `max(1.2m/s, 20%×max(v0,1))` 的方向，STOP 优先。
四帧稀疏 RGB 能否稳定预测这些细边界，尚须逐条复核和后续模型评估，不能保证精确可学。

### U3：对象和车道关系比 scenario 名重要

- DynamicObjectCrossing/Town01，113–115：原图稳定前车，旧 U3 隔离。
- DynamicObjectCrossing/Town07，116–119：红皮卡沿同一双黄线右侧弯道行驶；
  隔离旧 U3 117–119，不将未确认正例强制转为负例。
- DynamicObjectCrossing/Town02，69–71：远处路口横向车辆，雾中距离/冲突归属不足；
  仍为待审，不根据缩略图擅自删改。
- ParkingCutIn/Town13，208–212：黄色出租车静止，**绿色车**在 210–212 开始向通道
  外摆。保留候选，但 208–209 起点仍需核查；不能因盯错车而删除整个事件。

### U5：同 road + 连续新 lane_id 仍不足以证明变道

`InvadingTurn/Town13_Rep0_…` 完整 route_id 见
`lateral_rgb_uncertainties_v1.jsonl`（notes 79）。road=355，98 时 lane -1→-2，128 时
-2→-1。原图 96–99、127–130 仍位于同一黄线与右侧边线内，98 附近 theta 约
0.175rad 且轨迹平滑；可能涉及 lane-section 编号或地图投影，原始地图未核实。

已加入精确横向审计否决：未来窗包含上述切换时标记 `lateral_observation_complete=false`；
FULL_MANEUVER 候选跳过，纵向问组可保留，但不把这两个切换当作已确认变道。
这不是自动发现所有类似错误的通用地图算法，其他 lane-id 正例仍需 RGB/车道段回查。

### U4、U6、U7 与无灯常规行为

- HazardAtSideLane / TwoWays 跨 Town 骑行者常沿路行驶，U4 必须容纳安全绕行，不能
  仅问“行人是否横穿”；VehicleTurningRoute 的 U4 可从 R4/R5 延续到出口 R1。
- OppositeVehicleRunningRedLight/Town02（notes 101），16–19 ego 可见绿灯，
  28–32 白色大车横穿冲突区，支持具体 U6 候选；其余 Town 的远处接近段不能照搬。
- CrossJunctionDefectTrafficLight/Town03（notes 42），有信号硬件与横向冲突；
  图组小图不能独立证明所有灯态，保留 U6/U7 并发来源，灯态精确时间仍待审。
- VehicleTurningRoute/Town12（notes 193），120–134 清楚 STOP 口等待，135 后右转，
  141 后出口骑行者；普通让行是 R-E5，随后 U4，不是 U7。
- SignalizedJunctionRightTurn/Town02（notes 166），19–24 红灯，25 变绿，随后右转；
  正常灯态变化是 U7 负边界。夜暗或雾中辨不清灯不能反推灯坏。

### R-E3 与提前 R4

HighwayExit/Town12_Rep0_4081_0_route0_01_08_20_49_10：55/58 为减速并向右，
59 已在 lane -4；目标 y 为负仍不改变右侧出口动作。沿匝道自然接续不必有跨线。
StaticCutIn/Town13（notes 176）4–32 虽源标 R-E3，多帧仅见稳定高速跟车，局部
合流拓扑缺乏明确证据；不可认定整个 R-E3 span 已确认。

多个正常转弯 scenario 从远处直路就标 R4，例如 notes 168/172/174/178；另一些
出口直路仍保留 R4，例如 notes 188。默认风险过滤保留，但它不能保证捕获所有错帧。
这是源 span 的局部质量问题，不宜通过放宽 RS4 prompt 来迁就。

## 4. 实现、采样与复现

- 映射：`context_taxonomy.py` / `source_mapping.py`；精确排除在
  `mapping_rgb_decisions_v2.jsonl`，不会写回 Phase1/2。
- 动作：`trajectory_action.py`，未来轨迹只进入离线 expert label/evidence，prompt
  只含 RGB、当前语境、已发生历史及目标坐标。车道段冲突另由 `lateral_rgb_audit.py` 审核。
- prompt：五动作、R-E5、沿路 U4、恢复等待、最终目标不决定车道侧、速度时域同步更新。
- invalid：覆盖全部十个 asked context，按 source/true RS/错误 asked context 审计。
  通过明确错误 RS 或明显几何错配构造；不可把低能见度或合法全 NO 作为 invalid。
- 默认均衡：十个有效 context 各 N，invalid 2N；invalid/valid=20%，占总量16.7%。
  构建、训练默认、定额 generation/eval 一致；每个 context 内按可用动作签名分配。
  这不等于每种动作 YES:NO 均为1:1；缺少动作正例时不能凭空补标签。
- 诊断 split seed=20260912、val/test 各20%，只是保证缓存子集各桶可检查的划分；
  不是正式全量划分，更不是按模型分数挑 seed。未执行模型训练/推理。

从仓库根目录生成本次诊断索引（缓存仅为同版本轨迹复用，重新应用最新审计否决）：

```bash
python AutoMoT/qwen3vl_local/sft_new_loop_phase3/build_dataset.py \
  --candidate-cache AutoMoT/qwen3vl_local/sft_new_loop_phase3/probe_output/mapping_audit_20260905/balanced_v3/candidate_frames.jsonl \
  --split-seed 20260912 --test-ratio 0.2 --val-ratio 0.2 --target-per-context 32 \
  --output-dir AutoMoT/qwen3vl_local/sft_new_loop_phase3/probe_output/mapping_audit_20260905/filtered_v5
```

查看 `probe_output/mapping_audit_20260905/review.html`：354 条被索引或 notes 引用的
route，可按左右键逐帧翻原图，对照候选速度窗、当前/错误 RS、目标坐标和真实 prompt。
有图片但无候选记录不是动作全 NO。HTML 只引用原图，不复制 RGB。
可用 `build_mapping_review.py --candidate-index <frame_index.jsonl> --output <review.html>` 重生。

## 5. 仍需要审核的内容

验证：41 项映射/轨迹/解析回归通过；三 split 的训练与 eval 实际采样入口均为
10×32+64；route 级 split 无交集，1152 条历史 RGB 路径存在；Python AST、shell 语法、
HTML JavaScript 语法与 git diff 空白检查通过。Phase1/2 对 HEAD 无差异。
这些检查证明实现合同和可读性，不代表模型成绩或全量标签精度。

1. 七类映射来源已厘清，但所有 582 条 route 的事件起止与动作切换尚未逐帧人工确认；
   197 个图组主要补齐跨 Town 特征覆盖。其余 route 和已看图组之外的帧仍要继续审核。
2. U3 的远处路口交通、U5 的清空尾段、U6 的远处接近段、R4 的提前/滞后边界需要
   route/frame 级决定；暂不为凑桶改成确定标签。详见 notes 与 raw followup。
3. R-E3 的真实过渡 gate、R-E2 的普通导航目标车道不能由最终目标坐标唯一决定；
   当前离线显式源事件仍只是候选 gate。在线还需要可观察的导航/拓扑及恢复状态接入。
4. 同 road lane-id 切换不是充分证据，需原始 xodr 的 lane-section 连通关系或可靠的
   RGB 跨线确认。当前保守跳过跨 road/缺观察的 FULL_MANEUVER 会损失部分真实合流样本。
5. Phase2 对 interrupted junction 上 U3 的显式保留，与泛化的 wrong-domain invalid
   文案之间仍有边界需核查；本轮没有改 Phase2，以免影响现有 adapter 合同。
6. R-E2 诊断集已含绕障历史和非绕障两类；目前比例由候选容量/签名采样形成，尚非
   固定子型配额。正式集须审计各 split 的历史覆盖，不能用虚构历史填足比例。

**本轮产物可用于审核规则和数据链路，不能以“完美标定/全量确认”名义直接宣告结束。**
