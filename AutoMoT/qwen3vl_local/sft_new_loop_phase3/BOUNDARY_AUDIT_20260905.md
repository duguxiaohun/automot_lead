# Phase3 边界审计：当前结论与已落实处理

Phase1/2 的代码、prompt、输出格式和答案表没有改动。Phase3 继续只使用五种动作：
`DECELERATE / STOP / RESUME / LANE_CHANGE_LEFT / LANE_CHANGE_RIGHT`。
七个 UE 与 R-E2/R-E3/R-E5 共十个**事件情境**共用这些动作，不是十种动作。

当前实现已补上“常规事件候选核对 → 动作问答 → 拒绝候选/回常规流程”的调度接口，
并将同 RS 错事件负例接入数据构建、train/eval 采样与指标。尚未运行 LoRA 训练，
因此这里说明的是代码和监督合同，不是模型已达到的驾驶精度。

## 逐类结论

完整 Phase1/2 答案映射见 `MAPPING_AUDIT_20260905.md` 与运行说明。
完整 RS 四问全 NO 才能恢复 R3，未问不作 NO；HIGHWAY 是独立可见事实。
多个异常 YES 保留并发。普通无灯路口不自动算 U-E7，也不自动算他车违规 U-E6。

| 情境 | 当前规则与已落实处理 |
|---|---|
| U1 前车急刹 | 保留纵向三动作。Town13 f213 的孤立速度峰值不再抢先触发 RESUME，先发生的真实减速为 DECELERATE。未达到 `max(1.2 m/s,20%)` 的轻微速度变化不额外创立动作类；全 NO 不表示精确恒速。 |
| U2 静态占道 | 保留五动作，含先等待对向/侧后车、再绕行。停车带内的正常静止车不是占路障碍；ConstructionObstacleTwoWays/Town12 f64–67 已成为明确负例依据。横向未知的窗口不监督变道 NO。 |
| U3 动态侵入 | 保留动态跨线而非稳定跟车的边界；ParkingCutIn/Town13 f96–98 的漏标已补。近处静止警车与远处移动出租车要分开追踪，不因盯错目标删除整段。去掉 prompt 中“ego必定保持车道”的断言，未问横向保持未知。 |
| U4 行人/骑车人 | 包含横穿与沿路骑行，两者共用五动作。沿路骑车人可先等待侧后/对向车，再绕行；旁边普通停车和无决策关系的远处用户不自动触发。小幅速度变化仍遵守同一纵向阈值。 |
| U5 对向侵入 | InvadingTurn/Town13 `1445_0` 原图 f50–61 的侵入关系不可辨，已隔离正例，未强行造 invalid；f68–72 灰车压中线且ego避向路肩，保留纵向监督。f73后还有第二辆对向车，不能删除整个尾段。 |
| U6 路口违规 | 保留“对方违规且侵入ego冲突路径”的前提。Town03/04 普通无灯路口的正常对向左转、横流已用来监督同RS的错误违规前提；正常有路权车辆不是U6。 |
| U7 信号异常 | 仍取 Phase1 显式灯故障事实。无灯、红灯、雾中看不清灯都不等于故障。Town03/04 普通无灯路口已作为同RS的U7负例；对真正故障源 span 不用单组绿灯或速度曲线猜测修补。 |
| R-E2 目标/恢复变道 | 保留绕障后恢复与普通导航变道。EnterActorFlow/Town13 f146–149 原图确认向左跨线，meta同road1216由lane-4到-3，支持主路R-E2；不能额外算匝道R-E3。等待或动作全NO不清除恢复状态；最终目标y符号不决定变道方向。 |
| R-E3 合流/驶出 | 保留真实活动匝道/分合流；自然车道接续可以没有跨线。StaticCutIn/Town12 f23–26、EnterActorFlowV2/Town12 f82–85 实为主路行驶，已从正例隔离，并在核对过完整history的anchor构造同RS错事件负例。 |
| R-E5 常规无灯让行 | 保留STOP/yield及正常路权语境。VehicleTurningRoute/Town04 f2–10、Town03 f0–7、DynamicObjectCrossing/Town05 f0–3 的普通接近/跟车段已隔离，不直接改Phase1 RS。 |

“隔离”是当前训练集合的明确排除决定，不等于确认其反面。未看清的事件不做负例。
源 span 其余帧仍是自动候选；以上结论没有把一个已确认 anchor 扩大成全route确认。

## 功能与标定合同

- `dispatch.plan_candidate_requests` 保留明确异常，常规阶段可在R3补问R-E3、在R5
  补问R-E5，并核对导航/绕障恢复R-E2候选。RS只产生可问候选，不宣布事件成立。
  `candidate_response` 严格检查完整输出；`CANDIDATE_REJECTED` 可回常规流程，
  `NOT_REJECTED_NO_ACTION` 不表示事件不存在或恢复完成。接口未接入CARLA。
- 仍使用原 `INVALID_ACTION_CONTEXT` 行，不新增第六个动作。该行明确核对道路与
  事件两方面，允许“道路正确但事件前提错误”。INVALID为YES时所有动作必须NO。
  INVALID为NO表示未发现明确反证，不能当作额外的精确RE分类置信度。
- 横向只接受完整Driving waypoint窗内的同road稳定lane切换。LEAD原始road/lane
  来自Any查询，ego_lane_id来自另一遍Driving查询，不能拼接两套身份。Shoulder、
  Parking、未知类型、跨road仍不监督横向NO。这里核对了meta地图身份和RGB；没有
  可用完整XODR，未声称已完成lane-section拓扑验证。
- 纵向仍为STOP 1.5s、速度变化2s；RESUME需连续两个样本满足加速阈值，已停起步
  允许轻微回落但即时窗内须保持≥2m/s。STOP与变道YES对应不同时间窗。
- prompt为 `v4_context_recheck`，动作规则仍为 `ordered_speed_driving_lane_v5`。
  index额外绑定 `mapping_contract_hash`：上游证据表、显式U3补标、隔离决定、同RS
  人工决定发生变化时，train/eval拒绝旧索引，避免换了规则还训练旧数据。

## 同 RS 负例与均衡结果

`same_rs_invalid_review_v1.jsonl` 现在有9条route/anchor人工决定，构造34个同RS
错事件候选问题；每条只使用已看过的四张实际输入RGB，先剔除异常route，沿用同一
route级train/val/test划分。已被明确判错的源R-E3标签只作来源桶，不冒充真实事件。
这34题通过 `same_rs_invalid.py` 接入主构建器；独立challenge页复用同一证据，
**不能另外当独立holdout成绩**。

最新 `probe_output/mapping_audit_20260905/filtered_v12/` 有12,896个有效候选。
相对filtered_v8，删除32个正候选：R-E5 12、R-E3 8、U5 12；这只是机器差分，
不代表已测得标注精度。v10从582条既有review路线的原始meta重算，v12复用该版本
动作候选并执行最新语义过滤与采样；仍是审核子集，不是全量数据。

| split | 有效情境样本 | wrong-RS invalid | same-RS invalid | same-RS覆盖 |
|---|---:|---:|---:|---|
| train | 10×32 | 54 | 10 | 七个UE、R-E2、R-E3 |
| val | 10×32 | 56 | 8 | 七个UE |
| test | 10×32 | 56 | 8 | 七个UE、R-E3 |

总计1152题。invalid/valid仍为20%；十个asked context均有invalid。
构建与运行时共用一个source/联合签名配额实现；同RS已具备的asked-context覆盖
必须保留，数量在不破坏配额的前提下争取invalid的25%，容量不足报告实际比例。
不为凑数把不确定图像改成负例。R-E5的同RS反例尚无明确证据，继续由清晰wrong-RS
负例监督；R-E2/R-E3同RS的跨split覆盖尚不完整，已在report列出，不能据当前小集
声称它们泛化已验证。

R-E2有效行中，绕障历史/通用导航分别为train 19/13、val 19/13、test 17/15。
这些是实际覆盖，不是硬编码子型配额。所有事件有效桶严格1:1。

## 人工证据与验证

本轮逐张查看15条路线的84张原始RGB，记录在
`boundary_resolution_rgb_notes_20260905.jsonl`；其中有4条路线此前未在累计notes
出现。加上历史记录，当前274条notes、248条不同路线。没有把重复路线或缩略图
当新增全程确认，也没有将未看的帧写成已看。

61项回归通过；train/eval实际入口对三个split各20个seed检查通过，保持十桶1:1、
invalid来源/联合签名配额及已有同RS题覆盖；旧mapping索引被拒绝。结果保存在
`runtime_validation_v12.json`。未加载模型、未训练。

`review.html` 已更新：逐帧看原图、当前prompt、速度/车道证据，并直接显示
`invalid_reason`。`same_rs_challenge_20260905/` 导出全部34道人工负例便于检查。

当前可以继续全量生成**待审核候选**，但不应把1152题审核索引当正式全量训练集。
本轮已处理的错标通过代码排除；未完成全量人工确认和LoRA实际评测，不能承诺
“完美标定”或已实现可靠闭环驾驶。
