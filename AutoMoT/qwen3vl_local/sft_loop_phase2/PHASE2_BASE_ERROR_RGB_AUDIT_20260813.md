# Phase2 Base 错例 RGB 审计（2026-08-13）

## 范围与证据边界

本轮只使用 base Qwen 的 production 评估
`base_rs_four_binary_final_4rgb/20260813_222916`。每个 case 读取的都是模型实际输入的四张
三视角拼接 RGB，路径保存在相应 `error_cases/<task>/case_*/rgb/` 与 `case.json`；同时回查了
`lead_data/<scenario>/<route>/rgb/` 原图、`frame_index.jsonl` 的 RS/Event，以及已有全帧人工审计
`manual_full_sheet_notes_20260809.jsonl`。

既有全帧审计覆盖 42 个 scenario、197 个 scenario/Town 对；该覆盖是本轮逐帧抽查的背景依据，
不等同于本轮重新人工查看全部数据集。下面只记录本轮实际逐帧查看过的代表性错误序列，避免把未看过的
帧写成视觉结论。

## 已查看的代表性序列

| 主任务 | case | 四帧可见事实 | 结论 |
| --- | --- | --- | --- |
| RS1 | `ParkedObstacle/Town05/...001869.../f60` | 夜雨下仍能看到连续同向多车道、路缘/隔离边界和前车；没有岔口或对向共享证据。 | GT R1 合理；模型把低可见度误当成四项都 NO。 |
| RS1 | `DynamicObjectCrossing/Town05/...Scenario3_10.../f88` | 雾雨中的连续城市/地面道路与同向车辆；没有可见高速匝道或局部路口控制。 | GT R1 合理；应以普通地面道路作兜底。 |
| RS1 | `AccidentTwoWays/Town15/...001465.../f42` | 黄中心线、对向车流/占道车辆形成不分隔的双向走廊。 | GT R2 有可见支持；模型错把 R2 当 R1。 |
| RS1 | `PedestrianCrossing/Town13/...65_0.../f100` | 夜间多车道城市道路、路缘连续，当前无路口/对向借道证据。 | GT R1 合理；不能因夜色输出全 NO。 |
| RS2 | `AccidentTwoWays/Town15/...001465.../f18`、`ParkedObstacleTwoWays/Town*/.../f115`、`HazardAtSideLaneTwoWays/Town*/.../f56` | 可见中心线、无物理中央隔离、直接相关的对向车辆；后两类还存在路侧车辆/障碍压缩有效空间。 | 应明确：R2 不要求当前帧已经完成借对向车道动作，窄且未分隔的双向冲突本身即可成立。 |
| RS2 | `ConstructionObstacleTwoWays/.../f121`、`VehicleOpensDoorTwoWays/.../f122`、`InvadingTurn/.../f71` | 黑暗/雾或宽道路，当前四帧没有足够清楚的对向约束。原标注还带缺少 XODR/完整拓扑确认、或高投影误差提示。 | 不能用提示词要求模型从锥桶、中心线或场景名臆测 R2；属于视觉标签风险候选。 |
| RS4 | `PedestrianCrossing/.../f16`、`ControlLoss/.../f57`、`RedLightWithoutLeadVehicle/.../f255` | 交通灯头/灯杆或悬臂与本地停止线、交叉口/转向空间可同时看到；雨雾时灯具小且常处于上方/远侧。 | GT R4 有可见支持；提示词要显式要求扫描上方与两侧物理灯具，而非只找红绿像素。 |
| RS4 | `VehicleTurningRoute/.../f6`、`PriorityAtJunction/.../f99`、`SignalizedJunctionLeftTurn/.../f1` | 当前四帧为普通道路、早期边界或灯具/路口均不可辨。 | 当前四帧无法从 RGB 支持 R4；记录为风险候选，而非把远处灯光猜成 R4。 |
| RS5 | `HardBreakRoute/Town12/...1902.../f250`、`NonSignalizedJunctionLeftTurnEnterFlow/Town13/...003150.../f55` | 清楚的 STOP 标志、停止线和横向道路/路口开口；后者即使有雾，STOP 与交叉口几何仍可见。 | R5 正例的必要视觉锚点明确。 |
| RS5 | `VehicleTurningRoutePedestrian/Town12/...3448.../f23`、`VehicleTurningRoute/.../f15`、`NonSignalizedJunctionLeftTurn/.../f9`、`BlockedIntersection/.../f30`、`InterurbanActorFlow/.../f89` | 分别是直行围栏路、夜雨道路或几乎全黑的道路；在给模型的四帧中没有局部十字/T 口、STOP/YIELD、横向道路或优先冲突可确认。 | 不应用“车辆要转弯”或“看不见信号”强推 R5；这些是视觉标签风险候选。 |

`Town*` 表示此处不以场景名泛化结论；精确 route/frame 已保留在相应 eval 的 `case.json`。所有结论均只针对列出的
四帧时间窗，不外推到同名 scenario 的所有 route。

## 由证据产生的提示词修订

1. 四行输出本质是一个互斥 RS 决策的编码。先查 RS4、RS5、RS2，再以 RS1 作为可见普通地面道路的兜底，避免把四个问题孤立判断后退化为全 NO。
2. 全 NO 必须有正向的高速/匝道/merge/gore/高架连接结构证据。夜雨、雾、黑暗、直道、空路或遮挡不是全 NO 的理由。
3. RS2 需要未分隔的双向走廊与直接相关对向交通/有效通行空间受压；不要求当前帧已经正在借对向车道，也不能只凭双黄线、锥桶或一辆路边车判断。
4. RS4 需“物理灯具/灯杆/悬臂 + 本地交叉口线索”，并显式扫描上方、远侧和侧视图。RS5 的 STOP/YIELD 在可见横向道路/路口开口时是强正证据；普通 driveway、弯道和“未来可能转弯”不是。

## 标签风险处理

原始 RS 标签不在本轮被改写。现有 `visual_label_risk` 已会排除标注中明确要求 RGB/XODR/信号上下文复核的帧；但本轮还发现少量“标签时间边界早于当前四帧可见路口”的案例，原有规则尚未覆盖。

它们不应靠扩大提示词来硬拟合。下一轮若要重构数据索引，应先把这些 exact route/frame 窗口汇总为轻量 override 清单，再作为 `visual_label_risk` 排除；当前先保留为审计记录，避免用少量案例草率改变大规模训练集。
