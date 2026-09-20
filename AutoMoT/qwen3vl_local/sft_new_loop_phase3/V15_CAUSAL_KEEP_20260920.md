# Phase3 v15：场景事实、因果动作与显式 KEEP

2026-09-20。修改覆盖 UE1–UE7、RE2/RE3/RE5 共十类，默认四图单选。
这是基于已有 RGB 的开发复核和标签/提示词实现，不是新模型效果结论，也不是全数据集人工逐帧审计。

## 标注如何变化

旧默认 choice 的 NONE 表示当前问题域没有满足阈值的主要变化动作；binary 的 NO 只是每个动作命题的否定。
两者均不应解释为“看不清”或事件不存在，但 NONE 没有向模型正面说明车辆正在怎样行进。
新 choice 用 **KEEP** 明确表示继续当前行进阶段，允许低于变化阈值的小幅调速。

| 题域 | KEEP 的正向含义 | 必须满足的离线证据 |
| --- | --- | --- |
| UE2/UE4/RE2/RE3 机动域 | 沿当前车道继续，包括已经变道后继续绕行 | 2秒速度窗无合格阶段变化、3秒内无新的首次跨线、速度与横向观测完整 |
| UE1/UE3/UE5/UE6/UE7/RE5 纵向域 | 维持当前速度阶段 | 2秒速度观测完整、无合格阶段变化；不推断横向行为 |

当前持续近停仍是 STOP；缺帧、非有效速度、含糊的增速再制动窗、机动域横向身份不完整、无效场景都不能补 KEEP。
保持没有要求严格恒速：例如 UE5 #513 的降幅为约1.827m/s，略低于当时约1.8334m/s的既有阈值，随后回升，属于速度阶段保持。
该窗同时存在横向变道，所以**纵向 KEEP 不能声称车道保持**。

保留 v8 的时间窗口/阈值和 STOP > 首次未来跨线 > 速度动作优先级。
`choice_semantics.py` 在完整证据上标注 `primary_action`、`keep_scope`、`primary_action_evidence_status`，再核对原始速度/横向判据；
候选与最终索引都写入这些字段。原始五个布尔动作位保留作轨迹证据与 binary 诊断，默认模型 target 是动作名称或 KEEP。
解析器不再接受 NONE；KEEP 有独立 support/precision/recall/F1 和选优门槛。
审计拼图也按新合同显示 KEEP，解析失败显示 UNPARSED，不冒充保持；已有历史审计文件不重写。

## 因果放在哪里

`[SCENE_CONTEXT]` 只含 RS、事件和已提供的历史事实。UE2 的事件描述覆盖仍在进行的绕行，避免把已经跨线后的每一帧都描述成当前车道仍被堵住。
各动作选项各用一至两句连续的因果说明，所有候选一起显示且按 case seed 乱序；不会根据本帧答案挑解释、补意图或透露未来轨迹。

| 场景 | 减速/停车的目的 | 恢复/保持及横向阶段的区别 |
| --- | --- | --- |
| UE1 前车急刹 | 防追尾、保留跟车空间、等前车释放 | 前车拉开可增速；已经恢复稳定跟随可保持 |
| UE2 静态阻挡 | 避碰，同时观察邻车、来车和间隙，为可能绕行创造时机 | 可向左或右绕行/归位；已完成跨线后可保持，事件不必结束 |
| UE3 切入 | 为切入车辆让空间，避免逼近其行驶路径 | 切入发展、已建立跟随和可用空间打开对应不同速度阶段 |
| UE4 行人/骑车人 | 降低碰撞风险、保留侧向距离、等其通过 | 区分同向骑行、路径占用/释放、车道内通过与真正跨线 |
| UE5 对向侵入 | 减小接近速度、给对方通过空间、必要时等待 | 通行空间开始打开即可恢复，不强求对方完全离开画面 |
| UE6 路口违规冲突 | 即使有优先权也避免交叉碰撞、调整到达时机 | 交通动态与通行空间决定阶段，优先权本身不证明安全 |
| UE7 灯故障 | 独立观察各方向交通、判断交叉空隙、必要时让行 | 一个绿灯不证明故障解除，故障存在不等于每帧都停车 |
| RE2 目标车道/绕行回归 | 保留障碍物距离、匹配目标车道空隙、必要时等车流 | pending 不等于即将跨线；可能仍需先左借道，不能固定右归位 |
| RE3 合流/驶出 | 调速匹配间隙与车流，拥堵/无入口时等待 | 平行邻车不等于挡路；速度调整与首个真实跨线分开 |
| RE5 无灯路口 | 满足STOP/让行优先权，观察交叉车流及进入时机 | 区分接近、持续等待、起步、通过；过了STOP牌不能重复要求停车 |

这些是场景条件下的**可能目的**，不是人工标定了每一帧驾驶员的真实心理动机；模型仍预测采集行为。
“减速寻找空隙”不证明下一动作一定变道，“变道需要可用间隙”也不能反推所有邻车均不存在或已验证安全。

## RGB 依据与局部重建

复用 20260916/20260920 的既有17面板逐帧图，先检查异常路线；每例前四帧为实际输入，后13帧仅用于离线标定核验。
本次复核26个片段、442个帧面板，覆盖十类；复核条目、RGB SHA256、原始动作及新主要动作见
[causal_action_rgb_notes_20260920.jsonl](causal_action_rgb_notes_20260920.jsonl)。这些都是已曝光开发路线，继续 train-only。

| 复核例子（日期/编号） | 时序观察 | v15 主要动作 |
| --- | --- | --- |
| 0916/254 UE2 Accident f74 | 输入末帧已从lane3到lane2；之后保持新车道绕行，速度仅小幅/孤立起伏 | KEEP |
| 0920/314 RE2 ParkedObstacle f44 | 输入也已跨线，但随后0.75秒明显减速 | DECELERATE，不能所有“已跨线”都标KEEP |
| 0916/448 UE2 开门场景 f169 | 先减速等对向蓝车通过，随后借对向车道左跨；短暂近0未持续 | LANE_CHANGE_LEFT |
| 0916/23 UE2 Accident f71 | 左侧事故车，随后向右跨线且加速 | LANE_CHANGE_RIGHT |
| 0920/585 UE2 Accident f82 | 邻车从右驶入前方，自车仍等待到f89 | STOP，不把别人变道当作自车变道 |
| 0920/513 UE5 f64 | 速度变化未到阈值，同时有横向变化 | KEEP，仅纵向 |
| 0920/439 RE2 f72 | 施工仍在前方，自车首次从lane1左借到-1 | LANE_CHANGE_LEFT，pending恢复不是RIGHT证据 |
| 0920/556 RE3 f124 | 左侧有车，随后实际左跨；减速未到阈值 | LANE_CHANGE_LEFT |
| 0920/942 RE3 f67 | 前两秒保持车道和速度阶段，明显减速在窗口之后 | KEEP |
| 0920/860 RE5 f151 | 当前0.014m/s，但下一采样已0.604，之后持续起步 | RESUME，单帧近0不是持续STOP |
| 0920/836 UE7 f29 | 故障灯路口有横向车辆，自车继续当前速度阶段 | KEEP，亮绿灯不否定故障 |

其余 UE1/UE3/UE4/UE6 的逐帧观察与对应目的也记录在同一 JSONL。
26条路线的原始 meta 与原始标注经生产构建器产生 **1183候选**：STOP 558、RESUME 101、KEEP 222、DECELERATE 154、LEFT 89、RIGHT 59。
每类取32行再加64 invalid 得到 **384行局部索引**；320有效行全部通过标签重算、target/解析往返及scene字段检查。
这是训练数据生成路径的局部冒烟，不是全量生产索引，也没有生成重复 RGB 文件。

重放已审证据（在 AutoMoT 下）：

```bash
python qwen3vl_local/sft_new_loop_phase3/audit_causal_keep.py \
  --index qwen3vl_local/sft_new_loop_phase3/probe_output/causal_keep_20260920/index/frame_index.jsonl \
  --output qwen3vl_local/sft_new_loop_phase3/probe_output/causal_keep_20260920/verification.json
```

## 实际 UE2 提示词

以下由当前代码生成，R1、速度6m/s、导航目标(40,-2)、四图输入；候选顺序是该示例seed的结果。
这不是某个真实帧的答案演示。文字长度542个英文空白词，包含system与导航，未计图像token；不是Qwen tokenizer计数。
十类相同配置的完整文字长度为368–542词。真实历史和并发事件会增加长度。

System:

```text
Predict the recorded ego vehicle's next action. Output one listed high-level action only.
```

User（四图之外的文字）：

```text
RGB: four-frame history at t-0.75 s, t-0.50 s, t-0.25 s and t=0 (missing early history repeats frame 0). Each image is left/front/right stitched views.
Predict actual driving, not recommended driving. Only past RGB and current state are observed.
Use temporal gaps and lane boundaries; do not invent hidden actors. Predict from now, not completed history.

[SCENE_CONTEXT]
Proposed road: an ordinary same-direction road.
Situation: a static obstacle affects ego's travel corridor, including an ongoing bypass. Parking-bay vehicles outside the path alone are insufficient.
[/SCENE_CONTEXT]
Current speed: 6.000 m/s.
[NAVIGATION_GOAL]
ROUTE_TARGET_XY: (x=+40.0 m, y=-2.0 m). Final destination, not the next lane. Ego coordinates: x forward; y negative LEFT, positive RIGHT. Its sign cannot choose a lane-change side.
[/NAVIGATION_GOAL]

Choose one primary action including KEEP. Speed: next 2 seconds.
STOP: two consecutive 4-Hz samples at or below 0.5 m/s within 1.5 seconds. Include the current sample: still waiting at the next sample counts, even if ego accelerates later. STOP takes priority.
Otherwise, use the FIRST qualifying change from current speed: a drop of at least max(1.2 m/s, 20% of current speed) means DECELERATE; a gain of that size for two consecutive samples means RESUME. An isolated gain is insufficient. A stop beyond 1.5 seconds does not cancel DECELERATE.
For lane options, use only the FIRST ego lane-boundary crossing within 3 seconds; ignore crossings already in the input and later return crossings. A curve, steering, or another vehicle changing lanes is not ego lane change.
Choose qualifying STOP first; otherwise the first listed lane crossing takes priority over its accompanying speed change; otherwise choose the speed action or KEEP. KEEP allows small speed adjustments; it does not mean the event has ended or visibility is poor.

Action meanings and possible purposes (not proof of intent, safe gaps, or the next action):
RESUME: Gain speed as usable forward space opens during or after passing the obstruction; the obstacle need not have disappeared and a previous stop is not required.
KEEP: No new ego lane-boundary crossing within 3 seconds or qualifying speed-stage change within 2 seconds: Continue in the current lane with roughly the current speed, including after a completed bypass lane change while the obstacle event remains active. Small adjustments or passing within the lane do not constitute another lane change.
LANE_CHANGE_RIGHT: Cross the right boundary to use a suitable bypass or recovery gap, judging adjacent traffic and clearance from the obstruction. A rightward crossing can begin a bypass or return from one; the visible phase determines its purpose.
DECELERATE: Slow to avoid the static obstruction while assessing adjacent-lane vehicles, approaching traffic and gaps for a possible bypass. This creates time to find a suitable crossing opportunity; slowing alone does not mean a lane change follows.
STOP: Stop or keep waiting before the obstruction to avoid collision while checking passing clearance and waiting for a usable adjacent-lane or oncoming-traffic gap. Waiting can prepare a bypass without establishing that a crossing is next.
LANE_CHANGE_LEFT: Cross the left boundary to use a suitable bypass or recovery gap, judging approaching vehicles and passing clearance rather than requiring an empty lane. Borrowing an opposing lane also requires attention to oncoming traffic.

Output one listed action name only:
<ACTION_NAME>
```

答案仍只输出一个名称，例如已完成变道且后续窗口保持的那一帧输出 `KEEP`，持续停车那一帧输出 `STOP`。

## 版本和验证边界

- prompt：`sft_new_loop_phase3_high_level_action_v15_causal_actions_keep`。
- 正向主要动作合同：`primary_choice_v2_explicit_domain_keep`；新索引默认 `sft_new_loop_phase3_data_v15`。
- 旧 v14/更早 run 用原源码恢复；不能向旧 adapter 临时换 prompt 或把旧 NONE 当作 v15合法生成。
- 本轮仅升级 Phase3；action_prior 继续用原始布尔 oracle 证据和旧 NONE协议，v15 KEEP文本尚不是其外部预测接口。
- 本机没有 torch，本轮没有运行模型、DDP或GPU训练测试，也不宣称精度提升。CPU合同测试和实际数据冒烟检查见下方结果。

本轮最终验证：**391项CPU测试通过**（Phase3 334项、已有action_prior高层prompt回归57项）；
另排除了依赖torch的三个测试模块及五项测试，没有把它们计作通过。
全Phase3 Python AST解析、四个shell入口语法、`git diff --check`通过；26组输入图像哈希与原审计记录一致。
训练和独立eval的实际索引读取函数另用AST抽离模型初始化执行：验证先查完整布尔证据与版本，再读取答案；
None、字符串NO、整数0不能经bool转换变成伪监督。缺manifest的旧choice行也会被版本检查拦下。
根目录AGENTS/CLAUDE/PROJECT_CONTEXT已同步新版约定。
