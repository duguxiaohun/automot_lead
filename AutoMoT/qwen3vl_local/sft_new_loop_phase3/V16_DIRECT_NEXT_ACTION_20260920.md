# Phase3 v16：直接判断接下来动作

用户要求所有场景直接根据RGB、当前速度和场景判断接下来的动作，不在输入中要求“未来几秒内”完成什么。
本次统一覆盖UE1–UE7、RE2/RE3/RE5、choice/binary及四图/两端点模式。

## 模型输入

- 删除未来1.5/2/3秒时间窗、4Hz连续采样数、0.5m/s近停阈值、max(1.2m/s,20%)变化公式。
- 开头直接说明用图像序列、当前速度和场景判断下一动作；不要把图中已经完成的动作重复算作下一动作。
- STOP表示停车或继续近停等待，DECELERATE表示明显减速，RESUME表示持续增速；KEEP表示继续当前行进阶段，允许小幅调整。
- 保留当前等待优先、首次跨线、已完成跨线不重复、小幅调速与阶段变化不同的语义。
- 各场景动作原因与目的保留：例如UE2减速避碰同时观察邻车/来车与间隙，为可能绕行创造时机；不会把目的当作空隙已安全的证据。
- scene_context仍只给道路、事件和历史事实。默认只输出一个动作名称，不要求模型输出分析或解释。
- 输入图像的`t-0.75/t-0.50/t-0.25/t=0`仅标记过去的观察顺序；它们不是未来动作倒计时。

## 标注与版本

数值判据仍在 `trajectory_action.py` 中执行，**没有修改轨迹窗口、速度阈值、主要动作优先级或KEEP标注条件**。
提示词用自然动作语义引导模型，监督标签仍按固定规则生成；不声称自然语言能唯一确定所有接近阈值的样本。
本轮没有新的RGB人工审计，复用v15已审开发路线；未运行模型或GPU训练，不宣称“下意识”能力或准确率改善。

prompt为 `sft_new_loop_phase3_high_level_action_v16_direct_next_action`，默认索引为 `sft_new_loop_phase3_data_v16`。
更新了Python和shell的新训练/eval默认路径，所有输入模式的prompt哈希都会变化；旧run必须用原源码恢复。
KEEP标注协议仍为 `primary_choice_v2_explicit_domain_keep`，action_prior的旧协议未升级。

40种标准示例（十场景×两种输出×两种RGB）已检查：提示词哈希全部改变、目标全部不变，标定源码指纹不变。
相同四图choice示例含system/导航的英文空白分词从v15的368–542词降为337–511词；不是Qwen token数。

## 当前实际UE2示例

R1、速度6m/s、导航终点(40,-2)，四张历史图像；所有候选同时展示，释义不随答案变化。

System:

```text
Predict the recorded ego vehicle's next action. Output one listed high-level action only.
```

User（另有四张RGB）：

```text
RGB: four-frame history at t-0.75 s, t-0.50 s, t-0.25 s and t=0 (missing early history repeats frame 0). Each image is left/front/right stitched views.
Use the image sequence, current speed and scene to identify what ego does next. Judge visible traffic motion, gaps and lane boundaries; do not invent hidden actors or repeat an action already completed in the images.

[SCENE_CONTEXT]
Proposed road: an ordinary same-direction road.
Situation: a static obstacle affects ego's travel corridor, including an ongoing bypass. Parking-bay vehicles outside the path alone are insufficient.
[/SCENE_CONTEXT]
Current speed: 6.000 m/s.
[NAVIGATION_GOAL]
ROUTE_TARGET_XY: (x=+40.0 m, y=-2.0 m). Final destination, not the next lane. Ego coordinates: x forward; y negative LEFT, positive RIGHT. Its sign cannot choose a lane-change side.
[/NAVIGATION_GOAL]

Choose one primary action including KEEP.
STOP means stopping or continuing to wait at a near-stop; current waiting still counts even if ego moves off later. Otherwise, use the first meaningful speed change: slowing is DECELERATE; a sustained speed increase is RESUME, without requiring a previous stop. Small speed adjustments are continued driving, not a new speed stage.
For lane options, use only the FIRST upcoming ego lane-boundary crossing; ignore crossings already in the input and later return crossings. A curve, steering, or another vehicle changing lanes is not ego lane change.
Choose STOP first; otherwise the first listed lane crossing takes priority over its accompanying speed change; otherwise choose the speed action or KEEP. KEEP allows small speed adjustments; it does not mean the event has ended or visibility is poor.

Action meanings and possible purposes (not proof of intent, safe gaps, or the next action):
STOP: Stop or keep waiting before the obstruction to avoid collision while checking passing clearance and waiting for a usable adjacent-lane or oncoming-traffic gap. Waiting can prepare a bypass without establishing that a crossing is next.
LANE_CHANGE_RIGHT: Cross the right boundary to use a suitable bypass or recovery gap, judging adjacent traffic and clearance from the obstruction. A rightward crossing can begin a bypass or return from one; the visible phase determines its purpose.
KEEP: Continue in the current lane with roughly the current speed, including after a completed bypass lane change while the obstacle event remains active. Small adjustments or passing within the lane do not constitute another lane change. This means continuing without a new ego lane crossing or meaningful speed-stage change.
DECELERATE: Slow to avoid the static obstruction while assessing adjacent-lane vehicles, approaching traffic and gaps for a possible bypass. This creates time to find a suitable crossing opportunity; slowing alone does not mean a lane change follows.
RESUME: Gain speed as usable forward space opens during or after passing the obstruction; the obstacle need not have disappeared and a previous stop is not required.
LANE_CHANGE_LEFT: Cross the left boundary to use a suitable bypass or recovery gap, judging approaching vehicles and passing clearance rather than requiring an empty lane. Borrowing an opposing lane also requires attention to oncoming traffic.

Output one listed action name only:
<ACTION_NAME>
```

标注为已完成跨线后保持行进的样本输出`KEEP`，当前持续等待的样本输出`STOP`，不输出数值时间窗或解释。

## 本轮验证结果

- 378项Phase3 CPU回归通过；依赖torch的三个模块和四项测试未执行。本机未安装torch，未运行模型。
- 从v15已审的26路线原始标注/meta重建局部索引：1183候选和384行均衡索引；逐条比较标签、原始证据、行顺序及split全部不变。
- 384行严格标签验证、320有效行choice target验证、四种prompt合同哈希、Python AST、四个shell语法及diff空白检查通过。
- 旧candidate cache因源码合同变化被正确拒绝，随后从原始标注重建；未绕过合同、未覆盖旧产物。
- 此产物位于probe_output，仅为局部冒烟，不能替代全量生产索引。新训练由pipeline重建v16索引。
