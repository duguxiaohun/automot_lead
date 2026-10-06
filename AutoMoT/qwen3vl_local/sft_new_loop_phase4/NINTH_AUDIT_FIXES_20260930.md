# 第九轮审计修复：小预算覆盖与 RE3 逐段接续

修复小预算永久漏选，新增 RE3 导航分段接续、分段回执、提示词和覆盖检查。
任务合同 v10，运行快照 v8，提示词 v6，采样策略 `transition_single_ring_v2`。
需要新建数据与 run；旧数据/adapter/快照不得改哈希强行沿用。人工标注仍 v4，标定规则仍 v5。
Phase3 v23_io1 和 Action 稳定默认未改。当前仍不能正式开训。

## 采样

旧实现分别旋转路线内列表、合并列表和分组列表，单路线四题退化为步长二。
现先按固定种子构造路线/分组交错的完整题目队列，再仅用 `epoch % 题数` 轮转一次，
随后应用共享物理帧 cap、预算截断和 world size 整除要求。每题每轮最多一次，容量不足仍拒绝。

在题池、种子固定且预算可行并大于零时，每道题在 N 轮内至少一次位于队首，因此即使共享帧
cap 与预算截断同时存在，也不会永久漏选；N 是题数。此界限不承诺任意小预算七轮全覆盖，
不承诺小预算每轮事件等量。无需隐藏游标，给定 epoch 可重建同一顺序，输入行顺序不影响题 ID 顺序。
策略改变了后续轮次的呈现顺序，不能冒称旧采样轨迹完全等价。

原复现四题、budget=1、十二轮：四题各三次。另覆盖偶数池、多路线、多分组、共享帧 cap1、
world4、默认整池与小预算；真实默认 453 题在 budget1 下 453 轮全部覆盖。

## RE3 多段路线合同

导航在建立同一 RE3 实例时提供 `route_segments`，每段为一个已确认的相邻走廊转换：

```json
[
  {"segment_id":"middle", "source_corridor":"left lane", "target_corridor":"middle lane", "direction":"RIGHT", "adjacent":true},
  {"segment_id":"right", "source_corridor":"middle lane", "target_corridor":"right lane", "direction":"RIGHT", "adjacent":true}
]
```

`segment_index` 默认 0，Episode 的 `direction/target_corridor` 必须匹配活动段。
拒绝断链、重复段 ID、非相邻声明、越界索引、活动目标不符和非 RE3 使用此合同。
支持 LEFT/RIGHT/FORWARD；FORWARD 不生成横向变道先验。

`adjacent=true` 是已验证导航对拓扑的声明，代码检查链条结构，不能替代地图验证。
本轮没有实现从 RGB 自动推断完整路线或实际 CARLA 导航适配器。不要把跨两条车道的最终出口
伪称一个相邻目标，也不要把后续独立路线事件无限串入同一实例。

每段复用 WAIT→CROSS→几何完成，不增加新的事件状态。中间段 complete=YES 后选择下一目标、
回到 WAIT；本次完成仅对当前段有效，下一段 enter 要到下一观察重新构题。中间段完成不会
把实例标 DONE，也不会重跑 Phase1/2。HOLD/APPROACH 等纵向事实原样保留。
若同帧有待执行 release，先完成其确认或撤销，再接续，不把未执行许可带入下一段。
只有末段几何完成且纵向稳定，实例才真正退出。

分段实例执行回执额外要求 `segment_id`，同时仍绑定实例、当前决定帧、边集合、因果观察与来源。
`loop.acknowledge(instance_id, frame, segment_id=...)` 也需指定活动段；旧段/缺段回执拒绝，
公开 tick 异常仍整批回滚。分段实例禁用无段标识的 `execution_committed=True` 便捷入口。
历史 `segment_id` 表示当时接受问题所属段，`route_continuation` 记录该段完成后选择的下一段；
动作先验实例详情与快照保存活动段。模型 YES 不等于执行证据。

文字仍只有当前状态和候选下一状态、回答 YES/NO；新增“仅当前相邻段”的语义，
只呈现活动段目标与方向，不呈现后续路线、段 ID、序号或全流程。非分段题的实际文字保持原样。
数据构建和序列推理允许分段字段；相同可见输入的答案冲突检查仍生效，隐藏未来路线不能区分问题。
回放的逐帧真值也需 `segment_id`；缺失/不匹配计未知，跨段的未结束真值区间删失为
`route_segment_changed`，不能串用前一段的延迟起点。

可运行的合成 demo（从 AutoMoT 目录执行）：

```bash
python -m qwen3vl_local.sft_new_loop_phase4.demo --segmented-exit
```

动作输出为右变道→STOP→STOP→右变道→UNCOND；含中间等待、新许可、逐段执行回执。
这是控制器示例，不是模型预测或新增 RGB 标注。

## 证据、标注与覆盖

本轮读取第九轮报告和逐序列记录，并实际复看 107/109 各 f96–143 的四张连续面板，共96张
既有 RGB；新增完整序列0，新增标注0。107可见右侧车辆先后经过、两次横移间仍有车流；
109夜雨中的侧后间隙不能可靠确认，不批准精确YES帧。仅凭首次横移开始不能授权后续所有横移。
此次没有扩窗；各段 readiness/catchup 仍须独立逐帧标定，旧段补问须在下一段/新冲突处截止。
面板来源与 SHA 在 `probe_output/ninth_audit_fixes_v10/local_visual_review.json`。

第九轮新曝光的 `PedestrianCrossing/Town13_1736_0` 和 `PedestrianCrossing/Town13_1757_0`
已登记 train-only，`ninth_audit_exposure_20260930.json` 保留原报告来源摘要并绑定合同。

默认2/4RGB仍各453题（YES190/NO263，readiness339/catchup114）、133待审，val/test为空。
与v9逐行比较，除随提示词版本更新的 `model_input_sha256` 外，全部字段一致。
原389个转移支持格缺额保持，新增独立 `missing_segmented_route_support`：三split×
中间/末段×enter/complete，共12项正负readiness检查，目前全缺；普通单目标或catchup不能补齐。
这些是训练证据缺额，不是新发现401个代码错误。

完整审计进度沿用用户第九轮报告：283条完整序列、32,537帧，120/278格达到三个ID；
仍剩417条所选序列、66,131帧，18格来源不足三ID。本次复看不增加上述计数，全覆盖尚未完成。

## 验证

- 284项Phase4测试通过，含28项新增回归及既有真实小型Qwen3.5答案mask/LoRA反向检查。
- 2RGB/4RGB实际 `run_full_pipeline.sh MODE=check` 均通过，各453题RGB来源与解码内容核验通过。
- 两模式各7轮×world1/4共14个整池采样计划通过；另各453轮budget1全题覆盖。
- 默认输入无相反答案，标注文件未改；分段测试的合成事实仅用于测试，不写入默认训练库。
- 两模式preflight仍ready=false，本机默认 `checkpoints/Qwen3.5-4B` 目录不存在；未正式4B/GPU/DDP/CARLA验收。

本轮产物：`probe_output/ninth_audit_fixes_v10/validation.json`、两模式数据/pipeline日志；
测试日志 `/tmp/phase4_ninth_fixes_tests.log`。测试通过不代表多段驾驶效果或完整数据覆盖已验收。
