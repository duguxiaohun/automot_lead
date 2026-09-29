# Phase4 第四轮部分审计修复

依据 `/tmp/phase4_fourth_audit_20260930/AUDIT.md` 修复旧格式catchup入口和回放真值缺口问题。原审计文件、原标注、曝光登记及提示词未改。本轮没有新增RGB目视审阅，不增加序列/帧数，也不批准新转移窗口。

## 旧格式catchup

`per_frame_conditions`仅保留readiness入口；旧 `slice=catchup` 在读取RGB、写出数据集之前明确拒绝，错误提示要求迁移为带逐帧条件及冲突作用范围的 `reviewed_transition_band`。直接调用旧 `catchup_label` 同样拒绝，避免另一个入口继续根据“曾经完成＋在时间区间内”生成YES。非readiness的旧slice也不被当作readiness悄悄接受。

迁移需要重新审核每个当前帧，填入reference、stop、phase、当帧facts、后继确认、当前冲突及其作用范围；不能从旧completed_at/valid_until自动补造这些字段。新格式仍沿用逐帧校验：当前明确条件不成立时，即使后继曾经确认也为NO；几何完成事实为真且停车为独立纵向限制时，可为YES并保留HOLD。旧格式中原本有一条catchup，因此整份历史 `reviewed_intervals_20260929.json` 不再是可直接构建的新训练输入，历史证据原文件保留。

合成反例在正常条件函数中为NO，新构建器明确拒绝，不再写出四题YES。2/4图回归也通过实际构建入口验证：已逐帧审核的catchup中加入明确当前阻挡，会生成NO。这些仅为测试中的人为构造，不声称实际RGB存在该阻挡，不加入生产标注。

## 连续成立区间与延迟

回放的精确延迟仅在**逐帧都有审核YES的连续区间**内起算。未知/缺失真值关闭当前可证明的连续区间并记录 `truth_unknown` 删失；后续YES另起统计，不把缺口隐含为持续成立。每个删失记录同时保留ready_since、last_ready_frame、已观察到的answer_frame/accepted_frame和结束原因，end_frame是发现中断的帧，不表示条件在该帧被证明关闭。

整帧缺失记录 `observation_gap`；状态改变后不再询问某条边，记录 `transition_not_queried`。明确NO仍记录 `condition_closed`；序列结束/复核仍保留未确认区间。未知帧上的执行确认仍出现在steps，但没有足够真值时不写入精确延迟数组。缺口前已观测到的回答/接受延迟不会被删除，未确认执行则单独删失。

| 输入与外部执行 | 回答/接受/执行延迟 | 旧区间 |
|---|---|---|
| f10 YES，f11未知，f12 YES并执行 | 各[0] | ready_since=10、last_ready_frame=10，f11以truth_unknown删失 |
| f10 YES，f11 NO，f12 YES并执行 | 各[0] | f11以condition_closed删失 |
| f10–12连续YES，f12才回答YES并执行 | 各[2] | 无未确认区间 |
| 只有f10和f12两个观察，f12执行 | 各[0] | observation_gap删失，不能推定f11持续成立 |

输出标记 `delay_policy=phase4_contiguous_reviewed_readiness_v1`。这些是从连续审核区间首个可见YES起算的延迟；新起点不代表已经证明真实机会恰好在该帧开启。旧跨缺口统计不能与新指标直接混比。此修改不改变在线控制器的状态推进或动作先验。

## 验证、版本及未完成项

222项Phase4测试通过，包括旧helper/构建器拒绝、新逐帧构建中的条件优先、缺失/UNKNOWN/INVALID真值、连续YES/明确NO对照、整帧缺失、未知帧执行、区间末尾删失和不再询问的边；原HOLD、事务及几何完成修复仍通过。数据回环测试改用当前默认v3标注，不再把历史catchup入口视为合法新训练输入。

2/4RGB均重新构建并加载429道训练开发题、121条待审，三split题目和待审队列均与v6逐行一致；每种模式全部429题的因果RGB加载/哈希核验及各7轮×world1/4采样回放通过。产物、修后反例和来源审计SHA保存在 `probe_output/fourth_audit_fixes_v7/`，未覆盖原只读报告。

任务合同v7、标定代码v5、train.sh默认data_v7；提示词仍v5、人工标注仍v3、运行快照仍v6、先验source仍v2。本次运行时源码未改，快照结构不变。数据及adapter仍绑定完整源码合同，需要新构建产物、新run；旧模型/数据必须使用原合同版本，不能硬改hash复用。

ready=false。此前完整审计报告累计224条完整序列、26,418帧，92/278个单元达到三个ID，仍剩476条计划序列，18个单元来源不足三个ID；本轮不增加上述计数。全事件/场景/Town审核及纵向边、分支、独立val/test标定仍未完成，完整4B权重缺失，无正式4B/GPU/DDP/CARLA验收或模型效果提升结论。Phase3/Action稳定默认未改。
