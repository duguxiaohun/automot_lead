# 第八轮审计修复：不确定回答与可见输入冲突

本轮修复两项已复现问题，未修改默认人工标注或提示词。任务合同升级为
`phase4_state_pair_binary_v9`，运行快照升级为 `phase4_loop_v7`；须新数据目录、新 run。
旧快照/数据/adapter 不自动改合同。标注 v4、提示词 v5、标定规则 v5 保持。
Phase3 v23_io1 和 Action 稳定默认未改。

## 同帧独立停车判断

原先 UE4/YIELD 的 `hold=YES, proceed=UNKNOWN` 会提前返回，STABLE/FOLLOW 输出空动作，
APPROACH 仅输出 DECELERATE。现在有效上下文下，明确的 hold 独立推进纵向 HOLD，
实例仍为 UNKNOWN，单实例、汇总与先验导出均为 STOP。缺答与 UNKNOWN 使用相同规则。

不确定分支仅接纳明确的收紧约束（hold/restrict/renewed_restriction/re_yield），
不授予起步、变道、返回、恢复或完成许可；已成立 HOLD 也不会被 re_yield 降级。
历史记录 `partial_restriction`、实际接纳边及未解决问题，不产生执行许可或虚构执行回执。
超过不确定次数阈值进入 RECHECK，仍保留已成立 STOP。

INVALID/无效上下文/外部故障仍进入复核。明确收紧与明确推进相互矛盾时优先复核，
不会因为第三题 UNKNOWN 而漏掉矛盾，也不把矛盾中的新 hold 当成已确认事实；原有停车约束保留。

## 模型可见输入一致性

`input_identity.py` 与训练共用 `prompts.messages`，用下列内容生成输入身份：

- 按输入顺序排列的、RGB 解码后像素及尺寸摘要；编码文件的元信息不参与可见身份。
- 实际 system 消息和状态对文字，并核验原 `prompt_sha256`。

隐藏的 instance ID、证据 ID、路线 ID、速度或绝对帧号不能区分模型看起来完全相同的问题。
构建器在创建输出目录之前跨全部 split 核对 YES/NO；加载器再次核对身份与答案，
即使人为更新 JSONL 文件哈希，也不能直接放过相反答案。UNKNOWN/INVALID 待审题不是相反监督。
同答案的重复输入保留原记录并报告数量，不暗中删题或改变采样权重。
`load_images` 校验解码内容，防止记录的像素身份与实际图片不一致。

此身份检查针对相同解码 RGB 和消息。它不证明不同图片经过处理器缩放/归一化后是否碰巧
变成相同张量，也不代替逐帧语义审核。相同输入的不同参与者问题若需要相反答案，必须重新
审核可见的指代和冲突范围，不能只改隐藏 ID 或自动改标签。

## RGB 复看与曝光隔离

审计来源：`/tmp/phase4_eighth_audit_20260930/REPORT.md`。
本轮实际复看该批三个既有连续面板，共 71 张既有 RGB：

| 序列 | 复看帧 | 本轮可支持的观察 |
| --- | --- | --- |
| NonSignalizedJunctionLeftTurnEnterFlow / Town04_Rep0_route_001046_route0_01_11_01_51_57 | 24–70 | 自车开始左转后，44–49 附近又有黑车经过近处冲突区域；首次许可不能充当整个转弯持续无冲突的证明。 |
| VehicleTurningRoute / Town04_Rep0_Town04_Scenario4_23_route0_01_09_00_34_45 | 48–71 | 骑行者在右侧草地仍可见；目标可见与占用自车通道是两个判断。雾中远处条件仍不明确，不据此批量标 YES。 |

这是部分复看，新增完整序列数为零；没有新增转移窗口或默认标签。
第八轮报告的新曝光物理组 `NonSignalizedJunctionLeftTurnEnterFlow/Town04_route_001046`
已通过 `eighth_audit_exposure_20260930.json` 登记 train-only，来源文件 SHA 随合同绑定。

现有规则继续用转移专属冲突范围区分几何完成与独立纵向限制；再次侵入需重新让行，
补问到新冲突、实例边界或标定异常截止。没有恢复固定 ±0.5 秒窗口，没有增加状态。

## 验证结果与边界

- Phase4 全部 256 项测试通过，含新增 26 项；已有真实小型 Qwen3.5 的答案 mask 与 LoRA 反向检查通过。
- 2RGB/4RGB 的 `run_full_pipeline.sh` 实际 `MODE=check` 入口通过。
- 两模式各 453 道题全部 RGB 来源与解码内容检查通过；每模式另核验 7 轮 × world1/4 共 14 个采样计划。未运行实际多卡训练。
- 两模式均为 453 个不同可见输入、零答案冲突、零同答案重复；190 YES / 263 NO，133 条待审。
- 与本轮前 v8 产物逐行比较，除新增 `image_rgb_sha256` 和 `model_input_sha256` 两项身份字段外，全部监督与待审字段一致。未改标注或扩大窗口。
- 独立 val/test 仍为空，389 个转移支持格缺额（train119/val135/test135，其中公共纵向270）仍在，`ready=false`。

验证产物：`probe_output/eighth_audit_fixes_v9/validation.json`、两模式 pipeline 日志与数据目录；
测试日志 `/tmp/phase4_eighth_fix_tests.log`。复现作为合成回归测试保存于 `tests/test_eighth_audit_fixes.py`，不进入训练标注。

全覆盖进度沿用第八轮报告：累计 259 条完整序列/29,698 帧，107/278 格至少三个 ID，
仍剩 441 条选定序列，另有 18 格源数据不足三个 ID。本轮复看不增加这些计数。
全场景、全 Town、每格至少三个 ID 的完整审计尚未完成；没有正式 4B/GPU/CARLA 效果验收，不建议正式开训。
