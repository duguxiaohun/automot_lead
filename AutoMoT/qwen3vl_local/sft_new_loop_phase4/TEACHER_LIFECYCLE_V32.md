# Phase4 v32：截止过期等待实例并重建弱监督

## 本次修复

离线老师只有先给 YES 才能接收执行回执，因而可能错过专家实际起步，旧 UE1 长期留在 YIELD/HOLD，并给后续骑车人导致的再次停车贴错误标签。现在独立记录实例内可观察的停车历史；之后连续五帧前向运动、累计前向位移至少 2 米时，截止仍在 YIELD 的旧实例。每段位移都在当时车体坐标中检验，兼容缓弯；倒退、侧移、位姿跳变及缺帧不能触发。截止绑定过去/当前源文件 SHA，记为 `observed_resumption_censors_stale_wait`，不是 complete，也不产生 YES。运行时控制器的许可及执行回执约束未改。

前车连续三次可见且整个框超过 30 米实例创建范围时，同样截止旧 UE1。新实例必须重新满足限制证据，不能从普通跟车直接创建首帧 YES。

预检新增 `stationary_readiness_support`：逐事件 × proceed/depart/enter/return/release 报告静止 readiness YES、物理路线数、人工来源、运动 YES、未知运动及 catchup YES。默认参考下限为 1，缺额只报警，不新增试训硬门槛；运动或 catchup 样本不填补静止层。

UE4 `cyclist_follow` 分支使用单独状态文本：骑车人可以仍在前方，允许的是恢复跟随；这不授权超车或驶入对向车道。横穿分支、状态机、原人工标注、采样权重及重复上限未改。可见角色销毁仍保留异常，不能伪造 UE4 complete。

## 对审计结论的两处更正

旧库五条 VehicleTurningRoute 的静止 UE1 proceed YES 实际为 19 道（5+1+7+3+3），不是 20 道；其余 HardBreakRoute 17 道、noScenarios 1 道，共 18 道。对照按旧题的物理路线及帧号逐项核验，不能只比较总数。

BlockedIntersection/Town06 的 18/37/38/39 并非全程静止：全帧 metas 显示最高速度 7.2–11.6 m/s、距起点最大水平位移 56–82 米，末帧速度分别约 10.7/4.9/3.3/3.1 m/s。长时间中段等待不等于整条采集卡死，因此保留四条，不按名字列黑名单。新候选过滤会排除至少 500 帧、完整连续元数据中速度绝对值始终 ≤0.1 m/s 且水平位移始终 ≤0.25 米的整条静止序列；缺位姿/缺帧仅记未知，短等待和最后恢复行驶的路线不会被误剔除。这次全库没有新增符合条件的排除项。逐路线证据见机器报告 `blocked_route_audit`。

## 核验和产物

833 项专项测试通过。全部 8,614 条合格路线、1,062,401 帧已重新回放，耗时 1243.7 秒；训练/验证/测试仍为 7,268/658/688 条，保留原物理路线 split。2/4 图真实 pipeline check、world1/world4 七轮采样与所有 RGB 文件及像素身份检查均通过。

| 模式 | 训练题 | 弱题 | 训练物理路线 | 每轮呈现 | 七轮覆盖 world1 / world4 | RGB 核验题数 |
|---|---:|---:|---:|---:|---:|---:|
| 2 图 | 75,373 | 74,930 | 1,837 | 400 | 705 / 705 | 76,371 |
| 4 图 | 75,373 | 74,930 | 1,837 | 400 | 705 / 705 | 76,371 |

原 443 道人工训练题逐行保持；val273/test186/review539 逐字节保持。合同 v32、老师 v7、独立预约协议 v12，全部使用新产物；v31 数据未覆盖。

| UE1 proceed 弱标签 | v31 | v32 |
|---|---:|---:|
| YES / stationary | 37 | 18 |
| YES / moving | 0 | 0 |
| NO / stationary | 55,359 | 55,432 |
| NO / moving | 5,730 | 4,955 |

19 道已定位错误 YES 均已清除，另外 18 道旧静止正例逐帧保留。821 第167–175帧不再生成旧实例 proceed。上述减少不是独立准确率测量，剩余标签仍属弱监督。运动 UE1 proceed 弱 readiness YES 和弱 restrict 继续为0。

可复现的数值筛查（不等同于人工错标率）：

- `no_after_observed_stop_and_8_moving_frames`：794 → 66 题。
- `no_ego_and_original_lead_above_3_similar_speed_within_1`：424 → 138 题。

静止正例缺额：R-E2/enter, R-E2/release, R-E3/enter, R-E3/release, R-E5/proceed, R-E5/release, U-E1/release, U-E2/depart, U-E2/release, U-E2/return, U-E3/release, U-E4/release, U-E5/proceed, U-E5/release, U-E6/release, U-E7/proceed, U-E7/release。这些缺额只作覆盖警示。

留出老师参考每模式 val 8,345 / test 7,952，只衡量老师一致率。盲审包 961 张卡、56 条路线，全部未填写，严格批准仍为0。八类事件老师仍缺，独立准确率未测；每轮 400 次的瓶颈保持，不声称大题池已充分训练。弱监督试训入口 `trainable=true`；正式 `formal_data_ready=false`。本轮没有启动 GPU 训练。

实际目视仅复核两条训练路线十个关键帧的两张联系表，未新增完整逐帧序列，未查看独立/验证/测试 RGB。用户具名的七条 RGB 路线及821数值审阅路线共八组显式登记 train-only；两条行人消失按用户报告和自动边界记录风险窗口，未将未看原图的边界升级为本轮人工确认。

完整产物：`/home/codon/automot_lead/AutoMoT/checkpoints/phase4_v32_full`，包括 `data2`/`data4`、`full_production`、`references2`/`references4`、`preflight2.json`/`preflight4.json`、`independent_review2` 以及绑定源码和资产的 `frozen_source`。详细计数见 [thirty_second_audit_verification_20261005.json](thirty_second_audit_verification_20261005.json)。
