# Phase3/4 开训收敛审计（2026-10-17）

本轮依据用户给出的总方案、Phase3 指南和 Phase4 指南，核对当前 v81 源码与已收到的服务器证据，并完成十事件既有训练路线的本地因果重放和逐帧 RGB 复核。**现有自动规则数据可以继续用于开发训练，无需逐题人工批准。本轮未发现必须先修改生产标定、提示词或训练代码的新确定缺陷。** 这不等于十事件全部能力可靠，也不等于联合 G0 或部署验收通过。

**当前顺序：保留 Phase3 两模式三轮结果；只接续 Phase4 当前目录的 RGB2 test，然后 RGB4 首轮/val/reload/test。** 不重训 RGB2，不重跑已经通过的 reload，不重新建原生库、迁移 view、做 batch 选型或扩大审计前置。方案中的性能研究、全量人工审核、Action/CARLA 是独立后续工作，不作为本次开发训练的附加阻塞。

## 1. 当前完成状态和 1:1

| 项目 | 已有实际证据 | 尚缺的最小证据 |
| --- | --- | --- |
| Phase3 两图/四图三轮基线 | 两模式每轮十事件各1024，三轮各3072；val各32/test各38；重载一致；test270/380和274/380 | 该基线无需重跑；新B2/连续采样/全四卡评估与恢复组合的完整P3服务器运行，不能由历史三轮冒称已通过 |
| Phase4 RGB2 | 当前目录训练10240呈现，各事件1024；val各37；四rank步日志匹配；370题保存/重载相同 | 固定test18/event尚未成功执行 |
| Phase4 RGB4 | 正式数据、四卡B2短运行验证及编排已有证据 | 当前基线首轮、val、重载和test尚未开始 |
| 自动规则数据 | 已验收正式原生池298787训练题，路线缓存可复用 | U-E5 proceed/readiness YES仍无支持；不能签全能力覆盖 |

1:1 指十个事件的训练呈现配额及各 split 评测题数等量。**不要求 YES:NO=1:1，也不要求事件内各动作强制等量。** 评估不复制题目补桶；缺事件拒绝。连续选择与无放回不会自动带来路线多样性，现有 val 某些事件只有一个物理组，这是效果估计的限制。

本轮重算 P4 四rank实际消费并匹配计划；事件名来自绑定计划，未在本机重新读取完整服务器池或权重。P3六份消费回执均为 verified；重新解析2040条raw，预测/正确性及事件题数一致。P4重新解析370条raw，重载案例与保存验证完全一致。这些是已收到结果的再次核对，不是新GPU实验或新独立测试。

## 2. 剩余问题及其影响

| 问题 | 证据与边界 | 是否阻塞当前步骤 |
| --- | --- | --- |
| P4 test 快照读取失败 | 原服务器在rgb2.receipt.json哈希时异常；v81已改为打开文件描述符前后核对，CPU反例通过。原日志缺stat细节，根因未确证 | 需要服务器执行现有v81接续；不能在本机宣称已修复验收 |
| U-E5缺放行正例 | 正式池proceed/readiness YES=0；本轮画面对向车仍接近侵入走廊，不支持补YES | 不阻塞已批准的existing_data_development；阻塞“全部条件均已覆盖”的结论 |
| RGB2偏向NO | val316/370=85.41%，全NO基线304/370=82.16%；YES召回18/66=27.27%；U-E4为0/29，U-E6为4/22 | 属于模型开发弱项；先取得RGB4对照，不能直接断言错标或无正例，也不自动续七轮 |
| 部分事件评测单一 | U-E1/2/3、R-E5本次val均单NO；UE2 return、UE7 complete覆盖不足 | 不妨碍现有训练；不能把这些事件100%当成完整能力达标 |
| U-E7误判限制条件 | 六条NO→YES分别是renewed_restriction、restrict、hold各2条；proceed的24条NO均预测NO | 是条件识别错误，不应把六条一律统计为错误放行 |
| P3动作弱项 | 三轮val减速27/58、25/58；test左变道11/27、13/27。减速每轮2080次监督 | 不需先改规则；已曝光test不能反过来调标签/提示词或选超参 |
| 短图可观察性与规则真值不同 | 原生规则可能依赖停车义务历史、完整几何与参与者轨迹；两图/四图不能保证每题都包含足够信息 | 自动生成可继续；不得称规则一致率为独立真实准确率或将所有YES直接用于完整机动许可 |
| 方案进度落后于当前实现 | 总方案§14.1仍留早期not_run/v42内容，指南顶部更新也晚于正文 | 应使用本文和v81接续页确定当前步骤；不重新实施已完成优化 |

U-E4训练proceed YES已有262次呈现，因此本次0/29不能简单归因为没有正例。U-E5 val中的10个YES是restrict/hold，不是缺失的proceed YES；100%仍不能证明放行能力。

## 3. 本轮本地十事件逐帧复核

只使用已曝光的训练路线；9条路线从其原始起点因果重放369帧，耗时9.75秒。核验9条路线既有1842份源绑定前后SHA；实际查看每事件连续4帧、共40个不同RGB，120份对应RGB/meta/bbox源SHA通过。窗口与历史开发审核重叠，不计作40个全新独立样本，不是盲审。

40道选中RGB4规则提议的edge/target/facts与历史对应帧完全相同：35道YES/NO、5道UNKNOWN。没有重新编译或准入正式题库。P3同时导出的是原始轨迹诊断，3帧缺完整动作信号仍保留空值，不伪造KEEP，也不宣称重建了P3索引。

| 事件 | 本次帧 | 画面、当前规则及结论 |
| --- | --- | --- |
| U-E1 | InvadingTurn 13–16 | 红色前车在前方，不能仅据相对外观认定恢复条件；proceed UNKNOWN，未强制转成NO/YES |
| U-E2 | VehicleOpensDoorTwoWays 51–54 | 右侧白车开门，自车沿其左侧通过；depart YES针对指定可见通道。动作预测出现左右跨线不直接推翻条件题；未认证未见侧后方 |
| U-E3 | ParkingCutIn 38–41 | 黑车从右侧斜向并入前方；proceed NO。P3的未来RESUME不等于当前已获放行 |
| U-E4 | VehicleTurningRoute 48–51 | 骑行者在雾中从右向左横穿当前路径；proceed NO有可见冲突依据；远端可见性不作保证 |
| U-E5 | InvadingTurn 56–59 | 对向车沿中线附近持续接近，未完成可见退出；proceed NO，不制造放行正例 |
| U-E6 | OppositeVehicleRunningRedLight 31–34 | 消防车由右向左横穿路口；proceed NO，不能用自车方向信号覆盖横向冲突 |
| U-E7 | CrossJunctionDefectTrafficLight 13–16 | 蓝车横穿路口；proceed NO。故障事件身份不能仅凭四张图中的红灯重新认证 |
| R-E2 | noScenarios 34–37 | 目标左通道；左侧蓝车逐渐进入视野，enter为YES→UNKNOWN→NO→NO，未把右侧黑车直接当作左通道障碍 |
| R-E3 | HighwayExit 13–16 | 雾雨中前方及邻近车道均有车；enter NO；不据短窗口认证后方安全 |
| R-E5 | BlockedIntersection 49–52 | 前方排队与STOP标志可见；proceed NO，原始P3由DECELERATE转STOP；本片段未新增验证STOP放行正例 |

## 4. 当前代码/提示词核对

- 训练事件等量：P3 `train._contiguous_work`、P4 `rgb_short.batch_sampling.plan`接共享`sequence_batch.plan`；缺桶、非整除配额及历史不匹配拒绝。未见通过补标签凑1:1的路径。
- 验证/test：共享`sequence_evaluation.select`固定标签盲选、无放回、十事件等量。P4 `evaluation_policy.prepare/restore`绑定参考及评估成员；单事件/单答案覆盖不足与算法实现正确须分开报告。
- P3标签：未来轨迹提供预测监督，当前/历史RGB提供学生输入；不把P4当前许可作为动作真值。当前候选prompt保留STOP、首个明显速度变化、真实首跨和KEEP的语义。
- P4实际encoder调用`route_prompts.messages`，不能只审旧`prompts.SYSTEM`。活跃v12明确全部条件合取、移动/消失不等于许可、context不是条件证据；depart/enter/return明确仅可见条件，完整通道执行另需当前规划器批准。未见把老师facts直接渲染为生成答案的路径。
- 固定推荐配置：P3每卡B2/累积2，P4每卡B2/累积4，四卡；P3 warmup500更新，避免照搬旧2000更新。现有旧P3 adapter不能伪装成带optimizer的恢复点；当前新入口才保存完整epoch训练状态。
- v81 `phase4_continue`先核对当前权重/optimizer及成功reload回执，再CPU校验既有快照→RGB2 test→RGB4首轮/重载/test；已有输出拒绝、首错停止、缺快照不隐式重建。

284项现有针对性回归通过，9.58秒：事件采样、P3验证/提示词/训练状态、十事件teacher、U-E5及STOP修复、规则评估、接续和哈希边界。是CPU验证，未新增服务器GPU测试，也不与历史186项等累计为独立测试数。最初默认Python缺torch而收集失败，随后使用既有`/tmp/automot_batch_env/bin/python`完整通过；未安装依赖或放宽校验。

## 5. 服务器仅需当前接续

先确认已同步v81（本机提交`45dd0fc86`，564源码SHA匹配），在服务器现有AutoMoT根目录、原训练环境执行一次。完整说明见[PHASE4_TEST_SNAPSHOT_20261017.md](PHASE4_TEST_SNAPSHOT_20261017.md)。

```bash
P4_PRODUCTION_CACHE=checkpoints/p4_verified_routes_v69 \
python -m qwen3vl_local.audit_joint.phase4_continue \
  --root checkpoints/phase4_restart_20261017_162511 \
  --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B
```

默认自动选空闲四卡。需要显式固定四卡时使用同一命令并前置`GPU_IDS=0,1,2,3`，不要同时执行两条。保留整个`phase4_restart_20261017_162511`，尤其adapter、training_state.pt、view_upgrade及validated_data；旧125707目录已删除，不能使用旧命令。

成功/失败交回该目录的`handoff_continued/phase4_epoch_report.zip`及`phase4_epoch_package.json`。前置核验直接拒绝而未产包时，只交终端错误。已经运行过接续的目录不要重复执行或删除输出；依其实际回执判断下一步。

本轮只新增本地诊断脚本、证据、审计报告及轻量回执；未改生产源码/标签/提示词，未远程执行，未push。完整本地证据在`checkpoints/training_closure_review_20261017/`；未上传RGB、权重或数据。开训前本地收敛审计到此，剩余执行验收由上述服务器接续结果决定。
