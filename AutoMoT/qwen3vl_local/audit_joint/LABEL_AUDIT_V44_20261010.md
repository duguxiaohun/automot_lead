# Phase3/4 十事件逐帧 RGB 续审与 v44 标定修正（2026-10-10）

> 后续纠正（v45）：本文“过早放行指标”实际只指同实例 8 帧内复查筛查；Town04 f33 的已查触发项是 STOP hazard 导致优先权变化，不能仅据该序列认定 f31 的对向车冲突放行错误。原统计数保留历史含义，见 [v45 归因与限制](LABEL_AUDIT_V45_20261010.md)。

承接 [v43 原生接线](NATIVE_LABEL_FIXES_V43_20261010.md)。本轮先看连续 RGB，再看同帧 Phase3 动作与 Phase4 实例/条件，只有 RGB 与几何/元数据共同支持时才改规则。Phase4 合同 v44、老师 v12、同预约的新生产审核计划 v24（仅重新冻结源码 SHA）、空严格 registry v19。**这是有版本的开发审计与弱老师修正，不是全量标定完成、盲审或独立批准。**

机器证据：[label_audit_v44_verification_20261010.json](label_audit_v44_verification_20261010.json)；产物目录 `AutoMoT/checkpoints/joint_label_v44_20261010/`（`replay/`、`question_differences.json`、`visual_review.json`、`rgb/`、`data2/`、`data4/`、`native_build_verification.json`）。`superseded_run1/` 是中途发现 Accident 仍误建 U-E3 后作废的一次完整重算，只留作对照。

## 1. 逐帧 RGB 发现与判断

本轮实际查看 13 张联系表/放大图，**109 个不同 RGB 帧**，每帧 SHA 记录在 `visual_review.json`。选窗由已知标签和 Phase3/4 分歧驱动，不是盲审。

| 事件 | RGB + 几何结论 | 处理 |
| --- | --- | --- |
| U-E3（ParkingCutIn 1681_0） | actor 3697 在 f102–112 从停车道切入（横向 −3.0→−0.4 m），属真实 U-E3；f120–236 是车道中央静止前车（横向 −0.1），f244–256 起步，Phase3 同步 RESUME。f246 新建 U-E3 只来自整条路线常驻的 `cut_in_actors_ids`。 | 显式切入身份不能在历史起点已处于本车道走廊时再次建实例。 |
| U-E3（Accident 1819 的 88/89、ParkedObstacle 1923 的 163/164/167、ConstructionObstacle 2000 的 196、noScenarios ll_2 的 176） | 七个车辆的导出 road/lane 在整个因果历史中不变，RGB 中均在相邻车道行驶；“进入走廊”来自自车绕行规划路径移到相邻车道，或弯道上旋转足迹放大了横向范围。Phase3 同帧为 R-E2/U-E2，不是 U-E3。 | 通用（非显式）U-E3 需要参与者自身朝固定本车道走廊靠近 ≥0.5 m；导出车道全程不变则直接否决。 |
| U-E5（InvadingTurn 1450_0） | 侵入车 3752 在 f53 从左侧会车，f54 起在自车后方；旧实例仍保持 proceed UNKNOWN 至 f74。第二辆红车 3753 始终约 3 m 横向，未与本车道足迹重叠，Phase3 的 U-E5 情境不升级为第二实例。 | 决定性参与者连续两帧位于自车后方时截止等待（不记完成、不造 YES）。 |
| U-E7（Town03 002133） | f20–23 一辆横向车以约 6 m/s 从 15–21 m 外驶向路口，旧老师已给 proceed YES，f24 即 re_yield YES；专家全程停车（Phase3 STOP）。 | 交叉路口事件（U-E6/U-E7/R-E5）在自车近静止时用 4 s 而非 2 s 线性预测；行驶中保持 2 s。 |
| U-E4（DynamicObjectCrossing Town07 3_1） | 行人 140 在 f52 已离开本车道（proceed YES 合理），f58–62 站在路缘清晰可见，f63 突然消失；专家恰在 f63 起步。放大图确认。 | 登记为确认可见消失；Phase3 f63–66 起步候选及横跨该帧的人工 proceed 题隔离为 UNKNOWN。 |
| 其它源异常候选 | Town01 4_0 168→169 骑行者、Town02 3_3 68→69 行人、Town04 4_103 90→91 骑行者、Town13 1148_0 16→17 前方车流、Town13 1230_12 232→233 绿色 SUV、Town06 4_17 57→58 骑行者，前后两帧原图均确认可见后消失。Town03 8_107 46→47 的 actor 110 横向约 40 m、雾中无法确认，**不登记**。 | 共享隔离账本 2→10 条，每条绑定前后 RGB/meta/bbox 六份 SHA。 |

## 2. 代码修改

- [teacher_events.py](../sft_new_loop_phase4/teacher_events.py)：`cut_in_identity` 显式身份加“历史起点不在走廊内”；通用路径先做导出车道不变否决，再由 `crossed_fixed_corridor` 用冻结的绕行前参考路径（缺失时退回历史起点导航）计算参与者横向靠近量；`retirement` 新增 `decisive_actor_passed_behind_censors_stale_wait`；`clearance` 预测步长可配置，`facts` 对静止起步的路口事件用 `junction_start_prediction_s=4`。
- [label_quarantine_20261010.json](label_quarantine_20261010.json)：新增 8 条确认消失（7 条路线）。
- 版本：`teacher_rules.VERSION` v12、任务 `phase4_state_pair_binary_v44`、计划 v24、registry v19。
- 测试：新增 [test_teacher_v44.py](../sft_new_loop_phase4/tests/test_teacher_v44.py) 6 项；通用 U-E3 抽取夹具补齐与位置一致的 lane_id；5 个真实人工建库测试按新增隔离更新计数（训练 435→423、待审 +12）。

## 3. 前后差异（同一 45 条训练路线、5673 帧）

tmux 单进程低优先级重算 350.94 秒，峰值 RSS 约 361 MiB，退出 0。

- **Phase3**：5673 帧原始动作与轨迹证据逐帧不变；原生候选 1637→**1588**，减少 49 条全部来自新增隔离（Town02 13、Town01 13、Town04 13、Town07 6、Town06 4），其中 Town07 f63–66 是行人消失后的起步。没有新增候选。
- **Phase4 原始问题**（两模式含 UNKNOWN）9844→**8590**；精确同状态匹配 8554 题。
  - U-E3 实例 11→**3**（仅保留 ParkingCutIn 三个真实显式切入）；被删实例的后继状态问题全部消失，含原先进入弱库的 U-E3 proceed YES 70、NO 185 等/模式。
  - U-E5 两实例各在会车后由新截止结束，删除 23 道/模式的长尾 UNKNOWN，无监督标签变化。
  - U-E7：Town03 002133 f20–21 proceed YES→NO、f22–23 YES→UNKNOWN；新状态下新增 16 道 proceed NO/模式。
  - 隔离：Town04 4_103 的 R-E2 实例（f71 建立、f91 跨越骑行者消失）其后 152 道 release NO/模式转 UNKNOWN；Town13 1148_0 U-E2 29 道/模式。这沿用 v43“证据包络跨越异常起点即隔离”的规则，对长实例偏保守，已列为后续核查项，未放宽。
- **路口过早放行指标**（rgb4 proceed YES 后 8 帧内同实例 re_yield YES）：U-E7 9/10→**5/6**，U-E3 9/70→0（实例已删），R-E5 6/9 不变。复核 R-E5 1748_1：f36–39 放行与 Phase3 RESUME 一致，f44 的 re_yield 是行驶中出现的新限制，不属于本次修正对象。

## 4. 原生题库

`data2/`、`data4/` 均完成原生编译与加载核验，每模式 **train 3084 / val 273 / test 186**；3084 对按路线/末帧/实例/边/状态/答案完全配对；RGB4 val/test 与 v43 逐字节一致。

训练集 = **431 人工（443 中 12 道新隔离）+ 2653 弱监督**，YES 503 / NO 2581（v43 为 577/2913）。各事件 YES/NO：

| 事件 | v43 | v44 |
| --- | --- | --- |
| U-E1 | 70/1028 | 70/1028 |
| U-E2 | 95/431 | 84/413 |
| U-E3 | 60/184 | **13/14** |
| U-E4 | 62/59 | 50/59 |
| U-E5 | 12/39 | 12/39 |
| U-E6 | 29/38 | 29/38 |
| U-E7 | 47/145 | 43/153 |
| R-E2 | 108/560 | 108/408 |
| R-E3 | 62/146 | 62/146 |
| R-E5 | 32/283 | 32/283 |

U-E3 训练支持大幅减少是去掉错误实例后的真实状态，不能用重复采样掩盖；U-E4 YES 减少的 12 道正是人工 proceed 带跨越骑行者/行人消失帧的题。这些计数仍远未 1∶1，也不代表全池。

## 5. 验证

相关完整回归 **2032 项通过**（Phase4 tests、audit_joint tests、Phase3 包内测试，含真实 P3 扫描/缓存准入专项）。旧 v43 题库在当前源码下按合同拒绝加载，需原源码复验，未被改写。未执行 GPU 训练、模型重载、CARLA 或新的独立批准。

## 6. 仍未完成

- 转弯穿越对向车流的放行：Town04 002163 f31 在对向车靠近时给出 proceed YES、f33 re_yield，需对向车道冲突和转弯路径建模，本轮未改。
- 行驶中路口时间窗、长实例证据包络的隔离宽度需要单独设计与 RGB 校验。
- U-E3 通用（非显式）真实切入在本批 45 条路线没有正例，召回只能在扩大路线后验证。
- G1 实例匹配、G2 盲审、Phase3 正式全池、六事件独立评估与全量生产仍未完成；G1 in_progress、G2 not_run 不变。
