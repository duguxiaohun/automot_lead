# Phase3 v24：训练审计总结、连续 RGB 复核与小幅修订

日期：2026-09-26。历史指标详表见 [AUDIT_COMPARISON_20260926.md](AUDIT_COMPARISON_20260926.md)。本次按要求先看真实连续 RGB 和原始 meta，再修改代码；没有按模型答案反改真值，也没有运行新模型训练。

## 1. 这次训练有没有进步

四包均为 20260923 的 v23 训练；A/B 为四图/两端点 binary，C/D 为四图/两端点 choice。

| 指标 | A 四图 binary | B 两图 binary | C 四图 choice | D 两图 choice |
| --- | ---: | ---: | ---: | ---: |
| 本轮测试整题正确率 | 70.24% | 71.32% | 68.48% | 67.39% |
| 最近 v21 对照 | 63.57% | 62.14% | 66.57% | 63.71% |
| 表观变化 | +6.66pp | +9.18pp | +1.91pp | +3.68pp |
| 本轮仅有效题正确率 | 65.65% | 66.96% | 68.48% | 67.39% |
| 本轮动作 macro F1 | 62.22% | 63.92% | 63.57% | 62.09% |

最近与本轮测试逐题身份交集为零，不能把这些差值当作相同测试集上的模型增益。binary 的错误道路 INVALID 比较容易且计入总分，与不含 INVALID 的 choice 比较时应看有效题。

所查历史完整 binary 最好总分 67.71%，本轮 B 表观超过；历史完整 choice v13 为 68.77%，本轮 C 仍低 0.29pp。旧单动作 choice 的 79.77% 任务合同不同，不作可比最优值。动作宏平均也没有全面超过历史最好。

四种设置相对 v21 的 STOP/RESUME F1 都提高，但 RIGHT/KEEP 都下降。C/D 的 100 道主要减速题只对 30/33；晚于 0.5 秒出现显著减速的 43 题只对 4/5。STOP 的 139 道“当前已在等待”题对 137，而 47 道“当前行驶、未来停车”题只对 30。模型对当前状态的识别明显好于未来阶段预测。

格式已 100% 合法；这不是输出格式问题。四包均未产生通过守卫的 best_generation，production_ready 为 false。四图与两图的差异方向不一致，路线聚类区间跨零，不能确定四图更优。本轮实际仍使用旧 `primary_action_capacity_return_v2`，没有验证后来接入的 smooth_cap/完整训练池采样效果。

## 2. 本次实际看了多少、看了什么

- 连续目视复核 **60 段、58 个物理路线组、1,020 张不同 RGB**，覆盖全部十个 context。14 段为模型正确对照，46 段为错误样本。
- 每段固定查看 anchor−3 至 anchor+13 的全部 17 帧：0.75 秒历史、当前帧、至未来 3.25 秒。每帧保留完整左/前/右拼接视图；绿色是模型历史输入，灰色未来只供审计。
- 逐段检查交通参与者、道路与车道线、绕障/回归方向、等待/起步/调速、历史与未来动作先后；结合原始速度、brake/throttle、road/lane 身份核对。没有将转向、曲线或邻车运动直接当成自车跨线。
- 对 17、19、29、31、57 号另看原始分辨率四帧输入；这些是重复放大，不增加 1,020 的计数。未增亮、生成或修补原始图像。
- 原始数据回放覆盖 **117 条录制路线、15,297 个 meta 记录、460 道有效题**，验证全部题的速度与动作；这不是目视看完 15,297 帧，也不是完整生产 train 池审计。

按 context 的片段数：LEAD_BRAKE 4、STATIC_BLOCKAGE 12、DYNAMIC_CUTIN 4、VULNERABLE_CROSSING 3、ONCOMING_INVASION 6、JUNCTION_RULE_CONFLICT 2、SIGNAL_FAILURE 3、POST_BYPASS_RETURN 7、UNSIGNALIZED_PRIORITY 14、RAMP_MERGE_EXIT 5。

这是针对错误和边界、兼顾路线多样性与正确对照的定向复核，非随机抽样，不能据此估计全数据错标率。目视结论来自本次模型辅助审查，未声称经过多人独立盲标。没有逐帧驾驶意图的独立真值，场景中的物体与某次制动同时出现也不等于已证明因果。

逐段笔记见 [rgb_review_notes_20260926.jsonl](rgb_review_notes_20260926.jsonl)。原始文件路径、逐帧 RGB/meta SHA256、原题及输出保存在本地 [evidence.json](probe_output/rgb_review_20260926/evidence.json)，可由本地 `render_review.py` 重现面板。文末附全部 60 段观察与链接。

## 3. 证据支持哪些结论

### 3.1 短暂停顿、真正等待、立即起步不能混为一类

1 号 StaticCutIn/Town13_1704_1/f92：右前蓝 SUV 附近减速，f96 单次近停后立即继续运动。GT DECELERATE 有原始运动支持，不能因为预测 STOP 就重标。

33/34 号则不同：分别在 anchor+6（1.5 秒）开始近停，anchor+7（1.75 秒）确认，之后继续停住。它们是**持续停车的确认截止边界**，并非瞬时停顿。原规则要求确认在立即窗内，因此仍标 DECELERATE；本轮没有把窗长调到模型答案更好看的位置。

4/5/12/13 号当前接近零速，但未来立即持续起步；15/18/32 号则先等待、之后才释放。连续画面支持保留 v9 已确认起步例外，同时保留先等待的 STOP 优先级。

### 3.2 横向错误中有主动作选择和阶段辨认问题

6/7/8/9/27/43 号的左借道或右回归，在连续 RGB 与 lane 身份切换中一致，不能把模型预测伴随减速/加速当成横向标定错误。28 号正确对照先在历史左跨、再在未来右归，说明不能重复报告历史动作。42 号确实先停车等待再左跨，主要动作仍为 STOP；47 号只有邻车从左经过，ego 没有在窗口内左跨。

本次未确认需要新增的地图重编号假变道隔离。既有横向优先级、车道检查和精确例外保留。

### 3.3 KEEP 不等于无风险，过去速度也不能替代当前基准

14/24/40/41/44/59 号存在周围交通或后续明显减速，但当前固定窗口内只是相对当前速度的小幅变化；KEEP 合同成立。21/49/54 号历史在加速，未来却先减速；30/50 号历史减速，未来开始加速。仅延续历史趋势会答反。

26/46/58 号说明阈值附近的 KEEP/RESUME/DECELERATE 从短 RGB 历史本来就难精确区分。不能将“符合数值规则”直接等同于“输入足以无歧义预测”，也不能以低正确率直接认定标注错误。

### 3.4 有可见性限制，但不以场景或模型错例作批量过滤

17 号极暗，前车轮廓/灯光可见但制动细节不足；19 号锥桶可辨，侵入车辆在历史中未能可靠辨认；29/57 号难确认优先车辆；31 号行人细节不足。这五段保持未解决的可见性备注，不生成 INVALID，不凭未来出现对象倒推历史里已看见。

3 号施工案例反过来修正了最初印象：完整连续 RGB 确认输入远处已有黄色施工设施，之后接近并停车。因此旧报告的“前视较开阔”不能当作缺乏障碍证据的结论，已更正。

综合判断：当前主要证据指向**模型尚未稳定掌握动作阶段/优先级，加上短历史预测与硬窗口边界的可辨识性限制**。没有证据支持“全局标定误差太大，所以全面改阈值”；也不能证明只改提示词就会解决。60 段保留原标签，不等于证明全数据零错标。

## 4. 实际改动：小范围修订

### 提示词 v24_motion_reference

仅压缩改写共享速度说明，两种题型同源：

- 明确以最新速度为基准，不以更早峰值或整段趋势为基准。
- 明确瞬间降至近停后继续运动，不等同于持续等待。
- 保留低速不必然 STOP、立即起步可以 RESUME、减速后恢复仍可 DECELERATE、RESUME 不要求此前停过。

OBSERVATION、横向规则、十类场景条件与因果目的文案保持原意。原有最大 610 个英文词的测试预算未放宽。没有加入未来速度、控制、阈值、确认帧数或审计标志。

### 标定复核方法：细分近停证据，不变更动作阈值

`action_review.near_stop_review` 在固定九个速度采样内枚举连续近停片段，保存开始、末次近停、采样数、确认、释放和窗末是否尚未闭合。新增四个诊断桶：

| 新诊断桶 | 含义 |
| --- | --- |
| single_near_stop_sample_released_in_window | 只有一个近停采样，且窗内已观察到恢复运动 |
| near_stop_pair_confirmation_crosses_immediate_boundary | 立即窗截止处开始近停，确认落在下一帧 |
| near_stop_pair_starts_after_immediate_window | 两点近停片段在立即窗之后才开始 |
| near_stop_at_window_end_unconfirmed | 最后一个速度采样近停，窗内不能确认或判断释放 |

旧 `isolated_near_stop_in_1_5s` 保留用于历史兼容，但它只表示“立即窗内未形成确认对”，不能按名字解释成真实孤立点。新字段同时接入逐题 action_review、原始 meta 边界审计和 temporal slices。审计片段本身不覆盖已确认起步例外，不成为训练特征、过滤条件或采样权重。

C 的主要 DECELERATE 子集中，新“单点后释放”桶为 13 题、正确 0；确认跨立即截止为 7 题、正确 0；晚起近停对为 11 题、正确 1；窗末单点未确认为 8 题、正确 2。各桶可重叠，不能相加，也不等于错标率。详见 [replay_summary.json](probe_output/rgb_review_20260926/replay_summary.json)。

## 5. 验证、隔离与使用边界

- **760 项 CPU 回归通过**：Phase3 635 项，加相关 Action 数据准备、划分、动作 token/文字衔接 125 项；新增回归覆盖真实边界曲线、窗末未知、尾部不回写、起步例外和物理路线隔离。
- 460 道有效题重读原始 meta 后，原动作标签变化 **0**，原纵向判定字段变化 **0**；两种题型×两种图数 **1,840 次 prompt/target 回放通过**，新增诊断不进入输入。
- 本轮四包 test/导出 generation-val 共 **176 个新物理组**加入 train-only，累计 **1,955 组**。只有58组被本轮连续目视复核；其余因指标/逐题分析曝光而隔离，不能冒称全部目视。导出周期验证有500行截断，不声称覆盖未导出的验证身份。
- 用历史 v21 候选去掉 anchor<4 后的 **190,000 条**做划分容量回放，移动64组，可维持 val/test 十类各≥32帧、≥5物理组。该检查沿用旧标签，**不替代当前全量生产重建和哈希校验**，也不证明泛化充分。
- 默认新目录改为 `sft_new_loop_phase3_data_v24`，新 prompt/诊断/开发隔离名单已进入 mapping 合同。Phase3 需新索引、新 run；共享 Action full map 也需重建。旧 run 用原源码，不手改 manifest 哈希。
- 未跑新 GPU 训练或新模型推理，因此**不宣称修改后准确率提高**。下一轮优先按同数据/seed/预算配对比较提示词；不要再用本次已曝光测试路线报告盲测提升。

初次全量测试曾因系统 Python 缺 torch 无法收集，随后使用已有 `pvi` 环境完整通过；未安装新依赖或绕过生产检查。数据、RGB、训练权重与原审计包均未覆盖。

## 6. 全部逐段观察

以下编号对应 `case_000.jpg` 至 `case_059.jpg`；范围内17帧全部查看，说明记录关键变化，不能把每段一个观察当作独立帧级意图真值。

| 编号/面板 | 路线与帧 | 原标签 → 模型输出 | 连续观察与处理 |
| --- | --- | --- | --- |
| [00](probe_output/rgb_review_20260926/case_000.jpg) | MergerIntoSlowTrafficV2 / Town13_1249_7_route0_01_10_04_50_09<br>f54–70，anchor 57 | DECELERATE → STOP | 前方蓝车与左侧车队持续可见；f58仅短暂近停后继续跟行，f68再近停。保留减速，不把两次短暂停顿合并为持续停车。 |
| [01](probe_output/rgb_review_20260926/case_001.jpg) | StaticCutIn / Town13_1704_1_route0_01_10_13_20_24<br>f89–105，anchor 92 | DECELERATE → STOP | 蓝色SUV位于右前并逐渐贴近，自车f96短暂近停后立即加速从其左侧通过；不是持续等待。保留减速，需明确短暂近停与STOP区别。 |
| [02](probe_output/rgb_review_20260926/case_002.jpg) | VehicleTurningRoute / Town05_Town05_Scenario4_102_route0_01_10_08_07_05<br>f6–22，anchor 9 | DECELERATE → KEEP | 前方黄车始终可见，自车先调速再接近排队并f17后停住；当前锚点的停车在立即窗之后。保持原减速判据，标记晚停车边界。 |
| [03](probe_output/rgb_review_20260926/case_003.jpg) | ConstructionObstacleTwoWays / Town12_route_002692_route0_01_09_02_14_42<br>f155–171，anchor 158 | DECELERATE → KEEP | 施工拖车/黄色围挡在输入远处，随后明显接近并停车；不是无障碍空路，原报告只看历史缩略的‘较开阔’不能当缺证据结论。保留减速，强化当前基准的后续速度变化。 |
| [04](probe_output/rgb_review_20260926/case_004.jpg) | InvadingTurn / Town12_4195_0_route0_01_09_08_28_04<br>f111–127，anchor 114 | RESUME → STOP | 对向灰车由远及近向左侧通过，自车随即起步，后续明显前移；到+3秒又制动不回写2秒内起步。保留RESUME，输入端仍低速/静止且对向车未通过，存在预测不确定性。 |
| [05](probe_output/rgb_review_20260926/case_005.jpg) | HardBreakRoute / Town12_649_0_route0_01_10_16_26_23<br>f73–89，anchor 76 | RESUME → STOP | 前方黑车起步，自车f77后逐渐跟进，三图均支持队列释放；保留起步，不以t=0静止硬判STOP。 |
| [06](probe_output/rgb_review_20260926/case_006.jpg) | ParkedObstacle / Town06_route_001939_route0_01_09_16_27_48<br>f20–36，anchor 23 | DECELERATE+LANE_CHANGE_LEFT → DECELERATE | 左侧黑车先切入，自车减速后加速接近右前停放黑车，f34-35跨左侧虚线，meta -6到-5一致。主要左变道成立，速度先发生不能替代横向主动作。 |
| [07](probe_output/rgb_review_20260926/case_007.jpg) | AccidentTwoWays / Town15_route_001376_route0_01_11_01_56_10<br>f71–87，anchor 74 | RESUME+LANE_CHANGE_LEFT → RESUME | 对向车通过后自车加速向左跨中心线绕过右侧警车及事故车；f76-77 lane -1到+1，与RGB一致。保留RESUME+LEFT。 |
| [08](probe_output/rgb_review_20260926/case_008.jpg) | VehicleOpensDoorTwoWays / Town12_3790_1_route0_01_09_00_29_54<br>f68–84，anchor 71 | DECELERATE+LANE_CHANGE_RIGHT → DECELERATE | 右侧警车已从视野移出，自车随后的回正跨线与道路中心线相对位移可见，f72 lane -1到+1；需按进入road的方向解释为右回归。减速只是伴随动作，保留RIGHT。 |
| [09](probe_output/rgb_review_20260926/case_009.jpg) | MergerIntoSlowTraffic / Town12_2787_1_route0_01_10_20_02_09<br>f27–43，anchor 30 | LANE_CHANGE_RIGHT → KEEP | 夜间高速虚线仍可辨，f35-38自车跨向右侧车道而非仅沿曲线，lane +4到+5；原RIGHT有过程支持，不能因前方空旷改KEEP。 |
| [10](probe_output/rgb_review_20260926/case_010.jpg) | StaticCutIn / Town12_1344_0_route0_01_09_11_29_53<br>f57–73，anchor 60 | DECELERATE → STOP | 黄车与右前黑车、对向车辆可见，f67-68后才持续近停；比立即停车窗口晚，保留减速并记录截止边界。 |
| [11](probe_output/rgb_review_20260926/case_011.jpg) | CrossJunctionDefectTrafficLight / Town03_route_002140_route0_01_09_14_56_53<br>f19–35，anchor 22 | DECELERATE → STOP | 横向黄色/橙色车辆连续穿过路口，自车先蠕行再减速，f31之后才停。STOP道路字样不能直接当实际停车动作证据；保留减速。 |
| [12](probe_output/rgb_review_20260926/case_012.jpg) | CrossJunctionDefectTrafficLight / Town03_route_002140_route0_01_09_14_56_53<br>f46–62，anchor 49 | RESUME → RESUME | 红色横穿车辆从左至右经过，自车从静止加速通过路口，模型正确。说明存在当前速度为零但RESUME的有效对照，保留规则。 |
| [13](probe_output/rgb_review_20260926/case_013.jpg) | InvadingTurn / Town12_4195_0_route0_01_09_08_28_04<br>f93–109，anchor 96 | RESUME → STOP | 对向灰车从近前向左离开，ego f97以后起步；f106以后的再停在2秒窗外，不应撤回RESUME。保留起步。 |
| [14](probe_output/rgb_review_20260926/case_014.jpg) | DynamicObjectCrossing / Town05_Town05_Scenario3_41_route0_01_09_05_51_12<br>f47–63，anchor 50 | KEEP → DECELERATE | 白天弯路跟随橙车，f50-58速度仅小幅波动，对向车通过；f62以后才明显减速。保留KEEP，不能用更晚减速或存在交通参与者来否定。 |
| [15](probe_output/rgb_review_20260926/case_015.jpg) | ParkingCutIn / Town13_74_0_route0_01_08_04_53_20<br>f152–168，anchor 155 | STOP → STOP | 近前橙色对向车通过时自车f153-157持续等待，之后才释放起步。保留STOP，与立即起步规则形成对照。 |
| [16](probe_output/rgb_review_20260926/case_016.jpg) | CrossJunctionDefectTrafficLight / Town05_route_002063_route0_01_10_00_26_32<br>f24–40，anchor 27 | STOP → STOP | 大雾路口但近前横穿红车和后续MINI可辨；ego当前停稳，随后释放。保留STOP，不因雾或绿灯判事件无效。 |
| [17](probe_output/rgb_review_20260926/case_017.jpg) | HardBreakRoute / Town12_2325_0_route0_01_10_01_35_10<br>f30–46，anchor 33 | KEEP → DECELERATE | 夜间照度很低，近前/对向车辆轮廓可见但关键车制动不易从面板确认；数值窗主要在5-8m/s波动，KEEP是当前基准口径。需放大原图，不凭黑暗直接删除。 原始分辨率四帧复核：前车轮廓和灯光可见，但无法可靠判定其制动强度；输入可辨识性偏弱。 |
| [18](probe_output/rgb_review_20260926/case_018.jpg) | HardBreakRoute / Town12_2032_0_route0_01_10_01_01_05<br>f67–83，anchor 70 | STOP → STOP | 绿色前车近距离等待，对向SUV驶过；ego保持静止至f76后才释放。原STOP及模型正确，作为延迟起步对照。 |
| [19](probe_output/rgb_review_20260926/case_019.jpg) | InvadingTurn / Town13_1477_0_route0_01_08_04_11_51<br>f39–55，anchor 42 | DECELERATE → KEEP | 输入弯路远处锥桶渐近，明显对向车在后续更接近时出现，f50减速f51停车。原速度标签正确；事件在输入是否已有足够可见证据需原图放大，不直接裁错。 原始分辨率四帧复核：锥桶可辨，侵入车辆在输入中未能可靠辨认；事件文字为给定条件，不能由后续车辆出现反推输入可见。 |
| [20](probe_output/rgb_review_20260926/case_020.jpg) | InvadingTurn / Town13_1469_0_route0_01_10_18_43_57<br>f107–123，anchor 110 | DECELERATE → DECELERATE | 夜间对向红MINI在历史输入已有轮廓，随后近前通过，自车7.8降到5.5m/s，正确减速；黑暗不等于无可见证据。 |
| [21](probe_output/rgb_review_20260926/case_021.jpg) | StaticCutIn / Town13_53_0_route0_01_09_03_51_17<br>f2–18，anchor 5 | DECELERATE+LANE_CHANGE_LEFT → RESUME | 输入历史正在加速，右前蓝SUV与左前车辆可见；随后先明显减速并左跨线，再恢复加速。应预测锚点之后，不能外推此前加速；保留DECEL+LEFT。 |
| [22](probe_output/rgb_review_20260926/case_022.jpg) | AccidentTwoWays / Town13_route_003098_route0_01_09_04_52_58<br>f71–87，anchor 74 | RESUME+LANE_CHANGE_LEFT → LANE_CHANGE_LEFT | 右侧警车/事故障碍，自车加速左跨中心线绕行，lane+1到-1与画面一致；正确LEFT对照。 |
| [23](probe_output/rgb_review_20260926/case_023.jpg) | MergerIntoSlowTrafficV2 / Town12_968_41_route0_01_08_08_35_19<br>f125–141，anchor 128 | KEEP → LANE_CHANGE_RIGHT | 右车道灰车与弯曲道路可见，自车当前车道内近匀速，road/lane全窗不变，未见自身跨线；保留KEEP，邻车/曲线不能当自车变道。 |
| [24](probe_output/rgb_review_20260926/case_024.jpg) | MergerIntoSlowTrafficV2 / Town13_1398_4_route0_01_11_00_16_01<br>f24–40，anchor 27 | KEEP → KEEP | 夜间汇入车队，前车及右侧车可见；2秒内速度约7.4-8，保持车道。后续加速在窗口外，正确KEEP。 |
| [25](probe_output/rgb_review_20260926/case_025.jpg) | CrossJunctionDefectTrafficLight / Town05_route_002065_route0_01_10_08_45_03<br>f24–40，anchor 27 | DECELERATE → KEEP | 雾中多车横穿路口，自车先继续前进再明显制动，f34-35持续停住比立即窗晚。保持DECEL，灯色不能取代实际运动。 |
| [26](probe_output/rgb_review_20260926/case_026.jpg) | CrossJunctionDefectTrafficLight / Town05_route_002060_route0_01_10_17_43_04<br>f31–47，anchor 34 | KEEP → KEEP | 夜间蓝车横穿且自车仍前进，速度波动但相对锚点最大下降接近而未达到阈值。正确KEEP为阈值附近对照；不能以存在横穿车硬判STOP。 |
| [27](probe_output/rgb_review_20260926/case_027.jpg) | AccidentTwoWays / Town15_route_001591_route0_01_09_17_11_11<br>f84–100，anchor 87 | RESUME+LANE_CHANGE_RIGHT → DECELERATE | 事故车在右侧，自车先加速沿借道通过，f97回到右侧lane-1，与RGB回归一致；减速出现在速度窗外，主要RIGHT成立。 |
| [28](probe_output/rgb_review_20260926/case_028.jpg) | ConstructionObstacleTwoWays / Town15_route_001123_route0_01_09_10_05_15<br>f72–88，anchor 75 | RESUME+LANE_CHANGE_RIGHT → LANE_CHANGE_RIGHT | 施工路障已在右前，历史已经左跨但未来f84右归；正确RIGHT说明不能重复历史LEFT。随后大幅减速在速度窗外。 |
| [29](probe_output/rgb_review_20260926/case_029.jpg) | OppositeVehicleTakingPriority / Town12_1644_1_route0_01_09_18_48_46<br>f33–49，anchor 36 | STOP → RESUME | 路口前减速后f40起持续停住，STOP运动明确；优先车辆救护车至f47才明显出现，输入证据可见性待放大，不能把后见之明当输入证据。 原始分辨率四帧复核：停车标志、道路可辨，未能可靠辨认后续救护车；保持运动标签，因果可见性未确认。 |
| [30](probe_output/rgb_review_20260926/case_030.jpg) | AccidentTwoWays / Town15_route_001381_route0_01_09_02_45_54<br>f6–22，anchor 9 | RESUME → RESUME | 夜间障碍灯柱与前车可见，自车输入末减速后随即加速，保持加速后的速度；正确RESUME，历史减速不等于未来减速。 |
| [31](probe_output/rgb_review_20260926/case_031.jpg) | VehicleTurningRoutePedestrian / Town13_1694_1_route0_01_10_11_13_36<br>f146–162，anchor 149 | STOP → RESUME | 路口穿行车辆及左侧人影，ego f150起持续停稳，运动支持STOP，模型RESUME错误；行人可见性需放大历史。 原始分辨率四帧复核：路口车辆清楚，疑似人影过小，无法可靠确认行人运动细节；不把猜测当标定依据。 |
| [32](probe_output/rgb_review_20260926/case_032.jpg) | VehicleTurningRoute / Town04_Town04_Scenario4_107_route0_01_10_05_34_01<br>f156–172，anchor 159 | STOP → STOP | 蓝色对向SUV持续通过，自车当前及未来2秒均等待，之后才起步。正确STOP，不能把末尾释放倒推为立即起步。 |
| [33](probe_output/rgb_review_20260926/case_033.jpg) | DynamicObjectCrossing / Town03_Town03_Scenario3_5_route0_01_07_21_27_49<br>f10–26，anchor 13 | DECELERATE → KEEP | 夜间跟车旁有橙车，ego1.5秒首次近停，1.75秒确认持续停车，属于确认帧越过立即窗而非真正孤立近停。DECEL规则一致，但应独立标记确认截止边界。 |
| [34](probe_output/rgb_review_20260926/case_034.jpg) | ConstructionObstacleTwoWays / Town12_4437_1_route0_01_10_16_58_03<br>f71–87，anchor 74 | DECELERATE → KEEP | 前方施工拖车清楚可见，对向黄车通过；1.5秒首次近停、1.75秒确认，停车延续。与33同类边界，不能混称瞬时近停。 |
| [35](probe_output/rgb_review_20260926/case_035.jpg) | AccidentTwoWays / Town13_route_003063_route0_01_10_18_25_23<br>f67–83，anchor 70 | RESUME+LANE_CHANGE_LEFT → LANE_CHANGE_LEFT | 对向黑车通过后立即起步，随后左跨中心线绕过右侧事故车。正确LEFT，锚点单次低速不是等待。 |
| [36](probe_output/rgb_review_20260926/case_036.jpg) | InvadingTurn / Town12_3367_0_route0_01_09_14_35_06<br>f75–91，anchor 78 | DECELERATE → KEEP | 对向红MINI已近前通过，之后极暗但锥桶连续可见；ego7.44降5.47后恢复，后又下降，非持续停车。单次阈值越过敏感但真实减速不宜强改KEEP。 |
| [37](probe_output/rgb_review_20260926/case_037.jpg) | ParkingExit / Town10HD_route_001688_route0_01_10_21_40_02<br>f12–28，anchor 15 | DECELERATE → KEEP | 左侧黑车向前并行、右侧停放车辆；ego加速历史之后出现短周期调速，首次过减速阈值在f22。保留DECEL并标记短暂阈值事件，不能用整个片段均速取代当前基准。 |
| [38](probe_output/rgb_review_20260926/case_038.jpg) | VehicleTurningRoute / Town07_Town07_Scenario4_90_route0_01_09_21_14_28<br>f26–42，anchor 29 | DECELERATE → KEEP | 夜間弯路迎面车辆可见，路口road多次改变但没有自车横跨同路车道；相对锚点先下降后恢复，保留速度DECEL，不误当横向。 |
| [39](probe_output/rgb_review_20260926/case_039.jpg) | VehicleTurningRoute / Town06_Town06_Scenario4_28_route0_01_10_22_31_34<br>f179–195，anchor 182 | STOP → STOP | 数名骑行者在车前连续通过，ego全窗停稳，正确STOP。亚阈值速度变化只来自近零抖动，不是实质减速边界。 |
| [40](probe_output/rgb_review_20260926/case_040.jpg) | AccidentTwoWays / Town15_route_001595_route0_01_08_11_17_31<br>f4–20，anchor 7 | KEEP → KEEP | 雾中跟前车直行，当前6.52后短暂5.7再7左右，无显著相对变化；正确KEEP，制动控制不直接定义动作。 |
| [41](probe_output/rgb_review_20260926/case_041.jpg) | InvadingTurn / Town12_4223_0_route0_01_10_14_42_08<br>f57–73，anchor 60 | KEEP → DECELERATE | 对向黑车借道通过锥桶区，自车7.2-9.2调速，相对当前8.05未达到显著变化。KEEP并不代表风险清除，模型DECEL过响应。 |
| [42](probe_output/rgb_review_20260926/case_042.jpg) | Accident / Town06_route_001856_route0_01_10_09_51_57<br>f48–64，anchor 51 | STOP+LANE_CHANGE_LEFT → LANE_CHANGE_LEFT | 事故在右前，左车先通过；ego f54-57停稳等待后起步，f63左跨。STOP优先于稍晚LEFT合理，模型只报LEFT漏掉先等待。 |
| [43](probe_output/rgb_review_20260926/case_043.jpg) | AccidentTwoWays / Town13_route_003110_route0_01_08_21_24_29<br>f126–142，anchor 129 | RESUME+LANE_CHANGE_RIGHT → DECELERATE | 历史已左借道，ego先加速通过右侧事故车，f139向右回原车道，RGB/meta一致；主要RIGHT，不是DECEL。 |
| [44](probe_output/rgb_review_20260926/case_044.jpg) | HighwayExit / Town13_1396_2_route0_01_11_03_59_35<br>f60–76，anchor 63 | KEEP → DECELERATE | 高速右侧车道前进，左侧车辆并行，13.8至13.1小幅调速，明显减速在+2.75秒后。KEEP成立，不外推全出口流程。 |
| [45](probe_output/rgb_review_20260926/case_045.jpg) | ConstructionObstacleTwoWays / Town12_1945_0_route0_01_11_03_50_56<br>f123–139，anchor 126 | DECELERATE → KEEP | 历史右下锥桶表明刚绕过施工，自车锚点已完成跨线，未来沿当前车道减速。远处迎面车辆可见；保留DECEL，过去跨线不重复标RIGHT。 |
| [46](probe_output/rgb_review_20260926/case_046.jpg) | AccidentTwoWays / Town15_route_001478_route0_01_09_22_17_52<br>f5–21，anchor 8 | RESUME → KEEP | 蓝色前车雾中直行，ego相对当前6.12先小降再持续超过7.35，属于阈值附近RESUME。画面难精确区分KEEP，属可辨识性/离散阈值敏感，不认定错标。 |
| [47](probe_output/rgb_review_20260926/case_047.jpg) | ParkedObstacle / Town06_route_001949_route0_01_09_15_36_49<br>f31–47，anchor 34 | STOP → LANE_CHANGE_LEFT | 左车流经过且右前障碍车阻挡，ego短暂前移后f38-43停稳再释放，没有窗口内自身左跨；STOP正确，不能按障碍场景默认LEFT。 |
| [48](probe_output/rgb_review_20260926/case_048.jpg) | AccidentTwoWays / Town13_route_003257_route0_01_09_14_15_08<br>f16–32，anchor 19 | DECELERATE → RESUME | 前方排队车辆和对向来车，自车先4.57降到不足1后蠕行再停；立即窗未持续近停，DECEL合理，不能把稍后的速度回升当RESUME。 |
| [49](probe_output/rgb_review_20260926/case_049.jpg) | ConstructionObstacleTwoWays / Town15_route_001206_route0_01_09_22_48_31<br>f4–20，anchor 7 | DECELERATE → KEEP | 初始已过f0，跟前车历史加速，随后6.35骤降3.42再恢复；画面显示连续位移无初始化跳变。保留DECEL，不能扩大排除到所有早期加速/调速。 |
| [50](probe_output/rgb_review_20260926/case_050.jpg) | ParkingExit / Town03_route_001721_route0_01_07_22_41_03<br>f42–58，anchor 45 | RESUME → STOP | 侧方车辆经过后自车从约1m/s立即加速到5-6，保持前进；历史刚大幅减速不意味着还将STOP。保留RESUME。 |
| [51](probe_output/rgb_review_20260926/case_051.jpg) | NonSignalizedJunctionLeftTurnEnterFlow / Town12_route_002769_route0_01_10_15_39_29<br>f63–79，anchor 66 | DECELERATE → KEEP | 夜间转入有车流道路，迎面与前车可见，自车9.93降4再恢复但低于锚点，DECEL明确，模型KEEP漏掉速度下降。 |
| [52](probe_output/rgb_review_20260926/case_052.jpg) | VehicleTurningRoute / Town07_Town07_Scenario4_92_route0_01_09_04_16_08<br>f10–26，anchor 13 | STOP → KEEP | 路口右侧骑行者可见，ego随即f14-16停稳后释放；虽很快再起步，先停车阶段明确，STOP成立。 |
| [53](probe_output/rgb_review_20260926/case_053.jpg) | NonSignalizedJunctionLeftTurnEnterFlow / Town05_route_000960_route0_01_08_02_07_39<br>f35–51，anchor 38 | DECELERATE → KEEP | 雾中路口转弯并入道路，迎面车流密集，ego先约10.7维持再降到7-8；DECEL是较晚但仍窗内的变化，KEEP错误。 |
| [54](probe_output/rgb_review_20260926/case_054.jpg) | VehicleTurningRoute / Town07_Town07_Scenario4_108_route0_01_10_07_18_29<br>f3–19，anchor 6 | DECELERATE → RESUME | 雨雾跟车，历史加速后短周期调速，f12降到2.75再恢复，保留DECEL；早期录制不等于初始化异常。 |
| [55](probe_output/rgb_review_20260926/case_055.jpg) | HardBreakRoute / Town13_1540_0_route0_01_09_06_34_45<br>f208–224，anchor 211 | STOP → RESUME | 前方黑车刹车灯亮且近距离排队，ego当前已停稳后持续等待，模型RESUME无后续运动支持。保留STOP。 |
| [56](probe_output/rgb_review_20260926/case_056.jpg) | NonSignalizedJunctionLeftTurnEnterFlow / Town13_route_003128_route0_01_08_20_19_21<br>f127–143，anchor 130 | DECELERATE → KEEP | 路口左转结束并入住宅道路，当前之后先11-12再降7-8。保留DECEL；输入前方比较空但不能把后续调速自动判错标，原因仅凭RGB无法确定。 |
| [57](probe_output/rgb_review_20260926/case_057.jpg) | OppositeVehicleTakingPriority / Town13_1418_2_route0_01_09_08_53_46<br>f42–58，anchor 45 | STOP → DECELERATE | 雨雾停车标志路口，自车进入持续近停/蠕行等待，STOP成立；远方优先车辆身份无法从缩略确认，需放大而不凭标志推事件。 原始分辨率四帧复核：浓雾中路口与STOP标志清楚，关键优先车辆仍不可可靠辨认；不因未看清自动判INVALID。 |
| [58](probe_output/rgb_review_20260926/case_058.jpg) | ConstructionObstacleTwoWays / Town15_route_001089_route0_01_08_09_04_56<br>f5–21，anchor 8 | RESUME → KEEP | 同类树荫道路跟车，自车短暂调速后到8.4保持，比锚点6.13有持续增速，RESUME成立，视觉上与KEEP接近。 |
| [59](probe_output/rgb_review_20260926/case_059.jpg) | AccidentTwoWays / Town12_route_002726_route0_01_08_20_35_19<br>f140–156，anchor 143 | KEEP → DECELERATE | 右侧停放车和远处事故/迎面车辆，自车2秒内下降未越阈值，2.5秒后才明显减速。保持KEEP并记录窗外变化，不按完整避障流程统一DECEL。 |
