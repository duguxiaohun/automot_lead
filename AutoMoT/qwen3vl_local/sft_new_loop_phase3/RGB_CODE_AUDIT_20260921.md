# Phase3 逐帧 RGB / 标定 / 提示词代码审计（2026-09-21）

## 结论

问题同时来自**局部真实标定错误、提示词边界不够准确、以及预测任务自身的可观测性与窗口限制**。本轮有证据支持的改动已落到 v20；没有模型重训，不能声称准确率已经提升。

- **确认两处错误横向监督来源**：InterurbanActorFlow 的 f30、MergerIntoSlowTraffic 的 f23，把相邻 lane section 中连续车道的 1→2 重编号误判成 RIGHT。RGB、原始坐标与本地 XODR 的 predecessor 链接一致支持该结论。精确隔离受影响窗口，不改成 KEEP/NO。
- **修正共享速度提示语**：低速单帧不等于继续等待；立即起步与继续等待后才释放需要区分。速度变化以当前速度为基准；显著减速后回升仍可属于 DECELERATE。两种题型共享。
- **多数精选错例不能直接认定为错标**：先减速再左跨线的主要动作仍是 LEFT；已完成跨线后可 KEEP；存在风险、制动请求、前车离开均不单独决定答案。
- **保留尚未解决的限制**：STOP 与 DECELERATE/KEEP 的固定窗口边界、零速抖动与立即起步，以及部分暗光/未来才出现的交通变化。没有用这批测试预测调阈值，也没有把夜间或遮挡统一过滤。

前一份四组指标及历史对比见 [AUDIT_COMPARISON_20260921.md](AUDIT_COMPARISON_20260921.md)。本报告补充逐帧证据及代码修复，原四组分数保持不变。

## 证据范围与方法

主要逐帧样本来自 C：`sft_new_loop_phase3_20260921_044359_4rgb_choice_audit_bundle`。47 道错题（含补查 #200/#207）与十类各一道正确对照，共 **57 个片段、44 条采集路线、964 个有效帧面板、874 张不重复原始 RGB**。每段四张实际输入图，另查看未来逐帧至 +3.25s；末帧仅作跨线确认。#12 缺三张额外尾帧、#282 缺两张，均不影响完整的纵向标注窗口。

先按异常时长规则检查全部 138 条 C 测试路线，再准备证据。实际输入 RGB SHA256 和原始未来速度均与包内记录一致；每张已看帧保留 RGB/meta 指纹。地图疑点另看原分辨率图，并读取本地 `lead/3rd_party/fail2drive/toolbox/carla_xodr/Town12.xodr`、`Town13.xodr`。未修改源 RGB/meta，未下载模型或数据。

这是 Codex 对图像的定向视觉复核，**不是独立人工真值、随机错标率估计或全量逐帧审计**。只把实际看过的帧登记为已看。未来图像用于离线核验，未进入模型提示词。

修复前另对 C 全部 **320 道题**回读原 meta：原始动作与主要动作投影均为 **0 数值重算不一致**。这一结果说明导出/规则执行一致，不能证明规则代表的物理语义全部正确；两处 section 重编号正是反例。

逐例观察及帧级哈希见 [rgb_review_notes_20260921.jsonl](rgb_review_notes_20260921.jsonl)。拼图与原始验证产物位于 `probe_output/rgb_review_20260921/`、`rgb_controls_20260921/`、`rgb_extra_20260921/`，仅本地产物、不入库。

## 1. 标定规则：确认的地图重编号问题

| 案例 | 旧目标 | 原始证据 | 地图验证 | 处理 |
|---|---|---|---|---|
| #34 InterurbanActorFlow f27 | RIGHT | f29 lane1 → f30 lane2；分界线在左侧展开，未见主动跨线 | Town13 road720，边界 s=80.4m；f29/f30 投影 s=80.444252/77.746333m；高 s section 的 lane+1 predecessor 为低 s section 的 lane+2 | 含 f30 未来切换的横向窗口未知 |
| #144 MergerIntoSlowTraffic f22 | RIGHT | f22 lane1 → f23 lane2；x 只变化0.003906m，沿护栏旁直行 | Town12 road20228，边界 s=74.991485m；f22/f23 投影 s=76.895010/74.772692m；相同1→2连续链接 | 含 f23 未来切换的横向窗口未知 |

对应路线：

- `InterurbanActorFlow/Town13_Rep0_1304_2_route0_01_10_10_53_07`
- `MergerIntoSlowTraffic/Town12_Rep0_route_002661_route0_01_09_02_51_11`

原始 meta 缺 `section_id`，旧检查把两侧 `None == None` 当作没有 section 变化，再用 lane-id 差值判 RIGHT。原检查能处理显式 section 变化，但不能识别上述缺字段重编号。

本轮在 [lateral_rgb_uncertainties_v1.jsonl](lateral_rgb_uncertainties_v1.jsonl) 增加精确 transition 审计记录，通过既有 `lateral_observation_complete=False` 路径阻止不可靠机动域监督；不把未知转成训练用 NO/KEEP。相同采集的 f18–29、f11–22 是各自可能受影响的 anchor 范围；实际有多少候选取决于上游事件映射。已经到达切换帧的 anchor 不因历史切换再被屏蔽，其他路线不受这两条记录直接影响。

这是**两处已验证问题的修复，不是全地图拓扑问题已解决**。没有把“缺 section_id 的所有变道”一律删除，否则会丢掉有效借道。未为生产构建新增对 `lead/` 地图文件的隐式依赖。地图文件指纹、坐标和链接证据见 [lane_topology_evidence_20260921.json](lane_topology_evidence_20260921.json)。

## 2. 提示词：等待、起步与速度基准

旧句 `current waiting still counts even if ego moves off later` 没有区分“等待继续保持后再释放”和“当前恰好低速、接下来立即释放”，容易与规则的判断边界冲突：

| 对照 | RGB与速度事实 | 合理目标 |
|---|---|---|
| #252 HardBreakRoute f75 | 四张历史都近零、蓝色前车离开；下一帧0.537m/s，随后持续增速 | RESUME |
| #162 路口 f18 | 警车横过后离开；当前0.010，下一帧0.611，随后持续增速 | RESUME |
| #248 同#267路线更早 | 当前仍等待，下一帧继续近零，之后才释放 | STOP |
| #267 同路线f41 | 当前0.157，下一帧1.022并继续增加；绿灯变化主要发生在未来 | RESUME，兼有输入可观测性限制 |
| #0 HardBreakRoute f90 | 1.533→0.005→0.527→1.530…；只有瞬态近停 | DECELERATE |

v20 的共享提示语明确：STOP 是即时持续近停/继续等待，低速本身不证明继续等待；立即起步可以是 RESUME。另明确相对当前速度判断首次显著变化，先减速后恢复不能仅因最终恢复就改为 KEEP。

这属于**消除语义歧义的有依据修订**；仅凭错例不能证明旧句是所有混淆的因果根源。需要同数据、同训练预算的提示词对照才能量化效果。

[本轮代码](prompts.py) 保留条件性动作目的、KEEP 的阶段内轻微调速/风险未清空含义，保留 STOP > 首次跨线 > 速度的主要动作优先级；不把未来秒数、阈值、速度曲线或控制变量放进 prompt。删除一条重复的 binary 轻微调速说明，继续通过原610英文词预算检查（不是token计数）。

## 3. 骑行者与开门案例：不能把准备动作或事件目的当成主要动作

- **#200 / #89 HazardAtSideLane，同一路线 f49 / f52**：右侧骑行者可见，同时有黑车从左侧进入前方。ego先减速/短暂近停，随后f60从lane2左跨入lane1。binary可同时有DECELERATE与LEFT；choice按主要动作投影选择LEFT。#200预测DECELERATE是只命中了准备动作，#89预测KEEP漏了之后跨线。
- **#207 同路线 f63**：左跨线已经完成，后续在lane1约9.9m/s继续行进。KEEP成立，不能看到右侧骑行者就推出立即右归。
- **#172 VehicleOpensDoorTwoWays f43**：后续确有向左跨中心线绕行。开门/跨线的部分变化出现在未来，当前输入预测有难度，但不能据此把已有未来动作改为KEEP。
- **#4 VehicleTurningRoute f207**：后续静止支持STOP；暗光原图不足以让我可靠确认骑行者/行人前提。保留视觉不确定性，不把“看不清”当作事件不存在，也不直接制造invalid负例。

## 4. 剩余窗口与可观测性限制

现有 v8 数值判定未改：STOP立即窗口、速度窗口与跨线窗口不同；速度阈值相对当前速度，增速需要连续确认。以下是规则与日常动作描述的边界，不宜靠提示词承诺全部解决：

- #125 停牌前近停/爬行，连续近停确认落在立即窗之外，数值标签DECELERATE，视觉语义容易答STOP。
- #192 短暂爬行后等待，增速未连续确认，停车确认越过立即窗，得到KEEP。需要作为边界难例单独看，不能将其描述为明确“畅通巡航”。
- #225 降幅仅略越减速阈值，肉眼可能视为随行波动。
- #103 面对对向车，采集车辆实际加速；任务预测记录行为，不能直接拿“应当让行”的规范驾驶建议重新标为DECELERATE。
- 暗光、未来才出现的交通变化、0.25s尺度上的起步/停车抖动，仅凭0.75s图像历史与当前速度不一定能可靠预测。增加提示文字不能补足不可见信息。

C全320题的原有边界诊断结果如下；桶可重叠，正确率不是错标率，也不能单凭相关性证明原因。

| 离线诊断桶 | 题数 | 正确 | 正确率 |
|---|---:|---:|---:|
| `all` | 320 | 215 | 67.19% |
| `no_boundary_flag` | 209 | 143 | 68.42% |
| `isolated_near_stop_in_1_5s` | 20 | 6 | 30.00% |
| `first_drop_single_sample` | 10 | 5 | 50.00% |
| `isolated_gain_present` | 32 | 17 | 53.12% |
| `near_drop_threshold` | 5 | 3 | 60.00% |
| `gain_unconfirmed_at_2s_boundary` | 11 | 8 | 72.73% |
| `stop_pair_crosses_1_5s_boundary` | 2 | 1 | 50.00% |
| `speed_before_crossing` | 44 | 39 | 88.64% |

特别是孤立近停桶只有6/20正确，值得优先区分短暂近停、继续等待与立即释放；而“速度动作先于跨线”桶39/44正确，不能笼统认为主要动作投影整体失效。

## 5. 四个模型在相同物理片段上的表现

A=4RGB binary；B=2RGB binary；C=4RGB choice；D=2RGB choice。binary允许合法纵横同时YES，不能直接用多标签文本与choice单标签做等价比较；LEFT/RIGHT为缩写。

| C案例ID | C主要目标 | A预测 | B预测 | C预测 | D预测 |
|---|---|---|---|---|---|
| #0 | DECELERATE | RESUME | STOP | STOP | STOP |
| #23 | KEEP | DECELERATE | DECELERATE | DECELERATE | DECELERATE |
| #34 | RIGHT | DECELERATE | DECELERATE | LEFT | KEEP |
| #89 | LEFT | LEFT | DECELERATE+LEFT | KEEP | KEEP |
| #144 | RIGHT | RESUME | RESUME | KEEP | KEEP |
| #162 | RESUME | STOP | STOP | STOP | STOP |
| #192 | KEEP | RESUME+LEFT | RESUME | RESUME | LEFT |
| #200 | LEFT | LEFT | LEFT | DECELERATE | KEEP |
| #207 | KEEP | KEEP | KEEP | RIGHT | RIGHT |
| #252 | RESUME | STOP | RESUME | STOP | STOP |
| #267 | RESUME | STOP | STOP | STOP | STOP |

此表沿用旧包目标；#34/#144已经确认不应作为可靠RIGHT监督，不能以“谁碰巧预测旧标签正确”决定哪一模型更好。没有删除旧测试错题后重新宣称模型提升。

## 6. 已落地修改、验证与使用方式

- [prompts.py](prompts.py)：`v20_grounded_motion`，共享速度语义修订；默认题型仍4RGB choice。
- [lateral_rgb_uncertainties_v1.jsonl](lateral_rgb_uncertainties_v1.jsonl)：两个精确拓扑冲突；纵向数值阈值、首次真实跨线方向规则不改。
- [development_route_groups_20260921.json](development_route_groups_20260921.json)、[build_dataset.py](build_dataset.py)、[source_mapping.py](source_mapping.py)：新增223个本轮已导出test/val物理组（154 test + 69 val），累计1454组train-only，Rep/采集时间不能绕过；新文件纳入mapping哈希。
- 训练、评估与shell新入口默认索引改为`checkpoints/sft_new_loop_phase3_data_v20`；旧索引不覆盖复用。
- [prepare_error_review.py](prepare_error_review.py)：审计面板中未记录的brake/hazard/throttle保持未知，不能补False/0；本次964个帧面板对应这些原始字段均有值，既有结论未因此变化。
- [test_rgb_review_20260921.py](test_rgb_review_20260921.py)：精确窗口隔离、真实跨符号借道仍有效、立即释放与继续等待、曝光路线和缺失审计字段回归；现有提示词测试同步新合同。

验证：

- **482项Phase3无需PyTorch测试通过**；3个依赖PyTorch的测试文件未收集，另4个测试实例未运行。初次尝试受本机缺PyTorch阻塞，不把这些计作通过。**9项Action数据准备测试通过**；额外Action主要动作索引测试因依赖PyTorch未运行。
- **44条已审路线局部重建**：候选1762→1756，删除6个包含上述拓扑切换的机动候选；其余候选动作无变化、无新增候选。平衡索引384行，全部回读原meta速度/被问动作和合同通过；binary/choice × 4RGB/2RGB共1408次目标渲染/解析重放通过。
- 局部集合全部train-only；要求完整train/val/test的生产`preflight`会正确拒绝该冒烟索引。没有为了通过检查伪造val/test或关闭生产守卫；此处报告的是逐行原始数据与合同核验，不是全量生产预检通过。
- 本轮未加载Qwen、未跑GPU训练/推理、未重建全量生产索引。仍不能判断v20真实准确率、best守卫或production_ready是否改善。

局部复现产物见 [verification.json](probe_output/v20_grounded_motion_20260921/verification.json)。为了使复核可重做，本地证据目录还保存了选定路线的collection子集与新索引；未把原始数据提交到仓库。

从`AutoMoT/`运行新训练入口：

```bash
# 默认4RGB + choice；重建v20全量索引并训练、评测
bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```

若环境显式设过`INDEX`或`DATA_DIR`，应指向v20；首次不要设置`SKIP_BUILD=1`。旧v19模型/断点应使用原源码与原合同恢复，不能直接拿新提示词评旧adapter并当成训练对照。Action的共享动作描述文本本轮未变，但上游映射/候选哈希变化后，新的oracle数据产物仍需按现有合同重新准备。

新版本收益应在未曝光的路线中验证。要区分提示语与数据修复的贡献，可用同一份新划分、新标定数据、相同预算分别训练旧/新速度提示语；不要把本轮已审57题继续当盲测。无需为启动训练另设人工负例数量门槛。

## 7. 逐例观察索引

以下均为已逐帧看过的实际有效帧；完整路线、输入/未来区分、每帧SHA及原始数值在JSONL中。`保留`表示没有足够证据改标签，不等价于逐帧意图真值已证明。

| C案例 | 场景 / anchor | 旧GT → C预测 | 观察与决定 |
|---|---|---|---|
| [#0](probe_output/rgb_review_20260921/case_000.jpg) | HardBreakRoute f90 | DECELERATE → STOP | 蓝色前车离开；ego短暂接近零速后立即增速，未形成连续近停，DECEL而非STOP；瞬态边界难例。 |
| [#4](probe_output/rgb_review_20260921/case_004.jpg) | VehicleTurningRoute f207 | STOP → RESUME | 夜间可见对向蓝车与路侧红色信号，后续ego持续静止，STOP动作成立；原图0207也不能可靠确认行人/骑行者前提，保留场景证据不确定，不把不可见改成invalid。 |
| [#8](probe_output/rgb_review_20260921/case_008.jpg) | CrossJunctionDefectTrafficLight f33 | DECELERATE → STOP | 暗光路口后续横穿白车；ego降速后恢复，没有持续近停；DECEL成立，视觉可观测性较弱。 |
| [#9](probe_output/rgb_review_20260921/case_009.jpg) | HardBreakRoute f32 | STOP → KEEP | 接近灰色前车后持续静止；未来RGB和速度支持STOP，KEEP漏报。 |
| [#12](probe_output/rgb_review_20260921/case_012.jpg) | CrossJunctionDefectTrafficLight f44 | KEEP → DECELERATE | 暗光路口继续行进、速度小幅波动；KEEP成立，不表示风险清空。额外横向确认帧缺失不影响已完整纵向窗口。 |
| [#16](probe_output/rgb_controls_20260921/case_016.jpg) | HighwayCutIn f43 | KEEP → KEEP | 正确KEEP对照：侧方紫车并行/切入过程中ego保持约6.9和原车道，风险存在不等于必须变速。 |
| [#18](probe_output/rgb_review_20260921/case_018.jpg) | HardBreakRoute f42 | STOP → RESUME | 白色前车仍在前方，ego后续静止；STOP成立，不应凭前车位置猜RESUME。 |
| [#20](probe_output/rgb_controls_20260921/case_020.jpg) | ConstructionObstacleTwoWays f66 | RESUME → RESUME | 正确RESUME对照：双向路对向车通过，ego在原车道持续增速，不需要新的跨线动作。 |
| [#21](probe_output/rgb_review_20260921/case_021.jpg) | NonSignalizedJunctionLeftTurnEnterFlow f40 | STOP → RESUME | 无灯路口停牌、红车横过，ego后续继续近停；STOP成立，出现间隙不能替代实际起步证据。 |
| [#23](probe_output/rgb_review_20260921/case_023.jpg) | HardBreakRoute f20 | KEEP → DECELERATE | 灰色前车前方行驶，ego速度围绕当前值波动；KEEP成立，不能从历史最高速度计算未来减速。 |
| [#24](probe_output/rgb_controls_20260921/case_024.jpg) | CrossJunctionDefectTrafficLight f26 | STOP → STOP | 正确STOP对照：路口横向SUV驶过，当前与下一帧继续近停，随后才起步；与#162立即释放区分。 |
| [#27](probe_output/rgb_review_20260921/case_027.jpg) | ConstructionObstacleTwoWays f148 | KEEP → LEFT | 对向蓝车通过，ego始终留在原车道；KEEP，静态障碍场景不必然触发新的LEFT。 |
| [#28](probe_output/rgb_controls_20260921/case_028.jpg) | HazardAtSideLane f32 | LEFT → LEFT | 正确LEFT对照：城市路右侧骑行者可见，ego从lane2左跨入lane1后通过。 |
| [#34](probe_output/rgb_review_20260921/case_034.jpg) | InterurbanActorFlow f27 | RIGHT → LEFT | 道路展开时分界线在左侧出现，ego未见主动跨越分界；f30 lane1→2。XODR road720 s80.4两侧是连续lane1/2，属于section重编号；隔离包含该切换的横向窗口。 |
| [#36](probe_output/rgb_controls_20260921/case_036.jpg) | HighwayExit f110 | KEEP → KEEP | 正确KEEP对照：雨雾高速多车，ego跟随蓝车、约8稳定，无新跨线。 |
| [#37](probe_output/rgb_review_20260921/case_037.jpg) | NonSignalizedJunctionLeftTurn f67 | DECELERATE → KEEP | 夜间弯转道路，ego从11.31显著降到约5.53；数值支持DECEL，暗光下运动证据较弱，保留标签。 |
| [#38](probe_output/rgb_review_20260921/case_038.jpg) | MergerIntoSlowTrafficV2 f34 | DECELERATE → RESUME | 右侧黑车切入到前方；ego短暂急减速后恢复。输入历史仍加速不能外推为未来RESUME；先发生的减速成立。 |
| [#40](probe_output/rgb_controls_20260921/case_040.jpg) | HardBreakRoute f58 | STOP → STOP | 正确STOP对照：灰色前车前ego持续静止，图像和速度一致。 |
| [#47](probe_output/rgb_review_20260921/case_047.jpg) | ConstructionObstacleTwoWays f96 | RESUME → LEFT | 暗光道路保持lane1，ego明显持续增速；RESUME成立，不能由施工场景自动推出LEFT。 |
| [#51](probe_output/rgb_review_20260921/case_051.jpg) | HazardAtSideLaneTwoWays f94 | KEEP → LEFT | 右侧骑行者、左侧对向车；ego低速随行且不跨线；KEEP可与风险并存。 |
| [#62](probe_output/rgb_review_20260921/case_062.jpg) | Accident f69 | LEFT → RIGHT | 夜间多车事故场景，ego后续从lane3进入lane2；可见左侧虚线跨越，LEFT成立，不能选相反方向。 |
| [#64](probe_output/rgb_controls_20260921/case_064.jpg) | HazardAtSideLaneTwoWays f48 | KEEP → KEEP | 正确KEEP对照：停牌路口通过后速度围绕当前6.61波动；有制动请求也不必构成新的减速动作。 |
| [#79](probe_output/rgb_review_20260921/case_079.jpg) | VehicleTurningRoute f65 | DECELERATE → KEEP | 雾中接近路口/转弯并与车辆交互，先从7.93降到5.59再回升；不能因最终回升而改KEEP。 |
| [#89](probe_output/rgb_review_20260921/case_089.jpg) | HazardAtSideLane f52 | LEFT → KEEP | 右前方骑行者，黑车从左侧进入前方；先急减速后f60左跨线lane2→1，choice LEFT与binary DECEL可同时合理。 |
| [#91](probe_output/rgb_review_20260921/case_091.jpg) | InvadingTurn f69 | DECELERATE → KEEP | 对向侵入车相继通过；ego先增后明显降速。以当前17.96为基准未确认显著增速而确认后续下降，DECEL成立。 |
| [#103](probe_output/rgb_review_20260921/case_103.jpg) | InvadingTurn f60 | RESUME → DECELERATE | 与#91同路线更早时刻，对向车接近并通过，ego实际持续加速；RESUME是采集行为，不能用应当让行的安全建议改成DECEL。 |
| [#107](probe_output/rgb_review_20260921/case_107.jpg) | OppositeVehicleRunningRedLight f22 | KEEP → DECELERATE | 多车路口继续行进，ego约6.43稳定；KEEP成立，事件文字不保证需要减速。 |
| [#108](probe_output/rgb_review_20260921/case_108.jpg) | InvadingTurn f69 | STOP → DECELERATE | 对向车接近时ego减速并短暂持续近停后释放；STOP符合确认规则，不因最终起步抹去先前停车。 |
| [#122](probe_output/rgb_review_20260921/case_122.jpg) | ParkedObstacleTwoWays f51 | DECELERATE → KEEP | 暗光双向路，前车和对向车交互；6.38降至短暂近零再恢复，DECEL成立；可见性弱且瞬态近停。 |
| [#124](probe_output/rgb_review_20260921/case_124.jpg) | InvadingTurn f79 | DECELERATE → KEEP | 对向车已通过，ego先波动后持续减速并在较晚处停车；STOP发生在立即窗之外，DECEL为当前规则结果，时间边界局限。 |
| [#125](probe_output/rgb_review_20260921/case_125.jpg) | InterurbanAdvancedActorFlow f46 | DECELERATE → STOP | 可见停牌及横向车流，ego近停/爬行再等待；持续近停的确认落在立即窗外，DECEL数值一致但自然语义容易答STOP，保留边界诊断。 |
| [#140](probe_output/rgb_controls_20260921/case_140.jpg) | VehicleOpensDoorTwoWays f64 | LEFT → LEFT | 正确LEFT对照：门开启、对向车通过后ego跨中心线左绕，lane1→-1；避免把有效跨符号借道屏蔽。 |
| [#144](probe_output/rgb_review_20260921/case_144.jpg) | MergerIntoSlowTraffic f22 | RIGHT → KEEP | 沿护栏旁固定直线前进，f23 lane1→2时x仅变化0.003906m；XODR road20228 s74.991485两侧lane1/2有连续链接，属于section重编号；隔离横向窗口。 |
| [#151](probe_output/rgb_review_20260921/case_151.jpg) | DynamicObjectCrossing f34 | DECELERATE → KEEP | 前车行驶、左侧来车通过，ego先降至4.61后回升；DECEL成立，最终回升不代表KEEP。 |
| [#160](probe_output/rgb_review_20260921/case_160.jpg) | HazardAtSideLaneTwoWays f44 | RESUME → KEEP | 夜间对向车通过、右侧骑行对象不清晰，ego先持续增速再制动；RESUME满足先后规则，可见性有限。 |
| [#162](probe_output/rgb_review_20260921/case_162.jpg) | OppositeVehicleRunningRedLight f18 | RESUME → STOP | 横过警车离开；当前近零但下一帧已起步，随后持续增速，RESUME成立；单帧低速不等于继续等待。 |
| [#172](probe_output/rgb_review_20260921/case_172.jpg) | VehicleOpensDoorTwoWays f43 | LEFT → KEEP | 停靠车辆/开门风险，ego后续跨中心线向左绕行；LEFT成立，当前到实际跨线有准备阶段，不强改为KEEP。 |
| [#181](probe_output/rgb_review_20260921/case_181.jpg) | InvadingTurn f78 | KEEP → DECELERATE | 夜间锥桶对向车场景，速度随行波动但对当前基准未出现新动作；KEEP成立，不应仅因已制动/风险而选DECEL。 |
| [#184](probe_output/rgb_controls_20260921/case_184.jpg) | InvadingTurn f58 | DECELERATE → DECELERATE | 正确DECEL对照：对向车接近时ego实际从8.57降到6.16，减速成立。 |
| [#192](probe_output/rgb_review_20260921/case_192.jpg) | AccidentTwoWays f21 | KEEP → RESUME | 双向事故场景，对向蓝车通过；ego先短暂爬行后较晚近停。增速未连续确认、停车确认越过立即窗，KEEP数值成立但阶段边界语义不稳。 |
| [#200](probe_output/rgb_extra_20260921/case_200.jpg) | HazardAtSideLane f49 | LEFT → DECELERATE | 与#89同一骑行者路线更早帧：先减速/让过左侧黑车，后f60左跨线；choice LEFT成立，预测DECEL只覆盖准备动作。 |
| [#202](probe_output/rgb_review_20260921/case_202.jpg) | CrossingBicycleFlow f24 | DECELERATE → KEEP | 接近路口、远处骑行者横过，ego由6.35逐渐降至4.74；DECEL成立，未来更晚进一步降速。 |
| [#207](probe_output/rgb_extra_20260921/case_207.jpg) | HazardAtSideLane f63 | KEEP → RIGHT | 同路线f63已经进入lane1，随后在lane1稳定约9.9行驶并经过骑行者；KEEP成立，右侧骑行者不代表要立即RIGHT。 |
| [#210](probe_output/rgb_review_20260921/case_210.jpg) | HardBreakRoute f89 | RESUME → STOP | 蓝色前车离开，当前孤立近零后即刻起步；未来又有一次瞬态近零但随后增速，RESUME符合原规则，STOP提示语容易过度泛化。 |
| [#216](probe_output/rgb_review_20260921/case_216.jpg) | HighwayExit f69 | RIGHT → KEEP | 多车拥堵道路，ego连续向右从lane2到3再4；首次右跨线成立，选择RIGHT。 |
| [#221](probe_output/rgb_review_20260921/case_221.jpg) | VehicleTurningRoute f14 | STOP → DECELERATE | 暗光跟车接近路口，未来持续近停；STOP成立，DECEL遗漏停车阶段。 |
| [#225](probe_output/rgb_review_20260921/case_225.jpg) | HighwayExit f95 | DECELERATE → KEEP | 雨雾跟车，9.30降至7.38，较减速阈值只略超；DECEL数值成立，肉眼易当作随行波动的阈值边界。 |
| [#231](probe_output/rgb_review_20260921/case_231.jpg) | MergerIntoSlowTrafficV2 f28 | STOP → RESUME | 右侧车辆接连汇入，ego先近停后释放；STOP成立，输入中的增速不保证后续继续增速。 |
| [#234](probe_output/rgb_review_20260921/case_234.jpg) | InvadingTurn f46 | STOP → RESUME | 对向来车、锥桶场景，ego先短暂持续近停再起步；STOP成立，之后增速不撤销已发生的近停。 |
| [#243](probe_output/rgb_review_20260921/case_243.jpg) | ParkedObstacle f80 | LEFT → RIGHT | 夜间跟车，ego向左跨入lane3，LEFT成立；恢复场景不自动表示RIGHT，不能由动作目的替代首次方向。 |
| [#247](probe_output/rgb_review_20260921/case_247.jpg) | ConstructionObstacleTwoWays f54 | DECELERATE → LEFT | 前方施工阻挡、对向车通过；ego先显著减速，停车确认在立即窗外；DECEL成立，场景不必然先LEFT。 |
| [#248](probe_output/rgb_controls_20260921/case_248.jpg) | OppositeVehicleRunningRedLight f38 | STOP → STOP | 正确STOP对照：与#267同路更早时刻，当前继续近停，后续绿灯释放不撤销STOP。 |
| [#252](probe_output/rgb_review_20260921/case_252.jpg) | HardBreakRoute f75 | RESUME → STOP | 蓝色前车开始离开，四帧历史ego都近零，下一帧即起步并持续增速；RESUME成立，历史等待不能强制当前继续STOP。 |
| [#267](probe_output/rgb_review_20260921/case_267.jpg) | OppositeVehicleRunningRedLight f41 | RESUME → STOP | 红灯路口即将转绿，ego近零后立即起步；RESUME是记录结果，变化主要在未来，可观测性限制明显。 |
| [#282](probe_output/rgb_review_20260921/case_282.jpg) | CrossJunctionDefectTrafficLight f45 | DECELERATE → KEEP | 路口侧向车流，ego先减速后回升；DECEL成立。额外+3秒尾帧缺失，完整纵向证据保留。 |
| [#298](probe_output/rgb_review_20260921/case_298.jpg) | EnterActorFlowV2 f87 | KEEP → LEFT | 白色侧车并行，ego当前保持原车道且约20.37稳定；KEEP成立，未来更晚减速不回写当前动作。 |
| [#311](probe_output/rgb_review_20260921/case_311.jpg) | NonSignalizedJunctionRightTurn f59 | STOP → RESUME | 停牌前红车连续横穿，ego先短停后右转起步；STOP成立，输入正在起步不代表未来不会再次等待。 |
