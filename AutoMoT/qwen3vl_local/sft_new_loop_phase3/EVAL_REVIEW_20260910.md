# Phase3 v6：逐帧 RGB、动作标定与提示词联合审计（2026-09-10）

**当前较低成绩同时包含真实模型错误和监督问题，不能归因于单一的“数据噪声大”或“提示词不够详细”。** 已确认高速 R4 错前提、出口 lane-id 伪变道；另有首事件/时间阈值、未来放行不可见、弱光与绕出/归位阶段混淆。优先修正监督一致性和可观测性，再做固定预算消融。

本报告延续 [原始成绩摘要](AUDIT_SUMMARY_20260910.md)，保留 **470/767，61.28%** 原始分数，不根据看过的错误改标签后重算一个“提升分数”。分析的是离线高层动作问答，不是 CARLA 闭环成绩。

## 1. 实际看了什么，哪些工作只是机器核对

- 逐帧目视 **67 个 production 错例窗口、54 条采集 run、948 张去重原始 RGB**；其中这些窗口首行实际输入去重为 258 帧。覆盖十个 context，并额外检查 INVALID。选择以错误和边界为导向，**不能把其中的错标比例推广为数据集噪声率**。
- 审计包 `lora_production_audit_samples/` 有 45 个目录、44 个去重 case；44 个的实际输入及至少 2 秒未来全部看过。#438/#465/#671 与 #518/#213 共用连续路线证据，不重复冒充独立路线。其余 FULL_MANEUVER 窗口也查看到 3 秒。
- 拼图从本地 `AutoMoT/lead_data/<Scenario>/<run>/rgb/` 的 **1152×384 原始三视角拼接图**制作，逐格按时间阅读；不是只读包内自动生成的 `audit_note.md`。四列首行是模型实际四帧输入，后三行是未来取证，**未来图没有输入模型**。
- 关键争议另开原图：#255 的 f86–89，#613 的 f35–39，#518 的 f44，#43 的 f127；#518 同时核对包内压缩 f44。拼图显示会缩小，低可见性样本没有靠自动增亮、图像补全或生成图猜测目标。
- 先用既有异常路线规则检查评测涉及的 242 个 run：异常剔除命中 0、本地缺失 0。全评测为 767 题、727 个去重锚点，297 题错误。
- **机器核对**：640 个有效题均能从包内精确速度与横向判据重算原 GT；67 个目视窗口的 9 个纵向速度样本均与本地 raw meta 一致；67 个实际 `history_rgb_paths_used` 的帧号都与首行一致。它们证明序列/合同可复现，**不证明所有 GT 语义正确，也不是全量视觉审计**。
- #20 的 f97–98、#173 的 f49–52 不存在，均为本次扩展到 3 秒后超出路线末尾的取证格；其 2 秒纵向窗口完整。拼图已明确写 `SOURCE FRAME DOES NOT EXIST`，不把缺图当全黑 RGB 或零速。

当前源码与包内身份一致：

| 合同 | 值 |
|---|---|
| 动作规则 | `current_wait_first_crossing_v6` |
| 动作源码 SHA256 | `028adf55f74a12711ec6032125b945b32e32d2fd3d35078e9c7611d3795e24a0` |
| production prompt | `sft_new_loop_phase3_high_level_action_v5_current_phase` |
| prompt SHA256 | `8e1ce857bf3c0f367574dd422f2401005b178cb148ece725451d2ba91a84337f` |
| mapping SHA256 | `9cdd2c60cbd9f7141065babdcaf7cf8da22ae731ca99bf0fa8b93d2a145e1951` |
| 权重/图像 | final step 9216，4rgb |

本轮没有加载模型重跑推理，没有修改训练代码、标注源、原始 RGB 或 checkpoint。新增审计文档/笔记和本地取证工具，修正摘要归因。

## 2. 已确认：DYNAMIC_CUTIN 的 21 个 NONE 错例并非全部动作误触发

这 21 题分成 **13 题只答 INVALID + 8 题真的额外报动作**。原摘要把两类混在一起，会把修复方向带偏。

13 题来自同一条：

`HighwayCutIn/Town13_Rep0_1586_9_route0_01_08_19_01_08`，锚点 **f44–56**。

case 索引按帧为：`518,465,701,8,676,0,147,716,533,438,671,219,213`。

[连续证据 f41–56](probe_output/rgb_review_20260910/case_518.jpg) 与 [f53–68](probe_output/rgb_review_20260910/case_213.jpg)：

- f41–44：警车在左邻，接近车道分隔线。
- f45–52：警车逐渐接近/跨入 ego 邻近通行区域。
- f53–56：警车位于分隔线附近并向前。
- f57–68：警车继续向前，侵入阶段结束或远离。

道路是连续护栏多车道高速走廊，没有图像支持“本车由局部路口信号灯控制”。模型的五/三动作行全 NO，只有 `INVALID_ACTION_CONTEXT=YES`；其拒绝错误 R4 前提符合 prompt 的定义。**UE3 的切入补标并不是这组问题所在。**

归因已追到源文件，而不是仅凭视觉猜测：

1. `collection_output/HighwayCutIn_result.json` 的 f44 给 `R4=0.90`、`R3=0.78`，主 RS 取 R4；明确记录 `r4_light_hazard`、`decision_source=meta_light_hazard`。
2. 同条记录 `traffic_light_state="None"`、`traffic_light_count=0`、`traffic_light_affects_ego=false`；`is_junction=true`，但 CARLA highway junction 不能直接等价为信号控制局部路口。
3. [collector.py](../../keyframe_filter/collector.py) 的 `r4_light_hazard` 分支用 `light_hazard + light_hazard_control_context` 加 R4 分数。该例源记录已设 `review_required=true`，仍进入下游。
4. [build_dataset.py](build_dataset.py) 的 `_rs_label` 优先保留 `primary_road_structure`；[source_mapping.py](source_mapping.py) 的 `mapped_contexts` 追加人工 UE3，但没有同步复核/替换 RS，最终成为 **真实高速 RGB + 切入事件 + R4 错前提 + GT INVALID=NO**。
5. 本地 LEAD `expert.py` 中 `light_hazard` 是控制器目标速度分支的标志，不能单独作为可见交通灯存在/控制本车的证明。为什么采集时该 flag 为 true 尚未做同版本控制器状态重演，不能声称已找到其底层生成 bug。

此路线 source xodr 的 `xodr_topology_trusted=false`、投影误差 **3021.8 m**；不能用这份静态地图强行证明道路身份。

其余 8 题：Town06 f20/21/23/24/25、Town04 f27/28、Town10HD f113。已看对应完整连续帧，分别存在动态前提可见阶段存疑、雨夜邻车与阈值、历史减速被延续成未来停车。详见 #291/#509/#542/#582。不能把这 8 题也自动洗成标错。

## 3. 已确认：同 road_id 内仍存在 lane-id 伪变道

**#255 HighwayExit，f87，GT LEFT，模型 NONE。**

[16 帧连续证据](probe_output/rgb_review_20260910/case_255.jpg)。原始图中 f84–99 始终沿出口右侧车道自然向右弯曲，f86–89 逐张放大后没有向左跨越本车边界。

| 帧 | road_id | lane_id | 相对 f87 的位置 `(前,右)` m | steer |
|---|---:|---:|---:|---:|
| 86 | 186 | -4 | (-2.7472, +0.0070) | +0.0031 |
| 87 | 4520 | -4 | (0, 0) | +0.0199 |
| 88 | 4520 | -2 | (+2.7459, +0.0398) | +0.0294 |
| 89 | 4520 | -2 | (+5.4920, +0.1486) | +0.0284 |

f88 的 -4→-2 在下一帧确认后触发 LEFT，但同期车辆沿右弯平滑前进。位移/steer 只是与 RGB 相互印证，不是单独用它们定义变道。源事件证据还记 `changed_route=false`、`route_centered=true`。

[trajectory_action.py](trajectory_action.py) 的 `lateral_window_issue` 排除了跨 road、非 Driving 和缺失；`lane_change` 却仍把同 road 的 lane-id 排序变化当实际跨线，**没有 lane-section/successor 物理连续性核验**。锚点刚进入 road4520 后，后续窗口全为同 road，已有 guard 无法挡住本例。这是 prompt 明文要求“连接道路自然成为下一车道但未跨线应为 NO”与 GT 的直接冲突。

可以确认本例 LEFT 不符合 RGB 跨线定义；究竟是 section 重编号、junction waypoint 归属还是地图投影边界，仍需采集同源 xodr，不能只凭 id 跳两级断言具体地图机制。源静态 xodr 已不可信（投影误差 202.547 m）。

**#613 InterurbanActorFlow，f36，GT RIGHT，模型 LEFT** 是另一个高优先候选：[连续帧](probe_output/rgb_review_20260910/case_613.jpg)。f35–39 道路展宽、黄线变白线，ego 始终在延续道路右侧；f38 lane1→2，但相对 f36 横移约 **-0.362 m（略向左）**，随后回摆；没有清楚的向右物理跨线。GT RIGHT 缺少证据，但模型 LEFT 同样不能据此宣告正确，需将整个横向监督置入拓扑待审队列。

## 4. 源事件的“场景窗口”与 prompt 的“当前事件阶段”不同

**#334/#675 StaticCutIn，f7/f11**：[连续帧](probe_output/rgb_review_20260910/case_334.jpg)。f4–11 是桥上连续多车道跟车，f12 确实左跨 -3→-2；模型漏掉 LEFT 是记录层面的错误。但“当前 active ramp”前提缺乏对应可见出口、汇入口或车道并接。

源 `StaticCutIn_result.json` 的 f7 是 `event_highway_trigger_core_r3`：距 scenario trigger **15.21 m < 20 m**，并非确认当前正在汇入。该记录同时写 `highway_ramp_merge_split_hint=false`、`highway_scenario_merge_approach=false`、xodr 不可信。

[context_taxonomy.py](context_taxonomy.py) 对 RAMP 明确说：普通主路跟车或平行车流不是 active merge，主路导航变道应属 route-lane transition。比较 **#197**：同为可见连续主路，R3 正确、人为换成 RAMP 被标 INVALID。于是正样本与 same-RS 负样本可能按不同的事件阶段标准组织。

处理对象应是 **源 RE3 的激活证据/范围**，不能靠 scenario 名把整个 StaticCutIn 排除，也不能把全部 HighwayExit/EnterActorFlow 改 NO。#27 确有护栏夹窄的匝道跟车，#255 也确处于出口；它们的问题与 #334 不同。

## 5. 纵向标签通常能复现，但时间合同会制造语义边界

[trajectory_action.py](trajectory_action.py) 的实际规则：

- 4 Hz；STOP 取 f..f+6 内任意连续两点 ≤0.5 m/s；当前 f、f+1 都停则优先 STOP。
- 减速/恢复取 f..f+8，幅度 `max(1.2, 0.2*max(v0,1))` m/s。
- 减速一次越阈就成立，恢复需两次连续越阈；选择首次合格事件，部分先加速再制动的混合窗被隔离。
- FULL_MANEUVER 另外看最多 3 秒首跨，因此同题可组合“现在停等 + 近 3 秒窗末才归位”。

| RGB 已核实例 | 实际逐帧行为 | 对错误归因的影响 |
|---|---|---|
| #127，施工 f69 | f70 降到0.252；f71–75 加速至7.051，f78左跨 | GT DECEL+LEFT；模型 RESUME+LEFT描述了后段，但不符首事件口径。不是标签与速度文件错配 |
| #95，路口 f7 | f9最低0.545，之后起伏；f16起才连续停 | GT DECEL；STOP宏观可理解，但不符1.5秒/0.5阈值 |
| #492，高速 f99 | 9.712→7.759，降1.953，阈值约1.942 | 单个样本仅约0.011 m/s越阈，NONE/DECEL差异不能当明显视觉误判 |
| #43，路口 f127 | f128–134约11；f135尚9.477；f136之后才显著降速 | DECEL发生在2秒窗口之外，GT NONE有截窗依据 |
| #733，信号 f32 | 四输入全停；f33已0.649，随后启动，横穿车仍在离开 | GT RESUME依赖下一帧放行。模型 STOP 受当前可见冲突支持，存在不可提前确定的放行时点 |
| #684，施工 f157 | f157–164等待，f165启动，f169右归 | STOP+RIGHT混合了当前纵向与窗末横向，模型 RESUME跳过等待 |

系统问的是 **“ego should take next / now”**，GT 则是 **“采集控制器未来实际怎么开”**。两者在清楚避让时重合，在短暂急刹、启动边界和风格选择上不等价。

#632 的突然停车特别说明这个区别：f97–99 `vehicle_hazard=true` 且 brake=true，f100解除，速度也确实接近零；不能说速度字段乱跳。但当前 RGB 没有清楚到足以解释急停的近处阻挡。还需要 `vehicle_affecting_id`、预测 bbox/控制器路径确认具体原因；不能把记录过的每次急刹都定义为 RGB 下唯一应采取的动作。

建议先确定任务到底是有限时域行为预测还是可见证据下的规范决策。若保持行为预测，应把边界/首次事件置信度单列，并对小幅越阈做训练集上的敏感性审计；若改规范决策，则需要额外的可见性与决策监督。**不能仅调 test 阈值提高本次分数。**

## 6. 模型确实还存在的阶段、方向与强度错误

这些例子的动作 GT 有连续 RGB 支持，不能归咎于标注噪声：

- **绕出与归位相反**：#13 f42–45 已在骑行者左侧，f53后右归，模型仍 LEFT；#220 f73尚未左跨，f74先左、f82后右，模型先报 RIGHT。
- **障碍还没全通过就归位**：#42/#134 右侧仍有骑行者，#141 右侧施工锥筒未结束，后续都没有右跨，模型却 RIGHT。
- **横穿用户被当作需要绕行**：#190 和 #72 都是 ego 持续停等行人，模型除 STOP 又报 LEFT/RIGHT；不应把其他车流的横移解释成 ego 变道。
- **跳过等待**：#179 夜间施工前等对向车却报 LEFT；#228 STOP 标志清楚且继续停让，却报 RESUME；#442 先停再启动，模型直接 RESUME。
- **停车强度不足**：#254/#321 后续明确持续停等，模型仅 DECEL；#86 进入冲突路口后停车，模型 NONE。
- **纵向跟随历史而非预测下一段**：#20/#51 历史减速、未来恢复/稳速，模型仍 DECEL。

提示词已经包含“FIRST crossing”“当前等待优先”“不把曲线/其他车变道当自车变道”“不由终点左右推断近端车道”。继续堆叠相同句子缺少依据。更值得做的是沿独立路线构造**绕出前—跨线—通过未清空—归位待机—归位完成**的对照监督，并保持相邻阶段的正负例配对。

## 7. 四图输入的信息边界与成像问题

当前四图只覆盖 **0.75 秒历史**，却要预测 2 秒纵向/3 秒横向。比如 #462 f90 到首次右跨 f102 整整 3 秒，前四帧基本稳速跟车；#392 四图全静止，3秒窗尾才启动跨线。历史内没有明显动作趋势，不等于 GT 错，但会使任务成为含隐藏意图的预测。

`goal_ego_xy` 来自 `next_target_points[-1]`，是远端路线终点。#334 终点 `(193.52,+9.906)` 在右前，但真实先左跨；#462 终点在175米以外。它没有提供近端目标车道/下一路口路线选择。prompt 用“route target”同时询问 target-lane transition，容易让模型借终点方向填补缺失信息。应评估部署可获得的近端导航指令/车道目标；不能把未来真实 lane-id 偷喂作为修复。

**低可见性是真问题，但不等于错标。** #156/#243、#382/#429 几乎全黑；雨雾和车灯遮蔽使骑行者/车道线难辨。虽然索引 `include_visual_risk=false`，已有风险过滤没有排除所有不可判帧，应增加“具体任务目标是否可见”的审计项，而不是只看场景曾完成 RGB review。

另有 **冷启动非稳定图像**：#77 的 f0干亮、f1雨雾；#764 的 f0空路、f1突然出现前车；均在 f2 采用 f0,f0,f1,f2 左填充。可确认天气/actor渲染和有效历史不足，不能笼统说图像—meta错位。应单列 startup 桶，先在训练数据上核查渲染稳定窗口。

目前没有证据支持“所有左右都反了”“四RGB顺序整体反了”或“全数据时间戳整体错一帧”。许多真实跨线与 lane变化、速度阶段能够对应。相机外参/时间戳的全量标定误差本轮没有测量，不宣称已排除。

## 8. 评测/训练采样对成绩解释的影响

十类各64题不等于各64次独立事件：

| context | 正确/64 | 独立采集 run | 单run最多题数 | 有效GT却只答INVALID的题数 |
|---|---:|---:|---:|---:|
| DYNAMIC_CUTIN | 33 | 15 | 13 | 13 |
| LEAD_BRAKE | 49 | 7 | 17 | 0 |
| ONCOMING_INVASION | 41 | 10 | 13 | 0 |
| JUNCTION_RULE_CONFLICT | 51 | 15 | 15 | 0 |
| SIGNAL_FAILURE | 53 | 16 | 7 | 0 |
| RAMP_MERGE_EXIT | 22 | 24 | 9 | 4 |
| VULNERABLE_CROSSING | 19 | 24 | 6 | 1 |
| STATIC_BLOCKAGE | 28 | 45 | 4 | 0 |
| POST_BYPASS_RETURN | 28 | 48 | 4 | 0 |
| UNSIGNALIZED_PRIORITY | 35 | 51 | 3 | 1 |

这里 run 是采集标识，尚不等同于跨 scenario 地理去重后的物理道路。至少应报告 route/event-span 聚合，不能给相邻帧当独立伯努利样本计算置信度。

- LoRA 对 base：319题由错变对、56题由对变错、151题双方正确、241题双方错误。总体提升真实存在，但不代表每种情境都提升。
- production→audit prompt：51题由对变错、17题由错变对，净降34题。audit解释文本不是模型错误原因的独立真值；也不能证明只要改 prompt 就会改善。
- `train_balance.json` 的可用 INVALID 池为2052个 wrong-RS、30个 same-RS wrong-event；记录的训练采样中 same-RS仅30个独立case/8个run，占2048个INVALID约1.46%。因此87.40%的总INVALID成绩主要反映wrong-RS，不能代替细粒度事件拒绝能力。
- 当前 final 是无 best_generation 时按既定流水线打包的权重，guard没有全过；本轮没有用test选择另一个checkpoint。

源 UE3 审计记录中的 `expected_split=train` 属上游历史协议；Phase3使用自己的物理route切分，**不能仅凭这一个字段宣判Phase3训练测试泄漏**。本次新看过的test路线若用于下一轮规则/标签开发，应登记开发集合并提供新的未看holdout；原test可保留为固定回归集，但不再称全新盲测。

## 9. 建议的修订次序与验收方式

| 优先级 | 具体修改对象 | 本轮证据 | 验收方式 |
|---|---|---|---|
| P0 | RS前提与控制字段分离，隔离确认的高速R4冲突 | f44–56、collector来源链 | 先复核同类训练run；RGB/灯控证据联合判断，禁止仅由light_hazard或is_junction定R4 |
| P0 | 横向身份检测补物理连续性/同源地图依据 | #255确认冲突，#613待审 | 同road的section/分叉/新生车道专题；同时保住#13/#220等真跨线。未确认样本不回写横向NO |
| P0 | RE3的active transition门控与same-RS负例统一 | #334与#197 | 主路跟车/普通导航变道/真实匝道三组RGB配对，先审训练集；不按scenario整类改标 |
| P1 | 纵向“应然/实然”、第一事件、边界置信度 | #127/#95/#492/#733 | 在训练侧固定规则敏感性报告；阈值边缘、窗末动作、停后释放单列指标，不在test寻优 |
| P1 | 四帧历史、近端导航与任务可见性 | #462/#392、黑夜、startup | 同一split/seed/预算比较历史跨度与部署可用近端导航；未来帧仅作GT/审计 |
| P1 | 绕出—通过—归位与横穿用户对照训练 | #13/#220/#141/#190/#228 | 独立route阶段配对，分别看误归位、首跨反向、STOP漏报；增加same-RS负例route覆盖 |
| P2 | 评测独立性与提示词精简实验 | run集中与audit翻转 | 固定旧回归集+新盲测；按run/span聚合。清标签、改输入、改prompt分别做配对消融 |

不建议现在直接全量重训或继续加长提示词。首先让“图中事实—前提—监督动作—时间定义”互相一致，才能解释下一轮训练增益。

## 10. 可复查产物

- [逐case人工笔记 JSONL](EVAL_RGB_REVIEW_20260910.jsonl)：原case身份、GT/预测、实际输入帧号、已看帧、逐段视觉观察、证据来源与限度。
- [完整复算统计](probe_output/rgb_review_20260910/review_statistics.json)、[异常路线预检](probe_output/rgb_review_20260910/route_precheck.json)。
- [源标注摘录](probe_output/rgb_review_20260910/selected_source_annotations.json)、[关键原始meta](probe_output/rgb_review_20260910/focused_metadata.json)。
- [原图取证脚本](probe_output/rgb_review_20260910/inspect_cases.py)、[计数/合同复算脚本](probe_output/rgb_review_20260910/summarize_review.py)。这些可再生产物保留在git忽略的 `probe_output/`，不提交RGB。

以下附录来自实际已看的67个窗口，按case保留路线和帧号；同一路段合并笔记会明确列出所用证据case。这里只标记录与视觉的一致性，不把每个错误强制归到互斥的单一原因。

## 附录：逐窗口视觉记录

<!-- REVIEW_CASE_APPENDIX -->

### #4 EnterActorFlowV2 / f84

`Town12_Rep0_975_6_route0_01_08_21_24_13`；RAMP_MERGE_EXIT；GT `DECELERATE+LANE_CHANGE_LEFT` → 预测 `LANE_CHANGE_LEFT`。

实际输入 [81, 82, 83, 84]；本窗口已看存在帧 f81–96；合并笔记证据 case [4]。[逐帧图](probe_output/rgb_review_20260910/case_004.jpg)。

f81–84右道直行，左邻黑车，速度16.74基本稳定；f85–89减速至9.8让黑车先行；f90–94恢复且左移；f95–96首次跨左5→4。GT DECEL+LEFT支持，预测LEFT漏减速；从稳速历史预测减速需理解邻车间距。

### #13 HazardAtSideLane / f45

`Town12_Rep0_3988_1_route0_01_09_07_50_04`；VULNERABLE_CROSSING；GT `LANE_CHANGE_RIGHT+DECELERATE` → 预测 `DECELERATE+LANE_CHANGE_LEFT`。

实际输入 [42, 43, 44, 45]；本窗口已看存在帧 f42–57；合并笔记证据 case [13]。[逐帧图](probe_output/rgb_review_20260910/case_013.jpg)。

f42–45 已在靠黄线的左侧车道，逐一经过右侧骑行者；f46–49 经过最后骑行者并减速；f50–52 右侧空出；f53–57 向右跨线，lane -1→-2。GT DECEL+RIGHT 有视觉支持，模型 DECEL+LEFT 混淆绕出与归位阶段。

### #20 HardBreakRoute / f86

`Town13_Rep0_1327_0_route0_01_08_13_35_25`；LEAD_BRAKE；GT `RESUME` → 预测 `DECELERATE`。

实际输入 [83, 84, 85, 86]；本窗口已看存在帧 f83–96；合并笔记证据 case [20]。[逐帧图](probe_output/rgb_review_20260910/case_020.jpg)。

有效RGB f83–96（f97–98源帧不存在，不能当黑夜/零速证据）：f83–86雾中跟前车、对向车接近，历史先升后降；f87最低7.189，降幅距8.397仅1.208，低于1.679门槛；f88–94加速至12.379；f95–96继续加速。GT RESUME符合合同，DECEL延续历史减速/对向车风险。

### #27 EnterActorFlow / f9

`Town12_Rep0_4097_0_route0_01_10_16_45_22`；RAMP_MERGE_EXIT；GT `DECELERATE` → 预测 `NONE`。

实际输入 [6, 7, 8, 9]；本窗口已看存在帧 f6–21；合并笔记证据 case [27]。[逐帧图](probe_output/rgb_review_20260910/case_027.jpg)。

f6–9护栏夹窄匝道跟白车，历史先加后减；f10–11降至5.445/5.182；f12–17加速，f18–21恢复约11。GT DECEL源于第一段减速，NONE漏报。

### #39 MergerIntoSlowTraffic / f107

`Town13_Rep0_1587_7_route0_01_08_12_20_56`；RAMP_MERGE_EXIT；GT `RESUME` → 预测 `INVALID_ACTION_CONTEXT`。

实际输入 [104, 105, 106, 107]；本窗口已看存在帧 f104–119；合并笔记证据 case [39]。[逐帧图](probe_output/rgb_review_20260910/case_039.jpg)。

f104–107 雨夜已在lane-3，黑车左邻；f108–115 加速并超过左车；f116–119 前方道路延伸，未见当前近处汇入口。RESUME由11→17.8支持；模型INVALID与事件当前阶段、夜间弱可见性相关，但仅缺少可见匝道不能确认非法前提。

### #42 HazardAtSideLane / f102

`Town13_Rep0_1619_1_route0_01_09_03_48_09`；VULNERABLE_CROSSING；GT `NONE` → 预测 `LANE_CHANGE_RIGHT`。

实际输入 [99, 100, 101, 102]；本窗口已看存在帧 f99–114；合并笔记证据 case [42, 134]。[逐帧图](probe_output/rgb_review_20260910/case_042.jpg)。

同路段 f99–105 在左侧车道跟车，右侧骑行者尚未全部超过；f106–110 右前仍有骑行者；f111–117 保持lane-2，未向右跨线。两锚点GT分别NONE/RESUME，模型均RIGHT，属于过早归位。

### #43 NonSignalizedJunctionLeftTurnEnterFlow / f127

`Town12_Rep0_route_002772_route0_01_09_06_58_12`；UNSIGNALIZED_PRIORITY；GT `NONE` → 预测 `DECELERATE`。

实际输入 [124, 125, 126, 127]；本窗口已看存在帧 f124–139；合并笔记证据 case [43]。[逐帧图](probe_output/rgb_review_20260910/case_043.jpg)。

f124–127极暗路口，斑马线和转向箭头可辨，左上绿光点但无法确认是否本车控制灯；f128–134保持约11并转弯，f135尚9.477未达减速门槛；f136–139才明显减速至约5.5。GT NONE受2秒截窗影响，模型DECEL提前报窗外动作；R5灯控关系需全分辨率核实。 补看原图f127仍极暗；meta traffic_light_state=None、lane marking=NONE，不足以确认先前绿光点控制本车。保留R5待核查，不以光点推断错标。

### #51 HardBreakRoute / f49

`Town13_Rep0_1645_0_route0_01_08_01_52_34`；LEAD_BRAKE；GT `NONE` → 预测 `DECELERATE`。

实际输入 [46, 47, 48, 49]；本窗口已看存在帧 f46–61；合并笔记证据 case [51]。[逐帧图](probe_output/rgb_review_20260910/case_051.jpg)。

f46–49左侧红色对向车接近，远处同向前车，历史速度下降；f50–53对向车通过；f54–57另一对向车通过，ego约7.4–8.6；f58–61继续跟车。GT NONE，模型DECEL把历史变化或对向来车当未来制动。

### #72 DynamicObjectCrossing / f65

`Town10HD_Rep0_Town10HD_Scenario3_14_route0_01_08_23_27_14`；VULNERABLE_CROSSING；GT `STOP` → 预测 `STOP+LANE_CHANGE_RIGHT`。

实际输入 [62, 63, 64, 65]；本窗口已看存在帧 f62–77；合并笔记证据 case [72]。[逐帧图](probe_output/rgb_review_20260910/case_072.jpg)。

f62–65 行人在前方横穿，右侧车辆通过；f66–77 ego持续停车，行人到左侧，其他车辆从右通过。GT STOP，预测RIGHT没有ego跨线支持。

### #77 ConstructionObstacleTwoWays / f2

`Town12_Rep0_route_001147_route0_01_11_15_29_11`；STATIC_BLOCKAGE；GT `RESUME` → 预测 `STOP`。

实际输入 [0, 0, 1, 2]；本窗口已看存在帧 f0–14；合并笔记证据 case [77]。[逐帧图](probe_output/rgb_review_20260910/case_077.jpg)。

初始左填充四图实际f0,f0,f1,f2：f0晴亮干路，f1–2已雨雾湿路，天气渲染存在冷启动变化；当前几乎零速，前方通道较空，施工牌尚远。f3–10连续启动加速，f11–14才更接近施工牌。GT RESUME支持，预测STOP可能受零速/静态障碍前提影响。该例还有历史有效帧不足和首帧天气跳变。

### #80 MergerIntoSlowTrafficV2 / f133

`Town12_Rep0_968_11_route0_01_09_03_24_48`；RAMP_MERGE_EXIT；GT `LANE_CHANGE_RIGHT+DECELERATE` → 预测 `LANE_CHANGE_RIGHT`。

实际输入 [130, 131, 132, 133]；本窗口已看存在帧 f130–145；合并笔记证据 case [80]。[逐帧图](probe_output/rgb_review_20260910/case_080.jpg)。

f130–133高速多车道，右邻灰车，ego约11稳定；f134–139仍稳速；f140–141才减到8.396；f142–143右跨-3→-4，f144–145继续右跨-5。GT DECEL+RIGHT支持，模型RIGHT对而漏窗尾减速，预测时差约2秒。

### #83 CrossJunctionDefectTrafficLight / f39

`Town15_Rep0_route_002125_route0_01_09_20_36_54`；SIGNAL_FAILURE；GT `NONE` → 预测 `DECELERATE`。

实际输入 [36, 37, 38, 39]；本窗口已看存在帧 f36–51；合并笔记证据 case [83]。[逐帧图](probe_output/rgb_review_20260910/case_083.jpg)。

f36–39暗雨信号路口多车流，本车进入并转向；f40–47继续穿越、约6.5–7.4；f48–51才逐渐减至5.8。GT NONE符合2秒幅度门槛，模型DECEL偏保守；图像难以独立确认灯损坏，不能仅凭场景名宣布失效可见。

### #84 InvadingTurn / f160

`Town12_Rep0_21_0_route0_01_09_15_19_14`；ONCOMING_INVASION；GT `DECELERATE` → 预测 `NONE`。

实际输入 [157, 158, 159, 160]；本窗口已看存在帧 f157–172；合并笔记证据 case [84]。[逐帧图](probe_output/rgb_review_20260910/case_084.jpg)。

f157–160左侧施工锥筒、对向车在前，ego通道仍有空间；f161–162急减到近零，f163已0.674；f164–168加速且对向车贴近经过；f169–172正常前进。GT DECEL来自单次近停，NONE漏记录，需区分短暂停与持久STOP。

### #86 CrossJunctionDefectTrafficLight / f28

`Town03_Rep0_route_002143_route0_01_09_23_29_15`；SIGNAL_FAILURE；GT `STOP` → 预测 `NONE`。

实际输入 [25, 26, 27, 28]；本窗口已看存在帧 f25–40；合并笔记证据 case [86]。[逐帧图](probe_output/rgb_review_20260910/case_086.jpg)。

f25–28雾中进入信号路口，多方向车流，右侧近车接近路径；f29–31继续前进后急减；f32–36停等横穿车；f37–40启动。GT STOP有视觉/速度支持，预测NONE漏冲突。灯色混合不单独证明失效，动作不依赖此推断。

### #95 CrossJunctionDefectTrafficLight / f7

`Town05_Rep0_route_002075_route0_01_09_03_24_21`；JUNCTION_RULE_CONFLICT；GT `DECELERATE` → 预测 `STOP`。

实际输入 [4, 5, 6, 7]；本窗口已看存在帧 f4–19；合并笔记证据 case [95]。[逐帧图](probe_output/rgb_review_20260910/case_095.jpg)。

f4–7白天驶向信号路口，横向车流可见；f8–9减至1.787/0.545（仍高于0.5）；f10–12又启动；f13–15再次低速；f16–19才持续停止等横穿车。GT DECEL依赖1.5秒停窗和0.5阈值，模型STOP宏观合理但不符合当前时间合同。

### #127 ConstructionObstacle / f69

`Town12_Rep0_1395_1_route0_01_11_00_03_25`；STATIC_BLOCKAGE；GT `LANE_CHANGE_LEFT+DECELERATE` → 预测 `LANE_CHANGE_LEFT+RESUME`。

实际输入 [66, 67, 68, 69]；本窗口已看存在帧 f66–81；合并笔记证据 case [127]。[逐帧图](probe_output/rgb_review_20260910/case_127.jpg)。

f66–69 接近当前车道施工牌，左侧红白车正在通过；f70 急降至0.252，但只有一帧；f71–75 加速至7.051；f76–78 从左跨-3→-2；f79–81 持续加速绕行。LEFT双方正确，GT DECEL来自首个急减速样本，模型RESUME对应后续主段，说明首事件合同与宏观动作描述有差异。

### #134 HazardAtSideLane / f105

`Town13_Rep0_1619_1_route0_01_09_03_48_09`；VULNERABLE_CROSSING；GT `RESUME` → 预测 `LANE_CHANGE_RIGHT`。

实际输入 [102, 103, 104, 105]；本窗口已看存在帧 f102–117；合并笔记证据 case [42, 134]。[逐帧图](probe_output/rgb_review_20260910/case_134.jpg)。

同路段 f99–105 在左侧车道跟车，右侧骑行者尚未全部超过；f106–110 右前仍有骑行者；f111–117 保持lane-2，未向右跨线。两锚点GT分别NONE/RESUME，模型均RIGHT，属于过早归位。

### #141 ConstructionObstacle / f84

`Town12_Rep0_2184_0_route0_01_10_23_34_59`；STATIC_BLOCKAGE；GT `NONE` → 预测 `LANE_CHANGE_RIGHT`。

实际输入 [81, 82, 83, 84]；本窗口已看存在帧 f81–96；合并笔记证据 case [141]。[逐帧图](probe_output/rgb_review_20260910/case_141.jpg)。

f81→82 已由右lane-2跨左lane-1；f83–84 与施工牌并行；f85–90 右側连续锥筒尚未通过，左侧有对向车；f91–96 逐渐离开施工段但仍保持lane-1。模型RIGHT提前归位，GT NONE支持。

### #150 ConstructionObstacle / f59

`Town03_Rep0_route_001977_route0_01_10_05_03_33`；RAMP_MERGE_EXIT；GT `INVALID_ACTION_CONTEXT` → 预测 `NONE`。

实际输入 [56, 57, 58, 59]；本窗口已看存在帧 f56–71；合并笔记证据 case [150]。[逐帧图](probe_output/rgb_review_20260910/case_150.jpg)。

f56–59暗雨，城市地面道路有中心黄线和对向来车，f59跨右；f60–67加速跟车；f68–71仍地面城市道路，连续建筑围墙。输入RAMP/R3与可见道路/源R1不符，INVALID标签有依据，模型NONE未拒绝前提。

### #152 HardBreakRoute / f45

`Town12_Rep0_3031_0_route0_01_10_21_49_39`；UNSIGNALIZED_PRIORITY；GT `INVALID_ACTION_CONTEXT` → 预测 `STOP`。

实际输入 [42, 43, 44, 45]；本窗口已看存在帧 f42–57；合并笔记证据 case [152]。[逐帧图](probe_output/rgb_review_20260910/case_152.jpg)。

f42–45雾中单向一车道跟前车制动至停；f46–53持续等待，左侧对向车经过；f54–57仍停。未见近处局部交叉口/横向流，本例人为换成UNSIGNALIZED/R5而源R1，INVALID有视觉支持；预测STOP行为合理但没有识别错误前提。

### #156 HazardAtSideLaneTwoWays / f69

`Town12_Rep0_1469_0_route0_01_10_14_02_40`；VULNERABLE_CROSSING；GT `LANE_CHANGE_RIGHT` → 预测 `NONE`。

实际输入 [66, 67, 68, 69]；本窗口已看存在帧 f66–81；合并笔记证据 case [156, 243]。[逐帧图](probe_output/rgb_review_20260910/case_156.jpg)。

合并 f63–81，前段近乎全黑，只有稀疏黄白线和光点；f79–81 对向灯光增强。不能可靠识别骑行者及物理跨线。元数据提示先左后右，但视觉不足以判模型错还是标签错，单列低可见性。

### #158 ParkedObstacle / f83

`Town12_Rep0_4059_1_route0_01_10_00_40_24`；STATIC_BLOCKAGE；GT `DECELERATE+LANE_CHANGE_LEFT` → 预测 `NONE`。

实际输入 [80, 81, 82, 83]；本窗口已看存在帧 f80–95；合并笔记证据 case [158]。[逐帧图](probe_output/rgb_review_20260910/case_158.jpg)。

f80–83 雨夜右车道前方停警车，左侧车辆靠近；f84 降0.318、f85已0.745，未持续两帧停；f86–90 加速；f91–92 再减速；f93 跨左-3→-2绕警车；f94–95前进。GT DECEL+LEFT由首减速和后续跨线支持，模型NONE漏报；也存在短时速度起伏。

### #173 CrossJunctionDefectTrafficLight / f40

`Town13_Rep0_route_002189_route0_01_08_16_45_27`；SIGNAL_FAILURE；GT `RESUME` → 预测 `NONE`。

实际输入 [37, 38, 39, 40]；本窗口已看存在帧 f37–48；合并笔记证据 case [173]。[逐帧图](probe_output/rgb_review_20260910/case_173.jpg)。

有效f37–48（49–52缺源帧）：f37–40通过路口，横向红车离开，前方/左侧有车；f41短时升8.46，f42–44降约5.6，f45–48再升约9。GT RESUME依赖后段连续越阈；NONE遗漏未来恢复。

### #179 ConstructionObstacleTwoWays / f48

`Town02_Rep0_route_001339_route0_01_08_14_19_04`；STATIC_BLOCKAGE；GT `STOP` → 预测 `LANE_CHANGE_LEFT`。

实际输入 [45, 46, 47, 48]；本窗口已看存在帧 f45–60；合并笔记证据 case [179]。[逐帧图](probe_output/rgb_review_20260910/case_179.jpg)。

f45–48 夜雾暗，施工牌在前方，左侧对向来车；f49–51 对向车经过、ego减速；f52接近停，f53–60持续停在牌前，无借道跨线。GT STOP支持，模型LEFT是将最终绕障需求提前当作当前动作。

### #190 DynamicObjectCrossing / f55

`Town05_Rep0_Town05_Scenario3_57_route0_01_09_19_43_04`；VULNERABLE_CROSSING；GT `STOP` → 预测 `LANE_CHANGE_LEFT+STOP`。

实际输入 [52, 53, 54, 55]；本窗口已看存在帧 f52–67；合并笔记证据 case [190]。[逐帧图](probe_output/rgb_review_20260910/case_190.jpg)。

f52–55 行人横穿ego前方并有其他车流；f56–59 行人继续向左；f60–67 ego静止等待。GT STOP，预测额外LEFT无实际轨迹支持；不能把横穿行人当沿路绕行骑行者。

### #193 HighwayCutIn / f13

`Town12_Rep0_2457_0_route0_01_09_06_17_46`；POST_BYPASS_RETURN；GT `LANE_CHANGE_RIGHT` → 预测 `LANE_CHANGE_LEFT`。

实际输入 [10, 11, 12, 13]；本窗口已看存在帧 f10–25；合并笔记证据 case [193]。[逐帧图](probe_output/rgb_review_20260910/case_193.jpg)。

f10–13 护栏连续主路，左侧黑车、右侧红车；f14–19 ego右移；f20 跨线-3→-4；f21–25 跟随红车。GT RIGHT有视觉支持，预测LEFT错误；四帧和远端终点未明确近端目标车道。

### #197 HighwayCutIn / f16

`Town12_Rep0_2457_0_route0_01_09_06_17_46`；RAMP_MERGE_EXIT；GT `INVALID_ACTION_CONTEXT` → 预测 `NONE`。

实际输入 [13, 14, 15, 16]；本窗口已看存在帧 f13–28；合并笔记证据 case [197]。[逐帧图](probe_output/rgb_review_20260910/case_197.jpg)。

f13–16高速护栏多车道普通跟车，无当前汇出入口；f17–20右移；f21–28在右道跟红车。与#193同一run，RAMP合成错事件而RS不变；NONE未拒绝前提。注意同样可见主路的#334却被当valid RAMP，需检查事件阶段真值的一致性。

### #213 HighwayCutIn / f56

`Town13_Rep0_1586_9_route0_01_08_19_01_08`；DYNAMIC_CUTIN；GT `NONE` → 预测 `INVALID_ACTION_CONTEXT`。

实际输入 [53, 54, 55, 56]；本窗口已看存在帧 f53–68；合并笔记证据 case [518, 213]。[逐帧图](probe_output/rgb_review_20260910/case_213.jpg)。

逐帧查看 f41–68：f41–44 左侧警车接近分隔线；f45–52 接近/跨入本车道；f53–56 位于分隔线附近并向前；f57–68 向前离开。连续护栏多车道，未见信号控制本车的局部路口，与 R4 前提冲突。同路线 f44–56 的13个 NONE 错例均只答 INVALID，不能算动作误触发。 补看bundle压缩f44与本地原图，同场景内容一致。源collector的f44为R4分数0.90、R3分数0.78，触发r4_light_hazard；traffic_light_state=None、bbox traffic_light_count=0，xodr_topology_trusted=false（投影误差3021.8m），已标review_required却仍流入。Phase3只追加UE3，不修正RS。

### #218 HazardAtSideLaneTwoWays / f101

`Town12_Rep0_2735_0_route0_01_11_04_12_01`；VULNERABLE_CROSSING；GT `RESUME+LANE_CHANGE_RIGHT` → 预测 `DECELERATE+LANE_CHANGE_RIGHT`。

实际输入 [98, 99, 100, 101]；本窗口已看存在帧 f98–113；合并笔记证据 case [218]。[逐帧图](probe_output/rgb_review_20260910/case_218.jpg)。

雾中 f98–99 加速靠近骑行者；f100–101 借入左侧；f102–109 从左经过；f110–113 向右回归。8.889→11.675，GT RESUME+RIGHT 符合记录，预测 DECEL+RIGHT 更保守。

### #220 HazardAtSideLaneTwoWays / f73

`Town12_Rep0_299_0_route0_01_11_01_52_51`；VULNERABLE_CROSSING；GT `RESUME+LANE_CHANGE_LEFT` → 预测 `LANE_CHANGE_RIGHT`。

实际输入 [70, 71, 72, 73]；本窗口已看存在帧 f70–85；合并笔记证据 case [220]。[逐帧图](probe_output/rgb_review_20260910/case_220.jpg)。

f70–73 接近骑行者并加速，尚未跨中心线；f74 首次跨入对向lane1；f75–81 从左经过；f82 返回lane-1。GT RESUME+LEFT，预测 RIGHT 错把第二次回归当首次跨线。

### #228 DynamicObjectCrossing / f22

`Town13_Rep0_1603_0_route0_01_09_20_38_32`；UNSIGNALIZED_PRIORITY；GT `STOP` → 预测 `RESUME`。

实际输入 [19, 20, 21, 22]；本窗口已看存在帧 f19–34；合并笔记证据 case [228]。[逐帧图](probe_output/rgb_review_20260910/case_228.jpg)。

f19–22 STOP标志清楚，本车跟随前车低速挪动；f23–30持续近零速停让，横向主路有车流；f31–34继续停。GT STOP充分支持，预测RESUME误把前车离开当本车获得路权。

### #243 HazardAtSideLaneTwoWays / f66

`Town12_Rep0_1469_0_route0_01_10_14_02_40`；VULNERABLE_CROSSING；GT `LANE_CHANGE_LEFT+RESUME` → 预测 `NONE`。

实际输入 [63, 64, 65, 66]；本窗口已看存在帧 f63–78；合并笔记证据 case [156, 243]。[逐帧图](probe_output/rgb_review_20260910/case_243.jpg)。

合并 f63–81，前段近乎全黑，只有稀疏黄白线和光点；f79–81 对向灯光增强。不能可靠识别骑行者及物理跨线。元数据提示先左后右，但视觉不足以判模型错还是标签错，单列低可见性。

### #252 CrossJunctionDefectTrafficLight / f5

`Town07_Rep0_route_002120_route0_01_08_05_32_26`；JUNCTION_RULE_CONFLICT；GT `DECELERATE` → 预测 `STOP`。

实际输入 [2, 3, 4, 5]；本窗口已看存在帧 f2–17；合并笔记证据 case [252]。[逐帧图](probe_output/rgb_review_20260910/case_252.jpg)。

f2–5启动驶近有灯路口，前方黑车；f6–7继续加速；f8减至2.679；f9–13起伏慢行，前车向左穿过；f14–17仍0.5以上缓爬。GT DECEL有记录支持，STOP比合同停车标准更宽泛。

### #254 InvadingTurn / f120

`Town12_Rep0_21_0_route0_01_09_15_19_14`；ONCOMING_INVASION；GT `STOP` → 预测 `DECELERATE`。

实际输入 [117, 118, 119, 120]；本窗口已看存在帧 f117–132；合并笔记证据 case [254]。[逐帧图](probe_output/rgb_review_20260910/case_254.jpg)。

f117–120弯道锥筒，前方对向目标尚远；f121–124减速；f125–132持续停等，对向橙车逐渐进入窄处。GT STOP有未来证据，模型DECEL识别风险但低估停车强度；R1/R2在同一路线不同段变化不自动代表错标。

### #255 HighwayExit / f87

`Town13_Rep0_1396_11_route0_01_09_23_28_37`；RAMP_MERGE_EXIT；GT `LANE_CHANGE_LEFT` → 预测 `NONE`。

实际输入 [84, 85, 86, 87]；本窗口已看存在帧 f84–99；合并笔记证据 case [255]。[逐帧图](probe_output/rgb_review_20260910/case_255.jpg)。

f84–87行驶于高速出口右侧连续车道，左侧橙车/主路分离；f88 lane-4→-2但RGB中ego仍沿右边界前进，未见跨到左邻车道；f89–95沿出口自然弯曲；f96–99护栏鼻端分出，ego仍右支。GT LEFT疑似lane编号/section变换伪变道，需原图和元数据补证。 补看原始1152×384的f86/87/88/89：左右连续边界之间前进，无向左跨线。f87→88同road4520，lane-4→-2；ego坐标位移(2.745924,+0.039825)m，f89横移+0.148591m，正号为右；微小正steer随右弯。可确认LEFT标签与视觉跨线定义不符，具体section/waypoint归属原因待同源xodr。源标注也记changed_route=false、route_centered=true。

### #259 HazardAtSideLane / f36

`Town12_Rep0_3988_1_route0_01_09_07_50_04`；VULNERABLE_CROSSING；GT `LANE_CHANGE_LEFT` → 预测 `LANE_CHANGE_LEFT+DECELERATE`。

实际输入 [33, 34, 35, 36]；本窗口已看存在帧 f33–48；合并笔记证据 case [259]。[逐帧图](probe_output/rgb_review_20260910/case_259.jpg)。

同一路线较早的 f33–36 在外侧 lane -2 接近骑行者；f37–40 左移；f41 跨入 -1；f42–48 经过骑行者。GT LEFT 符合首跨；12.487降到11.103的幅度1.384小于2.497门槛，多报 DECEL 属阈值/应然与实然差异。

### #291 DynamicObjectCrossing / f20

`Town06_Rep0_Town06_Scenario3_26_route0_01_08_10_19_22`；DYNAMIC_CUTIN；GT `NONE` → 预测 `DECELERATE`。

实际输入 [17, 18, 19, 20]；本窗口已看存在帧 f17–32；合并笔记证据 case [291, 509]。[逐帧图](probe_output/rgb_review_20260910/case_291.jpg)。

合并查看 f17–33：f17–25 主要为相邻车道/前方车辆并行行驶，ego 前方有间距；f26–32 未见足以要求停车的明确近距切入；f33 底部有近处局部目标，身份不能据缩图确定。速度约6.7–8.0，无实际停车。NONE 合同可复现，STOP 偏强；UE3 生效区间可见性存疑，不能仅凭未见宣布事件不存在。

### #321 StaticCutIn / f59

`Town13_Rep0_1718_0_route0_01_08_17_17_56`；DYNAMIC_CUTIN；GT `STOP` → 预测 `DECELERATE`。

实际输入 [56, 57, 58, 59]；本窗口已看存在帧 f56–71；合并笔记证据 case [321]。[逐帧图](probe_output/rgb_review_20260910/case_321.jpg)。

f56–59雨中右前红车、右侧警车，左侧对向蓝车经过；f60–62向红车接近并减速；f63–71持续近零停等，红车偏斜占前方。GT STOP支持，模型DECEL低估停车强度。

### #334 StaticCutIn / f7

`Town13_Rep0_1264_1_route0_01_08_01_53_22`；RAMP_MERGE_EXIT；GT `LANE_CHANGE_LEFT+RESUME` → 预测 `RESUME`。

实际输入 [4, 5, 6, 7]；本窗口已看存在帧 f4–19；合并笔记证据 case [334, 675]。[逐帧图](probe_output/rgb_review_20260910/case_334.jpg)。

合并 f4–23：f4–7 桥上多车道跟车，右侧连续护栏，无清楚的当前匝道口；f8–11 开始左移；f12 跨线-3→-2；f13–23 保持中间车道。LEFT确实发生，模型漏报。RAMP当前阶段前提可疑，R3不等于正进行匝道动作；近端目标车道没有输入。 源f7事件由event_highway_trigger_core_r3触发，距离场景trigger15.21m<20m；highway_ramp_merge_split_hint=false、highway_scenario_merge_approach=false、xodr_topology_trusted=false。这解释了普通主路被当active ramp的来源，需与同RS错事件#197一致复核。

### #354 CrossJunctionDefectTrafficLight / f27

`Town05_Rep0_route_002067_route0_01_09_04_45_44`；SIGNAL_FAILURE；GT `DECELERATE` → 预测 `NONE`。

实际输入 [24, 25, 26, 27]；本窗口已看存在帧 f24–39；合并笔记证据 case [354]。[逐帧图](probe_output/rgb_review_20260910/case_354.jpg)。

f24–27雨雾路口蓝SUV横穿；f28–31蓝车离开，其他车仍近冲突区；f32升8.201，f33–35降到5.525/5.735；f36–39起伏前进。GT DECEL是7.001基准的20%越阈，模型NONE遗漏纵向起伏；可见路口冲突，不必把波动全解释为视觉错标。

### #382 CrossingBicycleFlow / f46

`Town12_Rep0_684_2_route0_01_09_23_50_11`；VULNERABLE_CROSSING；GT `NONE` → 预测 `INVALID_ACTION_CONTEXT`。

实际输入 [43, 44, 45, 46]；本窗口已看存在帧 f43–58；合并笔记证据 case [382, 429]。[逐帧图](probe_output/rgb_review_20260910/case_382.jpg)。

合并 f28–58 夜间极暗，对向车灯和局部中心黄线可见，难以辨认信号灯和骑行者。不能断言R4错误；f31缓慢速度上升低于1.2门槛，NONE与RESUME差异受阈值影响。

### #392 VehicleTurningRoute / f60

`Town06_Rep0_Town06_Scenario4_47_route0_01_09_10_51_25`；VULNERABLE_CROSSING；GT `STOP+LANE_CHANGE_RIGHT` → 预测 `STOP+LANE_CHANGE_LEFT`。

实际输入 [57, 58, 59, 60]；本窗口已看存在帧 f57–72；合并笔记证据 case [392]。[逐帧图](probe_output/rgb_review_20260910/case_392.jpg)。

雨夜 f57–60 停车，骑行者向左横穿；f61–67 持续等待；f68–71 启动；f72 元数据右跨-5→-6，位于3秒窗尾。STOP 可确认，当前静止四帧无法确定3秒后目标车道；右跨物理几何仍宜原图/拓扑复核。

### #395 HazardAtSideLane / f118

`Town13_Rep0_1710_11_route0_01_08_17_09_31`；POST_BYPASS_RETURN；GT `LANE_CHANGE_RIGHT+RESUME` → 预测 `LANE_CHANGE_LEFT`。

实际输入 [115, 116, 117, 118]；本窗口已看存在帧 f115–130；合并笔记证据 case [395]。[逐帧图](probe_output/rgb_review_20260910/case_395.jpg)。

f115–118已驶出路口到直路，靠近黄线，左侧对向蓝车经过；f119–121加速并右移；f122跨1→2；f123–130保持右车道继续加速。GT RESUME+RIGHT的动作支持，模型LEFT反向；R4当前有效性另需核查，不能把动作正确与前提正确混为一谈。

### #396 AccidentTwoWays / f82

`Town02_Rep0_route_001608_route0_01_07_21_43_19`；DYNAMIC_CUTIN；GT `INVALID_ACTION_CONTEXT` → 预测 `STOP`。

实际输入 [79, 80, 81, 82]；本窗口已看存在帧 f79–94；合并笔记证据 case [396]。[逐帧图](probe_output/rgb_review_20260910/case_396.jpg)。

f79–82白天右侧黄色事故车已被经过，与ego相对后退，未见主动侵入；f83–86事故车退到视野外；f87–94道路弯曲ego减速仍行驶。DYNAMIC合成错事件的INVALID有视觉依据，STOP既未拒绝前提也不符随后的记录。

### #429 CrossingBicycleFlow / f31

`Town12_Rep0_684_2_route0_01_09_23_50_11`；VULNERABLE_CROSSING；GT `NONE` → 预测 `RESUME`。

实际输入 [28, 29, 30, 31]；本窗口已看存在帧 f28–43；合并笔记证据 case [382, 429]。[逐帧图](probe_output/rgb_review_20260910/case_429.jpg)。

合并 f28–58 夜间极暗，对向车灯和局部中心黄线可见，难以辨认信号灯和骑行者。不能断言R4错误；f31缓慢速度上升低于1.2门槛，NONE与RESUME差异受阈值影响。

### #442 DynamicObjectCrossing / f26

`Town05_Rep0_Town05_Scenario3_57_route0_01_09_19_43_04`；UNSIGNALIZED_PRIORITY；GT `STOP` → 预测 `RESUME`。

实际输入 [23, 24, 25, 26]；本窗口已看存在帧 f23–38；合并笔记证据 case [442]。[逐帧图](probe_output/rgb_review_20260910/case_442.jpg)。

f23–26雨雾路口右前有行人/车辆，横向车辆通过，ego已明显减速；f27–31短暂停等；f32–38重新前行、右侧黑车出现并向前。GT STOP来自先停，模型RESUME跳过前置等待。

### #462 MergerIntoSlowTraffic / f90

`Town13_Rep0_1587_6_route0_01_09_23_40_39`；RAMP_MERGE_EXIT；GT `LANE_CHANGE_RIGHT+RESUME` → 预测 `NONE`。

实际输入 [87, 88, 89, 90]；本窗口已看存在帧 f87–102；合并笔记证据 case [462, 549]。[逐帧图](probe_output/rgb_review_20260910/case_462.jpg)。

合并 f87–105：f87–93 雾雨、前车在左侧相邻位置，ego速度约9保持；f94–98 加速约11；f99–101 向右移；f102 跨线-2→-3；f103–105 稳定在右侧，红车在左。RESUME+RIGHT与记录一致；首次跨线距f90达3秒，四帧中尚无明确转向趋势，要求提前预测而非单纯视觉识别。

### #486 VehicleTurningRoute / f282

`Town13_Rep0_1050_1_route0_01_09_21_06_45`；UNSIGNALIZED_PRIORITY；GT `INVALID_ACTION_CONTEXT` → 预测 `STOP`。

实际输入 [279, 280, 281, 282]；本窗口已看存在帧 f279–294；合并笔记证据 case [486]。[逐帧图](probe_output/rgb_review_20260910/case_486.jpg)。

f279–282城市直路停止，骑行者从右往左横穿，绿色对向车通过；f283–290仍停等骑行者；f291–294才启动。输入UNSIGNALIZED/R5与源R1及当前横穿事件不同，模型STOP忽略前提错配。局部侧街是否存在仍应地图核验，不用scenario名称作为证据。

### #489 AccidentTwoWays / f105

`Town12_Rep0_2919_0_route0_01_11_09_44_00`；RAMP_MERGE_EXIT；GT `INVALID_ACTION_CONTEXT` → 预测 `STOP`。

实际输入 [102, 103, 104, 105]；本窗口已看存在帧 f102–117；合并笔记证据 case [489]。[逐帧图](probe_output/rgb_review_20260910/case_489.jpg)。

f102–105极暗窄路、对向车经过，前方事故车灯；f106–109本车加速向左绕；f110–117通过车辆，源lane由-1→1。RAMP/R3与窄双向事故路段不符，但图像较暗需保留可见性限制；模型STOP未拒绝前提且未匹配未来加速。

### #492 HighwayExit / f99

`Town12_Rep0_4136_4_route0_01_08_20_21_44`；RAMP_MERGE_EXIT；GT `DECELERATE` → 预测 `NONE`。

实际输入 [96, 97, 98, 99]；本窗口已看存在帧 f96–111；合并笔记证据 case [492]。[逐帧图](probe_output/rgb_review_20260910/case_492.jpg)。

f96–99雨天高速最右道跟黄车，左邻黑车；f100–101从9.712减到7.759，降1.953，刚超过20%阈1.942；f102–107约8；f108–111约8继续跟车。GT DECEL几乎贴阈值，模型NONE不能据此判明显感知失败；背景是高速跟车，当前出口阶段不清。

### #509 DynamicObjectCrossing / f21

`Town06_Rep0_Town06_Scenario3_26_route0_01_08_10_19_22`；DYNAMIC_CUTIN；GT `NONE` → 预测 `STOP`。

实际输入 [18, 19, 20, 21]；本窗口已看存在帧 f18–33；合并笔记证据 case [291, 509]。[逐帧图](probe_output/rgb_review_20260910/case_509.jpg)。

合并查看 f17–33：f17–25 主要为相邻车道/前方车辆并行行驶，ego 前方有间距；f26–32 未见足以要求停车的明确近距切入；f33 底部有近处局部目标，身份不能据缩图确定。速度约6.7–8.0，无实际停车。NONE 合同可复现，STOP 偏强；UE3 生效区间可见性存疑，不能仅凭未见宣布事件不存在。

### #518 HighwayCutIn / f44

`Town13_Rep0_1586_9_route0_01_08_19_01_08`；DYNAMIC_CUTIN；GT `NONE` → 预测 `INVALID_ACTION_CONTEXT`。

实际输入 [41, 42, 43, 44]；本窗口已看存在帧 f41–56；合并笔记证据 case [518, 213]。[逐帧图](probe_output/rgb_review_20260910/case_518.jpg)。

逐帧查看 f41–68：f41–44 左侧警车接近分隔线；f45–52 接近/跨入本车道；f53–56 位于分隔线附近并向前；f57–68 向前离开。连续护栏多车道，未见信号控制本车的局部路口，与 R4 前提冲突。同路线 f44–56 的13个 NONE 错例均只答 INVALID，不能算动作误触发。 补看bundle压缩f44与本地原图，同场景内容一致。源collector的f44为R4分数0.90、R3分数0.78，触发r4_light_hazard；traffic_light_state=None、bbox traffic_light_count=0，xodr_topology_trusted=false（投影误差3021.8m），已标review_required却仍流入。Phase3只追加UE3，不修正RS。

### #529 HazardAtSideLaneTwoWays / f81

`Town13_Rep0_1475_1_route0_01_08_22_30_14`；VULNERABLE_CROSSING；GT `RESUME+LANE_CHANGE_RIGHT` → 预测 `LANE_CHANGE_RIGHT+DECELERATE`。

实际输入 [78, 79, 80, 81]；本窗口已看存在帧 f78–93；合并笔记证据 case [529]。[逐帧图](probe_output/rgb_review_20260910/case_529.jpg)。

f78–81 向左跨线阶段；f82–88 加速经过骑行者；f89 右归；f90–93 保持。GT RESUME+RIGHT，模型 DECEL+RIGHT 横向对而纵向保守。

### #542 DynamicObjectCrossing / f27

`Town04_Rep0_Town04_Scenario3_5_route0_01_10_15_28_06`；DYNAMIC_CUTIN；GT `NONE` → 预测 `DECELERATE`。

实际输入 [24, 25, 26, 27]；本窗口已看存在帧 f24–39；合并笔记证据 case [542]。[逐帧图](probe_output/rgb_review_20260910/case_542.jpg)。

f24–27 雨夜右侧蓝车很近；f28–31 蓝车向前；f32–39 ego 持续前行加速。速度18.469至20.080，变化未达20%门槛，GT NONE 有算术依据；模型 DECEL 与记录不同，弱可见性和动作门槛共同影响。

### #549 MergerIntoSlowTraffic / f93

`Town13_Rep0_1587_6_route0_01_09_23_40_39`；RAMP_MERGE_EXIT；GT `LANE_CHANGE_RIGHT+RESUME` → 预测 `NONE`。

实际输入 [90, 91, 92, 93]；本窗口已看存在帧 f90–105；合并笔记证据 case [462, 549]。[逐帧图](probe_output/rgb_review_20260910/case_549.jpg)。

合并 f87–105：f87–93 雾雨、前车在左侧相邻位置，ego速度约9保持；f94–98 加速约11；f99–101 向右移；f102 跨线-2→-3；f103–105 稳定在右侧，红车在左。RESUME+RIGHT与记录一致；首次跨线距f90达3秒，四帧中尚无明确转向趋势，要求提前预测而非单纯视觉识别。

### #582 DynamicObjectCrossing / f113

`Town10HD_Rep0_Town10HD_Scenario3_4_route0_01_10_03_26_36`；DYNAMIC_CUTIN；GT `NONE` → 预测 `STOP`。

实际输入 [110, 111, 112, 113]；本窗口已看存在帧 f110–125；合并笔记证据 case [582]。[逐帧图](probe_output/rgb_review_20260910/case_582.jpg)。

f110–113 ego 制动历史明显，右侧有车，前方仍有空间；f114–117 继续滚动；f118–125 经过更多右侧车辆。未来最低约3.7，未停车；回升仅最后样本达到门槛，不能构成两帧 RESUME。模型 STOP 把历史减速延续成未来停车。

### #613 InterurbanActorFlow / f36

`Town13_Rep0_1294_4_route0_01_10_03_53_41`；POST_BYPASS_RETURN；GT `LANE_CHANGE_RIGHT` → 预测 `LANE_CHANGE_LEFT`。

实际输入 [33, 34, 35, 36]；本窗口已看存在帧 f33–48；合并笔记证据 case [613]。[逐帧图](probe_output/rgb_review_20260910/case_613.jpg)。

f33–36乡间道路前方从单方向车道展宽，白线分叉；f37–38出现lane1→2，ego进入右側分支；f39–44保持前行；f45–48两道再次收束且lane2→1。这里车道增减/延续与主动跨线难分，不能仅lane-id断定RIGHT是真变道，列lane-section拓扑复核候选。 补看原图f35–39：始终位于延续道路右侧，中心黄线渐变成白线，未见明确向右跨线；f36→38横向-0.362m，实际上略向左，随后回摆。RIGHT缺少物理跨线支持，优先核对展宽处lane-section/新生车道归属，不直接用坐标代替边界标定。

### #632 ConstructionObstacle / f97

`Town03_Rep0_route_001980_route0_01_08_14_06_07`；POST_BYPASS_RETURN；GT `STOP` → 预测 `NONE`。

实际输入 [94, 95, 96, 97]；本窗口已看存在帧 f94–109；合并笔记证据 case [632]。[逐帧图](probe_output/rgb_review_20260910/case_632.jpg)。

f94–97 雾中跟红车、主路弯曲，当前车道前方仍有间距；f98突然减速；f99–101接近静止；f102–109再加速，同路其他车辆通过。STOP记录存在，但四帧没有明确近处阻挡，需回查控制器hazard/近端路线才能解释急停原因，不能仅RGB断定STOP是合理应然动作。 补查raw meta：f97–99 vehicle_hazard=true且brake=true，f100转false；说明急停有控制器车辆风险触发，不是速度随机字段错位。哪个目标和预测路径触发仍未完成bbox ID归因。

### #637 StaticCutIn / f10

`Town13_Rep0_1266_1_route0_01_09_05_38_22`；RAMP_MERGE_EXIT；GT `DECELERATE` → 预测 `NONE`。

实际输入 [7, 8, 9, 10]；本窗口已看存在帧 f7–22；合并笔记证据 case [637]。[逐帧图](probe_output/rgb_review_20260910/case_637.jpg)。

f7–10桥上三车道车流，ego跟蓝车、右邻黑车；f11–13减到5.258；f14–18起伏跟车；f19–22加速，右黑车仍近邻。GT DECEL第一事件可复现，NONE漏纵向变化；当前是否匝道动作未见直接证据。

### #646 Accident / f80

`Town12_Rep0_2882_0_route0_01_09_07_57_43`；STATIC_BLOCKAGE；GT `DECELERATE` → 预测 `LANE_CHANGE_RIGHT`。

实际输入 [77, 78, 79, 80]；本窗口已看存在帧 f77–92；合并笔记证据 case [646]。[逐帧图](probe_output/rgb_review_20260910/case_646.jpg)。

f77–79 从右车道朝左接近，右侧事故警车，左侧对向车辆；f80已跨到lane1；f81–86沿左侧继续经过事故车辆；f87–88速度降6.799/5.911；f89–92仍lane1。GT DECEL符合记录，预测RIGHT提前归位，当前障碍右侧并未全清。

### #653 HazardAtSideLaneTwoWays / f77

`Town13_Rep0_1475_1_route0_01_08_22_30_14`；VULNERABLE_CROSSING；GT `LANE_CHANGE_LEFT+RESUME` → 预测 `RESUME`。

实际输入 [74, 75, 76, 77]；本窗口已看存在帧 f74–89；合并笔记证据 case [653]。[逐帧图](probe_output/rgb_review_20260910/case_653.jpg)。

f74–77 浓雾，骑行者前方，ego速度降到3.035；f78–80 加速左移；f81 跨黄线1→-1；f82–88 左侧经过；f89 右归1。GT RESUME+LEFT 支持，模型漏 LEFT。

### #675 StaticCutIn / f11

`Town13_Rep0_1264_1_route0_01_08_01_53_22`；RAMP_MERGE_EXIT；GT `LANE_CHANGE_LEFT` → 预测 `NONE`。

实际输入 [8, 9, 10, 11]；本窗口已看存在帧 f8–23；合并笔记证据 case [334, 675]。[逐帧图](probe_output/rgb_review_20260910/case_675.jpg)。

合并 f4–23：f4–7 桥上多车道跟车，右侧连续护栏，无清楚的当前匝道口；f8–11 开始左移；f12 跨线-3→-2；f13–23 保持中间车道。LEFT确实发生，模型漏报。RAMP当前阶段前提可疑，R3不等于正进行匝道动作；近端目标车道没有输入。 源f7事件由event_highway_trigger_core_r3触发，距离场景trigger15.21m<20m；highway_ramp_merge_split_hint=false、highway_scenario_merge_approach=false、xodr_topology_trusted=false。这解释了普通主路被当active ramp的来源，需与同RS错事件#197一致复核。

### #684 ConstructionObstacleTwoWays / f157

`Town13_Rep0_route_003039_route0_01_10_14_43_08`；STATIC_BLOCKAGE；GT `STOP+LANE_CHANGE_RIGHT` → 预测 `RESUME`。

实际输入 [154, 155, 156, 157]；本窗口已看存在帧 f154–169；合并笔记证据 case [684]。[逐帧图](probe_output/rgb_review_20260910/case_684.jpg)。

f154–157在借道位置面对来车，速度反复接近零；f158–164继续停等/微动；f165–168启动向右归回；f169从lane-1→1。GT STOP+RIGHT由当前持续等待和未来首跨分别构成；预测RESUME忽略优先等待，而右跨发生在窗末。

### #703 HazardAtSideLane / f40

`Town12_Rep0_2193_3_route0_01_08_23_09_14`；VULNERABLE_CROSSING；GT `LANE_CHANGE_RIGHT+DECELERATE` → 预测 `LANE_CHANGE_RIGHT`。

实际输入 [37, 38, 39, 40]；本窗口已看存在帧 f37–52；合并笔记证据 case [703]。[逐帧图](probe_output/rgb_review_20260910/case_703.jpg)。

f37–40 已在骑行者左侧；f41–44 经过右侧骑行者，左侧有对向车；f45–48 向右回lane2并减速；f49–52 右侧稳定。14.011→9.467支持 DECEL+RIGHT，模型漏未来减速。

### #720 InterurbanAdvancedActorFlow / f103

`Town12_Rep0_417_3_route0_01_08_21_11_07`；POST_BYPASS_RETURN；GT `RESUME` → 预测 `LANE_CHANGE_LEFT`。

实际输入 [100, 101, 102, 103]；本窗口已看存在帧 f100–115；合并笔记证据 case [720]。[逐帧图](probe_output/rgb_review_20260910/case_720.jpg)。

f100–103雨夜，道路展开，f102已到lane-2；f104–107前进加速；f108–111加速到15.4，对向车从左通过；f112–115维持lane-2。GT RESUME支持，预测LEFT无未来跨线支持，弱光下车道展宽与既往变化可能干扰。

### #733 CrossJunctionDefectTrafficLight / f32

`Town04_Rep0_route_002175_route0_01_09_03_48_04`；SIGNAL_FAILURE；GT `RESUME` → 预测 `STOP`。

实际输入 [29, 30, 31, 32]；本窗口已看存在帧 f29–44；合并笔记证据 case [733]。[逐帧图](probe_output/rgb_review_20260910/case_733.jpg)。

f29–32一直近零等待，右侧黑车正在接近并横穿；f33已0.649，f34–40在黑车穿过时连续启动到6.9；f41–44继续通过。GT RESUME来自下一帧就解除等待，模型STOP与四张静止图和仍横穿车辆相符，是未来放行边界/行为预测难题，不能据未来启动要求当前视觉必然知道。

### #764 CrossJunctionDefectTrafficLight / f2

`Town07_Rep0_route_002120_route0_01_08_05_32_26`；SIGNAL_FAILURE；GT `RESUME` → 预测 `STOP`。

实际输入 [0, 0, 1, 2]；本窗口已看存在帧 f0–14；合并笔记证据 case [764]。[逐帧图](probe_output/rgb_review_20260910/case_764.jpg)。

左填充输入f0,f0,f1,f2：f0空路，f1突然出现前车，f2仍近零；f3–7启动加速；f8–14驶近灯口起伏减速，前车左转。GT RESUME支持，但首帧actor生成/天气亮度变化与历史不足会降低四帧运动证据质量。
