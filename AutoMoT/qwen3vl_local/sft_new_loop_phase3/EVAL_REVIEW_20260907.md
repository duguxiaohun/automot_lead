# Phase3 20260907 审计包复核

复核对象：`AutoMoT/checkpoints/sft_new_loop_phase3_20260907_095517_4rgb_audit_bundle`。
结论：LoRA 学到了部分停车、横向动作和道路不匹配拒绝，但纵向动作预测还不可用。
失败同时来自模型运动理解、当前动作与未来计划的语义差异、规则的时间/幅度边界，以及上游情境标签噪声。
不能把所有错题归为提示词问题，也不能直接把模型答案当作重标注依据。

本文记录最初的只读审计阶段：当时只新增审计文档、人工复核笔记和本地可视化，没有改训练代码、提示词、规则或既有标签，没有重训模型。用户随后授权的逐帧复核、规则/提示词/代码修订及索引重建见[后续修订记录](REPAIR_20260907.md)；下述旧包成绩保持原样，不代表修订后模型成绩。

## 核对范围与证据边界

- 重新统计 base production、LoRA production、LoRA audit 各768题；三组的case、GT、prompt spec和输入路径一一对应。
- 人工查看52道题的四帧历史，对应51个不同anchor、47条run；包括42道错误题和10道正确对照。
  错题来自审计包的15类目标桶，属于定向抽检，不是随机噪声率估计。
- 12个关键anchor继续读取本地原始RGB和meta，逐帧查看`[t-3,t+13]`；未来图只用于离线判定标签，未作为模型输入。
- 读取原始数据前，对这47条run使用已有`is_abnormal_lead_route`复核，均未触发异常时长过滤。
  结束前进一步核对本次评测全部258条run，均存在且通过同一过滤；未发现异常时长route混入本次评测。
- #8与#167是完全相同四图、同prompt、同anchor的重复题；只计一个独立视觉证据。
- 原包图像最大边640、JPEG质量55；可用时以本地原图补查。暗夜/雾中的弱标线不强行判左右真值。
- 全部逐例结论见[52条人工笔记](EVAL_RGB_REVIEW_20260907.jsonl)；图组和统计位于
  [本地复核目录](probe_output/eval_review_20260907/)。该目录受现有`.gitignore`排除，不入库。
- 原包`visual_audit_manifest.json`明确`action_boundary_verification_complete=false`。
  582条既有RGB审计缓存的可用性不等于本次训练动作标签全部经人工确认；manifest训练数据覆盖2914条run。

## 成功率：只能称离线问答正确率

| 指标 | Base production | LoRA production | LoRA audit |
|---|---:|---:|---:|
| 严格整题正确 | 104/768，13.54% | 304/768，39.58% | 258/768，33.59% |
| 有效情境题严格正确 | 99/640，15.47% | 186/640，29.06% | 183/640，28.59% |
| INVALID题严格正确 | 5/128，3.91% | 118/128，92.19% | 75/128，58.59% |
| 只解析答案前缀的整题正确 | 104/768 | 304/768 | 302/768，39.32% |
| 完整格式失败 | 0 | 0 | 52 |

LoRA比base多答对200题，其中113题增益来自INVALID。有效驾驶情境仍有454/640题不匹配规则标签。
这不是CARLA路线完成率、碰撞率或闭环驾驶成功率，也不是串联真实Phase1/Phase2预测后的端到端成功率。
当前输入情境来自离线构造的前提；尚未验证真实上游错误分布。

768题只有258条run、740个不同anchor、767份不同输入，邻近帧和重复题相关，不能按768个独立场景解释统计把握。
评测为十情境均衡抽样、有效题640加INVALID128，不代表自然4Hz驾驶分布。

## 具体动作指标

| 动作 | GT YES | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| DECELERATE | 169 | 9 / 34 / 160 | 20.93% | **5.33%** | 8.49% |
| STOP | 155 | 90 / 131 / 65 | 40.72% | 58.06% | 47.87% |
| RESUME | 167 | 21 / 83 / 146 | 20.19% | **12.57%** | 15.50% |
| LANE_CHANGE_LEFT | 77 | 41 / 34 / 36 | 54.67% | 53.25% | 53.95% |
| LANE_CHANGE_RIGHT | 85 | 55 / 33 / 30 | 62.50% | 64.71% | 63.58% |

纵向三题每题768份；横向仅在FULL_MANEUVER的313题中评分，未问横向不能算NO。
减速单题accuracy虽为74.74%，永远答NO就有599/768=78.00%；不能用多数NO撑起的accuracy评价正类能力。
Base的减速召回27/169=15.98%，LoRA降到9/169，说明总分提升掩盖了这个动作的退化。

仅有效题的纵向混淆（NONE表示三纵向全NO，横向可另有动作）：

| GT → 预测 | DECELERATE | STOP | RESUME | NONE |
|---|---:|---:|---:|---:|
| DECELERATE，169题 | 9 | 42 | 43 | 75 |
| STOP，155题 | 14 | 90 | 18 | 33 |
| RESUME，167题 | 9 | 69 | 21 | 68 |
| NONE，149题 | 11 | 19 | 22 | 97 |

模型既漏减速，也把需要减速的题答成加速；STOP的131个假阳性中包含69道RESUME题。
不能仅概括成“模型比较保守”。

组合动作更弱：DECELERATE+RIGHT为0/24，RESUME+RIGHT为0/19，
DECELERATE+LEFT为2/23；单独RIGHT为19/24。横向有一定能力，组合中主要受纵向拖累。

## 各情境整题正确率

| 情境 | 正确/64 | 正确率 |
|---|---:|---:|
| 前车急刹 LEAD_BRAKE | 18 | 28.13% |
| 静态占道 STATIC_BLOCKAGE | 15 | 23.44% |
| 动态侵入 DYNAMIC_CUTIN | 16 | 25.00% |
| 行人/骑车人 VULNERABLE_CROSSING | 9 | **14.06%** |
| 对向侵入 ONCOMING_INVASION | 22 | 34.38% |
| 路口违规 JUNCTION_RULE_CONFLICT | 28 | 43.75% |
| 信号故障 SIGNAL_FAILURE | 32 | 50.00% |
| 目标/恢复变道 POST_BYPASS_RETURN | 8 | **12.50%** |
| 常规无灯让行 UNSIGNALIZED_PRIORITY | 19 | 29.69% |
| 匝道过渡 RAMP_MERGE_EXIT | 19 | 29.69% |

POST_BYPASS_RETURN是兼容ID，实际包括通用R-E2目标车道变换，不全是绕障后的右回。
该桶低分不能简单用“全部应该向右”修正。

## INVALID高分的局限

- wrong-road题118/126=93.65%；same-RS wrong-event题0/2。
- 后两题就是#8/#167：EnterActorFlowV2 / Town12 `2448_0` / f85，完全重复输入。
  去重后仅**一个**同RS错误事件案例，不能估计这一能力的总体成功率。
- train索引的同RS候选仅31行，却采样为178行；训练首轮165/2048个INVALID呈现来自同RS，约8.06%。
  test索引只有1个同RS候选，展开成3行，本次评测取到2行。
- train同RS缺POST_BYPASS_RETURN和UNSIGNALIZED_PRIORITY；test只覆盖RAMP_MERGE_EXIT。
  旧filtered_v12小诊断集的覆盖数字不能套到这份全量构建manifest上。

需要扩充独立路线/独立可见事件证据，再分别报告wrong-road和same-road wrong-event；重复抽同一题不能补覆盖。

## RGB与规则共同定位的关键案例

以下case编号均为原包`case_index`，完整run_id保存在JSONL笔记。

| 案例 | 观察证据 | 归因与处理建议 |
|---|---|---|
| #350 VehicleTurningRoute/Town04 f81 | 四帧自车视点几乎不动；当前和未来9点速度全0；模型DECELERATE，GT STOP | 明确的模型运动状态错误。STOP包括已经停车继续等待，不需要看到STOP牌或刹车灯。 |
| #616 VehicleTurningRoute/Town05 f21 | 路口前跟车，当前0.003m/s并继续等待；模型全NO | “没有速度变化”不能推导没有STOP动作。补静止等待对照与当前ego状态。 |
| #221 ConstructionObstacle/Town06 f20 | 逐帧f17–33沿右弯保持车道，road20/lane6不变，约8m/s；模型RIGHT | 模型把曲率/标线扫动当跨线；现有prompt已经明确禁止，继续加同义句不足以解决。 |
| #653 VehicleTurningRoute/Town04 f31 | f28–44是多车道高速/出口，护墙、之后分流鼻；source却给R5/R-E5 | 高置信情境标签噪声。模型INVALID有依据；先隔离该精确span、回查RS/RE5映射，不以场景名给整条route改标签。 |
| #658 HighwayCutIn/Town12 f105 | f102–118灰黑车从右侧斜入当前通道；存在显式RGB UE3正例span98–118 | 模型以“highway/no intruder”拒绝错误。高速不是invalid理由，需保留高速UE3硬例。该例未来速度另有先增后减边界。 |
| #13 BlockedIntersection/Town13 f38 | 原图显示路口车流阻挡、短行后排队；速度2.80→4.33→5.37→2.69→0.38 | 连续两点增速不等于持续恢复通行；规则给RESUME符合现有实现，却与自然语言sustained/clear-path有落差。应作为规则噪声候选标定，不能自动改STOP。 |
| #14 CrossJunctionDefectTrafficLight/Town05 f22 | anchor仍停着且冲突车在路口；未来0.5–1.25s开始起步 | GT RESUME是预测释放，模型STOP描述当前等待。需统一“此刻必须执行”与“未来窗内会执行”的监督语义。 |
| #147 StaticCutIn/Town13 f89 | anchor停车、对向车经过；f91后起步 | 与#14同类。不能要求四帧静止历史直接给出精确释放时刻，再把所有STOP视为无依据。 |
| #104 PedestrianCrossing/Town12 f66（正确对照） | anchor静止，t+1s才缓行，t+1.5s达2.297m/s，触发pulling_away例外 | production答RESUME得分、audit答STOP失败。规则正确不等于“现在就起步”的语义已清楚。 |
| #128 ConstructionObstacleTwoWays/Town13 f85 | f86向左借道，f95向右回原道，均在3s内；规则只保留首次LEFT；模型RIGHT | 确认有左右连续动作。prompt的“ends up in a different lane”和恢复语境会混淆首次借道与之后右回；需明确输出首次横向动作，或把多阶段窗口列为不确定。 |
| #204 Accident/Town10HD f20 | 最初沿右弯lane2行进，直到t+3s f32转lane1向左；模型RIGHT | 当前弯道和未来导航换道混杂。只给远端终点，缺少近端目标车道证据；不能靠终点y符号修侧向。 |
| #31 HazardAtSideLane/Town12 f57 | 未来明显制动、右跨线，12.93降至0后低速跳动；模型只答RIGHT | 漏纵向是模型错误；0速后马上1.41，未连续两点≤0.5，规则未给STOP，另暴露爬行/停车边界。 |

其它案例的保守裁定、正确对照和未确认部分都保存在人工笔记中。没有从52道定向样本外推“噪声占全部错误的百分之多少”。

## 提示词是否合适

已有设计值得保留：最后一帧为anchor；速度与横向不同horizon；纵向互斥；曲线行驶不等于变道；
最终目的地y不决定左右；低能见度不等于invalid；invalid时全部动作NO。
本地production prompt重算SHA256为`058c797e5a76a5bb4e6d6acf7bbbfeca24e1b1fc94781916107dc6527a328cbd`，与包完全一致。
这次没有发现“本地已经换prompt，包用的是旧prompt”的错配。

需要改的是可执行语义和输入信息：

1. **应当怎么开与专家实际怎么开混用。** prompt问`Should ego ... now?`，GT只看专家未来速度/lane-id。
   两者在临时等待、缓慢爬行、释放前静止、控制器速度摆动时不等价。必须先决定任务是即时策略、短窗首动作预测还是整段规划。
2. **历史没有时间标记和当前速度。** 实际四帧是4Hz连续采样、覆盖0.75s，但文本只说four-frame history。
   应明确t−0.75/t−0.50/t−0.25/t；可加入在线可获取的当前ego速度/短历史速度，禁止未来速度进入输入。
   本阶段加入这些输入需要新合同/重训，当前结果不能证明一定提升多少。
3. **RESUME自然语言与两点确认仍不一致。** “sustained”“clear path”比两个相邻采样满足阈值更强；
   #13等先走后等会得到RESUME。应处理阶段性速度曲线，不能只增加一句“ignore isolated pulse”。
4. **横向需要首次动作与完成时刻分开。** 首次lane-id切换、车身完成跨线、3s末停留车道是三个不同量。
   #128确认同窗先左后右，现有两个互斥布尔只能表达其中一段。
5. **补near-term导航/已确认恢复状态。** 最终目的地不含下一个目标车道；用长文本提醒模型不要用终点符号，
   仍不能补足缺失信息。已发生的变道/借道历史只能来自过去观测，不能把future lane真值注入。

建议的短版合同草案（未替换现有prompt）：

> Use four observations at t−0.75, t−0.50, t−0.25 and t. Choose the first longitudinal action and the first lane transition after t. A later recovery must not replace an earlier slowdown or lane transition. Staying stopped is STOP even without a stop sign. RESUME requires evidence of a sustained release, not merely an open-looking road. Interpret left and right relative to ego. A proposed event may remain unresolved under occlusion; reject it only when visible evidence contradicts it.

这段是讨论稿：尤其“first longitudinal action”需要先与STOP/pulling-away标定规则统一，不能直接配旧GT训练。

## 规则标定建议与已做的静态诊断

640道有效题的纵向GT都能由包内四舍五入后的future速度按当前规则重算出来（640/640）。
因此没有证据表明本次主要是label计算错位；问题在规则是否代表所问的驾驶语义，以及情境前提是否真实。

只做敏感性诊断、不修改GT：把速度变化阈值整体改为原来的0.8/1.2，分别40/640和42/640题换类；
STOP低速阈值从0.5改为0.25或0.75，分别13/640题换类。二者可能重叠，不可相加。
167个RESUME中11个在同一个2s窗内又跌到低于初速一个完整变化阈值；这些是值得逐帧复核的候选，不是已确认11个错标。
这些试算基于test证据，只用于发现不稳定处，不能据模型得分择阈值；正式标定必须在独立train/val审计集完成。

推荐优先顺序：

- **P0，清理情境标签：** 精确复核#653的R5/RE5区间，检查RE5由接近路口提前扩展到高速/普通路段的来源；
  复核#128的静态阻挡/借道/恢复阶段，维护显式candidate uncertainty，先隔离未确认窗。
- **P0，定义阶段：** 停车等待、停车后可释放、持续减速、先加后刹、先左后右分别分段，明确首次动作或即时动作。
  STOP使用低速驻留与释放滞回；RESUME除最少持续时间外还需考虑随后反转；不在这里凭空指定新数值。
- **P1，横向几何：** Driving+同road lane-id只能作候选；核对lane-section连通与实际边界跨越。
  对一窗多次切换、lane-id改号、弯道和junction连接段保留不确定，不把全部元数据变更视为变道。
- **P1，数据与输入：** 增加真实同RS错事件的独立route覆盖；按事件阶段/动作/能见度分层，补DECELERATE与RESUME对照。
  训练首轮DECELERATE正例2294、RESUME2868、STOP3012，减速并非没有训练正例；问题不宜只靠过采样解决。
- **P2，配对消融：** 同seed、同物理route分割、同预算比较旧版/仅新prompt/仅清标签/两者组合；
  再单独测当前速度输入与更长历史。先固定人工holdout，避免按本次错题调好后仍用同题宣称泛化。

## 评测与训练记录本身的问题

1. **审计解释不稳定，不能当teacher。** production与audit答案有97/768题变化，尽管答案正确数304与302接近。
   audit有52题完整格式不合法，44道本来答案全对的题因evidence格式被扣分。
   检查52题中有166条空evidence，1题行数不符，没有超过14词的evidence。故33.59%与39.58%的差距主要是格式代价。
   还见“ego must yield”被用来支持RESUME、“ego still stopped”被用来支持STOP=NO、断言不可见的brake pedal。
   正确的YES/NO与可信解释需分开评分；解释不能独立证明失败原因。
2. **选优guard有实质性指标引用错误。** `train.py:generation_checkpoint_guards`把
   `no_action_context_exact`赋值为`slice/ramp_merge_exit_exact`，并非GT动作全NO子集exact。
   RAMP_MERGE_EXIT可含变道和速度动作；guard应使用实际NONE签名，并为DECELERATE/RESUME增加明确指标。
   该错误不改变上述重新统计的test准确率，但影响checkpoint选择语义。
3. **此次是final，不是通过守卫的best_generation。** final step9216、3epoch；
   `generation_selection_status.json`声明`best_generation_available=false`、`production_ready=false`。
   step8000 fallback的lane recall45.71%<60%、STOP recall57.38%<80%、所谓no-action15.625%<50%；仅invalid过门。
   不能把完成训练/正常打包当作达标。
4. **训练未显示充分收敛证据。** generation val exact在2000/4000/6000/8000步为31.25/33.07/34.90/36.72%；
   减速召回仍只有3.41/3.41/5.68/6.82%，不是偶然test失常。
   teacher-forced loss为0.1804/0.1778/0.1751/0.1810；低加权loss受到格式token和NO答案影响，不等于动作学好。
   包内train_metrics在500行截断、只到step4990，不能据此断言最后一轮过拟合/没有过拟合。
   `token_acc/value_token_acc`实现是样本内所有相应token同时正确，不是普通逐token平均准确率，读指标时要注明。
5. **可追溯与holdout有限。** adapter保存的训练Git commit为空；README的Git是打包环境记录，不能替代训练源码身份。
   当前prompt hash一致，仍不证明所有训练依赖字节一致。`build_dataset._split`按带Rep/时间戳的完整run_id分割，
   未归并相同物理route的不同采集；原frame_index不在包内，未能量化跨split物理路线重叠。

## 建议决策

优先做一轮“情境/动作阶段标定 + 输入/提示词合同统一 + 评测guard修正”，然后重建索引并重训。
若只想先做低成本排查，可以冻结当前权重进行prompt诊断消融，但必须另存指标并标注训练外prompt，不能据此宣称问题已修复。
本次39.58%是对现有规则标签的一致率；清洗后驾驶语义正确率是多少、哪一类改动贡献最大，仍需独立人工标注集和配对实验回答。
