# Phase3 v24 四包训练审计与历史比较（2026-09-27）

结论：相对上一轮 v23，四组 test 总分和六类 macro F1 均下降，没有超过所查历史最优。减速 P/R/F1 四组均有表观改善，但 STOP、RESUME、左右变道、KEEP 的 F1 四组均下降。主要问题是动作阶段与未来动作预测不稳定，叠加横向训练呈现减少、测试路线大幅收缩和部分提示词/硬窗口语义不一致。当前证据不支持把结果统一归因于“全库错标”，也不能证明回退提示词就能恢复成绩。

本次只增加审计文档及本地证据产物，未改训练、采样、提示词、标签、split 或原审计包，未运行新训练/模型推理。

## 1. 数据来源与核验边界

- A 四图binary：`checkpoints/sft_new_loop_phase3_20260926_182651_4rgb_binary_audit_bundle`。
- B 两图binary：`checkpoints/sft_new_loop_phase3_20260926_222647_2rgb_endpoints_binary_audit_bundle`。
- C 四图choice：`checkpoints/sft_new_loop_phase3_20260927_010408_4rgb_choice_audit_bundle`。
- D 两图choice：`checkpoints/sft_new_loop_phase3_20260927_032511_2rgb_endpoints_choice_audit_bundle`。

四包 production 全部 rank 逐题合并重算；exact、动作 GT/预测计数及召回与 metrics 一致。base/LoRA 对应题的 GT、输入 RGB SHA 和实际 user prompt 完全相同。同题型两/四图全部身份、GT相同；跨题型340题原始动作证据一致，端点RGB SHA一致。四包dataset manifest字节一致，82个保存源码SHA均与本机相应文件一致，无缺失文件。

最近 v23 来自 [09/26审计](AUDIT_COMPARISON_20260926.md) 及保留的 [重算指标](../../checkpoints/phase3_audit_20260926/metrics.json)；旧 v23 四份完整原包本次目录中已不在，未冒称重新读取旧包全部题。v21四包仍在AutoMoT/checkpoints，已重新计算。更早v19/v13/单动作最佳依据既有09/21、09/20等报告。

本轮原索引 train/val/test=14364/408/408；训练实际另用189265行候选池，其中186871正例、2394 INVALID，不能将14364当完整训练池。binary test为340有效题+68 INVALID；choice仅340有效题；有效题每事件34题。

340有效题回读1280份不同原始meta、核对807张不同输入RGB SHA：九点速度窗、anchor brake/throttle、lane/road字段全部一致。155题含微小负速度，最小−0.01364m/s，按原合同归零后没有差异。38条录制路线/4725份meta重放横向规则，与包内lane_change_direction全部一致；纵向规则、raw答案→主要动作投影也一致。这是管线/代码一致性，不等于独立证实地图或驾驶语义。

连续目视复核20段、14物理组、302张不同完整拼接RGB（每段17帧，−0.75至+3.25秒，16错误+4正确对照），覆盖8个context。查看的是缩放联系图面板，未来只用于审计；不是随机抽样或多人盲标，不能估计全库错标率。逐段记录见文末及本地产物。

## 2. 总指标与历史

| 指标 | A 四图binary | B 两图binary | C 四图choice | D 两图choice |
| --- | --- | --- | --- | --- |
| v24 test 整题正确率 | 262/408 = 64.22% | 260/408 = 63.73% | 215/340 = 63.24% | 218/340 = 64.12% |
| v24 仅有效题正确率 | 204/340 = 60.00% | 201/340 = 59.12% | 215/340 = 63.24% | 218/340 = 64.12% |
| v24 六类macro F1 | 52.24% | 53.37% | 56.75% | 54.77% |
| v24 base整题正确率 | 19.12% | 16.91% | 32.35% | 31.47% |
| LoRA相对同题base | +45.10pp | +46.81pp | +30.88pp | +32.65pp |
| 原始单变化动作题 | 179/270 = 66.30% | 176/270 = 65.19% | 166/270 = 61.48% | 169/270 = 62.59% |
| 原始组合证据题 | 3/24 = 12.50% | 4/24 = 16.67% | 10/24 = 41.67% | 9/24 = 37.50% |

| 历史比较 | A 四图binary | B 两图binary | C 四图choice | D 两图choice |
| --- | --- | --- | --- | --- |
| v19 test exact（旧报告） | 60.68% | 61.46% | 67.19% | 65.00% |
| v21 test exact | 63.57% | 62.14% | 66.57% | 63.71% |
| v23 test exact | 70.24% | 71.32% | 68.48% | 67.39% |
| v24−v23 exact | -6.02pp | -7.60pp | -5.24pp | -3.27pp |
| v24−v21 exact | +0.64pp | +1.58pp | -3.34pp | +0.40pp |
| v23 macro F1 | 62.22% | 63.92% | 63.57% | 62.09% |
| v24−v23 macro F1 | -9.98pp | -10.54pp | -6.81pp | -7.32pp |

binary需全部YES/NO正确，choice只答主要动作；不能按总分跨题型排名。binary合法答案保留INVALID误拒/互斥冲突/非法全NO为错误，再投影主要动作，A/B为209/340=61.47%、207/340=60.88%，C/D为63.24%/64.12%。这是离线诊断，不是同预算训练消融。

所查历史完整binary最高表观总分是v23两图71.32%，本轮binary最高64.22%，低7.11pp；完整choice主要动作/NONE历史v13为68.77%，本轮最高64.12%，低4.66pp。旧单动作choice最高79.77%合同不同，不作为当前六类含KEEP任务的可比冠军。历史macro F1也未刷新：v19四组63.02/64.11/66.93/64.25%，均高于本轮。

跨版本题集不同：新v24与v21逐题交集为0；与已保存v23测试物理组交集为0，所以也无相同物理题。以上均为表观差异，不是相同测试集上的模型因果增减。按v23六类GT比例重加权本轮choice召回为64.30%/65.68%，仍低于v23的68.48%/67.39%；类比例变化不能单独解释差值，但这没有控制类内场景难度。

两图−四图：binary净−2题，choice净+3题。物理路线聚类bootstrap 5000次（seed20260927）95%区间分别[−2.86,+1.94]pp、[−2.62,+4.60]pp，均跨零，不能判定四图或两图稳定优越。

## 3. 六类动作与场景

| P / R / F1（%） | A 四图binary | B 两图binary | C 四图choice | D 两图choice |
| --- | --- | --- | --- | --- |
| DECELERATE | 58.95 / 72.73 / 65.12 | 61.54 / 72.73 / 66.67 | 64.00 / 45.71 / 53.33 | 68.89 / 44.29 / 53.91 |
| STOP | 85.48 / 75.71 / 80.30 | 87.93 / 72.86 / 79.69 | 84.38 / 77.14 / 80.60 | 85.71 / 81.43 / 83.52 |
| RESUME | 58.14 / 43.86 / 50.00 | 52.00 / 45.61 / 48.60 | 65.62 / 48.84 / 56.00 | 67.74 / 48.84 / 56.76 |
| LANE_CHANGE_LEFT | 71.43 / 38.46 / 50.00 | 73.33 / 42.31 / 53.66 | 77.78 / 30.43 / 43.75 | 66.67 / 17.39 / 27.59 |
| LANE_CHANGE_RIGHT | 75.00 / 16.67 / 27.27 | 57.14 / 22.22 / 32.00 | 80.00 / 44.44 / 57.14 | 80.00 / 44.44 / 57.14 |
| KEEP | 35.48 / 47.83 / 40.74 | 35.00 / 45.65 / 39.62 | 35.14 / 84.78 / 49.68 | 34.78 / 86.96 / 49.69 |

| F1相对v23变化 | A 四图binary | B 两图binary | C 四图choice | D 两图choice |
| --- | --- | --- | --- | --- |
| DECELERATE | +6.14pp | +7.59pp | +14.37pp | +12.66pp |
| STOP | -6.35pp | -6.73pp | -5.26pp | -2.79pp |
| RESUME | -12.89pp | -16.40pp | -10.67pp | -7.31pp |
| LANE_CHANGE_LEFT | -1.85pp | -7.21pp | -17.12pp | -36.05pp |
| LANE_CHANGE_RIGHT | -39.39pp | -30.50pp | -13.97pp | -6.49pp |
| KEEP | -5.52pp | -10.02pp | -8.24pp | -3.94pp |

binary GT支持数依次77/140/57/26/18/46；choice为70/140/43/23/18/46。左右各仅5个物理组，不能把18或23题当作独立路线。

四图choice：70个减速32正确，30→KEEP，8→STOP；46个KEEP有39正确，但总计预测KEEP 111次，precision仅35.14%。两图KEEP预测115次、40正确，precision34.78%。KEEP召回变高的代价是大量漏动作，不是KEEP能力全面提高。

四图choice左变道23题对7，右变道18题对8；左错16中5个预测了真实伴随速度动作，另有8→KEEP；右错10中9→KEEP、1→LEFT。右变道事故绕行场景为8/9正确，HighwayExit及MergerIntoSlowTraffic共0/9，显示场景内差异；后两者各只有1个物理组，不能泛化为所有匝道都失败。

binary右变道更弱：A只3/18，B只4/18；其中5/6题被直接预测INVALID。A的解释模式对MergerIntoSlowTraffic f98称“not a limited-access highway or ramp corridor”，这是模型生成的理由，不是独立道路真值，但说明应把场景拒绝与动作漏报分开复核。

| 有效题context（各34题） | 物理组 | A 四图binary | B 两图binary | C 四图choice | D 两图choice |
| --- | --- | --- | --- | --- | --- |
| DYNAMIC_CUTIN | 7 | 82.35% | 73.53% | 55.88% | 70.59% |
| JUNCTION_RULE_CONFLICT | 6 | 70.59% | 73.53% | 85.29% | 85.29% |
| LEAD_BRAKE | 5 | 52.94% | 64.71% | 58.82% | 50.00% |
| ONCOMING_INVASION | 5 | 61.76% | 55.88% | 50.00% | 58.82% |
| POST_BYPASS_RETURN | 6 | 26.47% | 29.41% | 47.06% | 44.12% |
| RAMP_MERGE_EXIT | 5 | 41.18% | 32.35% | 29.41% | 29.41% |
| SIGNAL_FAILURE | 5 | 79.41% | 79.41% | 82.35% | 82.35% |
| STATIC_BLOCKAGE | 5 | 44.12% | 41.18% | 55.88% | 52.94% |
| UNSIGNALIZED_PRIORITY | 6 | 52.94% | 52.94% | 79.41% | 79.41% |
| VULNERABLE_CROSSING | 6 | 88.24% | 88.24% | 88.24% | 88.24% |

四图choice合流/出口仅29.41%，前车制动58.82%；无灯路口让行79.41%则高于上一轮63.04%。不是所有场景都下降。

## 4. 学习不足、提示词与标定分别判断

**模型学到了，但动作阶段和未来预测仍弱。** 相对同题base四组增加30.88–46.81pp；production格式均100%合法。相对base修好/改坏分别205/21、217/26、114/9、118/7，不支持“完全没学会”或主要解析故障。

STOP必须拆开：choice当前已确认等待98题，C/D为94/98=95.92%、95/98=96.94%；当前尚在运动、未来才停车42题，仅14/42=33.33%、19/42=45.24%。整体STOP召回77.14%/81.43%掩盖了未来停车弱点。6个确认起步题，四组整题均仅2/6；控制释放和未来速度参与真值，但模型只见RGB历史和当前速度，正确标签不等于输入足以唯一预测。

**v24减速改进有迹象，但不是受控证明。** choice早于/等于0.5秒触发减速36题，C/D均17/36=47.22%；更晚触发34题为15/34=44.12%、14/34=41.18%，上一轮晚减速仅4/43、5/43。近停未在立即窗形成确认对的15题，本轮6/15、5/15；剔除这些边界题后剩余55题也仅各26正确（47.27%）。边界不是全部错误来源，也不能把差值全算作v24提示词收益。

**提示词有一处具体措辞风险。** v23写 `a sustained speed increase is RESUME without requiring a previous stop`；v24改成 `RESUME is a sustained speed increase without prior stopping`。后者容易被读成“之前没有停车才算RESUME”，与同段 `immediate pull-away can be RESUME` 存在理解张力。建议的最小对照措辞是 `RESUME means a sustained speed increase, including pulling away from a stop; a previous stop is not required.` 本次未修改源码；要固定样本、采样、训练预算后比较，不能凭文字发现宣称已找到性能下降根因。

**自然语言与硬标签边界仍有不一致风险。** 速度看未来2秒、STOP两连续0.25秒采样且确认≤1.5秒、横向首跨看3秒；prompt只写immediate/sustained/meaningful。RGB复核StaticCutIn f11为0.25秒和0.5秒两个近停采样、0.75秒已运动，规则标STOP；“持续等待”对这种短停并非天然明确。另一方面StaticCutIn f65到+2.25秒才停，规则必须DECEL；HighwayExit f69直到+2秒才越20%减速阈值，超出仅约0.021m/s。这些说明任务定义/可预测性值得审查，不是数据读取错位，也不应顺着模型答案改标签。

**没有证据支持全局标定误差过大。** 原始数值、图像身份、规则重放一致；连续画面确认了多例真实跨线与等待，并出现“左跨已完成却再答LEFT”的阶段错误。本次未确认需新增的错标修复；这不等于全库零错标，未独立重建全部地图lane section、未随机盲审完整train池，也没有真实驾驶意图的独立逐帧标签。暗处、雾中与阈值附近样本保留不确定性，不按低正确率批量过滤。

## 5. 训练分布和评估设计是重要混杂因素

本轮实际用v24 prompt、`primary_action_capacity_return_v5_fair_cursor` / smooth_cap、完整train候选池、cap8；上一轮实际用v2旧池。invalid容量回流版本写入配置，但包中没有完整逐epoch采样审计，不能据配置断言每轮是否触发、真实cap峰值及累计覆盖都已独立验收。

三轮、seed20260904、4 ranks、视觉LoRA关闭。正例每事件每轮1024，binary另2048 INVALID；三轮总呈现binary36864、choice30720，并非这么多不同物理帧。global_step9216/7680是训练循环样本步，不直接称optimizer更新数。

| choice保存epoch快照 | v21 | v23 | v24 |
| --- | --- | --- | --- |
| DECELERATE | 1881 | 2053 | 2036 |
| STOP | 3600 | 3452 | 3732 |
| RESUME | 1506 | 1683 | 1412 |
| LANE_CHANGE_LEFT | 892 | 684 | 531 |
| LANE_CHANGE_RIGHT | 890 | 684 | 533 |
| KEEP | 1471 | 1684 | 1996 |

左右主要动作从v23各684降到531/533，约−22%；相对v21的892/890累计约−40%。KEEP从1684增到1996（+18.53%），RESUME从1683降到1412（−16.10%），STOP从3452增到3732。事件等量不保证动作等量。这个变化与“更易选KEEP、漏横向/起步”方向一致，但还不是因果证据；STOP呈现增多而测试仍退，说明单一配额解释不够。

同时训练路线覆盖确实更广：choice快照3031→3265组，binary3173→3407组。不能因测试下降就说完整池或公平游标没有价值。只有epoch1快照，不能当三轮累计曝光；完整池也不是每轮全部扫描训练。

测试物理组从v23约117/118降到38，choice左右各从10/15组降到5/5。保存manifest中补齐前val十类候选均0、test仅静态障碍36和无灯路口60，说明大量旧路线被开发隔离后，新holdout主要依赖补齐。本轮满足最低支持不等于有稳定泛化估计。反复查看test后移入train-only会持续改变难度与来源，跨日期排行榜因此很不稳定。

本轮test38组和已导出val32组无交叉，70组与当前1955开发组无交叉；与v23已保存test组亦无交叉。包中无完整训练索引/训练池，不能再独立证明全train/val/test无重叠。已经保存本轮70组曝光清单，未改隔离合同；这些结果应视为开发诊断，下一次盲测必须另留未曝光路线，接入新隔离名单时按原合同重建产物。

## 6. 训练曲线与选优

| 自由生成val | A 四图binary | B 两图binary | C 四图choice | D 两图choice |
| --- | --- | --- | --- | --- |
| 2000 | 57.70% | 57.70% | 58.13% | 58.75% |
| 4000 | 64.49% | 67.36% | 63.12% | 62.19% |
| 6000 | 66.32% | 67.62% | 69.69% | 70.94% |
| 8000 | 69.45% | 71.80% | — | — |
| final | 69.45% | 70.76% | 65.94% | 67.50% |

binary val去重383题，choice320题。相比上一轮final，本轮val四组分别+2.00/+4.61/+0.31/+3.13pp，test却均下降；不能用验证提升替代测试提升，也不应从test下降推断训练整体失效。

choice的step6000→final：C从69.69%降65.94%，D从70.94%降67.50%，分别净少12/11题，存在训练末段退化信号。两者teacher-forced val loss仍从约0.565/0.556降至0.475/0.473（6000），没有final teacher loss及中间权重test结果，不足确诊严重过拟合。binary teacher loss持续下降；B的8000→final下降约1.04pp。

四包都没有通过守卫的best_generation，pipeline均评final，production_ready=false。fallback A/B=8000，C/D=6000；fallback不是合格best，且没有其test成绩，不能声称换fallback一定提升泛化。choice当前实际守卫是六类P/R，binary还看STOP/变道等门槛，不应混用二者阈值。

本轮test失败项：A/B为横向召回、STOP召回、RESUME召回、KEEP P/R等；C/D为减速/RESUME/左右召回以及KEEP precision。全部production格式合法。binary解释模式A/B为62.75%/64.95%，较production−1.47/+1.23pp，格式98.77%/99.26%；要求解释没有一致收益，不能当修复方案。

binary INVALID为A58/68=85.29%、B59/68=86.76%，低于上一轮85/91=93.41%；全部是错误道路负例。没有same-RS错误事件测试支持，evaluation_complete=false；这是证据缺口，不新增开训门槛。choice不评INVALID，无法比较该拒绝能力。

train_metrics.jsonl只导出前500行，完整train.log有到末轮的窗口loss，但窗口波动不能当全rank整轮平均或完整学习曲线。验证墙钟秒/题本轮A/B约1.50/1.30，C/D约0.46/0.29；两图开销较低，但不是控制硬件后的端到端性能测量。

## 7. 建议的下一步顺序

1. **先固定开发比较集与最终盲测集。** 本次及历史已曝光路线用于开发；保留一份未曝光最终测试，所有对照固定数据、标签合同、seed、预算并按物理路线统计区间。历史模型按各自prompt在可比标签的固定开发集重评；适配应明确版本，不能绕过哈希检查。每次换测试集后不能再把差值当升级收益。
2. **优先验证横向呈现下降和KEEP偏置。** 在完整池及cap8内，按同预算做仅改变动作配额的对照，比较当前smooth_cap与恢复较高横向呈现的方案；先容量预检，不破坏每事件1024或同来源INVALID总额。记录每轮各动作实际呈现、不同帧/物理路线/阶段覆盖，不能只看“事件均衡”。
3. **将prompt最小修订独立成对照。** 首先澄清RESUME不要求先停但允许从停起步；再审定短停、速度/横向预测窗口的产品定义，让prompt和标签合同相符。一次只改一个因素，保留原标签；不要同时改阈值、过滤、采样和prompt后解释为某一个词的收益。
4. **保留并评测中间权重。** 下一轮保存6000/8000/final等候选，按预先规定的val守卫选择；当前可在权重仍在训练机时补做fallback开发集诊断，但不能用已看test挑权重，不能降低守卫来制造best。现有包仅元数据，无权重，未执行该推理。
5. **单列难例类型而非批量改标。** 当前等待、未来停车、确认起步、晚减速、窗末阈值、绕障左跨/右归和匝道变道分别报告。对疑似地图假变道做RGB+XODR lane section精确核验；对短历史无法唯一预测的样本标记可辨识性。不要把未来速度/控制喂给模型以虚增准确率。

这轮最值得先做的是固定题集后的采样配额对照、RESUME措辞最小对照和中间权重比较；没有依据先大规模重标全库，也没有依据直接多加训练轮数。

## 8. 连续RGB逐段记录与复现

本地产物：[指标](../../checkpoints/phase3_audit_20260927/metrics.json)、[原始记录核验](../../checkpoints/phase3_audit_20260927/raw_checks.json)、[扩展检查](../../checkpoints/phase3_audit_20260927/extended_checks.json)、[配对区间/场景切片](../../checkpoints/phase3_audit_20260927/extra.json)、[逐段JSONL](../../checkpoints/phase3_audit_20260927/rgb_review_notes.jsonl)、[原始帧SHA与案例](../../checkpoints/phase3_audit_20260927/rgb/evidence.json)、[曝光组](../../checkpoints/phase3_audit_20260927/exposure_groups.json)。重算脚本位于同目录，所有数据留本地。

| 面板 | 案例 / raw动作→输出 | 观察 |
| --- | --- | --- |
| [00](../../checkpoints/phase3_audit_20260927/rgb/case_000.jpg) | VehicleOpensDoorTwoWays f156<br>DECELERATE+LANE_CHANGE_LEFT → DECELERATE | 右侧停放车辆、对向车辆可见；先减速后向左跨中心线，f166 lane -1→+1。预测减速抓到伴随动作，漏掉主要左跨。车门细节在缩略面板中不作独立确认。 |
| [01](../../checkpoints/phase3_audit_20260927/rgb/case_001.jpg) | EnterActorFlow f58<br>LANE_CHANGE_LEFT → KEEP | 雾中高速车道及左侧灰车可见；road切换后f60 lane -5→-4，后续车道线相对位置支持左移。不是仅凭邻车移动认定自车变道；未独立复核XODR lane section。 |
| [02](../../checkpoints/phase3_audit_20260927/rgb/case_002.jpg) | EnterActorFlowV2 f75<br>LANE_CHANGE_LEFT → KEEP | 夜间左前黑车、虚线可见；f85 lane -5→-4，与后续向左移动相符，发生在+2.5秒，输入时尚未跨。KEEP漏掉较晚横向动作。 |
| [03](../../checkpoints/phase3_audit_20260927/rgb/case_003.jpg) | AccidentTwoWays f77<br>RESUME+LANE_CHANGE_LEFT → RESUME | 对向黄车刚通过，右前警车与事故车辆清楚；自车加速后f80从-1到+1左借道。预测RESUME有运动依据，但未遵守主要横向优先级。 |
| [04](../../checkpoints/phase3_audit_20260927/rgb/case_004.jpg) | MergerIntoSlowTraffic f98<br>RESUME+LANE_CHANGE_RIGHT → KEEP | 雨夜跟随黑车，输入仍在原车道；f108从-2到-3，未来右侧车道线与前车相对位置支持右移。发生在+2.5秒；黑暗和后续车道展开增加判断难度，不能据此认定错标。 |
| [05](../../checkpoints/phase3_audit_20260927/rgb/case_005.jpg) | HighwayExit f46<br>LANE_CHANGE_RIGHT → KEEP | 夜间密集车流中，自车f48从-3到-4，左侧黑车相对移向左，右移后沿新车道前进。预测KEEP漏掉实际右跨。 |
| [06](../../checkpoints/phase3_audit_20260927/rgb/case_006.jpg) | AccidentTwoWays f80<br>RESUME+LANE_CHANGE_RIGHT → LANE_CHANGE_LEFT | 与03同路线晚3帧：左借道已在输入末完成；未来f89从+1回-1，回到事故车右侧原路线。预测LEFT重复历史动作，未来应为RIGHT。 |
| [07](../../checkpoints/phase3_audit_20260927/rgb/case_007.jpg) | DynamicObjectCrossing f24<br>STOP → KEEP | 雾中多车、车道可见；自车由6.49m/s快速降至近停，f26至f29连续等待后再释放。STOP数值和画面支持；KEEP漏停车，不能凭远处物体猜独立事件因果。 |
| [08](../../checkpoints/phase3_audit_20260927/rgb/case_008.jpg) | HardBreakRoute f196<br>STOP → RESUME | 前方黑车与对向红车可见；输入已近停，后续3秒保持等待，无起步位移支持。RESUME错误，STOP有明确连续证据。 |
| [09](../../checkpoints/phase3_audit_20260927/rgb/case_009.jpg) | StaticCutIn f11<br>STOP → LANE_CHANGE_RIGHT | 极暗，前车尾灯、部分车道线可见；f12速度归零、f13为0.442，f14为1.449，短暂停顿后迅速起步。规则两点近停因此标STOP；自然语言sustained与这种短停的对应需澄清。未看到自车右跨依据。 |
| [10](../../checkpoints/phase3_audit_20260927/rgb/case_010.jpg) | CrossJunctionDefectTrafficLight f41<br>RESUME → STOP | 路口输入显示刚停下；当前控制释放，未来f42起持续起步通过。RESUME有记录支持，但释放控制不进入模型；输入静止不能确定下一瞬间释放，不能简单改STOP。 |
| [11](../../checkpoints/phase3_audit_20260927/rgb/case_011.jpg) | OppositeVehicleRunningRedLight f45<br>RESUME → STOP | 雾中路口历史几乎静止，未来立即加速并转入路口。支持已确认起步；前后动作边界与不可见控制均可能导致STOP误判，未据灯态推断故障或因果。 |
| [12](../../checkpoints/phase3_audit_20260927/rgb/case_012.jpg) | AccidentTwoWays f16<br>DECELERATE → KEEP | 右前警车、对向来车可见；先明显减速，一次近停后略动，再于+1.5秒近停、+1.75秒确认持续等待。DECEL符合现合同，STOP确认跨硬截止；KEEP仍漏掉明显下降。 |
| [13](../../checkpoints/phase3_audit_20260927/rgb/case_013.jpg) | StaticCutIn f65<br>DECELERATE → STOP | 右侧绿色车辆与对向车流可见；窗内先减速再恢复，持续停车到+2.25秒才开始。GT DECEL，预测STOP对应更远未来；并非原始速度错位。 |
| [14](../../checkpoints/phase3_audit_20260927/rgb/case_014.jpg) | HighwayExit f69<br>DECELERATE → KEEP | 同05路线后段，已完成右跨并直行；速度13.684到+2秒10.926，下降2.758，仅略超20%阈值2.737。KEEP/DECEL是明显阈值敏感例，短RGB不易精确预测2秒端点。 |
| [15](../../checkpoints/phase3_audit_20260927/rgb/case_015.jpg) | InvadingTurn f59<br>KEEP → DECELERATE | 锥桶、对向蓝车可见；当前7.849，未来约7.74–8.91，未达到显著减速阈值。GT KEEP合理；可见风险与历史减速不等于未来必为DECEL。 |
| [16](../../checkpoints/phase3_audit_20260927/rgb/case_016.jpg) | EnterActorFlowV2 f76<br>LANE_CHANGE_LEFT → LANE_CHANGE_LEFT | 与02相邻一帧；同样未来左跨，四图choice此时预测正确。说明此场景不是完全不能识别，但预测对锚点/输入变化不稳定。两题不是独立路线证据。 |
| [17](../../checkpoints/phase3_audit_20260927/rgb/case_017.jpg) | Accident f62<br>RESUME+LANE_CHANGE_RIGHT → LANE_CHANGE_RIGHT | 暗处绕过右侧事故车辆；输入先前左移，未来f69 lane -5→-6，向右恢复后前进。正确RIGHT对照，不能由夜景一概判不可学或错标。 |
| [18](../../checkpoints/phase3_audit_20260927/rgb/case_018.jpg) | AccidentTwoWays f89<br>DECELERATE → DECELERATE | 与03/06同路线，右归已经在输入末完成，后续保持原车道并明显减速。正确DECEL说明历史跨线不应重复输出。 |
| [19](../../checkpoints/phase3_audit_20260927/rgb/case_019.jpg) | CrossJunctionDefectTrafficLight f41<br>RESUME → RESUME | 夜间路口输入已经开始增速，未来继续前进。正确RESUME对照；与10/11的输入仍近零速不同，不能把所有起步题混为同等难度。 |
