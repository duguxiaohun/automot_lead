# 2026-09-11 四图错误逐帧复核与修订

续审更新：累计89例、1,217张不同RGB；新增12例与车道变化时间表见 [第二轮续审](EVAL_REVIEW_20260911_CONTINUED.md)。下方保留首轮范围与结论。

本报告对应 `sft_new_loop_phase3_20260910_203334_4rgb_audit_bundle`，整体指标见
[AUDIT_SUMMARY_20260911.md](AUDIT_SUMMARY_20260911.md)。这里把“GT计算是否一致”和“GT是否有可见语义支持”分开。

当前证据支持三类原因并存：局部道路/事件前提确有错标或确认不足，模型会混淆自车/他车与首次跨线阶段，
短历史也无法充分预示部分停车/释放时刻。长prompt存在重复和建议动作措辞，但其因果影响尚需新模型配对实验。
因此本轮同时做精确标定隔离、短prompt和独立评测划分，不把全部错误归结为某一项。

## 1. 实际查看范围

逐帧查看 **77 个案例：66 个 production 错例、11 个正确对照，58 个采集 run、1,082 张不同原始 RGB**。
覆盖十类context；每例将模型实际四图放在首行，后面是每0.25秒一帧、共3秒的未来取证。
拼图每张RGB缩到576×192以便比较；对道路结构、暗处目标、归位边界另打开原分辨率图细查。
`EVAL_RGB_REVIEW_20260911.jsonl` 记录每例帧号、GT/预测、输入SHA256、观察与结论；
`probe_output/rgb_review_20260911/evidence.json` 保存逐帧速度、road/lane、刹车及原始RGB/meta SHA256。
**这是针对性复核，不是247个错例全部看完，也不能用66个精选错例估计全数据集噪声率。**

先检查评测涉及的207个run：无缺失、无异常时长route。77例模型输入RGB SHA256均匹配包内记录，
未来速度与原始meta完全一致，有效题按现有规则复算的被问动作无不一致。
这只能排除本次样本的路径/数值串错，不能证明道路、事件前提或lane身份全部正确。
未来帧只作标定取证，绝不放入模型输入。

## 2. 已确认的问题和保留的不确定性

|问题|连续帧证据|判断和落实|
|---|---|---|
|错误的R5道路前提|#724/#716 f2–33：护栏旁连续多车道，无当前局部无灯路口；原图f5复看一致|模型INVALID有道理。仅隔离该run实际看过的f2–33，不猜替代R1/R3|
|锥桶被扩展成当前对向侵入|#2 f40–43、#440 f75–78；源规则均为 `event_invading_turn_final_cone_occupation_u5`、vehicle_hazard=false|当前侵入未获可见支持，分别隔离4帧正例；不把雾、远目标或未看清转换为invalid负例|
|真正的当前对向冲突反例|#625：STOP口和右转路径上的SUV，源 `event_oncoming_lane_invasion_vehicle_hazard_confirmed`、hazard=true，目标20.810m|保留。不能把所有InvadingTurn、所有早期帧或所有meta非junction都删掉|
|把别人的动作当自车动作|#5：绿车进入，自车持续静止；模型额外RIGHT|有充分轨迹与RGB支持的模型横向误报|
|把已发生动作重复预测到未来|#7：f31 lane2，anchor f32已lane1，之后保持；模型LEFT+RESUME|取未来首次跨线，不能重复算历史动作|
|左右方向与动作阶段混淆|#27、#555尚需左绕却答右；#572施工在左侧却答左；#587已应右回却答左|保留GT；增加的不是更多绕障例子，而是一条共享首跨规则|
|用最终车道代替首次跨线|#715 f154先左、f161再右；模型RIGHT|GT LEFT依据首次跨线有支持，不能以3秒末位置改标签|
|近停持续时间/窗口边界|#8/#9只有一帧近零；#329/#590第一个近零在1.5s、第二个在1.75s|现有DECEL算术成立，模型STOP反映更宽泛的“即将停车”。新增诊断，保持主指标与阈值|
|短速度脉冲与历史趋势|#88历史制动但未来小降不足20%，之后增速；#422先降后升；#694已恢复增速|不是“建议减速”任务；明确从当前起、按首次达到阈值的变化判断|
|等待与释放时机|#319/#508持续等待却答RESUME；#513/#615立即释放却答STOP|部分是真漏判，部分仅0.25秒采样边界。不能在短历史中假设未来控制动作已可知|
|单纯保持被风险措辞挤掉|#221路口横向车经过、#477对向流经过，但ego未达到减速阈值|原长prompt的让行/等待指令与实际采集行为冲突；改成事实描述|
|启动/可观测性困难|#159/#584启动演员和天气跳变；#15/#83后1.5–2秒才刹车；#261目标极暗|单独报告，不用删难例制造分数，也不声称压短prompt就会解决|

#724的规则链已定位：`turning_trigger_core` 可在没有独立局部路口证据时保留 `junction_window`，
随后 `vehicle_turning_junction_space` 给R5。该帧trigger仅1.218m，但
`turning_local_junction_evidence=false`、`meta_is_junction=false`，无STOP/yield/信号支持，且静态地图不可信。
**scenario触发点不等于局部交叉口**。这与历史VehicleTurningRoute错R5同类，当前先用精确span隔离，
没有凭一个错例全局提高距离阈值。后续应在独立训练路线同时审核“trigger很近但无路口”和真实路口反例，
再决定是否修改通用collector门控。

对向侵入的源规则也有概念问题：锥桶占道/场景仍响应可让标签尾段补成U5，
而本任务U5指的是对向车辆当前侵入。当前只隔离实际复核的8帧；不能用 `vehicle_hazard=false`
直接删除所有U5，它不是完整的视觉actor证据。#625保留是必要反例。

#4初始R5、#25隧道静态占道前提、#26路口退出边界仍保留待核记录；没有足够证据就不改答案。
#234原图为连续建筑街道，支持其人工错R5负例；#672远端道路开口不自动证明当前局部让行。

补查覆盖了全部15个INVALID位分歧（9个未拒绝、6个拒绝），以及2个同RS负例。
9个未拒绝中，8个是错道路前提，1个是#604同RS的错合流事件；模型多回答STOP或NONE而没有核查前提。
6个拒绝中，#716/#724同一条route的R5前提确实有问题；#26/#541/#625属于同一条对向侵入run的路口时序边界；
#552极暗匝道输入仍需更长历史及可信拓扑确认。这些相关帧不能算六条独立视觉证据。
#604和正确对照#637也是完全相同的四图、同一anchor，只换合流/静态障碍问题，进一步证实同RS负例的独立覆盖不足。

## 3. 给Qwen的提示词精简

旧prompt同时有长system、看图流程、场景SCOPE、QUESTIONS、逐动作定义、决策顺序、
纵向互斥、横向边界、导航解释、invalid解释，多处重复。部分SCOPE直接说“must give time and space”或
“Slow or wait while the path is unsafe”，会把实际行为预测推向建议动作。审计中的#20/#221/#477支持删除这类措辞，
但“它导致了多少模型错误”仍需配对实验，不能从生成答案反推内部原因。

当前改成一个短system和一份简短规则表；没有增加思维链或要求模型写长分析：

- system：`Predict the recorded ego vehicle's next actions. Follow the requested YES/NO format.`
- 一行历史时间说明；每图左/前/右拼接仅说明一次。
- 道路和事件只写事实及必要排除边界；已知历史模板压短，未确认绕障不会被写成已完成绕障。
- 纵向只保留STOP优先、1.5秒两点近停、2秒首个幅度变化及20%/1.2m/s阈值。
- 只有问横向时才出现横向规则：3秒内ego首次跨线、忽略已发生和稍后回归，排除弯道/他车/连接路。
- 导航保留本项目Phase3的x前、y负左正右口径，明确终点不等于下一车道；删除重复坐标翻译。
- audit保留严格行格式，允许非空 `unclear`，避免为NO编造“已证明不存在”的证据；解析器没有放宽为空。

按765题真实旧消息和对应新消息统计，含system+user、不含图像token：

|文本|旧版英文词数|新版英文词数|缩减|
|---|---:|---:|---:|
|system|120|12|90.0%|
|仅纵向，均值|1476.1|266.9|81.9%|
|含横向，均值|1838.9|349.4|81.0%|

词数按空白分割，**不是Qwen tokenizer token数**。本地没有对应tokenizer/模型权重，未下载；
图像token开销未下降。新prompt版本 `v7_compact_observed_forecast`，需要按新合同训练，
不能把旧adapter切到新prompt后称为同分布评测。完整示例：
[纵向](probe_output/rgb_review_20260911/prompt_LEAD_BRAKE.txt)、
[含横向](probe_output/rgb_review_20260911/prompt_POST_BYPASS_RETURN.txt)。

## 4. 标定和评测方法改进

新增 `prepare_error_review.py`，自动核验原始图像指纹、原始meta与包内速度，生成输入/未来严格分离的逐帧取证。
人工决定继续通过已有 `mapping_rgb_decisions_v2.jsonl` 生效：本次3条run共40帧，保留原始上游标签以便追溯。
没有改STOP、DECEL、RESUME阈值，也没有将模型答案回填为GT。

`audit_temporal_slices.py` 新增首次减速、增速起点/确认、停车起点/确认的时刻诊断，区分以下重叠切片：

|诊断切片|本轮exact|含义|
|---|---:|---|
|1.5s内有近零但未形成窗内两点确认|15/42|包含单点近停、启动释放和跨窗确认，不是统一标错|
|停车两点确认跨过1.5s边界|7/11|第一点恰在1.5s、第二点1.75s|
|首次达减速幅度仅一采样点|7/23|容易与持续动作语义冲突，先报告，不调阈值|
|确有下降但未达减速幅度|120/191|“稍微慢了”不等于该合同的DECEL|
|静止anchor、未来立即恢复|5/18|最难的释放时机切片之一|
|1.5s后才达减速阈值|12/32|短历史中不一定有充分预示|

这些都是离线诊断，不注入prompt，也不删除难例。后续若要换更平滑/更长期的动作定义，
应另开标签版本、先在训练开发集检验持续时长与幅度，再在未参与调参路线评测，不能直接重算旧test宣称提升。

本次整个765题池已参与指标/提示词开发，因此将207个run对应的206个物理路线组加入开发集合，
后续新val/test排除同物理route的Rep/重复采集；不只排除77个实际看图案例。
原split seed 20260819的test已大量进入开发集合；完整源重建证实其test缺4类，不能继续沿用空桶。
因此按固定日期seed **20260911** 重划新holdout（未看新版模型分数、未搜索分数最优seed）；完整原始候选中val/test十类都有覆盖。
默认新索引目录切为 `checkpoints/sft_new_loop_phase3_data_v8`，mapping/prompt hash已改变，旧v7索引和旧adapter拒绝混用。
新版分数应与同一份新test上的base/LoRA配对比较，旧765题只作回归挑战集。

旧评测同RS错事件负例只有1条独立route，不能靠重复采样通过guard。新日期seed下旧负例的test有2条独立route，val为0。
为补标验证样本，在**短prompt冻结后**，从新val中选择3条未参与错误分析的物理路线，只查看各自4张过去/当前RGB，
不读取任何模型答案、不据这些图继续改prompt。两条空旷高速连接车道确认STATIC_BLOCKAGE与VULNERABLE_CROSSING为错误前提；
另一条高速跟车只确认VULNERABLE_CROSSING错误，不把前车状态或匝道阶段猜成负例。

新增5个同RS问题、3条独立val路线，记录在 `same_rs_invalid_review_v1.jsonl`，带图像SHA256、
`model_outputs_inspected=false` 和冻结prompt SHA256
`7e3c31e89fbb905aae82e38d9c6eb7d91c99e2b86e9bec68e56488bb9f2f3f06`。
这是建标签的盲审，与前面根据错误输出调prompt的77例分开：这12张额外RGB不计入77例的1,082张。
[盲审1](probe_output/rgb_review_20260911/blind_negative_0.jpg)、
[盲审2](probe_output/rgb_review_20260911/blind_negative_1.jpg)、
[盲审3](probe_output/rgb_review_20260911/blind_negative_2.jpg)。
覆盖量仍小，不能宣称覆盖全部十类错事件；最终选优时保留同RS精度和独立路线数门槛。

## 5. 验证及执行

CPU合同、标定边界、配对评测、prompt预算、开发路线与盲标holdout隔离回归：**107项通过**。
小范围每scenario取2条run的构建因val缺少8个context被原有完整性检查拒绝，说明这类过小smoke不足以验证平衡构建；
未降低正式训练的context覆盖要求。全部审计缓存仍缺2个val类别；随后完整源在旧seed下缺4个test类别。
固定日期seed20260911的全源重建已通过，候选193,726行，最终索引14,472行：

|划分|总行数|每个有效context|invalid|同RS错事件题／独立route|
|---|---:|---:|---:|---:|
|train|13,524|1,127|2,254|37／10|
|val|396|33|66|2／2|
|test|552|46|92|2／2|

索引中的val/test各题不重复，train单个题目身份最大重复4次。新补标val负例有5个候选，正式平衡索引选入2题、2条独立route；
不能把3条已盲标route都写成实际入选。val/test同RS负例均仅覆盖2类事件，仍是最小覆盖，不能宣称充分评估十类拒绝能力。
五种被问动作在val/test均有正负例，三个划分的十类有效context全部齐全。

最终索引含3,274个run、3,273个物理路线组，划分间物理路线交叉为0；518个开发组均未进入val/test。
本次隔离的40帧在193,726行候选和14,472行最终索引中均无残留；确认实际生效，不只检查规则函数。
逐行prompt/答案复算通过，检查35,682条不同RGB文件路径均存在。完整精度速度的标签不一致为0。
两条train样本若使用展示用三位小数重算会改变边界判断，因此复核必须用 `future_speeds_exact_mps` 或原始meta，
不能从拼图的三位小数认定GT错误。最终回读 **3,274条run的14,472行原始meta**，所有行速度一致，有效题被问动作复算不一致为0；
无异常时长route、无横向观测不完整却监督横向的有效题。这是原始记录一致性验证，不等于全量RGB人工确认。

构建产物：[final_index/manifest.json](probe_output/rgb_review_20260911/final_index/manifest.json)。
审计产物：[索引/输入核验](probe_output/rgb_review_20260911/rebuilt_index_audit.json)、
[原始meta全量核验](probe_output/rgb_review_20260911/raw_index_audit.json)、
[隔离与开发集核验](probe_output/rgb_review_20260911/review_completion_checks.json)、
[时间边界统计](probe_output/rgb_review_20260911/temporal_index_audit.json)。

也调用了真实train/eval采样函数：默认generation val为384次呈现、383个独立题，同RS两条route均保留、无重复。
其中一个普通错RS负例重复一次，应区分呈现数与独立题数。默认test每类64的采样在每类仅46的此索引上先补齐再去重，
最终549题，比完整索引少3个普通负例，来源类别最大差异超过1。
因此本次推荐 **`CASES_PER_BIN=0`**，使用既有全量评测接口；实测恰为552个独立题，同RS两条route保留，
来源类别、真实RS、错问context和配额检查全部通过。未改模型输出或借重复题扩大独立覆盖。
[实际采样核验](probe_output/rgb_review_20260911/runtime_sampling_check.json)。

**尚未运行新版Qwen生成或训练；提示词变短是已测事实，准确率提升尚未验证。**

从AutoMoT目录重现这77例的原始数据核验和逐帧取证：

```bash
python qwen3vl_local/sft_new_loop_phase3/prepare_error_review.py \
  --bundle checkpoints/sft_new_loop_phase3_20260910_203334_4rgb_audit_bundle \
  --data-root lead_data \
  --output qwen3vl_local/sft_new_loop_phase3/probe_output/rgb_review_20260911 \
  --extra-ids 106 297 523 541 552 604 637 716 744
```

此脚本只准备证据，不会自动声明“人工看过”。人工结论以持久化JSONL为准。

从AutoMoT目录正式重建并训练：

```bash
CASES_PER_BIN=0 bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```

脚本默认使用新v8目录，不设置SKIP_BUILD跳过合同变更；如果需保留旧实验，使用其原始源码/提示词合同重现。

## 6. 逐例笔记

完整记录见 [EVAL_RGB_REVIEW_20260911.jsonl](EVAL_RGB_REVIEW_20260911.jsonl)。

### #0 StaticCutIn f69 — 正确对照

GT：`STOP`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_000.jpg)

f66–69自车静止，右前蓝车贴近；f70仍低速，随后起步，f76后再跟车减速。STOP符合当前等待优先，不能仅看未来起步改RESUME。

### #2 InvadingTurn f43 — 正确动作／情境时序待核

GT：`DECELERATE`；预测：`DECELERATE`。[连续帧](probe_output/rgb_review_20260911/case_002.jpg)

f40–43沿林间弯道前进，远端锥桶和车辆小；f48–51开始明显减速，对向灰车f52–55才近身。DECEL正确；anchor是否已构成明确侵入需源标注回查，不能凭锥桶认定。 源规则补查：event_invading_turn_final_cone_occupation_u5，vehicle_hazard=false，trigger=56.516m。锥桶尾段补全不能确认当前对向侵入；f40–43按前提未确认隔离，不作为invalid负例。

### #4 NonSignalizedJunctionLeftTurnEnterFlow f0 — 正确动作／启动与情境边界

GT：`STOP`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_004.jpg)

输入四张都为f0；f1天气和前车出现，随后起步。当前及下一帧静止符合STOP。首帧道路是住宅走廊，局部路口未明显进入视野，R5情境可能提前，不能将正确动作等同于正确上下文。

### #5 DynamicObjectCrossing f67 — 模型横向误报

GT：`STOP`；预测：`LANE_CHANGE_RIGHT+STOP`。[连续帧](probe_output/rgb_review_20260911/case_005.jpg)

f64–67自车静止，行人及右侧绿车活动；f68–75继续静止，f76后沿原lane1起步至f79。另车切入和道路弯曲均不能证明ego右变道，GT STOP有支持。

### #6 HardBreakRoute f12 — 正确对照／短脉冲标定

GT：`DECELERATE`；预测：`DECELERATE`。[连续帧](probe_output/rgb_review_20260911/case_006.jpg)

f9–12雨雾中行进跟车；速度先升7.115后回6.081，下一帧3.497，再回升，车辆相对距离变化不大。单帧减速符合规则，但该规则监督短控制脉冲而非持续减速，不能从答对推导精确运动理解。

### #7 ParkingExit f32 — 已发生跨线／未来时窗误报

GT：`NONE`；预测：`LANE_CHANGE_LEFT+RESUME`。[连续帧](probe_output/rgb_review_20260911/case_007.jpg)

f29–32从右侧停车位/边缘驶向主路，f33–44沿新车道跟车；模型把输入中已发生的左移与加速外推为未来动作。需核对跨线发生在anchor前后；GT NONE还受到速度变化阈值影响。 原始meta补查：f31仍lane2、f32已lane1，确认跨线发生在anchor，不应重复算未来LEFT。

### #8 CrossJunctionDefectTrafficLight f32 — 停车持续时间边界

GT：`DECELERATE`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_008.jpg)

f29–32夜雨接近横向车辆，明显减速；f33速度近0，但f34已0.855，随后加速。只有一个近零采样，规则DECEL而模型STOP。是短停车与两点持续停车定义边界，RGB无法可靠分辨0.25s驻留；不直接把GT改STOP。

### #9 CrossJunctionDefectTrafficLight f32 — 停车持续时间边界（同窗）

GT：`DECELERATE`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_009.jpg)

与#8完全相同输入与未来窗口，改问信号故障context；同为DECEL预测STOP，不计作第二条独立视觉证据。

### #12 CrossJunctionDefectTrafficLight f40 — 正确对照

GT：`DECELERATE`；预测：`DECELERATE`。[连续帧](probe_output/rgb_review_20260911/case_012.jpg)

f37–40雾中通过信号路口并沿左弯，f41–42速度8.183降5.545后起伏继续通过。DECEL正确；f50–52不存在，但所问纵向2秒窗口完整。

### #15 NonSignalizedJunctionLeftTurnEnterFlow f40 — 未来窗末减速／输入信息不足

GT：`DECELERATE`；预测：`NONE`。[连续帧](probe_output/rgb_review_20260911/case_015.jpg)

f37–40在路口转入新道路，输入速度持续增加至11.289；f41–46基本保持，f47–48才减到7.725，左侧对向车经过。NONE漏掉2秒窗末减速，但输入0.75秒不能直接观察后续控制时机。

### #18 OppositeVehicleRunningRedLight f22 — 模型漏停车／提前恢复

GT：`STOP`；预测：`RESUME`。[连续帧](probe_output/rgb_review_20260911/case_018.jpg)

f19–22接近信号路口，输入末段已明显制动至4.504；f23–34持续静止，右侧厢式车f26后横穿。RESUME与实际持续等待相反，不能因前方局部开阔就视为已放行；需原图补查anchor右侧目标可见程度。 原图补查：anchor右侧厢式车未清楚进入视野；停车有记录支持，但不能把稍后横穿车当成模型已经看见的输入证据。

### #20 ConstructionObstacle f116 — 行为目标与安全措辞冲突／速度阈值

GT：`LANE_CHANGE_LEFT`；预测：`DECELERATE+LANE_CHANGE_LEFT`。[连续帧](probe_output/rgb_review_20260911/case_020.jpg)

f113–116加速接近占道施工牌，f117跨左线并持续通过；GT LEFT，模型额外DECEL。实测速度窗口未低于初速，减速不成立；接近障碍的安全减速建议与采集驾驶员继续加速不同。RESUME受两点阈值判据限制，不可凭视觉重标。

### #21 AccidentTwoWays f82 — 弱光与控制速度波动

GT：`DECELERATE`；预测：`NONE`。[连续帧](probe_output/rgb_review_20260911/case_021.jpg)

f79–82夜间林道，对向车辆近身并向左后方通过；f83–94沿原车道继续行驶，速度7.431降5.100再反复波动。GT DECEL算术成立，画面弱光且缺历史速度，不能仅据NONE认定标定错。

### #22 ConstructionObstacle f66 — 正确横向对照

GT：`LANE_CHANGE_LEFT`；预测：`LANE_CHANGE_LEFT`。[连续帧](probe_output/rgb_review_20260911/case_022.jpg)

f63–66接近施工牌，f67 lane2→1跨左线，之后跟车沿锥桶左侧行驶。LEFT符合RGB与meta；速度窗口波动但未达减速阈值，GT无纵向动作。

### #24 OppositeVehicleRunningRedLight f97 — 正确停车对照

GT：`STOP`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_024.jpg)

f94–97雾中跟车静止，侧向白车经过；f98–109自车持续不动，前方与右前车辆未放行。STOP有直接历史和未来支持。

### #25 Accident f72 — 漏减速／双动作

GT：`DECELERATE+LANE_CHANGE_LEFT`；预测：`LANE_CHANGE_LEFT`。[连续帧](probe_output/rgb_review_20260911/case_025.jpg)

f69–72隧道内跟红车行进；f75速度8.246→6.557达20%减速，f79–80开始左移并跨到lane-1。LEFT正确但漏DECEL，当前跟车状态与2秒后的左移是不同阶段；静态占道前提在本窗口缺清楚近端证据，待标注源核查。 原图补查：隧道前方拥车可见，但当前ego路径是否被静态事故车直接占据仍不够明确；保留前提待核，不重标。

### #26 InvadingTurn f36 — 情境边界／可能过早拒绝

GT：`DECELERATE`；预测：`INVALID_ACTION_CONTEXT`。[连续帧](probe_output/rgb_review_20260911/case_026.jpg)

f33–36经过住宅路口出口，黑色对向SUV靠中心线，f37–40近身并伴随7.151降1.603；R5局部路口在身后/边缘，模型INVALID可能源于道路适用边界。需明确有效前提保留到何时；不能只按scenario断定R5正确或错。

### #27 HazardAtSideLaneTwoWays f187 — 明确首跨方向错误

GT：`LANE_CHANGE_LEFT+RESUME`；预测：`LANE_CHANGE_RIGHT`。[连续帧](probe_output/rgb_review_20260911/case_027.jpg)

f184–187贴近同向骑行者，f188 lane1→-1向左借道，f189–199继续加速从骑行者左侧通过，没有右回。GT RESUME+LEFT，模型RIGHT把待绕出误成归位；输入尚未清掉骑行者。

### #29 HighwayExit f82 — 正确无动作对照

GT：`NONE`；预测：`NONE`。[连续帧](probe_output/rgb_review_20260911/case_029.jpg)

f79–82右侧高速车道已减速到8.140，f83–94保持约8m/s同lane-4；邻车在左侧移动，自车未跨边界。NONE符合记录；匝道箭头出现不自动等于变道。

### #31 StaticCutIn f60 — STOP与DECEL时间边界

GT：`DECELERATE`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_031.jpg)

f57–60雨雾中右侧切入车接近；f61–65仍滚动，f66(1.5s)1.969，f67(1.75s)才近零并持续等待。GT DECEL因停车超出1.5s，模型STOP符合较长窗内停车但不符当前合同，不能简单算不懂风险。

### #36 InvadingTurn f48 — 模型漏减速

GT：`DECELERATE`；预测：`NONE`。[连续帧](probe_output/rgb_review_20260911/case_036.jpg)

f45–48高速接近窄道上对向灰车，f49–51降速19.447→14.667，f52–53车辆近身通过后回升。NONE漏掉明确减速；所问纵向2秒完整，后续f57起缺帧不影响该动作。

### #41 VehicleTurningRoute f197 — 弱光／漏减速与错加速

GT：`DECELERATE`；预测：`RESUME`。[连续帧](probe_output/rgb_review_20260911/case_041.jpg)

f194–197极暗道路前方红尾灯，速度约2.8；f198–199小增速未达1.2阈值，f201降0.188，随后低速爬行。RESUME与原始轨迹相反，GT DECEL因近零非连续；灯点外缺可靠车道/冲突证据，不臆测具体车辆意图。

### #44 InvadingTurn f93 — 侵入已过／窗末控制减速

GT：`DECELERATE`；预测：`NONE`。[连续帧](probe_output/rgb_review_20260911/case_044.jpg)

f90–93雨中对向车已贴近并通过，f94–100维持约8.3，f101才降6.6、f102降2.35。GT DECEL由窗末变化触发，输入容易支持保持行驶。未来晚刹车的触发对象需meta/bbox核查，不能只加长风险描述解决。

### #50 EnterActorFlow f59 — 正确无动作对照／历史跨线

GT：`NONE`；预测：`NONE`。[连续帧](probe_output/rgb_review_20260911/case_050.jpg)

f56–59极暗高速车流，lane-2→-3已发生在输入期间，之后f60–71同lane-3跟车。速度下降未达20%阈值，NONE正确。历史变道与未来变道应分开。

### #58 Accident f70 — 首跨方向／未来窗末

GT：`LANE_CHANGE_RIGHT`；预测：`LANE_CHANGE_LEFT`。[连续帧](probe_output/rgb_review_20260911/case_058.jpg)

f67–70雨夜沿事故车辆左侧行进，f71–78仍保持，f79–82才向右跨回lane2。GT RIGHT与窗末回归有支持，模型LEFT似继续套用绕出；短历史未给明确已完成绕障状态，需原图确认实际边界。 原图f82补查：自车位于右侧车道、左边虚线与右边缘清楚，支持最后一帧右回而非左绕。

### #70 HazardAtSideLaneTwoWays f76 — 正确联合动作对照

GT：`RESUME+LANE_CHANGE_LEFT`；预测：`RESUME+LANE_CHANGE_LEFT`。[连续帧](probe_output/rgb_review_20260911/case_070.jpg)

f73–76夜雨接近右前骑行者，当前降2.814；f77后加速，f84→85 lane1→-1左借道，f86–88通过。RESUME+LEFT正确，晚跨线仍是未来预测。

### #83 MergerIntoSlowTraffic f77 — 晚减速／可观测性

GT：`DECELERATE`；预测：`NONE`。[连续帧](probe_output/rgb_review_20260911/case_083.jpg)

f74–77右侧合流支路约11m/s，左邻橙车并行；f78–82保持，f83–84才减到8.494。GT DECEL来自1.75秒处，输入没有明显速度/间距变化，NONE属于未来时机漏判，不能据此判RGB标签错误。

### #88 HardBreakRoute f264 — 阈值与历史趋势相反

GT：`RESUME`；预测：`DECELERATE`。[连续帧](probe_output/rgb_review_20260911/case_088.jpg)

f261–264夜间跟车已制动，v11.578降8.749；f265再降7.557但不足1.750阈值，之后加速14.210。规则RESUME，模型DECEL延续历史制动。真实短减速存在但不够幅度，属于离散阈值和历史/未来阶段边界。

### #90 HazardAtSideLane f67 — 联合动作漏报／晚减速

GT：`LANE_CHANGE_LEFT+DECELERATE`；预测：`NONE`。[连续帧](probe_output/rgb_review_20260911/case_090.jpg)

f64–67夜雨跟车接近右前骑行者，f70–71向左跨线，f74(1.75秒)才达到减速阈值，随后经过骑行者。GT LEFT+DECEL有支持；NONE同时漏首跨和晚减速，弱光下骑行者较小但可见。

### #106 StaticCutIn f73 — 模型未拒绝错道路

GT：`INVALID_ACTION_CONTEXT`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_106.jpg)

f70–73雨雾普通双黄线道路，右侧红SUV贴近，f74仍低速后自车起步。题目给高速匝道R3，RGB无匝道布局；STOP虽符合当前等待，未先拒绝错前提。

### #109 EnterActorFlow f9 — 短减速／暗夜

GT：`DECELERATE`；预测：`NONE`。[连续帧](probe_output/rgb_review_20260911/case_109.jpg)

f6–9暗夜单通道跟车，输入末段已从8.834降7.414，f10–11约5.4后再加速。GT DECEL支持；NONE漏掉已有制动趋势，远端红灯点不能单独解释模型错误。

### #159 AccidentTwoWays f2 — 启动帧不连续／未来释放

GT：`RESUME`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_159.jpg)

输入f0重复、f1/f2天气由白天突变夜间并出现右侧黄车；当前0.002，下一帧1.519后持续起步。GT RESUME符合记录，模型STOP基于静止历史；启动环境跳变使运动证据更弱，不能从历史精确推出下一帧释放。

### #160 ConstructionObstacleTwoWays f173 — 模型未拒绝错情境

GT：`INVALID_ACTION_CONTEXT`；预测：`RESUME`。[连续帧](probe_output/rgb_review_20260911/case_160.jpg)

f170–173道路为连续城市走廊、前方施工牌和对向车；f174–185左借道绕施工，未见R5局部路口让行。题目人为给R5无灯路口，模型RESUME虽吻合速度增长却没有先拒绝错误情境。

### #208 StaticCutIn f50 — 漏预测持续停车

GT：`STOP`；预测：`DECELERATE`。[连续帧](probe_output/rgb_review_20260911/case_208.jpg)

f47–50城市走廊左侧蓝车通过，前方红车，输入末段已制动；f51–53继续降速，f54–56持续近零。GT STOP支持，模型DECEL低估减速终点；不是一帧零速歧义。

### #219 ParkedObstacle f66 — 模型未拒绝错情境

GT：`INVALID_ACTION_CONTEXT`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_219.jpg)

f63–66宽直城市道路跟红车静止，左邻车辆通过；f67–78始终静止，无局部无灯交叉口。STOP符合实际动作但题目人为R5前提不符，GT INVALID有视觉依据。

### #221 CrossJunctionDefectTrafficLight f26 — 减速误报／风险描述与实际行为

GT：`NONE`；预测：`DECELERATE`。[连续帧](probe_output/rgb_review_20260911/case_221.jpg)

f23–26驶入路口，红车从右向左横穿；f27–38继续转弯通过，v7.484仅降6.616，未达1.497阈值。模型DECEL可由冲突风险联想，但GT NONE符合实际速度幅度；不能以应当让行取代记录预测。

### #223 StaticCutIn f49 — 持续停车漏报（相邻窗）

GT：`STOP`；预测：`DECELERATE`。[连续帧](probe_output/rgb_review_20260911/case_223.jpg)

与#208同run前一anchor，f46–49输入仍增速但蓝车近身；f50–53连续减速，f54–55在1.25–1.5s持续近零。GT STOP，模型DECEL；同路线相关错例不计独立场景。

### #234 HardBreakRoute f53 — 错情境拒绝失败／道路边界待细查

GT：`INVALID_ACTION_CONTEXT`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_234.jpg)

f50–53前方灰车和左邻橙车，f54–65自车持续静止；模型STOP符合轨迹但未拒绝人工R5前提。建筑转角容易诱导路口解释，需原图核查右侧是否真实交叉道路，不能仅用源R1宣称RGB明确矛盾。 原图补查：两侧连续建筑与人行道，当前是跟车走廊，GT错R5前提有支持。

### #243 ConstructionObstacleTwoWays f82 — 提前归位和加速／低速爬行边界

GT：`DECELERATE`；预测：`RESUME+LANE_CHANGE_RIGHT`。[连续帧](probe_output/rgb_review_20260911/case_243.jpg)

f79–82借到对向车道面对近白车、右侧施工锥桶；f83–94车流未清、自车低速起伏，没有右跨，预测RESUME+RIGHT明显提前。GT DECEL由2.207降0.251触发，近停采样不连续；要保留等待与归位阶段区别。

### #261 HazardAtSideLaneTwoWays f99 — 弱光首跨错误／纵向阶段错误

GT：`DECELERATE+LANE_CHANGE_LEFT`；预测：`RESUME+LANE_CHANGE_RIGHT`。[连续帧](probe_output/rgb_review_20260911/case_261.jpg)

f96–99极暗窄道接近骑行者，f101–102再次减到3.134，f106后加速、f108 lane1→-1左借道；预测RESUME+RIGHT错两项。图中目标很暗，左右判别需原图补查，不用元数据独立证明可见性。 原图补查：骑行者在右侧白线附近极暗可辨，不能声称目标不存在；未来左借跨线与meta吻合。

### #297 InvadingTurn f127 — 模型未拒绝错路口

GT：`INVALID_ACTION_CONTEXT`；预测：`NONE`。[连续帧](probe_output/rgb_review_20260911/case_297.jpg)

f124–127乡间窄道左侧锥桶、右側木栏，前方对向车；f128–139减速后让过对向车，全窗无局部交叉口。GT人工R5错误前提有支持，模型NONE未拒绝。

### #319 AccidentTwoWays f17 — 模型跳过当前等待

GT：`STOP`；预测：`RESUME`。[连续帧](probe_output/rgb_review_20260911/case_319.jpg)

f14–17暗夜驶近STOP路口，前车刚离开；f18后降至零并持续等待至f29，右侧STOP牌可见。模型RESUME过早，GT STOP得到当前减速、停车位置及未来速度共同支持。

### #321 ConstructionObstacleTwoWays f77 — 漏加速／归位方向正确

GT：`LANE_CHANGE_RIGHT+RESUME`；预测：`LANE_CHANGE_RIGHT`。[连续帧](probe_output/rgb_review_20260911/case_321.jpg)

f74–77已左借道，当前8.174；f78–80持续加速9.449、10.951、11.415，f86跨回本侧lane1。RIGHT正确但漏RESUME，施工仍在前侧不代表采集驾驶员不加速。

### #329 HardBreakRoute f8 — 停车确认跨窗边界

GT：`DECELERATE`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_329.jpg)

f5–8夜间跟车制动，当前2.709；随后速度多次起伏，f14(1.5s)首次近零、f15(1.75s)第二次近零。GT DECEL符合两点必须在1.5s内的规则，模型STOP抓到稍晚停车。

### #389 HardBreakRoute f7 — 小幅减速低于阈值

GT：`NONE`；预测：`DECELERATE`。[连续帧](probe_output/rgb_review_20260911/case_389.jpg)

f4–7白天跟车，之后最低6.171，相对当前7.327下降1.156，小于1.465阈值。GT NONE算术一致，模型DECEL可能响应小幅跟车调整，不能直接认定无减速视觉变化。

### #397 HazardAtSideLaneTwoWays f89 — 减速超窗／归位正确

GT：`LANE_CHANGE_RIGHT`；预测：`DECELERATE+LANE_CHANGE_RIGHT`。[连续帧](probe_output/rgb_review_20260911/case_397.jpg)

f86–89已借到对向lane-1通过骑行者，f92右跨回lane1；2s内从11.140降9.236未达2.228阈值，f98(2.25s)才明显降7.111。RIGHT正确，额外DECEL对应稍晚的真实减速。

### #415 HazardAtSideLane f36 — 模型漏左跨／轻减速不足阈值

GT：`LANE_CHANGE_LEFT`；预测：`DECELERATE`。[连续帧](probe_output/rgb_review_20260911/case_415.jpg)

f33–36夜间城市场景，前方骑行者小，左侧灰车随后近身超越；ego到f48(3s)才跨lane2→1，2s最低8.561相对9.502不足1.900减速阈值。GT LEFT，模型DECEL；要区分左邻车超越与自车更晚的左跨。

### #422 Accident f59 — 先减速后加速／纵向阶段错误

GT：`LANE_CHANGE_RIGHT+DECELERATE`；预测：`LANE_CHANGE_RIGHT+RESUME`。[连续帧](probe_output/rgb_review_20260911/case_422.jpg)

f56–59白天事故车流，自车加速至6.015但anchor制动；f60立即降2.618，然后增速9.044，f64跨右lane2。GT DECEL+RIGHT先减后增，模型RESUME+RIGHT沿历史加速，右跨正确。

### #440 InvadingTurn f78 — 情境提前候选／阈值边界

GT：`NONE`；预测：`DECELERATE`。[连续帧](probe_output/rgb_review_20260911/case_440.jpg)

f75–78住宅直道，前方无近处对向侵入，锥桶f87后更清楚；2s速度9.239最低7.676，下降1.563小于1.848阈值。GT NONE计算一致，模型DECEL；ONCOMING_INVASION当前前提值得源标注检查，不能把未来锥桶视为已侵入。 源规则补查：event_invading_turn_final_cone_occupation_u5，vehicle_hazard=false，trigger=42.614m。当前原图无可确认侵入目标，f75–78按前提未确认隔离，不把不可见等同于不存在。

### #455 StaticCutIn f77 — 明确持续停车漏报

GT：`STOP`；预测：`DECELERATE`。[连续帧](probe_output/rgb_review_20260911/case_455.jpg)

f74–77雾中多车交汇，绿车停在本车前方，输入已减速；f79降1.148，f80–87持续近零。GT STOP而预测DECEL，阻挡和停车得到连续帧支持。

### #477 InterurbanAdvancedActorFlow f60 — 风险引导的减速误报

GT：`NONE`；预测：`DECELERATE`。[连续帧](probe_output/rgb_review_20260911/case_477.jpg)

f57–60雨雾无灯路口面对对向车，f61–72连续约11m/s完成转入新路。无达阈值减速，GT NONE；模型DECEL更像对交叉流让行的常规反应，不能把安全措辞当行为标签。

### #484 VehicleOpensDoorTwoWays f132 — 漏归位／减速幅度不足

GT：`LANE_CHANGE_RIGHT`；预测：`DECELERATE`。[连续帧](probe_output/rgb_review_20260911/case_484.jpg)

f129–132借到左侧绕开右侧开门蓝车，f133–136右回lane1，之后沿本侧通过。当前8.378最低7.570，未达1.676阈值；GT RIGHT，模型DECEL既漏右回又把小幅波动离散成减速。

### #508 InterurbanAdvancedActorFlow f54 — 当前等待被提前起步替代

GT：`STOP`；预测：`RESUME`。[连续帧](probe_output/rgb_review_20260911/case_508.jpg)

f51–54在STOP线前制动、近零和爬行交替，f55–66继续近零等待，横向车辆后来通过。GT STOP，模型RESUME；前方暂时空不等于采集驾驶员下一帧释放。

### #513 OppositeVehicleRunningRedLight f23 — 启动时机漏判／安全与记录分离

GT：`RESUME`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_513.jpg)

f20–23在信号路口静止后开始0.181滚动，f24–29持续加速到7.099，之后左弯行进。GT RESUME，模型STOP沿历史等待；画面信号状态不能替代实际记录，不能据起步标签认证通行合规。

### #523 InvadingTurn f92 — 模型未拒绝错路口

GT：`INVALID_ACTION_CONTEXT`；预测：`NONE`。[连续帧](probe_output/rgb_review_20260911/case_523.jpg)

f89–92住宅直道对向蓝车沿中心线经过，锥桶在左；f93–104继续直行并有后续对向车。当前不在先前的STOP口，GT人工R5前提不符，模型NONE把无动作混同于前提有效。

### #541 InvadingTurn f28 — 路口事件时序／不能据未看清否定

GT：`DECELERATE`；预测：`INVALID_ACTION_CONTEXT`。[连续帧](probe_output/rgb_review_20260911/case_541.jpg)

与#625/#26同run中间anchor：f25–28仍在明确STOP口右转，黑SUV在转入道路上接近，f30速度3.716→1.189；随后SUV近身。GT DECEL有原始记录支持，R5道路明确；事件的当前侵入可见度有限，不因模型INVALID直接把正例改负。

### #552 EnterActorFlow f15 — 极暗输入／事件前提待核

GT：`RESUME`；预测：`INVALID_ACTION_CONTEXT`。[连续帧](probe_output/rgb_review_20260911/case_552.jpg)

f12–15护栏间同lane2跟车，之后7.327增速至11.3；GT RESUME算术成立。画面极暗，当前匝道过渡证据不清楚，不能只凭scenario判真实合流，也不能因为暗而判INVALID。保留待拓扑及更长历史回查。

### #555 Accident f50 — 首跨方向错误／虚报减速

GT：`LANE_CHANGE_LEFT`；预测：`DECELERATE+LANE_CHANGE_RIGHT`。[连续帧](probe_output/rgb_review_20260911/case_555.jpg)

f47–50雾中前方事故车辆，自车到f51跨lane4→3向左，之后维持8m/s附近。GT LEFT；模型RIGHT+DECEL，当前仍是绕出并非右回，2s最低7.611未达1.705幅度。

### #572 ConstructionObstacle f50 — 施工绕障方向错误

GT：`LANE_CHANGE_RIGHT`；预测：`DECELERATE+LANE_CHANGE_LEFT`。[连续帧](probe_output/rgb_review_20260911/case_572.jpg)

f47–50前方本车道施工牌，右邻黑车继续行驶，f51自车向右跨lane2→3经过施工。GT RIGHT，模型LEFT+DECEL；绕障不能固定答左，当前7.980到2s最低7.603不够减速幅度。

### #584 ConstructionObstacle f1 — 启动输入与左右错误

GT：`LANE_CHANGE_RIGHT+STOP`；预测：`STOP+LANE_CHANGE_LEFT`。[连续帧](probe_output/rgb_review_20260911/case_584.jpg)

输入f0重复三次、f1突然出现车流；f2仍近零符合STOP，f7向右跨lane1→2跟车，之后保持。GT STOP+RIGHT，模型STOP+LEFT；启动时演员突现使历史运动不可信，但实测首跨与RGB方向吻合。

### #587 ConstructionObstacle f88 — 明确归位方向错误

GT：`LANE_CHANGE_RIGHT`；预测：`LANE_CHANGE_LEFT`。[连续帧](probe_output/rgb_review_20260911/case_587.jpg)

f85–88沿左侧车道接近蓝车，f89向右跨lane1→2后保持右侧行驶。GT RIGHT，模型LEFT；右侧空间和跨线得到白天连续RGB支持，不由终点左右符号决定。

### #588 EnterActorFlowV2 f69 — 把连接路拓扑切换当变道

GT：`NONE`；预测：`LANE_CHANGE_LEFT`。[连续帧](probe_output/rgb_review_20260911/case_588.jpg)

f66–69夜间驶过入口连接处，road1374→72已发生在历史；之后稳定20m/s保持lane-5，无自车左跨。GT NONE，模型LEFT。道路编号和连接路延续不等于跨线，不能按合流context固定触发。

### #590 StaticCutIn f60 — 停车确认跨窗边界

GT：`DECELERATE`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_590.jpg)

f57–60晴天右侧黑车经过，多车随后交叉；f66(1.5s)首次近零、f67(1.75s)第二次近零。GT DECEL预测STOP，与#329同类，两点确认超出当前窗口。

### #598 StaticCutIn f75 — 明确停车漏报

GT：`STOP`；预测：`DECELERATE`。[连续帧](probe_output/rgb_review_20260911/case_598.jpg)

f72–75白天右侧蓝车近身，输入末段仍8.333；f78降6.393，f79–81持续近零，之后等待。GT STOP预测DECEL，未来停车在1–1.5s内有充分原始记录，不是跨窗歧义。

### #604 HighwayCutIn f16 — 同RS错事件／把邻车并行当合流

GT：`INVALID_ACTION_CONTEXT`；预测：`NONE`。[连续帧](probe_output/rgb_review_20260911/case_604.jpg)

f13–16高速主线右侧护栏连续，蓝车在左邻车道、绿车前左，自车lane3约7.2保持至f28。无当前匝道/分合流几何，GT同RS的RAMP_MERGE_EXIT错误前提有支持；模型NONE未拒绝。

### #613 AccidentTwoWays f125 — 等待优先及归位遗漏

GT：`LANE_CHANGE_RIGHT+STOP`；预测：`RESUME`。[连续帧](probe_output/rgb_review_20260911/case_613.jpg)

f122–125借道面对近黑车，右侧黄车占近端；f126仍近零后才持续加速，f131跨回原侧并继续行驶。GT STOP+RIGHT，模型RESUME跳过下一采样点等待且漏首跨；STOP与稍后右回并不矛盾。

### #615 CrossJunctionDefectTrafficLight f33 — 单帧近停后释放／历史延续错误

GT：`RESUME`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_615.jpg)

f30–33夜雨信号路口明显制动到0.007，但f34立即0.855并持续加速，后续有车横穿。仅一个近零点，GT RESUME，模型STOP；与#8相邻窗展示规则随anchor一帧翻转，视觉预测时机难。

### #625 InvadingTurn f26 — 事件前提时序疑问

GT：`DECELERATE`；预测：`INVALID_ACTION_CONTEXT`。[连续帧](probe_output/rgb_review_20260911/case_625.jpg)

f23–26白天STOP口起步，路口结构明确，右转目标道路的黑SUV尚远，f33后才贴近。模型INVALID不能归因R5错误；ONCOMING_INVASION是否在anchor已成立需独立看当前侵入证据，GT DECEL由下一帧控制脉冲5.459→1.277触发。 源规则补查：event_oncoming_lane_invasion_vehicle_hazard_confirmed，vehicle_hazard=true、目标距离20.810m，停止牌可见。与锥桶单独补全不同，保留标签；不能因早期SUV较远就直接删UE5。

### #637 HighwayCutIn f16 — 同RS正确拒绝对照

GT：`INVALID_ACTION_CONTEXT`；预测：`INVALID_ACTION_CONTEXT`。[连续帧](probe_output/rgb_review_20260911/case_637.jpg)

与#604同一run、同一anchor、同四图，仅改问STATIC_BLOCKAGE。画面为高速平行车流，没有静态物体占ego通道，模型正确INVALID。两题共用一处图像证据，不能当两条独立route。

### #672 HardBreakRoute f75 — 错情境拒绝失败／需局部路口核查

GT：`INVALID_ACTION_CONTEXT`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_672.jpg)

f72–75雾中连续城市走廊跟车静止，建筑端部和路灯位于前方；f76–87先等后跟车起步。模型STOP符合轨迹却未拒绝R5。远端交叉道路可见性不等于当前由路口优先权支配，原图补查后再判标注边界。 原图补查：当前车道、人行道连续，左侧路口在较远前方；现有证据支持当前跟车而非当地路口让行，保留原invalid，不新增重标。

### #694 HazardAtSideLaneTwoWays f83 — 历史制动延续错误

GT：`LANE_CHANGE_RIGHT+RESUME`；预测：`DECELERATE+LANE_CHANGE_RIGHT`。[连续帧](probe_output/rgb_review_20260911/case_694.jpg)

f80–83夜间接近骑行者，输入速度9.171→4.790→5.720→7.327，f84后持续加速，f85向右跨回lane1。GT RESUME+RIGHT，模型DECEL+RIGHT；当前已恢复加速，右跨正确，漏纵向阶段切换。

### #699 HardBreakRoute f216 — 持续停车漏报

GT：`STOP`；预测：`DECELERATE`。[连续帧](probe_output/rgb_review_20260911/case_699.jpg)

f213–216夜间前车尾灯明显，当前6.729；f217降1.869，f221–222(1.25–1.5s)两次近零且之后等待。GT STOP有支持，模型DECEL未预测停车终点。

### #714 MergerIntoSlowTraffic f134 — 连接路延续误报变道

GT：`NONE`；预测：`LANE_CHANGE_RIGHT`。[连续帧](probe_output/rgb_review_20260911/case_714.jpg)

f131–134高速连接车道汇入宽主路，f135–146自车一直lane-4、约9.8m/s，不跨相邻车道边界。GT NONE预测RIGHT；左侧接入线和箭头不是自车右变道。

### #715 ParkedObstacle f151 — 第一次绕出被后续回归替代

GT：`DECELERATE+LANE_CHANGE_LEFT`；预测：`LANE_CHANGE_RIGHT`。[连续帧](probe_output/rgb_review_20260911/case_715.jpg)

f148–151夜雨接近右侧停放车辆，f154自车lane2→1向左，f157降9.542相对12.933足够減速，f161才向右回lane2。GT DECEL+LEFT，模型RIGHT可能读成最终回归，必须取3秒内第一次跨线。

### #716 VehicleTurningRoute f21 — 同一路线错误R5持续段

GT：`RESUME`；预测：`INVALID_ACTION_CONTEXT`。[连续帧](probe_output/rgb_review_20260911/case_716.jpg)

接#724：f18–33连续多车道、右护栏和限速牌，f24进入连接路、f31后出现分流边界；无当前无灯交叉口。GT RESUME有速度支持但R5前提错误，模型INVALID合理；将已逐帧看过的隔离span延长至f33，不删除未看的后段。

### #724 VehicleTurningRoute f5 — 明确上下文错标候选

GT：`NONE`；预测：`INVALID_ACTION_CONTEXT`。[连续帧](probe_output/rgb_review_20260911/case_724.jpg)

f2–17连续多车道道路、右侧护栏和匝道/宽主线布局，无当前局部无灯路口；GT把R5无灯路口判valid但模型INVALID。应回查源RS并对实际看过的帧隔离R5，不把该例当模型误拒；不能仅由RGB强行指定替代R1/R3。 原图补查及源规则：vehicle_turning_junction_space；trigger=1.218m，但turning_local_junction_evidence=false、meta_is_junction=false、无stop/yield/信号支持、xodr_topology_trusted=false。明确隔离已看过f2–17的R5前提，不猜替代RS。

### #744 StaticCutIn f83 — 模型未拒绝错路口／当前等待

GT：`INVALID_ACTION_CONTEXT`；预测：`STOP`。[连续帧](probe_output/rgb_review_20260911/case_744.jpg)

与#455同run稍后anchor：f80–83雾中前方绿车阻挡，自车零速；f84–88继续等，之后起步。GT人工R5不符当前连续道路，模型STOP对轨迹有支持，但漏了前提拒绝。
