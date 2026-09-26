# AGENTS.md

### 2026-09-26 Phase3 v24 连续RGB审计与小幅提示词修订

四包历史对比、60段/1020张不同RGB逐帧观察及修订见
[Phase3 v24审计](AutoMoT/qwen3vl_local/sft_new_loop_phase3/V24_RGB_CALIBRATION_20260926.md)。
新默认data_v24/prompt v24_motion_reference；动作v9阈值、窗口及采样策略保留。
新索引/full map、新run；旧run原源码。技术细节与验证边界见PROJECT_CONTEXT.md同日条目。


### 2026-09-23 分层子池公平性与完整训练池审计补齐

smooth_cap联合分配在动作目标和本轮不同帧数同样最优时，优先补偿连续未选轮数，再按每帧累计曝光排序。
每个归属子池审计容量、本轮/累计呈现和未选轮数；Action三入口checkpoint保存本轮起始历史，轮末才提交，
Phase3同次训练跨轮推进。主要约束排除某子池时不承诺强行覆盖，不改变事件预算、标签或全epoch重复上限。
Phase3 RGB/原始meta审计覆盖带哈希的完整train池＋原val/test；完整管线显式要求训练池，缺失拒绝回退。
RGB按路径、原始signals按物理帧复用；不同上下文/证据变体仍分别核验，仅完全相同行去重。
报告input_coverage按来源记行/帧/case数量，counts_scope标明原索引预检口径；未执行人工RGB复审。
907项相关CPU回归通过，2项缺只读mot_lead_offline_runner.py的合同测试排除，未绕过生产校验。
Action正式入口14帧/每轮12次/cap1的7轮回归，前两轮全覆盖；三入口中途/第二轮/轮末恢复历史与参数一致。
含新增池缺图、原始证据冲突拒绝回归；未全量生产数据/GPU验收。采样hash变化须重建索引/full map并新run。
详见 sft_new_loop_phase3/HIERARCHICAL_SAMPLING_PLAN_20260923.md 及三包运行说明。

### 2026-09-23 分层采样接线、全局容量与跨轮游标修复

Phase3 与 Action 主线/qwen_simple/bev_only 新训默认 smooth_cap：事件内 N^0.5 动作配额、全epoch帧上限8。
Action 默认模式名仍 event_balanced，事件1:…:1:2和背景1/6保留；显式 action-balanced 保留 global_action 对照，
不可与 smooth_cap 混用。cycle_even 保留旧配额对照；token/文字先验、标签和split规则不变。
两包共用压缩最小费用流，共享帧可回退重分配，同事件动作缺额回流；固定主种子和规范输入顺序，环形队列跨周期不洗牌。
Action checkpoint保存当前epoch起始游标，整轮完成才提交下一位置，三入口中途/第二轮/轮末恢复顺序和CPU参数一致。
Phase3新建带SHA256的train_sampling_pool.jsonl，完整train正例不再受构建均衡索引截断；验证/测试仍用原索引。
binary保留INVALID分层配额和人工无重复题，自动负例与正例联合分配；manifest/adapter/epoch审计保存实际策略和游标。
898项相关CPU回归通过；2项合同测试受缺只读mot_lead_offline_runner.py阻断，未绕过校验。
10万帧合成池7轮、每轮95136次回放，累计全覆盖、实际最大重复2，单轮约1.1–1.2秒；非生产数据容量结论。
尚未全量生产重建或真实GPU训练；新索引/full map、新run，旧run原源码。Phase3无optimizer断点恢复入口。
详见 sft_new_loop_phase3/HIERARCHICAL_SAMPLING_PLAN_20260923.md 及三包运行说明。


### 2026-09-23 Action全局动作比例与温和事件加权（覆盖事件内均衡/上限2）

新action-balanced全局六语义动作等量、普通背景UNCOND保留1/6，整数余数按epoch seed轮换。
动作内按支持事件样本量平方根倒数加权、最高2倍；取消事件1:1硬约束，KEEP/UNCOND标签不改。
并发事件按集合只入一个动作池、权重取均值，池内路线轮转优先覆盖不同帧；全epoch单帧上限仍严格。
两模式新训默认重复上限统一8；action自动预算复用同源同cap/world的event预算，容量不足明确报错不缩轮。
显式epoch预算继续生效；旧116256需当前容量预检，不保证v23过滤后event仍可达。动作输入开关独立。
三入口共用action_balance.py，计划/epoch审计记录全局动作配额和真实事件分布，新run，旧run原源码。
402项相关CPU回归通过；3项因缺只读runner未执行，未绕过校验。历史843913帧1/4rank各7轮回放通过：
95136次/轮与同池event一致，不同帧88910–88912，实际最多重复3次；非v23生产重建/真实GPU验收。
Phase3标签、过滤及split规则未改；实现与demo见action_prior/GLOBAL_ACTION_BALANCED_20260923.md及两包run.md。

### 2026-09-23 Phase3 v23 RGB输入、主要动作采样与路线支持

Phase3默认data_v23/prompt v23_grounded_stage，动作v9物理阈值/窗口及主要动作优先级保留。
RGB复核确认f0初始化突变，anchor<4统一排除；Phase3两/四图和Action三入口单/四图共用有效帧。
稀少纵横组合并入主要动作配额，binary证据保留；构建/train/eval一致，不按低频改KEEP/UNCOND。
Phase3按去除Rep/录制时间的物理路线轮转，构建holdout也优先多样性；支持补齐默认32帧/5组。
新增170曝光组，累计1779组train-only；Action用独立split_support计划整组补未曝光路线，三入口共享。
final提交累积尾部并保存后单独验证，输出final_generation.json及cases，不替换best守卫。
旧候选回放Phase3排除3696条、移动54组；Action排除34456帧、移动18组；两者holdout各事件≥32帧/5组。
这些是旧标签容量回放，不是当前生产重建；601项Phase3与328项相关Action/消融CPU回归通过。
更广检查受缺runner/peft/matplotlib限制；未跑真实GPU。新索引/full map/新run，旧run须原源码。
详见sft_new_loop_phase3/V23_RGB_SUPPORT_20260923.md；不宣称模型效果提升或五组足以证明泛化。

### 2026-09-22 Action 默认事件均衡、仅保留两种采样

Action主线与qwen_simple/bev_only新训练默认event_balanced，--action-balanced切换动作均衡；不再支持uniform。
移除--no-event-balanced/--no-action-balanced及EVENT_BALANCED=0；ACTION_BALANCED=0回到event，CLI优先。
默认事件采样同样自动准备full map，动作token/文字先验保持独立开关；十特殊事件各一份、背景两份不变。
event默认重复上限8、action默认2；显式预算/上限仍可覆盖，比较应对齐预算，验证/测试不重采样。
训练循环/容量审计只走两种均衡器；续训恢复保存模式，不注入默认，历史uniform或缺模式run须原源码。
525项相关CPU回归通过；7项因缺只读mot_lead_offline_runner.py未执行，未绕过合同校验；Python/Bash语法检查通过。
说明与简易demo见action_prior/run.md及action_expert_ablation/run.md；未跑真实GPU训练。

### 2026-09-22 Action低重复默认与Phase3支持量诊断（覆盖上限8的新训默认）

三条Action入口共用SamplingArgumentParser，新action-balanced默认全局单帧上限2，event-balanced仍8。
显式CLI/环境值及保存配置优先；旧配置缺字段按历史8恢复，旧run仍须原源码。不自动延长轮数补预算。
保留事件1:…:1:2和真实动作；七token跨事件共享，不因低频变KEEP/UNCOND、不新增条件屏蔽。
Action计划/epoch保存support.cells/events；Phase3构建保存signature_support，共用support_diagnostic。
100帧/10物理路线阈值只标复查线索，不参与过滤/配额/标签；未来UNCOND实验应保留原真值和mask理由。
历史843913帧在新默认下1/4rank各7轮回放：23784次/轮，21251不同帧，最多2次，10次动作配额回流。
UE3 RESUME呈现328→82（仍41独立帧）；65563/95136为显式上限8旧对照，不冒称新默认。
595项Phase3+313项Action/消融CPU检查通过；2项缺只读runner未执行，未绕过校验，未真实GPU训练。
Phase3仍v22目录，采样/构建hash更新须新产物；默认预算不因诊断变化。说明及demo见
Action/ACTION_BALANCED_20260922.md、两包run.md与Phase3/SFT_NEW_LOOP_PHASE3_RUN.md。


### 2026-09-22 Phase3 / Action 稀少动作容量回流（覆盖同日严格等量方案）

Phase3 新默认 data_v22，提示词/轨迹规则仍沿用v21；构建、binary/choice及均衡验证共用
sampling.support_aware_quota：完整自然池循环＋余量容量内均分，去掉整桶随机补额。
Action主线及两消融 --action-balanced 共用函数，并按Phase3 taxonomy过滤域外event×全局动作归属；
RE5右变道4帧仍保留RE2归属/原token/事件事实，不重标KEEP/UNCOND，不按任意低频阈值删合法标签。
事件仍1:…:1:2；动作容量不足回流，联合共享帧冲突可回流，先最少偏离目标再最大化不同帧覆盖。
预算倍数改为lcm(12,world)，不受最小动作格子约束；目标/实际配额、域外排除与overflow写入审计。
旧843913训练帧经新采样器1/4rank各7轮回放：95136次/轮，65563不同帧，最多8次，域外归属排除4。
Phase3完整历史候选十context、7轮容量内/超容量回放通过。合法小事件仍随整体循环，并非取消所有重复。
594项Phase3及305项Action/消融相关CPU检查通过；两项缺只读runner的旧合同测试未执行，未绕过校验。
采样源码加入mapping hash，manifest/config写sampling_policy；新索引/full map新run，自动准备按hash重建。
未全量生产重建或真实GPU训练；详见Phase3/SFT_NEW_LOOP_PHASE3_RUN.md与action_prior/ACTION_BALANCED_20260922.md。


### 2026-09-22 Action 事件内动作两层均衡

三条入口新增 --action-balanced / ACTION_BALANCED=1，与 uniform/event-balanced 互斥。
每事件内对实际支持的逐帧主要动作（含KEEP）等量，十特殊事件各一份、确认普通背景UNCOND两份（用户确认）。
沿用Phase3统一token投影，共享帧全epoch重复上限及唯一帧/路线优先；缺动作报告不合成，缺事件拒绝。
采样与token输入独立；关闭token仍读标签选样，模型不接动作。复用event-balanced-epoch-samples等预算开关。
动作域LCM和world联合预检，训练计划/逐轮审计绑定规则及标签身份；验证不重采样，旧run须原源码。
263项CPU检查通过；两项依赖缺失只读runner源码的旧合同测试未执行，生产校验未绕过。
已有全量有效843913/70058/79518帧，1/4rank各7轮回放通过；RE5右变道仅4帧限制自动预算1440，
每事件120/背景240、不同帧1420、最多重复6次；旧event-balanced为95136，公平比较需同预算。
未跑真实GPU训练；容量表与命令见 action_prior/ACTION_BALANCED_20260922.md，demo见两包run.md。
新增 action_prior/action_balance.py 与 tests/test_action_balance.py 属于已授权代码/测试目录，临时审计JSON不入库。

### 2026-09-22 Action token 弱分离正则

三条入口开启 high-level-action-token 的新训练默认加完整七类（含UNCOND）21对 cosine hinge，
margin=0.5、weight=0.01；action-token-separation-weight=0 可作对照，CLI/同名大写环境变量可覆盖。
仅正则分支归一化，FP32计算，FM与实际concat不变；累积/DDP不额外放大，验证/选优仍用原指标。
配置/版本绑定三入口合同与计划；旧配置缺字段按关闭解释，旧run仍须原源码。
日志记录FM/正则分项、21对cosine及七类范数，审计ZIP收录近期窗口；软约束不保证模型使用token。
226项相关CPU回归通过，含双进程Gloo梯度等价、BF16、三入口真实循环及恢复配置；未跑真实GPU效果实验。
实现见 action_prior/action_token.py、training_core.py，demo见 action_prior/run.md 与 action_expert_ablation/run.md。

### 2026-09-22 Action 多模型配对可视化与训练审计

新增 `action_prior/compare_checkpoints.sh`，两个消融目录同名脚本共用此入口；
编辑 CKPT_DIRS 即可传任意多个带时间的训练目录，按原完整val选best并使用EMA。
按同源event/action标签对有效train/test池各类默认准备最多50个候选，优先物理路线多样性；
所有模型同帧同评估噪声；event/action可逐类及逐split设配额，0跳过，优先物理路线多样性。
ENABLE_EVENT/ENABLE_ACTION默认均true；关闭风格不采样/搜索/输出目录，启用风格仍独立覆盖train/test；两者皆关启动即拒绝。
保存实际RGB投影、道路/车辆框俯视与GT/多模型拼图PNG/PDF、历史输入和简洁JSON；
RGB优先同帧meta实际标定，缺失才回退名义标定；显式JSON可覆盖。三图为横向拼接，各相机独立投影。
已对照LEAD采集顺序及Bench2Drive逆外参，避免直接沿用旧录像器[3,2,1]/欧拉/FOV约定。
真实第三人称图须已有录制及标定；hdmap可绘制语义俯视图，RGB地面投影不做遮挡判断。
已知非三相机/坏meta拒绝回退，RGB折线精确裁剪视野及近平面，图注/报告显示实际标定来源。
分类动作不注入关闭token的模型；仍严格检查原源码/权重/索引/条件合同，新增工具不改变旧指纹。
输出到AutoMoT/test/run_<时间>/（与checkpoints同级）；默认最多4张最空闲GPU，卡不足自动减少。
一卡一worker，模型少于卡时按case分片（4卡2模型为2/2），模型多则排队；GPU_IDS显式pin优先。
独立日志及scheduler记录分配；失败/中断回收本次进程组。86项相关CPU回归通过，尚无真实GPU验收。
预检按阶段每15秒输出耗时/RSS/调用位置到preflight.json/log；选帧仅哈希选中case并提前释放标签池。
GPU worker完成后仍有CPU绘图；render.json/log每15秒报case进度，优先生成GT+所有模型主图。
本轮58项CPU回归通过；分类目录在每个split绘制后发布，总完成以status.json为准。
ERROR_ONLY默认true，仅waypoint任意模型-GT或模型对ADE>1米或FDE>3米触发；route不参与筛选。
CASES_PER_CATEGORY默认50候选预算，ERROR_CASES_PER_CATEGORY默认5命中目标；分批搜索每类达标即停止单独派发。
常驻GPU服务跨批复用当前模型，跨类case去重；已派发批次完成，最终每类不超额，search.json记录预算/缺额/原因。
132项CPU检查通过，未跑真实GPU；规则/日志/早停统计偏置见CHECKPOINT_COMPARISON.md。
SAMPLING_SEED默认auto（时间+系统随机源），每次重新选例并打乱split内执行顺序；所有模型共用同序计划。
EVAL_SEED独立默认2026，原数据划分/训练条件不变；sampling.json/manifest/报告保存种子供复现，76项CPU回归通过。
GPU分片保留公共逻辑顺序的子序列，逐模型case恰好一次；独立日志/缓存/输出，按case身份合并，拒绝缺帧/重复。
汇总逐case计算，不平均分片均值；种子职责/复现/多卡方案见CHECKPOINT_COMPARISON.md。117项CPU检查通过，未验收真实多GPU。
两份bev_only审计：自然加权ADE改善0.82%、均衡waypoint ADE改善17.31%，普通背景/UE4退化；
详见 action_expert_ablation/bev_only/TRAINING_AUDIT_20260922.md 与 action_prior/CHECKPOINT_COMPARISON.md。

### 2026-09-21 全量容量检查与 v21 划分补齐

全量193696候选发现旧固定哈希+开发隔离令val缺4类、test缺5类；仍有未曝光来源。
Phase3默认min-holdout-context-frames=32，仅从未曝光train物理组补容量、同组整体移动，
保护train覆盖及1609开发组train-only；源不足仍失败，split_coverage.json/manifest保存调整。
本轮移动11组，train/val/test=11580/408/384；全局context/动作签名数量未变。
choice/binary各7轮及1/4rank、16/32验证预算回放通过；四类holdout仍仅1物理组，非泛化证明。
Action准备器v2使用独立候选容器，Action自己的split/隔离不变；全量full map=1051417帧。
三入口共享token零计数/独立帧/物理路线/UNCOND原因检查，全UNCOND训练拒绝，每轮采样再检。
Action有效split为843913/70058/79518帧，七类token各split均有支持；uniform/event-balanced各7轮、
world1/4共28个整轮计划通过。均衡95136次/轮，单帧最多8次；帧数不等于独立路线支持。
591项Phase3及136项Action/消融测试通过。产物在/tmp/p3audit/capacity_v21，未覆盖生产目录，
未跑真实模型/GPU；新mapping需新产物新run，旧run用原源码。详见Phase3/CAPACITY_AUDIT_20260921.md。

### 2026-09-21 Action 接入 Phase3 v21 与文案补齐

主线自然先验的 UE1 覆盖减速后的响应/等待/恢复，信号异常改为给定系统故障；
普通/紧凑文案、prefill、摘要、复核和fallback同源，prefill v2/analysis v7绑定合同。
Phase1/2检测合同未改；qwen_simple保留简短导航，bev_only无Qwen，不注入Phase3问答。
三条路径共享v21候选/动作投影及可选token，full map共享开发路线隔离，旧产物须按新hash重建。
150项Action及14项消融入口测试通过；7900候选经文字动作/token投影一致，其中确认起步84帧
为RESUME64/LEFT18/RIGHT2。回放用构造eligible记录，不代表生产full map/门控全量验收。
未全量生产重建、未跑真实Qwen/BEV或GPU；新条件新run，旧run原源码。详见action_prior/run.md。

### 2026-09-21 Phase3 v21 已确认起步与审计分层

新训练默认4rgb+choice、data_v21、prompt v21_confirmed_pullaway，入口seed仍20260920。
v9在近零速等待分支前识别明确brake=False/throttle>0.1且及时持续起步：至增速确认非递减，
确认后保留显著净增速，允许后续调速。缺控制保留null；控制/未来数据不进模型输入。
原速度阈值、有限窗口、首跨规则及STOP>首跨>速度优先级保留，v20精确lane-section隔离继续生效。
提示词明确事件持续阶段、U-E7给定故障前提及记录路线终点；不将灯态查询None/越线代理当物理故障证据。
manifest与epoch快照各报主要动作投影及INVALID分母，路线支持补Town/scenario；容量回流机制不改。
242路线31668帧数值回放，带context变更94帧（29个录制初期），真实构建候选84帧，横向位未变；
沿用此前927张不同RGB复审，不把数值回放当全路线目视。新增155物理组train-only，累计1609。
587项Phase3及63项Action衔接CPU测试通过；7900候选→384行局部开发索引，1408次prompt重放一致。
局部train-only索引未通过也不替代生产三split预检；未全量生产重建或GPU训练，无效果提升结论。
新索引新训，旧run用原源码；共享Action映射必须按新合同重建。详见
AutoMoT/qwen3vl_local/sft_new_loop_phase3/V21_CALIBRATION_20260921.md。

### 2026-09-21 Action 动作 token 与单当前图

三条 action 入口共用默认关闭的 `--high-level-action-token` / `HIGH_LEVEL_ACTION_TOKEN=1`。
五变化动作＋一个 KEEP＋UNCOND，`Embedding(7,1024)` 在 BEV projector 后沿序列维追加1 token，
默认142→143；embedding随FM loss训练并走AdamW。三组共用Phase3 candidate/full map与主要动作优先级，
不经过主线文字动作门控、不自动改变Qwen prompt或特殊RE场景先验。KEEP不细分，仍要求域内完整证据；
普通/过滤/未确认/映射外帧为UNCOND并分原因审计，eligible缺候选/坏哈希/冲突报错。
`--rgb-frame-count 1` / `RGB_FRAME_COUNT=1` 只给当前anchor完整拼接RGB，默认4；主线Phase1/2问答、
可选摘要、最终prefill均单图并适配提示词，旧LoRA输入分布变化明确记录；dataset-priors不跑LoRA。
BEV仍单帧RGB+LiDAR，bev_only图数开关不改变有效条件。CLI优先，resume/eval恢复原开关和图数；
源文件、词表、图数绑定合同，新条件新训，旧run用原源码。oracle token无在线provider，闭环拒绝。
公平对比需共用full map/split/采样/seed/预算；uniform显式给full map也隔离开发路线但不启用均衡。
本机回归635通过、7跳过；另22项因缺只读mot_lead_offline_runner.py（17项）或peft（5项）未通过。
未验证真实Qwen/BEV训练或远端多卡，无效果提升结论。demo见action_prior/run.md、action_expert_ablation/run.md。


### 2026-09-21 Action 七轮快速验证与首轮 warmup（覆盖此前总预算比例）

主线 action_prior 与 action_expert_ablation 的 qwen_simple/bev_only 同步默认7轮。
共享 action_optimization_v5：warmup_ratio=0.05 只按第一轮 optimizer updates 计算，包含梯度累积尾窗口；
warmup占用首周期，周期依次1/2/4轮，第1/3/7轮末到0，第2/4轮开始回原峰值，总计仍7轮。
每轮1000更新时为warmup50＋余弦950/2000/4000；单余弦基线也按首轮warmup。
显式延长轮数则继续8/16…；max_train_steps截断但不压缩余弦曲线，极短预算warmup留一次有效更新。
不新增开关，Muon/AdamW路由、每轮完整验证和training_audit.zip保持；启动日志/计划明确记录实际步数。
新旧schedule合同严格隔离：新方案新开run，旧61轮/总步数warmup run用原源码恢复，不能直接续训换计划。
详见 AutoMoT/qwen3vl_local/action_prior/OPTIMIZATION.md；不宣称真实模型收敛提升。


### 2026-09-21 Phase3 v20 逐帧错例复核与等待/起步语义

新训练默认 `4rgb + choice`、索引 `sft_new_loop_phase3_data_v20`，seed仍20260920，prompt为
`v20_grounded_motion`。57片段/44路线/964帧面板（874张不重复RGB）视觉复核，另320题raw动作/投影一致性回读。
RGB+原meta+本地XODR确认两处lane section连续车道重编号误判RIGHT；精确隔离未来含这两次切换的
横向窗口，不改成NO/KEEP，不泛化删除所有缺section_id的变道。等待/立即起步、当前速度基准与减速后恢复
提示语共享于两种题型；v8阈值、窗口和STOP>首次跨线>速度优先级不变。审计缺失控制字段不补False/0。
本轮四包导出test/val的223物理组新增train-only，累计1454，文件绑定mapping合同；复审后不能继续作盲测。
482项无torch Phase3测试及9项Action准备测试通过；44路线局部候选1762→1756（6条精确隔离），
其余候选标签不变，384行raw核验、1408次题型/RGB重放通过。局部全train索引不代表生产预检通过。
未全量重建或GPU训练，无新模型提升结论；v20需新索引新训，旧run用原源码。共享Action动作描述未改，
上游映射哈希变化仍需新数据产物。详见 AutoMoT/qwen3vl_local/sft_new_loop_phase3/RGB_CODE_AUDIT_20260921.md。

### 2026-09-20 Action 与 Phase3 manifest 格式衔接修复

Action `_candidate_membership` 原写死v3，但Phase3实际产物早已为v5_binary_keep，导致扫描完成后发布失败。
现构建器和Action读取器共用 `build_dataset.FRAME_INDEX_FORMAT`；仅接受当前格式，保留映射哈希、
文件存在性及候选计数检查，错误分别报告format/hash/frame-index具体原因。临时目录发布路径不是此错误根因。
测试不再写死旧v3；新增真实Phase3均衡/manifest写入→Action发布/复用回归（仅替换原始输入扫描）。
小集合真实产物走通1148候选→3657全帧映射→1143动作索引及二次缓存复用；非全量远端/GPU验证。
本轮46项Action数据准备测试及474项Phase3无torch测试通过。旧run用原代码；映射哈希变化需新产物。
本地修改需更新到训练机后重跑原Action入口；不要通过改manifest字段或关闭校验绕过错误。
详见 AutoMoT/qwen3vl_local/action_prior/run.md 的同日manifest衔接记录。

### 2026-09-20 Phase3 自动 INVALID 组合与可选人工事件诊断

按用户要求，不再以 val/test 人工 same-RS 负例至少两条作为 binary 开训门槛。
自动负例枚举几何规则允许的全部错误 RS × asked event，保留十来源及真实R1–R5/十事件覆盖，
按来源、真实RS/事件、错误RS分层抽样；不把未标注事件当不存在，不为全笛卡尔积制造错标。
每个有自动负例的来源在覆盖规划中保留一条，防止人工种子占满索引后运行时无法重采样；
不足预算仍按类型化配额自动增容（五人工种子加一自动种子的回归为30→51），正例预算不变。
人工负例仍无重复输入；覆盖不足标insufficient_support、排除该子组checkpoint守卫，
支持足够时仍检查事件拒绝率。production_ready不因子组缺证据而变为true。
manifest/训练验证报告增加错误RS和真实RS/错误RS/事件分布；raw审计重查自动组合的几何条件。
v19提示词、seed20260920、原动作标签/KEEP不变；映射哈希改变须重建，新训用新代码，旧run用原代码。
不新增人工事件负例，不需要靠补盲审路线才能开训；474项无torch/41项Action数据准备测试通过，
小集合384行原始meta回读通过，正例不变；全量远端构建及GPU尚未验证。
详见 AutoMoT/qwen3vl_local/sft_new_loop_phase3/INVALID_COMBINATIONS_20260920.md。

### 2026-09-20 Action 仅保留所选 high-level 的一句因果描述

移除 --high-level-planning / HIGH_LEVEL_PLANNING；--high-level-action-prior 独立开启。
保留原自然 RS/EVENT，场景末尾只追加一句 Next action，直接复用最新 Phase3 choice_semantics
的 action_description（五种变化动作逐场景条件→动作→目的），不枚举其它动作/目的。
并发事实全部保留，描述只在门控通过且支持所选动作的 context 中按 taxonomy 固定顺序选一个，
审计记录 description_context_id；不把动作 scope 变成场景证据。
干净 dataset-priors + action-prior 自动提供独立 full map 的特殊 RE，策略 dataset_action_special_re_v2；
LoRA/带噪声/动作关闭时不补 RE。UE 门控、标定、采样和 NONE/no_action 空状态不变，不合成 KEEP。
新条件版本 upstream_gated_phase3_causal_sentence_v3 及 Phase3 choice 源码哈希绑定合同/缓存；
prefill/摘要/复核/fallback 同源。旧 run 用原源码，新 prompt 新训；无真实模型效果结论。
详见 AutoMoT/qwen3vl_local/action_prior/run.md 和 DESIGN.md。


### 2026-09-20 Action 精简开关与每轮中途审计（覆盖此前优化细节开关）

三条入口统一 `action_optimization_v4`：本轮新增优化配置只保留 optimizer/lr_scheduler 选择，
周期4段/倍率2、Muon momentum0.95/NS5/RMS倍率1、shared decay、监控100步固定默认，
移除这些细节及cycle/fixed验证CLI和环境开关。每个epoch和共同参考周期完整验证，
单余弦/重启余弦在同预算下用相同完整验证候选点，无需用户选择验证策略。
共用 training_audit.py：首步/定期checkpoint、每轮训练结束、完整验证后及安全终止时，
原子更新当前run的 training_audit.zip（≤30MB）。包含最新提交step、轮次完整性/验证待办、
全rank训练/验证历史、近期loss/LR/更新幅度、实际配置与合同，不含权重/缓存/RGB/完整日志。
审计计数三入口均支持中断恢复；打包中断保留旧ZIP，强杀只能审计上次已发布结果。
无需新开关、不自动上传；恢复训练仍用latest.pt，旧run用原源码。
详见 AutoMoT/qwen3vl_local/action_prior/OPTIMIZATION.md。


### 2026-09-20 Action 吞吐与公平验证计划

三条入口共用 `action_optimization_v3`：默认 adaptive 延续 epoch/周期完整验证；
显式 `full_validation_policy=fixed` 按 `full_validation_steps`（默认1000）及最终步完整验证，
覆盖 epoch/cycle 触发，四组合使用同一候选 step，策略和间隔绑定恢复合同。
完整最终 val 在保存 latest.pt 后写 `validation/final.json`，区分最终与 best 成绩。
吞吐同时记录扣除验证/保存的训练口径和含全部循环开销的整体口径；最终性能汇总包含末尾
验证/保存，`performance/session_*.json` 按进程会话计时，不混入恢复停机时间。
EMA 仍0.999，已有 raw/EMA 配对评估说明见 action_prior/OPTIMIZATION.md；旧run用原源码。
真实条件模型多卡恢复/吞吐尚需远端验收，不能用小矩阵检查替代。


### 2026-09-20 Action 优化配置复审完善

三条 action 入口共用 `action_optimization_v2`：默认 `decay_policy=shared`，AdamW/Muon
对 embedding 的 decay 一致；`legacy` 独立保留旧分组。周期末完整 val 参与 best，与
期末/最终/小验证去重；待办验证及 epoch 审计计数支持中断恢复。默认 rank0 首步、每100步及
周期首尾监控各优化组代表参数真实更新 RMS/相对范数及算法耗时，可设0关闭。
周期峰值保持原值；新配置进入严格恢复合同，旧 run 用原源码。
`action_prior/check_optimization_cuda.py --gpus 1/2` 共用自动选卡/GPU_IDS，独立小矩阵
验证不加载 Qwen/BEV。单卡 CUDA 已检查，双卡 NCCL 与真实收敛/吞吐仍需远端验证。
详见 `AutoMoT/qwen3vl_local/action_prior/OPTIMIZATION.md`。


### 2026-09-20 Action 共享 Muon 与 cosine restart

`action_prior` 和 `action_expert_ablation/{qwen_simple,bev_only}` 新训练统一默认
`optimizer=muon_adamw`、`lr_scheduler=cosine_restarts`：5% warmup 后按1:2:4:8分配剩余
optimizer steps，周期内降到0后回到相同峰值，最终预算后保持0；短预算自动减少周期。
只对 Prefix-KV/轨迹 Transformer 隐藏矩阵及向量场隐藏层用 Muon，输入/最终输出、embedding、
query、norm/bias 留在 AdamW；FP32 参数/优化器状态/EMA，Muon CUDA NS 矩阵乘为 BF16。
基础 LR2e-4、Muon RMS 对齐尺度、momentum0.95/NS5、辅助 AdamW betas=(0.9,0.95)。
共用 `optimization_config.py` / `optimization.py` / `training_core.py`，`adamw + cosine`
保留新 run 基线；算法/周期/预算/路由绑定 checkpoint，续训不能切换，旧 run 用原源码。
配置/对照/日志见 `AutoMoT/qwen3vl_local/action_prior/OPTIMIZATION.md`；不改变数据、先验、
FM 损失或采样，无真实模型提升结论。

### 2026-09-20 Phase3 构建 INVALID 配额自动增容

全量构建可能在候选扫描完成后遇到 INVALID target=30、覆盖方案需要41；此前仅训练验证采样自动增容，
build_dataset入口遗漏。现仅捕获InvalidQuotaError，按required_target重试，保留随机状态；
缺来源/坏签名等数据错误仍抛出。只增加不足的INVALID桶，正例每类预算不变，充足配额抽样不变。
manifest记录requested_target_invalid、target_invalid及invalid_balance.quota，日志打印split和增容原因。
v19名称/动作/prompt不变，源码合同更新需重建；459项CPU测试（含6项无torch构建回归）通过，
局部1148候选/384行除mapping哈希均相同，未在本机跑全量构建或Qwen。
四组对照须显式指定binary/choice；默认choice，省略题型会重复；索引构建一次后可SKIP_BUILD=1复用。
详见 AutoMoT/qwen3vl_local/sft_new_loop_phase3/SFT_NEW_LOOP_PHASE3_RUN.md。

### 2026-09-20 Phase3 v19 风险响应与运动阶段

新训练默认 `4rgb + choice`、索引 `sft_new_loop_phase3_data_v19`，prompt 为
`v19_response_aware_keep`。复看5片段/91帧面板覆盖12条KEEP+车辆零目标速度请求题，
全部来自已有train-only路线，先过滤异常时长，无新增holdout。十类KEEP允许阶段内短暂制动，
不暗示风险已清空；减速/停车的避碰、观察和等待目的保留，两种题型共享，scene_context仍为事实。
新增action_review由构建/审计共用：控制响应、约束对象、原速度/跨线判据节点和记录变化；
缺失不补false/0，invalid证据注明来源上下文，review_only不进入提示词/标签/采样。
边界审计独立response/eval_response桶并回读核验新字段；局部220KEEP中67运动边界、40制动请求、
12车辆零目标速度请求，桶重叠且非错标率。v8标定及STOP>首跨>速度优先级未改，非完整阶段/动机真值。
442项CPU测试、1148候选重算及384行索引/704次题型回放通过；原标签和采样一致，未跑Qwen或全量重建。
需重建v19索引并新训，旧run用原源码；action_prior旧NONE协议不变。
同日开训前修复：action_review 支持明确 None→ID 首次约束出现，未知字段仍中断连续证据；
453项CPU测试通过，1148候选/384行重建补齐257/105条出现记录，动作/采样/prompt不变。
v19名称不变，mapping哈希更新，开训前重建索引；未运行全量生产预检或GPU训练。
详见 AutoMoT/qwen3vl_local/sft_new_loop_phase3/V19_RECORDED_RESPONSE_20260920.md。

### 2026-09-20 Phase3 v18 RGB 因果复核

新训练默认 `4rgb + choice`、索引 `sft_new_loop_phase3_data_v18`，prompt 为
`v18_causal_maneuver`；两种题型显式 KEEP 沿用 v17。复看十类26片段/506帧面板，
其中64帧为补充早期历史；全部为已有 train-only 开发路线，无新增 holdout。
两段 UE4 前提问题按精确 route/frame 隔离35帧，不改成 NO/KEEP/invalid，不推广整条路线。
十类动作说明统一条件→动作→作用；scene_context 仍为事实，目的不是逐帧意图真值。
choice 明确主要动作，准备减速可以先于被选跨线；binary 纵横 YES 可先后发生，仍不输出阶段序列。
KEEP 允许阶段内调速。新增 audit_label_boundaries.py 只报告阈值/窗边界/速度先于跨线，
不修改 v8 标定或 STOP > 首跨 > 速度优先级，不自动过滤或降权。未来数值窗仍不进入模型提示词。
本轮局部候选1183→1148，其余原始动作一致；410项CPU测试及384行索引重放通过，未跑Qwen或全量生产重建。
新默认需要重建索引并新训，旧 run 用原源码；action_prior 旧 NONE/门控接口保持原样。
同日独立复审修复：RE5 RESUME 覆盖等待后起步和未停车的持续增速；边界审计兼容
`prompt_spec.road_structure`，不回退 true_rs，并报告匹配/漏配、对全漏配告警。
417项CPU测试通过，局部1148候选/384行内容未变；v18名称不变，prompt哈希更新，重建刷新manifest。
详见 `AutoMoT/qwen3vl_local/sft_new_loop_phase3/V18_RGB_CAUSAL_REFINEMENT_20260920.md`。

### 2026-09-20 Phase3 v17 判断题显式 KEEP

判断题与选择题共用十类场景的 KEEP 语义和动作因果。binary 纵向域三动作+KEEP+invalid 共五行，
机动域五动作+KEEP+invalid 共七行；有效且证据完整、所有所问变化动作均否时 KEEP:YES。
KEEP 与变化动作互斥；持续等待仍 STOP:YES；invalid 时所有动作包括 KEEP 均 NO。
逐动作纵横证据仍可同时为 YES，choice 沿用主要动作投影。原始五动作标定、采样和 action_prior 协议不变。
模型提示词继续不显示未来数值时间窗。新索引 `sft_new_loop_phase3_data_v17`；训练/评估拒绝缺 KEEP 的旧索引，
KEEP 支持/P/R 纳入 binary 评估和 best 守卫。旧 run 用原源码恢复，新训练重建索引。
本轮复用既有26片段的轨迹及RGB指纹核验，无新增人工RGB审计或模型效果结论。
详见 `AutoMoT/qwen3vl_local/sft_new_loop_phase3/V17_BINARY_KEEP_20260920.md`。


### 2026-09-20 Phase3 v16 直接判断下一动作

按用户要求，UE1–UE7、RE2/3/5 的模型提示词不再显示未来1.5/2/3秒时间窗、连续采样数或速度阈值，
只依据图像序列、当前速度和场景判断接下来动作；各场景动作因果、KEEP题域及当前等待/已完成跨线区分保留。
未来窗口与精确数值判据仅在离线标注代码执行，v8标定、主要动作标签和采样规则不变。历史RGB时间仍说明已观察的输入。
新prompt为 `v16_direct_next_action`，默认索引 `sft_new_loop_phase3_data_v16`；需重建新索引并新训，旧run用原源码。
本轮无新增RGB人工审计或模型效果结论，复用v15开发素材验证。详见 `AutoMoT/qwen3vl_local/sft_new_loop_phase3/V16_DIRECT_NEXT_ACTION_20260920.md`。


### 2026-09-20 Phase3 v15 因果动作与 KEEP（覆盖此前 Phase3 NONE 默认）

Phase3 新训练默认 `4rgb + choice`、索引 `sft_new_loop_phase3_data_v15`，seed仍为20260920。
十类 UE/特殊RE 的 scene_context 只给RS/事件/历史事实，动作选项各给连续的场景因果说明；
减速/停车保留避碰、观察和等待可用间隙的目的，不把目的当作安全空隙或未来跨线的证据。
证据完整的保持阶段输出 KEEP：机动域为车道及速度阶段保持，纵向域只保持速度阶段；允许小幅调速。
缺失/歧义/invalid不能补KEEP；当前持续等待仍为STOP。`choice_semantics.py` 绑定正向标签、题域、
解析与KEEP指标，v8窗口和原STOP>首跨>速度优先级不变；原始布尔动作证据和binary诊断保留。
复用已曝光的26片段/442帧面板覆盖十类，26路线重建1183候选/384行局部冒烟，非全量逐帧审计或模型效果。
仅升级Phase3输出；action_prior仍用旧NONE门控协议，不能直接消费v15 KEEP文本。旧run用原源码恢复。
详见 `AutoMoT/qwen3vl_local/sft_new_loop_phase3/V15_CAUSAL_KEEP_20260920.md`。


### 2026-09-20 Action high-level 提示词精简

当前 planning 协议为 `phase3_inspired_conditional_high_level_v5_explicit_purpose`：缩短 system、道路、
条件性纵横动作及各 UE/特殊 RE 观察目的；有门控通过的主要动作时省去通用动作选项，只保留事实、
动作目的及一个 `Next action`。目的必须保留“减速/等待或调速可以帮助什么”的联系，不能退化为观察清单。
NONE/不可用/拒绝动作仍保留 planning-only；目的不证明空隙或随后变道。
合同、摘要生成和最终 prefill 共用 system 选择；关闭两个开关沿用原提示词。标定/门控/采样/四图导航不变。
R1+UE2 场景文字96→75词，加STOP后126→55词；均排除导航/标签，按英文空白分词而非Qwen token。
用于新训练，旧run用原源码恢复；无真实模型效果结论。详见 action_prior/run.md、DESIGN.md。


### 2026-09-20 Phase3 v14 逐帧审计修订

新训练默认 `4rgb + choice`、索引 `sft_new_loop_phase3_data_v14`、split seed `20260920`。
44片段/748帧重点复核合流、对向侵入、无灯路口及主要动作混淆；目的句区分占道/清空与动作阶段，
NONE不表示看不清，正常雨雾/黑夜保留。四个已审区间精确隔离错误道路前提或输入突变，
不回填NONE/invalid、不改v8速度规则及v1主要动作优先级；346个已曝光物理路线组train-only。
审计器分别核验raw动作和主要动作投影。旧run用原源码恢复；v14效果待新训练验证。
详见 `AutoMoT/qwen3vl_local/sft_new_loop_phase3/EVAL_REVIEW_20260920.md`。

### 2026-09-19 Action 特殊 RE 自动场景先验（覆盖此前独立开关）

新训练在无噪声 `--dataset-priors --high-level-planning` 下自动从独立 full map 提供已确认
RE2/3/5 的场景事实和条件性目的，无需 `--event-balanced-scene-priors`；该独立 CLI/环境开关已移除。
普通 RE 不补成特殊事件，UE 仍来自实际 Phase1/2，动作索引 scope 不补事实。再开动作先验时经同一
RE gate 注入一个主要动作，NONE 保留 planning。LoRA、带噪声或关闭 planning 时不自动补干净 RE。
采样开关独立，十特殊桶各一份/普通背景两份不变。planning-only 自动准备完整映射但不生成动作索引。
内部 `event_balanced_scene_priors` 保存实际启用值，`scene_prior_policy=dataset_planning_special_re_v1`
绑定合同；resume/eval/probe 保留保存条件，不套用新默认值。旧 run 用原源码恢复，闭环仍拒绝离线
场景标注条件。详见 action_prior/run.md、DESIGN.md、scene_policy.py。


### 2026-09-19 Phase3 / Action 统一主要动作与 NONE（覆盖此前多标签输入要求）

用户已要求 Phase3 只提供一个主要 high-level 动作，并允许有效 UE/特殊 RE 的 NONE。
新训练默认 v13 四图 choice：纵向三动作+NONE，机动五动作+NONE；有效全 NO 和组合行进入训练，
仅 invalid 前提剔除。两包共用 `sft_new_loop_phase3/primary_action.py`：STOP（当前等待/近端停车）
优先，其余首次跨线优先于配合速度变化，再取纵向动作，空集 NONE；原始纵横证据仍保留，binary 显式诊断。
NONE 支持/P/R 进入 choice 选优守卫。新索引 `sft_new_loop_phase3_data_v13`，旧 run 用原源码。
Action 使用 `scoped_phase3_primary_action_v4` / `primary_choice_v1`，oracle candidate_actions 保留原始证据，
先经真实 Phase1/2/RE gate，再归并一个主要动作。NONE/no_action 不删事件/planning，只省略具体动作段；
普通 RE 为 not_applicable，缺失/非法为 unavailable。外部预测只接受主要动作/NONE，不伪造 oracle 证据。
旧 schema/来源/缓存合同不混用；默认两个开关及摘要仍关闭，无在线 Phase3 provider。
详见 `AutoMoT/qwen3vl_local/sft_new_loop_phase3/V13_PRIMARY_ACTION_20260919.md` 和 action_prior/run.md。


### 2026-09-19 Action high-level 场景目的

`action_prior --high-level-planning` 按已接受事件/独立 scene context 加入简短动作目的，
UE2 观察相邻车道、接近车辆和通过空间；减速/停车不推出随后变道。配合 `--high-level-action-prior`
仍只注入门控后的统一 selected 动作，索引 scope 不补场景事实，RE gate 与多标签语义不变。
planning 版本升级为 `phase3_inspired_conditional_high_level_v3_compact_purpose` 并绑定条件/缓存；
旧 run 用原源码恢复。目的留在 user，摘要/复核共用，fallback 保持短预算。详见 action_prior/DESIGN.md、run.md。

### 2026-09-19 Phase3 v12 默认四帧单选

`sft_new_loop_phase3` 新训练默认 `4rgb + choice`，索引 `sft_new_loop_phase3_data_v12`；两帧/binary 可显式覆盖。
v11 紧凑 prompt 每题补充一句条件性场景目的：UE2 观察相邻车道、接近车辆和通过空间，
减速不推出变道；其余 context 同样不把动机当动作证据。标定/映射决定/划分不改，无新增人工 RGB 审计。
prompt/source hash 变化需重建新索引并新训；LoRA eval 仍从保存合同恢复，binary 显式可选，
choice 不代替 NONE/invalid/多动作或 action_prior 多标签输入。v12 尚待用户训练验证。
运行及边界见 `AutoMoT/qwen3vl_local/sft_new_loop_phase3/V12_DEFAULT_20260919.md` 与运行手册；覆盖下文历史默认值。

### 2026-09-17 Action 自动 Phase3 high-level 动作输入

在 high-level planning 基础上，`--high-level-action-prior` / `HIGH_LEVEL_ACTION_PRIOR=1`
可追加“接下来具体采取什么动作”；默认关闭。新训练无需 `--high-level-action-index`：
自动复用 Phase3 candidate/full map，缺失时按原构建器生成，投影当前三/五动作域并对齐 action 三 split。
2026-09-18 门控修订：仅 `special_eligible` 的 UE1–7、RE2/3/5 提供动作候选，索引上下文不补入场景事实。
动作统一经过实际 Phase1/2 条件（含噪声/复核）门控：UE 要求 YES/域有效，RS 必须兼容；RE 需要独立显式
scene-priors transition gate。只有最终 selected 进入 prompt，全 NO/不可用/不适用/被挡下动作仅记录审计。
普通背景维持原 prompt 与两份权重；动作开关不改采样方式。输入 v3 明确 binary 多标签语义，拒绝 choice 冒充等价输入；
门控版本与 Phase3 taxonomy 哈希绑定合同。原始动作、有效动作及门控理由分开记录；新训练自动生成 v3 索引。
这是显式授权的离线未来动作真值条件实验，记录 `phase3_oracle` / privileged 属性，不是 Phase3 模型预测；
不加载 Phase3 adapter。该开关是下文“默认不注入逐帧动作”的例外，单独 planning 不读取动作标签。
来源、规则、split、文件和开关绑定合同，逐帧动作绑定缓存；resume/eval/probe 恢复原产物，不重新标注。
索引参数仅保留搬迁/高级输入，自动索引需同目录 manifest；无在线 provider，Bench2Drive/CARLA 拒绝该模式。
接口与 demo 见 `AutoMoT/qwen3vl_local/action_prior/run.md` / `DESIGN.md` / `prepare_action_priors.py`。

### 2026-09-17 Action 可选 high-level planning

`action_prior` 新增默认关闭的 `--high-level-planning` / `HIGH_LEVEL_PLANNING=1`：
保留自然场景事实，以 Phase3 三/五动作语义的简短条件性规划替换旧措辞；单独开启时不接 Phase3 模型或逐帧动作标签。
与摘要开关独立，默认仍直接图文 prefill；模式绑定缓存/合同，resume/eval/probe/闭环沿用保存值。
设计及开启/关闭 demo 见 `AutoMoT/qwen3vl_local/action_prior/DESIGN.md`、`run.md` 和训练 shell 入口。

### 2026-09-15 Action 默认直接图文 KV（覆盖下文历史摘要流程）

`action_prior` 默认 `generate_analysis=False`：四图＋自然 RS/EVENT 先验＋速度/导航直接
base prefill，跳过摘要生成、复核、fallback 和 assistant 回答；dataset-priors 默认零文字生成，
LoRA 来源仍做 Phase1/2 先验问答。`--generate-analysis` / `GENERATE_ANALYSIS=1` 保留原摘要＋完整 KV
路径，`--no-generate-analysis` / `GENERATE_ANALYSIS=0` 显式关闭，CLI 优先。开关绑定缓存与 checkpoint
合同，resume/eval/probe/闭环沿用保存值，不允许同一 decoder 临时切换 KV 模式。旧 run 需原源码恢复。
运行 demo 见 `AutoMoT/qwen3vl_local/action_prior/run.md`、`train.sh`、`run_full_pipeline.sh`；
设计见同目录 `DESIGN.md`。
底层 `train.sh --resume`（含等号/环境变量写法）也先进入共用恢复逻辑，禁止用新训练默认值
覆盖原摘要开关、LR 或索引；显式环境覆盖仍有效，CLI 优先，checkpoint 合同检查保持严格。

> 2026-09-15 action_prior：默认四图＋自然 RS/EVENT 先验＋导航直接 base prefill 得到 KV；
> 仅显式 `--generate-analysis` / `GENERATE_ANALYSIS=1` 才生成摘要并追加其 KV。
> 不向下游注入 YES/NO/UNKNOWN、类别 JSON 或逐帧动作。轨迹 decoder 为联合轨迹条件 Flow Matching，
> 从高斯噪声以 Euler 积分生成 route/waypoint；旧 Linear+cumsum 及逐点独立 FM checkpoint 不兼容。

> 给所有后续 AI / coding agent 的项目入口说明。
> 目标是让新会话在改代码前快速知道：这个工作区在做什么、必须先读什么、哪些文件能动、哪些操作不要做。
>
> 本项目同时维护 [`CLAUDE.md`](CLAUDE.md) 作为 Claude Code 的自动加载入口。
> **AGENTS.md 与 CLAUDE.md 必须保持规则同步**：任何一边新增/修改文件白名单、
> git 规则、工作流偏好、禁止事项、项目入口说明时，必须同步更新另一边。

---

## 1. 先读顺序

开始任何代码分析、改动、提交之前，按这个顺序读：

1. `CLAUDE.md`：Claude Code 自动加载的镜像规则入口；Codex 也要读，确保两边规则一致。
2. `AGENTS.md`：当前通用 agent 入口；Claude 读到 `CLAUDE.md` 后也要读本文件。
3. `PROJECT_CONTEXT.md`：核心技术背景，包含 `lead/` 与 `AutoMoT/` 的数据、推理、BEV、RGB、LiDAR 对齐结论。
4. 当前任务相关源码：通常优先看 `AutoMoT/qwen3vl_local/` 或
   `AutoMoT/keyframe_filter/` 中的白名单实现，必要时再查 `lead/`、
   `AutoMoT/Automot/` 或 `AutoMoT/leaderboard/team_code/` 中的本地只读参考源码。

不要跳过 `PROJECT_CONTEXT.md` 直接从源码重新推断。这个项目里很多结论来自多轮核对，重新凭印象推断很容易犯错。

如果修改了 `AGENTS.md` 中任何规则，也必须同步修改 `CLAUDE.md`；如果发现
`CLAUDE.md` 比本文件更新，也必须把对应规则同步回本文件。不要让 Claude 和 Codex
看到两套不同规则。

---

## 2. 项目一句话

这个工作区在做的是：

把 `lead/` 采集/训练出来的 CARLA 离线数据，整理成本地 Qwen3-VL-Instruct frozen prefill + LeadMoT decoder 能直接消费的离线输入，并逐步分析两边数据分布、坐标系、RGB/LiDAR/BEV/target_point 的差异。

当前主要战场：

- `AutoMoT/qwen3vl_local/`
- `AutoMoT/keyframe_filter/`
- `PROJECT_CONTEXT.md`

---

## 3. 当前技术状态

关键结论以 `PROJECT_CONTEXT.md` 为准，下面只是快速索引：

- `lead/`：数据采集、训练、闭环评测仓库。CARLA 20Hz，每 5 tick 落盘 1 帧，即 4Hz。
- `AutoMoT/`：在线驾驶仓库。慢路径是 Qwen3-VL + KV cache，快路径依赖 BEV encoder + DP heads。
- 当前离线 runner 只走本地 `AutoMoT/qwen3vl_local` 的
  `LocalQwen3VLInstructEngine` 做 frozen Qwen prefill，再接 LeadMoT decoder；
  已移除 AutoMoT legacy `kv_cache_fixed_inference(...)` / `InterleaveInferencer`
  / 原 fast head 接口，不再保留 `--enable-automot-slow` 或 `enable_fast_inference`。
- runner 已切换到 LEAD 风格的 `LeadTransfuserBackbone` / `LeadBEVEncoder`，其输出直接供 LeadMoT 使用；不能再接 AutoMoT 原快推理 decoder。
- LEAD RGB 是三视角拼接 `(W=1152, H=384)`；当前本地 Qwen frozen prefill 直接喂整图，不切片、不 resize、不选前视。
- `vlm_paradigm_a_runner.py` 的 `qwen` backend 必须只读本地 `AutoMoT/checkpoints/Qwen3-VL-4B`（`local_files_only=True`），并用 HF 标准 `past_key_values` 显式 prefill/decode 做文字输出；AutoMoT 现有 `InterleaveInferencer` / `qwen3vl_template_inference` 绑定 AutoMoT 自定义 MoT 架构，不要拿来直接支撑 standalone Qwen 的完整自由文本生成。
- `qwen3vl_instruct_paradigm_a_runner.py` 是 standalone Qwen-only 范式 A runner，只跑本地 `AutoMoT/checkpoints/Qwen3-VL-4B-Instruct`；该目录对应 HuggingFace `repo_id=Qwen/Qwen3-VL-4B-Instruct`，用户远程环境已下载。必须 `local_files_only=True` 且设置 HF/Transformers offline 环境变量，禁止下载；不 import `vlm_paradigm_a_runner.py`，不接 AutoMoT `InterleaveInferencer`。
- `AutoMoT/qwen3vl_local/` 保存 Qwen3-VL-Instruct 本地可魔改代码：`prompt_pipeline.py` 从 `vlm_paradigm_a_runner.py` 的迁移块同步完整提示词/状态机；另含 LEAD RGB 读取、显式 prefill/decode、KV cache summary 与可选 `torch.save`。Qwen3-VL 自定义 KV 增量 decode 必须复用 `mrope_utils.py` 的 `qwen3vl_incremental_forward` 显式复算 M-RoPE `position_ids`，禁止再依赖 `prepare_inputs_for_generation` 组装 decode 输入（PEFT wrapper 会丢 `cache_position`）。`engine.py` 的 `cache_system_prompt` 只允许纯文本 suffix 复用 system-prefix cache；含 `pixel_values` / `image_grid_thw` 的多模态输入必须回退完整 prefill，避免半截图文 M-RoPE cache 错位。`engine.py` 的 `_clone_cache` 必须优先保持 Transformers `Cache` 对象类型（如带 `get_mask_sizes` / `get_seq_length` 的新版 cache），legacy tuple 只能作为旧版兜底；新版 Qwen3-VL forward 会直接调用 Cache 方法，不能把它退化成普通 tuple。
- `0026.json` 是 LEAD meta.pkl 转 JSON 的固定参考样本，只读，绝对不要修改或入库。

---

## 4. 文件修改范围

未经用户明确同意，只允许修改：

- `.vscode/settings.json`（用户授权：排除大数据/产物目录的文件监视；保留源码监视，不隐藏或删除文件）。
- `AGENTS.md`
- `CLAUDE.md`
- `PROJECT_CONTEXT.md`
- `AutoMoT/qwen3vl_local/eval_carla/`（LeadMoT 闭环评测子包，全部子文件白名单内）
  - `__init__.py` / `EVAL_CARLA_PLAN.md` / `EVAL_CARLA_RUN.md`
  - `agent.py`
    （LEAD 风格 CARLA Bench2Drive 实时 agent：3 摄像头 1152×384 + IMU/GPS/Speedometer +
    可选双 LiDAR/4 radar（按 ckpt `decoder_config.use_bev` 决定）；no-BEV 模型不产生未使用 LiDAR/radar 输入。
    ckpt `decoder_config.use_subgoal=True` 当前不支持闭环（CARLA 在线无法获得 SUBGOAL keyframe RGB），
    agent 加载时立即 `raise NotImplementedError` 并留 `TODO(subgoal)` 接口，由后续 SUBGOAL 图像生成/代理输入填补。
    **LEAD 训练分布对齐 (v2)**：RGB 拼接后 JPEG round-trip (JPEG_QUALITY=85)、
    LiDAR 轻量去地面 (z+LSQ, LIDAR_REMOVE_GROUND=1)、radar 4 路 → ego + 近车 duplicate (factor=5, radius=8m) 拼到 LiDAR、
    5 sweep 累积 0.25s 窗对齐 anchor frame。
    推理直接复用 `LeadOfflineMoTRunner`，每 5 tick 调一次模型，中间 tick PID 跟踪 (desired speed 用 wp[1]/wp[3] 即 0.5s/1.0s 两点平均)。
    target_point / next_target_point：训练与在线都走 P1 speed×lookahead 弧长前推：
    `max(speed*lookahead_s, 5m)`，默认 tp=1.0s / ntp=2.0s；final_goal 为 route 真实终点：
    训练取 LEAD 采集保存的 `meta["next_target_points"][-1]` 转 ego，在线 eval_carla 取
    `scenario_picker.py` 对应 route XML 最后一个 waypoint 转 ego；不能再用 `meta["route"][-1]`
    或固定局部 horizon，ego frame (x_forward, y_left)。
    warmup 改 **LEAD 风格 left-pad** 复制 frame 0 立即推理，不再等历史 (与 build_clip line 1808-1815 同款)；
    UKF + route_planner + 基本 PID + SafetyMixin 兜底；Python class/function 已补中文 docstring，shell/HTML/CSS 关键逻辑块有中文注释）
  - `safety.py`
    （SafetyMixin：`stuck_helper` 累计 300 帧低速 → force_move 14 帧 creep / `parking_start`
    前 200 帧位移 < 6m 禁用 force_move / `parking_escape` 1500 帧窗口位移 < 5m 触发 phase1
    强转角 -0.65 + 油门 0.45 / 限速 35 km/h；与 mot_b2d_agent.py 行为完全一致）
  - `video_recorder.py`
    （input/debug/bev_debug/demo/grid **五路** mp4，ffmpeg crf=18/22/28；
    bev_debug 是 LEAD 风格顶视 LiDAR 散点 + pred_route + pred_waypoints + tp/ntp + ego box，
    与 LEAD `video_recorder.py` 的 BEV pseudo-image 等价；demo 在首帧通过
    `CarlaDataProvider.get_world()` 找到 `role_name=hero` 后 spawn cinematic + BEV 临时 carla camera）
  - `visualizer.py`（无依赖 pinhole 投影 + 三视角 overlay；从 LEAD common_utils.project_points_to_image 移植）
  - `scenario_picker.py`
    （LEAD `data/lead/<Scenario>/<Town>_<route_key>.xml` 反向映射；CLI 支持 `--scenario` / `--route-id` /
    `--random N --seed K` 子集筛选与 `--list-scenarios`）
  - `aggregate.py`
    （按 scenario 聚合 leaderboard `eval_<route_id>.json` 写 `scenarios/<Scenario>/summary.json` + `summary_all.json`）
  - `run_eval.sh`
    （一键 launcher：必填 `--leadmot-ckpt`；自动空闲 GPU + 端口槽，支持 `--num-gpus N` / `EVAL_GPU_COUNT=N`
    多卡 worker round-robin 分 route；三种跑法：
    全量（无过滤）/ 按场景 `--scenario <Name>` / 随机 `--random N --seed K`，可叠加；
    `--single-test` / `--route-id` / `--no-input|--no-debug|--no-demo|--no-grid`；跑完自动调 aggregate）
  - `webapp/{__init__.py, app.py, templates/index.html, static/style.css}`
    （Flask：signature 下拉切换 ckpt；Routes tab 按 scenario 分组列 route + 4 路视频切换 + leaderboard
    scores + infractions；Scenarios tab 表格列每个 scenario 平均分）
- `AutoMoT/lead_video_tools/`
  （按用户同意新增到白名单：LEAD 离线 RGB 视频转换工具；只读
  `/datashare/IOL4SGH/data/data/<Scenario>/<run_id>/rgb/*.jpg`，按 4Hz 生成
  `/data/lead_video/<Scenario>/<run_id>/{input,left,front,right}.mp4`（默认 input，`--views`
  可选三视角裁剪），默认在左上角写 frame id，支持异常 route 剔除、断点续跑、
  ffprobe 完整性检查、运行文档和 `--workers` route 级 CPU 并行（`--workers 0`
  自动按 CPU 估计）；`rgb_to_video.py` 普通转换默认剔除异常时长 route，
  `abnormal_duration_filter.py` 按硬规则输出异常采集名单：
  4Hz 下 `frames >= 361`（严格大于 1 分 30 秒 / 90s）且不在白名单内的 route
  全部视为异常并写入 `abnormal_confirmed_over_90s.txt`；
  `BlockedIntersection` 与 `ControlLoss` 是唯一时长白名单不写入名单；
  `Accident`、`park*`、`dynamic*` 不再有 90-100 秒存疑段豁免；
  `abnormal_possible_90s_to_100s.txt` 只为旧接口兼容保留，正常应为空。
  **凡是 AutoMoT/keyframe_filter、AutoMoT/qwen3vl_local 或其它入口使用 LEAD 数据集，
  都必须在构建样本/调研/probe 前先剔除这些异常 route**；
  筛选时打印 discover + route 级进度条，
  两个 txt 名单只保留 `Scenario/run_id`，详情保留在 `abnormal_duration_summary.json`；
  只有显式传 `rgb_to_video.py --abnormal-route-list-dir` 才复用筛选目录只转名单 route）
- `AutoMoT/data/lead/`
  （按用户同意纳入白名单：`lead_data` 对应 route XML 根目录，由
  `AutoMoT/data/data_routes` 提取整理而来。命名规范固定为
  `data/lead/<Scenario>/<Town>_<route_key>.xml`：旧数字 route 为
  `Town03_route_001783.xml`，新版子编号为 `Town12_route_1054_0.xml`，
  命名本身带 Town 的 legacy key 为 `Town06_route_Town06_13.xml`，legacy key
  内部带 route 编号时保留完整 key，如 `Town12_route_Town12_route15.xml`。
  从 `lead_data/<Scenario>/<run_id>` 找 XML 时，`Scenario` 必须取 run 的父目录；
  run_id 先剥末尾 `MM_DD_HH_MM_SS` 时间戳，再只在存在时剥尾部采集后缀
  `_route0`；`Town12_route15` 这类 legacy key 本体里的 `route15` 不能剥，
  也不能要求它带 `_route0`。XML 文件名公式：`route_key` 以 `route_`
  开头时用 `<Town>_<route_key>.xml`，否则用 `<Town>_route_<route_key>.xml`。
  2026-07-03 全量核对结果：`lead_data` 9715 个 run 去重后 9294 个
  `(Scenario,Town,route_key)`，`data/lead` 正好 9294 个 XML，缺失 0、冗余 0；
  命名不规范 0、XML 解析失败 0、内容结构异常 0；XML 内 `<weathis_juncer>`
  拼写已统一修正为 `<weather>`。40 个 XML 的
  `data_routes` 源文件位于不同 scenario 目录（36 个 `noScenarios`、4 个
  `ConstructionObstacleTwoWays`），不是缺失；另有
  `ParkedObstacle/Town12_route_Town12_route15.xml` 覆盖有效并与
  `lead_data/ParkedObstacle/Town12_Rep0_Town12_route15_*` 对应，但未在
  `AutoMoT/data/data_routes` 找到直接源文件。使用时以 `lead_data` / `data/lead`
  的 scenario 目录为准，不能把该项当作 XML 缺失。）
- `AutoMoT/keyframe_filter/`
  （按用户同意新增到白名单：旧版 LEAD 关键帧选择器与新 ROAD/EVENT 语义重标注方案目录。
  `rule_based_keyframe_filter.py` 旧逻辑按 scenario 固定抽 initial / 3 middle / final，主要依赖
  meta 距离字段、speed/accel/brake，并 fallback 到 bbox / RGB motion；只作为突发事件 span
  提议器和验证工具参考，不再视为最终帧级 STATUS/SUBGOAL 真值。`classifier_logic.txt`
  是用户人工调研的道路结构与事件分类草案；`ROAD_EVENT_CLASSIFICATION_PLAN.md`
  是 ROAD/EVENT canonical 总方案，已合并 ROAD_STRUCTURE 调研协议、runtime 门控和错帧回查流程；
  `ROAD_EVENT_CANDIDATE_MAPPING.md` 保留为 Qwen/probe 可解析的候选表；
  `ROAD_EVENT_RGB_AUDIT_ARCHIVE_202607.md` 归并 2026-07 一次性 RGB/RS/EVENT 审计记录，
  旧散落审计 MD 不再恢复；`COLLECTION_OUTPUT_INDEX.md` 说明 `collection_output/`
  大产物、代码读取关系和白名单边界。
  **按用户同意扩展为 clean push 白名单，但默认排除输出产物**：
  `AutoMoT/keyframe_filter/` 下代码、方案文档、规则配置、README、HTML/CSS/JS、
  verification 工具和手写说明允许修改、追踪、commit 和 push。
  `AutoMoT/keyframe_filter/collection_output/` 默认仍是本地数据/审计/证据产物，
  不入库、不 push；**唯一例外**是 Phase1 四问标签的轻量 JSON/JSONL：
  `collection_output/phase1_four_question_audit/phase1_four_question_answer_table.json`、
  `answer_table_partial.json`、`manual_visual_audit_notes.jsonl`、
  `除 no_scenarios_batch 外的 *_batch/phase1_four_question_matrix.json`、
  `full_route_rgb_label_review_20260809/manual_full_sheet_notes_20260809.jsonl`、
  `full_route_rgb_label_review_20260809/manual_table_gap_combo_notes_20260810.jsonl`。
  这些文件是 scenario × RS × EVENT 四问监督标签/人工审计笔记，可精确 add 和 push；
  但 `sheets/*.jpg`、contact sheet、montage、candidate anomalies、route/town/scenario/global
  summary 等 RGB/可再生证据产物仍不入库。Phase1 `collection_output` 目录主入口、
  legacy/superseded 关系和复用流程以
  `AutoMoT/keyframe_filter/PHASE1_COLLECTION_OUTPUT_INDEX.md` 为准；后续类似复核必须先复用
  `full_route_rgb_label_review_20260809/` 与已有 notes，不要重新批量生成重复 RGB 文件夹。
  顶层旧证据产物 `rgb_r4_r5_audit_results/`、`keyframes_all_scenarios.json`、
  `R2_ROUTE_RGB_REVIEW_INDEX_*.csv`、`ROAD_EVENT_INTERRUPTED_OVERLAY_*_IDS_*.csv`、
  `ROAD_EVENT_INTERRUPTED_OVERLAY_IDS_SUMMARY_*.json` 已清理；若后续重生也默认不入库、不 push，
  需要共享时应先整理为方案文档或小型规则配置。ROAD_STRUCTURE / ROAD_EVENT 规则迭代不是手工凭空调参：必须按
  “先把思路写成可执行代码 → 跑小范围样本并生成可视化/逐帧注释 → 查看错帧与证据归因 →
  修正规则/阈值 → 再跑 smoke”的闭环推进。push 前可精确执行 `git add AutoMoT/keyframe_filter/`，
  依赖该目录内 `.gitignore` 排除输出产物；若要提交新产物，必须先确认它不是可再生 evidence）
- `AutoMoT/qwen3vl_local/`（含 `tb_serve.sh` 通用 TensorBoard launcher；`goalgen/` 子包详见 PROJECT_CONTEXT.md §15；`eval_carla/` 子包详见上）
- `AutoMoT/qwen3vl_local/action_prior/`
  （按用户同意新增：Phase1/2 先验 → 禁用所有 LoRA 的 base Qwen 直接编码四图/自然场景先验/
  导航 KV（默认不生成摘要，`--generate-analysis` 才追加摘要）+ 冻结 LEAD BEV → 条件 Flow Matching 轨迹 decoder。自然先验只由确认的 RS/EVENT YES
  查表组成，不泄露 NO/UNKNOWN 或类别 JSON；FM 联合 route(B,10,2)+waypoint(B,8,2)，训练回归直线流的向量场，
  每步以小型 trajectory Transformer 让全部带噪点交互，推理从高斯噪声以默认10步 Euler 采样。自动选择仅接受 best_generation 并核验 prompt/hash/Git/RGB
  与权重指纹；Phase1 做全问+RS 分层复核，Phase2 使用已训练的 EVENT 双域全问+域内续问，
  不伪造未训练的 EVENT hierarchical 接口。invalid 字段留空但保留轨迹监督并统计原因。
  frozen 问答/简述可按合同与实际图像缓存文本，最终 KV 每次由 base 完整 prefill；不接 Phase3。
  全量4Hz索引、物理 route 分割、61 epoch 起始配置、DDP/EMA/TB/频繁验证和独立 eval/probe，
  运行见 action_prior/run.md。代码/脚本/测试/文档可追踪；权重、SQLite、审计和训练输出不入库。）
  **2026-09-06 审查修订**：action_prior 可训练参数/AdamW/EMA 保持 FP32，BF16 仅用于 decoder autocast；仅开启摘要时 base 按先验/当前速度/导航自行组织短分析，不提供标准答案；独立文本模型复核五项判定，通过保留原文，失败才 fallback，模型判定不保证语义正确；导航 CLI 覆盖索引，未接通的多帧 BEV 直接拒绝。跨 rank 共享原子文本缓存，执行指纹按真实入口依赖展开，覆盖共享 Qwen/LeadMoT/只读 runner 与 BEV 工具，排除未接入 Phase3；只读源码只计算哈希，不入库。提供 history/independent/compare 复核审计、上游训练候选池重叠/未知分组及同预算 base/prior 配对消融；审计来源与生成 identity 分离，来源移动/缺失不阻断恢复；续训沿用原审计快照，eval 支持来源重映射并单列内容变化。轨迹分组区分全部确认/仅正常域外/实际未确认；compare 仍按 history 接受且不要求跨模式共识，不把候选池重叠当实际采样命中、不把一致率当准确率。checkpoint 容器为 v4（联合轨迹 conditional Flow Matching）；旧模板、Linear+cumsum 与逐点独立 FM 合同不兼容。
  **2026-09-07 dataset-priors 补充**：`--dataset-priors` 直接读取标定 RS/Phase1/EVENT 标签并默认关闭 analysis review，不加载 Phase1/2 LoRA；默认冷启动每帧仅 1 次最终 base KV prefill；显式 `--generate-analysis` 才增加 base 分析生成。`PRIOR_NOISE` 可注入 RS/EVENT confusion 或 invalid，噪声率、invalid 占比和 seed 均进入先验合同身份；eval/probe 默认沿用 checkpoint 记录。标签搬迁续训可只传 `--prior-labels /新路径`，pipeline 从旧 `config.json` 恢复 dataset 模式并贯穿最终 test/probe；闭环没有 dataset 标签，必须显式切回 LoRA 并披露条件迁移。**2026-09-09 修订**：`RS_HIGHWAY` 是独立 Phase1 事实，R3 不能反推高速；验证按样本身份固定 `eps/t` 和 Euler 初始噪声，并以从纯噪声 Euler 采样得到的加权 route/waypoint ADE 选取 best，FM MSE 仅作诊断。
  **2026-09-09 采样修订**：训练默认仅计算向量场 MSE，不执行 ODE 诊断采样；只在显式 `TRAIN_SAMPLED_METRICS=1` 时采样，且轨迹 Transformer 的 dropout 会临时关闭。闭环由 `--policy-seed` / `ACTION_POLICY_SEED` 派生每条 route 的独立 FM 高斯序列，优先级固定为 CLI > 环境变量 > Traffic Manager `--seed`，记录在 benchmark manifest/model contract。CPU BF16 eval/no_grad 在轨迹 Transformer 内安全回退 FP32，CUDA 路径保持原生 autocast。
- `AutoMoT/qwen3vl_local/action_expert_ablation/`
  （按用户同意新增：action expert 两个无 RS/EVENT 先验消融实验，代码/脚本/测试/文档可追踪，权重、日志、训练输出不入库。`qwen_simple/` 使用四张 LEAD stitched RGB + LeadMoT 原简短导航 prompt 的 base Qwen KV + frozen BEV；不跑 Phase1/2 LoRA、不生成分析摘要、不读取 dataset prior。`bev_only/` 不初始化 Qwen、不传图文 KV，给 LeadMoT Prefix-KV attention 提供 zero-length prefix，保留 frozen LEAD BEV（当前 stitched RGB + LiDAR BEV 融合）、speed、target_point、next_target_point、final_goal 和 query token 训练；它测的是移除 Qwen 图文分支，不是纯 LiDAR/完全无视觉。两者与主线共同调用 `action_prior/training_core.py` 的模型构造、FP32 AdamW/EMA、DDP 分片/梯度累积、FM 训练/验证、日志、checkpoint 保存及恢复校验，复用 `build_dataset.py` 索引；入口仅提供不同 runtime、条件合同与审计接口，后续公共训练行为必须改共享模块，不再复制循环。三组统一记录 `train/samples_seen` 累计训练呈现数和 `train/step_samples` 本次更新样本数，默认完整 step 为 64 case；本次重构改变执行源码指纹，旧 run 需原代码恢复；共享索引首次构建用 `.build.lock` 文件配合 `flock` 加锁，进程退出自动释放，残留锁文件不阻塞后续启动，拿锁后重查 split 完整性，避免两个变体并发写同名 tmp。epoch 尾部不足完整累积窗口时只在同索引/同卡数/同累积/同 seed 下可比。full pipeline 训练前固定本次 `RUN_TAG` 和真实 run dir，`--resume` 指向 `latest/latest.pt` 等软链接时先解析真实 checkpoint，最终 eval 直接读同一 run 的 `best.pt`；CLI `--data-root/--data-dir` 和显式 `MODEL_DIR`/`--model-dir`、`LEAD_BEV_CKPT`/`--lead-bev-ckpt` 贯穿构建、训练和 eval。仅传 `--resume` 时训练入口先从 run `config.json` 恢复原 LR/epoch/梯度累积/索引等参数，launcher 在选 GPU 前从 `training_plan.json` 恢复原 `world_size` 默认值，`train.sh --resume` 不注入脚本默认 LR/epoch/梯度累积/索引；显式 CLI 或环境变量覆盖仍优先生效。resume 会归档 TB 中 checkpoint step 之后的旧 event，并保留 checkpoint step。执行指纹覆盖消融入口、共享 action_prior/LeadMoT/BEV 依赖和关键运行库版本，但不绑定未使用的 Phase1/2 prompt。默认 uniform 的 TensorBoard 只保留核心 `loss`、`route_fm_mse`、`waypoint_fm_mse`、ADE/FDE、LR、grad_norm、吞吐与显存，不记录 RS/EVENT/UNKNOWN 分桶。运行见 `action_expert_ablation/run.md`。）
- `AutoMoT/qwen3vl_local/tb_serve.sh`
  （SFT / GoalGen / LeadMoT / VAE 共用 TensorBoard 启动器；从 `AutoMoT/` 目录下用
  `bash qwen3vl_local/tb_serve.sh <logdir>` 启动）
- `AutoMoT/qwen3vl_local/leadmot/__init__.py`
- `AutoMoT/qwen3vl_local/leadmot/ARCHITECTURE.md`
- `AutoMoT/qwen3vl_local/leadmot/LEADMOT_PLAN.md`
- `AutoMoT/qwen3vl_local/leadmot/LEADMOT_RUN.md`
- `AutoMoT/qwen3vl_local/leadmot/build_dataset.py`
- `AutoMoT/qwen3vl_local/leadmot/train.py`
- `AutoMoT/qwen3vl_local/leadmot/train.sh`
- `AutoMoT/qwen3vl_local/leadmot/eval.py`
- `AutoMoT/qwen3vl_local/leadmot/probe.py`
- `AutoMoT/qwen3vl_local/leadmot/config.py`
- `AutoMoT/qwen3vl_local/leadmot/projectors.py`
- `AutoMoT/qwen3vl_local/leadmot/query_bank.py`
- `AutoMoT/qwen3vl_local/leadmot/heads.py`
- `AutoMoT/qwen3vl_local/leadmot/mot_block.py`
- `AutoMoT/qwen3vl_local/leadmot/decoder.py`
- `AutoMoT/qwen3vl_local/leadmot/subgoal_prompt.py`
  （LEAD-MoT 快推理 decoder 子包及 v1 decoder-only 训练/eval/probe 入口：route(B,10,2) + waypoint(B,8,2)，Linear+cumsum head；gen 路独立 12 层 + frozen Qwen prefix K/V attention（不过 Linear）；hidden=1024=8x128 对齐 Qwen K/V 子空间；gen Q/K 按 `input_len + rope_deltas` 加 1D RoPE，language K/V 已由 Qwen prefill 带 M-RoPE 不重复旋转。训练时冻结 Qwen3-VL-Instruct 与 LeadBEVEncoder，只训练 LeadMoT decoder；GT 包含 route / future_waypoints 两类 ego-frame 累计点，head 内 Linear+cumsum 后直接对绝对点算 loss；`eval.py` 汇总 loss/ADE/FDE，`probe.py` 随机 case-level dump 预测与 GT 对比图。runner 必须用 `LocalQwen3VLInstructEngine` 单独跑 frozen Qwen prefill，只接受同源 HF `past_key_values`；不复用 AutoMoT InterleaveInferencer 的 `gen_context`，也不保留 AutoMoT legacy slow/fast 接口；`--leadmot-ckpt` 显式加载 decoder 权重，先读 checkpoint 的 `decoder_config.use_bev` 再实例化 decoder，并 `strict=True` 加载：`use_bev=True` 必须导入已有 BEV projector 参数，`use_bev=False` 则完全不实例化 / 不 forward BEV，禁止混入随机 BEV；不传 ckpt 仅作为随机初始化链路调试。**`use_subgoal`（离线专用）**：与 `use_bev` 正交的 prefix-only 开关，开启时 build_dataset `--with-subgoal-fields` 反查 `keyframes_all_scenarios.json` 写 scenario/run_id/status/subgoal/subgoal_frame/subgoal_rgb_path/subgoal_lookup_ok 字段；train/eval/probe 通过 `LeadMoTTrainRuntime._run_subgoal_qwen_prefill` 在 prefix 多喂 1 张 SUBGOAL stitched RGB + `[GROUND_TRUTH_STATE]` 文本块（prompt 由 `leadmot/subgoal_prompt.py` 提供，prompt 内仍保留 navigation 文本以维持 tp/ntp/final_goal 对齐）；ckpt `decoder_config.use_subgoal` 与训练 args 必须严格一致，cross-load 由 `_require_subgoal_match` 拒绝；state_dict 形状不受影响（subgoal 不引入新模块），但 prefix KV 分布不兼容；`mot_lead_offline_runner.py` 会按 ckpt 自动走 subgoal prefill 并要求 clip 注入 subgoal 字段，CLI demo 可通过 `--keyframes` 自动反查；eval_carla 在线 agent 暂不支持该开关，加载 use_subgoal=True ckpt 时立即 `raise NotImplementedError`。详见 `leadmot/ARCHITECTURE.md`、`leadmot/LEADMOT_PLAN.md` 与 `PROJECT_CONTEXT.md`）
- LeadMoT frozen Qwen adapter 合同
  （`leadmot/train.py` 支持 `--qwen-adapter-dir`，`train.sh` 支持 `QWEN_ADAPTER_DIR`；
  LoRA merge 到内存中的 frozen Qwen 后仍只训练 decoder。checkpoint `qwen_backbone`
  绑定 base config 与 adapter 实际权重 SHA256，eval/probe/eval_carla 自动恢复并拒绝
  错配；旧 checkpoint 没有该合同时只允许 base。base/LoRA A/B 必须用同 seed 分别训练
  decoder，不能拿同一个 decoder 临时切换 prefix。）
- `AutoMoT/qwen3vl_local/sft/__init__.py`
- `AutoMoT/qwen3vl_local/sft/SFT_PLAN.md`
- `AutoMoT/qwen3vl_local/sft/SFT_RUN.md`
- `AutoMoT/qwen3vl_local/sft/build_dataset.py`
- `AutoMoT/qwen3vl_local/sft/build_teacher.py`
- `AutoMoT/qwen3vl_local/sft/train.py`
- `AutoMoT/qwen3vl_local/sft/train.sh`
- `AutoMoT/qwen3vl_local/sft/eval.py`
- `AutoMoT/qwen3vl_local/sft/probe.py`
- `AutoMoT/qwen3vl_local/sft/check_loss_mask.py`
- `AutoMoT/qwen3vl_local/sft/inspect_teacher_outputs.py`
  （以上是统一 LoRA SFT 子包，已废弃 v1/v2 双轨与 ms-swift。`build_dataset.py` 只产 `dataset_version="pending"` jsonl（assistant 含 `__TEACHER_PENDING__` 占位）；`train.sh` → `train.py` 用 `peft.LoraConfig` + `get_peft_model` 直接把 LoRA 注入 base，torch DDP + 手写 train loop；每个 train batch 内部禁用 adapter，并调用底层 Qwen base model 现场 greedy 生成 ANALYSIS（避开 `PeftModel.generate` 的 Qwen3-VL 生成错位问题），再启用 adapter 跑 student forward + 内置 per-token 加权 loss（ANALYSIS body `SFT_ANALYSIS_WEIGHT`/默认 0.5，学习大致语言推理但不逐字压过状态监督；其余 assistant 段 1.0，prompt 段 0；旧 v2 "结构字面 mask=0" 致命陷阱不再保留）。**不再离线物化 teacher / 不再写 manifest / 不再有 runtime_teacher_data 复用**；`build_teacher.py` 仅作为可选离线 dump 工具。`eval.py` / `probe.py` 默认 `merge_and_unload`，case dump 保存 `expert_analysis.txt` / `language_compare.json` 对比 base-teacher 专家语言与模型 ANALYSIS；`probe.py` 的 token loss 使用训练同款权重汇总；`check_loss_mask.py` 静态验证 train.py 内置 mask；`inspect_teacher_outputs.py` 支持 `--live` 现场重跑 teacher 抽检。详见 `SFT_PLAN.md` / `SFT_RUN.md`）
- `AutoMoT/qwen3vl_local/sft_loop_phase2_augment/`
  （按用户同意新增到白名单：从 `sft_loop_phase2` 复制出的 Phase2 道路结构四问数据增强子包。允许修改、追踪、commit 和 push 代码、prompt、训练/eval/probe/audit 脚本与运行文档；训练/eval/checkpoint 等大产物仍应写入 `AutoMoT/checkpoints/` 或本地输出目录，不随目录白名单入库。）
- `AutoMoT/qwen3vl_local/sft_new_loop_phase1/`
  （按用户同意新增到白名单：融合 `sft_loop_phase1` 最终四问提示词与 `sft_loop_phase2_augment`
  最新 ROAD_STRUCTURE 三类增强问法的一次性 YES/NO 子包；每个样本固定包含 Phase1 四问，
  并把 Phase2 的 `all_random_order` / `subset_random` / `hierarchical_probe` 按训练 4:1:1、
  eval/generation 2:1:1 融入同一轮输出。数据构建复用 Phase2 最新异常 route
  剔除、full-frame RGB review 覆盖检查和默认视觉风险过滤；Phase1 标签来自已审计四问答案表，
  且只有结构化 RGB audit notes/annotations 中的 visual/topology subgroup 能触发覆盖，
  不能从自由文本 `audit_evidence` 推断 route 标签；JSONL 的 RGB 路径默认保存为相对
  `--data-root` 的路径，train/eval 支持 `--data-root` 重映射旧绝对 `lead_data` 路径；
  Phase2 标签来自逐帧 RS 标注。训练/eval
  使用双层采样审计：Phase1 四个 focus 问题各自 YES:NO=1:1，Phase2 四个 focus
  (`RS1/RS2/RS4/RS5`) 也各自 YES:NO=1:1，并在此前提下迁移 `sft_loop_phase2_augment`
  的 all/subset/hierarchical augment balance key、多边际配额、variant report、
  answer-pattern diagnostics、subset 未问行泄漏检查、`RS_HIGHWAY` 与 `GROUP:<id>` 指标；
  all/subset/hierarchical 三类 variant 总量、Phase2 `(focus_bucket, variant)` 配额和
  `all_random_order/RS*:YES|NO` 桶都是硬约束，subset/hierarchical 具体 augment key
  逐桶偏差必须写入 deviation report；四个 Phase1 focus 与四个 Phase2 focus 总量 1:1；
  manifest/train_balance/metrics 必须记录这些比例、每 epoch `balance/epoch_*.json`、窗口
  `augment_counts`、`all_random_order_target_deviation`、`phase2_focus_variant_*` 与重复率审计。
  默认 `FOCUS_BALANCE_COUNT=9216`，对应每轮 147,456 sampled cases，与旧 Phase2 augment
  总 case 数对齐；Phase1 桶先自然抽样，只按 all-random 的全局 RS 缺口从兼容 focus 的未用样本
  换入，不允许为了二级 RS 均匀而循环稀缺子桶；all-random 用容量匹配精确分配 YES/NO。
  默认 `MAX_TRAIN_FRAME_REPEAT=10`，任一 sampled frame 单轮复用超限必须在模型加载前中止。
  训练期 teacher/generation eval 与 checkpoint 默认步频为 2000/2000/20000，
  generation eval 默认 `generation_eval_balance_count=16`，避免小样本漏审 subset/hierarchical。冲突或不确定处以 Phase2 最新 RS 定义为 ROAD_STRUCTURE 权威，同时保留
  Phase1 审计标签作为对应可见事实标签。训练/eval/checkpoint 大产物仍写入 `AutoMoT/checkpoints/`
  或本地输出目录，不随目录白名单入库。）
- `AutoMoT/qwen3vl_local/sft_new_loop_phase2/`
  （按用户同意新增到白名单：融合 Phase1 后使用的单轮 EVENT YES/NO 子包。从
  `sft_loop_phase3` 迁移数据过滤、LoRA DDP、频繁 eval/TensorBoard 和审计框架，但彻底移除
  synthetic Phase2 RS user/assistant 与 KV 前缀；模型输入只有 RGB 和当前 EVENT prompt，内部
  `question_domain` 只用于采样与审计，绝不渲染成已回答 RS。`ROAD_CORRIDOR` 问 UE1/UE3/UE5，
  `LOCAL_JUNCTION` 问 UE6，每题保留 `INVALID_EVENT_CONTEXT`；UE 正类在 train/val/test 保持
  1:1:1:1，RE 默认等于一个 UE 桶，其中默认 25% 为 R3/highway valid all-NO hard negatives，
  invalid 默认约 20% 且只由清晰的跨问题域错配构造，所有 UE 必须为 NO。RGB 路径按
  `--data-root` 相对保存和重映射；训练期 teacher-forced loss eval / generation eval 与 checkpoint 默认步频为
  2000/2000/20000，generation eval 默认 balance count=32；UE3 recall 默认门槛为 0，只统计
  不阻断 checkpoint/流水线；其余 generation guard 仍用于 `best_generation` 诊断选优。
  `run_full_pipeline.sh` 在有效的 `best_generation/`（含 adapter 配置）存在时优先使用它继续完整 eval 和压缩；
  否则使用本轮 `final/`，即使没有 `best_generation` 也不得停在训练结束。旧 v3 冻结 multi-seed/unseen、UE3 rescore、专用 RGB 包、label-alignment、route-balance 与失败 adapter 的 LeadMoT A/B 可执行链已删除，避免与当前 v5 混用；历史成绩文档只作证据。
  历史严格可比基线中，v3 production/audit exact 为 `316/384` / `314/384`；
  2026-08-29 的 v4 实验虽恢复部分 UE3 recall，但 production 降至 `308/384` 且 UE6
  明显退化。v3 保留真正静态路边车/事故/施工和 ego 视差不是 UE3 的边界；这些结果只作
  历史基线，当前训练/评测合同已是 v5，必须重训后才能产生新的可比成绩。
  **2026-09-04 高速 UE3 候选合同**：逐帧 RGB 已确认高速/匝道他车跨分道线进入 ego
  当前通道仍属于原 UE3；`HIGHWAY_CUTIN` 只作 UE3 审计子型，不新增输出类别。数据构建
  只能用 `highway_ue3_rgb_decisions_v1.jsonl` 的显式 RGB-YES span 覆盖，不能从 R3 或
  scenario 名自动造正例；普通并行/稳定跟车/ego 超车仍是高速 all-NO。新 v5 prompt 必须
  重训且不能混用旧 v3/v4 adapter；manifest、generation/独立 eval 与 RGB audit 同时报
  UE3 总体和 HIGHWAY_CUTIN/OTHER_UE3。源 taxonomy 中显式 `U-E3` 的
  DynamicObjectCrossing/ParkingCutIn/StaticCutIn 全部保留；即使它与 R4/R5 interrupted overlay
  共存也必须通过 ROAD_CORRIDOR 问组监督，不能被 RS gate 静默丢弃。
  自由生成完整输出必须做
  顺序/行数/无额外文本严格解析，格式违规时整条
  format/exact 同时失败；adapter 加载前硬校验 production prompt hash、history RGB mode 和
  解析后的 base-model 路径；数据构建把实际扫描 scenario/Town 与 RGB review coverage 做差集；
  train/eval 要求 UE1/UE3/UE5/UE6/RE/INVALID 六桶齐全，`focus_balance_count=0` 只取六桶最小值；
  `invalid_source` 必须贯穿 train/eval，采样继续按 source class 及其联合
  `source+true_rs+wrong-domain` 签名分层轮转，train balance/TB、generation eval 与独立 eval
  必须统计 source class、true RS、错误问题域和联合签名的数量、guard 与 exact；eval 的
  `cases_per_bin=0` 保留全量行但仍强制 INVALID 签名/覆盖校验，错例 audit manifest、summary
  和单例 note 必须直接携带 INVALID 子组。
  代码、prompt、训练/eval/audit 脚本、测试和运行文档允许修改、追踪、commit 和 push；训练/eval/checkpoint 大产物仍写入
  `AutoMoT/checkpoints/` 或本地输出目录。）
- `AutoMoT/qwen3vl_local/sft_new_loop_phase3/`
  （按用户同意新增到白名单：Phase1/Phase2 之后的五动作 high-level LoRA 子包。动作固定为
  `DECELERATE` / `STOP` / `RESUME` / `LANE_CHANGE_LEFT` / `LANE_CHANGE_RIGHT`；未来 meta
  仅用于离线 expert-label（纵向速度窗、同 road 的 OpenDRIVE lane-id 切换），绝不能写入 prompt。
  Phase3 v2 保持五动作；完整 RS 四问全 NO 恢复 R3，未问不作 NO，HIGHWAY 为独立事实；并发异常保留。普通无灯路口不自动 U-E7，原 U7 用既有灯故障答案表适配；新增 R5/R-E5 常规让行，与七异常及 R-E2/R-E3 共十个 context 1:1。R-E2 包含目标变道及绕障恢复，不按 24 帧截断，两条变道 NO 不清除恢复状态；最终目标 y 符号不决定变道侧。invalid 必须覆盖每个 asked context；未来轨迹只用于离线标签，默认异常 route/RGB 风险过滤。逐帧人工审计与机器覆盖分开记录；详见 sft_new_loop_phase3/MAPPING_AUDIT_20260905.md。
  `ACTION_OUTPUT_MODE=binary|choice` 只切换 prompt/target/parser：默认 binary 保持逐题 YES/NO；choice 严格按有效事件 context 给三选一/五选一 high-level 动作词组集合，不添加 `NONE`、invalid 或动作组合。候选词组按 case seed 稳定打乱，模型只输出选中的完整动作词组，不能输出 A/B/C。全 NO、invalid、多个动作 YES 的旧多标签行无法从真值导出唯一动作，必须在 choice 训练/评测显式排除并报告数量，不能编造优先级。choice adapter 绑定独立 prompt hash，必须重训；eval.sh 从 adapter 配置读取并硬校验该模式，代码、训练/eval/audit 脚本和运行文档同步维护。
  代码、prompt、训练/eval/probe/audit 脚本、测试和
  运行文档允许修改、追踪、commit 和 push；训练/eval/checkpoint 与 RGB sheet 等大产物仍写
  `AutoMoT/checkpoints/` 或本地输出目录，不入库。）
- `AutoMoT/qwen3vl_local/sft_loop_phase3/`
  （按用户同意新增到白名单：Phase3 事件级 RS-gated 二值问答子包。复用 Phase2 风格构造已回答且默认正确的 RS context，并在训练/eval 中渲染成上一轮 assistant answer 作为 KV 前缀；`build_phase3_prompt` 默认只表示实际后一轮 user turn，不 inline Phase2，单串审计视图才显式开启 inline；eval case 必须保存实际多轮 messages 或拆开的 phase2 user / phase2 assistant / phase3 user prompt，避免 audit 误读 inline RS context；RS1/RS2 只问 UE1/UE3/UE5，RS4/RS5 只问 UE6，RE 统一为所有 UE=NO；UE2/UE4/UE7 由 Phase1 处理，UE8 默认并入 regular/RE。数据构建需剔除异常时长 route，训练/验证/测试保持 UE1:UE3:UE5:UE6 为 1:1:1:1，并默认加入约 20% wrong-RS invalid/not-applicable 样本；invalid 按 source_class / true_rs / fake_rs 均衡，R3/highway invalid 同时展开到 RS1/RS2/RS4/RS5，要求所有 UE=NO 且 `INVALID_RS_CONTEXT=YES`，eval/TB 必须记录 invalid joint/all-UE-NO 指标；prompt v2 强调弱 RGB 证据时保持 RE/all-NO、普通路口车辆不等于 UE6、事故/静态拥堵不等于 UE3、invalid 只表示 RS gate 明显不适用；训练默认 `REGULAR_FOCUS_MULTIPLIER=2.0` 只放大 RE hard negatives，UE 正类仍为 1:1:1:1，eval/generation 仍用均衡口径；DDP 训练必须按 global step 对齐各 rank，skip/超长样本跑短图文 DDP forward 并用 logits zero loss backward，避免 reducer、barrier 和 eval 分叉；`GRAD_ACCUM>1` 结尾残余梯度必须 flush，`SAVE_STEPS` 落在累积窗口中间时 checkpoint 延迟到下一次 optimizer step 后保存；训练/eval/probe/audit 脚本、prompt、运行文档允许修改、追踪、commit 和 push，训练/eval/checkpoint 等大产物仍写入 `AutoMoT/checkpoints/` 或本地输出目录。）
- `AutoMoT/qwen3vl_local/sft_v2/__init__.py`
- `AutoMoT/qwen3vl_local/sft_v2/SFT_V2_PLAN.md`
- `AutoMoT/qwen3vl_local/sft_v2/SFT_V2_RUN.md`
- `AutoMoT/qwen3vl_local/sft_v2/prompts.py`
- `AutoMoT/qwen3vl_local/sft_v2/build_dataset.py`
- `AutoMoT/qwen3vl_local/sft_v2/train.py`
- `AutoMoT/qwen3vl_local/sft_v2/train.sh`
- `AutoMoT/qwen3vl_local/sft_v2/eval.py`
- `AutoMoT/qwen3vl_local/sft_v2/probe.py`
- `AutoMoT/qwen3vl_local/sft_v2/check_loss_mask.py`
  （按用户同意新增到白名单：SFT v2 两段式串行选择题子包。输入仍为 LEAD stitched RGB + 语言 prompt；stage-1 只列 `SCENE_CHOICES` 并输出 `SCENE`，stage-2 作为同一条对话的后续 user prompt，按预测 scene 的 `EVENT_SEQUENCE` 输出 `STATUS/SUBGOAL`，推理时必须复用 stage-1 已吃图像和场景 prompt 后的 KV cache；默认 `--samples-per-scenario 0` 全量保留合法候选，默认 `--wrong-scene-ratio 0.15` 只增强 train rows；不再有 ANALYSIS / teacher / pending cache；训练 loss 只监督 scene/status/subgoal 值 token，格式 token 为 0 loss；LoRA 默认只注入语言侧 Linear，视觉侧通过 `--lora-vision-scope` / `LORA_VISION_SCOPE` 选择 `off` / `merger` / `last4` / `all` 四档（`--lora-vision` / `LORA_VISION=1` 作为 `all` 的 legacy 别名保留）；开启视觉 LoRA 时默认带"视觉组单独 LR 倍率 `--vision-lr-scale=0.1`（受 `--max-vision-lr-scale=0.25` 上限约束）+ 分组梯度裁剪 `--language-clip-norm=1.0` / `--vision-clip-norm=0.3` + TB 观测 `grad_norm/{language,vision}` / `param_norm/lora_{language,vision}` / `vision_guard_bad_steps` + `STRICT_VISION_SCOPE=1` 命名漂移硬拒绝 + `VISION_GUARD_ENABLED=1` 运行时熔断"保险；熔断时写 `fuse_stop_step_<N>/` 与 `fuse_reason.txt`，并跳过正常 `final/` 保存，防止视觉表征被冲坏且避免误用异常产物；base Qwen checkpoint 始终只读，训练只保存 adapter delta，并写 `sft_v2_adapter_config.json`（含 `lora_vision_scope` 与保险参数）；eval/probe 加载前按 adapter 配置判断普通 LoRA / 视觉 LoRA 并校验权重 key，不一致直接拒绝。自由生成评估中 scene 不在白名单则中断，scene 合法但错误时仍按预测 scene 进入 stage-2 并用串行口径计错，同时输出 `valid_total` / `*_valid_scene` 指标。运行文档见 `SFT_V2_RUN.md`）
- `AutoMoT/qwen3vl_local/sft_v3/__init__.py`
- `AutoMoT/qwen3vl_local/sft_v3/SFT_V3_PLAN.md`
- `AutoMoT/qwen3vl_local/sft_v3/SFT_V3_RUN.md`
- `AutoMoT/qwen3vl_local/sft_v3/prompts.py`
- `AutoMoT/qwen3vl_local/sft_v3/build_dataset.py`
- `AutoMoT/qwen3vl_local/sft_v3/train.py`
- `AutoMoT/qwen3vl_local/sft_v3/train.sh`
- `AutoMoT/qwen3vl_local/sft_v3/eval.py`
- `AutoMoT/qwen3vl_local/sft_v3/probe.py`
- `AutoMoT/qwen3vl_local/sft_v3/check_loss_mask.py`
- `AutoMoT/qwen3vl_local/sft_v3/test_memory_update.py`
- `AutoMoT/qwen3vl_local/sft_v3/test_kv_reuse.py`
- `AutoMoT/qwen3vl_local/sft_v3/test_gt_leak_filter.py`
  （按用户同意新增到白名单：SFT v3 代码已落地，采用 sub-scenario 时间序列训练 + 学生自维护 memory + 三步内循环 teacher/student 蒸馏；Phase A 学生自更新 memory，Phase B 每帧弱纠偏 scene=GT 反向学习“对的别改”；δ 允许 0 且只封顶 10，`EGO_TO_GOAL_XY` 严格来自 meta `next_target_points[-1]` 并在帧末预取下一帧，step3 触发统一走 `should_trigger_step3`；loss 为分析与离散值 token 混合监督，LoRA 视觉接口与 v2 同构并默认关闭；`train.sh` 默认 `ddp`（历史模式名），每卡默认 batch=1；多卡训练采用 work-stealing + local-SGD：不包 DDP、不静态分片、不截断尾部，通过 TCPStore 抢 episode，NCCL collective 前先 TCPStore rendezvous，先广播 rank0 LoRA 初始权重，按本轮 optimizer step 数加权平均 LoRA 参数，并且 `checkpoint-*` / `final/` 都在参数平均后保存；sync 日志/TB 记录 `all_rank_steps`、`round_eps`、`total_eps` 用于审计训练量。详见 `SFT_V3_PLAN.md` / `SFT_V3_RUN.md` 与同目录脚本。）
- `AutoMoT/qwen3vl_local/sft_v4/__init__.py`
- `AutoMoT/qwen3vl_local/sft_v4/SFT_V4_PLAN.md`
- `AutoMoT/qwen3vl_local/sft_v4/SFT_V4_RUN.md`
- `AutoMoT/qwen3vl_local/sft_v4/prompts.py`
- `AutoMoT/qwen3vl_local/sft_v4/build_dataset.py`
- `AutoMoT/qwen3vl_local/sft_v4/train.py`
- `AutoMoT/qwen3vl_local/sft_v4/train.sh`
- `AutoMoT/qwen3vl_local/sft_v4/eval.py`
- `AutoMoT/qwen3vl_local/sft_v4/probe.py`
- `AutoMoT/qwen3vl_local/sft_v4/check_loss_mask.py`
- `AutoMoT/qwen3vl_local/sft_v4/test_memory_update.py`
- `AutoMoT/qwen3vl_local/sft_v4/test_kv_reuse.py`
- `AutoMoT/qwen3vl_local/sft_v4/test_kv_vs_native.py`
- `AutoMoT/qwen3vl_local/sft_v4/test_gt_leak_filter.py`
- `AutoMoT/qwen3vl_local/sft_v4/replay.py`
- `AutoMoT/qwen3vl_local/sft_v4/collect.py`
- `AutoMoT/qwen3vl_local/sft_v4/learn.py`
- `AutoMoT/qwen3vl_local/sft_v4/launch_offpolicy.sh`
- `AutoMoT/qwen3vl_local/sft_v4/inspect_teacher.py`
  （按用户同意新增到白名单：SFT v4 是 sequence-memory OPD 的 off-policy actor-learner 路线；生产入口为 `launch_offpolicy.sh`，默认 4×H20 部署为 GPU0 跑单进程 learner、GPU1/GPU2/GPU3 各 1 个 collector；确认服务器允许单卡多 CUDA 进程后，可手动调 `COLLECTORS_PER_GPU=2/3`。collector 不进 DDP/NCCL，只异步用 LoRA snapshot rollout 并写 `replay/ready/*.jsonl`；learner 不进 DDP/NCCL，单进程随机读取 replay 做 teacher-forced loss/backward，并周期发布 `latest_lora/v_<step>/`。`learn.py` 日志/TB 记录 `replay_ready/replay_pending/replay_failed/wait_events/wait_total` 与 `train/replay/*`，用于判断 collector 和 learner 谁是吞吐瓶颈。`replay.py` 负责 trajectory schema、原子写、文件锁 counter、FIFO 驱逐；`collect.py` 负责 Phase A 50% 正确初始化、Phase B 0.15 噪声扰动、teacher/student generate 和 trajectory 写盘；`learn.py` 负责 replay 采样、无 generate 的 loss/backward、checkpoint/final/snapshot；`train.py` / `train.sh` 仅保留为 on-policy 兼容调试入口，生产训练不要走它。自定义 KV decode 已本地化到 `qwen3vl_local/mrope_utils.py`，`test_kv_vs_native.py` 对比本地增量 KV 与全量无 cache / 原生 generate；旧 bug 污染过的 v4 checkpoint 需作废后重训。三步 student prompt 与 teacher target 共用 `Scene Description` / `Critical Object Description` / `Reasoning on Intent` / `Memory Judgment` 四个公开 heading；step1 student 只读 road-only memory，step2/3 才读完整 memory，teacher 可看 answer 字段但 teacher prompt 不列 label 占位符，标签由脚本追加并清洗成学生视角。scene 训练标签使用 canonical 口径：`EnterActorFlowV2 -> EnterActorFlow`、`MergerIntoSlowTrafficV2 -> MergerIntoSlowTraffic`，原始 CARLA scenario 仅保留在 `scenario/raw_gt_scene` 元数据中。`inspect_teacher.py` 是离线老师抽检脚本：随机采样 episode × 帧 × 5 种 memory 模式（all_keep / rs_change / scene_change_same_rs / event_change / scene_change_cross_rs），先做 prompt contract 自检，再 lazy import torch/model runtime，全程 `disable_adapter` 走 frozen base Qwen，逐 step 记录 teacher-private prompt/raw、student-facing prompt、adapter-enabled student 初始输出、supervised target 与 token 统计，产物为 `teacher_report.md` + `teacher_report.jsonl`，供人工评估老师推理质量并指导 prompt 迭代。）
  （v3/v4 prompt 同步硬约束：`AutoMoT/qwen3vl_local/sft_v4/prompts.py` 是唯一 prompt、Memory、状态机、target span 源；`sft_v3/prompts.py` 只能 re-export v4 并保留兼容别名。v3 是 offline on-policy OPSD：student rollout 更新 memory，`disable_adapter()` privileged teacher logits 对同一批 student step token 做 forward-KL 分布监督；v4 是 off-policy actor-learner/replay 路线。任何 prompt 或状态机改动必须同时验证 v3 和 v4。）
- `AutoMoT/qwen3vl_local/sft_v5/`
  （按用户同意新增到白名单：SFT v5 是 RS / EVENT 两问串行 OPSD 路线。数据来自
  `AutoMoT/keyframe_filter/collection_output/*_result.json`，但训练前跳过
  `noScenarios_result.json`、异常时长 route、数据缺失 skip、缺 XML/RGB/meta/逐帧 annotation
  的 route；`review_required=true` 正常参与训练。每帧 meta 会抽取
  `next_target_points[-1]` 转 ego frame 写成学生可见 `EGO_TO_GOAL_XY`，缺该坐标的
  frame/旧 index 行会被跳过，不能继续显示 UNKNOWN。Q1 使用精简
  `Scene Description / Critical Object Description / Reasoning on Intent` 三段式 CoT 后输出
  `RS`；Q2 保留同样的三段分析后输出 `EVENT`，候选项显式标注
  `[RE | REGULAR]` / `[UE | UNUSUAL]`，直接合并 normal/abnormal 与具体事件判断，
  不再单问当前是否异常；
  prompt 合同固定为 `sft_v5_compact_prompt_v1`：system 只简短保留跨问题共享的
  视觉证据、memory 不可信和禁止泄漏规则，Q1/Q2 user 只放短 memory、短候选、本题
  一句任务和四行格式；代表性二选一预算为 system≤70、Q1≤160、Q2≤175 words，
  版本必须写入 adapter/eval/probe。完整工程标签定义保留在 `labels.py`，真正 prompt
  用短判别描述，禁止在 system/memory/候选/question 重复同一规则。Q1 memory 只渲染自然语言
  `PREVIOUS_RS_HYPOTHESIS + PREVIOUS_RS_HYPOTHESIS_AGE + EGO_TO_GOAL_XY`，不带
  `PREVIOUS_EVENT_HYPOTHESIS`，Q2 才渲染自然语言
  `PREVIOUS_EVENT_HYPOTHESIS + PREVIOUS_EVENT_HYPOTHESIS_AGE`；两个 memory block 都必须显式写
  `MEMORY_RELIABILITY=unverified`，memory 文本不写 A-E 选项字母或 `RE/U-E*` 标签代码。
  Q2 在当前 RS gate 正确后进入，候选优先使用逐帧
  `frame_event_annotation.allowed_events`，缺失时才 fallback 到
  `scenario_event_candidates ∩ EVENT_CANDIDATES_BY_RS[current_rs]`；所有 `R-E*`
  在 prompt 中折为一个 `RE`，原始 `event_code` / `regular_event_codes` 只作审计和 RE
  细分文案。RS 采用慢思考、EVENT 采用快思考：稳定正确 RS 默认以 4 帧为中心，
  每次从 3/4/5 个 4Hz frame 中可复现随机选择下一次 RS_SLOW 间隔，中间帧复用
  RS memory；EVENT_FAST 在每个 RS gate
  正确的帧都重新读当前 RGB 并训练，禁止复用前帧 normal/abnormal/EVENT。RS
  错误、UNKNOWN 或 recovery 时 RS_SLOW 恢复逐帧，当帧 RS 错就跳过 EVENT。
  正式训练默认 `RS_REPAIR_MODE=EVENT_REPAIR_MODE=ground_truth`：RS 连错 4 帧且
  到达 2 帧 review slot、EVENT 连错 3 次且到达每帧 review slot 后才延迟
  写回 GT，绝不在错误下一帧立刻纠正。`unknown` 软擦除只作消融；它在
  纯 memory-copy 压力测试中可长期卡住并饿饿 EVENT，不得作为正式长训默认。
  修复后答对必须与干预前自主恢复分开记录，禁止把 forced repair 帧算作
  `self_recovered_after_streak`。
  训练用 torchrun 多进程同步 on-policy OPSD：慢帧 EVENT_FAST 作为 Q1 assistant
  输出后的第二轮 user turn 复用当帧 Q1 KV cache；快帧没有 Q1 turn，EVENT_FAST
  必须对本帧 RGB fresh prefill，
  再用 privileged teacher logits 对同一批 token 的监督 span 做 forward-KL；每帧
  loss 立刻 backward，只累计 LoRA 梯度，并在 optimizer step 前手动 all-reduce；
  不能包 `DistributedDataParallel(model)` wrapper，
  因为动态 Q2 分支会造成 rank 间 forward 次数不一致并触发 NCCL watchdog。当前不是
  v4 的 collector/learner 异步 replay 分卡架构。collate
  只做本 rank local padding，主训练进程 all-reduce 得到 global `max_T` 后补齐，
  padding frame 不读图、不进 Qwen、不产 loss；多卡默认使用
  `LengthBalancedDistributedSampler` / `SAMPLER_MODE=length_balanced`，在每个 rank
  route 数一致的前提下按 route frame 数均衡分片，减少长 route rank 拖住其它 rank；
  `SAMPLER_MODE=distributed` 可切回 PyTorch 原生 `DistributedSampler` 做对照；
  `train.sh` 支持 `single/ddp/check`，遵循 GPU 自动选址、`GPU_IDS` pin 卡和
  `run_<RUN_TAG>/latest` 防覆盖约定；四卡 `ddp` 默认 H20 max_util 8 路口径：
  `BATCH_PROFILE=max_util`、`PER_DEVICE_BATCH_SIZE=8`、`QWEN_BATCH_SIZE=8`、
  `PARALLEL_KL_MICROBATCH_SIZE=2`、
  `MAX_NEW_TOKENS_Q1=1024`、`MAX_NEW_TOKENS_Q2=1024`、`PROGRESS_FRAMES=20`，
  启动时打印 `[batch]` 配置，第一条 `[batch-start]` 应显示
  `routes=8 / qwen_batch=8`；`BATCH_PROFILE=balanced` 退回 6 路，
  `BATCH_PROFILE=debug` 退回 4 路；
  `single/check` 默认仍保守 `1/1`。rank0 会输出 batch/frame/sync 心跳，默认 `LOGGING_STEPS=1`，
  可用 `PROGRESS_FRAMES` 和 `HEARTBEAT_SECONDS` 调整日志密度；阶段 1 batched Qwen
  通过 `QWEN_BATCH_SIZE` 启用，批量化同一 timestep 多 route 的 Q1/Q2 student rollout，
  需要配合 `PER_DEVICE_BATCH_SIZE>1`；阶段 1 Q1/Q2 student rollout 允许 mixed-length
  padded batch，padded past_key_values 只用于 no-grad 采样 Q1/Q2 文本/token，
  不写回 memory；默认 `PARALLEL_KL=1` / `--parallel-kl`，但 8 路 rollout 与有 autograd
  graph 的 KL 微批解耦：Q1/Q2 teacher/student scoring 默认按 2+2+2+2 微批并逐批 backward；
  Q2 student rollout 与 parallel KL 都必须按精确 `q1_ids` 续接 Q1 KV 后再追加
  Q2 user turn，不允许用
  `q1_ids -> q1_text -> full-dialog tokenizer` 回环替代；KL forward OOM 只允许在尚未
  backward 时二分当前微批，不能降低 token 上限或整块重新 rollout；backward OOM 和
  普通异常必须中止，避免部分梯度后 fallback 重复累计；batched Q1 必须按 `attention_mask` 取最后真实
  token logits、repetition penalty 不得包含 padding token，CUDA OOM 不允许静默 fallback，
  开大前用 `test_batched_qwen_smoke.py --check-parallel-kl` 做 single-vs-batch Q1/Q2
  续接、训练 logits 和 parallel-KL-vs-逐帧-KL 总 loss / case loss / parts 对照；
  parallel KL 的显存峰值主要来自 `KL microbatch x context length` attention activation/logits；必须记录 `parallel_kl/{microbatches_per_chunk,frames_per_microbatch,oom_splits}`，并观察 `train/q1_token_cap_hit_rate` / `train/q2_token_cap_hit_rate`；student rollout 缺少可监督 span 时必须返回 graph-connected zero，不能返回 no-grad 纯 0 破坏 backward；
  只有报告里的 `actual_batched_group_sizes` / `actual_batched_frames` 能证明真实 batched rollout
  被测到，强制验证时必须加 `--require-batched-group`；`qwen/q1_batched_frame_rate`
  是全训练 Q1 frame 的真实 batch 比例，若长期接近 0 应优先检查 `[warn] q1 batch fallback`；
  batched Qwen 相关代码必须保留中文注释解释
  padded rollout、单样本 KV 重建、last-valid logits、padding 排除、EOS active batch 移除、KL OOM 安全二分
  和 TensorBoard 分母口径；纯 batched rollout 不得物化/返回逐样本 final KV，Q2 state
  构造后必须及时释放旧 Q1/Q2 KV；`rope_deltas` 必须兼容 `(batch,1)` / `(1,batch)` 两种方向，
  避免 active batch 缩小时 M-RoPE delta 切片错误；后续改这些逻辑时同步更新注释。每帧 loss
  按全局有效 frame 数归一化，手动 all-reduce 后保持 frame 等权；TensorBoard 必须记录
  `train/loss/{q1_analysis,q1_rs,q2_analysis,q2_event}` 分项，以及
  `memory/{allocated,reserved,max_allocated,max_reserved}_gb`；长期显存风险以
  `allocated` 为主，不能只凭 `nvidia-smi` 或 allocator `reserved` 高水位判断泄漏。
  `probe.py` 公开选帧模式只保留 `random` / `rs_transition` / `ue_transition`；默认
  `random` 用固定 seed 抽取 1 条完整 route ID 并测试全部帧，`--num-routes` 控制
  完整 ID 数；`--num-cases` 只用于 RS/UE 专项预算，RS/UE 默认 context radius 为 8；
  RS 专项必须保留同一次变化的前帧/新 RS 首帧/后帧，UE 专项必须保留同一
  UE span 的全部 UE 帧并按 context radius 补进入前/退出后邻帧，不能被
  `num_cases` 从中间截断；专项找不到变化时不用
  无关帧 fallback 凑数。测试窗口首帧初始化 student/reference；随后 student RS/EVENT
  只由自身 Q1/Q2 输出推进，reference 只作真值比较，禁止回写纠错；逐帧导航坐标可刷新。
  `results.json.memory_recovery_report` 必须统计 RS/UE 变化后 student 首次自行对齐的延迟。
  默认 `--artifact-level review` 按 `scenarios/<scenario>__<route>/frame_<id>/` 保存连续帧；
  每帧只写 `input_rgb_*.jpg`、`input.json`、`output.json`、`memory.json`。output 并列
  student/teacher raw 与 parsed、teacher target、场景 GT 和正确性；memory 并列两问
  student 转换与 comparison-only reference。`compact` 只写顶层 `results.json`；只有
  `--artifact-level full` 才额外保存 system/user/messages 分离视图、student prompt/output、teacher privileged prompt、脚本化 teacher target、
  可选 `q*_teacher_output.txt`、memory_before/after、flags、timeline.json/png 和
  manifest.json，每帧另写完整 `case_record.json`。probe 输出目录启动时必须为空，
  非空直接拒绝且不自动删除；运行中保留 `.probe_in_progress.json`，只有 artifact 校验
  通过并原子提交 `format_version=5` 的 `results.json` 后才删除，`run_integrity` 必须
  记录 route/frame/artifact 完整性；超长/非法 scenario-route 目录名需追加短哈希防碰撞；
  `--with-teacher` 是兼容标志，真正生成 teacher 模型文本必须显式使用
  `--with-teacher-model`；训练前 base Qwen OPSD 能力体检必须不传 `--adapter-dir`、
  不加载任何 LoRA；teacher model output 应和 student 一样从 `Scene Description:`
  开始输出分析与 `RS/EVENT`，不能复读 MEMORY、choices 或 REFERENCE；可视化分为训练前 base Qwen OPSD 能力体检、
  训练中 base/checkpoint/final 固定样本自动对比、训练前 grouped/parallel 等价性、
  训练后 adapter 学生深入可视化、静态 prompt/target 快检五类。
  v5 每个 Python 模块需用中文 docstring 说明用法和入口，所有 class/function
  （含 CLI、嵌套 helper 和魔术方法）都需有中文 docstring；非显然的 padding、KV、
  loss 分母、DDP collective 和显存生命周期逻辑需注释设计原因，不写逐行复述。
  后续改标签协议、prompt、memory、loss、probe 或 DDP 训练逻辑时必须同步维护相邻注释。
  `SFT_V5_RUN.md` 保持为精简的可执行命令手册；设计合同放在 `SFT_V5_PLAN.md`，
  完整 probe 产物和人工检查项放在 `SFT_V5_VISUALIZATION_RECORD.md`，不在三份文档间重复铺开。
  2026-07 本轮详细中文注释覆盖数据过滤/坐标转换、标签与动态候选、memory curriculum、
  local/global padding、batched KV/M-RoPE、精确 `q1_ids` 续接、OPSD span/KL、OOM 安全二分、
  global-frame 梯度归一化与分桶 all-reduce、closed-loop eval、probe 选帧与 artifact 落盘；
  并修正了 forced-repair 恢复统计与 eval/probe oracle 调度泄漏。代码阅读顺序固定参考
  `SFT_V5_PLAN.md` §9.3：`labels.py -> prompts.py -> build_dataset.py -> train.py ->
  metrics.py -> eval.py -> probe.py -> tests`。
  正式训练默认 `UPDATE_MODE=streaming_frames`：每个完整 global timestep 后汇总实际
  有效 frame，累计 `TARGET_GLOBAL_FRAMES_PER_STEP=512` 或达到
  `MAX_TIMESTEPS_PER_STEP=32` 时同步 LoRA 梯度并 optimizer step；不能在同一帧
  Q1/Q2/KL 中间更新。梯度按窗口实际 global frame 数归一化，optimizer step 后保留
  route memory；无本地 frame 的 rank 也必须补零参加 collective，epoch 尾窗口必须
  flush。LoRA 梯度按 device/dtype 合并成约 64 MiB bucket 后再 all-reduce，禁止退回
  数百个小参数逐个 collective。`GRAD_ACCUM` 是流式窗口倍率，`UPDATE_MODE=batch` 只作旧实验兼容；默认
  learning rate 为 `1e-5`。TensorBoard 还必须记录每步 global frame/timestep、更新原因、
  梯度同步 bucket 数、梯度同步和 optimizer 耗时；adapter 元数据必须同时记录原始与
  effective 窗口阈值、LR 和梯度同步策略。正式 launcher 默认 `SAVE_STEPS=40`（按用户
  实测约 80 step/day，即约半天一版）；默认开启 checkpoint probe：step 0 保存
  `probes/base/`，每个 `checkpoint-*` 和 `final/` 保存后用固定 8 个、相同 seed 和相同
  `random` 规则选出的完整 validation route ID 生成对应 probe，并在 `probes/comparison.json`
  聚合各版本 `results.json` 摘要。自动 probe 必须
  复用 rank0 当前训练 bundle，base student/teacher 临时 `disable_adapter()`，LoRA
  checkpoint student 保持 adapter 开启；禁止另起进程或加载第二份 Qwen。其它 rank
  必须在 probe 前后 barrier，probe 完成后恢复 train 模式并清理 CUDA cache；probe
  失败写 `error.txt` 后继续训练。probe 的 256/192 token 上限只用于可视化，不能改变
  训练的 1024/1024。慢帧 teacher EVENT 能力指标只在 teacher 自身 Q1 RS 正确时
  触发，并必须续接 teacher 自己的 Q1 KV/解析 memory；快帧 teacher EVENT 对本帧
  RGB fresh prefill，不能混用 student Q1 prompt；训练 privileged
  输入写 `q2_teacher_training_prompt.txt`，默认 `q2_teacher_prompt.txt` 必须和
  `q2_teacher_output.txt` 实际配对。
  v5 训练 memory 必须按“可疑 hypothesis”而非答案使用：route 首帧 RS/EVENT 分别以
  0.5 概率使用 GT，否则为 UNKNOWN；原本正确的 RS memory 以 0.05/0.07 概率注入
  contradiction/UNKNOWN omission，EVENT 额外注入为 0.20/0.12。UNKNOWN 代表固定 memory
  schema 内的 no-prior，不整块删除 prompt。普通帧 RS/EVENT age 分别累加：对应 label
  真正改变时归零，周期确认同一 label 不归零，padding/skip 不累加；但 EVENT 是
  `EVENT | RS` 条件状态，RS hypothesis 真正改变时旧 EVENT 必须失效为 UNKNOWN/age=0，
  只有新 RS gate 下的 Q2 能重新建立。
  新注入的 wrong/UNKNOWN 因为刚改变 hypothesis，age 必须从 0 开始；若学生继续复制，
  才随后续真实帧自然形成 age>0 的 stale 样本，禁止随机伪造旧 age。
  稳定正确 RS 默认 `rs_slow_interval=4, rs_slow_interval_jitter=1`，即每次在 3/4/5
  帧中可复现抽取下一次复核间隔；快帧不产生
  RS rollout/loss，但必须产生 EVENT rollout/loss。RS 错误只跳过本帧 EVENT，下一帧恢复
  逐帧 RS 分析，直到学生自行纠正或训练期 delayed repair 真正执行。RS 默认
  连续错 4 帧后申请修复并每 2 个有效帧 review，EVENT 默认连续错 3 次后申请修复并
  每帧 review；`rs_repair_interval` 只控制脚本兜底，与 `rs_slow_interval` 独立；
  正式默认在 patience/review 后延迟写回 GT，`unknown` 只是软擦除消融；
  forced repair 后答对与干预前自主恢复必须分开统计；
  EVENT wrong 扰动优先从本帧 `event_option_map` 的其它可见候选中选择，单选题无替代项
  时才回退全局 EVENT 表；EVENT repair/augmentation 只在 RS memory 本帧扰动后仍正确
  时执行。RS 变化导致 EVENT 失效时还必须清空旧 RS 语境的 EVENT streak/pending；若
  同帧 Q2 仍错误，从新语境 streak=1 重新累计，禁止继承旧 pending 立即修复。
  EVENT 的 RE/UE family 完全由当帧 `[RE | REGULAR]` / `[UE | UNUSUAL]` EVENT
  选项推导，不存在独立 ABNORMAL 状态。以上参数必须可由 `train.py` CLI/
  `train.sh` 环境变量覆盖，并写入 adapter metadata。合法 Q1/Q2 最终高权重 span 只监督
  单个选项字符；若存在 `RS:`/`EVENT:` 行但值是 `R4`/`RE` 等非法语义标签，严格 parser
  仍拒绝且不更新 memory，但 loss 必须监督答案起始 token 以直接纠正选项格式；
  训练/TensorBoard 必须记录 wrong-memory copy、wrong/UNKNOWN recovery、
  injected wrong/UNKNOWN、forced repair、Q1/Q2 aligned/omission/contradiction 实际比例、
  RS/EVENT input age、RS 变化导致 EVENT 失效率、随机 RS interval 均值/方差、RS/EVENT input anomaly rate、RS error streak、
  因 RS 错跳过 Q2 的比例，以及由 EVENT 选项折叠出的 UE/RE TP/FP/TN/FN 与 P/R/F1。
  大样本 `eval.py` 与小样本 `probe.py` 必须共用 `metrics.py`：统计 RS/UE 边界、Q1/Q2
  precision/recall/F1、假阳性/假阴性、端到端 EVENT 与 route macro 指标；另外必须用相邻帧
  GT/预测状态分别统计 RS 变化、RE->UE 进入和 UE->RE 退出的 TP/FP/TN/FN/invalid、
  precision/recall/F1 与 false-positive-rate。小样本三种 artifact mode 都把变化报告内嵌在
  `results.json.transition_report`，full 模式再另写 `transition_report.json`；eval
  可用 `--transition-jsonl` 只落盘变化和 FP/FN 的轻量记录。所有指标输出保存中文定义和方向；
  eval 默认流式累计，只有显式 `--output-jsonl` 才落盘全量逐帧输入输出，不能为统计把全量
  prompt/output 常驻内存。
  eval/probe 的 student 默认从 RS/EVENT=UNKNOWN 启动，
  `rs_schedule_policy=deployable` 仅使用 UNKNOWN/非法输出、RS 变化后确认和可复现随机周期复核，
  不能再用 GT mismatch 触发下一帧 recovery；`ground_truth/oracle` 只复现旧报告。
  为实现“RS 真错就跳过 EVENT”，离线 EVENT gate 仍用 GT correctness，输出必须显式
  记录 `event_gate_uses_ground_truth=true` / `fully_deployable_end_to_end=false`，不得误称整条链路可部署。
  数据量审计以 42 个有效场景、7241 route、914466 帧为上限；10% validation 后约
  82.3 万训练帧。恒定 GT、当帧自纠模拟中 Q1 trigger≈30.5%，Q1 relation≈
  59.7/24.2/16.1，Q2 relation≈59.6/23.0/17.4；纯 memory-copy 到 delayed repair
  的压力测试中 Q1 trigger≈55.5%、Q2 gate≈64.0%、Q2 relation≈38.6/43.5/17.9。GT UE=15.55% 与 wrong/UNKNOWN
  memory 异常不能直接相加，最终比例必须看 TensorBoard。
  运行与可视化方法见
  `SFT_V5_RUN.md` / `SFT_V5_PLAN.md` / `SFT_V5_VISUALIZATION_RECORD.md`。）
- `AutoMoT/qwen3vl_local/sft_base/`
  （按用户同意新增到白名单：SFT v5 的直接监督基线。复用 v5 的 collection_output
  数据构建、异常 route 剔除、4 帧 RGB history、`EGO_TO_GOAL_XY`、RS/EVENT 候选池和
  串行 memory 状态；但训练不做 OPSD、不采 student rollout、不跑 privileged teacher、
  不输出 CoT。Q1 target 只有 `RS: <A-E>` 与 `ABNORMAL: <YES|NO>`，Q2 target 只有
  `EVENT: <option>`；Q2 option-letter 扰动使用 v5 seed namespace，同 route/frame/seed
  下 A/B/C 字母映射与 v5 一致；训练为 teacher-forced weighted CE，memory 由 GT answer 更新，
  作为干净直接监督 baseline，不宣称继承 v5 的 on-policy student memory 分布；eval
  仍按学生输出自维护离散 memory，`EGO_TO_GOAL_XY` 每帧刷新为当前帧 ego-frame goal。
  默认 `LORA_VISION_SCOPE=merger`，即默认微调视觉桥接层，并启用视觉 fuse guard；
  eval 加载 adapter 前校验 `sft_base_adapter_config.json` 的 route/dataset/base-model/
  vision-scope，避免误用 v2/v5 adapter；仍可用 `off/last4/all` 做对照。
  运行见 `SFT_BASE_RUN.md` / `SFT_BASE_PLAN.md`。）
- `AutoMoT/qwen3vl_local/sft_baseline/`
  （按用户同意新增到白名单：从 `sft_base` 复制后降维的简化单问 baseline。输入仍为
  LEAD stitched RGB history + `EGO_TO_GOAL_XY` + 轻量 memory；每帧只输出两行
  `ROAD: HIGHWAY|NON_HIGHWAY` 与 `EVENT: RE|UE`，其中 `HIGHWAY` 只对应内部 RS=R3
  的高速/匝道/merge/split/exit/connector/lane-join 结构，`NON_HIGHWAY` 覆盖城市/
  郊区/乡村非结构化 local road、窄双向路、红绿灯路口、无灯/优先权路口等非高速场景；
  `RE` 折叠 regular `R-E*`，`UE` 折叠 unusual `U-E*`。训练是 teacher-forced
  weighted CE，不做 OPSD、不跑 privileged teacher、不输出 CoT；保留训练时
  wrong/UNKNOWN/dropout memory curriculum，但 wrong ROAD 必须跨 HIGHWAY/NON_HIGHWAY
  边界、wrong EVENT 必须跨 RE/UE 边界。默认 `LORA_VISION_SCOPE=off`，只训练语言侧
  LoRA；视觉 LoRA 仅作显式消融。eval 按学生输出 closed-loop 自维护 memory，输出
  `metrics.json` / `frames.jsonl` / `summary.md` / 自包含 `report.html` / 简易 TB；
  `report.html` 不依赖本地数据、外部 JSON/CSS/JS，直接内嵌 ROAD/EVENT 二分类
  confusion matrix 与 change matrix。运行见 `SFT_BASE_RUN.md` / `SFT_BASE_PLAN.md`。）
- `AutoMoT/qwen3vl_local/sft_base_simple/`
  （按用户同意新增到白名单：从 `sft_baseline` 继续简化的 HIGHWAY/NON_HIGHWAY + RE/UE
  单问直接监督基线。显式 transition 采样/API 已撤掉，训练默认先跨 route 聚合
  `FOURBIN_ROUTES_PER_BATCH=16` 条 route，再按当前帧 GT 四格 `HIGHWAY:UE` /
  `HIGHWAY:RE` / `NON_HIGHWAY:UE` / `NON_HIGHWAY:RE` 做 exact balance，默认
  `JOINT_TARGET_BALANCE_COUNT=8`、`UE_FRAME_REPEAT=1`、`UE_EVENT_LOSS_WEIGHT=1.0`、
  repeat mode 为 `none`，避免四格均衡后再向 UE 重复倾斜；eval 默认同样按当前帧 GT
  四格随机均衡，但 joint case 会按 route 顺序闭环 rollout 到最远受评帧，只在抽中帧计
  ROAD/EVENT/JOIN accuracy，change matrix 来自 rollout 相邻帧，`--initial-memory-noise none`
  与 joint eval 组合会被拒绝防止 GT memory 泄漏。transition 帧只作为普通当前帧落入
  对应四格，不再单独抽样或 repeat。训练日志/TB 记录 balance 后四桶实际样本数与
  early-UE prompt memory 的 `RE/UE/UNKNOWN/HIDDEN` 分布。基础 RS/EVENT
  memory wrong/UNKNOWN/dropout 概率沿用 baseline，连续 UE span 前
  `MEMORY_EARLY_UE_FRAMES=4` 帧额外提高 EVENT memory wrong/UNKNOWN/dropout 与重采概率，
  放大后 wrong+UNKNOWN 显式归一化并在启动日志打印 effective 概率，避免模型靠
  `PREVIOUS_EVENT=UE` 续答 UE。当前 `DATASET_VERSION=sft_base_simple_highway_reue_fourbin_v1`，
  adapter route 为
  `sft_base_simple_highway_reue_fourbin_random`，运行见 `SFT_BASE_RUN.md` / `SFT_BASE_PLAN.md`。）
- `AutoMoT/qwen3vl_local/goalgen/GOALGEN_PLAN.md`
- `AutoMoT/qwen3vl_local/goalgen/GOALGEN_RUN.md`
- `AutoMoT/qwen3vl_local/goalgen/GOALGEN_V1.md`
- `AutoMoT/qwen3vl_local/goalgen/GOALGEN_V2.md`
- `AutoMoT/qwen3vl_local/goalgen/build_dataset.py`
- `AutoMoT/qwen3vl_local/goalgen/train.py`
- `AutoMoT/qwen3vl_local/goalgen/train.sh`
- `AutoMoT/qwen3vl_local/goalgen/eval.py`
- `AutoMoT/qwen3vl_local/goalgen/probe.py`
  （以上 9 个是子目标 latent 生成路线 v1/v2 共用数据/训练/eval/probe/文档，详见 PROJECT_CONTEXT.md §7；`GOALGEN_PLAN.md` / `GOALGEN_RUN.md` 只保留索引，版本细节分别写入 `GOALGEN_V1.md` / `GOALGEN_V2.md`；MD 与代码同位于 goalgen 子包内，不要再在 tools/ 下创建重复 MD。GoalGen 训练默认必须导入 `AutoMoT/checkpoints/patch_unpatch_v1/latest/weights/patch_unpatch_best.safetensors`（再兜底无 run_subdir 与最新 `run_*`）并冻结；找不到直接报错，不再随机初始化 patch/unpatch）
- `AutoMoT/vae_standalone/train_patch_unpatch.py`
  （patch/unpatch 端到端图像重建训练脚本：image→VAE.encode→patch→unpatch→VAE.decode→image；VAE 冻结。产物 `patch_unpatch_*.safetensors` 可被 `DiTMoT.load_patch_unpatch` 直接加载，state_dict key 与 DiTMoT 内 `self.patch` / `self.unpatch` 一一对应。`AutoMoT/vae_standalone/` 下其它原始文件仍为只读参考，除非已单独列入白名单）
- `AutoMoT/vae_standalone/vae_reconstruct.py`
  （按用户同意新增到白名单：VAE / patch-unpatch 诊断脚本。支持 VAE-only 与 VAE+patch/unpatch 两种重建链路，对比 VAE 前后 loss、patch 前后 latent loss，按 v1/v2 选择默认模式，支持 TensorBoard 批量 loss 与随机小批量 PNG 对比可视化）

其它文件默认只读，尤其是：

- `lead/` 整个目录
- `AutoMoT/Automot/` 整个目录（只保留本地参考，禁止修改、追踪或 push）
- `AutoMoT/leaderboard/team_code/` 整个目录（只保留本地参考，禁止修改、追踪或 push）
- `AutoMoT/` 中除上述白名单外的源码、配置、权重、数据
- `0026.json`
- 仓库根目录或 `AutoMoT/lead_data` 下的 `keyframes_all_scenarios.json` 数据参考文件
- `AutoMoT/keyframe_filter/collection_output/`
  （本地自动调研输出目录，默认不入库、不 push，保留在本机即可；Phase1 四问标签轻量
  JSON/JSONL 例外：`phase1_four_question_answer_table.json`、`answer_table_partial.json`、
  `manual_visual_audit_notes.jsonl`、`除 no_scenarios_batch 外的 *_batch/phase1_four_question_matrix.json`、
  `full_route_rgb_label_review_20260809/manual_full_sheet_notes_20260809.jsonl`、
  `full_route_rgb_label_review_20260809/manual_table_gap_combo_notes_20260810.jsonl`
  可以精确 add；RGB/contact sheet/summary 等证据产物仍禁止入库）

如果确实需要改白名单外文件，先在对话里说明原因并等待用户确认。

---

## 5. Git 规则

本轮用户授权将 `.vscode/settings.json` 纳入精确 git add 白名单；本机健康日志与系统维护工具不入库。

### GitHub SSH 连接（2026-09-20）

用户已在 GitHub 账号 duguxiaohun 添加名为 ubuntu 的认证密钥。本机
`~/.ssh/id_ed25519.pub` 指纹为 `SHA256:jB6/u0qC/uwkfXtOCeiMhnhQ5wMEtlDrjuxfhYILNTM`，
与用户提供值一致；已通过 `git@github.com:22` 身份认证和仓库 SSH fetch。
公钥指纹不是服务器主机指纹，不能用它替代 GitHub 主机密钥校验。

本机后续 GitHub fetch/push 优先使用这条已验证的 SSH 路径，尤其在 HTTPS 代理
`127.0.0.1:17890` 未运行或直连 TLS 中断时；该代理状态只是此次观察，不当作永久事实。
`origin` 保持 `https://github.com/duguxiaohun/automot_lead.git`，通过单次命令重写传输地址，
不修改全局 Git/代理配置；仍只按用户授权推 main，并执行下文完整历史与白名单检查：

```bash
git -c 'url.ssh://git@github.com/.insteadOf=https://github.com/' fetch --prune origin
git -c 'url.ssh://git@github.com/.insteadOf=https://github.com/' push origin main:main
```

连接诊断：`ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -T git@github.com`。
返回 `Hi duguxiaohun! You've successfully authenticated, but GitHub does not provide shell access.`
表示认证成功，即使退出码为1；仓库读写权限还要分别以实际 fetch/push 结果确认。
本次即使没有可连接的 ssh-agent，默认密钥认证仍成功，不必因此重新生成密钥。
其它机器需使用其自身已授权密钥，不能假定该路径和认证状态相同；不关闭主机校验，
不把私钥、口令或访问令牌写进文档/日志/仓库。

### 清理历史后的 push 约定（2026-09-16）

- 远程 `origin` 为 `https://github.com/duguxiaohun/automot_lead.git`，日常只推
  `main`，显式使用 `git push origin main:main`；执行前确认当前工作分支是 `main`。
  `tune-batched-training-defaults-h20` 已按用户要求删除远程分支，本地同名分支仅保留参考，
  不得自动重新发布；新建、恢复或删除远程分支须在用户授权范围内。
- 日常禁止 `git push --all`、`--mirror`、`--tags`、`--force` 或带 `+` 的强推 refspec；
  不推 `refs/codex/*`、`refs/original/*`、备份引用或旧标签。再次重写历史必须有专项授权、
  外部备份和验证，并使用绑定已核验远程 SHA 的 `--force-with-lease=<ref>:<sha>`。
- push 前先成功 `git fetch --prune origin`，再检查 `git status`、
  `git diff --cached --name-status`、`git log --oneline origin/main..main`、
  `git log --name-status origin/main..main`，以及
  `git rev-list --objects main --not origin/main` 列出的新增历史对象和 blob 大小。
  必须检查全部待推送提交，不能只看当前文件或最终 diff：曾提交后又删除的产物仍会随历史上传。
  fetch 失败时先解决连接问题，不能把旧的 `origin/main` 当作最新远程。
- 正常 push 必须满足 `git merge-base --is-ancestor origin/main main`；
  该检查不能代替历史产物审核。发现分叉、旧历史合并或白名单外新增对象时先处理，
  不用强推或 `--allow-unrelated-histories` 绕过。
- 继续精确 add 原白名单；目录白名单不包含其数据、权重、缓存、视频、RGB 证据和压缩包。
  `collection_output/` 只保留原先明确允许的 Phase1 标签 JSON/JSONL；
  已清理的旧审计产物、索引和 `AutoMoT-main.zip` / `lead.zip` / `lead_xml.zip` 不得重新入库。
  `.gitignore` 不会清除已追踪文件或历史对象，文档白名单也不是 Git 自动拦截器。
- 清理后的 main 基点为 `3c7e627b71bda5d549f04bbd6f870d45298b37ca`。
  旧 clone、旧提交 SHA、bundle 和历史备份只用于回查；不能把清理前历史 merge/push 回远程。
  旧机器先保护本地修改、数据和权重，再迁移必要代码差异到新历史；不得借机 `git clean`。
  旧 checkpoint 需要原源码时使用隔离备份或提交映射，不能篡改 checkpoint 合同。备份索引见
  `PROJECT_CONTEXT.md`「2026-09-16 Git 历史清理」。
- 本节不构成后续 push 的永久授权，仍遵守本文件的用户授权规则。

### 5.1 拉取远程更新

当用户说“拉取远程最新代码覆盖本地”“更新到远程最新代码”或类似表达时，含义是：

- 只更新 / 覆盖 git 已跟踪代码文件；优先用 `git fetch` 后按远程分支处理 tracked 文件。
- 只有与远程 tracked 文件发生冲突或本地 tracked 改动挡住更新时，才覆盖这些 tracked 文件。
- 不要删除未跟踪文件、未跟踪目录、本地数据、权重、缓存、软链接、外部同步目录或用户放在工作区里的参考资料。
- 禁止把这类请求自动扩展成 `git clean -fd`、`git clean -ffd`、`rm -rf` 或任何清理未跟踪文件的操作。
- 如果确实需要清理未跟踪内容，必须先单独列出将删除的路径，并得到用户明确确认。

简言之：用户要的是“更新代码”，不是“清空工作区”。除非用户明确说要删除其它本地内容，否则不要动与远程 tracked 代码无关的东西。

不要使用：

- `git add .`
- `git add -A`
- `git add *`
- `git add lead/`
- `git add AutoMoT/`
- `git add AutoMoT/Automot/` 或其中任何文件
- `git add AutoMoT/leaderboard/team_code/` 或其中任何文件
- `git add 0026.json`
- `git add keyframes_all_scenarios.json`
- `git add AutoMoT/lead_data/keyframes_all_scenarios.json`
- `git add AutoMoT/keyframe_filter/collection_output`
  （禁止整目录 add；只允许精确 add Phase1 四问标签白名单 JSON/JSONL）

只精确 add 白名单文件。例如：

```bash
git add AGENTS.md CLAUDE.md PROJECT_CONTEXT.md
git add AutoMoT/qwen3vl_local/eval_carla/__init__.py AutoMoT/qwen3vl_local/eval_carla/EVAL_CARLA_PLAN.md AutoMoT/qwen3vl_local/eval_carla/EVAL_CARLA_RUN.md AutoMoT/qwen3vl_local/eval_carla/agent.py AutoMoT/qwen3vl_local/eval_carla/safety.py AutoMoT/qwen3vl_local/eval_carla/video_recorder.py AutoMoT/qwen3vl_local/eval_carla/visualizer.py AutoMoT/qwen3vl_local/eval_carla/scenario_picker.py AutoMoT/qwen3vl_local/eval_carla/aggregate.py AutoMoT/qwen3vl_local/eval_carla/run_eval.sh AutoMoT/qwen3vl_local/eval_carla/webapp/__init__.py AutoMoT/qwen3vl_local/eval_carla/webapp/app.py AutoMoT/qwen3vl_local/eval_carla/webapp/templates/index.html AutoMoT/qwen3vl_local/eval_carla/webapp/static/style.css
git add AutoMoT/lead_video_tools/__init__.py AutoMoT/lead_video_tools/abnormal_duration_filter.py AutoMoT/lead_video_tools/rgb_to_video.py AutoMoT/lead_video_tools/LEAD_VIDEO_RUN.md
git add AutoMoT/data/lead/  # 目录白名单：route XML 与同目录轻量审计记录
git add AutoMoT/keyframe_filter/  # 依赖 AutoMoT/keyframe_filter/.gitignore，只会带入代码/文档和 Phase1 四问标签轻量 JSON/JSONL，不带 RGB/contact sheet
git add AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/phase1_four_question_answer_table.json AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/answer_table_partial.json AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/manual_visual_audit_notes.jsonl
git add AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/critical_batch/phase1_four_question_matrix.json AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/highway_flow_batch/phase1_four_question_matrix.json AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/motion_parking_batch/phase1_four_question_matrix.json AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/obstacle_single_batch/phase1_four_question_matrix.json AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/obstacle_twoways_batch/phase1_four_question_matrix.json AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/remaining_batch/phase1_four_question_matrix.json AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/signal_control_batch/phase1_four_question_matrix.json AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/vehicle_turning_batch/phase1_four_question_matrix.json
git add AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809/manual_full_sheet_notes_20260809.jsonl AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809/manual_table_gap_combo_notes_20260810.jsonl
git add AutoMoT/qwen3vl_local/__init__.py AutoMoT/qwen3vl_local/cache_utils.py AutoMoT/qwen3vl_local/engine.py AutoMoT/qwen3vl_local/image_io.py AutoMoT/qwen3vl_local/mrope_utils.py AutoMoT/qwen3vl_local/prompt_pipeline.py AutoMoT/qwen3vl_local/run_log.py AutoMoT/qwen3vl_local/tb_serve.sh
git add AutoMoT/qwen3vl_local/goalgen/__init__.py AutoMoT/qwen3vl_local/goalgen/vae.py AutoMoT/qwen3vl_local/goalgen/prompt.py AutoMoT/qwen3vl_local/goalgen/qwen_kv.py AutoMoT/qwen3vl_local/goalgen/keyframes.py AutoMoT/qwen3vl_local/goalgen/dit.py AutoMoT/qwen3vl_local/goalgen/flow.py
git add AutoMoT/qwen3vl_local/sft/__init__.py AutoMoT/qwen3vl_local/sft/SFT_PLAN.md AutoMoT/qwen3vl_local/sft/SFT_RUN.md AutoMoT/qwen3vl_local/sft/build_dataset.py AutoMoT/qwen3vl_local/sft/build_teacher.py AutoMoT/qwen3vl_local/sft/train.py AutoMoT/qwen3vl_local/sft/train.sh AutoMoT/qwen3vl_local/sft/eval.py AutoMoT/qwen3vl_local/sft/probe.py AutoMoT/qwen3vl_local/sft/check_loss_mask.py AutoMoT/qwen3vl_local/sft/inspect_teacher_outputs.py
git add AutoMoT/qwen3vl_local/sft_v2/__init__.py AutoMoT/qwen3vl_local/sft_v2/SFT_V2_PLAN.md AutoMoT/qwen3vl_local/sft_v2/SFT_V2_RUN.md AutoMoT/qwen3vl_local/sft_v2/prompts.py AutoMoT/qwen3vl_local/sft_v2/build_dataset.py AutoMoT/qwen3vl_local/sft_v2/train.py AutoMoT/qwen3vl_local/sft_v2/train.sh AutoMoT/qwen3vl_local/sft_v2/eval.py AutoMoT/qwen3vl_local/sft_v2/probe.py AutoMoT/qwen3vl_local/sft_v2/check_loss_mask.py
git add AutoMoT/qwen3vl_local/sft_v3/__init__.py AutoMoT/qwen3vl_local/sft_v3/SFT_V3_PLAN.md AutoMoT/qwen3vl_local/sft_v3/SFT_V3_RUN.md AutoMoT/qwen3vl_local/sft_v3/prompts.py AutoMoT/qwen3vl_local/sft_v3/build_dataset.py AutoMoT/qwen3vl_local/sft_v3/train.py AutoMoT/qwen3vl_local/sft_v3/train.sh AutoMoT/qwen3vl_local/sft_v3/eval.py AutoMoT/qwen3vl_local/sft_v3/probe.py AutoMoT/qwen3vl_local/sft_v3/check_loss_mask.py AutoMoT/qwen3vl_local/sft_v3/test_memory_update.py AutoMoT/qwen3vl_local/sft_v3/test_kv_reuse.py AutoMoT/qwen3vl_local/sft_v3/test_gt_leak_filter.py
git add AutoMoT/qwen3vl_local/sft_v4/__init__.py AutoMoT/qwen3vl_local/sft_v4/SFT_V4_PLAN.md AutoMoT/qwen3vl_local/sft_v4/SFT_V4_RUN.md AutoMoT/qwen3vl_local/sft_v4/prompts.py AutoMoT/qwen3vl_local/sft_v4/build_dataset.py AutoMoT/qwen3vl_local/sft_v4/train.py AutoMoT/qwen3vl_local/sft_v4/train.sh AutoMoT/qwen3vl_local/sft_v4/eval.py AutoMoT/qwen3vl_local/sft_v4/probe.py AutoMoT/qwen3vl_local/sft_v4/check_loss_mask.py AutoMoT/qwen3vl_local/sft_v4/test_memory_update.py AutoMoT/qwen3vl_local/sft_v4/test_kv_reuse.py AutoMoT/qwen3vl_local/sft_v4/test_kv_vs_native.py AutoMoT/qwen3vl_local/sft_v4/test_gt_leak_filter.py AutoMoT/qwen3vl_local/sft_v4/replay.py AutoMoT/qwen3vl_local/sft_v4/collect.py AutoMoT/qwen3vl_local/sft_v4/learn.py AutoMoT/qwen3vl_local/sft_v4/launch_offpolicy.sh AutoMoT/qwen3vl_local/sft_v4/inspect_teacher.py
git add AutoMoT/qwen3vl_local/sft_loop_phase2_augment/
git add AutoMoT/qwen3vl_local/sft_new_loop_phase1/
git add AutoMoT/qwen3vl_local/sft_new_loop_phase2/
git add AutoMoT/qwen3vl_local/sft_new_loop_phase3/
git add AutoMoT/qwen3vl_local/action_prior/
git add AutoMoT/qwen3vl_local/action_expert_ablation/
git add AutoMoT/qwen3vl_local/sft_loop_phase3/
git add AutoMoT/qwen3vl_local/sft_v5/
git add AutoMoT/qwen3vl_local/sft_base/
git add AutoMoT/qwen3vl_local/sft_baseline/
git add AutoMoT/qwen3vl_local/sft_base_simple/
git add AutoMoT/qwen3vl_local/goalgen/GOALGEN_PLAN.md AutoMoT/qwen3vl_local/goalgen/GOALGEN_RUN.md AutoMoT/qwen3vl_local/goalgen/GOALGEN_V1.md AutoMoT/qwen3vl_local/goalgen/GOALGEN_V2.md AutoMoT/qwen3vl_local/goalgen/build_dataset.py AutoMoT/qwen3vl_local/goalgen/train.py AutoMoT/qwen3vl_local/goalgen/train.sh AutoMoT/qwen3vl_local/goalgen/eval.py AutoMoT/qwen3vl_local/goalgen/probe.py
git add AutoMoT/qwen3vl_local/leadmot/__init__.py AutoMoT/qwen3vl_local/leadmot/ARCHITECTURE.md AutoMoT/qwen3vl_local/leadmot/LEADMOT_PLAN.md AutoMoT/qwen3vl_local/leadmot/LEADMOT_RUN.md AutoMoT/qwen3vl_local/leadmot/build_dataset.py AutoMoT/qwen3vl_local/leadmot/train.py AutoMoT/qwen3vl_local/leadmot/train.sh AutoMoT/qwen3vl_local/leadmot/eval.py AutoMoT/qwen3vl_local/leadmot/probe.py AutoMoT/qwen3vl_local/leadmot/config.py AutoMoT/qwen3vl_local/leadmot/projectors.py AutoMoT/qwen3vl_local/leadmot/query_bank.py AutoMoT/qwen3vl_local/leadmot/heads.py AutoMoT/qwen3vl_local/leadmot/mot_block.py AutoMoT/qwen3vl_local/leadmot/decoder.py AutoMoT/qwen3vl_local/leadmot/subgoal_prompt.py
git add AutoMoT/vae_standalone/train_patch_unpatch.py AutoMoT/vae_standalone/vae_reconstruct.py
```

commit 前先看：

```bash
git status
```

如果 status 里出现白名单外改动，停下来问用户。

`AutoMoT/keyframe_filter/` 是目录白名单；`AutoMoT/keyframe_filter/collection_output/`
默认仍不是白名单，但 Phase1 四问标签轻量 JSON/JSONL 是明确例外。目录下代码、方案文档、
规则配置和手写说明可精确 add；RGB/contact sheet、summary 和其它自动调研输出只能留本地。
不要和仓库根目录或 `AutoMoT/lead_data` 下的只读参考 JSON 混淆。

push 前也问用户，不要替用户决定是否 push 到 main。

当用户同意新增/修改白名单外文件时：

- 在 `CLAUDE.md` 的默认追踪文件列表里添加同一个文件。
- 在本文件的文件修改范围 / git 规则里添加同一个文件。
- 若新增文件位于 `AutoMoT/keyframe_filter/` 下且不在 `collection_output/` 内，无需逐文件更新白名单。
- commit message 注明"按用户同意新增 XXX"。

当修改 AI 规则文档时：

- 修改 `CLAUDE.md` 时必须检查并同步 `AGENTS.md`。
- 修改 `AGENTS.md` 时必须检查并同步 `CLAUDE.md`。
- 如果新增的是项目技术事实，优先写入 `PROJECT_CONTEXT.md`；同时在 `CLAUDE.md` / `AGENTS.md` 加入口提醒或索引。
- 提交时精确执行：`git add CLAUDE.md AGENTS.md PROJECT_CONTEXT.md`（只 add 实际改动过的文件）。

---

## 6. 不要运行

本机只有源码，没有完整运行环境。不要运行这些重型或仿真相关操作：

- `lead/scripts/*.sh`
- `AutoMoT/test.sh`
- `AutoMoT/start_carla.sh`
- CARLA 仿真脚本
- 大规模数据集构建/下载脚本
- `pip install -r requirements.txt`
- 会下载大型模型、数据集、CARLA 的命令

可以做轻量静态检查，例如：

- `rg`
- `Get-Content`
- `git status`
- 小范围 Python 语法检查
- 针对单个文件的只读搜索

GPU 运行入口统一规则：

- SFT、GoalGen、LeadMoT 与 VAE patch/unpatch 的训练、eval、probe、teacher / 推理入口默认都要自动寻找空闲 GPU。
- 文档示例不要写裸的 `export CUDA_VISIBLE_DEVICES=...` 选卡片段。**唯一允许的 pin 写法**：前置 `GPU_IDS=0` / `GPU_IDS=0,1,2,3`（白名单训练入口在 `GPU_IDS` 非空时跳过 nvidia-smi 选址，直接当 `CUDA_VISIBLE_DEVICES` 用）。
- 白名单内所有 GPU 运行入口默认自动选址：单进程入口默认用 `nvidia-smi` 自动挑 1 张最空闲 GPU，并覆盖已有 mask；`torchrun --nproc_per_node=N` 入口默认自动挑 N 张最空闲 GPU，并覆盖已有 mask，再按 `LOCAL_RANK` pin 到对应可见卡。`GPU_IDS` 显式 pin 时覆盖以上自动选址，卡数从 `GPU_IDS` 逗号数推断。
- 训练 launcher 的 `DDP_GPU_COUNT=N` / `NPROC_PER_NODE=N` 只表示默认自动选址时需要 N 张卡；具体卡号默认由脚本自动挑最空闲的 N 张。`GPU_IDS` 非空时，SFT / GoalGen / LeadMoT 这类 bash launcher 的卡数从 `GPU_IDS` 推断并忽略 `DDP_GPU_COUNT`；直接 `torchrun` 的 VAE 示例仍要让 `--nproc_per_node` 与 `GPU_IDS` 数量一致。
- 运行文档里每个单卡/多卡训练示例后面都要补显式 pin demo：单卡用 `GPU_IDS=0`，
  4 卡多卡用 `GPU_IDS=0,1,2,3`，照原命令保留其它 env。
- `eval_carla/run_eval.sh` 的 `--num-gpus N` / `EVAL_GPU_COUNT=N` 只表示闭环评测 worker 数；具体 GPU id 仍由 `nvidia-smi` 自动挑空闲卡，并为每张卡分配独立 CARLA 端口槽。
- 白名单内 bash launcher 开头必须保留 `ulimit -S -c 0 2>/dev/null || true`，禁用 core dump，避免工具进程异常时生成 `core.*`；新增运行入口也要继承该约定，若工作区已有 `core.*`，不要入库，先问用户是否清理。

训练 launcher 防覆盖目录约定（详见 PROJECT_CONTEXT.md §11）：

- 所有白名单训练入口（GoalGen / LeadMoT / SFT / VAE patch-unpatch）在用户给的 `OUTPUT_DIR`（或 `--output-dir`）下再套 `run_<RUN_TAG>/` 子目录，base 层维护 `latest` symlink，连跑同名 OUTPUT_DIR 不互相覆盖。
- `RUN_TAG` 默认 `$(date +%Y%m%d_%H%M%S)`，bash 段算一次再传给所有 worker；Python 入口用 rank0 strftime + `dist.broadcast_object_list` 同步。
- `NO_RUN_SUBDIR=1` 回退到顶层覆盖式行为（vae 入口也接受同名 env，并兼容旧名 `PATCH_UNPATCH_NO_RUN_SUBDIR`），仅排查兼容性时用。
- 共享缓存必须挂 base 层：`HF_HOME=${OUTPUT_DIR_BASE}/.hf_cache`；不能跟着 run 子目录，否则会每次重新下载。SFT 已不再保留 `runtime_teacher_data/` 共享 cache（teacher 在 train batch 内现场跑、不写盘）。
- 新增训练入口必须遵循同一范本。

运行文档路径口径：

- 运行手册默认当前目录就是远端 `AutoMoT/`。
- 命令示例统一写相对 `AutoMoT/` 的路径，例如 `bash qwen3vl_local/...`、
  `python qwen3vl_local/...`、`leaderboard/...`、`checkpoints/...`。
- 不要在文档里额外写切目录步骤，也不要给 `qwen3vl_local/...` 命令加 `AutoMoT/` 前缀。
- 只有仓库根视角的文件白名单、git add 路径、或明确说明 repo root 路径时，才保留
  `AutoMoT/` 前缀。
- LEAD 数据根目录统一假设在 `AutoMoT/lead_data`，也就是用户远端在 `AutoMoT/` 下
  将原始 LEAD 数据软链接后的目录。运行文档、脚本默认值和示例命令不要再写原始
  datashare 绝对路径；数据根写 `--data-root lead_data`，keyframes 写
  `--keyframes lead_data/keyframes_all_scenarios.json`。保存路径仍写
  `checkpoints/...`。

---

## 7. 和用户协作偏好

- 用简体中文交流。
- 改复杂代码前，先解释思路和方案取舍。
- 代码注释可以用简体中文，变量名/函数名保持英文。
- 不要把大段源码复制到文档里；文档写结论、边界、源码锚点。
- 如果发现 `PROJECT_CONTEXT.md` 与源码不一致，核对后同步修正文档。


### Action prior 闭环与审计包（2026-09-06）

`action_prior/run_full_pipeline.sh` 默认训练+频繁 val+最终离线 test/probe+审计包；
显式 `BENCH2DRIVE=1` 追加正式 220 路线闭环。`action_prior/eval.sh --bench2drive` 是独立闭环入口，
不将 action checkpoint 交给旧 LeadMoT launcher。专用 agent 复用 eval_carla 的传感器/PID，
通过 `_create_runner` 恢复训练同款 action runtime，`_route_endpoint` 读取正式 benchmark XML。
闭环源代码与传感器配置另存评测身份，不绑定无关 Phase3。220 条/44 类结果包含 DS/SR/RC/IS、
效率、舒适性、五能力与每场景明细；Traffic Signs 按官方 0.0.4 单次计数口径，缺地图/记录为 N/A。
运动学只供指标，禁止写入 policy。`audit.zip` 硬限制 30,000,000 字节，核心指标必须完整，
可选案例/历史按预算选入并列遗漏；权重/缓存/完整视频/原始 TB 和运动学大产物不入包、不入库。
正式 220 test 不参与训练期选优。CPU/合成检查不代表实际 CARLA 或真实模型已验证。
### Action prior 现有权重训练授权（2026-09-06，覆盖此前仅 best/Git 非空限制）

用户明确要求共享两服务器现有权重后自动搜索、打印选择并训练，不写死 run 日期。
`action_prior` 默认 `selection_policy=available`：仅新 Phase1/2 训练包；Phase1 当前 v5 best，
Phase2 优先兼容且 guard 通过的 best，没有时允许 fallback，分别按保存步 validation Exact 选优。
支持新 Phase2 当前 v5 与冻结 v3 原提示词，原 Git 缺失/guard 失败如实警告，不篡改元数据；
同名 Qwen3-VL-4B-Instruct 允许共享路径重映射，原服务器 base 字节一致性未证明，实际本地权重
哈希进入 action 身份。未知 prompt、错 RGB、错 base family、缺权重/错 step 仍拒绝。
`--checkpoint-roots` 支持多个共享根并跟随软链接；预检保存选择清单，训练固定此清单，
resume/eval 固定 checkpoint 合同，不重新择优。`strict` 保留原 best/current-prompt/Git/path 规则。
不使用旧 sft_loop_phase2_augment LoRA，不接 Phase3，最终 KV 仍是禁用所有 LoRA 的 base。
详见 action_prior/run.md；新增选择代码/冻结 prompt/测试在现有代码白名单内，权重和输出不入库。
用户进一步要求 LoRA 独立保存与迁移：action 训练前实际复制选中 LoRA/原配置/指标/Git 到 run/lora，
禁止仅软/硬链接源文件；checkpoint 恢复/评测优先从旁边副本核验加载，缺失不重新搜索。
rank_loras 默认导出最优组合 tar.gz+SHA256，包内仅选中两阶段 slot 与必要指标/配置/来源/提示词，
逐文件和归档解压流校验；可用 --no-export-bundle 仅审计。--lora-bundle 固定包内组合训练。
真实权重迁移包不受30MB审计包限制；权重、迁移包和 run/lora 均为本地产物，不入库。

### 2026-09-07 Phase3 审计后动作合同修订

`sft_new_loop_phase3` 使用 `current_wait_first_crossing_v6` 动作规则与 `v5_current_phase` prompt：
当前确认等待优先于未来释放，增速后明显制动的混合窗隔离，横向预测第一次确认跨线。
输入新增最新帧实测速度，未来轨迹仍只用于离线标签。新索引/adapter拒绝旧合同。
按物理路线剥Rep/采集时间分组，旧审计258条路线固定train-only；同RS人工负例单轮同输入最多一次，
自由生成/eval去重并报告实际覆盖。NONE守卫取真实全NO签名；新增纵向precision/recall与独立负例支持检查。
逐帧隔离与新增负例以版本化JSONL为准，不可把场景/Town机器覆盖当全路线人工动作确认。
Phase1/2/action_prior未改；训练只读完整本地Qwen权重，不下载。详见 `sft_new_loop_phase3/REPAIR_20260907.md`。

2026-09-08 Phase3 DDP 验证等待：rank0 串行自由生成，其余 rank 等 barrier；
`DDP_TIMEOUT_SECONDS` 默认3600秒，新增生成进度/耗时与同步日志。仅缓解等待超时，
不表示验证加速或远端GPU已通过；见 PROJECT_CONTEXT.md「Phase3 DDP 验证超时缓解」
与 `sft_new_loop_phase3/SFT_NEW_LOOP_PHASE3_RUN.md`。

### 2026-09-10 Action 训练安全终止

`action_prior` 与 `action_expert_ablation` 的共享训练循环捕获 `SIGTERM/SIGINT` 后，只在
optimizer 安全点跨 rank 同步并原子保存 `latest.pt` / `termination.json`；validation 中止不发布
残缺指标，resume 归档旧终止标记，DataLoader iterator 显式清理。保存后以 143/130 退出，阻止
full pipeline 继续 eval；`SIGKILL`、掉电或永久卡死仍只能退回周期 checkpoint。该变化属于严格
执行指纹，旧 checkpoint 必须使用对应旧代码。细节见 `PROJECT_CONTEXT.md` 与
`qwen3vl_local/action_expert_ablation/run.md`。


### 2026-09-10 Phase3 RGB 审计修复

Phase3 默认索引为 `sft_new_loop_phase3_data_v7`，动作规则为
`current_wait_first_crossing_v7_rgb_guard`，prompt 为 `v6_observed_behavior_forecast`。
采集规则不允许 light_hazard 单独证明 R4，普通 trigger 不独立激活合流事件；
Phase3 读旧 collection 时保留源字段并执行有指纹的修复，不覆盖 Phase1/2 原标签。
横向 section 变化、同侧非相邻 lane-id 跳变及已审 RGB 冲突记未知，不能写为横向 NO；
-1↔+1 的正常借道仍保留。本次 54 条 RGB 开发路线与此前名单共 312 组只进 train。
prompt 与实际采集行为预测对齐，不把未来轨迹作为模型输入。评测逐例保存 RGB SHA256，
base/LoRA 配对拒绝不同真值、缺样本及输入错配。实际训练因本地缺完整 Qwen 权重未启动；
数据构建和原 meta 回读不等于新模型提升。用户已明确自行迁移后在另一台机器训练/测试，
用户使用 GitHub 同步现有白名单源码；checkpoints 内产物不入库，远端 pipeline 重建 v7 索引后训练。详见 `AutoMoT/qwen3vl_local/sft_new_loop_phase3/REPAIR_20260910.md`。


### 2026-09-11 Phase3 四图结果与短提示词审计

收到 `sft_new_loop_phase3_20260910_203334_4rgb_audit_bundle`：production 518/765（67.71%）、valid 402/640；这是旧v6 prompt/v7动作规则的成绩。
逐帧复核77例（66错例+11对照），另冻结prompt后盲标3条新val路线的同RS负例。
Phase3当前prompt为 `v7_compact_observed_forecast`，system 120→12英文词，总文本约缩短81%，保留四图及原时间/幅度合同。
默认索引目录为 `sft_new_loop_phase3_data_v8`，split seed为20260911；旧765题所属206个物理路线组加入train-only开发集合。
精确隔离3条run的40帧错误/未确认前提；不根据模型答案改速度阈值，不把视觉未确认自动写成invalid负例。
新增停车确认跨窗、单点减速、小幅变化诊断。新版尚无Qwen训练/生成成绩，旧adapter/索引不能混用新合同。
审计与操作见 `AutoMoT/qwen3vl_local/sft_new_loop_phase3/EVAL_REVIEW_20260911.md` 和 `AUDIT_SUMMARY_20260911.md`。
全源重建已通过：train/val/test为13,524/396/552行；3,274条run原meta回读无速度或有效动作不一致，物理route划分无交叉。本次完整test用 `CASES_PER_BIN=0` 评测552个独立题。
第二轮续审累计89例（78错例+11对照）、64个run、1,217张不同RGB；新增12例未支持扩大隔离或改阈值，短prompt保持冻结。新增 `audit_review_transitions.py` 仅报告身份变化/确认时刻，不自动推断视觉左右；详见 `EVAL_REVIEW_20260911_CONTINUED.md`。


### 2026-09-14 Phase3 binary/choice 全错例 RGB 审计

20260911_174046 包：binary production 373/552=67.57%，choice在306个单动作题上241/306=78.76%；同题binary203/306。choice不评NONE/INVALID/联合动作，不当完整任务替代品。
本次逐帧复核202题、122个run、2566个不同主审计帧，覆盖binary179及choice65个production错例的193题并集，另9题正确对照；不是全数据随机噪声调查，自动源规则命中不得记作人工确认。
当前默认索引v9、split seed20260914、prompt v8_shared_temporal_rules；STOP两帧≤0.5m/s都须在1.5s内，普通速度变化窗口2s、首次越线窗口3s，+3.25s仅确认末端越线。数值动作规则仍v7，没有为模型答案调阈值。
collector的R4恢复逐帧要求局部路口空间证据；Phase3只撤回有明确R1来源的stable_meta_light_with_untrusted_xodr弱恢复，保留独立事件。DynamicObjectCrossing hazard-only切入标待审，不整类改NO；精确RGB排除两条U-E3帧段、隔离两条局部RS帧段及一处lane_id/视觉跨线未确认转移。
原collection与原audit bundle不回写；精确修订通过Phase3映射层，Phase1/2既有权重不会自动更正。所有已暴露test的191个物理路线组加入train-only开发集合，累计709组；新holdout不得复用。
v9全源重建train/val/test=13500/348/468，14316行及3272run原meta回读无速度/有效动作不一致、物理路线无跨split；139项回归通过。新prompt/mapping与旧adapter/索引不兼容，新模型尚未训练，CPU索引验证不代表新模型提升。
运行使用CASES_PER_BIN=0完整评测；详见AutoMoT/qwen3vl_local/sft_new_loop_phase3/EVAL_REVIEW_20260914.md、AUDIT_COMPARISON_20260914.md及SFT_NEW_LOOP_PHASE3_RUN.md。代码、精确修订、轻量手写笔记/文档可追踪，probe_output RGB/HTML与checkpoints索引审计大产物不入库。


### 2026-09-14 Action 消融共用事件均衡课程

`action_expert_ablation/{qwen_simple,bev_only}/run_full_pipeline.sh --event-balanced` 与主线共用
`action_prior/event_balance_common.sh`、`prepare_event_balance.py`、`event_balance.py`、
`config.py`、`metrics.py`、`training_core.py`；开关/比例/算法/预算/验证聚合不维护消融副本。
支持 `EVENT_BALANCED=1`、`--sampling-mode event_balanced`、`--no-event-balanced`；CLI 优先。
full map 只用于采样与真实事件桶指标，不注入模型；消融拒绝 `--event-balanced-scene-priors`，
保持 qwen_simple 简短导航 prompt 与 bev_only 空 Qwen KV。主线 planning/分析 prompt 仍只在
`action_prior/prompts.py` 维护，不被无先验消融消费。均衡时追加事件桶/覆盖/ADE/FDE 与 sampling 审计；
uniform 仍仅核心指标。checkpoint 绑定同一采样合同，续训恢复课程且不自动重建 full map；
`--event-balance-index` 搬迁贯穿训练与最终 eval。三组配对须使用同一 action split 索引、seed、
world size、预算和源码；这次执行指纹变化要求新 run，旧 checkpoint 使用原代码。
实现与开启/关闭/调参/续训 demo 见 `AutoMoT/qwen3vl_local/action_expert_ablation/run.md`。

同日复审：开发路线名单直接复用 Phase3 当前构建器并转成 action physical-route key，
当前隔离 709 组，禁止在 action 另写日期列表。验证覆盖按全部语义桶（含 special_filtered）统计，
训练 eligibility 不变；自动 epoch 容量逐桶使用 EVENT_BALANCE_WEIGHTS 的正整数权重。
三组 pipeline 首次构建 action 索引共用 `.build.lock`，完整性检查包含 manifest 与三个 split。
本次源码及有效划分变化用于新 run，旧 checkpoint 使用原代码；远端均衡 smoke 见同一运行文档。


2026-09-14 主线 pipeline 续训入口修复：`run_full_pipeline.sh --resume 路径` /
`--resume=路径` / `RESUME=路径` 共用 `resume.py` 恢复配置，CLI 优先；进入流程前解析
checkpoint 真实路径，最终 test/probe 固定原 run 的 best.pt，不受 latest 改指影响。
显式 data-root/data-dir/model-dir/lead-bev-ckpt 路径（含环境变量）和标签/full-map 路径
贯穿恢复与最终评测；未显式提供时不把脚本默认值覆盖到旧配置。续训不自动构建索引或重选 LoRA。
操作与搬迁 demo 见 `AutoMoT/qwen3vl_local/action_prior/run.md`；此修复不放宽 checkpoint 合同。

### 2026-09-15 Phase3 INVALID 验证预算修复

Phase3 先规划同 RS 人工负例/RS/问题域覆盖，再分配均衡 source 余数；只有明确预算不足才用
`InvalidQuotaError` 触发 loss/generation 各自自动增容，保持十类与 INVALID 的 10:2 呈现比例。
缺数据/签名错误仍失败，同 RS 人工输入不重复；实际请求/有效预算、呈现与独立题数进入
`validation_sampling`。`AUTO_EVAL_BALANCE_COUNT=0` 可关闭增容；`train.py --sampling-only`
复用正式 CPU 采样预检，不加载图像/权重、不初始化 NCCL 或写运行目录，直接 Python 默认索引已对齐 v9。
细节见 `AutoMoT/qwen3vl_local/sft_new_loop_phase3/SFT_NEW_LOOP_PHASE3_RUN.md`；本次未验证远端真实训练。

### 2026-09-15 Action 自动准备缓存发布修复

`prepare_event_balance.py` 对候选/full map 的目录发布冲突重新校验，内容一致才复用；
残缺自动缓存改名到 `.invalid-*` 保留后重建，不删除 `.prepare.lock`。显式索引仍严格校验。
原因、恢复与验证边界见 `PROJECT_CONTEXT.md` 和 `AutoMoT/qwen3vl_local/action_prior/run.md`。

### 2026-09-16 Phase3 逐帧续审与覆盖预检

Phase3 当前新训练默认索引 v10、split seed20260916；prompt v8_shared_temporal_rules 和动作规则 v7 不变。
79 个定向 RGB 窗口（60 个 run）及 8 条冻结 prompt 后的新负例候选已逐帧审阅；6 条接受、2 条证据不足拒绝。
新同 RS 错事件负例仅覆盖 R3 的两个事件，val 2/test 4 个独立物理路线，不代表全域拒绝能力。
binary preflight 在模型/NCCL 前要求 val/test 各至少 2 个同 RS 负例物理路线，生成验证实际采样再检查；
Rep/采集时间不增加独立支持，choice 豁免该项。eval 默认全量 CASES_PER_BIN=0，2RGB 证据图只标实际两张输入。
本轮暴露的 176 个 test 物理组后续 train-only；精确 U-E3 撤回仅限核验的121–124窗口，真实源只有124是正例且默认风险过滤已排除，不造NO。
新 mapping 必须重建索引，旧 adapter/run 要原源码与原合同；本轮未全量重建或训练新模型。
证据、边界及运行见 `AutoMoT/qwen3vl_local/sft_new_loop_phase3/EVAL_REVIEW_20260916.md` 和同目录运行文档。

### 2026-09-16 Phase3 标定与提示词继续完善（覆盖同日首轮冻结状态）

用户进一步要求完善标定及提示词后，当前 prompt 为 v9_explicit_window_baseline，动作实现为
current_wait_first_crossing_v8_bounded_window，数值阈值和横向规则不变；新索引目录仍为待全量构建的v10。
longitudinal_decision 内部只使用当前至+2s的九个采样，窗外尾部不能触发动作或隔离窗内标签；
标定、离线action_evidence和时间诊断共用判定轨迹，缺帧/非法速度/混合阶段分开记录，不注入模型输入。
提示词明确当前速度基准、增速两次确认都在2s内；binary/choice共用横向规则，已完成历史跨线忽略，
已开始但未来才跨线仍可成立。输出模式、STOP优先级和场景定义不变。
198项测试通过；40,000标准合成窗口及60条已审run的8,587帧原meta与旧实现配对，标签变化0；79个已审案例回查通过。
未全量重建/训练；必须按新prompt/规则源码/mapping合同重建并新训，旧adapter要原源码。
6条负例保留旧prompt冻结时的盲审SHA，不倒写历史，也不声称它们在最终v9冻结后新增。
详见 `AutoMoT/qwen3vl_local/sft_new_loop_phase3/TEMPORAL_REFINEMENT_20260916.md`。
