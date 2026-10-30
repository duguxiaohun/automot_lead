# PROJECT_CONTEXT — automot_lead Compact Guide

### 2026-10-23 base ctime-only 输入清单恢复

用户要求解决远程probe的input_bindings.json散列中ctime漂移，确认尚未重新启动。新增phase3_base_hash_retry，仅bounded base的receipt绑定清单在首次仅ctime变化时追加两次完整严格读取，身份/size/mtime/ctime稳定且两次SHA匹配原回执才放行；持续变化/内容改写/替换/其它错误拒绝，stderr记rank。通用io、冻结manifest/旧SHA及原生pool生成合同不变；bounded合同显式绑定新助手/策略，旧base ready须重新cpu准入，兼容原生池验证复用。single/abc与Phase4不改，不覆盖失败_probe，不认证远程元信息漂移根因已查明。

112项相关CPU回归54.82s通过，含真实ctime注入和兼容子进程原生validator。未本轮正式全池/GPU；旧ready字节保留但base合同不再兼容。说明AutoMoT/qwen3vl_local/audit_joint/PHASE3_HASH_RETRY_FIX_20261023.md；远程在尚无正式run时sync_automot.sh→cpu base→check base→train base，后者仍先四卡探针。已运行或完成任务保持旧源码，不热改或自动迁移。

### 2026-10-23 Phase3 五类发布复核

用户授权修复并 push 精确白名单。当前 canonical 为 base bounded_v3／single v3／abc v2，五类本机正式 CPU ready 已通过；每轮12800呈现、3200更新、三轮9600，允许独立采样，base按授权有限重复。base外层使用已验证兼容快照收尾，verify base支持已完成任务补验收。

本轮补齐 single／abc 原先仅凭完成摘要和metrics文件即可通过的漏洞：包外phase3_completion核验当前来源/数据/准入/计划、三轮state.pt散列、完成步数与累计曝光、完整test成员/标签，并从cases重算metrics；不加载模型、不重新推理，不改绑定的训练代码或ready。compare.sh继承禁core设置，运行说明统一当前版本。精确暂存副本不含.git，350项不同相关CPU测试通过；共享测试文件保留本机Phase4测试、仅暂存本轮范围，16项复核计入350不累加。结果和限制见AutoMoT/qwen3vl_local/audit_joint/PHASE3_RELEASE_REVIEW_20261023.md及phase3_release_verification_20261023.json。原13份base证据及single162/abc160项绑定源码保持，旧回执不重写。数据/权重/RGB/完整日志与其它任务改动不发布。

真实4B四卡探针仍未执行，不能把CPU回归当作正式GPU训练验收。稀疏test、val缺类、已知开发曝光与3/13米边界视觉审查限制保留。服务器同步仍bash ./sync_automot.sh；已有base完成先verify base，按未完成方法继续，不能重跑train all覆盖已有实验。

### 2026-10-23 base 外层完成验收修复

生产代码仅改phase3_compare.py，base外层收尾调用已校验phase3_native_runtime快照中的真实verify_training，固定cwd/PYTHONPATH和-B，保留全部SHA/9600步/38400消费/曝光/adapter完整性检查；成功写回执后train all继续后四组。新增verify base可为已有内层completed_quality_pending任务补验收，不建库、不GPU、不改实验产物，不应因此重训；失败或损坏证据仍拒绝。完成base后按缺少方法train single/abc，不重跑train all绕过碰撞。

108相关CPU测试59.44s通过（含此前99和新增9，不累加），真实兼容子进程+合成完整消费回执，训练与权重为测试替身。正式a1fd2101快照合同仍匹配base ready，13历史证据不变，single/abc182源仅compare外层变、其余181一致。五类CPU已ready，无需重建；未GPU/远程/push。说明PHASE3_COMPLETION_FIX_20261023.md、回执phase3_completion_verification_20261023.json。single切入test12查询STOP/RESUME质量3/9而非全STOP，仍一物理组；缺类和4/13米候选视觉审查待完成。

### 2026-10-23 base 有限重复 v3 正式 CPU 全池通过

用户明确“适当允许部分稀少数据重复”。新增 phase3_base_bounded.py/.sh 与独立 bounded_sampling，canonical phase3_compare base已切bounded_v3，旧coverage_v2不改采样规则、旧拒绝证据保留。完整原生train候选索引复用，不动标签/风险隔离/val-test/source SHA。每事件1280、每轮3200更新、三轮9600/warmup2500不变。稀缺事件先覆盖全部有效帧，同事件单轮最多2，充足事件内最多1；整数最小费用流联合处理共享帧，先最少物理重复再最少累计曝光，跨事件物理帧单轮最多2；同事件run上限ceil(3840/容量)或3，跨事件共享物理上限6并为后续轮次保留容量，无隐式续训。

真实全池发现SIGNAL_FAILURE/JUNCTION_RULE_CONFLICT共享447帧、并集2430<合计2560；不能只补355+25事件内重复。最终每轮额外物理呈现485=380事件内+105跨事件，3.789%；独特物理帧12315，三轮新帧12315/8960/7703；实际累计单帧最大5（dynamic5，signal4，junction同事件3/跨事件4）。正式来源校验/原生复用/三轮投影均通过，checkpoints/phase3_base_bounded_v3/ready_1280.json为ready。旧strict仍rejected。单帧/abc冻结源码不改且既有ready保留。

最终99项CPU回归26.61s通过，含原生小CPU真实优化/失败曝光、共享容量穷举、adapter配置/评测、新旧入口；另真实四Gloo+Linear DDP三轮120次合成呈现、逐rank ID/原生消费一致，不叠加pytest数、不称4B GPU验收。模型权重保存测试替身明确。本机缺完整4B模型，未GPU/SSH/push。发布后sync_automot.sh→前台compare cpu base→check base→probe base全部成功再后台train base；正式run为phase3_compare_base/run_base_bounded_v3，探针另有_probe。老coverage命令和geometry_study six不代表新版本。

说明PHASE3_BASE_BOUNDED_RUN_20261023.md，回执phase3_base_bounded_verification_20261023.json，证据checkpoints/phase3_startup_review_20261023。一次诊断漏-B增加独立缓存62个pyc，校验正确拒绝；原源码不变，带旧回执完整保留到审计目录后原生重建同源快照，未改SHA。中间共享容量拒绝报告保留。保留其它未提交工作。

### 2026-10-23 base 真实启动失败与完整训练池审计

用户补回底层日志确认旧 base 在 prepare 因 DYNAMIC_CUTIN 容量不足退出；旧 geometry_study six 则缺自身版本 ready，不等于新版 single。新增独立 phase3_base_pool：保留全部原生 train 候选，原 val/test/标签/INVALID 不变；兼容运行时绑定，CPU 无需模型/GPU，正式 coverage launcher 同池。主入口提前核验 ready/拒绝原因，底层错误直出；新增前台 probe base 独立目录，不训练、不写 latest、拒绝覆盖。旧冻结源和旧产物保留。

本机完整原生扫描、依赖前后校验、选择题有效性与容量实际执行完毕：train187982（含INVALID1850），动态切入925、信号灯故障1255有效不同帧，正式ready为rejected。不是正式三轮通过。动态切入原始事件候选1416，327旧视觉/标注风险（238主体未确认、89道路上下文），51纵向混合/4速度窗口不足，留下1034=train925/val45/test64；逐帧重放与原生全池一致，本事件没有四图缺失损失。旧原生采样器重放1280时分别重复355/25呈现，最大2；不声称历史某run实际曝光。即使327全部有效且可入train，上界1252仍不足1280。用户追问数量原因，尚未授权改配额/放重复；原1280、单轮1全run3保持，不能提供旧后台开训成功承诺。

85项相关回归+最终入口10项有重叠，去重87项通过。single v3/abc v2原生ready重载仍为三轮3200；此前182源仅公共compare入口变，其余181匹配，旧回执不改。无4B模型，未GPU/远程/push/新RGB人工审查。当前证据checkpoints/phase3_startup_review_20261023，正式base拒绝指针checkpoints/phase3_base_pool_v1/ready_1280.json；说明PHASE3_BASE_STARTUP_REVIEW_20261023.md及phase3_base_startup_verification_20261023.json。旧运行文档已加阻断提示，保留其它工作。

### 2026-10-23 single／abc 新版正式 CPU 全池准入完成

按审计结论推进真实全池，single 首次统一入口复现 output_overlaps_input：CPU 把地图视图放在 study-root/maps，触发 prepare 输入保护。修复为 study-root 同级的 <name>_maps，保留地图校验和重叠拒绝；测试取消地图解析器替身，串接真实默认／显式发现与原生小池 CPU。26 项直接相关测试 8.49s 通过，不与历史批数相加。上轮30文件回执中仅 cpu.py 和对应测试改变，旧回执保留。

修复后 single v3 正式 CPU 退出0／completed／ready，260719父锚点→204910有效（train202516/val573/test1821），三split十事件齐全。完整来源校验、六动作重算、缓存重载和三轮预算通过。移动3个完整已知非曝光train组：HardBreakRoute/Town12_4357_0→test；EnterActorFlowV2/Town12_975_0→test；MergerIntoSlowTrafficV2/Town13_1398_14→val。原val/test不动。test跟车制动58锚点219查询1组，train剩3106；信号灯故障1查询1组。动态切入12查询含STOP／RESUME，目标质量3／9，与上轮replay_final逐字段相同，并非全STOP。val无纯左变道支持。

abc v2 也正式 CPU 退出0／completed／ready，259493有效（train256469/val740/test2284），三split十事件齐全；a/b/c共享成员和计划，本轮源码未改。两包各三轮12800呈现／3200更新，总9600，每帧单轮1全run3；single新帧12800/10306/9506，abc12800/11272/9689，均仅计划投影。abc val a/b缺左近、c缺左右近；test跟车制动2锚点1组，仍沿原划分。

本机正式状态见 checkpoints/phase3_single_v3 与 geometry_abc_v2 的 ready_1280.json、cpu_status.json；任务均已结束，无需重复启动。single162／abc160绑定源码运行前后匹配。说明 PHASE3_CPU_ADMISSION_20261023.md、回执 phase3_cpu_admission_verification_20261023.json，证据 checkpoints/phase3_cpu_admission_20261023，首次失败日志与状态保留。未执行本轮base全池、四卡GPU、在线、RGB边界盲审、远程或push；不称五类训练全部跑通，不认证未知外部曝光，保持其它改动与旧库权重。

### 2026-10-23 single 真实 test 缺失与本机地图入口修复

用户审计确认旧single v2父库test跟车制动4候选全部原生六动作无效；授权修复。新增独立phase3_single_v3，原生六动作有效性过滤后补齐val/test事件，仅从未列入当前development／额外曝光清单的train完整物理组选择；原val/test固定，标签不改，abc不受限制。保护每事件1280训练不同帧并用原生共享帧规划核验，失败回溯；unknown/infeasible不发布部分移动，不放宽准入。保存parent_split、原/新assignment、移动组及曝光来源SHA，缓存/准入/模型绑定；未认证未知远程曝光。额外--exposure-manifest支持已知记录。信号灯故障与动态切入稀疏测试支持仍警示。

phase3_compare.sh的single现指向v3、新输出checkpoints/phase3_single_v3；abc v2运行时保持冻结。新phase3_maps优先项目CARLA完整十二Town，再查应用内/兄弟lead已知目录；统一CPU入口传已校验平坦视图给single/abc，开放--maps/--geometry，显式路径不隐式回退、不混拼不完整源、冲突与旧视图变更拒绝。single v3直接CPU同样接入。真实本机默认根lead/3rd_party/fail2drive/toolbox/carla_xodr十二Town通过，不推断远程缺图。旧abc v2独立CPU默认逻辑保持历史语义，新入口已绕开其本机路径阻断。

本轮真实重放三个事件全部候选425路线：LEAD_BRAKE train3164/val34/test0，SIGNAL_FAILURE 1430/9/1（test1查询），DYNAMIC_CUTIN 1346/3/3（test12查询1组）。发现14个未列入绑定曝光名单的有效跟车制动train供体组；固定哈希首候选HardBreakRoute/Town12_4357_0有58有效锚点219查询、仅该事件，移出后train3106不同帧。此为真实候选证据，非全事件正式新划分；旧test346/347/349/350仍全部无效。重放绑定父SHA、来源代码、metadata前后SHA和地图，不修改父库。

最终本轮172项CPU回归37.32s、1项未改动正式预算大合成测试排除；不能与此前同为172的不同测试批相加。含真实规则合成三split十事件后过滤修复、旧父字节不变、ready/缓存复用/篡改拒绝、曝光禁供体、穷举可行性对照、小Qwen/expert及旧study/abc回归。81个旧运行源码字节未变；新正式全池CPU/4B GPU/在线/新RGB边界审查/push均未执行，不伪造ready。说明PHASE3_SINGLE_REVIEW_FIX_20261023.md，回执phase3_single_review_fix_verification_20261023.json，本机证据checkpoints/phase3_single_review_fix_20261023。保持其它已有工作；发布后远程同步仍bash ./sync_automot.sh。

### 2026-10-23 Phase3 五类预算与区域分类 loss 独立版本

用户根据附件最后澄清授权实现：仅统一数量／更新，允许各自采样；abc明确九类内部softmax分类loss。新增 canonical phase3_compare.py/.sh（base/single/a/b/c，cpu/check/status/train），base沿用coverage_v2；新phase3_single_v2独立重算六动作、仅原生当前与未来有效性，无abc未来交集或采样质量；共享已验证父库的当前观测／划分要求，父库几何未来无效记录仍可进入single。新geometry_abc_v2九logits内部归一化，预测概率和1，保留全词表诊断；阈值不变，旧库／权重不自动迁移。

五组正式固定每轮12800呈现／3200更新，三轮9600，warmup2500，seed20260904/lr1e-5/原生余弦，最终第三轮test；single/abc全有效val诊断，宏最佳轮仅记录。物理帧每轮1全run3、成功优化后曝光、容量不足拒绝。新增支持与留一物理组敏感性诊断（非置信区间）、训练集3/4/5m候选支持，不自动改边界。c左右近3m为96/28，4m947/256，5m1822/908；无新RGB边界验收，保留3/13m。

主回归171项119.10s、收尾11项2.98s，有重叠共172不同测试；原生小Qwen、两Gloo、expert保存重载、原生六动作有效而几何未来全无效、缓存／源码篡改／失败入口等通过。真实26路线父204：single151、abc202，最终源码绑定重载及全原始源验证通过；仅train小池按设计拒绝ready。全池abc流式绑定父SHA复算260719→259493，val740，三轮3200，新帧12800/11272/9689；不代替全量原始来源重校验，不创建新版ready。

旧geometry_v1/study/abc/single_frame共61运行源码字节未变，旧abc v1已完成ready但不代表新版ready；新输出phase3_compare_base/phase3_single_v2/geometry_abc_v2。未运行single真实全池正式容量、4B四卡／在线／新RGB盲审／push。旧公共入口保持旧版本复现含义；本轮新入口和步骤见PHASE3_COMPARE_20261023.md，精确回执phase3_compare_verification_20261023.json，证据checkpoints/phase3_compare_verification_20261023。保留其它已有改动。远程发布后同步仍bash ./sync_automot.sh，不要求Git历史。

### 2026-10-23 a／b／c 与六动作有效性彻底解耦

用户明确区域标定不应依赖旧六动作，授权重新完善代码。新增独立 geometry_abc 包，沿验证后的原生几何父库派生 a/b/c，不调用 six.build/label_actions，无当前六动作门或六动作未来交集mask；只按已知道路/至少一个有效几何端点保留。当前原点、速度、车道参考及过去位置历史沿用；a 阈值、b 高速20/24m其它3/13m、c统一3/13m均不变。三组共享原始成员/有效mask/顺序，标签各异，采样质量只含abc；默认每事件1280/三轮各9600、warmup2500、每帧每轮1/全run3。新合同 current_reference_geometry_abc_v1，模型/准入/计划/expert均绑定；旧四组库和权重不自动迁移。

geometry_study_abc.py/.sh 公共入口现转向新包，默认输出 checkpoints/geometry_abc_v1；cpu/status/check/train all流程保留，CPU新增互斥锁/阶段状态，容量通过不覆盖覆盖拒绝。原 geometry_study/geometry_v1 源码与旧165绑定完全未变。新包为冻结运行时的独立版本，避免破坏six与旧模型复现。读库对同一份原始字节hash及JSON解析；缓存验证后避免立即重复全原始源扫描。已有其它未提交工作保留。

129个不同相关CPU测试通过（整批128项22.59s；独立文件4项1.11s，与整批重叠3项），含真实规则记录结尾six当前/未来全部无效而几何四端点有效、原生小Qwen、两Gloo、expert保存重载、旧study回归。最终代码流式复算本地260719父锚点→259493有效，三split十事件齐全，固定验证三组纯类可定义，正式1280三轮计划各3200更新通过；每轮新帧12800/11270/9689、每帧单轮1全run最多3。此复算绑定父库SHA，但不替代全量原始源校验，不伪造ready。

正式CPU建库仍在运行：checkpoints/geometry_abc_v1/cpu_status.json、geometry_abc_cpu_verified_20261023.log，写本条时PID195850（exec43243）。先读状态，不重复启动或修改其绑定源码。先前本轮PID195045为修正新入口读库绑定主动中断，失败目录及geometry_abc_cpu_20261023.log保留，未动其它任务。未GPU/在线/新RGB盲审/push。test跟车制动仅2锚点1物理组，覆盖不是精度可靠性认证；本地val匝道11与远程25不同，不称两机等价。说明 geometry_abc/README.md，精确边界及运行快照 verification_20261023.json；完整复算 checkpoints/geometry_abc_verification_20261023/full_pool_report.json。

### 2026-10-22 四图独立 base 增步与白名单入口

用户再次明确 sequence 已彻底放弃，授权补齐四图 base 9600步、独立采样、未见优先及重复限制后发布命令。新包外 phase3_base_coverage.py/.sh 和 sampling：十事件各1280、四卡B1/累积1、三轮9600/warmup2500；基于原生有效choice锚点，独立打乱，事件等额、动作自然/sqrt温和偏好，跨事件同帧每轮1/全run3，容量不足拒绝。CPU先投影三轮并复核实际形状；完整优化与消费后推进历史，失败无隐式续训。原提示词/动作/四RGB/单题loss及最终第三轮test政策保留；新增逐rank ID/曝光/预算验收与最终adapter SHA合同。不是sequence回滚，也不是只改环境变量绕过旧7680校验。

显式入口 phase3_launch --method base --base-budget coverage_v2；旧default原样。发布必要兼容源码bootstrap、公共/单帧数据复用与prompt helper，原prompt仍默认，未启动澄清prompt训练。runtime使用既有发布484快照核验原生261源，服务器无Git历史要求，不改旧SHA。精确index构造无.git应用副本，106项CPU回归43.61秒通过（含新增19项）；真实小CPU原生优化/失败/保存、完整38400呈现合成投影、四Gloo旧周期验证与原生单图回归包含其中。165 geometry绑定源完全未变，不操作既有geometry任务。

未实测真实全池base1280容量、4B/GPU、服务器耗时或新精度；单帧新版是geometry_study six，同9600预算但共同池/四查询/验证选模与base不同，不能声称完全配对对照。服务器sync_automot.sh后运行phase3_base_coverage.sh；six须当前CPU成功ready后remote.sh train six，无重复cpu/no隐式续训。两组各9600共19200更新，同一组四卡顺序跑。说明PHASE3_BASE_COVERAGE_RUN_20261022.md、回执phase3_base_coverage_verification_20261022.json。仅精确白名单相关代码/测试/说明，保留其它未提交工作、旧权重与数据。


### 2026-10-22 geometry a/b/c 发布入口复核

用户授权继续完善并 push 白名单、给出 abc 测试训练步骤。复核 v2 供体/退出竞争修复；新增包外 geometry_study_abc.py/.sh：CPU v2 入口、只读 status、正式准入/三轮计划/模型分片/四卡与全部输出碰撞前置检查，a→b→c 串行训练和原生验证选模重载 test，首错停止，操作日志与完成回执绑定表示/数据/9600步。新外层不接收隐式同名续训，不改159冻结源码或当前165源接续job、不重启全池任务，不启动本机GPU。每组四卡三轮9600、三组合计28800更新；旧2.12s/步粗估17h纯训练，另加准备评测，非本次GPU实测。

159项相关CPU回归26.95秒通过（含新增11入口测试）；从精确Git index构造无.git应用副本，补齐测试依赖后192项回归49.22秒通过；两批有重叠不相加，结果见geometry_study_abc_release_verification_20261022.json。发布仅geometry study/CPU审计/运行入口及原生数据准备必要文件与轻量记录，.vscode、其它P3/P4改动、RGB/原始数据/权重保留；data_prepare本轮验收限geometry路径，其它阶段新增集成不作为本次发布验收。当前全池仍在运行、尚无准入结论；服务器先sync_automot.sh→abc.sh cpu，完整通过后check→train all；拒绝则交CPU报告，不降低覆盖/预算。详见audit_joint/GEOMETRY_STUDY_ABC_RUN_20261022.md。


### 2026-10-22 CPU 审计供体与接续竞争 v2 修复

用户复核指出填补供体查车道失败时证据回落错帧、/proc存在检查与stat读取间退出竞争，另要求图册优先不同路线/起点。新增包外 geometry_study_cpu_audit_v2.py/.sh 和 geometry_study_cpu_continue_v2.py；原159冻结源、旧审计3文件及v90 587源均未改，不热改旧SHA。v2供体从imputation.selected_endpoint_frame优先确定，缺失/冲突/缺供体原证据拒绝；不依赖成功写direction_origin_frame。候选按类型/Town/事件有界散列，去重并优先新物理组/Town/事件/原生起点，报告实际支持，非独立变道实例覆盖认证。接续单次读/proc，消失视退出后查回执；PID复用/僵尸终止等待，权限错误不掩盖；job源SHA和单次启动锁/输出拒绝覆盖保持。

81项CPU回归16.30秒通过（原64+新增17），含真实transform地图失败保留供体9、退出竞争有回执继续、真实短命进程、源变化/重复启动拒绝。真实26路线小池v2 CLI与v1 pool_report一致，仍151共同/覆盖及1280容量拒绝/退出2；190不变方向、99去重，无真实翻转或新RGB人工验收。

完成验证后只交接等待进程：通过pidfd暂止旧supervisor确认waiting且无子进程，归档旧状态后退出旧163657；原生PID162503/start_time10087944未变，未发信号/重启原生建库。新supervisor168192（exec75733）绑定continuation_v2_job.json的165源，原生成功后用v2写full_report_v2与study_audit_v2.log；旧工具/脚本/结果保留。当前原生仍native.build，新接续waiting_for_native，无全池准入结论。继续先读checkpoints/geometry_study_pool_20261022/pipeline_status.json，不重复启动或热改绑定文件。交接证据checkpoints/geometry_study_cpu_review_20261022/handoff_from_v1.json。新说明GEOMETRY_STUDY_CPU_REVIEW_20261022.md及geometry_study_cpu_review_verification_20261022.json；未GPU/SSH/push/标签阈值预算改动，保留其它工作。

### 2026-10-22 geometry_study 转入全池 CPU 证据阶段

用户复核四项修复通过，要求停止反复修补并转向全池准入。新增包外 geometry_study_cpu_audit.py/.sh，不改已验收 geometry_study 159 源及表示/阈值/预算，v90 587 源亦匹配。原生版本缓存父库→原 study prepare→独立覆盖/1280三轮容量报告；即使holdout拒绝仍可测训练容量，诊断不写ready、不盖过拒绝。固定验证报告缺类/混合事件/最大单物理组占比，不加新门槛或挑选成员。可选全train原生父候选方向重放，含共同排除锚点，保存翻转/近垂直/其它无效、原生跨线证据和实际填补来源、连续RGB SHA/本地图册；自动图册非人工验收。

64项CPU测试15.90秒通过（原54+新增10），真实小池CLI 26路线204父锚点→151共同锚点，覆盖与1280容量均rejected，完整落报告退出2；190个D/E出现位置均不变，99个事件/原点/类型去重，不外推全池或新RGB验收。当前真实全池已启动但尚未完成：checkpoints/geometry_study_pool_20261022/native_prepare.log、pipeline_status.json；native成功后本机continuation自动启动共同池报告及训练方向重放，无GPU。写本条时native PID162503、supervisor PID163657在运行（exec会话91171/89877）；后续先读状态，不重复启动。第一轮候选枚举105.49秒、依赖SHA443.38秒，原生仍会重复完整来源复核，不删证明/跳校验提速。若冻结源码改变则接续拒绝。尚无真实全池准入/1280三轮/真实方向边界验收结论，失败缓存保留。

默认CPU命令（发布后已有AutoMoT根）bash qwen3vl_local/audit_joint/geometry_study_cpu_audit.sh --directions；同步仍 bash ./sync_automot.sh。无需模型权重或GPU；旧报告碰撞拒绝，另给语义化output。GPU启动/重载/评测仍在全池通过之后，当前未执行。说明 GEOMETRY_STUDY_CPU_AUDIT_20261022.md 与 geometry_study_cpu_audit_verification_20261022.json，后者是运行中快照；最终状态看本机pipeline_status。未SSH/push/新RGB人工审计，保留其它工作。

### 2026-10-22 geometry_study 训练前四项复核修复

用户复核要求处理共同池缺测试事件、六动作无事件解码限制、固定验证全混合导致选模不可定义及 expert 权重未绑表示。当前独立 study 合同升为 current_reference_matched_study_v2；新增 admission：共同过滤后 train/val/test 十事件覆盖先验，CPU 固定每事件ID前16验证成员并报四组纯/混合/缺失、类别/物理组支持，任一纯类macro不可定义即拒绝，再做容量。准入/计划内容寻址，就绪指针 preparing/rejected/ready，旧ready不兼容；训练重算准入并把SHA/成员绑checkpoint，恢复/评测核验。未补齐真实缺事件，未移动旧test或改变split要求。

six prompt/predict 恢复原事件 action_keys+KEEP（LEAD_BRAKE A/B/C/F），保留全词表/未限制输出及域外诊断，全词表准确率按原始输出；a/b/c仍九类。RepresentationExpert 在新expert训练前绑定完整合同，state_dict额外状态/权重回执SHA/调用均校验；缺/错合同包括strict=False拒绝，旧未知权重不自动迁移。原始源仅训练rank0完整检查，shell检查数据JSON/代码/准入/计划避免重复原始源扫描。

188项相关CPU回归63.44秒通过（study54，新增28）；最后避免重复扫描改动后5项准入4.34秒再通过。含原生合成30路线三split十事件完整建库/准入/缓存复用、全混合与缺test反例、16组expert表示加载、原生小FM保存重载同噪声输出一致。26既有真实train路线重派生仍151共同锚点/517查询、保留24路线，所有records及小计划不变，建库/复用SHA一致；明确拒绝正式ready，非全池覆盖/1280验证。159依赖绑定，旧非study142源和v90 587源不变；旧回执不改。未新RGB盲审/新真实D-E翻转/4B GPU/SSH/在线/push。见 geometry_study/REVIEW_FIXES_20261022.md 与 review_verification_20261022.json；完整证据 checkpoints/geometry_study_review_20261022。保留其它工作。

### 2026-10-22 六动作与 geometry a/b/c 共同样本实现

用户附件由方案审计转为代码实现。新增独立 audit_joint/geometry_study：six 保留原生 A/B/C/F 局部速度及未来证据，已确认 D/E 投影到当前车道方向，近垂直 abs(cos)<0.1 缺失；原填补后按实际来源参考转换。a 原标定不改；b 当前R3高速20/24m、其它3/13m，c统一3/13m，所有时距/历史共用绝对边界，中界叫MIDDLE，FAR不截断。缺道路不猜。六动作离线历史允许后续证据，geometry因果位置历史，不能称等信息比较。

新共同池从原生验证geometry与原始六动作规则派生，当前六动作有效、未来共同mask、无有效查询剔除，原split隔离保留；不是旧single_frame锚点列表逐字复刻。十事件等额、跨事件物理帧每轮1/全run3、实际成功更新曝光及路线/4秒片段优先，四表示未来质量一半natural/一半sqrt软偏好，共用ID/顺序；增广容量检查失败不放宽。候选1280/world4/B1/三轮9600、warmup2500；1024/7680控制保留，正式全池容量未验。固定每事件最多16验证成员按纯类macro recall选模型，完成全预算后test只收选中checkpoint。

新增历史冲突/拷贝、纯/混合分开、转换率/误报率及分事件时距类别支持；val配对历史清空/反序/黑图，diagnose-legacy保留原geometry权重prompt合同。expert四token条件/FM同噪声接口与共同ADE/FDE，混合hard oracle拒绝；未默认接在线agent。按内容SHA的本机缓存/锁/失败保留，整库级派生复用、尚无逐路线增量；remote沿用项目资产，sync仍bash ./sync_automot.sh。旧生产模块/库/权重不改，不改旧SHA。

160相关CPU回归57.82秒通过（26新增），含四表示原生小Qwen backward/token/reload、真实两Gloo曝光、原生FM梯度、共享容量穷举、源篡改拒绝/缓存复用；旧geometry remote测试夹具同步当前profiler入口与重复复用，非改生产。26真实train路线204候选→151共同锚点/517查询，十事件小计划三轮四组一致，a有效值不变，D/E本小池变化0；最终新缓存重载与逐字复用通过。v90 587源仍匹配。仅复看既有两图册12RGB面板，无新盲审/全池正式容量/4B GPU/SSH/在线/push；已用test的划分只能开发对照，独立结论须新未曝光物理组。详见geometry_study/README.md与verification_20261022.json；原始证据checkpoints/geometry_study_verification_20261022。保留其它已有工作。

### 2026-10-21 远程同步命令永久约定

用户明确要求：以后提供远程服务器代码同步命令，统一使用 `bash ./sync_automot.sh`，不再使用 `git pull --ff-only origin main` 或直接 git pull 作为替代。命令在该机已有 AutoMoT 应用根执行，不重复拼接 AutoMoT、不修改同步脚本、不清理本机数据和权重。本条覆盖历史运行说明中相冲突的远程同步示例；本机经授权的 GitHub fetch/push 发布流程不受影响。详见 PROJECT_CONTEXT.md 同名条目。

远程应用目录只接收白名单文件覆盖，不代表本机发布仓库，也不要求包含本机提交历史或可用 `.git`。部署和运行代码禁止依赖远程 git rev-parse、git archive、git checkout 或历史提交；旧模型兼容源码必须作为明确白名单文件随 push → sync_automot 分发，并按文件 SHA 校验。不能要求用户修复远程 Git 或补拉历史来满足运行入口。Phase3 本轮新增的 `qwen3vl_local/audit_joint/phase3_behavior_diagnostic_v1/runtime_sources.json` 及 `compat_sources/qwen3vl_local/leadmot/{config,decoder}.py` 纳入既有源码白名单；它们是兼容源码与清单，不含权重、RGB 或运行数据。


### 2026-10-18 v86 磁盘写入额度修复与完整回放恢复

用户日志8614/8614、15854.8秒后guard低空间。确认旧CPU入口max(20,free-12)人为限制约12GiB新增写入；两个CPU入口改绝对20GiB保留（可提高），记录free/reserve/writeable及完整需求未知，guard日志打印实际字节。不能认定服务器磁盘已满/空间必够。新增RESUME_FROM本机旧输出＋新OUTPUT_DIR：全部原生gzip路线核验通过才复用，缺索引原子独占恢复，已存在索引只核验；缺失/坏文件/旧合同/部分索引/发布后索引丢失明确阻塞，绝不退回重新计算。旧回放永久作为新库引用，不删除；两机任选但不混路径。不改teacher v24/学生/prompt/采样/评估，UE5能力限制保留。
116相关测试11.08秒通过；真实2路线231帧副本恢复索引字节一致、路线SHA不变、0重算。CPU CLI恢复后正确在局部池覆盖不足门阻塞；本机shell被既有piped core保护拒绝，前置/576源码/空间报告/失败包通过，不称完整服务器验证。未GPU/远程/新RGB目视。原始副本checkpoints/v86_recovery_verification_20261018，轻量回执随白名单。服务器同步后按BUILD_DISK_RECOVERY_20261018.md只续接本机本次v24旧输出，交新handoff两fullflow文件，缺目录交launch.log；不再readiness/盘点/全量回放。保留.vscode/paper等其它任务。


### 2026-10-18 v85 白名单发布与本地连续RGB

用户明确授权完善/push白名单并行RGB。发布local_paths及机器无关入口，不指定机器，服务器步骤为同步后后台一次v24正式池更新，非重复label_release_check/readiness；保留两机独立数据/权重/缓存，使用本机DATA_ROOT/MODEL_DIR、新OUTPUT_DIR。文档LABEL_RELEASE_RECEIVED_20261018.md含nohup/进度/交包，前置失败交launch.log；正常或中途失败交handoff两fullflow文件。不看到teacher完成就中止。
162回归再次10.12秒通过，573源码匹配；原生teacher、学生、prompt、采样及两模式窗口不改。2完整train路线231帧重放/693源SHA、两模式各48train/无valtest，questions/instance_ends与前v24逐字段相同。实际目视24RGB/72源SHA，相对前140+40+16集合新增10；UE4 65–67UNKNOWN→68YES，69不连续终止非完成；UE6 61–68实例前/69–72横穿仍NO，信号关联可观察性限制不扩大为确证错误。未发现新必修改标依据，不调整阈值提示词；UE5proceed YES仍0，不作完整能力认证。RGB/回放留checkpoints/v85_release_rgb_20261018，轻量v85_release_rgb_verification随白名单。未GPU/远程任务，保留.vscode/论文及其它产物。

### 2026-10-18 用户最终机器选择约定与正式历史池采样包

用户明确禁止指定必须去哪台服务器执行：以后用户任选cr02/cee，名称仅报告元数据，命令必须基于当前机器实际路径，不假设数据/权重/缓存互通，不猜最新目录，不自动迁移。新指令覆盖此前“只让cr02运行”的限制。
收到label_release_report.zip 33796字节，CRC/成员SHA/独立JSON及v84的571源码一致；173.637秒诊断完成，最后多粘贴右括号发生在程序完成后，无需重跑。298770纯自动规则题，manifest/train SHA与原完整准入一致；v5两模式十事件各1024，独立10036/单题最多2次，UE2returnYES1/1、UE7completeYES3/2，UE5proceedYES仍0。八合同差异是两处teacher修复、dataset/identity指针及plan/registry更替，旧SHA全匹配v83，不是8新缺陷。training_authorized false是只读工具固定属性。旧val37/test18每事件只引用此前已核验包，本包未重读holdout。
本轮新增local_paths检查本机显式数据/模型配置/输出，build/train脚本从自身定位AutoMoT根，缺资产/危险输出先于扫描拒绝；取消label_release_check强制机器标签。162回归5.46秒通过、实际无AUTOMOT_SERVER本地CLI通过、v85清单573源码。teacher v24/plan39/registry34、学生、采样、eval、prompt均未改，不引入新标签版本。未新增RGB目视/训练/push。当前不再跑readiness/盘点/旧诊断；下一步发布后任选有原始数据机器，一次新v24正式池更新与准入，再新实验首轮。不存在已验证的旧teacher免回放迁移，不改SHA绕过，不承诺重建耗时或现有空间够。命令和完整边界在audit_joint/LABEL_RELEASE_RECEIVED_20261018.md，交新输出handoff两个fullflow文件；保留所有旧数据/权重与.vscode。

### 2026-10-18 U-E5 本地通过时序与独立否决修复

按用户要求继续本地代码/标定/RGB，不安排训练。72条train路线109实例截止窗口重读1765不同帧/5295源SHA；截止诊断78几何已在车后但非可见放行、11未观察到、9部分可见未分离、8身份断裂、3无离场证据。侵入后没有可见且分离的观察，不把出画/几何车后变YES。departure_probe新增passage_evidence，yaw-aware完整车体后方间隙、侵入后时序和身份断裂分别报告，明确只诊断不授权，7新增测试。
原生teacher_events在UE3/UE5隐藏目标时直接返回空事实，确实抹掉其它可见阻挡/STOP否决；现只保留独立False，未知目标不补True、不增放行。8新增原生样例旧版4失败4通过。3完整真实路线266帧新旧回放、582两模式原始题成员/实例结束不变；1176_0 f73-75每模式3个UNKNOWN→NO，已编译入两模式各130train、无val/test，配对重载通过。独立占用者为前方黄色车4753，其lane1与ego lane2不符合既有ordinary_follow豁免；RGB确认前车存在，不等同全部跟随条件独立视觉认证，不据此放宽阈值。实际目视3路线31RGB（1176_0 67-79、4591_0 54-62、1378_0 71-79），93目视源、合并5946源SHA通过。1403回归通过2旧生产池缺失跳过119.79秒；35专项包含其中。教师v24/plan39/registry34，旧合同保留、提示词阈值不变；不改旧服务器SHA，不立即全池重建。UE5 proceed YES仍0，修复成功的是负证据保留而非完整放行能力。报告audit_joint/UE5_PASSAGE_REVIEW_20261018.md、ue5_passage_verification_20261018.json，本地checkpoints/ue5_passage_20261018。未GPU/远程/push，保留其它任务文件。

### 2026-10-17 用户纠正：当前先本地 Phase4 标定，不安排训练

用户明确不需要训练，要求本地产生标定、逐帧 RGB 和原生修复后判断初步质量。此前 cr02 补包通过不等于标签质量验收，不再按该页安排服务器首轮。cr02 只有题库无训练/缓存，cee 独立，历史结果不混用。
本轮真实 57 条 train 路线/7101 帧，v22、v23 各原生回放+编译+两模式重载约471秒；各3187训练题、无val/test。实际目视19路线20窗口140不同RGB（开发复核非盲审），21303源SHA及420目视来源核验。确认并修复teacher_rules的UE1/UE4明确阻挡False被附加不可见占用覆盖None：三值合取保留NO，原本可放行但未知占用仍UNKNOWN；4新增用例旧版2失败。新teacher v23/producer plan v38/registry v33保留原阈值、提示词及未认证状态，旧合同不改。57路线没有命中该分支，10662原始两模式问题事实/答案/实例结束精确无变化，不宣称修正了真实错标。1382回归通过2跳过171.84秒，teacher60包含其中。B2/world1和4各模式每事件1024CPU计划通过，同world两模式同序、跨world不同，最多38次重复不是补支持。
UE5 proceed YES仍0；极暗InvadingTurn Town12_1450_1 f51-57局部亮裁剪通过但学生难辨车体，作为可观察性限制，非几何错标确证。不宣称十事件全条件/全源视觉通过。本地其它关键YES缺格不外推cr02全池（UE2return2、UE3proceed277、UE6proceed708、UE7complete13）；本地STOP带完成回执proceedYES10。新合同不能给旧服务器池/缓存/view改SHA复用，先确定全池影响再安排更新，不让用户立即20小时重建。未GPU/远程/push；完整产物checkpoints/label_rule_closure_20261017，报告audit_joint/LOCAL_LABEL_CLOSURE_20261017.md及轻量local_label_closure_20261017.json。保留用户.vscode及其它任务文件。

### 2026-10-17 cr02完整准入补包通过

收到1476055字节13成员新fullflow包，SHA/CRC通过；14.97MB automatic_admission与上轮缺失记录SHA一致，v83打包修复服务器已验证。构建558源仍为v78历史，collector当前v83；不能把历史源码回执当565项最新全机检查。8626输入身份及data4/view/reference摘要关联一致，原始全池未在本地重开。cr02训练298770纯自动规则题；两模式报告计划B1下十事件各1024、固定val各37/test各18，通过逐字段核对。实际推荐B2/四卡/累积4由正式入口重新准入重算，不称本包已有B2实际消费。最终关键YES为UE2return2/2组、UE3proceed277/95、UE5proceed0、UE6proceed708/97、UE7complete13/13、RE5proceed759/182。多数评测事件仅1物理组，RGB2/4评测成员不同；不强求YES:NO等量、不宣布独立准确率或联合G0。116相关回归14.96秒通过，无新生产代码/标签/提示词修改、无新RGB目视、未训练或push。可推进cr02限定范围首轮，不重建库；用户随后明确确认cr02只有本次题库，尚未训练且没有路线缓存：按首次训练新建本机路线缓存，不寻找或引用cee权重/缓存。命令audit_joint/CR02_ADMISSION_ACCEPTANCE_20261017.md；冷缓存+v78view升级仍有CPU耗时和900秒上限，不承诺首轮耗时，不自动重试。预算仅30.30GiB空闲/26.20GiB声明需求，保留20GiB守护。交新P4_OUT/handoff两epoch文件；cee现有训练及Phase3不用重跑。


### 2026-10-17 双服务器约定与cr02标定包接收

用户明确说明有两台独立服务器cr02和cee，数据、权重、缓存、运行目录不互通，也不保证每次在哪台执行。后续每份报告和命令必须先确定服务器，使用该机AutoMoT工作目录和已核验资产；不得把cee的checkpoint/view/cache路径或成功结论直接套给cr02，反之亦然。源码可用既有sync_automot.sh流程分别同步，数据不会随源码同步。AUTOMOT_SERVER=cr02/cee作为用户声明标识，配合记录真实hostname和本机绝对路径；不能凭容器hostname猜机器、不能自动复制大数据或重建缺失资产。跨机器搬迁已有绝对路径绑定产物需显式重新核验，不能改SHA绕过。
cr02本次已收fullflow_report.zip 46888字节12成员，558源对应v78；原目录checkpoints/phase4_labels_20261017_125230_304771113，日志仓库根/home/cruser1/lda/AutoMoT。回放8614路线1062401帧、train298770，自动留出val27801/432路线、test26105/452路线，build_result=admitted_existing_data_development/finished_cpu/exit0。打包漏14969873字节automatic_admission.json（旧10MB输入限）导致外层blocked_or_interrupted；不是标定建库失败，不重跑。准入完整明细未随包到达，不能独立宣布cr02两模式训练/val/test1:1最终核验；先补交该JSON即可，不需任何GPU/全库任务。UE5 proceed YES仍0、raw UE2return YES2、UE7complete YES18/13组、STOPproceed YES868/187组，raw支持非最终采样。cr02与cee训练池SHA/题数不同（298770 vs298787），不视为同一数据副本。
cee当前已知run为checkpoints/phase4_restart_20261017_162511，RGB2训练/val/reload已通过，test失败、RGB4尚未收到结果；已删除的125707目录不能引用。cee续接不适用于cr02新标定目录；Phase3旧验收也必须区分服务器资产是否存在。

本轮本地修复fullflow打包器：准入JSON限64MB、未压缩合计128MB、实际ZIP仍30MB；新增只打包--handoff-output保留原回执/数据，并记录collection/执行机器上下文；首轮/续接/多轮workflow也记录本机路径。137相关13.97秒，最终23专项5.20秒（22重叠），v83清单565源。完整说明audit_joint/CR02_FULLFLOW_REVIEW_20261017.md，当前仅需cr02补交现有automatic_admission.json，无需先同步或重算。本轮未push、未训练、未新增RGB目视；未认证缺失准入明细。

用户随后明确授权本轮白名单push：精确发布v83打包修复、机器上下文、测试和轻量回执；保留其它任务改动。发布同步后cr02仅执行package --handoff-output补打既有报告，交fullflow_report.zip与fullflow_package.json，不重跑标定、建库或训练；cee不套用cr02目录。

### 2026-10-17 v81已收RGB2重载通过，test快照读取边界及接续

旧125707目录及权重用户已删除；当前必须保留phase4_restart_20261017_162511。新包3270100字节30成员SHA/CRC、563源通过；RGB2四rank实际10240呈现=十事件各1024，val各37，316/370/YES18of66，与前轮统计相同；训练4416.45秒/val32.85秒，reload384.96秒370题及raw一致，资产/optimizer绑定通过服务器回执。test在rgb2.receipt.json哈希报属性变化、RGB4未开始；不是训练失败，严禁再重训当前RGB2或重跑已通过reload。旧torchrun --run问题真实关闭。
哈希改打开文件后fstat前后+路径身份+字节数检查，异常输出device/inode/size/mtime/ctime；不删除检查、不自动重试。open元数据刷新合成反例通过，读中写动/替换/链接改指拒绝；日志缺前后stat，服务器根因未确证，不能声称并发写或挂载刷新已实证。接续核对成功reload/案例/当前adapter及optimizer后先CPU快照检查→RGB2test→RGB4首轮/重载/test，缺快照拒绝隐式重建，失败诊断入handoff_continued。学生/teacher/提示词/采样/两缓存适配器字节未改，现有权重/view/快照复用。186相关回归14.73秒含真实四CPU与torchrun测试，首批42重叠；非服务器GPU验收。v81清单564源。命令audit_joint/PHASE4_TEST_SNAPSHOT_20261017.md，轻量snapshot_read_verification_20261017.json。没有新增RGB目视或修改标定阈值，不为此重新建库/回放/P3/readiness。UE4/UE6低YES召回及UE5无proceed正例仍待两模式效果对照，不能宣布全能力/联合G0通过。沿当前故障修复发布授权精确白名单同步，保留.vscode/paper/checkpoints等其它任务。

### 2026-10-17 v80 RGB2已训练、重载启动歧义与免重训接续

收到3156117字节27成员v79包及相同副本，只计一次；ZIP/成员SHA/CRC、558源及运行学生合同通过。view升级525.23秒/转换146.19秒成功；RGB2四H20每卡B2/累积4，10240实际呈现逐步匹配计划（十事件各1024）、370固定val各37/四卡94,92,92,92无重复；训练4375.21秒/val33.41秒。规则一致316/370=85.41%，YES18/66=27.27%；UE4 YES0/29但训练proceed YES呈现262，不能都归因缺正例。UE1/2/3/RE5 val单NO且各1物理组；不把100%当全能力。test18/event计划未执行，RGB4未开始，禁止重训RGB2或直接续7轮。
重载torchrun解析--run为--run-path歧义退出2，真实解析器复现。外围python_dispatch.sh/py显式插入--，首轮/多轮/新phase4_continue接入，学生/teacher/合同/已升级view/准备快照/v69缓存字节保持，可直接复用当前权重；直接旧CUDA CLI/旧evaluate.sh需经适配入口，不改SHA。phase4_continue验证已完成RGB2消费及合同后reload→test→RGB4首轮/reload/test，保存旧workflow及handoff，首错停止/已有输出拒绝；新包handoff_continued。打包分离128MB未压缩预算与30MB实际ZIP上限，保持逐题/步证据，超限不签成功；旧pilot默认不变。81回归7.87秒含两个真实CPU torchrun worker及压缩边界，非GPU重载验收。v80清单563源；本轮无新RGB目视/改标签提示词/远程/push。命令audit_joint/PHASE4_RGB2_EPOCH_CONTINUE_20261017.md，轻量epoch_continuation_verification_20261017.json。保留.vscode/paper及其它任务；下一步发布同步后只接续125707_138502059目录，不新增view/全池/benchmark任务。

本轮用户随后明确授权白名单发布。发布前仅在两条既有train横穿路线并行原生重放126帧、核验378源SHA，实际逐帧查看16RGB；16道U-E4 proceed题答案/布尔条件/ID一致，仅12处浮点末位最大1.42e-14，未回写生产。雾天及隐藏条件限制保留，无新确定标定/提示词缺陷，不作为服务器接续前置。轻量摘要在v80回执，RGB/重算留checkpoints/v80_release_rgb_20261017；保留用户.vscode/paper等其它任务，按精确白名单发布。

### 2026-10-17 v79视图重复转换超时修复

收到38303字节7成员包，ZIP/成员SHA/CRC及558源匹配v78。原生298787train完整校验完成、8614路线缓存全复用，paired RGB2完成后RGB4未完即900.263秒退出124；139未复现，两模式尚未训练，不再问日志或跑readiness。short_row保留完整原题digest但不拷贝将删除的证明；四图复用独立短行副本；paired_views一次转换两模式，upgrade单线程事务复用结果，仍核对双模式排除/成员/输入及最终SHA，异常恢复。原生teacher/标定/提示词/采样及phase4_load/production_cache字节不变，v69缓存保留。学生data.py合同变化，已知v78及此前view显式迁移，旧checkpoint原版续训，不改SHA绕门禁。
真实17路线1457train新旧逐字段一致，双转换1.788→0.358秒约4.99倍，完整迁移2.116秒；仅本地小库，非服务器速度承诺。205相关回归15.94秒+42采样/分布式评估4.67秒通过（早前45含于205）。v79清单558源；本轮无新RGB目视/标签/提示词/GPU。事件1:1逻辑不变，服务器本轮实际消费未验收，UE5/proceed/YES仍开发限制。命令audit_joint/PHASE4_VIEW_CONVERSION_20261017.md及view_conversion_verification_20261017.json。用户本轮明确授权白名单push，发布核验后同步再正式两模式首轮入口，不追加benchmark/老师重放/独立全池测试；保留用户.vscode/paper及其它任务。

本轮发布前74项回归8.83秒通过（与247项重叠）。另实际目视两已知train路线32–39各8帧共16RGB，原生从起点因果重放至39共80帧/240源SHA，重新建立停车历史；16道题布尔事实/答案不变，STOP28–30完成后35帧hazard撤销，36帧开始加速，无新改标依据。浮点末位366处最大4.55e-13、98题ID变化，仅诊断重算以1e-10容差比较，未回写生产/放宽SHA，不能声称旧数据逐字节恢复。轻量摘要随v79发布，原RGB和重算证据留本地。

### 2026-10-17 v78升级139可疑定时栈打印隔离

用户交回同PID1662801两层日志：8614绑定完成、298787题JSON62.13秒、协议25万次173.4秒，validate_model_inputs时30秒faulthandler输出截于半行File，随后139/总210.27秒。30秒Timeout是诊断输出非900秒任务超时，未GPU。形态类似CPython官方116008/131580，但缺C栈/解释器补丁信息，未确证或复现同一根因，不归因json.dumps/OOM/坏标签。
benchmark_runtime取消周期watchdog栈遍历，保留fatal诊断；父进程30秒输出PID/耗时/上限/存活（明确不代表完成），原超时/信号/进程清理保留。98相关回归19.65秒通过含真实CPU worker及清理，两新增反例先失败；非服务器修复验收。v78清单558源，仅诊断器/测试和三个指针变化；学生/原生/teacher、phase4_load/production_cache等合同字节不变，现有缓存/view不因补丁重建。发布后仅既定B2两模式单轮流程一次，新输出、复用原库/旧view升级/已完成v69缓存，不追加readiness/benchmark/全池/GPU测试；若再失败直接交新版含日志ZIP。不改标定提示词采样、不承诺耗时，不自动重试。用户随后明确授权本轮白名单push；实际逐帧复看两既有train路线各40–46共14RGB，42目视源SHA/54重算源SHA（去重60）通过，4题stable/re_yield事实一致，STOP历史沿用已绑定回放，非新全路线重放/盲审，无新改标签提示词依据，不作为服务器前置。命令audit_joint/PHASE4_WORKER_DIAGNOSTICS_20261017.md和worker_diagnostics_verification_20261017.json。保留用户.vscode/paper等其它任务。

### 2026-10-17 v77 Phase4视图升级失败交接补强

收到34888字节/4成员ZIP，SHA/CRC通过，558源码匹配v76、磁盘预算通过。新版workflow仅view_upgrade执行210.27秒退出139，两模式训练均未启动；47缺项是未生成证据，非47缺陷。通常139为SIGSEGV但缺原日志不能定根因，不判OOM/坏标签/磁盘或900秒超时。确认collector漏view_upgrade.log及子目录load.log/status，已加固定白名单60KB日志尾、SHA/漂移检查及可用status/result，缺训练记录改明确missing证据错误，保留阻塞态及30MB上限。两新反例先失败后修复，50相关回归通过；v77清单558matched。生产/学生合同、训练提示词标定及缓存适配器不改，不为此重新迁移。服务器只需交既有phase4_b2_epoch1_20261017_113332_626467217的view_upgrade.log、view_upgrade/load.log尾及status；不重跑建库/benchmark/GPU/P3。尚未取得崩溃栈，不能称全流程修好。报告audit_joint/PHASE4_UPGRADE_FAILURE_20261017.md和phase4_upgrade_handoff_verification_20261017.json；原包本地checkpoints/phase4_failed_upgrade_received_20261017。本轮未远程/GPU/push，保留.vscode/paper等其它任务。

### 2026-10-17 v76 Action缺依赖前置检查与v75四卡证据补录

用户复审两项：Action qwen_simple/bev_only实际依赖只读runner，本机缺失，不能删源码哈希或造占位实现。新增action_expert_ablation.source_check（纯标准库/CPU）输出外部runner/BEV工具存在性与SHA；launcher/直接train/eval/合同构建在选卡、输出、数据及大权重前汇总缺失项。保留真实内容绑定。合同单测用显式临时源码夹具，生产缺失/变更必须拒绝；旧优化器toy无事件标签，明确natural评估，与主线toy一致，生产十事件默认不变。94消融测试8.11秒、181相关测试14.74秒通过（含前组），不是Action真实运行通过。本机仍缺runner，已向用户询问当前原版来源；旧Git有c28935ab历史副本，未据此自动恢复或发布只读文件。
已收到服务器v75日志：CPU41项28.38秒、四H20-3e两/四图每卡B2、DDP反向+分片生成correctness_passed/exit0；loss一致，梯度最大相对误差0.008143/0.003575，四rank消费[2,2,2,2]。补four_gpu_received回执并将four_gpu_verification的未执行状态更新为服务端已执行，原本机测试记录保留。单真实样本重复/dropout0，不是全池/完整epoch/Phase3独立GPU验收；不重跑GPU或batch选型。
413项Phase3/4、共享batch/sequence及读取适配字节与v75一致，已有v75 view/缓存/checkpoint不需迁移；Action自身合同新增前置检查文件，旧Action checkpoint仍用原版。v76清单558源。命令audit_joint/ACTION_SOURCES_AND_GPU_RECEIPT_20261017.md：服务器仅python -S -m qwen3vl_local.action_expert_ablation.source_check，纯CPU缺文件exit2。代码/测试在既有目录白名单内，沿持续授权精确发布；不上传外部runner、聊天原文、权重或其它任务产物，保留.vscode/paper/checkpoints。

### 2026-10-17 v75用户要求正式全流程四卡与普通默认参数

用户明确要求Phase3/4训练、验证、重载和test全四卡。已选训练B2、P3累积2/P4累积4及sequence采样不变；评估B2/冻结RGB复用改普通默认，无需特殊export。P3 loss改整批分片、按样本聚合；生成val/final由全部rank参与；独立eval整批四卡分片。P4每轮val全rank，新增evaluate.sh四卡reload/test，直接CUDA CLI同样自动启动四卡；CPU准备/明确CPU诊断仍CPU。训练是真DDP梯度同步，推理为四卡独立生成后汇总，不混称推理需要DDP反向。完整batch分片、尾批不重复、成员一致与一次消费检查、某rank推理失败全rank拒绝，rank0单写结果。
277相关回归16.80秒通过，含真实四CPU/Gloo两epoch Phase4 DDP训练和各rank验证、P3/P4生成、尾批/错误传播及P3 B1/B2非等长loss一致；追加已知v72/v73/v74迁移及缺失绑定拒绝8项3.78秒（与主组有重叠）。实际限时CPU CLI41项5.37秒通过（重叠），不是4B GPU验收。服务器已有v74单H20两模式B2生成/loss/梯度通过，仅收到日志；本轮不冒称四卡新路径已实测。短命令audit_joint/FULL_FOUR_GPU_20261017.md：CPU180秒后GPU四卡240秒，无整库/view/容量矩阵，首错停止。
本轮147个原生合同/资产/依赖及读取适配SHA未变，Qwen3.5已跟踪后端文件亦逐字与HEAD一致；v69路线缓存保留。v75清单551源；新增evaluate.sh进入学生合同，已知v74运行差异可显式迁移新view，旧checkpoint仍需原版源码续训，不绕过SHA。新tests/test_distributed_evaluation.py按本轮全四卡需求纳入文件白名单；沿持续授权精确发布源码/测试/轻量文档回执，保留.vscode/paper/checkpoints及其它任务。

### 2026-10-17 v74可选评估batch与冻结视觉复用

承接用户继续完善/push/短测试要求，新增共享batch_runtime和当前P3/P4评估batch路径：左padding、保留成员/顺序/尾批；P3保持原EOS和1.05重复惩罚，pad不入惩罚。冻结Qwen3.5视觉塔可在单次前向按实际预处理像素+grid去重，再展开原图槽；不跨rank/step缓存，视觉可训练/随机层/输入需梯度/多时刻grid/额外输出走原路径。训练已选B2、累积2/4和采样v4不变；新优化显式启用，默认评估B1、复用关，不冒称目标4B已验收或更快。P4 reload/test继承训练配置，多轮恢复预检拒绝开关漂移。
新增audit_joint.batch_runtime_check：CPU180秒专项，通过后单卡GPU240秒检查两/四图重复单真实样本B2生成及loss/梯度；dropout0，无全池读取/view构建/权重保存/容量搜索，首错停止。命令audit_joint/BATCH_RUNTIME_20261017.md。GPU尚未执行，不能称DDP/全池/效果验收。
187专项回归18.64秒通过；后端39项独立进程2.83秒通过；实际CPU CLI36项2.95秒/exit0（与前两组重叠，不相加）。最初混合后端测试后10个读取测试触发后台线程隔离守卫，拆为独立进程通过，未放宽生产隔离。224个冻结producer/后端/适配文件字节未变，v69路线缓存保留；v74清单549源matched。
当前学生源码及P3恢复绑定改变：新实验view需显式迁移，已知v72/v73纳入允许列表，仍全父库/成员/sidecar/曝光核验；短测试不需view迁移。不将旧checkpoint跨合同恢复，既有训练留原版续训；不重建原生库/重放老师。共享batch_runtime.py和tests/test_batch_runtime.py按用户本轮授权新增白名单，审计代码/测试/轻量回执按既有目录白名单发布，保留.vscode/paper/checkpoints等其它任务。

### 2026-10-17 v73用户确认续训与多轮入口范围

用户异步明确选择本轮先完成Phase3 epoch边界续训和Phase4多轮/续训入口，评估batch和视觉编码去重暂缓；每卡B2、P3累积2/P4累积4、输入时间窗口与采样v4不变。P3已选入口默认SAVE_TRAINING_STATE=1，完整epoch/optimizer边界保存训练参数、AdamW、scheduler、各rank Python/Torch RNG、实际曝光history及checkpoint本地best副本；临时目录完整SHA回执后发布，再更新latest_training_state。恢复至新run，绑定源码/索引/配置/基座stat及元数据SHA/world，原定NUM_EPOCHS与LR计划不变；仅max_steps0且每epoch更新整除。STOP_AFTER_EPOCH保留完整计划；旧仅adapter不可恢复，中断未保存epoch须重跑。
P4新增audit_joint.phase4_training，通过既有train_phase4_batch.sh的EPOCHS及RESUME_RGB2/4接入；EPOCHS绝对目标，两模式分别指定恢复点/起始epoch。两模式CPU恢复预检均通过后才GPU，保留正式训练/逐轮val/最新epoch重载及固定test，预算扩大到全部目标epoch；新training_workflow.json记录阶段，失败即停/信号只清本次进程。无参数仍原首轮入口；多轮不误用只支持epoch000的旧ZIP收集器。可显式复用P4_VALIDATED_DATA。
419相关CPU回归27.89秒通过：实际P3小模型连续3轮与1+恢复2轮参数/optimizer/scheduler/RNG/history与后续batch日志一致；P4两模式相同分拆下参数/optimizer/history/样本一致并继承best。同步旧消费测试的cache/batch模拟接口。Phase4学生/producer及路线适配器148项绑定字节未变，v72有效view/快照、v69路线缓存保留；新v73清单545源matched。没有目标4B/四卡GPU恢复或服务器完整训练验收，不保证GPU逐位确定性。完整命令audit_joint/EPOCH_RECOVERY_20261017.md，轻量epoch_recovery_verification_20261017.json；沿已授权白名单发布，保留.vscode/paper/checkpoints和其它任务。

### 2026-10-17 v72段内action边际补偿与开发曝光缓存绑定

两项复审缺陷先复现（两步长action及两RGB缓存共4项旧代码失败），现已修正。共享训练采样v4按段内provisional计数累计实际归一化欠额减少，达到目标后不加分；保留覆盖/连续性/尾部拼接/同锚点机会优先。四个8题段反例20seed×两步长均8A8B、两段8、16新题，次epoch覆盖剩余16；仍非全局最优。旧v1/v2/v3 history拒绝混用。
运行快照v2每次准备和复用前后动态调用正式exposure_inventory，比较view完整文件SHA清单并绑定物理group集合，检测新登记、删改及读中变化；直接/reviewed缓存同样拒绝曝光漂移，不推断已有泄漏。phase4_view_upgrade加入已知v71源码，保留v61/v70，仍全父库/两模式成员/sidecar核验且曝光变化拒绝，不覆盖旧view或用于旧checkpoint恢复。旧快照需新输出准备；phase4_load/production_cache和原生teacher源码不改，v69路线缓存保留。
327相关CPU回归26.23秒通过，含真实reviewed小库v71合同fixture升级、两模式数据一致、四CPU worker禁用native验证仍读成功。v72源码清单541源；每卡B2、P3累积2/P4累积4及时间窗口不变，无新benchmark/GPU/服务器全池验收；P3跨进程续训/批量评估生成/共享视觉编码去重边界不变。说明audit_joint/ACTION_EXPOSURE_FIXES_20261017.md，沿已授权白名单发布，保留用户.vscode/paper/checkpoints及其它任务。

### 2026-10-16 v71评估连续性、重复上限与正式全池加载闭环

用户复审三项已修正，每卡B2决定不变，不补benchmark。评估共享选择器升v2：跨路线最长实际stride连续锚点优先、同帧多题作候选不制造断点、标签盲选无放回十事件等量；两/四图成员可不同，正式准入保留训练成员相等，分别记录评估SHA，不冒称逐题配对。连续训练max_question_repeat明确每题每epoch硬限，容量不足前置拒绝，绑定history且改变/取消上限拒绝混用；默认None不限，回执区分配置上限与实际重复，保留既有少见/连续/action优先。
新增rgb_short.runtime_data：正式run.sh在torchrun前单CPU完整准备、run_automatic准入与两模式首轮/重载/test共享派生JSON。每模式native.load_dataset一次，四rank只读已验证学生快照，不再独立校验原生老师证明；直接DDP无准备目录拒绝。源码及适配器SHA、源数据/view/生产路径stat（device/inode/size/mtime/ctime）和准备JSON SHA读前后绑定，失败不发布回执、20GiB磁盘余量；可信本机派生缓存，非外部认证。原参考包与跨split检查不跳过，每rank仍读学生数据/做评估准备，不宣称全I/O消失。
学生合同改变，新增phase4_view_upgrade仅接受v61/v70已知运行/采样差异，完整父库及两模式成员/排除/sidecar核验后写新view；phase4_first_epoch可自动先升级已知旧view，不覆盖旧view、不恢复旧checkpoint。固定B2入口要求P4_PRODUCTION_CACHE；phase4_load和production_cache源码字节保持，既有v69路线缓存可用，不重建原生库或重放老师。命令audit_joint/SAMPLING_RUNTIME_FIXES_20261016.md，原生/teacher合同字节未改。
315相关回归21.97秒通过，含4真实CPU worker禁用native loader仍读一致、连续反例、上限/取消绑定、坏缓存/变更/低磁盘、shell和升级接线。真实17路线1457train以v70合同fixture升级2.60秒，两模式准备1.61/1.63秒，各四次读取一致，共2次native加载而非8次，仅本地小池；v71清单541源matched。无新服务器4B/GPU/整轮，生成评估逐题和独立样本重复图编码范围不变。沿已授权白名单发布，保留.vscode/paper/checkpoints等其它任务。

### 2026-10-16 v70用户确定两／四图每卡B2，选型关闭

用户明确接受Phase4两图、四图每卡B2，不需要再补任何batch测试。Phase3沿已测B2/累积2/四卡16样本更新；Phase4固定B2/累积4/四卡32样本更新；均开启GC。不得再以B4比较、全库准入、路线校验或view迁移作为本轮batch选型前置。已有P4小池8/update单重复支持运行选择，不宣称正式32/update新实测或全局最优。服务器另已交回8614路线全复用、完整原生native_validated/exit0/298787train（val/test0，独立references4另用），学生view迁移尚未收到成功回执，不编造有效新路径。
phase4_first_epoch函数/CLI默认改B2，两模式实际env均B2/累积4并入workflow；新增固定train_phase4_batch.sh要求已有有效view，冲突环境拒绝、按既有两模式首轮/重载/test执行，无隐式benchmark或view构建。底层rgb_short通用入口保留历史默认并由推荐入口显式传参，避免改变合同字节；sequence_batch、原生/学生合同、phase4_load/production_cache均不改，已有有效view与v69缓存不因这次发布失效。历史操作页加最终决策入口，操作见audit_joint/BATCH_SELECTION_FINAL_20261016.md。
108相关CPU回归6.95秒通过，覆盖真实shell参数流、两模式默认、旧环境冲突、训练batch回执、连续段/未见优先及Action等量评估；无新增服务器GPU、全库、迁移或训练。v70_selected_b2清单537源，按既有授权白名单发布；保留.vscode/paper/checkpoints及其它任务。后续只在用户要求正式训练时用已选B2入口，不再追加性能验收待办。

### 2026-10-16 v69原生路线校验续接

服务器v68完整读取240秒退出124，无result；JSON读取62.96秒、25万协议复用184.1秒、model_inputs校验15.08秒完成，后续visible_scope和生产路线accounting尚无成功回执，不解释为GPU失效或数据已坏。新增同次读取question内容SHA+teacher身份核对后复用首次完整证明校验（上限30万缓存项，超限仍原验证）；增加visible_scope/risk/candidate阶段日志。
新增audit_joint.phase4_production_cache：不读train大JSON，直接对已有生产路线调用原inspect_route逐条全校验后原子保存本地回执。绑定老师源码/适配器/父库manifest/pool/index/路线内容和物理路径，复用重新SHA核对，源码输入漂移/坏回执拒绝；每条通过即保存，真实超时后保留完成项，独占锁随进程释放。默认240秒、无自动循环；只报production_routes_validated/full_pool_admitted=false，不是训练准入或签名认证。phase4_load及显式旧view迁移支持--production-cache，未自动替换所有正式训练入口，不建库/老师回放、不改冻结生产或学生合同。
85相关回归11.23秒通过，含真实子进程超时后续接/篡改/恢复；17路线1457train真实原生返回一致，路线首查1.13秒/复用0.019秒、完整适配读取1.64→0.56秒，仅本地小池；两个真实CPU CLI通过。v69清单536源，沿已授权故障修复白名单发布。服务器下一步只跑一次240秒路线缓存命令，失败不自动排队；进度有新增且124时可保持cache目录手动续接，成功后完整读取/视图仍需独立验收。命令audit_joint/PHASE4_ROUTE_CACHE_20261016.md；保留.vscode/paper/checkpoints及其它任务。

### 2026-10-16 v68完整池读取老师源码SHA复用

服务器两/四图B1/B2小池全部measured/headroom通过，B2吞吐+7.89%/+6.20%，正式32/update及完整池尚未验收，不再重复smoke。只读审计发现visible_scope经validate_question未传expected_teacher_sha，每次rules.identity重复46个源码SHA。phase4_load隔离读取事务内复用teacher_rules.file_sha的已知源，未列路径实时读取；结束恢复原函数重读完整身份/参数，变化拒绝，每题完整校验保持，不持久缓存、不改冻结producer/teacher/学生合同。原生read_rows/model_input/coverage/admission增加阶段计时；正式训练原入口未自动替换。
56相关回归8.86秒通过；真实1457train原生返回与原始/协议适配两路径完全一致，错误question证明拒绝；协议适配1.95秒→加teacher复用1.69秒，仅本地小池，不外推服务器。源码v68_teacher_snapshot共534源。沿已授权故障修复白名单发布；下一步仅phase4_load --native --timeout240一次完整CPU校验，成功native_validated再推进view迁移/正式入口，超时交日志不自动重跑。命令audit_joint/PHASE4_FULL_LOAD_NEXT_20261016.md；保留.vscode/paper/checkpoints和其它任务。

### 2026-10-16 v67 Phase4限时十事件小池对照

服务器交回H20-3e四卡两/四图均model_step_passed/exit0：B1 allocated12.43/15.00GiB，最大reserved18.23/23.40GiB，仅同题单步。新增phase4_quick_benchmark，原生大文件32分散连续窗口、最多20000行/256MiB、32候选/event；选中图身份/正式两四图转换/候选内冲突排除，十事件各至少8独立题，不足CPU拒绝。默认CPU60秒、B1/B2各180秒，首错停；十事件各8题计时+各8预热，每update8、B1累积2/B2累积1，成员/update固定，正式encode_batch及既有DDP benchmark消费核验，不存权重。准备文件可显式复用、源码/输入stat/成员SHA绑定；只报provisional_subset_candidates及full_pool_admitted=false，不伪称全train SHA/老师证据/跨split准入、正式32/update不改。
70相关回归11.38秒通过；真实1457题小池U-E5仅7题，默认8/event正确拒绝；降为4/event、world2的CPU准备两模式各40独立题通过，仅验证接线非四卡GPU消费。v67清单534源，命令audit_joint/PHASE4_QUICK_BENCHMARK_20261016.md，先RGB2两组约7分钟预算。完整池加载/迁移性能仍未解决，不重建原生库/view/老师回放，沿现有故障修复白名单发布，保留.vscode/paper及其它任务文件。

### 2026-10-16 v66 GPU诊断软链接路径修正

服务器交回四rank transport_passed，CUDA矩阵乘/NCCL数值核验通过；model在读取首题RGB时被诊断脚本物理路径越界检查拒绝，尚未加载4B。修正phase4_gpu_smoke复用正式encoder的data_path，接受数据树内指向外盘的route/asset软链接，仍拒绝绝对路径/点段穿越并核对图SHA。34相关CPU回归0.66秒通过，覆盖两模式外盘route/单文件链接、篡改拒绝及路径穿越；非GPU模型验收。v66清单532源；学生/原生合同不变，无需重建库或view。按既有故障修复白名单发布，服务器同步后仅重跑model RGB2（180秒），不重跑transport或12组矩阵。具体命令沿PHASE4_GPU_SMOKE_20261016.md。

### 2026-10-16 v65独立GPU运行诊断

v64服务器180秒迁移退出124，160.6秒协议复用20万次、逐题老师证据校验仍进行，全池及view未完成。新增audit_joint.phase4_gpu_smoke避免整库成为硬件诊断前置：transport限时60秒CUDA矩阵乘+NCCL数值核验；model限时180秒读取第一条原生train YES/NO及所需2/4图、源码/图SHA检查，正式转换/encode_batch、GC/LoRA16/32、四卡一次前反向/AdamW。各rank重复同一题，仅运行诊断，不是事件均衡训练；不保存权重，summary固定full_pool_admitted=false，不当batch排名/质量准入。正式合同/全池规则不改，未修复所有整库耗时。
38相关测试11.90秒通过；本地真实首题两图[32,34]、四图[31,32,33,34]输入各约0.07/0.02秒，仅CPU图检查，4B/CUDA未验收。v65清单532源，命令PHASE4_GPU_SMOKE_20261016.md，轻量phase4_gpu_smoke_verification_20261016.json。沿当前Phase4故障修复及白名单发布流程交付，只发布代码/测试/轻量文档回执，保留.vscode/paper/checkpoints和其它任务文件。先transport成功再model两图，失败交worker.log，不跑12组/整轮或重建原生库。

### 2026-10-16 v64原生读取协议复用与旧view显式迁移

服务器30/60秒在JSONL解析、90秒进入逐题冻结协议SHA；每题55文件约1.34MB、298787题约1643万哈希调用，不等于物理磁盘字节或全部耗时。新增audit_joint.phase4_load隔离单线程读取适配层，原生函数开始/结束真实验证、计划SHA前后复核、每题返回归属副本；期间变更拒绝，异常恢复，不跨调用缓存。benchmark准备接入；原生producer/teacher/dataset/identity及学生合同源码字节不变，正式训练器仍用原路径，不声称全入口加速。
旧服务器view四SHA均v61，显式迁移只接受已知sequence_batch/evaluate/reload_check/run.sh差异，其余合同完全相同；完整原生校验、两RGB成员/排除项/配额复算后新目录写当前view和迁移来源记录，RGB sidecar前后SHA核验后复用，不覆盖旧view/checkpoint、不重放老师/建原生库，不将迁移用于旧checkpoint恢复或声称新RGB认证。真实1457train小池以v61运行SHA构造历史描述符fixture迁移，两模式重载ID/标签配对；原生新旧返回完全一致，本地4.07→1.91秒仅小池。123回归15.52秒通过，含真实冻结文件中途变更拒绝、未知迁移源拒绝及清理；当前v64_read_protocol清单530源。用户授权本轮白名单push，命令PHASE4_LOAD_V64_20261016.md：先CPU迁移180秒，再两图B1单trial准备180秒/GPU300秒，失败交日志不继续排队。服务器整库/GPU待验收，保留.vscode、paper、checkpoints及其它任务产物。

### 2026-10-16 v62／v63白名单授权发布

用户明确要求push本轮连续段尾部、可见GPU、分模式batch及限时benchmark修补。发布v63_bounded_benchmark，528源码SHA matched；118相关回归通过（13.69秒），承接v62既有261项验证，不叠加为独立总数。提交本轮源码、测试、轻量回执和命令文档，保留用户.vscode、paper、checkpoints、权重、RGB及其它任务未跟踪文件。服务器先沿sync_automot.sh同步，按PHASE4_SHORT_DIAGNOSIS_20261016.md执行120秒CPU定位，通过后单trial CPU120秒／GPU300秒；有效v62 view继续复用，不重放老师或重建原生库。目标4B四卡尚未验收，本次发布不代表GPU结果通过。

### 2026-10-16 Phase4 benchmark限时定位与准备阶段拆分

服务器四组各1800秒超时，worker各约14GiB CPU内存、GPU仅上下文占用；日志无模型完成或训练进度，具体全池热点仍需调用栈，不把NCCL弃用warning当根因。用户已清理并确认1个torchrun与4个worker退出。新增benchmark_runtime：Phase4完整load/validate/schedule在单CPU子进程每模式一次，通过后仅将SHA绑定的选中小集合交给各rank／batch／repeat；正式encode_batch、事件配额和更新成员不变。新增prepare-only／prepare-timeout默认120秒，阶段日志实时输出及30秒调用栈，默认首错停止矩阵；显式continue-on-error才继续。按PID与启动时间跟踪并清理独立session后代，支持超时／Ctrl+C／SIGTERM，不清理其它任务。
本次不改学生／原生合同，既有有效v62 view可复用，不重新建库或回放老师。真实1457题小池CPU CLI通过，278不同题／320呈现；目标4B权重本机缺失，未GPU测试。命令见audit_joint/PHASE4_SHORT_DIAGNOSIS_20261016.md：旧代码也可直接120秒栈定位；同步补丁后单trial预算CPU120秒+GPU300秒，先两图B1再四图，避免重跑12组。118相关测试13.69秒通过，含两模式小集合与真实信号清理；当前v63_bounded_benchmark清单528源matched，回执benchmark_bounded_verification_20261016.json；未push，保留.vscode及paper。

### 2026-10-16 Phase4 RGB2 首轮交回核验与原版本接续

Downloads 两文件：3740004字节ZIP/27成员SHA与CRC通过；四rank实际ID/顺序匹配10240计划，十事件各1024、10036独立题/298787池（3.36%），val实际各37共370、347正确（规则一致率93.78%），YES31/50、NO316/320；raw重解析、370题重载完整案例一致。RGB2固定test和RGB4结果缺，原workflow SIGINT130保留。R-E5 proceed YES0/3（训练暴露158）、R-E2 YES0/2；val无UE2 return/UE7 complete，UE5 proceed无YES，不以总体高分宣称全能力。训练1h30m55、内置val179s、独立reload2h49m、view2h58m；后两者无细分计时，不归因全部GPU生成。
启动523源匹配v60，运行15学生源匹配v61采样v2，sequence_batch启动/运行SHA不同；不能称v62实验或完整同版启动快照。保持原合同继续RGB2 test→RGB4单轮/重载/test，先check_contract，若已同步v62则停止并恢复独立原版本环境，不能改SHA。33相关回归6.15秒及文档命令语法通过，无生产修改/新RGB目视/训练/远程/push。既有v62工作树及用户.vscode保留。报告audit_joint/PHASE4_RGB2_RECEIVED_20261016.md、轻量phase4_rgb2_received_verification_20261016.json；全证据checkpoints/phase4_rgb2_received_20261016。无需重建库/老师回放/重做P3/RGB2训练或reload，不要求YESNO等额，联合G0仍未认证。

### 2026-10-15 v62复审收尾与Phase4实测交接

用户交回独立只读审计摘要：205专项、150随机池×4轮及真实1457题覆盖通过，前轮三项问题关闭，无新增确定缺陷；未接收该轮原始日志，不与本地261项相加或声称本轮重跑。核对保留v62运行源码／采样v3合同，仅修正文档残留v2说明、补齐实测后复用同一view的分模式首轮命令，记录证据来源与待验收项。下一步发布同步v62→既有有效v57 data4生成新view→Phase4目标4B四卡实测→分别选batch→首轮／重载／test；不重做原生建库或老师回放，不用P3结果替代P4实测。尚未push／远程／GPU，保留用户.vscode。详见audit_joint/PHASE4_BENCHMARK_20261015.md及batch_tail_modes_verification_20261015.json。

### 2026-10-15 连续段尾部修正、可见GPU与分模式batch

用户复审三项均复现并修正。共享采样升v3：曝光层级不变，batch开头最长段，尾部在本层用完整段恰好补齐（最少段数、再长段），无完整组合才截长段；保留同锚点action软补偿，旧v1／v2 history拒绝。6/6/2/2→6+2、6+2。真实1457train、四卡B2、1024/event：两／四图可避免同action截断选择156／83→0／0，总段数4214→4004／3497→3266；各模式全覆盖1457题，16/event五轮累计151/278/383/479/575保持、U-E3仍14proceedNO+2holdYES。空留出仅CPU计划，非GPU消费／全局最优排布。
P3 batch入口与P4 benchmark复用原devices空闲筛选（显存<4096MiB、利用率<20%），尊重CUDA_VISIBLE_DEVICES索引／完整UUID，不足四卡／无查询即停止，无0..3回退；P4当前学生run/evaluate/reload也不清除mask。GPU_IDS显式pin优先仍可覆盖mask／空闲筛选；原生devices／teacher合同不改。phase4_first_epoch新增--batch-size-rgb2/--batch-size-rgb4，--batch-size作后备，每模式计算32样本/update累积8/B；实际配置写workflow.mode_configuration与阶段记录并进入交接，混合模式顶层batch_size/accumulation=null不误报。
261相关回归17.01秒通过，含真实shell+selector模拟忙卡／单可见卡／UUID／显式pin及混合模式环境／交接、枚举组合检查；七shell和文档命令语法、CLI help通过。新v62_batch_tail_modes清单526源matched无额外Python，三个发布指针更新。原生producer和题目成员SHA不变；共享采样及当前学生入口源码改变合同，最终同步后从有效v57 data4新建view，不重放老师／20h建库、不改旧SHA续训。详见audit_joint/BATCH_TAIL_MODES_20261015.md及batch_tail_modes_verification_20261015.json；本机checkpoints/batch_tail_modes_20261015。未push／远程／GPU，保留用户.vscode。

### 2026-10-15 v61复审补丁与Phase4实测授权发布

用户明确要求完善代码push并给出Phase4实测完整命令。本轮发布此前v61四项修正（通用P3 warmup比例、v2同锚点action选择、probe配置可用性交集、旧回执测试），另将Phase4 benchmark默认扩为两／四图B1／2／4两重复共12trial，有效更新32、累积8／4／2、正式encode_batch路径。完整命令在audit_joint/PHASE4_BENCHMARK_20261015.md：现有v57 data4→新学生view→四卡实测，包含来源校验／磁盘guard／失败停止／日志与交回文件，不启动正式整轮、不重放老师。未收到Phase4 GPU实测结果，不能用Phase3结果替代。
发布前224相关回归18.68秒通过，六shell及完整命令语法／内联Python编译通过，v61清单525源matched；三个发布指针一致。源码测试轻量文档／回执白名单发布，checkpoints、RGB、权重、paper、用户.vscode留本机。同步最终源码后生成view；待用户执行服务器实测并交summary.json与request.json。无远程／GPU执行。

### 2026-10-15 连续batch四项复审修正

对v60复核：P4 benchmark已用正式encode_batch，保留并回归；P3 B2专用入口已有warmup500，通用入口仍固定2000，现默认按实际optimizer总更新10%预热，显式步数必须小于总更新、模型加载前拒绝错误，运行清单及adapter记录计划；B2专用500保留。共享采样升event_equal_contiguous_unseen_first_v2，保留同锚点全部合格候选，曝光层级／最长段优先后按action欠额与剩余机会选题，旧v1 history拒绝。probe用各卡实际安全batch交集，分别报告容量及training_compatible_candidates／排除原因，检查四卡1024/event和有效更新量；不把B3／5／6／7容量当正式入口推荐。修复旧short_diagnostics对train.encode和单id字段引用。
224相关回归15.77秒通过，更新发布指针后16子集再通过（不另计），六shell语法及P3直接脚本help通过。既有1457train小池新view，两模式U-E3首轮16proceed/NO→14proceed/NO+2hold/YES，同锚点／段成员长度，2RGB中间两段顺序交换；五轮累计151／278／383／479／575保持。两模式1024/event计划覆盖全部1457题，空留出仅CPU非GPU消费；原生producer及题目成员SHA不变，旧学生合同拒绝。新release_sources_v61_batch_audit_20261015.json共525源matched无额外Python，三个入口指针更新。共享采样改变学生合同，最终同步后从有效v57 data4新建view，不重放老师／20h建库，不改旧SHA续训。详见audit_joint/BATCH_AUDIT_FIXES_20261015.md和batch_audit_fixes_verification_20261015.json；本机checkpoints/batch_audit_fixes_20261015。未push／远程／GPU／改标定提示词，保留用户.vscode。

### 2026-10-15 Phase4首轮工作流与batch配置授权发布

用户明确授权本轮白名单push。当前统一release_sources_v60_batch_profiles_20261015.json，523源SHA全部匹配，发布前相关回归209项通过（13.49秒），含自动准入/失败交接/真实CPU进程清理/连续batch与性能配置。提交源码、测试、轻量报告与操作文档，不提交checkpoints、RGB、权重、压缩包、paper或用户.vscode。Phase4沿已有sync_automot.sh同步后使用audit_joint.phase4_first_epoch，复用v57原生库生成当前视图并完成两模式各一轮/重载/test/交接；GPU结果未验收，UE5能力缺格仍保留。性能profile与工作流变更已合并检查，最终同步后再生成学生视图。

### 2026-10-15 Phase3四卡B2实测入口与Phase4固定更新量衔接

用户交回Phase3四H20 end_to_end八trial，均measured/headroom通过；同160呈现／每更新16，两图B2吞吐较B1+9.53%、四图+8.60%，最大reserved29.47／40.55GiB。仅依据交回摘要，未独立接收rank文件，不推断质量。新增audit_joint/train_phase3_batch.sh，B2／累积2／checkpointing，RGB2[-2,0]步长2、RGB4[-3,-2,-1,0]步长1；四卡固定，旧环境冲突拒绝，新run子目录防覆盖。warmup500更新、eval1000微步按旧默认呈现数换算，显式预算覆盖仍需核对。旧通用train.sh默认未偷偷改，不再要求Phase3重复probe。
Phase4 benchmark改用正式encode_batch设备路径，分段明确encode_batch_device；新增benchmark_phase4_batch.sh默认B1／B2、32/event、有效更新32、累积8／4、end_to_end。正式run_automatic以UPDATE_SAMPLES=32计算累积，冲突在输出前拒绝；尚无Phase4四卡性能结论，不沿用P3速度选型。run_automatic属学生合同，需最终同步后从已完成v57 data4生成一次当前view，不重做20h建库／老师。
100相关回归+16工作流回归通过，五shell语法通过；既有17路线1457train小库新view两模式ID／目标配对、四组320呈现计划同成员／更新组，旧view拒绝、原生producer与成员SHA不变，空留出仅CPU接线验证。新v60_batch_profiles清单523源，承接并保留既有P4验收／首轮工作流改动，三个发布指针一致；未push／远程／GPU，未改用户.vscode。命令见audit_joint/BATCH_TRAINING_20261015.md，轻量batch_profiles_verification_20261015.json；本机checkpoints/batch_profiles_20261015。

### 2026-10-15 Phase4单命令首轮编排与失败自动交接

新增audit_joint.phase4_first_epoch：从指定已完成v57 data4新建当前学生view，或显式复用同父库有效view；来源与预算先查，顺序两模式run_automatic准入/单轮/重载/test，共同RUN_TAG与输出，最终自动30MB内交接。默认自动选空闲四卡，GPU_IDS显式pin，B默认1；失败停后续，已有目录拒绝覆盖。每阶段日志/耗时/workflow.json；SIGINT/TERM通知本次进程组，等待内层guard清理独立worker组，不碰其它任务。交接器新增可选workflow/source_check/launch_budget及view_build日志尾，失败状态不可被完整采样记录掩盖，旧交接兼容。
并行修改整合后209相关回归12.29秒通过，含真实CPU子进程+guard信号清理；实CLI缺资产退出2、无GPU阶段、30050字节3成员ZIP/CRC/SHA通过。新523源v59_phase4_workflow matched，两CPU指针更新。只增编排与交接，本入口不改学生合同，但并行run_automatic固定32样本/update修改改变SHA，最终同步后生成学生view，原生/标定/提示词未改，无新回放/RGB/GPU/push。命令见audit_joint/PHASE4_V57_ACCEPTANCE_20261015.md，一条python -m qwen3vl_local.audit_joint.phase4_first_epoch，交phase4_epoch两文件；有效当前view无需再次生成。UE5缺格/联合G0边界不变。轻量phase4_workflow_verification_20261015.json，证据checkpoints/phase4_workflow_20261015；保留用户.vscode。

### 2026-10-15 Phase4 v57正式CPU包交回与当前视图准入收尾

Downloads/fullflow_report.zip 867630字节/12成员CRC与SHA通过，501源与原v57_handoff清单一致；8614路线1062401帧、298787规则train，旧视图train1024/event、val37/test18无放回，RE5 proceed760YES/182组。总20h30m25、回放7h16m43；CPU finished/exit0、未GPU。UE5 proceedYES仍0，仅existing_data_development豁免；UE2 return2YES/2组、UE7 complete13YES/13组稀少，不声称全能力/独立准确率/联合G0。
当前批采样measure仅约束已有事件，补expected_events十类硬拒绝；原生prepare/训练已有十类门，不声称正式训练曾漏检。修正旧测试patch错评测模块与错误原因被掩盖，156相关回归8.24秒通过。新增518源v59_phase4_admission清单matched，补旧清单遗漏的VISUAL_REVIEW_RUBRIC_V25.md，两CPU shell指新清单。原生生产合同未改；rule_baseline属学生视图绑定，补丁后需新学生视图，勿改旧SHA。以既有17路线v57小库仅重建视图，两模式1457train ID/标签配对、空留出，world1/4 B1各1024/event非GPU消费。未新老师回放/RGB目视/改阈值提示词，不再要求服务器P3/readiness/20h全库。
下一步同步本补丁→复用checkpoints/phase4_v57_20261014_232157_762998141/data4生成[-2,0]/[-3,-2,-1,0]学生视图→run_automatic检查正式当前池后两模式首轮/重载/固定test。实际命令及共同OUTPUT_DIR/RUN_TAG=first_epoch交接在audit_joint/PHASE4_V57_ACCEPTANCE_20261015.md；交phase4_epoch_report.zip+phase4_epoch_package.json。轻量phase4_v57_received_verification_20261015.json、全证据checkpoints/phase4_v57_received_20261015。本轮未push/远程/GPU，未改用户.vscode。

### 2026-10-15 固定样本四卡性能对照工具与授权发布

收到Phase3 H20探测：checkpointing开后两图B8预留66.31GiB满足15%余量，四图B8预留89.33GiB不满足；旧probe各B样本成员/数量不同，不作严格速度排名。按用户要求新增audit_joint/batch_benchmark.py及专项测试，支持P3/P4、同时torchrun四卡、固定十事件样本/每update16样本，B1/2/4相应累积4/2/1。预热每event1update与计时每event16呈现都等量，分开统计；默认两重复正反B顺序，逐rank成员/顺序/更新完成与余量全部通过才报最快中位候选。分段CPU准备/H2D/forward-loss/backward含DDP/clip-AdamW及显存快照，end_to_end不把异步提交耗时说成GPU耗时；失败/超时保留诊断并终止本组worker，无权重保存。P4工具为分段在CPU组批后传输，正式encoder传输未改，部署吞吐需确认该差异。
31相关回归通过，含真实两进程CPU/Gloo累积与整批更新一致、两阶段loss/梯度对照、缺rank/篡改/超时；真实v57既有1457train两模式六组计划同160呈现/151唯一/10更新及成员SHA，无GPU验收。生产/学生/训练器合同保持、有效v58视图无需重建；新release_sources_v59_fixed_batch_benchmark_20261015.json 517源，两P4 CPU shell指新清单。命令见audit_joint/SEQUENCE_BATCH_20261015.md固定样本段、轻量batch_benchmark_verification_20261015.json，本机checkpoints/batch_benchmark_20261015。用户随后明确授权本轮白名单push；未远程/GPU、未改.vscode。

### 2026-10-15 显存probe空变量错误修正与授权发布

服务器8trial均报PosixPath('.') has an empty name；本机check_index(Path(""))原生复现，INDEX空或为.在with_name定位manifest前失败，非OOM/模型前向。补probe argparse非空路径与启动前文件/目录检查，6专项通过，旧代码可先正确设置现有正式索引及data/model/prompt变量直接恢复，无需建库。新release_sources_v58_probe_inputs_20261015.json共515源，两个P4 CPU launcher指新清单；原v58清单、训练器/学生视图合同不改。用户随后明确授权本补丁按白名单commit/push，无GPU远程。详见audit_joint/SEQUENCE_BATCH_20261015.md空变量修正段。

### 2026-10-15 v58白名单发布

用户明确授权push本轮Phase3／4连续batch改动。精确发布源码、测试、操作文档和轻量回执；新增共享模块 `AutoMoT/qwen3vl_local/sequence_batch.py` 与专项 `AutoMoT/qwen3vl_local/tests/test_sequence_batch.py` 同步登记文件白名单。225相关回归已通过，515源SHA再次matched；目标4B显存仍待服务器probe，不因发布宣称GPU验收。用户既有.vscode修改、checkpoints/RGB/权重及其它未跟踪参考文件保留本机。当前入口与历史实验边界见audit_joint/SEQUENCE_BATCH_20261015.md。

### 2026-10-15 Phase3／4真实batch、连续锚点与等量评估

最终用户明确确认：两图[t-2,t]、锚点推进2；四图[t-3,t-2,t-1,t]、锚点推进1，覆盖此前中间四图[-6,-4,-2,0]提议。Phase3 train及Phase4 rgb_short.train（两四图共用）已接真实每卡batch单次前向，梯度累积另计；事件严格等量，少见题/锚点优先、连续段优先、action软补偿，同event短段拼接独立样本，transition历史可重叠但不跨split/路线造历史。跨epoch曝光落盘；P4 epoch边界resume绑定history，P3仅同次多epoch延续不声称优化器恢复。val/test新固定十事件等量无放回；Action prior/两类消融共享评估默认十事件等量，背景不入第11事件，联合分配不重复计总体分母。
原生P4 producer/teacher22/v57合同及旧legacy trainer保持，不为新学生输入重放老师；从有效v57 data4新建视图及RGB身份sidecar，四图标签标记transferred_reference，未人工认证新可见性。P4当前入口rgb_short两模式；旧server_phase3_epoch固定旧采样验收不可用于本新实验，按audit_joint/SEQUENCE_BATCH_20261015.md直接新训练和显存probe。两个P4 CPU shell指新release_sources_v58_sequence_batch_20261015.json，515SHA matched无额外Python，五shell语法通过，未push。
最终225相关回归通过含tiny实际Qwen CPU两/四图logits/loss/梯度与真实P4训练循环mock模型两epoch；前扩展两Action消融合同检查缺mot_lead_offline_runner.py未完成，不报全套。真实v57既有1457train两模式ID/标签一致，原父库SHA不变；4卡B2计划1024/event，另16/event五轮累计覆盖151/278/383/479/575，非GPU实际消费、空留出小池非正式准入。当前有效本机checkpoints/sequence_batch_20261015/real_view_final/view及verification。目标4B权重本机缺失，新增batch_probe逐GPU/2与4RGB/B1..8真实前反向AdamW显存命令，但未实测容量/四卡DDP；无新采集、远程训练、生产改标、push，未改用户.vscode。

### 2026-10-14 Phase3重新训练入口清单修复

用户在白名单发布后询问Phase3能否正常训练并索取命令。实际复现server_phase3_epoch仍绑定v54_existing_data，当前17成员漂移→mismatch，不能照抄旧命令。仅更新其RELEASE及两P4入口指针到新release_sources_v57_phase3_entry_20261014.json（501matched），无P3生产/标定/提示词/参数改动，真实已收首轮ZIP verified_completed_epoch仍verified。基础环境23入口测试通过，14失败/7错误均缺torch未完成，保留失败日志不报全套。用户要新实验可从基座重新训固定三轮，两模式顺序、4卡、1024/event/epoch，val32/test38；旧三轮已验收无需重跑，无优化器断点续训/不支持直接EPOCHS7。命令和反馈见audit_joint/PHASE3_RETRAIN_V57_20261014.md，沿既有PREPARED与COMPLETED首轮目录，交phase3_epoch两个文件。此为此前授权发布后必要入口纠正，无远程/GPU训练，未改.vscode。

### 2026-10-14 v57白名单发布与后续状态RGB复核

用户授权本轮白名单push。186相关回归通过、1模型依赖meta_profile未选，501源v57_handoff matched、两个P4 shell语法通过。此次发布包含此前v56/57原生STOP接线与失败交接；原生合同不再改。服务器沿sync_automot.sh分发（源duguxiaohun/automot_lead），按audit_joint/SERVER_V57_RELEASE_RGB_20261014.md一次新版正式CPU池，交handoff/fullflow_report.zip+fullflow_package.json，正式等量/关键支持/隔离预算通过后两模式首轮重载test。P3/readiness/旧fit/STOP盘点不重复，无新采集；UE5仅既有开发豁免，YESNO不要求等额。当前未提供GPU新命令以免对未验收目录开训。
实际复看HardBreak1902_0与Parked1748_1第40–46帧，共14RGB/42目视源SHA；相对上一轮Accident两窗不重叠，仍是既有train非盲审。两路线f43 stable YES、HardBreak f46/Parked f44 re_yield YES，原生4题7帧因果事实复算一致/54源SHA（和目视有重叠），STOP历史取既有有效回放；不以重新让行推定此前错标，无新改标签提示词依据。无新标定生成/全库/远程/GPU训练；源码测试轻量文档回执发布，checkpoints/RGB/权重和用户.vscode留本机。见v57_release_rgb_verification_20261014.json及checkpoints/v57_release_rgb_20261014。

### 2026-10-14 v57独立复审与正式池收尾

本轮未发现新增确定生产阻塞，v56横穿范围/v57停车位置已接入，不重复旧“未接线”结论。501源matched、117相关回归3.19秒通过及两个P4 shell语法通过；未完整回归，上一轮torch/tokenizers依赖固定ID测试未补跑。另一批既有12train路线780帧原生gzip→共享RGB4→短视图54.54秒，两模式244train空留出十事件齐、2340源SHA前后及目视后通过；对同路线v54问题episode/edge/mode/slice/target/facts及instance_ends零差异，不称字节/全部证据不变。两模式/world1/4四计划同序1024/event、小池最大114次/题，非GPU消费；488提示词SHA和短两图取四图最后两张通过，缺参考预算准入blocked。独立重载上一轮v57原生20RE5YES/4组，不计新生成。
实际10连续窗口70展示帧去重65RGB/195源，含既有开发窗口非盲审；10所选题另读7帧几何原生facts一致(当前STOP证据来自新回放)。未见新增确定改标/阈值/提示词依据。可以停止扩展本地审计，发布同步v57后一次正式CPU池关键支持/隔离/train-val-test十事件等量→两模式首轮重载，交fullflow_report.zip+fullflow_package.json；本轮未push勿把旧服务器当新版。P3/盘点/readiness不重做，UE5仅已有开发豁免，不补采、不强求YESNO/子场景等量；联合G0仍未认证。无生产修改/训练/远程/push，保留.vscode。见audit_joint/V57_INDEPENDENT_REVIEW_20261014.md和v57_independent_review_verification_20261014.json；本机checkpoints/v57_independent_review_20261014。

### 2026-10-14 v57 STOP停车位置原生接入完成

用户指出本地可生成数据，纠正此前停在诊断未接线。本轮已原生接入stop_position/Catalog→StopDutyTracker，采用匹配驶入车道端点平面（非绘制线认证），保留0–3m/≤0.1m/s/连续3帧及横穿门；不按专家起步补标。已有12地图自动生成1706支持边界、766269字节资产随源码分发，Town15/151/-1不支持几何排除；无需服务器全地图。道路/车道/控制位置/固定导出形状匹配，冻结目录SHA/地图SHA/控制ID/建立帧源，缺控制/漂移/坏几何/断帧/目录变更撤销完成不消债。建立源纳入instance_sources/编译SHA/风险包络，绑定失败无旧近缘回退。task57/teacher22/plan37同预约/空registry32，501发布源matched，P4两入口已指新清单，未push。
四路线929帧原生回放，最终建库35.62秒，两短/四图各771train空留出、883原始配对；RE5 proceed20YES/199NO/4UNKNOWN，20YES/4物理组全部经过原生编译，停车记录间距2.503–2.744m/最高速度绝对值0.08965，非monkeypatch。对v56共同键29UNKNOWN→YES/4→NO，另233新增67移除。2787源及12地图末次SHA核验，实际4连续28RGB/84源复看非盲审。另17完整train路线2205帧163.33秒，各1457题十事件齐，6615源SHA前后/重载后通过；两模式/world1/4首轮同序1024/event非GPU消费，RE5YES10/2组与四路线重叠不相加。596相关回归通过，1项固定ID测试base缺torch（此前pvi缺tokenizers）未完成，不报全套通过。
“停车位置未接入”代码阻塞关闭；不冒称历史地图/绘制线/全源标签认证。下一步发布同步后一次v57正式CPU池及train/val/test十事件等量、关键支持和隔离准入→两模式首轮重载。server_phase4_existing.sh新输出，交fullflow_report.zip+fullflow_package.json；当前未push勿跑旧入口。P3/盘点/readiness不重做，不新采集、不旧新拼库、不要求YESNO等量；UE5仍仅既有开发豁免。未改提示词、未远程/GPU/训练/push，保留.vscode。见audit_joint/STOP_POSITION_V57_20261014.md及stop_position_v57_verification_20261014.json；有效本机checkpoints/stop_position_v57_20261014/native_bundle_final和ten_events，中间旧小库不复用。

### 2026-10-14 v56 STOP题目范围修复与固定绑定拒绝

已改生产crossing_clear为同题nav范围、连续2s/静止4s预测，相关性先于可见性，去旧15m筛选；缺速度/相关不可见仍UNKNOWN。local_priority接nav；未改停车速度/0–3m阈值或提示词。诊断FrozenApproach拒绝初始位置错配、非同junction、错前驱车道、导出形状及冻结漂移；float32八ULP/≤1cm仅数值匹配假设非停车授权。任务v56/teacher21，新plan36同预约、空registry31，494源matched，两个P4launcher新清单，未push。
4完整train路线929帧原生21.47秒，717原始两四图配对，相同语义目标/结构对v55零变化，RE5仍0YES/199NO/28UNKNOWN。固定位置+新横穿隔离50.47秒，8锚点/1初始错配拒绝，20YES/199NO/4UNKNOWN四组，883对；33共同键变化之外新增233/移除67，monkeypatch无有效合同禁止建库训练。原生独立小库最终31.90秒，短两/四图各586train空留出仅4事件，ID/标签配对；有效目录native_bundle_final，初次teacher_scope说明纠正后旧小库合同拒绝未改SHA。2787源末次核验，实际4连续28RGB/84源复看非盲审。571不同回归通过，1配对测试base缺torch/pvi缺tokenizers未完成，失败日志保留，不报全套通过。
停车位置尚未生产接入，地图语义/身份输入合同和绑定失效撤销完成记录仍需落实，不能只替换before_area或把二维端点当认证绘制线。先关闭这一项→当前正式池train/val/test十事件1:1→两模式首轮重载，不要求YESNO等量，不重复P3/盘点/readiness，不立刻20h重建。无GPU/远程/训练/push，未改用户.vscode。见audit_joint/STOP_SCOPE_V56_20261014.md及stop_scope_v56_verification_20261014.json；完整checkpoints/stop_scope_fix_20261014。

### 2026-10-14 STOP组合隔离验证与28帧RGB复核

当前生产未变。4完整train路线929帧原生隔离组合实验55.31秒，2787源SHA重放及目视后通过；原v55/仅位置两组RE5 proceed均0YES/199NO/28UNKNOWN，位置+相关性优先实验20YES/199NO/4UNKNOWN、4物理组，883两四图原始题配对，33共同键变化(29YES/4NO)不是全部结构差分。8锚点实验1mm筛选不作为生产阈值；monkeypatch记录保留原生身份字段但无有效生产合同，禁止训练建库。实际四段连续28RGB/84源，既有窗口非盲审。四首YES旧横穿区域约[-2,6]m，侧向车辆前11–17m/距控制15.95–20.94m被15m筛选排除，近静止未证实这20题错，但不能仅调顺序即认证完整路口。固定绑定几何/适用条件缺口与停车参照、相关性仍需统一修正；不把合并稀少子场景代替修规则。55现有回归+7隔离合成边界通过，491源码matched；不新改提示词/阈值标签，不正式建库/GPU/push，保留用户.vscode。P3及服务器盘点不重复，不立即20h重建；仅STOP定向验收→新版正式池等量准入→首轮重载。见audit_joint/STOP_COMBINED_REVIEW_20261014.md和stop_combined_review_verification_20261014.json；本机checkpoints/stop_combined_review_20261014。

### 2026-10-14 STOP固定边界独立审计与后方跟车弃权定位

用户提出稀少细分类可合并：同事件同语义子场景合并、不强求子类/YESNO等量，仍保留十事件间1:1；不将其它边YES或UNKNOWN补缺格，未授权改十事件定义。独立491源码matched、61相关回归通过；两个诊断接入缺口复现：初始stop横移100m仍冻结且次帧same_control_verified；connecting road非junction仍套incoming端点上游模型。诊断label_authorization仍false，不冒称已生产错标，接入前需几何匹配与上游适用条件硬拒绝。
隔离进程只改固定边界停车位置、保留速度3帧/身份/退休/横穿，4完整train路线929帧55.37秒，2787源SHA，8候选锚点(实验≤1mm筛选非标定阈值)，5检查停车点满足；共同题13变化=9UNKNOWN→YES/4→NO均UE2，非全部结构差分。RE5仍0YES/199NO/28UNKNOWN。五真实点crossing_clear因自车后9.92–10.48m不可见跟车UNKNOWN，当前与2秒预测均不相交；副本只移除此车均True，确定相关性筛选过宽，不能忽略所有不可见车或直接删演员。实际14RGB/42源复核，含既有窗口非盲审；无生产源码/标签/阈值提示词修改，无正式库/GPU/push。下一步绑定拒绝与横穿范围修正后当前正式池等量准入，P3及服务器盘点不重跑。见audit_joint/STOP_BINDING_REVIEW_20261014.md与stop_binding_review_verification_20261014.json；原证据checkpoints/stop_binding_review_20261014。

### 2026-10-14 STOP上游偏移来源与signalReference固定位置续审

核对CARLA官方0.9.15 StopSignComponent源码：STOP路口引用适用时先移非路口前驱再后退1.5+1.5m；本机LEAD依赖亦0.9.15，但历史捕获版本未认证。新增诊断沿road/lane→junction laneLink→连接前驱→signalReference有效车道→唯一type206定义，五组signal21636/20072/20397/20232/21633重建真实触发中心误差0.112–0.738mm；70帧唯一关联，1帧进入junction保留拒绝。不是用最近绘制线猜控制，也不是放宽阈值或把3m硬编码为生产修正。
FrozenApproachProbe只读固定原位置/地图SHA/来源/控制ID，跨road和affects_ego清零不重绑，坏身份/缺演员/跳帧/坏几何/地图漂移拒绝；车头投影考虑当前车辆姿态。两服务器样例各18观察，f41进入路口仍保留旧边界，穿越不等于停车完成。累计4路线81不同帧243源SHA通过；本轮此前21RGB复看，这次继续未新增目视不重复计数。87回归通过含累计36新增。原489生产源码字节仍matched，新491 stop_binding诊断清单matched尚未push，中间stop_line清单保留原SHA不再作为当前发布清单。生产v55/老师20/标签提示词/阈值不变。
服务器盘点关闭/P3不重做/P4不旧池开训或全库重建。位置定义的道路边界对应、绘制线高程异常/曲线多车道适用边界仍需验收后才接生产停车计数，再当前正式池train/val/test等量及首轮重载。见audit_joint/STOP_CONTROL_BINDING_20261014.md与stop_control_binding_verification_20261014.json；完整证据checkpoints/stop_control_binding_20261014。无远程/GPU/训练/push，未改用户.vscode。

### 2026-10-14 STOP车道端点与地图停止线诊断

服务器STOP盘点已关闭，不再索取重跑。新增只读stop_line_probe：地图逻辑/解析路径+SHA绑定，RHT正负车道选前方端点，解析line/arc与车道宽度并检查junction laneLink；缺归属/未知几何/地图漂移拒绝，始终label_authorization=false。没有改生产老师/标签/提示词/阈值。
4已有train路线5控制点71帧/213源SHA前后和目视后通过，约3.95秒非全路线回放；5控制中心均在前方车道端点前3.000–3.018m，停车时车头距端点2.526–2.550m、距候选StopLine2约2.549–2.630m，旧前缘gap约-1.95至-1.99m，支持位置参照不合适。最近StopLine离端点0.008–0.093m但road不属于直接连接road，2D/高程/捕获地图身份与控制归属未认证，不按最近线/硬编码3m/扩阈值补YES；5帧原生crossing_clear均UNKNOWN。AccidentTown12 f41当前端点不直连junction明确拒绝，不能重绑旧债务到新车道。
实际复看两Accident f28–34与HardBreak第二STOP f253–259共21RGB，非盲审/新样本；未看val/test题。68回归通过含17新增，原生正车道CLI诊断通过；491新源码清单matched，原489生产源码字节未改，现有launcher不改，未push。P3不重复，P4先落实位置归属合同后正式池及train/val/test等量再首轮重载，不拼旧新回放、不盲重跑20小时。见audit_joint/STOP_LINE_CALIBRATION_20261014.md与stop_line_calibration_verification_20261014.json，完整证据checkpoints/stop_line_calibration_20261014。无新标定/老师回放/GPU/远程/训练，未改用户.vscode。

### 2026-10-14 正式STOP盘点交回与两条服务器样例本地重建

收到693980字节JSON/SHA f8a23eac...，inventoried_not_relabelled/无错误，8614路线1062401帧；index a4ec26e7与既有正式包一致，1327条报告路线artifact SHA逐项对齐原绑定、各分项加总一致。STOP证据路线train1139/val92/test96；记录消失后proceed YES候选train1455题215路线、val106题19路线、test130题15路线，共1691题249路线。非全错标数/最终库删除数/全RE5；未独立读服务器全回放、无val/test逐题审图，不需再次盘点。无记录7287路线不证明无STOP，不能仅重建249或1327后拼旧合同。
本地选报告前两AccidentTwoWays train路线原生重建492帧34.09秒，两模式294题配对、1476源SHA前后通过、空留出仅4事件诊断。服务器8条RE5 YES按路线帧事件边对齐均UNKNOWN；未获服务器逐源绑定，不冒称RGB与服务器全SHA核验。实际4连续窗口28RGB/84源查看后核验，Town12停车gap约-1.958m/Town13约-1.993m，义务保留修复有效但停车位置校准仍未关闭，不改YES/阈值/提示词。30专项通过，489发布源matched；未生产修改/远程/GPU/push，保留用户.vscode。下一步集中现有资产停车位置对应，落定后一次当前正式池及train/val/test等量准入，再P4首轮重载；不重复P3三轮、不要求再跑旧20h建库。见audit_joint/STOP_INVENTORY_RECEIVED_20261014.md与stop_inventory_received_verification_20261014.json，本机证据checkpoints/stop_inventory_received_20261014。

### 2026-10-14 v55白名单发布与STOP第二停车点RGB续审

用户授权本轮白名单push。489源v55_re5_admission逐项matched、101专项回归2.74秒通过、两P4 shell语法通过；原生只读inventory复核17路线2205帧，旧8候选/新0、无错误。沿用户sync覆盖同步后按audit_joint/SERVER_V55_RELEASE_RGB_20261014.md先源码SHA校验，再只读盘点既有P4 PREPARED/inputs/production/index.json；交phase4_stop_inventory_v54_20261014.json，已有则直接交、不重跑readiness/20h建库。P3已关闭不重做，P4仍待停车位置校准与当前正式池准入。
本地实际3连续窗口21不同RGB/63源查看后SHA核验，相对上一轮49帧重叠0，非全历史新见/盲审。HardBreak第二STOP528 f160–166前车后等候gap7.17m，f253–259移至路口停车gap约-1.954至-1.957m、速度绝对值<0.011仍False；后窗口无题，不补标。Parked f17–23接近STOP速度0.104–1.493，NO无充分改标依据。未改生产/标签/阈值/提示词、未全库重放/远程/GPU训练。白名单发布仅源码/测试/轻量文档回执，checkpoints/RGB/权重和用户.vscode不入本次提交。见新发布手册与v55_release_rgb_verification_20261014.json；完整本机证据checkpoints/v55_release_rgb_20261014。

### 2026-10-14 R-E5关键支持硬门与STOP导出代理定位

原始回放及最终过滤准入共用CRITICAL_EDGES，新增R-E5/proceed/readiness正反支持；existing_data_development仍仅豁免UE5 YES。旧1471题删8RE5YES后新报告8题/2组→0并硬阻塞；v55父库复用创建23752字节新view，两模式各1381配对，RE5 0YES/140NO(3组)，旧view拒绝源码合同漂移。不重放/改旧SHA，生产仍v55/teacher20，标签阈值提示词未改。
本机采集expert.py把STOP导出extent固定为1.5/1.5/0.5，非原trigger尺寸/道路停止线；历史采集源码未认证。两路线32帧96源SHA前后及目视后通过，实际复看14既有RGB/42源非盲审。f30框中心距原点约1.98m，中心距车头-0.46m、前缘-1.96m、后缘+1.04m；不能任选后缘或扩阈值补YES。新增只读stop_proxy明确unproven_stop_line/label_authorization=false，停车位置校准仍未关闭。已有Town12/13 xodr找到约3.09/3.05m外StopLine2候选，最近road不是当前lane直接连接road；地图版本/车道控制归属未认证，map_probe保留线索，不按最近线自动改标。
117相关回归通过含16新增，另2依赖测试在pvi缺peft/tokenizers未完成，保留失败日志不报全过；新489源v55_re5_admission matched且无额外Python，两个P4launcher接新清单但未push。不要求服务器重复readiness或20h重建；先按STOP_V55_FIX交旧正式回放只读inventory，P3三轮不重做，P4停车代理/当前正式池等量准入仍待完成。见audit_joint/RE5_SUPPORT_PROXY_20261014.md及re5_support_proxy_verification_20261014.json，完整证据checkpoints/re5_support_proxy_20261014。无远程/GPU/训练/新标定，未改用户.vscode。

### 2026-10-14 v55独立复审与R-E5关键覆盖漏检

487发布源SHA matched；本地原生3路线479帧37.41秒，两模式各346train配对、空留出，1437源SHA前后通过。实际7窗口49不同RGB/147源查看后核验，与上一轮148帧重叠6帧，非盲审。独立40 STOP及29采样准入回归共69通过，未跑全套。17路线v55原生两模式/world1/4仍1024/event同序，单题最高147次，仅小池算法验证。
确定P2：rule_baseline.critical_support漏R-E5/proceed；旧1471题删除全部8个该边YES后关键报告不变，不宣称整个准入误通过。v55同17路线RE5 YES8→0、UE2 depart YES12→3、RE2 enter YES36→29。建议补RE5 readiness正反支持硬门，不能扩大仅UE5允许缺格的开发豁免。STOP生命周期修复未见复发，但实际停车代理负间距未解决；Parked f95–99 gap clear但旧STOP失去身份使priority=None/UNKNOWN，债务持续至末154，HardBreak至末281。不能直接改YES或靠超时/越过/丢演员清债。
新旧同源v55回放91/479帧非字节相同：8018浮点末位差异最大9.10e-13及派生question_id，1e-12容差下无其它标签/事实/结束语义差异；不伪称ID/SHA跨构建相同、不改旧SHA。692新库提示词SHA通过，无新改提示词依据。仅新增审计文档/回执/本机诊断脚本，未改生产/标定/阈值，未远程/GPU/push，保留用户.vscode。P3已关闭不重复；P4先补守卫和代理验收、按STOP_V55_FIX只读服务器盘点再定正式池更新，不盲重跑20小时或混旧合同。详见audit_joint/V55_INDEPENDENT_REVIEW_20261014.md及v55_independent_review_verification_20261014.json，完整证据checkpoints/v55_independent_review_20261014。

### 2026-10-14 v55 STOP证据链修复与定向重算

修复STOP未履行义务因affects_ego清零消失、ego无road/lane副本无法满足检查；meta合法归属可用，矛盾副本拒绝。local_priority消费保留义务，跳帧/演员丢失/坏几何/不可读源只撤销证据不删除义务；原生读源失败新建tracker路径亦修复。完成后仍检查横穿交通，只有已履行且同物理触发体完整在自车后方才退休；未履行不因越过/起步/车道变化解除。未放宽0–3m代理位置或速度阈值、未改提示词，真实-1.965m偏差仍未校准，保守UNKNOWN可能影响后续阶段。
任务v55/老师v20、新协议v35/空registry v30、同预约不改旧SHA；新487源清单matched，两个P4建库launcher指向新清单但不要求立即全库重建。17路线2205帧重算156.53秒，两模式各1381train(旧1471)、空留出、十事件齐，6615源SHA前后及与旧一致；已报8条R-E5 YES全UNKNOWN且只恢复flag不翻转。两模式合计52YES→UNKNOWN/44NO→UNKNOWN，其余实例状态结构变化保留差分，非全池数量。两模式/world1/4计划1024/event同序非GPU消费；复看既有14RGB/42源，非盲审。69最终专项通过含25新增；广泛base1251通过4跳过13缺依赖，pvi相关89通过1跳过4缺huggingface_hub/peft/tokenizers，另44通过7缺tokenizers，不宣称完整全过。
新增只读stop_inventory，旧17路线复现8候选/新0；旧问题无control_evidence不证明无STOP，不授权旧路线复用。发布同步后按audit_joint/STOP_V55_FIX_20261014.md读取服务器既有PREPARED/inputs/production/index.json，交phase4_stop_inventory_v54_20261014.json，再定更新范围与新池准入。P4暂停旧“直接首轮”指示，不盲重跑20小时、不混旧新合同；已跑旧训练只保留不删除不续训。P3已验收无需重做。证据checkpoints/stop_v55_20261014/final及comparison；无远程/GPU/训练/push，未改用户.vscode设置。

### 2026-10-14 扩展本地RGB确认STOP证据链缺陷

撤回上一轮“Phase4没有必修标定问题”结论；CPU构建/等量证据仍有效但不等于标签证据通过。排除上一轮12路线，另17现有train路线2205帧按当前v54完整生成163.18秒，两模式1471题ID/标签配对、6615源SHA前后通过。实际看22联系表去重148RGB/444源，十事件各两窗口加两STOP补查，非盲审/全池认证；相对紧接上一轮70帧无路线/帧重叠，不声称全历史未见。
确定P1：StopDutyTracker在affects_ego=false时删除未履行义务，local_priority只检查active停止牌，导致2路线f36–39共8条R-E5 proceed YES入库；同RGB/速度/位置/提示词仅把f36标志恢复true，YES变UNKNOWN，单保留旧记录仍YES。P2：tracker强求ego bbox重复road/lane，真实meta225/-1而ego缺字段，合成有效代理位置三帧只补一致副本False变True。真实停车代理gap约-1.965m另需校准，不能只补字段或扩大阈值就放行。画面确有停车，不声称8YES全是行为错误；确定问题是自动证据链绕过自身要求。17路线4路线208条STOP记录全False，未测服务器全池影响数量。
23关键时刻原生几何事实复算一致不证明规则正确；63已有STOP/十事件回归0.36秒通过，未覆盖新真实形态。其它绕行YES固定目标段解释保留，只可见条件不保证完整扫掠；UE5仍无真实放行YES。生产/标签/阈值/提示词未改，按用户标定先报告要求保留反例，不远程/训练/push。P3验收不重复；P4先修STOP归属及义务生命周期/定向验收，再盘点既有正式回放影响，勿盲重跑20小时/改旧SHA兼容。见audit_joint/LOCAL_RGB_EXPANSION_STOP_FINDING_20261014.md及local_rgb_expansion_verification_20261014.json，原证据checkpoints/local_rgb_expansion_20261014。

### 2026-10-14 本地十事件连续RGB复核与开训收尾

当前480发布源码逐项SHA仍matched，新增交接/规则准入/采样45回归3.23秒通过。已有12训练路线780帧原生重建57.23秒，两模式244题ID/标签一致、十事件齐；2340源SHA生成前后核验、相对既有同路线v54 questions/instance_ends差分0。两模式/world1/4原生计划1024/event同序，小池单题最多114次且空留出，只诊断不正式训练。
实际查看十段连续窗口70不同RGB、210源SHA查看后核验；相对上一轮对应窗口后移7帧，非盲审/全源认证，不声称此前从未见。UE2 49/54/55及RE2 33–36重读因果7帧几何/提示词：RE2左侧目标，右车不相交，左车177从后侧不可见到局部不确定再可见相交，YES→UNKNOWN→NO一致。YES仅可见条件，不能替代规划器完整机动放行。UE1消失不补YES，UE5未见真实可见离场，其余冲突NO无充分改标依据。
没有新的确定生产阻塞，未改标定/阈值/提示词、未远程/GPU/push。P3三轮已验收不重复；P4正式CPU池已验收，直接按PHASE4_EXISTING_ACCEPTANCE_20261014.md两模式首轮/重载/test，不再20小时建库。UE5缺格/稀疏组/联合G0边界保留。见audit_joint/LOCAL_RGB_REVIEW_20261014.md及local_rgb_review_verification_20261014.json；完整本机证据checkpoints/local_rgb_review_20261014。未改用户.vscode设置。

### 2026-10-14 Phase4首轮失败交接补强

旧内联收集脚本遇到截断runtime/JSONL复现JSONDecodeError、退出1无ZIP；新增独立audit_joint/phase4_epoch_handoff.py，缺失/损坏/超预算/输入漂移保留不完整诊断，步与逐题记录不使用大列表投影，固定白名单不含权重。核对四rank完成步ID/位置/顺序/epoch/有限loss及采样SHA，事件计数继承原计划，不冒称重新核全池归属、优化器效果或模型质量。合成损坏样例新CLI退出2、1643字节包CRC/SHA通过；65相关回归2.18秒通过含16新测试与原生采样格式验证。既有480发布源码逐项matched，生产/学生合同、标定提示词/正式池均未变，无重建/训练/新RGB目视/远程/push。新工具发布同步后收集两模式首轮，但已有训练命令可以直接开始，不等待新工具；P3三轮不重复、UE5缺格保留为开发限制。见audit_joint/PHASE4_EPOCH_HANDOFF_20261014.md及轻量verification；完整本机证据checkpoints/phase4_epoch_handoff_20261014。未改用户.vscode设置。

### 2026-10-14 Phase4现有数据正式CPU池验收与直接首轮训练

收到fullflow 866063字节/12成员ZIP，SHA69d2ff8e...，外层/成员SHA/CRC与480源逐项匹配36449cc62。finished_cpu/admitted_existing_data_development/退出0，8614路线1062401帧、301596条纯规则train，自动留出28058/26520，实际选定val37/test18每事件，两模式ID摘要/题序和world1/4一致；首轮计划1024/event非GPU实际消费。U-E5池820题最多重复2次；proceed YES仍0、NO711；UE2 return2YES/2组，UE7 complete13YES/13组；UE5 val/test3/2组、UE7 2/1组。只批准已有数据独立开发，不冒充能力完整或联合G0。
原生预算复算新增6.20GiB+20GiB通过，回执盘约2.54TiB空闲。本次实际资产根/mnt/workspace/cee3sgh/AutoMoT，PREPARED=checkpoints/phase4_existing_20261013_184628_526027234；不要套旧/home/cruser1路径。CPU构建71522.58秒约19h52m，回放收集27275.2秒约7h35m，另12h17m无细分计时，不认定死锁；无需重建补计时。原val/test空但独立references4已被rule_consistency正确接入。两图weak_supervision_rows=0是转换层统计，原来源301596规则，非人工。
P3 ZIP与已验收三轮同SHA，不重复。无新确定必修生产缺陷/标签提示词改动，新RGB目视0；84相关回归通过、2项本机缺peft/tokenizers失败，短RGB收集亦缺tokenizers，旧/tmp环境不存在，失败日志保留不声称完整通过。独立核验8625绑定记录关联/预算及包，非本机全库重载。按audit_joint/PHASE4_EXISTING_ACCEPTANCE_20261014.md现有已发布入口两模式各一轮→重载→固定test；无需等待push。文档含只读≤30MB反馈命令，交phase4_epoch_report.zip+phase4_epoch_package.json，未完成也保留missing。1294字节合成不完整反馈包CRC/SHA通过，非GPU。证据checkpoints/phase4_existing_received_20261014；phase4_existing_acceptance_verification_20261014.json。仅文档/回执，未push/远程训练，未改用户.vscode设置。

### 2026-10-14 Phase3三轮交回独立验收

收到807215字节/52成员ZIP，SHA566f7cf8...，外层/成员SHA/CRC及478源与71cc49636逐项一致；当前6处后续Phase4发布差异有解释，非服务器混版。两模式three_epoch_verified/退出0、各7680步；六份实际消费回执每轮十事件各1024、三轮各3072，总30720呈现，val32/test38事件等量。原生parser独立重算2040条输出、固定ID/标签/两四图[1,3]子集、评分与能力守卫一致，重载各0差异；同首轮题集/RGB回执一致。
两图val240/320=75%、test270/380=71.05%；四图val242/320=75.63%、test274/380=72.11%。两模式val减速27/58、25/58未过0.5，test左变道11/27、13/27未过0.5；不批准生产。每轮减速2080监督，三轮6240，不归因为缺题。val路口全STOP/test弱势交通全STOP仍有覆盖局限；不据test调标签/提示词。没有新确定生产缺陷，未改标定/阈值/提示词，新增RGB目视0，未重复2478完整回归。
Phase3三轮流程关闭，不重建/readiness/重复一轮三轮；保留服务器phase3_three_epochs_20261013_174517_669756526两final。下一步仅现有数据Phase4当前正式CPU包，已跑则直接交fullflow_report.zip+fullflow_package.json；未跑按PHASE4_EXISTING_DATA_20261013.md，不采集。P4现有数据开发模式保留UE5 YES缺口，正式池/联合G0未验收。见audit_joint/PHASE3_THREE_EPOCH_ACCEPTANCE_20261014.md及phase3_three_epoch_acceptance_verification_20261014.json；证据checkpoints/phase3_three_epoch_received_20261014。仅文档/回执更新，未训练/远程/push。

### 2026-10-13 用户明确不补采：现有数据Phase4开发入口

最新用户纠正：只使用现有数据集，不采集新RGB/CARLA；撤销上一条补采方向，不再索取采集环境路径。本轮新增显式existing_data_development准入，仅容许U-E5/proceed/readiness/YES已知缺格，实际0和unsupported_capabilities保留，成功状态admitted_existing_data_development不冒充能力完整/独立准确率/联合G0。strict默认仍阻断；其它关键格、十事件缺失、原生隔离/等量留出、来源/合同/两模式配对/预算守护继续硬拒绝。生产v54/老师v19/标签/提示词/采样均未改；仅短视图准入源码合同变化，新描述文件不能伪改旧SHA兼容。
新增server_phase4_existing.sh：对已有lead_data新目录CPU候选扫描→v54gzip回放→共享RGB4→自动留出→短视图→首轮预算/开发准入→≤30MB两文件包，无仿真/新RGB/训练；20GiB保留/约12GiB净写盘守护。服务器旧v52池不能混新版，需要已有原始数据按当前规则生成一次正式库。发布同步后按audit_joint/PHASE4_EXISTING_DATA_20261013.md命令执行，交handoff/fullflow_report.zip+fullflow_package.json，先核正式池后两模式各首轮重载/test。跨任务完整G0尚未过，不阻塞独立开发实验而不冒称全部审完。
完整2478通过108.66秒；复用两真实v54父库，两模式各244train ID/标签配对、21/13输入绑定不变，空留出仍blocked。480源matched、真实P3已收首轮原生三轮准入verified；缺模型原生prepare退出2/550字节包通过。无新目视/全池重放/远程/GPU运行。新release_sources_v54_existing_data_20261013.json及P3入口清单指向同步，P3三轮数据与命令不变。见phase4_existing_data_verification_20261013.json及checkpoints/phase4_existing_development_20261013。

### 2026-10-13 U-E5缺口收敛与截止后原始窗口检查

用户要求补齐而非重复报告缺口。当前v54对23运动截止+8源不连续实例（28训练路线）检查最多20帧后续：含前史536不同帧/1608源SHA前后核验，截止起350帧均无侵入后可见且含原裕量分离；10可见仍相交、340未通过可见门，5.80秒。实际看Town12 3131_0 f139–145、Town13 1176_1 f130–136共14不同RGB/42源，非盲审。未全89路线重放、未复活实例/改标/降门/跑仿真/训练/push。原生场景本机源码使用固定侵入offset且无显式可见撤回阶段，与画面相符；未核采集服务器源码SHA，不断言所有旧数据由此版本生成。
收敛为少量“真实侵入→镜头内平滑回本车道→自车等待后放行”的补采机制验证；保留同ID、三相机/同步源、前后7帧和正常路线运动，现有原生老师及最终两模式过滤自动验收真实proceed/readiness YES，不要求逐题人工/YESNO等量。成功后再一次当前正式池+联合隔离；不重复旧池找已否定的截止窗口。已询问服务器是否有CARLA+LEAD采集环境，尚未答复，不编造可运行采集命令。Phase3已发布三轮入口照常独立推进。详见audit_joint/UE5_LOCAL_RGB_20261013.md顶部补采条件和ue5_completion_verification_20261013.json；完整证据checkpoints/ue5_completion_20261013。生产合同/478源清单不变。

### 2026-10-13 v54白名单发布与U-E7双条件RGB续审

用户授权本轮白名单push。当前478源v54_admission matched、真实P3首轮ZIP原生三轮准入verified，79专项回归4.70秒通过，shell语法通过；只发布源码/测试/轻量文档与回执，checkpoints/RGB/权重及白名单外资料留本机。服务器沿既有sync覆盖同步，按audit_joint/SERVER_V54_RELEASE_RGB_20261013.md运行固定三轮P3，原PREPARED_ROOT及COMPLETED_EPOCH_ROOT不变，交phase3_epoch_report.zip+phase3_epoch_package.json；不重复readiness/建库/首轮。Phase4仍缺UE5真实放行YES、v54正式池与联合隔离，暂不启动旧全库/fit/训练。
本机当前v54两条U-E7训练路线107帧重建，两模式各110train/空留出，14.46秒；321源SHA前后通过，实际两联系表14不同RGB/42源，属于既有开发窗口复看非盲审。Town12 002113 f43先解除、f48才稳定；Town15 002214 f49先稳定、f52才解除，两条件齐备才complete YES；f47/48无题不补标。未发现新确定标定/提示词缺陷，无生产修改/远程/GPU训练。见发布手册和v54_publish_rgb_verification_20261013.json，完整证据checkpoints/v54_publish_rgb_20261013。

### 2026-10-13 v54自动准入缺项诊断补强

真实小库复现缺自动留出即modes={}、源池诊断丢失；rule_baseline分开处理参考绑定，保留两模式源分布/关键readiness支持，缺项仍blocked。预算缺失/截断/非对象也落阻塞报告。关键支持只计train自动原来源+指定事件边+readiness，人工/其它边/catchup/val不补格；最终选择、等量、输入漂移及预算门仍在。460 audit_joint测试9.11秒通过含12新增。复用两v54父库，新轻量视图23839/23904字节，原生两模式191+53题ID/标签配对，21/13输入绑定不变；两次原生CLI缺参考+坏预算均退出2且完整诊断。短视图冻结准入源码，旧view不改SHA，需新描述文件但无需重放/复制RGB。生产v54/老师v19/标定提示词不变，新478源v54_admission matched、实际P3首轮ZIP三轮准入verified，P3命令不变。未重复2450/readiness/全库/首轮，无新RGB/回放/训练/远程/push。P4仍缺UE5真实YES/正式等量留出/联合隔离；见audit_joint/V54_ADMISSION_DIAGNOSTICS_20261013.md及对应verification，证据checkpoints/v54_admission_diagnostics_20261013。

### 2026-10-13 v54独立十事件自动数据与RGB复审

未发现新的确定生产代码缺陷；v54离场时序与partial诊断隔离通过，478源matched、7既有证据SHA及2450全套日志核验，实跑179相关回归39.52秒通过。已收P3首轮ZIP原生三轮准入仍verified，不重建/readiness/首轮。新建10短路线642帧两模式各191题；旧两条路线对应事件仅人工band支持，另补2规则路线138帧各53题，两个独立池合计780帧/244纯weak_rule_teacher题覆盖十事件。2340原文件SHA前后核验。原生3轮计划各event1024、两模式/world1/4同序，未模型训练；小池单题最多114次，不能用于正式训练。训练池上无放回选题90=9/event仅算法验证非val/test；空留出拒绝。
实际十联系表70不同RGB帧/210源，开发定向非盲审；UE2 depart与RE2 enter YES额外重载7帧几何，visible-condition提示词边界一致；其余冲突NO未见充分改标依据。UE5原始proceed8NO/3UNKNOWN/0YES，其它边YES不补放行；P4仍需真实正例、当前v54正式池/隔离及首轮重载。无生产/标定/提示词修改、远程/GPU/push。见audit_joint/V54_FLOW_REVIEW_20261013.md及v54_flow_review_verification_20261013.json，原证据checkpoints/v54_flow_review_20261013。

### 2026-10-13 U-E5 v54离场时序与部分投影修复

确认原生合成反例：侵入前可见分离被visible_departure误用，旧facts/answer为YES；限制历史证据严格晚于restriction_frame，缺失/非法时不授权，新结果UNKNOWN。近裁剪增加12边求交，partial_crops独立诊断不进入完整quality_pass/运动匹配/放行。任务v54/老师v19，新同预约plan v34/空registry v29，旧产物合同拒绝，不改提示词/阈值。两真实路线193帧新建，两模式各87train/空留出，478原始题语义差分0；109窗口763帧2289源SHA核验，101离场False/8源不连续，153帧部分投影。实际复看21帧RGB，非盲审；4591/f57可裁剪但固定走廊仍相交，真实UE5 YES仍0。2450完整回归108.15秒通过含19新增；首次pvi缺tokenizers收集失败，用既有完整环境通过。478源v54清单matched，Phase3生产/学生合同不变、已收首轮ZIP三轮准入verified，服务器三轮命令不变；不重复readiness/建库/首轮。见audit_joint/UE5_VISIBILITY_V54_20261013.md及ue5_visibility_v54_verification_20261013.json，原证据checkpoints/ue5_visibility_v54_20261013。P4正式新池/真实正例/联合隔离仍缺，无训练/远程/GPU/push。

### 2026-10-13 U-E5本地小库与35帧RGB定向续审

本机重新读取109截止窗口/763不同帧，2289原RGB/meta/bbox SHA前后通过；实际目视5连续窗口35不同RGB另1原图含在35内，非盲审/全763目视。101可计算窗口visible_departure全False，8源不连续；56截止帧当前可见范围clearance=True不当完整许可。5可见且分离帧全部早于restriction_frame，是驶入前而非离场。确认4591/f57部分红车可见、visible_pixels32315，但投影8角点近裁剪要求使crops空/visible未知；0.25/0.75裕量仍相交，不能因此改YES，只报告投影保守盲区。
两训练路线193帧原生v53回放→共享RGB4→短2/4各87train加载配对通过，14.62秒；193条U-E5范围记录与既有v53逐行相同。小库仅UE2/5且空留出，UE5 proceed14NO/0YES，非正式训练准入；没有新标定修复或正例。见audit_joint/UE5_LOCAL_RGB_20261013.md及ue5_local_rgb_verification_20261013.json，产物checkpoints/ue5_local_rgb_20261013。生产/阈值/标签/提示词未改，无GPU/远程/训练/push。后续定向可见离场观测，正式v53全池/联合完整索引隔离另验，不重复readiness或全89重放。

### 2026-10-13 Phase3三轮失败诊断与常驻回归

复现截断epoch_result使package直接JSONDecodeError且无ZIP；修复损坏/非对象/非法轮数进入incomplete诊断包，记录errors/missing_files，shell把EPOCHS传入package，启动前阻塞仍保留三轮范围，轮数矛盾拒绝通过。正常训练/标定/提示词/超参未改；临时CPU三轮检查固化常驻回归并补第三轮缺题拒绝。51相关测试2.07秒通过含12新增，原生失败CLI两包544/661字节SHA/CRC通过；新475源three_epoch_hardening清单matched，实际已收首轮ZIP准入仍verified，旧清单/复审证据不改。首次默认Python无torch，改用已有pvi；新测试漏INVALID=0的3断言修正，失败日志保留。未重复完整2419/readiness/建库/首轮，无新RGB/回放/远程/GPU/push。发布同步后顶部固定三轮命令不变，交两个phase3_epoch文件；P4真实UE5正例/v53池/联合隔离仍缺。见audit_joint/PHASE3_THREE_EPOCH_HARDENING_20261013.md及对应verification，原证据checkpoints/phase3_three_epoch_hardening_20261013。

### 2026-10-13 Phase3固定三轮独立复审通过

三轮入口增量复审未发现新的确定阻塞缺陷；475源matched，已收首轮843428字节/43成员SHA/CRC重核及原生三轮准入通过。实跑107相关回归6.26秒；既有2419完整日志SHA核验，未重复全套。另原生CPU线性模型三轮120题、每轮十事件各4全部消费；第二轮首题注入拒绝即停止，只留第一轮完成回执。合成池三轮1024/event、world1/4题序重组通过，非正式池/GPU质量证明。7680步warmup2000=26.04%，固定final/val/test，不新改标签/提示词/采样/超参。可以发布现有三轮入口，不重建/readiness/首轮；旧test已查看，后续重复比较不冒称未曝光最终测试。P4 UE5真实YES、新v53池与联合隔离仍缺；无新RGB/全源回放/生产修改/远程/GPU/push。见AutoMoT/qwen3vl_local/audit_joint/PHASE3_THREE_EPOCH_REVIEW_20261013.md及phase3_three_epoch_review_verification_20261013.json；证据checkpoints/phase3_three_epoch_review_20261013。

### 2026-10-13 Phase3首轮全流程验收与固定三轮开发入口

收到843428字节/43成员phase3_epoch包，外层/成员SHA/CRC及475源与364e1e4b1一致；两模式均first_epoch_verified、退出0。两图旧adapter解析目标/SHA不变，四图新训；原生parser独立复算2040条输出，val/test固定题集及逐动作分数/能力守卫重算一致，短两图是四图[1,3]子集，同题同标签。实际每模式训练十事件各1024、val32/test38，重载各0差异；Phase3独立首轮链路关闭，不重建/重复readiness/首轮。
两图val215/320=67.1875%、test227/380=59.7368%；四图val214/320=66.875%、test222/380=58.4211%。test减速19/81及16/81、恢复均20/52、左变道均8/27，能力守卫未过，非生产批准。实际每轮DECELERATE2080/RESUME1680/STOP3422/左右各684/KEEP1690，不能认定缺监督。首轮warmup2000/2560=78.125%是预算局限，不证明弱项唯一原因；未新改标定/提示词/采样/阈值，无新RGB目视。
扩展已有server_phase3_epoch支持EPOCHS=3：仅接受已审完成首轮ZIP SHA b555c258...，绑定同准备/生产合同，拒绝REUSE_EPOCH_ROOT三轮跳训；两模式从基座新训3轮各30720呈现、每轮实际等量检查、final-only、最终val重载/test、≤30MB包带三轮回执，20GiB守护/预算重算，无优化器恢复或自动七轮。成功three_epoch_verified只是开发链路通过。新475源multiepoch清单；旧清单不改。完整2419通过106.86秒（新增7），真实已收ZIP原生准入及新source matched；CPU模拟子进程不冒称GPU三轮。未push/未远程GPU。
发布同步后按audit_joint/PHASE3_ADMISSION_FIRST_EPOCH_20261013.md顶部新命令，COMPLETED_EPOCH_ROOT=checkpoints/phase3_epoch_recovery_20261013_121248_683529276，复用原PREPARED_ROOT，EPOCHS=3、新OUTPUT，交phase3_epoch_report.zip+phase3_epoch_package.json。按实测两图77分钟/四图130分钟每轮，顺序三轮粗估10–12小时而非保证。原证据checkpoints/phase3_epoch_completed_20261013，轻量回执phase3_epoch_completed_verification_20261013.json；三份指南最新状态同步。P4UE5真实proceed YES、新v53全池及跨任务隔离/联合G0仍未关闭，不要求全量人工审核才能推进P3独立开发实验。

### 2026-10-13 Phase3两图首轮交接、重载入口修复与只读复用

收到phase3_epoch_report.zip246461字节/17成员，外层/成员SHA/CRC及475源与81a486f28逐项一致。两图已完整训练2560四卡步/10240次，十事件各1024实际消费；原生parser复算320val各事件32题、固定ID集合一致，215正确/67.1875%。减速4/58（38答KEEP）、恢复18/38、左变道3/8；路口32/32参考全STOP，不能以整体分数声称全动作已会。test与四图未运行，无新RGB目视/规则修改。
修复两确定编排缺陷：LoRA eval禁用history-rgb-mode覆盖却被显式传入；epoch_result使用独占write_json反复写导致FileExistsError掩盖原错误且阻止第二模式。改为读取持久化合同、调用前核对choice/RGB/prompt；仅进度原子替换，不可变证据不动。新增REUSE_EPOCH_ROOT仅接受本次固定ZIP SHA，核验准备/生产源码/报告/六adapter资产/验证RGB，原两图train只读引用，先补重载320+test380再新训四图10240及重载/test，不重训两图、不复制权重，不是优化器恢复。旧目录保留，新输出≤30MB两文件；新475源recovery清单，旧清单不改。用户随后授权push，按白名单发布；未远程GPU。
25专项含16新增通过；完整2412通过106.69秒。合成编排替代模型子进程，不冒称实测GPU；实际交回adapter元数据原生CLI通过。按audit_joint/PHASE3_ADMISSION_FIRST_EPOCH_20261013.md顶部新命令发布同步后续跑，PREPARED_ROOT=checkpoints/phase3_prepare_20261012_231356_193369568，REUSE_EPOCH_ROOT=checkpoints/phase3_epoch_20261013_100331_732771255，交phase3_epoch_report.zip+phase3_epoch_package.json。证据checkpoints/phase3_epoch_received_20261013，轻量回执phase3_epoch_received_verification_20261013.json。P3正式池不重建/readiness不重复；P4UE5真实YES缺口/v53全池与联合隔离仍未关闭，不声称两阶段全部验收。

### 2026-10-13 Phase3正式数据通过与实际等量首轮入口

收到phase3_prepare_report.zip163418字节/19成员，外层/成员SHA/CRC、471源与3f9daefe1420e199d0c6772536912b987cda1cec一致；manifest无尾换行原生序列化重现登记SHA，非漂移。服务器CPU正式索引15096行=14124/516/456，3510run/3509物理组，原始速度/动作不一致0、跨split物理重叠0；choice训练1024/event、val16/32/test38，评测各事件至少5物理组，正式建库关闭，不再readiness/重建。只核交接证据，本机未独立重算全库/新RGB。
发现正式train超长输入零损失推进及val跳题仍评分会破坏实际1:1；修复完整轮次任一rank打包拒绝即同步停止、DDP填充损失归零、各rank消费计数汇总核对计划/十事件、固定val漏题拒绝。显式max_steps smoke不冒充完整轮次。新增server_phase3_epoch.sh/.py及475源清单：冻结已验收索引SHA，CPU LoRA预算→choice短2/4各从基座1轮10240→final-only adapter→未合并LoRA同320题重载/raw重解析→380题固定test→≤30MB包；20GiB守护，无optimizer/RGB复制，不自动7轮。按既有4B拓扑预算约4.734GiB增量+20保留，实际启动重算。无P3优化器恢复能力，不声称可无损续训。
完整2396回归通过103.77秒；首轮2392通过4旧测试期待静默跳题，修正断言后全套通过，失败日志保留。真实CPU线性小模型40题原生循环验证各事件4及缺题拒绝（替代图像/损失，不是QwenGPU）；原生CLI本机缺服务器索引退出2/未训练，26344字节3成员包及475源通过。用户随后授权push，按白名单发布。见audit_joint/PHASE3_ADMISSION_FIRST_EPOCH_20261013.md和phase3_admission_verification_20261013.json；证据checkpoints/phase3_prepare_received_20261013。发布同步后仅运行新epoch入口PREPARED_ROOT=checkpoints/phase3_prepare_20261012_231356_193369568，交phase3_epoch_report.zip+phase3_epoch_package.json。
P3部分动作评测支持窄（JUNCTION_RULE_CONFLICT val全STOP、VULNERABLE_CROSSING test全STOP），binary同RS错事件留出0；不以事件等量冒称所有动作准确，但可先choice基线。P4UE5真实proceed YES仍0、新v53全池和联合隔离未验收；不阻塞P3独立开发实验。生产标定/阈值/提示词未改，无新RGB/本机GPU/远程训练，G0仍未通过。

### 2026-10-12 Phase3缺索引交接核验与自动建库入口

收到Downloads Phase3包25977字节/2成员，外层及成员SHA/CRC、468源码与本机及4e59862ed01ef9dae8266024e1e210a2fc074f36逐项一致。结果index_not_selected/paths空/truncated=true，仅限定路径盘点，非全服务器缺失证明或正式数据准入。不重跑readiness/路径盘点。新增server_phase3_prepare.sh/.py：471源清单，fresh输出，原生全源build+候选隔离+binary/choice原始meta/标签/RGB路径核验+等量计划，四图/短两图合同与提示词SHA，训练1024/event、val16/32上限/test固定无放回、world1/4题序；默认CPU2worker，20GiB余量/约8GiB净写盘轮询保护，成功/失败≤30MB包。不复制RGB/回放/权重，不训练；完整原始信号检查不冒称全源RGB视觉认证。
214相关回归通过29.16秒含9新增；原生CLI缺collection退出2、26279字节/3成员包核验，471源matched。见audit_joint/PHASE3_RECEIVED_NEXT_20261012.md及phase3_received_verification_20261012.json；本机证据checkpoints/phase3_received_20261012。用户随后授权push，新入口按白名单发布；同步后运行一次交phase3_prepare_report.zip+phase3_prepare_package.json。P3正式索引/首轮预算后可独立推进；P4仍缺UE5真实proceed YES、新v53全池待准入，旧v52不可混用。生产标签/阈值/提示词未改，无新RGB目视/全源重放/GPU/远程训练，联合G0未过。

### 2026-10-12 Phase3白名单发布与并行RGB续审

用户授权push。本轮新增server_phase3_index：468源SHA自动核验，未指定索引仅深度4/最多100条路径盘点、不自动选择；指定正式索引绑定逻辑/解析路径与前后SHA，完整准入并带manifest，正常/缺失/阻塞均交≤30MB ZIP+独立SHA回执，无训练。29专项通过1.78秒（含3新测试），原生CLI合法/缺失两包27990/25881字节核验；已有2368完整回归不重跑。服务器沿sync_automot.sh同步后按audit_joint/SERVER_PHASE3_NEXT_20261012.md执行一次，交phase3_index_report.zip+phase3_index_package.json；不重复readiness/旧fit/旧全库，旧v52不能混新版。
本机实际查看Town13 1148_0 f58–64和1159_1 f77–83，共14不同RGB/42源SHA；均目标失去可见后才分离，未获真实proceed YES，不改标签/阈值/提示词，非G2；无完整路线重放。证据checkpoints/phase3_publish_rgb_20261012、phase3_publish_rgb_verification_20261012.json。P3正式索引验收后独立推进，不等待UE5；当前联合G0仍阻塞，无远程/GPU训练。

### 2026-10-12 Phase3正式train/eval统一索引准入修复

check_index强制manifest与提示词合同，候选variant额外校验冻结开发清单；独立eval在NCCL/输出/选题前执行全索引隔离，train含sampling-only共用，train.sh传实际variant；four_runs保留baseline也使用候选开发隔离池的原保护。未改P3标定/提示词/等量采样，P4 v53生产/学生合同与既有view一致。新增26回归，完整P3/P4/audit_joint2368通过105.36秒；6原生CLI合成验证5非法输入退出1/无输出目录，合法候选CPU采样40题=十事件各4、两类val各事件1/退出0，无模型加载或训练。466源清单matched未push。
见audit_joint/PHASE3_ENTRY_FIX_20261012.md与phase3_entry_fix_verification_20261012.json，产物checkpoints/phase3_entry_fix_20261012。代码发布后仅收实际P3索引的preflight JSON+manifest，不重复readiness；缺索引先生成正式库，不能删manifest兼容。从基座新训不要求旧adapter，P3准入后可独立推进不等UE5。UE5真实YES仍0/未重复回放RGB，P3正式索引及联合隔离未验收，G0仍阻塞。无远程/GPU训练/推理/push。

### 2026-10-12 Phase3独立入口准入续审

确认两处入口缺口：独立eval未接完整索引/候选开发隔离检查；baseline preflight缺manifest时跳过提示词及额外开发清单。36行原生合成反例中，有manifest拒绝开发组入test，移开同一清单后CPU采样通过；恢复清单后候选eval仍生成十事件各1题并到模型加载哨兵（无加载/推理）。不声称实际正式库已泄漏；four_runs总入口已有前置保护。90相关回归4.54秒通过，未重复完整套件。目录盘点InvadingTurn train91/val8/test3，train过滤2后89均已重放，无新合格训练路线/帧号变化；未重读源内容、无新RGB/事件回放。U-E5真实正例及P3正式索引/联合隔离仍缺。仅报告建议统一正式入口准入，生产/标签/提示词未改，无远程/GPU/训练/push。见AutoMoT/qwen3vl_local/audit_joint/PHASE3_ENTRY_REVIEW_20261012.md及phase3_entry_verification_20261012.json；原始证据checkpoints/phase3_entry_review_20261012。

### 2026-10-12 v53修复独立验收与开训收尾

上轮U-E5走廊混用缺陷关闭，本轮未发现新的确定生产代码缺陷。实跑547相关回归14.20秒通过；既有完整2342/137.96秒日志SHA及退出0核验，本轮未重跑全套。11证据SHA/465源清单matched，逐行复算10799帧1219题语义差分0、32397来源回执SHA相同（未重读全部原文件）；原生v53短视图2/4各28train加载、旧v52合同拒绝。新增RGB目视0，无89路线重放。真实UE5 proceed仍711NO/290UNKNOWN/0YES；后续集中可观察源、新合同正式池准入与P3正式索引/跨任务隔离，不重复服务器readiness/旧fit/旧建库。旧adapter不是从基座新训前置条件。见AutoMoT/qwen3vl_local/audit_joint/V53_ACCEPTANCE_REVIEW_20261012.md及v53_acceptance_verification_20261012.json。生产/标签/提示词未改，无远程/GPU/训练/push。

### 2026-10-12 v53 U-E5同实例走廊离场修复

修复visible_departure历史导航/当前固定走廊混用：历史及当前均逐帧投影identity_world/width；缺几何不给离场证据，U-E3/可见性/阈值/优先权/截止/提示词未改。任务v53/老师v18、同预约plan v33/空registry v28，旧v52合同实测拒绝，不修改旧hash。14新增原生反例与否决测试；完整P3/P4/audit_joint2342通过137.96秒。真实一路线原生共享RGB4→短视图，两模式28train/空留出，4.84秒通过，非正式准入。
本机89条训练路线10799帧单CPU重放496.74秒/58964KiB，1219道UE5四图问题语义差分0、32397源SHA一致；proceed711NO/290UNKNOWN/0YES、109实例截止78/23/8不变。复用既有RGB证据，本轮新增目视0/非盲审。修复确定机制缺陷但没有补出真实放行正例；定向寻找/补可观察源，不用出画/恢复运动/截止补YES。465源v53清单matched未push；server_finish指向新清单，旧v52手册命令已标历史不可混用，服务器不重跑readiness/旧fit/五小时建库。旧等量及预算仅为v52正式证据，v53全池待正例收敛后重建准入；Phase3正式资产/联合隔离仍缺。见audit_joint/UE5_CORRIDOR_V53_20261012.md和ue5_corridor_v53_verification_20261012.json；产物checkpoints/ue5_corridor_v53_20261012。无GPU/远程/训练/push。

### 2026-10-12 v52流程复审与U-E5离场走廊反例

完整P3/P4/audit_joint回归2328通过/107.07秒。原生合成反例定位teacher_events.visible_departure历史使用各帧导航、当前U-E5使用固定实例走廊；仅改历史导航、车辆/可见性/固定走廊不变，UNKNOWN变YES。确定口径缺陷，本轮只报告未修，不声称真实库已有此错标或修复会补出正例。复算既有89路线10799帧仍711NO/290UNKNOWN/0YES，8证据SHA通过；实际复看4联系表28不同RGB/84源SHA，非新独特帧/盲审，无全源重放。v51索引覆盖已修，正式P4池/等额/首轮预算已通过，不重复服务器建库/readiness；P3正式资产/联合隔离仍未验收。见AutoMoT/qwen3vl_local/audit_joint/V52_FLOW_CODE_REVIEW_20261012.md和v52_flow_code_review_verification_20261012.json，反例及日志checkpoints/v52_flow_review_20261012。生产/标签/提示词未改，无远程/GPU/训练/push。

### 2026-10-12 U-E5原始支持提前检查与离场RGB续审

本机按冻结split/异常时长过滤重放89条InvadingTurn训练路线10799帧，504.16秒/58728KiB；U-E5 proceed711NO/290UNKNOWN/0YES、109实例78后方截止/23运动截止/8源不连续。实际收尾四联系表28不同RGB、84源SHA，非G2/全源认证；几何变空不等于可见离场或完整许可，未改生产标定/提示词/学生合同。
server_finish新增原始train/readiness/RGB4五关键边正反支持提前检查：缺格直接报告并跳过昂贵编译，非零仍需最终过滤准入。event_gap加入按物理组限量训练样例和可选原始七帧绑定；departure_probe分报固定走廊/历史导航与忽略旧目标消融，绝不生成标签；原始样例失败保留原生统计、退出2。真实五路线复用2.11秒提前停止，49原文件不变、2813字节包通过；tiny预算夹具非4B/GPU。392联合审计回归7.04秒通过，新462源fullflow3未push。见audit_joint/UE5_DEPARTURE_AUDIT_20261012.md与ue5_departure_verification_20261012.json；产物checkpoints/ue5_continuation_20261012。服务器发布后仅手册第0步加--data-root读取既有证据，不重跑五小时建库/readiness/旧fit；本地可继续同实例离场与截止规则研究。UE5正例缺口仍阻塞首轮、G0未过/G2未运行，无GPU/远程/训练/push。

### 2026-10-12 正式全池交接通过与U-E5唯一首轮阻塞

收到fullflow ZIP 864998字节/11成员，外层及成员SHA、458源码与bafcedf21一致；两模式/world1/4题序及绑定对照通过。8614路线1062401帧完成，CPU18508.81秒/增量回放7980.4秒；规则train301588，实际每事件1024，val每事件37/test18（370/180），自动留出原池28052/26525；两模式首轮预算新增6.20GiB+20GiB保留、当前39.49GiB通过。仅缺UE5/proceed/readiness/YES：原始764NO/320UNKNOWN/0YES，实际train711NO/0YES；不能用hold/restrict YES代替。UE2 return2YES/2组、UE7 complete13YES/13组仍稀疏，UE5留出3/2组、UE7留出2/1组，不冒称独立泛化或所有状态覆盖。
修复server_finish打包：保留原build_result、区分finished_cpu的admission_blocked与执行中断；新增只读event_gap绑定交回index SHA，原生核验事件路线收据、统计条件/截止，仅train限量示例，val/test汇总。生产/学生合同不变，旧正式库/view保留；新460源清单尚未push。服务器同步发布后只按SERVER_V52_FULLFLOW第0步读取既有回放，交v52_ue5_gap_20261012.json，不重跑五小时建库/readiness/旧fit，不训练。
本机tmux单CPU12训练路线1634帧/73.95秒/58304KiB，proceed89NO/51UNKNOWN；16实例11后方截止/3运动截止/2源不连续。实际三联系表30不同RGB、90源SHA，非G2；未证实可直接改YES，标签阈值提示词不改。真实原生5路线集合中1条UE5新诊断核验通过；382联合审计测试7.30秒通过（先行10专项不相加）。证据checkpoints/fullflow_received_20261012，报告audit_joint/FULLPOOL_RECEIVED_20261012.md。服务器完整数据未传本机，核验交接证据非全库独立重载。Phase3/Action/联合G0仍未完成，无GPU/训练/push。

### 2026-10-12 服务器pilot通过与全自动开训收尾

收到Downloads/pilot_report.zip 30799字节/6成员，外层与成员SHA及454发布源码核验通过。服务器30路线3858帧/236.74秒，原生回放104.6秒、gzip合计1966539字节；正式候选8614路线1062401帧。样本只覆盖7事件，未含UE3/5/6不能推断全源无支持；pilot关闭，不重复readiness/旧fit。
修复rule_baseline首模式失败吞掉另一模式诊断；新增server_finish.sh/.py和first_epoch_budget.py（原生meta模块+同LoRA目标选择器计算rank16/vision-off两模式首轮预算，不加载权重/GPU）。新458源清单；正式池CPU入口复用30回放→完成全池→单份共享RGB4规则库→自动留出→短视图→预算/准入→≤30MB包，20GiB保留/约12GiB净写盘轮询上限；不调用人工总入口、不启动训练。原生v52/老师v17/标定提示词不变，学生准入合同更新需新view。发布同步后按audit_joint/SERVER_V52_FULLFLOW_20261012.md执行，交fullflow_report.zip+fullflow_package.json，只有admitted_rule_baseline才两模式各一轮。
本机12路线1335帧真实链路83.36秒/598512KiB，规则train498、自动val48/test75；train缺RE5、val缺9/test缺8事件，两模式一致，按预期阻塞。预算使用tiny config测试夹具，不冒称服务器4B预算。实际三联系表18不同训练RGB/54源SHA（UE4/UE7/RE3），无新确定错标/盲审；雾天远端可见性仍有限制。2311完整回归103.90秒通过；收尾8工具专项通过含1新增回执测试。见FULLFLOW_REVIEW_20261012.md与fullflow_verification_20261012.json；checkpoints/fullflow_audit_20261012存原始证据。Phase3/Action缺失及联合隔离未完成，G0未过/G2未运行；未远程/GPU/训练/push。


### 2026-10-12 v52白名单发布准备与服务器限量试建

用户授权push。新增server_v52_pilot.sh及454源SHA清单：正式候选扫描后最多30条/每条≤400帧，split/scenario轮转、单worker gzip；不编译封存部分回放/不训练。20GiB余量+约2GiB净写盘轮询保护，≤30MB ZIP及独立SHA回执；样本不保证全量容量/事件覆盖。服务器沿sync_automot.sh覆盖同步，仅运行该新入口，交pilot_report.zip+pilot_package.json；readiness/旧fit不重跑。详见audit_joint/SERVER_V52_NEXT_20261012.md。
本机两UE7路线107帧重算11.23秒/123172KiB；P4语义及P3动作证据与v50同范围一致。实际两联系表28不同RGB+两个已包含原图，84源SHA；Town12 002113先解除后稳定、Town15 002214先稳定后解除，分别48/52才complete YES，保留并发身份和雾天观察限制，非G2。37专项通过含5新测试；原生试建CLI真实2050字节包通过，shell因本机管道core门退出2仍产25606字节包/454源SHA通过。补自动训练脚本core软限制后新建view_release，2/4各464train/空留出。见RELEASE_RGB_V52_20261012.md及release_rgb_v52_verification_20261012.json；完整产物checkpoints/v52_release_rgb_20261012，未远程/GPU/训练，未新改标签/阈值/提示词。


### 2026-10-12 v52累计回放索引、自动首轮门与RGB续审

修复分批produce覆盖index：写锁内核验累计回执、按候选顺序合并，仅计算未登记路线。共享安装同锁封存published.json；已有子集请求不重写，已发布集合禁止扩展/需新目录，旧产物不改。合同v52/老师算法v17不变/同预约v32/空registry v27。固定自动入口rgb_short/run_automatic.sh：rule_baseline先核验实际过滤后的规则train/留出、十事件等量、world1/4及两模式同序、五关键边readiness正反支持、2模式各1检查点与20GiB首轮预算，再四卡一轮10240/重载/本次epoch000固定test；不走原生人工双库总入口。预算不认证操作者上限或所有历史入口。
本机真实5路线2+3累计5，封存后2/3/5子集重复全字节不变、共享库仍可载；755帧1116问题/P3动作与v50语义一致。只建RGB4及短视图，两模式各464train/空val-test，自动准入按预期empty val/test+missing budget阻塞。68.87秒/262300KiB；2302完整回归121.89秒，追加4入口测试（1新增）通过，最终学生视图重建于view_final。实际5联系表70不同RGB、210源SHA，UE5十帧参与者跟踪30源与前者去重219；3752/3753/3754不同车，ParkingCutIn后继U-E1不回填旧U-E3正例。部分窗口与历史重叠，非G2/全源认证，标签阈值提示词不改。见audit_joint/BATCH_BASELINE_V52_20261012.md及batch_baseline_v52_verification_20261012.json；产物checkpoints/batch_rgb_v52_20261012。服务器readiness已关闭/不重跑旧fit；正式池、真实首轮预算及服务器新训练仍待交付，无远程/GPU/push。


### 2026-10-12 v51开训代码复审与索引覆盖反例

本机完整CPU回归2294通过/105.92秒；v51合同与记录一致，10证据SHA及210源SHA通过。实际复看既有5联系表70不同RGB，无新增独特帧/盲审/新错标。真实produce在独立副本复现5路线index被子集请求改为2、再下一批改为3，完整性丢失且既有共享引用拒绝加载；原产物未改，缺陷本轮只报告未修。原生full_pipeline/train.sh/preflight仍按原val/test准入，自动参考未统一；首轮可固定采用RGB4库→短视图→rule_consistency入口，不必扩大改造原生总入口。正式池、关键条件覆盖和首轮容量上限仍待交付；服务器readiness保持关闭。见AutoMoT/qwen3vl_local/audit_joint/V51_CODE_REVIEW_20261012.md及v51_code_review_verification_20261012.json，复现副本/日志在checkpoints/v51_code_review_20261012。未改生产/规则/提示词，无全池构建、远程、GPU、训练或push。

### 2026-10-12 v51共享回放、原生gzip与RGB续审

新库默认共享production引用（逻辑/解析路径+index/逐路线SHA），缺失、漂移、同字节软链接改指拒绝；短视图/pipeline/统计绑定同步。原生CLI/full_pipeline流式gzip6、无未压缩临时副本；旧目录编码/合同变化禁止覆盖；显式copy保留。合同v51/老师算法v17不变/同预约v31/空registry v26，旧资产不改hash。新增training_budget逐项上限与证据，含两模式每轮adapter/optimizer、自动留出/P3索引/临时峰值/20GiB；缺项sufficient=null退出2，不用小样本压缩率保证全池空间。
最终tmux低优先级单CPU5训练路线755帧1116原始题，与v50状态/答案/证据及P3动作一致，两模式原生464条完整配对/建载通过，6事件空val-test/trainable=false。回放14,251,070→499,958字节，RGB4引用库6,558,759字节，共享回放另计；64.84秒/237348KiB。实际5联系表70不同RGB/1原图含在70内、210源SHA，扩展UE2/3/5/6/7前史；没有足够新证据改标签/阈值/提示词，非G2。2292完整回归118.23秒+gzip留出/抽样4专项通过（2新增）。见audit_joint/STORAGE_RGB_V51_20261012.md及storage_rgb_v51_verification_20261012.json；最终产物checkpoints/storage_rgb_v51_final_20261012，首轮独立产物保留。服务器CPU已关闭，不再readiness/空间盘点/旧拟合；正式池/自动留出和全源预算仍待补，无远程/GPU/训练/push。


### 2026-10-12 服务器CPU正式反馈通过与原生存储实测

收到08:24启动/08:31交回readiness3包：53,866字节/7成员及外层SHA通过，440源SHA、日志111测试文件清单逐项一致；服务器2262通过/3历史本地产物依赖跳过/9543弃用告警，165.52秒，source/pytest/preflight均0。额外6文件保留，覆盖同步与CPU入口问题关闭，不要求再readiness/空间盘点/旧拟合。29资产记录仍旧v42未认证元数据，当前45.700GiB余量，不冒称新正式池/GPU训练。
本机tmux低优先级单CPU复用v50的5路线既有回放，原生build/load464条RGB4规则题，7.34秒/102712KiB；库20,805,713字节，其中14,268,323为SHA相同、inode不同的production副本。逐文件gzip6诊断合计1,243,439字节且还原一致，仅5路线不外推；小库6事件/空val-test/trainable=false，不当正式库。新复制式建库按旧规模超可新增预算约13.432GiB，优先新产物单份回放/压缩及完整峰值预算，存储实现尚未改。见audit_joint/CPU_PASS_TRAINING_AUDIT_20261012.md及server_cpu_pass_verification_20261012.json，原包/编译产物checkpoints/server_cpu_pass_received_20261012。生产/提示词/标签未改，无新增RGB目视/标定回放/远程/GPU/push。


### 2026-10-12 覆盖同步适配：发布测试清单与导入检查

用户确认sync_automot.sh使用正确仓库、rsync -a --checksum或cp覆盖白名单且保留本地其它文件；440文件SHA已证同步成功，额外文件不是同步失败。修正此前readiness2的全目录额外文件硬阻断：readiness3逐文件核验发布SHA，仅运行清单中已发布CPU测试，关闭本地pytest配置/conftest/自动插件扩展，审计进程拒绝受检包清单外模块/另一目录模块导入；额外文件记录并保留，不改sync或要求新目录。外部依赖/子进程不冒充全环境认证，生产v50/标定/提示词不改。
15项专项通过；完整隔离副本保留服务器6个额外文件名及额外conftest哨兵，实际入口2264通过/1项本地历史collection依赖跳过，101.76秒退出0，26,892字节/6成员诊断包通过。日志SHA见audit_joint/overlay_readiness_verification_20261012.json，原始产物checkpoints/server_readiness_overlay_20261012。新版发布后沿用原同步，在原AutoMoT运行server_readiness.sh交report.zip/package.json；不用重复空间盘点或旧拟合。当前报告SERVER_READINESS_REVIEW_20261012.md已修正部署建议，仍待正式池/新建存储预算/服务器GPU新基线，无新RGB目视或标定回放。


### 2026-10-12 第二份服务器CPU包与独立快照验收

51,716字节/7成员ZIP及外层SHA通过，440发布文件与本机一致；原应用目录额外6个Phase3 Python导致source_preflight=2、pytest_started=false，模型preflight=0。29资产记录仍旧v42未认证声明，当前45.713GiB空闲，留20后25.713，旧规模复制式新建约差13.419GiB，未确认新正式池/P3索引。不要重复原目录失败或旧拟合。
本机只复制440清单不能独立运行（27导入错误，清单非依赖闭包）；完整Git 93f05dd7a快照缺lead_data时67失败/4错误，补显式原始数据链接后2259通过/1本地历史collection依赖跳过，101.52秒退出0；前两次失败保留，不冒称2260全通过。完整Git快照+数据链接远程步骤见audit_joint/SERVER_READINESS_REVIEW_20261012.md，核验server_readiness_received_verification_20261012.json，原包/日志在checkpoints/server_readiness_received_20261012。生产/规则/提示词未改，无新RGB目视/回放/GPU/远程/push；事件等量及自动规则选优已接通，收尾集中干净CPU、存储减少复制、正式池覆盖与一次新基线。


### 2026-10-11 服务器混版前置检查与755帧RGB续审

server_readiness在pytest前检查440文件SHA及声明包中额外Python，混版不执行测试、不删文件，失败仍交两文件包；readiness2清单不改生产v50合同。4新增回归、完整2260通过103.57秒，真实隔离混版入口退出2/pytest未启动/50306字节包核验。服务器按SERVER_V50_START_20261011.md最新命令，真正新目录同步，沿用旧模型和资产绝对路径；同盘45.71GiB已知，不重复盘点、不建库/训练。
本机tmux单CPU5训练路线755帧1116题，45.81秒/162460KiB；原始语义与既有同范围一致，两模式各464语义配对。实际5联系表70不同RGB/3原图含在70内，210源SHA；非盲审/全路线目视。UE2 f92导航截止无return，UE3 f112恢复运动、UE5 f56目标到后方、UE6 f65出路口均截止不记完成；UE7 f54 complete YES仍有雾天/短输入可观察性限制，不改标。5回放gzip诊断14,250,605→499,687字节，逐字节还原，仅候选存储方案不外推全池。见audit_joint/LOCAL_RGB_FOLLOWUP_V50_20261011.md及local_rgb_followup_v50_verification_20261011.json；完整产物checkpoints/label_rgb_followup_v50_20261011，无GPU/远程/生产修改。

### 2026-10-11 v50白名单发布与U-E7补覆盖RGB审计

新增服务器只读server_readiness入口及440文件发布SHA清单，自动CPU回归/preflight/元数据盘点，≤30MB ZIP带回执；旧元数据不认证，失败也留包。用户授权push仅源码/轻量文档证据；checkpoint/权重/RGB/完整回放留本机。服务器立即按audit_joint/SERVER_V50_START_20261011.md同步后运行server_readiness.sh，交report.zip+package.json，未确认正式资产前不建库/训练。
本机14训练路线766帧2786题原生v50回放，原始语义与v49相同；两/四图各795条完整配对，U-E7 complete/readiness YES各13条/13物理组，旧13候选均保留；Town12/002115 f70仍不可比。原生两模式各隔离4条历史歧义题，不改答案；766帧P3动作证据另算，不强行跨阶段同义。实际7联系表80不同RGB帧、另2原图含在80内，240源SHA通过；雾天可观察性仍待审，非盲审/新批准，未合入正式全池/重建短视图。32.28s/401688KiB；完整2256回归通过100.06s。服务器脚本本机core管道门+缺基座退出2/1仍产26313字节已核验包，非服务器成功。生产v50合同/提示词未再变，G0未过/G1继续/G2未运行，无GPU/远程训练。见LABEL_RGB_V50_RELEASE_20261011.md及label_rgb_v50_release_verification_20261011.json；完整产物checkpoints/label_rgb_v50_release_20261011。

### 2026-10-11 v50生产等量评测、规则选优与历史隔离

按用户最新授权补生产三处接线：P3去重后固定十事件等量val/eval（INVALID另报）；P4原生/短视图训练、评测、重载绑定固定无放回清单，入口传固定十事件全集，缺事件提前失败。新增显式rule_consistency+teacher_references，仅规则train及物理组隔离自动留出选优，排除独立老师审核预约；不冒充独立准确率。已知路口同目标/输入的停车历史UNKNOWN/YES在弃权过滤前隔离，回执和加载拒绝重引入，不改答案/合并实例。任务v50、老师仍v17但SHA更新、同预约v30/空registry v25；旧数据/adapter须原合同，不改hash兼容。
本机2路线110帧394问题原生重放与v49原始语义相同，8歧义题（两模式各4）隔离，各编译96；6.36s/240640KiB，重引入反例拒绝。完整2250回归通过98.96s，含真实CPU小模型规则选优测试；新RGB目视0、无Qwen GPU/远程/全池重建/push。正式池及五项关键条件支持仍待补，局部val/test缺事件会被新版拒绝；G0未通过/G1继续/G2未运行。见AutoMoT/qwen3vl_local/audit_joint/TRAINING_SCOPE_20261011.md §9、production_wiring_v50_verification_20261011.json及NEXT_TESTS_20261011.md新步骤。完整本机证据checkpoints/production_wiring_v50_20261011/scope_replay，旧夹具/试跑失败日志保留。

### 2026-10-11 开训统计v2输入绑定与等量评测计划

只读training_scope v2绑定描述文件/父库/实际索引/逻辑路径/SHA，统计结束复查；Phase3显式缺索引missing、缺manifest incomplete并保留P4统计。原生核验后分报候选池/回放覆盖及五项关键自动readiness支持；新增不看答案、固定seed、物理组轮转、无放回等量评测ID清单，缺事件不生成部分清单，未接入训练选优。确认原生完整十事件macro缺事件返回null，评分器不改。v49局部七轮两模式/world1/4计划及train_cells与旧报告完全一致；候选82/回放47路线，五项关键格仍0、val缺6/test缺7。29.77s/245820KiB、9统计输入绑定通过；13新增回归、完整2223通过107.83s。未新增RGB目视/回放/标定修改/提示词/训练/GPU/push，两处路口历史准入仍未解决。见audit_joint/TRAINING_SCOPE_20261011.md §7及training_scope_v2_verification_20261011.json；本地产物checkpoints/training_scope_v2_20261011。服务器交回当前正式池v2统计，不重跑旧fit；G1继续/G2未运行。

### 2026-10-11 开训收敛、Action均衡参考与十事件续审

用户仅要求事件间1∶1，不要求YES/NO等额；标定问题只报告。本轮新增只读training_scope，复用原生加载/采样及Action权重。v49局部3263/239/123，七轮RGB2/4与world1/4同序、各事件1024；val缺6/test缺7事件，U-E3仅27题/单题最多38次；规则U-E3/5/6缺proceed正例、U-E7缺complete正例，不推断全源。Phase3原生生成去重可破坏题数等额，binary/choice实测回归；正式索引缺失。tmux10路线1579帧120.76秒/211916KiB，492P3候选/2874P4问题与旧v49同范围逐条一致；实际12联系表82不同RGB/12路线、246源SHA/40学生像素通过，非盲审。新增10测试、完整2210项通过（107.57s）；无生产/提示词/标签修改、GPU/远程/push。E1–E6等不阻塞开训，收尾集中当前正式池、历史准入与均衡评测缺口、一次服务器新基线。见AutoMoT/qwen3vl_local/audit_joint/TRAINING_SCOPE_20261011.md及training_scope_verification_20261011.json；产物checkpoints/training_scope_final_20261011及training_scope_rgb_20261011。G1继续/G2未运行。

### 2026-10-11 v49并发路口身份与RGB续审

新增junction_review只读回执/帧全集/老师合同/源SHA与逻辑目标绑定、真实渲染输入比较。18路线1023帧中15路线198并发帧253对（有效同路口217/未知36），470同目标重复输入组、2 UNKNOWN/YES弃权分歧、0二元矛盾；不自动归并。Town15 002210 f45/002091 f44旧实例保留停车NO而新锚点无历史，同RGB提示词分别catchup UNKNOWN/readiness YES，须准入复核，告警未接生产硬门。13保留complete正例均不在当前47路线池，弱必要门/风险窗口通过、严格批准0；另1实例截止不可比，未补训。实际13联系表117不同RGB/6路线，351源SHA；候选543源、审计1032源SHA，非盲审。19新测试/audit_joint315通过；6.73秒/239532KiB，未改生产/提示词/训练/push。见AutoMoT/qwen3vl_local/audit_joint/JUNCTION_RGB_REVIEW_V49_20261011.md及junction_rgb_review_v49_verification_20261011.json；最终产物checkpoints/junction_review_v49_20261011/audit_bound。G1继续/G2not_run，服务器仅CPU/盘点。


### 2026-10-11 v49路口生命周期与整实例RGB续审

原生teacher_replay先refresh_uncommitted再执行生命周期门，修正临时PROCEED使过期等待漏截止；已提交执行保持，提示词/阈值/P3动作不改。任务v49/老师v17/同预约v29/空registry v24，旧v48合同拒绝。47路线5847帧重算365.09s/369804KiB；9866原始题/目标和P3动作证据1588候选不变。原生2/4各3263/273/186、短视图3263/239/123；val/test各自与v48字节相同，train语义相同证据ID变；505YES/2758NO，world1/4及两模式10240题序一致、各事件1024、呈现2773YES/7467NO。2181完整回归105.23s通过，含2新生命周期测试。
另18诊断路线1023帧v48/v49原生回放3384→3294条：Town12 002115 f61重新运动截止旧等待，删除后继90条；Town07另41帧62→60，f40出路口截止。全部completed=false、无新YES/改NO。14旧weak complete YES中13同命题仍YES且两模型均NO，1旧RGB4正确题当前命题不再可比，不重算改善率。实际26联系表193不同RGB/19路线、579源SHA；非盲审/整路线目视。4歧义路线完整回放41并发帧，仍不合并自车/车辆/STOP锚点。最终仅bound_*回放可信，已断言装载路径/版本/SHA；初次缺依赖和错误装载版本试跑保留但排除。G1继续/G2not_run，无GPU/训练/push。见AutoMoT/qwen3vl_local/audit_joint/LABEL_AUDIT_V49_20261011.md及label_audit_v49_verification_20261011.json，完整产物checkpoints/joint_label_v49_20261011及ue7_complete_instances_v49_20261011。服务器只补CPU/资产盘点，不重做四组推理/不续训。

### 2026-10-11 U-E7拟合交接与逐帧RGB复核

收到1232文件U-E7包、原一轮1264文件SHA通过；四组723题/1446输出，采样/原始解析/检查点回执绑定通过。本机缺原视图/权重，internal_consistency_verified+source_binding verified，不冒称查询全集/实测GPU。complete YES命中2图1/29、4图3/29，proceed15/108、24/108；1853输入RGB文件及像素SHA通过。专用tmux旧/新各9路线476帧，31.47/31.19秒约72MiB，P3动作证据不变；实际15联系表142不同RGB/2原图（含在142内）、426源SHA，非盲审。104道覆盖题78规则不变、6变化、7不可比、13人工待审，619未回放；不改生产标签/提示词。新增fit_review保守实例/状态/阶段/RGB关联及物理组分层，修复跨Python浮点8ULP复验；22新测试/audit_joint296通过，两Python真实包通过。四组不重推/不续训，服务器仅补CPU/资产盘点，本地继续完成定义/路口身份/可观察性。G1继续/G2not_run。见AutoMoT/qwen3vl_local/audit_joint/UE7_FIT_RGB_REVIEW_20261011.md与ue7_fit_rgb_verification_20261011.json，完整产物checkpoints/ue7_fit_received_20261011；未GPU/训练/push。


### 2026-10-11 v48分阶段RGB与并发锚点复核

新增audit_joint/rgb_review.py，来源绑定的短两图锁定→四图锁定→因果来源揭示，早期屏不含参考/事实/预测，独占判断及阶段SHA链；明确开发流程不冒充G2。实际5训练题三阶段、29并发帧候选4路线；目视65不同RGB/8路线/20联系表，195源SHA。两个修复锚点投影稳定通过，U-E1普通反例有真实闭合/间距波动；UE7 f51几何已解但速度波动1.881不满足稳定门。v48最终train UE7 complete人工15YES/9NO、弱0YES/7NO，仅局部池。29歧义锚点在自车/横穿车/STOP间变化，待查重复路口身份，确认跨阶段匹配仍0。两/四图各1YES4UNKNOWN，来源阶段1YES2NO2UNKNOWN，不倒填或估计准确率。15新测试、audit_joint274通过；生产合同/标签未改、未GPU/训练/push。见AutoMoT/qwen3vl_local/audit_joint/RGB_REVIEW_V48_20261011.md及rgb_review_verification_20261011.json；产物checkpoints/rgb_review_v48_20261011。服务器仍按NEXT_TESTS旧合同推理/盘点，不续训。


### 2026-10-11 学生输入与老师历史来源盘点

新增只读input_evidence，正式v48短视图3625语义题/7250两模式记录，精确文字/RGB身份、递归老师来源及模型输入SHA绑定。2832弱训练题证据包均含RGB2窗口前历史，RGB4有1392题；仅来源范围，不判必要性/错标/不可观察。431人工train与239val/123test缺原生老师包保留空，不否认原审核。172条件格分报题数/物理组，16新增测试、联合审计259通过；12.73秒/213816KiB。完整59MB逐题产物留checkpoints/input_evidence_v48_20261011_release；无新RGB复验/目视/GPU/训练，G1in_progress/G2not_run。见AutoMoT/qwen3vl_local/audit_joint/INPUT_EVIDENCE_20261011.md及input_evidence_verification_20261011.json。服务器仍按NEXT_TESTS_20261011.md执行CPU盘点和旧合同U-E7四组推理，暂不训练。

### 2026-10-11 白名单发布前验收

v48最终回执补齐，47路线5847帧、原生2/4各3263/273/186（431人工+2832弱）、短视图3263/239/123；旧v47合同拒绝，同预约/空registry及源SHA通过。新G1工具纳入联合回归2126项通过（106.74秒）；服务器下一步为CPU检查/资产盘点及旧合同U-E7 complete/proceed已采训练题推理，暂不训练。具体命令见AutoMoT/qwen3vl_local/audit_joint/NEXT_TESTS_20261011.md。白名单仅源码/轻量文档证据；完整数据、权重和checkpoints留本机。

### 2026-10-11 G1 只读时间线与条件区间

新增audit_joint/timeline.py，冻结映射/回执绑定、同录制同帧全外连接、事件存在性/未匹配队列及YES/NO/UNKNOWN区间。只读核验v47 45路线5673帧/8566题、v48 47路线5847帧/9866题，分别1650/1788区间；v48有174帧缺Phase3原标注保留未知。1007单实例同事件候选缺参与者证据、29并发歧义；确认匹配/可比命题均0，准入门不冒充最终题库。离线回执/状态同步不当真实许可；未知/停止问/源消失不改NO或完成。29新增专项、audit_joint 243项通过；约6秒/高水位约116MiB。G1in_progress/G2not_run，无生产修改、GPU、训练、RGB目视或push；原工作区v48生产任务成果另报。见AutoMoT/qwen3vl_local/audit_joint/G1_TIMELINE_20261011.md及timeline_verification_20261011.json，产物checkpoints/joint_timeline_delivery_20261011。

### 2026-10-11 拟合结果逐帧 RGB 标定复核

旧规则点查询237题/104路线/218锚点/934因果帧，tmux单进程28.77秒、RSS约161MiB；177弱跟车稳定目标复现。实际目视175不同RGB帧/10路线/19联系表及4原图，525源文件SHA；已知预测选窗，非G2。U-E1仅骑行跟车做路线投影，普通跟车仍用自车轴速度，弯道沿车道运动被横向门误判；v47相关函数与旧版一致。路线投影诊断筛出19候选（两路线18+1），未完整实例回放，不改为YES、不宣称15错题收益。人工UE1 f83与弱速度范围门有定义差异候选；UE7八道complete人工YES中六道不满足弱三帧速度门，仅必要条件对照，五道历史有负几何，不冒充完整老师NO。保留真实模型错例；P3同帧原始动作重算，68缺失保留null，未完成实例匹配。G1in_progress/G2not_run；生产规则/数据/标签未改，无GPU/训练/push。详见AutoMoT/qwen3vl_local/audit_joint/LABEL_FIT_RGB_REVIEW_20261011.md及label_fit_rgb_review_verification_20261011.json，产物checkpoints/label_fit_rgb_audit_20261011。

### 2026-10-11 已接收训练题拟合包并复算

本机拟合解压包1224文件、原一轮1264文件全集/SHA通过；独立runtime恢复128原run合同文件，无门绕过。两模式181题raw解析/指标/采样/检查点绑定通过，缺原完整视图仍为internal_consistency_verified+source_binding verified，全集完整性仅服务器回报；原ZIP/实时服务器/权重未本机验证。U-E1 complete/readiness训练YES87/NO94各呈现一次：RGB2正确115/181、YES75/87、NO40/94；RGB4正确101/181、YES82/87、NO19/94。NO52/94集中两物理组，非独立样本；4人工题全答YES（3NO错），实际看13不同输入RGB/4联系表，非盲审。U-E7 val56全NO，12YES全错，其中8道complete已有类别训练支持；20缺支持val含16答对NO，不能全归因缺边；未收到U-E7训练拟合。两fit工具SHA不同如实保留，当前已独立复算输出与来源，不声称两次执行代码完全一致。
不续七轮/不重复181题；本机标定及Phase3/Action跨任务隔离与正式池补建准备优先；服务器如需定位仅新增旧合同U-E7 complete/proceed训练题推理，不能用v47续旧adapter。未改训练/标定代码、未GPU/训练/push。见AutoMoT/qwen3vl_local/audit_joint/TRAIN_FIT_RECEIVED_20261011.md及train_fit_received_verification_20261011.json，产物checkpoints/train_fit_received_20261011。

2026-10-10服务器回报两模式181题拟合已verified，待本机接收ZIP复核，不要求重复推理；v47推送后先交包、旧环境保留、新源码独立worktree，具体见AutoMoT/qwen3vl_local/audit_joint/NEXT_AFTER_V47_20261010.md。

### 2026-10-10 v47切入历史、STOP同时间预测与上下文提示词

修复显式cut-in把历史走廊UNKNOWN当作在外，必须明确False；STOP横穿由起终点大包围盒改两轴同时间区间求交，保留AABB/2s匀速/可见性与停车义务。Phase4活跃提示词v11明确事件名称/导航目标仅为问题上下文、不作条件证据，冻结prompts.py及P3语义不改。任务v47/老师v15/同预约计划v27/空registry v22。
45训练路线5673帧tmux重算360.78秒/368764KiB，P3动作证据及1588候选、P4全部8566原始问题/准入目标与v46相同，无真实改标或模型收益声明。实际目视16不同RGB帧/4联系表，48源SHA；ParkingCutIn两窗口首次走廊相交f104/f35而lane_id不变，保留显式分支，通用召回仍缺。原生2/4各3082/273/186、431人工+2651弱，YES495/NO2587；完整配对及留出参考不变，仅输入SHA变更。
短视图46794字节、零production复制，2/4均3082/239/123；原生/短视图world1/4首轮10240题序一致、事件各1024、YES2763/NO7477。2078完整回归通过（新增20），旧v46合同拒绝，三阶段退出0。G1partial/G2notrun、199readiness缺格及六事件评估仍缺；无GPU/训练/push。旧服务器fit保留原源码/原视图。见AutoMoT/qwen3vl_local/audit_joint/LABEL_AUDIT_V47_20261010.md及label_audit_v47_verification_20261010.json；产物checkpoints/joint_label_v47_20261010。

### 2026-10-10 v46车道连续预测与条件提示词

车道预测分支改为有限折线逐段同时间两轴求交，复用原三帧身份/速度门，不推断连续转向或延长端点。活跃Phase4提示词v10明确条件合取、运动不等于许可、消失不等于清空、新限制不反证旧判断；冻结prompts.py不改，P3动作语义不改。任务v46/老师v14/同预约计划v26/空registry v21。
tmux45训练路线5673帧重算347.84秒/369228KiB；P3证据及1588候选、P4全部8566原始问题/准入目标均与v45相同，无真实改标或效果提升声明。实际目视14不同RGB帧（4联系表/2原图，原图含在14帧内）、42源SHA绑定，非盲审/新批准。原生2/4各3082/273/186、431人工+2651弱、YES495/NO2587，完整配对；留出仅model_input_sha256改变，不能称字节相同。
新短视图46794字节、零production复制，两模式3082/239/123、97曝光排除，RGB2[-2,0]/RGB4[-6,-4,-2,0]；原生及短视图world1/4首轮题序一致10240、每事件1024、YES2763/NO7477。2058完整回归通过（新增13）；旧v45合同拒绝，旧服务器fit须原源码/原视图。G1partial/G2notrun、199readiness缺格及六事件评估仍缺；无GPU/训练/push。见AutoMoT/qwen3vl_local/audit_joint/LABEL_AUDIT_V46_20261010.md及label_audit_v46_verification_20261010.json，产物checkpoints/joint_label_v46_20261010。

### 2026-10-10 v45连续预测与逐帧RGB复核

实际目视84不同RGB帧（11联系表、4原图），覆盖9个标签变化锚点及Town04路口归因复核；非盲审/完整路线/新批准。通用匀速预测改为有限走廊两轴同时间解析求交，修复半秒离散检查漏掉短占用；车道预测接收调用方2/4s时间窗。固定速度/朝向/宽度仍是弱假设，极短擦边及小目标待盲审，曲线车道分支仍离散。Town04 002163 f33由stop_sign_hazard转真导致优先权NO，corridor_clear仍true，不能据f31→33认定对向冲突过早放行；新transition_rechecks明确同实例复查筛查、原始/准入计数及右删失，不输出安全结论。
合同v45/老师v13/同预约计划v25/空registry v20；默认流水线/训练目录从遗留v43更新v45。tmux低优先级单进程45训练路线5673帧，351.60秒/369144KiB；P3原始动作证据及1588候选不变，P4原始8590→8566。原生2/4各3082train=431人工+2651弱、273val/186test，两模式留出各自与v44逐字节一致，3082完全配对。YES495/NO2587；epoch0 world1/4两模式题序一致10240，各事件1024，实际呈现YES2763/NO7477，仍非1∶1。2041完整回归+4新增筛查专项通过。旧v44合同拒绝加载，旧产物未改。G1 in_progress/G2 not_run/P3正式全池及六事件评估仍缺；无GPU/训练/push。见AutoMoT/qwen3vl_local/audit_joint/LABEL_AUDIT_V45_20261010.md、label_audit_v45_verification_20261010.json，产物checkpoints/joint_label_v45_20261010。

### 2026-10-10 v44十事件逐帧RGB续审与切入/路口/源消失修正

逐帧看109个不同RGB帧（13张联系表/放大图，非盲审），结合同帧P3动作与P4实例/几何。确认8个误建U-E3：Accident/ParkedObstacle/Construction/noScenarios的7车导出车道全程不变，“进入走廊”来自自车绕行规划或弯道足迹旋转；ParkingCutIn actor3697在f246只因常驻cut_in身份再建。修正：显式身份历史起点已在走廊不得再建；通用切入需固定本车道走廊靠近≥0.5m且导出车道不变即否决。U-E5决定性车辆连续两帧在自车后截止等待（不记完成）；路口U-E6/U-E7/R-E5自车近静止用4s预测（行驶仍2s），Town03 002133 f20–23过早proceed YES消除。共享源消失账本2→10条（8条前后原图确认、SHA绑定），Town03 8_107远处车不登记。
Phase4合同v44/老师v12/同预约计划v24/空registry v19。tmux单进程45训练路线5673帧重算350.94秒/RSS约361MiB；P3动作证据全帧不变，候选1637→1588（全为新隔离）；P4原始9844→8590，U-E3实例11→3，U-E7过早放行9/10→5/6，R-E5 6/9不变。原生2/4图库各train3084=431人工+2653弱、val273/test186，3084完全配对，RGB4 val/test与v43逐字节一致；YES503/NO2581，U-E3训练降为13/14，仍非1∶1/全池。相关回归2032通过。转弯穿越对向车、长实例隔离宽度、通用U-E3正例召回、G1/G2/P3全池仍缺；未GPU/续训/push。见AutoMoT/qwen3vl_local/audit_joint/LABEL_AUDIT_V44_20261010.md及label_audit_v44_verification_20261010.json，产物checkpoints/joint_label_v44_20261010。

### 2026-10-10 v43原生标定接线与十事件续审

Phase4新合同v43/老师v11/同预约计划v23/空严格registry v18；明确阻挡不再被另一项UNKNOWN抹去，许可合取未放宽。共享RGB消失隔离接入P3普通/缓存准入及P4原生编译/逐题人工准入/加载核验；6源SHA绑定，人工已UNKNOWN题仅增加隔离凭据。P3动作v23本体未改，mapping合同更新；旧v42快照/数据保留，旧adapter须原源码复验。
tmux单CPU重算45训练路线5673帧，368.65秒/RSS约362MiB；P3动作证据全帧相同，候选1638→1637；P4原始10920→9844，同状态44题hold/restrict UNKNOWN→YES，无新增放行YES，后继状态变化单列。176不同RGB帧局部目视覆盖十事件，非盲审/完整路线认证。原生2/4局部库均train3490=443人工+3047弱、val273/test186；3490完全配对，world1/4首轮题序相同，事件各1024但YES2642/NO7598。RGB4 val/test与旧v42参考逐字节一致。
相关完整回归2025通过，另真实P3扫描/缓存专项1通过。80原人工路线盘点、45路线回放，明确partial production，不冒充全池；P3正式全池、G0/G1实例匹配/G2及六事件独立评估仍缺。U3 actor3697在246新建二次切入身份待核，不能误称104旧实例未结束。未GPU/续训/新批准/push。见AutoMoT/qwen3vl_local/audit_joint/NATIVE_LABEL_FIXES_V43_20261010.md和native_label_v43_verification_20261010.json；原生题库在checkpoints/joint_label_v43_20261010/data2、data4。

### 2026-10-10 十事件本机逐帧标定续审与源异常隔离

本机tmux单进程重算45既有训练路线/5673帧，1638条P3原生候选、10920条P4原始问题；新增247不同RGB帧局部目视及4原图复核，非完整路线/盲审/独立批准。确认RE3 f58→59 actor4745、UE6 f43→44 actor138可见车辆消失，六份RGB/meta/bbox SHA绑定。局部隔离保留明确当前STOP：粗筛8条收窄为1条P3；P4原始6条、实际可保留弱候选4条。最终P3 1637、P4两模式各3585，仅诊断候选不冒充正式库。默认v42生产/学生合同及P3 mapping未改，正式建库尚未接入隔离；无GPU/续训/全池/新批准/push。
新增audit_joint/local_replay.py、label_quarantine.py、label_inventory.py及tmux入口；全帧请求、源SHA/回执/隔离范围和明确UNKNOWN保留。两次完整重算P4逐字节/P3动作证据一致，480.02秒/约383MiB、CLI362.83秒/约364MiB；14新专项，audit_joint207项通过。详情AutoMoT/qwen3vl_local/audit_joint/TEN_EVENT_LOCAL_LABEL_AUDIT_20261010.md，总方案§14.1保持G1 in_progress/G2 not_run。本机标定主线不等待服务器训练例拟合。

### 2026-10-10 拟合验收绑定原始采样与原始输出

verify-fit新增显式--run或--package/--result，绑定原源码/日志/采样/检查点回执SHA；完整验收另需--dataset原视图，从原池重算请求事件/边/阶段的已采ID全集、逐题身份和曝光次数。只有结果自洽或缺视图时返回internal_consistency_verified、退出2；显式来源错误直接failed，不猜服务器路径或降级绕过。已生成拟合结果可复验，不要求重新推理。
evidence、fit及verify-fit均用冻结parse_answer重解析raw，失败映射MALFORMED，缺文本或prediction矛盾拒绝。新增22项反例/边界，拟合专项49项、相关完整回归1429项通过；真实接收478条val原始输出复解析通过，112/239支持缺口不变。学生/生产合同未改，未GPU推理/训练；服务器命令更新见AutoMoT/qwen3vl_local/audit_joint/TRAIN_FIT_20261010.md，核验见同目录train_fit_verification_20261010.json。

### 2026-10-10 覆盖与已采训练题拟合审计入口

新增audit_joint/train_fit.py独立工具，避免修改冻结学生/生产源码而失效现有adapter。support先核验接收包，再核对全量val身份、有序采样绑定、逐rank实际执行及更新记录和完整池汇总；实跑再次复现两模式112/239缺类别支持、U-E1 complete/readiness 87YES/94NO已呈现，不能据此推断训练拟合。
fit复用一轮checkpoint与原视图，只选真实已采训练ID，同题重复只推理一次并分报不同题/曝光权重；默认U-E1 complete/readiness全部181题，按状态/纵向/分支/运动/来源分层，与完整池/val状态支持对照。加载前核验模型身份，部分推理失败留阶段和已完成cases；无样本显式not_run，不把假设状态当监督。新增verify-fit独立核对所选全集/身份/结果SHA并重算指标；采样目标一致性、val指标复算及同字节软链接改指/权重漂移均设失败门，Ctrl-C退出130。新增结果进30MB白名单，权重仍排除。
相关完整回归1407通过，含新增专项27项；接收两模式student/producer合同与本机逐字节一致。本轮未实际GPU推理、训练、G2盲审或修改源几何/可见性门；负尺寸及夜间门只保留待证问题。现有data4/view/adapter无需重建；先服务器181题拟合，不续七轮。见audit_joint/TRAIN_FIT_20261010.md、train_fit_verification_20261010.json。

### 2026-10-10 本机逐帧审计与一轮训练支持缺口

收到Phase4完整一轮解压交接1264文件：两模式10240呈现、各rank2560前向/320更新，239道val保存前后预测及原始输出一致；仅核验服务器证据，本机未加载权重、原ZIP未核验。U-E7全NO；RGB4 U-E1九道NO误答YES分proceed2/release2/complete5。val中112题的事件×边×阶段×答案在训练池零支持（UE5 60、UE6 32、UE7 20），不等于112个独立实例或缺覆盖必致错；UE1 complete/readiness的87YES/94NO全采到仍有5道NO误答YES。
本机tmux低优先级单进程重算11开发路线792帧，374条P3原生候选/356锚点、1772条P4原始2/4图问题；候选未完成正式split覆盖和准入，不能直接训练。核验24文件回执/792RGB SHA；实际看4完整序列300帧联系表、7原图，共301不同帧；非盲审/独立批准。主任务56.47秒/RSS约131MiB；第二遍21.17秒/81MiB，均退出0。
定位UE7 release状态域/catchup UNKNOWN门；Town05/002062全43帧及002066末8帧负静态extent，原生22米走廊诊断包络不相交仍不能认证几何；UE1夜间亮度门有欠覆盖候选。160条UE7 proceed原条件复现，假设PROCEED/HOLD的release为15YES/145NO，仅诊断不入库。G1/E0/S0前置/S4效果部分推进，G0未通过、G2未运行。未改规则/旧标签/split/采样，无续训或CARLA，产物仅checkpoints。详见AutoMoT/qwen3vl_local/audit_joint/LOCAL_FRAME_AUDIT_20261010.md和总方案唯一进度表。

### 2026-10-09 重载失败报告与交接补齐

重载前创建独立报告目录；逐阶段原子写入in_progress，资产/优化器漂移、加载或验证异常记录failed、失败阶段、错误及已核验身份，非零退出。Ctrl-C退出130；强制终止可能只留in_progress，不计通过；既有报告拒绝覆盖，外层latest仅成功更新。失败报告进入原交接白名单，权重仍排除。
相关回归1379项通过（退出0）；另复验已有latest保留3项通过。补测真实验证循环第二题失败、模拟加载/资产异常、重试保护及ZIP字节核验。首次专项在14项通过后退出139，原因未定；开启故障追踪的专项重跑与完整回归均正常退出。学生源码合同变化仍只需从现有data4另建小视图；原生产合同保持。未执行服务器/GPU/4B重载。执行与限制见rgb_short/NEXT_RUN_20261009.md，复核记录见rgb_short/reload_failure_verification_20261009.json。

### 2026-10-09 短跑实审与完整一轮运行证据

接收phase4_short_smoke_handoff_20261009.audit：1244文件及绑定通过，两组源码/基座/视图一致；239道val完整身份和指标独立复算吻合。RGB2一致率52.30%、YES召回11.32%；RGB4全NO、一致率55.65%、YES召回0%。两组40次呈现、按配置2次更新，不能证明学习效果。原ZIP压缩大小、实时服务器输入及权重重载未本机复验，G0为历史缺失态未更新。
新增原始弱监督计数、全局/各rank有序采样绑定、四rank逐微步loss/耗时/更新记录、训练峰值显存及验证保存耗时。逐事件YES/NO召回、常量回答与多数类基线诊断不改原选优指标。保存权重/优化器身份，launcher结束后新进程重载最后adapter并完整val比对，失败保留证据且不更新latest；新增记录纳入30MB打包白名单，权重仍排除。
相关完整回归1376通过；CPU真实optimizer记录、模拟重载/资产漂移/预测不一致、模拟四卡启动失败路径通过。原Phase4生产合同完全不变；学生源码合同变更须从现有data4生成新小视图，旧短跑权重不续用，不重新回放/复制production。下一步两模式各完整预算1轮（10240次/四卡320更新），审核后再续至总计7轮。未执行本轮服务器/GPU/4B重载或优化器恢复实验。操作见sft_new_loop_phase4/rgb_short/NEXT_RUN_20261009.md，验证见diagnostics_verification_20261009.json。

### 2026-10-09 短两图输入与可复用四图库实验入口

用户确认Phase3/Phase4新RGB2均为[-2,0]（4Hz下0.5秒），各自RGB4保持[-3,-2,-1,0]/[-6,-4,-2,0]；历史两图名称不改义。Phase3新增2rgb_short并接通训练/评测/打包，短输入adapter须有明确合同；Phase4新增独立rgb_short入口，原冻结生产/老师合同与已收服务器recovery报告完全一致。
只从已核验full_production编译一份原生四图库，再以小视图选两图；完整四图库已存在时零production复制。两模式同题集排除已知开发曝光留出与同split短输入相反标签，不以test答案改变train选题。新短输入仅继承实验参考目标，不继承四图人工可见性/老师批准，评测分报参考一致率，联合独立性仍false。
本机原生人工题443/273/186，经共同筛选443/239/123，97题因已知开发曝光排除；805题真实RGB身份核验及world1/4两模式题序一致，小视图45620字节。相关完整回归1361通过；最终短输入专项35通过（含真实小模型CPU反传、四图编码等价、绑定漂移、真实建库及模拟四卡启动），未服务器/GPU训练或全量回放。新入口20GiB守护，服务器下一步准备四图库/视图→CPU采样→四卡40次呈现短跑；P3索引另行重建，E0/E1/G1/G2及缺失资产仍未完成。操作见sft_new_loop_phase4/rgb_short/README.md，证据见同目录verification_20261009.json；本轮不删旧产物。

### 2026-10-09 首份服务器交接实审与只读复用检查

收到checkpoints/capture.audit解压目录，1216文件大小/SHA/全集及包内回执绑定通过；原ZIP大小、服务器实时55项外部引用未在本机复验。44份冻结曝光/预约声明复算用途矩阵与路线成员一致，3对冲突涉及10物理组，为Phase3开发曝光与Phase4留出预约重叠，非已证实际训练泄漏；G0仍in_progress，9项全池、P3权重及执行证据缺失，E0/E1/G1/G2未运行。服务器旧HEAD但1170共有源码与0279ac735逐字节相同，另39服务器独有；不得据HEAD猜运行版本或清理工作树。
verify-package支持解压目录及跨清单绑定，拒绝链接/特殊文件。新增recovery-check只读核验原生请求/候选/registry/全路线回执，输出题库外新JSON；不建库、删除、改名或修改生产合同。原生续建会保留残data2再编译，每套复制production；空间只报下限，不承诺77GB足够。新增22项测试，相关1257项通过；本轮无远程执行/GPU/全量生产。结论、10组名单及服务器下一步见audit_joint/REMOTE_CAPTURE_REVIEW_20261009.md，机器证据见received_capture_verification_20261009.json。

### 2026-10-09 联合审计Action隐式候选依赖

依赖条件对齐原生annotate_tokens：token开启或action_balanced；登记candidate及同目录manifest/counts/frame_index，均SHA核验，未读取候选的event_balanced/token关闭不强加依赖。v3格式保持但源码合同更新，旧小索引须另目录重导，不重建原始题库。真实read_rows/原生候选校验/真实合同四组合及跨cwd、split搬迁、四文件删除和改字节测试；无GPU/远程/全量重放。另修复测试读取已退出worker的/proc竞态，guard生产代码未变。新增4组合集成测试不替换原生读取器/合同，相关回归1235通过（G0共122）；验证见audit_joint/candidate_dependencies_verification_20261009.json。

### 2026-10-09 联合审计图片失败保留证据与Action路径基准

原生图片核验失败记certification_errors并继续完整扫描，路线用途/疑似冲突/候选排除保留；不计完整池及人工支持，image_hashes_verified=false，记录检查/失败行数。索引结构损坏仍不提交部分来源。Action导出v3保留argv，登记export_cwd/resolved_args/dependency_paths，换cwd及原始split搬迁按登记路径与SHA复验；旧缺基准记录拒绝猜测，需新目录重导小索引。Phase4生产/原始题库不变，无GPU/远程或全量重放。新增15项回归，相关全套1231通过（G0共118）；真实合成RGB经原生加载器核验，Action跨cwd/搬迁/token候选及旧记录拒绝通过。详见audit_joint/images_paths_verification_20261009.json。

### 2026-10-09 联合审计仓库与应用目录识别

prepare以模块位置查询Git根，识别仓库/AutoMoT和扁平应用两种布局；project_root/application_root分开，默认data/source/checkpoint随应用根。显式根目录歧义拒绝，不以cwd猜测。Git缺失/无HEAD/超时转git_identity阻塞并留诊断包，不再晚期崩溃。旧错误请求不重写，须新prepare/输出；服务器手册新增缺失资产完整分支。生产/Action导出合同不变，无题库重建/训练/远程执行。新增13项布局/Git专项，相关回归1216通过（G0共103）；扁平/嵌套×应用/无关cwd四组合真实CLI及Git、阻塞捕获/ZIP通过，worktree/no-Git/旧错误root均覆盖；本机实际prepare路径核对通过。见audit_joint/layout_verification_20261009.json。

### 2026-10-09 联合审计原始池绑定、逻辑引用与未认证证据

Action有效池按本次action_raw三split内容SHA绑定，旧请求也从原始来源提取路径；允许原始索引搬迁并核验相同字节，无须保留旧索引目录，RGB/其它依赖仍按原合同检查。轻量引用同时登记逻辑路径/解析目标/SHA，verify-inputs分报target_status与binding_status；同字节改指仍失败，旧记录无逻辑路径为incomplete，不改历史回执。
未认证但完整可读索引保留路线用途和带source_status的疑似冲突，禁止贡献完整池/人工支持；相关候选排除，缺RGB目录也保留逻辑组证据。损坏索引仍不提交部分来源。生产合同/标签不变；无服务器、GPU或全量重放。新增16项专项，完整相关回归1203通过、最终G0专项90通过；现有8058训练配对、66项引用目标与逻辑绑定核验通过，阻塞态ZIP8317688字节。缺模型/5对冲突仍在，验证详情见audit_joint/followup_verification_20261009.json。

### 2026-10-09 联合审计六项边界修复

resolved-route别名传递归并参与跨任务冲突/候选排除，图像重复不当路线等价；Phase3/Action/候选实际data_root接线。guard主进程退出后清理本进程组残留并保留退出码。verify-inputs覆盖已登记基座资产与资产集合，缺模型incomplete、漂移failed。
Action默认/显式路径统一v2导出合同及实际准入门，覆盖稳定Phase3外部依赖、时长规则与原始路线帧数/判定；旧v1需另目录重新导出小索引，不重写旧hash。损坏manifest/metadata转invalid诊断并可打包。两/四图库不同目录、模式、绑定及全训练语义/共享RGB流式配对核对。Phase4生产合同未改；服务器仅见18GB回放/13GB data2，未认定完整，不重建或删除30GB。

相关回归1187项、最终G0专项74项通过（新增28项）；本机8058对训练题完整配对通过，原生产合同保持。新版阻塞态G0的66项外部引用、回执及8309301字节ZIP核验通过；模型缺失明确incomplete，5对用途冲突仍保留，E0/E1/G1/G2未运行。证据：`AutoMoT/qwen3vl_local/audit_joint/boundary_verification_20261009.json`。

### 2026-10-09 联合审计低磁盘占用与启动保护

G0默认references：冻结源码/小JSON，完整索引及adapter只记原路径/大小/SHA；full输入副本须显式选择，原数据必须保留，verify-inputs核对引用漂移。新增storage-plan预估及storage-inventory只读盘点，绝不自动删core/模型/历史产物。run_guarded.sh每次设core软硬限制0，管道收集器拒绝启动；每秒检查指定文件系统余量，低于预留仅停止本次进程组，周期检查非硬配额。
服务器手册先盘点/复用已有题库，不再默认新时间戳全量建库；缺库先报missing，不冒称完整验收。原Phase4逐帧生产格式/标签/合同未改，仍可能有大产物；30MB仅ZIP上限。新增实现和本机验证见audit_joint；无服务器清理、core崩溃根因认定或新GPU训练。
本机同一批输入完整快照296575294字节→轻量捕获28697111字节；66项外部引用SHA通过，除存储字段外隔离报告相同、账本逐字节相同、原合同不变；ZIP8296154字节及旧/新回执通过。新增10项存储/保护测试，相关回归87通过。本机管道core收集器确实在启动前被拒绝，未制造崩溃或修改系统设置。

### 2026-10-08 联合审计自动交接包（30 MB）

G0 run 在报告/回执完成后自动生成同级 .audit.zip，阻塞态仍先打包再返回2；pack 支持旧捕获及显式指定训练/评测目录，verify-package 核验文件全集/SHA。压缩后硬限30000000字节，完整报告/逐题结果不截断；超限仅可登记省略源码正文，核心仍超限拒绝发布。数据/权重/RGB不入包，包与本机完整快照留checkpoints，不自动上传/入Git；额外结果不自动升级G0或E0/E1状态。
36项G0/打包测试及扩展相关回归77项通过；四套历史Phase3结果合包12486935字节；旧真实捕获ZIP为8271720字节（含冻结源码），旧回执/ZIP核验通过；无新训练、标签或GPU实验。服务器手册第4/6步说明自动交接与训练评测合包。

### 2026-10-08 联合 G0 审计入口与服务器交接

新增 qwen3vl_local/audit_joint（既有 qwen3vl_local 目录白名单内）：实际源码本机快照、原合同/SHA核验、跨 Phase3/Phase4/Action 物理组用途矩阵、原生 Action 采样前池导出和开发val候选排除；不改旧split/标签/训练默认。66项相关测试通过，推送前兼容环境完整回归1138通过；本机v42两图四图各8058/273/186题源/像素核验及快照回执通过，1152源码文件快照留checkpoints，不push。
可用清单发现13个物理组/5对用途交集；全Phase3/Action池与权重缺失，独立性未通过。G0仍in_progress，E0/E1/G1/G2未执行；服务器先补全量v42题库及G0，不将旧七轮弱训练当新方案性能验收。见 AutoMoT/qwen3vl_local/audit_joint/SERVER_RUN_20261008.md 与总方案唯一进度表。

### 2026-10-08 Phase4 v42 事件身份、运动分层与弯道跟车

核对外部v41审计并修正：静止NO后移动才释放记catchup候选/UNKNOWN，不伪造视觉后继；七类YIELD执行后截止过期等待，横向WAIT不凭前进放行。UE5原车道/上游侵入身份，UE6须异常优先权身份，UE3区分普通路口汇入；动态ID不直接当静态障碍。跟车按路线切线与间距/闭合TTC，平行车沿道路预测。
默认10240/轮、各事件1024保持；主边权重8/恢复4/其他1，单答案边1；答案×切片×运动容量分层，不靠无限重复补缺格。1090专项通过；140训练路线15372帧程序回放，2/4各train8058=443人工+7615弱题，8058完全配对，world1/4各七轮题序一致；val273/test186/review539与v41逐字节保持。
实际16联系表127RGB帧，一条47帧完整序列、五条局部；六组核验train并显式登记，无独立目视/新人工标签。InvadingTurn具名路线确有锥桶，不能全删UE2；正常对向车误UE5、普通红灯队列误UE6已在具名回放消除。
合同v42/老师v10/同预约v22/空registry v17；run.sh自动全路线新建匹配库，旧hash不改。未全量v42重放、GPU/CARLA；140路线中UE5推进自动YES为0、UE2 return及UE4 complete YES仍缺，冻结导航耗尽仍截止，不能声称召回/十事件分支或独立准确率完备。见Phase4/TEACHER_AUDIT_V42.md及teacher_v42_verification_20261008.json。

### 2026-10-08 Phase4 v41 十事件自动弱老师接线

新增UE2/UE3/UE5/UE6/UE7/RE2/RE3/RE5因果规则、实例检索、离线执行状态同步、横向可见范围入库；通用规则面向全部合格路线，非开发路线白名单。障碍当前规划仅定目标；固定世界走廊判切入；metas车道所有权不再要求bbox重复字段；正常跟车不当占道；故障信号身份与普通红灯分开，STOP/可见冲突仍约束。运行时BEV许可保持。
34训练路线/3601帧程序回放，新增八事件主边均有YES/NO；两图四图真实建库各train3071=443人工+2628弱题，val273/test186/review539保持，3071完全配对。1067专项通过，world1/4各七轮题序配对、事件各1024次保持，默认预算10240不变。
合同v41、老师v9、预约v21同物理组同split、严格registry v16仍空。五个当前RGB训练帧联系表抽看并显式train-only，无完整序列、独立目视或新人工标签。未全量v41重放、GPU/CARLA训练；run.sh会自动全路线重建。
返回规则有合成正例路径，本次34路线无return监督；自动不返回/本车道通过/骑行绕行分支抽取仍缺，不能宣称所有分支完备或独立准确率达标。见Phase4/TEN_EVENT_TEACHER_V41.md及ten_event_teacher_v41_verification_20261008.json。

### 2026-10-08 Phase4 v40 按Phase3比例抽样，替代默认逐轮全覆盖

用户澄清全量指通用规则扫描全部合格路线，不是每轮每题必见再大量重复；本条取代v37默认采样要求。run.sh/run_full_pipeline改phase3_balanced：每事件1024次，共10240；事件内按边/答案容量回流，完整事件池循环后再补余量，物理路线轮转及曝光历史恢复。四卡accum8每卡2560次计算、320次更新；train_full.sh只保留为旧策略显式入口。
仍全路线生产、不按路线截断、保持完整两图四图配对及冻结split；通用老师仅UE1/UE4，八事件人工题不冒称自动规则。当前小事件仍重复，不宣称无重复或七轮全覆盖。
对SHA核验的v38两库88986题实跑world1/4各七轮：每轮10240、首轮2304不同题、七轮7807不同题、最大单题49次，两图四图题序一致。1019专项通过。合同v40/预约v20/空registry v15，事实规则v8未变；未全量重建v40、未GPU训练/新增RGB审计。见Phase4/PHASE3_STYLE_SAMPLING_V40.md及配套verification JSON。

### 2026-10-07 Phase4 v39 外部软链接数据兼容

修复 candidate route escapes data root：数据路径按逻辑 scenario/route/frame 读取，允许场景、路线、子目录及文件链接到外部磁盘；扫描、回放、建库、审核、RGB加载统一处理。绝对路径/父目录穿越仍拒绝，帧身份与文件/像素SHA保留。
合同v39、预约v19、空严格registry v14绑定新源码；老师v8条件、独立预约路线及split不变。run.sh自动新建匹配题库，旧产物不改hash、不删除，日常命令与全覆盖/严格1:1/四图→两图保持。
完整回归1005项通过，追加老师入库四组合专项通过（含两项新增外部软链接组合）；18项路径专项包含于全套。真实人工两图/四图建库经外部路线链接后监督逐字节一致。无全量重放、GPU训练或新增人工RGB审计；详见Phase4/SYMLINK_DATA_V39.md及symlink_data_v39_verification_20261007.json。

### 2026-10-07 Phase4 无参数一行入口

新增 `Phase4/run.sh` + `auto_run.py`：自动选择兼容训练环境、模型及匹配的全量配对题库，按四图→两图串行完成训练/测试/打包；用户无需 DATA_DIR/MODEL_DIR/SKIP_BUILD 参数。缺题库自动生成，源码变化自动另选内容寻址目录，匹配 v38 时直接复用。
缺基座时仅准备阶段从 Qwen/Qwen3.5-4B 固定 revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a 下载模型资产（不下载远程代码）；本地缓存优先、训练评估继续离线、telemetry/隐式token禁用。已有不兼容模型不覆盖。
每轮全覆盖、严格事件呈现1:1、七轮和两图配对沿用已验收 v38；两个实验分目录，失败立即停止。外层只处理主机配置，单独记录启动源码 SHA，不改标签生产/源合同/旧manifest。实测现有 v38 request 与冻结源码保持一致。
15 项入口专项通过，含进程模拟与真实只读目录选择；本轮无权重下载、GPU训练或新RGB审计。日常命令：在 AutoMoT 下 `bash qwen3vl_local/sft_new_loop_phase4/run.sh`。见 Phase4/QUICKSTART.md。


### 2026-10-07 Phase4 v38 一条命令全量流水线

run_full_pipeline默认全量配对建库→train_full严格事件呈现1:1→完整人工test→audit.zip；HISTORY_RGB_MODE=4rgb/2rgb_endpoints、SKIP_BUILD/SKIP_TRAIN/SKIP_EVAL、TRAIN_MODE/EPOCHS/ACCUMULATION兼容Phase3用法。Phase4仅binary，拒绝choice及路径/采样策略覆盖。
首次自动回放全部合格路线并生成data2/data4；源码绑定请求/完成回执、目录锁、完整配对与不截断校验。模型缺失在昂贵建库前失败；check/preflight/host-preflight不读取best或测试旧adapter。
全8614路线1062401帧实回放；split7268/658/688、2/4各88986题、443人工及val273/test186/review539保持，v37题ID全保留。每轮793040次、十事件各79304次；两模式真实七轮及world4七轮100%覆盖。预算10000拒绝。
972专项通过；已实际验证首次建库/两图SKIP_BUILD复用。模型缺失，未GPU训练/真实测试/打包/CARLA；这些阶段只验证编排边界。无新增人工标签/RGB目视/独立评估，弱监督范围与关键覆盖缺口保持。
合同v38、老师v8规则未改、预约v18同隔离、严格registry v13为空；新产物checkpoints/phase4_v38_full，旧v37保持。详见Phase4/PIPELINE_V38.md及pipeline_v38_verification_20261007.json。


### 2026-10-06 Phase4 v37 全量训练与严格事件1:1

用户明确选择全量合格监督每轮全覆盖、呈现次数事件1:1，接受大量重复。新增full_event_equal/专用train_full.sh 2/4；不应用旧cap4/cap8，最小预算由最大事件题池和DDP整除决定，过小预算拒绝。
建库max-teacher-questions-per-route=0，manifest记录无路线截断；全量入口拒绝截断库或配对丢题。checkpoint绑定全量采样历史；主机预检报告重复量/实际world/优化步数。
全8614路线1062401帧重放，split7268/658/688保持；2/4各88986题且全部配对，v36所有题ID保留，新增13497题；443人工及val273/test186/review539保持。
941专项通过，真实pipeline/RGB/world1-4七轮全覆盖及题序配对通过；world4每轮793040次、各事件79304次，每轮100%覆盖；7轮5551280呈现。预算10000拒绝，缺4B未GPU/CARLA。
老师v8、合同v37、预约v17、严格批准0；关键监督/六事件人工评估缺口保留，无新增人工标签或目视审计，不把全量弱监督称为驾驶认证。
产物AutoMoT/checkpoints/phase4_v37_full，旧v36保持。见Phase4/FULL_TRAINING_V37.md及full_training_v37_verification_20261006.json。

### 2026-10-06 Phase4 v36 稳定选题、配对训练与异常起点

按不含图数/源码的语义身份稳定选题；显式paired-with核对双库合同/状态/答案/共享RGB，训练交集重算准入，恢复绑定双方manifest；val/test完整保留。
全8614路线/1062401帧重放，split7268/658/688保持；2/4训练75489/75489，配对75489，原443人工及val273/test186/review539保持。
920专项通过，实际pipeline/全部RGB/配对启动及world1-4七轮通过；两模式每轮题序一致，400次/轮、七轮705题；预算10000拒绝，缺4B未GPU/CARLA。
9精确风险、6新曝光训练组接入；新确认相邻异常按首次受影响帧生效，旧审核约束保持。97帧YES因果证据止于97，98消失不反向否决；新窗口旧库题重叠0。
实际仅1训练路线2原图，无新完整序列/人工标签/独立RGB；继承229/278格三ID筛查、140条待审、18源不足。
老师v8、合同v36、预约v16、严格批准0、formal_data_ready=false；关键转移/六事件评估/八事件老师缺口保持，1037盲审卡未填。
新产物AutoMoT/checkpoints/phase4_v36_full，旧v35保持。见Phase4/PAIRED_TRAINING_V36.md及thirty_sixth_audit_verification_20261006.json。

### 2026-10-06 Phase4 v35 局部控制、弯道骑行与采样准入

局部释放排除有范围证据的远处控制，当前hazard/近处红灯/STOP义务仍约束；骑行按当前路线切线判断，同向跟随不授权绕行。
采样预算/cap/重复/seed/epochs/world贯通预检，require-trainable拒绝不可行计划；直接训练在模型加载前检查容量。
全8614路线/1062401帧重放，train7268/val658/test688保持；2/4训练75488/75489，443人工及val273/test186/review539保持。
900专项通过，2/4真实pipeline/全部RGB/world1-4七轮通过；预算10000预检均退出2。默认每轮400，七轮覆盖705/75489；缺模型未GPU/CARLA训练。
17精确风险接入，旧库每模式11弱题重叠排除，人工/留出无重叠；本轮实际3训练路线6原图，无新增完整序列/人工标签/独立RGB。
继承216/278三ID筛查、166待审及18源不足。老师v8、合同v35、预约v15、严格批准0、formal_data_ready=false；八事件老师/六事件评估/关键转移仍缺。
2/4模式独有8141/8142题，当前训练不是严格配对图数消融；question_id绑定图数/源码造成逐路线限额重选，已记录未改。
新产物AutoMoT/checkpoints/phase4_v35_full，旧v34不变。见Phase4/LOCAL_SCOPE_AND_PREFLIGHT_V35.md及thirty_fifth_audit_verification_20261006.json。

### 2026-10-06 Phase4 v34 参数缩写、主机预检退出与依赖重建

train/preflight/launch禁用参数缩写，shell提前拒绝路径参数全部前缀；pipeline的host-preflight成功后退出，不读best或评估旧adapter。
合同错误列出具体源码/依赖及新旧SHA；保留当前Phase3开发组隔离修改，按最终依赖重建v34，不修改v33 manifest。
新增13实际已看训练组、11精确不连续窗口，未看val组不曝光。新窗口覆盖完整人工因果包络含图间隔；旧整路线审核约束保持，卡/编译/加载范围一致。
新增风险逐题检查：每模式相关18训练/3待审题与11窗口重叠0，没有据源异常撤回标签。本轮2训练路线4原图复核，无新完整序列/标签/独立RGB。
全8614路线/1062401帧重放，train7268/val658/test688保持；2/4训练75373/75373，原443人工及val273/test186/review539保持。
878专项通过，2/4真实pipeline、全部RGB、world1/4七轮通过；最终默认2/4启动再验通过，验收时间/SHA保存。默认MODEL_DIR仍缺4B，host-preflight明确失败，未GPU/CARLA训练。
默认每轮400，七轮覆盖705（0.9353%）；关键边/六事件评估/八事件老师缺口保持；严格批准0，盲审961卡未填，formal_data_ready=false。
继承筛查203/278格三ID、余215条及18格源不足，非逐题认证。合同v34/data_v34、老师规则v7保持、预约v14，新产物AutoMoT/checkpoints/phase4_v34_full。
见Phase4/LAUNCH_AND_RISK_V34.md及thirty_fourth_audit_verification_20261006.json；final_source_acceptance只证明记录时点，后续依赖修改须重新验收。

### 2026-10-06 Phase4 v33 分支覆盖、配对评估与启动检查

逐分支/返回策略/边/阶段/答案/split计数，零样本分支显式列出；补问题不填readiness缺额，新增补标任务清单与七轮覆盖比例，无新增准入硬门。
evaluate输出绑定完整因果输入和参考的paired_identity；paired_eval拒绝漏题/重复/换图/换答案，人工准确率与老师一致率分报。
启动默认使用v33全量弱库；缺模型/数据在GPU前明确报错，路径覆盖不能绕过预检；构建区分审核题模式与全量老师模式。
全8614路线/1062401帧重放，物理train7268/val658/test688保持；2/4训练75373/75373题，原443人工及val273/test186/review539保持。
850专项通过，2/4真实pipeline check、全部RGB核验及world1/4七轮通过；默认每轮400，七轮覆盖705，比例0.9353%。
12既有训练曝光组及7确认消失窗口接入，撤回/未确认猜测不升级。本轮3训练路线8帧/3缩放联系表，无新完整序列/人工标签/独立RGB。
继承筛查199/278格，剩228选定序列；筛查不作标签认证。关键绕行/进入/返回和六事件独立评估仍缺，八事件老师仍缺。
新盲审961卡/56路线未填，严格批准0；弱trainable=true、formal_data_ready=false。默认模型仍缺，Action/CARLA在线提供者未接通，未GPU训练/CARLA验收。
合同v33/data_v33、老师规则v7未改、预约v13；新产物AutoMoT/checkpoints/phase4_v33_full，v32保留。见Phase4/AUDIT_COVERAGE_V33.md及thirty_third_audit_verification_20261006.json。

### 2026-10-05 Phase4 v32 过期等待截止与静止正例诊断

UE1在已观察停车后持续实际前向运动而仍YIELD时截止旧实例，不造YES/complete，不放宽运行时执行许可；前车连续可见超30m范围同样截止。
全8614路线/1062401帧重放，train7268/val658/test688保持；2/4弱库训练75373/75373题，原443人工及val273/test186/review539保持。
已定位VehicleTurningRoute错误静止YES实际19道，均清除；其他旧静止YES18道逐项保留；821 f167–175旧proceed清零。
预检按事件/边列静止readiness YES缺额，仅报警；cyclist_follow提示明确允许继续跟随，不要求骑车人消失，不授权超车。
新增至少500帧全程静止的元数据证据过滤；具名BlockedIntersection四路实际起末均行驶，保留并记录反证，本轮无新剔除。
833专项通过，2/4真实pipeline check、全部RGB核验、world1/4七轮通过；默认每轮400，七轮覆盖705，八事件老师仍缺。
弱试训trainable=true，formal_data_ready=false，严格批准0。新盲审961卡/56路线全部待审，无独立准确率。
本轮仅两训练路线10RGB关键帧联系表复核，无新增完整序列/人工标签/独立RGB/GPU训练；八个具名曝光组及两条用户报告行人消失窗口登记。
合同v32/data_v32、老师v7、预约v12；新产物AutoMoT/checkpoints/phase4_v32_full，v31保留。见Phase4/TEACHER_LIFECYCLE_V32.md及thirty_second_audit_verification_20261005.json。

### 2026-10-05 Phase4 v31 实例范围、运动偏差与弱监督重建

UE1旧前车连续可见离开走廊/被正常移动前车替换后截止；普通跟车不作为额外堵塞，1765_0 f158–180旧实例错NO清零。
新增保留图像变化门槛的静止释放路径；运动后UE1 proceed原始YES保留回放但不进弱readiness，自动视觉catchup未批准；人工UNKNOWN否决弱覆盖。
同向骑行者用cyclist_follow分支，横穿转跟随须新实例；不凭场景名猜绕行许可。可见删除仍异常，已完成实例不被下一帧删除改写。
弱restrict排除；单答案边权重降1；事件等额/边内答案等额/重复cap4/帧cap8保持，readiness:catchup权重4:1、内部运动分层并报告容量回退。
全8614路线/1062401帧重放，物理train7268/val658/test688保持。2/4弱库训练76871/76871题；原443人工题及val273/test186/review539保持。
UE1静止proceed弱YES=37，运动弱readinessYES=0；默认每轮400次，七轮覆盖708/708，八事件老师仍缺，不声称全池已训练。
813项不同专项测试通过；2/4真实pipeline check、world1/4七轮和全部RGB身份核验通过。严格批准0，formal_data_ready=false，弱实验trainable=true。
93活动/16退役独立预约保持，新盲审945卡/56路线全部未填；未看独立RGB。实际5联系表/24训练帧，无新增完整序列/人工监督/GPU训练。
用户6精确训练物理组登记，4_47 f51→52可见骑行者消失登记窗口；未具名的额外6条UE1路线未猜ID。
合同v31/data_v31、老师v6、预约v11，新产物AutoMoT/checkpoints/phase4_v31_full，v30保留；见Phase4/TEACHER_SCOPE_V31.md及thirty_first_audit_verification_20261005.json。

### 2026-10-05 Phase4 v30 全量弱监督、时长过滤与加权采样

过滤963异常时长及135额外无metas路线，合格8614路线/1062401帧，train7268/val658/test688；原物理split保持，排除明细留账。
独立预约109→93，16条退役仍保留val/test隔离；新2图盲审933卡/57路线，全部未填，12类最佳情形仍0批准。
风险按登记帧包络核对完整因果历史及STOP证据，窗口外仍过统一自动检查；人工逐帧风险审核保持。
最终全量老师回放、账本、2/4弱库及留出一致率参考均完成。train2=81377/train4=81377，弱题80934/80934，训练物理组1914/1914；原443题逐行保持，val273/test186/review539逐字节保持。
新event_weighted：事件等额→主边权重8/恢复跟车4/其他1→边内YES/NO等额，先不同题后重复，单题cap4、共享帧cap8；预算按实际容量计算。
每轮400次受RE5仅5道YES限制，七轮world1覆盖711/711，未覆盖80666/80666；不声称全池学完。
795专项通过；2/4真实pipeline check、world1/4七轮及各82375/82375题RGB身份核验。初版重复分配缺陷已修正并重新生产，旧产物及绑定源码保留作对照。
留出老师参考每模式val8736/test8369，仅一致率，人工指标选优保持；其他8类仍无老师。弱trainable=true，严格formal_data_ready=false。
合同v30/data_v30、老师v5、抽检v10；阈值/冻结语义/人工标签未改。实际新目视0、独立人工参考0、GPU训练0、CARLA验收0。
用户3个精确曝光组已train-only，第4个OppositeVehicleRunningRedLight/Town01缺ID待补；旧逐帧审计覆盖缺口保持。
产物AutoMoT/checkpoints/phase4_v30_full；详见Phase4/FULL_WEAK_TEACHER_V30.md及thirtieth_audit_verification_20261005.json。

### 2026-10-05 Phase4 v29 弱监督实验入口、限制证据与停车义务

复核v28独立池12个2图规则类，假设全对仍0类可达严格批准；新增best_case_feasibility明确判断/路线/转移缺额。
保留严格批准v2，新增显式命名/逐类白名单/源码绑定的weak_experiment_train_only；弱题仅train、label_basis=weak_rule_teacher，不能冒充已批准或人工准确率。
实例创建须先观察本事件自身限制，3_26首帧即YES实例已排除；新增StopDutyTracker与当前速度横穿检查，记录带SHA的连续停稳历史，断档/换ID/车道清除。
STOP框是停止控制区域代理而非已标注停止线；2858_0 f50/2679_1 f48仍距区域13.55/16.30m，提前等行人不能算履行停车。真实停车释放正例尚未确认。
23开发路线2598帧每模式57YES/170NO/68UNKNOWN，proceed37、恢复跟车YES1、完成YES1，完成实例1。
过滤风险后实际新增73弱监督（18YES/55NO），train443→516/45物理组；原443题保持，val273/test186/review539与v28逐字节相同。
新增event_edge_answer：事件等额、事件内边×答案等额、共享帧cap8、缺额不借NO、恢复绑定历史。弱库默认该政策；原人工库仍event_equal。
782专项通过；2/4实际完整构建/check及975监督+539待审RGB核验；预算100、world1/4七轮采样357/516，159未覆盖。默认整池预算520不满足稀缺格容量，须指定已验证预算。
同109独立路线21608帧重放生成963张2图盲审卡，均未填写/未目视独立RGB；本轮无新人工标签/RGB目视/完整序列/曝光组。
合同v29/data_v29、老师v4、抽检v9同预约同split；严格注册表v4为空，弱实验可训但formal_data_ready=false。9712仅全目录清点，非全量标签生产。
八类老师/独立误差/十事件评估仍缺；未GPU/4B/DDP/CARLA验收。控制器/冻结规则/安全/选优/人工标签/Phase3/Action未改，采样及训练入口按本版修改。
详见Phase4/WEAK_TEACHER_V29.md及twenty_ninth_audit_verification_20261005.json；产物/tmp/phase4_v29_final，训练须--epoch-samples 100。

### 2026-10-05 Phase4 v28 局部异常、释放回放与独立池扩充

异常按参与者/局部走廊删失，干净历史后可重建同ID；自车已动不再抹掉readiness YES，修复旧推进回执重复提交。
5观察跟车恢复含间距/相对速度/闭合约束；0.5s仅待校准状态阈值，不是认证安全距离。本车道控制替代路口距离一刀切，红灯仍hold；候选30m及减速前车召回。
原17开发路线1615帧每模式18YES/55NO/125UNKNOWN、proceed12、完成1；扩展23路线2598帧60YES/264NO/697UNKNOWN、proceed41、完成1。
1030_0 f72–73推进、f75恢复跟车、f76完成；3_26仍无可靠UE4、暗图/决定性异常仍弃答，不宣称所有实例可完成。
按有实例而非答案筛384留出路线，新增69物理组，UE1/UE4各40条；原40预约及split保留，共109路线。19260次帧源异常未当标签。
实回放独立109路线21608帧，仅程序处理；2图开发240卡/独立1485卡全未填写，独立RGB/逐题答案未用于调规则，4图不继承批准。
766专项通过，2/4图真实pipeline check及902监督+539待审RGB核验；四JSONL与v27相同，train443/45路线、val273/test186。
实复看3既有训练路线8原图，无新完整序列/曝光组/人工标签；批准0、自动监督0，formal_data_ready=false、有限trainable=true。
合同v28/data_v28、老师v3、抽检v8、注册表v3空；新增批准预算和采样重复诊断，旧标注/冻结状态规则/控制器/安全/采样/训练/选优保持。
未全9712生产或GPU/4B/DDP/CARLA验收，八类老师/视觉catchup/独立准确率仍缺，Phase3/Action未改。详见Phase4/TEACHER_RELEASE_V28.md及twenty_eighth_audit_verification_20261005.json。

### 2026-10-04 Phase4 v27 运动证据修复与可执行盲审包

UE1间距增长需前车前向运动；执行回执需车体前向位移/侧移及朝向约束，倒退/侧移不能确认纵向起步。
严格bool/None条件与来源SHA；完成实例经连续可见解冲突观察可重开同ID，缺图/消失/待复核不重开，终止原因登记。
新增teacher_review prepare/import/assemble：分层限额选题、公开包隐藏老师答案、全样本逐项审核导入、逐类组合批准。
去重不含源路径/帧号；YES/NO各需20物理路线，未知参考计错误分母，漏题/空证据/身份状态未确认的二元参考拒绝。
实跑17开发路线1615帧生成212卡；40冻结独立路线9759帧生成152卡，仅程序生成，未看独立RGB/参考/调规则，全部待填写。
736专项通过，2/4图真实pipeline check及902监督+539待审RGB核验；四JSONL与v26逐字节相同，train443/45路线、val273/test186。
真实批准0、新标签0、新RGB目视0，formal_data_ready=false、有限trainable=true；8事件老师/视觉catchup/真实恢复正例仍缺。
合同v27/data_v27、老师v2、批准政策v2、抽检v7同40路线同split；标注v9、快照v11、冻结规则/控制器/安全/采样/训练/选优保持。
未全9712路线生产或GPU/4B/DDP/CARLA验收，Phase3/Action未改。详见Phase4/TEACHER_REVIEW_V27.md及twenty_seventh_audit_verification_20261004.json。

### 2026-10-02 Phase4 v26 因果规则老师与逐类批准链路

新增teacher_rules/replay/approval/data：先支持UE1/UE4，参与者身份提示+可见局部冲突建实例，真实Episode逐帧提问/推进/运动回执。
规则原始YES/NO/UNKNOWN与批准监督分离；每题绑定老师源码及因果输入，独立开发/留出参考、Wilson下界和不同路线数逐类批准。
实际17开发路线1615帧，2/4图各238提议：15YES/97NO/126UNKNOWN；其中proceed正例11（UE1一帧、UE4十帧），均未批准监督。
精确开发参考每模式17题：10一致/5弃答/2人工参考UNKNOWN；对应1030_0 f72和Town07 3_1 f52，不声称100%准确或独立认证。
修复+inf路口距离哨兵、三值未知传播、已在走廊外停下的行人；最近前车筛选，冲突可见输入排除/重复审核不重复计数。
完整生产按全路线逐帧处置索引计算，不再硬编码False；全UNKNOWN不能过正式准入。按路线并行/完整产物续跑可用，未跑9712条全量。
真实规则批准清单为空，自动catchup无监督，其他八事件老师未实现；40抽检v6保留同路线同split未审，20路线/规则要求仍需前瞻扩样。
703专项通过；2/4图真实pipeline check及902监督+539待审RGB核验；四JSONL与v25逐字节一致，训练443/45路线、val273/test186。
实际仅复看2既有训练路线6原图，无新完整序列/曝光组/人工标签；recover_follow真实正例仍缺，合成状态路径通过不算真实支持。
合同v26/data_v26、默认标注v9、快照v11；评估老师一致率与人工准确率分列，正式formal_data_ready=false，有限训练trainable=true。
冻结四规则/标注/holdout/状态机/安全许可/采样/训练/选优不变，Phase3/Action未改，无4B/GPU/DDP/CARLA验收。
详见Phase4/RULE_TEACHER_V26.md及twenty_sixth_audit_verification_20261002.json。

### 2026-10-02 Phase4 v25 实际输入证据、弱标签撤回与分层审核

UE1 f70–72及UE4 f50–52六题退为UNKNOWN；保留v24增量8题，训练443/45路线，待审539，旧435训练题及val/test保持。
新增逐模式visual_review：精确2/4RGB与SHA、可定位当前证据、补问后继视觉证明；4图支持/2图未知可分流，构建加载均检查。
1m完整轮廓余量/6px可见变化等为保守开发口径，未认证驾驶阈值；规则作者复核不冒充独立盲审。
抽卡按事件×场景×Town×边×建议答案平衡并优先不同物理路线，隐藏逐卡建议；独立val/test包不经开发编译，排除40生产者抽检。
17路线1615帧3700常规题+4身份提示，分层79卡/117层，45层未选；UE7源场景仅检索身份卡，不能确认为故障或监督。
实际看5路线87RGB/20面板含7原图，002053完整45帧为既有路线复核；灯色非始终绿色，适用灯头匹配/故障身份待审，无新曝光组。
653专项通过，2/4图真实check及902监督+539待审RGB核验；world1/4七轮第三轮覆盖443，恢复历史一致。
合同v25/data_v25、默认标注v9、快照v11，须新产物新run；40抽检v5同路线同split未审，独立任务清单仅待分配。
预检新增每边每答案20不同路线/每评估事件5路线缺额；关键边、六事件独立评估、全量可靠监督仍缺，formal_data_ready=false。
旧冻结规则/标注/holdout/状态机/安全许可/采样/训练/选优及Phase3/Action保持；未4B/GPU/DDP/CARLA验收，继承192/278格余约240序列。
详见Phase4/TWENTY_FIFTH_AUDIT_FIXES_20261002.md与twenty_fifth_audit_verification_20261002.json。

### 2026-10-02 Phase4 v24 半自动转移审核与首批监督增量

选择方案B：局部释放建议与稳定跟车分开，当前参与者附近走廊排除下一路口远车污染；缺证据仍UNKNOWN，自动建议不作监督。
路口候选要求当前冲突/适用停车标志/待核异常灯态，已过车仅短暂延续；17路线1615帧候选6066→3700，局部释放76真仍待审。
新增有预算的转折审核队列、显式目标请求、带相机分界的因果RGB面板及逐项审核编译；缺审核/SHA/风险/可见范围证据拒绝。
真实复看6路线局部84张不同RGB/20面板，其中4原图，无新完整序列/曝光组；疑似双行人与相机重叠吻合，1150_0后段存在新车辆。
原v7的152条记录保持；v8新增14题：6readiness YES、4readiness NO、4catchup YES。训练449/45路线，RE3 enter新增2YES、UE2 depart新增2NO。
630专项通过；2/4图真实check及908监督+533待审RGB核验，旧435训练题不变，val/test/review逐字节保持；七轮world1/4第三轮覆盖449。
合同v24/data_v24、快照v11，须新候选/数据/run；40条抽检v4保留同路线同split，尚未审核，无独立教师准确率。
return/recover_follow及UE2 depart YES、UE7 proceed YES等仍缺，评估仅四事件，formal_data_ready=false；未完整4B/GPU/DDP/CARLA验收。
原标注/holdout/状态机/安全许可/采样/训练/选优SHA保持，Phase3/Action未改；继承192/278格与余约240序列。
详见Phase4/TWENTY_FOURTH_AUDIT_FIXES_20261002.md及twenty_fourth_audit_verification_20261002.json。

### 2026-10-02 Phase4 v23 静态冲突、跟车语义与自动上下文

静态障碍不再因语义像素0被跳过；未解决占据使间隙UNKNOWN；正常跟车可作为UE1/UE3推进，行人等待不再限2m。
缺失/跳变区分RGB可能可见与相机外清单风险，后者仍拒绝安全许可；跳变两端任一可能入镜均保留RGB风险。
当前未改规划路线逐段几何接入，UE5的1150_0第127–151帧召回，676_0第57–87仍无误候选。
新增路口UE6/UE7/RE5假设、交通控制/连续身份过车部分上下文；不把红灯当故障、绿灯当完整优先权或过车当事件完成。
负static_prop_car extent保留诊断且拒绝许可，两个Town05样本114帧受影响；交通控制代理框与物理碰撞检查分开，仍须优先权凭据。
17路线1615帧6066候选全待审；1465帧有当前未改导航；133异常中64可能RGB/69清单风险，均非全部目视确认。
实际复看5路线局部82张RGB、其中4原图，无新完整序列；5曝光组train-only、4风险记录接入，继承192/278格与余约240序列。
610专项通过；2/4RGB真实check及894监督+533待审核验、四JSONL与v22相同，七轮world1/4第3轮覆盖435题。
合同v23/data_v23，快照v11；需新产物新run。40留出抽检v3保留同路线/原split，尚无审核标签或独立准确率。
自动可靠实例/优先权/可见性/完整机动导航与安全三前提仍缺，无新增监督；训练435/42路线、关键四边0、评估仅四事件。
formal_data_ready=false，无完整4B/GPU/DDP/CARLA验收；Phase3/Action与冻结标注/采样/选优不变。详见Phase4/TWENTY_THIRD_AUDIT_FIXES_20261002.md。

### 2026-10-02 Phase4 v22 瞬移检测、可见性与上下文接线

同ID跳变用ego_matrix补偿自车运动，146_1 18→19的3824/3823实测位移约59.4/59.9m；缺位姿安全许可拒绝。
保留base_type/type_id，自行车/摩托车归UE4、静态障碍归UE2候选；弯道正常对向车不凭直线带判侵入，676_0 57–87误候选为0。
投影RGB亮度/对比度/过曝门槛只作开发筛查，未认证可见性；来源绑定的当帧状态/导航/场景contexts接入生产，新范围横向参考可校准。
缺上下文时假设状态target仍UNKNOWN，几何结果单列；默认12路线1087帧2462候选全待审，68缺失ID+4跳变未逐一确认。
旧校准仍320参考311弃答/8对/1错；全量可靠上下文及安全三前提的自动来源未实现，无新增监督。
九组显式train-only、五风险记录；仅复看3原图，001997人形无对应walker记录仍身份待核，无新完整序列目视。
40条留出路线冻结生产者人工抽检v2计划，保持原split并禁止开发/选优用途；只是预约，尚无人审标签或十事件评估。
592专项通过；2/4RGB真实check及894监督+533待审核验，四JSONL与v21相同；七轮world1/4第3轮覆盖435题。
合同v22/data_v22/快照v11，须新候选/数据/run；训练435/42路线，关键四边0、val/test仅四事件，formal_data_ready=false。
冻结四规则/旧标注/旧holdout/采样/训练/选优SHA保持，Phase3/Action默认未改；无完整4B/GPU/DDP/CARLA验收。
见Phase4/TWENTY_SECOND_AUDIT_FIXES_20261002.md与twenty_second_audit_verification_20261002.json；继承192/278格及余约240序列。


### 2026-10-01 Phase4 v21 因果候选生产与离线安全适配

新增白名单bboxes/metas因果几何、逐帧候选/异常生产与开发校准，排除未来数组/专家动作；候选不作监督。
实扫六路线712帧，26个ID缺失候选；320有效readiness参考311弃答/8对/1错，排除98旧横向范围参考，未批准自动标签。
离线回放接上当前导航包络＋全参与者运动几何安全适配，仍须真实覆盖/可行驶空间/通行权证据；不是CARLA适配验收。
明确安全拒绝正常等待、缺信息仍未知；新增矛盾/缺许可/拒绝统计，修复旧范围编译题隔离后加载来源核对。
六组显式train-only、七风险记录接入；仅复看2954_1五张原图，无新完整RGB序列/训练标签，146_1沿用v20不重复。
567专项通过；2/4RGB真实check及各894监督+533待审RGB通过，四JSONL与v20逐字节一致，七轮world1/4覆盖435题。
合同v21/data_v21/快照v11，须新候选/数据/run；冻结四规则/标签/holdout/采样/训练/选优SHA保持，Phase3/Action默认未改。
train435/42路线、val273/test186仅四事件，关键depart/enter/return/recover_follow仍0，formal_data_ready=false；无完整4B/GPU/DDP/CARLA验收。
继承192/278格、余约240序列，不将程序读取计作目视；见Phase4/TWENTY_FIRST_AUDIT_FIXES_20261001.md及twenty_first_audit_verification_20261001.json。


### 2026-10-01 Phase4 v20 跟车恢复与可见范围安全准入

RECOVER新增recover_follow到FOLLOW，不再伪造renewed_restriction完成正常跟车；新边真实监督仍0。
depart/enter/return仅判断可见范围，接受机动另需当前规划器/BEV全走廊及后侧许可，绑定实例/帧/边/目标/分段；缺失不授权，执行回执不能替代，整帧事务保持。
旧横向题须逐帧新范围复审，128监督转待审；两RGB现train435/val273/test186、待审533，训练42路线，val/test仍仅四事件。
15物理组显式train-only、10风险路线登记；仅复看146_1两张原图确认18→19消失，其余疑似不升级确认，无新标签/完整序列审计。
529专项通过；原标注/四冻结规则/采样/训练/选优不变，Phase3/Action默认未改。合同v20/data_v20/快照v10，须新候选/数据/run。
缺415转移格、12分段格、9环岛格；formal_data_ready=false，未真实BEV适配器/4B/GPU/DDP/CARLA验收，未放宽HOLD为蠕行。
详见Phase4/TWENTIETH_AUDIT_FIXES_20261001.md与twentieth_audit_verification_20261001.json。


### 2026-09-30 Phase4 将审计风险接入构建准入

新增risk_review.py，十/十一轮12条风险路线必须逐帧复审并覆盖全部因果输入RGB，旧整段/缺审核/过期风险ID拒绝。
usable要求身份/转移范围/实例边界复核及本帧SHA；任一历史图uncertain/excluded则整题UNKNOWN待审，不作YES/NO监督。
加载再次校验，manifest/preflight记录处置；新增待填写审核模板CLI，内部证据不进入模型提示词。
任务v11/data_v11，需新产物新run；prompt v6/快照v8/标注v4/标定v5不变，Phase3/Action稳定默认未改。
311项专项通过（新增20）；两模式实际check、各453题RGB、14整池计划及cap1小预算各453轮覆盖通过。
四个JSONL与第十一轮逐字节一致，453题/133待审；无新人工标签/目视审阅，风险机制不自动识别异常或替代人工证据。
ready=false，仍缺389转移格/12分段格及独立val/test/完整4B；覆盖155/278格，剩361条，未正式GPU/DDP/CARLA验收。
详见Phase4/RISK_ADMISSION_20260930.md。

### 2026-09-30 Phase4 第十一轮曝光与证据登记

ParkedObstacleTwoWays/Town12_4608_0显式登记train-only，不能依赖默认seed恰好分train；七条重点路线/16张定位RGB摘要保存复审线索。
仅改曝光合并/合同资产绑定及说明，不改控制器/采样/提示词/标定/人工标签；569消失点保留原图纠正的f74→75。
任务名v10、提示词v6、快照v8、标注v4保持，源码/曝光摘要更新须新产物新run；Phase3/Action默认未改。
291项专项通过；2/4RGB实际pipeline check、各453题RGB与14整池计划、cap1小预算各453轮覆盖通过。
四个JSONL与第十轮登记产物逐字节一致；453题/133待审，val/test空，缺389转移格及12分段格，ready=false。
本轮复看2面板34张既有RGB，无新增完整序列/标签；报告累计339条/38369帧、155/278格三ID、剩361条，18格源不足。
完整审计及正式4B/GPU/DDP/CARLA仍未验收，本机默认完整模型缺失。见Phase4/ELEVENTH_AUDIT_REGISTRATION_20260930.md。

### 2026-09-30 Phase4 第十轮曝光与风险登记

四个新增曝光物理组正式登记train-only；五条消失/碰撞/出画风险保存关键帧SHA及报告来源，均未入默认题库。
仅修改曝光合并/合同资产绑定与说明，不改状态机/采样/提示词/标定/人工标签；风险表不自动生成标签或排除样本。
任务名v10/提示词v6/快照v8/标注v4保持，但源码及曝光摘要改变，须新产物新run，旧manifest不可改hash复用。
289项专项通过；2/4RGB实际pipeline check、各453题全量RGB、14整池采样及cap1小预算各453轮全覆盖通过。
四个JSONL与此前v10逐字节一致，453题/133待审、val/test空，缺389转移单元及12分段单元，ready=false。
本轮复看三面板72张既有RGB，无新增完整序列/标注；报告累计311条/35059帧，137/278格三ID，剩389条、18格源不足。
完整审计和正式4B/GPU/DDP/CARLA未验收，本机默认4B目录仍缺；Phase3/Action稳定默认未改。
详见Phase4/TENTH_AUDIT_REGISTRATION_20260930.md。

### 2026-09-30 Phase4 第九轮采样与RE3分段修复

小预算改为固定题目交错队列只轮转一次；四题budget1十二轮各三次，固定池可行正预算N轮覆盖每题。
RE3提供相邻route_segments，复用WAIT/CROSS逐段接续，每段新观察重新问enter；保留纵向约束，末段稳定才退出。
提示词仅当前段状态对，执行回执绑定segment_id；错段/缺段拒绝且tick原子回滚；回放真值跨段删失。
任务v10/data_v10、提示词v6、快照v8、采样v2，须新产物新run；人工标注v4/标定v5保持，Phase3/Action默认未动。
284项专项通过；2/4RGB实际check、各453题RGB核验、各14整池采样计划及453轮budget1覆盖通过。
原453题/133待审除模型输入身份摘要外逐行一致；原389转移格缺额保持，新增12项分段支持检查全缺，ready=false。
登记第九轮两个PedestrianCrossing/Town13曝光组train-only；本轮复看四面板96张既有RGB，无新增完整序列/标注。
报告累计283条/32537帧、120/278格三ID，剩417条，18格源不足；完整逐帧审计未完成。
默认4B目录不存在，无正式GPU/DDP/CARLA验收；邻接关系须上游导航确认，不从RGB自动生成完整计划。
详见Phase4/NINTH_AUDIT_FIXES_20260930.md。

### 2026-09-30 Phase4 第八轮审计修复

UNKNOWN/缺答不再吞掉有效上下文中同帧明确的hold；仅接纳独立收紧约束，保留UNKNOWN/STOP，
不授予推进许可；显式矛盾仍复核，已有HOLD不因再次让行或不确定超时丢失。
构建/加载按有序解码RGB和实际system/状态对消息核对相反答案，隐藏instance ID不能区分；
load_images核对像素身份。新增输入身份字段，任务合同v9/data_v9/快照v7，须新数据新run。
256项专项通过；2/4RGB实际pipeline check、各453题全量RGB核验和各14采样计划通过。
默认453题/133待审，除两项身份字段外逐行保留，无实际输入冲突；标注v4/提示词v5/标定规则v5未改。
第八轮新增曝光组Town04_route_001046已登记train-only；本轮只复看3面板71张既有RGB，新增完整序列0。
审计报告累计259条/29698帧，107/278格三ID，剩441条，18格来源不足；全覆盖未完成。
ready=false，仍缺389转移支持格/独立holdout/完整4B，无正式GPU/DDP/CARLA验收；Phase3/Action默认未改。
详见Phase4/EIGHTH_AUDIT_FIXES_20260930.md。

### 2026-09-30 Phase4 第六/七轮候选标注接入

四段proceed候选逐帧复核后纳入reviewed_state_pairs_v4.json，默认完整流水线更新；原v3的57条逐条保留。
新增24题、12UNKNOWN、4截止排除，开发库453题/133待审；UE4条件已成立但尚未起步仍YES，突变消失不生成YES。
本次实际复看7面板、43原尺寸，去重169张既有RGB；无新增完整序列/曝光组，增量证据SHA绑定。
230项专项通过，默认入口新断言单独通过；2/4图默认pipeline check、全453题哈希和各14个采样计划通过。
原429题/121待审逐行保留，新库逐行等于第七轮候选；曝光集合不变，全部train-only。
任务合同v8/data_v8/标注v4，提示词v5/标定规则v5/快照v6保持；新数据新run，Phase3/Action默认未动。
ready=false；缺389转移格（119/135/135，其中270公共纵向）、独立holdout及完整4B，无正式GPU/DDP/CARLA验收。
第七轮报告累计238条/27633帧、97/278格三ID，剩462条，18格源不足；完整审计未完成。
详见Phase4/RGB_CANDIDATE_INTEGRATION_20260930.md。

### 2026-09-30 Phase4 第四轮部分审计修复

旧per_frame_conditions仅保留readiness；legacy catchup构建和直接helper均拒绝，须逐帧复审为reviewed_transition_band。
回放真值缺失/UNKNOWN在truth_unknown处删失，后续YES另起；整帧缺失/不再询问的边也截止，记录last_ready_frame。
f10 YES/f11未知/f12 YES且执行，三种延迟均[0]并保留旧区间删失；连续YES对照仍[2]。
222项专项通过；2/4RGB各429题及121待审与v6逐行一致，全量哈希、各7轮×world1/4采样通过。
任务合同v7/标定代码v5/data_v7，提示词v5/人工标注v3/快照v6不变，须新数据新run。
本轮无新增RGB目视/曝光/标注；完整审计仍224条/26418帧、92/278单元三ID，剩476条、18单元来源不足。
ready=false；缺完整逐帧覆盖、监督/独立holdout及完整4B，无正式GPU/DDP/CARLA验收。
Phase3/Action稳定默认未改；见Phase4/FOURTH_AUDIT_FIXES_20260930.md。

### 2026-09-30 Phase4 第三轮部分审计修复

re_yield保留尚未解除的HOLD，七类事件连续hold→re_yield仍STOP。
tick先整批检查再在副本推进/核验回执，异常时快照不变；advance/revalidate/acknowledge同样避免部分写入。
catchup新增transition_blocking_conflict作用范围：独立纵向限制不否定几何完成；有冲突但无范围拒绝构建。
新增instance_boundary/calibration_anomaly人工审核截止，不自动扩窗或从目标消失生成YES。
本次复看七面板155张既有RGB、不增加完整序列计数；报告11个实际曝光组补入Phase4 train-only。
用户报告累计224条/26418帧，92/278单元三ID，剩476条，18单元来源不足；完整审计未完成。
202项专项通过；2/4RGB各429题逐行等于v5、121待审，哈希及各7轮×world1/4采样通过。
任务/快照v6、标定代码v4；提示词v5/人工标注v3未改，须新数据新run。Phase3/Action稳定默认未动。
ready=false；缺完整分支/纵向及独立holdout监督、完整4B权重，无正式GPU/CARLA或实时吞吐验收。
详见Phase4/THIRD_AUDIT_FIXES_20260930.md；事务不回滚外部预测器副作用，副本随历史增长的开销待实测。

### 2026-09-29 Phase4 第二轮部分审计修复

用户报告两轮累计121条完整序列/10900帧，278单元仅37个三ID，剩579条，18单元来源不足。
本次只复看StaticCutIn/Town13三条已审路线72–95帧共72张，不增加完整序列计数、不批准新YES区间。
修复横向目标冲突丢STOP/无可用复核入口；汇总/export返回冲突及待审实例，runtime落实needs_recheck，
取消未执行冲突许可但保留已确认制动，公开revalidate逐项恢复。re_yield+stable原子拒绝，不能变KEEP。
返回完成后继续使用原车道目标，其他路线分支不受影响；DONE文字明确阶段完成且纵向约束继续保留。
UE3恢复提示词覆盖可跟随/旁侧通过，仍受继续侵入、来车、通道和优先权约束，不硬编码左绕。
报告中的23个新曝光组登记Phase4 train-only，来源SHA绑定；原逐帧标注/边界证据SHA未改。
172项专项通过，原复现通过；2/4RGB各429题＋121待审、全量哈希及各7轮×world1/4采样回放通过。
任务/提示词v5、快照v5、先验source v2，人工标注仍v3；须新产物新run，旧版本不硬复用。
ready=false，全量RGB/COMMON/re_yield/分支/独立三split仍未验收；未正式4B/GPU/CARLA训练或集成。
Phase3/Action稳定默认未改；详见Phase4/REAUDIT_FIXES_20260929.md。

### 2026-09-29 Phase4 部分审计后的控制器与训练门槛修复

用户提供的只读审计仅部分完成：38条完整序列/4290帧，278单元仅7个三ID，
仍缺662条/94378帧，18单元来源不足三ID。本轮无新增目视审计、不改标注与曝光SHA。
修复COMMON覆盖漏检、许可撤销误回退独立里程碑、几何完成与减速冲突、正常NO等待误超时；
事件轴DONE不等于实例退出，纵向APPROACH/HOLD/RECOVER继续保留；复核保留已有制动约束。
回放分回答/接受/执行确认延迟及未确认右删失；handoff要求完整显式有效性判断；
训练/adapter/runtime/replay绑定2/4RGB及历史间隔。新增当前实例/边/决定的因果执行回执，
支持已开始动作的旧状态同步及新快照恢复，旧/未来/错边回执拒绝；未真实跟踪器/CARLA验收。
流水线接上独立test与审计包，单独train默认data_v4；未认证Phase3所有功能完全对齐。
138项专项通过（含原复现、小模型、观察合同及流程stub）；2/4RGB各429题＋121待审、
全量哈希及各7轮×world1/4采样回放通过。公共纵向缺额270，正式ready=false，缺完整4B。
任务合同v4，提示词/人工标注仍v3，须新产物新run，旧快照拒绝；Phase3/Action稳定默认未改。
详见Phase4/AUDIT_FIXES_20260929.md及README；未GPU/DDP/正式训练。

### 2026-09-29 Phase4 按逐帧RGB标定转移范围（覆盖同日固定半秒窗口）

用户纠正：0.5秒只是例子，必须根据不同事件/Town/路线逐帧RGB决定前后范围，不能固定扩窗。
本轮实际复查45窗、720次呈现、701张不同RGB、36物理组、10Town、十事件；含9个续窗，
新增125张不同RGB，累计1541张，复查旧图不重复算新增曝光；均属于既有train-only路线。
v3移除时间距离自动YES，改为46条逐帧reviewed_transition_band＋11条保留旧审核区间；
区分条件成立与转移后旧状态补问，明确下一阶段/新冲突/普通行进截止；未见终点标右删失。
完成类不得提前YES，窗外/不确定不补NO；修正UE3误NO、RE2新切入后的完成标签及UE2起步/横移混淆。
模型仍只看当前/候选下一状态和因果2/4RGB，回答YES/NO；动作先验/流程/路线分支保留。
新默认reviewed_state_pairs_v3.json，task/prompt v3，默认标注与审阅证据绑定合同，须新产物新run。
两模式各429道二元开发题（323 readiness、106 catchup）＋121待审；catchup不补readiness缺额。
107项专项通过，含真实小模型；2/4RGB全量哈希及各7轮×world1/4采样回放通过。
无独立val/test、缺完整分支支持/4B基座，ready=false，未正式训练/GPU/DDP/CARLA验收。
Phase3/Action稳定默认未改；详见Phase4/RGB_BOUNDARIES_V3_20260929.md及README.md。

### 2026-09-29 Phase4 状态对二元问答与半秒窗口（覆盖首版四分类/不扩窗约定）

用户明确文字只给当前状态和候选下一状态，因果条件/动作含义融入状态描述，仅回答YES/NO。
Phase4 prompt/task升级state_pair_binary_v2；速度/帧号/全流程不进入文字，因果2/4RGB保留。
新增转移点前后各0.5秒，4Hz共5帧，同一旧状态构题；明确阻挡NO、逐帧条件优先，事件/下一转移裁边。
窗口仅是用户指定训练容差，指标独立于严格readiness；未来点不进模型，窗外未知不补NO。
复用原92窗1416张RGB审阅的9个转折，未增加目视计数。55条标注产出214道二元开发题及20条待审，
UNKNOWN/INVALID仅内部审计/控制器保留，不作模型答案。全train-only，缺独立三split/完整基座，ready=false。
88项专项通过；2/4RGB全量开发题哈希核验及各7轮×world1/4采样回放通过，未真实4B/GPU/CARLA验收。
默认reviewed_state_pairs_v2.json，新合同须新产物新run；Phase3/Action稳定默认未动。细节见Phase4/README.md。

### 2026-09-29 Phase4 条件转移实验与逐帧RGB审阅

用户授权新增 `AutoMoT/qwen3vl_local/sft_new_loop_phase4/`（代码、测试、轻量审核记录与说明）到白名单。
LoRA只判断状态转移条件，动作先验由事件/纵向状态派生；覆盖十事件，保留UE2左右绕行/不返回/
本车道通过、UE4沿路跟随/绕行、RE3沿路线汇入驶出。正常事件内不重复Phase1/2，许可未执行则重查，
未知/矛盾/卡住复核，完成才退出。新实验不修改Phase3 v23_io1或Action稳定默认，不兼容旧adapter。
实际审阅92个连续窗、1416张不同RGB、81物理组、11Town；全部新增曝光组train-only，兼并Phase3旧隔离。
46个审核区间构建204道开发题；未以未来动作时刻膨胀许可。catchup独立分层，不能填readiness缺额。
代码含独立构建/预检/单卡DDP/2与4图/生成评估/预测状态回放/先验导出/demo/审计入口。
当前无独立val/test和完整分支正负标注，正式训练ready=false；缺完整4B权重，未真实GPU/多卡/CARLA验收。
专项77项通过，包含真实小型Qwen3.5的2/4图答案mask、LoRA反向与实际204题构建/哈希拒绝；另两模式各14个epoch采样计划通过。
详细证据、边界和运行见Phase4/README.md与RGB_AUDIT_20260929.md。RGB、probe_output、权重不入库。


### 2026-09-29 旧 BEV-only 评估兼容接线（补充只读诊断）

训练机报告run_20260928_101609/best.pt合同完整、train/val/test哈希一致，真实EMA在恢复
旧RoPE后严格加载/CPU前向通过；差异恰为六个已审查Qwen迁移文件和transformers4.57.3→5.3.0。
新增evaluation_compatibility.py，仅eval/compare接受指定旧→新源码SHA对与该版本对，
其它源码/运行库/BEV权重/条件/精度/标签/采样字段保持一致，split仍验hash。
共同eval入口本轮加载改动另用精确SHA对核对，避免当前qwen_simple因这一纯入口修改失效。
完整旧BEV配置恢复partial_rotary_factor=1.0、mrope_interleaved=false，原层数/头数/theta/
dropout保留；父预检与worker均核对，不靠用户报告授权。manifest/metrics记录两份身份、
兼容规则源码SHA及差异，不改checkpoint。训练resume仍原require_contract，旧Qwen基座不可替换。

专项31通过（含旧源码数值回放、单eval及comparison worker真实小网络EMA加载）；
相关共138通过、11失败：1缺只读runner、8缺matplotlib、2既有shell默认30与测试50不符。
未绕过缺失依赖、未训练机完整RGB/LiDAR或GPU验收，不宣称CUDA数值/效果等价。
使用原compare_checkpoints.sh即可；详见action_prior/CHECKPOINT_COMPARISON.md与消融run.md。

### 2026-09-29 旧 BEV-only 在 Qwen3.5 环境下的兼容性诊断

用户要求检验升级环境能否继续使用旧模型。复核发现除源码/依赖指纹变化外，旧 BEV-only
decoder_config 缺少新增 partial_rotary_factor/mrope_interleaved，直接构造会套用0.25/true，
旧 mrope section 校验失败；空 Qwen prefix 不代表 generated-token self-attention 不用 RoPE。
新增 action_prior/audit_checkpoint_compatibility.py 只读诊断：逐项源码/依赖/完整合同差异、
三split身份、原EMA严格加载和固定合成BEV输入CPU FP32 forward/Euler；完整旧配置可在内存中
尝试1.0/false，保留原层数/头数/theta。记录源码扫描独立于数据/runner可用性，不改旧checkpoint、
正式合同或比较器，报告evaluation_authorized=false，不对旧Qwen条件模型替换基座。

14项专项通过：以59a5d7fe21de66e1bae95d95fd7848cccfe64f7b真实源码在独立进程生成小模型
权重与参考，3种RoPE×token开关的条件特征/速度场/轨迹与候选实现atol=rtol=1e-6一致；
另覆盖旧默认报错、完整配置要求、缺权重/NaN拒绝、合同失败仍留报告及CLI不改原run。
比较器扩展检查51通过、10失败（8缺matplotlib，2既有shell参数30与测试硬编码50不符）。
未修改绘图/比较器/训练实现；未安装依赖。此数值对照为同一当前Torch环境的新旧源码比较，
没有训练机真实checkpoint、完整外部runner/GPU或旧新依赖环境的端到端验收。
命令及判读见action_prior/CHECKPOINT_COMPARISON.md；正式兼容放行仍待具体合同差异和同帧实测。

### 2026-09-28 修正 Qwen3.5 官方资产预检

训练机报告 transformers 4.57.3 与缺 generation_config.json。核对官方 Qwen/Qwen3.5-4B
仓库确认未提供该文件；此前将其列为必需资产是本地预检错误。现改为可选，沿用固定
Transformers 5.3.0 从本地 config.json 构建默认生成配置，不伪造模型文件、不联网补取。
版本要求仍严格 5.3.0，训练机需安装 qwen35/requirements.txt；其报告已找到两片权重，
但尚不代表完整权重内容、实际模型或GPU验收。新增无 generation_config 的真实小模型
离线加载/生成、预检可选资产与缺必需处理器拒绝测试；专项52项通过，diff检查通过。
此修复在956b55c4c之后新增；训练机需同步本次预检修正及固定依赖。

### 2026-09-28 Qwen3.5 复核补齐：system 缓存、选优与 K/V 模式

system-prefix 纯文本复用走 inference_mode；增量 helper 返回本条缓存传入的 rope_deltas，
避免模型对象上其它图像分支的 delta 污染后续 decode。真实小模型覆盖先图像 delta=-2、
后同 system 文本缓存 delta=0、连续三步 decode 与完整重算 logits 一致，未靠重算清除旧值。
Action available/strict 两种选择均在打分前比较当前本地基座资产；validate_adapter 即使
只检查目录也拒绝缺资产合同/权重身份。一次扫描只哈希一次基座，显式路径及最终加载仍核对。
新增 Phase1/2 高分不兼容候选、缺合同/缺权重哈希拒绝回归；原跨机器来源警告同步准确表述。
GoalGen 完整/latest 保存 qwen_kv_segment_mode，评估及 init-from-ckpt 共用严格校验；
兼容读取已保存 args 中的模式，缺失/冲突/不同模式拒绝，8 段默认同样遵守合同。

专项与相关扩展回归 753 passed、4 failed、5 deselected；四项打包失败均明确为缺少外部
mot_lead_offline_runner.py，未关闭来源检查。五个排除项为已记录的 runner、缺数据及 shell
fixture 问题。Qwen3.5 专项现 50 项通过；另 v5 KV helper、GoalGen 两个 CLI 与12文件语法通过。
测试日志 /tmp/qwen35-followup-final.log；git diff --check 通过。无真实4B/GPU/吞吐验收，
完整模型与外部 runner 仍缺失，preflight ready=false；未 commit/push，冻结 release 未改。

### 2026-09-28 Qwen3.5 复核修复：padding 与内容身份

针对复核的三个问题补齐修复。DeltaNet 保留 cached suffix mask，仅有效 token 推进卷积/
循环状态，scatter 保留 v5 右 padding 打分位置；整行空 suffix 不改变状态或 next logits。
基座身份增加实际 safetensors/全部分片的流式 SHA256，同时绑定 tokenizer、模板及处理参数；
backend 更新 qwen35_local_v2，拒绝缺权重身份的 adapter，旧 v1 不自动迁移。
LeadMoT/GoalGen 共用 v2 backbone 合同；GoalGen 完整与 latest checkpoint 保存合同，
评估及 init-from-ckpt 在载入 DiT 参数前核对；同路径替换拒绝，同内容搬迁允许，旧 mismatch
开关不能跳过校验。权重哈希增加启动读取，ragged DeltaNet 分支按样本处理，未验证 CUDA 吞吐。
新增代码/回归仍在已有白名单，vendor 来源清单同步本地补丁 SHA；冻结数据 release 不改。
最终相关 CPU 回归 732 项通过（Qwen3.5 专项 46 项），另 v5 KV helper 与 GoalGen train/eval
CLI 检查通过；覆盖 v5 真实小网络 Q1→Q2 状态/logits/梯度、同结构换基座、
分片替换/缺失、规划模板/adapter 内容替换与搬迁、GoalGen 实际保存/评估路径；无真实4B/GPU验收。
完整权重和外部 runner 仍缺失，preflight ready=false；临时日志 /tmp/qwen35-review-final.log。
离线运行与兼容边界见 qwen35/README.md；本轮未 commit/push。

### 2026-09-28 Qwen3.5-4B 本地源码迁移

按用户明确要求，当前 SFT/Phase1–3、Action/qwen_simple、GoalGen、LeadMoT 和道路事件探测
默认基座改为 `AutoMoT/checkpoints/Qwen3.5-4B`。包名与 engine 旧导入别名保留兼容；
bev_only 仍无 Qwen。冻结 Phase3 release、历史审计和旧 checkpoint 不改写，数据语义仍 v23_io1。

新增 `qwen3vl_local/qwen35/`：复制官方 Transformers 5.3.0 的模型/配置/tokenizer/视觉处理器
共七个源码文件，保留 Apache 许可证、上游 wheel/源码与本地 SHA256 清单；修改只在项目副本。
运行时显式导入本地类，强制 offline/local_files_only、禁用 remote-code/远程媒体/Hub kernels。
固定通用框架依赖，在独立临时环境验证，未修改原 pvi 或全局 site-packages。

适配 32 层 hybrid cache（8 个 full-attention 层，K/V 4×256）、线性层卷积/循环状态、
单/多 token 续写及分支复制、partial interleaved M-RoPE、左 padding、多图 processor、
非思考模板与答案 loss 边界；保留历史空 think 使模板与缓存前缀一致。
修复固定上游版本的多 token 缓存状态重置及新纯文本 prefill 沿用旧视觉 rope delta。
LoRA 加入 DeltaNet 投影；GoalGen/LeadMoT 默认 8 段/8 block/4 heads，宽度仍 1024。
Action 新增本地 runner bridge，外部 runner 源码仍缺失，不能据桥接单测宣称其端到端兼容。
新 adapter 保存 backend/基础资产合同，选择/复制/打包携带合同，执行指纹包括本地副本。
旧 Qwen3-VL LoRA 和规划 checkpoint 不兼容新基座，须原源码恢复；本次升级须新 run。

验证：联合 CPU 回归 768 passed、4 deselected；随后新增真实 Phase3 binary/choice
小模型 LoRA backward 两项均通过，本地 Qwen3.5 专项共 32 passed（已含于累计 770 项）。
覆盖小型随机真实网络、多图非零 RoPE delta、缓存续写/梯度、padding/分支、原生 generate
对照、官方模板/processor、2/4 图 loss-mask、禁止 socket 联网的本地 save/load；并覆盖
Phase3、Action 训练恢复、稳定 release 等现有回归。15 个 CLI 在仓库外 cwd 的 --help 通过。
早期扩展回归 5 项失败：3 项缺外部 mot_lead_offline_runner.py；1 项 preflight 测试触发
数据准备但缺 lead_data；1 项 shell fixture 未复制 prepare_event_balance.py。最终联合检查
排除前四个 available_adapters 用例且未包含 test_lora_bundle；没有绕过生产合同或伪称全绿。
日志位于本机 `/tmp/automot-qwen35-source/`（临时诊断，不作为持久产物合同）。

本机无完整 Qwen3.5-4B 权重，亦缺外部 runner；只读 preflight 明确 ready=false。
未执行真实 4B/GPU 训练、CUDA 加速核验收、真实数据效果对比或完整 Action/闭环验收。
训练机需完整本地模型资产、固定依赖及原 runner 后再验证；不声称模型效果或速度已提高。
源码来源、离线安装/预检命令、迁移差异详见
[Qwen3.5 本地运行说明](AutoMoT/qwen3vl_local/qwen35/README.md)。

### 2026-09-28 Action pending mkdir EEXIST 恢复

准备器先检查.pending再mkdir，创建阶段EEXIST原未进入缓存恢复，导致训练前退出。
新增锁内有限重试，每次重新核验ready/合同/hash；完整结果续发，残缺结果隔离，保留原锁。
日志不足以确认训练机底层是可见性延迟还是外部写入；未连接训练机或执行GPU验收。
124项相关CPU回归通过（新增18项），覆盖两阶段冲突、mkdir ESTALE、持久失败及产物字节等价。
仅改外层prepare_event_balance.py；v23_io1快照、语义、mapping及缓存身份不变，旧run合同仍严格。
部署范围、恢复操作与验证边界见action_prior/run.md同日条目。

### 2026-09-27 Action launcher父进程导入路径修复

`python path/launch.py` 原只设置子进程PYTHONPATH，当前进程sys.path仍缺AutoMoT，
导致probe子评测和audit.zip已完成后，父进程延迟import audit_bundle仍报找不到qwen3vl_local。
现启动时同时按__file__初始化当前sys.path和子进程环境；不依赖调用cwd、用户export或pytest路径。
独立进程在AutoMoT内/外均复现旧失败，修复后父进程真实打包通过；另覆盖12个实际CLI入口--help。
118项相关CPU回归通过，1项因缺只读runner的既有合同检查排除；未GPU/训练机验收。
这是工程修复，不改v23_io1快照、提示词、标签、采样或mapping合同；旧run源码合同仍严格校验。
`dataset_label_missing`是独立的逐帧先验缺失计数，不是本次Python导入异常。
详见action_prior/run.md同日条目；新增测试在Action已授权目录。


### 2026-09-27 区分工程修复与效果版本（覆盖同日逐文件回退约定）

用户明确：ESTALE、缓存续发、异常恢复等不改变模型语义的工程bug修复可保留/回移到稳定版，
不能因修复最早出现在v24就随效果实验一起撤回。工程修复需故障回归、正常路径输出等价及代码复核，
不要求先证明GPU分数提高；提示词、标定、标签、阈值/窗口、过滤/split、采样分布、优化/模型条件等
会改变数据或学习行为的改动仍需效果验证才能替换默认，不能仅以“bug修复”名义绕过。
当前语义基线仍v23；Action稳定发布为v23_io1（v23＋ESTALE工程修订），三入口同步。
补回Phase3候选/索引/元信息和并行扫描的有限重试与已提交rename核验；Action外层ready/哈希续发保留。
旧v23快照不改，新工程revision也绑定源码/manifest/hash；不掺入v24 prompt/标定/smooth_cap/完整训练池。
工程发布走独立engineering_fix审查分支，保护提示词/轨迹规则/采样/标注文件；改动构建器还必须审查输出等价。
本轮908项相关CPU回归通过（含15项原v23/工程修订产物等价与故障注入检查）；未真实挂载/GPU验收。
mapping身份因源码及filesystem依赖绑定变化而更新；旧索引不可改hash硬复用，旧run须原源码。
详见 [稳定版本约定](AutoMoT/qwen3vl_local/action_prior/PHASE3_STABLE_RELEASES.md) 与
[工程维护记录](AutoMoT/qwen3vl_local/action_prior/V23_IO_MAINTENANCE_20260927.md)。


### 2026-09-27 初次回退记录（工程部分已由上方同日修订覆盖）

用户明确指定v23为当前稳定基线。v24四包审计与回退前源码已留档；Phase3 Python/shell恢复
b433aa605的prompt/构建/训练/评估/采样，默认data_v23、v23_grounded_stage。
Action主线/qwen_simple/bev_only统一经phase3_stable读取phase3_releases/stable.json，目前v23；
禁止直接跟随Phase3实验目录、最新版本号或仅通过CPU审计的v24/v25。快照及外部依赖验SHA，
缺失/不匹配拒绝，不回退实验源码；缓存/训练合同记录稳定release，新旧run继续严格隔离。
只有真实同题评估相对当前稳定版改善、关键分组及Action影响复核并正式晋升后，三入口新进程才自动同步；
运行中进程及builder子进程固定原版。旧checkpoint用原源码，不因晋升热换条件。
Action采样回到v23事件/事件内动作容量回流；独立指定116256/cap11保留，容量不足拒绝。
不得把跨题集历史差值当因果改善，不得把曝光路线重称盲测；本轮未GPU效果验收。
细节、归档位置与晋升流程见
[稳定版本约定](AutoMoT/qwen3vl_local/action_prior/PHASE3_STABLE_RELEASES.md)及PROJECT_CONTEXT同日记录。
新增release加载器、稳定快照/manifest、测试和说明均属于已授权Action目录；冻结内容禁止随实验修改。
本轮相关1064项CPU回归通过；Phase3的80个Python/shell文件与v23逐字节一致。
更广检查剩余37项未通过，涉及缺只读runner/peft/matplotlib及4项已在回退前归档源码复现的既有测试问题；
未绕过生产校验，日志与restore_record.json保存在本机源码归档目录。


当前稳定登记：`action_prior/phase3_releases/stable.json`。v23 manifest SHA256：
`d83d0a5a43cab1368cd45a7541985a28b86a3206b1ad9ee4bc949b322a3f1945`。
冻结的运行模块/标注及外部依赖逐一验SHA，内部导入/目录根位置做迁移适配；原mapping/rule身份保留。
`phase3_release.py` 提供show/promote，晋升检查候选和报告哈希、当前baseline、同题评估声明、
关键回归复核及主要指标严格提升，随后加锁原子发布；不能代替审查人核验真实证据。
`phase3_stable`只从登记快照加载，构建子进程继承已验证selection，缓存和训练合同绑定release身份。
当前v23历史例外来自用户明确选择，不伪造跨版本同题实验；晋升不热更新运行中模型，也不迁移旧checkpoint。
本机源码归档：`checkpoints/phase3_version_archive/v24_20260927_before_v23_restore/`；不入远程。

### 2026-09-26 Phase3 binary 根据服务器实际池进行 INVALID 来源内容量回流

用户补充真实报告：DYNAMIC_CUTIN/true-R4/prompt-R3的七个asked-context细分组共用3帧，
目标54而cap8容量24，缺30；固定人工题52，总呈现12288。来源总配额容量上界可行，但此前未保覆盖。
用户要求各服务器数据略有差异也应兼容，继续保留正例每类1024与全局cap8，不缩轮。

新增训练专用invalid_capacity.py。压缩图把INVALID细分组接到来源父节点，来源预算保持；
原非空细分组通过硬下限边各保留至少一次。正例context预算不变，人工题固定且占用物理帧容量。
图考虑正例/自动/人工共享帧，整数费用先最少细分配额偏差，再原正例动作偏差、重复和历史公平性。
只返回调整后的细分配额；实际选帧/路线队列/游标和历史仍由原hierarchical sampler负责。
不固定服务器路径、帧数、配额数字或缺额；同来源容量可重分配则继续，否则保留明确容量错误。
不会跨INVALID来源借额、丢掉原非空细分覆盖、修改标签或屏蔽坏索引校验。

train只在binary非cycle_even且原配额容量失败后调用。正常成功路径不变；
重试恢复选样前随机状态，失败不提交游标/历史。rank0的phase3-capacity-reallocated与epoch
审计记原/新配额、转移次数、来源总额、覆盖下限；不可行报告标明coverage_preserving_reallocation。
sampling_config保存回流版本和独立源码SHA256，进run/adapter/epoch配置。
未更改sampling/build/source_mapping及其哈希输入，本补丁不要求重建当前匹配索引；
先前版本不匹配仍严格拒绝。新训练使用本源码，已有run原源码。

113项相关CPU回归通过（已有pvi Python/torch环境）：
合成七组共享2/3/4/8帧，分别配52/52/53/0条人工题，四种数据差异各7轮；
每轮正例各1024、总12288、cap8、INVALID各来源总額与人工具体集合不变，
需要回流时分别转移38/30/22次，8帧时保留原成功路径；
另含45个随机小池穷举最小偏差、输入顺序不变性、正例与人工共占负例帧、跨来源借额拒绝、
覆盖下限不足拒绝、训练配置合同及原binary/INVALID/构建/validation/support回归。
未访问训练机完整候选或跑GPU；实际服务器须对本机匹配索引sampling-only预检，
通过后可复用原四组pipeline，已有完整匹配索引用SKIP_BUILD=1。

### 2026-09-26 Phase3 binary 固定配额容量失败诊断

用户提供四卡CPU采样预检失败：joint请求12236、feasible12206、repeat_cap8；前置same-RS零路线
提示明确Training allowed，不是中断原因。用户选择保持每类1024和cap8，只排查是否能重新分配。
现有最小费用流已经允许共享帧反向重分配和同事件动作回流；当前候选/固定细分配额下缺30次
不等于全train缺30张不同图，不能靠改shuffle宣称解决，也不自动降低预算/提高上限/变更标签。
只在train._balanced_work容量失败分支调用新增capacity_diagnostic：独立无费用压缩最大流，
残量最小割给出bottleneck_events、bottleneck_unique_frames、bottleneck_target/available_presentations，
并区分预留人工题和联合分配次数。minimum_repeat_cap仅诊断数值，绝不实际提高上限。
另报告合并INVALID细分组但保持来源总配额的乐观容量上界；signature/prompt-RS覆盖未保证，
标记executable_plan=false，不能当符合原覆盖约束的解决方案；不足则说明仅放松这些细分配额也无解。
错误仍为ValueError子类、打印rank0的phase3-capacity JSON；成功路径无诊断调用/随机数或游标变化。
66项CPU回归通过（已有/home/codon/anaconda3/envs/pvi含torch），覆盖最小割、预留、缺池、
随机小池穷举容量/最小cap、真实_balanced_work失败保持状态和原support/validation回归。
12236/12206/差30的数值测试是合成拓扑，不能认作远端实池复现；未读取训练机索引、未GPU训练。
采样器、构建器、mapping hash依赖文件和prompt未变，此诊断补丁本身不要求重建匹配索引。
保留此前版本合同检查，旧run仍按其源码要求；不自动push。运行命令见Phase3运行说明。

### 2026-09-26 Action / Phase3 数据产物发布 ESTALE

训练机日志定位：BEV-only全流程在Phase3扫描完成、写出15180行索引后，候选目录rename返回Errno116。
真实shell回归确认bev_only/qwen_simple的event/action两模式均调用共享prepare_event_balance。
不能仅凭日志确认具体挂载类型/服务端原因，也不能保证原始数据读取和模型保存永不遇到ESTALE。
新增action_prior/filesystem.py：0.5/1/2/4秒有限退避、发布前SHA256快照、已提交rename结果复核，
元信息写独立临时文件后原子发布；权限、空间、EIO等其它错误仍失败。
接线覆盖Action三split/full map/prior labels与Phase3候选、训练池、frame index、并行扫描及相关元信息。
prepare_event_balance和prepare_action_priors共用完成产物保留机制：.pending-<最终缓存名>/index，
ready.json绑定目标及payload/manifest哈希，同版本持锁重跑验证后续发，不再扫描完整缓存。
未完成、损坏或来源变化不能盲目续发；子构建器非零退出也保留残留用于诊断，下次隔离重建。
独立Phase3构建的文件发布是当次重试，不新增跨进程全扫描恢复；原始流式读写、checkpoint和模型IO
仍可能因挂载故障失败，不对这些非幂等操作包住整轮重试。恢复挂载后按原有效合同重跑。
标签阈值/采样规则不变，但build_dataset/source_mapping/helper源码进入mapping哈希；新索引/full map、新run，
旧run须原源码。此次不能只同步prepare_event_balance.py，运行文件清单见action_prior/run.md。
203项CPU回归通过：文件与目录短暂/持续ESTALE、已提交rename、核验再遇ESTALE、其它errno拒绝、
损坏恢复状态、来源变化、真实flock、Phase3真实写入流程及两个消融真实shell传参（重型Python入口用stub）。
此前7项event_balance扩展检查和本轮training_pool_audit模块收集因当前Python缺torch受阻；未绕过生产校验。
未在报错机器真实挂载/GPU验收；findmnt -T checkpoints/action_prior_prepared 可定位挂载交存储管理员检查。
旧版可能清理临时产物，且新hash不能复用旧身份待发布缓存，首次升级不能保证免扫描。

### 2026-09-26 Phase3 v24 连续RGB与训练结果复核

四份20260923包对最近v21总分表观提高，但测试逐题交集为0；STOP/RESUME进步，RIGHT/KEEP退化，
choice减速仅30/100与33/100；不宣称全面超过历史最好或通过production守卫。
连续复核60段、58物理组、1020张不同RGB（全三视图、每段17帧，14正确对照/46错误），另五段原分辨率复看。
保留60段原标签，不能由定向样本宣称全池零错标；暗光/对象可见性五段仍标未解决。
近停诊断分开单点已释放、截止确认、晚起停车对、窗末未知，不再把旧isolated字段都解释为短暂停顿；
action_review及两类边界审计共用，未来/控制不进入提示词。v9动作阈值、窗口与主动作优先级未改。
prompt v24_motion_reference只改共享速度说明：最新速度基准、瞬时近停与等待；保持原610词测试预算。
新data_v24，176个已曝光test/导出val物理组加入train-only，累计1955；mapping合同绑定名单。
460题/117录制路线/15297meta回放，原动作/纵向判定0变化，1840次两题型/两图数回放通过；非全train审计。
760项相关CPU测试通过；历史190000候选划分回放移动64组，val/test各类≥32帧/5组，非当前生产重建。
未跑GPU新训/新模型效果验收，不能声称效果改善；索引/full map须重建并新run，旧run原源码。
报告及逐段观察见 [V24_RGB_CALIBRATION_20260926.md](AutoMoT/qwen3vl_local/sft_new_loop_phase3/V24_RGB_CALIBRATION_20260926.md)，
指标详表见 [AUDIT_COMPARISON_20260926.md](AutoMoT/qwen3vl_local/sft_new_loop_phase3/AUDIT_COMPARISON_20260926.md)。


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


本文只保留新会话改代码前必须知道的项目事实。细节以源码为准；不要把长源码片段复制到这里。

### 2026-09-19 Action high-level 场景目的

`action_prior/prompts.py` 借鉴 Phase3 v12，为 high-level planning 的已接受事件/独立 transition
增加目的说明：统一减速/停车动作在 UE1 为保持跟车距离，在 UE2 为观察相邻车道车辆、接近趋势、
借道时的对向来车及通过空间。RE2 各阶段分别描述，RE3 关注车辆位置/相对运动与目标空隙。
目的不构成空隙或下一动作证据，不建立固定动作顺序；并发事实/目的去重保留。
同时开启 high-level-action-prior 时，selected 动作释义保持统一；目的边界只说明一次，不追加重复关联句。
动作门控、RE 独立 scene gate、binary 多标签语义及标定均未因此修改。普通背景/关闭模式不增加目的。
planning 版本为 `phase3_inspired_conditional_high_level_v3_compact_purpose`，现有 config 身份计算
将其传入 checkpoint 与文本缓存合同。prefill/生成/复核共享 user 目的，fallback 保留短预算。
58 项 CPU 回归通过，缺 torch 未执行完整 runtime/GPU 验证；新代码用于新 run，旧 checkpoint 用原源码。
本轮压缩通用规划/目的约束并删除动作关联重复句：R1+UE2 场景文本 120→96 个空白分词，
加 STOP、排除导航/区块标签为 164→126；这是英文空白分词口径，不是模型 token 数。
上述为输入模板长度，不设 80 词输入上限，也不按该预算截断。80 词仅约束显式开启的分析摘要及
fallback；默认 base prefill 跳过这些分支。LoRA 来源的上游 Phase1/2 问答生成仍保留。
详见 `action_prior/DESIGN.md`、`action_prior/run.md`。

### 2026-09-19 Phase3 v12 默认配置

`sft_new_loop_phase3` 新训练/base eval 默认 `4rgb + choice`，两帧/binary 可显式覆盖，索引目录 v12，
prompt 为 `v12_compact_context_purpose`。沿用 v11 紧凑措辞，每题在场景前提下加一句目的：
UE1 保持跟车间距，UE2 观察相邻车道、接近车辆（借道时含对向来车）与通过空间寻找绕障空隙；
减速不推出变道，目的不作为下一动作证据。其余八类按 context 给对应条件性动机。
标定、映射决定与划分规则不改；提示词/源码合同变化要求独立重建 v12 索引并新训。
LoRA eval 仍恢复保存模式，旧缺字段配置保持四图/binary 回退语义及严格 prompt 检查。
choice 仅支持有效单动作，不能替代 action_prior 的多标签输入；本次不修改 action_prior。
无新增人工 RGB 审计。v11 四帧 choice 581/762=76.25%、双帧 589/762=77.30% 为历史参考，不是 v12 成绩。
同划分用于开发对照，已参与版本选择的 test 不能称为独立新测试。
入口与边界：`sft_new_loop_phase3/SFT_NEW_LOOP_PHASE3_RUN.md`、`V12_DEFAULT_20260919.md`。

## 0. 项目目标

把 `lead/` 采集/训练出来的 CARLA 离线数据，整理成本地
Qwen3-VL-Instruct frozen prefill + LeadMoT / GoalGen decoder 能直接消费的输入，
并逐步分析 RGB、LiDAR、BEV、target_point、prompt 与训练分布差异。

主要战场：

- `qwen3vl_local/`（从 `AutoMoT/` 当前目录看）
- `AutoMoT/keyframe_filter/`
- `PROJECT_CONTEXT.md`

`AutoMoT/Automot/` 与 `AutoMoT/leaderboard/team_code/` 仅作为本地参考源码，
不再由本仓库追踪或推送；下文涉及其中实现的内容只表示技术背景，不表示 Git 白名单。

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

### 2026-09-16 Git 历史清理

按用户要求从远程历史移除 8754 个旧审计产物、索引及 ZIP 路径；清理时两个分支的
527 个不同提交逐个核验，当前文件 tree、作者、日期、提交消息与父子关系保持不变。
提交 SHA 已重写，21 个旧签名保存在原始备份中。新远程完整克隆对象包约 8.63 MiB
（清理前约 84.22 MiB）；随后远程 `tune-batched-training-defaults-h20` 已删除，仅保留 main。
清理后 main 基点为 `3c7e627b71bda5d549f04bbd6f870d45298b37ca`，后续提交在此基础上继续。
本机备份目录为 `/home/codon/git-cleanups/automot_lead_20260916_130159/`，其中
`before-cleanup.bundle` 保存原历史，`commit-map.tsv` 保存旧/新 SHA 对照，`report.json`
保存验证结果；这些是仓库外本地备份，不随代码 push，其它机器不能假定该路径存在。
旧 checkpoint 如需原源码，使用隔离备份回查，不把旧历史合并回 main，也不修改其合同。
日常 push 的分支、白名单和全历史审核要求见 AGENTS.md / CLAUDE.md 同名 Git 规则。

## 1. 目录角色

| 目录/文件 | 角色 |
|---|---|
| `lead/` | 数据采集、训练、闭环评测参考仓库。只读 |
| `AutoMoT/` | 在线驾驶仓库；本仓库只追踪明确列入白名单的本地改造 |
| `AutoMoT/Automot/` | AutoMoT 原始实现的本地只读参考；不修改、不追踪、不 push |
| `AutoMoT/leaderboard/team_code/` | leaderboard agent/runner 的本地只读参考；不修改、不追踪、不 push |
| `AutoMoT/lead_data` | 远端 LEAD 数据软链接入口，等价于用户在 `AutoMoT/` 下执行 `ln -s /datashare/IOL4SGH/data/data/* lead_data/` 后的目录；运行命令里用相对路径 `lead_data` / `lead_data/keyframes_all_scenarios.json` |
| `AutoMoT/data/lead` | `lead_data` 对应的 route XML 根目录，由 `AutoMoT/data/data_routes` 提取整理而来；命名规范固定为 `data/lead/<Scenario>/<Town>_<route_key>.xml`。旧数字 route 使用 `Town03_route_001783.xml`，新版子编号使用 `Town12_route_1054_0.xml`，命名本身带 Town 的 legacy key 使用 `Town06_route_Town06_13.xml`，legacy key 内部带 route 编号时保留完整 key，如 `Town12_route_Town12_route15.xml`。从 `lead_data/<Scenario>/<run_id>` 找 XML 时，`Scenario` 必须取 run 的父目录；run_id 先剥末尾 `MM_DD_HH_MM_SS` 时间戳，再只在存在时剥尾部采集后缀 `_route0`，剩余部分就是 route_key；`Town12_route15` 这类 legacy key 本体里的 `route15` 不能剥，也不能要求它带 `_route0`。XML 文件名公式：`route_key` 以 `route_` 开头时用 `<Town>_<route_key>.xml`，否则用 `<Town>_route_<route_key>.xml`。2026-07-03 全量核对：`lead_data` 9715 个 run 去重后 9294 个 `(Scenario,Town,route_key)`，`data/lead` 正好 9294 个 XML，缺失 0、冗余 0、命名不规范 0、XML 解析失败 0、内容结构异常 0；XML 内 `<weathis_juncer>` 拼写已统一修正为 `<weather>`。40 个 XML 的 `data_routes` 源文件位于不同 scenario 目录（36 个 `noScenarios`、4 个 `ConstructionObstacleTwoWays`），不是缺失；另有 `ParkedObstacle/Town12_route_Town12_route15.xml` 覆盖有效并与 `lead_data/ParkedObstacle/Town12_Rep0_Town12_route15_*` 对应，但未在 `AutoMoT/data/data_routes` 找到直接源文件。使用时以 `lead_data` / `data/lead` 的 scenario 目录为准，不能把该项当作 XML 缺失。 |
| `AutoMoT/lead_video_tools/` | 按用户同意新增：LEAD 离线 RGB 视频转换工具。只读 `/datashare/IOL4SGH/data/data/<Scenario>/<run_id>/rgb/*.jpg`，按 4Hz 生成 `/data/lead_video/<Scenario>/<run_id>/{input,left,front,right}.mp4`（默认 input，`--views` 可选三视角裁剪），默认在左上角写 frame id，支持异常 route 剔除、断点续跑、ffprobe 完整性检查和 `--workers` route 级 CPU 并行（`--workers 0` 自动按 CPU 估计）；`rgb_to_video.py` 普通转换默认剔除异常时长 route；`abnormal_duration_filter.py` 按硬规则输出异常采集名单到 `lead_video_tools/abnormal_duration_filter/`：4Hz 下 `frames >= 361`（严格大于 1 分 30 秒 / 90s）且不在白名单内的 route 全部视为异常并写入 `abnormal_confirmed_over_90s.txt`；`BlockedIntersection` 与 `ControlLoss` 是唯一时长白名单不写入名单；`Accident`、`park*`、`dynamic*` 不再有 90-100 秒存疑段豁免；`abnormal_possible_90s_to_100s.txt` 只为旧接口兼容保留，正常应为空。凡是 `AutoMoT/keyframe_filter`、`AutoMoT/qwen3vl_local` 或其它入口使用 LEAD 数据集，都必须在构建样本、调研、probe 前先剔除这些异常 route；筛选时打印 discover + route 级进度条，两个 txt 名单只保留 `Scenario/run_id`，帧数/秒数/RGB 路径/视频目录保留在 `abnormal_duration_summary.json`；只有显式传 `rgb_to_video.py --abnormal-route-list-dir ...` 才只对筛选目录里的异常 route 生成巡检视频 |
| `AutoMoT/keyframe_filter/` | 按用户同意新增到 clean push 白名单：旧版 LEAD 关键帧选择器与新 ROAD/EVENT 语义重标注方案目录。旧 `rule_based_keyframe_filter.py` 按 scenario 固定抽 initial / 3 middle / final，主要依赖 `metas/*.pkl` 的 `dist_to_*`、speed、accel/brake，缺失时 fallback 到 bbox / RGB motion；适合作为突发事件 span 提议器和验证工具输入，不再作为最终帧级 STATUS/SUBGOAL 真值来源。`classifier_logic.txt` 是用户逐场景调研得到的道路结构与事件分类草案；`ROAD_EVENT_CLASSIFICATION_PLAN.md` 是 ROAD/EVENT canonical 总方案，已合并 ROAD_STRUCTURE 调研协议、runtime 门控和错帧回查流程；`ROAD_EVENT_CANDIDATE_MAPPING.md` 保留为 Qwen/probe 可解析的候选表；`ROAD_EVENT_RGB_AUDIT_ARCHIVE_202607.md` 归并 2026-07 一次性 RGB/RS/EVENT 审计记录，旧散落审计 MD 不再恢复；`COLLECTION_OUTPUT_INDEX.md` 说明 `collection_output/` 大产物、代码读取关系和白名单边界。代码、方案文档、规则配置、README、HTML/CSS/JS、verification 工具和手写说明允许修改、追踪、commit 和 push；`collection_output/` 默认仍是本地数据/审计/证据产物，不入库、不 push，唯一例外是 Phase1 四问标签轻量 JSON/JSONL：`phase1_four_question_answer_table.json`、`answer_table_partial.json`、`manual_visual_audit_notes.jsonl`、`除 no_scenarios_batch 外的 *_batch/phase1_four_question_matrix.json`、`full_route_rgb_label_review_20260809/manual_full_sheet_notes_20260809.jsonl`、`full_route_rgb_label_review_20260809/manual_table_gap_combo_notes_20260810.jsonl` 可精确 add 和 push；RGB contact sheet、montage、candidate anomalies、route/town/scenario/global summary 等证据产物仍不入库。顶层旧证据产物 `rgb_r4_r5_audit_results/`、`AutoMoT/keyframe_filter/keyframes_all_scenarios.json`、`R2_ROUTE_RGB_REVIEW_INDEX_*.csv`、`ROAD_EVENT_INTERRUPTED_OVERLAY_*_IDS_*.csv`、`ROAD_EVENT_INTERRUPTED_OVERLAY_IDS_SUMMARY_*.json` 已清理；若后续重生也默认不入库、不 push，需要共享时先沉淀为文档或小型配置。 |
| `qwen3vl_local/`（`AutoMoT/` 主目录内） | 本地 Qwen3-VL-Instruct frozen prefill、prompt、GoalGen、LeadMoT；`tb_serve.sh` 是通用 TensorBoard 启动器 |
| LeadMoT frozen Qwen adapter 合同 | `--qwen-adapter-dir` / `QWEN_ADAPTER_DIR` 把 LoRA merge 到内存中的 frozen Qwen，仍只训练 decoder；checkpoint `qwen_backbone` 绑定 base config 与 adapter 实际权重 SHA256，eval/probe/eval_carla 自动恢复并拒绝错配，旧 checkpoint 无合同时只允许 base。base/LoRA 必须同 seed 分别训练 decoder，不能临时切 prefix。 |
| `qwen3vl_local/sft/` | SFT 数据、训练、eval、probe（统一一套，已废弃 v1/v2 双轨与 ms-swift） |
| `qwen3vl_local/sft_v2/` | 新版 SFT v2 串行选择题路线：SCENE → STATUS/SUBGOAL，无 ANALYSIS teacher |
| `qwen3vl_local/sft_new_loop_phase1/` | Phase1 + Phase2 融合 YES/NO 路线：同一轮 prompt 固定回答 `HIGHWAY/STATIC_OBSTACLE/VULNERABLE/TRAFFIC_LIGHT_ABNORMAL`，并嵌入 Phase2_augment 的 `all_random_order` / `subset_random` / `hierarchical_probe` 三类 ROAD_STRUCTURE 问法；训练按 4:1:1，eval/generation 按 2:1:1。数据构建沿用 Phase2_augment 最新过滤（异常 route、full-frame RGB review 覆盖、默认剔除 visual-risk），Phase1 标签取已审计四问答案表并支持显式 visual subgroup override，但覆盖只来自结构化 RGB audit notes/annotations 中的 visual/topology subgroup，不能从自由文本 `audit_evidence` 推断 route 标签；JSONL 的 RGB 路径默认保存为相对 `--data-root` 的路径，train/eval 支持 `--data-root` 重映射旧绝对 `lead_data` 路径；Phase2 标签取逐帧 RS 标注；冲突处以 Phase2 最新 ROAD_STRUCTURE 定义作为 RS 权威，Phase1 审计标签仍作为独立可见事实标签。训练/eval 使用双层采样审计：Phase1 四问 focus YES:NO=1:1，Phase2 四问 focus (`RS1/RS2/RS4/RS5`) 也 YES:NO=1:1，并在此前提下迁移 all/subset/hierarchical augment balance key、多边际配额、variant report、answer-pattern diagnostics、subset 未问行泄漏检查、`RS_HIGHWAY` 与 `GROUP:<id>` 指标；all/subset/hierarchical 三类 variant 总量、Phase2 `(focus_bucket, variant)` 配额和 `all_random_order/RS*:YES|NO` 桶都是硬约束，subset/hierarchical 具体 augment key 逐桶偏差写入 deviation report；四个 Phase1 focus 与四个 Phase2 focus 总量 1:1。默认 `FOCUS_BALANCE_COUNT=9216`，对应每轮 147,456 sampled cases，与旧 Phase2 augment 总 case 数对齐；Phase1 桶先自然抽样，只按 all-random 的全局 RS 缺口从兼容 focus 的未用样本换入，不循环稀缺二级子桶；all-random 用容量匹配精确分配 YES/NO；默认 `MAX_TRAIN_FRAME_REPEAT=10`，任一 sampled frame 单轮复用超限会在模型加载前中止。训练 balance 记录每 epoch `balance/epoch_*.json`、窗口 `augment_counts`、`all_random_order_target_deviation`、`phase2_focus_variant_*` 和重复率审计；训练期 teacher/generation eval 与 checkpoint 默认步频为 2000/2000/20000，generation eval 默认 `generation_eval_balance_count=16`，运行见 `SFT_NEW_LOOP_PHASE1_RUN.md`。 |
| `qwen3vl_local/sft_new_loop_phase2/` | 新 Phase2 单轮 EVENT YES/NO 路线：保留 `sft_loop_phase3` 的逐帧 RGB 数据、异常 route/视觉覆盖过滤、LoRA DDP、频繁 eval/TB 与 case audit，但完全删除 synthetic Phase2 ROAD_STRUCTURE user/assistant 和 KV prefix。实际对话严格只有 system + 当前 RGB/历史 RGB + 一个 EVENT user prompt；`question_domain` 仅为数据采样/审计元数据，不向模型泄露已回答 RS。道路域问 UE1/UE3/UE5，路口域问 UE6，每题同时输出 `INVALID_EVENT_CONTEXT`；UE1:UE3:UE5:UE6 在 train/val/test 精确 1:1:1:1，RE 默认数量等于一个 UE 桶，且默认 25% 专门来自 R3/highway 作为 valid all-NO hard negative。invalid 默认约占基础有效样本的 20%，只由道路域与路口域明确错配构造，并要求所有 UE=NO、invalid=YES；同域 RE、拥堵、弱证据、无目标 UE、高速道路均保持 valid。UE2/UE4/UE7 由新 Phase1 承担，UE8 折入同域 RE。RGB 路径相对 `--data-root` 保存，train/eval 支持重映射；teacher-forced loss eval / generation eval / checkpoint 步频为 2000/2000/20000。默认 generation eval 每桶 32 条；UE3 recall 默认门槛为 0，只统计不阻断 checkpoint/流水线，`run_full_pipeline.sh` 在有效的 `best_generation/`（含 adapter 配置）存在时优先使用它继续完整 eval 和压缩；否则使用本轮 `final/`，即使没有 `best_generation` 也不得停在训练结束。旧 v3 冻结 multi-seed/unseen、UE3 rescore/route-balance 与失败 adapter 的 LeadMoT A/B 可执行链已删除；历史成绩文档只作证据。历史严格可比基线中，v3 production/audit exact 为 `316/384` / `314/384`，v4 production 为 `308/384` 且 UE6 退化；这些成绩只作基线，当前训练/评测合同已是 v5，必须重训后才能产生新成绩。v3 保留静态事故/施工、路边停车、队列车辆和 ego 视差不能充当 UE3 证据；v4 的历史逐帧审计结论仍保存在 `V3_RETRAIN_RGB_AUDIT_20260829.md`。 源 taxonomy 中显式 `U-E3` 的 DynamicObjectCrossing/ParkingCutIn/StaticCutIn 全部保留；即使它与 R4/R5 interrupted overlay 共存也固定通过 ROAD_CORRIDOR 问组监督，不能被 RS gate 静默丢弃。自由生成严格 parser 对完整字符串校验规定顺序、恰好行数和无额外文本，任何格式违规让 format/exact 同时失败；audit 另报告非评分 `answer_only_diagnostics`，只用于拆分事件答案与 evidence 完整性。adapter eval 在载入权重前硬校验 production prompt hash、history RGB mode 和 resolve 后 base model 路径。构建索引会把实际扫描到的 scenario/Town 与 RGB review coverage 做差集；训练与评测要求 UE1/UE3/UE5/UE6/RE/INVALID 六桶齐全，截断缺桶直接失败，`focus_balance_count=0` 只按六桶最小值采样与记账。`invalid_source` 贯穿 train/eval case，采样继续按 source class 及其联合 `source+true_rs+wrong-domain` 签名分层轮转；train balance/TB、generation eval 和独立 eval 都报告 source class、true RS、错误问题域、联合签名的数量、guard 与 exact。`cases_per_bin=0` 保留全量评测行但仍执行 INVALID 签名/覆盖守卫；错例 audit 的 manifest、summary 和逐例 note 直接保存 INVALID 子组。运行见 `SFT_NEW_LOOP_PHASE2_RUN.md`。 |
| new loop Phase1/2 launcher 与 eval 共用合同 | 两个 full-pipeline 都默认 4RGB + 请求四卡，`2rgb_endpoints` 只取原四帧 history 的 `[0,3]`，也可显式按模式分别训练；训练结束默认调用各自 `eval.sh`。指定 checkpoint 后，RGB mode 只能从 adapter config 恢复，调用方不再覆盖；bundle 构建前校验所有 eval metrics 的 mode 与 checkpoint 一致，并把 mode/count/selected-indices 写入包名、README、manifest 和 adapter metadata。两个 eval 包都保留 base/LoRA production+audit 指标、数据/采样 metadata 与按错误桶压缩的真实 RGB，排除权重/checkpoint/TensorBoard，并强制不超过 30MB。 |
| `qwen3vl_local/sft_v3/` | SFT v3 offline on-policy OPSD 路线：学生自维护 memory + `disable_adapter()` privileged teacher full-vocabulary logits 分布监督；v3 不维护独立 prompt，只 re-export v4 prompt / Memory / 状态机 / target span；δ 允许 0 且只封顶 10，`EGO_TO_GOAL_XY` 严格来自 meta `next_target_points[-1]`，帧末预取下一帧 goal，step3 触发统一走 `should_trigger_step3`；多卡训练采用 work-stealing + local-SGD（TCPStore 抢 episode、NCCL collective 前先 TCPStore rendezvous、先广播 rank0 LoRA 初始权重、按本轮 optimizer step 数加权平均 LoRA 参数，sync 后保存 averaged checkpoint；sync 日志/TB 记录 `all_rank_steps`、`round_eps`、`total_eps`），不再 DDP 分片或截断尾部；运行看 `SFT_V3_RUN.md` |
| `qwen3vl_local/sft_v4/` | SFT v4 off-policy actor-learner 路线：`launch_offpolicy.sh` 默认四卡部署为 GPU0 单进程 learner + GPU1/GPU2/GPU3 各 1 个异步 collector；collector 不进 DDP/NCCL，只用 adapter snapshot 采集 sequence-memory rollout 并写 `replay/ready/*.jsonl`；learner 不进 DDP/NCCL，单进程随机读取 replay 做 teacher-forced loss/backward，并周期发布 `latest_lora/v_<step>/`。确认服务器允许单卡多 CUDA 进程后，可手动调 `COLLECTORS_PER_GPU=2/3`；`learn.py` 日志/TB 记录 `replay_ready/replay_pending/replay_failed/wait_events/wait_total` 与 `train/replay/*`，用于判断 collector 和 learner 谁是吞吐瓶颈。当前 v4 使用 ROAD_STRUCTURE→SCENE→STATUS/SUBGOAL 三层 memory：Phase A 初始 `P_INIT_CORRECT=0.7`（road_structure 命中 GT 桶后 scene 同桶 50% 正确）、Phase B 噪声率 0.15、上一帧 step1 后 road_structure 仍未命中 GT 时下一帧帧首触发一次 skip 纠偏（scene 大概率 GT / 0.15 同桶扰动，status/subgoal 回 init）、stair-step 触发门要求上层在本帧前后都稳定正确才继续下钻（road 刚纠正不跑 step2，scene 刚纠正不跑 step3）。step1 学生 prompt 只读 road-only `[STEP1_ROAD_MEMORY]`（believed road + goal），不提前暴露 scene/status/subgoal；公共证据规则默认 keep believed memory，只有清晰可见证据矛盾才改，弱证据写 not contradicted，不编造 braking/merging/cut-in/active-flow 等隐藏线索；Step1 只看 road-layout cues，Step2 是 road bucket 内 fine-grained scene verification，Step3 明确区分当前 `STATUS` 和下一目标 `SUBGOAL`。step2/3 才读完整 `[MEMORY]`。student prompt 与 teacher target 共用四行 analysis contract（Scene Description / Critical Object Description / Reasoning on Intent / Memory Judgment），区别只是 teacher prompt 可看 answer 字段且标签由脚本追加；`build_step*_teacher_target` 必须把 analysis 清成学生视角，禁止把 `GROUND_TRUTH_*` / `ANSWER_*` / `REFERENCE_*` 私有字段名写进监督文本。replay schema 为 `sft_v4_rollout_v2`，显式保存 `memory_after_step1`，learner 重放 step2 必须用该 memory 构造收窄后的 SCENE_CHOICES；旧 v1 trajectory 会被拒收。`inspect_teacher.py` 默认 4 种常规模式，`scene_change_cross_rs` 为显式 stress-only，并在报告中一对一展示 teacher-private prompt/raw、student-facing prompt、adapter-enabled student 初始输出、target/memory transition。`replay.py` / `collect.py` / `learn.py` / `launch_offpolicy.sh` 已实现；`train.py` / `train.sh` 仅为 on-policy 兼容调试入口。运行见 `SFT_V4_RUN.md` |
| `qwen3vl_local/sft_loop_phase3/` | Phase3 事件级 RS-gated 二值问答路线：从 `keyframe_filter/collection_output/*_result.json` 构建逐帧样本，先剔除异常时长 route，再用 Phase2 风格的 synthetic RS context 模拟“上一步 RS 已答对”，训练/eval 中渲染为上一轮 assistant answer 后继续问 EVENT，更贴近真实 KV 续接；`build_phase3_prompt` 默认只表示实际后一轮 user turn，不 inline Phase2，eval case 保存实际多轮 messages / phase2 user / phase2 assistant / phase3 user prompt，避免 audit 误读 inline RS context。RS1/RS2 只问 UE1/UE3/UE5，RS4/RS5 只问 UE6；RE 统一为所有 UE=NO，不再细分 regular event；UE2/UE4/UE7 由 Phase1 路线承担，UE8 默认折入 RE。数据集按 split 保持 UE1:UE3:UE5:UE6 为 1:1:1:1，默认 RE 数量等于单个 UE 桶，并额外加入约 20% wrong-RS invalid/not-applicable 样本；invalid 按 source_class / true_rs / fake_rs 均衡，R3/highway invalid 同时展开到 RS1/RS2/RS4/RS5，标签为所有 UE=NO 且 `INVALID_RS_CONTEXT=YES`。训练/eval 复用 phase2_augment 的 LoRA、TensorBoard、频繁 eval/checkpoint 与 audit case 框架，并在 metrics/TB 中记录 invalid joint/all-UE-NO 指标；prompt v2 强调弱 RGB 证据时保持 RE/all-NO、普通路口车辆不等于 UE6、事故/静态拥堵不等于 UE3、invalid 只表示 RS gate 明显不适用；训练默认 `REGULAR_FOCUS_MULTIPLIER=2.0` 只放大 RE hard negatives，UE 正类仍为 1:1:1:1，eval/generation 仍用均衡口径；DDP 训练按 global step 对齐各 rank，skip/超长样本跑短图文 DDP forward 并用 logits zero loss backward；`GRAD_ACCUM>1` 结尾残余梯度会 flush，`SAVE_STEPS` 落在累积窗口中间时 checkpoint 延迟到下一次 optimizer step 后保存；运行见 `SFT_LOOP_PHASE3_RUN.md`。 |
| `qwen3vl_local/sft_v5/` | SFT v5 RS_SLOW / EVENT_FAST 双频 OPSD 路线：Q1 用当前 RGB 和不可信 RS hypothesis 判断慢变量 RS；Q2 在 RS gate 正确时逐帧重新读取 RGB，从 `[RE | REGULAR]` / `[UE | UNUSUAL]` 混合候选判断 EVENT，不再单问 ABNORMAL。Q1/Q2 memory 使用固定 schema 内的 UNKNOWN/no-prior 与错误/陈旧 hypothesis 做 aligned/omission/contradiction 课程；普通帧 RS/EVENT age 分别累计，但 EVENT 是 `EVENT | RS` 条件状态，RS hypothesis 真正变化时旧 EVENT 立即失效为 UNKNOWN/age=0，只能由新 RS gate 下的 Q2 重建。Prompt 合同为 `sft_v5_compact_prompt_v1`：system 只放共享证据原则，user 只放短 memory/候选/本题说明/四行格式，代表性二选一预算为 system≤70、Q1≤160、Q2≤175 words，版本写入 adapter/eval/probe。稳定 RS 默认以 4 帧为中心，从 3/4/5 帧中可复现随机选择下一次 RS_SLOW；错误/UNKNOWN/recovery 时逐帧慢问，RS 错的当帧跳过 EVENT。慢帧 EVENT 精确续接当帧 Q1 KV，快帧对当前 RGB fresh prefill。训练使用 torchrun 同步 on-policy OPSD、batched rollout、独立 parallel-KL 微批和手动 LoRA 梯度 all-reduce；完整数据过滤、prompt、KV、padding、指标和 probe 合同见 `SFT_V5_RUN.md` / `SFT_V5_PLAN.md` / `SFT_V5_VISUALIZATION_RECORD.md`。 |
| `qwen3vl_local/sft_v5/` batched Qwen 补充约束 | Q1/Q2 student rollout 允许 mixed-length padded batch；padded KV 只用于 no-grad 采样，不写回 memory。默认保持 `QWEN_BATCH_SIZE=8`，但有 autograd graph 的 parallel KL 使用独立 `PARALLEL_KL_MICROBATCH_SIZE=2`，即 8 路 rollout 后按 2+2+2+2 teacher/student scoring 并逐微批 backward。Q2 student rollout 和 Q2 KL 都必须按精确 `q1_ids` 续接 Q1 KV 后追加 Q2 user turn，禁止文本回环重 tokenize，保证采样与 scoring 上下文一致。KL forward OOM 只允许在尚未 backward 时二分当前微批，不降低 token 上限、不重新 rollout；backward OOM 或普通异常必须中止，避免部分梯度后 fallback 重复累计。`test_batched_qwen_smoke.py --check-parallel-kl` 验证真实模型等价性，`test_parallel_kl_microbatch.py` 验证微批/OOM 二分梯度等价性。显存峰值按 `KL microbatch x context length` 审计，TB 记录 `parallel_kl/{microbatches_per_chunk,frames_per_microbatch,oom_splits}`。相关代码必须保留中文注释解释 padded rollout、精确 Q2 KV 续接、KL OOM 安全二分和 TensorBoard 分母口径。 |
| `qwen3vl_local/sft_v5/` TensorBoard 补充约束 | 除 `train/loss_frame` 外，必须记录 `train/loss/{q1_analysis,q1_rs,q2_analysis,q2_event}`，其中 Q1 分项按实际触发 RS_SLOW 的 frame 平均，Q2 分项按实际进入 EVENT_FAST 的 frame 平均；还要记录 `train/rs_slow_trigger_rate` / `train/rs_reuse_fast_rate`、`memory/q{1,2}_relation_{aligned,omission,contradiction}_rate`、`memory/q1_rs_age_frames_mean`、`memory/q2_event_age_frames_mean`、`memory/event_invalidated_by_rs_change_rate`、`memory/rs_periodic_interval_{mean,std}`。同时记录 `memory/{allocated,reserved,max_allocated,max_reserved}_gb`，长期显存风险以活跃引用 `allocated` 为主，不能只凭 `nvidia-smi` 或 allocator `reserved` 高水位判断泄漏。 |
| `qwen3vl_local/sft_v5/` 流式优化补充约束 | 正式训练默认 `UPDATE_MODE=streaming_frames`：每个完整 global timestep 后 SUM all-reduce 实际有效 frame 数，累计 `TARGET_GLOBAL_FRAMES_PER_STEP=512` 或达到 `MAX_TIMESTEPS_PER_STEP=32` 时同步 LoRA 梯度并 optimizer step；不能在同一帧 Q1/Q2/KL 中间更新。每帧 loss 先按 effective target 缩放，梯度 SUM all-reduce 后再按窗口实际 global frame 数修正，保证 frame 等权而非 rank 等权；无本地 frame 的 rank 也必须补零梯度参与 collective。LoRA 梯度按 device/dtype 合并成约 64 MiB bucket，减少小参数逐个 NCCL collective 的同步开销。optimizer step 后保留各 route 离散 memory，epoch 尾窗口必须 flush。`GRAD_ACCUM` 是窗口倍率；`UPDATE_MODE=batch` 只作旧实验兼容。scheduler 总步数按全量训练 frame / effective target 估算，默认 LR 为 `1e-5`。TB 额外记录 `train/global_frames_per_step`、`train/timesteps_per_step`、`train/update_reason_code`、`ddp/grad_allreduce_buckets`、`time/grad_sync_seconds`、`time/optimizer_step_seconds`；adapter 元数据同时保存原始/effective 阈值、LR 和梯度同步策略。 |
| `qwen3vl_local/sft_v5/` checkpoint probe 补充约束 | launcher 默认 `SAVE_STEPS=40`，step 0、checkpoint、final 都用固定 seed 测同一条完整 validation route ID，从首帧运行到末帧；`--num-routes` 控制 random ID 数，`--num-cases` 只用于 RS/UE 专项，专项默认边界前后 8 帧且 UE span 不截断。完整 ID 首帧初始化 student/reference，之后 student memory 由 RS_SLOW/EVENT_FAST 输出推进，reference 只比较不纠错，逐帧只刷新导航坐标。`results.json.memory_recovery_report` 统计变化后 student 首次自行对齐的延迟。默认 review 每帧只写 `input_rgb_*.jpg`、`input.json`、`output.json`、`memory.json`；output 含 student/teacher raw+parsed、teacher target、场景 GT、RS_SLOW 触发原因和 EVENT KV 来源，快帧必须标记 `fresh_rgb_prefill`。`compact` 只写顶层 results，`full` 额外保留 legacy 文件。每次 probe 输出目录必须为空，非空直接拒绝且不自动删除；运行期间保留 `.probe_in_progress.json`，只有 artifact 校验通过并原子提交 `format_version=5` 的 `results.json` 后才移除，`run_integrity` 记录本次 route/frame/artifact 完整性；超长/非法 scenario-route 目录名追加短哈希防碰撞。自动 probe 复用 rank0 bundle，base/teacher 临时关闭 adapter，checkpoint/final student 使用 LoRA；其它 rank barrier，结束恢复 train 并清 cache。 |
| `qwen3vl_local/sft_v5/` eval/probe 指标补充约束 | `eval.py` 与 `probe.py` 共用 `metrics.py`，统一统计 RS/UE 边界、Q1/Q2 precision/recall/F1、FP/FN、端到端 EVENT 和 route macro；相邻帧统计 RS change、RE->UE、UE->RE 的 TP/FP/TN/FN/invalid。student closed-loop 测试中实际 GT reset 应为 0，训练规则建议 reset 的频率单独记录。小样本变化指标在 summary，自主恢复延迟在 `memory_recovery_report`；大样本 eval 默认流式累计，只有显式 `--output-jsonl` 才落盘逐帧证据。 |
| `qwen3vl_local/sft_v5/` repair / eval 去 oracle 补充约束 | 正式训练默认 `RS_REPAIR_MODE=EVENT_REPAIR_MODE=ground_truth`，但只在 RS 连错 4 帧并到 2 帧 review slot、EVENT 连错 3 次并到每帧 review slot 后延迟写回，不是错误下一帧即时纠正。`unknown` 软擦除只作消融；纯 memory-copy 压力测试中它会让 RS anomaly 约 95.7%、Q2 gate 约 4.3%，不作长训默认。forced-repair 后答对与干预前自主恢复分开统计。eval/probe student 默认 RS/EVENT=UNKNOWN 启动，deployable RS scheduler 不使用 GT mismatch，只由 UNKNOWN/非法输出、RS 变化确认和周期复核驱动；旧 GT/oracle 口径只显式复现。离线 EVENT gate 为保持“RS 真错就跳过 EVENT”仍使用 GT correctness，summary 必须写 `event_gate_uses_ground_truth=true` / `fully_deployable_end_to_end=false`。 |
| `qwen3vl_local/sft_v5/` 显存生命周期与 teacher probe 补充约束 | 纯 batched rollout 只返回文本/token ids，不物化逐样本 final KV；Q2 state 构造后立即释放旧 Q1/Q2 KV。loss backward 后释放计算图，optimizer step 使用 `zero_grad(set_to_none=True)`；正常/异常退出统一销毁 process group 并做一次 GC/CUDA cache 清理，训练 step 内不频繁 `empty_cache()`。慢帧 teacher EVENT 在 teacher 自身 Q1 RS 正确时触发并续接 teacher 自己的 Q1 KV；快帧 teacher EVENT 对当前 RGB fresh prefill。训练 privileged prompt 与 teacher 自主 prompt 分文件保存。 |
| `qwen3vl_local/sft_v5/` 注释与文档分工补充约束 | 每个 Python 模块用中文 docstring 说明用法和入口，所有 class/function（含 CLI、嵌套 helper 和魔术方法）保留中文 docstring；padding、KV、loss 分母、DDP collective 和显存生命周期等非显然逻辑需注释设计原因。2026-07 本轮详细注释覆盖数据过滤与坐标转换、标签/动态候选、Q1/Q2 memory curriculum、local/global padding、batched KV/M-RoPE、精确 `q1_ids` 续接、OPSD span/KL、forward-OOM 安全二分、global-frame 梯度归一化、分桶 all-reduce、closed-loop eval、probe 选帧与 artifact 落盘；本轮同时修正了 repair 统计与 eval/probe oracle 调度泄漏。推荐阅读顺序为 `labels.py -> prompts.py -> build_dataset.py -> train.py` 的 dataset/sampler 与单帧语义基准 -> grouped rollout/KL/optimizer -> `metrics.py -> eval.py -> probe.py -> test_*.py`，完整函数导航见 `SFT_V5_PLAN.md` §9.3。compact `results.json` 只减少文件数量，不能减少人工审计字段：每帧保存 RGB 路径、实际 student/teacher messages、完整 student/base-teacher CoT 输出、脚本化 teacher target、RS/EVENT 场景 GT、memory 和变化检测结果；base 与 LoRA probe 使用同一 schema。UE 专项必须保留一个连续 UE span 的全部 UE 帧，并按 context radius 补进入前/退出后邻帧，不得被 `num_cases` 从中间截断；RS 专项只保留变化点前后数帧。`SFT_V5_RUN.md` 只作精简命令手册，设计合同归 `SFT_V5_PLAN.md`，完整 probe 产物和人工检查项归 `SFT_V5_VISUALIZATION_RECORD.md`。 |
| `qwen3vl_local/sft_v5/` memory curriculum 补充约束 | 旧 `BELIEVED_*` 名称已废弃：Q1 使用 `PREVIOUS_RS_HYPOTHESIS + PREVIOUS_RS_HYPOTHESIS_AGE + MEMORY_RELIABILITY + EGO_TO_GOAL_XY`，Q2 才额外加入 `PREVIOUS_EVENT_HYPOTHESIS + PREVIOUS_EVENT_HYPOTHESIS_AGE`；memory 是可能过期或错误的 hypothesis。“没有 memory”使用固定 schema 内的 UNKNOWN/no-prior，不删除 block。普通帧两个 age 分别累加，对应 label 改变时归零；重复确认不归零，padding/skip 不累加。EVENT 是 `EVENT | RS`：RS hypothesis 变化会把 EVENT 失效为 UNKNOWN/age=0，并清空旧 RS 语境的 EVENT streak/pending；同帧 Q2 错误从新语境 streak=1 重新累计。新注入的 wrong/UNKNOWN 因为刚改变 hypothesis，age 必须从 0 开始；只有学生继续复制，才由后续真实帧自然形成 age>0 的 stale 样本，不能随机伪造旧 age。route 首帧 RS/EVENT 各以 0.5 概率使用 GT，否则 UNKNOWN；正确 RS memory 默认按 0.05/0.07 注入 contradiction/omission，EVENT 额外注入为 0.20/0.12。稳定 RS 默认 `rs_slow_interval=4, rs_slow_interval_jitter=1`，即按 3/4/5 帧可复现随机复核；快帧不产生 RS rollout/loss，但 gate 正确时必须产生 EVENT_FAST。RS 错误只跳过本帧 EVENT，下一帧逐帧 RS 分析，直到学生纠正或 delayed repair。RS 连错 4 帧并到 2 帧 review slot、EVENT 连错 3 次并到每帧 review slot 后才延迟修复；EVENT 不存在独立 ABNORMAL 状态。合法 Q1/Q2 最终高权重 span 只监督单个选项字符；若存在 `RS:`/`EVENT:` 行但值是 `R4`/`RE` 等非法语义标签，则严格 parser 仍拒绝且不更新 memory，但 loss 会监督答案起始 token 以直接纠正选项格式。train/eval/probe 必须记录 memory 关系、age、RS 变化导致 EVENT 失效、复制、恢复、门控及 UE/RE P/R/F1。 |
| `qwen3vl_local/sft_v5/` memory 数据量审计 | 42 个有效 `collection_output` 场景（排除 `noScenarios`）共有 7241 条 success route、914466 帧；原始 GT EVENT 中 RE=772286（84.45%）、UE=142180（15.55%）。默认 10% route-level validation 后约 82.3 万训练帧，最终值以远端 `checkpoints/sft_v5_data/summary.json` 为准。GT UE 与人为 wrong/UNKNOWN memory 是不同异常，不能相加。恒定 GT、当帧自纠模拟中 Q1 trigger≈30.5%，Q1 aligned/omission/contradiction≈59.7/24.2/16.1，Q2≈59.6/23.0/17.4；纯 memory-copy 到 delayed repair 的压力测试中 Q1 trigger≈55.5%、Q2 gate≈64.0%、Q2 relation≈38.6/43.5/17.9。最终比例以 TensorBoard 为准。 |
| `qwen3vl_local/sft_base/` | SFT v5 的直接监督基线。复用 v5 的 collection_output 数据构建、异常 route 剔除、4 帧 RGB history、`EGO_TO_GOAL_XY`、RS/EVENT 候选池和串行 memory 状态；Q2 候选顺序使用 v5 seed namespace，但本路线不使用 A/B/C 字母，学生直接输出语义 token。当前 `DATASET_VERSION=sft_base_rs_event_token_choice_rs_regular_mapped`：regular 不再折叠成 `RE`，且以 RS 为准做 canonical 映射，R4 任意 regular -> `SIGNAL_COMPLIANCE`，R5 任意 regular -> `PRIORITY_NEGOTIATION`，R1/R2/R3 中不属于该 RS 静态表的路口 regular -> `LANE_FOLLOWING`；原始 code 保留在 `event_code_raw` / `regular_event_codes` / `event_labels_raw` 审计字段。audit remap 只统计最终 GT 为 regular 的 pure regular 帧，UE 帧即使同时带 R-E 标注也只进入 `frames_with_regular_annotation_by_rs`，不计入 `raw_regular_remap_total`；`pure_regular_frames_by_rs` 应等于各 RS 总帧减 UE 帧。旧 A/B/C adapter、旧 `RE -> REGULAR` adapter 和旧未映射 index/adapter 必须重建/重训。训练不做 OPSD、不采 student rollout、不跑 privileged teacher、不输出 CoT；Q1 target 只有 `RS: <token>`，Q2 target 只有 `EVENT: <token>`，UE/regular 指标从 Q2 EVENT 折叠。训练为 teacher-forced weighted CE，memory 由 GT answer 推进，但 prompt memory 默认强扰动且只展示 RS/EVENT token，不重复长解释；RS 描述只写静态几何，EVENT 描述写本帧动态/规则行为，R3 regular 按稳定车道内行驶、普通主路横向跨线、正在执行匝道/连接段/分流/驶出动作拆开，并由 `check_loss_mask.py` 与 `test_prompt_snapshots.py` 守住 prompt 分工。Q1 prompt 不显示 `BELIEVED_EVENT`，Q1 后 RS hypothesis 改变会让旧 EVENT 失效，训练侧随后为 Q2 按当前 RS 池重采 EVENT memory。UE loss 按 RS 条件 UE 率 inverse-sqrt 缩放；regular loss 与 regular frame repeat 按 R-E 子类频次 inverse-sqrt 缩放，UE 子类仍按逆频率 repeat 放大长尾。eval 可省略 `--adapter-dir` 直接跑 base 零样本基线；LoRA eval 会校验 adapter config。eval Q2 候选按学生 RS 的静态候选全集生成；逐帧 `allowed_events` 只用于 GT 解析和审计，不参与 prompt 候选构造；集合与 dataset 候选相同时复用 dataset 顺序。GT EVENT 不在 dataset 自己候选表中的帧与训练 Q2 skip 对齐并单独报告；GT EVENT 在 dataset 候选中但不在学生 RS 候选中仍算模型错误。eval 默认首帧 UNKNOWN 冷启动、Q1 错也继续问 Q2，并输出 `joint_acc=P(RS 对且 EVENT 对)`、端到端全局多数 regular 下界 `event_global_majority_baseline`（永远答 `LANE_FOLLOWING`）、GT-RS oracle regular 参照 `event_regular_baseline_given_gt_rs`（全量映射后参考 76.85%，不能作为端到端门槛，当前子集 oracle majority 另列审计字段）、RS/EVENT 13 类混淆矩阵、regular 内部混淆、R3 shortcut 监控、UE-vs-regular P/R/F1、单/多候选 × regular/UE 四格 EVENT 指标、`q2_candidate_count_report`、`q2_rs_candidate_count_report`、直接按 RS 分组的 `ue_fp_on_multi_candidate_re_by_rs` / `q2_multi_re_by_rs_report` / `q2_multi_ue_by_rs_report`、raw regular 映射帧诊断 `event_raw_regular_remap_report`、相邻帧 RS change / regular->UE / UE->regular 指标和 EVENT 不可达率；`report.html` 单文件内嵌 metrics 数据并可视化 RS/EVENT/regular/UE-vs-RE 矩阵，脱离 JSON 也能直接打开；`frames.jsonl` 同步写 `gt_event_code_raw` 与 `gt_regular_remapped` 供错帧回查。默认 `LORA_VISION_SCOPE=last4`、rank=32、alpha=64、vision_lr_scale=0.2，并启用视觉 fuse guard；checkpoint 目录除 adapter 外会写 `trainer_state.pt`，`--resume-from-checkpoint` / `RESUME_FROM_CHECKPOINT` 可在原 run 目录内接着 step、TB、optimizer/scheduler/RNG 续训，resume 默认归档并修剪 `tb/` 中大于 checkpoint step 的旧 event，避免 200-300 等未来曲线污染续训；旧 checkpoint 无 state 时至少恢复 adapter 与目录名 step；仍可用 `off/merger/all` 做对照。运行见 `qwen3vl_local/sft_base/SFT_BASE_RUN.md`。 |
| `qwen3vl_local/sft_base/` RS×EVENT 静态表补充约束 | `EVENT_CANDIDATES_BY_RS` 按严格方案 A 维护 UE 组合：复用 build_dataset 的异常/缺失/失败 route 过滤后，只保留 `count >= 20` 且 `rs_frame_rate >= 0.1%` 的 RS×UE 组合；regular 使用 RS canonical 映射后的语义保守表，当前候选数固定为 R1=7、R2=5、R3=3、R4=5、R5=5，R3 不开放 UE 但保留 3 个 regular 候选。`audit_rs_event_cooccurrence.py` 必须报告 RS×UE、mapped RS×R-E regular 分布、多 raw regular 标签比例、UE/regular 两侧 missing、low-rate 和 spurious 静态组合；raw regular remap 只统计最终 GT 为 regular 的 pure regular 帧，并输出 scenario/route top-k 归因；新协议下 regular missing/spurious 应为空，GT static mismatch 应只剩严格阈值拒绝的低频 UE 组合。 |
| `qwen3vl_local/sft_base_simple/` | 按用户同意新增：从 `sft_baseline` 继续简化的 HIGHWAY/NON_HIGHWAY + RE/UE 单问直接监督基线。显式 transition 采样/API 已撤掉，训练默认先跨 route 聚合 `FOURBIN_ROUTES_PER_BATCH=16` 条 route，再按当前帧 GT 四格 `HIGHWAY:UE` / `HIGHWAY:RE` / `NON_HIGHWAY:UE` / `NON_HIGHWAY:RE` 做 exact balance，默认 `JOINT_TARGET_BALANCE_COUNT=8`、`UE_FRAME_REPEAT=1`、`UE_EVENT_LOSS_WEIGHT=1.0`、repeat mode 为 `none`，避免四格均衡后再向 UE 重复倾斜；eval 默认同样按当前帧 GT 四格随机均衡，但 joint case 会按 route 顺序闭环 rollout 到最远受评帧，只在抽中帧计 ROAD/EVENT/JOIN accuracy，change matrix 来自 rollout 相邻帧，`--initial-memory-noise none` 与 joint eval 组合会被拒绝防止 GT memory 泄漏。transition 帧只作为普通当前帧落入对应四格，不再单独抽样或 repeat。训练日志/TB 记录 balance 后四桶实际样本数与 early-UE prompt memory 的 `RE/UE/UNKNOWN/HIDDEN` 分布。基础 RS/EVENT memory wrong/UNKNOWN/dropout 概率沿用 baseline，连续 UE span 前 `MEMORY_EARLY_UE_FRAMES=4` 帧额外提高 EVENT memory wrong/UNKNOWN/dropout 与重采概率，放大后 wrong+UNKNOWN 显式归一化并在启动日志打印 effective 概率，避免模型靠 `PREVIOUS_EVENT=UE` 续答 UE。当前 `DATASET_VERSION=sft_base_simple_highway_reue_fourbin_v1`，adapter route 为 `sft_base_simple_highway_reue_fourbin_random`，运行见 `qwen3vl_local/sft_base_simple/SFT_BASE_RUN.md`。 |
| `AutoMoT/vae_standalone/train_patch_unpatch.py` | patch/unpatch 端到端重建训练 |
| `0026.json` | LEAD meta 固定参考样本，只读，绝对不要入库 |
| `keyframes_all_scenarios.json` | 仓库根目录或 `AutoMoT/lead_data` 下的远端数据参考，只读；`AutoMoT/keyframe_filter/` 下的旧副本已清理，不再恢复 |

2026-09-03 的旧 UE3 rescore、label-alignment 与 route-balance 排障链已结束，相关专用脚本和审计文档已删除；当前只保留通用错例审计与 2026-09-04 的 highway RGB 真值合同。

2026-09-04 Phase2 高速 UE3 补充：继续逐帧查看 HighwayCutIn 的完整 RGB sheets 后确认，
高速/匝道他车相对可见分道线持续横移、车头/车身跨入 ego 当前通道时仍属于原 `UE3`；
普通并行、稳定跟车和 ego 超车视差仍是 all-NO。由于源 HighwayCutIn 标注没有 U-E3，
仅改 prompt 会造成监督冲突；因此数据构建只接受
`sft_new_loop_phase2/highway_ue3_rgb_decisions_v1.jsonl` 的显式 RGB-YES span 覆盖，
scenario/R3 不自动造正例。模型输出类别不变，`HIGHWAY_CUTIN` 只作为 UE3 审计子型；
manifest、generation eval、独立 eval 与 RGB audit 同时报告总体 UE3 和
HIGHWAY_CUTIN/OTHER_UE3，默认每个 UE3 评测桶至少保留高速子型（目标比例 12.5%），RE
仍保留无 cut-in 高速负例。prompt 升级为待验收的
`sft_new_loop_phase2_direct_event_visual_v5_highway_ue3`，旧 v3/v4 adapter 与其 hash
不兼容；v3 的历史 316/384 仍只是基线，不能冒充 v5 结果。详细证据见
`sft_new_loop_phase2/HIGHWAY_UE3_RGB_AUDIT_20260904.md`。

同日全量扫描 `collection_output/*_result.json`：源 taxonomy 只有
DynamicObjectCrossing（251 帧/84 routes）、ParkingCutIn（330/80）和
StaticCutIn（770/54）显式含 U-E3，共 1351 帧；其中 33 帧是 R4 interrupted overlay。
Phase2 现保留所有显式 U-E3，并固定通过 ROAD_CORRIDOR 问组监督；没有 U-E3 的 R4/R5
仍不生成 UE3。加上 HighwayCutIn 的 73 个 RGB-YES 帧，当前兼容四类 UE3 来源。
UE3 recall 默认门槛改为 0，只统计不阻断；`run_full_pipeline.sh` 优先用通过 generation guard 的
`best_generation/`，缺失时回退训练结束的 `final/` 继续 eval 与压缩。旧 UE3 专用 rescore/label-alignment/RGB 包/
route-balance 排障脚本及对应文档已删除，避免与当前 v5 合同混用。

SFT v4 scene canonicalization rule: `EnterActorFlowV2 -> EnterActorFlow` and
`MergerIntoSlowTrafficV2 -> MergerIntoSlowTraffic`. These raw CARLA scenario
variants keep their original `scenario/raw_gt_scene` metadata, but student
`SCENE` choices, memory, teacher targets, and eval comparisons use the canonical
scene label because the paired variants share the same visible semantics and
event sequence.
SFT v4 prompt contract: teacher prompts generate only the four analysis lines
(`Scene Description`, `Critical Object Description`, `Reasoning on Intent`,
`Memory Judgment`) and never include label placeholders; `build_step*_teacher_target`
appends supervised labels and enforces the scripted KEEP/CHANGE/ADVANCE opener,
while student prompts ask the adapter to write labels on separate lines.

SFT v3/v4 prompt-sync rule: `qwen3vl_local/sft_v4/prompts.py` is the single
canonical prompt, Memory state machine, trigger helper, target-span source, and
public analysis-heading source (`Scene Description` / `Critical Object
Description` / `Reasoning on Intent` / `Memory Judgment`).
`qwen3vl_local/sft_v3/prompts.py` re-exports it and only keeps v3 compatibility
aliases. v3 is the offline on-policy OPSD route: student rollout tokens update
memory, and `eval() + no_grad + disable_adapter()` privileged teacher logits
supervise the same generated token ids that entered student KV with forward-KL
rather than hard teacher-text CE. Student text is kept unstripped only for parsing
labels/spans; step1 empty/EOS output skips the frame instead of injecting a GT
teacher target. v4 keeps the off-policy actor-learner/replay route. Any
prompt or state-machine edit must be validated on both v3 and v4.
`inspect_teacher.py` runs prompt-contract self-checks before lazy-loading torch
and model helpers; keep this order so prompt-only regressions are caught before
runtime dependency failures.

2026-08-09 `AutoMoT/keyframe_filter` 追加完成 Phase1 四问答案表的全量人工 RGB + 原始标签复核，
摘要见 `AutoMoT/keyframe_filter/PHASE1_FOUR_QUESTION_RGB_AUDIT_20260809.md`，Phase1
`collection_output` 目录索引/legacy 关系/复用流程见
`AutoMoT/keyframe_filter/PHASE1_COLLECTION_OUTPUT_INDEX.md`；本地证据位于
`AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/`，后续复核应优先复用
`full_route_rgb_label_review_20260809/` 与已有 notes，不要重新批量生成重复 RGB 目录；其中最终四问标签表、
batch matrix 和人工 JSONL notes 是轻量标签产物，已列入白名单可精确 push；RGB contact
sheet、montage、candidate anomalies 与 route/town/scenario/global summary 等证据产物仍默认不入库、不 push。
统一证据目录 `full_route_rgb_label_review_20260809/` 覆盖 42 个非 `noScenarios` 场景、
197 个 scenario-Town、582 条 route、68,073 帧、2,003 张逐帧 RGB+RS/EVENT 标签 sheet；
5 个源数据不足 3 条的 Town 已审完全部可用 route。
关键口径：`noScenarios` 排除；`HIGHWAY` 必须看匝道/出入口/导流 gore/连续隔离/受控通行等拓扑，
不能由直道、宽路、空旷或单独护栏推出；`EnterActorFlow/R1/R-E1` 与
`EnterActorFlowV2/R1/R-E1` 四问 `HIGHWAY=YES`，`InterurbanActorFlow/R3/R-E1`
四问 `HIGHWAY=NO`；`Accident/R1/{R-E1,R-E2}` 已纠正为 `HIGHWAY=NO`（旧表错误套用了
actor-flow 理由）；`ParkedObstacle × U-E2` 组合级保持 `OBSTACLE=YES`。答案表现在是 v2：
默认 `scenario × RS × EVENT` 行由 RGB 审计策略生成，`ParkedObstacle/Town12` 的受控快速路
子组只有在 route-level RGB topology 标为 `limited_access_fast_road` 时才覆盖成 `HIGHWAY=YES`，
不能仅凭 Town、场景名、宽直道路或护栏触发。

## 1.1 运行命令目录约定

运行手册默认当前目录就是远端 `AutoMoT/`。命令示例统一写相对 `AutoMoT/`
的路径，例如 `bash qwen3vl_local/...`、`python qwen3vl_local/...`、
`leaderboard/...`、`checkpoints/...`；不要额外写切目录步骤，也不要给
`qwen3vl_local/...` 命令加 `AutoMoT/` 前缀。只有仓库根视角的文件白名单、git add 路径、
或明确说明 repo root 路径时，才保留 `AutoMoT/` 前缀。

LEAD 数据根目录统一假设在 `AutoMoT/lead_data`，即远端原始 LEAD 数据软链接后的目录。
运行文档、脚本默认值和示例命令不要再写原始 datashare 绝对路径；数据根写
`--data-root lead_data`，keyframes 写 `--keyframes lead_data/keyframes_all_scenarios.json`。
训练、eval、probe 产物仍写 `checkpoints/...`。

## 2. 时间与输入约定

- CARLA 20Hz；LEAD 每 5 tick 存 1 帧，所以离线帧率 4Hz，每帧约 0.25s。
- LEAD RGB 是三视角拼接图，`PIL.size=(1152,384)`，当前本地 Qwen prefill 直接喂整图。
- LEAD `.laz` 单帧已含 5 sweep；不要额外按 AutoMoT 在线双帧融合逻辑乱拼。
- `future_positions[[5,10,...,40]]` 对应约 2s future waypoints。
- route / future_waypoints 都是 ego-frame 累计点，不是相邻 delta。

## 3. 当前路线决策

当前离线 runner 只走本地 `LocalQwen3VLInstructEngine` 做 frozen Qwen prefill，
再接 LeadMoT 或 GoalGen。已经移除 / 禁用这些旧路径：

- AutoMoT legacy `kv_cache_fixed_inference(...)`
- `InterleaveInferencer` 直接复用
- 原 fast head / `enable_fast_inference`
- `--enable-automot-slow`

原因：AutoMoT 自定义 MoT 架构和 standalone Qwen3-VL-Instruct 的 HF
`past_key_values` 不同源，不能混用。

## 4. Qwen / Prompt 规则

- `qwen` backend 只读本地 checkpoint，必须 `local_files_only=True`。
- Qwen3-VL-Instruct standalone runner 只跑
  `AutoMoT/checkpoints/Qwen3-VL-4B-Instruct`，不 import
  `vlm_paradigm_a_runner.py`。
- `prompt_pipeline.py` 是范式 A prompt / 状态机来源；改 prompt 后要同步影响 SFT v2 pending 数据。
- AutoMoT `InterleaveInferencer` / `qwen3vl_template_inference` 绑定自定义 MoT 架构，不能支撑 standalone Qwen 自由文本生成。
- Qwen3-VL 自定义 KV 增量 decode 必须走 `qwen3vl_local/mrope_utils.py` 的
  `qwen3vl_incremental_forward`，不能再依赖 `prepare_inputs_for_generation` 拼 decode
  输入。已确认 PEFT wrapper 会裁掉 `cache_position`，使每个续写 token 的 M-RoPE 位置
  退化为 0，导致 logits 从第 1 个续写 token 起大幅漂移、老师/学生生成复读，并污染
  teacher-forced loss。`engine.py`、`sft_v2/eval.py`、`sft_v3/train.py`、`sft_v3/eval.py`、
  `sft_v4/train.py`、`sft_v4/eval.py` 和 `vlm_paradigm_a_runner.py` 的文本续写路径均应
  复用该本地 helper；decode 阶段不重传图像，位置来自本条 KV state 的 `rope_deltas`。
  `engine.py` 的 `cache_system_prompt` 仅允许纯文本 suffix 复用 system-prefix cache；
  若 full input 含 `pixel_values` / `image_grid_thw` 等多模态字段，必须回退完整 prefill，
  避免把 Qwen3-VL 的图文 M-RoPE 计算拆成半截 cache 后错位。
  `_clone_cache` 必须优先保持 Transformers `Cache` 对象类型（如带 `get_mask_sizes` /
  `get_seq_length` 的新版 cache），legacy tuple 只能作为旧版兜底；新版 Qwen3-VL
  forward 会直接调用 Cache 方法，不能把它退化成普通 tuple。受旧 bug 训练出的
  SFT v4 checkpoint 和旧 `teacher_report.md` 抽检结果需要作废，修复后先重跑
  `inspect_teacher.py`，再重新训练。
- PEFT 仍可用于**加载 LoRA adapter 并立即 `merge_and_unload`**；禁止的是让 PEFT
  wrapper 参与增量 decode / `generate` 的 `prepare_inputs_for_generation` 路径。
  v3/v4 eval/probe 默认 `--merge-lora=True`，加载 adapter 后后续文本生成走 merged
  base + `qwen3vl_incremental_forward`。`engine.py` 的 adapter 自检允许
  `sft_v*_adapter_config.json` 保存完整 target module 路径，而 PEFT
  `adapter_config.json` 保存 `q_proj/down_proj/...` 这类短名；二者按 PEFT 后缀匹配语义
  判定兼容，同时继续校验视觉 LoRA scope 与权重 key，避免静默漏挂视觉 adapter。
- `LocalQwen3VLInstructEngine` 的 prefill/decode 必须在 `torch.inference_mode()` 下运行；
  v3/v4 eval/probe 的 `_generate_next_with_kv` 也必须用 inference mode 包住 suffix forward
  和 decode。否则 `--with-teacher` 的双模型诊断会在纯推理阶段构建 autograd graph，
  4 张 LEAD RGB 多次 full prefill 很容易把 80-90GB 显存吃满。每次新的 full
  `engine.generate()` 前必须先清旧 `_last_decode_state`；teacher step1/2/3 是独立专家问答，
  生成后也必须立刻清 teacher `_last_decode_state`，避免上一轮 KV 和下一轮 prefill 同时常驻。
  `--with-teacher` 本身仍会同卡常驻 student merged LoRA + base teacher 两份 Qwen，
  只适合小样本诊断。

## 5. VLM 两种范式

| 范式 | 目的 | 当前状态 |
|---|---|---|
| A：VLM 直接输出 `ANALYSIS/STATUS/SUBGOAL` | 文字状态跟踪 | standalone Qwen runner 可用；AutoMoT ckpt 不能可靠自由文本生成 |
| B：Qwen 当视觉语言编码器，decoder attend KV | 轨迹 / 子目标生成 | 当前 LeadMoT / GoalGen 主路线 |

记忆法：要文字走 standalone Qwen；要规划走 frozen prefill + decoder。

## 6. LeadMoT

文件：

- `qwen3vl_local/leadmot/train.py`
- `qwen3vl_local/leadmot/eval.py`
- `qwen3vl_local/leadmot/probe.py`
- `qwen3vl_local/leadmot/decoder.py`
- `qwen3vl_local/leadmot/LEADMOT_RUN.md`

核心结构：

- 输出 `pred_route (B,10,2)` 和 `pred_future_waypoints (B,8,2)`。
- head 是 `Linear -> cumsum`，loss 直接对累计 ego-frame 点。
- 训练冻结 Qwen3-VL-Instruct 与 LeadBEVEncoder，只训练 LeadMoT decoder。
- gen 路 12 层，hidden=1024，8 heads，head_dim=128，对齐 Qwen K/V 子空间。
- language K/V 来自 Qwen prefill，已经带 M-RoPE；LeadMoT 不重复旋转语言 K/V。

BEV 开关：

- 默认 `USE_BEV=1` / `use_bev=True`。
- `USE_BEV=0` 时 decoder 完全不实例化 / 不 forward BEV projector。
- checkpoint 加载必须二选一：`use_bev=True` 就导入已有 BEV projector 参数；`use_bev=False` 就彻底不用 BEV。禁止随机初始化 BEV projector 混入推理。
- LeadMoT `ema_state_dict` 当前 schema 是 `{"decay": ..., "shadow": {...}}`；`eval.py` /
  `probe.py` 会 unwrap `shadow` 后再 strict load。

Subgoal 开关：

- 默认 `USE_SUBGOAL=0` / `use_subgoal=False`。`USE_SUBGOAL=1` 会让 frozen Qwen prefix
  额外接收 1 张 SUBGOAL stitched RGB + `[GROUND_TRUTH_STATE]` 文本块
  （scenario/status/subgoal/event meanings），原 navigation prompt 仍保留。
- `use_subgoal` 与 `use_bev` 正交；不改变 decoder state_dict 形状，但 prefix KV 分布
  不兼容，`train.py` 的 resume/init-from-ckpt 会按 checkpoint
  `decoder_config.use_subgoal` 拒绝跨开关加载。
- `leadmot/build_dataset.py --with-subgoal-fields` 默认读取
  `keyframes_all_scenarios.json`，写入
  `run_id/subgoal_lookup_ok/status/subgoal/subgoal_frame/subgoal_rgb_path/subgoal_skip_reason`。
  subgoal 反查只接受 keyframes run status 为 `Completed/Perfect` 的轨迹，失败 run 会写
  `run_status_not_accepted:*` skip reason；`use_subgoal=True` 训练启动时只保留
  `subgoal_lookup_ok=True` 的行，过滤后无样本直接报错。
- eval / probe / `mot_lead_offline_runner.py` 从 ckpt 自动读取 `use_subgoal`。offline runner
  对 subgoal ckpt 要求 `lead_clip` 带
  `subgoal_rgb_path/subgoal_scenario/subgoal_status/subgoal_event`；CLI demo 可用
  `--keyframes` 自动反查并注入。
- `eval_carla` 在线闭环暂不支持 `use_subgoal=True`，因为在线拿不到未来 SUBGOAL keyframe RGB；
  agent 加载该类 ckpt 时立即 `raise NotImplementedError`，保留后续图像生成/代理输入 TODO。

DataLoader worker：

- LeadMoT `train.py` / `eval.py` 默认 `--num-workers 8 --prefetch-factor 2`
  （train.sh: `NUM_WORKERS=8`），把 CPU 侧 JPG/lzma/LAZ 解码预取到 worker；
  worker 默认 `multiprocessing_context=spawn`，避免 Qwen/CUDA 初始化后 fork。
- worker 只提升 GPU util，不改变 B=1 训练语义，也不提升 GPU 显存占用；真正吃满显存
  需要后续 batch>1 + prefix padding mask 改造。
- DDP train 先截掉尾部不足 `world_size` 的样本，再每 rank 取无重复等长 shard；
  val/eval 手动 `rank::world_size` 分片，避免 `DistributedSampler` padding 重复计数。
- `eval.py` 同样默认 `--num-workers 8`；`probe.py` 是小规模 case-level dump（默认 24 case），
  不接 DataLoader worker，避免为了少量样本引入额外启动开销。

### 6.1 LeadMoT CARLA 闭环评测（eval_carla）

入口从 `AutoMoT/` 当前目录看是 `qwen3vl_local/eval_carla/`，操作文档：
`EVAL_CARLA_PLAN.md` / `EVAL_CARLA_RUN.md`。

- `agent.py` 是 leaderboard 实时 agent，RGB 固定 LEAD 3cam：
  3 路 384×384、FOV=60，横拼 1152×384。
- LiDAR / radar 由 checkpoint 的 `decoder_config.use_bev` 决定：
  `use_bev=True` 才声明/读取 LEAD 双 LiDAR + 4 radar；`use_bev=False` 不产生未使用输入，
  只传空点云占位给统一 `lead_clip` 结构。
- **LEAD 训练分布对齐 (v2)**：
  - RGB：拼接后做 JPEG round-trip（`JPEG_QUALITY=85`），模拟 LEAD `.jpg` 训练数据。
  - LiDAR：轻量去地面（z 阈值 + LSQ 平面拟合，`LIDAR_REMOVE_GROUND=1`）；LEAD 用 RANSAC，
    我们因不引 numba 重依赖改用 LSQ 近似。
  - Radar：4 路 → ego frame 后拼到 LiDAR，近车 <8m 的 radar 点 duplicate 5 次
    （`USE_RADAR=1`，与 LEAD `save_radar_pc_as_lidar`+`duplicate_radar_near_ego` 同源）。
- warmup 改 LEAD 风格：第一个 4Hz 采样点（约 0.25s）就开始推理；
  历史不足时 left-pad 复制 frame 0（与 build_clip line 1808-1815 同款）。
- **target_point / next_target_point — P1 全 lookahead 弧长（用户最终路线，
  完全舍弃 P2 automot_route_index）**：
  - 训练侧 `mot_lead_offline_runner._extract_tp_route_lookahead(meta, lookahead_s, min_m=5)`
    沿 meta["route"] 弧长前推 `max(speed*lookahead_s, 5m)` 米取 ego-frame 点；
  - 在线 `agent._lookahead_world_point(speed, lookahead_s, gps, compass)` 用同款公式
    沿 RoutePlanner 剩余 route 前推；
  - **默认 tp=1.0s, ntp=2.0s**（与 wp 视野 8*0.25=2s 对齐，ntp 落 wp 末端），
    MIN_LOOKAHEAD=5m 让停车/红灯仍有方向；
  - 终点近时（route 弧长 < target_dist）tp/ntp 自动 fallback 到当前剩余 route 末端；
  - build_clip 加 `tp_mode={route_lookahead, future_truth}`（默认 route_lookahead）；
    future_truth 是 v1 兼容模式保留。
- **final_goal token 新增**（LeadMoT decoder 第 4 个 status token）：
  - 训练用 LEAD 采集器写入的 `meta["next_target_points"][-1]`，即
    `_command_planner.route` 当前剩余 command route 的真实末端，world frame 转 ego frame；
    不能再用 `meta["route"][-1]`，后者只是局部 dense route 监督片段；
  - 在线 eval_carla 用 `scenario_picker.load_route_endpoint(route_id)` 按 leaderboard
    整数 `route_id` 查旧数字 route XML（如 `data/lead/<Scenario>/Town03_route_001783.xml`）
    的最后一个 waypoint，再按当前 ego pose 转 ego frame；这是在线 220 route 的
    route_id 入口，不改变 `lead_data` 全量映射使用 `(Scenario,Town,route_key)` 的命名规则。
    找不到 XML 时才临时 fallback 到 RoutePlanner 剩余 route 末端；
  - `LeadMoTPlanningDecoderConfig.use_final_goal=True` 默认；与 tp/ntp 共享
    `WaypointInputAdaptor` MLP 让坐标语义同空间；
  - `_LEADMOT_QWEN_SYSTEM_PROMPT` + `build_cleaned_prompt_and_modes` prompt 加
    `your final destination is (X, Y)`；当前路线要求 7 元 `[speed,tp,ntp,final_goal]`，
    不再对 5 元旧输入做自动兼容；
  - **老 LeadMoT v1 ckpt 不兼容**（gen sequence 多 1 token）。train/eval/probe
    均要求 checkpoint 显式记录 `decoder_config.use_final_goal=True`。
- BEV 模型的实时 LiDAR 使用最近 `STEP_STRIDE=5` 个 20Hz sweep（≈0.25s 窗），
  按 (dx, dy, dyaw) 对齐到当前 anchor ego frame 后 concat。
- PID desired speed 用 `wp[1]` / `wp[3]`（0.5s 与 1.0s 两段距离平均），
  与 LEADMOT_PLAN.md §32 一致。
- 旧 AutoMoT 原模型里仍有 `l2_3s` / 6 个 0.5s waypoint 的 legacy 轨迹 head 注释；
  这是原 fast head 的内部监督/指标，不代表当前 tp/ntp prompt 又回到 1.5s/3s。
  当前 prompt 决策语义固定为 `now/+1s/+2s`，LeadMoT waypoint 监督固定覆盖 2s。
- `run_eval.sh` 必填 `--leadmot-ckpt`，支持 scenario / route_id / random / full；
  默认自动选 1 张空闲 GPU，`--num-gpus N` 或 `EVAL_GPU_COUNT=N` 会自动选 N 张空闲 GPU，
  每张卡一个 worker、独立端口槽，round-robin 分 route。launcher 只扫描空闲
  `[rpc..rpc+3, tm]` 端口块并实时 tail worker log；CARLA server 由
  `leaderboard_evaluator.py` 在 worker 进程内启动并清理，避免双重启动抢端口。worker log
  只落 `/tmp/leadmot_eval_workers.*` 临时目录，退出后删除，不在结果目录持久保存。
- **输出按跑法分目录**：
  - `<signature>/route<id>/` 视频与 `<signature>/eval_per_route/eval_<id>.json` 跨跑法共享
    （断点续跑）
  - `<signature>/runs/<RUN_LABEL>/` 本批次聚合，按 `full` / `scenario_X` /
    `random_NN_SK` / `routes_A+B` / `smoke_<id>` 等自动命名；始终写
    `summary_all.json`、`summary_report.md`、`scenario_table.csv`、
    `route_results.csv`、`run_manifest.json` 与 `scenarios/<Scenario>/summary.json`
    （`log.txt` 记录本批终端 stdout/stderr；`run_manifest.json` 含 started/finished 时间、attempted_count、failed_routes、worker_fail）
  - `summary_report.md` 是人类可读实验总结，解释 planned/evaluated/coverage/success_rate/
    perfect_rate/score/infractions；`scenario_table.csv` 是论文表格友好汇总，
    `route_results.csv` 是每条 route 状态/分数/违规明细
  - `<signature>/summary_all.json` 是跨批次总聚合（所有已评估 route）；有 manifest 的
    run 聚合会把计划但缺失 eval JSON 的路线记为 `MISSING_EVAL_JSON` 并计入成功率分母
- 输出 signature 包含 ckpt 父目录、ckpt stem、`bev{0|1}`、`ema{0|1}`，避免不同模型/BEV/raw-EMA 覆盖。
- 五路视频：`input` / `debug`（相机 overlay） / `bev_debug`（顶视 LiDAR+pred+tp+ego box，
  LEAD `video_recorder` BEV pseudo-image 等价）/ `demo`（spawn cinematic + 顶视 carla camera）
  / `grid`（demo+input 上下拼接）。
- 子包内 Python class/function 已补中文 docstring；shell / HTML / CSS 在关键逻辑块前有中文注释。
  后续改传感器、坐标、warmup、GPU worker、输出 JSON 或 webapp API 时，同步更新代码注释和
  `EVAL_CARLA_*` 文档。

## 7. GoalGen

文件：

- `qwen3vl_local/goalgen/build_dataset.py`
- `qwen3vl_local/goalgen/train.py`
- `qwen3vl_local/goalgen/eval.py`
- `qwen3vl_local/goalgen/probe.py`
- `qwen3vl_local/goalgen/GOALGEN_RUN.md`
- `qwen3vl_local/goalgen/GOALGEN_V1.md`
- `qwen3vl_local/goalgen/GOALGEN_V2.md`

语义：

- 输入：history RGB -> frozen Qwen prefill/KV；history/target RGB -> frozen VAE latent。
- 目标：生成未来 subgoal keyframe latent。
- 训练：rectified flow，`z_t=(1-t)z0+t z1`，预测 `v=z1-z0`。
- 推理：Euler 从 t=0 到 t=1，decode 成 RGB。
- **z0 默认为纯噪声 `z0 ~ N(0, I)`**（`z0_prior_alpha=0.0`）：
  之前默认 `alpha=1.0, sigma=1.0` 把当前帧 latent 掺进 z0 → 低 t 区域 `z_t` 主要由
  "当前帧 + 噪声"主导，`v_target = z1 - z0` 里 z0 含 z_current → 模型偷懒学
  "还原当前帧"而不是 subgoal。**统一改回纯噪声起点**（flow.py / train.py / eval.py /
  probe.py / train.sh / GOALGEN_V1.md / GOALGEN_V2.md 同步），让模型必须从噪声生成 subgoal。
  image-to-image ablation 时显式 `--z0-prior-alpha 1.0`（此时推理 `z_init` 必须用同样
  混合方式构造，分布才一致）。**v1 和 v2 共享同一份代码，默认值改动同时覆盖两者**。

v1/v2：

| 项 | v1 | v2 |
|---|---|---|
| 数据 mode | `--mode v1` | `--mode v2` |
| transition | 4 类，含 initial/final 两端 | 2 类，只保留 middle 之间 |
| 默认训练 | 从零 | 从 v1 `latest/best.pt` warm start |
| 代码 | 同一套 | 同一套 |

v2 train/eval/probe 仍复用 `goalgen/train.py` / `goalgen/eval.py` / `goalgen/probe.py`；
通过 `--val-jsonl checkpoints/goalgen_v2_data/val.jsonl`、`--save-root
checkpoints/goalgen_v2_dit` 或 `VERSION=v2` 切换。v2 train/eval/probe 都会校验样本
只能是 middle 子目标之间的转换，并拒绝明显的 `goalgen_v1_*` train/val/save/ckpt 路径
（但 v2 train 的 `INIT_FROM_CKPT=goalgen_v1_dit/...` warm start 是允许的）；误传 v1
数据时不再静默混跑 initial/final 两端样本。v2 的 counterfactual 干预默认
`counterfactual_scope=middle_transitions`：
只允许 `middle[0]→middle[1]` / `middle[1]→middle[2]`，显式排除
`initial→middle[0]` 与 `middle[2]→final`；若 v2 probe 选中两端样本会直接报错。
v2 下 `--counterfactual-scope all` 会被拒绝，内置 `--counterfactual-config default`
只生成 middle-only 候选，不复用 v1/full-scope 候选表。
GoalGen 文档已拆分：`GOALGEN_PLAN.md` / `GOALGEN_RUN.md` 只保留索引；
版本细节分别看 `GOALGEN_V1.md` 与 `GOALGEN_V2.md`。

当前共享架构，不属于某个 dataset mode：

- VAE latent `(C=16,T=1,H=48,W=144)`
- patch size `4`，token 网格 `12*36`
- hidden `1024`，heads `8`
- DiT layers `12`
- Qwen 36 层切 12 段，head_dim=128

## 8. SFT

统一文档入口：

- `qwen3vl_local/sft/SFT_PLAN.md`
- `qwen3vl_local/sft/SFT_RUN.md`
- `qwen3vl_local/sft_v2/SFT_V2_PLAN.md`
- `qwen3vl_local/sft_v2/SFT_V2_RUN.md`

**v1 / v2 双轨已废弃**（含 ms-swift 入口、loss_scale plugin、runtime_teacher_data
manifest 复用机制）。现在只有一套统一 SFT：

- `qwen3vl_local/sft/build_dataset.py` 产 `pending` jsonl，assistant 段 ANALYSIS
  为 `__TEACHER_PENDING__` 占位。
- `qwen3vl_local/sft/train.sh` → `train.py`：torch DDP + 手写 train loop +
  `peft.LoraConfig` / `get_peft_model` 直接把 LoRA 注入 base，不再走 swift。
- 每个 train batch 内部禁用 adapter，并调用底层 Qwen base model 现场 greedy
  生成 ANALYSIS 真值，立即喂进 student forward；这样避开 `PeftModel.generate`
  在 Qwen3-VL 上的生成错位问题。**不再离线物化 teacher、不再写 manifest、
  不再有 runtime_teacher_data 复用**。代价：训练时间约为 base LoRA 的 3-4 倍
  （每 step 多一次 4B 生成）。
- loss 在 `train.py` 内显式按 char-range 切段加权：ANALYSIS body 权重
  `SFT_ANALYSIS_WEIGHT`（默认 0.5，学习大致语言推理但不逐字压过状态监督）；
  `ANALYSIS:` / `\nSTATUS:` / `\nSUBGOAL:`
  字面、STATUS / SUBGOAL event_name、tail / EOS 全部 1.0；user / system prompt 段 0。
  旧版 v2 "结构字面 mask=0" 是已确认致命陷阱，新 mask 不留这个坑。
- ANALYSIS 语言学习仍走 token-level teacher 蒸馏，不引入 embedding / 偏好式
  semantic loss；如果需要减少固定措辞，训练时显式设
  `SFT_TEACHER_TEMPERATURE=0.2~0.3` 让现场 teacher 轻微改写。
- `qwen3vl_local/sft/build_teacher.py` 仅保留作可选离线工具（手动 dump teacher
  输出供 review / inspect）；不再被训练入口自动调用，也不再写 manifest。

eval 端固定坑：

- Qwen3-VL 上 PEFT wrapper forward 可能错位；`eval.py` / `probe.py` 默认
  `merge_and_unload` 把 LoRA 合并进 base 再推理。
- SFT v2/v3/v4 的 `sft_v*_adapter_config.json` 会记录完整 target module 路径用于审计；
  PEFT 自带 `adapter_config.json` 可能只保留短名 target modules。加载端按后缀兼容校验，
  不要求两份 JSON 的字符串集合逐字相等。
- `--max-gen-tokens` 默认 256（teacher ANALYSIS 80-150 token + STATUS/SUBGOAL 段，
  必须 ≥ 200，否则解析不到 STATUS）。
- partial-continue fallback 是永久兜底，不代表模型健康。
- `dataset_version="pending"` 时 GT ANALYSIS 段是占位，STATUS/SUBGOAL 评测不受
  影响；想做 ANALYSIS 内容对照，跑 `build_teacher.py` 物化 val 后再传 `--val-jsonl`。
- `eval.py` 小样本 full dump 与 `probe.py` 默认保存专家语言对照：
  `expert_analysis.txt` / `language_compare.json`。expert 来自 base teacher prompt +
  PRIVILEGED，model 来自 LoRA 自己生成的 ANALYSIS；用于检查是否学到大致语言推理，
  而不只看 STATUS/SUBGOAL。

### 8.1 SFT v2 串行选择题路线

`qwen3vl_local/sft_v2/` 是用户明确要求新增的独立路线，不替换旧 `sft/`：

- `prompts.py` 是唯一 prompt 来源。prompt 先列出全部 `SCENE_CHOICES` 及自然语言描述，
  stage-1 只要求模型输出 `SCENE`；stage-2 作为同一条对话里的后续 user prompt，
  只列出预测 scene 的 `EVENT_SEQUENCE` 和事件描述，再要求输出 `STATUS/SUBGOAL`。
  推理时 stage-2 要复用 stage-1 已经吃过图像和场景 prompt 的 KV cache。
- assistant 目标拆成两段：stage-1 `SCENE:`，stage-2 `STATUS:` / `SUBGOAL:`。没有
  ANALYSIS，没有 `__TEACHER_PENDING__`，训练时不跑 teacher.generate。
- `build_dataset.py` 复用旧 SFT 的 keyframe timeline 与 keep/advance 采样，输入仍是
  4 张 LEAD stitched RGB；`SCENE` 监督来自 scenario，`STATUS` 来自 anchor 所在 GT
  interval，`SUBGOAL` 由 `prompt_pipeline.get_full_sequence()` 推导；默认
  `--samples-per-scenario 0` 表示每个场景保留全部合法候选，正数才启用下采样；
  默认 `--wrong-scene-ratio 0.15` 只增强 train rows，把一部分 stage-2 selected scene
  替换成错误场景但仍监督真实 `STATUS/SUBGOAL`，val rows 保持 GT 分支。
- `train.py` 直接 LoRA 注入本地 Qwen3-VL-4B-Instruct；默认只注入语言侧 Linear。
  视觉侧通过 `--lora-vision-scope` / launcher `LORA_VISION_SCOPE` 选择
  `off` / `merger` / `last4` / `all` 四档；`--lora-vision` / `LORA_VISION=1`
  仅作为 `all` 的 legacy 别名保留。开启视觉 LoRA 时默认带视觉组低 LR
  (`--vision-lr-scale=0.1`，受 `--max-vision-lr-scale=0.25` 上限约束)、
  语言/视觉分组梯度裁剪 (`--language-clip-norm=1.0` / `--vision-clip-norm=0.3`)
  与 TensorBoard 梯度/参数范数观测；`merger/last4` 会解析视觉 block 编号，
  默认 `--strict-vision-scope`，解析不到 block 编号时直接报错，只有显式
  `--no-strict-vision-scope` 才退化为只训 merger/patch_embed 并 warning。
  `--vision-guard-enabled` 默认开启，连续视觉 grad/param norm 异常达到
  `--vision-guard-patience` 会停训并写 `fuse_stop_step_<N>/` + `fuse_reason.txt`，
  同时跳过正常 `final/` 保存，避免把熔断产物误当完整训练结果。
  base Qwen checkpoint 始终只读，训练产物只保存 adapter delta；adapter 目录同时写
  PEFT `adapter_config.json` 和 `sft_v2_adapter_config.json`（含 scope 与保险参数），
  eval/probe 加载前按配置判断普通 LoRA / 视觉 LoRA 并校验权重 key，不一致直接拒绝。
  prompt token 权重为 0，
  只监督 scene/status/subgoal 值 token；
  `SCENE:` / `STATUS:` / 换行等格式 token 为 0 loss。每条样本是一条多轮 teacher-forced chat：图像只在第一轮 user，
  第二轮 status prompt 作为文本 follow-up 接在 scene assistant 后，单次 forward
  里同时计算三个值 token 的 loss。
- `eval.py` 先自由生成 `SCENE`；若 scene 不在白名单，样本立即中断并计 invalid；
  若 scene 合法，即使 scene 错，也按预测 scene 的 event sequence 构造 stage-2 prompt，
  接到 stage-1 KV cache 后继续生成 `STATUS/SUBGOAL`。统计 `scene_accuracy/status_accuracy/subgoal_accuracy/all_accuracy`；
  其中 status/subgoal 主指标是串行口径，scene 错时后续即使 event token 偶然同名也算错。
  同时记录 raw exact 与 `invalid_scene_rate`、`invalid_status_for_pred_scene_rate`、
  `subgoal_not_next_rate`，用于观察串行错误传播；另记录 `valid_total` 与
  `*_valid_scene` 指标，用合法 scene 子集作分母。

## 9. VAE Patch/Unpatch

入口：`AutoMoT/vae_standalone/train_patch_unpatch.py`。

训练目标：`image -> VAE.encode -> patch -> unpatch -> VAE.decode -> image`。
VAE 冻结，只训练 patch/unpatch。产物 `patch_unpatch_*.safetensors` 可被
`DiTMoT.load_patch_unpatch` 直接加载，key 与 `self.patch` / `self.unpatch` 对齐。
默认产物路径为
`AutoMoT/checkpoints/patch_unpatch_v1/<run_TS>/weights/patch_unpatch_best.safetensors`，
base 层维护 `latest -> run_TS`；`vae_reconstruct.py` 与 GoalGen 训练共用
latest / 无 run_subdir / 最新 run_* 的兜底解析顺序。

GoalGen checkpoint 记录 patch/unpatch 来源：

- `patch_unpatch_source=external`：训练由 `PATCH_UNPATCH_WEIGHTS` /
  `--patch-unpatch-weights` 加载外部 `patch_unpatch_*.safetensors`；空参时
  `goalgen/train.py` 自动解析默认 best。找到后默认冻结，并把绝对路径写入
  `dit_config.patch_unpatch_weights`。eval/probe/runner 在 strict load DiT 后会按该
  路径再次覆盖 patch/unpatch 并恢复冻结语义。三处默认路径都找不到时直接报错，
  不再回退到随机初始化。
- `patch_unpatch_source=checkpoint`：仅用于显式 `PATCH_UNPATCH_UNFREEZE=1`
  将外部权重作为初始化并继续联合训练，或 warm start 时显式
  `PATCH_UNPATCH_CKPT_FALLBACK=1` 使用 ckpt 内自带 patch/unpatch。eval/probe/runner
  此时直接使用 `--dit-checkpoint` 内自带的 patch/unpatch。
- GoalGen eval/probe/runner 使用 EMA 时，先以完整 `dit_state_dict` 打底，再用
  `ema_state_dict` 覆盖可训练参数；外部冻结 patch/unpatch 不在 EMA shadow 中也
  不会缺 key。
- warm start 继承 `patch_unpatch_source=external` 时默认要求原 safetensors 仍存在；
  只有显式 `PATCH_UNPATCH_CKPT_FALLBACK=1` / `--allow-patch-unpatch-ckpt-fallback`
  才使用 ckpt 内自带 patch/unpatch 继续训练，并把新产物记为 `source=checkpoint`。

DDP 选卡：Python 内部 rank0 选卡，写临时文件，其它 rank 读取，避免每 worker
各自 `nvidia-smi` 导致 GPU 子集 race；前置 `GPU_IDS=0,1,2,3` 时跳过自动选卡，
直接把这组卡写入 `CUDA_VISIBLE_DEVICES`。

## 10. GPU 选址统一规则

适用：SFT、GoalGen、LeadMoT 与 VAE patch/unpatch 的训练、
eval、probe、teacher / 推理入口。

- 默认调用 `nvidia-smi` 自动挑空闲 GPU，并覆盖外层残留的 `CUDA_VISIBLE_DEVICES`。
- 单进程默认挑 1 张；进程内通常用 `cuda:0`。
- `torchrun --nproc_per_node=N` 默认挑 N 张，并按 `LOCAL_RANK` pin。
- `DDP_GPU_COUNT=N` / `NPROC_PER_NODE=N` 只表示默认自动选址时需要 N 张卡，具体卡号默认自动挑。
- 训练脚本显式 pin 卡统一用 `GPU_IDS=0` / `GPU_IDS=0,1,2,3` 前置；非空时跳过
  `nvidia-smi` 自动选址。SFT / GoalGen / LeadMoT bash launcher 的卡数从 `GPU_IDS`
  逗号数推断，`DDP_GPU_COUNT` 被忽略；直接 `torchrun` 的 VAE 示例仍要让
  `--nproc_per_node` 与 `GPU_IDS` 数量一致。
- 文档示例不要写 shell 手动 `CUDA_VISIBLE_DEVICES=...`。
- 显式 `--device cpu` / `--device cuda:N` 的 Python 入口通常视为用户锁设备，不覆盖。
- GoalGen eval/probe 的 `--gpu N` 只在单进程下锁进程内 GPU；默认保持 0。
- `eval_carla/run_eval.sh` 用 `--num-gpus N` / `EVAL_GPU_COUNT=N` 表示闭环评测 worker 数；
  具体 GPU id 仍按 `nvidia-smi` 空闲排序自动选，并给每张卡分配独立 CARLA 端口槽。
- 白名单 bash launcher 开头统一执行 `ulimit -S -c 0 2>/dev/null || true`，禁用 core dump，
  避免工具进程异常时生成 `core.*`；新增运行入口也要加，若工作区已有 `core.*`，不要入库，先问用户是否清理。

## 11. Run 目录防覆盖规则

训练入口默认写：

```text
<OUTPUT_DIR_BASE>/run_<RUN_TAG>/
<OUTPUT_DIR_BASE>/latest -> run_<RUN_TAG>
```

规则：

- bash launcher 在启动 torchrun 前计算一次 `RUN_TAG`。
- Python VAE 入口由 rank0 生成 run tag 后 broadcast 给其它 rank。
- `NO_RUN_SUBDIR=1` 回到旧式覆盖行为，只作排查。vae 入口也接受 `NO_RUN_SUBDIR`，旧名 `PATCH_UNPATCH_NO_RUN_SUBDIR` 作为兼容别名保留。
- `HF_HOME` 挂在 base 层：`<OUTPUT_DIR_BASE>/.hf_cache`。
- `qwen3vl_local` 下训练 / eval / probe / eval_carla launcher 默认会在本次产物同目录追加
  `log.txt` 保存终端 stdout/stderr；外层 shell 已 tee 时用 `QWEN3VL_LOG_ACTIVE=1`
  防止重复记录，可用 `QWEN3VL_LOG_TO_FILE=0` 临时关闭。
- SFT 不再保留 runtime teacher cache；teacher 在 train batch 内现场生成且不写盘。

## 12. 不要做

- 不要改 `lead/`。
- 不要把 `0026.json` 或仓库根目录 / `AutoMoT/lead_data` 下的 `keyframes_all_scenarios.json` 入库。
- 不要运行 CARLA、`AutoMoT/test.sh`、`start_carla.sh`、大规模下载或安装命令。
- 不要把 AutoMoT legacy slow/fast 接口重新接回本地 Qwen/LeadMoT 路线。
- 不要把当前共享 GoalGen 架构描述成某个 dataset mode 专属架构。

## 13. 快速导航

| 任务 | 文档 |
|---|---|
| SFT 跑法 | `qwen3vl_local/sft/SFT_RUN.md` |
| SFT v2 串行选择题跑法 | `qwen3vl_local/sft_v2/SFT_V2_RUN.md` |
| SFT v3 offline OPSD 跑法 | `qwen3vl_local/sft_v3/SFT_V3_RUN.md` |
| SFT new loop phase2 单轮 EVENT 跑法 | `qwen3vl_local/sft_new_loop_phase2/SFT_NEW_LOOP_PHASE2_RUN.md` |
| SFT loop phase3 事件 gate 跑法 | `qwen3vl_local/sft_loop_phase3/SFT_LOOP_PHASE3_RUN.md` |
| SFT v5 RS/EVENT OPSD 跑法与可视化 | `qwen3vl_local/sft_v5/SFT_V5_RUN.md`；可视化记录见 `qwen3vl_local/sft_v5/SFT_V5_VISUALIZATION_RECORD.md` |
| SFT base 直接选项基线 | `qwen3vl_local/sft_base/SFT_BASE_RUN.md` |
| SFT base simple 四格均衡基线 | `qwen3vl_local/sft_base_simple/SFT_BASE_RUN.md` |
| GoalGen 跑法 | `qwen3vl_local/goalgen/GOALGEN_RUN.md` 索引；版本细节看 `GOALGEN_V1.md` / `GOALGEN_V2.md` |
| LeadMoT 跑法 | `qwen3vl_local/leadmot/LEADMOT_RUN.md` |
| LeadMoT 架构 | `qwen3vl_local/leadmot/ARCHITECTURE.md` |
| LeadMoT CARLA 闭环评测 | `qwen3vl_local/eval_carla/EVAL_CARLA_RUN.md` |
| LEAD RGB 批量转视频 | `lead_video_tools/LEAD_VIDEO_RUN.md` |
| 关键帧 / ROAD-EVENT 重标注 | `keyframe_filter/ROAD_EVENT_CLASSIFICATION_PLAN.md` |
| 规则入口 | `AGENTS.md` / `CLAUDE.md` |


## Phase3 映射与动作审计补充（2026-09-05）

Phase3 v2 保持五动作；完整 RS 四问全 NO 恢复 R3，未问不作 NO，HIGHWAY 为独立事实；并发异常保留。普通无灯路口不自动 U-E7，原 U7 用既有灯故障答案表适配；新增 R5/R-E5 常规让行，与七异常及 R-E2/R-E3 共十个 context 1:1。R-E2 包含目标变道及绕障恢复，不按 24 帧截断，两条变道 NO 不清除恢复状态；最终目标 y 符号不决定变道侧。invalid 必须覆盖每个 asked context；未来轨迹只用于离线标签，默认异常 route/RGB 风险过滤。逐帧人工审计与机器覆盖分开记录；`sft_new_loop_phase3/eval.sh` 与 `run_full_pipeline.sh` 会在 eval 结束后生成默认不超过 30MB 的 `*_audit_bundle.tar.gz`，只包含 metrics/summary/manifest/adapter 文本元数据和抽样下采样 RGB，排除权重/checkpoint/TensorBoard。详见 sft_new_loop_phase3/MAPPING_AUDIT_20260905.md。

本轮不修改 sft_new_loop_phase1/2 的代码、prompt、已审计答案表。原始 taxonomy 的
OppositeVehicleTakingPriority/R5/U-E7 与 Phase1 TRAFFIC_LIGHT_ABNORMAL=NO 不一致，
只在 Phase3 source_mapping 适配；ROAD_EVENT_CLASSIFICATION_PLAN 已注明历史语义差异。
当前动作规则使用4Hz真实速度顺序、连续lane身份及完整观察窗口；跨road连接段暂不强标横向NO，
planned route不作为原车道中心线真值。现有582条RGB审计缓存可复用，机器扫描不表示人工完整动作确认。

2026-09-05 扩展复核覆盖197个scenario×Town连续图组（42场景、11 Town、7826个缩略帧格），
不等于582条route全程人工确认。InvadingTurn/Town13的同road355 lane-id在98/128变化却未见
RGB跨线，已加入Phase3精确横向未确认清单：FULL_MANEUVER跳过，纵向组保留并记录证据不足。
同road+连续lane-id仍只是候选证据，原始xodr lane-section连通尚未核验。
本轮诊断索引filtered_v5共1152条，三split各十context×32+64invalid；训练默认和定额eval均
对齐invalid/valid=20%（总量16.7%）。详见Phase3映射审计报告及本地review.html，未训练模型。

2026-09-05 后续边界审计：在上述记录上新增两轮52个连续图组条目及10条原图/补充图组
记录，合计259条notes、244条不同route；仍不是582条路线全程人工确认。
已查明 `expert.py` 保存的 road_id/lane_id 来自 LaneType.Any，InvadingTurn/Town13
1777_0 的 f98–127 为 Shoulder，而另一遍 Driving 查询的 ego_lane_id 仍为-1。
Phase3 横向监督要求完整 Driving 窗口，不混用缺少配套 road_id 的 ego_lane_id。
动作规则 `ordered_speed_driving_lane_v5` 排除孤立加速峰值，并允许起步后在正常速度范围
轻微回落；prompt 为 `v3_boundary_audited`，train/eval 显式拒绝旧规则索引。
新增 Phase3 专用 ParkingCutIn/Town13 1008_0 f96–98 U3 正例，不写回Phase1/2；另隔离
VehicleTurningRoute/Town04 的 f2–10 错误 R5/RE5 候选。最新诊断数据 filtered_v8 共
12,928候选、1152平衡行。58项回归、三split实际train/eval采样20seed检查通过，无模型加载。
`dispatch.py` 是保留并发UE/恢复状态并报告缺失RE gate的接口，尚未接入在线runner；
`build_same_rs_challenge.py` 仅有1route/5题同RS错事件诊断，不是独立holdout，未并入训练。
当前正式索引invalid仍只覆盖错误RS，不能宣称已具备全情境同RS事件纠错能力。
后续入口：`sft_new_loop_phase3/BOUNDARY_AUDIT_20260905.md`；Phase1/2对HEAD仍无改动。


### 2026-09-05 Phase3 后续落实：同 RS 负例与常规候选核对

Phase1/2未改。本轮原图15路线84张，累计274 notes/248不同路线；不是全route确认。
新增 `same_rs_invalid.py` 将9个route/anchor的34道人工同RS错事件候选接入主构建；
train/eval保留invalid_reason、同RS已具备的asked-context覆盖，构建与运行时共用
source/联合配额。`filtered_v12` 为582路线审核子集的12,896有效候选/1152均衡题；
相比v8排除32个正候选（过早RE5 12、主路伪RE3 8、雾中U5提前段12）。
三个split分别54/10、56/8、56/8个wrong-RS/same-RS invalid，七UE的同RS题跨三split
均有；RE同RS覆盖不全，不凭空补负例。十个有效context仍各32、每split总384题。
`dispatch.plan_candidate_requests/candidate_response` 提供常规候选核对，不把R3/R5
当RE真值；未驳回且动作全NO不表示恢复完成，接口未接CARLA。五动作不变。
prompt更新v4_context_recheck；动作规则仍Driving v5；训练/eval新增mapping_contract_hash
校验绑定人工决定，旧索引必须重建。61项回归与三split各20seed实际采样通过；未训练。
最新状态以 `sft_new_loop_phase3/BOUNDARY_AUDIT_20260905.md` 为准，v8及此前数字为历史。


## Action prior 轨迹训练（2026-09-05）

新增 `qwen3vl_local/action_prior/`，运行入口 `run.md` / `run_full_pipeline.sh`。
它冻结 Qwen3-VL-4B-Instruct、Phase1/2 LoRA 和 LEAD BEV；v5 将最终图文/分析 KV
作为条件，使用条件 Flow Matching 生成连续轨迹。旧的 LeadMoT Linear+cumsum 轨迹头、
坐标回归 loss 和 checkpoint 仅是历史实现，不能与 v5 混用。
新 Phase1 已融合事实四问和 RS，先全问再四个 RS 分层复核；新 Phase2 没有训练
hierarchical EVENT 接口，采用两个已有问题域全问和真实 assistant 后的同域续问复核。
两次一致只是 condition 接受，不代表真实准确率；invalid 字段留空，样本继续参加轨迹训练。

新增 `qwen3vl_local/action_expert_ablation/` 作为 action expert 对照路线，不读取
RS/EVENT 标定或 Phase1/2 LoRA。`qwen_simple/` 只使用四图 + LeadMoT 原简短导航 prompt
的 base Qwen KV + frozen BEV；`bev_only/` 不初始化 Qwen，给 Prefix-KV attention
传 zero-length prefix，保留 frozen LEAD BEV（当前 stitched RGB + LiDAR BEV 融合）、
speed、target_point、next_target_point、final_goal 和 query token；它不是纯 LiDAR/完全无视觉。
两者复用 action_prior 的 shared dataset index、联合轨迹
Flow Matching decoder、FP32 AdamW/EMA、DDP/验证节奏和 step 样本预算；共享索引首次构建用
`.build.lock` 文件配合 `flock` 加锁，进程退出自动释放，残留锁文件不阻塞后续启动，并在锁内重查 split 完整性，避免两个变体并发写同名 tmp；epoch 尾部不足
完整累积窗口时只在同索引/同卡数/同累积/同 seed 下可比。full pipeline 训练前固定本次
`RUN_TAG` 和真实 run dir，`--resume` 指向 `latest/latest.pt` 等软链接时先解析真实 checkpoint，
最终 eval 直接读同一 run 的 `best.pt`；CLI `--data-root/--data-dir` 和显式
`MODEL_DIR`/`--model-dir`、`LEAD_BEV_CKPT`/`--lead-bev-ckpt` 贯穿构建、训练和 eval。
仅传 `--resume` 时训练入口先从 run `config.json` 恢复原 LR/epoch/梯度累积/索引等参数，
launcher 在选 GPU 前从 `training_plan.json` 恢复原 `world_size` 默认值，`train.sh --resume`
不注入脚本默认 LR/epoch/梯度累积/索引；显式 CLI 或环境变量覆盖仍优先生效。
resume 会归档 TB 中 checkpoint step 之后的旧 event，并保留 checkpoint step。
执行指纹覆盖消融入口、共享 action_prior/LeadMoT/BEV 依赖和关键运行库版本，但不绑定未使用的 Phase1/2 prompt。
TB 只记录核心 FM/planning loss、ADE/FDE、LR、grad_norm、吞吐和显存，不记录 RS/EVENT/UNKNOWN 分桶。
运行见 `qwen3vl_local/action_expert_ablation/run.md`。

2026-09-09 共享训练重构：主线 `action_prior/train.py` 与消融 `common.py` 都调用
`action_prior/training_core.py`，统一模型构造、FP32 AdamW/EMA、分片/累积、训练和验证、
日志、best/最新 checkpoint 调度及恢复配置校验。入口只注入各自 runtime、条件合同和审计接口；
主线保留先验分组，消融不产生 RS/EVENT 分组和先验 case 审计。
后续修改公共训练行为必须落在共享模块，不能在消融复制循环。
三组新增 `train/samples_seen` 累计训练呈现数与 `train/step_samples` 本次更新样本数；
`train/samples` 仍是日志窗口计数。默认完整 step 为四卡×16=64 case，epoch 尾部按实际分母。
`action_prior` 对 `qwen_simple` 默认衡量先验+prompt 的整体变化；仅开启 `--generate-analysis` 才额外含分析，不能孤立归因于推理文字；
`bev_only` 仍含 BEV RGB 融合。FM MSE 仅为训练诊断，最终比较同口径采样 ADE/FDE 与 test。
共享循环兼容两种历史 pending cursor 字段，但代码指纹已改变，旧 run 仍需原代码恢复。
CPU 回归覆盖同模拟条件三入口更新/指标一致、两个消融中途及验证中断恢复、预算完成后不超训；
未运行真实 Qwen/BEV GPU/DDP、CARLA 或真实 TB event 恢复。

2026-09-10 训练终止恢复：共享 `training_core.py` 捕获 `SIGTERM/SIGINT` 时只置请求标记，
在完整梯度累积窗口后的 optimizer 安全点跨 rank 同步，再由全部 rank 参与 RNG 汇总并原子保存
`latest.pt`；validation 内信号会让各 rank 按带空轮的同步协议一起退出，不发布残缺 validation 指标。
rank0 写 `termination.json`，checkpoint 落盘后 barrier，再以 143/130 退出阻止 full pipeline 继续 eval；
resume 会归档旧终止标记。launcher 多卡时优先向 torchrun worker 转发信号，单卡直接通知训练进程；
退出显式关闭已登记 DataLoader iterator，短期 validation loader 禁用 persistent workers。
该机制无法处理 `SIGKILL`、节点掉电或永久卡死，届时仍回退周期 checkpoint。本次执行代码变化会改变
主线和两个消融的严格指纹，旧 checkpoint 仍只能用对应旧代码恢复。


自动权重选择只在 `best_generation/` 内查实际权重，校验生产 prompt name/hash、Git commit、
base 路径和 RGB 2/4图配置，并回查保存 step 的 generation 验证分数；没有 final 兜底。
允许显式指定兼容 adapter；来源、完整权重指纹和代码指纹写入 config/selected_priors/checkpoint。
两个 LoRA 共用 base 但独立启停，禁止 merge。所有最终图文+自然场景先验+短摘要 KV 在
`disable_adapter()` 下完整 prefill；禁用上下文退出也强制冻结参数，防止 PEFT 自动恢复梯度。
本入口还在构造 BEV 时禁用旧 timm pretrained 下载，随后 strict 导入本地 LEAD backbone。

冻结问答/简述可按权重与代码合同、实际四图字节、导航和 sample seed 缓存压缩文本；
首次和命中采用相同完整 assistant transcript 重建 base KV，不缓存 GPU KV 或图片。
缓存命中样本的 invalid 仍计入每轮真实呈现计数；这不是 SFT teacher 离线物化机制。

默认索引保留所有合法 4Hz anchor，先排除异常时长 route，按物理路线聚合 Rep/时间戳后
约80/10/10划分。默认61 epoch、4卡×1clip×16累积、LR2e-4、5%warmup+cosine，
每250 optimizer steps验证256帧，每完整epoch全量val选best.pt；实际数据量与步数写training_plan。
此配置参考LEAD/现有LeadMoT，尚未针对新KV分布调参验证。支持DDP、EMA、TB、精确cursor/RNG
断点恢复、独立eval/probe和CPU测试；尚未执行真实权重、多卡训练或闭环评测。
新checkpoint使用独立qwen_backbone schema，旧LeadMoT/eval_carla明确拒绝，不得直接交给旧入口。
Phase1/2/3源码均未修改；Phase3未接入。


### Action prior 首轮审查修订（2026-09-06，部分行为已由下节替代）

用户审查指出直接 BF16 AdamW 小更新舍入、简述仅格式验收、CLI 被索引字段覆盖等问题。
现可训练 decoder/AdamW/EMA 保持 FP32，仅 decoder 前向 autocast，预测转 FP32 再算 loss；
受控三段简述完整表达全部 Phase1/RS/EVENT/域适用性 YES/NO/UNKNOWN，仅容忍空白变化，
拒绝反义/遗漏/额外断言，失败用相同完整模板，原始输出只作审计，不声称自由文本语义验证能力。
导航 CLI 在 read_rows 统一覆盖，单帧 BEV / 4Hz / route10 / waypoint8 非默认值直接拒绝。
共享文件缓存跨 rank 复用，POSIX 锁与原子发布保证并发同 key 首次只计算一次；最终 KV 仍每次 base prefill。
执行指纹覆盖共享 helper/LeadMoT/prompt 家族、只读 runner/BEV 源码及依赖版本，参考源码不入库。
训练 Git 是溯源记录，实际代码哈希才参与兼容比较；不同驱动硬件仍需 smoke。新 checkpoint 为 v2，旧 v1 明确拒绝。

复核保持 history 默认，提供 independent 和 compare（condition 仍按 history）的无训练 audit_priors 入口；
显式统计 UE6 prompt 相同，不能把独立 greedy 重复或带历史一致当成筛错能力证据。
从所选 adapter 同 run manifest 实际 index 读取训练候选池，缺失来源标 unknown；显式迁移 index 标记 override。
报告 action 各 split 的物理路线池重叠/池外/未知；这不是实际 sampled route 追溯，池外也不等于整个系统未见。
轨迹指标按预测事件条件、invalid、简述来源、上游池关系分别归一化；run_ablation.sh 同初始化/数据/预算分别训练
原 base prefill 与 prior，两组完整 test 后再配对256帧子组比较，组归属统一采用 prior case，不能当 GT 事件指标。
真实权重/生成质量/吞吐/多卡效果仍未验证，61 epoch 和 LR2e-4 仍是起始配置，不能据此直接推断成本与增益。


### Action prior 后续需求对齐（2026-09-06，以本节为当前状态）

撤掉 VERIFIED_SUMMARY/唯一模板验收。base 自行组织 Scene/Interaction/Planning context 三段简析，
当前速度、导航几何和接受条件共同约束分析；随后同一禁用 LoRA 的 base 重新做纯文本 prefill，
复核一致性/道路覆盖、正类覆盖、未知字段、额外断言和导航依据。严格解析五个布尔项，全部通过保留原文，
失败才用完整 fallback；fallback 的 planning 也随已解析速度/目标方位/正类条件变化。
这是模型审查而非确定性语义保证，同源误判仍可能发生；真实模型质量尚未验证。
缓存保存草稿/复核及配对 SHA，最终 transcript 不包含复核或失败草稿，最终 KV 仍全由禁用 LoRA 的 base 计算。
语言协议 v3，FP32 checkpoint 容器仍 v2；旧照抄模板合同不可混用。

执行哈希由显式真实/延迟入口加模块初始化 import 展开，当前48个源码，未接入 Phase3 为0；
修改无关 Phase3 不再影响 action 恢复，真实 engine/M-RoPE/decoder/runner/prompt 改动仍失效。
新增延迟执行依赖须同步 seeds，参考源码只哈希不入库。上游训练索引审计的路径/获取模式/可用状态/
内容与生成 identity 分离；eval/probe 支持两份 training-index 重映射，来源异常标 unknown 并记录错误。
续训保留原审计路线快照使 epoch 累加口径稳定，同时报告新获取内容差异；独立 eval 使用当前来源，
同内容移动、显式指定或暂时缺失不会阻断相同生成条件。来源变化仍不改变“非严格系统未见 holdout”的限制。
轨迹分组新增 confirmation/all_confirmed、expected_domain_only、unconfirmed，避免正常域外被解读成复核失败。
Phase2 compare 明确记录仅 history 接受、不要求共识、跨模式分歧；仍是观察工具，未证明能筛掉真实错误，
也未伪造 EVENT hierarchical 接口。默认首次每帧至多11次生成（含纯文本复核），compare至多17次，另做最终 base prefill。


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

新增入口 `action_prior/bench2drive.py/.sh`、`carla_agent.py`、`carla_runtime.py`、
`benchmark_report.py`、`audit_bundle.py`。训练期仍每250更新验证256帧、每轮全量val。
原 action 段中“eval_carla 明确拒绝”仍指旧 launcher，专用 action 入口现已实现源码接入。
闭环 launcher 默认完整220、支持 subset smoke/显式resume/report-only；运行前CPU合同校验，
多GPU独立route与端口；使用当前Python，不改只读leaderboard/lead源码，不启动本地CARLA验证。
`--dry-run` 已按本地正式 XML 核对220/44；真实模型/传感器/控制表现和论文分数尚未验证。
官方 https://github.com/Thinklab-SJTU/Bench2Drive/blob/0.0.4/tools/ability_benchmark.py 的
Traffic Signs 新口径只给未成功但合法过路口的路线补成功，不沿用本地旧脚本的重复计数。
Comfort 沿用本地官方函数及其原始 angular_velocity 处理，采集10Hz与dt=.1对齐；指标版本、
训练数据/传感器/安全兜底差异必须披露。缺记录DS/SR以planned分母列零贡献并标provisional，
能力缺观测不给完整均值，不能用少于220条的已完成子集冒充完整得分。

Action prior 新增 `rank_loras.py/.sh`：CPU 只读扫描 Phase1/2 `best_generation`，命令行
打印保存 step 的全部 generation/teacher-forced/guard 指标及 prompt/Git/RGB/来源，分别推荐。
复用 `inspect_adapter` / `selection_score` 与生产同序排名，不混入 final/balanced/test 或其它step；
无推荐写报告后 exit 2。默认报告 `checkpoints/action_prior_lora_audit/run_<时间>/`，不入库。
它是已有验证记录对比，不是重新评测，不证明不同run在共同holdout上可比；运行见 action_prior/run.md。
2026-09-06 发现审计更新：`lora_audit.py` 跟随目录软链接并防循环去重，报告非 best 保存点、
旧/缺失配置、未识别 phase、压缩包、审计元数据副本和读目录错误。`rank_loras` v2 报告分开
记录发现状态与逐项合同 actual/expected，不再用第一个 prompt 错误遮住 Git/RGB/权重等问题。
Git 只检查训练 commit 非空，不要求与当前 checkout 相等；不篡改旧权重来源。推荐仍严格只取
生产合同通过的真实 best_generation；额外跟随链接发现的推荐需用显式 adapter 路径加载。
本地已有四个 sft_new_loop_phase2 审计包含 adapter_metadata/adapter 元数据但无实际权重，
这类副本可审计版本，不能作为可训练权重。两个远程服务器的实际目录需在对应服务器重跑新版确认。
`rank_loras` v3 修正来源归类：仅精确新 Phase adapter 配置或完整新包目录线索进入该 Phase
审计；旧 sft_loop_phase2_augment 等其它包只记录 excluded_other_packages，不计入候选/拒绝。
包名与提示词版本分开：sft_new_loop_phase2 的 v3 仍属新 EVENT 包，只是与当前 v5 协议不同。
143409 最新远程报告没有发现新 Phase2 adapter 配置；143428 有新 Phase2 的 12 个真实保存点
和 1 个无权重副本，但没有 best_generation。旧包的两个 best 不应算作新 Phase2 候选。

2026-09-06 用户随后明确授权先用现有模型训练 action：默认 `selection_policy=available`，
支持 `--checkpoint-roots` 合并共享目录/软链接。新 Phase1 当前 v5 best 优先按 validation Exact；
新 Phase2 优先兼容且 guard 通过的 best，没有时按 fallback validation Exact 选优，不使用 final。
支持新 Phase2 v3 冻结 prompt（源 6c722de11，2/4 RGB 哈希与远程记录一致）及当前 v5，
问组和运行文字按所选协议恢复；Git 缺失、原 fallback guard 失败只作警告，不篡改元数据。
同名 Qwen3-VL-4B-Instruct 可跨服务器路径重映射，旧 base 字节同一性未证明；实际本地 base、
BEV、LoRA 字节和真实执行代码进入合同。`strict` 保留原限制。完整 pipeline 预检打印候选与选择，
保存 selection_$RUN_TAG.json 并固定传给训练，防止数据构建后重选；resume/eval 固定 checkpoint
路径和哈希，不依赖原 manifest 物理路径。rank_loras 默认 available（独立报告 schema），strict
仍输出 v3 审计。真实模型/GPU 训练未在本机运行；新代码改变旧 action 执行合同。

2026-09-06 用户要求保存所选 LoRA 防止上游删除：训练 rank0 在模型加载前通过
`lora_bundle.preserve_for_training` 实际复制所选权重/原配置/保存步指标/来源到 action run/lora，
复制字节 SHA256 不变；checkpoint 合同记录本地相对路径。resume/eval/probe/闭环优先使用
checkpoint 旁 lora 的核验副本，整个 run 可搬迁，缺失/损坏不静默重选。Qwen/BEV 仍外部准备。
rank_loras 默认导出所选组合目录及 tar.gz+SHA256，逐文件复制校验、逐归档成员解压流校验；
只含选中 slot 和必要旁车指标/配置/提示词依赖/训练与导出 Git，不含其它中间 checkpoint。
输出包路径/每阶段模型/step/prompt/Exact，report.weight_bundle 保存清单；完整权重包不限30MB，
原 audit.zip 仍限30MB且不装权重。`--no-export-bundle` 只审计，`--lora-bundle` 固定已认可组合，
解压到目标 AutoMoT/checkpoints 后可直接预检/训练；仅有单阶段时可导出单阶段包用于合并搜索，
不能冒称双阶段完整。CPU 已验证删除原模型、压缩迁移及 action run 搬迁恢复，不代表真实GPU已跑。

2026-09-07 dataset-priors 模式可直接读取标定 RS/Phase1/EVENT 标签并默认关闭 analysis review，
冷启动每帧为 1 次 base 分析生成 + 1 次最终 base KV prefill；可用 PRIOR_NOISE 注入 RS/EVENT
confusion 或 invalid。噪声率、invalid 占比和 seed 都进入先验合同身份，因为 seed 会决定具体被破坏的帧；
eval/probe 默认沿用 checkpoint 记录的先验来源与噪声。dataset 标签搬迁后，run_full_pipeline resume 可只传
`--prior-labels /新路径`，脚本从原 config.json 恢复 dataset 模式，并把新路径贯穿续训、最终 test 和 probe；
闭环没有 dataset 标签，必须显式切回 LoRA 并披露条件迁移。

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

同日代码复审修复四处边界：action 的开发路线隔离原先只有 312 组，漏了最近两次审计的
397 组；现直接调用 Phase3 当前构建器的名单并规范为 action route_group，当前 709 组在三个
入口均强制 train-only。验证预检原先统计 eligible 桶，现按全部语义桶统计，含 special_filtered，
与实际 ADE/FDE 分母一致；训练 eligibility 不变。自动 epoch 容量不再写死背景除以 2，
而是逐桶除以共享 EVENT_BALANCE_WEIGHTS 的正整数权重。主线与两个消融首次构建索引
共用 event_balance_common.sh 内的 `.build.lock`，同时检查 manifest 和三个 split 才复用。
源码及有效 split 变化要求新 run；原始 action 索引可继续作为读取来源，由 read_rows 执行开发路线迁移。
full map 自动准备按当前源码/内容生成缓存；旧 checkpoint 仍用原代码。CPU/合成回归为
282 passed / 1 skipped（缺 TensorBoard），未运行真实 Qwen/BEV GPU/DDP 训练。独立均衡短预算
pipeline demo 见 action_expert_ablation/run.md，完整 val/test 仍可能耗时较长。

2026-09-14 主线 pipeline 续训入口修复：`run_full_pipeline.sh --resume 路径` /
`--resume=路径` / `RESUME=路径` 共用 `resume.py` 恢复配置，CLI 优先；进入流程前解析
checkpoint 真实路径，最终 test/probe 固定原 run 的 best.pt，不受 latest 改指影响。
显式 data-root/data-dir/model-dir/lead-bev-ckpt 路径（含环境变量）和标签/full-map 路径
贯穿恢复与最终评测；未显式提供时不把脚本默认值覆盖到旧配置。续训不自动构建索引或重选 LoRA。
操作与搬迁 demo 见 `AutoMoT/qwen3vl_local/action_prior/run.md`；此修复不放宽 checkpoint 合同。

### Action prior UE 规划经验（2026-09-07）

> 历史 v4 记录。以下语言协议已由 2026-09-09 v5 替代。

`action_prior/prompts.py` 的语言协议更新为 v4：按最终接受条件的 YES 查七类 UE 经验表，
`STATIC_OBSTACLE/VULNERABLE/TRAFFIC_LIGHT_ABNORMAL` 对应 UE2/UE4/UE7，其余来自 UE1/3/5/6。
经验摘要依据新 Phase3 的 context/action 定义，但不 import Phase3、不加载其模型或逐帧动作标签。
LoRA 与 dataset-priors（包括噪声后条件）共用 `[PLANNING_EXPERIENCE]`，生成、纯文本复核与
最终 base transcript prefill 均使用同一块；fallback 也保留条件式经验，并发合并以保持 60 词。
保留全部并发正类；缺失不当 NO，Phase2 RE 不覆盖 Phase1 特殊事实；没有特殊正类时按已知
道路/导航正常驾驶，RE 不细分，不加入 RE2/RE3/RE5 经验。空隙、变道方向、恢复阶段均不由
经验表自动确认为事实。新提示词进入执行/语言合同，旧 v3 action 缓存和 checkpoint 不兼容，
需新协议训练；FP32 checkpoint 容器仍 v2。详见 `action_prior/run.md` 的经验映射表。

### Action prior v5 自然场景先验与条件 Flow Matching（2026-09-09，当前）

`action_prior/prompts.py` 不再把 `CONDITION_MEANINGS`、接受条件 JSON、
`PLANNING_EXPERIENCE`、YES/NO/UNKNOWN 或类别词组塞入 Qwen 上下文。它只从最终接受的
RS/EVENT **YES** 选择短的英文自然描述：道路结构、独立 highway 事实及每个当前正事件均可提供
一条可用于近端规划的背景句；NO/null 不被渲染成反例或“正常”断言。罕见多事件并发会压成一条
自然并列句，但不丢任何已接受正类。2026-09-15 起默认将该短段、四张 chronological RGB、当前速度/导航
作为 system/user 图文输入，直接由 base prefill 得到 KV，不执行文字 decode，也不追加 assistant 起始头或回答。
仅 `--generate-analysis` 开启时，base 才输出不超过 80 词的一段总结；复核/fallback 使用同一自然先验，
最终完整 prefill 加入 assistant 摘要。两种模式都保留完整四图与先验/导航提示词。

2026-09-11 入口简化：`run_full_pipeline.sh --dataset-priors --event-balanced` 即可启用均衡课程；
兼容 `EVENT_BALANCED=1` / `--sampling-mode event_balanced`，`--no-event-balanced` 显式关闭。
新训练自动准备 action 索引、缺失的默认 Phase1 索引/标定先验标签，并调用 `prepare_event_balance.py`
自动构建候选与 full map；按原始标注内容、data-root、规则/构建源码与 action split hash 复用私有缓存，
flock 串行构建、临时目录校验成功后原子发布，不依赖手工先建 Phase3 产物，也不训练 Phase1/Phase3。
2026-09-15 发布冲突修复：候选与 full map 共用校验/发布 helper，处理 EEXIST/ENOTEMPTY；
并发目标校验通过且数据文件内容相同才复用，残缺自动缓存改名到 `.invalid-*` 保留后重建。
full map 额外核验实际 candidate SHA256；非目录冲突的 I/O 错误和内容不一致仍失败。
`.prepare.lock` 不删除，进程退出自动释放内核锁；等待时输出日志。显式 full-map 输入仍严格校验。
真实双进程锁竞争、持锁者被 SIGTERM 终止后的恢复、损坏 JSON/缺文件、发布冲突与 I/O 错误均有回归；
准备/均衡/入口三组测试 88 项通过，消融扩展另 14 项通过、2 项因本地缺只读
`leaderboard/team_code/mot_lead_offline_runner.py` 未通过。未复现远端共享文件系统或运行全量数据构建。
CLI data-root/data-dir 贯穿构建与训练；显式标签/full-map 路径保持用户指定，缺失不被替换。
续训沿用原配置和索引，自动准备不参与续训；最终 eval/probe 使用同一 full map。
`run.md` 只保留训练/续训/测试/TensorBoard 常用命令；可选审计移至 `AUDIT.md`，实现合同移至 `DESIGN.md`。

2026-09-10 新增可选 `--sampling-mode event_balanced` 课程：先由
`action_prior/build_event_balance_index.py` 用当前、完整的 Phase3 candidate 和**全部逐帧标注**写
`full_event_mapping.jsonl`。映射将 UE1–UE7、RE2、RE3、RE5 真实 special、确认常规、未确认和
special-but-filtered 分开；只有确认常规进入权重 2 的 `REGULAR_BACKGROUND`，所以 candidate 因动作窗口/
视觉风险过滤而缺席的 UE 不会被伪装为普通。每个 special bucket 权重 1，全局先配齐 `1:…:1:2`
presentation 再切 DDP rank；单帧总重复受 epoch budget 和 `event_balance_max_frame_repeats` 硬限制，
按事件归属压缩的小图使用全局最小费用流：每帧首次使用免费，重复使用费用为 1，
在精确配额/全局 frame cap 下最大化整个 epoch 的唯一帧数；跨桶共享帧由残量网络统一重分配。
同归属组按物理路线轮转展开，跨桶共用帧游标，先用完不同帧再重复；route 多样性是组内顺序偏好，不是路线硬配额。
`sampling/epoch_*.json` v4 记录最优与实际唯一帧数、额外重复呈现数、各桶唯一帧/实际最大重复次数与压缩组数。
七帧抽八次和充足共享帧不得无谓重复；CPU 小图穷举作为独立最优值参照。此次算法改变执行指纹，新代码用于新 run，
旧 checkpoint 仍需原代码恢复；语义未变的 v2 full map 可复用。运行文档与 shell 入口开头提供预检、smoke、
开启/关闭均衡、独立 scene-prior、续训/索引搬迁 demo。固定的全帧
scene context 在 train/val/test 都附加，且 Phase3 规则开发 physical routes 强制 train-only。默认 `uniform`
不读取该 source，保持原始全量 shuffle。索引 manifest/映射合同/内容 hash 进入 checkpoint identity，绝对路径
只作审计；离线文件搬迁可在 eval/resume 用新 `--event-balance-index` 重映射，纯采样闭环不依赖它。
full-map schema 已升级至 v2；旧 v1 因可能将隔离/未解析 R-E2/3/5 误作普通背景而被强制拒绝并必须重建。
参与 best 选择的验证始终全量遍历 val，不受 `val_max_samples` 限制。验证同时报自然分布总体与真实桶 ADE/FDE、桶覆盖；当前 source mapping 没有可靠的“已绕过且待恢复”
帧级状态，因此不生成 UE2→RE2 recovery 专项，保留接口供未来审计证据接入。可用
`best_selection_metric=event_balanced_ade` 按固定权重选 best，但缺任一 validation bucket 会预检失败。

均衡课程默认**不改** Qwen 输入。单独的 `--event-balanced-scene-priors` 只允许
`--dataset-priors --prior-noise 0` 的离线特权实验；它把 explicit context 转为简短自然英文背景，
不渲染 UE/RE code、YES/NO 或动作标签。当前 RE2 仅区分普通导航变道和早先障碍记录；“已通过障碍且
恢复待完成”等待可靠帧级证据接入，不能从 RE2 编号、all-NO 或当前图像不足凭空产生。CARLA/Bench2Drive
没有该 transition/history 标签，因此会明确拒绝这种 checkpoint，不能把它用于闭环或宣称闭环可迁移。

轨迹 decoder 为 `ConditionalFlowMatchingDecoder`。将 route `(B,10,2)` 和 future waypoint
`(B,8,2)` 分别按 30m/20m 缩放后拼成连续变量；训练采样
`eps~N(0,I), t~U(0,1), x_t=(1-t)eps+t*x_gt`，回归向量场 `x_gt-eps` 的 route/waypoint 加权 MSE。
评测与闭环从高斯噪声开始，默认以 10 步 Euler 积分 `x<-x+dt*v_theta(x,t|C)`，再反缩放回既有
route/waypoint 接口；ADE/FDE 来自这个采样轨迹，不能用 GT 加噪重建替代。每个样本的 Qwen KV、BEV
和 Prefix-KV 条件 block 只编码一次，ODE 多步只复用该条件跑轻量的联合 trajectory Transformer 向量场；
每一步的全部当前带噪点可彼此交互，不能退化成逐点独立 MLP。`flow_config`（缩放、时间编码、步数、
trajectory layers/heads）与 `trajectory_decoder=conditional_joint_trajectory_flow_matching_v2` 存进
`action_prior_checkpoint_v4`；验证按 `(scenario, run_id, anchor, seed)` 固定 `eps/t` 与 ODE 初始噪声，
best 按纯噪声 Euler 采样的加权 route/waypoint ADE 选取，FM MSE 仅作诊断。`RS_HIGHWAY` 必须来自独立
确认事实，R3 不得反推高速。旧 Linear+cumsum、逐点 FM checkpoint、缓存和 resume/eval/closed-loop 均明确拒绝。真实模型训练、
采样质量、吞吐和闭环表现仍未在此环境验证。

训练默认只跑 vector-field MSE，不再为每个 micro-batch 额外执行 10 步 ODE；仅显式
`TRAIN_SAMPLED_METRICS=1` 时记录训练采样指标，采样会暂时关闭 trajectory Transformer 的 dropout，
固定噪声不消耗额外 dropout RNG。Bench2Drive 的 `--policy-seed`（默认沿用但独立记录 `--seed`）通过
`ACTION_POLICY_SEED` 为每条 route 创建独立可重放的 FM 噪声 generator，写入 run manifest 和 model contract；
policy seed 优先级为显式 `--policy-seed`、环境 `ACTION_POLICY_SEED`、最后才是 Traffic Manager `--seed`；
后续多 seed 稳定性评测可只改变该 seed。PyTorch 2.3 CPU BF16 的 eval/no_grad Transformer fastpath 在
轨迹块内部回退 FP32；CUDA BF16 没有被这一兼容分支降级。

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

### 2026-09-15 Action 默认直接图文 KV

`action_prior/config.py::DEFAULTS.generate_analysis=False`，CLI 为 `--generate-analysis` /
`--no-generate-analysis`，`train.sh` 与 `run_full_pipeline.sh` 支持 `GENERATE_ANALYSIS=1/0`（CLI 优先）。
`runtime.py::PriorEngine.condition` 默认只获取先验，再用 `prompts.py::PREFILL_SYSTEM_PROMPT` /
`prefill_prompt` 构造 system＋user 四图/自然先验/导航，`add_generation_prompt=False`，一次 base prefill
返回完整 `past_key_values`，位置偏移仍为输入 token 长度加 `rope_deltas`。无摘要生成、复核、fallback，
也没有空 assistant 回答。dataset 模式默认零文字生成；LoRA 模式保留原 Phase1/2 问答。

开启后保留原摘要生成与可选复核/fallback，再完整 prefill 图像、提示词和 assistant 摘要。
`analysis_review` 只在开启摘要时生效；开关、prompt 版本及 `final_cache_content` 进入合同身份，
文本缓存另显式区分开关，冷启动与命中均重新 prefill。逐例审计保留空 analysis 以兼容指标，
用 `analysis_acceptance=disabled` 和 `final_cache_content=inputs_only` 标记默认模式。
训练计划区分实际摘要复核和文字生成预算；dataset 为 0 / 1 / 2 次（关闭 / 摘要 / 摘要＋复核）。

resume、eval/probe 和闭环从保存参数恢复模式，不提供评测时临时切换 KV 模式；分别训练才能比较。
缺字段旧配置按历史摘要语义解释，但不豁免严格源码身份检查，本次代码用于新 run。
开启/关闭与迁移边界见 `AutoMoT/qwen3vl_local/action_prior/run.md` 和 `DESIGN.md`。

同日续训修复：底层 `train.sh --resume 路径` / `--resume=路径` / `RESUME=路径` 也在注入
新训练默认参数之前转入共用 `resume.py`，避免默认关闭摘要覆盖原 run 的开启状态；原 LR、
累积步数、索引与卡数一并恢复，仅显式环境值作为覆盖，CLI 优先。仍严格校验 checkpoint 合同。

### Action prior DDP 梯度布局与日志（2026-09-07）

`leadmot/projectors.py` 的 BEV `pos_embed` 参数仍为 `(1,120,1024)`，前向改用
`squeeze(0)` 二维视图广播，避免 B=1 的 cat 反向将 142-token 步长带回参数。
CPU Gloo 单 rank DDP 回归覆盖 B=1/2、FP32/BF16 autocast、no_sync 累积和清零，
与原公式比较输出/参数梯度；不代表远端真实 GPU 多卡性能已验证。
`action_prior/train.py` 保留旧 invalid_any 统计并增加 domain_only，终端同时显示
unconfirmed/fallback 与字段原因次数；前三项为全 rank 日志窗口样本比例，
invalid_any = domain_only + unconfirmed，正常 domain_inapplicable 不当失败。
全局窗口 loss 与 rank0 心跳 loss 明确区分。执行代码指纹校验未放宽，
旧 action checkpoint 续训/评测需原代码，本次更新用于新 run；运行细节见 action_prior/run.md。

### 2026-09-07 Phase3 审计后动作合同修订

`sft_new_loop_phase3` 使用 `current_wait_first_crossing_v6` 动作规则与 `v5_current_phase` prompt：
当前确认等待优先于未来释放，增速后明显制动的混合窗隔离，横向预测第一次确认跨线。
输入新增最新帧实测速度，未来轨迹仍只用于离线标签。新索引/adapter拒绝旧合同。
按物理路线剥Rep/采集时间分组，旧审计258条路线固定train-only；同RS人工负例单轮同输入最多一次，
自由生成/eval去重并报告实际覆盖。NONE守卫取真实全NO签名；新增纵向precision/recall与独立负例支持检查。
逐帧隔离与新增负例以版本化JSONL为准，不可把场景/Town机器覆盖当全路线人工动作确认。
Phase1/2/action_prior未改；训练只读完整本地Qwen权重，不下载。详见 `sft_new_loop_phase3/REPAIR_20260907.md`。

Phase3本次最终重建14,832条（train/val/test=12,492/528/1,812），物理路线交叉0，79项回归和实际采样检查通过。已尝试pipeline训练入口，本机缺本地Qwen权重而中止，未训练新模型；不能宣称新成功率提升。最终索引位于`AutoMoT/checkpoints/sft_new_loop_phase3_data_v6/`。

### 2026-09-08 Phase3 DDP 验证超时缓解

Phase3 自由生成验证只在 rank0 串行运行，其余 rank 等 NCCL barrier；默认去重前约384题
可能越过原600秒进程组超时。`train.py/train.sh` 新增 `--ddp-timeout-seconds` /
`DDP_TIMEOUT_SECONDS`，默认3600秒，新旧初始化分支均显式传入；整个进程组共用该预算。
新增生成开始/样本边界/完成数/耗时/ETA 与 barrier 日志，`GENERATION_EVAL_LOG_EVERY=1`
可逐题定位，默认10；日志不是后台心跳。采样、prompt、评分和checkpoint guard不变。
这是等待预算与可观测性缓解，不是多卡生成加速或卡死修复；本机仅合成/CPU回归，远端需
跨过一次完整生成验证后确认恢复训练。运行和短验说明见 `sft_new_loop_phase3/SFT_NEW_LOOP_PHASE3_RUN.md`。


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

2026-09-11 新增 `ACTION_OUTPUT_MODE=choice`，只修改 Phase3 的 prompt、target 与严格 parser：纵向有效事件严格在减速、停车等待、增速恢复三选一；机动事件严格在五动作中单选。默认仍是兼容旧 adapter 的 `binary` 逐题 YES/NO；choice 不加入 `NONE`、invalid 或组合动作。候选 high-level 动作词组按 case seed 稳定打乱，模型必须输出选中的完整词组而非 A/B/C。全 NO、invalid、多个动作 YES 的旧多标签行无法无依据地选择一个动作，choice 训练/评测明确剔除并记录原因和数量；choice 写入独立 prompt hash 和 adapter config，必须新训，eval/base/LoRA/audit bundle 强制使用同一 mode。运行示例见 `sft_new_loop_phase3/SFT_NEW_LOOP_PHASE3_RUN.md`。

Phase3 choice 候选补充一句英文动作释义（`prompts.py::CHOICE_ACTION_DESCRIPTIONS`）：
说明减速与 STOP 的优先关系、STOP 包含继续等待、RESUME 不要求此前停车，左右跨线以自车朝向为准。
只渲染所属三/五项，名称和释义一起乱序；阈值与时间窗保持原规则，target/parser 仍仅接受动作名称。
释义由完整渲染自动进入 choice prompt hash，旧 choice adapter 不能与新提示词混用；binary hash 不变。
横向释义明确从最新帧之后预测，规则排除输入历史中的跨线及首次跨线之后的归位；
运行文档提供带释义的三选一示例，输出仍仅为动作名称。


### 2026-09-14 Phase3 binary/choice 全错例 RGB 审计

20260911_174046 包：binary production 373/552=67.57%，choice在306个单动作题上241/306=78.76%；同题binary203/306。choice不评NONE/INVALID/联合动作，不当完整任务替代品。
本次逐帧复核202题、122个run、2567个不同主审计帧，覆盖binary179及choice65个production错例的193题并集，另9题正确对照；不是全数据随机噪声调查，自动源规则命中不得记作人工确认。
当前默认索引v9、split seed20260914、prompt v8_shared_temporal_rules；STOP两帧≤0.5m/s都须在1.5s内，普通速度变化窗口2s、首次越线窗口3s，+3.25s仅确认末端越线。数值动作规则仍v7，没有为模型答案调阈值。
collector的R4恢复逐帧要求局部路口空间证据；Phase3只撤回有明确R1来源的stable_meta_light_with_untrusted_xodr弱恢复，保留独立事件。DynamicObjectCrossing hazard-only切入标待审，不整类改NO；精确RGB排除两条U-E3帧段、隔离两条局部RS帧段及一处lane_id/视觉跨线未确认转移。
原collection与原audit bundle不回写；精确修订通过Phase3映射层，Phase1/2既有权重不会自动更正。所有已暴露test的191个物理路线组加入train-only开发集合，累计709组；新holdout不得复用。
v9全源重建train/val/test=13500/348/468，14316行及3272run原meta回读无速度/有效动作不一致、物理路线无跨split；139项回归通过。新prompt/mapping与旧adapter/索引不兼容，新模型尚未训练，CPU索引验证不代表新模型提升。
运行使用CASES_PER_BIN=0完整评测；详见AutoMoT/qwen3vl_local/sft_new_loop_phase3/EVAL_REVIEW_20260914.md、AUDIT_COMPARISON_20260914.md及SFT_NEW_LOOP_PHASE3_RUN.md。代码、精确修订、轻量手写笔记/文档可追踪，probe_output RGB/HTML与checkpoints索引审计大产物不入库。

本轮全源审计覆盖42个scenario、7241个run、914466帧；弱R4恢复撤回13459帧为自动规则结果，仅定向RGB样本及4个额外四帧对照被查看，不能外推为13459个人工确认错误。U-E3待审912帧保留候选，精确排除32帧单列。RS精确隔离32帧，完整决定绑定run/frame。
新mapping hash=d90ef1eca13fe31987690ad13f0a8cf0cb994a0256c84da118ba7dec6a1ca62d。索引独立题数13387/348/468，train重复呈现113、最大input复用7；3272run对应3271物理组。相同RS错误事件测试仅2条独立路线，拒绝泛化证据不足。

### 2026-09-15 Phase3 INVALID 验证预算修复

Phase3 先规划同 RS 人工负例/RS/问题域覆盖，再分配均衡 source 余数；只有明确预算不足才用
`InvalidQuotaError` 触发 loss/generation 各自自动增容，保持十类与 INVALID 的 10:2 呈现比例。
缺数据/签名错误仍失败，同 RS 人工输入不重复；实际请求/有效预算、呈现与独立题数进入
`validation_sampling`。`AUTO_EVAL_BALANCE_COUNT=0` 可关闭增容；`train.py --sampling-only`
复用正式 CPU 采样预检，不加载图像/权重、不初始化 NCCL 或写运行目录，直接 Python 默认索引已对齐 v9。
细节见 `AutoMoT/qwen3vl_local/sft_new_loop_phase3/SFT_NEW_LOOP_PHASE3_RUN.md`；本次未验证远端真实训练。

### 2026-09-16 Phase3 逐帧续审与覆盖预检

Phase3 当前新训练默认索引 v10、split seed20260916；prompt v8_shared_temporal_rules 和动作规则 v7 不变。
79 个定向 RGB 窗口（60 个 run）及 8 条冻结 prompt 后的新负例候选已逐帧审阅；6 条接受、2 条证据不足拒绝。
新同 RS 错事件负例仅覆盖 R3 的两个事件，val 2/test 4 个独立物理路线，不代表全域拒绝能力。
binary preflight 在模型/NCCL 前要求 val/test 各至少 2 个同 RS 负例物理路线，生成验证实际采样再检查；
Rep/采集时间不增加独立支持，choice 豁免该项。eval 默认全量 CASES_PER_BIN=0，2RGB 证据图只标实际两张输入。
本轮暴露的 176 个 test 物理组后续 train-only；精确 U-E3 撤回仅限核验的121–124窗口，真实源只有124是正例且默认风险过滤已排除，不造NO。
新 mapping 必须重建索引，旧 adapter/run 要原源码与原合同；本轮未全量重建或训练新模型。
证据、边界及运行见 `AutoMoT/qwen3vl_local/sft_new_loop_phase3/EVAL_REVIEW_20260916.md` 和同目录运行文档。

验证：Phase3 186 项回归通过；1 条真实 collection route 双模式 source smoke、6 条真实负例原图指纹/meta 加载通过。79 窗口是定向审计而非全量错例覆盖或随机噪声率调查。两图/四图各有得失，旧/新 holdout 不同，当前无证据认定回滚版最优。

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

### 2026-10-18 v84 标定修复与关键格曝光发布

用户授权完善/白名单push及本地RGB；发布教师v23/v24已验证否决保留、plan39/registry34与诊断，不修改旧合同或阈值提示词。Phase4采样v5在每事件1024内对已有原生自动train/readiness关键边YES/NO各预留至少1不同题，再填连续低曝光段；缺格只报、history绑定拒绝旧v4混用。P3无minimum仍v4算法，但共享源码SHA变化，旧checkpoint保留原版，不安排重训。P3测试夹具补齐tokenizer/device，非生产改动。
本轮再原生回放3路线266帧/798源SHA，两模式各130train与前v24逐字段一致，无val/test；实际复看27既有RGB（1176_0 71–79、4591_0 54–62、1378_0 71–79），非新增独立盲审。1457回归通过2旧池缺失跳过，另127入口/采样/分布式通过有重叠。57路线旧3187题两模式CPU各事件1024模拟，缺格不外推全池。新label_release_check只读历史manifest/train SHA及版本差异、CPU采样，自动<=30MB ZIP；本机真实CLI通过，571源码匹配，不授予训练准入。
当前服务器只cr02按LABEL_SAMPLING_RELEASE_20261018.md同步后检查既有phase4_labels_20261017_125230_304771113/data4，交label_release_report.zip及label_release_package.json；无RGB/模型/缓存/回放/训练。cee已收两模式最终结果，不重跑phase4_continue或RGB2/4。UE5正例仍0，新正式池迁移未验收；禁止修改旧SHA越过合同。不上传数据/RGB/权重，保留用户.vscode等其它任务。

### 2026-10-27 Phase1/2 独立新训练入口

用户要求改标定/提示词后必须重新训练再评测。入口 `AutoMoT/qwen3vl_local/audit_joint/phase12_retrain/run.sh all 1` 或 `all 2`；两台服务器各自在 AutoMoT 根运行各自阶段。Phase1只接入5条已逐帧复核train修正，RS2/RS4说明在独立训练快照内合并，val/test字节不变；Phase2沿现有v5独立新训练。保留原采样和四卡三轮9216/5376预算，两图/四图分别从基座训练。新旧源码/数据/权重隔离，CPU准备与失败日志先于GPU，训练后原生评测交包。新训练效果仍待远程实测，不把CPU或旧权重换提示词当新模型验收。详细边界见该目录README.md。

### 2026-10-30 Phase2 错例根因审计与证据修订（用户要求先微调问题场景）

用户要求结合错例和本地逐帧 RGB 判断是标定、提示词、数据量还是遮挡等客观原因，并先微调有问题的场景、不把好的改坏。本轮逐例看完 cee 新训练 81 个生产错例的四帧原始 RGB，并回放 LEAD metas/bboxes 客观量；结论与处理记录在 `AutoMoT/qwen3vl_local/audit_joint/phase12_retrain/EVIDENCE_20261030.md` 与 `sft_new_loop_phase2/HIGHWAY_UE3_OBJECTIVE_SPANS_20261030.md`。
- 高速切入 0/8 是数据问题：v1 训练正例只有 Town13 两条左侧匝道路线 28 帧，客观上车辆始终停留在相邻车道；测试是 Town12 右侧跨线。新增 `highway_ue3_objective_spans.py` 用切入车横向位置/偏航角客观给出窗口，生成 `highway_ue3_rgb_decisions_v2.jsonl`（训练 50 条路线 573 帧、验证 5 条 56 帧，测试三行与 v1 逐字相同；Town13 两条 v1 YES 改为 NO 对照），抽查 14 条路线 RGB 无误报。建库支持 YES 行 `true_rs_override: R3`，沿用 10-27 的 23 帧修正口径。
- UE1 漏检主要是标定窗口过宽：训练池 67% 的 UE1 正例是停稳 >2 s 的静止队列、9% 是匀速无刹车前车；模型把静止队列答对、真正刹车的帧反而漏检；其它场景中前车明显刹车却标 NO 的矛盾负例 221 行。UE6 正例 33% 违规车缺席/车后/已远离；真实漏检集中在夜雨中应急车横穿。新增 `phase12_retrain/phase2_evidence.py` 只对 train 行做客观排除（不翻转标签、val/test 逐字节不变、六类桶必须齐全），排除行写 `evidence_exclusions.jsonl`；`--audit-holdouts` 只读列出 holdout 争议帧。
- 提示词 v7（`direct_event_visual_v7_evidence_20261030`）在 v6 上补 UE1 尾灯/起步/队列边界、UE6 应急车与无违规车边界、UE3 左右两侧同判；UE5/问题域/INVALID/解析/采样/预算不变。入口 `run.sh all 2 --revision evidence_20261030 --budget-profile phase2_10k`，父库另建 `checkpoints/sft_new_loop_phase2_data_evidence_20261030`。
- 必须预先说明：现有测试集与证据口径矛盾，每模式 39 个 UE1 静止队列正例和 25 个 UE6 无可见违规车正例当前靠场景记忆答对，新口径训练后最坏可能 −64/384；新权重须同时报告原评分和剔除争议帧后的评分，测试标签修正单列不改主评分。本机 114 项相关 CPU 回归通过，v2 文件重跑逐字节一致；codon 真实 CPU 准入通过（`p2_evidence_cpu_20261030`）：v2 建库 650 高速帧全匹配、test 794 行逐字节同成员、val 成员变化，证据排除 2127 行后 UE1 327/UE6 857/RE 1051，两模式六轮采样投影通过。未训练、未 GPU、未改默认模型。
同日开训复核后补修（提交见 Git）：val 与 train 同策略过滤作为选优视图（原 val 另存、test 不变、争议帧冻结）；UE1“前车久停而 ego 仍行驶”仅在 ego 两秒内降速 ≥2 m/s 且 ≤20 m 时保留；UE6 提示词删去“停车等待即优先权”，要求独立信号/优先权证据并加正常让行/红灯/应急车正常通行的 NO 保护；meta/bbox 缺失直接阻断并绑定来源 SHA；复用父库校验 v2 决策 SHA（目录名带 SHA 前缀）；训练后自动输出 `dual_score.json`（原评分 + 剔除冻结争议帧评分）。口径更正：phase2_10k 为完整六轮；v2 父库 val 成员已变，只有 test 逐字节不变。

2026-10-30 二次开训复核收尾：Phase2 evidence_policy_v3 要求实际读取的非负历史 meta/bbox 完整，缺失拒绝，避免队列被误判为刚停；双评分只读取两模式原生 eval_review/lora_production 并核验完成回执、分片及题数，排除打包副本但保留合法抽样重复。相关五模块 85 passed/1 不适用 skipped，真实旧输出复算仍为 316/384、324/384。最终源码须新建 CPU run，旧 ready 不重写；没有新权重或 GPU 验收。此前“−64/384 最坏退步”不是经过验证的上限；当前 test 原始 798 行、原生 384 题，val 为单独过滤选优视图。详见 phase12_retrain/EVIDENCE_20261030.md 第九节。
