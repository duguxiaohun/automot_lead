# AGENTS.md

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



### 2026-10-01 Phase4 v19 环岛目标与十八/十九轮风险接入

RE5目标走廊/方向进入状态对；显式roundabout上下文要求导航证据与完成边界，complete增加route_exit_reached并独立保留STOP。
仍用YIELD/PROCEED/DONE及re_yield；catchup不能替代出口证据。回执/回放真值绑定route_context_id，改目标须新实例。
route_prompts/route_calibration为实际入口；非路线题委托冻结旧规则，原四文件SHA不变，旧holdout协议不批准环岛扩展。
登记两轮22风险/21路线、44定位RGB核SHA、三个noScenarios/Town03曝光组train-only；无新标签。
本次复看14连续面板/3既有序列309RGB及6原尺寸，不增加覆盖；继承460条56872帧、192/278格，余240条41796帧及18格源不足。
485专项通过，2/4RGB实际check及各1022监督+405待审RGB通过，四JSONL与第十七轮逐字节一致。
world1/4七轮事件1:1/cap8，第3轮全覆盖563；历史恢复一致。原采样/训练/标注/冻结资产不变，Phase3/Action默认不改。
合同v19/data_v19/快照v9，须新候选/数据/run；缺366转移格/12分段格/新增9环岛格，三关键正例仍0、六事件val-test缺。
formal_data_ready=false，无全量可靠条件生产或正式4B/GPU/DDP/CARLA验收。见Phase4/ROUTE_V19_20261001.md及route_v19_verification_20261001.json。

### 2026-10-01 Phase4 第十七轮曝光与风险登记

新增隔离Town03_route_001041/001040两个NonSignalizedJunctionLeftTurnEnterFlow物理组为train-only；
001041原默认val，重建候选实际移到train，001040原train也显式隔离，两组不属冻结holdout且未入现有题库。
登记20处视觉不连续/14路线，40条相邻RGB证据仅核SHA；保留127→128纠正，不登记三处未确认候选。
任务名v18保持，新增资产及构建SHA须新候选清单/数据/run；旧产物原源码，不改hash复用。
462项专项通过；2/4RGB实际check及各1022监督+405待审RGB通过，四JSONL与第十六轮逐字节一致。
七轮world1/4题序和累计历史不变，第3轮覆盖563；候选路线train8166/val765/test781，总9712/1637680帧不变。
采样/训练/控制器/提示词/标定/人工标签/冻结holdout已核SHA未变，Phase3/Action稳定默认未改。
无新增目视/标签；继承审计432条/52344帧、180/278格三ID，余268条/46324帧，18格源不足。
全量可靠条件生产及处置核算、关键正例、六事件独立评估仍缺，formal_data_ready=false；无正式4B/GPU/DDP/CARLA验收。
详见Phase4/README.md、seventeenth_audit_exposure_20261001.json及seventeenth_audit_registration_verification.json。

### 2026-09-30 Phase4 第十六轮风险登记接入

登记9处车辆突变、1条画面倾斜及2条骑行姿态/身份风险，共12记录/11路线；29条定位RGB仅核SHA，无新增目视/标签。
12审阅组均原train-only，合并后曝光集合不变；复用逐帧risk_review，同路线多风险必须完整覆盖全部因果历史。
自然出画不登记为突变，“仍可见但可能不阻挡”保留复审候选，不自动生成YES。
任务名保留v18/data_v18，风险资产及构建源码SHA改变须新候选清单/数据/run，旧产物原源码，不改hash复用。
采样/训练/控制器/提示词/标定/人工标签/冻结holdout已核SHA不变，Phase3/Action稳定默认未改。
457项专项通过；2/4RGB实际check、各1022监督+405待审RGB通过；四JSONL与第十五轮逐字节一致。
单进程/四进程七轮题序及历史一致，第3轮覆盖563题；无正式4B/GPU/DDP/CARLA验收。
继承审计415条/49929帧、172/278格三ID、余285条/48739帧、18格源不足；全量标签及关键正例/独立评估缺口保持。
详见Phase4/README.md、sixteenth_audit_exposure_20260930.json及sixteenth_audit_registration_verification.json。

### 2026-09-30 Phase4 第十五轮默认七轮覆盖与恢复采样历史

修复event_equal事件内每轮仅移一位造成七轮漏选：按实际累计次数优先，再按最久未选，再固定分层顺序补选。
帧内重复同样看实际次数；跳过题不记曝光，保留事件1:1/共享帧cap8/不可行配额诊断。
checkpoint保存采样历史与池/seed/cap/budget/world/next_epoch绑定，恢复历史/计划等于连续训练；缺失错配拒绝。
预检新增默认七轮world1/world4累计覆盖及缺题ID，实际每轮审计同样记录；容量约束下不承诺任意池七轮必全覆盖。
旧池单进程七轮413/563、world4为416/563；修复两RGB均单进程401→540→563、world4为404→543→563，第3轮全覆盖。
第十五轮10处消失/6路线接入逐帧risk_review，20条定位RGB仅核SHA；无新标签/目视，12组原train-only。
合同v18/data_v18，需新候选清单/数据/run；标注v7、提示词/状态/标定/split/事件宏平均保持，Phase3/Action稳定默认未改。
450项专项通过；2/4RGB实际check、各1022题+405待审RGB、14默认等额计划及563轮budget10/cap1全覆盖通过。
四JSONL与v17逐字节一致；全量可靠因果标注/关键正例/六事件val-test仍缺，formal_data_ready=false。
继承审计403条/48357帧、169/278格三ID、余297条/50311帧、18格源不足；无正式4B/GPU/DDP/CARLA验收。
详见Phase4/README.md及fifteenth_audit_fixes_v18_verification.json。

### 2026-09-30 Phase4 第十四轮事件等额、选优与全目录清点

默认event_equal：十事件同配额，事件内转移/答案/来源/路线轮转；共享帧容量流联合分配，全epoch cap8含重复。
预算须整除lcm(事件数,world_size)，不足报错给事件缺额；旧legacy_ring仅显式对照。默认单卡57/事件，world4为58/事件。
验证先逐事件准确率再事件宏平均；缺事件时固定十事件分数null，已观测宏平均单列，selector拒绝旧分组宏分数。
默认pipeline先扫描冻结候选物理路线split，再构建/加载核对候选身份；实扫9712路线/1637680帧，非目视或自动标注。
生产覆盖单列1074帧有题/782帧监督，full causal production=false；可靠条件生产者仍缺，不能称全量标签生产已接通。
第十四轮15处消失/11路线接入逐帧risk_review，30条定位图只核SHA；无新标签/目视审阅，12开发组均原train-only。
合同v17/data_v17，须新数据新run；标注v7/提示词/状态/标定/冻结holdout保持，Phase3/Action稳定默认未改。
435项专项通过；2/4RGB实际check、各1022题+405待审RGB、14等额计划及563轮budget10/cap1全训练题覆盖通过。
四JSONL与v16逐字节一致；trainable=true仅有限子集，formal_data_ready=false，仍缺366转移格/12分段格/六事件val-test。
继承审计391条/46070帧、166/278格三ID、余309条/52598帧、18格源不足；无正式4B/GPU/DDP/CARLA验收。
详见Phase4/README.md及fourteenth_audit_fixes_v17_verification.json。采样/指标改动尚无GPU效果结论。

### 2026-09-30 Phase4 第十二/十三轮恢复训练与风险登记修复

每个 checkpoint 保存当时最佳 adapter 路径/分数/epoch/推理资产 SHA，恢复不读父 run 后来的 best.json；
验证历史边界、分数、selection、合同及资产，拒绝缺失/替换；搬迁须保留相对历史引用，不自动兼容旧 run。
两轮30处消失（29路线）及3条ControlLoss身份待审接入逐帧风险门槛，补登记Town02_route_001609 train-only；
无新人工标签/目视审阅；66条定位RGB证据只核SHA。新增readiness YES/NO明细及缺正/负例清单。
合同v16/data_v16、标注v7；提示词/状态/标定/冻结holdout不变，Phase3/Action稳定默认未改。
414项专项通过；2/4RGB实际check、各1022题及405待审RGB、14整池计划及563轮cap1全覆盖通过。
四JSONL与v15逐字节一致，train563/val273/test186；保留有限覆盖训练准入trainable=true，complete_coverage=false。
仍缺366转移格/12分段格；UE2 return/RE3 enter/UE7 proceed readiness YES均0，不能称完整闭环已学会。
继承审计379条/44204帧、165/278格三ID、余321条，18格源不足；无正式4B/GPU/DDP/CARLA验收。
详见Phase4/TWELFTH_THIRTEENTH_AUDIT_FIXES_20260930.md及audit_fixes_v16_verification.json。

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


### 2026-09-29 旧 BEV-only 评估兼容（补充诊断入口）

训练机报告原合同/三split一致、真实EMA旧RoPE探针通过。eval/compare新增精确审查的兼容
路径：仅六个Qwen迁移源码SHA对、指定transformers版本对及本轮eval入口SHA对可接受，
恢复旧完整decoder配置；其余资产/条件/源码仍严格核对，未知改动拒绝。报告记录双合同身份。
不修改checkpoint、不放宽训练resume、不替换旧Qwen基座。专项31通过，完整GPU尚未验收。
细节与相关检查限制见PROJECT_CONTEXT同日条目及action_prior/CHECKPOINT_COMPARISON.md。

### 2026-09-29 旧 BEV-only 兼容性诊断入口

新增只读 `action_prior/audit_checkpoint_compatibility.py`，输出源码/依赖/合同差异及可选
EMA decoder CPU 探针；旧完整配置可测试恢复原 RoPE，不修改 checkpoint 或放宽正式合同。
14项专项通过，含迁移前真实源码的小网络数值对照；未训练机真实权重/完整RGB-LiDAR/GPU验收。
报告不是评估放行凭据。命令与验证边界见 `action_prior/CHECKPOINT_COMPARISON.md`
及 `PROJECT_CONTEXT.md` 同日条目。

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

### 2026-09-28 Qwen3.5-4B 本地迁移（新训默认覆盖旧基座约定）

用户明确授权当前 Qwen 使用链路整体迁移至 Qwen3.5-4B，并复制所需上游源码到本地修改。
新增授权目录 `AutoMoT/qwen3vl_local/qwen35/`（代码、vendor/许可证/来源哈希、测试、说明）。
当前运行入口默认本地 `checkpoints/Qwen3.5-4B`，模型/处理器用本地副本，禁止运行时下载、
remote-code 或远程媒体；通用框架依赖固定并在独立环境验证，不改全局 site-packages。
此模型升级为新 run，不是 v23 数据语义晋升；冻结 Phase3 releases 和旧审计/checkpoint 不改写。
旧 LoRA/规划 checkpoint 须原源码与原基座；新 adapter 绑定 backend/模板/基础资产合同。
Qwen3.5 hybrid cache、8 个 full-attention 层、4×256 K/V、partial interleaved RoPE、
非思考答案边界和 DeltaNet LoRA 均须一起迁移，不得只替换模型目录。
本机缺完整 Qwen3.5 权重及外部 mot_lead_offline_runner.py，未完整 GPU/runner 验收。
详见 `AutoMoT/qwen3vl_local/qwen35/README.md` 与 PROJECT_CONTEXT 同日迁移条目。


### 2026-09-28 Action pending mkdir EEXIST 恢复

准备器补齐mkdir EEXIST/ESTALE的锁内有限重试，每次重查ready/合同/hash；完整续发、残缺隔离。
124项相关CPU回归通过（新增18项），未训练机真实挂载/GPU验收；底层触发原因不能仅凭日志确定。
仅改外层prepare_event_balance.py，v23_io1快照/语义/mapping/缓存身份不变；不删除锁、不放宽旧run合同。
详见PROJECT_CONTEXT及action_prior/run.md的2026-09-28条目。

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


### 2026-09-26 Phase3 binary 跨机器容量差异与来源内回流

训练机诊断为七个DYNAMIC_CUTIN自动负例细分组共用3帧：54次需求、cap8容量24，缺30。
新增训练专用invalid_capacity，仅固定细分配额失败后按实际池联合重分配；不写死帧数/缺额。
保留正例各1024、INVALID各来源总额、全epoch cap8、具体人工题和原非空细分组至少一次覆盖；
只同来源转移次数，最少偏离细分目标；随后复用原采样器/游标/历史。仍不可行则诊断报错。
113项相关CPU回归通过：2/3/4/8帧及0/52/53人工题四种合成差异各7轮，每轮12288次，
另含随机小池穷举、正负例/人工共帧和成功路径不调用回流。未训练机完整池/GPU验收。
回流版本与源码哈希写入训练采样配置；未改共享sampling/build/mapping/prompt，匹配索引可复用，
旧run原源码；各服务器须sampling-only预检，不保证缺数据/真实容量不足也能开训。
详见Phase3运行说明及PROJECT_CONTEXT.md。本条覆盖此前“仅诊断、等待报告”的处理阶段。

### 2026-09-26 Phase3 固定配额容量诊断（不放宽训练约束）

用户要求保留每类1024与单帧cap8，只排查重分配。train失败分支新增capacity_diagnostic，
精确最大流/最小割报告受限分组、物理帧、可用次数和预留人工题；原ValueError子类兼容。
INVALID仅来源配额的容量上界单列，未保留细签名覆盖的上界不能当可执行计划，不自动采用。
66项相关CPU回归通过（使用已有pvi环境）；同12236/12206数值为合成拓扑，未复现训练机实际池。
原sampler/build/mapping/prompt未改，现有匹配索引可直接CPU sampling-only诊断；不绕过旧索引校验。
远端须贴phase3-capacity报告后才能确认合法重分配空间；详见Phase3运行说明及PROJECT_CONTEXT.md。

### 2026-09-26 Action / Phase3 数据发布 ESTALE 恢复

BEV-only和qwen_simple真实shell入口确认共用prepare_event_balance，候选/full map/actions完成缓存
带ready.json与哈希保留，持锁重跑复核后续发；未完成/损坏目录隔离重建，不删除锁。
Action三split/full map/prior labels及Phase3候选、索引、训练池、并行扫描和元信息共用
filesystem.py有限ESTALE重试，核验rename已提交但报错的目标；不承诺原始输入/训练权重IO免故障。
203项CPU回归通过；torch相关扩展检查受当前环境缺依赖限制，未真实挂载/GPU验收。
构建与helper源码绑定mapping哈希，需同步完整相关改动、重建索引/full map并新run，旧run原源码。
同版本完成缓存才可免扫描续发，旧版临时结果可能已清理；持续挂载故障需训练机恢复存储服务。
细节见action_prior/run.md、Phase3运行说明及PROJECT_CONTEXT.md同日条目。

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
- `AutoMoT/qwen3vl_local/qwen35/`（按用户要求本地化 Qwen3.5-4B 新增的目录白名单；
  允许修改、追踪、commit 和在用户授权 push 后推送本地源码、`vendor/` 上游源码及许可证/来源哈希、
  官方模板参考、测试、依赖清单、README 和 `.gitignore`。不包含模型权重、下载 wheel、
  `__pycache__`、pytest 缓存、日志或训练/评估产物；`AutoMoT/checkpoints/Qwen3.5-4B/` 不入库。）
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
- Qwen3.5 本地实现的 Git 白名单为 `AutoMoT/qwen3vl_local/qwen35/` 中上述源码与轻量配套文件，
  包括新增且尚未追踪的文件；push 前须精确 add 并审核，不能只提交已追踪文件而漏掉此包。
  模型权重目录 `AutoMoT/checkpoints/Qwen3.5-4B/` 不在 push 白名单内。
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
本次逐帧复核202题、122个run、2567个不同主审计帧，覆盖binary179及choice65个production错例的193题并集，另9题正确对照；不是全数据随机噪声调查，自动源规则命中不得记作人工确认。
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
