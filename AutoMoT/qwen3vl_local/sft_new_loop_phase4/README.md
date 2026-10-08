一行运行（在 `AutoMoT` 目录）：

```bash
bash qwen3vl_local/sft_new_loop_phase4/run.sh
```

自动从全部合格路线准备配对题库，依次运行四图和两图；每事件每轮抽样1024次（类似Phase3），不再把小事件重复到最大事件规模。无需手填路径或参数。见 [简明说明](QUICKSTART.md)。

[当前 v40：全路线生产与 Phase3 式有限采样](PHASE3_STYLE_SAMPLING_V40.md)

[v38 一条命令完成全量建库、两图／四图训练、测试与打包](PIPELINE_V38.md)

[v37全量训练与两图/四图运行命令](FULL_TRAINING_V37.md)

[v36稳定选题、配对训练与异常起点验收](PAIRED_TRAINING_V36.md)

[v35局部控制、弯道骑行与采样准入验收](LOCAL_SCOPE_AND_PREFLIGHT_V35.md)

# Phase4：条件成立即推进

### 2026-10-06 Phase4 v34 参数缩写、主机预检退出与依赖重建

train/preflight/launch禁用参数缩写，shell提前拒绝路径参数全部前缀；pipeline的host-preflight成功后退出，不读best或评估旧adapter。
合同错误列出具体源码/依赖及新旧SHA；保留当前Phase3开发组隔离修改，按最终依赖重建v34，不修改v33 manifest。
新增13实际已看训练组、11精确不连续窗口，未看val组不曝光。新窗口覆盖完整人工因果包络含图间隔；旧整路线审核约束保持，卡/编译/加载范围一致。
新增风险逐题检查：每模式相关18训练/3待审题与11窗口重叠0，没有据源异常撤回标签。本轮2训练路线4原图复核，无新完整序列/标签/独立RGB。
全8614路线/1062401帧重放，train7268/val658/test688保持；2/4训练75373/75373，原443人工及val273/test186/review539保持。
878专项通过，2/4真实pipeline、全部RGB、world1/4七轮通过；最终默认2/4启动再验通过，验收时间/SHA保存。默认MODEL_DIR仍缺4B，host-preflight明确失败，未GPU/CARLA训练。
默认每轮400，七轮覆盖705（0.9353%）；关键边/六事件评估/八事件老师缺口保持；严格批准0，盲审961卡未填，formal_data_ready=false。
继承筛查203/278格三ID、余215条及18格源不足，非逐题认证。合同v34/data_v34、老师规则v7保持、预约v14，新产物AutoMoT/checkpoints/phase4_v34_full。
见[LAUNCH_AND_RISK_V34.md](LAUNCH_AND_RISK_V34.md)及thirty_fourth_audit_verification_20261006.json；final_source_acceptance只证明记录时点，后续依赖修改须重新验收。

### 2026-10-06 Phase4 v33 分支覆盖、配对评估与启动检查

逐分支/返回策略/边/阶段/答案/split计数，零样本分支显式列出；补问题不填readiness缺额，新增补标任务清单与七轮覆盖比例，无新增准入硬门。
evaluate输出绑定完整因果输入和参考的paired_identity；paired_eval拒绝漏题/重复/换图/换答案，人工准确率与老师一致率分报。
启动默认使用v33全量弱库；缺模型/数据在GPU前明确报错，路径覆盖不能绕过预检；构建区分审核题模式与全量老师模式。
全8614路线/1062401帧重放，物理train7268/val658/test688保持；2/4训练75373/75373题，原443人工及val273/test186/review539保持。
850专项通过，2/4真实pipeline check、全部RGB核验及world1/4七轮通过；默认每轮400，七轮覆盖705，比例0.9353%。
12既有训练曝光组及7确认消失窗口接入，撤回/未确认猜测不升级。本轮3训练路线8帧/3缩放联系表，无新完整序列/人工标签/独立RGB。
继承筛查199/278格，剩228选定序列；筛查不作标签认证。关键绕行/进入/返回和六事件独立评估仍缺，八事件老师仍缺。
新盲审961卡/56路线未填，严格批准0；弱trainable=true、formal_data_ready=false。默认模型仍缺，Action/CARLA在线提供者未接通，未GPU训练/CARLA验收。
合同v33/data_v33、老师规则v7未改、预约v13；新产物AutoMoT/checkpoints/phase4_v33_full，v32保留。见[AUDIT_COVERAGE_V33.md](AUDIT_COVERAGE_V33.md)及thirty_third_audit_verification_20261006.json。

### 2026-10-05 Phase4 v32 过期等待截止与静止正例诊断

UE1在已观察停车后持续实际前向运动而仍YIELD时截止旧实例，不造YES/complete，不放宽运行时执行许可；前车连续可见超30m范围同样截止。
全8614路线/1062401帧重放，train7268/val658/test688保持；2/4弱库训练75373/75373题，原443人工及val273/test186/review539保持。
已定位VehicleTurningRoute错误静止YES实际19道，均清除；其他旧静止YES18道逐项保留；821 f167–175旧proceed清零。
预检按事件/边列静止readiness YES缺额，仅报警；cyclist_follow提示明确允许继续跟随，不要求骑车人消失，不授权超车。
新增至少500帧全程静止的元数据证据过滤；具名BlockedIntersection四路实际起末均行驶，保留并记录反证，本轮无新剔除。
833专项通过，2/4真实pipeline check、全部RGB核验、world1/4七轮通过；默认每轮400，七轮覆盖705，八事件老师仍缺。
弱试训trainable=true，formal_data_ready=false，严格批准0。新盲审961卡/56路线全部待审，无独立准确率。
本轮仅两训练路线10RGB关键帧联系表复核，无新增完整序列/人工标签/独立RGB/GPU训练；八个具名曝光组及两条用户报告行人消失窗口登记。
合同v32/data_v32、老师v7、预约v12；新产物AutoMoT/checkpoints/phase4_v32_full，v31保留。见[TEACHER_LIFECYCLE_V32.md](TEACHER_LIFECYCLE_V32.md)及thirty_second_audit_verification_20261005.json。

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
合同v31/data_v31、老师v6、预约v11，新产物AutoMoT/checkpoints/phase4_v31_full，v30保留；见[TEACHER_SCOPE_V31.md](TEACHER_SCOPE_V31.md)及thirty_first_audit_verification_20261005.json。

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
产物AutoMoT/checkpoints/phase4_v30_full；详见[FULL_WEAK_TEACHER_V30.md](FULL_WEAK_TEACHER_V30.md)及thirtieth_audit_verification_20261005.json。

## 2026-10-05 v29：弱监督实验已接通（当前）

严格批准保留，新增显式弱监督train-only入口，区分未认证规则标签与人工/批准标签。实际新增73道弱监督（18YES/55NO），训练443→516题；原443题、val273/test186/review539保持。
实例先有本事件限制才可创建，排除3_26首帧YES。新增因果STOP台账与横穿检查；提前十几米等行人不算履行停车义务。STOP区域代理误差仍待审核。
新增事件内边×答案等额采样与容量缺额报告；弱库默认使用。**当前可执行预算为每轮100题，启动加 `--epoch-samples 100`；默认整池520不可行。**
782专项、2/4真实pipeline check、975监督+539待审RGB核验通过；world1/4七轮357/516题覆盖，不能称七轮全覆盖。独立109路线重放生成963卡，全部未填，未目视独立RGB。
合同v29/data_v29、老师v4；严格批准仍0，弱实验trainable=true、formal_data_ready=false。八类老师和独立准确率未补齐，未GPU/4B/DDP/CARLA验收。
[策略、实测、数据路径与可执行命令](WEAK_TEACHER_V29.md) · [验证记录](twenty_ninth_audit_verification_20261005.json)。

## 2026-10-05 v28：实例局部异常与可完成的释放回放（历史）

修复无关目标消失终止全部实例、自车起步抹掉 readiness YES、跟车恢复无法完成；异常窗口后允许重新建立同 ID 实例。扩大近距实例召回，路口附近检查本车道控制，暗图和决定性异常继续弃答。
原17开发路线每模式18YES/55NO/125UNKNOWN，完成1实例；扩展23路线2598帧为60YES/264NO/697UNKNOWN，其中proceed YES41。1030_0 f72–73推进、f75恢复跟车、f76完成。
按实例存在筛选新增69独立物理组，UE1/UE4各40条，连同原40共109路线；2图盲审包开发240卡、独立1485卡，未填写、未批准。4图不自动继承2图批准。
766专项及2/4图真实pipeline check通过；902监督+539待审RGB核验，训练仍443题/45路线，四JSONL与v27相同。实看3既有训练路线8原图，无新完整序列/人工标签。
合同v28/data_v28、老师v3、抽检v8、真实批准表teacher_registry_v3为空。有限trainable=true，formal_data_ready=false；其余八类老师和独立准确率仍缺。
[修改、实测及审核包](TEACHER_RELEASE_V28.md) · [验证记录](twenty_eighth_audit_verification_20261005.json)。

## 2026-10-04 v27：运动证据与盲审材料（历史）

修复倒退/侧移误认前向动作；同ID完成后可在明确解冲突后重建实例。新增分层抽样、隐藏老师答案的RGB审核包、完整逐项导入和逐类批准组装。
736专项通过；17开发路线212卡、40冻结独立路线152卡已生成，全部未填写，独立RGB未目视。2/4图902监督+539待审核验、pipeline check通过，四JSONL与v26保持。
合同v27/data_v27，默认标注v9，老师v2；真实批准清单 `teacher_registry_v2.json` 仍为空。有限trainable=true，formal_data_ready=false。
[审核包位置、修改说明与命令](TEACHER_REVIEW_V27.md) · [实测记录](twenty_seventh_audit_verification_20261004.json)。

## 2026-10-02 v26：规则老师、真实状态回放、逐类批准（历史）

UE1/UE4 原始规则标签已能产生 YES/NO；新入口 `teacher_replay` 默认面向全冻结路线，按路线并行，实际处置索引计算覆盖。
逐类批准后通过 `teacher_data` 进入现有训练入口，真实批准列表仍为空。自动题不伪称人工审核；老师一致率与人工准确率分列。
703专项通过；17开发路线1615帧，2/4图各15YES/97NO/126UNKNOWN，proceed正例11。
开发参考17题中10一致、5弃答、2人工仍UNKNOWN，不是独立准确率。原训练443题/45路线与val/test及待审JSONL保持。
合同v26/data_v26、默认标注v9、快照v11；需新候选/数据/run。有限训练trainable=true，formal_data_ready=false。
[实现范围、批准规则与运行命令](RULE_TEACHER_V26.md) · [实测记录](twenty_sixth_audit_verification_20261002.json)。

## 2026-10-02 v25：实际输入审核、撤回弱标签、分层抽卡（历史）

撤回UE1 f70–72及UE4 f50–52六条弱证据YES为UNKNOWN；训练**443题/45路线**，待审539，旧435训练题与val/test保持。新visual_review要求逐模式2/4RGB来源、当前证据定位及补问后继视觉证明；模式间可分别保留或弃答。保守余量/位移口径见[视觉审核规则](VISUAL_REVIEW_RUBRIC_V25.md)，未经独立校准，不是驾驶阈值。

抽卡按场景、Town、边、建议答案分层并优先不同物理路线，隐藏逐卡规则建议；val/test可导出独立待审包，40条生产者准确率抽检另行保留。UE7源场景可检索身份卡，但不自动确认故障。预检新增每边每答案20路线、每评估事件5路线的缺额统计。

**653专项通过**；2/4图真实check及902监督＋539待审RGB核验，默认七轮world1/4第3轮覆盖443题。实看87不同RGB，包含002053完整45帧复核；没有新增全库独立序列覆盖或监督。**formal_data_ready=false**，关键转移、独立40路线审核和六事件评估未完成。

合同v25/data_v25、默认标注v9、快照v11，须新候选/数据/run。详见[v25报告](TWENTY_FIFTH_AUDIT_FIXES_20261002.md)、[验证记录](twenty_fifth_audit_verification_20261002.json)、[独立审核待分配清单](twenty_fifth_independent_review_handoff_20261002.json)。以下均为历史记录。

## 2026-10-02 v24：半自动审核与首批监督增量（历史）

采用方案B：规则定位局部释放/转折，逐帧RGB审核后编译入库。分开前车驶离与稳定跟车，避免下一路口车辆污染行人事件；压缩路口与旧参与者候选，新增有预算审核队列、显式方向/目标请求、相机分界面板及严格证据编译。自动建议仍不能直接作为标签。

实际复看6路线局部84张不同RGB（20面板，其中4原图），无新完整序列。新增14题：6 readiness YES、4 readiness NO、4 catchup YES；训练**449题/45路线**，首次补入RE3 enter的2YES与UE2 depart的2NO。原435训练题逐行保持，val273/test186/待审533逐字节保持；评估仍仅四事件。

**630专项通过**；2/4图真实check及908监督＋533待审RGB核验通过，七轮world1/4第3轮覆盖449题。**formal_data_ready=false**：return/recover_follow等关键支持、六事件独立评估及40条人工抽检仍缺。无完整4B/GPU/DDP/CARLA验收。

合同v24/data_v24、默认标注v8、快照v11，须新候选/数据/run。详见[v24说明与命令](TWENTY_FOURTH_AUDIT_FIXES_20261002.md)、[验证记录](twenty_fourth_audit_verification_20261002.json)、[RGB审核](twenty_fourth_rgb_review_20261002.json)。以下均为历史记录。

## 2026-10-02 v23：静态冲突、跟车语义与自动上下文（历史）

修正静态物体零语义计数导致空走廊、正常跟车误作阻挡、仅2m停车等待门槛；区分可能影响RGB的异常和相机外清单删除，安全许可仍检查全部。接入未改当前规划路线的分段走廊、交通控制和过车部分事实，补路口待审假设与UE5近中心线候选；负静态extent只作诊断且拒绝许可。

610专项通过；实跑17路线1615帧6066待审状态对，尚无批准的自动监督。复看5路线局部82张RGB/其中4原图，无新增完整序列。2/4图check及894监督＋533待审RGB核验通过，四JSONL与v22相同；训练435/42路线、关键四边0，评估仍仅四事件。**formal_data_ready=false**，40条独立抽检预约仍未审核。

合同v23/data_v23、快照v11，必须新候选/数据/run。详见[v23说明](TWENTY_THIRD_AUDIT_FIXES_20261002.md)、[验证记录](twenty_third_audit_verification_20261002.json)及[逐帧RGB记录](twenty_third_rgb_review_20261002.json)。以下均为历史记录。

## 2026-10-02 v22：瞬移、类别、可见性与上下文接线（历史）

新增自车运动补偿的同ID跳变检测，修正自行车/静态障碍类别和弯道对向车误候选；加入投影RGB亮度/对比度/过曝筛查。接通当帧来源绑定的状态/导航/场景上下文与新范围横向参考校准；有充分证据时正例路径可达，默认假设状态候选仍UNKNOWN，不冒充真实恢复历史。

592专项通过；12路线1087帧扫描有72异常候选，未批准自动标签。旧校准仍311弃答/8对/1错；可靠上下文及可见性自动来源未完成。2/4图check与894监督＋533待审RGB通过，四JSONL与v21相同；训练435题/42路线，四关键边仍0，val/test仍仅四事件。

九组train-only、五风险记录接入；40条未曝光留出路线预留独立生产者人工抽检，尚未审核或产生评估标签。任务v22/data_v22、快照v11须新产物，formal_data_ready=false。详见[v22说明](TWENTY_SECOND_AUDIT_FIXES_20261002.md)与[验证记录](twenty_second_audit_verification_20261002.json)。

## 2026-10-01 v21：候选生产与离线安全适配（历史）

新增因果bboxes/metas候选生产、开发校准及离线安全适配器；只计算当前/历史字段，不读取未来轨迹或专家动作。安全适配器仍需要真实当帧导航包络和覆盖/通行权证据，缺失不授权；明确安全拒绝改为正常等待，新增回放冲突与安全等待统计。

六路线712帧实扫得到26个ID缺失候选；320个有效readiness参考中311弃答、8对1错，尚不能批准自动标签。六物理组显式train-only、七风险记录接入；只复看5张原图，无新增完整序列或监督。训练仍435题/42路线，关键depart/enter/return/recover_follow仍0，val/test仍只四事件，formal_data_ready=false。

567专项通过，2/4RGB实际check与各894监督＋533待审RGB核验通过，四JSONL与v20逐字节一致。合同v21/data_v21/快照v11须新产物新run；不是全量可靠标签或真实CARLA验收。详见[v21说明](TWENTY_FIRST_AUDIT_FIXES_20261001.md)与[验证记录](twenty_first_audit_verification_20261001.json)。

## 2026-10-01 v20：跟车恢复与可见范围安全准入（历史）

审计确认 RECOVER 缺少直接进入普通跟车的边，现增加 recover_follow→FOLLOW；depart/enter/return 改为可见范围判断，运行时须另有当前规划器/BEV安全凭据，缺失不会授权机动。尚未接入真实BEV适配器，不将专家蠕行当作冲突已解除。

旧横向标签需逐帧复核新范围，128道原监督题转待审；当前两RGB均 train435/val273/test186、待审533，训练42路线，独立评估仍仅四事件。新恢复边监督为0，正式数据未就绪。15组显式train-only、10条路线风险登记，无新增训练标签或完整序列审计。

任务v20/data_v20、快照v10，须新候选/数据/run。529项专项通过；边界、接口、审计判断及验证见 [第二十轮修复说明](TWENTIETH_AUDIT_FIXES_20261001.md)。以下为历史记录，旧题数和版本不能当作当前状态。


## 2026-10-01 v19：环岛目标出口与实例边界

RE5 路线目标进入状态对；显式环岛上下文要求目标出口、结束边界及导航证据。保留三状态和再次让行，完成必须到指定出口，独立 STOP 不撤销；catchup 不替代出口证据，执行回执/回放真值绑定路线。新增九个环岛 readiness 支持格；已接入两轮22处风险/21路线、三个新train-only组。详细接口、原图复看、合同边界见 [v19修复说明](ROUTE_V19_20261001.md)。

485项专项及两种RGB实际check通过；各1022监督+405待审全RGB核验通过，四JSONL与第十七轮逐字节一致，七轮采样和恢复保持。当前默认单独训练目录升为data_v19，快照v9；必须新候选/数据/run。完整数据准入仍false，关键正例、独立评估、全量条件生产与GPU验收未补齐。以下v18及更早条目为历史记录。

当前为 **合同 v18/data_v18，默认审核标注仍为 v7**。已补齐事件 1:1 采样、事件等权选优及全目录候选清点入口；**全量可靠因果标注生产仍未完成，不能称已满足全事件正式训练条件。** 有限题库探索训练保留 `trainable=true`，训练主机仍须检查完整基座、依赖和 RGB。

第十七轮补齐曝光隔离和风险准入，任务名仍 v18。`NonSignalizedJunctionLeftTurnEnterFlow/Town03_route_001041` 原默认分 val，现在固定 train-only；`Town03_route_001040` 原默认分 train，也显式登记为 train-only。两组均不属于冻结 holdout，当前题库均无题。重建后的全量候选路线数为 train **8166** / val **765** / test **781**，总9712路线/1637680帧不变；实际只有001041对应路线由候选val移入train，既有独立监督val/test不受影响。

`seventeenth_audit_exposure_20261001.json` 登记20处视觉不连续/14条路线，40条相邻原图证据核SHA；保留001041原图纠正的127→128，三处未确认候选不计入确认清单。消失、遮挡物消失和整组车消失不能生成“冲突解除”YES；同路线多个风险必须共同覆盖全部因果输入。序列结束但返回尚未确认仍须右删失，UE3对象身份不确定仍须复审，未调整提示词/状态/标定窗口。

本次 **462 项专项测试通过**（新增5项曝光/风险回归）；2/4RGB实际pipeline check、各1022监督题+405待审因果图身份核验通过。四份题库JSONL与第十六轮逐字节一致，单进程/四进程七轮题序及累计历史一致，第3轮覆盖563题。两新增曝光组在100个测试seed下均为train；旧曝光摘要的候选清单即使重算自身摘要也拒绝加载。验证见 [第十七轮登记记录](seventeenth_audit_registration_verification.json)，临时产物 `/tmp/phase4_seventeenth_registration_20261001/data2`、`data4`。

风险/曝光资产及构建源码SHA改变，须**新候选清单、新数据、新run**，旧产物须原源码，不改hash绕过检查。采样、训练、控制器、提示词、标定、人工标签、冻结holdout的文件SHA未变，Phase3/Action稳定默认保持。本次没有新增目视或标签；继承审计432条/52344帧、180/278格三ID，余268条/46324帧，18格源不足。全量可靠因果生产及合格/无关/无效/待审处置核算尚未完成；不能仅把 `complete_causal_production` 改True。三个关键readiness正例、六事件独立val/test缺口保持，未正式4B/GPU/DDP/CARLA验收。

第十六轮只做风险登记与构建准入接线，**任务名仍 v18，采样策略保持**。新增 `sixteenth_audit_exposure_20260930.json`：9 处车辆突变消失、1 条绕行处画面倾斜、2 条骑行姿态/身份不确定记录，共 12 条记录、11 条风险路线；12 个已审物理组原本均为 train-only，合并后曝光集合不变。29 条定位图证据核对文件 SHA，复用已有逐帧风险审核门槛；同一路线的消失与倾斜记录必须一并审核，任一因果历史帧 uncertain/excluded 都不能产生当前 YES 监督。自然出画和“仍可见但可能不阻挡”的路线不被误记为突变消失，也不自动生成标签。

本次 **457 项专项测试通过**，2/4RGB 实际 pipeline check、各 1022 道监督题+405 待审题全部因果 RGB 身份核验通过；四份 JSONL 与第十五轮逐字节一致，单进程/四进程七轮题序及累计历史一致，第3轮均覆盖563题。采样/训练/状态机/提示词/标定/默认标注/冻结holdout资产已核SHA未变；Phase3/Action稳定默认保持。验证见 [第十六轮登记记录](sixteenth_audit_registration_verification.json)，临时产物 `/tmp/phase4_sixteenth_registration_20260930/data2`、`data4`。

虽然任务名不变，风险清单和构建源码 SHA 已变化，**需要新建候选清单、数据产物与 run**；旧产物/checkpoint 使用原源码，不修改旧合同 hash 复用。新风险没有进入现有题库，登记不能当作新增有效标签。继承第十六轮视觉审计累计415条/49,929帧、172/278格三ID，余285条/48,739帧，18格源不足；本次没有新增目视审阅。全量可靠条件生产、三个关键readiness正例及六事件独立val/test仍缺，`formal_data_ready=false`，没有正式4B/GPU/DDP/CARLA验收。

第十五轮修复默认 7 轮漏选：旧策略只将事件内起点每轮移一位，本池单进程/四进程七轮累计仅 413/416 题。现在按**实际累计选中次数优先、同次数下优先更久未选题**安排事件内队列，保留固定转移/答案/来源/路线顺序作为最后的平局规则；只有实际选中的题增加计数，受 cap 阻挡的题不会被误算为已使用。帧内重复分配也参考实际次数。当前两种 RGB、world1/world4 均在第 3 轮覆盖 563/563，七轮保持全覆盖；逐轮累计分别为 401→540→563 与 404→543→563，事件 1:1 与全局 cap8 均保持。

采样历史保存到每轮 checkpoint 的 `sampling_history`，绑定题池内容、seed、cap、预算、world size、next_epoch；恢复后顺序和累计计数与连续训练一致。缺历史或题池/参数/轮次不符拒绝；输入历史不就地修改，失败不会提交部分计数。直接调用 `plan` 时传入上一轮审计中的 `next_history` 可线性顺序回放；未传入时从第零轮确定性重建，仅用于随机访问，不应在大规模逐轮调用中反复重建。预检新增 `default_seven_epoch_coverage`，按 world1/world4 明确列出累计覆盖及缺题 ID；每轮采样审计同样报告累计缺额。容量约束下仍不承诺任意题池在七轮必定覆盖。

第十五轮验证：**450 项专项测试通过**，包括独立物理帧配额窗口、实际 563 题池、共享帧阻挡后的补选、历史重排/JSON 往返/错配拒绝以及实际 torch 训练编排的恢复历史一致性。2/4RGB 实际 pipeline check、各 1022 题+405 待审 RGB 身份、14 个默认等额采样计划及 563 轮 budget10/cap1 检查通过。四份 JSONL 与 v17 逐字节一致。第十五轮新增 10 处消失风险/6 条路线接入已有逐帧审核门槛；本次只核对 20 条定位 RGB SHA，没有新增目视或标签。验证见 [v18 记录](fifteenth_audit_fixes_v18_verification.json)，临时产物 `/tmp/phase4_fifteenth_fixes_20260930/data2`、`data4`。

本次需新候选清单、新数据、新 run；v17 及更早 run 保留原源码，不修改合同 hash 强行恢复。标签、提示词、事件状态、标定、独立 split、选优口径及 Phase3/Action 稳定默认保持。全量可靠条件生产仍未完成，三个关键 readiness 正例缺口和六事件 val/test 缺口仍在；继承视觉审计为 403 条/48,357 帧、169/278 格三 ID、余 297 条/50,311 帧、18 格源不足。没有正式 4B/GPU/DDP/CARLA 验收。

第十四轮工程修订：默认 pipeline 先扫描全部 RGB 路线并冻结物理组 split，写出 `DATASET.candidates.json`，随后构建时绑定该清单。实际发现 9,712 条路线、1,637,680 帧，路线数 train/val/test 为 8,165/766/781；这是文件清点，不是新增目视审阅或标签。使用 `CANDIDATE_POOL=/path/to/frozen.json` 可在同一合同下复用冻结清单，构建/加载拒绝清单外路线、历史帧和 split。清单不判断事件身份、不把未标注帧补 NO；`production_coverage` 单列未构题帧，`formal_data_ready=false`。扩池仍需经校准的因果条件生产者产出 `CONDITION_STREAM` 或逐帧审核标注，不能称现有编译器已能自动读完全部 RGB 并可靠标定。

新增 15 处消失风险/11 条路线已接入 risk_review，定位图 SHA 校验通过，未批准新监督标签。继承最新视觉审计累计 391 条/46,070 帧、166/278 格三 ID、剩 309 条/52,598 帧，18 格源不足；本次新增目视审阅为零。
第十四轮验证：**435 项专项测试通过**，含共享帧容量小拓扑穷举对照、事件等额/预算拒绝、事件等权指标、候选冻结与越界拒绝、新风险准入、实际构建/加载和旧选优报告拒绝。2/4RGB 实际 `MODE=check` 通过，各 1022 道监督题和 405 道待审题 RGB 身份核验通过，四份 JSONL 与 v16 逐字节一致。每模式 14 个等额采样计划及 563 轮 budget10/cap1 覆盖全部 563 训练题；这是本池的实测结果，不是所有共享帧拓扑的无条件覆盖保证。全量清点中仅 1,074 个不同帧已有题目、782 帧有监督；剩余帧未经事件相关性与因果条件审核，不能称其均为应训练样本。验证详见 [v17 记录](fourteenth_audit_fixes_v17_verification.json)，临时产物 `/tmp/phase4_fourteenth_fixes_20260930/final2`、`final4`。

本次改变训练采样分布与选优指标，尚无真实 GPU 效果对照；必须重建数据并新建 run，v16 及更早 checkpoint 继续使用原源码，不改 hash 强行续训。提示词、状态、标定边界及默认人工标签保持；Phase3/Action 稳定默认保持。

当前 2/4RGB 各有 **train 563、val 273、test 186，共 1022 道 YES/NO 题**，另有 405 条 UNKNOWN；四份 JSONL 与 v14 逐行保持。新 `admission.py` 要求三个独立 split 非空、有 YES/NO，训练还须有 readiness YES/NO。物理路线隔离、曝光隔离、冻结评估协议、源图/像素身份、因果历史、条件来源、输入冲突和风险审核检查继续执行。`trainable=true` / 数据 manifest 的 `ready=true` 只表示训练数据准入；`ready_scope=training_data`，不代表完整流程效果通过。

**完整覆盖单独报告：`complete_coverage=false`**，仍缺 366 个转移支持格、12 个 RE3 分段支持格。它们不再默认阻止训练或训练后的独立 test；需要完整覆盖验收时显式使用 `--require-complete-coverage`。训练输出 `data_admission.json`，验证/测试报告明确为已标注子集评估，缺失组不能据此声称通过。最佳模型按已有 val 子集选取，test 不参与选优。

新增 `condition_builder.py`，统一复用 taxonomy 中的转移条件：输入同一帧、同一实例的因果条件，自动枚举当前状态适用的状态对并构题；没有条件的题保留 UNKNOWN，不从动作执行、固定前后窗口或未来轨迹生成许可。十事件及 UE2 不返回、UE4 跟随/绕行等分支共用同一套编译逻辑。**该入口完成的是“因果条件→题目”，尚未实现/验收从任意原始 RGB 自动提取全部条件的视觉系统。** 当前默认仍使用已审核的 563 道训练题，不能把新增接口称为已完成全数据集自动标定。

此前 v15 的 399 项专项测试通过，包括实际 torch 优化/保存流程的轻量替身测试、训练后 test 入口、不完整覆盖可训练、空/单类 split 拒绝、未来条件及源内容错配拒绝。轻量替身只证明接线，不是完整 Qwen/GPU 效果验收。新编译器对既有审核中的 721 个 readiness 判断回放一致；未增加标注或曝光。2/4RGB 实际完整流水线的数据预检通过，不要求本机权重；全题图像和采样核验结果见 [v15 验证记录](training_admission_v15.json)。临时产物位于 `/tmp/phase4_general_pipeline_20260930/`，其中 final2/final4 为最终版本。

全量逐帧覆盖仍单独推进：预留评估组已查看 25 条/2294 帧，尚有 26 组未查看，另有两条 UE7 上下文待确认；第十三轮开发审计曾累计 379 条/44204 帧，165/278 格三 ID，剩 321 条/54464 帧，18 格来源不足；这些是继承的审计统计，本次工程修复未新增目视审阅。这些进度用于审计完整性，不再合并为首次训练准入条件。历史结果保留于 [v14 数据与 RGB 记录](formal_data_status_v14.json)，其中旧版“尚不能开训”属于当时的严格门槛结论，已被本轮准入策略覆盖。

本次修复旧轮次恢复选错最佳模型：每个 checkpoint 的 `training_state.pt` 绑定当时最佳 adapter 的相对路径、分数、epoch 和推理资产 SHA；`selection.json` 保存该 adapter 自身验证分数。恢复不读取父 run 后来更新的 `best.json`，缺失/被替换的最佳资产、分数错配、未来 epoch 均拒绝。搬迁须保留被引用的历史 checkpoint 与相对目录关系，单独复制一个引用外部最佳模型的 epoch 不足以恢复。v15 及更早产物须原源码；本次须重建数据并新建 run，不能改旧合同 hash 强行兼容。

第十二/十三轮的 30 处消失风险（29 条路线）和 3 条 ControlLoss 事件身份待审路线已接入逐帧风险门槛，补登记 `AccidentTwoWays/Town02_route_001609` 为 train-only。未批准新标签；提示词、状态、标定窗口及冻结 holdout 规则保持。构建/预检新增 `transition_support` 的 readiness YES/NO 明细及 `train_missing_readiness_yes/no`：UE2 return、RE3 enter、UE7 proceed 仍分别为 0/11、0/9、0/4。catchup 不能填充这些正例缺口，当前结果只能评估已覆盖子集。修复与验证见 [第十二/十三轮修复记录](TWELFTH_THIRTEENTH_AUDIT_FIXES_20260930.md)。

在明确接受有限题库探索范围时，从仓库根目录在训练服务器运行以下一行，按顺序完成 2RGB、4RGB 的候选清点、构建、预检、训练、独立 test 和审计打包（使用服务器已配置好的 Python 环境；默认模型为 `AutoMoT/checkpoints/Qwen3.5-4B`）：

```bash
(set -e; for P4_RGB_MODE in 2 4; do RGB_MODE="$P4_RGB_MODE" bash AutoMoT/qwen3vl_local/sft_new_loop_phase4/run_full_pipeline.sh; done)
```

只在本机检查数据可用性，可将 `MODE=preflight` 加到调用环境；结果中的 `model_checked=false`、`ready=null` 表示未检查训练主机，数据结论看 `data_ready`。正式 single/ddp 入口必须在训练主机通过 `--require-ready` 检查模型与 RGB，不能用数据检查代替。

此前代码完善见 [RISK_ADMISSION_20260930.md](RISK_ADMISSION_20260930.md)：已知风险路线现在必须提供覆盖全部输入RGB的逐帧复审，缺证据拒绝构建；不确定/排除帧使整题进入待审，加载再次核验。新增审核模板CLI和预检计数；311项测试通过，默认两模式453题/133待审逐字节保持。任务v11/data_v11，须新数据新run；提示词v6/快照v8/标注v4保持。未新增目视审阅或标签，ready=false。

此前接入见 [ELEVENTH_AUDIT_REGISTRATION_20260930.md](ELEVENTH_AUDIT_REGISTRATION_20260930.md)：Town12_4608_0已固定train-only，七条重点路线复审证据登记；未改默认标签/提示词/状态机。291项专项通过，两模式453题/133待审与第十轮产物逐字节一致，RGB与采样核验通过。任务名v10保持但合同摘要更新，须新数据新run。覆盖155/278格三ID、剩361条；本轮只复看34张既有RGB，完整审计仍未完成，ready=false。

此前审计接入见 [TENTH_AUDIT_REGISTRATION_20260930.md](TENTH_AUDIT_REGISTRATION_20260930.md)：登记四个曝光组train-only及五条风险路线复审证据，未改默认标签/提示词/状态机。289项测试通过；两模式453题/133待审与此前v10逐字节一致，RGB与整池/小预算核验通过。任务名仍v10，源码/曝光合同摘要已更新，须新产物新run。覆盖进度137/278格三ID、剩389条；本轮仅复看72张既有RGB，完整审计未完成，ready=false。

此前修复见 [NINTH_AUDIT_FIXES_20260930.md](NINTH_AUDIT_FIXES_20260930.md)：小预算采样改为单次全局轮转；RE3新增相邻段接续，每段重问间隙、绑定执行回执，沿用WAIT/CROSS并保留纵向约束。284项测试通过，2/4RGB各453题核验与整池/小预算采样检查通过。任务v10、提示词v6、快照v8，须新数据新run；人工标注v4未改。原389个转移支持格及新增12项分段支持检查仍缺，ready=false。全审计进度120/278格三ID、剩417条；本次只复看96张既有RGB，无新增完整序列。

此前修复见 [EIGHTH_AUDIT_FIXES_20260930.md](EIGHTH_AUDIT_FIXES_20260930.md)：UNKNOWN/缺答时保留同帧明确的停车约束，按有序解码RGB与实际消息拒绝相反答案。256项测试通过，2/4RGB实际check入口、各453题RGB核验及14个采样计划通过；默认标签保持453题/133待审，未发现实际输入冲突。任务合同v9、快照v7，须新数据新run；标注v4/提示词v5/标定规则v5保持。第八轮报告107/278格三ID、仍剩441条；本轮只复看71张既有RGB，完整审计未完成，ready=false。

此前数据接入见[RGB_CANDIDATE_INTEGRATION_20260930.md](RGB_CANDIDATE_INTEGRATION_20260930.md)：第六、七轮四段候选经逐帧复核纳入默认标注v4，共453题、133条待审；原429题逐行保留。230项测试通过，两种RGB模式默认流水线check及全题核验通过。任务合同v8，提示词v5/标定规则v5/运行快照v6保持，须新数据、新run。累计完整审计仍为238条/27,633帧，97/278格三ID，剩462条、18格来源不足；本次只复看169张既有RGB（43张原尺寸），不增加完整序列计数。ready=false。

Phase4维护事件实例与当前阶段，LoRA只判断某条转移的条件：`YES / NO`。动作先验由状态机派生，不再训练动作选择问答。Phase1/2建立事件后，把结果交给`runtime.handoff`；正常推进不重新问Phase1/2，事件轴完成且纵向约束已稳定后退出实例；失效、持续不确定、许可长期未执行或外部执行故障才请求复核。可并发维护多个实例。

本目录是独立实验，不替换Phase3稳定版v23_io1、Action默认来源、旧adapter或冻结数据。默认本地Qwen3.5-4B；旧Qwen3-VL adapter不兼容。运行不下载模型。

首批边界标定见 [RGB_BOUNDARIES_V3_20260929.md](RGB_BOUNDARIES_V3_20260929.md)，后续四段增量见上方接入报告。首批45窗/701张RGB及其历史累计计数不与后续完整序列审计简单相加；所有计数均不代表全数据集逐帧认证。

| 事件/路线分支 | 最小事件流程 |
|---|---|
| UE1、UE3、UE5 | YIELD → PROCEED → DONE |
| UE4横穿、沿路跟随 | YIELD → PROCEED → DONE |
| UE6、UE7、RE5 | YIELD → PROCEED → DONE |
| UE2绕行、UE4沿路骑行绕行 | WAIT → DEPART → PASS → RETURN → DONE |
| 上述绕行且路线不要求返回 | WAIT → DEPART → PASS → DONE |
| UE2本车道通过 | YIELD → PROCEED → DONE |
| RE2变道、RE3汇入/驶出 | WAIT → CROSS → DONE |

PROCEED期间出现新冲突可回到YIELD。纵向用共享小状态因子区分STABLE、APPROACH、FOLLOW、HOLD、RECOVER，对应KEEP、DECELERATE、KEEP、STOP、RESUME；因此“等待进入走廊”不强制停车，已稳定跟随也不持续输出减速。几何完成可与减速/停车同时成立：此时state=DONE但instance_complete=false，横向KEEP，纵向继续处理直到STABLE/FOLLOW。不能仅凭state字符串清空实例。左右方向、是否返回及目标走廊来自当时可用的导航/因果跟踪，不从未来GT路线回推。RE3允许FORWARD；路段弯曲不能产生变道token。

`controller.Episode`每个新观察至多推进事件轴和纵向轴各一次。已开始横向运动时，对“完成了吗”回答NO仍保持该运动先验。YES许可需要执行器回执`acknowledge(instance_id, frame_id)`；未执行的许可下一观察重新判断，不一直锁存。执行端应确认**实际开始执行**，不能以模型YES自动充当回执。返回的是目标走廊约束和先验，不是直接执行的油门/转向命令；到达目标后不能重复横移到下一条车道。

`action_prior.export`输出七类token名称/编号、文字、每个实例的纵横向约束。UNKNOWN/INVALID不冒充UNCOND；并发停车/减速约束保留，冲突横向目标交回复核，并保留STOP/DECELERATE及conflict_instances/recheck_instances；runtime持久化实例复核状态，可调用revalidate逐项解决。临时UNKNOWN仍为UNKNOWN。只有instance_complete=true才释放该实例约束；复核时保留已有STOP/DECELERATE约束。

提示词、训练与推理共用`prompts.py`。视觉输入为原始2/4张因果RGB；文字只展示“当前状态”和“候选下一状态”，把转移原因、方向及动作含义融入这两个描述，最后问是否转移。候选下一状态是流程目标，不是下一帧真实状态。速度及帧号仅用于输入检查，不渲染到提示词；不展示整个状态链或后续分支。场景名、Town、route ID、未来轨迹、控制量、标注事实和审计文字不进入模型。图片有因果帧号和内容哈希检查；初始帧<4不使用，缺历史不复制填充。训练及adapter保存observation_contract；实时/序列与训练图数和历史间隔必须一致。Phase4Loop(rgb_mode=2或4)显式选择，快照也保存该合同。

默认标注使用`reviewed_state_pairs_v7.json`。用户要求按不同事件、Town和路线的实际RGB决定前后范围，上一版直接固定±0.5秒的实现已撤销。

- `reviewed_transition_band`逐帧声明readiness/catchup/uncertain/excluded，并记录可见参考帧和停止采旧问题的理由。参考帧只用于汇总偏移，不能自动产生YES；不同路线的前后范围可以不对称。
- readiness要求当前条件成立；完成类转移必须已经观察到完成，不因“快完成”提前标YES。catchup仍问同一旧状态对，须确认后继已达成并审核冲突是否阻止本条转移，单独统计。current_conflict=true时必须明确transition_blocking_conflict；独立纵向限制不否定已经完成的几何事实，缺少作用范围则要求复审。
- 在下一阶段、新冲突或普通行进占主导时停止采旧问题；instance_boundary和calibration_anomaly分别记录人工确认的实例边界与采集异常截止；截止之后不自动标NO。`review_end`表示只看到窗口结束，还未确定最晚边界。
- `context_valid=false`为内部INVALID，证据不足为内部UNKNOWN，进入`review_queue.jsonl`，不进入YES/NO模型监督。每帧绑定RGB SHA256、当帧观察和审核来源。
- 旧`transition_point_window`构建输入明确拒绝；v2标注仅保留历史。旧逐帧事实格式仅保留readiness；旧slice=catchup和旧catchup_label均拒绝，须人工复审为reviewed_transition_band，不自动迁移时间区间。

当前默认含 152 条审核记录（含两条上下文待确认），产出 1022 道二元题（751 readiness、271 catchup），按上述独立物理路线分为三 split。**训练数据准入已通过；完整覆盖缺额见顶部，尚未生成正式训练的 Phase4 LoRA。** 下列命令从`AutoMoT/`运行。

```bash
# 看完整绕行/路线不返回/骑行绕行demo；无需GPU，答案是合成条件
python -m qwen3vl_local.sft_new_loop_phase4.demo
python -m qwen3vl_local.sft_new_loop_phase4.demo --no-return --direction RIGHT
python -m qwen3vl_local.sft_new_loop_phase4.demo --event U-E4 --branch cyclist_bypass
python -m qwen3vl_local.sft_new_loop_phase4.demo --event R-E3 --direction FORWARD
python -m qwen3vl_local.sft_new_loop_phase4.demo --segmented-exit

# 审核后的区间→训练/评估数据；output必须是新目录
python -m qwen3vl_local.sft_new_loop_phase4.dataset \
  --annotations qwen3vl_local/sft_new_loop_phase4/reviewed_state_pairs_v7.json \
  --output-dir checkpoints/sft_new_loop_phase4_data_v18 --rgb-mode 4

# 只检查/回放采样，不伪装成正式训练通过
DATASET=checkpoints/sft_new_loop_phase4_data_v18 bash qwen3vl_local/sft_new_loop_phase4/train.sh check
python -m qwen3vl_local.sft_new_loop_phase4.preflight \
  --dataset checkpoints/sft_new_loop_phase4_data_v18 --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B

# 在实际训练服务器执行；默认审核库已经有独立三split
bash qwen3vl_local/sft_new_loop_phase4/run_full_pipeline.sh
GPU_IDS=0 DATASET=/path/to/ready_dataset bash qwen3vl_local/sft_new_loop_phase4/train.sh single
GPU_IDS=0,1,2,3 DATASET=/path/to/ready_dataset bash qwen3vl_local/sft_new_loop_phase4/train.sh ddp
# 2/4RGB分别构建、分别新run，不混用身份
bash qwen3vl_local/sft_new_loop_phase4/run_rgb_mode_matrix.sh

# 严格原合同的epoch边界恢复；不能恢复epoch中途
DATASET=/path/to/ready_dataset bash qwen3vl_local/sft_new_loop_phase4/train.sh single \
  --resume checkpoints/sft_new_loop_phase4_runs/OLD/epoch_002 --epochs 7

# 指定adapter目录；best.json记录本run最佳完整val，test不选优
bash qwen3vl_local/sft_new_loop_phase4/eval.sh \
  --dataset /path/to/ready_dataset --adapter /path/to/run/epoch_003 \
  --split test --output-dir checkpoints/sft_new_loop_phase4_eval_NEW
python -m qwen3vl_local.sft_new_loop_phase4.replay \
  --sequence /path/to/sequence.json --adapter /path/to/run/epoch_003 \
  --output checkpoints/phase4_replay_NEW.json
python -m qwen3vl_local.sft_new_loop_phase4.audit_bundle \
  --run /path/to/run --output /path/to/phase4_audit.zip
```

默认 `--sampling-policy event_equal`：各事件相同配额，事件内按转移/答案/readiness、catchup和物理路线轮转。全局物理帧 cap8 包括重复题和跨事件共用帧；优先取不同题，允许在 cap 内重复小事件题，不能补出缺失正例。配额通过共享帧容量流联合分配，容量不足给出事件缺额并拒绝，不静默降配。显式总预算必须为 lcm(事件数, world_size) 的正整数倍；默认将池大小向上取整到该倍数，当前单卡每事件57题/总570，world4每事件58题/总580。小预算十事件最少10题（world4最少20）；budget1不可能满足十事件1:1。旧单环策略可显式 `--sampling-policy legacy_ring` 用于对照，默认已切换；此策略改变训练分布，尚无GPU效果验证。队列按实际曝光历史跨epoch补选，共享帧竞争可能约束全题覆盖，审计报告七轮累计选中量与缺题，不宣称无条件N轮全覆盖。验证/测试完整遍历已构建的独立合格题池，不重采样。2/4RGB使用同一标注、同一物理路线划分与seed，输入长度改变需独立合同和run。

选优先计算各事件内逐题准确率，再对已观测事件等权平均（`event_macro_accuracy_observed`）。固定十事件的 `event_macro_accuracy` 只有十事件全有验证题时才有值，否则为 null 并列出 `missing_events`；子分组数量不再改变事件权重。当前 val/test 仍只有 UE1/5/6/7，故选优只代表四事件子集，不能描述为完整十事件效果。`selection.select_best` 拒绝旧分组宏平均报告。

训练复用本地Qwen3.5/DeltaNet LoRA加载，监督仅assistant答案与结束符；支持视觉LoRA off/merger/last4/all、单卡/DDP、梯度累积尾部、逐epoch完整自由生成验证、best/latest/final与优化器epoch边界恢复。源码、提示词、数据manifest、采样参数、基座资产均绑定合同。`selection.select_best`先核验兼容性再比较val；不会按“最新目录”捡adapter。

与Phase3的功能对应：构建/预检/单卡-DDP/两四图矩阵/完整生成评估/错误题JSONL/审计包/源码合同已有独立入口；原Phase3六动作choice或多动作binary标签不继承，因为本任务输出必须是转移条件。RGB审阅面板用`rgb_audit.py`；在线调用入口用`Phase4Loop`，未接管现有CARLA/Action生产runner，未修改其默认先验来源。没有宣称所有Phase3旧实验开关逐一兼容。

序列文件包含`initial`（Episode构造字段）和按帧升序的`observations`；每项有`frame_id`、`observation={frame_id,history_frames,speed_mps}`、`images`相对路径、`image_sha256`、可选`truth={edge:答案}`、`context_valid`和来自外部的`execution_committed`。必须按当前**预测状态**重新构题，不逐帧灌入GT阶段。缺边真值计uncovered，不能补NO；已开启区间在未知真值处以truth_unknown删失，后续YES重新起算。整帧缺失和不再询问的边也分别截止；last_ready_frame记录最后一个已证实成立帧。精确延迟仅覆盖逐帧已审核的连续YES区间，不能把旧跨缺口指标直接混比。报告提前YES、答案/控制器接受/执行确认各自的延迟、未执行右删失记录和整体完成；delay_frames仅指执行确认延迟，不能再当作答案延迟。正常明确NO的等待不判停滞；这是固定录制观测上的预测状态回放，不能冒称车辆动作改变后的闭环仿真。

审计包仅打包指定run的指标/错误题/合同，30MB上限，超限明确报错，不上传、不包含模型权重或本机聊天。RGB与生成输出位于忽略目录或checkpoints，不进源码仓库。

本机验证使用独立`/tmp/automot-qwen35-env`（Transformers 5.3.0），真实小型Qwen3.5多模态网络做过条件答案mask和LoRA反向测试；完整4B权重缺失，因此未真实4B/GPU/DDP多卡或CARLA验收。此前第二轮172项专项通过；另完成2/4图真实开发索引构建、各7轮×world_size 1/4采样回放、全部历史图像哈希核验及shell语法检查。完整流程效果仍需服务器模型训练与独立序列对照；这与本轮已通过的数据训练准入分开记录。

`coverage_report`同时检查十事件的独立路线/正负支持和每个事件流程边及实际可达公共纵向边的训练转移正负支持（仅readiness），包含UE2不返回/本车道通过、UE4跟随/绕行及不返回分支；catchup的YES不能填补训练转移边的YES缺额，缺额字段为`missing_transition_edges`，其中公共纵向边另列`missing_longitudinal_edges`（v15 数据共260项缺额）。这是完整覆盖诊断；默认不阻止训练，不代表泛化能力证明。

完整流水线现在串接全目录候选清点/冻结划分→因果条件编译及审核题构建→预检/训练→按val所选best做独立test→审计打包；MODE=check 只做采样检查；MODE=preflight 校验数据准入与全部监督输入 RGB，不加载模型。此接线有进程边界测试，未真实GPU验收，仍不能认证Phase3所有实验开关/原始证据审计已完全对齐。

handoff接收明确完成的Phase1八项（HIGHWAY/STATIC_OBSTACLE/VULNERABLE/TRAFFIC_LIGHT_ABNORMAL/RS1/RS2/RS4/RS5）和Phase2五项（UE1/UE3/UE5/UE6/INVALID_EVENT_CONTEXT）布尔判断；建立实例时缺项拒绝，有效性必须明确False。旧状态补问可用executor或causal_tracker的结构化execution_receipt确认动作已经开始，绑定当前实例/决定/边/因果观察；具体协议及快照恢复用例见修复说明，不以模型YES充当执行证据。

第二轮审计报告的23个新曝光物理组已写入Phase4 train-only隔离（reaudit_exposure_20260929.json），不加入盲测，不新增监督标签。UE3旁侧通过仍须有既定通道、足够间隙和优先条件，不由提示词自动指定左绕或产生未经确认的变道先验。

第三轮已审的11个新曝光组已登记train-only（third_audit_exposure_20260930.json）。tick在副本上完成整批输入/预测/回执核对后统一提交；抛异常时旧状态及待执行许可不变，允许重试或确认原有效决策。跨路口的新目标使用独立instance_id，不能沿用旧许可。

新增UE4在骑行者仍可见且已离开所需通道的f56–57标readiness YES，自车未起步不延迟许可；f58突变消失为calibration_anomaly截止，不据此构造YES或执行确认。标注v4、复核证据rgb_boundary_review_v4.json均绑定新合同，原v3历史文件保留。

RE3分段支持单独检查中间段/末段的enter、complete正负readiness，缺额见`missing_segmented_route_support`。相邻段导航合同、分段回执和回放真值绑定见第九轮修复说明；模型不看完整路线计划。


条件流批量入口使用 JSONL，首行为 `{"policy":"phase4_causal_conditions_v1","producer":...}`。producer 包含非空 `name`、`version`、`source`（`rgb_review` 或 `causal_geometry_review`）及 `rubric_sha256`（由 `condition_builder.rubric_identity()` 获取）。这只是来源与规则绑定，不是对上游条件提取准确性的认证。

后续每行是一条当前实例的条件记录：

| 字段 | 含义 |
|---|---|
| `scenario`、`route_id` | 数据根目录下真实路线，不能跨路线拼图 |
| `episode` | 当前实例/状态/路线分支，使用既有 Episode 字段；不能提供未来真实阶段 |
| `frame_id`、`observed_until` | 两者相同；来源帧不得晚于当前帧 |
| `context_valid` | 明确 true/false/null；不能靠目录名猜有效实例 |
| `facts` | taxonomy 中条件名到 true/false/null 的映射，一帧提供一次，无须逐题指定答案 |
| `sources` | 非空列表，每项有 `path`、`sha256`、`frame_id`、`kind`（rgb/metas）；至少绑定当前 RGB，所有来源都核验内容及因果路径 |
| `evaluation_annotation_protocol` | 预留评估路线必须明确绑定冻结计划，与原规则相同 |

在 AutoMoT 目录执行 `CONDITION_STREAM=/absolute/conditions.jsonl bash qwen3vl_local/sft_new_loop_phase4/run_full_pipeline.sh` 即从条件流构建。可同时显式设置 `ANNOTATIONS` 合并已审核区间；同一道题有不同答案或不同 readiness/catchup 身份时拒绝合并，不静默覆盖。未显式设置 ANNOTATIONS 时，条件流模式不自动混入默认题库。`condition_compilation.json` 记录输入流 SHA 和构题统计；每题绑定原条件及来源，但这些内部证据不进入模型提示词。

编译器不会把 catchup 自动当成 readiness，也不根据动作起点固定扩窗。旧状态补问继续复用现有逐帧审核带；后续若要规模化生成补问，仍须有经验证的因果状态完成/实例边界证据。当前原始 LEAD 中的 `brake`、`throttle`、`future_speeds` 等并未被接成许可规则；通用原始数据条件提取器是后续独立工作，不应通过逐 ID 定制规则或把控制量直接换成 YES 来假装完成。
