# 2026-10-10 本机逐帧审计与任务完成度

当前已经具备标定器、数据生产和服务器训练留痕，尚未完成标签可靠性、联合隔离和下游效果验收。本机缺少可运行的对应 Qwen 基座/adapter，不妨碍逐帧重算与 RGB 诊断；本次实际执行了这部分工作，没有启动续训。

继续审计的新增结论：239 道 val 中 112 道的精确“事件×边×阶段×答案”组合在训练池计数为零，涉及 U-E5/U-E6/U-E7；而 U-E1 complete 的正负例本轮已全部采到，仍有 5 道 NO 全答 YES。两种问题应分开处理。下文 §4.7 是后续补充，不把训练缺覆盖当作模型错误的充分解释。

唯一实验进度仍见[总方案 §14.1](../sft_new_loop_phase3/PHASE3_PHASE4_AUDIT_ROADMAP_20261008.md)。本文是一次有范围的开发审计，不把 11 条路线的结果外推到全部路线或十事件。

## 1. 完成度：哪些已有，哪些仍缺

| 工作 | 可确认的完成部分 | 未完成部分 |
| --- | --- | --- |
| 标定规则 | Phase3 原轨迹标定器、context 映射和 choice 投影；Phase4 v42/老师 v10、双轴控制器、三态规则和弱监督准入均已实现 | 独立 RGB 可观察性、各边/分支覆盖、逐实例事实一致性未全部验收；“已定义”不等于“已认证” |
| Phase4 新输入与一轮训练 | 收到 RGB2/RGB4 各 10240 次呈现、每卡 2560 次前向/320 次更新；题序一致，239 道 val 的保存前后预测和原始输出一致 | 本机未重载权重；尚未验证训练例预测，未批准续到七轮 |
| G0 | 已有源码/身份/曝光审计工具及历史服务器捕获 | 完整 Phase3/Action 资产和联合用途隔离仍缺；本次不更新历史 G0 为通过 |
| G1 | 本次完成 11 条路线/792 帧原生重算、0 帧差连接、源身份和事件存在差异账本 | 实例参与者/目标逐一匹配、差异六类的最终裁定、完整条件区间和物理动作起止人工复核仍缺 |
| G2 | 本次有探索性 RGB 目视和可复核材料 | 未做第一轮有限输入锁答、独立双审、严格批准；六事件开发 val 仍缺 |
| E0 | 已有服务器逐步/各 rank 时间与峰值显存，本次有 CPU 回放时间/RSS | 分段 profiler、真实快速内核、三级帧复用分布、GPU 利用率与生成路径未测齐 |
| E1–E6、R、P 系列 | 设计与部分通用记录基础已有 | 稀疏 logits、缓存、批处理、四卡重分配、预取、验证分片、前缀梯度与新采样实验均未验收 |
| S4 | Phase4 短两图/四图完成第一轮服务器对照；Phase3 短输入入口已接线 | 非独立可观察性结论；Phase3 新输入模型结果缺失；不能代替短/长两图跨度对照 |
| S0、其他 S 与 Action/CARLA | 本次补少量标记分布诊断 | 全训练池/曝光权重分布、实验卡、训练对照与下游验收仍缺 |

不提供一个合计百分比：源码实现、一次训练成功、标签正确和闭环行为不是同等权重的完成项。

服务器证据详见[接收复核 JSON](../../checkpoints/phase4_epoch1_received_review_20261010.json)。仅收到解压目录，原 ZIP 字节与服务器当前权重未在本机核验。人工 val 仍只有 4 事件、11 物理组；同一开发 val 已用于定位问题，不能充作新的独立最终测试。

## 2. 本次实际生成了什么

产物目录：[checkpoints/joint_frame_audit_20261010](../../checkpoints/joint_frame_audit_20261010)。不覆盖现有题库，不生成新的训练批准或发布 manifest。

| 产物 | 实际范围及用途 |
| --- | --- |
| [request.json](../../checkpoints/joint_frame_audit_20261010/request.json)、source.json | 6 条既有训练审核路线、5 条已用于诊断的开发 val 路线；冻结帧清单、源码合同与匹配协议 |
| [phase3_native_candidates.jsonl](../../checkpoints/joint_frame_audit_20261010/phase3_native_candidates.jsonl) | 调用原生 `iter_base_frames`、`_make_row`、`choice_annotation`，生成 374 条合格候选、356 个不同锚点；未做配额、INVALID 增补和最终 split coverage |
| [phase3_timeline.jsonl](../../checkpoints/joint_frame_audit_20261010/phase3_timeline.jsonl) | 792 帧轨迹信号、decision trace、原动作、Phase1/2 context 来源及候选准入；尾部窗口不足和无 context 也保留 |
| phase4_routes/、[phase4_questions.jsonl](../../checkpoints/joint_frame_audit_20261010/phase4_questions.jsonl) | 连续执行原生 `route_records`，保留实例、问题、UNKNOWN、删失和拒绝原因；共 1772 条原始问题，含原生两图/四图，不是 1772 个独立锚点 |
| [joint_timeline.jsonl](../../checkpoints/joint_frame_audit_20261010/joint_timeline.jsonl) | 路线＋同帧精确连接；事件映射引用 `CONTEXT_TO_EVENT`，没有通过偏移帧凑匹配 |
| [analysis.json](../../checkpoints/joint_frame_audit_20261010/analysis.json)、case_comparison.json | 重算计数、标记分布、同帧人工验证/原生老师候选对照；未把不同状态或实例视为同一命题 |
| [visual_review.json](../../checkpoints/joint_frame_audit_20261010/visual_review.json)、rgb_sheets/ | 看完 4 条完整序列共 300 帧/21 张缩小联系表；另看 7 张原分辨率 RGB，其中 6 张属于上述序列，目视共 301 个不同帧 |
| exposure_ledger.json、receipt.json | 本次开发用途/目视范围；原生成产物大小及 SHA。新分析文件单独列入 review_receipt.json，不改原回执 |

Phase4 原生两图仍是 `[-4,0]`，不是新学生的 `[-2,0]`。回放的原生规则值不能因此获得短两图批准。`admission_target` 仅重放 registry/预约/风险门；未执行后续完整编译的去重、可观察冲突与人工优先，不等同于最终入库数量。

Phase3 候选中的 split 是原生初步散列结果，不是当前服务器训练成员：本批 225 条来自 Phase4 val 的候选被初步散列为 Phase3 train。最终 development override、split coverage 和实际训练索引未在本次重建，**这不能证明已发生训练泄漏，也不能把该诊断 JSONL 直接用于训练**。任何正式重建必须先完成 G0 联合隔离。

原生成回执 24 个文件、792 张当前帧 RGB SHA、792 帧双侧全集、374 条候选与逐帧原动作一致性均重新核验。没有运行 Qwen 前向，也没有用模型推测补写结果。

## 3. 标定规则已经确定到什么程度

Phase3 [trajectory_action.py](../sft_new_loop_phase3/trajectory_action.py) 的当前规则已经固定并能逐帧运行：纵向 2 秒/9 个速度样本、1.5 秒停车确认窗、当前停等与确认起步优先关系、速度变化确认和横向首跨等；[choice_semantics.py](../sft_new_loop_phase3/choice_semantics.py) 再生成主要动作。未来轨迹是监督来源，不应进入学生当前输入。

Phase4 [teacher_replay.py](../sft_new_loop_phase4/teacher_replay.py) 使用当前及过去的因果历史、事件实例、状态机和条件规则；必须先观察到本事件的限制，才建立实例。条件成立、已发生后继和可执行许可仍是不同命题。两套任务保留各自的翻转点，不能统一成一个“早几帧都 YES”的标定器。

例如 HardBreakRoute/Town13_1520_0 的 f94、f96：Phase3 为 `STOP/current_confirmed_wait`，Phase4 原生 `proceed=NO`；f98 Phase3 已为 `RESUME/first_confirmed_gain`，Phase4 原生 `proceed=UNKNOWN`，人工旧状态补问为 `catchup YES`；f99/f100 原生补问也为 UNKNOWN。这里既有行为/条件语义差异，又有补问题批准缺口，不能只见答案不同就修标签。

## 4. 找到的问题与修改优先级

### 4.1 U-E7 release：问题生成范围缺失，先于采样

服务器全池统计已有 `release` 无训练支持；本次选中路线原始问题中同样没有 U-E7 release。人工 val 的 18 道 release 全部使用 `PROCEED/HOLD`。

[taxonomy.applicable](../sft_new_loop_phase4/taxonomy.py) 禁止在 WAIT/YIELD 时 release；[controller](../sft_new_loop_phase4/controller.py) 接受 proceed 时会把 APPROACH/FOLLOW/HOLD 同步到 RECOVER，[synchronize_observed](../sft_new_loop_phase4/teacher_events.py) 也有类似同步。因此沿普通成功推进的自动轨迹，release 所需的 `PROCEED/HOLD` 并不自然由这个转移产生。后续重新 hold 等其他路径仍可能进入该状态，不能声称全局不可达。

应先写明哪些真实状态/滞后状态属于问答域，再为同一域设计训练候选或补问题。保留双轴限制；不能为了填格取消 RECOVER 同步、直接复制 proceed 标签到 release，或删除人工 val 的缺失边来提高分数。

第二遍使用同一历史和原实例种子重新计算 160 条 U-E7 proceed（129 个不同路线锚点，部分帧有多个实例），原 criteria/`state_rule_target` 全部复现。只把查询状态假设为 `PROCEED/HOLD`，计算 release，得到 YES 15、NO 145，与原条件值一致；其中 train 106 条，YES 10。这证明本批缺题并非所有相关条件值都算不出来，但假设状态不是可达性证明，更不是新标签。保留在[counterfactual_states.jsonl](../../checkpoints/joint_frame_audit_20261010/counterfactual_states.jsonl)，不进入训练。另有一条原生 catchup 被降为 UNKNOWN 的题，假设 release 仍算出 YES，恰好说明跨边复制会绕开原补问准入门。

### 4.2 U-E7 catchup：有意 UNKNOWN，不能靠重采样修复

老师对“此前静止 NO，首次 YES 已在运动之后”的推进问题标为 `post_stop_release_requires_visual_catchup`、target UNKNOWN。默认批准也不会把这类 UNKNOWN 变成监督。本批 val 有此类候选，所选 train 没有 proceed/catchup；服务器全池也缺该训练支持。

补齐路径应为：真实可见后继与当前冲突证据 → 精确输入的有限观察审核 → 独立命名的补问目标/准入。不能直接去掉 UNKNOWN 门、把“自车动了”当作当前放行许可。训练中已有 proceed YES 的离线预测仍待服务器，用来区分已覆盖题没学会与未覆盖状态泛化失败。

### 4.3 U-E7 Town05/002062：全路线被源几何门阻断

[source_geometry_probe.json](../../checkpoints/joint_frame_audit_20261010/source_geometry_probe.json) 显示该路线 43 帧都有静态车辆负 extent；f0 记初始化，f1–42 全为 `invalid_geometry`。例 f20 有 5 个负尺寸静态车辆实体。显式故障场景身份存在，阻断发生在实例检索之前。

[privileged_geometry.project](../sft_new_loop_phase4/privileged_geometry.py) 保留问题并仅作诊断包络，[teacher_replay](../sft_new_loop_phase4/teacher_replay.py) 对任意 `source_geometry_issues` 重置整帧历史。RGB 联系表及原图可见正常路口车辆运动，不能将几何格式问题等同于所有 RGB 标签无效。

建议作为单独修正候选：先查负尺寸的导出语义与问题实体空间范围，确定能否以可靠包络证明其与本题走廊/相机证据无关；只有证明无关才讨论局部隔离，无法证明继续 UNKNOWN。不能只取绝对值后认证，也不能直接移除整帧保护。修改需加源异常影响域回归、升合同并在新目录重放。

第二遍扩大检查到全部 792 帧：负尺寸出现在 Town05/002062 的 43 帧，以及 Town05/002066 的 f49–56 共8帧。按代码现有绝对值**诊断**包络，51 帧中的问题实体均不与原生 22 米局部导航走廊（margin 0.25米）相交，但并非都在相机之外。见[geometry_scope22.json](../../checkpoints/joint_frame_audit_20261010/geometry_scope22.json)。第一次18米探针不是原生22米口径，已另做22米复核；没有据此豁免全局几何门。

当前本地 `lead/lead/expert/expert_data.py` 的静态车导出直接写入 `get_level_bbs` 返回的 extent，无绝对值处理；源码 SHA 已记录。尚未绑定此次采集时的导出器版本，也未验证负号来源，因此只能定位到原始导出字段，不能认定是可无损修复的镜像缩放。

### 4.4 U-E1 夜间路线：可见性门的欠覆盖候选

HardBreakRoute/Town13_1056_0 的 108 帧自动回放均无问题。f55 的近处前车 5086 位于约 8.47 米，记录 8256 visible pixels，但 RGB 投影裁剪 p95≈43，小于当前 64 的门槛；contrast≈23.77 已超过 20，最终 visibility 为 UNKNOWN。见[night_visibility_probe.json](../../checkpoints/joint_frame_audit_20261010/night_visibility_probe.json)。原图可辨车体和刹车灯，但这不等于所有运动/许可条件都可判断。

先分报白天/夜间、车体颜色、距离、遮挡的候选损失，补跨路线裁剪审核，再决定是否改变亮度门。单个反例不足以把阈值从 64 调到 43；通过画面质量门也不能自动证明目标可见或空间为空。

第二遍该夜间路线有107帧包含30米内同车道车辆，其中91帧至少一辆车辆 visibility 为 UNKNOWN；另外三条 HardBreak 开发路线对应67/69/95帧，这个条件下未出现 UNKNOWN。此处只是候选过滤的分层计数，不是91次人工确认的误拒，也不是白天/夜间总体统计；需检查各实例参与者、亮度裁剪与完整 criteria，不能替换成人工可观察率。

### 4.5 U-E1 的错误要按边拆开

RGB4 验证中 9 道 NO 全答 YES，其中 proceed 2、release 2、complete 5。因此问题既包括过早放行，也包括过早认定稳定完成，不能统一叫作“前车起步误判”。同帧的多道问题来自共享输入/状态构造，也不是 9 个独立事件。

本次查看的 Town13_1520_0 f94：人工与原生 proceed 都是 NO，模型为 YES，不能用“两个标定器本来不同”消解这个错误。f100/f101 的 complete 错例则需另查 FOLLOW/STABLE 证据与训练覆盖。先在服务器对已采到的训练正负例做相同协议预测，按事件×边×阶段×运动×标签来源对照，不先加七轮预算。

### 4.6 Phase3 S0：先保持所有明确标签

374 条诊断候选中五标记并集 149（39.8%），其中 subthreshold_drop_present 105、isolated_near_stop_in_1_5s 21、first_drop_single_sample 17、gain_unconfirmed_at_2s_boundary 15、stop_pair_crosses_1_5s_boundary 5；存在重叠。分 context/主要动作计数在 analysis.json。

这批路线是针对问题选取的小样本，不是全训练池，也没有对应新模型错误率；不能据此删掉 39.8% 的训练监督。S0 仍需全候选池及实际曝光权重分布、开发语义子集和按事件/动作匹配的等量随机排除对照。

### 4.7 验证域与训练支持：不能只查两个事件

将已接收的 `data_admission.json` 全池计数、`epoch_000_sampling.json` 实际呈现和239道逐题验证按“事件×边×readiness/catchup×答案”连接，完整表在[support_matrix.csv](../../checkpoints/joint_frame_audit_20261010/support_matrix.csv)及同名JSON，三种输入文件均绑定SHA。`admission.training_report` 遍历全部训练行累加，因此缺失的该类别键可作为该类别零支持；这不是按状态、分支、实例进一步匹配的支持证明。

| 缺训练支持的事件 | 对应val题数 | 具体范围 | RGB4答对 |
| --- | ---: | --- | ---: |
| U-E5 | 60 | complete两阶段、proceed/catchup、release两阶段 | 52 |
| U-E6 | 32 | complete/catchup、release两阶段、stable/catchup | 17 |
| U-E7 | 20 | proceed/catchup、release两阶段 | 16 |
| 合计 | 112/239 | 54 YES、58 NO；不是112条独立路线 | 85 |

RGB2相同112题答对70。这里的“答对”沿用当前开发参考，不升级为独立审核。U-E5的52/60说明模型可能跨边/事件迁移，不能断言缺训练支持就必然错误；U-E7的16/20则全来自NO，不能据此说缺口已解决。

已覆盖类别同样需要诊断：U-E1 complete/readiness 全池87 YES/94 NO，本轮全部采到；RGB4对应val为2 YES/5 NO，七题全答YES。U-E1 proceed/readiness 实际呈现58 YES/319 NO，也不能以“没见NO”解释误放行。服务器优先补这些训练题的逐题预测及相同状态分层；本机只有采样/池/val记录，不能伪造训练拟合结果。

后续若补监督，应按“训练缺失但val存在”“已采到但拟合未知”“同类别但状态/可观察性分布不同”三类列任务；先补真实状态域和审核，不以平均每事件1024次呈现当作各边充分覆盖。

## 5. G1/G2 尚不能宣称完成的原因

792 帧中：两边均无事件 243，同非空事件集合 166，只有 Phase3 有事件 180，只有 Phase4 有事件 120，两边非空但集合不同 83。后面三类合计 383，是事件集合差异候选，**不是 48.4% 错标率**。初始化/历史不足、上下文范围、并发实例、源异常与实例生命周期都会影响计数。

Phase3 context 还没有与 Phase4 的 actor/instance/target 一一确认；本次所有对照保留 `contradiction_label=null`。下一步对已定位的代表窗口确认同一事实，再按任务语义、标签、不可观察、边界、源异常、事件存在差异分类，不能默认同帧同事件就是同实例。

本次 RGB 审核先看过标签/元数据，属于助手探索性分析。不能回填成第一轮盲答，也不能升级严格 registry。G2 需要另外安排精确两图→锁答→四图→锁答→完整证据的流程，并在独立子集上双审；新短输入集合只能在第一轮给出。

## 6. 本机运行与下一批工作

主任务在 tmux `p34_audit_20261010` 执行，已完成并退出 0；56.47 秒、峰值 RSS 134180 KiB（约 131 MiB）。保留 [run.sh](../../checkpoints/joint_frame_audit_20261010/run.sh)、run.log、progress.json、exit_code。源几何探针另在 tmux `p34_source_probe_20261010` 完成。会话任务结束后自动退出，不是还在后台训练。

继续审计的第二遍在 tmux `p34_deep_audit_20261010` 完成、退出0，21.17秒、峰值RSS82976KiB（约81MiB），逐帧导出见deep_frames.jsonl。22米复核在独立tmux小任务完成。deep_summary.json中历史字段 `verified_native_proceed_frames` 实际按问题累加（160），不同锚点数为129；以counterfactual_states和review_receipt的显式口径为准，不将多实例问题重复算作帧。

单进程、两核亲和、nice 15、idle I/O、数值库单线程；主任务地址空间上限 6 GiB，可用 RAM 小于 6 GiB 暂停，磁盘预留 20 GiB，诊断输出预算 2 GiB、超时 45 分钟。均为本任务局部措施，未终止已有 CARLA/Python，未改系统 core 配置。主 worker 自身设置并核验 non-dumpable；没有改写或绕过生产 `run_guarded.sh` 的管道 core 拒绝逻辑。这些措施降低争抢，不构成全机绝不卡顿的保证。

当前无需复制四图库或重做百万帧生产。下一批按以下依赖推进：

1. **本机先做**：U-E7 状态域/补问缺口审计、几何门影响范围、U-E1 明暗/距离过滤分层、Phase3 全候选池标记与联合 split 清单。新任务用新输出目录、固定路线批次、tmux、低并发与回执；不得复用本次不完整候选作为正式训练集。
2. **需要审核输入**：代表窗口实例对齐与 G2 有限输入锁答。已看过的开发样本只能作诊断，不冒充独立批准池。六个缺 val 事件先做跨 Phase/Action 物理组排除再预约。
3. **服务器并行低成本任务**：当前 adapter 对已采训练例的分层预测及 E0 分段/内核/利用率记录；不必等待全量标签重建，也不必先续七轮。
4. **有依据后才改规则**：一次只改一个标定/准入条件，新合同和新产物，复核逐帧前后差异；再训练并做 Action 分层下游验收。

本轮仅新增本机审计产物和更新指南事实；未修改生产规则、原题库、训练采样、旧标签或 split，未启动 Qwen/CARLA 训练，未提交或推送 RGB/数据产物。
