# Phase4 v25：撤回弱证据标签、按实际输入审核、分层抽卡

用户对 v24 标签证据和规模问题的判断成立。此次优先撤回无法从实际输入确认的正例，完善方案 B；没有批准规则自动标注，没有启动正式训练。训练从449变为**443题、45条路线**，待审从533变为539。原435训练题逐行保留，val273/test186逐字节保持。

## 争议标签复核结果

| 原标签 | v25处理 | 依据 |
| --- | --- | --- |
| UE1 1030_0 f70 readiness YES | UNKNOWN待审 | `[66,70]`中前车微小变化不足以明确支持释放，bbox位移不能代替可见证据 |
| UE1 f71、f72 catchup YES | UNKNOWN待审 | 静态环境位移与自车后继状态证据偏弱；f72也保守撤回 |
| UE1 f73、f74 catchup YES | 保留2题 | 左侧同一静态灯杆基座在`[69,73]`、`[70,74]`明显向外移动，当前车道与前车关系清楚；没有算作readiness |
| UE4 Town07 3_1 f50–52 readiness YES | UNKNOWN待审 | 仍在路面，无法从画面核实完整轮廓到走廊边界的1米保守下界或分隔人行区；未改成NO |
| UE4 f47–48 NO | 保留2题 | 当前图行人仍占据近处行进走廊 |
| UE2 002000 f33–34 depart NO | 保留2题 | 右侧黄色出租车及前方黑车占据可见进入区域 |
| RE3 000839 f44–45 enter YES | 保留2题 | 可见近处左出口分支可用，两辆Mini仍沿右主线；后方与浓雾远处不获安全认证 |

v24增量现保留8题：**2 readiness YES、4 readiness NO、2 catchup YES**，另6题显式待审。新默认 `reviewed_state_pairs_v9.json` 保留旧v7的152条标注，替换其中附加的14条为本次审核结果；旧v8文件未覆盖。此次没有新增监督题。

## 实际输入证据进入编译、构建与加载

新增 `visual_review.py`，口径详见 [VISUAL_REVIEW_RUBRIC_V25.md](VISUAL_REVIEW_RUBRIC_V25.md)。这些数字是保守开发口径，尚未在独立样本上校准，不是驾驶安全阈值。

- 分别绑定2图`[f-4,f]`和4图`[f-6,f-4,f-2,f]`及其SHA。面板按模式分别渲染，补问不能借中间帧/未来帧证明后继状态。
- 每种模式须填写当前图中可定位的证据区域、观察及支持/未知处置。允许4图有支持而2图未知，数据构建按模式送入不同监督/待审集合，加载再核对。真实测试覆盖此情形。
- 自车前行类补问须记录同一相机、同一静态地标在共享4帧间隔的两个像素坐标（至少6像素变化），说明相机运动影响。完成/入道须在当前图定位可见几何；单独`successor_confirmed=True`不再足够。
- UE1 readiness释放须同时有可见前车变化（至少6像素）、当前近处通道证据，以及4帧间隔车头间距增长至少1米的辅助测量。编译时核对参与者ID、当帧几何来源SHA及实际测量差值；辅助几何不会替代RGB审核。
- UE4释放要求完整轮廓的横向余量保守下界至少1米并远离，或完全处于分隔人行区域，须定位可见边界。几何建议也改用1米缓冲：走廊外但缓冲内记未知，不能按“未过YES门槛”补NO。Town07 f50当前几何建议未知；f51/52仍可有局部正向建议，但实际RGB审核证据不足，依旧待审。
- 已知风险与新范围横向审核仍沿用原门槛。缺模式证据、错输入、错哈希、未来证据、相机拼接边界伪运动、错误目标/阶段及旧勾选式审核拒绝。

校验器检查的是**证据合同与可复核测量**，不是自动视觉真实性判定器；填写坐标或勾选项不能证明审核者看对了。本轮Codex已见旧规则建议，是规则作者复核，不能称独立盲审。旧冻结人工标签未被宣称已经重审满足新口径，原四事件评估仍限于其旧审核协议。

## 分层抽卡与独立评估边界

`review_queue.select` 按事件内的场景、Town、转移边、建议YES/NO/UNKNOWN组织候选，优先不同物理路线，再平衡各维度。每层只保留有限路线锚点，逐事件最终预算仍是硬上限。数量庞大的单场景不能靠更多帧淹没其它场景；无预算覆盖的层明确报告，不伪装成概率代表样本。

审核卡和空白表不带逐卡规则建议/排名/预填答案，建议答案只在分层统计里出现。未选中帧不生成NO。显式请求入口保留，可提供当前已知方向/目标，处理仍未实现自动实例发现的RE2/RE3与绕行边。

新增 `--split train|val|test`，固定路线分属不变。val/test只导出给独立审核者的待审包；开发编译器拒绝将其批准为监督，扩展评估须另行独立审核并更新评估协议。40条预留路线用于**生产者误差测量**，自动从模型评估抽卡中排除，不能同时用于模型选优。此次没有读取这40条的RGB，没有评估标签。

同17条训练路线1615帧真实重跑：3700个常规待审状态对，加4个UE7身份检索提示；具备历史长度的3444题经分层得到**79张卡、117个已观察层，其中45层未入选**。16张/事件只是本次开发预算，不能认为已满足全量规模；RE2/RE3自动实例仍无覆盖。未扫描生成其余全部训练路线的条件，更未生成全量可靠标签。

`preflight.review_capacity` 新增不同物理路线支持统计：每事件的事件轴转移及recover_follow，每种readiness答案至少20路线；val/test各事件至少5路线。重复帧、补问及训练重复不填缺额。旧公共纵向、分支和状态覆盖仍另查。数值门槛加入正式数据检查，探索性训练准入保持独立。

## UE7身份与此次逐帧RGB审计

本轮实看20个连续面板、**87张不同RGB**；其中7张再看原图，不重复计数。5条路线均已显式train-only，无新曝光组。

| 路线 | 实看帧 |
| --- | --- |
| HardBreakRoute 1030_0 | 64–74 |
| DynamicObjectCrossing Town07 3_1 | 41–55 |
| ConstructionObstacle 002000 | 27–34 |
| HighwayExit 000839 | 38–45 |
| CrossJunctionDefectTrafficLight 002053 | **0–44完整序列复核** |

002053存在可见灯色与逻辑解释的疑问，但“画面灯头一直绿色”不准确：第20帧可见黄灯，21帧高处有小绿光、侧灯不清，23帧绿灯；适用灯头与逻辑记录的精确匹配尚未确认。横穿黑车经过及自车随后前行，不足以证明“已知故障、不等修复、自行协商”。**本轮不批准它作为UE7正例。**

为防止Off/Unknown筛选漏掉这类路线，新增独立的`identity_review_hints`：已知故障类源场景只用于检索身份复核，在首个可构卡帧及适用控制清单变化时给出待审提示。本路线为10/21/27/38帧，经路线去重抽中1卡。提示不进入已确认实例或条件事实，原红灯不等于故障的规则保留。新UE7半自动审核要求适用灯头匹配、可见故障与普通信号等待排除，源场景名不能自动获准。

本轮完整复核的002053此前用户已经全看过，不增加全库已审独立序列或三ID覆盖格。继承192/278格、余约240条序列的缺口仍保留。

## 验证、产物与剩余工作

**653项专项测试通过**。2/4图真实默认pipeline check均通过，各**902题监督＋539题待审RGB**全部解码核验；旧435训练题及原533待审题保留，val/test逐字节保持。默认七轮world1/4事件1:1、cap8、恢复采样历史一致，第3轮覆盖443题。

合同v25/data_v25，默认标注v9，快照v11不变。冻结taxonomy/prompts/calibration/observation、状态机、安全准入、旧标注/holdout、采样/训练/选优源码SHA保持。全目录重建仍9712路线、1637680帧，split8166/765/781。须使用新候选、新数据、新run，旧材料不能只改hash复用。

40条抽检v5保留原路线原split，在未查看这些RGB前绑定本轮源码；待分配清单见 `twenty_fifth_independent_review_handoff_20261002.json`。这是准备好的审核任务，不是已有人审核，也未向任何人发送消息。

**formal_data_ready=false。** UE2 depart YES/return、RE2 enter、recover_follow、UE7 proceed YES及RE5充分正例仍缺，没有达到20/5路线规模。六事件独立评估、40条独立审核、可靠全量标签和完整4B/GPU/DDP/CARLA验收均未完成。上述独立人工审核需要另外的审核者，本轮不能用同一规则作者替代并自报独立准确率。

证据文件：`twenty_fifth_rgb_review_20261002.json`、`twenty_fifth_review_queue_20261002.json`、`twenty_fifth_review_decisions_20261002.json`、`twenty_fifth_audit_verification_20261002.json`。
本机产物：`/tmp/phase4_v25_work/candidates.json`、`data2`、`data4`、`proposals.jsonl`、`triage.json`、`pending_reviews.json`。

## 可复现入口

仓库根目录，输出必须使用新路径：

```bash
PYTHONPATH=AutoMoT /tmp/automot-qwen35-env/bin/python -m pytest -q AutoMoT/qwen3vl_local/sft_new_loop_phase4/tests
PYTHONPATH=AutoMoT python -m qwen3vl_local.sft_new_loop_phase4.review_queue \
  --proposals /tmp/phase4_v25_work/proposals.jsonl --split train --per-event 64 \
  --output /tmp/new_stratified_cards.json --template /tmp/new_blind_form.json --panels /tmp/new_mode_panels
# 填完逐模式实际输入证据后编译；pending/uncertain不生成二元标签，deferred可记录为UNKNOWN。
PYTHONPATH=AutoMoT python -m qwen3vl_local.sft_new_loop_phase4.review_queue \
  --queue /tmp/new_stratified_cards.json --reviews /tmp/new_blind_form.json --output /tmp/new_reviewed_annotations.json
```

全训练路线候选生产可使用 `privileged_producer --candidate-pool <新全量清单> --splits train --output <新路径>`，不传`--route-list`；此次仅17条回归实跑，不能把此命令的可用性说成已经跑完全量。val/test待审包同理从对应split生产，再以`review_queue --split val|test`分层导出，须由独立审核者处理，不能纳入开发调参。
