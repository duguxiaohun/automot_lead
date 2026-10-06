# Phase4 v20：正常跟车恢复与可见范围许可

本轮核实第二十轮审计报告并修复代码；没有新增训练标签、完整逐帧序列审阅或正式模型训练。当前只能运行有限题库探索，不能据此宣称全流程已经学会或部署安全已验收。

## 对审计结论的判断

1. **RECOVER 到正常跟车缺边属实。** 冻结 taxonomy 的 settle 只能 APPROACH→FOLLOW；proceed 从 HOLD/APPROACH/FOLLOW 恢复到 RECOVER，原运行流程缺少直接进入普通跟车的路径。不能用 renewed_restriction 伪造新限制绕过去。
2. **前向 RGB 不能证明后侧间隙安全属实。** 本地 eval_carla 三相机配置为 yaw −54.5/0/+54.5、各 FOV 60，横拼 1152×384，不提供正后方观测。但“部署模型一定回答 YES”尚未实测；本轮修复的是信息边界和执行准入，不能宣称解决了真实后向感知。
3. **蠕行是未支持的能力，不能直接据专家起步授权。** 代码的 YIELD 与纵向轴独立：APPROACH 可 DECELERATE，FOLLOW 可 KEEP；并非所有 YIELD 都 STOP。不过已确认 HOLD 在冲突未解除时缺少受限前移能力。1304_11 的报告说明这一缺口，不能证明任何时刻前移都安全。本轮保留 STOP，不增加未经标定的 creep 动作；后续需停止义务完成证据、当前可用前移空间、禁入冲突边界及规划器限速/距离能力。
4. **数据与评估缺口属实。** 旧训练题覆盖十类事件，独立 val/test 只覆盖 UE1/5/6/7；“只能代表四类”适用于评估，不能当作训练集只有四类。UE4 pass 的 readiness YES 3、catchup YES 16 仍存在。缺少正例不能靠补问、事件均衡采样或工程测试补足。
5. **源标签与视觉可观测性的风险合理，但不能泛化成整类误标结论。** 本轮把报告中的实例身份、夜雨、小目标等疑问接入逐帧复审；未用其自动生成 NO/YES。10–15 像素高度不能直接证明“不足一个 token 就不可学”，还需检查实际 processor 缩放和模型输入/识别效果。

## 正常跟车路径

活动入口 `route_context.episode_edges` 增加 `recover_follow: RECOVER → FOLLOW`，要求当前已形成稳定移动跟车间距。流程可为：

`YIELD/HOLD → proceed + 执行确认 → PROCEED/RECOVER → recover_follow → PROCEED/FOLLOW → complete → DONE/FOLLOW`。

每次观察最多推进当前适用的边，不能在同帧连跳。stable 仍用于恢复到无约束稳定行进；同时回答 stable/recover_follow 或跟车恢复与 hold/renewed_restriction/re_yield 相矛盾时拒绝推进并复核。DONE 仍可保留 HOLD/APPROACH/RECOVER，待独立纵向轴结束。新边也用于几何完成后的 DONE/RECOVER。

活动提示词、条件编译、标定、覆盖检查都能识别新边。原 `taxonomy.py/prompts.py/calibration.py/observation.py` 及冻结评估计划 SHA 保持；新边不借用旧 holdout 协议证明评估有效。当前新边的真实监督为零，没有从 complete 标签复制出“已学会跟车恢复”。

## 可见范围与外部安全许可

`depart/enter/return` 的实际消息明确：只判断当前相机覆盖内的可见冲突，不能断言后方/侧后方无车。模型仍只输出 YES/NO，安全凭据不进入提示词。`pass/complete` 的几何完成和独立纵向约束保持原有含义。

`Episode.advance(..., maneuver_clearances=...)`、`Phase4Loop.tick`、序列回放统一执行安全检查。RGB YES 必须另有当前帧规划器或 BEV 安全模块的许可，才能接受机动：

- policy=`current_full_corridor_clearance_v1`；绑定 instance_id、frame_id、edge、目标走廊、方向、segment_id 和 route_context_id。
- source 只能 planner 或 bev_safety，提供非空 evidence_id、严格布尔 clear 和 rear_side_coverage_confirmed=True。
- 缺失或 clear=False：不产生新机动许可，进入不确定状态，保留已有 STOP/DECELERATE。后续新证据可恢复，持续不确定仍沿用超时复核。
- 过期、错实例/目标/分段、重复或格式异常：拒绝，整次调用原子回滚。
- 外部许可不是执行确认，仍需已有执行回执；仅设置 execution_committed=True 或提交执行回执不能绕过安全检查。未执行许可在下一观察过期，须重新提供当前安全证据。

`maneuver_safety.binding(ep, edge, frame)` 只构造绑定字段，**不会生成许可**。真实适配器必须先完成全机动走廊及后侧检查，再填入证据和 clear；本轮没有实现或验收这个 BEV/规划器适配器。demo 与测试中的 synthetic 许可仅服务合成流程，不可接入实车/CARLA。进入后的持续避碰仍由规划器/跟踪器负责。

回放的横向真值还要求 `visible_truth_scope=visible_maneuver_conditions_v1`；旧全间隙真值未声明新范围时按未知删失，不拿来计算新的可见条件延迟。回答、接受、执行三种延迟仍独立统计。

## 旧数据如何处理

改变问题范围后不能保留旧监督假装自动兼容。`visible_scope.py` 要求 `visible_scope_review`：当前 rubric 的 policy、reviewer/evidence_id、全部因果输入帧的 SHA、逐帧 observation 和 observed_until、visible_conditions_reviewed，以及当前帧明确复核且与标注一致的 target。构建与加载均检查；缺证据的旧 depart/enter/return 题保留为 UNKNOWN 待审，不改原人工标注文件、不把未知补成 NO。

两种 RGB 重建均为：

| 项目 | v19 | v20 |
| --- | ---: | ---: |
| train | 563 / 45 物理路线 | 435 / 42 物理路线 |
| val | 273 / 13 路线 | 273 / 13 路线 |
| test | 186 / 10 路线 | 186 / 10 路线 |
| 待审题 | 405 | 533 |
| 缺转移支持格 | 366 | 415 |

128 道旧监督题退出训练；已待审的相关横向题也增加可见范围复核原因。新缺额包含45个跟车恢复支持格及4个受旧标签隔离影响的格；12分段格、9环岛格仍缺。原 UE2 return/RE3 enter 的 NO 也移入待审，不能继续报告旧11/9作为新 rubric 支持；其新 YES/NO 都为0。UE7 proceed 仍为0/4。新 admission 报告列出事件/物理路线范围、逐转移 readiness/catchup、RECOVER 当前状态支持和六类缺失评估事件。

新增15个物理组显式 train-only，实际候选划分仍为8166/765/781、9712路线/1637680 RGB帧。登记10条路线的风险/复审要求；仅146_1的18→19突变在本轮复看原图确认。2858_0及两条行人消失仍标未确认，未混入确认异常计数；其它条目是后侧不可观测、实例身份、夜雨或蠕行等复审需求。所有条目均不授予训练标签。

用户报告3条完整、7条局部、5条未审。本轮只复看2张原图，未新增完整序列；未重算覆盖格，保留继承192/278格及约240条待审，不把15条已渲染算作15条已审。

## 验证与版本

任务 v20、横向/新边 prompt v8、活动标定 v7、快照 v10、单独训练默认 data_v20；需新候选、新数据和新run。旧adapter/快照使用原源码，不改旧合同hash复用。原标注v7、四个冻结规则文件、采样、训练、选优、Phase3/Action默认均未在本轮修改。

专项529项通过，含真实本地题库构建/加载、真实小模型已有回归、新增状态流程与安全许可回归。实际2/4RGB流水线check均通过，各894监督题和533待审题全RGB身份核验通过；保留训练题逐行不变，val/test与v19逐字节一致。world1/world4七轮累计均340→428→435，第3轮全覆盖，事件1:1/cap8和历史序列化恢复一致。详细输出差异见 `twentieth_audit_verification_20261001.json`。所有检查都是CPU/数据或合成控制器验证，未训练或验收真实4B/GPU/DDP/CARLA。

`trainable=true` 仅保留有限子集探索准入；`complete_coverage=false`、`formal_data_ready=false`。先完成新范围标签复审、RECOVER关键转移与另外六类事件的独立评估，再推进剩余逐帧审计。正常跟车、后侧安全、蠕行和正式闭环效果都不能用本轮测试通过替代验证。
