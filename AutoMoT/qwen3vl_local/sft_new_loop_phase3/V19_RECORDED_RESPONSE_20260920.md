# Phase3 v19：保留运动阶段，补充风险响应证据

本轮实际修改了十类场景共享的提示词、候选/索引审查字段和边界审计器。
新训练默认 `4rgb + choice`、索引 `sft_new_loop_phase3_data_v19`，prompt 为
`sft_new_loop_phase3_high_level_action_v19_response_aware_keep`。

## RGB 复核与处理依据

在上一轮 [三个边界复核](V18_BOUNDARY_JUDGMENT_20260920.md) 的基础上，
本轮复看 **5 个片段、91 个帧面板、4 条路线**，覆盖此前 12 条
`KEEP + brake=True + vehicle_hazard=True + target_speed=0` 的上下文题。
这 12 题对应 11 个物理 anchor；一个路口帧同时有 UE6、UE7 两题。
全部沿用已曝光 train-only 路线，查看前执行异常时长过滤，无新增 holdout。
这是有既有标签背景的开发复核，不是盲审；91 是面板次数，不是新增独立测试帧。

| 场景与路线 | 复看片段 | 关键观察与处理 |
| --- | --- | --- |
| InvadingTurn / Town12…1826_0…09_13_43 | f41–57、f65–81 | 对向车沿受限通道接近并通过，自车仍移动；保留 f44、f68 的 KEEP，单列控制响应。 |
| InvadingTurn / Town13…1448_0…04_10_29 | f58–80 | f61–67 有零目标速度与制动请求，但没有实际持续停车；不能凭命令改成 STOP。 |
| Accident / Town05…001774…11_33_13 | f71–87 | 绕行跨线后的推进与调速，事件继续不意味着需要再跨线；保留 f75 KEEP。 |
| CrossJunctionDefectTrafficLight / Town05…002078…17_55_06 | f26–42 | 横向车辆通过、自车仍推进；f30 两个上下文都记录风险响应，不能抹成“无风险”。 |

完整 route、逐帧 RGB/meta 指纹、所审题及判断见
[rgb_response_review_20260920.jsonl](rgb_response_review_20260920.jsonl)。
其中 3 张面板为这次针对性补看生成，另外 2 张复用已有面板；证据图留在忽略的 probe_output 中。
信号失效仍来自已接受的场景先验，图像中的灯色本身不能独立证明信号失效。

## 1. 十类 KEEP 的释义

choice 与 binary 共用 `choice_semantics.action_description`。
去掉 KEEP 独有的“已经有足够安全空间”式前提，保留各类跟车、绕障、让行、合流和路口监测的具体内容。
共享补充句为：

> Brief braking can occur within this stage; KEEP does not imply that traffic risks have cleared.

这与原来的“KEEP allows speed adjustments within the current stage”连续表达。
机动域继续要求没有新的跨线或明确速度阶段变化；纵向域不声称保持车道。
明显减速仍按原规则为 DECELERATE，实际持续等待仍为 STOP；该句不把所有制动都归入 KEEP。

UE2 的减速解释保持：

> As the obstruction restricts the path, slow to preserve collision clearance and reaction time while assessing adjacent-lane vehicles, approaching traffic and gaps, creating time to prepare a possible bypass.

其 STOP 仍说明避碰、观察通过空间、等待可用绕行间隙；RE5 RESUME 继续涵盖未停车的持续增速。
`scene_context` 保留道路、事件与历史事实。控制器字段、未来时间和阈值都不进入模型输入。
因果句是动作在相应条件下的作用，不声称每帧驾驶主体确实在观察或已经确认安全。

## 2. 所有新候选与索引保存 action_review

新增 `action_review.py` 由构建器与审计器共用，独立于原 `action_evidence`，字段包括：

- `controller`：当前制动/风险标志、目标速度、记录的约束对象及其名单成员关系。缺失或非法值保留 null，不能补成 false/0。
- `motion_milestones`：原速度判据触发、首次跨线及确认时刻；标出该速度节点是否实际被纵向判据选中。
- `logged_limiter_change`：连续、已知记录中约束对象首次改变；字段缺失不能写成释放。
- `positive_target_after_zero`：零目标速度后首次连续两帧目标速度为正，两次都限在审查窗口内；不把它称为“已经安全”。
- `flags`：KEEP 伴随制动请求、KEEP 伴随车辆零目标速度请求、速度先于跨线。
- `review_only=True`、`purpose_status=conditional_not_verified_intent`：明确证据用途和因果边界。

对象不在 scenario 名单中不等于已经证实是背景车；名单也不能证明最终控制的唯一原因。
invalid 题保留物理来源的审查证据，`source_context_id` 指向原上下文，
`applies_to_prompt_context=False`，不能把来源动作当作 invalid 题答案。
该字段不参与标签、采样、loss 或提示词；构建合同包含源码，旧候选缓存需重建。

UE2 开门案例 f169 的实际审查节点为：

| 节点 | 相对 f169 的时间 | 标签关系 |
| --- | ---: | --- |
| DECELERATE 判据触发 | +0.50 秒 | 被选中的纵向变化 |
| RESUME 幅度达到、随后确认 | +1.50 / +1.75 秒 | 后续节点，原纵向判据仍选更早的减速 |
| LEFT 地图跨线、随后确认 | +2.00 / +2.25 秒 | choice 的主要动作 |

choice 仍为 `LANE_CHANGE_LEFT`；binary 为 `DECELERATE:YES`、`LANE_CHANGE_LEFT:YES`，其余动作与 KEEP 为 NO。
这里的时刻只存在于离线审查产物。节点是原判据的触发顺序，**不是完整动作阶段分割真值**；
不把后续增速节点直接改成 binary RESUME:YES，也不把主要 LEFT 说成最先发生的反应。

## 3. 审计器能直接检索并评测风险响应样本

`audit_label_boundaries.py` 保留原 `bins` / `eval_bins`，另加 `response_bins` / `eval_response_bins`，
继续支持训练验证记录中的 `prompt_spec.road_structure` 和匹配/漏配计数。
新索引存在 `action_review` 时逐题与原 meta 重算结果核对，发现不一致直接报错。

局部 1148 候选中共有 220 KEEP：原运动边界仍为 67 题，新响应桶有 40 题带制动请求，
其中 12 题同时有车辆风险和零目标速度。12 题中只有 4 题命中原运动边界桶。
桶之间重叠、连续 anchor 相关；这些数值都不是错标率。
本轮 12 题全部补看 RGB，未将控制器命令强行改写为轨迹动作。

从 AutoMoT 目录运行：

```bash
python qwen3vl_local/sft_new_loop_phase3/audit_label_boundaries.py \
  --index checkpoints/sft_new_loop_phase3_data_v19/candidate_frames.jsonl \
  --output qwen3vl_local/sft_new_loop_phase3/probe_output/v19_boundary_audit
# --cases 可追加同一索引对应的评测 JSONL，报告各桶模型正确率。
```

## 验证、产物与尚未解决的范围

- **442 项 CPU 测试通过**。环境无 PyTorch，忽略 3 个依赖 torch 的测试模块，并 deselect 4 项相关测试；未运行 Qwen。
- 从 26 条既有开发路线重建 **1148 条候选、384 行索引**，十类齐全；与上版逐行比较，仅 `action_review` 和 `mapping_contract_hash` 改变。
- 原始动作布尔、完整速度证据、KEEP/invalid 标签及采样逐项一致；原 v8 标定和 STOP > 首跨 > 速度优先级未变。
- **704 次题型目标/解析回放**通过，并检查两种历史 RGB 模式的提示词不含新增离线字段；choice 按合同跳过 64 个 invalid，binary 保留。
- 边界审计从原 meta 重算全部 1148 条候选，核验新增审查字段；并复算 UE2 f169 实例的时刻和两种答案。

本地产物为 `probe_output/v19_response_review_20260920/`：`index/`、`boundary_audit/`、
`verification.json`、`ue2_prompt_examples.json` 和三张针对性面板；不入库、不上传。

本轮把前述三个边界变成可核查的数据字段和更准确的提示词，并没有消除阈值临界学习难度，
也没有宣称完成逐帧真实动机或完整动作序列监督。后续可用 response/边界桶评测实际模型误差，
再决定是否需要独立的阶段任务；不能凭少量样本改掉共享标定。

需要重建全量 **v19** 索引并新训；本轮只完成局部构建，没有启动生产全量构建或 Qwen 训练。
旧 run 用原源码恢复。`action_prior` 仍保持旧 NONE/门控协议，不直接消费 Phase3 KEEP 文本。

## 同日开训前修复：首次限速对象出现

已修复独立复审发现的 `None → ID` 遗漏：显式无对象参与首次变化比较；
anchor 或中途字段缺失/非法时仍中断证据链，不将未知当作无对象。
新增获得、解除、替换及未知值回归，**453 项 Phase3 CPU 测试通过**（4 项取消选择、3 个 torch 模块未收集）。
局部重建并对照 1148 候选/384 行索引，分别补齐 257/105 条对象出现记录；
UE3 f69 正确记录 `at_s=0.75, from_id=null, to_id=3739`。
其余审查字段、动作标签、采样和四组 prompt 合同不变，索引 mapping 哈希随修复更新。
构建/原 meta 复算证据见本地 `probe_output/v19_limiter_fix_20260920/`。

当前适合进入新训练与模型验证，不需要继续以措辞修改替代效果实验。
开训前在完整训练环境重建 v19 全量索引、运行 preflight 和 sampling-only；
本地无 torch，也没有全量 v19 生产索引，不能宣称 GPU、DDP 或全量划分已通过。
保留 v19 名称，旧 candidate cache 不能跳过新合同；旧 run 用原源码恢复。
本次修复只核验代码与原始 meta，没有新增 RGB 人工审计或模型训练。
