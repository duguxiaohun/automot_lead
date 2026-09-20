# Phase3 v18：RGB 复核、条件性动作因果与主要机动

本轮落实 [v17 独立复审](V17_INDEPENDENT_RGB_REVIEW_20260920.md) 的问题：隔离两段不可靠的 UE4 前提，
改写全部十类动作说明，并让两种题型明确速度动作与跨线动作可能先后发生。
KEEP 继续表示当前题域的动作阶段保持；新增边界诊断，不凭一个临界案例修改全局速度阈值。

新训练默认 `4rgb + choice`、`sft_new_loop_phase3_data_v18`、split seed `20260920`。
prompt 为 `sft_new_loop_phase3_high_level_action_v18_causal_maneuver`；输出标签协议沿用
`primary_choice_v3_choice_and_binary_keep`。需要重建索引并新训；旧 run 用原源码恢复。

## 1. 实际查看了什么

重新查看覆盖 UE1–7、RE2/3/5 的 **26 个片段、506 个帧面板**：复用此前 442 个面板，
额外补看 UE4 两例及夜间 UE3 的 **64 帧早期历史**，另放大关键原图确认道路和行人位置。
这些都是已经曝光并设为 train-only 的开发路线；本轮没有新 holdout。
查看前执行异常时长路线过滤，查看后核验 506 帧 RGB/meta 的 SHA256 与物理路线隔离。

逐片段结果：[rgb_causal_review_v18_20260920.jsonl](rgb_causal_review_v18_20260920.jsonl)。
文件记录原图/元数据哈希、帧范围、可见阶段、观察结论、因果证据边界及处理决定。
原来的 v15/v17 笔记保留当时结论，不倒写历史。
审阅者已看到旧标签，且使用未来帧核验离线标注，所以这不是盲审，也不是仅凭输入四图的模型能力评估。
额外历史和未来证据只用于审计，没有进入训练输入。

| 上下文 | 本轮可见阶段与动作说明重点 |
| --- | --- |
| UE1 前车制动 | 前车阻挡时等待；减速保留跟车距离和反应空间，前车拉开空间后再增速。 |
| UE2 静态绕障 | 接近、等待、跨线、跨线后继续通过分别成立；减速兼顾避碰与观察邻车/间隙，为可能绕行准备。 |
| UE3 动态切入 | 补看夜间切入前后过程；减速给切入车辆建立前向位置的空间，更晚的停车不回写当前 KEEP。 |
| UE4 行人/骑车人 | 同向骑车人也可形成通道冲突；目的强调运动趋势和通过间隔，不强写成横穿。两段前提问题见下。 |
| UE5 对向侵入 | 减速减少接近速度并留通过空间；对方仍在附近但空间已打开时也可增速。 |
| UE6 路口违规冲突 | 有优先权仍需避碰和判断路径交叠；四图本身未必能还原完整违规过程。 |
| UE7 信号故障 | 调速用于判断各进口交通和进入时机；故障事实由既有先验提供，不从单帧灯色推断。 |
| RE2 目标换道/恢复 | 既有 ID 不意味着已绕完；依历史区分出发、等待、跨线及恢复，不能固定向右。 |
| RE3 合流/驶出 | 调速用于匹配车流并准备间隙；沿连接走廊继续行进不一定有新跨线。 |
| RE5 无灯路口 | 区分停车要求、等待和释放；已经完成停车后起步，不因输入仍有停止标志重复标 STOP。 |

## 2. 有证据支持的标注修正

新增 [mapping_rgb_decisions_v18_20260920.jsonl](mapping_rgb_decisions_v18_20260920.jsonl)，
经 `source_mapping.mapped_contexts()` 精确隔离如下 route/frame，且纳入映射合同哈希：

| Scenario / route | 排除帧 | RGB 核验结论 |
| --- | --- | --- |
| `VehicleTurningRoute / Town06_Rep0_Town06_Scenario4_4_route0_01_10_10_47_36` | 130–162，33 帧 | 补看更早历史后，近处信号灯、停止线及路口渠化持续可见；源 R1 前提不成立。行人在对侧岛附近活动，不能确认是本车等待原因。 |
| `DynamicObjectCrossing / Town01_Rep0_Town01_Scenario3_2_route0_01_08_08_41_45` | 89–90，2 帧 | 行人在售货机附近、路缘内侧；补看此前历史也未确认占道及随后释放，不能把该段解释成已发生行人冲突后的恢复。 |

处理是排除已审前提区间，没有改造成 NO、KEEP、invalid，也没有凭这一段推广到整个 scenario。
区间外未审帧不自动改标。原始 LEAD/Phase1/Phase2 文件保持只读；排除作用于 Phase3 候选构建。
构建 manifest 的 `visual_label_risk_counts["mapping_excluded/rgb_quarantine"]` 记录数量。

同一 26 路线局部输入重建：候选 **1183 → 1148**，差集恰好为以上 35 帧。
其余候选的原始动作布尔值、精确轨迹证据、主要动作和 split 均逐项一致。
采样池改变，因此最终采样身份可能变化；局部索引仍为 384 行，其中 320 有效、64 invalid，78 行 KEEP。

## 3. 提示词的实际变化

`scene_context` 继续只放道路、事件和已有历史，不用动作目的反向补充场景事实。
[choice_semantics.py](choice_semantics.py) 的十类动作说明均采用连续的“条件 → 动作 → 作用”句子，
选择题和判断题共享同一份文本。例如 UE2：

```text
DECELERATE: As the obstruction restricts the path, slow to preserve collision clearance and reaction time while assessing adjacent-lane vehicles, approaching traffic and gaps, creating time to prepare a possible bypass.
STOP: When the obstruction or passing traffic blocks progress, stop or keep waiting to avoid collision while checking passing clearance and waiting for a usable bypass gap.
LANE_CHANGE_LEFT: When a usable bypass or recovery gap opens, cross the left boundary to pass the obstruction or regain the route lane, checking adjacent vehicles, oncoming traffic and passing clearance.
```

这些句子保留“减速/等待为何能帮助准备变道”，但不把目的当成每条采集行为的真实动机。
公共边界只说一次：场景本身不能证明动作原因、安全空隙或必然发生的机动。
KEEP 明确允许当前阶段内的速度调整；机动域要求没有新的跨线或明确速度阶段变化，
纵向域只回答速度，不能从 KEEP 推断车道保持。持续等待仍为 STOP。

选择题现在询问 `main upcoming action`，机动域明确：

```text
Choose STOP first; otherwise the first upcoming lane crossing takes priority over its accompanying speed change; otherwise choose the speed action or KEEP. Preparatory slowing may happen before the selected crossing.
```

判断题说明 `Speed and lane YES can coexist and may happen in sequence, such as slowing before crossing.`
纵向域选择题省去无关的跨线优先级和准备跨线说明。两种题型仍有显式 KEEP，invalid 仍仅判断题可输出。
没有重新加入 `Stop within the next 1.5 s`、未来采样数或阈值；四帧历史的时间标识仍保留。

开门绕障 f169 是这个区别的实证：f171 先减速到单帧近零速，f177 的地图车道身份才切换，f178 确认保持。
离线阈值触发约 +0.5 秒、首次跨线 +2.0 秒；不满足持续 STOP。
（同日逐帧勘误：此前文字写成 f176 / +1.75 秒；原 meta 和边界审计一直是 f177 / +2.0 秒，标签未改。）
选择题 LEFT、判断题 DECELERATE:YES 和 LANE_CHANGE_LEFT:YES 仍与原主要动作合同一致。
**这没有解决“逐阶段预测第一个反应”的问题**：当前任务选择主要机动，判断题也不输出动作顺序。
若未来改成最先执行的阶段，必须单独改标定、目标协议和评估，不能只换一句问法。

本轮并非全面缩短提示词。固定速度 6 m/s、目标 (40,-2)、无附加 history 的演示中，
system + user 按英文空白分词，UE2 choice **511 → 497**、binary **599 → 593**；
其余类因增加明确条件和 KEEP 边界，有些略长。该计数不含图像/回答，不是 Qwen token，
不表示注意力或训练效果已改善。完整 20 份演示保存在本地 `probe_output/v18_review_20260920/prompt_*.txt`。

## 4. KEEP 临界样本单独报告

新增 [audit_label_boundaries.py](audit_label_boundaries.py)，回读 meta 并核验索引动作，按场景和主要动作分桶：

- 距速度变化阈值的精确余量；固定 ±0.10 m/s 仅作为诊断带宽，阈值两侧都检查。
- 孤立增速/近停、单点降速、连续采样确认跨窗等情况。
- 速度保持窗结束后紧邻一秒内出现的明显变化；只用于说明边界，不回填未来动作。
- 速度判据触发早于首次跨线的样本；触发时刻不是主观反应或驾驶员意图的真值。

| 局部候选池 | 有效候选 | 主要 KEEP | KEEP 命中任一诊断 | 速度判据先于跨线 |
| --- | ---: | ---: | ---: | ---: |
| v17 | 1183 | 222 | 67 | 116 |
| v18 | 1148 | 220 | 67 | 109 |

这些诊断重叠，不能相加当作错误数，**也不是错标率**。没有按这些标记过滤、降权或重标。
例如 UE5 f64 降速约 1.8274578 m/s，阈值约 1.83344232 m/s，差 0.00598452 m/s，
KEEP 符合现有算法，却不能据此声称 RGB 呈现“近似匀速”。
该例随后涉及 Shoulder 身份，不能当作已确认的 Driving 车道变更；这是对此前模糊描述的澄清。

从 `AutoMoT/` 可对新生产索引运行：

```bash
python qwen3vl_local/sft_new_loop_phase3/audit_label_boundaries.py \
  --index checkpoints/sft_new_loop_phase3_data_v18/frame_index.jsonl \
  --output qwen3vl_local/sft_new_loop_phase3/probe_output/v18_boundaries
```

可加 `--cases /path/to/matching_eval_cases.jsonl` 连接同题同标签协议的评测结果，以 `all_ok` 分桶统计。
连接按 scenario/route/frame/context/所问RS，兼容顶层 `prompt_road_structure` / `rs` 和训练验证的
`prompt_spec.road_structure`，不回退 `true_rs`。缺少所问道路、道路字段冲突或重复题结果冲突均拒绝；
工具不会自行证明外部评测的模型和 prompt 合同一致，
使用者须提供与被审索引匹配的评测。没有匹配的 case 不计入正确率。本轮未运行 Qwen，`eval_bins` 为空。

## 5. 验证与后续使用

已完成：

- 410 项本机可运行 CPU 测试通过，4 项依赖 torch 的选择测试取消选择，另 3 个导入 torch 的模块未收集。
- 26 个已审片段原始标签重放通过；新版局部索引 384 行的 choice/binary 目标、解析、KEEP 及证据一致性通过。
  重放器读取的是历史 442 面板笔记，不应把它的计数当作本轮 506 面板总数。
- 新笔记 506 帧 RGB/meta 哈希、异常路线过滤及 train-only 身份核验通过。
- 1148 条保留候选逐条回读 meta，边界审计通过；35 条删除区间与人工规则完全匹配。
- `trajectory_action.py` 与 `primary_action.py` 源码哈希相对本轮开始未变。

本地证据位于 `probe_output/v18_review_20260920/`：`comparison.json`、`replay.json`、
`boundaries_before/`、`boundaries_after/` 和 `index/`。这些可再生产物不入库。
正式重建/训练方法见 [运行说明](SFT_NEW_LOOP_PHASE3_RUN.md)。

本轮未重建全量生产索引、未训练 Qwen，也未验证 GPU/DDP。
这里只确认了所审开发片段及代码合同；其他路线的事件区间、真实动作原因和临界 RGB 可辨性仍需继续核验。
`action_prior` 的旧 NONE、门控和原始五动作协议保持原样，不把 Phase3 KEEP 文本直接写入其外部预测接口。

## 6. 独立复审后的两项修复（同日）

依据 [v18 独立复审](V18_INDEPENDENT_REVIEW_20260920.md)，修复两项 P2：

1. RE5 的 RESUME 不再要求等待阶段结束。两种题型共用的新说明为：
   `As a usable crossing gap or forward space opens, gain speed to enter or continue through the junction while monitoring priority traffic, whether leaving a wait or already moving.`
   主条件是可通行空间，等待后起步和未停车的持续增速均被覆盖。原速度标定不变。
2. 边界审计兼容训练验证的嵌套道路字段；`eval_matching` 保存输入行数、重复行数、去重题数、
   匹配/漏配题数及已审索引中没有评测结果的题数。统计只连接实际通过过滤并完成审计的有效题，
   输入 eval 中被排除的 invalid/异常路线题也计入漏配。非空评测完全漏配时保存 `no_matches` 状态并发出明确告警；
   部分匹配、空文件、未提供评测分别标为 `partial`、`empty_cases`、`not_requested`，不混作正确率零。
   CLI 也打印匹配统计。审计输出版本为 `phase3_boundary_diagnostics_v2_eval_matching`。

复用报告中的 ControlLoss f37 复现输入，嵌套训练格式和顶层独立评测格式均匹配 **1/1**；
其中 `all_ok=True` 是合成的接口测试输入，不是模型成绩。
新增回归覆盖非停车 RESUME、各道路字段、禁止 true_rs 回退、字段冲突、重复题、部分漏配、全漏配告警和空文件。
**417 项 CPU 测试通过，4 项取消选择、3 个 torch 模块未收集**；仍未运行 Qwen/GPU。
局部重建的 **1148 条候选、384 行索引逐项与修复前一致**，原始速度及主要动作标定源码哈希也一致。
本轮没有新增人工 RGB 审阅。

保留 v18 名称及默认索引路径；RE5 文本改动使 choice/binary × 两种 RGB 模式的 prompt 哈希全部变化，
已有索引目录应通过构建器刷新 manifest 的 prompt 合同，新训练使用新文本，旧 run 仍用原源码。
本轮复现和重建保存在本地 `probe_output/v18_review_fixes_20260920/`，未改写上轮审计产物。

后续对三个任务边界的 RGB/meta/bbox 核验见 [V18_BOUNDARY_JUDGMENT_20260920.md](V18_BOUNDARY_JUDGMENT_20260920.md)：
补充了控制器约束对象、速度 KEEP 与制动响应的区别及跨线帧号勘误，没有改生产标签或提示词。
