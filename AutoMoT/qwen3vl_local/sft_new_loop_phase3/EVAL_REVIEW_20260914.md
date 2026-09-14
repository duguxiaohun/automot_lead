# Phase3：20260911 binary / choice 逐帧审计与修订

本次问题同时来自上游道路/事件标定、提示词时间口径和模型行为预测。**没有证据支持为了迎合模型答案修改速度阈值**。已完成全部 LoRA production 错例的 RGB 复核，并实现精确标签隔离、源规则修复、共享时序提示词及 v9 索引重建；尚未训练新模型，不能宣称准确率已提高。

## 交付与覆盖

- [指标比较](AUDIT_COMPARISON_20260914.md)：历史总体、各动作 P/R/F1、十类 context、同题 binary/choice、base 对照、训练曲线与 guard。
- [逐帧笔记](EVAL_RGB_REVIEW_20260914.jsonl)：202 个题目，122 个 run，2,566 个不同的 `(scenario, run, frame)`；另有扩展上下文和四个源规则对照。每题记录真实输入 SHA256、观察帧号、原 GT、binary 预测、归因、处理决定和证据路径。
- [可搜索 RGB 审计页](probe_output/rgb_review_20260914/index.html)：显示同题 choice 预测与其原编号，点击拼图可看大图；引用既有图片，不另复制证据。`case_index` 默认使用 binary 编号。
- [覆盖核验](probe_output/rgb_review_20260914/index.summary.json)：binary 179/179 个错例、choice 65/65 个错例，重叠 51 个，错例并集 193/193；另有 9 个两种适用输出均正确的对照。标签为 `control_*` 的笔记也可能是 binary 正确、choice 错误，不能靠分类名统计正确对照。
- [四个源规则 RGB 对照](RULE_RGB_CONTROLS_20260914.jsonl)：只确认所列四帧的场景，不代表全部自动修订帧均经过人工视觉确认。

先复核 96 个定向样本，再补齐剩余 106 个 production 错例，原图片复用。每题第一行是模型实际四帧输入；后续逐帧 RGB 仅作离线取证，旧拼图到 +3s，新拼图到 +3.25s。+3.25s 只确认 +3s 的首次越线，不扩大动作预测窗口。#42、#537 另看原始 +3.25s 图确认。暗图保留原始亮度；不能由“看不清”推导事件为 NO。

#21、#211、#65、#474、#532 在路线尾部缺少部分远期图，笔记与拼图明确标记；这些纵向题均有完整 2s 速度窗口，不能声称看过不存在的远期帧。以上是 **production 错例全覆盖**，没有声称覆盖 base 的全部错例或 audit 输出专属错例，也不是数据集随机噪声率调查。

生成证据前检查原测试 191 条 run：没有异常时长或缺失路线；202 题实际输入 RGB SHA256 和原 meta 速度与包内记录一致。输入四张 1152×384 stitched RGB 仍保持原合同，审计拼图的缩小显示不改变训练输入。

## 指标变化说明

| 指标 | 上一版 | 本次 | 解释 |
|---|---:|---:|---|
| binary production exact | 518/765，67.71% | 373/552，67.57% | -0.14 pp，基本持平；holdout 不同 |
| binary valid exact | 62.81% | 62.17% | -0.64 pp |
| binary INVALID exact | 92.80% | 94.57% | +1.77 pp；同道路错事件仅 2 题 |
| binary NONE exact | 65.06% | 61.43% | -3.63 pp |
| DECELERATE F1 | 55.86% | 50.91% | 主要短板，召回 47.06% |
| STOP F1 | 87.18% | 85.71% | 略降 |
| RESUME F1 | 62.14% | 65.41% | 小幅上升 |
| LEFT / RIGHT F1 | 69.94% / 68.61% | 85.96% / 86.40% | 横向明显改善 |
| audit strict exact | 58.17% | 64.13% | 格式失误减少，但非独立视觉正确性证明 |

同一批 306 个单动作题：binary 203/306=66.34%，choice 241/306=78.76%，净增 38 题；52 题修好、14 题改坏。choice 排除了 92 INVALID、70 NONE、84 联合动作题，不能与完整 binary 直接横比，也不能替代完整动作合同。binary 联合动作仅 40/84=47.62%。

binary 最终 test 未通过 DECEL recall≥50% 的 guard；训练 best 通过 val 不等于 test 通过。choice 通过的是单动作任务 guard。旧版和本版使用不同 holdout，且 binary/choice 训练筛选、预算、保存步不同，不能把差值归因于某一个 prompt 改动。

## 逐帧归因：动作标签与提示词

### 停车确认必须完整落在 1.5s 内

原数值规则要求两帧连续 ≤0.5m/s；两帧都须在 1.5s 内。#1、#11、#150、#336 等第一次近停在 +1.5s，下一帧 +1.75s 才确认，应为 DECEL。#354、#540 等只有孤立近停，也不能改 STOP。反例 #2、#352 在 +1.25/+1.5s 两帧已确认，应为 STOP；#544 在 f23/f24 停稳，模型只答 DECEL 是漏停车终态。

提示词旧版没有同样明确地说明“两帧都在窗内”，容易把首次触及阈值或孤立低点当停车。现已在 binary/choice 共用 `SPEED_ACTION_RULES` 明确这点，并保留 STOP 优先、当前帧可计入、继续等待优先随后释放。

**普通 DECEL/RESUME 的窗口仍是 2s，不是 1.5s。** #344 在 +1.75/+2s 才减速，仍然有效；#442 在 +1.75/+2s 两次增速达标，仍为 RESUME。#549 直到 +2s 才第一次增速达标、确认在 +2.25s，不能标 RESUME。

### 历史状态、未来行为和环境建议混淆

- #195、#306、#365、#520、#473、#538：输入历史曾减速/等待，但当前已释放并在未来增速。不能复述历史动作。
- #119、#47、#446、#250、#398：当前和下一帧仍在等待，稍后释放不改变当前阶段 STOP。
- #375、#489、#407、#434、#396、#411、#509：看到障碍、行人或“恢复”情境就提前填变道/制动；需要预测采集行为和时间，而非输出一般安全建议。
- #31、#327、#426、#449：邻车绕行或通过，不等于自车跨越边界。
- #42、#405、#415、#470、#480、#511：方向可能正确但漏速度动作；binary 需要保留联合标签，不能用 choice 的单选结果替代。
- #529：首次右跨线发生在 +3.25s，超出 3s；#42 则在 +3s 首次跨线，+3.25s 用于确认。两者不能混为同一边界口径。

当前 prompt 明确从最新输入帧开始预测，先核验前提；history 和 situation 只提供证据，不是要求必然执行的动作。保持 NONE 与 INVALID 的不同含义，车道规则继续排除弯道、道路连接、邻车运动和已经发生的跨线。没有引入逐帧动作答案或未来 RGB。

这些是有 RGB/数值支持的错误模式；不能仅凭错误模式证明“全部由提示词导致”。四帧过去 RGB 对未来 2–3s 行为并不总有唯一确定答案，尤其在遮挡、晚出现的参与者和采集控制抖动中。#266 的 f0 四图重复、f1 参与者才出现；#143/#347 的起步光照/参与者突变单列待审，不为了改善分数擅自重标。

## 逐帧归因：源标定规则与精确修订

### 1. 持续灯态把连续道路恢复成 R4

#521 DynamicObjectCrossing f42、#522 HazardAtSideLane f71 的源记录明确是 `R1 → R4`、原因 `stable_meta_light_with_untrusted_xodr`，而 RGB 是连续街道、地图对齐误差分别约 2239m/3880m。重复灯态并不证明当前局部有交叉口。

`collector.py` 现要求每个恢复帧有局部几何支持：near_junction、bbox junction hint，或可信 XODR 的 is_junction；trigger、停车、持续灯态不单独构成道路几何。统计记录真正恢复的帧列表，避免把整个区间计为恢复。

Phase3 读取既有源文件时，仅在源明确记录上述弱恢复、原标签来自 R1 且没有局部几何支持时，撤回到原 R1，并去掉伴生 R-E4，保留独立 U-E4/R-E2。这表示撤销缺乏依据的恢复步骤，**不是从“缺证据”推导任意 R4 必定为 R1**。

### 2. 两个 R5 局部路段与 RGB 不符

| binary 题号 | run / anchor | 连续复核范围 | 决定 |
|---|---|---|---|
| #40 | VehicleTurningRoute / Town04…Scenario4_116… / f8 | f5–20 | 连续弯道，源 trigger/静态拓扑不足以证明当前无灯路口，隔离 RS 为 UNKNOWN |
| #459 | PedestrianCrossing / Town13…105_0… / f14 | f11–26 | 连续街道，不能由行人横穿直接认定局部路口，隔离 RS 为 UNKNOWN |

完整身份与帧段见 [annotation_repairs_20260914.json](annotation_repairs_20260914.json)。没有把这些题改成模型预测答案，也没有扩展到整条路线。后续 #218、#479、#532 的局部路口在雾/暗图中不清晰，证据不足，仍为 `UNRESOLVED_KEEP_SOURCE`。

### 3. DynamicObjectCrossing 的 hazard 并不等于车辆切入

#353/#215 同 run f70/f71，补看 f45–81：行人在 f51–66 横穿、f69 已离开，自车前方无切入，随后蓝车正常对向通过。#464/#124 同 run f15/f16，前方灰车随弯道正常行驶，黑车在对向通过，行人在更晚的 f30 后才出现。两条源 U-E3 仅由 scenario + vehicle_hazard 提出，无 cutin 距离或 brake_cutin 支持。

[event_rgb_exclusions_20260914.jsonl](event_rgb_exclusions_20260914.jsonl) 只排除这两条 run 的 U-E3：f67–82、f12–27，共 32 帧。保留其它事件，不由未来行人状态补造当前 U-E4，不自动生成 INVALID 负例。

同样的 hazard-only 字段也出现在 #113/#287 真正黑皮卡侵入样本，所以通用规则仅标记 `dynamic_cutin_actor_unverified` 待审，**不会把所有这类样本改 NO 或删除**。这部分尚未逐一视觉复核的候选仍可能含噪声，状态和原因已进入审计/构建记录。

### 4. lane_id 跳变与物理越线不一致

#325 ParkedObstacle f44，f49/f50 在同一 roundabout connector 上 lane4→3，但 f40–56 RGB 中边界连续，无法确认真实跨线。新增 transition49 的精确 lateral uncertainty；覆盖该未来转移的窗口隔离，已经在历史中的转移不再影响后续窗口。不直接改成 LEFT=NO。

### 5. 被进一步证据澄清的疑似噪声

#80/#278/#413/#496 HighwayCutIn 的 SUV：扩大到 f90–129、看原图后，f114/115 仍在侵入，f120 才稳定；源 Phase2 的 YES96–119 有支持，保留。

#15/#21/#211/#302/#330 CrossJunctionDefectTrafficLight：夜间拼图初看像无路口；扩大上下文后，f0 可见信号路口、f4–20 红灯、f24–32 绿灯，原 meta f40–52 为 junction。保留 Phase1 已审计的故障事实，不由暗图看不到灯认定前提错误。

最初 16 个疑点中 9 个被进一步证据澄清；最终 202 题中，188 条 KEEP、5 条保留来源待审、2 条 RS 隔离、2 条撤回弱 R4、4 条 U-E3 排除、1 条横向隔离。KEEP 表示本轮不改变来源，并非每题所有语义均已获独立证明。这些比例不能当作数据集标注噪声率。

## 实现、全量影响与验证

代码修改涉及 `evidence_guards.py / collector.py`、Phase3 `annotation_repair.py / source_mapping.py / build_dataset.py / prompts.py`、审计工具及两条 launcher。所有读取既有 collection 结果的修订在 Phase3 映射层执行，原 collection 大文件和原始 audit bundle 未回写。新 collector 防止以后重新生成时重复相同弱恢复错误；Phase1/2 已训练权重不会因此自动修好。

全源 CPU 审计覆盖 42 scenario、7,241 run、914,466 帧。包括旧规则的合计改变 16,162 帧；各原因可叠加，不能直接相加当独立帧：

| 原因 | 帧数 | 本次性质 |
|---|---:|---|
| 撤回无局部几何支持的弱 R4 恢复 | 13,459 | 新通用源规则；自动命中，未逐帧看完这 13,459 帧 |
| 精确 RGB 反证的局部 RS 隔离 | 32 | 新，两条连续审计区间 |
| hazard-only signalized road 隔离 | 1,582 | 旧规则继续生效 |
| RGB mainline 修订 | 13 | 旧规则继续生效 |
| generic trigger-only ramp event 清理 | 1,053 | 旧规则继续生效 |
| U-E3 参与者待审 | 912 | 新 review 标志，保留候选，不算自动改标 |
| 精确 U-E3 排除 | 32 | 新，两条连续审计区间，单独统计 |

四个追加源规则对照 RGB 与撤回弱 R4 相容，但不能外推所有命中均已被视觉证明。完整源审计在 `checkpoints/sft_new_loop_phase3_data_v9/annotation_audit/`。

已构建 `AutoMoT/checkpoints/sft_new_loop_phase3_data_v9`，split seed=20260914：

| 项目 | train | val | test |
|---|---:|---:|---:|
| 索引行数 | 13,500 | 348 | 468 |
| 每个有效 context | 1,125 | 29 | 39 |
| INVALID | 2,250 | 58 | 78 |
| 独立题数 | 13,387 | 348 | 468 |

train 有 113 个重复呈现；最大输入帧复用 7，val/test 为 1。3,272 run 对应 3,271 个物理路线组，划分交叉为 0。此次暴露的旧 test 全部 191 个物理组加入 train-only，连同既有开发组共 709；不能仅换 Rep 编号绕过隔离。新 val/test 不属于这些开发组。

- [静态索引审计](probe_output/rgb_review_20260914/v9_index_audit.json)：35,392 张 RGB 文件检查，按完整精度重算动作无不一致；不能用展示时的三位小数重算边界。
- [原 meta 回读](probe_output/rgb_review_20260914/v9_raw_index_audit.json)：14,316 行、3,272 run，原始速度不一致 0、有效动作不一致 0、物理路线划分交叉 0。这证明实现和数据合同一致，不等于所有语义已人工核实。
- Phase3 回归测试涵盖 STOP 确认窗口、共享提示词、弱信号恢复、真实/未知切入保留、精确排除、横向未来窗口及开发路线隔离。执行 `PYTHONPATH=AutoMoT pytest -q AutoMoT/qwen3vl_local/sft_new_loop_phase3`：139 passed。
- 新 mapping hash：`d90ef1eca13fe31987690ad13f0a8cf0cb994a0256c84da118ba7dec6a1ca62d`；prompt `v8_shared_temporal_rules`；动作数值规则仍为 v7。

## 下一次运行与剩余限制

运行命令见 [SFT_NEW_LOOP_PHASE3_RUN.md](SFT_NEW_LOOP_PHASE3_RUN.md)。新 prompt/hash/索引与旧 adapter 合同不兼容，需要新训练；不应把旧权重直接配新 prompt 的输出当作重训结果。远端从源码重建 v9，完整 eval 使用 `CASES_PER_BIN=0`；binary 和 choice 分别训练、配对比较单动作交集，完整 binary 另报 NONE/INVALID/联合动作。

后续提升必须在未参与本次 RGB 修改的新 holdout 评价。当前同 RS 错事件负例只有两条独立测试路线，难以验证通用事件拒绝能力；尚未逐一看完的 912 帧 U-E3 待审、自动撤回弱 R4 的全体语义，以及真实 Phase1/2 串联误差均是仍需量化的限制。已交付的是可追溯修订和验证结果，不是新的训练或闭环成绩。
