# Phase3 v6 动作审计摘要（2026-09-10）

## 结论

`sft_new_loop_phase3_20260909_095810_4rgb_audit_bundle` 的 LoRA production
在本次离线问答测试取得 **470/767 = 61.28%** 严格整题正确率；同一测试集的
base production 是 **207/767 = 26.99%**，LoRA 增益为 **34.29 个百分点**。

这不是 CARLA 闭环路线完成率，也不是串联真实 Phase1/Phase2 输出后的端到端指标。
评测输入的道路结构和事件前提来自离线构造，且十个有效情境做了均衡抽样，不能代表自然
4Hz 驾驶帧分布。

本版使用 `current_wait_first_crossing_v6` 动作规则、
`sft_new_loop_phase3_high_level_action_v5_current_phase` prompt、4 张历史 RGB，
权重槽为 final step 9216。完整身份见审计包的 `bundle_manifest.json`。

**逐帧复核补充**：已实际查看 67 个错例窗口、54 条采集 run、948 张去重 RGB，
覆盖包内 44 个精选错例；详见 [RGB 联合审计](EVAL_REVIEW_20260910.md) 和
[逐 case 笔记](EVAL_RGB_REVIEW_20260910.jsonl)。已确认高速 R4 错前提与出口 lane-id
伪变道，也确认模型存在提前归位、首跨方向混淆和跳过等待。其余失分还包括时间阈值、
未来放行/近端导航信息不足和弱光。该抽检不是全数据噪声率估计，原始成绩保持不变。

**代码修复进展**：9月10日已按上述 RGB 证据修复采集门控、Phase3 旧标注适配、横向不确定窗口与行为预测 prompt，并实际重建 v7 数据。训练因本机缺完整 Qwen 权重尚未启动；本页 61.28% 仍是旧模型成绩。执行状态与限制见 [修复记录](REPAIR_20260910.md)。

## 评测范围与可比性

- 本次：767 题 = 640 个有效情境题（十个 context 各 64）+ 127 个 INVALID 题。
- 旧包：`sft_new_loop_phase3_20260907_095517_4rgb_audit_bundle` 为 768 题 = 640 有效题
  + 128 INVALID 题。
- 两版间已更换动作合同、提示词、物理 route 切分、开发路线隔离与部分标签；因此下文旧版
  对照仅用于定位趋势，**不得把分数差值归因于某一项修复，也不得作为严格 A/B 结论**。
  旧包的审计、逐帧证据与新合同变更见 `EVAL_REVIEW_20260907.md` 和
  `REPAIR_20260907.md`。

## 总体结果

| 指标 | Base production | 新 LoRA production | 新 LoRA audit prompt | 旧 LoRA production（仅趋势） |
|---|---:|---:|---:|---:|
| 严格整题正确 | 207/767，26.99% | **470/767，61.28%** | 436/767，56.84% | 304/768，39.58% |
| 有效情境严格正确 | 205/640，32.03% | **359/640，56.09%** | 338/640，52.81% | 186/640，29.06% |
| INVALID 严格正确 | 2/127，1.57% | **111/127，87.40%** | 98/127，77.17% | 118/128，92.19% |

同一 LoRA 从 production prompt 切换到 audit prompt 时，整题正确率下降
**34/767 = 4.43 个百分点**。这说明审计提示词仍会实质改变动作答案；audit 结果只能用于
诊断，不能替代 production 能力结论。

## 各 Phase3 event/context 表现

下表的每个有效 context 都有 64 个均衡样本。`相对 Base` 是本次同一测试集的百分点差；
`相对旧版` 仅为前述非严格可比的趋势参考。

| event/context | 新 LoRA 正确/64 | 新 LoRA | Base | 相对 Base | 旧 LoRA | 相对旧版 |
|---|---:|---:|---:|---:|---:|---:|
| SIGNAL_FAILURE | 53 | **82.81%** | 35.94% | +46.88 | 50.00% | +32.81 |
| JUNCTION_RULE_CONFLICT | 51 | **79.69%** | 65.63% | +14.06 | 43.75% | +35.94 |
| LEAD_BRAKE | 49 | **76.56%** | 50.00% | +26.56 | 28.13% | +48.44 |
| ONCOMING_INVASION | 41 | **64.06%** | 28.13% | +35.94 | 34.38% | +29.69 |
| UNSIGNALIZED_PRIORITY | 35 | 54.69% | 37.50% | +17.19 | 29.69% | +25.00 |
| DYNAMIC_CUTIN | 33 | 51.56% | **57.81%** | **-6.25** | 25.00% | +26.56 |
| STATIC_BLOCKAGE | 28 | 43.75% | 14.06% | +29.69 | 23.44% | +20.31 |
| POST_BYPASS_RETURN | 28 | 43.75% | 7.81% | +35.94 | 12.50% | +31.25 |
| RAMP_MERGE_EXIT | 22 | 34.38% | 14.06% | +20.31 | 29.69% | +4.69 |
| VULNERABLE_CROSSING | 19 | **29.69%** | 9.38% | +20.31 | 14.06% | +15.63 |

### 主要观察

1. **强项：信号、路口、前车制动。** `SIGNAL_FAILURE`、`JUNCTION_RULE_CONFLICT` 与
   `LEAD_BRAKE` 均超过 75%；前两者的 STOP / RESUME 判别较稳定。
2. **DYNAMIC_CUTIN 是本次唯一低于 base 的有效 context。** 该桶 21 个标注为 `NONE`
   的样本没有一个整题答对，但其中 **13 个只答 INVALID、8 个才是动作误触发**。
   前 13 个集中于同一 HighwayCutIn run 的 f44–56；逐帧 RGB 和源标注确认高速走廊被
   `light_hazard` 规则选成 R4，Phase3 补入 UE3 后保留了错误 RS 前提。因此不能将
   这 21 题全部归为 LoRA 过度触发动作。
3. **VULNERABLE_CROSSING、RAMP_MERGE_EXIT 是最弱的有效情境。** 前者主要被减速和横向
   联合决策拖累；后者对需要的动作整体偏保守、偏漏报。
4. **STATIC_BLOCKAGE 与 POST_BYPASS_RETURN 仍是横纵向联合难题。** 静态占道中
   DECELERATE 的命中较差；恢复/目标变道中，左右方向与 RESUME 的联合正确率不足。

## 动作级指标

横向动作只在被问及的 FULL_MANEUVER 样本上统计，未问的横向动作不当作 NO。

| 动作 | 新版 Precision | 新版 Recall | 新版 F1 | 旧版 Precision / Recall / F1（仅趋势） |
|---|---:|---:|---:|---:|
| DECELERATE | 61.74% | 54.12% | 57.68% | 20.93% / 5.33% / 8.49% |
| STOP | 79.31% | 86.10% | 82.56% | 40.72% / 58.06% / 47.87% |
| RESUME | 73.91% | 52.31% | 61.26% | 20.19% / 12.57% / 15.50% |
| LANE_CHANGE_LEFT | 71.43% | 57.47% | 63.69% | 54.67% / 53.25% / 53.95% |
| LANE_CHANGE_RIGHT | 72.06% | 62.82% | 67.12% | 62.50% / 64.71% / 63.58% |

需要关注的细项：

- `RAMP_MERGE_EXIT`：DECELERATE 仅命中 1/17；左、右变道分别命中 10/21、6/15，
  RESUME 命中 7/19。
- `POST_BYPASS_RETURN`：左、右变道分别命中 12/22、10/21，RESUME 命中 7/15。
- `VULNERABLE_CROSSING`：DECELERATE 命中 9/18；左、右变道分别命中 12/22、15/21。
- `DYNAMIC_CUTIN`：需分别统计 13 个错误 R4 前提下的 INVALID-only 失分与 8 个 `NONE`
  动作误触发；两者需要不同修复，不能合并解释。

## Guard 与部署结论

该包的 `production_ready=false`，生成 guard 未全数通过：

| Guard | 实际 | 门槛 | 状态 |
|---|---:|---:|---:|
| 有效情境 exact | 56.09% | ≥50% | 通过 |
| INVALID exact | 87.40% | ≥80% | 通过 |
| STOP recall | 86.10% | ≥80% | 通过 |
| 左右变道联合 recall | 57.47% | ≥60% | 未通过 |
| 真实 NONE exact | 49/103，47.57% | ≥50% | 未通过 |
| same-RS wrong-event exact | 1/6，16.67% | ≥50% | 未通过 |

其中 same-RS wrong-event 只有 6 个测试样本、3 条独立 route，统计覆盖本身很弱；它是明确
的失败信号，但不能据此估计全部同 RS 错事件的泛化率。

因此当前权重适合继续用作**离线误差分析候选**，不应标记为已通过生产/闭环部署门槛。

## 建议的下一轮优先级

1. 优先处理已经逐帧确认的监督冲突：高速 R4 前提、#255 出口 lane-id 伪变道，以及
   RE3 场景触发窗口与 active-ramp 文本的阶段一致性。#613 道路展宽处的横向真值另列待审。
2. 明确未来实际行为与“应该做什么”的目标差异；按首次横向跨线、当前等待、
   减速不停车、持续恢复四种时间阶段分层，优先补
   `RAMP_MERGE_EXIT`、`POST_BYPASS_RETURN` 与 `VULNERABLE_CROSSING` 的独立路线。
3. 对横向弱例继续做 RGB + lane-section 连通性核验；不能只用 lane-id 或远端 target point
   的横向符号作为左右变道真值。
4. 增加同 RS 错事件的独立 route 覆盖后再评估 INVALID；不要把当前 1/6 当成充分的
   能力估计。
5. 若要说明规则、提示词或数据清理的因果贡献，必须以同一物理 split、同预算、同 seed
   的配对消融验证；当前新旧包不具备该条件。本轮看过的 test 路线若用于下一轮开发，
   应登记开发集合并补新的未看 holdout；旧 test 保留为固定回归集。

## 证据入口

- 当前 bundle：`checkpoints/sft_new_loop_phase3_20260909_095810_4rgb_audit_bundle/`
- 当前 production 指标：`lora_production/metrics.json` 与 `lora_production/summary.md`
- 当前 audit-prompt 指标：`lora_audit/metrics.json` 与 `lora_audit/summary.md`
- 当前 base 对照：`base_production/metrics.json`
- 旧包复核：`EVAL_REVIEW_20260907.md`
- 新合同、标签修订和已知限制：`REPAIR_20260907.md`
- 本轮实际 RGB、源标注和代码联合复核：[EVAL_REVIEW_20260910.md](EVAL_REVIEW_20260910.md)
- 本轮逐 case 人工观察：[EVAL_RGB_REVIEW_20260910.jsonl](EVAL_RGB_REVIEW_20260910.jsonl)
