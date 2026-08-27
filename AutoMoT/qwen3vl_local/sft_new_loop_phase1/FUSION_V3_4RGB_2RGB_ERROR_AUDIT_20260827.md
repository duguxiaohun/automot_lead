# Fused Phase1+Phase2 v3：4RGB / 2RGB 结果与逐帧错例审计（2026-08-27）

## 1. 审计对象与结论

本审计只分析以下两个已训练 adapter 的正式 test bundle，并据此形成下一版代码和 prompt
修改。没有根据 scenario 名称或标签文本猜图像内容；第 5 节列出的结论均来自打开原始
stitched RGB 后逐帧查看。

- `checkpoints/sft_new_loop_phase1_20260827_095804_4rgb_audit_bundle`
- `checkpoints/sft_new_loop_phase1_20260827_100333_2rgb_endpoints_audit_bundle`

核心结论：

1. 融合训练相对 base Qwen 是显著有效的：两种 RGB 模式的 production 联合 exact 都从约
   26% 提升到约 70%。模型已经学会八任务联合输出，不是整体失效。
2. 相对 2026-08-24 的旧融合结果，当前 v3 的联合 exact、Phase1-only exact 和
   Phase2-only exact 均下降；相对独立 Phase2 augment，主要退化在 hierarchical。
3. 逐帧 RGB 不支持全面重写或放宽现有 prompt。VULNERABLE/RS5 的若干 FN 在图中没有清楚
   witness，另有暗帧 FP；追随这些 GT 加规则会制造更多幻觉。
4. 存在一个明确 prompt 缺口：fused `hierarchical_probe` 会问 `RS_HIGHWAY`，原 v3 却没有
   渲染它自己的定义。普通双黄线/双向城际道路在 4RGB 和 2RGB 错例中都被误判为高速。
5. 训练只给 focus 主答案语义 loss。两套 eval 中约 18% 样本出现“focus 正确但请求的副行
   仍有错”。但 RGB 同时确认 Phase1 HIGHWAY 与 Phase2 RS 标签存在少量局部冲突，因此不应
   回退到所有副行 1.0 loss。
6. 下一版采用最小修订：保留 v3 主规则；只补 `RS_HIGHWAY` RGB 定义、加强 audit 输出格式；
   focus 维持 1.0，非 focus 主答案默认给 0.1 的类别平衡语义 loss。

这里的“下一版”是新 prompt/训练合同，必须重新训练后才能判断是否提升；本文不会把代码
修改直接写成已经获得的模型分数。

## 2. 两个 bundle 不是严格配对实验

| 项目 | 4RGB bundle | 2RGB endpoints bundle |
|---|---:|---:|
| RGB 输入 | 原始索引 `[0,1,2,3]` | 原始索引 `[0,3]` |
| adapter | 4RGB `best_generation`, step 64000 | 2RGB `best_generation`, step 48000 |
| git commit | `6c7b21e7...` | `ed8ca42f...` |
| git 状态 | clean | dirty（manifest 已记录） |
| train / val / test rows | 581051 / 38207 / 67630 | 581505 / 38106 / 67758 |

所以 4RGB 与 2RGB 的差值同时混入训练步数、代码版本和数据行变化，不能解释成纯 RGB 帧数
因果。后续矩阵实验必须从同一 commit、同一 frame index、同一采样 seed 起跑。

## 3. 当前 v3 production 结果

### 3.1 Base → LoRA

| RGB 模式 | 模型 | 联合 exact | Phase1-only exact | Phase2-only exact |
|---|---|---:|---:|---:|
| 4RGB | base | 26.855% | 68.262% | 31.445% |
| 4RGB | LoRA | **70.801%** | **85.059%** | **78.711%** |
| 2RGB endpoints | base | 26.172% | 65.039% | 31.641% |
| 2RGB endpoints | LoRA | **70.312%** | **82.812%** | **77.637%** |

两种 LoRA production 的严格格式率均为 100%。4RGB 联合 exact 只比 2RGB 高 0.488 pp，
在上述非配对条件下不足以宣称 4RGB 一定更好。

### 3.2 八个均衡 focus 任务

| focus | 4RGB F1 | 2RGB endpoints F1 |
|---|---:|---:|
| HIGHWAY | 99.225% | 96.970% |
| STATIC_OBSTACLE | 90.909% | 87.603% |
| VULNERABLE | 83.478% | 86.207% |
| TRAFFIC_LIGHT_ABNORMAL | 93.443% | 92.562% |
| RS1 | 68.376% | 59.813% |
| RS2 | 82.051% | 78.947% |
| RS4 | 88.189% | 95.238% |
| RS5 | 81.739% | 86.726% |
| Phase1 focus Macro F1 | **91.764%** | **90.835%** |
| Phase2 focus Macro F1 | **80.089%** | **80.181%** |

最稳定的是 Phase1 HIGHWAY；最弱的是 RS1，尤其 2RGB 的 YES recall 只有 50%。VULNERABLE
仍主要是漏检，不是大规模乱报：4RGB/2RGB focus YES recall 分别为 75.0%/78.1%。

### 3.3 Phase2-only variant exact

| variant | 4RGB | 2RGB endpoints |
|---|---:|---:|
| all_random_order | 72.852% | 76.172% |
| subset_random | 83.203% | 77.344% |
| hierarchical_probe | 85.938% | 80.859% |

hierarchical 仍是当前三类里最高，但相对旧独立 Phase2 augment 的 97.1% 明显下降，且
`RS_HIGHWAY` 相关 false positive 在人工 RGB 中可复现。

## 4. 与旧记录的可比结论

### 4.1 原 Phase1 独立训练

来源：`sft_loop_phase1/SFT_LOOP_PHASE1_EVAL_REPORT_20260813.md`。

| 模式 | 原 Phase1 Macro F1 / exact | 当前 fused Phase1 focus Macro F1 / Phase1-only exact |
|---|---:|---:|
| 4RGB | 86.03% / 84.77% | **91.76% / 85.06%** |
| 2RGB endpoints | 86.62% / **85.16%** | **90.84%** / 82.81% |

当前融合模型的均衡 focus Macro F1 更高；4RGB 四问 exact 基本持平略升 0.29 pp，2RGB exact
下降 2.35 pp。分任务看，融合显著改善 STATIC 和灯异常，但 VULNERABLE 仍低于原报告，
不能说 Phase1 所有能力都提升。

### 4.2 原 Phase2 augment checkpoint-20000

来源：`sft_loop_phase2_augment/CKPT20000_EVAL_ANALYSIS.md`。旧结果为总 exact 82.8%，
all/subset/hierarchical 分别 76.5%/81.1%/97.1%。当前 4RGB Phase2-only 总 exact 78.71%，
三 variant 为 72.85%/83.20%/85.94%。因此 subset 略升，但总分、all-random 和尤其 hierarchy
下降。旧评测有 6144 cases，当前为 1024 cases，方向可参考但不是严格同集显著性检验。

### 4.3 2026-08-24 旧 fused 结果

来源：`FUSED_PHASE1_PHASE2_EVAL_ANALYSIS_20260824.md`。

| 指标 | 旧 fused | 当前 4RGB v3 | 当前 2RGB v3 |
|---|---:|---:|---:|
| 联合 exact | **77.54%** | 70.80% | 70.31% |
| Phase1-only exact | **89.75%** | 85.06% | 82.81% |
| Phase2-only exact | **84.47%** | 78.71% | 77.64% |

当前 v3 相对旧 fused 约下降 7 pp 联合 exact。v3 的随机 Phase1 输出顺序、focus-only loss、
训练数据/步数和代码版本都发生过变化，不能只把下降归咎于某一句 prompt；这也是本次只做
小改、要求重训 A/B 的原因。

## 5. 真实 RGB 逐帧审计

方法：4RGB case 查看全部四帧；2RGB case 先按模型实际输入看首尾帧，再查看原始四帧中的
中间帧，判断是否存在“模型未看到的瞬时证据”。以下是有代表性的 10 个序列，不是对全部
错例重新标注。

| 模式 / case | 模型错误 | 逐帧可见证据与人工判断 | 对修改的约束 |
|---|---|---|---|
| 4RGB `PedestrianCrossing/.../0025-0028` | GT VULNERABLE/RS5=YES；预测 NO，并把 RS1 判 YES | 雨雾局部道路，可见横道/道路开口，但所谓行人至多是极小且不可确认目标；RS5 的局部控制关系也弱。此序列不足以支持放宽 YES。 | 保留“crosswalk alone / unreadable object 不是 witness”。列入标签/可见性复核。 |
| 4RGB `DynamicObjectCrossing/.../0130-0133` | GT VULNERABLE/RS5=YES、HIGHWAY=NO；预测 VULNERABLE/RS5=NO、RS1=YES | 清楚的分隔多车道受控走廊，有车辆但没有可读行人/骑行者，也没有局部路口。GT 的三项可见语义都可疑。 | 不追 GT 扩写 prompt；应审标签投影和事件时间边界。 |
| 4RGB `DynamicObjectCrossing/.../0011-0014` | VULNERABLE FP | 四帧近乎全黑，无法辨认弱势参与者。 | 现有“darkness/unreadable object 不是 witness”正确，不能再加想象性目标规则。 |
| 4RGB `MergerIntoSlowTraffic/.../0059-0062` | focus RS4 正确，但副行 RS1 FP | 虽暗但护栏、分隔、多车道受控高速走廊连续可见；HIGHWAY=YES、RS1=NO 有 RGB 支持。 | 不是缺规则；支持给请求中的非 focus 行小权重监督。禁止加 `HIGHWAY ⇒ RS1=NO` 硬约束。 |
| 4RGB `InterurbanActorFlow/.../0090-0093` | HIGHWAY/RS_HIGHWAY FP | 雨夜普通双向城际道路，双黄线/对向车道与普通道路边缘可见，没有受控高速拓扑链。 | 直接支持新增 `RS_HIGHWAY` 的双黄线/普通双向道路 NO 边界。 |
| 4RGB `ParkedObstacleTwoWays/.../0077-0080` | HIGHWAY/RS_HIGHWAY FP，GROUP/RS2/STATIC FN | 普通无分隔双向地面道路；对向车逐帧接近，蓝车固定占据右侧走廊。不是高速。 | 再次支持 `RS_HIGHWAY` 补定义；STATIC/RS2 已有规则，不继续堆 prompt。 |
| 2RGB `EnterActorFlow/.../0000-0001` | RS1 FN；GT 同时 HIGHWAY=YES、RS1=YES | 首尾帧均清楚显示隔离护栏、分隔车行道和连续高速走廊；模型 HIGHWAY=YES、RS1=NO 与画面一致，GT 的跨阶段组合冲突。 | 证明不能加入 Phase1/Phase2 硬一致性 loss，也不能让副行使用 1.0 统一压制。 |
| 2RGB `ParkedObstacleTwoWays/.../0100-0103` | RS1 FP、RS2 FN | 四帧极暗，仅能弱看到中心线、路边结构和前车；不足以可靠判断当前是否需借用/让行对向空间。 | 不为暗帧加新正类规则；保留不确定时依赖清楚拓扑 witness。 |
| 2RGB `HardBreakRoute/.../0020-0023` | HIGHWAY/RS_HIGHWAY FP，GROUP/RS1 FN | 四帧为雨雾普通无分隔双向道路，对向车辆沿相邻车道接近，没有 median、ramp、gore 或受控出入口链。 | 直接支持 `RS_HIGHWAY` surface-road 反例定义。 |
| 2RGB `ParkingCrossingPedestrian/.../0054-0057` | VULNERABLE FN | 行人从左侧逐帧进入道路，首尾帧都可见，属于真实视觉漏检，不是 2RGB 恰好漏掉中间帧。 | 现有 prompt 已要求扫 sidewalks/crosswalk/每帧；优先改善训练监督/视觉能力，不重复堆同义规则。 |

代表性 case JSON 位于两个 bundle 的：

- `audit_lora_production/rs_highway_fp/`
- `audit_lora_production/group_fn/`
- `audit_lora_production/vulnerable_fn/` 与 `vulnerable_fp/`
- `audit_lora_production/rs1_fn/` 与 `rs1_fp/`

原始 RGB 均来自 `lead_data/<Scenario>/<route_id>/rgb/<frame>.jpg`；审计没有修改这些只读数据。

## 6. focus 与副行诊断

每套 production eval 有 1024 cases。focus 总准确率约 87%；在 focus 已正确的样本中：

| 模式 | focus 正确 | focus 正确但至少一个其它请求主行错误 | 比例 |
|---|---:|---:|---:|
| 4RGB | 891 | 163 | 18.29% |
| 2RGB endpoints | 892 | 168 | 18.83% |

这不是说所有副行都差：副行通常自然 NO 更多，普通 accuracy 反而偏高。问题是当前训练把
非 focus 主答案值 token 全设为 0，只用低权重格式 token 学会“输出这一行”，没有在当前图像
和当前组合里直接纠正其 YES/NO。旧 fused exact 更高也说明完全忽略副行可能过于激进。

修订不能走另一个极端。数据 manifest 已记录少量 Phase1 HIGHWAY 与 Phase2 R3 不一致，
人工审计的 `EnterActorFlow` 又确认其中至少部分冲突在 RGB 语义上真实存在。因此采用：

- focus 主行：基础语义权重 1.0；
- 非 focus 主行：默认基础语义权重 0.1；
- hierarchical `RS_HIGHWAY/GROUP`：基础语义权重 1.0；
- 每个 metric 的 YES/NO 类别权重按上述“有效基础质量”计算，而不是按原始副行数计算。

这样副行会得到弱纠错信号，但大量自然 NO 和局部标签冲突不会重新压过 1:1 focus 桶。

## 7. Audit prompt 结果不是纯问答能力

| 模式 | audit strict exact | answers valid | evidence/contract valid | 只取首次合法答案块的语义 joint exact |
|---|---:|---:|---:|---:|
| 4RGB LoRA | 33.789% | 51.172% | 44.043% | 71.191% |
| 2RGB LoRA | 57.715% | 82.617% | 80.957% | 70.312% |

当前 parser 已正确地区分普通答案与 `EVIDENCE_*`，不是旧 parser 把 evidence 当答案的 bug。
raw output 的典型问题是：先输出一组合法答案，随后 evidence 行丢失 `EVIDENCE_` 前缀，或在
证据开始后再次重复普通答案行。首次答案块重算接近 production，说明主要退化是 evidence
合同遵循，而不是分类语义整体崩溃。parser 必须继续严格，不能靠宽松解析掩盖格式问题。

因此 v4 只补一句明确纪律：每个 evidence 行必须保留完整前缀；证据开始后不得再次输出答案
行；最后一个 evidence 后停止。该修改能否提高 contract valid 仍需新 adapter/新 eval 验证。

## 8. 已实施的 v4 最小修改

1. `prompts.py`
   - prompt 名升级为 `sft_new_loop_phase1_phase1_phase2_combined_v4_rgb_audited_rs_highway`；
   - hierarchical 时渲染独立 `RS_HIGHWAY` 定义；
   - 用本次 RGB 中的普通双黄线、无分隔城际道路、暗/雾/单护栏反例收紧高速 YES；
   - 加强 audit evidence 前缀和停止规则；
   - 未改变其它主问题定义。
2. `train.py` / `train.sh`
   - 新增 `--non-focus-semantic-loss-weight` /
     `NON_FOCUS_SEMANTIC_LOSS_WEIGHT`，默认 0.1，可设 0 复现 focus-only；
   - 类别平衡改为按有效基础质量计算；
   - adapter config、run manifest、balance report 持久化新合同；
   - balance 新增 `semantic_answer_base_mass`。
3. `test_refined_contract.py`
   - 覆盖 RS_HIGHWAY RGB 边界、audit 输出纪律、0.1 副行权重、0 权重兼容和派生行 1.0。

## 9. 下一轮必须如何验证

1. 固定同一 commit 和同一 frame index，分别训练 4RGB 与 2RGB；不要继续拿两个不同数据
   manifest 做纯 RGB 因果比较。
2. 先做短程 A/B：`NON_FOCUS_SEMANTIC_LOSS_WEIGHT=0` 对 `0.1`，其它参数、seed 和采样完全
   相同。不要同时继续扩 prompt。
3. checkpoint 选择同时看：联合 exact、Phase1-only、Phase2-only、all-random、hierarchical、
   RS1 YES recall、VULNERABLE YES recall、RS_HIGHWAY FP 和格式率。
4. audit 需要分开报告首次答案语义与 strict evidence contract；contract 提升不能替代
   production 指标。
5. 对本节标为“标签/可见性可疑”的 route 建轻量复核名单，核对 frame-level RGB 标签投影和
   事件时间边界；在人工确认前不根据这些 case 自动改标签，也不扩 prompt。

验收门槛建议：production 格式率保持 99% 以上；联合 exact 至少恢复到旧 fused 77.54%
附近；hierarchical Phase2-only 明显高于当前 85.94%/80.86%；同时 VULNERABLE 和 RS1 的
正类 recall 不因副行 loss 再次塌成全 NO。若 0.1 A/B 没有提高联合 exact，先回退该权重到
0，再单独保留已由 RGB 证实的 `RS_HIGHWAY` prompt 修复。
