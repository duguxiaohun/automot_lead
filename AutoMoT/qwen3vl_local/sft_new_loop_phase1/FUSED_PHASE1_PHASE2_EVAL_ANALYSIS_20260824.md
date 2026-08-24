# SFT New Loop Phase1：Phase1 + Phase2 融合训练完整评测分析

> 分析日期：2026-08-24
> 评测 bundle：`AutoMoT/checkpoints/sft_new_loop_phase1_20260824_103639_audit_bundle`
> 模型：Qwen3-VL-4B-Instruct base + `sft_new_loop_phase1` LoRA
> LoRA 权重：`checkpoints/sft_new_loop_phase1_runs/latest/best_generation`
> checkpoint global step：`24000`
> 正式输入：`4rgb`，原始 history 索引 `[0, 1, 2, 3]`
> production prompt：`sft_new_loop_phase1_phase1_phase2_combined_v1`

## 1. 结论摘要

本次 Phase1 + Phase2 融合训练整体成功，没有出现多任务融合后整体能力下降。

核心结论如下：

1. 融合 LoRA 的 production 联合 exact 为 `77.54%`，相对 base 的 `26.37%`
   提升 `51.17` 个百分点；production 格式合法率为 `100%`。
2. 从逐例输出中拆开计算：
   - Phase1 四问单独 exact 为 `89.75%`，高于原 Phase1 4RGB LoRA 的 `84.77%`；
   - Phase2 当前变体问题单独 exact 为 `84.47%`，高于原 Phase2 augment
     checkpoint-20000 的 `82.8%`。
3. Phase1 的净提升主要来自 `STATIC_OBSTACLE`，F1 从 `73.79%` 提升到
   `87.93%`；`HIGHWAY` 和 `TRAFFIC_LIGHT_ABNORMAL` 也有小幅提升。
4. Phase1 的主要退化是 `VULNERABLE`，F1 从 `88.70%` 降到 `80.37%`，
   当前仍有明显的正例漏检。
5. Phase2 的 `RS2`、`RS4`、`RS1` 均有提升；`RS5` 小幅下降。
6. Phase2 的 `subset_random` 提升明显，但 `hierarchical_probe` 从旧模型的
   `97.1%` 降到 `91.80%`，说明融合训练对原本极强的层级决策树能力有一定稀释。
7. bundle 中 LoRA audit exact `58.98%` 不是实际语义能力下降，而是 audit parser
   的解析顺序缺陷导致大量合法输出被整条判为 invalid。宽松地仅提取 expected answer
   可得 `77.54%`；新 v2 分离解析器保留重复答案等语义严格性后为 `77.25%`，同时单独报告
   evidence 合同合法率 `97.75%`。两种口径都说明 audit prompt 没有实质破坏问答能力。
8. 当前训练没有跑满计划的 3 个 epoch，是因为观察到过拟合迹象后主动在
   `29090 / 110592` 停止并转入测试，不是训练异常退出。generation exact 在
   step-24000 达到 `81.64%`，随后 step-26000 为 `81.25%`、step-28000 降到
   `75.00%`；因此 step-24000 是有验证依据的 early-stop checkpoint。
9. 逐帧 RGB 复核与训练 work 重放发现：低召回不能只归因于 prompt。epoch-0 实际
   emitted label 严重偏 NO，例如 `VULNERABLE=14198:133258`、
   `RS5=18816:97514`、`TRAFFIC_LIGHT_ABNORMAL=9633:137823`（YES:NO）。
   这解释了当前“precision 很高、recall 偏低”的共同形态。
10. 已据此形成 v2 修订：只对当前不可见 focus 主任务施加 YES/NO 语义 loss，保留
    层级派生键监督；prompt 只加入逐帧证据支持的两遍扫描和三个窄边界复核；同时修复
    audit evidence 被误判为 answer 的解析顺序。该修订需要重新训练后评测，不能把
    本报告 v1 checkpoint 的分数当成 v2 效果。

综合判断：融合路线值得保留。它扩大了问法覆盖，同时带来中等幅度的整体净提升；
当前需要优先处理的是 audit 解析器、`VULNERABLE` 漏检、`RS1/RS5` 边界和评测采样口径，
而不是推翻融合 prompt。

---

## 2. 分析对象与数据来源

### 2.1 融合模型结果

本报告读取以下正式产物：

- `bundle_manifest.json`：base/LoRA、production/audit 四组总指标；
- `base_production/metrics.json`；
- `base_audit_prompt/metrics.json`；
- `lora_production/metrics.json`；
- `lora_audit_prompt/metrics.json`；
- 四组结果各自的 `cases_rank0.jsonl` 到 `cases_rank3.jsonl`；
- `adapter_metadata/adapter/sft_new_loop_phase1_adapter_config.json`；
- `adapter_metadata/run_root/train_run_manifest.json`；
- `adapter_metadata/run_root/train.log`；
- `adapter_metadata/run_root/train_eval_metrics.jsonl`。

### 2.2 原 Phase1 对照

主要对照：

- `sft_loop_phase1/SFT_LOOP_PHASE1_EVAL_REPORT_20260813.md`；
- 原 Phase1 4RGB production LoRA；
- 512 个评测 case；
- 四个固定输出：`HIGHWAY / STATIC_OBSTACLE / VULNERABLE /
  TRAFFIC_LIGHT_ABNORMAL`。

### 2.3 原 Phase2 对照

主要对照：

- `sft_loop_phase2_augment/CKPT20000_EVAL_ANALYSIS.md`；
- 原 Phase2 augment checkpoint-20000；
- 6144 个评测 case；
- `all_random_order / subset_random / hierarchical_probe` 三类问法。

### 2.4 可比性边界

以下差异意味着新旧结果不是严格 paired test：

- 原 Phase1、新 Phase1+Phase2 使用的数据索引和过滤条件并不完全相同；
- 新融合数据沿用了 Phase2 最新异常 route 剔除、full-frame RGB review 覆盖检查和
  visual-risk 默认过滤；
- 新评测每个 prompt 固定包含四个 Phase1 问题，并额外包含当前 Phase2 variant 的输出；
- 原 Phase1 exact 只要求四个答案全部正确；新联合 exact 最多要求八个答案同时正确；
- 原 Phase2 报告和新融合报告的样本规模不同；
- 新融合测试对八个 focus task 分别做 YES:NO 平衡，原报告的全局指标不一定使用完全相同的
  focus sampling 口径。

因此，本报告把结果分成三类：

1. 同一 bundle 内的 base → LoRA：最可靠；
2. 从融合逐例结果重算的 Phase1-only / Phase2-only：比联合 exact 更适合与旧模型比较；
3. 新旧任务 F1 对比：用于判断方向，但不能当作严格同样本显著性检验。

---

## 3. 融合代码和提示词设计分析

### 3.1 融合任务结构

每个样本固定询问四个 Phase1 visible-fact 问题：

- `HIGHWAY`
- `STATIC_OBSTACLE`
- `VULNERABLE`
- `TRAFFIC_LIGHT_ABNORMAL`

随后加入当前 Phase2 ROAD_STRUCTURE variant：

- `all_random_order`：RS1/RS2/RS4/RS5 全部询问并随机顺序；
- `subset_random`：随机询问其中一部分；
- `hierarchical_probe`：按 `RS_HIGHWAY -> GROUP -> DETAIL` 做层级拆问。

训练 variant 比例为 `4:1:1`，正式 eval/generation eval 为 `2:1:1`。训练默认
`FOCUS_BALANCE_COUNT=9216`，每轮构造 `147456` 个 sampled cases。

### 3.2 正向设计

#### 3.2.1 复用已验证的两套语义定义

融合 prompt 没有重新发明标签含义，而是：

- 从最终 Phase1 prompt 中提取完整 visible-fact 决策规则；
- 从最新 Phase2 prompt 中渲染当前 ROAD_STRUCTURE 问题和规则；
- Phase1 负责可见交通事实；
- Phase2 负责当前驾驶规则对应的道路结构。

这减少了重新改写 prompt 带来的标签漂移风险。

#### 3.2.2 显式处理 Phase1/Phase2 语义重叠

融合 prompt 增加两组关键约束：

1. `ROAD STRUCTURE PRIORITY`
   - limited-access highway/ramp；
   - opposing-lane sharing constraint；
   - local signalized junction；
   - local no-signal priority junction；
   - 最后才允许 `RS1=YES`。
2. `INDEPENDENT ANSWER CHECK`
   - Phase1 的 YES/NO 不得直接强制 RS 输出；
   - 每一行必须按照自己的视觉定义独立判断；
   - Phase2 定义对 RS 输出具有最终权威。

这对解决 `RS1` catch-all、道路事件干扰结构判断、Phase1/Phase2 逻辑泄漏都有直接价值。

#### 3.2.3 解决 HIGHWAY 键名冲突

Phase1 的 `HIGHWAY` 是人工审计后的可见 limited-access 事实；Phase2 hierarchical 中的
高速判断原本也叫 `HIGHWAY`。融合代码把后者改为 `RS_HIGHWAY`，避免同一个输出中出现
两个相同 key，也降低解析和监督冲突。

#### 3.2.4 问法增强抑制固定位置记忆

随机顺序、随机子集和层级问答让模型不能只学习固定四行的位置映射。融合结果中：

- production 格式合法率达到 `100%`；
- subset 未询问 RS 行泄漏率为 `0`；
- all-random/subset 没有异常 multi-YES pattern；
- subset Phase2-only exact 明显高于旧模型。

这些结果说明模型较好地学习了问题 key 与视觉语义，而不是只依赖固定输出位置。

### 3.3 设计风险

#### 3.3.1 prompt 和输出长度显著增加

新 prompt 同时包含：

- 完整 Phase1 四问长规则；
- 当前 Phase2 问题；
- 当前 Phase2 规则；
- ROAD_STRUCTURE 优先级；
- 独立回答约束；
- 最多八个答案行。

这会增加 instruction-following 和生成联合正确的难度。因此，不能直接把新联合 exact
`77.54%` 与旧 Phase1 四答案 exact `84.77%` 比较。

#### 3.3.2 两套高速语义仍可能竞争模型容量

虽然输出 key 已拆成 `HIGHWAY` 和 `RS_HIGHWAY`，prompt 也明确了各自定义，但二者仍共享
相似的高速、匝道、护栏和 controlled corridor 视觉特征。如果标签边界在少数帧上不同，
语言侧 LoRA 仍然需要学习两套相近但不完全相同的决策边界。

#### 3.3.3 只训练语言侧 LoRA

本次 adapter 配置为：

- `lora_vision_scope = off`；
- LoRA 注入语言模型各层 attention/MLP；
- base 视觉编码器保持冻结。

因此，本次提升主要表示模型更会：

- 读取已有视觉 token；
- 执行规则；
- 区分问法；
- 输出正确的结构化结果。

它不等价于视觉编码器对小目标、遮挡和低可见度获得了新表征。`VULNERABLE` 是最依赖
小行人、骑行者、画面边缘和遮挡细节的任务，恰好成为融合后的主要退化项，这与训练范围一致。

#### 3.3.4 多任务 token 监督并非简单的八任务等权

每个样本固定监督四个 Phase1 值；Phase2 根据 variant 监督 1 到 4 个细问，或三层层级值。
虽然 focus module 总量按 Phase1:Phase2=`1:1` 控制，但每个样本实际产生的答案 token 数不同。
训练损失中的答案值 token 与格式 token 权重也不同，因此“focus case 平衡”不代表所有问题获得
完全相等的梯度预算。

---

## 4. 融合 bundle 总体结果

### 4.1 官方联合 exact

| 模型/模式 | cases | 联合 exact | 说明 |
|---|---:|---:|---|
| Base production | 1024 | 26.37% | 原始 Qwen，答案行模式 |
| Base audit | 1024 | 24.12% | 原始 Qwen，额外要求 evidence |
| LoRA production | 1024 | **77.54%** | 当前正式可用指标 |
| LoRA audit（官方 parser） | 1024 | 58.98% | 被 parser 缺陷严重低估 |

LoRA 相对 base 的 production 联合 exact 提升：

`77.54% - 26.37% = 51.17 pp`

这是同一套 1024-case 评测中的直接对照，是本报告最可靠的训练有效性证据。

### 4.2 从逐例结果重算的任务族 exact

重算规则：

- Phase1-only exact：四个 Phase1 key 全部正确；
- Phase2-only exact：当前 prompt 中实际询问的所有 Phase2 key 全部正确；
- 不要求另一任务族同时正确。

| 模型 | 联合 exact | Phase1-only exact | Phase2-only exact | 格式合法率 |
|---|---:|---:|---:|---:|
| Base production | 26.37% | 66.60% | 32.13% | 99.51% |
| LoRA production | **77.54%** | **89.75%** | **84.47%** | **100.00%** |
| LoRA 增益 | **+51.17 pp** | **+23.14 pp** | **+52.34 pp** | +0.49 pp |

这个拆分说明：

- LoRA 同时学会了 Phase1 和 Phase2；
- 联合 exact 较低主要来自多行联合事件，而不是某一任务族完全失效；
- Phase2 不是靠 Phase1 的固定答案“带分”，其自身 exact 从 `32.13%` 提升到 `84.47%`。

### 4.3 按 variant 拆分

#### 4.3.1 联合 exact

| variant | cases | Base | LoRA | 提升 |
|---|---:|---:|---:|---:|
| all_random_order | 512 | 30.47% | **71.09%** | +40.62 pp |
| subset_random | 256 | 23.44% | **79.69%** | +56.25 pp |
| hierarchical_probe | 256 | 21.09% | **88.28%** | +67.19 pp |

#### 4.3.2 LoRA 内部拆分 exact

| variant | Phase1-only | Phase2-only | 联合 exact |
|---|---:|---:|---:|
| all_random_order | 88.67% | 78.91% | 71.09% |
| subset_random | 88.67% | 88.28% | 79.69% |
| hierarchical_probe | 92.97% | 91.80% | 88.28% |

all-random 一次需要输出四个 Phase1 + 四个 RS，总共八行，联合 exact 最低是合理的。
hierarchical 的输出更具结构约束，而且 Phase1-only 也更高，因此联合 exact 最高。

---

## 5. Phase1 与原模型对比

### 5.1 总体对比

原 Phase1 4RGB LoRA：

- Macro F1：`86.03%`；
- 四答案 exact：`84.77%`。

融合 LoRA：

- Phase1 focus Macro F1：`88.66%`；
- Phase1-only exact：`89.75%`。

方向性变化：

- Macro F1：`+2.63 pp`；
- Phase1-only exact：`+4.98 pp`。

### 5.2 逐问题对比

| Phase1 问题 | 原 4RGB LoRA F1 | 融合 LoRA F1 | 变化 |
|---|---:|---:|---:|
| HIGHWAY | 95.93% | **97.64%** | +1.71 pp |
| STATIC_OBSTACLE | 73.79% | **87.93%** | **+14.14 pp** |
| VULNERABLE | **88.70%** | 80.37% | **-8.33 pp** |
| TRAFFIC_LIGHT_ABNORMAL | 85.71% | **88.70%** | +2.99 pp |
| Macro F1 | 86.03% | **88.66%** | **+2.63 pp** |

### 5.3 融合模型 Phase1 confusion matrix

| 问题 | Accuracy | Precision | Recall | F1 | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|
| HIGHWAY | 97.66% | 98.41% | 96.88% | 97.64% | 62 / 1 / 2 / 63 |
| STATIC_OBSTACLE | 89.06% | 98.08% | 79.69% | 87.93% | 51 / 1 / 13 / 63 |
| VULNERABLE | 83.59% | 100.00% | 67.19% | 80.37% | 43 / 0 / 21 / 64 |
| TRAFFIC_LIGHT_ABNORMAL | 89.84% | 100.00% | 79.69% | 88.70% | 51 / 0 / 13 / 64 |

### 5.4 Phase1 结论

#### HIGHWAY

当前已经接近饱和：只有 1 个 FP 和 2 个 FN。融合 Phase2 的 highway/road structure
监督没有破坏 Phase1 高速判断，反而略有提升。

#### STATIC_OBSTACLE

这是本次融合最大的收益项。可能原因包括：

- Phase2 的 usable corridor / opposing constraint / junction structure 帮助模型更准确地追踪
  ego 可用路径；
- Phase1 prompt 强调 road-fixed、lane occupation 和四帧历史；
- 融合训练中更多道路结构变化降低了“附近物体 = 占道”的误判。

当前主要问题是 recall：仍漏掉 13 个正例，但误报仅 1 个。

#### VULNERABLE

这是最明确的退化项：

- F1 下降 `8.33 pp`；
- 64 个正例漏掉 21 个；
- 没有 FP，说明模型明显偏保守。

错误主要集中在：

- `VehicleTurningRoute`；
- `VehicleTurningRoutePedestrian`；
- `DynamicObjectCrossing`；
- `HazardAtSideLaneTwoWays`；
- `PedestrianCrossing`。

可能原因：

- 冻结视觉编码器对小目标的可辨性有限；
- 长 prompt 和多任务监督让模型更依赖道路结构主线；
- 行人/骑行者在侧视图边缘、路口转弯区或遮挡中的证据容易被忽略；
- 当前定义要求“decision-relevant”，模型可能把临近但未进入车道的 VRU 判得过于保守。

#### TRAFFIC_LIGHT_ABNORMAL

融合后小幅提升，且保持零 FP；13 个错误全部是 FN，集中在
`CrossJunctionDefectTrafficLight`。模型没有把普通红绿灯误报为异常，但在信号头较小、跨视角
对应关系不清晰时仍容易漏掉同一冲突区的 contradictory green witness。

---

## 6. Phase2 与原 augment 模型对比

### 6.1 总体和 variant 对比

| Phase2 指标 | 原 Phase2 ckpt-20000 | 融合 LoRA | 变化 |
|---|---:|---:|---:|
| 综合 exact | 82.8% | **84.47%** | +1.67 pp |
| all_random_order | 76.5% | **78.91%** | +2.41 pp |
| subset_random | 81.1% | **88.28%** | **+7.18 pp** |
| hierarchical_probe | **97.1%** | 91.80% | **-5.30 pp** |

融合的主要价值仍然是增强问法覆盖：

- all-random 略有提升；
- subset 泛化显著提升；
- hierarchy 仍然很高，但没有保持旧模型接近满分的专项表现。

### 6.2 逐问题 F1 对比

| 问题 | 原 Phase2 F1 | 融合 LoRA F1 | 变化 |
|---|---:|---:|---:|
| RS1 | 72.4% | **75.97%** | +3.57 pp |
| RS2 | 84.3% | **91.73%** | **+7.43 pp** |
| RS4 | 87.4% | **90.63%** | +3.23 pp |
| RS5 | **81.8%** | 80.70% | -1.10 pp |
| Macro F1 | 81.48% | **84.76%** | **+3.28 pp** |

注意：原报告这里是对应问题的 production 全局指标，新报告采用每个 focus task 的 64 YES +
64 NO 主问题指标。两者适合看方向，不是完全同口径复现。

### 6.3 融合模型 Phase2 confusion matrix

| 问题 | Accuracy | Precision | Recall | F1 | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|
| RS1 | 75.78% | 75.38% | 76.56% | 75.97% | 49 / 16 / 15 / 48 |
| RS2 | 91.41% | 88.41% | 95.31% | 91.73% | 61 / 8 / 3 / 56 |
| RS4 | 90.63% | 90.63% | 90.63% | 90.63% | 58 / 6 / 6 / 58 |
| RS5 | 82.81% | 92.00% | 71.88% | 80.70% | 46 / 4 / 18 / 60 |

### 6.4 Phase2 结论

#### RS1

RS1 仍是最难问题：31 个主问题错误，FP 和 FN 基本对称。错误主要集中在施工、动态穿越、
转弯、interurban 场景。它仍然容易在以下边界混淆：

- 普通 same-direction corridor；
- 双向/对向共享约束；
- 局部路口控制；
- limited-access 高速拓扑；
- 事件目标存在但道路本身仍属于 RS1。

虽然 F1 比旧模型提高，但距离稳定可用仍有空间。

#### RS2

这是 Phase2 最大提升项，recall 达到 `95.31%`。模型已经能较好识别对向车道和共享/受限走廊，
剩余问题主要是 8 个 FP：可能把路边停放车、中心线或道路狭窄过度解释为 opposing constraint。

#### RS4

precision 和 recall 都为 `90.63%`，表现最均衡。融合 Phase1 灯异常监督并没有把“灯异常”与
“信号灯控制路口”混为一谈，说明独立回答约束基本有效。

#### RS5

模型偏保守：precision `92.00%`，recall `71.88%`。错误集中在
`VehicleTurningRoute`，模型容易把可见 road mouth / local turning conflict 判成普通弯道或 RS1。

#### hierarchical_probe

融合后的层级指标仍然很强：

- `RS_HIGHWAY` F1：`96.45%`；
- `PLAIN_LANE_FOLLOWING_CORRIDOR` group F1：`82.05%`；
- `OPEN_SURFACE_PATH`：`92.86%`；
- `JUNCTION_CONTROL_ZONE`：`96.30%`；
- `LOCAL_RIGHT_OF_WAY_RULE`：`100%`；
- `CONSTRAINED_SHARED_SPACE`：`96.30%`。

hierarchical Phase2-only exact 的下降主要不是高速顶层失效，而更可能来自 plain-lane group
和 detail 联合正确率下降。融合训练扩大任务覆盖后，旧模型高度专项化的 hierarchy 优势有所稀释。

---

## 7. Audit prompt 结果与 parser 缺陷

### 7.1 官方表面结果

| 模型 | Production exact | Audit exact | 表面变化 | Audit format valid |
|---|---:|---:|---:|---:|
| Base | 26.37% | 24.12% | -2.25 pp | 100.00% |
| LoRA | 77.54% | 58.98% | -18.55 pp | 75.88% |

如果只读 summary，会误以为 evidence prompt 严重破坏 LoRA 的问答能力。

### 7.2 parser 的具体缺陷

audit 要求先输出答案行，再输出：

```text
EVIDENCE_<KEY>: <one short RGB cue; max 12 words>
```

模型经常生成：

```text
HIGHWAY: NO
EVIDENCE_HIGHWAY: NO
```

`EVIDENCE_HIGHWAY: NO` 对人类来说是格式正确但信息不足的 evidence；生成该 bundle 时的 v1
parser 处理顺序是：

1. 先用通用答案正则匹配 `KEY: YES|NO`；
2. `EVIDENCE_HIGHWAY: NO` 会先被匹配成答案键 `EVIDENCE_HIGHWAY`；
3. 该键不在 expected answer keys 中，于是设置 `extra=True`；
4. 最终把该 case 的所有 parsed answers 全部置为 `None`；
5. 只有答案正则没有命中时，代码才检查 evidence 正则。

因此，合法的 bare YES/NO evidence 会触发整条无效。

### 7.3 invalid 统计

LoRA audit 共 1024 个 case：

- invalid：247；
- 其中包含 bare `EVIDENCE_*: YES|NO`：231；
- 包含 `[END]` / `[END OF OUTPUT]`：5；
- evidence key 拼写或残缺：5；
- 其它额外行：6。

231/247，也就是 `93.52%` 的 invalid，直接由 bare evidence 与 parser 匹配顺序冲突导致。

### 7.4 修正口径后的 audit

仅提取 expected answer keys，忽略 evidence 行和结束标记后重算：

| 指标 | Production | Audit 重新解析 | 差异 |
|---|---:|---:|---:|
| 联合 exact | 77.54% | **77.54%** | 0.00 pp |
| Phase1-only exact | 89.75% | 89.55% | -0.20 pp |
| Phase2-only exact | 84.47% | 84.18% | -0.29 pp |

只在官方 parser 判定为 valid 的 777 个 audit case 中：

- 联合 exact：`77.73%`；
- Phase1-only exact：`89.06%`；
- Phase2-only exact：`84.04%`。

这与 production 一致，说明 evidence prompt 本身没有明显损伤答案语义。

v2 代码采用更严格且可解释的双通道口径：答案缺失/重复仍会影响语义分；evidence 缺失、重复、
未知行或额外结束标记只影响 audit contract，不再把已合法的答案擦除。对同一批 1024 条 raw
output 重放得到：

| v2 parser 指标 | 结果 |
|---|---:|
| 联合 semantic exact | 77.25% |
| Phase1-only semantic exact | 89.06% |
| Phase2-only semantic exact | 83.79% |
| answers valid | 99.22% |
| evidence complete | 98.54% |
| audit contract valid | 97.75% |

宽松提取的 `77.54%` 与严格语义的 `77.25%` 相差仅 3/1024，差异来自重复/异常答案行是否仍
处罚；正式代码采用后者。

### 7.5 Audit 结论

bundle 中固化的 `58.98%` 不应继续用于：

- 判断融合是否失败；
- 与原 Phase1 audit 分数比较；
- 与原 Phase2 audit 分数比较；
- 选择 checkpoint；
- 据此继续修改 production prompt。

v2 parser 已改为先匹配 `EVIDENCE_` 行，再匹配普通 answer 行，并保留对未知 evidence、
重复 answer 和其它额外行的严格拒绝；回归测试覆盖 bare `EVIDENCE_*: YES|NO`。旧 bundle 的
summary 不会自动改写，阅读时应使用第 7.4 节的离线重算结果；新 audit 可直接使用修复后的 parser。

---

## 8. 训练过程和 checkpoint 可靠性

### 8.1 计划训练量

训练 manifest：

- train rows：`581051`；
- 每轮 global sampled work：`147456`；
- 每 rank 每轮：`36864`；
- epochs：`3`；
- 每 rank 计划总 step：`110592`。

### 8.2 实际训练终点

日志最后一行：

```text
epoch=1 step=29090/110592 ...
```

也就是说训练只跑到总计划的 `26.30%`，第一个 epoch 约完成 `78.91%`。这是用户在
观察到验证集生成指标从 24k 平台转为下降后主动停止并开始正式测试，属于人工 early stop，
不能写成意外中断，也不能用“没有跑满三轮”反推模型应该继续训练。

### 8.3 best-generation 选择

generation validation 每次只有 256 个样本，即每个 focus 32 个。关键节点：

| step | generation-val exact | format valid |
|---:|---:|---:|
| 2000 | 57.03% | 非最终稳定状态 |
| 8000 | 73.83% | 逐步提高 |
| 12000 | 76.95% | 接近平台 |
| 18000 | 76.95% | 波动 |
| 20000 | 78.52% | 提高 |
| 22000 | 80.08% | 提高 |
| **24000** | **81.64%** | **100%** |
| 26000 | 81.25% | 100% |
| 28000 | 75.00% | 100% |

step-24000 是已运行区间内最高 generation exact，因此 `best_generation` 的选择有依据。
但需要注意：

- 24000 与 26000 只差 `0.39 pp`；
- 每次 generation validation 只有 256 个 case，单 focus 只有 32 个；
- 24k validation 中 `VULNERABLE=100%`，正式 128-case focus 测试只有 `83.59%`，说明
  小 validation 对单任务波动非常敏感；
- 28k 明显下降，是用户停止继续训练的直接依据；其幅度也可能混有小验证集波动；
- 正式 test 联合 exact `77.54%`，比 24k validation 低 `4.10 pp`。

所以 step-24000 是当前已跑区间里合理的 early-stop checkpoint。现有证据不支持从 29k
继续盲目训练；后续应从修正后的 loss 合同重新训练，并继续按 generation validation 选择最优点。

---

## 9. 评测采样审计

### 9.1 已满足的硬约束

正式测试满足：

- 八个 focus task 各 128 个 case；
- 每个 focus YES=64、NO=64；
- Phase1 focus 总量与 Phase2 focus 总量相等；
- Phase2 `(focus, label, variant)` 配额精确；
- 总 variant 数量符合 eval `2:1:1`；
- 1024 个 case 中有 1014 个 unique frame；
- 单帧最大重复次数为 2；
- subset 未问 RS 行泄漏为 0。

### 9.2 Phase1 focus × variant 偏差

`focus_variant_target_deviation` 显示：

- `max_abs_delta = 35`；
- 共有 23 个组合偏离目标；
- Phase2 focus×variant 的最大偏差为 0；
- 偏差主要发生在 Phase1 focus 的 label 与 variant 组合。

例如某些 Phase1 YES 桶几乎不进入 all-random，却大量进入 hierarchy。虽然每个 Phase1 focus
总体仍保持 64:64，但 variant 难度与标签发生相关，会产生评测混杂：

- 某任务的 YES 可能主要在更容易或更难的 variant；
- 当前 main F1 仍是平衡二分类结果，但不再代表三种 variant 中同等难度的平均；
- 与原 Phase1 单一固定 prompt 的比较会混入 variant distribution 差异。

后续若要做严格新旧归因，建议额外构建一个固定 Phase2 variant、固定 frame list 的 Phase1 paired
eval；或者让 Phase1 `(focus, label, variant)` 也成为硬约束。

### 9.3 focus 平衡没有传递到实际语义 loss

用本次 `frame_index.jsonl`、训练 seed 和 `4:1:1` variant 配额重放 epoch-0 的 147,456 条
work item，再统计每个 target 实际输出的答案行，得到：

| 主任务 | emitted YES | emitted NO | YES 占比 |
|---|---:|---:|---:|
| HIGHWAY | 17,776 | 129,680 | 12.06% |
| STATIC_OBSTACLE | 18,897 | 128,559 | 12.82% |
| VULNERABLE | 14,198 | 133,258 | 9.63% |
| TRAFFIC_LIGHT_ABNORMAL | 9,633 | 137,823 | 6.53% |
| RS1 | 41,013 | 75,416 | 35.23% |
| RS2 | 31,517 | 85,048 | 27.04% |
| RS4 | 29,196 | 90,789 | 24.33% |
| RS5 | 18,816 | 97,514 | 16.14% |

原因不是 focus sampler 失效，而是 v1 训练代码对 target 中所有输出行都赋予 `1.0` 的答案值
loss。一个样本虽然只属于一个平衡 focus 桶，但其它 4--7 个非 focus 主任务仍反向传播；原始
数据中的自然 NO 因而绕过 focus 平衡重新进入梯度。低 YES 占比与本次测试的低召回桶高度一致：

- VULNERABLE：precision 100%，recall 67.19%；
- RS5：precision 92.00%，recall 71.88%；
- LIGHT：precision 100%，仍有 13 个 FN。

这不是单靠追加提示词能够修复的问题。v2 训练合同改为：八个主任务只监督当前不可见 focus
行；非 focus 值 token 权重为 0，但字段、换行和结束符仍按低权重学习；没有独立 focus 的
`RS_HIGHWAY/GROUP` 继续监督，并按当轮实际 YES/NO 数量做等质量类别加权。训练 balance JSON
同时新增 `emitted_answer_counts`、`semantic_answer_counts` 和 `semantic_class_weights`，
避免以后再次把 focus case 平衡误认为实际 token loss 平衡。

完整 epoch-0 dry-run 已验证：八个主任务的 semantic YES/NO 都是 `9216/9216`、权重均为
`1.0/1.0`；派生层级权重范围为 `0.7243--1.6143`，属于温和校正，重点补偿本次退化最大的
`GROUP:PLAIN_LANE_FOLLOWING_CORRIDOR` 正类，而不是使用失控的大倍率 rare-positive 权重。

---

## 10. 错误分布和后续优先级

### 10.1 第一优先级：VULNERABLE

现状：

- F1 `80.37%`；
- precision `100%`；
- recall `67.19%`；
- 21 个 FN，0 个 FP；
- 相对原 Phase1 F1 下降 `8.33 pp`。

逐帧复核结果：抽查的 FN 中，多数夜间/雾天帧看不到可确认行人或骑行者，不能为了追 GT
把“场景像有行人”“有 crosswalk”“黑暗处可能有人”写成 YES。明确可教的例子是
`PedestrianCrossing` 一组：小行人在较早帧的路口/右侧区域可见、最新帧更难辨认。这支持
“对四帧做独立第二遍小目标扫描、一个仍与当前决策相关的清晰较早帧可作证”，但不支持扩张
VULNERABLE 的空间边界。v2 只加入这个窄提醒，主要修复仍交给 focus-only 语义 loss；若重训后
清晰小目标仍漏检，再做受控 vision LoRA，而不是现在直接开启。

### 10.2 第二优先级：RS1

现状：

- F1 `75.97%`；
- 16 FP、15 FN；
- 是 Phase2 最大主问题错误源。

逐帧复核同时看到两类情况：一类是真实模型错误——普通 surface corridor 中出现 cyclist、
pedestrian、事故/施工车辆后，模型把事件参与者错误地当成道路结构；另一类是 GT/时间边界风险——
标成 R1 的帧已出现清晰 crossroad 或连续护栏、多车道 controlled corridor，或标成 R3/R4/R5
但 RGB 仍明显像普通 surface road。v2 因而只强化“先不看事件主体判断道路拓扑”和
“连续 access-controlled 多车道+barrier corridor 不能判 RS1”，不增加通用 RS1 fallback，
也不按 projection-error 字段一刀切过滤，因为已看到 projection error 但 RGB 的 R1 仍成立的样本。

### 10.3 第三优先级：RS5

现状：

- F1 `80.70%`；
- precision `92.00%`；
- recall `71.88%`；
- 18 个 FN；
- `VehicleTurningRoute` 占 10 个错误。

逐帧复核中真正清晰的 FN 共同模式是 `STOP` 标志与本地道路开口/横向冲突同时可见，且侧视
区域比正前方更容易看到道路连接；这应判 RS5。其余多个 FN 只有黑暗、普通弯道、事件车辆或
连续道路，RGB 不足以支持 RS5。v2 因而要求 paired witness，并明确 bend、turning actor、
darkness 或“看不见 signal”都不能单独触发 RS5。

### 10.4 第四优先级：hierarchical plain-lane group

hierarchical 总体仍强，不应整体重写。优先检查
`GROUP:PLAIN_LANE_FOLLOWING_CORRIDOR`，因为它的 F1 `82.05%` 明显低于其它 group。

### 10.5 第五优先级：STATIC 和灯异常 recall

这两项已经显著可用，并且几乎无 FP。逐帧检查 STATIC FN 时，少数样本能看到较小但持续占据
可用车道的物体；更多样本只见正常停车位车辆、运动队列、正常对向车，或夜间完全不可读，说明
存在明显 temporal/label-risk。灯异常 FN 也有类似分裂：清晰路口中可读的暗灯/冲突灯值得学习，
但多组 GT=YES 帧根本没有可读信号硬件。现有 Phase1 规则已覆盖这些边界，因此 v2 不再放宽
STATIC/LIGHT 定义，只使用两遍逐帧扫描并修复 loss 失衡：

- STATIC：13 FN、1 FP；
- LIGHT：13 FN、0 FP。

---

## 11. 已落地修订与下一轮验证顺序

### P0（已完成）：修正 audit 评测口径

1. parser 已改为 evidence 优先于 answer 匹配；
2. 现有 raw output 已离线重算：宽松 expected-answer exact 为 `77.54%`，v2 严格语义
   exact 为 `77.25%`，audit contract valid 为 `97.75%`；
3. 后续 audit 应单独报告：
   - answer semantic exact；
   - evidence format valid；
   - evidence non-trivial rate；
   - unexpected-line rate；
4. 新 regression test 覆盖 `EVIDENCE_HIGHWAY: NO ...`，不再让合法 evidence 把所有答案清零。

### P1：构建严格 paired comparison

固定同一批 RGB frame、同一 variant、同一 output keys，对以下 adapter 同时跑：

- 原 Phase1 4RGB LoRA；
- 原 Phase2 checkpoint-20000；
- 融合 step-24000。

这样才能把数据变化、prompt 变化和训练变化真正分开。

### P2（已完成首轮）：逐题 RGB 错例审计

已按 VULNERABLE、RS1、RS5、STATIC/LIGHT 的四帧历史复核形成第 10 节结论。审计导出模板
新增 `label_support / newest_frame_witness / older_frame_witness / temporal_relevance /
error_source / recommended_action` 等结构化字段，后续不再把自动导出的 GT/pred note 当成人工
视觉结论。下一轮应继续补齐 hierarchical plain-lane group，并把人工 verdict 保存成轻量表，
不要根据 scenario 名或自由文本 evidence 反推标签。

### P3：固定样本复核 early-stop 稳定性

对 20k、22k、24k、26k、28k 使用同一份固定 1024-case test index，比较：

- 联合 exact；
- Phase1-only exact；
- Phase2-only exact；
- 八个 focus F1；
- 三种 variant exact；
- 关键错误集合的 paired flip。

这不是要求恢复已停止的旧训练；只在 checkpoint 仍可用时做离线 paired eval，用来量化用户已观察到的
过拟合转折。不要只依赖每次 256-case、每 focus 32-case 的 generation validation。

### P4：先重训 v2 loss 合同，再决定视觉 LoRA

下一轮先用 v2 prompt + focus-only semantic loss 做短程训练，并重点观察正类 recall、联合 exact、
hierarchical exact 与 format valid。只有在这之后再判断：

- 如果 generation exact 再次连续下降，应沿用本次主动 early stop；
- 如果 VULNERABLE 错例多数肉眼清晰，先调监督和采样；
- 如果多数是细小、遮挡、远距目标，才值得做受控的 vision LoRA scope 实验；
- 不建议仅依据当前错误直接扩大 prompt，因为现有 prompt 已很长，继续追加规则可能加剧多任务竞争。

---

## 12. 最终判断

### 是否提升整体问答能力？

是，整体提升。

最强证据是同 bundle base → LoRA：

- 联合 exact：`26.37% -> 77.54%`；
- Phase1-only exact：`66.60% -> 89.75%`；
- Phase2-only exact：`32.13% -> 84.47%`；
- production 格式合法率达到 `100%`。

### 是否超过原 Phase1？

总体超过：

- Phase1-only exact 方向性提高约 `4.98 pp`；
- Macro F1 提高约 `2.63 pp`；
- STATIC 提升最大；
- 但 VULNERABLE 明显下降，需要专项修复。

### 是否超过原 Phase2 augment？

综合略微超过：

- Phase2-only exact 提高约 `1.67 pp`；
- subset、RS2、RS4 提升明显；
- RS5 小幅下降；
- hierarchical 专项能力下降约 `5.30 pp`，但仍维持较高水平。

### audit 是否说明模型不稳定？

不能。官方 audit `58.98%` 主要是 parser 缺陷。重新解析答案行后，audit 联合 exact 为
`77.54%`，与 production 相同。

### 当前 checkpoint 是否可以作为最终模型？

它可以作为 v1 已训练区间内的 best checkpoint 和后续 paired audit 基线。没有跑满三轮是基于
验证回落的主动 early stop，不是缺陷；但它也暴露了 actual semantic token 失衡，不能再从该点
原样续训并期待 recall 自动恢复。

最终建议：保留融合架构，采用已经落地的 v2 窄 prompt 修订、focus-only 主任务语义 loss、
严格 audit parser 和结构化 RGB verdict 模板重新做短程训练。先验证 VULNERABLE/RS5/LIGHT
recall 是否在 precision 可控的前提下恢复，以及 hierarchical 是否停止下滑；只有肉眼清晰的小目标
在新合同下仍持续漏检时，才开启受控视觉 LoRA。
