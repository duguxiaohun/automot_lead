# SFT Loop Phase2 Augment Checkpoint-20000 指标分析

数据来源：

- Augment 训练日志：`AutoMoT/checkpoints/checkpoints/sft_loop_phase2_augment_runs/run_rs_augmented_format_supervised_4rgb/20260818_101413/`
- Augment eval：`AutoMoT/checkpoints/checkpoints/sft_loop_phase2_augment_eval/`
- Augment 错例抽样：`AutoMoT/checkpoints/checkpoints/sft_loop_phase2_augment_audit_samples/ckpt20000_audit/`
- 旧版 four-binary 对照：`AutoMoT/checkpoints/sft_loop_ohase2_eval/`

## 1. 结论

`checkpoint-20000` 相比 base Qwen 是显著有效的：production 总 exact 从 `31.8%` 提到 `82.8%`，audit 从 `29.4%` 提到 `74.8%`。四个 RS 细问里，提升最大的是原本 base 几乎不会判断的 `RS2`，F1 从 `11.6%` 到 `84.3%`；`HIGHWAY` 层级问法也从近乎失效的 `0.8%` F1 到 `97.3%`。

但问题也很清楚：augment 总分高，很大一部分来自新增 `hierarchical_probe` 的高分；如果只看最像旧版四问任务的 `all_random_order`，LoRA exact 是 `76.5%`，低于旧版 four-binary LoRA 的 `79.3%`。所以增强方法提升了多问法泛化和层级问答，但没有在原四问全量格式上压过旧版。

## 2. 训练过程

训练配置是 `4rgb`，4 卡，训练集 `581,503` 行，计划 `73,728` rank step；本地日志停在 `step=26000/73728`，所以这里只分析中途 checkpoint。`best_val.json` 记录的 best-val 在 `step=6000`：val loss `0.0892`，teacher-forced value token acc `79.8%`，小样本 free-generation exact `74.5%`。

训练中验证指标：

| step | teacher-forced loss | value acc | free-gen exact | all_random | subset | hierarchical |
|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 0.1081 | 71.8% | 68.2% | 49.0% | 79.2% | 95.8% |
| 6000 | 0.0892 | 79.8% | 74.5% | 63.5% | 75.0% | 95.8% |
| 16000 | 0.1156 | 80.5% | 83.9% | 77.1% | 87.5% | 93.8% |
| 20000 | 0.1068 | 81.6% | 77.1% | 65.6% | 85.4% | 91.7% |
| 26000 | 0.0953 | 82.7% | n/a | 76.8% | 83.3% | 93.8% |

说明：loss/小样本生成有波动，主要因为每个窗口混合了 `all_random_order/subset_random/hierarchical_probe` 和不同 RS bin；格式学习很稳定，teacher-forced `format_acc` 全程 `100%`。

## 3. Augment Base vs LoRA

总体：

| prompt | model | cases | exact | 相对 base |
|---|---|---:|---:|---:|
| production | base | 6144 | 31.8% | - |
| production | LoRA ckpt-20000 | 6144 | 82.8% | +51.0 pp |
| audit | base | 6144 | 29.4% | - |
| audit | LoRA ckpt-20000 | 6144 | 74.8% | +45.4 pp |

按问法：

| prompt | variant | base exact | LoRA exact | delta | LoRA format valid |
|---|---|---:|---:|---:|---:|
| production | all_random_order | 30.9% | 76.5% | +45.6 pp | 100.0% |
| production | subset_random | 41.3% | 81.1% | +39.7 pp | 100.0% |
| production | hierarchical_probe | 24.0% | 97.1% | +73.0 pp | 100.0% |
| audit | all_random_order | 23.8% | 68.5% | +44.7 pp | 92.6% |
| audit | subset_random | 43.8% | 76.1% | +32.4 pp | 93.9% |
| audit | hierarchical_probe | 26.3% | 86.1% | +59.8 pp | 95.1% |

解读：

- `hierarchical_probe` 是最大收益点：把 `HIGHWAY -> GROUP -> DETAIL` 分解后，LoRA 几乎学成了结构化决策树。
- `subset_random` 比全四问更稳，说明随机子集训练确实提升了“只答被问项”的服从性，且未问 RS 泄漏为 0。
- audit prompt 会诱导长证据输出，LoRA 的 format valid 掉到 `92.6%-95.1%`，这部分直接压低 audit exact；production 没这个问题。

按问题：

| prompt | question | base acc | LoRA acc | base F1 | LoRA F1 | 主要变化 |
|---|---|---:|---:|---:|---:|---|
| production | RS1 | 43.5% | 81.4% | 49.7% | 72.4% | 明显改善，但仍是最大错源 |
| production | RS2 | 68.3% | 91.0% | 11.6% | 84.3% | 从几乎不召回变成可用 |
| production | RS4 | 80.4% | 93.3% | 44.9% | 87.4% | recall 大幅改善 |
| production | RS5 | 81.9% | 91.0% | 53.9% | 81.8% | 中等幅度改善 |
| production | HIGHWAY | 50.4% | 97.3% | 0.8% | 97.3% | base 基本只会答 NO，LoRA 学会高速/匝道拓扑 |
| audit | RS1 | 40.0% | 75.7% | 51.2% | 67.2% | audit 下仍偏误报 |
| audit | RS2 | 57.1% | 85.9% | 28.5% | 79.9% | 仍是有效提升 |
| audit | RS4 | 82.0% | 88.7% | 52.2% | 84.6% | 被长证据 prompt 拖低 |
| audit | RS5 | 82.1% | 86.4% | 56.2% | 78.5% | 漏报仍较多 |
| audit | HIGHWAY | 54.6% | 92.9% | 17.1% | 95.1% | 高速判断整体稳，但会被“无可见 merge/split/exit”误导 |

错例抽样/全量错误计数显示，production 错误里 `RS1` 仍最多：`RS1=771`、`RS2=371`、`RS5=371`、`RS4=276`。audit 错误更严重：`RS1=973`、`RS2=552`、`RS5=527`、`RS4=450`，并额外有 `HIGHWAY=109`、`GROUP=183`、`DETAIL=170`。

## 4. 主要问题与原因

1. `RS1` 误报仍重。错例中模型常把普通双向路的黄线/白线、弯道、窄路也解释成“plain lane-following corridor”，尤其在 `VehicleTurningRoute`、`DynamicObjectCrossing`、`AccidentTwoWays` 等场景。根因是 `RS1` 的视觉否定条件比较抽象：它不是“有车道线就是 YES”，而是要排除对向共享、路口控制和高速拓扑。

2. `HIGHWAY` 漏报集中在 `MergerIntoSlowTraffic*`、`HighwayExit`、`HighwayCutIn`。audit case 里经常出现“多车道/护栏/同向车流”被模型描述出来，但最终仍写 `HIGHWAY: NO`，原因是 prompt 强调 merge/split/exit/connector 等显式拓扑，模型在局部帧没看到清晰 gore 或出入口时过度保守。

3. audit prompt 带来的长证据输出有副作用。production 格式全对，但 audit 下 LoRA 格式有效率降到 `92.6%-95.1%`，且有重复描述、输出过长、结尾标签缺失的情况；这不是分类能力本身下降，而是证据生成和格式约束竞争。

4. `all_random_order` 是最硬的泛化位。它要求一次独立回答四个 YES/NO，错误 exact 会被任一问题拖垮；相比之下 `subset_random` 少问几项，`hierarchical_probe` 把问题拆成粗到细，因此得分更高。

## 5. 与旧版 sft_loop_ohase2_eval 对比

旧版 four-binary 是四个 RS 问题固定全问；augment 的 `all_random_order` 与它最接近。总体对比：

| eval | model | cases | production exact | audit exact |
|---|---|---:|---:|---:|
| old four-binary | base | 512 | 30.7% | 20.3% |
| old four-binary | LoRA ckpt-40000 | 512 | 79.3% | 76.2% |
| augment all variants | base | 6144 | 31.8% | 29.4% |
| augment ckpt-20000 | LoRA | 6144 | 82.8% | 74.8% |
| augment all_random only | LoRA | 3072 | 76.5% | 68.5% |

关键对比：

- Base：augment 和旧版 production base 几乎一样，`31.8%` vs `30.7%`；说明 base Qwen 自身没有稳定掌握 RS 四问。
- LoRA 总体：augment production 总分高于旧版 `+3.5 pp`，但这是因为包含高分 `hierarchical_probe`。
- LoRA 同类任务：augment `all_random_order=76.5%`，低于旧版 four-binary `79.3%`；audit 下差距更大，`68.5%` vs `76.2%`。
- 稳健性：旧版 audit 相对 production 只掉 `3.1 pp`，augment 总体掉 `8.0 pp`，all_random 掉 `8.0 pp`；说明 augment 的 audit 长证据模板更容易破坏格式或引入过度解释。

逐问题 production 对比：

| question | old LoRA acc/F1 | augment LoRA acc/F1 | 变化 |
|---|---:|---:|---|
| RS1 | 81.2% / 72.3% | 81.4% / 72.4% | 基本持平 |
| RS2 | 92.6% / 84.3% | 91.0% / 84.3% | acc 小降，F1 持平 |
| RS4 | 93.6% / 83.6% | 93.3% / 87.4% | F1 更好，召回更均衡 |
| RS5 | 92.6% / 78.2% | 91.0% / 81.8% | F1 更好，acc 小降 |

逐问题 audit 对比：

| question | old LoRA acc/F1 | augment LoRA acc/F1 | 变化 |
|---|---:|---:|---|
| RS1 | 78.7% / 70.9% | 75.7% / 67.2% | 下降，RS1 误报更明显 |
| RS2 | 90.2% / 82.0% | 85.9% / 79.9% | 下降，audit 输出干扰判断 |
| RS4 | 90.6% / 79.0% | 88.7% / 84.6% | acc 降，F1 升 |
| RS5 | 91.6% / 80.2% | 86.4% / 78.5% | 下降 |

所以对比结论是：augment 让模型学会了随机子集和层级拆问，production 综合分更漂亮；但对旧版同类“四问一次答完”的任务，`checkpoint-20000` 没有超过旧版 `checkpoint-40000`，尤其 audit prompt 更脆。增强的主要价值不是单纯提高旧任务分数，而是扩大问法覆盖、改善 RS4/RS5 F1，并把 HIGHWAY/GROUP/DETAIL 这类分解式诊断做得很稳。

## 6. 后续建议

- 若目标是最终 production 分类，可继续用 augment 路线，但需要单独提高 `all_random_order` 权重或 hard-negative 采样，避免总分被 hierarchical 掩盖。
- 若目标是可审计输出，应把 audit 的证据输出限长，并强制最后答案行完整；当前 audit 下降主要是格式/冗长问题。
- 针对 `RS1` 增加“普通车道线不等于 RS1”的反例，特别是双向窄路、转弯场景、事故/施工/动态目标场景。
- 针对 `HIGHWAY` 增加局部帧高速正例：没有清晰 gore/merge/split 但仍处于 limited-access mainline/ramp 的样本。

## 7. 追加 RGB 复核与代码调整

按上述结论追加查看了 `checkpoint-20000` 的 production 四问错例和 audit 抽样 RGB：

- `ControlLoss/Town13_Rep0_1274_0.../f68`：夜雨多车道、路侧护栏明显，但仍像普通连续道路；不能继续把“有护栏/低可见度”硬压成高速，否则会伤 `RS1` 正例。
- `MergerIntoSlowTraffic/Town13_Rep0_1230_6.../f147`：多条同向高速车道、连续隔离/护栏和分离车行道清楚；这是 `R3`，模型误报 `RS1`，需要强调 high-speed lane + separated carriageway/shoulder/barrier 的组合。
- `AccidentTwoWays/Town12_Rep0_487_0.../f22`：雾中可见未隔离双向走廊和迎面/占道车辆；这是 `RS2` 正例，不能要求 ego 已经跨线才算 opposing-lane constraint。
- `RedLightWithoutLeadVehicle/Town13_Rep0_route_003320.../f96`：有双黄线和路边停放车，但 ego 车道仍打开；这是 `RS2` 负例，不能把中心线/停放车单独当正证据。
- `Accident/Town10HD_Rep0_route_001821.../f58-f61`：极暗雨夜下当前 RGB 难以确认信号灯/路口结构；这类 `RS4` 标签更像低可见度标签风险，不应继续靠 prompt 猜场景。
- `VehicleTurningRoute/Town03_Rep0_Town03_Scenario4_6.../f95`：局部 turning conflict/road mouth 有一定证据，但雾中不强；`RS5` 可以补充 side-road mouth/curb break，但仍不能把普通弯道、行人或车辆事件当成 RS5。

已据此将 prompt 合同升级到 `sft_loop_phase2_rs_augmented_visual_v2`：

- `RS1`：区分普通路侧护栏/低可见度与真正 limited-access 高速走廊，避免两头误伤。
- `RS2`：同时补强正负边界；迎面车头/车灯占据相邻对向通行空间可算正证据，但中心线、雾、路边停放车且 ego lane 完整打开不能单独算正。
- `RS5`：补充可见 side-road mouth / curb break / local turning conflict，但明确车辆/行人事件和普通弯道不是 catch-all。
- audit 输出改为先写所有 `KEY: YES/NO`，再写每题一行短证据，避免 evidence-first 的长文本重复导致答案行缺失。
- 训练保存新增 `best_generation/`：按 free-generation exact 选择，并用 `all_random_order` exact 作 tie-breaker；`eval.py` 解析 run/latest 时优先 `best_generation/`，再 `best_val/`，最后 `final/`。
- 错例抽样默认 target 扩为 `invalid_answer`、四个 RS 的 FP/FN、`HIGHWAY` FP/FN 与 `multi_yes`，避免只审计旧的少数错误桶；同时在 eval 包缺少 `error_cases/` 时自动从 `cases_rank*.jsonl` 筛 `all_ok=false` 后复制 RGB，修复 production 抽样 `selected=0` 的问题。

注意：v2 改动会改变 production prompt hash，旧 `checkpoint-20000` 是 v1 adapter，不能把它和 v2 prompt 当作同一合同直接横比；后续应重训/继续训练一版 v2，再按 production full eval 和 `all_random_order` exact 选 checkpoint。
