# New Phase2 v4 重训总对比与 RGB 转移审计（2026-08-29）

## 1. 审计范围

本记录汇总以下四个 New Phase2 bundle，并与旧 RS-gated Phase3 记录比较：

- v1 / 4RGB：`checkpoints/sft_new_loop_phase2_20260825_133640_audit_bundle`
- v2 / 2RGB endpoints：`checkpoints/sft_new_loop_phase2_20260825_150826_2rgb_endpoints_audit_bundle`
- v3 / 2RGB endpoints：`checkpoints/sft_new_loop_phase2_20260827_144749_2rgb_endpoints_audit_bundle`
- v4 / 2RGB endpoints：`checkpoints/sft_new_loop_phase2_20260829_235430_2rgb_endpoints_audit_bundle`
- 旧 Phase3：`qwen3vl_local/sft_loop_phase3/EVAL_PROMPT_V2_V3_20260821.md`

指标直接读取各 bundle 的 `base_production`、`lora_production`、
`base_audit_prompt`、`lora_audit_prompt` 和 adapter metadata。RGB 归因使用实际 case
保存的 `history_rgb_paths_all4`，逐帧查看 v3→v4 全部 46 个正确性发生变化的 case，
共 184 张原始 stitched RGB。2RGB 模型只实际接收 `t0/t3`；`t1/t2` 只用于离线判断
运动连续性和标签边界，不作为模型可见证据。临时审计页没有写入仓库。

## 2. 可比性与限制

### 2.1 v2、v3、v4 是固定测试集配对比较

三者满足：

- `dataset_metadata/manifest.json` SHA256 都是
  `a7f8f15164d97091390adffbc5b6295943033565fb75cbf870495008c2a55b60`；
- 384 个 production case 的 `case_index/scenario/town/route/frame/GT` 完全相同，
  排序身份 SHA256 都是
  `ca040922412fcbc4241b10c1601dd1ac1f78933545879466cf0bdae796520a67`；
- base 都是 `checkpoints/Qwen3-VL-4B-Instruct`，视觉 LoRA 都为 `off`；
- RGB 都是原四帧 history 的 `[0,3]` 两个端点；
- train 主采样不变。v3→v4 的 `train_balance.json` 差异只来自 generation eval
  从 16 增至 32 条/桶，以及新增 UE3 recall 选优字段。

因此 v2→v3 可以主要看作 prompt 变化；v3→v4 除 prompt 外，还同时改变了 generation
validation 样本数和 checkpoint 选择规则，不能把全部差异只归因于新增的 UE3 句子。

### 2.2 v1 和旧 Phase3 不是严格因果对照

v1 使用 4RGB；旧 Phase3 还带 synthetic RS user/assistant KV prefix，而 New Phase2
完全删除该前缀，只输入 RGB 和当前 EVENT 问题。因此它们可比较最终任务分数，但不能做
逐 case 的单变量因果结论。v1→v2 又同时变化了 RGB mode、prompt 和数据过滤，不能声称
提升单独来自“减少到 2RGB”。

此外，这 384 条已经用于 v2/v3/v4 prompt 误差归因，后续只能作为 dev/audit set，不能再
声称是独立泛化测试。真正的泛化结论必须来自尚未用于调 prompt 的 456 条 unseen cases。

## 3. 总结结论

1. 融合后的单轮 direct EVENT 路线确实优于旧 Phase3，但最佳结果是 v3，不是最新 v4。
2. v4 相对 v3 从 `316/384` 降到 `308/384`，下降 8 个 case、2.08 个百分点；
   audit strict exact 也从 `314/384` 降到 `305/384`。这不是格式问题，两者格式有效率
   都是 100%。
3. v4 的目标性收益成立：UE3 recall 从 v3 的 0.7188 恢复到 0.7969，UE3 F1 从
   0.7931 提高到 0.8293。但它同时把 UE6 recall 从 0.8750 拉低到 0.7656，并降低
   RE、INVALID 和 UE5；因此属于局部修复换取整体退化。
4. v3→v4 配对转移是“修复 19、回退 27”，McNemar 双侧精确检验 `p=0.302`。
   当前 384 条不足以证明总体差异具有统计显著性，但 v4 已明确违反 production 的
   `UE6 recall >= 0.80` 安全门槛，工程上仍不能晋升。
5. 当前 production prompt 回退并保留 v3 是合理的。v4 只应保留为 UE3 定向实验记录，
   不应替换 v3。

## 4. 总体分数

### 4.1 LoRA production 主指标

| 路线 | RGB / 上下文 | production exact | 错例 | 相对旧 Phase3 v2 |
|---|---|---:|---:|---:|
| 旧 Phase3 prompt v2 | 4RGB + oracle-like RS prefix | 299/384 = 77.86% | 85 | - |
| New Phase2 v1 | 4RGB，direct EVENT | 298/384 = 77.60% | 86 | -0.26pp |
| New Phase2 v2 | 2RGB endpoints，direct EVENT | 315/384 = 82.03% | 69 | +4.17pp |
| New Phase2 v3 | 2RGB endpoints，direct EVENT | **316/384 = 82.29%** | **68** | **+4.43pp** |
| New Phase2 v4 | 2RGB endpoints，direct EVENT | 308/384 = 80.21% | 76 | +2.34pp |

v1 说明“移除 RS prefix”本身没有自动带来提升；真正的提升出现在 v2 之后的 prompt、过滤、
采样审计和 2RGB endpoints 组合。v3 比 v2 只多 1 个 case，属于基本持平；v4 又回落到
v2/v3 之下，但仍高于 v1 和旧 Phase3。

### 4.2 Base 与 LoRA

v4 base production strict exact 是 `50/384=13.02%`，answer-only 是
`64/384=16.67%`；base audit strict exact 是 `64/384=16.67%`。其主要行为仍是保守
all-NO，只命中一个均衡 RE 桶，且 production 下还有格式损失。v4 LoRA production 达到
`308/384=80.21%`，说明事件问答能力主要来自 LoRA 训练，而不是 base prompt 零样本能力。

### 4.3 Audit prompt

| 版本 | audit strict exact | format valid | answer-only exact |
|---|---:|---:|---:|
| v2 | 263/384 = 68.49% | 327/384 = 85.16% | 316/384 = 82.29% |
| v3 | **314/384 = 81.77%** | **384/384 = 100%** | 314/384 = 81.77% |
| v4 | 305/384 = 79.43% | **384/384 = 100%** | 305/384 = 79.43% |

v3 已经修复 v2 的空 `EVIDENCE_*` 行问题。v4 保持 100% 格式有效，但答案本身少 9 个
正确 case，所以 v4 audit 退化是语义/分类退化，不是 parser 或 evidence schema 退化。

## 5. 分桶与逐题能力

### 5.1 每个互斥 GT 桶的整例 exact

| 版本 | UE1 | UE3 | UE5 | UE6 | applicable RE | highway RE | INVALID | 总计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 49 | 42 | 54 | 55 | 30 | 15 | 53 | 298 |
| v2 | 52 | 52 | 57 | 54 | 30 | 16 | 54 | 315 |
| v3 | 50 | 46 | **60** | **56** | **33** | **16** | **55** | **316** |
| v4 | **52** | 51 | 59 | 49 | 29 | **16** | 52 | 308 |

v4 相对 v3 的净变化为：UE1 `+2`、UE3 `+5`、UE5 `-1`、UE6 `-7`、
applicable RE `-4`、highway RE `0`、INVALID `-3`。UE3 的收益无法覆盖其它桶的退化。

### 5.2 逐问题 P/R/F1

| 题目 | v2 P/R/F1 | v3 P/R/F1 | v4 P/R/F1 | v4 判断 |
|---|---:|---:|---:|---|
| UE1 | .883/.828/.855 | .926/.781/.847 | .898/.828/.862 | recall 恢复，F1 小升 |
| UE3 | .825/.812/.819 | .885/.719/.793 | .864/.797/.829 | 目标修复有效，三版最高 F1 |
| UE5 | .983/.891/.934 | .968/.938/.952 | .937/.922/.929 | 相对 v3 回退 |
| UE6 | .947/.844/.893 | .918/.875/.896 | .942/.766/.845 | 明显漏报，低于 0.80 门槛 |
| INVALID | .931/.844/.885 | .948/.859/.902 | .929/.812/.867 | precision/recall 都回退 |

最新 v4 不是“问答能力全面下降”：UE1、UE3 的局部判别更好；但整体 EVENT 问答、路口冲突
召回、问题域适用性和 regular hard-negative 稳定性下降。因此总体结论必须记为“局部提升、
整体下降”。

## 6. v3→v4 配对转移与 RGB 证据

同一 384 case 上：

- 两版都正确：289；
- 两版都错误：49；
- v4 修复：19；
- v4 回退：27。

### 6.1 UE3 定向修改确实修复了目标样本

v4 修复 8 个 v3 的 UE3 错例：`28, 220, 221, 232, 233, 270, 277, 356`。
其中 7 个是 `ParkingCutIn`，1 个是 `StaticCutIn`。四帧 RGB 中可见停车位/路边车辆从
oldest 到 newest 持续靠近车道边界、车头或车身逐步进入 usable corridor；这些样本符合
v4 新增的“即使 newest 仍带 parked-looking angle，只要相对边界持续侵入也判 UE3”规则。
这部分修复有直接 RGB 证据，不是根据 scenario 名倒推。

但 v4 同时丢失 3 个原本正确的 UE3：`39, 103, 334`，三例都是
`DynamicObjectCrossing`。RGB 显示它们位于宽路口/开放交叉区域，有横向车辆进入或穿过
ego future path，而不是停车位 pull-out。新增文字没有显式删除 dynamic crossing，故不能
把这 3 个回退简单归因于规则定义错误；更像训练/选 checkpoint 后模型把注意力偏向了
parking-side cue。

### 6.2 最大损失来自未改 prompt 的 UE6

v4 修复 4 个 UE6，但回退 11 个 UE6，净损失 7。回退主要是
`OppositeVehicleRunningRedLight` 的不同天气/时刻帧。四帧 RGB 中多例可以看到横向冲突车，
但红灯/路权违规线索有时很小、被雨雾遮挡或只在部分帧更清楚。更关键的是：v3 与 v4 的
UE6 prompt 没有变化，同一场景中又同时出现修复和回退。因此这不是一条可由 RGB 支持的
统一 UE6 prompt 规则变化，而是 checkpoint 选择和训练波动造成的类别漂移。

### 6.3 RE、INVALID 和其它桶的回退没有共同的新视觉规则

v4 新增的其它回退包括：

- 6 个 RE 被误报为 UE：静态施工/停车、普通 lead、blocked intersection 或多车交互；
- 4 个 INVALID 漏报：连续道路与局部路口域的边界帧；
- 2 个 UE1 回退：一例多 actor 画面被同时报 UE1+UE3，一例夜间弱证据；
- 1 个 UE5 回退：画面几乎没有可见的仍在侵入车辆，更接近事件 span/可观察性边界。

这些 case 不共享一个能由 v4 新增 UE3 句子解释的视觉模式。尤其 UE6、INVALID、UE5 的
prompt 都未修改，说明不能继续通过追加 prompt 文字追这些标签；应优先修 checkpoint 选优、
多 seed 稳定性和独立 unseen 验收。

### 6.4 v4 的一个新冲突信号

v4 出现 1 个 `MULTI:UE1+UE3`，而 v3 没有 multi-YES。该 case 同时包含同车道前车和侧向
车辆，说明强化 UE3 后，原有 UE1/UE3 互斥边界并非在所有 checkpoint 上都稳定。数量只有
1，暂不应继续扩写 prompt，但应保留 multi-YES guard。

## 7. 训练与 checkpoint 选优问题

v4 把 generation eval 从 16 提高到 32 条/桶，五次 validation 如下：

| step | total exact | UE3 target recall | UE6 target recall | applicable RE | INVALID |
|---:|---:|---:|---:|---:|---:|
| 2000 | .7604 | .3125 | 1.0000 | .5417 | .9375 |
| 4000 | .8073 | **.4062** | 1.0000 | .6250 | .9062 |
| 6000 | .7604 | .3438 | .6250 | .6667 | .9062 |
| 8000 | .7917 | .3125 | .9688 | .7083 | .8750 |
| 10000 | **.8281** | .3438 | .9062 | .8333 | .9375 |

没有任何 step 达到 UE3 recall `0.625`。bundle 中的 `best_generation.json` 明确记录
`ue3_target_recall_guard_ok=false`，但仍把 UE3 recall 最高的 step 4000 放进
`best_generation` 并交给 full pipeline 测试。结果它在 384-case 上意外恢复 UE3，
却把 UE6 拉到 0.7656。

因此只守一个 UE3 指标会把类别风险转移到其它桶。当前代码采用的多门槛方向是必要的：

- UE3 recall `>=0.625`；
- UE6 recall `>=0.80`；
- INVALID exact `>=0.80`；
- applicable RE exact `>=0.50`；
- 全部门槛通过后才按总 exact 选优；否则只保存明确命名的 fallback，full pipeline 不自动晋升。

不过门槛只能降低误晋升风险，不能替代多 seed。v2/v3/v4 使用单次训练，v4 的 UE6 大幅波动
说明后续应按 frozen protocol 先跑 3 个 seed，只用 validation 选择，再对 456 条 unseen
做一次性验收。

## 8. 生产决策与后续验收

### 8.1 当前生产选择

- production prompt：保留/回退 `sft_new_loop_phase2_direct_event_visual_v3`；
- 不晋升 20260829 v4 adapter；
- v4 文本和 bundle 仅保留作 UE3 定向实验与 RGB 审计证据；
- 不再为本 384-case 继续追加 prompt 规则。

v3 的优点是总体 exact、audit strict、UE5、UE6、RE、INVALID 最均衡；缺点是 UE3 recall
偏低。v2 的 UE3 recall 略高，但 audit evidence 合同明显不稳定。综合当前生产合同，v3
仍是更合理的默认。

### 8.2 下一次可以声称“提升”前必须满足

1. 训练前冻结 prompt、数据 manifest、RGB mode、seed 列表和 validation 门槛；
2. 三个 seed 只看 validation，历史 384 条不能参与 checkpoint/prompt 选择；
3. 一次性评测剩余 456 条 unseen cases，并硬校验与历史 384 条身份无交集；
4. 同时报总 exact、UE1/UE3/UE5/UE6/INVALID P/R/F1、applicable/highway RE、multi-YES；
5. 只有在 UE3 恢复且 UE6/INVALID/RE 不跨门槛退化时，才把结果描述为整体提升；
6. 若 unseen 仍出现类别交换，只保留 v3 production，转向数据/时序建模，不继续堆 prompt。

## 9. 最终判断

New Phase2 的融合方向总体有效：它在不使用 synthetic RS prefix 的前提下，v2/v3 已把固定
384-case 的 direct EVENT exact 从约 77.6% 提高到约 82.0%～82.3%，说明相关 RGB 事件
问答能力较旧 Phase3/早期 direct EVENT 有提升。

但 20260829 v4 不是进一步提升。它成功修复了一批有真实跨帧横向侵入证据的 UE3，代价是
UE6、RE、INVALID 和 UE5 的回退，总 exact 降到 80.21%。因此当前结论是：
**融合路线提升成立，v4 局部 UE3 提升成立，但最新 v4 的整体问答能力相对 v3 下降；生产应保留 v3。**
