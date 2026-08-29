# New Phase2 v3 重训结果与全错例 RGB 审计（2026-08-29）

## 1. 审计范围

本次主要对比：

- v1/4RGB：`checkpoints/sft_new_loop_phase2_20260825_133640_audit_bundle`
- v2/2RGB endpoints：`checkpoints/sft_new_loop_phase2_20260825_150826_2rgb_endpoints_audit_bundle`
- v3/2RGB endpoints：`checkpoints/sft_new_loop_phase2_20260827_144749_2rgb_endpoints_audit_bundle`
- 旧 Phase3 对照：`qwen3vl_local/sft_loop_phase3/EVAL_PROMPT_V2_V3_20260821.md`

v2 与 v3 是严格的主对照：两个 bundle 的 `dataset_metadata/manifest.json` SHA256 均为
`a7f8f151...a55b60`，`train_balance.json` SHA256 均为 `abe31103...25ff`；训练数据、
采样、四卡配置、训练步数、384 个测试 case 及 2RGB `[0,3]` 都相同。主要变量是
production prompt v2 到 v3。v1 使用 4RGB，与 v2/v3 的差异不能全部归因于 prompt。

## 2. 结果结论

| 版本 | RGB | production exact | 错例 | 相对前版 |
|---|---:|---:|---:|---:|
| v1 | 4 | 298/384 = 77.60% | 86 | - |
| v2 | 2 endpoints | 315/384 = 82.03% | 69 | +4.43pp |
| v3 | 2 endpoints | 316/384 = 82.29% | 68 | +0.26pp |

整体判断：

1. 融合后的 new Phase2 路线相对 v1/旧 Phase3 是有效提升；v3 相对 v1 净增 18 个正确 case。
2. v3 相对同设置 v2 只多 1 个 case。配对转移为 v3 修复 21、回退 20，
   McNemar 精确检验 `p=1.0`；不能声称语义能力显著提升。
3. v3 是“精度上升、UE3 recall 下降”的交换，不是全面更好。

### 2.1 逐题指标

| 题目 | v2 TP/FP/FN | v2 P/R/F1 | v3 TP/FP/FN | v3 P/R/F1 | 判断 |
|---|---:|---:|---:|---:|---|
| UE1 | 53/7/11 | .883/.828/.855 | 50/4/14 | .926/.781/.847 | 更保守，F1 小降 |
| UE3 | 52/11/12 | .825/.812/.819 | 46/6/18 | .885/.719/.793 | FP -5，但 FN +6，明显过度收紧 |
| UE5 | 57/1/7 | .983/.891/.934 | 60/2/4 | .968/.938/.952 | 提升 |
| UE6 | 54/3/10 | .947/.844/.893 | 56/5/8 | .918/.875/.896 | 基本持平 |
| INVALID | 54/4/10 | .931/.844/.885 | 55/3/9 | .948/.859/.902 | 小幅提升 |

v2 到 v3 的分类 exact 变化：UE1 `52→50`、UE3 `52→46`、UE5 `57→60`、
UE6 `54→56`、applicable RE `30→33`、highway RE `16→16`、INVALID `54→55`。

## 3. 训练与 checkpoint 选优审计

| 项目 | v2 | v3 |
|---|---:|---:|
| best_generation step | 4000 | 10000 |
| val generation exact | 85/96 = 88.54% | 79/96 = 82.29% |
| val UE3 slice exact | 10/16 = 62.50% | 5/16 = 31.25% |
| 同 step teacher-forced loss | .09190 | .18008 |
| best_val step/loss | 4000 / .09190 | 4000 / .09641 |

v3 在 2k/4k/6k/8k/10k 的 UE3 validation slice 依次为
`.125/.3125/.3125/.1875/.3125`，而 v2 best_generation 是 `.625`。这说明问题不仅是最终
test 波动：v3 训练全程都没学好 UE3 正类。原选优只看总 exact，会容许某个关键事件类
被牺牲。

本次代码修正：

- generation eval 默认从 16 增加到 32 条/桶，降低小样本选优噪声；
- 新增 `slice/ue*_target_recall`，直接度量正类目标行是否被召回；
- `best_generation` 先守 `UE3 target recall >= 0.625` 的已验证 v2 底线，达标后再按总 exact
  选优；如果全部 step 都不达标，保留 UE3 recall 最高的明示 fallback，不静默丢失
  `best_generation`。

## 4. audit prompt 与语义分数必须分开

| 版本 | audit strict exact | strict format valid | answer-only exact |
|---|---:|---:|---:|
| v2 | 263/384 = 68.49% | 327/384 = 85.16% | 316/384 = 82.29% |
| v3 | 314/384 = 81.77% | 384/384 = 100% | 314/384 = 81.77% |

v3 audit 的主要收益是 evidence 行从 57 个空缺变成 0，即格式/指令遵循提升；
answer-only 反而少 2 个 case。因此 audit strict 的 +13.28pp 不能当作事件问答语义的提升。

## 5. 68 个 production 错例的四帧 RGB 全量复核

审计方法：从每个错例的 `history_rgb_paths_all4` 取出 `t0/t1/t2/t3`，按 `GT→pred`
分组，对 68 个 case 的 272 张 RGB 全部逐帧查看。2RGB 模型实际只看 `t0/t3`；
`t1/t2` 只用来判断错误是 prompt/模型问题，还是 endpoint 信息不足，不把未输入信息当作模型证据。
临时 contact sheet 仅用于本地查看，不作为可再生大产物入库。

| GT→pred | 数量 | case_index | 逐帧视觉归因 |
|---|---:|---|---|
| UE3→RE | 15 | 28, 220, 232, 356, 65, 205, 221, 233, 277, 46, 270, 71, 115, 279, 363 | 主体是 ParkingCutIn/侧方车，多例能看到车头或车身逐帧向车道边界内移；v3 静态排除过强。少数雾天/黑夜只有弱证据。 |
| UE1→RE | 14 | 92, 308, 360, 37, 89, 185, 285, 309, 321, 2, 62, 134, 178, 327 | 大多只能确认同车道前车存在；黑夜/雨天下急减速幅度很难由 RGB 确证。不宜靠放宽 prompt 追标签。 |
| UE6→RE | 8 | 236, 304, 376, 225, 289, 26, 98, 263 | 部分有明显横穿冲突；部分最新帧冲突车已离开或视野中几乎无车，属于时序/标签边界，不做统一放宽。 |
| INVALID→RE | 7 | 32, 164, 13, 105, 241, 257, 175 | wrong-domain 是人工构造的提问域错配，部分画面本身不足以稳定区分 continuous corridor/local junction；保留现有定义。 |
| RE→UE3 | 6 | 4, 153, 50, 274, 350, 63 | 包含事故/静态车、ego 经过侧车和已驶离冲突；说明 v3 的静态反例不能删，v4 只增加“相对边界持续侵入”例外。 |
| UE5→RE | 4 | 40, 180, 196, 113 | 对向侵入车往往很小、黑夜或雾中对比度低；无重复性 prompt 误导证据。 |
| RE→INVALID | 3 | 344, 135, 287 | 一例普通道路、一例高速、一例街道入口；是局部布局误判，现有“highway 不 invalid”已覆盖。 |
| RE→UE6 | 3 | 288, 137, 91 | 只看到普通路口/远处灯光/阻塞路口，缺少明确违规与优先权证据；现有 UE6 边界正确。 |
| UE3→UE1 | 3 | 296, 90, 191 | 两例前车+侧方动态交互，一例雾天侧车；属 UE1/UE3 证据归属混淆，v4 的侧车跨边界提示可直接针对。 |
| INVALID→UE6 | 2 | 58, 278 | 画面有路口/侧向车，但对人工 wrong-domain 的适用性判断不稳；不改 UE6 语义。 |
| RE→UE5 | 2 | 97, 31 | 新帧中几乎没有可见对向侵入车；现有 UE5 时序边界正确。 |
| RE→UE1 | 1 | 88 | 黑夜中普通前车/尾灯，无可见突然减速；不改。 |

## 6. prompt v4：只修 UE3，其他保留

修改为 `sft_new_loop_phase2_direct_event_visual_v4`。在 v3 的静态/视差边界上只增加：

> 当 oldest-to-newest 显示停车位或路边车辆持续向可见车道边界靠近、跨越，或逐步侵入
> usable corridor 时，即使最新帧还有部分车身在停车区、姿态仍像停车，也判 UE3 YES。

同时保留：单帧斜姿、单纯变大/图像位移、静态事故/施工、ego 经过真正未动的
路边车都不是 UE3。UE1、UE5、UE6、INVALID 没有可证实的统一 prompt 问题，本次不改。

## 7. 后续训练的验收口径

1. 本 384-case 已被用来设计 v3/v4 prompt，从现在起只能视为 dev/audit set，
   不再当作独立 test 声称泛化提升。
2. 当前 test index 有 840 行（6 桶各 140）；相对当前 384 行还有 456 行（6 桶各 76）
   未参与本轮误差归因。应冻结这 456 行做一次性 unseen 验收。
3. 训练内部首先要求 32/桶 generation val 上 UE3 target recall `>=0.625`；
   再比较总 exact，不能用 UE5/UE6/INVALID 的收益遮住 UE3 崩塌。
4. unseen 验收至少报告总 exact、UE1/UE3/UE5/UE6/INVALID P/R/F1、RE highway/local，
   并单独比较 `UE3→RE` 和 `RE→UE3`。
5. audit evidence 合规率只表示指令遵循，不代替 production 语义分数。
