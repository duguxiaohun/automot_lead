# Fused Phase1 + Phase2 v5：重复训练结果审计与冻结决策（2026-09-02）

## 1. 审计范围

本次审计比较以下两个最新 bundle：

- `checkpoints/sft_new_loop_phase1_20260902_105430_2rgb_endpoints_audit_bundle`
- `checkpoints/sft_new_loop_phase1_20260902_120501_4rgb_audit_bundle`

严格历史基线为：

- `checkpoints/sft_new_loop_phase1_20260831_144428_2rgb_endpoints_audit_bundle`
- `checkpoints/sft_new_loop_phase1_20260831_125632_4rgb_audit_bundle`

另外对照：

- `FUSION_V5_4RGB_2RGB_RESULT_RGB_AUDIT_20260831.md`；
- `FUSED_PHASE1_PHASE2_EVAL_ANALYSIS_20260824.md`；
- `sft_loop_phase2_augment/CKPT20000_EVAL_ANALYSIS.md`。

本文只比较同一评测合同下的结果，不把 scenario 名当视觉证据，也不根据错例名称推断标签。

## 2. 总结结论

1. **融合训练本身有效。** 最新 2RGB/4RGB LoRA 相对 base 的联合 exact 仍分别提高
   `50.195 pp` / `46.192 pp`。
2. **最新重复训练没有超过 2026-08-31 的 v5 最优权重。** 2RGB 基本持平但净退化 5/1024；
   4RGB 净退化 50/1024，不能晋升。
3. **退化不是新 prompt 或测试集变化造成。** 同一 RGB mode 的新旧 bundle 使用相同 v5
   prompt/hash、相同 1024 个 case、相同 base 逐例输出；严格配对差异来自 adapter/checkpoint。
4. **融合后 Phase1 总体仍优于独立 Phase1；Phase2 是混合结果。** subset 问法改善，
   hierarchical 能力相对独立 Phase2 明显被稀释。
5. **不继续创建 v6，也不回退到 v4。** v5 已在同一协议下产生过当前全局最佳结果；重复训练
   回退不能证明 prompt 有错。围绕同一 test 错例继续扩写，会把固定 test 逐步变成开发集。
6. **从本审计起冻结 v5 prompt 与当前训练/评测协议。** production 使用 2026-08-31 已验证的
   `best_generation/step-40000`；`best_generation_balanced` 继续只作诊断候选。

## 3. 严格可比性

| 项目 | 2RGB 新旧对比 | 4RGB 新旧对比 |
|---|---|---|
| production prompt | v5，相同 | v5，相同 |
| production hash | `a932e5a86d89ca553339a7828a8a4da38170f20d1d841e43217390d7b1cbee0c` | `5b7f8c60d175960c0f22e23d1cb827b8010258db6f3043182c21611f7bec1430` |
| RGB indices | `[0, 3]` | `[0, 1, 2, 3]` |
| case 数 | 1024 | 1024 |
| case identity | 逐例完全相同 | 逐例完全相同 |
| base 逐例结果 | 完全相同 | 完全相同 |
| adapter step | 旧 40000 / 新 40000 | 旧 40000 / 新 20000 |

因此同 mode 的旧→新变化不是抽样误差、prompt 漂移或 base 推理漂移。

## 4. 核心结果

### 4.1 上一轮 v5 → 最新重复训练

| RGB mode | 指标 | 2026-08-31 v5 | 2026-09-02 | 变化 |
|---|---:|---:|---:|---:|
| 2RGB endpoints | 联合 exact | **76.562%** | 76.074% | -0.488 pp |
| 2RGB endpoints | Phase1-only exact | 89.844% | **90.039%** | +0.195 pp |
| 2RGB endpoints | Phase2-only exact | **83.691%** | 82.617% | -1.074 pp |
| 4RGB | 联合 exact | **77.930%** | 73.047% | **-4.883 pp** |
| 4RGB | Phase1-only exact | **89.941%** | 87.012% | -2.930 pp |
| 4RGB | Phase2-only exact | **83.984%** | 80.273% | -3.711 pp |

严格配对翻转：

- 2RGB：34 个旧错→新对、39 个旧对→新错，净 `-5`；Phase1 净 `+2`，Phase2 净 `-11`。
- 4RGB：48 个旧错→新对、98 个旧对→新错，净 `-50`；Phase1 净 `-30`，Phase2净 `-38`。

2RGB 可以视为平台附近的小幅波动；4RGB 是广泛、明确的泛化回退。

### 4.2 相对 base Qwen

| RGB mode | Base 联合 exact | 最新 LoRA | 提升 |
|---|---:|---:|---:|
| 2RGB endpoints | 25.879% | 76.074% | **+50.195 pp** |
| 4RGB | 26.855% | 73.047% | **+46.192 pp** |

所以本轮不能解释成融合任务没有学会；问题是新训练实例没有复现历史最佳泛化。

### 4.3 相对独立 Phase1

| RGB mode | 原 Phase1 macro F1 / exact | 最新 fused Phase1 macro F1 / exact |
|---|---:|---:|
| 2RGB endpoints | 86.62% / 85.16% | **88.30% / 90.04%** |
| 4RGB | 86.03% / 84.77% | **89.15% / 87.01%** |

融合后 Phase1 整体仍提升，但最新 4RGB 的 `VULNERABLE` focus F1 降到 `77.78%`，不能把
整体提升理解成所有四问都提升。

### 4.4 相对独立 Phase2 augment

独立 Phase2 checkpoint-20000 为总 exact `82.8%`，all/subset/hierarchical 为
`76.5% / 81.1% / 97.1%`。

| RGB mode | Phase2 总 exact | all | subset | hierarchical |
|---|---:|---:|---:|---:|
| 独立 Phase2 | 82.8% | 76.5% | 81.1% | **97.1%** |
| 最新 2RGB fused | 82.617% | 76.758% | **85.938%** | 91.016% |
| 最新 4RGB fused | 80.273% | 73.047% | **85.156%** | 89.844% |

融合显著改善 subset，但没有保持独立 Phase2 的 hierarchical 上限；最新 4RGB 在总量、all 和
hierarchical 上均下降。

## 5. 退化分布

### 5.1 2RGB

- `all_random_order`：72.461% → 71.289%；
- `subset_random`：78.516% → 78.516%；
- `hierarchical_probe`：82.813% → 83.203%；
- 八个 focus macro F1：88.65% → 87.44%。

退化较小，主要来自 Phase2 RS1/RS2/RS4 和 Phase1 VULNERABLE 的少量净回退；没有出现单一
variant 全面崩溃。

### 5.2 4RGB

- `all_random_order`：72.070% → 65.430%，下降 **6.641 pp**；
- `subset_random`：80.078% → 76.172%；
- `hierarchical_probe`：87.500% → 85.156%；
- 八个 focus macro F1：88.77% → 86.22%；
- `VULNERABLE` 主 focus：0 修复 / 9 回退；
- RS1/RS2/RS4 均下降，只有 RS5 focus 有小幅改善。

这不是一条 prompt 边界导致的孤立错误，而是新 4RGB checkpoint 的广泛能力回退。

## 6. audit prompt 不能挽救本轮结果

| RGB mode | 上一轮 audit exact | 最新 audit exact | 变化 |
|---|---:|---:|---:|
| 2RGB endpoints | 76.660% | 75.781% | -0.879 pp |
| 4RGB | 77.344% | 72.363% | -4.981 pp |

最新 audit parser 中证据完整性也下降：2RGB `evidence_complete` 为 997/1024，4RGB 为
992/1024。切换到 audit prompt 不会解决 production 语义退化，production prompt 应继续作为
正式推理合同。

## 7. 代表性逐帧 RGB 核查

以下观察均来自实际四帧 stitched RGB，不依据 scenario 名推断。

| case | route / frames | GT→最新输出 | 四帧视觉结论 | 归因 |
|---|---|---|---|---|
| 108 | `CrossingBicycleFlow/...2344_2...`, 107–110 | VULNERABLE YES→NO | 道路和车辆清楚，但四帧没有可靠可辨识的骑行者 | 当前窗口弱证据/标签强于 RGB |
| 121 | `VehicleTurningRoute/...Scenario4_6...`, 73–76 | VULNERABLE YES→NO；RS4 YES→NO | 浓雾、远处目标很小；没有足够清晰的 vulnerable user 或交通灯头 | 弱证据，不应加入场景名先验 |
| 207 | `VehicleTurningRoutePedestrian/...398_0...`, 155–158 | VULNERABLE YES→NO | 极暗夜景，主要可见车辆灯光，没有可靠行人证据 | 不宜用 prompt 强迫 YES |
| 299 | `ConstructionObstacleTwoWays/...1963_0...`, 48–51 | STATIC YES→NO；RS2 YES→NO | 前方箱式车辆及其底部橙色施工/警示物可见 | 真实视觉漏检，说明新 checkpoint 有能力回退 |
| 371 | `CrossJunctionDefectTrafficLight/...002189...`, 39–42 | LIGHT YES→NO | 路口已在后方/边缘，未见清晰可读的异常灯头 | 当前帧证据不足 |
| 48 | `InterurbanAdvancedActorFlow/...3489_8...`, 105–108 | RS1 YES→NO | 极暗且下雨，拓扑只能局部辨认 | RS 标签视觉可判性弱 |
| 330 | `BlockedIntersection/...Town06_44...`, 101–104 | RS1 NO→YES | 浓雾且主要被前车遮挡，局部道路开口不清 | 遮挡/能见度导致结构歧义 |

样本同时包含“模型真实漏检”和“当前 RGB 不足以支持 GT”两类。正确处理方式是继续保留
`visible RGB evidence` 边界，并在标签/评测侧标记视觉可判性；不能在 prompt 中加入
`CrossingBicycleFlow => VULNERABLE=YES` 一类 scenario 先验。

## 8. 训练过程与 checkpoint 波动

| RGB mode | 选中权重 | bundle 内日志末步 | 计划总步数 | generation validation 最优 |
|---|---:|---:|---:|---:|
| 最新 2RGB | step-40000 | 55960 | 110592 | step-40000 / 76.562% |
| 最新 4RGB | step-20000 | 46460 | 110592 | step-20000 / 79.297% |

两次训练都在计划训练尚未结束时打包。上一轮 4RGB 的 validation 最优为 step-40000 / 81.25%，
也高于本轮 79.297%；因此 4RGB test 回退与 validation 趋势一致，不只是 test 偶然波动。

generation validation 默认每个 focus×YES/NO 桶 16 个，共 256 case。它适合训练期快速选点，
但不足以证明多个只差 1–2 pp 的 checkpoint 有稳定排序。固定 test 已被多轮 prompt/权重审计
使用，不应再用它驱动逐句 prompt 搜索。

## 9. 冻结版本决定

### 9.1 冻结 prompt

- 名称：`sft_new_loop_phase1_phase1_phase2_combined_v5_rgb_audited_rs4_hardware`
- 4RGB production SHA256：
  `5b7f8c60d175960c0f22e23d1cb827b8010258db6f3043182c21611f7bec1430`
- 2RGB endpoints production SHA256：
  `a932e5a86d89ca553339a7828a8a4da38170f20d1d841e43217390d7b1cbee0c`

不创建 v6，不回退 v4，不再根据同一 1024-case test 的零散错误修改 prompt。

### 9.2 冻结代码协议

冻结当前 `sft_new_loop_phase1` 的以下行为：

- Phase1 四问 + Phase2 all/subset/hierarchical 单轮融合；
- 八个 focus 各自 YES:NO=1:1，Phase1/Phase2 focus 总量 1:1；
- train variant `4:1:1`，eval/generation variant `2:1:1`；
- `FOCUS_BALANCE_COUNT=9216`、`MAX_TRAIN_FRAME_REPEAT=10`；
- focus semantic weight `1.0`、非 focus semantic weight `0.1`；
- production prompt/hash 与 adapter RGB mode 在加载前硬校验；
- 自由生成严格行数/顺序/重复/额外文本解析；
- `best_generation` 作为 production 主候选，`best_generation_balanced` 仅作诊断。

冻结时核心文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `prompts.py` | `53574f9543005b351d574af976651f4e5b2d0b7d2b9e9c3ea4d5eb44ad16e471` |
| `build_dataset.py` | `8aeef46b9e54745b22cc765cc9b29e1e593a0f0e4f78b321e2bf151e1cb8e1c1` |
| `train.py` | `066c1a8161f37ddd5b7c8c71748d077f97b9bb5aaf8a55554dc488c795b7fe9c` |
| `eval.py` | `0db16457a3d2c2090eadffae476282355d433d089bc3b573475ce65a08cceb90` |
| `history_rgb.py` | `742e33c70c085e87a6c972fcc6304cacebe1e28a5bc53fa21b19e42cf2e6d074` |
| `run_full_pipeline.sh` | `33900e512455346a2b39c51f035ee0997d92ed98eaba0a0a50449424878ab92e` |
| `train.sh` | `3c2b4d8d831fd91693a16c9f72a7f11ab8a2ecb2cb70abac81fdbaabfa3087cc` |
| `eval.sh` | `bda6e1411faef28f1b85701fcf07beebaeed376194447690c466aae5cab642e9` |

文档更新不改变上述指纹。今后可以修复明确的实现 bug、路径兼容或日志问题，但任何会改变
prompt、样本语义、采样比例、loss 权重、解析口径或 checkpoint 排序的改动，都属于协议解冻，
必须重新建立独立验证依据。

### 9.3 冻结 production 权重

质量优先默认使用已验证的 4RGB：

```text
checkpoints/sft_new_loop_phase1_runs/run_20260829_235223_4rgb_combined_phase1_phase2_4rgb/best_generation
step-40000, strict joint exact=77.930%
```

显存/时延优先可使用已验证的 2RGB endpoints：

```text
checkpoints/sft_new_loop_phase1_runs/run_20260829_235241_2rgb_endpoints_combined_phase1_phase2_2rgb_endpoints/best_generation
step-40000, strict joint exact=76.562%
```

以下最新重复训练权重不晋升：

- `run_20260831_170526_4rgb.../best_generation`（step-20000，73.047%）；
- `run_20260831_170554_2rgb_endpoints.../best_generation`（step-40000，76.074%）。

## 10. 解冻条件

只有满足以下至少一项，才重新讨论 prompt/训练协议：

1. 在预先冻结、此前从未用于 prompt 或 checkpoint 决策的 unseen 集上出现可复现退化；
2. 至少两个 seed 均出现同方向、同视觉子类的显著错误；
3. 对同一错误子类逐帧看完 RGB 后，确认 GT 有清晰视觉证据，且当前 v5 定义确实缺失或矛盾；
4. 发现实现 bug 导致图像顺序、标签、loss、解析或 checkpoint 加载与文档合同不一致。

不构成解冻依据：单个 scenario 名、不可见/极暗/严重遮挡样本、同一 test 上少量计数波动、
或只在 audit 文风而非 production 语义上发生的变化。

## 11. 最终判断

`sft_new_loop_phase1` 已达到适合冻结的阶段：v5 prompt 与当前严格代码协议保留；4RGB 历史
step-40000 是质量优先 production，2RGB 历史 step-40000 是低成本候选。本轮最新权重说明训练
存在 seed/checkpoint 波动，但没有提供继续改 prompt 的证据。后续研发资源应转向独立 unseen
验收、下游驾驶效果或下一阶段任务，而不是继续围绕同一 1024-case test 做 prompt 微调。
