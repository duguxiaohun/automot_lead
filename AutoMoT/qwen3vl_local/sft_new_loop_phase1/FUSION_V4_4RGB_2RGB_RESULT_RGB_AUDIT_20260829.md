# Fused Phase1+Phase2 v4：最新结果与逐帧 RGB 审计（2026-08-29）

## 1. 审计范围与结论

本轮只分析以下正式 test bundle，并与 2026-08-27 的 v3 同 case 结果、原 Phase1、原
Phase2 augment 和 2026-08-24 旧 fused 记录对比：

- `checkpoints/sft_new_loop_phase1_20260829_170137_2rgb_endpoints_audit_bundle`
- `checkpoints/sft_new_loop_phase1_20260829_171816_4rgb_audit_bundle`

逐帧审计实际打开了：

- 4RGB `TRAFFIC_LIGHT_ABNORMAL` 全部 17 个 focus FN 所覆盖的 9 条 route/连续窗口；
- 2RGB 相对 v3 新增的 4 个 RS4 focus 回退 case，并额外查看模型未输入的两个中间帧；
- 4RGB hierarchical `RS_HIGHWAY` 全部 7 个 false positive 的四帧 RGB。

核心结论：

1. v4 相对 v3 明确提升，两种模式的联合 exact、Phase1-only、Phase2-only 都提高；4RGB
   联合 exact 从 70.80% 恢复到 76.27%。
2. v4 的两项 RGB 驱动修改有效：`RS_HIGHWAY` false positive 大幅减少，audit evidence
   合同从不稳定恢复到 94.82%/99.22%。不回退 v4。
3. 4RGB 灯异常 F1 的下降不能直接解释为 prompt 变差。17 个 FN 中，多数窗口只有正常的
   “一个方向红、另一个方向绿”，或完全没有可读信号头；旧模型答 YES 更接近贴 scenario/
   route 标签，而不是更好的 RGB 判断。不能为追表面分数放宽灯异常规则。
4. 2RGB 的两个 RS4 false positive 在 RGB 中确实只有装饰路灯、裸杆/灯臂、雾中车灯，
   没有可识别交通信号头。v5 只增加这一条硬件辨识边界；其余 prompt 保留。
5. 单一 generation joint exact 会掩盖最弱 focus。代码继续保留 `best_generation/`，并新增
   `best_generation_balanced/` 作为额外候选；新代码/新 prompt 的效果必须重训后验证，本文
   不把尚未训练的 v5 写成已经获得提升。

## 2. Adapter 与可比性

| 项 | 2RGB endpoints | 4RGB |
|---|---:|---:|
| 实际输入索引 | `[0,3]` | `[0,1,2,3]` |
| best_generation step | 26000 | 48000 |
| prompt | v4 RGB-audited RS_HIGHWAY | v4 RGB-audited RS_HIGHWAY |
| non-focus 语义权重 | 0.1 | 0.1 |
| test cases | 1024 | 1024 |
| adapter/prompt hash 匹配 | 是 | 是 |

两个新 bundle 与各自 2026-08-27 v3 bundle 的 1024 个 test case 完全相同，因此 v3→v4
可以逐例配对。2RGB 与 4RGB 之间不是严格帧数消融：checkpoint step、训练行数和 bundle
记录的 commit/dirty 状态不同，只能比较当前产物，不能作纯帧数因果结论。

## 3. 最新 production 结果

### 3.1 Base → LoRA

| 模式 | 模型 | 联合 exact | Phase1-only | Phase2-only |
|---|---|---:|---:|---:|
| 2RGB | base | 25.781% | 63.574% | 31.152% |
| 2RGB | LoRA | **73.340%** | **87.695%** | **81.543%** |
| 4RGB | base | 26.855% | 67.969% | 30.957% |
| 4RGB | LoRA | **76.270%** | **88.965%** | **82.910%** |

### 3.2 相对 v3 的同 case 变化

| 模式 | 指标 | v3 | v4 | 变化 |
|---|---|---:|---:|---:|
| 2RGB | 联合 exact | 70.312% | **73.340%** | **+3.027 pp** |
| 2RGB | Phase1-only | 82.812% | **87.695%** | **+4.883 pp** |
| 2RGB | Phase2-only | 77.637% | **81.543%** | **+3.906 pp** |
| 4RGB | 联合 exact | 70.801% | **76.270%** | **+5.469 pp** |
| 4RGB | Phase1-only | 85.059% | **88.965%** | **+3.906 pp** |
| 4RGB | Phase2-only | 78.711% | **82.910%** | **+4.199 pp** |

逐例迁移：

- 2RGB joint：v3 错→v4 对 113，v3 对→v4 错 82，净修正 31；
- 4RGB joint：v3 错→v4 对 112，v3 对→v4 错 56，净修正 56。

### 3.3 八个 focus F1

| focus | 2RGB v3 → v4 | 4RGB v3 → v4 |
|---|---:|---:|
| HIGHWAY | 96.970% → **99.213%** | **99.225%** → 98.438% |
| STATIC_OBSTACLE | 87.603% → **89.831%** | **90.909%** → 89.256% |
| VULNERABLE | 86.207% → **88.136%** | 83.478% → **90.909%** |
| TRAFFIC_LIGHT_ABNORMAL | 92.562% → **92.683%** | **93.443%** → 84.685% |
| RS1 | 59.813% → **73.585%** | 68.376% → **78.049%** |
| RS2 | 78.947% → **88.406%** | 82.051% → **93.846%** |
| RS4 | **95.238%** → 92.063% | 88.189% → **88.710%** |
| RS5 | 86.726% → **88.333%** | 81.739% → **83.478%** |
| Phase1 Macro F1 | 90.835% → **92.465%** | **91.764%** → 90.822% |
| Phase2 Macro F1 | 80.181% → **85.597%** | 80.089% → **86.021%** |

4RGB Phase1 macro 的小降几乎全部来自灯异常；第 6 节的 RGB 显示这部分混有明显的标签/
可见性冲突，不能机械地用 prompt 放宽追回。

### 3.4 Phase2-only variant exact

| variant | 2RGB v3 → v4 | 4RGB v3 → v4 |
|---|---:|---:|
| all_random_order | **76.172%** → 75.195% | 72.852% → **75.977%** |
| subset_random | 77.344% → **86.328%** | 83.203% → **87.891%** |
| hierarchical_probe | 80.859% → **89.453%** | 85.938% → **91.797%** |

v4 对 subset/hierarchical 的恢复明显；2RGB all-random 仍是一个真实短板。

## 4. 与原独立训练和旧 fused 对比

### 4.1 原 Phase1

| 模式 | 原 Phase1 Macro F1 / exact | 最新 fused Macro F1 / Phase1-only |
|---|---:|---:|
| 2RGB | 86.62% / 85.16% | **92.47% / 87.70%** |
| 4RGB | 86.03% / 84.77% | **90.82% / 88.96%** |

融合后的 Phase1 总体超过原独立训练，不是以牺牲全部 Phase1 能力换取 Phase2。

### 4.2 原 Phase2 augment checkpoint-20000

原模型为总 exact 82.8%，all/subset/hierarchical 为 76.5%/81.1%/97.1%。最新 4RGB
Phase2-only 为 82.91%，三类为 75.98%/87.89%/91.80%。综合已追平，subset 更强，但
hierarchical 专项仍低 5.30 pp。

### 4.3 2026-08-24 旧 fused

| 指标 | 旧 fused | 最新 4RGB v4 | 差值 |
|---|---:|---:|---:|
| 联合 exact | **77.54%** | 76.27% | -1.27 pp |
| Phase1-only | **89.75%** | 88.96% | -0.78 pp |
| Phase2-only | **84.47%** | 82.91% | -1.56 pp |

所以 v4 已恢复 v3 的大部分损失，但还不是所有历史口径上的绝对最优。

## 5. v4 已验证有效的两项修改

### 5.1 RS_HIGHWAY

| 模式 | v3 F1 / FP | v4 F1 / FP |
|---|---:|---:|
| 2RGB | 84.163% / 35 | **94.359% / 10** |
| 4RGB | 88.288% / 24 | **95.098% / 7** |

明确 limited-access 拓扑链，并把普通双黄线/城际道路/雾/黑暗/单护栏列为不足证据，是有效
修复，应保留。

### 5.2 Audit evidence 合同

| 模式 | answers valid v3 → v4 | contract valid v3 → v4 | audit exact v3 → v4 |
|---|---:|---:|---:|
| 2RGB | 82.617% → **100%** | 80.957% → **94.824%** | 57.715% → **73.828%** |
| 4RGB | 51.172% → **100%** | 44.043% → **99.219%** | 33.789% → **76.172%** |

强制 `EVIDENCE_*` 前缀、证据开始后不再重复答案行的修改已经被新 adapter 验证，应保留。

## 6. 逐帧 RGB 审计

### 6.1 4RGB 灯异常 17 个 FN

| route / anchor | 实际查看帧 | RGB 结论 | prompt 决策 |
|---|---|---|---|
| Town12 route_002113 / f43 | 0040-0043 | 路口逐渐离开视野；没有可读的冲突信号头组合。 | 保留 NO witness 门槛。 |
| Town13 route_002189 / f39,f42 | 0036-0042 | 早帧是左侧红、前向绿，属于正常相位候选；后帧信号已离开视野。 | 不把普通 red+green 改成异常。 |
| Town05 route_002090 / f47 | 0044-0047 | 仅最早帧右缘有一个小绿灯，其后不可见；无同冲突区异常组合。 | 不改。 |
| Town03 route_002134 / f51 | 0048-0051 | 雾中普通道路和车辆，完全没有可读信号头。 | GT 可见性冲突；不追标签。 |
| Town04 route_002178 / f48,f49,f52 | 0045-0052 | 暗雨中沿普通道路前进，没有可读交通信号硬件。 | GT 可见性冲突；不追标签。 |
| Town04 route_002168 / f39 | 0036-0039 | 雾中住宅道路，只有装饰路灯，无交通信号。 | GT 可见性冲突；不追标签。 |
| Town05 route_002067 / f17,f18 | 0014-0018 | 宽路口可见前向红与右侧绿；这是正常互斥相位的典型外观，没有两组冲突绿灯。 | 保留“red+green 默认正常”。 |
| Town12 route_002111 / f31,f36,f41,f44 | 0028-0044 | 暗雨中左侧红、前/右一个绿，未见两条冲突 approach 同时绿；后期信号离开视野。 | 不放宽。 |
| Town12 route_002110 / f31,f34 | 0028-0034 | 雾中左红、前绿；没有额外冲突绿灯证据。 | 不放宽。 |

这 17 个 FN 全部 GT=YES，但逐帧 RGB 没有形成当前定义所需的异常硬件 witness。此处若让
prompt 看到普通红绿组合就答 YES，会系统性制造普通路口 false positive。

### 6.2 2RGB RS4 的 4 个 v3→v4 回退

| case | GT / v4 | 四帧 RGB 结论 | 处理 |
|---|---:|---|---|
| DynamicObjectCrossing Town13 f196 | YES / NO | 雾中三岔/弯路，只有普通灯杆，没有可识别 signal head。 | 标签/可见性冲突，不放宽。 |
| VehicleTurningRoute Town05 f82 | YES / NO | 宽直城市道路，没有局部路口控制，也没有信号头。 | 标签/可见性冲突，不放宽。 |
| DynamicObjectCrossing Town07 f34 | NO / YES | 雾中乡间双向道路；可见车灯和路边环境，无 signal head。 | 真实 FP，强化硬件辨识。 |
| ControlLoss Town10HD f141 | NO / YES | 雾中城市道路、车辆与装饰向下路灯，无 traffic-signal head。 | 真实 FP，强化硬件辨识。 |

因此 v5 只补一句：RS4 必须看见治理本地路口的可识别交通信号头；装饰/向下路灯、裸杆/
灯臂和车辆灯光都不是信号硬件。Phase2 原规则已有相同方向，本次是在融合 prompt 的最终
recheck 中提高显著性，不改 RS4 正类定义。

### 6.3 4RGB RS_HIGHWAY 的 7 个残余 FP

| case 类型 | RGB 结论 | 处理 |
|---|---|---|
| PriorityAtJunction 暗夜双黄线道路 | 普通有对向中心线的地面道路，不是 limited-access。 | v4 已明确覆盖，不继续堆词。 |
| InterurbanAdvancedActorFlow 开放田野道路 | 单车道/普通道路边缘，无隔离/匝道/受控出入口链。 | v4 已覆盖。 |
| ConstructionObstacleTwoWays 林荫城市道路 | 路缘、普通地面道路、开放侧向环境。 | v4 已覆盖。 |
| VehicleTurningRoute 大型环岛/地面冲突区 | 有高架背景，但 ego 正受 at-grade 冲突区治理。 | v4 的 urban-overpass NO 已覆盖。 |
| Accident 夜间多车道 | 多同向车道、边缘护栏和受控走廊清晰；模型 YES 有充分 RGB 依据。 | 记录为 RS 标签/视觉语义冲突。 |
| ConstructionObstacle 夜间施工 | 多同向车道、双侧连续护栏、无普通道路接入，视觉上是 highway。 | 记录为标签/视觉语义冲突。 |
| ControlLoss 夜间多车道 | 双侧连续隔离/声屏障、多同向车道、受控走廊。 | 记录为标签/视觉语义冲突。 |

7 个 FP 中至少 3 个从 RGB 看更支持模型而非 GT；继续收紧 RS_HIGHWAY 会伤害真正高速正例，
所以保留 v4 定义。

## 7. 已实施的 v5 最小修改

### 7.1 Prompt

- prompt 名升级为 `sft_new_loop_phase1_phase1_phase2_combined_v5_rgb_audited_rs4_hardware`；
- 只在最终 visual recheck 加一条 RS4 硬件辨识句；
- 不修改 Phase1 四问主体、RS1/RS2/RS4/RS5 canonical 定义、RS_HIGHWAY 定义、audit 合同；
- 不根据 CrossJunctionDefectTrafficLight 名称或 GT 放宽灯异常。

### 7.2 Checkpoint 选择

- 保留 `best_generation/`：format gate 后最大联合 exact，兼容旧行为；
- 新增 `best_generation_balanced/`：format gate 后按
  `(八 focus 最低 accuracy, 联合 exact, focus macro accuracy)` 字典序选优；
- 两个 checkpoint 都应进入正式 test，不能用 balanced val 指标直接宣称 test 已提升。

以本轮已保存的 validation 日志离线代入新规则：

- 2RGB balanced 仍会选择 step 26000；
- 4RGB balanced 会选择 step 42000，而 exact-only 为 step 48000。

这只说明新规则会留下一个更均衡候选，不代表 step 42000 的正式 test 一定优于 step 48000。

## 8. 下一轮验收

1. v5 必须重新训练；v4 adapter 的 prompt hash 不兼容，禁止强挂。
2. 4RGB/2RGB 各自同时评测 `best_generation` 与 `best_generation_balanced`。
3. 主指标继续报告联合、Phase1-only、Phase2-only 和三 variant，不用单一 exact 覆盖细项。
4. 灯异常必须单列视觉可证实错例；普通 red+green、无可读信号头的 GT=YES 不作为扩 prompt
   依据，先进入标签/时间边界复核。
5. 重点观察 RS4 NO precision 是否改善，同时确保 RS4 YES recall、TRAFFIC_LIGHT_ABNORMAL
   正类 precision 不下降。
6. 若 v5 没有改善 RS4 FP，先回退新增的一句到 v4；不要继续扩写其它表现已稳定的规则。
