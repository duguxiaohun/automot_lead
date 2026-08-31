# Fused Phase1 + Phase2 v5：4RGB / 2RGB 结果与逐帧 RGB 审计（2026-08-31）

## 1. 审计范围与结论

本次对比以下两个最新正式 bundle，并回查 v4、v3、原 Phase1、原 Phase2 augment 记录：

- `checkpoints/sft_new_loop_phase1_20260831_125632_4rgb_audit_bundle`
- `checkpoints/sft_new_loop_phase1_20260831_125839_2rgb_endpoints_audit_bundle`

两者都使用 production prompt
`sft_new_loop_phase1_phase1_phase2_combined_v5_rgb_audited_rs4_hardware`，都选择
`best_generation/step-40000`，每个 RGB mode 各评测 1024 个按八个 focus 严格平衡的 test case。

结论：融合后的整体问答能力相对原 Phase1、原 Phase2 augment 和 fused v3 都有明确提升；
v5 相对 v4 的联合 exact 也继续提升。但提升并不均匀，4RGB 的 Phase2 focus macro、2RGB 的
Phase1 focus macro 出现回退。逐帧 RGB 表明其中一部分是标签/当前帧可见性冲突，不能直接据此
继续扩写提示词。v5 新增的 RS4 灯头辨识句语义正确，但没有消除雾天车灯/路灯假阳性，且部分
表面收益来自模型拟合了与 RGB 不一致的 RS4=YES 标签。因此本轮保留 v5、不新增 v6、也不回退
到 v4；代码侧改为正式流程自动补测不同权重的 balanced checkpoint。

## 2. 总体结果

### 2.1 最新 base 与 LoRA

| RGB mode | 模型 | 联合 exact | Phase1 exact | Phase2 exact |
|---|---:|---:|---:|---:|
| 4RGB | base | 26.855% | 66.992% | 31.543% |
| 4RGB | LoRA v5 | **77.930%** | **89.941%** | **83.984%** |
| 2RGB endpoints | base | 25.879% | 64.258% | 32.520% |
| 2RGB endpoints | LoRA v5 | **76.562%** | **89.844%** | **83.691%** |

LoRA 相对 base 的增益很大，说明融合 SFT 确实学习到了四个可见事实问题和 Phase2 道路结构
问法，而不是只靠 base prompt。4RGB 当前略高于 2RGB，但两套 test case 只重合 2/1024，且
训练行数也不同，不能把 1.37 个百分点直接解释成“中间两帧的因果收益”。

### 2.2 v4 → v5，同一 RGB mode、同一 1024 case 的配对对比

| RGB mode | 指标 | v4 | v5 | 变化 |
|---|---:|---:|---:|---:|
| 4RGB | 联合 exact | 76.270% | **77.930%** | **+1.660pp** |
| 4RGB | Phase1 exact | 88.965% | **89.941%** | **+0.977pp** |
| 4RGB | Phase2 exact | 82.910% | **83.984%** | **+1.074pp** |
| 2RGB | 联合 exact | 73.340% | **76.562%** | **+3.223pp** |
| 2RGB | Phase1 exact | 87.695% | **89.844%** | **+2.148pp** |
| 2RGB | Phase2 exact | 81.543% | **83.691%** | **+2.148pp** |

配对切换也支持整体净提升：4RGB 联合 wrong→right 72、right→wrong 55，净增 17；2RGB 为
88/55，净增 33。相对 fused v3，4RGB 联合/Phase1/Phase2 分别提升
7.129/4.882/5.273pp，2RGB 分别提升 6.250/7.032/6.054pp。

与 2026-08-24 的旧 fused v1 4RGB 同身份样本相比，v5 联合 exact +0.391pp、Phase1
+0.195pp，但 Phase2 -0.489pp。也就是说整体没有倒退，但 Phase2 已接近平台期，不能只看
联合 exact 就宣布所有 RS 问题都变好。

### 2.3 与融合前单独训练结果

- 原 Phase1：4RGB focus macro/exact 为 86.03%/84.77%，最新为
  **92.98%/89.94%**；2RGB 原为 86.62%/85.16%，最新为 **89.89%/89.84%**。
- 原 Phase2 augment：总 exact 82.8%，all/subset/hierarchical 为
  76.5%/81.1%/97.1%。最新 4RGB 为 **83.984%**，三类为
  **77.734%/88.672%/91.797%**；最新 2RGB 为 **83.691%**，三类为
  **77.930%/87.109%/91.797%**。

融合对 Phase1 和 Phase2 总体都有收益，尤其 subset；hierarchical 则低于旧独立 Phase2，
表明共享输出负担对层级问法仍有代价。

## 3. 分任务变化

| focus F1 | 4RGB v4→v5 | 2RGB v4→v5 | 判断 |
|---|---:|---:|---|
| HIGHWAY | 98.438→98.438 | 99.213→99.213 | 稳定 |
| STATIC_OBSTACLE | 89.256→91.200 | 89.831→87.931 | mode 间方向不一致 |
| VULNERABLE | 90.909→87.179 | 88.136→87.719 | 4RGB 表面回退，需看 RGB |
| TRAFFIC_LIGHT_ABNORMAL | 84.685→95.082 | 92.683→84.685 | mode 间方向相反 |
| RS1 | 78.049→75.758 | 73.585→77.311 | mode 间方向相反 |
| RS2 | 93.846→91.339 | 88.406→88.550 | 4RGB 小幅回退 |
| RS4 | 88.710→90.476 | 92.063→93.023 | raw F1 提升，但含标签冲突 |
| RS5 | 83.478→80.702 | 88.333→90.756 | mode 间方向相反 |

4RGB Phase1 focus macro 从 90.822% 升至 92.975%，Phase2 focus macro 从 86.021% 降至
84.569%；2RGB Phase1 focus macro 从 92.465% 降至 89.887%，Phase2 从 85.597% 升至
87.410%。这类“一个 mode 升、另一个 mode 降”的结果不支持继续对所有提示词做统一增删。

## 4. 逐帧 RGB 错误审计

以下结论均查看 `history_rgb_paths_all4` 的四张原始 stitched RGB；即使模型输入是
2RGB endpoints，也查看了被省略的中间两帧，避免凭首尾帧猜测。

### 4.1 2RGB RS4：全部 5 个当前 focus 假阳性

| case | route / frames | GT→pred | 四帧视觉结论 | 归因 |
|---:|---|---|---|---|
| 81 | `NonSignalizedJunctionLeftTurn/Town13`, 82–85 | NO→YES | 多个真实、完整的横臂交通信号灯头清楚可见，虽灯面偏暗 | **标签/定义冲突**；按当前 RS4 定义视觉上应为 YES |
| 415 | `DynamicObjectCrossing/Town07`, 31–34 | NO→YES | 雾天乡间双向路，只有车辆前灯，无信号灯硬件 | **真实假阳性** |
| 462 | `VehicleTurningRoute/Town05`, 30–33 | NO→YES | 交叉口左侧横臂和立杆上有清晰绿色交通灯头 | **标签/定义冲突**；视觉上应为 YES |
| 723 | `ControlLoss/Town10HD`, 138–141 | NO→YES | 雾天城市道路、车辆尾灯和向下照明装饰路灯，无信号灯头 | **真实假阳性** |
| 904 | `DynamicObjectCrossing/Town05`, 27–30 | NO→YES | 雾天道路、车辆灯和普通高杆照明，无局部信号交叉口 | **真实假阳性** |

真实 RS4 FP 为 3/5，标签/定义冲突为 2/5。v5 新句已经明确“装饰路灯、裸杆、车辆灯不是
信号灯硬件”，但 case 415/723/904 仍误判，说明继续重复同一句边界的边际价值很低。

### 4.2 2RGB RS4：v5 相对 v4 修正的两个代表漏检

| case | route / frames | GT / v4→v5 | 四帧视觉结论 | 归因 |
|---:|---|---|---|---|
| 77 | `VehicleTurningRoute/Town05`, 79–82 | YES / NO→YES | 宽城市道路、转向车辆和行人，但没有可辨认交通信号灯头或受信号控制的局部交叉口 | **标签/可见性冲突，非视觉修正** |
| 898 | `DynamicObjectCrossing/Town13`, 193–196 | YES / NO→YES | 大雨浓雾中的乡间支路、车辆前灯和普通路灯杆，无信号灯头 | **标签/可见性冲突，非视觉修正** |

因此 RS4 raw F1 上升不能全部归功于硬件辨识能力增强；至少这两个“修正”更像标签拟合。

### 4.3 2RGB TRAFFIC_LIGHT_ABNORMAL：10 个 v4 正确→v5 错误

逐帧合并重复窗口后共有 7 条 route：

- case 72：夜间仍能看到不同方向的红/绿灯，属于弱光下的真实漏检。
- case 104/450/521/619：同一 Town05 route 的 17–20 帧被四个 case 重复计分；宽交叉口
  前方红灯、侧向绿灯清晰，属于真实漏检，但测试指标对该 route 有重复加权。
- case 296：农田岔路只有车辆和普通灯杆，没有交通信号灯硬件；GT=YES 与 RGB 冲突。
- case 330：雾中左向红灯与前向绿灯清楚可见，属于真实漏检。
- case 532：多个交叉方向红/绿灯清楚可见，属于真实漏检。
- case 658：夜雨中多组红/绿灯跨帧可读，属于真实漏检。
- case 832：已驶离/看不到信号灯的雾天道路，GT=YES 缺少当前视觉证据。

按 case 计，8/10 是真实漏检、2/10 是标签/当前可见性冲突；但 4/8 真实漏检来自同一路段
连续窗口。现有提示词已经要求“同一冲突区比较不同方向灯头”和“一帧清晰 witness 即可”，
缺的不是定义文字，而是 checkpoint 在弱光/雾雨下的稳定性。

### 4.4 4RGB VULNERABLE：6 个 v4 正确→v5 错误

- case 176 与 867 是同一 Town10HD 夜雨窗口 31–35：逐帧无可辨认行人/骑行者，GT=YES
  缺乏视觉证据。
- case 344：四帧几乎全黑，无法形成可辨认 vulnerable witness，属于可见性冲突。
- case 635：多车道道路仅见机动车，无行人/骑行者，属于标签冲突。
- case 636：乡间双向路仅见机动车，无 vulnerable road user，属于标签冲突。
- case 997：夜雨中骑行者在 58–60 帧清楚位于冲突区、61 帧已离开；这是唯一明确的真实
  漏检，且现有 prompt 的“one clear older-frame witness”规则本应覆盖。

因此 6 个表面回退中 5 个不应驱动提示词扩写，1 个是真实时序漏检。保留既有旧帧 witness
规则，比新增更宽泛的“场景推断 vulnerable”更安全。

## 5. Prompt 决策

### 5.1 保留 v5，不创建 v6

保留 v5 中这一条最小修订：RS4 需要可识别交通信号灯头；装饰/向下照明路灯、裸杆/灯臂、
车辆灯不算信号硬件。理由：该句与逐帧 RGB 和 RS4 定义一致，且不会把真实信号头排除。

本轮不继续增加以下内容：

- 不用“雾天一律 RS4=NO”，因为雾中仍可能有清楚灯头；
- 不为匹配 case 77/898 的 GT 放宽到“路口/车辆灯即可 RS4=YES”；
- 不进一步放宽 VULNERABLE，因为多数表面 FN 根本没有可见目标；
- 不重写 TRAFFIC_LIGHT_ABNORMAL，其现有规则已经覆盖此次真实漏检，问题主要在训练/选优稳定性。

这也意味着 prompt 名与 production hash 保持不变，现有 v5 adapter 不需要因为本次代码更新
而重训。

### 5.2 何时才允许下一次改 prompt

下一版 prompt 必须先满足：同一 frozen case 集上出现跨 seed 或跨 RGB mode 重复的同类真实
视觉错误；逐帧能指出现有规则缺失的具体边界；新规则不能只修正与 RGB 冲突的 GT。单个
checkpoint 的 raw F1 波动不够作为改词依据。

## 6. 代码改进：正式流程自动补测 balanced checkpoint

最新 2RGB run 的 validation 选择出现重要分叉：

- `best_generation/step-40000`：exact 76.95%，minimum focus 78.12%；本次 bundle 只测了它。
- `best_generation_balanced/step-28000`：exact 76.17%，minimum focus **84.38%**；未进入正式 test bundle。

2RGB test 中 `TRAFFIC_LIGHT_ABNORMAL` 正是最明显退化桶，因此漏测 balanced 候选会让“是否
应换 checkpoint”无法闭环。`run_full_pipeline.sh` 已改为默认
`RUN_BALANCED_EVAL=1`：primary 为 `best_generation/` 且同 run 的 balanced 权重不同时，
自动生成独立 `eval_review_balanced/` 和 `*_balanced_audit_bundle.tar.gz`；若权重逐字节相同则
跳过。它不改变训练、prompt 或 primary 权重，仅补齐正式候选验证。

## 7. 下一步验收建议

以下建议已由同日 2RGB 补测闭环，正式结果见第 8 节。补测前的验收原则仍保留为记录：必须在
同一个 frozen 1024-case 身份上比较 step-40000 exact 与 step-28000 balanced，并同时查看联合
exact、八个 focus、minimum focus、Phase1/Phase2、三类 variant 和逐帧 RGB，而不能只看
validation 的单一选优分数。

## 8. 2RGB primary / balanced 补测审计

### 8.1 可比性与候选身份

补测输入：

- `sft_new_loop_phase1_20260831_144428_2rgb_endpoints_audit_bundle`
- `sft_new_loop_phase1_20260831_144428_2rgb_endpoints_balanced_audit_bundle`

两包均使用 v5 production prompt、相同 production prompt hash、`2rgb_endpoints` 的 `[0,3]`
帧位、同一个 frozen 1024-case test 集。逐条核对后，1024 个 case id 以及
`scenario/route/frame/focus/variant/四帧路径` 身份全部一致，可以做严格配对比较。区别只有权重：

- primary：`best_generation/step-40000`
- balanced：`best_generation_balanced/step-28000`

两包 base production/audit 的逐例结果一致，因此下面的 LoRA 差异不是测试集或 base 漂移造成。

### 8.2 总体与阶段结果

| 指标 | primary step-40000 | balanced step-28000 | balanced - primary |
|---|---:|---:|---:|
| 联合严格 exact | **784/1024 = 76.563%** | 774/1024 = 75.586% | **-0.977 pp** |
| Phase1 requested-lines exact | **920/1024 = 89.844%** | 908/1024 = 88.672% | **-1.172 pp** |
| Phase2 requested-lines exact | **857/1024 = 83.691%** | 846/1024 = 82.617% | **-1.074 pp** |
| focus accuracy | **916/1024 = 89.453%** | 914/1024 = 89.258% | -0.195 pp |
| 八 focus macro F1 | 88.649% | **88.853%** | +0.205 pp |
| 最弱 focus accuracy | 78.906% | **81.250%** | **+2.344 pp** |

严格配对的联合 exact 切换为：primary 错、balanced 对 67 例；primary 对、balanced 错 77 例，
净减少 10 例。balanced 确实完成了“抬高最弱桶”的目标，但代价是联合 exact、Phase1 exact
和 Phase2 exact 同时下降；macro F1 的小幅上升不足以抵消完整问答严格正确率的下降。

### 8.3 八个 focus 的迁移

每个 focus 恰有 128 个 YES:NO 平衡样本：

| focus | primary accuracy / F1 | balanced accuracy / F1 | 主要变化 |
|---|---:|---:|---|
| HIGHWAY | 99.219% / 99.213% | 99.219% / 99.213% | 不变 |
| STATIC_OBSTACLE | **89.063% / 87.931%** | 87.500% / 87.692% | recall 上升，但 FP 1→9，accuracy 下降 |
| VULNERABLE | **89.063% / 87.719%** | 85.938% / 83.636% | TP 50→46，继续偏保守 |
| TRAFFIC_LIGHT_ABNORMAL | 86.719% / 84.685% | **92.188% / 91.525%** | TP 47→54 且仍为 0 FP，明确改善 |
| RS1 | 78.906% / 77.311% | **81.250% / 80.952%** | 最弱桶改善，TP 46→51、FP 9→11 |
| RS2 | **88.281% / 88.550%** | 86.719% / 87.218% | FP 9→11，回退 |
| RS4 | **92.969% / 93.023%** | 92.188% / 92.063% | raw 小幅回退，RGB 归因见下 |
| RS5 | **91.406% / 90.756%** | 89.063% / 88.525% | FP 1→4，回退 |

balanced 的优势高度集中在 `TRAFFIC_LIGHT_ABNORMAL` 和 `RS1`；与此同时
`STATIC_OBSTACLE`、`VULNERABLE`、`RS2`、`RS5` 均下降。因此它不是整体视觉问答能力提升，
而是 checkpoint 早期状态在不同任务间重新分配了偏置。

### 8.4 variant 与 audit 输出合同

| variant | primary exact | balanced exact | 配对净变化 |
|---|---:|---:|---:|
| `all_random_order`（512） | **72.461%** | 69.922% | 37 修复 / 50 回退，净 -13 |
| `subset_random`（256） | 78.516% | **81.250%** | 21 修复 / 14 回退，净 +7 |
| `hierarchical_probe`（256） | **82.813%** | 81.250% | 9 修复 / 13 回退，净 -4 |

production 格式合法率两者都是 100%。audit prompt 下，balanced 的语义 exact 从 76.660%
降至 74.609%，但输出合同明显更稳定：answer-valid `1021→1024`、evidence-complete
`1011→1017`、contract-valid `1001→1017`，unexpected answer `14→1`、missing evidence
`49→10`、duplicate answer `15→0`。这说明 balanced 更会遵守 audit 格式，但不能把格式改善
等同于语义问答提升。

### 8.5 关键切换 case 的逐帧 RGB 归因

以下只把人工实际查看过的四帧写入结论；未查看的切换 case 不做视觉推断。

#### TRAFFIC_LIGHT_ABNORMAL

- balanced 的 8 个 focus 修复中，case 72、104、330、450、521、532、658 都能在至少一帧
  读到同一宽交叉口不同方向的红/绿灯，其中 case 104/450/521 是同一路段连续窗口，属于真实
  能力改善但存在重复计权；case 832 四帧没有可读信号，属于 GT/当前可见性冲突。
- 唯一 raw 回退 case 158 是极暗雨雾道路，四帧都没有可辨认信号灯头或矛盾灯态；balanced
  输出 NO 反而符合 RGB 证据。

所以该桶的提升不仅是 raw 指标改善，也有明确视觉支持；按独立 route 计仍需扣除重复窗口影响。

#### RS4

- case 415/723：primary 把车辆灯、雾中照明或装饰灯杆当成信号硬件，balanced 改为 NO，是真实修复。
- case 77/898：raw GT 为 YES，但四帧没有可辨认信号灯头；balanced 的 NO 是视觉上合理的
  “表面回退”。
- case 326：清楚的分隔多车道桥面/高速走廊，没有局部交叉口或交通信号硬件；balanced 改成
  YES 是真实新增假阳性。

因此 RS4 raw F1 小降不能直接解释为硬件辨识退化；balanced 更少把弱灯光当硬件，但仍产生了
新的道路结构幻觉。现有 v5 边界无需改写。

#### STATIC_OBSTACLE（抽样切换）

- raw 修复 case 342/708 都处于极暗场景，看不到固定物占用 ego corridor；case 557 的车辆跨帧
  横向移动并离开，不是静态障碍。这 3 个 YES 更像标签拟合，不能作为放宽 prompt 的依据。
- raw 回退 case 404/489/516 都只有道路相对运动中的车辆，没有可辨认固定占道物；case 575
  的红色 SUV 从停车区动态驶出。balanced 的 YES 均为真实假阳性。

这与全 requested-lines 指标一致：balanced 虽把 STATIC TP `110→122`，同时把 FP
`11→40`，F1 `86.614%→82.712%`。它明显放松了 STATIC 判定，视觉上不如 primary 稳健。

#### VULNERABLE（全部 8 个 focus 切换）

- 两个修复 case 796/613 均有清楚视觉证据：前者雾中路口有多名行人，后者右侧人行道边有
  骑行者，balanced 的 YES 是真实修复。
- 六个 raw 回退中，case 36/739 四帧近乎全黑且没有可辨认 vulnerable witness，balanced 的
  NO 与 RGB 一致；case 391 右侧路口边缘有行人，case 657 左侧路口有骑行者，case 755 右侧
  近处有清楚骑行者，属于真实漏检；case 606 只在最右边缘出现被裁切的人体，证据较弱但按
  当前“边缘/旧帧 witness”定义仍应警惕。

所以 raw 的 2 修复 / 6 回退包含标签可见性冲突，但至少 3 个、连同边缘 case 606 至多 4 个
回退是实际小目标漏检。balanced 在 VULNERABLE 上确实更保守，不能用 LIGHT/RS1 的收益掩盖。

### 8.6 最终 checkpoint 与 prompt 决策

1. **production 继续使用 `best_generation/step-40000`。** 它在同一 1024-case 集上联合 exact
   高 0.977 pp，Phase1/Phase2 strict exact 都高约 1.1 pp，且 STATIC、VULNERABLE、RS2、RS5
   更稳。`best_generation_balanced/step-28000` 不晋升为全局 production。
2. balanced 可保留为诊断候选：它证明较早 checkpoint 对弱光灯异常和 RS1 有更好状态，且
   audit 合同更干净；但这是任务间 trade-off，不是整体能力提升。
3. **不修改 v5 prompt，也不因此重训。** LIGHT 的规则本来已经覆盖真实修复，STATIC/VULNERABLE
   的问题是 checkpoint 偏置、小目标/弱光视觉稳定性和部分标签冲突；继续堆同义提示词会扩大误报。
4. 下一轮若修改 checkpoint 选优，应给 minimum focus 增益增加联合/阶段守门条件，例如要求
   联合 exact 的下降显著小于本次 0.977 pp，并限制 Phase1、Phase2 strict exact 以及任一关键
   focus 的回退；不能只最大化 minimum focus 后直接晋升。
