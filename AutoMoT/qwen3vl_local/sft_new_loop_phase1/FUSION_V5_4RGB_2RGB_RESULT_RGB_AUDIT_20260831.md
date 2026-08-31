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

1. 若远端仍保留本次 2RGB run 的 `best_generation_balanced/step-28000` 权重，优先直接补跑
   balanced test，不需要重训；本地只有 audit bundle 元数据，没有该 adapter 权重，无法代跑。
2. 使用同一个 frozen 1024-case test 身份比较 step-40000 exact 与 step-28000 balanced，至少
   报告联合 exact、八个 focus F1、minimum focus、Phase1/Phase2 macro、三类 variant exact。
3. 选择 production checkpoint 时，不只看联合 exact：若 balanced 能显著恢复 2RGB 灯异常
   且联合 exact 下降小于约 1pp，应优先考虑 balanced；最终仍以正式 test 和逐帧错例为准。
4. 当前不建议立即开启新一轮训练。先补齐 balanced test，确认问题来自 checkpoint 选择还是
   整轮训练分布，再决定是否需要新 seed/重训。
