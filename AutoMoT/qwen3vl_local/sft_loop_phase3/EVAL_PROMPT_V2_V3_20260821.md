# Phase3 Prompt v2/v3 Eval Analysis, 2026-08-21

本文汇总 `sft_loop_phase3` 在 2026-08-21 两个 audit bundle 上的指标对比与问题归因。

## 1. 对比对象

两个 bundle 都来自 `sft_loop_phase3` 工具链，不是 `qwen3vl_local/sft_v3` 代码路线。这里的 `v2/v3` 指的是 `sft_loop_phase3` 的 prompt 版本。

| bundle | prompt | adapter | 备注 |
|---|---|---|---|
| `AutoMoT/checkpoints/sft_loop_ohase3_eval/20260821_100309_audit_bundle` | `sft_loop_phase3_event_gate_visual_v3` | `checkpoints/sft_loop_phase3_runs/run_event_gate_format_supervised_4rgb/20260820_201322/best_generation` | 新 prompt v3 |
| `AutoMoT/checkpoints/sft_loop_ohase3_eval/20260821_105049_audit_bundle` | `sft_loop_phase3_event_gate_visual_v2` | `checkpoints/sft_loop_phase3_runs/run_event_gate_format_supervised_4rgb/20260820_104131/best_generation` | 回退 prompt v2 |

共同设置：

- eval cases: 384
- history RGB: `4rgb`, indices `[0, 1, 2, 3]`
- LoRA vision scope: `off`
- eval variant: `all_random_order`
- format valid rate: 1.0000
- 每个 GT pattern 在 eval 中均衡为 64 个：`ALL_NO`、`INVALID_RS_CONTEXT`、`UE1`、`UE3`、`UE5`、`UE6`

## 2. 总体结论

回退 v2 是合理的。v2 在 production eval 上明显优于 v3：

| 指标 | v3 / `100309` | v2 / `105049` |
|---|---:|---:|
| Base Qwen exact | 0.1667, 64/384 | 0.1667, 64/384 |
| LoRA production exact | 0.6354, 244/384 | 0.7786, 299/384 |
| LoRA production errors | 140 | 85 |
| LoRA audit-prompt exact | 0.6042, 232/384 | 0.7578, 291/384 |
| invalid joint ok rate | 0.7031 | 0.7656 |
| invalid UE all-NO rate | 0.9375 | 0.9844 |

Base Qwen 两边完全一致，且都只会给出全 NO 式保守答案；差异主要来自对应 prompt 版本下训练出的 LoRA adapter。

## 3. Per-question 指标

Production LoRA 的分项指标如下。注意 UE1/UE3/UE5 只在 RS1/RS2 gate 下出现，UE6 只在 RS4/RS5 gate 下出现，因此各问题的 `total` 不相同。

| question | v3 acc | v3 P/R/F1 | v2 acc | v2 P/R/F1 |
|---|---:|---|---:|---|
| UE1 | 0.8858 | 0.857 / 0.656 / 0.743 | 0.9213 | 0.907 / 0.766 / 0.831 |
| UE3 | 0.8307 | 0.889 / 0.375 / 0.527 | 0.9252 | 0.895 / 0.797 / 0.843 |
| UE5 | 0.9331 | 0.898 / 0.828 / 0.862 | 0.9646 | 1.000 / 0.859 / 0.924 |
| UE6 | 0.7769 | 0.830 / 0.688 / 0.752 | 0.8077 | 0.898 / 0.688 / 0.779 |
| INVALID_RS_CONTEXT | 0.9089 | 0.738 / 0.703 / 0.720 | 0.9505 | 0.925 / 0.766 / 0.838 |

最主要的退化是 UE3：v3 的 recall 只有 0.375，v2 是 0.797。v3 对动态 cut-in、动态占道、横穿进入 ego corridor 的判断明显过保守。

## 4. Pattern 诊断

| pattern 统计 | v3 | v2 | 说明 |
|---|---:|---:|---|
| GT `ALL_NO` | 64 | 64 | eval 均衡固定 |
| Pred `ALL_NO` | 135 | 116 | 两者都偏保守，v3 更严重 |
| GT `UE3` | 64 | 64 | eval 均衡固定 |
| Pred `UE3` | 27 | 57 | v3 严重少报 UE3 |
| GT `INVALID_RS_CONTEXT` | 64 | 64 | eval 均衡固定 |
| Pred `INVALID_RS_CONTEXT` | 61 | 53 | v3 报 invalid 数量更多，但质量更差 |
| Pred multi-YES | 0 | 0 | 两者都没有多 UE 同时 YES 的格式/策略问题 |

关键 pattern pair：

| pattern pair | v3 | v2 | 解读 |
|---|---:|---:|---|
| `UE3=>UE3` | 24 | 51 | v2 对 UE3 的命中明显更好 |
| `UE3=>ALL_NO` | 37 | 10 | v3 最大错误源 |
| `ALL_NO=>ALL_NO` | 36 | 51 | v2 对 regular/all-NO 更稳 |
| `ALL_NO=>INVALID_RS_CONTEXT` | 14 | 4 | v3 更容易把 regular 误判成 wrong RS context |
| `INVALID_RS_CONTEXT=>INVALID_RS_CONTEXT` | 45 | 49 | v2 invalid recall 更好 |
| `INVALID_RS_CONTEXT=>ALL_NO` | 15 | 14 | 两者仍都有 invalid FN |
| `UE6=>UE6` | 44 | 44 | UE6 recall 两者相同 |
| `UE6=>ALL_NO` | 18 | 20 | UE6 仍是 v2 主要剩余问题之一 |

## 5. 错误来源

Production LoRA 的错误按 GT bin 统计：

| GT bin | v3 errors | v2 errors |
|---|---:|---:|
| UE1 | 22 | 15 |
| UE3 | 40 | 13 |
| UE5 | 11 | 9 |
| UE6 | 20 | 20 |
| RE / ALL_NO | 28 | 13 |
| INVALID | 19 | 15 |
| total | 140 | 85 |

场景层面，v3 的高频错误集中在：

- `HardBreakRoute`: 24
- `OppositeVehicleRunningRedLight`: 23
- `DynamicObjectCrossing`: 20
- `StaticCutIn`: 15
- `InvadingTurn`: 12
- `ParkingCutIn`: 12

v2 仍有错误，但分布更可控：

- `HardBreakRoute`: 17
- `OppositeVehicleRunningRedLight`: 16
- `DynamicObjectCrossing`: 15
- `InvadingTurn`: 13
- `CrossJunctionDefectTrafficLight`: 5
- `BlockedIntersection`: 4

这说明 v2 的主要剩余问题是 UE6、UE1 和 invalid 的少量 FN/FP；v3 则把 UE3 和 RE/all-NO 一起拉坏了。

## 6. Prompt 差异归因

从 bundle 中保存的实际 prompt diff 看，v3 相比 v2 对 UE3、UE5、UE1 都做了更谨慎的边界收窄。

UE3 关键变化：

- v2: `about to occupy ego's immediate future corridor`
- v3: `already occupying ego's immediate future corridor`
- v2: `dynamic crossing into the path`
- v3: `a cross-traffic vehicle that has reached the ego path`
- v3 额外排除 `a bus/car close to the side but still outside the ego lane`

这些改动本意是减少 FP，但实际让模型更倾向 `ALL_NO`，尤其把 `DynamicObjectCrossing`、`StaticCutIn`、`ParkingCutIn` 中刚进入或即将进入 ego corridor 的早期/中期帧漏掉。

UE5 关键变化：

- v3 强化了 narrow/rural two-way road 的 intrusion 条件，同时排除了 distant headlights。
- 该项没有崩，但收益不明显，v3 UE5 仍低于 v2。

UE1 关键变化：

- v3 更强调 `brake lights plus rapid closing distance`，并显式排除 `brake lights alone`。
- 这降低了部分误报风险，但 v3 UE1 recall 也比 v2 低。

Invalid 关键变化：

- v3 把 wrong-RS context 限定为 `newest RGB road layout` 的明显不兼容，并添加 fog/night 不应覆盖明显 topology mismatch。
- 结果 v3 invalid FP 从 4 增到 16，invalid FN 从 15 增到 19，说明该描述没有带来稳定收益，反而可能让模型把复杂场景的 topology 不确定性当成 wrong gate。

## 7. 训练与评测口径问题

训练时自动 `best_generation` 使用的 generation eval 只有 12 个样本，选择噪声很大：

| bundle | best generation / best val 小样本 | 384-case production |
|---|---:|---:|
| v3 | best val generation exact 0.8333 | 0.6354 |
| v2 | best val generation exact 0.9167 | 0.7786 |

因此后续不要仅靠 12-case generation eval 选 checkpoint。至少应该增加 generation eval 的 balance count，或者固定一组更大的 audit validation set。

此外，两个 bundle 的 adapter metadata 存在 step 记录不完全一致的问题：

- v3 `sft_loop_phase3_adapter_config.json` 中 `global_step=2000`，但 `best_generation.json` 记录 `step=4000`。
- v2 `sft_loop_phase3_adapter_config.json` 中 `global_step=8000`，但 `best_generation.json` 记录 `step=2000`。

当前结论以 bundle 的 production `metrics.json` 为准；后续建议修正 adapter 保存时的 `global_step` / `best_generation` 元数据同步，避免误判 checkpoint 来源。

## 8. 建议

1. 保持回退到 `sft_loop_phase3_event_gate_visual_v2`。
2. 不建议继续在 v3 prompt 上追加训练；v3 的主要问题是语义边界收窄过度，而不是训练不够。
3. 后续 prompt 迭代应从 v2 出发，只小步修：
   - UE6 FN: `OppositeVehicleRunningRedLight`、`CrossJunctionDefectTrafficLight` 中仍有 20 个漏报。
   - UE1 FN: `HardBreakRoute` 仍有 15 个漏报。
   - invalid FN/FP: v2 仍有 15 个 FN 和 4 个 FP。
4. 保留 v2 对 UE3 的 `about to occupy` / `dynamic crossing into the path` 口径，不要再改成必须已经到达 ego path。
5. 增大 generation eval 样本量，避免 12-case best selection 误导。
6. 修复 adapter metadata 的 step 同步问题，确保 `adapter_config`、`best_generation.json`、实际 adapter 目录一致。

