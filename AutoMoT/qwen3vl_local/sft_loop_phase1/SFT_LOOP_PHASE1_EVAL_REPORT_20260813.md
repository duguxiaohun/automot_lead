# Phase1 最新 4RGB / 2RGB 指标简报

## 实验口径

本报告读取当前 `sft_loop_phase1_eval/` 的 8 组结果：`4rgb` 和
`2rgb_endpoints` 各有 base production、base audit、LoRA production、LoRA audit。

- `4rgb` 输入原始 history 的 `[0,1,2,3]` 四帧。
- `2rgb_endpoints` 输入原始 history 的首尾两帧 `[0,3]`。
- 每个主问题都是 128 个样本，且 `YES=64 / NO=64`；四个 production 结果的 512 个
  case 完全相同，因此可以直接比较。
- 两个 LoRA eval 都从 adapter config 自动读取 RGB 模式，并且
  `adapter_prompt_matches_current_production=True`，不存在 LoRA 与 prompt/input
  不匹配的问题。
- 正式排名只看 production。audit 增加 `EVIDENCE_*`，属于不同输入 prompt，只用于诊断。

## Production：Base 与 LoRA 对比

| 模式 / 模型 | Macro F1 | 四答案完全正确率 | HIGHWAY F1 | STATIC_OBSTACLE F1 | VULNERABLE F1 | TRAFFIC_LIGHT_ABNORMAL F1 |
|---|---:|---:|---:|---:|---:|---:|
| 4rgb base | 0.3287 | 0.5566 | 0.0606 | 0.4286 | 0.8257 | 0.0000 |
| 4rgb LoRA | 0.8603 | 0.8477 | 0.9593 | 0.7379 | 0.8870 | 0.8571 |
| 2rgb_endpoints base | 0.3764 | 0.5605 | 0.2222 | 0.4471 | 0.8364 | 0.0000 |
| 2rgb_endpoints LoRA | **0.8662** | **0.8516** | 0.9593 | **0.7619** | **0.8966** | 0.8468 |

### LoRA 是否有效

有效，而且两种 RGB 输入下都很明显。

| 模式 | Macro F1：base -> LoRA | 完全正确率：base -> LoRA | 结论 |
|---|---:|---:|---|
| 4rgb | 0.3287 -> 0.8603（+0.5316） | 0.5566 -> 0.8477（+0.2911） | 高速、灯异常从接近全 NO 变成可用。 |
| 2rgb_endpoints | 0.3764 -> 0.8662（+0.4898） | 0.5605 -> 0.8516（+0.2911） | 同样大幅提升，静态障碍和弱势参与者略强。 |

production LoRA 的主要剩余问题仍是正例漏报，而非误报：

| 模式 | HIGHWAY TP/FP/FN/TN | STATIC TP/FP/FN/TN | VULNERABLE TP/FP/FN/TN | LIGHT TP/FP/FN/TN |
|---|---|---|---|---|
| 4rgb LoRA | 59 / 0 / 5 / 64 | 38 / 1 / 26 / 63 | 51 / 0 / 13 / 64 | 48 / 0 / 16 / 64 |
| 2rgb_endpoints LoRA | 59 / 0 / 5 / 64 | 40 / 1 / 24 / 63 | 52 / 0 / 12 / 64 | 47 / 0 / 17 / 64 |

## Production：4rgb 与 2rgb_endpoints 对比

`2rgb_endpoints LoRA` 的 macro F1 高 `0.0059`，完全正确率高 `0.0039`，即相同 512 个
case 中多 2 个四答案全对。这是轻微优势，尚不足以宣称两帧已经确定优于四帧。

- HIGHWAY：完全相同，F1 都是 `0.9593`。
- STATIC_OBSTACLE：两帧更好，`0.7619` vs `0.7379`；多 2 个 TP，FP 相同。
- VULNERABLE：两帧更好，`0.8966` vs `0.8870`；多 1 个 TP，FP 都为 0。
- TRAFFIC_LIGHT_ABNORMAL：四帧略好，`0.8571` vs `0.8468`；多 1 个 TP，FP 都为 0。

因此当前可把 `2rgb_endpoints LoRA` 作为 macro F1 / exact-match 的暂时最佳结果；但两个模式
差距很小，后续应继续看错例 RGB 和更多固定 split，而不是仅凭这一轮选择最终输入帧数。

## Audit：开启 EVIDENCE 后的变化

audit 不能和 production 混作部署指标，但它能说明“要求模型显式写视觉证据”对答案有何影响。
本表最后四列是 **`Audit F1 - Production F1` 的变化量**，不是 audit 的绝对 F1。

| 模式 / 模型 | Production Macro F1 -> Audit Macro F1 | Production Exact -> Audit Exact | HIGHWAY F1 变化 | STATIC F1 变化 | VULNERABLE F1 变化 | LIGHT F1 变化 |
|---|---:|---:|---:|---:|---:|---:|
| 4rgb base | 0.3287 -> 0.5615 | 0.5566 -> 0.5059 | +0.8061 | +0.1698 | -0.0447 | 0.0000 |
| 2rgb_endpoints base | 0.3764 -> 0.5769 | 0.5605 -> 0.5039 | +0.6083 | +0.1810 | -0.0182 | +0.0308 |
| 4rgb LoRA | 0.8603 -> 0.7947 | 0.8477 -> 0.7285 | +0.0167 | +0.0590 | -0.0098 | -0.3284 |
| 2rgb_endpoints LoRA | 0.8662 -> 0.8059 | 0.8516 -> 0.7559 | +0.0167 | +0.0714 | -0.0096 | -0.3193 |

### 为什么有任务升、有任务降，但 Macro F1 和 Exact 都下降？

以 `4rgb LoRA` 为例：audit 让 HIGHWAY 增加 `+0.0167`、STATIC 增加 `+0.0590`，但
VULNERABLE 减少 `-0.0098`，LIGHT 大幅减少 `-0.3284`。四项变化平均为：

```text
(+0.0167 +0.0590 -0.0098 -0.3284) / 4 = -0.0656
```

所以 Macro F1 自然从 `0.8603` 降到 `0.7947`。两帧模式也是同一个原因：STATIC 的增益
抵不过 LIGHT 的大幅下降。

`Exact` 更严格：512 个 case 中，只有四个答案都正确才算该 case 正确。即使 audit 在静态障碍
模块找回了一些正例，只要同一个 case 的灯异常、弱势参与者或高速其中任一项被 audit 改错，这个 case
就不再计入 Exact。`4rgb LoRA` 的 Exact 从 `434/512` 降到 `373/512`；两帧 LoRA 从
`436/512` 降到 `387/512`。此外，四个主任务 F1 分别在各自独立的 128 个 1:1 主任务集合上计算，
不能把某一项 F1 上升直接理解为所有 512 个 case 的四答案组合都会变好。

简要判断：

- 对 base，audit 明显改变输出倾向，尤其把 HIGHWAY 从保守全 NO 拉高；但 exact 反而下降，不能当作
  base 本身变强。
- 对 LoRA，audit 会提高静态障碍 recall，但同时增加 FP：4rgb 为 `38/1/26/63 -> 51/13/13/51`，
  两帧为 `40/1/24/63 -> 50/6/14/58`。
- audit 明显伤害灯异常：4rgb `0.8571 -> 0.5287`，两帧 `0.8468 -> 0.5275`。因此灯异常的正式
  评估、训练选择和部署都应使用 production prompt；audit 只用于导出 RGB 错例和模型证据。

## 当前结论

1. 两个 LoRA 都有效，且 production prompt 下性能稳定；不要用 base 或 audit 的 F1 替代 LoRA
   production 结论。
2. 当前最好指标是 `2rgb_endpoints LoRA production`，但只比四帧高 `0.0059` macro F1，结论应是
   “首尾两帧至少没有明显损失，并有轻微优势”，不是“已证明两帧全面更好”。
3. 若继续优化，优先围绕 static 与 light 的 production FN 逐帧审计；不要因为 audit 的 static F1
   更高就直接切换 audit prompt。
