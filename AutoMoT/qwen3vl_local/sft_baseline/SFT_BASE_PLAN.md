# SFT Baseline Plan

`sft_baseline` 是从 `sft_base` 复制后降维出来的直接监督基线，用来测试更粗的视觉问题是否更容易学到：

- `ROAD`: `HIGHWAY` / `NON_HIGHWAY`
- `EVENT`: `RE` / `UE`

## 1. Label Folding

- `HIGHWAY`: 原 RS 为 `R3`，即 highway / ramp / merge / split / exit / connector / lane-join 结构。
- `NON_HIGHWAY`: 原 RS 为 `R1/R2/R4/R5`。
- `UE`: 原 EVENT target `abnormal=True`，即 `U-E*`。
- `RE`: 原 EVENT target `abnormal=False`，即 regular `R-E*`。

数据构建仍复用 `collection_output` 的 RS/EVENT 解析、异常 route 剔除、4 帧 RGB history、`EGO_TO_GOAL_XY` 与 route-level sequence index；但训练监督只用折叠后的两行答案。

## 2. Prompt Contract

单帧只问一次，不再有 Q1/Q2 串行：

```text
ROAD: HIGHWAY|NON_HIGHWAY
EVENT: RE|UE
```

Prompt 仍包含轻量 memory：

```text
PREVIOUS_ROAD: UNKNOWN|HIGHWAY|NON_HIGHWAY
PREVIOUS_EVENT: UNKNOWN|RE|UE
EGO_TO_GOAL_XY: (...)
```

memory 是弱先验，不是答案；system/user prompt 明确要求视觉证据优先。

## 3. Training

训练仍是 teacher-forced weighted CE，不做 OPSD、不跑 privileged teacher、不写 CoT。loss 只落在 `ROAD` 与 `EVENT` 两个值 token 上。

默认只训练语言侧 LoRA：`--lora-vision-scope=off`。视觉塔 LoRA 只作为显式消融开关，
需要时手动设为 `merger` / `last4` / `all`。

保留原 `sft_base` 的 wrong/UNKNOWN/dropout memory curriculum，但已按二分类语义修正：

- ROAD wrong 必须跨 `HIGHWAY` / `NON_HIGHWAY` 边界；
- EVENT wrong 必须跨 `RE` / `UE` 边界；
- 首帧默认 `UNKNOWN/UNKNOWN`；
- route_state 扰动仍按连续 3-5 帧块状保持，模拟错误 memory 持续后再恢复。

UE/RE loss reweight 继续生效：`EVENT` 值 token 在 UE 帧按 `--ue-event-loss-weight` 加权，在 RE 帧按 `--re-event-loss-weight` 加权。

## 4. Eval

`eval.py` 每帧 fresh prefill 生成一次两行答案，并用学生输出维护下一帧 memory，不做脚本纠偏。

主指标：

- `road_acc`
- `highway_precision/recall/f1`
- `event_acc`
- `ue_precision/recall/f1`
- `joint_acc`
- `road_change_f1`
- `event_change_f1`

逐帧 `frames.jsonl` 保存 GT/PRED ROAD、GT/PRED EVENT、原始生成文本和可选 prompt。

## 5. Compatibility

当前协议：

- `DATASET_VERSION=sft_baseline_highway_reue_joint_v1`
- adapter route: `sft_baseline_highway_reue_joint`
- adapter config: `sft_baseline_adapter_config.json`

旧 `sft_base` 或旧两问 adapter 必须重建/重训，eval 会拒绝 route 或 dataset version 不匹配的 adapter。
