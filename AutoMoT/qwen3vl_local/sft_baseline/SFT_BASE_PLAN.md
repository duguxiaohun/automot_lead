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

ckpt-200 诊断后，训练采样从“整条 route 均匀吃完”改为可配置的短片段均衡路线：

- launcher 默认 `TRAIN_SAMPLING_MODE=transition_segments`：围绕折叠后 `ROAD` 或 `EVENT` 发生变化的帧抽 24 帧片段，并补少量纯负片段。Python 入口仍可显式 `--train-sampling-mode full_route` 复现旧行为。
- transition 判定默认按折叠后的二分类标签：`HIGHWAY/NON_HIGHWAY` 与 `RE/UE`。旧细粒度 `R1->R2` 或 `R-E1->R-E4` 不再被当作二分类起跳帧；需要复现旧逻辑时设 `TRANSITION_LABEL_MODE=fine`。
- transition repeat 默认用 `add` 合入已有 UE/regular/joint repeat，而不是旧的 `max`。这样 UE 起跳帧会在 UE 过采样之外额外加权。
- loss 归一化使用 chunk 内 `sum(w * CE) / sum(w)`，不再先对每帧除以自己的权重和；DDP 下分母会跨 rank all-reduce 成 `sum_all(w) / world`，并保留梯度 all-reduce 后的 `div(world)`，因此 ROAD/UE 权重能真正改变跨帧梯度份额且不改变学习率量纲。
- 切 chunk 前用稳定 seed 打乱整个 work 列表，避免同一帧 repeat 和同一 transition 片段连续落进同质 chunk 后再次抵消类别权重；`frames_per_sync` 只作为 heartbeat，chunk loss 会再除以全局最大 chunk 数，避免切分后梯度随 chunk 数放大。
- ROAD 是近似 route-level 属性：launcher 默认 `HIGHWAY_ROUTE_SAMPLE_TARGET=0.5`，优先把含 HIGHWAY 的 route 采样概率拉到 1:1 口径，降低整步没有 ROAD 正例的概率。对应地 `ROAD_LOSS_BALANCE_MODE` 默认改为 `none`，避免 route 采样和帧级 ROAD loss 权重叠加后继续推高 HIGHWAY 先验。基础 ROAD 权重默认保持 `1.4`，与 prompt 侧常量一致。
- 训练帧 repeat 支持 `JOINT_BALANCE_REPEAT_MODE=inverse_sqrt|inverse|none`，默认 `inverse_sqrt`，按 `(ROAD, EVENT)` 四格提高长尾组合曝光；`JOINT_BALANCE_REPEAT_COMBINE=add` 避免被 UE repeat 吞掉；`JOINT_BALANCE_DROP_MAJORITY=1` 会按同一 scale 稳定丢弃部分多数类非 transition 帧。
- 最终 work list 默认再经过 `JOINT_TARGET_BALANCE_MODE=exact`，对已有的 `HIGHWAY/NON_HIGHWAY x UE/RE` 四格做可复现下采样/重复采样，使训练帧的 ROAD 边际和 EVENT 边际都尽量接近 1:1。`JOINT_TARGET_BALANCE_COUNT=0` 表示下采样到当前非空四格的最小数量；显式给正数时才会过采样到指定每格数量。它不会凭空造缺失类别，因此仍需配合 route-level HIGHWAY 采样和 transition segment 抽样。
- `SEGMENTS_PER_ROUTE` 默认 4，避免 transition 多的 route 覆盖回整条 route。训练日志/TB 输出 `selected_frame_rate`、`road_highway_rate`、`event_ue_rate`、teacher-forced ROAD/EVENT accuracy 与 HIGHWAY/UE recall，用来判断采样和监督是否真的对准目标。

UE/RE loss reweight 继续生效：`EVENT` 值 token 在 UE 帧按 `--ue-event-loss-weight` 加权，在 RE 帧按 `--re-event-loss-weight` 加权。默认 `UE_EVENT_LOSS_WEIGHT=2.0`，因为当前采样已经把 UE 帧显著拉高；继续用旧 4.0 容易把 EVENT loss 推得过度偏向 UE。

纯视觉上限实验使用 `PROMPT_MEMORY_MODE=hidden`：prompt 仍显示当前 `EGO_TO_GOAL_XY`，但隐藏 `PREVIOUS_ROAD/PREVIOUS_EVENT`，用于判断视觉信息本身是否足够支撑 ROAD/EVENT。

Prompt 已改为显式比较 4 帧 history 中的相对运动、减速/closing speed、横向进入和道路几何，不再只要求看最新单帧；hidden-memory 模式下 QUESTION 段不会再提示使用不存在的 previous memory。

可选 closed-loop probe：`CLOSED_LOOP_PROBE_STEPS=N` 时，rank0 每 N 个 optimizer step
保存临时 adapter 并连续调用三路 eval：`full` 自然分布 route、
`road_transition` HIGHWAY/NON_HIGHWAY 变化窗口、`event_transition` RE/UE 变化窗口。
原因是 HIGHWAY route 自然率只有约 7%，小样本 full probe 很容易没有 HIGHWAY 正例，
无法判断 ROAD 召回；三路 probe 能把自然分布准确率、ROAD 变化和 EVENT 变化拆开看。
其它 rank barrier 等待，任一任务失败都会中止训练。probe 子进程会清掉 torchrun 的
分布式环境变量，避免误入 4 卡 eval；需要固定 probe 用卡时设
`CLOSED_LOOP_PROBE_GPU_IDS`。默认关闭，避免无意中放慢长训。

## 4. Eval

`eval.py` 每帧 fresh prefill 生成一次两行答案，并用学生输出维护下一帧 memory，不做脚本纠偏。自然 full eval 保留部署分布诊断；新增 `--full-balance-mode joint` 会按 `HIGHWAY/NON_HIGHWAY x UE/RE` 四格均衡抽单帧 case，用来判断模型是否真的学会 HIGHWAY 和 UE，而不是被自然分布里的 NON_HIGHWAY/RE 多数类掩盖。`road_transition` / `event_transition` 可加 `--transition-balance-mode label`，只在转折邻域采样，但分别强制 `HIGHWAY/NON_HIGHWAY` 或 `UE/RE` 正负接近 1:1。

主指标：

- `road_acc`
- `highway_precision/recall/f1`
- `event_acc`
- `ue_precision/recall/f1`
- `joint_acc`
- `road_change_f1`
- `event_change_f1`

逐帧 `frames.jsonl` 保存 GT/PRED ROAD、GT/PRED EVENT、原始生成文本和可选 prompt。

阈值诊断：`eval.py --prediction-mode score` 会对四个 `ROAD x EVENT` 组合做
teacher-forced 值 token log-prob 打分，然后用 `--road-logit-bias` /
`--event-logit-bias` 选择输出。该模式仍按学生预测 closed-loop 更新 memory，
用于区分“模型有判别力但阈值偏保守/偏激进”和“图像判别力本身不足”。

## 5. Compatibility

当前协议：

- `DATASET_VERSION=sft_baseline_highway_reue_joint_v1`
- adapter route: `sft_baseline_highway_reue_joint`
- adapter config: `sft_baseline_adapter_config.json`

旧 `sft_base` 或旧两问 adapter 必须重建/重训，eval 会拒绝 route 或 dataset version 不匹配的 adapter。
