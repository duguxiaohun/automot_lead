# SFT Base Simple Plan

`sft_base_simple` 是在 `sft_baseline` 上继续简化的直接监督基线：

- `ROAD`: `HIGHWAY` / `NON_HIGHWAY`
- `EVENT`: `RE` / `UE`
- 训练和测试主集合都按当前帧 GT 四格随机均衡：`HIGHWAY:UE`、`HIGHWAY:RE`、`NON_HIGHWAY:UE`、`NON_HIGHWAY:RE`

## Label Folding

- `HIGHWAY`: 原 RS 为 `R3`。
- `NON_HIGHWAY`: 原 RS 为 `R1/R2/R4/R5`。
- `UE`: 原 EVENT target `abnormal=True`。
- `RE`: 原 EVENT target `abnormal=False`。

数据构建仍复用 `collection_output` 的 RS/EVENT 解析、异常 route 剔除、4 帧 RGB history、`EGO_TO_GOAL_XY` 与 route-level sequence index。

## Prompt

每帧只问一次，assistant 只输出：

```text
ROAD: HIGHWAY|NON_HIGHWAY
EVENT: RE|UE
```

Prompt 保留轻量 memory：

```text
PREVIOUS_ROAD: UNKNOWN|HIGHWAY|NON_HIGHWAY
PREVIOUS_EVENT: UNKNOWN|RE|UE
EGO_TO_GOAL_XY: (...)
```

memory 是弱先验，不是答案；视觉证据优先。

## Training

训练是 teacher-forced weighted CE，不做 OPSD、不跑 privileged teacher、不写 CoT。loss 只落在 `ROAD` 与 `EVENT` 两个值 token 上。

显式 transition 采样已撤掉：

- 不再对 transition center/window 额外 repeat。
- transition 帧如果进入训练，只是因为它的当前帧 GT 落入某个四格桶。
- 训练默认先跨 route 聚合 `FOURBIN_ROUTES_PER_BATCH=16` 条 route，再对聚合后的 frame work list 执行 `JOINT_TARGET_BALANCE_MODE=exact`。
- 默认 `JOINT_TARGET_BALANCE_COUNT=8`，避免 `min(nonempty_sizes)` 把一个 optimizer step 压成很小样本数；空桶仍不会凭空生成，必须看 TB 的 missing-bin 审计。
- 四格均衡后默认关闭额外 UE/regular repeat 与 UE loss 倾斜：`UE_FRAME_REPEAT=1`、`UE_EVENT_LOSS_WEIGHT=1.0`、`UE_REPEAT_MODE=none`、`REGULAR_REPEAT_MODE=none`。
- TensorBoard/日志会记录 balance 后四桶实际样本数：`train/fourbin_*_last_batch`、`train/fourbin_*_rate_last_batch`、`train/fourbin_nonempty_bins_last_batch`。

memory curriculum 保留 baseline 基础比例：

- `MEMORY_RS_WRONG_PROB=0.30`
- `MEMORY_RS_UNKNOWN_PROB=0.40`
- `MEMORY_EVENT_WRONG_PROB=0.35`
- `MEMORY_EVENT_UNKNOWN_PROB=0.35`
- `MEMORY_DROPOUT_PROB=0.15`

新增 early-UE 课程：当前帧处在连续 UE span 的前 `MEMORY_EARLY_UE_FRAMES=4` 帧时，提高 EVENT memory 的 wrong/UNKNOWN/dropout 概率，并更频繁重采 EVENT 扰动段。这样 UE 刚触发时不会总是看到 `PREVIOUS_EVENT=UE`，模型必须从 RGB history 里学触发证据。

early-UE 的 EVENT wrong/UNKNOWN 放大后 cap 在 0.85（保留 >=15% keep 地板），非 early-UE 归一化上限仍为 1.0。early-UE resample guard 改为 post-perturbation：先应用扰动，再检查结果是否落在 UE family，如果是则按 `early_ue_resample_prob` 强制推离为 RE 或 UNKNOWN。训练启动会打印 effective wrong/UNKNOWN/dropout 概率。训练日志和 TensorBoard 同时记录 early-UE prompt memory 的真实 EVENT 分布：`memory/early_ue_event_re_rate_last_batch`、`memory/early_ue_event_ue_rate_last_batch`、`memory/early_ue_event_unknown_rate_last_batch`、`memory/early_ue_event_hidden_rate_last_batch`。

默认只训练语言侧 LoRA：`LORA_VISION_SCOPE=off`。视觉 LoRA 仍可作为显式消融设为 `merger/last4/all`。

## Eval

`eval.py --task full` 默认使用四格 balanced case，但不会把 case 截成单帧。joint 模式会先按当前帧 GT 四格抽受评帧，再按 route 顺序从首帧闭环推进 student memory 到最远受评帧，只在抽中的帧上计 ROAD/EVENT/JOIN accuracy。change matrix 来自这些 route rollout 的相邻帧，`metrics.json` 会同时写 `frames` 与 `rollout_frames`。

```bash
GPU_IDS=0 python qwen3vl_local/sft_base_simple/eval.py \
  --adapter-dir checkpoints/sft_base_simple_runs/latest/final \
  --task full
```

默认 `--full-balance-mode joint --full-balance-cases-per-bin 64`。如需更大测试集，显式提高 `--full-balance-cases-per-bin`。

`--initial-memory-noise none` 不能与 joint balanced eval 同用，避免 sampled frame 直接看到当前帧 GT memory。需要完整闭环变化矩阵时，用 `--full-balance-mode none` 跑整 route eval。

输出仍包含 ROAD/EVENT confusion matrix、change matrix、`metrics.json`、`frames.jsonl`、`summary.md`、`report.html` 和 eval TensorBoard。

## Compatibility

- `DATASET_VERSION=sft_base_simple_highway_reue_fourbin_v1`
- adapter route: `sft_base_simple_highway_reue_fourbin_random`
- adapter config: `sft_base_simple_adapter_config.json`

旧 `sft_baseline` adapter 会被 eval/resume 拒绝，避免混用。
