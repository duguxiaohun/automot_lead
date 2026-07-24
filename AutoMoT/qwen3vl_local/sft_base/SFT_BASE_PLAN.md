# SFT Base Plan

`sft_base` 是 `sft_v5` 的直接监督对照组，用来测试“不做 OPSD、不写 CoT、不跑 teacher，只让 Qwen3-VL 直接输出选项”能否训好 RS/EVENT。

## 核心差异

- 复用 `sft_v5` 的数据构建、异常 route 剔除、4 帧 stitched RGB history、`EGO_TO_GOAL_XY`、RS/EVENT 候选池与随机 EVENT 选项。
- Q2 option-letter 扰动使用 `sft_v5` 的 seed namespace，同一 route/frame/seed 下候选集合和 A/B/C 字母映射都与 v5 对齐。
- Q1 仍判 `RS` 与 `ABNORMAL`，Q2 仍在串行上下文中判 `EVENT`。
- 训练目标是普通 teacher-forced CE，不让 student 先 rollout，也不使用 privileged teacher logits。
- 训练时 memory 由 GT answer teacher-forced 更新，用来提供干净直接监督；它不是 v5 的 on-policy student 自维护 memory 分布。`EGO_TO_GOAL_XY` 每帧刷新为当前帧 ego-frame goal；eval 仍按学生自己的 Q1/Q2 输出维护离散 memory。
- Prompt 禁止 CoT；assistant target 只有：

```text
RS: A
ABNORMAL: NO
```

和：

```text
EVENT: B
```

## LoRA

默认只保存 adapter delta，base Qwen checkpoint 只读。和 v2/v5 一样支持 `off/merger/last4/all` 四档视觉 LoRA；本路线默认 `LORA_VISION_SCOPE=merger`，也就是默认微调视觉桥接部分。视觉 LoRA 默认启用 fuse guard，连续异常时保存 `fuse_stop_step_<N>/` 并跳过正常 `final/`。

eval 加载 adapter 前必须读 `sft_base_adapter_config.json`，校验 route、dataset version、base model path 与 vision scope，避免误拿 v2/v5/base adapter 产出无意义指标。

## 实现约束

- `labels.py` 中 `DATASET_VERSION` 标识 sft_base 自己的训练协议；`OPTION_MAP_DATASET_VERSION` 固定为 v5，用来保证 Q2 字母扰动逐样本对齐。
- `prompts.py` 中 memory 的 RS/EVENT 可以跨帧延续，但 `EGO_TO_GOAL_XY` 必须在每帧 prompt 前刷新，不能沿用首帧坐标。
- `train.py` 的 DDP 累积使用手动 `_sync_trainable_grads()`：每帧 forward/backward 都放在 `no_sync()` 内，本 micro-batch 末尾对所有 trainable LoRA 参数补零并 all-reduce 一次。
- `train.py` 里的 memory 更新是 teacher-forced baseline 逻辑；如果要测 v5 on-policy student memory 分布，应回到 `sft_v5` 或另开路线，不在 base 内混用。
