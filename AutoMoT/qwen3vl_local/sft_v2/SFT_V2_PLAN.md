# SFT v2 方案说明

SFT v2 是一条独立的 LoRA 串行选择题监督路线，不替代
`qwen3vl_local/sft/`。

当前任务明确拆成两个阶段：

1. 输入 RGB clip 和场景选择 prompt，只要求模型输出 `SCENE`。
2. 如果预测出的场景合法，就在同一条对话后追加一个新的 user prompt，使用该预测
   场景自己的 `EVENT_SEQUENCE`，复用第一阶段已经吃过图像和场景 prompt 后的 KV
   cache，只要求模型输出 `STATUS` 和 `SUBGOAL`。

第二阶段必须复用第一阶段的视觉和 prompt KV cache，避免重新编码 RGB clip。若第一
阶段预测的 scene 不在 `SCENE_CHOICES` 里，eval 立即终止该样本并计入 invalid scene。
若第一阶段预测出合法但错误的 scene，第二阶段仍然按预测 scene 的事件序列继续走；
串行指标会把后续 status/subgoal 计错，除非 scene 本身也正确。

## 数据

`build_dataset.py` 复用旧 SFT 的 keyframe 时间线和 keep/advance 采样逻辑：

- 输入仍是 4 张 LEAD 三视角拼接 RGB，按从旧到新的顺序排列。
- `SCENE` 标签来自当前 run 的 scenario。
- `STATUS` 标签来自包含 anchor frame 的 GT interval。
- `SUBGOAL` 是 `prompt_pipeline.get_full_sequence(scene)` 中当前事件的下一步。
- `PREVIOUS_STATUS_HINT` 来自 `anchor - K`，用来保留记忆语义。
- 默认 `--samples-per-scenario 0`，表示每个 scenario 保留全部合法候选；正数才启用
  旧的按场景下采样路径。
- train 默认启用 `--wrong-scene-ratio 0.15` 增强：一部分第二阶段 prompt 会故意写入
  一个错误但合法的 selected scene。previous hint 和监督的 `STATUS/SUBGOAL` 会按状态机
  相位映射到该 selected scene 自己的 `EVENT_SEQUENCE`，让模型见过 eval 时“scene 合法
  但错了”的情况，同时不破坏第二阶段的选择约束。
- val 不做 wrong-scene 增强，保持真实验证分布。

每条 jsonl row 都保存 `stage_messages.scene` 和 `stage_messages.status`。只有 scene
阶段包含图像占位符；status 阶段是追加的纯文本 follow-up turn。

## Prompt

`prompts.py` 是唯一 prompt 来源：

- 第一阶段使用 `SCENE_SYSTEM_PROMPT` 和 `build_scene_user_prompt(...)`。
- 第二阶段追加 `build_status_user_prompt(selected_scene=...)` 作为后续 user turn。
- 第一阶段列出完整 `SCENE_CHOICES`。
- 第二阶段只列 selected scene 自己的 `EVENT_SEQUENCE` 和事件描述。
- system/user prompt 正文保持英文，因为它直接参与 Qwen 训练和评估分布；开发文档和
  代码注释使用中文。

## Loss

`train.py` 加载本地 `Qwen3-VL-4B-Instruct`，冻结 base model，注入 PEFT LoRA，并对每个
样本跑一次多轮 teacher-forced forward。默认 LoRA 只注入语言侧 Linear；需要让视觉侧也
参与 LoRA 微调时，训练入口显式加 `--lora-vision`（或 launcher 设 `LORA_VISION=1`）：

- 第一段 assistant turn：只监督 `SCENE` 的值 token。
- 第二段 assistant turn：只监督 `STATUS` 和 `SUBGOAL` 的值 token。
- prompt、图像、system token 的 loss 为 0。
- `SCENE:`、`STATUS:`、换行等格式 token 的 loss 为 0。
- wrong-scene 增强 row 仍然监督第二段 assistant turn，但监督值已经映射成 selected scene
  内合法的事件。
- 不再有 teacher、ANALYSIS、pending placeholder，也不再有离线 teacher cache。

这样可以让 status/subgoal 训练条件化在前面的 scene turn 上，同时只把 loss 打在真正
需要复制的离散选择值上。

## Eval

`eval.py` 按真实串行协议自由生成：

- 生成第一阶段 `SCENE`。
- 如果 scene 非法，终止该样本并计入 `invalid_scene`。
- 如果 scene 合法，使用预测 scene 而不是 GT scene 构造第二阶段 prompt，并从第一阶段
  KV cache 继续解码。
- 生成 `STATUS/SUBGOAL`。
- previous-status hint 会先映射到预测 scene 的同相位事件，保证第二阶段 prompt 内部
  自洽。

主要指标：

- `scene_accuracy`
- `status_accuracy`
- `subgoal_accuracy`
- `all_accuracy`
- `status_raw_accuracy` / `subgoal_raw_accuracy`，只用于诊断
- `invalid_scene_rate`
- `invalid_status_for_pred_scene_rate`
- `subgoal_not_next_rate`
- `status_kv_reuse_rate` / `status_kv_fallback_rate`
- `valid_total` 与 `*_valid_scene` 指标，其中 invalid-scene 样本不进入分母

`status_accuracy` 和 `subgoal_accuracy` 是串行指标：scene 必须也正确。

## TensorBoard

训练时 rank0 会把标量写到当前 run 的 `tb/` 子目录，例如
`checkpoints/sft_v2_lora/latest/tb`。使用项目通用启动器查看：

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_v2_lora/latest/tb
```

如果想让 TensorBoard 扫描整个 SFT v2 输出根目录，也可以指向：

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_v2_lora/latest
```

当前训练写入的主要曲线是 `train/loss`、`train/lr`，以及按 `EVAL_STEPS` 触发时写入的
`val/loss`、`val/samples`、`val/skipped`。
