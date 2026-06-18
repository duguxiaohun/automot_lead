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
参与 LoRA 微调时，训练入口通过 `--lora-vision-scope` 或 launcher 的 `LORA_VISION_SCOPE`
选择覆盖范围：

- `off`（默认）：完全不挂视觉侧 LoRA。
- `merger`：只挂 vision merger / patch_embed 等桥接 Linear，最小适配 LEAD stitched RGB
  分布，风险最低。
- `last4`：`merger` 范围 + vision transformer 最后 4 个 block。
- `all`：视觉侧所有 `nn.Linear`，最激进。`--lora-vision` / `LORA_VISION=1` 仍作为该档
  的 legacy 别名保留。

`merger` / `last4` 会通过视觉模块名中的 `blocks.N`、`layers.N` 或
`encoder.layers.N` 解析 vision transformer block 编号。默认
`--strict-vision-scope` 开启；如果真实模型命名漂移导致无法识别任何编号，训练入口会
直接报错，避免误以为在训 `last4` 实际只训 bridge。只有显式
`--no-strict-vision-scope` 时，才退化为只保留 merger / patch_embed 这类非 block 桥接
Linear 并打印 warning。

为了防止视觉表征被冲坏，训练入口默认对视觉组施加三重保险：

- 视觉组 LR = 主 LR × `--vision-lr-scale`（默认 `0.1`），且开启视觉 LoRA 时受
  `--max-vision-lr-scale`（默认 `0.25`）硬上限约束；
- 语言组与视觉组分别 `clip_grad_norm_`，阈值由 `--language-clip-norm`（默认 `1.0`）和
  `--vision-clip-norm`（默认 `0.3`）控制；
- TensorBoard 记录 `train/grad_norm/{language,vision}`、`train/lr_vision`、
  `train/param_norm/lora_{language,vision}` 与 `train/vision_guard_bad_steps`，便于观察
  视觉侧是否异常放大；
- 运行时视觉熔断默认开启：`--vision-guard-enabled` 会监控视觉 LoRA 的梯度范数和
  参数范数，连续异常达到 `--vision-guard-patience` 时停止训练，保存
  `fuse_stop_step_<N>/` 应急 adapter 和 `fuse_reason.txt`，并跳过正常 `final/` 保存，
  避免把异常停训产物误当成完整训练结果。

无论选哪档 scope，原始 Qwen checkpoint 权重仍然只读，不会被覆盖；训练产物只保存
adapter delta，并额外写 `sft_v2_adapter_config.json` 记录 `lora_vision_scope`、
`lora_vision`（bool）、target modules 与保险参数。eval/probe 通过 adapter 配置
自动判断普通 LoRA 还是视觉+语言 LoRA，配置与权重 key 不一致时直接拒绝加载。

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
