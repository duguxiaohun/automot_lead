# SFT v5 可视化记录

这个文件记录 SFT v5 教师/学生输入输出的可视化方法。它只保存方法说明，
不保存真实 probe case；真实产物建议放在具体 run 目录下面，例如
`checkpoints/sft_v5_runs/latest/probe_*`。

SFT v5 每帧分成两个问题：

- Q1：判断当前道路结构 `RS`，以及当前是否发生或处在异常事件中。
- Q2：在 Q1 的道路结构基础上，从当前帧候选里判断 `EVENT`。

这里需要区分三类检查，它们目的不同，不应该混在一起看。

## A. 训练前：默认 Qwen 的 OPSD 能力与 prompt 检查

训练前检查分两类：

1. `probe.py`：检查默认 Qwen 在学生 prompt / privileged teacher prompt 下的 OPSD
   能力，以及 system/user/messages、teacher target、memory、候选池是否合理。
2. `test_batched_qwen_smoke.py`：检查阶段 1 grouped Qwen 路径和单样本路径是否等价，
   尤其是 Q1/Q2 token、文本和训练 KL logits。

### A.1 默认 Qwen prompt / teacher-student 能力检查

目的：在真正开始 OPSD / DDP 训练前，先用默认 `Qwen3-VL-4B-Instruct`
同时跑学生 prompt 和 privileged teacher prompt。这个检查必须是纯 base Qwen，
不导入任何 LoRA，也不要传 `--adapter-dir`，用来确认：

- 默认 Qwen 作为 student 时，是否能理解 Q1/Q2 选择题格式、RS 选项和 EVENT 选项。
- 默认 Qwen 作为 teacher 时，吃到 XML weather、GT RS、GT event 等私有参考后，
  是否能给出稳定、合理、可被解析的 teacher 分析与答案。
- prompt 是否诱导模型泄漏私有字段、复读候选、漏掉三段式 CoT
  `Scene Description / Critical Object Description / Reasoning on Intent`
  或 `RS/ABNORMAL/EVENT` 等关键输出字段。
- system prompt 是否简洁提醒模型关注交通灯/标志、周围车辆/行人/障碍物、
  车道线/道路结构和影响自车决策的关键因素。
- `q1_*_user_prompt.txt` 的 `[MEMORY]` 是否只包含自然语言 `BELIEVED_RS` 和
  `EGO_TO_GOAL_XY=(+x, +y) m`，不包含 `BELIEVED_EVENT`，也不包含 `A -` 这类选项前缀。
- `q2_*_user_prompt.txt` 的 `[MEMORY]` 才包含自然语言 `BELIEVED_EVENT`，但仍不写
  `RE -` 或 `U-E* -` 标签前缀。
- 如果看到 `EGO_TO_GOAL_XY=UNKNOWN`，先检查 `labels.json` 里的 `ego_to_goal_xy`
  是否为 `null`；这表示 probe 使用了旧 sequence index，需要重跑 build_dataset 和 probe。
- Q2 的 `RE` 文案和当前帧 `U-E*` 候选是否足够清晰。
- Q2 是否确实作为 Q1 assistant 输出后的第二轮 user turn 续接 KV cache，而不是重新
  fresh prefill 同一帧。

从 `AutoMoT/` 目录运行：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_v5_runs/pre_opsd_base_probe \
  --num-cases 8 \
  --with-model \
  --with-teacher-model \
  --with-teacher
```

这个命令不传 `--adapter-dir`，因此 student 和 teacher 都是默认/base Qwen；
区别是 student 只看 RGB + 学生 prompt，teacher 看 RGB + privileged teacher prompt。
这一类训练前体检不要加载任何 adapter，否则看到的就不是普通 Qwen 是否足够支撑
OPSD。

会额外生成：

- `q1_student_output.txt` / `q2_student_output.txt`：默认 Qwen 在学生输入下的输出。
- `q1_teacher_output.txt` / `q2_teacher_output.txt`：默认 Qwen 在 teacher 私有输入下的输出。
  合格内容应当像学生输出一样，从 `Scene Description:` 开始写分析和答案；如果看到它
  复读 `[MEMORY]`、`[RS_CHOICES]`、`[REFERENCE]` 等输入块，说明这份 demo 是旧
  prompt 产物，或默认 Qwen 没有遵守格式，需要用当前代码重新跑 probe。
- `flags.json` 里的 `parsed_teacher_q1`、`parsed_teacher_q2`、
  `q1_teacher_rs_correct`、`q1_teacher_abnormal_correct`、`q2_teacher_event_correct`、
  `q2_student_continued_from_q1_kv`、`q2_teacher_continued_from_q1_kv`。

如果同卡同时加载 student 和 teacher 两份 Qwen 显存不够，可以分两次跑：

```bash
# 只看默认 Qwen 学生能力，不传 --adapter-dir
GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_v5_runs/pre_opsd_base_student_probe \
  --num-cases 8 \
  --with-model

# 只看默认 Qwen teacher 能力，不传 --adapter-dir
GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_v5_runs/pre_opsd_base_teacher_probe \
  --num-cases 8 \
  --with-teacher-model \
  --with-teacher
```

训练前重点看：

- `q1_student_output.txt` 能否稳定输出 `RS: <A-E>` 和 `ABNORMAL: YES/NO`。
- `q2_student_output.txt` 能否只从当前 `EVENT_CHOICES` 里选，不编造选项。
- `q1_teacher_output.txt` / `q2_teacher_output.txt` 是否能利用私有参考做更稳的分析，
  但最终表述不要依赖学生看不到的字段名。
- `q1_teacher_output.txt` / `q2_teacher_output.txt` 是否从 `Scene Description:`
  直接开始；若复读输入 prompt，优先检查 `q*_teacher_user_prompt.txt` 是否包含
  `Output exactly these lines:`，并重新生成 demo。
- `q1_student_messages.json` / `q2_student_messages.json` 是否清楚区分 system role
  和 user role；`q2_*_messages.json` 里的 prompt 是第二轮 user turn 的内容，模型输出
  实际由 Q1 KV cache 续接得到。
- `flags.json` 里 teacher/student 解析字段是否为空；为空说明 prompt 或解析合同要先修。
- `flags.json` 里的 `student_adapter_dir` 必须为空；否则说明训练前体检误加载了 LoRA，
  需要重跑纯 base Qwen 检查。

### A.2 grouped / parallel Qwen 等价性检查

目的：在启用 `QWEN_BATCH_SIZE>1` 和默认 `PARALLEL_KL=1` 前，确认 Q1/Q2 batched
student rollout、Q1/Q2 parallel KL 都没有改变训练语义。这个检查也必须使用
默认/base Qwen，不传 `--adapter-dir`；只有当你想专门检查某个已训练 adapter 的
grouped/parallel 路径时，才显式传 `--adapter-dir`。

默认命令偏向检查“混长 padded rollout 是否和单样本等价”，并会主动制造 padding 压力：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/test_batched_qwen_smoke.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --num-cases 2 \
  --output-json checkpoints/sft_v5_runs/batched_qwen_smoke.json
```

强制验证真实 batched rollout 和 parallel KL 时，用：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/test_batched_qwen_smoke.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --num-cases 2 \
  --candidate-pool 256 \
  --require-batched-group \
  --no-prefer-different-lengths \
  --check-parallel-kl \
  --output-json checkpoints/sft_v5_runs/batched_qwen_smoke_require_batch.json
```

默认命令会优先从 `--candidate-pool` 里挑 Q1 input length 差异大的 case，主动制造
padding 压力；`--require-batched-group` 要求实际运行到 size>=2 的 batched rollout。
`--check-parallel-kl` 会额外比较新 chunk parallel KL 与旧逐帧 KL 的总 loss、
逐 case loss 和 Q1/Q2 loss parts。Q2 KL scoring 必须按旧逐帧语义先把精确
`q1_ids` 追加到 Q1 prompt KV，再追加 Q2 user turn；不能把 `q1_ids` decode 成
`q1_text` 后重新 tokenize 成 full dialog 来替代。
合格时需要看到：

- `ok=true`。
- `padding_pressure=true` 时，混长 case 仍能通过，因为 padded KV 只用于 no-grad
  student 采样，后续 Q1/Q2 KL 会重建 batched student/teacher prompt state；
  Q2 KL 使用精确 `q1_ids` 续接 Q1 KV。
- `actual_batched_group_sizes` 非空且 `actual_batched_frames>=2`，才说明这次真的测到了
  size>=2 的 batched rollout。
- 每个 case 的 `q1_ids_equal=true`、`q1_text_equal=true`。
- 每个 case 的 `q2_ids_equal=true`、`q2_text_equal=true`。
- `q1_logits_max_abs <= logit_atol`，默认 `logit_atol=0.5`；这是训练真正使用的
  `_append_token_ids_with_logits` 路径，不只是自由生成文本。
- 开启 `--check-parallel-kl` 时，`parallel_kl.ok=true`，并且 `loss_abs_diff`、
  `case_loss_max_abs_diff`、`parts_max_abs_diff` 都不超过 `parallel_loss_atol`。
- `adapter_dir=null`，表示没有误加载 LoRA。

训练时再看 TensorBoard：

- `qwen/q1_batched_frame_rate` 是所有已训练 Q1 frame 的真实 batched 比例。
- `qwen/q1_batched_frame_rate_grouped` 只表示进入 grouped 路径后的内部比例，不能当成
  全局 batch 生效率。
- `parallel_kl/frame_rate` 表示当前 logging window 内走 chunk 级 batched KL 的 frame
  比例；如果长期为 0，说明 parallel KL 没有真正生效。
- `parallel_kl/fallbacks` 非 0 时，优先用 `PARALLEL_KL_TRACEBACK=1` 定位普通兼容问题；
  CUDA OOM 不会 fallback。
- `train/q1_token_cap_hit_rate` / `train/q2_token_cap_hit_rate` 记录 Q1/Q2 是否打满
  `MAX_NEW_TOKENS_Q1/Q2`；如果长期非 0，说明 1024 安全上限正在截断输出，远端要优先
  检查模型是否不出 EOS / `<|im_end|>`。
- 如果 `qwen/q1_batched_frame_rate` 接近 0，优先查 `[warn] q1 batch fallback`；
  没有 fallback 时默认 max_util 8 路配置应稳定看到 `batched_frames=8`。

四卡多 batch 训练 demo：

```bash
BATCH_PROFILE=max_util \
LOGGING_STEPS=1 PROGRESS_FRAMES=20 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp
```

现在 `GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp` 默认就是
`BATCH_PROFILE=max_util / PER_DEVICE_BATCH_SIZE=8 / QWEN_BATCH_SIZE=8`：会启动 4 个 rank，
并让每卡同一 timestep 有 8 个 frame 可尝试 Q1 grouped/batched
rollout。是否真的并行，以 `actual_batched_frames`、`[q1-grouped] batched_frames=...`
和 TensorBoard 的 `qwen/q1_batched_frame_rate` 为准。训练日志第一条 `[batch-start]`
应显示 `routes=8` / `qwen_batch=8`；如果是 `routes=6` / `qwen_batch=6`，说明本次
run 使用了 `BATCH_PROFILE=balanced`；如果是 4 路，则是 debug 或显式覆盖了 batch。

如果 8 路不稳，先退回 balanced 6 路；如果 6 路仍不稳，再退回 debug 4 路：

```bash
BATCH_PROFILE=balanced \
LOGGING_STEPS=1 PROGRESS_FRAMES=20 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp
```

```bash
BATCH_PROFILE=debug \
LOGGING_STEPS=1 PROGRESS_FRAMES=20 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp
```

注意：Qwen 输出不做 `ABNORMAL:` / `EVENT:` 字段早停，但仍保留
`MAX_NEW_TOKENS_Q1=1024` / `MAX_NEW_TOKENS_Q2=1024` 作为安全上限。parallel KL 的显存
峰值主要来自近似 `batch x rollout_len x vocab` 的 student/teacher logits；如果 8 路
或 1024 上限导致 OOM，先退回 `BATCH_PROFILE=balanced` / `BATCH_PROFILE=debug`，
必要时再用 `PARALLEL_KL=0` 定位，不要改成字段早停。

代码审阅时同步检查注释：

- `_kv_start_state_batch_padded` 必须写清楚：padding KV 只允许用于 no-grad rollout，
  不能直接写回 memory；parallel KL 会重建 batched student/teacher prompt state。
- `_slice_kv_state_batch` 与 `mrope_utils.py` 必须兼容 `(batch,1)` / `(1,batch)` 两种
  `rope_deltas` 方向；如果 log 里出现
  `Target sizes: [1, -1]. Tensor sizes: [2, 1]` 的 `[warn] q1 batch fallback`，
  通常说明 active batch 缩小时 M-RoPE delta 没跟着样本行正确切片。
- `_student_generate_kv_batch` 必须写清楚：EOS 样本要从 active batch 移除，Q2 才能接在
  干净的 Q1 assistant KV 后。
- `_q2_full_messages` 只允许用于 Q2 student rollout 采样；Q2 KL 必须通过精确
  `q1_ids` 续接 Q1 KV 后再追加 Q2 user turn。
- `test_batched_qwen_smoke.py` 必须写清楚：默认 smoke 主要验证 mixed-length padded
  rollout，`--require-batched-group` 且 `actual_batched_frames>=2` 才证明真实 batch。

## B. 训练后：adapter 学生输入输出可视化

目的：训练结束后检查当前 adapter 学生在真实推理状态机下的表现。此时重点不是看
base Qwen 强不强，而是看训练出的学生是否：

- Q1 RS 错时正确停止本帧 Q2，并在下一帧恢复 `GT RS + RE` 默认 memory。
- Q1 正确时进入 Q2，且 Q2 只在当前帧候选里输出 `RE` 或 `U-E*`。
- `memory_before.json` / `memory_after.json` 符合 v5 状态机。
- 错误样本能通过 RGB、prompt、label、flags 定位到原因。

从 `AutoMoT/` 目录运行：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_v5_runs/latest/final \
  --output-dir checkpoints/sft_v5_runs/latest/probe_with_adapter \
  --num-cases 8 \
  --with-model \
  --with-teacher
```

这个命令只加载 student adapter，不额外加载 teacher Qwen。它会填充：

- `q1_student_output.txt`
- `q2_student_output.txt`
- `flags.json` 里的 `parsed_q1`、`parsed_q2`、`q1_rs_correct`、
  `q1_abnormal_correct`、`q2_event_correct`、`q2_invalid_output`、`rs_wrong_reset`

如果训练后也想把 adapter student 和 base teacher 放在同一个 case 里对照，可以额外加
`--with-teacher-model`，但显存会同时常驻两份 Qwen。

## C. 静态 prompt / target 合同快检

目的：不加载模型，只快速检查数据、候选池、teacher/student prompt 隔离、teacher target
清洗、RGB history 复制和 timeline 是否完整。

从 `AutoMoT/` 目录运行：

```bash
python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --output-dir checkpoints/sft_v5_runs/latest/probe_static \
  --num-cases 24 \
  --with-teacher
```

这个命令不占 GPU。它会写 teacher prompt 和脚本化 teacher target，但不会生成
`q*_student_output.txt` 或 `q*_teacher_output.txt` 的模型文本。

## 输出结构

```text
probe*/
  manifest.json
  route_<idx>__<scenario>__<route_id>/
    timeline.json
    timeline.png
    frame_<frame_id>/
      rgb_00.jpg
      rgb_01.jpg
      rgb_02.jpg
      rgb_03.jpg
      rgb_paths.json
      q1_system_prompt.txt
      q1_student_user_prompt.txt
      q1_student_messages.json
      q1_student_prompt.txt
      q1_student_output.txt
      q1_teacher_user_prompt.txt
      q1_teacher_messages.json
      q1_teacher_prompt.txt
      q1_teacher_target.txt
      q1_teacher_output.txt
      q2_system_prompt.txt
      q2_student_user_prompt.txt
      q2_student_messages.json
      q2_student_prompt.txt
      q2_student_output.txt
      q2_teacher_user_prompt.txt
      q2_teacher_messages.json
      q2_teacher_prompt.txt
      q2_teacher_target.txt
      q2_teacher_output.txt
      step1_user.txt
      step1_student.txt
      step1_teacher_user.txt
      step1_teacher.txt
      step1_teacher_output.txt
      step2_user.txt
      step2_student.txt
      step2_teacher_user.txt
      step2_teacher.txt
      step2_teacher_output.txt
      labels.json
      memory_before.json
      memory_after.json
      flags.json
```

其中：

- `q1_*` 是 v5 原生命名，对应第一问。
- `q2_*` 是 v5 原生命名，对应第二问。
- `q*_system_prompt.txt` / `q*_student_user_prompt.txt` / `q*_teacher_user_prompt.txt`
  把 system prompt 和 user prompt 分开保存，解决旧版 demo 里 role 边界不清的问题。
- `q*_messages.json` 是可序列化的 Qwen chat messages：system 为固定 v5 协议，user
  content 中先列 4 张 RGB，再列文本 prompt。图片用文件名和原路径表示，不嵌入 PIL。
- `step1_*` / `step2_*` 是仿 v3 的别名，方便用同一套人工检查习惯对比 v3/v5。
- `rgb_00.jpg` 到 `rgb_03.jpg` 是真实输入 Qwen 的 4 帧 stitched RGB history。
- `labels.json` 保存 RS/EVENT 标签、候选池、`event_code`、`regular_event_codes`
  `ego_to_goal_xy` 和 teacher-only weather 文本。
- `memory_before.json` / `memory_after.json` 保存该帧前后的内部
  `RS + EVENT + EGO_TO_GOAL_XY` memory；真正写入 Qwen 的 prompt 里，Q1 只渲染
  road-only memory，Q2 才渲染 event memory。
- `flags.json` 保存解析出的学生输出、teacher 输出、是否 RS 正确、是否进入 Q2、
  是否 candidate mismatch、是否 reset 下一帧、是否误传 `student_adapter_dir`
  以及 Q2 是否续接 Q1 KV cache 等诊断字段。

## Timeline 颜色

- 红色：Q1 的 RS 错误，下一帧会恢复 `GT RS + RE`。
- 蓝色：Q1 的 RS 正确，本帧进入 Q2。
- 绿色：未加载 student 模型的静态 teacher-forced dump。
- 灰色：没有特别转折的普通帧。

## 人工检查清单

- `q1_student_user_prompt.txt` / `q2_student_user_prompt.txt` 不应包含 `XML_WEATHER`、
  `ANSWER_`、`REFERENCE`、GT label 或 scenario name。
- `q1_system_prompt.txt` / `q2_system_prompt.txt` 应为固定 v5 system prompt；
  `q*_messages.json` 应能看到 system/user 分离和 4 帧 RGB 顺序。
- `q1_teacher_user_prompt.txt` / `q2_teacher_user_prompt.txt` 可以包含 teacher 私有参考信息。
- `q1_teacher_target.txt` / `q2_teacher_target.txt` 必须是学生视角文本，不能泄漏
  `ANSWER_`、`REFERENCE`、`XML_WEATHER` 这类私有字段名。
- `q1_teacher_output.txt` / `q2_teacher_output.txt` 是模型生成文本，只在
  `--with-teacher-model` 时非空，用来评估默认 Qwen 老师能力和 prompt 合理性。
  它不是脚本化标签；合格输出应包含 `Scene Description / Critical Object Description /
  Reasoning on Intent / RS 或 EVENT`。如果文件内容像 prompt 续写，先确认是否为旧
  probe 产物，再重跑当前版本。
- `q1_student_output.txt` 应按三段式 CoT 输出 `Scene Description`、
  `Critical Object Description`、`Reasoning on Intent`，然后输出 `RS` 和 `ABNORMAL`；
  `q2_student_output.txt` 应按同样三段式 CoT 输出后给出 `EVENT`。
- `q2_student_user_prompt.txt` 应该显示 `RE` 加当前帧允许的 `U-E*` 候选；`RE` 文案里
  应覆盖当前帧 `regular_event_codes` 对应的 regular 行为。
- `memory_before.json` / `memory_after.json` 应符合 v5 状态机：
  Q1 RS 错误时跳过本帧 Q2，并在下一帧恢复 `GT RS + RE`；Q2 非法输出不污染 memory。
- `flags.json` 中 `q2_student_continued_from_q1_kv=true` 表示 student Q2 是接在 Q1
  KV cache 后继续问；`q2_teacher_continued_from_q1_kv=true` 表示 teacher 模型输出也
  是同样的第二轮对话体检。
