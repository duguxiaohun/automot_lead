# SFT v5 可视化记录

这个文件是 SFT v5 教师/学生输入输出、产物目录和人工检查项的完整说明。
`SFT_V5_RUN.md` 只保留常用命令，不再重复本文的文件级细节。本文不保存真实
probe case；真实产物建议放在具体 run 目录下面，例如
`checkpoints/sft_v5_runs/latest/probe_*`。

SFT v5 是双频两问协议：

- Q1 / `RS_SLOW`：保留三段分析并判断 `RS`；稳定时默认每 4 帧运行，
  错误/UNKNOWN/recovery 时恢复逐帧。
- Q2 / `EVENT_FAST`：每个 RS gate 正确的帧都重新分析本帧 RGB，直接从
  显式标注 `[RE | REGULAR]` / `[UE | UNUSUAL]` 的混合候选里判断 `EVENT`，
  不再单问当前是否异常。

这里需要区分五类检查：训练前 base 能力、batched 等价性、训练中自动版本对比、
训练后手动 adapter 检查、静态合同快检。它们目的不同，不应该混在一起看；大样本
eval 则负责总体统计，不属于小样本可视化。

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
  或 `RS/EVENT` 等关键输出字段。
- system prompt 是否简洁提醒模型关注交通灯/标志、周围车辆/行人/障碍物、
  车道线/道路结构和影响自车决策的关键因素。
- `q1_*_user_prompt.txt` 的 `[MEMORY]` 是否只包含自然语言 `PREVIOUS_RS_HYPOTHESIS`、
  `MEMORY_RELIABILITY` 和 `EGO_TO_GOAL_XY=(+x, +y) m`，不包含
  `PREVIOUS_EVENT_HYPOTHESIS`，也不包含 `A -` 这类选项前缀。
- `q2_*_user_prompt.txt` 的 `[MEMORY]` 才包含自然语言 `PREVIOUS_EVENT_HYPOTHESIS`，但仍不写
  `RE -` 或 `U-E* -` 标签前缀。
- 如果看到 `EGO_TO_GOAL_XY=UNKNOWN`，先检查 `labels.json` 里的 `ego_to_goal_xy`
  是否为 `null`；这表示 probe 使用了旧 sequence index，需要重跑 build_dataset 和 probe。
- Q2 的所有选项是否显式标注 `[RE | REGULAR]` / `[UE | UNUSUAL]`，且文案足够清晰。
  完整 family 标识只出现在当帧 choices；memory 仍只保存无标签前缀的自然语言 hypothesis。
- 慢帧 Q2 是否作为 Q1 assistant 输出后的第二轮 user turn 续接当帧 KV；快帧
  是否标注 `fresh_rgb_prefill`，并且不伪造、不复用上一慢帧 Q1 分析。

从 `AutoMoT/` 目录运行：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_v5_runs/pre_opsd_base_probe \
  --num-routes 1 \
  --with-model \
  --with-teacher-model \
  --with-teacher
```

这个命令不传 `--adapter-dir`，因此 student 和 teacher 都是默认/base Qwen；
区别是 student 只看 RGB + 学生 prompt，teacher 看 RGB + privileged teacher prompt。
这一类训练前体检不要加载任何 adapter，否则看到的就不是普通 Qwen 是否足够支撑
OPSD。

每帧重点看：

- `output.json.student`：默认 Qwen 在学生输入下的 Q1/Q2 原始输出和解析结构。
- `output.json.teacher`：默认 Qwen 在 teacher 私有输入下的 Q1/Q2 原始输出和解析结构。
  合格内容应当像学生输出一样，从 `Scene Description:` 开始写分析和答案；如果看到它
  复读 `[MEMORY]`、`[RS_CHOICES]`、`[REFERENCE]` 等输入块，说明这份 demo 是旧
  prompt 产物，或默认 Qwen 没有遵守格式，需要用当前代码重新跑 probe。
- `input.json`：teacher/student Q1 prompt 与 Q2 KV 续接 user turn。
- `memory.json`：两问的 student memory 输入/输出和只读 truth 对照。

如果同卡同时加载 student 和 teacher 两份 Qwen 显存不够，可以分两次跑：

```bash
# 只看默认 Qwen 学生能力，不传 --adapter-dir
GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_v5_runs/pre_opsd_base_student_probe \
  --num-routes 1 \
  --with-model

# 只看默认 Qwen teacher 能力，不传 --adapter-dir
GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_v5_runs/pre_opsd_base_teacher_probe \
  --num-routes 1 \
  --with-teacher-model \
  --with-teacher
```

训练前重点看：

- 触发 RS_SLOW 的帧，`output.json.student.q1_output` 能否稳定输出 `RS: <A-E>`；
  快帧该字段应为空，且 `q1_triggered=false`。
- `output.json.student.q2_output` 能否只从当前 `[RE | REGULAR]` /
  `[UE | UNUSUAL]` 混合
  `EVENT_CHOICES` 里选，不编造选项。
- `output.json.teacher` 是否能利用私有参考做更稳的分析，
  但最终表述不要依赖学生看不到的字段名。
- teacher 输出是否从 `Scene Description:` 直接开始；若复读输入，检查 `input.json`。
- `input.json` 是否清楚区分 system/user，以及 EVENT_FAST 是否正确标记
  `student.q1_output_kv` 或 `fresh_rgb_prefill`。
- `output.json` 的 teacher/student 解析字段是否为空；为空说明 prompt 或解析合同要先修。
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
逐 case loss 和 Q1/Q2 loss parts。grouped Q2 rollout 与 Q2 KL scoring 都必须先把
精确 `q1_ids` 追加到 Q1 prompt KV，再追加 Q2 user turn；不能把 `q1_ids` decode 成
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
- 每个 case 的 `grouped_q2_ids_equal=true`、`grouped_q2_text_equal=true`，证明训练实际
  grouped Q2 采样与单样本精确 Q1 KV 续接一致。
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

注意：Qwen 输出不做 `RS:` / `EVENT:` 字段早停，但仍保留
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
- grouped Q2 student rollout 与 Q2 KL 都必须先 prefill 当帧 Q1 图文 prompt，再通过
  原始 `q1_ids` 精确续接 Q1 KV 后追加 Q2 user turn；两条路径都禁止把 `q1_text`
  放回 full dialog 重新 tokenize。
- `test_batched_qwen_smoke.py` 必须写清楚：默认 smoke 主要验证 mixed-length padded
  rollout，`--require-batched-group` 且 `actual_batched_frames>=2` 才证明真实 batch。

## B. 训练中：base 与 checkpoint LoRA 自动对比

正式 `train.sh ddp` 默认启用自动 probe。按当前约 80 optimizer steps/day 的实测速度，
默认每 `40 step`（约半天）保存一版：

```text
SAVE_STEPS=40
CHECKPOINT_PROBE=1
CHECKPOINT_PROBE_BASE=1
CHECKPOINT_PROBE_WITH_TEACHER=1
CHECKPOINT_PROBE_NUM_CASES=24
CHECKPOINT_PROBE_NUM_ROUTES=1
CHECKPOINT_PROBE_SAMPLE_MODE=random
CHECKPOINT_PROBE_ARTIFACT_LEVEL=review
CHECKPOINT_PROBE_CONTEXT_RADIUS=8
```

训练开始时生成 `probes/base/`，每次保存 `checkpoint-40/80/...` 后生成对应
`probes/checkpoint-000040/000080/...`，正常结束保存 `final/` 后生成 `probes/final/`。
所有版本使用相同 seed 和相同 `random` 规则固定选择 1 条完整 route ID，从首帧运行到
末帧。完整 ID 用于观察学生 memory 的全部 step-by-step 变化：

- `base`：student 与 teacher 都临时关闭 LoRA，记录未训练 Qwen 的表现。
- `checkpoint-*` / `final`：student 使用当前 LoRA；teacher 临时关闭 LoRA，保持纯 base
  privileged teacher，便于判断学生是否相对同一个老师改善。
- rank0 复用训练进程里已经加载的 Qwen+LoRA，不另起 `probe.py` 子进程，也不加载
  第二份 Qwen；其它 rank 在 barrier 等待，完成后恢复训练并清理 CUDA cache。

训练显存趋势不从 probe 文件判断，而看同一 run 的 TensorBoard：
`memory/allocated_gb`、`memory/reserved_gb`、`memory/max_allocated_gb`、
`memory/max_reserved_gb`。其中 `allocated` 是活跃引用主口径；`reserved` 或
`nvidia-smi` 进程显存停在历史高位，不能单独证明泄漏。

错误 memory 课程是否达到预期则看：
`memory/{rs,event}_input_anomaly_rate`、`memory/{rs,event}_wrong_copy_rate`、
`memory/{rs,event}_recovery_rate`、`memory/{rs,event}_error_streak_mean` 和
`train/q2_skip_due_rs_rate`。RS 默认目标不是固定比例，但长期 anomaly 超过 30% 或
Q2 trigger 低于 70% 通常说明 RS 扰动/延迟过强，会挤压 EVENT 训练。

主要入口：

```text
checkpoints/sft_v5_runs/latest/probes/comparison.json
checkpoints/sft_v5_runs/latest/probes/base/results.json
checkpoints/sft_v5_runs/latest/probes/checkpoint-000040/results.json
checkpoints/sft_v5_runs/latest/probes/final/results.json
```

`comparison.json` 会集中记录 RS_SLOW 触发/准确率、EVENT_FAST、由 EVENT 折叠的
UE 假阳性/假阴性和端到端
指标。默认 review 已逐帧复制真实 RGB，并保存精简 input/output/memory；只有 legacy
逐项 TXT/JSON 需要设置 `CHECKPOINT_PROBE_ARTIFACT_LEVEL=full`。

关闭自动检查，或临时改成少量 RS 变化帧专项：

```bash
CHECKPOINT_PROBE=0 GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp

CHECKPOINT_PROBE_SAMPLE_MODE=rs_transition CHECKPOINT_PROBE_NUM_CASES=4 \
GPU_IDS=0,1,2,3 \
bash qwen3vl_local/sft_v5/train.sh ddp
```

自动 probe 的 `256/192` token 是可视化安全上限，不会改变训练的 `1024/1024`。
probe 失败会写 `error.txt` 并继续训练，不会因为旁路可视化终止长跑。
手工 `probe.py` 默认使用 1024/1024，并在 `results.json.summary.generation_limits`
记录实际值；因此手工完整检查和自动轻量对比可以明确区分。

## C. 训练后：adapter 学生输入输出可视化

目的：训练结束后检查当前 adapter 学生在真实推理状态机下的表现。此时重点不是看
base Qwen 强不强，而是看训练出的学生是否：

- Q1 RS 错时停止本帧 Q2，但下一帧必须再次运行 Q1，并继续使用学生自己的 memory
  观察后续自主纠正；不能把 RS repair interval 当成 Q1 跳帧间隔。
- Q1 正确时进入 Q2，且 Q2 只输出当前帧候选字母；parser 再按
  `event_option_map` 还原为 `RE` 或 `U-E*`。
- `memory.json` 分清 Q1/Q2 输入、输出、下一帧 student memory 和只读 truth memory。
- 错误样本能通过 RGB、prompt、label、flags 定位到原因。

从 `AutoMoT/` 目录运行：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_v5_runs/latest/final \
  --output-dir checkpoints/sft_v5_runs/latest/probe_with_adapter \
  --num-routes 1 \
  --artifact-level review \
  --sample-mode random \
  --context-radius 8 \
  --with-model \
  --with-teacher-model \
  --with-teacher
```

这个命令让 student 加载训练后的 LoRA，同时让 teacher 使用未加载 LoRA 的纯 base
Qwen，和训练前能力检查保持同一输入输出合同。默认结果集中在
`scenarios/<scenario>__<route_id>/frame_*/`；每帧只写输入 RGB、`input.json`、
`output.json`、`memory.json`。加 `--artifact-level full` 后再额外生成 legacy 文件。

小样本只有三种选帧模式：

- `--sample-mode random --num-routes 1 --seed <N>`：随机完整 route ID，默认从首帧测到末帧。
- `--sample-mode rs_transition`：对每一次 RS 变化连续保留变化前帧、新 RS 首帧和变化后帧，
  用于检查 RS 识别和 memory 切换。
- `--sample-mode ue_transition`：完整保留一个连续 UE span 的全部 UE 帧，并按
  `--context-radius` 补进入前和退出后的邻帧，同时检查进入、持续和退出。

`--num-cases` 只用于 RS/UE 专项；random 不会按帧预算截断完整 ID。
`--context-radius` 控制专项边界前后的邻帧数（最少 1）。专项数据不存在时会少于
`--num-cases`，这是正常审计结果，不会用无关 RE 帧填满；UE 为避免截断真实 span，
实际帧数也允许超过 `--num-cases`。

独立 CLI 使用 `--with-teacher-model` 时显存会同时常驻 student 与 teacher 两份 Qwen；
显存不足时可去掉该参数，只运行 LoRA student。脚本化 teacher target 和场景 GT 不依赖
teacher 模型，仍会完整写入 `results.json`。

## D. 静态 prompt / target 合同快检

目的：不加载模型，只快速检查数据、候选池、teacher/student prompt 隔离、teacher target
清洗、RGB history 复制和 timeline 是否完整。

从 `AutoMoT/` 目录运行：

```bash
python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --output-dir checkpoints/sft_v5_runs/latest/probe_static \
  --num-routes 1 \
  --artifact-level full \
  --with-teacher
```

这个命令不占 GPU。它会写 teacher prompt 和脚本化 teacher target，但不会生成
`q*_student_output.txt` 或 `q*_teacher_output.txt` 的模型文本。

## 输出结构

默认 `--artifact-level review` 的人工检查结构：

```text
probe*/
  results.json
  scenarios/
    <scenario>__<route_id>/
      frame_<frame_id>/
        input_rgb_00.jpg
        input_rgb_01.jpg
        input_rgb_02.jpg
        input_rgb_03.jpg
        input.json
        output.json
        memory.json
```

先选测试场景，再进入连续 frame。各文件职责如下：

- `input_rgb_*.jpg`：该帧模型实际读取的 RGB history 原始字节。
- `input.json`：Q1 student/teacher messages，以及标明 KV 续接来源的 Q2 user turn。
- `output.json`：学生/老师完整 CoT、解析结构、teacher target、RS/EVENT 真值和正确性。
- `memory.json`：Q1 输入→学生输出、Q2 输入→学生输出、下一帧 student memory 与
  reference truth memory。reference 只作对比，`forced_correction_applied=false`。

根目录 `results.json.memory_recovery_report` 统计 RS/UE 变化后学生首次自行与 reference
对齐的 `recovery_delay_frames`；窗口内未改对时记录 `recovered=false`。

`--artifact-level compact` 才只写顶层 `results.json`，用于机器汇总，不作为默认人工入口。

只有显式传 `--artifact-level full` 才额外展开以下深度审计结构：

```text
probe*/
  results.json
  selection_plan.json
  manifest.json
  summary.json
  scenarios/
    <scenario>__<route_id>/
      timeline.json
      timeline.png
      frame_<frame_id>/
        # 上述 review 文件仍保留，并额外增加以下 legacy 文件：
      case_record.json
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
      q2_teacher_training_messages.json
      q2_teacher_training_prompt.txt
      q2_teacher_training_target.txt
      q2_teacher_model_messages.json
      q2_teacher_model_prompt.txt
      q2_teacher_model_target.txt
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

full 模式中：

- `selection_plan.json` 记录选帧模式、边界半径、seed、每个 case 的选择原因和类别计数。
- `case_record.json` 是单 case 完整入口，集中保存 RGB 来源、实际 system/user messages、
  teacher target、student/teacher 原始与解析输出、memory 和 flags。
- `summary.json` 汇总当前版本的 Q1/Q2 student 与 base teacher 指标，包括 teacher Q2
  trigger rate；训练自动 probe 还会把各版本摘要写进 run 级
  `probes/comparison.json`。
- `q1_*` 是 v5 原生命名，对应第一问。
- `q2_*` 是 v5 原生命名，对应第二问。
- `q2_teacher_prompt/target/messages` 与 `q2_teacher_output.txt` 实际配对；
  `q2_teacher_training_*` 是基于 student Q1 rollout 的 OPSD 训练输入，不能拿它解释
  teacher 自主输出。
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

## E. 大样本：完整统计与假阳性指标

小样本 probe 用于人工看完整证据，不能代替总体统计。正式 adapter 评估使用：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/eval.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_v5_runs/latest/final \
  --output-json checkpoints/sft_v5_runs/latest/eval_metrics.json \
  --output-jsonl checkpoints/sft_v5_runs/latest/eval_frames.jsonl \
  --transition-jsonl checkpoints/sft_v5_runs/latest/eval_transitions.jsonl
```

`eval_metrics.json` 保存总体、边界、分 RS、分 EVENT、混淆矩阵和 route macro 指标；
`eval_frames.jsonl` 可选保存每帧完整输入输出，用于反查 FP/FN。聚合本身是流式的，
不传 `--output-jsonl` 时不会把全量 prompt/output 留在内存。
`eval_transitions.jsonl` 只保存真实/预测的 RS 变化、UE 进入/退出及 FP/FN，
不包含大段 prompt，适合快速人工审计变化帧。
正式 eval 默认 Q1/Q2 均为 1024 token 安全上限，与训练 rollout 对齐；自动 checkpoint
probe 的 256/192 只是小样本可视化上限，不应替代正式指标。

指标方向：`rs_acc`、`rs_transition_acc`、RS 变化检测、UE 进入/退出检测、
`abnormal_acc`、各类 precision/recall/F1、
`event_acc_when_rs_correct`、`ue_acc/re_acc`、端到端准确率/召回率均越高越好；
`abnormal_false_positive_rate`、`abnormal_false_negative_rate`、
`q2_false_positive_rate`、`q2_false_negative_rate`、
`event_end_to_end_false_positive_rate`、非法输出率和每百帧 reset 次数均越低越好；
`rs_change_false_positive_rate`、`ue_entry_false_positive_rate`、
`ue_exit_false_positive_rate` 也均越低越好；
`q2_trigger_rate`、`q2_skip_due_rs_rate` 与样本数只作门控诊断。每个字段的完整中文定义和方向同时内嵌在
`eval_metrics.json.metric_definitions`，以代码输出为最终口径。

其中 `abnormal_*` 只是旧报告 schema 的兼容名称，值由 EVENT_FAST 的 RE/UE family
派生，不代表 prompt、memory 或模型输出里仍有独立 `ABNORMAL` 状态。

变化指标保存在 `results.json.summary`，自主纠正延迟保存在
`results.json.memory_recovery_report`；full 模式另写完整 `transition_report.json`。
`random` 默认抽 1 条完整 route ID 并测试全部帧；要确保命中变化边界，使用 `rs_transition` 或
`ue_transition`。全量 eval 按 validation route 的所有连续帧计算。

## Timeline 颜色

- 红色：RS_SLOW 的 RS 错误；本帧跳过 EVENT_FAST，下一帧仍沿用学生 memory
  并再次运行 RS_SLOW，
  测试不做 GT 纠错。
- 蓝色：RS gate 正确，本帧进入 EVENT_FAST；可以是慢帧新 RS，也可以是快帧复用 RS。
- 绿色：未加载 student 模型的静态 teacher-forced dump。
- 灰色：没有特别转折的普通帧。

## 人工检查清单

- `input.json` 的 student prompt 不应包含 `XML_WEATHER`、`ANSWER_`、`REFERENCE`、GT
  label 或 scenario name；teacher prompt 可以包含私有参考。
- 慢帧 `output.json.student/teacher` 应包含 RS 三段式 CoT 和 EVENT 三段式 CoT；
  快帧 Q1 为空，EVENT 仍必须包含三段分析与最终 `EVENT`。
- `output.json.teacher_targets` 必须是学生视角文本，不能泄漏私有字段名。
- `memory.json.q1/q2` 应能看到两问各自的 student input、student output 和 reference；
  `reference_is_comparison_only=true`、`forced_correction_applied=false`。
- RS_SLOW 错误时本帧 EVENT_FAST 应跳过，但 `next_frame.student` 仍保留学生结果；
  后续改对必须来自
  新一帧学生输出。Q2 非法输出不覆盖已有 student EVENT。
