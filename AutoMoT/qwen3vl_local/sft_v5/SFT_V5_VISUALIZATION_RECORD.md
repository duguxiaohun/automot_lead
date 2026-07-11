# SFT v5 可视化记录

这个文件记录 SFT v5 教师/学生输入输出的可视化方法。它只保存方法说明，
不保存真实 probe case；真实产物建议放在具体 run 目录下面，例如
`checkpoints/sft_v5_runs/latest/probe_*`。

SFT v5 每帧分成两个问题：

- Q1：判断当前道路结构 `RS`，以及当前是否发生或处在异常事件中。
- Q2：在 Q1 的道路结构基础上，从当前帧候选里判断 `EVENT`。

这里需要区分三类检查，它们目的不同，不应该混在一起看。

## A. 训练前：默认 Qwen 的 OPSD 能力与 prompt 检查

目的：在真正开始 OPSD / DDP 训练前，先用默认 `Qwen3-VL-4B-Instruct`
同时跑学生 prompt 和 privileged teacher prompt。这个检查必须是纯 base Qwen，
不导入任何 LoRA，也不要传 `--adapter-dir`，用来确认：

- 默认 Qwen 作为 student 时，是否能理解 Q1/Q2 选择题格式、RS 选项和 EVENT 选项。
- 默认 Qwen 作为 teacher 时，吃到 XML weather、GT RS、GT event 等私有参考后，
  是否能给出稳定、合理、可被解析的 teacher 分析与答案。
- prompt 是否诱导模型泄漏私有字段、复读候选、漏掉 `ANALYSIS/RS/ABNORMAL/EVENT`
  等关键输出字段。
- Q2 的 `RE` 文案和当前帧 `U-E*` 候选是否足够清晰。

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
- `flags.json` 里的 `parsed_teacher_q1`、`parsed_teacher_q2`、
  `q1_teacher_rs_correct`、`q1_teacher_abnormal_correct`、`q2_teacher_event_correct`。

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
- `flags.json` 里 teacher/student 解析字段是否为空；为空说明 prompt 或解析合同要先修。
- `flags.json` 里的 `student_adapter_dir` 必须为空；否则说明训练前体检误加载了 LoRA，
  需要重跑纯 base Qwen 检查。

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
      q1_student_prompt.txt
      q1_student_output.txt
      q1_teacher_prompt.txt
      q1_teacher_target.txt
      q1_teacher_output.txt
      q2_student_prompt.txt
      q2_student_output.txt
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
- `step1_*` / `step2_*` 是仿 v3 的别名，方便用同一套人工检查习惯对比 v3/v5。
- `rgb_00.jpg` 到 `rgb_03.jpg` 是真实输入 Qwen 的 4 帧 stitched RGB history。
- `labels.json` 保存 RS/EVENT 标签、候选池、`event_code`、`regular_event_codes`
  和 teacher-only weather 文本。
- `memory_before.json` / `memory_after.json` 保存该帧前后的 `RS + EVENT` memory。
- `flags.json` 保存解析出的学生输出、teacher 输出、是否 RS 正确、是否进入 Q2、
  是否 candidate mismatch、是否 reset 下一帧、是否误传 `student_adapter_dir`
  等诊断字段。

## Timeline 颜色

- 红色：Q1 的 RS 错误，下一帧会恢复 `GT RS + RE`。
- 蓝色：Q1 的 RS 正确，本帧进入 Q2。
- 绿色：未加载 student 模型的静态 teacher-forced dump。
- 灰色：没有特别转折的普通帧。

## 人工检查清单

- `q1_student_prompt.txt` / `q2_student_prompt.txt` 不应包含 `XML_WEATHER`、
  `ANSWER_`、`REFERENCE`、GT label 或 scenario name。
- `q1_teacher_prompt.txt` / `q2_teacher_prompt.txt` 可以包含 teacher 私有参考信息。
- `q1_teacher_target.txt` / `q2_teacher_target.txt` 必须是学生视角文本，不能泄漏
  `ANSWER_`、`REFERENCE`、`XML_WEATHER` 这类私有字段名。
- `q1_teacher_output.txt` / `q2_teacher_output.txt` 是模型生成文本，只在
  `--with-teacher-model` 时非空，用来评估默认 Qwen 老师能力和 prompt 合理性。
- `q2_student_prompt.txt` 应该显示 `RE` 加当前帧允许的 `U-E*` 候选；`RE` 文案里
  应覆盖当前帧 `regular_event_codes` 对应的 regular 行为。
- `memory_before.json` / `memory_after.json` 应符合 v5 状态机：
  Q1 RS 错误时跳过本帧 Q2，并在下一帧恢复 `GT RS + RE`；Q2 非法输出不污染 memory。
