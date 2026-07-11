# SFT v5 可视化记录

这个文件放在 `AutoMoT/checkpoints/` 下，用来记录 SFT v5 教师/学生输入输出的
可视化方法。它只是轻量 markdown 说明，不保存模型权重，也不保存实际 probe case；
真实的 probe 产物应放在具体 run 目录下面，例如
`checkpoints/sft_v5_runs/latest/probe/`。

## 用途

SFT v5 每帧分成两个问题：

- Q1：判断当前道路结构 RS，以及当前是否发生或处在异常事件中。
- Q2：在 Q1 的道路结构基础上，从当前帧候选里判断 EVENT。

可视化流程会把这些内容落盘：

- 学生实际看到的 prompt。
- privileged teacher prompt 里包含的私有参考信息。
- 清洗成学生视角后的 teacher target。
- 真实输入 Qwen 的 4 帧 RGB history 副本。
- 标签、候选池、memory 前后状态。
- route 级时间线 `timeline.json/png`。

## 静态检查教师和学生输入

从 `AutoMoT/` 目录运行：

```bash
python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --output-dir checkpoints/sft_v5_runs/latest/probe \
  --num-cases 24 \
  --with-teacher
```

这个命令不加载 Qwen，不占用 GPU。它用于快速检查：

- Q1/Q2 student prompt 是否干净。
- teacher prompt 是否只在 teacher 侧包含 XML weather 和答案字段。
- teacher target 是否已经清洗成学生视角。
- Q2 的 `RE` 是否带上当前帧 `regular_event_codes` 的自然语言解释。
- RGB、label、memory、flags、timeline 是否完整。

## 生成学生输出并可视化

从 `AutoMoT/` 目录运行：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_v5_runs/latest/final \
  --output-dir checkpoints/sft_v5_runs/latest/probe_with_model \
  --num-cases 8 \
  --with-model \
  --with-teacher
```

`--with-model` 会加载 student adapter，并填充：

- `q1_student_output.txt`
- `q2_student_output.txt`

`--with-teacher` 是为了和 v3 probe 命令习惯兼容。v5 probe 始终会写 teacher
privileged prompt 和脚本化 teacher target，但不会因为这个参数额外加载第二份
teacher Qwen。

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
      q2_student_prompt.txt
      q2_student_output.txt
      q2_teacher_prompt.txt
      q2_teacher_target.txt
      step1_user.txt
      step1_student.txt
      step1_teacher_user.txt
      step1_teacher.txt
      step2_user.txt
      step2_student.txt
      step2_teacher_user.txt
      step2_teacher.txt
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
- `flags.json` 保存解析出的学生输出、是否 RS 正确、是否进入 Q2、是否 candidate
  mismatch、是否 reset 下一帧等诊断字段。

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
- `q2_student_prompt.txt` 应该显示 `RE` 加当前帧允许的 `U-E*` 候选；`RE` 文案里
  应覆盖当前帧 `regular_event_codes` 对应的 regular 行为。
- `memory_before.json` / `memory_after.json` 应符合 v5 状态机：
  Q1 RS 错误时跳过本帧 Q2，并在下一帧恢复 `GT RS + RE`；Q2 非法输出不污染 memory。
