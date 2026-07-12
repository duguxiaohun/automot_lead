# SFT v5 Runbook

SFT v5 是 RS / EVENT 两问串行的 OPSD 训练入口。数据来自
`keyframe_filter/collection_output/*_result.json`，一条样本是一条 route sequence；
训练时 student 自己维护 `RS + EVENT` memory，privileged teacher 只用来在 student
自由生成 token 上提供 forward-KL logits。

本文默认当前目录是 `AutoMoT/`。

## 1. 构建数据

```bash
python qwen3vl_local/sft_v5/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_v5_data \
  --val-ratio 0.1 \
  --seed 42
```

快速 smoke：

```bash
python qwen3vl_local/sft_v5/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir /tmp/sft_v5_smoke \
  --scenarios Accident \
  --max-routes 1 \
  --max-frames-per-route 3
```

构建阶段会跳过 `noScenarios_result.json`、异常时长 skip、数据缺失 skip、
`status != success`、缺 XML、缺 RGB、缺 meta 或缺逐帧 annotation 的 route。
`review_required=true` 仍保留训练。

Q2 候选优先使用逐帧 `frame_event_annotation.allowed_events`，旧结果缺字段时才
fallback 到 `scenario_event_candidates ∩ EVENT_CANDIDATES_BY_RS[current_rs]`。
所有 `R-E*` 会在 prompt 中折叠为一个 `RE`，原始 regular code 只保存在
`event_code` / `regular_event_codes` 用于审计。

## 2. 静态检查

```bash
python qwen3vl_local/sft_v5/check_loss_mask.py
python qwen3vl_local/sft_v5/test_memory_update.py
python qwen3vl_local/sft_v5/test_dataset_contract.py
python -m py_compile qwen3vl_local/sft_v5/*.py
```

检查 dataset / padding / prompt / memory，不加载模型：

```bash
python qwen3vl_local/sft_v5/train.py \
  --train-index checkpoints/sft_v5_data/train_sequence_index.jsonl \
  --output-dir /tmp/sft_v5_train_check \
  --check
```

## 3. 训练

### 3.0 OPSD 采样与训练关系

OPSD 不是先离线生成一批 teacher 数据再训练。v5 当前实现是同步 on-policy：

1. 每个 DDP rank 读取自己的 route sequence batch。
2. 对每个有效 frame，当前 student 先自由生成 Q1；Q1 RS 正确时再自由生成 Q2。
3. 这些 student 生成出来的 token 就是本 step 的 on-policy 采样数据，只在内存中临时保存。
4. Teacher 关闭 LoRA，读取 privileged prompt，在同一批 student token 上 forward 得到 logits。
5. Student/teacher logits 做 weighted forward-KL，然后当前 rank 反向传播，DDP 同步梯度。

因此 H20 四卡下，当前 v5 是“四张卡都边采样边训练”，不是“几张卡专门采数据、几张卡专门训练”。
这和 v4 的 off-policy collector/learner 不同。若要改成异步架构，可以参考 v4：
`3` 张 H20 跑 collector 持续写 replay，`1` 张 H20 跑 learner 训练；但那需要新增 v5
collector/replay/learner 代码，当前 `train.py` 没有这个分卡角色。

### Launcher 模式

`train.sh` 参考 v3 的运行口径，支持 `single/ddp/check` 三种模式，并会维护
`OUTPUT_DIR/run_<RUN_TAG>/` 与 `OUTPUT_DIR/latest`：

```bash
bash qwen3vl_local/sft_v5/train.sh ddp
```

显式单卡：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_v5/train.sh single
```

显式 4 卡 DDP：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp
```

自动选择 4 张空闲卡：

```bash
DDP_GPU_COUNT=4 bash qwen3vl_local/sft_v5/train.sh ddp
```

`GPU_IDS` 非空时优先使用显式卡号；否则 launcher 会用 `nvidia-smi` 按显存占用和
GPU util 选择较空闲的卡。`MASTER_PORT` 默认自动找空闲端口。`QWEN3VL_LOG_TO_FILE=1`
时 stdout/stderr 会写入当前 run 的 `log.txt`。

`check` 模式只检查 dataset / padding / prompt / memory，不加载 Qwen 权重、不保存
adapter：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_v5/train.sh check
```

通常不要绕过 `train.sh` 手写 `torchrun`；launcher 会统一处理 GPU 自动选址、
`GPU_IDS` 显式 pin、`MASTER_PORT`、日志落盘和 `run_<RUN_TAG>/latest` 防覆盖目录。

常用环境变量：

```bash
OUTPUT_DIR=checkpoints/sft_v5_runs \
RUN_TAG=debug_v5 \
NUM_EPOCHS=1 \
LR=3e-5 \
PER_DEVICE_BS=1 \
GRAD_ACCUM=1 \
MAX_ROUTES=0 \
MAX_FRAMES_PER_ROUTE=0 \
SAVE_STEPS=200 \
LOGGING_STEPS=5 \
GPU_IDS=0 \
bash qwen3vl_local/sft_v5/train.sh single
```

视觉 LoRA 默认关闭。需要打开时：

```bash
LORA_VISION_SCOPE=merger VISION_LR_SCALE=0.1 GPU_IDS=0 bash qwen3vl_local/sft_v5/train.sh single
```

关键训练口径：

- DataLoader collate 只做本 rank local padding；主训练进程再 all-reduce 得到
  当前 batch 的 global `max_T`，padding timestep 不读图、不进 Qwen、不产 loss。
- Q1 输出 `WEATHER / SCENE DESCRIPTION / CRITICAL OBJECT DESCRIPTION / REASONING /
  MEMORY JUDGMENT / RS / ABNORMAL`；天气只写在 `WEATHER` 行，没有单独天气分类 loss。
- `MEMORY` 内含 `EGO_TO_GOAL_XY`，由 build_dataset 从当前帧 meta
  `next_target_points[-1]` 转 ego frame 写入；缺失该坐标的 route 不进入新数据集。
- Student 不看 XML weather；XML weather 只进入 teacher prompt，并且 teacher target
  清洗成学生视角。若 XML weather 与 RGB 可见天气或能见度冲突，teacher 以 RGB
  证据为准。
- Q2 是 Q1 assistant 输出后的第二轮 user turn，训练、eval 和 probe 的模型路径都复用
  Q1 KV cache，不重新问一轮 fresh Q2。
- Q1 RS 错误时只结束当前帧：跳过本帧 Q2，下一有效帧恢复 `GT RS + RE`。
- Q2 非法 option 不污染 memory，下一有效帧同样恢复 `GT RS + RE`。
- Q2 候选字母每帧可复现随机，不能假设某个字母固定代表 `RE`。

## 4. TensorBoard

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_v5_runs/latest/tb
```

当前 v5 训练脚本至少写：

- `train/loss_frame`：当前 logging window 内所有 rank 聚合后的 frame 平均 loss。
- `train/q2_trigger_rate`：Q1 RS 正确后进入 Q2 的比例，也就是第二问实际采样率。
- `train/q1_rs_acc_window` / `train/q1_abnormal_acc_window`：当前窗口的 Q1 解析正确率。
- `train/q2_event_acc_window`：进入 Q2 的帧中，EVENT 解析是否命中动态真值。
- `train/q2_invalid_output` / `train/reset_next`：非法输出和下一帧 reset 次数。
- `train/rollout_tokens_per_frame`：student on-policy 采样出的 Q1+Q2 token 平均长度。
- `ddp/padding_rate`：global padding 后的 None frame 占位比例。
- `ddp/max_T_global_avg`：logging window 内 DDP 对齐后的平均 `max_T_global`。

后续如果扩展详细 loss，可沿用 v3 的命名习惯拆到
`train/loss/{q1_analysis,q1_rs,q1_abnormal,q2_analysis,q2_event}`、
`grad_norm/{language,vision}` 和 `param_norm/lora_{language,vision}`。

## 5. Eval

默认只跑学生，不加载 teacher，不做 GT 注入；memory 全程由学生 Q1/Q2 输出维护。
Q1 RS 错时跳过本帧 Q2，下一有效帧恢复 `GT RS + RE`，与训练采样口径一致。

自由生成评估：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/eval.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_v5_runs/latest/final \
  --output-json checkpoints/sft_v5_runs/latest/eval_metrics.json
```

只检查 index 可读：

```bash
python qwen3vl_local/sft_v5/eval.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --check
```

输出：

- `eval_metrics.json`

核心指标：

- `rs_acc`
- `abnormal_acc`
- `event_acc_when_rs_correct`
- `ue_acc`
- `re_acc`
- `q2_trigger_rate`
- `q2_candidate_mismatch`
- `q2_invalid_output`
- `rs_wrong_resets`

## 6. Probe / 可视化输入输出

### 6.1 方法速查

| 目的 | 命令 | 是否加载模型 | 主要产物 |
|---|---|---|---|
| 训练前 base Qwen OPSD 能力体检 | `GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py --index checkpoints/sft_v5_data/val_sequence_index.jsonl --model-dir checkpoints/Qwen3-VL-4B-Instruct --output-dir checkpoints/sft_v5_runs/pre_opsd_base_probe --num-cases 8 --with-model --with-teacher-model --with-teacher` | 是，纯默认/base Qwen，不传 `--adapter-dir`，不加载任何 LoRA | RGB 副本、system/user/messages、student output、teacher target/output、memory、flags、timeline |
| 训练后 adapter 学生可视化 | `GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py --index checkpoints/sft_v5_data/val_sequence_index.jsonl --model-dir checkpoints/Qwen3-VL-4B-Instruct --adapter-dir checkpoints/sft_v5_runs/latest/final --output-dir checkpoints/sft_v5_runs/latest/probe_with_adapter --num-cases 8 --with-model --with-teacher` | 是，只加载 student adapter | 上述静态产物 + `q1_student_output.txt` / `q2_student_output.txt` |
| 静态检查教师/学生输入合同 | `python qwen3vl_local/sft_v5/probe.py --index checkpoints/sft_v5_data/val_sequence_index.jsonl --output-dir checkpoints/sft_v5_runs/latest/probe_static --num-cases 24 --with-teacher` | 否 | RGB 副本、student prompt、teacher prompt、teacher target、memory、flags、timeline |
| 检查 teacher 合同 | `python qwen3vl_local/sft_v5/inspect_teacher.py --index checkpoints/sft_v5_data/train_sequence_index.jsonl --output-dir checkpoints/sft_v5_runs/latest/teacher_inspect --num-cases 64` | 否 | `teacher_report.json` / `teacher_report.md` |

同一份速查记录也保存在 `qwen3vl_local/sft_v5/SFT_V5_VISUALIZATION_RECORD.md`，
方便在 v5 子包内直接回看。

### 6.2 Probe 入口

`probe.py` 是 v5 的主要可视化/审计入口。它有三种用途，必须分开理解：

- 训练前 base Qwen OPSD 体检：`--with-model --with-teacher-model`，不传
  `--adapter-dir`，不加载任何 LoRA，用默认 Qwen 分别跑 student prompt 和
  privileged teacher prompt，判断模型基础能力与 prompt 合同是否足够支撑 OPSD。
- 训练后 adapter 学生可视化：`--with-model --adapter-dir ...`，只看训练出的学生在
  真实状态机下的 Q1/Q2 输出。
- 静态 prompt / target 快检：不加载模型，只 dump RGB、student prompt、teacher prompt、
  脚本化 teacher target、label、memory 和 timeline。

训练前 base Qwen OPSD 能力体检：

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

这个命令不传 `--adapter-dir`，所以 student 和 teacher 都是默认/base Qwen；
区别是 student 只看 RGB + 学生 prompt，teacher 看 RGB + privileged teacher prompt。
这一类训练前体检不要加载任何 adapter，否则就不是在测试普通 Qwen 是否足够支撑
OPSD。
如果同卡同时加载两份 Qwen 显存不够，可以拆成 student-only 的 `--with-model` 和
teacher-only 的 `--with-teacher-model` 两次跑。

训练后 adapter 学生可视化：

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

静态 prompt / target 快检：

```bash
python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --output-dir checkpoints/sft_v5_runs/latest/probe_static \
  --num-cases 24 \
  --with-teacher
```

`--with-teacher` 是 v3 兼容参数；v5 始终写 teacher privileged prompt 和脚本化
teacher target。只有显式加 `--with-teacher-model` 时，才会额外加载 base Qwen 并生成
`q1_teacher_output.txt` / `q2_teacher_output.txt`。

输出结构仿 v3 probe：

- 顶层 `manifest.json`：列出每条 route dump 的目录、scenario、route_id 和帧数。
- 每条 route 一个 `route_<idx>__<scenario>__<route_id>/` 目录。
- 每条 route 下有 `timeline.json` 和 `timeline.png`，红点表示 RS 错/reset，蓝点表示
  进入 Q2，绿点表示无模型的 teacher-forced 静态 dump。
- 每帧一个 `frame_<frame_id>/` 目录。

每个 `frame_*` 目录会保存：

- `rgb_00.jpg` / `rgb_01.jpg` / `rgb_02.jpg` / `rgb_03.jpg`：真实进入 Qwen 的 4 帧
  stitched RGB history 副本。
- `rgb_paths.json`：RGB 原路径与复制状态。

- `q1_system_prompt.txt` / `q2_system_prompt.txt`：v5 固定 system prompt。
- `q1_student_user_prompt.txt` / `q1_teacher_user_prompt.txt`：Q1 的 student / teacher
  user prompt；`q1_student_prompt.txt` / `q1_teacher_prompt.txt` 仍作为兼容别名保留。
- `q2_student_user_prompt.txt` / `q2_teacher_user_prompt.txt`：Q2 的 student / teacher
  user prompt；`q2_student_prompt.txt` / `q2_teacher_prompt.txt` 仍作为兼容别名保留。
- `q1_student_messages.json` / `q1_teacher_messages.json` / `q2_student_messages.json` /
  `q2_teacher_messages.json`：可序列化的 system + user messages，图片用 `rgb_*.jpg`
  文件名和原路径表示，方便直接检查 role 分界。
- `q1_student_prompt.txt`：Q1 学生真实输入，不含 XML weather / GT。
- `q1_teacher_prompt.txt`：Q1 privileged teacher 输入，含 XML weather、GT RS、GT abnormal、
  原始 `event_code`。
- `q1_teacher_target.txt`：脚本化学生视角 target，用于审计合同和 loss mask。
- `q1_teacher_output.txt`：只有 `--with-teacher-model` 时非空，用于训练前检查 base teacher。
- `q2_student_prompt.txt`：Q2 学生真实输入，含逐帧随机 `EVENT_CHOICES`；`RE` 会展开
  当前帧 `regular_event_codes` 的自然语言含义。
- `q2_teacher_prompt.txt`：Q2 privileged teacher 输入，含 answer event option 与
  `event_code` 审计字段。
- `q2_teacher_target.txt`：脚本化 Q2 target。
- `q2_teacher_output.txt`：只有 `--with-teacher-model` 时非空，用于训练前检查 base teacher。
- `q1_student_output.txt` / `q2_student_output.txt`：目录结构固定；只有 `--with-model` 时内容非空。
- `step1_user.txt` / `step1_student.txt` / `step1_teacher_user.txt` / `step1_teacher.txt`：
  v3 风格别名，对应 Q1。
- `step2_user.txt` / `step2_student.txt` / `step2_teacher_user.txt` / `step2_teacher.txt`：
  v3 风格别名，对应 Q2。
- `memory_before.json` / `memory_after.json`：该帧前后的 `RS + EVENT + EGO_TO_GOAL_XY`
  memory。
- `flags.json`：解析出的 student/teacher 输出、是否 RS 正确、是否进入 Q2、
  是否 candidate mismatch、是否 reset 下一帧，以及
  `q2_student_continued_from_q1_kv` / `q2_teacher_continued_from_q1_kv` 等诊断字段。
- `labels.json`：`history_rgb_paths`、`rs_label/rs_option`、`event_label/event_code`、
  `abnormal`、`event_option_map`、`frame_allowed_events_raw`、`regular_event_codes`、
  `event_candidate_codes`、`ego_to_goal_xy` 与 `weather_text_teacher_only`。

### 6.3 Teacher 合同抽检

Teacher 合同抽检：

```bash
python qwen3vl_local/sft_v5/inspect_teacher.py \
  --index checkpoints/sft_v5_data/train_sequence_index.jsonl \
  --output-dir checkpoints/sft_v5_runs/latest/teacher_inspect \
  --num-cases 64
```

`inspect_teacher.py` 会写：

- `teacher_report.json`
- `teacher_report.md`

检查项包括：

- XML weather 只出现在 teacher prompt，不进入 student prompt。
- teacher target 不泄漏 `ANSWER_` / `REFERENCE` / `XML_WEATHER` 等私有标记。
- Q2 option map 非空。
- Q2 student prompt 不泄漏 scenario name。

## 7. 代码注释维护要求

`sft_v5` 代码已经按中文注释口径补充：

- 函数/docstring 说明入口职责。
- 关键逻辑块说明“为什么这样做”，包括逐帧 `allowed_events` 优先、`R-E* -> RE`
  折叠、双标签单标签化、Q1 RS 错误截断、DDP global padding、OPSD teacher/student
  logits 对齐，以及训练前纯 base Qwen 体检不加载 LoRA。
- 测试脚本说明各自防止哪类回归。

后续改 prompt、候选池、memory、训练状态机、probe 输出或 DDP padding 时，需要同步更新
对应代码块注释和本运行文档，避免实现和人工检查口径漂移。
