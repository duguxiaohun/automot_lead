# SFT v5 运行手册

SFT v5 是 RS / EVENT 两问串行 OPSD。本文只保留实际命令、关键参数和排障方法；
设计与标签规则见 `SFT_V5_PLAN.md`，完整可视化产物说明见
`SFT_V5_VISUALIZATION_RECORD.md`。以下命令默认从 `AutoMoT/` 目录运行。

## 1. 数据与静态检查

构建全量 route sequence index：

```bash
python qwen3vl_local/sft_v5/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_v5_data \
  --val-ratio 0.1 \
  --seed 42
```

构建时跳过 `noScenarios_result.json`、异常时长 route、失败结果，以及缺 XML、RGB、
meta 或逐帧 annotation 的 route；`review_required=true` 正常保留。Q2 优先使用逐帧
`frame_event_annotation.allowed_events`，缺失时才使用静态候选表。

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

代码和合同检查：

```bash
python -m py_compile qwen3vl_local/sft_v5/*.py
python qwen3vl_local/sft_v5/check_loss_mask.py
python qwen3vl_local/sft_v5/test_memory_update.py
python qwen3vl_local/sft_v5/test_dataset_contract.py
python qwen3vl_local/sft_v5/test_streaming_optimizer.py
python qwen3vl_local/sft_v5/test_parallel_kl_microbatch.py
python qwen3vl_local/sft_v5/test_checkpoint_probe.py
python qwen3vl_local/sft_v5/test_probe_selection_and_metrics.py
python qwen3vl_local/sft_v5/test_batched_kv_helpers.py
```

只检查 DataLoader、prompt、memory 和 padding，不加载 Qwen：

```bash
python qwen3vl_local/sft_v5/train.py \
  --train-index checkpoints/sft_v5_data/train_sequence_index.jsonl \
  --output-dir /tmp/sft_v5_train_check \
  --check
```

## 2. 四卡训练

正式运行：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp
```

默认配置：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `BATCH_PROFILE` | `max_util` | 使用 8 路 route/Qwen rollout |
| `PER_DEVICE_BATCH_SIZE` | `8` | 每个 rank 同时维护的 route 数 |
| `QWEN_BATCH_SIZE` | `8` | 同 timestep 的 Q1/Q2 rollout batch |
| `PARALLEL_KL_MICROBATCH_SIZE` | `2` | 带反传的 KL 微批大小 |
| `UPDATE_MODE` | `streaming_frames` | 按全局有效帧流式更新 |
| `TARGET_GLOBAL_FRAMES_PER_STEP` | `512` | 常规 optimizer step 帧阈值 |
| `MAX_TIMESTEPS_PER_STEP` | `32` | 最迟更新的 timestep 阈值 |
| `LEARNING_RATE` | `1e-5` | LoRA 学习率 |
| `MAX_NEW_TOKENS_Q1/Q2` | `1024/1024` | 无字段早停时的安全上限 |
| `RS_ERROR_PATIENCE` / `EVENT_ERROR_PATIENCE` | `4/3` | 连续错误后才申请 GT 兜底；错误期间仍持续自主思考 |
| `RS_REPAIR_INTERVAL` / `EVENT_REPAIR_INTERVAL` | `2/1` | 只控制脚本兜底检查；不降低 Q1 的逐帧运行频率 |
| `RS_MEMORY_CORRUPT_PROB/RS_MEMORY_UNKNOWN_PROB` | `0.06/0.02` | 正确 RS memory 的 wrong/UNKNOWN 扰动，总条件概率 8% |
| `EVENT_MEMORY_CORRUPT_PROB/EVENT_MEMORY_UNKNOWN_PROB` | `0.10/0.05` | 正确 EVENT memory 的 wrong/UNKNOWN 扰动，总条件概率 15% |
| `RS_INITIAL_GT_PROB/EVENT_INITIAL_GT_PROB` | `0.5/0.5` | route 首帧其余样本使用 UNKNOWN/no-prior |
| `SAVE_STEPS` | `40` | 约半天保存一次 checkpoint |

v5 在每张卡上同步边采样边训练：student 先自由生成 Q1/Q2，再由关闭 LoRA 的
privileged teacher 对相同 token span 提供 forward-KL。**每个有效帧都运行 Q1**；
Q1 RS 错误只跳过当前帧 Q2，下一帧仍继续回答 RS，直到学生自行答对或 delayed repair
真正执行。EVENT 只有在本帧 RS 正确、实际进入 Q2 后才产生 rollout/loss 并累计自己的
error streak。Q1 `ABNORMAL=NO` 不再脚本化覆盖 EVENT，必须由 Q2 自己选 RE。
模型不包 DDP wrapper，只在 optimizer step 前手动 all-reduce LoRA 梯度。
EVENT wrong memory 优先从本帧 Q2 可见候选中选其它事件；只有单选题没有替代项时才
使用全局 EVENT 作为 stale hypothesis。EVENT repair/augmentation 还要求本帧 RS
memory 已对齐；RS 已错误/UNKNOWN 时保留 EVENT 状态，避免统计学生没看到的增强样本。

当前 42 个有效场景 collection 文件含 7241 条 success route、914466 个标注帧；默认
10% route-level validation 后，训练规模约 82.3 万帧，最终精确值以远端构建出的
`checkpoints/sft_v5_data/summary.json` 为准。原始 GT 中 UE 为 142180 帧（15.55%），
RE 为 772286 帧（84.45%）；这和下面人为注入的“错误 memory 异常”是两个概念。
在模型能当帧纠偏时，预计 RS 异常输入约
8.4%（约 6.9 万帧，含首帧 UNKNOWN），EVENT 异常输入约为实际 Q2 帧的 15%；如果
模型持续复制错误 memory 直到脚本兜底，按当前平均约 126 帧/route 的模拟上界约为
RS 27.5%、EVENT/Q2 31.7%。

只检查 launcher 参数：

```bash
DRY_RUN=1 GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp
```

显存不稳时先降低并行路数，不缩短 1024 token 上限：

```bash
BATCH_PROFILE=balanced GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp  # 6 路
BATCH_PROFILE=debug GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp     # 4 路
```

若 OOM 只发生在 parallel KL，单独缩小 KL 微批：

```bash
PARALLEL_KL_MICROBATCH_SIZE=1 GPU_IDS=0,1,2,3 \
bash qwen3vl_local/sft_v5/train.sh ddp
```

单卡和静态 launcher：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_v5/train.sh single
GPU_IDS=0 bash qwen3vl_local/sft_v5/train.sh check
```

启动日志第一条 `[batch-start]` 应显示 `routes=8`、`qwen_batch=8`。GPU 利用率以一段
时间的平均值判断；自回归 decode、CPU/IO 和 rank 等待会让瞬时值波动。

## 3. 保存与自动 Probe

当前四卡实测约 80 optimizer steps/day，默认每 40 step 保存：

```text
checkpoint-40/
checkpoint-80/
final/
```

不足 40 step 的正常训练仍会保存 `final/`。每次 run 还会生成：

```text
probes/base/
probes/checkpoint-000040/
probes/final/
probes/comparison.json
```

自动 probe 复用 rank0 已加载的模型，不额外加载第二份 Qwen。默认用固定 seed 的
`random` 模式选择 1 条完整 route ID，从首帧逐步测试到末帧，保证
base/checkpoint/final 始终对比同一 ID。默认 `review` 每帧只保存输入 RGB、
`input.json`、`output.json` 和 `memory.json`。
自动 probe 使用 256/192 token 旁路上限，不影响训练的 1024/1024。

常用覆盖：

```bash
# 关闭自动 probe，仍按 40 step 保存。
CHECKPOINT_PROBE=0 SAVE_STEPS=40 GPU_IDS=0,1,2,3 \
bash qwen3vl_local/sft_v5/train.sh ddp

# 改成 UE 进入/退出专项小样本。
CHECKPOINT_PROBE_SAMPLE_MODE=ue_transition CHECKPOINT_PROBE_NUM_CASES=24 \
CHECKPOINT_PROBE_CONTEXT_RADIUS=8 GPU_IDS=0,1,2,3 \
bash qwen3vl_local/sft_v5/train.sh ddp
```

## 4. TensorBoard 与日志

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_v5_runs/latest/tb
```

优先查看：

| Tag | 用途 |
|---|---|
| `train/loss_frame` | 全局帧平均总 loss |
| `train/loss/{q1_analysis,q1_rs,q1_abnormal,q2_analysis,q2_event}` | 各监督部分 loss |
| `train/q1_rs_acc_window` / `train/q2_event_acc_window` | 当前窗口解析准确率 |
| `train/q2_skip_due_rs_rate` | 因本帧 RS 错而跳过 Q2 的比例；应与 RS 错误率一致 |
| `train/abnormal_{precision,recall,f1}` | Q1 UE/RE 混淆指标 |
| `train/q2_ue_{precision,recall,f1}` | Q2 UE/RE 混淆指标 |
| `memory/rs_wrong_copy_rate` / `memory/event_wrong_copy_rate` | 已知错误 memory 被原样复制的比例，越低越好 |
| `memory/rs_recovery_rate` / `memory/event_recovery_rate` | wrong/UNKNOWN memory 的自主纠偏率 |
| `memory/rs_input_anomaly_rate` / `memory/event_input_anomaly_rate` | 实际进入 prompt 的 wrong+UNKNOWN 比例；EVENT 分母是 Q2 帧 |
| `memory/rs_error_streak_mean` / `memory/event_error_streak_mean` | 延迟纠偏持续长度；用于判断 patience 是否过长 |
| `memory/{rs,event}_{injected_wrong,injected_unknown,forced_repair}` | 课程扰动与兜底修复实际样本数 |
| `train/q1_token_cap_hit_rate` / `train/q2_token_cap_hit_rate` | 是否经常打满 1024 |
| `qwen/q1_batched_frame_rate` | Q1 真正进入 batch rollout 的帧比例 |
| `parallel_kl/frame_rate` | 走并行 KL 的帧比例 |
| `parallel_kl/oom_splits` | KL 微批自动二分次数 |
| `ddp/padding_rate` | global timestep padding 比例 |
| `memory/allocated_gb` | 活跃 tensor/计算图显存，判断泄漏的主指标 |
| `memory/reserved_gb` | allocator 缓存高水位，单独升高不等于泄漏 |

首个 optimizer step 前可看 `run/*` 与 `progress/*`。若 events 文件只有 header 且没有
这些 tag，通常是旧 run 或 writer 尚未创建。

排查卡顿时提高心跳密度：

```bash
PROGRESS_FRAMES=1 HEARTBEAT_SECONDS=60 GPU_IDS=0,1,2,3 \
bash qwen3vl_local/sft_v5/train.sh ddp
```

- 停在 `[frame-start]`：检查图像 IO 或单帧生成。
- 停在 `[batch-local-done]`：检查其它 rank 是否仍在处理长 route。
- 停在 `[sync-start]`：检查 NCCL 和梯度 all-reduce。

## 5. 训练后大样本 Eval

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/eval.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_v5_runs/latest/final \
  --output-json checkpoints/sft_v5_runs/latest/eval_metrics.json \
  --output-jsonl checkpoints/sft_v5_runs/latest/eval_frames.jsonl \
  --transition-jsonl checkpoints/sft_v5_runs/latest/eval_transitions.jsonl
```

`eval_metrics.json` 是流式聚合结果；`eval_frames.jsonl` 可选保存逐帧 RGB 路径、prompt、
原始输出、解析、memory 和 GT/prediction，便于回查 FP/FN。不需要逐帧证据时移除
`--output-jsonl`，可减少磁盘写入。`eval_transitions.jsonl` 更小，只保存真实或预测的
RS 变化、UE 进入/退出帧以及 FP/FN，每行直接给出 `TP/FP/FN/TN/invalid`。

核心指标：

| 指标 | 含义 | 方向 |
|---|---|---|
| `rs_acc` / `rs_transition_acc` | 全帧 RS / RS 变化首帧准确率 | 越高越好 |
| `rs_change_detection_precision/recall/f1` | 模型是否真正在正确帧切换 RS | 越高越好 |
| `rs_change_false_positive_rate` | 真值 RS 稳定时模型误切换的比例 | 越低越好 |
| `ue_entry_detection_precision/recall/f1` | RE->UE 进入帧检测 | 越高越好 |
| `ue_exit_detection_precision/recall/f1` | UE->RE 退出帧检测 | 越高越好 |
| `ue_entry_false_positive_rate` / `ue_exit_false_positive_rate` | UE 进入/退出边界误报率 | 越低越好 |
| `abnormal_precision/recall/f1` | Q1 对 UE 的查准率、召回率和 F1 | 越高越好 |
| `abnormal_false_positive_rate` | 真实 RE 被 Q1 错报为异常的比例 | 越低越好 |
| `abnormal_false_negative_rate` | 真实 UE 未被 Q1 正确报异常的比例 | 越低越好 |
| `event_acc_when_rs_correct` | 进入 Q2 后的具体 EVENT 准确率 | 越高越好 |
| `q2_ue_precision/recall/f1` | Q2 的 UE/RE 二分类指标 | 越高越好 |
| `q2_trigger_rate` / `q2_skip_due_rs_rate` | RS 正确进入 Q2 / RS 错误跳过 Q2 的互补覆盖率 | 诊断 |
| `q2_false_positive_rate` / `q2_false_negative_rate` | Q2 的 UE 假阳性/假阴性率 | 越低越好 |
| `event_end_to_end_acc` / `ue_end_to_end_recall` | 包含 Q1 门控失败的端到端指标 | 越高越好 |
| `event_end_to_end_false_positive_rate` | 所有真实 RE 中最终误报 UE 的比例 | 越低越好 |
| `rs_wrong_memory_copy_rate` / `event_wrong_memory_copy_rate` | 错误 memory 被直接照抄的比例 | 越低越好 |
| `rs_wrong_or_unknown_memory_recovery_rate` / `event_wrong_or_unknown_memory_recovery_rate` | 错误或无先验输入上的自主恢复率 | 越高越好 |
| `mean_resets_per_100_frames` | 测试中每百帧真值强制纠错次数；学生闭环应为 0 | 越低越好 |
| `mean_training_reset_recommendations_per_100_frames` | 兼容旧报告的即时失败信号（Q1 RS 错或 Q2 非法）；不代表当前延迟修复实际执行次数 | 仅诊断 |
| `q2_trigger_rate` / 样本数 | 门控覆盖与评估规模 | 仅诊断 |

完整定义与方向保存在 `eval_metrics.json.metric_definitions`。分母无样本时写 `null`。

## 6. 小样本 Probe

`build_dataset.py` 只负责把所有合法 route 建成连续序列索引，不执行模型测试。小样本
连续片段检查使用 `probe.py`；验证集所有 route 的所有连续帧指标使用上一节的 `eval.py`。

训练前纯 base Qwen 能力检查，不加载 LoRA：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_v5_runs/pre_opsd_base_probe \
  --num-routes 1 \
  --artifact-level review \
  --with-model \
  --with-teacher-model \
  --with-teacher
```

训练后 adapter 检查：

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

这条训练后命令与训练前 base 检查使用同一批输入和 schema：student 加载 LoRA，teacher
仍是纯 base Qwen。测试窗口首帧建立共同起点，此后 student memory 只由学生 Q1/Q2
输出推进；truth memory 只在 `memory.json` 对照，绝不回写纠错。

选帧模式：

- `random`：按 `--seed` 随机选择完整 route ID，并从该 ID 首帧测试到末帧；
  `--num-routes` 控制完整 ID 数，默认 1。
- `rs_transition`：按同一次 RS 变化连续取变化前帧、新 RS 首帧和变化后帧。
- `ue_transition`：完整保留同一 UE 从首帧到末帧，再向前后各补
  `--context-radius` 个 RE/邻帧；长 UE 可以超过 `--num-cases`，不会从中间截断。

`--num-cases` 只用于 RS/UE 专项，不会截断 random 的完整 ID。`--context-radius 8`
表示专项模式保留边界前后最多 8 帧。如果所读数据中没有
对应的 RS/UE 变化，专项模式不会用无关帧凑数。RS 结果可能少于 `--num-cases`；UE
结果为保证 span 完整性也可能多于 `--num-cases`。

默认从 `scenarios/` 进入测试场景，再进入对应 `frame_*`。每帧只有：

- `input_rgb_*.jpg`：模型实际读取的连续 RGB history。
- `input.json`：Q1 student/teacher 输入及 Q2 KV 续接 user turn。
- `output.json`：学生和老师的完整输出、解析结构、teacher target、场景真值与正确性。
- `memory.json`：Q1 输入/输出、Q2 输入/输出、下一帧 student memory 与只读 truth 对照。

顶层 `results.json` 只保留指标、帧目录索引和 `memory_recovery_report`；后者统计变化后
学生首次自主改对的延迟帧数。`--artifact-level full` 才增加 legacy TXT/JSON。
手工 probe 默认 1024/1024 token；自动 checkpoint probe 才使用 256/192。

训练前 grouped/parallel 等价性检查：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/test_batched_qwen_smoke.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --num-cases 2 \
  --candidate-pool 256 \
  --require-batched-group \
  --no-prefer-different-lengths \
  --check-parallel-kl \
  --output-json checkpoints/sft_v5_runs/batched_qwen_smoke.json
```

静态 teacher 合同：

```bash
python qwen3vl_local/sft_v5/inspect_teacher.py \
  --index checkpoints/sft_v5_data/train_sequence_index.jsonl \
  --output-dir checkpoints/sft_v5_runs/latest/teacher_inspect \
  --num-cases 64
```

## 7. 高频故障

### NCCL watchdog

确认没有把 `bundle.model` 包进 DDP wrapper。v5 的 Q2 分支在不同 rank 上触发次数可能
不同，只允许 torchrun 多进程加手动 LoRA 梯度 all-reduce。

### Cache 类型错误

`AttributeError: 'tuple' object has no attribute 'get_mask_sizes'` 表示新版 Transformers
Cache 被退化成 legacy tuple。必须保持带 `get_mask_sizes/get_seq_length` 的 Cache 类型。

### CUDA OOM

先看 `parallel_kl/oom_splits` 和 traceback：KL OOM 先减
`PARALLEL_KL_MICROBATCH_SIZE`；rollout OOM 再退 `BATCH_PROFILE`。不要先缩短 Qwen 输出。
长期显存判断看 `memory/allocated_gb`，不要只看 `nvidia-smi` 或 `reserved`。

### loss 没有 grad

若出现 `loss does not require grad`，检查对应输出是否缺 `RS:`、`ABNORMAL:` 或
`EVENT:`。当前代码会用可训练参数构造 graph-connected zero，避免单个坏输出直接中断
所有 rank，但这类警告仍表示 prompt/output 需要检查。

## 8. 注释与文档维护

v5 的函数、CLI、状态机、并行 rollout、KL 微批、memory/reset、数据过滤和指标分母均
使用中文注释说明“如何调用”和“为什么这样做”。改动这些合同后，必须同步更新
`SFT_V5_PLAN.md`、本手册、`SFT_V5_VISUALIZATION_RECORD.md` 及相邻测试。
