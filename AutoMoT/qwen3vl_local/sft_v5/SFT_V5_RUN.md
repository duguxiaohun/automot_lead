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
| `RS_SLOW_INTERVAL` / `RS_SLOW_INTERVAL_JITTER` | `4/1` | RS 稳定正确时从 3/4/5 个 4Hz frame 中可复现抽取下一次慢思考间隔；jitter=0 为固定周期 |
| `RS_ERROR_PATIENCE` / `EVENT_ERROR_PATIENCE` | `4/3` | 连续错误后才申请 GT 兜底；错误期间仍持续自主思考 |
| `RS_REPAIR_INTERVAL` / `EVENT_REPAIR_INTERVAL` | `2/1` | 只控制 pending 后的脚本兜底检查，与慢思考频率独立 |
| `RS_REPAIR_MODE` / `EVENT_REPAIR_MODE` | `ground_truth/ground_truth` | patience+review 后才延迟写回 GT；`unknown` 仅用于软擦除消融 |
| `RS_MEMORY_CORRUPT_PROB/RS_MEMORY_UNKNOWN_PROB` | `0.05/0.07` | 正确 RS memory 的 contradiction/omission 条件概率；最终 Q1 比例受额外慢问影响 |
| `EVENT_MEMORY_CORRUPT_PROB/EVENT_MEMORY_UNKNOWN_PROB` | `0.20/0.12` | eligible EVENT memory 的额外 contradiction/omission 条件概率；RS 变化还会自然失效 EVENT，合并后把 Q2 校准到约 60/23/17 |
| `RS_INITIAL_GT_PROB/EVENT_INITIAL_GT_PROB` | `0.5/0.5` | route 首帧其余样本使用 UNKNOWN/no-prior |
| `SAVE_STEPS` | `40` | 约半天保存一次 checkpoint |

当前 prompt 合同为 `sft_v5_compact_prompt_v1`：system 只保留跨问题通用证据原则，
Q1/Q2 user 只保留短 memory、短候选、本题一句任务和四行输出模板。代表性 R1/RE
二选一输入约为 system 64 words、Q1 141 words、Q2 156 words；版本会写入 adapter、
eval 和 probe summary。若现有 adapter 由旧长 prompt 训练，严格版本对比应重训。

v5 在每张卡上同步边采样边训练：student 先自由生成当前需要的 RS_SLOW /
EVENT_FAST，再由关闭 LoRA 的 privileged teacher 对相同 token span 提供 forward-KL。
RS 稳定正确时，快帧直接复用 RS memory，不采集也不训练 RS；但 EVENT_FAST 每个
RS gate 正确的帧都重新读本帧 RGB，保留三段分析，并直接在混合标注的
`[RE | REGULAR]` / `[UE | UNUSUAL]` 选项中选 EVENT，没有独立 normal/abnormal
问题。RS 一旦答错就跳过
当帧 EVENT，下一帧恢复逐帧 RS 慢思考，直到自我修正或 delayed repair 执行。
正式 delayed repair 不是错误后下一帧就改成答案：RS 至少要经历 4 次
连续错误并等到 2 帧 review slot，EVENT 至少 3 次实际 Q2 错误。修复帧答对
只记为 `recovered_after_forced_repair`，不冒充 `self_recovered_after_streak`。
换句话说，旧概念的 `EVENT_FAST_1` / `EVENT_FAST_2` 已合并为单个 `EVENT_FAST`。
模型不包 DDP wrapper，只在 optimizer step 前手动 all-reduce LoRA 梯度。
EVENT wrong memory 优先从本帧 Q2 可见候选中选其它事件；只有单选题没有替代项时才
使用全局 EVENT 作为 stale hypothesis。EVENT repair/augmentation 还要求本帧 RS
memory 已对齐。EVENT 是条件状态 `EVENT | RS`：RS hypothesis 真正变化时，旧 EVENT
立即失效为 UNKNOWN/age=0，并清空旧语境的 EVENT streak/pending；若 RS 没有再次变化，
则保持当前 EVENT/age，不会每帧反复重置。
“没有 memory”统一表示为固定 `[MEMORY]` schema 内的 UNKNOWN/no-prior，不整块删除
prompt。RS/EVENT 各自带 `*_HYPOTHESIS_AGE`：普通帧独立增加，对应 label 改变时归零；
此外 RS 改变会使条件 EVENT 一起归零。周期核验同一 RS 不会把任一 age 清零。
新注入的 wrong/UNKNOWN 因为刚刚改变了 hypothesis，age 也从 0 开始；只有模型后续
继续复制它，才会自然得到 age>0 的 stale-memory 训练轨迹。不要在离线 index 中人为
随机填写较大的 age，也不要把 age 当成“越大越一定错误”的硬标签：稳定 RS 可以长期正确，
EVENT 才通常衰减得更快，两者最终都必须由当前 RGB 复核。

RS/EVENT 的输出 parser 始终只接受本帧选项字母，`RS: R4`、`EVENT: RE` 等语义标签
仍记为 invalid，不能写入 memory。为了避免这类最需要纠正的 rollout 只剩低权重分析
loss，训练会把答案值的第一个生成 token 纳入高权重 teacher-KL；privileged teacher
在冒号后的起始位置推动合法选项字母。若整行 `RS:`/`EVENT:` 都缺失，则不猜测离散
监督位置，只保留实际存在的分析 span。

当前 42 个有效场景 collection 文件含 7241 条 success route、914466 个标注帧；默认
10% route-level validation 后，训练规模约 82.3 万帧，最终精确值以远端构建出的
`checkpoints/sft_v5_data/summary.json` 为准。原始 GT 中 UE 为 142180 帧（15.55%），
RE 为 772286 帧（84.45%）；这和下面人为注入的“错误 memory 异常”是两个概念。
默认稳定间隔在 3/4/5 帧中随机，平均仍是 4 帧。按 3000 条、每条 126 帧的恒定 GT
序列模拟：模型若能当帧纠偏，Q1 触发约 30.5%（约 25.1 万帧），其
`aligned/omission/contradiction` 约为 `59.7/24.2/16.1`；Q2 gate 约 100%，关系约为
`59.6/23.0/17.4`。若模型只复制 memory 直到 delayed repair，压力测试中 Q1 触发约
55.5%，Q2 gate 约 64.0%，Q2 关系约 `38.6/43.5/17.9`。这些是策略仿真，不替代
TensorBoard 的实际关系比例与 gate 统计。
上述上界依赖正式默认的延迟 GT 兜底。若显式改成 `unknown` 软擦除，
“学生只复制 memory”压力测试中 RS anomaly 会达约 95.7%、Q2 gate 只剩约
4.3%，因此不建议用作长训默认。

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
| `train/loss/{q1_analysis,q1_rs,q2_analysis,q2_event}` | 各监督部分 loss |
| `train/q1_rs_acc_window` / `train/q2_event_acc_window` | 当前窗口解析准确率 |
| `train/rs_slow_trigger_rate` / `train/rs_reuse_fast_rate` | RS 慢思考触发率 / 快帧复用率 |
| `train/q2_skip_due_rs_rate` | 因本帧 RS 错而跳过 Q2 的比例；应与 RS 错误率一致 |
| `train/abnormal_{precision,recall,f1}` / `train/q2_ue_{precision,recall,f1}` | EVENT_FAST 选项折成 UE/RE 的同口径混淆指标 |
| `memory/q1_relation_{aligned,omission,contradiction}_rate` | 真正进入 Q1 的三类 memory 关系；三项应约等于 1 |
| `memory/q2_relation_{aligned,omission,contradiction}_rate` | 真正进入 Q2 的三类 memory 关系；受 RS gate 影响 |
| `memory/q1_rs_age_frames_mean` / `memory/q2_event_age_frames_mean` | 两个 hypothesis 在实际监督 prompt 中的平均持续帧数 |
| `memory/rs_periodic_interval_mean/std` | 真正触发 periodic Q1 时抽到的 3/4/5 间隔；均值应接近 4 |
| `memory/rs_wrong_copy_rate` / `memory/event_wrong_copy_rate` | 已知错误 memory 被原样复制的比例，越低越好 |
| `memory/rs_recovery_rate` / `memory/event_recovery_rate` | wrong/UNKNOWN memory 的自主纠偏率 |
| `memory/rs_input_anomaly_rate` / `memory/event_input_anomaly_rate` | 实际进入 prompt 的 wrong+UNKNOWN 比例；EVENT 分母是 Q2 帧 |
| `memory/rs_error_streak_mean` / `memory/event_error_streak_mean` | 延迟纠偏持续长度；用于判断 patience 是否过长 |
| `memory/{rs,event}_{injected_wrong,injected_unknown,forced_repair}` | 课程扰动与兜底修复实际样本数 |
| `memory/event_invalidated_by_rs_change_rate` | RS hypothesis 变化导致旧条件 EVENT 失效的帧比例；用于核对 EVENT age 是否正确重置 |
| `memory/{rs,event}_repaired_to_{ground_truth,unknown}` | 延迟 GT 硬修复与 UNKNOWN 软擦除分开计数 |
| `memory/{rs,event}_self_recovered_after_streak` | 脚本干预前学生自行退出连续错误，越多越好 |
| `memory/{rs,event}_recovered_after_forced_repair` | 已介入修复后才答对；不能当成自主纠偏 |
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

默认 eval 口径已改为真实无先验启动：`--initial-memory unknown` 会在 route 首帧
用 RS/EVENT=UNKNOWN；`--rs-schedule-policy deployable` 只根据 UNKNOWN/非法输出、RS
标签变化后一帧确认和可复现的 3/4/5 帧随机周期来调度 RS_SLOW，不使用 GT mismatch。
旧报告时才显式传：

```bash
--initial-memory ground_truth --rs-schedule-policy oracle
```

新旧口径不能放在同一条曲线直接比较；新 summary schema 会记录
`student_initial_memory_mode` / `rs_schedule_policy` / `rs_schedule_uses_ground_truth`。
当前大样本摘要为 `schema_version=sft_v5_eval_v6`，probe 为
`format_version=5`；两者都记录普通帧独立累加的 RS/EVENT age、RS 变化导致 EVENT
上下文失效、随机 interval 的中心/jitter/seed 以及逐帧实际 interval draw。
另外，为了实现“RS 真错就跳过 EVENT”，离线评分的 EVENT gate 仍需 GT
判断一个合法 R1-R5 是否真错。因此 summary 显式写
`event_gate_policy=offline_ground_truth_rs_correctness` 和
`fully_deployable_end_to_end=false`；上线前需要 RS 置信度/几何一致性 verifier。

核心指标：

`abnormal_*` 是旧报告 schema 的兼容名称，实际都由 EVENT_FAST 最终选择的 RE/UE
family 派生，不表示仍有独立的 `ABNORMAL` 问题或输出字段。

| 指标 | 含义 | 方向 |
|---|---|---|
| `rs_acc` / `rs_transition_acc` | 全帧 RS / RS 变化首帧准确率 | 越高越好 |
| `rs_change_detection_precision/recall/f1` | 模型是否真正在正确帧切换 RS | 越高越好 |
| `rs_change_false_positive_rate` | 真值 RS 稳定时模型误切换的比例 | 越低越好 |
| `ue_entry_detection_precision/recall/f1` | RE->UE 进入帧检测 | 越高越好 |
| `ue_exit_detection_precision/recall/f1` | UE->RE 退出帧检测 | 越高越好 |
| `ue_entry_false_positive_rate` / `ue_exit_false_positive_rate` | UE 进入/退出边界误报率 | 越低越好 |
| `abnormal_precision/recall/f1` | EVENT_FAST 选项折成 UE/RE 后的查准率、召回率和 F1 | 越高越好 |
| `abnormal_false_positive_rate` | 真实 RE 被 EVENT_FAST 错选为 UE 的比例 | 越低越好 |
| `abnormal_false_negative_rate` | 真实 UE 未被 EVENT_FAST 正确选中的比例 | 越低越好 |
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
PROBE_DIR=checkpoints/sft_v5_runs/pre_opsd_base_probe_$(date +%Y%m%d_%H%M%S)
GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir "$PROBE_DIR" \
  --num-routes 1 \
  --artifact-level review \
  --with-model \
  --with-teacher-model \
  --with-teacher
```

训练后 adapter 检查：

```bash
PROBE_DIR=checkpoints/sft_v5_runs/latest/probe_with_adapter_$(date +%Y%m%d_%H%M%S)
GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_v5_runs/latest/final \
  --output-dir "$PROBE_DIR" \
  --num-routes 1 \
  --artifact-level review \
  --sample-mode random \
  --context-radius 8 \
  --with-model \
  --with-teacher-model \
  --with-teacher
```

`probe.py` 现在要求 `--output-dir` 为空：发现任何旧文件就直接拒绝，不自动删除证据。
上面用时间戳创建独立目录，避免新 `results.json` 与旧 `scenarios/frame_*` 混写。运行中
会出现 `.probe_in_progress.json`；只有逐帧 artifact 校验通过且 `results.json` 原子写完后
才删除。成功结果必须同时满足 `format_version=5`、`run_integrity.status=complete`，且
`--num-routes 1` 时 `run_integrity.selected_route_count=1`。根目录没有 `results.json`、仍有
隐藏 marker，或 route 数不符，都应视为中断/旧版混合产物并重跑，不能继续判断模型。

这条训练后命令与训练前 base 检查使用同一批输入和 schema：student 加载 LoRA，teacher
仍是纯 base Qwen。测试窗口首帧默认从 UNKNOWN 建立共同起点，第一次产生合法
RS 后会在下一帧再做一次无 GT 确认，此后 student memory 只由学生 Q1/Q2 输出
推进；truth memory 只在 `memory.json` 对照，绝不回写纠错。probe 同样支持
`--initial-memory ground_truth --rs-schedule-policy oracle` 仅复现旧结果；EVENT gate 的
离线 GT 边界与 eval 相同。

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
`teacher_q2_trigger_rate` 的分母是所选全部 frame，因为 teacher EVENT_FAST 同时覆盖
慢帧 Q1-KV 续接与快帧 fresh-RGB；它不再除以 teacher Q1 帧数，因此范围固定在 `[0,1]`。
旧兼容字段 `q1_rs_accuracy` 与 `rs_gate_accuracy` 都表示每帧实际使用 RS 的准确率；
只看慢思考输出时使用 `rs_slow_accuracy`，不能混用两个分母。

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

若出现 `loss does not require grad`，检查对应输出是否缺 `RS:` 或
`EVENT:`。当前代码会用可训练参数构造 graph-connected zero，避免单个坏输出直接中断
所有 rank，但这类警告仍表示 prompt/output 需要检查。

## 8. 注释与文档维护

v5 的函数、CLI、状态机、并行 rollout、KL 微批、memory/reset、数据过滤和指标分母均
使用中文注释说明“如何调用”和“为什么这样做”。改动这些合同后，必须同步更新
`SFT_V5_PLAN.md`、本手册、`SFT_V5_VISUALIZATION_RECORD.md` 及相邻测试。

代码阅读最快顺序：

1. 先读 `labels.py` 和 `prompts.py`，确定 RS/EVENT、动态选项、memory 与 repair 语义。
2. 再读 `build_dataset.py`，确认一条 collection route 怎样变成连续 frame sequence。
3. 从 `train.py:main` 进入 Dataset/sampler/collate，再看 `_run_frame`（慢帧语义基准）和
   `_run_event_only_frame`（快帧语义基准）。
4. 之后再看 grouped rollout、精确 Q2 KV 续接、parallel-KL 微批和 streaming optimizer；
   这些函数内部中文注释会特别说明 padding、M-RoPE、loss 分母、OOM 二分和 collective
   次序为什么不能随意简化。
5. 最后按 `metrics.py` → `eval.py` → `probe.py` 阅读评估与证据落盘，并用相邻
   `test_*.py` / `check_loss_mask.py` 对照每项 correctness 合同。

本轮在补全中文 docstring 和函数内部解释的同时，还修正了 forced-repair
恢复统计、eval/probe 首帧 GT memory 泄漏和用 GT 驱动 RS recovery 的 oracle 调度。
完整函数级导航见 `SFT_V5_PLAN.md` §9.3；probe 文件生成链路见
`SFT_V5_VISUALIZATION_RECORD.md`。
