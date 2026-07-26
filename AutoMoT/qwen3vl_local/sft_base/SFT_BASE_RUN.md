# SFT Base Run

默认当前目录是远端 `AutoMoT/`。

## 1. 构建数据

当前协议使用固定语义 token 作为答案，例如 `RS: SIGNAL_INTERSECTION`、
`EVENT: RULE_VIOLATION`，不再输出 A/B/C 选项。`DATASET_VERSION` 已更新，
旧 A/B/C adapter 不能直接用于这版评估；切换协议后需要重新构建数据并重新训练。

```bash
python qwen3vl_local/sft_base/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_base_data
```

Smoke：

```bash
python qwen3vl_local/sft_base/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_base_data_smoke \
  --max-routes 2 \
  --max-frames-per-route 4
```

## 2. 静态检查

```bash
python qwen3vl_local/sft_base/check_loss_mask.py
python qwen3vl_local/sft_base/test_dataset_contract.py
GPU_IDS=0 bash qwen3vl_local/sft_base/train.sh check
```

## 3. 训练

单卡：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_base/train.sh single
```

4 卡：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_base/train.sh ddp
```

默认输出到 `checkpoints/sft_base_runs/run_<RUN_TAG>/`，并维护 `checkpoints/sft_base_runs/latest`。默认 `LORA_VISION_SCOPE=merger`，并启用视觉 fuse guard；如果要纯语言 LoRA 对照：

多卡训练默认 `FRAMES_PER_SYNC=64`，会在长 route 内按固定帧数做梯度同步 heartbeat，避免不同 rank 的 route 帧数差异导致 NCCL all-reduce 等待超时。排查时可调小到 `32`，或在确认单条 route 很短时设为 `0` 回到整条 route 结束后同步。

针对 checkpoint-600 里 `ue_acc=0` 的问题，当前默认加强 UE 监督：

```bash
UE_EVENT_LOSS_WEIGHT=3.0 RE_EVENT_LOSS_WEIGHT=1.0 UE_FRAME_REPEAT=2 \
GPU_IDS=0 bash qwen3vl_local/sft_base/train.sh single
```

`UE_EVENT_LOSS_WEIGHT` 只加 Q2 的 UE EVENT token loss；`UE_FRAME_REPEAT` 只重复异常帧的训练样本，不改变 route memory 推进。训练日志和 TensorBoard 会写 `train/q2_ue_rate_last_batch`，用于确认本轮 batch 里确实喂到了 UE 监督。

```bash
LORA_VISION_SCOPE=off GPU_IDS=0 bash qwen3vl_local/sft_base/train.sh single
```

关闭视觉 fuse guard 只建议排查用：

```bash
VISION_GUARD_ENABLED=0 GPU_IDS=0 bash qwen3vl_local/sft_base/train.sh single
```

## 4. TensorBoard 与日志

训练脚本会在 rank0 写 TensorBoard events：

```text
checkpoints/sft_base_runs/run_<RUN_TAG>/tb/
checkpoints/sft_base_runs/latest/tb/
```

远端从 `AutoMoT/` 目录启动：

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_base_runs/latest/tb
```

也可以把 logdir 指到 run 根目录，后续如果增加 `eval_tb/` 等子目录，TensorBoard 会在左侧 run 列表中分开展示：

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_base_runs/latest
```

`tb_serve.sh` 会自动选空闲端口、用 `--bind_all` 启动 TensorBoard，并在 stdout 打印本地浏览器 URL。常用覆盖：

```bash
TB_PORT=6007 bash qwen3vl_local/tb_serve.sh checkpoints/sft_base_runs/latest/tb
TB_BIND=127.0.0.1 bash qwen3vl_local/tb_serve.sh checkpoints/sft_base_runs/latest/tb
TB_EXTRA="--samples_per_plugin images=200" bash qwen3vl_local/tb_serve.sh checkpoints/sft_base_runs/latest/tb
```

优先查看：

| Tag | 用途 |
|---|---|
| `train/loss` | optimizer step 后的全局平均训练 loss |
| `train/q2_rate_last_batch` | 最近一次同步 batch 中包含 Q2 的帧比例 |
| `train/q2_ue_rate_last_batch` | 最近一次 Q2 监督中 UE 帧比例，用于排查 UE 是否被 RE 淹没 |
| `train/grad_norm/language` | 语言 LoRA 梯度范数 |
| `train/grad_norm/vision` | 视觉 LoRA/merger 相关梯度范数；`LORA_VISION_SCOPE=off` 时可能没有 |
| `train/param_norm/lora_vision` | 视觉侧可训练参数范数，用于观察 fuse 是否异常漂移 |
| `train/vision_guard_bad_steps` | 视觉 fuse guard 连续异常步数 |
| `val/loss` | 评估间隔触发的 teacher-forced 验证 loss |
| `val/samples` / `val/skipped` | 验证样本数与跳过帧数 |
| `val/q2_rate` | 验证集中 Q2 监督帧比例 |

训练 stdout/stderr 默认同步写到当前 run 的 `log.txt`，路径为：

```text
checkpoints/sft_base_runs/run_<RUN_TAG>/log.txt
checkpoints/sft_base_runs/latest/log.txt
```

如果 TensorBoard 里只有 events header 没有曲线，先确认训练已经完成至少 1 个 optimizer step；`LOGGING_STEPS` 只影响写入频率，默认每 5 step 记录一次，step 1 也会记录。

## 5. 评估

评估分三类，并且完全不做脚本纠正：Q1 预测错 RS 时只跳过当前帧 Q2，下一帧继续沿用模型自己维护出的 memory；Q2 输出非法时也不重置。

随机完整路线闭环测试：随机抽若干条 route，从起点跑到终点，看模型在自然路径里长期记忆是否漂移。

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --index checkpoints/sft_base_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --eval-mode full_route \
  --sample-routes 16 \
  --seed 20260724 \
  --output-json checkpoints/sft_base_runs/latest/eval_full_route_metrics.json \
  --output-jsonl checkpoints/sft_base_runs/latest/eval_full_route_frames.jsonl
```

RS 转折专项测试：只取 RS 变化点前后 `--transition-window` 帧，并用 `--transition-tolerance` 允许提前或滞后若干帧；重点看模型是否能在容忍窗口内切到正确 RS，而不是逐帧和数据标注完全同拍。

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --index checkpoints/sft_base_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --eval-mode rs_transition \
  --transition-window 8 \
  --transition-tolerance 3 \
  --max-transition-cases 128 \
  --seed 20260724 \
  --output-json checkpoints/sft_base_runs/latest/eval_rs_transition_metrics.json \
  --output-jsonl checkpoints/sft_base_runs/latest/eval_rs_transition_frames.jsonl
```

UE/RE/EVENT 转换专项测试：只取 EVENT 或 ABNORMAL 状态变化点前后窗口，检查模型是否能在容忍窗口内切到正确 EVENT；`event_transition_abnormal_hit_rate` 另看 Q1 的 YES/NO 是否先切对。

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --index checkpoints/sft_base_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --eval-mode event_transition \
  --transition-window 8 \
  --transition-tolerance 3 \
  --max-transition-cases 128 \
  --seed 20260724 \
  --output-json checkpoints/sft_base_runs/latest/eval_event_transition_metrics.json \
  --output-jsonl checkpoints/sft_base_runs/latest/eval_event_transition_frames.jsonl
```

`eval.py` 会先校验 adapter 目录中的 `sft_base_adapter_config.json`。如果 route、
dataset version、base model path 或 vision scope 不匹配，会直接报错。

起始 memory 噪声只在每条完整 route 或每个转折窗口的第一帧注入，后续仍然不纠正，用来测模型能不能自己恢复：

```bash
--initial-memory-noise rs
--initial-memory-noise event
--initial-memory-noise both
--initial-memory-noise random
```

常用指标口径：

| 指标 | 含义 |
|---|---|
| `rs_acc` / `event_acc_when_rs_correct` | 全部评估帧上的 RS 准确率、RS 正确时的 EVENT 准确率 |
| `q2_trigger_rate` | Q1 RS 正确后进入 Q2 的比例；RS 漂移会直接压低这个值 |
| `script_resets` | 脚本纠偏审计字段；评测不允许纠偏，正常必须恒为 0 |
| `rs_transition_hit_rate` | RS 转折 case 在容忍窗口内切到目标 RS 的比例 |
| `event_transition_hit_rate` | UE/RE/EVENT 转换 case 在容忍窗口内切到目标 EVENT 的比例 |
| `event_transition_abnormal_hit_rate` | UE/RE 转换 case 在容忍窗口内 Q1 YES/NO 切对的比例 |
| `ue_q1_abnormal_acc` | 所有 UE 帧里 Q1 是否先报 `ABNORMAL=YES` |
| `ue_pred_regular_rate` | UE 帧进入 Q2 后仍被判成 `REGULAR` 的比例 |
| `*_hit_offset_avg` / `*_abs_hit_offset_avg` | 命中帧相对标注转折帧的平均偏移和平均绝对偏移，负数表示提前 |
| `*_early_hits` / `*_on_time_hits` / `*_late_hits` | 命中发生在标注转折前、同帧或后几帧的数量 |
| `output-jsonl` 每行 | 单帧复盘和 `transition_case_summary`，包含 route/frame、转折点、容忍窗口、GT/PRED RS、GT/PRED EVENT、原始生成文本 |

## 6. 维护检查

```bash
python -m py_compile qwen3vl_local/sft_base/*.py
python qwen3vl_local/sft_base/check_loss_mask.py
python qwen3vl_local/sft_base/test_dataset_contract.py
```

`test_dataset_contract.py` 会检查 sft_base 与 sft_v5 的 Q2 option-letter 映射是否保持一致。
多卡训练相关改动需要额外关注 `train.py` 中 `_sync_trainable_grads()` 与
`run_batch(..., sync_grads=True)` 的调用边界，确保每个 rank 的 collective 次数一致。
