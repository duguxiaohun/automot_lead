# SFT Base Run

默认当前目录是远端 `AutoMoT/`。

## 1. 构建数据

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

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --index checkpoints/sft_base_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --output-json checkpoints/sft_base_runs/latest/eval_metrics.json
```

`eval.py` 会先校验 adapter 目录中的 `sft_base_adapter_config.json`。如果 route、
dataset version、base model path 或 vision scope 不匹配，会直接报错。

## 6. 维护检查

```bash
python -m py_compile qwen3vl_local/sft_base/*.py
python qwen3vl_local/sft_base/check_loss_mask.py
python qwen3vl_local/sft_base/test_dataset_contract.py
```

`test_dataset_contract.py` 会检查 sft_base 与 sft_v5 的 Q2 option-letter 映射是否保持一致。
多卡训练相关改动需要额外关注 `train.py` 中 `_sync_trainable_grads()` 与
`run_batch(..., sync_grads=True)` 的调用边界，确保每个 rank 的 collective 次数一致。
