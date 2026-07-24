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

```bash
LORA_VISION_SCOPE=off GPU_IDS=0 bash qwen3vl_local/sft_base/train.sh single
```

关闭视觉 fuse guard 只建议排查用：

```bash
VISION_GUARD_ENABLED=0 GPU_IDS=0 bash qwen3vl_local/sft_base/train.sh single
```

## 4. 评估

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --index checkpoints/sft_base_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --output-json checkpoints/sft_base_runs/latest/eval_metrics.json
```

`eval.py` 会先校验 adapter 目录中的 `sft_base_adapter_config.json`。如果 route、
dataset version、base model path 或 vision scope 不匹配，会直接报错。

## 5. 维护检查

```bash
python -m py_compile qwen3vl_local/sft_base/*.py
python qwen3vl_local/sft_base/check_loss_mask.py
python qwen3vl_local/sft_base/test_dataset_contract.py
```

`test_dataset_contract.py` 会检查 sft_base 与 sft_v5 的 Q2 option-letter 映射是否保持一致。
多卡训练相关改动需要额外关注 `train.py` 中 `_sync_trainable_grads()` 与
`run_batch(..., sync_grads=True)` 的调用边界，确保每个 rank 的 collective 次数一致。
