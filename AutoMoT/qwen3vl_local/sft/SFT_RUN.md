# SFT Runbook

本手册默认当前目录就是远端 `AutoMoT/`。下面命令都写相对 `AutoMoT/` 的路径，
例如 `bash qwen3vl_local/sft/sft_v1_train.sh`，不再额外写切目录步骤。
SFT v1/v2 共享数据构建、评估、probe 和 TensorBoard 工具；差异主要在
ANALYSIS 段监督方式。

## 0. 准备

```bash
git pull
ls checkpoints/Qwen3-VL-4B-Instruct/ | head -5
```

`MODEL_DIR` 默认是 `checkpoints/Qwen3-VL-4B-Instruct`。如模型在别处，在命令前
加 `MODEL_DIR=/abs/path/to/Qwen3-VL-4B-Instruct`。

## 1. 选模式

| 模式 | 适用场景 | ANALYSIS | 训练入口 |
|---|---|---|---|
| v1 | 先稳住 STATUS/SUBGOAL，不学视觉分析正文 | 固定占位 `Observations recorded.`，loss=0 | `sft_v1_train.sh` |
| v2 | 在 v1 基础上蒸馏真实视觉分析，降低过早推进 | frozen Qwen teacher 物化，body loss 默认 0.3 | `sft_v2_train.sh` |

v1 和 v2 的评估都用 `eval_sft_v1.py` / `probe_sft_v1.py`。脚本会按 jsonl
字段自动识别 v1/v2；v2 eval 必须传已经物化后的 runtime teacher jsonl。

## 2. 构建数据

v1：

```bash
python qwen3vl_local/sft/build_sft_dataset_v1.py \
  --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
  --data-root /datashare/IOL4SGH/data/data \
  --samples-per-scenario 800 \
  --output-dir checkpoints/sft_v1_data
```

v2 pending：

```bash
python qwen3vl_local/sft/build_sft_dataset_v1.py \
  --mode v2 \
  --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
  --data-root /datashare/IOL4SGH/data/data \
  --output-dir checkpoints/sft_v2_data_pending
```

v1 产物为 `checkpoints/sft_v1_data/{train,val}.jsonl`；v2 pending 样本的
`dataset_version == "v2_pending"`，assistant 中包含 `__TEACHER_PENDING__`。

## 3. Sanity

v1：

```bash
python qwen3vl_local/sft/check_loss_mask.py
bash qwen3vl_local/sft/sft_v1_train.sh check

# 想固定到某张卡（默认 GPU 0）
GPU_IDS=0 bash qwen3vl_local/sft/sft_v1_train.sh check
```

v2：

```bash
RUNTIME_TEACHER_REFRESH=1 bash qwen3vl_local/sft/sft_v2_train.sh check
python qwen3vl_local/sft/check_loss_mask_v2.py \
  --jsonl checkpoints/sft_v2_lora/runtime_teacher_check_data/train.jsonl \
  --sample-idx 0

# 想固定到某张卡（默认 GPU 0）
GPU_IDS=0 RUNTIME_TEACHER_REFRESH=1 bash qwen3vl_local/sft/sft_v2_train.sh check
```

通过条件：check 模式 2 step 正常前后向，无 NaN/OOM；v1 只有 STATUS/SUBGOAL
事件名 token 参与 loss；v2 的 ANALYSIS body 权重约 0.3，三段结构字面和事件名参与 loss。

## 4. Teacher 预览与缓存

v2 可先看 teacher 输出：

```bash
python qwen3vl_local/sft/inspect_teacher_outputs.py \
  --jsonl checkpoints/sft_v2_data_pending/train.jsonl \
  --save-root checkpoints/sft_v2_teacher_preview_live \
  --num-per-scenario 1 --seed 42 \
  --live --serve --port 0 \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct
```

`sft_v2_train.sh` 启动时会处理 teacher cache：

- 完整 `runtime_teacher_data/` 存在且 manifest 匹配：直接复用，和 GPU 数无关。
- cache 不完整或 manifest 不匹配：自动重物化。
- `RUNTIME_TEACHER_REFRESH=1`：强制清掉旧 cache 后重跑。
- `check` 模式默认写 `runtime_teacher_check_data/`，不污染正式 cache。

如果改过 `prompt_pipeline.py` 的 system/user/memory prompt，重建 pending 并刷新 runtime cache。

## 5. 训练

v1：

```bash
bash qwen3vl_local/sft/sft_v1_train.sh single
bash qwen3vl_local/sft/sft_v1_train.sh ddp
DDP_GPU_COUNT=4 bash qwen3vl_local/sft/sft_v1_train.sh ddp

# 想固定到指定卡（单卡默认 GPU 0，多卡默认 GPU 0,1,2,3）
GPU_IDS=0 bash qwen3vl_local/sft/sft_v1_train.sh single
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft/sft_v1_train.sh ddp
```

v2：

```bash
bash qwen3vl_local/sft/sft_v2_train.sh single
bash qwen3vl_local/sft/sft_v2_train.sh ddp
DDP_GPU_COUNT=4 bash qwen3vl_local/sft/sft_v2_train.sh ddp

# 想固定到指定卡（单卡默认 GPU 0，多卡默认 GPU 0,1,2,3）
GPU_IDS=0 bash qwen3vl_local/sft/sft_v2_train.sh single
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft/sft_v2_train.sh ddp
```

关键 env：

| env | v1 默认 | v2 默认 | 说明 |
|---|---:|---:|---|
| `OUTPUT_DIR` | `checkpoints/sft_v1_lora` | `checkpoints/sft_v2_lora` | base 输出目录 |
| `RUN_TAG` | 时间戳 | 时间戳 | 写到 `OUTPUT_DIR/run_<tag>` |
| `NO_RUN_SUBDIR` | `0` | `0` | 置 `1` 回到顶层覆盖写法 |
| `DDP_GPU_COUNT` | `8` | `8` | DDP 需要的 GPU 数；`GPU_IDS` 非空时忽略 |
| `GPU_IDS` | 空 | 空 | 显式 pin 卡号；空 = nvidia-smi 自动选址；`GPU_IDS=0` / `GPU_IDS=0,1,2,3` |
| `RUNTIME_TEACHER_REFRESH` | - | `0` | v2 强制重跑 teacher |
| `SFT_V2_ANALYSIS_WEIGHT` | - | `0.3` | v2 ANALYSIS body loss 权重 |

GPU 规则：脚本默认用 `nvidia-smi` 自动挑空闲卡并覆盖旧 `CUDA_VISIBLE_DEVICES`。
显式 pin 时前置 `GPU_IDS=<id1,id2,...>`，跳过自动选址；卡数从 `GPU_IDS` 逗号数推断。

每个训练 run 目录会追加 `log.txt` 保存本次终端 stdout/stderr。

## 6. TensorBoard

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_v1_lora
bash qwen3vl_local/tb_serve.sh checkpoints/sft_v2_lora
```

脚本自动选空闲端口并打印访问地址。

## 7. 评估

v1：

```bash
python qwen3vl_local/sft/eval_sft_v1.py \
  --lora-dir checkpoints/sft_v1_lora/latest \
  --save-root checkpoints/sft_v1_lora/latest \
  --max-samples 100

torchrun --standalone --nproc_per_node=4 qwen3vl_local/sft/eval_sft_v1.py \
  --lora-dir checkpoints/sft_v1_lora/latest \
  --save-root checkpoints/sft_v1_lora/latest
```

v2：

```bash
python qwen3vl_local/sft/eval_sft_v1.py \
  --lora-dir checkpoints/sft_v2_lora/latest \
  --val-jsonl checkpoints/sft_v2_lora/runtime_teacher_data/val.jsonl \
  --save-root checkpoints/sft_v2_lora/latest \
  --max-samples 100
```

重点指标：`keep_accuracy` 越高越好，`advance_accuracy` 越高越好，
`early_advance_rate` 越低越好，`anchor12_sanity=True` 必须保持。
Eval 的终端输出会追加到 `<save-root>/eval/log.txt`。

## 8. Case Probe

v1：

```bash
python qwen3vl_local/sft/probe_sft_v1.py \
  --lora-dir checkpoints/sft_v1_lora/latest \
  --save-root checkpoints/sft_v1_lora/latest \
  --num-per-scenario 4 --seed 0 --case-suffix "_v1"
```

v2：

```bash
python qwen3vl_local/sft/probe_sft_v1.py \
  --lora-dir checkpoints/sft_v2_lora/latest \
  --val-jsonl checkpoints/sft_v2_lora/runtime_teacher_data/val.jsonl \
  --save-root checkpoints/sft_v2_lora/latest \
  --num-per-scenario 4 --seed 0 --case-suffix "_v2"
```

产物在 `<save-root>/eval_cases/`：每个 case 子目录里有图像、prompt、GT、pred、token loss 和 overview；
`<save-root>/eval_cases/log.txt` 是本次 probe 的终端 stdout/stderr 汇总日志（不在每个 case 内）。

## 9. 常见问题

| 现象 | 处理 |
|---|---|
| `KeyError: sft_v1_analysis_mask` | 确认当前目录就是默认的 `AutoMoT/`，且 `qwen3vl_local/sft/sft_v1_loss_scale_plugin.py` 存在 |
| `dataset_version=v2_pending` 直接进 eval | 先让 `sft_v2_train.sh` 物化 runtime teacher 数据 |
| runtime cache 被误复用 | `RUNTIME_TEACHER_REFRESH=1` 或删 `runtime_teacher_data/` |
| `invalid device ordinal` | 训练入口锁卡用 `GPU_IDS=0` / `GPU_IDS=0,1,2,3`；不要手写 CVD；DDP 卡数从 `GPU_IDS` 推断 |
| 输出反复 `STATUS:` | 过训；按 checkpoint 曲线选 early_advance 最低且 advance 不退化的点 |
| teacher 太短 / 套话 | 先看 `inspect_teacher_outputs.py --live`，必要时改 teacher prompt 后刷新 cache |
