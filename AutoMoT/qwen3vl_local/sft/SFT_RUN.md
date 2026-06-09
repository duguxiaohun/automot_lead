# SFT Runbook

所有命令默认在远端 `AutoMoT/` 目录执行。SFT v1/v2 共享数据构建、评估、
probe 和 TensorBoard 工具；差异主要在 ANALYSIS 段监督方式。

## 0. 准备

```bash
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
```

v2：

```bash
RUNTIME_TEACHER_REFRESH=1 bash qwen3vl_local/sft/sft_v2_train.sh check
python qwen3vl_local/sft/check_loss_mask_v2.py \
  --jsonl checkpoints/sft_v2_lora/runtime_teacher_check_data/train.jsonl \
  --sample-idx 0
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
```

v2：

```bash
bash qwen3vl_local/sft/sft_v2_train.sh single
bash qwen3vl_local/sft/sft_v2_train.sh ddp
DDP_GPU_COUNT=4 bash qwen3vl_local/sft/sft_v2_train.sh ddp
```

关键 env：

| env | v1 默认 | v2 默认 | 说明 |
|---|---:|---:|---|
| `OUTPUT_DIR` | `checkpoints/sft_v1_lora` | `checkpoints/sft_v2_lora` | base 输出目录 |
| `RUN_TAG` | 时间戳 | 时间戳 | 写到 `OUTPUT_DIR/run_<tag>` |
| `NO_RUN_SUBDIR` | `0` | `0` | 置 `1` 回到顶层覆盖写法 |
| `DDP_GPU_COUNT` | `8` | `8` | DDP 需要的 GPU 数 |
| `RUNTIME_TEACHER_REFRESH` | - | `0` | v2 强制重跑 teacher |
| `SFT_V2_ANALYSIS_WEIGHT` | - | `0.3` | v2 ANALYSIS body loss 权重 |
| `PER_DEVICE_BS` | single=22 / ddp=15 | single=11 / ddp=10 | 每卡 micro-batch；默认优先避开 Qwen3-VL full-logits fp32 loss 峰值 |
| `GRAD_ACC` | single=2 / ddp=2 | single=4 / ddp=3 | 梯度累积步数；与 PER_DEVICE_BS 共同决定等效 batch |
| `USE_LOGITS_TO_KEEP` | `true` | `true` | 传给 ms-swift 的 `--use_logits_to_keep`，只为 label 区间算 logits 以降低显存 |

GPU 规则：脚本用 `nvidia-smi` 自动挑空闲卡并覆盖旧 `CUDA_VISIBLE_DEVICES`。
不要在文档命令里手写卡号。

端口规则：SFT v1/v2 的 `check` / `single` / `ddp` 都会在进入 `swift sft` 前自动选择空闲
`MASTER_PORT`，并同步导出 PyTorch launcher 会读取的 `PET_MASTER_PORT`；只有显式同时设置
`MASTER_PORT` 与 `SFT_RESPECT_MASTER_PORT=1` 时才严格使用指定端口。

`check` 模式只用于 2 step loss_scale sanity，默认传 `--report_to none`，不写 TensorBoard，也不触发
swift 训练结束后的 matplotlib loss 曲线绘图；正式 `single` / `ddp` 仍写 `tb/`。

### 5.1 H20 96GB 显存与 batch 调优

2026-06 起先经过**四轮吞吐调优**，随后第五轮针对 Qwen3-VL loss 峰值做稳定性修正：

- 第一轮（v1/v2 同步）：等效 global batch 从早期保守值翻 2x。
- 第二轮：再翻 2x。两轮合计 4x，LR 总倍率 2x（v1 5e-5→1.0e-4，v2 3e-5→5.9e-5）。
- 第三轮：默认再乘 1.5x，瞄准 H20 更高显存利用率（v1 1.0e-4→1.23e-4，v2 5.9e-5→7.2e-5）。
- 第四轮：默认从 global 192 推到 240（v1 1.23e-4→1.37e-4，v2 7.2e-5→8.0e-5）；single 略降，避免峰值过近 96GB 上限。
- 第五轮：显式启用 `--use_logits_to_keep true`，并把 micro-batch 拆小、用 `GRAD_ACC` 补回等效 batch，避免 `logits.float()` 在第一步额外申请数十 GB。

最终默认 8 卡 ddp 等效 global batch = **240**；single 等效：v1 **44**，v2 **44**。
默认已打开 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，降低 allocator 碎片导致的峰值失败概率。

**用户无需显式设置 batch size，直接 `bash qwen3vl_local/sft/sft_v*_train.sh ddp` 即可**。

若 `nvidia-smi` 看到训练时单卡显存很宽松，可继续在 shell 里覆盖（LR 按 sqrt 法则同步上调）：

```bash
# v1 ddp 再 +1.07x 等效 batch（global 256），LR 1.37e-4 × sqrt(256/240)=1.41e-4
PER_DEVICE_BS=16 GRAD_ACC=2 LR=1.41e-4 bash qwen3vl_local/sft/sft_v1_train.sh ddp

# v2 ddp 再 +1.07x 等效 batch（global 256），LR 8.0e-5 × sqrt(256/240)=8.3e-5
PER_DEVICE_BS=8 GRAD_ACC=4 LR=8.3e-5 bash qwen3vl_local/sft/sft_v2_train.sh ddp
```

若 OOM，先 ÷2 PER_DEVICE_BS、×2 GRAD_ACC（等效 batch 不变，仅省 activation 显存），
并确认日志里 `USE_LOGITS_TO_KEEP=true`。如果某个 ms-swift / Transformers 组合启用该参数后报错，
可临时 `USE_LOGITS_TO_KEEP=false`，但需要继续降低 `PER_DEVICE_BS`。check 模式始终保持 `PER_DEVICE_BS=1 / GRAD_ACC=1`
不动 —— 它只用来 2 步快速判读 loss_scale，不参与吞吐优化。

历史回退：想跑回 2026-06 之前的"保守"路径，用 `PER_DEVICE_BS=2 GRAD_ACC=2 LR=5e-5`
（v1）/ `LR=3e-5`（v2），即可复现早期 ckpt 训练动力学。

## 6. TensorBoard

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_v1_lora
bash qwen3vl_local/tb_serve.sh checkpoints/sft_v2_lora
```

脚本自动选空闲端口并打印访问地址。

SFT v1/v2 的 loss_scale plugin 会保留 `tb/` 下的 TensorBoard event 写入，但会跳过
ms-swift 训练结束后额外生成 `images/` loss PNG 的步骤。这个 PNG 导出依赖
matplotlib；远程环境里若 numpy/matplotlib 版本不匹配，训练已经结束后仍可能在
`plot_images(...)` 里崩掉。当前默认不生成这份 PNG，不需要修改 conda 包；直接用
`tb_serve.sh` 看曲线即可。

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

产物在 `<save-root>/eval_cases/`，包含图像、prompt、GT、pred、token loss 和 overview。

## 9. 常见问题

| 现象 | 处理 |
|---|---|
| `KeyError: sft_v1_analysis_mask` | 确认从 `AutoMoT/` 运行，且 `qwen3vl_local/sft/sft_v1_loss_scale_plugin.py` 存在 |
| `dataset_version=v2_pending` 直接进 eval | 先让 `sft_v2_train.sh` 物化 runtime teacher 数据 |
| runtime cache 被误复用 | `RUNTIME_TEACHER_REFRESH=1` 或删 `runtime_teacher_data/` |
| 训练结束后在 `plot_images(...)` / `numpy.core.umath ERR_IGNORE` 崩掉 | 训练主体已完成；v1/v2 plugin 默认跳过 swift 的 matplotlib PNG 导出，保留 `tb/` events，不需要改 conda 包 |
| `invalid device ordinal` | 不手写 CVD；DDP 用 `DDP_GPU_COUNT=N` |
| 输出反复 `STATUS:` | 过训；按 checkpoint 曲线选 early_advance 最低且 advance 不退化的点 |
| teacher 太短 / 套话 | 先看 `inspect_teacher_outputs.py --live`，必要时改 teacher prompt 后刷新 cache |
