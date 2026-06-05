# SFT v1 Runbook

所有命令默认在远端 `AutoMoT/` 目录执行。`MODEL_DIR` 默认是
`checkpoints/Qwen3-VL-4B-Instruct`；如模型在别处，给命令前缀
`MODEL_DIR=/abs/path/to/Qwen3-VL-4B-Instruct`。

## 0. 准备

```bash
cd ~/automot_lead
git pull
cd AutoMoT
ls checkpoints/Qwen3-VL-4B-Instruct/ | head -5
```

## 1. 构建数据

```bash
python tools/build_sft_dataset_v1.py \
  --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
  --data-root /datashare/IOL4SGH/data/data \
  --samples-per-scenario 800 \
  --output-dir checkpoints/sft_v1_data
```

产物：`checkpoints/sft_v1_data/{train,val}.jsonl` 和 `stats.json`。
默认采样已包含 `advance_ratio=0.35`；推进类不足时收集全部可用样本，不复制补齐。

## 2. Sanity

```bash
python tools/check_loss_mask.py
bash tools/sft_v1_train.sh check
```

通过条件：
- `check_loss_mask.py` 中只有 `STATUS` / `SUBGOAL` 事件名 token 是 `[LOSS]`。
- `check` 模式 2 step 能正常前后向，loss 没有 NaN/OOM。

## 3. 训练

```bash
# 单卡
bash tools/sft_v1_train.sh single

# DDP 默认自动挑 8 张最空闲 GPU
bash tools/sft_v1_train.sh ddp

# 指定需要几张卡，卡号仍由脚本自动挑
DDP_GPU_COUNT=4 bash tools/sft_v1_train.sh ddp
```

关键 env：

| env | 默认 | 说明 |
|---|---:|---|
| `OUTPUT_DIR` | `checkpoints/sft_v1_lora` | base 输出目录 |
| `RUN_TAG` | 时间戳 | 写到 `OUTPUT_DIR/run_<tag>` |
| `NO_RUN_SUBDIR` | `0` | 置 `1` 回到旧式覆盖写法 |
| `LR` | `5e-5` | LoRA 学习率 |
| `NUM_EPOCHS` | `2` | 小数据集避免过训 |
| `DDP_GPU_COUNT` | `8` | DDP 需要的 GPU 数 |

GPU 规则：脚本用 `nvidia-smi` 自动挑空闲卡并覆盖旧 `CUDA_VISIBLE_DEVICES`。
DDP 的 `NPROC_PER_NODE` 跟随实际挑到的 GPU 数。不要在文档命令里手写
`CUDA_VISIBLE_DEVICES=...`。

产物：
- `checkpoints/sft_v1_lora/run_<tag>/`：本次 run。
- `checkpoints/sft_v1_lora/latest`：指向最新 run 的 symlink。
- `tb/`：训练 TensorBoard。
- `invocations/`：命令、环境、git commit 记录。

## 4. TensorBoard

```bash
bash tools/tb_serve.sh checkpoints/sft_v1_lora
```

脚本自动选空闲端口并打印访问地址。

## 5. 评估

```bash
# base 小样本，带 case dump
python tools/eval_sft_v1.py \
  --lora-dir "" \
  --save-root checkpoints/sft_v1_base_eval \
  --max-samples 100

# LoRA 小样本
python tools/eval_sft_v1.py \
  --lora-dir checkpoints/sft_v1_lora/latest \
  --save-root checkpoints/sft_v1_lora/latest \
  --max-samples 100

# 全量多卡分片
torchrun --standalone --nproc_per_node=4 tools/eval_sft_v1.py \
  --lora-dir checkpoints/sft_v1_lora/latest \
  --save-root checkpoints/sft_v1_lora/latest
```

`eval_sft_v1.py` 默认自动挑 GPU；显式 `--device cpu` / `--device cuda:N`
时不覆盖用户设备设置。重要指标：

| 指标 | 含义 |
|---|---|
| `keep_accuracy` | keep 样本 STATUS 是否保持 |
| `advance_accuracy` | advance 样本 STATUS 是否推进 |
| `early_advance_rate` | keep 样本是否过早推进，越低越好 |
| `anchor12_sanity` | 早推进典型样本是否恢复正常 |

## 6. Case Probe

```bash
python tools/probe_sft_v1.py \
  --lora-dir checkpoints/sft_v1_lora/latest \
  --save-root checkpoints/sft_v1_lora/latest \
  --num-per-scenario 4 --seed 0 --case-suffix "_latest"
```

产物在 `<save-root>/eval_cases/`，包含图像、prompt、GT、pred、token loss 和
overview，适合给 AI 或人工 review。

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `KeyError: sft_v1_analysis_mask` | 确认从 `AutoMoT/` 运行，且 `tools/sft_v1_loss_scale_plugin.py` 存在 |
| `invalid device ordinal` | 不手写 CVD；DDP 用 `DDP_GPU_COUNT=N` |
| 输出反复 `STATUS:` | 过训；按 checkpoint 曲线选 early_advance 最低且 advance 不退化的点 |
| eval 缺 `STATUS/SUBGOAL` | 先看 `raw_text`，必要时启用默认 fallback，重查 LoRA 健康度 |

## 8. v1 / v2 关系

v1 使用固定 `ANALYSIS: Observations recorded.` 占位训练；v2 用 teacher 生成
真实 ANALYSIS。v2 数据、训练和评估入口见 `SFT_V2_RUN.md`。
