# SFT v2 Runbook

v2 与 v1 共享 student 训练入口，但数据先生成 `v2_pending`，再由 frozen
Qwen teacher 在训练启动时物化真实 ANALYSIS。所有命令默认在远端 `AutoMoT/`
目录执行。

## 0. 准备

```bash
cd ~/automot_lead
git pull
cd AutoMoT
ls checkpoints/Qwen3-VL-4B-Instruct/ | head -5
```

如果改过 `prompt_pipeline.py` 的 system/user/memory prompt，必须重建 pending
并刷新 runtime teacher cache：

```bash
rm -rf checkpoints/sft_v2_data_pending
RUNTIME_TEACHER_REFRESH=1 bash tools/sft_v2_train.sh check
```

## 1. 生成 Pending 数据

```bash
python tools/build_sft_dataset_v1.py \
  --mode v2 \
  --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
  --data-root /datashare/IOL4SGH/data/data \
  --output-dir checkpoints/sft_v2_data_pending
```

通过条件：
- `train.jsonl` / `val.jsonl` / `stats.json` 存在。
- 样本 `dataset_version == "v2_pending"`。
- assistant 形如 `ANALYSIS: __TEACHER_PENDING__\nSTATUS: ...\nSUBGOAL: ...`。

## 2. Teacher 预览

```bash
python tools/inspect_teacher_outputs.py \
  --jsonl checkpoints/sft_v2_data_pending/train.jsonl \
  --save-root checkpoints/sft_v2_teacher_preview_live \
  --num-per-scenario 1 --seed 42 \
  --live --serve --port 0 \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct
```

检查重点：图像顺序正确；PRIVILEGED 块匹配样本；teacher ANALYSIS 按
“看图 -> 变化 -> 结论”写，且不泄漏 `PRIVILEGED` 字样。

## 3. Teacher 物化规则

`sft_v2_train.sh` 启动时处理 teacher：

- 已有完整 runtime cache：直接复用，和 GPU 数无关。
- 无完整 cache：先调用 `build_sft_dataset_v2_teacher.py` 全量物化，再训练。
- `RUNTIME_TEACHER_REFRESH=1`：清掉旧 runtime cache 后重跑 teacher。
- `check` 模式写独立 `runtime_teacher_check_data/`，不污染正式 cache。

默认 runtime cache：

```text
checkpoints/sft_v2_lora/runtime_teacher_data/
checkpoints/sft_v2_lora/runtime_teacher_check_data/
```

manifest 必须匹配 schema、pending/runtime 行数、model_dir、seed、生成参数。
半截 debug cache 不会被误复用。

## 4. Sanity

```bash
python tools/check_loss_mask_v2.py \
  --jsonl checkpoints/sft_v2_lora/runtime_teacher_data/train.jsonl \
  --sample-idx 0

bash tools/sft_v2_train.sh check
```

通过条件：
- ANALYSIS body 权重约 `0.3`。
- `ANALYSIS:`、`\nSTATUS:`、`\nSUBGOAL:` 字面本身参与 loss。
- STATUS/SUBGOAL 事件名权重 `1.0`。
- check 2 step 正常前后向，无 NaN/OOM。

## 5. 训练

```bash
# DDP 默认自动挑 8 张最空闲 GPU
bash tools/sft_v2_train.sh ddp

# 指定需要几张卡，卡号仍自动挑
DDP_GPU_COUNT=4 bash tools/sft_v2_train.sh ddp
```

关键 env：

| env | 默认 | 说明 |
|---|---:|---|
| `OUTPUT_DIR` | `checkpoints/sft_v2_lora` | base 输出目录 |
| `RUN_TAG` | 时间戳 | 写到 `OUTPUT_DIR/run_<tag>` |
| `NO_RUN_SUBDIR` | `0` | 置 `1` 回到旧式覆盖写法 |
| `RUNTIME_TEACHER_REFRESH` | `0` | 强制重跑 teacher |
| `SFT_V2_ANALYSIS_WEIGHT` | `0.3` | ANALYSIS body loss 权重 |
| `DDP_GPU_COUNT` | `8` | DDP 需要的 GPU 数 |

GPU 规则同 v1：自动挑空闲卡并覆盖旧 `CUDA_VISIBLE_DEVICES`；不要手写卡号。

## 6. 评估与 Probe

复用 v1 脚本，换成 v2 runtime val：

```bash
python tools/eval_sft_v1.py \
  --lora-dir checkpoints/sft_v2_lora/latest \
  --val-jsonl checkpoints/sft_v2_lora/runtime_teacher_data/val.jsonl \
  --save-root checkpoints/sft_v2_lora/latest \
  --max-samples 100

python tools/probe_sft_v1.py \
  --lora-dir checkpoints/sft_v2_lora/latest \
  --val-jsonl checkpoints/sft_v2_lora/runtime_teacher_data/val.jsonl \
  --save-root checkpoints/sft_v2_lora/latest \
  --num-per-scenario 4 --seed 0 --case-suffix "_v2"
```

重点看 `early_advance_rate` 是否低于 v1，且 raw text 三段完整，不再只剩
ANALYSIS。

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| teacher 太短 / 套话 | 先看 `inspect_teacher_outputs.py --live`，必要时改 teacher prompt 后刷新 cache |
| runtime cache 被误复用 | `RUNTIME_TEACHER_REFRESH=1` 或删 `runtime_teacher_data/` |
| `dataset_version=v2_pending` 直接进 eval | 先让 `sft_v2_train.sh` 物化 runtime teacher 数据 |
| v2 不优于 v1 | 对比相同 seed 的 probe case，先确认 teacher ANALYSIS 是否真的提供视觉证据 |

## 8. 与 v1 的区别

v1 只训 STATUS/SUBGOAL 事件名，ANALYSIS 是固定占位；v2 同时让 student 学
teacher 视觉分析文本，目标是降低 keep 样本的过早推进。
