# SFT Runbook

本手册默认当前目录就是远端 `AutoMoT/`。下面命令都写相对 `AutoMoT/` 的路径，
例如 `bash qwen3vl_local/sft/train.sh`，不再额外写切目录步骤。

## 0. 准备

```bash
git pull
ls checkpoints/Qwen3-VL-4B-Instruct/ | head -5
```

`MODEL_DIR` 默认是 `checkpoints/Qwen3-VL-4B-Instruct`。如模型在别处，在命令前
加 `MODEL_DIR=/abs/path/to/Qwen3-VL-4B-Instruct`。

## 1. 总体流程

```
build_dataset.py  →  train.sh  →  eval.py / probe.py
   (pending jsonl)    (LoRA 训练，          (评估 & case dump)
                       train.py 内部 每个
                       batch 现场跑 teacher)
```

teacher ANALYSIS 不再离线物化、不再写持久 cache。train.py 在每个 train batch
里禁用 adapter，并调用底层 Qwen base model 现场 greedy 生成 ANALYSIS 真值；
随后启用 LoRA 跑 student forward + 加权 loss。这样保留一份模型显存，同时避开
`PeftModel.generate` 在 Qwen3-VL 上的生成错位问题。

如果需要离线 dump 或在浏览器抽检 teacher 输出，再单独跑 `build_teacher.py`
+ `inspect_teacher_outputs.py`，不进入训练主路径。

## 2. 构建数据

```bash
python qwen3vl_local/sft/build_dataset.py \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --samples-per-scenario 800 \
  --output-dir checkpoints/sft_data_pending
```

产物：`checkpoints/sft_data_pending/{train,val,stats}.json[l]`，所有样本
`dataset_version == "pending"`，assistant 含 `__TEACHER_PENDING__`。

## 3. Sanity

token-level weight 静态校验（无 GPU）：

```bash
python qwen3vl_local/sft/check_loss_mask.py
```

train.py 端到端 2 step 自检：

```bash
GPU_IDS=0 bash qwen3vl_local/sft/train.sh check

# 默认 GPU 0
GPU_IDS=0 bash qwen3vl_local/sft/train.sh check
```

通过条件：check 模式 2 step 正常前后向，无 NaN/OOM；初始 loss 在 3-8 区间；
`check_loss_mask.py` 输出里 ANALYSIS body token 权重为 0.5，其余 assistant
token 权重 1.0。

## 4. Teacher 预览（可选）

想在训练前看 teacher 输出（pending jsonl 上）：

```bash
GPU_IDS=0 python qwen3vl_local/sft/inspect_teacher_outputs.py \
  --jsonl checkpoints/sft_data_pending/train.jsonl \
  --save-root checkpoints/sft_teacher_inspect \
  --num-per-scenario 1 --seed 42 \
  --live --serve --port 0 \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct
```

想离线一次性 dump 全集 teacher 输出供后续 review / 分布统计：

```bash
GPU_IDS=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    qwen3vl_local/sft/build_teacher.py \
    --pending-dir checkpoints/sft_data_pending \
    --output-dir checkpoints/sft_teacher_dump \
    --model-dir checkpoints/Qwen3-VL-4B-Instruct \
    --seed 20260601
```

dump 不会被 train.sh 自动复用，下次启动 train 仍然现场跑 teacher。

## 5. 训练

```bash
GPU_IDS=0 bash qwen3vl_local/sft/train.sh single
DDP_GPU_COUNT=4 bash qwen3vl_local/sft/train.sh ddp

# 想固定到指定卡（单卡默认 GPU 0，多卡默认 GPU 0,1,2,3）
GPU_IDS=0 bash qwen3vl_local/sft/train.sh single
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft/train.sh ddp
```

关键 env：

| env | 默认 | 说明 |
|---|---:|---|
| `OUTPUT_DIR` | `checkpoints/sft_lora` | base 输出目录 |
| `RUN_TAG` | 时间戳 | 写到 `OUTPUT_DIR/run_<tag>` |
| `NO_RUN_SUBDIR` | `0` | 置 `1` 回到顶层覆盖写法 |
| `DDP_GPU_COUNT` | `8` | DDP 需要的 GPU 数；`GPU_IDS` 非空时忽略 |
| `GPU_IDS` | 空 | 显式 pin 卡号；空 = nvidia-smi 自动选址 |
| `SFT_ANALYSIS_WEIGHT` | `0.5` | ANALYSIS body loss 权重；默认中等强度学习语言推理 |
| `SFT_TEACHER_MAX_NEW_TOKENS` | `256` | teacher 单次生成上限 |
| `SFT_TEACHER_TEMPERATURE` | `0.0` | teacher 采样温度（0 = greedy） |
| `NUM_EPOCHS` / `LR` / `MAX_LENGTH` / `LORA_RANK` 等 | 见 train.sh | 都可以直接 env override |

`SFT_ANALYSIS_WEIGHT=0.5` 是折中口径：让 LoRA 学 ANALYSIS 的大致语言推理套路，
但不把 teacher 的逐字措辞压成主任务。若需要更多 paraphrase 式语言多样性，可在训练命令前
显式加 `SFT_TEACHER_TEMPERATURE=0.2` 或 `0.3`；默认仍保持 greedy，优先保证稳定复现。

GPU 规则：脚本默认用 `nvidia-smi` 自动挑空闲卡并覆盖旧 `CUDA_VISIBLE_DEVICES`。
显式 pin 时前置 `GPU_IDS=<id1,id2,...>`，跳过自动选址；卡数从 `GPU_IDS` 逗号数推断。

每个训练 run 目录会追加 `log.txt` 保存本次终端 stdout/stderr。
LoRA adapter 默认每 `SAVE_STEPS` 写 `checkpoint-<step>/`，训练结束再写 `final/`。

## 6. TensorBoard

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_lora
```

脚本自动选空闲端口并打印访问地址。

## 7. 评估

```bash
GPU_IDS=0 python qwen3vl_local/sft/eval.py \
  --lora-dir checkpoints/sft_lora/latest/final \
  --save-root checkpoints/sft_lora/latest \
  --max-samples 100

GPU_IDS=0,1,2,3 torchrun --standalone --nproc_per_node=4 qwen3vl_local/sft/eval.py \
  --lora-dir checkpoints/sft_lora/latest/final \
  --save-root checkpoints/sft_lora/latest
```

重点指标：`keep_accuracy` 越高越好，`advance_accuracy` 越高越好，
`early_advance_rate` 越低越好，`anchor12_sanity=True` 必须保持。
Eval 的终端输出会追加到 `<save-root>/eval/log.txt`。
当 `--max-samples > 0` 或显式 `--full-dump` 时，`<save-root>/eval/cases/` 里每个
case 也会保存 `outputs/expert_analysis.txt` 与 `outputs/language_compare.json`，
用于对比专家语言和模型自己的 ANALYSIS。

如果想跑 base 模型对照（不挂 LoRA）：传 `--lora-dir ''`。

## 8. Case Probe

```bash
GPU_IDS=0 python qwen3vl_local/sft/probe.py \
  --lora-dir checkpoints/sft_lora/latest/final \
  --save-root checkpoints/sft_lora/latest \
  --num-per-scenario 4 --seed 0 --case-suffix "_final"
```

产物在 `<save-root>/eval_cases/`：每个 case 子目录里有图像、prompt、GT、pred、`expert_analysis.txt`、`language_compare.json`、token loss 和 overview；
`expert_analysis.txt` 是 base teacher 在带 PRIVILEGED 的专家 prompt 下生成的 ANALYSIS，`language_compare.json` 把专家语言、模型自己的 ANALYSIS、物化 GT ANALYSIS（若有）并排保存。
`<save-root>/eval_cases/log.txt` 是本次 probe 的终端 stdout/stderr 汇总日志（不在每个 case 内）。
`token_loss.json` 里的 `mean_loss_weighted_train` 按训练真实权重汇总：
ANALYSIS body=`SFT_ANALYSIS_WEIGHT`，结构字面 / STATUS / SUBGOAL / tail 全部为 1.0。

## 9. 常见问题

| 现象 | 处理 |
|---|---|
| 训练 step 异常慢 | 正常；teacher 每 step 跑 frozen base generate，整体 ~3-4x base LoRA 用时 |
| `dataset_version=pending` 进 eval 后 GT ANALYSIS 是占位 | STATUS / SUBGOAL 评测不受影响；想看真 ANALYSIS 跑 `build_teacher.py` 物化 val 后再传 |
| `invalid device ordinal` | 训练入口锁卡用 `GPU_IDS=0` / `GPU_IDS=0,1,2,3`；不要手写 CVD；DDP 卡数从 `GPU_IDS` 推断 |
| eval 输出乱码 "ANALERTA" / "ANAL" | 没 `merge_and_unload`；用默认 `--merge-lora`，或检查 PEFT 版本 |
| teacher 太短 / 套话 | 先看 `inspect_teacher_outputs.py --live`，必要时改 teacher prompt 后重启训练 |
| DDP 启动时卡死 | 看 stdout `[gpu]` 行；`GPU_IDS` 卡数与 `DDP_GPU_COUNT` 不一致会预先报错，所有 rank 必须看到一致的 mask |
