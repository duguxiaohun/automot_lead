# SFT v2 运行手册

以下命令默认在远端 `AutoMoT/` 目录下执行。

## 1. 构建数据

默认构建会保留每个 scenario 的全部合法候选：

```bash
python qwen3vl_local/sft_v2/build_dataset.py \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --output-dir checkpoints/sft_v2_data
```

默认 `--wrong-scene-ratio 0.15`：一部分 train row 的第二阶段 prompt 会故意写入一个
错误但合法的 selected scene；previous hint 和监督的 status/subgoal 会按相位映射到
该 selected scene 自己的事件序列。若要关闭增强，设为 `--wrong-scene-ratio 0`。

可选的下采样构建：

```bash
python qwen3vl_local/sft_v2/build_dataset.py \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --samples-per-scenario 800 \
  --output-dir checkpoints/sft_v2_data_800
```

快速 dry-run：

```bash
python qwen3vl_local/sft_v2/build_dataset.py \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --dry-run \
  --output-dir checkpoints/sft_v2_data_dry
```

输出文件：

- `train.jsonl`
- `val.jsonl`
- `stats.json`

每条 row 的两阶段消息保存在 `stage_messages` 下：

```text
stage_messages.scene   -> 图像 turn + SCENE
stage_messages.status  -> 文本 follow-up turn + STATUS / SUBGOAL
```

## 2. 检查 Loss Mask

```bash
python qwen3vl_local/sft_v2/check_loss_mask.py
```

预期输出里 `ok: true`。这表示 `SCENE`、`STATUS`、`SUBGOAL` 的值区间定位正确。若本地
存在 `--model-dir`，脚本还会检查 tokenizer 级别的 0/1 value-token mask；格式 token
不会参与训练。

## 3. 训练

单卡训练：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_v2/train.sh single
```

4 卡 DDP，自动选择空闲 GPU：

```bash
DDP_GPU_COUNT=4 bash qwen3vl_local/sft_v2/train.sh ddp
```

4 卡 DDP，显式指定 GPU：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v2/train.sh ddp
```

轻量链路检查：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_v2/train.sh check
```

常用环境变量：

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `MODEL_DIR` | `checkpoints/Qwen3-VL-4B-Instruct` | 本地模型目录 |
| `TRAIN_JSONL` | `checkpoints/sft_v2_data/train.jsonl` | 训练 jsonl |
| `VAL_JSONL` | `checkpoints/sft_v2_data/val.jsonl` | 验证 jsonl |
| `OUTPUT_DIR` | `checkpoints/sft_v2_lora` | 输出根目录 |
| `MAX_LENGTH` | `8192` | prompt 会包含全部 scene 或单个事件序列 |
| `RUN_TAG` | 时间戳 | 写入 `OUTPUT_DIR/run_<tag>` |
| `NO_RUN_SUBDIR` | `0` | 设为 `1` 时回退到旧的顶层覆盖行为 |
| `GPU_IDS` | 空 | 显式指定 GPU |
| `DDP_GPU_COUNT` | `8` | 自动 DDP 选卡时需要的 GPU 数 |
| `LABEL_WEIGHT` | `1.0` | 值 token loss 权重 |

训练对每个样本跑一次多轮 forward：

1. 图像 + scene prompt，只监督 `SCENE` 值 token。
2. 追加 selected scene 的 status prompt，只监督 `STATUS/SUBGOAL` 值 token。

wrong-scene 增强 row 的 selected scene 会故意不同于 GT scene；监督的 status/subgoal
已经映射为该 selected scene 内合法的同相位事件。

## 4. TensorBoard

训练会在当前 run 下写入 `tb/`，例如：

```text
checkpoints/sft_v2_lora/run_<RUN_TAG>/tb
checkpoints/sft_v2_lora/latest/tb
```

查看最新一次训练：

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_v2_lora/latest/tb
```

查看某个固定 run：

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_v2_lora/run_<RUN_TAG>/tb
```

也可以把 logdir 指到 run 根目录，让 TensorBoard 自动扫描子目录：

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_v2_lora/latest
```

`tb_serve.sh` 会自动选择空闲端口并打印浏览器 URL。常用曲线：

- `train/loss`
- `train/lr`
- `val/loss`
- `val/samples`
- `val/skipped`

若端口需要手动固定：

```bash
TB_PORT=6007 bash qwen3vl_local/tb_serve.sh checkpoints/sft_v2_lora/latest/tb
```

## 5. Eval

快速验证：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v2/eval.py \
  --jsonl checkpoints/sft_v2_data/val.jsonl \
  --lora-dir checkpoints/sft_v2_lora/latest/final \
  --save-root checkpoints/sft_v2_lora/latest \
  --max-samples 100
```

完整 val：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v2/eval.py \
  --jsonl checkpoints/sft_v2_data/val.jsonl \
  --lora-dir checkpoints/sft_v2_lora/latest/final \
  --save-root checkpoints/sft_v2_lora/latest
```

eval 使用真实两阶段协议：

1. 生成 `SCENE`。
2. 如果 `SCENE` 非法，终止该样本。
3. 如果 `SCENE` 合法，用预测 scene 的事件序列追加第二阶段 prompt，并从 scene-step
   KV cache 继续生成 `STATUS/SUBGOAL`。previous-status hint 会映射到预测 scene 的同相位
   事件，保证 prompt 即使在预测 scene 错误时也内部一致。

输出目录：`checkpoints/sft_v2_lora/latest/eval_v2/`

- `metrics.json`
- `scenario_metrics.json`
- `predictions.jsonl`
- `predictions_diff.jsonl`
- `cases/`

`status_accuracy` / `subgoal_accuracy` 是串行指标：scene 也必须正确。
`status_raw_accuracy` / `subgoal_raw_accuracy` 只用于诊断。`valid_total` 和
`*_valid_scene` 指标会排除 invalid-scene row。`status_kv_reuse_rate` 应接近 1.0；
fallback 表示第二阶段不得不重建完整 multi-turn 上下文。

base 对照：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v2/eval.py \
  --lora-dir '' \
  --save-root checkpoints/sft_v2_base_eval \
  --max-samples 100
```

## 6. Case Probe

```bash
GPU_IDS=0 python qwen3vl_local/sft_v2/probe.py \
  --lora-dir checkpoints/sft_v2_lora/latest/final \
  --save-root checkpoints/sft_v2_lora/latest \
  --num-per-scenario 4 --seed 0
```

指定场景：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v2/probe.py \
  --lora-dir checkpoints/sft_v2_lora/latest/final \
  --save-root checkpoints/sft_v2_lora/latest \
  --scenarios Accident,ConstructionObstacle \
  --num-per-scenario 6 --seed 7
```

输出目录：`checkpoints/sft_v2_lora/latest/eval_cases_v2/`。
