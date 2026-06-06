# LeadMoT 训练运行说明

> **版本说明**：本子包目前**只有 v1**（一套完整的 frozen Qwen prefix K/V + LEAD BEV
> + planning decoder 训练栈），与 GoalGen 不同，**没有 v2 数据分布裁剪**。命名风格
> 与 GoalGen 对齐：`build_dataset.py` / `train.py` / `train.sh` / `eval.py` / `probe.py`，
> 未来扩展 v2 时复用同一份脚本，靠 `--mode` / `VERSION` env 切换（同 SFT/GoalGen 套路）。
>
> **现有可控开关（v1 内部）**：
> - `USE_BEV=0/1`（env，默认 `1`）：是否让 decoder 在 gen 序列里融合 BEV(120) token。
>   `USE_BEV=0` 时 decoder 完全靠 frozen Qwen prefix K/V + 自车状态做 planning，
>   BEV encoder 整个 forward 也会跳过；适合做"语言模型是否够用"的消融。**注意切换
>   `USE_BEV` 会让 `decoder.bev_projector` 子模块存在性变化，state_dict 不兼容**，
>   不能跨 `USE_BEV` 用 `--init-from-ckpt`，必须从头训或单独 warm start。
>   eval / probe / runner 会从 checkpoint 的 `decoder_config.use_bev` 自动恢复开关；
>   runner 加载 decoder 权重使用 `strict=True`。`USE_BEV=1` 必须导入已有
>   `bev_projector` 参数，`USE_BEV=0` 则完全不实例化 / 不 forward BEV，绝不混入随机 BEV。

## 1. 构建训练索引

```bash
cd AutoMoT
python qwen3vl_local/leadmot/build_dataset.py \
  --data-root /datashare/IOL4SGH/data/data \
  --output-dir checkpoints/leadmot_v1_data \
  --samples-per-scenario 0
```

快速抽样调试：

```bash
python qwen3vl_local/leadmot/build_dataset.py \
  --data-root /datashare/IOL4SGH/data/data \
  --output-dir checkpoints/leadmot_v1_data_debug \
  --samples-per-scenario 50 \
  --stride 5
```

`--samples-per-scenario 0` 与 GoalGen 一致，表示每个 scenario 保留所有合法 anchor；传正整数时按 route-balanced 方式抽样。构建器输出 `train.jsonl` / `val.jsonl` / `stats.json`，train/val 按 route 切分，避免同一路线相邻 anchor 同时进入训练和验证。

`--check-readable` 是可选项，**默认不开**：开了之后每个 anchor 要做 6 次 lzma + 12 次 file stat，几百 route 的数据集会变成几小时。train 已经有 DDP-safe 占位 loss 兜底坏样本，不需要在构建期预校验。

## 2. Sanity check

```bash
cd AutoMoT
TRAIN_JSONL=checkpoints/leadmot_v1_data/train.jsonl \
VAL_JSONL=checkpoints/leadmot_v1_data/val.jsonl \
bash qwen3vl_local/leadmot/train.sh check
```

`check` 默认只跑 2 个训练 step，不写 TensorBoard，不做验证，用来确认 Qwen prefill、BEV、decoder 和两类轨迹监督全部能接上。

## 3. 单卡训练

```bash
cd AutoMoT
bash qwen3vl_local/leadmot/train.sh single
```

脚本会自动用 `nvidia-smi` 选择显存占用最低的一张卡，并覆盖外层残留的 `CUDA_VISIBLE_DEVICES`。

常用覆盖：

```bash
LR=1e-4 \
NUM_EPOCHS=4 \
GRAD_ACC=8 \
bash qwen3vl_local/leadmot/train.sh single
```

## 4. 多卡 DDP

```bash
cd AutoMoT
DDP_GPU_COUNT=4 bash qwen3vl_local/leadmot/train.sh ddp
```

规则：

- 设置 `DDP_GPU_COUNT=N` 时，脚本自动挑 N 张空闲 GPU，并覆盖 `CUDA_VISIBLE_DEVICES`。
- 不设置 `DDP_GPU_COUNT` 时，默认尝试挑 8 张空闲 GPU。
- `MASTER_PORT` 未设置时自动找空闲端口；已设置但端口被占用会直接报错。

### 4.x USE_BEV 开关（v1 内部消融）

```bash
# 默认 USE_BEV=1：v1 完整行为，decoder 在 gen 序列里融合 BEV(120) token
DDP_GPU_COUNT=4 bash qwen3vl_local/leadmot/train.sh ddp

# USE_BEV=0：消融配置，decoder 不接 BEV 信息，纯靠 Qwen + 自车状态做 planning
DDP_GPU_COUNT=4 USE_BEV=0 bash qwen3vl_local/leadmot/train.sh ddp
```

行为差异：

| 维度 | `USE_BEV=1`（默认 / v1） | `USE_BEV=0`（消融） |
|---|---|---|
| gen 序列长度 | 141（BEV 120 + status 3 + queries 18） | 21（无 BEV，只 status + queries） |
| BEV encoder forward | 跑（LEAD TransfuserBackbone 全套） | 完全跳过，省一份显存/时间 |
| `decoder.bev_projector` | 实例化 | 不实例化（state_dict 少这一组 key） |
| state_dict 互通 | ❌ 不兼容；**不能跨 USE_BEV 用 `--init-from-ckpt`** | 同左 |
| 适用场景 | 主训练 | 消融对比"语言模型是否够用"，或限显存机器训练 |

切换 `USE_BEV` 必须从头训或单独 warm start，eval / probe 时从 ckpt 里自动读
`use_bev` 字段；旧 ckpt 缺字段时按 `bev_projector.*` key 推断。与训练时一致即可，不需要重复指定 CLI。接入
`mot_lead_offline_runner.py` 时也同理：先读 checkpoint 配置再实例化 decoder，并用
`strict=True` 加载权重，避免 `USE_BEV=0` checkpoint 被错误装进带随机 `bev_projector` 的模型。

## 5. 默认训练参数

```text
LR=2e-4
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.05
NUM_EPOCHS=3
GRAD_ACC=8
ROUTE_LOSS_WEIGHT=0.5
WAYPOINT_LOSS_WEIGHT=1.0
LOSS_TYPE=l1
LEADMOT_ROPE_TYPE=mrope
DECODER_DTYPE=bfloat16
QWEN_DTYPE=bfloat16
QWEN_LOAD_STAGGER_S=2.0
SAVE_STEPS=500
KEEP_RECENT_CHECKPOINTS=3
STEP_SAVE_EVERY=10000
KEEP_RECENT_STEP_CHECKPOINTS=3
VAL_STEPS=500
VAL_MAX_SAMPLES=64
VAL_SAMPLE_SEED=202607
DECODER_DROPOUT=0.1
EMA=1
EMA_DECAY=0.999
IMAGE_LOG_EVERY=1000
IMAGE_LOG_SAMPLES=4
IMAGE_LOG_SEED=20260101
```

只有 LeadMoT decoder 更新参数；Qwen3-VL-Instruct 与 LeadBEVEncoder 都是 frozen eval。不传位置参数时 `bash qwen3vl_local/leadmot/train.sh` 默认走 `ddp`（与 GoalGen 一致）。保存逻辑对齐 GoalGen，共 4 类产物：`best.pt`/`best.json`（val 最优）、`latest.pt`（每 `SAVE_STEPS` 步覆盖 + epoch 末）、`checkpoint-epochNN.pt`（epoch 末池，保留最近 `KEEP_RECENT_CHECKPOINTS` 份）、`step-checkpoint-NNNNNN.pt`（每 `STEP_SAVE_EVERY` 步独立池，保留最近 `KEEP_RECENT_STEP_CHECKPOINTS` 份）。epoch 池与 step 池互不淘汰；`best.pt` / `latest.pt` 永远保留。

DDP 训练中的 validation 会按 `VAL_MAX_SAMPLES` 截断后在所有 rank 间分片，每张卡各跑自己的 Qwen/BEV/decoder forward，再 all-reduce 聚合 loss / ADE / FDE，避免 rank0 串行扫 val、其它 rank 空等。开 EMA 时 val 会跑两遍（raw + EMA），TB 上分别记到 `val/*` 和 `val_ema/*`，best.pt 选 EMA val/loss 作为指标。
val 子集不是固定取 jsonl 头部，而是用 `VAL_SAMPLE_SEED` 从 val 全量中确定性抽样，减少场景分布偏置。
DDP 加载 Qwen 时默认按 `LOCAL_RANK * QWEN_LOAD_STAGGER_S` 错峰，降低多 rank 同时读 4B checkpoint 对共享文件系统的压力。

`IMAGE_LOG_EVERY` 步触发一次 rank0 渲染：从 val 抽 `IMAGE_LOG_SAMPLES` 条画 pred vs gt 拼图贴到 TB（`samples/planning_overlay_raw` 与开 EMA 时的 `samples/planning_overlay_ema`），便于训练过程肉眼看模型质量进化。设 `IMAGE_LOG_EVERY=0` 关闭，check 模式默认关闭。

EMA：默认 `EMA=1` `EMA_DECAY=0.999`，关掉用 `EMA=0`。eval/probe 默认 `--use-ema`，旧 ckpt 不带 EMA 字段时自动回落到 raw 权重。

## 6. Eval

```bash
cd AutoMoT
python qwen3vl_local/leadmot/eval.py \
  --jsonl checkpoints/leadmot_v1_data/val.jsonl \
  --save-root checkpoints/leadmot_v1_decoder \
  --max-samples 256
```

多卡分片：

```bash
torchrun --standalone --nproc_per_node=4 qwen3vl_local/leadmot/eval.py \
  --jsonl checkpoints/leadmot_v1_data/val.jsonl \
  --save-root checkpoints/leadmot_v1_decoder
```

输出：

```text
checkpoints/leadmot_v1_decoder/eval/eval_v1_summary.json
checkpoints/leadmot_v1_decoder/eval/eval_v1_perline.jsonl
checkpoints/leadmot_v1_decoder/eval_tb/<ckpt>_<时间戳>/
checkpoints/leadmot_v1_decoder/invocations/*.txt
```

`eval_tb/` 每跑一次 eval 落一个独立 run，可用 `bash tools/tb_serve.sh checkpoints/leadmot_v1_decoder` 把训练曲线和多次 eval 标量叠在同一块 TensorBoard 上对比。

## 7. Probe

```bash
cd AutoMoT
python qwen3vl_local/leadmot/probe.py \
  --jsonl checkpoints/leadmot_v1_data/val.jsonl \
  --save-root checkpoints/leadmot_v1_decoder \
  --num-per-scenario 2 \
  --max-cases 24
```

`eval.py` / `probe.py` 未显式传 `--checkpoint` 时，会在 `--save-root` 下依次尝试 `best.pt` -> `latest.pt` -> 最新 `step-checkpoint-*.pt` -> 最新 `checkpoint-epoch*.pt`；不传 `--save-root` 则使用默认 `checkpoints/leadmot_v1_decoder`。需要消融或指定中间 ckpt 时再显式传 `--checkpoint`。

`--use-ema` 默认开（与训练侧 EMA on/off 无关，只判断 ckpt 里有没有 `ema_state_dict`）：有就用 EMA shadow，没有就回落 raw 并 print 警告。训练侧当前保存的 EMA schema 是 `{"decay": ..., "shadow": {...}}`；`eval.py` / `probe.py` 会先 unwrap `shadow` 再 strict load。想强制对比 raw 加 `--no-use-ema`。`probe` 会在 `eval_cases/probe_meta.json` 记录本次跑的 `use_ema`，方便 raw vs ema 两次 probe 结果摆在一起比。

每个 case 会写：

```text
checkpoints/leadmot_v1_decoder/eval_cases/<case>/
planning_overlay.png
predictions.json
metrics.json
sample.json
overview.md
```

## 8. 断点与加载

恢复训练：

```bash
RESUME=checkpoints/leadmot_v1_decoder/latest.pt \
bash qwen3vl_local/leadmot/train.sh single
```

只加载 decoder 权重、重置 optimizer/scheduler：

```bash
INIT_FROM_CKPT=checkpoints/leadmot_v1_decoder/latest.pt \
bash qwen3vl_local/leadmot/train.sh single
```

推理 demo 加载：

```bash
python leaderboard/team_code/mot_lead_offline_runner.py \
  --leadmot-ckpt checkpoints/leadmot_v1_decoder/best.pt
```

如果没有 val 集或还没有 `best.pt`，用 `latest.pt`。

训练输出目录同样会包含 `invocations/`，记录每次 `train.py` 启动时的 argv、关键环境变量和 git commit，方便之后追溯 ckpt 来源。
