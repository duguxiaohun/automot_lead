# LeadMoT 训练运行说明

本手册默认当前目录就是远端 `AutoMoT/`。下面命令都写相对 `AutoMoT/` 的路径，
例如 `bash qwen3vl_local/leadmot/train.sh`，不再额外写切目录步骤。

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
> - `USE_SUBGOAL=0/1`（env，默认 `0`）：是否让 frozen Qwen prefix 额外看到
>   1 张 SUBGOAL keyframe RGB + STATUS/SUBGOAL 真值文本块。它与 `USE_BEV` 正交，
>   不改变 decoder state_dict 形状，但 prefix KV 分布不兼容；resume/init-from-ckpt
>   会按 checkpoint 的 `decoder_config.use_subgoal` 严格拒绝跨开关加载。
>   eval / probe / offline runner 会从 checkpoint 自动恢复该开关；eval_carla 在线闭环
>   暂不支持 `use_subgoal=True` ckpt，会在 agent 加载时直接报错并保留 TODO 接口。

## 1. 构建训练索引

```bash
python qwen3vl_local/leadmot/build_dataset.py \
  --data-root lead_data \
  --output-dir checkpoints/leadmot_v1_data \
  --with-subgoal-fields \
  --samples-per-scenario 0
```

快速抽样调试：

```bash
python qwen3vl_local/leadmot/build_dataset.py \
  --data-root lead_data \
  --output-dir checkpoints/leadmot_v1_data_debug \
  --with-subgoal-fields \
  --samples-per-scenario 50 \
  --stride 5
```

`--with-subgoal-fields` 默认就是开启的，显式写出来只是提醒：构建器会读取
`--keyframes lead_data/keyframes_all_scenarios.json`，给每行写入
`run_id/subgoal_lookup_ok/status/subgoal/subgoal_frame/subgoal_rgb_path/subgoal_skip_reason`。
同一份 jsonl 可同时给 `USE_SUBGOAL=0` 和 `USE_SUBGOAL=1` 使用；前者忽略这些字段，
后者只保留 `subgoal_lookup_ok=True` 的样本训练。只有 keyframes 文件不可用时才用
`--no-with-subgoal-fields`，此时不能训练 `USE_SUBGOAL=1` 模型。
反查 SUBGOAL 时只接受 keyframes run status 为 `Completed/Perfect` 的轨迹；失败或中断
run 会写入 `run_status_not_accepted:*`，避免把不可达未来帧当成真值。

注意：上面的 `--data-root lead_data` 假设当前目录是 `AutoMoT/`，且 `lead_data/`
已经软链接到远端 LEAD 数据内容；输出仍保存在 `checkpoints/...`。本机或其它机器重建
jsonl 时，如果没有这个 keyframes 文件，必须显式传 `--keyframes <有效 keyframes_all_scenarios.json>`；
只是想构建普通 no-subgoal 训练索引时，可传 `--no-with-subgoal-fields` 跳过反查。

`--samples-per-scenario 0` 与 GoalGen 一致，表示每个 scenario 保留所有合法 anchor；传正整数时按 route-balanced 方式抽样。构建器输出 `train.jsonl` / `val.jsonl` / `stats.json`，train/val 按 route 切分，避免同一路线相邻 anchor 同时进入训练和验证。

`--check-readable` 是可选项，**默认不开**：开了之后每个 anchor 要做 6 次 lzma + 12 次 file stat，几百 route 的数据集会变成几小时。train 已经有 DDP-safe 占位 loss 兜底坏样本，不需要在构建期预校验。

## 2. Sanity check

```bash
TRAIN_JSONL=checkpoints/leadmot_v1_data/train.jsonl \
VAL_JSONL=checkpoints/leadmot_v1_data/val.jsonl \
GPU_IDS=0 bash qwen3vl_local/leadmot/train.sh check

# 想固定到某张卡（默认 GPU 0）
GPU_IDS=0 \
TRAIN_JSONL=checkpoints/leadmot_v1_data/train.jsonl \
VAL_JSONL=checkpoints/leadmot_v1_data/val.jsonl \
GPU_IDS=0 bash qwen3vl_local/leadmot/train.sh check
```

`check` 默认只跑 2 个训练 step，不写 TensorBoard，不做验证，用来确认 Qwen prefill、BEV、decoder 和两类轨迹监督全部能接上。

## 3. 单卡训练

```bash
GPU_IDS=0 bash qwen3vl_local/leadmot/train.sh single

# 想固定到某张卡（默认 GPU 0）
GPU_IDS=0 bash qwen3vl_local/leadmot/train.sh single
```

脚本会自动用 `nvidia-smi` 选择显存占用最低的一张卡，并覆盖外层残留的 `CUDA_VISIBLE_DEVICES`。前置 `GPU_IDS=<id>` 时跳过自动选址，直接用指定卡。

常用覆盖：

```bash
LR=1e-4 \
NUM_EPOCHS=4 \
GRAD_ACC=8 \
GPU_IDS=0 bash qwen3vl_local/leadmot/train.sh single

# 想固定到某张卡（默认 GPU 0）
GPU_IDS=0 \
LR=1e-4 \
NUM_EPOCHS=4 \
GRAD_ACC=8 \
GPU_IDS=0 bash qwen3vl_local/leadmot/train.sh single
```

## 4. 多卡 DDP

```bash
DDP_GPU_COUNT=4 bash qwen3vl_local/leadmot/train.sh ddp

# 想固定到指定 4 张卡（默认 GPU 0,1,2,3）
GPU_IDS=0,1,2,3 bash qwen3vl_local/leadmot/train.sh ddp
```

规则：

- 设置 `DDP_GPU_COUNT=N` 时，脚本自动挑 N 张空闲 GPU，并覆盖 `CUDA_VISIBLE_DEVICES`。
- 不设置 `DDP_GPU_COUNT` 时，默认尝试挑 8 张空闲 GPU。
- 前置 `GPU_IDS=0,1,2,3` 时跳过 nvidia-smi 自动选址，直接用指定卡号；`DDP_GPU_COUNT` 此时被忽略，卡数从 `GPU_IDS` 逗号数推断。
- `MASTER_PORT` 未设置时自动找空闲端口；已设置但端口被占用会直接报错。

### 4.z DataLoader 多 worker 预取（GPU 利用率优化）

LeadMoT 单 sample 的 IO 大头是 4 张 JPG 解码 + 4 个 lzma pickle 解压 + 4 个 LAZ 点云读取（可选 +1 张 SUBGOAL JPG），全是主进程 CPU/磁盘等待，GPU 在此期间 idle。

**默认已开启 `NUM_WORKERS=8 / rank`，无需显式指定。** H20 节点上跑下来 `nvidia-smi` GPU util 稳定 90%+，4 卡之间差异控制在 ±10% 内。worker 启动方式默认 `WORKER_MULTIPROCESSING_CONTEXT=spawn`，避免在 Qwen/CUDA 初始化后 fork 子进程。

```bash
# 默认就有 worker 预取，跟原命令一字不变
DDP_GPU_COUNT=4 bash qwen3vl_local/leadmot/train.sh ddp
GPU_IDS=0,1,2,3 bash qwen3vl_local/leadmot/train.sh ddp

# 想再激进（CPU 核数充裕时）
NUM_WORKERS=12 PREFETCH_FACTOR=4 DDP_GPU_COUNT=4 bash qwen3vl_local/leadmot/train.sh ddp
NUM_WORKERS=12 PREFETCH_FACTOR=4 GPU_IDS=0,1,2,3 bash qwen3vl_local/leadmot/train.sh ddp

# Linux 远端想试更低启动开销时，可显式改成 forkserver；确认稳定后再长期使用
WORKER_MULTIPROCESSING_CONTEXT=forkserver DDP_GPU_COUNT=4 bash qwen3vl_local/leadmot/train.sh ddp
WORKER_MULTIPROCESSING_CONTEXT=forkserver GPU_IDS=0,1,2,3 bash qwen3vl_local/leadmot/train.sh ddp

# 想退回同步 IO 做对照 / debug
NUM_WORKERS=0 DDP_GPU_COUNT=4 bash qwen3vl_local/leadmot/train.sh ddp
NUM_WORKERS=0 GPU_IDS=0,1,2,3 bash qwen3vl_local/leadmot/train.sh ddp
```

#### 经验值

- 默认 `NUM_WORKERS=8` / rank：保守安全，H20 节点（~96 物理核）4 卡 / 8 卡 DDP 都顶得住。
- `NUM_WORKERS ≈ 物理 CPU 核数 / 每节点 rank 数` 是上限；超过这个值 worker 之间会争 CPU 反而变慢。
- CPU 内存代价：每 rank ~ `NUM_WORKERS × PREFETCH_FACTOR × clip_size (~50MB)` ≈ 800MB，相对 H20 节点 512GB+ RAM 可忽略。
- `spawn` 是最稳的默认值；`fork` 不推荐，因为 train/eval 都会先初始化 Qwen/CUDA，再创建 DataLoader worker。
- DDP train 会先截掉尾部不足 `world_size` 的样本，再每 rank 取无重复 shard，保证每张卡 backward 次数一致；val/eval 用 `rank::world_size` 手动分片，不用 `DistributedSampler` padding，因此不会重复计入样本。
- `eval.py` 同款 `--num-workers` 默认也是 8；`probe.py` 因数据量极小（默认 24 case）未接 DataLoader。

#### ⚠️ worker 不影响 GPU 显存

`NUM_WORKERS` 是 **CPU 子进程数**，影响的是 GPU 计算利用率（util），**不影响 GPU 显存**：

- 当前 ~14GB / 卡 = frozen Qwen3-VL 4B (~9GB bf16) + LeadBEVEncoder (~150MB) + LeadMoT decoder (~300MB) + B=1 单 sample 激活/KV cache (~4GB)。
- 想吃满显存（>60GB / 卡）必须做真正的 batch_size > 1：
  1. 改 `PrefixKVAttention` 接受 `lang_kv_attention_mask` 处理不同 prefix 长度的 padding；
  2. 改 `forward_sample` 接受 batched samples；
  3. 改 train loop + `grad_accum_steps` 语义。
- 这是另一条 3–5 天的改造路线，与当前 worker 预取正交。当前路线只把 GPU util 拉到 90%+，吞吐受 B=1 上限约束。

### 4.x USE_BEV 开关（v1 内部消融）

```bash
# 默认 USE_BEV=1：v1 完整行为，decoder 在 gen 序列里融合 BEV(120) token
DDP_GPU_COUNT=4 bash qwen3vl_local/leadmot/train.sh ddp
GPU_IDS=0,1,2,3 bash qwen3vl_local/leadmot/train.sh ddp

# USE_BEV=0：消融配置，decoder 不接 BEV 信息，纯靠 Qwen + 自车状态做 planning
DDP_GPU_COUNT=4 USE_BEV=0 bash qwen3vl_local/leadmot/train.sh ddp
GPU_IDS=0,1,2,3 USE_BEV=0 bash qwen3vl_local/leadmot/train.sh ddp
```

行为差异：

| 维度 | `USE_BEV=1`（默认 / v1） | `USE_BEV=0`（消融） |
|---|---|---|
| gen 序列长度 | 142（BEV 120 + status 4 + queries 18） | 22（无 BEV，只 status + queries） |
| BEV encoder forward | 跑（LEAD TransfuserBackbone 全套） | 完全跳过，省一份显存/时间 |
| `decoder.bev_projector` | 实例化 | 不实例化（state_dict 少这一组 key） |
| state_dict 互通 | ❌ 不兼容；**不能跨 USE_BEV 用 `--init-from-ckpt`** | 同左 |
| 适用场景 | 主训练 | 消融对比"语言模型是否够用"，或限显存机器训练 |

切换 `USE_BEV` 必须从头训或单独 warm start，eval / probe 时从 ckpt 里自动读
`use_bev` 字段；`use_final_goal` 必须显式为 true，旧 LeadMoT ckpt 缺该字段会直接报错。与训练时一致即可，不需要重复指定 CLI。接入
`mot_lead_offline_runner.py` 时也同理：先读 checkpoint 配置再实例化 decoder，并用
`strict=True` 加载权重，避免 `USE_BEV=0` checkpoint 被错误装进带随机 `bev_projector` 的模型。

### 4.y USE_SUBGOAL 开关（离线专用）

```bash
# 默认 USE_SUBGOAL=0：普通 history/current RGB + navigation prompt
DDP_GPU_COUNT=4 bash qwen3vl_local/leadmot/train.sh ddp
GPU_IDS=0,1,2,3 bash qwen3vl_local/leadmot/train.sh ddp

# USE_SUBGOAL=1：额外把 SUBGOAL keyframe RGB 和 STATUS/SUBGOAL 真值文本喂给 Qwen prefix
DDP_GPU_COUNT=4 USE_SUBGOAL=1 bash qwen3vl_local/leadmot/train.sh ddp
GPU_IDS=0,1,2,3 USE_SUBGOAL=1 bash qwen3vl_local/leadmot/train.sh ddp

# 可与 USE_BEV=0 组合，做 no-BEV + subgoal prefix 消融
DDP_GPU_COUNT=4 USE_BEV=0 USE_SUBGOAL=1 bash qwen3vl_local/leadmot/train.sh ddp
GPU_IDS=0,1,2,3 USE_BEV=0 USE_SUBGOAL=1 bash qwen3vl_local/leadmot/train.sh ddp
```

行为约定：

| 维度 | `USE_SUBGOAL=0`（默认） | `USE_SUBGOAL=1` |
|---|---|---|
| Qwen 图像输入 | 历史/当前 stitched RGB | 历史/当前 stitched RGB + SUBGOAL stitched RGB |
| Qwen 文本输入 | navigation prompt | `[GROUND_TRUTH_STATE]` + navigation prompt |
| decoder state_dict | 不变 | 不变 |
| prefix KV 分布 | 普通分布 | subgoal-aware 分布，不能与普通分布混用 |
| 数据要求 | 普通 LeadMoT jsonl | jsonl 必须有 `subgoal_lookup_ok=True` 样本 |
| 在线 eval_carla | 支持 | 暂不支持，agent 加载时报 `NotImplementedError` |

训练入口会把 `USE_SUBGOAL=1` 透传成 `train.py --use-subgoal`，并在启动时过滤掉
`subgoal_lookup_ok` 不是 `True` 的行；若过滤后没有训练样本会直接报错。checkpoint 会写入
`decoder_config.use_subgoal`，后续 eval / probe / `mot_lead_offline_runner.py` 自动按 ckpt
选择普通 prefill 或 subgoal prefill，不需要再手动指定。
`eval.py --max-samples N` 会先按 ckpt 的 `use_subgoal` 过滤有效 subgoal 行，再截取前 N 条，
因此快速评测不会被 jsonl 前缀中的无效兼容行污染。

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
USE_BEV=1
USE_SUBGOAL=0
```

只有 LeadMoT decoder 更新参数；Qwen3-VL-Instruct 与 LeadBEVEncoder 都是 frozen eval。不传位置参数时 `bash qwen3vl_local/leadmot/train.sh` 默认走 `ddp`（与 GoalGen 一致）。保存逻辑对齐 GoalGen，共 4 类产物：`best.pt`/`best.json`（val 最优）、`latest.pt`（每 `SAVE_STEPS` 步覆盖 + epoch 末）、`checkpoint-epochNN.pt`（epoch 末池，保留最近 `KEEP_RECENT_CHECKPOINTS` 份）、`step-checkpoint-NNNNNN.pt`（每 `STEP_SAVE_EVERY` 步独立池，保留最近 `KEEP_RECENT_STEP_CHECKPOINTS` 份）。epoch 池与 step 池互不淘汰；`best.pt` / `latest.pt` 永远保留。

DDP 训练中的 validation 会按 `VAL_MAX_SAMPLES` 截断后在所有 rank 间分片，每张卡各跑自己的 Qwen/BEV/decoder forward，再 all-reduce 聚合 loss / ADE / FDE，避免 rank0 串行扫 val、其它 rank 空等。开 EMA 时 val 会跑两遍（raw + EMA），TB 上分别记到 `val/*` 和 `val_ema/*`，best.pt 选 EMA val/loss 作为指标。
val 子集不是固定取 jsonl 头部，而是用 `VAL_SAMPLE_SEED` 从 val 全量中确定性抽样，减少场景分布偏置。
DDP 加载 Qwen 时默认按 `LOCAL_RANK * QWEN_LOAD_STAGGER_S` 错峰，降低多 rank 同时读 4B checkpoint 对共享文件系统的压力。

`IMAGE_LOG_EVERY` 步触发一次 rank0 渲染：从 val 抽 `IMAGE_LOG_SAMPLES` 条画 pred vs gt 拼图贴到 TB（`samples/planning_overlay_raw` 与开 EMA 时的 `samples/planning_overlay_ema`），便于训练过程肉眼看模型质量进化。设 `IMAGE_LOG_EVERY=0` 关闭，check 模式默认关闭。

EMA：默认 `EMA=1` `EMA_DECAY=0.999`，关掉用 `EMA=0`。eval/probe 默认 `--use-ema`，ckpt 不带 EMA 字段时自动回落到 raw 权重；但 ckpt 必须带 `decoder_config.use_final_goal=True`。

## 6. Eval

```bash
GPU_IDS=0 python qwen3vl_local/leadmot/eval.py \
  --jsonl checkpoints/leadmot_v1_data/val.jsonl \
  --save-root checkpoints/leadmot_v1_decoder \
  --max-samples 256
```

多卡分片：

```bash
GPU_IDS=0,1,2,3 torchrun --standalone --nproc_per_node=4 qwen3vl_local/leadmot/eval.py \
  --jsonl checkpoints/leadmot_v1_data/val.jsonl \
  --save-root checkpoints/leadmot_v1_decoder
```

GPU 规则：`torchrun --nproc_per_node=N` 表示需要 N 张 GPU，脚本会自动挑 N 张空闲物理卡；
`GPU_IDS=0,1,2,3` 表示人为指定具体物理卡号，DDP 时数量必须覆盖 `nproc_per_node`。
单进程 `eval.py` / `probe.py` 默认自动挑 1 张空闲卡；想固定卡号时统一前置 `GPU_IDS=0`。

输出：

```text
checkpoints/leadmot_v1_decoder/eval/log.txt
checkpoints/leadmot_v1_decoder/eval/eval_v1_summary.json
checkpoints/leadmot_v1_decoder/eval/eval_v1_perline.jsonl
checkpoints/leadmot_v1_decoder/eval_tb/<ckpt>_<时间戳>/
checkpoints/leadmot_v1_decoder/invocations/*.txt
```

`eval_tb/` 每跑一次 eval 落一个独立 run，可用 `bash qwen3vl_local/tb_serve.sh checkpoints/leadmot_v1_decoder` 把训练曲线和多次 eval 标量叠在同一块 TensorBoard 上对比。

## 7. Probe

```bash
GPU_IDS=0 python qwen3vl_local/leadmot/probe.py \
  --jsonl checkpoints/leadmot_v1_data/val.jsonl \
  --save-root checkpoints/leadmot_v1_decoder \
  --num-per-scenario 2 \
  --max-cases 24
```

`eval.py` / `probe.py` 未显式传 `--checkpoint` 时，会在 `--save-root` 下依次尝试 `best.pt` -> `latest.pt` -> 最新 `step-checkpoint-*.pt` -> 最新 `checkpoint-epoch*.pt`；不传 `--save-root` 则使用默认 `checkpoints/leadmot_v1_decoder`。需要消融或指定中间 ckpt 时再显式传 `--checkpoint`。

`--use-ema` 默认开（与训练侧 EMA on/off 无关，只判断 ckpt 里有没有 `ema_state_dict`）：有就用 EMA shadow，没有就回落 raw 并 print 警告。训练侧当前保存的 EMA schema 是 `{"decay": ..., "shadow": {...}}`；`eval.py` / `probe.py` 会先 unwrap `shadow` 再 strict load。想强制对比 raw 加 `--no-use-ema`。`probe` 会在 `eval_cases/probe_meta.json` 记录本次跑的 `use_ema`，方便 raw vs ema 两次 probe 结果摆在一起比。

每个 case 会写：

```text
checkpoints/leadmot_v1_decoder/eval_cases/log.txt        ← 本次 probe 终端 stdout/stderr（每次 probe 一份，覆盖全部 case）
checkpoints/leadmot_v1_decoder/eval_cases/<case>/
planning_overlay.png
predictions.json
metrics.json
sample.json
overview.md
```

其中 `predictions.json` 会额外写 `navigation_input`，包含本次实际喂给 Qwen/decoder
的 `speed_mps`、`target_point`、`target_point_next`、`final_goal`，方便逐 case 核对
tp/ntp/final_goal 是否和训练分布一致。

## 8. 断点与加载

恢复训练：

```bash
RESUME=checkpoints/leadmot_v1_decoder/latest.pt \
GPU_IDS=0 bash qwen3vl_local/leadmot/train.sh single

GPU_IDS=0 \
RESUME=checkpoints/leadmot_v1_decoder/latest.pt \
GPU_IDS=0 bash qwen3vl_local/leadmot/train.sh single
```

只加载 decoder 权重、重置 optimizer/scheduler：

```bash
INIT_FROM_CKPT=checkpoints/leadmot_v1_decoder/latest.pt \
GPU_IDS=0 bash qwen3vl_local/leadmot/train.sh single

GPU_IDS=0 \
INIT_FROM_CKPT=checkpoints/leadmot_v1_decoder/latest.pt \
GPU_IDS=0 bash qwen3vl_local/leadmot/train.sh single
```

推理 demo 加载：

```bash
GPU_IDS=0 python leaderboard/team_code/mot_lead_offline_runner.py \
  --leadmot-ckpt checkpoints/leadmot_v1_decoder/best.pt
```

如果没有 val 集或还没有 `best.pt`，用 `latest.pt`。

训练输出目录同样会包含 `log.txt` 和 `invocations/`：`log.txt` 是本次终端 stdout/stderr 追加日志，`invocations/` 记录每次 `train.py` 启动时的 argv、关键环境变量和 git commit，方便之后追溯 ckpt 来源。
