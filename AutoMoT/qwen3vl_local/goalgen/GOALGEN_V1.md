# GoalGen v1

本文是 GoalGen v1 的独立说明。v1 和 v2 共享同一套代码入口：
`build_dataset.py` / `train.py` / `train.sh` / `eval.py` / `probe.py`；区别只在数据分布、
训练初始化和 counterfactual 干预范围。

## 版本边界

| 项 | v1 |
|---|---|
| 数据构建 | `--mode v1` |
| 数据目录 | `checkpoints/goalgen_v1_data` |
| 训练产物 | `checkpoints/goalgen_v1_dit` |
| transition | 4 类，含 `initial` / `final` 两端 |
| 默认初始化 | 从零训练 DiT |
| counterfactual 范围 | 默认全状态机，可包含 initial/final 相关转换 |

v1 保留完整事件链：

```text
initial -> middle[0] -> middle[1] -> middle[2] -> final
```

训练样本覆盖 4 段 transition：

```text
initial    -> middle[0]
middle[0]  -> middle[1]
middle[1]  -> middle[2]
middle[2]  -> final
```

## 模型与数据流

GoalGen 在 VAE latent 空间生成未来 subgoal keyframe：

```text
history RGB        -> frozen Qwen prefill -> token-level KV
history RGB        -> frozen VAE          -> z_history
target keyframe RGB -> frozen VAE         -> z1
z0 ~ N(0, I), z_t = (1 - t) * z0 + t * z1
DiT-MoT(z_t, z_history, t, segmented KV) -> v_pred
loss = MSE(v_pred, z1 - z0)
```

共享架构参数：

| 项 | 当前值 |
|---|---|
| VAE latent | `(C=16, T=1, H=48, W=144)` |
| patch | `4`，token 网格 `12*36` |
| hidden / heads | `1024 / 8` |
| DiT layers | `12` |
| Qwen KV | 36 层切 12 段，默认 `select_last`，head_dim=128 |
| z0 prior | `z0_prior_alpha=0.0`, `z0_prior_sigma=1.0`，纯噪声起点 |

Qwen、VAE 默认冻结。patch/unpatch 默认从
`checkpoints/patch_unpatch_v1/latest/weights/patch_unpatch_best.safetensors`
导入并冻结，找不到直接报错，不回退随机初始化。

## 构建数据

构建前会自动剔除异常时长 LEAD route：4Hz 下 `rgb/*.jpg >= 361`
（严格大于 90s）且不在 `BlockedIntersection/ControlLoss` 白名单内的 run
不会进入 train/val；统计写入 `stats.json.skipped_runs`。

```bash
python qwen3vl_local/goalgen/build_dataset.py \
  --mode v1 \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --samples-per-scenario 0
```

产物：

```text
checkpoints/goalgen_v1_data/train.jsonl
checkpoints/goalgen_v1_data/val.jsonl
checkpoints/goalgen_v1_data/stats.json
```

样本语义：`status` 是 anchor 帧真值状态，`subgoal` 是下一事件，
`target_frame` 是该 subgoal 开始发生的 keyframe，且必须满足 `target_frame > anchor`。

## 训练

```bash
# 冒烟
GPU_IDS=0 bash qwen3vl_local/goalgen/train.sh check

# 单卡
GPU_IDS=0 bash qwen3vl_local/goalgen/train.sh single

# 多卡，自动选空闲卡
DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp

# 多卡，显式 pin
GPU_IDS=0,1,2,3 bash qwen3vl_local/goalgen/train.sh ddp
```

常用 env：

| env | v1 默认 |
|---|---|
| `VERSION` | `v1` |
| `OUTPUT_DIR` | `checkpoints/goalgen_v1_dit` |
| `INIT_FROM_CKPT` | 空，从零训练 |
| `RUN_TAG` | 时间戳 |
| `NO_RUN_SUBDIR` | `0` |
| `DDP_GPU_COUNT` | 自动选址时需要的卡数 |
| `GPU_IDS` | 显式 pin 物理卡 |

训练产物：

```text
checkpoints/goalgen_v1_dit/run_<tag>/
checkpoints/goalgen_v1_dit/latest -> run_<tag>
```

`latest/best.pt` 是 eval/probe 默认首选；`latest/latest.pt` 是训练末尾权重。

## Eval

```bash
# 小样本完整 dump
GPU_IDS=0 python qwen3vl_local/goalgen/eval.py \
  --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
  --save-root checkpoints/goalgen_v1_dit \
  --max-samples 100

# 全量多卡分片
GPU_IDS=0,1,2,3 torchrun --standalone --nproc_per_node=4 qwen3vl_local/goalgen/eval.py \
  --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
  --save-root checkpoints/goalgen_v1_dit
```

核心指标：`latent_mse` / `latent_cos` / `pixel_l1` / `psnr` / `velocity_cos`。
最重要的人工检查图是 case 目录里的 `compare.png`：
`target_raw | pred | target_vae_recon`。

## Probe

```bash
GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --version v1 \
  --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
  --save-root checkpoints/goalgen_v1_dit \
  --num-per-scenario 4 --seed 0
```

## Counterfactual 干预

v1 的 counterfactual 可以覆盖完整状态机。默认 `--counterfactual-scope auto`
在 v1 下等价于 `all`，因此候选可以包含 initial/final 相关转换。

```bash
GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --version v1 \
  --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
  --save-root checkpoints/goalgen_v1_dit \
  --scenarios NonSignalizedJunctionLeftTurn,PriorityAtJunction,HazardAtSideLane \
  --num-per-scenario 1 --seed 7 \
  --case-suffix "_v1_cf_default" \
  --counterfactual-mode scenario_swap \
  --counterfactual-config default \
  --counterfactual-seed-replicates 3
```

`scenario_swap` 会同时替换 `scenario/event_sequence/status/subgoal/completed_events`
以保持 prompt 自洽；`subgoal_only` 只替换 SUBGOAL token，并要求
`(STATUS, SUBGOAL)` 至少是某个场景状态机里的合法相邻 pair。

floor 由 truth 自己跨 `z_init` seed 的 pairwise 差异估计；
`--counterfactual-seed-replicates >= 2` 才能计算 ratio。
