# GoalGen v2

本文是 GoalGen v2 的独立说明。v2 仍复用 v1 的同一套代码入口，但数据分布、
训练初始化和 counterfactual 干预范围都按 v2 约束收窄。

## 版本边界

| 项 | v2 |
|---|---|
| 数据构建 | `--mode v2` |
| 数据目录 | `checkpoints/goalgen_v2_data` |
| 训练产物 | `checkpoints/goalgen_v2_dit` |
| transition | 只保留 middle 子目标之间两段 |
| 默认初始化 | 从 `checkpoints/goalgen_v1_dit/latest/best.pt` warm start |
| counterfactual 范围 | 默认 `middle_transitions`，禁止 init/final 干预 |

v2 只保留完整事件链中间的两段：

```text
middle[0] -> middle[1]
middle[1] -> middle[2]
```

显式排除：

```text
initial   -> middle[0]
middle[2] -> final
```

动机：`initial` 视觉上通常还没有任务进度信息，`final` 往往退化成减速/停车；
这两端对“从当前状态生成未来子目标关键帧”的方向信号较弱。v2 聚焦 middle 子目标之间的实质场景演变。

## 模型与数据流

v2 不引入新模型结构，仍使用 GoalGen 共享 DiT-MoT：

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

```bash
python qwen3vl_local/goalgen/build_dataset.py \
  --mode v2 \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --samples-per-scenario 0
```

产物：

```text
checkpoints/goalgen_v2_data/train.jsonl
checkpoints/goalgen_v2_data/val.jsonl
checkpoints/goalgen_v2_data/stats.json
```

v2 数据构建会过滤 `status == "initial"` 和 `subgoal == "final"` 的样本。
训练 / eval / probe 都会校验 v2 数据只能出现 middle 子目标之间的 pair；
如果误把 v1 train/val 喂给 `VERSION=v2`，脚本会直接报错。

## 训练

```bash
# v2 多卡 fine-tune，默认从 v1 latest/best.pt warm start
VERSION=v2 DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp

# 显式 pin
VERSION=v2 GPU_IDS=0,1,2,3 bash qwen3vl_local/goalgen/train.sh ddp

# 指定 warm-start ckpt
VERSION=v2 INIT_FROM_CKPT=checkpoints/goalgen_v1_dit/latest/latest.pt \
  DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp

# 完全从零训 v2（不推荐作默认，只用于 ablation）
VERSION=v2 INIT_FROM_CKPT= DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp
```

常用 env：

| env | v2 默认 |
|---|---|
| `VERSION` | `v2` |
| `OUTPUT_DIR` | `checkpoints/goalgen_v2_dit` |
| `INIT_FROM_CKPT` | `checkpoints/goalgen_v1_dit/latest/best.pt` |
| `RUN_TAG` | 时间戳 |
| `NO_RUN_SUBDIR` | `0` |
| `DDP_GPU_COUNT` | 自动选址时需要的卡数 |
| `GPU_IDS` | 显式 pin 物理卡 |

训练产物：

```text
checkpoints/goalgen_v2_dit/run_<tag>/
checkpoints/goalgen_v2_dit/latest -> run_<tag>
checkpoints/goalgen_v2_dit/run_<tag>/checkpoint-XXXXXX/goalgen_v2.pt
```

v2 warm start 只继承 DiT 权重与 EMA shadow，不继承 optimizer / scheduler / step。
patch/unpatch 会按当前默认路径重新加载并冻结，避免 ckpt 内残留权重改变最终映射。

## Eval

```bash
# 小样本完整 dump
VERSION=v2 GPU_IDS=0 python qwen3vl_local/goalgen/eval.py \
  --save-root checkpoints/goalgen_v2_dit \
  --max-samples 100

# 全量多卡分片
VERSION=v2 GPU_IDS=0,1,2,3 torchrun --standalone --nproc_per_node=4 qwen3vl_local/goalgen/eval.py \
  --save-root checkpoints/goalgen_v2_dit
```

eval 仍没有独立模型分支，但已经支持 `VERSION=v2` / `--version v2`：未显式传
`--val-jsonl` 时会默认使用 `checkpoints/goalgen_v2_data/val.jsonl`，并在加载后校验样本
只能是 `middle[0]->middle[1]` / `middle[1]->middle[2]`。若误传明显的
`goalgen_v1_*` 路径，或数据里出现 initial/final 两端样本，脚本会直接报错。

## Probe

推荐使用 `VERSION=v2`，这样 `probe.py` 会自动使用
`checkpoints/goalgen_v2_data/val.jsonl` 和 `checkpoints/goalgen_v2_dit`。

```bash
VERSION=v2 GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --num-per-scenario 4 --seed 0
```

也可以显式写全路径：

```bash
GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --version v2 \
  --val-jsonl checkpoints/goalgen_v2_data/val.jsonl \
  --save-root checkpoints/goalgen_v2_dit \
  --num-per-scenario 4 --seed 0
```

v2 probe 会校验选中的样本必须是 middle-transition pair。若误传明显的
`goalgen_v1_*` 路径，或选到 initial/final 两端样本时会直接报错，而不是静默产出混合实验。

## Counterfactual 干预

v2 干预实验必须围绕三个 middle 子目标之间的转换设计，不能设计 init/final。
`--counterfactual-scope auto` 在 v2 下等价于 `middle_transitions`：
脚本会拒绝 `--version v2 --counterfactual-scope all`，避免把全状态机干预误用于 v2。
内置 `--counterfactual-config default` 在 v2 下也会自动切成 **v2 middle-only 配置**：
每个 scenario 只把自己的 `middle[1]` / `middle[2]` 作为 CF SUBGOAL 候选，
不会使用 v1/full-scope 的跨语义候选表。

```text
允许：middle[0] -> middle[1]
允许：middle[1] -> middle[2]
禁止：initial -> middle[0]
禁止：middle[2] -> final
```

默认推荐：

```bash
VERSION=v2 GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --scenarios NonSignalizedJunctionLeftTurn,PriorityAtJunction,HazardAtSideLane \
  --num-per-scenario 1 --seed 7 \
  --case-suffix "_v2_cf_default" \
  --counterfactual-mode scenario_swap \
  --counterfactual-config default \
  --counterfactual-seed-replicates 3
```

最小干预：

```bash
VERSION=v2 GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --scenarios NonSignalizedJunctionLeftTurn,SignalizedJunctionLeftTurn \
  --num-per-scenario 1 --seed 7 \
  --case-suffix "_v2_cf_subgoal_only" \
  --counterfactual-mode subgoal_only \
  --counterfactual-config default \
  --counterfactual-seed-replicates 3
```

同 case 对照两种干预强度：

```bash
VERSION=v2 GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --scenarios NonSignalizedJunctionLeftTurn \
  --num-per-scenario 1 --seed 7 \
  --case-suffix "_v2_cf_both" \
  --counterfactual-mode both \
  --counterfactual-config default \
  --counterfactual-seed-replicates 3
```

输出中的 `cf_summary.json` / `cf_report.md` 会写明：

```text
version = v2
counterfactual_scope = middle_transitions
request_source = config
```

任何手动 CLI / 自定义 JSON 中会落到 init/final 两端的 CF 候选都会被 skip，并记录到
`cf_summary.json.modes.<mode>.skipped`；内置 default 本身不会生成这类候选。
