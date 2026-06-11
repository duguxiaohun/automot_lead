# GoalGen Runbook

本手册默认当前目录就是远端 `AutoMoT/`。下面命令都写相对 `AutoMoT/` 的路径，
例如 `bash qwen3vl_local/goalgen/train.sh`，不再额外写切目录步骤。
GoalGen v1/v2 共用 `qwen3vl_local/goalgen/` 下同一套代码；版本只由 `--mode`
或 `VERSION` env 决定。

## 0. 版本速查

| 项 | v1 | v2 |
|---|---|---|
| 数据构建 | `--mode v1` | `--mode v2` |
| 训练入口 | `bash qwen3vl_local/goalgen/train.sh ddp` | `VERSION=v2 bash qwen3vl_local/goalgen/train.sh ddp` |
| 数据目录 | `checkpoints/goalgen_v1_data` | `checkpoints/goalgen_v2_data` |
| 产物目录 | `checkpoints/goalgen_v1_dit` | `checkpoints/goalgen_v2_dit` |
| 默认初始化 | 从零 | 从 `goalgen_v1_dit/latest/best.pt` warm start |
| transition | 4 类，含 initial/final 两端 | 2 类，只保留 middle 之间 |

eval/probe 没有版本概念，只看 `--save-root` / `--dit-checkpoint`。

> 显式 pin 卡统一用 `GPU_IDS=...` 前置，例如 `GPU_IDS=0 bash qwen3vl_local/goalgen/train.sh single`、
> `GPU_IDS=0,1,2,3 bash qwen3vl_local/goalgen/train.sh ddp`，详见 §3.2。

## 1. 准备

```bash
git pull
ls checkpoints/Qwen3-VL-4B-Instruct/ | head -5
ls vae_standalone/weights/vae_only.safetensors
ls vae_standalone/config/vae_only.yaml
ls /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json
```

## 2. 构建数据

```bash
# v1
python qwen3vl_local/goalgen/build_dataset.py \
  --mode v1 \
  --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
  --data-root /datashare/IOL4SGH/data/data \
  --samples-per-scenario 0

# v2
python qwen3vl_local/goalgen/build_dataset.py \
  --mode v2 \
  --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
  --data-root /datashare/IOL4SGH/data/data \
  --samples-per-scenario 0
```

样本语义：`status` 是 anchor 帧 GT；`subgoal` 是下一事件；`target_frame`
是该事件 keyframe，必须满足 `target_frame > anchor`。v2 额外去掉
`initial` 和 `final` 两端样本。

产物：

```text
checkpoints/goalgen_v1_data/{train,val}.jsonl + stats.json
checkpoints/goalgen_v2_data/{train,val}.jsonl + stats.json
```

## 3. 训练

```bash
# check：2 个 optimizer step
bash qwen3vl_local/goalgen/train.sh check

# 单卡
bash qwen3vl_local/goalgen/train.sh single

# DDP 默认自动挑 8 张最空闲 GPU
bash qwen3vl_local/goalgen/train.sh ddp

# 指定需要几张卡，卡号仍自动挑
DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp

# 想固定到指定卡（单卡默认 GPU 0，多卡默认 GPU 0,1,2,3）
GPU_IDS=0 bash qwen3vl_local/goalgen/train.sh check
GPU_IDS=0 bash qwen3vl_local/goalgen/train.sh single
GPU_IDS=0,1,2,3 bash qwen3vl_local/goalgen/train.sh ddp

# v2：默认从 v1 latest/best.pt warm start
VERSION=v2 bash qwen3vl_local/goalgen/train.sh ddp
VERSION=v2 GPU_IDS=0,1,2,3 bash qwen3vl_local/goalgen/train.sh ddp
```

常用 env：

| env | 默认 | 说明 |
|---|---:|---|
| `VERSION` | `v1` | `v1` / `v2` |
| `OUTPUT_DIR` | 随版本 | base 输出目录 |
| `RUN_TAG` | 时间戳 | 写到 `OUTPUT_DIR/run_<tag>` |
| `NO_RUN_SUBDIR` | `0` | 置 `1` 回到旧式覆盖写法 |
| `INIT_FROM_CKPT` | v1 空，v2 指向 v1 best | warm start |
| `PATCH_UNPATCH_WEIGHTS` | 空 → 默认 best | 留空时自动取 `checkpoints/patch_unpatch_v1/{latest,run_*}/weights/patch_unpatch_best.safetensors`（详见 §3.0），加载并默认冻结；找不到直接报错；显式传路径则覆盖默认 |
| `PATCH_UNPATCH_UNFREEZE` | `0` | 设 `1` 时外部 patch/unpatch 只作初始化、继续联合训练 |
| `PATCH_UNPATCH_CKPT_FALLBACK` | `0` | warm start 继承外部 patch/unpatch 但 safetensors 缺失时，显式允许使用 ckpt 内自带权重 |
| `DDP_GPU_COUNT` | `8` | DDP 需要的 GPU 数；`GPU_IDS` 非空时忽略 |
| `GPU_IDS` | 空 | 显式 pin 卡号；空 = nvidia-smi 自动选址；`GPU_IDS=0` / `GPU_IDS=0,1,2,3` |
| `COMPILE_DIT` | `1` | 只 compile DiT |
| `GRAD_CKPT` | `1` | per-block gradient checkpointing |

GPU 规则：launcher 默认用 `nvidia-smi` 自动挑空闲卡并覆盖旧 `CUDA_VISIBLE_DEVICES`，
卡数用 `DDP_GPU_COUNT=N` 控制。想固定卡号时前置 `GPU_IDS=0,1,2,3`，跳过自动选址，
卡数从 `GPU_IDS` 逗号数推断（`DDP_GPU_COUNT` 此时不起作用）。

训练产物：

```text
checkpoints/goalgen_v*_dit/run_<tag>/
checkpoints/goalgen_v*_dit/latest -> run_<tag>
```

每个训练 run 目录会追加 `log.txt` 保存本次终端 stdout/stderr。`latest/best.pt` 是 eval/probe 默认首选；`latest/latest.pt` 是训练末尾权重。

### 3.0 patch/unpatch 默认导入约定

- **默认行为（推荐）**：不传 `PATCH_UNPATCH_WEIGHTS`，`goalgen/train.py` 会调
  `qwen3vl_local.goalgen.dit.default_patch_unpatch_weights()` 按以下顺序兜底：
  1. `<AutoMoT>/checkpoints/patch_unpatch_v1/latest/weights/patch_unpatch_best.safetensors`
  2. `<AutoMoT>/checkpoints/patch_unpatch_v1/weights/patch_unpatch_best.safetensors`
     （`NO_RUN_SUBDIR=1` 跑出来的旧式覆盖目录）
  3. `<AutoMoT>/checkpoints/patch_unpatch_v1/run_*/weights/patch_unpatch_best.safetensors`
     中 mtime 最新的一份。
  找到则按 freeze=True 加载，并把绝对路径回填进 `args.patch_unpatch_weights`、
  写进 ckpt 的 `dit_config.patch_unpatch_source=external` / `patch_unpatch_weights`，
  eval/probe/runner 走 `restore_patch_unpatch_from_config()` 时自动重新加载同一份。
  三处默认路径都找不到时，训练入口直接 `FileNotFoundError`；正式 GoalGen 训练不再
  回退到随机 patch/unpatch。
- **显式覆盖**：消融实验里想对比另一份 patch/unpatch 时传
  `PATCH_UNPATCH_WEIGHTS=/path/to/...safetensors`，逻辑跟原来一致；显式路径绝对
  优先于默认解析。
- **VAE 默认权重**：VAE 仍由 `qwen3vl_local.goalgen.vae.default_vae_paths()` 读取
  `vae_standalone/config/vae_only.yaml` 与 `vae_standalone/weights/vae_only.safetensors`；
  patch/unpatch 是额外的 best safetensors 导入，不改变 VAE 权重来源。
- **`PATCH_UNPATCH_UNFREEZE=1`**：仍把 best 权重作为初始化，但继续联合微调；产物
  会被记为 `source=checkpoint`、`weights=""`，eval/probe/runner 走 ckpt 内自带权重。
- **EMA load 顺序**：先完整 `dit_state_dict` 打底，再 `ema_state_dict` 覆盖可训练参数；
  冻结的 patch/unpatch 即使不在 EMA shadow 里，strict load 也不缺 key。
- **warm start 行为**：
  - 默认（v2 从 v1 best.pt warm-start）：v2 `train.py` 先按默认路径加载 patch/unpatch
    (上面 §3.0)，再 `dit.load_state_dict(warm_ckpt["dit_state_dict"], strict=True)`，
    然后**再次**用 §3.0 解析路径覆盖一次 patch/unpatch；保证 best.pt 里残留的同一份
    patch/unpatch 不会改变最终落到 DiT 的权重。
  - warm ckpt 自带 `patch_unpatch_source=external` 但本机找不到 safetensors（跨机器
    迁移 + 没有重新跑 train_patch_unpatch.py）时，显式
    `PATCH_UNPATCH_CKPT_FALLBACK=1` 允许使用 ckpt 内已保存的 patch/unpatch，
    并把新产物记为 `source=checkpoint`。

### 3.1 v2 多卡启动（自动选址）

下面这一组示例都让脚本自动挑空闲卡。需要固定卡号见 §3.2。

```bash
# 默认 8 卡 + 默认 best.pt（自动从 checkpoints/goalgen_v1_dit/latest/best.pt warm start）
VERSION=v2 bash qwen3vl_local/goalgen/train.sh ddp

# 默认 8 卡 + 指定 pt
VERSION=v2 INIT_FROM_CKPT=checkpoints/goalgen_v1_dit/latest/latest.pt \
    bash qwen3vl_local/goalgen/train.sh ddp

# 指定 N 卡 + 默认 best.pt（这里 N=4，卡号仍由脚本自动挑最空闲的）
VERSION=v2 DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp

# 指定 N 卡 + 指定 pt
VERSION=v2 DDP_GPU_COUNT=4 INIT_FROM_CKPT=checkpoints/goalgen_v1_dit/latest/latest.pt \
    bash qwen3vl_local/goalgen/train.sh ddp
```

### 3.2 显式 pin 卡（GPU_IDS）

`GPU_IDS` 非空时脚本跳过 nvidia-smi 自动选址，直接把指定卡号写进 `CUDA_VISIBLE_DEVICES`；
DDP 卡数从 `GPU_IDS` 逗号数推断，`DDP_GPU_COUNT` 此时被忽略。

```bash
# 单卡 pin（默认 GPU 0）
GPU_IDS=0 bash qwen3vl_local/goalgen/train.sh single
GPU_IDS=0 bash qwen3vl_local/goalgen/train.sh check

# 4 卡 DDP pin（默认 GPU 0,1,2,3），v1 / v2 同写法
GPU_IDS=0,1,2,3 bash qwen3vl_local/goalgen/train.sh ddp
VERSION=v2 GPU_IDS=0,1,2,3 bash qwen3vl_local/goalgen/train.sh ddp

# 4 卡 DDP pin + 指定 pt（与 INIT_FROM_CKPT 等其它 env 自由叠加）
VERSION=v2 GPU_IDS=0,1,2,3 \
    INIT_FROM_CKPT=checkpoints/goalgen_v1_dit/latest/latest.pt \
    bash qwen3vl_local/goalgen/train.sh ddp
```

## 4. TensorBoard

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/goalgen_v1_dit
```

训练 TB 在每个 run 的 `tb/` 下；eval TB 在 `eval_tb/<run_tag>/` 下。

## 5. 单条前向冒烟

```bash
python leaderboard/team_code/qwen3vl_dit_goalgen_runner.py \
  --route-dir /datashare/IOL4SGH/data/data/Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46 \
  --anchor 12 \
  --save-root eval_json/qwen3vl_dit_goalgen_smoke
```

用于检查 prompt、Qwen prefill、KV segmentation、VAE latent、DiT forward 和 loss。
数据集训练仍走 `build_dataset.py` + `train.sh`。

## 6. Eval

```bash
# 小样本，完整 dump compare.png
python qwen3vl_local/goalgen/eval.py \
  --save-root checkpoints/goalgen_v1_dit \
  --max-samples 100

# 全量多卡分片
torchrun --standalone --nproc_per_node=4 qwen3vl_local/goalgen/eval.py \
  --save-root checkpoints/goalgen_v1_dit

# 绑定具体 ckpt
python qwen3vl_local/goalgen/eval.py \
  --dit-checkpoint checkpoints/goalgen_v1_dit/run_YYYYmmdd_HHMMSS/best.pt \
  --save-root checkpoints/goalgen_v1_dit/run_YYYYmmdd_HHMMSS \
  --max-samples 100
```

默认自动解析 checkpoint：`<save-root>/latest/best.pt` ->
`latest/latest.pt` -> `best.pt` -> `latest.pt`。显式 `--dit-checkpoint` 时按用户路径。
Eval 的终端输出会追加到 `<save-root>/eval/log.txt`。

`--gpu` 语义：默认保持 `0`，脚本自动挑物理卡并映射为 `cuda:0`；单进程显式
传 `--gpu N` 时不覆盖 `CUDA_VISIBLE_DEVICES`。DDP 下按 `LOCAL_RANK`。

关键指标：

| 指标 | 含义 |
|---|---|
| `latent_mse` / `latent_cos` | latent 预测质量 |
| `pixel_l1` / `psnr` | 解码图像质量 |
| `velocity_cos` | flow velocity 方向是否对 |

最重要的可视化是 case 目录里的 `compare.png`：
`target_raw | pred | target_vae_recon`。

## 7. Probe

```bash
python qwen3vl_local/goalgen/probe.py \
  --save-root checkpoints/goalgen_v1_dit \
  --num-per-scenario 4 --seed 0

python qwen3vl_local/goalgen/probe.py \
  --dit-checkpoint checkpoints/goalgen_v1_dit/checkpoint-000500/goalgen_v1.pt \
  --save-root checkpoints/goalgen_v1_dit \
  --num-per-scenario 4 --seed 0 --case-suffix "_ckpt500"
```

Probe 适合看单 case 的输入图、目标图、生成图、Euler trace 和指标；终端输出会追加到 `<save-root>/eval_cases/log.txt`。

### 7.1 Counterfactual SUBGOAL 干预实验

这个实验回答一个问题：**GoalGen 生成的未来 subgoal 图像是否真的会跟随
teacher-forced prompt 里的 SUBGOAL 改变？** 每个 case 固定同一段 history RGB、
同一份 target RGB、同一组 `z_init` seed，只改 Qwen prefill prompt 里的任务语义。

有两个 mode：

| mode | 做法 | 适合回答 |
|---|---|---|
| `scenario_swap`（默认） | 同时替换 `scenario/scenario_label/event_sequence/status/subgoal/completed_events`，保证 prompt 内部自洽 | 反事实场景语义是否能驱动图像变化 |
| `subgoal_only` | 保留原 scenario / STATUS，只替换 SUBGOAL；若 `(STATUS, SUBGOAL)` 不是任何场景里的合法相邻 pair 会跳过 | DiT 是否直接读 SUBGOAL token；注意 prompt 可能出现 sequence mismatch |
| `both` | 同一 case 下同时保存两套目录 | 对照两种干预强度 |

每个启用 counterfactual 的 case 都会自动加入 `noop_<truth_subgoal>` 控制实验：
noop 使用和 truth 完全一样的 SUBGOAL，但仍完整跑一次 Qwen prefill + DiT。理论上
noop 与 truth 的 pred-vs-pred 差异应接近 0；`counterfactual_metrics_summary.json`
会把这个数写成 `noop_floor`，后续 CF 的 delta 会给出 `ratio_over_noop_floor_*`。
经验上 ratio > 5 才值得说“SUBGOAL 可能显著影响输出”，最后仍以并排图为准。

**Demo A：默认推荐，左转 case 做 prompt 自洽的 scenario_swap。**

```bash
python qwen3vl_local/goalgen/probe.py \
  --save-root checkpoints/goalgen_v1_dit \
  --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
  --scenarios NonSignalizedJunctionLeftTurn,SignalizedJunctionLeftTurn \
  --num-per-scenario 1 --seed 7 \
  --case-suffix "_demo_leftturn_swap" \
  --counterfactual-mode scenario_swap \
  --counterfactual-subgoals assert_priority,turn_on_green,brake_at_light,proceed_resume
```

这里会把左转 clip “假装成”能合法到达 `assert_priority` / `turn_on_green`
等 SUBGOAL 的场景状态机，prompt 不自相矛盾。

**Demo B：最小干预 subgoal_only，用来测 SUBGOAL token 本身。**

```bash
python qwen3vl_local/goalgen/probe.py \
  --save-root checkpoints/goalgen_v1_dit \
  --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
  --scenarios NonSignalizedJunctionLeftTurn,SignalizedJunctionLeftTurn \
  --num-per-scenario 1 --seed 7 \
  --case-suffix "_demo_leftturn_subgoal_only" \
  --counterfactual-mode subgoal_only \
  --counterfactual-subgoals assert_priority,turn_on_green,brake_at_light,proceed_resume
```

`subgoal_only` 会跳过所有与当前 STATUS 不构成合法 pair 的 CF，并把跳过原因写入
`counterfactual_summary.json/skipped_variants`。若 SUBGOAL 不在原始
`EVENT_SEQUENCE` 里，但 `(STATUS, SUBGOAL)` 在别的场景中合法，会保留并标记
`prompt_consistency=sequence_mismatch`。

**Demo C：同一 case 同时保存两种 mode。**

```bash
python qwen3vl_local/goalgen/probe.py \
  --save-root checkpoints/goalgen_v1_dit \
  --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
  --scenarios NonSignalizedJunctionLeftTurn \
  --num-per-scenario 1 --seed 7 \
  --case-suffix "_demo_both_modes" \
  --counterfactual-mode both \
  --counterfactual-subgoals assert_priority,turn_on_green,brake_at_light
```

输出中会同时有：

```text
counterfactual/scenario_swap/...
counterfactual/subgoal_only/...
counterfactual_compare_scenario_swap.png
counterfactual_compare_subgoal_only.png
```

**Demo D：标准跨场景实验，使用内置 per-scenario 配置。**

```bash
python qwen3vl_local/goalgen/probe.py \
  --save-root checkpoints/goalgen_v1_dit \
  --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
  --scenarios NonSignalizedJunctionLeftTurn,SignalizedJunctionLeftTurn,NonSignalizedJunctionRightTurn,SignalizedJunctionRightTurn,PriorityAtJunction,EnterActorFlow,HazardAtSideLane,PedestrianCrossing,Accident \
  --num-per-scenario 2 --seed 7 \
  --case-suffix "_cf_default" \
  --counterfactual-mode scenario_swap \
  --counterfactual-config default
```

内置配置等价于：

```json
{
  "NonSignalizedJunctionLeftTurn": {"swap_in_subgoals": ["assert_priority", "turn_on_green", "brake_at_light"]},
  "NonSignalizedJunctionRightTurn": {"swap_in_subgoals": ["yield_and_turn", "brake_at_light", "assert_priority"]},
  "SignalizedJunctionLeftTurn": {"swap_in_subgoals": ["assert_priority", "turn_on_green", "brake_at_light"]},
  "SignalizedJunctionRightTurn": {"swap_in_subgoals": ["yield_and_turn", "brake_at_light", "assert_priority"]},
  "PriorityAtJunction": {"swap_in_subgoals": ["brake_at_light", "yield_and_turn", "wait_or_turn_on_green"]},
  "EnterActorFlow": {"swap_in_subgoals": ["yield_and_turn", "max_brake_or_min_gap", "passing_hazard"]},
  "HazardAtSideLane": {"swap_in_subgoals": ["gap_accept_merge", "yield_and_turn", "max_brake_or_min_gap"]},
  "PedestrianCrossing": {"swap_in_subgoals": ["assert_priority", "proceed_resume", "turn_on_green"]},
  "Accident": {"swap_in_subgoals": ["assert_priority", "proceed_resume", "gap_accept_merge"]}
}
```

也可以把同结构 JSON 存成文件，然后传 `--counterfactual-config path/to/cf.json`。
若当前 scenario 命中 config，优先用 config；否则回退到 `--counterfactual-subgoals`。

**Demo E：CFG sweep 和 z_init seed replicates。**

```bash
python qwen3vl_local/goalgen/probe.py \
  --save-root checkpoints/goalgen_v1_dit \
  --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
  --scenarios NonSignalizedJunctionLeftTurn,PriorityAtJunction,HazardAtSideLane \
  --num-per-scenario 1 --seed 7 \
  --case-suffix "_cf_sweep_rep3" \
  --counterfactual-mode scenario_swap \
  --counterfactual-config default \
  --cfg-scale-sweep 0.0,1.0,2.0,4.0 \
  --counterfactual-seed-replicates 3
```

如果 CFG=0 时 CF 基本无效，而 CFG=2/4 时 pred-vs-truth-pred delta 明显变大，
说明 DiT 的 conditional branch 确实在使用 SUBGOAL 相关 KV。seed replicates 用来确认
这种变化不是单个 `z_init` 的偶然采样。

每个 case 的关键输出：

```text
<save-root>/eval_cases/<case>/
  counterfactual_compare_<mode>.png
  counterfactual_summary.json
  counterfactual_metrics_summary.json
  counterfactual/<mode>/<truth|noop|cf_XX_event>/chat_text.txt
  counterfactual/<mode>/<truth|noop|cf_XX_event>/memory.json
  counterfactual/<mode>/<truth|noop|cf_XX_event>/cfg_<scale>/seed_<NN>/pred.png
  counterfactual/<mode>/<truth|noop|cf_XX_event>/cfg_<scale>/seed_<NN>/metrics_vs_original_target.json
  meta.json
  overview.md
```

读图优先看 `counterfactual_compare_<mode>.png`：第一列是真值目标帧，第二列是
truth SUBGOAL 生成图，第三列是 noop，后面是人工 CF。数值优先看
`counterfactual_metrics_summary.json`：

- `noop_floor.latent_mse/pixel_l1`：数值/采样地板。
- `per_cf[].delta_*_vs_truth_pred`：CF pred 相对 truth pred 的变化。
- `ratio_over_noop_floor_latent/pixel`：变化量相对 noop floor 的倍数。
- `per_cf[].per_cfg`：CFG sweep 曲线；每个 CFG 单独统计 delta 和 noop floor ratio。

注意：CF 分支的 `metrics_vs_original_target.json` 仍然是相对原始 truth target 计算；
故意换目标时它变差不代表模型错。真正判断“听不听 SUBGOAL”看 pred-vs-pred delta、
ratio 和并排图像的语义方向。`chat_text.txt` 用来核对 Qwen 实际收到的 prompt。

## 8. 默认形状

| 项 | 当前共享架构 |
|---|---|
| VAE latent | `(C=16, T=1, H=48, W=144)` |
| patch | `4`，token 数 `12*36=432` |
| hidden / heads | `1024 / 8` |
| DiT layers | `12` |
| Qwen KV | 36 层切 12 段，head_dim=128 |

这套是当前 v1/v2 共享架构；不要把它描述成某个 dataset mode 专属架构。

## 9. 常见问题

| 现象 | 处理 |
|---|---|
| `target_frame must be in the future` | 目标帧不在 anchor 之后；重建数据或换 anchor |
| patch/unpatch key 缺失 | 确认权重来自 `vae_standalone/train_patch_unpatch.py` |
| `invalid device ordinal` | 训练入口锁卡用 `GPU_IDS=0` / `GPU_IDS=0,1,2,3`；eval/probe 的 `--gpu N` 只锁进程内可见 GPU 编号 |
| 生成图像像 VAE 重建上限差 | 先看 `target_vae_recon`，判断是 VAE 上限还是 DiT 失败 |
