# GoalGen Runbook

本手册默认当前目录就是远端 `AutoMoT/`。下面命令都写相对 `AutoMoT/` 的路径，
例如 `bash qwen3vl_local/goalgen/train.sh`，不再额外写切目录步骤。
GoalGen v1/v2 共用 `qwen3vl_local/goalgen/` 下同一套代码；版本只由 `--mode`
或 `VERSION` env 决定。

## 0. 版本速查

| 项 | v1 | v2 |
|---|---|---|
| 数据构建 | `--mode v1` | `--mode v2` |
| 训练入口 | `DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp` | `VERSION=v2 DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp` |
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
GPU_IDS=0 bash qwen3vl_local/goalgen/train.sh check

# 单卡
GPU_IDS=0 bash qwen3vl_local/goalgen/train.sh single

# DDP 默认自动挑 8 张最空闲 GPU
DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp

# 指定需要几张卡，卡号仍自动挑
DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp

# 想固定到指定卡（单卡默认 GPU 0，多卡默认 GPU 0,1,2,3）
GPU_IDS=0 bash qwen3vl_local/goalgen/train.sh check
GPU_IDS=0 bash qwen3vl_local/goalgen/train.sh single
GPU_IDS=0,1,2,3 bash qwen3vl_local/goalgen/train.sh ddp

# v2：默认从 v1 latest/best.pt warm start
VERSION=v2 DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp
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
VERSION=v2 DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp

# 默认 8 卡 + 指定 pt
VERSION=v2 INIT_FROM_CKPT=checkpoints/goalgen_v1_dit/latest/latest.pt \
    DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp

# 指定 N 卡 + 默认 best.pt（这里 N=4，卡号仍由脚本自动挑最空闲的）
VERSION=v2 DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp

# 指定 N 卡 + 指定 pt
VERSION=v2 DDP_GPU_COUNT=4 INIT_FROM_CKPT=checkpoints/goalgen_v1_dit/latest/latest.pt \
    DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp
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
    GPU_IDS=0,1,2,3 bash qwen3vl_local/goalgen/train.sh ddp
```

## 4. TensorBoard

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/goalgen_v1_dit
```

训练 TB 在每个 run 的 `tb/` 下；eval TB 在 `eval_tb/<run_tag>/` 下。

## 5. 单条前向冒烟

```bash
GPU_IDS=0 python leaderboard/team_code/qwen3vl_dit_goalgen_runner.py \
  --route-dir /datashare/IOL4SGH/data/data/Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46 \
  --anchor 12 \
  --save-root eval_json/qwen3vl_dit_goalgen_smoke
```

用于检查 prompt、Qwen prefill、KV segmentation、VAE latent、DiT forward 和 loss。
数据集训练仍走 `build_dataset.py` + `train.sh`。

## 6. Eval

```bash
# 小样本，完整 dump compare.png
GPU_IDS=0 python qwen3vl_local/goalgen/eval.py \
  --save-root checkpoints/goalgen_v1_dit \
  --max-samples 100

# 全量多卡分片
GPU_IDS=0,1,2,3 torchrun --standalone --nproc_per_node=4 qwen3vl_local/goalgen/eval.py \
  --save-root checkpoints/goalgen_v1_dit

# 绑定具体 ckpt
GPU_IDS=0 python qwen3vl_local/goalgen/eval.py \
  --dit-checkpoint checkpoints/goalgen_v1_dit/run_YYYYmmdd_HHMMSS/best.pt \
  --save-root checkpoints/goalgen_v1_dit/run_YYYYmmdd_HHMMSS \
  --max-samples 100
```

默认自动解析 checkpoint：`<save-root>/latest/best.pt` ->
`latest/latest.pt` -> `best.pt` -> `latest.pt`。显式 `--dit-checkpoint` 时按用户路径。
Eval 的终端输出会追加到 `<save-root>/eval/log.txt`。

Eval / probe GPU 规则与训练入口一致：

- 想指定用几张 GPU：多卡 eval 用 `torchrun --nproc_per_node=N`，脚本按 `WORLD_SIZE=N`
  自动挑 N 张空闲物理卡。
- 想指定用哪几张 GPU：前置 `GPU_IDS=0` 或 `GPU_IDS=0,1,2,3`；脚本跳过自动选址，
  直接把这些物理卡写入 `CUDA_VISIBLE_DEVICES`，进程内从 `cuda:0` 开始编号。
- 都不指定：单进程 eval/probe 自动挑 1 张最空闲物理卡，并覆盖外层残留的
  `CUDA_VISIBLE_DEVICES`。
- 文档命令只保留两种 GPU 写法：单进程显式 pin 用 `GPU_IDS=0`，DDP/torchrun
  显式 pin 用 `GPU_IDS=0,1,2,3`，训练 launcher 指定卡数用 `DDP_GPU_COUNT=N`。

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
GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --save-root checkpoints/goalgen_v1_dit \
  --num-per-scenario 4 --seed 0

GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --dit-checkpoint checkpoints/goalgen_v1_dit/checkpoint-000500/goalgen_v1.pt \
  --save-root checkpoints/goalgen_v1_dit \
  --num-per-scenario 4 --seed 0 --case-suffix "_ckpt500"
```

Probe 适合看单 case 的输入图、目标图、生成图、Euler trace 和指标；终端输出会追加到 `<save-root>/eval_cases/log.txt`。

### 7.1 Counterfactual SUBGOAL 干预实验

**问题**：把 Qwen teacher-forced prompt 里的 SUBGOAL 换成别的 token 后，
GoalGen 解出的子目标图像会不会跟着语义变化？

**核心控制变量**：同一段 history RGB、同一份 target、同一组 `z_init` seed；
只改 Qwen prefill 里的任务语义。

两个 mode：

| mode | 做法 | 适合回答 |
|---|---|---|
| `scenario_swap`（默认） | 同时换 `scenario/event_sequence/status/subgoal/completed_events`，prompt 内部自洽 | 反事实场景语义是否能驱动图像变化 |
| `subgoal_only` | 保留原 scenario / STATUS，只换 SUBGOAL；`(STATUS, SUBGOAL)` 不构成任何场景的合法相邻 pair 时直接跳过；如果本轮候选全被跳过，会按当前 STATUS 自动补最多 3 个合法 SUBGOAL；合法但不在原 EVENT_SEQUENCE 里时打 `prompt_consistency=sequence_mismatch` 警告 | DiT 是否直接读 SUBGOAL token |
| `both` | 同 case 下同时保存两套 | 对照两种干预强度 |

**Floor 概念（v2 修订）**：不再用 `noop` 控制分支（那玩意儿 bit-identical，floor 永远 0）。
改用 truth 自身**跨 `z_init` seed 的两两 pairwise 差异**作为采样噪声 floor：
等价于「换 seed 跑同 prompt 会变多少」。所以 `--counterfactual-seed-replicates 2`
（或更大）才有 floor / ratio；默认 1 时只画图、ratio 字段为 null。

**Verdict 阈值（基于 ratio = ΔCF / floor）**：

| ratio | verdict | 含义 |
|---|---|---|
| `< 2`  | `near_floor`         | 基本是采样噪声 |
| `[2,5)` | `weak_response`     | 模型有反应但弱 |
| `[5,15)` | `responsive`       | 明显跟 SUBGOAL 走 |
| `≥ 15` | `highly_responsive` | 强 SUBGOAL 条件化 |
| n/a    | `insufficient_seeds` | seed_replicates < 2 |

**每个 case 默认只产生 3 个 CF 产物**（精简模式）：

```text
<case_dir>/
  cf_overview_<mode>.png   一张拼图：行=variant，列=CFG sweep；标注 Δpix 和 ratio
  cf_summary.json          含 noise_floor / per_cf delta / ratio / verdict
  cf_report.md             人类可读 markdown，可直接拷给别人看
```

case_root 还会写 `_cf_index<suffix>.jsonl`，每个 case 一行 max_ratio / max_verdict，
方便后续 `jq` 筛 "responsive" 的 case。

加 `--cf-verbose-artifacts` 时才落子目录细产物：

```text
counterfactual/<mode>/<tag>/chat_text.txt
counterfactual/<mode>/<tag>/memory.json
counterfactual/<mode>/<tag>/cfg_<x>/seed_<NN>/pred.png
counterfactual/<mode>/<tag>/cfg_<x>/seed_<NN>/metrics.json
```

---

**Demo A：默认推荐，scenario_swap + 内置 config + 3 seed floor。**

```bash
GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --save-root checkpoints/goalgen_v1_dit \
  --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
  --scenarios NonSignalizedJunctionLeftTurn,PriorityAtJunction,HazardAtSideLane \
  --num-per-scenario 1 --seed 7 \
  --case-suffix "_cf_default" \
  --counterfactual-mode scenario_swap \
  --counterfactual-config default \
  --counterfactual-seed-replicates 3
```

跑完看 `cf_report.md` 一页结论 + `cf_overview_scenario_swap.png` 一张图就够。
ratio ≥ 5 时认为模型对 SUBGOAL 有显著响应。

**Demo B：最小干预 `subgoal_only`，只换 SUBGOAL token。**

```bash
GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --save-root checkpoints/goalgen_v1_dit \
  --scenarios NonSignalizedJunctionLeftTurn,SignalizedJunctionLeftTurn \
  --num-per-scenario 1 --seed 7 \
  --case-suffix "_cf_subgoal_only" \
  --counterfactual-mode subgoal_only \
  --counterfactual-config default \
  --counterfactual-seed-replicates 3
```

非法 `(STATUS, SUBGOAL)` pair 会在 stdout 打 `[probe][cf][skip]` 并记进
`cf_summary.json.modes.subgoal_only.skipped`。如果默认 config 给出的候选全都不匹配当前
STATUS（常见于随机抽到 `STATUS=initial` 的 case），probe 会再打印
`[probe][cf][auto]`，并从当前 STATUS 的合法相邻转移里自动补候选，避免拼图只剩 truth 行。
`cf_overview_*.png` 左侧行标签只保留 variant id 与 `subgoal=...`（scenario_swap 时额外显示 CF scenario）；
候选来源、`prompt_consistency` 与完整 warning 请看 `cf_report.md` 或 `cf_summary.json`。

**Demo C：同 case 下同时跑 scenario_swap 和 subgoal_only。**

```bash
GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --save-root checkpoints/goalgen_v1_dit \
  --scenarios NonSignalizedJunctionLeftTurn \
  --num-per-scenario 1 --seed 7 \
  --case-suffix "_cf_both" \
  --counterfactual-mode both \
  --counterfactual-config default \
  --counterfactual-seed-replicates 3
```

每个 case 会同时产出两张 png：

```text
cf_overview_scenario_swap.png
cf_overview_subgoal_only.png
```

`cf_report.md` 会包含两个 mode 的对照表；`cf_summary.json.modes` 也会有两条 key。

**Demo D：CFG sweep — 用来证明 DiT 真的在用 SUBGOAL KV。**

```bash
GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --save-root checkpoints/goalgen_v1_dit \
  --scenarios NonSignalizedJunctionLeftTurn,PriorityAtJunction,HazardAtSideLane \
  --num-per-scenario 1 --seed 7 \
  --case-suffix "_cf_cfg_sweep" \
  --counterfactual-mode scenario_swap \
  --counterfactual-config default \
  --counterfactual-seed-replicates 3 \
  --cfg-scale-sweep 0.0,1.0,2.0,4.0
```

`cf_overview_scenario_swap.png` 的列数会变成 4，对应 CFG=0/1/2/4。
预期：CFG=0 时 CF 与 truth 几乎一样（near_floor），CFG≥2 时 ratio 显著上升。

**Demo E：用 CLI fallback 而非 config（quick demo）。**

```bash
GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --save-root checkpoints/goalgen_v1_dit \
  --scenarios NonSignalizedJunctionLeftTurn \
  --num-per-scenario 1 --seed 7 \
  --case-suffix "_cf_cli" \
  --counterfactual-mode scenario_swap \
  --counterfactual-subgoals assert_priority,turn_on_green,brake_at_light \
  --counterfactual-seed-replicates 3
```

不传 `--counterfactual-config` 时全场景都用 `--counterfactual-subgoals` 这一组 CF。
适合临时跑一个 case 看看，但不适合做跨场景大批量实验。

**Demo F：保留细产物用于深度排查（性能差但材料全）。**

```bash
GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --save-root checkpoints/goalgen_v1_dit \
  --scenarios NonSignalizedJunctionLeftTurn \
  --num-per-scenario 1 --seed 7 \
  --case-suffix "_cf_verbose" \
  --counterfactual-mode scenario_swap \
  --counterfactual-config default \
  --counterfactual-seed-replicates 3 \
  --cfg-scale-sweep 0.0,2.0 \
  --cf-verbose-artifacts
```

加 `--cf-verbose-artifacts` 后会额外落每个 (mode, variant, cfg, seed) 的 `pred.png`、
`metrics.json`、以及 per-variant 的 `chat_text.txt` / `memory.json`。
平时不用——`cf_overview_<mode>.png` + `cf_report.md` 已经够看；
只有需要逐张图核对 prompt / 排查异常 case 时才开。

---

**内置 per-scenario config 概览**（`--counterfactual-config default`）：

```text
左转  NonSignalizedJunctionLeftTurn / SignalizedJunctionLeftTurn / *EnterFlow
右转  NonSignalizedJunctionRightTurn / SignalizedJunctionRightTurn
路口  PriorityAtJunction / CrossJunctionDefectTrafficLight / T_Junction
汇入  EnterActorFlow / EnterActorFlowV2 / InterurbanActorFlow / InterurbanAdvancedActorFlow
      MergerIntoSlowTraffic / MergerIntoSlowTrafficV2
避障  HazardAtSideLane(TwoWays) / ConstructionObstacle(TwoWays) / ParkedObstacle(TwoWays)
      BlockedIntersection / InvadingTurn / VehicleOpensDoorTwoWays
行人  PedestrianCrossing / DynamicObjectCrossing / ParkingCrossingPedestrian
      CrossingBicycleFlow / VehicleTurningRoute / VehicleTurningRoutePedestrian
事故  Accident(TwoWays) / OppositeVehicleRunningRedLight / OppositeVehicleTakingPriority
高速  HardBreakRoute / HighwayCutIn / HighwayExit / StaticCutIn / ParkingCutIn
红灯  RedLightWithoutLeadVehicle
其它  ControlLoss / ParkingExit
```

每个 scenario 自动配 3 个跨语义簇的 CF subgoal（让行/通行/避让混搭）。
完整内容见 `qwen3vl_local/goalgen/probe.py` 顶部 `DEFAULT_COUNTERFACTUAL_CONFIG`。

未列入的 scenario 命中失败时 stdout 会打 `[probe][cf][warn] scenario=... 不在
counterfactual config 中`；要么把它加进 config，要么显式传 `--counterfactual-subgoals`。

---

**怎么读结果（一分钟看完）**：

1. 打开 `cf_overview_<mode>.png`：
   - 第一行是 truth；下面每行是一个 CF variant；
   - 列数 = CFG sweep 个数（不开 sweep 时只有 1 列）；
   - 每个 cell 上方两三行小字：`dpix=0.087  r=7.3x  responsive`
     （`dpix` 是 pred-vs-truth-pred 像素 L1；ASCII 写法是为了 PIL 默认字体兼容）。

2. 打开 `cf_report.md`，看每个 mode 的表格：

   ```text
   | tag           | subgoal         | scenario(CF)        | Δpixel_l1 | Δlatent_mse | ratio_pix | verdict |
   | cf_01_assert_priority | assert_priority | PriorityAtJunction | 0.087 | 0.0034 | 7.3x | responsive |
   ```

3. 想筛 case：

   ```bash
   jq 'select(.max_verdict=="responsive" or .max_verdict=="highly_responsive")' \
     checkpoints/goalgen_v1_dit/eval_cases/_cf_index*.jsonl
   ```

4. ratio 字段 = null 时：加大 `--counterfactual-seed-replicates`（≥2）才能算 floor。
   `cf_report.md` 顶部 Setup 段会显示当前 baseline cfg / cfg sweep / seed_replicates，
   读 report 时不用回过头查命令行。

5. CF 列对原始 target 的传统 `metrics_vs_gt` 变差是预期的——目标已经被你换了。
   听不听 SUBGOAL 看 ratio + 并排图像的语义方向，**不看原始 target 距离**。

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
| `invalid device ordinal` | 显式 pin 用 `GPU_IDS=0` / `GPU_IDS=0,1,2,3`；指定卡数用 `DDP_GPU_COUNT=N`；DDP 下让 `GPU_IDS` 数量与 `--nproc_per_node` 一致 |
| 生成图像像 VAE 重建上限差 | 先看 `target_vae_recon`，判断是 VAE 上限还是 DiT 失败 |
