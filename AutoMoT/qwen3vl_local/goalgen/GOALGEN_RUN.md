# GoalGen Runbook

所有命令默认在远端 `AutoMoT/` 目录执行。GoalGen v1/v2 共用
`qwen3vl_local/goalgen/` 下同一套代码；版本只由 `--mode` 或 `VERSION` env 决定。

## 0. 版本速查

| 项 | v1 | v2 |
|---|---|---|
| 数据构建 | `--mode v1` | `--mode v2` |
| 训练入口 | `bash train.sh ddp` | `VERSION=v2 bash train.sh ddp` |
| 数据目录 | `checkpoints/goalgen_v1_data` | `checkpoints/goalgen_v2_data` |
| 产物目录 | `checkpoints/goalgen_v1_dit` | `checkpoints/goalgen_v2_dit` |
| 默认初始化 | 从零 | 从 `goalgen_v1_dit/latest/best.pt` warm start |
| transition | 4 类，含 initial/final 两端 | 2 类，只保留 middle 之间 |

eval/probe 没有版本概念，只看 `--save-root` / `--dit-checkpoint`。

## 1. 准备

```bash
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

# v2：默认从 v1 latest/best.pt warm start
VERSION=v2 bash qwen3vl_local/goalgen/train.sh ddp
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
| `DDP_GPU_COUNT` | `8` | DDP 需要的 GPU 数 |
| `COMPILE_DIT` | `1` | 只 compile DiT |
| `GRAD_CKPT` | `1` | per-block gradient checkpointing |
| `MICRO_BS` | `16` | DiT 单步 forward 处理的 sample 数；=1 走旧 per-sample 路径（与早期 ckpt 字节级等价），>1 启用 pad_pooled_kv_batch + lang_key_padding_mask（详见 GOALGEN_PLAN.md 显存预算表与 §H20 batched 训练章节） |
| `GRAD_ACC` | `2` | 梯度累积步数；等效 global batch = world_size × MICRO_BS × GRAD_ACC（默认 8卡 × 16 × 2 = 256） |
| `LR` | `5.66e-4` | v1 已按默认等效 batch 256 调好（vs 早期 batch=1 路径的 32，8x，LR sqrt(8)=2.83x 从 2e-4 上调）。VERSION=v2 warm start 仍默认 LR=1e-4；再加大 MICRO_BS 时按 sqrt 法则继续调 |

GPU 规则：launcher 用 `nvidia-smi` 自动挑空闲卡并覆盖旧
`CUDA_VISIBLE_DEVICES`。不要手写卡号；用 `DDP_GPU_COUNT=N` 控制卡数。

### 3.0.0 H20 96GB batched 训练示例

**2026-06 新默认**（H20 上尽量靠近 80% 显存）：`MICRO_BS=16 / GRAD_ACC=2 / LR=5.66e-4`，
8 卡 ddp 等效 global batch=256，单卡显存预期 ~75-88GB。
直接 `bash qwen3vl_local/goalgen/train.sh ddp` 不用设任何 env。

显存还有余量想继续上调：

```bash
# 默认（无需 env）
bash qwen3vl_local/goalgen/train.sh ddp

# 再 +1.5x 等效 batch（global 384）：LR 5.66e-4 × sqrt(384/256)=6.93e-4，高 OOM 风险
MICRO_BS=24 GRAD_ACC=2 LR=6.93e-4 bash qwen3vl_local/goalgen/train.sh ddp

# 保持等效不变只省显存（用于稳定性回退）：MICRO_BS÷2 / GRAD_ACC×2
MICRO_BS=8 GRAD_ACC=4 LR=5.66e-4 bash qwen3vl_local/goalgen/train.sh ddp

# 保守回退到上一版默认（global 128）
MICRO_BS=8 GRAD_ACC=2 LR=4e-4 bash qwen3vl_local/goalgen/train.sh ddp

# 回到 2026-06 之前的 batch=1 路径（与历史 ckpt 字节级等价）
MICRO_BS=1 GRAD_ACC=4 LR=2e-4 bash qwen3vl_local/goalgen/train.sh ddp
```

`MICRO_BS>1` 时单卡显存随 batch 约 45-90GB（详见 PLAN §显存预算表；默认 16 约 75-88GB）；OOM 时先 ÷2
`MICRO_BS`、×2 `GRAD_ACC` 保持等效 batch 不变，再视情况按 sqrt 法则反向降 LR。
若 `torch.compile(dit)` 触发频繁 recompile（首几步 log 出现 `recompile` 关键字），
临时设 `COMPILE_DIT=0` 排查；compile 与 dynamic batch + dynamic seq_len 的
交互在新版 PyTorch 上一般稳定，但跨版本可能差异较大。

训练产物：

```text
checkpoints/goalgen_v*_dit/run_<tag>/
checkpoints/goalgen_v*_dit/latest -> run_<tag>
```

`latest/best.pt` 是 eval/probe 默认首选；`latest/latest.pt` 是训练末尾权重。

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

### 3.1 v2 多卡启动

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

Probe 适合看单 case 的输入图、目标图、生成图、Euler trace 和指标。

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
| `invalid device ordinal` | 默认不要传 `--gpu`；需要锁卡时先确保外部 CVD 可见对应编号 |
| 生成图像像 VAE 重建上限差 | 先看 `target_vae_recon`，判断是 VAE 上限还是 DiT 失败 |
