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
cd ~/automot_lead
git pull
cd AutoMoT
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
| `PATCH_UNPATCH_WEIGHTS` | 空 | 可选 `vae_standalone/train_patch_unpatch.py` 产出的 `patch_unpatch_*.safetensors`；非空时加载并默认冻结，路径会写入 GoalGen ckpt |
| `PATCH_UNPATCH_UNFREEZE` | `0` | 设 `1` 时外部 patch/unpatch 只作初始化、继续联合训练 |
| `PATCH_UNPATCH_CKPT_FALLBACK` | `0` | warm start 继承外部 patch/unpatch 但 safetensors 缺失时，显式允许使用 ckpt 内自带权重 |
| `DDP_GPU_COUNT` | `8` | DDP 需要的 GPU 数 |
| `COMPILE_DIT` | `1` | 只 compile DiT |
| `GRAD_CKPT` | `1` | per-block gradient checkpointing |

GPU 规则：launcher 用 `nvidia-smi` 自动挑空闲卡并覆盖旧
`CUDA_VISIBLE_DEVICES`。不要手写卡号；用 `DDP_GPU_COUNT=N` 控制卡数。

训练产物：

```text
checkpoints/goalgen_v*_dit/run_<tag>/
checkpoints/goalgen_v*_dit/latest -> run_<tag>
```

`latest/best.pt` 是 eval/probe 默认首选；`latest/latest.pt` 是训练末尾权重。

patch/unpatch 保存约定：

- 传 `PATCH_UNPATCH_WEIGHTS=/path/to/patch_unpatch_*.safetensors` 且保持默认冻结时，训练会加载这份权重并冻结，checkpoint 的 `dit_config.patch_unpatch_source=external`，`dit_config.patch_unpatch_weights` 记录该 safetensors 绝对路径。eval/probe/runner 加载 GoalGen ckpt 后会按这个路径再次覆盖 patch/unpatch 并恢复冻结语义，避免悄悄使用不匹配的随机 patch。
- 若同时设 `PATCH_UNPATCH_UNFREEZE=1`，外部 patch/unpatch 只作为初始化，后续联合训练后的权重随 GoalGen ckpt 保存，source 记为 `checkpoint`。
- 不传 `PATCH_UNPATCH_WEIGHTS` 时，patch/unpatch 随 DiT 随机初始化并联合训练，权重跟随 `dit_state_dict` / `ema_state_dict` 一起保存；checkpoint 记录 `patch_unpatch_source=checkpoint`，eval/probe/runner 直接使用 `--dit-checkpoint` 内自带的 patch/unpatch。
- 使用 EMA 推理时，加载逻辑先以完整 `dit_state_dict` 打底，再用 `ema_state_dict` 覆盖可训练参数；因此外部冻结的 patch/unpatch 即使不在 EMA shadow 里，也不会在 strict load 时缺 key。
- warm start 读到旧 ckpt 的 `patch_unpatch_source=external` 时，默认要求记录的 safetensors 路径仍存在；若跨机器迁移确实只有 ckpt，可显式设 `PATCH_UNPATCH_CKPT_FALLBACK=1`，此时使用 ckpt 内已保存的 patch/unpatch，并把新产物记为 `source=checkpoint`。

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
bash tools/tb_serve.sh checkpoints/goalgen_v1_dit
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
