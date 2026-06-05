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

### 3.1 v2 详解（warm start / 启动方式 / 默认 pt）

**启动命令**（GPU 选址、`RUN_TAG`、`latest` symlink 与 v1 共用同一套规则）：

```bash
# DDP，默认自动挑 8 张最空闲 GPU，从 v1 best.pt warm start
VERSION=v2 bash qwen3vl_local/goalgen/train.sh ddp

# 改 DDP 卡数（卡号仍自动挑）
VERSION=v2 DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp

# 单卡（smoke / 显存够时的小规模）
VERSION=v2 bash qwen3vl_local/goalgen/train.sh single

# 2 step sanity（端到端跑通，看初始 loss 与 warm start 是否对齐）
VERSION=v2 bash qwen3vl_local/goalgen/train.sh check
```

**默认导入哪个 pt**：v2 启动时若不显式设 `INIT_FROM_CKPT`，会自动取：

```text
INIT_FROM_CKPT = checkpoints/goalgen_v1_dit/latest/best.pt
```

`latest` 是 v1 train.sh 维护的 symlink，永远指向最新一次 v1 run。`train.py` 加载时**只接 DiT 权重 + EMA shadow，不接 optimizer / scheduler / global_step**——等同于"继承 v1 学好的权重，换数据子集重新训"。文件不存在会硬报错，不会偷偷从零训。

**怎么导入其它 pt**：

```bash
# 从 v1 最末权重（不是 best.pt）warm start
VERSION=v2 INIT_FROM_CKPT=checkpoints/goalgen_v1_dit/latest/latest.pt \
    bash qwen3vl_local/goalgen/train.sh ddp

# 锁定某个具体 v1 run（不跟 latest 漂）
VERSION=v2 INIT_FROM_CKPT=checkpoints/goalgen_v1_dit/run_20260605_143000/best.pt \
    bash qwen3vl_local/goalgen/train.sh ddp

# 老 schema（v1 训练时还没启用 run 子目录，best.pt 落 base 顶层）
VERSION=v2 INIT_FROM_CKPT=checkpoints/goalgen_v1_dit/best.pt \
    bash qwen3vl_local/goalgen/train.sh ddp

# 完全从零训 v2（不要 warm start；显式清空，不要写 NONE）
VERSION=v2 INIT_FROM_CKPT="" \
    bash qwen3vl_local/goalgen/train.sh ddp
```

**v2 默认超参（保守 fine-tune 配方）**：起点已经是 v1 best.pt，初期 LR 过大会一步把学好的权重打散。

| 超参 | v1 默认 | v2 默认 | 说明 |
|---|---:|---:|---|
| `LR` (AdamW) | `2e-4` | `1e-4` | 减半 |
| `MUON_LR` | `2e-3` | `1e-3` | 减半 |
| `WARMUP_RATIO` | `0.05` | `0.02` | warmup 缩短 |
| `NUM_EPOCHS` | `2` | `2` | 不变 |

想恢复 v1 from-scratch 风格的 LR，显式覆盖即可（例如 `LR=2e-4 MUON_LR=2e-3 ...`）。

**可选：接 SFT v1 LoRA**（让 Qwen prefill 走 SFT 微调后的语言编码）：

```bash
VERSION=v2 QWEN_ADAPTER_DIR=checkpoints/sft_v1_lora/latest \
    bash qwen3vl_local/goalgen/train.sh ddp
```

注意传的是 LoRA adapter 目录，不是合并后的整模型目录。v1/v2 都支持这个开关。

**v2 产物路径**：

```text
checkpoints/goalgen_v2_dit/run_<tag>/{best.pt, latest.pt, tb/, checkpoint-XXXXXX/, ...}
checkpoints/goalgen_v2_dit/latest -> run_<tag>
```

eval / probe 默认指 `checkpoints/goalgen_v2_dit/latest/best.pt`，无需关心具体 `run_<tag>`。

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
