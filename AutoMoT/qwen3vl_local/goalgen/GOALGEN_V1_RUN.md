# GoalGen v1 Run

> All commands below are run from `AutoMoT/` on the remote machine.
> GoalGen files live under `qwen3vl_local/goalgen/`.

## 0. Check Inputs

```bash
cd ~/automot_lead
git pull
cd AutoMoT

ls checkpoints/Qwen3-VL-4B-Instruct/ | head -5
ls vae_standalone/weights/vae_only.safetensors
ls vae_standalone/config/vae_only.yaml
ls /data/lead_data/data/Accident | head -3
ls /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json
```

## 1. Build Dataset

```bash
python qwen3vl_local/goalgen/build_dataset_v1.py \
  --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
  --data-root /data/lead_data/data \
  --samples-per-scenario 1000 \
  --output-dir checkpoints/goalgen_v1_data
```

The builder follows the SFT v1 timeline idea:

- `status` is the GT status at `anchor`.
- `subgoal` is the next event after `status`.
- `target_frame` is the keyframe where `subgoal` starts.
- Samples require `target_frame > anchor`.

Use `--samples-per-scenario 0` to keep every valid anchor. The default `1000`
keeps a large balanced subset per scenario.

Outputs:

```text
checkpoints/goalgen_v1_data/train.jsonl
checkpoints/goalgen_v1_data/val.jsonl
checkpoints/goalgen_v1_data/stats.json
```

## 2. Train

```bash
# Two optimizer steps, for pipeline sanity.
bash qwen3vl_local/goalgen/train_v1.sh check

# Single GPU.
bash qwen3vl_local/goalgen/train_v1.sh single

# DDP. Defaults to the 8 idlest GPUs and an auto-selected free port.
bash qwen3vl_local/goalgen/train_v1.sh ddp

# DDP with 4 auto-selected GPUs.
DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train_v1.sh ddp
```

Common overrides:

```bash
TRAIN_JSONL=checkpoints/goalgen_v1_data/train.jsonl \
OUTPUT_DIR=checkpoints/goalgen_v1_dit \
DDP_GPU_COUNT=4 \
QWEN_KV_SEGMENT_MODE=select_last \
bash qwen3vl_local/goalgen/train_v1.sh ddp
```

`QWEN_KV_SEGMENT_MODE` defaults to `select_last`; only override to
`concat_layers` for ablation comparison.

Training freezes Qwen and VAE. Only DiT-MoT is optimized.

Outputs:

```text
checkpoints/goalgen_v1_dit/latest.pt
checkpoints/goalgen_v1_dit/checkpoint-000200/goalgen_v1.pt
checkpoints/goalgen_v1_dit/tb/                # TensorBoard event 文件
```

## 2.1 TensorBoard

Training writes scalars and image samples to `OUTPUT_DIR/tb/` (rank 0 only). Open
it on the remote machine, then port-forward to your laptop:

```bash
# 远程：起 tb server，绑 0.0.0.0 让本地能连
tensorboard --logdir checkpoints/goalgen_v1_dit/tb --port 6006 --bind_all
# 本地：ssh port forward
ssh -L 6006:localhost:6006 user@remote
# 浏览器打开 http://localhost:6006
```

### 端口冲突自适应

TensorBoard 默认不会自动避让端口（`--port 6006` 占用直接报 `Address already in
use` 退出）。多个用户共用同一台机器或重启留有僵尸 tb 进程时建议用下面两种方案
之一：

**方案 A：让 OS 分配空闲端口**

```bash
# --port 0 时 tb 会从 OS 拿一个空闲端口；启动后从 stdout 里读"TensorBoard ... http://localhost:NNNNN"
tensorboard --logdir checkpoints/goalgen_v1_dit/tb --port 0 --bind_all
```

读出来的 N 同样可以 `ssh -L N:localhost:N`。缺点：tb 没起来之前你不知道端口号，
适合人工跟 tb stdout 时用。

**方案 B：脚本里先探空闲端口**

```bash
# 在远程 shell 里一行算出空闲端口（与 train_v1.sh 里 find_free_master_port 同手法）
TB_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "[tb] picked port ${TB_PORT}"
tensorboard --logdir checkpoints/goalgen_v1_dit/tb --port "${TB_PORT}" --bind_all &
# 本地用同一个端口 ssh forward
ssh -L "${TB_PORT}:localhost:${TB_PORT}" user@remote
```

这种方式 tb 起之前端口就确定了，适合后台 `&` 跑 + 远程自动化脚本。注意 Python 探完
端口到 tb 实际 bind 之间有几毫秒窗口可能被别的进程抢，**生产场景**直接重跑一次
即可（再次探出的端口几乎肯定不同）。

Tags:

| Tag | Meaning |
|---|---|
| `train/loss` | flow matching MSE on `v_pred - v_target`，越低越好 |
| `train/cos` | `cosine_similarity(v_pred, v_target)`，越接近 1 越好（健康训练 ~0.5+） |
| `train/lr` | 当前学习率（cosine 调度后的值） |
| `diag/grad_norm` | clip 前的全梯度范数；正常应稳定在 1–10，持续上涨说明在炸 |
| `diag/kv_seq_len` | 每条样本 Qwen prefill 后 token 数，监控 prompt 是否异常变长 |
| `val/loss` | 在 val 子集（默认前 64 条）上同样口径的 loss |
| `val/cos` | val 子集 velocity 余弦 |
| `samples/pred_vs_gt` | 每 `IMAGE_LOG_EVERY` 步生成的 pred / gt 并排图，依次：pred₀, gt₀, pred₁, gt₁, … |

控制开销：

```bash
# 仅保留标量曲线，关掉 image sample（每次约 32 步 euler，含 VAE decode）
IMAGE_LOG_EVERY=0 bash qwen3vl_local/goalgen/train_v1.sh ddp

# val / image 频率可分别调
VAL_STEPS=200 IMAGE_LOG_EVERY=1000 bash qwen3vl_local/goalgen/train_v1.sh ddp

# 完全关闭 tb（仅 stdout 日志，用于 check 模式快速跑 2 step）
bash qwen3vl_local/goalgen/train_v1.sh check  # check 模式默认仍写 tb，需要时手动加 --no-tb
```

DDP 下 tb writer 只在 rank 0 起，其它 rank 不写文件；val / image sample 同样
只在 rank 0 跑（用 DDP 解包后的裸 DiT），不参与 all-reduce。

## 3. Forward Smoke

The old single-step runner is still useful for inspecting one route:

```bash
python leaderboard/team_code/qwen3vl_dit_goalgen_runner.py \
  --route-dir /data/lead_data/data/Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46 \
  --anchor 12 \
  --num-frames 4 \
  --keyframes-json /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
  --qwen-kv-segment-mode select_last \
  --save-root eval_json/qwen3vl_dit_goalgen
```

To inspect a trained DiT instead of a random initialized model, add:

```bash
  --dit-checkpoint checkpoints/goalgen_v1_dit/latest.pt
```

The runner validates `STATUS/SUBGOAL`, requires `target_frame > anchor`, and
feeds all history latents to DiT, matching the training interface. Dataset
training should still use `build_dataset_v1.py`.

## 3.5 离线 Eval

`eval_v1.py` 跑 `val.jsonl`，对每条样本：teacher-forced Qwen prefill → VAE encode →
Euler 采样 → VAE decode，输出四个指标和图像并排。

```bash
python qwen3vl_local/goalgen/eval_v1.py \
  --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
  --dit-checkpoint checkpoints/goalgen_v1_dit/latest.pt \
  --out-dir eval_json/goalgen_v1 \
  --max-samples 200 \
  --euler-steps 32 \
  --image-dump-count 32
```

四个指标（与 5.3 设计一致）：

| 指标 | 含义 | 期望方向 |
|---|---|---|
| `latent_mse` | `MSE(z1_pred, z1_gt)`，与训练 loss 同口径但对 z 而非 v | 越低越好 |
| `latent_cos` | `cosine(z1_pred, z1_gt)` | 越接近 1 越好 |
| `pixel_l1` / `psnr` | VAE.decode 后 [-1,1] RGB 的 L1 与 PSNR | l1 越低 / psnr 越高 |
| `velocity_cos` | 5 个 t 点 (0.1/0.3/0.5/0.7/0.9) v 余弦平均 | 训练健康性，与 train/cos 同口径 |

输出：

```text
eval_json/goalgen_v1/eval_v1_summary.json    # overall + by_scenario 聚合
eval_json/goalgen_v1/eval_v1_perline.jsonl   # 每条样本一行
eval_json/goalgen_v1/samples/00000_pred.png  # 前 image-dump-count 条 pred / gt PNG 并排（自己 ssh 拉回看）
eval_json/goalgen_v1/samples/00000_gt.png
```

`pixel_l1 / psnr` 是直接对比 VAE.decode 后的 RGB；地板取决于 VAE 重建质量本身，
所以光看绝对值意义有限——做"base ckpt vs 训练后 ckpt"或"step-200 vs step-1000"
横向对比时 delta 才有意义。

## 4. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `keyframes json not found` | Wrong remote path | Use `/datashare/IOL4SGH/data/data/keyframes_all_scenarios.json` |
| `RGB image not found` | `--data-root` does not match LEAD data | Use `/data/lead_data/data` or the actual mounted path |
| Qwen prefill OOM | Too many frames / large KV | Use fewer GPUs per process only if needed; otherwise keep `num_frames=4` and use H20-class GPUs |
| DDP port conflict | Existing `MASTER_PORT` is occupied | Launcher auto-selects a port unless `GOALGEN_RESPECT_MASTER_PORT=1` |
| Slow training | Qwen prefill and VAE encode run per sample | This is expected for v1; later versions can cache segmented KV/latents |
| Qwen/DiT OOM after switching to full KV ablation | `QWEN_KV_SEGMENT_MODE=concat_layers` keeps 3 Qwen layers per DiT block (3x language tokens) | Drop back to default `QWEN_KV_SEGMENT_MODE=select_last`; use `mean` only for old ablation. |
| `target_frame must be in the future` | Manual `--target-frame` or keyframes event is <= `--anchor` | Choose a later anchor target, or let the dataset builder select valid anchors. |
| `SUBGOAL ... does not match STATUS` | CLI override violates the scenario event chain | Use STATUS only; runner derives the next SUBGOAL automatically. |
| `language KV batch ... != vision batch ...` | DDP rank received KV from a different sample | Make sure each sample has its own `teacher_forced_prefill`; do not stack different samples' KV. |
| `pooled_kv segments X != DiT layers Y` | `--num-layers` and `num_segments` drifted apart | Keep `--num-layers` (trainer) and the segmenter's `num_segments` (qwen_kv.py default 12) equal. |
| `RuntimeError: shape mismatch` inside JointAttention | Hardcoded `language_kv_input_dim` no longer matches base model's `n_kv_heads * head_dim` | Use `--language-kv-input-dim auto` (default); trainer infers it from the first sample's segmented KV. |

## 5. Shape Defaults

These values are the contract between `build_dataset_v1.py`, `train_v1.py`,
and `qwen3vl_dit_goalgen_runner.py`. **If you change one of them, update both
this file and `GOALGEN_V1_PLAN.md`.**

| Param | Default | Effect when changed |
|---|---|---|
| LEAD stitched RGB | 1152x384 | Hard-coded by data; do not change without re-stitching. |
| VAE latent | [B, 4, 48, 144] | Derived from RGB via /8 downsample. |
| `--patch-size 2` | grid (24, 72) = 1728 token / latent | Use 4 to cut tokens 4x; lower fidelity. |
| `--hidden-dim 768` | 768 / head_dim 64 | Must be divisible by `--n-heads`. |
| `--n-heads 12` | head_dim = 64 | Must divide `--hidden-dim`. |
| `--num-layers 12` | == KV segments | Must equal `segment_kv_for_dit(num_segments=...)`. Default 12. |
| `--mlp-ratio 4.0` | MLP hidden = 768 * 4 | DiT-XL convention. |
| `--cond-dim 256` | Timestep embedding dim | Affects per-layer AdaLN modulation size. |
| `--max-history-frames 8` | Allows builder-default 4 history latents plus room for longer clips | Must be >= `len(history_rgb_paths)` in the jsonl. |
| `--qwen-kv-segment-mode select_last` | Each segment is the last Qwen layer of its 3-layer group; token-level K/V `[B, 8, S, 128]`. Default. | Switch to `concat_layers` only for ablation (3x language tokens, much heavier). |
| `--dit-checkpoint` | Optional path to `latest.pt` or `checkpoint-*/goalgen_v1.pt` | Omit only for structure smoke tests with random DiT weights. |
| `--language-kv-input-dim auto` | Inferred from `n_kv_heads * head_dim` of segmented KV (Qwen3-VL-4B-Instruct = 1024) | Set to a fixed int only if you know the base model's KV shape and want to skip the auto probe. |

## 6. Memory Expectations (training, batch=1, bf16 DiT)

Per rank on H20 96GB:

- Qwen 4B (bf16) ~8 GB
- VAE (fp32) ~0.4 GB
- Segmented KV with `select_last` (12 segments, bf16) ~1 GB
- DiT + activations + backward + AdamW: higher than old single-frame latent path because DiT now sees all history latents

If this OOMs under default `select_last`, reduce `HIDDEN_DIM` or history frames.
Switching to `concat_layers` will triple language tokens per block, so only use
it for ablation. DDP replicates Qwen + VAE on every rank, so v2 should
offline-cache segmented KV + latents to remove this duplication.

## 7. Forward Smoke vs Training Smoke

Two different sanity entry points:

| Entry | Purpose |
|---|---|
| `qwen3vl_dit_goalgen_runner.py` | One-shot forward on a specific LEAD route. STATUS/SUBGOAL are validated against the scenario chain. Use for inspecting one sample's shapes, `step.json`, and optional trained DiT checkpoint behavior. |
| `train_v1.sh check` | Two optimizer steps on the actual jsonl. Verifies that backward + DDP + optimizer step all wire up. Use before launching a full DDP run. |

Always run `check` before `single` / `ddp`.

## 8. What v1 does NOT do

- No multi-step Euler sampling at training time; loss is computed at a single random `t`.
- No EMA / CFG / latent caching.
- No image-decode evaluation; metric = loss + velocity cosine only.

See `GOALGEN_V1_PLAN.md` "v1 / v2 Boundary" for the full list.
