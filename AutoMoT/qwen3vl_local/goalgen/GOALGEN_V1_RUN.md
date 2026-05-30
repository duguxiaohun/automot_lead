# GoalGen v1 运行手册

> 下面所有命令都在远端机器上 `AutoMoT/` 目录里执行。
> GoalGen 相关文件都放在 `qwen3vl_local/goalgen/` 下。本手册只覆盖 v1：
> 先构建 jsonl 数据，再做两步训练检查，确认无误后再跑单卡或 DDP。

## 0. 检查输入

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

## 1. 构建数据集

```bash
python qwen3vl_local/goalgen/build_dataset_v1.py \
  --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
  --data-root /data/lead_data/data \
  --samples-per-scenario 1000 \
  --output-dir checkpoints/goalgen_v1_data
```

数据构建器沿用 SFT v1 的时间线思路，但目标从"文本标签"换成"未来子目标关键帧"：

- `status` 是 `anchor` 帧的 GT status；
- `subgoal` 是 `status` 之后的下一个事件；
- `target_frame` 是 `subgoal` 触发所在的 keyframe；
- 样本必须满足 `target_frame > anchor`（子目标只能在未来）。

传 `--samples-per-scenario 0` 可以保留所有合法锚点帧；默认 `1000` 会按场景
做一个**较大但已经平衡过的子集**。

输出：

```text
checkpoints/goalgen_v1_data/train.jsonl
checkpoints/goalgen_v1_data/val.jsonl
checkpoints/goalgen_v1_data/stats.json
```

## 2. 训练

```bash
# 跑 2 个优化器步，验证完整链路是否通
bash qwen3vl_local/goalgen/train_v1.sh check

# 单卡训练
bash qwen3vl_local/goalgen/train_v1.sh single

# DDP，默认自动挑 8 个最闲的 GPU 和一个空闲端口
bash qwen3vl_local/goalgen/train_v1.sh ddp

# DDP，只用 4 张自动挑选的 GPU
DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train_v1.sh ddp
```

常用覆盖参数：

```bash
TRAIN_JSONL=checkpoints/goalgen_v1_data/train.jsonl \
OUTPUT_DIR=checkpoints/goalgen_v1_dit \
DDP_GPU_COUNT=4 \
QWEN_KV_SEGMENT_MODE=select_last \
bash qwen3vl_local/goalgen/train_v1.sh ddp
```

`QWEN_KV_SEGMENT_MODE` 默认是 `select_last`；只有在做消融对比时才切到
`concat_layers`。

训练阶段 **Qwen 与 VAE 全程冻结，只更新 DiT-MoT**。

**默认跑基础 Qwen，不挂任何 LoRA / 适配器。** 想接续 SFT v1 微调后的语言编码再看
§2.0.1；当前训练生成模型阶段无需关心适配器，直接用上面命令即可。

### 2.0.1 接入 LoRA / PEFT 适配器（接续 SFT v1 微调后的 Qwen）

> **默认不挂 LoRA**：训练 / 评测 / 单步 runner 三个入口的 `--qwen-adapter-dir` 默认为空字符串，
> 这种情况下完全走基础 Qwen，不会导入 `peft`，也不会读适配器目录。
> 当前训练生成模型阶段**不需要**做任何事，直接 `bash qwen3vl_local/goalgen/train_v1.sh ddp`
> 即可——下面这一节只有在你后续想接续 SFT v1 微调后的语言编码时再读。

如果想让 GoalGen 直接吃 SFT v1 微调后的语言编码，传 LoRA 适配器目录即可，**不需要**
事先合并：

```bash
QWEN_ADAPTER_DIR=checkpoints/sft_v1_lora \
OUTPUT_DIR=checkpoints/goalgen_v1_dit_sftv1 \
bash qwen3vl_local/goalgen/train_v1.sh ddp
```

实现细节（`engine.attach_lora_adapter`）：

- 基础 Qwen 不变；适配器用 `PeftModel.from_pretrained(base, adapter_dir)` 包一层；
- 默认 `merge=True` 走 `merge_and_unload()`：LoRA 权重合进基础矩阵，之后 `engine.model`
  上不再有 PEFT 包装，预填充 / KV 提取 / 分段切分对 LoRA 的存在完全无感知；
- LoRA 不改变 `n_kv_heads / head_dim / num_layers`，所以 `language_kv_input_dim` 与
  基础 Qwen 一致，DiT 形状不用动；
- 合并后训练前向比 PeftModel 包装快约 5–10%，且更省一份 LoRA 分支显存；
- 想保留 PEFT 包装（调试用）传 `--no-qwen-adapter-merge`。

**评测 / runner 必须传同一个 `--qwen-adapter-dir`**：训练用适配器而评测用基础模型
会让 KV 分布偏移，指标完全不可比。

为了防止"忘了传"导致的静默错误生成，评测 / runner 在加载 DiT 检查点时会从
`payload["args"]` 读训练时的 `qwen_adapter_dir`，**与当前 CLI 严格比对**：

- 训练 + 当前都是基础模型（空串）→ OK
- 训练 + 当前都是同一适配器目录（绝对路径比较）→ OK，输出提示
- 训练 + 当前都挂适配器但合并开关不同 → 输出提示（数学等价，只有浮点精度差异）
- **适配器路径不一致 → 默认抛 `RuntimeError`，明确告诉你训练时是什么、当前是什么**
- 想做跨适配器消融对比时传 `--allow-qwen-adapter-mismatch`，转为警告后继续

```bash
# eval
python qwen3vl_local/goalgen/eval_v1.py \
  --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
  --dit-checkpoint checkpoints/goalgen_v1_dit_sftv1/latest.pt \
  --qwen-adapter-dir checkpoints/sft_v1_lora \
  --out-dir eval_json/goalgen_v1_sftv1

# runner（单步前向冒烟测试）
python leaderboard/team_code/qwen3vl_dit_goalgen_runner.py \
  --route-dir /data/lead_data/data/Accident/Town03_... \
  --anchor 12 \
  --qwen-adapter-dir checkpoints/sft_v1_lora \
  --dit-checkpoint checkpoints/goalgen_v1_dit_sftv1/latest.pt
```

输出：

```text
checkpoints/goalgen_v1_dit/latest.pt
checkpoints/goalgen_v1_dit/checkpoint-000200/goalgen_v1.pt
checkpoints/goalgen_v1_dit/tb/                # TensorBoard event 文件
```

## 2.1 TensorBoard

训练时只在 0 号进程写标量与图像样例到 `OUTPUT_DIR/tb/`。先在远端起 TensorBoard
服务，再把端口转发到本机：

```bash
# 远程：起 TensorBoard 服务，绑定 0.0.0.0 让本地能连
tensorboard --logdir checkpoints/goalgen_v1_dit/tb --port 6006 --bind_all
# 本地：SSH 端口转发
ssh -L 6006:localhost:6006 user@remote
# 浏览器打开 http://localhost:6006
```

### 端口冲突自适应

TensorBoard 默认不会自动避让端口（`--port 6006` 占用会直接报 `Address already in
use` 并退出）。多个用户共用同一台机器或重启后留有旧进程时，建议用下面两种方案
之一：

**方案 A：让 OS 分配空闲端口**

```bash
# --port 0 时 tb 会从 OS 拿一个空闲端口；启动后从 stdout 里读"TensorBoard ... http://localhost:NNNNN"
tensorboard --logdir checkpoints/goalgen_v1_dit/tb --port 0 --bind_all
```

读出来的 N 同样可以 `ssh -L N:localhost:N`。缺点：服务没起来之前你不知道端口号，
适合人工盯启动输出时用。

**方案 B：脚本里先探空闲端口**

```bash
# 在远程 shell 里一行算出空闲端口（与 train_v1.sh 里 find_free_master_port 同手法）
TB_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "[tb] picked port ${TB_PORT}"
tensorboard --logdir checkpoints/goalgen_v1_dit/tb --port "${TB_PORT}" --bind_all &
# 本地用同一个端口做 SSH 转发
ssh -L "${TB_PORT}:localhost:${TB_PORT}" user@remote
```

这种方式在服务启动前端口就确定了，适合后台 `&` 跑 + 远程自动化脚本。注意 Python 探完
端口到 tb 实际 bind 之间有几毫秒窗口可能被别的进程抢，**生产场景**直接重跑一次
即可（再次探出的端口几乎肯定不同）。

标签含义：

| Tag | 含义 |
|---|---|
| `train/loss` | 流匹配均方误差，对比 `v_pred` 与 `v_target`，越低越好 |
| `train/cos` | `cosine_similarity(v_pred, v_target)`，越接近 1 越好（健康训练 ~0.5+） |
| `train/lr` | 当前学习率（cosine 调度后的值） |
| `diag/grad_norm` | clip 前的全梯度范数；正常应稳定在 1–10，持续上涨说明在炸 |
| `diag/kv_seq_len` | 每条样本 Qwen prefill 后 token 数，监控 prompt 是否异常变长 |
| `val/loss` | 在验证子集（默认前 64 条）上同样口径的损失 |
| `val/cos` | val 子集 velocity 余弦 |
| `samples/pred_vs_gt` | 每 `IMAGE_LOG_EVERY` 步生成的预测 / 真值并排图，依次：pred₀, gt₀, pred₁, gt₁, … |

控制开销：

```bash
# 仅保留标量曲线，关掉图像样例（每次约 32 步 Euler，含 VAE 解码）
IMAGE_LOG_EVERY=0 bash qwen3vl_local/goalgen/train_v1.sh ddp

# 验证 / 图像频率可分别调
VAL_STEPS=200 IMAGE_LOG_EVERY=1000 bash qwen3vl_local/goalgen/train_v1.sh ddp

# 完全关闭 TensorBoard（仅保留 stdout 日志，用于 check 模式快速跑 2 步）
bash qwen3vl_local/goalgen/train_v1.sh check  # check 模式默认仍写 TensorBoard，需要时手动加 --no-tb
```

DDP 下 TensorBoard 写入器只在 0 号进程启动，其它进程不写文件；验证 / 图像样例同样
只在 0 号进程跑（用 DDP 解包后的裸 DiT），不参与跨卡归约。

## 3. 单条前向冒烟测试

老的单步 runner 用来检查单条 route 仍然有用：

```bash
python leaderboard/team_code/qwen3vl_dit_goalgen_runner.py \
  --route-dir /data/lead_data/data/Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46 \
  --anchor 12 \
  --num-frames 4 \
  --keyframes-json /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
  --qwen-kv-segment-mode select_last \
  --save-root eval_json/qwen3vl_dit_goalgen
```

如果要喂训练好的 DiT 而不是随机初始化的，加上：

```bash
  --dit-checkpoint checkpoints/goalgen_v1_dit/latest.pt
```

runner 会校验 `STATUS/SUBGOAL`、强制要求 `target_frame > anchor`，并且把所有
历史潜变量喂给 DiT（和训练接口一致）。**数据集训练仍走 `build_dataset_v1.py`**。

## 3.5 离线评测

`eval_v1.py` 跑 `val.jsonl`，对每条样本执行：teacher-forced Qwen 预填充 → VAE 编码 →
Euler 采样 → VAE 解码，输出四个指标和图像并排。

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
| `latent_mse` | `MSE(z1_pred, z1_gt)`，与训练损失同口径但对 z 而非 v | 越低越好 |
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

`pixel_l1 / psnr` 是直接对比 VAE 解码后的 RGB；下限取决于 VAE 重建质量本身，
所以光看绝对值意义有限——做"基础检查点 vs 训练后检查点"或"step-200 vs step-1000"
横向对比时 delta 才有意义。

## 4. 排障

| 现象 | 可能原因 | 修复 |
|---|---|---|
| 关键帧 JSON 不存在 | 远端路径错了 | 使用 `/datashare/IOL4SGH/data/data/keyframes_all_scenarios.json` |
| RGB 图像不存在 | `--data-root` 与 LEAD 数据不匹配 | 使用 `/data/lead_data/data` 或实际挂载路径 |
| Qwen 预填充显存溢出 | 历史帧太多 / KV 太大 | 必要时减少每进程占用，否则保持 `num_frames=4`、用 H20 级 GPU |
| DDP 端口冲突 | 已有 `MASTER_PORT` 被占 | launcher 默认自动选端口；如需保留固定端口设 `GOALGEN_RESPECT_MASTER_PORT=1` |
| 训练慢 | Qwen 预填充与 VAE 编码每个样本都重算 | v1 已知瓶颈；后续版本将缓存分段 KV / 潜变量 |
| 切到完整 KV 消融后显存溢出 | `QWEN_KV_SEGMENT_MODE=concat_layers` 让每个 DiT 层包含 3 层 Qwen（语言 token 3 倍） | 回到默认 `QWEN_KV_SEGMENT_MODE=select_last`；`mean` 仅作旧消融 |
| `target_frame must be in the future` | 手动 `--target-frame` 或 keyframes 事件 <= `--anchor` | 选更晚的目标帧，或让数据构建器自动选合法锚点 |
| `SUBGOAL ... does not match STATUS` | CLI 覆盖打破了 scenario 事件链 | 只指定 STATUS，让 runner 自己推下一个 SUBGOAL |
| `language KV batch ... != vision batch ...` | DDP 进程拿到的 KV 来自别的样本 | 保证每个样本各自走 `teacher_forced_prefill`；不要堆别人样本的 KV |
| `pooled_kv segments X != DiT layers Y` | `--num-layers` 与 `num_segments` 漂移 | 保持 `--num-layers`（训练器）与分段函数的 `num_segments`（qwen_kv.py 默认 12）一致 |
| JointAttention 内部 `RuntimeError: shape mismatch` | 硬编码的 `language_kv_input_dim` 与基础模型 `n_kv_heads * head_dim` 不一致 | 使用 `--language-kv-input-dim auto`（默认），让训练器从第一条样本的分段 KV 推断 |

## 5. 默认形状

下列默认值是 `build_dataset_v1.py`、`train_v1.py`、
`qwen3vl_dit_goalgen_runner.py` 三方共同遵守的契约。**任何一个改动都必须同步
更新本文件与 `GOALGEN_V1_PLAN.md`**。

| 参数 | 默认值 | 修改后影响 |
|---|---|---|
| LEAD 拼接 RGB | 1152x384 | 由数据决定，不重新拼图就不要改 |
| VAE latent | [B, 4, 48, 144] | 来自 RGB / 8 下采样 |
| `--patch-size 2` | 网格 (24, 72) = 每个潜变量 1728 个 token | 改 4 会让 token 数减 4 倍，细节精度下降 |
| `--hidden-dim 768` | 隐藏维度 768 / 每头维度 64 | 必须能整除 `--n-heads` |
| `--n-heads 12` | 每头维度 = 64 | 必须整除 `--hidden-dim` |
| `--num-layers 12` | 等于 KV 段数 | 必须等于 `segment_kv_for_dit(num_segments=...)`，默认 12 |
| `--mlp-ratio 4.0` | MLP 隐藏维度 = 768 * 4 | DiT-XL 常用约定 |
| `--cond-dim 256` | 时间步嵌入维度 | 影响每层 AdaLN 调制向量的输出大小 |
| `--max-history-frames 8` | 容纳数据构建器默认 4 帧历史，余量给更长片段 | 必须 >= jsonl 中 `len(history_rgb_paths)` |
| `--qwen-kv-segment-mode select_last` | 每段取 3 层 Qwen 中的最后一层；token-level K/V `[B, 8, S, 128]`，默认 | 只在做消融时切到 `concat_layers`（语言 token 3 倍，明显更重） |
| `--dit-checkpoint` | 可选，传 `latest.pt` 或 `checkpoint-*/goalgen_v1.pt` | 只有做随机初始化结构冒烟测试时才省略 |
| `--language-kv-input-dim auto` | 从分段 KV 的 `n_kv_heads * head_dim` 推断（Qwen3-VL-4B-Instruct = 1024） | 只有在你确知 base 模型 KV shape 且想跳过 auto probe 时才传定值 |

## 6. 显存预期（训练，batch=1，DiT 用 bfloat16）

H20 96GB 上每个进程：

- Qwen 4B（bfloat16）约 8 GB
- VAE（float32）约 0.4 GB
- 分段 KV（`select_last`，12 段，bfloat16）约 1 GB
- DiT 权重 + 激活 + 反传 + AdamW：比老的单帧潜变量路径要大，因为现在 DiT
  会看到所有历史潜变量

如果默认 `select_last` 下都显存溢出，可以减小 `HIDDEN_DIM` 或者减少历史帧数。切到
`concat_layers` 会把每个 DiT 层的语言 token 数翻 3 倍，**只用于消融实验**。
DDP 下每个进程都重复加载一份 Qwen + VAE，所以 v2 必须把分段 KV + 潜变量
**离线缓存**，消除这种复制开销。

## 7. 前向冒烟测试与训练冒烟测试

这是两个**不同**的最小验证入口：

| 入口 | 用途 |
|---|---|
| `qwen3vl_dit_goalgen_runner.py` | 在某条 LEAD route 上跑一次前向。STATUS/SUBGOAL 会按场景链做校验。用于检查一条样本的形状、`step.json` 以及可选 DiT 检查点的行为。 |
| `train_v1.sh check` | 在真实 jsonl 上跑 2 个优化器步。验证反传 + DDP + 优化器更新全链路正常。**全量 DDP 跑之前必跑**。 |

跑 `single` / `ddp` 之前**永远先跑** `check`。

## 8. v1 不做的事

- 训练阶段**不**跑多步 Euler 采样；损失只在单个随机 `t` 上计算。
- 不上 EMA / CFG / latent caching。
- 不做图像解码评测；指标只看损失 + 速度余弦。

完整边界见 `GOALGEN_V1_PLAN.md` 的 "v1 / v2 边界" 一节。
