# GoalGen 设计方案（v1 / v2 共用）

> **代码组织约定**：GoalGen 子包采用"单一代码、双版本"风格——v1 / v2 共享同一份
> 脚本（`build_dataset.py` / `train.py` / `train.sh` / `eval.py` / `probe.py`），
> 靠 `--mode` 参数与 `VERSION` env 切换两套行为。本文档默认描述的是 v1（GoalGen 的
> 原始版本）；任何 v2 differs 的地方都用 **[v2 only]** 标签明确标出，v1 / v2
> 都共用的部分用 **[v1/v2 通用]** 标识或不加标签（即默认通用）。
>
> 路线意图：v1 = 在 LEAD 全部 4 类 status→subgoal transition 上从零训 DiT；
> v2 = 只用 middle 之间 2 类 transition + 从 v1 best.pt warm start 做 fine-tune。
> 完整对比见 [`GOALGEN_RUN.md`](GOALGEN_RUN.md) 顶部的"版本与模式速查"。

## 目标

GoalGen 训练一个在 VAE 潜变量空间上工作的 DiT-MoT 生成器。它不直接预测
RGB 像素，而是预测"子目标关键帧"对应的潜变量速度场；条件信息来自冻结的
Qwen3-VL-Instruct 预填充后得到的键值缓存。这样做的核心原因是：Qwen 负责把
历史视觉、当前状态、下一子目标压进语言/视觉上下文，DiT 只学习"在这个上下文下，
未来子目标画面应该往哪里走"。

整条数据流如下：

```text
历史 RGB        ->  冻结 Qwen prefill   ->  分段的 token-level KV
历史 RGB        ->  冻结 VAE            ->  z_history
子目标关键帧 RGB ->  冻结 VAE            ->  z1
z0 = z_current + eps，t ~ logit-normal(0, 1)，z_t = (1 - t) * z0 + t * z1
DiT-MoT(z_t, z_history, t, 分段 KV)   ->  v_pred
loss = MSE(v_pred, z1 - z0)
```

当前第二轮训练默认已经把第一档 diffusion 标配全部纳入 v1 代码路径：共享 patchify、
type embedding normal 初始化、EMA、logit-normal t 采样、CFG 训练/推理、VAE latent
per-channel 标准化、z_current prior 起点，以及学习率 `2e-4` + warmup `0.05`。
这批改动与旧 ckpt 不兼容，第二轮应从零重训。

Qwen 和 VAE 全程冻结，**只有 DiT-MoT 参与训练**。这条边界非常重要：
训练损失只能更新 DiT，不能让语言模型或 VAE 被反向传播污染，否则后续所有
缓存、评测和旧检查点都不可比。

## 数据

数据集构建脚本：

```text
qwen3vl_local/goalgen/build_dataset.py
```

它在结构上对齐 `tools/build_sft_dataset_v1.py`，但监督目标不同：SFT v1 监督
文本中的状态/子目标，GoalGen v1 监督未来子目标关键帧的 VAE 潜变量。流程如下：

1. 读取 `/datashare/IOL4SGH/data/data/keyframes_all_scenarios.json`。
2. 只保留 `Completed` / `Perfect` 的 run。
3. 把 keyframes 展开成每条 run 的 status 时间线。
4. 对每一个合法的 `anchor` 帧，确定：
   - `status` = `anchor` 处的 GT status；
   - `subgoal` = `status` 之后的下一个事件；
   - `target_frame` = `subgoal` 触发所在的 keyframe。
5. 只保留 `target_frame > anchor` 的样本（也就是子目标必须发生在未来）。
6. 按 scenario 和 `status -> subgoal` 做分层采样。

默认 `--samples-per-scenario 0`，保留所有合法锚点帧；如需做小规模消融，可显式传
正整数上限。

**`--mode v1` vs `--mode v2`（同一脚本，输出目录隔离）：**

- `--mode v1`（默认）：保留全部 4 类 `status → subgoal` 转换，即
  `initial → middle[0]` / `middle[0] → middle[1]` / `middle[1] → middle[2]` / `middle[2] → final`。
  默认输出 `checkpoints/goalgen_v1_data/`。
- `--mode v2`：只保留三个 `middle` 子目标之间的两段转换，即
  `middle[0] → middle[1]` / `middle[1] → middle[2]`；排除 `status == "initial"` 与
  `subgoal == "final"` 两端样本。默认输出 `checkpoints/goalgen_v2_data/`。
  动机：起手 `initial` 帧视觉上没有"任务进度"信息、收尾 `final` 子目标视觉上常常退化为
  "减速 / 停车"，对 DiT 生成未来关键帧几乎不携带方向信号；v2 把这两端剔除以让训练
  分布只聚焦在子目标之间的"实质性场景演变"上。

`--mode` 字段会写进 `stats.json` 的 `config`，下游训练 / 评测脚本通过 `--data-dir` 指向
`goalgen_v2_data/` 即可直接复用 v1 的 train / eval / probe 入口，**不需要新代码路径**。

样本字段结构：

```json
{
  "scenario": "Accident",
  "run_id": "...",
  "anchor": 12,
  "status": "initial",
  "subgoal": "hazard_detect",
  "target_event": "hazard_detect",
  "target_frame": 37,
  "history_frames": [9, 10, 11, 12],
  "history_rgb_paths": ["..."],
  "current_rgb_path": ".../rgb/0012.jpg",
  "target_rgb_path": ".../rgb/0037.jpg",
  "memory": {
    "scenario": "Accident",
    "event_sequence": ["initial", "...", "final"],
    "status": "initial",
    "subgoal": "hazard_detect",
    "completed_events": ["initial"]
  }
}
```

## 模型默认配置（v2 架构，2026-06 切换）

> 与 v1 的关键差异：**hidden_dim 1024 / n_heads 8 / head_dim 128 / patch=4**，
> 直接对齐 Qwen3-VL-4B-Instruct 的 `(num_key_value_heads=8, head_dim=128)`，
> 让语言 K/V 不再经过任何线性投影。`lang_k_proj` / `lang_v_proj` 已删除。
> 同时 MLP 走 SwiGLU、所有 norm 走 RMSNorm、attention 内加 q_norm/k_norm，
> 与 Qwen3 / SD3 / Flux 现代 transformer 标配对齐。

- LEAD RGB 尺寸：`1152 x 384`
- VAE latent 形状：`[B, 4, 48, 144]`
- DiT 图块大小：`4`（v1 是 2）
- DiT 上的潜变量图块网格：`(12, 36) = 432` 个图块 token（v1 是 1728）
- 数据构建器默认 4 帧历史时，DiT 视觉 token 总数：
  `z_t` token + `4 * z_history` token = `2160`（v1 是 8640）
- DiT 隐藏维度：`1024`（v1 是 768）
- DiT 注意力头数：`8`（v1 是 12）
- DiT 单头维度：`128`（v1 是 64）
- DiT 层数：`12`
- DiT MLP：SwiGLU，`mlp_ratio=4.0`（inner=4096）
- DiT norm：全部 RMSNorm（block norm1/norm2/final_norm 无 affine + AdaLN modulation；
  attention 内的 q_norm/k_norm 走带 affine 的 RMSNorm，仅对 vision Q/K 归一化，
  language K 保持 Qwen 自己 k_norm 过的形态不再二次归一化）
- Qwen 分段 KV：默认 `select_last` 模式。Qwen 共 36 层，被切成 12 段，
  每段取其 3 层小组里的最后一层（token-level K/V，shape 为
  `[B, 8, S, 128]`）。`concat_layers` 只保留为消融实验选项。
- 语言 K/V 接入方式：**直接复用 Qwen 的 `(8, 128)` 子空间**，无任何线性投影。
  vision Q/K/V 在 DiT 内部投影到同一 `(8, 128)` 空间，与语言 K/V 沿 token 维 concat，
  做一次 joint attention。

**如果以上任何一个默认形状被修改，必须同步修改本文件和运行手册**。
这些数字不是说明性文字，而是数据构建、训练、单步 runner、离线评测共同遵守的
接口契约。换 Qwen 基础模型（不同 `num_key_value_heads × head_dim`）时，
DiT `hidden_dim` 与 `n_heads` 必须同步改动让 `hidden_dim / n_heads == Qwen head_dim`，
否则 `DiTMoT.forward` 第一步就会抛 `RuntimeError`。

## 训练

训练入口：

```text
qwen3vl_local/goalgen/train.py
qwen3vl_local/goalgen/train.sh
```

启动模式：

- `check`：单卡，跑 2 个优化器步，做最小冒烟测试。
- `single`：单卡训练。
- `ddp`：多卡 DDP。自动挑可用 GPU、自动选空闲端口。

Optimizer 设置（v2 双 optimizer）：

- **Muon**（接管 2D 权重矩阵 = attention/MLP/AdaLN 的所有 Linear）：
  - 学习率默认 `2e-3`（比 AdamW 大 10×；Newton-Schulz 正交化后单位步长更稳）。
  - momentum `0.95`、Nesterov `True`、Newton-Schulz 5 步。
  - weight decay `0.0`（2D 矩阵的 Muon 通常不挂 wd）。
- **AdamW**（接管其它 = Conv2d patch.proj、norm 1D weight、embeddings、null_lang_k/v、t_mlp 等）：
  - 学习率默认 `2e-4`，weight decay `0.01`，betas `(0.9, 0.95)`。
- 两个 optimizer 共享同一份 cosine + warmup ratio `0.05` 的 LR 调度（`_DualScheduler` 同步驱动）。
- `t_sampler=logit_normal`（SD3 配方，t 集中在 0.5 附近）。
- `z0_prior_alpha=1.0, z0_prior_sigma=1.0`，起点为当前帧 latent + 噪声。
- CFG：训练 `cfg_drop_prob=0.1`，推理 `cfg_scale=2.0`。
- EMA：`ema_decay=0.9999`，val / image log / eval / probe / runner 默认使用 EMA 权重。
- VAE latent stats：训练启动时在 `train.jsonl` 同目录缓存 `latent_stats.json`，encode 后标准化、decode 前反标准化。

Runtime 优化（v2 默认全部开启，可通过 sh 环境变量关闭）：

- `torch.compile(dit)`：mode=default、dynamic=True、fullgraph=False；首次 step
  会编译 30-90 秒。`COMPILE_DIT=0` 关闭。
- Gradient checkpointing：per-block + `use_reentrant=False`，显存省 ~40% / wall-clock
  多 ~30%。`GRAD_CKPT=0` 关闭。
- Flash-attention 由 PyTorch SDPA 自动调度（H100/A100 上自动走 flash 后端）。

检查点中只保存 DiT 自身的权重以及优化器 / 调度器状态
（仅用于断点续训和诊断，不包含冻结模型）。

## 参数预算（v2 默认配置）

下表只统计 DiT-MoT（Qwen 与 VAE 冻结、不计入）：

| 模块 | 大致参数量 |
|---|---|
| Shared Patchify（Conv2d 4 -> 1024，kernel=4） | ~66K |
| Type embedding（2, 1024） | 2K |
| Frame embedding（8, 1024） | 8K |
| CFG null KV（12 层 × K+V × `[1,8,1,128]`） | ~24K |
| Timestep MLP（cond_dim=256，4x） | ~0.5M |
| 单层 JointAttention（q/k/v/o，**已删 lang_k/v_proj**） | ~4.2M |
| 单层 SwiGLU MLP（1024 → 4096 × 2 gate/up + 4096 → 1024 down） | ~12.6M |
| 单层 AdaLN modulation（256 → 6144） | ~1.6M |
| 单层 q_norm/k_norm（RMSNorm head_dim=128，仅 vision） | 256 |
| 单个 block 合计 | ~18.4M |
| 12 个 block 合计 | ~221M |
| Final norm + final_mod | ~1.6M |
| Unpatchify Linear(1024 → 4×16=64) | ~66K |
| **DiT 总计（粗算）** | **~225M** |

`bfloat16` 权重大约占 450MB。Qwen 4B 与 VAE 自身需要额外显存，但**不产生梯度**。

参数量较 v1（~120M）增长 ~1.88×，主要来自 hidden 768→1024（attention/MLP 全线扩张）
以及 MLP 从 GELU 2-Linear → SwiGLU 3-Linear（×1.5）。但 patch=4 后视觉 token 数砍到
1/4，attention FLOPs 净下降约 4×，wall-clock 仍更快。

## 显存预算（H20 96GB，batch=1，DiT 用 bfloat16）

| 阶段 | 估算显存 |
|---|---|
| Qwen3-VL-4B-Instruct（bf16） | ~8GB |
| VAE（fp32） | ~0.4GB |
| Qwen prefill（~2300 token，36 层 KV） | ~3-4GB |
| 分段后的 12 个 KV（select_last，bf16） | ~1GB |
| DiT-MoT 权重（bf16） | ~0.25GB |
| DiT EMA shadow（fp32） | ~0.5GB |
| DiT 前向激活（默认 `select_last` 下：视觉 N=8640、语言 S~2300 / 每层） | 中等；若改用 `concat_layers`，语言 token 数会涨到约 6900，对应激活也线性放大 |
| DiT 反传 + AdamW 状态 | ~2GB |
| **单卡训练总计（batch=1）** | **~17-20GB** |

注意：DDP 下每个进程各自加载 Qwen + VAE，8 卡情况下 Qwen 占用大约是单
卡的 8 倍。这在 v1 里是浪费但可以接受；v2 必须把分段 KV、history latent、
target latent 都**离线缓存到磁盘**，避免每个进程重复计算。

## 风险与回退预案

| 风险 | 触发条件 | 应对方式 |
|---|---|---|
| KV 段数 != DiT 层数 | DiT 前向内部抛出 `pooled_kv segments ... != DiT layers ...` | 保持 `--num-layers` 与分段函数的 `num_segments` 一致。两侧默认都是 12。 |
| 换 Qwen 模型后 K/V 头数 / head_dim 变化 | `DiTMoT.forward` 抛 `pooled_kv[0] K 形状 ... 与 DiT (n_heads=..., head_dim=...) 不匹配` | v2 起 DiT 直接消费 Qwen K/V，没有 lang_k/v_proj 兜底；必须改 `--hidden-dim` / `--n-heads` 让 `hidden_dim // n_heads == Qwen head_dim`、`n_heads == Qwen num_key_value_heads`。 |
| 完整 KV 模式显存溢出 | `concat_layers` 会把每个 DiT 层的语言显存放大约 3 倍 | 默认保持 `QWEN_KV_SEGMENT_MODE=select_last`；只在做消融实验时切到 `concat_layers`；`mean` 仅作历史消融保留。 |
| `bfloat16` Qwen KV 与 `float32` DiT 之间数据类型不一致 | SDPA 内部抛 `RuntimeError` | 训练器会在前向之前把分段 KV、z_history、z1 显式 `.to(dtype=dit_dtype)`。保持 `--qwen-dtype` 与 `--dit-dtype` 兼容，或者依赖这一行显式转换。 |
| Qwen 预填充序列过长（LEAD `num_frames > 4`） | H20 96GB 上 Qwen 预填充显存溢出 | 减小 `--num-frames`，或者用 `--qwen-dtype float16` 降低 Qwen 推理显存。 |
| 目标关键帧在磁盘上不存在 | 抛出"RGB 图像不存在"类错误 | 在能挂载到 LEAD 数据的机器上重新跑 `build_dataset.py`。 |
| 给 `--status` / `--subgoal` 传了错误的字符串 | runner 抛错，因为 STATUS/SUBGOAL 违反事件链 | 优先只指定 STATUS，让 runner 自己推 SUBGOAL；训练数据走 `build_dataset.py` 生成的 jsonl。 |
| 目标关键帧不在未来 | runner 抛出"target_frame 必须在未来" | 使用数据集自动构出的样本，或者保证 `--target-frame > --anchor`。 |
| DDP 进程卡死 | 某个进程因切片长度对不齐多跑了一次 `loss.backward()` | `usable_per_epoch = (N // world_size) * world_size` 已经保证所有进程切片等长。**不要去改切片逻辑**。 |
| 训练慢、瓶颈在 Qwen 预填充 | 每个样本都要做一次完整 Qwen 前向 | 这是 v1 已知瓶颈。v2 必须把分段 KV、history/target latent 全部预先算好并缓存到磁盘。 |

## 默认形状同步约定（修改任何几何参数之前必读）

上面那张默认形状表是数据构建器与训练器共同遵守的契约。任何一个
值发生变化时，必须同步执行：

1. 更新本文件的默认配置小节。
2. 同步更新 `GOALGEN_RUN.md` 里对应的数字。
3. 如果改的是 `RGB_FRAME_COUNT` 或 `RGB_FRAME_STEP`，**必须重跑**
   `build_dataset.py`（jsonl 里编码了具体的帧索引）。
4. 如果改的是 `patch_size`, `hidden_dim`, `n_heads` 或 `num_layers`，
   必须**从零重训** DiT（老检查点全部不兼容）。

不要让代码与本文档发生漂移。

## 已知局限

- v1 在线计算 Qwen KV 与 VAE 潜变量，逻辑简单但**慢**。
- DiT 现在直接消费 `history_rgb_paths` 里所有历史潜变量。
- 仍未做完整 KV / history latent / target latent 离线缓存；Qwen/VAE 仍在线计算，因此训练慢。
- 已有 EMA / CFG / latent stats 缓存 / 解码图像评测；第二轮默认都开启。

## v1 / v2 边界

v1 显式**不做**的事情（写在这里防止未来 agent 擅自扩张范围）：

- 不做分段 KV / 潜变量的离线缓存。
- 不做完整离线 KV / latent 数据集缓存（只缓存 per-channel latent stats）。
- 不做多目标监督（每条样本就一个子目标关键帧）。

**当前"v2"的范围仅限数据集分布裁剪 + 默认从 v1 warm start**（见上一节 `--mode v2`
和下面 `VERSION=v2`）：

- 复用 `build_dataset.py` 同一脚本，靠 `--mode` 切换 transition 集合；
- 输出 jsonl 字段 schema 与 v1 完全一致（`scenario / run_id / anchor / status /
  subgoal / target_frame / history_rgb_paths / current_rgb_path / target_rgb_path /
  memory`），train / eval / probe 入口不动；
- DiT 架构、Qwen KV 分段、VAE 编码、CFG、EMA 等所有训练侧配置保持 v1 默认；
- `train.sh` 新增 `VERSION` env：`VERSION=v2` 时自动把数据切到 `goalgen_v2_data/`、
  产物落到 `goalgen_v2_dit/`、并把 `--init-from-ckpt` 默认指向 `goalgen_v1_dit/latest/best.pt`
  （latest 是脚本自动维护的 symlink，下条），做 **DiT 权重 + EMA shadow 双 strict=True
  warm start**（不接 optimizer / scheduler / step），实质等同于"换数据子集 + 继承架构
  权重"重新训练。架构默认完全沿用 v1，strict=True 在这条路径上是"防 env 漂移"的
  护栏（v1→v2 默认配置不会触发）。
- `train.sh` 同时引入 **run 子目录隔离 + latest symlink**（v1/v2 通用）：每次启动
  把 OUTPUT_DIR 自动改写成 `${OUTPUT_DIR_BASE}/run_${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}/`，
  所有 ckpt / TB events / eval 产物都落在 run 子目录里；base 顶层维护一个相对路径的
  `latest` symlink 指向最新 run。HF weights 缓存放 base 层共享。`NO_RUN_SUBDIR=1`
  退回老的"顶层覆盖"行为，仅用于排查脚本兼容性。这样反复跑同一 VERSION 的训练
  不会再覆盖旧 ckpt，TB 也能天然多 run 对比。

后续需要"真正的 v2 训练栈"（例如离线缓存分段 KV / latent、多目标监督、新的损失项），
应在 v1 远端冒烟测试通过后另起 `train_v2.py` 等文件，并按项目规则同步白名单。
