# GoalGen v1 设计方案

## 目标

GoalGen v1 训练一个在 VAE 潜变量空间上工作的 DiT-MoT 生成器。它不直接预测
RGB 像素，而是预测"子目标关键帧"对应的潜变量速度场；条件信息来自冻结的
Qwen3-VL-Instruct 预填充后得到的键值缓存。这样做的核心原因是：Qwen 负责把
历史视觉、当前状态、下一子目标压进语言/视觉上下文，DiT 只学习"在这个上下文下，
未来子目标画面应该往哪里走"。

整条数据流如下：

```text
历史 RGB        ->  冻结 Qwen prefill   ->  分段的 token-level KV
历史 RGB        ->  冻结 VAE            ->  z_history
子目标关键帧 RGB ->  冻结 VAE            ->  z1
z0 ~ N(0, I)，z_t = (1 - t) * z0 + t * z1
DiT-MoT(z_t, z_history, t, 分段 KV)   ->  v_pred
loss = MSE(v_pred, z1 - z0)
```

Qwen 和 VAE 全程冻结，**只有 DiT-MoT 参与训练**。这条边界非常重要：
训练损失只能更新 DiT，不能让语言模型或 VAE 被反向传播污染，否则后续所有
缓存、评测和旧检查点都不可比。

## 数据

数据集构建脚本：

```text
qwen3vl_local/goalgen/build_dataset_v1.py
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

默认每个场景最多取 `1000` 条样本。如需保留所有合法锚点帧，传
`--samples-per-scenario 0`。

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

## 模型默认配置

- LEAD RGB 尺寸：`1152 x 384`
- VAE latent 形状：`[B, 4, 48, 144]`
- DiT 图块大小：`2`
- DiT 上的潜变量图块网格：`(24, 72) = 1728` 个图块 token
- 数据构建器默认 4 帧历史时，DiT 视觉 token 总数：
  `z_t` token + `4 * z_history` token = `8640`
- DiT 隐藏维度：`768`
- DiT 注意力头数：`12`
- DiT 层数：`12`
- Qwen 分段 KV：默认 `select_last` 模式。Qwen 共 36 层，被切成 12 段，
  每段取其 3 层小组里的最后一层（token-level K/V，shape 为
  `[B, 8, S, 128]`）。`concat_layers` 只保留为消融实验选项。
- Qwen KV 输入维度：`8 * 128 = 1024`（在 `select_last` 与 `concat_layers`
  下都是同一个值）

**如果以上任何一个默认形状被修改，必须同步修改本文件和运行手册**。
这些数字不是说明性文字，而是数据构建、训练、单步 runner、离线评测共同遵守的
接口契约。

## 训练

训练入口：

```text
qwen3vl_local/goalgen/train_v1.py
qwen3vl_local/goalgen/train_v1.sh
```

启动模式：

- `check`：单卡，跑 2 个优化器步，做最小冒烟测试。
- `single`：单卡训练。
- `ddp`：多卡 DDP。自动挑可用 GPU、自动选空闲端口。

Optimizer 设置：

- AdamW，**只更新 DiT 参数**。
- 学习率 `1e-4`。
- weight decay `0.01`。
- cosine 学习率调度。
- warmup ratio `0.02`。

检查点中只保存 DiT 自身的权重以及优化器 / 调度器状态
（仅用于断点续训和诊断，不包含冻结模型）。

## 参数预算（默认配置）

下表只统计 DiT-MoT（Qwen 与 VAE 冻结、不计入）：

| 模块 | 大致参数量 |
|---|---|
| Patchify x2（Conv2d 4 -> 768，kernel=2） | ~12K × 2 |
| Type embedding（2, 768） | 1.5K |
| Timestep MLP（cond_dim=256，4x） | ~0.5M |
| 单层 JointAttention（q/k/v/o + lang_k/v） | ~3.9M |
| 单层 MLP（768 -> 3072 -> 768） | ~4.7M |
| 单层 AdaLN modulation（256 -> 4608） | ~1.2M |
| 单个 block 合计 | ~9.8M |
| 12 个 block 合计 | ~118M |
| Unpatchify Linear(768 -> 16) | ~12K |
| **DiT 总计（粗算）** | **~120M** |

`bfloat16` 权重大约占 240MB。Qwen 4B 与 VAE 自身需要额外显存，但**不产生梯度**。

## 显存预算（H20 96GB，batch=1，DiT 用 bfloat16）

| 阶段 | 估算显存 |
|---|---|
| Qwen3-VL-4B-Instruct（bf16） | ~8GB |
| VAE（fp32） | ~0.4GB |
| Qwen prefill（~2300 token，36 层 KV） | ~3-4GB |
| 分段后的 12 个 KV（select_last，bf16） | ~1GB |
| DiT-MoT 权重（bf16） | ~0.25GB |
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
| `language_kv_input_dim` 硬编码但基础模型变了 | DiT 第一层语言投影矩阵形状不匹配 -> `RuntimeError` | `train_v1.py` 现在会从第一个样本的分段 KV 自动推断该值。传 `--language-kv-input-dim auto`（默认）即可。 |
| 完整 KV 模式显存溢出 | `concat_layers` 会把每个 DiT 层的语言显存放大约 3 倍 | 默认保持 `QWEN_KV_SEGMENT_MODE=select_last`；只在做消融实验时切到 `concat_layers`；`mean` 仅作历史消融保留。 |
| `bfloat16` Qwen KV 与 `float32` DiT 之间数据类型不一致 | SDPA 内部抛 `RuntimeError` | 训练器会在前向之前把分段 KV、z_history、z1 显式 `.to(dtype=dit_dtype)`。保持 `--qwen-dtype` 与 `--dit-dtype` 兼容，或者依赖这一行显式转换。 |
| Qwen 预填充序列过长（LEAD `num_frames > 4`） | H20 96GB 上 Qwen 预填充显存溢出 | 减小 `--num-frames`，或者用 `--qwen-dtype float16` 降低 Qwen 推理显存。 |
| 目标关键帧在磁盘上不存在 | 抛出"RGB 图像不存在"类错误 | 在能挂载到 LEAD 数据的机器上重新跑 `build_dataset_v1.py`。 |
| 给 `--status` / `--subgoal` 传了错误的字符串 | runner 抛错，因为 STATUS/SUBGOAL 违反事件链 | 优先只指定 STATUS，让 runner 自己推 SUBGOAL；训练数据走 `build_dataset_v1.py` 生成的 jsonl。 |
| 目标关键帧不在未来 | runner 抛出"target_frame 必须在未来" | 使用数据集自动构出的样本，或者保证 `--target-frame > --anchor`。 |
| DDP 进程卡死 | 某个进程因切片长度对不齐多跑了一次 `loss.backward()` | `usable_per_epoch = (N // world_size) * world_size` 已经保证所有进程切片等长。**不要去改切片逻辑**。 |
| 训练慢、瓶颈在 Qwen 预填充 | 每个样本都要做一次完整 Qwen 前向 | 这是 v1 已知瓶颈。v2 必须把分段 KV、history/target latent 全部预先算好并缓存到磁盘。 |

## 默认形状同步约定（修改任何几何参数之前必读）

上面那张默认形状表是数据构建器与训练器共同遵守的契约。任何一个
值发生变化时，必须同步执行：

1. 更新本文件的默认配置小节。
2. 同步更新 `GOALGEN_V1_RUN.md` 里对应的数字。
3. 如果改的是 `RGB_FRAME_COUNT` 或 `RGB_FRAME_STEP`，**必须重跑**
   `build_dataset_v1.py`（jsonl 里编码了具体的帧索引）。
4. 如果改的是 `patch_size`, `hidden_dim`, `n_heads` 或 `num_layers`，
   必须**从零重训** DiT（老检查点全部不兼容）。

不要让代码与本文档发生漂移。

## 已知局限

- v1 在线计算 Qwen KV 与 VAE 潜变量，逻辑简单但**慢**。
- DiT 现在直接消费 `history_rgb_paths` 里所有历史潜变量。
- 还没有 EMA / CFG / 缓存 latent 数据集这些设施。
- 还没有解码图像层面的评测；目前只用损失与速度余弦做早期健康检查。

## v1 / v2 边界

v1 显式**不做**的事情（写在这里防止未来 agent 擅自扩张范围）：

- 不做分段 KV / 潜变量的离线缓存。
- 不给 DiT 加 EMA。
- 不做无分类器引导（语言流不做随机丢弃）。
- 不做多目标监督（每条样本就一个子目标关键帧）。
- 不做解码图像层面的评测，只跟踪损失 + 速度余弦。

v2 在 v1 的前向单步验证 + 训练冒烟测试在远端 H20 集群跑通之后再启动。
