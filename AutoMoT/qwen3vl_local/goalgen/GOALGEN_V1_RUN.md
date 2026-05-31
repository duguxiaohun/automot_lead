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
  --samples-per-scenario 0 \
  --output-dir checkpoints/goalgen_v1_data
```

数据构建器沿用 SFT v1 的时间线思路，但目标从"文本标签"换成"未来子目标关键帧"：

- `status` 是 `anchor` 帧的 GT status；
- `subgoal` 是 `status` 之后的下一个事件；
- `target_frame` 是 `subgoal` 触发所在的 keyframe；
- 样本必须满足 `target_frame > anchor`（子目标只能在未来）。

默认 `--samples-per-scenario 0` 会保留所有合法锚点帧；如需小规模消融，再显式传
正整数上限。

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

第二轮默认启用：共享 patchify、EMA `0.9999`、logit-normal t 采样、CFG drop `0.1` /
推理 scale `2.0`、VAE latent per-channel 标准化、z_current prior 起点、LR `2e-4`、
warmup `0.05`。这些改动与旧 ckpt 不兼容，按方案 A 从零重训。
首次训练会在 `checkpoints/goalgen_v1_data/latent_stats.json` 缓存 latent mean/std；
后续训练默认复用，必要时传 `--recompute-latent-stats` 重算。

**默认跑基础 Qwen，不挂任何 LoRA / 适配器。** 当前 GoalGen 阶段先不要传
`QWEN_ADAPTER_DIR` / `--qwen-adapter-dir`，直接用上面命令即可。

### 2.0.1 LoRA / PEFT 适配器（暂不启用，仅保留边界）

> **默认不挂 LoRA**：训练 / 评测 / 单步 runner 三个入口的 `--qwen-adapter-dir` 默认为空字符串，
> 这种情况下完全走基础 Qwen，不会导入 `peft`，也不会读适配器目录。
> 当前训练生成模型阶段**不需要**做任何事，直接 `bash qwen3vl_local/goalgen/train_v1.sh ddp`
> 即可。下面只记录以后如果重新启用 adapter 时需要同步检查的边界；常规命令不要照抄这里。

如果以后确实要让 GoalGen 接 SFT v1 微调后的语言编码，训练 / 评测 / runner 必须显式传
同一个 adapter 目录；当前手册的主流程不提供可直接照抄的 LoRA 命令，避免误触发 `peft`
导入。

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

当前阶段不要给 eval / runner 传 adapter；如果后续恢复该分支，再单独补一组带 adapter
的命令，并同步检查 DiT ckpt 里的 `qwen_adapter_dir` 一致性。

输出：

```text
checkpoints/goalgen_v1_dit/latest.pt
checkpoints/goalgen_v1_dit/checkpoint-000200/goalgen_v1.pt
checkpoints/goalgen_v1_dit/tb/                # TensorBoard event 文件
```

## 2.1 TensorBoard（推荐用 tools/tb_serve.sh 一条命令）

训练 + eval 都往 `OUTPUT_DIR` 下平铺保存：

```
checkpoints/goalgen_v1_dit/
├─ checkpoint-*/ + latest.pt  DiT 权重
├─ tb/                        训练 TB events（train/* val/* samples/pred_vs_gt）
├─ eval/                      eval_v1.py 产物
│  ├─ eval_v1_summary.json    聚合指标 + _metric_doc 说明
│  ├─ eval_v1_perline.jsonl   每条样本一行
│  ├─ samples/                前 N 条 pred / gt 分开 PNG（轻量预览）
│  └─ cases/                  小样本完整 dump（compare.png + 输入图文）
├─ eval_tb/<ckpt-tag>/        eval_v1.py 写的指标 scalar + pred_vs_gt 图（每 ckpt 一个 run）
└─ eval_cases/                probe_v1.py 随机场景 case dump（含 euler trace）
```

启动 TB 指 `--logdir checkpoints/goalgen_v1_dit` 时，左侧 run 列表会同时显示 `tb`
（训练）和 `eval_tb/<ckpt>`（每次 eval 一个）。

**推荐：用一条命令搞定**（自动选端口 + bind_all）：

```bash
# 远端，在 AutoMoT/ 目录下
bash tools/tb_serve.sh checkpoints/goalgen_v1_dit
```

`tb_serve.sh` 会打印类似：

```
[tb] TensorBoard 已启动 → 在本地浏览器直接打开：
      http://localhost:41273
```

VSCode Remote 会自动把端口转发到本地，浏览器直接打开 URL 即可。Ctrl-C 关本脚本
TB 服务一起退。固定端口：`TB_PORT=6008 bash tools/tb_serve.sh ...`。

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

`--save-root` **必填**，产物落到 `<save_root>/eval/` 与 `<save_root>/eval_tb/<run_tag>/`。
推荐 `--save-root` = 训练 OUTPUT_DIR（`checkpoints/goalgen_v1_dit`）。

### 3.5.1 两个入口

| 脚本 | 跑全集出聚合指标 | 每条样本完整 dump | 额外提供 |
|---|---|---|---|
| `qwen3vl_local/goalgen/eval_v1.py` | ✅ | ✅（默认 `--max-samples > 0` 时开） | summary + perline + 可选 TB（默认开）|
| `qwen3vl_local/goalgen/probe_v1.py` | ❌（按 scenario 抽样） | ✅ | per-step euler trace（v_cos / z_l2 单调下降曲线）|

**简单决策**：先跑 `eval_v1.py --max-samples 100` 就能拿到 compare.png 三联对比 +
全部指标；只有想看"Euler 32 步轨迹是否单调收敛"时再跑 probe。

`eval_v1.py` 对每条样本：teacher-forced Qwen 预填充 → VAE 编码 → Euler 32 步采样 →
VAE 解码 → 5 指标 + 输入图文 + pred/gt 对比图全部本地保存。

### 3.5.2 eval_v1.py

```bash
# 推荐：小样本 + 完整 dump（每条样本一个目录，含 compare.png 三联图）
python qwen3vl_local/goalgen/eval_v1.py \
  --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
  --dit-checkpoint checkpoints/goalgen_v1_dit/latest.pt \
  --save-root checkpoints/goalgen_v1_dit \
  --max-samples 100

# 跑全集只出聚合指标 + TB（不 dump，磁盘友好）
python qwen3vl_local/goalgen/eval_v1.py \
  --dit-checkpoint checkpoints/goalgen_v1_dit/latest.pt \
  --save-root checkpoints/goalgen_v1_dit

# 多卡分片
torchrun --standalone --nproc_per_node=4 qwen3vl_local/goalgen/eval_v1.py \
  --dit-checkpoint checkpoints/goalgen_v1_dit/latest.pt \
  --save-root checkpoints/goalgen_v1_dit
```

单进程 eval 默认会在加载 Qwen/VAE 前调用 `nvidia-smi`，按 `memory.used`、
`utilization.gpu` 从小到大自动选择一张最空闲的 GPU，并设置
`CUDA_VISIBLE_DEVICES=<选中卡>`；进程内仍使用 `cuda:0`。如果外部已经设置
`CUDA_VISIBLE_DEVICES`、显式传 `--gpu N`，或使用 `torchrun`，脚本会尊重外部设置。
要关闭自动选卡：`GOALGEN_EVAL_DISABLE_AUTO_GPU=1 python qwen3vl_local/goalgen/eval_v1.py ...`。

**关键参数**：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--save-root` | （必填） | 产物根目录，通常等于训练 OUTPUT_DIR |
| `--run-tag` | 自动（`ckpt200` / `latest` 等） | TB run 子目录名 |
| `--max-samples` | 0 = 全集 | 截断 val 样本数 |
| `--full-dump` / `--no-full-dump` | 自动 | 默认 `--max-samples > 0` 时开 |
| `--full-dump-limit N` | 0 = 不限 | dump 上限，防止误开铺满磁盘 |
| `--euler-steps` | 32 | Euler 采样步数（rectified flow 下 32 通常足够）|
| `--cfg-scale` | 2.0 | classifier-free guidance 强度；训练默认 drop=0.1 |
| `--z0-prior-alpha` / `--z0-prior-sigma` | 1.0 / 1.0 | 推理起点 = 当前帧 latent + 噪声，需与训练一致 |
| `--use-ema` / `--no-use-ema` | True | 默认读取 ckpt 里的 EMA 权重；旧 ckpt 无 EMA 时回退 raw 权重并 warning |
| `--image-dump-count` | 32 | `samples/` 目录里轻量预览 PNG 的条数（与 cases/ 完整 dump 独立）|
| `--no-tb` | False | 关闭 TB（默认开；步骤二 TB 是项目主入口）|
| `--qwen-adapter-dir` | 空字符串 | 默认跑基础 Qwen 且不会导入 `peft`；当前 GoalGen 阶段先不要传 |
| `--gpu` | 0 | 进程内 GPU 编号；未显式传时会先自动选择空闲物理 GPU 并映射为 `cuda:0` |

**产物布局**（每次 eval 后）：

```
checkpoints/goalgen_v1_dit/eval/
├─ eval_v1_summary.json        聚合指标 + _metric_doc 含义说明
├─ eval_v1_perline.jsonl       每条样本一行（5 指标 + PNG 路径）
├─ samples/                    轻量预览：前 N 条 pred / gt 分开 PNG
│  ├─ 00000_pred.png
│  ├─ 00000_gt.png
│  └─ ...
└─ cases/                      完整 dump（小样本时自动开）
   └─ 00017__Accident__Town03_Rep0_route_001783__anchor12/
      ├─ inputs/
      │  ├─ system_prompt.txt          teacher-forced system 原文
      │  ├─ user_prompt.txt            teacher-forced user 原文（含 ground-truth state）
      │  ├─ memory.json                DrivingMemory（scenario / status / subgoal / event_sequence）
      │  ├─ history_00.jpg ... 03.jpg  history RGB，**复制**到本地
      │  └─ target_raw.jpg             真值 keyframe 原图（VAE 输入前）
      ├─ outputs/
      │  ├─ pred.png                   DiT 采样 + VAE 解码（模型生成）
      │  ├─ target_vae_recon.png       真值经 VAE encode→decode（生成质量天花板）
      │  └─ compare.png                **横拼三联图：target_raw | pred | target_vae_recon**
      ├─ metrics.json                  单 case 5 指标 + _metric_doc
      ├─ step.json                     完整元信息（dit_ckpt / qwen_adapter / euler_steps / seed）
      └─ summary.md                    一页 markdown，**顶部直接引用 compare.png**
```

`summary.md` 渲染后顶部：

```markdown
## 最关心的可视化：target_raw | pred | target_vae_recon
![compare](outputs/compare.png)

- target_raw：真值 keyframe 原图
- pred：DiT 采样 + VAE 解码（模型生成的子目标图像）
- target_vae_recon：真值经 VAE encode→decode；生成质量天花板（pred 不会比这清）
```

`compare.png` 是用户最关心的可视化——一眼能看出"模型生成的子目标 vs 真值"差距，
不用打开三个文件分别看。

### 3.5.3 metrics 字段

`eval_v1_summary.json` 顶层带 `_metric_doc`：

```json
{
  "_metric_doc": {
    "latent_mse": "MSE(z1_pred, z1_gt)；与训练损失同口径，越小越好",
    "latent_cos": "cosine(z1_pred, z1_gt)；越接近 1 越好",
    "pixel_l1": "解码 RGB [-1,1] L1；越小越好；地板 = VAE 重建误差",
    "psnr": "解码 RGB PSNR (dB)；越大越好；地板 = VAE 重建 PSNR",
    "velocity_cos": "5 个固定 t 上 v_pred vs v_target cosine 平均；越接近 1 越好"
  },
  "overall": {
    "latent_mse_mean": 0.34, "latent_mse_std": 0.12,
    "latent_cos_mean": 0.78, "latent_cos_std": 0.09,
    "psnr_mean": 21.4, ...
  },
  "by_scenario": { ... }
}
```

**`pixel_l1 / psnr` 的绝对值意义有限**——下限取决于 VAE 重建质量本身。做 base /
step-200 / step-1000 横向对比时看 delta；单看绝对值不要直接当"生成质量"。
`target_vae_recon.png` 给出 VAE 重建天花板作参照。

### 3.5.4 TB tag（写到 eval_tb/&lt;run_tag&gt;/）

| Tag | 含义 |
|---|---|
| `eval/latent_mse` / `latent_cos` / `pixel_l1` / `psnr` / `velocity_cos` | overall mean，按 ckpt step 形成横向曲线 |
| `eval/<...>_std` | overall std |
| `eval_by_scenario/<sc>/<metric>` | 拆场景看哪些场景拉低指标 |
| `eval/pred_vs_gt` | image 面板：前 8 条 pred + gt 交错排 |

`--logdir checkpoints/goalgen_v1_dit` 一条命令同时看训练 `tb/` 和多个 ckpt 的
`eval_tb/<ckpt>/`；不同 ckpt 在 TB 左侧 run 列表里并列。

### 3.5.5 probe_v1.py（深度诊断）

```bash
python qwen3vl_local/goalgen/probe_v1.py \
  --dit-checkpoint checkpoints/goalgen_v1_dit/latest.pt \
  --save-root checkpoints/goalgen_v1_dit \
  --num-per-scenario 4 --seed 0

# 多 ckpt 横向对比：同 seed + --case-suffix 防覆盖
python qwen3vl_local/goalgen/probe_v1.py \
  --dit-checkpoint checkpoints/goalgen_v1_dit/checkpoint-000500/goalgen_v1.pt \
  --save-root checkpoints/goalgen_v1_dit \
  --num-per-scenario 4 --seed 0 --case-suffix "_ckpt500"
```

probe 的 case 目录布局（与 eval cases 类似，但多 `euler_trace.json`）：

```
checkpoints/goalgen_v1_dit/eval_cases/<scenario>__<run>__<anchor>/
├─ input_history/00.jpg ... 03.jpg
├─ target_raw.jpg / target_vae_recon.png / pred.png
├─ euler_trace.json   ← per-step t / v_cos_vs_gt_direction / z_l2_to_gt
├─ memory.json / metrics.json / meta.json
└─ overview.md
```

`euler_trace.json` 里：
- `v_cos_vs_gt_direction`：每步 v_pred 与真值方向 `z1_gt - z_init` 的 cosine；理想轨迹整段接近 1
- `z_l2_to_gt`：当前 z_t 到真值 z1_gt 的 L2 距离；理想应单调下降

什么时候用 probe 而不是 eval cases：
- 想看 Euler 32 步轨迹是否单调收敛（v_cos 曲线是否一路 ≈1、z_l2 是否单调下降）
- 怀疑 DiT 采样在中段"走偏"——eval cases 只给最终输出，probe 给每一步

### 3.5.6 训练完小批量诊断 → 打包发给 AI 审阅（推荐流程）

**适用场景**：DiT 训练刚结束、还没决定挑哪个 ckpt、想快速判断"生成出来的图是不是已经像样、问题是 VAE 瓶颈还是 DiT 没学好"。
不要等全集 eval 跑完再判断方向，先用小批量 + AI 审阅做粗筛。

**这一节解决的关键问题**：图像生成不像文本可以 print 出来扫一眼——pred 模糊？颜色漂移？结构缺失？只看 `pixel_l1=0.18` 这种数字判断不了原因，必须把 `compare.png` 三联图给人 / AI 看才能说出"是 VAE 瓶颈、还是 DiT 没收敛、还是 prompt 没对齐"。

#### 3.5.6.1 一条命令产出诊断材料

```bash
# 推荐：每个场景抽 3 条，~30 个 case，3–5 分钟（基本是采样时间）
python qwen3vl_local/goalgen/probe_v1.py \
  --dit-checkpoint checkpoints/goalgen_v1_dit/latest.pt \
  --save-root checkpoints/goalgen_v1_dit \
  --num-per-scenario 3 --seed 42 \
  --case-suffix "_latest"

# 想横向比 ckpt-500 vs latest：同 seed + 不同 case_suffix 防覆盖
python qwen3vl_local/goalgen/probe_v1.py \
  --dit-checkpoint checkpoints/goalgen_v1_dit/checkpoint-000500/goalgen_v1.pt \
  --save-root checkpoints/goalgen_v1_dit \
  --num-per-scenario 3 --seed 42 \
  --case-suffix "_ckpt500"
```

- `--num-per-scenario 3` × ~10 个场景 ≈ 30 个 case
- 同 `--seed 42`：不同 ckpt 跑出的样本是同一批，可以拿同一 case 直接对比
- `--case-suffix` 防互相覆盖

#### 3.5.6.2 产物结构（这是要发给 AI 看的"诊断包"）

```
checkpoints/goalgen_v1_dit/eval_cases/
├─ Accident__Town03_Rep0_route_001783__anchor12_latest/
│  ├─ input_history/00.jpg ... 03.jpg     ← 历史 4 帧（条件输入）
│  ├─ target_raw.jpg                      ← 真值 keyframe 原图（VAE 输入前）
│  ├─ target_vae_recon.png                ← 真值经 VAE encode→decode（生成质量天花板）
│  ├─ pred.png                            ← DiT 采样 + VAE 解码（**最关心的输出**）
│  ├─ euler_trace.json                    ← per-step v_cos / z_l2 曲线
│  ├─ memory.json                         ← DrivingMemory（scenario / status / subgoal / completed_events）
│  ├─ metrics.json                        ← 单 case 5 指标 + _metric_doc
│  ├─ meta.json                           ← dit_ckpt / qwen_adapter / euler_steps / seed
│  └─ overview.md                         ← 一页 markdown，串起上面全部
├─ Accident__..._ckpt500/                 ← 同一 case 的 ckpt-500 版本（同 seed）
│  └─ （同上结构）
├─ Construction__... × 数十个 case
└─ _index_latest.jsonl / _index_ckpt500.jsonl   ← 全部 case 的扁平索引
```

eval_v1.py 的 cases/ 比 probe 多一张 `compare.png`（target_raw | pred | target_vae_recon 横拼三联），如果走 `eval_v1.py --max-samples 30 --full-dump` 而不是 probe，**`compare.png` 才是 AI 审阅时最先看的那张**。两种产物都可以送审，结构基本一致。

#### 3.5.6.3 怎么发给 Claude / AI

| 想问的问题 | 最少要带的文件 | 怎么发 |
|---|---|---|
| "模型整体生成效果怎么样、有没有跑歪" | 整个 `eval_cases/` zip（或 eval/cases/ zip） | 上传 zip，让 AI 抽看几个 case 的 PNG |
| "生成图模糊 / 有伪影 / 颜色漂移" | 任一 case 的 `compare.png`（或 `target_raw.jpg` + `pred.png` + `target_vae_recon.png` 三张并排） | 直接把图拖进对话 |
| "是 VAE 瓶颈还是 DiT 没学好" | 同一 case 的 `target_vae_recon.png` + `pred.png` | 两张图并排发，AI 会比较两边差距 |
| "Euler 采样轨迹是不是收敛" | 任一 case 的 `euler_trace.json` + `pred.png` | 拖文件 |
| "ckpt-500 vs latest 谁更好" | 同 seed 同 case 的两个目录 `..._ckpt500/` 和 `..._latest/` 各自 `pred.png` + `metrics.json` | 拖两个文件夹 |
| "某场景特别差（如 Construction）" | 该 scenario 下 3–5 个 case 的 `overview.md` + `compare.png` | 拖文件 |

最省事的姿势：**把整个 `eval_cases/` zip 上传**，问"看看生成效果怎么样、问题在哪里"。AI 会按下面 3.5.6.4 的清单做一遍。

#### 3.5.6.4 Claude 能从这堆文件里看出什么

按"先看哪个、看到什么得出什么结论"列：

**第一眼看 `compare.png` 三联图（或 pred.png + target_vae_recon.png 并排）**：

| 现象 | 结论 |
|---|---|
| pred 整体一片**糊**、几乎看不出场景结构 | DiT 没学好；ckpt 太早 / lr 不够 / EMA 没生效。结合 metrics.json 的 `latent_cos` 看，< 0.3 基本确认 |
| pred 结构对、但**细节糊** + `target_vae_recon` 也糊 | **VAE 瓶颈**，不是 DiT 的锅。pred 不会比 target_vae_recon 清，得回去看 VAE 重训或换更高 spatial resolution |
| pred 结构对、**比 target_vae_recon 还糊**很多 | DiT 收敛不够，但已经在干活；继续训 / 拿更晚的 ckpt |
| pred 整体**颜色偏冷 / 偏暖**、与 target_raw 系统性色差 | latent 标准化 mean/std 漂移；查 `latent_stats.json` 是否用对了 ckpt 对应的统计 |
| pred **黑屏 / 全灰 / 全噪声** | 推理起点 `z0_prior_alpha/sigma` 与训练不一致；或 CFG `--cfg-scale` 偏离训练 drop（默认 0.1 对应推理 2.0） |
| pred **左右翻转 / 上下颠倒** | history 帧顺序错（应该 oldest → newest，`00 → 03`），看 meta.json 里 image 路径 |
| pred 有规则**网格 / 方块**伪影 | VAE 解码器问题（多半在 patchify / decoder upsample 模块） |
| pred **几乎复制 target_raw 上一帧**、没"未来感" | DiT 直接学了"复制 z_current"，flow matching 信号被忽略；查 `--z0-prior-alpha` 是不是太大 |

**第二眼看 `metrics.json`**：
- `latent_mse_mean` > 0.5：DiT 还远没收敛；ckpt 太早
- `latent_cos_mean` < 0.3：方向都不对，重训
- `latent_cos_mean` > 0.7 但 `pixel_l1` 仍偏大：latent 对了、VAE 解码丢了细节，多半是 VAE 瓶颈
- `psnr_mean` 单看绝对值无意义（参考 `target_vae_recon` 的天花板），重要的是 ckpt 之间的 delta
- `velocity_cos` < 0.7：rectified flow 训练信号弱，确认 t 采样是不是 logit-normal、CFG drop 是不是 0.1

**第三眼看 `euler_trace.json`（probe 才有）**：
- `v_cos_vs_gt_direction` 整段接近 1（如 [0.92, 0.94, 0.93, ...]）→ 健康
- `v_cos` 一头一尾接近 1、**中段掉到 0.3** → DiT 中间时间步 t=0.5 附近没训透
- `v_cos` 整段都低（< 0.5）→ DiT 整体没学
- `z_l2_to_gt` **单调下降**（如 [4.2, 3.1, 2.4, 1.8, ...]）→ 健康
- `z_l2_to_gt` 中段反弹（先下降再上升）→ Euler 步长太大 / 训练 t 分布与推理不匹配，考虑加 `--euler-steps`
- `z_l2_to_gt` 一直平 → DiT 输出几乎是 0，可能 CFG 缩放过大把信号抵消了

**第四眼看 `input_history/` + `memory.json` + `meta.json`**：
- 4 张 jpg 顺序 `00 → 03` 是 oldest → newest（最后一张是 anchor 帧）
- `memory.json` 里的 `status` 应该是 anchor 当前 GT，`subgoal` 是下一段事件
- `meta.json` 的 `qwen_adapter` 应为空字符串（GoalGen 阶段不挂 LoRA）；非空但你**没准备挂 LoRA** → eval 路径混进了 SFT 的 adapter，结果不可信
- `meta.json` 的 `euler_steps` / `cfg_scale` / `z0_prior_*` 必须与训练 args 一致；不一致时 pred 失真不是模型问题

#### 3.5.6.5 反过来：AI 看不出什么、必须靠 metrics 全集

probe / 小批量 eval 是 case-level 视角，**不报聚合指标**。下面这些必须靠
`eval_v1.py` 不带 `--max-samples` 的全集跑：

- `overall.latent_mse_mean / latent_cos_mean / psnr_mean` 真值
- `by_scenario` 拆分（哪些场景特别差）
- TB 上多 ckpt 横向曲线
- ckpt 选优（哪一步 `latent_cos` 拐点）

所以推荐顺序是：先跑 §3.5.6.1 拿到 30 个 case 发给 AI 做粗筛（判定方向 / 找问题类型）→ 如果方向对，再跑全集 `eval_v1.py` 拿数字 + TB 曲线挑 ckpt。

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
- 已上 EMA / CFG / latent stats caching / 图像解码评测；仍未做完整 KV 与 latent 离线数据集缓存。

完整边界见 `GOALGEN_V1_PLAN.md` 的 "v1 / v2 边界" 一节。
