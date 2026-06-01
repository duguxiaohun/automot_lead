# SFT v1 运行教程 — 从生成数据到拿到评估指标

> 本文档是 [SFT_V1_PLAN.md](SFT_V1_PLAN.md) 的"操作手册"对照：PLAN 讲设计与
> 决策依据，本 RUN 讲实际怎么跑。
>
> **关键约定**：所有命令默认 **从 `AutoMoT/` 目录执行**（远程默认 cwd），
> 不是从仓库根 `automot_lead/`。所以脚本路径写 `tools/...` 不是 `AutoMoT/tools/...`，
> checkpoint 路径写 `checkpoints/...` 不是 `AutoMoT/checkpoints/...`。
> 唯一例外：远程环境的 `keyframes_all_scenarios.json` 固定放在
> `/datashare/IOL4SGH/data/data/keyframes_all_scenarios.json`，下面示例统一使用这个绝对路径。

---

## 0. 准备：远程同步代码 + 确认模型权重

```bash
cd ~/automot_lead          # 仓库根
git pull                   # 拉到最新
cd AutoMoT                 # 进入 AutoMoT/ 作为后续所有命令的 cwd

# 确认 base 模型已下载
ls checkpoints/Qwen3-VL-4B-Instruct/ | head -5
# 期望：config.json / tokenizer.json / *.safetensors / ...
```

如果模型不在 `checkpoints/Qwen3-VL-4B-Instruct/`，后续命令都可以前缀
`MODEL_DIR=/真实绝对路径` 临时 override，例如：

```bash
MODEL_DIR=/datashare/IOL4SGH/AutoMoT/models/Qwen3-VL-4B-Instruct \
  bash tools/sft_v1_train.sh ddp
```

---

## 1. 生成 SFT 数据集（CPU，约 1–3 分钟）

```bash
python tools/build_sft_dataset_v1.py \
    --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
    --data-root /datashare/IOL4SGH/data/data \
    --samples-per-scenario 800 \
    --output-dir checkpoints/sft_v1_data
```

> 默认值已经是 `--samples-per-scenario 800 --advance-ratio 0.35`（脚本里写死，无需手动传）。
> 前一轮 v1 训练用 200/0.25 + lr 1e-4 + epoch 3 跑出了 ckpt-8100 严重过训
> （STATUS 答对但模型陷入复读、EOS 被刷崩）。当前默认配合 sft_v1_train.sh 的 epoch=2 / lr=5e-5
> 重新校准训练 step 数。

**预期输出**（节选）：

```
[load] 7326 total runs in keyframes
[filter] kept 7326 runs; skipped by status: {}
[stratify] Accident         keep= 686 adv= 20 -> chosen=706 (adv=20)
[stratify] AccidentTwoWays  keep=1024 adv= 20 -> chosen=800 (adv=20)
...
[split] train=~14400  val=~1600
[write] checkpoints/sft_v1_data/train.jsonl
[write] checkpoints/sft_v1_data/val.jsonl
[write] checkpoints/sft_v1_data/stats.json
```

**通过条件**：

- `train.jsonl` + `val.jsonl` + `stats.json` 三个文件都生成；
- `stats.json` 里 `transition_in_train` ≈ 总数的 30% 左右（推进类天然稀少，达不到 35% 目标时会自动收所有可得的；不会复制样本）；
- 单条样本里 `messages[1].content` 含 4 个 `<image>` 占位符；
- `images` 列表里 4 个 RGB 路径都指向 `/datashare/IOL4SGH/data/data/<scenario>/<run_id>/rgb/*.jpg`。

**边界采样口径**：

- 保持类只丢弃转换帧前的 buffer 帧，默认 `buffer=2`，也就是 `f_t-2` / `f_t-1`；
- 转换帧 `f_t` 起已经属于新 STATUS，若 `prev_anchor=anchor-K` 仍在旧 STATUS，则作为推进类保留；
- 例如 `f_t=37`、`K=4` 时，`anchor=35/36` 的 keep 被丢弃，`anchor=37/38/39/40` 都是 advance。

**常见报错**：

| 现象 | 原因 | 处理 |
|---|---|---|
| `keyframes_all_scenarios.json` 找不到 | 路径写错（远程不在仓库根） | 用 `--keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json` |
| 某些 scenario 提示样本不足 800 | 该场景 `Completed/Perfect` run 太少 | 不影响，会自动按现有量取；看 `stats.json` 里 `chosen_total` 哪些场景 < 800 |
| `images` 路径全是 `0000.jpg / 0001.jpg / ...` 字面值 | `--data-root` 在本机不可访问 | 远程跑时 data-root 必须可见，不然 fallback 会退到字面路径，训练时找不到图 |

---

## 2. 静态 sanity：token 级 mask 是否对（CPU，<10 秒）

```bash
python tools/check_loss_mask.py
```

**预期输出**：

```
[load] jsonl=checkpoints/sft_v1_data/train.jsonl sample_idx=0
[load] scenario=Accident run_id=... anchor=...

===== assistant text =====
ANALYSIS: Observations recorded.
STATUS: hazard_detect
SUBGOAL: max_brake_or_min_gap
==========================

[loss-range] STATUS  chars [41,54)  -> 'hazard_detect'
[loss-range] SUBGOAL chars [64,82)  -> 'max_brake_or_min_gap'

 idx tag        id     char_range  decoded
--------------------------------------------------------------------------------
   0 [MASK]  19394          [0,9)  'ANALYSIS:'
   1 [MASK]    220         [9,10)  ' '
   2 [MASK]   4571        [10,13)  'Obs'
   ...
   7 [MASK]    198        [32,33)  '\n'
   8 [MASK]  31650        [33,40)  'STATUS:'
   9 [MASK]    220        [40,41)  ' '
  10 [LOSS]  ...           [41,...) 'hazard'
  11 [LOSS]  ...                   '_detect'
  12 [MASK]    198                 '\n'
  13 [MASK]  ...                   'SUBGOAL:'
  ...
  16 [LOSS]  ...                   'max'
  17 [LOSS]  ...                   '_brake'
  ...

[summary] total tokens = ~20, mask = ~14, loss = 4~6
```

**通过条件**（**必须全部满足**）：

- ANALYSIS 占位段每个 token 都打 `[MASK]`；
- `STATUS:` 与 `SUBGOAL:` 这两个**字面关键词**的 token 也打 `[MASK]`；
- 只有 STATUS 行和 SUBGOAL 行的 **事件名 token** 打 `[LOSS]`；
- `summary` 里 `2 ≤ n_loss ≤ 10`、`n_mask ≥ 5`；
- plugin sanity 里两个 `literal=STATUS:/SUBGOAL: in_loss=False, in_mask=True`；
- **无 `[WARN]` 行输出**。

**异常处理**：看到 `[WARN]` 必须先修再继续，常见原因：

| WARN | 原因 | 处理 |
|---|---|---|
| `FULL_PATTERN 匹配不到` | `PLACEHOLDER_ANALYSIS` 与 `FULL_PATTERN` 漂移；或 jsonl 三段格式被破坏 | 检查 `build_sft_dataset_v1.py` 的 `assistant_content` 模板、确认仍是 `ANALYSIS:\nSTATUS:\nSUBGOAL:` 三段 |
| `算 loss 的 token 太少（<2）` | regex 把事件名吃进了 mask 段，或 jsonl 里 STATUS / SUBGOAL 行缺事件名 | 看 token 表里 status_start / subgoal_start 落点；常见是事件名前多了空格 / 制表符让 regex 错切 |
| `算 loss 的 token 太多（>10）` | regex 把 `STATUS:` / `SUBGOAL:` 字面也算进 loss | 比对 `sft_v1_loss_scale_plugin.py::_FULL_PATTERN` 与 `check_loss_mask.py::FULL_PATTERN` 是否一致 |
| `字面 STATUS:/SUBGOAL: in_loss=True` | 插件回退到 `_ANALYSIS_ONLY_REGEX` fallback 路径（context 被 swift 切碎传入） | 升级 ms-swift 或回退到上一个验证过的 swift 版本；fallback 仅 mask ANALYSIS、字面会进 loss |

---

## 3. 动态 sanity：跑 2 step 看真实 loss 数值（**需要 GPU**，约 1–2 分钟）

```bash
bash tools/sft_v1_train.sh check
```

这个命令会通过 `--external_plugins tools/sft_v1_loss_scale_plugin.py` 注册
`sft_v1_analysis_mask`，再用 `--loss_scale sft_v1_analysis_mask` 把 ANALYSIS 占位 +
`STATUS:` / `SUBGOAL:` 字面 + 段间空白全部 mask=0，只让两段事件名 token 算 loss。
ms-swift 3.12.x 不接受 JSON regex 形式的 `--loss_scale`。`check` 模式默认用
`nvidia-smi` 自动选择当前最空闲的一张 GPU，并且不传 `--val_dataset`，所以只跑 2
个训练 step，不会加载/评估 val 集的约 800 条样本。

建议先过完 §2 静态 sanity，再跑这里。

**预期 loss 数值**（健康范围）：

```
{'loss': 0.3~6, 'grad_norm': ..., 'learning_rate': ..., 'epoch': 0.0x}
```

> v1 mask 升级后，每条样本只剩 2–6 个事件名 token 算 loss。基模对常见事件名（如
> `initial`、`hazard_detect`）预测难度本来就低，所以早期 loss 比上一版（含字面 token）
> 系统性偏低 1–2 个量级是正常现象。重点不是 loss 绝对值，而是 plugin sanity 通过 +
> 训练曲线收敛 + eval 指标。

与 v2 一样，重点看“mask 是否生效 + loss 是否有限非 NaN”，不要只盯绝对值。

**判读规则**：

| 现象 | 判读 | 处理 |
|---|---|---|
| `python tools/check_loss_mask.py` 的 plugin sanity 显示两段 `event_name in_loss=True, in_mask=False`，且字面 `STATUS:/SUBGOAL: in_loss=False, in_mask=True`，并且 `check` loss 有限非 NaN | ✅ 训练侧 mask 大方向正常 | 可进 step 4 |
| `loss < 1` 但 plugin sanity 通过 | ⚠️ 事件名 token 数少 + 基模对短词预测容易，正常现象 | 继续看正式训练/评估指标 |
| `loss < 0.01` 或 `grad_norm=0` | ❌ 可能两段 `event_name` 也被 mask 了（全 0 权重） | 先查 `check_loss_mask.py` 的 `n_loss` 是否 ≥ 2 |
| `loss > 8` | ⚠️ 可能 `STATUS:` / `SUBGOAL:` 字面或 ANALYSIS 占位误入 loss；也可能 swift fallback | 先查 plugin sanity 是否走主路径 `_FULL_PATTERN`；若回退到 fallback，按 PLAN §11 回退：写 `tools/sft_v1_preprocessor.py` 手动 mask labels |
| check 结束保存了 `checkpoint-2` | ❌ check 模式不该保存 checkpoint | 拉最新脚本，确认含 `--save_strategy no` |

**常见启动报错**：

| 现象 | 原因 | 处理 |
|---|---|---|
| `swift: command not found` | 当前环境没装 ms-swift 或 PATH 不对 | 先确认 `which python && which swift && pip show ms-swift` |
| `KeyError: 'sft_v1_analysis_mask'` | 插件没被加载，loss_scale 策略未注册 | 确认从 `AutoMoT/` 目录运行；检查 `tools/sft_v1_loss_scale_plugin.py` 是否存在 |
| `KeyError: '{"ANALYSIS...": 0.0}'` | 仍在用旧版 JSON regex 命令 | 拉最新脚本，确认 `sft_v1_train.sh` 里有 `--external_plugins` |
| `invalid device ordinal` / CUDA 选错卡 | 远程调度只分配了部分卡，或 `CUDA_VISIBLE_DEVICES` 与实际可见卡不一致 | 不手动指定时脚本会自动挑空闲卡；若调度系统已分配卡，显式使用它给出的 `CUDA_VISIBLE_DEVICES` |

---

## 4. 正式训练（**8×H20 DDP，约 2 小时**）

```bash
bash tools/sft_v1_train.sh ddp
```

默认会用 `nvidia-smi` 自动挑最空闲的 8 张 GPU。想指定卡数但仍自动挑最闲的
N 张，用 `DDP_GPU_COUNT`：

```bash
# 自动挑最空闲的 4 张 GPU
DDP_GPU_COUNT=4 bash tools/sft_v1_train.sh ddp

# 自动挑最空闲的 2 张 GPU
DDP_GPU_COUNT=2 bash tools/sft_v1_train.sh ddp
```

注意：DDP 模式会默认让 `NPROC_PER_NODE` 跟随最终的 `CUDA_VISIBLE_DEVICES` 数量。
`DDP_GPU_COUNT` 显式传入时会覆盖外层残留的 `CUDA_VISIBLE_DEVICES`，避免远程环境里已有
`CUDA_VISIBLE_DEVICES=0` 或 `NPROC_PER_NODE=1` 导致实际只起单卡。
如果你确实要严格沿用外部已经设置好的 `CUDA_VISIBLE_DEVICES`，加：

```bash
SFT_RESPECT_CUDA_VISIBLE_DEVICES=1 DDP_GPU_COUNT=4 bash tools/sft_v1_train.sh ddp
```

如果你确实要严格沿用外部已经设置好的 `NPROC_PER_NODE`，加：

```bash
SFT_RESPECT_NPROC_PER_NODE=1 bash tools/sft_v1_train.sh ddp
```

DDP rendezvous 端口也会自动处理：脚本默认设置 `MASTER_ADDR=127.0.0.1`，如果
`MASTER_PORT` 没有设置，或已设置但端口被占用，会自动选择一个空闲端口并在启动日志里打印。
如果你确实要严格沿用外部已经设置好的 `MASTER_PORT`，加：

```bash
SFT_RESPECT_MASTER_PORT=1 MASTER_PORT=29501 bash tools/sft_v1_train.sh ddp
```

如果你已经知道要用哪几张卡，直接显式指定 `CUDA_VISIBLE_DEVICES`，不要同时传
`DDP_GPU_COUNT`：

```bash
CUDA_VISIBLE_DEVICES=2,5,6,7 bash tools/sft_v1_train.sh ddp
```

**预期**：

- 8 卡总 step ≈ 900（按 ~14400 train 样本 / 等效 bs 32 × 2 epoch）；如果改成 4 卡，等效 bs 约 16，step 约翻倍；
- 每 100 step 保存一次 LoRA adapter 到 `checkpoints/sft_v1_lora/checkpoint-XXX/`；
- 训练 loss 大致从 check 阶段量级继续下降，最终以 eval 指标为准。

**中途检查 + 选 ckpt 策略**（吸取 ckpt-8100 教训）：

- swift 会把日志写到 `checkpoints/sft_v1_lora/v*/logging.jsonl`；
- 每 `save_steps=100` 会保存一次 LoRA adapter checkpoint，最多保留 5 个（`save_total_limit=5`）。
- **不要无脑用最后一个 ckpt**。训练完后对每个保留的 ckpt 跑一次 eval 抽样，
  挑 `early_advance_rate ↓` + `advance_accuracy ↑` 拐点的那个：

  ```bash
  for s in 200 400 600 800 900; do
      python tools/eval_sft_v1.py \
          --lora-dir checkpoints/sft_v1_lora/v*/checkpoint-${s} \
          --save-root checkpoints/sft_v1_lora/v*/checkpoint-${s} \
          --max-samples 200 --no-full-dump
  done
  ```

  对照各 `eval/metrics.json`，曲线"训过头"的典型征兆是 `advance_accuracy` 涨到峰值后回落、
  `early_advance_rate` 抬头，同时 `predictions_diff.jsonl` 里出现 `raw_text` 含
  重复 `STATUS:` 行 → 选回退到峰值那个 ckpt，不要用更晚的。

**单卡退回**（如果 8 卡 NCCL 出问题）：

```bash
bash tools/sft_v1_train.sh single
```

单卡约 8–10 小时。

**显存观察**（H20 96GB，bf16 LoRA r=16）：

| 阶段 | 单卡占用 |
|---|---|
| 模型 + LoRA 加载完 | ~10 GB |
| forward + activation | ~25 GB |
| backward + adam state | ~32 GB |
| **稳态峰值** | **~30-35 GB** |

如果超过 80 GB，先把 `per_device_train_batch_size` 从 2 降到 1，再排查是否
`--gradient_checkpointing` 没生效。

### 4.1 TensorBoard（推荐用 tools/tb_serve.sh 一条命令搞定）

训练 + eval 现在统一往 `OUTPUT_DIR` 下平铺保存：

```
checkpoints/sft_v1_lora/
├─ checkpoint-*/      LoRA adapter
├─ tb/                训练 TB events（swift 写，--report_to tensorboard）
├─ eval/              eval_sft_v1.py 写的 metrics.json + predictions.jsonl
├─ eval_tb/<ckpt>/    eval_sft_v1.py 写的指标 scalar + text（每个 ckpt 一个 TB run）
└─ eval_cases/        probe_sft_v1.py 随机场景 case dump
```

TensorBoard 启动 `--logdir checkpoints/sft_v1_lora` 时，左侧 run 列表会同时列出
`tb`（训练）和 `eval_tb/<ckpt-tag>`（每次 eval 一个 run）。

推荐用脚本一条命令搞定（自动选空闲端口 + bind_all）：

```bash
# 远端：起 TB（在 AutoMoT/ 目录下）
bash tools/tb_serve.sh checkpoints/sft_v1_lora
```

`tb_serve.sh` 会打印类似：

```
[tb] TensorBoard 已启动 → 在本地浏览器直接打开：
      http://localhost:41273
```

VSCode Remote 会自动把端口转发到本地，直接浏览器打开 URL 即可。Ctrl-C 关掉
tb_serve.sh 时 TB 服务一起退。固定端口用 `TB_PORT=6007 bash tools/tb_serve.sh ...`。

常见 tag：

| Tag | 含义 |
|---|---|
| `train/loss` | per-step LoRA loss（含 loss_scale plugin 的权重） |
| `train/learning_rate` | cosine 调度后的 lr |
| `train/grad_norm` | swift 内部已记录 |
| `eval/loss` | val 子集上的 loss（swift 训练里 EVAL_STEPS 触发） |

**注意**：`eval_sft_v1.py` 现在**默认不写 TB**（步骤一 TB 入口已让位给步骤二 GoalGen
那侧；用户反馈本侧 TB 容易加载失败）。需要时显式加 `--tb`，eval scalar 才会落到
`OUTPUT_DIR/eval_tb/<run_tag>/`。

`check` 模式把 `EVAL_STEPS=999999` 关掉 eval，所以 check 跑只有 train 曲线；
若想 check 时也看 eval 曲线，临时改 `EVAL_STEPS=1` 加 `VAL_ARGS=(--val_dataset ...)`。

---

## 5. 评估（约 10–30 分钟，取决于 val 大小）

`--save-root` **必填**，所有产物落到 `<save_root>/eval/` 与 `<save_root>/eval_tb/<run_tag>/`。
推荐 `--save-root` = 训练 OUTPUT_DIR（即 `checkpoints/sft_v1_lora`），与训练 ckpt
平铺在同一根下。

### 5.0 两个入口

| 脚本 | 跑全集出聚合指标 | 每条样本完整 dump | 额外提供 |
|---|---|---|---|
| `tools/eval_sft_v1.py` | ✅ | ✅（默认 `--max-samples > 0` 时开） | predictions.jsonl + metrics.json + 可选 TB |
| `tools/probe_sft_v1.py` | ❌（按 scenario 抽样） | ✅ | per-token NLL + 训练 loss_scale 同口径 mask |

**简单决策**：先跑 `eval_sft_v1.py --max-samples 100` 就能拿到 100 个 case 完整
dump + 全部指标；只有需要看"具体哪些 token 拉高了 loss"时再跑 probe。

### 5.1 eval_sft_v1.py

```bash
# 推荐：先跑 base 小样本 + 完整 dump；默认不会导入 peft / LoRA
python tools/eval_sft_v1.py \
    --save-root checkpoints/sft_v1_lora \
    --max-samples 100

# 跑全集只出聚合指标（不 dump，磁盘友好）；默认同样是 base
python tools/eval_sft_v1.py \
    --save-root checkpoints/sft_v1_lora

# 多卡分片（适合全集）
torchrun --standalone --nproc_per_node=4 tools/eval_sft_v1.py \
    --save-root checkpoints/sft_v1_lora

# 只有明确要评估 LoRA adapter 时，才传 --lora-dir
python tools/eval_sft_v1.py \
    --lora-dir checkpoints/sft_v1_lora/checkpoint-900 \
    --save-root checkpoints/sft_v1_lora/checkpoint-900 \
    --max-samples 100
```

eval 默认会在加载模型前调用 `nvidia-smi`，按 `memory.used`、`utilization.gpu`
从小到大自动选择空闲 GPU：单进程挑 1 张，`torchrun --nproc_per_node=N` 时挑 N 张。
进程内仍使用 `cuda:0` / `--device auto`。如果外部已经设置 `CUDA_VISIBLE_DEVICES`
或显式传 `--device cuda:N`，脚本会尊重外部设置。要关闭自动选卡：
`SFT_EVAL_DISABLE_AUTO_GPU=1 python tools/eval_sft_v1.py ...`。

**关键参数**：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--save-root` | （必填） | 产物根目录，通常等于训练 OUTPUT_DIR |
| `--run-tag` | 自动派生 | TB run 子目录名（`base` / `ckpt300` / `lora`） |
| `--max-samples` | 0 = 全集 | 截断 val 样本数 |
| `--full-dump` / `--no-full-dump` | 自动 | 默认 `--max-samples > 0` 时开；显式覆盖 |
| `--full-dump-limit N` | 0 = 不限 | dump 上限，防止误开铺满磁盘 |
| `--lora-dir` | 空字符串 | 默认跑 base 且不会导入 `peft`；只有明确评估 LoRA 时才传 adapter 目录 |
| `--device` | `auto` | 默认配合自动选空闲 GPU；显式 `cuda:N` 时关闭自动 mask |
| `--tb` / `--no-tb` | `--no-tb` | 默认不写 TB（步骤一 TB 已让位给步骤二） |
| `--skip-anchor12-sanity` | False | 跳过 anchor=12 单例检查 |

**产物布局**（每次 eval 后）：

```
checkpoints/sft_v1_lora/eval/
├─ metrics.json                聚合指标 + _metric_doc 含义说明
├─ predictions.jsonl           全部样本一行（含 raw_text / parsed / error_kind）
├─ predictions_diff.jsonl      只挑 error_kind != "ok" 的样本（人工查错入口）
└─ cases/                      完整 dump（小样本时自动开）
   └─ 00017__Accident__Town03_Rep0_route_001783__anchor12__early_advance/
      ├─ inputs/
      │  ├─ system_prompt.txt              system 原文
      │  ├─ user_prompt.txt                user 原文（去 <image> 占位）
      │  └─ image_00.jpg ... image_03.jpg  history 4 帧，**复制**到本地
      ├─ outputs/
      │  ├─ raw_text.txt                   模型 raw 输出
      │  └─ parsed.json                    pred/gt status+subgoal + status_match + subgoal_match
      ├─ step.json                         完整元信息（sample_idx / lora_dir / 图像路径）
      └─ summary.md                        一页 markdown，**顶部就是 SUBGOAL/STATUS 对比表**
```

`summary.md` 渲染后，顶部直接看到：

```
## GT vs Pred
| field    | GT (truth)      | Pred (model)        | match |
| STATUS   | `initial`       | `hazard_detect`     | ❌    |
| SUBGOAL  | `hazard_detect` | `max_brake_or_min_gap` | ❌ |
```

再往下是输入图、system/user prompt、GT 全文、Pred raw 全文——人工 review 单文件够。

**case 目录名编码**：`<sample_idx>__<scenario>__<run_id>__anchor<N>__<error_kind>`，
按 `error_kind` 排序后 `early_advance` / `none` / `other` 的 case 自然聚在一起。

### 5.2 metrics.json 字段

```json
{
  "_metric_doc": {
    "keep_accuracy": "保持类样本 STATUS == GT 的比例（越大越好）",
    "advance_accuracy": "推进类样本 STATUS == GT 的比例（越大越好）",
    "early_advance_rate": "保持类样本 STATUS == next(GT) 的比例（越小越好；核心痛点）",
    "anchor12_sanity": "anchor=12 固定 fail case 上 STATUS 是否回到 initial；passed=true 即原始 bug 已修",
    "per_scenario": "按 scenario 拆开的细分计数"
  },
  "n_total": 840,
  "n_keep": 630,
  "n_advance": 210,
  "keep_accuracy": 0.96,
  "advance_accuracy": 0.65,
  "early_advance_rate": 0.03,
  "anchor12_sanity": { "enabled": true, "passed": true, "pred_status": "initial", "expected_status": "initial" }
}
```

`_metric_doc` 是 JSON 内嵌的指标含义，省得回 doc 查。

**通过条件**（与 PLAN §8 一致）：

| 指标 | v1 目标 | 优先级 |
|---|---|---|
| `keep_accuracy` | ≥ 0.95 | 高 |
| `advance_accuracy` | ≥ 0.60 | 中 |
| **`early_advance_rate`** | **≤ 0.05** | **最高（核心痛点）** |
| **`anchor12_sanity.passed`** | **= true** | **必须** |

### 5.3 predictions.jsonl 字段

每行一条样本：

```json
{
  "sample_idx": 17, "scenario": "Accident", "run_id": "...", "anchor": 12,
  "is_transition_sample": false,
  "gt_status": "initial", "gt_subgoal": "hazard_detect",
  "pred_status": "hazard_detect", "pred_subgoal": "max_brake_or_min_gap",
  "raw_text": "ANALYSIS: ...\nSTATUS: hazard_detect\nSUBGOAL: max_brake_or_min_gap",
  "error_kind": "early_advance", "error": null
}
```

`error_kind` 分类：

| 值 | 含义 |
|---|---|
| `ok` | pred == gt |
| `early_advance` | keep 样本上 pred == next(gt)，最关心的失败模式 |
| `none` | 输出格式坏，没解析到 STATUS |
| `inference_error` | generate 阶段抛异常（OOM / 路径错） |
| `other` | 其它（跳更远状态 / 非法 token / advance 样本未对齐） |

**base vs LoRA 横向对比**：跑两次 eval，分别用 `--run-tag base` / `--run-tag lora`，
然后 diff 两份 predictions.jsonl：

```bash
python -c "
import json
def load(p): return [json.loads(l) for l in open(p)]
b = {r['sample_idx']: r['pred_status'] for r in load('checkpoints/sft_v1_lora/eval/predictions.jsonl')}
# ... 另起一个 save-root 跑 base，比较"
```

(同一个 save-root 跑两次会覆盖 predictions.jsonl；要并存就拆 `--save-root checkpoints/sft_v1_base`。)

### 5.4 probe_sft_v1.py（深度诊断）

```bash
# 默认跑 base，抽 16 条样本（4 场景 × 4 条）
python tools/probe_sft_v1.py \
    --save-root checkpoints/sft_v1_lora \
    --num-per-scenario 4 --seed 0

# 只有明确要看 LoRA adapter 时才传 --lora-dir，--case-suffix 防覆盖
python tools/probe_sft_v1.py \
    --lora-dir checkpoints/sft_v1_lora/checkpoint-900 \
    --save-root checkpoints/sft_v1_lora \
    --num-per-scenario 4 --seed 0 --case-suffix "_lora"
```

probe 的 case 目录布局（与 eval cases 类似，但多 `token_loss.json`）：

probe 不接 torchrun；未显式设置 `CUDA_VISIBLE_DEVICES` 或 `--device cuda:N` 时，
默认自动挑 1 张空闲 GPU。

```
checkpoints/sft_v1_lora/eval_cases/<scenario>__<run>__<anchor>/
├─ input_images/00.jpg ... 03.jpg
├─ system_prompt.txt / user_prompt.txt / gt.txt / pred.txt
├─ token_loss.json   ← per-token NLL + 仅两段 event_name token 标 mask=1（与训练 loss_scale 同口径）
├─ meta.json
└─ overview.md
```

`token_loss.json` 三个均值（v1 mask 升级后的语义）：
- `mean_loss_raw`：所有 token 平均（含被 mask 的字面 token）
- `mean_loss_status_subgoal_only`：训练时真正在优化的两段事件名 token 平均
- `mean_loss_masked_literals`：训练时 mask=0 的字面 token 平均（ANALYSIS 占位 + `STATUS:` / `SUBGOAL:` 字面 + 空白），参考值

什么时候用 probe 而不是 eval cases：
- 想看模型在 STATUS / SUBGOAL 具体哪个 token 上犹豫（per-token NLL 曲线）
- 怀疑 loss_scale plugin mask 边界漂移（probe 用同一份 regex 反推 token mask）

否则 eval cases 已经够人工 review。

### 5.5 训练完小批量诊断 → 打包发给 AI 审阅（推荐流程）

用途：训练刚结束时先抽 ~30 个 case 粗筛，判断模型是否在干活；方向对再跑全集
`eval_sft_v1.py` 拿正式指标。

```bash
# 推荐：base + LoRA 各跑一份 probe（同 seed 选中样本完全一致，便于并排比较）
# 1) base
python tools/probe_sft_v1.py \
    --save-root checkpoints/sft_v1_lora \
    --num-per-scenario 3 --seed 42 \
    --case-suffix "_base"

# 2) 当前 LoRA ckpt（按需选 ckpt 编号）
python tools/probe_sft_v1.py \
    --lora-dir checkpoints/sft_v1_lora/checkpoint-900 \
    --save-root checkpoints/sft_v1_lora \
    --num-per-scenario 3 --seed 42 \
    --case-suffix "_lora_ckpt900"
```

产物在 `checkpoints/sft_v1_lora/eval_cases/`：每个 case 含 4 帧输入、
`user_prompt.txt`、`gt.txt`、`pred.txt`、`token_loss.json`、`meta.json`、
`overview.md`，另有 `_index_*.jsonl`。给 AI 审阅时直接打包 `eval_cases/`。

快速判断顺序：先看 `pred.txt` vs `gt.txt` 的 STATUS/SUBGOAL 是否正确、是否复读；
再看 `token_loss.json` 里事件名 token 的 NLL；最后查输入帧顺序和 `meta.json` 的
`lora_dir`。probe 只做 case-level 粗筛，正式的 `keep_accuracy`、`advance_accuracy`、
`early_advance_rate`、per-scenario 指标仍以全集 `eval_sft_v1.py` 为准。

---

## 一行串起来（happy path，不推荐生产用）

```bash
python tools/build_sft_dataset_v1.py --data-root /datashare/IOL4SGH/data/data \
  --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
  --output-dir checkpoints/sft_v1_data && \
python tools/check_loss_mask.py && \
bash tools/sft_v1_train.sh check && \
bash tools/sft_v1_train.sh ddp && \
python tools/eval_sft_v1.py --save-root checkpoints/sft_v1_lora --max-samples 100
```

**强烈建议分步跑**，每步看输出确认再进下一步——尤其是 step 2/3 sanity，
跳过会让 step 4 烧 1.5 小时但什么都没学到。

---

## 6. 出问题时贴什么内容方便排查

| 步骤 | 贴这些 |
|---|---|
| step 1 后 | `checkpoints/sft_v1_data/stats.json` 完整内容 |
| step 2 后 | `check_loss_mask.py` 完整 stdout |
| step 3 后 | `sft_v1_train.sh check` 输出最后 30 行（含 loss 数值与 warning） |
| step 4 中 | 每 100 step 的训练 log（loss / grad_norm / lr 趋势）即可，不需要全部 |
| step 5 后 | `checkpoints/sft_v1_lora/eval/metrics.json` 全文 + 任意 1 个 `cases/.../summary.md` |

---

## 7. 与 v2 / 后续迭代的关系

- v1 完成（4 项指标全过）后，进 [SFT_V1_PLAN.md §9](SFT_V1_PLAN.md) 列出的 v2 计划：
  ANALYSIS 蒸馏（用 v1 LoRA + GT status 反向写 analysis 真值）；
- v1 没过的话，按 [SFT_V1_PLAN.md §11](SFT_V1_PLAN.md) 风险表逐条排查，
  不要直接堆 v2 上去。
