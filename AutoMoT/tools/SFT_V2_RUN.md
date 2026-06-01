# SFT v2 运行教程 — pending 数据 + 运行时 teacher 真值

> 本文档是 [SFT_V2_PLAN.md](SFT_V2_PLAN.md) 的"操作手册"对照：PLAN 讲设计与决策依据，
> 本 RUN 讲实际怎么跑。
>
> **关键约定**：所有命令默认 **从 `AutoMoT/` 目录执行**（远程默认 cwd），不是从仓库根
> `automot_lead/`。脚本路径写 `tools/...` 不是 `AutoMoT/tools/...`，checkpoint 路径写
> `checkpoints/...` 不是 `AutoMoT/checkpoints/...`。`keyframes_all_scenarios.json`
> 远程固定放在 `/datashare/IOL4SGH/data/data/keyframes_all_scenarios.json`。

---

## 0. 准备

```bash
cd ~/automot_lead
git pull
cd AutoMoT

# 确认 base 模型已下载（teacher 与训练共用）
ls checkpoints/Qwen3-VL-4B-Instruct/ | head -5
```

模型路径不在默认位置时所有命令都可前缀 `MODEL_DIR=/真实路径`。

---

## 1. 阶段 1：生成 pending jsonl（CPU，约 1–3 分钟）

```bash
python tools/build_sft_dataset_v1.py \
    --mode v2 \
    --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
    --data-root /data/lead_data/data \
    --output-dir checkpoints/sft_v2_data_pending
```

与 v1 完全相同的采样 + 拼装逻辑，唯一区别是 assistant content 里 ANALYSIS 段是占位
`__TEACHER_PENDING__`，并写 `dataset_version: "v2_pending"`。

**通过条件**：
- `train.jsonl` / `val.jsonl` / `stats.json` 三个文件生成；
- 任意行 `dataset_version == "v2_pending"`；
- 任意行 `messages[2].content` 形如 `ANALYSIS: __TEACHER_PENDING__\nSTATUS: ...\nSUBGOAL: ...`。

快速 spot check：

```bash
python -c "
import json
with open('checkpoints/sft_v2_data_pending/train.jsonl') as f:
    r = json.loads(f.readline())
print('version:', r['dataset_version'])
print(repr(r['messages'][2]['content']))
"
```

---

## 2. teacher ANALYSIS 生成策略

**默认不要长期维护写死 ANALYSIS 的训练集。**

v2 的长期数据集是 `checkpoints/sft_v2_data_pending/`：里面只保存图像、MEMORY、
STATUS/SUBGOAL 与 `__TEACHER_PENDING__` 占位。冻结 teacher 只在两种场景现场推理：

- 训练前预览：少量样本，确认 teacher 输出是否符合预期，不写训练 jsonl。
- 训练启动时：`sft_v2_train.sh` 检测到 `dataset_version == "v2_pending"` 后，自动调用
  `build_sft_dataset_v2_teacher.py` 生成临时 runtime jsonl，默认写到
  `checkpoints/sft_v2_lora/runtime_teacher_data/`。

说明：ms-swift 训练入口需要 jsonl 文件，所以这里的“实时”是训练启动时临时物化 teacher
真值，不是每个 batch 在线调用 teacher；pending 源数据不会被回写。

### 2.1 训练前 teacher 预览（不写训练集，推荐先跑）

如果 keyframes / prompt / 数据采样之后会改，先从 pending jsonl 里抽少量样本现场跑
teacher，打开网页看 ANALYSIS 是否符合预期。这个步骤只写 inspect 目录，不会把
teacher ANALYSIS 回写到训练 jsonl。

如果只想生成一个很小的预览数据集，先跑：

```bash
python tools/build_sft_dataset_v1.py --mode v2 --dry-run \
    --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
    --data-root /data/lead_data/data \
    --output-dir checkpoints/sft_v2_preview_pending
```

然后把下面命令里的 `checkpoints/sft_v2_data_pending/train.jsonl` 换成
`checkpoints/sft_v2_preview_pending/train.jsonl`。

```bash
python tools/inspect_teacher_outputs.py \
    --jsonl checkpoints/sft_v2_data_pending/train.jsonl \
    --save-root checkpoints/sft_v2_teacher_preview_live \
    --num-per-scenario 1 --seed 42 \
    --live --serve --port 0 \
    --model-dir checkpoints/Qwen3-VL-4B-Instruct
```

脚本会自动挑 1 张空闲 GPU，并打印类似 `http://127.0.0.1:<PORT>/index.html` 的地址。
VSCode Remote / SSH 端口转发后，在浏览器打开这个地址检查：

- 输入 4 帧图像是否对；
- `teacher_user.txt` 里的 PRIVILEGED 块是否符合当前样本；
- `teacher_analysis_live.txt` 是否按“看图 -> 变化 -> 结论”写，且没有泄漏 PRIVILEGED 字样。

### 2.2 可选：手动临时物化 teacher jsonl

通常不需要手动跑这一步，正式训练会自动做。只有你想提前做 plugin 静态 sanity、
复用同一份 teacher 输出，或排查 fallback 比例时才需要：

```bash
python tools/build_sft_dataset_v2_teacher.py \
    --pending-dir checkpoints/sft_v2_data_pending \
    --output-dir checkpoints/sft_v2_runtime_debug \
    --max-samples 32
```

teacher 脚本默认用 `nvidia-smi` 自动挑空闲 GPU：单进程挑 1 张，
`torchrun --nproc_per_node=N` 时挑 N 张。已有 `CUDA_VISIBLE_DEVICES` 时尊重外部设置；
要关闭自动选卡，设 `SFT_TEACHER_DISABLE_AUTO_GPU=1`。

### 2.3 通过条件

- `<output-dir>/train.jsonl` 行数 == `<pending-dir>/train.jsonl` 行数；val 同理。
- 任意行 `dataset_version == "v2"`、`teacher_meta.model_dir` 不为空。
- `teacher_meta.fallback == true` 的比例 < 5%（PLAN §8）。

快速统计：

```bash
python -c "
import json
n_total = n_fb = 0
with open('checkpoints/sft_v2_runtime_debug/train.jsonl') as f:
    for line in f:
        r = json.loads(line)
        n_total += 1
        if r.get('teacher_meta', {}).get('fallback'): n_fb += 1
print(f'total={n_total} fallback={n_fb} ratio={n_fb/max(n_total,1):.2%}')
"
```

### 2.4 抽检 teacher 输出质量

推荐改用专用可视化脚本（见 §2.1），这里保留最小抽检：

```bash
python -c "
import json, random
random.seed(0)
with open('checkpoints/sft_v2_runtime_debug/train.jsonl') as f:
    rows = [json.loads(l) for l in f]
for r in random.sample(rows, 5):
    print('---', r['scenario'], r['run_id'], 'anchor=', r['anchor'])
    print(r['messages'][2]['content'])
    print()
"
```

人工 review 判读：
- ✅ ANALYSIS 单行、2–4 句、按“看图 -> 变化 -> 结论”三步式
- ❌ 出现 PRIVILEGED 字眼或直接泄漏 GT（例如 `the current STATUS is X`）
- ❌ 大量样本同一句套话（teacher 表达塌缩）

### 2.5 teacher 可视化抽检（推荐）

新增工具：`tools/inspect_teacher_outputs.py`

- 默认模式 A（只读 jsonl，不重跑 teacher）：快、稳定、可一次抽几十条
- `--live` 模式 B：现场重跑 teacher，多产出 `teacher_raw.txt` 和 `teacher_postprocess.json`
- 采样默认按 scenario 均匀抽样（每场景 `--num-per-scenario` 条）
- 加 `--serve --port 0` 会自动选空闲端口并启动 `index.html` 预览服务

```bash
# 模式 A：只读已物化 v2 jsonl，按场景均匀抽样
python tools/inspect_teacher_outputs.py \
    --jsonl checkpoints/sft_v2_runtime_debug/train.jsonl \
    --save-root checkpoints/sft_v2_teacher_inspect \
    --num-per-scenario 3 --seed 42

# 模式 B：从 pending 现场重跑 teacher（推荐用于改 prompt / 数据后复核）
python tools/inspect_teacher_outputs.py \
    --jsonl checkpoints/sft_v2_data_pending/train.jsonl \
    --save-root checkpoints/sft_v2_teacher_inspect_live \
    --num-per-scenario 3 --seed 42 \
    --live --serve --port 0 \
    --model-dir checkpoints/Qwen3-VL-4B-Instruct
```

每个 case 目录会生成：

```
checkpoints/sft_v2_teacher_inspect/cases/<sample_idx>__<scenario>__<run_id>__anchor<N>/
├─ input_images/00.jpg ... 03.jpg
├─ teacher_user.txt
├─ teacher_analysis.txt
├─ student_assistant.txt
├─ meta.json
└─ overview.md
```

`--live` 额外多两个文件：
- `teacher_raw.txt`
- `teacher_postprocess.json`

重点看 `overview.md` 顶部三项：
- `target_status` 与 `transition` 是否和样本语义一致
- `teacher_analysis` 是否符合“看图 -> 变化 -> 结论”
- `teacher_fallback_flag` 是否大量为 true（若高于 5%，先修 teacher 再训）

---

## 3. 静态 sanity：v2 plugin mask 是否对（CPU，< 30 秒）

v2 现在有专用静态 sanity 脚本：`tools/check_loss_mask_v2.py`。

它会做两层检查：
- token 级可视化：把 assistant 每个 token 标成 `[W0.0] / [W0.3] / [W1.0]`
- plugin 主路径 sanity：直接调用 `sft_v2_loss_scale_plugin.py`，验证切片和权重

```bash
python tools/check_loss_mask_v2.py --jsonl checkpoints/sft_v2_runtime_debug/train.jsonl

# 看第 N 条样本
python tools/check_loss_mask_v2.py --jsonl checkpoints/sft_v2_runtime_debug/train.jsonl --sample-idx 7

# 训练跑过后，也可以检查 runtime teacher 数据
python tools/check_loss_mask_v2.py --jsonl checkpoints/sft_v2_lora/runtime_teacher_data/val.jsonl --sample-idx 3
```

**通过条件**（必须全部满足）：

- token 表里 ANALYSIS 正文 token 为 `[W0.3]`
- token 表里 STATUS/SUBGOAL 事件名 token 为 `[W1.0]`
- `ANALYSIS:` / `STATUS:` / `SUBGOAL:` 字面 token 都是 `[W0.0]`
- plugin 输出中，`ANALYSIS body in_loss=True`、两段 `event_name in_loss=True`
- plugin 输出中，`literal='ANALYSIS:'/'STATUS:'/'SUBGOAL:' in_loss=False`
- plugin 分段数量为 6 或 7（有 tail 时为 7）

**预期切片形状**（示例）：

```
w=0.00: 'ANALYSIS: '
w=0.30: 'I see a foggy tunnel ... advancing to hazard_detect.'
w=0.00: '\nSTATUS: '
w=1.00: 'hazard_detect'
w=0.00: '\nSUBGOAL: '
w=1.00: 'max_brake_or_min_gap'
```

---

## 4. 动态 sanity：跑 2 step 看真实 loss（**GPU**，约 1–2 分钟）

```bash
bash tools/sft_v2_train.sh check
```

与 v1 一样，2 step、不保存 ckpt、不跑 val。check 模式会自动从 pending 数据生成最多
32 条 runtime teacher 样本；不会改写 pending 数据集。

**预期 loss 数值**（健康范围）：

```
{'loss': 3~8, 'grad_norm': ..., 'learning_rate': ..., 'epoch': 0.0x}
```

说明：
- v2 会监督 ANALYSIS 正文（权重 0.3），所以 loss 统计口径与 v1 不同
- 重点看“mask 是否生效 + loss 是否有限非 NaN”，不是盯绝对值

判读：

| 现象 | 判读 | 处理 |
|---|---|---|
| `check_loss_mask_v2.py` 通过 + check loss 有限且非 NaN | ✅ 训练侧权重大方向正常 | 进 §5 正式训练 |
| loss < 1 或 `grad_norm=0` | ❌ 可能事件名 token 也被 mask 掉了（全 0 权重） | 先看 §3 的 `w1` token 数是否 ≥ 2 |
| loss > 12 | ⚠️ 可能字面 token 误入 loss，或 teacher ANALYSIS 异常长 | 先看 §3 plugin literal 检查；再查 teacher 可视化 §2.6 |
| check 模式仍保存了 checkpoint | ❌ check 不该落盘 | 拉最新 `tools/sft_v2_train.sh`，确认 `--save_strategy no` |

---

## 5. 正式训练（**8×H20 DDP**）

```bash
bash tools/sft_v2_train.sh ddp
```

GPU / 端口 / DDP rendezvous 行为与 v1 完全一致（自动选最空闲卡、自动找空闲 MASTER_PORT、
NCCL_P2P_LEVEL=NVL 等）。所有 v1 的 `DDP_GPU_COUNT` / `SFT_RESPECT_*` 环境变量在 v2 同名。

默认 `ddp` 按 8 卡跑。如果机器只想用 N 张卡，不要手动写死卡号，直接用
`DDP_GPU_COUNT=N`，脚本会自动挑 N 张最空闲 GPU，并让 teacher runtime 物化和后面的
ms-swift 训练都使用同一组卡：

```bash
# 自动挑最空闲的 4 张 GPU
DDP_GPU_COUNT=4 bash tools/sft_v2_train.sh ddp

# 自动挑最空闲的 2 张 GPU
DDP_GPU_COUNT=2 bash tools/sft_v2_train.sh ddp
```

注意：`DDP_GPU_COUNT` 显式传入时会覆盖外层残留的 `CUDA_VISIBLE_DEVICES`，避免远程
环境里已有单卡 mask 导致实际只起 1 张卡。如果调度系统已经分配好卡、你要严格沿用外部
mask，再加：

```bash
SFT_RESPECT_CUDA_VISIBLE_DEVICES=1 DDP_GPU_COUNT=4 bash tools/sft_v2_train.sh ddp
```

如果已经明确知道要用哪几张卡，才手动设置 `CUDA_VISIBLE_DEVICES`，并且不要同时传
`DDP_GPU_COUNT`：

```bash
CUDA_VISIBLE_DEVICES=2,5,6,7 bash tools/sft_v2_train.sh ddp
```

正式训练第一步会先把 `checkpoints/sft_v2_data_pending/` 临时物化到
`checkpoints/sft_v2_lora/runtime_teacher_data/`，再把这份 runtime jsonl 交给 ms-swift。
默认 `RUNTIME_TEACHER_REFRESH=1`，每次训练启动都会刷新这个 runtime 缓存，避免 keyframes /
prompt 改了以后复用旧 teacher 文本。如果上次 teacher 物化中断、想续跑同一份 pending，可显式：

```bash
RUNTIME_TEACHER_REFRESH=0 bash tools/sft_v2_train.sh ddp
```

**预期**：
- teacher runtime 物化：8 卡约 100 分钟，单卡小样本 check 约 1–2 分钟；
- LoRA 训练：8 卡总 step ≈ 900（与 v1 同）；4 卡 ≈ 1800；
- 每个 epoch 末尾保存一次 LoRA adapter 到 `checkpoints/sft_v2_lora/v*/checkpoint-XXX/`，
  保留最近 3 个；
- 训练 loss 大致从 check 阶段量级下降到 0.5–1.5 区间（v2 ANALYSIS 段 loss 不会到 0，
  因为 teacher 文本本身有随机性，模型不可能逐 token 完美复现）。

---

## 6. 评估（约 10–30 分钟）

完全复用 v1 的 eval / probe 脚本，自动检测 v2 格式：

```bash
# 小样本 + 完整 dump
python tools/eval_sft_v1.py \
    --lora-dir checkpoints/sft_v2_lora \
    --val-jsonl checkpoints/sft_v2_lora/runtime_teacher_data/val.jsonl \
    --save-root checkpoints/sft_v2_lora \
    --max-samples 100

# 全集分片
torchrun --standalone --nproc_per_node=4 tools/eval_sft_v1.py \
    --lora-dir checkpoints/sft_v2_lora \
    --val-jsonl checkpoints/sft_v2_lora/runtime_teacher_data/val.jsonl \
    --save-root checkpoints/sft_v2_lora
```

**关键参数变化**：v1 默认 val 路径写死 `checkpoints/sft_v1_data/val.jsonl`，v2 必须显式
`--val-jsonl checkpoints/sft_v2_lora/runtime_teacher_data/val.jsonl`，否则会评 v1 数据集或 pending 占位数据。
eval / probe 会默认自动挑空闲 GPU；多卡 eval 用 `torchrun --nproc_per_node=N`
时会自动挑 N 张。

### probe 也复用 v1 脚本

```bash
python tools/probe_sft_v1.py \
    --lora-dir checkpoints/sft_v2_lora \
    --val-jsonl checkpoints/sft_v2_lora/runtime_teacher_data/val.jsonl \
    --save-root checkpoints/sft_v2_lora \
    --num-per-scenario 3 --seed 42 \
    --case-suffix "_v2"
```

case 目录里的 `gt.txt` 现在是真实的 teacher ANALYSIS（不再是 `Observations recorded.`），
对照 `pred.txt` 可以直观看出 LoRA 学到了多少 teacher 风格。

---

## 7. v1 vs v2 横向比较（推荐做法）

最简方式：同一份 val（用 v1 数据集或 v2 数据集都行，关键是同一份）分别跑两个 LoRA 的 eval。

```bash
# v1 LoRA 在 v1 val 上的 baseline
python tools/eval_sft_v1.py \
    --lora-dir checkpoints/sft_v1_lora \
    --val-jsonl checkpoints/sft_v1_data/val.jsonl \
    --save-root checkpoints/sft_v1_lora \
    --max-samples 200

# v2 LoRA 在 v2 val 上的指标（注意 GT 不同，但 STATUS/SUBGOAL GT 一致）
python tools/eval_sft_v1.py \
    --lora-dir checkpoints/sft_v2_lora \
    --val-jsonl checkpoints/sft_v2_lora/runtime_teacher_data/val.jsonl \
    --save-root checkpoints/sft_v2_lora \
    --max-samples 200
```

对比四项指标 + 任选 case 的 `summary.md`，确认：
- `keep_accuracy` v2 ≥ v1（v1 这项本就好）
- `early_advance_rate` v2 < v1（v1 这项是核心痛点）
- v2 case 的 `pred.txt` 三段输出齐全，不再像 v1 那样只剩 ANALYSIS 复读

---

## 8. 出问题时贴什么内容方便排查

| 步骤 | 贴这些 |
|---|---|
| 阶段 1 | `checkpoints/sft_v2_data_pending/stats.json` |
| teacher 预览 / runtime | 预览网页截图，或 runtime 目录的 fallback 比例统计 + 随机 5 条 ANALYSIS |
| §3 sanity | inline 脚本完整 stdout |
| §4 check | check 模式输出最后 30 行（含 loss 数值） |
| §5 训练中 | 每 100 step 的 loss / grad_norm 趋势 |
| §6 评估后 | `metrics.json` 全文 + 任意 1 个 case 的 `summary.md` |

---

## 9. 与 v3 / KL 正则的关系

v2 通过 4 项指标且 probe ANALYSIS 段不复读 → v2 收官，进 [SFT_V2_PLAN.md §9](SFT_V2_PLAN.md) v3 的 KL 路线
（或直接跳过 v3，转入 GoalGen 那条线）。

v2 没通过（例如 ANALYSIS 段还在漂移）→ 按 PLAN §8 风险表逐条排查，不要直接堆 v3 上去。
