# SFT v2 运行教程 — 从生成 teacher 数据到拿到评估指标

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

## 2. 阶段 2：teacher 推理填 ANALYSIS（**GPU**，8 卡分片约 100 分钟）

这是 v2 最重的一步。teacher 加载冻结 base Qwen，对每条 pending 样本跑一次推理产 ANALYSIS。

### 2.1 8 卡分片跑（推荐）

```bash
torchrun --standalone --nproc_per_node=8 tools/build_sft_dataset_v2_teacher.py \
    --pending-dir checkpoints/sft_v2_data_pending \
    --output-dir checkpoints/sft_v2_data \
    --model-dir checkpoints/Qwen3-VL-4B-Instruct \
    --seed 20260601
```

每个 rank 处理 `sample_idx % world_size == rank` 的样本，各自落盘到
`<output-dir>/train.jsonl.rank<R>` / `val.jsonl.rank<R>`，跑完后 rank0 自动合并为
`train.jsonl` / `val.jsonl`，删 rank 分片文件。

### 2.2 单卡跑（小批量调试或 8 卡不可用）

```bash
CUDA_VISIBLE_DEVICES=0 python tools/build_sft_dataset_v2_teacher.py \
    --pending-dir checkpoints/sft_v2_data_pending \
    --output-dir checkpoints/sft_v2_data \
    --max-samples 32           # 只跑前 32 条，验证流水线
```

### 2.3 中断后续跑

teacher 脚本启动时会扫描 `<output-dir>/train.jsonl` 与 `val.jsonl` 已有内容，
按 `(scenario, run_id, anchor)` 三元组做指纹，跳过已生成样本。直接重跑同一条命令即可继续：

```bash
# 中断了，重跑（同一条命令）
torchrun --standalone --nproc_per_node=8 tools/build_sft_dataset_v2_teacher.py \
    --pending-dir checkpoints/sft_v2_data_pending \
    --output-dir checkpoints/sft_v2_data \
    --model-dir checkpoints/Qwen3-VL-4B-Instruct \
    --seed 20260601
```

### 2.4 通过条件

- `<output-dir>/train.jsonl` 行数 == `<pending-dir>/train.jsonl` 行数；val 同理。
- 任意行 `dataset_version == "v2"`、`teacher_meta.model_dir` 不为空。
- `teacher_meta.fallback == true` 的比例 < 5%（PLAN §8）。

快速统计：

```bash
python -c "
import json
n_total = n_fb = 0
with open('checkpoints/sft_v2_data/train.jsonl') as f:
    for line in f:
        r = json.loads(line)
        n_total += 1
        if r.get('teacher_meta', {}).get('fallback'): n_fb += 1
print(f'total={n_total} fallback={n_fb} ratio={n_fb/max(n_total,1):.2%}')
"
```

### 2.5 抽检 teacher 输出质量

```bash
python -c "
import json, random
random.seed(0)
with open('checkpoints/sft_v2_data/train.jsonl') as f:
    rows = [json.loads(l) for l in f]
for r in random.sample(rows, 5):
    print('---', r['scenario'], r['run_id'], 'anchor=', r['anchor'])
    print(r['messages'][2]['content'])
    print()
"
```

人工 review 几条：
- ✅ ANALYSIS 单行、2–4 句、按"看图 → 变化 → 结论"三步式
- ❌ 出现 PRIVILEGED 字眼 / 直接说 `"the current STATUS is X"` → teacher 没遵守约束，
  考虑改 §3.1 system prompt 或加强后处理 strip
- ❌ 大量样本同一句套话 → teacher temperature=0 + 简单 prompt 太死板，加 `--teacher-temperature 0.3`

---

## 3. 静态 sanity：v2 plugin mask 是否对（CPU，< 30 秒）

v2 没有专用 `check_loss_mask.py`，但可以用一段 inline 脚本验证 plugin regex 切片是否正确。

```bash
python -c "
import sys, json
sys.path.insert(0, 'tools')
from sft_v2_loss_scale_plugin import SftV2AnalysisSupervisedLossScale
plugin = SftV2AnalysisSupervisedLossScale()
ctx = 'ANALYSIS: I see a foggy tunnel with cars ahead. Compared to the earlier frame the ego has slowed down. This visual evidence supports advancing to hazard_detect.\nSTATUS: hazard_detect\nSUBGOAL: max_brake_or_min_gap'
parts, scales = plugin.get_loss_scale(ctx)
for p, s in zip(parts, scales):
    print(f'w={s:.2f}: {p!r}')
"
```

**预期输出**（6 段；上下文末尾无 EOS 时无 tail 段，加上 EOS 时多 1 段，最多 7 段）：

```
w=0.00: 'ANALYSIS: '
w=0.30: 'I see a foggy tunnel ... advancing to hazard_detect.'
w=0.00: '\nSTATUS: '
w=1.00: 'hazard_detect'
w=0.00: '\nSUBGOAL: '
w=1.00: 'max_brake_or_min_gap'
```

**通过条件**：
- `"".join(parts) == ctx`（必须，swift 内部对齐要求）
- ANALYSIS body 权重 0.3
- 两个 event_name 权重 1.0
- 其余字面 0

---

## 4. 动态 sanity：跑 2 step 看真实 loss（**GPU**，约 1–2 分钟）

```bash
bash tools/sft_v2_train.sh check
```

与 v1 一样，2 step、不保存 ckpt、不跑 val。预期初始 loss 在 **3–8 区间**（比 v1 偏高，因为
v2 多了 ANALYSIS 段约 30 个 token 参与 loss，初始 nll 也包含进去）。

判读：

| 现象 | 判读 | 处理 |
|---|---|---|
| loss ∈ [3, 8] | ✅ ANALYSIS + STATUS + SUBGOAL 都在算 loss | 进 §5 正式训练 |
| loss < 1 | ❌ plugin 把所有段都 mask 了 | 跑 §3 sanity 看 regex 是否漂移 |
| loss > 12 | ⚠️ teacher ANALYSIS 太长或字面被错误算 loss | 看 `MAX_LENGTH` 是否截断；看 plugin 切片是否对 |

---

## 5. 正式训练（**8×H20 DDP，约 2 小时**）

```bash
bash tools/sft_v2_train.sh ddp
```

GPU / 端口 / DDP rendezvous 行为与 v1 完全一致（自动选最空闲卡、自动找空闲 MASTER_PORT、
NCCL_P2P_LEVEL=NVL 等）。所有 v1 的 `DDP_GPU_COUNT` / `SFT_RESPECT_*` 环境变量在 v2 同名。

**预期**：
- 8 卡总 step ≈ 900（与 v1 同）；4 卡 ≈ 1800；
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
    --val-jsonl checkpoints/sft_v2_data/val.jsonl \
    --save-root checkpoints/sft_v2_lora \
    --max-samples 100

# 全集分片
torchrun --standalone --nproc_per_node=4 tools/eval_sft_v1.py \
    --lora-dir checkpoints/sft_v2_lora \
    --val-jsonl checkpoints/sft_v2_data/val.jsonl \
    --save-root checkpoints/sft_v2_lora
```

**关键参数变化**：v1 默认 val 路径写死 `checkpoints/sft_v1_data/val.jsonl`，v2 必须显式
`--val-jsonl checkpoints/sft_v2_data/val.jsonl`，否则会评 v1 数据集。

### probe 也复用 v1 脚本

```bash
python tools/probe_sft_v1.py \
    --lora-dir checkpoints/sft_v2_lora \
    --val-jsonl checkpoints/sft_v2_data/val.jsonl \
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
    --val-jsonl checkpoints/sft_v2_data/val.jsonl \
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
| 阶段 2 | §2.4 的 fallback 比例统计 + §2.5 随机 5 条 ANALYSIS |
| §3 sanity | inline 脚本完整 stdout |
| §4 check | check 模式输出最后 30 行（含 loss 数值） |
| §5 训练中 | 每 100 step 的 loss / grad_norm 趋势 |
| §6 评估后 | `metrics.json` 全文 + 任意 1 个 case 的 `summary.md` |

---

## 9. 与 v3 / KL 正则的关系

v2 通过 4 项指标且 probe ANALYSIS 段不复读 → v2 收官，进 [SFT_V2_PLAN.md §9](SFT_V2_PLAN.md) v3 的 KL 路线
（或直接跳过 v3，转入 GoalGen 那条线）。

v2 没通过（例如 ANALYSIS 段还在漂移）→ 按 PLAN §8 风险表逐条排查，不要直接堆 v3 上去。
