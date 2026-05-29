# SFT v1 运行教程 — 从生成数据到拿到评估指标

> 本文档是 [SFT_V1_PLAN.md](SFT_V1_PLAN.md) 的"操作手册"对照：PLAN 讲设计与
> 决策依据，本 RUN 讲实际怎么跑。
>
> **关键约定**：所有命令默认 **从 `AutoMoT/` 目录执行**（远程默认 cwd），
> 不是从仓库根 `automot_lead/`。所以脚本路径写 `tools/...` 不是 `AutoMoT/tools/...`，
> checkpoint 路径写 `checkpoints/...` 不是 `AutoMoT/checkpoints/...`。
> 唯一例外：`keyframes_all_scenarios.json` 在仓库根而非 `AutoMoT/` 下，
> 所以从 AutoMoT/ 视角是 `../keyframes_all_scenarios.json`。

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
MODEL_DIR=/data/lead_data/checkpoints/Qwen3-VL-4B-Instruct \
  bash tools/sft_v1_train.sh ddp
```

---

## 1. 生成 SFT 数据集（CPU，约 1–3 分钟）

```bash
python tools/build_sft_dataset_v1.py \
    --keyframes ../keyframes_all_scenarios.json \
    --data-root /data/lead_data/data \
    --samples-per-scenario 200 \
    --output-dir checkpoints/sft_v1_data
```

**预期输出**（节选）：

```
[load] 7326 total runs in keyframes
[filter] kept 7326 runs; skipped by status: {}
[stratify] Accident         keep= 686 adv= 20 -> chosen=200 (adv=50)
[stratify] AccidentTwoWays  keep=1024 adv= 20 -> chosen=200 (adv=50)
...
[split] train=~7560  val=~840
[write] checkpoints/sft_v1_data/train.jsonl
[write] checkpoints/sft_v1_data/val.jsonl
[write] checkpoints/sft_v1_data/stats.json
```

**通过条件**：

- `train.jsonl` + `val.jsonl` + `stats.json` 三个文件都生成；
- `stats.json` 里 `transition_in_train` ≈ 总数的 25%；
- 单条样本里 `messages[1].content` 含 4 个 `<image>` 占位符；
- `images` 列表里 4 个 RGB 路径都指向 `/data/lead_data/data/<scenario>/<run_id>/rgb/*.jpg`。

**常见报错**：

| 现象 | 原因 | 处理 |
|---|---|---|
| `keyframes_all_scenarios.json` 找不到 | 路径写错（仓库根 vs AutoMoT/） | 用 `--keyframes ../keyframes_all_scenarios.json` 或 `/data/lead_data/keyframes_all_scenarios.json` 绝对路径 |
| 某些 scenario 提示样本不足 200 | 该场景 `Completed/Perfect` run 太少 | 不影响，会自动按现有量取；看 `stats.json` 里 `chosen_total` 哪些场景 < 200 |
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

[mask] regex matched chars [0,32)  -> 'ANALYSIS: Observations recorded.'

 idx tag        id     char_range  decoded
--------------------------------------------------------------------------------
   0 [MASK]  19394          [0,9)  'ANALYSIS:'
   1 [MASK]    220         [9,10)  ' '
   2 [MASK]   4571        [10,13)  'Obs'
   ...
   7 [LOSS]    198        [32,33)  '\n'
   8 [LOSS]  31650        [33,40)  'STATUS:'
   ...

[summary] total tokens = 18, mask = 7, loss = 11
```

**通过条件**（**必须全部满足**）：

- ANALYSIS 段每个 token 都打 `[MASK]`；
- `STATUS:` / `SUBGOAL:` 行的 token 都打 `[LOSS]`；
- `summary` 里 `n_mask ≥ 5`、`n_loss ≥ 8`；
- **无 `[WARN]` 行输出**。

**异常处理**：看到 `[WARN]` 必须先修再继续，常见原因：

| WARN | 原因 | 处理 |
|---|---|---|
| `regex 匹配不到` | `PLACEHOLDER_ANALYSIS` 与 `LOSS_SCALE_REGEX` 漂移 | 检查 `build_sft_dataset_v1.py` 与 `sft_v1_train.sh` 是否同步 |
| `没有任何 token 被 mask` | tokenizer 把 ANALYSIS 整段合成单 token 并跨过 `\nSTATUS:` 边界 | 换一个不会被 tokenizer 合并的占位句 |
| `算 loss 的 token 太少` | STATUS / SUBGOAL 行被吞掉了 | 检查 jsonl 里 assistant content 三段格式是否完整 |

---

## 3. 动态 sanity：跑 2 step 看真实 loss 数值（**需要 GPU**，约 1–2 分钟）

```bash
bash tools/sft_v1_train.sh check
```

这个命令会通过 `--external_plugins tools/sft_v1_loss_scale_plugin.py` 注册
`sft_v1_analysis_mask`，再用 `--loss_scale sft_v1_analysis_mask` mask 掉
ANALYSIS 占位段。ms-swift 3.12.x 不接受 JSON regex 形式的 `--loss_scale`。
`check` 模式默认用 `nvidia-smi` 自动选择当前最空闲的一张 GPU，并且不传 `--val_dataset`，
所以只跑 2 个训练 step，不会加载/评估 val 集的约 800 条样本。

**预期 loss 数值**（健康范围）：

```
{'loss': 1~10, 'grad_norm': ..., 'learning_rate': ..., 'epoch': 0.0x}
```

**判读规则**：

| 现象 | 判读 | 处理 |
|---|---|---|
| `python tools/check_loss_mask.py` 的 plugin sanity 显示 STATUS/SUBGOAL `in_loss=True, in_mask=False`，且 `check` loss 有限非 NaN | ✅ 训练侧 mask 大方向正常 | 可进 step 4 |
| `loss < 3` 但 plugin sanity 通过 | ⚠️ base 模型对固定格式和短 event token 预测很容易，不单独视为失败 | 继续看正式训练/评估指标 |
| `loss < 0.1` 或 `grad_norm=0` | ❌ 可能 STATUS/SUBGOAL 也被 mask 了 | 先查 `check_loss_mask.py` 的 plugin sanity |
| `loss > 12` | ❌ ANALYSIS 段也算 loss 了 | 走 PLAN §11 回退：写 `tools/sft_v1_preprocessor.py` 手动 mask labels |
| check 结束保存了 `checkpoint-2` | ❌ check 模式不该保存 checkpoint | 拉最新脚本，确认含 `--save_strategy no` |

**常见启动报错**：

| 现象 | 原因 | 处理 |
|---|---|---|
| `swift: command not found` | 当前环境没装 ms-swift 或 PATH 不对 | 先确认 `which python && which swift && pip show ms-swift` |
| `KeyError: 'sft_v1_analysis_mask'` | 插件没被加载，loss_scale 策略未注册 | 确认从 `AutoMoT/` 目录运行；检查 `tools/sft_v1_loss_scale_plugin.py` 是否存在 |
| `KeyError: '{"ANALYSIS...": 0.0}'` | 仍在用旧版 JSON regex 命令 | 拉最新脚本，确认 `sft_v1_train.sh` 里有 `--external_plugins` |
| `invalid device ordinal` / CUDA 选错卡 | 远程调度只分配了部分卡，或 `CUDA_VISIBLE_DEVICES` 与实际可见卡不一致 | 不手动指定时脚本会自动挑空闲卡；若调度系统已分配卡，显式使用它给出的 `CUDA_VISIBLE_DEVICES` |

---

## 4. 正式训练（**8×H20 DDP，约 1.5 小时**）

```bash
bash tools/sft_v1_train.sh ddp
```

**预期**：

- 总 step ≈ 710（按 7560 train 样本 / 等效 bs 32 × 3 epoch）；
- 每 100 step 保存一次 LoRA adapter 到 `checkpoints/sft_v1_lora/checkpoint-XXX/`；
- 训练 loss 从 ~7-8 降到 ~1-2。

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

---

## 5. 评估（约 10–30 分钟，取决于 val 大小）

```bash
# 完整 val 集 + anchor=12 fail case sanity
python tools/eval_sft_v1.py \
    --lora-dir checkpoints/sft_v1_lora

# 或先快速验收前 100 条样本看趋势
python tools/eval_sft_v1.py \
    --lora-dir checkpoints/sft_v1_lora \
    --max-samples 100
```

**预期输出末尾**（节选）：

```json
{
  "n_total": 840,
  "n_keep": 630,
  "n_advance": 210,
  "keep_accuracy": 0.96,
  "advance_accuracy": 0.65,
  "early_advance_rate": 0.03,
  "anchor12_sanity": {
    "enabled": true,
    "passed": true,
    "pred_status": "initial",
    "expected_status": "initial"
  }
}
```

**通过条件**（与 PLAN §8 一致）：

| 指标 | v1 目标 | 优先级 |
|---|---|---|
| `keep_accuracy` | ≥ 0.95 | 高 |
| `advance_accuracy` | ≥ 0.60 | 中 |
| **`early_advance_rate`** | **≤ 0.05** | **最高（核心痛点）** |
| **`anchor12_sanity.passed`** | **= true** | **必须** |

### Baseline 对照（强烈推荐）

先跑 base 模型留个对照，再跑 LoRA 看是否真的解决了"过早推进"：

```bash
# base 模型（不挂 LoRA），跑前 200 条 val 做 baseline
python tools/eval_sft_v1.py \
    --lora-dir "" \
    --max-samples 200 \
    --output-json eval_json/sft_v1_metrics_base.json

# 训完再跑同样规模 LoRA
python tools/eval_sft_v1.py \
    --lora-dir checkpoints/sft_v1_lora \
    --max-samples 200 \
    --output-json eval_json/sft_v1_metrics_lora.json
```

对比 `early_advance_rate`：base 通常 0.3–0.5，LoRA 应降到 < 0.05。
如果 LoRA 后仍 > 0.15，说明 v1 LoRA 没真正学到"默认保持"——
进 PLAN §11 风险表排查 chat template 一致性问题。

---

## 一行串起来（happy path，不推荐生产用）

```bash
python tools/build_sft_dataset_v1.py --data-root /data/lead_data/data \
  --keyframes ../keyframes_all_scenarios.json \
  --output-dir checkpoints/sft_v1_data && \
python tools/check_loss_mask.py && \
bash tools/sft_v1_train.sh check && \
bash tools/sft_v1_train.sh ddp && \
python tools/eval_sft_v1.py --lora-dir checkpoints/sft_v1_lora
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
| step 5 后 | `eval_json/sft_v1_metrics.json` 全文 |

---

## 7. 与 v2 / 后续迭代的关系

- v1 完成（4 项指标全过）后，进 [SFT_V1_PLAN.md §9](SFT_V1_PLAN.md) 列出的 v2 计划：
  ANALYSIS 蒸馏（用 v1 LoRA + GT status 反向写 analysis 真值）；
- v1 没过的话，按 [SFT_V1_PLAN.md §11](SFT_V1_PLAN.md) 风险表逐条排查，
  不要直接堆 v2 上去。
