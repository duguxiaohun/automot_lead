# SFT v2 设计文档 — ANALYSIS 监督升级 + 双 prompt 蒸馏

> v1 的核心痛点是"监督信号只覆盖 5% 的 token、但 LoRA 梯度扰动 100% 的 forward"，
> 导致无论怎么调 lr / epoch / 早停，ANALYSIS 段都会被无监督扰动拉崩，最终连
> `STATUS:` / `SUBGOAL:` 行都生成不出来。v2 用"冻结 base Qwen 蒸馏 ANALYSIS"把
> 监督信号补满到 100%，从根上修复这个范式缺陷。
>
> 关键约定：所有命令默认 **从 `AutoMoT/` 目录执行**，操作流程见 [SFT_V2_RUN.md](SFT_V2_RUN.md)。

---

## 1. 为什么要 v2 — v1 的根因诊断

v1 [SFT_V1_PLAN.md](SFT_V1_PLAN.md) 的设计是"只让 STATUS / SUBGOAL 的事件名 token 算 loss，其他都 mask=0"。
实测发现（详见 v1 ckpt-8100 与 ckpt-3764 两轮失败 case）：

| 现象 | 根因 |
|---|---|
| ANALYSIS 段复读 `Observations recorded.` → `tunnel, tunnel, tunnel` | LoRA 低秩增量同时作用于全 forward path，无监督段没有 loss 约束，分布漂移 |
| 训练 loss 还在缓慢下降时 STATUS / SUBGOAL 已不再生成 | 事件名 token nll 已 ≈ 0.01（早就过收敛），继续训只在压剩余边际，副作用扩散 |
| 早停 ckpt（200/600/900）也崩 | 监督信号覆盖面是结构问题，不是训练时长问题 |

→ **必须让 ANALYSIS 段也带监督信号**，从而约束 LoRA 不漂移。但 ANALYSIS 段在 keyframes
原始标注里**没有真值**，只能蒸馏出来。

v1 PLAN §9 原本写"用 v1 LoRA + GT status 自蒸馏"，但前提是 v1 能跑通 — 现在 v1 自己就崩了，
自蒸馏直接退化。**v2 改用冻结 base Qwen 做 teacher**。

---

## 2. 核心设计 — 两条 prompt、pending 数据与运行时 teacher

### 2.1 Teacher 管线（训练前预览 / 训练启动时临时产 ANALYSIS）

- **输入**：4 帧 RGB（与 student 完全一致）+ MEMORY（anchor-K 的 STATUS）+ **PRIVILEGED 块**
  （当前 anchor 的 GT STATUS 与 transition flag）。
- **任务**：从已知答案反推视觉证据，按"先描述图片可见内容 → 描述帧间变化 → 简要说明
  支持当前 STATUS 的视觉依据"三步式写 2–4 句 ANALYSIS。
- **执行者**：冻结的 `Qwen3-VL-4B-Instruct` base 模型（不带 LoRA）。
- **输出**：纯文本 ANALYSIS 正文（不带 `ANALYSIS:` 前缀），只写入预览目录或训练时
  runtime jsonl；不回写长期维护的 pending 数据集。

### 2.2 Student 管线（LoRA SFT 训练）

- **输入**：4 帧 RGB + MEMORY（anchor-K 的 STATUS）。**永远不见 PRIVILEGED**。prompt 与
  v1 完全一致（复用 `qwen3vl_local.prompt_pipeline.build_system_prompt` / `build_user_prompt`）。
- **GT 三段**：
  - `ANALYSIS: <teacher 生成的视觉证据>` ← 蒸馏自 base
  - `STATUS: <GT>` ← keyframes 标注
  - `SUBGOAL: <GT>` ← 状态机推导
- **loss 覆盖**：100% assistant token 都参与 loss，但差异化加权（详见 §5）。

### 2.3 两条 prompt 必须严格分离

**绝对禁止**把 PRIVILEGED 块写进 student jsonl 的 user content。一旦 student 训练时见过
PRIVILEGED 字段，推理分布就废了 —— 因为推理时根本拿不到 GT STATUS。

实现层面：v2 jsonl 的 `messages` 字段里 system / user 与 v1 完全 byte 级相同；
PRIVILEGED 只出现在 teacher 推理时临时拼装的 prompt 里。长期 pending jsonl 不落盘
teacher ANALYSIS；训练时 runtime jsonl 会落盘 teacher 结果，作为本次训练的临时缓存。

---

## 3. Teacher prompt 模板（v2 实现固定文本）

### 3.1 Teacher system prompt

```
You are a vision-grounded annotation teacher for an autonomous driving
status-tracking task.

Input:
- 4 RGB frames (oldest -> newest), stitched three-camera view.
- MEMORY: the previous anchor (anchor-K) STATUS and EVENT_SEQUENCE.
- PRIVILEGED: the ground-truth current STATUS at the newest frame, and
  whether this anchor is KEEP (state unchanged) or ADVANCE (state moved
  forward from MEMORY STATUS).

Task:
Produce a single line of ANALYSIS that a student model (which does NOT see
PRIVILEGED) could plausibly infer from images alone. Sentence order MUST be:
1. First sentence: concretely describe what is visible in the LAST frame.
2. Second sentence: describe what CHANGED between the earliest and the
   latest frame.
3. Third sentence: state whether the observed evidence supports staying at
   MEMORY STATUS or advancing to the current STATUS, tying it to the visual
   evidence above.

Constraints:
- Do NOT mention or reference the PRIVILEGED block; write as if from images only.
- Do NOT invent visual content not actually present.
- Be concise, grounded, factual; 2-4 sentences total, all on a single line.
- Do NOT output STATUS or SUBGOAL; only the ANALYSIS body text (no "ANALYSIS:" prefix).

Output EXACTLY one line of text (the ANALYSIS body, no prefix, no trailing newline).
```

### 3.2 Teacher user prompt（在 student user prompt 基础上加 PRIVILEGED）

```
<image><image><image><image>
The 4 images above are ordered oldest to newest; the last image is the current moment.

[MEMORY]
SCENARIO: ...
EVENT_SEQUENCE: ... -> ... -> ... -> ... -> final
EVENT_DESCRIPTIONS:
- ...
STATUS: <prev_anchor_status>
SUBGOAL: <next_in_sequence>
COMPLETED: ...
[/MEMORY]

[PRIVILEGED]
CURRENT_GT_STATUS: <target_status>
TRANSITION: <keep|advance>
PREV_STATUS: <prev_anchor_status>
[/PRIVILEGED]

Given the observations, memory, and privileged ground truth, output the
ANALYSIS body that the student should plausibly produce from images alone.
```

### 3.3 Teacher 输出后处理

teacher 是生成模型，不保证严格遵守"单行无前缀"。后处理 pipeline：

1. `lstrip("ANALYSIS:")` + `strip()` — 兜底去掉可能的前缀。
2. 截断到第一个 `\nSTATUS:` / `\nSUBGOAL:` / `\n\n` 之前 — 防 teacher 自作主张续写。
3. 把剩余的 `\n` 替换为空格 — 强制单行。
4. 截断到最长 480 字符 — 防个别样本 teacher 跑飞拉长 jsonl。
5. 如果空白后剩余 < 20 字符（teacher 输出垮了）→ 兜底用 `"Observations recorded."` 占位，
   并在 `teacher_meta.fallback=true` 里 mark，便于事后排查比例。

---

## 4. 数据格式 + 运行时物化管线

### 4.1 v2 jsonl 字段

在 v1 schema 基础上加：

```json
{
  "dataset_version": "v2",            // "v2_pending" 长期保存；"v2" 仅用于 runtime/调试物化产物
  "scenario": "...", "run_id": "...", "anchor": N, "prev_anchor": M,
  "images": [...],
  "messages": [
    {"role": "system",    "content": <v1 system prompt 原样>},
    {"role": "user",      "content": <v1 user prompt 原样，含 <image> 占位>},
    {"role": "assistant", "content":
        "ANALYSIS: <teacher 生成内容>\nSTATUS: <GT>\nSUBGOAL: <GT>"}
  ],
  "is_transition_sample": bool,
  "teacher_meta": {                    // 仅 dataset_version == "v2" 时存在
    "model_dir": "checkpoints/Qwen3-VL-4B-Instruct",
    "seed": 20260601,
    "generated_at": "2026-06-01T15:30:00Z",
    "analysis_chars": 287,
    "fallback": false               // teacher 跑飞回退到 Observations recorded.
  }
}
```

`dataset_version` 字段让 `eval_sft_v1.py` / `probe_sft_v1.py` 自动检测格式（v1 跳过、v2 沿用）。

### 4.2 阶段 1 — `build_sft_dataset_v1.py --mode v2`

复用 v1 全部采样逻辑，只改 `build_messages` 的 assistant 段：把 ANALYSIS 文本换成特殊占位
`__TEACHER_PENDING__`，并写 `dataset_version: "v2_pending"`。

产物：`<output-dir>/train.jsonl` / `val.jsonl` / `stats.json`（schema 同 v1，区别靠目录隔离）。

### 4.3 运行时 teacher 物化 — `sft_v2_train.sh` + `build_sft_dataset_v2_teacher.py`

默认训练入口是 `bash tools/sft_v2_train.sh ddp|single|check`。它会先读取
`TRAIN_JSONL` 第一条样本：

- 如果 `dataset_version == "v2_pending"`：脚本**启动时自动 bulk 物化** runtime teacher
  jsonl 到 `RUNTIME_TEACHER_DIR`（默认 `checkpoints/sft_v2_lora/runtime_teacher_data/`），
  然后再进 swift sft。整个过程不需要显式 opt-in flag。
  - 若 `RUNTIME_TEACHER_DIR` 下已有完整 `dataset_version == "v2"` 的 `train.jsonl` /
    `val.jsonl`，脚本直接复用，跳过 100 min 全量物化。
  - 若没有可复用 runtime 数据，脚本会自动调用 `build_sft_dataset_v2_teacher.py`
    跑一份完整 runtime jsonl，落盘到 `RUNTIME_TEACHER_DIR`，pending 源数据不被修改。
  - `check` 模式只物化最多 32 条小样本用于 sanity，默认写到
    `runtime_teacher_check_data/`，不跑全集也不覆盖正式 runtime cache。
- 如果 `dataset_version == "v2"`：说明用户显式传入了已物化 jsonl，训练脚本直接使用。

默认 `RUNTIME_TEACHER_REFRESH=0` = 已有 runtime cache 时直接复用；显式
`RUNTIME_TEACHER_REFRESH=1` 会清掉旧 runtime jsonl 并强制重跑 teacher，用于 keyframes /
prompt 改过、必须丢弃旧 ANALYSIS 的场景。teacher 物化本身支持断点续跑
（详见下文 step 3 与 step 4），所以上次 100 min 中途断了，下一次默认就会续完。

`build_sft_dataset_v2_teacher.py` 输入 pending jsonl，流程：

1. 加载冻结 base Qwen（`local_files_only=True`，bf16，eval mode）。
2. 对每条 pending 样本：
   - 解析 MEMORY STATUS / target_status / transition flag。
   - 拼 teacher system + user prompt（§3）。
   - `engine.generate(...)` 跑一次，max_new_tokens=256，temperature=0（greedy）。
   - 走 §3.3 后处理拿到干净 ANALYSIS 文本。
   - 把 `messages[2].content` 里的 `__TEACHER_PENDING__` 替换为生成结果。
   - `dataset_version` 改 `"v2"`，填 `teacher_meta`。
3. **resumable**：每完成 100 条 flush 一次输出 jsonl；启动时如果输出 jsonl 已存在，
   按 `(scenario, run_id, anchor)` 去重跳过已完成样本。
4. **多卡分片**：读 `RANK / WORLD_SIZE`，每个 rank 处理 `sample_idx % world_size == rank`
   的样本，各自落盘到 `train.jsonl.rank<R>`；最后 rank0 合并。

预计耗时：14400 train + 1600 val ≈ 16000 样本 × 3 秒/样本 ÷ 8 卡 ≈ 100 分钟。
这一步是训练启动时自动落盘的 runtime 缓存，不再要求用户维护 `checkpoints/sft_v2_data/`
这种固定 teacher 数据集，也不会回写长期 pending 数据集；下次启动若 cache 仍在则复用。

### 4.4 student GT 拼接位置

teacher 只产出 ANALYSIS 正文。student GT 三段在 runtime 物化时
直接拼接：

```python
assistant = (
    f"ANALYSIS: {teacher_text}\n"
    f"STATUS: {target_status}\n"
    f"SUBGOAL: {target_subgoal}"
)
```

注意末尾**不带换行**（与 v1 完全对齐，便于 plugin regex 复用同款边界）。

---

## 5. Loss 权重设计 — `sft_v2_loss_scale_plugin.py`

策略名：`sft_v2_analysis_supervised`。切片如下（**2026-06-02 修订**：结构字面从
0 改为 1.0；tail/EOS 若由 ms-swift chat template 放进 plugin context，也必须为
1.0，详见下方"结构字面为什么不能 mask"）：

| 段 | 权重 | 理由 |
|---|---|---|
| `ANALYSIS:` 字面 + 空格 | **1.0** | **起手结构信号**（v2 修订）— 不能只靠 base/chat template 先验 |
| ANALYSIS body text | **0.3** | 蒸馏目标 — teacher 输出有随机性，0.3 让 student 学到"形状与事实"但不强制逐 token 复现 |
| 换行 `\n` + `STATUS:` 字面 + 空格 | **1.0** | **段切换信号**（v2 修订）— 0 会让 ANALYSIS 循环复读 |
| STATUS event_name | **1.0** | 核心监督 |
| 换行 `\n` + `SUBGOAL:` 字面 + 空格 | **1.0** | **段切换信号**（v2 修订）— 0 会让 STATUS 后无切换 |
| SUBGOAL event_name | **1.0** | 核心监督 |
| 末尾 tail / EOS / `\n`（若进入 plugin context） | **1.0** | **停止信号**（v2 修订）— 若该 token 由 ms-swift 模板传入，不能 mask |

regex 在 v1 `_FULL_PATTERN` 基础上加一个 `(?P<analysis>...)` 捕获组，把 ANALYSIS body
单独切出来 weight 0.3。

### 结构字面为什么不能 mask（v2.0 踩坑，2026-06-02）

v2.0 初版照搬 v1 思路把 `ANALYSIS:` / `\nSTATUS:` / `\nSUBGOAL:` 以及可能进入 context 的 tail/EOS 全部 mask 成 0，理由是
"关键词字面无学习价值"。实测在 ckpt-7526 上推理：

- 模型输出陷入 `ANALYSIS: ... ANALYSIS: ... ANALYSIS: ...` 循环复读，
  顶到 `max_gen_tokens=512` 都不出 `STATUS:`；
- `pred_status` / `pred_subgoal` 全部 None；
- 训练 loss 顺利下降，但自由生成完全失控。

**根因**：cross-entropy 下 weight=0 ⇔ 这个位置 next-token-prediction 没梯度。
v1 ANALYSIS 是固定占位（`Observations recorded.`，7 token），段切换 mask=0 也没事，
因为 base 先验在 7 token 后切到 `\n` 几乎是必然的；v2 ANALYSIS 升到 80-150 token 后，
LoRA 学到的"ANALYSIS body 风格"先验在长上下文中累积，**没有梯度推它切到 `\n`**，
自由生成时自然倾向继续写下一段 ANALYSIS。

**修法**：所有结构性字面（`ANALYSIS:` / `\nSTATUS:` / `\nSUBGOAL:`）必须 weight ≥ 1.0；
tail/EOS 若由 ms-swift chat template / runtime 拼进 plugin context，也必须 weight ≥ 1.0。
三个结构字面加起来约十几个 token，相对 ANALYSIS body 80-150 token × 0.3 ≈
24-45 的等效梯度量，**不会主导 loss**，只是补一条结构性约束。静态脚本只能用
synthetic tail 验证 plugin 对 tail/EOS 的权重规则；真实 EOS 是否进 loss 以 ms-swift
实际 context 为准。

### 为什么 ANALYSIS body 是 0.3 不是 1.0

- teacher 输出是采样结果，**不是真值**。把 ANALYSIS 也给 1.0 等于强迫 student 完美复现
  teacher 的语气和措辞，等于让 student 学到 teacher 的"随机噪声"。
- 0.3 是经验值（GoalGen 等多任务训练里常用区间），保证 ANALYSIS 段有梯度信号约束 LoRA
  不漂移，又不会主导整体 loss。
- 等效"loss 权重比"（v2 修订后）：单条样本里 ANALYSIS ≈ 80 token × 0.3 = 24，
  STATUS+SUBGOAL event_name ≈ 6 token × 1.0 = 6，结构字面约十几个 token × 1.0，
  tail/EOS 若进入 context 则再约 1 × 1.0。结构性信号合计约 20-21，
  ANALYSIS 占 ~53-55%，结构占 ~45-47%。
- 实测发现 ANALYSIS 段还在漂移 → ANALYSIS 权重 0.3 → 0.5；过拟合 teacher 措辞 → 0.3 → 0.1。

---

## 6. 训练超参（与 v1 对照）

| 超参 | v1 | v2 | 说明 |
|---|---|---|---|
| LoRA rank / alpha | 16 / 32 | 16 / 32 | 不变 |
| lr | 5e-5 | **3e-5** | v2 监督信号 token 数 × 7.5 左右（v1 ~6 → v2 ~44-45，含 ANALYSIS body + 结构字面 + 可选 tail/EOS，2026-06-02 修订；旧版 ~30），lr 同步下调避免过冲 |
| num_train_epochs | 2 | 2 | 不变（先看 v1 step 上限管不管用） |
| weight_decay | 0.05 | 0.05 | 不变 |
| lora_dropout | 0.1 | 0.1 | 不变 |
| per_device_bs × grad_acc | 2 × 2 | 2 × 2 | 不变（DDP 8 卡等效 bs=32） |
| max_length | 3072 | **3584** | ANALYSIS 段比 v1 长（占位 7 token → teacher 30+ token），适度放宽 |
| loss_scale | `sft_v1_analysis_mask` | `sft_v2_analysis_supervised` | 见 §5 |
| save / eval strategy | epoch + best on val/loss | epoch + best on val/loss | 不变 |

预计 total step ≈ 900（与 v1 同），但 v2 每个 step 的有效梯度来自 ~44-45 个 token（含 ANALYSIS
body weight=0.3 + 结构字面 weight=1.0 + 可选 tail/EOS weight=1.0，2026-06-02 修订；旧版 ~30）而不是
v1 的 ~6 个，**真实有效优化量 ≈ v1 的 7 倍**。这也是为什么 v2 第一版不上 KL — 监督信号本身
已经回到健康量级。

---

## 7. 评估

完全复用 `eval_sft_v1.py` 与 `probe_sft_v1.py`。它们按 `STATUS:` / `SUBGOAL:` 行 grep 解析
预测，ANALYSIS 变长不影响指标。

**v2 新增观察项**（不进 metrics.json，靠 probe overview.md / case-level 人工 review）：

- ANALYSIS 段是否依然在"先看图 → 再分析"三步式上？（teacher 灌进去后再被 student 学得是否走样）
- ANALYSIS 与 STATUS / SUBGOAL 是否一致？（出现"ANALYSIS 描述减速，STATUS=initial"是矛盾信号）
- 不同 ckpt 间 ANALYSIS 风格漂移是否可控？（如果某个 ckpt ANALYSIS 开始复读 teacher 高频短语，
  说明过拟合 teacher 风格，参考 §5 把权重从 0.3 调到 0.1）

通过条件（与 v1 §8 一致）：

| 指标 | 目标 | 优先级 |
|---|---|---|
| `keep_accuracy` | ≥ 0.95 | 高 |
| `advance_accuracy` | ≥ 0.60 | 中 |
| `early_advance_rate` | ≤ 0.05 | 最高 |
| `anchor12_sanity.passed` | = true | 必须 |
| **新增**：probe 中 ANALYSIS 段无复读 / 无截断 | 必须 | 必须（v1 翻车点） |

---

## 8. 失败回退路径

| 风险 | 触发条件 | 回退 |
|---|---|---|
| teacher 大批量输出"Observations recorded." 兜底 | `teacher_meta.fallback=true` 比例 > 5% | 检查 base Qwen 是否在 PRIVILEGED prompt 上 OOM 截断；调小 max_new_tokens，或把 §3.1 system prompt 缩短 |
| teacher 输出在 STATUS=keep 与 STATUS=advance 上风格一致（学不出区分） | probe 看 keep / advance 两类的 ANALYSIS 措辞几乎相同 | base 模型分辨力不足；考虑用 v1 LoRA 早期 ckpt（ANALYSIS 还没崩的那个 epoch）做 teacher |
| v2 训练 loss 不降 | 训 50 step 后 loss 还在 4+ | plugin regex 漂移或 max_length 截掉 ANALYSIS 段；先跑 `check_loss_mask_v2.py` 检查 runtime jsonl（见 RUN §3） |
| v2 训练 loss 降但 ANALYSIS 段输出仍复读 | ckpt-100/300/600 都见复读 | 把 §5 ANALYSIS 权重从 0.3 调到 0.5；仍不行进 §9 KL 正则 |
| ANALYSIS 段过拟合 teacher 措辞，每个场景都用同一句套话 | probe 看不同 case 的 ANALYSIS 几乎相同 | 权重 0.3 → 0.1；或在 teacher 阶段把 temperature 从 0 调到 0.3 增加多样性 |
| max_length 3584 仍触发 truncation warning | swift 日志 | 缩短 teacher system prompt 或把 ANALYSIS 后处理截断阈值从 480 char 降到 300 char |

---

## 9. v3 / KL 正则（v2 收官后再启动，本期不实现）

v2 第一版**故意不上 KL**，理由：齐全监督就是治根药，KL 是 belt-and-suspenders。先确认治根
药管用再决定要不要保险，否则会引入工程复杂度（swift 不原生支持自定义 compute_loss + 双
forward + reference model 显存翻倍）。

如果 v2 验证后仍出现 ANALYSIS 漂移（§8 倒数第二行场景），按下面路线进 v3：

1. **舍弃 swift，切 HF Trainer + custom compute_loss**（工程量 2–3 天）。
2. 在 `compute_loss` 里：
   - student forward 拿 `logits_lora`。
   - 冻结 base forward 拿 `logits_base`（no grad、bf16）。
   - 在 ANALYSIS body 或显式指定的结构 token 上算 `KL(softmax(logits_lora) ‖ softmax(logits_base))`；
     当前 v2 完整三段样本不再依赖 mask=0 区域。
   - 总 loss = `weighted_ce + λ * kl_loss`，λ 从 0.01 起步。
3. 显存代价：base forward 占 ~10GB，与 student forward 共享 ViT 输出可省一半 → 单卡 ~50GB。
4. 测试组：v3 train.sh 复用 v2 数据集，不重新生成。

---

## 10. 文件清单

| 文件 | 类型 | 作用 |
|---|---|---|
| `AutoMoT/tools/SFT_V2_PLAN.md` | 新增 | 本文件 |
| `AutoMoT/tools/SFT_V2_RUN.md` | 新增 | 操作手册 |
| `AutoMoT/tools/build_sft_dataset_v1.py` | 改 | 加 `--mode v2`：输出 pending jsonl |
| `AutoMoT/tools/build_sft_dataset_v2_teacher.py` | 新增 | 从 pending jsonl 运行 teacher，生成 runtime/调试用 ANALYSIS GT |
| `AutoMoT/tools/sft_v2_loss_scale_plugin.py` | 新增 | 注册 `sft_v2_analysis_supervised` |
| `AutoMoT/tools/sft_v2_train.sh` | 新增 | v2 训练入口（与 v1 同套 GPU/MASTER_PORT 自动选址） |
| `AutoMoT/tools/eval_sft_v1.py` | 改 | 按 `dataset_version` 字段自动检测，沿用同一份评估逻辑 |
| `AutoMoT/tools/probe_sft_v1.py` | 改 | 同上 |
| `CLAUDE.md` / `AGENTS.md` | 改 | 同步白名单 |

`check_loss_mask_v2.py` 是 v2 专用 token 级 sanity；输入必须是已经 runtime 物化后的
`dataset_version == "v2"` jsonl，不能直接检查 pending 占位数据。
