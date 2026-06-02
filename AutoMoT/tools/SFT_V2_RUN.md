# SFT v2 运行教程 — pending 数据 + 运行时 teacher 真值

> 本文档是 [SFT_V2_PLAN.md](SFT_V2_PLAN.md) 的"操作手册"对照：PLAN 讲设计与决策依据，
> 本 RUN 讲实际怎么跑。
>
> **关键约定**：所有命令默认 **从 `AutoMoT/` 目录执行**（远程默认 cwd），不是从仓库根
> `automot_lead/`。脚本路径写 `tools/...` 不是 `AutoMoT/tools/...`，checkpoint 路径写
> `checkpoints/...` 不是 `AutoMoT/checkpoints/...`。`keyframes_all_scenarios.json`
> 远程固定放在 `/datashare/IOL4SGH/data/data/keyframes_all_scenarios.json`。

> ⚠️ **如果近期改过下列任意一项 prompt / 数据格式 → 必须删旧 pending 重新跑 §1**：
>
> - [`qwen3vl_local/prompt_pipeline.py`](../qwen3vl_local/prompt_pipeline.py) 的 `_SYSTEM_PROMPT`
>   或 `build_user_prompt` / `build_memory_block`（student 端 prompt，影响 messages[0] / messages[1]）；
> - [`build_sft_dataset_v1.py`](build_sft_dataset_v1.py) 的拼装逻辑或采样规则。
>
> 因为 pending jsonl 一旦生成，里面的 `messages[0]['content']` / `messages[1]['content']`
> 就是当时的 prompt 文本快照；[`build_sft_dataset_v2_teacher.py`](build_sft_dataset_v2_teacher.py) 物化 runtime teacher
> **只替换 ANALYSIS 占位**，不会重写 system / user。直接复用旧 pending 训出来的 LoRA
> 会出现"训练时一套 prompt、eval 时另一套 prompt"的分布漂移。
>
> 删法：
>
> ```bash
> rm -rf checkpoints/sft_v2_data_pending
> ```
>
> 然后从 §1 重新跑。`sft_v2_train.sh` 默认 `RUNTIME_TEACHER_REFRESH=1`，下次启动
> 会自动清掉 `runtime_teacher_data/` 重跑 teacher，无需手动删 runtime cache。

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

> 📌 **重生成时机**：见本文顶部 ⚠️ 提醒。改过 `prompt_pipeline.py` 的 `_SYSTEM_PROMPT`、
> `build_user_prompt`、`build_memory_block` 中任意一个 → **必须先** `rm -rf checkpoints/sft_v2_data_pending`
> 再跑下面的命令。pending jsonl 是"prompt 文本快照"，teacher 物化时不重写 system/user。

```bash
python tools/build_sft_dataset_v1.py \
    --mode v2 \
    --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
    --data-root /datashare/IOL4SGH/data/data \
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
  `checkpoints/sft_v2_lora/runtime_teacher_data/`，再把这份 runtime jsonl 交给 ms-swift。
  默认 `RUNTIME_TEACHER_REFRESH=1` → 每次启动都清旧 cache 重跑，与当前 prompt / keyframes
  保持一致；只有续跑同一份 pending 的中断任务时才设 `RUNTIME_TEACHER_REFRESH=0`。

说明：ms-swift 训练入口需要 jsonl 文件，所以这里的"实时"是训练启动时临时物化 teacher
真值，不是每个 batch 在线调用 teacher；pending 源数据不会被回写。

### 2.1 训练前 teacher 预览（不写训练集，推荐先跑）

如果 keyframes / prompt / 数据采样之后会改，先从 pending jsonl 里抽少量样本现场跑
teacher，打开网页看 ANALYSIS 是否符合预期。这个步骤只写 inspect 目录，不会把
teacher ANALYSIS 回写到训练 jsonl。

如果只想生成一个很小的预览数据集，先跑：

```bash
python tools/build_sft_dataset_v1.py --mode v2 --dry-run \
    --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
    --data-root /datashare/IOL4SGH/data/data \
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

通常不需要手动跑这一步。只有你想提前做 plugin 静态 sanity、复用同一份 teacher 输出，
或排查 fallback 比例时才需要：

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

**通过条件**（必须全部满足，**2026-06-02 修订**：结构字面改为 W1.0；tail/EOS 若进入 plugin context 也必须 W1.0，详见 PROJECT_CONTEXT.md §18.5）：

- token 表里 ANALYSIS 正文 token 为 `[W0.3]`
- token 表里起手 `ANALYSIS:` 字面 token 为 **`[W1.0]`**（不能 mask，否则模型可能丢起手格式）
- token 表里 STATUS/SUBGOAL 事件名 token 为 `[W1.0]`
- token 表里 `\nSTATUS:` / `\nSUBGOAL:` 段切换字面 token 为 **`[W1.0]`**（旧版是 W0.0，修订后必须 W1.0）
- 若原始 assistant content 自带 tail，token 表里 tail token 为 **`[W1.0]`**；常见 jsonl 不含 EOS 时，
  脚本会额外打印 `synthetic tail/EOS sanity`，合成 `\n<eos>` tail 并确认 plugin 会把 tail/EOS 标为 **`[W1.0]`**。
  这只验证 plugin 行为；真实训练 EOS 是否进入 loss 由 ms-swift chat template / runtime context 决定
- 完整三段样本里不应再有 `[W0.0]` token；若出现，优先检查是否有预期外前缀/空白
- plugin 输出中，`ANALYSIS:` / `ANALYSIS body`、两段 `event_name`、`\nSTATUS:` / `\nSUBGOAL:` 都 `in_loss=True`；
  `synthetic tail/EOS sanity` 的 plugin check 中 tail/EOS `in_loss=True`
- plugin 分段数量为 6 或 7（有 tail 时为 7）

**预期切片形状**（2026-06-02 修订后；示例）：

```
w=1.00: 'ANALYSIS: '          # 起手结构字面：必须 1.0
w=0.30: 'I see a foggy tunnel ... advancing to hazard_detect.'
w=1.00: '\nSTATUS: '          # 段切换字面：必须 1.0
w=1.00: 'hazard_detect'
w=1.00: '\nSUBGOAL: '         # 段切换字面：必须 1.0
w=1.00: 'max_brake_or_min_gap'
w=1.00: '\n<|im_end|>'        # synthetic tail/EOS sanity：若 tail/EOS 进入 context，必须 1.0
```

---

### 3.5 重训前命中率门槛（**重要：不达标必查 teacher**）

`check_loss_mask_v2.py` 跑单条样本 OK 不够——必须确认 `_FULL_PATTERN`
在 runtime teacher 数据**整体**命中率 ≥ 95%。命中率低于 95% 说明 teacher 物化
后 ANALYSIS body 里混进了多行或异常字符，plugin 走 fallback `_split_analysis_only`
（fallback 下 STATUS/SUBGOAL 字面会被 weight=1.0 训练但段结构不对），训练会跑偏。

```bash
# 在 AutoMoT/ 根目录运行。importlib 路径加载 plugin，避免 tools/ 不是 Python 包导致
# `from tools.sft_v2_loss_scale_plugin import ...` 失败；同时绕开 ms-swift 重型导入
# （plugin 顶层 from swift...import LossScale 会拖一整套 ms-swift，单纯查 regex 不值）。
python - <<'PY'
import importlib.util, json, pathlib

plugin_path = pathlib.Path("tools/sft_v2_loss_scale_plugin.py").resolve()
spec = importlib.util.spec_from_file_location("sft_v2_loss_scale_plugin", plugin_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
_FULL_PATTERN = mod._FULL_PATTERN

path = "checkpoints/sft_v2_lora/runtime_teacher_data/train.jsonl"
hit = total = 0
samples_fail = []
with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        total += 1
        msgs = json.loads(line)["messages"]
        if _FULL_PATTERN.search(msgs[-1]["content"]):
            hit += 1
        elif len(samples_fail) < 5:
            samples_fail.append((i, msgs[-1]["content"][:200]))
print(f"hit_rate = {hit}/{total} = {hit/total:.2%}")
for i, snippet in samples_fail:
    print(f"--- sample {i} fail snippet ---\n{snippet}\n")
PY
```

> 备注：上面 python heredoc 仍然会触发 `import sft_v2_loss_scale_plugin` 里
> `from swift.plugin.loss_scale.loss_scale import LossScale` 这条顶层 import；
> 在远程训练机上 swift 是装好的，没问题。如果在没装 ms-swift 的环境跑（如本地
> mac dev box），把上面 plugin 加载替换成 inline 直接复制 regex 串：
>
> ```python
> import re
> _FULL_PATTERN = re.compile(
>     r"ANALYSIS:[ \t]*"
>     r"(?P<analysis>[^\n]*?)"
>     r"\s*\nSTATUS:[ \t]*"
>     r"(?P<status>\S[^\n]*?)"
>     r"\s*\nSUBGOAL:[ \t]*"
>     r"(?P<subgoal>\S[^\n]*)",
>     flags=re.DOTALL,
> )
> ```

**判读门槛**：

| 命中率 | 判读 | 处理 |
|---|---|---|
| ≥ 95% | ✅ teacher 物化分布健康 | 进 §4 跑 dynamic check |
| 90–95% | ⚠️ teacher 有少量异常样本（多行 / 异常长） | 看打印的 fail snippet；若是多行 ANALYSIS → 查 teacher postprocess `_truncate_at_sentence_boundary` 是否生效 |
| < 90% | ❌ teacher prompt / postprocess 已经垮了 | **不要训**，回 §2 查 teacher：抽样 `inspect_teacher_outputs.py --live`，重点看 ANALYSIS body 平均长度、是否多行、是否退化到 fallback `Observations recorded.`；调好 teacher prompt 长度约束后重新物化 |

### 3.6 inspect_teacher_outputs 抽样必看项（重训前一次性确认）

teacher prompt 加了 word-count target（"40-70 words"）+ postprocess 收紧上下限
（80-420 字符），重训前必须用 inspect 工具抽 20–50 条样本人眼过一遍，
确认 teacher 真的进入了新分布：

```bash
python tools/inspect_teacher_outputs.py \
    --jsonl checkpoints/sft_v2_data_pending/train.jsonl \
    --save-root /tmp/inspect_pre_retrain \
    --num-per-scenario 3 --seed 42 \
    --live --serve --port 0 \
    --model-dir checkpoints/Qwen3-VL-4B-Instruct
```

**抽样必看项**：

- 大多数 ANALYSIS body 长度落在 200–400 字符（≈ 30–70 词）。<150 字符或 >450 字符的样本应该是少数（< 5%）。
- ANALYSIS body 是单行（没有内嵌 `\n`），且以句号 / 问号 / 感叹号收尾（不在 word 中间被截）。
- ANALYSIS body 不出现 `STATUS:` / `SUBGOAL:` 字面（teacher 自己生成结构会污染 plugin regex 切片）。
- `truncated_sentence=True` 比例 < 30%——比例过高说明 teacher 普遍超长，需要再压缩 prompt 长度约束。
- `fallback=true` 比例 < 5%——比例过高说明 teacher 普遍过短或垮，需要看 prompt 是不是约束太紧。

抽样结果不满足以上任何一条 → 回去调 teacher prompt 或 postprocess 上下限，**不要直接训**。

---

## 4. 动态 sanity：跑 2 step 看真实 loss（**GPU**，约 1–2 分钟）

```bash
bash tools/sft_v2_train.sh check
```

与 v1 一样，2 step、不保存 ckpt、不跑 val。check 模式会自动从 pending 数据物化最多
32 条 runtime teacher 样本到 `checkpoints/sft_v2_lora/runtime_teacher_data/`（默认
`RUNTIME_TEACHER_REFRESH=1` 会先清旧 cache），然后跑 2 step。pending 数据集不会被改写。

**预期 loss 数值**（健康范围，**2026-06-02 修订**：plugin 升级后结构字面进 loss，effective 监督 token 从 v2.0 的 ~30 升到 ~44-45，loss 数值会略偏高；EOS 是否额外计入以 ms-swift runtime context 为准）：

```
{'loss': 3~10, 'grad_norm': ..., 'learning_rate': ..., 'epoch': 0.0x}
```

说明：
- v2 会监督 ANALYSIS 正文（权重 0.3）+ 起手/段切换结构字面（权重 1.0，2026-06-02 修订）；tail/EOS 若由 ms-swift 放进 plugin context 也按 1.0，所以 loss 统计口径与 v1 不同
- 重点看"mask 是否生效 + loss 是否有限非 NaN"，不是盯绝对值
- v2.0 旧版（commit ef0eb19 之前）健康范围曾是 3~8；修订后 effective 监督 token 增加约 50%，区间上沿略上抬

判读：

| 现象 | 判读 | 处理 |
|---|---|---|
| `check_loss_mask_v2.py` 通过 + check loss 有限且非 NaN | ✅ 训练侧权重大方向正常 | 进 §5 正式训练 |
| loss < 1 或 `grad_norm=0` | ❌ 可能事件名 token 也被 mask 掉了（全 0 权重） | 先看 §3 的 `w1` token 数是否 ≥ 2 |
| loss 在 8~10 区间 | ✅ v2 修订后正常（结构字面进 loss，tail/EOS 可能也进 loss，抬高了数值），**不是异常** | 直接进 §5 |
| loss > 14 | ⚠️ 可能 teacher ANALYSIS 异常长 / regex 大面积 fallback | 先看 §3.5 `_FULL_PATTERN` 命中率；再查 teacher 可视化 §2.6 |
| check 模式仍保存了 checkpoint | ❌ check 不该落盘 | 拉最新 `tools/sft_v2_train.sh`，确认 `--save_strategy no` |

---

## 5. 正式训练（**8×H20 DDP**）

```bash
bash tools/sft_v2_train.sh ddp
```

GPU / 端口 / DDP rendezvous 行为与 v1 完全一致（自动选最空闲卡、自动找空闲 MASTER_PORT、
NCCL_P2P_LEVEL=NVL 等）。所有 v1 的 `DDP_GPU_COUNT` / `SFT_RESPECT_*` 环境变量在 v2 同名。

默认 `ddp` 按 8 卡跑。如果机器只想用 N 张卡，不要手动写死卡号，直接用
`DDP_GPU_COUNT=N`，脚本会自动挑 N 张最空闲 GPU，让后面的 ms-swift 训练使用同一组卡：

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
默认 `RUNTIME_TEACHER_REFRESH=1` → 每次启动都会刷新 runtime cache，避免 keyframes /
prompt 改了之后复用旧 teacher 文本。如果上次 teacher 物化中断、想续跑同一份 pending，可显式：

```bash
RUNTIME_TEACHER_REFRESH=0 bash tools/sft_v2_train.sh ddp
```

teacher 物化本身天然支持断点续跑（rank 分片 + 每 100 条 flush），所以 REFRESH=0 时
下一次启动会基于已落盘的 rank 分片续完。

**预期**：
- teacher runtime 物化：8 卡约 100 分钟，单卡小样本 check 约 1–2 分钟；
- LoRA 训练：8 卡总 step ≈ 900（与 v1 同）；4 卡 ≈ 1800；
- 每 `SAVE_STEPS`（默认 10000）步保存一次 LoRA adapter 到 `checkpoints/sft_v2_lora/v*/checkpoint-XXX/`，
  `SAVE_TOTAL_LIMIT`（默认 3）控制保留数 → 等效最近 30k 步；epoch 边界不再单独 save，
  但训练结束时最后一份 ckpt ≈ 最后一个 epoch 末快照，best 仍由 `load_best_model_at_end`
  按 eval/loss 装回 `OUTPUT_DIR` 顶层 `adapter_model.*`。
  `SAVE_STEPS` / `SAVE_TOTAL_LIMIT` 是 env，按需 `SAVE_STEPS=200 bash tools/sft_v2_train.sh ddp`
  临时缩小间隔；
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
