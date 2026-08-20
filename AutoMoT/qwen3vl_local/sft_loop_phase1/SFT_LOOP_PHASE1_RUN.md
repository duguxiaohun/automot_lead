# SFT Loop Phase1 Run

这个子包训练/测试第一轮无 memory 的四个视觉事实问题：

```text
HIGHWAY: YES|NO
STATIC_OBSTACLE: YES|NO
VULNERABLE: YES|NO
TRAFFIC_LIGHT_ABNORMAL: YES|NO
```

所有命令都从 `AutoMoT/` 目录运行。默认只读本地
`checkpoints/Qwen3-VL-4B-Instruct`，不会联网下载。

正式输出约定：

- 数据索引写到 `checkpoints/sft_loop_phase1_data/`。
- 原始数据索引始终保存同一组时序四帧；RGB 输入模式是运行时合同，不改索引也不重构数据集。
  `4rgb`（默认）送入第 1/2/3/4 帧；`2rgb_endpoints` 只送入首尾两帧，即第 1 帧和第 4 帧、原始索引 `[0, 3]`。
- 正式训练分别写到 `checkpoints/sft_loop_phase1_runs/run_static_obstacle_final_4rgb/` 或
  `checkpoints/sft_loop_phase1_runs/run_static_obstacle_final_2rgb_endpoints/`；`latest` 只指向本次实际训练的模式。
- eval 默认按 base/LoRA/audit 和 RGB 模式自动写到
  `checkpoints/sft_loop_phase1_eval/*_static_obstacle_final_{4rgb|2rgb_endpoints}/<timestamp>/`；
  默认带时间戳，不会覆盖旧结果。
- `--output-dir` 只保留给临时 debug，不是正式流程参数。

## 1. 构建数据

```bash
python qwen3vl_local/sft_loop_phase1/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --answer-table keyframe_filter/collection_output/phase1_four_question_audit/phase1_four_question_answer_table.json \
  --output-dir checkpoints/sft_loop_phase1_data \
  --test-ratio 0.10 \
  --val-ratio 0.05
```

同一个 `frame_index.jsonl` 可以反复用于同一任务语义下的 base Qwen 测试、LoRA 训练和
LoRA 复测。只有下面情况需要重建：

- `phase1_four_question_answer_table.json` 更新了；
- `collection_output/*_result.json` 的 RS/EVENT 标注更新了；
- 异常 route 过滤逻辑或 `lead_data` 内容变了；
- 想改变 `--split-seed` / `--test-ratio` / `--val-ratio`；
- `build_dataset.py` 的字段 schema 改了。
- 四问的语义变了，例如本轮从混合 `OBSTACLE` 拆成只问 `STATIC_OBSTACLE`。

输出：

- `checkpoints/sft_loop_phase1_data/frame_index.jsonl`
- `checkpoints/sft_loop_phase1_data/manifest.json`

脚本按 route 稳定 split，默认 `test_ratio=0.10`、`val_ratio=0.05`，并剔除
`noScenarios`、异常时长 route 和 data-missing route。train / val / test 按 route
互斥，避免同一路线相邻帧泄漏。训练中定期验证依赖 `val` split；如果你之前是在
`val_ratio=0.00` 的旧默认下构建的数据，需要按上面的命令重建一次。

### 是否需要重构数据集

只有仍在使用“混合 `OBSTACLE`”旧 schema 的索引时才需要重构。当前第一轮训练真值是
`STATIC_OBSTACLE`（`primary EVENT == U-E2`），旧索引和旧 LoRA 与这个语义不兼容；重构后的索引
会保留 `val` split，供训练中定期验证。已有当前 `STATIC_OBSTACLE` 四帧索引时无需重构。

重构只会重写 `checkpoints/sft_loop_phase1_data/frame_index.jsonl` 和 `manifest.json`，
不会改 RGB，也不会改人工四问答案表。重构后建议先确认 manifest 里有三类 split：

```bash
python - <<'PY'
import json
m=json.load(open("checkpoints/sft_loop_phase1_data/manifest.json"))
print(m["counts"])
print(m["route_counts"])
PY
```

如果能看到 `frames/train`、`frames/val`、`frames/test`，就可以开始训练。后续只要
answer table、RS/EVENT 标注、异常 route 过滤或 split 参数不变，就不需要再次重构。

### 四帧与首尾两帧不需要重构

`history_rgb_paths` 在现有索引中固定为按时间排序的四张 RGB。`4rgb` 和 `2rgb_endpoints` 只是在训练/评测
运行时选择 `[0,1,2,3]` 或 `[0,3]`；八个 `问题 x YES/NO` 均衡桶、route split 和标签完全不变。
因此切换这两个模式时**不要**运行 `build_dataset.py`。它们是两种不同视觉输入分布，必须各自训练
LoRA，不能把 4RGB LoRA 当作 2RGB LoRA 正式评测或部署。

## 2. 先测原始 Qwen

这是 prompt 迭代最重要的一步。默认使用和训练/部署完全一致的 production prompt，模型只输出
四个 YES/NO。只有显式传 `--audit-prompt` 才会要求 `EVIDENCE_*`；parser 仍只解析四个
YES/NO。

最近一次 RGB 错例复核见
[`SFT_LOOP_PHASE1_RGB_ERROR_AUDIT_20260810.md`](SFT_LOOP_PHASE1_RGB_ERROR_AUDIT_20260810.md)。
已完成 base/LoRA、audit/production 和提示词边界的实验矩阵见
[`SFT_LOOP_PHASE1_EVAL_EXPERIMENT_MATRIX_20260811.md`](SFT_LOOP_PHASE1_EVAL_EXPERIMENT_MATRIX_20260811.md)。
它区分了模型确实漏看了的视觉证据，和仅凭当前四帧 RGB 无法回答的 scenario/RS/EVENT
标签。修 prompt 后必须先在同一份固定 test index 上复测；不要把“故障场景但故障尚未出现”
的正标签误当作模型应该从不可见 RGB 猜出的知识。

### Prompt 与 LoRA 的正确比较顺序

提示词本身也是模型输入。已有 LoRA 如果是在旧提示词上训练，它对新提示词的结果同时混合了
视觉能力、答案格式能力和输入措辞分布漂移；这类评测只用于检查旧 adapter 是否兼容，**不能**
单独决定新提示词好坏。

1. 先用 base Qwen 的 `--audit-prompt` 跑固定 index，观察它是否能在 RGB 中找出提示词要求的
   道路、目标物和信号证据。这是 prompt 的视觉诊断，不是 production 指标。
2. 再用 production prompt 跑 base Qwen，确认四行格式、解析和没有训练时的保守下限。未训练的
   base 可能几乎全答 `NO`，不能据此否定一个需要 LoRA 才能学会输出 YES 的视觉规则。
3. 旧 LoRA 是混合动静态 `OBSTACLE` 任务，schema 会被拒绝加载；它只能保留作历史参考，不能
   测新静态问题。
4. 选定 prompt 后，使用本节重建的 `frame_index.jsonl` 训练新的 LoRA；`train.py` 会在训练时
   动态调用 `build_phase1_prompt(audit=False)`。在新的静态 schema 内只改提示词时无需再次重构。
5. 用新 LoRA 和同一 production prompt 做正式 1:1 四任务评测；随后用同一 adapter 的 audit
   run 回查错例 RGB，才判断训练是否真正提高 answer-only loop 的视觉性能。

当前静态障碍仍以低成本代理 `primary_event == U-E2` 监督：不新增逐帧人工标签，也不把其它 EVENT
加入静态正例。因此 U-E2 的事件前后重叠、已驶过或严重遮挡帧会保留不可消除的视觉标签噪声；正式分析
必须把它与模型漏看真实施工物/固定占道物分开。2026-08-12 的逐帧审计已据 RGB 补强三项判据：
临时箭头拖车/导流施工设施占道为 YES；短历史中自身仍在切入/转向/横穿/前进的车辆为 NO；远处但
连续可辨、固定在 ego 车道内的橙黄封道设施为 YES。详见
`SFT_LOOP_PHASE1_STATIC_OBSTACLE_RGB_AUDIT_20260811.md` 和
`SFT_LOOP_PHASE1_LORA_RGB_ERROR_AUDIT_20260812.md`。经过同一固定 1:1 test index 的 v2/v3
prompt-aligned LoRA 对比，v2 是最终 production prompt：`STATIC_OBSTACLE` F1 `0.7500` 高于 v3 的
`0.7451`，且 `TRAFFIC_LIGHT_ABNORMAL` F1 `0.8571` 高于 v3 的 `0.8364`。最终 production prompt SHA
为 `827b59181b391657c7bd3c97640241902d8905f4cada90861ebb37d931bb633a`；v3 的
`334388...` adapter 不作为部署候选。这次只切回已验证的 production prompt，数据索引不需重建。

最终 prompt 已冻结。下面命令复用现有 `frame_index.jsonl`，不要重构数据集；模型、索引、1:1
采样、时间戳输出都由脚本默认处理。

```bash
# 先测固定 final prompt 下的四帧 base Qwen；它是下限，不是 LoRA 成绩。
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase1/eval.py \
  --history-rgb-mode 4rgb

# 单独测首尾两帧的 base Qwen。这里只能在 base eval 显式选模式。
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase1/eval.py \
  --history-rgb-mode 2rgb_endpoints

# 训练四帧 final LoRA；无需 RUN_TAG。
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase1/train.sh ddp

# 训练首尾两帧 final LoRA；只送第 1 帧和第 4 帧。
HISTORY_RGB_MODE=2rgb_endpoints \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase1/train.sh ddp
```

正式 F1/TP/FP/FN/TN 只比较 **prompt-aligned LoRA** 的 `prompt_mode=production` 结果。
`--audit-prompt` 是同一固定 case index 的第二个诊断 run，用来保存模型可见证据和错例 RGB；
它多了一段用户输入，不能和 production 的指标直接横比。未训练 base 的 production run 是
格式/下限检查，也不能代替 prompt-aligned LoRA 的正式结果。每个 `summary.md` 和
`metrics.json` 都会记录 prompt mode、`history_rgb_mode`、实际图片数量和原始帧索引。
新训练 adapter 还会记录 production prompt 的 SHA-256；eval 会同时写当前内容指纹和
`adapter_prompt_matches_current_production`。旧 adapter 没有该字段时显示 `unknown`，仍可
用于兼容性诊断，但不能冒充 prompt-aligned 对照。

本轮静态障碍 prompt 的 4 卡 base 评测应单独留存：

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase1/eval.py \
  --index checkpoints/sft_loop_phase1_data/frame_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_loop_phase1_eval/base_static_obstacle_4gpu \
  --cases-per-bin 64 \
  --timestamp-output
```

1 卡测原始 Qwen：

```bash
GPU_IDS=0 \
python qwen3vl_local/sft_loop_phase1/eval.py \
  --index checkpoints/sft_loop_phase1_data/frame_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_loop_phase1_eval/base_static_obstacle \
  --cases-per-bin 64
```

评估集按四个主任务分别做 YES/NO 1:1。也就是说 HIGHWAY 模块只保证
`HIGHWAY:YES / HIGHWAY:NO` 均衡，STATIC_OBSTACLE 模块只保证
`STATIC_OBSTACLE:YES / STATIC_OBSTACLE:NO` 均衡，以此类推：

```text
HIGHWAY:YES / HIGHWAY:NO
STATIC_OBSTACLE:YES / STATIC_OBSTACLE:NO
VULNERABLE:YES / VULNERABLE:NO
TRAFFIC_LIGHT_ABNORMAL:YES / TRAFFIC_LIGHT_ABNORMAL:NO
```

每个样本仍回答全部四个问题；`focus_question` / `task` 只用于采样和统计，不进入 prompt。
构建时 `manifest.json.focus_bin_availability` 必须显示 train/test 的八桶均为正数（开启 val 时 val 也必须如此）；
否则 `build_dataset.py` 会失败且不替换旧索引。训练/评测随后断言八个最终桶严格同数，任何缺桶或计数不等都会失败，
不会静默降级为不均衡测试。
例如 HIGHWAY 模块的主问题是“是否高速”，该模块会记录 HIGHWAY 的
TP/FP/FN/TN、precision/recall/F1，同时顺带记录这批 HIGHWAY 1:1 样本上
STATIC_OBSTACLE / VULNERABLE / TRAFFIC_LIGHT_ABNORMAL 的结果；这些副问题在该模块里不要求
YES/NO 均衡。其它三个模块同理。

输出：

- `metrics.json`：总结果 + `task_reports.{HIGHWAY,STATIC_OBSTACLE,VULNERABLE,TRAFFIC_LIGHT_ABNORMAL}`。
  每个 task report 都有主问题 1:1 balance、TP/FP/FN/TN、precision/recall/F1 和副问题统计。
- `cases.jsonl`：所有 case 放在一起，含 prompt、`history_rgb_paths_all4`、实际输入的
  `history_rgb_paths_used`、GT、parsed、raw output、ok_by_key。
- `task_cases/<TASK>/cases.jsonl`：按主任务拆开的 case 记录，方便只看某一类问题。
- `summary.md`：四个主任务模块的简短 Markdown 报告。
- `error_cases/<TASK>/case_*/rgb/`：主问题答错时模型实际看见的 RGB；文件名保留原始四帧索引，
  所以 `2rgb_endpoints` 只会有 `history_source_0_*` 和 `history_source_3_*`。
- `rgb_cases/<TASK>/case_*/rgb/`：只有显式加 `--save-all-rgb` 时才复制所有受评 RGB；默认不复制全量 RGB，
  避免输出太大。

你后续给我分析时，优先打包这些轻量文件即可，不需要传全量 RGB：

```bash
tar -czf /tmp/base_static_obstacle_records.tgz \
  checkpoints/sft_loop_phase1_eval/base_static_obstacle/<timestamp>/metrics.json \
  checkpoints/sft_loop_phase1_eval/base_static_obstacle/<timestamp>/summary.md \
  checkpoints/sft_loop_phase1_eval/base_static_obstacle/<timestamp>/cases.jsonl \
  checkpoints/sft_loop_phase1_eval/base_static_obstacle/<timestamp>/task_cases
```

如果某个主任务表现很差，再只补充该主任务的少量 RGB 错例，例如：

```bash
tar -czf /tmp/base_highway_error_rgb_sample.tgz \
  checkpoints/sft_loop_phase1_eval/base_static_obstacle/<timestamp>/error_cases/HIGHWAY
```

不建议第一次就传 `rgb_cases/`，除非你专门加了 `--save-all-rgb` 并且只想给我一个很小 smoke。

如果只想跑很小 smoke：

```bash
GPU_IDS=0 \
python qwen3vl_local/sft_loop_phase1/eval.py \
  --index checkpoints/sft_loop_phase1_data/frame_index.jsonl \
  --output-dir checkpoints/sft_loop_phase1_eval/smoke_base \
  --cases-per-bin 2 \
  --max-frames 200 \
  --overwrite
```

2 卡测原始 Qwen：

```bash
GPU_IDS=0,1 torchrun --nproc_per_node=2 \
  qwen3vl_local/sft_loop_phase1/eval.py \
  --index checkpoints/sft_loop_phase1_data/frame_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_loop_phase1_eval/base_static_obstacle_2gpu \
  --cases-per-bin 64
```

4 卡测原始 Qwen：

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase1/eval.py \
  --index checkpoints/sft_loop_phase1_data/frame_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_loop_phase1_eval/base_static_obstacle_4gpu \
  --cases-per-bin 64
```

根据 base 错例修过 prompt 后的复测，也放在 `checkpoints/` 下另起目录：

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase1/eval.py \
  --index checkpoints/sft_loop_phase1_data/frame_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_loop_phase1_eval/base_static_obstacle_refined_4gpu \
  --cases-per-bin 64 \
  --overwrite
```

多卡 eval 会写 `cases_rank0.jsonl`、`cases_rank1.jsonl` ...，rank0 汇总
`metrics.json` / `summary.md`；同时 `task_cases/<TASK>/` 下也会按 rank 拆分。

需要分析错因时，在上面**同一个模型和同一个 index**的命令额外加
`--audit-prompt`，但输出目录必须另起名（例如
`checkpoints/sft_loop_phase1_eval/base_static_obstacle_audit_4gpu`）。诊断 run 的
`EVIDENCE_*` 可交给我配合 `error_cases/<TASK>/` RGB 看，正式结果仍以 production run 为准。

## 3. 训练 LoRA

默认 LoRA 只挂语言侧，视觉侧保持 frozen。训练采样和测试采样保持同一个口径：
四个主问题各自 `YES:NO = 1:1`，也就是八个桶
`HIGHWAY:YES/NO`、`STATIC_OBSTACLE:YES/NO`、`VULNERABLE:YES/NO`、
`TRAFFIC_LIGHT_ABNORMAL:YES/NO` 每轮取同样数量。

RGB 输入模式是 adapter 的一部分：`train.sh` 默认 `HISTORY_RGB_MODE=4rgb`，输出 `4rgb` 目录；
显式 `HISTORY_RGB_MODE=2rgb_endpoints` 只送第 1/4 帧，输出 `2rgb_endpoints` 目录。训练中 val 也会使用同一
选择，因此 TensorBoard 曲线与正式 eval 的视觉输入一致。保存的
`sft_loop_phase1_adapter_config.json` 会固化 `history_rgb_mode`、`history_rgb_count` 和
`history_rgb_selected_indices`。

现在默认更接近正式训练，不再是 1000 step 快跑：

```text
FOCUS_BALANCE_COUNT=512  # 每个桶每个 epoch 512 条，全局每 epoch 4096 个 work item
NUM_EPOCHS=3             # MAX_STEPS=0 时完整跑 3 个 epoch
EVAL_STEPS=100           # 每 100 个本 rank train step 做一次轻量 val
EVAL_BALANCE_COUNT=16    # val 每个桶 16 条，全局 128 条，用来看 loss/acc 曲线
SAVE_STEPS=500           # 额外保存 checkpoint-<step>，final 始终保存
```

这里的训练中验证同时覆盖两种口径：每 `100` step 的 teacher-forced `val/loss`、`val/value_token_acc`、`val/format_token_acc` 和四问 token accuracy；每 `1,000` step 的 rank0 自由生成小验证集会记录 `val_generation/format_valid_rate`、`val_generation/exact_accuracy` 与四问 focus accuracy，并将原始输出、GT、解析结果和 RGB 原路径追加到运行目录的 `generation_val_cases.jsonl`。YES/NO 的 loss 权重固定 `1.0`，字段名、冒号、换行和 assistant 结束符只用 `0.25` 的格式权重；这只训练输出语法，绝不让四问互相约束。`train.sh check` 为快速链路检查，会关闭两种验证；正式训练不要关闭自由生成验证。
完整自由生成的 TP/FP/FN/TN、precision、recall、F1 仍然用第 4 节的 `eval.py` 跑。

单卡：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_loop_phase1/train.sh single
```

2 卡 DDP：

```bash
GPU_IDS=0,1 bash qwen3vl_local/sft_loop_phase1/train.sh ddp
```

4 卡 DDP：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase1/train.sh ddp
```

最终 prompt 已固定，正式重训不再需要 `RUN_TAG` 或版本名：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase1/train.sh ddp
```

快速 check / smoke：

1 卡：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_loop_phase1/train.sh check
```

2 卡：

```bash
MAX_STEPS=2 \
FOCUS_BALANCE_COUNT=2 \
EVAL_STEPS=0 \
SAVE_STEPS=0 \
GPU_IDS=0,1 bash qwen3vl_local/sft_loop_phase1/train.sh ddp
```

4 卡：

```bash
MAX_STEPS=2 \
FOCUS_BALANCE_COUNT=2 \
EVAL_STEPS=0 \
SAVE_STEPS=0 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase1/train.sh ddp
```

常用覆盖：

1 卡：

```bash
NUM_EPOCHS=5 \
MAX_STEPS=0 \
FOCUS_BALANCE_COUNT=512 \
EVAL_STEPS=100 \
EVAL_BALANCE_COUNT=16 \
LR=1e-5 \
LORA_RANK=16 \
LORA_VISION_SCOPE=off \
GPU_IDS=0 bash qwen3vl_local/sft_loop_phase1/train.sh single
```

2 卡：

```bash
NUM_EPOCHS=5 \
MAX_STEPS=0 \
FOCUS_BALANCE_COUNT=512 \
EVAL_STEPS=100 \
EVAL_BALANCE_COUNT=16 \
LR=1e-5 \
LORA_RANK=16 \
LORA_VISION_SCOPE=off \
GPU_IDS=0,1 bash qwen3vl_local/sft_loop_phase1/train.sh ddp
```

4 卡：

```bash
NUM_EPOCHS=5 \
MAX_STEPS=0 \
FOCUS_BALANCE_COUNT=512 \
EVAL_STEPS=100 \
EVAL_BALANCE_COUNT=16 \
LR=1e-5 \
LORA_RANK=16 \
LORA_VISION_SCOPE=off \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_loop_phase1/train.sh ddp
```

如果你想严格指定总训练步数，可以设置 `MAX_STEPS>0`；这会覆盖 `NUM_EPOCHS` 的总步数，
适合快速 ablation。正式训练更建议保持 `MAX_STEPS=0`，用 `NUM_EPOCHS` 控制，让模型完整多看几轮
均衡后的训练数据。

训练采样同样按四个主任务各自 YES/NO 1:1。每个 work item 有一个不可见
`focus_question`，但 loss 默认监督同一 assistant target 的四个 YES/NO 值 token；
也就是训练和测试都保持“主问题均衡，副问题顺带记录/监督”的口径。
当 `EVAL_STEPS>0` 时，val split 同样必须八桶完整；缺桶会终止训练，绝不会只打印 warning 后跳过验证。
训练产物：

- `checkpoints/sft_loop_phase1_runs/latest/final/adapter_model.safetensors`
- `checkpoints/sft_loop_phase1_runs/latest/final/adapter_config.json`
- `checkpoints/sft_loop_phase1_runs/latest/final/sft_loop_phase1_adapter_config.json`
- `checkpoints/sft_loop_phase1_runs/latest/checkpoint-<step>/`
- `checkpoints/sft_loop_phase1_runs/latest/tb/`
- `checkpoints/sft_loop_phase1_runs/latest/train_balance.json`

正式训练输出固定为：

```text
checkpoints/sft_loop_phase1_runs/run_static_obstacle_final_4rgb/
checkpoints/sft_loop_phase1_runs/run_static_obstacle_final_2rgb_endpoints/
checkpoints/sft_loop_phase1_runs/latest -> the mode trained most recently
```

训练时可以用 TensorBoard 看 `train/loss`、`val/loss`、`val/focus_*_acc`：

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_loop_phase1_runs/latest/tb
```

中间的 `checkpoint-<step>/` 也可以直接作为 `--adapter-dir` 跑第 4 节完整 eval。如果
`val/loss` 后期上升，或者某个 checkpoint 的完整 F1 明显好于 `final/`，就优先用那个 checkpoint。

## 4. 测试训练后的 LoRA

训练后的正式 eval 默认开启时间戳。每次测试会写到：

```text
checkpoints/sft_loop_phase1_eval/<eval_name>/YYYYmmdd_HHMMSS/
```

多卡时 timestamp 由 rank0 生成并广播，所有 rank 会写进同一个时间戳目录；不需要手动
`--overwrite`，也不会覆盖上一次测试。

LoRA eval **不要传** `--history-rgb-mode`：脚本会从
`<adapter>/sft_loop_phase1_adapter_config.json` 自动读取模式，随后自动把目录命名成 `4rgb` 或
`2rgb_endpoints`。旧 adapter 没有该字段时按历史兼容口径视为 `4rgb`，并在结果中标注
`legacy_adapter_default_4rgb`。

单卡：

```bash
GPU_IDS=0 \
python qwen3vl_local/sft_loop_phase1/eval.py \
  --adapter-dir checkpoints/sft_loop_phase1_runs/run_static_obstacle_final_2rgb_endpoints/final
```

2 卡：

```bash
GPU_IDS=0,1 torchrun --nproc_per_node=2 \
  qwen3vl_local/sft_loop_phase1/eval.py \
  --adapter-dir checkpoints/sft_loop_phase1_runs/run_static_obstacle_final_4rgb/final
```

4 卡：

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_loop_phase1/eval.py \
  --adapter-dir checkpoints/sft_loop_phase1_runs/run_static_obstacle_final_2rgb_endpoints/final
```

建议保留 base 和 LoRA 两份结果：

- `checkpoints/sft_loop_phase1_eval/base_static_obstacle_final_4rgb/<timestamp>/`
- `checkpoints/sft_loop_phase1_eval/base_static_obstacle_final_2rgb_endpoints/<timestamp>/`
- `checkpoints/sft_loop_phase1_eval/lora_static_obstacle_final_4rgb/<timestamp>/`
- `checkpoints/sft_loop_phase1_eval/lora_static_obstacle_final_2rgb_endpoints/<timestamp>/`

这样可以直接比较：

- base Qwen 是不是已经会答某些问题；
- LoRA 是否只修正了答案格式，还是确实改善了视觉判据；
- 错例 evidence 是否暴露 prompt 仍然含糊，例如把直路/护栏误判高速，或把他车闯红灯误判灯异常。

## 4.1 独立评测打包

给定一个 LoRA adapter/run 目录，直接跑 base + LoRA production/audit-prompt eval、
轻量 label/RGB audit matrix，并从 eval 自带 `error_cases/` 抽取适量 RGB，生成不超过
`30MB` 的审计压缩包：

```bash
ADAPTER_DIR=checkpoints/sft_loop_phase1_runs/run_static_obstacle_final_4rgb/final \
bash qwen3vl_local/sft_loop_phase1/eval.sh
```

默认四卡 `GPU_IDS=0,1,2,3`。base eval 默认从 adapter config 读取同一个
`history_rgb_mode`，确保 base/LoRA 输入合同一致；显式设置 `HISTORY_RGB_MODE=...`
时才覆盖。输出到 `checkpoints/sft_loop_phase1_eval_review/<timestamp>/`，压缩包为
`sft_loop_phase1_audit_bundle.tar.gz`。包内包含 metrics/report/case JSONL、
adapter/run-root 小型元信息（不含权重）、少量 label audit sheet，以及按错误 task
分层抽样的降采样 error RGB，供代码与 prompt 审计。
