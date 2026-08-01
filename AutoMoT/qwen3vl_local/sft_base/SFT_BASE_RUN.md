# SFT Base Run

默认当前目录是远端 `AutoMoT/`。

## 1. 构建数据

当前协议使用固定语义 token 作为答案，例如 `RS: SIGNAL_INTERSECTION`、
`EVENT: RULE_VIOLATION`。**代码里已经完全没有 A/B/C 选项字母**：Q2 候选在数据里
就是有序 list `event_candidates_ordered`，顺序即 prompt 展示顺序。当前协议下
Q2 候选固定来自当前 RS 的静态候选全集；逐帧 `allowed_events` 只保留为 GT 解析和
audit 字段，不参与出题，避免候选长度直接泄漏“本帧有没有异常”。

regular EVENT 以 RS 为准做 canonical 映射：R4 下任意 regular 都监督
`SIGNAL_COMPLIANCE`，R5 下任意 regular 都监督 `PRIORITY_NEGOTIATION`；
R1/R2/R3 下不属于本 RS 静态表的路口通行类 regular 映射回默认
`LANE_FOLLOWING`。原始 regular code 仍写入 `event_code_raw` /
`regular_event_codes` / `event_labels_raw` 供审计。

因此必须先重新构建数据，再重新训练：

- 旧 index 里的 `rs_option` / `event_option_map` 两个字母字段已被删除。
- 旧 `sft_base_rs_event_token_choice_re_expanded` index 没有做 RS regular 映射，
  会被当前 `DATASET_VERSION=sft_base_rs_event_token_choice_rs_regular_mapped` 拒绝。
- `RouteSequenceDataset` 读到缺 `event_candidates_ordered` 的旧 index 会直接报错，
  不做兼容降级 —— 静默兼容会让候选顺序悄悄变化，指标看着正常但训练分布已经错了。

如果训练/评估时报：

```text
frame missing 'event_candidates_ordered' ... Rebuild it with the current build_dataset.py
```

说明 `--index` 指向的是旧 schema 数据，重跑下面的 build_dataset 即可。

如果测试时报：

```text
adapter route mismatch: expected sft_base_token_choice, got 'sft_base_direct_choice'
```

说明 `--adapter-dir` 指向的是旧 ABC/direct-choice 权重，不是当前 token-choice 权重。
不要用这类 checkpoint 跑当前评测，也不要为了通过校验去改 adapter config；应换成
当前代码重新训练得到的 `sft_base_token_choice` adapter。

```bash
python qwen3vl_local/sft_base/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_base_data
```

Smoke：

```bash
python qwen3vl_local/sft_base/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_base_data_smoke \
  --max-routes 2 \
  --max-frames-per-route 4
```

重建数据前可先扫一遍 RS x EVENT 共现，检查静态候选表是否漏掉真实组合。该脚本复用
`build_dataset.py` 的异常时长 / 数据缺失 route 过滤口径，并同时报告 UE 与 regular
两侧的 missing / low-rate / spurious：

```bash
python qwen3vl_local/sft_base/audit_rs_event_cooccurrence.py \
  --collection-dir keyframe_filter/collection_output \
  --min-rate 0.001 \
  --output-json checkpoints/sft_base_data/rs_event_cooccurrence.json
```

重点看这些字段：

| 字段 | 含义 |
|---|---|
| `missing_ue_combinations` / `spurious_ue_combinations` | UE 侧静态表是否漏/多了高频组合 |
| `missing_regular_combinations` / `spurious_regular_combinations` | 映射后的 regular 侧静态表是否漏/多了高频组合；新协议应为空 |
| `regular_static_mismatch_total` / `regular_static_mismatch_rate_by_rs` | 映射后 GT regular 不在当前 RS 静态候选表里的比例；新协议应接近 0 |
| `raw_regular_by_rs` | 映射前原始 regular 分布，用于复盘标注噪声 |
| `raw_regular_remap_total` / `raw_regular_remap_breakdown` | 原始 regular 被 RS canonical 映射的总量与 scenario/route 归因 |
| `focus_combo_top_routes` | 默认聚焦原始 `R5:R-E4`，用于保留 R5 下 `SIGNAL_COMPLIANCE` 标注冲突证据 |

如果要改聚焦组合：

```bash
python qwen3vl_local/sft_base/audit_rs_event_cooccurrence.py \
  --collection-dir keyframe_filter/collection_output \
  --focus-combo R2:R-E4 \
  --top-k 30
```

静态 UE 表采用严格方案 A：先按 `build_dataset.py` 同款口径剔除 `noScenarios`、
异常时长 route、数据缺失 route 和失败 route，再只保留 `count >= 20` 且
`rs_frame_rate >= 0.1%` 的组合。regular 表采用 RS canonical 映射后的语义保守版：
R1/R2 只保留 `R-E1/R-E2`，R3 保留 `R-E1/R-E2/R-E3`，R4 固定 `R-E4`，
R5 固定 `R-E5`。重建数据后验证点：

```text
single_candidate_rate                    -> 0
regular_static_mismatch_total            -> 0
missing_regular_combinations             -> []
spurious_regular_combinations            -> []
gt_static_candidate_mismatch             -> 只剩低频 UE 组合，约等于全量 163 帧在 val split 的对应子集
R3 regular 分布                          -> R-E1/R-E3/R-E2 都存在，保持 3 选 1 判别
```

## 2. 静态检查

```bash
python qwen3vl_local/sft_base/check_loss_mask.py
python qwen3vl_local/sft_base/test_dataset_contract.py
python qwen3vl_local/sft_base/test_eval_metrics.py
GPU_IDS=0 bash qwen3vl_local/sft_base/train.sh check
```

## 3. 训练

单卡：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_base/train.sh single
```

4 卡：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_base/train.sh ddp
```

默认输出到 `checkpoints/sft_base_runs/run_<RUN_TAG>/`，并维护 `checkpoints/sft_base_runs/latest`。默认 `LORA_VISION_SCOPE=last4`、`LORA_RANK=32`、`LORA_ALPHA=64`，保持 LoRA scaling 为 2，并启用视觉 fuse guard；如果要纯语言 LoRA 对照：

多卡训练默认 `FRAMES_PER_SYNC=64`，会在长 route 内按固定帧数做梯度同步 heartbeat，避免不同 rank 的 route 帧数差异导致 NCCL all-reduce 等待超时。排查时可调小到 `32`，或在确认单条 route 很短时设为 `0` 回到整条 route 结束后同步。

针对 `ue_acc=0` 和 memory-copy shortcut，当前训练默认做几件事：

- 首帧 memory 使用 `UNKNOWN`，不再白送 GT RS。
- 训练 prompt 中的 `BELIEVED_RS/BELIEVED_EVENT` 会按高概率置错或置为 `UNKNOWN`，让抄 memory 不再稳定拿高分；Q1 只显示 `BELIEVED_RS`，Q2 才显示并使用 `BELIEVED_EVENT`。
- 另有 `MEMORY_DROPOUT_PROB` 会作为独立第一层整块隐藏离散先验，只保留 `EGO_TO_GOAL_XY`，制造必须看图的帧；route 首帧固定 UNKNOWN/UNKNOWN，不参与 dropout 或 EVENT 扰动。
- 默认 `MEMORY_PERTURBATION_MODE=route_state`：wrong/UNKNOWN 会在同一 route 内连续保持 3-5 个真实帧，到期恢复干净 GT 轨迹；`frame` 模式保留旧逐帧独立扰动，主要作消融。
- Q1 用 GT RS 更新后，训练侧会为 Q2 在当前 RS 池里单独重采 EVENT memory；keep 分支沿用进入本帧前的干净 EVENT memory（上一帧 GT），防止“扰动 RS 被纠正回 GT”把 Q2 EVENT memory 大量失效成 UNKNOWN，也防止 EVENT 转折帧把本帧答案写进 prompt。
- `regular->UE`、`UE->regular`、RS 变化帧及其前后 3 帧会被重点重复训练。
- Q1 只监督 RS；Q2 训练所有 GT 在候选表里的帧。regular 展开为 5 个语义 token，R3 不再是单候选；UE loss 会按 RS 条件 UE 率缩放，regular loss 会按 R-E 子类频次缩放。

当前默认值：

| 环境变量 | 默认 | 用途 |
|---|---:|---|
| `FIRST_FRAME_MEMORY_UNKNOWN` | `1` | 首帧 memory 是否置为 UNKNOWN |
| `MEMORY_RS_WRONG_PROB` | `0.30` | 非首帧把 RS memory 改成错误 RS 的概率 |
| `MEMORY_RS_UNKNOWN_PROB` | `0.40` | 非首帧把 RS memory 置为 UNKNOWN 的概率 |
| `MEMORY_EVENT_WRONG_PROB` | `0.35` | 非首帧把 EVENT memory 改成错误 EVENT 的概率 |
| `MEMORY_EVENT_UNKNOWN_PROB` | `0.35` | 非首帧把 EVENT memory 置为 UNKNOWN 的概率 |
| `RS_WRONG_EVENT_UNKNOWN_PROB` | `0.25` | RS 被置错时，EVENT 置 UNKNOWN 而不是从新 RS 候选池抽错项的概率 |
| `MEMORY_DROPOUT_PROB` | `0.15` | 非首帧独立触发隐藏 RS/EVENT memory、只保留导航 goal 的概率 |
| `MEMORY_PERTURBATION_MODE` | `route_state` | route 级块状扰动；设 `frame` 可回到旧逐帧独立扰动 |
| `MEMORY_PERTURB_DURATION_MIN/MAX` | `3` / `5` | wrong/UNKNOWN/keep 段的真实帧长度采样范围 |
| `TRANSITION_FRAME_REPEAT` | `4` | 转折邻域帧最少重复次数 |
| `TRANSITION_FRAME_WINDOW` | `3` | 转折点前后纳入重复的窗口半径 |
| `UE_EVENT_LOSS_WEIGHT` | `4.0` | Q2 UE token loss 权重 |
| `RE_EVENT_LOSS_WEIGHT` | `1.0` | Q2 regular 硬负样本 token loss 基础权重；会按 R-E 子类频次 inverse-sqrt 缩放 |
| `SINGLE_CANDIDATE_RE_SCALE` | `0.1` | 单候选 regular 相对 `RE_EVENT_LOSS_WEIGHT` 的缩放；新协议正常应几乎用不到 |
| `UE_FRAME_REPEAT` | `2` | UE 子类重复训练的基础倍率 |
| `UE_REPEAT_MODE` | `inverse_sqrt` | UE repeat 模式；`inverse_sqrt` 按子类逆频率放大长尾，`fixed` 保留旧固定 repeat |
| `UE_REPEAT_MAX` | `8` | UE 子类逆频率重复次数上限 |

如果要更激进地打断 memory-copy，可以临时提高扰动：

```bash
MEMORY_DROPOUT_PROB=0.25 MEMORY_RS_UNKNOWN_PROB=0.50 MEMORY_RS_WRONG_PROB=0.35 \
GPU_IDS=0 bash qwen3vl_local/sft_base/train.sh single
```

`UE_EVENT_LOSS_WEIGHT` 不建议无限拉高，`RE_EVENT_LOSS_WEIGHT` 也不建议直接设为 0；否则模型可能从“全 regular”翻到“全 UE”。多候选 regular 是最重要的硬负样本，用来约束“候选里有 UE 但画面不支持 UE”的假阳性。训练日志和 TensorBoard 会写 `train/q2_ue_rate_last_batch`，用于确认本轮 batch 里确实喂到了 UE 监督。

```bash
LORA_VISION_SCOPE=off GPU_IDS=0 bash qwen3vl_local/sft_base/train.sh single
```

关闭视觉 fuse guard 只建议排查用：

```bash
VISION_GUARD_ENABLED=0 GPU_IDS=0 bash qwen3vl_local/sft_base/train.sh single
```

## 4. TensorBoard 与日志

训练脚本会在 rank0 写 TensorBoard events：

```text
checkpoints/sft_base_runs/run_<RUN_TAG>/tb/
checkpoints/sft_base_runs/latest/tb/
```

远端从 `AutoMoT/` 目录启动：

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_base_runs/latest/tb
```

也可以把 logdir 指到 run 根目录，后续如果增加 `eval_tb/` 等子目录，TensorBoard 会在左侧 run 列表中分开展示：

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_base_runs/latest
```

`tb_serve.sh` 会自动选空闲端口、用 `--bind_all` 启动 TensorBoard，并在 stdout 打印本地浏览器 URL。常用覆盖：

```bash
TB_PORT=6007 bash qwen3vl_local/tb_serve.sh checkpoints/sft_base_runs/latest/tb
TB_BIND=127.0.0.1 bash qwen3vl_local/tb_serve.sh checkpoints/sft_base_runs/latest/tb
TB_EXTRA="--samples_per_plugin images=200" bash qwen3vl_local/tb_serve.sh checkpoints/sft_base_runs/latest/tb
```

优先查看：

| Tag | 用途 |
|---|---|
| `train/loss` | optimizer step 后的全局平均训练 loss |
| `train/q2_rate_last_batch` | 最近一次同步 batch 中包含 Q2 的帧比例 |
| `train/q2_ue_rate_last_batch` | 最近一次 Q2 监督中 UE 帧比例，用于排查 UE 是否被 regular 淹没 |
| `train/q2_ue_weight_share_last_batch` | 最近一次 Q2 监督中 UE loss 权重占比；配合 `ue_fp_on_multi_candidate_re_rate` 判断 regular 权重是否压得过低 |
| `train/grad_norm/language` | 语言 LoRA 梯度范数 |
| `train/grad_norm/vision` | 视觉 LoRA/merger 相关梯度范数；`LORA_VISION_SCOPE=off` 时可能没有 |
| `train/param_norm/lora_vision` | 视觉侧可训练参数范数，用于观察 fuse 是否异常漂移 |
| `train/vision_guard_bad_steps` | 视觉 fuse guard 连续异常步数 |
| `val/loss` | 评估间隔触发的 teacher-forced 验证 loss；prompt memory 使用训练同款 anti-copy 扰动，seed 固定为 `SEED`，不随 rank 漂移 |
| `val/samples` / `val/skipped` | 验证样本数与跳过帧数 |
| `val/q2_rate` | 验证集中 Q2 监督帧比例 |

训练 stdout/stderr 默认同步写到当前 run 的 `log.txt`，路径为：

```text
checkpoints/sft_base_runs/run_<RUN_TAG>/log.txt
checkpoints/sft_base_runs/latest/log.txt
```

如果 TensorBoard 里只有 events header 没有曲线，先确认训练已经完成至少 1 个 optimizer step；`LOGGING_STEPS` 只影响写入频率，默认每 5 step 记录一次，step 1 也会记录。

## 5. 评估

评估分三类，并且完全不做脚本纠正：Q1 预测错 RS 时仍按学生 RS 静态候选全集构造并继续询问当前帧 Q2，下一帧继续沿用模型自己维护出的 memory；Q2 输出非法时也不重置。

日常只需要改三类东西：

1. `GPU_IDS`：指定用哪张卡或哪几张卡。
2. `--adapter-dir`：测 LoRA 时显式指定 adapter；不传则跑 base 零样本。
3. `--task`：每次必须显式指定测 `rs`、`event`，或 `full`。

`--model-dir` 通常不用改，除非你换了 base 模型目录。

测 LoRA 时，`--adapter-dir` 必须指向当前 token-choice 协议训练出的 adapter。旧
`sft_base_direct_choice` / ABC checkpoint 会被 eval 直接拒绝，这是为了避免旧输出协议
和新语义 token prompt 混用后得到没有意义的指标。

测 base 零样本时可以省略 `--adapter-dir`，但必须手动给 `--output-dir`，否则默认输出根会落到
`checkpoints/sft_base_runs/latest`，容易和训练 run 混在一起。

默认值：

| 参数 | 默认 |
|---|---|
| `--index` | `checkpoints/sft_base_data/val_sequence_index.jsonl` |
| `--model-dir` | `checkpoints/Qwen3-VL-4B-Instruct` |
| `--adapter-dir` | 默认 `None`；不传时评估 base 模型 |
| `--task` | 无默认值，每次必须指定；`full` 等价于 `--eval-mode full_route` |
| `--image-ablation` | `none`；可设 `black/random` 做视觉消融 |
| `--ablate-goal` | `False`；设为 true 时隐藏 `EGO_TO_GOAL_XY`，可与黑图组合测导航文本捷径 |
| `--output-dir` | 默认自动生成；手动指定时会在该目录写 `metrics.json`、`frames.jsonl`、`summary.md` |
| RS/EVENT 转折 case 数 | `128` |
| full_route 随机 route 数 | `16` |
| 输出路径 | 自动写到 adapter run 目录下的 `eval_results/<task>/<YYYYMMDD_HHMMSS>/` |

旧的 `--eval-mode full_route/rs_transition/event_transition` 仍然兼容；日常推荐用更短的
`--task full/rs/event`。

默认不需要手写输出路径。每次评测都会按任务和时间自动保存，不会覆盖上一次结果。例如：

```text
checkpoints/sft_base_runs/latest/eval_results/rs_transition/20260730_143012/
├── metrics.json
├── frames.jsonl
└── summary.md
```

base 零样本基线建议单独放目录：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --task full \
  --output-dir checkpoints/sft_base_base_eval/full_original
```

黑图 + no-goal 消融建议单独放目录，和原图结果成对比较：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --task full \
  --image-ablation black \
  --ablate-goal \
  --output-dir checkpoints/sft_base_base_eval/full_black_no_goal
```

三个文件的用途：

| 文件 | 用途 |
|---|---|
| `metrics.json` | 汇总指标，适合后续脚本读取和横向对比 |
| `frames.jsonl` | 逐帧复盘，包含 GT/PRED RS、GT/PRED EVENT、原始生成文本、转折窗口和 case summary |
| `summary.md` | 中文摘要，包含本次任务、adapter、保存路径、关键指标和指标解释 |

`metrics.json` 里还会写 `q2_candidate_count_report` 与
`q2_rs_candidate_count_report`，分别按候选数、RS×候选数分层统计 EVENT acc 与
UE-vs-regular P/R/F1；R3 的 3 个 regular 候选、R1 的 7 选 1 与 R4/R5 的 5 选 1 不会再混在一个 event_acc 里。`regular_internal_confusion_report` 单独展示 R-E 子类混淆。

评估结束后，rank0 终端会打印：

```text
[eval] saved metrics=...
[eval] saved frames=...
[eval] saved summary=...
```

多卡评测时，每个 rank 会先写自己的 `frames.jsonl.rank*` 临时分片，rank0 结束后自动合并成最终
`frames.jsonl`，并删除临时分片。

### 5.1 简易单卡测试

RS 转折专项测试：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --task rs
```

EVENT 转换专项测试：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --task event
```

随机完整路线闭环测试：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --task full
```

换 adapter：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --adapter-dir checkpoints/sft_base_runs/run_20260728_120000/final \
  --task rs
```

检查某个 adapter 是否能用于当前评测：

```bash
cat checkpoints/sft_base_runs/latest/final/sft_base_adapter_config.json
```

其中必须包含：

```json
{
  "route": "sft_base_token_choice",
  "dataset_version": "sft_base_rs_event_token_choice_rs_regular_mapped"
}
```

换 base model：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --task event
```

### 5.2 简易多卡测试

`sft_base/eval.py` 现在支持 `torchrun` 多卡分片评估。每个 rank 加载一份模型，
按 `case_idx % WORLD_SIZE == RANK` 处理不同 case；full_route 模式里一个 case 是一条完整 route，
RS/EVENT transition 模式里一个 case 是一个转折窗口。这样每个 case 内部的 memory 仍然串行推进，
不会被跨卡切断。

2 卡 RS 转折专项测试：

```bash
GPU_IDS=0,1 torchrun --standalone --nproc_per_node=2 \
  qwen3vl_local/sft_base/eval.py \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --task rs
```

4 卡 RS 转折专项测试：

```bash
GPU_IDS=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  qwen3vl_local/sft_base/eval.py \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --task rs
```

2 卡 EVENT 转换专项测试：

```bash
GPU_IDS=0,1 torchrun --standalone --nproc_per_node=2 \
  qwen3vl_local/sft_base/eval.py \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --task event
```

4 卡 EVENT 转换专项测试：

```bash
GPU_IDS=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  qwen3vl_local/sft_base/eval.py \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --task event
```

2 卡完整路线随机闭环测试：

```bash
GPU_IDS=0,1 torchrun --standalone --nproc_per_node=2 \
  qwen3vl_local/sft_base/eval.py \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --task full
```

4 卡完整路线随机闭环测试：

```bash
GPU_IDS=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  qwen3vl_local/sft_base/eval.py \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --task full
```

多卡输出规则：

- `--output-json` 只由 rank0 写最终汇总指标，里面会记录 `world_size`。
- `--output-jsonl` 每个 rank 先写临时分片，例如 `frames.jsonl.rank0`；
  rank0 在所有 rank 结束后合并成用户指定的最终 jsonl，并删除临时分片。
- `GPU_IDS=4,5 torchrun --standalone --nproc_per_node=2 ...` 表示用物理 4、5 号卡；
  进程内部分别看到 `cuda:0` 和 `cuda:1`，这是 `CUDA_VISIBLE_DEVICES` 的正常映射。
- `--nproc_per_node` 必须和 `GPU_IDS` 里的卡数一致或小于它；常用写法是二者相同。

### 5.3 全量参数 demo

如果要临时改窗口大小、case 数、seed 或输出目录，可以展开成全量参数。例如 RS：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --index checkpoints/sft_base_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --task rs \
  --transition-window 8 \
  --transition-tolerance 3 \
  --max-transition-cases 128 \
  --seed 20260724 \
  --output-dir checkpoints/sft_base_runs/latest/eval_results/manual_rs_debug
```

`eval.py` 会先校验 adapter 目录中的 `sft_base_adapter_config.json`。如果 route、
dataset version、base model path 或 vision scope 不匹配，会直接报错。

起始 memory 噪声只在每条完整 route 或每个转折窗口的第一帧注入，后续仍然不纠正，用来测模型能不能自己恢复：

```bash
--initial-memory-noise rs
--initial-memory-noise event
--initial-memory-noise both
--initial-memory-noise random
```

常用指标口径：

| 指标 | 含义 |
|---|---|
| `rs_acc` / `event_acc_end_to_end` | 全部评估帧上的 RS 准确率、每帧都问 Q2 的端到端 EVENT 准确率 |
| `joint_acc` | 真实串行主指标：RS 正确且 EVENT 正确的比例 |
| `q2_trigger_rate` | 进入 Q2 的比例；新协议应接近 100% |
| `script_resets` | 脚本纠偏审计字段；评测不允许纠偏，正常必须恒为 0 |
| `rs_transition_hit_rate` | RS 转折 case 在容忍窗口内切到目标 RS 的比例 |
| `event_transition_hit_rate` | UE/regular/EVENT 转换 case 在容忍窗口内切到目标 EVENT 的比例 |
| `rs_transition_already_at_target_rate` / `event_transition_already_at_target_rate` | 命中 case 中窗口左边界已经等于目标值的比例；越高越说明 hit_rate 被锁死模型污染 |
| `event_unreachable_due_to_rs_rate` | GT EVENT 在学生 RS 候选下不可达的比例 |
| `ue_vs_re_f1` | 由 Q2 EVENT 折叠得到的 UE-vs-regular 二分类 F1 |
| `event_acc_multi_candidate` | 排除单候选送分题后的 EVENT 准确率 |
| `event_pred_ue_rate_multi_candidate` | 排除单候选送分题后的 UE 输出比例 |
| `ue_vs_re_f1_multi_candidate` | 排除单候选送分题后的 UE-vs-regular F1 |
| `event_acc_single_re` / `event_acc_single_ue` | 单候选 regular/UE 的 EVENT 准确率 |
| `event_acc_multi_re` / `event_acc_multi_ue` | 多候选 regular 硬负样本 / UE 硬正样本的 EVENT 准确率 |
| `ue_fp_on_multi_candidate_re_rate` | 多候选 regular 帧被误报成 UE 的比例，越低越好 |
| `ue_fp_on_multi_candidate_re_by_rs` | 上一项按 RS 分组的明细，避免 R2 假阳性被全局均值掩盖 |
| `single_candidate_invalid_rate` | 单候选题输出候选外 token 的比例 |
| `dataset_candidate_mismatch_rate` / `dataset_candidate_mismatch_ue_rate` | GT EVENT 不在 dataset 自己候选表中的比例；这些帧不进 EVENT 评分分母 |
| `q2_single_candidate_rate` / `q2_multi_candidate_rate` | Q2 单候选送分题比例 / 真正需要判别的多候选比例 |
| `rs_change_f1` | 相邻帧 RS 是否变化的 F1；同时约束该切和不该切 |
| `re_to_ue_f1` / `ue_to_re_f1` | 相邻帧异常起始 / 异常结束检测 F1，拆开看漏检和持续误报 |
| `false_transition_rate_when_gt_stable` | RS、regular->UE、UE->regular 合并后的 GT 稳定帧假转折比例 |
| `rs_transition_direction_confusion` / `event_transition_direction_confusion` | `(gt_source->gt_target) vs (pred_source->pred_target)` 的 sparse 转折方向混淆 |
| `rs_confusion_report` / `event_confusion_report` | RS 5 类与 EVENT 13 类混淆矩阵、per-class P/R/F1 |
| `regular_internal_confusion_report` | 只看 R-E1..R-E5 目标帧时，regular 子类被预测成哪类 |
| `ue_pred_regular_rate` | UE 帧进入 Q2 后仍被判成 regular 子类的比例 |
| `*_hit_offset_avg` / `*_abs_hit_offset_avg` | 命中帧相对标注转折帧的平均偏移和平均绝对偏移，单位是 frame，负数表示提前 |
| `*_early_hits` / `*_on_time_hits` / `*_late_hits` | 命中发生在标注转折前、同帧或后几帧的数量 |
| `output-jsonl` 每行 | 单帧复盘和 `transition_case_summary`，包含 route/frame、转折点、容忍窗口、GT/PRED RS、GT/PRED EVENT、原始生成文本 |

快速查看最近一次 RS 测试摘要：

```bash
ls -td checkpoints/sft_base_runs/latest/eval_results/rs_transition/* | head -1
cat $(ls -td checkpoints/sft_base_runs/latest/eval_results/rs_transition/* | head -1)/summary.md
```

快速查看 EVENT 测试里预测错的帧，可以先从 `frames.jsonl` 里 grep：

```bash
grep '"event_ok": false' checkpoints/sft_base_runs/latest/eval_results/event_transition/*/frames.jsonl | head
```

### 5.4 黑图 / 随机图诊断

这个测试用于判断模型到底有没有用视觉。它只替换 RGB history，不改 prompt、memory、
候选、adapter 或解析逻辑。

黑图 RS：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --task rs \
  --image-ablation black
```

随机图 RS：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --task rs \
  --image-ablation random
```

黑图 EVENT：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --task event \
  --image-ablation black
```

随机图 EVENT：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base/eval.py \
  --adapter-dir checkpoints/sft_base_runs/latest/final \
  --task event \
  --image-ablation random
```

输出会分到不同目录：

```text
eval_results/rs_transition_black/<时间>/
eval_results/rs_transition_random/<时间>/
eval_results/event_transition_black/<时间>/
eval_results/event_transition_random/<时间>/
```

重点看 `summary.md` 或 `metrics.json` 里的这些指标：

| 指标 | 判断 |
|---|---|
| `rs_visual_gain_over_first_pred_lock` | RS 相对“模型首帧预测锁死”的净增益；黑图/随机图下如果几乎不变，说明视觉贡献很弱 |
| `event_visual_gain_over_regular_baseline` | EVENT 相对“按 GT RS 答多数 regular 子类”的净增益；接近 0 说明 EVENT 坍缩 |
| `rs_pred_change_rate` vs `rs_gt_change_rate` | 预测变化率远低于 GT 变化率，说明 RS 被 memory 锁死 |
| `rs_locked_case_rate` | 整段 RS 预测完全不变的 case 比例 |
| `event_pred_ue_rate_multi_candidate` | 排除单候选后仍长期接近 0，表示 Q2 EVENT 坍缩到 regular |
| `ue_vs_re_f1_multi_candidate` | 排除单候选后的 UE-vs-regular 判别能力；比全量 F1 更接近真实能力 |
| `rs_change_f1` | 原图下也接近 0 表示模型没有学到 RS 变化；黑图下应明显更差 |
| `re_to_ue_f1` | 原图下也接近 0 表示模型没有学到异常起始；黑图下应明显更差 |
| `false_transition_rate_when_gt_stable` | 过高说明模型乱切；过低但 `*_change_f1` 接近 0 说明模型锁死 |

建议验收门槛：

| 指标 | 门槛 |
|---|---:|
| `rs_visual_gain_over_first_pred_lock` 原图 - 黑图 | > +10pt |
| `event_pred_ue_rate_multi_candidate` | > 5% |
| `rs_locked_case_rate` | < 20% |
| `rs_confusion_report.per_class.R3.predicted` | > 0 |
| `ue_vs_re_f1_multi_candidate` | > 0.35 |
| `rs_change_f1` | > 0.15 |
| `re_to_ue_f1` | > 0.20 |

## 6. 维护检查

```bash
python -m py_compile qwen3vl_local/sft_base/*.py
python qwen3vl_local/sft_base/check_loss_mask.py
python qwen3vl_local/sft_base/test_dataset_contract.py
python qwen3vl_local/sft_base/test_memory_curriculum.py
python qwen3vl_local/sft_base/test_eval_candidates.py
python qwen3vl_local/sft_base/test_eval_metrics.py
```

候选过滤偏差审计：

```bash
python qwen3vl_local/sft_base/audit_eval_candidate_drift.py \
  --index checkpoints/sft_base_data/val_sequence_index.jsonl \
  --expect-mismatch-combinations
```

审计输出会分开列出：

| 字段 | 含义 |
|---|---|
| `set_mismatch_examples` | eval 候选集合与 dataset 候选集合不同的样例 |
| `unreachable_scoreable_examples` | dataset 候选本身可达，但 eval 候选不可达的样例 |
| `dataset_candidate_mismatch_examples` | GT EVENT 不在 dataset 自己候选表中的上游数据缺陷样例 |
| `dataset_candidate_mismatch_ue_rate` | UE 帧中这类数据缺陷的占比 |
| `gt_static_candidate_mismatch_combinations` | GT EVENT 不在 GT RS 静态全集内的低频组合分布 |
| `unexpected_gt_static_candidate_mismatch_combinations` | 不在已知低频拒绝组合白名单内的新缺口；非空才是真问题 |

`test_dataset_contract.py` 会检查 sft_base 与 sft_v5 的 Q2 候选顺序是否保持一致。
`test_eval_candidates.py` 会检查 eval 侧候选构造不会用静态 RS 表过滤逐帧
allowed_events 事实、集合相同时会复用 dataset 顺序。
多卡训练相关改动需要额外关注 `train.py` 中 `_sync_trainable_grads()` 与
`run_batch(..., sync_grads=True)` 的调用边界，确保每个 rank 的 collective 次数一致。
