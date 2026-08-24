# SFT New Loop Phase2 运行说明

`sft_new_loop_phase2` 是融合版 Phase1 之后的 EVENT 级单轮问答。它从旧
`sft_loop_phase3` 继承数据过滤、LoRA/DDP、频繁 loss eval、generation eval、
TensorBoard、错例 RGB 审计与一键脚本，但彻底删除 synthetic Phase2 RS 对话：

- 输入只有一个 system turn 和一个带 RGB+EVENT prompt 的 user turn；
- 不生成 `RS1/RS2/RS4/RS5` 文本，不放伪 assistant answer，也不保留 RS KV 前缀；
- ROAD_CORRIDOR 问 `UE1/UE3/UE5`，LOCAL_JUNCTION 问 `UE6`；
- 每个问题组最后都回答 `INVALID_EVENT_CONTEXT`。

## 1. 标签合同

- `UE1`：前车急刹/突然减速。普通稳定跟车、正常红灯排队不是 UE1。
- `UE3`：他车正在进入或可见地即将进入 ego future corridor。保留旧 prompt v2
  的 `about to occupy / dynamic crossing` 边界，不要求他车已经完全居中进入 ego lane。
- `UE5`：对向车侵入自车可用通道；自车主动借对向绕障不是 UE5。
- `UE6`：可见局部路口内，他车违反信号/路权并迫使有优先权的 ego 避让。
- `UE2/UE4/UE7/UE8` 暂由融合版 Phase1 或其它阶段处理，本阶段折叠为 valid RE/all-NO。
- `INVALID_EVENT_CONTEXT=YES` 只表示“本题问题域与可见道路布局明确不相容”，并要求
  所有 UE 为 NO。雾、夜间、遮挡、拥堵、普通队列、静态事故或仅仅没有 UE 都不是 invalid。

高速/R3 是 ROAD_CORRIDOR 的 valid hard negative：正常输出 UE1/UE3/UE5 全 NO、
`INVALID_EVENT_CONTEXT=NO`。路口内没有 UE6 也是 valid RE，不是 invalid。

这些边界来自 `keyframe_filter/ROAD_EVENT_CLASSIFICATION_PLAN.md`、
`ROAD_EVENT_RGB_AUDIT_ARCHIVE_202607.md` 和旧 Phase3 的
`EVAL_PROMPT_V2_V3_20260821.md`。数据构建仍要求每个 scenario/Town 至少已有一条
完整逐帧 RGB review，并默认排除 visual-risk 帧和异常时长 route。

## 2. 构建数据

从 `AutoMoT/` 目录运行：

```bash
python qwen3vl_local/sft_new_loop_phase2/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_new_loop_phase2_data \
  --test-ratio 0.10 \
  --val-ratio 0.05 \
  --regular-multiplier 1.0 \
  --highway-regular-fraction 0.25 \
  --invalid-ratio 0.20
```

输出：

- `frame_index.jsonl`：RGB 路径相对 `--data-root` 保存，可跨机器重映射；
- `manifest.json`：split/route/class/answer、UE 1:1:1:1、RE 中 highway/local、
  invalid source/true-RS/question-domain 统计和 RGB review 覆盖。本次实际扫描到的
  scenario/Town 会与 review coverage 做差集，出现新增或漏审 pair 时构建直接失败。

每个 split 中四个 UE 正类严格 1:1:1:1；RE 总量默认等于一个 UE 桶，其中 25%
保留给 R3/highway；invalid 默认是 valid 主数据的 20%。invalid 按 source class、
true RS 和错误 question domain 轮转，避免大场景支配。

`UE1/UE3/UE5/UE6/RE/INVALID` 是训练和评测的六个必需桶；任何桶缺失都会硬失败，
不会因为 `--max-frames` 截断而静默只评剩余类别。`FOCUS_BALANCE_COUNT=0` 使用六桶中
最小原始桶作为共同采样基数，不再保留不均衡 raw rows。

## 3. 训练

轻量链路检查：

```bash
bash qwen3vl_local/sft_new_loop_phase2/train.sh check
```

单卡与四卡：

```bash
bash qwen3vl_local/sft_new_loop_phase2/train.sh single
GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase2/train.sh single

bash qwen3vl_local/sft_new_loop_phase2/train.sh ddp
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_new_loop_phase2/train.sh ddp
```

常用覆盖：

```bash
EVAL_STEPS=2000 \
GENERATION_EVAL_STEPS=2000 \
GENERATION_EVAL_BALANCE_COUNT=16 \
SAVE_STEPS=20000 \
FOCUS_BALANCE_COUNT=1024 \
REGULAR_FOCUS_MULTIPLIER=2.0 \
HIGHWAY_REGULAR_FRACTION=0.25 \
bash qwen3vl_local/sft_new_loop_phase2/train.sh ddp
```

训练默认每 2000 optimizer step 跑 teacher-forced val 和固定均衡的自由生成 val，
每 20000 step 保存 checkpoint；generation eval 每个 UE/RE/invalid class 默认 16 条，
避免旧 Phase3 仅 2 条/桶带来的 checkpoint 选择噪声。训练会输出：

- `train_balance.json`：class、question domain、true RS、highway/local RE，以及 invalid 四维子类别采样；
- `balance/epoch_*.json`：每轮 invalid 的 source class、true RS、错误问题域、联合签名与均衡 guard；
- `train_eval_metrics.jsonl` / generation records；
- `tb/`：loss、每个问题准确率、invalid joint/all-UE-NO、invalid 四维子类别、highway/local/UE slice；
- `best_val/`、`best_generation/`、`checkpoint-*`、`final/`；
- `sft_new_loop_phase2_adapter_config.json`：prompt hash、history mode、sampling 与单轮输入合同。

## 4. 独立评测

Base：

```bash
GPU_IDS=0 python qwen3vl_local/sft_new_loop_phase2/eval.py \
  --index checkpoints/sft_new_loop_phase2_data/frame_index.jsonl \
  --data-root lead_data \
  --cases-per-bin 64
```

LoRA：

```bash
GPU_IDS=0 python qwen3vl_local/sft_new_loop_phase2/eval.py \
  --index checkpoints/sft_new_loop_phase2_data/frame_index.jsonl \
  --data-root lead_data \
  --adapter-dir checkpoints/sft_new_loop_phase2_runs/latest/best_generation \
  --cases-per-bin 64
```

重点查看：

- `per_question`：UE1/UE3/UE5/UE6/INVALID 的 accuracy/P/R/F1；
- `invalid_contract`：invalid line、UE all-NO、joint 三项；
- `sampling_verification.sampled_invalid_subgroups`：source class、true RS、错误问题域、联合签名的数量和 guard；
- `invalid_subgroup_accuracy`：上述四个维度逐桶的 exact；
- `slice_reports`：四个 UE、applicable RE、R3/highway RE、invalid；
- `cases.jsonl`：实际单轮消息（system + 单个 image/text user）、raw/parsed/GT 与 `invalid_source`；
- `error_cases/`：错误样本的实际输入 RGB。

`--cases-per-bin 0` 表示保留全量行，不限制每个 class 的最终数量；INVALID 行仍会执行
签名一致性以及五种 source、R1-R5、两个错误问题域的完整覆盖守卫，不能借全量模式绕过。
`audit_eval_cases.py` 生成的 `manifest.jsonl`、`summary.json/summary.md` 和每例
`audit_note.md` 也会直接展示 INVALID source class、true RS、错误问题域和联合签名。

生产 prompt 只输出 YES/NO。`--audit-prompt` 会额外要求每题一条短 RGB evidence，
仅用于人工诊断，不能与生产指标混为一谈。

自由生成解析使用完整字符串合同：答案必须严格按当前 prompt 规定的行顺序输出，不能缺行、
重复、换序或夹带解释；audit 模式只额外允许同顺序的 `EVIDENCE_*` 行。任意格式违规都会让
整条 case 的 `format_valid` 和 semantic exact 同时失败。LoRA eval 在加载权重前还会硬校验
`production_prompt_sha256`、`history_rgb_mode` 和解析后的 `base_model_dir`；不兼容 adapter
不会只写 warning/metrics，而会直接中止。

## 5. 一键流程与 RGB 模式矩阵

```bash
bash qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh
```

未设置 `GPU_IDS` 时，所有 launcher 都会通过 `nvidia-smi` 覆盖选择空闲 GPU；显式设置
`GPU_IDS` 时按列表长度推断进程数。`eval.sh` 和 RGB mode matrix 默认分别通过
`EVAL_GPU_COUNT=4` / `DDP_GPU_COUNT=4` 自动选择四张卡，不再固定物理卡 0–3。

只复评现有 adapter：

```bash
RUN_BUILD=0 RUN_BASE_EVAL=0 RUN_TRAIN=0 \
ADAPTER_DIR=checkpoints/sft_new_loop_phase2_runs/latest/best_generation \
bash qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh
```

4RGB/首尾2RGB 对照：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_new_loop_phase2/run_rgb_mode_matrix.sh
```

## 6. 错例审计与打包

```bash
python qwen3vl_local/sft_new_loop_phase2/audit_eval_cases.py \
  --eval-dir checkpoints/sft_new_loop_phase2_eval/<run> \
  --output-dir checkpoints/sft_new_loop_phase2_audit_samples/<name> \
  --data-root lead_data \
  --per-target 12 \
  --overwrite
```

或运行完整 base/LoRA production+audit 评测包：

```bash
ADAPTER_DIR=checkpoints/sft_new_loop_phase2_runs/latest/best_generation \
bash qwen3vl_local/sft_new_loop_phase2/eval.sh
```

人工看图时优先检查：UE1 是否把普通队列当急刹、UE3 是否遗漏“即将进入路径”的早期帧、
UE5 是否混入 ego 主动借道、UE6 是否把普通路口车辆当违规，以及 invalid 是否误伤高速、
低能见度或 valid RE。
