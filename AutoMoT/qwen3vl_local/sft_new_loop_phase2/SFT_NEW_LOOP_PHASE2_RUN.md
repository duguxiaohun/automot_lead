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

production prompt v2 根据 2026-08-25 训练错例的逐帧 RGB 复核补了四条边界：

- 最后一帧决定事件是否仍在发生；旧帧用于确认运动。交互及其即时影响都已结束时不能靠场景
  历史续标；目标刚驶离但 ego 仍明显因该冲突停车/避让时可以保持 YES；
- 同一交互中，前车纵向突然减速是 UE1，侧向进入 future corridor 是 UE3，不能只凭刹车灯或
  单帧距离同时打开两者；停着的斜车身姿态本身也不是 UE3；
- UE5 要求最后一帧仍看得到对向车侵入，空的锥桶路段或只在旧帧出现的车辆不够；
- UE6 要同时看得到冲突车辆与违规/优先权证据，普通转弯、横穿、已经驶离的路口车辆不够。

夜间、雾、眩光或遮挡下看不清事件证据时，对相应 UE 保守回答 NO；只要道路布局仍与问题域
相容，就仍是 valid，而不是 invalid。

production prompt v3 在 2026-08-27 的 69 个 2RGB production 错例复核后补了静态事故/
施工、路边停车、队列车辆和 ego 视差不等于横向进入。重训后 production 只从
`315/384` 到 `316/384`，但 UE3 recall 从 `81.25%` 降到 `71.88%`。2026-08-29 再对
v3 全部 68 个错例的 272 张四帧 RGB 逐帧复核后，production prompt v4 只修 UE3：
静态/视差排除仍保留，但停车位或路边车在 oldest-to-newest 之间持续跨向车道边界或侵入
usable corridor 时仍是 UE3，即使最新帧看起来仍像停车姿态。UE1、UE5、UE6、INVALID
没有可证实的统一 prompt 问题，不改。v4 改变 production prompt hash，旧 adapter 会在加载
权重前被硬拒绝，必须重训。

这些边界来自 `keyframe_filter/ROAD_EVENT_CLASSIFICATION_PLAN.md`、
`ROAD_EVENT_RGB_AUDIT_ARCHIVE_202607.md` 和旧 Phase3 的
`EVAL_PROMPT_V2_V3_20260821.md`。数据构建仍要求每个 scenario/Town 至少已有一条
完整逐帧 RGB review，并默认排除 visual-risk 帧和异常时长 route。
本次指标、代表性错帧及“证据→prompt/代码”的逐项映射见
`PROMPT_V2_RGB_AUDIT_20260825.md`、`FUSION_2RGB_ENDPOINTS_AUDIT_20260827.md` 与
`V3_RETRAIN_RGB_AUDIT_20260829.md`。

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

默认四卡、显式单卡与显式四卡：

```bash
bash qwen3vl_local/sft_new_loop_phase2/train.sh
bash qwen3vl_local/sft_new_loop_phase2/train.sh single
GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase2/train.sh single

bash qwen3vl_local/sft_new_loop_phase2/train.sh ddp
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_new_loop_phase2/train.sh ddp
```

常用覆盖：

```bash
EVAL_STEPS=2000 \
GENERATION_EVAL_STEPS=2000 \
GENERATION_EVAL_BALANCE_COUNT=32 \
GENERATION_EVAL_MIN_UE3_TARGET_RECALL=0.625 \
SAVE_STEPS=20000 \
FOCUS_BALANCE_COUNT=1024 \
REGULAR_FOCUS_MULTIPLIER=2.0 \
HIGHWAY_REGULAR_FRACTION=0.25 \
bash qwen3vl_local/sft_new_loop_phase2/train.sh ddp
```

训练默认每 2000 optimizer step 跑 teacher-forced val 和固定均衡的自由生成 val，
每 20000 step 保存 checkpoint；generation eval 每个 UE/RE/invalid class 默认 32 条。
`best_generation` 优先要求 UE3 正类 target recall 达到 `0.625`，达标后按总 exact 选优；
若所有 step 都不达标，则明示保留 UE3 recall 最高的 fallback。训练会输出：

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

eval 同时写入非评分的 `answer_only_diagnostics`：它只严格解析输出开头的有序 YES/NO 行，
忽略后续 evidence 是否完整，用于区分“事件答案错误”和“证据行格式错误”。正式成绩仍是完整
字符串合同下的 `exact_match_accuracy`，不能用 answer-only 指标替代。

自由生成解析使用完整字符串合同：答案必须严格按当前 prompt 规定的行顺序输出，不能缺行、
重复、换序或夹带解释；audit 模式只额外允许同顺序的 `EVIDENCE_*` 行。任意格式违规都会让
整条 case 的 `format_valid` 和 semantic exact 同时失败。LoRA eval 在加载权重前还会硬校验
`production_prompt_sha256`、`history_rgb_mode` 和解析后的 `base_model_dir`；不兼容 adapter
不会只写 warning/metrics，而会直接中止。

## 5. 一键流程与 RGB 模式矩阵

```bash
# 默认 4RGB + 自动选择四张空闲 GPU
bash qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh

# 只用原 history 的首帧和最新帧
HISTORY_RGB_MODES=2rgb_endpoints \
  bash qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh

# 分别训练/评测 4RGB 与首尾 2RGB
HISTORY_RGB_MODES="4rgb 2rgb_endpoints" \
  bash qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh
```

未设置 `GPU_IDS` 时，所有 launcher 都会通过 `nvidia-smi` 覆盖选择空闲 GPU；显式设置
`GPU_IDS` 时按列表长度推断进程数。`eval.sh` 和 RGB mode matrix 默认分别通过
`EVAL_GPU_COUNT=4` / `DDP_GPU_COUNT=4` 自动选择四张卡，不再固定物理卡 0–3。

`run_full_pipeline.sh` 现在默认在训练后直接调用本目录 `eval.sh`，完成 base/LoRA 的
production 与 audit-prompt 测试、错例 RGB 抽样、全量 visual-risk 统计，并在
`${PIPELINE_ROOT}/<rgb_mode>/eval_review/` 生成硬上限 30MB 的 `.tar.gz` 审计包。
因此默认把旧的内联 `RUN_BASE_EVAL` / `RUN_LORA_EVAL` 关掉，避免重复评测；可用
`RUN_EVAL_SH=0 RUN_BASE_EVAL=1 RUN_LORA_EVAL=1` 恢复旧式分段调试。压缩包上限可通过
`BUNDLE_MAX_MB` 调整，默认仍为 30；超过上限时脚本会逐级减少/压缩 RGB，仍超限则硬失败。

指定 checkpoint 后，`eval.sh` 只从
`sft_new_loop_phase2_adapter_config.json` 读取 `history_rgb_mode`，不接受调用方覆盖。
压缩包文件名、顶层 README、bundle manifest、adapter 配置副本和四组 metrics 都保存该模式；
manifest 还保存 `history_rgb_count/history_rgb_selected_indices`，可直接区分 4RGB 的
`[0,1,2,3]` 与首尾 2RGB 的 `[0,3]`。包内重点保留 UE1/UE3/UE5/UE6、RE highway/local、
INVALID 联合 guard/子组指标、production/audit 差异、数据/视觉风险 manifest 和分桶错例 RGB，
不包含 adapter 权重、checkpoint 或 TensorBoard 大产物。

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
低能见度或 valid RE。每个 `audit_note.md` 已包含 newest-frame、标签可见性、模型/标签责任和
UE1/UE3、UE5、UE6 专项检查项，避免只按目录名确认错误。`eval.sh` 默认传
`--scan-frame-risks`，所以 bundle 中的 `visual_audit_manifest.json` 不再只是 coverage 摘要，
还会统计数据构建真正使用的 RS/EVENT review-risk 原因。
