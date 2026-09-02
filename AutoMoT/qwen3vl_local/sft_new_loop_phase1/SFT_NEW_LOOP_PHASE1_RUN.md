# SFT New Loop Phase1 运行手册

本目录把 `sft_loop_phase1` 的四个可见事实问题和
`sft_loop_phase2_augment` 的 ROAD_STRUCTURE 三类增强问法融合到同一轮
YES/NO 问答里。每个样本固定包含 Phase1 四问：

- `HIGHWAY`
- `STATIC_OBSTACLE`
- `VULNERABLE`
- `TRAFFIC_LIGHT_ABNORMAL`

同时按 Phase2 最新增强方式加入 ROAD_STRUCTURE 问题：

- `all_random_order`：随机顺序回答 `RS1/RS2/RS4/RS5` 全部四问。
- `subset_random`：随机回答 1/2/3 个 RS 子问题。
- `hierarchical_probe`：先问 `RS_HIGHWAY` 和 `GROUP`，再问一个具体 RS 细节。

训练使用 Phase2 augment 的 `4:1:1` 比例，eval / generation eval 使用 `2:1:1`。
`RS_HIGHWAY` 是 Phase2 hierarchical 里的高速/R3 判断，和 Phase1 审计过的
`HIGHWAY` 是两个不同输出行，不要混淆。

正式训练默认 `FOCUS_BALANCE_COUNT=9216`，即每轮 147,456 个 sampled case，
和旧 `sft_loop_phase2_augment` 的每轮训练量对齐。八个主 focus 问题全部保持
YES:NO = 1:1，Phase1 四问与 Phase2 四问总量也是 1:1。默认还会检查
`MAX_TRAIN_FRAME_REPEAT=10`，任一帧在单轮采样里复用超过上限会在加载模型前中止。

从 2026-08-27 的 v4 修订开始，`focus` 不写入 prompt，但会决定主任务语义 loss：当前
focus 的 YES/NO 值 token 使用 `1.0` 基础权重；同一输出里的其它主答案默认使用
`NON_FOCUS_SEMANTIC_LOSS_WEIGHT=0.1`；hierarchical 的 `RS_HIGHWAY/GROUP` 派生值仍使用
`1.0`。类别权重不是按原始行数计算，而是按上述基础权重累积出的有效语义质量分别平衡
YES/NO，避免 0.1 副行中的自然 NO 重新压过 1:1 focus 监督。所有输出行的字段、冒号、换行
和结束符仍使用 `FORMAT_LOSS_WEIGHT`。如需复现实验前的 focus-only 合同，可显式设置
`NON_FOCUS_SEMANTIC_LOSS_WEIGHT=0`。

这一折中来自 2026-08-27 的真实 RGB 错例审计：两套 1024-case eval 中，约 18% 样本是
focus 行正确但其它请求行仍有错；但又观察到 Phase1 HIGHWAY 与 Phase2 RS 标签存在少量
独立标注冲突，因此没有恢复 v1 的“所有答案值都以 1.0 反传”。每个 balance JSON 会记录
`emitted_answer_counts`、`semantic_answer_counts`、`semantic_answer_base_mass`、
`semantic_class_weights` 和 `non_focus_semantic_loss_weight`，应重点检查有效质量与类别权重，
不能只看原始答案行数。

当前 prompt 名为
`sft_new_loop_phase1_phase1_phase2_combined_v5_rgb_audited_rs4_hardware`。v5 完整保留 v4 的
Phase1/Phase2 决策规则、RS_HIGHWAY RGB 边界和 audit 输出合同，只根据 2026-08-29 的
2RGB RS4 错帧增加一条硬件辨识边界：装饰/向下照明路灯、裸杆/灯臂和车辆灯光不是可识别的
交通信号头，不能单独触发 RS4。没有根据灯异常标签回退去放宽
`TRAFFIC_LIGHT_ABNORMAL`；逐帧 RGB 显示这批回退多数只有正常红/绿相位或根本没有可读信号头。
完整结果与逐帧记录见 `FUSION_V4_4RGB_2RGB_RESULT_RGB_AUDIT_20260829.md`。
2026-08-31 对 v5 两个正式 bundle 的二次逐帧核查表明：raw exact 整体提升，但 RS4 收益中
混有 RGB 与 GT 冲突，2RGB 灯异常和 4RGB vulnerable 的部分回退也来自不可见/冲突标签。
因此当前保留 v5，不继续堆提示词；完整对比、逐帧 case 归因和 balanced 补测决策见
`FUSION_V5_4RGB_2RGB_RESULT_RGB_AUDIT_20260831.md`。

### 冻结基线（2026-09-02）

2026-09-02 的同协议重复训练再次确认：2RGB 联合 exact 为 76.074%，相对上一轮 76.562%
净退化 5/1024；4RGB 为 73.047%，相对上一轮 77.930% 净退化 50/1024。新旧同 mode 的
prompt/hash、1024 个 case 和 base 逐例输出完全一致，且代表性错例已经逐帧查看 RGB，因此本轮
变化归因于 adapter/checkpoint 波动，不归因于 v5 prompt。

从本节起冻结以下 production 基线：

- prompt 固定为当前 v5，不创建 v6，也不回退 v4；
- 质量优先使用 `run_20260829_235223_4rgb_combined_phase1_phase2_4rgb/best_generation`
  的 step-40000（strict joint exact 77.930%）；
- 显存/时延优先使用
  `run_20260829_235241_2rgb_endpoints_combined_phase1_phase2_2rgb_endpoints/best_generation`
  的 step-40000（strict joint exact 76.562%）；
- `best_generation_balanced` 保留为诊断候选，不晋升为 production；
- 不再根据当前固定 1024-case test 的零散错误迭代 prompt。只有独立 unseen 集上的可复现退化、
  至少两个 seed 的同类回退、逐帧 RGB 证明的系统性规则缺失，或实现合同 bug 才能解冻。

完整指标、配对翻转、RGB 证据、核心代码 SHA256 指纹和解冻条件见
`FUSION_V5_REPEAT_RUN_AUDIT_AND_FREEZE_20260902.md`。

v4 相对 v3 的既有修订继续保留：Phase1 四行按 case seed 可复现随机排列；
`hierarchical_probe` 渲染独立 `RS_HIGHWAY` 定义，要求 limited-access 拓扑链，并把双黄线、
普通双向城际道路、黑暗、雾、单一护栏列为不足证据；audit prompt 强制每行保持
`EVIDENCE_<ANSWER_KEY>:` 前缀且证据开始后不再重复答案行。对应旧审计见
`FUSION_V3_4RGB_2RGB_ERROR_AUDIT_20260827.md`。所有旧 adapter 的 prompt fingerprint 与 v5
不兼容，必须重训后评测，不能把旧权重强行挂到新 prompt 上。

训练代码仍保存联合 exact 最优的 `best_generation/`，同时额外保存
`best_generation_balanced/`：在 generation format gate 通过后，先最大化八个 focus 中的
最低 accuracy，再比较联合 exact 和 focus macro accuracy。它不替换历史目录；正式 test 应
同时比较“最高联合 exact”和“最弱任务受保护”两个 checkpoint，避免单一联合 exact 掩盖某个
任务的回退。该流程用于研究性新实验；冻结 production 不会因为新 run 自动改指向，只有满足
上面的解冻条件才重新晋升权重或修改协议。

单独调用 `eval.sh` 评测 balanced 候选时，仍必须显式传 adapter 子目录；run root 默认优先
解析历史 `best_generation/`：

```bash
ADAPTER_DIR=checkpoints/sft_new_loop_phase1_runs/<run>/best_generation_balanced \
  bash qwen3vl_local/sft_new_loop_phase1/eval.sh
```

`run_full_pipeline.sh` 则默认设置 `RUN_BALANCED_EVAL=1`：primary 为
`best_generation/` 且同一 run 下存在不同权重的 `best_generation_balanced/` 时，会在 primary
审计包之后自动生成独立的 `eval_review_balanced/` 与 `*_balanced_audit_bundle.tar.gz`。若两个
目录的 adapter 权重逐字节相同则自动跳过，避免重复评测；需要临时只测 primary 时显式设置
`RUN_BALANCED_EVAL=0`。这一流程只扩大正式候选覆盖，不改变训练选优或覆盖
`best_generation/`。

2026-08-31 的同集 2RGB 补测已经比较 `best_generation/step-40000` 与
`best_generation_balanced/step-28000`：balanced 把最弱 focus accuracy 从 78.906% 提到
81.250%，并明显改善 `TRAFFIC_LIGHT_ABNORMAL` 和 RS1，但联合 strict exact 从 76.563% 降到
75.586%，Phase1/Phase2 exact 也分别下降 1.172/1.074 pp，STATIC、VULNERABLE、RS2、RS5
同时回退。逐帧 RGB 还确认 balanced 的 STATIC 误报增加和部分 vulnerable 小目标漏检是真实问题。
因此当前 production 仍使用 `best_generation/step-40000`，balanced 只作为诊断候选；完整配对
指标与逐帧归因见 `FUSION_V5_4RGB_2RGB_RESULT_RGB_AUDIT_20260831.md` 第 8 节。

数据构建沿用 Phase2 最新过滤：剔除异常时长 route、检查 full-frame RGB review 覆盖，
默认排除 visual-risk 帧。Phase1 标签来自已审计四问答案表；Phase2 标签来自逐帧 RS
标注。视觉子组覆盖只允许来自结构化 RGB audit notes / annotations，不能从自由文本
`audit_evidence` 推断 route 标签。

## 0. 目录与产物

以下命令都从 `AutoMoT/` 目录运行，路径不要再加 `AutoMoT/` 前缀。
所有 shell 入口不额外指定时都默认四卡。`train.sh` 和 `run_full_pipeline.sh`
默认自动选 4 张空闲卡；`eval.sh` 和 `run_rgb_mode_matrix.sh` 默认使用
`GPU_IDS=0,1,2,3`。需要单卡时显式传 `GPU_IDS=0`，训练还要显式传 `single`
或 `check`。

常用产物路径：

- 数据集：`checkpoints/sft_new_loop_phase1_data/frame_index.jsonl`
- 训练 run：`checkpoints/sft_new_loop_phase1_runs/run_<RUN_TAG>_combined_phase1_phase2_<rgb_mode>/`
- check run：`checkpoints/sft_new_loop_phase1_runs/check_<RUN_TAG>_combined_phase1_phase2_<rgb_mode>/`
- 一键 eval：`checkpoints/sft_new_loop_phase1_eval_review/<timestamp>/`
- 一键 pipeline：`checkpoints/sft_new_loop_phase1_pipeline/<timestamp>/`
- RGB 模式矩阵：`checkpoints/sft_new_loop_phase1_eval_matrix/<timestamp>/`
- 错例抽样：`checkpoints/sft_new_loop_phase1_audit_samples/`
- RGB 覆盖证明：`qwen3vl_local/sft_new_loop_phase1/phase2_rgb_audit_coverage.json`

## 1. 构建数据集

先生成视觉审计 manifest，再构建 fused index：

```bash
python qwen3vl_local/sft_new_loop_phase1/visual_audit.py
python qwen3vl_local/sft_new_loop_phase1/build_dataset.py
```

如果远程机器的数据根目录不同，只设置 `DATA_ROOT` 或显式传 `--data-root`：

```bash
python qwen3vl_local/sft_new_loop_phase1/build_dataset.py \
  --data-root /path/to/lead_data
```

`build_dataset.py` 默认把 `history_rgb_paths` 保存为相对 `--data-root` 的路径。
训练和评测会用 `--data-root` 解析相对路径，也能重映射旧 JSONL 里包含 `lead_data`
的绝对路径。若旧数据是在别的机器上构建的，长期方案是重新运行 `build_dataset.py`，
不要手工替换接近 1GB 的 JSONL。

构建后重点检查：

- `checkpoints/sft_new_loop_phase1_data/visual_audit_manifest.json` 已生成。
- `manifest.json` 的 train / val / test 都非空。
- train / val / test 都有 16 个非空 focus 桶。
- visual-risk 过滤数量符合预期。
- Town12 自由文本 HIGHWAY override 计数为 0。
- 随机抽查 RGB 路径能通过 `--data-root` 找到真实文件。

## 2. 快速 check

先跑小规模链路，确认数据、模型加载、loss、DDP 同步分支没有明显错误：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase1/train.sh check
```

常用远程数据根：

```bash
DATA_ROOT=/path/to/lead_data GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase1/train.sh check
```

check 模式默认只跑很少 step，但仍从完整 index 里构造均衡桶，不靠文件前几行取样。
每次 check 默认写入带时间戳的新目录，并更新
`checkpoints/sft_new_loop_phase1_runs/check_latest`，避免重复 smoke 混用旧 JSONL/TB。

## 3. 正式训练

单卡：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase1/train.sh single
```

四卡 DDP：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_new_loop_phase1/train.sh
```

`train.sh` 不传模式时默认 `ddp`，也就是默认四卡；如果只想单卡训练，需要显式传
`single`。
直接运行 `train.sh` 会把 stdout/stderr 同步写到当前 run 的 `train.log`；通过
`run_full_pipeline.sh` 启动时，外层还会额外记录 `pipeline.log`。

两帧端点输入对照：

```bash
HISTORY_RGB_MODE=2rgb_endpoints GPU_IDS=0,1,2,3 \
  bash qwen3vl_local/sft_new_loop_phase1/train.sh ddp
```

关键默认值：

- `FOCUS_BALANCE_COUNT=9216`
- `MAX_TRAIN_FRAME_REPEAT=10`
- `NON_FOCUS_SEMANTIC_LOSS_WEIGHT=0.1`
- `NUM_EPOCHS=3`
- `EVAL_STEPS=2000`
- `GENERATION_EVAL_STEPS=2000`
- `GENERATION_EVAL_BALANCE_COUNT=16`
- `SAVE_STEPS=20000`
- `WARMUP_STEPS=2000`
- `GRAD_ACCUM=1`
- `HISTORY_RGB_MODE=4rgb`

如果 generation validation 在最佳点后连续下降，可以像 2026-08-24 的 v1 训练一样主动
停止并使用 `best_generation/`。当时 24k/26k/28k exact 为
`81.64% / 81.25% / 75.00%`，在 29,090 step 停止是过拟合控制，不是训练故障。

如果显式把 `GRAD_ACCUM` 调大，teacher eval、generation eval 和 checkpoint 会在达到
触发 step 后延迟到下一次 optimizer step 执行，避免保存尚未应用当前累积梯度的 adapter。
延迟保存的周期 checkpoint 名称会写成 `checkpoint-<trigger_step>-applied-<step>`。

四卡 DDP 下，generation eval 只在 rank0 上做自由生成。该阶段可能超过 PyTorch/NCCL
默认 600 秒 watchdog，因此训练器会让其它 rank 通过 `.dist_sync/` 文件轮询等待
rank0 完成，再进入一次短 DDP barrier。默认 `DDP_TIMEOUT_SECONDS=7200`、
`GENERATION_EVAL_SYNC_TIMEOUT_SECONDS=7200`；如果远程机器自由生成更慢，可以显式调大。

训练产物会写入 `checkpoints/sft_new_loop_phase1_runs/`，非 check run 会更新
`checkpoints/sft_new_loop_phase1_runs/latest` 软链接。

## 4. base Qwen 评测

production prompt：

```bash
GPU_IDS=0 python qwen3vl_local/sft_new_loop_phase1/eval.py \
  --history-rgb-mode 4rgb \
  --split test \
  --cases-per-bin 64
```

audit prompt：

```bash
GPU_IDS=0 python qwen3vl_local/sft_new_loop_phase1/eval.py \
  --history-rgb-mode 4rgb \
  --audit-prompt \
  --split test \
  --cases-per-bin 64
```

audit parser 会先匹配 `EVIDENCE_*`，再匹配普通答案行；因此
`EVIDENCE_HIGHWAY: NO readable ramp cue` 不会再被误当成未知 answer key。parser 仍然严格：
重复/缺失 answer 会影响答案语义；未知/缺失/重复 evidence 或其它额外行会使 audit contract
invalid，但 `eval.py` 会把 `exact_match_accuracy` 与 `audit_parser_diagnostics` 分开统计，不再
用 evidence 格式错误清空已经合法的答案。

四卡评测：

```bash
GPU_IDS=0,1,2,3 torchrun --nproc_per_node=4 \
  qwen3vl_local/sft_new_loop_phase1/eval.py \
  --history-rgb-mode 4rgb \
  --split test \
  --cases-per-bin 64
```

正式 eval 的采样合同是八个主问题各自 YES:NO = 1:1，eval variant 比例为
`all_random_order/subset_random/hierarchical_probe = 2:1:1`。

## 5. LoRA 一键评测

`eval.sh` 会解析 run 目录或 adapter 目录，然后依次运行：

1. base production
2. base audit-prompt
3. LoRA production
4. LoRA audit-prompt
5. production 错例 RGB 抽样
6. 小型审计 tar 包

只要指定 checkpoint/run，`eval.sh` 就强制从其中的
`sft_new_loop_phase1_adapter_config.json` 读取 `history_rgb_mode`，调用时不再设置
`HISTORY_RGB_MODE`。配置缺字段或不是 `4rgb/2rgb_endpoints` 会直接失败。最终压缩包名称、
`BUNDLE_README.md`、`bundle_manifest.json`、adapter 配置副本和各组 metrics 都记录模式；
manifest 还记录 `history_rgb_count` 与 `history_rgb_selected_indices`，所以下载后可以明确区分
四帧 `[0,1,2,3]` 和首尾两帧 `[0,3]` 的结果。

推荐传 run 根目录：

```bash
bash qwen3vl_local/sft_new_loop_phase1/eval.sh \
  checkpoints/sft_new_loop_phase1_runs/latest
```

也可以直接传 adapter：

```bash
ADAPTER_DIR=checkpoints/sft_new_loop_phase1_runs/latest/final \
  bash qwen3vl_local/sft_new_loop_phase1/eval.sh
```

常用覆盖：

```bash
GPU_IDS=0,1,2,3 CASES_PER_BIN=64 AUDIT_PER_TARGET=8 \
  bash qwen3vl_local/sft_new_loop_phase1/eval.sh \
  checkpoints/sft_new_loop_phase1_runs/latest
```

如果只想直接跑 `eval.py`，不打包：

```bash
GPU_IDS=0 python qwen3vl_local/sft_new_loop_phase1/eval.py \
  --adapter-dir checkpoints/sft_new_loop_phase1_runs/latest/final \
  --split test \
  --cases-per-bin 64
```

## 6. 错例 RGB 抽样

`eval.sh` 默认已经调用错例抽样。若要单独从某个 eval 目录抽样：

```bash
python qwen3vl_local/sft_new_loop_phase1/audit_eval_cases.py \
  --eval-dir checkpoints/sft_new_loop_phase1_eval_review/<timestamp>/lora_production \
  --output-dir checkpoints/sft_new_loop_phase1_audit_samples/<tag> \
  --data-root lead_data \
  --per-target 12 \
  --overwrite
```

脚本会覆盖以下 fused 错误类型：

- Phase1 四问的 false positive / false negative。
- Phase2 `RS1/RS2/RS4/RS5` 的 false positive / false negative。
- hierarchical 的 `RS_HIGHWAY` 和 `GROUP` 错误。
- 输出格式非法。
- Phase2 多个 RS 同时 YES。
- subset 模式下额外输出未要求的问题行。

## 7. TensorBoard

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_new_loop_phase1_runs/latest
```

重点看：

- `train/loss`
- `train/learning_rate`
- `train/focus/*`
- `train/variant/*`
- `train/augment/*`
- `val/value_token_acc`
- `val_generation/exact_accuracy`

## 8. RGB 模式矩阵

`run_rgb_mode_matrix.sh` 用于做 phase2 风格的 4rgb / 2rgb_endpoints 对照：

```bash
bash qwen3vl_local/sft_new_loop_phase1/run_rgb_mode_matrix.sh
```

它会顺序执行 10 步：

1. base 4rgb production
2. base 4rgb audit-prompt
3. base 2rgb_endpoints production
4. base 2rgb_endpoints audit-prompt
5. 训练 4rgb LoRA
6. LoRA 4rgb production
7. LoRA 4rgb audit-prompt
8. 训练 2rgb_endpoints LoRA
9. LoRA 2rgb_endpoints production
10. LoRA 2rgb_endpoints audit-prompt

常用覆盖：

```bash
GPU_IDS=0 DATA_ROOT=/path/to/lead_data CASES_PER_BIN=64 \
  bash qwen3vl_local/sft_new_loop_phase1/run_rgb_mode_matrix.sh
```

## 9. 一键全流程

`run_full_pipeline.sh` 是按 phase3 的一键流程形式适配到 fused Phase1+Phase2 的入口。
默认流程为：

1. 生成 `visual_audit_manifest.json`
2. 构建 fused 数据集
3. 训练 LoRA
4. 调用本目录 `eval.sh` 跑 primary 的 base/LoRA production 与 audit-prompt eval
5. 对 base 和 LoRA production 错例做 RGB 抽样
6. 按 checkpoint 的 RGB mode 生成硬上限 30MB 的审计压缩包
7. 若 `best_generation_balanced/` 与 primary 权重不同，默认再独立评测并打包 balanced 候选

默认只跑 `4rgb`：

```bash
bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
```

只跑首帧+最新帧的 `2rgb_endpoints`：

```bash
HISTORY_RGB_MODES=2rgb_endpoints \
  bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
```

同时跑 4rgb 和 2rgb_endpoints：

```bash
GPU_IDS=0,1,2,3 HISTORY_RGB_MODES="4rgb 2rgb_endpoints" \
  bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
```

跳过训练，评测已有 adapter：

```bash
RUN_BUILD=0 RUN_TRAIN=0 \
ADAPTER_DIR=checkpoints/sft_new_loop_phase1_runs/latest/best_generation \
GPU_IDS=0,1,2,3 \
  bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
```

只构建数据，不训练不评测：

```bash
RUN_EVAL_SH=0 RUN_BASE_EVAL=0 RUN_TRAIN=0 RUN_LORA_EVAL=0 RUN_AUDIT_CASES=0 \
  bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
```

常用变量：

- `PIPELINE_TIMESTAMP`：控制 pipeline 输出目录名。
- `PIPELINE_ROOT`：覆盖 pipeline 输出根目录。
- `HISTORY_RGB_MODES`：空格分隔的 RGB 模式列表。
- `RUN_VISUAL_AUDIT/RUN_BUILD/RUN_TRAIN/RUN_EVAL_SH/RUN_BASE_EVAL/RUN_LORA_EVAL/RUN_AUDIT_CASES`
  控制各阶段开关。
- `RUN_EVAL_SH=1` 为默认值；此时旧内联 base/LoRA eval 默认关闭，避免把相同模型重复测两遍。
- `RUN_BALANCED_EVAL=1` 为默认值；只在 primary 是 `best_generation/`、balanced adapter
  存在且权重不同时追加完整 balanced 评测。设为 `0` 可关闭。
- `BUNDLE_MAX_MB=30` 是默认硬上限。包内保留四组 metrics/报告、截断后的 case JSONL、
  Phase1/Phase2 专项采样诊断、adapter 与数据 manifest，以及按错误桶抽样压缩的真实 RGB；
  不复制权重、checkpoint 或 TensorBoard 大产物。
- `TRAIN_FOCUS_BALANCE_COUNT`：默认继承 `FOCUS_BALANCE_COUNT=9216`。
- `TRAIN_MAX_FRAME_REPEAT`：默认继承 `MAX_TRAIN_FRAME_REPEAT=10`。
- `BASE_CASES_PER_BIN/LORA_CASES_PER_BIN`：默认 64。
- `AUDIT_PER_TARGET`：默认 12。

## 10. 均衡与审计合同

训练采样分两层：

- Phase1 四个 focus 问题各自 YES:NO = 1:1。
- Phase2 四个 focus 问题 `RS1/RS2/RS4/RS5` 也各自 YES:NO = 1:1。

在这个前提下，训练保持：

- Phase1 focus 总量 = Phase2 focus 总量。
- 三种 Phase2 variant 总量为 `4:1:1`。
- `all_random_order/RS*:YES|NO` 桶为硬约束。
- Phase2 `(focus_bucket, variant)` 配额为硬约束。
- subset / hierarchical 具体 `augment_balance_key` 使用目标驱动近似均衡，并写入偏差报告。

训练和评测会记录：

- `manifest.json`
- `train_balance.json`
- `balance/epoch_*.json`
- `balance/epochs.jsonl`
- `train_run_manifest.json`
- `train_metrics.jsonl`
- `train_eval_metrics.jsonl`
- eval `metrics.json`
- `augment_target_deviation`
- `all_random_order_target_deviation`
- `phase2_focus_variant_*`
- `repeat_audit`
- per-question / per-variant / pattern diagnostics

正式训练前建议看：

- `augment_target_deviation.max_abs_delta`
- `augment_target_deviation.total_abs_delta`
- `repeat_audit.max_repeat`
- `repeat_audit.mean_repeat`
- 八个 focus 是否全部 exact。
- 三种 variant 是否 exact。
