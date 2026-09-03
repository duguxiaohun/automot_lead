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
施工、路边停车、队列车辆和 ego 视差不等于横向进入。对应重训 bundle 的 production
达到 `316/384=82.29%`，是当前严格可比 v2/v3/v4 中总 exact 与 audit exact 最优版本。
2026-08-29 曾在逐帧复核 v3 全部 68 个错例后试验 prompt v4，只放宽停车位/路边车持续
跨向车道边界的 UE3 判定；v4 虽将 UE3 recall 从 `71.88%` 恢复到 `79.69%`，但 production
降至 `308/384=80.21%`，UE6 recall 同时降至 `76.56%`。因此当前 production prompt 已按
总体最优口径回退为 v3；使用时必须搭配 prompt hash 一致的 v3 adapter，不能与 v4 adapter
混用。

这些边界来自 `keyframe_filter/ROAD_EVENT_CLASSIFICATION_PLAN.md`、
`ROAD_EVENT_RGB_AUDIT_ARCHIVE_202607.md` 和旧 Phase3 的
`EVAL_PROMPT_V2_V3_20260821.md`。数据构建仍要求每个 scenario/Town 至少已有一条
完整逐帧 RGB review，并默认排除 visual-risk 帧和异常时长 route。
本次指标、代表性错帧及“证据→prompt/代码”的逐项映射见
`PROMPT_V2_RGB_AUDIT_20260825.md`、`FUSION_2RGB_ENDPOINTS_AUDIT_20260827.md` 与
`V3_RETRAIN_RGB_AUDIT_20260829.md`。

## 2. 构建数据

如果只想直接执行下一轮正式实验，默认当前目录已经是 `AutoMoT/`，不需要分别运行本页后续命令：

```bash
bash qwen3vl_local/sft_new_loop_phase2/run_next_experiment.sh
```

该脚本会自动完成“缺失时构建数据 → 冻结集合预检 → 3 seed 训练 → validation 选优 →
一次性 unseen-456 验收”，支持训练中断后用同一条命令复用已完成 seed。若
`unseen_acceptance.json` 已存在，脚本只显示原结论并拒绝再次评测。默认正式实验 ID 固定为
`v3_frozen_3seed_unseen456_20260831`；只有确实需要新建独立实验时才显式设置
`EXPERIMENT_ID=<new_id>`，不要因 unseen 结果不理想而换 ID 重测。
历史 dev 集不再依赖未入库的 checkpoints audit bundle；仓库内
`frozen_dev_cases_v3_384.jsonl` 只保存 384 条 case 身份，不含 RGB、模型输出或权重，脚本会
用它与新建 index 做 `840/384/456` 交集硬校验。
如果三个 seed 都未通过选优，脚本不会打开 unseen，而会自动调用
`build_ue3_validation_rgb_audit.py`：按各 seed 的 fallback step 收集 UE3 假阴性，补齐 index
中的四帧 RGB，生成逐例原图、2×2 contact sheet、`audit_note.md`、汇总 JSON/MD 和 tar.gz。
任何 prompt/标签修改都必须先逐例填写该审计模板，不能按 scenario 名直接推断。
`v3_frozen_3seed_unseen456_20260831` 的 25 个 UE3 假阴性已经完成四帧人工复核，见
`V3_FROZEN_3SEED_UE3_VALIDATION_RGB_AUDIT_20260902.md`。该审计发现 32 条 UE3 小切片被
同一 route 的连续 9 帧明显过度加权，所以不能把原始 UE3 guard 失败直接归因为 prompt。

先不要重训。复用三个现有 `final` adapter、对每个 class 按 `(scenario, route_id)` 轮转抽取同一组
validation cases；所有 guard 通过时脚本才会继续一次性 unseen-456：

```bash
bash qwen3vl_local/sft_new_loop_phase2/run_route_diverse_validation_rescore.sh
```

只想复评、不自动打开 unseen 时：

```bash
RUN_UNSEEN=0 \
  bash qwen3vl_local/sft_new_loop_phase2/run_route_diverse_validation_rescore.sh
```

若复评已经跑完，只需从现有 `cases.jsonl` 生成全量 UE3 四帧 RGB 审计包，不重跑模型：

```bash
bash qwen3vl_local/sft_new_loop_phase2/run_ue3_full_validation_rgb_audit.sh
```

输出默认为 `ue3_route_diverse_full_rgb_audit/` 及同名 `tar.gz`。它会硬校验三个 seed
是否评估了完全相同的 UE3 case 身份，并导出全部 32 个正例的 TP/FN 矩阵、四帧原图、
contact sheet 和逐例填写模板。只看假阴性不足以修改 prompt；必须同时对比稳定答对的对照例。
当 `run_route_diverse_validation_rescore.sh` 新跑并且无 seed 过 guard 时，脚本也会在保持
unseen 未触碰的同时自动构建该审计包。

完成逐帧审计后，可用入库的 32-case 决策表生成诊断性子集指标：

```bash
bash qwen3vl_local/sft_new_loop_phase2/run_ue3_label_alignment_audit.sh
```

入口会优先使用正式 frozen experiment 下的
`checkpoints/sft_new_loop_phase2_frozen_protocol/v3_frozen_3seed_unseen456_20260831/ue3_route_diverse_full_rgb_audit/`；
若旧机器只保留了顶层副本，则按 32-case decisions 身份自动定位。也可以显式设置
`AUDIT_ROOT=<包含 manifest.jsonl 的目录>`。输出 `decision_rescore.json/md`。该结果明确标记 `official_metric=false`，只用于区分
模型责任与 PRE/POST/DOMAIN/2RGB/AMBIGUOUS 标签责任；不修改 frame index、不参与
checkpoint 选优、不能触发 unseen。完整结论见 `V3_ROUTE_DIVERSE_FULL_UE3_RGB_AUDIT_20260902.md`。

继续训练前运行 RGB decision × source rule × index route 分布的自动联表审计：

```bash
bash qwen3vl_local/sft_new_loop_phase2/run_ue3_label_alignment_audit.sh
```

该命令不加载模型、不改标签、不触碰 unseen。当前联表表明 32 帧都来自同一条
`event_dynamic_cutin_or_occupancy`，但该规则同时覆盖 VISIBLE_ACTIVE 与
PRE/POST/DOMAIN/2RGB/AMBIGUOUS，因此不能凭规则名或单一距离阈值自动重标。
数据构建只对 train 启用 route-round-robin 抽样；val/test 保留旧 sampler 以维持 frozen
身份。由于 UE3 是最小桶，数据 smoke 中 UE3 构成不变是预期现象，不能据此判定正式训练
目标 2048 下的重复权重已经改善。详见
`UE3_LABEL_ALIGNMENT_AND_ROUTE_DIVERSE_DATA_20260903.md`。
训练入口除 `TRAIN_ROUTE_DIVERSE=1` 外，只对 UE3 默认启用
`TRAIN_UE3_ROUTE_BALANCED=1`：原始桶耗尽后仍按 route 轮转，并以
`MAX_TRAIN_UE3_FRAME_REPEAT=10` 限制同一帧重复。其它类别不改变，
`balance/epoch_*.json` 保存实际 route 和 UE3 frame-repeat 分布。

完成联表后，用一条 CPU-only 命令重建到新目录并自动比较新旧 index；它不会训练：

```bash
bash qwen3vl_local/sft_new_loop_phase2/run_route_diverse_data_smoke.sh
```

该脚本的 `passed: true` 只证明 index/frozen split 合同成立。随后还必须运行真正对应
训练目标 2048 的 CPU sampling smoke：

```bash
bash qwen3vl_local/sft_new_loop_phase2/run_ue3_train_route_balance_smoke.sh
```

它不加载模型、不改 prompt/标签/index、不触碰 unseen；输出到
`checkpoints/ue3_train_route_balance_smoke/`。只有所有 guard 为 true 才运行单 seed pilot：

```bash
GPU_IDS=0,1,2,3 \
HISTORY_RGB_MODE=2rgb_endpoints \
INDEX=checkpoints/sft_new_loop_phase2_data/frame_index.jsonl \
OUTPUT_DIR=checkpoints/sft_new_loop_phase2_ue3_route_balance_pilot/seed_20260810 \
SEED=20260810 MAX_STEPS=4000 SAVE_STEPS=4000 \
FOCUS_BALANCE_COUNT=2048 TRAIN_ROUTE_DIVERSE=1 \
TRAIN_UE3_ROUTE_BALANCED=1 MAX_TRAIN_UE3_FRAME_REPEAT=10 \
GENERATION_EVAL_ROUTE_DIVERSE=0 \
bash qwen3vl_local/sft_new_loop_phase2/train.sh ddp
```

这里显式关闭 generation route-diverse，复用旧 seed 20260810 的 validation sampler；
因此这轮唯一实验变量是 UE3 训练采样。不要直接启动三 seed，也不要打开 unseen-456。

### 下游自动 A/B（独立可选实验，非当前 Phase2 排障路径）

完整逐帧 RGB 审计不支持继续修改 v3 prompt；三个 seed 也都未通过 UE3 guard，因此
不再重训本 Phase2，且不打开 unseen-456。只保留 seed 20260810 作为研究候选，测试它的
LoRA 是否能改善 LeadMoT 规划。整个下游流程不需要人工复核：

```bash
# 1. 核对 model/dataset/adapter/prompt/hash/2RGB/seed 合同，不加载 GPU；
#    若默认 LeadMoT JSONL 不存在，会先从 lead_data 自动构建一次
bash qwen3vl_local/sft_new_loop_phase2/run_leadmot_qwen_ab.sh preflight

# 2. 两臂各 2 step + 同 8 case eval，只检查端到端链路
GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase2/run_leadmot_qwen_ab.sh smoke

# 3. smoke 通过后，base 与 LoRA 分别从同 seed 训练 decoder，再做全量 paired eval
GPU_IDS=0,1,2,3 TRAIN_LAUNCH_MODE=ddp \
  bash qwen3vl_local/sft_new_loop_phase2/run_leadmot_qwen_ab.sh train
```

默认 LoRA 是 frozen protocol 的 `seed_20260810/fallback_generation`（validation 选中的
step 4000 研究候选）；不是训练结束于 step 10752 的 `final/`。`fallback_generation.json`
与该目录下 adapter config 的 `global_step` 都必须为 4000。它没有通过 UE3 production guard，
本 A/B 只回答“该视觉表征是否改善下游规划”，不能据此把 Phase2 晋升为 production。
adapter 搬家时用 `ADAPTER_DIR=...` 覆盖，但仍会校验 step/prompt/hash/2RGB/seed 和实际权重 SHA256。
默认规划索引是 `checkpoints/leadmot_v1_data/{train,val}.jsonl`。两者都不存在时脚本会从
`DATA_ROOT=lead_data` 自动调用 `leadmot/build_dataset.py --no-with-subgoal-fields`：保留所有
合法 anchor、按 route 切 train/val，并先剔除异常时长 route。只有一个文件存在时会中止，
避免静默混用两次构建的 split；不希望自动构建可设 `AUTO_BUILD_LEADMOT_DATASET=0`。
当前收口 A/B 固定 `USE_SUBGOAL=0`，这样获胜 checkpoint 才能进入 CARLA；`USE_BEV` 默认 1。
每个 LeadMoT checkpoint 都绑定 base/adapter 真实 SHA256，eval/CARLA 自动恢复并拒绝错配。
`comparison.json/md` 对同一 validation case 严格配对，并按 route 做 cluster bootstrap；
只有 route/waypoint 的 ADE/FDE 四项 95% CI 上界全部小于 0，才写
`carla_allowed=true`。否则实验在离线阶段停止，不再改 Phase2 prompt 或继续训练。
若只想重跑一轮已完成实验的 eval，必须把原目录显式传回，避免时间戳生成空目录：

```bash
AB_ROOT=checkpoints/leadmot_qwen_adapter_ab/<原实验目录> \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_new_loop_phase2/run_leadmot_qwen_ab.sh eval
```

复评输出写入原实验目录的 `route_diverse_validation_rescore/`，原 frozen 指标不覆盖。训练期
generation validation 后续也默认采用 route-diverse 采样，并使用与训练 seed 无关的固定
`GENERATION_EVAL_SAMPLING_SEED=20260831`，保证不同 seed 真正比较同一 validation case 集。
历史独立 eval 为保持可比性仍默认关闭该选项；需要新口径时显式设
`ROUTE_DIVERSE_SAMPLING=1` 或传 `eval.py --route-diverse-sampling`。
脚本坚持离线运行，默认要求本地模型已经位于 `checkpoints/Qwen3-VL-4B-Instruct`；若训练机
使用其它本地路径，可在脚本开头的 `MODEL_DIR` 默认值处统一修改一次。

下面是各阶段的独立命令，主要用于排障；这些独立命令从 `AutoMoT/` 目录运行。

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
GENERATION_EVAL_MIN_UE6_TARGET_RECALL=0.80 \
GENERATION_EVAL_MIN_INVALID_EXACT=0.80 \
GENERATION_EVAL_MIN_APPLICABLE_REGULAR_EXACT=0.50 \
SAVE_STEPS=20000 \
FOCUS_BALANCE_COUNT=1024 \
REGULAR_FOCUS_MULTIPLIER=2.0 \
HIGHWAY_REGULAR_FRACTION=0.25 \
bash qwen3vl_local/sft_new_loop_phase2/train.sh ddp
```

训练默认每 2000 optimizer step 跑 teacher-forced val 和固定均衡的自由生成 val，
每 20000 step 保存 checkpoint；generation eval 每个 UE/RE/invalid class 默认 32 条。
`best_generation` 必须同时满足 UE3 recall `>=0.625`、UE6 recall `>=0.80`、INVALID
exact `>=0.80` 与 applicable RE exact `>=0.50`，达标后再按总 exact 选优。未全部达标的
权重只写入 `fallback_generation/`，不会被 full pipeline 自动当作 production checkpoint；fallback
先按通过门槛数、最差归一化达标比例和总 exact 排序，避免只救一个问题类。训练会输出：

- `train_balance.json`：class、question domain、true RS、highway/local RE，以及 invalid 四维子类别采样；
- `balance/epoch_*.json`：每轮 invalid 的 source class、true RS、错误问题域、联合签名与均衡 guard；
- `train_eval_metrics.jsonl` / generation records；
- `tb/`：loss、每个问题准确率、invalid joint/all-UE-NO、invalid 四维子类别、highway/local/UE slice；
- `best_val/`、通过全部门槛时的 `best_generation/`、仅诊断用的 `fallback_generation/`、
  `generation_selection_status.json`、`checkpoint-*`、`final/`；
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

### 4.1 冻结 prompt、多 seed 与一次性 unseen 验收

当前停止继续改 prompt，固定 v3 名称与 production hash
`cd564634257fe0f072de70947200a820d6dd2b43375981b60120a1fe2296dd7f`。标准实验先训练
3 个 seed，只使用 validation 和上述四项门槛选一个正式 checkpoint；随后从 840 条 test 中
精确排除历史 384 条 dev/audit cases，对剩余 456 条一次性验收：

```bash
# 只训练 3 个 seed，并按 validation 选优；不读取 test/unseen 指标
bash qwen3vl_local/sft_new_loop_phase2/run_frozen_protocol.sh train

# 确认 seed_selection.json 后，只执行一次 456 条 unseen 验收
EXPERIMENT_ID=<与训练相同的实验 id> \
bash qwen3vl_local/sft_new_loop_phase2/run_frozen_protocol.sh unseen
```

也可用 `all` 连续执行；生产实验更推荐分成两条命令，中间先核对所有 seed 的
`generation_selection_status.json`。默认 unseen 硬门槛为总 exact/UE3 recall/UE6 recall/
INVALID recall 分别 `0.80/0.80/0.80/0.80`、format valid `1.0`、applicable RE exact
`0.50`。结果写入 `unseen_acceptance.json`；文件已存在时脚本拒绝重跑，防止依据 unseen
结果继续调 prompt。只有基础设施失败才可显式 `ALLOW_UNSEEN_RERUN=1`。

底层 `eval.py` 也支持重复传 `--exclude-cases-jsonl <file-or-eval-dir>`，并通过
`--expected-excluded-cases 384 --expected-total-cases 456` 在模型加载前硬校验冻结集合。

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
