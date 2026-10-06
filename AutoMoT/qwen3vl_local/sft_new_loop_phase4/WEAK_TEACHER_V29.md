# v29：可运行的弱监督实验、实例限制证据与 STOP 台账

2026-10-05。合同 v29/data_v29、老师 v4；严格批准政策仍是 v2。新数据、新索引、新 run；历史 hash 不可改写复用。

## 结论与策略选择

复核 v28 的 109 条独立路线原始输出，2 图共 12 个规则类，**即使二元判断全对，也没有一个类能满足现行严格批准门槛**。已经新增 `best_case_feasibility`，在审核卡统计中给出判断数、不同路线数及转移附近判断的缺额，避免把题源不足误写成“只等审核”。完整旧池上界复核见本版验证 JSON。

本版采用“保留严格批准 + 显式弱监督实验”。没有合并语义不同的 proceed/recover_follow，也没有用全对样本的退化 bootstrap 区间宣称高置信度。弱监督本来就不等于经过独立认证的标签，应能单独建库、训练、测误差。

- `causal_teacher_class_approval_v2`：沿用独立审核与数值门槛；真实 `teacher_registry_v4.json` 为空。
- `causal_teacher_weak_experiment_v1`：必须显式命名实验并列出版本绑定的规则类；仅允许 train。输出 `label_basis=weak_rule_teacher`、`reference_kind=rule_teacher`，包含原始事实、源码、图片和因果来源证明。UNKNOWN、留出路线、未解决的已登记风险不放行。
- 构建和加载都验证该区分。弱监督不进入 approved 类清单，不能让 `formal_data_ready` 变真。现有评估继续把老师一致率与人工准确率分开。
- 自动选择弱监督训练的 `event_edge_answer` 采样政策；原人工库默认 `event_equal` 保持。可以用配置显式做对照，实际政策写入训练配置和采样恢复状态。

## 规则修复

### 实例必须先确有本事件限制

创建 Episode 前，当前可见参与者必须提供本事件自身的限制证据；红灯本身或无关车辆造成的 NO 不足以证明该参与者引发事件。实例记录限制发生帧。此次真实回放中，Town04 3_26 那个首帧就释放的 UE1 不再生成任何问题。

### 停车位置、连续停稳与横穿间隙分开

新增 `teacher_controls.StopDutyTracker`：连续观察同一个适用 STOP 的区域、车道身份、自车速度和车头相对区域位置，满足后保存带源 SHA 的因果证明；换 ID/车道、观测断档、无效几何会使证明失效。历史证明可以早于当前 2/4 图输入，但不能包含未来帧，其全部引用在建库时核对。

数据的 `stop_sign` 框由 LEAD `expert_data.py` 输出停止控制区域，**不是人工标注的路面停止线**。本版使用区域近侧边界作开发代理，要求车头仍在前方 0–3 m 范围内，连续 3 次观测绝对速度不超过 0.1 m/s；不是认证法规阈值。完成停车后，仍检查附近可见横穿参与者及其当前速度外推；未知可见性/运动保持 UNKNOWN，红灯和已知危险仍否决推进。

用户提出的“已停下就解除 STOP 未知”需要位置约束。实际读取的开发帧：2858_0 f50 的 STOP 区域中心在前方约 13.55 m，2679_1 f48 约 16.30 m，自车几乎静止；这是提前等行人，不能证明已到停止位置。本版不把这些帧冒充完成停车义务。STOP 修复尚无新增人工确认的真实释放正例，不能声称已经解决全部正例稀缺。

### 事件内按边 × 答案平衡

新政策用共享物理帧容量约束分配每个事件的等额配额，再对该事件实际存在的边×答案格等额分配；整数余量按 epoch 轮转。恢复绑定题池、seed、cap、预算、world 和实际曝光历史。

缺失答案格单独报告，不补成 NO。稀缺格/共享帧无法支持配额时报 `CapacityError`，不把缺少的 YES 配额转给 NO。严格分层也不保证七轮覆盖全部多数类题。

## 实测

- **782 项专项测试通过**，包括 STOP 位置/连续性/ID/未来证据、横穿冲突、首帧释放排除、弱监督 train-only 与来源防混、分层配额/容量/恢复一致性，以及真实文件构建加载测试。
- 23 条既有开发路线、2598 帧：每模式 57 YES / 170 NO / 68 UNKNOWN，共 295 个提议。proceed YES 37（UE1 2、UE4 35），恢复跟车 YES 1、完成 YES 1，完成实例 1。与 v28 相比，减少的是未经本事件限制建立的实例，不追求机械增加 YES。
- 每模式 227 道二元提议中，154 道仍因已登记风险整条路线需要另审而未入库；**73 道进入弱监督实验库（YES18、NO55）**。
- 训练 **443 → 516 题**，45 个物理组；原 443 条逐行保持。新增含 recover_follow YES 1 道。
- val273/test186/review539 与 v28 逐字节一致。2/4 模式各 **975 道监督题 + 539 道待审题 RGB** 全部核验，构建 API 与真实 `run_full_pipeline.sh MODE=check` 的四个 JSONL 一致。
- 每轮 100 题，十事件各 10 题；world1/world4、两种图数的七轮采样通过，共覆盖 357/516 道题，159 道未见。默认整池预算 520 不可行，会报告 R-E3 enter YES 和 U-E4 restrict YES 的容量缺额；**不要省略下面命令里的预算**。
- 原 109 条独立预约及 split 保持；当前源码重新回放并生成 963 张 2 图独立审核卡。只程序处理，没有目视独立 RGB、填写人工答案或据新版逐题结果调规则。
- 本轮本人没有新增 RGB 目视、人工标签、完整序列或曝光组。记录用户对 3_10 的纠正：栗色车在左侧相邻车道并行，不应因它认定发生切入；未据此私自改写冻结人工题。

## 可执行入口

以下从仓库根目录运行，Python 使用已配置依赖环境。现成产物在 `/tmp/phase4_v29_final`；这些是临时工作产物，不是持久备份。

现成 2 图弱监督库采样检查（已实际执行）：

```bash
PYTHON=/tmp/automot-qwen35-env/bin/python \
DATASET=/tmp/phase4_v29_final/weak_data2 \
OUTPUT_DIR=/path/new_check \
bash AutoMoT/qwen3vl_local/sft_new_loop_phase4/train.sh check --epoch-samples 100
```

4 图改为 `weak_data4`。有完整模型和 GPU 的训练主机将 `check` 改为 `single` 或 `ddp`，设置 `MODEL_DIR`，并保留 `--epoch-samples 100`。本轮没有启动 GPU 训练，没有验收完整模型、DDP 通信或 CARLA。

全流水线重建弱监督实验库（已验证对应 2/4 图入口）：

```bash
PYTHON=/tmp/automot-qwen35-env/bin/python MODE=check RGB_MODE=2 \
CANDIDATE_POOL=/tmp/phase4_v29_final/candidates.json \
PRODUCTION_INDEX=/tmp/phase4_v29_final/development_production/index.json \
TEACHER_REGISTRY=/tmp/phase4_v29_final/weak_registry.json \
DATASET=/path/new_weak_data2 OUTPUT_DIR=/path/new_check2 \
bash AutoMoT/qwen3vl_local/sft_new_loop_phase4/run_full_pipeline.sh --epoch-samples 100
```

新的弱监督策略文件可用 `teacher_approval --weak-experiment <name> --rule-class <exact class> ... --output <new.json>` 显式生成；只选本次实验需要的规则类，不能将旧源码的注册表改 hash 后使用。严格批准的原入口保持。

独立审核公开材料：`/tmp/phase4_v29_final/independent_review/index.html` 与同目录 `decisions.json`。私有快照 `independent_private.json` 不交给盲审者。新版审核可先用于测量弱监督误差，不能因没达到严格门槛而把审核结果当成无用，也不能把没有填写的参考视为全对。

## 实际边界

弱监督实验现在可以通过正式批准门槛之外的明确入口开展，不再需要为“允许实验训练”先凑够每类 20 条 YES/NO 路线。但这些标签的人工误差仍未知，严格批准仍为 0，不能称已认证的正式标签。

只实现 UE1/UE4 老师，其余八类仍依赖人工题；十事件独立评估未补齐。9712 路线仅完成冻结清点，当前回放为开发23 + 独立109，不是全量因果生产。全量生产可沿用 `teacher_replay` 入口，当前证据不足以声称全量已完成或承诺运行用时。

冻结 taxonomy/prompts/calibration/observation、控制器、安全许可、人工标签、选优和 Phase3/Action 保持。训练入口与采样代码有本版明确修改，不能再写“采样/训练完全未改”。
