# geometry_v1：九类车道方向 × 速度参照前进程度

独立实验入口，2026-10-21。旧 single_frame 六动作入口、数据和权重保留。新版不读取旧动作标签作为监督，也不将旧 adapter 解释成九分类模型。

## 固定定义

- ID 0–8 / A–I 依次为 LEFT、CENTER、RIGHT 各自的 NEAR、REFERENCE、FAR。
- 当前最近 Driving 车道切线向自车前向定向；以自车位置为原点。横向小于 −2 m 为 LEFT，大于 +2 m 为 RIGHT，边界属于 CENTER。这不是实际车道白线。
- `delta(v)=min(1,max(0.5,0.2*v))` m/s；`q=r/t-v`，`r` 是水平首尾直线距离。小于 `-delta` 为 NEAR，大于 `delta` 为 FAR，其余 REFERENCE。实现用等价的连续半径比较，减少浮点相减导致的边界翻转，不取整半径。
- 未来 `+0.5,+1,+1.5,+2s` 使用原始 20 Hz 轨迹的第 10/20/30/40 点，四点共享当前冻结的原点、速度、车道方向。没有窗口混合软标签、侧向加权或事件内 A–F 动作限制。
- 历史端点为 `−2,−1.5,−1,−0.5s`：位移取 **过去位置减当前位置**，四点统一使用当前原点、当前速度和当前车道参考，半径计算使用正的回看时长。四个 A–I 标签明确写入提示词；UNKNOWN 保留缺失槽位。这是用户本轮确认的新定义，取代上一版各历史起点参考系。只读取截至当前的真实位姿，不读取任何未来轨迹字段，不插值、不填补。过去点在当前左侧不等于过去向左驾驶。
- `[-0.1,0)` 的微小负速度归零；更负、缺失、NaN/Inf 均不可用。历史 UNKNOWN 不属于九个监督类别；未来缺失按端点掩码，整帧没有有效端点不参加训练。

这九类描述几何位置，不能恢复严格 STOP、瞬时加速度或先加后减的完整过程。所有阈值是本次数据评估后固定的首版，尚无模型效果最优的结论。

## 数据与预算

`build` 直接读取原始 collection 的逐帧事件标注，沿用事件修订、异常路线排除、物理路线隔离与已开发路线 train-only 政策；在旧 `label_actions`、旧动作可用性及四图历史要求之前取候选。几何资格由当前图像、导航、速度、参考方向和轨迹完整性决定。旧 collection 的事件覆盖本身仍是上游边界，不能称为全部驾驶帧都有事件。

建库默认合并 Phase3 的全部 `development_route_groups_*.json`，包括 20260926 和 RGB-stage 追加曝光；也可重复传入 `--development-manifest PATH`，继承既有合同的 `additional_development.groups`。名单内容和源 SHA 一同绑定，建库、重载和训练一致拒绝曝光路线进入 holdout。

先在原始候选上做覆盖可行性检查，几何过滤后再以有效端点记录补全 train/val/test 十事件覆盖，仅移动未曝光的完整物理路线。重复采集与无有效端点的同组成员一起移动，选择不使用九类标签值或模型表现。候选与最终结果分别写入 `candidate_split_coverage.json`、`split_coverage.json`。贪心补全成功时保留原结果；失败时撤销本轮移动，对原 train 中未曝光的完整物理组做精确回溯，允许它们重新分配到 train/val/test。原始 val/test 固定，开发组始终 train-only，不依据模型表现重选 holdout。当前最低是一事件一有效帧／物理组，只证明覆盖；各事件物理组数单独报告，不称泛化认证。局部功能构建必须显式 `--scoped-smoke-only`，该产物禁止正式训练。

回溯仅针对当前“一事件一有效帧／组”门槛：相同事件覆盖组合的路线可合并为一类，每类最多保留三个可交换代表；这不适用于更高帧数／组数配额。结果的 `feasibility` 为 `feasible`、`infeasible` 或 `unknown`。只有完整搜索证明当前固定 holdout／开发政策下无解，才报数据覆盖不足；达到 `--coverage-search-states`（默认 100000 个状态）时，报告 `coverage_search_inconclusive` 并停止，不能称为数据不足。可按需提高该状态预算，用新输出目录重试。无解或未判定时不会发布贪心留下的部分划分。该修复只在 geometry_v1 中启用，共享的旧 Phase3 补全实现保留。

当前图像完整解码；绑定实际读取的 meta、图像、地图、标注文件、隔离记录及代码 SHA。输出必须为新目录，旧输出和源数据不覆盖。已知源不连续窗口屏蔽，不把不可用当作静止。多余晚期轨迹损坏不影响更早的完整端点。缺失地图直接报错，不切换成自车朝向。

默认十事件各 1024 个锚点呈现，B1/累积1/四卡每轮 2560 次更新，三轮 7680；warmup 2000。事件内按本 run 的成功 optimizer 更新曝光优先，其次任务组合与路线覆盖，不强行九类等量。实际曝光按物理帧跨事件共享；每轮每帧最多 10 次。贪心选择遇到共享容量冲突时，通过事件—物理帧残余路径重新分配先前选择；只有找不到增广路径才报容量不足，保留事件配额和曝光优先规则。计划投影不写实际曝光，下一轮重新使用实际记录。

训练限同一机器的 torchrun。启动时 rank 0 完整核验一次原始源文件，其余 rank 校验同一份已核验数据文件的 SHA；没有可跨运行复用的跳过核验开关。模型源 SHA 也只扫描一次。processor 启动检查为 train/val 每事件各一条、总计至多 20 条，在 rank 间分片；本 rank 保留这些打包结果，其他输入消费时再逐条完整检查。不会在加载模型后预处理整个 train/val/test 池。

`--max-startup-hours` 默认 1 小时，独立覆盖源核验、计划、模型加载和小样本 processor 检查；`startup.json` 记录各阶段耗时及检查成员，日志同时打印阶段。它是协作式预算：文件哈希逐块检查，阻塞的库调用在返回时检查，超时禁止开始更新；不是可强制中断任意底层阻塞的硬实时保证。启动预算与训练预算分别计量，不能将三轮 10 小时解释为总作业最多 30 小时（另含启动、保存和评估）。

默认每轮 4096、总计 12288 步上限；100 步后依据实测速度检查 10 小时单轮上限，超限各 rank 协调停止并保存 `budget_stop`。这不是一天完成保证，也不能视为 Qwen 单步提速。普通异常没有自动续跑；保存 optimizer/scheduler/RNG 是诊断和后续恢复的基础，目前 CLI 不提供 resume，不能拿部分 epoch 冒充完成。

每轮保存新 checkpoint，并在按 ID 固定、每事件至多 16 帧的验证子集上评估。完整 `evaluate` 使用新数据内固定 split 全体成员，缺失标签计入覆盖而不计入准确率。只有完整训练预算后的 checkpoint 可评 test。报告包含每时距/类别召回、缺格、低速、事件、边界 0.1m 子集、物理路线支持和原始词表输出合法性；不与旧六动作准确率直接相减。

## 本机明确路径的入口

从 `AutoMoT` 根目录运行。使用已经安装项目 Qwen3.5 runtime 的 Python。建库是显式操作，本次只运行小范围源验证，未启动全池构建或 GPU 训练。

```bash
ENTRY=qwen3vl_local/audit_joint/geometry_v1/run.sh
bash "$ENTRY" build --collection "$COLLECTION_DIR" --data-root "$DATA_ROOT" \
  --maps "$MAPS_DIR" --output "$GEOMETRY_DATA"
bash "$ENTRY" plan --dataset "$GEOMETRY_DATA/dataset.json" \
  --output "$GEOMETRY_PLAN_JSON" --world-size 4
bash "$ENTRY" train --dataset "$GEOMETRY_DATA/dataset.json" \
  --model-dir "$MODEL_DIR" --output "$GEOMETRY_RUN"
```

训练默认自动选择四张空闲 GPU；显式四卡：`GPU_IDS=0,1,2,3 bash "$ENTRY" train ...`。单卡必须显式降低 `--per-event`，否则三轮总步数会超过默认门限，例如 `GPU_IDS=0 bash "$ENTRY" train ... --per-event 256`；这会改变曝光预算，不能当作等量四卡实验。机器任选，各变量指向当前机器资产，不跨机器猜路径或迁移数据。输出使用明确方法目录，例如 `geometry_v1`，碰撞拒绝。

```bash
bash "$ENTRY" evaluate --dataset "$GEOMETRY_DATA/dataset.json" \
  --model-dir "$MODEL_DIR" --checkpoint "$GEOMETRY_RUN/epoch_3" \
  --split test --output "$GEOMETRY_TEST"
# 显式单卡：GPU_IDS=0 bash "$ENTRY" evaluate ...
```

## 连续 expert 与推理边界

`sources.observation(...)` 是离线和在线调用者共用的因果观测适配器；在线调用者只需提供截至当前的 meta 形状观测和同一地图。`objective.predict(...)` 不读取 record 的 targets，输出四个类别、完整规则合同、本帧参考方向和 `sample_binding`。绑定包含样本 ID、场景／路线／帧／图像标识、事件、道路结构，以及完整因果观测的 SHA；不包含未来监督。expert 消费时逐项重算比较，缺少绑定或换帧、速度、历史、导航均拒绝，不能跨样本复用预测。图像标识由调用者正确关联到传入 RGB；此接口不能证明调用者传入的像素就属于该标识。当前事件/道路信息由调用者提供；离线真值事件不等于在线 Phase1/2 预测已验收。

`LeadMoTPlanningDecoderConfig(use_geometry_tokens=True)` 启用四个条件 token，原六动作开关必须关闭。decoder 输入新增：

```python
prediction = objective.predict(bundle, record, current_image)
condition = expert.condition(record, mode="predicted", prediction=prediction, device=device)
outputs = decoder(**observed_expert_inputs, **condition)
```

`geometry_token_ids` 为 `[B,4]`，`geometry_reference_yaw` 为 `[B]`。token embedding 加时距编码和车道相对方向编码；共享 expert 仍产生 route10 和 waypoint8 连续点，FM 路线同样接通。`expert.training_loss` 对接现有 FM loss、保持连续监督；调用者继续负责冻结 KV/BEV、optimizer、预算及 checkpoint 合同。`expert.condition` 默认要求真实模型预测，不允许缺预测回退到真值；oracle 必须显式指定，none 模式四个 ID 都为内部 UNCOND=9，UNCOND 不是 Phase3 第十类。

`expert.compare_conditions` 在同一 checkpoint、同一输入和相同 FM 噪声上运行 none/oracle/predicted；`paired_trajectory_metrics` 汇总 ADE/FDE 和末速度不超过 0.2m/s 的终点误差。这是条件干预诊断，不能代替三组分别训练的消融实验。

当前交付接通了 decoder 与 FM 的程序接口，**没有把现有 action_prior launcher 或 CARLA 在线 agent 默认改成 geometry_v1**。训练新 expert 和在线控制器的实际消费仍需以新模型及对应资产进行端到端验收。旧数据、权重及实验入口保留；共享 LeadMoT 两文件新增可选接口，因此依赖源码指纹的旧 checkpoint 仍应使用原源码，不能改 SHA 绕过。

## 本地验证范围

最新划分回溯修复见同目录 `split_search_verification_20261021.json`；`review_verification_20261021.json` 和 `verification_20261021.json` 保留为前两轮历史快照，不能用于当前源码验收。本轮历史位置标签、九类 codec、提示词和训练采样器源码逐字不变。历史定义、曝光合同和源码绑定已经变化，旧 geometry 小库／checkpoint 不自动兼容，须用新目录生成当前合同产物，禁止重写旧 SHA。包含合成边界与异常输入、原始地图/meta/RGB 小样本建库重载、真实小型 Qwen3.5 单图四查询 backward/保存重载、两 rank Gloo 实际曝光、Linear/FM expert 条件编码和梯度。没有下载模型权重、没有服务器 GPU/SSH/仿真、没有新模型效果或闭环安全结论。
