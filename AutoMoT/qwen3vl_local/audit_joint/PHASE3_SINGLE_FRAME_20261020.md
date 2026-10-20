# Phase3 单帧方案一：四个半秒原生意图查询

当前正式启动请使用 [base 与新方案一运行入口](PHASE3_BASE_SINGLE_FRAME_RUN_20261020.md)。该入口补齐原生准备检查、资源探针、失败报告及base最终test参数修复；下文保留标签和底层worker协议说明。

最新动作 token 校验与自然 UNKNOWN 评测说明见 [PHASE3_SINGLE_FRAME_TOKEN_IDENTITY_20261020.md](PHASE3_SINGLE_FRAME_TOKEN_IDENTITY_20261020.md)。保存合同、续训和配对评测见 [PHASE3_SINGLE_FRAME_REVIEW_FIXES_20261020.md](PHASE3_SINGLE_FRAME_REVIEW_FIXES_20261020.md)。当前源码绑定数据为 `token_identity_train/val`；下文旧回执保留为开发历史。

## 已确认的协议

用户最终选择：暂停方案二；当前一帧拼接 RGB、速度、导航目标、当前 event；过去两秒每半秒一个 token；未来两秒每半秒严格输出一个动作。未来和历史都保留原 Phase3 标定，不新增动作阶段规则；用户最新明确允许历史标签使用后续轨迹，取消确认延迟门禁。此实现位于独立 `phase3_single_frame`，旧 base、旧 A/B、原生规则和默认入口均未改动。

单帧指一个时刻的拼接 RGB 文件，不是把多相机拼接再拆成单相机。未来四个查询分别汇总原生时点 (+.25,+.5)、(+.75,+1)、(+1.25,+1.5)、(+1.75,+2) 秒。它们是两个时点各自“接下来主要动作意图”的压缩，不是这半秒实际执行的动作。每个时点仍从原始 meta 调用原生规则及当前锚点 event 的题域投影；不因后续 event/RS 改变题域，不改 STOP/首跨/速度的原优先级。

输入前提保留当前速度和当前导航目标的自车相对坐标，不输入未来真实速度、位置、图像、答案或全程 ABC 模板。清除依赖历史图像的旧 context_detail 文本。沿用已有 A–F，四个 query 一次前向，未来真值从不进入语言输入，无跨样本缓存或连续段优先。

## 历史直接使用原生离线标签（最新决定）

最新协议为 `offline_native`，版本 `phase3_single_rgb_half_second_native_intents_v4_nearest_history_default`。过去与未来都直接调用原 Phase3 原生规则，使用完整记录路线，不截断到当前帧、不要求标签的后续证据在当前时刻已到齐。用户明确允许这种历史输入包含后续轨迹信息。

过去四个半秒窗口取原生标签起点 (-1.75,-1.5)、(-1.25,-1)、(-.75,-.5)、(-.25,0) 秒，沿用当前锚点 event 题域。每窗严格压缩成一个动作；两个等宽点一致就保留，不一致时取较晚点。先保留原生点标签和四格原历史，再对历史输入执行下述短缺口填补；填补后仍有未知点的窗口保持 UNKNOWN。UNKNOWN不再由确认时间导致。未来四窗与原版相同，不填补未来监督。

提示词同步明确历史是过去时点的原生意图标签，不要求未来照抄历史顺序。模型输入仍为单帧、当前速度/导航/event、四个历史token；历史不使用模型自己的预测。记录仍保留标签起点、证据窗口与当前时刻，便于核对，但不作为阻塞门禁。

前一版 `confirmed_native` 及其全部UNKNOWN的结果仅是旧开发记录，不是当前策略。旧数据不会被改写成新版，当前数据按新协议单独生成并绑定源码。

## 历史短缺口填补（最新实现）

对照模式 `--history-fill same_short`：同一路线、当前锚点event题域，两端均有原生标签且动作相同，连续未知原点不超过2个（每点0.25秒），才补成两端动作。0.5秒指未知点覆盖时长，两端原点的时间差最大0.75秒，不是允许两个0.5秒历史token未知。

默认 `--history-fill nearest_short`：包含上述同动作填补，另允许同样不超过0.5秒的异动作双端缺口。按时间距离选较近端；距离相同时选后端。`A ? ? B → A A B B`，`A ? B → A B B`。这是推断过渡时刻，不是原生规则新增了确定答案。`--history-fill none` 保留不填补对照。

不改已有明确动作（包括短B），不以填补结果再次扩展，不作单边延续，不跨原始meta缺失或数据隔离。原始 `history_points` / `history_original` 保存不变，派生 `history_resolved_points` 记录每点原action/reason、imputed标记、两端帧及动作、缺口时长、方法与选中端；四格另有 `history_imputed` 和 `history_changed`。训练消费与评测案例均保存原历史/标记。部分点填补后窗口仍可能UNKNOWN，因此imputed与changed分别统计。

本地同一批302 train/320 val：原UNKNOWN为214/235格；same_short对照后159/180格；nearest_short后88/116格。按路线/event/完整缺口去重，训练有61段双端可判定缺口，其中47段≤0.5秒；val有78段，其中50段≤0.5秒。异动作双端缺口分别36/40段，其中27/29段≤0.5秒；另有32/10段缺少某一端，不能用双端法填补。该统计只含与本次历史输入相交的缺口，不是全路线无条件分布。

未来标签、三轮训练锚点顺序、event/action配额与填补前完全一致；164项相关测试通过（39.10秒）。这些统计不证明插值正确或模型性能提升。这批历史对照资产在 `checkpoints/phase3_single_frame_v1/fill_same_train`、`fill_same_val`；当时nearest仅输出比较JSON。该次回执为 `phase3_single_frame_history_fill_verification_20261020.json`，不作为当前源码绑定。

## 当前默认与新资产

用户确认采用短异动作双端填补后，Python数据构建、CPU CLI、本地数据复核三处默认统一为 `nearest_short`，入口可省略 `--history-fill`。保留显式 `same_short` 与 `none` 对照。数据配置/训练合同记录实际选择，GPU worker直接消费绑定数据，不在训练时偷偷改变标签。

本轮165项相关测试通过（39.92秒），包含实际CPU CLI默认nearest_short及显式same_short/none。新数据逐条与上轮nearest比较一致，历史已知比例train为1120/1208（92.72%）、val为1164/1280（90.94%）；未来标签与三轮采样逐项保持一致。新数据由默认入口生成在 `checkpoints/phase3_single_frame_v1/nearest_default_train`、`nearest_default_val`。旧版产物保留；源码绑定变化后不能给旧产物改SHA复用。最新核验回执：`phase3_single_frame_nearest_default_verification_20261020.json`。同一半秒内原生动作混合时仍压缩成一个动作，选择较晚点；这项既定表示损失与填补规则分开，填补不会覆盖已知原生点。

## soft loss、稀缺动作与采样

两个时点都有效才监督该窗口。一个缺失则整个窗口 mask，不能把剩下一点重归一化成100%。记录 `raw` 原始采样占比和 `target` 训练偏好两份向量。

初始显式参数 `lateral_weight=1.5`，可配置1–3：仅对真实存在于混合答案中的 LEFT/RIGHT 增权，再归一化；例如减速/左变道的原始50/50，训练40/60。纯动作保持100%，原生单时点标签/优先级不变。该系数是未验证的实验设置，不是置信度或动作持续时间。没有人为增加不存在的变道。

使用完整词表 softmax 的软交叉熵，每个有效窗口计算，再在该锚点有效窗口内平均，跨rank平均。没有旧 A 的当前动作额外监督、50/50当前/未来预算，也没有在软标签外再次加类别权重。等权双动作的理想交叉熵为log2而非0；报告 target entropy、KL，不能直接与旧单标签loss比较。

预测每窗口从当前event允许动作中选择一个A–F，同时记录完整词表自由argmax及六动作原始概率。质量指标使用未偏置的raw目标：集合命中、采样占比得分、raw CE/KL、six-plus-other Brier、逐事件/时距/混合与纯窗口、动作召回、预测切换次数。短动作和窗内顺序可能丢失，软标签不保证argmax包含每个短动作。

采样先用原独立base方法确定 event/action 配额，再从至少有一个完整未来窗口的锚点中，按原每格配额使用原route_diverse_sample抽样。若某动作完全失去支持，明确拒绝，不把其配额分给KEEP。只有训练锚点可在同一格内替换；固定val/test不换成员。保留每事件1024、B1/累积1、四卡、三轮7680更新，无少见题/连续片段优先。报告原锚点配额和新未来目标的原始/偏置loss总量；**原action配额不意味着新未来标签各动作也等量**。重叠物理时点和窗口的重复曝光分别统计，不声称独立支持增加。

## 本地真实数据结果

填补前的offline_native版本使用同一批302个去重训练开发锚点和历史固定320个val锚点，重新生成历史标签并逐条比较未来窗口、锚点与采样配额。结果：训练历史994/1208格有效（82.28%），295/302个锚点至少有一格历史；val历史1045/1280格有效（81.64%），310/320个锚点至少有一格历史。4976个历史原点逐项与原生规则输出一致；未来窗口、三轮锚点顺序和event/action配额与前版完全一致。153项相关测试通过（38.43秒）。最新结果、历史已知比例、原生动作分布和测试数量见 `phase3_single_frame_offline_history_verification_20261020.json`。

填补前的数据和日志位于 `checkpoints/phase3_single_frame_v1/offline_train`、`offline_val` 及同级日志。旧 `accepted_*`、`local_*`、`final_*` 是旧源码/旧历史策略产物，不作为新版复用数据。开发池有大量重复采样，仅验证规则与训练接线，不代替服务器全池。固定val缺失目标保留mask、不更换锚点；本地标签覆盖率不是模型准确率，没有执行新4B模型或历史最佳模型推理。

## 可调用入口与验证边界

CPU数据入口（不会训练）：

```bash
python -m qwen3vl_local.audit_joint.phase3_single_frame build \
  --index /本机/原生/frame_index.jsonl --data-root /本机/lead_data \
  --output /本机/新数据目录 --history-policy offline_native --history-fill nearest_short --lateral-weight 1.5
python -m qwen3vl_local.audit_joint.phase3_single_frame audit \
  --index /本机/原生/frame_index.jsonl --data-root /本机/lead_data \
  --dataset /本机/新数据目录/dataset.json --output /本机/新采样报告.json
```

数据路径明确提供；已有输出拒绝覆盖，不猜最新、不生成时间戳目录，不复制RGB/权重。artifact绑定原索引、原始meta及当前RGB SHA/stat/逻辑挂载、规则和代码、主机/时间。可重复读取已验证artifact，不重新生成。

A-only GPU worker 的操作是 `prepare / probe / train / evaluate`。prepare使用历史冻结原生索引、固定val/test/loss成员和所有三轮计划；扫描训练和评测输入，并检查每事件/窗口至少两个不同支持目标。probe要求四个rank，训练输入实际反向/AdamW两步，评测输入只前向；探针权重不保存。train要求匹配成功探针及新输出，仅训练此方案A；evaluate重载三轮7680步完整checkpoint，分别评估固定val/test，输出独立JSON。没有运行B、没有重复旧最佳模型，也没有修改旧base/A/B总入口。

当前已通过的CPU测试覆盖真实小Qwen四读出反向、原生trainer十次更新/完整checkpoint/40次监督消费、单RGB实际输入、固定评测形状、GPU准入拒绝分支、历史与原生离线标定逐点一致、允许后续证据改变历史标签、原生标签不受偏好权重影响、缺失mask、动作支持丢失拒绝、采样三轮预算及RGB/meta篡改拒绝。最新测试日志及SHA见本轮离线历史核验回执。

**尚未执行四H20 probe/正式训练/新模型性能验收，也未做服务器全池准入。** 本轮本地结果证明数据和训练接线可测试，不证明稀缺动作召回提高、恢复历史70%以上。后续正式服务器命令应使用新A-only worker，不再执行旧base/A/B联合命令。
