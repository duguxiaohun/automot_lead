# Action prior 实现与合同

## 2026-09-22 七类动作 token 弱分离

共享 `action_token.separation_loss` 对七类 embedding（含 UNCOND）的 21 对余弦相似度施加 squared hinge，
默认 margin=0.5、weight=0.01，仅开启 token 的新训练生效；weight=0 为对照。
共享训练循环在 FM loss 后、累积除数前加入正则，FP32 计算；不改实际 token 的范数或 concat。
`separation_contract` 将版本/系数绑定三入口条件合同与训练计划；旧配置缺字段按关闭解释，但仍须原源码恢复。
首步/定期/轮末/最终步记录所有类对 cosine 和范数，审计窗口保留 FM 与正则分项；验证/选优仍按原指标。
此软约束并非 SIGReg 复现，也不证明 decoder 使用 token 或轨迹性能提高。运行示例见 [run.md](run.md)。

## 2026-09-21 v21 共享标定与独立提示词合同

候选来源、动作 token 与文字动作主要投影直接复用 Phase3 v21；不维护另一套速度规则。主线自然先验将 UE1 改为可持续的响应/等待/恢复，将信号异常改为给定系统故障；普通、紧凑、摘要与直接 prefill 路径共用并升级 prompt 版本。Phase1/2 检测合同及消融简短/无 Qwen 条件保持独立。数据产物与实际验证范围见 [run.md](run.md) 的 v21 同步说明。

## 2026-09-21 Decoder 动作 token 与图数条件

`--high-level-action-token` 独立于文字动作先验，默认关闭。`action_token.py` 以当前 Phase3 candidate/full map 为唯一标注源，沿用完整证据检查和主要动作优先级；KEEP 不细分，普通/隔离/未确认/覆盖外帧显式 UNCOND。词表七类，`nn.Embedding(7, hidden_size)` 默认 7×1024，经 FM 轨迹损失学习、路由 AdamW。BEV projector 输出后沿序列维追加一个 token，BEV/action/status/query 一同经过已有 Prefix-KV attention；143 个 token 的 route/wp 切片由配置统一维护。条件编码一次，FM 各步共享；不增加文字描述或 Phase1/2 gate。三条路径共享同一标注、模型和训练实现。

`--rgb-frame-count 1` 选择当前 anchor 的完整拼接图，默认4；索引仍为四帧构建记录，运行时覆盖输入采样数。`image_condition.py` 将 simple/base/先验/可选摘要的视觉说明切为单图，主线 LoRA 问题和输出 schema 保留，道路几何定义复用，依赖时序证据的指令适配当前图，记录旧 LoRA 的输入分布变化。BEV 不变，bev_only 没有 Qwen 历史输入。

token 内容身份、词表、源码及 RGB 图数绑定 checkpoint/缓存；开关关闭保持既有模型参数初始化，新增 embedding 的初始化保存/恢复随机流。eval/resume 不临时换条件，oracle token 在闭环加载时拒绝。详见 [run.md](run.md) 的开启/关闭和单图 demo。

本机回归：635通过、7跳过；另外22项失败源于本机缺失只读 `leaderboard/team_code/mot_lead_offline_runner.py`（17项）或 `peft`（5项）。保留源码身份校验，没有用缺省哈希绕过。新增测试覆盖真实小模型 FP32/BF16 反传、UNCOND/KEEP、三入口合同错配、单图两阶段问答、缓存和 CLI/恢复；未运行真实 Qwen/BEV 或远端多卡。


优化细节已统一为默认值，无需追加新开关。每个 epoch 训练结束及完整验证后，自动更新当前 run 的 **`training_audit.zip`**；训练被中途终止时可直接带走此文件审计。包内有进度、各轮训练/验证指标、loss/LR/更新幅度和实际配置，详见 [默认训练与中途审计](OPTIMIZATION.md)。

2026-09-21 优化更新：主线与两个消融共用 `optimization_config.py` / `optimization.py`，新训练默认 Muon＋辅助 AdamW、首轮5%更新warmup＋1/2/4轮cosine restart（共7轮，warmup占用首周期）。仅更新指定隐藏矩阵，其余参数保留 AdamW；完整路由和恢复合同见 [OPTIMIZATION.md](OPTIMIZATION.md)。

操作入口见 [run.md](run.md)，可选复核和评测解释见 [AUDIT.md](AUDIT.md)。

## 模型

四张 4 Hz stitched RGB、当前速度与导航进入冻结 Qwen。默认用 Phase1/2 LoRA 提供先验，再禁用所有 LoRA，
将四图、自然 RS/EVENT 描述和导航直接 prefill 得到 KV；不生成摘要，不追加 assistant 消息。
`dataset-priors` 直接读取标定标签，默认冷启动没有文字生成，仅一次完整 base KV prefill。
默认最终 KV 不注入类别 JSON、NO/UNKNOWN 或逐帧动作标签；显式具体动作开关的扩展见下节。

`--generate-analysis` / `GENERATE_ANALYSIS=1` 保留原摘要路径：base 生成短摘要、按配置复核/fallback，
再把 system＋user 四图/先验/导航＋assistant 摘要完整 prefill。`--no-generate-analysis` 为默认。
直接模式用 `PREFILL_SYSTEM_PROMPT` / `prefill_prompt`，没有要求写摘要的指令；
两条路径都使用 `add_generation_prompt=False`，直接模式也不追加空的 assistant 起始头。
摘要关闭时忽略复核设置，不产生摘要 fallback。先验问答、invalid 处理和轨迹监督照常执行。
开关进入条件身份和缓存 key，最终 KV 每次重建，M-RoPE 偏移仍按输入长度加 `rope_deltas` 计算。

冻结 LEAD BEV，与 KV 共同条件化联合 Flow Matching 轨迹 decoder：route `(10,2)`、waypoint `(8,2)`。
训练默认仅向量场 MSE；验证从高斯噪声默认 10 步 Euler 采样，按样本身份固定噪声。
可训练参数、优化器状态、EMA 为 FP32，decoder 可用 BF16 autocast。best 按采样 ADE 选取。
旧 Linear+cumsum 和逐点 FM checkpoint 不兼容当前 `action_prior_checkpoint_v4`。

主线与两个 action-expert 消融共用 `training_core.py`；统一 DDP、梯度累积、EMA、验证、checkpoint 与恢复。
中断只在 optimizer 安全点保存，SIGKILL/掉电仍依赖周期 checkpoint。源码身份变化需新 run。

## 先验边界

- RS_HIGHWAY 是独立事实，R3 不能推出高速；缺失标签不当 NO。
- Phase1 全问加 RS 分层复核；Phase2 使用已训练的双域问法，不伪造 EVENT hierarchical 接口。
- 复核一致只表示接受条件，摘要模型复核也不保证语义正确；失败可 fallback。
- 无噪声 `dataset-priors + high-level-action-prior` 自动提供已确认特殊 RE 场景；不再提供独立场景开关。LoRA、带噪声或未开启动作输入 时不自动补入。
- RE2 当前可描述导航变道或早先障碍记录；没有可靠证据时不声称“UE2 刚结束且恢复待完成”。

## 所选动作的场景因果句

移除 `--high-level-planning` / `HIGH_LEVEL_PLANNING` 及整套候选动作/通用目的列表。
自然 RS/EVENT 沿用原 `scene_description`，只有门控后的 `selected` 在场景末尾追加一句
`Next action: ` + 当前 Phase3 `choice_semantics.action_description(context_id, action)`。
不另维护动作原因副本，确保和 action choice 完全一致；不带数值标定窗或让 base 输出 choice 的指令。
并发事件的事实全部保留，原因只在接受且支持所选动作的 context 中按 taxonomy 顺序选一个，
记录 `description_context_id`。例如横向动作不会错误选用只支持纵向的 UE1 原因。
RE2 三种独立场景事实仍分别保留，动作句统一复用 Phase3 的 POST_BYPASS_RETURN，
其描述使用可见历史区分当前阶段，不断言先前障碍已经绕过。

直接 prefill、摘要、复核、fallback 共用渲染函数；摘要失败时仅重述所选动作句。
无有效动作时不补猜测，保留自然场景输入；默认 system 仍为 PREFILL_SYSTEM_PROMPT。
80词只约束可选摘要，不裁切输入。默认不运行摘要、复核或 fallback。
`ACTION_CONDITIONING_VERSION=upstream_gated_phase3_causal_sentence_v3` 与 choice_semantics.py
源码 SHA256 进入动作条件身份；新旧措辞/规则不能混用缓存或 decoder。
旧开关和环境变量明确拒绝，旧 run 必须使用原源码恢复。

## 具体 high-level 动作输入

默认关闭的 `high_level_action_prior` 可独立开启。新训练无需填写动作索引：
`prepare_action_priors.py` 复用自动准备器的 Phase3 全量候选和 full map，缺失时调用原构建器，
读取当前规则的 `action_labels`，按 Phase3 三/五动作域投影后对齐 action 三 split。
候选只是特殊帧 eligibility，缺席不能反推普通背景；不使用最终均衡 frame_index 作为全帧动作表。

仅 UE1–7、RE2/3/5 中 `special_eligible` 的帧提供动作候选，索引上下文不再注入场景事实。
`gate_action` 在获得实际上游先验（含噪声/复核）之后运行：UE 要求对应 YES，Phase2 的 UE 还要求
问题域有效；所有上下文检查 Phase3 允许的道路结构，RE 要求显式的独立 transition 上下文。
新训练在无噪声 dataset + action prior 下自动从独立 full map 提供 RE2/3/5；不由动作文件、RS 或上游全 NO 推导。
`scene_policy.py` 统一 shell/Python 判断，自动模式只补特殊 RE，UE 始终取实际 Phase1/2 条件。
内部保存 `event_balanced_scene_priors` 和 `scene_prior_policy=dataset_action_special_re_v2`，后者绑定合同和缓存。
resume/eval/probe 保留原条件，不按新默认重新推导；旧 run 仍须原源码。动作模式自动准备完整映射与动作索引。
并发事件按已确认域投影动作，未确认的机动域不能贡献横向标签。

只有最终 `selected` 渲染动作段；`no_action / unavailable / not_applicable` 和被门控拒绝的动作
都保持同一上游条件下的原始 prompt，摘要/复核/fallback 同步。审计保存原始输入、有效动作和门控理由。
普通背景两份配额不变；动作开关不隐式改变 uniform/event_balanced 采样模式。

这是用户显式开启的离线真值条件实验，来源标记 `phase3_oracle` 和
`privileged_action_conditioning=True`，不冒充 Phase3 模型推理。候选标签依赖未来轨迹证据，
动作域、规则代码、映射、candidate/full map、action split 和动作文件内容均绑定来源合同。
先剔除异常 route；缓存构建有锁和原子发布，损坏的自动产物隔离后重建，续训不重新标注。
开发路线强制 train-only；来源/状态/覆盖写入训练计划，逐帧动作进入文本缓存 key。

`action_input.py` 负责严格输入校验、精确帧查表和统一门控；`prompts.rendered_action` 共用 Phase3 逐场景动作因果句。
输入 schema 为 `scoped_phase3_primary_action_v4`，`action_format=primary_choice_v1`。
与 Phase3 v13 choice 共用 `primary_action.py`：STOP > 首次跨线 > 纵向动作，空集为 NONE。
自动索引 `actions` 最多一个；额外 `candidate_actions` 保存原始纵横证据，runtime 先按上游确认域门控，
再投影为唯一动作（防止被挡下的机动域抢走有效纵向动作）。原始证据、有效域动作和最终动作分别审计。
`prediction/provided` 只接受单动作或空状态，不能伪造原始 oracle 证据；`choice_action_input` 严格转换
词组输出，NONE→no_action，非法/缺失→unavailable，绝不凭空补次选动作。
格式、门控、主要动作版本/源码、choice 因果描述源码及 taxonomy 哈希进入条件身份，旧 v3/binary/旧 choice 文件需重建。
此次只对齐五个变化动作的文案，沿用 NONE/no_action 空状态，不凭它合成最新版 KEEP。
普通 RE 为 not_applicable；有效 UE/特殊 RE 无动作是 no_action；二者均无动作段，原有场景事实保留。
默认 dataset 条件仍零文字生成；摘要/复核/fallback 使用同一动作，背景不出现额外动作段。
`--high-level-action-index` 仅用于搬迁或高级 prediction/provided 输入，后者也必须提供事件作用域，
不能给普通背景塞入动作。自动索引搬迁需携带 manifest，内容必须与 checkpoint 相同。
闭环尚无在线 provider，入口拒绝此模式；后续 v13 choice provider 可复用规范化动作/作用域接口和同一门控，
仍须接通在线预测、独立 RE transition gate 与来源合同。统一接口不保证 oracle/预测无分布差异，不能直接换源评测。
无先验消融不暴露此开关；开启/关闭与恢复 demo 见 run.md。

## 均衡采样

全帧 map 将 `special_eligible`、`special_filtered`、`confirmed_regular`、`unconfirmed` 分开。
Phase3 candidate 只标记特殊帧 eligibility，candidate 缺席不等于普通背景；只有确认常规进入两份背景池。
UE1–7、RE2/3/5 十桶各一份，普通背景两份；开发路线强制 train-only，val/test 保留自然分布。

相同事件归属的帧压缩为小型最小费用流网络。首次使用费用 0，额外使用费用 1，固定总配额下精确最大化全局唯一帧覆盖。
共享帧可跨桶重分配，展开时跨桶共用游标，组内路线轮转；路线轮转是顺序偏好，不是硬性路线配额。
全局 presentation 构建完成后再按 DDP rank 分片。默认每帧单轮上限 8，预算须满足 `lcm(12, world_size)`。

## 文件与恢复

自动准备只构建缺失的中间产物，使用现有原始数据、人工标注和本地权重，不下载或伪造标签。
候选/full map 使用锁和临时目录原子发布；全帧 map 绑定候选内容、规则版本和 action 三 split 的 SHA256。
v1 map 必须重建。候选规则或构建代码变化产生新的自动缓存目录，旧产物保留。

模型/adapter 字节指纹、prompt、导航、RGB 和执行源码共同约束 checkpoint 与文本缓存。
LoRA 选择会固定来源并复制到 run/lora；先验及可选摘要的文本缓存可跨 rank 共享，但最终 KV 每次完整 prefill。
续训、离线评测和闭环从保存配置恢复摘要开关，不能用同一 decoder 临时换模式；缺字段的旧配置按历史摘要语义解释，
但仍需通过严格执行指纹检查。本次变更用于新 run，旧 run 用原代码恢复。开启/关闭 demo 见 run.md。
`train.sh --resume`（含等号写法和 `RESUME` 环境变量）在注入任何新训练默认值前进入同一
`resume.py`；恢复原摘要开关、LR、索引等参数，只转发显式环境覆盖和 CLI，CLI 优先。
审计来源身份与实际生成身份分开，来源不可访问记 unknown；同内容路径迁移不应改变生成条件。
详细历史事实以仓库根目录 PROJECT_CONTEXT.md 为准。
