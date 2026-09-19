# Action prior 实现与合同

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
可训练参数、AdamW、EMA 为 FP32，decoder 可用 BF16 autocast。best 按采样 ADE 选取。
旧 Linear+cumsum 和逐点 FM checkpoint 不兼容当前 `action_prior_checkpoint_v4`。

主线与两个 action-expert 消融共用 `training_core.py`；统一 DDP、梯度累积、EMA、验证、checkpoint 与恢复。
中断只在 optimizer 安全点保存，SIGKILL/掉电仍依赖周期 checkpoint。源码身份变化需新 run。

## 先验边界

- RS_HIGHWAY 是独立事实，R3 不能推出高速；缺失标签不当 NO。
- Phase1 全问加 RS 分层复核；Phase2 使用已训练的双域问法，不伪造 EVENT hierarchical 接口。
- 复核一致只表示接受条件，摘要模型复核也不保证语义正确；失败可 fallback。
- 无噪声 `dataset-priors + high-level-planning` 自动提供已确认特殊 RE 场景；不再提供独立场景开关。LoRA、带噪声或未开启 planning 时不自动补入。
- RE2 当前可描述导航变道或早先障碍记录；没有可靠证据时不声称“UE2 刚结束且恢复待完成”。

## 可选 high-level planning

`high_level_planning=False` 默认保持原提示词。开启后，`prompts.py` 保留确认的 RS/HIGHWAY 和全部事件事实，
用一段条件性的减速、停车/继续等待、持续增速文字替换旧规划；静态障碍/弱势参与者及显式 RE2/RE3 才补左右首次未来跨线。
语义参考 Phase3 的 `CHOICE_ACTION_DESCRIPTIONS` 与三/五动作域，但不 import Phase3 runtime，
不增加 adapter/生成，不读取未来动作标签，也不复制其单选格式和标定阈值。未确认事实与绕障历史不自动补齐。

2026-09-19：planning 协议升级为 `phase3_inspired_conditional_high_level_v3_compact_purpose`。
在 user 的场景描述中按已接受事件/独立 scene context 补充动作目的，借鉴 Phase3 v12，
仍在本包维护自然文字。UE1 减速/等待用于保持跟车间距；UE2 用于观察相邻车道交通、接近车辆
（借道时含对向来车）和通过空间，判断绕障空隙；RE3 根据目标车道车辆位置和相对运动判断空隙。
RE2 导航转换、先前障碍和确认恢复分别描述，不把早先障碍记录写成已经绕过。
并发事件保留各自目的、同一事件去重；目的不证明存在空隙或已选动作，也不规定先减速后变道。
`high_level_action_prior` 继续渲染统一动作释义，目的边界只说明一次，不另加重复关联句，不新增动作类别，
不从动作 scope 反向填充场景。两个开关关闭时的 prompt 和无有效动作时相对 planning-only 的 prompt 保持原行为。

开关独立于 `generate_analysis`，直接 prefill、可选摘要、复核、fallback 都一致使用；高层 fallback 只重复短规划/导航，
完整事实留在 user prompt。`HIGH_LEVEL_PLANNING_VERSION` 与开关进入条件身份，缓存另显式区分开关，审计记录实际模式。
场景目的同样保留在 user，可选摘要的生成与复核共用；仅显式开启摘要时，fallback 才受 80 词预算约束。
`MAX_ANALYSIS_WORDS` 不用于输入提示词校验/截断。默认 prefill 无摘要/复核/fallback，使用没有
“within 80 words”指令的 `PREFILL_SYSTEM_PROMPT`；输入精简靠模板去重，模型 token/上下文容量另计。
v3 压缩通用条件性规划与目的约束，保留横向首次跨线、观察空隙及不推导额外动作的语义。
固定 R1+UE2 场景文本 120→96 个英文空白分词，加 STOP 且排除导航/区块标签为 164→126；非模型 token 数。
resume 从原配置恢复，eval/probe/闭环从 checkpoint args 恢复；改先验来源也不能放宽 planning 合同。
旧配置缺字段解释为关闭，严格源码校验不变。开启/关闭、CLI/环境优先级与续训 demo 见 run.md 和两个训练 shell 入口。

## 具体 high-level 动作输入

默认关闭的 `high_level_action_prior` 依赖 `high_level_planning`。新训练无需填写动作索引：
`prepare_action_priors.py` 复用自动准备器的 Phase3 全量候选和 full map，缺失时调用原构建器，
读取当前规则的 `action_labels`，按 Phase3 三/五动作域投影后对齐 action 三 split。
候选只是特殊帧 eligibility，缺席不能反推普通背景；不使用最终均衡 frame_index 作为全帧动作表。

仅 UE1–7、RE2/3/5 中 `special_eligible` 的帧提供动作候选，索引上下文不再注入场景事实。
`gate_action` 在获得实际上游先验（含噪声/复核）之后运行：UE 要求对应 YES，Phase2 的 UE 还要求
问题域有效；所有上下文检查 Phase3 允许的道路结构，RE 要求显式的独立 transition 上下文。
新训练在无噪声 dataset + planning 下自动从独立 full map 提供 RE2/3/5；不由动作文件、RS 或上游全 NO 推导。
`scene_policy.py` 统一 shell/Python 判断，自动模式只补特殊 RE，UE 始终取实际 Phase1/2 条件。
内部保存 `event_balanced_scene_priors` 和 `scene_prior_policy=dataset_planning_special_re_v1`，后者绑定合同和缓存。
resume/eval/probe 保留原条件，不按新默认重新推导；旧 run 仍须原源码。planning-only 自动准备完整映射但不生成动作索引。
并发事件按已确认域投影动作，未确认的机动域不能贡献横向标签。

只有最终 `selected` 渲染动作段；`no_action / unavailable / not_applicable` 和被门控拒绝的动作
都保持同一上游条件下的原始 prompt，摘要/复核/fallback 同步。审计保存原始输入、有效动作和门控理由。
普通背景两份配额不变；动作开关不隐式改变 uniform/event_balanced 采样模式。

这是用户显式开启的离线真值条件实验，来源标记 `phase3_oracle` 和
`privileged_action_conditioning=True`，不冒充 Phase3 模型推理。候选标签依赖未来轨迹证据，
动作域、规则代码、映射、candidate/full map、action split 和动作文件内容均绑定来源合同。
先剔除异常 route；缓存构建有锁和原子发布，损坏的自动产物隔离后重建，续训不重新标注。
开发路线强制 train-only；来源/状态/覆盖写入训练计划，逐帧动作进入文本缓存 key。

`action_input.py` 负责严格输入校验、精确帧查表和统一门控；`prompts.rendered_action` 共用固定自然语言渲染。
输入 schema 为 `scoped_phase3_primary_action_v4`，`action_format=primary_choice_v1`。
与 Phase3 v13 choice 共用 `primary_action.py`：STOP > 首次跨线 > 纵向动作，空集为 NONE。
自动索引 `actions` 最多一个；额外 `candidate_actions` 保存原始纵横证据，runtime 先按上游确认域门控，
再投影为唯一动作（防止被挡下的机动域抢走有效纵向动作）。原始证据、有效域动作和最终动作分别审计。
`prediction/provided` 只接受单动作或空状态，不能伪造原始 oracle 证据；`choice_action_input` 严格转换
词组输出，NONE→no_action，非法/缺失→unavailable，绝不凭空补次选动作。
格式、门控、主要动作版本/源码哈希及 taxonomy 哈希进入条件身份，旧 v3/binary/旧 choice 文件需重建。
普通 RE 为 not_applicable；有效 UE/特殊 RE 无动作是 no_action；二者均无动作段，原有事实/planning 保留。
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
