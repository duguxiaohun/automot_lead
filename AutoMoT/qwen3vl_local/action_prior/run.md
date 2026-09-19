# Action prior 使用说明

以下命令在训练机的 `AutoMoT/` 目录执行。准备好 `lead_data`、已有人工标注和本地 Qwen/BEV 权重；脚本自动构建所需索引，不下载模型。

## Qwen KV 输入与摘要开关

默认 `generate_analysis=False`：**四张图像＋含自然 RS/EVENT 先验、当前速度与导航的提示词 → frozen base Qwen 一次 prefill → KV 交给轨迹 decoder**。

**输入精简与摘要限长分开处理**：high-level planning/动作目的属于输入文字，默认也会编码进 KV。
输入没有这里的 80 词上限，不会为凑词数截断场景或导航；仍须满足模型自身的 token/上下文容量。
文中 96/126 等英文词数仅衡量模板是否简洁，不是硬阈值，也不等于模型 token 数。
`MAX_ANALYSIS_WORDS=80` 及“within 80 words”仅用于显式 `--generate-analysis` 的摘要模式；
默认 prefill 使用另一份 system prompt，跳过摘要生成、格式验收、复核和 fallback。
这里“默认不生成文字”指 base 分析摘要；若使用 LoRA 先验来源，上游 Phase1/2 仍进行问答生成。
不生成 talk/CoT/摘要，不追加 assistant 回答，也不运行摘要复核或 fallback。LoRA 来源仍执行 Phase1/2 的先验问答；
`--dataset-priors` 则直接查标签，因此默认每帧没有文字生成，只有一次最终 base KV prefill。

```bash
# 显式关闭摘要（默认行为）；可与 --event-balanced 等原开关组合。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --no-generate-analysis
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --no-generate-analysis

# 保留原路径：生成摘要，然后把图像、先验提示词及摘要一起编码到最终 KV。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --generate-analysis
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --generate-analysis

# 等价环境变量，0 关闭 / 1 开启；显式 CLI 优先。
GENERATE_ANALYSIS=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors
GPU_IDS=0,1,2,3 GENERATE_ANALYSIS=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors
```

`train.sh` 和 `train.py` 同样接受 `--generate-analysis` / `--no-generate-analysis`。
`--analysis-review` / `ANALYSIS_REVIEW` 仅在摘要开启时生效，不会单独开启生成；开启摘要后，
LoRA 来源默认复核，dataset 来源在 shell 入口默认不复核，可显式覆盖。摘要失败时保留原 fallback。

模式写入 `config.json`、checkpoint 条件合同和训练计划；逐例审计用 `final_cache_content=inputs_only` /
`inputs_and_analysis` 区分，关闭时 `analysis_acceptance=disabled`。两种模式不共用文本缓存条目。
**续训、eval、probe 和闭环自动沿用 checkpoint 模式**，测试命令无需重复开关，也不接受临时切换 KV 模式。
比较两种模式需要分别训练 decoder。本次源码指纹改变，旧 run 必须用原源码恢复；开启摘要并不绕过旧 checkpoint 兼容检查。

2026-09-15 本地验证：直接图文 KV、冷/热缓存、摘要开启及复核/fallback、CLI/环境优先级、
真实 resume 配置恢复、生成预算和 `summary_disabled` 分组回归通过，Python/shell 语法检查通过。
未运行真实 Qwen/BEV GPU 训练；当前环境缺 peft 和只读 `leaderboard/team_code/mot_lead_offline_runner.py`，
真实双 LoRA 测试及完整执行源码指纹集成检查未完成。模式合同单测固定无关源码身份；正式指纹守卫没有放宽。

## 可选 high-level planning 先验

`--high-level-planning` / `HIGH_LEVEL_PLANNING=1` 把原来“保留空间、等待冲突消除”等规划措辞替换成
参考 `sft_new_loop_phase3/prompts.py` 和 `context_taxonomy.py` 的短条件性规划段落。
默认关闭，`--no-high-level-planning` / `HIGH_LEVEL_PLANNING=0` 保留原提示词；CLI 优先。
它与摘要开关独立，开启后默认仍一次图文 prefill，不增加文字生成或加载 Phase3 模型。

- 急刹前车、切入、对向侵入、路口冲突、灯故障：减速、停车/继续等待、条件允许后持续增速；增速不要求之前停车。
- 静态障碍、弱势交通参与者：再考虑相对自车方向的左右首次未来跨线，区分弯道和已完成跨线。
- 自动确认的特殊 RE2/RE3 可增加横向语义，RE5 仅纵向；不从 RS 或全 NO 推断这些事件。
- 多事件事实全部保留，只写一段规划；没有确认事件时只保留车道/导航与按当前证据调速的通用提示。

例如静态障碍对应的规划文字（前面仍有道路和障碍事实）：

> If progress is constrained, decelerate or stop/continue waiting; once the path and priority permit, sustain a speed increase, without requiring a previous stop. If needed and clear, cross left or right relative to ego's heading; consider the first future lane-boundary crossing, not a curve or an already completed crossing.

单独开启本开关时，这是供轨迹生成参考的条件性动作语义，不是该帧动作答案；不注入 Phase3 标签、未来轨迹、动作 code 或单选回答要求。
Phase3 的预测任务/数值标定时间阈值不搬进 action prompt。导航、四图和轨迹训练目标照常保留。

```bash
# 开启 high-level 替换，默认仍不生成摘要；也可追加 --event-balanced。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --high-level-planning
GPU_IDS=0 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --high-level-planning
GPU_IDS=0,1,2,3 HIGH_LEVEL_PLANNING=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors

# 显式保留原提示词；CLI 覆盖环境变量。
HIGH_LEVEL_PLANNING=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --no-high-level-planning
GPU_IDS=0 HIGH_LEVEL_PLANNING=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --no-high-level-planning

# 已备好索引的底层入口；可选再生成摘要。
bash qwen3vl_local/action_prior/train.sh --dataset-priors --high-level-planning --generate-analysis
GPU_IDS=0 bash qwen3vl_local/action_prior/train.sh --dataset-priors --high-level-planning --generate-analysis

# 续训自动恢复保存的模式，eval/probe/闭环也只读 checkpoint 设置。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt
```

开关与 planning 版本进入合同、训练计划、文本缓存及逐例审计。两种模式需分别训练 decoder；
更换先验来源的评测许可也不能绕过 planning 模式校验。缺字段旧配置按关闭处理，但旧 run 仍需原源码通过执行指纹检查。
本开关支持 LoRA 与 dataset 来源，不要求均衡采样；`condition-mode base` 无先验消融会拒绝它。
若同时开启摘要，生成与复核读取同一新版先验；失败 fallback 只重述条件性规划及简短导航，全部事实仍留在 user prompt。

2026-09-17 本地验证：137 项提示词/缓存/合同/续训/入口回归通过，Python 与 shell 语法、
无先验消融不暴露该开关的检查通过。未运行真实 Qwen/BEV 训练或闭环，尚无动作质量提升结论。

## 再加入“接下来具体采取什么动作”

2026-09-19 提示词更新：这两个开关现在共用场景目的说明。统一的减速/停车等动作不拆类，
UE1 对应保持跟车间距；UE2 对应观察相邻车道车辆、接近趋势、借道时的对向来车和通过空间，
判断绕障空隙；减速/停车本身不意味着随后变道。RE2/RE3 的目的只随独立 scene context 提供，
不会由动作索引补出。默认关闭时不添加目的；同时开启动作输入时，仍只有门控后 selected 才追加具体动作。
原有 `--high-level-planning --high-level-action-prior` 命令即可生效，无新参数。
planning 版本已升级为 `phase3_inspired_conditional_high_level_v3_compact_purpose`，进入条件/缓存身份；
请用于新 run，旧 checkpoint 仍使用原源码恢复，不能拿原 decoder 临时切换新提示词。
本次 58 项 CPU 提示词/目的/动作门控回归通过；本地缺 torch，完整 runtime/GPU 测试未运行，尚无性能提升结论。
v3 合并重复约束并缩短通用规划，具体观察条件与动作门控保持不变；R1+UE2 场景文本
120→96 个英文空白分词，加 STOP（排除导航及区块标签）164→126。80 词限制仍只针对摘要。

`--high-level-action-prior` / `HIGH_LEVEL_ACTION_PRIOR=1` 默认关闭，需要同时开启
`--high-level-planning`。**新训练不需要提供 `--high-level-action-index`**：自动复用当前 Phase3
候选与全帧事件映射，缺少产物时调用 `sft_new_loop_phase3/build_dataset.py` 的原标注逻辑生成，
再按 `(scenario, run_id, anchor)` 对齐 action 的 train/val/test。使用完整 `candidate_frames.jsonl`，
不会拿 Phase3 均衡抽样后的 `frame_index.jsonl` 代替全量标签。

只有 `special_eligible` 的 UE1–UE7、RE2/RE3/RE5 可提供动作候选；**动作索引的 `planning_contexts`
只用于作用域核对，不再补入场景事实或 planning 段落**。`gate_action` 在实际 Phase1/2 条件
（含噪声和复核结果）上统一门控 oracle/外部动作，再由同一函数渲染固定自然语言释义：

- UE 必须对应上游 YES；Phase2 的 UE 还要求所属问题域已确认有效，道路结构须在 Phase3 允许范围内。
- RE2/RE3/RE5 需要独立的 transition gate。无噪声 `--dataset-priors --high-level-planning` 会自动从 full map 提供这些上下文；
  动作文件和 Phase1/2 全 NO 都不能替代该证据。LoRA、带噪声及未开启 planning 时不自动补入。
- 三动作域只允许 DECELERATE / STOP / RESUME；已确认的五动作域才允许左右首次未来跨线。
  并发事件只确认纵向域时，会剔除候选中的横向动作。
- 只有门控后仍为 `selected` 才追加 `[UPCOMING_HIGH_LEVEL_ACTION]`。全 NO 的 `no_action`、缺帧/无效的
  `unavailable`、背景/过滤/未确认的 `not_applicable` 以及被门控挡下的动作，**均不改变原 prompt**，
  但审计状态保持区分。直接 prefill、摘要、复核及 fallback 共用此规则。

因此第二个开关本身不再改善 Phase1/2 事件覆盖，也不会补回 `PRIOR_NOISE` 删除的事件。
“保持一样”指固定同一上游和 scene-priors 条件下的图文 prompt；数据划分、来源合同和审计仍按开启模式记录。
只有开启 `--event-balanced` 时采用十个特殊桶各一份、确认普通背景两份的配额；动作开关本身不改变采样方式。
Phase3 已参与规则开发的物理路线仍强制放入 train，避免进入 val/test。

此模式使用 **Phase3 离线标注真值（`phase3_oracle`）**，其中动作标签依赖未来轨迹证据，
属于带额外真值条件的实验，不能当作 Phase3 模型预测的评测结果。没有加载 Phase3 adapter；
默认 dataset-priors 仍只做一次最终图文 prefill，零文字生成。显式开启摘要时，摘要/复核/fallback 使用同一动作。

```bash
# 自动准备动作标签并按上游门控；十桶各一份、普通背景两份，不用填写动作索引。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --high-level-planning --high-level-action-prior
GPU_IDS=0 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --high-level-planning --high-level-action-prior

# 环境变量等价写法。
HIGH_LEVEL_PLANNING=1 HIGH_LEVEL_ACTION_PRIOR=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced
GPU_IDS=0,1,2,3 HIGH_LEVEL_PLANNING=1 HIGH_LEVEL_ACTION_PRIOR=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced

# 关闭具体动作输入，保持上一版条件性 planning。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --high-level-planning --no-high-level-action-prior
GPU_IDS=0 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --high-level-planning --no-high-level-action-prior

# 底层训练入口也会自动准备动作及其依赖，普通 uniform 采样保持不变。
bash qwen3vl_local/action_prior/train.sh --dataset-priors --high-level-planning --high-level-action-prior
GPU_IDS=0 bash qwen3vl_local/action_prior/train.sh --dataset-priors --high-level-planning --high-level-action-prior

# 续训自动恢复保存的开关和索引，不重新标注。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt
GPU_IDS=0 bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt
```

自动缓存位于 `checkpoints/action_prior_prepared/` 的 `phase3_*`、`full_*`、`actions_*` 目录，
使用锁、校验和、临时目录与原子发布。需要本地原始 LEAD 数据及现有人工标注；不会下载数据或训练 Phase3。
先剔除异常时长 route，再对齐动作。旧缓存规则/内容不匹配时隔离保留并重建；显式输入不静默替换。
来源、规则、三 split 内容和文件 SHA256 进入 checkpoint 合同，逐帧动作进入文本缓存 key。
训练计划记录来源、真值条件属性及各 split 的原始状态/覆盖统计；运行审计分别记录
`high_level_action_input`（原始）、`high_level_action`（实际注入）、`high_level_action_gate`（原因/接受上下文/剔除动作）。
原始覆盖不等于实际注入率，训练/评测按同一门控结果统计。

`--high-level-action-index` / `HIGH_LEVEL_ACTION_INDEX` 仅保留为路径搬迁或高级外部输入接口，
正常开启不需要它。自动索引搬迁须同时携带同目录 `manifest.json`，full map 搬迁另用
`--event-balance-index`；resume/eval/probe 只接受与 checkpoint 同内容的产物。不能在同一个 decoder
上临时切换动作开关或替换标注，旧 run 仍需原源码恢复。

高级输入使用 `scoped_phase3_primary_action_v4`，必须声明 `action_format="primary_choice_v1"`，并保留精确帧身份、
来源、status/actions、`event_status`、`event_buckets`、`planning_contexts`；后者仅为作用域元数据。
`actions` 最多一个。自动 oracle 索引额外保存 `candidate_actions`，门控后按共享规则选主要动作：
**STOP > 首次跨线 > 纵向动作 > NONE**；不能先压掉纵向证据再门控。

后续 v13 choice 预测经 `choice_action_input(text)` 转换：动作词组→selected，NONE→no_action，
非法输出→unavailable，再携带预测时采用的 scope 输入 `PriorEngine.condition`。
外部 prediction/provided 不接受复合动作或 oracle candidate_actions；旧 binary/旧 choice 索引不能改名混用。
普通 RE 不调用特殊动作分支（not_applicable）；有效 UE/特殊 RE 的 NONE 保留事件和 planning，仅不追加动作段。

当前仍没有在线 provider，Bench2Drive/CARLA 拒绝此模式。接口语义统一不保证模型准确率，
也不能用同一 decoder 临时替换来源规避合同。新训练自动生成 v4 动作索引；旧 run 用原源码恢复。
完整示例与边界见 [Phase3 v13](../sft_new_loop_phase3/V13_PRIMARY_ACTION_20260919.md)。

2026-09-18 门控修订验证：315 项针对性回归通过，覆盖七类 UE 上游确认/拒绝、RS/问题域失效、
RE 独立 transition gate、组合动作域投影、真实 confusion/invalid 噪声及冷/热缓存的完整 prompt 对比。
全 NO、不可用、缺帧、背景及被挡下动作在直接 prefill/摘要/复核/fallback 下均不追加文字；
binary/choice 格式校验、自动准备、合同/续训与 CLI 回归通过，Python/shell 语法和 diff 格式检查通过。
本机仍缺少 PEFT 和只读 `leaderboard/team_code/mot_lead_offline_runner.py`，此前全目录测试有19项因此失败；
本次未运行真实模型、全量数据构建或闭环，不将合成回归视为 oracle/预测迁移效果验证。

## 开始训练

```bash
# 数据集标定先验：自动准备标签、训练并完成 test/probe。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors

# 开启 UE/特殊 RE 均衡采样，只需增加一个开关。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced

# 默认自动选卡；需要指定四张卡时：
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced
```

`--event-balanced`：UE1–UE7、RE2、RE3、RE5 各一份，确认的普通 RE 合计两份。
默认每帧单 epoch 最多重复 8 次，自动确定可行预算，并优先覆盖不同帧。省略此开关就是自然采样；`--no-event-balanced` 可覆盖环境变量中开启的设置。

脚本自动准备 action 索引、Phase1 标签索引、先验标签以及均衡采样所需的候选/full map；已存在的有效结果会复用。
这只构建数据，不训练 Phase1/Phase3。首次准备可能较慢，终端和 pipeline 日志会显示进度。
不需要手写 `DATA_DIR` 或 `EVENT_BALANCE_INDEX`。

如果旧版在准备阶段报 `staging.rename(...)` 的 `FileExistsError`，同步新版源码后直接重跑原命令。
候选与 full map 发布时会重新校验已存在的目录；并发结果完整且内容一致时复用，残缺缓存移到
`checkpoints/action_prior_prepared/.invalid-*` 保留后重建。不要删除 `.prepare.lock` 来解锁：
进程退出会自动释放锁，文件残留不妨碍重跑；等锁时会打印 `[prepare] waiting for cache lock`。
该恢复仅作用于自动准备的缓存，显式指定的 `--event-balance-index` 仍严格校验。
不同内容的完整发布冲突、权限、磁盘空间及 I/O 错误仍会中止并报告原因。

```bash
# 使用 Phase1/2 LoRA 现场生成先验（耗时更长，需已有 LoRA 权重）。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --event-balanced
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --event-balanced

# 缩短本次训练，例如训练 3 个 epoch；其它 train.py 参数同样可直接追加。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --num-epochs 3
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --num-epochs 3

# 可选：向标定先验注入 10% 噪声。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --prior-noise 0.1
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --prior-noise 0.1
```

均衡采样本身不改变 Qwen 提示词。无噪声 `--dataset-priors --high-level-planning` 自动注入已确认特殊 RE 的场景描述及目的，独立于采样是否均衡；全流程、train.sh 和直接 Python 都自动准备所需映射。这种模型使用离线场景标注，不能用于闭环。

旧 `--event-balanced-scene-priors` / `--no-event-balanced-scene-priors` 和 `EVENT_BALANCED_SCENE_PRIORS` 已移除；旧命令请删除这些参数。内部同名布尔字段仅用于保存合同，不再是用户开关。续训/评估恢复原配置，不自动迁移旧 run；源码指纹变化后旧 checkpoint 仍须原源码。噪声实验不补干净 RE，保持已有噪声含义。

## 查看 TensorBoard 和输出

另开一个终端：

```bash
# 最新一个 run：
bash qwen3vl_local/tb_serve.sh checkpoints/action_prior/latest/tb
# 指定端口：
TB_PORT=6006 bash qwen3vl_local/tb_serve.sh checkpoints/action_prior/latest/tb

# 同时看 action_prior 的所有历史 run（包含 latest 软链接）：
bash qwen3vl_local/tb_serve.sh checkpoints/action_prior
# 看 checkpoints 下所有实验：
bash qwen3vl_local/tb_serve.sh checkpoints
```

启动器默认只在最多 4 层内枚举标准的 `tb/`、`eval_tb/` 目录（不逐文件扫描模型和权重），解析 `latest -> run_时间戳` 软链接并显式交给 TensorBoard；因此上面两个父目录命令在 TensorBoard 2.21 fast loader 下也能显示 run，且传 `checkpoints/` 不会因全量检索权重而长时间不显示端口。启动时会先打印快速发现提示和发现的 event 叶目录数。若某个旧实验将 event 放在非标准、超过 4 层的位置，再显式使用完整扫描：

```bash
TB_DISCOVER_DEEP=1 bash qwen3vl_local/tb_serve.sh checkpoints
```

若训练刚创建、尚未写出 event，待首次写入后重启一次启动器即可把新的子 run 纳入列表。

打开终端显示的地址。主要看 `train/loss`、`train/samples_seen`、`val_epoch/route_ade_m` 和 `val_epoch/waypoint_ade_m`；均衡训练还可查看各事件组指标。FM loss 只反映训练拟合，轨迹质量以验证 ADE/FDE 为准。

TensorBoard 的横轴是 optimizer step，不是数据量；`train/samples_seen` 是用于核对实际累计 case 呈现数的独立 tag，不能直接被 Scalars 页面选作横轴。不同 run 的曲线密度/平滑程度因而不必然可比：训练日志每 `logging_steps` 个 update 写一个“该窗口的样本均值”，而 Scalars 的 smoothing 又按**点数**而非样本数做 EMA。看真实性能时先把 smoothing 设为 `0`，并核对 `samples_seen`、相同的 GPU 数与 `GRAD_ACCUM`；事件组还应同时查看同路径的 `/samples`，小分母的组损失天然更抖。

很长的 run 还可能被 TensorBoard 对 scalar 的默认 reservoir 抽样而显得稀疏。需要逐点查看时用（`0` 表示不抽样；大日志会增加浏览器内存和加载时间）：

```bash
TB_SCALAR_SAMPLES=0 bash qwen3vl_local/tb_serve.sh checkpoints/action_prior
```

若目标是让不同总步数的**新实验**在图上各有近似相同数量的、等间隔的点，给每个 run 预先取相同目标点数 `N`，设置 `LOGGING_STEPS=ceil(总 optimizer steps / N)`（例如想保留约 250 点，14,000 step 的 run 取约 `56`）。这会改变每点的平均窗口宽度，适合看总体收敛趋势；要比较短期噪声，应保持相同的 `LOGGING_STEPS` 与有效 global batch，而不是人为下采样。

每次训练使用独立的 `checkpoints/action_prior/run_时间戳/`，`latest` 指向最新 run：

| 文件/目录 | 用途 |
| --- | --- |
| `train.log`、`pipeline.log` | 训练/数据准备日志 |
| `best.pt`、`latest.pt` | 验证最优/最新 checkpoint |
| `tb/` | TensorBoard |
| `sampling/` | 每轮采样配额、唯一帧及重复次数 |
| `test/`、`probe/` | 最终测试指标和案例 |

## 续训

```bash
# 继续训练并完成最终 test/probe；采样开关和索引从原配置恢复。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt

# 兼容环境变量写法；CLI --resume 优先于 RESUME。
RESUME=checkpoints/action_prior/latest/latest.pt bash qwen3vl_local/action_prior/run_full_pipeline.sh
GPU_IDS=0,1,2,3 RESUME=checkpoints/action_prior/latest/latest.pt bash qwen3vl_local/action_prior/run_full_pipeline.sh

# 仅续训：
bash qwen3vl_local/action_prior/resume.sh checkpoints/action_prior/latest/latest.pt
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/resume.sh checkpoints/action_prior/latest/latest.pt

# 底层训练入口也会先恢复配置，包含原摘要开关；兼容 --resume=路径 和 RESUME=路径。
bash qwen3vl_local/action_prior/train.sh --resume checkpoints/action_prior/latest/latest.pt
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/train.sh --resume checkpoints/action_prior/latest/latest.pt
```

续训使用原代码版本和原配置。更改采样算法或实验条件时开新 run；不要修改旧 checkpoint 的合同。
`--resume 路径`、`--resume=路径` 与 `RESUME=路径` 共用专用恢复入口，恢复原 LR、epoch、累积、
采样课程、标签路径与卡数默认值，不执行新训练的数据构建或 LoRA 重选。
`train.sh` 同样在拼接新训练默认参数前转入共用恢复入口，不再用默认
`--no-generate-analysis` 覆盖原 run 的 `True`。仅显式环境变量才作为覆盖传入，CLI 优先；
不要显式切换摘要模式续训，条件合同仍会拒绝不同 KV 模式。
pipeline 在读取配置前解析 checkpoint 的真实路径；即使另一个实验修改 `latest`，
续训和最终 test/probe 也固定使用原 run 及其 `best.pt`。缺值或不存在的 checkpoint 会提前报错。

```bash
# 搬迁后仅覆盖路径；full map 要连同 manifest.json 一起移动，内容身份仍严格核验。
bash qwen3vl_local/action_prior/run_full_pipeline.sh \
  --resume checkpoints/action_prior/latest/latest.pt \
  --data-root lead_data --data-dir checkpoints/action_prior_data \
  --prior-labels checkpoints/moved_labels/prior_labels.jsonl \
  --event-balance-index checkpoints/moved_event_map/full_event_mapping.jsonl
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh \
  --resume checkpoints/action_prior/latest/latest.pt \
  --data-root lead_data --data-dir checkpoints/action_prior_data \
  --prior-labels checkpoints/moved_labels/prior_labels.jsonl \
  --event-balance-index checkpoints/moved_event_map/full_event_mapping.jsonl
```

上述标签搬迁示例适用于 dataset-priors run；LoRA run 不传 `--prior-labels`。
显式 `--data-root`、`--data-dir`、`--model-dir`、`--lead-bev-ckpt` 及对应环境变量，
连同标签/full-map 路径覆盖都会贯穿续训与最终 test/probe，CLI 优先。
未显式指定时沿用原配置和 checkpoint，脚本默认路径不会覆盖旧 run。

## 独立测试

```bash
bash qwen3vl_local/action_prior/eval.sh --checkpoint checkpoints/action_prior/latest/best.pt
GPU_IDS=0 bash qwen3vl_local/action_prior/eval.sh --checkpoint checkpoints/action_prior/latest/best.pt

# 小样本可视化：
bash qwen3vl_local/action_prior/probe.sh --checkpoint checkpoints/action_prior/latest/best.pt
GPU_IDS=0 bash qwen3vl_local/action_prior/probe.sh --checkpoint checkpoints/action_prior/latest/best.pt
```

默认沿用 checkpoint 的先验设置，test 不进行均衡重采样。闭环、审计、权重迁移等少用操作见 [AUDIT.md](AUDIT.md)，模型与采样合同见 [DESIGN.md](DESIGN.md)。

## 两个无先验消融使用同一均衡课程

`action_expert_ablation/qwen_simple` 与 `bev_only` 现在直接引用本目录的开关解析、候选/full-map
自动准备、采样/预算/合同校验、训练循环和事件桶指标。调整比例只改 `event_balance.py` 的
`EVENT_BALANCE_WEIGHTS`，不需要复制逻辑；三个实验须统一源码版本、action 索引、seed 与训练预算。
主线本脚本默认每次创建 run 索引；配对实验请显式 `DATA_DIR=checkpoints/action_prior_data`，
与两个消融的默认索引对齐。主线 planning 先验和分析 prompt 只由本目录维护；消融保持各自的无先验输入定义。

```bash
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced
```

完整三组配对、课程参数、续训/索引搬迁命令见[消融运行文档](../action_expert_ablation/run.md#与主线完全共用-event-balanced)。

2026-09-14 复审后，三组同时首次启动也共用 action 索引构建锁；索引必须同时有 manifest 和三个 split。
均衡采样直接使用 Phase3 当前开发路线名单隔离 holdout（当前 709 组），验证覆盖按全部真实事件桶统计，
不会把 `special_filtered` 误报成缺桶。修改正整数权重表后，自动 epoch 预算与配额均同步更新。
本次执行源码和有效数据划分有变化，使用新 run；旧 checkpoint 仍用原代码恢复。


2026-09-14 验证：主线与消融回归共 **291 passed / 1 skipped**，Python/shell 语法检查通过。
新增测试执行真实 `resume.py` 配置恢复，仅替换模型 launcher，覆盖三种续训写法、CLI 优先、
路径搬迁、latest 改指与非法 checkpoint 提前失败。跳过项为缺少 TensorBoard 依赖的事件文件测试；
尚未运行真实 Qwen/BEV GPU/DDP 训练，首次上机先使用消融文档中的独立短预算 pipeline。
