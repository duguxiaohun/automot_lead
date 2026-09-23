# Action prior 使用说明

## 2026-09-23 RGB输入和稀少组合审计同步

三条Action入口共用v23输入质量过滤：anchor<4排除；单图/四图/BEV-only用同一有效帧范围。
有效split在开发路线隔离后，从未曝光train整组补足事件支持，目标每事件32帧、5物理组；
原数据文件不改，实际迁移/缺额见训练计划的 `event_balance_source_audit.split_support`。
`--action-balanced`按下述全局动作比例采样，重复上限与event统一为8；合法稀少动作不改KEEP/UNCOND。
文字动作先验自动复用更新后的Phase3因果句，token/文字开关仍独立；默认event-balanced不变。
需按新hash准备候选/full map并新开run，旧run使用原源码。
详细RGB判断、容量回放与限制见 [v23审计](../sft_new_loop_phase3/V23_RGB_SUPPORT_20260923.md)。

## 2026-09-23 全局动作均衡与小事件温和加权

默认 `--event-balanced` 保持十特殊事件各一份、普通背景两份。加入 `--action-balanced` 后：

- 六种语义动作（DECELERATE、STOP、RESUME、LEFT、RIGHT、KEEP）在全局等量，背景 UNCOND 保留总预算的1/6。
  六类配额因整数余数最多差1，余数随epoch轮换；不要求各事件1:1。七种token并非全部等量。
- 在动作内部，小事件按 `min(2, sqrt(最大事件帧数 / 当前事件帧数))` 加权。权重最高2倍，
  避免4帧之类的小事件×动作格子被强拉到与大格子一样多；这是初始实验设置，不代表效果最优。
- 并发帧按“动作＋支持该动作的事件集合”只入一个池；事件权重取均值，容量和权重不重复相加。
  池配额按帧数×权重分配，饱和缺额回到同动作其它池；池内优先不同帧和物理路线，硬上限仍检查。
- 两模式新训练的单帧重复上限统一为8；显式值与保存值优先。8是允许上限，不要求每帧重复8次。
  默认预算0使用**同一数据、repeat cap和world size下event模式的自动预算**，action不再自动缩到1/4。
  动作容量不足会在模型加载前列出缺口及最低重复上限，不静默缩短epoch或放宽限制。

只保留event/action两种采样；`ACTION_BALANCED=1`选action、`ACTION_BALANCED=0`选event，CLI优先。
`--no-event-balanced`、`--no-action-balanced`、`EVENT_BALANCED=0`和uniform仍拒绝。
采样不会自动开启token或文字动作先验；token关闭时动作标签只用于选样。验证/测试仍按有效split遍历。
完整epoch满足全局比例，micro-batch、rank子序列或max_train_steps截断前缀不保证等量。

若要复现旧run的116256次/轮，给两组都显式加 `--event-balanced-epoch-samples 116256`；
4卡、grad_accum_steps=16时为1817次更新/轮。这个预算**仍需两模式各自通过容量校验**，
最新v23过滤/划分后不保证event模式上限8能满足；报错时应选共同可行预算或明确调整上限。
仅改成action模式不保证恢复某个历史run的精确数量，默认对齐的是当前同源event基线。

`training_plan.json` / `sampling/epoch_*.json`记录动作配额、事件权重、事件归属次数、真实唯一帧和重复直方图。
并发帧的配额审计每次只归属一个事件，原事件事实全部保留；`group/event_balance/*`事实指标可以重叠。
小类审计阈值仍只供复查，不改KEEP/UNCOND真值。旧两层采样/上限2的run使用原源码，当前方案需新run。
算法详见 [全局动作均衡说明](GLOBAL_ACTION_BALANCED_20260923.md)。

```bash
# AutoMoT/ 下；默认event，同源自动预算
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors
# 全局action比例 + 温和事件加权 + token + 单当前图
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --action-balanced --high-level-action-token --rgb-frame-count 1
# 可在两组追加同一个通过容量预检的 --event-balanced-epoch-samples N
```

## 2026-09-22 动作 token 弱分离正则

开启 `--high-level-action-token` 的新训练默认增加弱分离：`--action-token-separation-weight 0.01`、
`--action-token-separation-margin 0.5`。仍为七个可学习的 1024 维向量，KEEP 不细分，UNCOND 也参与。
对完整 embedding 表的 21 对不同类别计算 `mean(relu(cos(e_i,e_j)-margin)^2)`，
总训练 loss 为 `FM loss + weight * separation loss`。只惩罚过高相似度，不要求动作全部正交；
归一化只用于正则分支，BEV concat 与注意力输入不变。该设计参考防坍塌思想，**不是 LeJEPA/SIGReg 的复现**。

正则使用 FP32（包括 BF16 autocast 内），所有类别等权，每个 micro-step 与 FM loss 一起除以实际累积窗口长度，
DDP 不额外乘 world size。标签/Phase3 规则、采样和 Qwen prompt 不变。
验证仍计算原始 FM MSE 和轨迹指标，best 仍按原 ADE 策略选取，不把正则加入验证分数。
权重设 `0` 可做无正则对照；token 关闭时正则不执行。CLI 优先于环境变量
`ACTION_TOKEN_SEPARATION_WEIGHT` / `ACTION_TOKEN_SEPARATION_MARGIN`。

TensorBoard 和 `training_audit/windows` 记录 `fm_loss`、`action_token_separation_loss`（未加权）、
`action_token_separation_weighted`（加权），`loss` 是总训练目标。
`action_token/cosine/<类别>__<类别>` 共 21 对、`cosine_max/mean` 和每类 `norm` 在首步、日志窗口、轮末与最终步记录；
`action_token_initial.json` 保存本次进程起点（恢复时为恢复步）的范数/相似度。
审计 ZIP 包含近期窗口。权重为 0 时两个 separation loss 日志为 0，相似度仍监控。

这是软约束，不保证 decoder 一定利用 token，也不保证轨迹指标提升；相似度不能代替正确/替换 token 的固定噪声轨迹对照。
默认系数是实验起点，尚未做真实模型/GPU效果验证。算法版本、权重和 margin 写入条件合同与训练计划，
更改需新 run；旧 run 使用原源码，不能跨源码直接续训。缺这些字段的历史配置按权重 0 解释，仍保留严格源码检查。

```bash
# 在 AutoMoT/ 下执行；默认四图，开启 token 即启用弱分离
bash qwen3vl_local/action_prior/run_full_pipeline.sh --high-level-action-token
# 单当前图 + token + 弱分离；固定四卡可前置 GPU_IDS=0,1,2,3
bash qwen3vl_local/action_prior/run_full_pipeline.sh --high-level-action-token --rgb-frame-count 1
# 独立新 run 的无正则对照
bash qwen3vl_local/action_prior/run_full_pipeline.sh --high-level-action-token --action-token-separation-weight 0
```

## 2026-09-21 后续：防止候选容量与 token 支持不足

自动准备的 Phase3 候选容器现在独立于 SFT 问答 split，避免其 val/test 空桶阻塞 Action；Action 继续使用自己的三 split、full map 和开发隔离。准备缓存身份升级，需要按当前源码生成派生产物，旧 run 用原代码。

三条 Action 路径共享训练前 token 支持检查：七类零计数、独立帧/物理路线、UNCOND 原因、缺失动作和 eval 有而 train 无的动作都会记录。开启 token 且训练集全为 UNCOND 时拒绝；每轮采样后再次检查，覆盖 DDP 尾部截断，写 `sampling/epoch_*.json`。稀缺动作只报告，不补 KEEP 或伪造覆盖。事件均衡不可行时给出请求量、重复上限、桶容量，自动预算0仍由联合分配器求解。

新增无模型的数据检查器 `audit_data_capacity.py`，可用真实配置回放七轮、1/4卡的配额和支持；不启动GPU或NCCL。全量结果和命令见 [容量审计](../sft_new_loop_phase3/CAPACITY_AUDIT_20260921.md)。新产物目前在独立审计目录，未覆盖现有生产目录。

## 2026-09-21 Phase3 v21 标定与文案同步

当前共享来源为 Phase3 v21 / `current_wait_first_crossing_v9_confirmed_pullaway`。确认释放制动且及时持续起步的近零速帧不再一律标 STOP；缺控制仍保留等待判定。Action 读取完整 candidate，文字动作和 token 复用同一主要动作投影，原有 STOP＞首次跨线＞速度优先级不变。具体判据及审计边界见 [Phase3 v21](../sft_new_loop_phase3/V21_CALIBRATION_20260921.md)。

主线自然先验同步修订：UE1 可处于前车减速后的响应、等待或恢复；信号异常描述为已经给定的信号系统故障，不要求可见灯具损坏，不把查询灯态 None 或越线作为物理故障证明。普通及多事件紧凑文案共用新语义，直接 prefill、摘要、复核与 fallback 同步。`PREFILL_VERSION=natural_scene_prior_input_only_v2_response_and_system_fault`，`ANALYSIS_VERSION=natural_scene_prior_concise_summary_v7_response_and_system_fault`，参与缓存/恢复合同。Phase1/2 视觉检测问题的判定合同未在本次改写；给定场景的动作规划与从 RGB 检测事件须分别评估。

| 使用路径 | v21 的影响 |
| --- | --- |
| 主线自然先验 | 上述事件文案更新；开启文字动作先验后使用共享 v21 动作及条件性因果句 |
| 主线 / qwen_simple / bev_only 的动作 token | 开关仍默认关闭；开启时使用当前 candidate/full map 的主要动作，KEEP 与 UNCOND 分开 |
| qwen_simple | 保留原简短导航 prompt；不注入 Phase3 问答或 RS/EVENT 先验 |
| bev_only | 无 Qwen prompt；标定只在启用相应采样/动作条件时发挥作用 |

Action 导航继续来自其 LeadMoT 输入合同，不能把 Phase3 原始 ego 坐标轴或字段模板直接复制过来。三条路径使用 full map 时共享更新后的开发路线隔离；event/action 两种模式都使用 full map 同步隔离，不改基础 split 文件。

**代码同步不等于已有数据更新。** 新训练自动准备路径包含当前 mapping/source 身份，会准备新候选和 full map；显式指定旧产物仍须通过哈希校验。已有生产索引、full map、动作索引、文字 KV 缓存和 checkpoint 本轮未全量重建或回写。请更新整套源码后为新条件启动新 run；不能用 `SKIP_BUILD`、手改 manifest 或旧缓存跳过合同。旧 run 用原源码恢复。

本轮验证：150 项 Action 提示词/准备/token 测试通过；另 14 项消融入口测试通过（未执行依赖本机缺失离线 runner 的 fingerprint 测试）。实际 v21 开发候选 7900 行通过 Action 读取、文字动作及 token 投影；其中 84 个确认起步帧投影为 RESUME 64、LEFT 18、RIGHT 2。文字动作的 KEEP 仍保持 `no_action` 协议。此检查用候选 context 构造 eligible 记录验证消费者，不替代实际生产 full map 构建和门控验收。记录位于 `/tmp/p3audit/action_v21_sync/consumer_replay.json`；未运行全量生产构建、真实 Qwen/BEV 或 GPU 训练。

## 2026-09-21 动作 token 与单当前图

三条入口共用两个独立选项（只用于新 run）：

- `--high-level-action-token` / `HIGH_LEVEL_ACTION_TOKEN=1`：默认关闭。五种变化动作＋一个 KEEP＋UNCOND，共七类；`Embedding(7,1024)` 在 BEV projector 后沿 token 维拼接一个向量，默认序列 142→143。UNCOND 同样可学习、参与注意力，不是 padding。FM 主损失不变，新训练另加上述可关闭的弱分离正则；不改变采样权重或 Qwen 文本，不自动开启旧 `--high-level-action-prior`。
- `--rgb-frame-count 1` / `RGB_FRAME_COUNT=1`：只向 Qwen 提供当前 anchor 的一张完整 1152×384 三视角拼接图，不裁前视、不复制成四图。默认 `4`；`--rgb-frame-count 4` 恢复四张连续图。BEV 继续单帧 RGB＋LiDAR。`bev_only` 接受同一参数，但本来就不使用 Qwen 图像历史，单/四图不改变它的 BEV 条件。

动作来自当前 Phase3 原始候选和完整帧映射，不从均衡抽样题库补标签、不运行 Phase3 模型。新训缺映射时自动复用/生成；三条路径共用主要动作投影（STOP＞首次跨线＞速度＞KEEP），不走仅主线才有的 Phase1/2 预测门控。KEEP 仍要求题域内证据完整，但不细分类；普通 RE、被过滤、未确认及映射覆盖外帧为 UNCOND，原因分别审计。eligible 帧缺候选、证据不完整、冲突或哈希不符均报错。

这属于 `phase3_oracle` 离线条件实验。token 经共用 Prefix-KV attention 与 BEV/status/query 交互；`qwen_simple` / 主线同时读取冻结 Qwen KV，`bev_only` 用空 prefix。embedding 由轨迹损失训练，路由到 AdamW，冻结模型保持冻结。`training_plan.json` 记录七类覆盖与源合同，验证按 `group/action_token/*` 报告 ADE/FDE 等指标，逐例保存 UNCOND 原因。

单图模式覆盖主线 Phase1/2 LoRA 问答、可选摘要和最终 base prefill。相关提示词明确只有当前图，移除请求比较历史图像的指令；adapter 原训练元数据/输出格式不回写。这会改变原多帧 LoRA 的推理输入分布，应与四图分别新训 decoder 并报告该条件变化。使用 `--dataset-priors` 时没有上游 LoRA 问答。

以下命令默认当前目录为 `AutoMoT/`，自动选择空闲 GPU。共享 `DATA_DIR`、采样方式、full map、seed、卡数和预算；特别是接入 Phase3 映射会隔离已参与开发的物理路线，不能拿未做同一隔离的旧基线比较。下面均启用同一事件均衡课程以保持有效 split 一致。

```bash
# 主线：四图＋动作 token；关闭旧文字动作先验以单独测 token
DATA_DIR=checkpoints/action_prior_data bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --high-level-action-token --no-high-level-action-prior
# 同样实验只看当前一张图
DATA_DIR=checkpoints/action_prior_data bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --high-level-action-token --no-high-level-action-prior --rgb-frame-count 1
# 显式固定四张 GPU（可选）
GPU_IDS=0,1,2,3 DATA_DIR=checkpoints/action_prior_data bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --high-level-action-token --rgb-frame-count 1
# 基线：四图，关闭两种动作条件，另开 run
DATA_DIR=checkpoints/action_prior_data bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --no-high-level-action-token --no-high-level-action-prior --rgb-frame-count 4
```

CLI 优先于环境值；两开关可独立组合，也可同时打开文字动作先验做单独实验。数据索引仍保留四图构建记录，单图在运行时选当前帧，不需只为图数重建 action split。

resume / eval / probe 从 checkpoint 恢复开关、图数、词表、映射和候选内容身份。续训不重建标注，不能在旧 run 中途切图数或动作条件；旧 run 必须用其原源码。搬迁用现有 `--event-balance-index` 指定相同内容的 full map，并确保其 manifest 中引用的 Phase3 candidate 及同目录 manifest/frame_index/candidate_counts 可访问。动作 token 尚无在线 provider，CARLA/Bench2Drive 明确拒绝 oracle checkpoint。CPU 小模型验证不代表真实 Qwen/BEV、多卡吞吐或模型效果已验证。


## 2026-09-20：Phase3 manifest 衔接修复

报错 `candidate manifest is stale, incomplete, or not the current Phase3 frame-index artifact` 的已确认原因：
Phase3 构建器输出 `sft_new_loop_phase3_frame_index_v5_binary_keep`，Action 校验器仍写死 v3。
因此刚构建完成的候选也会被拒绝；不是本次 RGB 损坏，也不是临时目录路径必须提前存在。

现两边共用 `build_dataset.FRAME_INDEX_FORMAT`，测试 fixture 同步；增加真实Phase3构建/写manifest
经过Action校验、原子发布和复用的回归，避免模拟产物和校验器一起停留在旧格式而测试仍通过。
旧/未知format、旧mapping hash、缺失或错误frame-index及候选计数不一致仍然拒绝，报错分别给出原因。

更新整套源码后重跑原 Action 命令即可，自动缓存会按当前来源身份准备，不需要手改manifest或删除所有缓存。
这轮修改改变映射哈希，不直接复用旧候选；旧训练run需保持原源码。
`TemporaryDirectory` 通常会在失败退出时清理日志中的 `.candidate-*` 目录，所以不能承诺该次扫描结果仍可恢复。

验证：46项Action数据准备测试、474项Phase3无torch测试通过；还以实际小集合产物走通
1148候选→3657全帧场景映射→1143动作索引，并验证三层缓存二次复用。
小集合 Action split 仅包含 train；候选由真实构建器生成后复制进准备器临时目录，避免重复同一小集合扫描，
full map和动作索引均调用正式实现。产物位于Phase3 `probe_output/action_manifest_bridge_20260920/`，不入库。
尚未运行远端全量构建或GPU训练，不把这些检查当成完整生产验收。


优化细节已统一为默认值，无需追加新开关。每个 epoch 训练结束及完整验证后，自动更新当前 run 的 **`training_audit.zip`**；训练被中途终止时可直接带走此文件审计。包内有进度、各轮训练/验证指标、loss/LR/更新幅度和实际配置，详见 [默认训练与中途审计](OPTIMIZATION.md)。

新训练默认 `muon_adamw + cosine_restarts`：隐藏矩阵 Muon、其余参数 AdamW，默认7轮，首轮更新步数的5%用于warmup（包含在首周期内），三个周期占1/2/4轮，在第1/3/7轮末降到0。主线与两个消融共用实现。参数、四组合对照和续训边界见 [共享优化说明](OPTIMIZATION.md)。

以下命令在训练机的 `AutoMoT/` 目录执行。准备好 `lead_data`、已有人工标注和本地 Qwen/BEV 权重；脚本自动构建所需索引，不下载模型。

## Qwen KV 输入与摘要开关

默认 `generate_analysis=False`：**四张图像＋含自然 RS/EVENT 先验、当前速度与导航的提示词 → frozen base Qwen 一次 prefill → KV 交给轨迹 decoder**。

**输入精简与摘要限长分开处理**：RS/EVENT 与所选动作因果句属于输入文字，默认也会编码进 KV。
输入没有这里的 80 词上限，不会为凑词数截断场景或导航；仍须满足模型自身的 token/上下文容量。
文中 75/55 等英文词数仅衡量模板是否简洁，不是硬阈值，也不等于模型 token 数。
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

## 一句所选 high-level 动作及场景原因

`--high-level-planning`、其否定形式及 `HIGH_LEVEL_PLANNING` 已移除；旧命令会明确报错，
请删去它们。保留默认关闭的 `--high-level-action-prior` / `HIGH_LEVEL_ACTION_PRIOR=1`，
独立于 `--generate-analysis` 和采样方式，不再需要其它 planning 开关。

保留原来的自然 RS/EVENT 介绍，在同一 `[SCENE_DESCRIPTION]` 尾部只追加一句 `Next action: ...`。
该句直接调用当前 Phase3 action choice 的 `choice_semantics.action_description(context_id, action)`，
逐字复用所选动作的**场景条件 → 动作 → 目的**；不列所有动作，不追加所有事件的动作目的列表。
例如 R1 + UE2 + STOP（四图仍由消息构造器附加）：

```text
[SCENE_DESCRIPTION]
The vehicle is travelling along an ordinary surface-road corridor with the route ahead. A static obstacle is restricting the usable route; any bypass needs lawful clear space.
Next action: When the obstruction or passing traffic blocks progress, stop or keep waiting to avoid collision while checking passing clearance and waiting for a usable bypass gap.
[/SCENE_DESCRIPTION]
[CURRENT_NAVIGATION]
<当前速度与导航文本>
[/CURRENT_NAVIGATION]
```

同为 STOP，UE1 解释防追尾、UE2 解释等待绕障间隙、UE5 解释让侵入车辆通过、
RE3 解释等待合流/驶出空间、RE5 解释遵守停车/让行路权与等待交叉车流。
并发事件的事实仍全部保留；动作原因仅从**已经门控通过且支持所选动作**的 context 中，
按 Phase3 taxonomy 固定顺序选一个，`high_level_action_gate.description_context_id` 记录选择。
这是一种明确的并发文本取舍，不把它当作唯一真实驾驶意图；保留 choice 的条件性表述，不据目的断言安全间隙已存在。

```bash
# 自动准备动作标签及 full map，十个特殊桶各一份、普通背景两份。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --high-level-action-prior
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --high-level-action-prior

# 环境变量等价；CLI 优先。省略 --event-balanced 仍是默认事件均衡。
HIGH_LEVEL_ACTION_PRIOR=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors
GPU_IDS=0 HIGH_LEVEL_ACTION_PRIOR=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors

# 显式关闭，恢复自然 RS/EVENT 输入；也不再自动补特殊 RE。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --no-high-level-action-prior
GPU_IDS=0 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --no-high-level-action-prior

# 已准备好 action 数据索引时，底层入口也自动准备动作依赖。
bash qwen3vl_local/action_prior/train.sh --dataset-priors --high-level-action-prior
GPU_IDS=0 bash qwen3vl_local/action_prior/train.sh --dataset-priors --high-level-action-prior

# 新版本 run 恢复保存配置和索引，不重新标注。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt
GPU_IDS=0 bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt
```

**来源与门控。** 新训练无需填写 `--high-level-action-index`：复用 Phase3 完整 `candidate_frames.jsonl`
及独立 full map，缺失时调用原构建器，按 `(scenario, run_id, anchor)` 对齐 action 的三 split。
只接受 `special_eligible` 的 UE1–7、RE2/3/5；候选缺失不反推为普通背景。
使用未来轨迹标注真值 `phase3_oracle`，保存 `privileged_action_conditioning=True`，不加载 Phase3 adapter，
不是 Phase3 模型预测。先剔除异常时长路线，已参与规则开发的物理路线仍强制 train-only。

UE 要求实际 Phase1/2 的 YES、有效问题域和兼容 RS；带噪声条件下不能补回被删除的事件。
RE2/3/5 要求独立 transition gate；仅干净的 `--dataset-priors --high-level-action-prior`
自动从独立 full map 提供这些场景事实，策略为 `dataset_action_special_re_v2`。
LoRA、带噪声或动作关闭时不自动补 RE；动作索引 scope、RS、上游全 NO 都不能替代该证据。
只确认纵向域时会先过滤横向候选，再按 STOP > 首次跨线 > 纵向动作选唯一主要动作。

**空状态保持原协议。** 此次复用最新 choice 的五种变化动作措辞；没有改轨迹标定或动作索引 schema。
Action 仍为 `scoped_phase3_primary_action_v4` / `primary_choice_v1`：NONE→`no_action`，
缺帧/无效→`unavailable`，普通背景/过滤/未确认→`not_applicable`。
这些状态或被拒绝的动作不追加句子，也不凭旧 NONE 合成新版 KEEP。
最新 Phase3 KEEP 文本仍不能直接当成该旧接口的输入；未来接真实预测时需单独对齐标签协议。
在相同上游/scene 条件下，无有效动作的 prompt 与关闭动作输入相同，场景事实仍保留。

自动缓存位于 `checkpoints/action_prior_prepared/{phase3_*,full_*,actions_*}/`，有锁、校验和及原子发布。
`--high-level-action-index` 只供搬迁或高级 prediction/provided 输入；自动索引搬迁须带同目录 manifest，
full map 搬迁用 `--event-balance-index`。外部输入也须精确帧身份和合法作用域，不能伪造 oracle candidate_actions。
原始动作、有效动作、门控理由和描述 context 分开审计；覆盖率不等于实际注入率。

直接 prefill、可选摘要、复核及 fallback 共用一句动作描述；默认仍零摘要生成。
合同绑定 `upstream_gated_phase3_causal_sentence_v3`、Phase3 choice 源码 SHA256、taxonomy、数据来源及 split；
逐帧动作/作用域也进入缓存键。续训/eval/probe 读取保存条件，不能临时切换提示词或换源绕过合同。
**新提示词用于新训练；旧 run 使用原源码恢复。** Bench2Drive/CARLA 仍因没有在线 provider 而拒绝动作模式。

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
默认每帧单 epoch 最多重复 8 次，自动确定可行预算，并优先覆盖不同帧。省略此开关仍为事件均衡；`--action-balanced` 切换为动作均衡。

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

均衡采样本身不改变 Qwen 提示词。无噪声 `--dataset-priors --high-level-action-prior` 自动注入已确认特殊 RE 的场景事实，动作目的只随门控通过的所选动作提供，独立于采样是否均衡；全流程、train.sh 和直接 Python 都自动准备所需映射。这种模型使用离线场景标注，不能用于闭环。

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
与两个消融的默认索引对齐。主线自然先验/分析 prompt 由本目录维护，所选动作因果句共用 Phase3 choice；消融保持各自的无先验输入定义。

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


### 2026-09-22 多模型同帧可视化

新增 `compare_checkpoints.sh`：只改开头的多个训练run目录，自动校验并选择best，
支持主线/bev_only/qwen_simple任意组合，分别按event/action对train/test分层选例。
脚本支持逐event/action及train/test配置案例数，0跳过；输出实际RGB投影、道路/车辆框俯视GT/多模型拼图PNG/PDF、完整历史输入、逐例简洁JSON和分桶报告到新的时间目录。
已保存第三人称CARLA图时可用显示标定JSON加入该视角；默认RGB优先同帧meta实际标定、缺失才回退名义标定；采用地面近似，不做遮挡推理。
完整用法、搬迁与验证边界见 [CHECKPOINT_COMPARISON.md](CHECKPOINT_COMPARISON.md)。

2026-09-22 对比入口补齐：结果默认写到 `AutoMoT/test/run_<时间>/`，与checkpoints同级。
默认最多自动选4张最空闲GPU，卡不足时减少并发；模型少于卡时按case拆片（4卡2模型各2份），模型多于卡则排队动态补位；
`--gpus`/`GPU_COUNT`设置自动上限，`GPU_IDS`显式pin优先。独立日志与scheduler记录，失败/中断回收本次子进程。
并发相关检查使用真实CPU子进程，未验证实际多GPU模型推理。

投影源码核对与录制标定读取见 [PROJECTION_AUDIT_20260922.md](PROJECTION_AUDIT_20260922.md)。

2026-09-22 预检静默排查：对比入口新增阶段开始/完成及每15秒心跳，记录到preflight.json/log（耗时、RSS、主线程函数位置）。
GPU在完整CPU合同与选帧计划后才启动；选帧取消全池case-ID哈希，临时标签池选完即释放。
共享训练源码及内容校验不变；4项新增无模型检查通过，未测远端真实耗时。

2026-09-22 误差搜索更新：shell默认每类每split最多50候选、最多保留5例，waypoint任一模型-GT/模型对ADE>1m或FDE>3m触发，route不参与筛选。
分批搜索满额类别停止单独派发，常驻GPU服务复用当前模型；search.json报告检查/命中/缺额，原始预测保留，早停统计不代表全量表现。
详细参数、重放条件及输出布局见 [CHECKPOINT_COMPARISON.md](CHECKPOINT_COMPARISON.md)。
