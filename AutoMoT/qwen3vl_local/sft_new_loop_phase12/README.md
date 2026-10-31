# 单个 LoRA 的 Phase1 + Phase2 独立问答实验

用户于2026-10-29明确要求新增融合实验。现有独立 Phase1/Phase2、历史权重与运行保持原状。
当前新运行合同为 `phase12_shared_lora_v4_phase2_authority_20261030`；旧v1/v2/v3冻结源码/ready不覆盖或迁移。
每个图像模式训练一个共用 LoRA：两图模型同时回答P1/P2，四图模型也同时回答P1/P2；
两图与四图仍是两个独立实验，并非把两种图数混成一个训练集。


## 2026-10-31 r11：评测与部署绑定同一缓存配置

新 `fusion_config.json` 及独立 Phase1/2/3 adapter 配置携带 `text_cache_policy`，
保存实际缓存开关、槽位、字节预算、token 上限、输入布局和分段版本。保存前核验启动配置；
融合权重额外核对 CPU-ready。拷走 adapter 后无需访问原 ready，在线加载在模型加载前检查，
每次预测、PriorEngine 闭环调用及底层 prefill 再核验环境和实际 bank。
配置不同明确拒绝，报错显示期望/实际值；不自动调整容量或缩短前缀。

旧 text-first 权重缺少合同字段时拒绝部署，不按当前默认值追认；旧 image-first 路径保持。
该字段是配置约束，不是新质量认证。改缓存配置须重新验收；完整前向对照保留在内部诊断，
不能通过切换环境变量绕过已绑定的推理路径。新 Phase3 源码清单 r11 保留所有旧清单。

下方 r10 的新建 CPU-ready → 单卡 probe 命令继续适用，需先同步当前源码并新建运行，
不覆盖已有源码快照/ready/adapter。新训练保存的权重再用于对应配置的在线评测和闭环。
不需要重建原始标签库，精度、训练反传、采样、标定和预算均保持。

## 2026-10-31 r10：两题实际驻留及跨图闭环准入（覆盖 r9 的容量与检查范围）

新文本验收协议 `text_reuse_resident_stream_v3_20261031`：默认每个模型/rank 两项、
总计 **1024 MiB**、每项最多 **8192 token**。这只是按需分配的保留缓存上限，
不预分配 1 GiB，不包含模型、图片、解码状态及临时 clone。多个 adapter 各有自己的 bank。
可通过 `AUTOMOT_TEXT_PREFIX_CACHE_ENTRIES/MIB/TOKENS` 显式设置正整数；
CPU-ready 绑定三项实际配置，后续 probe/训练/评测配置不一致拒绝。不能降低设置绕过验收。
旧失败运行的诊断会从原回执恢复容量；旧回执无容量字段时沿用其冻结源码默认，不继承当前 shell 设置。

默认扩大不代表任意问题一定放得下。准入实际计算每题完整文本缓存的 `prefix_bytes`，
报告 `required_bytes` 和 `required_entries`；两题总和超过预算、只有一个槽位、
token 上限不能容纳完整前缀，都明确拒绝，要求用户根据实际空闲显存调整配置再新建运行。
不会自动突破用户预算，也不通过只缓存共同短前缀冒称两种完整问题已驻留。

共享训练前 probe 与正式评测首例，从既有验证/评测成员中读取另一组真实 RGB，
保持当前两题文本不变，在**同一个与生产容量相同的 bank**执行：

```text
A/图组1 → B/图组1 → A/图组2 → B/图组2 → A/图组1 → B/图组1
```

必须前两次 miss、后四次完整 hit，两项同时驻留，无相互淘汰。每一步比较独立冷重算的
完整 K/V/conv/recurrent 状态、采样隐藏状态/位置与完整 token/EOS/截断；
图组切换、返回和完整解码之后，保留前缀的字节指纹必须不变。哈希忽略 LRU 顺序，
不能把命中导致的排序变化误判为状态污染。两组图像张量必须真正不同，缺少替代 RGB 拒绝。
准入在单独 bank 中回放整条流程，保护正式流的计时和命中计数；不再把单题隔离命中当成两题驻留。
报告 `resident_text_stream` 保存六步及实际容量，打包审计缺项/假命中/不同容量均拒绝。

Phase1/2/3/1+2 原生在线基准同样执行上述驻留控制（单题为三步子集），并验证生产回放
出现完整命中；零完整命中不再报告复用验收通过。两调度槽若实际生成完全相同的题序，
按真实不同前缀数记录，不冒称两个缓存项。诊断中固定问题与动态文字仅用于同题换图控制，
实际 Phase3 回放仍使用每帧真实速度/导航，正式成员、随机题序和评分不变。

训练、精度、标定、采样、预算及完整前向数值对照规则不变。8192 上限可能改变旧 4096
切分点的数值路径，必须用新源码、新 CPU-ready 做 GPU 验证，不能沿用 r9 的回执。
无需重建原始标签库。cr02/cee 均在当前 AutoMoT 根目录及激活环境下执行：

```bash
bash ./sync_automot.sh
export AUTOMOT_TEXT_PREFIX=1 AUTOMOT_TEXT_PREFIX_CACHE=1
export AUTOMOT_TEXT_PREFIX_CACHE_ENTRIES=2
export AUTOMOT_TEXT_PREFIX_CACHE_MIB=1024
export AUTOMOT_TEXT_PREFIX_CACHE_TOKENS=8192
python -B -m qwen3vl_local.sft_new_loop_phase12 cpu
# 保留原先实际使用的数据/历史合同参数；下行替换为新打印的 CPU_READY 目录。
env -u GPU_IDS python -B -m qwen3vl_local.sft_new_loop_phase12 probe \
  --run-dir checkpoints/phase12_shared/NEW_CPU_READY_RUN
```

单卡 probe 仍零 optimizer 更新，依次检查两图/四图和两个问题域。结果写在新 run 的
`probes/run_*/`；失败时 `gpu_prefix_probe.json` 保留容量或跨图检查证据。
新 Phase3 源码清单 r10 单独发布，旧清单/冻结源码/失败目录保持原样。

## 2026-10-31：文本 KV 正式路径的新验收（r9，覆盖下文旧的逐 token 硬门槛）

新建 `AUTOMOT_TEXT_PREFIX=1 AUTOMOT_TEXT_PREFIX_CACHE=1` 运行采用
`text_reuse_same_partition_v2_20261031`。训练仍文字前置、完整前向反传；
不使用脱离梯度的文本缓存训练，不跨 optimizer 更新复用。BF16 加载及既有必要的
FP32 运算保持，未改变提示词、标定、采样、数据成员、训练预算或质量选优规则。

- 正式验证、选优、测试及在线推理使用文本缓存的回答。旧正确损失否决规则保持。
- 完整前向是独立对照：保存完整 token/EOS/截断、首次分歧、logits 及其下一 token 分歧，
  不再以两条不同数值路径的逐 token 相同为硬门槛。非有限值和形状异常仍拒绝。
- 每个正式评测请求额外独立重算文本分段、完整解码，与生产缓存答案严格比较；
  这些诊断不能暖生产缓存，也不进入主速度计时。两域训练前 probe 和首个评测样本另检查
  实际命中、最终 K/V/conv/recurrent 全部张量、采样位置/隐藏状态及保留前缀指纹。
  不存储/不命中不能冒充复用验收通过；同分段状态变化仍阻断，不能用 BF16 容差绕过。
- 完整前向对照也按相同标签重新解析评分，`full_forward_control` 另报有效决策变化、
  无效/截断及双方各自答对的数量；不把完整前向当成真值，不用对照分数选权重。
- 缓存中只有较短问题时，完整重建新问题的规范前缀，不再从短前缀续算；
  避免数值结果依赖历史分段。`promotions` 统计这种重建，当前请求记 miss、复用 0 token。
  4096 token / 256 MiB 默认容量未变，因此部分前缀命中和拒绝存储仍须如实记录。
- 完整前向仍是耗时基线，不偷换成更慢的“每次重建文本前缀”。输出长度可能不同，
  因此同时报告生成 token 数及输出差异，不能把时间比全称为等输出工作量的纯 KV 收益。
- 独立文本流及 Phase1/2/3/1+2 原生在线基准同步采用 v2：同路径重复与冷重算严格检查，
  完整前向差异单列。它们不含真实标签，不认证任务质量或 CARLA 闭环。
- 旧 image-first 路径和同布局关缓存对照保留原严格合同；旧冻结源码/清单/失败回执不改。
  新 CPU ready 绑定缓存验收协议，不能准备时开缓存、训练/评测时关缓存绕过。

远程先更新，在当前 AutoMoT 根、已激活环境下**新建** CPU-ready 运行：

```bash
bash ./sync_automot.sh
AUTOMOT_TEXT_PREFIX=1 AUTOMOT_TEXT_PREFIX_CACHE=1 \
  python -B -m qwen3vl_local.sft_new_loop_phase12 cpu
```

保留此前实际使用的 `--phase1-data`、`--phase2-data` 等资源参数；不指定时仍按原默认资源。
将下面占位路径替换成上述命令打印的 **新 `[CPU_READY]` 路径**，不能填旧失败 run：

```bash
env -u GPU_IDS AUTOMOT_TEXT_PREFIX=1 AUTOMOT_TEXT_PREFIX_CACHE=1 \
  python -B -m qwen3vl_local.sft_new_loop_phase12 probe \
  --run-dir checkpoints/phase12_shared/NEW_CPU_READY_RUN
```

`probe` 自动选择一张空闲卡（或 `GPU_IDS=单个空闲物理卡号`），执行已绑定冻结源码，
同时验证两图/四图的两个问题域；3600 秒超时、零 optimizer 更新，不创建训练 `runs/`。
输出在新运行的 `probes/run_*/`，包括逐域完整回执、worker 日志和 `result.json`。
通过只表示新验收在该基座/初始化 LoRA/样本上通过，不等于新训练质量或速度通过。
随后使用同一个新 CPU-ready 目录执行原 `train` 命令，正式训练仍独立重做两域 probe。
旧 run 仅作证据保留，不迁移、不重写 SHA，也不要求重建原始标签库。

## 问答与缓存

```text
同一个 Qwen3.5-4B 基座 + 同一个新 LoRA
共同 system → 相同 RGB 历史 → 完整图像前缀状态
                              ├─ 独立副本 → P1 user 后缀 → P1 答案
                              └─ 独立副本 → P2 user 后缀 → P2 答案
```

P1每次全问八项：HIGHWAY、STATIC_OBSTACLE、VULNERABLE、TRAFFIC_LIGHT_ABNORMAL、
RS1、RS2、RS4、RS5。八个输出键整体随机排列，取消subset/GROUP/DETAIL/层次提问。
P2沿用原生领域问法：ROAD_CORRIDOR询问UE1/UE3/UE5及INVALID，LOCAL_JUNCTION询问UE6及INVALID。
两路分别作答，P2不读取P1的输出，不用P1预测RS来自动改P2问题域。

为满足共同前缀，两路使用同一system。任务定义放在图像后的user后缀：P1沿用20261029直接八问
及局部VULNERABLE说明；P2沿用20261029切入边界说明。它们都与新训练绑定。

推理先校验共同token前缀和图像预处理一致，再在最后一个vision_end之后分叉。
复制完整Qwen3.5混合缓存，包括K/V、卷积与循环状态，保留attention mask与rope_deltas；
不截断第一路回答后的完整缓存冒充图像状态，不跨权重/LoRA/样本/optimizer更新缓存。
这遵循[Hugging Face的共同前缀缓存机制](https://huggingface.co/docs/transformers/kv_cache#prefill-a-cache-prefix-caching)。

训练每个逻辑样本有两个独立的完整梯度forward/backward，loss各占1/2，之后只更新一次optimizer。
训练不用detach后的推理缓存，以免丢失图像前缀位置的LoRA梯度。缓存加速在推理阶段验证；
不能据此承诺训练也加速，单步两组监督可能比独立P2更慢。

## 训练预算与采样

- 四卡DDP，每卡每次一个逻辑样本，累积1；1024配额、六轮，每模式10752次optimizer更新。
- 每轮UE1/UE3/UE5/UE6各1024、RE2048、INVALID1024，合计7168个逻辑样本、1792次更新。
- 训练INVALID为1/7（14.29%），与现有P2训练相同；原生评测六桶等量，INVALID为1/6。
  “相当于valid数据20%”不是整体训练数据20%，RE加倍后应按上述实际分母计算。
- 保留P2原生route-diverse、highway RE25%、highway UE3 12.5%、INVALID子组、coverage_v2及
  单轮同帧上限10。不为达到万步把单轮配额强升1536/2048；完整六轮CPU投影须先通过。
- P1事实按实际入选物理帧关联。P2的INVALID只说明其当前问题域不适用，不把P1事实改成INVALID/全NO。
- P1八项不是各自YES:NO平衡，因为用户指定按P2取样；回执另列各P1正例曝光，不暗中重采样。

本机初步保护后训练记录4734条，其中VULNERABLE正例31、STATIC_OBSTACLE正例59。
第一轮实际投影分别呈现79次和145次；这种正负失衡可能影响P1，不能只看P2指标判断融合成功。
上一轮新增修正的6个P1训练锚点不在本机P2候选池中，本融合计划没有使用这6条；
用户本轮询问这几帧是否可忽略：不专门强插这6帧，继续保留Phase2原采样。父索引仍先按已审计规则派生，
但不能据此宣称已通过这6帧加强弱项。训练正例稀少仍未解决，新增验证保护不等于补足训练证据。
六轮约10752步不等于新增同等数量独立数据；本机第一轮4360个独特帧，后续没有新增物理帧。
跨epoch仍使用实际成功反传的ExposureLedger，在原配额/语义/子组/标注可用性内优先零曝光帧，
再按累计曝光和路线均衡选择。CPU投影不冒充已消费历史，失败样本不会登记成训练成功。
新增每轮candidate_unique_frames、unseen_before/after、compatible_unseen_after、pool_exhausted回执；
真实训练的下一轮覆盖必须与CPU准入一致。当前融合池首轮已耗尽，不通过改问序或增加轮数虚报新图；
若需要后续epoch持续出现新图，须扩大已标定且隔离合格的候选池，不能挪用val/test。

## 2026-10-30：以 Phase2 标定为主

用户明确要求融合版以Phase2的RS/event为权威。新派生集保留Phase2原有每个槽位、顺序、
true_rs、event/event_codes、问题域、事件答案、目标类别和INVALID来源；不回写两个父索引。
P1分支的RS1/RS2/RS4/RS5直接由Phase2 true_rs生成；R3时这四项均为NO。
HIGHWAY、STATIC_OBSTACLE、VULNERABLE、TRAFFIC_LIGHT_ABNORMAL仍取经过审计的视觉标定，
不从event或场景名猜测，不将RS3自动等同于这四项的完整视觉监督。
同帧P1/P2的RS不一致不再中断，写入 `data/label_authority.json` 记录原值和对齐值；
原生Phase2采样方法、比例、重复上限、六轮10752更新不变。
共享的弱项验证帧也采用对齐后的RS；仅P1有标定的辅助验证帧保留P1原值并注明来源。
固定历史参考题及其原始评分标签保持不变，不把新融合评分口径冒称历史同协议成绩。

cr02报告7317个Phase2独特帧中，仅HazardAtSideLane/
Town12_Rep0_3753_1_route0_01_08_21_52_10第150帧缺P1标定。
本地原生P1索引有此帧，已目视复核147–150历史RGB及151帧控制；
`missing_phase1_review.json` 固定四张输入RGB的SHA及相邻视觉标定。
只有该训练锚点缺失、四张RGB完全一致且不属于任一holdout时，
才在融合内存连接阶段补四项视觉事实，RS/event仍取Phase2；已有标签不覆盖。
不新增或删除Phase2槽位，不更改其INVALID占比，回执 `data/missing_label_recovery.json`
绑定进CPU准入和最终包。本地本来完整的父索引不会触发补标。
相邻帧标定仅供诊断：缺失或视觉标签差异逐项记录并提示，不作为该目标帧的补标依据，
也不覆盖相邻帧已有标签。目标帧的四图SHA先核验；相邻帧涉及val/test时仍执行整组隔离拒绝。
这修复了cr02第149帧触发的过宽门槛；旧失败运行和冻结源码不修改，更新代码后新建运行。
未知缺标、不同图像历史、RGB改变、Phase2同帧自身RS冲突或非法值仍拒绝并打印具体成员。
这项补充解决已审计单帧的监督缺口，不声称已确认远程原始采集为何漏掉该帧。

RGB解析与标定模块统一：支持数据根、场景、路线、rgb目录及单文件链接到外部数据盘，
例如cr02的 `lead_data/...` 实际位于 `/datashare/datasets/lead/...`。
只按逻辑成员 `scenario/route/rgb/file` 取图，拒绝 `..`；历史绝对路径仍映射到当前数据根。
补标、连接、历史参考题、保护集与准入复核保留逐图SHA检查，不以物理路径超出根目录判断缺图。
失效链接报错同时给出请求路径、解析路径及链接信息。源码快照的链接限制不变。

## 两阶段旧测试必须保护

按物理帧连接两份已标定索引；RS冲突按上文Phase2权威规则记录对齐。
未知视觉缺标或历史RGB不一致仍拒绝，绝不从场景名猜P1视觉事实。
原始索引和既有split不修改。在新融合派生集里，以原生physical_group统一保护两边holdout：
test优先于val、val优先于train。另读取已发布的真实历史测试合同，将其中所有物理路线强制保护为test，
包括当前两份父索引均未将其标成test、或者一边索引完全缺失的路线。
本次登记10份已验证历史评测输入、581个物理路线组，核验4974个RGB SHA；只声称覆盖登记来源，
不声称自动知道所有从未收到的旧实验。ConstructionObstacleTwoWays/Town07...001189整组已排除于保护集。
原始数据/旧回执不改；新派生index重新记录隔离结果，交包带历史合同和全物理帧split以便复核。

本机原两份索引6462条P2记录中，1575条原split不同。保护后：train4734、val469、test1259；
696条原P2 train移出融合训练，15条原P2 val进入融合test。P1/P2原数据都保持原样。
这些是新的融合划分，不能声称与独立P2训练成员完全相同，收益不是纯粹“一LoRA与两LoRA”的单因素实验。

原P2验证集高速切入路线属于旧P1 test，保护后融合val没有HIGHWAY_CUTIN正例。
CPU回执显式标记validation_limitation；验证仍有六类任务，但不能认证高速子组。
没有把旧test搬回val、降低训练重复限制或伪造高速标定来填洞。
最终必须评测原P2测试集中的高速案例；若需用该子组选模，须另建独立验证证据，不能偷用旧test。

## 入口：本机准备，远程新训练

在服务器当前AutoMoT根目录激活训练环境；继承已激活Python，不下载模型、不安装环境。
默认本机 `checkpoints/Qwen3.5-4B`、`lead_data`、各阶段标准数据目录。
所有源码冻结在本机项目内，新run不覆盖旧目录，不隐式续训。

历史清单含题目、标签、图像SHA和旧模型已保存的回答，不含RGB/权重，也不上传Git。
启动时优先核验现有合同，缺失则自动读取AutoMoT根/checkpoints/合同目录中的
`phase12_history_20261030.tar.gz`，或从服务器checkpoints已有原生评测输出、历史审计包重建。
源码的 `history_release.json` 仅保存固定合同和十份来源的SHA/名称；合同SHA保持原值。
恢复只读取绑定的JSON，不解压其它成员，不联网、不覆盖错误的现有合同。
历史合同读取绑定文件描述符及路径的身份/大小/mtime/ctime，解析的是同一批已校验字节。
针对硬链接发布及临时链接清理可能引起的ctime漂移，仅当内容SHA仍匹配发布值且变化字段
只有ctime时，额外要求两次完整稳定读取；两次之间的全部元数据也必须一致。
身份、大小、mtime、内容改变或持续ctime漂移仍拒绝，并报告具体变化字段。
这只适用于固定SHA的历史合同，通用proof.sha、训练数据、adapter及源码校验不放宽。
发现目录受120秒/十万目录限制，有10秒扫描心跳，跳过冻结源码和下级目录软链接；
checkpoints自身为共享目录软链接的布局仍支持。所有来源齐备时无需上传；若只在另一台服务器有
部分来源，会逐项报缺，在创建新run前停止，不能退回重抽测试题或悄悄缩小隔离范围。
需要补齐时仍可使用本机 `/home/codon/Downloads/phase12_history_20261030.tar.gz`（约2.4MiB）；
可用 `--history-contract` 指向已存在的相同合同，显式路径缺失/错配直接拒绝。
历史合同仅作为测试与隔离输入，不加载历史LoRA。新运行仍在服务器派生数据→从基座新训→新权重测试→打包。

融合版本一行式完整命令：

```bash
bash -c 'set -Eeuo pipefail; trap "echo [FAILED]-shared-Phase1-Phase2-see-error-above >&2" ERR; bash ./sync_automot.sh; bash qwen3vl_local/sft_new_loop_phase12/run.sh all'
```

独立Phase2约一万步，仍是单独LoRA、单独运行：

```bash
bash -c 'set -Eeuo pipefail; trap "echo [FAILED]-Phase2-10k-see-error-above >&2" ERR; bash ./sync_automot.sh; bash qwen3vl_local/audit_joint/phase12_retrain/run.sh all 2 --revision visual_20261029 --budget-profile phase2_10k'
```

融合入口自动选择4张空闲GPU；明确选卡时在run.sh前加 `GPU_IDS=0,1,2,3`，卡忙时拒绝，
不操作他人进程或GPU计算模式。两个入口可放不同服务器运行，不互相等待。
cr02与cee均使用当前目录解析路径，不复制另一台机器的ready或权重。
自定义资源用CLI `--model-dir`、`--data-root`、`--phase1-data`、`--phase2-data`、`--collection-dir`。

融合 `cpu` 只作数据派生、RGB校验和六轮投影；`train --run-dir <精确CPU-ready目录>` 再启动训练。
独立P2同理用retrain入口的cpu/train，并保持同一revision/profile。
缺失父数据沿原生CPU建库；残缺数据拒绝覆盖。CPU准备有10秒心跳、4小时总超时。
每模式训练含验证最多120小时、测试最多24小时；单轮训练检查24小时上限。失败保留证据，不自动跳过或重跑。

融合GPU训练前还会对两个问题域做实际processor/监督边界/缓存分叉探针。
启动时明确绑定每rank的CUDA设备，先做四卡NCCL求和检查；通信超时为300秒。
模型加载、rank0前缀探针等启动阶段各有900秒硬截止、每rank阶段回执和等待进度。
rank0独占探针、选模验证及最终保存时，其余rank通过torchrun的CPU Store协调，
不再提前挂起NCCL barrier等待。选模验证仍允许7200秒、最终保存1800秒，
这与真正梯度通信的300秒超时分开；不会用延长NCCL超时掩盖卡死。
本进程超时打印线程栈并退出124，由torchrun停止同任务其它rank；不操作其它GPU作业。
异常由torchrun的record处理器保留原始Python栈，失败路径不再先执行可能阻塞的集体销毁。
`diagnostics/<mode>/` 保存阶段JSON（随审计包交付）；`torchrun/<mode>/` 保存各rank日志和error.json，
主 `<mode>_train.log` 仍汇总输出。日志另明确记录第一次两路反传和optimizer更新。
2026-10-30补充日志确认：rank0在训练前缓存探针的logits等价断言失败（最大差8.3125），
其余rank等待barrier两小时后NCCL超时；该次尚无optimizer更新。不能把它描述为已训练两小时。
上述修复解决错误传播和等待问题，不意味着4B缓存数值差异已解决；原等价阈值和完整token检查保留。

单卡只读缓存诊断（在当前AutoMoT根、已激活训练环境执行）：
`python -B -m qwen3vl_local.sft_new_loop_phase12.cache_diagnose --source-run checkpoints/phase12_shared/<failed_run>`。
自动选择一张空闲卡（或GPU_IDS指定一张）；使用失败运行的原冻结源码、真实验证题和基座，
不加载旧LoRA、不做optimizer更新、不写失败目录。先保持原BF16模型，再真实 `.float()` 对照，
对两个问题域记录全部prompt-logits差异统计、完整输出token和EOS差异。比较按16个token分块，
避免同时分配十亿logits的全尺寸差异缓冲。小模型CPU不能代替4B/H20结果。
默认1800秒超时；无论成功或失败均保存独立诊断包（日志、报告、执行回执及补充脚本），
输出 `[DIAGNOSTIC_BUNDLE]`。诊断完成不表示训练完成、缓存验收通过或允许放宽阈值。
2026-10-30收到H20诊断包 `run_20261030_203032_1951629.tar.gz`：107.55秒、0次更新；
两图、两个验证样本、BF16/FP32共8路输出的token与停止状态逐项一致。
BF16最大prompt-logit差5.6875–10.203125，FP32降至0.001069–0.002979；两种精度原逐元素阈值均未通过。
其中路口Phase1在两种精度下均触及160-token上限：一致的是受限生成，不是格式/质量通过。
单次顺序计时未经交替预热，不作为速度收益验收；四图、训练后LoRA和全测试集仍待远程验证。
精度提升显著降低差异，支持数值舍入/计算顺序影响，但这两个样本不能证明所有实现路径均正确。

当前合同v5明确调整验收对象，并非让旧logits断言变成通过：FP32的atol2e-5/rtol2e-4、
BF16的atol0.08/rtol0.03保留作诊断，分16-token块统计全部logits；逐元素不一致不再单独阻断训练。
`full_forward_equivalence_checked`记录是否比较，`full_forward_equivalence_passed`如实记录是否接近；
`cache_acceptance_protocol=exact_generated_tokens_and_stop_v1_20261030`绑定新的输出验收规则。
非有限值、形状不一致、下一token决策变化仍直接拒绝。
探针、每一道选模验证题和最终融合test均比较共享缓存与两次独立完整推理的全部生成token和停止token，
任何token/停止状态差异直接失败。两路径均使用相同160-token上限；触顶会记录，不冒充正常EOS结束。
审计器复核启动探针、全部融合val/test输出与数值诊断，拒绝缺少对照或将不接近标成通过的证据。
这不代表缓存logits数学等价得到认证，不改变BF16训练、两路完整梯度、标签、提示词、采样或10752更新预算。
新合同只用于新运行；旧失败目录、原诊断脚本和历史回执不重写。启动仍先CPU准备，再从基座新训练、评测和打包。
其中最多8个覆盖不同类别/子组的案例先分别预热，再交替执行两轮配对计时，记录原始耗时与汇总节省比例。
计时只含模型prefill、完整decode和缓存复制，使用相同已编码输入，不含processor或额外logits检查；
允许结果为负（共享路径更慢），不预设加速。峰值显存是含基线/诊断的整个评测峰值，不冒称共享路径独占显存。
启动探针失败发生在optimizer更新之前；训练中验证或最终测试失败会保留该次运行并停止，不冒充完整交付。
训练中的cache只用于无梯度验证，不跨更新持久复用。
审计器逐项强制检查预热次数、交替执行顺序、device、dtype、160-token上限、batch=1、
确定性生成、repetition penalty、同一编码输入、同步状态和计时scope，并与本次新权重的评测绑定对照。
缺项、条件冲突、汇总不一致均拒绝，不再接受只有trials的计时证据。

## 2026-10-31 完整输出分歧的诊断与证据保存

本次 cr02 `run_20261031_003849_2055418` 在 ROAD_CORRIDOR 的第一分支出现实际生成 token 差异，不能用前一天另两个样本的诊断通过来覆盖，也不能以放宽 logits 容差、跳过断言或改标签继续训练。本轮保持输出 token/EOS/截断的一致性门和原精度阈值；尚未确认该 4B 失败究竟来自数值敏感还是状态/位置实现。

新训练的 `gpu_prefix_probe.json` 从检查开始就逐域落盘，保存实际 input_layout、文本缓存开关、dtype、题序、样本身份、已通过域与失败域。`CacheParityError` 仍是 ValueError，会阻断训练；附带两条路径的完整输出，以及首次分歧的分支、零起始 token 下标和终止类型，不再只留下截断的异常字符串。失败状态是 `failed_before_optimizer_updates`，不能作通过回执；本轮不更改 optimizer 创建顺序、标定、提示词、采样和更新预算。

单卡诊断入口已支持旧图片前缀和新文本前缀。它先读取原 run 的 ready / training_inputs / execution，恢复绑定布局，不依赖当前 shell 的 AUTOMOT_TEXT_PREFIX。若有新 probe 回执则恢复记录的缓存开关；旧运行没有记录该开关时，明确标为 `default_not_recorded_in_old_run`，文本布局按启用缓存诊断，不能宣称恢复了未知环境。基座默认读取原 training_inputs；`--model-dir` 可显式指定本机资源，仍由冻结加载器校验。

在当前服务器 AutoMoT 根、已激活训练环境下执行。先做无需 GPU 的只读检查：

```bash
python -B -m qwen3vl_local.sft_new_loop_phase12.cache_diagnose \
  --source-run checkpoints/phase12_shared/run_20261031_003849_2055418 --check-only
```

这只确认保存的策略/路径和部分回执关联，不加载模型、不认证完整源码或权重。随后单卡实测：

```bash
python -B -m qwen3vl_local.sft_new_loop_phase12.cache_diagnose \
  --source-run checkpoints/phase12_shared/run_20261031_003849_2055418 \
  --mode 2rgb_endpoints
```

默认自动选择一张空闲卡；若指定 `GPU_IDS`，只可填一个空闲物理卡号，不是训练时的四卡列表。诊断采用原冻结源码和原验证成员，核验原准备文件/RGB，零次 optimizer 更新；不会重新训练、重建数据或改旧目录。两台 GPU 服务器通用，但上述 run 名仅对应 cr02，cee 必须使用它自己实际存在的失败运行，不能跨机器伪造恢复路径。不要在 codon/wang 执行 GPU 部分，也不需要先安装加速扩展来改变复现环境。

每个精度分别按原问题域顺序重放；后一个域恢复前一个域已建立的缓存历史，两题第二分支的追踪也恢复首题状态。缓存和独立路径各重复一次，另报重复输出是否稳定。首次分歧追踪让两边在分歧前消费完全相同的 token 历史，记录经过原生 repetition penalty=1.05 的 top-5 候选、top-2 分差、双方所选 token 的分数、文本与 EOS。若重放不再分歧，会如实记录，不能用一次重放取代原失败。

BF16 保持基座加载时的原旋转缓冲；FP32 是把同一份 BF16 已加载权重提升为 FP32 运算，并禁用 TF32，不是重新获得原始 FP32 权重。只在编码输入 SHA 一致时给出精度对照：`precision_sensitive...` 表示该样本对精度敏感，不是根因或全量缓存认证；FP32 仍不一致则继续查状态/位置路径；重复路径本身不稳定则优先报告不稳定。fast-path 可用性和原模型/cache 源码 SHA 同时记录。

输出仍为独立 `checkpoints/phase12_cache_diagnose/run_*/report.json` 及 `[DIAGNOSTIC_BUNDLE] ...tar.gz`。包内附本次诊断脚本和 `cache_checks.py` 补充证据代码，计算模型与缓存则来自原冻结目录；未改旧源码 SHA。完整成对输出在额外分歧追踪和 logits 诊断之前保存，后续异常仍保留已得到的证据。外部超时以 worker.json / worker.log 为准，不能把 running 状态视为完成。默认 1800 秒上限、160 个生成 token；本工具不是速度 benchmark 或质量验收。

## 2026-10-31 r7：局部精度对照（等待远端验收）

已核对 cr02 返回的 `run_20261031_101418_2289572.tar.gz`（SHA256：
`f5772f5506158de450c3fe518e8c5d76c475264034ae93ac12d206df48d1f45c`）。
ROAD_CORRIDOR 的 Phase1 第 48 个生成 token（下标 47）首次分歧：缓存路径为 YES，独立路径为 NO。
重复运行各自稳定；经过 1.05 重复惩罚后，缓存路径 YES/NO 为 21.875/21.75，独立路径为 21.875/21.875。
FP32 对照两域两题的完整输出相同，支持“该样本对精度敏感”，尚不能确定是哪个运算环节造成分歧。
失败请求为 miss、复用 token 为 0，首次拆分计算就能触发；不能归因于沿用了上一张图片的状态。
该报告未保存惩罚前分数，所以不直接断言是重复惩罚造成，也不将正式训练改为 FP32。

本轮增加两个可选诊断控制，主干保持原 BF16：

| 控制 | 词表投影 | 重复惩罚 |
| --- | --- | --- |
| `bf16` | 原 BF16 | 原 BF16 |
| `fp32_penalty` | 原 BF16 | 对已得到的 logits 提升为 FP32 后运算 |
| `fp32_head` | 用同一份输出权重的临时 FP32 副本，实际执行 FP32 线性投影 | FP32 |
| `fp32` | 整体模型提升为 FP32，仍是同一份 BF16 加载权重 | FP32 |

`fp32_head` 同时包含 FP32 重复惩罚，与 `fp32_penalty` 的差别才是投影运算；它不是将已舍入的 BF16 logits 转成 FP32 冒称恢复精度。禁用该投影的 TF32；不改 tied embedding 的权重/类型，不对参数 `.data` 赋值。上下文退出或异常时恢复方法和后端开关。补充文件 `precision_controls.py` 只由诊断入口使用，记录 SHA 并附入诊断包。默认未传 `--controls` 仍为旧 BF16/全模型 FP32 对照，`fp32` 必须排最后。

在 cr02 当前 AutoMoT 根、已激活原训练环境执行：

```bash
bash ./sync_automot.sh
python -B -m qwen3vl_local.sft_new_loop_phase12.cache_diagnose \
  --source-run checkpoints/phase12_shared/run_20261031_003849_2055418 \
  --controls bf16 fp32_penalty fp32_head --check-only
env -u GPU_IDS python -B -m qwen3vl_local.sft_new_loop_phase12.cache_diagnose \
  --source-run checkpoints/phase12_shared/run_20261031_003849_2055418 \
  --controls bf16 fp32_penalty fp32_head --timeout 3600
```

最后一条自动选一张空闲卡；需要指定时把 `env -u GPU_IDS` 换为 `env GPU_IDS=<单个空闲卡号>`。
cee 同一入口可用，但必须替换为它本机存在并通过验证的 source-run；没有原运行时明确拒绝，不伪造恢复。
不启动训练、不改旧目录，结果另存并打印 `[DIAGNOSTIC_BUNDLE]`；将该 tar.gz 下载回本机供审计。

`control_comparison` 对齐样本/题序/图像/提示词/编码输入 SHA，报告各自输出一致性、重复稳定性及相对 BF16 的完整输出变化。
若某控制在当前样本恢复一致，仅标 `parity_restored_on_this_case`，不是正确答案、缓存全量通过或加速证明。
新追踪同时保存原始与惩罚后 logits 分数/类型。`fixed_history_probe` 让两路径消费 BF16 首次分歧前的固定历史，
在原决策位置重算候选分数，并记录控制的贪心路径是否更早偏离该历史；它是条件诊断，不能代替独立自由生成输出。
到达生成预算末尾才发现的终止/文本差异不额外越界生成。完整输出仍先落盘，附加诊断失败不会抹去已有证据。

生产默认精度、完整 token/EOS/截断门、标定、采样、训练预算及缓存 4096-token/256-MiB 上限保持。
原样本 Phase1 文本前缀 5063 token，不能据命中率称整题已缓存；全模型 FP32 对照有容量拒存，不能据此认证跨图收益。
LOCAL_JUNCTION 的未训练 Phase1 输出包含分析并达到 160-token 上限，缓存/独立相同也不代表输出质量合格。
本轮 CPU 测试验证干预实际生效、异常恢复、两种布局下真实小模型完整生成、诊断缓存隔离与交包；
4B/H20 上局部干预是否消除该分歧、训练能否通过严格探针及闭环是否加速仍待远端结果。

## 选模、评测、交包

每1000步及最终步评测融合val192题，并评测新增Phase1弱项保护集。本机保护集合并去重后254题，
行人和静态障碍物各抽64正例、64负例，每桶来自64个物理路线组；逐题仍回答全部八问。
它只来自原Phase1 val，并按physical_group排除两边父test、已登记真实历史test以及融合train；不改训练采样、原生val或旧test。
每桶至少8帧和2个物理组才标记覆盖足够，该最低门只是工程检查，不是统计学充分性证明。

第一个完成验证的训练checkpoint作为初始候选。以后相对当前已选checkpoint，融合val和弱项保护集
任何此前正确的单问变错都会阻止晋升，并保存全部损失。符合保护约束后按两弱项最低召回、
两任务最低exact、平均exact、弱项平均F1排序选同一个LoRA。严格保护可能导致早期checkpoint一直保留，
不能暗中放宽或分别挑两任务的不同权重。打包时重算每次晋升过程。
这是本轮训练内部验证保护，不是相对历史最优权重的不退化证明；高速验证依然缺失，
质量状态一直保留待审计，原始测试和历史对比仍必须完成。额外保护集及缓存基线会增加验证/测试耗时。
训练完成先核对每rank的10752次更新、两个分支监督次数、完整实际曝光，再保存final。

另起进程从CPU读adapter再搬到指定GPU，重载选中的同一个权重，必须完成：

1. 融合test384题：共同图像前缀分叉，P1/P2各自独立输出。
2. 实际收到的历史P1主包1024题：严格按该模式保存的成员、图序、标签和seed，改为全八问；不从当前父池重抽。
3. 实际收到的历史P2主包384题：严格按该模式历史成员单独问P2；不依赖相同seed来假定跨机成员相同。

两图、四图分别绑定对应历史包，不能用四图清单推测两图。新的固定成员覆盖是100%，这只是数据准备证据，
不是模型正确率。P1旧题中GROUP/RS_HIGHWAY不在新八问输出内，主比较保留共同问题，明确列出未比较问题；
不会将新八问准确率冒充旧混合题型的整题准确率。P2按原问题域比较。
每份reference输出自动产生historical_comparison.json，保留全部旧正确损失，交包时重算。
历史case id由输入文件SHA与原case_index共同绑定，用来消除同一物理帧重复题的歧义；仍强制检查图序/图SHA/标签。
新权重评测另写weight_binding.json，绑定本run ready、完整训练回执、选中adapter SHA及device/dtype；
错配新数据/旧adapter或另一轮训练回执会拒绝，历史LoRA从未成为评测加载对象。

保存完整原始输出、GT、问序、原问键、RGB/SHA、逐问题混淆计数与precision/recall/F1、场景指标、
格式有效率、同题两任务联合正确数、缓存/时延/显存和训练曝光。原参考集只作最终测试，不能选模。
P1原混合题型总分不能直接与新八问总分视为同协议比较。
`compare.py`要求baseline evaluation_inputs的RGB SHA，以唯一物理帧、图序、图像SHA、标签及共同问题
逐项比较，列全部旧正确损失；净分数持平也保留review_required。唯一匹配不依赖case_index相等，
重编号会记录renumbered_matches；同帧重复任务无法唯一确定时标记歧义，不按有利答案挑选。
零匹配、部分匹配、问题覆盖不全都返回insufficient_comparable_evidence和review_required，
同时输出新旧两端覆盖率；CLI退出2。只有完整可比且无旧正确损失才返回通过。
手动CLI比较原始旧包与带historical_case_id的新reference时，会通过baseline-inputs旁的
evaluation_complete.json核验原生结果分片，并要求传入的旧case逐条等于已绑定的原始输出，
再以输入SHA及原case_index补建ID。原case可合并分片或选择子集，部分覆盖仍要求复核；
错来源、改答案、缺回执不能借补ID绕过核验。自动historical_comparison保持同一严格逻辑。

融合每个模式一个完整包，交回两个 `[BUNDLE] .../{2rgb_endpoints,4rgb}_audit_bundle.tar.gz`。
包内含数据/源码绑定、完整val/test/reference输出、实际曝光、选中与final权重的SHA和元数据；权重本体留服务器。
打包前重跑parser/评分并核验题目/RGB/GT；超过30MiB报错，保留完整文件，不截断证据。
已解压包可在其code目录运行 `python -m qwen3vl_local.sft_new_loop_phase12.audit <包目录>` 复算。
独立Phase2仍交回两模式原生包和training_revision_audit.tar.gz。

单独使用融合LoRA时，`infer.py --task phase1|phase2|both` 不需要GT。
提供原始四帧 `--rgb f0.jpg f1.jpg f2.jpg f3.jpg`、与权重一致的 `--mode`、`--model-dir`、`--adapter`；
P2问题域用 `--domain ROAD_CORRIDOR|LOCAL_JUNCTION`。both才共享前缀；只问一路时无需算另一路。
直接调用该Python推理入口时用 `CUDA_VISIBLE_DEVICES=<一张空闲物理卡>` 指定GPU；
`GPU_IDS` 是训练shell/launcher约定，不会由直接Python调用自动转换。

## 当前证据边界

本地CPU可核对数据、容量、严格打包，以及真实小型Qwen3.5混合缓存和LoRA梯度。
本机无完整4B权重，不跑GPU训练/测试。远程4B吞吐、峰值显存、收敛质量、旧正确损失和推理节省比例
都必须等新训练后的完整审计包，不能把CPU通过描述为效果已好或保证加速。


2026-10-31 本轮验证：CPU 完整组合 **670 passed、1 skipped，63.73 秒**；最终诊断/失败回执/原有缓存验收及 Phase3 来源清单专项 **65 passed，6.64 秒**，范围与组合重叠，不相加。包含真实小型 Qwen3.5 在两种布局下的 BF16/FP32、缓存隔离、重复输出及同历史追踪；强制分歧、终止边界与训练失败落盘另用夹具验证。没有运行 4B GPU、新训练或 CARLA，未修改标签库；cr02 原失败仍需上述单卡诊断后定位，不能称本轮已经修复数值根因。

## 2026-10-31 r8：先区分分段数值误差与缓存复用错误

官方依据、上游 Qwen3.5 调查、容忍度和改进条件见
[TEXT_PREFIX_CACHE_20261030.md 的 r8 调研](../qwen35/TEXT_PREFIX_CACHE_20261030.md#r8缓存数值差异实现错误与验收容忍度2026-10-31-调研)。
BF16 的完整与分段计算不保证逐 token 一致；这既不是必然改变答案，也不是可忽略所有差异的理由。
新增 `output_assessment` 用原生严格解析器区分有效字段改变、仅表面差异、无效/截断；
新 `stability_bound` 报告同历史候选间隔与误差上界关系。两者不自动放宽现有训练/评测门。

原 cr02 回包的分歧出现在首次 miss，之前的“重复缓存路径”也会重新清空并预热缓存，
不能单凭重复输出稳定就认证真实跨图 hit。新入口直接分开检查 full / fresh split / hit / 换图及返回：

```bash
bash ./sync_automot.sh
python -B -m qwen3vl_local.sft_new_loop_phase12.cache_diagnose \
  --source-run checkpoints/phase12_shared/run_20261031_003849_2055418 \
  --controls bf16 fp32 --structure-only --check-only
env -u GPU_IDS python -B -m qwen3vl_local.sft_new_loop_phase12.cache_diagnose \
  --source-run checkpoints/phase12_shared/run_20261031_003849_2055418 \
  --controls bf16 fp32 --structure-only --timeout 3600
```

在 cr02 AutoMoT 根、原训练环境执行；单卡自动选择空闲 GPU，不装/换依赖，不训练，只做 prefill，
不再重复生成 160-token 分析文本。只要 BF16 对照可传 `--controls bf16`；生成诊断仍至少两种控制，旧命令保持兼容。
cee 同入口可用，但 source-run 必须是其本机实际存在的 text-first/cache-enabled 运行；旧布局或禁用缓存的回执在占卡前拒绝。
输出仍为新目录及 `[DIAGNOSTIC_BUNDLE]` tar.gz，新增结构脚本随包绑定；旧 run/source/SHA 不改。

`structure` 的四类结果：`structural_contract_violation`（位置/长度/形状/类型/非有限值问题）、
`reuse_not_exercised`（容量等原因没有真实 hit）、`repeat_or_reuse_difference_requires_investigation`（重复或真实复用不同）、
`reuse_matches_fresh_split_on_observed_tensors_not_full_forward_certification`（观察到的复用状态与 fresh split 一致）。
任何一种都不是训练或质量通过凭据。full-vs-split 单独保留逐层采样与全量最终状态误差；这次输出可决定应修状态实现，
还是制定数值/任务质量分开的新验收合同。Phase3 新启动清单 r8 只绑定这些共享诊断依赖，r7 及更早清单不变。
