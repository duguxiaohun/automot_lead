# 单个 LoRA 的 Phase1 + Phase2 独立问答实验

用户于2026-10-29明确要求新增融合实验。现有独立 Phase1/Phase2、历史权重与运行保持原状。
当前新运行合同为 `phase12_shared_lora_v2_guard_20261029`；旧v1冻结源码/ready不覆盖或迁移。
每个图像模式训练一个共用 LoRA：两图模型同时回答P1/P2，四图模型也同时回答P1/P2；
两图与四图仍是两个独立实验，并非把两种图数混成一个训练集。

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

## 两阶段旧测试必须保护

按物理帧连接两份已标定索引；缺标签、RS矛盾、历史RGB不一致会拒绝，绝不从场景名猜P1答案。
原始索引和既有split不修改。在新融合派生集里，以原生physical_group统一保护两边holdout：
test优先于val、val优先于train，任一旧测试路线不进入融合训练或选模验证。

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
FP32 logits使用atol2e-5/rtol2e-4，BF16使用atol0.08/rtol0.03，同时要求下一token决策一致。
探针进一步比较两路生成的全部token和停止token；最终融合test的每题均做共享缓存与两次独立完整推理对照，
任何token/停止状态差异直接失败。两路径均使用相同160-token上限；触顶会记录，不冒充正常EOS结束。
其中最多8个覆盖不同类别/子组的案例先分别预热，再交替执行两轮配对计时，记录原始耗时与汇总节省比例。
计时只含模型prefill、完整decode和缓存复制，使用相同已编码输入，不含processor或额外logits检查；
允许结果为负（共享路径更慢），不预设加速。峰值显存是含基线/诊断的整个评测峰值，不冒称共享路径独占显存。
失败发生在optimizer更新之前。训练中的cache只用于无梯度验证，不跨更新持久复用。

## 选模、评测、交包

每1000步及最终步评测融合val192题，并评测新增Phase1弱项保护集。本机保护集合并去重后254题，
行人和静态障碍物各抽64正例、64负例，每桶来自64个物理路线组；逐题仍回答全部八问。
它只来自原Phase1 val，并按physical_group排除两边任何旧test以及融合train；不改训练采样、原生val或旧test。
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
2. 原P1 test的原生1024题成员：改为全八问，单独问P1，检验P2取样下是否损害P1覆盖。
3. 原P2 test的原生384题成员：单独问P2；默认种子20260810、原生非route-diverse评测抽样。

保存完整原始输出、GT、问序、原问键、RGB/SHA、逐问题混淆计数与precision/recall/F1、场景指标、
格式有效率、同题两任务联合正确数、缓存/时延/显存和训练曝光。原参考集只作最终测试，不能选模。
P1原混合题型总分不能直接与新八问总分视为同协议比较。
`compare.py`要求baseline evaluation_inputs的RGB SHA，以唯一物理帧、图序、图像SHA、标签及共同问题
逐项比较，列全部旧正确损失；净分数持平也保留review_required。唯一匹配不依赖case_index相等，
重编号会记录renumbered_matches；同帧重复任务无法唯一确定时标记歧义，不按有利答案挑选。
零匹配、部分匹配、问题覆盖不全都返回insufficient_comparable_evidence和review_required，
同时输出新旧两端覆盖率；CLI退出2。只有完整可比且无旧正确损失才返回通过。

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
