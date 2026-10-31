# Phase1 / Phase2：独立新训练、评测和交包

**2026-10-31 Phase2 任务组合 `taskmix_20261031`：** UE1/UE5 继承旧版，UE3/UE6/RE/INVALID
继承证据版，默认文字前移并复用文本 KV；不新增标定规则。用户已选择 UE6 保留新版。
入口 `bash qwen3vl_local/audit_joint/phase12_retrain/taskmix.sh all`，默认六轮 10752 更新。
来源矩阵、输入协议、远程命令与限制见 [TASKMIX_20261031.md](TASKMIX_20261031.md)。

**2026-10-30 Phase2 证据修订 `--revision evidence_20261030`：** 对新训练 81 个错例逐帧 RGB 与
metas 回放后，高速切入改用客观标定 v2（训练正例 28→573 帧、Town12 右侧切入入池），
训练池按客观证据排除无刹车的 UE1 正例、无可见违规车的 UE6 正例、前车明显刹车的 RE 负例
和路线起始填充帧（不翻转标签、holdout 不变），提示词升 v7。命令、预期测试分数影响和
争议帧口径见 [EVIDENCE_20261030.md](EVIDENCE_20261030.md)；下文其它命令不带该参数时仍是旧修订。

**2026-10-30 历史路线隔离修复：** 新启动的独立 Phase1 在标定派生后，按
`physical_group` 剔除与已登记 Phase1 历史测试重叠的 train/val 记录，不移入 test；
保留行和原 test 逐字节不变。原始数据、旧 ready、旧训练及冻结源码不改。
本机发现的是 `VehicleOpensDoorTwoWays/Town12_3315_0` 的53条训练记录，
不同采集批次仍属于同一物理路线；不是宣称相同RGB泄漏。新准入重新检查容量，
保留40320步预算、原采样算法和prompt；删去候选后不能要求采样成员SHA与旧版相同。
独立 Phase2 不依赖这项 Phase1 数据准备，仍按自己的入口训练。

Phase1 启动先检查本地历史合同；缺失时自动从服务器已有的合同压缩包、原生评测输出或
历史审计包恢复，无需事先手动解压。完整来源已在该服务器时无需额外上传。
`checkpoints` 本身可为共享目录软链接；不遍历源码快照或下级目录软链接。
可以用 `--history-contract` 显式指定已存在的相同合同（显式路径缺失不会自动搜索）。
合同SHA固定于源码，来源不齐逐项列出，错配直接拒绝，均在占用新run前停止；
不会联网获取历史、用当前池重抽旧测试题或加载旧LoRA。如果两台服务器各自只保留一部分历史，
仍需补齐缺少的元数据或使用完整合同包，不能据此声称无条件免传文件。
若清单硬链接发布后只发生ctime漂移，新读取器核验固定内容SHA后再要求两次稳定读取；
其它元数据变化、内容变化或无法稳定均报错，训练数据/权重的通用散列检查不变。
`train` 接续只使用本run已保存并绑定的合同，不依赖外部原文件。
旧 Phase1 ready 缺少隔离证据时不能由新入口继续训练，应新建CPU准入。
`training_revision_audit.tar.gz` 增带历史合同、剔除明细及隔离回执；标定派生回执保留在
新 `training_revision.json` / `preservation.json` 的 `label_derivation` 字段，避免把旧的
“全部holdout不变”描述套在可能剔除了验证记录的新数据上。下文旧版本统计作为历史记录保留。

**2026-10-29 新增可选配置：** 独立Phase2使用 `--budget-profile phase2_10k`，
保持配额1024及原采样比例，六轮共10752更新。默认bounded仍5376，旧运行不迁移。
用户另要求的单LoRA双任务实验位于独立目录 [sft_new_loop_phase12](../../sft_new_loop_phase12/README.md)，
不把本目录两个独立阶段的权重串联。该新目录包含共同图像前缀分叉、测试保护和完整交包说明。

**2026-10-29 新修订：** 当前用户要求的 Phase1 四万步、直接八问和局部视觉复核，
以及 Phase2 保持预算的 UE3 提示词修订，使用显式 `--revision visual_20261029`。
新命令、验证范围和未解决限制见 [VISUAL_20261029.md](VISUAL_20261029.md)。
下文无 revision 参数的命令仍是 2026-10-27 版本，不能用来启动此次新方案。

用户于 2026-10-27 明确：修改标定或提示词后，必须用该版本重新训练，再评测新权重。
旧 adapter 换提示词只能作为辅助诊断，不代表新训练后的效果。

## 本版本实际改变什么

- Phase1：5 个逐帧确认的 **train** 帧修正（1 个 STATIC_OBSTACLE、4 个 R1→R5），
  原生八问中 RS2/RS4 定义分别加入 v2 的同向通道与可识别灯头说明。训练、验证、测试
  用同一个新提示词身份；两个说明合并用于本次新训练，尚无新权重精度结果。
- 不把上轮 test 标定候选写入本次数据：val/test 字节保持，原评分口径保持。
  不按场景或路线扩散5条修正，不强行让它们入选训练；标签改变可能影响原生分箱成员，
  采样算法、比例和重复上限保持。5条修正不等于全训练池已重新逐帧审计。
- Phase2：保持 `direct_event_visual_v5_highway_ue3` 提示词；针对独立逐帧审核的23个train
  高速切入锚点修订 R4/R-E4→R3/R-E1，UE3答案不变。另5个已正确锚点原样保护。
  修复该跨域INVALID子组候选，详见下文。两阶段从Qwen3.5基座独立新训练，无权重传递。
- 两图固定原 history 的首尾 `[0,3]`，四图 `[0,1,2,3]`，均为 YES/NO。
- 每模式四卡、累积1、三轮：Phase1 配额768 / 9216更新；Phase2 配额1024 / 5376更新。
  每阶段两个模式先后训练；两台服务器上的两个阶段完全独立。不能把两模式合计更新误作单模型步数。

## 一行式入口

在各服务器自己的 `AutoMoT/` 根目录激活训练环境，先同步已发布的白名单源码。
默认使用本机 `checkpoints/Qwen3.5-4B`、`lead_data` 和本阶段标准数据目录；
不假定另一台服务器的路径或 run 存在。

服务器一（Phase1）：

```bash
bash qwen3vl_local/audit_joint/phase12_retrain/run.sh all 1
```

服务器二（Phase2）：

```bash
bash qwen3vl_local/audit_joint/phase12_retrain/run.sh all 2
```

自动选4张空闲卡；明确选卡时前置 `GPU_IDS=0,1,2,3`。入口拒绝单卡套用四卡预算。
继承已激活 Python 环境，可显式前置 `PYTHON=/path/to/python`；不安装环境或下载模型。
本次命令固定训练/评测/预算参数，会清除旧 shell 中 RUN_TRAIN=0、PLAN_ONLY、旧 ADAPTER_DIR
等原生脚本参数，防止冒充重训。资源路径使用 CLI `--model-dir`、`--data-root`、
`--parent-data`、`--collection-dir`，不要用旧脚本环境变量重定向新版本。

`all` 每次创建独立 run，先模型资源检查，再 CPU 数据准备/真实RGB检查/三轮采样投影，
全部通过后才开始正式 GPU 训练。已有父数据复用；缺失时原生 CPU 建库，残缺目录拒绝覆盖。
每个 CPU 子阶段默认7200秒超时、10秒心跳；有错立即输出并非零退出，不隐式继续训练。
仅“暂时没有报错”不等于保证质量或保证不卡住，请看心跳和阶段状态。

本机可先仅 CPU 验证：`bash .../run.sh cpu 1 --run-dir <新的本机目录>`。
同一台服务器、同一份准入可 `bash .../run.sh train 1 --run-dir <该CPU目录>` 接续；
逐文件核验原数据/准入/源码，缺目录或已尝试过GPU训练均拒绝，不复用另一台机器的回执。
不能从本机 CPU ready 推断远程已准入。不同机器缺少部分审核帧会记录 absent_reviews；
至少匹配一条审核记录，全部缺失则拒绝。匹配后允许本机标签为审核前值或已确认的修正目标；
已是目标的行逐字节保留，计入 already_correct_train_reviews，不能冒称新改了一条。
只有仍需修改的行计入 applied_train_reviews；即使全部已修正，仍用新提示词从基座新训练。
所有匹配行（包括已正确行）都检查 train 归属、RS/answers一致性、四帧顺序和RGB SHA。
未审核字段变化、第三种RS标签或RGB不符仍拒绝；一次收集已匹配帧冲突，保存
`train_review_conflicts.json` 中的本机实际值/审核原值/修正目标，不自动跳过或覆盖。
原审核JSON与失败目录不改，selected_train_reviews.json另存本机实际前态和对齐记录。

## 隔离与公平比较

运行创建独立内容寻址源码版本，只在新副本中改变 Phase1 提示词渲染和训练输出根。
原源码、旧版本、旧数据、旧 adapter、旧 latest 不改；新训练输出到
`checkpoints/sft_new_loop_phase{1,2}_retrain_runs/`。源码目录链接仍拒绝；
`Automot/checkpoints` 资产链接及本机RGB/权重目录按现有规则共享。

每次训练从同机 Qwen3.5 基座初始化独立 LoRA；两种模式不传递 adapter。
训练后沿原生 val 选模、重载新 adapter，做 test/base/LoRA/审计提示词评测并打包。
训练消费步数、来源与 adapter SHA 仍由原生完成校验检查。
`[DONE]` 仅指新训练、评测、打包完成，精度是否更好还要对审计包比较。

下载日志中全部 `[BUNDLE]` 文件：每模式的原生审计包（可能另有 balanced 包）
和小型 `training_revision_audit.tar.gz`。后者含5条训练修正、本机实际应用情况、
holdout散列、CPU准入和新源码版本绑定。权重保留服务器，不塞进审计包。

先核验新旧测试成员/问题/标签/RGB SHA，再报告整题、逐类 accuracy/precision/recall/F1、
格式、时延/显存和原正确损失。基座相同的“数据+提示词联合改动”不等于单因素实验。
已曝光的测试集只能作开发回归，最终泛化需独立未开发测试；本入口不偷偷重划测试集。

## Phase2 远程容量拒绝取证

若 CPU 报 `Coverage quota exceeds compatible frame capacity`，建库可能已经完成，
不能重复建库或擅自提高重复上限、改变1024配额。不同机器的兼容子组容量需要分别审计。
在 AutoMoT 根目录运行（替换具体失败目录及新的输出文件名）：

```bash
python -B -m qwen3vl_local.audit_joint.phase12_retrain.collect_capacity --run-dir checkpoints/phase12_retrain/<failed_p2_run> --output checkpoints/phase2_capacity_audit.tar.gz
```

仅接受 Phase2 CPU准入的容量失败；按失败计划里实际的 `--index` 收集原索引及manifest、
失败目录JSON/log、绑定快照manifest与四个相关采样源码，并只读计算当前有效action hints。
不建库、不读取RGB、不加载模型、不写训练准入、不改变采样规则，不覆盖同名包。
Action hints是收集时重建结果；原失败进程未保存其回执，不能冒称已证明历史来源未变。
将此小包带回本机后重放原生三轮规划、核对稀缺子组与物理帧共享；未完成前不承诺1024可开训，
也不把本机较少总量但子组容量足够的库当作远程同一份数据。

## cr02 的23帧RS修订与1024容量修复

收到的真实索引有7968行，train6666/val336/test966。两路线共28个已审核高速切入正例，
原23帧误为R4/R-E4，5帧已是R3/R-E1；用户核验34张历史RGB的聚合SHA与本机完全一致。
`phase2_highway_review.json` 固定23修订与5保护锚点及逐图SHA；执行时再次逐图校验，
拒绝第三种标签、非train归属、答案/历史顺序变化及缺帧。不会因scenario名字自动改标签。

新Phase2数据派生只读现有父索引，不扫描原始全池，不写回旧库或原collection。
23个正例仅修订true_rs/event/event_codes。受影响INVALID子组保留父库的111个位置和数量，
从这28个已审核train正例通过原生 `_make_row` 与 `_cycle_sample` 重新生成跨域负例，
固定新修订seed；其余7834行（包括5个正确正例、全部val/test）逐字节保留。
这明确是新的局部数据派生，不冒称完整原生建库的逐条重放或原始采样成员完全相同。
训练采样代码、四UE比例、RE2倍、INVALID1倍、子组平衡、路线规则、1024配额、上限10不变。
已正确且INVALID候选完整的本机6462行可逐字节保持；只有正例已改而负例池尚旧时补派生负例。
父库旧sampling统计移入parent_statistics，不能冒称新成员统计。

使用收到的cr02真实索引，本机两图/四图原生三轮计划均通过：每轮7168呈现/1792更新，
共5376更新，warmup1000，三轮单帧最多9。独特帧5856/5750/5752，新增5856/280/0，
仅为CPU计划，不代表GPU完成、真实耗时或精度提高。新权重仍须重新训练、评测、交包审计。
`all 2` 自动使用run/data中的修订索引，并将修订回执与保护证明绑定至ready和审计包。
旧运行快照和失败目录保留；Phase1已运行任务不重启。

## Exclusive_Process 下 LoRA 评测加载与原运行恢复（2026-10-28）

cr02 两图完成训练后在 LoRA production 加载 safetensors 时，rank1/2/3 报 CUDA busy/unavailable；
用户确认0–3为 Exclusive_Process、失败退出后空闲，4–7由其他任务占用。评测初始化已设置
local_rank，但PEFT未显式指定反序列化设备时推断为通用cuda；不能依赖底层库把它当作当前rank。
新Phase1/2 eval显式传 `torch_device="cpu"`，原 `model.to(device)` 仍负责最终设备放置。
不改模型参数、提示词、采样、步数、GPU计算模式或他人进程，也不修改qwen35后端身份合同。

已有旧权重必须在原冻结源码中验证，不能直接拿更新后的源码替换旧源码/SHA。恢复入口：

```bash
GPU_IDS=0,1,2,3 python -B -m qwen3vl_local.audit_joint.phase12_retrain.recover_eval --run-dir checkpoints/phase12_retrain/p2_20261028_112100_432848
```

适用本机Phase2 fresh_train_eval失败，且两图原生5376更新/完整训练回执/数据及adapter身份通过。
支持的原源码按 `recovery_profile.json` 精确绑定；不兼容版本拒绝，不改旧回执或SHA。
入口锁住原launch锁，在原run下新建recovery目录，补充代码复制后留证，在原快照中运行。
仅torchrun的进程入口增加单独记录的adapter CPU读取包装；训练、评测与打包仍使用原生源码。
本补充属于显式运行时修改，不能将native包中的旧源码清单误称为完整执行来源，必须连同
`recovery_audit.tar.gz` 中的补充代码、绑定、加载trace和native包SHA一起审计。

两图：仅重跑完整评测/打包，旧失败评测目录和已有基座结果保留在原处，另写新目录。
四图：原计划输出目录缺失才从基座新训练；若已完整训练并通过原生验收，复用其权重补评测；
存在不完整训练目录则拒绝，不隐式续训、不覆盖、不重复两图训练。仍为四卡、1024配额、
每模型三轮5376更新，两个模式各自独立基座初始化。任何子步骤失败就停止，保留所有产物。
原run/status.json保留历史failed，恢复状态以新recovery/completion.json为准。
交付所有 `[BUNDLE]`，包含两模式原生包及额外恢复包。不得再用 `all 2` 当作此运行的续跑。

本机测试包括真实小Qwen3.5+PEFT权重保存重载、强制库默认cuda的失败对照与CPU修复、
真实两进程torchrun/Gloo入口测试，以及显式GPU阶段替身的恢复顺序/保护测试。
这些不是远程H20/GPU验收，不承诺新模型精度已经达标。

本轮相关CPU回归221项通过（64.00秒）；最终修正恢复输出目录后，13项针对性测试
再次通过（3.78秒，包含在221项内），覆盖原生独占输出目录检查。首次运行有两项
shell测试因本机PATH缺少Python环境失败，补齐测试环境PATH后通过，未更改检查条件。

## 只读源码打包修复与已完成两图评测复用（2026-10-28晚）

`recovery_20261028_205911_691534` 已通过CPU adapter加载并完成两图四组推理，
在打包 `copytree` 时失败。冻结源码是0444：文本遍历先复制了code中的JSON/TXT，
随后源码树复制再次写同一只读目标，导致Permission denied。两个阶段的eval.sh均让
code只由一次copytree复制；不chmod冻结源码/旧证据、不删旧结果、不放松SHA/完整性检查。

旧快照保持原样。恢复入口增加有记录的Python入口补充，只匹配精确绑定eval.sh中的
inline packer，将同一条code排除策略应用于内存中的打包程序。其它Python调用原样转交；
补充代码、原始/执行片段SHA随recovery审计包交付。四图原生流水线也使用这一补充。

本次失败可用以下命令复用两图评测：

```bash
GPU_IDS=0,1,2,3 python -B -m qwen3vl_local.audit_joint.phase12_retrain.recover_eval --run-dir checkpoints/phase12_retrain/p2_20261028_112100_432848 --reuse-two-eval checkpoints/phase12_retrain/p2_20261028_112100_432848/recovery_20261028_205911_691534/pipeline/2rgb_endpoints/2rgb_endpoints/eval_review
```

复用只接受本轮原恢复程序的精确源码、失败回执和同run/adapter/data/source绑定，
四组推理均经原生raw输出重算与完成/性能回执校验；错误审计、原CPU加载trace及原源码证据
同时检查。只将已完成输出复制到新恢复目录，旧失败包不带入；重新执行原生delivery校验、
打包、包内逐文件验证和30MB限制。原结果及补充来源散列随新包保存，末尾再检查未变化。
缺项、篡改、错配直接拒绝，不自动重推理。两图打包后继续原四图流程：缺失才从基座新训练，
完整则只评测，不完整训练目录仍拒绝。各阶段采样、标定、提示词与更新预算均未改变。

本轮240项CPU回归107.21秒通过；最终加强子进程打包测试后24项针对性复核15.55秒通过
（包含在240项范围内）。非root环境真实复现旧逻辑Permission denied，验证两阶段只读
源码打包及缩图重试、包内散列、原生评分、完成回执、复用后的真实shell/Python子进程打包、
篡改拒绝及恢复顺序；VLM输出/小训练fixture明确为替身，不称真实模型精度或H20本轮验收。
