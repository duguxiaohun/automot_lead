# Phase1 / Phase2：独立新训练、评测和交包

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
