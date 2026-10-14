# Phase4 现有数据正式池验收：开始两模式首轮训练

> **当前指示（2026-10-14 v55）：** STOP证据链已修复并定向验证，以下旧v54“直接开训”说明仅保留为历史。现在先按 [STOP修复与影响盘点](STOP_V55_FIX_20261014.md) 的只读命令交回正式池范围，不执行旧首轮/全库重建，不混用新旧合同。Phase3无需重做。

> **2026-10-14 后续 RGB 审计修正：** 本地另17路线2205帧生成、实际148帧目视，确认 STOP 未履行记录可因关联标志清零被删除并生成已入库 YES，另有 ego 车道字段兼容缺口。撤回此前“Phase4没有必修标定问题”的结论；CPU等量/构建通过不等于标签证据完整。Phase3不重做，Phase4先收敛STOP问题并定位正式池影响，不立即重跑全库。详见[STOP实证与修改位置](LOCAL_RGB_EXPANSION_STOP_FINDING_20261014.md)。

2026-10-14：本次Phase4包 **CPU正式建库与现有数据开发准入通过**，可以直接使用已经生成的库训练，无需再执行 `server_phase4_existing.sh`。Phase3 ZIP SHA仍为已验收的三轮包，复用 [Phase3结论](PHASE3_THREE_EPOCH_ACCEPTANCE_20261014.md)，不重复训练或审计。

## 本次证据

ZIP 866,063字节、12成员，SHA `69d2ff8e475808c3d449a5f62e28d8bec13ec561998062e34e61af982aa3a704`。大小、CRC、全部成员SHA和480份源码逐项核验，与本机及发布 `36449cc6268622ecb10c75eca39fa747662dbbf0` 一致。`admitted_existing_data_development`、`finished_cpu`、退出0、errors为空，**尚未GPU训练**。

处理8614路线、1,062,401帧，当前v54/老师v19回放完整；118个历史歧义问题ID被记录隔离。自动训练池301596条，来源字段原生或追溯后均为 `weak_rule_teacher`。RGB4原库/短视图的原val/test为空，是该入口的正常结构：实际验证与test由独立 `references4` 提供，已在 `rule_consistency` 中校验完整候选池、隔离并选定等量题集。不能绕过此入口改跑默认人工参考模式。

本机核验8625项输入绑定记录的唯一性及相关SHA之间的一致性、原生预算复算、两模式题集/题序摘要、world1/4计划和关键条件支持。原始大库与RGB在服务器，**没有冒称本机重载了全库或逐帧视觉认证**。

| 项目 | 短两图 | 四图 |
| --- | ---: | ---: |
| 原始可用训练题 | 301596 | 301596 |
| 首轮训练计划／每事件 | 1024 | 1024 |
| 首轮总呈现 | 10240 | 10240 |
| 固定val／每事件 | 37 | 37 |
| 固定test／每事件 | 18 | 18 |
| 两模式题集、训练题序 | 一致 | 一致 |
| world1／4训练题序 | 一致 | 一致 |

事件间1:1满足**CPU计划与固定选题**；实际训练是否完成10240次要核验下一包的逐rank日志，当前不提前宣称实际消费通过。U-E5只有820道独立训练题，1024次呈现中有重复，单题最多2次；其它事件本轮各1024道独立题。不要求YES:NO等量。

## 仍存在的问题及处理

1. **能力覆盖不足，不是新的编排失败。** U-E5/proceed/readiness/YES仍0，NO711；只能训练已有有证据的条件，不能声称学会侵入解除后放行。U-E2/return YES仅2条/2组，U-E7/complete YES仅13条/13组。保留 `existing_data_development` 明确限制，不伪造正例、不降标定阈值，不要求重新采集。
2. **部分留出独立场景很少。** U-E5 val37题/3组、test18题/2组；U-E7 val37题/2组、test18题/1组。题数等量不能变成独立泛化认证。后续分事件、分条件和物理组数一并报告，不仅看整体YES/NO一致率。
3. **慢任务缺细分进度，是确定的可观测性不足。** 构建函数记录71522.58秒，即19小时52分03秒；日志回放收集完成于27275.2秒，即7小时34分35秒，另约12小时17分包含回放末尾复验、编译、留出、视图和准入。候选扫描及外层打包不包含在构建函数计时内。没有各阶段持续计时或CPU/I/O采样，不能认定卡死、也不能把全部时间归因于双worker。`REPLAY_WORKERS`只控制回放，后续并不会自动变成2路并行。这次已成功，不为补计时重跑或修改旧库；后续性能工作应先加入阶段计时/心跳，再定位重复验证开销。
4. **两图弱监督计数的显示口径有局限。** `training_admission.weak_supervision_rows=0`只数当前层的 `label_basis`；两图经student转换后来源保存在 `student_reference.original_label_basis`。来源统计及自动训练过滤已追溯为301596条规则题，并非变成人工标注，也未因此丢训练题。当前按原来源字段解释，不为改诊断字段使已建视图合同失效。

未发现新的必须修改标定、提示词或训练生产代码才能开训的问题。此次没有新增RGB目视，也没有依照test错误反向改标。P3三轮开发链路已通过；本次关闭P4正式CPU池准入，下一步验证P4真实GPU首轮、重载与test。跨任务联合G0仍未完成，不影响这里明确限定的独立开发实验。

## 现在服务器执行：复用现有库，各一轮

**使用本次报告里的实际路径 `/mnt/workspace/cee3sgh/AutoMoT`。** 若你通过软链接进入同一目录，保持现有资产的解析目标不变。本轮没有修改生产代码，以下命令使用已发布入口，不需要等待新push，也不用再同步或建库。训练开始前原入口仍会重核输入与预算，属于资产校验，不会重放或重新生成库。

```bash
cd /mnt/workspace/cee3sgh/AutoMoT
conda activate automot
P4_PREPARED=checkpoints/phase4_existing_20261013_184628_526027234
P4_TRAIN_OUT="checkpoints/phase4_first_epoch_$(date +%Y%m%d_%H%M%S_%N)"
export P4_TRAIN_OUT
mkdir "$P4_TRAIN_OUT"

(
  set -euo pipefail
  for P4_RGB in 2 4; do
    AUTOMATIC_COVERAGE_POLICY=existing_data_development \
    DATASET="$P4_PREPARED/view" \
    TEACHER_REFERENCES="$P4_PREPARED/references4" \
    FIRST_EPOCH_BUDGET="$P4_PREPARED/first_epoch_budget.json" \
    DATA_ROOT=lead_data MODEL_DIR=checkpoints/Qwen3.5-4B \
    RGB_MODE="$P4_RGB" RUN_TAG=first_epoch OUTPUT_DIR="$P4_TRAIN_OUT" \
    bash qwen3vl_local/sft_new_loop_phase4/rgb_short/run_automatic.sh \
      2>&1 | tee "$P4_TRAIN_OUT/rgb${P4_RGB}.log"
  done
)
```

默认自动选择4张GPU；若需固定卡号，在对应调用前加 `GPU_IDS=0,1,2,3`。顺序运行两模式，各从基座一轮10240次，原生val370题、保存adapter及优化器、独立进程重载370题、固定test180题。唯一一轮选优对应本次 `epoch_000`，不读取其它任务的latest。任何阶段失败即停止后续模式，日志保留，不自动从头重训；先交失败证据。

本次预算两模式各保留1个检查点，新增上界6.20GiB，加20GiB余量共26.20GiB；回执所在盘约2.54TiB空闲，启动时仍重新检查。没有P4此次实测GPU耗时，不能把P3耗时或CPU建库的20小时当作P4训练ETA。

## 交回哪些文件

在上述同一终端完成或失败后，用新版独立工具收集。只读取本次运行目录，不加载模型、不重训、不建库；权重、RGB、优化器张量不打包。新增工具需要发布同步后才能使用，但**训练入口与正式池合同未变化，现有首轮训练不必等它**。

如果换了终端，先将 `P4_TRAIN_OUT` 设回这次实际输出目录，**不要重新运行训练命令找回变量**。任务停止写入后执行：

```bash
python -m qwen3vl_local.audit_joint.phase4_epoch_handoff \
  --root "$P4_TRAIN_OUT"
```

交回 `handoff/phase4_epoch_report.zip` 和 `handoff/phase4_epoch_package.json`。旧handoff拒绝覆盖；需要补交时指定新的 `--output "$P4_TRAIN_OUT/handoff_retry_1"`。

- 正常收集退出0仅表示证据已收集，不代表训练或模型质量通过。
- 缺文件、损坏JSON、截断JSONL、输入变化或步记录不一致时退出2，仍交回诊断包。可读JSONL行保留，但该来源不算完整。
- 按四rank检查每条步记录的ID、位置、顺序、轮次、有限损失与采样SHA；事件数量继承已绑定采样计划，不能替代重新读取正式池核验事件归属。
- 元数据大列表以数量和SHA表示，逐步及逐题记录不使用此投影。超出30MB预算的成员明确登记遗漏并使包变为不完整，不能静默声称完整。
- 始终保留U-E5缺能力的准入说明。包的CRC/SHA通过与训练通过分开；收到后还需复算验证/test、重载一致性与两模式配对。

本工具替换此前内联收集脚本：旧脚本遇到损坏JSON会在打包前退出。无需重跑任何训练来修复交接。代码与测试说明见[首轮交接补强记录](PHASE4_EPOCH_HANDOFF_20261014.md)。

## 原CPU验收轮的本机验证范围（历史记录）

480源码/12成员SHA、内部证据关联、原生预算重算通过。执行相关回归84项通过，2项因本机pvi缺peft/tokenizers依赖失败；短RGB测试也因缺tokenizers无法收集，旧/tmp完整环境已不存在。失败日志保留，没有把依赖缺失写成生产代码故障或宣称完整回归通过。生产源码未改，未重跑既有2478项全套。反馈收集命令通过1294字节合成不完整包的SHA/CRC验证，missing=42按预期；不是Qwen GPU训练验证。

证据位于 `checkpoints/phase4_existing_received_20261014/`；[轻量核验回执](phase4_existing_acceptance_verification_20261014.json)。该CPU验收轮仅更新文档/回执。后续收集工具的代码修改及65项回归见上方首轮交接补强记录；两轮均未远程训练、未push。
