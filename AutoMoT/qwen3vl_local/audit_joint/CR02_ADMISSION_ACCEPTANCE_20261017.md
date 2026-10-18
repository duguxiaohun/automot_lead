# cr02 完整准入包验收与首轮入口（2026-10-17）

> **后续纠正：本页是历史包验收记录，不是当前执行指令。用户要求先做本地标定/RGB审计，以下训练建议暂停；最新缺陷修复与质量边界见 [本地标定闭环报告](LOCAL_LABEL_CLOSURE_20261017.md)。**

## 结论

本次补包通过；cr02 的标定建库和自动规则数据开发准入可以关闭，不再重跑。允许现有数据限定范围的两模式各一轮开发训练，不代表独立标签准确率、U-E5放行能力或联合G0通过。没有发现必须修改生产标定、提示词或训练代码才能继续的新缺陷，本轮保留这些源码不变。

机器是 cr02（包内hostname=szh-gpu-cr02），资产为 `checkpoints/phase4_labels_20261017_125230_304771113`。不能使用cee的路径、缓存、权重或实际训练结论。

## 独立核验

- ZIP 1476055字节、13成员SHA及CRC通过，ZIP SHA256 `da1aa9f5f0596fa83591d9fcf896a920d09554753f00cbd60049c6393e39a39f`。
- 完整14969873字节automatic_admission.json已收到，SHA与上轮漏包诊断完全相同：`5d76e18cc8586f89ede5a58d5402666bcd9e6a023b567912882cc153c5eacce8`。补包修复已在服务器验证，外层状态也恢复admitted_existing_data_development。
- 原构建558项源码逐项匹配v78发布清单；collector_sha256匹配当前v83打包器。构建来源与收集器来源分开核验，不能把历史source_check说成整台机器565项当前源码已核验。
- 8626项输入身份记录，以及data4/view/references4摘要和两模式准入报告互相一致。只有身份记录到达，本地没有服务器原始全池，未逐项重新打开这些原始文件。
- 原生Phase4生产源码与上一轮比较的203项保持一致；这次发现的14处历史来源差异属于学生转换/外围运行及其测试等，不需重放老师。v78视图由既有显式升级入口处理，不能手工改合同SHA。
- 116项入口、准入、已有数据开发策略、视图升级、epoch交接回归通过，14.96秒；非新增GPU验收。

## 事件等量与自动监督

| 项目 | RGB2 | RGB4 |
|---|---:|---:|
| 自动训练池 | 298770 | 298770 |
| 首轮计划 / 每事件 | 1024 | 1024 |
| 固定val / 每事件 | 37 | 37 |
| 固定test / 每事件 | 18 | 18 |
| val YES / NO | 66 / 304 | 69 / 301 |
| test YES / NO | 16 / 164 | 16 / 164 |

已逐事件核对十事件集合、总数、unique_questions和采样action计数相加；两模式训练成员SHA相同。训练原始池不要求等量，采样后的事件曝光才要求1:1。YES:NO不要求1:1。

本包计划batch/card=1，实际推荐入口为B2/四卡/累积4，每次更新32样本。run_automatic将BATCH_SIZE传入rule_baseline，GPU前按实际B2重算，训练器再绑定并核对消费。不能声称当前包已经验证B2实际训练，也不能把B1题序跨world或跨RGB强行视为相同：连续锚点分组可改变题序。固定val/test按各模式stride选择，成员SHA不同，但都十事件等量、无放回。

所有298770训练题来自weak_rule_teacher，全部固定val/test来自heldout_rule_reference，选优指标是ten_event_rule_consistency。training_report在选小子集前对完整合格train/val/test检查physical_group跨split，收到的报告trainable=true、errors=[]；这是Phase4内部隔离检查，不代替Phase3/Action完整联合隔离。人工逐题标注不是当前开训前置。

## 必须保留的质量限制

以下为最终过滤后训练/readiness正例，不能用原始回放数量代替：

| 关键条件 | YES题数 | 物理组 |
|---|---:|---:|
| U-E2 return | 2 | 2 |
| U-E3 proceed | 277 | 95 |
| U-E5 proceed | 0 | 0 |
| U-E6 proceed | 708 | 97 |
| U-E7 complete | 13 | 13 |
| R-E5 proceed | 759 | 182 |

U-E5已有其它边的YES，但不能监督proceed放行。U-E2返回、U-E7完成支持仍少，不能以重复次数代替独立场景。多数事件固定评测只有1个物理组，连续邻帧相关性高；等量通过不等于泛化充分，准确率不能直接称真实道路能力。

本包没有新RGB或逐帧几何，不能从SHA/CPU准入证明每一个规则标签都正确。本轮无新增RGB目视，没有足以修改阈值或提示词的新证据。既有STOP/横穿修复继续使用，不为补正例放宽条件。

## cr02最短执行流程

用户已明确确认：cr02只有本次题库，尚未训练，也没有生产路线缓存。因此本机按首次训练执行，下面创建新缓存；不引用cee的权重或缓存。同步已发布v83后，在cr02的AutoMoT目录、原Python环境执行；不运行server_finish build、readiness或benchmark。

下面路线缓存路径是**本次允许创建的新本地派生产物**，并非宣称已存在：原生读取适配器会逐路线完整核验后写入，后续复用；不生成新老师标签或复制整份回放。若已知cr02有对应有效缓存，可将此变量改成那个路径，不能套用cee缓存。首次冷缓存和旧view升级仍需CPU时间；升级现有900秒上限不变，失败交包，不自动循环重跑。

```bash
export P4_OUT="checkpoints/phase4_cr02_epoch1_$(date +%Y%m%d_%H%M%S_%N)"
(
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
export AUTOMOT_SERVER=cr02
export PREPARED=checkpoints/phase4_labels_20261017_125230_304771113
export DATASET="$PREPARED/view"
export P4_PRODUCTION_CACHE="$PREPARED/verified_routes_v83"
export DATA_ROOT=lead_data MODEL_DIR=checkpoints/Qwen3.5-4B
export BATCH_SIZE=2 ACCUMULATION=4 UPDATE_SAMPLES=32
export EVAL_BATCH_SIZE=2 REUSE_FROZEN_RGB=1
export GRADIENT_CHECKPOINTING=1 EPOCHS=1
unset RESUME_RGB2 RESUME_RGB4 P4_VALIDATED_DATA
export OUTPUT_DIR="$P4_OUT"
test -f "$PREPARED/data4/manifest.json"
test -f "$PREPARED/references4/references.json"
test -f "$DATASET/student_view.json"
test -f "$MODEL_DIR/config.json"
test -d "$DATA_ROOT"
echo "本次输出：$OUTPUT_DIR"
bash qwen3vl_local/audit_joint/train_phase4_batch.sh
)
```

入口自动核对v83发布源码、view/父库绑定和剩余空间；必要时升级已知v78学生view到新输出目录，保留原view和原题库。随后CPU准备、RGB2训练/验证/重载/test，再RGB4同流程；自动选择空闲四卡。未指定GPU_IDS时使用自动选卡，若旧shell保留了GPU_IDS则按既有显式pin规则执行。

收到时空闲30.30GiB，首轮声明需求约26.20GiB（含20GiB预留），不是cee的TB级余量。该预算以声明上限为条件，入口重查当前空间、保留运行时磁盘守护；新view、路线缓存和学生快照也会占空间，不承诺完全零新增磁盘或服务器固定耗时。

成功或失败均交当前P4_OUT/handoff下：

- phase4_epoch_report.zip
- phase4_epoch_package.json

若未生成handoff，直接交当前workflow.json及末段日志，不重启相同输出、不删除已有权重/快照/缓存。完整首轮后依据实际事件消费、重载一致性及固定test结果决定后续轮数，不盲目续7轮。Phase3和cee已通过的训练不因此重跑。
