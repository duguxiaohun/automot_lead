# Phase4 正式包验收与当前连续视图首轮

> 发布更新：本轮已获用户授权按白名单发布，统一 v60_batch_profiles_20261015（523源），发布前209项相关回归通过。同步最终发布源码后执行本文入口；历史段落中的“未push”描述保留当时状态。

本次收到 Downloads/fullflow_report.zip（867630字节，SHA256 `b5d54761918ccbf53a00cb998ef250dd28a727754807feca485b24a8898c9b6f`）及 fullflow_package.json。ZIP CRC、12个成员SHA和501份服务器来源记录与原v57发布清单一致；汇总之间的输入绑定、配额、关键支持、预算均交叉核对。**v57正式CPU建库完成，可以复用；不再运行server_finish/server_phase4_existing建库。** 包内没有GPU训练结果。

## 本次结果及界限

| 项目 | 结果 |
|---|---|
| 完成路线／帧 | 8614／1062401 |
| 自动规则训练题 | 298787 |
| 原视图首轮计划 | 十事件各1024呈现，共10240 |
| 原视图固定val／test | 十事件各37／18题，分别370／180，无放回 |
| R-E5 proceed YES | 760题／182物理路线组，STOP关键支持通过 |
| U-E2 return YES | 2题／2物理组，支持稀少 |
| U-E7 complete YES | 13题／13物理组，支持稀少 |
| U-E5 proceed YES | 0；仅existing_data_development允许此既有缺格 |
| 两模式首轮新增预算 | 6.20GiB，另预留20GiB；入口重查实时余量 |
| 状态 | admitted_existing_data_development；未训练；联合G0及独立准确率未认证 |

本次服务器总耗时20小时30分25秒，其中回放7小时16分43秒，其余13小时13分42秒；没有足够的分段日志判断后半段具体瓶颈。完成回执只能在全流程结束时出现，teacher计数到8614并不等于其余建库／核验已结束。现在收到退出0的完成回执，该次任务已经成功结束。

原始事件题数不相等，训练通过配额实现事件1:1；评测通过固定无放回清单实现事件1:1。YES:NO不要求相等。以上是旧视图的CPU计划与准入证据，实际GPU消费要用首轮四rank记录核对。原生data4的val/test为空是正常的，留出来自独立references4；已核对外部引用关联及全池隔离报告，未把空内部留出误判成缺失。

这份服务器包使用旧学生时间视图。当前要求RGB2 `[t-2,t]`、RGB4 `[t-3,t-2,t-1,t]`，分别推进2／1帧，必须从已完成的原生data4生成当前学生视图，并在实际过滤后重新检查训练、留出配额和关键支持。旧37／18容量不能无条件套用于新版；入口会固定等量选择，缺任一事件就停止。该步骤读取RGB并生成身份sidecar，不回放老师、不复制生产库、不采集新数据。

## 本轮代码审计及修正

新采样审计的measure原先只对现有事件统计1024次，没有显式传十事件集合。原生evaluation_policy.prepare和训练入口已有十事件检查，因此没有证据证明服务器漏事件放行；本轮补上审计统计层expected_events硬检查，逐一删除十事件的反例均拒绝。

existing_data_development的部分测试仍patch旧evaluation_policy，导致4个失败，另有拒绝测试被“缺参考”错误掩盖。现统一patch当前rgb_short模块，并要求隔离、参考、选题错误命中指定原因。相关回归156项通过，含真实小Qwen batch数值、原生训练循环、STOP、消费回执和自动准入；不称全套或目标四卡验收。

原生v57生产合同未改，发布清单内126个生产依赖与旧包一致。旧发布清单漏列VISUAL_REVIEW_RUBRIC_V25.md，不能说旧包单独核验过该文件；当前新清单补入，合计518项。加载器原有完整生产合同校验保留。rule_baseline的SHA是学生视图合同的一部分，所以本补丁后的视图需新建，不能改旧SHA或沿用旧学生checkpoint。

本地以已有17路线v57真实小库重建当前视图并加载，1457训练题，两模式ID／目标相同；单卡／四卡B1计划各1024/event。小库留出为空，仅验证新视图和采样接线，不代替服务器正式池准入。没有新增老师回放或RGB目视，没有新证据要求改标定阈值／提示词。自动规则监督无需人工逐题审核；规则一致率也不能充当独立真实准确率，尤其不能宣称U-E5缺格已解决。

## 服务器操作：一次新视图，然后两模式各一轮

先按既有sync_automot.sh流程同步本轮修改；本轮审计本身未push。不要用服务器遗留.git核对来源，不重跑readiness/P3/旧拟合/STOP盘点。以下在AutoMoT目录、原有可训练环境执行。确认四张卡没有被另一个Phase3任务占用；两种Phase4模式顺序运行。

这是保守B1首轮，不等待性能优化；checkpointing按当前P4默认开启、accumulation=8。尚未实测当前4B四卡容量，不承诺不会OOM；失败保留诊断。若要选更快B值，使用SEQUENCE_BATCH_20261015.md里的固定样本benchmark，不能只按最大可容纳batch选择。

推荐使用新的一体化入口，默认由原训练launcher选择空闲四卡；显式选卡时用 `GPU_IDS=0,1,2,3` 前置同一命令。输出必须是新目录，旧数据与旧训练目录不会被覆盖。

```bash
TRAIN_ROOT="checkpoints/phase4_current_epoch1_$(date +%Y%m%d_%H%M%S_%N)"
python -m qwen3vl_local.audit_joint.phase4_first_epoch \
  --prepared checkpoints/phase4_v57_20261014_232157_762998141 \
  --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
  --batch-size 1 --output "$TRAIN_ROOT"
```

显式pin示例：`GPU_IDS=0,1,2,3 python -m qwen3vl_local.audit_joint.phase4_first_epoch`，其余参数同上；确认这些卡没有被其它任务占用。无需手填DATASET、两个模式的RUN_TAG或不同输出目录，入口统一设置。每阶段在终端打印日志位置，并将状态、退出码和耗时写入workflow.json。

入口先做本轮523来源清单和首轮磁盘预算检查，再受20GiB磁盘保护生成学生视图；随后顺序调用run_automatic，完成每个模式的自动准入、单轮训练、重载及固定test。源码/输入/预算/任一模式失败即停止后续步骤，仍尝试打包已有诊断。SIGINT/SIGTERM会通知本次拥有的进程组，并留时间给内部guard清理其torchrun组；SIGKILL、断电或不可写磁盘不能保证自动交接，恢复后手动调用交接器。

若已经生成**当前合同有效、属于同一个data4**的视图，可加 `--view checkpoints/已有运行/student/view`，只复用视图，仍重新准入；新输出目录必需，不自动恢复或覆盖半成品训练。原始v57旧学生视图不适用。预算和条件准入失败时不跳过门禁，也不拿旧报告替代。

完整和失败任务都尝试生成handoff/phase4_epoch_report.zip及phase4_epoch_package.json。workflow.json与view_build.log尾部已纳入白名单交接，失败原因无需额外猜测。工作流退出0只表示执行及证据收集完成，不宣称模型质量或联合G0通过。一次首轮不是七轮训练。

另一个终端查看时，将下行设为命令最初打印的真实目录（不要重新生成时间戳）：

```bash
TRAIN_ROOT=checkpoints/phase4_current_epoch1_实际时间戳
ls -lh "$TRAIN_ROOT"
tail -n 30 "$TRAIN_ROOT/view_build.log"
tail -n 30 "$TRAIN_ROOT/rgb2.log" "$TRAIN_ROOT/rgb4.log"
```

交回 `$TRAIN_ROOT/handoff/phase4_epoch_report.zip` 和 `phase4_epoch_package.json`。包限制30MB，不含权重、优化器或RGB。若交接失败，另交workflow.json、handoff_error.json及相关日志；若训练中断，交接命令退出2但仍保留缺失态诊断包，不要把“ZIP verified”理解为训练通过。本地审计完整证据位于checkpoints/phase4_v57_received_20261015，轻量回执见phase4_v57_received_verification_20261015.json。

## 一体化入口验证（2026-10-15续审）

新增phase4_first_epoch编排器及失败交接字段，不改老师、标定、提示词、学生视图或训练器合同。运行期间保留了并行修改的batch benchmark／profile代码，补跑整合回归；这些性能修改不冒称本入口独立完成。单独验证视图生成失败／两模式失败／源码缺失／预算不合格／复用视图父库不符均不启动后续阶段；真实CPU子进程+guard的SIGTERM测试确认worker退出，未运行GPU。默认自动选卡，显式GPU_IDS保持；core限制继承，保留既有输出，不自动删除。

本轮并行修改整合后209项相关回归通过（12.29秒），523份发布源码匹配；实际CLI缺资产退出2并生成30050字节诊断ZIP，CRC与全部成员SHA通过。本入口自身不改学生合同；并行run_automatic更新了固定32样本/update接线并改变学生合同SHA，所以在最终代码同步后生成新视图，之后有效视图可复用。详细结果及失败包SHA见phase4_workflow_verification_20261015.json。工具不能认证所有标签的真实准确率；U-E5既有缺格、U-E2/U-E7稀少支持继续报告。无需再扩展本地审计或重新采集，下一步同步后执行以上首轮入口。本轮尚未push。
