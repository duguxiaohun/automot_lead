# Phase4 v79：已收900秒超时包与视图转换修复

## 本次包核验与真实进度

Downloads 的 phase4_epoch_report(1).zip（38303字节）及 package(1).json 已核验：ZIP SHA、7个成员SHA、CRC一致，source_check的558项逐项与本地v78发布清单及源码一致。完整证据留在 checkpoints/phase4_timeout_received_20261017。

本次运行 phase4_b2_epoch1_20261017_121728_998368263 在view_upgrade耗时900.263秒，退出124。此前139没有复现；不据一次运行断言崩溃永久解决。日志明确完成8614条路线缓存复用（新校验0），原生298787道train题完整检查结束，然后两图转换完成、四图尚未完成便超时。两模式训练、重载和test均没有开始，这不是新的训练结果。

原生val/test为0符合独立references4留出设计，不能据此判断缺少正式评测。预算sufficient=true，有约2.79TB可用、声明需求约28.13GB，不是此次磁盘不足；预算仍以声明上限为条件。

## 已修复的重复计算

旧paired_view每次生成两种视图，却只返回一种；迁移循环又调用两次。每次生成四图先重做short_row，而short_row深拷贝整条老师证明后随即丢弃，完整来源digest也反复计算。

- short_row只深拷贝保留字段，仍对完整原始题（含全部证明）求SHA；删改证明仍改变来源身份。
- 两图只转换一次，四图从独立副本产生。两模式字段不共享可变对象，图片SHA、观测窗口与完整输入身份检查保留。
- paired_views一次计算共同冲突/开发曝光排除集合及两种结果。upgrade在隔离单线程事务内消费两种结果，检查传入对象与曝光集合，异常恢复函数；仍逐模式核对排除、成员数量及有序ID SHA，最终重新检查父库/sidecar/来源/曝光。
- 普通学生读取同样受益于每题去重；未新增持久缓存或跳过原生校验，没有增大900秒超时。

未修改teacher、原生库、标定规则、提示词、采样、时间窗口、GPU配置或phase4_load/phase4_production_cache。已有v69路线缓存继续有效。data.py学生合同变化：新增实验会显式迁移已知旧view，旧view不覆盖；已有checkpoint仍须原版源码续训，不能改SHA恢复。本次服务器尚未训练，无checkpoint需迁移。

## 本地验证

真实17路线局部库1457train：直接执行旧v78代码与新代码，逐字段比较两模式所有题、顺序、标签、来源SHA和排除结果，完全一致。旧两次转换0.906+0.882=1.788秒，新一次双模式0.358秒，约4.99倍。完整原生校验及迁移2.116秒通过。该局部库val/test为空，仅验证转换，不能代替服务器正式均衡评测。此测速不是服务器298787题性能保证，也不是训练速度提升。

205项相关回归15.94秒通过；另42项连续采样及分布式评测4.67秒通过，共247项。覆盖证明删改绑定、模式间可变字段独立、冲突/曝光排除、图片SHA/重复ID拒绝、单次转换/输入变化拒绝、异常恢复及真实小库升级。先前45项专项包含在205项内，不重复计数。没有运行远程、GPU训练或本轮新增RGB目视。

## 开训结论与范围

事件之间1:1的生产逻辑不变：每轮十事件各1024次；正式val/test按既定固定等量清单，不要求YES:NO等量。此次包没有实际训练/评测消费证据，不能宣布本轮GPU均衡验收。

没有新证据要求修改标签阈值或提示词。自动规则数据可继续限定范围开发训练，无需逐题人工标注；仍不代表所有标签都正确或联合G0已通过。U-E5/proceed/YES缺失仍是明确不支持能力，不能用其它边YES或事件采样次数补足。

## 同步v79后的下一步（不单独再做readiness或benchmark）

代码发布并经原sync_automot.sh流程同步后，在服务器AutoMoT目录执行下面一条正式工作流。先通过既有sync_automot.sh同步本轮v79发布，不能在旧源码上重跑。源清单自动核验失败即停，不能删除SHA门禁。

```bash
export P4_OUT="checkpoints/phase4_b2_epoch1_$(date +%Y%m%d_%H%M%S_%N)"
(
set -euo pipefail
export PREPARED=checkpoints/phase4_v57_20261014_232157_762998141
export DATASET=checkpoints/phase4_epoch1_20261015_214227_800375064/student/view
export P4_PRODUCTION_CACHE=checkpoints/p4_verified_routes_v69
export DATA_ROOT=lead_data MODEL_DIR=checkpoints/Qwen3.5-4B
export BATCH_SIZE=2 ACCUMULATION=4 UPDATE_SAMPLES=32
export GRADIENT_CHECKPOINTING=1 EPOCHS=1
unset RESUME_RGB2 RESUME_RGB4 P4_VALIDATED_DATA
export OUTPUT_DIR="$P4_OUT"
test -d "$PREPARED/data4"
test -f "$DATASET/student_view.json"
test -d "$P4_PRODUCTION_CACHE"
echo "本次输出：$OUTPUT_DIR"
bash qwen3vl_local/audit_joint/train_phase4_batch.sh
)
```

不重新回放老师、不重建正式库、不重跑Phase3。默认空闲四卡B2，完成迁移后直接两模式各一轮、四卡验证/重载及固定test。完整原生CPU核验仍需要时间，本次只删除已定位重复转换，不承诺总耗时。若再失败不自动重试，保留日志。

成功或失败均交回新输出handoff中的phase4_epoch_report.zip及phase4_epoch_package.json（维持30MB上限）。不要再次交旧121728目录包。日志应新增paired conversion both modes DONE及两个paired view DONE，随后进入准备/训练阶段。

## 本轮授权发布与并行RGB复核

用户已明确授权白名单push。本轮发布前74项入口/视图回归8.83秒通过，与上文247项重叠，不叠加。558项发布清单核验通过，原生teacher及v69路线缓存适配器SHA保持不变。只发布源码、测试、指南及轻量回执，不发布RGB、权重、checkpoints或其它任务文件。

本地实际逐帧查看HardBreakRoute/Town12_Rep0_1902_0_route0_01_09_03_48_13与ParkedObstacleTwoWays/Town13_Rep0_1748_1_route0_01_08_18_54_13各32–39帧，共16张RGB。两路线均从起始帧通过原生route_records因果重放至39帧，共80帧，重新建立STOP历史，未借用旧回放中的停车完成状态；校验240份源SHA，其中目视帧对应48份源文件。两条路线不是新路线，属于开发复核，不是盲审。

16道readiness题与旧记录的答案/布尔事实一致。两条路线均在28–30帧累积停车完成；32–34帧仍有stop_sign_hazard，priority_satisfied=false，输出NO；35帧该标志撤销后，已保留的停车义务和横穿检查仍成立，输出YES；36帧才开始明显加速。实际画面与速度支持先停再动，但仅凭短RGB不能独立证明全部隐藏优先权，本次没有足以修改规则或提示词的新证据。

重放并非逐字节一致：有366处浮点末位差异，最大4.55e-13，引起98个question_id变化。比较明确以1e-10绝对容差核对浮点，其余非ID字段一致；不把重放输出称作旧数据的SHA恢复。该差异不进入生产库、不改变当前既有题集或视图ID；本次迁移仍复用原题并做精确SHA/成员检查，不放宽生产校验。完整本地记录checkpoints/v79_release_rgb_20261017。

另一终端查看进度时，将P4_OUT设为启动时打印的实际目录：

```bash
cat "$P4_OUT/workflow.json"
tail -n 40 "$P4_OUT/view_upgrade.log"
tail -n 40 "$P4_OUT/rgb2.log"
tail -n 40 "$P4_OUT/rgb4.log"
```

尚未开始的阶段可能没有对应日志；以workflow.json的stage/status和已存在日志判断，不把缺rgb4.log当作训练静默运行。保持本次进程使用的源码固定，期间不在同目录继续同步其它版本。
