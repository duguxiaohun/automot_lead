# Phase4 本次失败交接审计与补证

本次 Downloads 的 phase4_epoch_report.zip 是失败诊断包，不是训练结果。34888字节、4成员，ZIP SHA256 `2dd2a5a7b377af6658ff32ea2d3ea5680b373e01ee8a3f80fb542c1a0fadf492`，成员SHA及CRC全部通过。源码记录558项逐项匹配v76清单；磁盘预算fits_declared_bounds，所需含预留28133425152字节，记录可用2785600864256字节，不是该预算门拦截。

唯一执行阶段为view_upgrade，210.2658秒后退出139。RGB2/RGB4训练均未启动，未到规则准入、训练消费、验证、重载或test。47项缺失是尚未生成的后续证据，不是47个代码缺陷。旧RGB2已验收结果仍保留，不能作为此次新版执行的证据。

## 已确认问题与修复

交接器只采集view_build.log，漏掉新增升级入口的view_upgrade.log、view_upgrade/load.log及status.json；服务器包因此仅保留退出码，无法定位崩溃栈。现增加固定文件白名单的限长日志尾（每份最多60000字节）、原文件SHA与读后漂移检查，以及存在的status/result元数据；失败阶段应有日志缺失时明确登记。无权重、RGB、core或任意递归目录收集，30MB上限继续生效。

同时将缺训练证据时的NoneType报错改为明确的missing sample, sampling or rank evidence，不改变失败状态、不误报训练完成。两项新增反例修复前均失败；交接/首轮编排共50项回归通过。v77_upgrade_handoff清单558源matched、无额外Python；生产/学生合同、标定、提示词、采样、训练、读取及路线缓存适配器字节未变，不需要为本次交接修复迁移有效view或重建缓存。本轮未push、未运行远程/GPU。

139按当前进程包装器通常对应SIGSEGV（128+11），但仅凭退出码不能识别具体故障库/函数，也不能排除程序主动返回139。它不是默认900秒升级超时的124；没有证据将其归因CUDA OOM、NCCL、数据损坏或磁盘不足。升级worker已有faulthandler，所需日志目前仍在服务器，先取已有日志，不重复迁移。

## 服务器现在只补交日志

无需同步或重跑，AutoMoT根目录执行以下只读命令，把输出交回：

```bash
P4_RUN=checkpoints/phase4_b2_epoch1_20261017_113332_626467217
tail -n 160 "$P4_RUN/view_upgrade.log"
tail -n 160 "$P4_RUN/view_upgrade/load.log"
cat "$P4_RUN/view_upgrade/status.json"
```

若尾部没有崩溃栈，可另执行 `rg -n -C 25 'Fatal Python|Segmentation|Traceback|Error|Killed' "$P4_RUN/view_upgrade/load.log"`。文件缺失的报错同样保留。不要交core或整个数据目录。

下一步根据实际栈定位并修复升级故障，再复用现有v57原生库、v69路线缓存及旧view走新输出。每卡B2/累积4/四卡32样本更新决定不变，不再做batch选型或GPU smoke，不重放老师、不重做Phase3。此次包无法核验新版实际train/val/test十事件1:1，也不能提供逐事件模型质量或标签正确率；仍使用自动规则监督，不要求逐题人工标注、不强求YES:NO等量，U-E5放行缺支持按既有开发限制报告。没有新增依据改标签阈值或提示词。
