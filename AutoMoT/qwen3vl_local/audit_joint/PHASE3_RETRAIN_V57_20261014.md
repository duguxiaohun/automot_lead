# Phase3 v57同步后的三轮重新训练

用户要求提供可直接运行的Phase3训练命令。已有两模式三轮已验收，重复训练不是流程验收的必要步骤；如要启动一组新的开发实验，可复用正式索引，从基座重新训练。现有保存不含优化器，不能称为从旧adapter无损续训。仅支持固定1或3轮，不能把此入口直接设为7轮。

本次确认并修复发布接线：server_phase3_epoch原先绑定v54_existing_data清单，当前代码有17个清单成员漂移，重新启动会在加载训练前被正确阻断。改为新的release_sources_v57_phase3_entry_20261014.json，同时更新两个P4入口，501成员逐项matched。没有修改P3标定、学生提示词、训练参数或数据合同；旧清单不覆盖。真实已收首轮ZIP调用verified_completed_epoch仍verified。当前本机23项入口测试通过，另14失败/7错误均因缺torch不能完成，保留日志，不声称完整回归；shell语法通过。

发布同步后，在服务器tmux中运行：

```bash
cd /mnt/workspace/cee3sgh/AutoMoT
conda activate automot
bash ./sync_automot.sh

PREPARED_ROOT=checkpoints/phase3_prepare_20261012_231356_193369568 \
COMPLETED_EPOCH_ROOT=checkpoints/phase3_epoch_recovery_20261013_121248_683529276 \
REUSE_EPOCH_ROOT= \
EPOCHS=3 \
DATA_ROOT=lead_data \
MODEL_DIR=checkpoints/Qwen3.5-4B \
OUTPUT_DIR="checkpoints/phase3_three_epochs_new_$(date +%Y%m%d_%H%M%S_%N)" \
bash qwen3vl_local/audit_joint/server_phase3_epoch.sh
```

保留上述已审准备/首轮目录及其ZIP/回执，入口会检查实际资产及合同；COMPLETED_EPOCH_ROOT是首轮流程准入证据，不作为训练初始化权重。先短两图再四图，默认选择4卡，各从基座新训3轮；每轮每事件1024次，固定val每事件32、test每事件38，逐轮核验实际消费，最终重载核验。只保留新目录的final，原三轮final不覆盖。

完成或失败后交回新输出目录下phase3_epoch_report.zip和phase3_epoch_package.json，不上传权重。已有实测顺序完成两模式三轮约10–12小时，仅供排期，不保证当前共享GPU速度。可以与Phase4 CPU建库并行，但共享CPU/磁盘可能变慢，磁盘守护仍生效；不要同时占用同一组GPU启动另一个训练。

已有训练验收通过不代表生产能力门通过；验证减速与test左变道仍有局限，本次命令不能保证效果提高。Phase4是否建库完成不阻塞这个P3独立开发实验。证据checkpoints/v57_phase3_entry_20261014。
