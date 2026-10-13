# v54白名单发布：Phase3三轮与本地RGB续审

用户授权发布到 origin/main。发布范围为源码、测试、版本清单及轻量审计文档；完整题库、回放、RGB、权重、checkpoints和其它未跟踪资料留本机。沿用现有 sync_automot.sh 覆盖同步，旧实验保留。

## 服务器只执行Phase3固定三轮

进入服务器既有 AutoMoT 目录，建议在持久终端会话中执行。以下路径对应已验收的准备目录及完整首轮目录；不重复readiness、建库或首轮恢复。

```bash
cd /home/cruser1/lda/AutoMoT
conda activate automot
bash ./sync_automot.sh

MODEL_DIR=checkpoints/Qwen3.5-4B \
DATA_ROOT=lead_data \
PREPARED_ROOT=checkpoints/phase3_prepare_20261012_231356_193369568 \
COMPLETED_EPOCH_ROOT=checkpoints/phase3_epoch_recovery_20261013_121248_683529276 \
REUSE_EPOCH_ROOT= \
EPOCHS=3 \
OUTPUT_DIR="checkpoints/phase3_three_epochs_$(date +%Y%m%d_%H%M%S_%N)" \
bash qwen3vl_local/audit_joint/server_phase3_epoch.sh
```

模型不在默认位置时只替换MODEL_DIR。默认自动选择四张GPU，两模式顺序执行；需要指定卡时在训练命令前增加 `GPU_IDS=0,1,2,3`。脚本内部记录 `epoch.log`，保留20GiB磁盘守护并重算本次预算。入口核验当前478源清单 `release_sources_v54_admission_20261013.json` 及既有首轮ZIP/准备合同。

两模式分别从基座训练三轮，不是旧优化器续训。每轮十事件各1024次，三轮每模式30720次；固定val各事件32、test各事件38，最终adapter重载验证。只保留本次最终adapter，不复制RGB或保存优化器。粗估10–12小时，非时间保证。

成功或失败均交回脚本最后打印的新输出目录中的两个文件：

- `phase3_epoch_report.zip`（不超过30MB）
- `phase3_epoch_package.json`（大小、SHA及核验信息）

不要交权重。若启动前的路径/环境错误导致未生成包，交终端报错和已生成的 `epoch.log`；不要删旧目录、改SHA或重复建库来绕过错误。预期成功状态为 `three_epoch_verified`，只代表开发链路通过，效果仍看本次结果。

## 本次发布前独立检查

- 79项专项回归通过，4.70秒：自动准入缺项诊断、Phase3三轮编排和v54近裁剪/离场时序。
- 478源SHA全部匹配；真实已收Phase3首轮包的原生三轮准入仍verified；两个shell入口语法通过。
- 新增准入诊断会在缺留出、预算缺失/损坏时保留两模式的训练源分布与关键readiness支持，保持blocked。人工标签、其它边YES或catchup不能填自动放行支持格。

## 本地同步完成的标定审计

本轮以当前v54/老师v19重新生成两条已曝光U-E7训练路线，共107帧；原生共享gzip→RGB4→短视图，两图/四图各110train且ID/标签配对，val/test为空，14.46秒。321份原始文件SHA前后核验，实际逐帧查看14个不同RGB帧、42份对应来源。是既有开发窗口复核，不是盲审或历史新增独特帧。

| 路线窗口 | 规则与图像核对 |
| --- | --- |
| Town12 002113，f42–48 | 驶离路口，对向交通仍可见；f43开始event_resolved=True，但稳定行驶条件未成立，complete仍NO；直到f48两条件同时True才YES。 |
| Town15 002214，f46–52 | 雾雨中驶离路口；f49–51稳定行驶已成立而event_resolved=False，仍NO；f52两条件同时成立才YES。f47–48无该complete题，保留为空，不补标签。 |

本轮未发现足以修改标签、阈值或提示词的新反例。雾天远端可观察性、有限输入与完整历史之间的差异仍是证据边界；当前小库不满足正式十事件等量留出准入。

Phase4暂不运行旧全池/旧fit或训练入口。仍需U-E5侵入后真实可观察的proceed YES、当前v54正式池及跨任务完整索引隔离，再进入两模式首轮与重载。这里的U-E7正例不解决U-E5缺口，Phase3可独立推进。

轻量回执：`v54_publish_rgb_verification_20261013.json`；完整本机证据：`checkpoints/v54_publish_rgb_20261013/`。
