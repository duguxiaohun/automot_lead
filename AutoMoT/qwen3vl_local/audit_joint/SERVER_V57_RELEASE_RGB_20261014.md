# v57发布：服务器只跑新版正式CPU池

2026-10-14发布接线更新：当前三入口统一核验`release_sources_v57_phase3_entry_20261014.json`。用户如需新的Phase3三轮实验，按[重新训练命令](PHASE3_RETRAIN_V57_20261014.md)；已验收旧实验不要求重跑，Phase4命令不变。

本次按用户白名单发布STOP原生位置绑定、同题横穿检查、失败交接容错及配套测试/文档。当前入口清单是`release_sources_v57_handoff_20261014.json`，501个源码/规则资产逐项核验。服务器仍使用`sync_automot.sh`的clone+覆盖同步，源头为`duguxiaohun/automot_lead`；其旧本地`.git`不是本次分发依据。

## 服务器步骤

建议在tmux会话中运行（例如先`tmux new -s phase4_v57`），避免SSH断开中断长任务。在服务器执行：

```bash
cd /mnt/workspace/cee3sgh/AutoMoT
conda activate automot
bash ./sync_automot.sh

export P4_V57_OUT="checkpoints/phase4_v57_$(date +%Y%m%d_%H%M%S_%N)"
printf '本次输出目录：%s\n' "$P4_V57_OUT"
DATA_ROOT=lead_data \
MODEL_DIR=checkpoints/Qwen3.5-4B \
REPLAY_WORKERS=2 \
OUTPUT_DIR="$P4_V57_OUT" \
bash qwen3vl_local/audit_joint/server_phase4_existing.sh
```

该入口自动执行源码SHA核验→现有原始数据候选扫描→v57回放→单份共享RGB4题库→自动规则留出→短两图视图→预算和准入→交接打包。没有新采集、GPU训练或权重下载。源码不匹配会提前停止并交诊断；不要跳过核验。

仅需一次当前版本正式池。不要将旧v54–56回放拼入v57，也不要删除现有数据来腾空间。已有完整且有效的v57正式池则直接交其结果，不因本次交接器改动重复建库。旧全源CPU流程曾约20小时，新版未有全源耗时测量，建议为这次任务保留一天时间；这不是训练耗时预测。

脚本自动关闭core dump并守护磁盘，当前预留至少20GiB且限制约12GiB净写盘增长；触发保护后交回诊断，不擅自降低保护或清理模型。其它并发写盘也会影响可用空间。

另一个终端可查看（将目录替换为上面打印的实际值）：

```bash
tail -n 40 checkpoints/phase4_v57_实际时间戳/build.log
```

## 给本地反馈

无论成功、准入阻塞或构建失败，交回该输出目录中的两个文件：

- `handoff/fullflow_report.zip`
- `handoff/fullflow_package.json`

包受30MB上限约束，不含模型权重或RGB。若连这两个文件都没有，交脚本最后约40行输出及本次目录路径，不先重跑整个构建。不要只发终端“verified”：外层verified仅说明ZIP完整，仍要核验内部`finish_receipt.json`、`automatic_admission.json`及预算。

CPU正式准入需验证：两模式题集/题序配对、关键监督支持、物理路线隔离，train/val/test十事件之间1:1。训练计划每事件1024次，val/test采用无放回的固定等量清单；本次新版池每事件实际留出数量以报告为准，不预填旧库37/18。YES:NO和同语义子场景不强求等量。U-E5放行YES缺格仅按existing_data_development保留能力限制，其它关键格仍硬检查。

正式包通过后才运行两模式各一轮、保存重载和固定test；这一步待实际新输出目录核验后给命令。Phase3三轮已验收，不重复；readiness、旧fit和STOP盘点不重跑。独立Phase4开发准入不等于联合G0全部通过。

## 同步进行的本地RGB复核

发布前186项相关回归通过，1项模型依赖meta_profile未选，不宣称全套或GPU验收。两份shell语法及501份发布源码核验通过。

实际查看HardBreakRoute Town12 `1902_0`与ParkedObstacleTwoWays Town13 `1748_1`各第40–46帧，共14不同RGB；42份目视源SHA核验。相对紧接上一轮两个Accident窗口无路线重叠，但均是既有train开发路线，不称盲审或全历史未见。读取已有有效v57原生回放，没有重新生成标签。

两窗口中自车继续经过路口，侧路车辆逐渐靠近画面。两路线第43帧stable为YES，HardBreak第46帧/Parked第44帧re_yield为YES；重读各题7帧原始几何原生facts与回放一致，54份复算源SHA核验（与目视源有重叠，不相加为96）。本轮STOP历史证据来自已验收回放，非独立重新标定；后续再让行不直接证明之前放行错。第45–46帧Parked无题，不补标签。没有新增足以改规则、阈值或提示词的确定证据，不据此延期正式池验收。

本机详细证据：`checkpoints/v57_release_rgb_20261014/`；轻量回执`v57_release_rgb_verification_20261014.json`。RGB、完整源证据、checkpoint和用户`.vscode`均不入本次发布。
