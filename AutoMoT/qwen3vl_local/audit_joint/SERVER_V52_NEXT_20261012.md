# v52：服务器现在执行什么

已通过的CPU readiness、模型预检和旧拟合不重跑。同步仍用你现有的 `sync_automot.sh`（源仓库 `duguxiaohun/automot_lead`，clone+rsync覆盖白名单）；服务器 AutoMoT 内遗留 Bosch `.git` 不作为版本核验依据。保留其它文件和旧实验。

## 本次只执行一个CPU数据准备入口

同步最新发布后，在服务器 **AutoMoT/** 下、已激活 automot 环境运行：

```bash
bash qwen3vl_local/audit_joint/server_v52_pilot.sh
```

默认原数据 `lead_data`；默认输出 `checkpoints/v52_pilot_<时间戳>/`。可显式指定一个不存在的新目录：

```bash
DATA_ROOT=lead_data OUTPUT_DIR=checkpoints/v52_pilot_first \
  bash qwen3vl_local/audit_joint/server_v52_pilot.sh
```

入口依次：

1. 核对新版发布源码SHA（不运行pytest、模型preflight或旧`.git`检查）。缺文件/不一致时立即停止，交 `source_check.json`，不要删本地文件绕过。
2. 扫描正式数据根，冻结全部符合原生准入的候选路线及train/val/test分配。这是新正式池准备，不是重复检查旧checkpoint目录。
3. 在同一候选池内，按split与scenario轮转，选择最多30条、每条最多400帧的完整路线，用单worker原生gzip回放。不是从每条路线截前400帧；超过400帧的路线不进这次试建。场景分层不保证十事件齐全。
4. 不编译/封存部分共享库，不训练；输出逐路线压缩字节数、原生条件分布、候选总量及运行耗时，自动打包并复核ZIP成员与外层SHA。

运行期间保留至少20GiB，并在本次盘净新增约2GiB时停下（一秒轮询，不是硬配额；其它任务写盘也会触发）。禁用core。若守护停止或回放失败，正常可用磁盘下仍尝试打包错误与日志尾部；不会把失败报成训练就绪。机器被强制关机等情况下无法保证留下包，保留输出目录与日志即可。

**本次跑完就停，把终端打印的两个文件交回：**

- `checkpoints/v52_pilot_<时间戳>/pilot_report.zip`
- 同目录 `pilot_package.json`

ZIP限制为30,000,000字节；不收权重、优化器、RGB、完整回放或原始候选逐帧清单。`pilot_completed`只代表试建成功，`training_ready=false`、`sufficient_for_full_build=null`仍正确；包核验通过不是正式池通过。

## 收到这两个文件之后

根据真实压缩量、路线长度分布和条件分布确认后续分批容量安排；短路线样本不能直接外推全源保证。保留这份 `candidates.json` 和 `production/`，后续可在相同源码/候选合同下累计追加未登记路线，无需重跑已完成路线。**所有批次完成前不要编译共享库**，因为发布会封存准确的回放集合。

正式池完成后执行[自动首轮步骤](BATCH_BASELINE_V52_20261012.md#4-正式池完成后的执行步骤)：单份RGB4库 → 短两图/四图视图 → 自动规则留出 → 实际过滤后十事件等量和关键YES/NO覆盖 → 完整首轮预算 → 两模式各一轮、保存重载、固定test。YES:NO无需1:1，不续旧adapter，也不直接跑7轮。

Phase3正式索引/adapter、Action缺失和联合隔离仍单列跟踪；本入口不声称修复这些资产或通过联合G0。本地逐帧标定审计与此CPU试建可并行，不会改写服务器正在使用的旧实验合同。
