# Phase3 新版正式 CPU 准入（2026-10-23）

本轮沿用已经审计的 single v3 与 abc v2 标定、划分政策、事件覆盖和三轮预算，执行正式全池 CPU 准入。CPU 结果不代表四卡训练或评测可靠性验收。

## 正式入口暴露的地图视图重叠

第一次执行统一入口 `phase3_compare.sh cpu single --geometry ...`，真实父库读取完成后报 `ValueError: output_overlaps_input`。原因是 single CPU 把地图视图创建到 `<study-root>/maps`，而 prepare 的输入保护拒绝输出根包含地图输入。此前 CPU 测试替换了地图解析器并返回根外路径，未覆盖这一组合。

single v3 的视图现改为 `<study-root>_maps`，与实验输出根同级；默认和自定义 `--study-root` 使用同一规则。地图解析和内容校验、prepare 的重叠保护均保留。旧视图、失败日志和状态快照保留。abc v2 接收统一入口位于 `checkpoints/phase3_compare/maps` 的地图路径，不存在这个嵌套问题，运行源码未修改。

新增回归使用真实地图解析器串接原生小池 CPU 建库、划分、准入与拒绝报告，分别覆盖默认地图发现和显式 `--maps`。LaneMap 使用合成几何夹具，未把模拟地图文件称为真实地图解析验收。

## 证据位置

- 证据目录：`checkpoints/phase3_cpu_admission_20261023/`。
- 首次失败日志：`checkpoints/phase3_single_cpu_admission_20261023.log`；失败状态快照：证据目录中的 `single_attempt1_status.json`。
- 修复后日志：证据目录中的 `single_attempt2.log`。
- 源码快照：证据目录中的 `runtime_sources.json`，绑定 single 162 个、abc 160 个运行文件。
- 直接相关 CPU 回归：`regression.xml`，26 项通过（8.49 秒）；不与历史测试数相加。
- 正式就绪结论以各包 `ready_1280.json`、`cpu_status.json` 和其绑定的准入、三轮计划为准。运行中状态不能当作通过。

保留 `independent_holdout_claim=false`：曝光保护只覆盖已登记清单，不能认证其他机器未知的使用历史。少查询、单物理组与缺类诊断继续保留。未改变区域边界或九类内部 softmax loss，未启动 GPU、在线评测或 push。

## single v3 正式结果

修复后正式 CPU 入口退出 0，`cpu_status.json` 为 `completed`，`ready_1280.json` 为 `ready`，完整运行约 28.4 分钟。父库原始来源完整校验、原生六动作重算、派生库重载验证均已执行。

260,719 个父锚点过滤后保留 204,910 个：train 202,516、val 573、test 1,821；三个 split 均覆盖十事件。划分搜索访问 4 个状态，得到以下完整物理组移动，原 val/test 组归属不变：

| 物理组 | 原归属 → 新归属 | 该组事件 |
| --- | --- | --- |
| `HardBreakRoute/Town12_4357_0` | train → test | LEAD_BRAKE |
| `EnterActorFlowV2/Town12_975_0` | train → test | POST_BYPASS_RETURN、RAMP_MERGE_EXIT |
| `MergerIntoSlowTrafficV2/Town13_1398_14` | train → val | POST_BYPASS_RETURN、RAMP_MERGE_EXIT |

三轮各 12,800 次呈现／3,200 次更新，共 9,600 次更新；投影中的未见物理帧数为 12,800／10,306／9,506。同帧单轮最多 1 次、全程最多 3 次。计划投影未推进实际训练曝光。

跟车制动 test 为 58 个锚点、219 个有效查询、1 个物理组，移出后 train 保留 3,106 个锚点。信号灯故障 test 为 1 个有效查询、1 个物理组。动态切入 test 为 3 个锚点、12 个查询、1 个物理组，目标包含 STOP 和 RESUME（目标质量合计 3／9），并非全部 STOP；其三个锚点的完整目标与上轮 `replay_final.json` 逐字段相同，见 `cutin_crosscheck.json`。

single 验证集无纯 LANE_CHANGE_LEFT 查询。test 的 LEAD_BRAKE、DYNAMIC_CUTIN、ONCOMING_INVASION、SIGNAL_FAILURE、RAMP_MERGE_EXIT 仍触发稀疏支持诊断。正式准入通过不是独立泛化或六类评估充分的认证。

数据 SHA256：`bcf1d2fd2f1fc381f20ce7a13b278824e547055e108cbea56846a018e126b031`。

## abc v2 正式结果

统一 `cpu abc` 入口退出 0，`cpu_status.json` 为 `completed`，`ready_1280.json` 为 `ready`，完整运行约 15.1 分钟。a/b/c 共享的全池完成原始来源校验、几何派生、缓存重载验证、准入和三轮计划投影；本轮未修改 abc 运行源码。

保留 259,493 个锚点：train 256,469、val 740、test 2,284；三个 split 均覆盖十事件。每个表示三轮各 12,800 次呈现／3,200 次更新，共 9,600 次更新。未见物理帧为 12,800／11,272／9,689；同帧单轮最多 1 次、全程最多 3 次。

验证集 a/b 的 LEFT_NEAR 无支持，c 的 LEFT_NEAR、RIGHT_NEAR 无支持。abc test 跟车制动仍仅 2 个锚点、1 个物理组；本轮没有套用 single 新划分补齐它。覆盖通过不表示稀疏类别评估可靠，也不证明候选 4／13 米边界更合理。

数据 SHA256：`db2c6e39fa891c4ebdde0018dec489394119edc7b15a3ffbfde7e38348411fc7`。

## 当前使用与范围

本机两个正式 ready 已生成，无需重复 CPU 建库。可从应用根只读查看：

```bash
PYTHON=/tmp/geometry_v1_verify_env/bin/python bash qwen3vl_local/audit_joint/phase3_compare.sh status all
```

该 Python 路径是本机测试环境，不能照搬到远程。未执行本轮 base 正式全池验收、GPU 四卡训练、RGB 边界盲审或远程验证，因此不声称五类正式训练流程全部跑通。single 与 abc 各自使用已绑定的划分，不声称是同成员配对测试。

汇总回执：`phase3_cpu_admission_verification_20261023.json`。证据目录保存两个完成状态和 ready 快照、日志、测试 XML、动态切入交叉核对，以及运行前后均一致的源码清单。此前回执保留为历史记录；其中 single CPU 和对应测试的旧 SHA 因本轮修复发生变化，未补写旧 SHA。
