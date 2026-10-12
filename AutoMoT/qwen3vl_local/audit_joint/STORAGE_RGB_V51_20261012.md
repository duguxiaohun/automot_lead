# v51：共享回放、原生压缩与连续 RGB 复核

服务器 CPU 检查已关闭；本轮没有重复 readiness、远程盘点或旧拟合。改动集中在新产物存储，标签算法、阈值、提示词和事件等额策略保持原语义。旧产物未删除、未原地升级；当前源码合同为 v51，不能修改旧 manifest 的 hash 来复用旧 checkpoint。

## 存储实现

- `dataset.build` 默认 `production_storage="references"`。只保存小型 `production/index.json` 快照及 `production_reference.json`，两图/四图共享外部路线回放，不复制路线文件。
- 引用登记绝对逻辑路径、解析路径、SHA；逐条路线同时绑定 index receipt。加载核对描述文件、快照、当前 index 和所有路线。缺失、字节漂移、同字节软链接改指均拒绝；切换 cwd 不改变解释。短视图、pipeline_ready 和 training_scope 纳入新描述文件；统计结束也复查外部引用。
- Python API 保留显式 `production_storage="copy"`，用于独立归档或对照；不能把引用库当成可脱离源回放的独立包。移动源目录需要重新建立并核验引用，不自动猜路径。
- `teacher_replay` CLI 与 full_pipeline 新建回放默认 gzip6。逐行压缩到独占 pending，正常关闭后原子更名，没有未压缩临时副本；固定 gzip 时间戳和文件名，压缩结果可重复。原生 accounting、compile、teacher_review、automatic heldout 均支持 `.jsonl` 和 `.jsonl.gz`。
- Python `produce(..., compression="none")` 保留原 API 默认；正式 full_pipeline 显式传 gzip。恢复时检查编码及既有 index 合同，禁止在旧目录切换编码覆盖 index。磁盘守护仍需启用；压缩不代替剩余空间保护。
- 同预约计划 v31 / 空严格 registry v26；teacher 算法版本仍 v17，源码身份更新，无新增人工批准。

## 原生实测范围

5 条既有训练路线，755 帧完整重放、1116 条原始问题；逐条比较 v50 的状态、事实、来源、答案等字段，仅排除源码派生的 question_id / teacher_sha256。Phase3 同帧动作/证据逐条比较。没有扩大正式全池或宣称全源标定完成。

本轮编译的是原生RGB2 `[-4,0]` / RGB4 `[-6,-4,-2,0]`，没有重建短两图 `[-2,0]` 视图，四图跨度不变。两模式各 464 条规则训练题，语义完全配对；原生 build/load 成功，引用版与复制版加载结果一致。仅覆盖 6 事件、没有 val/test，`trainable=false`，不能把该小库用于正式训练或等量评测验收。

最终完整回归 **2292项通过（118.23秒）**；追加gzip自动留出导出/重载及审核抽样集成测试4项通过，其中2项是原完整回归之后新增的压缩分支。最终原生重放/两库建载及复制对照 **64.84秒、峰值237348 KiB**。回放14,251,070→499,958字节，RGB4引用库6,558,759字节（共享路线回放另计）。

最终运行数据、耗时和测试结果见同目录 `storage_rgb_v51_verification_20261012.json`。首轮结果保留在 `checkpoints/storage_rgb_v51_20261012/`；补齐旧目录恢复保护后重新冻结合同并完整复跑，最终证据在 `checkpoints/storage_rgb_v51_final_20261012/`。两次均为独立 tmux、低优先级单 CPU、core=0、6 GiB 地址空间限制，无 GPU。

首轮实测：5 份回放 14,251,070 → 500,071 字节；RGB4 引用库本体 6,558,687 字节，外部共享回放另计一次。此前 v50 同范围库 20,805,713 字节，其中重复回放 14,268,323 字节。源码元数据变化使字节数略不同；此对比说明去掉重复文件，不是全源容量预测。

## 实际 RGB 复核

实际查看 5 张连续联系表、70 个不同 RGB 帧；另放大 U-E7 f40 原图，已包含在 70 帧内。每帧 RGB/metas/bboxes 共 210 份源 SHA 核对。已知标签和历史案例后选窗，属于开发复核，非 G2 盲审，也不代表整条路线逐帧目视。

| 事件与窗口 | RGB 观察及原生证据 | 本轮处置 |
| --- | --- | --- |
| U-E2，ConstructionObstacleTwoWays / Town01 001275，f72–85 | 前方施工牌仍在本车道，对向绿车接近并从左侧经过；f84–85视角开始绕行。原生 depart 在72/80为NO，84/85为catchup UNKNOWN；原后继f92导航截止仍无return | 本窗口不证明返回完成，不能补return YES |
| U-E3，ParkingCutIn / Town12 1681_0，f94–107 | 路旁蓝车逐步靠近行驶带，f104–107有切入运动；前后仍有其它车辆，不能把移动等同稳定恢复 | f104、107 proceed NO，保留运动/稳定门；不靠新增重复题补正例 |
| U-E5，InvadingTurn / Town12 1450_0，f36–49 | 雨夜反光明显，f36左下有近车，之后道路/锥桶与新靠近车辆可见；不能只按scenario名字把全过程当同一参与者 | f49 U-E5 proceed NO；后继55不可见UNKNOWN、56截止不改完成，需继续按实例身份核对 |
| U-E6，OppositeVehicleRunningRedLight / Town03 8_107，f44–57 | 雾雨路口有对向/侧向车辆；f52以后信号变化、自车开始转动，车辆仍在场景中 | 原生44/57 proceed NO，不能把绿灯或自车运动直接当全部让行条件满足 |
| U-E7，CrossJunctionDefectTrafficLight / Town03 002134，f27–40 | 横穿黑车由左向右通过，雾雨限制远方可见性；f40原图车已向右侧离开视野中心 | 原生38/39 corridor_clear=false、proceed NO；40变为true/YES，优先权true。本轮没有充分反证，也不能由视觉判断单独认证几何安全；既有f54 complete YES保留，未新增批准 |

以上未发现足以修改生产阈值或提示词的确定缺陷。可观察性与全池关键格覆盖仍需保留，不能因压缩/回归通过宣布标签全部正确。

## 峰值预算与下一步

新增只读入口：

```bash
python -m qwen3vl_local.audit_joint.training_budget \
  --request /实际路径/budget_request.json \
  --output /新的路径/budget_report.json
```

输入模板见 `training_budget_request_v51_20261012.json`。缺任一上限或证据，输出 `sufficient:null`、退出2；完整且符合声明上限时退出0。它不是建库启动器，也不会根据小样本压缩率自动填入全池容量。

增量预算逐项包括：共享回放一份、实际编译库、自动留出、Phase3索引、**每个保存轮次的 adapter 与 optimizer（两套模型分别计算）**、日志/审计包、临时峰值及20 GiB预留。既有基座、旧实验、残缺 data2 已计入当前 free，不再次当作新增或默认删除。`retained_checkpoints_per_mode=7` 会计算14份检查点；首轮实验应明确设为1，不能把首轮预算拿来担保七轮。`temporary_peak` 必须覆盖并发写入和临时保存；多挂载点需分别核算。

目前只有服务器既有可用量49,070,149,632字节和本机5路线测量，没有当前全池/留出/优化器的可靠新增上限，因此模板保持缺项，**尚不能确认全源建库与七轮训练一定装得下**。恢复工具已改为“新引用式编译不新增路线副本”，但不会因该下限为0就给 sufficient_for_build=true。

后续集中做当前源的候选清单与分批压缩回放/编译统计，补齐上述预算与全池五类关键格：U-E2 return、U-E3/5/6 proceed YES、U-E7 complete YES。沿用服务器已通过的环境，不再跑readiness或旧fit。当前规则全池及自动留出通过原生准入、十事件等量实际采样/评测检查后，才运行新合同两模式各一整轮和保存重载。G0仍未完整通过、G1继续、G2未运行。

本轮未训练、未远程执行、未push；完整回放、RGB和测试日志留在checkpoints，不进入源码白名单。
