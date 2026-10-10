# 十事件本机标定续审 — 2026-10-10

后续已将本页局部隔离接入原生v43生产，并完成新一轮十事件重算及局部原生题库。最新状态见[原生标定修复](NATIVE_LABEL_FIXES_V43_20261010.md)；下文保留前一里程碑的历史范围。

本轮推进的是本机标定和数据准入，不是模型训练。完成 45 条既有人工作为 train 的路线、5,673 帧原生重算；实际目视 8 条路线的连续局部窗口、247 个不同 RGB 帧（16 张联系表，其中 4 张原图再核），没有新增完整路线目视、G2 盲标或独立批准。**十事件有候选，不代表所有状态/分支/答案完整，更不代表训练、验证、测试都是 1∶1。**

## 已完成的代码与产物

- [local_replay.py](local_replay.py)：可复用的 Phase3/4 本机逐帧生成入口。要求请求列出整条路线的 RGB 帧，拒绝用抽样锚点替代；校验原生物理组与 split，仅接受训练路线。Phase3 重算动作/证据和原生候选，Phase4 连续回放，按同路线同帧关联；来源、UNKNOWN、原因、实例结束和原始标签均保留。
- [label_quarantine.py](label_quarantine.py) 与 [证据登记](label_quarantine_20261010.json)：按原图、meta、bbox 六份 SHA 绑定两处已确认车辆消失。Phase3 检查输入及实际标定依赖范围；Phase4 检查完整因果历史、实例和 STOP 证据。只隔离，不翻转 YES/NO。
- [label_inventory.py](label_inventory.py)：先核验回执/逐帧全集，再生成隔离后候选、十事件计数、逐分支零支持表和事件存在性复核队列。无 Phase4 问题归为 abstention，不能解释成“事件不存在”。两个阶段事件集合不同也只记候选分歧，不冒充错标结论。
- [run_local_replay.sh](run_local_replay.sh)：tmux 内可调用的单 CPU 进程入口。单线程数学库、低 I/O 优先级、nice 15、最多两核亲和、6 GiB 地址空间上限；可用内存低于 6 GiB 暂停，磁盘余量低于 20 GiB 停止。限制只作用于本任务。

原始重算：`checkpoints/ten_event_label_audit_20261010/`。正式 CLI 实跑复验：`checkpoints/ten_event_label_cli_20261010/`。**最终局部候选以 `checkpoints/ten_event_label_final_20261010/` 为准**；先前 `curated`/`curated_v2` 为保留的开发中间产物。

最终包含 Phase3 候选 1,637 条；Phase4 准入门后候选 7,170 条，为原生两图、四图各 3,585 条，不能当作 7,170 个独立实例。原始 Phase4 问题 10,920 条，包含 UNKNOWN。

**这些 JSONL 仍不是可直接训练的完整数据集**：Phase3 尚未完成全局 split/覆盖和 INVALID 生成；Phase4 尚未执行最终去重、同输入矛盾排查和人工优先合并。所有保留记录都标 `diagnostic_only`，最终报告 `approved_for_training=false`。新增隔离已作用于本次局部候选，未接入默认 v42 正式建库/服务器训练；旧库不能被称为已经应用了本次修复。

原生 Phase4 两图仍是 `[-4,0]`；本轮不是短两图 `[-2,0]` 的可学性认证。后者应沿用现有 `rgb_short` 输入投影，但需要按实际有限输入重新审核，不能迁移四图的批准。Phase3 四图为 `[-3,-2,-1,0]`。

## 两处确认的问题及精确影响

1. **R-E3，MergerIntoSlowTraffic / Town13_Rep0_1230_12_route0_01_10_00_03_59，第 58→59 帧。** 红色前车在距自车约 9.6 米时直接消失；原图无自然出视野或遮挡解释，actor 4745 同时从 bbox 消失。绿色横穿车是另一参与者，不能把它与前车合并。新的因果隔离命中第 65 帧 enter/hold/restrict，两个图数共 6 个原始问题；restrict 原来已经被准入门拒绝，所以真正减少的是 4 条可保留弱候选。
2. **U-E6，OppositeVehicleRunningRedLight / Town07_Rep0_Town07_Scenario8_19_route0_01_10_20_11_38，第 43→44 帧。** 消防车仍清晰显示在左前方（actor 138，6,741 可见像素），下一帧整车消失，自车近乎静止。Phase3 最终仅隔离第 43 帧候选，因为其停车确认读到了第 44 帧。

首次用完整 +2 秒包络筛出 8 条 Phase3 STOP 只是粗筛；复核原标定器后，第 36–42 帧均在控制未释放时已确认当前等待，不依赖后续车辆消失，故保留 7 条明确 STOP。最终是 **1 条待复核候选，不是认定 8 条错误标签**。横向问题仍需要完整横向证据，最后一次跨线的确认会读到 `t+13`，不能漏算成只到 `t+12`。

两处新增证据与现有人工题的输入包络交集为零（原生 train 443 / val 273 / test 186，只读身份核对）；不据此撤回人工答案。也不声称已经检查过服务器 330,902 条全池的受影响数量。

U-E4 第 168→169 帧有骑行者与对向车重叠，联系表不足以区分遮挡和删除，**未登记为确认消失**。夜间亮度、负静态 extent、U-E7 假设状态等仍没有通过本轮被放宽。

## 局部覆盖：不要与服务器全池混用

下表 Phase4 为四图单模式、准入门和本轮隔离之后，尚未完成最终编译。两图的相同计数不作为额外独立支持。

| 事件 | Phase3 原候选 | Phase3 保留 | Phase4 YES | Phase4 NO |
| --- | ---: | ---: | ---: | ---: |
| U-E1 | 366 | 366 | 31 | 1005 |
| U-E2 | 260 | 260 | 98 | 461 |
| U-E3 | 9 | 9 | 101 | 407 |
| U-E4 | 170 | 170 | 17 | 47 |
| U-E5 | 153 | 153 | 2 | 14 |
| U-E6 | 29 | 28 | 3 | 54 |
| U-E7 | 34 | 34 | 40 | 165 |
| R-E2 | 168 | 168 | 92 | 544 |
| R-E3 | 42 | 42 | 53 | 132 |
| R-E5 | 407 | 407 | 23 | 296 |

这些计数显然不是答案 1∶1；事件等额抽样也不能补出没有证据的边或阶段。`phase4_branch_support.json` 显式列出本批弱候选中 199 个 readiness 空格，是局部、未合并人工题的缺额，**不是全池缺额**。

值得优先继续补证据：

- U-E5/U-E6：本批没有 proceed/readiness YES，complete/release 也有结构缺口；不能凭其他边的 YES 宣称推进覆盖完整。
- U-E7：本批仍无 release 正负候选；需要审核可到达/滞后状态与同一参与者，而非强行把 proceed 复制成 release 或将 catchup UNKNOWN 改 YES。
- U-E4：四图 705 个原始问题中 636 UNKNOWN；495 次带 `decisive_actor_not_visible` 原因。先针对参与者可见性和实例范围查 RGB，不能靠降低亮度门解决所有情况。
- Phase3 U-E3：本批原生合格候选仅 9 条，需要继续分清事件映射、动作窗准入与可观察性，不能靠邻帧重复掩盖来源不足。

最终队列把 Phase4 无问题的帧与事件集合分歧分开；仅事件名相同仍不是实例/参与者匹配完成。统计见最终 `summary.json`，不要把初版旧 `presence_status` 的事件不一致粗计数当成错标率。

## 复现与资源

以下命令从 AutoMoT 目录执行，选择新的输出目录，拒绝覆盖旧证据：

```bash
tmux new-session -d -s local-label-audit   'bash qwen3vl_local/audit_joint/run_local_replay.sh --request checkpoints/ten_event_label_audit_20261010/request.json --output-dir checkpoints/label_replay_new > checkpoints/label_replay_new.log 2>&1'

python -m qwen3vl_local.audit_joint.label_inventory   --replay checkpoints/label_replay_new   --output-dir checkpoints/label_candidates_new
```

可用 `PYTHON_BIN=/path/to/python` 指定兼容的现有 CPU 环境；不下载 Qwen。`--collection-dir` 默认原始 collection；复验时可指向已冻结的完整所选路线 collection，并在新回执保留其 SHA。tmux 保证终端断开后继续运行；不意味着机器重启、OOM 或异常终止后自动续建，失败须看 progress/log 并用新目录重跑。

第一次原始扫描/重算 480.02 秒，峰值 RSS 392,036 KiB（约 383 MiB）；复用所选 collection 的 CLI 复验 362.83 秒、372,616 KiB。45 路线均退出 0，两次 Phase4 原始问题逐字节一致，5,673 帧 Phase3 动作与证据一致。原 v42 生产合同、Phase3 mapping 均保持。专项新增 14 项、audit_joint 全套 **207 项通过**。证据见 [local_label_verification_20261010.json](local_label_verification_20261010.json)。

本轮没有训练、GPU/CARLA、全池重建、独立人工批准或 push。下一阶段把已确认隔离通过有版本合同的新生产入口接入并全池复核，同时继续十事件分支/状态补审；G0、G1 实例匹配、G2 和六事件开发验证集仍未完成。本机标定工作不等待服务器 181 题拟合检查。
