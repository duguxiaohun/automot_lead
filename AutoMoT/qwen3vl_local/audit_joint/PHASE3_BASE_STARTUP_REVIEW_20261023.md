# Base 启动失败与真实全池容量复核

**后续进展：用户已明确允许稀缺数据有限重复，新版 bounded_v3 正式全池 CPU 已通过。** 当前运行方式见 [Base v3 运行说明](PHASE3_BASE_BOUNDED_RUN_20261023.md)。本文保留授权调整之前的严格去重拒绝证据；旧报告不改写。

当前结论：**base 的 1,280／事件、每帧每轮一次合同仍未通过真实容量准入。不要重新提交旧后台训练命令。** 本轮修复了训练索引提前截断和入口诊断，完成本机原生全池建库、原始依赖复核及实际选择题容量检查；没有通过重复、修改标签或挪动评估集制造 ready。

## 服务器失败原因

用户回传的 `prepare.log` 明确报 `insufficient_capacity: DYNAMIC_CUTIN; no unlimited repetition`。公共建库已经成功，错误发生在 coverage 采样准入；没有生成 `train.log` 是因为尚未进入训练。此前 106 项回归包含合成正式预算，但没有真实全池容量证据，不能证明该命令在真实数据上可训练。

旧 `geometry_study/remote.sh train six` 则要求其自身版本的 CPU ready。它不识别新版 `phase3_single_v3` ready；不要用旧入口运行新版 single。旧脚本保留原实验语义，新版入口是 `phase3_compare.sh`。

## 为什么数据很多，仍然不够

旧原生索引先把所有 train 事件裁到最少事件的候选数量，coverage 采样器又要求每事件 1,280 个不同帧。这两个要求没有经过真实全池联合验证。新增 `phase3_base_pool.py` 在独立目录保留全部原生训练候选；四图、标签、历史、原生 val/test 选择、INVALID 及旧目录均保留。

本机新索引：train 187,982 条，其中 1,850 条 INVALID 不进入 choice 训练；有效十事件候选如下。它不是服务器旧 12,384 条索引的逐字副本，两机数据数量有差异。

| 事件 | train 有效不同帧 |
|---|---:|
| LEAD_BRAKE | 2,782 |
| STATIC_BLOCKAGE | 58,063 |
| DYNAMIC_CUTIN | **925** |
| VULNERABLE_CROSSING | 28,183 |
| ONCOMING_INVASION | 3,641 |
| JUNCTION_RULE_CONFLICT | 1,622 |
| SIGNAL_FAILURE | **1,255** |
| POST_BYPASS_RETURN | 24,568 |
| UNSIGNALIZED_PRIORITY | 56,091 |
| RAMP_MERGE_EXIT | 9,002 |

另从原始标注重新追踪动态切入，逐帧结果与完整原生扫描完全一致：

- 映射后的事件候选 1,416 个，来自 218 条路线。
- 327 个被既有视觉／标注风险门排除：238 个为 `dynamic_cutin_actor_unverified`，89 个为道路结构 R4／信号灯上下文待确认。风险标记表示尚不能直接采用，不等于本轮已证明全部标签错误。
- 51 个为原生纵向动作混合阶段，4 个速度窗口不完整。
- 剩余 1,034 个：train 925、val 45、test 64。本事件本次没有因四图缺失或初始化历史门损失候选；不能把此次短缺泛称为四图必然比单图少。
- 新 single 的父候选入口没有同样的旧 `frame_visual_risk` 门，不能拿 single 的容量证明 base 同样够用，也不能为补数直接移除 base 的风险隔离。

用同一个新全池重放旧 `_balanced_work`：动态切入 925 个不同帧被呈现 1,280 次，其中 355 次重复；信号灯故障 1,255 个不同帧呈现 1,280 次，其中 25 次重复；两者单轮最大重复均为 2。旧采样器能跑与新帧去重合同通过是两件事。此为 CPU 重放，不代表恢复了某个历史训练运行的实际曝光。

## 本轮入口修复

- `phase3_compare.sh cpu base` 不要求模型或 GPU。独立内容绑定的全池生成、准入报告和 ready 指针位于 `checkpoints/phase3_base_pool_v1`；拒绝时留下各事件有效容量及原因。
- coverage 正式 launcher 使用同一完整训练池。旧默认预算入口保留。
- `check/train/probe` 在读大库和查询 GPU 前检查当前 ready；已经拒绝的容量会显示具体原因和报告，不再让用户把相同输入反复提交。
- 外层阶段失败会把底层日志尾部打印到启动日志，避免只看到 `prepare exited 1`。
- 新增 `phase3_compare.sh probe base`：前台执行数据／模型准备和真实四卡 probe 后退出，独立输出 `run_base_coverage_v2_probe`，不开始正式训练，不写 latest；同名输出拒绝覆盖。CPU 准入通过后才允许此操作。

本机 base 当前 ready 状态是 **rejected**，不是成功。继续实现需要确定容量策略：补充经核验的真实候选、调整事件配额，或明确允许有限重复。用户尚未选择；当前保持原 1,280／事件、单轮一次、全程三次的限制。即使重新接纳全部待审 327 个候选，也必须先核实标签和物理组归属，不能据此预先承诺容量通过。

在保留现有有效 holdout 成员及动作规则的前提下，极端假设 327 个风险候选全部通过且全部可进入 train，训练容量上界也只有 925＋327＝1,252，仍低于 1,280。这个上界不是允许清除风险门的理由。当前没有运行新增 RGB 人工审查或补标任务。

## 验证与交接

完整本机证据：`checkpoints/phase3_startup_review_20261023/`。其中 `cutin_filter_audit.json` 记录逐帧过滤原因，`legacy_sampling_replay.json` 记录旧采样重放；正式拒绝报告由 `checkpoints/phase3_base_pool_v1/ready_1280.json` 指向。源码／测试／报告散列见 `phase3_base_startup_verification_20261023.json`。

既有 single v3 和 abc v2 正式 ready 已重新通过原生加载校验，三轮均 3,200 更新；没有改其冻结标定或准入源码。旧 CPU 回执的 182 个文件中，仅公共入口 `phase3_compare.py` 因本轮改动不同，其余 181 个匹配，旧回执未重写。

本机缺少完整 Qwen3.5-4B 模型，未执行真实四卡 GPU probe／训练。CPU 通过、模型输入准备和 GPU 实测分别记录，不能互相替代。此次未 push。发布后服务器同步仍用 `bash ./sync_automot.sh`；在 base 容量策略解决之前，不提供可直接后台开训的成功承诺。
