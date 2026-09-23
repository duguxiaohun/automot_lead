# 全局动作比例与小事件温和加权（2026-09-23）

本方案替代此前“事件先1:1、事件内动作容量回流”的 action-balanced。主线与
`action_expert_ablation/{qwen_simple,bev_only}` 共用实现；默认入口仍为 event-balanced。

## 目标与预算

| 模式 | 特殊样本分配 | 普通背景 | 默认重复上限 |
| --- | --- | --- | ---: |
| event-balanced | 十种特殊事件等配额 | 总量1/6 | 8 |
| action-balanced | 六种主要动作全局等配额，事件比例可变 | 总量1/6 | 8 |

六种语义动作是 DECELERATE、STOP、RESUME、LANE_CHANGE_LEFT、LANE_CHANGE_RIGHT、KEEP。
因此 UNCOND 加六种动作不是七类1:1；沿用用户此前指定的背景两份比例。
总预算须为 `lcm(12, world_size)` 的倍数。六类不能整除时次数最多差1，余数按epoch seed轮转，
连续六个完整epoch各动作累计相等。micro-batch、rank子序列和提前截断前缀不保证等量。

`event_balanced_epoch_samples=0` 时，action调用同源、同重复上限、同world的event预算函数，
再检查六类动作及背景的独立帧容量。足够则两模式每轮呈现数、同累积参数下更新数一致；
不足则报告 `action_capacity_deficits`，包含所需次数、独立帧、容量和最低重复上限。
预检检查余数轮转的最大配额，避免首轮可行而后续失败。不自动缩短epoch或增大重复上限。

这恢复与**当前event基线**相同的训练量，不承诺恢复任意历史数据版本的原数字。
旧run的116256次/轮可通过显式 `--event-balanced-epoch-samples 116256` 请求；
两组须各自通过容量检查。4卡、累积16时对应1817次更新/轮；v23过滤和划分后，
event模式在上限8下可能无法达到旧预算。选择共同可行N或显式改变两组上限，不能绕过检查。
自动event参考仍需要完整十事件支持；显式action预算不要求每事件都存在，但必须有六类动作和背景。

## 温和提高小事件权重

仅从训练池计算支持所选主要动作的事件帧数 `n_e`，每事件权重为：

```text
w_e = min(2, sqrt(max_e(n_e) / n_e))
```

大事件基准权重1，小事件最高2，不把每个事件或每个event×action都拉到等量。
例如同一动作的大事件1000帧、小事件10帧，加权质量约为1000:20，而不是1:1。
提升是相对于该动作内的自然按帧分配；不保证比旧event硬均衡的次数更多。
2倍是配额权重上限，实际帧选择还受整数分配、容量和物理路线优先影响。

同一帧只有一个主要动作；先按“动作＋支持该动作的事件集合”分成互不重叠的池。
并发事件权重取均值，不相加，避免多事件帧额外获得倍数权重或被重复统计容量。
例如RE2+RE5的右变道只保留RE2采样归属，原RE5事实和全局右变道标签仍保留。

每个动作内部按“池独立帧数×平均事件权重”分配整数配额；池容量为帧数×重复上限，
饱和池的余量回到同动作其它池。池内使用独立seed的物理路线轮转，完整覆盖池后再循环。
由此每帧全epoch重复严格不超上限；允许为温和事件加权在小池循环，同时大池尚有未抽帧，
所以不宣称跨所有池全局最大化不同帧。最后共同打乱全epoch，再按rank切片。

这些权重是初始实验设置，没有模型效果最优结论；不依据val/test计数或错误率调整权重。
合法稀少动作不改KEEP/UNCOND，不放回被过滤/未确认帧，不伪造缺失动作。

## 记录与恢复

`policy_contract.version=action_balanced_global_soft_event_v4` 绑定规则和动作标签来源。
`training_plan.json` 中 `sampling.action_balance` 记录全局动作支持、权重和首轮配额；
`epoch_quotas`为首轮实际事件归属数，不再声明事件硬性1:1。
`sampling/epoch_*.json`记录实际动作/事件/格子次数、唯一帧、路线和重复直方图。

并发样本在配额报告中每次只归属一个支持事件，均匀轮转归属仅用于计数；
输入和事实指标保留全部事件，因此 `group/event_balance/*` 可以重叠，不能直接相加当总样本数。
`support.cells/events`及低支持诊断继续保留；这些诊断阈值不参与过滤或配额。

动作采样不自动开启token、文字动作先验或改变Qwen输入；token关闭也读取标签用于选样。
Phase3原动作标注、初始化过滤、开发路线隔离、holdout支持补齐、验证/测试遍历均保持。
新规则需新run；旧两层采样/上限2的run用原源码恢复，不自动迁移checkpoint条件或训练计划。

## 历史全量池回放

使用已有的843913帧v21标注训练池，直接测试新采样器。1/4 rank各7轮均通过；
该池不是当前v23生产索引，没有重新标注、重建生产数据或加载真实Qwen/BEV。
本地机器结果为 `/tmp/global_action_replay.json`，不上传完整数据产物。

| 项目 | 新action模式 |
| --- | ---: |
| 每轮呈现（与同池event参考一致） | 95136 |
| 背景UNCOND | 15856 |
| 每种语义动作 | 13213或13214 |
| 每轮不同帧 | 88910–88912 |
| 实际最大单帧重复 | 3 |

第一轮88911个不同帧中82691帧出现1次、6215帧出现2次、5帧出现3次。
上限8没有导致每帧重复8次。实际event归属数不等量，例如UE3为446、RE2为20242；
事件均衡效果与动作均衡效果应在共同预算、split、seed、GPU数、累积设置下比较。
这些数值只证明这份历史池的预算/比例/容量，不证明新模型轨迹指标提升。

## 验证

402项相关CPU回归通过；3项依赖本机缺失的只读 `mot_lead_offline_runner.py` 未执行，未绕过生产合同校验。
覆盖三入口真实小模型训练/恢复、共同预算、全局动作比例、整数轮转、温和权重、并发去重、容量失败及来源准备。
另通过修改文件的Python/Bash语法检查，未运行真实GPU训练。

## 简易demo

从 `AutoMoT/` 执行。默认自动选卡；需要固定卡时前置 `GPU_IDS=0,1,2,3`。

```bash
# 默认event参考
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors
# 全局action + 小事件温和加权 + token + 单当前图
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --action-balanced --high-level-action-token --rgb-frame-count 1
# 两个消融共享同一规则
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --action-balanced --high-level-action-token
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --action-balanced --high-level-action-token --rgb-frame-count 1
# 若需要固定旧预算，在两组命令都追加以下参数，并通过各自容量预检
# --event-balanced-epoch-samples 116256
```
