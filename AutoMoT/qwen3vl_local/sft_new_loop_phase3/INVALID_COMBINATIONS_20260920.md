# 自动 INVALID 组合与稀疏人工负例

用户要求用有根据的 RS/event 错误组合构造更均衡的 INVALID，避免开训依赖不断补充人工负例。
本次保留 v19 提示词、KEEP、动作标定、split seed=20260920；修改负例候选/抽样和评估范围。

## 原因与修改

此前配额不足修复后，binary 预检继续因 val/test 的同道路错事件负例只有一条独立物理路线而退出。
现在人工事件负例是补充诊断：0/1 条路线会显示 `insufficient_support`，允许训练，也不将其子组准确率
用于 checkpoint 守卫。达到至少两条时，仍要求该子组 exact >= 0.5。支持不足时 `passed=null`，
`evaluation_complete=false`，独立评测 `production_ready` 保持 false；best 可按已支持的指标选择。
这不等于充分验证了事件前提拒绝能力。标签/来源缺失、物理路线泄漏、必需训练类别缺失仍严格拒绝。

自动负例依赖已有道路标注及几何证据，枚举下面所有与被问事件兼容的错误 RS：

| 真实道路与几何条件 | 可构造的错误 RS |
| --- | --- |
| R1/R2，不在路口，距下一路口至少25m | R3、R4、R5 |
| R3，不在路口 | R4、R5 |
| R4/R5 | R3 |
| 不满足上述条件 | 不自动构造 |

所有十类事件均参与候选枚举，但每个事件仅搭配其 `allowed_rs` 支持的错误道路。
例如同一真实 R3 图像可分别问 R4/静态障碍与 R5/静态障碍；INVALID 的依据是道路前提错误，
不声称图像中必然没有静态障碍。不会把“未标某个事件”直接当成“事件不存在”。
没有足够互斥证据时不自动伪造 R1/R2，也不承诺所有五种错误RS或全笛卡尔积数量相等。

抽样采用来源均衡（最大差1）、来源内真实RS/事件联合签名均衡，再在联合签名内平衡错误RS。
覆盖种子计入配额；人工负例无重复输入，25%为软补充目标。
每个有自动负例的来源至少保留一条，25%补充不得替换掉最后一条自动负例。
这修复了连通测试发现的另一失败：五个人工题占满 RE5 来源后，验证预算增大无法重采样。
该回归所需预算从41变成51；已有自动增容按新覆盖计划处理，只调整 INVALID 桶。

manifest 的 `invalid_balance` 新增候选 `candidate_true_prompt_rs_context`、抽样 `prompt_rs`、
`true_prompt_rs_context`、候选与抽样人工覆盖报告。训练验证采样报告也输出相同维度。
raw index 审计从原始 meta 重查自动负例的路口条件与可构造组合。

## 重跑

当前映射哈希已变化，需要重建索引。不要直接跳过构建复用之前失败的索引，旧 checkpoint 用原代码恢复。
无需手动提高配额或修改 seed。从远端 `AutoMoT/` 运行，先只构建和审计：

```bash
SKIP_TRAIN=1 SKIP_EVAL=1 ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```

再用实际 PyTorch 环境执行训练共用采样预检（不加载模型权重、不初始化 NCCL）：

```bash
python qwen3vl_local/sft_new_loop_phase3/train.py --sampling-only --action-output-mode binary \
  --index checkpoints/sft_new_loop_phase3_data_v19/frame_index.jsonl
```

通过后复用该索引启动训练；默认自动选卡，或显式指定：

```bash
SKIP_BUILD=1 ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
GPU_IDS=0,1,2,3 SKIP_BUILD=1 ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```

两图添加 `HISTORY_RGB_MODE=2rgb_endpoints`；选择题显式设 `ACTION_OUTPUT_MODE=choice`，分别新训。
Action 自动准备候选复用同一个构建器，也会获得更新的自动负例实现；其下游动作索引仍读取有效候选原始动作。

## 验证范围

无 torch 回归覆盖安全组合、错误RS内均衡、无/单路线人工负例的构建→运行时公共抽样、
预检、重复路线去重、泄漏拒绝和 checkpoint 守卫。已有 PyTorch 测试的预期也同步更新。
本机未运行完整7000余路线构建、PyTorch训练入口或Qwen模型，不据此保证全量数据或模型效果。
本轮不新增人工事件负例决定；此前临时查看的 RGB 不作为新增生产标注。

本轮验证：474项Phase3无torch测试通过（另4项依赖测试未选，3个依赖torch模块未收集）；
41项Action数据准备测试通过。扩展Action测试中的5项及high-level测试收集因缺torch未能执行。
26条既有正候选开发路线的小集合重建出1148条正候选、384行索引；纳入既有纯负例后实际27路线，
原始meta回读384行无动作错配，RGB路径检查728个文件。320条正例除mapping哈希外与上版逐字段一致；
64条INVALID的提示RS分布为R3=39、R4=10、R5=15。
这些是真实小集合重放，不是全量生产验证；全三split覆盖和无/单人工路线行为由合成回归验证。
审计产物位于同目录 `probe_output/invalid_combinations_20260920/`，不入库。
