# v76：Action 依赖前置检查与 v75 四卡证据补录

Phase3/4 的已选 B2、四卡、sequence 采样、训练/评估实现本轮未改，不再重跑 batch 选型。v75 目标4B四卡成功日志已经收到：服务器输出目录 `checkpoints/v75_four_gpu_20261017_105725_728264199`，CPU 41 项28.38秒，GPU两模式均 correctness_passed、四个rank、每卡B2，退出码0。两图/四图loss与未复用基线相同，最大梯度相对误差0.008143029893082774/0.0035750892768338194。原先的“GPU未执行”记录是发布时状态，现补充服务端证据到 `four_gpu_received_20261017.json` 并更新总回执，未改写既有CPU测试结果。

该GPU检查是重复一个真实样本、关闭dropout，核验DDP反向和分布式生成；不包含完整epoch、优化器恢复、全池效果或Phase3独立入口GPU验收。30秒Timeout文本是周期性调用栈；实际GPU父进程成功退出，不是240秒硬超时。

## Action 缺依赖的处理

`qwen_simple` 和 `bev_only` 均调用 LeadMoT runtime，真实依赖：

- `leaderboard/team_code/mot_lead_offline_runner.py`
- `Automot/mot/modeling/bev_encoder/bev_encoder_utils.py`

本机第一项缺失；它不是多余合同字段。保留全部源码哈希和缺失拒绝，新增统一前置检查，覆盖 launcher、直接 train/eval 和合同构建，在选卡、创建输出、读取数据及大权重前报告缺失项。无需启动GPU定位这个问题。

合同采样/路径迁移单测改为显式临时源码夹具；它们验证内容绑定、参数漂移和路径迁移，不再隐式要求本机有完整外部环境。另测生产缺失拒绝、内容变更、启动顺序；不是将生产缺文件标记为通过。优化器循环的无事件标签toy数据显式用natural评估，与已有Action主线toy测试一致，生产默认十事件等量不变，专项等量评估测试保留。

本机只读Git历史有移除跟踪前的旧runner，最后修改提交 `c28935ab3e8fb16aae7b3d707c0334084b559cc9`；没有证据表明它就是当前服务器版本，未自动恢复、改写或发布这个外部文件。需要当前运行环境的原始依赖；不能用占位实现或去掉哈希解决。**Action真实消融运行仍受本机缺依赖阻断**，测试夹具通过不代表该阻断已消除。

## 服务器只需一次CPU依赖检查

同步代码后在AutoMoT目录执行，通常不到1秒，不导入torch/模型、不使用GPU、不读取数据/权重、不产生输出目录：

```bash
python -S -m qwen3vl_local.action_expert_ablation.source_check
```

`sources_present` / exit0仅表示两个外部文件存在、可读取并给出SHA；不代表完整导入、模型或训练验收。`missing_or_unreadable_sources` / exit2会列出具体缺失项。服务器若原本存在正确版本，不需要复制或覆盖它。本机恢复依赖前需确认当前原版文件来源；交回此JSON可区分本机缺文件与服务器同样缺文件。

可选CPU专项（最多60秒）：

```bash
timeout --signal=TERM --kill-after=5s 60s python -m pytest -q \
  qwen3vl_local/action_expert_ablation/tests \
  qwen3vl_local/action_prior/tests/test_equal_event_evaluation.py
```

Phase3/4学生、原生、训练源码以及运行缓存绑定未改，已有v75 view/快照/checkpoint不因本次发布要求迁移；v75 GPU短检查无需重跑。Action消融自身执行合同新增前置检查源码，旧Action checkpoint仍应使用原版合同续训，不能跨源码强行恢复。外部源码、权重、日志不进Git发布。
