# v13：Phase3 与 action_prior 共用主要动作 / NONE

Phase3 默认保持四图 choice，每次输出一个主要动作或 `NONE`。纵向域四选一，机动域六选一；
保留条件性场景目的，不生成解释。binary 仍显式可选，用于原始纵横标签诊断。

## 统一标定

`primary_action.py` 是两包唯一的主要动作投影实现。先限制当前允许的动作域，然后：

1. 原判据确认当前持续等待或 1.5 秒内近停：`STOP` 优先。
2. 否则，未来 3 秒内首次跨线为主要动作，配合的 DECELERATE/RESUME 不另输出。
3. 无跨线时保留未来 2 秒内符合原判据的纵向动作。
4. 没有动作达到判据：`NONE`。缺帧、未知、无效前提不伪装成 NONE。

因此 DECELERATE+LEFT→LEFT，RESUME+RIGHT→RIGHT，STOP+LEFT→STOP。
这是明确的有限窗口优先级定义，不声称精确识别了动作起始阶段，也不因事件成立而强制动作。
速度阈值、STOP 双采样确认、首次跨线、原始 RGB 审计/隔离和路线划分保持原规则。
索引保留原始 answers/action_signature，另写 primary_action 与版本；证据重算仍针对原始标签。

## 训练、评测与下游

有效 NONE 和原始组合样本均进入 choice，组合在构造监督时投影；invalid 仍剔除。
按 context 内主要动作（含 NONE）平衡。严格 parser 接受一行候选名称；NONE 解析为动作全 false、
invalid false，不会与格式错混淆。验证与独立 eval 同步统计 NONE 支持、precision、recall、F1，
best 守卫要求五动作及 NONE 各有支持且 P/R 达门槛；不能靠总分掩盖 NONE 误触发。
choice 的 gt/exact 是主要动作，action_answers 与 action_signature 保留原始证据。
binary/choice 的完整指标不能直接混比，历史单动作子集成绩也不能沿用为 v13 成绩。

action 自动索引改为 scoped_phase3_primary_action_v4 / primary_choice_v1：actions 最多一个，
oracle 的 candidate_actions 保留原始动作，门控后才再次选择主要动作。
UE 需上游 YES/域有效/RS 兼容，RE2/3/5 需独立 transition gate，scope 不补场景事实。
只有 selected 注入一个动作句；NONE→no_action 保留场景事实和条件性 planning，不写 NONE/NO 字样。
普通 RE 为 not_applicable，文件缺帧/解析错为 unavailable，三者审计不混淆。

未来 provider 可调用 action_input.choice_action_input：词组→单动作、NONE→no_action、非法→unavailable。
prediction/provided 文件必须使用新 schema/format 和精确帧/scope，不能夹带 oracle 原始证据。
本次不加载 Phase3 LoRA，不实现在线 provider；自动标注仍是显式离线未来真值实验。
统一语义不保证预测准确率或 oracle 到预测的效果一致；闭环仍拒绝该离线模式。

## 新训练 demo（从 AutoMoT/ 运行）

```bash
# Phase3：新建 v13 索引，默认 4rgb + 主要动作/NONE choice
bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh

# 构建后先做两个训练 step；跳过 pipeline 的最终独立 eval
TRAIN_MODE=check SKIP_EVAL=1 bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh

# 显式 binary 诊断实验，使用独立训练输出
ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh

# Action：自动重建候选与 v4 动作索引；默认不生成摘要
bash qwen3vl_local/action_prior/run_full_pipeline.sh \
  --dataset-priors --event-balanced --high-level-planning --high-level-action-prior

# 上述无噪声 dataset + planning 已自动包含 RE2/3/5 的独立场景条件；也可显式选卡
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh \
  --dataset-priors --event-balanced \
  --high-level-planning --high-level-action-prior

# planning-only 对照：不读取逐帧动作
bash qwen3vl_local/action_prior/run_full_pipeline.sh \
  --dataset-priors --event-balanced --high-level-planning
```

Phase3 新索引目录为 checkpoints/sft_new_loop_phase3_data_v13。旧索引、adapter、action v3 索引和
旧 checkpoint 不在新代码下续训；原产物保留，用原源码恢复。自动准备器按新来源哈希另建缓存。
不用 SKIP_BUILD=1 指向旧索引，也不通过手工改版本字段绕过合同。

本次验证以合成合同、实际 CPU 采样与张量回归为主；未跑全源重建、Qwen 真训练或闭环。
新任务是否提高轨迹效果，需要在完整训练环境重新评测。

## 本地验证结果

使用现有 pvi Python 环境运行两个包完整 pytest：817 通过，19 失败。失败均为已有环境缺口：
14 项依赖缺失的只读 mot_lead_offline_runner.py，5 项缺 peft；未安装或伪造依赖来绕过检查。
覆盖 120 个 context×纵横组合的一致投影、NONE/格式错区分、真实生成评测计数、上游门控、
自动索引发布/复用/篡改拒绝、缓存及原有训练采样回归。122 个 Python 文件 AST、16 个 shell
语法和 git diff --check 通过。真实 Qwen/BEV 权重及新索引未在本机准备，未启动训练。
