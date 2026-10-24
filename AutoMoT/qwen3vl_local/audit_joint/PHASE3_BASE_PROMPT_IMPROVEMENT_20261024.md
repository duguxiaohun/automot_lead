# Base 审计结论落地：提示词配对实验与整体回归保护

2026-10-24。依据 RGB_REVIEW 与 RELABEL_REVIEW 两轮结果，新增可运行的四图单选提示词实验，补强全量验证集回归保护。没有修改现行训练标签、采样预算或默认提示词；本轮没有新增目视 RGB 审查，没有模型推理或性能改善结论。

> 发布修复：事件内各动作 precision/F1、事件 macro recall/F1 和格式有效数现均触发退步复核，不能再由另一事件抵消。评分 CLI 同时接受 JSON 数组和 JSONL，可直接读取 replay 导出。64 项相关测试通过。新建并校验的本地计划为 `checkpoints/phase3_base_prompt_release_v2_20261024`；下文旧 `_final` 计划保留，但其评分器源码绑定已过期。当前操作步骤见 [base 训练与提示词测试](PHASE3_BASE_PROMPT_RUN_20261024.md)。

## 已实现

`phase3_base_prompt_experiment.py` 提供 `prepare` 和 `replay`。新候选 `base_horizon_clarity_v1_20261024` 在实际使用的 `v23_rgb_stage_candidate_20261006` 前增加统一时间说明：

- STOP 的近零连续两帧确认必须落在当前至 +1.5 秒；更晚停车不能单独覆盖先发生的减速。
- 纵向观察到 +2 秒，包含恰好在末端跨减速阈值的情况；减速不额外要求双帧，增速仍要求双帧。
- 允许横向动作的题域看 +3 秒内首次未来跨线；原有 STOP、横向、纵向优先级和所有动作选项原样保留。
- 区分当前继续等待与立即确认起步，不因未来可能放行就推断已经起步。

这是通用任务定义，不向输入加入每例控制量、未来速度、未来 RGB 或答案。每条构造都会翻转答案再次渲染，确认文本不变。现有场景、速度、目标、四图与题序保留。候选增加 727 个英文字符；全部原文本长度 2758–4026，候选 3485–4753 字符。字符不是 tokenizer 长度；真实 replay 使用完整 processor 检查多模态输入与模型 context 上限，不截断。

准备必须覆盖参考文件的全部 val，拒绝缺项、重复、目标/旧预测不一致、提示词错配及四图顺序异常。图像按 scenario/run/rgb/frame 精确重定位至指定 data-root，绑定本地 SHA。原包 val 没有逐图远端 SHA，因此本地绑定不能称为远端图像一致认证。test 不生成候选请求。

Replay 对每条样本使用同一个 adapter、同样四图，分别运行原文本和候选，fresh KV、固定 seed，保存实际原始生成文本和输入 token 数。原 adapter 合同由原生 validator 检查，不改元数据、不绕过模型路径检查。记录模型/adapter 文件内容散列、源码与请求散列，前后复核；失败保留 started/部分生成，只有完整成功才有 report.json。输出目录已存在则拒绝。

主比较是**本次原提示词推理对本次候选推理**；另报原提示词与收到的旧结果差异。不会把旧结果重放不一致算作新提示词收益。候选使用旧 adapter，属于推理期提示词迁移实验，不代表新提示词训练效果。默认单进程执行，共 640 次生成；未实测显存/时长。

## 回归保护修复

`phase3_base_regression_review.py` 原先只在子集净正确数下降时触发警示，可能将同组一修一损视为无退步。现在新增：

- 任意原正确样本损失均明确 `review_required=true`；没有净退步但存在损失时为 `tradeoff_review_required`。
- 动作 precision/F1 下降也触发检查，防止靠大量误报提高某类召回。
- 增加 event×action，事件/路线宏平均下降也纳入检查。
- CLI 对净退步或抵消损失返回退出码 2，同时完整保存 fixed/lost；自对照无损失返回 0。

这些是保守的开发复核信号，不把小组损失当统计显著性，不自动授权替换。已开发曝光的 val 不能证明全池泛化不退步。需要保留原标签的独立评测；本轮不利用 test 选提示词。

## 实际验证

恢复原包 SHA 为 `c7ecb14d5f76d0488c4d7a84a53a2afe5b907859a842e0baf5f75213f7d301f1`。最终计划位于 `checkpoints/phase3_base_prompt_improvement_20261024_final`，完整 320 条 val、1280 次图像引用均已读图像文件散列，原预测自对照 251/320，无 fixed/lost。这不是模型运行。

52 项相关 CPU 测试通过，含 10 项新入口测试、10 项回归检查测试、16 项标定实验室测试、16 项既有 prompt 测试。覆盖真实 prompt 渲染、成员/图像/源码篡改、输出碰撞、成对推理编排、context 超限、同分抵消及 precision 下降。推理编排测试使用替身，不是 Qwen 前向。原生 prompts/prompt_candidate/prompt_contract/trajectory_action/choice_semantics/eval 六文件与 Git HEAD 字节一致；相关旧标定边界回归通过。

最终证据：`manifest.json`、`verification.json`、`selfcheck.json`、`tests.xml`。初次准备目录 `phase3_base_prompt_improvement_20261024` 保留为开发中间结果；它的源码指纹已过期，不能用于 replay，应使用 `_final`。

## 使用

在 AutoMoT 根、已安装项目依赖的 Python 环境下，重新准备时必须使用新输出目录：

```bash
PYTHONPATH=. python -B -m qwen3vl_local.audit_joint.phase3_base_prompt_experiment prepare \
  --archive '/home/codon/Downloads/phase3_report(4).zip' \
  --reference checkpoints/phase3_base_relabel_audit_20261024/eval_boundary_reference.json \
  --data-root lead_data --output checkpoints/base_prompt_plan_new
```

本机最终计划已准备完成。有完整原模型和本次 base adapter 的机器上可执行（路径替换为实际资产，adapter 中模型路径须与原生合同一致）：

```bash
PYTHONPATH=. python -B -m qwen3vl_local.audit_joint.phase3_base_prompt_experiment replay \
  --plan checkpoints/phase3_base_prompt_improvement_20261024_final \
  --data-root lead_data --model /path/to/original/model \
  --adapter /path/to/base/final/adapter \
  --device cuda:0 --output checkpoints/base_prompt_paired_new
```

读取 `report.json` 的 `saved_baseline_predictions_changed`、`paired_prompt_effect`、每类/每事件/语义/路线 fixed 与 lost。整个 replay 完成不等于通过效果审查；`promotion_authorized` 始终为 false。更改实验源码后须重新 prepare，不能改清单 SHA 继续用旧计划。

## 为什么保留标定与采样

两轮审计已实际证明三种简化重标各有副作用，本轮把它们继续作为独立实验而非默认“修正”。动作曝光偏移值得验证，但这次固定权重、标签、图像单独测试提示词；没有同时重采样以免混淆原因。候选提示词可能无收益甚至退步，必须接收完整配对推理结果后再决定是否训练新版本。现有资产不足以在本机执行这一步，未下载模型、未启动远程任务、未 push。
