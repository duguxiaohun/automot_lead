# Base 训练与候选提示词配对测试

2026-10-24。本次发布包外提示词实验、标定实验室及防退步检查。生产 base 仍为四图 bounded_v3，原标签/原采样/原提示词、每轮 12,800 呈现与 3,200 更新、三轮 9,600 更新。新 `base_horizon_clarity_v1_20261024` 只用于推理期配对实验，没有接入训练入口；`native_clarified_v1` 是另一项单帧实验，不能当作本候选传给 base。

先在服务器运行现有同步脚本，再进入其 AutoMoT 应用根目录。应用根可直接看到 `qwen3vl_local/`、`checkpoints/`、`lead_data/`；不要求有 `.git`。

```bash
bash ./sync_automot.sh
```

激活服务器原有 Qwen3.5 训练 Python 环境。若使用 `.venv`，shell 训练入口会优先选它；下面的 Python 命令也必须用同一解释器。`checkpoints/Qwen3.5-4B` 必须是本次 adapter 对应的完整本地模型，含原 tokenizer/processor；工具不下载模型。不要修改 adapter 合同中的路径/SHA 来绕过核验。

## 已有完成的 base：先测试候选，不必重训

你已返回训练完成的 base 包。原生完成情况可只读查看；若还没有外层完成回执，可补验收：

```bash
bash qwen3vl_local/audit_joint/phase3_compare.sh status base
bash qwen3vl_local/audit_joint/phase3_compare.sh verify base
```

`verify base` 只验收 canonical `checkpoints/phase3_compare_base/run_base_bounded_v3`，不适用于下面另起的 repeat 路径；它不训练、不重新评测。不要重新 `train all` 或删除旧输出。

本机已生成当前代码绑定的 320 条完整 val 计划：

`checkpoints/phase3_base_prompt_release_v2_20261024/{manifest.json,reference.json,requests.json}`

这三份是本机审计数据，**不在 Git 白名单发布中**。在服务器配对测试前，需将这三份原样放到服务器同名目录（用户自行传输；不是重新生成 SHA）。请求中的图像为 data-root 相对路径，服务器会核验每张图的本地 SHA。若服务器图像不同会拒绝，不能改清单绕过。旧 `_final` 计划因评分器源码变化不再兼容。也可在具备原包和完整参考的机器上重新准备到新目录：

```bash
PYTHONPATH=. python -B -m qwen3vl_local.audit_joint.phase3_base_prompt_experiment prepare \
  --archive /path/to/phase3_report.zip \
  --reference /path/to/eval_boundary_reference.json \
  --data-root lead_data --output checkpoints/base_prompt_plan_new
```

`eval_boundary_reference.json` 是上一轮本地审计参考，包含全部 val 成员和语义切片；不是任何其它模型的预测文件，也不在 Git 中。已有完整三文件计划时无需再 prepare。

用已完成的 canonical base 第三轮 final adapter 执行原/候选两组推理；路径若不同请按实际 `experiment/train/final` 替换：

```bash
PYTHONPATH=. python -B -m qwen3vl_local.audit_joint.phase3_base_prompt_experiment replay \
  --plan checkpoints/phase3_base_prompt_release_v2_20261024 \
  --data-root lead_data --model checkpoints/Qwen3.5-4B \
  --adapter checkpoints/phase3_compare_base/run_base_bounded_v3/experiment/train/final \
  --device cuda:0 --output checkpoints/base_prompt_pair_review_v1
```

入口在导入推理环境前默认选择空闲 GPU；需要指定物理卡时仅在命令前加 `GPU_IDS=0`，`--device cuda:0` 指选择后的可见设备。使用单进程，两种提示词共 640 次生成；没有显存和耗时实测。输出目录存在则拒绝，不隐式续跑。

`report.json` 分开记录 `saved_baseline_replay`（历史输出与本次原提示词）及 `paired_prompt_effect`（本次原提示词与本次候选）。候选评估必须使用 `fresh_reference.json`，不能误用历史参考：

```bash
PYTHONPATH=. python -B -m qwen3vl_local.audit_joint.phase3_base_regression_review \
  --reference checkpoints/base_prompt_pair_review_v1/fresh_reference.json \
  --predictions checkpoints/base_prompt_pair_review_v1/candidate_predictions.json \
  --output checkpoints/base_prompt_pair_review_v1/independent_review.json
```

CLI 支持 JSON 数组/JSONL；需要复核时写完整报告后返回 2，无观察到退步返回 0。查看总体/动作/事件内 precision、F1、语义分组以及 fixed/lost；任意原正确损失也须复核。`replay` 生成完成不等于质量通过，始终不自动授权替换。原生准确率不能用重新标注或挑选 errors 的分母替代。

## 确实要另训一个原生 base

已有模型仍保留。另起独立输出目录，不与旧运行碰撞：

```bash
bash qwen3vl_local/audit_joint/phase3_compare.sh cpu base && \
bash qwen3vl_local/audit_joint/phase3_base_bounded.sh \
  --output checkpoints/phase3_base_repeat_20261024 --run-tag base_repeat_20261024
```

第二条依次做输入准备、真实四卡探针、三轮训练、摘要、最终第三轮 test 和完成验收。正式任务内已先跑四卡探针，无需为了开始正式训练额外重复 probe。需要固定四张物理卡时在第二条命令前加 `GPU_IDS=0,1,2,3`；默认自动选空闲四卡。前台执行便于看到首个错误，CPU 或探针拒绝时不会跳过继续训练。必须保持服务器原数据和完整模型路径可读，并满足至少 20 GiB 磁盘保留要求。

新运行目录：`checkpoints/phase3_base_repeat_20261024/run_base_repeat_20261024/`。

- 权重：`experiment/train/final/`
- 状态：`launch_workflow.json`
- 最终 test：`experiment/test/metrics.json`
- 交包：`handoff/phase3_report.zip` 与 `handoff/phase3_package.json`

同名输出拒绝覆盖；失败时保留原目录，根据首个错误诊断，不能将删目录重试视为修复。重新训练上述原生 base **不会使用本次候选提示词**；候选的现有权重配对效果尚未验证，本发布不提供伪装成新候选训练的命令。

## 本次发布验证

64 项相关 CPU 回归通过，含跨事件误报抵消反例、两种预测格式的 CLI 成功/复核退出码、以及 replay 导出到独立 CLI 的真实子进程衔接。推理编排使用模型替身，不代表 4B 前向。当前 320 val/1,280 图引用的计划重建、来源核验与 251/320 旧预测自对照通过。无新 GPU 推理/训练或新精度结论。
