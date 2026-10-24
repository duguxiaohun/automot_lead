# Base 训练与候选提示词配对测试

## 当前默认：回到上一轮 bounded_v3（2026-10-24 用户决定）

默认 base 固定为 `checkpoints/phase3_compare_base/run_base_bounded_v3/experiment/train/final/`：
val **251/320（78.4375%）**，test **267/380（70.2632%）**。
这是上一轮模型，不是历史四图 72.11% 的另一版本，也不是最新 `phase3_base_new_20261024/run_base_new_20261024`。
新训练、旧模型、审计证据全部保留，不覆盖或删除；默认路径不跟随 `latest` 或文件时间。

机器可读选择为本目录 `phase3_base_default.json`，由包外 `phase3_base_default.sh` / `.py` 读取和验证。统一 `phase3_compare.sh` 的 base 入口原本已经指向上一轮 canonical 路径，本次保留其源码和冻结合同，不需要重训或重建数据来完成选择回退。

在 AutoMoT 根目录查看当前选择（不需要模型/GPU）：

```bash
bash qwen3vl_local/audit_joint/phase3_base_default.sh status
```

在保存旧模型的服务器，核验旧交包 SHA、交包与运行目录中的完整 JSON/JSONL 记录一致、base 提示词/四图单选身份及原生训练完成/adapter 文件合同后输出所选 adapter 路径：

```bash
bash qwen3vl_local/audit_joint/phase3_base_default.sh adapter
```

`adapter` 成功时 stdout 只输出路径，诊断发到 stderr；验证前后复核交包及运行记录，任一改变拒绝。日志在旧导出器中经过截断，因此不与完整运行日志逐字比较。此命令调用原有兼容运行时验收，可准备其源码缓存，但不训练、不重新推理、不改模型、不写外层完成回执；缺旧交包或权重时拒绝，不回落到新模型。旧交包不包含完整权重，原生校验依据服务器保留的 adapter receipt，不冒称审计包本身提供了独立权重副本。

**base 标定和提示词改进暂时停止。** 原标定、original 提示词、四图单选和 bounded_v3 预算保持。下面的候选配对及重新训练命令仅保留为历史复现说明，不再作为当前建议；除非用户重新要求，不继续执行或改进 base 候选。

**base 与 single 是独立方法分支。** single 的 `phase3_single_v3`、`phase3_single_refine/frozen`、`phase3_single_refine/prompt.py`、各自数据/ready/权重保持原样。不得把 base 的窗口提示词/标定复制到 single，也不得把 single 的方向、历史填补、四未来查询或候选提示词反向合入 base；`single_refine` 当前候选状态不因本次 base 默认选择而改变。这里的“分支”指独立方法及源码路径，不执行整个 Git 工作树的 reset/revert。

本机缺上述服务器 adapter，仅已完成默认选择与代码隔离核验；不能把本机 status 的 `weights_verified=false` 描述成已经部署或加载服务器模型。旧质量门槛问题保留，用户选为默认不等于新的质量认证。

本次默认选择发布复核：38 项 CPU 测试通过（63.08 秒），含真实兼容子进程对合成完整 9,600 步记录及权重回执的验收、损坏权重拒绝、旧包与运行目录混用拒绝、single 身份拒绝和跨目录 shell 入口。477 个既有受保护文件散列未变。合成权重测试不等于服务器 4B 模型推理；本次未训练或重新评测。

## 以下为暂停前的实验说明

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
