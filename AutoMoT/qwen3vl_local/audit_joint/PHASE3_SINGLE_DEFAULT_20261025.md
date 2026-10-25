# Phase3 single 默认选择：上一版 single v3

2026-10-25 用户决定：暂停新的 single 审计和改进，保留 single_refine，默认使用上一版 single v3。这个选择不是新的闭环效果认证。

| 项目 | 默认 |
|---|---|
| 标定、历史、提示词、loss、训练与测试 | `phase3_single_v3` 原实现及其原生依赖 |
| 默认运行 | `checkpoints/phase3_single_v3/runs/six_1280` |
| 选定模型 | `epoch_3/state.pt`，最终第三轮，9600 次更新 |
| 旧版测试纯正确 / 有效集合命中 | 5768/6707；6092/7077 |
| 保留的候选 | `phase3_single_refine` 与 `checkpoints/phase3_single_refine_train_v1` |

统一 `phase3_compare.sh ... single` 本来就指向 v3，本次不改动已绑定的标定、prompt、训练/测试代码、ready、采样或合同，不把 refine 的澄清文本写回旧版。base、abc、Phase4 的选择保持原状。没有删除、移动或覆盖两版数据、结果、权重；本机没有服务器 state.pt，不声称已经切换远程闭环 agent。

新增机器可读 `phase3_single_default.json` 固定版本、路径、旧运行的18份JSON证据SHA及第三轮权重SHA。不依赖 latest/mtime，不因 refine 更新自动提升，也不自动重启训练。`phase3_single_default` 是包外选择器，因此不改变两版源码合同。

查看选择（只读，不需要模型、GPU或 .venv）：

```bash
bash qwen3vl_local/audit_joint/phase3_single_default.sh status
```

在保有旧运行的服务器上取得已核验的 checkpoint 目录：

```bash
bash qwen3vl_local/audit_joint/phase3_single_default.sh checkpoint
```

后者先核验选定旧运行证据和权重，再调用原生完成验收检查源码、ready、三轮权重、训练曝光、测试成员和重算指标；验收前后复核选定证据。缺失或错配即报错，不回落到 refine，不写新完成回执、不加载模型或运行推理。返回的是 checkpoint 目录，内含 state.pt；并非通用 Hugging Face adapter 目录，必须配旧版模型/提示词接口使用。

今后明确要重新训练上一版时，沿用原入口（仅对尚未存在的运行目录）：

```bash
bash qwen3vl_local/audit_joint/phase3_compare.sh cpu single
bash qwen3vl_local/audit_joint/phase3_compare.sh check single
bash qwen3vl_local/audit_joint/phase3_compare.sh train single
```

`train single` 包含三轮训练、最终第三轮 test 和完成验收；不是 refine。已有完成的 six_1280 不要重新执行 train，它会按原规则拒绝覆盖。已有结果只需 `phase3_compare.sh verify single` 做保存结果验收，无需重新训练或测试。新训练产物即使路径相同也不自动替代本次绑定的历史权重，变更默认需另行明确决定。

`phase3_single_refine/run.sh` 保留显式实验含义，不将其偷偷重定向到 v3；文档内旧训练命令仅供历史复现。后续除非用户重新要求，不继续 refine 的标定/提示词审计、实验或训练。shell 继承显式 PYTHON、已激活 virtualenv/Conda，之后才查项目 .venv 与 PATH。
