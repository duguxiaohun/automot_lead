# ABC 提示词小幅改进实验：从基座新建 LoRA

2026-10-25 起流程为 `abc_fresh_lora_posttraining_regression_v2`。提示词版本仍为 `abc_endpoint_clarity_v1`：只在 a 输入明确各未来时距的内外半径，b/c、九区域标定、数据划分和训练预算保持不变。半径只来自当前观测，不读未来真值，不添加“静止必须继续静止”的限制。

每组从原始 Qwen3.5-4B 新建 LoRA。**不加载原 Phase3 adapter，不把旧 adapter 换提示词重放作为开训条件**。旧模型的完整 test JSON 仅用于成员预检和训练后的逐例回归审计。独立探针的权重只用于其自身保存重载验证，不用来初始化正式训练；正式评测只加载本轮最终第三轮权重。

## 命令

从 AutoMoT 根目录执行，先同步修复代码、激活原 Python 环境。父数据路径取旧 abc v2 `ready_1280.json` 的 `dataset`。`--baseline-cases-root` 下只需三份原始完整 JSON：`a_1280/test/cases.json`、`b_1280/test/cases.json`、`c_1280/test/cases.json`，不需要旧权重。旧参数 `--baseline-runs-root` 仍兼容，但也只读 JSON。

```bash
ABC_BASE_DATASET=$(python -c 'import json; from pathlib import Path; print(json.loads(Path("checkpoints/geometry_abc_v2/ready_1280.json").read_text())["dataset"])')

bash qwen3vl_local/audit_joint/phase3_abc_refine/run.sh all \
  --baseline-dataset "$ABC_BASE_DATASET" \
  --baseline-cases-root checkpoints/geometry_abc_v2/runs \
  --model-dir checkpoints/Qwen3.5-4B \
  --output checkpoints/abc_refine_fresh_v2
```

`PYTHON` 可指定解释器，`GPU_IDS` 可指定四卡。all 创建新 run：完整父库/来源及三轮计划 → 旧 test JSON 成员预检 → 26MB 容量演练 → 实际 processor 全池检查 → 三组独立四卡探针 → 三组新 LoRA 训练 → 第三轮完整 test → 回归审计 → 自动打包。每组每轮 12,800 呈现、3,200 更新，共三轮 9,600 更新。各组均从基座重新初始化。

可单独先做 CPU 准备，正式 all 仍准备其自身新目录：

```bash
bash qwen3vl_local/audit_joint/phase3_abc_refine/run.sh prepare \
  --baseline-dataset "$ABC_BASE_DATASET" \
  --output checkpoints/abc_refine_fresh_v2_cpu/data
```

## 失败、质量复核与交付

远程旧 `run_20261024_232204` 在 a 的旧 adapter 重放阶段退出 2，尚未进入正式训练。收到的 739 条 val 重放证明换提示词影响旧 adapter，但不能推断重新训练后的效果。旧 `replay` 命令已明确拒绝，`--replay-root` 不再接受。旧目录和原包保留；**不要向新 all 传旧失败目录的 `--run-dir`，不要改旧 SHA 或复用旧 processor/probe**。

训练后对照完整旧 test JSON。任意旧正确端点损失、实际有定义的分层 precision/recall/F1 或组宏平均下降仍标记 `tradeoff_review_required`。三组训练与结果打包会全部完成，然后退出 2；不把此质量状态伪装成运行异常或保护通过。退出 0 表示本次观察队列保护通过，不代表所有场景无退步或闭环安全。

运行异常仍失败并保存诊断包；其中含 pipeline、已有回归/准入/探针摘要及各阶段日志末 64KB，历史重放 JSON 存在时也会收录。失败包的 `core_complete` 仅表示已选择诊断成员齐全，`final_result:false`，不表示训练完成。所有打包均禁止权重/RGB，主包硬限 30,000,000 字节。

正式包 `<run>/delivery/abc_refine_audit.zip` 包含完整成员、全部 test 新旧紧凑预测、重算指标/逐例回归、训练完成证据、processor、三组 probe 和四 rank 回执。程序打印结果目录、包路径、大小、SHA。已完成本版本运行可单独补包：

```bash
bash qwen3vl_local/audit_joint/phase3_abc_refine/run.sh pack \
  --run-dir <本版本已有run目录> --output <本版本已有run目录>
```

补包仍需原基线 JSON/数据、当前基础模型文件和本轮训练/探针产物可读，重验散列但不加载大模型。all 对本版本完整已有证据可重验；部分训练仍不隐式续训。

## 保留的验证约束

processor 分 a/b/c 和 train/val/test 记录文本及视觉网格极值；反向探针只选训练集极值并补足四条。val/test 极值只 eval/no_grad 前向，零优化更新。真实输入形状必须匹配 processor 记录。probe 保存后原生重载并逐参数核验，回执绑定模型、表示、数据、源码和 processor；正式开训及打包重新核验。原生训练代码和旧运行不改。

RGB 人工审阅仍只有四个训练物理组共 68 帧，不是全场景边界盲审。本机 CPU 验证和合成打包不能替代真实模型 processor、四卡探针、长训或精度验收；重新训练的效果尚待实测。

## 运行期间源码固定

run.sh 自动在本地源码版本中执行整条流程，后续同步主树不会影响已启动的新任务。其它阶段的外部包装方式及显式原版本补包见 [运行版本说明](../../RUNTIME_VERSIONING_20261025.md)。此改动不迁移旧训练权重、数据或 processor/probe 合同。
