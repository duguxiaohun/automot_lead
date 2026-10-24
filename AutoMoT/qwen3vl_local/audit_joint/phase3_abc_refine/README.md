# ABC 提示词小幅改进实验

独立版本 `abc_endpoint_clarity_v1`，保持旧 abc v2 标签、阈值、成员和三轮预算。只在 a 的输入中明确给出每个未来时距的内外半径；b/c 文本不变。半径仅来自当前观测和既有标定，不读取未来真值。不添加“当前静止则未来静止”或强制单调约束。

当前新增的训练集 RGB 审阅覆盖四个物理组、各 17 帧，共 68 帧，证据在 `checkpoints/abc_implementation_20261024/rgb_review`。包含近零速度、短暂前挪和后续起步；不足以支持修改阈值。候选提示词仍需同 adapter 成对推理验证，不能宣称已经提升精度。

## 使用

从 `AutoMoT` 根目录执行。`--baseline-dataset` 应取当前 abc v2 `ready_1280.json` 的 `dataset`；`--baseline-runs-root` 指包含原 `a_1280/b_1280/c_1280` 的目录，各组必须保留 `epoch_3/state.pt`、receipt、training_complete 和完整 test/cases。收到的轻量审计包通常没有 adapter，不能代替远程原 run。

```bash
# 独立 CPU：验证父库、保持标签、三轮计划、逐例完整成员
bash qwen3vl_local/audit_joint/phase3_abc_refine/run.sh prepare \
  --baseline-dataset <原abc_v2数据集绝对路径> \
  --output checkpoints/abc_refine_v1_cpu/data

# 远程具备原 adapter 和完整模型后：自动完成整个实验
bash qwen3vl_local/audit_joint/phase3_abc_refine/run.sh all \
  --baseline-dataset <原abc_v2数据集绝对路径> \
  --baseline-runs-root <原a_1280_b_1280_c_1280所在目录> \
  --model-dir checkpoints/Qwen3.5-4B \
  --output checkpoints/abc_refine_v1
```

`PYTHON` 可指定解释器，`GPU_IDS` 可指定四卡。新运行建立 `run_<tag>`/`latest`。顺序为：完整旧 test 成员预检 → 26MB 打包演练 → 同一原 adapter 全 val 原/新文本各 fresh KV 重放 → 每个旧正确样本保护 → 实际 processor 全池检查 → 三个独立四卡探针 → a/b/c 各三轮训练、最终第三轮完整 test → 回归审计 → 自动打包。训练仍是每轮 12,800 次呈现/3,200 更新，共 9,600 步。

`replay` 可单独执行，参数为 `--baseline-dataset --variant a --checkpoint <a_1280/epoch_3> --model-dir --output`，另可传 `--received-cases` 报告旧收到结果与本次原文本重放的差异。主比较始终为本次同模型成对推理，不能用不同机器旧结果差异冒充提示词收益。

任何旧正确损失、分层 precision/recall/F1 下降或物理组宏平均下降，都会标记 `tradeoff_review_required`，退出 2；即使净准确率提高也不能自动发布为保护通过。val 回归不通过时停止长训练，保留成对案例供复核。训练后 test 发现损失时仍自动交付完整结果包并退出 2，便于审查。

最终主包 `<run>/delivery/abc_refine_audit.zip` 不超过 30,000,000 字节，包含全成员、全部 test 紧凑案例、指标/回归与训练完成证明；完整轨迹无关，本版本是九分类。模型权重和 RGB 保留在本地/服务器原目录。打印包目录、路径、大小及 SHA256，失败另有诊断包。已完整训练只需补包时：

```bash
bash qwen3vl_local/audit_joint/phase3_abc_refine/run.sh pack \
  --run-dir <已有run目录> --output <已有run目录>
```

重跑 `all --run-dir` 会核验已有证据；部分训练不能隐式续训。旧源码未改，运行适配仅发生在独立进程内，退出后恢复原函数引用。此版本不接入路线二，不能覆盖旧权重或旧 ready。

## 2026-10-24 独立审计后的修复

正式打包的指标重算保持新合同上下文，错误合同仍拒绝。precision/F1 从 0 变为 N/A（消除了最后一个误报）不再当成下降；任意旧正确损失、实际有定义的指标下降及组宏平均下降仍需复核。

suite、成对重放、全池 processor、每个 a/b/c probe 均绑定当前基础模型目录内的权重、config、tokenizer 和 processor 文件内容。原 adapter 的模型绑定必须相同。probe 还绑定表示、processor 回执和保存的 checkpoint/receipt，执行原生权重重载并逐参数比较；短探针包含各表示实际最长的训练输入，val/test 极值仅前向。替换模型、交换 probe 或修改保存权重后不能沿用旧成功回执开长训练。

正式主包现在包含完整成对 val 的紧凑原/新预测、回归报告、processor 及三组 probe/保存回执。外部 `--replay-root` 的绝对来源与文件 SHA 冻结到 suite，打包重新核验。全部训练前核心证据齐全后再做一次 26MB 演练，仍预留 4MB 给正式结果回执。单独 `pack` 需要原基线数据、模型文件、adapter 和外部重放路径可读，以便重验；不会加载大模型推理。

修复改变了运行源码指纹，修复前准备的 refine 数据和 processor/probe 回执不直接兼容。保留旧目录，重新 `prepare` 或使用新输出目录运行 `all`；禁止手工修改旧 SHA。最终本地重验记录见 `checkpoints/abc_review_fixes_20261024`。68 帧训练 RGB 审阅仍只有四个 AccidentTwoWays 物理组，未补成全场景边界盲审，也不构成模型正确保护已通过。

## 第二轮审计：探针训练与留出集隔离

完整 processor 扫描分别记录 a/b/c × train/val/test 的文本和视觉网格极值。四卡反向探针只选训练集最大文本及最大视觉网格样本，去重后按稳定 ID 补足四条；任何非 train 成员在调用原生 train_epoch 前拒绝。val/test 极值单独 eval + no_grad 前向，使用未包 DDP 的模型以避免各 rank 不等量前向触发集体通信；没有 loss/backward/optimizer 更新。实际 pack 的文本长度和视觉网格必须匹配全池报告。前向检查后恢复训练模式，随后保存和重载独立探针权重。

probe_complete 绑定四 rank 回执，记录反向成员及 holdout 前向成员/零更新数；正式开训及打包从实际数据重算选择并验证回执，主包收齐这些证据。原提示词、标定和三轮预算未改。由于独立 refine 源码绑定更新，需使用新目录重新 prepare；旧目录保留。当前验证记录在 `checkpoints/abc_coverage_probe_fixes_20261024`，本轮未实测完整模型 processor、原 adapter 成对重放或四卡 GPU 探针。
