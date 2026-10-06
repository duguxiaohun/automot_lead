# 第十二/十三轮审计后的修复

本次实施代码与风险准入修复，不新增目视审阅或人工标签。合同 v16/data_v16，标注仍 v7；保留 v15 的有限覆盖训练准入策略。状态、提示词、标定规则和冻结 holdout 协议不变，Phase3/Action 稳定默认不变。

## 恢复训练只使用该 checkpoint 当时的最佳模型

原问题是 `training_state.pt.best` 来自所选旧 epoch，而最佳模型路径取父 run 后来更新的 `best.json`。现在每个 checkpoint 同时保存最佳模型的相对路径、分数、所属 epoch 和推理资产 SHA；最佳 adapter 自身的 `selection.json` 保存其验证分数。恢复检查分数一致、epoch 早于 next_epoch、合同和实际文件身份，再加载模型/优化器。父目录后来的 best.json 不参与恢复。

使用真实 `train.main`、torch 优化器、保存和恢复编排的回归：原三轮分数 0.5/0.7/0.9，从 epoch_000 恢复，新分数 0.4 或 0.5 均选择原 epoch_000；0.6 选择新 epoch_001，不能选择原未来 epoch_002。删除父 run 的 best.json 后仍可恢复；从继承最佳模型的新 checkpoint 再恢复，以及整条历史相对目录搬迁均通过。缺旧格式记录、未来 epoch、错误分数/epoch、非有限分数、资产缺失/替换、合同/selection 错配均拒绝。

这组测试的模型、生成指标、adapter 验证为轻量替身，只证明训练编排，未完成 Qwen4B/GPU/DDP 效果或性能验收。恢复时仍需保存被引用的祖先最佳 checkpoint；单独搬迁一个包含外部相对引用的 epoch 不足以恢复。旧 v15 及更早数据/run 不自动兼容，须保留原源码；本次要重建数据并新建 run，不能改 hash 硬套旧合同。

## 将已报告风险接入审核

新增 `twelfth_thirteenth_audit_exposure_20260930.json`，绑定原审计报告、观察记录和定位 RGB SHA。49 个已审物理组保持 train-only，其中真正新增的隔离组为 `AccidentTwoWays/Town02_route_001609`。

两轮共 30 处消失记录、29 条路线，加上 3 条 ControlLoss 事件身份待审路线，共 33 条风险记录/32 条路线；风险清单均未与当前 v7 题库重叠。66 条定位 RGB 证据均核对文件 SHA，这是文件身份校验，不是本次新增目视审阅。保留原图更正的 86→87、32→33，Town12_262_2 两处异常分别登记。

新增路线复用现有 `risk_review`：每道题所有因果历史图均需身份、转移范围、实例边界逐帧审核；缺审核拒绝构建，uncertain/excluded 保持 UNKNOWN。不能从消失生成 YES，也不能用单个施工牌出画代替整组障碍和整车净空。同一路线多个 STOP 使用独立实例；RE3 每段重新判断间隙。已有 taxonomy/route_segments 能表达这些情况，本次没有新增状态或固定扩窗。

## 明确缺失正例，保留探索训练准入

`coverage.transition_support` 按 split/事件/分支/边列出 readiness 的 YES/NO；`training_admission.train_missing_readiness_yes/no` 明确给出训练缺额，catchup 不计入。当前重点缺额：

| readiness 转移 | YES | NO |
| --- | ---: | ---: |
| UE2 return（返回分支） | 0 | 11 |
| RE3 enter | 0 | 9 |
| UE7 proceed | 0 | 4 |

两模式各 train 563 / val 273 / test 186，另 405 待审；四份 JSONL 与 v15 逐字节一致。`trainable=true` 是已标注子集的数据准入；`complete_coverage=false`，仍缺 366 个转移格、12 个 RE3 分段格、12 个事件格。严格完整覆盖模式继续拒绝。不能把可运行训练解释为已学会上述没有正例的转移或十事件闭环验收。

## 验证与剩余边界

- 414 项 Phase4 专项测试通过（原 399 项加 15 项回归）。
- 2/4RGB 实际 `run_full_pipeline MODE=check` 通过；每模式 1022 题及 405 待审题 RGB 身份核验通过。
- 每模式 14 个整池采样计划（7 epoch × world_size 1/4）、563 轮 cap1/budget1 覆盖全部训练题通过。该 world_size 检查是采样计划验证，不是实际 DDP 训练。
- 两模式数据 preflight 通过；未提供模型时 `ready=null`、`model_checked=false`，不能据此声称训练主机验收。
- 验证摘要见 `audit_fixes_v16_verification.json`；构建和测试日志在 `/tmp/phase4_resume_fixes_20260930/`。diff 空白和 shell 语法检查通过。

继承第十三轮审计统计：379 条/44204 帧，165/278 格三 ID，尚余 321 条/54464 帧，18 格源不足。本次没有增加这些视觉审阅计数，完整审计仍未完成。正式训练仍需训练主机完整基座/依赖/RGB/GPU 预检；本次未启动正式训练，也未完成 CARLA 或 Phase3 全部功能对等验收。
