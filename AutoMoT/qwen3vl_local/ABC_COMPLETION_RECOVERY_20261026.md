# ABC 已完成训练在收尾报错：原六组接续

## 2026-10-27：兼容 cr02 的原 BEV 三组

已核验新诊断包 `diagnostic(2).zip` 的 32 个成员，包 SHA256 为 `6dc42e47ded325901dd3de6209420b1290ef3c539828ff8752dad605769f62f0`。cr02 运行是 `geometry_abc_bev_only/run_20261026_122636`，固定源码 `03d54a7b338d815c80a2b92a9777dc9f8d000d27f81e6bafb852615649a32c1b`。四 rank 均已到第 7 轮 / 12719 步；320 条最终 val、两次 checkpoint 保存和 training_audit 已完成，随后同样在性能 all_gather 使用已销毁的进程组时报错。内存回执的 oom / oom_kill 前后均为 0。权重本体不在诊断包中，仍须在服务器严格验收。

本次仅扩展包外恢复入口，训练核心、worker、数据、提示词、标定、划分和预算均不改。入口按已审查的 core / worker / audit / group_scope 精确源码组合选择兼容路径，读取原 admission 的 selected_groups；该 cr02 运行只能继续 a_bev_only / b_bev_only / c_bev_only，不启动或要求 Qwen 组探针。a 完成核验后直接测试，b/c 各按原预算新训、测试，原生审计重算和 30MB 自动打包保持。其余验收要求见下文。

在 cr02 已激活 automot 环境下执行：

```bash
cd /home/cruser1/lda/AutoMoT
bash ./sync_automot.sh && \
python -B qwen3vl_local/abc_completion_recovery.py \
  --run-dir checkpoints/action_expert_ablation/geometry_abc_bev_only/run_20261026_122636
```

可加 `--check-only` 先做 CPU 验收。数据和模型路径读取这台机器原配置，包括实际 BEV 权重路径；不复制另一台机器的权重或准入。不要重跑普通 all / --recover-from。新状态在原 run 的 `completion_recovery/status.json`；失败时保留原诊断并另写 `completion_recovery/failure_delivery/diagnostic.zip`。

另一台服务器的六组接续命令保持下文所列。若它已经安装 2026-10-26 发布的恢复 runner，入口核验该文件已知 SHA 及原 supplement 后继续使用它，不替换正在运行的版本；未知文件或变化的绑定仍拒绝。新恢复目录按原 admission 自动确定六组或单分支三组，不能用 CLI 偷换范围。源码未知、训练实际未完成、缺权重或消费证据时不会伪造通过。

以下为 2026-10-26 首次六组故障和恢复协议记录；其中“原六组”描述该次六组运行。本次 cr02 采用同等完成验收，仅按自己的三组准入接续。

收到 `diagnostic(1).zip` 对应 `geometry_abc_memory_fixed/run_20261026_100720`，固定源码版本 `16915828325dfb81b7f8bfff24d3f3886d7cc7a7eff12848a27564740fee27cf`。实际命令未指定 branch，用户确认继续原六组。

完整日志显示 a_bev_only 已完成 7/7 轮、12719/12719 次更新、每 rank 203448 次呈现。最终 320 条验证于 18:32:52 完成，日志 val loss=0.0149121132、route ADE=0.1239496m、waypoint ADE=0.3404772m。18:33:03/06 完成 best/latest 保存，约各 1.8GB；训练审计包生成后四 rank 都在 worker 的性能 all_gather_object 报“Default process group has not been initialized”。这些是日志证据，不替代实际权重/消费验收，也不是 test 成绩。

原因是共享 training_core.run_training_loop 的 finally 销毁进程组，而 ABC worker 返回后仍需汇总性能并写 training_complete。训练前后 oom_kill 都为 1，没有本次新增 OOM 计数；此故障有明确代码栈，不按 OOM 处理。8500 步只是先前贴出的日志片段，完整日志已修正该判断。

## 新任务修复

共享循环新增 destroy_process_group 参数，默认 True 保留旧语义 Action 行为；ABC 显式 False，由它自己的 finally 在性能汇总和完成回执之后清理。优化器、实际数值循环、标签、提示词、数据划分、采样及预算不变。精确 processor 兼容表只新增这两份生命周期改动。

## 既有任务恢复

`abc_completion_recovery.py` 是独立、记录 SHA 的补充执行程序，只接受上面已审查的旧 worker/core 源码组合。它复用原服务器上的固定快照，逐字验证快照/原 admission；不修改原快照、源码 SHA、原 admission、processor、探针、权重或失败 pipeline/diagnostic。

1. 复核同机资源、完整 split SHA、路线回执、隔离与开发约束、原计划和六组已完成探针。只允许原六组，不在恢复时更改实验范围。
2. 对每个已有训练目录先校验：checkpoint schema/配置/来源/世界大小、12719 更新且 cursor epoch7/micro0 且无待验证、最终 val、原验证选 best、七轮逐例消费与来源证明、四 rank 最终梯度/槽位曝光和 best/latest 散列。
3. 在临时只读视图构造候选完成回执并执行原生 audit.verify_training；前后输入散列一致后才补写缺失 training_complete.json。已有不一致回执拒绝，不覆盖。检查不完整的既有训练目录会在启动任何新组前停止，不将其当新组重训。
4. a 不调用 optimizer 或重新训练，直接执行原 best 的测试/条件诊断；未开始的五组继续原六组次序，各自新 decoder 初始化。运行补充仅把原公共循环的进程组销毁推迟到原 worker 的 finally；原 _run_training_loop 不变。补充代码、SHA、原版本、补验收输入清单均单独留证，不伪称未使用补充代码。
5. 最终使用原生 audit 重算并自动交包，加入 completion_recovery 证据，30MB 上限不变；条件诊断需复核时保留原 review 状态。原始失败记录保留，新状态看 completion_recovery/status.json。恢复失败包在 completion_recovery/failure_delivery/diagnostic.zip。

a 缺失的各 rank 性能汇总不编造。保留原训练会话计时及已有指标，补充报告明确该缺项。未完成的测试输出在重试前保留到独立 test_incomplete_before_recovery_*，不覆盖失败证据。若已开始的新训练组再次中断且预算未完成，此工具不会自动恢复优化器；届时要根据实际 checkpoint 另行审查。

## 两台服务器命令

先进入本次运行实际所在服务器的 AutoMoT 根目录，使用已激活环境。不能将另一台服务器的运行目录/快照视作本机存在，也不能用 `--recover-from` 代替下面的完成恢复。此次目录在 `/mnt/workspace/cee3sgh/AutoMoT`。

```bash
bash ./sync_automot.sh && \
python -B qwen3vl_local/abc_completion_recovery.py \
  --run-dir checkpoints/action_expert_ablation/geometry_abc_memory_fixed/run_20261026_100720
```

可选只检查（不补写 training_complete、不训练或测试；会写独立检查报告）：

```bash
python -B qwen3vl_local/abc_completion_recovery.py \
  --run-dir checkpoints/action_expert_ablation/geometry_abc_memory_fixed/run_20261026_100720 \
  --check-only
```

路径、模型、数据参数自动读取原 config。不是从旧 Action/Phase3 权重初始化新实验；a 的权重仅用于这个已完成原实验的验收和评测。不要再运行普通 all 从零启动，也不要删旧 run 或手改 frozen SHA。

## 验证范围

本地相关 CPU 回归包括真实四进程 Gloo 的完整小模型训练循环、最终验证、原生保存与返回后 all_gather；新生命周期与旧任务补充均成功，旧默认提前销毁对照复现该错误。完成验收通过原生预算/消费/权重检查，覆盖篡改与未完成拒绝、检查不写完成标记、重复运行幂等、a 不重训且后五组依次训练/测试、打包补充证据、两种服务器路径。模拟模型/数据仅验证功能，不认证实际4B/GPU或服务器 checkpoint。

169 份数据构建来源逐字不变；共享 core 只有 run_training_loop 生命周期包装改变，内部数值循环 AST 与原发布相同。证据 checkpoints/action_completion_fix_20261026。实际服务器权重和七轮完整产物仍由上面入口现场校验；本机只有诊断包，不能提前宣称已补验收完成或六组训练成功。
