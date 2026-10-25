# 运行版本隔离与旧准入恢复

## 本次故障

Action run_20261024_232223 的 admission 保存旧 preflight.py/worker.py，processor 保存更新后版本。四卡初始化后才发现差异，尚未开始第一个 a_bev_only 探针更新。NCCL 初始化成功，底层原因是同一准备流程混入两版代码，不是 GPU 通信或训练 loss。

仅改版本号或用当前 SHA 覆盖旧回执不正确。本轮引入本机内容寻址源码快照：入口运行前复制源码/配置，校验复制期间来源不变，随后整个 shell、延迟导入和 Python/torchrun 子进程在同一快照下运行。更新主代码目录不再改变已启动任务的源码。新版本自然生成新目录，旧版本不覆盖、不自动删除。模型、数据、输出通过明确链接复用；数据/权重本身仍由各阶段原有校验保护，不能在任务中改写。Python 环境沿用当前激活环境，此功能不隔离 pip/Conda 环境。

## 全目录适用方式

本轮逐个清点 qwen3vl_local 下 122 个 shell 文件，语法检查并核对快照字节。通用包装器支持该目录的 shell、Python 脚本与 `-m` 模块。并非 122 个入口都做了真实 GPU 训练验收。

以下公共入口自动启用版本隔离：

- action_expert_ablation/geometry_abc/run.sh
- audit_joint/phase3_abc_refine/run.sh
- audit_joint/phase3_compare.sh（执行操作；只读 status/help 保留轻量入口）
- audit_joint/geometry_study_abc.sh（同上）
- run_with_env.sh 改为固定版本后运行目标脚本。

其它历史绑定脚本保留原始字节，通过外部包装器运行。直接绕过包装器执行旧 Python/旧 shell 不会自动获得版本隔离。这样避免为改启动方式而让历史数据、权重与原生源码回执全部失效。

```bash
# Phase1 与 Phase2 分别独立执行，不互为前置，不传递权重。
bash qwen3vl_local/run_versioned.sh -- qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
bash qwen3vl_local/run_versioned.sh -- qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh

# 其它既有命令只需加同一个包装器；保留原参数、原阶段准入要求。
bash qwen3vl_local/run_versioned.sh -- qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
bash qwen3vl_local/run_versioned.sh -- qwen3vl_local/sft_new_loop_phase4/run_full_pipeline.sh
# Python 模块：
bash qwen3vl_local/run_versioned.sh -- -m qwen3vl_local.audit_joint.phase3_compare status abc
```

以上为包装方式示例，不代表本轮授权启动这些实验；各阶段原有容量准入仍须通过，base/single 暂停的新实验仍不启动。训练预算、prompt、标定、采样和原生阶段代码未改。自定义输入/输出推荐用绝对路径；默认 checkpoints/lead_data 等保持原项目目录。输出应放共享 checkpoints 或显式绝对目录，不写入冻结的源码/配置目录。

入口打印版本目录与执行记录：`checkpoints/runtime_versions/<SHA>/manifest.json`、`checkpoints/runtime_executions/*.json`。后续评测、补包或原生允许的恢复可显式选择相同版本：

```bash
bash qwen3vl_local/run_versioned.sh --version <上次打印的版本目录> -- <原脚本及原参数>
```

Action 新运行还保存 execution_version.json；其 run.sh 带 --run-dir 时自动读取并验证原版本。失败包和正式包包含版本/恢复摘要。不改已完成结果或给旧 checkpoint 自动升级合同。已经在主树启动的旧进程无法事后转成源码快照；须使用保留的原运行版本及原生恢复规则处理，不能继续热更新它。

## 本次 Action 恢复

--recover-from 只接受尚无六组正式训练目录的旧运行，创建新 run，旧目录、失败包、回执不改。重新核对完整原始数据、历史、七轮计划及模型/BEV 资源；数据构建代码未改，既有合格数据可继续使用。若旧数据的构建代码合同确实不兼容，则在独立 `_version_<SHA>` 数据目录重新生成，不覆盖旧库；本次两个文件的更新没有改变数据构建合同。旧 processor 的 SHA、完整成员数、表示、split、极值及数据/模型绑定均须通过。

只有 processor_compatibility.json 中逐文件精确列出的已审查 SHA 变更才允许复用。本次两版 preflight/worker 差异是旧权重拒绝与版本检查/编排，不改 observation_text、图像 packing、极值算法。新增回执保存旧报告 SHA、兼容策略 SHA 和变更清单；旧报告仍原样。未知源码变更、模型变化、数据变化、报告不完整时自动重新执行 processor 检查，不以改 SHA 放行。不沿用旧 probe，全部六组探针重新执行；已有正式训练产物时拒绝准备阶段恢复，避免悄悄重训或改变看过的数据量。

从本次服务器目录执行；先确认没有仍使用主树的旧任务，再同步代码：

```bash
cd /mnt/workspace/cee3sgh/AutoMoT
bash ./sync_automot.sh
ABC_ACTION_MAPS=$(python -c 'from pathlib import Path; from qwen3vl_local.audit_joint.phase3_maps import resolve; print(resolve(Path.cwd(), Path("checkpoints/abc_action_maps_20261025"))[0])')
bash qwen3vl_local/action_expert_ablation/geometry_abc/run.sh all \
  --recover-from checkpoints/action_expert_ablation/geometry_abc/run_20261024_232223 \
  --maps "$ABC_ACTION_MAPS" \
  --data-root lead_data \
  --data-dir checkpoints/abc_trajectory_v1 \
  --model-dir checkpoints/Qwen3.5-4B \
  --lead-bev-ckpt checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth \
  --output checkpoints/action_expert_ablation/geometry_abc_recovered
```

日志 `processor_reused` 表示复用了完整检查；`recompute_required` 会列原因并自动重做 processor。不会重新标定或更改七轮配额。四卡之前先对照源码与 processor，差异直接列文件并进入失败包；worker 也在 runner/NCCL 初始化之前重验。

## 验证边界

本机实际源码树完成快照与旧方法合同一致性核对；测试包含运行途中热改主树、延迟导入/子进程隔离、显式旧版本复用、快照内容/新增文件篡改拒绝、仅转发本任务信号、旧目录保留、兼容/不兼容恢复和六组完整编排。源码捕获连续变化最多重试三次，始终在长工作开始前处理。真实大模型、四卡 GPU、远程旧目录恢复没有在本机执行，不能保证所有历史脚本的外部资产和环境都已满足。

证据位于 checkpoints/runtime_version_fix_20261025。原始数据和模型不随源码快照上传，快照仅存本机；最终审计交包仍硬限 30MB，不把整份源码快照或权重塞入主包。

本轮最终 355 项不同 CPU 测试通过（完整批次 351 项，最后相关复测 111 项，按测试名称去重）；122 个 shell 语法及快照字节检查通过。真实 Action 数据 843,059 条的 train/val/test 文件散列重新核对不变，数据构建源码及既有七轮计划散列不变；本轮没有重新扫描全部原始传感器，恢复入口会重新执行该检查。

## 2026-10-25 固定依赖路径兼容修复

远程恢复 run_20261025_155610 在数据准入、GPU 初始化之前失败。旧 snapshot 把 keyframe_filter 整目录链接回主目录，原生 Phase3 stable 的 verify_release 同时要求依赖 SHA 正确且 resolve() 后位于本项目 ROOT 内；链接跳出了快照 ROOT，因而即使字节正确也报 `Phase3 stable dependency changed`。本机原目录通过、旧快照同位置失败，已复现。上一轮只做 CLI help/status 与源码合同核验，没有覆盖这条真实延迟导入链，是验收遗漏。

新版按已存在的 frozen release.json 外部依赖清单，将审计 JSON、JSONL 及辅助源码逐文件复制到快照内部；仅展开必要父目录，其它数据/输出仍共享。原生 release、expected SHA、背景确认规则、Action 数据构建代码、prompt、标定、划分和训练预算均不改。真实内容不匹配仍由原生校验拒绝，不更新旧回执掩盖变化。旧失败快照和运行保留。

同步后重新执行上面的 all --recover-from 原始 run_20261024_232223 命令即可；创建新的执行版本与新运行。不要用 --version 指向失败的 4e4f5e16... 快照，也不要用 --run-dir 继续这个准备失败的目录，否则会按设计固定回有问题的旧执行版本。仍从原始 run 恢复，才能尝试复用其中完整的 processor 证据。

本轮回归 156 项通过；额外一个历史测试 test_historical_v23_matches_restored_phase3 仍失败，该测试要求冻结 v23 与当前开发 Phase3 的 mapping hash 相同。装入修复前 b91758a48f 的运行器也同样失败，属于已有的跨版本等同假设；本轮保留测试与两个方法，未修改它使之通过。新增测试实际调用原生 verify_release，覆盖目录内依赖、JSONL、缺失/篡改/路径越界拒绝与共享数据保留。证据 checkpoints/runtime_dependency_fix_20261025；不代表真实四卡训练验收。

真实全量候选扫描：原目录与新快照均得到 7,241 条路线、901,221 个事件/背景候选锚点，逐条顺序/字段摘要、事件计数及 169 份数据源码/324 份运行源码身份完全相同。这里是过滤前候选，不是 843,059 条最终有效训练库；本轮未重建训练库或重新解码全部 RGB/LiDAR。
