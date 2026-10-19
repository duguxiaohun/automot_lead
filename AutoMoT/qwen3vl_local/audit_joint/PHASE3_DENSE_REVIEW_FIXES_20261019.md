# Phase3 base / dense v2：审查修复与验收边界

> 本页保留为 v2 历史说明。当前启动及验收以 [v3 第二轮审查修复](PHASE3_DENSE_V3_20261019.md) 为准。

本页取代 `PHASE3_BASE_DENSE_20261019.md` 中的 v1 实验说明与启动建议。恢复 base 的选题、提示词、标定、固定评测和 7680 更新预算未改；密集目标升级为 `phase3_dense_external_labels_v2`。这是实现及本地验证记录，不是 H20 训练通过或效果恢复证明。历史 75.625% val / 72.105% test 是固定题库动作准确率，不是闭环驾驶成功率。

## 审查问题对应的修改

1. **窗口过滤与动作支持**：先在未过滤的历史 base 训练池上计算独立 event/action 硬配额，再从可构成窗口的候选中填同一配额。任何原池支持的 event/action 被过滤为空立即拒绝，不能用 KEEP 替代 STOP。无需少见题、连续片段优先。实际窗口候选数、独立锚点数、重复呈现数均可从计划审计核查。
2. **辅助标签**：`--index` 仍必须等于冻结历史 SHA，固定 val/test 不变；`--aux-index` 是独立原生索引，须带 `manifest.json` 并通过完整原生规则/提示词/物理路线隔离检查，index 和 manifest SHA 独立绑定。只读其 train 行，且必须属于 base 已有 train 路线/物理组；同帧观察/标签冲突拒绝。辅助行仅补精确时点的监督或历史窗口，不成为新的训练锚点，不修改 base 或评测成员。没有辅助标签时仍可运行，但覆盖不足会明确阻塞；不自动重建全库，不插值，不补 KEEP。
3. **覆盖与实际消费**：future 每个事件、每个非零时点默认有效标签比例至少 0.8，可在 `[0.5,1]` 配置；single 对照也必须通过相同门禁。预检对三轮剩余计划逐轮检查，接续从 `resume_epoch` 开始；字段明确为 `planned_*`，不再叫 actual。每个成功 backward 的 batch 保存 query/权重，整轮 optimizer 更新结束后四 rank 汇总实际 query 数及 loss 权重质量，核对计划并生成 `query_consumed_epoch_*.json`。中断 batch 日志不等于整轮更新完成。
4. **时间定义**：`--forecast-step-seconds auto` 从正在使用的原生 `trajectory_action.py` 读取 6/8/12 帧窗口和 0.25 秒帧间隔，取最长窗口 3 秒；默认跨度 6 秒，所以 N=3，位置为 0/+3/+6 秒。`--forecast-span-seconds` 是最后一个预测锚点，必须能被间隔整除，最多 8 个位置。可显式设间隔 1 秒，报告会标记原生窗口重叠。输出仍是各时点的未来窗口主要动作；不能把它们当成依次执行的、固定时长的 action token，也不去重/重标同一次变道。
5. **方案二语义**：`stream` 是一次八图、五读出，共享因果语言前缀，可见历史依次 4/5/6/7/8 帧；`windows` 是五个严格四帧语言上下文并行，只复用冻结视觉计算。两者不混称“严格滑窗且共享语言状态”。Qwen3.5 的 DeltaNet 递归状态不能仅靠普通 attention mask 清除；本实现没有引入未经验证的递归状态减除算法。[官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)
6. **匹配对照**：每条路线支持 `--supervision single` 与 `all`。使用同一已完成 base adapter、同一辅助标签、时间配置、seed、三轮额外 7680 更新、候选/锚点顺序、提示词、图片与读出位置；single 仅监督 future 的当前位置，或 stream/windows 的最后位置。`input_plan_sha256` 应逐 epoch 相同。输入中从不填真实答案。对照完成后才讨论“新增监督的收益”，不能只拿密集路线与少训练 7680 步的 base 相比。
7. **资源准入**：CPU 读取全部剩余计划中的不同锚点和图片尺寸，按实际 processor 展开 token，审计每条语言序列 ≤8192，整个 padded batch 默认 ≤16384（可显式设 8192..32768）。windows 合计可超过 8192，但不能超过单独声明的总预算。扫描保留 token 长度/视觉 patch 成本的 Pareto 最大形状，全部进入真实四卡 DDP、相同 AdamW 配置的两步反向探针，每 rank 要求至少 8 GiB 空闲且 peak reserved≤85%。训练要求成功探针与本次合同/shape audit SHA 匹配，并逐输入复核 token 形状。`--prepare-only` 只做 CPU 准入/形状审计并交包，不选择 GPU。不存在“八张图一定不爆显存”的承诺。
8. **默认入口**：`run_full_pipeline.sh`、`train.sh`、`audit_joint/train_phase3_batch.sh` 默认转到独立 base 入口；支持新 CLI 参数，或显式 `INDEX` / 新 `OUTPUT_DIR` / `P3_STAGE`。旧 B2 等环境设置拒绝静默继承。原生 `train.py` 直接 CLI 被 guard 拦住，正式新 worker 通过 import 使用原有计算函数。旧 sequence shell/CLI 必须显式 `LEGACY_PHASE3_SEQUENCE=1`，旧断点仍须原版 checkout。

多 query 的**原始呈现数不保证动作等量**。单独以原生 support-aware action 配额分配 CE 权重，使每个事件每轮总 loss 权重质量为 1024，再在其动作间分配；每张卡对锚点的加权 CE 求和，DDP 平均。原始可用标签、实际有效监督数、加权 loss 质量分开报告。这样避免大量 KEEP query 仅凭数量稀释 STOP，也避免简单把多 query loss 乘 N 改变梯度尺度。连续重叠监督仍具有相关性，不能当作 N 个独立场景。

## 版本与兼容

本次改变了公开入口文件及实验合同，新的 `phase3_base_trial_sources.json` 独立绑定全部依赖。没有改旧 sequence/Phase4 发布清单，也不改旧断点 SHA 来绕过恢复检查。已有任务、旧 base v1 或 Phase4 的恢复应留在原版 checkout；不要对运行中的代码目录执行同步。训练计算函数和共享采样模块未改，原生 `train.py` 的修改仅为直接 CLI guard。

本次不清理任何已有数据/权重、不训练旧模型、不调用 SSH。辅助索引需要原生生成并通过准入，不能手填一个 manifest 冒充审核通过。若没有可用的辅助索引，先交 CPU 覆盖失败包定位缺少的事件/位置，不启动 GPU，不降低门槛冒充密集监督完成。

## 执行顺序

在具有原始数据和模型的那台机器的 AutoMoT 根目录执行；机器名称只作元数据，两机资产不能混用。先用本版本从 Qwen 训练/验证新 base。示例使用已知 cr02 的历史索引相对路径；其它机器需使用它本机已经核验的同 SHA 索引。

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/audit_joint/phase3_base_trial.sh \
  --experiment base --stage quick \
  --index checkpoints/phase3_four_data_20261017_223254_120427458/frame_index.jsonl \
  --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
  --output phase3_base_v2_quick/result
```

日志和 handoff 都在此输出目录；后台运行时把外层 `launch.log` 放在当前根目录下的实验目录。输出必须不存在，不覆盖旧实验。quick 在首轮 2560 更新后评测固定 val，保留三轮调度和完整 optimizer/RNG 断点。确认后以同版本 `--stage complete --resume <quick/result/train/training_states/epoch_0001>`、新 `--output` 续剩余两轮。只有完成 7680 更新的 base final adapter 才能作为下面的初始化。

两条密集路线先 CPU 准入，再分别运行 matched single/all；不要让两任务抢同一组 GPU。下面变量必须由操作者指向明确的本机文件，不猜最新目录：

```bash
P3_INDEX=checkpoints/phase3_four_data_20261017_223254_120427458/frame_index.jsonl
P3_BASE_FINAL=/本机已完成base的输出/train/final
# 可选：原生完整辅助索引，单独SHA绑定，不替换P3_INDEX。
P3_AUX_ARGS=()
# P3_AUX_ARGS=(--aux-index /本机已准入的稠密原生索引/frame_index.jsonl)

bash qwen3vl_local/audit_joint/phase3_base_trial.sh \
  --experiment future --supervision all --prepare-only \
  --forecast-span-seconds 6 --forecast-step-seconds auto --min-future-coverage 0.8 \
  --index "$P3_INDEX" "${P3_AUX_ARGS[@]}" --init-adapter "$P3_BASE_FINAL" \
  --output phase3_future_v2_prepare/result
```

若准入通过，分别在新输出运行以下两次。每次 quick 都会重新核验本次完整剩余计划、执行四卡探针并训练首轮；两组后续都应完成相同额外更新预算再比较。

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/audit_joint/phase3_base_trial.sh \
  --experiment future --supervision single --stage quick \
  --index "$P3_INDEX" "${P3_AUX_ARGS[@]}" --init-adapter "$P3_BASE_FINAL" \
  --output phase3_future_v2_single/result

GPU_IDS=0,1,2,3 bash qwen3vl_local/audit_joint/phase3_base_trial.sh \
  --experiment future --supervision all --stage quick \
  --index "$P3_INDEX" "${P3_AUX_ARGS[@]}" --init-adapter "$P3_BASE_FINAL" \
  --output phase3_future_v2_all/result
```

方案二把两条命令的 experiment 同时改为 `windows`（严格四帧）或同时改为 `stream`（共享增长前缀），并更换输出名。不要拿不同 experiment 的 single/all 当作匹配对照。`--max-batch-tokens`、预测时间配置、aux-index、init-adapter 在同一组两次运行和各自接续中必须相同。

每次交 `result/handoff/phase3_base_report.zip` 与 `phase3_base_package.json`；前置尚未创建输出时交外层 `launch.log`。CPU 准入失败也交包。实际资源/全池覆盖/效果仍以这些服务器回执为准，本地小模型测试不能替代。
