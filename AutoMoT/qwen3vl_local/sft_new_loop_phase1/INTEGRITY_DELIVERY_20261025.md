# Phase1/2：保留采样方法的完成验收与完整审计交付

本轮只完善训练完成判定、内容身份、测试证据、打包与比较，不调整采样方法、标签、prompt、划分或训练预算。适用于新运行；旧运行、旧完成回执和权重保留，不能为旧权重补写一个未实际执行的新完成认证。

> 独立运行更正：Phase1、Phase2 是两个独立实验，各自从 Qwen3.5 基座训练，不读取另一阶段的权重、ready、latest 或完成状态。此前文档中的 `&&` 顺序命令已撤回；不应把一个阶段的成功设为另一个阶段的前提。Phase1 历史命名里的“融合 Phase1+Phase2”指它自身已有的八问（可见事实+道路结构），不代表接着训练这里的独立 EVENT Phase2，原任务与采样保持。

> **真实全池审计结论（2026-10-25）**：本机 Phase1 默认768/三轮9216通过CPU准入；用户已明确选择保持原采样方法的Phase2 1024/三轮5376为新默认。原1536在本机数据上容量不足，拒绝记录保留，不再作为默认。实际数据、原因和独立命令见 [INDEPENDENT_DATA_20261025.md](INDEPENDENT_DATA_20261025.md)。

## 采样和预算

Phase1 沿用 v5 的八个 focus、每个 focus YES/NO=1:1、Phase1/道路结构任务各半，训练问法 all/subset/hierarchical=4:1:1；评测问法比例及确定性选题规则不变。

Phase2 保留 UE1/UE3/UE5/UE6 四正类等额、RE 两倍/INVALID 一倍、已有 highway/INVALID 子类及路线处理。当前数据与 prompt 仍是 v5_highway_ue3，不把它冒称为历史 v3 已验收成绩，也不回滚其标定。本轮保留已授权的 coverage_v2 跨轮低曝光优先和旧 action 选样辅助，不引入新 action 标定。

`sampling_preservation_20261025.json` 固定修改前 27 个训练/评测采样函数 AST 和 10 个采样、prompt、RGB及建库文件 SHA，回归校验实现未漂移。历史最优的绝对大训练量不会恢复。

| 四卡 B1、累积1、三轮，每个 RGB 模式 | 每轮呈现 | 每轮更新 | 全程更新 |
|---|---:|---:|---:|
| Phase1 | 12288 | 3072 | 9216 |
| Phase2 | 7168 | 1792 | 5376 |

默认 4096/12288 步门及 10 小时单轮耗时投影门不变。两种 RGB 分别训练，总预算相加；建库、加载和额外评测耗时不包含在循环步数中。

## 训练完成必须满足什么

- 超长输入在各 rank 统一检查后立即拒绝，不丢弃已选样本、不替换样本、不自动追加步数。
- 无有效监督、非有限 loss、非有限梯度拒绝继续更新。
- 完成时检查每个 rank 的循环步数、实际 optimizer 更新数、有效 frames/tasks 曝光、零跳过、零 pending，以及跨轮/总计一致性。
- 保存所选 adapter 和 final 等槽位的真实权重及配置 SHA，绑定训练索引/manifest、参数、预算、环境、源码和曝光证据。训练完成后的输入/来源漂移拒绝认证。
- `training_complete.json` 成功写出且 shell 再次验证通过后才更新 latest。保留失败现场，失败目录不能作为完成训练发布。

训练索引/标签与曝光明细受绑定；评测实际使用的 RGB 逐文件散列。这里不宣称给整个原始训练 RGB 库提供了物理不可变存储。

## 测试与打包

测试前保存原生采样器产生的完整 WorkItem、标签、问法顺序和实际 RGB SHA；测试后核验输入、基座、adapter未漂移。所有 rank 的完整 raw_output/cases 必须齐全，不能重复或错位。独立 CPU 重放使用同一原生 evaluator/parser，仅把保存的原始输出作为输入，重新计算所有原生评分字段；不加载模型、不重新推理。

`evaluation_complete.json` 绑定指标和各 rank 逐题文件。打包时再次重算，确认 base/LoRA 使用同一批题和基座，确认选中 adapter 属于该完成 run，再纳入：

- 完整 cases、metrics、测试计划及评分验收；
- 选中与 final 等槽位的权重 SHA/配置（不含大权重）；
- 训练输入绑定、预算、各 rank 和各轮曝光证据；
- dataset_receipt、pipeline preflight（完整流水线提供时）；
- 相关源码快照和包内逐文件散列清单；
- 缩小后的抽样 RGB、辅助报告和日志。

JSON/JSONL 评分证据不再截断到 500 行。30 MB 大小控制仅压缩 RGB/省略非必要大日志；完整证据仍超限时失败，不输出一个截断但声称完整的成功交付。原始结果保留在 eval 目录，可通过显式 BUNDLE_MAX_MB 调整本地交付上限。

推理性能记录实际 fresh prefill+decode 的同步耗时、输入/生成 token、峰值显存及环境；每 rank 前两题仅作预热，仍正常计入准确率。CPU 耗时只代表 CPU、显存留空；预热后样本不足或没有真实计时则标记未测。吞吐是每设备活跃推理速率，不冒称整个多卡作业吞吐或训练耗时。

## 入口与比较

两阶段分别使用各自的 `run_full_pipeline.sh`，没有调用或先后依赖。训练入口默认 `RUN_BUILD=0`：数据缺失立即退出，不隐式开始长时间建库。先在本地分别准备数据（不需要模型或 GPU），并检查对应 CPU 状态为 ready：

```bash
bash qwen3vl_local/sft_new_loop_phase1/prepare_data.sh
```

```bash
bash qwen3vl_local/sft_new_loop_phase2/prepare_data.sh
```

准备入口独立建库或复用本阶段数据、检查 RGB/val/test 采样、投影原生三轮训练计划；不训练，不更新 latest。状态分别位于 `checkpoints/sft_new_loop_phase1_data_cpu/coverage_v2/status.json` 和 `checkpoints/sft_new_loop_phase2_data_cpu/coverage_v2_default1024/status.json`。已有审核目录拒绝覆盖，重试用新的 `DATA_AUDIT_DIR`，旧失败证据保留。原生数据/采样失败仍拒绝，不自动改标定或配额。CPU 阶段各自有日志、10秒心跳和默认7200秒超时（`DATA_STAGE_TIMEOUT_SECONDS`），只终止自己启动的进程组。

在已激活环境的 AutoMoT 根目录，按需单独启动 Phase1：

```bash
PIPELINE_TIMESTAMP=coverage_v2_integrity_v1 bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
```

或单独启动 Phase2：

```bash
PIPELINE_TIMESTAMP=coverage_v2_integrity_v1 bash qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh
```

默认 4RGB；需要两种模式时，对所选阶段的准备和训练命令设置 `HISTORY_RGB_MODES="4rgb 2rgb_endpoints"`。数据、运行、latest、日志和包各属本阶段；同名运行拒绝覆盖。若在同一服务器同时运行两阶段，需显式设置不重叠的 `GPU_IDS`，两个独立入口不会替用户预留 GPU。传输索引/manifest 后仍需目标机器核验本地 RGB 与模型环境；本机 CPU ready 不认证远程文件或 GPU。命令是使用说明，不自动执行正式训练。

本地解包后的内容验收（`BUNDLE_DIR` 为解包后包含 delivery.json 的目录）：

```bash
python -m qwen3vl_local.sft_new_loop_phase1.delivery verify "$BUNDLE_DIR"
```

同一阶段的两个 production 评测目录比较（目录可位于审计包内）：

```bash
python -m qwen3vl_local.sft_new_loop_phase1.delivery compare "$OLD_EVAL_DIR" "$NEW_EVAL_DIR" --output comparison.json
```

结果区分全量同题系统对照、共同子集和缺少旧协议/RGB身份的历史参考。列出标签/图像冲突、旧对新错、旧错新对、原生各项分数和支持量。`exact_delta` 和翻转统计仅针对匹配题；`old_metrics/new_metrics` 保留各自完整原评测的原生指标。只有全量同题时才直接比较这些完整指标，部分重合时不能把它们当作共同子集的 P/R/F1。硬件、精度、注意力实现、生成上限、RGB模式、prompt或成员不匹配时不认证速度比较。

旧包没有新回执时不伪造认证；旧 parser/源码与当前不兼容时使用它自己的原始实现，不能绕过来源校验。历史 Phase1 v5 和 Phase2 v3 总分可列为参考，但新旧基座、预算及 Phase2 highway标签变化须单列，不能据总分差声称单一改动的收益，也不能把已开发曝光的 test 当作新盲测。

本轮仅 CPU/合成及小模型验收；完整 4B 四卡速度、显存、最终精度和远程数据准入仍需目标机器实测。
