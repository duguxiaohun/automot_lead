# Action prior 使用说明

以下命令在训练机的 `AutoMoT/` 目录执行。准备好 `lead_data`、已有人工标注和本地 Qwen/BEV 权重；脚本自动构建所需索引，不下载模型。

## Qwen KV 输入与摘要开关

默认 `generate_analysis=False`：**四张图像＋含自然 RS/EVENT 先验、当前速度与导航的提示词 → frozen base Qwen 一次 prefill → KV 交给轨迹 decoder**。
不生成 talk/CoT/摘要，不追加 assistant 回答，也不运行摘要复核或 fallback。LoRA 来源仍执行 Phase1/2 的先验问答；
`--dataset-priors` 则直接查标签，因此默认每帧没有文字生成，只有一次最终 base KV prefill。

```bash
# 显式关闭摘要（默认行为）；可与 --event-balanced 等原开关组合。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --no-generate-analysis
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --no-generate-analysis

# 保留原路径：生成摘要，然后把图像、先验提示词及摘要一起编码到最终 KV。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --generate-analysis
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --generate-analysis

# 等价环境变量，0 关闭 / 1 开启；显式 CLI 优先。
GENERATE_ANALYSIS=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors
GPU_IDS=0,1,2,3 GENERATE_ANALYSIS=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors
```

`train.sh` 和 `train.py` 同样接受 `--generate-analysis` / `--no-generate-analysis`。
`--analysis-review` / `ANALYSIS_REVIEW` 仅在摘要开启时生效，不会单独开启生成；开启摘要后，
LoRA 来源默认复核，dataset 来源在 shell 入口默认不复核，可显式覆盖。摘要失败时保留原 fallback。

模式写入 `config.json`、checkpoint 条件合同和训练计划；逐例审计用 `final_cache_content=inputs_only` /
`inputs_and_analysis` 区分，关闭时 `analysis_acceptance=disabled`。两种模式不共用文本缓存条目。
**续训、eval、probe 和闭环自动沿用 checkpoint 模式**，测试命令无需重复开关，也不接受临时切换 KV 模式。
比较两种模式需要分别训练 decoder。本次源码指纹改变，旧 run 必须用原源码恢复；开启摘要并不绕过旧 checkpoint 兼容检查。

2026-09-15 本地验证：直接图文 KV、冷/热缓存、摘要开启及复核/fallback、CLI/环境优先级、
真实 resume 配置恢复、生成预算和 `summary_disabled` 分组回归通过，Python/shell 语法检查通过。
未运行真实 Qwen/BEV GPU 训练；当前环境缺 peft 和只读 `leaderboard/team_code/mot_lead_offline_runner.py`，
真实双 LoRA 测试及完整执行源码指纹集成检查未完成。模式合同单测固定无关源码身份；正式指纹守卫没有放宽。

## 开始训练

```bash
# 数据集标定先验：自动准备标签、训练并完成 test/probe。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors

# 开启 UE/特殊 RE 均衡采样，只需增加一个开关。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced

# 默认自动选卡；需要指定四张卡时：
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced
```

`--event-balanced`：UE1–UE7、RE2、RE3、RE5 各一份，确认的普通 RE 合计两份。
默认每帧单 epoch 最多重复 8 次，自动确定可行预算，并优先覆盖不同帧。省略此开关就是自然采样；`--no-event-balanced` 可覆盖环境变量中开启的设置。

脚本自动准备 action 索引、Phase1 标签索引、先验标签以及均衡采样所需的候选/full map；已存在的有效结果会复用。
这只构建数据，不训练 Phase1/Phase3。首次准备可能较慢，终端和 pipeline 日志会显示进度。
不需要手写 `DATA_DIR` 或 `EVENT_BALANCE_INDEX`。

如果旧版在准备阶段报 `staging.rename(...)` 的 `FileExistsError`，同步新版源码后直接重跑原命令。
候选与 full map 发布时会重新校验已存在的目录；并发结果完整且内容一致时复用，残缺缓存移到
`checkpoints/action_prior_prepared/.invalid-*` 保留后重建。不要删除 `.prepare.lock` 来解锁：
进程退出会自动释放锁，文件残留不妨碍重跑；等锁时会打印 `[prepare] waiting for cache lock`。
该恢复仅作用于自动准备的缓存，显式指定的 `--event-balance-index` 仍严格校验。
不同内容的完整发布冲突、权限、磁盘空间及 I/O 错误仍会中止并报告原因。

```bash
# 使用 Phase1/2 LoRA 现场生成先验（耗时更长，需已有 LoRA 权重）。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --event-balanced
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --event-balanced

# 缩短本次训练，例如训练 3 个 epoch；其它 train.py 参数同样可直接追加。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --num-epochs 3
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --num-epochs 3

# 可选：向标定先验注入 10% 噪声。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --prior-noise 0.1
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --prior-noise 0.1
```

均衡采样不改变 Qwen 提示词。独立的 `--event-balanced-scene-priors` 是离线场景文字实验，要求 `--dataset-priors` 且噪声为 0；全流程也会自动准备它的映射，但这种模型不能用于闭环。

## 查看 TensorBoard 和输出

另开一个终端：

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/action_prior/latest/tb
# 指定端口：
TB_PORT=6006 bash qwen3vl_local/tb_serve.sh checkpoints/action_prior/latest/tb
```

打开终端显示的地址。主要看 `train/loss`、`train/samples_seen`、`val_epoch/route_ade_m` 和 `val_epoch/waypoint_ade_m`；均衡训练还可查看各事件组指标。FM loss 只反映训练拟合，轨迹质量以验证 ADE/FDE 为准。

每次训练使用独立的 `checkpoints/action_prior/run_时间戳/`，`latest` 指向最新 run：

| 文件/目录 | 用途 |
| --- | --- |
| `train.log`、`pipeline.log` | 训练/数据准备日志 |
| `best.pt`、`latest.pt` | 验证最优/最新 checkpoint |
| `tb/` | TensorBoard |
| `sampling/` | 每轮采样配额、唯一帧及重复次数 |
| `test/`、`probe/` | 最终测试指标和案例 |

## 续训

```bash
# 继续训练并完成最终 test/probe；采样开关和索引从原配置恢复。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt

# 兼容环境变量写法；CLI --resume 优先于 RESUME。
RESUME=checkpoints/action_prior/latest/latest.pt bash qwen3vl_local/action_prior/run_full_pipeline.sh
GPU_IDS=0,1,2,3 RESUME=checkpoints/action_prior/latest/latest.pt bash qwen3vl_local/action_prior/run_full_pipeline.sh

# 仅续训：
bash qwen3vl_local/action_prior/resume.sh checkpoints/action_prior/latest/latest.pt
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/resume.sh checkpoints/action_prior/latest/latest.pt

# 底层训练入口也会先恢复配置，包含原摘要开关；兼容 --resume=路径 和 RESUME=路径。
bash qwen3vl_local/action_prior/train.sh --resume checkpoints/action_prior/latest/latest.pt
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/train.sh --resume checkpoints/action_prior/latest/latest.pt
```

续训使用原代码版本和原配置。更改采样算法或实验条件时开新 run；不要修改旧 checkpoint 的合同。
`--resume 路径`、`--resume=路径` 与 `RESUME=路径` 共用专用恢复入口，恢复原 LR、epoch、累积、
采样课程、标签路径与卡数默认值，不执行新训练的数据构建或 LoRA 重选。
`train.sh` 同样在拼接新训练默认参数前转入共用恢复入口，不再用默认
`--no-generate-analysis` 覆盖原 run 的 `True`。仅显式环境变量才作为覆盖传入，CLI 优先；
不要显式切换摘要模式续训，条件合同仍会拒绝不同 KV 模式。
pipeline 在读取配置前解析 checkpoint 的真实路径；即使另一个实验修改 `latest`，
续训和最终 test/probe 也固定使用原 run 及其 `best.pt`。缺值或不存在的 checkpoint 会提前报错。

```bash
# 搬迁后仅覆盖路径；full map 要连同 manifest.json 一起移动，内容身份仍严格核验。
bash qwen3vl_local/action_prior/run_full_pipeline.sh \
  --resume checkpoints/action_prior/latest/latest.pt \
  --data-root lead_data --data-dir checkpoints/action_prior_data \
  --prior-labels checkpoints/moved_labels/prior_labels.jsonl \
  --event-balance-index checkpoints/moved_event_map/full_event_mapping.jsonl
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh \
  --resume checkpoints/action_prior/latest/latest.pt \
  --data-root lead_data --data-dir checkpoints/action_prior_data \
  --prior-labels checkpoints/moved_labels/prior_labels.jsonl \
  --event-balance-index checkpoints/moved_event_map/full_event_mapping.jsonl
```

上述标签搬迁示例适用于 dataset-priors run；LoRA run 不传 `--prior-labels`。
显式 `--data-root`、`--data-dir`、`--model-dir`、`--lead-bev-ckpt` 及对应环境变量，
连同标签/full-map 路径覆盖都会贯穿续训与最终 test/probe，CLI 优先。
未显式指定时沿用原配置和 checkpoint，脚本默认路径不会覆盖旧 run。

## 独立测试

```bash
bash qwen3vl_local/action_prior/eval.sh --checkpoint checkpoints/action_prior/latest/best.pt
GPU_IDS=0 bash qwen3vl_local/action_prior/eval.sh --checkpoint checkpoints/action_prior/latest/best.pt

# 小样本可视化：
bash qwen3vl_local/action_prior/probe.sh --checkpoint checkpoints/action_prior/latest/best.pt
GPU_IDS=0 bash qwen3vl_local/action_prior/probe.sh --checkpoint checkpoints/action_prior/latest/best.pt
```

默认沿用 checkpoint 的先验设置，test 不进行均衡重采样。闭环、审计、权重迁移等少用操作见 [AUDIT.md](AUDIT.md)，模型与采样合同见 [DESIGN.md](DESIGN.md)。

## 两个无先验消融使用同一均衡课程

`action_expert_ablation/qwen_simple` 与 `bev_only` 现在直接引用本目录的开关解析、候选/full-map
自动准备、采样/预算/合同校验、训练循环和事件桶指标。调整比例只改 `event_balance.py` 的
`EVENT_BALANCE_WEIGHTS`，不需要复制逻辑；三个实验须统一源码版本、action 索引、seed 与训练预算。
主线本脚本默认每次创建 run 索引；配对实验请显式 `DATA_DIR=checkpoints/action_prior_data`，
与两个消融的默认索引对齐。主线 planning 先验和分析 prompt 只由本目录维护；消融保持各自的无先验输入定义。

```bash
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced
```

完整三组配对、课程参数、续训/索引搬迁命令见[消融运行文档](../action_expert_ablation/run.md#与主线完全共用-event-balanced)。

2026-09-14 复审后，三组同时首次启动也共用 action 索引构建锁；索引必须同时有 manifest 和三个 split。
均衡采样直接使用 Phase3 当前开发路线名单隔离 holdout（当前 709 组），验证覆盖按全部真实事件桶统计，
不会把 `special_filtered` 误报成缺桶。修改正整数权重表后，自动 epoch 预算与配额均同步更新。
本次执行源码和有效数据划分有变化，使用新 run；旧 checkpoint 仍用原代码恢复。


2026-09-14 验证：主线与消融回归共 **291 passed / 1 skipped**，Python/shell 语法检查通过。
新增测试执行真实 `resume.py` 配置恢复，仅替换模型 launcher，覆盖三种续训写法、CLI 优先、
路径搬迁、latest 改指与非法 checkpoint 提前失败。跳过项为缺少 TensorBoard 依赖的事件文件测试；
尚未运行真实 Qwen/BEV GPU/DDP 训练，首次上机先使用消融文档中的独立短预算 pipeline。
