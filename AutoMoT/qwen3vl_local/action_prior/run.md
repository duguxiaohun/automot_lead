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

## 可选 high-level planning 先验

`--high-level-planning` / `HIGH_LEVEL_PLANNING=1` 把原来“保留空间、等待冲突消除”等规划措辞替换成
参考 `sft_new_loop_phase3/prompts.py` 和 `context_taxonomy.py` 的短条件性规划段落。
默认关闭，`--no-high-level-planning` / `HIGH_LEVEL_PLANNING=0` 保留原提示词；CLI 优先。
它与摘要开关独立，开启后默认仍一次图文 prefill，不增加文字生成或加载 Phase3 模型。

- 急刹前车、切入、对向侵入、路口冲突、灯故障：减速、停车/继续等待、条件允许后持续增速；增速不要求之前停车。
- 静态障碍、弱势交通参与者：再考虑相对自车方向的左右首次未来跨线，区分弯道和已完成跨线。
- 显式 `event-balanced-scene-priors` 的 RE2/RE3 可增加横向语义，RE5 仅纵向；不从 RS 或全 NO 推断这些事件。
- 多事件事实全部保留，只写一段规划；没有确认事件时只保留车道/导航与按当前证据调速的通用提示。

例如静态障碍对应的规划文字（前面仍有道路和障碍事实）：

> If progress is constrained, decelerate or stop/continue waiting; once the path and priority permit, sustain a speed increase, without requiring a previous stop. If needed and clear, cross left or right relative to ego's heading; consider the first future lane-boundary crossing, not a curve or an already completed crossing.

单独开启本开关时，这是供轨迹生成参考的条件性动作语义，不是该帧动作答案；不注入 Phase3 标签、未来轨迹、动作 code 或单选回答要求。
Phase3 的预测任务/数值标定时间阈值不搬进 action prompt。导航、四图和轨迹训练目标照常保留。

```bash
# 开启 high-level 替换，默认仍不生成摘要；也可追加 --event-balanced。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --high-level-planning
GPU_IDS=0 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --high-level-planning
GPU_IDS=0,1,2,3 HIGH_LEVEL_PLANNING=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors

# 显式保留原提示词；CLI 覆盖环境变量。
HIGH_LEVEL_PLANNING=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --no-high-level-planning
GPU_IDS=0 HIGH_LEVEL_PLANNING=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --no-high-level-planning

# 已备好索引的底层入口；可选再生成摘要。
bash qwen3vl_local/action_prior/train.sh --dataset-priors --high-level-planning --generate-analysis
GPU_IDS=0 bash qwen3vl_local/action_prior/train.sh --dataset-priors --high-level-planning --generate-analysis

# 续训自动恢复保存的模式，eval/probe/闭环也只读 checkpoint 设置。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt
```

开关与 planning 版本进入合同、训练计划、文本缓存及逐例审计。两种模式需分别训练 decoder；
更换先验来源的评测许可也不能绕过 planning 模式校验。缺字段旧配置按关闭处理，但旧 run 仍需原源码通过执行指纹检查。
本开关支持 LoRA 与 dataset 来源，不要求均衡采样；`condition-mode base` 无先验消融会拒绝它。
若同时开启摘要，生成与复核读取同一新版先验；失败 fallback 只重述条件性规划及简短导航，全部事实仍留在 user prompt。

2026-09-17 本地验证：137 项提示词/缓存/合同/续训/入口回归通过，Python 与 shell 语法、
无先验消融不暴露该开关的检查通过。未运行真实 Qwen/BEV 训练或闭环，尚无动作质量提升结论。

## 再加入“接下来具体采取什么动作”

`--high-level-action-prior` / `HIGH_LEVEL_ACTION_PRIOR=1` 默认关闭，需要同时开启
`--high-level-planning`。**新训练不需要提供 `--high-level-action-index`**：自动复用当前 Phase3
候选与全帧事件映射，缺少产物时调用 `sft_new_loop_phase3/build_dataset.py` 的原标注逻辑生成，
再按 `(scenario, run_id, anchor)` 对齐 action 的 train/val/test。使用完整 `candidate_frames.jsonl`，
不会拿 Phase3 均衡抽样后的 `frame_index.jsonl` 代替全量标签。

只有 `special_eligible` 的 UE1–UE7、RE2/RE3/RE5 追加条件性规划域和具体动作块
`[UPCOMING_HIGH_LEVEL_ACTION]`。三动作域只保留 DECELERATE / STOP / RESUME，五动作域再允许
LANE_CHANGE_LEFT / LANE_CHANGE_RIGHT，严格复用 Phase3 的动作标签与域定义，不另设轨迹阈值。
RE2 沿用原映射门控和导航变道/早先障碍区分，不把所有变道描述成“绕障恢复”。

**普通背景保持原先提示词，不追加具体动作块**；`special_filtered` 和 `unconfirmed` 也不补动作。
它们记录 `not_applicable`；有效特殊帧的全 NO 记录 `no_action`，缺失外部结果记 `unavailable`，不混同。
只有开启 `--event-balanced` 时采用十个特殊桶各一份、确认普通背景两份的配额；动作开关本身不改变采样方式。
Phase3 已参与规则开发的物理路线仍强制放入 train，避免进入 val/test。

此模式使用 **Phase3 离线标注真值（`phase3_oracle`）**，其中动作标签依赖未来轨迹证据，
属于带额外真值条件的实验，不能当作 Phase3 模型预测的评测结果。没有加载 Phase3 adapter；
默认 dataset-priors 仍只做一次最终图文 prefill，零文字生成。显式开启摘要时，摘要/复核/fallback 使用同一动作。

```bash
# 自动准备动作标签，特殊十桶各一份、普通背景两份；不用填写动作索引。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --high-level-planning --high-level-action-prior
GPU_IDS=0 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --high-level-planning --high-level-action-prior

# 环境变量等价写法。
HIGH_LEVEL_PLANNING=1 HIGH_LEVEL_ACTION_PRIOR=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced
GPU_IDS=0,1,2,3 HIGH_LEVEL_PLANNING=1 HIGH_LEVEL_ACTION_PRIOR=1 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced

# 关闭具体动作输入，保持上一版条件性 planning。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --high-level-planning --no-high-level-action-prior
GPU_IDS=0 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --high-level-planning --no-high-level-action-prior

# 底层训练入口也会自动准备动作及其依赖，普通 uniform 采样保持不变。
bash qwen3vl_local/action_prior/train.sh --dataset-priors --high-level-planning --high-level-action-prior
GPU_IDS=0 bash qwen3vl_local/action_prior/train.sh --dataset-priors --high-level-planning --high-level-action-prior

# 续训自动恢复保存的开关和索引，不重新标注。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt
GPU_IDS=0 bash qwen3vl_local/action_prior/run_full_pipeline.sh --resume checkpoints/action_prior/latest/latest.pt
```

自动缓存位于 `checkpoints/action_prior_prepared/` 的 `phase3_*`、`full_*`、`actions_*` 目录，
使用锁、校验和、临时目录与原子发布。需要本地原始 LEAD 数据及现有人工标注；不会下载数据或训练 Phase3。
先剔除异常时长 route，再对齐动作。旧缓存规则/内容不匹配时隔离保留并重建；显式输入不静默替换。
来源、规则、三 split 内容和文件 SHA256 进入 checkpoint 合同，逐帧动作进入文本缓存 key。
训练计划记录来源、真值条件属性及各 split 的状态/覆盖统计。

`--high-level-action-index` / `HIGH_LEVEL_ACTION_INDEX` 仅保留为路径搬迁或高级外部输入接口，
正常开启不需要它。自动索引搬迁须同时携带同目录 `manifest.json`，full map 搬迁另用
`--event-balance-index`；resume/eval/probe 只接受与 checkpoint 同内容的产物。不能在同一个 decoder
上临时切换动作开关或替换标注，旧 run 仍需原源码恢复。

高级输入使用 `scoped_phase3_high_level_action_v2`，除精确帧身份、来源、status/actions 外，
必须明确 `event_status`、`event_buckets` 和 `planning_contexts`，普通背景必须为 `not_applicable`。
`prediction/provided` 供后续显式接入，来源声明本身不证明模型质量；任意自由文本不注入 prompt。
当前没有在线 Phase3 provider，Bench2Drive/CARLA 拒绝此模式。后续可复用
`PriorEngine.condition(..., high_level_action=...)`，仍需接通预测、事件门控和来源合同。

2026-09-18 本地验证：本次 266 项针对性测试全部通过，包含自动准备/复用、候选缺失与动作冲突拒绝、
异常 route 剔除、背景 prompt 不变、文件搬迁、续训和 CLI/环境变量优先级；23 个修改文件的 Python/shell
语法与 git diff 格式检查通过。全目录测试为 397 通过、19 失败，失败来自本机缺少 PEFT 或只读
`leaderboard/team_code/mot_lead_offline_runner.py`；未放宽正式依赖/指纹检查。
未执行真实数据全量构建、Qwen/BEV 训练或闭环，自动准备流程以小型合成索引验证。

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
# 最新一个 run：
bash qwen3vl_local/tb_serve.sh checkpoints/action_prior/latest/tb
# 指定端口：
TB_PORT=6006 bash qwen3vl_local/tb_serve.sh checkpoints/action_prior/latest/tb

# 同时看 action_prior 的所有历史 run（包含 latest 软链接）：
bash qwen3vl_local/tb_serve.sh checkpoints/action_prior
# 看 checkpoints 下所有实验：
bash qwen3vl_local/tb_serve.sh checkpoints
```

启动器默认只在最多 4 层内枚举标准的 `tb/`、`eval_tb/` 目录（不逐文件扫描模型和权重），解析 `latest -> run_时间戳` 软链接并显式交给 TensorBoard；因此上面两个父目录命令在 TensorBoard 2.21 fast loader 下也能显示 run，且传 `checkpoints/` 不会因全量检索权重而长时间不显示端口。启动时会先打印快速发现提示和发现的 event 叶目录数。若某个旧实验将 event 放在非标准、超过 4 层的位置，再显式使用完整扫描：

```bash
TB_DISCOVER_DEEP=1 bash qwen3vl_local/tb_serve.sh checkpoints
```

若训练刚创建、尚未写出 event，待首次写入后重启一次启动器即可把新的子 run 纳入列表。

打开终端显示的地址。主要看 `train/loss`、`train/samples_seen`、`val_epoch/route_ade_m` 和 `val_epoch/waypoint_ade_m`；均衡训练还可查看各事件组指标。FM loss 只反映训练拟合，轨迹质量以验证 ADE/FDE 为准。

TensorBoard 的横轴是 optimizer step，不是数据量；`train/samples_seen` 是用于核对实际累计 case 呈现数的独立 tag，不能直接被 Scalars 页面选作横轴。不同 run 的曲线密度/平滑程度因而不必然可比：训练日志每 `logging_steps` 个 update 写一个“该窗口的样本均值”，而 Scalars 的 smoothing 又按**点数**而非样本数做 EMA。看真实性能时先把 smoothing 设为 `0`，并核对 `samples_seen`、相同的 GPU 数与 `GRAD_ACCUM`；事件组还应同时查看同路径的 `/samples`，小分母的组损失天然更抖。

很长的 run 还可能被 TensorBoard 对 scalar 的默认 reservoir 抽样而显得稀疏。需要逐点查看时用（`0` 表示不抽样；大日志会增加浏览器内存和加载时间）：

```bash
TB_SCALAR_SAMPLES=0 bash qwen3vl_local/tb_serve.sh checkpoints/action_prior
```

若目标是让不同总步数的**新实验**在图上各有近似相同数量的、等间隔的点，给每个 run 预先取相同目标点数 `N`，设置 `LOGGING_STEPS=ceil(总 optimizer steps / N)`（例如想保留约 250 点，14,000 step 的 run 取约 `56`）。这会改变每点的平均窗口宽度，适合看总体收敛趋势；要比较短期噪声，应保持相同的 `LOGGING_STEPS` 与有效 global batch，而不是人为下采样。

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
