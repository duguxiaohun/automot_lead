# 项目规则 (CLAUDE.md)

> 本文件会被 Claude Code 在每次启动时**自动加载**到上下文。所有规则对所有
> 后续对话有效，无需用户重复说明。
>
> 本项目同时维护 [`AGENTS.md`](AGENTS.md) 作为 Codex / 通用 coding agent 的入口。
> **CLAUDE.md 与 AGENTS.md 必须保持规则同步**：任何一边新增/修改文件白名单、
> git 规则、工作流偏好、禁止事项、项目入口说明时，必须同步更新另一边。

---

## 1. 第一动作 — 先读项目文档

**在开始任何工作（包括回答简单问题、写代码、改文件）之前**，请先读取
[`AGENTS.md`](AGENTS.md) 和 [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)。

- `AGENTS.md` 是给 Codex / 通用 agent 的镜像入口；Claude 也要读，确保两边规则一致。
- `PROJECT_CONTEXT.md` 是项目技术事实来源；Claude 和 Codex 都必须读。

这个文件已经把：

- `lead/` 仓库的数据采集 / 加载 / BEV 栅格化参数
- `AutoMoT/` 仓库的在线推理慢/快路径与 KV cache 流程
- LiDAR / RGB / target_point / scenario_type 在两边的所有差异
- `mot_lead_offline_runner.py` 当前已知偏离训练分布的具体点（含 ⚠ 标记）
- `vlm_paradigm_a_runner.py` 的本地 Qwen 文字生成规则：`qwen` backend 只读
  `AutoMoT/checkpoints/Qwen3-VL-4B`（`local_files_only=True`），并用 HF 标准
  `past_key_values` 显式 prefill/decode；AutoMoT `InterleaveInferencer` /
  `qwen3vl_template_inference` 绑定自定义 MoT 架构，不要拿来直接跑
  standalone Qwen 完整自由文本生成
- `qwen3vl_instruct_paradigm_a_runner.py` 是 standalone Qwen-only 范式 A runner，
  只跑本地 `AutoMoT/checkpoints/Qwen3-VL-4B-Instruct`；该目录对应
  HuggingFace `repo_id=Qwen/Qwen3-VL-4B-Instruct`，用户远程环境已下载。
  必须 `local_files_only=True` 且设置 HF/Transformers offline 环境变量，
  禁止下载；不 import `vlm_paradigm_a_runner.py`，不接 AutoMoT `InterleaveInferencer`
- `AutoMoT/qwen3vl_local/` 保存 Qwen3-VL-Instruct 本地可魔改代码：
  `prompt_pipeline.py` 从 `vlm_paradigm_a_runner.py` 的迁移块同步完整提示词/状态机；
  另含 LEAD RGB 读取、显式 prefill/decode、KV cache summary 与可选 `torch.save`。
  Qwen3-VL 自定义 KV 增量 decode 必须复用 `mrope_utils.py` 的
  `qwen3vl_incremental_forward` 显式复算 M-RoPE `position_ids`，禁止再依赖
  `prepare_inputs_for_generation` 组装 decode 输入（PEFT wrapper 会丢 `cache_position`）。
  `engine.py` 的 `cache_system_prompt` 只允许纯文本 suffix 复用 system-prefix cache；
  含 `pixel_values` / `image_grid_thw` 的多模态输入必须回退完整 prefill，避免半截图文
  M-RoPE cache 错位。
  `engine.py` 的 `_clone_cache` 必须优先保持 Transformers `Cache` 对象类型（如带
  `get_mask_sizes` / `get_seq_length` 的新版 cache），legacy tuple 只能作为旧版兜底；
  新版 Qwen3-VL forward 会直接调用 Cache 方法，不能把它退化成普通 tuple。
- `mot_lead_offline_runner.py` 只走
  `AutoMoT/qwen3vl_local.engine.LocalQwen3VLInstructEngine` 做 frozen Qwen prefill，
  再接 LeadMoT decoder；已移除 AutoMoT legacy `kv_cache_fixed_inference(...)`、
  `InterleaveInferencer` 与原 fast head 接口。

整理成可直接消费的形式。**不要从源码重新扒**，会浪费 token 且容易得出错误结论
（前几轮迭代已经证实凭印象推断会犯多种事实错误，详见 PROJECT_CONTEXT.md 的修订历史）。

如果对 PROJECT_CONTEXT.md 里某处描述有疑问 → 去源码核对 → 核对后**直接修正
文档**（修正方式见下面"修改范围"）。

如果修改了本文件中任何规则，也必须同步修改 `AGENTS.md`；如果发现 `AGENTS.md`
比本文件更新，也必须把对应规则同步回本文件。不要让 Claude 和 Codex 看到两套不同规则。

---

## 2. 修改范围限制（**强制**）

未经用户明确同意时，**只允许修改以下文件**：

| 文件 | 用途 |
|---|---|
| `PROJECT_CONTEXT.md` | 项目说明文档，需要随代码修改持续更新 |
| `AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py` | 用户主战场：把 LEAD 数据离线喂给 AutoMoT 推理的桥接脚本。`LeadOfflineMoTRunner` 加载 ckpt 时按 `decoder_config.use_bev` / `use_subgoal` / `use_final_goal` 自描述切换分支：use_subgoal=True ckpt 必须在 `lead_clip` 里塞 `subgoal_rgb_path/subgoal_scenario/subgoal_status/subgoal_event`，由 `build_clip_from_real_lead_route(..., subgoal_*=...)` 写入；runner 的 `main()` 在加载 ckpt 后按 use_subgoal 自动调 `--keyframes` 反查 anchor 对应 STATUS → SUBGOAL → SUBGOAL keyframe RGB，无需用户额外指定 |
| `AutoMoT/leaderboard/team_code/vlm_paradigm_a_runner.py` | 范式 A 对照脚本，保留 automot/qwen 双 backend |
| `AutoMoT/leaderboard/team_code/qwen3vl_instruct_paradigm_a_runner.py` | standalone Qwen3-VL-4B-Instruct 范式 A 脚本 |
| `AutoMoT/leaderboard/team_code/qwen3vl_dit_goalgen_runner.py` | 子目标 latent 生成新路线 runner（teacher-forced prefill → DiT-MoT → flow matching，详见 PROJECT_CONTEXT.md §15） |
| `AutoMoT/leaderboard/team_code/automot_utils.py` | 按用户同意纳入白名单：AutoMoT legacy prompt helper；`build_cleaned_prompt_and_modes` 必须接收 7 元 `[speed,tp,ntp,final_goal]` 并在 prompt 写入 final destination |
| `AutoMoT/Automot/team_code/automot_utils.py` | 按用户同意纳入白名单：AutoMoT 原始副本 prompt helper；必须与 `AutoMoT/leaderboard/team_code/automot_utils.py` 的 7 元 final destination prompt 保持同步 |
| `AutoMoT/Automot/mot/evaluation/inference.py` / `AutoMoT/Automot/mot/modeling/automot/automot.py` | 按用户同意纳入白名单：AutoMoT 原始副本中的 prompt 示例注释；涉及 target_point / next_target_point 的 prompt 示例必须包含 final destination，并与 2s 规划视野同步 |
| `AutoMoT/leaderboard/team_code/mot_b2d_agent.py` | 按用户同意纳入白名单：legacy AutoMoT 在线 agent；涉及 wp/nwp prompt 时必须同步按 `max(speed*lookahead_s, 5m)` 弧长生成 tp/ntp，生成局部 final_goal 并传入 `automot_utils.build_cleaned_prompt_and_modes` |
| `AutoMoT/leaderboard/team_code/display_interface.py` / `AutoMoT/Automot/team_code/display_interface.py` | 按用户同意纳入白名单：AutoMoT 显示层；decision 三元组只表示 now/+1s/+2s，不要再沿用旧 3s 命名 |
| `AutoMoT/qwen3vl_local/eval_carla/` | LeadMoT 闭环评测子包（全部白名单内）：实时 agent + 5 路视频 + 投影 overlay + scenario 反向映射 + 聚合 + Flask webapp。`agent.py` 直接复用 `LeadOfflineMoTRunner`；target_point / next_target_point 与训练同走 `max(speed*lookahead_s, 5m)` route 弧长前推，默认 tp=1.0s / ntp=2.0s；final_goal 为 route 真实终点：训练取 LEAD 采集保存的 `meta["next_target_points"][-1]` 转 ego，在线 eval_carla 取 `scenario_picker.py` 对应 route XML 最后一个 waypoint 转 ego，不能再用 `meta["route"][-1]` 或固定局部 horizon；warmup 为 LEAD 风格 left-pad 复制 frame 0 立即推理；按 ckpt `decoder_config.use_bev` 决定是否声明/读取 LiDAR/radar；ckpt `decoder_config.use_subgoal=True` 当前不支持闭环，agent.py 加载时立即 `raise NotImplementedError` 并留 `TODO(subgoal)` 接口；其余细节以 `EVAL_CARLA_PLAN.md` / `EVAL_CARLA_RUN.md` 为准。 |
| `AutoMoT/lead_video_tools/` | 按用户同意新增到白名单：LEAD 离线 RGB 视频转换工具；只读 `/datashare/IOL4SGH/data/data/<Scenario>/<run_id>/rgb/*.jpg`，按 4Hz 生成 `/data/lead_video/<Scenario>/<run_id>/{input,left,front,right}.mp4`（默认 input，`--views` 可选三视角裁剪），默认在左上角写 frame id，支持异常 route 剔除、断点续跑、ffprobe 完整性检查、运行文档和 `--workers` route 级 CPU 并行（`--workers 0` 自动按 CPU 估计）；`rgb_to_video.py` 普通转换默认剔除异常时长 route；`abnormal_duration_filter.py` 按硬规则输出异常采集名单：4Hz 下 `frames >= 361`（严格大于 1 分 30 秒 / 90s）且不在白名单内的 route 全部视为异常并写入 `abnormal_confirmed_over_90s.txt`；`BlockedIntersection` 与 `ControlLoss` 是唯一时长白名单不写入名单；`Accident`、`park*`、`dynamic*` 不再有 90-100 秒存疑段豁免；`abnormal_possible_90s_to_100s.txt` 只为旧接口兼容保留，正常应为空。**凡是 AutoMoT/keyframe_filter、AutoMoT/qwen3vl_local 或其它入口使用 LEAD 数据集，都必须在构建样本/调研/probe 前先剔除这些异常 route**；筛选时打印 discover + route 级进度条，两个 txt 名单只保留 `Scenario/run_id`，详情保留在 `abnormal_duration_summary.json`；只有显式传 `rgb_to_video.py --abnormal-route-list-dir` 才复用筛选目录只转名单 route。 |
| `AutoMoT/data/lead/` | 按用户同意纳入白名单：`lead_data` 对应 route XML 根目录，由 `AutoMoT/data/data_routes` 提取整理而来。命名规范固定为 `data/lead/<Scenario>/<Town>_<route_key>.xml`：旧数字 route 为 `Town03_route_001783.xml`，新版子编号为 `Town12_route_1054_0.xml`，命名本身带 Town 的 legacy key 为 `Town06_route_Town06_13.xml`，legacy key 内部带 route 编号时保留完整 key，如 `Town12_route_Town12_route15.xml`。从 `lead_data/<Scenario>/<run_id>` 找 XML 时，`Scenario` 必须取 run 的父目录；run_id 先剥末尾 `MM_DD_HH_MM_SS` 时间戳，再只在存在时剥尾部采集后缀 `_route0`；`Town12_route15` 这类 legacy key 本体里的 `route15` 不能剥，也不能要求它带 `_route0`。XML 文件名公式：`route_key` 以 `route_` 开头时用 `<Town>_<route_key>.xml`，否则用 `<Town>_route_<route_key>.xml`。2026-07-03 全量核对结果：`lead_data` 9715 个 run 去重后 9294 个 `(Scenario,Town,route_key)`，`data/lead` 正好 9294 个 XML，缺失 0、冗余 0、命名不规范 0、XML 解析失败 0、内容结构异常 0；XML 内 `<weathis_juncer>` 拼写已统一修正为 `<weather>`。40 个 XML 的 `data_routes` 源文件位于不同 scenario 目录（36 个 `noScenarios`、4 个 `ConstructionObstacleTwoWays`），不是缺失；另有 `ParkedObstacle/Town12_route_Town12_route15.xml` 覆盖有效并与 `lead_data/ParkedObstacle/Town12_Rep0_Town12_route15_*` 对应，但未在 `AutoMoT/data/data_routes` 找到直接源文件。使用时以 `lead_data` / `data/lead` 的 scenario 目录为准，不能把该项当作 XML 缺失。 |
| `AutoMoT/keyframe_filter/` | 按用户同意新增到 clean push 白名单：旧版 LEAD 关键帧选择器与新 ROAD/EVENT 语义重标注方案目录。`rule_based_keyframe_filter.py` 旧逻辑按 scenario 固定抽 initial / 3 middle / final，主要依赖 meta 距离字段、speed/accel/brake，并 fallback 到 bbox / RGB motion；只作为突发事件 span 提议器和验证工具参考，不再视为最终帧级 STATUS/SUBGOAL 真值。`classifier_logic.txt` 是用户人工调研的道路结构与事件分类草案；`ROAD_EVENT_CLASSIFICATION_PLAN.md` 是 ROAD/EVENT canonical 总方案，已合并 ROAD_STRUCTURE 调研协议、runtime 门控和错帧回查流程；`ROAD_EVENT_CANDIDATE_MAPPING.md` 保留为 Qwen/probe 可解析的候选表。代码、方案文档、规则配置、README、HTML/CSS/JS、verification 工具和手写说明允许修改、追踪、commit 和 push；`collection_output/`、`rgb_r4_r5_audit_results/`、`keyframes_all_scenarios.json`、`R2_ROUTE_RGB_REVIEW_INDEX_*.csv`、`ROAD_EVENT_INTERRUPTED_OVERLAY_*_IDS_*.csv`、`ROAD_EVENT_INTERRUPTED_OVERLAY_IDS_SUMMARY_*.json` 都是本地数据/审计/证据产物，默认不入库、不 push。需要共享时应先整理为方案文档或小型规则配置。ROAD_STRUCTURE / ROAD_EVENT 规则迭代不是手工凭空调参：必须按“先把思路写成可执行代码 → 跑小范围样本并生成可视化/逐帧注释 → 查看错帧与证据归因 → 修正规则/阈值 → 再跑 smoke”的闭环推进。push 前可精确执行 `git add AutoMoT/keyframe_filter/`，依赖该目录内 `.gitignore` 排除输出产物；若要提交新产物，必须先确认它不是可再生 evidence。 |
| `AutoMoT/qwen3vl_local/` | Qwen3-VL-Instruct 本地 helper 包；其中 `tb_serve.sh` 是通用 TensorBoard 启动器（SFT / GoalGen / LeadMoT / VAE 共用），`mrope_utils.py` 是 Qwen3-VL 增量 decode 的本地 M-RoPE position_ids 实现，`goalgen/` 子包是 §15 新路线全部模块（vae/prompt/qwen_kv/keyframes/dit/flow），`eval_carla/` 子包是上述闭环评测子包 |
| `AutoMoT/qwen3vl_local/sft/__init__.py` / `SFT_PLAN.md` / `SFT_RUN.md` / `build_dataset.py` / `build_teacher.py` / `train.py` / `train.sh` / `eval.py` / `probe.py` / `check_loss_mask.py` / `inspect_teacher_outputs.py` | 统一 LoRA SFT 子包（已废弃 v1/v2 双轨与 ms-swift）：`SFT_PLAN.md` / `SFT_RUN.md` 是设计与运行入口；`build_dataset.py` 只产 `dataset_version="pending"` jsonl（assistant 含 `__TEACHER_PENDING__` 占位）；`train.sh` → `train.py` 用 `peft.LoraConfig` + `get_peft_model` 直接把 LoRA 注入 base，torch DDP + 手写 train loop，每个 train batch 内禁用 adapter，并调用底层 Qwen base model 现场 greedy 生成 ANALYSIS 真值（避开 `PeftModel.generate` 的 Qwen3-VL 生成错位问题），再启用 adapter 跑 student forward + per-token 加权 loss（ANALYSIS body `SFT_ANALYSIS_WEIGHT`/默认 0.5，学习大致语言推理但不逐字压过状态监督；其余 assistant 段 1.0，prompt 段 0）。**不再离线物化 teacher、不再写 manifest、不再有 runtime_teacher_data 复用机制**；`build_teacher.py` 仅作为可选离线 dump 工具，不被训练入口自动调用。`eval.py` / `probe.py` 默认 `merge_and_unload` 加载 LoRA，case dump 保存 `expert_analysis.txt` / `language_compare.json` 对比 base-teacher 专家语言与模型 ANALYSIS；`probe.py` 的 token loss 使用训练同款权重汇总；`check_loss_mask.py` 验 train.py 内置 mask；`inspect_teacher_outputs.py` 支持 `--live` 现场重跑 teacher 抽检 |
| `AutoMoT/qwen3vl_local/sft_v2/__init__.py` / `SFT_V2_PLAN.md` / `SFT_V2_RUN.md` / `prompts.py` / `build_dataset.py` / `train.py` / `train.sh` / `eval.py` / `probe.py` / `check_loss_mask.py` | 按用户同意新增到白名单：SFT v2 两段式串行选择题子包。输入仍为 LEAD stitched RGB + 语言 prompt；stage-1 只列 `SCENE_CHOICES` 并输出 `SCENE`，stage-2 作为同一条对话的后续 user prompt，按预测 scene 的 `EVENT_SEQUENCE` 输出 `STATUS/SUBGOAL`，推理时必须复用 stage-1 已吃图像和场景 prompt 后的 KV cache；默认 `--samples-per-scenario 0` 全量保留合法候选，默认 `--wrong-scene-ratio 0.15` 只增强 train rows；不再有 ANALYSIS / teacher / pending cache；训练 loss 只监督 scene/status/subgoal 值 token，格式 token 为 0 loss；LoRA 默认只注入语言侧 Linear，视觉侧通过 `--lora-vision-scope` / `LORA_VISION_SCOPE` 选择 `off` / `merger` / `last4` / `all` 四档（`--lora-vision` / `LORA_VISION=1` 作为 `all` 的 legacy 别名保留）；开启视觉 LoRA 时默认带"视觉组单独 LR 倍率 `--vision-lr-scale=0.1`（受 `--max-vision-lr-scale=0.25` 上限约束）+ 分组梯度裁剪 `--language-clip-norm=1.0` / `--vision-clip-norm=0.3` + TB 观测 `grad_norm/{language,vision}` / `param_norm/lora_{language,vision}` / `vision_guard_bad_steps` + `STRICT_VISION_SCOPE=1` 命名漂移硬拒绝 + `VISION_GUARD_ENABLED=1` 运行时熔断"保险；熔断时写 `fuse_stop_step_<N>/` 与 `fuse_reason.txt`，并跳过正常 `final/` 保存，防止视觉表征被冲坏且避免误用异常产物；base Qwen checkpoint 始终只读，训练只保存 adapter delta，并写 `sft_v2_adapter_config.json`（含 `lora_vision_scope` 与保险参数）；eval/probe 加载前按 adapter 配置判断普通 LoRA / 视觉 LoRA 并校验权重 key，不一致直接拒绝。自由生成评估中 scene 不在白名单则中断，scene 合法但错误时仍按预测 scene 进入 stage-2 并用串行口径计错，同时输出 `valid_total` / `*_valid_scene` 指标。运行文档见 `SFT_V2_RUN.md` |
| `AutoMoT/qwen3vl_local/sft_v3/` 子包（含 `__init__.py` / `SFT_V3_PLAN.md` / `SFT_V3_RUN.md` / `prompts.py` / `build_dataset.py` / `train.py` / `train.sh` / `eval.py` / `probe.py` / `check_loss_mask.py` / `test_memory_update.py` / `test_kv_reuse.py` / `test_gt_leak_filter.py`） | 按用户同意新增到白名单：SFT v3 代码已落地。训练单元为 sub-scenario 时间序列（`[f1-δ, f3]`），Memory 为学生自维护文本状态（场景/状态/子目标 + EGO_TO_GOAL_XY），每帧三步内循环（step1 纯视觉分析、step2 场景判断、step3 状态/子目标判断）并用 teacher/student 蒸馏；Phase B 每帧开头弱纠偏 scene=GT 反向学习“对的别改”；δ 允许 0 且只封顶 10，`EGO_TO_GOAL_XY` 严格来自 meta `next_target_points[-1]` 并在帧末预取下一帧，step3 触发统一走 `should_trigger_step3`。`train.sh` 默认 `ddp`（历史模式名），每卡默认 batch=1；多卡训练采用 work-stealing + local-SGD：不包 DDP、不静态分片、不截断尾部，通过 TCPStore 抢 episode，NCCL collective 前先 TCPStore rendezvous，先广播 rank0 LoRA 初始权重，按本轮 optimizer step 数加权平均 LoRA 参数，并且 `checkpoint-*` / `final/` 都在参数平均后保存；sync 日志/TB 记录 `all_rank_steps`、`round_eps`、`total_eps` 用于审计训练量；LoRA 视觉接口与 v2 同构并默认 `off`，保存 adapter delta + `sft_v3_adapter_config.json`。 |
| `AutoMoT/qwen3vl_local/sft_v4/` 子包（含 `__init__.py` / `SFT_V4_PLAN.md` / `SFT_V4_RUN.md` / `prompts.py` / `build_dataset.py` / `train.py` / `train.sh` / `eval.py` / `probe.py` / `check_loss_mask.py` / `test_memory_update.py` / `test_kv_reuse.py` / `test_kv_vs_native.py` / `test_gt_leak_filter.py` / `replay.py` / `collect.py` / `learn.py` / `launch_offpolicy.sh` / `inspect_teacher.py`） | 按用户同意新增到白名单：SFT v4 是 sequence-memory OPD 的 off-policy actor-learner 路线；生产入口为 `launch_offpolicy.sh`，默认 4×H20 部署为 GPU0 跑单进程 learner、GPU1/GPU2/GPU3 各 1 个 collector；确认服务器允许单卡多 CUDA 进程后，可手动调 `COLLECTORS_PER_GPU=2/3`。collector 不进 DDP/NCCL，只异步用 LoRA snapshot rollout 并写 `replay/ready/*.jsonl`；learner 不进 DDP/NCCL，单进程随机读取 replay 做 teacher-forced loss/backward，并周期发布 `latest_lora/v_<step>/`。`learn.py` 日志/TB 记录 `replay_ready/replay_pending/replay_failed/wait_events/wait_total` 与 `train/replay/*`，用于判断 collector 和 learner 谁是吞吐瓶颈。`replay.py` 负责 trajectory schema、原子写、文件锁 counter、FIFO 驱逐；`collect.py` 负责 Phase A 50% 正确初始化、Phase B 0.15 噪声扰动、teacher/student generate 和 trajectory 写盘；`learn.py` 负责 replay 采样、无 generate 的 loss/backward、checkpoint/final/snapshot；`train.py` / `train.sh` 仅保留为 on-policy 兼容调试入口，生产训练不要走它。自定义 KV decode 已本地化到 `qwen3vl_local/mrope_utils.py`，`test_kv_vs_native.py` 对比本地增量 KV 与全量无 cache / 原生 generate；旧 bug 污染过的 v4 checkpoint 需作废后重训。三步 student prompt 与 teacher target 共用 `Scene Description` / `Critical Object Description` / `Reasoning on Intent` / `Memory Judgment` 四个公开 heading；step1 student 只读 road-only memory，step2/3 才读完整 memory，teacher 可看 answer 字段但 teacher prompt 不列 label 占位符，标签由脚本追加并清洗成学生视角。scene 训练标签使用 canonical 口径：`EnterActorFlowV2 -> EnterActorFlow`、`MergerIntoSlowTrafficV2 -> MergerIntoSlowTraffic`，原始 CARLA scenario 仅保留在 `scenario/raw_gt_scene` 元数据中。`inspect_teacher.py` 是离线老师抽检脚本：随机采样 episode × 帧 × 5 种 memory 模式（all_keep / rs_change / scene_change_same_rs / event_change / scene_change_cross_rs），先做 prompt contract 自检，再 lazy import torch/model runtime，全程 `disable_adapter` 走 frozen base Qwen，逐 step 记录 teacher-private prompt/raw、student-facing prompt、adapter-enabled student 初始输出、supervised target 与 token 统计，产物为 `teacher_report.md` + `teacher_report.jsonl`，供人工评估老师推理质量并指导 prompt 迭代。 |
| `AutoMoT/qwen3vl_local/sft_v3/prompts.py` / `AutoMoT/qwen3vl_local/sft_v4/prompts.py` | v3/v4 prompt 同步硬约束：`sft_v4/prompts.py` 是唯一 prompt、Memory、状态机、target span 源；`sft_v3/prompts.py` 只能 re-export v4 并保留兼容别名。v3 是 offline on-policy OPSD：student rollout 更新 memory，`disable_adapter()` privileged teacher logits 对同一批 student step token 做 forward-KL 分布监督；v4 是 off-policy actor-learner/replay 路线。任何 prompt 或状态机改动必须同时验证 v3 和 v4。 |
| `AutoMoT/qwen3vl_local/sft_v5/` 子包 | 按用户同意新增到白名单：SFT v5 是 RS_SLOW / EVENT_FAST 双频 OPSD 路线。Q1 判断慢变量 RS，Q2 在 `[RE | REGULAR]` / `[UE | UNUSUAL]` 混合候选中逐帧判断 EVENT，不再单问 ABNORMAL。memory 是带 RS/EVENT age 的不可信 hypothesis；EVENT 是 `EVENT | RS` 条件状态，RS 变化会使旧 EVENT 失效为 UNKNOWN/age=0。Prompt 合同固定为 `sft_v5_compact_prompt_v1`：system 只放共享证据原则，user 只放短 memory/候选/一句任务/四行格式；代表性二选一预算为 system≤70、Q1≤160、Q2≤175 words，版本写入 adapter/eval/probe，禁止在多段重复同一规则。稳定 RS 默认按 3/4/5 帧可复现随机间隔复核，错误/UNKNOWN 时恢复逐帧。训练采用 aligned/omission/contradiction 课程、延迟 repair、batched rollout + parallel-KL 微批和手动 LoRA 梯度 all-reduce。完整数据过滤、prompt、KV、padding、指标、probe 与注释合同以 `SFT_V5_PLAN.md` / `SFT_V5_RUN.md` / `SFT_V5_VISUALIZATION_RECORD.md` 为准。 |
| `AutoMoT/qwen3vl_local/sft_v5/` batched Qwen 补充约束 | Q1/Q2 student rollout 允许 mixed-length padded batch；padded `past_key_values` 只用于 no-grad 采样，不写回 memory。默认保持 `QWEN_BATCH_SIZE=8`，但有 autograd graph 的 parallel KL 使用独立 `PARALLEL_KL_MICROBATCH_SIZE=2`，即 8 路 rollout 后按 2+2+2+2 teacher/student scoring 并逐微批 backward。Q2 student rollout 与 Q2 KL 都必须按精确 `q1_ids` 续接 Q1 KV 后再追加 Q2 user turn，不允许用 `q1_ids -> q1_text -> full-dialog tokenizer` 回环替代，保证采样与 scoring 上下文一致。KL forward OOM 只允许在尚未 backward 时把当前 2 路微批二分为单帧，不降低 token 上限、不重新 rollout；backward OOM 或普通异常必须中止，避免部分梯度后 fallback 重复累计。`test_batched_qwen_smoke.py --check-parallel-kl` 必须对照 single-vs-batch 生成、Q2 续接、训练 logits 和 KL loss；`test_parallel_kl_microbatch.py` 必须验证微批与 OOM 二分后的梯度等价性。显存峰值按 `KL microbatch x context length` 审计，TensorBoard 必须记录 `parallel_kl/{microbatches_per_chunk,frames_per_microbatch,oom_splits}`。必须观察 `train/q1_token_cap_hit_rate` / `train/q2_token_cap_hit_rate`；student rollout 缺少可监督 span 时必须返回 graph-connected zero。只有 `actual_batched_group_sizes` / `actual_batched_frames` 能证明真实 rollout batch；强制验证时加 `--require-batched-group`。相关代码必须保留中文注释解释 padded rollout、单样本 KV 重建、last-valid logits、padding 排除、EOS active batch 移除、KL OOM 安全二分和 TensorBoard 分母口径；`rope_deltas` 必须兼容 `(batch,1)` / `(1,batch)`。 |
| `AutoMoT/qwen3vl_local/sft_v5/` TensorBoard 补充约束 | 除 `train/loss_frame` 外，必须记录 `train/loss/{q1_analysis,q1_rs,q2_analysis,q2_event}`，其中 Q1 分项按实际触发 RS_SLOW 的 frame 平均，Q2 分项按实际进入 EVENT_FAST 的 frame 平均；还要记录 `train/rs_slow_trigger_rate` / `train/rs_reuse_fast_rate`。同时记录 `memory/{allocated,reserved,max_allocated,max_reserved}_gb`，长期显存风险以活跃引用 `allocated` 为主，不能只凭 `nvidia-smi` 或 allocator `reserved` 高水位判断泄漏。 |
| `AutoMoT/qwen3vl_local/sft_v5/` 流式优化补充约束 | 正式训练默认 `UPDATE_MODE=streaming_frames`：每个完整 global timestep 后汇总实际有效 frame，累计 `TARGET_GLOBAL_FRAMES_PER_STEP=512` 或达到 `MAX_TIMESTEPS_PER_STEP=32` 时同步 LoRA 梯度并 optimizer step；不能在同一帧 Q1/Q2/KL 中间更新。梯度按窗口实际 global frame 数归一化，optimizer step 后保留 route memory；无本地 frame 的 rank 也必须补零参加 collective，epoch 尾窗口必须 flush。LoRA 梯度按 device/dtype 合并成约 64 MiB bucket 后再 all-reduce，禁止退回数百个小参数逐个 collective。`GRAD_ACCUM` 是流式窗口倍率，`UPDATE_MODE=batch` 只作旧实验兼容；默认 learning rate 为 `1e-5`。TensorBoard 还必须记录每步 global frame/timestep、更新原因、梯度同步 bucket 数、梯度同步和 optimizer 耗时；adapter 元数据必须同时记录原始与 effective 窗口阈值、LR 和梯度同步策略。 |
| `AutoMoT/qwen3vl_local/sft_v5/` checkpoint probe 补充约束 | 正式 launcher 默认 `SAVE_STEPS=40`（约半天一版）。step 0、每个 checkpoint 和 final 都用固定 seed 测同一条完整 validation route ID，从首帧运行到末帧；`--num-routes` 控制 random 完整 ID 数，`--num-cases` 只用于 RS/UE 专项，专项默认边界前后 8 帧且 UE span 不得截断。完整 ID 首帧初始化 student/reference，随后 student RS/EVENT 只由自身 Q1/Q2 输出推进；reference 只作真值比较，禁止回写纠错，逐帧导航坐标可刷新。`results.json.memory_recovery_report` 统计 RS/UE 变化后 student 首次自行对齐的延迟。默认 review 每帧只写 `input_rgb_*.jpg`、`input.json`、`output.json`、`memory.json`；output 并列 student/teacher raw 与 parsed、teacher target、场景 GT 和正确性，memory 并列两问 student 转换与 comparison-only reference。`compact` 只写顶层 results，`full` 额外保留 legacy 深度审计文件。probe 输出目录启动时必须为空，非空直接拒绝且不自动删除；运行中保留 `.probe_in_progress.json`，只有 artifact 校验通过并原子提交 `format_version=5` 的 `results.json` 后才删除，`run_integrity` 记录 route/frame/artifact 完整性；超长/非法 scenario-route 目录名追加短哈希防碰撞。自动 probe 复用 rank0 当前 bundle，base/teacher 临时 `disable_adapter()`，checkpoint/final student 保持 LoRA；其它 rank barrier，结束恢复 train 并清 CUDA cache。256/192 token 上限只控制旁路 probe，不改变训练 1024/1024。 |
| `AutoMoT/qwen3vl_local/sft_v5/` eval/probe 指标补充约束 | 大样本 `eval.py` 与小样本 `probe.py` 共用 `metrics.py`，统一统计 RS/UE 边界、Q1/Q2 precision/recall/F1、假阳性/假阴性、端到端 EVENT 与 route macro 指标；相邻帧另外统计 RS 变化、RE->UE、UE->RE 的 TP/FP/TN/FN/invalid。`mean_resets_per_100_frames` 表示实际 GT 强制纠错，student closed-loop 测试应为 0；训练规则建议 reset 的频率单独记录。小样本变化指标在 results summary，自主恢复延迟在 `memory_recovery_report`；full 另写 transition report。eval 仅显式传 `--output-jsonl` 才落盘逐帧证据，不能把全量 prompt/output 常驻内存。 |
| `AutoMoT/qwen3vl_local/sft_v5/` memory repair 与无 GT 调度补充约束 | 正式训练默认 `RS_REPAIR_MODE=EVENT_REPAIR_MODE=ground_truth`，但只在 RS 连错 4 帧/2 帧 review 或 EVENT 连错 3 次/每帧 review 后延迟写回，禁止错误下一帧立刻纠正。`unknown` 软擦除只作消融，不作长训默认。forced-repair 后答对必须与干预前 `self_recovered_after_streak` 分开统计。eval/probe student 默认 RS/EVENT=UNKNOWN 启动，`rs_schedule_policy=deployable` 不读 GT mismatch，只用 UNKNOWN/非法输出、RS 变化确认和周期复核；`ground_truth/oracle` 只复现旧报告。离线 EVENT gate 为保持“RS 真错就跳过 EVENT”仍使用 GT correctness，必须输出 `event_gate_uses_ground_truth=true` 和 `fully_deployable_end_to_end=false`。 |
| `AutoMoT/qwen3vl_local/sft_v5/` 检查分类补充约束 | v5 检查按训练前 base 能力、grouped/parallel 等价性、训练中版本对比、训练后 adapter 可视化、静态合同五类维护；大样本 eval 单独负责总体统计。compact results 内嵌完整审计字段，review 则逐帧保存 RGB 与 input/output/memory。base 与 LoRA probe 使用同一 schema。UE 专项保留完整 span 并补边界邻帧，RS 专项保留变化点前后多帧，确保能看到延迟纠正。 |
| `AutoMoT/qwen3vl_local/sft_v5/` 显存生命周期与 teacher probe 补充约束 | 纯 batched rollout 只返回文本/token ids，不得为每个完成样本物化或返回 final KV；Q2 state 构造后立即释放旧 Q1/Q2 KV。每个 loss backward 后释放计算图引用，optimizer step 使用 `zero_grad(set_to_none=True)`；正常/异常退出统一销毁 process group 并执行一次 GC/CUDA cache 清理，但训练 step 内不得频繁 `empty_cache()`。慢帧 teacher EVENT 只在 teacher 自身 Q1 RS 正确时触发，必须续接 teacher 自己的 Q1 KV 和 RS；快帧 teacher EVENT 对当前 RGB fresh prefill，不存在独立 ABNORMAL 状态。训练用 `q2_teacher_training_prompt.txt` 与自主 teacher 的 `q2_teacher_model_prompt.txt` 分开保存，默认 `q2_teacher_prompt.txt` 必须与 `q2_teacher_output.txt` 配对。 |
| `AutoMoT/qwen3vl_local/sft_v5/` 注释与文档分工补充约束 | 每个 Python 模块需用中文 docstring 说明用法和入口，所有 class/function（含 CLI、嵌套 helper 和魔术方法）都需有中文 docstring；非显然的 padding、KV、loss 分母、DDP collective 和显存生命周期逻辑需注释设计原因，不写逐行复述。2026-07 本轮详细中文注释覆盖数据过滤/坐标转换、标签与动态候选、memory curriculum、local/global padding、batched KV/M-RoPE、精确 `q1_ids` 续接、OPSD span/KL、OOM 安全二分、global-frame 梯度归一化与分桶 all-reduce、closed-loop eval、probe 选帧与 artifact 落盘；本轮同时修正了训练 repair 统计和 eval/probe oracle 调度泄漏。代码阅读顺序固定参考 `SFT_V5_PLAN.md` §9.3：`labels.py -> prompts.py -> build_dataset.py -> train.py -> metrics.py -> eval.py -> probe.py -> tests`。后续改标签协议、prompt、memory、loss、probe 或 DDP 训练逻辑时必须同步维护相邻注释。`SFT_V5_RUN.md` 保持为精简的可执行命令手册；设计合同放在 `SFT_V5_PLAN.md`，完整 probe 产物和人工检查项放在 `SFT_V5_VISUALIZATION_RECORD.md`，不在三份文档间重复铺开。 |
| `AutoMoT/qwen3vl_local/sft_base/` 子包 | 按用户同意新增到白名单：SFT v5 的直接监督基线。复用 v5 的 collection_output 数据构建、异常 route 剔除、4 帧 RGB history、`EGO_TO_GOAL_XY`、RS/EVENT 候选池和串行 memory 状态；Q2 option-letter 扰动使用 v5 seed namespace，同 route/frame/seed 下 A/B/C 字母映射与 v5 一致；但训练不做 OPSD、不采 student rollout、不跑 privileged teacher、不输出 CoT。Q1 target 只有 `RS: <A-E>` 与 `ABNORMAL: <YES|NO>`，Q2 target 只有 `EVENT: <option>`；训练为 teacher-forced weighted CE，memory 由 GT answer 更新，作为干净直接监督 baseline，不宣称继承 v5 的 on-policy student memory 分布；eval 仍按学生输出自维护离散 memory，`EGO_TO_GOAL_XY` 每帧刷新为当前帧 ego-frame goal。默认 `LORA_VISION_SCOPE=merger`，即默认微调视觉桥接层，并启用视觉 fuse guard；eval 加载 adapter 前校验 `sft_base_adapter_config.json` 的 route/dataset/base-model/vision-scope，避免误用 v2/v5 adapter；仍可用 `off/last4/all` 做对照。运行见 `SFT_BASE_RUN.md` / `SFT_BASE_PLAN.md`。 |
| `AutoMoT/qwen3vl_local/goalgen/GOALGEN_PLAN.md` / `GOALGEN_RUN.md` / `GOALGEN_V1.md` / `GOALGEN_V2.md` | 子目标 latent 生成路线文档；PLAN/RUN 为索引，v1/v2 细节分别写入 `GOALGEN_V1.md` / `GOALGEN_V2.md`（与 goalgen 子包代码同目录） |
| `AutoMoT/qwen3vl_local/goalgen/build_dataset.py` / `train.py` / `train.sh` / `eval.py` / `probe.py` | GoalGen v1/v2 共用数据集构建 + 训练入口（DDP / 单卡 / check 三模式）+ 离线 eval（latent / pixel / velocity + PNG + TB scalar/image，支持 torchrun 分片）+ `probe.py` 随机场景 case-level dump；训练默认必须导入 `AutoMoT/checkpoints/patch_unpatch_v1/latest/weights/patch_unpatch_best.safetensors`（再兜底无 run_subdir 与最新 `run_*`）并冻结，找不到直接报错，不再随机初始化 patch/unpatch |
| `AutoMoT/qwen3vl_local/leadmot/` 子包（含 v1 训练/eval/probe 文件 + `subgoal_prompt.py`） | LEAD-MoT 快推理 decoder：route(B,10,2) + waypoint(B,8,2)，Linear+cumsum head；gen 路独立 12 层 + frozen Qwen prefix K/V attention（不过 Linear）；hidden=1024=8×128 对齐 Qwen K/V 子空间；status 按 AutoMoT velocity MLP + 共享 WaypointInputAdaptor；block 用 Qwen3 风格 RMSNorm + q/k_norm + SwiGLU；gen Q/K 按 `input_len + rope_deltas` 加 1D RoPE，language K/V 已由 Qwen prefill 带 M-RoPE 不重复旋转。v1 训练入口只训练 decoder，冻结 Qwen3-VL-Instruct 与 LeadBEVEncoder；GT 包含 route / future_waypoints 两类 ego-frame 累计点，head 内 Linear+cumsum 后直接对绝对点算 loss；`eval.py` 汇总 loss/ADE/FDE，`probe.py` 随机 case-level dump 预测与 GT 对比图。runner 必须用 `LocalQwen3VLInstructEngine` 单独跑 frozen Qwen prefill，只接受同源 HF `past_key_values`；**不能复用 AutoMoT InterleaveInferencer 的 `gen_context`**，也不保留 AutoMoT legacy slow/fast 接口；`--leadmot-ckpt` 显式加载 decoder 权重，先读 checkpoint 的 `decoder_config.use_bev` 再实例化 decoder，并 `strict=True` 加载：`use_bev=True` 必须导入已有 BEV projector 参数，`use_bev=False` 则完全不实例化 / 不 forward BEV，禁止混入随机 BEV；不传 ckpt 仅作为随机初始化链路调试。**`use_subgoal`（离线专用）**：与 `use_bev` 正交的 prefix-only 开关，开启时 build_dataset `--with-subgoal-fields` 反查 `keyframes_all_scenarios.json` 写入 scenario/run_id/status/subgoal/subgoal_frame/subgoal_rgb_path/subgoal_lookup_ok 字段；训练/eval/probe 通过 `LeadMoTTrainRuntime._run_subgoal_qwen_prefill` 在 prefix 多喂 1 张 SUBGOAL stitched RGB + `[GROUND_TRUTH_STATE]` 文本块（system/user prompt 由 `leadmot/subgoal_prompt.py` 提供），prompt 内仍保留原 navigation 文本以维持 tp/ntp/final_goal 对齐；ckpt 里 `decoder_config.use_subgoal` 与训练 args 必须严格一致，cross-load 由 `_require_subgoal_match` 拒绝；state_dict 形状不受影响（subgoal 不引入新模块），但 prefix KV 分布不兼容；`mot_lead_offline_runner.py` 会按 ckpt 自动走 subgoal prefill 并要求 clip 注入 subgoal 字段，CLI demo 可通过 `--keyframes` 自动反查；eval_carla 在线 agent 暂不支持该开关，加载 use_subgoal=True ckpt 时立即 `raise NotImplementedError`。详见 `leadmot/ARCHITECTURE.md`、`leadmot/LEADMOT_PLAN.md` 与 `PROJECT_CONTEXT.md` |
| `AutoMoT/vae_standalone/train_patch_unpatch.py` | patch / unpatch 端到端图像重建训练脚本：image→VAE.encode→patch→unpatch→VAE.decode→image，VAE 冻结；产物 `patch_unpatch_*.safetensors` 可被 `DiTMoT.load_patch_unpatch` 直接加载并默认冻结。`AutoMoT/vae_standalone/` 下其它原始文件（vwm/、config/、weights/ 等）仍为只读参考，除非已单独列入白名单 |
| `AutoMoT/vae_standalone/vae_reconstruct.py` | 按用户同意新增到白名单：VAE / patch-unpatch 诊断脚本，支持 VAE-only 与 VAE+patch/unpatch 两种重建链路、批量 loss、TensorBoard 与随机小批量 PNG 对比可视化 |
| `CLAUDE.md` | 本规则文件（仅在调整规则时修改） |
| `AGENTS.md` | 通用 AI / coding agent 入口说明文件 |

SFT v5 的上表 `BELIEVED_*` 名称已废弃：Q1 使用
`PREVIOUS_RS_HYPOTHESIS + PREVIOUS_RS_HYPOTHESIS_AGE + MEMORY_RELIABILITY + EGO_TO_GOAL_XY`，
Q2 才额外加入 `PREVIOUS_EVENT_HYPOTHESIS + PREVIOUS_EVENT_HYPOTHESIS_AGE`；memory
必须明确是可能过期或错误的 hypothesis。普通帧两个 age 各自在对应 label 改变时归零，
否则每个真实 4Hz 帧持续增加；重复确认同一 label 不归零。但 EVENT 是 `EVENT | RS`
条件状态：RS hypothesis 真正改变时旧 EVENT 必须失效为 UNKNOWN/age=0，只能由新 RS
gate 下的 Q2 重新建立。训练 route 首帧 RS/EVENT 分别以
0.5 概率用 GT，否则为 UNKNOWN/no-prior。新注入的 wrong/UNKNOWN 因为刚改变
hypothesis，age 必须从 0 开始；只有学生继续复制，才由后续真实帧自然形成 age>0 的
stale 样本，不能随机伪造旧 age。正确 RS memory 默认按 0.05/0.07 注入
contradiction/omission，EVENT 额外注入为 0.20/0.12。稳定正确 RS 默认以 4 帧为中心，在
3/4/5 帧中可复现随机选择下一次 RS_SLOW；中间快帧只复用 RS memory；EVENT_FAST
每个 RS gate 正确的帧都重新
分析当前 RGB。RS 错误只跳过本帧 EVENT，下一帧恢复逐帧 RS 分析，直到学生纠正
或 delayed repair 执行。RS 连续
错 4 帧后申请修复并每 2 个有效帧 review，EVENT 连续错 3 次后申请修复并每帧 review；
`rs_repair_interval` 只控制脚本兜底，与 `rs_slow_interval` 独立。正式默认在这些
patience/review 后延迟写回 GT，`unknown` 软擦除只作消融；forced repair 后答对不计为
干预前自主恢复。EVENT 的 normal/abnormal
完全由当帧 `[RE | REGULAR]` / `[UE | UNUSUAL]` 选项推导，不保存、不复用独立
ABNORMAL。EVENT wrong
扰动优先从本帧 `event_option_map` 其它可见
候选中选择，单选题无替代项时才回退全局 EVENT 表；EVENT repair/augmentation 只在
RS memory 本帧扰动后仍正确时执行。RS 变化导致 EVENT 失效时还必须清空旧 RS 语境的
EVENT streak/pending；同帧 Q2 错误从新语境 streak=1 重新累计。参数必须可从 CLI/
`train.sh` 覆盖并写入 adapter metadata。合法 Q1/Q2 最终高权重 span 只监督单个选项字符；
若存在 `RS:`/`EVENT:` 行但值是 `R4`/`RE` 等非法语义标签，严格 parser 仍拒绝且不更新
memory，但 loss 必须监督答案起始 token 以直接纠正选项格式；
训练/TensorBoard 记录 wrong-memory copy、wrong/UNKNOWN recovery、注入、强制修复、
Q1/Q2 aligned/omission/contradiction 比例、RS/EVENT age、RS 变化导致 EVENT 失效率、随机 interval 均值/方差、
input anomaly rate、error streak、因 RS 错跳过 Q2 的比例，以及由 EVENT 折叠出的 UE/RE
TP/FP/TN/FN 与 P/R/F1；eval/probe 同步输出 memory 依赖与 Q2 门控指标。
数据量审计以 42 个有效场景、7241 route、914466 帧为上限；10% validation 后约
82.3 万训练帧。恒定 GT、当帧自纠模拟中 Q1 trigger≈30.5%，Q1 relation≈
59.7/24.2/16.1，Q2 relation≈59.6/23.0/17.4；纯 memory-copy 压力测试中
Q1 trigger≈55.5%、Q2 gate≈64.0%、Q2 relation≈38.6/43.5/17.9。GT UE=15.55% 与 wrong/UNKNOWN memory 异常
不能直接相加，最终比例必须看 TensorBoard。

**其它所有文件**（`lead/` 整个目录、`AutoMoT/` 其余文件、配置等）**不准动**——
它们是用户从远程服务器同步下来的参考源码，作只读资料用。

特别提示：仓库根目录有 **`0026.json`**（用户提供的 LEAD meta.pkl 转 JSON 标准参考样本，
详见 PROJECT_CONTEXT.md §2.3）——**绝对禁止修改其内容，绝对禁止 `git add 0026.json`**。
它是固定参考"标尺"，修改或入库都会破坏历史推论的可追溯性。

仓库根目录或 `AutoMoT/lead_data` 下的 **`keyframes_all_scenarios.json`** 是远端数据参考，
默认只读且不要入库。`AutoMoT/keyframe_filter/` 下的同名文件属于该目录递归白名单，
不按这里的只读参考文件处理。

> 例外流程：如果确实有必要新建文件或在其他已有文件上打补丁（例如发现 utils 类
> 缺函数、需要新建测试脚本等），**必须先用 AskUserQuestion 或直接在对话里
> 请求用户确认**。得到同意后，按 §3 把新文件纳入 git 追踪列表。

---

## 3. Git 提交规则

仓库根目录：`c:\Users\11509\Desktop\automot_lead`
远程：`https://github.com/duguxiaohun/automot_lead.git`（main 分支）

### 拉取远程更新

当用户说“拉取远程最新代码覆盖本地”“更新到远程最新代码”或类似表达时，含义是：

- 只更新 / 覆盖 git 已跟踪代码文件；优先用 `git fetch` 后按远程分支处理 tracked 文件。
- 只有与远程 tracked 文件发生冲突或本地 tracked 改动挡住更新时，才覆盖这些 tracked 文件。
- 不要删除未跟踪文件、未跟踪目录、本地数据、权重、缓存、软链接、外部同步目录或用户放在工作区里的参考资料。
- 禁止把这类请求自动扩展成 `git clean -fd`、`git clean -ffd`、`rm -rf` 或任何清理未跟踪文件的操作。
- 如果确实需要清理未跟踪内容，必须先单独列出将删除的路径，并得到用户明确确认。

简言之：用户要的是“更新代码”，不是“清空工作区”。除非用户明确说要删除其它本地内容，否则不要动与远程 tracked 代码无关的东西。

### 默认追踪文件（git add 白名单）

- `PROJECT_CONTEXT.md`
- `CLAUDE.md`
- `AGENTS.md`
- `AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py`
- `AutoMoT/leaderboard/team_code/vlm_paradigm_a_runner.py`
- `AutoMoT/leaderboard/team_code/qwen3vl_instruct_paradigm_a_runner.py`
- `AutoMoT/leaderboard/team_code/qwen3vl_dit_goalgen_runner.py`
- `AutoMoT/leaderboard/team_code/automot_utils.py`
- `AutoMoT/Automot/team_code/automot_utils.py`
- `AutoMoT/Automot/mot/evaluation/inference.py`
- `AutoMoT/Automot/mot/modeling/automot/automot.py`
- `AutoMoT/leaderboard/team_code/mot_b2d_agent.py`
- `AutoMoT/leaderboard/team_code/display_interface.py`
- `AutoMoT/Automot/team_code/display_interface.py`
- `AutoMoT/qwen3vl_local/eval_carla/__init__.py`
- `AutoMoT/qwen3vl_local/eval_carla/EVAL_CARLA_PLAN.md`
- `AutoMoT/qwen3vl_local/eval_carla/EVAL_CARLA_RUN.md`
- `AutoMoT/qwen3vl_local/eval_carla/agent.py`
- `AutoMoT/qwen3vl_local/eval_carla/safety.py`
- `AutoMoT/qwen3vl_local/eval_carla/video_recorder.py`
- `AutoMoT/qwen3vl_local/eval_carla/visualizer.py`
- `AutoMoT/qwen3vl_local/eval_carla/scenario_picker.py`
- `AutoMoT/qwen3vl_local/eval_carla/aggregate.py`
- `AutoMoT/qwen3vl_local/eval_carla/run_eval.sh`
- `AutoMoT/qwen3vl_local/eval_carla/webapp/__init__.py`
- `AutoMoT/qwen3vl_local/eval_carla/webapp/app.py`
- `AutoMoT/qwen3vl_local/eval_carla/webapp/templates/index.html`
- `AutoMoT/qwen3vl_local/eval_carla/webapp/static/style.css`
- `AutoMoT/lead_video_tools/__init__.py`
- `AutoMoT/lead_video_tools/abnormal_duration_filter.py`
- `AutoMoT/lead_video_tools/rgb_to_video.py`
- `AutoMoT/lead_video_tools/LEAD_VIDEO_RUN.md`
- `AutoMoT/keyframe_filter/`（目录白名单，但排除 `AutoMoT/keyframe_filter/collection_output/` 输出产物）
- `AutoMoT/qwen3vl_local/__init__.py`
- `AutoMoT/qwen3vl_local/cache_utils.py`
- `AutoMoT/qwen3vl_local/engine.py`
- `AutoMoT/qwen3vl_local/image_io.py`
- `AutoMoT/qwen3vl_local/mrope_utils.py`
- `AutoMoT/qwen3vl_local/prompt_pipeline.py`
- `AutoMoT/qwen3vl_local/run_log.py`
- `AutoMoT/qwen3vl_local/tb_serve.sh`
- `AutoMoT/qwen3vl_local/goalgen/__init__.py`
- `AutoMoT/qwen3vl_local/goalgen/vae.py`
- `AutoMoT/qwen3vl_local/goalgen/prompt.py`
- `AutoMoT/qwen3vl_local/goalgen/qwen_kv.py`
- `AutoMoT/qwen3vl_local/goalgen/keyframes.py`
- `AutoMoT/qwen3vl_local/goalgen/dit.py`
- `AutoMoT/qwen3vl_local/goalgen/flow.py`
- `AutoMoT/qwen3vl_local/goalgen/build_dataset.py`
- `AutoMoT/qwen3vl_local/goalgen/train.py`
- `AutoMoT/qwen3vl_local/goalgen/train.sh`
- `AutoMoT/qwen3vl_local/goalgen/eval.py`
- `AutoMoT/qwen3vl_local/goalgen/probe.py`
- `AutoMoT/qwen3vl_local/leadmot/__init__.py`
- `AutoMoT/qwen3vl_local/leadmot/ARCHITECTURE.md`
- `AutoMoT/qwen3vl_local/leadmot/LEADMOT_PLAN.md`
- `AutoMoT/qwen3vl_local/leadmot/LEADMOT_RUN.md`
- `AutoMoT/qwen3vl_local/leadmot/build_dataset.py`
- `AutoMoT/qwen3vl_local/leadmot/train.py`
- `AutoMoT/qwen3vl_local/leadmot/train.sh`
- `AutoMoT/qwen3vl_local/leadmot/eval.py`
- `AutoMoT/qwen3vl_local/leadmot/probe.py`
- `AutoMoT/qwen3vl_local/leadmot/config.py`
- `AutoMoT/qwen3vl_local/leadmot/projectors.py`
- `AutoMoT/qwen3vl_local/leadmot/query_bank.py`
- `AutoMoT/qwen3vl_local/leadmot/heads.py`
- `AutoMoT/qwen3vl_local/leadmot/mot_block.py`
- `AutoMoT/qwen3vl_local/leadmot/decoder.py`
- `AutoMoT/qwen3vl_local/leadmot/subgoal_prompt.py`
- `AutoMoT/vae_standalone/train_patch_unpatch.py`
- `AutoMoT/vae_standalone/vae_reconstruct.py`
- `AutoMoT/qwen3vl_local/goalgen/GOALGEN_PLAN.md`
- `AutoMoT/qwen3vl_local/goalgen/GOALGEN_RUN.md`
- `AutoMoT/qwen3vl_local/goalgen/GOALGEN_V1.md`
- `AutoMoT/qwen3vl_local/goalgen/GOALGEN_V2.md`
- `AutoMoT/qwen3vl_local/sft/__init__.py`
- `AutoMoT/qwen3vl_local/sft/SFT_PLAN.md`
- `AutoMoT/qwen3vl_local/sft/SFT_RUN.md`
- `AutoMoT/qwen3vl_local/sft/build_dataset.py`
- `AutoMoT/qwen3vl_local/sft/build_teacher.py`
- `AutoMoT/qwen3vl_local/sft/train.py`
- `AutoMoT/qwen3vl_local/sft/train.sh`
- `AutoMoT/qwen3vl_local/sft/eval.py`
- `AutoMoT/qwen3vl_local/sft/probe.py`
- `AutoMoT/qwen3vl_local/sft/check_loss_mask.py`
- `AutoMoT/qwen3vl_local/sft/inspect_teacher_outputs.py`
- `AutoMoT/qwen3vl_local/sft_v2/__init__.py`
- `AutoMoT/qwen3vl_local/sft_v2/SFT_V2_PLAN.md`
- `AutoMoT/qwen3vl_local/sft_v2/SFT_V2_RUN.md`
- `AutoMoT/qwen3vl_local/sft_v2/prompts.py`
- `AutoMoT/qwen3vl_local/sft_v2/build_dataset.py`
- `AutoMoT/qwen3vl_local/sft_v2/train.py`
- `AutoMoT/qwen3vl_local/sft_v2/train.sh`
- `AutoMoT/qwen3vl_local/sft_v2/eval.py`
- `AutoMoT/qwen3vl_local/sft_v2/probe.py`
- `AutoMoT/qwen3vl_local/sft_v2/check_loss_mask.py`
- `AutoMoT/qwen3vl_local/sft_v3/__init__.py`
- `AutoMoT/qwen3vl_local/sft_v3/SFT_V3_PLAN.md`
- `AutoMoT/qwen3vl_local/sft_v3/SFT_V3_RUN.md`
- `AutoMoT/qwen3vl_local/sft_v3/prompts.py`
- `AutoMoT/qwen3vl_local/sft_v3/build_dataset.py`
- `AutoMoT/qwen3vl_local/sft_v3/train.py`
- `AutoMoT/qwen3vl_local/sft_v3/train.sh`
- `AutoMoT/qwen3vl_local/sft_v3/eval.py`
- `AutoMoT/qwen3vl_local/sft_v3/probe.py`
- `AutoMoT/qwen3vl_local/sft_v3/check_loss_mask.py`
- `AutoMoT/qwen3vl_local/sft_v3/test_memory_update.py`
- `AutoMoT/qwen3vl_local/sft_v3/test_kv_reuse.py`
- `AutoMoT/qwen3vl_local/sft_v3/test_gt_leak_filter.py`
- `AutoMoT/qwen3vl_local/sft_v4/test_kv_vs_native.py`
- `AutoMoT/qwen3vl_local/sft_v4/inspect_teacher.py`
- `AutoMoT/qwen3vl_local/sft_v5/`
- `AutoMoT/qwen3vl_local/sft_base/`

### 硬性规则

- **禁止** `git add .` / `git add -A` / `git add *`，会污染仓库
- **禁止** `git add lead/` / `git add AutoMoT/`（除了白名单里那一个具体路径）
- **禁止** `git add 0026.json`——它是 LEAD meta 参考样本，永远只读、永远不入库
- **禁止** `git add keyframes_all_scenarios.json` 或 `AutoMoT/lead_data/keyframes_all_scenarios.json`；
  `AutoMoT/keyframe_filter/` 下的同名文件属于该目录白名单，可随目录精确 add
- **禁止** `git add AutoMoT/keyframe_filter/collection_output`；该目录是本地自动调研输出，
  已从白名单移除，不入库、不 push
- 每次 commit 前先 `git status` 确认改动只在白名单内；如果发现别的文件有改动 →
  停下来问用户，不要 commit
- 不要执行 `git push --force` 之类的破坏性操作
- 不要 `git config` 修改用户配置

### 标准提交流程

```bash
# 1. 确认改动范围
git status

# 2. 精确 add 白名单文件（举例）
git add PROJECT_CONTEXT.md
git add AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py
git add AutoMoT/qwen3vl_local/sft_v5/
git add AutoMoT/qwen3vl_local/sft_base/

# 3. commit
git commit -m "<一句话说明本次改了什么、为什么>"

# 4. push
git push
```

### 当用户同意新建/修改白名单外文件时

1. 在本文件 §3 的"默认追踪文件"列表里**添加新文件**
2. 在 `AGENTS.md` 的文件修改范围 / git 规则里同步添加同一个文件
   - 若新增文件位于 `AutoMoT/keyframe_filter/` 下且不在 `collection_output/` 内，无需逐文件更新白名单
3. 把新文件一并 `git add`
4. commit message 注明"按用户同意新增 XXX"

### 当修改 AI 规则文档时

- 修改 `CLAUDE.md` 时必须检查并同步 `AGENTS.md`
- 修改 `AGENTS.md` 时必须检查并同步 `CLAUDE.md`
- 如果新增需要后续 agent 记住的项目事实，优先写入 `PROJECT_CONTEXT.md`；同时在
  `CLAUDE.md` / `AGENTS.md` 加入口提醒或索引
- 提交时精确执行：`git add CLAUDE.md AGENTS.md PROJECT_CONTEXT.md`（只 add 实际改动过的文件）

---

## 4. 不要做的事

- **不要执行 `lead/scripts/*.sh`、`AutoMoT/test.sh`、`AutoMoT/start_carla.sh` 等仿真脚本**——
  本机没有 CARLA、没有 LEAD 数据集（约 TB 级）、也没有模型权重
- **不要尝试 `pip install`** lead / AutoMoT 的 requirements——大量重型依赖
  （CARLA Python API、laspy、imgaug 等）会污染本机环境
- **不要从源码大段抄代码到 PROJECT_CONTEXT.md**——文档应该是浓缩结论 + 源码锚点
  （`[path:line](path#Lxxx)`），不是源码副本
- **不要替用户决定是否 push**——commit 可以自己做，push 之前问一下
  （push 一旦发到 main，外部可见，难撤回）
- **不要在运行文档里直接写 `CUDA_VISIBLE_DEVICES=...` 这种裸 shell 选卡片段**。
  SFT、GoalGen、LeadMoT、VAE patch/unpatch 以及白名单 runner 的训练、eval、probe、teacher / 推理
  入口默认都自动寻找空闲 GPU，并覆盖已有 mask。
  唯一允许的 pin 写法：`GPU_IDS=0` / `GPU_IDS=0,1,2,3` 前置环境变量（白名单训练入口在
  `GPU_IDS` 非空时跳过 nvidia-smi 自动选址，直接用给定卡号）。
  禁止在文档命令里手写 `export CUDA_VISIBLE_DEVICES=...`。

---

## 5. 工作流提示

- 改 `mot_lead_offline_runner.py` 前先看 PROJECT_CONTEXT.md §7（lead vs AutoMoT
  对照表）+ §8（runner 当前不匹配点列表）
- 改完 runner 后**同步更新** PROJECT_CONTEXT.md §8 的相应条目（标记为已修复，
  或把新的不匹配点加进去）
- 写或改 SFT / GoalGen / LeadMoT / VAE 运行命令时，保持 GPU 选址规则一致：
  单进程默认 `nvidia-smi` 自动挑 1 张空闲 GPU，并覆盖已有 mask；
  `torchrun --nproc_per_node=N` 默认自动挑 N 张最空闲 GPU，并覆盖已有 mask；
  `DDP_GPU_COUNT=N` / `NPROC_PER_NODE=N` 只表示默认自动选址时需要 N 张卡，具体卡号默认由脚本自动挑；
  显式 pin 卡只写 `GPU_IDS=0`（单卡示例）或 `GPU_IDS=0,1,2,3`（4 卡示例），并紧跟在
  原单卡/多卡训练命令后作为 demo；直接 `torchrun` 的 VAE 示例仍要让 `--nproc_per_node` 与
  `GPU_IDS` 数量一致。
- `eval_carla/run_eval.sh` 的 `--num-gpus N` / `EVAL_GPU_COUNT=N` 只表示闭环评测 worker 数；
  具体 GPU id 仍由 `nvidia-smi` 自动挑空闲卡，并为每张卡分配独立 CARLA 端口槽。
- 写或改白名单内 bash launcher 时，开头必须保留 `ulimit -S -c 0 2>/dev/null || true`，
  禁用 core dump，避免工具进程异常时生成 `core.*`；新增运行入口也要继承该约定，若工作区已有
  `core.*`，不要入库，先问用户是否清理。
- 写或改训练 launcher 时，保持**防覆盖目录约定**一致（详见 PROJECT_CONTEXT.md §11）：
  在用户给的 `OUTPUT_DIR` 下再套 `run_<RUN_TAG>/` 子目录（`RUN_TAG` 默认时间戳，bash 段算一次），
  base 层维护 `latest` symlink，`NO_RUN_SUBDIR=1` 回退；共享缓存（`HF_HOME`）必须钉在 base 层、
  不进 run 子目录，避免每次重新下载。SFT 不再有 runtime_teacher_data 共享 cache（teacher 在
  train batch 内现场跑、不写盘）。
- 写或改运行文档时，默认当前目录就是远端 `AutoMoT/`。命令示例统一写相对
  `AutoMoT/` 的路径，例如 `bash qwen3vl_local/...`、`python qwen3vl_local/...`、
  `leaderboard/...`、`checkpoints/...`；不要额外写切目录步骤，也不要给
  `qwen3vl_local/...` 命令加 `AutoMoT/` 前缀。只有仓库根视角的文件白名单、git add 路径、
  或明确说明 repo root 路径时，才保留 `AutoMoT/` 前缀。
- LEAD 数据根目录统一假设在 `AutoMoT/lead_data`，也就是用户远端在 `AutoMoT/` 下
  将原始 LEAD 数据软链接后的目录。运行文档、脚本默认值和示例命令不要再写原始
  datashare 绝对路径；数据根写 `--data-root lead_data`，keyframes 写
  `--keyframes lead_data/keyframes_all_scenarios.json`。保存路径仍写
  `checkpoints/...`。
- 用户偏好：先解释思路 → 列方案优缺点 → 等用户选 → 才开始改代码。不要"先斩后奏"
- 用户用简体中文交流，代码注释也用简体中文，变量名 / 函数名保持英文
