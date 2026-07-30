# PROJECT_CONTEXT — automot_lead Compact Guide

本文只保留新会话改代码前必须知道的项目事实。细节以源码为准；不要把长源码片段复制到这里。

## 0. 项目目标

把 `lead/` 采集/训练出来的 CARLA 离线数据，整理成本地
Qwen3-VL-Instruct frozen prefill + LeadMoT / GoalGen decoder 能直接消费的输入，
并逐步分析 RGB、LiDAR、BEV、target_point、prompt 与训练分布差异。

主要战场：

- `AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py`
- `AutoMoT/leaderboard/team_code/vlm_paradigm_a_runner.py`
- `AutoMoT/leaderboard/team_code/qwen3vl_dit_goalgen_runner.py`
- `qwen3vl_local/`（从 `AutoMoT/` 当前目录看）

## 1. 目录角色

| 目录/文件 | 角色 |
|---|---|
| `lead/` | 数据采集、训练、闭环评测参考仓库。只读 |
| `AutoMoT/` | 在线驾驶仓库；当前本地改造主要放这里 |
| `AutoMoT/lead_data` | 远端 LEAD 数据软链接入口，等价于用户在 `AutoMoT/` 下执行 `ln -s /datashare/IOL4SGH/data/data/* lead_data/` 后的目录；运行命令里用相对路径 `lead_data` / `lead_data/keyframes_all_scenarios.json` |
| `AutoMoT/data/lead` | `lead_data` 对应的 route XML 根目录，由 `AutoMoT/data/data_routes` 提取整理而来；命名规范固定为 `data/lead/<Scenario>/<Town>_<route_key>.xml`。旧数字 route 使用 `Town03_route_001783.xml`，新版子编号使用 `Town12_route_1054_0.xml`，命名本身带 Town 的 legacy key 使用 `Town06_route_Town06_13.xml`，legacy key 内部带 route 编号时保留完整 key，如 `Town12_route_Town12_route15.xml`。从 `lead_data/<Scenario>/<run_id>` 找 XML 时，`Scenario` 必须取 run 的父目录；run_id 先剥末尾 `MM_DD_HH_MM_SS` 时间戳，再只在存在时剥尾部采集后缀 `_route0`，剩余部分就是 route_key；`Town12_route15` 这类 legacy key 本体里的 `route15` 不能剥，也不能要求它带 `_route0`。XML 文件名公式：`route_key` 以 `route_` 开头时用 `<Town>_<route_key>.xml`，否则用 `<Town>_route_<route_key>.xml`。2026-07-03 全量核对：`lead_data` 9715 个 run 去重后 9294 个 `(Scenario,Town,route_key)`，`data/lead` 正好 9294 个 XML，缺失 0、冗余 0、命名不规范 0、XML 解析失败 0、内容结构异常 0；XML 内 `<weathis_juncer>` 拼写已统一修正为 `<weather>`。40 个 XML 的 `data_routes` 源文件位于不同 scenario 目录（36 个 `noScenarios`、4 个 `ConstructionObstacleTwoWays`），不是缺失；另有 `ParkedObstacle/Town12_route_Town12_route15.xml` 覆盖有效并与 `lead_data/ParkedObstacle/Town12_Rep0_Town12_route15_*` 对应，但未在 `AutoMoT/data/data_routes` 找到直接源文件。使用时以 `lead_data` / `data/lead` 的 scenario 目录为准，不能把该项当作 XML 缺失。 |
| `AutoMoT/lead_video_tools/` | 按用户同意新增：LEAD 离线 RGB 视频转换工具。只读 `/datashare/IOL4SGH/data/data/<Scenario>/<run_id>/rgb/*.jpg`，按 4Hz 生成 `/data/lead_video/<Scenario>/<run_id>/{input,left,front,right}.mp4`（默认 input，`--views` 可选三视角裁剪），默认在左上角写 frame id，支持异常 route 剔除、断点续跑、ffprobe 完整性检查和 `--workers` route 级 CPU 并行（`--workers 0` 自动按 CPU 估计）；`rgb_to_video.py` 普通转换默认剔除异常时长 route；`abnormal_duration_filter.py` 按硬规则输出异常采集名单到 `lead_video_tools/abnormal_duration_filter/`：4Hz 下 `frames >= 361`（严格大于 1 分 30 秒 / 90s）且不在白名单内的 route 全部视为异常并写入 `abnormal_confirmed_over_90s.txt`；`BlockedIntersection` 与 `ControlLoss` 是唯一时长白名单不写入名单；`Accident`、`park*`、`dynamic*` 不再有 90-100 秒存疑段豁免；`abnormal_possible_90s_to_100s.txt` 只为旧接口兼容保留，正常应为空。凡是 `AutoMoT/keyframe_filter`、`AutoMoT/qwen3vl_local` 或其它入口使用 LEAD 数据集，都必须在构建样本、调研、probe 前先剔除这些异常 route；筛选时打印 discover + route 级进度条，两个 txt 名单只保留 `Scenario/run_id`，帧数/秒数/RGB 路径/视频目录保留在 `abnormal_duration_summary.json`；只有显式传 `rgb_to_video.py --abnormal-route-list-dir ...` 才只对筛选目录里的异常 route 生成巡检视频 |
| `AutoMoT/keyframe_filter/` | 按用户同意新增到 clean push 白名单：旧版 LEAD 关键帧选择器与新 ROAD/EVENT 语义重标注方案目录。旧 `rule_based_keyframe_filter.py` 按 scenario 固定抽 initial / 3 middle / final，主要依赖 `metas/*.pkl` 的 `dist_to_*`、speed、accel、brake，缺失时 fallback 到 bbox / RGB motion；适合作为突发事件 span 提议器和验证工具输入，不再作为最终帧级 STATUS/SUBGOAL 真值来源。`classifier_logic.txt` 是用户逐场景调研得到的道路结构与事件分类草案；`ROAD_EVENT_CLASSIFICATION_PLAN.md` 是 ROAD/EVENT canonical 总方案，已合并 ROAD_STRUCTURE 调研协议、runtime 门控和错帧回查流程；`ROAD_EVENT_CANDIDATE_MAPPING.md` 保留为 Qwen/probe 可解析的候选表。代码、方案文档、规则配置、README、HTML/CSS/JS、verification 工具和手写说明允许修改、追踪、commit 和 push；`collection_output/`、`rgb_r4_r5_audit_results/`、`keyframes_all_scenarios.json`、`R2_ROUTE_RGB_REVIEW_INDEX_*.csv`、`ROAD_EVENT_INTERRUPTED_OVERLAY_*_IDS_*.csv`、`ROAD_EVENT_INTERRUPTED_OVERLAY_IDS_SUMMARY_*.json` 都是本地数据/审计/证据产物，默认不入库、不 push，需要共享时先沉淀为文档或小型配置。 |
| `qwen3vl_local/`（`AutoMoT/` 主目录内） | 本地 Qwen3-VL-Instruct frozen prefill、prompt、GoalGen、LeadMoT；`tb_serve.sh` 是通用 TensorBoard 启动器 |
| `qwen3vl_local/sft/` | SFT 数据、训练、eval、probe（统一一套，已废弃 v1/v2 双轨与 ms-swift） |
| `qwen3vl_local/sft_v2/` | 新版 SFT v2 串行选择题路线：SCENE → STATUS/SUBGOAL，无 ANALYSIS teacher |
| `qwen3vl_local/sft_v3/` | SFT v3 offline on-policy OPSD 路线：学生自维护 memory + `disable_adapter()` privileged teacher full-vocabulary logits 分布监督；v3 不维护独立 prompt，只 re-export v4 prompt / Memory / 状态机 / target span；δ 允许 0 且只封顶 10，`EGO_TO_GOAL_XY` 严格来自 meta `next_target_points[-1]`，帧末预取下一帧 goal，step3 触发统一走 `should_trigger_step3`；多卡训练采用 work-stealing + local-SGD（TCPStore 抢 episode、NCCL collective 前先 TCPStore rendezvous、先广播 rank0 LoRA 初始权重、按本轮 optimizer step 数加权平均 LoRA 参数，sync 后保存 averaged checkpoint；sync 日志/TB 记录 `all_rank_steps`、`round_eps`、`total_eps`），不再 DDP 分片或截断尾部；运行看 `SFT_V3_RUN.md` |
| `qwen3vl_local/sft_v4/` | SFT v4 off-policy actor-learner 路线：`launch_offpolicy.sh` 默认四卡部署为 GPU0 单进程 learner + GPU1/GPU2/GPU3 各 1 个异步 collector；collector 不进 DDP/NCCL，只用 adapter snapshot 采集 sequence-memory rollout 并写 `replay/ready/*.jsonl`；learner 不进 DDP/NCCL，单进程随机读取 replay 做 teacher-forced loss/backward，并周期发布 `latest_lora/v_<step>/`。确认服务器允许单卡多 CUDA 进程后，可手动调 `COLLECTORS_PER_GPU=2/3`；`learn.py` 日志/TB 记录 `replay_ready/replay_pending/replay_failed/wait_events/wait_total` 与 `train/replay/*`，用于判断 collector 和 learner 谁是吞吐瓶颈。当前 v4 使用 ROAD_STRUCTURE→SCENE→STATUS/SUBGOAL 三层 memory：Phase A 初始 `P_INIT_CORRECT=0.7`（road_structure 命中 GT 桶后 scene 同桶 50% 正确）、Phase B 噪声率 0.15、上一帧 step1 后 road_structure 仍未命中 GT 时下一帧帧首触发一次 skip 纠偏（scene 大概率 GT / 0.15 同桶扰动，status/subgoal 回 init）、stair-step 触发门要求上层在本帧前后都稳定正确才继续下钻（road 刚纠正不跑 step2，scene 刚纠正不跑 step3）。step1 学生 prompt 只读 road-only `[STEP1_ROAD_MEMORY]`（believed road + goal），不提前暴露 scene/status/subgoal；公共证据规则默认 keep believed memory，只有清晰可见证据矛盾才改，弱证据写 not contradicted，不编造 braking/merging/cut-in/active-flow 等隐藏线索；Step1 只看 road-layout cues，Step2 是 road bucket 内 fine-grained scene verification，Step3 明确区分当前 `STATUS` 和下一目标 `SUBGOAL`。step2/3 才读完整 `[MEMORY]`。student prompt 与 teacher target 共用四行 analysis contract（Scene Description / Critical Object Description / Reasoning on Intent / Memory Judgment），区别只是 teacher prompt 可看 answer 字段且标签由脚本追加；`build_step*_teacher_target` 必须把 analysis 清成学生视角，禁止把 `GROUND_TRUTH_*` / `ANSWER_*` / `REFERENCE_*` 私有字段名写进监督文本。replay schema 为 `sft_v4_rollout_v2`，显式保存 `memory_after_step1`，learner 重放 step2 必须用该 memory 构造收窄后的 SCENE_CHOICES；旧 v1 trajectory 会被拒收。`inspect_teacher.py` 默认 4 种常规模式，`scene_change_cross_rs` 为显式 stress-only，并在报告中一对一展示 teacher-private prompt/raw、student-facing prompt、adapter-enabled student 初始输出、target/memory transition。`replay.py` / `collect.py` / `learn.py` / `launch_offpolicy.sh` 已实现；`train.py` / `train.sh` 仅为 on-policy 兼容调试入口。运行见 `SFT_V4_RUN.md` |
| `qwen3vl_local/sft_v5/` | SFT v5 RS_SLOW / EVENT_FAST 双频 OPSD 路线：Q1 用当前 RGB 和不可信 RS hypothesis 判断慢变量 RS；Q2 在 RS gate 正确时逐帧重新读取 RGB，从 `[RE | REGULAR]` / `[UE | UNUSUAL]` 混合候选判断 EVENT，不再单问 ABNORMAL。Q1/Q2 memory 使用固定 schema 内的 UNKNOWN/no-prior 与错误/陈旧 hypothesis 做 aligned/omission/contradiction 课程；普通帧 RS/EVENT age 分别累计，但 EVENT 是 `EVENT | RS` 条件状态，RS hypothesis 真正变化时旧 EVENT 立即失效为 UNKNOWN/age=0，只能由新 RS gate 下的 Q2 重建。Prompt 合同为 `sft_v5_compact_prompt_v1`：system 只放共享证据原则，user 只放短 memory/候选/本题说明/四行格式，代表性二选一预算为 system≤70、Q1≤160、Q2≤175 words，版本写入 adapter/eval/probe。稳定 RS 默认以 4 帧为中心，从 3/4/5 帧中可复现随机选择下一次 RS_SLOW；错误/UNKNOWN/recovery 时逐帧慢问，RS 错的当帧跳过 EVENT。慢帧 EVENT 精确续接当帧 Q1 KV，快帧对当前 RGB fresh prefill。训练使用 torchrun 同步 on-policy OPSD、batched rollout、独立 parallel-KL 微批和手动 LoRA 梯度 all-reduce；完整数据过滤、prompt、KV、padding、指标和 probe 合同见 `SFT_V5_RUN.md` / `SFT_V5_PLAN.md` / `SFT_V5_VISUALIZATION_RECORD.md`。 |
| `qwen3vl_local/sft_v5/` batched Qwen 补充约束 | Q1/Q2 student rollout 允许 mixed-length padded batch；padded KV 只用于 no-grad 采样，不写回 memory。默认保持 `QWEN_BATCH_SIZE=8`，但有 autograd graph 的 parallel KL 使用独立 `PARALLEL_KL_MICROBATCH_SIZE=2`，即 8 路 rollout 后按 2+2+2+2 teacher/student scoring 并逐微批 backward。Q2 student rollout 和 Q2 KL 都必须按精确 `q1_ids` 续接 Q1 KV 后追加 Q2 user turn，禁止文本回环重 tokenize，保证采样与 scoring 上下文一致。KL forward OOM 只允许在尚未 backward 时二分当前微批，不降低 token 上限、不重新 rollout；backward OOM 或普通异常必须中止，避免部分梯度后 fallback 重复累计。`test_batched_qwen_smoke.py --check-parallel-kl` 验证真实模型等价性，`test_parallel_kl_microbatch.py` 验证微批/OOM 二分梯度等价性。显存峰值按 `KL microbatch x context length` 审计，TB 记录 `parallel_kl/{microbatches_per_chunk,frames_per_microbatch,oom_splits}`。相关代码必须保留中文注释解释 padded rollout、精确 Q2 KV 续接、KL OOM 安全二分和 TensorBoard 分母口径。 |
| `qwen3vl_local/sft_v5/` TensorBoard 补充约束 | 除 `train/loss_frame` 外，必须记录 `train/loss/{q1_analysis,q1_rs,q2_analysis,q2_event}`，其中 Q1 分项按实际触发 RS_SLOW 的 frame 平均，Q2 分项按实际进入 EVENT_FAST 的 frame 平均；还要记录 `train/rs_slow_trigger_rate` / `train/rs_reuse_fast_rate`、`memory/q{1,2}_relation_{aligned,omission,contradiction}_rate`、`memory/q1_rs_age_frames_mean`、`memory/q2_event_age_frames_mean`、`memory/event_invalidated_by_rs_change_rate`、`memory/rs_periodic_interval_{mean,std}`。同时记录 `memory/{allocated,reserved,max_allocated,max_reserved}_gb`，长期显存风险以活跃引用 `allocated` 为主，不能只凭 `nvidia-smi` 或 allocator `reserved` 高水位判断泄漏。 |
| `qwen3vl_local/sft_v5/` 流式优化补充约束 | 正式训练默认 `UPDATE_MODE=streaming_frames`：每个完整 global timestep 后 SUM all-reduce 实际有效 frame 数，累计 `TARGET_GLOBAL_FRAMES_PER_STEP=512` 或达到 `MAX_TIMESTEPS_PER_STEP=32` 时同步 LoRA 梯度并 optimizer step；不能在同一帧 Q1/Q2/KL 中间更新。每帧 loss 先按 effective target 缩放，梯度 SUM all-reduce 后再按窗口实际 global frame 数修正，保证 frame 等权而非 rank 等权；无本地 frame 的 rank 也必须补零梯度参与 collective。LoRA 梯度按 device/dtype 合并成约 64 MiB bucket，减少小参数逐个 NCCL collective 的同步开销。optimizer step 后保留各 route 离散 memory，epoch 尾窗口必须 flush。`GRAD_ACCUM` 是窗口倍率；`UPDATE_MODE=batch` 只作旧实验兼容。scheduler 总步数按全量训练 frame / effective target 估算，默认 LR 为 `1e-5`。TB 额外记录 `train/global_frames_per_step`、`train/timesteps_per_step`、`train/update_reason_code`、`ddp/grad_allreduce_buckets`、`time/grad_sync_seconds`、`time/optimizer_step_seconds`；adapter 元数据同时保存原始/effective 阈值、LR 和梯度同步策略。 |
| `qwen3vl_local/sft_v5/` checkpoint probe 补充约束 | launcher 默认 `SAVE_STEPS=40`，step 0、checkpoint、final 都用固定 seed 测同一条完整 validation route ID，从首帧运行到末帧；`--num-routes` 控制 random ID 数，`--num-cases` 只用于 RS/UE 专项，专项默认边界前后 8 帧且 UE span 不截断。完整 ID 首帧初始化 student/reference，之后 student memory 由 RS_SLOW/EVENT_FAST 输出推进，reference 只比较不纠错，逐帧只刷新导航坐标。`results.json.memory_recovery_report` 统计变化后 student 首次自行对齐的延迟。默认 review 每帧只写 `input_rgb_*.jpg`、`input.json`、`output.json`、`memory.json`；output 含 student/teacher raw+parsed、teacher target、场景 GT、RS_SLOW 触发原因和 EVENT KV 来源，快帧必须标记 `fresh_rgb_prefill`。`compact` 只写顶层 results，`full` 额外保留 legacy 文件。每次 probe 输出目录必须为空，非空直接拒绝且不自动删除；运行期间保留 `.probe_in_progress.json`，只有 artifact 校验通过并原子提交 `format_version=5` 的 `results.json` 后才移除，`run_integrity` 记录本次 route/frame/artifact 完整性；超长/非法 scenario-route 目录名追加短哈希防碰撞。自动 probe 复用 rank0 bundle，base/teacher 临时关闭 adapter，checkpoint/final student 使用 LoRA；其它 rank barrier，结束恢复 train 并清 cache。 |
| `qwen3vl_local/sft_v5/` eval/probe 指标补充约束 | `eval.py` 与 `probe.py` 共用 `metrics.py`，统一统计 RS/UE 边界、Q1/Q2 precision/recall/F1、FP/FN、端到端 EVENT 和 route macro；相邻帧统计 RS change、RE->UE、UE->RE 的 TP/FP/TN/FN/invalid。student closed-loop 测试中实际 GT reset 应为 0，训练规则建议 reset 的频率单独记录。小样本变化指标在 summary，自主恢复延迟在 `memory_recovery_report`；大样本 eval 默认流式累计，只有显式 `--output-jsonl` 才落盘逐帧证据。 |
| `qwen3vl_local/sft_v5/` repair / eval 去 oracle 补充约束 | 正式训练默认 `RS_REPAIR_MODE=EVENT_REPAIR_MODE=ground_truth`，但只在 RS 连错 4 帧并到 2 帧 review slot、EVENT 连错 3 次并到每帧 review slot 后延迟写回，不是错误下一帧即时纠正。`unknown` 软擦除只作消融；纯 memory-copy 压力测试中它会让 RS anomaly 约 95.7%、Q2 gate 约 4.3%，不作长训默认。forced-repair 后答对与干预前自主恢复分开统计。eval/probe student 默认 RS/EVENT=UNKNOWN 启动，deployable RS scheduler 不使用 GT mismatch，只由 UNKNOWN/非法输出、RS 变化确认和周期复核驱动；旧 GT/oracle 口径只显式复现。离线 EVENT gate 为保持“RS 真错就跳过 EVENT”仍使用 GT correctness，summary 必须写 `event_gate_uses_ground_truth=true` / `fully_deployable_end_to_end=false`。 |
| `qwen3vl_local/sft_v5/` 显存生命周期与 teacher probe 补充约束 | 纯 batched rollout 只返回文本/token ids，不物化逐样本 final KV；Q2 state 构造后立即释放旧 Q1/Q2 KV。loss backward 后释放计算图，optimizer step 使用 `zero_grad(set_to_none=True)`；正常/异常退出统一销毁 process group 并做一次 GC/CUDA cache 清理，训练 step 内不频繁 `empty_cache()`。慢帧 teacher EVENT 在 teacher 自身 Q1 RS 正确时触发并续接 teacher 自己的 Q1 KV；快帧 teacher EVENT 对当前 RGB fresh prefill。训练 privileged prompt 与 teacher 自主 prompt 分文件保存。 |
| `qwen3vl_local/sft_v5/` 注释与文档分工补充约束 | 每个 Python 模块用中文 docstring 说明用法和入口，所有 class/function（含 CLI、嵌套 helper 和魔术方法）保留中文 docstring；padding、KV、loss 分母、DDP collective 和显存生命周期等非显然逻辑需注释设计原因。2026-07 本轮详细注释覆盖数据过滤与坐标转换、标签/动态候选、Q1/Q2 memory curriculum、local/global padding、batched KV/M-RoPE、精确 `q1_ids` 续接、OPSD span/KL、forward-OOM 安全二分、global-frame 梯度归一化、分桶 all-reduce、closed-loop eval、probe 选帧与 artifact 落盘；本轮同时修正了 repair 统计与 eval/probe oracle 调度泄漏。推荐阅读顺序为 `labels.py -> prompts.py -> build_dataset.py -> train.py` 的 dataset/sampler 与单帧语义基准 -> grouped rollout/KL/optimizer -> `metrics.py -> eval.py -> probe.py -> test_*.py`，完整函数导航见 `SFT_V5_PLAN.md` §9.3。compact `results.json` 只减少文件数量，不能减少人工审计字段：每帧保存 RGB 路径、实际 student/teacher messages、完整 student/base-teacher CoT 输出、脚本化 teacher target、RS/EVENT 场景 GT、memory 和变化检测结果；base 与 LoRA probe 使用同一 schema。UE 专项必须保留一个连续 UE span 的全部 UE 帧，并按 context radius 补进入前/退出后邻帧，不得被 `num_cases` 从中间截断；RS 专项只保留变化点前后数帧。`SFT_V5_RUN.md` 只作精简命令手册，设计合同归 `SFT_V5_PLAN.md`，完整 probe 产物和人工检查项归 `SFT_V5_VISUALIZATION_RECORD.md`。 |
| `qwen3vl_local/sft_v5/` memory curriculum 补充约束 | 旧 `BELIEVED_*` 名称已废弃：Q1 使用 `PREVIOUS_RS_HYPOTHESIS + PREVIOUS_RS_HYPOTHESIS_AGE + MEMORY_RELIABILITY + EGO_TO_GOAL_XY`，Q2 才额外加入 `PREVIOUS_EVENT_HYPOTHESIS + PREVIOUS_EVENT_HYPOTHESIS_AGE`；memory 是可能过期或错误的 hypothesis。“没有 memory”使用固定 schema 内的 UNKNOWN/no-prior，不删除 block。普通帧两个 age 分别累加，对应 label 改变时归零；重复确认不归零，padding/skip 不累加。EVENT 是 `EVENT | RS`：RS hypothesis 变化会把 EVENT 失效为 UNKNOWN/age=0，并清空旧 RS 语境的 EVENT streak/pending；同帧 Q2 错误从新语境 streak=1 重新累计。新注入的 wrong/UNKNOWN 因为刚改变 hypothesis，age 必须从 0 开始；只有学生继续复制，才由后续真实帧自然形成 age>0 的 stale 样本，不能随机伪造旧 age。route 首帧 RS/EVENT 各以 0.5 概率使用 GT，否则 UNKNOWN；正确 RS memory 默认按 0.05/0.07 注入 contradiction/omission，EVENT 额外注入为 0.20/0.12。稳定 RS 默认 `rs_slow_interval=4, rs_slow_interval_jitter=1`，即按 3/4/5 帧可复现随机复核；快帧不产生 RS rollout/loss，但 gate 正确时必须产生 EVENT_FAST。RS 错误只跳过本帧 EVENT，下一帧逐帧 RS 分析，直到学生纠正或 delayed repair。RS 连错 4 帧并到 2 帧 review slot、EVENT 连错 3 次并到每帧 review slot 后才延迟修复；EVENT 不存在独立 ABNORMAL 状态。合法 Q1/Q2 最终高权重 span 只监督单个选项字符；若存在 `RS:`/`EVENT:` 行但值是 `R4`/`RE` 等非法语义标签，则严格 parser 仍拒绝且不更新 memory，但 loss 会监督答案起始 token 以直接纠正选项格式。train/eval/probe 必须记录 memory 关系、age、RS 变化导致 EVENT 失效、复制、恢复、门控及 UE/RE P/R/F1。 |
| `qwen3vl_local/sft_v5/` memory 数据量审计 | 42 个有效 `collection_output` 场景（排除 `noScenarios`）共有 7241 条 success route、914466 帧；原始 GT EVENT 中 RE=772286（84.45%）、UE=142180（15.55%）。默认 10% route-level validation 后约 82.3 万训练帧，最终值以远端 `checkpoints/sft_v5_data/summary.json` 为准。GT UE 与人为 wrong/UNKNOWN memory 是不同异常，不能相加。恒定 GT、当帧自纠模拟中 Q1 trigger≈30.5%，Q1 aligned/omission/contradiction≈59.7/24.2/16.1，Q2≈59.6/23.0/17.4；纯 memory-copy 到 delayed repair 的压力测试中 Q1 trigger≈55.5%、Q2 gate≈64.0%、Q2 relation≈38.6/43.5/17.9。最终比例以 TensorBoard 为准。 |
| `qwen3vl_local/sft_base/` | SFT v5 的直接监督基线。复用 v5 的 collection_output 数据构建、异常 route 剔除、4 帧 RGB history、`EGO_TO_GOAL_XY`、RS/EVENT 候选池和串行 memory 状态；Q2 候选顺序使用 v5 seed namespace，但本路线不再使用 A/B/C 字母，学生直接输出语义 token。训练不做 OPSD、不采 student rollout、不跑 privileged teacher、不输出 CoT；Q1 target 只有 `RS: <token>`，`ABNORMAL` 已删除，Q2 target 只有 `EVENT: <token>`，UE/RE 指标从 Q2 EVENT 折叠。训练为 teacher-forced weighted CE，memory 由 GT answer 推进，但 prompt memory 默认强扰动：RS wrong/UNKNOWN=0.30/0.40、EVENT wrong/UNKNOWN=0.35/0.35、memory dropout=0.15，转折邻域 repeat=4/window=3；route 首帧固定 UNKNOWN/UNKNOWN 并直接返回；非首帧 dropout 是独立第一层，命中就隐藏 RS/EVENT 先验并跳过其它扰动；RS unknown 时 EVENT 仍独立扰动，RS wrong 时 EVENT 在 UNKNOWN 与新 RS 自洽错误候选之间分流；R3 对抗只保留 GT=R3 -> memory=R1，不再做 R1 -> R3 反向灌入。Q1 prompt 不显示 `BELIEVED_EVENT`，只用 RS memory 和 goal；Q1 后 RS hypothesis 真正改变时旧 EVENT 立即失效为 UNKNOWN，训练侧随后为 Q2 按当前 RS 池单独重采 EVENT memory，且 keep 分支使用进入本帧前的干净 EVENT memory（上一帧 GT），避免 Q2 EVENT 边际塌成 UNKNOWN 或泄漏本帧答案。`EGO_TO_GOAL_XY` 每帧刷新，eval 默认首帧 UNKNOWN 冷启动、Q1 错也继续问 Q2，Q2 候选按学生预测/维护的 RS 生成，并输出 RS/EVENT 混淆矩阵、UE-vs-RE P/R/F1、相邻帧 RS change / RE->UE / UE->RE 的 TP/FP/FN/TN/F1/invalid、GT 稳定帧假转折率、窗口左边界已在目标值比例、转折方向 sparse confusion 和 EVENT 不可达率；`*_hit_offset_avg` 在 summary 中按 frame 显示，不再误格式化成百分比。`audit_eval_candidate_drift.py` 可量化 eval 候选过滤相对训练候选的偏差。默认 `LORA_VISION_SCOPE=last4`、rank=32、alpha=64、vision_lr_scale=0.2，并启用视觉 fuse guard；仍可用 `off/merger/all` 做对照。运行见 `qwen3vl_local/sft_base/SFT_BASE_RUN.md`。 |
| `AutoMoT/vae_standalone/train_patch_unpatch.py` | patch/unpatch 端到端重建训练 |
| `0026.json` | LEAD meta 固定参考样本，只读，绝对不要入库 |
| `keyframes_all_scenarios.json` | 仓库根目录或 `AutoMoT/lead_data` 下的远端数据参考，只读；`AutoMoT/keyframe_filter/` 下的同名文件属于该目录递归白名单 |

SFT v4 scene canonicalization rule: `EnterActorFlowV2 -> EnterActorFlow` and
`MergerIntoSlowTrafficV2 -> MergerIntoSlowTraffic`. These raw CARLA scenario
variants keep their original `scenario/raw_gt_scene` metadata, but student
`SCENE` choices, memory, teacher targets, and eval comparisons use the canonical
scene label because the paired variants share the same visible semantics and
event sequence.
SFT v4 prompt contract: teacher prompts generate only the four analysis lines
(`Scene Description`, `Critical Object Description`, `Reasoning on Intent`,
`Memory Judgment`) and never include label placeholders; `build_step*_teacher_target`
appends supervised labels and enforces the scripted KEEP/CHANGE/ADVANCE opener,
while student prompts ask the adapter to write labels on separate lines.

SFT v3/v4 prompt-sync rule: `qwen3vl_local/sft_v4/prompts.py` is the single
canonical prompt, Memory state machine, trigger helper, target-span source, and
public analysis-heading source (`Scene Description` / `Critical Object
Description` / `Reasoning on Intent` / `Memory Judgment`).
`qwen3vl_local/sft_v3/prompts.py` re-exports it and only keeps v3 compatibility
aliases. v3 is the offline on-policy OPSD route: student rollout tokens update
memory, and `eval() + no_grad + disable_adapter()` privileged teacher logits
supervise the same generated token ids that entered student KV with forward-KL
rather than hard teacher-text CE. Student text is kept unstripped only for parsing
labels/spans; step1 empty/EOS output skips the frame instead of injecting a GT
teacher target. v4 keeps the off-policy actor-learner/replay route. Any
prompt or state-machine edit must be validated on both v3 and v4.
`inspect_teacher.py` runs prompt-contract self-checks before lazy-loading torch
and model helpers; keep this order so prompt-only regressions are caught before
runtime dependency failures.

## 1.1 运行命令目录约定

运行手册默认当前目录就是远端 `AutoMoT/`。命令示例统一写相对 `AutoMoT/`
的路径，例如 `bash qwen3vl_local/...`、`python qwen3vl_local/...`、
`leaderboard/...`、`checkpoints/...`；不要额外写切目录步骤，也不要给
`qwen3vl_local/...` 命令加 `AutoMoT/` 前缀。只有仓库根视角的文件白名单、git add 路径、
或明确说明 repo root 路径时，才保留 `AutoMoT/` 前缀。

LEAD 数据根目录统一假设在 `AutoMoT/lead_data`，即远端原始 LEAD 数据软链接后的目录。
运行文档、脚本默认值和示例命令不要再写原始 datashare 绝对路径；数据根写
`--data-root lead_data`，keyframes 写 `--keyframes lead_data/keyframes_all_scenarios.json`。
训练、eval、probe 产物仍写 `checkpoints/...`。

## 2. 时间与输入约定

- CARLA 20Hz；LEAD 每 5 tick 存 1 帧，所以离线帧率 4Hz，每帧约 0.25s。
- LEAD RGB 是三视角拼接图，`PIL.size=(1152,384)`，当前本地 Qwen prefill 直接喂整图。
- LEAD `.laz` 单帧已含 5 sweep；不要额外按 AutoMoT 在线双帧融合逻辑乱拼。
- `future_positions[[5,10,...,40]]` 对应约 2s future waypoints。
- route / future_waypoints 都是 ego-frame 累计点，不是相邻 delta。

## 3. 当前路线决策

当前离线 runner 只走本地 `LocalQwen3VLInstructEngine` 做 frozen Qwen prefill，
再接 LeadMoT 或 GoalGen。已经移除 / 禁用这些旧路径：

- AutoMoT legacy `kv_cache_fixed_inference(...)`
- `InterleaveInferencer` 直接复用
- 原 fast head / `enable_fast_inference`
- `--enable-automot-slow`

原因：AutoMoT 自定义 MoT 架构和 standalone Qwen3-VL-Instruct 的 HF
`past_key_values` 不同源，不能混用。

## 4. Qwen / Prompt 规则

- `qwen` backend 只读本地 checkpoint，必须 `local_files_only=True`。
- Qwen3-VL-Instruct standalone runner 只跑
  `AutoMoT/checkpoints/Qwen3-VL-4B-Instruct`，不 import
  `vlm_paradigm_a_runner.py`。
- `prompt_pipeline.py` 是范式 A prompt / 状态机来源；改 prompt 后要同步影响 SFT v2 pending 数据。
- AutoMoT `InterleaveInferencer` / `qwen3vl_template_inference` 绑定自定义 MoT 架构，不能支撑 standalone Qwen 自由文本生成。
- Qwen3-VL 自定义 KV 增量 decode 必须走 `qwen3vl_local/mrope_utils.py` 的
  `qwen3vl_incremental_forward`，不能再依赖 `prepare_inputs_for_generation` 拼 decode
  输入。已确认 PEFT wrapper 会裁掉 `cache_position`，使每个续写 token 的 M-RoPE 位置
  退化为 0，导致 logits 从第 1 个续写 token 起大幅漂移、老师/学生生成复读，并污染
  teacher-forced loss。`engine.py`、`sft_v2/eval.py`、`sft_v3/train.py`、`sft_v3/eval.py`、
  `sft_v4/train.py`、`sft_v4/eval.py` 和 `vlm_paradigm_a_runner.py` 的文本续写路径均应
  复用该本地 helper；decode 阶段不重传图像，位置来自本条 KV state 的 `rope_deltas`。
  `engine.py` 的 `cache_system_prompt` 仅允许纯文本 suffix 复用 system-prefix cache；
  若 full input 含 `pixel_values` / `image_grid_thw` 等多模态字段，必须回退完整 prefill，
  避免把 Qwen3-VL 的图文 M-RoPE 计算拆成半截 cache 后错位。
  `_clone_cache` 必须优先保持 Transformers `Cache` 对象类型（如带 `get_mask_sizes` /
  `get_seq_length` 的新版 cache），legacy tuple 只能作为旧版兜底；新版 Qwen3-VL
  forward 会直接调用 Cache 方法，不能把它退化成普通 tuple。受旧 bug 训练出的
  SFT v4 checkpoint 和旧 `teacher_report.md` 抽检结果需要作废，修复后先重跑
  `inspect_teacher.py`，再重新训练。
- PEFT 仍可用于**加载 LoRA adapter 并立即 `merge_and_unload`**；禁止的是让 PEFT
  wrapper 参与增量 decode / `generate` 的 `prepare_inputs_for_generation` 路径。
  v3/v4 eval/probe 默认 `--merge-lora=True`，加载 adapter 后后续文本生成走 merged
  base + `qwen3vl_incremental_forward`。`engine.py` 的 adapter 自检允许
  `sft_v*_adapter_config.json` 保存完整 target module 路径，而 PEFT
  `adapter_config.json` 保存 `q_proj/down_proj/...` 这类短名；二者按 PEFT 后缀匹配语义
  判定兼容，同时继续校验视觉 LoRA scope 与权重 key，避免静默漏挂视觉 adapter。
- `LocalQwen3VLInstructEngine` 的 prefill/decode 必须在 `torch.inference_mode()` 下运行；
  v3/v4 eval/probe 的 `_generate_next_with_kv` 也必须用 inference mode 包住 suffix forward
  和 decode。否则 `--with-teacher` 的双模型诊断会在纯推理阶段构建 autograd graph，
  4 张 LEAD RGB 多次 full prefill 很容易把 80-90GB 显存吃满。每次新的 full
  `engine.generate()` 前必须先清旧 `_last_decode_state`；teacher step1/2/3 是独立专家问答，
  生成后也必须立刻清 teacher `_last_decode_state`，避免上一轮 KV 和下一轮 prefill 同时常驻。
  `--with-teacher` 本身仍会同卡常驻 student merged LoRA + base teacher 两份 Qwen，
  只适合小样本诊断。

## 5. VLM 两种范式

| 范式 | 目的 | 当前状态 |
|---|---|---|
| A：VLM 直接输出 `ANALYSIS/STATUS/SUBGOAL` | 文字状态跟踪 | standalone Qwen runner 可用；AutoMoT ckpt 不能可靠自由文本生成 |
| B：Qwen 当视觉语言编码器，decoder attend KV | 轨迹 / 子目标生成 | 当前 LeadMoT / GoalGen 主路线 |

记忆法：要文字走 standalone Qwen；要规划走 frozen prefill + decoder。

## 6. LeadMoT

文件：

- `qwen3vl_local/leadmot/train.py`
- `qwen3vl_local/leadmot/eval.py`
- `qwen3vl_local/leadmot/probe.py`
- `qwen3vl_local/leadmot/decoder.py`
- `qwen3vl_local/leadmot/LEADMOT_RUN.md`

核心结构：

- 输出 `pred_route (B,10,2)` 和 `pred_future_waypoints (B,8,2)`。
- head 是 `Linear -> cumsum`，loss 直接对累计 ego-frame 点。
- 训练冻结 Qwen3-VL-Instruct 与 LeadBEVEncoder，只训练 LeadMoT decoder。
- gen 路 12 层，hidden=1024，8 heads，head_dim=128，对齐 Qwen K/V 子空间。
- language K/V 来自 Qwen prefill，已经带 M-RoPE；LeadMoT 不重复旋转语言 K/V。

BEV 开关：

- 默认 `USE_BEV=1` / `use_bev=True`。
- `USE_BEV=0` 时 decoder 完全不实例化 / 不 forward BEV projector。
- checkpoint 加载必须二选一：`use_bev=True` 就导入已有 BEV projector 参数；`use_bev=False` 就彻底不用 BEV。禁止随机初始化 BEV projector 混入推理。
- LeadMoT `ema_state_dict` 当前 schema 是 `{"decay": ..., "shadow": {...}}`；`eval.py` /
  `probe.py` 会 unwrap `shadow` 后再 strict load。

Subgoal 开关：

- 默认 `USE_SUBGOAL=0` / `use_subgoal=False`。`USE_SUBGOAL=1` 会让 frozen Qwen prefix
  额外接收 1 张 SUBGOAL stitched RGB + `[GROUND_TRUTH_STATE]` 文本块
  （scenario/status/subgoal/event meanings），原 navigation prompt 仍保留。
- `use_subgoal` 与 `use_bev` 正交；不改变 decoder state_dict 形状，但 prefix KV 分布
  不兼容，`train.py` 的 resume/init-from-ckpt 会按 checkpoint
  `decoder_config.use_subgoal` 拒绝跨开关加载。
- `leadmot/build_dataset.py --with-subgoal-fields` 默认读取
  `keyframes_all_scenarios.json`，写入
  `run_id/subgoal_lookup_ok/status/subgoal/subgoal_frame/subgoal_rgb_path/subgoal_skip_reason`。
  subgoal 反查只接受 keyframes run status 为 `Completed/Perfect` 的轨迹，失败 run 会写
  `run_status_not_accepted:*` skip reason；`use_subgoal=True` 训练启动时只保留
  `subgoal_lookup_ok=True` 的行，过滤后无样本直接报错。
- eval / probe / `mot_lead_offline_runner.py` 从 ckpt 自动读取 `use_subgoal`。offline runner
  对 subgoal ckpt 要求 `lead_clip` 带
  `subgoal_rgb_path/subgoal_scenario/subgoal_status/subgoal_event`；CLI demo 可用
  `--keyframes` 自动反查并注入。
- `eval_carla` 在线闭环暂不支持 `use_subgoal=True`，因为在线拿不到未来 SUBGOAL keyframe RGB；
  agent 加载该类 ckpt 时立即 `raise NotImplementedError`，保留后续图像生成/代理输入 TODO。

DataLoader worker：

- LeadMoT `train.py` / `eval.py` 默认 `--num-workers 8 --prefetch-factor 2`
  （train.sh: `NUM_WORKERS=8`），把 CPU 侧 JPG/lzma/LAZ 解码预取到 worker；
  worker 默认 `multiprocessing_context=spawn`，避免 Qwen/CUDA 初始化后 fork。
- worker 只提升 GPU util，不改变 B=1 训练语义，也不提升 GPU 显存占用；真正吃满显存
  需要后续 batch>1 + prefix padding mask 改造。
- DDP train 先截掉尾部不足 `world_size` 的样本，再每 rank 取无重复等长 shard；
  val/eval 手动 `rank::world_size` 分片，避免 `DistributedSampler` padding 重复计数。
- `eval.py` 同样默认 `--num-workers 8`；`probe.py` 是小规模 case-level dump（默认 24 case），
  不接 DataLoader worker，避免为了少量样本引入额外启动开销。

### 6.1 LeadMoT CARLA 闭环评测（eval_carla）

入口从 `AutoMoT/` 当前目录看是 `qwen3vl_local/eval_carla/`，操作文档：
`EVAL_CARLA_PLAN.md` / `EVAL_CARLA_RUN.md`。

- `agent.py` 是 leaderboard 实时 agent，RGB 固定 LEAD 3cam：
  3 路 384×384、FOV=60，横拼 1152×384。
- LiDAR / radar 由 checkpoint 的 `decoder_config.use_bev` 决定：
  `use_bev=True` 才声明/读取 LEAD 双 LiDAR + 4 radar；`use_bev=False` 不产生未使用输入，
  只传空点云占位给统一 `lead_clip` 结构。
- **LEAD 训练分布对齐 (v2)**：
  - RGB：拼接后做 JPEG round-trip（`JPEG_QUALITY=85`），模拟 LEAD `.jpg` 训练数据。
  - LiDAR：轻量去地面（z 阈值 + LSQ 平面拟合，`LIDAR_REMOVE_GROUND=1`）；LEAD 用 RANSAC，
    我们因不引 numba 重依赖改用 LSQ 近似。
  - Radar：4 路 → ego frame 后拼到 LiDAR，近车 <8m 的 radar 点 duplicate 5 次
    （`USE_RADAR=1`，与 LEAD `save_radar_pc_as_lidar`+`duplicate_radar_near_ego` 同源）。
- warmup 改 LEAD 风格：第一个 4Hz 采样点（约 0.25s）就开始推理；
  历史不足时 left-pad 复制 frame 0（与 build_clip line 1808-1815 同款）。
- **target_point / next_target_point — P1 全 lookahead 弧长（用户最终路线，
  完全舍弃 P2 automot_route_index）**：
  - 训练侧 `mot_lead_offline_runner._extract_tp_route_lookahead(meta, lookahead_s, min_m=5)`
    沿 meta["route"] 弧长前推 `max(speed*lookahead_s, 5m)` 米取 ego-frame 点；
  - 在线 `agent._lookahead_world_point(speed, lookahead_s, gps, compass)` 用同款公式
    沿 RoutePlanner 剩余 route 前推；
  - **默认 tp=1.0s, ntp=2.0s**（与 wp 视野 8*0.25=2s 对齐，ntp 落 wp 末端），
    MIN_LOOKAHEAD=5m 让停车/红灯仍有方向；
  - 终点近时（route 弧长 < target_dist）tp/ntp 自动 fallback 到当前剩余 route 末端；
  - build_clip 加 `tp_mode={route_lookahead, future_truth}`（默认 route_lookahead）；
    future_truth 是 v1 兼容模式保留。
- **final_goal token 新增**（LeadMoT decoder 第 4 个 status token）：
  - 训练用 LEAD 采集器写入的 `meta["next_target_points"][-1]`，即
    `_command_planner.route` 当前剩余 command route 的真实末端，world frame 转 ego frame；
    不能再用 `meta["route"][-1]`，后者只是局部 dense route 监督片段；
  - 在线 eval_carla 用 `scenario_picker.load_route_endpoint(route_id)` 按 leaderboard
    整数 `route_id` 查旧数字 route XML（如 `data/lead/<Scenario>/Town03_route_001783.xml`）
    的最后一个 waypoint，再按当前 ego pose 转 ego frame；这是在线 220 route 的
    route_id 入口，不改变 `lead_data` 全量映射使用 `(Scenario,Town,route_key)` 的命名规则。
    找不到 XML 时才临时 fallback 到 RoutePlanner 剩余 route 末端；
  - `LeadMoTPlanningDecoderConfig.use_final_goal=True` 默认；与 tp/ntp 共享
    `WaypointInputAdaptor` MLP 让坐标语义同空间；
  - `_LEADMOT_QWEN_SYSTEM_PROMPT` + `build_cleaned_prompt_and_modes` prompt 加
    `your final destination is (X, Y)`；当前路线要求 7 元 `[speed,tp,ntp,final_goal]`，
    不再对 5 元旧输入做自动兼容；
  - **老 LeadMoT v1 ckpt 不兼容**（gen sequence 多 1 token）。train/eval/probe
    均要求 checkpoint 显式记录 `decoder_config.use_final_goal=True`。
- BEV 模型的实时 LiDAR 使用最近 `STEP_STRIDE=5` 个 20Hz sweep（≈0.25s 窗），
  按 (dx, dy, dyaw) 对齐到当前 anchor ego frame 后 concat。
- PID desired speed 用 `wp[1]` / `wp[3]`（0.5s 与 1.0s 两段距离平均），
  与 LEADMOT_PLAN.md §32 一致。
- 旧 AutoMoT 原模型里仍有 `l2_3s` / 6 个 0.5s waypoint 的 legacy 轨迹 head 注释；
  这是原 fast head 的内部监督/指标，不代表当前 tp/ntp prompt 又回到 1.5s/3s。
  当前 prompt 决策语义固定为 `now/+1s/+2s`，LeadMoT waypoint 监督固定覆盖 2s。
- `run_eval.sh` 必填 `--leadmot-ckpt`，支持 scenario / route_id / random / full；
  默认自动选 1 张空闲 GPU，`--num-gpus N` 或 `EVAL_GPU_COUNT=N` 会自动选 N 张空闲 GPU，
  每张卡一个 worker、独立端口槽，round-robin 分 route。launcher 只扫描空闲
  `[rpc..rpc+3, tm]` 端口块并实时 tail worker log；CARLA server 由
  `leaderboard_evaluator.py` 在 worker 进程内启动并清理，避免双重启动抢端口。worker log
  只落 `/tmp/leadmot_eval_workers.*` 临时目录，退出后删除，不在结果目录持久保存。
- **输出按跑法分目录**：
  - `<signature>/route<id>/` 视频与 `<signature>/eval_per_route/eval_<id>.json` 跨跑法共享
    （断点续跑）
  - `<signature>/runs/<RUN_LABEL>/` 本批次聚合，按 `full` / `scenario_X` /
    `random_NN_SK` / `routes_A+B` / `smoke_<id>` 等自动命名；始终写
    `summary_all.json`、`summary_report.md`、`scenario_table.csv`、
    `route_results.csv`、`run_manifest.json` 与 `scenarios/<Scenario>/summary.json`
    （`log.txt` 记录本批终端 stdout/stderr；`run_manifest.json` 含 started/finished 时间、attempted_count、failed_routes、worker_fail）
  - `summary_report.md` 是人类可读实验总结，解释 planned/evaluated/coverage/success_rate/
    perfect_rate/score/infractions；`scenario_table.csv` 是论文表格友好汇总，
    `route_results.csv` 是每条 route 状态/分数/违规明细
  - `<signature>/summary_all.json` 是跨批次总聚合（所有已评估 route）；有 manifest 的
    run 聚合会把计划但缺失 eval JSON 的路线记为 `MISSING_EVAL_JSON` 并计入成功率分母
- 输出 signature 包含 ckpt 父目录、ckpt stem、`bev{0|1}`、`ema{0|1}`，避免不同模型/BEV/raw-EMA 覆盖。
- 五路视频：`input` / `debug`（相机 overlay） / `bev_debug`（顶视 LiDAR+pred+tp+ego box，
  LEAD `video_recorder` BEV pseudo-image 等价）/ `demo`（spawn cinematic + 顶视 carla camera）
  / `grid`（demo+input 上下拼接）。
- 子包内 Python class/function 已补中文 docstring；shell / HTML / CSS 在关键逻辑块前有中文注释。
  后续改传感器、坐标、warmup、GPU worker、输出 JSON 或 webapp API 时，同步更新代码注释和
  `EVAL_CARLA_*` 文档。

## 7. GoalGen

文件：

- `qwen3vl_local/goalgen/build_dataset.py`
- `qwen3vl_local/goalgen/train.py`
- `qwen3vl_local/goalgen/eval.py`
- `qwen3vl_local/goalgen/probe.py`
- `qwen3vl_local/goalgen/GOALGEN_RUN.md`
- `qwen3vl_local/goalgen/GOALGEN_V1.md`
- `qwen3vl_local/goalgen/GOALGEN_V2.md`

语义：

- 输入：history RGB -> frozen Qwen prefill/KV；history/target RGB -> frozen VAE latent。
- 目标：生成未来 subgoal keyframe latent。
- 训练：rectified flow，`z_t=(1-t)z0+t z1`，预测 `v=z1-z0`。
- 推理：Euler 从 t=0 到 t=1，decode 成 RGB。
- **z0 默认为纯噪声 `z0 ~ N(0, I)`**（`z0_prior_alpha=0.0`）：
  之前默认 `alpha=1.0, sigma=1.0` 把当前帧 latent 掺进 z0 → 低 t 区域 `z_t` 主要由
  "当前帧 + 噪声"主导，`v_target = z1 - z0` 里 z0 含 z_current → 模型偷懒学
  "还原当前帧"而不是 subgoal。**统一改回纯噪声起点**（flow.py / train.py / eval.py /
  probe.py / train.sh / GOALGEN_V1.md / GOALGEN_V2.md 同步），让模型必须从噪声生成 subgoal。
  image-to-image ablation 时显式 `--z0-prior-alpha 1.0`（此时推理 `z_init` 必须用同样
  混合方式构造，分布才一致）。**v1 和 v2 共享同一份代码，默认值改动同时覆盖两者**。

v1/v2：

| 项 | v1 | v2 |
|---|---|---|
| 数据 mode | `--mode v1` | `--mode v2` |
| transition | 4 类，含 initial/final 两端 | 2 类，只保留 middle 之间 |
| 默认训练 | 从零 | 从 v1 `latest/best.pt` warm start |
| 代码 | 同一套 | 同一套 |

v2 train/eval/probe 仍复用 `goalgen/train.py` / `goalgen/eval.py` / `goalgen/probe.py`；
通过 `--val-jsonl checkpoints/goalgen_v2_data/val.jsonl`、`--save-root
checkpoints/goalgen_v2_dit` 或 `VERSION=v2` 切换。v2 train/eval/probe 都会校验样本
只能是 middle 子目标之间的转换，并拒绝明显的 `goalgen_v1_*` train/val/save/ckpt 路径
（但 v2 train 的 `INIT_FROM_CKPT=goalgen_v1_dit/...` warm start 是允许的）；误传 v1
数据时不再静默混跑 initial/final 两端样本。v2 的 counterfactual 干预默认
`counterfactual_scope=middle_transitions`：
只允许 `middle[0]→middle[1]` / `middle[1]→middle[2]`，显式排除
`initial→middle[0]` 与 `middle[2]→final`；若 v2 probe 选中两端样本会直接报错。
v2 下 `--counterfactual-scope all` 会被拒绝，内置 `--counterfactual-config default`
只生成 middle-only 候选，不复用 v1/full-scope 候选表。
GoalGen 文档已拆分：`GOALGEN_PLAN.md` / `GOALGEN_RUN.md` 只保留索引；
版本细节分别看 `GOALGEN_V1.md` 与 `GOALGEN_V2.md`。

当前共享架构，不属于某个 dataset mode：

- VAE latent `(C=16,T=1,H=48,W=144)`
- patch size `4`，token 网格 `12*36`
- hidden `1024`，heads `8`
- DiT layers `12`
- Qwen 36 层切 12 段，head_dim=128

## 8. SFT

统一文档入口：

- `qwen3vl_local/sft/SFT_PLAN.md`
- `qwen3vl_local/sft/SFT_RUN.md`
- `qwen3vl_local/sft_v2/SFT_V2_PLAN.md`
- `qwen3vl_local/sft_v2/SFT_V2_RUN.md`

**v1 / v2 双轨已废弃**（含 ms-swift 入口、loss_scale plugin、runtime_teacher_data
manifest 复用机制）。现在只有一套统一 SFT：

- `qwen3vl_local/sft/build_dataset.py` 产 `pending` jsonl，assistant 段 ANALYSIS
  为 `__TEACHER_PENDING__` 占位。
- `qwen3vl_local/sft/train.sh` → `train.py`：torch DDP + 手写 train loop +
  `peft.LoraConfig` / `get_peft_model` 直接把 LoRA 注入 base，不再走 swift。
- 每个 train batch 内部禁用 adapter，并调用底层 Qwen base model 现场 greedy
  生成 ANALYSIS 真值，立即喂进 student forward；这样避开 `PeftModel.generate`
  在 Qwen3-VL 上的生成错位问题。**不再离线物化 teacher、不再写 manifest、
  不再有 runtime_teacher_data 复用**。代价：训练时间约为 base LoRA 的 3-4 倍
  （每 step 多一次 4B 生成）。
- loss 在 `train.py` 内显式按 char-range 切段加权：ANALYSIS body 权重
  `SFT_ANALYSIS_WEIGHT`（默认 0.5，学习大致语言推理但不逐字压过状态监督）；
  `ANALYSIS:` / `\nSTATUS:` / `\nSUBGOAL:`
  字面、STATUS / SUBGOAL event_name、tail / EOS 全部 1.0；user / system prompt 段 0。
  旧版 v2 "结构字面 mask=0" 是已确认致命陷阱，新 mask 不留这个坑。
- ANALYSIS 语言学习仍走 token-level teacher 蒸馏，不引入 embedding / 偏好式
  semantic loss；如果需要减少固定措辞，训练时显式设
  `SFT_TEACHER_TEMPERATURE=0.2~0.3` 让现场 teacher 轻微改写。
- `qwen3vl_local/sft/build_teacher.py` 仅保留作可选离线工具（手动 dump teacher
  输出供 review / inspect）；不再被训练入口自动调用，也不再写 manifest。

eval 端固定坑：

- Qwen3-VL 上 PEFT wrapper forward 可能错位；`eval.py` / `probe.py` 默认
  `merge_and_unload` 把 LoRA 合并进 base 再推理。
- SFT v2/v3/v4 的 `sft_v*_adapter_config.json` 会记录完整 target module 路径用于审计；
  PEFT 自带 `adapter_config.json` 可能只保留短名 target modules。加载端按后缀兼容校验，
  不要求两份 JSON 的字符串集合逐字相等。
- `--max-gen-tokens` 默认 256（teacher ANALYSIS 80-150 token + STATUS/SUBGOAL 段，
  必须 ≥ 200，否则解析不到 STATUS）。
- partial-continue fallback 是永久兜底，不代表模型健康。
- `dataset_version="pending"` 时 GT ANALYSIS 段是占位，STATUS/SUBGOAL 评测不受
  影响；想做 ANALYSIS 内容对照，跑 `build_teacher.py` 物化 val 后再传 `--val-jsonl`。
- `eval.py` 小样本 full dump 与 `probe.py` 默认保存专家语言对照：
  `expert_analysis.txt` / `language_compare.json`。expert 来自 base teacher prompt +
  PRIVILEGED，model 来自 LoRA 自己生成的 ANALYSIS；用于检查是否学到大致语言推理，
  而不只看 STATUS/SUBGOAL。

### 8.1 SFT v2 串行选择题路线

`qwen3vl_local/sft_v2/` 是用户明确要求新增的独立路线，不替换旧 `sft/`：

- `prompts.py` 是唯一 prompt 来源。prompt 先列出全部 `SCENE_CHOICES` 及自然语言描述，
  stage-1 只要求模型输出 `SCENE`；stage-2 作为同一条对话里的后续 user prompt，
  只列出预测 scene 的 `EVENT_SEQUENCE` 和事件描述，再要求输出 `STATUS/SUBGOAL`。
  推理时 stage-2 要复用 stage-1 已经吃过图像和场景 prompt 的 KV cache。
- assistant 目标拆成两段：stage-1 `SCENE:`，stage-2 `STATUS:` / `SUBGOAL:`。没有
  ANALYSIS，没有 `__TEACHER_PENDING__`，训练时不跑 teacher.generate。
- `build_dataset.py` 复用旧 SFT 的 keyframe timeline 与 keep/advance 采样，输入仍是
  4 张 LEAD stitched RGB；`SCENE` 监督来自 scenario，`STATUS` 来自 anchor 所在 GT
  interval，`SUBGOAL` 由 `prompt_pipeline.get_full_sequence()` 推导；默认
  `--samples-per-scenario 0` 表示每个场景保留全部合法候选，正数才启用下采样；
  默认 `--wrong-scene-ratio 0.15` 只增强 train rows，把一部分 stage-2 selected scene
  替换成错误场景但仍监督真实 `STATUS/SUBGOAL`，val rows 保持 GT 分支。
- `train.py` 直接 LoRA 注入本地 Qwen3-VL-4B-Instruct；默认只注入语言侧 Linear。
  视觉侧通过 `--lora-vision-scope` / launcher `LORA_VISION_SCOPE` 选择
  `off` / `merger` / `last4` / `all` 四档；`--lora-vision` / `LORA_VISION=1`
  仅作为 `all` 的 legacy 别名保留。开启视觉 LoRA 时默认带视觉组低 LR
  (`--vision-lr-scale=0.1`，受 `--max-vision-lr-scale=0.25` 上限约束)、
  语言/视觉分组梯度裁剪 (`--language-clip-norm=1.0` / `--vision-clip-norm=0.3`)
  与 TensorBoard 梯度/参数范数观测；`merger/last4` 会解析视觉 block 编号，
  默认 `--strict-vision-scope`，解析不到 block 编号时直接报错，只有显式
  `--no-strict-vision-scope` 才退化为只训 merger/patch_embed 并 warning。
  `--vision-guard-enabled` 默认开启，连续视觉 grad/param norm 异常达到
  `--vision-guard-patience` 会停训并写 `fuse_stop_step_<N>/` + `fuse_reason.txt`，
  同时跳过正常 `final/` 保存，避免把熔断产物误当完整训练结果。
  base Qwen checkpoint 始终只读，训练产物只保存 adapter delta；adapter 目录同时写
  PEFT `adapter_config.json` 和 `sft_v2_adapter_config.json`（含 scope 与保险参数），
  eval/probe 加载前按配置判断普通 LoRA / 视觉 LoRA 并校验权重 key，不一致直接拒绝。
  prompt token 权重为 0，
  只监督 scene/status/subgoal 值 token；
  `SCENE:` / `STATUS:` / 换行等格式 token 为 0 loss。每条样本是一条多轮 teacher-forced chat：图像只在第一轮 user，
  第二轮 status prompt 作为文本 follow-up 接在 scene assistant 后，单次 forward
  里同时计算三个值 token 的 loss。
- `eval.py` 先自由生成 `SCENE`；若 scene 不在白名单，样本立即中断并计 invalid；
  若 scene 合法，即使 scene 错，也按预测 scene 的 event sequence 构造 stage-2 prompt，
  接到 stage-1 KV cache 后继续生成 `STATUS/SUBGOAL`。统计 `scene_accuracy/status_accuracy/subgoal_accuracy/all_accuracy`；
  其中 status/subgoal 主指标是串行口径，scene 错时后续即使 event token 偶然同名也算错。
  同时记录 raw exact 与 `invalid_scene_rate`、`invalid_status_for_pred_scene_rate`、
  `subgoal_not_next_rate`，用于观察串行错误传播；另记录 `valid_total` 与
  `*_valid_scene` 指标，用合法 scene 子集作分母。

## 9. VAE Patch/Unpatch

入口：`AutoMoT/vae_standalone/train_patch_unpatch.py`。

训练目标：`image -> VAE.encode -> patch -> unpatch -> VAE.decode -> image`。
VAE 冻结，只训练 patch/unpatch。产物 `patch_unpatch_*.safetensors` 可被
`DiTMoT.load_patch_unpatch` 直接加载，key 与 `self.patch` / `self.unpatch` 对齐。
默认产物路径为
`AutoMoT/checkpoints/patch_unpatch_v1/<run_TS>/weights/patch_unpatch_best.safetensors`，
base 层维护 `latest -> run_TS`；`vae_reconstruct.py` 与 GoalGen 训练共用
latest / 无 run_subdir / 最新 run_* 的兜底解析顺序。

GoalGen checkpoint 记录 patch/unpatch 来源：

- `patch_unpatch_source=external`：训练由 `PATCH_UNPATCH_WEIGHTS` /
  `--patch-unpatch-weights` 加载外部 `patch_unpatch_*.safetensors`；空参时
  `goalgen/train.py` 自动解析默认 best。找到后默认冻结，并把绝对路径写入
  `dit_config.patch_unpatch_weights`。eval/probe/runner 在 strict load DiT 后会按该
  路径再次覆盖 patch/unpatch 并恢复冻结语义。三处默认路径都找不到时直接报错，
  不再回退到随机初始化。
- `patch_unpatch_source=checkpoint`：仅用于显式 `PATCH_UNPATCH_UNFREEZE=1`
  将外部权重作为初始化并继续联合训练，或 warm start 时显式
  `PATCH_UNPATCH_CKPT_FALLBACK=1` 使用 ckpt 内自带 patch/unpatch。eval/probe/runner
  此时直接使用 `--dit-checkpoint` 内自带的 patch/unpatch。
- GoalGen eval/probe/runner 使用 EMA 时，先以完整 `dit_state_dict` 打底，再用
  `ema_state_dict` 覆盖可训练参数；外部冻结 patch/unpatch 不在 EMA shadow 中也
  不会缺 key。
- warm start 继承 `patch_unpatch_source=external` 时默认要求原 safetensors 仍存在；
  只有显式 `PATCH_UNPATCH_CKPT_FALLBACK=1` / `--allow-patch-unpatch-ckpt-fallback`
  才使用 ckpt 内自带 patch/unpatch 继续训练，并把新产物记为 `source=checkpoint`。

DDP 选卡：Python 内部 rank0 选卡，写临时文件，其它 rank 读取，避免每 worker
各自 `nvidia-smi` 导致 GPU 子集 race；前置 `GPU_IDS=0,1,2,3` 时跳过自动选卡，
直接把这组卡写入 `CUDA_VISIBLE_DEVICES`。

## 10. GPU 选址统一规则

适用：SFT、GoalGen、LeadMoT、VAE patch/unpatch、白名单 runner 的训练、
eval、probe、teacher / 推理入口。

- 默认调用 `nvidia-smi` 自动挑空闲 GPU，并覆盖外层残留的 `CUDA_VISIBLE_DEVICES`。
- 单进程默认挑 1 张；进程内通常用 `cuda:0`。
- `torchrun --nproc_per_node=N` 默认挑 N 张，并按 `LOCAL_RANK` pin。
- `DDP_GPU_COUNT=N` / `NPROC_PER_NODE=N` 只表示默认自动选址时需要 N 张卡，具体卡号默认自动挑。
- 训练脚本显式 pin 卡统一用 `GPU_IDS=0` / `GPU_IDS=0,1,2,3` 前置；非空时跳过
  `nvidia-smi` 自动选址。SFT / GoalGen / LeadMoT bash launcher 的卡数从 `GPU_IDS`
  逗号数推断，`DDP_GPU_COUNT` 被忽略；直接 `torchrun` 的 VAE 示例仍要让
  `--nproc_per_node` 与 `GPU_IDS` 数量一致。
- 文档示例不要写 shell 手动 `CUDA_VISIBLE_DEVICES=...`。
- 显式 `--device cpu` / `--device cuda:N` 的 Python 入口通常视为用户锁设备，不覆盖。
- GoalGen eval/probe 的 `--gpu N` 只在单进程下锁进程内 GPU；默认保持 0。
- `eval_carla/run_eval.sh` 用 `--num-gpus N` / `EVAL_GPU_COUNT=N` 表示闭环评测 worker 数；
  具体 GPU id 仍按 `nvidia-smi` 空闲排序自动选，并给每张卡分配独立 CARLA 端口槽。
- 白名单 bash launcher 开头统一执行 `ulimit -S -c 0 2>/dev/null || true`，禁用 core dump，
  避免工具进程异常时生成 `core.*`；新增运行入口也要加，若工作区已有 `core.*`，不要入库，先问用户是否清理。

## 11. Run 目录防覆盖规则

训练入口默认写：

```text
<OUTPUT_DIR_BASE>/run_<RUN_TAG>/
<OUTPUT_DIR_BASE>/latest -> run_<RUN_TAG>
```

规则：

- bash launcher 在启动 torchrun 前计算一次 `RUN_TAG`。
- Python VAE 入口由 rank0 生成 run tag 后 broadcast 给其它 rank。
- `NO_RUN_SUBDIR=1` 回到旧式覆盖行为，只作排查。vae 入口也接受 `NO_RUN_SUBDIR`，旧名 `PATCH_UNPATCH_NO_RUN_SUBDIR` 作为兼容别名保留。
- `HF_HOME` 挂在 base 层：`<OUTPUT_DIR_BASE>/.hf_cache`。
- `qwen3vl_local` 下训练 / eval / probe / eval_carla launcher 默认会在本次产物同目录追加
  `log.txt` 保存终端 stdout/stderr；外层 shell 已 tee 时用 `QWEN3VL_LOG_ACTIVE=1`
  防止重复记录，可用 `QWEN3VL_LOG_TO_FILE=0` 临时关闭。
- SFT 不再保留 runtime teacher cache；teacher 在 train batch 内现场生成且不写盘。

## 12. 不要做

- 不要改 `lead/`。
- 不要把 `0026.json` 或仓库根目录 / `AutoMoT/lead_data` 下的 `keyframes_all_scenarios.json` 入库。
- 不要运行 CARLA、`AutoMoT/test.sh`、`start_carla.sh`、大规模下载或安装命令。
- 不要把 AutoMoT legacy slow/fast 接口重新接回本地 Qwen/LeadMoT 路线。
- 不要把当前共享 GoalGen 架构描述成某个 dataset mode 专属架构。

## 13. 快速导航

| 任务 | 文档 |
|---|---|
| SFT 跑法 | `qwen3vl_local/sft/SFT_RUN.md` |
| SFT v2 串行选择题跑法 | `qwen3vl_local/sft_v2/SFT_V2_RUN.md` |
| SFT v3 offline OPSD 跑法 | `qwen3vl_local/sft_v3/SFT_V3_RUN.md` |
| SFT v5 RS/EVENT OPSD 跑法与可视化 | `qwen3vl_local/sft_v5/SFT_V5_RUN.md`；可视化记录见 `qwen3vl_local/sft_v5/SFT_V5_VISUALIZATION_RECORD.md` |
| SFT base 直接选项基线 | `qwen3vl_local/sft_base/SFT_BASE_RUN.md` |
| GoalGen 跑法 | `qwen3vl_local/goalgen/GOALGEN_RUN.md` 索引；版本细节看 `GOALGEN_V1.md` / `GOALGEN_V2.md` |
| LeadMoT 跑法 | `qwen3vl_local/leadmot/LEADMOT_RUN.md` |
| LeadMoT 架构 | `qwen3vl_local/leadmot/ARCHITECTURE.md` |
| LeadMoT CARLA 闭环评测 | `qwen3vl_local/eval_carla/EVAL_CARLA_RUN.md` |
| LEAD RGB 批量转视频 | `lead_video_tools/LEAD_VIDEO_RUN.md` |
| 关键帧 / ROAD-EVENT 重标注 | `keyframe_filter/ROAD_EVENT_CLASSIFICATION_PLAN.md` |
| 规则入口 | `AGENTS.md` / `CLAUDE.md` |
