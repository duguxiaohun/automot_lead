# PROJECT_CONTEXT — automot_lead Compact Guide

本文只保留新会话改代码前必须知道的项目事实。细节以源码为准；不要把长源码片段复制到这里。

## 0. 项目目标

把 `lead/` 采集/训练出来的 CARLA 离线数据，整理成本地
Qwen3-VL-Instruct frozen prefill + LeadMoT / GoalGen decoder 能直接消费的输入，
并逐步分析 RGB、LiDAR、BEV、target_point、prompt 与训练分布差异。

主要战场：

- `qwen3vl_local/`（从 `AutoMoT/` 当前目录看）
- `AutoMoT/keyframe_filter/`
- `PROJECT_CONTEXT.md`

`AutoMoT/Automot/` 与 `AutoMoT/leaderboard/team_code/` 仅作为本地参考源码，
不再由本仓库追踪或推送；下文涉及其中实现的内容只表示技术背景，不表示 Git 白名单。

## 1. 目录角色

| 目录/文件 | 角色 |
|---|---|
| `lead/` | 数据采集、训练、闭环评测参考仓库。只读 |
| `AutoMoT/` | 在线驾驶仓库；本仓库只追踪明确列入白名单的本地改造 |
| `AutoMoT/Automot/` | AutoMoT 原始实现的本地只读参考；不修改、不追踪、不 push |
| `AutoMoT/leaderboard/team_code/` | leaderboard agent/runner 的本地只读参考；不修改、不追踪、不 push |
| `AutoMoT/lead_data` | 远端 LEAD 数据软链接入口，等价于用户在 `AutoMoT/` 下执行 `ln -s /datashare/IOL4SGH/data/data/* lead_data/` 后的目录；运行命令里用相对路径 `lead_data` / `lead_data/keyframes_all_scenarios.json` |
| `AutoMoT/data/lead` | `lead_data` 对应的 route XML 根目录，由 `AutoMoT/data/data_routes` 提取整理而来；命名规范固定为 `data/lead/<Scenario>/<Town>_<route_key>.xml`。旧数字 route 使用 `Town03_route_001783.xml`，新版子编号使用 `Town12_route_1054_0.xml`，命名本身带 Town 的 legacy key 使用 `Town06_route_Town06_13.xml`，legacy key 内部带 route 编号时保留完整 key，如 `Town12_route_Town12_route15.xml`。从 `lead_data/<Scenario>/<run_id>` 找 XML 时，`Scenario` 必须取 run 的父目录；run_id 先剥末尾 `MM_DD_HH_MM_SS` 时间戳，再只在存在时剥尾部采集后缀 `_route0`，剩余部分就是 route_key；`Town12_route15` 这类 legacy key 本体里的 `route15` 不能剥，也不能要求它带 `_route0`。XML 文件名公式：`route_key` 以 `route_` 开头时用 `<Town>_<route_key>.xml`，否则用 `<Town>_route_<route_key>.xml`。2026-07-03 全量核对：`lead_data` 9715 个 run 去重后 9294 个 `(Scenario,Town,route_key)`，`data/lead` 正好 9294 个 XML，缺失 0、冗余 0、命名不规范 0、XML 解析失败 0、内容结构异常 0；XML 内 `<weathis_juncer>` 拼写已统一修正为 `<weather>`。40 个 XML 的 `data_routes` 源文件位于不同 scenario 目录（36 个 `noScenarios`、4 个 `ConstructionObstacleTwoWays`），不是缺失；另有 `ParkedObstacle/Town12_route_Town12_route15.xml` 覆盖有效并与 `lead_data/ParkedObstacle/Town12_Rep0_Town12_route15_*` 对应，但未在 `AutoMoT/data/data_routes` 找到直接源文件。使用时以 `lead_data` / `data/lead` 的 scenario 目录为准，不能把该项当作 XML 缺失。 |
| `AutoMoT/lead_video_tools/` | 按用户同意新增：LEAD 离线 RGB 视频转换工具。只读 `/datashare/IOL4SGH/data/data/<Scenario>/<run_id>/rgb/*.jpg`，按 4Hz 生成 `/data/lead_video/<Scenario>/<run_id>/{input,left,front,right}.mp4`（默认 input，`--views` 可选三视角裁剪），默认在左上角写 frame id，支持异常 route 剔除、断点续跑、ffprobe 完整性检查和 `--workers` route 级 CPU 并行（`--workers 0` 自动按 CPU 估计）；`rgb_to_video.py` 普通转换默认剔除异常时长 route；`abnormal_duration_filter.py` 按硬规则输出异常采集名单到 `lead_video_tools/abnormal_duration_filter/`：4Hz 下 `frames >= 361`（严格大于 1 分 30 秒 / 90s）且不在白名单内的 route 全部视为异常并写入 `abnormal_confirmed_over_90s.txt`；`BlockedIntersection` 与 `ControlLoss` 是唯一时长白名单不写入名单；`Accident`、`park*`、`dynamic*` 不再有 90-100 秒存疑段豁免；`abnormal_possible_90s_to_100s.txt` 只为旧接口兼容保留，正常应为空。凡是 `AutoMoT/keyframe_filter`、`AutoMoT/qwen3vl_local` 或其它入口使用 LEAD 数据集，都必须在构建样本、调研、probe 前先剔除这些异常 route；筛选时打印 discover + route 级进度条，两个 txt 名单只保留 `Scenario/run_id`，帧数/秒数/RGB 路径/视频目录保留在 `abnormal_duration_summary.json`；只有显式传 `rgb_to_video.py --abnormal-route-list-dir ...` 才只对筛选目录里的异常 route 生成巡检视频 |
| `AutoMoT/keyframe_filter/` | 按用户同意新增到 clean push 白名单：旧版 LEAD 关键帧选择器与新 ROAD/EVENT 语义重标注方案目录。旧 `rule_based_keyframe_filter.py` 按 scenario 固定抽 initial / 3 middle / final，主要依赖 `metas/*.pkl` 的 `dist_to_*`、speed、accel/brake，缺失时 fallback 到 bbox / RGB motion；适合作为突发事件 span 提议器和验证工具输入，不再作为最终帧级 STATUS/SUBGOAL 真值来源。`classifier_logic.txt` 是用户逐场景调研得到的道路结构与事件分类草案；`ROAD_EVENT_CLASSIFICATION_PLAN.md` 是 ROAD/EVENT canonical 总方案，已合并 ROAD_STRUCTURE 调研协议、runtime 门控和错帧回查流程；`ROAD_EVENT_CANDIDATE_MAPPING.md` 保留为 Qwen/probe 可解析的候选表；`ROAD_EVENT_RGB_AUDIT_ARCHIVE_202607.md` 归并 2026-07 一次性 RGB/RS/EVENT 审计记录，旧散落审计 MD 不再恢复；`COLLECTION_OUTPUT_INDEX.md` 说明 `collection_output/` 大产物、代码读取关系和白名单边界。代码、方案文档、规则配置、README、HTML/CSS/JS、verification 工具和手写说明允许修改、追踪、commit 和 push；`collection_output/` 默认仍是本地数据/审计/证据产物，不入库、不 push，唯一例外是 Phase1 四问标签轻量 JSON/JSONL：`phase1_four_question_answer_table.json`、`answer_table_partial.json`、`manual_visual_audit_notes.jsonl`、`除 no_scenarios_batch 外的 *_batch/phase1_four_question_matrix.json`、`full_route_rgb_label_review_20260809/manual_full_sheet_notes_20260809.jsonl`、`full_route_rgb_label_review_20260809/manual_table_gap_combo_notes_20260810.jsonl` 可精确 add 和 push；RGB contact sheet、montage、candidate anomalies、route/town/scenario/global summary 等证据产物仍不入库。顶层旧证据产物 `rgb_r4_r5_audit_results/`、`AutoMoT/keyframe_filter/keyframes_all_scenarios.json`、`R2_ROUTE_RGB_REVIEW_INDEX_*.csv`、`ROAD_EVENT_INTERRUPTED_OVERLAY_*_IDS_*.csv`、`ROAD_EVENT_INTERRUPTED_OVERLAY_IDS_SUMMARY_*.json` 已清理；若后续重生也默认不入库、不 push，需要共享时先沉淀为文档或小型配置。 |
| `qwen3vl_local/`（`AutoMoT/` 主目录内） | 本地 Qwen3-VL-Instruct frozen prefill、prompt、GoalGen、LeadMoT；`tb_serve.sh` 是通用 TensorBoard 启动器 |
| LeadMoT frozen Qwen adapter 合同 | `--qwen-adapter-dir` / `QWEN_ADAPTER_DIR` 把 LoRA merge 到内存中的 frozen Qwen，仍只训练 decoder；checkpoint `qwen_backbone` 绑定 base config 与 adapter 实际权重 SHA256，eval/probe/eval_carla 自动恢复并拒绝错配，旧 checkpoint 无合同时只允许 base。base/LoRA 必须同 seed 分别训练 decoder，不能临时切 prefix。 |
| `qwen3vl_local/sft/` | SFT 数据、训练、eval、probe（统一一套，已废弃 v1/v2 双轨与 ms-swift） |
| `qwen3vl_local/sft_v2/` | 新版 SFT v2 串行选择题路线：SCENE → STATUS/SUBGOAL，无 ANALYSIS teacher |
| `qwen3vl_local/sft_new_loop_phase1/` | Phase1 + Phase2 融合 YES/NO 路线：同一轮 prompt 固定回答 `HIGHWAY/STATIC_OBSTACLE/VULNERABLE/TRAFFIC_LIGHT_ABNORMAL`，并嵌入 Phase2_augment 的 `all_random_order` / `subset_random` / `hierarchical_probe` 三类 ROAD_STRUCTURE 问法；训练按 4:1:1，eval/generation 按 2:1:1。数据构建沿用 Phase2_augment 最新过滤（异常 route、full-frame RGB review 覆盖、默认剔除 visual-risk），Phase1 标签取已审计四问答案表并支持显式 visual subgroup override，但覆盖只来自结构化 RGB audit notes/annotations 中的 visual/topology subgroup，不能从自由文本 `audit_evidence` 推断 route 标签；JSONL 的 RGB 路径默认保存为相对 `--data-root` 的路径，train/eval 支持 `--data-root` 重映射旧绝对 `lead_data` 路径；Phase2 标签取逐帧 RS 标注；冲突处以 Phase2 最新 ROAD_STRUCTURE 定义作为 RS 权威，Phase1 审计标签仍作为独立可见事实标签。训练/eval 使用双层采样审计：Phase1 四问 focus YES:NO=1:1，Phase2 四问 focus (`RS1/RS2/RS4/RS5`) 也 YES:NO=1:1，并在此前提下迁移 all/subset/hierarchical augment balance key、多边际配额、variant report、answer-pattern diagnostics、subset 未问行泄漏检查、`RS_HIGHWAY` 与 `GROUP:<id>` 指标；all/subset/hierarchical 三类 variant 总量、Phase2 `(focus_bucket, variant)` 配额和 `all_random_order/RS*:YES|NO` 桶都是硬约束，subset/hierarchical 具体 augment key 逐桶偏差写入 deviation report；四个 Phase1 focus 与四个 Phase2 focus 总量 1:1。默认 `FOCUS_BALANCE_COUNT=9216`，对应每轮 147,456 sampled cases，与旧 Phase2 augment 总 case 数对齐；Phase1 桶先自然抽样，只按 all-random 的全局 RS 缺口从兼容 focus 的未用样本换入，不循环稀缺二级子桶；all-random 用容量匹配精确分配 YES/NO；默认 `MAX_TRAIN_FRAME_REPEAT=10`，任一 sampled frame 单轮复用超限会在模型加载前中止。训练 balance 记录每 epoch `balance/epoch_*.json`、窗口 `augment_counts`、`all_random_order_target_deviation`、`phase2_focus_variant_*` 和重复率审计；训练期 teacher/generation eval 与 checkpoint 默认步频为 2000/2000/20000，generation eval 默认 `generation_eval_balance_count=16`，运行见 `SFT_NEW_LOOP_PHASE1_RUN.md`。 |
| `qwen3vl_local/sft_new_loop_phase2/` | 新 Phase2 单轮 EVENT YES/NO 路线：保留 `sft_loop_phase3` 的逐帧 RGB 数据、异常 route/视觉覆盖过滤、LoRA DDP、频繁 eval/TB 与 case audit，但完全删除 synthetic Phase2 ROAD_STRUCTURE user/assistant 和 KV prefix。实际对话严格只有 system + 当前 RGB/历史 RGB + 一个 EVENT user prompt；`question_domain` 仅为数据采样/审计元数据，不向模型泄露已回答 RS。道路域问 UE1/UE3/UE5，路口域问 UE6，每题同时输出 `INVALID_EVENT_CONTEXT`；UE1:UE3:UE5:UE6 在 train/val/test 精确 1:1:1:1，RE 默认数量等于一个 UE 桶，且默认 25% 专门来自 R3/highway 作为 valid all-NO hard negative。invalid 默认约占基础有效样本的 20%，只由道路域与路口域明确错配构造，并要求所有 UE=NO、invalid=YES；同域 RE、拥堵、弱证据、无目标 UE、高速道路均保持 valid。UE2/UE4/UE7 由新 Phase1 承担，UE8 折入同域 RE。RGB 路径相对 `--data-root` 保存，train/eval 支持重映射；teacher-forced loss eval / generation eval / checkpoint 步频为 2000/2000/20000。默认 generation eval 每桶 32 条；UE3 recall 默认门槛为 0，只统计不阻断 checkpoint/流水线，`run_full_pipeline.sh` 在有效的 `best_generation/`（含 adapter 配置）存在时优先使用它继续完整 eval 和压缩；否则使用本轮 `final/`，即使没有 `best_generation` 也不得停在训练结束。旧 v3 冻结 multi-seed/unseen、UE3 rescore/route-balance 与失败 adapter 的 LeadMoT A/B 可执行链已删除；历史成绩文档只作证据。历史严格可比基线中，v3 production/audit exact 为 `316/384` / `314/384`，v4 production 为 `308/384` 且 UE6 退化；这些成绩只作基线，当前训练/评测合同已是 v5，必须重训后才能产生新成绩。v3 保留静态事故/施工、路边停车、队列车辆和 ego 视差不能充当 UE3 证据；v4 的历史逐帧审计结论仍保存在 `V3_RETRAIN_RGB_AUDIT_20260829.md`。 源 taxonomy 中显式 `U-E3` 的 DynamicObjectCrossing/ParkingCutIn/StaticCutIn 全部保留；即使它与 R4/R5 interrupted overlay 共存也固定通过 ROAD_CORRIDOR 问组监督，不能被 RS gate 静默丢弃。自由生成严格 parser 对完整字符串校验规定顺序、恰好行数和无额外文本，任何格式违规让 format/exact 同时失败；audit 另报告非评分 `answer_only_diagnostics`，只用于拆分事件答案与 evidence 完整性。adapter eval 在载入权重前硬校验 production prompt hash、history RGB mode 和 resolve 后 base model 路径。构建索引会把实际扫描到的 scenario/Town 与 RGB review coverage 做差集；训练与评测要求 UE1/UE3/UE5/UE6/RE/INVALID 六桶齐全，截断缺桶直接失败，`focus_balance_count=0` 只按六桶最小值采样与记账。`invalid_source` 贯穿 train/eval case，采样继续按 source class 及其联合 `source+true_rs+wrong-domain` 签名分层轮转；train balance/TB、generation eval 和独立 eval 都报告 source class、true RS、错误问题域、联合签名的数量、guard 与 exact。`cases_per_bin=0` 保留全量评测行但仍执行 INVALID 签名/覆盖守卫；错例 audit 的 manifest、summary 和逐例 note 直接保存 INVALID 子组。运行见 `SFT_NEW_LOOP_PHASE2_RUN.md`。 |
| new loop Phase1/2 launcher 与 eval 共用合同 | 两个 full-pipeline 都默认 4RGB + 请求四卡，`2rgb_endpoints` 只取原四帧 history 的 `[0,3]`，也可显式按模式分别训练；训练结束默认调用各自 `eval.sh`。指定 checkpoint 后，RGB mode 只能从 adapter config 恢复，调用方不再覆盖；bundle 构建前校验所有 eval metrics 的 mode 与 checkpoint 一致，并把 mode/count/selected-indices 写入包名、README、manifest 和 adapter metadata。两个 eval 包都保留 base/LoRA production+audit 指标、数据/采样 metadata 与按错误桶压缩的真实 RGB，排除权重/checkpoint/TensorBoard，并强制不超过 30MB。 |
| `qwen3vl_local/sft_v3/` | SFT v3 offline on-policy OPSD 路线：学生自维护 memory + `disable_adapter()` privileged teacher full-vocabulary logits 分布监督；v3 不维护独立 prompt，只 re-export v4 prompt / Memory / 状态机 / target span；δ 允许 0 且只封顶 10，`EGO_TO_GOAL_XY` 严格来自 meta `next_target_points[-1]`，帧末预取下一帧 goal，step3 触发统一走 `should_trigger_step3`；多卡训练采用 work-stealing + local-SGD（TCPStore 抢 episode、NCCL collective 前先 TCPStore rendezvous、先广播 rank0 LoRA 初始权重、按本轮 optimizer step 数加权平均 LoRA 参数，sync 后保存 averaged checkpoint；sync 日志/TB 记录 `all_rank_steps`、`round_eps`、`total_eps`），不再 DDP 分片或截断尾部；运行看 `SFT_V3_RUN.md` |
| `qwen3vl_local/sft_v4/` | SFT v4 off-policy actor-learner 路线：`launch_offpolicy.sh` 默认四卡部署为 GPU0 单进程 learner + GPU1/GPU2/GPU3 各 1 个异步 collector；collector 不进 DDP/NCCL，只用 adapter snapshot 采集 sequence-memory rollout 并写 `replay/ready/*.jsonl`；learner 不进 DDP/NCCL，单进程随机读取 replay 做 teacher-forced loss/backward，并周期发布 `latest_lora/v_<step>/`。确认服务器允许单卡多 CUDA 进程后，可手动调 `COLLECTORS_PER_GPU=2/3`；`learn.py` 日志/TB 记录 `replay_ready/replay_pending/replay_failed/wait_events/wait_total` 与 `train/replay/*`，用于判断 collector 和 learner 谁是吞吐瓶颈。当前 v4 使用 ROAD_STRUCTURE→SCENE→STATUS/SUBGOAL 三层 memory：Phase A 初始 `P_INIT_CORRECT=0.7`（road_structure 命中 GT 桶后 scene 同桶 50% 正确）、Phase B 噪声率 0.15、上一帧 step1 后 road_structure 仍未命中 GT 时下一帧帧首触发一次 skip 纠偏（scene 大概率 GT / 0.15 同桶扰动，status/subgoal 回 init）、stair-step 触发门要求上层在本帧前后都稳定正确才继续下钻（road 刚纠正不跑 step2，scene 刚纠正不跑 step3）。step1 学生 prompt 只读 road-only `[STEP1_ROAD_MEMORY]`（believed road + goal），不提前暴露 scene/status/subgoal；公共证据规则默认 keep believed memory，只有清晰可见证据矛盾才改，弱证据写 not contradicted，不编造 braking/merging/cut-in/active-flow 等隐藏线索；Step1 只看 road-layout cues，Step2 是 road bucket 内 fine-grained scene verification，Step3 明确区分当前 `STATUS` 和下一目标 `SUBGOAL`。step2/3 才读完整 `[MEMORY]`。student prompt 与 teacher target 共用四行 analysis contract（Scene Description / Critical Object Description / Reasoning on Intent / Memory Judgment），区别只是 teacher prompt 可看 answer 字段且标签由脚本追加；`build_step*_teacher_target` 必须把 analysis 清成学生视角，禁止把 `GROUND_TRUTH_*` / `ANSWER_*` / `REFERENCE_*` 私有字段名写进监督文本。replay schema 为 `sft_v4_rollout_v2`，显式保存 `memory_after_step1`，learner 重放 step2 必须用该 memory 构造收窄后的 SCENE_CHOICES；旧 v1 trajectory 会被拒收。`inspect_teacher.py` 默认 4 种常规模式，`scene_change_cross_rs` 为显式 stress-only，并在报告中一对一展示 teacher-private prompt/raw、student-facing prompt、adapter-enabled student 初始输出、target/memory transition。`replay.py` / `collect.py` / `learn.py` / `launch_offpolicy.sh` 已实现；`train.py` / `train.sh` 仅为 on-policy 兼容调试入口。运行见 `SFT_V4_RUN.md` |
| `qwen3vl_local/sft_loop_phase3/` | Phase3 事件级 RS-gated 二值问答路线：从 `keyframe_filter/collection_output/*_result.json` 构建逐帧样本，先剔除异常时长 route，再用 Phase2 风格的 synthetic RS context 模拟“上一步 RS 已答对”，训练/eval 中渲染为上一轮 assistant answer 后继续问 EVENT，更贴近真实 KV 续接；`build_phase3_prompt` 默认只表示实际后一轮 user turn，不 inline Phase2，eval case 保存实际多轮 messages / phase2 user / phase2 assistant / phase3 user prompt，避免 audit 误读 inline RS context。RS1/RS2 只问 UE1/UE3/UE5，RS4/RS5 只问 UE6；RE 统一为所有 UE=NO，不再细分 regular event；UE2/UE4/UE7 由 Phase1 路线承担，UE8 默认折入 RE。数据集按 split 保持 UE1:UE3:UE5:UE6 为 1:1:1:1，默认 RE 数量等于单个 UE 桶，并额外加入约 20% wrong-RS invalid/not-applicable 样本；invalid 按 source_class / true_rs / fake_rs 均衡，R3/highway invalid 同时展开到 RS1/RS2/RS4/RS5，标签为所有 UE=NO 且 `INVALID_RS_CONTEXT=YES`。训练/eval 复用 phase2_augment 的 LoRA、TensorBoard、频繁 eval/checkpoint 与 audit case 框架，并在 metrics/TB 中记录 invalid joint/all-UE-NO 指标；prompt v2 强调弱 RGB 证据时保持 RE/all-NO、普通路口车辆不等于 UE6、事故/静态拥堵不等于 UE3、invalid 只表示 RS gate 明显不适用；训练默认 `REGULAR_FOCUS_MULTIPLIER=2.0` 只放大 RE hard negatives，UE 正类仍为 1:1:1:1，eval/generation 仍用均衡口径；DDP 训练按 global step 对齐各 rank，skip/超长样本跑短图文 DDP forward 并用 logits zero loss backward；`GRAD_ACCUM>1` 结尾残余梯度会 flush，`SAVE_STEPS` 落在累积窗口中间时 checkpoint 延迟到下一次 optimizer step 后保存；运行见 `SFT_LOOP_PHASE3_RUN.md`。 |
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
| `qwen3vl_local/sft_base/` | SFT v5 的直接监督基线。复用 v5 的 collection_output 数据构建、异常 route 剔除、4 帧 RGB history、`EGO_TO_GOAL_XY`、RS/EVENT 候选池和串行 memory 状态；Q2 候选顺序使用 v5 seed namespace，但本路线不使用 A/B/C 字母，学生直接输出语义 token。当前 `DATASET_VERSION=sft_base_rs_event_token_choice_rs_regular_mapped`：regular 不再折叠成 `RE`，且以 RS 为准做 canonical 映射，R4 任意 regular -> `SIGNAL_COMPLIANCE`，R5 任意 regular -> `PRIORITY_NEGOTIATION`，R1/R2/R3 中不属于该 RS 静态表的路口 regular -> `LANE_FOLLOWING`；原始 code 保留在 `event_code_raw` / `regular_event_codes` / `event_labels_raw` 审计字段。audit remap 只统计最终 GT 为 regular 的 pure regular 帧，UE 帧即使同时带 R-E 标注也只进入 `frames_with_regular_annotation_by_rs`，不计入 `raw_regular_remap_total`；`pure_regular_frames_by_rs` 应等于各 RS 总帧减 UE 帧。旧 A/B/C adapter、旧 `RE -> REGULAR` adapter 和旧未映射 index/adapter 必须重建/重训。训练不做 OPSD、不采 student rollout、不跑 privileged teacher、不输出 CoT；Q1 target 只有 `RS: <token>`，Q2 target 只有 `EVENT: <token>`，UE/regular 指标从 Q2 EVENT 折叠。训练为 teacher-forced weighted CE，memory 由 GT answer 推进，但 prompt memory 默认强扰动且只展示 RS/EVENT token，不重复长解释；RS 描述只写静态几何，EVENT 描述写本帧动态/规则行为，R3 regular 按稳定车道内行驶、普通主路横向跨线、正在执行匝道/连接段/分流/驶出动作拆开，并由 `check_loss_mask.py` 与 `test_prompt_snapshots.py` 守住 prompt 分工。Q1 prompt 不显示 `BELIEVED_EVENT`，Q1 后 RS hypothesis 改变会让旧 EVENT 失效，训练侧随后为 Q2 按当前 RS 池重采 EVENT memory。UE loss 按 RS 条件 UE 率 inverse-sqrt 缩放；regular loss 与 regular frame repeat 按 R-E 子类频次 inverse-sqrt 缩放，UE 子类仍按逆频率 repeat 放大长尾。eval 可省略 `--adapter-dir` 直接跑 base 零样本基线；LoRA eval 会校验 adapter config。eval Q2 候选按学生 RS 的静态候选全集生成；逐帧 `allowed_events` 只用于 GT 解析和审计，不参与 prompt 候选构造；集合与 dataset 候选相同时复用 dataset 顺序。GT EVENT 不在 dataset 自己候选表中的帧与训练 Q2 skip 对齐并单独报告；GT EVENT 在 dataset 候选中但不在学生 RS 候选中仍算模型错误。eval 默认首帧 UNKNOWN 冷启动、Q1 错也继续问 Q2，并输出 `joint_acc=P(RS 对且 EVENT 对)`、端到端全局多数 regular 下界 `event_global_majority_baseline`（永远答 `LANE_FOLLOWING`）、GT-RS oracle regular 参照 `event_regular_baseline_given_gt_rs`（全量映射后参考 76.85%，不能作为端到端门槛，当前子集 oracle majority 另列审计字段）、RS/EVENT 13 类混淆矩阵、regular 内部混淆、R3 shortcut 监控、UE-vs-regular P/R/F1、单/多候选 × regular/UE 四格 EVENT 指标、`q2_candidate_count_report`、`q2_rs_candidate_count_report`、直接按 RS 分组的 `ue_fp_on_multi_candidate_re_by_rs` / `q2_multi_re_by_rs_report` / `q2_multi_ue_by_rs_report`、raw regular 映射帧诊断 `event_raw_regular_remap_report`、相邻帧 RS change / regular->UE / UE->regular 指标和 EVENT 不可达率；`report.html` 单文件内嵌 metrics 数据并可视化 RS/EVENT/regular/UE-vs-RE 矩阵，脱离 JSON 也能直接打开；`frames.jsonl` 同步写 `gt_event_code_raw` 与 `gt_regular_remapped` 供错帧回查。默认 `LORA_VISION_SCOPE=last4`、rank=32、alpha=64、vision_lr_scale=0.2，并启用视觉 fuse guard；checkpoint 目录除 adapter 外会写 `trainer_state.pt`，`--resume-from-checkpoint` / `RESUME_FROM_CHECKPOINT` 可在原 run 目录内接着 step、TB、optimizer/scheduler/RNG 续训，resume 默认归档并修剪 `tb/` 中大于 checkpoint step 的旧 event，避免 200-300 等未来曲线污染续训；旧 checkpoint 无 state 时至少恢复 adapter 与目录名 step；仍可用 `off/merger/all` 做对照。运行见 `qwen3vl_local/sft_base/SFT_BASE_RUN.md`。 |
| `qwen3vl_local/sft_base/` RS×EVENT 静态表补充约束 | `EVENT_CANDIDATES_BY_RS` 按严格方案 A 维护 UE 组合：复用 build_dataset 的异常/缺失/失败 route 过滤后，只保留 `count >= 20` 且 `rs_frame_rate >= 0.1%` 的 RS×UE 组合；regular 使用 RS canonical 映射后的语义保守表，当前候选数固定为 R1=7、R2=5、R3=3、R4=5、R5=5，R3 不开放 UE 但保留 3 个 regular 候选。`audit_rs_event_cooccurrence.py` 必须报告 RS×UE、mapped RS×R-E regular 分布、多 raw regular 标签比例、UE/regular 两侧 missing、low-rate 和 spurious 静态组合；raw regular remap 只统计最终 GT 为 regular 的 pure regular 帧，并输出 scenario/route top-k 归因；新协议下 regular missing/spurious 应为空，GT static mismatch 应只剩严格阈值拒绝的低频 UE 组合。 |
| `qwen3vl_local/sft_base_simple/` | 按用户同意新增：从 `sft_baseline` 继续简化的 HIGHWAY/NON_HIGHWAY + RE/UE 单问直接监督基线。显式 transition 采样/API 已撤掉，训练默认先跨 route 聚合 `FOURBIN_ROUTES_PER_BATCH=16` 条 route，再按当前帧 GT 四格 `HIGHWAY:UE` / `HIGHWAY:RE` / `NON_HIGHWAY:UE` / `NON_HIGHWAY:RE` 做 exact balance，默认 `JOINT_TARGET_BALANCE_COUNT=8`、`UE_FRAME_REPEAT=1`、`UE_EVENT_LOSS_WEIGHT=1.0`、repeat mode 为 `none`，避免四格均衡后再向 UE 重复倾斜；eval 默认同样按当前帧 GT 四格随机均衡，但 joint case 会按 route 顺序闭环 rollout 到最远受评帧，只在抽中帧计 ROAD/EVENT/JOIN accuracy，change matrix 来自 rollout 相邻帧，`--initial-memory-noise none` 与 joint eval 组合会被拒绝防止 GT memory 泄漏。transition 帧只作为普通当前帧落入对应四格，不再单独抽样或 repeat。训练日志/TB 记录 balance 后四桶实际样本数与 early-UE prompt memory 的 `RE/UE/UNKNOWN/HIDDEN` 分布。基础 RS/EVENT memory wrong/UNKNOWN/dropout 概率沿用 baseline，连续 UE span 前 `MEMORY_EARLY_UE_FRAMES=4` 帧额外提高 EVENT memory wrong/UNKNOWN/dropout 与重采概率，放大后 wrong+UNKNOWN 显式归一化并在启动日志打印 effective 概率，避免模型靠 `PREVIOUS_EVENT=UE` 续答 UE。当前 `DATASET_VERSION=sft_base_simple_highway_reue_fourbin_v1`，adapter route 为 `sft_base_simple_highway_reue_fourbin_random`，运行见 `qwen3vl_local/sft_base_simple/SFT_BASE_RUN.md`。 |
| `AutoMoT/vae_standalone/train_patch_unpatch.py` | patch/unpatch 端到端重建训练 |
| `0026.json` | LEAD meta 固定参考样本，只读，绝对不要入库 |
| `keyframes_all_scenarios.json` | 仓库根目录或 `AutoMoT/lead_data` 下的远端数据参考，只读；`AutoMoT/keyframe_filter/` 下的旧副本已清理，不再恢复 |

2026-09-03 的旧 UE3 rescore、label-alignment 与 route-balance 排障链已结束，相关专用脚本和审计文档已删除；当前只保留通用错例审计与 2026-09-04 的 highway RGB 真值合同。

2026-09-04 Phase2 高速 UE3 补充：继续逐帧查看 HighwayCutIn 的完整 RGB sheets 后确认，
高速/匝道他车相对可见分道线持续横移、车头/车身跨入 ego 当前通道时仍属于原 `UE3`；
普通并行、稳定跟车和 ego 超车视差仍是 all-NO。由于源 HighwayCutIn 标注没有 U-E3，
仅改 prompt 会造成监督冲突；因此数据构建只接受
`sft_new_loop_phase2/highway_ue3_rgb_decisions_v1.jsonl` 的显式 RGB-YES span 覆盖，
scenario/R3 不自动造正例。模型输出类别不变，`HIGHWAY_CUTIN` 只作为 UE3 审计子型；
manifest、generation eval、独立 eval 与 RGB audit 同时报告总体 UE3 和
HIGHWAY_CUTIN/OTHER_UE3，默认每个 UE3 评测桶至少保留高速子型（目标比例 12.5%），RE
仍保留无 cut-in 高速负例。prompt 升级为待验收的
`sft_new_loop_phase2_direct_event_visual_v5_highway_ue3`，旧 v3/v4 adapter 与其 hash
不兼容；v3 的历史 316/384 仍只是基线，不能冒充 v5 结果。详细证据见
`sft_new_loop_phase2/HIGHWAY_UE3_RGB_AUDIT_20260904.md`。

同日全量扫描 `collection_output/*_result.json`：源 taxonomy 只有
DynamicObjectCrossing（251 帧/84 routes）、ParkingCutIn（330/80）和
StaticCutIn（770/54）显式含 U-E3，共 1351 帧；其中 33 帧是 R4 interrupted overlay。
Phase2 现保留所有显式 U-E3，并固定通过 ROAD_CORRIDOR 问组监督；没有 U-E3 的 R4/R5
仍不生成 UE3。加上 HighwayCutIn 的 73 个 RGB-YES 帧，当前兼容四类 UE3 来源。
UE3 recall 默认门槛改为 0，只统计不阻断；`run_full_pipeline.sh` 优先用通过 generation guard 的
`best_generation/`，缺失时回退训练结束的 `final/` 继续 eval 与压缩。旧 UE3 专用 rescore/label-alignment/RGB 包/
route-balance 排障脚本及对应文档已删除，避免与当前 v5 合同混用。

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

2026-08-09 `AutoMoT/keyframe_filter` 追加完成 Phase1 四问答案表的全量人工 RGB + 原始标签复核，
摘要见 `AutoMoT/keyframe_filter/PHASE1_FOUR_QUESTION_RGB_AUDIT_20260809.md`，Phase1
`collection_output` 目录索引/legacy 关系/复用流程见
`AutoMoT/keyframe_filter/PHASE1_COLLECTION_OUTPUT_INDEX.md`；本地证据位于
`AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/`，后续复核应优先复用
`full_route_rgb_label_review_20260809/` 与已有 notes，不要重新批量生成重复 RGB 目录；其中最终四问标签表、
batch matrix 和人工 JSONL notes 是轻量标签产物，已列入白名单可精确 push；RGB contact
sheet、montage、candidate anomalies 与 route/town/scenario/global summary 等证据产物仍默认不入库、不 push。
统一证据目录 `full_route_rgb_label_review_20260809/` 覆盖 42 个非 `noScenarios` 场景、
197 个 scenario-Town、582 条 route、68,073 帧、2,003 张逐帧 RGB+RS/EVENT 标签 sheet；
5 个源数据不足 3 条的 Town 已审完全部可用 route。
关键口径：`noScenarios` 排除；`HIGHWAY` 必须看匝道/出入口/导流 gore/连续隔离/受控通行等拓扑，
不能由直道、宽路、空旷或单独护栏推出；`EnterActorFlow/R1/R-E1` 与
`EnterActorFlowV2/R1/R-E1` 四问 `HIGHWAY=YES`，`InterurbanActorFlow/R3/R-E1`
四问 `HIGHWAY=NO`；`Accident/R1/{R-E1,R-E2}` 已纠正为 `HIGHWAY=NO`（旧表错误套用了
actor-flow 理由）；`ParkedObstacle × U-E2` 组合级保持 `OBSTACLE=YES`。答案表现在是 v2：
默认 `scenario × RS × EVENT` 行由 RGB 审计策略生成，`ParkedObstacle/Town12` 的受控快速路
子组只有在 route-level RGB topology 标为 `limited_access_fast_road` 时才覆盖成 `HIGHWAY=YES`，
不能仅凭 Town、场景名、宽直道路或护栏触发。

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

适用：SFT、GoalGen、LeadMoT 与 VAE patch/unpatch 的训练、
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
| SFT new loop phase2 单轮 EVENT 跑法 | `qwen3vl_local/sft_new_loop_phase2/SFT_NEW_LOOP_PHASE2_RUN.md` |
| SFT loop phase3 事件 gate 跑法 | `qwen3vl_local/sft_loop_phase3/SFT_LOOP_PHASE3_RUN.md` |
| SFT v5 RS/EVENT OPSD 跑法与可视化 | `qwen3vl_local/sft_v5/SFT_V5_RUN.md`；可视化记录见 `qwen3vl_local/sft_v5/SFT_V5_VISUALIZATION_RECORD.md` |
| SFT base 直接选项基线 | `qwen3vl_local/sft_base/SFT_BASE_RUN.md` |
| SFT base simple 四格均衡基线 | `qwen3vl_local/sft_base_simple/SFT_BASE_RUN.md` |
| GoalGen 跑法 | `qwen3vl_local/goalgen/GOALGEN_RUN.md` 索引；版本细节看 `GOALGEN_V1.md` / `GOALGEN_V2.md` |
| LeadMoT 跑法 | `qwen3vl_local/leadmot/LEADMOT_RUN.md` |
| LeadMoT 架构 | `qwen3vl_local/leadmot/ARCHITECTURE.md` |
| LeadMoT CARLA 闭环评测 | `qwen3vl_local/eval_carla/EVAL_CARLA_RUN.md` |
| LEAD RGB 批量转视频 | `lead_video_tools/LEAD_VIDEO_RUN.md` |
| 关键帧 / ROAD-EVENT 重标注 | `keyframe_filter/ROAD_EVENT_CLASSIFICATION_PLAN.md` |
| 规则入口 | `AGENTS.md` / `CLAUDE.md` |


## Phase3 映射与动作审计补充（2026-09-05）

Phase3 v2 保持五动作；完整 RS 四问全 NO 恢复 R3，未问不作 NO，HIGHWAY 为独立事实；并发异常保留。普通无灯路口不自动 U-E7，原 U7 用既有灯故障答案表适配；新增 R5/R-E5 常规让行，与七异常及 R-E2/R-E3 共十个 context 1:1。R-E2 包含目标变道及绕障恢复，不按 24 帧截断，两条变道 NO 不清除恢复状态；最终目标 y 符号不决定变道侧。invalid 必须覆盖每个 asked context；未来轨迹只用于离线标签，默认异常 route/RGB 风险过滤。逐帧人工审计与机器覆盖分开记录；详见 sft_new_loop_phase3/MAPPING_AUDIT_20260905.md。

本轮不修改 sft_new_loop_phase1/2 的代码、prompt、已审计答案表。原始 taxonomy 的
OppositeVehicleTakingPriority/R5/U-E7 与 Phase1 TRAFFIC_LIGHT_ABNORMAL=NO 不一致，
只在 Phase3 source_mapping 适配；ROAD_EVENT_CLASSIFICATION_PLAN 已注明历史语义差异。
当前动作规则使用4Hz真实速度顺序、连续lane身份及完整观察窗口；跨road连接段暂不强标横向NO，
planned route不作为原车道中心线真值。现有582条RGB审计缓存可复用，机器扫描不表示人工完整动作确认。

2026-09-05 扩展复核覆盖197个scenario×Town连续图组（42场景、11 Town、7826个缩略帧格），
不等于582条route全程人工确认。InvadingTurn/Town13的同road355 lane-id在98/128变化却未见
RGB跨线，已加入Phase3精确横向未确认清单：FULL_MANEUVER跳过，纵向组保留并记录证据不足。
同road+连续lane-id仍只是候选证据，原始xodr lane-section连通尚未核验。
本轮诊断索引filtered_v5共1152条，三split各十context×32+64invalid；训练默认和定额eval均
对齐invalid/valid=20%（总量16.7%）。详见Phase3映射审计报告及本地review.html，未训练模型。

2026-09-05 后续边界审计：在上述记录上新增两轮52个连续图组条目及10条原图/补充图组
记录，合计259条notes、244条不同route；仍不是582条路线全程人工确认。
已查明 `expert.py` 保存的 road_id/lane_id 来自 LaneType.Any，InvadingTurn/Town13
1777_0 的 f98–127 为 Shoulder，而另一遍 Driving 查询的 ego_lane_id 仍为-1。
Phase3 横向监督要求完整 Driving 窗口，不混用缺少配套 road_id 的 ego_lane_id。
动作规则 `ordered_speed_driving_lane_v5` 排除孤立加速峰值，并允许起步后在正常速度范围
轻微回落；prompt 为 `v3_boundary_audited`，train/eval 显式拒绝旧规则索引。
新增 Phase3 专用 ParkingCutIn/Town13 1008_0 f96–98 U3 正例，不写回Phase1/2；另隔离
VehicleTurningRoute/Town04 的 f2–10 错误 R5/RE5 候选。最新诊断数据 filtered_v8 共
12,928候选、1152平衡行。58项回归、三split实际train/eval采样20seed检查通过，无模型加载。
`dispatch.py` 是保留并发UE/恢复状态并报告缺失RE gate的接口，尚未接入在线runner；
`build_same_rs_challenge.py` 仅有1route/5题同RS错事件诊断，不是独立holdout，未并入训练。
当前正式索引invalid仍只覆盖错误RS，不能宣称已具备全情境同RS事件纠错能力。
后续入口：`sft_new_loop_phase3/BOUNDARY_AUDIT_20260905.md`；Phase1/2对HEAD仍无改动。


### 2026-09-05 Phase3 后续落实：同 RS 负例与常规候选核对

Phase1/2未改。本轮原图15路线84张，累计274 notes/248不同路线；不是全route确认。
新增 `same_rs_invalid.py` 将9个route/anchor的34道人工同RS错事件候选接入主构建；
train/eval保留invalid_reason、同RS已具备的asked-context覆盖，构建与运行时共用
source/联合配额。`filtered_v12` 为582路线审核子集的12,896有效候选/1152均衡题；
相比v8排除32个正候选（过早RE5 12、主路伪RE3 8、雾中U5提前段12）。
三个split分别54/10、56/8、56/8个wrong-RS/same-RS invalid，七UE的同RS题跨三split
均有；RE同RS覆盖不全，不凭空补负例。十个有效context仍各32、每split总384题。
`dispatch.plan_candidate_requests/candidate_response` 提供常规候选核对，不把R3/R5
当RE真值；未驳回且动作全NO不表示恢复完成，接口未接CARLA。五动作不变。
prompt更新v4_context_recheck；动作规则仍Driving v5；训练/eval新增mapping_contract_hash
校验绑定人工决定，旧索引必须重建。61项回归与三split各20seed实际采样通过；未训练。
最新状态以 `sft_new_loop_phase3/BOUNDARY_AUDIT_20260905.md` 为准，v8及此前数字为历史。


## Action prior 轨迹训练（2026-09-05）

新增 `qwen3vl_local/action_prior/`，运行入口 `run.md` / `run_full_pipeline.sh`。
它冻结 Qwen3-VL-4B-Instruct、Phase1/2 LoRA 和 LEAD BEV，复用已有 LeadMoT
Linear+cumsum 轨迹头、状态/导航、GT 和 loss，主要改变语言 KV 条件；不是扩散头。
新 Phase1 已融合事实四问和 RS，先全问再四个 RS 分层复核；新 Phase2 没有训练
hierarchical EVENT 接口，采用两个已有问题域全问和真实 assistant 后的同域续问复核。
两次一致只是 condition 接受，不代表真实准确率；invalid 字段留空，样本继续参加轨迹训练。

自动权重选择只在 `best_generation/` 内查实际权重，校验生产 prompt name/hash、Git commit、
base 路径和 RGB 2/4图配置，并回查保存 step 的 generation 验证分数；没有 final 兜底。
允许显式指定兼容 adapter；来源、完整权重指纹和代码指纹写入 config/selected_priors/checkpoint。
两个 LoRA 共用 base 但独立启停，禁止 merge。所有最终图文+先验+三句分析 KV 在
`disable_adapter()` 下完整 prefill；禁用上下文退出也强制冻结参数，防止 PEFT 自动恢复梯度。
本入口还在构造 BEV 时禁用旧 timm pretrained 下载，随后 strict 导入本地 LEAD backbone。

冻结问答/简述可按权重与代码合同、实际四图字节、导航和 sample seed 缓存压缩文本；
首次和命中采用相同完整 assistant transcript 重建 base KV，不缓存 GPU KV 或图片。
缓存命中样本的 invalid 仍计入每轮真实呈现计数；这不是 SFT teacher 离线物化机制。

默认索引保留所有合法 4Hz anchor，先排除异常时长 route，按物理路线聚合 Rep/时间戳后
约80/10/10划分。默认61 epoch、4卡×1clip×16累积、LR2e-4、5%warmup+cosine，
每250 optimizer steps验证256帧，每完整epoch全量val选best.pt；实际数据量与步数写training_plan。
此配置参考LEAD/现有LeadMoT，尚未针对新KV分布调参验证。支持DDP、EMA、TB、精确cursor/RNG
断点恢复、独立eval/probe和CPU测试；尚未执行真实权重、多卡训练或闭环评测。
新checkpoint使用独立qwen_backbone schema，旧LeadMoT/eval_carla明确拒绝，不得直接交给旧入口。
Phase1/2/3源码均未修改；Phase3未接入。


### Action prior 首轮审查修订（2026-09-06，部分行为已由下节替代）

用户审查指出直接 BF16 AdamW 小更新舍入、简述仅格式验收、CLI 被索引字段覆盖等问题。
现可训练 decoder/AdamW/EMA 保持 FP32，仅 decoder 前向 autocast，预测转 FP32 再算 loss；
受控三段简述完整表达全部 Phase1/RS/EVENT/域适用性 YES/NO/UNKNOWN，仅容忍空白变化，
拒绝反义/遗漏/额外断言，失败用相同完整模板，原始输出只作审计，不声称自由文本语义验证能力。
导航 CLI 在 read_rows 统一覆盖，单帧 BEV / 4Hz / route10 / waypoint8 非默认值直接拒绝。
共享文件缓存跨 rank 复用，POSIX 锁与原子发布保证并发同 key 首次只计算一次；最终 KV 仍每次 base prefill。
执行指纹覆盖共享 helper/LeadMoT/prompt 家族、只读 runner/BEV 源码及依赖版本，参考源码不入库。
训练 Git 是溯源记录，实际代码哈希才参与兼容比较；不同驱动硬件仍需 smoke。新 checkpoint 为 v2，旧 v1 明确拒绝。

复核保持 history 默认，提供 independent 和 compare（condition 仍按 history）的无训练 audit_priors 入口；
显式统计 UE6 prompt 相同，不能把独立 greedy 重复或带历史一致当成筛错能力证据。
从所选 adapter 同 run manifest 实际 index 读取训练候选池，缺失来源标 unknown；显式迁移 index 标记 override。
报告 action 各 split 的物理路线池重叠/池外/未知；这不是实际 sampled route 追溯，池外也不等于整个系统未见。
轨迹指标按预测事件条件、invalid、简述来源、上游池关系分别归一化；run_ablation.sh 同初始化/数据/预算分别训练
原 base prefill 与 prior，两组完整 test 后再配对256帧子组比较，组归属统一采用 prior case，不能当 GT 事件指标。
真实权重/生成质量/吞吐/多卡效果仍未验证，61 epoch 和 LR2e-4 仍是起始配置，不能据此直接推断成本与增益。


### Action prior 后续需求对齐（2026-09-06，以本节为当前状态）

撤掉 VERIFIED_SUMMARY/唯一模板验收。base 自行组织 Scene/Interaction/Planning context 三段简析，
当前速度、导航几何和接受条件共同约束分析；随后同一禁用 LoRA 的 base 重新做纯文本 prefill，
复核一致性/道路覆盖、正类覆盖、未知字段、额外断言和导航依据。严格解析五个布尔项，全部通过保留原文，
失败才用完整 fallback；fallback 的 planning 也随已解析速度/目标方位/正类条件变化。
这是模型审查而非确定性语义保证，同源误判仍可能发生；真实模型质量尚未验证。
缓存保存草稿/复核及配对 SHA，最终 transcript 不包含复核或失败草稿，最终 KV 仍全由禁用 LoRA 的 base 计算。
语言协议 v3，FP32 checkpoint 容器仍 v2；旧照抄模板合同不可混用。

执行哈希由显式真实/延迟入口加模块初始化 import 展开，当前48个源码，未接入 Phase3 为0；
修改无关 Phase3 不再影响 action 恢复，真实 engine/M-RoPE/decoder/runner/prompt 改动仍失效。
新增延迟执行依赖须同步 seeds，参考源码只哈希不入库。上游训练索引审计的路径/获取模式/可用状态/
内容与生成 identity 分离；eval/probe 支持两份 training-index 重映射，来源异常标 unknown 并记录错误。
续训保留原审计路线快照使 epoch 累加口径稳定，同时报告新获取内容差异；独立 eval 使用当前来源，
同内容移动、显式指定或暂时缺失不会阻断相同生成条件。来源变化仍不改变“非严格系统未见 holdout”的限制。
轨迹分组新增 confirmation/all_confirmed、expected_domain_only、unconfirmed，避免正常域外被解读成复核失败。
Phase2 compare 明确记录仅 history 接受、不要求共识、跨模式分歧；仍是观察工具，未证明能筛掉真实错误，
也未伪造 EVENT hierarchical 接口。默认首次每帧至多11次生成（含纯文本复核），compare至多17次，另做最终 base prefill。


### Action prior 闭环与审计包（2026-09-06）

`action_prior/run_full_pipeline.sh` 默认训练+频繁 val+最终离线 test/probe+审计包；
显式 `BENCH2DRIVE=1` 追加正式 220 路线闭环。`action_prior/eval.sh --bench2drive` 是独立闭环入口，
不将 action checkpoint 交给旧 LeadMoT launcher。专用 agent 复用 eval_carla 的传感器/PID，
通过 `_create_runner` 恢复训练同款 action runtime，`_route_endpoint` 读取正式 benchmark XML。
闭环源代码与传感器配置另存评测身份，不绑定无关 Phase3。220 条/44 类结果包含 DS/SR/RC/IS、
效率、舒适性、五能力与每场景明细；Traffic Signs 按官方 0.0.4 单次计数口径，缺地图/记录为 N/A。
运动学只供指标，禁止写入 policy。`audit.zip` 硬限制 30,000,000 字节，核心指标必须完整，
可选案例/历史按预算选入并列遗漏；权重/缓存/完整视频/原始 TB 和运动学大产物不入包、不入库。
正式 220 test 不参与训练期选优。CPU/合成检查不代表实际 CARLA 或真实模型已验证。

新增入口 `action_prior/bench2drive.py/.sh`、`carla_agent.py`、`carla_runtime.py`、
`benchmark_report.py`、`audit_bundle.py`。训练期仍每250更新验证256帧、每轮全量val。
原 action 段中“eval_carla 明确拒绝”仍指旧 launcher，专用 action 入口现已实现源码接入。
闭环 launcher 默认完整220、支持 subset smoke/显式resume/report-only；运行前CPU合同校验，
多GPU独立route与端口；使用当前Python，不改只读leaderboard/lead源码，不启动本地CARLA验证。
`--dry-run` 已按本地正式 XML 核对220/44；真实模型/传感器/控制表现和论文分数尚未验证。
官方 https://github.com/Thinklab-SJTU/Bench2Drive/blob/0.0.4/tools/ability_benchmark.py 的
Traffic Signs 新口径只给未成功但合法过路口的路线补成功，不沿用本地旧脚本的重复计数。
Comfort 沿用本地官方函数及其原始 angular_velocity 处理，采集10Hz与dt=.1对齐；指标版本、
训练数据/传感器/安全兜底差异必须披露。缺记录DS/SR以planned分母列零贡献并标provisional，
能力缺观测不给完整均值，不能用少于220条的已完成子集冒充完整得分。

Action prior 新增 `rank_loras.py/.sh`：CPU 只读扫描 Phase1/2 `best_generation`，命令行
打印保存 step 的全部 generation/teacher-forced/guard 指标及 prompt/Git/RGB/来源，分别推荐。
复用 `inspect_adapter` / `selection_score` 与生产同序排名，不混入 final/balanced/test 或其它step；
无推荐写报告后 exit 2。默认报告 `checkpoints/action_prior_lora_audit/run_<时间>/`，不入库。
它是已有验证记录对比，不是重新评测，不证明不同run在共同holdout上可比；运行见 action_prior/run.md。
