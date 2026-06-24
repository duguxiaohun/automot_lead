# AGENTS.md

> 给所有后续 AI / coding agent 的项目入口说明。
> 目标是让新会话在改代码前快速知道：这个工作区在做什么、必须先读什么、哪些文件能动、哪些操作不要做。
>
> 本项目同时维护 [`CLAUDE.md`](CLAUDE.md) 作为 Claude Code 的自动加载入口。
> **AGENTS.md 与 CLAUDE.md 必须保持规则同步**：任何一边新增/修改文件白名单、
> git 规则、工作流偏好、禁止事项、项目入口说明时，必须同步更新另一边。

---

## 1. 先读顺序

开始任何代码分析、改动、提交之前，按这个顺序读：

1. `CLAUDE.md`：Claude Code 自动加载的镜像规则入口；Codex 也要读，确保两边规则一致。
2. `AGENTS.md`：当前通用 agent 入口；Claude 读到 `CLAUDE.md` 后也要读本文件。
3. `PROJECT_CONTEXT.md`：核心技术背景，包含 `lead/` 与 `AutoMoT/` 的数据、推理、BEV、RGB、LiDAR 对齐结论。
4. 当前任务相关源码：通常优先看 `AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py`，必要时再查 `lead/` 或 `AutoMoT/` 中的参考源码。

不要跳过 `PROJECT_CONTEXT.md` 直接从源码重新推断。这个项目里很多结论来自多轮核对，重新凭印象推断很容易犯错。

如果修改了 `AGENTS.md` 中任何规则，也必须同步修改 `CLAUDE.md`；如果发现
`CLAUDE.md` 比本文件更新，也必须把对应规则同步回本文件。不要让 Claude 和 Codex
看到两套不同规则。

---

## 2. 项目一句话

这个工作区在做的是：

把 `lead/` 采集/训练出来的 CARLA 离线数据，整理成本地 Qwen3-VL-Instruct frozen prefill + LeadMoT decoder 能直接消费的离线输入，并逐步分析两边数据分布、坐标系、RGB/LiDAR/BEV/target_point 的差异。

当前主要战场：

- `AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py`
- `AutoMoT/leaderboard/team_code/vlm_paradigm_a_runner.py`
- `PROJECT_CONTEXT.md`

---

## 3. 当前技术状态

关键结论以 `PROJECT_CONTEXT.md` 为准，下面只是快速索引：

- `lead/`：数据采集、训练、闭环评测仓库。CARLA 20Hz，每 5 tick 落盘 1 帧，即 4Hz。
- `AutoMoT/`：在线驾驶仓库。慢路径是 Qwen3-VL + KV cache，快路径依赖 BEV encoder + DP heads。
- 当前离线 runner 只走本地 `AutoMoT/qwen3vl_local` 的
  `LocalQwen3VLInstructEngine` 做 frozen Qwen prefill，再接 LeadMoT decoder；
  已移除 AutoMoT legacy `kv_cache_fixed_inference(...)` / `InterleaveInferencer`
  / 原 fast head 接口，不再保留 `--enable-automot-slow` 或 `enable_fast_inference`。
- runner 已切换到 LEAD 风格的 `LeadTransfuserBackbone` / `LeadBEVEncoder`，其输出直接供 LeadMoT 使用；不能再接 AutoMoT 原快推理 decoder。
- LEAD RGB 是三视角拼接 `(W=1152, H=384)`；当前本地 Qwen frozen prefill 直接喂整图，不切片、不 resize、不选前视。
- `vlm_paradigm_a_runner.py` 的 `qwen` backend 必须只读本地 `AutoMoT/checkpoints/Qwen3-VL-4B`（`local_files_only=True`），并用 HF 标准 `past_key_values` 显式 prefill/decode 做文字输出；AutoMoT 现有 `InterleaveInferencer` / `qwen3vl_template_inference` 绑定 AutoMoT 自定义 MoT 架构，不要拿来直接支撑 standalone Qwen 的完整自由文本生成。
- `qwen3vl_instruct_paradigm_a_runner.py` 是 standalone Qwen-only 范式 A runner，只跑本地 `AutoMoT/checkpoints/Qwen3-VL-4B-Instruct`；该目录对应 HuggingFace `repo_id=Qwen/Qwen3-VL-4B-Instruct`，用户远程环境已下载。必须 `local_files_only=True` 且设置 HF/Transformers offline 环境变量，禁止下载；不 import `vlm_paradigm_a_runner.py`，不接 AutoMoT `InterleaveInferencer`。
- `AutoMoT/qwen3vl_local/` 保存 Qwen3-VL-Instruct 本地可魔改代码：`prompt_pipeline.py` 从 `vlm_paradigm_a_runner.py` 的迁移块同步完整提示词/状态机；另含 LEAD RGB 读取、显式 prefill/decode、KV cache summary 与可选 `torch.save`。
- `0026.json` 是 LEAD meta.pkl 转 JSON 的固定参考样本，只读，绝对不要修改或入库。

---

## 4. 文件修改范围

未经用户明确同意，只允许修改：

- `AGENTS.md`
- `CLAUDE.md`
- `PROJECT_CONTEXT.md`
- `AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py`
  （LeadOfflineMoTRunner 加载 ckpt 时按 `decoder_config.use_bev` / `use_subgoal` /
  `use_final_goal` 自描述切换分支；use_subgoal=True ckpt 必须在 `lead_clip`
  里塞 `subgoal_rgb_path/subgoal_scenario/subgoal_status/subgoal_event`，由
  `build_clip_from_real_lead_route(..., subgoal_*=...)` 写入；runner 的 `main()`
  加载 ckpt 后按 use_subgoal 自动调 `--keyframes` 反查 anchor 对应 STATUS →
  SUBGOAL → SUBGOAL keyframe RGB）
- `AutoMoT/leaderboard/team_code/vlm_paradigm_a_runner.py`
- `AutoMoT/leaderboard/team_code/qwen3vl_instruct_paradigm_a_runner.py`
- `AutoMoT/leaderboard/team_code/qwen3vl_dit_goalgen_runner.py`
- `AutoMoT/leaderboard/team_code/automot_utils.py`
  （按用户同意纳入白名单：AutoMoT legacy prompt helper；`build_cleaned_prompt_and_modes`
  必须接收 7 元 `[speed,tp,ntp,final_goal]` 并在 prompt 写入 final destination）
- `AutoMoT/Automot/team_code/automot_utils.py`
  （按用户同意纳入白名单：AutoMoT 原始副本 prompt helper；必须与
  `AutoMoT/leaderboard/team_code/automot_utils.py` 的 7 元 final destination prompt 保持同步）
- `AutoMoT/Automot/mot/evaluation/inference.py`
- `AutoMoT/Automot/mot/modeling/automot/automot.py`
  （按用户同意纳入白名单：AutoMoT 原始副本中的 prompt 示例注释；涉及 target_point /
  next_target_point 的 prompt 示例必须包含 final destination，并与 2s 规划视野同步）
- `AutoMoT/leaderboard/team_code/mot_b2d_agent.py`
  （按用户同意纳入白名单：legacy AutoMoT 在线 agent；涉及 wp/nwp prompt 时必须同步
  按 `max(speed*lookahead_s, 5m)` 弧长生成 tp/ntp，生成局部 final_goal 并传入
  `automot_utils.build_cleaned_prompt_and_modes`）
- `AutoMoT/leaderboard/team_code/display_interface.py`
- `AutoMoT/Automot/team_code/display_interface.py`
  （按用户同意纳入白名单：AutoMoT 显示层；decision 三元组只表示 now/+1s/+2s，
  不要再沿用旧 3s 命名）
- `AutoMoT/qwen3vl_local/eval_carla/`（LeadMoT 闭环评测子包，全部子文件白名单内）
  - `__init__.py` / `EVAL_CARLA_PLAN.md` / `EVAL_CARLA_RUN.md`
  - `agent.py`
    （LEAD 风格 CARLA Bench2Drive 实时 agent：3 摄像头 1152×384 + IMU/GPS/Speedometer +
    可选双 LiDAR/4 radar（按 ckpt `decoder_config.use_bev` 决定）；no-BEV 模型不产生未使用 LiDAR/radar 输入。
    ckpt `decoder_config.use_subgoal=True` 当前不支持闭环（CARLA 在线无法获得 SUBGOAL keyframe RGB），
    agent 加载时立即 `raise NotImplementedError` 并留 `TODO(subgoal)` 接口，由后续 SUBGOAL 图像生成/代理输入填补。
    **LEAD 训练分布对齐 (v2)**：RGB 拼接后 JPEG round-trip (JPEG_QUALITY=85)、
    LiDAR 轻量去地面 (z+LSQ, LIDAR_REMOVE_GROUND=1)、radar 4 路 → ego + 近车 duplicate (factor=5, radius=8m) 拼到 LiDAR、
    5 sweep 累积 0.25s 窗对齐 anchor frame。
    推理直接复用 `LeadOfflineMoTRunner`，每 5 tick 调一次模型，中间 tick PID 跟踪 (desired speed 用 wp[1]/wp[3] 即 0.5s/1.0s 两点平均)。
    target_point / next_target_point：训练与在线都走 P1 speed×lookahead 弧长前推：
    `max(speed*lookahead_s, 5m)`，默认 tp=1.0s / ntp=2.0s；final_goal 为 route 真实终点：
    训练取 LEAD 采集保存的 `meta["next_target_points"][-1]` 转 ego，在线 eval_carla 取
    `scenario_picker.py` 对应 route XML 最后一个 waypoint 转 ego；不能再用 `meta["route"][-1]`
    或固定局部 horizon，ego frame (x_forward, y_left)。
    warmup 改 **LEAD 风格 left-pad** 复制 frame 0 立即推理，不再等历史 (与 build_clip line 1808-1815 同款)；
    UKF + route_planner + 基本 PID + SafetyMixin 兜底；Python class/function 已补中文 docstring，shell/HTML/CSS 关键逻辑块有中文注释）
  - `safety.py`
    （SafetyMixin：`stuck_helper` 累计 300 帧低速 → force_move 14 帧 creep / `parking_start`
    前 200 帧位移 < 6m 禁用 force_move / `parking_escape` 1500 帧窗口位移 < 5m 触发 phase1
    强转角 -0.65 + 油门 0.45 / 限速 35 km/h；与 mot_b2d_agent.py 行为完全一致）
  - `video_recorder.py`
    （input/debug/bev_debug/demo/grid **五路** mp4，ffmpeg crf=18/22/28；
    bev_debug 是 LEAD 风格顶视 LiDAR 散点 + pred_route + pred_waypoints + tp/ntp + ego box，
    与 LEAD `video_recorder.py` 的 BEV pseudo-image 等价；demo 在首帧通过
    `CarlaDataProvider.get_world()` 找到 `role_name=hero` 后 spawn cinematic + BEV 临时 carla camera）
  - `visualizer.py`（无依赖 pinhole 投影 + 三视角 overlay；从 LEAD common_utils.project_points_to_image 移植）
  - `scenario_picker.py`
    （LEAD `<Scenario>/<route_id>.xml` 反向映射；CLI 支持 `--scenario` / `--route-id` /
    `--random N --seed K` 子集筛选与 `--list-scenarios`）
  - `aggregate.py`
    （按 scenario 聚合 leaderboard `eval_<route_id>.json` 写 `scenarios/<Scenario>/summary.json` + `summary_all.json`）
  - `run_eval.sh`
    （一键 launcher：必填 `--leadmot-ckpt`；自动空闲 GPU + 端口槽，支持 `--num-gpus N` / `EVAL_GPU_COUNT=N`
    多卡 worker round-robin 分 route；三种跑法：
    全量（无过滤）/ 按场景 `--scenario <Name>` / 随机 `--random N --seed K`，可叠加；
    `--single-test` / `--route-id` / `--no-input|--no-debug|--no-demo|--no-grid`；跑完自动调 aggregate）
  - `webapp/{__init__.py, app.py, templates/index.html, static/style.css}`
    （Flask：signature 下拉切换 ckpt；Routes tab 按 scenario 分组列 route + 4 路视频切换 + leaderboard
    scores + infractions；Scenarios tab 表格列每个 scenario 平均分）
- `AutoMoT/qwen3vl_local/`（含 `tb_serve.sh` 通用 TensorBoard launcher；`goalgen/` 子包详见 PROJECT_CONTEXT.md §15；`eval_carla/` 子包详见上）
- `AutoMoT/qwen3vl_local/tb_serve.sh`
  （SFT / GoalGen / LeadMoT / VAE 共用 TensorBoard 启动器；从 `AutoMoT/` 目录下用
  `bash qwen3vl_local/tb_serve.sh <logdir>` 启动）
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
  （LEAD-MoT 快推理 decoder 子包及 v1 decoder-only 训练/eval/probe 入口：route(B,10,2) + waypoint(B,8,2)，Linear+cumsum head；gen 路独立 12 层 + frozen Qwen prefix K/V attention（不过 Linear）；hidden=1024=8x128 对齐 Qwen K/V 子空间；gen Q/K 按 `input_len + rope_deltas` 加 1D RoPE，language K/V 已由 Qwen prefill 带 M-RoPE 不重复旋转。训练时冻结 Qwen3-VL-Instruct 与 LeadBEVEncoder，只训练 LeadMoT decoder；GT 包含 route / future_waypoints 两类 ego-frame 累计点，head 内 Linear+cumsum 后直接对绝对点算 loss；`eval.py` 汇总 loss/ADE/FDE，`probe.py` 随机 case-level dump 预测与 GT 对比图。runner 必须用 `LocalQwen3VLInstructEngine` 单独跑 frozen Qwen prefill，只接受同源 HF `past_key_values`；不复用 AutoMoT InterleaveInferencer 的 `gen_context`，也不保留 AutoMoT legacy slow/fast 接口；`--leadmot-ckpt` 显式加载 decoder 权重，先读 checkpoint 的 `decoder_config.use_bev` 再实例化 decoder，并 `strict=True` 加载：`use_bev=True` 必须导入已有 BEV projector 参数，`use_bev=False` 则完全不实例化 / 不 forward BEV，禁止混入随机 BEV；不传 ckpt 仅作为随机初始化链路调试。**`use_subgoal`（离线专用）**：与 `use_bev` 正交的 prefix-only 开关，开启时 build_dataset `--with-subgoal-fields` 反查 `keyframes_all_scenarios.json` 写 scenario/run_id/status/subgoal/subgoal_frame/subgoal_rgb_path/subgoal_lookup_ok 字段；train/eval/probe 通过 `LeadMoTTrainRuntime._run_subgoal_qwen_prefill` 在 prefix 多喂 1 张 SUBGOAL stitched RGB + `[GROUND_TRUTH_STATE]` 文本块（prompt 由 `leadmot/subgoal_prompt.py` 提供，prompt 内仍保留 navigation 文本以维持 tp/ntp/final_goal 对齐）；ckpt `decoder_config.use_subgoal` 与训练 args 必须严格一致，cross-load 由 `_require_subgoal_match` 拒绝；state_dict 形状不受影响（subgoal 不引入新模块），但 prefix KV 分布不兼容；`mot_lead_offline_runner.py` 会按 ckpt 自动走 subgoal prefill 并要求 clip 注入 subgoal 字段，CLI demo 可通过 `--keyframes` 自动反查；eval_carla 在线 agent 暂不支持该开关，加载 use_subgoal=True ckpt 时立即 `raise NotImplementedError`。详见 `leadmot/ARCHITECTURE.md`、`leadmot/LEADMOT_PLAN.md` 与 `PROJECT_CONTEXT.md`）
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
  （以上是统一 LoRA SFT 子包，已废弃 v1/v2 双轨与 ms-swift。`build_dataset.py` 只产 `dataset_version="pending"` jsonl（assistant 含 `__TEACHER_PENDING__` 占位）；`train.sh` → `train.py` 用 `peft.LoraConfig` + `get_peft_model` 直接把 LoRA 注入 base，torch DDP + 手写 train loop；每个 train batch 内部禁用 adapter，并调用底层 Qwen base model 现场 greedy 生成 ANALYSIS（避开 `PeftModel.generate` 的 Qwen3-VL 生成错位问题），再启用 adapter 跑 student forward + 内置 per-token 加权 loss（ANALYSIS body `SFT_ANALYSIS_WEIGHT`/默认 0.5，学习大致语言推理但不逐字压过状态监督；其余 assistant 段 1.0，prompt 段 0；旧 v2 "结构字面 mask=0" 致命陷阱不再保留）。**不再离线物化 teacher / 不再写 manifest / 不再有 runtime_teacher_data 复用**；`build_teacher.py` 仅作为可选离线 dump 工具。`eval.py` / `probe.py` 默认 `merge_and_unload`，case dump 保存 `expert_analysis.txt` / `language_compare.json` 对比 base-teacher 专家语言与模型 ANALYSIS；`probe.py` 的 token loss 使用训练同款权重汇总；`check_loss_mask.py` 静态验证 train.py 内置 mask；`inspect_teacher_outputs.py` 支持 `--live` 现场重跑 teacher 抽检。详见 `SFT_PLAN.md` / `SFT_RUN.md`）
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
  （按用户同意新增到白名单：SFT v2 两段式串行选择题子包。输入仍为 LEAD stitched RGB + 语言 prompt；stage-1 只列 `SCENE_CHOICES` 并输出 `SCENE`，stage-2 作为同一条对话的后续 user prompt，按预测 scene 的 `EVENT_SEQUENCE` 输出 `STATUS/SUBGOAL`，推理时必须复用 stage-1 已吃图像和场景 prompt 后的 KV cache；默认 `--samples-per-scenario 0` 全量保留合法候选，默认 `--wrong-scene-ratio 0.15` 只增强 train rows；不再有 ANALYSIS / teacher / pending cache；训练 loss 只监督 scene/status/subgoal 值 token，格式 token 为 0 loss；LoRA 默认只注入语言侧 Linear，视觉侧通过 `--lora-vision-scope` / `LORA_VISION_SCOPE` 选择 `off` / `merger` / `last4` / `all` 四档（`--lora-vision` / `LORA_VISION=1` 作为 `all` 的 legacy 别名保留）；开启视觉 LoRA 时默认带"视觉组单独 LR 倍率 `--vision-lr-scale=0.1`（受 `--max-vision-lr-scale=0.25` 上限约束）+ 分组梯度裁剪 `--language-clip-norm=1.0` / `--vision-clip-norm=0.3` + TB 观测 `grad_norm/{language,vision}` / `param_norm/lora_{language,vision}` / `vision_guard_bad_steps` + `STRICT_VISION_SCOPE=1` 命名漂移硬拒绝 + `VISION_GUARD_ENABLED=1` 运行时熔断"保险；熔断时写 `fuse_stop_step_<N>/` 与 `fuse_reason.txt`，并跳过正常 `final/` 保存，防止视觉表征被冲坏且避免误用异常产物；base Qwen checkpoint 始终只读，训练只保存 adapter delta，并写 `sft_v2_adapter_config.json`（含 `lora_vision_scope` 与保险参数）；eval/probe 加载前按 adapter 配置判断普通 LoRA / 视觉 LoRA 并校验权重 key，不一致直接拒绝。自由生成评估中 scene 不在白名单则中断，scene 合法但错误时仍按预测 scene 进入 stage-2 并用串行口径计错，同时输出 `valid_total` / `*_valid_scene` 指标。运行文档见 `SFT_V2_RUN.md`）
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
  （按用户同意新增到白名单：SFT v3 代码已落地，采用 sub-scenario 时间序列训练 + 学生自维护 memory + 三步内循环 teacher/student 蒸馏；Phase A 学生自更新 memory，Phase B 每帧弱纠偏 scene=GT 反向学习“对的别改”；δ 允许 0 且只封顶 10，`EGO_TO_GOAL_XY` 严格来自 meta `next_target_points[-1]` 并在帧末预取下一帧，step3 触发统一走 `should_trigger_step3`；loss 为分析与离散值 token 混合监督，LoRA 视觉接口与 v2 同构并默认关闭；`train.sh` 默认 `ddp`（历史模式名），每卡默认 batch=1；多卡训练采用 work-stealing + local-SGD：不包 DDP、不静态分片、不截断尾部，通过 TCPStore 抢 episode，NCCL collective 前先 TCPStore rendezvous，先广播 rank0 LoRA 初始权重，按本轮 optimizer step 数加权平均 LoRA 参数，并且 `checkpoint-*` / `final/` 都在参数平均后保存；sync 日志/TB 记录 `all_rank_steps`、`round_eps`、`total_eps` 用于审计训练量。详见 `SFT_V3_PLAN.md` / `SFT_V3_RUN.md` 与同目录脚本。）
- `AutoMoT/qwen3vl_local/sft_v4/__init__.py`
- `AutoMoT/qwen3vl_local/sft_v4/SFT_V4_PLAN.md`
- `AutoMoT/qwen3vl_local/sft_v4/SFT_V4_RUN.md`
- `AutoMoT/qwen3vl_local/sft_v4/prompts.py`
- `AutoMoT/qwen3vl_local/sft_v4/build_dataset.py`
- `AutoMoT/qwen3vl_local/sft_v4/train.py`
- `AutoMoT/qwen3vl_local/sft_v4/train.sh`
- `AutoMoT/qwen3vl_local/sft_v4/eval.py`
- `AutoMoT/qwen3vl_local/sft_v4/probe.py`
- `AutoMoT/qwen3vl_local/sft_v4/check_loss_mask.py`
- `AutoMoT/qwen3vl_local/sft_v4/test_memory_update.py`
- `AutoMoT/qwen3vl_local/sft_v4/test_kv_reuse.py`
- `AutoMoT/qwen3vl_local/sft_v4/test_gt_leak_filter.py`
- `AutoMoT/qwen3vl_local/sft_v4/replay.py`
- `AutoMoT/qwen3vl_local/sft_v4/collect.py`
- `AutoMoT/qwen3vl_local/sft_v4/learn.py`
- `AutoMoT/qwen3vl_local/sft_v4/launch_offpolicy.sh`
- `AutoMoT/qwen3vl_local/sft_v4/inspect_teacher.py`
  （按用户同意新增到白名单：SFT v4 是 sequence-memory OPD 的 off-policy actor-learner 路线；生产入口为 `launch_offpolicy.sh`，默认 4×H20 保守部署为 GPU0/GPU1 各 1 个 learner DDP rank、GPU2/GPU3 各 1 个 collector；确认服务器允许单卡多 CUDA 进程后，可手动调 `COLLECTORS_PER_GPU=2/3`。collector 不进 DDP/NCCL，只异步用 LoRA snapshot rollout 并写 `replay/ready/*.jsonl`；learner 才进 DDP，同步随机读取 replay 做 teacher-forced loss + gradient allreduce，并周期发布 `latest_lora/v_<step>/`。`replay.py` 负责 trajectory schema、原子写、文件锁 counter、FIFO 驱逐；`collect.py` 负责 Phase A 50% 正确初始化、Phase B 0.15 噪声扰动、teacher/student generate 和 trajectory 写盘；`learn.py` 负责 replay 采样、无 generate 的 loss/backward、checkpoint/final/snapshot；`train.py` / `train.sh` 仅保留为 on-policy 兼容调试入口，生产训练不要走它。`inspect_teacher.py` 是离线老师抽检脚本：随机采样 episode × 帧 × 3 种 memory 模式（all_keep / event_change / scene_change），全程 `disable_adapter` 走 frozen base Qwen，逐 step 记录 system / user / teacher-assistant 三类 role 的 prompt 与 token 统计，产物为 `teacher_report.md` + `teacher_report.jsonl`，供人工评估老师推理质量并指导 prompt 迭代。）
- `AutoMoT/qwen3vl_local/goalgen/GOALGEN_PLAN.md`
- `AutoMoT/qwen3vl_local/goalgen/GOALGEN_RUN.md`
- `AutoMoT/qwen3vl_local/goalgen/GOALGEN_V1.md`
- `AutoMoT/qwen3vl_local/goalgen/GOALGEN_V2.md`
- `AutoMoT/qwen3vl_local/goalgen/build_dataset.py`
- `AutoMoT/qwen3vl_local/goalgen/train.py`
- `AutoMoT/qwen3vl_local/goalgen/train.sh`
- `AutoMoT/qwen3vl_local/goalgen/eval.py`
- `AutoMoT/qwen3vl_local/goalgen/probe.py`
  （以上 9 个是子目标 latent 生成路线 v1/v2 共用数据/训练/eval/probe/文档，详见 PROJECT_CONTEXT.md §7；`GOALGEN_PLAN.md` / `GOALGEN_RUN.md` 只保留索引，版本细节分别写入 `GOALGEN_V1.md` / `GOALGEN_V2.md`；MD 与代码同位于 goalgen 子包内，不要再在 tools/ 下创建重复 MD。GoalGen 训练默认必须导入 `AutoMoT/checkpoints/patch_unpatch_v1/latest/weights/patch_unpatch_best.safetensors`（再兜底无 run_subdir 与最新 `run_*`）并冻结；找不到直接报错，不再随机初始化 patch/unpatch）
- `AutoMoT/vae_standalone/train_patch_unpatch.py`
  （patch/unpatch 端到端图像重建训练脚本：image→VAE.encode→patch→unpatch→VAE.decode→image；VAE 冻结。产物 `patch_unpatch_*.safetensors` 可被 `DiTMoT.load_patch_unpatch` 直接加载，state_dict key 与 DiTMoT 内 `self.patch` / `self.unpatch` 一一对应。`AutoMoT/vae_standalone/` 下其它原始文件仍为只读参考，除非已单独列入白名单）
- `AutoMoT/vae_standalone/vae_reconstruct.py`
  （按用户同意新增到白名单：VAE / patch-unpatch 诊断脚本。支持 VAE-only 与 VAE+patch/unpatch 两种重建链路，对比 VAE 前后 loss、patch 前后 latent loss，按 v1/v2 选择默认模式，支持 TensorBoard 批量 loss 与随机小批量 PNG 对比可视化）

其它文件默认只读，尤其是：

- `lead/` 整个目录
- `AutoMoT/` 中除上述白名单外的源码、配置、权重、数据
- `0026.json`
- `keyframes_all_scenarios.json`

如果确实需要改白名单外文件，先在对话里说明原因并等待用户确认。

---

## 5. Git 规则

不要使用：

- `git add .`
- `git add -A`
- `git add *`
- `git add lead/`
- `git add AutoMoT/`
- `git add 0026.json`

只精确 add 白名单文件。例如：

```bash
git add AGENTS.md CLAUDE.md PROJECT_CONTEXT.md
git add AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py
git add AutoMoT/leaderboard/team_code/vlm_paradigm_a_runner.py
git add AutoMoT/leaderboard/team_code/qwen3vl_instruct_paradigm_a_runner.py
git add AutoMoT/leaderboard/team_code/automot_utils.py AutoMoT/Automot/team_code/automot_utils.py AutoMoT/Automot/mot/evaluation/inference.py AutoMoT/Automot/mot/modeling/automot/automot.py AutoMoT/leaderboard/team_code/mot_b2d_agent.py AutoMoT/leaderboard/team_code/display_interface.py AutoMoT/Automot/team_code/display_interface.py
git add AutoMoT/qwen3vl_local/eval_carla/__init__.py AutoMoT/qwen3vl_local/eval_carla/EVAL_CARLA_PLAN.md AutoMoT/qwen3vl_local/eval_carla/EVAL_CARLA_RUN.md AutoMoT/qwen3vl_local/eval_carla/agent.py AutoMoT/qwen3vl_local/eval_carla/safety.py AutoMoT/qwen3vl_local/eval_carla/video_recorder.py AutoMoT/qwen3vl_local/eval_carla/visualizer.py AutoMoT/qwen3vl_local/eval_carla/scenario_picker.py AutoMoT/qwen3vl_local/eval_carla/aggregate.py AutoMoT/qwen3vl_local/eval_carla/run_eval.sh AutoMoT/qwen3vl_local/eval_carla/webapp/__init__.py AutoMoT/qwen3vl_local/eval_carla/webapp/app.py AutoMoT/qwen3vl_local/eval_carla/webapp/templates/index.html AutoMoT/qwen3vl_local/eval_carla/webapp/static/style.css
git add AutoMoT/qwen3vl_local/__init__.py AutoMoT/qwen3vl_local/cache_utils.py AutoMoT/qwen3vl_local/engine.py AutoMoT/qwen3vl_local/image_io.py AutoMoT/qwen3vl_local/prompt_pipeline.py AutoMoT/qwen3vl_local/run_log.py AutoMoT/qwen3vl_local/tb_serve.sh
git add AutoMoT/qwen3vl_local/goalgen/__init__.py AutoMoT/qwen3vl_local/goalgen/vae.py AutoMoT/qwen3vl_local/goalgen/prompt.py AutoMoT/qwen3vl_local/goalgen/qwen_kv.py AutoMoT/qwen3vl_local/goalgen/keyframes.py AutoMoT/qwen3vl_local/goalgen/dit.py AutoMoT/qwen3vl_local/goalgen/flow.py
git add AutoMoT/leaderboard/team_code/qwen3vl_dit_goalgen_runner.py
git add AutoMoT/qwen3vl_local/sft/__init__.py AutoMoT/qwen3vl_local/sft/SFT_PLAN.md AutoMoT/qwen3vl_local/sft/SFT_RUN.md AutoMoT/qwen3vl_local/sft/build_dataset.py AutoMoT/qwen3vl_local/sft/build_teacher.py AutoMoT/qwen3vl_local/sft/train.py AutoMoT/qwen3vl_local/sft/train.sh AutoMoT/qwen3vl_local/sft/eval.py AutoMoT/qwen3vl_local/sft/probe.py AutoMoT/qwen3vl_local/sft/check_loss_mask.py AutoMoT/qwen3vl_local/sft/inspect_teacher_outputs.py
git add AutoMoT/qwen3vl_local/sft_v2/__init__.py AutoMoT/qwen3vl_local/sft_v2/SFT_V2_PLAN.md AutoMoT/qwen3vl_local/sft_v2/SFT_V2_RUN.md AutoMoT/qwen3vl_local/sft_v2/prompts.py AutoMoT/qwen3vl_local/sft_v2/build_dataset.py AutoMoT/qwen3vl_local/sft_v2/train.py AutoMoT/qwen3vl_local/sft_v2/train.sh AutoMoT/qwen3vl_local/sft_v2/eval.py AutoMoT/qwen3vl_local/sft_v2/probe.py AutoMoT/qwen3vl_local/sft_v2/check_loss_mask.py
git add AutoMoT/qwen3vl_local/sft_v3/__init__.py AutoMoT/qwen3vl_local/sft_v3/SFT_V3_PLAN.md AutoMoT/qwen3vl_local/sft_v3/SFT_V3_RUN.md AutoMoT/qwen3vl_local/sft_v3/prompts.py AutoMoT/qwen3vl_local/sft_v3/build_dataset.py AutoMoT/qwen3vl_local/sft_v3/train.py AutoMoT/qwen3vl_local/sft_v3/train.sh AutoMoT/qwen3vl_local/sft_v3/eval.py AutoMoT/qwen3vl_local/sft_v3/probe.py AutoMoT/qwen3vl_local/sft_v3/check_loss_mask.py AutoMoT/qwen3vl_local/sft_v3/test_memory_update.py AutoMoT/qwen3vl_local/sft_v3/test_kv_reuse.py AutoMoT/qwen3vl_local/sft_v3/test_gt_leak_filter.py
git add AutoMoT/qwen3vl_local/sft_v4/__init__.py AutoMoT/qwen3vl_local/sft_v4/SFT_V4_PLAN.md AutoMoT/qwen3vl_local/sft_v4/SFT_V4_RUN.md AutoMoT/qwen3vl_local/sft_v4/prompts.py AutoMoT/qwen3vl_local/sft_v4/build_dataset.py AutoMoT/qwen3vl_local/sft_v4/train.py AutoMoT/qwen3vl_local/sft_v4/train.sh AutoMoT/qwen3vl_local/sft_v4/eval.py AutoMoT/qwen3vl_local/sft_v4/probe.py AutoMoT/qwen3vl_local/sft_v4/check_loss_mask.py AutoMoT/qwen3vl_local/sft_v4/test_memory_update.py AutoMoT/qwen3vl_local/sft_v4/test_kv_reuse.py AutoMoT/qwen3vl_local/sft_v4/test_gt_leak_filter.py AutoMoT/qwen3vl_local/sft_v4/replay.py AutoMoT/qwen3vl_local/sft_v4/collect.py AutoMoT/qwen3vl_local/sft_v4/learn.py AutoMoT/qwen3vl_local/sft_v4/launch_offpolicy.sh
git add AutoMoT/qwen3vl_local/goalgen/GOALGEN_PLAN.md AutoMoT/qwen3vl_local/goalgen/GOALGEN_RUN.md AutoMoT/qwen3vl_local/goalgen/GOALGEN_V1.md AutoMoT/qwen3vl_local/goalgen/GOALGEN_V2.md AutoMoT/qwen3vl_local/goalgen/build_dataset.py AutoMoT/qwen3vl_local/goalgen/train.py AutoMoT/qwen3vl_local/goalgen/train.sh AutoMoT/qwen3vl_local/goalgen/eval.py AutoMoT/qwen3vl_local/goalgen/probe.py
git add AutoMoT/qwen3vl_local/leadmot/__init__.py AutoMoT/qwen3vl_local/leadmot/ARCHITECTURE.md AutoMoT/qwen3vl_local/leadmot/LEADMOT_PLAN.md AutoMoT/qwen3vl_local/leadmot/LEADMOT_RUN.md AutoMoT/qwen3vl_local/leadmot/build_dataset.py AutoMoT/qwen3vl_local/leadmot/train.py AutoMoT/qwen3vl_local/leadmot/train.sh AutoMoT/qwen3vl_local/leadmot/eval.py AutoMoT/qwen3vl_local/leadmot/probe.py AutoMoT/qwen3vl_local/leadmot/config.py AutoMoT/qwen3vl_local/leadmot/projectors.py AutoMoT/qwen3vl_local/leadmot/query_bank.py AutoMoT/qwen3vl_local/leadmot/heads.py AutoMoT/qwen3vl_local/leadmot/mot_block.py AutoMoT/qwen3vl_local/leadmot/decoder.py AutoMoT/qwen3vl_local/leadmot/subgoal_prompt.py
git add AutoMoT/vae_standalone/train_patch_unpatch.py AutoMoT/vae_standalone/vae_reconstruct.py
```

commit 前先看：

```bash
git status
```

如果 status 里出现白名单外改动，停下来问用户。

push 前也问用户，不要替用户决定是否 push 到 main。

当用户同意新增/修改白名单外文件时：

- 在 `CLAUDE.md` 的默认追踪文件列表里添加同一个文件。
- 在本文件的文件修改范围 / git 规则里添加同一个文件。
- commit message 注明"按用户同意新增 XXX"。

当修改 AI 规则文档时：

- 修改 `CLAUDE.md` 时必须检查并同步 `AGENTS.md`。
- 修改 `AGENTS.md` 时必须检查并同步 `CLAUDE.md`。
- 如果新增的是项目技术事实，优先写入 `PROJECT_CONTEXT.md`；同时在 `CLAUDE.md` / `AGENTS.md` 加入口提醒或索引。
- 提交时精确执行：`git add CLAUDE.md AGENTS.md PROJECT_CONTEXT.md`（只 add 实际改动过的文件）。

---

## 6. 不要运行

本机只有源码，没有完整运行环境。不要运行这些重型或仿真相关操作：

- `lead/scripts/*.sh`
- `AutoMoT/test.sh`
- `AutoMoT/start_carla.sh`
- CARLA 仿真脚本
- 大规模数据集构建/下载脚本
- `pip install -r requirements.txt`
- 会下载大型模型、数据集、CARLA 的命令

可以做轻量静态检查，例如：

- `rg`
- `Get-Content`
- `git status`
- 小范围 Python 语法检查
- 针对单个文件的只读搜索

GPU 运行入口统一规则：

- SFT、GoalGen、LeadMoT、VAE patch/unpatch 以及白名单 runner 的训练、eval、probe、teacher / 推理入口默认都要自动寻找空闲 GPU。
- 文档示例不要写裸的 `export CUDA_VISIBLE_DEVICES=...` 选卡片段。**唯一允许的 pin 写法**：前置 `GPU_IDS=0` / `GPU_IDS=0,1,2,3`（白名单训练入口在 `GPU_IDS` 非空时跳过 nvidia-smi 选址，直接当 `CUDA_VISIBLE_DEVICES` 用）。
- 白名单内所有 GPU 运行入口默认自动选址：单进程入口默认用 `nvidia-smi` 自动挑 1 张最空闲 GPU，并覆盖已有 mask；`torchrun --nproc_per_node=N` 入口默认自动挑 N 张最空闲 GPU，并覆盖已有 mask，再按 `LOCAL_RANK` pin 到对应可见卡。`GPU_IDS` 显式 pin 时覆盖以上自动选址，卡数从 `GPU_IDS` 逗号数推断。
- 训练 launcher 的 `DDP_GPU_COUNT=N` / `NPROC_PER_NODE=N` 只表示默认自动选址时需要 N 张卡；具体卡号默认由脚本自动挑最空闲的 N 张。`GPU_IDS` 非空时，SFT / GoalGen / LeadMoT 这类 bash launcher 的卡数从 `GPU_IDS` 推断并忽略 `DDP_GPU_COUNT`；直接 `torchrun` 的 VAE 示例仍要让 `--nproc_per_node` 与 `GPU_IDS` 数量一致。
- 运行文档里每个单卡/多卡训练示例后面都要补显式 pin demo：单卡用 `GPU_IDS=0`，
  4 卡多卡用 `GPU_IDS=0,1,2,3`，照原命令保留其它 env。
- `eval_carla/run_eval.sh` 的 `--num-gpus N` / `EVAL_GPU_COUNT=N` 只表示闭环评测 worker 数；具体 GPU id 仍由 `nvidia-smi` 自动挑空闲卡，并为每张卡分配独立 CARLA 端口槽。
- 白名单内 bash launcher 开头必须保留 `ulimit -S -c 0 2>/dev/null || true`，禁用 core dump，避免工具进程异常时生成 `core.*`；新增运行入口也要继承该约定，若工作区已有 `core.*`，不要入库，先问用户是否清理。

训练 launcher 防覆盖目录约定（详见 PROJECT_CONTEXT.md §11）：

- 所有白名单训练入口（GoalGen / LeadMoT / SFT / VAE patch-unpatch）在用户给的 `OUTPUT_DIR`（或 `--output-dir`）下再套 `run_<RUN_TAG>/` 子目录，base 层维护 `latest` symlink，连跑同名 OUTPUT_DIR 不互相覆盖。
- `RUN_TAG` 默认 `$(date +%Y%m%d_%H%M%S)`，bash 段算一次再传给所有 worker；Python 入口用 rank0 strftime + `dist.broadcast_object_list` 同步。
- `NO_RUN_SUBDIR=1` 回退到顶层覆盖式行为（vae 入口也接受同名 env，并兼容旧名 `PATCH_UNPATCH_NO_RUN_SUBDIR`），仅排查兼容性时用。
- 共享缓存必须挂 base 层：`HF_HOME=${OUTPUT_DIR_BASE}/.hf_cache`；不能跟着 run 子目录，否则会每次重新下载。SFT 已不再保留 `runtime_teacher_data/` 共享 cache（teacher 在 train batch 内现场跑、不写盘）。
- 新增训练入口必须遵循同一范本。

运行文档路径口径：

- 运行手册默认当前目录就是远端 `AutoMoT/`。
- 命令示例统一写相对 `AutoMoT/` 的路径，例如 `bash qwen3vl_local/...`、
  `python qwen3vl_local/...`、`leaderboard/...`、`checkpoints/...`。
- 不要在文档里额外写切目录步骤，也不要给 `qwen3vl_local/...` 命令加 `AutoMoT/` 前缀。
- 只有仓库根视角的文件白名单、git add 路径、或明确说明 repo root 路径时，才保留
  `AutoMoT/` 前缀。
- LEAD 数据根目录统一假设在 `AutoMoT/lead_data`，也就是用户远端在 `AutoMoT/` 下
  将原始 LEAD 数据软链接后的目录。运行文档、脚本默认值和示例命令不要再写原始
  datashare 绝对路径；数据根写 `--data-root lead_data`，keyframes 写
  `--keyframes lead_data/keyframes_all_scenarios.json`。保存路径仍写
  `checkpoints/...`。

---

## 7. 和用户协作偏好

- 用简体中文交流。
- 改复杂代码前，先解释思路和方案取舍。
- 代码注释可以用简体中文，变量名/函数名保持英文。
- 不要把大段源码复制到文档里；文档写结论、边界、源码锚点。
- 如果发现 `PROJECT_CONTEXT.md` 与源码不一致，核对后同步修正文档。
