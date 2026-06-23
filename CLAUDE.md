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
  另含 LEAD RGB 读取、显式 prefill/decode、KV cache summary 与可选 `torch.save`
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
| `AutoMoT/qwen3vl_local/` | Qwen3-VL-Instruct 本地 helper 包；其中 `tb_serve.sh` 是通用 TensorBoard 启动器（SFT / GoalGen / LeadMoT / VAE 共用），`goalgen/` 子包是 §15 新路线全部模块（vae/prompt/qwen_kv/keyframes/dit/flow），`eval_carla/` 子包是上述闭环评测子包 |
| `AutoMoT/qwen3vl_local/sft/__init__.py` / `SFT_PLAN.md` / `SFT_RUN.md` / `build_dataset.py` / `build_teacher.py` / `train.py` / `train.sh` / `eval.py` / `probe.py` / `check_loss_mask.py` / `inspect_teacher_outputs.py` | 统一 LoRA SFT 子包（已废弃 v1/v2 双轨与 ms-swift）：`SFT_PLAN.md` / `SFT_RUN.md` 是设计与运行入口；`build_dataset.py` 只产 `dataset_version="pending"` jsonl（assistant 含 `__TEACHER_PENDING__` 占位）；`train.sh` → `train.py` 用 `peft.LoraConfig` + `get_peft_model` 直接把 LoRA 注入 base，torch DDP + 手写 train loop，每个 train batch 内禁用 adapter，并调用底层 Qwen base model 现场 greedy 生成 ANALYSIS 真值（避开 `PeftModel.generate` 的 Qwen3-VL 生成错位问题），再启用 adapter 跑 student forward + per-token 加权 loss（ANALYSIS body `SFT_ANALYSIS_WEIGHT`/默认 0.5，学习大致语言推理但不逐字压过状态监督；其余 assistant 段 1.0，prompt 段 0）。**不再离线物化 teacher、不再写 manifest、不再有 runtime_teacher_data 复用机制**；`build_teacher.py` 仅作为可选离线 dump 工具，不被训练入口自动调用。`eval.py` / `probe.py` 默认 `merge_and_unload` 加载 LoRA，case dump 保存 `expert_analysis.txt` / `language_compare.json` 对比 base-teacher 专家语言与模型 ANALYSIS；`probe.py` 的 token loss 使用训练同款权重汇总；`check_loss_mask.py` 验 train.py 内置 mask；`inspect_teacher_outputs.py` 支持 `--live` 现场重跑 teacher 抽检 |
| `AutoMoT/qwen3vl_local/sft_v2/__init__.py` / `SFT_V2_PLAN.md` / `SFT_V2_RUN.md` / `prompts.py` / `build_dataset.py` / `train.py` / `train.sh` / `eval.py` / `probe.py` / `check_loss_mask.py` | 按用户同意新增到白名单：SFT v2 两段式串行选择题子包。输入仍为 LEAD stitched RGB + 语言 prompt；stage-1 只列 `SCENE_CHOICES` 并输出 `SCENE`，stage-2 作为同一条对话的后续 user prompt，按预测 scene 的 `EVENT_SEQUENCE` 输出 `STATUS/SUBGOAL`，推理时必须复用 stage-1 已吃图像和场景 prompt 后的 KV cache；默认 `--samples-per-scenario 0` 全量保留合法候选，默认 `--wrong-scene-ratio 0.15` 只增强 train rows；不再有 ANALYSIS / teacher / pending cache；训练 loss 只监督 scene/status/subgoal 值 token，格式 token 为 0 loss；LoRA 默认只注入语言侧 Linear，视觉侧通过 `--lora-vision-scope` / `LORA_VISION_SCOPE` 选择 `off` / `merger` / `last4` / `all` 四档（`--lora-vision` / `LORA_VISION=1` 作为 `all` 的 legacy 别名保留）；开启视觉 LoRA 时默认带"视觉组单独 LR 倍率 `--vision-lr-scale=0.1`（受 `--max-vision-lr-scale=0.25` 上限约束）+ 分组梯度裁剪 `--language-clip-norm=1.0` / `--vision-clip-norm=0.3` + TB 观测 `grad_norm/{language,vision}` / `param_norm/lora_{language,vision}` / `vision_guard_bad_steps` + `STRICT_VISION_SCOPE=1` 命名漂移硬拒绝 + `VISION_GUARD_ENABLED=1` 运行时熔断"保险；熔断时写 `fuse_stop_step_<N>/` 与 `fuse_reason.txt`，并跳过正常 `final/` 保存，防止视觉表征被冲坏且避免误用异常产物；base Qwen checkpoint 始终只读，训练只保存 adapter delta，并写 `sft_v2_adapter_config.json`（含 `lora_vision_scope` 与保险参数）；eval/probe 加载前按 adapter 配置判断普通 LoRA / 视觉 LoRA 并校验权重 key，不一致直接拒绝。自由生成评估中 scene 不在白名单则中断，scene 合法但错误时仍按预测 scene 进入 stage-2 并用串行口径计错，同时输出 `valid_total` / `*_valid_scene` 指标。运行文档见 `SFT_V2_RUN.md` |
| `AutoMoT/qwen3vl_local/sft_v3/` 子包（含 `__init__.py` / `SFT_V3_PLAN.md` / `SFT_V3_RUN.md` / `prompts.py` / `build_dataset.py` / `train.py` / `train.sh` / `eval.py` / `probe.py` / `check_loss_mask.py` / `test_memory_update.py` / `test_kv_reuse.py` / `test_gt_leak_filter.py`） | 按用户同意新增到白名单：SFT v3 代码已落地。训练单元为 sub-scenario 时间序列（`[f1-δ, f3]`），Memory 为学生自维护文本状态（场景/状态/子目标 + EGO_TO_GOAL_XY），每帧三步内循环（step1 纯视觉分析、step2 场景判断、step3 状态/子目标判断）并用 teacher/student 蒸馏；Phase B 每帧开头弱纠偏 scene=GT 反向学习“对的别改”；δ 允许 0 且只封顶 10，`EGO_TO_GOAL_XY` 严格来自 meta `next_target_points[-1]` 并在帧末预取下一帧，step3 触发统一走 `should_trigger_step3`。`train.sh` 默认 `ddp`（历史模式名），每卡默认 batch=1；多卡训练采用 work-stealing + local-SGD：不包 DDP、不静态分片、不截断尾部，通过 TCPStore 抢 episode，NCCL collective 前先 TCPStore rendezvous，先广播 rank0 LoRA 初始权重，按本轮 optimizer step 数加权平均 LoRA 参数，并且 `checkpoint-*` / `final/` 都在参数平均后保存；sync 日志/TB 记录 `all_rank_steps`、`round_eps`、`total_eps` 用于审计训练量；LoRA 视觉接口与 v2 同构并默认 `off`，保存 adapter delta + `sft_v3_adapter_config.json`。 |
| `AutoMoT/qwen3vl_local/goalgen/GOALGEN_PLAN.md` / `GOALGEN_RUN.md` / `GOALGEN_V1.md` / `GOALGEN_V2.md` | 子目标 latent 生成路线文档；PLAN/RUN 为索引，v1/v2 细节分别写入 `GOALGEN_V1.md` / `GOALGEN_V2.md`（与 goalgen 子包代码同目录） |
| `AutoMoT/qwen3vl_local/goalgen/build_dataset.py` / `train.py` / `train.sh` / `eval.py` / `probe.py` | GoalGen v1/v2 共用数据集构建 + 训练入口（DDP / 单卡 / check 三模式）+ 离线 eval（latent / pixel / velocity + PNG + TB scalar/image，支持 torchrun 分片）+ `probe.py` 随机场景 case-level dump；训练默认必须导入 `AutoMoT/checkpoints/patch_unpatch_v1/latest/weights/patch_unpatch_best.safetensors`（再兜底无 run_subdir 与最新 `run_*`）并冻结，找不到直接报错，不再随机初始化 patch/unpatch |
| `AutoMoT/qwen3vl_local/leadmot/` 子包（含 v1 训练/eval/probe 文件 + `subgoal_prompt.py`） | LEAD-MoT 快推理 decoder：route(B,10,2) + waypoint(B,8,2)，Linear+cumsum head；gen 路独立 12 层 + frozen Qwen prefix K/V attention（不过 Linear）；hidden=1024=8×128 对齐 Qwen K/V 子空间；status 按 AutoMoT velocity MLP + 共享 WaypointInputAdaptor；block 用 Qwen3 风格 RMSNorm + q/k_norm + SwiGLU；gen Q/K 按 `input_len + rope_deltas` 加 1D RoPE，language K/V 已由 Qwen prefill 带 M-RoPE 不重复旋转。v1 训练入口只训练 decoder，冻结 Qwen3-VL-Instruct 与 LeadBEVEncoder；GT 包含 route / future_waypoints 两类 ego-frame 累计点，head 内 Linear+cumsum 后直接对绝对点算 loss；`eval.py` 汇总 loss/ADE/FDE，`probe.py` 随机 case-level dump 预测与 GT 对比图。runner 必须用 `LocalQwen3VLInstructEngine` 单独跑 frozen Qwen prefill，只接受同源 HF `past_key_values`；**不能复用 AutoMoT InterleaveInferencer 的 `gen_context`**，也不保留 AutoMoT legacy slow/fast 接口；`--leadmot-ckpt` 显式加载 decoder 权重，先读 checkpoint 的 `decoder_config.use_bev` 再实例化 decoder，并 `strict=True` 加载：`use_bev=True` 必须导入已有 BEV projector 参数，`use_bev=False` 则完全不实例化 / 不 forward BEV，禁止混入随机 BEV；不传 ckpt 仅作为随机初始化链路调试。**`use_subgoal`（离线专用）**：与 `use_bev` 正交的 prefix-only 开关，开启时 build_dataset `--with-subgoal-fields` 反查 `keyframes_all_scenarios.json` 写入 scenario/run_id/status/subgoal/subgoal_frame/subgoal_rgb_path/subgoal_lookup_ok 字段；训练/eval/probe 通过 `LeadMoTTrainRuntime._run_subgoal_qwen_prefill` 在 prefix 多喂 1 张 SUBGOAL stitched RGB + `[GROUND_TRUTH_STATE]` 文本块（system/user prompt 由 `leadmot/subgoal_prompt.py` 提供），prompt 内仍保留原 navigation 文本以维持 tp/ntp/final_goal 对齐；ckpt 里 `decoder_config.use_subgoal` 与训练 args 必须严格一致，cross-load 由 `_require_subgoal_match` 拒绝；state_dict 形状不受影响（subgoal 不引入新模块），但 prefix KV 分布不兼容；`mot_lead_offline_runner.py` 会按 ckpt 自动走 subgoal prefill 并要求 clip 注入 subgoal 字段，CLI demo 可通过 `--keyframes` 自动反查；eval_carla 在线 agent 暂不支持该开关，加载 use_subgoal=True ckpt 时立即 `raise NotImplementedError`。详见 `leadmot/ARCHITECTURE.md`、`leadmot/LEADMOT_PLAN.md` 与 `PROJECT_CONTEXT.md` |
| `AutoMoT/vae_standalone/train_patch_unpatch.py` | patch / unpatch 端到端图像重建训练脚本：image→VAE.encode→patch→unpatch→VAE.decode→image，VAE 冻结；产物 `patch_unpatch_*.safetensors` 可被 `DiTMoT.load_patch_unpatch` 直接加载并默认冻结。`AutoMoT/vae_standalone/` 下其它原始文件（vwm/、config/、weights/ 等）仍为只读参考，除非已单独列入白名单 |
| `AutoMoT/vae_standalone/vae_reconstruct.py` | 按用户同意新增到白名单：VAE / patch-unpatch 诊断脚本，支持 VAE-only 与 VAE+patch/unpatch 两种重建链路、批量 loss、TensorBoard 与随机小批量 PNG 对比可视化 |
| `CLAUDE.md` | 本规则文件（仅在调整规则时修改） |
| `AGENTS.md` | 通用 AI / coding agent 入口说明文件 |

**其它所有文件**（`lead/` 整个目录、`AutoMoT/` 其余文件、配置等）**不准动**——
它们是用户从远程服务器同步下来的参考源码，作只读资料用。

特别提示：仓库根目录有 **`0026.json`**（用户提供的 LEAD meta.pkl 转 JSON 标准参考样本，
详见 PROJECT_CONTEXT.md §2.3）——**绝对禁止修改其内容，绝对禁止 `git add 0026.json`**。
它是固定参考"标尺"，修改或入库都会破坏历史推论的可追溯性。

> 例外流程：如果确实有必要新建文件或在其他已有文件上打补丁（例如发现 utils 类
> 缺函数、需要新建测试脚本等），**必须先用 AskUserQuestion 或直接在对话里
> 请求用户确认**。得到同意后，按 §3 把新文件纳入 git 追踪列表。

---

## 3. Git 提交规则

仓库根目录：`c:\Users\11509\Desktop\automot_lead`
远程：`https://github.com/duguxiaohun/automot_lead.git`（main 分支）

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
- `AutoMoT/qwen3vl_local/__init__.py`
- `AutoMoT/qwen3vl_local/cache_utils.py`
- `AutoMoT/qwen3vl_local/engine.py`
- `AutoMoT/qwen3vl_local/image_io.py`
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

### 硬性规则

- **禁止** `git add .` / `git add -A` / `git add *`，会污染仓库
- **禁止** `git add lead/` / `git add AutoMoT/`（除了白名单里那一个具体路径）
- **禁止** `git add 0026.json`——它是 LEAD meta 参考样本，永远只读、永远不入库
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

# 3. commit
git commit -m "<一句话说明本次改了什么、为什么>"

# 4. push
git push
```

### 当用户同意新建/修改白名单外文件时

1. 在本文件 §3 的"默认追踪文件"列表里**添加新文件**
2. 在 `AGENTS.md` 的文件修改范围 / git 规则里同步添加同一个文件
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
