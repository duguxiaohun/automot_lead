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
- `AutoMoT/leaderboard/team_code/vlm_paradigm_a_runner.py`
- `AutoMoT/leaderboard/team_code/qwen3vl_instruct_paradigm_a_runner.py`
- `AutoMoT/leaderboard/team_code/qwen3vl_dit_goalgen_runner.py`
- `AutoMoT/qwen3vl_local/eval_carla/`（LeadMoT 闭环评测子包，全部子文件白名单内）
  - `__init__.py` / `EVAL_CARLA_PLAN.md` / `EVAL_CARLA_RUN.md`
  - `agent.py`
    （LEAD 风格 CARLA Bench2Drive 实时 agent：3 摄像头 1152×384 + IMU/GPS/Speedometer；
    按 ckpt 的 `decoder_config.use_bev` 自动决定是否声明/读取 LEAD 双 LiDAR；no-BEV 模型不产生未使用 LiDAR 输入；
    推理直接复用 `LeadOfflineMoTRunner`，按 ckpt 的 `decoder_config.use_bev` 自动决定 BEV 分支；
    每 5 tick 调一次模型，中间 tick PID 跟踪。**target_point / next_target_point 是未来 1.5s / 3s
    沿 global plan 弧长前瞻位置**（在线按 AutoMoT/CARLA route planner 做 route-lookahead 近似离线未来真值语义；ego frame
    约定 (x_forward, y_left)；低速 fallback 5m）；warmup 等真实 4 个 4Hz 历史帧，不复制首帧；
    UKF + route_planner + 基本 PID + SafetyMixin 兜底；Python class/function 已补中文 docstring，shell/HTML/CSS 关键逻辑块有中文注释）
  - `safety.py`
    （SafetyMixin：`stuck_helper` 累计 300 帧低速 → force_move 14 帧 creep / `parking_start`
    前 200 帧位移 < 6m 禁用 force_move / `parking_escape` 1500 帧窗口位移 < 5m 触发 phase1
    强转角 -0.65 + 油门 0.45 / 限速 35 km/h；与 mot_b2d_agent.py 行为完全一致）
  - `video_recorder.py`
    （input/debug/demo/grid 四路 mp4，ffmpeg crf=18/28；demo 在首帧通过
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
- `AutoMoT/qwen3vl_local/`（含 `goalgen/` 子包详见 PROJECT_CONTEXT.md §15；`eval_carla/` 子包详见上）
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
  （LEAD-MoT 快推理 decoder 子包及 v1 decoder-only 训练/eval/probe 入口：route(B,10,2) + waypoint(B,8,2)，Linear+cumsum head；gen 路独立 12 层 + frozen Qwen prefix K/V attention（不过 Linear）；hidden=1024=8x128 对齐 Qwen K/V 子空间；gen Q/K 按 `input_len + rope_deltas` 加 1D RoPE，language K/V 已由 Qwen prefill 带 M-RoPE 不重复旋转。训练时冻结 Qwen3-VL-Instruct 与 LeadBEVEncoder，只训练 LeadMoT decoder；GT 包含 route / future_waypoints 两类 ego-frame 累计点，head 内 Linear+cumsum 后直接对绝对点算 loss；`eval.py` 汇总 loss/ADE/FDE，`probe.py` 随机 case-level dump 预测与 GT 对比图。runner 必须用 `LocalQwen3VLInstructEngine` 单独跑 frozen Qwen prefill，只接受同源 HF `past_key_values`；不复用 AutoMoT InterleaveInferencer 的 `gen_context`，也不保留 AutoMoT legacy slow/fast 接口；`--leadmot-ckpt` 显式加载 decoder 权重，先读 checkpoint 的 `decoder_config.use_bev` 再实例化 decoder，并 `strict=True` 加载：`use_bev=True` 必须导入已有 BEV projector 参数，`use_bev=False` 则完全不实例化 / 不 forward BEV，禁止混入随机 BEV；不传 ckpt 仅作为随机初始化链路调试。详见 `leadmot/ARCHITECTURE.md`、`leadmot/LEADMOT_PLAN.md` 与 `PROJECT_CONTEXT.md §11.6/§11.7`）
- `AutoMoT/tools/SFT_V1_PLAN.md`
- `AutoMoT/tools/build_sft_dataset_v1.py`
- `AutoMoT/tools/sft_v1_train.sh`
- `AutoMoT/tools/sft_v1_loss_scale_plugin.py`
- `AutoMoT/tools/eval_sft_v1.py`
- `AutoMoT/tools/check_loss_mask.py`
- `AutoMoT/tools/SFT_V1_RUN.md`
- `AutoMoT/tools/tb_serve.sh`
- `AutoMoT/tools/probe_sft_v1.py`
  （以上 9 个是 LoRA SFT v1 / TB / probe 相关；`tb_serve.sh` 是通用 TensorBoard 启动器，GoalGen 也复用；`probe_sft_v1.py` 是随机场景 case-level dump；`AutoMoT/tools/` 下其它原始脚本仍为只读参考）
- `AutoMoT/tools/SFT_V2_PLAN.md`
- `AutoMoT/tools/SFT_V2_RUN.md`
- `AutoMoT/tools/build_sft_dataset_v2_teacher.py`
- `AutoMoT/tools/sft_v2_loss_scale_plugin.py`
- `AutoMoT/tools/sft_v2_train.sh`
- `AutoMoT/tools/check_loss_mask_v2.py`
- `AutoMoT/tools/inspect_teacher_outputs.py`
  （以上 7 个是 SFT v2 升级：长期数据集只保留 `v2_pending` 占位 jsonl；冻结 base Qwen + PRIVILEGED prompt 的 ANALYSIS 真值由 `sft_v2_train.sh` 在**首次**训练启动时一次性物化到 runtime 目录并写 `manifest.json`，之后任意卡数启动通过 manifest（schema_version=2 + max_samples==0 + model_dir/seed/gen 参数 + pending/runtime 行数严格匹配）校验，校验通过才直接复用（GPU 数无关），无需任何额外参数；32 条 debug cache、半截 val、`--max-samples N` 跑出来的不写 manifest 不会被误复用；无 manifest 的 final/rank 残留默认清掉重物化，避免旧 teacher 分片被 fingerprint 去重误用；改 prompt / keyframes 后想强制重跑 → `RUNTIME_TEACHER_REFRESH=1` 或手动 `rm -rf runtime_teacher_data/`；`check` 模式例外，默认 REFRESH=1 + 独立 `runtime_teacher_check_data/` 目录。也可由 `inspect_teacher_outputs.py --live` 做训练前预览；student 全段都算 loss（ANALYSIS body 0.3 / 起手字面 `ANALYSIS:`、段切换字面 `\nSTATUS:` / `\nSUBGOAL:`、STATUS+SUBGOAL event_name、可能进入 context 的 tail/EOS 全部 1.0；v2.0 旧版字面 mask=0 是致命陷阱，详见 PROJECT_CONTEXT.md §18.5）；`build_sft_dataset_v1.py` 同时承载 v1/v2 两个 `--mode`；`eval_sft_v1.py` / `probe_sft_v1.py` 自动按 jsonl 字段检测 v1/v2；`check_loss_mask_v2.py` 用于已物化 v2 jsonl 的 token 级静态 sanity）
- `AutoMoT/qwen3vl_local/goalgen/GOALGEN_PLAN.md`
- `AutoMoT/qwen3vl_local/goalgen/GOALGEN_RUN.md`
- `AutoMoT/qwen3vl_local/goalgen/build_dataset.py`
- `AutoMoT/qwen3vl_local/goalgen/train.py`
- `AutoMoT/qwen3vl_local/goalgen/train.sh`
- `AutoMoT/qwen3vl_local/goalgen/eval.py`
- `AutoMoT/qwen3vl_local/goalgen/probe.py`
  （以上 7 个是子目标 latent 生成路线 v1/v2 共用数据/训练/eval/probe/文档，详见 PROJECT_CONTEXT.md §15；MD 与代码同位于 goalgen 子包内，不要再在 tools/ 下创建重复 MD。GoalGen 训练默认必须导入 `AutoMoT/checkpoints/patch_unpatch_v1/latest/weights/patch_unpatch_best.safetensors`（再兜底无 run_subdir 与最新 `run_*`）并冻结；找不到直接报错，不再随机初始化 patch/unpatch）
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
git add AutoMoT/qwen3vl_local/eval_carla/__init__.py AutoMoT/qwen3vl_local/eval_carla/EVAL_CARLA_PLAN.md AutoMoT/qwen3vl_local/eval_carla/EVAL_CARLA_RUN.md AutoMoT/qwen3vl_local/eval_carla/agent.py AutoMoT/qwen3vl_local/eval_carla/safety.py AutoMoT/qwen3vl_local/eval_carla/video_recorder.py AutoMoT/qwen3vl_local/eval_carla/visualizer.py AutoMoT/qwen3vl_local/eval_carla/scenario_picker.py AutoMoT/qwen3vl_local/eval_carla/aggregate.py AutoMoT/qwen3vl_local/eval_carla/run_eval.sh AutoMoT/qwen3vl_local/eval_carla/webapp/__init__.py AutoMoT/qwen3vl_local/eval_carla/webapp/app.py AutoMoT/qwen3vl_local/eval_carla/webapp/templates/index.html AutoMoT/qwen3vl_local/eval_carla/webapp/static/style.css
git add AutoMoT/qwen3vl_local/__init__.py AutoMoT/qwen3vl_local/cache_utils.py AutoMoT/qwen3vl_local/engine.py AutoMoT/qwen3vl_local/image_io.py AutoMoT/qwen3vl_local/prompt_pipeline.py
git add AutoMoT/qwen3vl_local/goalgen/__init__.py AutoMoT/qwen3vl_local/goalgen/vae.py AutoMoT/qwen3vl_local/goalgen/prompt.py AutoMoT/qwen3vl_local/goalgen/qwen_kv.py AutoMoT/qwen3vl_local/goalgen/keyframes.py AutoMoT/qwen3vl_local/goalgen/dit.py AutoMoT/qwen3vl_local/goalgen/flow.py
git add AutoMoT/leaderboard/team_code/qwen3vl_dit_goalgen_runner.py
git add AutoMoT/tools/SFT_V1_PLAN.md AutoMoT/tools/SFT_V1_RUN.md AutoMoT/tools/build_sft_dataset_v1.py AutoMoT/tools/sft_v1_train.sh AutoMoT/tools/sft_v1_loss_scale_plugin.py AutoMoT/tools/eval_sft_v1.py AutoMoT/tools/check_loss_mask.py AutoMoT/tools/tb_serve.sh AutoMoT/tools/probe_sft_v1.py
git add AutoMoT/tools/SFT_V2_PLAN.md AutoMoT/tools/SFT_V2_RUN.md AutoMoT/tools/build_sft_dataset_v2_teacher.py AutoMoT/tools/sft_v2_loss_scale_plugin.py AutoMoT/tools/sft_v2_train.sh AutoMoT/tools/check_loss_mask_v2.py AutoMoT/tools/inspect_teacher_outputs.py
git add AutoMoT/qwen3vl_local/goalgen/GOALGEN_PLAN.md AutoMoT/qwen3vl_local/goalgen/GOALGEN_RUN.md AutoMoT/qwen3vl_local/goalgen/build_dataset.py AutoMoT/qwen3vl_local/goalgen/train.py AutoMoT/qwen3vl_local/goalgen/train.sh AutoMoT/qwen3vl_local/goalgen/eval.py AutoMoT/qwen3vl_local/goalgen/probe.py
git add AutoMoT/qwen3vl_local/leadmot/__init__.py AutoMoT/qwen3vl_local/leadmot/ARCHITECTURE.md AutoMoT/qwen3vl_local/leadmot/LEADMOT_PLAN.md AutoMoT/qwen3vl_local/leadmot/LEADMOT_RUN.md AutoMoT/qwen3vl_local/leadmot/build_dataset.py AutoMoT/qwen3vl_local/leadmot/train.py AutoMoT/qwen3vl_local/leadmot/train.sh AutoMoT/qwen3vl_local/leadmot/eval.py AutoMoT/qwen3vl_local/leadmot/probe.py AutoMoT/qwen3vl_local/leadmot/config.py AutoMoT/qwen3vl_local/leadmot/projectors.py AutoMoT/qwen3vl_local/leadmot/query_bank.py AutoMoT/qwen3vl_local/leadmot/heads.py AutoMoT/qwen3vl_local/leadmot/mot_block.py AutoMoT/qwen3vl_local/leadmot/decoder.py
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

- SFT v1/v2、GoalGen、LeadMoT、VAE patch/unpatch 以及白名单 runner 的训练、eval、probe、teacher / 推理入口默认都要自动寻找空闲 GPU。
- 文档示例不要写 shell 手动设置 `CUDA_VISIBLE_DEVICES` 的选卡片段。
- 白名单内所有 GPU 运行入口都按用户要求彻底放弃手动 `CUDA_VISIBLE_DEVICES` 选择：单进程入口默认用 `nvidia-smi` 自动挑 1 张最空闲 GPU，并覆盖已有 mask；`torchrun --nproc_per_node=N` 入口默认自动挑 N 张最空闲 GPU，并覆盖已有 mask，再按 `LOCAL_RANK` pin 到对应可见卡。
- 训练 launcher 的 `DDP_GPU_COUNT=N` / `NPROC_PER_NODE=N` 只表示需要 N 张卡；具体卡号仍由脚本自动挑最空闲的 N 张，不提供“尊重外部 mask”的分支。
- `eval_carla/run_eval.sh` 的 `--num-gpus N` / `EVAL_GPU_COUNT=N` 只表示闭环评测 worker 数；具体 GPU id 仍由 `nvidia-smi` 自动挑空闲卡，并为每张卡分配独立 CARLA 端口槽。

训练 launcher 防覆盖目录约定（详见 PROJECT_CONTEXT.md §11）：

- 所有白名单训练入口（GoalGen / LeadMoT / SFT v1 / SFT v2 / VAE patch-unpatch）在用户给的 `OUTPUT_DIR`（或 `--output-dir`）下再套 `run_<RUN_TAG>/` 子目录，base 层维护 `latest` symlink，连跑同名 OUTPUT_DIR 不互相覆盖。
- `RUN_TAG` 默认 `$(date +%Y%m%d_%H%M%S)`，bash 段算一次再传给所有 worker；Python 入口用 rank0 strftime + `dist.broadcast_object_list` 同步。
- `NO_RUN_SUBDIR=1` 回退到顶层覆盖式行为（vae 入口也接受同名 env，并兼容旧名 `PATCH_UNPATCH_NO_RUN_SUBDIR`），仅排查兼容性时用。
- 共享缓存必须挂 base 层：`HF_HOME=${OUTPUT_DIR_BASE}/.hf_cache`、SFT v2 `runtime_teacher_data/`（由 manifest 严格校验复用）；不能跟着 run 子目录，否则会每次重物化。
- 新增训练入口必须遵循同一范本。

---

## 7. 和用户协作偏好

- 用简体中文交流。
- 改复杂代码前，先解释思路和方案取舍。
- 代码注释可以用简体中文，变量名/函数名保持英文。
- 不要把大段源码复制到文档里；文档写结论、边界、源码锚点。
- 如果发现 `PROJECT_CONTEXT.md` 与源码不一致，核对后同步修正文档。
