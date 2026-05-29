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

把 `lead/` 采集/训练出来的 CARLA 离线数据，伪装成 `AutoMoT/` 在线 agent 的输入，让 AutoMoT 的 Qwen3-VL 慢推理路径可以在 LEAD 数据上离线跑起来，并逐步分析两边数据分布、坐标系、RGB/LiDAR/BEV/target_point 的差异。

当前主要战场：

- `AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py`
- `AutoMoT/leaderboard/team_code/vlm_paradigm_a_runner.py`
- `PROJECT_CONTEXT.md`

---

## 3. 当前技术状态

关键结论以 `PROJECT_CONTEXT.md` 为准，下面只是快速索引：

- `lead/`：数据采集、训练、闭环评测仓库。CARLA 20Hz，每 5 tick 落盘 1 帧，即 4Hz。
- `AutoMoT/`：在线驾驶仓库。慢路径是 Qwen3-VL + KV cache，快路径依赖 BEV encoder + DP heads。
- 当前离线 runner 的实际可用路径是慢推理路径：`kv_cache_fixed_inference(...)`。
- 快推理路径默认禁用：`enable_fast_inference=False`。
- runner 已切换到 LEAD 风格的 `LeadTransfuserBackbone` / `LeadBEVEncoder`，但其输出与 AutoMoT 原快推理 decoder shape 不兼容，因此不能直接打开快推理。
- LEAD RGB 是三视角拼接 `(W=1152, H=384)`；当前慢推理直接喂给 Qwen3-VL，不切片、不 resize、不选前视。
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
- `AutoMoT/qwen3vl_local/`
- `AutoMoT/tools/SFT_V1_PLAN.md`
- `AutoMoT/tools/build_sft_dataset_v1.py`
- `AutoMoT/tools/sft_v1_train.sh`
- `AutoMoT/tools/eval_sft_v1.py`
- `AutoMoT/tools/check_loss_mask.py`
- `AutoMoT/tools/SFT_V1_RUN.md`
  （以上 6 个是 LoRA SFT v1 微调相关；`AutoMoT/tools/` 下其它原始脚本仍为只读参考）

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
git add AutoMoT/qwen3vl_local/__init__.py AutoMoT/qwen3vl_local/cache_utils.py AutoMoT/qwen3vl_local/engine.py AutoMoT/qwen3vl_local/image_io.py AutoMoT/qwen3vl_local/prompt_pipeline.py
git add AutoMoT/tools/SFT_V1_PLAN.md AutoMoT/tools/SFT_V1_RUN.md AutoMoT/tools/build_sft_dataset_v1.py AutoMoT/tools/sft_v1_train.sh AutoMoT/tools/eval_sft_v1.py AutoMoT/tools/check_loss_mask.py
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

---

## 7. 和用户协作偏好

- 用简体中文交流。
- 改复杂代码前，先解释思路和方案取舍。
- 代码注释可以用简体中文，变量名/函数名保持英文。
- 不要把大段源码复制到文档里；文档写结论、边界、源码锚点。
- 如果发现 `PROJECT_CONTEXT.md` 与源码不一致，核对后同步修正文档。
