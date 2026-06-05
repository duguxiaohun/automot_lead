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
- `AutoMoT/qwen3vl_local/`

## 1. 目录角色

| 目录/文件 | 角色 |
|---|---|
| `lead/` | 数据采集、训练、闭环评测参考仓库。只读 |
| `AutoMoT/` | 在线驾驶仓库；当前本地改造主要放这里 |
| `AutoMoT/qwen3vl_local/` | 本地 Qwen3-VL-Instruct frozen prefill、prompt、GoalGen、LeadMoT |
| `AutoMoT/tools/` | SFT v1/v2 数据、训练、eval、probe |
| `AutoMoT/vae_standalone/train_patch_unpatch.py` | patch/unpatch 端到端重建训练 |
| `0026.json` | LEAD meta 固定参考样本，只读，绝对不要入库 |
| `keyframes_all_scenarios.json` | 远端数据参考，只读 |

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

## 7. GoalGen

文件：

- `qwen3vl_local/goalgen/build_dataset.py`
- `qwen3vl_local/goalgen/train.py`
- `qwen3vl_local/goalgen/eval.py`
- `qwen3vl_local/goalgen/probe.py`
- `qwen3vl_local/goalgen/GOALGEN_RUN.md`

语义：

- 输入：history RGB -> frozen Qwen prefill/KV；history/target RGB -> frozen VAE latent。
- 目标：生成未来 subgoal keyframe latent。
- 训练：rectified flow，`z_t=(1-t)z0+t z1`，预测 `v=z1-z0`。
- 推理：Euler 从 t=0 到 t=1，decode 成 RGB。

v1/v2：

| 项 | v1 | v2 |
|---|---|---|
| 数据 mode | `--mode v1` | `--mode v2` |
| transition | 4 类，含 initial/final 两端 | 2 类，只保留 middle 之间 |
| 默认训练 | 从零 | 从 v1 `latest/best.pt` warm start |
| 代码 | 同一套 | 同一套 |

当前共享架构，不属于某个 dataset mode：

- VAE latent `(C=16,T=1,H=48,W=144)`
- patch size `4`，token 网格 `12*36`
- hidden `1024`，heads `8`
- DiT layers `12`
- Qwen 36 层切 12 段，head_dim=128

## 8. SFT v1 / v2

SFT v1：

- `tools/build_sft_dataset_v1.py`
- `tools/sft_v1_train.sh`
- `tools/eval_sft_v1.py`
- `tools/probe_sft_v1.py`
- `tools/SFT_V1_RUN.md`

v1 assistant 使用固定 `ANALYSIS: Observations recorded.`，主要训练
STATUS/SUBGOAL 事件名。

SFT v2：

- `tools/build_sft_dataset_v1.py --mode v2` 生成 `v2_pending`。
- `tools/build_sft_dataset_v2_teacher.py` 用 frozen Qwen + PRIVILEGED prompt 物化真实 ANALYSIS。
- `tools/sft_v2_train.sh` 首次训练启动时物化 runtime teacher cache，后续按 manifest 复用。
- `tools/SFT_V2_RUN.md` 是操作入口。

v2 loss 规则：ANALYSIS body 默认权重 0.3；`ANALYSIS:`、`\nSTATUS:`、
`\nSUBGOAL:` 字面、事件名、EOS/tail 都参与 loss。旧版“结构字面 mask=0”
是致命陷阱，不要恢复。

eval 端固定坑：

- Qwen3-VL 上 PEFT wrapper forward 可能错位；默认 `merge_and_unload`。
- v2 `max_gen_tokens` 需要 256；96 会截到只剩 ANALYSIS。
- partial-continue fallback 是永久兜底，不代表模型健康。

## 9. VAE Patch/Unpatch

入口：`AutoMoT/vae_standalone/train_patch_unpatch.py`。

训练目标：`image -> VAE.encode -> patch -> unpatch -> VAE.decode -> image`。
VAE 冻结，只训练 patch/unpatch。产物 `patch_unpatch_*.safetensors` 可被
`DiTMoT.load_patch_unpatch` 直接加载，key 与 `self.patch` / `self.unpatch` 对齐。

DDP 选卡：Python 内部 rank0 选卡，写临时文件，其它 rank 读取，避免每 worker
各自 `nvidia-smi` 导致 GPU 子集 race。

## 10. GPU 选址统一规则

适用：SFT v1/v2、GoalGen、LeadMoT、VAE patch/unpatch、白名单 runner 的训练、
eval、probe、teacher / 推理入口。

- 默认调用 `nvidia-smi` 自动挑空闲 GPU，并覆盖外层残留的 `CUDA_VISIBLE_DEVICES`。
- 单进程默认挑 1 张；进程内通常用 `cuda:0`。
- `torchrun --nproc_per_node=N` 默认挑 N 张，并按 `LOCAL_RANK` pin。
- `DDP_GPU_COUNT=N` / `NPROC_PER_NODE=N` 只表示需要 N 张卡，具体卡号仍自动挑。
- 文档示例不要写 shell 手动 `CUDA_VISIBLE_DEVICES=...`。
- 显式 `--device cpu` / `--device cuda:N` 的 Python 入口通常视为用户锁设备，不覆盖。
- GoalGen eval/probe 的 `--gpu N` 只在单进程下锁进程内 GPU；默认保持 0。

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
- SFT v2 runtime teacher cache 挂在 base 层，靠 manifest 复用。

## 12. 不要做

- 不要改 `lead/`。
- 不要把 `0026.json` 或 `keyframes_all_scenarios.json` 入库。
- 不要运行 CARLA、`AutoMoT/test.sh`、`start_carla.sh`、大规模下载或安装命令。
- 不要把 AutoMoT legacy slow/fast 接口重新接回本地 Qwen/LeadMoT 路线。
- 不要把当前共享 GoalGen 架构描述成某个 dataset mode 专属架构。

## 13. 快速导航

| 任务 | 文档 |
|---|---|
| SFT v1 跑法 | `AutoMoT/tools/SFT_V1_RUN.md` |
| SFT v2 跑法 | `AutoMoT/tools/SFT_V2_RUN.md` |
| GoalGen 跑法 | `AutoMoT/qwen3vl_local/goalgen/GOALGEN_RUN.md` |
| LeadMoT 跑法 | `AutoMoT/qwen3vl_local/leadmot/LEADMOT_RUN.md` |
| LeadMoT 架构 | `AutoMoT/qwen3vl_local/leadmot/ARCHITECTURE.md` |
| 规则入口 | `AGENTS.md` / `CLAUDE.md` |
