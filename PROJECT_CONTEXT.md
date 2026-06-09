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
| `AutoMoT/qwen3vl_local/` | 本地 Qwen3-VL-Instruct frozen prefill、prompt、GoalGen、LeadMoT、通用 TensorBoard 启动器 `tb_serve.sh` |
| `AutoMoT/qwen3vl_local/sft/` | SFT v1/v2 数据、训练、eval、probe |
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

`LocalQwen3VLInstructEngine.prefill` 默认用 `logits_to_keep=1`，因为后续 decode
只需要最后一个位置的首 token logits；不再为整段 prompt 构造完整 vocab logits。

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
- 本地 Qwen 文本生成只需要最后一位 logits；`qwen3vl_local.engine` 和
  `vlm_paradigm_a_runner.py` 的 qwen backend 都默认用 `logits_to_keep=1` 裁掉整段 prompt logits。
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
- LeadMoT `ema_state_dict` 当前 schema 是 `{"decay": ..., "shadow": {...}}`；`eval.py` /
  `probe.py` 会 unwrap `shadow` 后再 strict load。

### 6.1 LeadMoT CARLA 闭环评测（eval_carla）

入口：`AutoMoT/qwen3vl_local/eval_carla/`，操作文档：
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
  - 在线 eval_carla 用 `scenario_picker.load_route_endpoint(route_id)` 读取
    `lead/data/benchmark_routes/bench2drive220/<Scenario>/<route_id>.xml`
    最后一个 waypoint，再按当前 ego pose 转 ego frame；找不到 XML 时才临时 fallback 到
    RoutePlanner 剩余 route 末端；
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
  `leaderboard_evaluator.py` 在 worker 进程内启动并清理，避免双重启动抢端口。
- **输出按跑法分目录**：
  - `<signature>/route<id>/` 视频与 `<signature>/eval_per_route/eval_<id>.json` 跨跑法共享
    （断点续跑）
  - `<signature>/runs/<RUN_LABEL>/{summary_all.json, scenarios/, run_manifest.json}`
    本批次聚合，按 `full` / `scenario_X` / `random_NN_SK` / `routes_A+B` 等自动命名
  - `<signature>/summary_all.json` 是跨批次总聚合（所有已评估 route）
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

语义：

- 输入：history RGB -> frozen Qwen prefill/KV；history/target RGB -> frozen VAE latent。
- 目标：生成未来 subgoal keyframe latent。
- 训练：rectified flow，`z_t=(1-t)z0+t z1`，预测 `v=z1-z0`。
- 推理：Euler 从 t=0 到 t=1，decode 成 RGB。
- **z0 默认为纯噪声 `z0 ~ N(0, I)`**（`z0_prior_alpha=0.0`）：
  之前默认 `alpha=1.0, sigma=1.0` 把当前帧 latent 掺进 z0 → 低 t 区域 `z_t` 主要由
  "当前帧 + 噪声"主导，`v_target = z1 - z0` 里 z0 含 z_current → 模型偷懒学
  "还原当前帧"而不是 subgoal。**统一改回纯噪声起点**（flow.py / train.py / eval.py /
  probe.py / train.sh / GOALGEN_PLAN.md 同步），让模型必须从噪声生成 subgoal。
  image-to-image ablation 时显式 `--z0-prior-alpha 1.0`（此时推理 `z_init` 必须用同样
  混合方式构造，分布才一致）。**v1 和 v2 共享同一份代码，默认值改动同时覆盖两者**。

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

统一文档入口：

- `qwen3vl_local/sft/SFT_PLAN.md`
- `qwen3vl_local/sft/SFT_RUN.md`

SFT v1：

- `qwen3vl_local/sft/build_sft_dataset_v1.py`
- `qwen3vl_local/sft/sft_v1_train.sh`
- `qwen3vl_local/sft/eval_sft_v1.py`
- `qwen3vl_local/sft/probe_sft_v1.py`

v1 assistant 使用固定 `ANALYSIS: Observations recorded.`，主要训练
STATUS/SUBGOAL 事件名。

SFT v2：

- `qwen3vl_local/sft/build_sft_dataset_v1.py --mode v2` 生成 `v2_pending`。
- `qwen3vl_local/sft/build_sft_dataset_v2_teacher.py` 用 frozen Qwen + PRIVILEGED prompt 物化真实 ANALYSIS。
- `qwen3vl_local/sft/sft_v2_train.sh` 首次训练启动时物化 base 层
  `runtime_teacher_data/`，后续按 manifest 复用；`check` 模式默认写独立
  `runtime_teacher_check_data/`，不污染正式 cache。

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
各自 `nvidia-smi` 导致 GPU 子集 race。

## 10. GPU 选址统一规则

适用：SFT v1/v2、GoalGen、LeadMoT、VAE patch/unpatch、白名单 runner 的训练、
eval、probe、teacher / 推理入口。

- 默认调用 `nvidia-smi` 自动挑空闲 GPU，并覆盖外层残留的 `CUDA_VISIBLE_DEVICES`。
- 单进程默认挑 1 张；进程内通常用 `cuda:0`。
- `torchrun --nproc_per_node=N` 默认挑 N 张，并按 `LOCAL_RANK` pin。
- `DDP_GPU_COUNT=N` / `NPROC_PER_NODE=N` 只表示需要 N 张卡，具体卡号仍自动挑。
- 训练 launcher 的 DDP rendezvous 端口默认自动选择空闲 `MASTER_PORT`，并同步导出 PyTorch launcher 会读取的 `PET_MASTER_PORT`；已有端口残留且被占用时自动换端口。只有显式同时设置 `MASTER_PORT` 与对应 `*_RESPECT_MASTER_PORT=1` 时才严格使用指定端口。SFT `check` / `single` 也要在进入 `swift sft` 前设置端口，因为 ms-swift 即使单进程也会走 torch distributed launcher。
- 文档示例不要写 shell 手动 `CUDA_VISIBLE_DEVICES=...`。
- 显式 `--device cpu` / `--device cuda:N` 的 Python 入口通常视为用户锁设备，不覆盖。
- GoalGen eval/probe 的 `--gpu N` 只在单进程下锁进程内 GPU；默认保持 0。
- `eval_carla/run_eval.sh` 用 `--num-gpus N` / `EVAL_GPU_COUNT=N` 表示闭环评测 worker 数；
  具体 GPU id 仍按 `nvidia-smi` 空闲排序自动选，并给每张卡分配独立 CARLA 端口槽。

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

## 11.5 Batched 训练规则（2026-06 新增 → H20 吞吐 + SFT logits 峰值稳定性）

H20 96GB 单卡上早期 LeadMoT 4 卡 DDP 实测每卡仅 ~14GB 显存占用，
为提高显存利用率与吞吐，`AutoMoT/qwen3vl_local/` 下的训练入口经过多轮调优。
SFT v1/v2 在 2026-06 又追加一轮稳定性修正：显式 `--use_logits_to_keep true`，
并把 micro-batch 拆小、用 `GRAD_ACC` 补回等效 batch，避免 Qwen3-VL loss 阶段
`logits.float()` 首步额外申请数十 GB 导致 OOM。**用户不需要显式设任何 batch 相关 env**：

| 子包 | env / CLI | 当前默认（2026-06 五轮调优后） | 等效 global batch | 单卡显存预期 | LR | NUM_EPOCHS / WARMUP_RATIO / EMA_DECAY |
|---|---|---|---|---|---|---|
| SFT v1 single | `PER_DEVICE_BS` / `GRAD_ACC` / `USE_LOGITS_TO_KEEP` | 22 / 2 / true | 44 | 避开 full-logits fp32 峰值 | 1.37e-4 | 4 / 0.03 / — |
| SFT v1 ddp 8卡 | 同上 | 15 / 2 / true | 240 | 避开 full-logits fp32 峰值 | 1.37e-4 | 4 / 0.03 / — |
| SFT v2 single | 同上 | 11 / 4 / true | 44 | 避开 full-logits fp32 峰值 | 8.0e-5 | 4 / 0.03 / — |
| SFT v2 ddp 8卡 | 同上 | 10 / 3 / true | 240 | 避开 full-logits fp32 峰值 | 8.0e-5 | 4 / 0.03 / — |
| GoalGen ddp 8卡 | `MICRO_BS` / `GRAD_ACC` | 16 / 2 | 256 | ~75-88GB | 5.66e-4（v1；v2 warm start 默认 1e-4） | 8（v1）/ 2（v2 warm）  /  0.15（v1）/ 0.02（v2）  /  0.9999 |
| LeadMoT ddp 8卡 | `BATCH_SIZE` / `GRAD_ACC` | 24 / 2 | 384 | ~70-85GB | 4.9e-4 | 10 / 0.20 / 0.9999 |

注：LeadMoT decoder 本身轻（hidden=1024、12 层），且 Qwen prefill frozen/no-grad/串行；
默认 `BATCH_SIZE=24` 是更靠近 H20 80% 显存的吞吐点；若坏样本密集或 OOM，
优先回退 `BATCH_SIZE=16`。

调优规则（详见各子包 RUN.md 的"H20 batched 训练"章节）：

- **等效 global batch 翻 N×，LR 按 sqrt(N) 上调**；保持等效不变则 LR 不动。
- **等效 batch 翻 N× 时单 epoch optimizer step 同步缩 1/N**：必须按需要上调 `NUM_EPOCHS` 弥补总 step 损失，否则训练步数严重不足；同时 `WARMUP_RATIO` 按比例上调（旧 ratio × N 大致保持 warmup 绝对 step 数不变），避免 LR 上调后 warmup 期 overshoot。LeadMoT/GoalGen 还应同步把 `EMA_DECAY` 从 0.999 升到 0.9999，否则平均窗与 step 数一起被砍。
- SFT OOM 时先确认日志里 `USE_LOGITS_TO_KEEP=true`；若该参数在某个 ms-swift/Transformers 组合报错，
  可临时设 `USE_LOGITS_TO_KEEP=false`，但必须继续降低 `PER_DEVICE_BS`。
- SFT / GoalGen / LeadMoT launcher 默认设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，
  用于缓解长跑中的 allocator fragmentation；VAE patch/unpatch 的 Python 入口会记录该 env。
- OOM 时先 ÷2 per-device batch、×2 grad_accum 保持等效 batch 不变，再视情况按 sqrt 法则反向降 LR。
- GoalGen `MICRO_BS=1` / LeadMoT `BATCH_SIZE=1` 时与 batched 之前的 per-sample 路径**字节级等价**（pad/prefix 工具退化为单条 stack，训练循环检测批内等长后传 `mask=None`，走无 mask SDPA），可作历史 ckpt 复现路径。
- LeadMoT eval/probe 不动，仍走 `runtime.forward_sample`；只有训练循环 BATCH_SIZE>1 时切到 `runtime.forward_batch`。
- LeadMoT batched 路径下"坏样本"会污染整个 mini-batch（整批 fallback 占位 loss）；默认 `BATCH_SIZE=24` 更吃显存，脏样本密集或 OOM 时优先回退 `BATCH_SIZE=16`。
- GoalGen CFG drop 粒度从 per-sample 升到 per-micro-batch（同一 mb 内 B 条共享 force_uncond）；`MICRO_BS=16` 是当前默认折中点，异常时先回退 `MICRO_BS=12 / LR=4.9e-4`，再回退 `MICRO_BS=8 / LR=4e-4`。
- SFT check 三段 `PER_DEVICE_BS=1 / GRAD_ACC=1` 不动以保留 2 步快速 loss_scale sanity，不参与吞吐优化。
- SDPA 在 bool attn_mask 时不走 flash backend（走 efficient/math），有 30-50% 性能损失；这是正确性必需，无法两全。
- `eval_carla/run_eval.sh` 不存在 batch 概念（per-route worker 独占 CARLA），不在 batched 训练规则范围内。

实现锚点：
- GoalGen: [dit.py](AutoMoT/qwen3vl_local/goalgen/dit.py) JointAttention `lang_key_padding_mask`；[qwen_kv.py](AutoMoT/qwen3vl_local/goalgen/qwen_kv.py) `pad_pooled_kv_batch`。
- LeadMoT: [mot_block.py](AutoMoT/qwen3vl_local/leadmot/mot_block.py) PrefixKVAttention `prefix_key_padding_mask`；[train.py](AutoMoT/qwen3vl_local/leadmot/train.py) `_pad_segmented_kv_batch` + `LeadMoTTrainRuntime.forward_batch`。

## 12. 不要做

- 不要改 `lead/`。
- 不要把 `0026.json` 或 `keyframes_all_scenarios.json` 入库。
- 不要运行 CARLA、`AutoMoT/test.sh`、`start_carla.sh`、大规模下载或安装命令。
- 不要把 AutoMoT legacy slow/fast 接口重新接回本地 Qwen/LeadMoT 路线。
- 不要把当前共享 GoalGen 架构描述成某个 dataset mode 专属架构。

## 13. 快速导航

| 任务 | 文档 |
|---|---|
| SFT v1/v2 跑法 | `AutoMoT/qwen3vl_local/sft/SFT_RUN.md` |
| GoalGen 跑法 | `AutoMoT/qwen3vl_local/goalgen/GOALGEN_RUN.md` |
| LeadMoT 跑法 | `AutoMoT/qwen3vl_local/leadmot/LEADMOT_RUN.md` |
| LeadMoT 架构 | `AutoMoT/qwen3vl_local/leadmot/ARCHITECTURE.md` |
| LeadMoT CARLA 闭环评测 | `AutoMoT/qwen3vl_local/eval_carla/EVAL_CARLA_RUN.md` |
| 规则入口 | `AGENTS.md` / `CLAUDE.md` |
