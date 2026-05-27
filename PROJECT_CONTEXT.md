# PROJECT_CONTEXT — `automot_lead` 工作区指南

> 目的：把 `lead` 与 `AutoMoT` 两个仓库的“怎么生成数据、怎么消费数据、怎么推理”讲清楚，
> 让接手者无需再从头扒源码就能理解“离线把 AutoMoT 喂 LEAD 数据”这件事的对齐边界。
>
> **本机环境提醒**：本机只有源码，没有 CARLA 仿真器、没有 LEAD 训练数据集（约 TB 级），
> 也没有模型权重。所有数据集断言都来自代码注释/常量，未跑过实测。
> 用户在 `mot_lead_offline_runner.py` 里写过 sample 跑通日志，能反推真实 shape，
> 关键 shape 会在文末汇总。

---

## 0. 一图看全

```
┌───────────────────────── lead 仓库（数据采集 + 训练 + 闭环评测）──────────────────────┐
│  CARLA 0.9.15 ──(expert.py 20Hz)──> 每 5 tick 落盘一帧 metas/.pkl + lidar/.laz       │
│                                     + rgb/.jpg + bev_semantic/.png + ...           │
│   .laz 第 4 维是 time stamp（不是 intensity）；含 5 个累积 sweep（lidar_pc_queue       │
│   maxlen=data_save_freq=5）；若 save_radar_pc_as_lidar=True 还会拼接 radar 检测点。   │
│  → carla_dataset.py / carla_dataset_video.py 把这些组织成训练 batch                  │
│  → tfv6/tfv7 / lead_pure_pmf_world_model 等模型训练                                   │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────── AutoMoT 仓库（在线驾驶：慢/快双路径 Qwen3VL + DP head） ────────┐
│  CARLA 0.9.15 ──(mot_b2d_agent.py 20Hz)──> tick() 组装：                            │
│      CAM_FRONT 1024×512 RGB                                                          │
│      LIDAR (CARLA 输入 N×4 含 intensity；lidar_to_ego_coordinate 后只留 N×3) → 两路： │
│         (A) lidar_bev 448×448×3 uint8 (R=密度 G=高度 B=0; B 通道最终被强制清零)        │
│         (B) bev_encoder_lidar_bev 1×1×256×256 float [0,1]                            │
│      GPS+IMU+SPEED → UKF → gps_filtered / compass_filtered                          │
│      RoutePlanner → target_point / next_target_point (ego frame)                    │
│   → 慢路径 InterleaveInferencer.kv_cache_fixed_inference                            │
│      （4 帧 RGB resize→512×256 + text prompt → Qwen3VL → KV cache）                  │
│   → 快路径 based_kv_cache_context_fast_qwen3vl_dp                                    │
│      （trans_feat=BEV encoder (1,1512,8,8) + v_target_point(1,5)                    │
│        + reasoning(8)+route(20)+waypoint(6) learnable query → heads）                │
│   → traj(1,6,2 cumsum)、route(1,20,2)、text "verb,verb,verb"                          │
│   → control_pid 把 traj 转 PID 油门/刹车，route 转 steer                              │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌── mot_lead_offline_runner.py（你的工作目录）───────────────────────────────────────┐
│  从 LEAD route 目录读取 12 帧 .pkl + .laz + .jpg →                                   │
│  伪装成 AutoMoT 的 tick_data → 调 InterleaveInferencer 复现一遍上面流程              │
│  ⚠ 已知不匹配点很多，详见 §8。当前“能跑通”不代表“推理是有意义的”。                  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 0.5 路线决策（**重要：决定了哪些差异需要修，哪些可以忽略**）

> 用户在反复讨论后确定的工程路线，所有后续修复优先级都基于此前提：

### 当前关注的路径

**慢推理（Qwen3-VL）** = 当前**唯一**实际有用的路径。

- Qwen3-VL backbone 是 **原始权重 frozen**（**未被 AutoMoT 作者 fine-tune**）
- 它是通用 vision-language tower，训练数据涵盖互联网各种 aspect ratio / 分辨率 / 视角
- 内部 vision processor 做 dynamic resolution（`smart_resize` **严格维持 aspect ratio**，详见 §5.7）
- ⇒ runner 喂 `(W=1152, H=384)` 三视角拼接图：smart_resize 后**仍是 (1152, 384) 不变**（已是 factor=32 倍数），aspect 3:1 完美保持，**图像不会被压扁/拉伸/失真**
- ⇒ **慢推理路径不需要做 RGB 切片、不需要 resize、不需要选前视**——直接喂当前的三视角拼接图就够
- ⚠ 唯一副作用：vision tokens 比 AutoMoT 在线 (512, 256) 多 **3.4×**（432 vs 128），慢推理显存/时间相应增加。详见 [§5.7](#57-qwen3-vl-image-processing-行为细节决定任意-shape-是否失真的关键)

### BEV encoder：已切换为 LEAD TransfuserBackbone（**架构调整**）

**之前**：runner 用 AutoMoT 自带 `BEVEncoderBackboneExtractor`（输出 `(1, 1512, 8, 8) bf16`），数据预处理被迫对齐 AutoMoT 训练分布（±32m 对称 / 256×256 / RGB crop / `z>0.2` 切地面）。

**现在**：runner 改用 **LEAD `TransfuserBackbone`（单帧 tfv6 框架）**，源码"抄"在 `mot_lead_offline_runner.py` 底部（`LeadTransfuserBackbone` / `LeadBEVEncoder` / `lead_rasterize_lidar` / `LeadBevConfig`），数据预处理整体回到 LEAD 训练分布：
- LiDAR 栅格：**±40m × [-32, 64]m / 4 px/m / z ∈ [-4, 10] 闭区间含地面**，输出 `(1, 1, 320, 384) float32 [0, 1]`
- RGB：三视角拼接 **(1, 3, 384, 1152) float32 [0, ~235]**，**不再 `crop_array`**
- 输出：`{bev_feature: (1, 512, 10, 12), image_feature_grid: (1, 512, 12, 36)}`
- 权重导入窗口：常量 `LEAD_BEV_CKPT_PATH` 当前指向已从 HuggingFace `ln2697/tfv6/tfv6_resnet34/model_0030_0.pth` 提取的 backbone-only ckpt（`/home/cruser1/lda/AutoMoT/checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth`）。**实测 `missing=0 / unexpected=0`**，`LeadTransfuserBackbone` state_dict 与 LEAD 训练 ckpt 100% 匹配。

> ⚠ LEAD backbone 输出 channel 数（512）与 AutoMoT 原（1512）不兼容，无法直接接 AutoMoT 自家 `bev_encoder_proj`——这是切换的代价。详见下面"快推理"。

### 快推理路径：默认禁用（**用户决定，等待 LEAD 版重设计**）

`bev_encoder_proj` + `reasoning_projector` + `route_head` + `waypoint_head` + 各种 learnable queries = AutoMoT 作者训过的部件。

- LEAD trans_feat `(1, 512, 10, 12)` 与 `bev_encoder_proj` 期望 `(1, 1512, 8, 8)` 不兼容，**shape mismatch 必然崩溃**
- runner 在 `run_step` 加了 `enable_fast_inference=False` 默认开关，**跳过 `based_kv_cache_context_fast_qwen3vl_dp` 调用**
- 想跑快推理需要重设计整个 decoder 链路（projector 维度、reasoning/route/waypoint queries、各 head），见 §12
- §8.2 表格里所有**仅影响 trans_feat / 快推理路径**的差异（相机物理位置、LiDAR sweep 数、bev_encoder RGB 视野等）**全部失效**——LEAD backbone 自己已经在 LEAD 数据上训练过，那些"差异"在新路径下不存在

### 当前 runner 对慢推理的输入状态

针对慢推理（`kv_cache_fixed_inference(rgb_pil_list + [prompt])`）的**所有输入项已经全部对齐**：

| 输入项 | 现状 | 是否对齐慢推理需求 |
|---|---|---|
| 4 帧 RGB PIL `(W=1152, H=384)` 三视角拼接 | 直接喂，不切、不 resize | ✅ Qwen3-VL 自适应消化 |
| `prompt_cleaned` 文本 | speed `:.2f`、tp/ntp `:.6f`、ego frame 米 | ✅ |
| `target_point/ntp`（未来 1.5s/3.0s 真值 → ego frame） | 距离量级 7–75 m，与 AutoMoT RoutePlanner 同分布 | ✅ |
| `theta` / `pos_global` | 弧度 + 米，与 `inverse_conversion_2d` 配对正确 | ✅ |

**runner 在慢推理路径上不需要任何额外修改**。BEV encoder 的 forward 仍会跑（拿到 `trans_feat`），但 `enable_fast_inference=False` 时不被消费。当前 `LEAD_BEV_CKPT_PATH` 已指向 LEAD tfv6_resnet34 backbone-only ckpt，backbone 走训练好的权重；即便回退到 None 走随机初始化也不影响慢推理输出。

---

## 1. 帧率与时间约定（重要！）

| 项 | lead | AutoMoT 在线 | 离线 runner |
|---|---|---|---|
| CARLA tick | 0.05 s (20 Hz) | 0.05 s (20 Hz) | 不跑 CARLA |
| 落盘 / 决策周期 | `data_save_freq=5` ⇒ **每 0.25 s 落盘 1 帧**（4 Hz） | 每 tick 决策但 RGB 历史**每 5 tick 抽一帧** ⇒ 0.25 s 间隔 | 每个 `.pkl` 已经是 0.25 s 间隔，`rgb_frame_step=1` 即 0.25 s |
| 预测视野 | `num_way_points_prediction=8`、`waypoints_spacing=5` ⇒ 8 个点 × 0.25 s = 2 s | traj 6 个点 × 0.5 s = 3 s（注意是 0.5 s 不是 0.25 s！） | 同 AutoMoT |
| 序列长度 | `sequence_length=12`（3 s 历史，4 Hz） | `rgb_history` deque maxlen = `obs_horizon*10 = 40 tick`（2 s 滚动缓冲）；采样取 `[-1, -6, -11, -16]` ⇒ 4 帧跨度 15 tick **= 0.75 s**；BUFFER_PHASE=31 tick (~1.55 s) | 显式输入 `--anchor`（默认 12），由 `rgb_frame_count=4`/`bev_frame_count=1` 反推 max_history=3，clip 实际加载 4 帧 |

**结论**：lead 的 0.25 s 帧 ≡ AutoMoT 的 5 个 0.05 s tick。
**但 AutoMoT 的 traj 是 0.5 s 间隔不是 0.25 s**，所以 traj 6 点覆盖 3 秒（对应 lead 的 12 个 sample）。

文件锚点：
- [lead/common/config_base.py:336-338](lead/lead/common/config_base.py#L336-L338) `carla_fps=20, data_save_freq=5`
- [lead/training/config_training.py:780-820](lead/lead/training/config_training.py#L780-L820) `num_way_points_prediction=8, sequence_length=12, waypoints_spacing=5`
- [AutoMoT/.../mot_b2d_agent.py:299-301](AutoMoT/leaderboard/team_code/mot_b2d_agent.py#L299-L301) `carla_fps=20, data_save_freq=5`
- [AutoMoT/.../mot_b2d_agent.py:1281](AutoMoT/leaderboard/team_code/mot_b2d_agent.py#L1281) `rgb_history_list[-1 - i*5]`

---

## 2. lead 数据集生成与目录结构

### 2.1 入口：`scripts/collect_all.sh`

- 调用 `lead/leaderboard_wrapper.py` 启 CARLA，跑 `lead/expert/expert.py` 在所有场景采集。
- 数据路径：`${DATA_ROOT}/data/<scenario>/<route_name>/`，其中 `<route_name>` 形如
  `999_Rep-1_Town06_13_route0_01_16_11_28_42`。
- 每个 route 内子目录由 `expert_data.py` 写出：
  ```
  metas/0001.pkl … 0NNN.pkl        # 主 meta，每 5 tick 一帧
  bboxes/0001.pkl …                # 当前帧场景内 actor 的 bbox + future
  rgb/0001.jpg …                   # 三视角拼接 (H=384, W=1152, leaderboard2_3cam)
  lidar/0001.laz                   # LARS 压缩点云（5 个累积 sweep + 可选 radar 检测点）
                                   # 第 4 维是 time stamp（0=最新,1,2,3,4），不是 intensity
                                   # save_radar_pc_as_lidar=True 时尾部还会拼 radar
  semantic/, depth/, bev_semantic/, bev_3rd_person/, radar/
  perturbated 版本同名 + _perturbated/
  ```
- 同一帧的 `metas/0001.pkl` 与 `lidar/0001.laz`、`rgb/0001.jpg` 文件名对齐。

### 2.2 expert 写入的 meta 关键字段

来自 [lead/expert/expert.py:3038-3178](lead/lead/expert/expert.py#L3038-L3178) 与
[lead/expert/expert_data.py:1142-1230](lead/lead/expert/expert_data.py#L1142-L1230)（offline 二次处理）。

**位姿/朝向**（**全部 CARLA 世界坐标系 + yaw 弧度**）：
- `pos_global`：ego 真实位置 `[x, y, z]`，单位米（直接来自 `actor.get_location()`）
- `noisy_pos_global`：加噪 GPS 位置 `[x, y]`
- `filtered_pos_global`：Kalman 滤波后位置 `[x, y]`
- `theta`：ego yaw 弧度。流程是 `theta = normalize_angle(compass_imu - π/2)`（[common_utils.py:520](lead/lead/common/common_utils.py#L520)
  **只是减 90°，没有取反**），再经 `np.unwrap` 累积，可超出 `[-π, π]`
- `privileged_yaw`：真实 yaw 弧度（=`np.deg2rad(transform.rotation.yaw)`）。**与 `theta` 偏置不同**——
  compass 与 CARLA transform.yaw 来自不同坐标约定，`theta` 又走过 unwrap。两者**是否同号取决于初始航向**，不是恒成立。若要拿 `privileged_yaw` 替代 `theta`，先在具体数据上对比一下，不要假设
- `ego_matrix`：4×4 ego→world 齐次变换（把世界点变到 ego frame：`T_world_to_ego = inv(ego_matrix)`）

> ⚠ 训练默认走哪个位姿？取决于两个开关（[carla_dataset.py:437-464](lead/lead/data_loader/carla_dataset.py#L437-L464)）：
> - `use_noisy_tp`（**默认 False**）控制 ego_position 与 target_points 的选源。False 时 ego_position = `pos_global`，tp 从 `next_target_points_3.25` 读（世界系真值）。
> - `use_noisy_tp=True` 时再看 `use_kalman_filter_for_gps`：True ⇒ filtered_pos_global + `next_target_points_3.25`，False ⇒ noisy_pos_global + `next_gps_target_points_3.25`。
> - sensor 扰动开关 `use_sensor_perburtation_prob=0.5` 只影响 image/lidar 是否走 perturbated 版本，不切换位姿源。
> ⇒ **本仓库默认配置下，训练样本一律用 pos_global（真值）**。离线 runner 已对齐：`_extract_pose_from_meta` 严格用 `pos_global + theta`，缺字段直接 raise，与训练默认完全一致。

**自车动力学**：
- `speed` (标量 m/s)、`accel_x/y/z`、`angular_velocity_x/y/z`（IMU）
- `target_speed`、`target_speed_limit`、`speed_limit`、`steer`、`throttle`、`brake`
- `privileged_acceleration`、`privileged_rotation_speed`（用下一帧 speed/yaw 数值差算的，仅 datagen 后处理填充）

**未来量**（在 `_offline_process_data` 中后处理一次性填入；以**当前帧 ego frame** 为参考）：
- `future_positions`：shape `(ego_num_temporal_data_points_saved+1, 3) = (61, 3)`，
  **每项对应 1 个原始 0.05 s tick → 总跨度 60 × 0.05 s = 3 s**。
- `future_yaws`：同上，相对当前帧的 Δyaw。
- `future_speeds`：同上。

> ⚠ **训练时如何采样未来点**：dataset 用 `future_waypoint_indices = [5, 10, 15, …, 40]`
>（`waypoints_spacing=5`）从这个数组里跳采，得到 `num_way_points_prediction=8` 个 0.25 s
> 间隔的点，覆盖 2 s。所以**模型监督的是 0.25 s 间隔 8 点**，不是 0.5 s 间隔 6 点。
> 但 AutoMoT 的输出是 6 点×0.5 s = 3 s，二者**不在同一个时间网格上**。

**过去量**（每帧实时写入，已是反序：最新→最旧）：
- `past_positions`、`past_filtered_state`、`past_speeds`、`past_yaws`、`privileged_past_positions`

**Target points / commands**（**世界坐标系**！需在 dataset 中转 ego）：
- 多套 `tp_pop_distance` 版本：`next_target_points_{k}`、`next_commands_{k}` 与 GPS 版 `next_gps_target_points_{k}`、`next_gps_commands_{k}`
- 训练默认 `tp_pop_distance = 3.25`，所以读 `next_target_points_3.25` 与 `next_commands_3.25`。
- 在 dataset 里：`target_point = inverse_conversion_2d(next_tp_list[1], ego_position, ego_yaw)`，
  `target_point_next = next_tp_list[2]`，`target_point_previous = next_tp_list[0]`。
- 离散指令维度 `discrete_command_dim = 6`（`carla_leaderboard_mode=True`），用 one-hot。

**Route**：
- `route`：dense route 世界坐标 `(N, ≥2)`，N≤`num_route_points_saved=50`。
- 训练取前 `num_route_points_smoothing=20` 个 → perturbate + smooth → 取前 `num_route_points_prediction=10` 作为模型 GT 路线。

**场景标签 / 危险标志**：`vehicle_hazard`、`light_hazard`、`walker_hazard`、`stop_sign_hazard`、`current_active_scenario_type`、`previous_active_scenario_type`、`town` 等。完整属性清单见 [carla_dataset.py:250-302](lead/lead/data_loader/carla_dataset.py#L250-L302)。

> ⚠ **`scenario_type` 是 dataset 派生字段，不是 meta 里直接存的**（[carla_dataset.py:362-374](lead/lead/data_loader/carla_dataset.py#L362-L374)）：
> ```python
> if current_active_scenario_type not in (None, "NA"):
>     scenario_type = current_active_scenario_type
> elif previous_active_scenario_type not in (None, "NA"):
>     scenario_type = previous_active_scenario_type
> else:
>     scenario_type = "NA"
> scenario_type_id = SCENARIO_TYPES.index(scenario_type)
> ```
> 直接 `pickle.load(*.pkl)["scenario_type"]` 会 **KeyError**——meta 里只有 `current_/previous_active_scenario_type`。
>
> `SCENARIO_TYPES` 列表见 [constants.py:334-385](lead/lead/common/constants.py#L334-L385)，总长 **50**（48 个真实场景 + `"noScenarios"` + `"NA"`），所以 `scenario_type_id ∈ [0, 49]`。常用场景如 `Accident, BlockedIntersection, DynamicObjectCrossing, ParkedObstacle, PedestrianCrossing, T_Junction, VehicleTurningRoute, ...`。

### 2.3 参考样本：`0026.json`（**只读、不参与 git**）

工作目录根有一个 `0026.json` —— **用户提供的一个 LEAD meta.pkl 转 JSON 后的标准参考样本**。

| 属性 | 值 |
|---|---|
| 用途 | 验证 meta 字段语义、推算坐标系、做 sanity check 时的参考"标尺" |
| 工程角色 | **只读资料**，**不参与 git**（在 `.gitignore` 之外，靠 CLAUDE.md §2-3 规则保护：禁止修改、禁止 `git add`） |
| 顶层 keys 数 | 350（含 29 套 `next_target_points_{k}` + 29 套 `next_gps_target_points_{k}`，`k` 是 tp_pop_distance ∈ {3.0, 3.25, …, 10.0}） |
| 涵盖字段 | §2.2 列出的所有 meta 字段，**实际数值版本** |

#### 关键字段实测值（供核对）

```json
{
    "speed": 16.69856909441179,                       // m/s, 高速直行
    "pos_global": [229.79, 151.20, 1.37],             // 世界 m
    "filtered_pos_global": [-3.61, 151.28],           // 注意只有 2 维, 且 x 大不同（GPS frame）
    "noisy_pos_global": [-3.32, 150.79],
    "theta": 1.8477753837916513,                      // 弧度 ≈ 105.87°
    "privileged_yaw": 1.8477754664913886,             // 与 theta 差 ~1e-7
    "target_speed": 17.94, "speed_limit": 25.0,
    "previous_target_points": [],                     // 空 ⇒ ego 还在第一个 tp 之前
    "next_target_points": [                           // 世界坐标 (3D)
        [233.16, 80.66, 0.58],
        [118.66, 196.73, 1.72],
        [18.66, 196.98, 0.0]
    ],
    "next_commands": [4, 4, 4]                        // 4 = LANEFOLLOW
}
```

#### 用这个样本核对过的关键推论（曾用于决定 runner 设计）

1. **`next_target_points[1]` 转 ego frame 后约 (74, 94) ≈ 120 m**，**`[2]` 约 (102, 190) ≈ 216 m**——LEAD milestone 远超 AutoMoT 训练分布（30–80 m）。这是 §8.2 ④ runner 选 future 1.5s/3.0s GT 而不是 `next_target_points_3.25` 的直接证据。
2. **`filtered_pos_global` 与 `pos_global` 数值差异巨大**（`x` 分别是 -3.61 vs 229.79）：两者**不在同一坐标系**！filtered_pos_global 是 GPS frame（以 GPS 起点为零），pos_global 是 CARLA world frame。⇒ runner 严格走 `pos_global`，**不能**和 filtered/noisy 混用。
3. **`theta` 与 `privileged_yaw` 差异 ~1e-7**：实质等价；§8.2 ⑨ runner 选 `theta` 是合理的。

#### 如何使用

- 新 AI 接手时若需要验证 meta 字段含义，**优先查 `0026.json` 而非去翻 `lead/expert/expert.py` 源码**（省 token）
- 若你需要重新生成或更新参考样本，要么覆盖本机 `0026.json` 文件（不 git），要么用户提供新的
- **永远不要 `git add 0026.json`**——它是参考资料，不是项目产出
- **永远不要修改其内容**——它是固定标尺，修改会让历史 §8.2 ④ 推论失效

---

### 2.4 配置入口与"carla_leaderboard_mode=True"下的关键参数

| 参数 | 值（leaderboard mode） | 含义 |
|---|---|---|
| `pixels_per_meter` | **4.0** | BEV / LiDAR raster 4 像素=1 米 |
| `min_x_meter, max_x_meter` | **-32, 64** | **ego 前向 -32m..+64m**（不对称！前面看 64m 后面只看 32m）。⚠ 源码里 `min_x_meter` **与 leaderboard_mode 无关**（仅 WAYMO 改为 0，其它都是 -32）；`max_x_meter` 才是 leaderboard or WAYMO 下 64，否则 32 |
| `min_y_meter, max_y_meter` | **-40, 40** | 左右 ±40m（仅 leaderboard_mode 走 ±40，否则 ±32） |
| `min_z, max_z` | -4, 4 | LiDAR z 过滤 |
| `max_height_lidar, min_height_lidar` | 10, -4 | LiDAR 全局高度过滤（栅格化前用 `min<=z<=max` 闭区间过滤） |
| `hist_max_per_pixel` | 5 | LiDAR 直方图最大值（用于归一化） |
| `training_used_lidar_steps` | 10 | 加载 `.laz` 后 `time < 10` 的点保留（**实际累积只有 5 个 sweep**，`time ∈ {0..4}`；这个阈值是上界保护，实际相当于全留） |
| `pixels_per_meter_collection` → `pixels_per_meter` | 2 → 4 | bev_semantic 落盘时 2 px/m，dataset 加载后 `np.repeat(2, axis=...)` **像素复制（无插值）**到 4 px/m |
| `num_way_points_prediction` | 8 | 路点 / 速度预测数 |
| `waypoints_spacing` | 5 | 在 future_positions 数组里跨 5 个 tick 采一个 |
| `sequence_length` | 12 | 训练 video clip 长度（3 s @ 4 Hz） |
| `num_route_points_prediction` | 10 | 路线预测点 |
| `num_route_points_smoothing` | 20 | smoothing 用的 route 长度 |
| `tp_pop_distance` | 3.25 | 选 `next_target_points_3.25` |
| `discrete_command_dim` | 6 | 高层指令 one-hot |
| `target_speed_classes` | `[0,4,8,10,13.88,16,17.77,20]` | 离散速度类（m/s） |
| `max_speed` | 25.0 | 速度归一化最大值 m/s |
| `target_points_normalization_constants` | `[[200, 50]]` | tp x_norm=200, y_norm=50 |
| `use_kalman_filter_for_gps` | True | 训练默认走 filtered_pos_global |
| `use_sensor_perburtation_prob` | 0.5 | 一半样本用扰动版本 sensor |
| `num_cameras` | 3（CARLA_LEADERBOARD2_3CAMERAS）→ 6（6CAM 配置时） | 见下 |

相机标定（leaderboard2_3cameras 下，**RGB 落盘是 3 视角横向拼接**）：

| idx | pos `[x,y,z]` | yaw | 单视角 W×H | fov |
|---|---|---|---|---|
| 1 (left)  | `[0.1, -0.35, 2.25]` | -54.5° | 384 × 384 | 60° |
| 2 (front) | `[0.35, 0.0, 2.25]`  | 0°     | 384 × 384 | 60° |
| 3 (right) | `[0.1, 0.35, 2.25]`  | +54.5° | 384 × 384 | 60° |

拼接后 `(H=384, W=1152, 3) uint8 JPEG`。LiDAR 是车顶 `[0,0,2.5]`、yaw=-90°，
`use_two_lidars=True` 还会加一个 yaw=-270° 的（数据 collection 决定）。

锚点：[lead/common/config_base.py:91-220](lead/lead/common/config_base.py#L91-L220),
[lead/training/config_training.py:80-820](lead/lead/training/config_training.py#L80-L820)。

---

## 3. lead 中 LiDAR / BEV / RGB 是怎么进入网络的

### 3.1 LiDAR 栅格化（**critical**）

[lead/data_loader/carla_dataset_utils.py:30-82](lead/lead/data_loader/carla_dataset_utils.py#L30-L82)
`rasterize_lidar(config, lidar, remove_ground_plane=False)`：

```
# 注意源码中先做了一次 splat_points(lidar) 但结果立刻被覆盖（写法有点绕但等价于）：
lidar = lidar[(min_height_lidar <= z) & (z <= max_height_lidar)]   # z 过滤 [-4, 10]，闭区间
xbins = linspace(-32, 64, (96*4)+1=385)   # 384 bins
ybins = linspace(-40, 40, (80*4)+1=321)   # 320 bins
hist  = histogramdd(lidar[:, :2], (xbins, ybins))   # shape (384, 320)，行=x 列=y
hist  = clip(hist, 0, hist_max_per_pixel=5) / 5     # → [0, 1]
return hist.T   # 转置后 shape (320, 384)：行=y（左右），列=x（前后）
```

- **输入**：`lidar[:, :3]` 是 **ego-local** 坐标，单位米，CARLA 朝向（x 前、y 右、z 上）。
  `.laz` 在采集时由 `accumulate_lidar` 在 ego frame 拼好；如果 `save_radar_pc_as_lidar=True`，文件里还含 radar 检测点（混在 LiDAR 点之间），离线 `laspy.read` 时无法区分。
- **输出**：`float32 (320, 384)`，**单通道**，归一化到 `[0, 1]`。
- 坐标轴：**row = ego y（右为正）；col = ego x（前为正）**，注意非典型朝向。
- 训练时加载单帧 `.laz` → 内部含 **5 个累积 sweep**（`lidar_pc_queue maxlen=data_save_freq=5`）+ 可选 radar；按 `time < training_used_lidar_steps=10` 过滤实际等于全留。**没有跨帧手动对齐**（采集时已对齐到当前 ego frame）。
- ⚠ **没有 `_perturbated.laz` 文件**：见 [carla_dataset.py:904](lead/lead/data_loader/carla_dataset.py#L904) 注释 *"LiDAR is always the same"*。
  perturbation 影响的是 RGB / semantic / depth / radar 是否走 `*_perturbated/` 目录的版本；
  LiDAR 永远读同一个 `.laz`，扰动是在加载时用
  [`align_lidar(pc, np.array([0, perturbation_translation, 0]), np.deg2rad(perturbation_rotation))`](lead/lead/data_loader/carla_dataset.py#L944-L949)
  对点云做**数学仿射变换**（y 平移 + yaw 旋转）模拟出来的，不是另存一份文件。

> ✅ **与 AutoMoT BEV encoder 的 `lidar_to_histogram_features` 轴序对照**：
> [bev_data_utils.py:4-34](AutoMoT/leaderboard/team_code/bev_data_utils.py) 的 `splat_points` 部分与 LEAD `rasterize_lidar` 的 `splat_points` **逐行同款**——
> 都是 `point_cloud[:, :2]`、`bins=(xbins, ybins)`、`hist[hist>max]=max`、`/=max`、最后 `.T`。
> ⇒ 两者输出**空间轴序完全一致**：`row = ego y（右为正）`，`col = ego x（前为正）`。
>
> ⚠ **但 z 过滤策略完全不同**（runner 想复用 AutoMoT 栅格时必须额外处理）：
>
> | | LEAD `rasterize_lidar` | AutoMoT `lidar_to_histogram_features` |
> |---|---|---|
> | z 上界 | `z <= max_height_lidar = 10` | `z < max_height_lidar = 100`（形同虚设） |
> | z 下界 | `min_height_lidar = -4 <= z`（**保留地面层**） | 无下界，但接着按 `lidar_split_height=0.2` 切 above/below |
> | 默认输出 | 单层（含地面） | `use_ground_plane=False` ⇒ 只 `stack([above])`，**`z <= 0.2` 全丢** |
> | 返回 shape | `(H, W) float32`（`.squeeze().astype(...)`） | `(1, H, W) float32`（`np.transpose((2,0,1))`） |
>
> ⇒ AutoMoT BEV encoder 训练时**根本看不到 z ≤ 0.2 的地面点**；LEAD 训练时是看得到的。
>
> 所以 runner 若想用 AutoMoT 风格栅格喂 BEV encoder：
> 1. **必须改 min/max 区间到 ±32**（lead 是 [-32,64]×[-40,40]，AutoMoT 是 ±32 对称），否则同一像素代表的米数差一倍；
> 2. **必须先 `z > 0.2` 切掉地面层**，否则会比训练分布多一整层地面密度；
> 3. 注意输出 shape 多了 channel 维（`(1, H, W)`），喂给 `bev_lidar_tensor.unsqueeze(0)` 时维度刚好对应 `(B=1, C=1, H, W)`。

### 3.2 BEV 占用 / 语义图

[lead/data_loader/carla_dataset_utils.py:800-…](lead/lead/data_loader/carla_dataset_utils.py#L800)
`build_bev_occupancy()`：

- 在 `1024 × 1024` (=`256 * 4`) 大栅格上绘制 bbox，4 px/m，覆盖 ±128 m。
- 像素坐标：`cx = (pos.x + 128) * 4`，`cy = (128 - (-pos.y)) * 4 = (128 + pos.y) * 4`。
  ⇒ **col = ego x（前为正）；row = ego y_carla（右为正）**（与 §3.1 LiDAR raster 同款，**非典型朝向**——和 OpenCV 习惯不同）
- 数据加载时切到 ego ±[min,max] meter，然后 `.repeat(2, axis=0/1)` 把 `pixels_per_meter_collection=2`
  的 bev_semantic 升采到 `pixels_per_meter=4`。
- 最终训练时 BEV semantic & 占用都是 **(320, 384) uint8 类别图**，类别从 `TransfuserBEVSemanticClass`/`TransfuserBEVOccupancyClass`
  来（`carla_leaderboard_mode=True` 不做 sim2real 转换）。

### 3.3 RGB

[lead/data_loader/carla_dataset.py:528-724](lead/lead/data_loader/carla_dataset.py#L528-L724)：

1. 读 JPEG bytes → `cv2.imdecode` → BGR2RGB；
2. `image_augmenter`（颜色扰动、模糊、噪声等，prob 0.2）；
3. 转 `(C, H, W)`；
4. 按 `used_cameras` 切分（如只用 1 个相机）；
5. `crop_height`（默认 0，因为 `cropped_height==height==384`）；
6. `horizontal_fov_reduction`（默认 0）。

最终输入网络 `rgb` 是 `(3, 384, 1152) uint8 → /255 float`（**视实现而异**，但**不做 ImageNet normalize**——
这部分由模型 backbone 自己负责）。

### 3.4 Boxes / Radar / Depth / Semantic

略，离线 runner 不用。

---

## 4. lead carla_dataset.py vs carla_dataset_video.py 的区别

| 维度 | `carla_dataset.py`（单帧） | `carla_dataset_video.py`（视频） |
|---|---|---|
| 一个样本 | 1 帧 | `sequence_length=12` 连续帧 |
| meta 加载 | 一个 `.pkl` | 12 个 `.pkl` 的 list |
| RGB | `(3, H, W)` | `(T, 3, H, W)` |
| LiDAR | 单帧 raster | T 帧 raster |
| 用于 | 老的 transfuser / 单帧 pretrain | tfv7 / world model / 视频后训练 |
| 数据组织 | `bucket.images[idx]` 1D | `bucket.images[idx]` 2D (`[T]` 帧路径) |
| index 字段 | `global_indices[idx]` 标量 | `global_indices[idx]` 长度 T 数组 |

逻辑都是先轻量 meta，再可选 sensor。`build_cache=True` / `build_buckets=True` 走早返回路径用于离线建索引。

锚点：
- 单帧 [lead/data_loader/carla_dataset.py](lead/lead/data_loader/carla_dataset.py)
- 视频 [lead/data_loader/carla_dataset_video.py](lead/lead/data_loader/carla_dataset_video.py)
  - 关键的 `__getitem__` 把所有量按 `[T]` 维度组织，注意 `meta_bytes` / `box_bytes` 是 bytes 数组要 `.decode("utf-8")`。

---

## 5. AutoMoT 仓库：在线测试链路

### 5.1 入口

- `AutoMoT/test.sh` → `leaderboard/scripts/run_evaluation_route.sh` → `leaderboard_evaluator_local.py`
- agent 实现：[AutoMoT/leaderboard/team_code/mot_b2d_agent.py](AutoMoT/leaderboard/team_code/mot_b2d_agent.py)
  `class MOTAgent(autonomous_agent.AutonomousAgent)` 入口 = `MOTAgent.get_entry_point() -> 'MOTAgent'`。

### 5.2 sensors()

| sensor id | type | 关键参数 |
|---|---|---|
| `CAM_FRONT` | rgb 相机 | x=-1.5, z=2.0, **1024×512, fov=110°** |
| `LIDAR`     | ray_cast lidar | x=0, z=2.5, yaw=**-90°**（与 lead lidar_pos_1 一致） |
| `IMU`       | imu | 0.05 s |
| `GPS`       | gnss | 0.01 s |
| `SPEED`     | speedometer | 20 Hz |
| `bev`（仅 Bench2Drive） | rgb 俯视 | z=50, pitch=-90°, fov=50°, 512×512（**仅可视化**） |

⚠ 与 lead 三视角拼接不一样：AutoMoT 在线只挂**单前视 1024×512 fov=110**。
模型权重就是按这种相机训练的。

### 5.3 tick()：把 CARLA 原始数据攒成 `result` dict

[mot_b2d_agent.py:561-869](AutoMoT/leaderboard/team_code/mot_b2d_agent.py#L561-L869)。重点：

1. **GPS+IMU+SPEED → UKF**（`USE_UKF=True`）→ `gps_filtered (2,), compass_filtered (标量)`。
2. **LiDAR**：
   - `lidar_to_ego_coordinate(input_data['LIDAR'])`：CARLA 给的是 (N, 4) `xyz+intensity`，
     这里**只取 `[:, :3]`**输出 N×3，套用 `lidar_rot=[0,0,-90]` 转回 x 前。
   - 与上一帧 ego-local 点云用 `algin_lidar(...)` 对齐拼接（双帧融合），得到 `lidar_combined (N, 3)`。
   - 拼接结果走**两条路径**：
     - **主模型 PIL 路径**：`generate_lidar_bev_images(lidar_combined, img_height=448, img_width=448)`
       → **`(448, 448, 3) uint8`**，3 通道 = (R=Density log/log(64), G=Height, B=Intensity)。
       ⚠ 因果链：输入 (N, 3) 进 `generate_lidar_bev_images` 后 line 113 **补一列 1（不是 0）**当伪 intensity，
       栅格化后 B 通道实际为 255（来自 intensity=1）；最后 line 124 `lidar_bev[:, :, 2] = 0.0`
       **再强制把 B 通道清零**。所以"B=0"是后处理重写，不是因为输入 intensity 是 0。
       范围 x,y ∈ [-32, 32]，z ∈ [-2.73, 1.27]，**7 px/m**（更准：64m / 448px ≈ 6.96），
       预处理 `lidar_pc[:, 0] *= -1`（把 CARLA 的"x 前"翻成"x 后"），然后 `+maxX/+maxY` 平移到非负，落像素时 `int_(x/discretization)` 取行 ⇒ **最终图像里 ego 位于中心，车头朝下、车尾朝上**（与 LEAD `rasterize_lidar` 输出 `row=y col=x` 的"车头朝右"完全不同朝向；调试画图时容易撞坑）；中心 |x|<2.4 且 |y|<1 的点被去掉（避免自车）。
       归一化到 [0, 255] uint8。
       ⚠ **这条 PIL 路径在 inferencer `__call__` 里被注释忽略**（详见 §5.5），实际只用作日志。
     - **BEV encoder 路径**：单独再调 `bev_encoder_t_u.lidar_to_ego_coordinate(config, …)` →
       手工 `_align_lidar_bev_encoder(...)` 跨帧 → `lidar_to_histogram_features(config)`
       → **`(1, 256, 256) float32 [0,1]`**（与 lead `rasterize_lidar` 几乎同款 splat+clip+/5+.T），
       `min/max_x=±32, min/max_y=±32`，4 px/m，`lidar_split_height=0.2` 但
       `use_ground_plane=False` 所以只用 above 一层。还存在 `min_z_projection=-10, max_z_projection=14`
       两个额外阈值（见 config.py:411-412），但当前路径不直接调用。
3. **RGB**：
   - `tick_data['rgb_front']` = `(H=512, W=1024, 3) uint8`（原始 CAM_FRONT）。
   - `tick_data['bev_encoder_rgb']` = `(1, 3, 384, 1024) float`，路径：JPEG re-encode（注入压缩伪影）→ `cv2.imdecode` → BGR2RGB → `crop_array(config)` 裁底/裁两侧 → CHW → float（**不 /255**，因为 backbone 内部 `normalize_imagenet` 自己做 `(x/255-mean)/std`，所以喂入要保持 [0, 255] 范围）。
4. **RoutePlanner** 在 ego frame 给出 `target_point (2,)` 和 `next_target_point (2,)`。
5. `result['next_command']` = `self.commands[-2]`（向前回退 2 个的高层指令）。

### 5.4 run_step() 流程

1. `tick(input_data)` → `tick_data`。
2. 把所有量塞历史 deque（`maxlen = obs_horizon*10 = 40`）。
3. `step < BUFFER_PHASE=31` 时强制 `VehicleControl(0, 0, 1)` 预热（UKF 收敛）。
4. `_build_obs_dict` → `rgb_pil_list`（4 帧 1024×512 PIL，时序由旧到新；间隔 5 tick = 0.25 s）
   + `lidar_pil_list`（**最新 1 帧** 448×448 PIL）。
5. `build_cleaned_prompt_and_modes(target_point_speed=cat([speed, tp, ntp]))` →
   `"Your current and next target point is (x,y), (x',y'), and your current velocity is X m/s. Predict the driving actions ( now, +1s, +2s) and plan the trajectory for the next 3 seconds."`
   + `understanding_output=False, reasoning_output=True`。
6. **BEV encoder 推理**：`bev_encoder(rgb=bev_encoder_rgb_bf16, lidar_bev=bev_encoder_lidar_bev_bf16)` → `bev_feature (1, 1512, 8, 8) bf16`。
7. **MoT inferencer 推理**（`__call__`）：
   ```python
   output = self.inferencer(
       image=rgb_pil_list,            # 4 帧前视
       front=[rgb_pil_list[-1]],      # ⚠ 实际未进 input_list（在 __call__ 中被注释）
       lidar=lidar_pil_list,          # ⚠ 同上，未进 input_list
       text=prompt_cleaned,
       understanding_output=False,
       reasoning_output=True,
       v_target_point=target_point_speed,   # (1, 5)
       trans_feat=bev_feature,              # (1, 1512, 8, 8)  ← 真正的 BEV 模态
       do_sample=False, text_temperature=0.0,
       frame_idx=self.step,
   )
   ```
8. 输出 `output = {text, traj (1,6,2) 已 cumsum, route (1,20,2), dp_vl_feature (8,2560)}`。
9. `control_pid(route_waypoints=(1,20,2), velocity, speed_waypoints=traj, target_point)`：
   - 纵向：`desired_speed = ||traj[1] - traj[0]|| * 2`（0.5 s 间隔 → ×2 转 m/s），PID 油门/刹车。
   - 横向：`route` (20×2) 经 `interpolate_waypoints` 0.1m 间距，传给 `LateralPIDController.step`。
10. 各种 force_move / parking_escape 覆盖控制；外加一层 35 km/h 硬限速。
    ⚠ 35 km/h 限速实际生效在 [run_step 末尾 line 1581-1583](AutoMoT/leaderboard/team_code/mot_b2d_agent.py#L1581-L1583)
    `if gt_velocity * 3.6 > 35: throttle=0, brake=1`；`control_pid` 内 line 1046-1048
    那处 35 km/h 早期实现是注释掉的兜底示例（**不要把这里当作生效逻辑**）。

### 5.5 慢/快路径（KV cache）细节

[AutoMoT/Automot/mot/evaluation/inference.py](AutoMoT/Automot/mot/evaluation/inference.py)：

**`InterleaveInferencer.__call__`** ([:1507](AutoMoT/Automot/mot/evaluation/inference.py#L1507))：
- 把 4 帧 RGB `resize_image(width=512, height=256)`（⚠ PIL 风格 W×H = 512×256，即图像高仅 256 行，是硬编码缩放，不管原图多大），加 text，组成 `input_list`。
- **lidar / front 参数在 `__call__` 内部被**注释掉**，不会进 input_list！**
  ⇒ 真正的 BEV 模态只能靠 `trans_feat`，PIL lidar 仅做日志/调试。
- 调用 `kv_cache_inference_slow_fast_dp(input_list, trans_feat=..., v_target_point=..., reasoning_tokens, action_tokens, frame_idx, slow_update_interval=2)`。

**慢路径 `kv_cache_fixed_inference(input_lists)`** ([:1233](AutoMoT/Automot/mot/evaluation/inference.py#L1233))：
- 走 `update_kv_cache_context_qwen3vl(user_prompt=USER_PROMPT, instruction_prompt=text, image_list, gen_context)`
- 用 Qwen3VL 模板，包含 system prompt `"You are a mature and professional driver."`。
- 调 `prepare_kv_cache` → `forward_cache_update_generation` 把整个上下文跑一遍并存进 `NaiveCache`。
- 返回 `gen_context = {kv_lens, ropes, past_key_values, packed_position_ids}`。
- 在线下默认每 `slow_update_interval=2` 帧刷一次。

**快路径 `based_kv_cache_context_fast_qwen3vl_dp(trans_feat, gen_context, reasoning_tokens=8, action_tokens=26, v_target_point)`** ([:340](AutoMoT/Automot/mot/evaluation/inference.py#L340))：
- 用 `prepare_fast_kvcache` 拼一个 101 tokens 的 packed sequence：
  ```
  [bev_token(64) | target_point(2) | velocity(1) | reasoning_query(8) | route_query(20) | waypoint_query(6)]
  ```
- BEV token 来自 `trans_feat.flatten(2).transpose(1,2)` → `bev_encoder_proj` → 1512→2560
  - **断言 `C==1512` 必须满足；`H*W==64` 的断言被注释掉了**。
  - ⚠ 实测：`prepare_fast_kvcache` ([automot.py:1973](AutoMoT/Automot/mot/modeling/automot/automot.py#L1973))
    会按 `bev_token_max_num_tokens = trans_feat.shape[-1] * trans_feat.shape[-2]` **动态分配**
    `packed_bev_token_indexes`。所以即便 `trans_feat` 是 `(1, 1512, 10, 12)` = 120 token，
    也不会 shape mismatch 崩溃，能跑通；但模型权重训练时见到的是 (8, 8) = 64 token 的 BEV 网格，
    现在 120 token 的特征对应错位，语义已偏离训练分布。
- target_point/velocity embed 来自专门的 encoder head。
- learnable query 来自 `reasoning_queries / route_queries / waypoint_queries`。
- 走 `language_model.forward_inference(..., update_past_key_values=False, is_causal=False)`，只算这 101 个新 token，复用 `past_key_values`。
- 输出：
  - `route_head(hidden[route_idx]).view(-1, 20, 2)` → route
  - `traj_head` 等 → `(1, 6, 2)`，**额外 `.cumsum(dim=1)` 把增量转累计位移**。
  - `gen_fast_reasoning_decision` 在 reasoning hidden state 上用 `lm_head` 解码出
    `"<|im_start|> verb, verb, verb<|im_end|>"`（verbs ∈ `{stop, keep, accelerate, slow, ...}`）。

### 5.6 AutoMoT 模型权重位置（仓库相对）

- `AutoMoT/checkpoints/AutoMoT/model.safetensors`（**Qwen3-VL 4B + heads + bev_encoder 全打包**）
- `AutoMoT/checkpoints/Qwen3-VL-4B/`（tokenizer/processor）
- `BEVEncoderBackboneExtractor` 从 `model.safetensors` 里取 `bev_encoder.*` 前缀的子集自行装载。

### 5.7 Qwen3-VL Image Processing 行为细节（决定"任意 shape 是否失真"的关键）

> 这一节解答："我直接把 (1152, 384) 三视角拼接图喂给 Qwen3-VL，会不会被偷偷 resize 改变 aspect 导致失真？" 答：**不会失真**，aspect 完美保持。但 vision token 数会变多。

#### 5.7.1 内部使用的 image processor

Qwen3-VL 复用 `Qwen2VLImageProcessor`（[transformers/models/qwen2_vl/image_processing_qwen2_vl.py:54-80](../../AppData/Roaming/Python/Python311/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)）。核心函数 `smart_resize`：

```python
def smart_resize(height, width, factor=28, min_pixels=56*56, max_pixels=28*28*1280):
    """
    1. Both dimensions divisible by 'factor'.
    2. Total pixels in [min_pixels, max_pixels].
    3. Aspect ratio maintained as closely as possible.   ← 关键
    """
    if max(h, w) / min(h, w) > 200:
        raise ValueError(...)
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        # 等比例缩小（保持 aspect）
        beta = sqrt(h * w / max_pixels)
        h_bar, w_bar = floor(height/beta/factor)*factor, floor(width/beta/factor)*factor
    elif h_bar * w_bar < min_pixels:
        # 等比例放大（保持 aspect）
        beta = sqrt(min_pixels / (h * w))
        h_bar, w_bar = ceil(height*beta/factor)*factor, ceil(width*beta/factor)*factor
    return h_bar, w_bar
```

#### 5.7.2 Qwen3-VL 实际参数

从 [automot.py:1700-1709](AutoMoT/Automot/mot/modeling/automot/automot.py#L1700-L1709) 注释里的 `grid_thw=[1, 16, 32]`、`pixel_values shape=(512, 1536)`、`128 vision tokens` 反推（基于 AutoMoT 在线 resize 后 (512, 256) 的实测值）：

| 参数 | 值 | 推理 |
|---|---|---|
| `patch_size` | **16** | grid_thw[1]=16 ⇒ H/patch_size=16 ⇒ patch_size = 256/16 = 16 |
| `merge_size` | **2** | tokens = 16×32 / 4 = 128 ⇒ merge_size² = 4 |
| `factor` | **32** | factor = patch_size × merge_size = 16 × 2 = 32 |
| `min_pixels` | 56×56 = 3,136 | 默认值 |
| `max_pixels` | 28×28×1280 = 1,003,520 | 默认值 |

#### 5.7.3 不同输入尺寸的实际处理结果

`vision_tokens = (h_bar / patch_size) × (w_bar / patch_size) / merge_size² = h_bar × w_bar / (16² × 2²) = h_bar × w_bar / 1024`

| 输入 PIL.size | smart_resize 后 | aspect 是否变化 | 总像素 | vision tokens 数 |
|---|---|---|---|---|
| **AutoMoT 在线** `(W=512, H=256)` | (512, 256) 不变（已是 32 倍数） | ✅ 保持 2:1 | 131,072 | **128** |
| **runner 当前** `(W=1152, H=384)` | (1152, 384) 不变（1152=36×32, 384=12×32） | ✅ 保持 3:1 | 442,368 | **432** |
| 假设原前视图 `(W=1024, H=512)` | (1024, 512) 不变 | ✅ 保持 2:1 | 524,288 | 512 |
| 极端例：`(W=2000, H=300)` | smart_resize 后约 (2016, 288) | ✅ 仍 aspect 近 7:1 | 580k | ~568 |

⇒ **结论**：runner (1152, 384) **既不会被 resize 也不会被压扁**，aspect 3:1 严格保持；图像内容不失真。

> 📊 **实测验证**（一次端到端跑通日志）：runner 喂 4 帧 PIL.size=(1152, 384) + 1 个 prompt 文本 → `kv_cache_fixed_inference` 返回 `gen_context = {kv_lens=[1840], ropes=[256], past_key_values: NaiveCache, packed_position_ids}`。**`kv_lens=1840 ≈ 1728 vision token（4×432）+ ~112 text/system token`**，与上面理论估算完全吻合 ✅。

#### 5.7.4 真正的副作用：vision token 数 / 显存 / 慢推理时间

| 维度 | runner | AutoMoT 在线 | 比例 |
|---|---|---|---|
| 每张图 vision tokens | 432 | 128 | **3.4×** |
| 4 帧总 vision tokens | 1728 | 512 | 3.4× |
| Attention KV cache 长度（含 text） | ~1800+ | ~700+ | ~2.6× |
| 慢推理时间（attention O(N²)） | baseline × ~7 | baseline | 显著变慢 |
| 显存（KV cache 与 attention map） | baseline × 3.4 | baseline | 3.4× |

> ⚠ 如果显存不够或速度受不了，可在 runner `_prepare_inference_inputs` 里加一行：
> `rgb_pil_list = [img.resize((512, 256), Image.LANCZOS) for img in rgb_pil_list]`
> 这会让 vision tokens 降到 128/帧，与 AutoMoT 在线对齐。**但用户当前不希望做此处理**（保留原始视野信息，让 Qwen3-VL 自己消化）。

#### 5.7.5 为什么这不影响"Qwen3-VL 慢推理质量"

1. Qwen3-VL backbone **冻结**，权重来自 Qwen 团队原始预训练（互联网通用图像-文本对，**见过各种 aspect / 各种 vision token 数**）
2. AutoMoT 训练只 fine-tune 下游 projector/heads，**不 fine-tune Qwen3-VL backbone**
3. 你预计放弃快推理 → 下游 projector/heads 不再用 → vision token 数 432 vs 128 的差异只影响**已搁置的路径**
4. 你保留的"慢推理"只是用 Qwen3-VL 算 KV cache（即 attention 中间状态）—— Qwen3-VL 通用 backbone 对此完全胜任

⇒ **runner 当前输入对慢推理无任何"隐藏失真"问题。**

---

## 6. AutoMoT BEV encoder（trans_feat 来源）的输入约束

> ⚠ **历史参考**：runner 已切换到 LEAD `TransfuserBackbone`（详见 §0.5），本节描述
> 的 AutoMoT BEV encoder **不再被 runner 使用**。保留是为了：(1) 与在线 agent 对照，
> (2) 将来若要切回 AutoMoT 自家 BEV encoder 时有 baseline 资料。
>
> runner 当前 BEV encoder 接口在 [`mot_lead_offline_runner.py`](AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py)
> 底部的 `LeadBEVEncoder` / `LeadTransfuserBackbone` 类（输出 `(1, 512, 10, 12)`，
> 数据预处理回到 LEAD 训练分布）。

[AutoMoT/Automot/mot/modeling/bev_encoder/](AutoMoT/Automot/mot/modeling/bev_encoder/)。

**config（默认）**：
- `seq_len = 1, img_seq_len = 1, lidar_seq_len = 1` — **训练时单帧输入**，不堆叠时序。
  ⇒ 在线 agent 与 runner 都只喂当前一帧的 RGB / LiDAR BEV 给 BEV encoder，不要误以为它消化多帧栈。
  时序信息只在主模型那边通过 4 帧 RGB PIL 给 Qwen3VL 慢路径处理。
- `lidar_resolution_width = lidar_resolution_height = 256`，**强 reshape 期望**。
- `pixels_per_meter = 4`，`min_x=max_x=±32, min_y=max_y=±32, min_z=-4, max_z=4`（后两者是 voxelization 参考，不直接用在 `lidar_to_histogram_features`）。
- `max_height_lidar = 100.0`（[config.py:763](AutoMoT/Automot/mot/modeling/bev_encoder/config.py#L763)），**实质无上界**；真正起 z 过滤作用的是 `lidar_split_height = 0.2`。
- `lidar_split_height = 0.2`，`use_ground_plane = False`。⇒ `splat_points` 之前 `above = lidar[z > 0.2]` 切片，只对 above 那层栅格化；**`z ≤ 0.2` 的地面点全部丢弃**，与 LEAD `rasterize_lidar` 保留 `[-4, 10]` 闭区间完全不同（详见 §3.1 ⚠ 表格）。
- `hist_max_per_pixel = 5` ⇒ 直方密度归一到 [0,1]，输出 `(1, 256, 256) float32`（注意比 LEAD 多一个 channel 维）。
- `min_z_projection = -10, max_z_projection = 14`：⚠ 这两个**不参与 `lidar_to_histogram_features`**，
  只在 [`bev_encoder_utils.py:595-613`](AutoMoT/Automot/mot/modeling/bev_encoder/bev_encoder_utils.py#L595-L613)
  的 `create_projection_grid` 里用——构造相机投影用的 voxel grid 高度范围。runner 做点云预处理时**不要参考它们**。
- `cropped_height = 384, cropped_width = 1024`（RGB 用 `crop_array` 自动裁）。
- `normalize_imagenet = True` ⇒ backbone 内部期望输入 `[0, 255]` 范围 float，自己除 255 + (x-mean)/std。
- `transformer_decoder_join = True`、`detect_boxes = True`、`use_bev_semantic = True`、`use_depth = True`、`use_semantic = True`。

**forward 输出**：
- `bev_feature: (1, 1512, 8, 8) bf16` ← `trans_feat`，flatten 后 64 个 token，喂给慢/快路径。
- `bev_feature_upscale: (1, 64, 64, 64)`、`fused_features: (1, 1512, 8, 8)`、`image_feature_grid: (1, 1512, 12, 32)`（**这些在 inferencer 里没用到**）。

---

## 7. lead vs AutoMoT 模态输入对照表（**最容易翻车的部分**）

| 模态 | lead 训练 | AutoMoT 主模型（trans_feat 之外） | AutoMoT BEV encoder（trans_feat 输入） |
|---|---|---|---|
| **LiDAR 范围** | x ∈ **[-32, 64]**, y ∈ **[-40, 40]**，z 过滤 [-4, 10] **闭区间**（含地面） | x ∈ [-32, 32], y ∈ [-32, 32], z ∈ [-2.73, 1.27]，**自车 \|x\|<2.4 且 \|y\|<1 中心去除** | x ∈ [-32, 32], y ∈ [-32, 32]。**有效 z 过滤 = `z > 0.2` (above-only)**，上界 `max_height_lidar=100` 形同虚设；**地面层 z ≤ 0.2 被丢弃** |
| **LiDAR 分辨率** | **4 px/m**，**`(320, 384) 单通道 float [0,1]`**，行=y(右正) 列=x(前正) | **~6.96 px/m**（64m/448px），**`(448, 448, 3) uint8 [0,255]`**（B 通道最终被置零） | **4 px/m**，**`(1, 1, 256, 256) float [0,1]`** |
| **LiDAR 通道含义** | 单通道：直方密度 / hist_max_per_pixel=5 | R=log(c+1)/log(64) 密度，G=高度归一，B=0（先把输入补的伪 intensity=1 喂入 → 再 `lidar_bev[:, :, 2]=0` 强制清零） | 单通道：直方密度 / 5（与 lead 同款，但范围不同） |
| **LiDAR 坐标轴** | ego-local，CARLA 朝向（x 前 y 右 z 上）；splat 后 `.T` 转置 ⇒ row=y col=x | **`pts[:, 0] *= -1` 先翻转 x**（伪装"x 右" image），然后 `+maxX/+maxY` 平移到非负 | ego-local，无翻转，与 lead 一致 |
| **LiDAR 是否跨帧累积** | `.laz` 内部 **5 个 sweep**（lidar_pc_queue maxlen=data_save_freq=5）+ 可选 radar；加载时不再跨帧手动对齐 | 在线手工双帧融合（current + algin_lidar(last)） | 同主模型路径，手工双帧融合 |
| **RGB** | 3 视角拼接 **PIL.size=(W=1152, H=384), uint8 JPEG**；多相机模式可切片 | 单前视 PIL.size=(W=1024, H=512) uint8 | 单前视裁切到 `(H=384, W=1024, 3)`，**直接给 float（保持 [0,255] 范围）** |
| **RGB normalize** | dataset 层不做 ImageNet normalize（含 `image_augmenter` 颜色扰动 prob=0.2）；**是否 normalize 由具体模型 class 决定**（tfv6/tfv7 各自实现，本文未逐个验证） | inferencer 里强制 `resize_image(W=512, H=256)`；bev_encoder 路径走 `normalize_imagenet`（内部 /255+mean/std） | `normalize_imagenet=True`，输入要保持 [0,255] |
| **target_point 坐标** | 训练默认走 `next_target_points_3.25`（世界坐标） → `inverse_conversion_2d(pos_global, theta)` 转 ego；可选归一化 `target_points_normalization_constants=[200,50]` 仅在 head 输入用 | ego-local 米（route planner 输出），不归一化 | 同 AutoMoT，不归一化 |
| **future 时间网格** | 训练监督是 `future_positions[[5,10,…,40]]` ⇒ 0.25 s × 8 = 2 s | **0.5 s × 6 = 3 s**（traj 输出） | 不直接用 |
| **route 长度** | 10 个点（smoothing 后） | 20 个点（route head 输出） | 不直接用 |
| **discrete command** | one-hot 6 维输入到 dataset（`data["command"]`） | inferencer 不输入 command 字符串（lidar/front/command 都被注释）；但 agent 内部仍维护 `self.commands = deque(maxlen=2)` 用于 force_move 等控制逻辑 | 不用 |
| **velocity 归一化** | 训练默认有 `max_speed=25` 用于离散化等场景，但 v_target_point 输入 `velocity_encoder(nn.Linear(1, 256, ...))` 不显式 /max_speed | 直接传 raw m/s | 直接传 raw m/s（与在线一致） |

**关键结论**：
- lead 训练数据**前后非对称**（前 64m / 后 32m），AutoMoT 训练数据**对称 ±32m**。**runner 当前用 LEAD backbone**（详见 §0.5），栅格按 LEAD 风格 ±40m × [-32, 64]m / 4 px/m / z ∈ [-4, 10] 闭区间含地面，输出 `(1, 320, 384)`，调用本文件 `lead_rasterize_lidar`。原 `bev_data_utils.lidar_to_histogram_features` 不再使用。
- lead 三视角拼接 RGB `(384, 1152)` 在 LEAD 训练时是直喂的（backbone 内部对 1152 宽自适应），runner 现在也**不再 crop**，三视角原样喂进 `LeadBEVEncoder`。原"crop 到 1024 仍含侧视"的问题不再存在。
- traj 时间网格不匹配是结构性问题：lead 学的是 4 Hz×2 s，AutoMoT 输出 2 Hz×3 s。LEAD 版快推理重训时可重新对齐。
- LiDAR 来源差异：lead `.laz` 含 5 个 ego-aligned sweep（**且可能混入 radar 检测点**），AutoMoT 在线只有当前+上一帧 N×3。runner 当前喂 LEAD backbone 的就是 LEAD `.laz` 同款分布，无差异；表中 AutoMoT 列保留作历史参考。

---

## 8. `mot_lead_offline_runner.py` 当前实现 — **批判性审查**

### 8.1 是什么

`AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py` 这个脚本：

1. 用户显式输入 `--anchor`（route 内绝对帧索引，默认 12）；由 `rgb_frame_count`/`bev_frame_count`/`step` 反推 `max_history`，决定要从 route 加载 `[anchor-max_history, anchor]` 这段帧。anchor 太靠前时打印 warning 并重复 frame 0 补 0，不报错。
2. 组成 `lead_clip = {rgb, lidar_points, pos_global, theta, speed, target_point, target_point_next}`（只存原始点云，不再缓存栅格化结果）。
3. `LeadOfflineMoTRunner.run_clip()` 用 clip 内最后一帧（=输入的 anchor）作 anchor_t，构造 **4 帧 RGB**（`[t-3, t-2, t-1, t]`，间隔 1 帧 = 0.25 s）+ **1 帧 LiDAR**（默认 `bev_frame_count=1`，对齐 LEAD 单帧 .laz 含 5 累积 sweep 的训练分布）。
4. LiDAR：在 `_prepare_inference_inputs` 内做"跨帧对齐到 anchor ego-local（用 `R(src_theta).T`，比在线 agent 的 `R(anchor_theta).T` 更严谨）→ `lead_rasterize_lidar(self.bev_encoder_config)` **LEAD 风格** 栅格化 → `(1, 320, 384)` 单通道"。
5. BEV encoder：runner 底部 `LeadBEVEncoder` 包装 LEAD `TransfuserBackbone`（tfv6 单帧），输出 `bev_feature: (1, 512, 10, 12)`。
6. 调用 `kv_cache_fixed_inference`（慢路径 Qwen3-VL）；快推理默认禁用（`enable_fast_inference=False`），不调 `based_kv_cache_context_fast_qwen3vl_dp`。

### 8.2 当前"跑通"不等于"对"——已识别的不匹配点

> ⚠ **本表大改前提**：BEV encoder 已切换为 LEAD TransfuserBackbone（详见 §0.5），数据预处理回到 LEAD 训练分布，原来"对齐 AutoMoT 训练分布"的努力**全部失效**：
> - ①②⑥（LiDAR 栅格 / z 过滤 / PIL）→ **🔄 已重写为 LEAD 风格栅格**，不再走 AutoMoT 风格
> - ③（RGB 视野不对齐）→ **🔄 已失效**：LEAD backbone 训练时就是三视角拼接，无需裁切
> - ⑤（LiDAR 多帧）→ 仍保留，但语义跟随 LEAD（单帧 .laz 含 5 sweep）
> - ⑦⑧⑩⑬⑭ 不受影响（与 BEV encoder 选择无关）
>
> 下面表格保留作历史记录，每一项的"现状"列描述的是**切换前**的实现，与 runner 当前代码已经不一致。runner 当前实际状态详见 §8.3 / §11 cheat sheet。

> **阅读提示**：以下表格中标 ⚪ 的项**仅影响快推理路径（trans_feat / bev_encoder / 下游 AutoMoT-trained head）**。按 §0.5 路线决策，快推理预计放弃，**这些项暂不优化**。标 ✅ 的是已修；其它项见对应注释。


| # | 问题点 | 现状 | 影响 |
|---|---|---|---|
| ① | ✅ **已修复**：BEV encoder lidar 输入尺寸 | 改用 `bev_data_utils.lidar_to_histogram_features(self.bev_encoder_config)`，直接按 AutoMoT config 出 `(1, 256, 256)`（±32m / 4 px/m / `z>0.2` 切地面）。`_rasterize_lidar_xy` 函数和 clip 的 `rasterized_lidar` 字段已删除 | trans_feat 回到训练分布的 `(1, 1512, 8, 8)`，BEV token 数恢复 64 |
| ② | ✅ **lidar PIL 改用 AutoMoT 栅格** | runner 现在的 `lidar_pil_list` 直接来自 `lidar_to_histogram_features` 输出 (1, 256, 256) [0, 1]，再 *255 转 uint8 复制 3 次成 RGB（R=G=B）。PIL.size=(W=256, H=256) | 仅日志用（PIL lidar 在 inferencer `__call__` 里被注释忽略，不进推理），但日志也已和真正喂模型的 `bev_lidar_tensor` 同源，调试体验对齐 |
| ③ | ⚪ **三视角拼接 RGB 直接喂模型**（**仅快推理相关，已搁置**） | LEAD `rgb` 是 `(384, 1152, 3)`，runner 不挑前视直接 PIL；`bev_encoder_rgb` 走 `crop_array` 裁到 `(384, 1024, 3)` | 慢路径 Qwen3-VL frozen 通用 backbone 能消化任意 aspect/分辨率（按 §0.5 路线决策不做处理）；bev_encoder 路径 crop 后保留中间 1024 列**仍含大量侧视像素**，且相机物理位置差 1.85 m（LEAD 前视 x=+0.35 vs AutoMoT x=-1.50），近场视差和"是否含车头"无法靠图像处理消除——但属于快推理路径，预计放弃 |
| ④ | ✅ **合理对齐 AutoMoT 训练分布**：target_point/ntp 用未来 1.5 s/3.0 s 真值 | `_extract_tp_ntp_from_future_frames`：取未来 `tp_lookahead_s=1.5 s` 与 `ntp_lookahead_s=3.0 s` 的 ego 真值位置，用 `inverse_conversion_2d(future_pos_global, cur_pos_global, cur_theta)` 转 ego frame | **AutoMoT 模型 ≠ LEAD 训练**：模型权重来自 `AutoMoT/checkpoints/AutoMoT`，用 AutoMoT 自家 RoutePlanner（`min_distance=7.5, max_distance=50`）训练，**期望 tp/ntp 在 30–80 m 区间**（用户在线实测 `TP≈(30, 0)`, `NTP≈(82, 2)`）。直接用 LEAD `next_target_points_3.25[1]/[2]` 转 ego frame 后实测**距离 100–200 m**（见 0026.json：`target_point` 转 ego = `(74, 94)` ≈120 m，`next` = `(102, 190)` ≈216 m），与 AutoMoT 训练分布严重错位。用 future 1.5 s/3.0 s 真值 ≈ `speed × lookahead` = 25–50 m（中速时）、~7–15 m（低速时）、~38–75 m（高速时），数量级落在 AutoMoT 训练分布内。⚠ caveat：距离随速度变化，红灯停车时退化到 ~(0, 0)，但用户接受此为预期行为 |
| ⑤ | ✅ **已修复**：LiDAR 多帧对齐 | `bev_frame_count` 默认改为 **1**——仅用 anchor 单帧 .laz（内含 5 个 ego-aligned sweep），完全对齐 LEAD 训练分布（5 sweep / 0.25 s）。`_align_lidar_points_to_anchor` 函数保留（设 `bev_frame_count>1` 时仍会工作，用 `R(src_theta).T` 严格平移），但默认路径不再触发跨帧拼接 | 单帧路径下完全无 sweep 累积偏差 |
| ⑥ | ✅ **已修复**：LiDAR z 过滤直接走 AutoMoT 风格 | runner 不再有自定义 z 过滤；`lidar_to_histogram_features` 内部按 `lidar_split_height=0.2` 切 above（`use_ground_plane=False`），与 BEV encoder 训练分布完全一致。LEAD `rasterize_lidar` 的 `[-4, 10]` 闭区间路径已废弃（连同 `_rasterize_lidar_xy` 函数一起删除） | 完全对齐训练分布；地面层 z ≤ 0.2 被丢弃，与训练一致 |
| ⑦ | **traj 时间网格不匹配** | runner 拿到 6×0.5 s = 3 s 输出，但 LEAD GT 是 8×0.25 s = 2 s | 如果用 LEAD GT 做 evaluation，需要把 traj 重新插值/采样到 0.25 s 网格，runner 当前没有这一步 |
| ⑧ | **command 队列状态机离线缺失** | runner 完全不维护 `self.commands = deque(maxlen=2)`；在线 inferencer 的 prompt 里确实不含 command，但 agent 内部 `commands` 队列参与 force_move / 路径切换 / `next_command` 计算 | 当前 runner 走"开环单点推理"，没用控制层；若未来要做 close-loop 评测，必须把 commands 队列搬过来 |
| ⑨ | ✅ **theta 单位一致** | LEAD `theta` = `preprocess_compass(IMU)` + `np.unwrap` 累积（弧度，可超 `[-π, π]`）；AutoMoT 在线 `compass_filtered` = `preprocess_compass(IMU)` + `UKF.normalize_angle`（弧度，∈`[-π, π]`）。两者来自**同一公式**（lead.common_utils.inverse_conversion_2d ≡ automot_utils.inverse_conversion_2d），仅 unwrap 边界处理不同 | `inverse_conversion_2d` 内部只用 `cos/sin`，**周期性下 unwrap 与否无影响** ✓。实测 0026.json `theta=1.8477753838` vs `privileged_yaw=1.8477754665` 差异 ~1e-7，runner 用 theta 完全正确 |
| ⑩ | ✅ **已修复**：位姿源 | `_extract_pose_from_meta` 已简化为**严格用 `pos_global` + `theta`**，不再回退到 filtered/noisy/privileged_yaw；缺字段直接 raise。与 LEAD 训练默认（`use_noisy_tp=False`）完全一致 | 完全对齐训练分布 |
| ⑪ | **clip_len 与 BUFFER_PHASE** | 离线 12 帧（0.25 s 间隔）= 3 s 历史 vs 在线 31 tick (~1.55 s) buffer 然后再决策 | 时间长度比在线还多，不算问题 |
| ⑫ | **gen_context 复用语义不同** | runner 每个 anchor 重新 `kv_cache_fixed_inference`（gen_context=None 时）；在线版 `kv_cache_inference_slow_fast_dp` 自己按 `slow_update_interval=2` 帧管理刷新 | 当前 `run_clip` 只用最后一帧 anchor 问题不暴露。**未来若多 anchor 复用，应直接调 `kv_cache_inference_slow_fast_dp` 而不是自己手动凑 gen_context**，否则会跳过 inferencer 内置的 interval 控制 |
| ⑬ | ✅ **已修复**：二次 JPEG 压缩 | 删除 `cv2.imencode + imdecode + BGR2RGB` 三步，直接 `np.array(rgb_pil_list[-1])` 取 PIL 解码结果。LEAD .jpg 本身已是 1 次 JPEG，与训练数据分布一致 | 完全对齐训练分布 |
| ⑭ | **LiDAR / Radar 混合点** | runner 用 `laspy.read(*.laz)` 拿到的可能是 LiDAR + Radar 混合点（取决于采集时 `save_radar_pc_as_lidar` 设置）；runner 不区分，全部喂栅格 | 与训练分布一致（训练时 dataset 也用 `laspy.read` 不区分），所以**不是 bug**；但要注意如果训练时 `duplicate_radar_near_ego=True`，ego 附近密度会被人为增厚 |

### 8.3 数据流形/数值范围（runner 当前实际状态）

`build_clip_from_real_lead_route` 输出（默认 anchor=12, max_history=3 → clip_len=4）：

```
rgb               : (4, 384, 1152, 3) uint8     [0, 255]
lidar_points      : list[4] 变长 float32 范围~[-93, 124]     # 每帧 ego-local 点云
pos_global        : (4, 2) float32                            # 世界 m（严格 pos_global，无回退）
theta             : (4,)   float32 ~1.59 rad
speed             : (4,)   float32 raw m/s
target_point      : (4, 2) float32                            # ego 前向米（未来 1.5s 真值）
target_point_next : (4, 2) float32                            # ego 前向米（未来 3.0s 真值）
```

`_prepare_inference_inputs` 输出（**LEAD 风格**，与 LEAD TransfuserBackbone 训练分布一致）：

```
rgb_pil_list      : list[4]，每张 PIL.size=(W=1152, H=384) RGB（三视角拼接，不 crop）
lidar_pil_list    : list[1]，PIL.size=(W=384, H=320) RGB（B=G=R，仅日志用）
target_point_speed: (1, 5) float32  [speed, tp.x, tp.y, ntp.x, ntp.y]
bev_rgb_tensor    : (1, 3, 384, 1152) float32 [0, ~235]              # 三视角直喂，不 crop
bev_lidar_tensor  : (1, 1, 320, 384)  float32 [0, 1]                 # LEAD 风格栅格（±40m × [-32,64]m, z∈[-4,10] 含地面）
```

→ `LeadBEVEncoder(rgb, lidar_bev)`（输出 dict）：
- `bev_feature`        : **(1, 512, 10, 12) float32** ← LEAD lidar branch（trans_feat 来源）
- `image_feature_grid` : (1, 512, 12, 36) float32 ← LEAD image branch

⚠ trans_feat shape 与 AutoMoT 原 `(1, 1512, 8, 8)` 不兼容，但慢推理路径不消费 trans_feat，因此不影响输出。快推理路径默认禁用（`enable_fast_inference=False`）。

### 8.4 当前剩余偏差（按路径分组）

**慢推理路径（用户**当前关心**的路径）**：✅ **所有偏差均不影响**
- Qwen3-VL frozen 通用 backbone，能消化 (1152, 384) 三视角拼接图
- prompt / target_point / theta / pos_global / speed 单位精度全部对齐
- BEV encoder 走 LEAD backbone（随机权重也无所谓），forward 跑通但 trans_feat 不消费
- ⇒ **runner 在慢推理路径上不需要任何额外修改**

**LEAD BEV ckpt 已下载并加载** ✅：
- 来源：HuggingFace [`ln2697/tfv6/tfv6_resnet34/model_0030_0.pth`](https://huggingface.co/ln2697/tfv6/tree/main/tfv6_resnet34)，经一次性脚本过滤 `backbone.*` 前缀（剥前缀）后保存为 `model_0030_0_backbone_only.pth`
- 路径：`/home/cruser1/lda/AutoMoT/checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth`
- 实测加载：`[LeadBEVEncoder] missing=0, unexpected=0` —— 与 `LeadTransfuserBackbone` state_dict 完全匹配
- 慢推理不消费 `bev_feature`，所以即便后续切回 None 走随机权重也不影响 KV cache 产出；但启用快推理 / 接 LEAD 自家 planning head 时这些权重才有意义

**快推理路径（默认禁用）**：⏸ 启用需重设计整个 decoder
- 见 §12 "🟡 中期可能做的"
- runner 中代码块保留（`enable_fast_inference=True` 一行即可触发），但不会跑通

**Close-loop / evaluation 才相关**：
- traj 时间网格（LEAD 版快推理重训时决定输出网格）
- `self.commands` deque 缺失（close-loop 状态机才需要）

### 8.5 精度与单位对照（speed / theta / tp / ntp）

第一次 KV 缓存调用之前所有数值输入的精度/单位/dtype 已与 AutoMoT 在线完全对齐：

| 字段 | AutoMoT 在线 | runner 当前 | 一致性 |
|---|---|---|---|
| **speed** 单位 | m/s | m/s（直接读 `meta["speed"]`） | ✅ |
| **speed** dtype | `torch.float32` tensor | `torch.float32` tensor | ✅ |
| **speed** prompt 格式 | `f"{speed:.2f} m/s"` ← `build_cleaned_prompt_and_modes` | runner 调**同一个函数** | ✅（自动 `.2f`） |
| **target_point/ntp** 坐标系 | ego frame（米，CARLA 顺时针 yaw 约定下 `inverse_conversion_2d` 输出，x=前向，y=右向） | 同款 `inverse_conversion_2d(future_pos_global, cur_pos_global, cur_theta)` | ✅ |
| **target_point/ntp** dtype | `torch.float32` (1, 5) | `torch.float32` (1, 5) | ✅ |
| **target_point/ntp** prompt 格式 | `f"({cur_x:.6f}, {cur_y:.6f})"` | runner 调同函数 | ✅（自动 `.6f`） |
| **yaw / theta** 单位 | 弧度 | 弧度 | ✅ |
| **yaw / theta** 来源 | `compass = preprocess_compass(IMU)` → UKF（输出 `normalize_angle` ∈ `[-π, π]`） | `meta["theta"] = preprocess_compass(IMU) + np.unwrap`（可超 `[-π, π]`） | ⚠ unwrap 与否；`inverse_conversion_2d` 用 `cos/sin` 周期性下**等价** ✅ |
| **inverse_conversion_2d** 公式 | `R(yaw).T @ (p - ego_pos)`，`R(yaw) = [[cos, -sin], [sin, cos]]` | **完全同款公式**（`lead.common.common_utils` ≡ `automot_utils`） | ✅ |
| **pos 源** | `gps_filtered`（UKF 滤波 GPS） | `pos_global`（真值，对齐 LEAD 训练默认 `use_noisy_tp=False`） | ⚠ 数值差极小（UKF 收敛后），且 future-cur 同字段做差，差值不变 ✅ |
| **wp_history** | agent 内 `waypoint_history` deque 维护但**不喂模型**（inferencer 不接收） | runner 也不维护 | ✅（都不用） |
| **LiDAR 点** | `(N, 3) float32` ego-local | `(N, 3) float32` ego-local（`laspy.read` 取 `las.x/y/z`） | ✅ |
| **LiDAR BEV** | `(1, 1, 256, 256)` bf16 [0, 1]（AutoMoT BEV encoder） | `(1, 1, 320, 384)` float32 [0, 1]（LEAD `lead_rasterize_lidar` 输出） | 🔄 已切到 LEAD 风格，对齐 LEAD TransfuserBackbone 训练分布 |
| **RGB tensor** | `(1, 3, 384, 1024) bf16` `[0, ~235]` | `(1, 3, 384, 1152) float32 [0, ~235]`（三视角拼接，不 crop） | 🔄 已切到 LEAD 风格 |
| **`v_target_point`** | `(1, 5) float32 = [speed, tp.x, tp.y, ntp.x, ntp.y]` | 同 | ✅ |
| **`trans_feat`** | `(1, 1512, 8, 8) bf16`（AutoMoT 训练分布） | `(1, 512, 10, 12) float32`（LEAD backbone 输出） | 🔄 与 AutoMoT 自家快推理 head 不兼容；慢推理不消费 |

**结论**：第一次调用 `kv_cache_fixed_inference` 之前的所有数值/单位/精度都与训练分布对齐。

> **针对当前路线（慢推理 Qwen3-VL 为主）**：上表所有项**全部 ✅**——Qwen3-VL frozen 对图像 shape 鲁棒，连 RGB 三视角拼接图都能消化，runner 慢推理路径**不需要任何修改**。
> **快推理 trans_feat 路径**的偏差（视野/sweep 数等）参见 §8.2 ⚪ 标记项，已搁置。

---

## 9. 修复进度（仅在 `mot_lead_offline_runner.py` 内可做的方向）

### 9.1 本轮（BEV encoder 切 LEAD）

- ✅ **BEV encoder 切换为 LEAD TransfuserBackbone（tfv6 单帧框架）**：
  - 在 runner 底部抄入 `LeadBevConfig` / `lead_rasterize_lidar` / `_LeadSelfAttention` /
    `_LeadBlock` / `_LeadGPT` / `LeadTransfuserBackbone` / `LeadBEVEncoder`
  - 源代码：`lead/lead/tfv6/transfuser_backbone.py`、`lead/lead/tfv6/transfuser_utils.py`、
    `lead/lead/data_loader/carla_dataset_utils.py`（去掉 lead.* import / 类型装饰器）
  - config 默认值取自 `lead/lead/training/config_training.py`（carla_leaderboard_mode=True）
- ✅ **数据预处理整体回到 LEAD 风格**：
  - LiDAR 栅格：`lidar_to_histogram_features(±32m / z>0.2)` → `lead_rasterize_lidar(±40m × [-32, 64]m / z∈[-4, 10] 含地面)`
  - shape：`(1, 1, 256, 256)` → `(1, 1, 320, 384)`
  - RGB：取消 `crop_array`，三视角拼接 `(1, 3, 384, 1024)` → `(1, 3, 384, 1152)`
  - dtype：bf16 → float32（LEAD backbone 期望 float32 输入）
- ✅ **快推理默认禁用**：`run_step` 新增 `enable_fast_inference=False`，跳过
  `based_kv_cache_context_fast_qwen3vl_dp` 调用，因 LEAD trans_feat shape 与 AutoMoT 自家
  `bev_encoder_proj` 不兼容
- ✅ **权重导入窗口 + 已下载 LEAD tfv6_resnet34 ckpt**：
  - 常量 `LEAD_BEV_CKPT_PATH = "/home/cruser1/lda/AutoMoT/checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth"`
  - 一次性脚本下载 HuggingFace `ln2697/tfv6/tfv6_resnet34/model_0030_0.pth` → 过滤 `backbone.*` 前缀（剥前缀）→ 另存 backbone-only → 删完整 ckpt 释放 ~276 MB
  - `LeadBEVEncoder._load_lead_weights` 走 `strict=False` 加载，**实测 `missing=0 / unexpected=0`**，state_dict 100% 匹配

### 9.2 之前轮次（保留）

- ✅ target_point/ntp 用未来 1.5 s/3.0 s 真值（速度依赖距离 7~75 m，比 LEAD `next_target_points_3.25` 100–200 m 更合理）
- ✅ `bev_frame_count` 默认 **1**，仅 anchor 单帧 .laz（含 5 sweep）
- ✅ `_extract_pose_from_meta` 严格用 `pos_global + theta`，不再回退到 filtered/noisy/privileged_yaw
- ✅ 去掉二次 JPEG re-encode（LEAD .jpg 已 1 次压缩）
- ✅ CLI 改 `--anchor` 显式输入，由采样参数反推 max_history
- ✅ 删除 `_rasterize_lidar_xy` 函数和 clip 的 `rasterized_lidar` 缓存字段

### 9.3 待办

- ✅ **LEAD BEV ckpt 已填**：HuggingFace `ln2697/tfv6/tfv6_resnet34/model_0030_0.pth` 提取 backbone-only 加载，missing=0/unexpected=0
- ⏸ **LEAD 版快推理 decoder 链路重设计**：见 §12 🟡
- ⏸ traj 时间网格（LEAD 版快推理重训时再决定）
- ⏸ `self.commands` deque（close-loop 评测才相关）

---

## 10. 文件清单（常用导航）

### lead
- 数据采集主入口：`lead/scripts/collect_all.sh`
- expert：`lead/lead/expert/expert.py`、`expert_data.py`、`expert_utils.py`、`config_expert.py`
- 数据加载：`lead/lead/data_loader/carla_dataset.py`、`carla_dataset_video.py`、`carla_dataset_utils.py`、`training_cache.py`
- 配置：`lead/lead/common/config_base.py`、`lead/lead/training/config_training.py`
- 模型：`lead/lead/tfv6/`、`lead/lead/tfv7/`
- 闭环评测：`lead/lead/inference/sensor_agent.py`、`closed_loop_inference.py`、`config_closed_loop.py`
- 评测脚本：`lead/scripts/evaluation.sh`、`eval_bench2drive.sh`

### AutoMoT
- 测试入口：`AutoMoT/test.sh` → `leaderboard/scripts/run_evaluation_route.sh`
- 在线 agent：`AutoMoT/leaderboard/team_code/mot_b2d_agent.py`
- 工具：`automot_utils.py`、`bev_data_utils.py`、`lidar_utils.py`、`ukf_utils.py`、`nav_planner.py`
- 模型：`AutoMoT/Automot/mot/modeling/automot/automot.py`
- 推理：`AutoMoT/Automot/mot/evaluation/inference.py`
- BEV encoder：`AutoMoT/Automot/mot/modeling/bev_encoder/`
- 数据预处理：`AutoMoT/Automot/preprocess/generate_lidar_bev_b2d.py`
- **离线 runner（用户主战场）**：`AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py`

---

## 11. 一次性 cheat sheet

```
【路线】慢推理 Qwen3-VL (frozen 原始权重) = 当前唯一跑通有意义的路径
       BEV encoder 已切换为 LEAD TransfuserBackbone (tfv6 单帧框架, 抄进 runner)
            ⇒ 数据预处理整体回到 LEAD 风格 (±40m × [-32,64]m / 4px/m / z∈[-4,10] 含地面)
            ⇒ trans_feat shape (1, 512, 10, 12), 与 AutoMoT 原 (1, 1512, 8, 8) 不兼容
            ⇒ 权重已加载 LEAD tfv6_resnet34 backbone-only ckpt (HuggingFace ln2697/tfv6)
              路径 /home/cruser1/lda/AutoMoT/checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth
              实测 missing=0 / unexpected=0, state_dict 100% 匹配
       快推理默认禁用 (enable_fast_inference=False), 因 trans_feat shape 不兼容下游 head
            ⇒ 想启用 LEAD 版快推理需重设计整个 decoder 链路 (proj/queries/heads), 见 §12
       runner 慢推理路径无需修改 (Qwen3-VL 通用 backbone 对图像 shape 鲁棒)

LEAD: 20Hz CARLA, 每0.25s落盘1帧, BEV 4px/m 范围 [-32,64]×[-40,40] 单通道[0,1]
       LiDAR 单帧.laz 内含 5 sweep (lidar_pc_queue maxlen=5 滚动覆盖, 相邻帧无重叠)
            + 可选 radar; 第4列是 time stamp 不是 intensity
       z 过滤 [-4,10] 闭区间; rasterize_lidar 输出 (320,384) 行=y 列=x
       LiDAR perturbation: 没有 _perturbated.laz, 加载时 align_lidar 数学变换扰动
       meta: pos_global/theta(世界); future_positions 每0.05s一项(ego frame),共61项(3s)
       训练默认: use_noisy_tp=False ⇒ 用 pos_global (真值)
       scenario_type 是 dataset 派生 (current/previous_active → fallback NA)
            SCENARIO_TYPES len=50 (含 noScenarios + NA)
       target_point: 用 next_target_points_3.25, dataset 内转 ego
       预测: 8点×0.25s=2s (future_positions[5,10,...,40]), route 10点
       TransfuserBackbone (tfv6): RGB (B,3,384,1152) + lidar (B,1,320,384) → 
            (lidar_feat (B,512,10,12), image_feat (B,512,12,36))
            backbone 自带 normalize_imagenet, 输入需保持 [0, 255] 范围
            权重 state_dict 前缀 backbone.*, ckpt 文件名 model_XXXX.pth

AutoMoT 在线: 20Hz, 每tick决策 (本仓库 runner 不复用其 BEV encoder/快推理)
       RGB 历史 deque(maxlen=40 tick=2s), 采样[-1,-6,-11,-16] 跨度15tick=0.75s
       LiDAR(主可视PIL): ~6.96px/m, [-32,32]², 448×448×3 uint8 (仅日志)
       inferencer: 4 PIL RGB(慢) + trans_feat(快) → traj/route/text
       慢: kv_cache_fixed_inference (Qwen3-VL frozen, 通用 backbone, runner 在用)
       快: based_kv_cache_context_fast_qwen3vl_dp (AutoMoT-trained head, runner 默认禁用)
       (历史 AutoMoT BEV encoder 细节见 §6, runner 已不再使用)

离线 runner: 当前状态
   ✅ BEV encoder 已切换为 LEAD TransfuserBackbone (tfv6, 抄进 runner)
   ✅ LiDAR 栅格回到 LEAD 风格 (320, 384), z [-4,10] 含地面
   ✅ RGB 三视角拼接 (1, 3, 384, 1152), 不再 crop_array
   ✅ 快推理默认禁用 (enable_fast_inference=False), trans_feat shape mismatch 不会崩
   ✅ 权重导入窗口 + 已加载 LEAD tfv6_resnet34 ckpt: 实测 missing=0 / unexpected=0
   ✅ 位姿源严格 pos_global + theta, 无回退
   ✅ 去掉二次 JPEG (LEAD .jpg 已 1 次压缩, 对齐训练)
   ✅ CLI 改 --anchor 显式输入, 由采样参数反推 max_history
   ✅ tp/ntp 用未来 1.5s/3.0s 真值: 距离 ≈ speed×lookahead = 7~75m
      精度: prompt 用 :.6f, tensor float32; ego frame: x=前向, y=右向
      速度依赖是预期: 红灯停车时退化 ~(0,0), 用户接受
   ✅ 端到端跑通: 4 帧 (1152,384) RGB + LEAD 栅格 LiDAR → 慢推理 kv_lens=1840
      (1728 vision token + ~112 text/system token), gen_context 4 个 key 齐全
   ⏸ 离线没维护 self.commands deque (close-loop 评测才相关)
   ⏸ traj 时间网格 6×0.5s vs LEAD GT 8×0.25s (LEAD 版快推理重设计时再决定输出网格)
   ⏸ LEAD 版快推理整体需重设计 (新 projector 512→2560 / 新 queries / 新 heads)
```

---

## 12. 未来工作 / 路线相关待办

> 这些是**已经讨论但暂不实施**的工作项，按优先级 / 工程量列出。新对话接手时若用户提到要做，可直接查这里。

### 🟢 当前阶段无需做的（基于 §0.5 路线决策）

1. **慢推理 RGB resize / 切片** — ❌ 不做
   - 理由：Qwen3-VL frozen 通用 backbone，对任意 aspect/分辨率鲁棒；当前 (1152, 384) 三视角拼接图直接喂没问题
   - 用户明确表态："不希望要做"
   - ⇒ runner 慢推理路径**保持现状**

2. **bev_encoder 输入对齐 AutoMoT 训练分布**（切视野、resize 到 256×256、sweep 数对齐等）— ❌ 不做
   - 理由：BEV encoder 已切换为 LEAD TransfuserBackbone，数据已回到 LEAD 训练分布，
     原 AutoMoT 风格预处理彻底废弃
   - 历史讨论见 §8.2 ②③④（现已失效）

### ✅ 已完成的路线调整

3. **BEV encoder 从 AutoMoT 切到 LEAD TransfuserBackbone**（**已完成**）
   - 在 `mot_lead_offline_runner.py` 底部抄入 `LeadBevConfig` / `lead_rasterize_lidar` /
     `LeadTransfuserBackbone` / `LeadBEVEncoder`（源代码来自 `lead/lead/tfv6/` 与
     `lead/lead/data_loader/carla_dataset_utils.py`，去掉 lead-only import 与类型装饰器）
   - 数据预处理（LiDAR ±40m × [-32, 64]m / RGB 三视角 1152、不 crop）
   - 权重导入窗口：`LEAD_BEV_CKPT_PATH` 当前指向已下载并提取的 LEAD tfv6_resnet34 backbone-only ckpt（HuggingFace `ln2697/tfv6`），实测 `missing=0 / unexpected=0`
   - 快推理默认禁用（`enable_fast_inference=False`）

### 🟡 中期可能做的（用户决定要恢复快推理时）

4. **LEAD 版快推理重设计**（**未做**）
   - 范围：LEAD trans_feat `(1, 512, 10, 12)` → 新 `bev_encoder_proj`（512→2560，token 数 120）
     + 新 reasoning/route/waypoint queries（维度匹配新 hidden_size）+ 新 traj/route/text 各 head
   - 数据：LEAD 数据集（已有 `bev_semantic / detect_boxes / depth / semantic /
     future_positions / target_speed_classes / next_target_points_3.25 / 离散 command` 等所有 GT）
   - Qwen3-VL backbone 仍 frozen；只训 decoder 链路
   - 工程量大（要搭训练 pipeline + Loss + GPU 资源）
   - **训完之后 runner 完全不需要做"AutoMoT 训练分布对齐"**

5. **如果走方案 4 后，PROJECT_CONTEXT.md 进一步调整**：
   - §0.5"快推理路径：默认禁用"改成"快推理 LEAD 版已重训上线"
   - run_step 中 `enable_fast_inference=False` 默认改为 True
   - cheat sheet 路线行重写

### 🔴 远期 / 若做 close-loop 评测才相关

6. **traj 时间网格对齐**（LEAD 版快推理重训时再决定输出网格）
   - LEAD GT 是 8×0.25s = 2s，重训快推理时可直接对齐 LEAD GT 网格
   - 不再走 AutoMoT 原 6×0.5s = 3s 的时序约定

7. **离线维护 `self.commands` deque**
   - 用于 force_move / parking_escape / 路径切换状态机
   - 当前 runner 走"开环单点推理"用不到
   - 真要做 close-loop 评测时再加

### 🧰 工程清理（任何时候都可以做）

8. **进一步精简快推理相关代码**（**用户明确放弃 AutoMoT 版快推理后可做**）
   - 当前 runner 已用 `enable_fast_inference=False` 开关跳过 AutoMoT 快推理调用，
     但 `based_kv_cache_context_fast_qwen3vl_dp` 代码块、`reasoning_tokens`/`action_tokens`
     参数等仍保留（方便用户测试时一行切换）
   - 若彻底放弃 AutoMoT 版快推理（用 LEAD 版替代），可删掉相关分支与 inferencer 字段引用

---

## 13. 关键帧 / VLM 提示词外挂三件套（远程服务器侧产物，本机只读）

> 这三个文件来自远程仓库路径 `/datashare/IOL4SGH/data/data/`，本机仓库根目录下也放了一份副本：
>
> - [`rule_based_keyframe_filter.py`](rule_based_keyframe_filter.py)
> - [`vlm_prompt_pipeline.py`](vlm_prompt_pipeline.py)
> - [`keyframes_all_scenarios.json`](keyframes_all_scenarios.json)
>
> **它们不归本项目维护**——不要改、不要 `git add`、不要 push（CLAUDE.md §2 已禁止）。
> 这里只记录"它们做了什么、字段怎么读、如果要接到 `mot_lead_offline_runner.py`
> 的 `prompt_cleaned` 上需要怎么用"。

### 13.1 三者的上下游关系

```
LEAD 数据集 (cache_ln/data/<scenario>/<run>/{metas,bboxes,rgb,results.json})
   │
   │  [离线一次性运行]
   ▼
rule_based_keyframe_filter.py
   │  按 43 个 CARLA 场景各自的规则，从 metas/*.pkl 的物理量
   │  (speed / accel_x / brake / 8 个 dist_to_*) 中挑出
   │  5 个关键帧：initial → 3 个 middle 事件 → final
   ▼
keyframes_all_scenarios.json          ← 静态查表，runtime 直接读
   │
   │  [运行时 / 推理时]
   ▼
vlm_prompt_pipeline.py
   │  DrivingMemory 状态机（scenario / status / subgoal / completed）
   │  + 给 VLM 拼 system + user prompt
   │  + 解析 VLM 输出推进 status
   ▼
VLM 调用（理论上可挂到 AutoMoT 慢推理的 prompt 上）
```

注意：`vlm_prompt_pipeline.py` 是**框架无关**的——它**不直接读**
`keyframes_all_scenarios.json`，只用 `SCENARIO_EVENT_SEQUENCES` 字典。两者通过
"事件名"约定耦合：filter 写出的 `event` 字符串必须在 pipeline 的事件序列里能找到。

### 13.2 `rule_based_keyframe_filter.py` 解读

**作用**：把每个 LEAD run 摘录成 5 个关键帧，供下游做事件级标注 / 评估 / VLM
监督信号。

**核心数据结构**：

- `SCENARIO_LABELS`（[L34-77](rule_based_keyframe_filter.py#L34)）— 场景 → 一句话英文标签，
  作为 initial 帧的 `label_text` 写入 JSON
- `SCENARIO_CONFIG`（[L89-271](rule_based_keyframe_filter.py#L89)）— 每个场景的
  `(dist_meta_field, approach_threshold_m, (event_A, event_B, event_C))` 元组
  - `dist_meta_field`：从 `metas/*.pkl` 里读哪个 `dist_to_*` 字段作为"距离接近"信号
  - `approach_threshold_m`：距离阈值（米），低于它认为"进入交互段"
  - `(A, B, C)`：三个中间事件的命名（如 `("hazard_detect", "max_brake_or_min_gap", "recover_or_pass")`）
- `BRAKE_ACCEL_PRIMARY`（[L274](rule_based_keyframe_filter.py#L274)）— `{HardBreakRoute, ControlLoss}`，
  这两个场景不靠距离信号，靠 brake/accel 找事件峰值
- `_ALL_DIST_FIELDS`（[L280-289](rule_based_keyframe_filter.py#L280)）— 全部 8 个距离字段名

**输入**（每个 run）：

| 文件 | 用途 |
|---|---|
| `results.json` | 取 `meta.duration_game` 算 `seconds_per_frame`、`status`、`infractions`、`route_id` |
| `metas/*.pkl`（lzma 压缩 pickle） | 取 `speed / accel_x / brake / throttle / dist_to_*` 8 个字段 |
| `bboxes/*.pkl`（lzma 压缩 pickle） | 当 metas 里相应距离字段缺失时回退，从 bbox `class/distance/position/extent` 算最近车辆/行人/自行车距离 |
| `rgb/*.jpg` | 最后兜底——靠相邻帧文件大小差找运动峰值 |

**事件挑选流程（[`pick_middle_events` L783](rule_based_keyframe_filter.py#L783)）**：

按优先级试 4 条规则：

1. **CrossingBicycleFlow 专用** — 自行车等待峰值更细的判定（[`_pick_bicycle_flow_events` L693](rule_based_keyframe_filter.py#L693)）
2. **Cut-in / Merge 专用** — `cutin_onset → caution_peak → stabilize_follow`，
   onset 用"距离持续下降"判定（[`_pick_cutin_events` L740](rule_based_keyframe_filter.py#L740)）
3. **Brake/accel 主导**（HardBreakRoute / ControlLoss）— 找 `accel_x` 最负
   或 `brake` 最大（[`_pick_brake_accel_events` L655](rule_based_keyframe_filter.py#L655)）
4. **通用距离规则**（[`_pick_distance_events` L599](rule_based_keyframe_filter.py#L599)）：
   - A = 距离首次低于阈值且开始减速的帧（同时满足 `dist < thresh` 且 `accel < -0.4` 或 `brake > 0.05`）
   - B = 该段内速度最低的帧（即"最大减速点 / 最近接近点"）
   - C = B 之后第一个持续 ≥2 帧 `speed > 2 m/s 且 accel > 0.1` 的恢复点
5. 上述都失败 → **RGB fallback**：相邻 JPG 文件大小差找 3 个峰值帧

**信号源优先级**：metas/*.pkl（confidence≈0.88）→ bboxes/*.pkl（≈0.7）→ rgb_fallback（0.5）。
对应 JSON 输出里的 `signal_source` 字段。

**几个工程细节**：

- 全部 pickle 都是 **lzma (xz) 压缩**的，[`load_pickle` L324](rule_based_keyframe_filter.py#L324) 有兜底
- `enforce_event_order`（[L552](rule_based_keyframe_filter.py#L552)）保证 3 个中间事件严格递增
  且彼此间有最小间隔（默认 2 帧），避免几个事件落在同一帧
- 帧→时间换算 `t = frame * seconds_per_frame`，其中
  `seconds_per_frame = duration_game / (total_frames - 1)`，CARLA 数据集典型≈0.25s/帧
- 默认 dataset_root 在远程：`/home/cruser1/lda/lead/cache_ln/data`（CLI `--dataset-root` 可改）

### 13.3 `keyframes_all_scenarios.json` 字段说明

**顶层结构**：

```json
{
  "dataset_root": "/home/cruser1/lda/lead/cache_ln/data",
  "scenarios": [41 个场景名 ...],
  "num_runs": 7326,
  "runs": [ { ...单个 run 条目... }, ... ],
  "failed_runs": [],
  "num_failed_runs": 0
}
```

注意：实际 JSON 里 `scenarios` 列了 **41** 个（缺 `Accident` 之外的 `AccidentTwoWays`
变体？——核对 [`keyframes_all_scenarios.json#L3-46`](keyframes_all_scenarios.json#L3) 发现
没有 `AccidentTwoWays`），与 `SCENARIO_CONFIG` 的 43 项略有差异，可能是数据集里
有些场景没采到。`num_runs = 7326`，全部成功无失败 run。

**每个 run 条目**：

| 字段 | 类型 | 含义 |
|---|---|---|
| `scenario` | str | CARLA 场景名（与 `SCENARIO_CONFIG` 键一致） |
| `run_id` | str | run 目录名，形如 `Town03_Rep0_route_001783_route0_01_11_02_37_46` |
| `route_id` | str | 来自 `results.json`，如 `RouteScenario_route_001783_rep0` |
| `status` | str | run 结果：`Perfect / Completed / Failed / Unknown` 等（来自 `results.json`） |
| `num_infractions` | int | 该 run 的违规次数（来自 `results.json`） |
| `signal_source` | str | 关键帧从哪挑出来的：`metas / bboxes / rgb_fallback` |
| `rule_confidence` | float | 三个中间事件 confidence 的平均，0.0–1.0 |
| `initial` | obj | 首帧（见下） |
| `middle` | list[3] | 三个中间关键帧（见下） |
| `final` | obj | 末帧（见下） |
| `diagnostics` | obj | `{total_frames, duration_game, seconds_per_frame}` |

**`initial`**：

```json
{
  "event": "initial",
  "frame": 0,
  "t": 0.0,
  "label_text": "Brake and avoid accident hazard",   // ← 来自 SCENARIO_LABELS
  "confidence": 1.0
}
```

**`middle` 每个元素**：

```json
{
  "event": "hazard_detect",          // 场景特定事件名（与 vlm_prompt_pipeline 的 SCENARIO_EVENT_SEQUENCES 对应）
  "frame": 37,                       // 在该 run rgb 序列中的 0-based 帧号
  "t": 9.2974,                       // 对应的游戏时间（秒）
  "confidence": 0.88                 // 信号强度：metas≈0.88, bboxes≈0.7, rgb≈0.5
}
```

**`final`**：

```json
{
  "event": "final",
  "frame": 117,                       // = total_frames - 1
  "t": 29.4,
  "final_success": true,              // status ∈ {Perfect, Completed} 且无 timeout/blocked
  "confidence": 1.0                   // 成功 1.0 / 失败 0.8
}
```

**重要：`frame` 是 LEAD 原始 rgb 序列的下标**，**不是** AutoMoT 慢推理那个 4 帧
clip 的下标。要在 runner 里用它，需要换算到当前 `anchor_t` 对应的原始帧号
（runner 里 `lead_clip` 已经按某种 frame_step 抽帧，参见 PROJECT_CONTEXT.md §5）。

### 13.4 `vlm_prompt_pipeline.py` 解读

**作用**：把"VLM 看一张前视图 + 一段 memory → 输出 STATUS/SUBGOAL/ANALYSIS"
封装成可复用的小模块。**不依赖具体 VLM 框架**——`run_pipeline_step` 收一个
`vlm_fn(system, user) → str` callable，让调用方接 Qwen3-VL / GPT-4V / 任意 VLM。

**核心组件**：

- **`SCENARIO_LABELS`**（[L38-81](vlm_prompt_pipeline.py#L38)）— 与 filter 里那份**完全一致**
  （独立维护，没共享代码）
- **`SCENARIO_EVENT_SEQUENCES`**（[L87-130](vlm_prompt_pipeline.py#L87)）— 场景 → 3 个中间事件
  的元组。**与 filter 里 `SCENARIO_CONFIG` 第 3 项一一对应**，这是两个文件唯一
  的"事件命名契约"
- **`EVENT_DESCRIPTIONS`**（[L136-209](vlm_prompt_pipeline.py#L136)）— 每个事件名 → 一句英文
  人类可读说明，给 VLM 看 memory 时同时贴出
- **`DrivingMemory`**（[L228](vlm_prompt_pipeline.py#L228)）— dataclass，字段：
  - `scenario`、`scenario_label`：标识
  - `event_sequence`：完整序列 `("initial", mid_a, mid_b, mid_c, "final")`
  - `status`：当前确认到哪个事件
  - `subgoal`：下一个目标事件
  - `completed_events`：已走过的事件列表
- **`build_system_prompt()`**（[L348](vlm_prompt_pipeline.py#L348)）— 返回一段固定 system prompt，
  让 VLM 严格按 `ANALYSIS: ... \n STATUS: ... \n SUBGOAL: ...` 三行格式输出
- **`build_memory_block(memory)`**（[L353](vlm_prompt_pipeline.py#L353)）— 把 memory 渲染成
  `[MEMORY] ... [/MEMORY]` 文本块插入 user prompt
- **`build_user_prompt(memory, image_description)`**（[L375](vlm_prompt_pipeline.py#L375)）— 拼接
  `<image>` + memory block + 提问语
- **`parse_vlm_output(text)`**（[L410](vlm_prompt_pipeline.py#L410)）— 三条正则抽 STATUS/SUBGOAL/ANALYSIS
- **`update_memory(memory, parsed, strict=False)`**（[L434](vlm_prompt_pipeline.py#L434)）— 推进状态：
  - 只允许沿 `event_sequence` **前进或保持**，不允许回退
  - 校验 VLM 返回的事件名必须在序列内（`strict=True` 时抛错，否则忽略）
  - subgoal 由 `final_status` 推导而非完全相信 VLM
- **`run_pipeline_step(memory, image_description, vlm_fn)`**（[L516](vlm_prompt_pipeline.py#L516)）— 一步走完
  build → call → parse → update

**system prompt 关键约束**（[L317-345](vlm_prompt_pipeline.py#L317)）：

- VLM 只能输出"ANALYSIS / STATUS / SUBGOAL"三行
- STATUS / SUBGOAL 必须是事件序列里的单个 token
- SUBGOAL 必须是 STATUS 的下一个事件，除非 STATUS 已经是最后一个中间事件（此时
  SUBGOAL = `final`）

### 13.5 接到 `mot_lead_offline_runner.py` 的 `prompt_cleaned` 上的思路

当前 runner 的 prompt 在
[`mot_lead_offline_runner.py:1061`](AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py#L1061)
来自
[`automot_utils.py:1204 build_cleaned_prompt_and_modes`](AutoMoT/leaderboard/team_code/automot_utils.py#L1204)，
内容是固定模板：

```
Your current and next target point is (..., ...), (..., ...),
and your current velocity is ... m/s. Predict the driving actions ...
and plan the trajectory for the next 3 seconds.
```

如果想接入"场景描述 + memory"风格的提示词，**几种可选切入方式**（先列出，
等用户挑了再实现，不要先斩后奏）：

**方案 A：注入静态场景标签**（最轻）

- 从 LEAD sample 的 `run_id` / 目录名解析出 scenario
- 查 `SCENARIO_LABELS[scenario]` 得到一句英文场景描述
- 拼到现有 prompt 前面：`Scenario: <label>. Your current and next target ...`
- 优点：实现最简单，0 状态；缺点：不利用关键帧 / memory，VLM 不知道当前进度

**方案 B：注入"当前事件 + 下一事件"**（中等）

- 加载 `keyframes_all_scenarios.json`，按 `(scenario, run_id)` 找到该 run 的 5 个关键帧
- 把当前推理帧号映射到 LEAD 原始序列里的对应帧（注意 anchor_t / frame_step 换算）
- 找到该帧号落在哪个事件区间（initial→mid0、mid0→mid1、mid1→mid2、mid2→final）
- 用 `vlm_prompt_pipeline.DrivingMemory(status=区间起点, subgoal=区间终点)`
  调 `build_memory_block` 拼到 prompt 里
- 优点：能给 VLM 时序上下文；缺点：runner 是离线单步推理，没法跨步推进 memory，
  得每步都从 JSON 重算一次

**方案 C：在线推进 memory 状态机**（最重）

- 在 runner 里持有一个 `DrivingMemory` 实例，跨 sample 推进
- 用 VLM 输出反过来更新 memory，下一步喂带新 status 的 prompt
- 需要修改 `run_step` 把 VLM 输出截出 STATUS/SUBGOAL（当前 runner 只关心
  traj 输出，不关心文字）
- 适合做闭环评测，**不适合当前的"开环单点推理"**模式

**推荐先做方案 A**：成本极低，能立刻验证"prompt 里加场景描述"对模型输出的影响
有没有意义；做完再考虑要不要升到 B/C。

实施前要先解决的事实问题（等数据/源码确认）：

1. LEAD sample 怎么拿到 `scenario` 名？看 `lead_clip` 的 meta 或目录名？
2. LEAD 原始帧号 ↔ runner 当前 `anchor_t` 的换算关系（PROJECT_CONTEXT.md §1 帧率约定可参考）
3. `keyframes_all_scenarios.json` 路径——本机副本只是参考，正式跑要么打成
   小表打包进 runner，要么读远程路径

> 修改 prompt 时**只动 [`mot_lead_offline_runner.py`](AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py)**
> 这一个文件——不要去改 `automot_utils.py` 的 `build_cleaned_prompt_and_modes`
> （那不在白名单里）。可以在 runner 里调它拿到默认 prompt，再在前/后做拼接。

---

## 14. VLM 的两种使用范式（**改 prompt_cleaned 前必读**）

> 这一节解决一个核心认知误区：**`vlm_prompt_pipeline.py` 那种 "VLM 输出三行 STATUS/SUBGOAL"
> 的范式，和 AutoMoT `kv_cache_fixed_inference` 里的 prompt 处理范式，根本不是一回事。**
> 把 vlm_prompt_pipeline 风格 prompt 直接塞进 prompt_cleaned，VLM **不会按那个格式答题**——
> 因为整条 AutoMoT 推理链路从来就没有"让 VLM 生成文字回答"这一步。

### 14.1 总览对比表

| 维度 | 范式 A：**生成式 (generative decoding)** | 范式 B：**Prefill + Learnable Query** |
|---|---|---|
| 谁用 | [`vlm_prompt_pipeline.py`](vlm_prompt_pipeline.py)（远程仓库工具，AutoMoT 没接入） | AutoMoT 慢推理 + 快推理链路（runner 真正跑的路径） |
| 输入 | system + user prompt（+ 图像） | 慢路径喂 prompt + 历史帧；快路径喂当前帧 + 可学 query |
| 模型动作 | autoregressive 一步步采样 token，到 EOS 或 max_tokens 停 | 慢路径：**只跑一次 forward 留 KV cache**；快路径：**让 query cross-attend cache**，一次 forward 结束 |
| 模型输出 | **一段字符串**（如 `"ANALYSIS: ...\nSTATUS: match_speed\nSUBGOAL: merge_complete"`） | 慢路径：**KV cache 张量**；快路径：**`last_hidden_state`**（不是文本！） |
| "解码"在哪 | tokenizer.decode + 正则抽字段 | 在 query 位置的 hidden 上**外接 head**（lm_head 分类 / MLP 回归） |
| 是否有 for-loop / EOS | **有**（生成循环） | **无**（一次性 forward） |
| 是否产生新 K/V | 是（每生成一个 token 都扩 cache） | 慢路径：是；快路径：`update_past_key_values=False`，不扩 |
| 格式约束方式 | system prompt 软约束 + 解析端兜底 | 用**可学 query** + **外接 head** 把输出形状硬约束成需要的张量 |
| 训练成本 | 复用预训练 LLM 即可（指令跟随能力来自预训练 / SFT） | 需要训 reasoning_queries + lm_head 偏置 + waypoints_head 等所有 head |

### 14.2 范式 A — `vlm_prompt_pipeline.py`：让 VLM 真的"说话"

典型调用栈（参见 [`run_pipeline_step` L516](vlm_prompt_pipeline.py#L516)）：

```python
system = build_system_prompt()                          # "你必须按三行格式输出..."
user   = build_user_prompt(memory, image_description)   # <image> + [MEMORY]块 + "请输出ANALYSIS..."
raw    = vlm_fn(system, user)                           # ← 一次完整 .generate() 调用，返回字符串
parsed = parse_vlm_output(raw)                          # 正则抽 ANALYSIS / STATUS / SUBGOAL
updated_memory = update_memory(memory, parsed)          # 校验 + 推进状态机
```

注意 `vlm_fn` 是个 callable `(system, user) -> str`，**本文件没有实现它**——它把"真正
调模型"留给上层，签名典型如：

```python
def vlm_fn(system, user):
    resp = qwen_vl.chat(
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
        max_new_tokens=200, temperature=0.0,
    )
    return resp.text
```

**格式约束是软约束**：[`_SYSTEM_PROMPT` L317](vlm_prompt_pipeline.py#L317) 写"你必须按三行答"，
模型 autoregressive 生成时可能不听话——所以兜两层：

1. [`parse_vlm_output` L410](vlm_prompt_pipeline.py#L410)：lenient 正则，抽不到字段返回 `None`
2. [`update_memory` L434](vlm_prompt_pipeline.py#L434)：
   - 校验 event 名必须在 `event_sequence` 内，否则丢弃
   - 只允许 status 在序列上**前进或保持**，不允许回退
   - subgoal 由代码从 status 推导，不完全相信 VLM

经典 "LLM-as-classifier" 套路：prompt 让模型说人话 → 代码把人话约束回合法状态。

### 14.3 范式 B — AutoMoT：让 VLM 当"编码器 + cross-attention 上下文池"

#### 14.3.1 慢路径 — 只 prefill，不 decode

[`kv_cache_fixed_inference` L1233](AutoMoT/Automot/mot/evaluation/inference.py#L1233)：

```python
def kv_cache_fixed_inference(self, input_lists, ...):
    gen_context = self.init_gen_context()                         # 空 KV cache
    gen_context = self.update_kv_cache_context_qwen3vl(           # ← 关键
        user_prompt, instruction_prompt, image_list, gen_context
    )
    return gen_context   # ← 返回 KV cache 张量，不是字符串！
```

`update_kv_cache_context_qwen3vl`（[L556](AutoMoT/Automot/mot/evaluation/inference.py#L556)）核心：

```python
past_key_values, _ = self.model.forward_cache_update_generation(...)   # 跑一次 forward
```

**整个过程无循环、无采样、无 EOS**。`prompt_cleaned` 那句 "Your current and next target
point is ... Predict ..." 和 4 张 RGB 一起塞进 Transformer 跑一遍，把每层 attention 的
K/V 张量留下。返回的 `gen_context` 长这样：

```python
{
    'kv_lens': [627],
    'ropes':   [179],
    'past_key_values': NaiveCache(...),    # 每层一组 (K, V)
    'packed_position_ids': (3, 1, 626),
}
```

**`prompt_cleaned` 的作用**：当 soft prompt——它的内容（速度、目标点）通过 K/V 影响后续 query
在每层 attention 里"看到"的上下文。**模型永远不输出对 prompt_cleaned 的文字回答。**

#### 14.3.2 快路径 — `learnable query + cross-attend KV cache + 外接 head`

[`based_kv_cache_context_fast_qwen3vl` L244](AutoMoT/Automot/mot/evaluation/inference.py#L244) 骨架：

```python
# Step 1: 组装 query 序列
packed_sequence_fast = ...                  # [L_new, hidden]，初始化 0
packed_sequence_fast[packed_text_indexes]      = text_embed       # 当前帧文本 token
packed_sequence_fast[packed_vit_token_indexes] = vit_embed        # 当前帧 RGB/LiDAR 视觉 token
packed_sequence_fast[packed_reasoning_token_indexes] = \
    self.model.reasoning_projector(self.reasoning_query_tokens)   # ← 8 个可学 query
packed_sequence_fast[packed_action_token_indexes] = \
    self.model.action_projector(self.action_query_tokens)         # ← action 可学 query

# Step 2: query 去 attend 慢路径留下的 KV cache
last_hidden_state = self.model.language_model.forward_inference(
    packed_query_sequence=packed_sequence_fast,
    past_key_values=past_key_values,    # ← 慢路径的 cache
    is_causal=False,                    # 关键 1：query 全方向 attend
    update_past_key_values=False,       # 关键 2：不扩 cache
    ...
)
```

两个 flag 揭示本质：

- **`is_causal=False`**：不是 LLM 因果 mask。query 之间互相 attend、query → cache 全 attend。
  **每个 query 一次性看到全部 cache + 当前帧所有 token**。
- **`update_past_key_values=False`**：query 跑完一次 forward 就走，不写回 cache。

**没有生成循环**。一次 Transformer forward，输入 query，输出 query 经过 attention 后的 hidden state。
这是 **Perceiver IO / BLIP-2 Q-Former / DETR object query** 的套路，不是 LLM 的 `.generate()`。

#### 14.3.3 `reasoning_query_tokens=8` 是什么 — 可学 `nn.Embedding`

[`automot.py:317-322`](AutoMoT/Automot/mot/modeling/automot/automot.py#L317)：

```python
self.reasoning_query_tokens = config.reasoning_query_tokens          # = 8
self.reasoning_queries = nn.Embedding(
    num_embeddings=8,
    embedding_dim=self.reasoning_query_dim,
)
self.reasoning_projector = MLPconnector(reasoning_query_dim, hidden_size, ...)
```

`reasoning_queries.weight` 就是个 `[8, query_dim]` 的**可训练参数矩阵**——和 DETR object query、
BLIP-2 Q-Former 32 learnable queries 同一概念。每次推理用 `arange(8)` 取出 8 行
（[`inference.py:33`](AutoMoT/Automot/mot/evaluation/inference.py#L33)）：

```python
self.reasoning_query_tokens = self.model.reasoning_queries(torch.arange(8, ...))
# shape: [8, query_dim]
```

类似还有：

| 名字 | 数量 | 用途 |
|---|---|---|
| `reasoning_queries` | 8 | 决策（stop / keep 二分类） |
| `route_queries` | 20 | route 编码 |
| `waypoint_queries` | 6 | 6 个轨迹 waypoint |
| `action_queries` | 26 | anchor + route + waypoint 合集（runner 里 `action_tokens=26`） |

#### 14.3.4 外接 head 才是真正的"解码"

`last_hidden_state` 按下标切出 query 对应位置，分别接 head：

**① Reasoning head — 用 lm_head 做分类（伪装成"生成"）**

[`gen_fast_reasoning_decision` L160](AutoMoT/Automot/mot/evaluation/inference.py#L160)：

```python
reasoning_hidden_states = last_hidden_state[packed_reasoning_token_indexes]  # [8, 2560]
reasoning_logits = self.model.language_model.lm_head(reasoning_hidden_states)  # [8, vocab]
logits_per_sample = reasoning_logits.view(B, 8, -1)
second_token_logits = logits_per_sample[:, 1, :]                # 只看第 2 个 query
action_token_ids = [tokenizer.encode(w)[0] for w in ["stop","keep"]]
action_logits = second_token_logits[:, action_token_ids]        # vocab 维度只留 2 列
pred = action_logits.argmax(dim=-1)                             # 0=stop, 1=keep
```

关键 trick：

- **复用 Qwen3 自带 `lm_head`**（`[hidden, vocab]`），不做 autoregressive，只对 8 个 query 各做一次单 token 分类
- 推理只取**第 2 个 query**（第 1 个对应 `<|im_start|>` 等特殊位置），vocab 维度只保留 `stop / keep` 两个 id，等价于一个**二分类 head**
- 训练时把这 8 个位置监督到 GT 文字 token（如 `"accelerate, slow, slow"`），所以叫 "gen"；推理时大部分位置被丢弃

**② Waypoints head — 轨迹回归**

[`gen_fast_reasoning_trajectory` L112](AutoMoT/Automot/mot/evaluation/inference.py#L112)：

```python
action_hidden = last_hidden_state[packed_action_token_indexes]   # [N, 2560]
action = self.model.waypoints_head(action_hidden)                # MLP -> [N, 12]
pred_traj = action.reshape(-1, 6, 2)                             # 6 个 (x, y) 轨迹点
```

**完全不经过 vocab、不调用 tokenizer**，跟语言生成无关。一个标准 MLP head。

**③ Anchor head — 轨迹模式分类**

```python
pred_anchor = self.model.anchor_head(last_hidden_state[0][packed_anchor_token_indexes])
```

类似 multi-modal trajectory 的 mode classification。

### 14.4 一帧推理的数据流总图

```
慢路径 (kv_cache_fixed_inference)
  输入: 4 张历史 RGB + prompt_cleaned (含速度 / 目标点的英文模板)
  ─────────────────────────────────────────────
    forward 一次 Qwen3-VL，把 prompt + 图像编码进每层 attention 的 K/V
  ─────────────────────────────────────────────
  输出: past_key_values 张量    ← 不是字符串
                │
                ▼
快路径 (based_kv_cache_context_fast_qwen3vl[_dp])
  query 序列 = [当前帧 text emb, 当前帧 vit emb, BEV emb,
                reasoning_query × 8 (可学),
                action_query  × 26 (可学), ...]
  ─────────────────────────────────────────────
    forward_inference (is_causal=False, no kv update)
    query 与 past_key_values 做 cross-attention，
    query 之间互相 attend
  ─────────────────────────────────────────────
  输出: last_hidden_state
         │
         ├─ reasoning 位置 → lm_head → argmax(stop|keep)   "决策"
         ├─ action    位置 → waypoints_head → (6, 2)        "轨迹"
         └─ anchor    位置 → anchor_head    → mode           "模式"
```

### 14.5 这两种范式对改 `prompt_cleaned` 的影响（关键结论）

1. **直接把 `vlm_prompt_pipeline.build_user_prompt(memory, ...)` 输出塞进 `prompt_cleaned`：**
   - 可以塞，但 VLM **不会按 ANALYSIS/STATUS/SUBGOAL 三行答**
   - 因为 `kv_cache_fixed_inference` 不解码、不生成 token，多塞的内容只是变长了 KV cache
   - 效果：reasoning_query 和 action_query cross-attend 时多看了一段 memory 文本，间接影响下游
     `(stop/keep, traj)` 输出。**不会有"VLM 答了什么"的可读字符串。**

2. **想真的让模型"按 STATUS/SUBGOAL 格式输出"，必须额外跑范式 A：**
   - 在调 `kv_cache_fixed_inference` **之前**，先用 `qwen_vl.chat()` 跑一次纯 generative
     调用，拿到 STATUS/SUBGOAL 文本
   - 再把这段文本拼进 `prompt_cleaned` 作为 soft prompt，照常喂给慢路径
   - 这是 §13.5 方案 B 的真正实现方式（多一次 VLM 前向，但范式 A 那条链路才能产文字）

3. **范式 A 和范式 B 在 AutoMoT 当前代码里没有接通：**
   - `vlm_prompt_pipeline.py` 是远程独立工具，没有 import 到 AutoMoT 代码里
   - 接通的成本 = 在 runner 里写胶水，封装一个 `vlm_fn` callable，调 Qwen3-VL 的 chat 接口
   - 训练时如果想把 STATUS/SUBGOAL 信号端到端融入 reasoning_query，就是另一个项目了（§13.5 方案 C）

### 14.6 一句话记忆法

- **范式 A** = LLM 当**对话模型**用。`.generate()` 出文本，正则解析。格式靠 prompt 软约束。
- **范式 B** = LLM 当**特征提取器 + cross-attention 上下文池**用。一次 forward，外接 head 解码。格式靠 head 的 output shape 硬约束。
- **AutoMoT 是范式 B**。`prompt_cleaned` 是给范式 B 的"上下文 soft prompt"，不是给范式 A 的"问题"。

