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

### 已搁置的路径

**快推理**（`bev_encoder` + `bev_encoder_proj` + `reasoning_projector` + `route_head` + `waypoint_head` + 各种 learnable queries）= **用户预计放弃**。

这部分都是 AutoMoT 作者训过的部件，对训练分布敏感。但：
- 用户表态："快推理的东西我之后大概率会放弃，你就可以暂时先不管了"
- ⇒ §8.2 表格里所有**仅影响 trans_feat / 快推理路径**的差异（相机物理位置差 1.85 m、LiDAR sweep 数 5 vs 2、bev_encoder RGB 视野不对齐等）**保留作历史记录但不再优化**

### 替代方案：未来若要彻底解决，重训 bev_encoder

如果未来用户决定不放弃 BEV/快推理路径：

- ✅ **推荐**：重训整个 bev_encoder + 下游 projector/head（用 LEAD 数据训练，Qwen3-VL 仍 frozen）。这样 runner 完全不需要做"AutoMoT 训练分布对齐"，所有视角/格式/sweep 数差异自然消失
- ❌ **不推荐**：只重训 bev_encoder backbone 但保留下游 projector/head 不动——风险高，bev_encoder 输出特征语义变了，下游 projector 没见过，整体性能可能反而变差

详细任务清单见 §12「未来工作」。

### 当前 runner 对慢推理的输入状态

针对慢推理（`kv_cache_fixed_inference(rgb_pil_list + [prompt])`）的**所有输入项已经全部对齐**：

| 输入项 | 现状 | 是否对齐慢推理需求 |
|---|---|---|
| 4 帧 RGB PIL `(W=1152, H=384)` 三视角拼接 | 直接喂，不切、不 resize | ✅ Qwen3-VL 自适应消化 |
| `prompt_cleaned` 文本 | speed `:.2f`、tp/ntp `:.6f`、ego frame 米 | ✅ |
| `target_point/ntp`（未来 1.5s/3.0s 真值 → ego frame） | 距离量级 7–75 m，与 AutoMoT RoutePlanner 同分布 | ✅ |
| `theta` / `pos_global` | 弧度 + 米，与 `inverse_conversion_2d` 配对正确 | ✅ |

**runner 在慢推理路径上不需要任何额外修改**。

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
- lead 训练数据**前后非对称**（前 64m / 后 32m），AutoMoT 训练数据**对称 ±32m**。
- AutoMoT 的 BEV encoder 期望 **256×256 单通道直方图**，与 lead 的 320×384 不一致。**仅 resize 不够**——栅格代表的米数也不同，必须改栅格 `min/max_x/y` 区间到 ±32。runner 当前**直接调 `bev_data_utils.lidar_to_histogram_features(self.bev_encoder_config)`** 完全沿用 AutoMoT 栅格逻辑，不再有自定义 LEAD 风格栅格函数。
- lead 三视角拼接 RGB **比 AutoMoT 训练数据宽 128 px**，靠 `crop_array` 裁掉两侧 64 px 才到 1024。这相当于**把侧视部分丢掉**，但拼接位置不对——拼接图中间 384 列是真正的前视，两侧是侧视，crop 后留中间 1024 列：左 320 + 中 384 + 右 320 = 1024，**仍混入大量侧视像素**，与 AutoMoT 单视角 fov=110° 的训练分布不一致。
- traj 时间网格不匹配是结构性问题：lead 学的是 4 Hz×2 s，AutoMoT 输出 2 Hz×3 s。
- LiDAR 来源差异：lead `.laz` 含 5 个 ego-aligned sweep（**且可能混入 radar 检测点**），AutoMoT 在线只有当前+上一帧 N×3。点云密度与时间堆叠语义都不同。

---

## 8. `mot_lead_offline_runner.py` 当前实现 — **批判性审查**

### 8.1 是什么

`AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py` 这个脚本：

1. 用户显式输入 `--anchor`（route 内绝对帧索引，默认 12）；由 `rgb_frame_count`/`bev_frame_count`/`step` 反推 `max_history`，决定要从 route 加载 `[anchor-max_history, anchor]` 这段帧。anchor 太靠前时打印 warning 并重复 frame 0 补 0，不报错。
2. 组成 `lead_clip = {rgb, lidar_points, pos_global, theta, speed, target_point, target_point_next}`（只存原始点云，不再缓存栅格化结果）。
3. `LeadOfflineMoTRunner.run_clip()` 用 clip 内最后一帧（=输入的 anchor）作 anchor_t，构造 **4 帧 RGB**（`[t-3, t-2, t-1, t]`，间隔 1 帧 = 0.25 s）+ **1 帧 LiDAR**（默认 `bev_frame_count=1`，对齐 LEAD 单帧 .laz 含 5 累积 sweep 的训练分布）。
4. LiDAR：在 `_prepare_inference_inputs` 内做"跨帧对齐到 anchor ego-local（用 `R(src_theta).T`，比在线 agent 的 `R(anchor_theta).T` 更严谨）→ `bev_data_utils.lidar_to_histogram_features(self.bev_encoder_config)` AutoMoT 风格栅格化 → (1, 256, 256)"。trans_feat 输出 `(1, 1512, 8, 8)` 与训练分布一致。
5. 调用 `kv_cache_fixed_inference`（慢路径）+ `based_kv_cache_context_fast_qwen3vl_dp`（快路径），返回 traj/route/text。

### 8.2 当前"跑通"不等于"对"——已识别的不匹配点

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

### 8.3 数据流形/数值范围（用户日志反推）

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

注意：相比修订前，`rasterized_lidar` 字段已删除（栅格化全部在 `_prepare_inference_inputs` 里按 AutoMoT config 重新算）。

`_prepare_inference_inputs` 输出：

```
rgb_pil_list      : list[4]，每张 PIL.size=(W=1152, H=384) RGB
lidar_pil_list    : list[1]，PIL.size=(W=256, H=256) RGB（B=G=R，仅日志用）
target_point_speed: (1, 5) float32  [speed, tp.x, tp.y, ntp.x, ntp.y]
bev_rgb_tensor    : (1, 3, 384, 1024) bf16   [0, ~235]              # crop_array 后，直接 PIL→array 不再 re-encode
bev_lidar_tensor  : (1, 1, 256, 256)  bf16   [0, 1]                 # AutoMoT 风格栅格（±32m, z>0.2 切地面）
```

→ `bev_encoder(rgb, lidar_bev)`：
- 输出 `bev_feature` (trans_feat) = **(1, 1512, 8, 8)**，BEV token 数 = 64，**与训练分布完全一致**
- `bev_feature_upscale: (1, 64, 64, 64)`、`fused_features: (1, 1512, 8, 8)`、`image_feature_grid: (1, 1512, 12, 32)`

### 8.4 当前剩余偏差（按路径分组）

**慢推理路径（用户**当前关心**的路径）**：✅ **所有偏差均不影响**
- Qwen3-VL frozen 通用 backbone，能消化 (1152, 384) 三视角拼接图
- prompt / target_point / theta / pos_global / speed 单位精度全部对齐
- ⇒ **runner 在慢推理路径上不需要任何额外修改**

**快推理路径（用户预计放弃，暂不优化）**：⚪ 以下偏差**保留为历史记录**：
- RGB 视野不对齐（含侧视、相机物理位置 x 差 1.85 m）
- LiDAR sweep 数（5 vs 在线 2）
- 这些只有重训 bev_encoder + 下游 head 才能根治（见 §12）

**Close-loop / evaluation 才相关**：
- traj 时间网格 6×0.5s vs LEAD GT 8×0.25s（eval 脚本内重采样即可）
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
| **LiDAR BEV** | `(1, 1, 256, 256)` bf16 [0, 1] | `(1, 1, 256, 256)` bf16 [0, 1] | ✅（用同一个 `lidar_to_histogram_features(self.bev_encoder_config)`） |
| **RGB tensor** | `(1, 3, 384, 1024) bf16` `[0, ~235]` | `(1, 3, 384, 1024) bf16` `[0, ~235]` | ✅ shape/dtype/范围；⚠ 内容含侧视（§8.4 第 1 条） |
| **`v_target_point`** | `(1, 5) float32 = [speed, tp.x, tp.y, ntp.x, ntp.y]` | 同 | ✅ |
| **`trans_feat`** | `(1, 1512, 8, 8) bf16` | 同（因 BEV LiDAR 已对齐到 256×256） | ✅ |

**结论**：第一次调用 `kv_cache_fixed_inference` 之前的所有数值/单位/精度都与训练分布对齐。

> **针对当前路线（慢推理 Qwen3-VL 为主）**：上表所有项**全部 ✅**——Qwen3-VL frozen 对图像 shape 鲁棒，连 RGB 三视角拼接图都能消化，runner 慢推理路径**不需要任何修改**。
> **快推理 trans_feat 路径**的偏差（视野/sweep 数等）参见 §8.2 ⚪ 标记项，已搁置。

---

## 9. 修复进度（仅在 `mot_lead_offline_runner.py` 内可做的方向）

1. ✅ **已实施**：`bev_lidar_tensor` 改用 `bev_data_utils.lidar_to_histogram_features(self.bev_encoder_config)`，直接出 (1, 256, 256)；`trans_feat = (1, 1512, 8, 8)` 恢复训练分布。
2. ⏸ **暂不动**（用户决定）：RGB 切前视。用户准备替换快路径，慢路径 Qwen3VL ViT 能读三视图。
3. ✅ **已对齐 AutoMoT 训练分布**：target_point/ntp 用未来 1.5 s/3.0 s 真值。LEAD `next_target_points_3.25` milestone 距离量级（100–200 m，0026.json 实测验证）远超 AutoMoT 训练分布（30–80 m）；future GT 1.5 s/3.0 s 数量级（速度依赖：7~75 m）匹配 AutoMoT 训练 RoutePlanner 输出。距离随速度变化是用户接受的预期行为。
4. ✅ **已实施**：BEV encoder LiDAR 走 `lidar_to_histogram_features`，AutoMoT 风格栅格。
5. ✅ **已实施**：`bev_frame_count` 默认改 **1**，仅 anchor 单帧 .laz（含 5 sweep）— 完全对齐 LEAD 训练分布。
6. ⏸ **未实施**：traj 时间网格 6×0.5s → 8×0.25s 重采样。仅 evaluation 阶段需要，runner 本身不涉及。

额外已实施（不在原列表）：
- ✅ `_extract_pose_from_meta` 严格用 `pos_global + theta`，不再回退到 filtered/noisy/privileged_yaw（对齐 LEAD `use_noisy_tp=False` 训练默认）
- ✅ 去掉 BEV encoder RGB 路径的二次 JPEG re-encode（LEAD .jpg 已是 1 次压缩，对齐训练数据分布）
- ✅ CLI 入口语义：`--start-frame/--clip-len` 替换为显式 `--anchor`，由采样参数反推 max_history
- ✅ 删除冗余的 `_rasterize_lidar_xy` 函数和 clip 的 `rasterized_lidar` 缓存字段

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
【路线】慢推理 Qwen3-VL (frozen 原始权重) = 唯一关心的路径
       快推理 (trans_feat / bev_encoder / 下游 AutoMoT-trained head) = 预计放弃, 暂不优化
       ⇒ runner 慢推理路径无需修改 (Qwen3-VL 通用 backbone 对图像 shape 鲁棒)
       未来若不放弃快推理 ⇒ 重训整个 decoder 链路, 见 §12

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

AutoMoT 在线: 20Hz, 每tick决策
       RGB 历史 deque(maxlen=40 tick=2s), 采样[-1,-6,-11,-16] 跨度15tick=0.75s
       LiDAR(主可视PIL): ~6.96px/m, [-32,32]², 448×448×3 uint8
            R=log密度, G=高度, B=0 (输入补1当伪intensity后,line124强制清零)
            x 取反 ⇒ 图像里车头朝下; |x|<2.4 ∧ |y|<1 中心去除
            ⚠ 实际不进 inferencer __call__ 的 input_list, 只为调试
       LiDAR(BEV enc): 4px/m, [-32,32]², 256×256×1 [0,1], shape (1, H, W)
                       上界 max_height_lidar=100 形同虚设
                       use_ground_plane=False ⇒ 有效 z 过滤 = z > 0.2 (above-only)
                       ⚠ 地面层 z<=0.2 全丢, 与 LEAD [-4,10] 闭区间含地面 不同!
                       splat 空间轴序与 LEAD 同款 row=y col=x
                       min_z_projection=-10 / max_z_projection=14 不在 splat 用,
                            仅 create_projection_grid 用
       BEV encoder: seq_len=img_seq_len=lidar_seq_len=1, 训练/推理都是单帧 (不堆时序)
       RGB: 单前视 PIL.size=(1024,512) fov=110, bev_encoder 路径 crop_array 裁到 (H=384,W=1024)
       trans_feat: (1,1512,8,8) bf16  ← 真正的BEV模态
       inferencer: 4 PIL RGB(resize PIL W=512 H=256) + text → 慢KV
                   + trans_feat + v_target_point(5) + reasoning(8)+route(20)+wp(6) → 快
       输出: traj(1,6,2 cumsum, 0.5s间隔), route(1,20,2), text "verb,verb,verb"
       control: traj→PID油门刹, route→PID转向; 末尾 line 1581 限速 35km/h 硬保护

离线 runner: 已修复 vs 仍存在
   ✅ bev_lidar 入网 (1,256,256) AutoMoT 风格 + trans_feat (1,1512,8,8) 训练分布对齐
   ✅ bev_frame_count 默认 1 (仅 anchor 单帧 .laz 含 5 sweep, 对齐 LEAD 训练)
   ✅ 位姿源严格 pos_global + theta, 无回退
   ✅ 去掉二次 JPEG (LEAD .jpg 已 1 次压缩, 对齐训练)
   ✅ CLI 改 --anchor 显式输入, 由采样参数反推 max_history
   ✅ 删除 rasterized_lidar 缓存字段和 _rasterize_lidar_xy 函数
   ✅ tp/ntp 用未来 1.5s/3.0s 真值: 距离 ≈ speed×lookahead = 7~75m, 落在 AutoMoT 训练分布 30-80m 内
      (LEAD next_target_points_3.25 milestone 实测 100-200m, 与 AutoMoT 训练不匹配)
      精度: prompt 用 :.6f, tensor float32; ego frame: x=前向, y=右向 (CARLA 顺时针 yaw 约定)
      速度依赖是预期: 红灯停车时退化 ~(0,0), 用户接受
   ⏸ RGB 三视角拼接图直喂 (用户: 慢路径 Qwen3VL 能读, 快路径将整体替换)
   ⏸ 离线没维护 self.commands deque (close-loop 评测才相关)
   ⏸ traj 时间网格 6×0.5s vs LEAD GT 8×0.25s (eval 时插值即可, runner 内不涉及)
```

---

## 12. 未来工作 / 路线相关待办

> 这些是**已经讨论但暂不实施**的工作项，按优先级 / 工程量列出。新对话接手时若用户提到要做，可直接查这里。

### 🟢 当前阶段无需做的（基于 §0.5 路线决策）

1. **慢推理 RGB resize / 切片** — ❌ 不做
   - 理由：Qwen3-VL frozen 通用 backbone，对任意 aspect/分辨率鲁棒；当前 (1152, 384) 三视角拼接图直接喂没问题
   - 用户明确表态："不希望要做"
   - ⇒ runner 慢推理路径**保持现状**

2. **bev_encoder 输入对齐**（切 [192:960] 视野、resize 到 (1024, 384)、LiDAR sweep 数对齐等） — ❌ 不做
   - 理由：仅影响快推理路径，用户预计放弃
   - 相关历史讨论见 §8.2 ②③④ 表格

### 🟡 中期可能做的（若改变路线 / 想搞快推理）

3. **重训整个 AutoMoT decoder 链路（Qwen3-VL 仍 frozen）**
   - 范围：`bev_encoder` backbone + `bev_encoder_proj` + 各 `*_projector` + 各 `*_head` + 各 `*_queries`
   - 数据：LEAD 数据集（已经有 `bev_semantic / detect_boxes / depth / semantic / future_positions / target_speed_classes / next_target_points_3.25 / 离散 command` 等所有需要的 GT）
   - **训完之后 runner 完全不需要做"AutoMoT 训练分布对齐"**——所有视角/格式/sweep 数差异自然消失
   - 工程量：大（要搭训练 pipeline、可能要写 Loss、需要 GPU 资源）
   - **不推荐只重训 bev_encoder backbone**：风险高，输出特征语义变了下游 projector/head 没见过

4. **如果走方案 3 后，PROJECT_CONTEXT.md 大改**：
   - 删掉 §0.5 中"快推理已搁置"的论断
   - §8.2 表格里 ②③④ 等"仅快推理"标记全部归零
   - cheat sheet 顶部路线行重写

### 🔴 远期 / 若做 close-loop 评测才相关

5. **traj 时间网格对齐**（6×0.5s → 8×0.25s 重采样）
   - 仅在用 LEAD GT 算 ADE/FDE 时需要
   - runner 内不必处理，加在 evaluation 脚本里即可

6. **离线维护 `self.commands` deque**
   - 用于 force_move / parking_escape / 路径切换状态机
   - 当前 runner 走"开环单点推理"用不到
   - 真要做 close-loop 评测时再加

7. **LiDAR sweep 数对齐**（如果想严格匹配 AutoMoT 在线 2 sweep）
   - LEAD `.laz` 已经 5 sweep 烧死了，且 `laspy.read` 读不到 `time` 字段无法过滤
   - 实际无解，除非重训 bev_encoder 让它适配 5 sweep

### 🧰 工程清理（任何时候都可以做）

8. **删除 runner 里和快推理相关的死代码**（如果用户明确放弃）
   - `BEVEncoderBackboneExtractor` 加载
   - `_lead_lidar_to_bev_encoder_channel` / 各种 LiDAR 栅格调用
   - `bev_encoder(...)` 前向调用
   - `based_kv_cache_context_fast_qwen3vl_dp` 调用
   - 把 `run_step` / `run_clip` 改成只产出 `gen_context`（慢推理 KV cache）
   - 体积大幅缩减，文件更聚焦
   - ⚠ 删除前请用户确认快推理真不要了
