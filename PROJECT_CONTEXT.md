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

## 1. 帧率与时间约定（重要！）

| 项 | lead | AutoMoT 在线 | 离线 runner |
|---|---|---|---|
| CARLA tick | 0.05 s (20 Hz) | 0.05 s (20 Hz) | 不跑 CARLA |
| 落盘 / 决策周期 | `data_save_freq=5` ⇒ **每 0.25 s 落盘 1 帧**（4 Hz） | 每 tick 决策但 RGB 历史**每 5 tick 抽一帧** ⇒ 0.25 s 间隔 | 每个 `.pkl` 已经是 0.25 s 间隔，`rgb_frame_step=1` 即 0.25 s |
| 预测视野 | `num_way_points_prediction=8`、`waypoints_spacing=5` ⇒ 8 个点 × 0.25 s = 2 s | traj 6 个点 × 0.5 s = 3 s（注意是 0.5 s 不是 0.25 s！） | 同 AutoMoT |
| 序列长度 | `sequence_length=12`（3 s 历史，4 Hz） | `rgb_history` deque maxlen = `obs_horizon*10 = 40 tick`（2 s 滚动缓冲）；采样取 `[-1, -6, -11, -16]` ⇒ 4 帧跨度 15 tick **= 0.75 s**；BUFFER_PHASE=31 tick (~1.55 s) | clip 默认 12 帧，`rgb_frame_count=4` |

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
> ⇒ **本仓库默认配置下，训练样本一律用 pos_global（真值）**。 离线 runner 当前是 filtered → pos → noisy 的优先级回退，跟训练默认不完全一致。

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

### 2.3 配置入口与"carla_leaderboard_mode=True"下的关键参数

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
- AutoMoT 的 BEV encoder 期望 **256×256 单通道直方图**，与 lead 的 320×384 不一致。**仅 resize 不够**——栅格代表的米数也不同，要同时改 `_rasterize_lidar_xy` 的 `min/max_x/y` 区间到 ±32 才对齐。
- lead 三视角拼接 RGB **比 AutoMoT 训练数据宽 128 px**，靠 `crop_array` 裁掉两侧 64 px 才到 1024。这相当于**把侧视部分丢掉**，但拼接位置不对——拼接图中间 384 列是真正的前视，两侧是侧视，crop 后留中间 1024 列：左 320 + 中 384 + 右 320 = 1024，**仍混入大量侧视像素**，与 AutoMoT 单视角 fov=110° 的训练分布不一致。
- traj 时间网格不匹配是结构性问题：lead 学的是 4 Hz×2 s，AutoMoT 输出 2 Hz×3 s。
- LiDAR 来源差异：lead `.laz` 含 5 个 ego-aligned sweep（**且可能混入 radar 检测点**），AutoMoT 在线只有当前+上一帧 N×3。点云密度与时间堆叠语义都不同。

---

## 8. `mot_lead_offline_runner.py` 当前实现 — **批判性审查**

### 8.1 是什么

`AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py` 这个脚本：

1. 从 LEAD 一个 route 目录读取 `clip_len=12` 连续 0.25 s 帧（`rgb/*.jpg, metas/*.pkl, lidar/*.laz`）。
2. 组成 `lead_clip = {rgb, rasterized_lidar, lidar_points, pos_global, theta, speed, target_point, target_point_next}`。
3. `LeadOfflineMoTRunner.run_clip()` 取 `anchor_t = clip_len - 1`（**最后一帧**作 anchor），
   构造 4 帧 RGB（`[t, t-1, t-2, t-3]`，间隔 1 帧 = 0.25 s）+ 2 帧 LiDAR 融合。
4. 调用 `kv_cache_fixed_inference`（慢路径）+ `based_kv_cache_context_fast_qwen3vl_dp`（快路径），返回 traj/route/text。

### 8.2 当前“跑通”不等于“对”——已识别的不匹配点

| # | 问题点 | 现状 | 影响 |
|---|---|---|---|
| ① | **BEV encoder lidar 输入尺寸错** | `_lead_lidar_to_bev_encoder_channel` 直接透传 LEAD 栅格 `(1, 320, 384)`；resize 到 256×256 的代码**被注释掉了**。注：仅 resize 不够，LEAD 栅格代表 `[-32,64]×[-40,40]m` 而训练时是 `[-32,32]×[-32,32]m`，米数也错 | bev_encoder 卷积输出 shape 从 (1,1512,8,8) 变成 **(1,1512,10,12)**，120 个 token 而非 64。⚠ `prepare_fast_kvcache` 中 `bev_token_max_num_tokens` 是按 `trans_feat.shape[-1]*trans_feat.shape[-2]` 动态算的，所以**不会 shape mismatch**也不会 crash，但模型权重训练时见到的是 8×8 网格，120 token 的特征对应错位 ⇒ **语义偏离训练分布** |
| ② | **主推理 lidar PIL 范围/通道错** | 离线版栅格化按 LEAD：`[-32,64]×[-40,40]`，4 px/m，单通道，复制 3 次成 RGB（R=G=B）；尺寸 320×384 | 不影响推理（PIL lidar 在 inferencer `__call__` 里被注释忽略），但保存日志会误导调试 |
| ③ | **三视角拼接 RGB 直接喂模型** | LEAD `rgb` 是 `(384, 1152, 3)`，runner 不挑前视直接 PIL；`bev_encoder_rgb` 走 `crop_array` 裁到 `(384, 1024, 3)` | inferencer 在 `__call__` 里强 `resize_image(512, 256)`，所以慢路径输入与训练（单视角 1024×512）的视野/比例严重不匹配；bev_encoder 路径 crop 后保留中间 1024 列，**左右仍混入大量侧视像素**，与 AutoMoT 训练分布偏离 |
| ④ | **target_point/next_target_point 语义不同** | `_extract_tp_ntp_from_future_frames`：取未来 `tp_lookahead_s=1.5 s` 与 `ntp_lookahead_s=3.0 s` 的 GT ego 位置 | 训练/在线时 tp 是 RoutePlanner 输出的“当前未访问的下一个 / 下下个里程碑”，距离可以是几十米的路口转弯点；这里改成短时未来真值（约 ego 前向 5–25m）→ prompt 字符串语义不同，模型理解会偏向“跟着未来真值走”而不是“规划到路口”，且在转弯点附近表现差异极大 |
| ⑤ | **LiDAR 多帧对齐与训练不一致** | runner 用 `_align_lidar_points_to_anchor`（用 `R(src_theta).T` 平移，比在线版 `R(anchor_theta).T` 数学上更严谨），把 `bev_frame_count=2` 帧手动拼到 anchor 帧；LEAD 训练时**直接用单帧 .laz**（采集时已含 5 个 ego-aligned sweep + 可选 radar），不做跨帧手工对齐 | sweep 在采集端 `lidar_pc_queue` (maxlen=5) 里是**滚动覆盖**——每个新 tick 进队列就挤掉最老的；`save_sensors` 每 5 tick 触发，落盘时 queue 里正好是当前这 5 个 tick 的 sweep。⇒ **相邻两个 .laz 的 5 sweep 完全不重叠**，单帧 .laz 覆盖时间窗 0.25 s。runner 拼 2 帧 .laz ⇒ **10 个不重复 sweep 压到 anchor**，覆盖时间窗 **0.5 s**，比训练（5 sweep / 0.25 s）**密度大一倍、时间窗长一倍**；且转弯时 `R(src_theta).T` vs `R(anchor_theta).T` 会产生明显不同的局部位移（示例 clip 几乎直行所以看不出来，转弯场景下会暴露） |
| ⑥ | **LiDAR 高度过滤几乎一致** | runner 用 `[-4, 10]` (`>=, <`)；LEAD `rasterize_lidar` 用 `[-4, 10]` (`<=, <=`) **闭区间** | 实质等价，仅相差 1 个采样面（z=10 是否保留），不影响推理 |
| ⑦ | **traj 时间网格不匹配** | runner 拿到 6×0.5 s = 3 s 输出，但 LEAD GT 是 8×0.25 s = 2 s | 如果用 LEAD GT 做 evaluation，需要把 traj 重新插值/采样到 0.25 s 网格，runner 当前没有这一步 |
| ⑧ | **command 队列状态机离线缺失** | runner 完全不维护 `self.commands = deque(maxlen=2)`；在线 inferencer 的 prompt 里确实不含 command，但 agent 内部 `commands` 队列参与 force_move / 路径切换 / `next_command` 计算 | 当前 runner 走"开环单点推理"，没用控制层；若未来要做 close-loop 评测，必须把 commands 队列搬过来 |
| ⑨ | **theta 单位** | LEAD `theta` 是 CARLA `ego_orientation_rad`（弧度），通过 `preprocess_compass` + unwrap 累积；和 `pos_global` 配合做 `inverse_conversion_2d` | 与 `automot_utils.inverse_conversion_2d` 期望一致 ✓。⚠ 若想用 `privileged_yaw`（=`np.deg2rad(transform.rotation.yaw)`）替代 `theta`，要确认是否同号（实测同号，但 privileged_yaw 不会 unwrap，跨 ±π 边界会有跳变） |
| ⑩ | **位姿源不严格一致** | `_extract_pose_from_meta` 优先 filtered → pos → noisy；**训练默认（`use_noisy_tp=False`）是 `pos_global`** | runner 跟训练默认偏离一档：在 LEAD 数据上 `filtered_pos_global` 与 `pos_global` 一般差异小（Kalman 已收敛），影响小但不是"完全一致" |
| ⑪ | **clip_len 与 BUFFER_PHASE** | 离线 12 帧（0.25 s 间隔）= 3 s 历史 vs 在线 31 tick (~1.55 s) buffer 然后再决策 | 时间长度比在线还多，不算问题 |
| ⑫ | **gen_context 复用语义不同** | runner 每个 anchor 重新 `kv_cache_fixed_inference`（gen_context=None 时）；在线版 `kv_cache_inference_slow_fast_dp` 自己按 `slow_update_interval=2` 帧管理刷新 | 当前 `run_clip` 只用最后一帧 anchor 问题不暴露。**未来若多 anchor 复用，应直接调 `kv_cache_inference_slow_fast_dp` 而不是自己手动凑 gen_context**，否则会跳过 inferencer 内置的 interval 控制 |
| ⑬ | **二次 JPEG 压缩** | `_prepare_inference_inputs` line 450 对最后一帧 RGB 再 `cv2.imencode('.jpg', …) → imdecode`；LEAD `rgb/*.jpg` 已是一次 JPEG，runner 再 encode 一次共 2 次 | 在线 agent 也做 1 次 encode（注入压缩伪影对齐训练），但训练数据本身是 1 次；离线再加一次 = 2 次，与训练分布有微小差异 |
| ⑭ | **LiDAR / Radar 混合点** | runner 用 `laspy.read(*.laz)` 拿到的可能是 LiDAR + Radar 混合点（取决于采集时 `save_radar_pc_as_lidar` 设置）；runner 不区分，全部喂栅格 | 与训练分布一致（训练时 dataset 也用 `laspy.read` 不区分），所以**不是 bug**；但要注意如果训练时 `duplicate_radar_near_ego=True`，ego 附近密度会被人为增厚 |

### 8.3 数据流形/数值范围（用户日志反推）

`build_clip_from_real_lead_route` 输出（实际跑通的统计）：

```
rgb               : (12, 384, 1152, 3) uint8     [0, 255]
rasterized_lidar  : (12, 1, 320, 384) float32    [0, 1]      # LEAD 栅格风格
lidar_points      : list[12] 变长 float32 范围~[-93, 124]    # ego-local，每帧 4k~35k 点
pos_global        : (12, 2) float32  示例 [-0.25, 92.8]      # 世界 m
theta             : (12,) float32   ~1.594 rad
speed             : (12,) float32   ~[7.5e-06, 8] m/s
target_point      : (12, 2) float32  ~[-9e-4, 10.6]          # ego 前向米
target_point_next : (12, 2) float32  ~[-6e-4, 25.1]
```

`_prepare_inference_inputs` 输出：

```
rgb_pil_list      : list[4]，每张 PIL.size=(W=1152, H=384) RGB
lidar_pil_list    : list[1]，PIL.size=(W=384, H=320) RGB（B=G=R）   # PIL 风格 W×H
target_point_speed: (1, 5) float32  [speed, tp.x, tp.y, ntp.x, ntp.y]
bev_rgb_tensor    : (1, 3, 384, 1024) bf16   [0, ~235]              # crop_array 后
bev_lidar_tensor  : (1, 1, 320, 384) bf16    [0, 1]                 # ⚠ runner 注释里说 256×256 是错的，实际是 320×384
```

→ `bev_encoder(rgb, lidar_bev)`：
- 因为输入 320×384 ≠ 256×256，输出 `bev_feature` 是 **(1, 1512, 10, 12)** 而非 (1, 1512, 8, 8)。
- 快路径 `prepare_fast_kvcache` 中 `bev_token_max_num_tokens = trans_feat.shape[-1] * trans_feat.shape[-2]`
  是**动态计算**的（[automot.py:1973](AutoMoT/Automot/mot/modeling/automot/automot.py#L1973)），
  所以 `packed_bev_token_indexes` 会自动扩到 120，**不会 shape mismatch / IndexError**——
  这就是为什么 runner 能跑通。
- 然而 trans_feat 的语义网格变成 (10×12 来自非对称 BEV `[-32,64]×[-40,40]`)，
  模型权重训练时见到的是 (8×8 来自对称 BEV `[-32,32]²`)，**特征对应错位**。
- 行为不报错但**推理质量已偏离训练分布**——这是 runner 当前最大的"沉默错误"。

### 8.4 “能跑通”但“不可信”的解释

- 模型权重对输入的健壮性会让前向不报错（注意力会“通融”各种尺寸）。
- 但是输出的 traj / route / decision 与 lead GT 几乎肯定**不可比**：
  1. trans_feat 的 BEV 网格与训练分布偏离；
  2. RGB 是三视角拼接而不是单前视；
  3. target_point 含义已变。

---

## 9. 推荐的修复方向（仅作参考，**不在本次任务范围**）

> **用户原则**：修改优先只在 `mot_lead_offline_runner.py`，别处需先同意。
> 下列建议都是该文件内可做的：

1. **`bev_lidar_tensor` resize 到 256×256**：把当前被注释的 resize 代码恢复，或改用
   AutoMoT 风格的栅格（[-32,32]×[-32,32] @ 4 px/m），让 `trans_feat = (1, 1512, 8, 8)`。
2. **RGB 切前视**：从 `(384, 1152, 3)` 拼接图中切出中段 384 列（前视相机 idx=2）做
   resize/pad 到 1024×512 后再喂；`bev_encoder_rgb` 同步用切出来的前视。
3. **target_point 用 LEAD meta 的 `next_target_points_3.25`**：直接读 `meta["next_target_points_3.25"]`
   的第 1 和第 2 个（世界坐标），用 `inverse_conversion_2d(pos_global, theta)` 转 ego。
   这样 prompt 语义和训练 / 在线一致。
4. **bev_encoder LiDAR 路径用 AutoMoT 风格栅格**：复用
   `team_code/bev_data_utils.lidar_to_histogram_features(self.bev_encoder_config)`，
   而不是 LEAD 风格。注意：
   - 它**内部已经做 `z > 0.2` 切片**（与 LEAD `[-4, 10]` 含地面层不一致，但这正是 BEV encoder 训练的分布），所以**不要手动叠加 LEAD 风格的 z 过滤**；
   - 它的返回 shape 是 `(1, H, W)`（多一个 channel 维），喂给 `bev_encoder` 时 `.unsqueeze(0)` 得到 `(1, 1, 256, 256)`，与当前 runner 透传 `(1, 320, 384)` 的 channel-1 习惯一致。
5. **多帧 LiDAR 对齐与训练对齐**：要么只用单帧 `lidar_points[anchor]`（与 LEAD 训练一致），
   要么模仿在线 agent 的“前后两帧融合”而不是 N 帧（runner 默认 `bev_frame_count=2` 这点已是 OK 的，
   但是要确认采用与训练分布一致的语义）。
6. **traj 时间网格**：若想与 LEAD 0.25 s GT 对齐 evaluation，需要在 runner 输出后
   把 6×0.5 s 累计位移插值到 8×0.25 s。

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

离线 runner: 当前已知问题（按严重度从高到低）
   - bev_lidar 入网 320×384（应 resize 256×256 + 改栅格范围到±32）
   - trans_feat 输出 (1,1512,10,12) 非 (1,1512,8,8) → bev_token_max_num_tokens
     是动态算的所以不会崩, 但语义偏离训练分布
   - tp/ntp 用未来1.5s/3.0s GT 而非 next_target_points_3.25 ⇒ prompt 语义变了
   - RGB 三视角拼接图直喂 (1152→1024 crop) 含大量侧视
   - 离线没维护 self.commands deque (close-loop 评测时会缺)
   - 二次 JPEG 压缩 (LEAD .jpg → encode → decode)
   - LiDAR 跨帧手动对齐叠加 (5-sweep .laz 已含累积; runner 又拼2帧)
   - 位姿源 filtered → pos 回退 vs 训练默认 pos_global (差异小)
```
