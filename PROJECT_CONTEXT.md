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

> **详细字段值查 `0026.json` 而非翻 expert.py 源码（省 token）**。下面只列关键字段名。

| 字段族 | 关键字段 | 备注 |
|---|---|---|
| **位姿/朝向** (CARLA world frame + yaw 弧度) | `pos_global` (3D 真值), `noisy_/filtered_pos_global` (GPS frame, **≠ world**), `theta` (compass-π/2, unwrap), `privileged_yaw` (transform.yaw), `ego_matrix` (4×4) | runner 严格用 `pos_global + theta`；训练默认 `use_noisy_tp=False` 也是同源 |
| **自车动力学** | `speed`, `accel_x/y/z`, `angular_velocity_*`, `target_speed`, `steer/throttle/brake`, `privileged_acceleration/rotation_speed` | |
| **未来量** (offline 后处理填，**当前帧 ego frame**) | `future_positions` (61, 3) = 60×0.05s = **3 s**, `future_yaws`, `future_speeds` | |
| **过去量** (反序，新→旧) | `past_positions`, `past_filtered_state`, `past_speeds`, `past_yaws`, `privileged_past_positions` | |
| **Target points** (**world frame**！) | `next_target_points_{k}`, `next_commands_{k}` (29 套, k∈[3.0..10.0]) + GPS 版 | 训练默认 `tp_pop_distance=3.25`；`target_point = inverse_conversion_2d(next_tp_list[1], pos, yaw)`；`discrete_command_dim=6` one-hot |
| **Route** | `route` (N≤50, world) | 训练取前 20 → smooth → 前 10 作 GT |
| **场景标签** | `current_/previous_active_scenario_type`, `vehicle_/light_/walker_/stop_sign_hazard`, `town` | 完整列表见 [carla_dataset.py:250-302](lead/lead/data_loader/carla_dataset.py#L250-L302) |

> ⚠ **训练采 8 点×0.25 s vs AutoMoT 输出 6 点×0.5 s**——dataset 用 `future_waypoint_indices=[5,10,…,40]`（`waypoints_spacing=5`）从 `future_positions` 跳采 `num_way_points_prediction=8` 个 0.25 s 间隔点，覆盖 2 s；AutoMoT 输出 6×0.5 s = 3 s。**两者不在同一时间网格**，重训快推理时要解决。

> ⚠ **`scenario_type` 是 dataset 派生字段**（[carla_dataset.py:362-374](lead/lead/data_loader/carla_dataset.py#L362-L374)）：current 非 NA 取 current，否则取 previous，否则 "NA"。直接 `pkl["scenario_type"]` 会 KeyError——meta 里只有 `current_/previous_active_scenario_type`。`SCENARIO_TYPES` 共 **50** 项（[constants.py:334-385](lead/lead/common/constants.py#L334-L385)）。

### 2.3 参考样本：`0026.json`（**只读，绝对禁止 `git add`**）

工作目录根的 `0026.json` 是 LEAD meta.pkl 转 JSON 标准参考样本，350 个顶层 key（含 29 套 `next_target_points_{k}` k∈{3.0..10.0}），实际数值版本的 §2.2 全部字段。**验证 meta 字段时优先查它，省去翻 `expert.py` 源码**。靠 CLAUDE.md §2-3 规则保护（禁修改、禁入库）。

**用它核对过的关键推论**（已落地到 runner 设计）：

1. `next_target_points[1]` 转 ego ≈120 m、`[2]` ≈216 m，**远超 AutoMoT 训练分布 30–80 m** → §8.2 ④ runner 用 future 1.5s/3.0s GT 而非 `next_target_points_3.25`
2. `filtered_pos_global` (GPS frame, x=-3.61) 与 `pos_global` (world frame, x=229.79) **不同坐标系** → runner 严格走 `pos_global`
3. `theta` ≈ `privileged_yaw`（差 ~1e-7）→ runner 用 `theta` 合理

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

### 3.1 LiDAR 栅格化（**与 AutoMoT 对照是 runner 关键**）

[`rasterize_lidar`](lead/lead/data_loader/carla_dataset_utils.py#L30-L82)：

```python
lidar = lidar[(min_height_lidar=-4 <= z) & (z <= max_height_lidar=10)]   # 保留地面
xbins = linspace(-32, 64, 385)   # 384 bins
ybins = linspace(-40, 40, 321)   # 320 bins
hist  = histogramdd(lidar[:, :2], (xbins, ybins))
hist  = clip(hist, 0, 5) / 5     # → [0, 1]
return hist.T                    # (320, 384) float32 单通道
```

- 输入 `lidar[:, :3]` 是 **ego-local** (CARLA 朝向 x 前、y 右、z 上)，采集时已用 `accumulate_lidar` 在 ego frame 拼好 5 sweep
- 输出 `(320, 384)`：**row=ego y（右正），col=ego x（前正）**——非典型朝向
- `.laz` 含 5 累积 sweep + 可选 radar 混在一起，`laspy.read` 不可区分
- **无 `_perturbated.laz` 文件**（[carla_dataset.py:904](lead/lead/data_loader/carla_dataset.py#L904)）；扰动靠 `align_lidar()` 数学变换

> ⚠ **runner 复用 AutoMoT 栅格的 3 个差异点**（[bev_data_utils.py](AutoMoT/leaderboard/team_code/bev_data_utils.py) `lidar_to_histogram_features`）：
>
> | | LEAD | AutoMoT |
> |---|---|---|
> | xy 范围 | `[-32,64]×[-40,40]` 不对称 | `±32` 对称 |
> | z 下界 | `-4 <= z`，**含地面** | 按 `lidar_split_height=0.2` 切 above；`use_ground_plane=False` 时 **z≤0.2 全丢** |
> | 输出 shape | `(H, W)` 单通道 | `(1, H, W)` 多 channel 维 |
>
> ⇒ runner 用 AutoMoT 栅格要：(1) 改区间到 ±32；(2) 先 `z > 0.2` 切地面；(3) shape `(1, H, W)` 配 `(B=1, C=1, H, W)`

### 3.2 BEV 占用 / 语义图

[`build_bev_occupancy`](lead/lead/data_loader/carla_dataset_utils.py#L800)：1024×1024 (4 px/m, ±128 m) 大栅格画 bbox，加载时切到 ego ±[min,max]，`bev_semantic` 用 `.repeat(2)` 从 2 px/m 升采到 4 px/m。最终训练用 `(320, 384) uint8` 类别图。坐标轴与 §3.1 LiDAR 同款（col=x 前正，row=y 右正）。

### 3.3 RGB

[carla_dataset.py:528-724](lead/lead/data_loader/carla_dataset.py#L528-L724)：JPEG → `cv2.imdecode` → BGR2RGB → 颜色扰动 (prob 0.2) → `(C, H, W)` → `used_cameras` 切分。最终 `(3, 384, 1152) uint8 → /255 float`，**不做 ImageNet normalize**（backbone 自己处理）。

### 3.4 Boxes / Radar / Depth / Semantic — 略，离线 runner 不用。

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

### 5.3 tick()：CARLA 原始数据 → `result` dict

[mot_b2d_agent.py:561-869](AutoMoT/leaderboard/team_code/mot_b2d_agent.py#L561-L869)。

1. **GPS+IMU+SPEED → UKF** → `gps_filtered (2,), compass_filtered`
2. **LiDAR**（双路径）：
   - `lidar_to_ego_coordinate` 取 `[:, :3]`、`lidar_rot=[0,0,-90]` 转回 x 前；与上帧 `algin_lidar` 拼接 → `lidar_combined (N, 3)`
   - **主模型 PIL 路径**（**仅日志**——`__call__` 内被注释）：`generate_lidar_bev_images` → `(448, 448, 3) uint8`，RGB=(Density, Height, Intensity 后被清零)，**7 px/m**，**车头朝下**（与 LEAD `rasterize_lidar` 车头朝右**不同朝向**，调试画图易撞坑）
   - **BEV encoder 路径**（实际用）：`lidar_to_histogram_features` → `(1, 256, 256) float32 [0,1]`，`±32 m × ±32 m`，4 px/m，`lidar_split_height=0.2, use_ground_plane=False` 只用 above 层
3. **RGB**：`rgb_front (H=512, W=1024, 3) uint8` 原始；`bev_encoder_rgb (1, 3, 384, 1024) float` 路径：JPEG re-encode → BGR2RGB → `crop_array` → CHW → float **不 /255**（backbone 内部 `normalize_imagenet` 处理）
4. RoutePlanner 给 ego frame `target_point (2,), next_target_point (2,)`
5. `result['next_command'] = self.commands[-2]`（向前回退 2 个）

### 5.4 run_step() 流程

1. `tick()` → `tick_data`，塞历史 deque (`maxlen=40`)
2. `step < BUFFER_PHASE=31` 强制 `VehicleControl(0, 0, 1)` 预热 UKF
3. `_build_obs_dict` → `rgb_pil_list` (4 帧 1024×512, 间隔 5 tick=0.25 s) + `lidar_pil_list` (最新 1 帧 448×448)
4. `build_cleaned_prompt_and_modes(cat([speed, tp, ntp]))` → 固定模板 prompt + `understanding=False, reasoning=True`
5. BEV encoder: `bev_encoder(rgb_bf16, lidar_bev_bf16)` → `bev_feature (1, 1512, 8, 8) bf16`
6. `self.inferencer(image=rgb_pil_list, text=prompt, v_target_point=tp_speed, trans_feat=bev_feature, frame_idx=self.step, ...)` —— **`front/lidar` 参数实际被 `__call__` 内部注释忽略**
7. 输出 `{text, traj (1,6,2) 已 cumsum, route (1,20,2), dp_vl_feature (8,2560)}`
8. `control_pid`: 纵向 `desired_speed = ||traj[1]-traj[0]|| * 2`（0.5 s ×2 转 m/s） + 横向 `route` 经 `interpolate_waypoints` 0.1 m
9. force_move / parking_escape 覆盖 + **35 km/h 硬限速**生效点在 [run_step 末尾 line 1581-1583](AutoMoT/leaderboard/team_code/mot_b2d_agent.py#L1581-L1583)，`control_pid` 内同名实现是注释掉的兜底示例

### 5.5 慢/快路径（KV cache）— **完整范式说明见 §14，本节只列在线特有事实**

- `InterleaveInferencer.__call__` ([:1507](AutoMoT/Automot/mot/evaluation/inference.py#L1507)) 在线把 4 帧 RGB **硬编码 resize 到 (W=512, H=256)** 再喂；**`lidar / front` 参数在 `__call__` 内部被注释**，不会进 input_list ⇒ BEV 模态完全靠 `trans_feat`，PIL lidar 仅日志
- 慢路径 system prompt：`"You are a mature and professional driver."`（Qwen3VL 模板）
- 在线默认 `slow_update_interval=2`（每 2 帧刷新一次慢路径 KV cache）
- 快路径 packed sequence 101 token：`[bev(64) | tp(2) | v(1) | reasoning(8) | route(20) | waypoint(6)]`
  - **断言 `C==1512` 必须满足；`H*W==64` 的断言被注释掉**——`trans_feat (1, 1512, 10, 12)` = 120 token 也能跑，但偏离训练 (8, 8) = 64 token 网格
- 输出末 `traj_head` 出 `(1, 6, 2)` 后**额外 `.cumsum(dim=1)`** 把增量转累计位移
- reasoning hidden → `lm_head` 解码出 `"<|im_start|> verb, verb, verb<|im_end|>"`（verbs ∈ {stop, keep, accelerate, slow, ...}）

### 5.6 AutoMoT 模型权重位置（仓库相对）

- `AutoMoT/checkpoints/AutoMoT/model.safetensors`（**Qwen3-VL 4B + heads + bev_encoder 全打包**）
- `AutoMoT/checkpoints/Qwen3-VL-4B/`（tokenizer/processor）
- `BEVEncoderBackboneExtractor` 从 `model.safetensors` 里取 `bev_encoder.*` 前缀的子集自行装载。

### 5.7 Qwen3-VL Image Processing —（**结论：(1152, 384) 输入不失真**）

Qwen3-VL 复用 `Qwen2VLImageProcessor.smart_resize`，参数：`patch_size=16, merge_size=2, factor=32`。
**只要 W、H 都是 32 倍数且总像素在 `[56², 28²×1280]` 内，就不 resize 不变 aspect**。

| 输入 PIL.size | aspect 保持 | vision tokens/帧 (=H×W/1024) |
|---|---|---|
| AutoMoT 在线 (512, 256) | ✅ 2:1 | 128 |
| runner 当前 (1152, 384) | ✅ 3:1 | **432** |

**实测验证**：runner 喂 4 帧 (1152, 384) + prompt 后 `kv_lens=[1840]` ≈ 1728 vision (4×432) + 112 text，吻合。

**副作用**：vision tokens 是 AutoMoT 在线的 3.4×，显存/attention 时间放大约 3.4×–7×。Qwen3-VL backbone 冻结（预训练见过各种 aspect），**不影响慢推理质量**。需要降显存时可在 `_prepare_inference_inputs` 里 `rgb.resize((512, 256))`，但用户当前不做。

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

### 8.2 历史不匹配点 — 已修复 / 仍存在

> 背景：BEV encoder 已切换为 LEAD TransfuserBackbone（§0.5），数据预处理回到 LEAD 训练分布。

**✅ 已修复**：
- ① LiDAR 栅格 — 已切 LEAD 风格 `lead_rasterize_lidar` → `(1, 320, 384)`
- ② LiDAR PIL 仅日志用，与 `bev_lidar_tensor` 同源
- ④ tp/ntp 用 future 1.5 s/3.0 s 真值（`_extract_tp_ntp_from_future_frames`），落在 AutoMoT 训练分布 30–80 m 内
- ⑤ LiDAR 单帧（`bev_frame_count=1`），对齐 LEAD 单帧含 5 sweep 训练分布
- ⑥ z 过滤已随 LEAD 风格栅格走（含地面层）
- ⑨ theta 用 `meta["theta"]`，与 AutoMoT `compass_filtered` 在 `inverse_conversion_2d` 周期性下等价
- ⑩ 位姿严格 `pos_global + theta`，缺字段 raise
- ⑬ 删除二次 JPEG 压缩，直接 `np.array(PIL)`

**仍存在 / 未来才处理**：

| # | 问题点 | 影响 |
|---|---|---|
| ③ ⚪ | bev_encoder RGB 仍含侧视像素 + 相机物理位置差 1.85 m | 仅快推理路径，已搁置（§0.5） |
| ⑦ | traj 时间网格：runner 拿 6×0.5 s=3 s vs LEAD GT 8×0.25 s=2 s | 用 LEAD GT eval 时需插值；当前未做 |
| ⑧ | `self.commands` deque 缺失 | 仅 close-loop 才需要 |
| ⑪ | clip_len 12 帧 (3 s) > 在线 BUFFER_PHASE 31 tick (~1.55 s) | 时间更长，不算 bug |
| ⑫ | gen_context 复用：runner 每 anchor 重算 vs 在线 `slow_update_interval=2` | 当前 run_clip 单 anchor 不暴露；若多 anchor 复用应直接调 `kv_cache_inference_slow_fast_dp` |
| ⑭ | LiDAR / Radar 混合点 (`save_radar_pc_as_lidar`) | 与训练分布一致，**不是 bug**；若 `duplicate_radar_near_ego=True` 则 ego 附近密度增厚 |

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

### 8.5 精度与单位对照（**慢推理路径全部对齐**）

第一次 `kv_cache_fixed_inference` 之前的标量/向量输入与 AutoMoT 在线完全等价：
- **speed**: m/s, float32, prompt `:.2f`（runner 调同款 `build_cleaned_prompt_and_modes`）
- **target_point/ntp**: ego frame，同款 `inverse_conversion_2d(future_pos, cur_pos, cur_theta)`，prompt `:.6f`
- **theta**: 弧度。LEAD 有 `np.unwrap`（可超 [-π, π]），AutoMoT 用 `normalize_angle` ∈ [-π, π]；`inverse_conversion_2d` 用 `cos/sin` 周期性下**等价** ✅
- **pos 源**: runner 用 `pos_global`（真值，对齐 LEAD 训练默认 `use_noisy_tp=False`），与在线 `gps_filtered` 数值差极小且 future-cur 做差不变
- **`v_target_point`**: `(1, 5) float32 = [speed, tp.x, tp.y, ntp.x, ntp.y]`

**🔄 切到 LEAD 风格（与 AutoMoT 原训练分布不同，但慢推理 Qwen3-VL frozen 鲁棒，不影响）**：
- LiDAR BEV: `(1, 1, 320, 384) float32 [0, 1]` (LEAD) ← 原 `(1, 1, 256, 256) bf16` (AutoMoT)
- RGB tensor: `(1, 3, 384, 1152) float32` 三视角不 crop ← 原 `(1, 3, 384, 1024) bf16`
- `trans_feat`: `(1, 512, 10, 12)` (LEAD backbone) ← 原 `(1, 1512, 8, 8)`；与 AutoMoT 快推理 head 不兼容，但慢推理不消费

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
- **独立 VAE 模块（冻结，子目标 latent 生成路线第一阶段）**：`AutoMoT/vae_standalone/`（详见 §15）
- **子目标 latent 生成 runner（KV cache + DiT flow matching 路线）**：`AutoMoT/leaderboard/team_code/qwen3vl_dit_goalgen_runner.py` + `AutoMoT/qwen3vl_local/goalgen/`（详见 §15）

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

**作用**：把每个 LEAD run 摘录成 5 个关键帧（initial / middle×3 / final），写入 JSON。

**核心字典**（详见源文件，本机有副本）：
- `SCENARIO_LABELS` ([L34](rule_based_keyframe_filter.py#L34)) — 场景 → initial 帧 `label_text`
- `SCENARIO_CONFIG` ([L89](rule_based_keyframe_filter.py#L89)) — 场景 → `(dist_field, threshold_m, (event_A, event_B, event_C))`
- `BRAKE_ACCEL_PRIMARY = {HardBreakRoute, ControlLoss}` — 走 brake/accel 分支
- `_ALL_DIST_FIELDS` — 8 个 `dist_to_*` 字段

**输入**：`results.json`（duration/status/infractions）+ `metas/*.pkl`（speed/accel/brake + dist_to_*，lzma 压缩）+ `bboxes/*.pkl`（距离回退）+ `rgb/*.jpg`（最后兜底）。

**事件挑选优先级**（[`pick_middle_events` L783](rule_based_keyframe_filter.py#L783)）：
1. CrossingBicycleFlow 专用（[`_pick_bicycle_flow_events`](rule_based_keyframe_filter.py#L693)）
2. Cut-in / Merge 专用（[`_pick_cutin_events`](rule_based_keyframe_filter.py#L740)）
3. Brake/accel 主导（[`_pick_brake_accel_events`](rule_based_keyframe_filter.py#L655)）
4. 通用距离规则（[`_pick_distance_events`](rule_based_keyframe_filter.py#L599)）：A = 距离首次 `< thresh` 且开始减速；B = 该段速度最低帧；C = B 后 ≥2 帧 `speed>2 ∧ accel>0.1` 的恢复点
5. RGB fallback：相邻 JPG 文件大小差找 3 峰值帧

**信号源 confidence**：metas≈0.88 / bboxes≈0.7 / rgb_fallback≈0.5（写入 `signal_source` 字段）。

**工程细节**：pickle 是 lzma xz 压缩；`enforce_event_order` 保证 3 中间事件严格递增（min_gap=2）；`t = frame * seconds_per_frame`，典型 ≈0.25 s/帧；远程默认 root `/home/cruser1/lda/lead/cache_ln/data`。

### 13.3 `keyframes_all_scenarios.json` 字段说明

**顶层**：`{dataset_root, scenarios[41], num_runs=7326, runs[], failed_runs=[], num_failed_runs=0}`。
（41 个场景，比 `SCENARIO_CONFIG` 的 43 项少 2，数据集没采全）

**每个 run 条目字段**：

| 字段 | 含义 |
|---|---|
| `scenario` | CARLA 场景名 |
| `run_id` | run 目录名，如 `Town03_Rep0_route_001783_...` |
| `route_id` / `status` / `num_infractions` | 来自 `results.json` |
| `signal_source` | `metas` / `bboxes` / `rgb_fallback` |
| `rule_confidence` | 3 中间事件 confidence 均值 |
| `initial` | `{event:"initial", frame:0, t:0.0, label_text, confidence:1.0}` ←  `label_text` 来自 `SCENARIO_LABELS` |
| `middle[3]` | `{event, frame, t, confidence}` — event 与 `vlm_prompt_pipeline.SCENARIO_EVENT_SEQUENCES` 对应 |
| `final` | `{event:"final", frame=total-1, t, final_success, confidence:1.0/0.8}` |
| `diagnostics` | `{total_frames, duration_game, seconds_per_frame}` |

> ⚠ **`frame` 是 LEAD 原始 rgb 序列的下标，不是 AutoMoT 4 帧 clip 内的下标**。runner 里用它需要先把 `anchor_t` 换算到原始帧号（§5 已抽帧）。

### 13.4 `vlm_prompt_pipeline.py` 解读

**作用**：把"VLM 看图 + memory → 输出 STATUS/SUBGOAL/ANALYSIS"封装成框架无关小模块。`run_pipeline_step(memory, image, vlm_fn)` 收一个 `vlm_fn(system, user)->str` callable，下游接任意 VLM。

**核心组件**（详见源文件）：
- `SCENARIO_LABELS` ([L38](vlm_prompt_pipeline.py#L38)) — 与 filter 里那份**完全一致**（独立维护）
- `SCENARIO_EVENT_SEQUENCES` ([L87](vlm_prompt_pipeline.py#L87)) — 场景 → 3 中间事件元组；**与 filter 的 `SCENARIO_CONFIG` 第 3 项一一对应**，两文件唯一的"事件命名契约"
- `EVENT_DESCRIPTIONS` ([L136](vlm_prompt_pipeline.py#L136)) — 事件名 → 人类可读英文说明
- `DrivingMemory` ([L228](vlm_prompt_pipeline.py#L228)) dataclass：`scenario, scenario_label, event_sequence=("initial", mid_a, mid_b, mid_c, "final"), status, subgoal, completed_events`
- `build_system_prompt / build_memory_block / build_user_prompt / parse_vlm_output / update_memory`

**`update_memory` 关键约束**：只允许沿 `event_sequence` **前进或保持，不许回退**；非法事件名 strict=True 抛错否则忽略；subgoal 由 `final_status` 推导而非信 VLM。

**system prompt 约束** ([L317](vlm_prompt_pipeline.py#L317))：只输出 ANALYSIS/STATUS/SUBGOAL 三行；STATUS/SUBGOAL 必须是序列内事件；SUBGOAL = STATUS 的下一个事件（最后一个中间事件时 = `final`）。

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

### 14.5 Prefill + Decode 是 Transformer autoregressive 推理的本质架构

> 解答"能不能不经过 `kv_cache_fixed_inference` 直接有端口生成？" —— **物理上做不到**。
> 所有"看起来一步生成"的端口（Qwen 原版 `.chat()`、HF `model.generate()`、AutoMoT
> `qwen3vl_template_inference`、llama.cpp、vLLM……）内部都是 **1 次 prefill + K 次 decode**。
> 这是 Transformer attention 算法的根本性约束，不是 AutoMoT 的实现选择。

#### 14.5.1 两阶段对比

| 阶段 | 输入 | 计算量 | KV cache | 并行度 | 计算特性 |
|---|---|---|---|---|---|
| **Prefill** | N 个 token 一起进 | N×N attention 一次算完 | 写入 N 个位置 | 高（所有 token 并行 attend） | compute-bound |
| **Decode** | 每步 1 个新 token | 1×(N+k) attention | 每步 append 1 个 | 低（必须串行，下一个依赖前一个） | memory-bound |

#### 14.5.2 为什么不能合并

- 第 N+1 个 token 要等模型把前 N 个 token 的 hidden 算完才知道——它是 `argmax(lm_head(hidden[-1]))` 出来的
- 第 N+2 个 token 又依赖 N+1 的 hidden ——必须先把 N+1 喂进去再算一次
- 这是自回归的本质 `p(x_{t+1} | x_1, ..., x_t)`，下一个永远依赖前一个

⇒ 生成 K 个 token **最少**要 1 + K 次 forward。

> 例外：non-autoregressive generation（一次性预测所有 token）理论存在但效果差，工业上几乎不用。
> speculative decoding / parallel decoding 是优化，**仍然遵循 prefill + decode 框架**，只是 decode 阶段一次跑多个 draft token。

#### 14.5.3 AutoMoT 为什么把 prefill 显式暴露成 `kv_cache_fixed_inference`

**为了跨帧复用 prefill 结果，省掉大图像（vision token 1728 个）的重复编码开销。**

看 [`kv_cache_inference_slow_fast` L1306](AutoMoT/Automot/mot/evaluation/inference.py#L1306)：

```python
# 慢路径每 slow_update_interval=2 帧才刷一次（贵但低频）
if frame_idx % slow_update_interval == 0:
    self._cached_gen_context = self.kv_cache_fixed_inference(slow_input_lists)

# 快路径每帧都跑，复用上面那份 cache（便宜但高频）
gen_text, gen_traj = self.based_kv_cache_context_fast_qwen3vl(
    fast_input_lists, self._cached_gen_context, ...
)
```

这是工业级 LLM serving 的标准优化（disaggregated prefill / decode），AutoMoT 把它显式暴露
在 Python 层方便控制。

### 14.6 AutoMoT 里的 generative 端口 — 一步到位看着像，内部仍两步

#### 14.6.1 端口对照表

| 端口 | 包装层数 | 内部展开 | 适用场景 |
|---|---|---|---|
| [`qwen3vl_template_inference`](AutoMoT/Automot/mot/evaluation/inference.py#L1409) | **最高 — 一行返回文本** | `update_context_qwen3vl + gen_text` | 单帧 demo，不复用 cache |
| [`slow_reasoning`](AutoMoT/Automot/mot/evaluation/inference.py#L1446) | 高 | 同上 | 在线 agent 早期版本 |
| [`kv_cache_fixed_inference`](AutoMoT/Automot/mot/evaluation/inference.py#L1233) **+** [`gen_text`](AutoMoT/Automot/mot/evaluation/inference.py#L820) | 中 — 手控两步 | 你手动拆 | **想跨帧复用 cache，或同时喂快推理 head** |
| `kv_cache_fixed_inference` **+** `based_kv_cache_context_fast_qwen3vl_dp` | 中 — 跳过 decode | — | 走范式 B（轨迹预测，不要文字） |

#### 14.6.2 一步端口内部展开（证明确实两步）

[`qwen3vl_template_inference`](AutoMoT/Automot/mot/evaluation/inference.py#L1409) 核心：

```python
gen_context = self.init_gen_context()                                  # 空 cache
gen_context = self.update_context_qwen3vl(                              # ★ prefill 阶段
    user_prompt, instruction_prompt, image_list, gen_context
)
gen_text = self.gen_text(gen_context, max_length=max_think_token_n)    # ★ decode 阶段
return [gen_text]
```

`update_context_qwen3vl` ≈ `update_kv_cache_context_qwen3vl`（同族），底层都走
`prepare_kv_cache → forward_cache_update_generation`。两步本质等价。

#### 14.6.3 `gen_text` 是真·autoregressive

[`gen_text`](AutoMoT/Automot/mot/evaluation/inference.py#L820) 直接消费 `kv_cache_fixed_inference` 返回的 `gen_context`，
内部调 [`model.generate_text`](AutoMoT/Automot/mot/modeling/automot/automot.py#L3037) 跑教科书循环：

```python
step = 0
curr_tokens = packed_start_tokens
while step < max_length:
    output = language_model.forward_inference(
        ..., past_key_values=past_key_values,
        is_causal=True,                # ← 与 fast 路径 False 正好相反
        update_past_key_values=True,   # ← 把新 token 的 K/V 写回 cache
    )
    pred_logits = self.language_model.lm_head(output.packed_query_sequence)
    curr_tokens = argmax(pred_logits) if not do_sample else multinomial(softmax(...))
    key_values_lens += 1
    if curr_tokens[0] == end_token_id: break       # 遇到 <|im_end|> 停
    step += 1
```

#### 14.6.4 三个函数对照（解答"长短不定 vs 长短一致"）

| | `kv_cache_fixed_inference` | `gen_text` | `based_kv_cache_context_fast_qwen3vl` |
|---|---|---|---|
| 是否有 while 循环 | ❌ 一次 forward | ✅ `while step < max_length` | ❌ 一次 forward |
| `is_causal` | True（decoder 内部） | **True** | **False** |
| `update_past_key_values` | True | **True**（每生成一 token 扩 cache） | **False** |
| 输出 | KV cache 张量 | **str（变长，到 EOS 或 max_length）** | hidden state 张量（固定 101 长度） |
| 长度规律 | cache K/V 长度 = **输入 token 数**（"长短一致"指这个） | 1..max_length，**遇 EOS 提前停 → 不定长** | 固定 = bev+tp+v+reasoning+route+waypoint |

### 14.7 这两种范式对改 `prompt_cleaned` 的影响（关键结论）

> ⚠⚠ **重要订正（2026-05-27 smoke test 实测）**：本节早期版本提到的
> "AutoMoT 自带 `gen_text` 端口 → 直接出三行 STATUS/SUBGOAL" **实测无效**。
> AutoMoT ckpt 跑 `gen_text` 会立即吐 EOS → **空字符串**。详见 §14.9。
>
> 想真正走范式 A 出文字 → **只能用本地 `AutoMoT/checkpoints/Qwen3-VL-4B`**
>（`vlm_paradigm_a_runner.py --backend qwen` 的 baseline runner），
> **不能用 AutoMoT ckpt**。详细对照实验见
> [`AutoMoT/leaderboard/team_code/vlm_paradigm_a_runner.py`](AutoMoT/leaderboard/team_code/vlm_paradigm_a_runner.py)
> 的 `--backend automot|qwen|both` smoke test。
>
> **加载规则**：`qwen` backend 不联网、不拉官方 hub，必须 `local_files_only=True`
> 读取 `AutoMoT/checkpoints/Qwen3-VL-4B`。AutoMoT 现有
> `InterleaveInferencer` / `qwen3vl_template_inference` 绑定的是 AutoMoT 自定义
> MoT 架构和 `gen_text` 起始 token 逻辑，不能直接拿来支撑 standalone Qwen 的完整
> 自由文本生成；本地 Qwen 的文字输出走 HF 标准 `AutoProcessor.apply_chat_template`
> + `past_key_values` 显式 prefill/decode（不是 AutoMoT `NaiveCache/gen_text`）。

1. **直接把 `vlm_prompt_pipeline.build_user_prompt(memory, ...)` 塞进 `prompt_cleaned`，但只跑 `kv_cache_fixed_inference`：**
   - 可以塞，但 VLM **不会按 ANALYSIS/STATUS/SUBGOAL 三行答**
   - 因为 `kv_cache_fixed_inference` 不解码（只 prefill），多塞的内容只是变长 KV cache
   - 效果：reasoning_query / action_query cross-attend 时多看了一段 memory，**间接**影响下游 `(stop/keep, traj)` 输出。**不会有可读文字。**

2. **~~AutoMoT 加一行 `gen_text` 即可~~ ⚠ 实测证伪；本地 Qwen baseline 也不能复用 AutoMoT `gen_text`：**

   - AutoMoT `gen_text` 吃的是 AutoMoT 自定义 MoT 模型的 `NaiveCache`、
     `new_token_ids` 和 `prepare_start_tokens(...)`，不是通用 Qwen 文本生成入口。
   - `AutoMoT/checkpoints/AutoMoT` 跑 `kv_cache_fixed_inference + gen_text` 会立即 EOS，
     `raw_vlm_text == ""`。
   - `AutoMoT/checkpoints/Qwen3-VL-4B` 要出三行文本，走
     `AutoProcessor.apply_chat_template` → `model(**inputs, use_cache=True)` 显式 prefill
     → 每步喂上一个 token + `past_key_values` decode。这个逻辑在
     `BaselineQwen3VLRunner._decode_with_explicit_cache(...)`，不是
     `InterleaveInferencer.gen_text(...)`。

3. **~~"模型按三行答的能力来自 Qwen3-VL 原始指令跟随"这条说法实测不成立。~~** 实测发现 AutoMoT 的 lm_head autoregressive 路径**完全丧失**指令跟随能力——即使训练脚本宣称"backbone 冻结、lm_head 原生未微调"，加载日志里 `Loaded weights: 0 missing` 表明**模型的每一个参数（含 decoder 层、lm_head）都来自 AutoMoT 自家的 `model.safetensors`**，不是从 Qwen3-VL-4B 基座加载的。叠加新加的驾驶 special tokens + `prepare_start_tokens` 用的也是 driving-task 起始 token，SFT 训练分布把 `gen_text` 路径完全特化成"短驾驶 token + 立即 EOS"。详见 §14.9。

4. **范式 A 和范式 B 共存的红利仍然有效，但配置受 ckpt 约束**：
   - 范式 A 路径：文字输出 —— **必须**用本地 `AutoMoT/checkpoints/Qwen3-VL-4B`，走 HF 标准 `past_key_values` 显式 prefill/decode
   - 范式 B 路径：`kv_cache_fixed_inference + based_kv_cache_context_fast_qwen3vl_dp` —— **必须**用 AutoMoT ckpt（因为下游 reasoning_head / waypoints_head 在 baseline Qwen 里根本不存在）
   - 同一份 `gen_context` 喂两条路径在**架构上**可行，**业务上**两条路径吃的是不同 ckpt / 不同加载链路，需要在外面起两个 runner 实例分别跑（见 `vlm_paradigm_a_runner.py` 的 `--backend both`）

### 14.8 一句话记忆法

- **范式 A** = LLM 当**对话模型**用。自回归 decode 出文本，正则解析。格式靠 prompt 软约束。**ckpt 必须是本地 `AutoMoT/checkpoints/Qwen3-VL-4B`**。
- **范式 B** = LLM 当**特征提取器 + cross-attention 上下文池**用。一次 forward，外接 head 解码。格式靠 head 的 output shape 硬约束。**ckpt 必须是 AutoMoT**。
- **AutoMoT 自带 `gen_text` 端口架构上能跑，业务上跑不出来**——SFT 把 lm_head 训成只会吐 stop/keep + EOS。
- **"一步到位"端口（`qwen3vl_template_inference` 等）内部仍是 prefill + decode**——这是 Transformer autoregressive 的物理本质，绕不开。
- `prompt_cleaned` 在范式 B 是 "soft prompt 上下文"，在范式 A 才是真正被回答的"问题"。


### 14.9 smoke test 实证：AutoMoT ckpt **不能**走范式 A

**实验脚本**：[`AutoMoT/leaderboard/team_code/vlm_paradigm_a_runner.py`](AutoMoT/leaderboard/team_code/vlm_paradigm_a_runner.py)

```bash
python leaderboard/team_code/vlm_paradigm_a_runner.py --backend both
```

输入完全一致（同一组合成 RGB 图 + 同一个 system/user prompt + 同一份 DrivingMemory），
输出对比：

| backend | raw VLM text | parsed | 结论 |
|---|---|---|---|
| `automot`（AutoMoT ckpt + `kv_cache_fixed_inference + gen_text`）| **`""`**（长度 0）| `{status: None, subgoal: None, analysis: None}` | ❌ 立即 EOS |
| `qwen`（本地 `AutoMoT/checkpoints/Qwen3-VL-4B` + HF `past_key_values` 显式 prefill/decode）| 完整三行 + 正确语义（"The image shows a static, colorful banner ... status has not changed"）| `{status: "initial", subgoal: "slow_traffic_detect", analysis: "..."}`| ✅ 指令跟随 + 视觉理解都正常 |

#### 14.9.1 根因（三层修改叠加）

| 层 | AutoMoT vs 本地 Qwen3-VL-4B baseline 的差异 | 对 `gen_text` 的影响 |
|---|---|---|
| **架构** | `layer_module="Qwen3VLMoTDecoderLayer"`；新增 `reasoning_queries`/`action_queries`/`route_queries`/`waypoint_queries`/`reasoning_head`/`waypoints_head`/`anchor_head`/`bev_encoder_proj`/各种 projector | decoder 层的 MoT 通路对 reasoning_query 等位置走另一条 FFN/attention，间接改变 lm_head 输入分布 |
| **权重** | `Loaded weights: 0 missing, 1146 unexpected` —— **整个模型**（含 decoder + lm_head）的参数都来自 AutoMoT 自家 `model.safetensors`，**不是**从 Qwen3-VL-4B base 加载。doc 早期"backbone 冻结、lm_head 原生未微调"的说法**只描述训练时的梯度策略**，不代表 ckpt 里的数值跟基座完全一致 | lm_head 的输出分布已被 SFT 拉到驾驶 token 上 |
| **Tokenizer** | [`add_special_tokens`](AutoMoT/Automot/data/reasoning/data_utils.py) 加了驾驶专用 special token；`gen_text` 内部 `prepare_start_tokens(kv_lens, ropes, self.new_token_ids)` 喂的起始 token 是这些新加的 driving-task token，不是普通的 `<\|im_start\|>assistant\n` | 模型一拿到驾驶起始 token，SFT 训练分布告诉它"下一步该立即结束"，第一个采样到的就是 `<\|im_end\|>` |

#### 14.9.2 三种"空输出"失败模式必须分清

| 失败模式 | 是不是 AutoMoT 模型问题 | runner 当前是否中招 | 怎么救 |
|---|---|---|---|
| **范式 A `gen_text` 立即 EOS → 空字符串** | ✅ 是模型问题（SFT 把 lm_head 训成只会出短驾驶 token） | ❌ runner 不调用 `gen_text`，**不会中招** | 想要文字 → 切本地 Qwen3-VL（baseline runner）|
| **范式 B `fast_qwen3vl_dp` 输出空** | ❌ **物理上不可能** —— `is_causal=False, update_past_key_values=False`，一次 forward 出固定 shape hidden state，没有 EOS 概念 | ❌ **不会中招** | —— |
| **范式 B 整条路被 `enable_fast_inference=False` 跳过 → traj/route 全 None** | ❌ 不是模型问题 —— LEAD trans_feat shape `(1, 512, 10, 12)` 与 AutoMoT `bev_encoder_proj` 期望 `(1, 1512, 8, 8)` 不兼容 | ✅ **当前 runner 默认就是这个状态** | 按 §12 重设计 BEV → projector → query 链路 |

#### 14.9.3 对 `mot_lead_offline_runner.py` 的影响

**不需要改 runner**。理由：

- runner [设计上](AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py#L1092-L1118) 就**从不调用 `gen_text`**，所以 §14.9.2 第 1 行的失败模式压根不会发生
- runner 的 `prompt_cleaned`（"Your current and next target point is ..."）对齐 SFT 训练分布，没问题
- runner 当前看着"输出全 None" 是因为 §14.9.2 第 3 行（BEV shape 不兼容），不是模型问题，换千问也救不了（千问没有 waypoints_head）

#### 14.9.4 实操推荐

| 想要什么 | 用什么 ckpt | 走哪条路 |
|---|---|---|
| 轨迹 / 动作（waypoints, stop/keep）| AutoMoT | 范式 B（要先修 BEV shape，见 §12）|
| ANALYSIS / STATUS / SUBGOAL 文字 | 本地 `AutoMoT/checkpoints/Qwen3-VL-4B` | 范式 A（HF `past_key_values` 显式 prefill/decode，不要套 AutoMoT `InterleaveInferencer`）|
| 两者都要 | 起两个 inferencer 并行跑 | `vlm_paradigm_a_runner.py --backend both` |

**绝对不要**：在 AutoMoT ckpt 上调 `gen_text` 期待出三行文本 —— 物理上能跑通（没报错），业务上是空字符串。

### 14.10 Qwen3-VL-4B-Instruct standalone 范式 A runner

新增入口 [`qwen3vl_instruct_paradigm_a_runner.py`](AutoMoT/leaderboard/team_code/qwen3vl_instruct_paradigm_a_runner.py)，只跑本地 `AutoMoT/checkpoints/Qwen3-VL-4B-Instruct`，不再保留 `Qwen3.5-4B` / `both` 分支；`vlm_paradigm_a_runner.py` 仍负责 AutoMoT vs Qwen 的旧对照实验。

模型来源固定为 HuggingFace `repo_id=Qwen/Qwen3-VL-4B-Instruct`，用户远程环境已下载到 `AutoMoT/checkpoints/Qwen3-VL-4B-Instruct`。后续运行必须只读本地目录，不允许联网下载。

本地可魔改代码放在 [`AutoMoT/qwen3vl_local/`](AutoMoT/qwen3vl_local/)：

- `prompt_pipeline.py`：从 `vlm_paradigm_a_runner.py` 的 `vlm_prompt_pipeline.py` 迁移块完整同步，包含 `EVENT_DESCRIPTIONS`、`build_memory_block`、system/user prompt、三行输出解析、memory update。
- `image_io.py`：合成 RGB 与 LEAD route `rgb/*.jpg` 读取。
- `engine.py`：显式拆分 `build_messages -> apply_chat_template -> prepare_inputs -> prefill -> decode`，decode 逐 token 更新 `past_key_values`。
- `cache_utils.py`：KV cache shape/dtype/device summary；可选 `--save-cache` 把 prefill/final cache 用 `torch.save` 落盘。

运行边界：

- 只允许本地 checkpoint，`local_files_only=True`，并设置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，禁止下载。
- `Qwen3-VL-4B-Instruct` 与原 `AutoMoT/checkpoints/Qwen3-VL-4B` 不能假设权重一致：前者是 instruct/post-training 版，后者是原本 baseline Qwen 目录；tokenizer/processor 可能同源，但模型权重不是同一个语义对象。
- 每次运行都会保存 prompt、raw text、parsed、memory、`generation_trace.json`。即使不加 `--save-cache`，trace 也包含 KV cache 结构摘要；加 `--save-cache` 会额外保存 `.pt` tensor 文件，体积可能很大。

示例：

```bash
python leaderboard/team_code/qwen3vl_instruct_paradigm_a_runner.py
python leaderboard/team_code/qwen3vl_instruct_paradigm_a_runner.py --save-cache
python leaderboard/team_code/qwen3vl_instruct_paradigm_a_runner.py --route-dir <LEAD route> --anchor 12
```

CLI 参数尽量对齐 `vlm_paradigm_a_runner.py`：`--route-dir` 默认同样指向
`/datashare/IOL4SGH/data/data/Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46`，
可传空字符串退回合成图。

---

## 15. VAE Standalone + 基于 KV cache 的子目标 latent 生成路线（新路线）

> 这一节描述 §14 范式 A 的姊妹路线：不再让 Qwen3-VL "说话"，而是把
> teacher-forced 的场景状态作为 prompt 让它只跑 prefill，拿到 KV cache 当作
> "语言/场景上下文池"，再驱动一个 **flow-matching DiT** 在 VAE latent 空间
> 上生成"子目标对应关键帧"的预测 latent。监督真值来自 keyframes_all_scenarios.json
> 指定的关键帧 RGB 经同一冻结 VAE 编码。

### 15.1 VAE Standalone 模块（冻结的图像编码 / 解码器）

目录：[`AutoMoT/vae_standalone/`](AutoMoT/vae_standalone/)。
白名单内只有 [`train_patch_unpatch.py`](AutoMoT/vae_standalone/train_patch_unpatch.py)；
其它 VAE 原始文件仍只读。

来源：Vista 项目的 first-stage VAE 单独抽出来，权重 372 keys。

文件：

- 入口脚本 [`vae_reconstruct.py`](AutoMoT/vae_standalone/vae_reconstruct.py)：encode → decode → 输出 MSE / PSNR / L1。
- 训练脚本 [`train_patch_unpatch.py`](AutoMoT/vae_standalone/train_patch_unpatch.py)：冻结 VAE，端到端训练 DiT 同款 `Patchify` / `Unpatchify` 做图像重建；在 `AutoMoT/` 下运行，默认用 `nvidia-smi` 自动挑空闲 GPU。
- 配置 [`config/vae_only.yaml`](AutoMoT/vae_standalone/config/vae_only.yaml)：`first_stage_config` 通过 `instantiate_from_config` 实例化 `vwm.models.autoencoder.AutoencodingEngine`。
- 权重 `weights/vae_only.safetensors`：来自 `extract_vae_weights.py` 从 Vista 完整 ckpt 抽取 `first_stage_model.*` 前缀。
- 子模块 `vwm/`：必要的最小依赖（`models.autoencoder`、`modules.latentmodules.model.Encoder`、`modules.autoencoding.temporal_ae.VideoDecoder`、regularizer、distributions、attention、util）。

关键事实（写代码前必看）：

| 项 | 值 |
|---|---|
| 输入归一化 | `mean = std = 0.5`，张量范围 `[-1, 1]`（不是 `[0,1]` 也不是 ImageNet 均值） |
| 输入尺寸约束 | H、W 必须是 **64 的倍数** |
| Encoder | 普通 2D `vwm.modules.latentmodules.model.Encoder`，`z_channels=4`，`ch=128`，`ch_mult=[1,2,4,4]`，下采 8 倍 |
| Decoder | `vwm.modules.autoencoding.temporal_ae.VideoDecoder`（temporal kernel size=3），单帧/batch=1 时取 `overlap=0` |
| 缩放因子 | `scale_factor = 0.18215`；encode 后 `z = z * scale_factor`，decode 前 `z = z / scale_factor` |
| 自动混精度 | `disable_first_stage_autocast=True`，即默认**关闭** autocast |
| 单轮 batch 上限 | `en_and_decode_n_samples_a_time = 14`（推理时按这个切片，避免大张量爆显存） |
| 不依赖 | 不需要 Vista 主干 UNet / conditioner / sample.py / sample_utils.py |

形状映射示例（路线里 LEAD 三视角直接走 1152×384）：

| 输入 `[B, 3, H, W]` | latent `[B, 4, H/8, W/8]` | 备注 |
|---|---|---|
| `[B, 3, 384, 1152]` | `[B, 4, 48, 144]` = **6912** token / batch | LEAD stitched 三视角，**当前路线默认** |
| `[B, 3, 576, 1024]` | `[B, 4, 72, 128]` | 1024×576 vista 测试样张 |
| `[B, 3, 256, 256]` | `[B, 4, 32, 32]` | 通用 256 方图 |

使用模式（新路线代码优先复用 `qwen3vl_local/goalgen/vae.py`；除白名单里的 `train_patch_unpatch.py` 外，不要再去碰 vae_standalone 原始文件）：

1. `python -c` 不可行（vae_standalone 依赖路径相对自身）；新路线把 `AutoMoT/vae_standalone` 加进 `sys.path` 后再 `from vwm.util import instantiate_from_config` / `from safetensors.torch import load_file` 走和 `vae_reconstruct.py` 同样的加载流程。
2. 加载后 `model.eval()` + 所有参数 `requires_grad_(False)`，全程**冻结**。
3. encode 用 `model.encode(x_in)`，注意 x_in 必须是 [-1,1] 归一化后的张量；不要漏 `* scale_factor`。
4. decode 仅用于可视化（推理后想把预测 latent 还原成 RGB），训练阶段不需要。

### 15.2 路线总览：teacher-forced Qwen prefill → DiT-MoT → flow matching

```
[hist + current RGB]                       [subgoal keyframe RGB]
        │                                            │
        ▼                                            ▼
┌───────────────────────────┐               ┌──────────────────┐
│ Qwen3-VL-Instruct (frozen)│               │  VAE encode      │
│ teacher-forced prompt:    │               │  (frozen)        │
│  告诉它 STATUS/SUBGOAL    │               └──────────────────┘
│  只 prefill,不 decode     │                        │
│  收 36 层 past_key_values │                  z1 (target latent)
└───────────────────────────┘
        │                                    z_t = (1-t)·z0 + t·z1
        │  默认 select_last:Qwen 36 层 / 3 段              ▲
        │  每段取最后一层 → 12 段 token-level (K, V)        │
        │  (concat_layers 模式留作 ablation)                │
        ▼                                                 │
┌──────────────────────────────────────────────────────────────┐
│ DiT (trainable, 12 layers, MoT joint attention)              │
│ vision token = concat[                                       │
│   proj(z_t),                       # noisy target latent     │
│   proj(VAE(history_frame_1..F)),   # 所有历史帧 latent       │
│ ] + type / frame / 位置编码 + timestep                        │
│                                                              │
│ 每层 block:                                                  │
│   Q = vision_token (only updated)                            │
│   K = concat[ vision_K_proj(q), language_K_seg[i] ]          │
│   V = concat[ vision_V_proj(q), language_V_seg[i] ]          │
│   language K/V 来自 Qwen, 全程冻结                            │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
   v_pred (velocity on subgoal latent)
        │
        ▼  flow matching loss
   L = ‖ v_pred − (z1 − z0) ‖²
```

关键设计点（与用户达成的决定）：

1. **融合方式**：MoT joint-attention。DiT block 不做单独 cross-attn；vision Q 与 (vision K/V + language K/V) 一起做一次 attention。
2. **层映射**：E3 分段。Qwen 共 36 层 → 12 段（每段 3 层），DiT 也设 12 层。默认 `select_last`：每段取该 3 层组的**最后一层** token-level K/V，shape `[B, 8, S, 128]`（省显存、loss 等价上不损失主要信息）；`concat_layers` 把 3 层 K/V 沿 token 轴 concat 留作 ablation；`mean` 是旧版层平均，已弃用。第 i 层 DiT block 使用第 i 段 KV。
3. **视觉锚点**：把所有历史帧（builder 默认 4 帧）的 VAE latent 分别 patchify 后 concat 到 vision token 序列；每帧用独立的 frame embedding 区分，最后一帧 = 当前 anchor。
4. **目录组织**：新模块全部放进 `AutoMoT/qwen3vl_local/goalgen/` 子包，CLI 入口为 `AutoMoT/leaderboard/team_code/qwen3vl_dit_goalgen_runner.py`。

### 15.3 文件分工（`AutoMoT/qwen3vl_local/goalgen/`）

| 文件 | 职责 |
|---|---|
| `__init__.py` | 子包导出索引 |
| `vae.py` | 把 `AutoMoT/vae_standalone` 临时加进 sys.path，封装 `FrozenVAE.load(cfg_path, weights_path)`，提供 `encode(pil_list)` / `decode(z)`；输入归一化、shape 校验、scale_factor 处理全部内聚在这里；加载即冻结 |
| `prompt.py` | teacher-forced prompt 模板。结构同 §14.10 但 system 删掉"输出 ANALYSIS/STATUS/SUBGOAL"那段，user 改为"当前 STATUS=X（描述）/ 子任务 SUBGOAL=Y（描述）"；输出函数 `build_teacher_system_prompt` + `build_teacher_user_prompt(memory)` |
| `qwen_kv.py` | 复用现有 `LocalQwen3VLInstructEngine` 的 prefill；`teacher_forced_prefill(...)` 返回 `PrefillResult`（含 `pooled_kv` 字段名是历史遗留 + 维度元信息）；`segment_kv_for_dit(past_key_values, num_segments=12, mode="select_last")` 把 36 层切成 12 段，默认每段取最后一层 token-level K/V，可选 `concat_layers`（3 层 token 维 concat）或 `mean`（旧版层平均）；`pool_kv_for_dit` 是同义别名；输出 K/V 全部 detach |
| `keyframes.py` | 读 `keyframes_all_scenarios.json`，按 `(scenario, run_id, subgoal_event)` 查 `frame_idx`；`load_keyframe_rgb(route_dir, frame_idx)` 返回 stitched 三视角 RGB PIL |
| `dit.py` | DiT-MoT 主体。`DiTMoTBlock`：vision Q + joint-attn(K=cat[vision_K, lang_K], V=cat[vision_V, lang_V]) + MLP；`DiTMoT`：patchify vision latent → token + AdaLN-Zero (timestep) → 12 个 block → unpatchify 回 latent shape；语言 K/V 由外部传入；附带 `language_kv_input_dim_from_pooled` 辅助函数 |
| `flow.py` | 1) 训练采样：`t ~ U[0,1]`，`z0 ~ N(0,I)`，`z_t=(1-t)z0+t z1`，`v_target = z1 - z0`；2) `flow_matching_loss(v_pred, v_target)`；3) 推理 Euler 积分 `euler_sample(velocity_fn, ...)`：`z = z + dt * v_pred` 从 t=0 到 t=1 |
| `build_dataset_v1.py` | 扫 `keyframes_all_scenarios.json`，按 `Completed/Perfect` 筛 run，把每个状态段展开成 (anchor, status, subgoal, target_frame, history/current/target RGB 路径) jsonl；按 `status->subgoal` 桶 stratified 抽样；按 run_id 8:2 划 train/val；每个 route 用 file-list 缓存避免 N 次 stat |
| `train_v1.py` | DDP / 单卡训练入口。`engine.load() + freeze_module()` 显式冻 Qwen；`FrozenVAE.load()` 内部已冻 VAE；`_probe_language_kv_dim()` 用首条样本推 language_kv_input_dim（避免硬编码）；AdamW 只优化 DiT 参数；DDP 模式 grad-accum 期间走 `dit.no_sync()` 减少 all-reduce；每 `--save-steps` 落盘 `goalgen_v1.pt` + `latest.pt` |
| `train_v1.sh` | check / single / ddp 三模式；按 `nvidia-smi` 自动挑空闲 GPU、自动选空闲 MASTER_PORT；环境变量 `LANGUAGE_KV_INPUT_DIM=auto` 默认走 trainer 内部 probe，传整数可跳过 |
| `GOALGEN_V1_PLAN.md` | v1 路线设计、形状默认表、参数量预估、显存预估、风险表、v1/v2 边界 |
| `GOALGEN_V1_RUN.md` | 0 检查输入 / 1 build dataset / 2 train check→single→ddp / 3 forward smoke / 4 troubleshooting / 5 形状默认表 / 6 显存预期 |

CLI 入口 `qwen3vl_dit_goalgen_runner.py`：

- 复用 `prepare_images` / `load_lead_rgb_clip` / `DrivingMemory` / `auto_detect_scenario_from_route`。
- 多一个 `--subgoal` 参数（默认从 `DrivingMemory.from_scenario(scenario).subgoal` 推；推理调试可显式覆盖）。
- 多一个 `--run-id` / 自动识别 run，方便 `keyframes.py` 查目标帧；找不到目标帧时打印警告并跳过 loss，仅做 forward。
- 单次执行：teacher-forced prefill → 分段 KV（默认 `select_last`）→ 准备 z0 / 历史多帧 latent z_history / 真值 latent z1 → DiT forward → loss。
- 未显式设置 `CUDA_VISIBLE_DEVICES` 或 `--device cuda:N` 时，默认自动挑 1 张空闲 GPU。
- 输出 step.json：保存 prompt、Qwen KV summary、分段后段 shape、DiT 输入 / 输出 latent shape、loss、目标帧路径。

### 15.4 与现有 §14.10 的关系

- **完全独立的入口**，不修改 [`qwen3vl_instruct_paradigm_a_runner.py`](AutoMoT/leaderboard/team_code/qwen3vl_instruct_paradigm_a_runner.py) 或 `qwen3vl_local/` 现有模块。
- 复用现有 `LocalQwen3VLInstructEngine` 的 `load`/`prepare_inputs`/`prefill` 方法（teacher-forced 路线只缺 decode 那段，不需要分叉 engine 实现）。
- 新模块都在 `goalgen/` 子包里，旧 runner 不会被影响。

---

## 16. GPU 选址统一规则

SFT v1/v2、GoalGen、VAE patch/unpatch 的训练、eval、probe、teacher 入口默认都自动寻找空闲 GPU。文档示例不要默认写 `CUDA_VISIBLE_DEVICES=0`；手动 CUDA mask 只作为用户显式覆盖。

- 单进程入口：默认调用 `nvidia-smi`，按 `memory.used`、`utilization.gpu` 从低到高挑 1 张卡。
- `torchrun --nproc_per_node=N`：默认按同一规则挑 N 张卡，并按 `LOCAL_RANK` pin 到对应可见卡。
- 已有 `CUDA_VISIBLE_DEVICES`：尊重外部 mask；训练 launcher 中显式 `DDP_GPU_COUNT=N` 表示重新自动挑 N 张卡。
- 自动选卡关闭开关按入口命名，例如 `SFT_EVAL_DISABLE_AUTO_GPU=1`、`GOALGEN_EVAL_DISABLE_AUTO_GPU=1`、`SFT_TEACHER_DISABLE_AUTO_GPU=1`。

---

## 17. SFT v2 teacher 数据规则

SFT v2 不再把冻结 teacher 生成的 ANALYSIS 作为长期维护数据集写死。长期数据集只保留
`checkpoints/sft_v2_data_pending/` 这种 `dataset_version == "v2_pending"` jsonl：
图像、MEMORY、STATUS/SUBGOAL 与 `__TEACHER_PENDING__` 占位会落盘，teacher 文本不会回写。

- 训练前预览：`tools/inspect_teacher_outputs.py --live --serve --port 0` 从 pending jsonl 抽样，
  现场调用冻结 teacher，并把网页预览写到 inspect 目录；不改训练 jsonl。
- 正式训练：`tools/sft_v2_train.sh` 检测到 `v2_pending` 后，训练启动阶段调用
  `tools/build_sft_dataset_v2_teacher.py`，把 ANALYSIS 临时物化到
  `checkpoints/sft_v2_lora/runtime_teacher_data/`（可用 `RUNTIME_TEACHER_DIR` 覆盖），再交给 ms-swift。
  默认 `RUNTIME_TEACHER_REFRESH=1` 会刷新 runtime cache，避免数据或 prompt 改动后复用旧
  teacher 文本；只有续跑同一份 pending 的中断物化任务时才设 `RUNTIME_TEACHER_REFRESH=0`。
- `check_loss_mask_v2.py`、`eval_sft_v1.py`、`probe_sft_v1.py` 需要用已经物化后的
  `dataset_version == "v2"` jsonl，例如 runtime teacher 目录下的 `val.jsonl`。

由于 ms-swift 训练入口读取 jsonl，当前“实时 teacher 真值”实现为训练启动时临时物化，
不是每个 mini-batch 在线调用 teacher；pending 源数据仍然可随 keyframes / prompt 改动重新生成。
