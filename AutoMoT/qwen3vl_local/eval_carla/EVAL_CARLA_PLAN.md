# EVAL_CARLA_PLAN

LeadMoT 在 CARLA Bench2Drive 上的闭环评测全套设施的**架构与对齐边界**。
实际操作命令见 [`EVAL_CARLA_RUN.md`](EVAL_CARLA_RUN.md)。

---

## 1. 子包结构

```
AutoMoT/qwen3vl_local/eval_carla/
├── __init__.py
├── EVAL_CARLA_PLAN.md      ← 本文（架构）
├── EVAL_CARLA_RUN.md       ← 操作手册
├── agent.py                ← leaderboard 实时 agent（MOTLeadAgent；继承 SafetyMixin）
├── safety.py               ← SafetyMixin：stuck / parking_escape / parking_start / 限速
├── video_recorder.py       ← input / debug / bev_debug / demo / grid 五路 mp4 + ffmpeg
├── visualizer.py           ← pinhole 投影 + 三视角 overlay + BEV 顶视调试
├── scenario_picker.py      ← 220 routes ↔ LEAD scenario 反向映射（含 --random N）
├── aggregate.py            ← 按 scenario 聚合 leaderboard 指标
├── run_eval.sh             ← 一键 launcher（scenario / random N / full 三种跑法）
└── webapp/                 ← Flask 浏览器查看
    ├── app.py
    ├── templates/index.html
    └── static/style.css
```

所有 Python class/function 都有中文 docstring；shell / HTML / CSS 在关键逻辑块前
有中文注释。新增功能时同步更新代码注释 + 本文件 + `EVAL_CARLA_RUN.md`。

---

## 2. 输入分布与训练严格对齐

### 2.1 传感器（LEAD `CARLA_LEADERBOARD2_3CAMERAS`）

| 项 | 取值 |
|---|---|
| 摄像头 | 3 路 384×384 fov=60，拼成 1152×384，外参 (0.10,±0.35,2.25)/(0.35,0,2.25) yaw ±54.5° |
| LiDAR | 双 LiDAR，attach (0,0,2.5)，yaw -90 / -270，`use_two_lidars=True`；只在 use_bev=True 声明 |
| Radar | 4 路（前左/前右/后左/后右），位置/朝向同 LEAD `radar_calibration`；只在 use_bev=True 声明 |
| IMU / GPS / Speedometer | 与 mot_b2d_agent.py 一致 |

`SENSOR_PROFILE` 固定 `3cam`，非 3cam 直接报错。

### 2.2 LiDAR / Radar 融合（与 LEAD `base_agent.tick()` 同源）

1. **sensor → ego**：`(Rzyx @ pts.T).T + translation`，末尾 `z -= sensor_pos[2]/2`
   （与 LEAD `lidar_to_ego_coordinate` 完全一致）
2. **radar 4 路 → ego** 同样公式，并拼到 LiDAR；近车 < 8m 的 radar 点 duplicate
   5 次（LEAD `duplicate_radar_near_ego=True`, factor=5）
3. **去 ego box**：`abs(x)>extent_x AND abs(y)>extent_y`
4. **BEV 范围裁切**：x∈[-32,64], y∈[-40,40], z∈[-4,10]
5. **轻量去地面**：z 阈值 + LSQ 平面拟合（LEAD 用 RANSAC 径向分段，依赖 numba 重依赖；
   按 PROJECT_CONTEXT.md §1 不引入）
6. **0.1m 量化** XYZ
7. **5 sweep 累积**：最近 `step_stride=5` 个 20Hz sweep（0.25s 窗）按 (dx, dy, dyaw)
   对齐到当前 anchor ego frame 后 concat —— 等价于 LEAD `.laz` 单帧含的 5 sweep

### 2.3 RGB（与 LEAD `sensor_agent.tick()` 同源）

- 三视角横拼后做 JPEG round-trip（`JPEG_QUALITY=85`）模拟训练 `.jpg` 分布。

### 2.4 推理节奏 4Hz

- CARLA 20Hz；每 `STEP_STRIDE=5` 个 tick 调一次模型，中间 4 tick 用上一拍预测做 PID
- 4 帧 RGB 历史跨 `(4-1)*5/20 = 0.75s`，与 LEAD `rgb_frame_step=1` 训练一致
- pred_future_waypoints `(B, 8, 2)`，dt=0.25s，跨度 2s
- pred_route `(B, 10, 2)` 是 navigation route 累计绝对点

### 2.5 warmup（LEAD 风格 left-pad）

- 首帧 brake=1，第一个 4Hz 采样点拿到后立即推理
- clip 历史不足时复制 frame 0 left-pad 到 4 帧（与 `build_clip` line 1808-1815 同款）
- LeadMoT dataloader 见过这种 pad 输入，不算 OOD

### 2.6 target_point / next_target_point

- 离线训练：未来 1.5s / 3.0s 真值位置（`_extract_tp_ntp_from_future_frames`）
- 在线推理：
  - 对齐 AutoMoT `mot_b2d_agent.py`：`RoutePlanner.run_step()` 推进 route 后，
    直接取剩余 route[1] / route[2] 作为 target_point / next_target_point
  - route 点不足时按 AutoMoT 兜底：只有 tp 时沿 ego→tp 方向延展 5m 生成 ntp；
    route 为空时沿当前航向前推 50m
  - 不再使用 `speed * lookahead_s`，也不再低速 tp=ego；RoutePlanner 的
    `min_distance=7.5m` 在静止起步时仍提供非零前方目标，避免 waypoint head 死锁
- world → ego：`inverse_conversion_2d(world_xy, gps_xy, theta)`
- ego frame 约定 `(x_forward, y_left)`

### 2.7 PID 控制

- desired speed：用 wp[1] (0.5s) 与 wp[3] (1.0s) 两段距离平均（LEADMOT_PLAN.md §32）
- steer：`LateralPIDController` 走 `pred_route`
- 限速 35 km/h 强制刹车（SafetyMixin）

---

## 3. 模型加载

agent 不重写 Qwen prefill / BEV encoder / LeadMoT decoder。每帧只组装 `lead_clip`
字典调 `LeadOfflineMoTRunner.run_clip(...)`。`use_bev` 从 ckpt 的
`decoder_config.use_bev` 自动判定；checkpoint 有 `ema_state_dict` 时默认走 EMA shadow
（`LEADMOT_USE_EMA=0` 强制 raw decoder）。

`--leadmot-ckpt` 可传文件或训练输出目录：自动解析 `best.pt → latest.pt →
latest/best.pt → latest/latest.pt → 最新 step-checkpoint-*.pt`。

---

## 4. 输出目录组织

```
${EVAL_OUTPUT_BASE}/closed_loop_eval/
  <ckpt_parent>__<ckpt_stem>__bev{0|1}__ema{0|1}/       ← signature，由 ckpt 自动命名
    config.json                                          ← 模型 + 传感器 + env 设定
    eval_per_route/eval_<route_id>.json                  ← leaderboard 原始结果（共享）
    route<route_id>/                                      ← 单 route 输出（共享，断点续跑）
      {input,debug,bev_debug,demo,grid}.mp4
      meta/<step>.json                                    ← 每 4Hz 推理 tick 的 pred + 输入统计
      logs/
    runs/<RUN_LABEL>/                                     ← 本次启动的聚合（按跑法隔离）
      scenarios/<Scenario>/summary.json
      summary_all.json
      run_manifest.json                                    ← 本批 route_id + 启动参数 + 时间戳
```

### 4.1 signature 命名规则

由 ckpt 路径 + use_bev（从 ckpt 读出）+ `LEADMOT_USE_EMA` 自动生成，**不接受 CLI 覆盖**。
不同 ckpt / 不同 use_bev / raw-vs-EMA 永不互相覆盖。

### 4.2 RUN_LABEL 规则

由跑法自动生成，让"全量"和"随机 N"等批次的聚合结果分目录保存：

| 跑法 | RUN_LABEL |
|---|---|
| 全量（无过滤） | `full` |
| `--scenario X` 单类 | `scenario_X` |
| `--scenario A --scenario B` | `scenario_A+B` |
| `--random N --seed K` | `random_N{N}_S{K}` |
| `--scenario X --random N` | `scenario_X__random_N{N}_S{K}` |
| `--route-id A --route-id B` | `routes_A+B`（最多列前 3 个） |
| `--single-test` | `smoke_<first_route_id>` |
| `--run-label X` | `X`（手动覆盖） |

route 视频 + leaderboard json 仍共享 signature 根，**只有聚合结果按 RUN_LABEL 分目录**——
跨批次断点续跑、聚合不互相覆盖。

---

## 5. 五路视频参考 LEAD `video_recorder.py`

| 视频 | 内容 | 依赖 carla world |
|---|---|---|
| `input.mp4` | 1152×384 拼接 RGB（已 JPEG round-trip） | 否 |
| `debug.mp4` | input 上叠加 pred_waypoints + tp 投影 | 否 |
| `bev_debug.mp4` | 顶视：LiDAR 散点 + pred_route + pred_waypoints + tp/ntp + ego box（等价 LEAD BEV pseudo-image） | 否 |
| `demo.mp4` | spawn cinematic（车后 6.5m 高 6m）+ BEV（顶视 22m）两个 RGB camera 横向拼 | 是（首帧 setup_demo_cameras） |
| `grid.mp4` | demo 在上、input 在下，按 input 宽度等比缩放 demo | 是（demo 帧来源） |

ffmpeg 压缩：input/demo/grid crf=18 preset=slow；bev_debug crf=22 preset=slow；
debug crf=28 preset=slower。路线结束时 `destroy()` 触发 release + 压缩。

---

## 5.x CARLA 启动与端口分配

`run_eval.sh` 只负责选 GPU、筛 route、预分配端口起点和实时回放 worker log；
真正的 CARLA server 由 `leaderboard_evaluator.py` 在 worker 进程内启动并在退出时清理。
这样避免 launcher / evaluator 双重启动 CARLA 抢端口。

| 项 | 实现 |
|---|---|
| 端口 | launcher 从 `PORT_BASE_START=5000` 开始按 `PORT_STRIDE=20` 扫描；要求 `[p, p+1, p+2, p+3, p+8000]` 都空闲，避免 evaluator 再改端口 |
| GPU 绑定 | `run_evaluation.sh` 以 `CUDA_VISIBLE_DEVICES=$gpu_rank` 启动 `leaderboard_evaluator.py`，evaluator 再启动 CARLA；模型和 CARLA 落在同一张物理卡 |
| CARLA 启动 | `leaderboard_evaluator.py` 调 `CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=<p> -carla-streaming-port=<p+1>` |
| TrafficManager | launcher 传入 `tm_port=p+8000`；evaluator 仍会做一次可用性检查 |
| 退出回收 | `leaderboard_evaluator.py` 注册 `atexit` 清理自己启动的 CARLA 进程组 |
| 命令行 | `--auto-carla` / `--no-auto-carla` 只兼容旧命令；launcher 默认不预启动 CARLA |
| 日志 | `<signature>/worker_logs/<ts>/worker<N>.log` 实时包含 `Launch CARLA`、`load_world`、agent 进度和 traceback |

主进程 fork `tail -F` 实时回放 worker log；agent 每 20 tick 打一行速度 / 控制 /
target point / LiDAR 点数，每次模型推理打一行 `INFER step=... dt=...`。

## 6. 场景反向映射 + 聚合

`scenario_picker.py` 扫 `lead/data/benchmark_routes/bench2drive220/` 建反向
`route_id → [scenario, ...]` 映射；CLI `--scenario` / `--route-id` / `--random N --seed K` /
`--list-scenarios`。

`aggregate.py` 把 `eval_per_route/eval_<route_id>.json` 里的 leaderboard route record
按 scenario 聚合，写到 `runs/<RUN_LABEL>/scenarios/<Scenario>/summary.json` +
`runs/<RUN_LABEL>/summary_all.json`。

多卡评测时 `run_eval.sh` 按 worker index round-robin 分 route，每个 worker 不同 GPU id
+ 端口槽，输出仍归档到同一个 signature 的 `eval_per_route/`。

---

## 7. SafetyMixin（兜底）

与 mot_b2d_agent.py 行为完全一致：

| 项 | 默认 | 行为 |
|---|---|---|
| `stuck_detector` | speed<0.1 累计 300 帧 | 触发 force_move 14 帧 creep_throttle=0.4 |
| `parking_start` | 前 200 帧位移 < 6m | 本局禁用 force_move（停车起步兼容） |
| `parking_escape` | 1500 帧窗口位移 < 5m | phase 1：steer=-0.65 + throttle=0.45 共 40 帧 |
| escape 提前结束 | 位移 > 6m 或航向变化 > 25° | 进入 2400 帧冷却 |
| 限速 | speed > 35 km/h | throttle=0, brake=1 强制刹车 |

LEAD 用 `ForceMovePostProcessor` + `StopSignPostProcessor`（基于场景理解），我们沿用
mot_b2d_agent.py 的位移阈值方案，工程量小且对 Bench2Drive 220 routes 已足够。

---

## 8. Webapp（`webapp/app.py`）

- 顶部下拉切换 ckpt signature 和 run_label
- Routes tab：按 scenario 分组列 route + 五路视频切换 + leaderboard scores + infractions
- Scenarios tab：当前 run_label 下每个 scenario 的平均分

视频路径：`/video/<signature>/route<id>/<v>.mp4`；后端有路径安全检查防止越界。
