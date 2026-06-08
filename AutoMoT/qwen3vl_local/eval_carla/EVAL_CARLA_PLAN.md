# EVAL_CARLA_PLAN

## 2026-06-08 实现边界更新

- 模型入口现在接受 checkpoint 文件或 LeadMoT 输出目录；目录按 `best.pt` / `latest.pt` /
  `latest/best.pt` / `latest/latest.pt` / 最新滚动 checkpoint 自动解析，解析后的文件写入
  signature 与 route 的 `config.json`。
- 传感器固定为 LEAD `CARLA_LEADERBOARD2_3CAMERAS`。保留非 3cam 会造成 RGB shape 与
  LeadMoT 训练输入不一致，因此 agent 和 launcher 都显式拒绝。
- 实时历史：模型看到 4 个真实 4Hz RGB 采样点，默认跨度 0.75s；warmup 阶段低速 creep，
  不再复制首帧填历史。BEV 模型首次推理还要求最近 `STEP_STRIDE` 个 20Hz LiDAR sweep 就绪。
- BEV/LiDAR：只有 `decoder_config.use_bev=True` 的 checkpoint 才声明和读取双 LiDAR；
  no-BEV 模型不产生未使用的 LiDAR 输入。BEV 模型中，双 LiDAR 先 sensor->ego，
  再按最近 `STEP_STRIDE` 个 20Hz sweep 对齐到 anchor。
  对齐后执行 LEAD 风格的 ego box / BEV range / z range / 0.1m 量化处理，再交给
  `LeadOfflineMoTRunner.run_clip(..., bev_frame_count=1)` 栅格化。
- 动作到控制：future waypoints 是累计 ego-frame 点，按 0.25s/点解释。速度控制优先使用
  第一个未来点并混合第二段，避免忽略即时停车/起步意图。
- target_point / next_target_point 在线只能按 route lookahead 近似真实未来位置；它是闭环可用的
  navigation hint，不应再描述成与离线真值完全等价。
- 评测 launcher 默认自动选择显存占用最低的 GPU；`--num-gpus N` / `EVAL_GPU_COUNT=N`
  会选择 N 张空闲 GPU，每张卡一个 worker、一个端口槽，按 route 分片并行评测。
- 代码内已按模块补中文注释：在线输入/坐标变换/GPU worker/视频写入/API 聚合这些容易误用的
  逻辑必须保持“代码注释 + 本 MD 说明”同步更新。

LeadMoT 在 CARLA Bench2Drive 上的闭环评测全套设施。

> 实际操作命令见 [`EVAL_CARLA_RUN.md`](EVAL_CARLA_RUN.md)；本文件只讲设计与边界。

---

## 1. 子包结构

```
AutoMoT/qwen3vl_local/eval_carla/
├── __init__.py
├── EVAL_CARLA_PLAN.md      ← 本文
├── EVAL_CARLA_RUN.md
├── agent.py                ← leaderboard 实时 agent（MOTLeadAgent；继承 SafetyMixin）
├── safety.py               ← SafetyMixin：stuck_helper / parking_escape / parking_start / 限速
├── video_recorder.py       ← input / debug / demo / grid 4 路视频 + ffmpeg
├── visualizer.py           ← pinhole 投影 + waypoints overlay
├── scenario_picker.py      ← 220 routes ↔ LEAD scenario 反向映射（含 --random N）
├── aggregate.py            ← 按 scenario 聚合 leaderboard 指标
├── run_eval.sh             ← 一键 launcher（scenario / random N / full 三种跑法）
└── webapp/                 ← Flask 浏览器查看
    ├── app.py
    ├── templates/index.html
    └── static/style.css
```

### 1.1 注释覆盖范围

新增代码按“后续接手能直接定位风险点”的粒度写中文注释：

- `agent.py`：解释 checkpoint.use_bev 如何决定传感器、LEAD 3cam / LiDAR 输入、warmup、
  4Hz 历史帧、LiDAR sweep 对齐、target_point world→ego 坐标转换、PID 控制与 destroy 清理。
- `run_eval.sh`：解释自动空闲 GPU 选择、单卡/多卡 worker、端口槽、checkpoint signature 与
  断点续跑。
- `scenario_picker.py` / `aggregate.py`：解释 LEAD scenario 反向映射、AutoMoT route JSON
  schema 兼容、leaderboard JSON fallback 与 summary 输出。
- `video_recorder.py` / `visualizer.py`：解释异步 CARLA camera、grid/demo 视频、ffmpeg 压缩、
  VideoWriter 懒创建、pinhole 投影、`y_left`/`y_right` 转换。
- `webapp/`：解释 Flask API、视频路径安全检查、前端 signature/route/scenario 数据流。

当前 Python 文件的 class/function 已全部补 docstring；shell / HTML / CSS 则在关键逻辑块前写中文注释。

后续新增功能时，如果涉及输入分布、坐标系、并行调度、输出目录或可视化字段，要同步更新
对应代码注释、本文件和 `EVAL_CARLA_RUN.md`。

## 2. 关键决策点

### 2.1 传感器与 LEAD 训练分布严格对齐

| 项 | 取值 | 来源 |
|---|---|---|
| 摄像头 | 3 路，单相机 384×384 fov=60，拼成 1152×384 | `lead/lead/common/config_base.py` `CARLA_LEADERBOARD2_3CAMERAS` |
| 摄像头外参 | (0.10,-0.35,2.25,yaw=-54.5) / (0.35,0,2.25,yaw=0) / (0.10,0.35,2.25,yaw=54.5) | 同上 |
| LiDAR | use_bev=True 时启用双 LiDAR，attach 在 (0,0,2.5)，yaw=-90 / yaw=-270 | `use_two_lidars=True` |
| IMU / GPS / Speedometer | 与 mot_b2d_agent.py 一致 | leaderboard 协议 |

`SENSOR_PROFILE` 现在固定只允许 `3cam`。AutoMoT 单前视 1024x512 不是 LEAD 训练分布，
会导致动作模型 RGB 输入尺寸不兼容。LiDAR 是否启用由 checkpoint 内
`decoder_config.use_bev` 决定；no-BEV 动作模型不会额外请求未使用的 LiDAR 传感器。

### 2.2 推理节奏 4Hz

CARLA 20Hz；LeadMoT 训练数据是每 5 tick 1 帧（4Hz）。
agent 每 `STEP_STRIDE=5` 个 tick 调一次 LeadMoT，中间 4 tick 沿用上一拍
`pred_route` / `pred_future_waypoints` 做 PID 跟踪。
RGB / speed / pose / target point 历史也只在这 4Hz 采样点进入模型 clip，
避免把 20Hz 相邻帧误当成训练时 0.25s 间隔。

warmup 会等待真实 4 个 4Hz 采样点，不复制首帧填满窗口；默认首次推理在第 15 个
20Hz tick 附近触发。LiDAR 使用最近 `STEP_STRIDE` 个 20Hz sweep；每个 sweep 先按当时 GPS/heading
还原到 world，再对齐到当前 anchor 帧 ego frame 后 concat。这样保留实时高频点云
密度，同时不会把未对齐的历史 ego-frame 点直接混进 anchor 帧。

### 2.2.1 target_point / next_target_point 与离线训练一致（**关键**）

| 项 | 取值 | 来源 |
|---|---|---|
| tp / ntp 含义 | **未来 1.5s / 3.0s 的位置**（沿 global plan 弧长前瞻） | `mot_lead_offline_runner._extract_tp_ntp_from_future_frames` |
| 弧长公式 | `dist = max(speed * lookahead_s, MIN_LOOKAHEAD_M)` | agent `_lookahead_world_point` |
| world → ego | `inverse_conversion_2d(world_xy, gps_xy, theta)` | `team_code.automot_utils` |
| ego frame 约定 | (x_forward, y_left) | 与 LeadMoT 训练分布一致 |
| 低速 fallback | 5 m | `MIN_LOOKAHEAD_M` env |

**离线训练用真值未来位置**（4Hz 帧间 0.25s，tp 取 t+6 帧、ntp 取 t+12 帧）；
**在线没真值**，只能按 expected 速度沿规划路径推。这是闭环里能拿到的最接近训练
分布的 navigation hint。不要走 `RoutePlanner.run_step()` 返回的几何 min/max
距离路点——那是 mot_b2d_agent.py 的旧风格，与 LeadMoT 训练 tp 语义不同。

### 2.3 模型加载完全交给 `LeadOfflineMoTRunner`

agent 不重新写 Qwen prefill / BEV encoder / LeadMoT decoder。每帧只组装
`lead_clip` 字典调 `runner.run_clip(...)`。`use_bev` 从 ckpt 的
`decoder_config.use_bev` 自动判定，agent 拿这个标志命名输出目录。checkpoint
有 `ema_state_dict` 时默认加载 EMA shadow；需要 raw decoder 对比时设置
`LEADMOT_USE_EMA=0`。

### 2.4 输出目录组织（路径全自描述）

```
${EVAL_OUTPUT_BASE}/closed_loop_eval/
  <ckpt_parent>__<ckpt_stem>__bev{0|1}__ema{0|1}/   ← agent 自动命名
    config.json                            ← ckpt / use_bev / sensor / 录像开关
    eval_per_route/
      eval_<route_id>.json                 ← leaderboard 评测原始 json
    route<route_id>/
      input.mp4   debug.mp4   demo.mp4   grid.mp4
      meta/<step>.json
      logs/
    scenarios/<Scenario>/summary.json     ← aggregate.py 写
    summary_all.json
```

`<ckpt_parent>` = ckpt 文件父目录名，`<ckpt_stem>` = ckpt 文件无后缀；
`__bev0/__bev1` 由 ckpt 内 `decoder_config.use_bev` 决定，不接受 CLI 覆盖；
`__ema0/__ema1` 由 `LEADMOT_USE_EMA` 决定。不同模型 / 不同 use_bev /
raw-vs-EMA 永不互相覆盖。`config.json` 同时写在 signature 根目录和每个
`route<id>/` 目录，前者服务 webapp / 聚合，后者方便单 route 复查。

## 3. 四路视频参考 LEAD

| 视频 | 内容 | 是否依赖 ego vehicle world 句柄 |
|---|---|---|
| `input.mp4` | 1152×384 三视角拼接 RGB（模型实际输入） | 否 |
| `debug.mp4` | input 上叠加 `pred_waypoints` / `target_point` 投影（按相机段分别画） | 否 |
| `demo.mp4`  | spawn cinematic（车后 6.5m 高 6m）+ BEV（顶视 22m）两个 RGB camera，横向拼 | 是（在首帧 setup_demo_cameras） |
| `grid.mp4`  | demo 在上、input 在下，按 input 宽度等比缩放 demo | 是（demo 帧来源） |

四路开关：`RECORD_INPUT / RECORD_DEBUG / RECORD_DEMO / RECORD_GRID` 环境变量
或 launcher 的 `--no-input / --no-debug / --no-demo / --no-grid`。

ffmpeg 压缩：`input/demo/grid` crf=18 preset=slow；`debug` crf=28 preset=slower。
路线结束时 `destroy()` 触发 release + 压缩。

## 4. 场景体系结合 LEAD

LEAD 按 `<Scenario>/<route_id>.xml` 组织 220 routes，AutoMoT 按 route_id 列表跑。
两边 route_id 同源。

- `scenario_picker.py` 扫 `lead/data/benchmark_routes/bench2drive220/` 建反向
  `route_id → [scenario, ...]` 映射；`--scenario <Name>`、`--route-id <ID>` 可
  多次叠加做子集筛选。
- launcher 跑完后自动调 `aggregate.py`，把当前 signature 下
  `eval_per_route/eval_<route_id>.json` 里的 leaderboard route record 指标按
  scenario 聚合写到 `scenarios/<Scenario>/summary.json`。
- 多卡评测时 `run_eval.sh` 会把 route 按 worker index 做 round-robin 分配；每个 worker
  调同一个 leaderboard `run_evaluation.sh`，但传入不同 GPU id 和端口槽，输出仍归档到同一个
  signature 的 `eval_per_route/` 下。

## 5. Webapp（`webapp/app.py`）

Flask app 扫一个 `closed_loop_eval/` 目录：
- 侧边栏：按 scenario 分组列出所有 route，显示 score_composed 等核心指标。
- 右上：视频播放器（标签页切换 input / debug / demo / grid）。
- 右下：当前 route 的 leaderboard json 解析（score_composed / score_route /
  score_penalty / infractions 列表）。
- 顶部 tab：还有一个 *Scenarios* 聚合视图，列每个 scenario 平均分。

启动：`python AutoMoT/qwen3vl_local/eval_carla/webapp/app.py --eval-base <path>`，
默认 5050 端口。

## 6. 安全兜底（SafetyMixin，与 mot_b2d_agent.py 一致）

`safety.py` 把 mot_b2d_agent.py 的安全策略全部抽出来作为独立 mixin，agent.py
只 import 即用：

| 项 | 默认 | 行为 |
|---|---|---|
| stuck_detector | `speed < 0.1 m/s` 累计 300 帧 | 触发 force_move 14 帧 creep_throttle=0.4 |
| parking_start | 前 200 帧位移 < 6m | 本局禁用 force_move（停车起步场景兼容） |
| parking_escape | 1500 帧窗口内最大位移 < 5m | 激活 phase 1：向左强转角 -0.65 + 油门 0.45 共 40 帧 |
| escape 提前结束 | 位移 > 6m 或航向变化 > 25° | 进入 2400 帧冷却 |
| 限速 | speed > 35 km/h | throttle=0, brake=1 强制刹车 |

不再用 pygame `DisplayInterface`：本子包用 mp4 视频（input/debug/demo/grid）替代实时可视化。

## 7. 状态

- [x] 传感器对齐（LEAD 3 摄像头 1152×384；use_bev=True 时启用双 LiDAR）
- [x] LeadMoT 推理桥接（复用 `LeadOfflineMoTRunner`）
- [x] target_point / next_target_point 时间 lookahead（1.5s / 3s）
- [x] 4 路视频（input / debug / demo / grid）+ ffmpeg
- [x] safety mixin（stuck / parking_start / parking_escape / 限速）
- [x] scenario 反向映射 + 聚合
- [x] 三种跑法（scenario / random N / full）+ 单卡/多卡自动 GPU worker
- [x] Flask webapp

跑通后还可以加：高级 BEV 顶视可视化（pred_route + 周车 bbox），与 LEAD
`visualizer.py` 对齐。当前先用 demo.mp4 的 CARLA 顶视 RGB 替代。

## 7. 与原 `AutoMoT/leaderboard/team_code/mot_b2d_agent.py` 的关系

- 沿用：`nav_planner.RoutePlanner / LateralPIDController`、`automot_utils.PIDController` /
  `inverse_conversion_2d`、`ukf_utils.*`、`run_evaluation.sh` 的 launcher 协议。
- 不复用：原 agent 内部 `InterleaveInferencer` 路径与 1024×512 单前视；这两点
  与 LeadMoT 训练分布冲突。
- 不修改 mot_b2d_agent.py 本身。LeadMoT 闭环走自己的 agent。
