# EVAL_CARLA_RUN

LeadMoT 闭环评测一键操作手册。架构与对齐细节见 [`EVAL_CARLA_PLAN.md`](EVAL_CARLA_PLAN.md)。

---

## 0. 前置

- 远程已装 CARLA（`$CARLA_ROOT` 或 `~/carla` 自动探测）
- LeadMoT decoder checkpoint 就位（`AutoMoT/qwen3vl_local/leadmot/train.py` 产物）
- LEAD benchmark_routes 就位：`lead/data/benchmark_routes/bench2drive220/<Scenario>/<route_id>.xml`
- 安装 Flask（webapp 用）：`pip install flask`

**CARLA 自带启动**（无需手动 `start_carla.sh`）：

- 每个 worker 进入 `leaderboard_evaluator.py` 后，在自己 GPU 上自动 spawn 一个 CARLA server
- **端口扫描空闲**：主进程从 `PORT_BASE_START=5000` 开始按 `PORT_STRIDE=20` 步进，
  扫到一个 `[port, port+1, port+2, port+3, port+8000]` 都空闲的块就分配给当前
  worker，之后从 `port+stride` 继续扫给下一个 worker。被占用的槽自动跳过。
- `CUDA_VISIBLE_DEVICES=$gpu_rank` 锁住 CARLA 看到的卡，CARLA 自身只见 `cuda:0`，
  和 leaderboard_evaluator.py 落在同一张物理卡
- worker 结束 / Ctrl+C / 异常退出时，`leaderboard_evaluator.py` 会清理自己启动的 CARLA
- `--auto-carla` / `--no-auto-carla` 现在只是兼容旧命令；本 launcher 不再预启动 CARLA，
  避免和 `leaderboard_evaluator.py` 双重启动抢端口

**实时输出**（默认）：

- 每个 worker 写独立 log 到 `<signature>/worker_logs/<ts>/worker<N>.log`
- 主进程 fork 一个 `tail -F` 把所有 worker log 行内容实时输出到终端
- 单 worker 直接 follow worker0.log，多 worker 时 tail 自带 `==> path <==` 头便于区分
- agent.py 每 20 tick (=1s) 打一行总览：`tick=N speed=X km/h ctrl=(thr,str,brk) tp=Xm wp_end=Ym lidar_pts=N`
- 每个 4Hz 推理 tick 打一行：`INFER step=N dt=Xms |wp[0]|=Xm |wp[-1]|=Ym |rt[-1]|=Ym`
- 所有 print 都 `flush=True` 保证不被 buffer 卡住

---

## 1. 三种跑法

> `--leadmot-ckpt` 可以传具体 checkpoint 文件，也可以传训练输出目录（自动解析
> `best.pt` / `latest.pt` / `latest/best.pt` / `latest/latest.pt` / 最新滚动 ckpt）。
>
> launcher 共享同一份代码 `run_eval.sh`，三种跑法只是过滤器不同。每种跑法的
> route 视频、leaderboard json 共享同一个 ckpt signature 目录（断点续跑），
> 但本次的 scenario 聚合结果写到 `runs/<RUN_LABEL>/` 下，**互不污染**。

### 1.1 全量 220 路线（默认）

```bash
cd AutoMoT
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt checkpoints/leadmot_v1_decoder/latest/best.pt
```

多卡并行：

```bash
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt checkpoints/leadmot_v1_decoder/latest/best.pt \
    --num-gpus 4
```

按显存占用最低自动选 GPU；CARLA 端口由 launcher 从 `PORT_BASE_START` 起按
`PORT_STRIDE` 扫描空闲 `[rpc..rpc+3, tm]` 端口块。

### 1.2 按 scenario 跑

```bash
# 单类
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt /path/best.pt \
    --scenario PedestrianCrossing

# 多类叠加（取并集）
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt /path/best.pt \
    --scenario PedestrianCrossing --scenario Accident
```

查看所有 LEAD scenario 名（带每类 route 数）：

```bash
python3 qwen3vl_local/eval_carla/scenario_picker.py --list-scenarios
```

### 1.3 随机 N 个

```bash
# 全量池里随机抽 20 条；seed 固定可复现
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt /path/best.pt \
    --random 20 --seed 42

# 与 --scenario 叠加：在 PedestrianCrossing + Accident 里随机抽 10 条
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt /path/best.pt \
    --scenario PedestrianCrossing --scenario Accident \
    --random 10 --seed 42
```

### 1.4 指定 route_id

```bash
# 单条 + 烟雾测试（跑完不聚合）
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt /path/best.pt \
    --route-id 1711 --single-test

# 多条
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt /path/best.pt \
    --route-id 1711 --route-id 1773
```

`--scenario` / `--route-id` / `--random` 任意叠加：先按 scenario 过滤、再按
route_id 精确筛、最后在剩余里 `--random N` 抽样。

---

## 2. 视频开关（五路，默认全开）

| 文件 | 内容 | 关闭方式 |
|---|---|---|
| `input.mp4` | 1152×384 三视角拼接 RGB（已 JPEG round-trip） | `--no-input` |
| `debug.mp4` | input 上叠加 pred_waypoints + tp 相机投影 | `--no-debug` |
| `bev_debug.mp4` | 顶视 BEV：LiDAR + pred_route + waypoints + tp/ntp + ego box | `--no-bev-debug` |
| `demo.mp4` | spawn cinematic + BEV 临时 carla camera | `--no-demo` |
| `grid.mp4` | demo 上、input 下，等宽拼接 | `--no-grid` |

任意跑法默认都录全 5 路视频。最省资源（只录 input + bev_debug）：

```bash
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt /path/best.pt \
    --no-debug --no-demo --no-grid
```

---

## 3. 推理 / 传感器 / target_point 配置

| 标志 / env | 默认 | 说明 |
|---|---|---|
| `--step-stride 5` | 5 | 每多少 tick 调一次模型；4Hz 与训练分布一致 |
| `--num-gpus N` | 1 | 自动选 N 张空闲 GPU；也可用 `EVAL_GPU_COUNT=N` |
| `--rope mrope` | mrope | mrope / mhrope / none |
| `--sensor-profile 3cam` | 3cam | 仅支持 LEAD 三相机档 |
| `--auto-carla` / `--no-auto-carla` | 兼容旧命令 | launcher 不预启动 CARLA；实际由 leaderboard_evaluator.py 启动 |
| env `CARLA_BOOT_TIMEOUT` | legacy | launcher 不预启动 CARLA 时不生效 |
| env `PORT_BASE_START` | 5000 | worker CARLA 空闲端口扫描起点 |
| env `PORT_STRIDE` | 20 | 空闲端口块扫描步长 |
| env `LEADMOT_USE_EMA` | 1 | checkpoint 里有 EMA shadow 时默认加载；设 0 用 raw decoder |
| env `JPEG_QUALITY` | 85 | RGB 拼接后 JPEG round-trip quality；设 0 关闭模拟 |
| env `TP_LOOKAHEAD_S` | 1.0 | target_point 未来时长（秒），落在 wp 视野 2s 内 |
| env `NTP_LOOKAHEAD_S` | 2.0 | next_target_point 时长（秒），与 wp 末端同步 |
| env `MIN_LOOKAHEAD_M` | 5.0 | 低速 fallback 最小前瞻：`dist = max(speed*lookahead, 5m)` |
| env `LIDAR_REMOVE_GROUND` | 1 | 轻量去地面（z+LSQ）；设 0 关闭 |
| env `LIDAR_GROUND_Z` | -1.4 | 地面 z 高度阈值（ego frame） |
| env `USE_RADAR` | 1 | use_bev=True 时是否声明 4 个 radar 并拼到 LiDAR |
| env `RECORD_BEV_DEBUG` | 1 | 是否写 bev_debug.mp4 |

在线 target_point / next_target_point 对齐训练侧 route lookahead：每 tick 调
`RoutePlanner.run_step()` 推进 route 后，按 `max(speed*lookahead_s, MIN_LOOKAHEAD_M)`
沿剩余 route 弧长插值；默认 tp=1.0s、ntp=2.0s。final_goal 来自
`scenario_picker` 按 route_id 读取的 LEAD route XML 最后一个 waypoint，再转成
当前 ego frame；这与训练侧 `meta["next_target_points"][-1]` 的剩余 route 终点语义一致。

---

## 4. 输出目录

```
${EVAL_OUTPUT_BASE:-AutoMoT/outputs/closed_loop_eval}/
  <ckpt_parent>__<ckpt_stem>__bev{0|1}__ema{0|1}/
    config.json                              ← ckpt / use_bev / 传感器 / 录像 / env 设定
    eval_per_route/eval_<route_id>.json      ← leaderboard 原始结果（跨跑法共享）
    route<route_id>/                          ← route 级输出（跨跑法共享，断点续跑）
      input.mp4 debug.mp4 bev_debug.mp4 demo.mp4 grid.mp4
      meta/<step>.json                        ← 每个推理 tick 的 pred + 耗时 + 输入统计（speed/tp/ntp/final_goal）
      logs/
    runs/<RUN_LABEL>/                          ← 本次启动的聚合结果（与其他批次隔离）
      scenarios/<Scenario>/summary.json
      summary_all.json
      run_manifest.json                        ← 本批次 route_id 列表 + 启动参数
```

`<RUN_LABEL>` 由跑法自动生成：

| 跑法 | RUN_LABEL |
|---|---|
| 全量（无过滤） | `full` |
| `--scenario PedestrianCrossing` | `scenario_PedestrianCrossing` |
| `--scenario A --scenario B` | `scenario_A+B` |
| `--random 20 --seed 42` | `random_N20_S42` |
| 组合 | `scenario_A__random_N5_S0` |
| `--route-id 1711 --route-id 1773` | `routes_1711+1773` |
| `--single-test` | `smoke_<first_route_id>` |
| 自定义 | `--run-label my_label` 覆盖 |

`<ckpt_signature>` 由 ckpt 路径、use_bev、`LEADMOT_USE_EMA` 自动生成，不接受 CLI
覆盖；不同 ckpt / 不同 use_bev / raw-vs-EMA 永不互相覆盖。

---

## 5. 聚合单独跑

```bash
cd AutoMoT
# 聚合 signature 下指定 run_label 批次
python3 -m AutoMoT.qwen3vl_local.eval_carla.aggregate \
    --eval-base outputs/closed_loop_eval \
    --leadmot-ckpt checkpoints/leadmot_v1_decoder/latest/best.pt \
    --run-label full

# --leadmot-ckpt 可省略：聚合所有 signature 目录
# --run-label 可省略：聚合该 signature 下所有 runs
```

---

## 6. Webapp 浏览器查看

```bash
pip install flask
python3 qwen3vl_local/eval_carla/webapp/app.py \
    --eval-base outputs/closed_loop_eval --port 5050
```

打开 `http://<远程ip>:5050`：
- 顶部下拉切换 ckpt signature 和 run_label
- Routes：左栏按 scenario 分组列 route 与 score_composed，点开右栏切五路视频 +
  leaderboard scores + infractions
- Scenarios：表格展示本 run 下每个 scenario 的平均分

---

## 7. 常见坑

- **CARLA 启动超时**：现在由 `leaderboard_evaluator.py` 启动并等待 CARLA。可以查
  `<signature>/worker_logs/<ts>/worker<N>.log` 里的 `Launch CARLA`、`load_world failed`
  和 evaluator traceback。
- **端口被占**：launcher 会自动跳过被占用的 `[rpc..rpc+3, tm]` 端口块；
  若扫描范围内都找不到空闲块，会报 `cannot find free CARLA port block`。可以调大
  `PORT_BASE_START` 或 `PORT_STRIDE`。
- **首帧 demo 摄像头 spawn 失败**：通常是 ego vehicle 还没注册 `role_name=hero`。
  agent 打印 `hero vehicle not found; demo cameras skipped`，本路线 demo / grid
  会缺，input / debug / bev_debug 不受影响。
- **ffmpeg 不在 PATH**：视频不压缩，留原始 mp4v 编码 mp4，可播放但体积大。
- **debug.mp4 在 warmup 阶段空白**：第一次 4Hz 采样点（默认第 5 个 tick）才有
  pred_waypoints；bev_debug.mp4 从 t=0 就有 ego box，更适合诊断 warmup。
- **target_point 起步死锁**：在线使用 `max(speed*lookahead_s, 5m)`，静止时 tp/ntp
  都给出 5m 前方方向；这是有意设计，不会因为 speed≈0 把 tp/ntp 置成 ego 当前点。
- **parking_escape 误触**：默认 1500 帧（125s）位移 < 5m 才触发，红灯等灯正常
  不会触发。如果场景普遍 lights 等待 > 2 分钟，在 `safety.py` 调大窗口。
