# EVAL_CARLA_RUN

## 2026-06-08 完善说明

- `--leadmot-ckpt` 可以传具体 checkpoint 文件，也可以传训练输出目录。目录解析顺序：
  `best.pt` -> `latest.pt` -> `latest/best.pt` -> `latest/latest.pt` -> 最新的
  `step-checkpoint-*.pt` / `checkpoint-epoch*.pt` / `*.pt` / `*.safetensors`。
- `run_eval.sh` 默认自动选择 1 张显存占用最低的 GPU；传 `--num-gpus N` 或设置
  `EVAL_GPU_COUNT=N` 时会自动选择 N 张空闲 GPU，每张卡一个 worker 并行跑 route。
- 实时 CARLA 评测只支持 LEAD 训练分布的 `3cam`：三路 384x384、FOV=60、拼接为
  1152x384。`1cam` 与动作模型输入尺寸不兼容，launcher 和 agent 都会拒绝。
- 过去 4 帧 RGB 是 4Hz 采样，默认 `STEP_STRIDE=5`，所以历史跨度是
  `(4-1)*5/20=0.75s`。动作头输出的 future waypoints 仍按 0.25s/点解释，8 点约 2s。
- warmup 不再复制首帧填历史；agent 会等到真实 4 个 4Hz 采样点后首次推理，默认约
  0.75s。使用 BEV 时同时要求最近 `STEP_STRIDE` 个 20Hz LiDAR sweep 已就绪，等待期间低速 creep。
- LiDAR 只在 checkpoint 的 `decoder_config.use_bev=True` 时声明和读取；no-BEV 模型只产生
  RGB / GPS / IMU / speedometer 等动作模型实际需要的输入。
- BEV 模型的 LiDAR 实时输入会在 sensor->ego 后和 anchor-frame 融合后执行 LEAD 风格的确定性处理：
  去 ego box、限制 BEV 范围、z in [-4, 10]、0.1m 量化。没有引入 RANSAC 去地面，避免在线重依赖。
- 每个推理 `meta/<step>.json` 会记录 resolved checkpoint、历史跨度、waypoint dt、
  LiDAR sweep 数、点数、target_point、预测轨迹和上一 tick 控制量，便于闭环诊断。
- 新增代码已补中文注释：重点覆盖输入适配、坐标变换、warmup、GPU worker、视频/可视化、
  scenario 聚合和 webapp API。以后改这些行为时要同步更新代码注释和本运行说明。
- 当前 Python class/function 已全部补 docstring；shell / HTML / CSS 也在关键逻辑块前补了中文说明。

LeadMoT 闭环评测一键操作手册。设计与边界见 [`EVAL_CARLA_PLAN.md`](EVAL_CARLA_PLAN.md)。

---

## 0. 前置

- 远程已装 CARLA（`$CARLA_ROOT` 或 `~/carla` 自动探测）
- LeadMoT decoder checkpoint 就位（`AutoMoT/qwen3vl_local/leadmot/train.py` 产物）
- LEAD benchmark_routes 就位：`lead/data/benchmark_routes/bench2drive220/<Scenario>/<route_id>.xml`
- 安装 Flask（webapp 用）：`pip install flask`

---

## 1. 三种跑法

> launcher 共享一份代码 `run_eval.sh`，区别只在过滤器。所有模式都自动 ffmpeg 压缩
> 四路视频、自动跑完后聚合到 `scenarios/`。

### 1.1 全量 220 路线（默认）

```bash
cd AutoMoT
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt checkpoints/leadmot_v1_decoder/latest/best.pt
```

已评估过的 route 自动跳过（对应 signature 下
`eval_per_route/eval_<id>.json` 存在即视为已跑），可断点续跑。不同动作模型、
不同 `use_bev`、raw-vs-EMA 会落到不同 signature，不会互相跳过或覆盖。

多卡并行跑全量：

```bash
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt checkpoints/leadmot_v1_decoder/latest/best.pt \
    --num-gpus 4
```

脚本会用 `nvidia-smi` 按显存占用从低到高选择 4 张卡，并按 GPU id 分配端口槽：
`PORT_BASE_START + gpu_id * PORT_STRIDE`；默认 `PORT_BASE_START=5000`、
`PORT_STRIDE=20`。每张卡一个 worker，route 以 round-robin 分配。

### 1.2 按场景跑（LEAD scenario 子集）

```bash
# 只跑 PedestrianCrossing 这一类
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
python3 AutoMoT/qwen3vl_local/eval_carla/scenario_picker.py --list-scenarios
```

### 1.3 随机 N 个 route

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
# 跑单条 + 烟雾（跑完不聚合）
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt /path/best.pt \
    --route-id 1711 --single-test

# 多条指定
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt /path/best.pt \
    --route-id 1711 --route-id 1773
```

`--scenario` / `--route-id` / `--random` 任意叠加：先按 scenario 过滤、再按
route_id 精确筛、最后在剩余里 `--random N` 抽样。

---

## 2. 视频开关

| 标志 | 效果 |
|---|---|
| `--no-input` | 不写 input.mp4 |
| `--no-debug` | 不写 debug.mp4 |
| `--no-demo`  | 不写 demo.mp4（同时跳过 cinematic / BEV 临时摄像头 spawn） |
| `--no-grid`  | 不写 grid.mp4 |

例：只录 input + debug 最省资源：

```bash
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt /path/best.pt --no-demo --no-grid
```

---

## 3. 推理 / 传感器档 / target_point lookahead

| 标志 | 默认 | 说明 |
|---|---|---|
| `--step-stride 5` | 5 | 每多少 tick 调一次模型；4Hz 与训练分布完全一致 |
| `--num-gpus N` | 1 | 自动选择 N 张空闲 GPU 并行跑 route；也可用 `EVAL_GPU_COUNT=N` |
| `--rope mrope` | mrope | mrope / mhrope / none |
| `--sensor-profile 3cam` | 3cam | 仅支持 LEAD 三相机档；非 3cam 会直接报错 |
| env `LEADMOT_USE_EMA` | 1 | 默认加载 checkpoint 里的 EMA shadow；设 0 强制 raw decoder |
| env `TP_LOOKAHEAD_S` | 1.5 | target_point 未来时长（秒），与离线 build_clip 一致 |
| env `NTP_LOOKAHEAD_S` | 3.0 | next_target_point 未来时长（秒） |
| env `MIN_LOOKAHEAD_M` | 5.0 | 低速 fallback 最小前瞻距离（米） |

例：如果你训练时用了 1.0s / 2.5s 的 lookahead，请保持环境变量一致：

```bash
TP_LOOKAHEAD_S=1.0 NTP_LOOKAHEAD_S=2.5 \
bash qwen3vl_local/eval_carla/run_eval.sh --leadmot-ckpt /path/best.pt
```

target_point / next_target_point 实现：每个 tick 都按当前速度 v 沿
`_global_plan_world_coord` 弧长前推 `max(v * lookahead_s, MIN_LOOKAHEAD_M)` 米，
取出 world 坐标后用 `inverse_conversion_2d(world, gps, theta)` 转 ego frame
(x_forward, y_left)。在线没有真实未来位置，这只是对离线 tp/ntp 语义的 route-lookahead 近似。

---

## 4. 聚合脚本单独跑

```bash
cd AutoMoT
python3 -m AutoMoT.qwen3vl_local.eval_carla.aggregate \
    --eval-base outputs/closed_loop_eval \
    --leadmot-ckpt checkpoints/leadmot_v1_decoder/latest/best.pt
```

`--leadmot-ckpt` 可省略，省略时聚合 `eval_base` 下所有
`*__*__bev{0|1}__ema{0|1}` 签名目录（也兼容旧的 `*__*__bev{0|1}`）。

---

## 5. Webapp 浏览器查看

```bash
pip install flask
python3 AutoMoT/qwen3vl_local/eval_carla/webapp/app.py \
    --eval-base outputs/closed_loop_eval --port 5050
```

打开 `http://<远程ip>:5050`：
- 顶部下拉切换 ckpt signature；tab 切 **Routes** / **Scenarios**
- Routes：左栏按 scenario 分组列 route 与 score_composed，点开右栏切
  input/debug/demo/grid 视频 + leaderboard scores + infractions
- Scenarios：表格展示每个 scenario 平均分

跨 ckpt 比较：直接把 `--eval-base` 指向 `closed_loop_eval`（含多个 signature 子目录）。

---

## 6. 输出目录

```
${EVAL_OUTPUT_BASE}/closed_loop_eval/
  <ckpt_parent>__<ckpt_stem>__bev{0|1}__ema{0|1}/
    config.json                                 ← ckpt / use_bev / 传感器 / 录像开关
    eval_per_route/eval_<route_id>.json         ← leaderboard 原始结果
    route<route_id>/
      input.mp4 debug.mp4 demo.mp4 grid.mp4
      meta/<step>.json                          ← 每个推理触发帧的 pred + 耗时
      logs/
    scenarios/<Scenario>/summary.json           ← aggregate 写入
    summary_all.json
```

`<ckpt_signature>` 由 ckpt 路径、use_bev（从 ckpt 读出来）和
`LEADMOT_USE_EMA` 自动生成，不接受 CLI 覆盖；不同 ckpt / 不同 use_bev /
raw-vs-EMA 永不互相覆盖。

---

## 7. 常见坑

- **首帧 demo 摄像头 spawn 失败**：通常是 ego vehicle 还没注册 `role_name=hero`。
  agent 打印 `hero vehicle not found; demo cameras skipped`，本路线 demo / grid
  会缺，input / debug 不受影响。
- **ffmpeg 不在 PATH**：视频不压缩，留原始 mp4v 编码 mp4，仍可播放但体积大。
- **debug.mp4 在 warmup 阶段是空白**：默认前约 15 个 tick 还没有 pred_waypoints，
  debug 那段不写帧。属于预期。
- **target_point 静止时跳动**：低速时 `MIN_LOOKAHEAD_M=5m` 兜底，避免 tp 退化
  为当前位置导致模型迷茫；如果你看到模型在停车场景里乱打方向，调大这个值。
- **parking_escape 误触**：默认参数 1500 帧（125s）位移 < 5m 才触发，红灯等灯
  正常不会触发。如果你的场景普遍 lights 等待 > 2 分钟，调大 `parking_deadlock_window`
  或在 `safety.py` 里临时关掉。

## 8. 维护注释约定

- 改 `agent.py` 的传感器、坐标系、warmup、target_point 或 LiDAR 融合时，同步更新函数旁中文注释。
- 改 `run_eval.sh` 的 GPU/端口/worker/断点续跑逻辑时，同步更新脚本注释和本文件的多卡说明。
- 改输出 JSON、summary 或 webapp API 时，同步更新 `aggregate.py` / `webapp/app.py` 注释和
  `EVAL_CARLA_PLAN.md` 的结构说明。
- 新增 Python 函数或类时，至少写清楚输入来源、输出去向、与 LEAD/AutoMoT 兼容点；
  新增 shell/前端逻辑时，在关键分支前写中文注释说明目的和边界。
