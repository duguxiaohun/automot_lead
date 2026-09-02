# EVAL_CARLA_RUN

LeadMoT 闭环评测一键操作手册。架构与对齐细节见 [`EVAL_CARLA_PLAN.md`](EVAL_CARLA_PLAN.md)。

本手册默认当前目录就是远端 `AutoMoT/`。下面命令都写相对 `AutoMoT/` 的路径，
例如 `bash qwen3vl_local/eval_carla/run_eval.sh`，不再额外写切目录步骤。

---

## 0. 前置

- 远程已装 CARLA（`$CARLA_ROOT` 或 `~/carla` 自动探测）
- LeadMoT decoder checkpoint 就位（`qwen3vl_local/leadmot/train.py` 产物）
- LEAD route XML 就位：`data/lead/<Scenario>/<Town>_<route_key>.xml`，旧数字 route
  形如 `data/lead/Accident/Town03_route_001783.xml`
  - 命名规范固定为 `data/lead/<Scenario>/<Town>_<route_key>.xml`：旧数字 route 用
    `Town03_route_001783.xml`，新版子编号用 `Town12_route_1054_0.xml`，命名本身带
    Town 的 legacy key 用 `Town06_route_Town06_13.xml`，legacy key 内部带 route 编号时
    保留完整 key，如 `Town12_route_Town12_route15.xml`
  - 从 run 目录解析 route_key 时只剥采集尾缀 `_route0`，不能剥 `Town12_route15`
    本体里的 `route15`
  - 2026-07-03 全量核对：`lead_data` 9715 个 run 去重后 9294 个
    `(Scenario,Town,route_key)`，`data/lead` 正好 9294 个 XML，缺失 0、冗余 0、
    命名不规范 0、XML 解析失败 0、内容结构异常 0；40 个 XML 的 `data_routes`
    源在其它 scenario 目录，不是缺失；`ParkedObstacle/Town12_route_Town12_route15.xml`
    覆盖有效，不能当作 `xml_available=false`
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

- 每个 worker 的 stdout/stderr 只写到 `/tmp/leadmot_eval_workers.*` 临时目录
- 主进程 fork 一个 `tail -F` 把 worker log 行内容实时输出到终端；退出时临时目录自动删除
- 单 worker 直接 follow worker0.log，多 worker 时 tail 自带 `==> path <==` 头便于区分
- 主进程终端输出会同步追加到 `runs/<RUN_LABEL>/log.txt`，方便跑完后回看完整 stdout/stderr
- 有用的运行状态会写回 `runs/<RUN_LABEL>/run_manifest.json`：`finished_at`、
  `attempted_count`、`failed_routes`、`worker_fail`
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

如果 LeadMoT checkpoint 是用 `QWEN_ADAPTER_DIR` 训练的，agent 会从 checkpoint 的
`qwen_backbone` 自动恢复同一份 LoRA，并在模型加载前校验 adapter SHA256。adapter 目录
搬家后，在原命令前加 `QWEN_ADAPTER_DIR=checkpoints/new/path/to/adapter`；哈希不一致会
直接中止。旧 checkpoint 没有该合同，只允许 base Qwen，不能在闭环时临时挂 LoRA。

### 1.1 全量 220 路线（默认）

推荐写法（全量 + 只录 input/bev_debug + 默认开断点续跑）：

```bash
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt checkpoints/leadmot_v1_decoder/latest/best.pt \
    --minimal-videos
```

多卡并行（默认按显存占用最低自动选 GPU）：

```bash
bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt checkpoints/leadmot_v1_decoder/latest/best.pt \
    --minimal-videos --num-gpus 4
```

显式指定哪几张 GPU（与项目 SFT / GoalGen / LeadMoT / VAE 训练入口规则一致：
`GPU_IDS` env 非空时跳过 `nvidia-smi` 自动选址）：

```bash
# 单卡 pin 到 GPU 0
GPU_IDS=0 bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt checkpoints/leadmot_v1_decoder/latest/best.pt \
    --minimal-videos

# 4 卡 pin 到 GPU 0,1,2,3
GPU_IDS=0,1,2,3 bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt checkpoints/leadmot_v1_decoder/latest/best.pt \
    --minimal-videos

# 2 卡 pin 到 GPU 2,5（worker 数自动跟 GPU_IDS 长度走；--num-gpus 可省）
GPU_IDS=2,5 bash qwen3vl_local/eval_carla/run_eval.sh \
    --leadmot-ckpt checkpoints/leadmot_v1_decoder/latest/best.pt \
    --minimal-videos
```

`GPU_IDS` 支持逗号或空格分隔；若同时传 `--num-gpus N` 且 `N` 与 `GPU_IDS` 数量不同，
以 `GPU_IDS` 数量为准并打印 `Note: --num-gpus=N overridden by GPU_IDS=...`。
CARLA 端口由 launcher 从 `PORT_BASE_START` 起按 `PORT_STRIDE` 扫描空闲
`[rpc..rpc+3, tm]` 端口块。

**断点续跑（默认开启）**：每次启动会先扫 `eval_per_route/eval_<id>.json`：

- `status` 是 leaderboard 终态(Completed / Perfect / Failed - ...) **且** `score_composed` 非空 → 该 route 视为已完成，跳过
- 文件存在但解析失败 / 无 status / 无 score → **删 `eval_<id>.json` + `eval_latest_<id>.json` + 整个 `route<id>/` 目录**，加入本次待跑队列
- 文件不存在 → 加入本次待跑队列；如果残留了 `eval_latest_<id>.json` 或 `route<id>/`，也会先清掉再跑

启动 banner 会打印：

```
Picker routes   : 220
Resume          : enabled
  already done  : 152
  partial cleaned: 3
  to run        : 68
```

跑的过程中每条 route 完成后主进程打印一行：

```
[done 17/68] route_id=1773 status=Completed
```

`k` 是本次启动跨 worker 累计完成数，`Y` 是本次实际下发给 worker 的 `to run` 数；
默认 resume 跳过的已完成 route 不进入这个计数。不想续跑（整张表强制重跑）传
`--no-resume`；这种情况下会先清理 picker 范围内旧的 `eval_<id>.json`、
`eval_latest_<id>.json` 和 `route<id>/`，然后 `to run = Picker routes`。

`--minimal-videos` 等价 `--no-debug --no-demo --no-grid`，只保留 `input.mp4` 和
`bev_debug.mp4`，跑全量时强烈推荐（省 ffmpeg / 磁盘 / 录像 spawn 开销）。

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
# 单条 + 烟雾测试（也会写 smoke_<route_id>/summary_report.md）
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
    --minimal-videos
# 等价于 --no-debug --no-demo --no-grid
```

---

## 3. 推理 / 传感器 / target_point 配置

| 标志 / env | 默认 | 说明 |
|---|---|---|
| `--step-stride 5` | 5 | 每多少 tick 调一次模型；4Hz 与训练分布一致 |
| `--num-gpus N` | 1 | 自动选 N 张空闲 GPU；也可用 `EVAL_GPU_COUNT=N` |
| env `GPU_IDS` | unset | 显式指定 GPU 卡号（逗号或空格分隔，如 `0,1,2,3`）；非空时跳过 nvidia-smi 自动选址，`GPU_COUNT` 强制取 `GPU_IDS` 数量 |
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
| `--minimal-videos` | off | 等价 `--no-debug --no-demo --no-grid`，只录 input + bev_debug |
| `--no-resume` | off | 关闭断点续跑：清掉 picker 范围旧 eval/video 产物，整张 picker 列表强制重跑 |

在线 target_point / next_target_point 对齐训练侧 route lookahead：每 tick 调
`RoutePlanner.run_step()` 推进 route 后，按 `max(speed*lookahead_s, MIN_LOOKAHEAD_M)`
沿剩余 route 弧长插值；默认 tp=1.0s、ntp=2.0s。final_goal 来自
`scenario_picker` 按 route_id 读取的 LEAD route XML 最后一个 waypoint，再转成
当前 ego frame；这与训练侧 `meta["next_target_points"][-1]` 的剩余 route 终点语义一致。

---

## 4. 输出目录

```
${EVAL_OUTPUT_BASE:-outputs/closed_loop_eval}/
  <ckpt_parent>__<ckpt_stem>__bev{0|1}__ema{0|1}/
    config.json                              ← ckpt / use_bev / 传感器 / 录像 / env 设定
    eval_per_route/eval_<route_id>.json      ← leaderboard 原始结果（跨跑法共享）
    route<route_id>/                          ← route 级输出（跨跑法共享，断点续跑）
      input.mp4 debug.mp4 bev_debug.mp4 demo.mp4 grid.mp4
      meta/<step>.json                        ← 每个推理 tick 的 pred + 耗时 + 输入统计（speed/tp/ntp/final_goal）
    runs/<RUN_LABEL>/                          ← 本次启动的聚合结果（与其他批次隔离）
      log.txt                                  ← 本次终端 stdout/stderr 追加日志
      scenarios/<Scenario>/summary.json
      run_manifest.json                        ← 本批次 route_id 列表 + 启动/完成参数 + 失败 route + resume 拆分（done/cleaned/to_run）
      summary_all.json                         ← 完整机器可读结果（含 route/scenario 明细）
      summary_report.md                        ← 人类可读总结：测试数、成功率、分数、指标解释
      scenario_table.csv                       ← scenario 级论文表格友好汇总
      route_results.csv                        ← route 级状态/分数/违规明细
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

## 5. 聚合单独跑 / 跑完没看到 summary 的补救

`aggregate.py` 是独立 CLI：**只读 `eval_per_route/eval_<route_id>.json` 这堆
leaderboard 原始 JSON，不碰 CARLA、不重跑 route**，所以跑完 eval 后任何时刻
都能离线补一份 summary。常见触发场景：

- 之前传了 `--no-aggregate`，`runs/<RUN_LABEL>/` 里只有 `run_manifest.json` + `log.txt`
- 聚合阶段失败（例如 data/lead 路径不对），4 件套没生成
- 想跨多次启动合并出一张总表（`__all__` 跨批次聚合）
- 想换一份 `--benchmark-root` 重算 scenario 反查

### 5.1 先确认原始结果在不在

```bash
# 找到 ckpt 对应的 signature 目录
ls outputs/closed_loop_eval/
# 应该有形如 leadmot_v1_decoder__best__bev0__ema1/ 的子目录

# 看里面是否有 leaderboard 原始 JSON
ls outputs/closed_loop_eval/<signature>/eval_per_route/
# 应有 eval_<route_id>.json；只要这堆在，summary 一定能补
```

如果 `eval_per_route/` 里一条 JSON 都没有，说明 route 根本没跑通，需要重跑
`run_eval.sh`，不是聚合的问题。

### 5.2 三种补救调用

```bash
# (a) 补某一批：只聚合 runs/<RUN_LABEL>/run_manifest.json 列出的 route
python3 -m qwen3vl_local.eval_carla.aggregate \
    --eval-base outputs/closed_loop_eval \
    --leadmot-ckpt checkpoints/leadmot_v1_decoder/latest/best.pt \
    --benchmark-root data/lead \
    --run-label full

# (b) 补该 signature 下所有批：枚举 runs/* 各聚合一次 + 跨批次总聚合
python3 -m qwen3vl_local.eval_carla.aggregate \
    --eval-base outputs/closed_loop_eval \
    --leadmot-ckpt checkpoints/leadmot_v1_decoder/latest/best.pt \
    --benchmark-root data/lead
# 跨批次总聚合落在 signature 根目录（run_label=__all__），合并所有
# eval_per_route/*.json，coverage 恒为 1.0，markdown 头部会写明这一点

# (c) 没指定 ckpt：聚合 EVAL_OUTPUT_BASE 下所有 signature 目录
python3 -m qwen3vl_local.eval_carla.aggregate \
    --eval-base outputs/closed_loop_eval \
    --benchmark-root data/lead
```

⚠ `--benchmark-root` 不传也能跑，但每条 route 的 `scenarios` 字段会塌成
`__unknown__`，`scenario_table.csv` 只会剩一行，论文表格基本没法用。**补救
聚合时强烈建议传**，从 `AutoMoT/` 当前目录看路径就是
`data/lead`。

⚠ `--leadmot-ckpt` 支持文件或目录，目录会按 `best.pt` / `latest/best.pt` 等
顺序自动解析，和 `run_eval.sh` 完全一致。

### 5.3 输出文件

聚合会在对应目录下同时写四类总结文件：

| 文件 | 用途 |
|---|---|
| `summary_report.md` | 最适合直接读或贴到实验记录：包含本次计划测多少、实际生成多少、成功率、每个 scenario 表格和指标解释 |
| `scenario_table.csv` | 每个 scenario 一行：planned/evaluated/coverage/success_rate/score/违规数，适合论文表格或 Excel |
| `route_results.csv` | 每条 route 一行：status、score、违规类型计数，适合定位失败和低分路线 |
| `summary_all.json` | 完整机器可读聚合，webapp 和后续脚本继续使用 |

核心口径：

| 指标 | 含义 |
|---|---|
| `planned_routes` | 本批计划评测路线数，来自 `run_manifest.json` |
| `evaluated_routes` | 已生成并成功读取 `eval_<route_id>.json` 的路线数 |
| `coverage` | `evaluated_routes / planned_routes`，表示本批实际完成落盘比例 |
| `success_rate` | status 为 `Completed` 或 `Perfect` 的路线数 / `planned_routes`；缺失 JSON 的路线计为未成功 |
| `perfect_rate` | status 为 `Perfect` 的路线数 / `planned_routes` |
| `score_composed` | leaderboard 主分数，约等于路线完成度分数乘违规惩罚分数 |
| `score_route` | 路线完成度分数 |
| `score_penalty` | 违规惩罚分数，碰撞、闯灯、偏航等会降低它 |

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

- **CARLA 启动超时**：现在由 `leaderboard_evaluator.py` 启动并等待 CARLA。运行中看终端
  tail 输出里的 `Launch CARLA`、`load_world failed` 和 evaluator traceback；跑完后只保留
  `run_manifest.json` 里的失败 route 状态，不长期保存 worker log。
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
- **use_subgoal=True ckpt 报 `NotImplementedError`**：闭环 agent 拿不到未来 SUBGOAL
  keyframe RGB，所以 `decoder_config.use_subgoal=True` 的 ckpt **当前不支持闭环**，
  agent 加载时直接抛错并提示。对应离线训练/eval/probe 走
  `leadmot/train.py --use-subgoal` + `mot_lead_offline_runner.py`；闭环跑请改用
  `USE_SUBGOAL=0` 训出来的 ckpt。后续若有 SUBGOAL 图像生成 / 代理输入，会在
  `agent.py` 的 `TODO(subgoal)` 处接入。
