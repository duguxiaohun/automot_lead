# LEAD_VIDEO_RUN

把远端 LEAD 数据集里的 `rgb/*.jpg` 按 route 转成可拖动进度条、可调倍速播放的 MP4。
默认视频左上角会写 `<view> frame N`，其中 `N` 与正常 LEAD 文件名 `000N.jpg` 对齐，
方便拖进度条后回查原始帧。

本手册默认当前目录就是远端 `AutoMoT/`。工具只读 LEAD 数据，不改原始数据。

依赖：`ffmpeg` 和 `ffprobe` 需要在 PATH 中。脚本用 `ffmpeg` 编码，用 `ffprobe`
做断点续跑完整性检查。

注意：`rgb_to_video.py` 默认不做异常时长筛选，也不会自动读取异常名单；它只按当前筛选范围做普通视频转换与断点续跑检查。只有显式运行 `abnormal_duration_filter.py` / `rgb_to_video.py --abnormal-only`，才会生成异常名单；只有显式传 `--abnormal-route-list-dir`，才会只转异常名单里的 route。

## 1. 数据与输出

默认输入：

```text
/datashare/IOL4SGH/data/data/<Scenario>/<run_id>/rgb/0000.jpg
```

默认输出：

```text
/data/lead_video/<Scenario>/<run_id>/input.mp4
/data/lead_video/<Scenario>/<run_id>/left.mp4       # 仅 --views 包含 left 时生成
/data/lead_video/<Scenario>/<run_id>/front.mp4      # 仅 --views 包含 front 时生成
/data/lead_video/<Scenario>/<run_id>/right.mp4      # 仅 --views 包含 right 时生成
/data/lead_video/<Scenario>/<run_id>/video_meta.json
/data/lead_video/lead_video_summary.json
```

异常时长名单由 `abnormal_duration_filter.py` 单独生成，默认写在工具目录：

```text
lead_video_tools/abnormal_duration_filter/abnormal_possible_90s_to_100s.txt
lead_video_tools/abnormal_duration_filter/abnormal_confirmed_over_100s.txt
lead_video_tools/abnormal_duration_filter/abnormal_duration_summary.json
```

例如：

```text
/datashare/IOL4SGH/data/data/BlockedIntersection/Town06_Rep0_Town06_14_route0_01_08_14_51_15/rgb
```

会生成：

```text
/data/lead_video/BlockedIntersection/Town06_Rep0_Town06_14_route0_01_08_14_51_15/input.mp4
```

LEAD 每 5 个 CARLA tick 落盘一帧，CARLA 为 20Hz，所以默认 `--fps 4.0`，即每帧 0.25s。

## 2. 先看计划

```bash
python3 lead_video_tools/rgb_to_video.py --dry-run --limit 5
```

全量数据第一次进入脚本时会先打印 discover 进度：

```text
[discover] scanning data_root=/datashare/IOL4SGH/data/data
[discover] scenarios=10 routes=...
[discover] done scenarios=... routes=... elapsed=...
```

`discover` 只确认 `<Scenario>/<run_id>/rgb/` 里是否存在 jpg，不在这一步统计所有帧。
帧数、连续性、首尾尺寸检查会进入后面的 `[scan]` 阶段再做，所以全量启动时不应该再长时间
停在没有输出的状态。

只看某个场景 / 某条 run：

```bash
python3 lead_video_tools/rgb_to_video.py \
    --scenario BlockedIntersection \
    --run-id Town06_Rep0_Town06_14_route0_01_08_14_51_15 \
    --dry-run
```

单独扫描异常采集时长名单、不转视频：

```bash
python3 lead_video_tools/abnormal_duration_filter.py
```

按默认 4Hz 换算，`360 <= frames <= 400`（1 分 30 秒到 1 分 40 秒）写入
`abnormal_possible_90s_to_100s.txt`，`frames >= 401`（大于 1 分 40 秒）写入
`abnormal_confirmed_over_100s.txt`。`BlockedIntersection` 和 `ControlLoss` 是筛选白名单，
即使超过阈值也不会写入异常/存疑名单；`Accident` 只对白名单存疑段生效，
即 360-400 帧不会写入存疑名单，但 401+ 帧仍会写入确定异常名单。场景名以
`park` 或 `dynamic` 开头的数据也只在 360-400 帧存疑段白名单内，401+ 帧仍会写入确定异常名单。
`abnormal_duration_summary.json` 保留同一批名单的
帧数、秒数、RGB 路径、视频输出目录和 scan 状态，方便后续脚本继续处理。
两个 txt 名单只保留 `Scenario/run_id`，不写路径，便于人工复制和给
`rgb_to_video.py --abnormal-route-list-dir` 复用。

这个筛选脚本只统计 jpg 数量，不调用 `ffprobe`，也不检查已有视频，所以比
`rgb_to_video.py` 的全局预扫描轻很多。普通 `rgb_to_video.py` 默认不会使用这些名单；筛完后，必须显式传 `--abnormal-route-list-dir` 才会只对筛选目录里的 route 生成视频：

筛选时会先打印 `[filter:discover]` 统计 route 数，再打印 `[filter]` route 级进度条、
elapsed / ETA 和当前候选数量；可用 `--progress-interval N` 调整每隔多少条 route 打印一次。

```bash
python3 lead_video_tools/rgb_to_video.py \
    --abnormal-route-list-dir lead_video_tools/abnormal_duration_filter \
    --abnormal-route-kind all \
    --workers 0
```

只跑“确定异常”（大于 1 分 40 秒）：

```bash
python3 lead_video_tools/rgb_to_video.py \
    --abnormal-route-list-dir lead_video_tools/abnormal_duration_filter \
    --abnormal-route-kind confirmed \
    --workers 0
```

兼容入口：`rgb_to_video.py --abnormal-only` 也会调用同一套轻量筛选逻辑并退出，但推荐日常直接用
`abnormal_duration_filter.py`，语义更清楚。

## 3. 生成视频

推荐远端全量跑法（默认只生成 `input.mp4`，自动 CPU 并行 + 断点续跑）：

```bash
python3 lead_video_tools/rgb_to_video.py \
    --workers 0
```

如果你已经知道某个场景还没生成过，想跳过全量 `[scan]` 预检查、直接进入编码队列：

```bash
python3 lead_video_tools/rgb_to_video.py \
    --scenario Accident \
    --workers 0 \
    --skip-scan
```

`--skip-scan` 仍会在每条 route 真正编码前做单条检查；它只是省掉开跑前那次全量
already_done / excluded / to_run 统计。

默认编码参数偏向“快速浏览”：`--preset veryfast --crf 18 --ffmpeg-threads 1`。如果你更在意速度、
可以接受视频文件变大，可进一步用：

```bash
python3 lead_video_tools/rgb_to_video.py \
    --workers 0 \
    --preset ultrafast
```

单条示例：

```bash
python3 lead_video_tools/rgb_to_video.py \
    --scenario BlockedIntersection \
    --run-id Town06_Rep0_Town06_14_route0_01_08_14_51_15
```

指定场景 demo：

```bash
python3 lead_video_tools/rgb_to_video.py \
    --scenario Accident \
    --workers 0
```

指定场景 + 跳过 scan demo：

```bash
python3 lead_video_tools/rgb_to_video.py \
    --scenario Accident \
    --workers 0 \
    --skip-scan
```

全量转换：

```bash
python3 lead_video_tools/rgb_to_video.py
```

只有需要单独看左右/前视裁剪时，再生成三视角拆分视频：

```bash
python3 lead_video_tools/rgb_to_video.py \
    --views input,left,front,right
```

其中：

- `input.mp4`：原始 1152×384 stitched RGB；
- `left.mp4`：左 384×384；
- `front.mp4`：中间前视 384×384；
- `right.mp4`：右 384×384。

共享存储接近满盘时建议先按场景分批跑：

```bash
python3 lead_video_tools/rgb_to_video.py --scenario BlockedIntersection
```

## 4. CPU 并行

脚本支持按 route 并行转换：每个 worker 处理一条 route，并启动一个独立 `ffmpeg`
进程。这样可以吃到多核 CPU，但也会同时读取 `/datashare`、写 `/data/lead_video`，
所以不要在共享存储压力很高或剩余空间很紧时开太大。

自动按 CPU 估计并行数：

```bash
python3 lead_video_tools/rgb_to_video.py --workers 0
```

`--workers 0` 会取 CPU 核数的一半、最多 16 个 worker，并且不超过待处理 route 数。
这是比较稳的默认加速档。

手动指定并行数：

```bash
python3 lead_video_tools/rgb_to_video.py --workers 4
```

默认只生成一路 `input.mp4`；如果显式 `--views input,left,front,right`，单条 route 会依次
编码四路视角，route 之间仍由 `--workers` 并行。

并行加速细节：

- 默认 `--ffmpeg-threads 1`，每个 ffmpeg 只用 1 个线程，把 CPU 主要留给 route 级并行；
- `--workers 0` 自动取 CPU 核数一半、最多 16 个 worker；
- 如果只跑少量 route，想让单条 route 编码更快，可以试 `--ffmpeg-threads 0`，让 ffmpeg 自己开线程；
- 如果共享盘 I/O 压力很大，手动降到 `--workers 4` 往往比盲目开大更稳。

## 5. 断点续跑与检查

默认启用断点续跑：

- 如果目标视频存在、`video_meta.json` 里的 views / frame-index 配置匹配，且 `ffprobe` 检查帧数或时长合理，跳过。
- 如果视频缺失、为空、帧数/时长明显不对，重新生成。
- 每条 route 写 `video_meta.json`，全局写 `lead_video_summary.json`。

每次正式编码前都会先打印本次计划：

```text
[plan] total=220 already_done=152 excluded=3 to_run=65
```

含义：

- `total`：本次筛选范围内的 route 总数；
- `already_done`：断点续跑检查通过，本次直接跳过；
- `excluded`：异常数据剔除，不生成视频；
- `to_run`：本次还需要真正编码的视频 route 数。

编码阶段会实时打印 route 级进度条：

```text
[progress] [#########-------------------] 21/65 ( 32.3%) elapsed=180.4s eta=378.0s converted Accident/...
```

进度条只统计 `to_run`，不把已经跳过的 `already_done` 算进去。

注意：脚本现在有两段进度：

- `[discover]`：快速发现 route，只检查每个 `rgb` 目录是否至少有 jpg，不统计全量帧；
- `[scan]`：断点续跑 / 异常数据预扫描进度；已有 `video_meta.json` 且视频完整的 route 会走
  manifest 快速跳过，不再扫描原始 jpg；
- `[progress]`：真正 ffmpeg 编码进度。

如果全量数据很多，先看到 `[discover]` 和 `[scan]` 是正常的，不是卡住。

跳过全局 scan：

```bash
python3 lead_video_tools/rgb_to_video.py --scenario Accident --workers 0 --skip-scan
```

跳过后计划行会变成：

```text
[plan] total=220 already_done=unknown excluded=unknown to_run=220 scan=skipped
```

这表示脚本不再提前统计哪些已经完成、哪些异常，而是把 discover 到的 route 直接提交给
worker。每条 route 进入 worker 后仍会执行：

- 已有完整视频则 `skipped`；
- RGB 异常则 `excluded`；
- 缺视频或视频不完整则 `converted`。

适合场景：第一次跑某个 scenario，或者你确认大部分 route 都需要生成。  
不适合场景：大部分视频已经生成好了，此时默认 scan 可以更早跳过，反而更省 worker 调度。

异常数据剔除：

- 默认要求 `rgb` 帧文件名是连续数字序列：`0000.jpg ... 00NN.jpg`。
- 默认要求至少 `--min-frames 2` 帧。
- 默认用 `ffprobe` 检查首尾帧可读、尺寸一致，且 stitched 宽度能被 3 整除。
- 不满足的 route 标记为 `excluded`，不会生成错位视频。
- 如果确实要容忍缺帧或非连续命名，可显式传 `--allow-noncontiguous`。

强制重做：

```bash
python3 lead_video_tools/rgb_to_video.py --scenario BlockedIntersection --overwrite
```

关闭帧号 overlay：

```bash
python3 lead_video_tools/rgb_to_video.py --no-frame-index
```

## 6. 视频路数说明

`eval_carla` 闭环评测能录五路：

| 文件 | 是否能从离线 LEAD 原始数据直接生成 | 原因 |
|---|---|---|
| `input.mp4` | 可以 | 直接来自 `rgb/*.jpg` 三视角拼接图 |
| `left/front/right.mp4` | 可以 | 从 stitched RGB 按宽度三等分裁剪 |
| `debug.mp4` | 不完整 | 需要模型预测 `pred_waypoints` 和相机投影 |
| `bev_debug.mp4` | 不完整 | 需要 LiDAR + 预测 route/waypoints + tp/ntp |
| `demo.mp4` | 不支持 | 需要 CARLA 在线 spawn cinematic / BEV camera |
| `grid.mp4` | 不支持 | 依赖 demo camera |

所以这个工具默认只生成最有用、最省空间的 `input.mp4`。`left/front/right` 只是从同一张
stitched RGB 裁出来的重复视角，需要时用 `--views input,left,front,right` 显式打开。
后续如果要加离线 BEV/标注 overlay，应另接 `metas/*.pkl`、`lidar/*.laz` 和模型预测结果。

## 7. 播放

输出为 H.264 `yuv420p` MP4，并带 `+faststart` 元数据。浏览器、VLC、mpv 都可以拖动进度条；
倍速可以用播放器自带的 playback speed / playback rate 控件调节。

## 8. 代码结构

`rgb_to_video.py` 已补中文注释，主要函数分工如下：

| 函数 / 数据结构 | 作用 |
|---|---|
| `RouteTask` | 描述一条 route 的输入 RGB 目录与输出视频目录 |
| `ConvertResult` | 记录一条 route 的最终状态，写入 `lead_video_summary.json` |
| `PlanItem` / `PlanSummary` | 正式编码前的断点续跑计划：already_done / excluded / to_run |
| `discover_routes()` / `build_tasks()` | 快速扫描 `<Scenario>/<run_id>/rgb/` 是否存在 jpg 并构造任务，不在 discover 阶段统计全量帧 |
| `validate_rgb_sequence()` | 剔除缺帧、非连续编号、首尾不可读、尺寸不一致等异常数据 |
| `is_video_complete()` | 用 `video_meta.json` + `ffprobe` 判断旧视频能否断点跳过 |
| `build_resume_plan()` | 开跑前统计本次 total / already_done / excluded / to_run |
| `--skip-scan` | 跳过 `build_resume_plan()` 的全局预扫描，直接把 discover 到的 route 交给 worker |
| `_view_filter()` | 生成 input/left/front/right 的裁剪与 frame id overlay filter |
| `_encode_one_view()` | 调 ffmpeg 编码单个 view，先写临时文件再原子替换 |
| `convert_route()` | 转换一条 route 的多个 view，并写 `video_meta.json` |
| `print_progress()` | 打印 scan / encode 阶段的文本进度条、elapsed 与 ETA |

维护原则：

- 不引入 OpenCV/PIL 作为运行依赖；图片尺寸检查用 `ffprobe`。
- 断点续跑必须同时检查视频和 `video_meta.json`，防止换了 `--views` 或帧号 overlay 配置后误跳过。
- 对 `/datashare` 和 `/data/lead_video` 都要温柔一点；全量跑优先用 `--workers 0`，不要盲目开很大。

