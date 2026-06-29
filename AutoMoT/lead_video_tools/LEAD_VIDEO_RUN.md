# LEAD_VIDEO_RUN

把远端 LEAD 数据集里的 `rgb/*.jpg` 按 route 转成可拖动进度条、可调倍速播放的 MP4。
默认视频左上角会写 `<view> frame N`，其中 `N` 与正常 LEAD 文件名 `000N.jpg` 对齐，
方便拖进度条后回查原始帧。

本手册默认当前目录就是远端 `AutoMoT/`。工具只读 LEAD 数据，不改原始数据。

依赖：`ffmpeg` 和 `ffprobe` 需要在 PATH 中。脚本用 `ffmpeg` 编码，用 `ffprobe`
做断点续跑完整性检查。

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

只看某个场景 / 某条 run：

```bash
python3 lead_video_tools/rgb_to_video.py \
    --scenario BlockedIntersection \
    --run-id Town06_Rep0_Town06_14_route0_01_08_14_51_15 \
    --dry-run
```

## 3. 生成视频

推荐远端全量跑法（自动 CPU 并行 + 四路 RGB 视角 + 断点续跑）：

```bash
python3 lead_video_tools/rgb_to_video.py \
    --views input,left,front,right \
    --workers 0
```

单条示例：

```bash
python3 lead_video_tools/rgb_to_video.py \
    --scenario BlockedIntersection \
    --run-id Town06_Rep0_Town06_14_route0_01_08_14_51_15
```

全量转换：

```bash
python3 lead_video_tools/rgb_to_video.py
```

生成三视角拆分视频：

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

`--workers 0` 会取 CPU 核数的一半、最多 8 个 worker，并且不超过待处理 route 数。
这是比较稳的默认加速档。

手动指定并行数：

```bash
python3 lead_video_tools/rgb_to_video.py --workers 4
```

如果同时生成四路视角，单条 route 会依次编码 `input/left/front/right`；route 之间仍由
`--workers` 并行。

## 5. 断点续跑与检查

默认启用断点续跑：

- 如果目标视频存在、`video_meta.json` 里的 views / frame-index 配置匹配，且 `ffprobe` 检查帧数或时长合理，跳过。
- 如果视频缺失、为空、帧数/时长明显不对，重新生成。
- 每条 route 写 `video_meta.json`，全局写 `lead_video_summary.json`。

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

所以这个工具支持 raw RGB 能可靠生成的 `input/left/front/right`。后续如果要加离线
BEV/标注 overlay，应另接 `metas/*.pkl`、`lidar/*.laz` 和模型预测结果。

## 7. 播放

输出为 H.264 `yuv420p` MP4，并带 `+faststart` 元数据。浏览器、VLC、mpv 都可以拖动进度条；
倍速可以用播放器自带的 playback speed / playback rate 控件调节。
