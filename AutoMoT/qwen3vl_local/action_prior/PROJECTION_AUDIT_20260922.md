# 2026-09-22 离线轨迹投影与 LEAD 源码核对

`LEAD_VIDEO_RUN.md` 的三视角是同帧相机的横向拼接与裁剪，不是多视角重建或生成高空图。对比工具对每个相机使用其自己的外参和内参投影 GT / 各模型轨迹，再放到同一张结果图中；不会把 1152×384 整图当作一台相机。

## 已核对的原始链路

| 只读参考 | 结论及对比工具采用方式 |
|---|---|
| [LEAD_VIDEO_RUN.md](../../lead_video_tools/LEAD_VIDEO_RUN.md)、[rgb_to_video.py](../../lead_video_tools/rgb_to_video.py) `_view_filter` | 原图1152×384；left/front/right各裁384×384。没有几何融合或轨迹投影 |
| [base_agent.py](../../../lead/lead/common/base_agent.py) `tick` 的相机处理 | 按 `rgb_1, rgb_2, rgb_3` 递增顺序 `concatenate(axis=1)`；对应当前三相机配置的左、前、右 |
| [sensor_setup.py](../../../lead/lead/common/sensor_setup.py) 相机创建、[config_base.py](../../../lead/lead/common/config_base.py) `camera_calibration` | 传感器使用配置中的实际位置、roll/pitch/yaw、尺寸和水平FOV |
| [expert.py](../../../lead/lead/expert/expert.py) meta写入 | `sensor_information.camera_calibration` 保存录制时标定；新增优先读取同anchor字段，按传感器1/2/3与实际RGB分块对应 |
| [expert_data.py](../../../lead/lead/expert/expert_data.py) future_positions写入 | `inverse(ego_matrix) @ future_world_position`，坐标为CARLA ego：x前、y右、z上 |
| [video_recorder.py](../../../lead/lead/inference/video_recorder.py) `draw_waypoints` / grid输入处理 | 可参考“每个相机独立投影、只显示可见轨迹”的流程；其旧顺序/旋转实现不能直接移植到离线录制标定 |
| [Bench2Drive/tools/utils.py](../../../lead/3rd_party/Bench2Drive/tools/utils.py) `get_matrix` / `build_projection_matrix` / `get_image_point` | 采用其CARLA逆外参、`(x,y,z)→(y,-z,x)`及水平FOV内参方法，已通过只读函数数值交叉检查 |
| [plant_visualizer.py](../../../lead/lead/plant/plant_visualizer.py) `visualize_plant_bev` | 俯视绘制使用 `u=origin_x+y*ppm`、`v=origin_y-x*ppm`，GT与预测同坐标 |
| [chauffeurnet.py](../../../lead/lead/expert/hdmap/chauffeurnet.py) `_get_warp_transform`、`hdmap_classes` | 录制道路图按米制变换生成，存盘顺时针转90°；对比工具反向旋回并按采集比例与像素中心映射显示 |

## 为什么没有直接调用旧 project_points_to_image

`lead/lead/common/common_utils.py:project_points_to_image` 是有价值的流程参考，但其源码使用正向标准欧拉旋转处理相机坐标，并把FOV解释为垂直角。`video_recorder.py` 的三视角又使用 `[3,2,1]` 标定映射，与录制端 `rgb_1,2,3` 的顺序不同。两者在部分零roll/pitch、方形图场景可能有符号抵消，不能据此直接套入实际保存的相机标定。

数值检查：对左/右相机分别取沿自身水平朝向、距相机10m的地面点，正确投影均为 `(192, 266.824595)`；直接把同一实际标定传入旧投影函数会将该点判断为相机后方。前视零旋转时两者一致。非方形相机还需保证 `fx=fy`，不能按图像高宽分别计算出不同焦距。

当前计算：

```text
T_camera_to_ego = CARLA相机位姿矩阵
p_camera       = inverse(T_camera_to_ego) @ [x, y, ground_z, 1]
p_optical      = [p_camera.y, -p_camera.z, p_camera.x]
f              = image_width / (2*tan(horizontal_fov/2))
u              = image_width/2  + f*p_optical.x/p_optical.z
v              = image_height/2 + f*p_optical.y/p_optical.z
```

点投影拒绝深度≤0.1m的点；线段绘制对深度0.1m近平面及四条视野边界做精确3D裁剪，保留跨边界可见部分，不依赖加密采样。离开视野或进入相机后方后再出现的部分以断点分开，不跨不可见区间连线；不平滑预测、不修改指标。所有方法和GT使用同一份相机标定及地面高度。

## 现在的标定优先级

1. route下显式 `comparison_camera.json`。
2. 显式 `--camera-config` 全局JSON。
3. 同anchor `metas/NNNN.pkl` 的 `sensor_information.camera_calibration`。
4. 只有旧数据缺该字段/缺meta时，回退内置LEAD三相机名义标定，并在case JSON记录回退原因。

录制标定存在但坏数据、非三相机或涉及未支持的裁剪时会报错，不静默换默认值。每例 `visualization_calibration.json` 保存完整有效配置，`case.json.visualization`记录来源、meta/配置哈希及回退状态。此次只修改新增对比模块，未修改只读LEAD文件及已有训练源文件。

## 没有高空RGB，也可以有几何BEV

本轮复用已曝光的train-only路线 `Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46`（先通过异常时长过滤），检查第39帧：有 `rgb`、`lidar`、`metas`、`hdmap`、`bboxes`，没有 `3rd_person`；道路图256×256，同帧21个目标框，录制三相机标定与原默认值一致。

因此该样本可显示：

- 真实左/前/右RGB上的轨迹投影。
- `hdmap`道路、车道线与`bboxes`目标框组成的米制俯视图，直接绘制GT/预测。
- 缺道路图时用LiDAR或米制坐标背景。

这不证明所有route均有相同素材。真实高空RGB必须来自已保存的第三人称相机及标定；没有这些文件时不能从简单拼接得到高空摄影效果。对比工具也未进行深度重建或RGB逆透视拼接。

## 验证及边界

新增10项回归，包含每帧录制标定/顺序、显式覆盖、坏标定拒绝，以及四组姿态/尺寸（左右相机、俯视、带roll/pitch的非方形相机）各300个测试点与Bench2Drive独立逆矩阵投影核对；可投影点 `rtol=1e-10, atol=1e-8`，后方点断开。累计相关测试75项通过。

真实录制样本只做GT绘图与标定检查，图中复画GT明确标为 `GT replay ONLY (no checkpoint inference)`，没有真实checkpoint或GPU推理。RGB仍是统一地面平面近似，不模拟遮挡；坡度、车辆俯仰及远端遮挡可能导致视觉贴地偏差。道路/目标标注只作为显示背景，不加入模型条件。

### 边界与来源补齐

继续补充7项回归，累计82项通过：已知非三相机但缺标定字段、非法meta/sensor_information均拒绝，不回退三相机默认值；精确线段裁剪覆盖两端均在画面外但中间穿过画面、跨近平面、完全不可见和不连续可见区间。
图注显示recorded/explicit/nominal标定来源；REPORT和manifest按去重case统计来源，event/action重复归档不重复计数。测试包含合成端到端渲染，真实多GPU模型推理尚未验收。
