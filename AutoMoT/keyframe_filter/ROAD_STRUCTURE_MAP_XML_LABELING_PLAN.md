# 基于 CARLA 地图 + Route XML + LEAD Meta 的 ROAD_STRUCTURE 帧级标注方案

## 1. 背景与目标

你当前的问题很明确：仅靠 VLM（Qwen）看图判断每帧属于哪个场景/道路结构，稳定性不足，尤其在场景切换边界和视觉歧义帧上容易漂移。

本方案目标是把 `ROAD_STRUCTURE` 的主判断从“视觉语义”切到“几何与规则语义”，即：

1. 用每帧自车全局坐标（`metas/*.pkl`）确定自车在地图中的位置与拓扑关系。
2. 用 `data/data_routes/lead/<Scenario>/route_xxxxxx.xml` 提供 route 先验和 scenario 触发信息。
3. 用 meta 中的 scenario 距离字段与激活字段，提供“事件区间”证据（尤其是 R2/R6 这类局部触发结构）。
4. 产出每帧 `ROAD_STRUCTURE`（R1~R6）和可解释的置信来源，作为后续 EVENT 标注与 Qwen 候选裁剪的上游输入。

一句话：把 Qwen 从“主裁判”降级为“补充裁判”，先用地图与轨迹给出强规则主标签。

---

## 2. 输入数据源规范

本节对三类输入的**文件位置、格式、可用字段及用途**逐一说明，以便没有实际数据的读者也能理解后续判定逻辑。

---

### 2.1 Route XML（路线定义文件）

**位置：** `data/data_routes/lead/<Scenario>/route_<六位编号>.xml`

**对应关系：** run 文件夹名包含 route 编号（如 `Town05_Rep0_route_001761_...`），可提取数字与 XML 文件名对应。

**完整结构示例（Accident）：**

```xml
<routes>
  <route id="route_001761" town="Town05">
    <weathers>
      <weather route_percentage="0" cloudiness="10.0" precipitation="0.0"
               sun_altitude_angle="45.0" fog_density="3.0" ... />
    </weathers>
    <waypoints>
      <position x="210.74" y="-11.27" z="0.0" />
      <position x="210.62" y="-21.27" z="0.0" />
      <!-- 约 10~30 个稀疏采样点，间隔约 10m -->
    </waypoints>
    <scenarios>
      <scenario name="Accident_13" type="Accident">
        <direction value="right" />
        <trigger_point x="210.68" y="-16.27" z="0.00" yaw="-90.7" />
      </scenario>
    </scenarios>
  </route>
</routes>
```

**完整结构示例（ParkingExit，含更多 trigger 参数）：**

```xml
<scenario name="ParkingExit_1" type="ParkingExit">
  <trigger_point x="222.4" y="183.1" z="2.0" yaw="317.17" />
  <direction value="right" />
  <front_vehicle_distance value="9" />   <!-- 障碍物前方距离（米） -->
  <behind_vehicle_distance value="9" />  <!-- 障碍物后方距离（米） -->
</scenario>
```

**字段用途说明：**

| 字段 | 含义 | 用途 |
|------|------|------|
| `route.id` | 路线编号 | 与 results.json.route_id 对齐，校验 run 匹配 |
| `route.town` | 地图名 | 加载对应 xodr；与 meta.town 交叉校验 |
| `waypoints/position` | 稀疏轨迹点（x,y,z） | 构造 polyline，计算每帧 s（沿线弧长进度） |
| `scenarios.scenario.type` | 场景类型字符串 | 决定该 run 所属 scenario，进而决定 R2/R3/R6 候选 |
| `scenarios.scenario.trigger_point` | 仿真触发的地理坐标（x,y,z,yaw） | 映射到路线弧长 `s_trigger`，构造"场景激活候选窗口" |
| `trigger_point.yaw` | 触发时自车航向 | 可辅助判断触发方向（左/右绕行等） |
| `direction` / `front_vehicle_distance` 等 | scenario 特定参数 | 了解障碍物布置，辅助计算可见距离 |

**重要约束：** `trigger_point` 只是 CARLA 仿真系统开始生成 NPC actor 的地理锚点，**不代表自车此时已在 RGB 中看到事件对象**。事件标签注入必须另加可见性门控（见第 6 节）。

---

### 2.2 每帧 Meta 文件（主要帧级信号）

**位置：** `<run_dir>/metas/<帧编号4位>.pkl`，如 `0000.pkl`、`0087.pkl`

**读取方式：** xz 压缩 pickle，必须用 `lzma.open(path, 'rb')` 解包，每帧约 6KB（压缩后），解包后是 `dict`，约 350 个键。

**核心字段分类：**

#### 自车位姿

| 字段 | 类型 | 说明 | 用途 |
|------|------|------|------|
| `ego_matrix` | 4×4 numpy array | 世界坐标系→自车坐标系的变换矩阵 | `[0][3]=x, [1][3]=y, [2][3]=z`，即自车世界坐标 |
| `theta` | float（弧度） | 自车航向角（偏航） | 辅助判断朝向 |
| `speed` | float（m/s） | 自车速度 | 判断停车/缓行/急刹 |
| `accel_x/y/z` | float | 自车加速度（m/s²） | 急刹检测（`accel_x` 为纵向） |
| `steer` | float | 方向盘转角 | 辅助检测转弯/变道 |

#### 道路拓扑（CARLA 服务端实时注入，离线数据已固化）

| 字段 | 类型 | 说明 | 用途 |
|------|------|------|------|
| `is_junction` | bool | 自车当前在 CARLA 路口拓扑区域内 | 路口判别主信号（但 R3 合流区也会为 True，需结合 TL 区分） |
| `is_intersection` | bool | 同 `is_junction`，部分版本使用不同字段 | 同上 |
| `junction_id` | int（-1=非路口） | 当前所在路口 ID | 追踪穿越同一路口的全过程 |
| `dist_to_junction` | float（米） | 到前方路口的距离 | 判断"即将进入路口"；`inf` 表示无路口在前 |
| `distance_to_next_junction` | float（米） | 同上，部分版本字段名不同 | 与上面字段取 min 使用 |
| `lane_id` | int | 当前所在车道 ID（负=正向，正=反向，CARLA 约定） | 辅助判断变道 |
| `ego_lane_id` | int | 同上 | 同上 |
| `ego_lane_width` | float（米） | 当前车道宽度 | 辅助判断窄道/双向单车道 |
| `lane_type_str` | str | 车道类型字符串，如 `"Driving"/"Shoulder"/"Parking"` | 判断是否在停车带 |
| `lane_change_str` | str | 允许变道方向，如 `"NONE"/"Right"/"Both"` | 辅助判断是否允许借对向 |

#### 信号灯

| 字段 | 类型 | 说明 | 用途 |
|------|------|------|------|
| `traffic_light_state` | str | `"Red"/"Yellow"/"Green"/"None"` | **R4 vs R5 判别的最强证据**；实测出现有效灯态时自车距路口 ≤50m，与 RGB 可见性对应 |
| `light_hazard` | bool | 当前有闯红灯风险 | 辅助信息 |
| `stop_sign_close` | bool | 附近有停止标志 | 辅助 R5 判别 |
| `stop_sign_hazard` | bool | 停止标志需要制动 | 辅助 R5 判别 |

#### 场景激活与事件距离（最关键的 EVENT 判断信号）

| 字段 | 类型 | 说明 | 用途 |
|------|------|------|------|
| `current_active_scenario_type` | str / `None` | 当前激活的 CARLA scenario 名称；非激活时为 `None` | 判断当前帧是否处于场景核心区间 |
| `dist_to_accident_site` | float（米） | 到事故障碍物的距离；无障碍时为 `inf` | Accident/AccidentTwoWays 的 U-E2 可见门控 |
| `dist_to_construction_site` | float（米） | 到施工障碍物的距离 | ConstructionObstacle 系列 U-E2 门控 |
| `dist_to_parked_obstacle` | float（米） | 到停放障碍物的距离 | ParkedObstacle 系列 U-E2 门控 |
| `dist_to_vehicle_opens_door` | float（米） | 到开门车辆的距离 | VehicleOpensDoorTwoWays / R6 门控 |
| `dist_to_cutin_vehicle` | float（米） | 到切入车辆的距离 | ParkingCutIn / StaticCutIn U-E3 门控 |
| `dist_to_pedestrian` | float（米） | 到最近行人的距离 | U-E4 门控 |
| `dist_to_biker` | float（米） | 到最近自行车的距离 | U-E4 门控 |
| `vehicle_hazard` | bool | 当前有车辆碰撞风险 | 辅助 U-E1/U-E3 |
| `walker_hazard` | bool | 当前有行人碰撞风险 | 辅助 U-E4 |

**注意：** 所有 `dist_to_*` 字段在障碍物不存在或超出感知范围时为 `inf`（Python `float('inf')`），使用前必须过滤 `inf`。

---

### 2.3 CARLA OpenDRIVE 地图（离线拓扑查询）

**位置：**
- Town01~Town07, Town10HD：`CARLA_0915/CarlaUE4/Content/Carla/Maps/OpenDrive/<TownName>.xodr`
- Town12, Town13（大地图）：`CARLA_0915/CarlaUE4/Content/Carla/Maps/<TownName>/OpenDrive/<TownName>.xodr`

**加载方式（已验证可离线使用）：**

```python
import carla
xodr_txt = open('CARLA_0915/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05.xodr').read()
map_obj = carla.Map('Town05', xodr_txt)   # 不需要连接 CARLA 服务端
```

**依赖：** automot conda 环境已安装 carla 包（含 `carla.libs/libjpeg`/`libtiff`），直接 `import carla` 即可。

**`carla.Map.get_waypoint(location)` 返回的 `Waypoint` 字段（实测）：**

| 字段 | 类型 | 说明 | 用途 |
|------|------|------|------|
| `road_id` | int | OpenDRIVE road 段 ID | 判断是否在同一路段 |
| `lane_id` | int | 车道 ID（负=正向行驶，正=反向，CARLA 约定） | 正负号变化 = 对向车道 |
| `lane_type` | `carla.LaneType` 枚举 | `Driving/Shoulder/Parking/Sidewalk/...` | 判断是否在可驾驶区域 |
| `lane_width` | float（米） | 车道物理宽度 | 辅助判断窄道 |
| `is_junction` | bool | 是否在路口拓扑区域 | 路口判别（注意：匝道合流区也为 True） |
| `junction_id` | int（-1=非路口） | 路口唯一 ID | 追踪完整路口穿越过程 |
| `s` | float（米） | 在该 road 上的纵向距离 | 辅助精确定位 |
| `transform` | `carla.Transform` | 位置+朝向 | 可视化、方向比对 |

**`Waypoint` 可调用的关键方法：**

| 方法 | 返回 | 说明 |
|------|------|------|
| `wp.get_left_lane()` | `Waypoint / None` | 相邻左侧车道；`lane_id` 符号改变 = 对向 |
| `wp.get_right_lane()` | `Waypoint / None` | 相邻右侧车道 |
| `wp.get_junction()` | `carla.Junction / None` | 当 `is_junction=True` 时获取路口对象 |
| `map_obj.generate_waypoints(spacing)` | `[Waypoint, ...]` | 全图等间距采样，用于离线预建路口/车道索引 |

**对向车道检测逻辑（实测）：**

```python
wp = map_obj.get_waypoint(carla.Location(x=ego_x, y=ego_y))
# 向左遍历相邻车道
cur = wp.get_left_lane()
while cur is not None and cur.lane_type == carla.LaneType.Driving:
    if (wp.lane_id > 0) != (cur.lane_id > 0):
        has_opposite_driving = True
        break
    cur = cur.get_left_lane()
```

**`carla.Junction.get_waypoints(LaneType.Driving)` 返回：**

路口内 `(入口 Waypoint, 出口 Waypoint)` 对的列表，可计算路口内有几条行驶轨迹，判断路口类型（十字/T形等）。

**xodr 文件内含的信号信息（不通过 map API，直接解析）：**

xodr 中 `<signal>` 标签记录路标和信号灯，字段包括：
- `type="206"`：停止标志（Stop Sign）
- `dynamic="yes"`：动态信号（含交通灯）；`"no"` 为静态标志

Town05.xodr 实测：`junctions=21, signals=59, controllers=106`，通过 `controllers` 可离线预建"哪些路口有受控信号灯"的索引，辅助 R4/R5 判别。

**注意：** 信号灯的实时颜色（Red/Green/Yellow）无法离线获取，必须依赖 meta 字段 `traffic_light_state`。

---

### 2.4 每帧 Bounding Box 文件（可选补充信号）

**位置：** `<run_dir>/bboxes/<帧编号>.pkl`（同样是 xz 压缩 pickle）

**内容：** 当前帧内所有感知到的障碍物列表，每个对象为 dict，含：

| 字段 | 说明 |
|------|------|
| `class` | 对象类型，如 `"car"/"pedestrian"/"walker"` |
| `distance` | 到自车的距离（米） |
| `position` | 相对自车坐标 `[x, y, z]`（米） |
| `extent` | 物体半尺寸 `[半长, 半宽, 半高]`（米） |

**用途：** 当 `dist_to_*` meta 字段为 `inf` 时，可从 bboxes 中计算最近同类对象距离，作为距离门控的 fallback。

---

### 2.5 run 结果文件（run 级元数据）

**位置：** `<run_dir>/results.json`

**关键字段：**

| 字段 | 说明 |
|------|------|
| `route_id` | 如 `"RouteScenario_route_001761_rep0"`，含 route 编号，用于与 XML 对齐 |
| `status` | `"Perfect"/"Failed"` 等，可过滤失败 run |
| `meta.duration_game` | 游戏时长（秒），用于计算帧率（`duration_game / total_frames`） |
| `meta.route_length` | 路线总长度（米） |
| `infractions` | 各类违规计数，可用于质量过滤 |


---

## 3. run 与 route XML 的对齐规则

## 3.1 目标

对每个 run 建立唯一映射：

- `scenario_name`
- `route_num`（如 001761）
- `town`
- `route_xml_path`

## 3.2 推荐实现

1. 从 run 文件夹名提取 route 编号：匹配 `route_(\d+)`。
2. 结合 scenario 目录拼接 XML：
   - `data/data_routes/lead/<scenario_name>/route_<route_num>.xml`
3. 若存在多个候选（极少数命名异常），用 `results.json.route_id` 二次确认。
4. 校验 `xml.route.town == meta.town`（抽样帧）防止错配。

## 3.3 对齐失败兜底

- 若 XML 缺失：该 run 仅用 meta 做弱规则标注，置信度降级。
- 若 town 不一致：打硬错误并进入人工复核队列。

---

## 4. 帧级几何定位与拓扑特征构建

## 4.1 从 meta 取每帧全局坐标

对每帧：

- `x = ego_matrix[0][3]`
- `y = ego_matrix[1][3]`
- `z = ego_matrix[2][3]`
- `yaw = theta`

## 4.2 route 进度 s（沿线弧长）

把 XML waypoints 组成 polyline，计算每帧点到 polyline 的最近投影，得到：

- `s_frame`：沿路线累计距离（米）
- `d_route`：到路线中心线距离（米）

用途：

- 同一 run 内时序对齐稳定
- 可把 trigger_point 投影成 `s_trigger`，构造“触发窗口”

## 4.3 地图拓扑特征（通过 carla.Map 离线查询）

`carla.Map` 已验证在 automot 环境下可离线加载 xodr（不需连接 CARLA 服务端），为每帧提供以下几何拓扑特征：

| 特征 | 获取方式 | 用于 |
|------|----------|------|
| `is_junction_map` | `wp.is_junction` | 更精确的路口判别（区分 R3 合流节点 vs 真实路口） |
| `junction_id_map` | `wp.junction_id` | 追踪整段路口穿越，避免边界帧抖动 |
| `lane_count_same_dir` | 遍历 `get_left_lane()/get_right_lane()` 统计同向 Driving 车道数 | 辅助判断是否可能为双向单车道（单车道更可能 R2） |
| `has_opposite_driving_lane` | 遍历左邻车道，检测 `lane_id` 符号反转 | R2 的地图几何证据 |
| `junction_lane_pairs` | `junction.get_waypoints(Driving)` 返回对数 | 区分 T 形路口（3 对入/出）和十字路口（4+ 对） |
| `has_signal_controller` | 离线解析 xodr `<controller>` 标签 | 预判该路口是否为信号灯控制路口，辅助 R4/R5 |

**实现建议：** 预建每个 town 的路口索引（`{junction_id: {has_signal, lane_pair_count, location}}`），每帧查询时用自车坐标找最近路口，避免逐帧重复解析。


---

## 5. ROAD_STRUCTURE 判定主逻辑（核心）

以下规则均以**实际 meta 数据探测结果为依据**，不是凭空推断。

---

## 5.0 判定框架总览

整体逻辑是"分层优先覆盖"，不是简单打分：

```
每帧先判断是否进入路口决策区（R4/R5）
  → 路口内：再判断有无信号灯（R4 vs R5）
非路口直道段：
  → 判断是否在 R3 事件窗口（仅限强相关 scenario）
  → 判断是否在 R2 事件窗口（obstacle 距离门控）
  → 判断是否在 R6 事件窗口（parking 距离门控）
  → 以上均否：R1（默认直道）
```

路口判定优先于 R2/R3/R6：**一旦满足 R4/R5 条件，R2/R3/R6 暂时让位。**

---

## 5.1 路口类型判别（R4/R5 入口）

### 哪些算"路口"

- **算**：十字路口、T 形路口、有停止线的多向交叉。
- **不算（→ R1）**：普通岔路、无主从关系的小路汇合、环岛、非结构化道路。

原因：环岛/岔路视觉上无明确停止线，标签意义不稳定，暂归 R1 便于分类。

### R3 的 `is_junction=True` 不等于灯控路口

实测 R3 全部场景（HighwayExit/HighwayCutIn/EnterActorFlow/MergerIntoSlowTraffic 等）
**tl% 均为 0.0%**，但 `is_junction` 可达 20%~43%。

这里的 `is_junction=True` 是匝道/合流区 CARLA 内部的拓扑节点，不是信号灯交叉口。
区分方式：**进入 `is_junction=True` 区域前后 TL 状态从未出现 Red/Yellow/Green → 合流/匝道节点，不切换 R4/R5。**

---

## 5.2 R4 / R5 判别标准（数据驱动）

### 信号灯可见距离阈值（实测）

对信号灯场景（SignalizedJunctionLeftTurn/RedLightWithoutLeadVehicle 等）样本：

| TL 状态 | is_junction | p10 距路口 | p50 | p90 | max |
|---------|-------------|-----------|-----|-----|-----|
| Red     | False       | 11m       | 19m | 25m | **49m** |
| Yellow  | False       | 31m       | 37m | 42m | 43m |
| Green   | False       | 5m        | 16m | 31m | **50m** |
| None    | False       | 58m       | 100m | 156m | — |

结论：`traffic_light_state` 从 `None` 变为有效灯态时，自车距路口均 **≤ 50m**。
这个 **50m** 是 RGB 里"可以看到红绿灯"的最远距离，作为 R4 注入的距离门控阈值。

### R4 注入条件（与 RGB 对应）

必须满足以下**至少一条可见确认**，才注入 R4 标签：

1. `traffic_light_state in {Red, Yellow, Green}` → 最强可见证据，直接确认 R4。
2. `distance_to_next_junction < 50m` 且 scenario 属于 signalized 类 → 预期可见范围，允许 R4。

两者同时不满足时，即使已在 `current_active_scenario_type` 激活范围，也不打 R4，维持 R1。

### R5 注入条件

1. 在路口区域内（`is_junction=True` 或 `distance_to_next_junction < 50m`）。
2. 且 `traffic_light_state` 全程为 `None`（连续 ≥ 5 帧，排除短暂抖动）。
3. 且 scenario 属于非信号/路权类，或为 CrossJunctionDefectTrafficLight。

R5 只在路口区间内有效，直道上不触发。

### R4 vs R5 区分决策树

```
进入路口区域（is_junction=True 或 distance_to_next_junction < 50m）？
  → 否：不走此分支
  → 是：
      traffic_light_state != None（连续 >= 3 帧）？
        → 是：R4
        → 否：scenario 是 signalized 类但灯短暂未感知？
               → 用滞后逻辑维持 R4 最多 5 帧
             scenario 是 non-signalized/defect 类？
               → R5
```

### 关于 nonsig 场景里出现 Green 的处理

实测 non-signalized 场景有少量 `Green` 出现（p50=24m）。
原因：route 经过有灯路口后才进入无灯目标路口，属于同一 run 内的 R4 前置片段。
处理方式：以 `traffic_light_state` 实际值为准，该段打 R4，不因 scenario 是无灯类就强制 R5。

---

## 5.3 R3（高速/匝道/合流）判别标准

### 关键发现：R3 强 scenario 相关，与 R4/R5 天然不重叠

实测 7 个 R3 场景均 tl%=0.0%，`is_junction` 可达 20~43% 但均为合流/匝道节点。

**R3 触发条件（同时满足）：**

1. `scenario` 属于强相关集合：
   `{HighwayExit, HighwayCutIn, EnterActorFlow, EnterActorFlowV2, MergerIntoSlowTraffic, MergerIntoSlowTrafficV2, InterurbanActorFlow}`
2. 不在路口区域（`is_junction=False` 且 `distance_to_next_junction >= 50m`）
3. `current_active_scenario_type` 激活，或 `s_frame` 在 trigger 窗口内

**不在以上 scenario 集合内的 run，不打 R3。**
其他场景里偶尔出现的变道行为用 R-E2/R-E3 事件表达，不升级道路结构。

### R3 内的 `is_junction=True` 片段

无 TL 状态出现时维持 R3，完全离开 R3 事件窗口后回 R1。

---

## 5.4 R2（双向单车道）判别标准

### 关键发现：TwoWays 场景约 50% 帧出现信号灯

实测：

| Scenario | junc% | tl% |
|---|---|---|
| AccidentTwoWays | 0.0% | 48.9% |
| ConstructionObstacleTwoWays | 0.0% | 49.1% |
| ParkedObstacleTwoWays | 0.0% | 0.0% |
| HazardAtSideLaneTwoWays | 1.6% | 33.1% |
| VehicleOpensDoorTwoWays | 0.0% | 3.9% |
| InvadingTurn | 3.6% | 36.6% |

**R2 不是全程标签**，只在事件激活窗口内有效；其余片段按 R4/R5 或 R1。

**R2 触发条件（同时满足）：**

1. scenario 属于 `{AccidentTwoWays, ConstructionObstacleTwoWays, ParkedObstacleTwoWays, HazardAtSideLaneTwoWays, VehicleOpensDoorTwoWays, InvadingTurn}`
2. 不在路口区域（当前帧不满足 R4/R5 条件）
3. 相关障碍/风险距离字段进入阈值（`dist_to_*` < `T_r2_on`，参考 U-E2 可见门控 25m）

**R2 窗口外：**

- 进入路口 → R4/R5
- 直道且事件距离不满足 → R1

---

## 5.5 R6（路边停车/停车占道）判别标准

### 关键发现：R6 场景大量帧在信号灯路口

实测：

| Scenario | junc% | tl% |
|---|---|---|
| ParkingCutIn | 2.9% | **82.9%** |
| ParkingCrossingPedestrian | 7.3% | **81.2%** |
| ParkingExit | 0.0% | 25.0% |

R6 绝对不能整段覆盖，否则把大量 R4 段误打为 R6。

**R6 触发条件（三段式，同 U-E2 可见门控逻辑）：**

1. scenario 属于 `{ParkingCutIn, ParkingExit, ParkingCrossingPedestrian, VehicleOpensDoorTwoWays（停车风险主导片段）, StaticCutIn（若确认在停车区）}`
2. 不在路口区域（路口优先 R4/R5）
3. 相关距离字段进入阈值：`dist_to_vehicle_opens_door` / `dist_to_cutin_vehicle` / `dist_to_parked_obstacle` < `T_r6_on`（推荐初始 25m）

**R6 活跃窗口之外：**

- 进入路口 → R4/R5（ParkingCutIn 大量帧应为 R4）
- 直道且距离不满足 → R1

---

## 5.6 R1（默认直道）适用范围

不满足任何 R2/R3/R4/R5/R6 触发条件的帧，均回落到 R1。

以下情况**强制 R1，不触发任何特殊结构**：

- 环岛内（无明确停止线，不算路口）
- 岔路/不规则汇合（非标准交叉，不算路口）
- TwoWays 类 route 在障碍事件段结束后的直道
- R6 类 route 在停车事件段结束后的直道
- R3 类 route 在合流事件段结束后的直道

---

## 5.7 各 Scenario 初步 R 分配总结

| Scenario | 直道默认 | 路口 | 特殊窗口 |
|---|---|---|---|
| Accident, ConstructionObstacle, ParkedObstacle | R1 | R4（灯态可见时） | — |
| AccidentTwoWays, ConstructionObstacleTwoWays | R1 | R4 | R2（障碍窗口，约路程50%以外回R4/R1） |
| ParkedObstacleTwoWays | R1 | 通常无 R4（tl%=0）| R2（障碍窗口） |
| HazardAtSideLaneTwoWays | R1 | R4（33% tl%） | R2（障碍窗口） |
| InvadingTurn | R1 | R4（37% tl%） | R2（对向侵占窗口） |
| VehicleOpensDoorTwoWays | R1 | 偶发 R4（4%） | R2/R6（开门距离窗口） |
| HighwayExit, HighwayCutIn | R3（事件窗口内） | — | — |
| EnterActorFlow(V2) | R3（合流窗口） | — | — |
| MergerIntoSlowTraffic(V2) | R3（合流窗口） | — | — |
| InterurbanActorFlow | R3（合流窗口） | R4/R5（路口窗口） | — |
| NonSignalizedJunction* | R1 | R5（路口+无灯） | — |
| PriorityAtJunction, OppositeVehicleTakingPriority | R1 | R5 | — |
| CrossJunctionDefectTrafficLight | R1 | R5（强制，灯故障） | — |
| SignalizedJunction*, RedLightWithoutLeadVehicle, T_Junction | R1 | R4 | — |
| ParkingCutIn, ParkingCrossingPedestrian | **R4**（路口段为主，82%帧有灯）| R4 | R6（停车窗口内，窗口外为 R4） |
| ParkingExit | R1 | R4（偶发） | R6（停车汇入窗口） |
| BlockedIntersection, CrossingBicycleFlow | R1 | R4 | — |
| HardBreakRoute, ControlLoss, noScenarios | R1 | R4（偶发） | — |
| DynamicObjectCrossing | R1 | R4（偶发） | — |
| PedestrianCrossing, VehicleTurningRoute(Pedestrian) | R1 | R4/R5（按灯态） | — |
| StaticCutIn | R1/R3（按位置） | R4（偶发） | R6（若在停车区） |
## 6. 场景触发窗口（Scenario Window）融合

你现有数据里 `current_active_scenario_type` 已经非常有价值，可直接并入规则层。

推荐策略：

1. 由 `current_active_scenario_type == scenario_name` 形成原始激活掩码。
2. 再与距离字段阈值交集，得到高置信触发窗口。
3. 窗口内允许 R2/R3/R6 获得更高先验分；窗口外大幅衰减。

这样可以抑制“某些场景名在视觉上像全程生效”的误判。

## 6.1 重要约束：XML trigger 不是事件“可见起点”

这个约束必须单独强调：

- XML 里的 `trigger_point` 代表 scenario 机制开始被仿真系统激活的地理锚点。
- 它不等于人类或模型在当前相机帧里“已经看见事件对象”。

因此，`trigger_point` 只能作为“候选时间窗起点”，不能直接作为 EVENT 注入起点。

否则会出现你说的问题：自车还没看见障碍，标签却已经打成 `U-E2`，导致学习被污染。

## 6.2 U-E2 注入时机（防过早标注）

对 `U-E2 静态障碍物占道`，推荐三段式门控：

1. 激活门（scenario-level）：
  - 满足 `current_active_scenario_type == scenario_name` 或 `s_frame >= s_trigger - pad_pre`。
  - 只说明“事件可能发生”，不直接打 U-E2。
2. 可见/可感知门（per-frame）：
  - 仅当 `dist_to_accident_site` / `dist_to_construction_site` / `dist_to_parked_obstacle` / `dist_to_vehicle_opens_door`
    之一进入阈值 `T_visible_on` 时，才允许注入 `U-E2`。
  - 若有 bbox/前向视锥信息，可加条件“障碍位于前向可视区域”。
3. 持续门（hysteresis）：
  - 注入后直到距离回升到 `T_visible_off`（`T_visible_off > T_visible_on`）并持续 K 帧才退出。
  - 防止边界抖动。

推荐阈值初始化：

- `T_visible_on = 25m`（远距先保守）
- `T_visible_off = 32m`
- `K = 3~5` 帧

最终以你抽检结果再调参。

## 6.3 与 ROAD_STRUCTURE 的关系

- ROAD_STRUCTURE 帧级标注本身不依赖“是否看见障碍”，它主要由道路规则空间决定。
- 上述“可见性门控”主要作用于 EVENT 注入时机（尤其 U-E2/U-E3/U-E4）。

也就是：

- 你可以先稳定输出 ROAD_STRUCTURE。
- 再在对应 ROAD_STRUCTURE 下，按可见门控注入突发事件标签。

---

## 7. 时序平滑（避免帧抖动）

建议最少做两层平滑：

1. 中值/众数滑窗（窗口 5~9 帧）
2. 最短持续约束（例如少于 4 帧的孤立标签回并到邻居）

可选：HMM/Viterbi，转移代价建议：

- 低代价：R1 <-> R4、R1 <-> R5、R1 <-> R6
- 中代价：R1 <-> R2、R1 <-> R3
- 高代价：R2 <-> R5、R3 <-> R5（通常不直接跳）

---

## 8. 产出格式建议

建议新增每 run 一个帧级标注文件，例如：

- `keyframe_filter/outputs/road_structure_labels/<Scenario>/<run_id>.json`

单帧结构示例：

```json
{
  "frame": 87,
  "timestamp_sec": 21.75,
  "ego": {"x": 210.74, "y": -11.27, "z": -0.02, "yaw": -1.58},
  "route_progress_m": 45.12,
  "road_structure": "R6",
  "confidence": 0.91,
  "evidence": {
    "is_junction": false,
    "traffic_light_state": "None",
    "current_active_scenario_type": "ParkingCutIn",
    "dist_to_cutin_vehicle": 9.8,
    "rules_fired": ["scenario_window", "r6_distance_trigger", "priority_override"]
  }
}
```

注意保留 `evidence.rules_fired`，便于后续人工抽检和阈值调参。

---

## 9. 与你现有 ROAD/EVENT 体系的衔接

你已经有两份关键文档：

- `ROAD_EVENT_CLASSIFICATION_PLAN.md`
- `ROAD_EVENT_CANDIDATE_MAPPING.md`

衔接方式：

1. 本方案先生成每帧 `ROAD_STRUCTURE` 单标签。
2. 再按映射表第 4/5 节，得到每帧 `EVENT` 候选集合。
3. 最后再决定是否让 Qwen 只在“候选集合内”做细粒度判别。

这样 Qwen 不再需要先猜你在哪种道路结构，错误会明显减少。

---

## 10. 分阶段落地计划（建议）

## Phase A（1~2 天）：不依赖 CARLA 地图，先跑通

输入：`metas + route_xml + candidate_mapping`

- 完成 run->xml 对齐
- 完成帧级 `R1~R6` 规则初版（主要依赖 meta 字段）
- 输出每帧标签与证据

目标：先把 Qwen 的结构判断替换掉 70% 以上。

## Phase B（2~4 天）：引入 CARLA 地图拓扑增强

- 用 `carla.Map.get_waypoint` / junction / lane topology 增强 R2/R3/R4/R5 边界
- 增加 ramp/merge 检测与对向车道参与检测

目标：减少 R2/R3 与 R4/R5 混淆。

## Phase C（持续）：阈值自动校准与评估

- 抽样人工标注 200~500 帧作验证集
- 调整阈值与优先级
- 增加失败案例回放清单

---

## 11. 关键实现细节与风险提示

1. `metas/*.pkl` 是 xz 压缩，读取必须 `lzma.open`。
2. 很多距离字段会出现 `inf`，计算阈值前要先过滤。
3. `current_active_scenario_type` 不是全程有效，常出现 `None`，必须与距离字段联合。
4. `traffic_light_state` 可能短时抖动，不可单帧硬判 R4/R5。
5. 若拿不到 CARLA 地图，不影响第一版上线；先做 meta+xml 版即可。

---

## 12. 伪代码（可直接转脚本）

```text
for run in all_runs:
    scenario = parse_scenario_from_path(run)
    route_num = parse_route_num(run)
    xml = load_route_xml(scenario, route_num)
    route_polyline = xml.waypoints
    trigger_points = xml.scenarios.trigger_point

    frames = load_all_metas_lzma(run)

    for t, meta in enumerate(frames):
        ego = extract_ego_xyz_yaw(meta)
        s, d = project_to_route_polyline(ego.xy, route_polyline)

        feat = build_features(meta, ego, s, d)
        # feat: is_junction, traffic_light_state, dist_to_*, active_scenario, ...

        score = init_scores(R1..R6)
        score += apply_r4_rules(feat, scenario)
        score += apply_r5_rules(feat, scenario)
        score += apply_r3_rules(feat, scenario, map_feat)
        score += apply_r2_rules(feat, scenario, map_feat)
        score += apply_r6_rules(feat, scenario)
        score += apply_r1_default(feat)

        label[t] = argmax_with_priority(score)
        evidence[t] = collect_fired_rules()

        # 事件注入要做“可见性门控”，不要把 xml trigger 直接当作 U-E2 起点
        event_gate = build_event_visibility_gate(feat, s, trigger_points)
        events[t] = inject_events_with_hysteresis(label[t], feat, event_gate)

    label = temporal_smoothing(label)
      events = temporal_smoothing_events(events)
      dump_framewise_road_and_events(run, label, events, evidence)
```

---

## 13. 结论

你的思路是对的，而且从现有数据字段看已经具备可实施条件。

最务实路径是：

1. 先用 `meta + route xml` 做强规则帧级 ROAD_STRUCTURE；
2. R6 做区间触发，不做全程标签；
3. 后续再接 CARLA 地图拓扑增强边界；
4. Qwen 只在候选空间内做细分，不再承担主结构判断。

这条线可以显著提升一致性，也更容易解释和调参。
