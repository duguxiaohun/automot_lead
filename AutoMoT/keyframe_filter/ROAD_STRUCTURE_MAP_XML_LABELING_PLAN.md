# 基于 CARLA 地图 + Route XML + LEAD Meta 的 ROAD_STRUCTURE 帧级标注方案

## 1. 背景与目标

你当前的问题很明确：仅靠 VLM（Qwen）看图判断每帧属于哪个场景/道路结构，稳定性不足，尤其在场景切换边界和视觉歧义帧上容易漂移。

本方案目标是把 `ROAD_STRUCTURE` 的主判断从“视觉语义”切到“几何与规则语义”，即：

1. 用每帧自车全局坐标（`metas/*.pkl`）确定自车在地图中的位置与拓扑关系。
2. 用 `data/lead/<Scenario>/*.xml` 提供 route 先验和 scenario 触发信息。
3. 用 meta 中的 scenario 距离字段与激活字段，提供“事件区间”证据（尤其是 R2/R6 这类局部触发结构）。
4. 产出每帧 `ROAD_STRUCTURE`（R1~R6）和可解释的置信来源，作为后续 EVENT 标注与 Qwen 候选裁剪的上游输入。

一句话：把 Qwen 从“主裁判”降级为“补充裁判”，先用地图与轨迹给出强规则主标签。

**重要目标修正：** 本方案不追求“每一帧绝对无歧义”。CARLA/LEAD 场景里天然存在
路口接近区、停车带与对向借道重叠、匝道拓扑节点被 `is_junction` 标记、信号灯来自相邻路口等情况。
因此最终产物应是：

- 一个用于训练主监督的 `primary_road_structure`；
- 少量用于审计的 `secondary_road_structures`；
- 每帧 `confidence_level` / `label_source` / `review_required`；
- 对边界帧、证据冲突帧和低置信帧保留复核队列。

也就是说，目标是“高一致性、可解释、可复核”的帧级主标签，而不是把所有过渡帧强行伪装成无歧义真值。

---

## 2. 输入数据源规范

本节对三类输入的**文件位置、格式、可用字段及用途**逐一说明，以便没有实际数据的读者也能理解后续判定逻辑。

---

### 2.1 Route XML（路线定义文件）

**位置：** `data/lead/<Scenario>/*.xml`

**对应关系：** run 文件夹名包含完整 `route_key`，不是只有数字 route。必须从
`lead_data/<Scenario>/<run_id>` 提取 `(town, route_key)`，再按固定命名公式查
`data/lead/<Scenario>/<Town>_<route_key>.xml`：
`route_001761 -> Town05_route_001761.xml`，`1054_0 -> Town12_route_1054_0.xml`，
`Town06_13 -> Town06_route_Town06_13.xml`，
`Town12_route15 -> Town12_route_Town12_route15.xml`。

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
| `pos_global` | array-like, len≥2 | 自车世界坐标真值 `[x,y,z]` | **主位姿来源**，与 LeadMoT runner 对齐 |
| `theta` | float（弧度） | 自车航向角（偏航） | **主航向来源**，与 LeadMoT runner 对齐 |
| `ego_matrix` | 4×4 numpy array | 自车 pose 矩阵；`[:3,3]` 与 `pos_global` 对齐 | 仅作校验/兜底，不作为首选 |
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
- `route_key`（如 `route_001761` / `1054_0` / `Town06_13` / `Town12_route15`）
- `town`
- `route_xml_path`

## 3.2 推荐实现

1. 从 `lead_data/<scenario_name>/<run_id>` 提取 `(town, route_key)`，其中
   `scenario_name` 必须取 run 的父目录，不能从 XML 内 scenario type 或
   `data_routes` 源目录反推：
   - `Town03_Rep0_route_001783_route0_...` -> `town=Town03`, `route_key=route_001783`
   - `Town12_Rep0_1054_0_route0_...` -> `town=Town12`, `route_key=1054_0`
   - `Town06_Rep0_Town06_13_route0_...` -> `town=Town06`, `route_key=Town06_13`
   - `Town12_Rep0_Town12_route15_...` -> `town=Town12`, `route_key=Town12_route15`
   - 解析时先剥末尾 `MM_DD_HH_MM_SS` 时间戳；尾部采集后缀 `_route0` 只有存在时才剥掉，`route15` 这类 legacy key 本体必须保留，且不能要求它带 `_route0`。
2. 结合 scenario 目录拼接规范 XML。文件名公式是：`route_key` 以 `route_`
   开头时用 `<Town>_<route_key>.xml`，否则用 `<Town>_route_<route_key>.xml`。
   - 旧数字 route: `data/lead/<scenario_name>/Town03_route_001783.xml`
   - 新版子编号: `data/lead/<scenario_name>/Town12_route_1054_0.xml`
   - legacy key: `data/lead/<scenario_name>/Town06_route_Town06_13.xml`
   - legacy key 内部带 route 编号: `data/lead/<scenario_name>/Town12_route_Town12_route15.xml`
3. 建索引时保留旧数字 route 的 `001783` / `route_001783` 互查能力，兼容已有脚本。
4. 校验 `xml.route.town == meta.town`（抽样帧）防止错配。

## 3.3 对齐失败兜底

- 若 XML 缺失：当前 2026-07-03 全量核对没有缺失；如果后续数据版本新增 run
  导致查不到 XML，先确认是否应从 `AutoMoT/data/data_routes` 提取补齐。确认无法补齐后，
  该 run 仅用 meta + XODR 做弱规则标注，置信度降级，并写入 review 队列。
- 若 town 不一致：打硬错误并进入人工复核队列。

---

## 4. 帧级几何定位与拓扑特征构建

## 4.1 从 meta 取每帧全局坐标

对每帧：

- 首选 `pos_global[:2]` / `theta`，这是当前 LeadMoT runner 和训练数据默认使用的真值位姿口径；
- `ego_matrix[:3,3]` 只用于交叉校验或 `pos_global` 缺失时的兜底；
- 若 `pos_global[:2]` 与 `ego_matrix[:2,3]` 偏差超过 0.5m，应将该帧标为 `review_required=true`；
- 不要把 `ego_matrix` 误当作“世界坐标系→自车坐标系”的逆矩阵直接用来转点。若要做 world→ego，
  必须显式使用 `pos_global + theta` 或经过验证的 inverse transform。

## 4.2 route 进度 s（沿线弧长）

把 XML waypoints 组成 polyline，计算每帧点到 polyline 的最近投影，得到：

- `s_frame`：沿路线累计距离（米）
- `d_route`：到路线中心线距离（米）

用途：

- 同一 run 内时序对齐稳定
- 可把 trigger_point 投影成 `s_trigger`，构造“触发窗口”

**重要限制：** XML waypoint 常常很稀疏，甚至存在 `ParkingExit` 这类只有 2 个点的 route。
因此 XML polyline 只能作为 Phase A 粗召回。进入正式标注前，应基于 CARLA map 沿 road/lane
把 route densify 到 1~2m 间隔，或至少用 `carla.Map.get_waypoint` 对帧级位置做路网吸附。
若 `d_route > 3~5m`，该帧不得输出高置信 ROAD_STRUCTURE，应降级为低置信并进入复核队列。

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

整体逻辑是"候选召回 + 主标签仲裁"，不是简单打分，也不是大量多标签：

```
每帧先生成候选：
  → R4/R5 候选：灯态 + junction 拓扑 + route 是否穿越路口
  → R3 候选：高速/匝道/合流/驶出 scenario 先验 + ramp/merge/exit 拓扑
  → R2 候选：TwoWays/InvadingTurn 先验 + 对向车道/窄路拓扑
  → R6 候选：停车场景先验 + parking lane/路边停车/停车汇入拓扑
  → R1 候选：默认常规道路

再仲裁主标签 primary_road_structure：
  → R4/R5 > R3 > R2/R6 > R1
  → 只有真实重叠且对审计有价值时，保留少量 secondary_road_structures
  → 绝大多数帧仍应只有一个 primary，secondary 不是训练主监督
```

路口判定优先于 R2/R3/R6：**一旦确认当前主决策由 R4/R5 支配，R2/R3/R6 暂时让位为
secondary 或被压制。** 例如停车区附近的信号灯路口可以输出
`primary=R4, secondary=[R6]`，但训练主标签仍用 R4，避免“到处都是多标签”。

ROAD_STRUCTURE 与 EVENT 必须解耦：

- ROAD_STRUCTURE 表示当前主要驾驶规则空间，由地图/route/拓扑/场景先验决定。
- EVENT 表示当前可见或已发生的交通事件，由 `dist_to_*`、bbox、速度/刹车和时序门控决定。
- `dist_to_* < T_visible` 只应控制 U-E2/U-E3/U-E4 等事件注入，不能单独决定 R2/R6 道路结构。

建议同时保存两个内部概念，防止物理道路与决策空间混淆：

- `physical_road_structure_hint`：地图/车道形态提示，例如双向单车道、停车带、junction；
- `primary_road_structure`：当前帧训练用主规则空间，必须考虑 route、可见/可用规则、当前决策任务。

例如 TwoWays 场景可能物理上长时间都是双向单车道，但只有当前对向车道参与决策或静态障碍窗口临近时，
`primary_road_structure` 才应切到 R2；普通前置直行片段仍可为 R1。

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
这个 **50m** 是 RGB 里"可以看到红绿灯"的最远距离，但只作为 R4/R5 **候选召回阈值**，
不作为最终标签阈值。最终是否打 R4/R5，必须再看灯态确认、route 是否实际穿越该路口、
CARLA junction/stopline 拓扑、以及短时序稳定性。

### R4 注入条件（与 RGB 对应）

必须满足以下强确认，才把 R4 作为 `primary_road_structure`：

1. `traffic_light_state in {Red, Yellow, Green}` 连续或经滞后确认 → 最强证据，但仍需做 route/junction 同源校验。
2. `distance_to_next_junction < 50m` 且 scenario 属于 signalized 类 → 只召回 R4 候选；
   还需满足 route 前方确实进入该 junction/stopline，或 xodr controller/signal 索引确认该路口受灯控。
3. 当前帧离路口较近但灯态短暂缺失时，只允许通过上一段稳定 R4 状态维持少量帧，不允许从远处单帧跳 R4。

以上强确认不满足时，即使已在 `current_active_scenario_type` 激活范围，也不打 primary R4；
最多保留为低置信候选，主标签按 R1/R3/R2/R6 仲裁。

**同源校验建议：**

- 优先沿 route densified polyline 找“即将穿越的 junction/stopline”，不要只取空间最近 junction；
- 若 meta 有 `light_hazard` / stopline 相关字段，可作为灯态属于当前 route 的增强证据；
- 若有效灯态出现，但 route 前方并不穿越受控 junction，标为 `R4_candidate_low_confidence`，
  不直接覆盖 R2/R3/R6；
- non-signalized 场景里出现的 Green 可打 R4，但必须满足“当前 route 正经过有灯路口”。

### R5 注入条件

1. route 前方实际进入 junction/stopline 决策区；`distance_to_next_junction < 50m`
   只召回候选，不能单独确认。
2. `traffic_light_state` 全程为 `None`（连续 ≥ 5 帧，排除短暂抖动），或
   scenario 为 CrossJunctionDefectTrafficLight 这类灯控失效场景。
3. scenario 属于非信号/路权类，或 xodr/controller 索引未显示该 junction 有可用灯控。

R5 只在路口区间内有效，直道上不触发。

R5 的反例也要显式处理：若 xodr 显示该 route junction 受灯控，但当前短时间没有灯态，
不要立即打 R5；应先维持上一段 R4 或输出低置信 `R4/R5_ambiguous`，直到连续缺灯足够长且
scenario/defect 语义支持 R5。

### R4 vs R5 区分决策树

```
进入路口候选召回区（is_junction=True 或 distance_to_next_junction < 50m）？
  → 否：不走此分支
  → 是：
      route 是否实际穿越该 junction/stopline？
        → 否：不打 R4/R5，继续看 R1/R2/R3/R6
        → 是：
      traffic_light_state != None（连续 >= 3 帧或滞后确认）？
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

**R3 候选条件（同时满足）：**

1. `scenario` 属于强相关集合：
   `{HighwayExit, HighwayCutIn, EnterActorFlow, EnterActorFlowV2, MergerIntoSlowTraffic, MergerIntoSlowTrafficV2, InterurbanActorFlow}`
2. 不被确认的 R4/R5 主导；注意 `distance_to_next_junction < 50m` 只是路口候选召回，
   不能直接压掉 R3。
3. `current_active_scenario_type` 激活，或 `s_frame` 在 trigger 窗口内。
4. 若可用 CARLA map，优先补充 ramp/merge/exit 证据：road/lane 分流、合流、lane count 变化、
   XML `road_id`、`start_actor_flow/end_actor_flow` 等。

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

**R2 不是全程标签**，只在双向单车道/对向车道参与决策的结构窗口内有效；其余片段按
R4/R5 或 R1。这里的结构窗口不等于障碍物已经可见，障碍物可见性应下放给 U-E2/U-E5。

**R2 候选条件（同时满足）：**

1. scenario 属于 `{AccidentTwoWays, ConstructionObstacleTwoWays, ParkedObstacleTwoWays, HazardAtSideLaneTwoWays, VehicleOpensDoorTwoWays, InvadingTurn}`
2. 当前帧不被确认的 R4/R5 主导。
3. 存在双向单车道/对向车道参与证据之一：
   - CARLA lane topology 显示相邻 driving lane 存在 lane_id 符号反转，且自车可用同向车道数不足；
   - XML / scenario 明确为 TwoWays 或 InvadingTurn，且当前在 trigger/active 的结构窗口附近；
   - meta/bbox 显示对向车辆或障碍导致对向车道参与，但这只增强置信度，不作为唯一条件。
4. `dist_to_accident_site` / `dist_to_construction_site` / `dist_to_parked_obstacle`
   / `dist_to_vehicle_opens_door` 进入阈值时，只表示 U-E2 可见或 R2 置信度升高。

**R2 窗口外：**

- 进入路口 → R4/R5
- 直道且没有双向单车道/对向车道参与证据 → R1

**R2 主标签约束：**

- `physical_road_structure_hint=two_way_single_lane` 不等于主标签必为 R2；
- 只有“对向车道/对向来车已经参与当前决策”或“即将被迫借对向绕障”的结构窗口内才打 primary R2；
- InvadingTurn 与主动借对向绕障共享 R2，但 EVENT 必须区分：InvadingTurn 对应 U-E5，
  AccidentTwoWays/ConstructionObstacleTwoWays 等对应 U-E2；
- 若同一帧同时有停车带开门风险与对向借道风险，按当前动作主因仲裁：
  借对向/等待对向车为主 → primary R2，停车风险可进 secondary R6；
  停车带车辆切入/开门为主且未涉及对向借道 → primary R6。

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

**R6 候选条件（三段式，结构证据与事件可见性分离）：**

1. scenario 属于 `{ParkingCutIn, ParkingExit, ParkingCrossingPedestrian, VehicleOpensDoorTwoWays（停车风险主导片段）, StaticCutIn（若确认在停车区）}`
2. 当前帧不被确认的 R4/R5 主导；若停车区与信号灯路口重叠，输出 `primary=R4, secondary=[R6]`。
3. 存在停车/路边占道结构证据之一：
   - CARLA waypoint/lane 显示附近有 `Parking` lane、shoulder/parking shoulder，或 lane 宽/相邻 lane 形态符合停车带；
   - scenario/XML 明确为 Parking*，且当前在停车汇入/停车风险结构窗口附近；
   - bbox/meta 显示路边停放车辆密集或自车从停车侧汇入。
4. `dist_to_vehicle_opens_door` / `dist_to_cutin_vehicle` / `dist_to_parked_obstacle`
   < `T_visible_on` 时，只表示 U-E2/U-E3 可见或 R6 置信度升高，不单独决定 R6。

**R6 活跃窗口之外：**

- 进入路口 → R4/R5（ParkingCutIn 大量帧应为 R4）
- 直道且没有停车/路边占道结构证据 → R1

**R6 证据分级：**

- 高置信：Parking* scenario + active/trigger 结构窗口 + 旁侧停车/parking lane/bbox 静态车列证据；
- 中置信：Parking* scenario + active/trigger 窗口，但地图未能确认 parking/shoulder；
- 低置信：只有 `dist_to_*` 或只有 scenario 名，缺少停车空间结构证据。

CARLA `LaneType.Parking` 不一定覆盖所有路边停车场景，不能把它作为唯一入口；应允许 bbox 横向分布、
停车车辆密度、scenario 参数和 route 位置共同投票。

---

## 5.6 R1（默认直道）适用范围

没有任何 R2/R3/R4/R5/R6 候选被确认并赢得主标签仲裁的帧，均回落到 R1。

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

## 5.8 全场景调研启动结果与初始阈值矩阵

已按 `ROAD_STRUCTURE_SCENARIO_RESEARCH_PROTOCOL.md` 启动全量调研包生成：

```bash
python keyframe_filter/rs_research.py --samples-per-town 5
```

调研输入会先执行异常时长剔除：4Hz 下 `rgb/*.jpg >= 361`（严格大于 90s）且不在
`BlockedIntersection/ControlLoss` 白名单内的 run 不进入任何 sample、阈值拟合或人工审核队列。
某个 town 若在过滤后没有可读 meta run，则记录缺口并跳过该 town 的样本；其它 town 仍必须
抽 5 个分散 id 或全量可读 run。

本轮输出：

```text
keyframe_filter/collection_output/rs_research/<Scenario>/
```

覆盖 43 个 scenario。每个 scenario 按全部涉及 town 抽样，每个 town 优先取
first/q1/middle/q3/last 五条有 meta 的 run，并生成 XML、XODR、meta、RGB、maps、rules
证据链。当前状态是 `auto_artifacts_ready`，不是人工最终完成；后续必须逐场景检查
`maps/*route_trigger_ego_trace.png` 与 `rgb/*sample_contact_sheet.jpg` 后，才能把
`manual_map_rgb_checked` 和 `runtime_rule_ready` 置为 true。

当前只有一个 scenario 自动输入完整性暂缺：

- `NonSignalizedJunctionLeftTurn`: `Town10HD` 本地 run 缺可读 meta，按协议跳过该 town 的样本调研。

`SignalizedJunctionRightTurn/Town07` 已自动换用可读 meta 的
`Town07_Rep0_route_002583_route0_01_09_12_58_34`；该 town 只有 1 条可读 run，
因此按“可读 run 全量读取”记为 auto-ok。

下面的阈值是基于当前 XML/XODR/meta 自动审计的**初始调研阈值**，必须在每个 scenario 的
`rules/thresholds.json` 中补齐 supporting runs、reviewed map/RGB artifacts 和 reason 后，
才能作为最终 runtime 阈值。`status=auto-ok` 只表示自动证据链已生成，不代表人工验收完成。

| Scenario | 调研 rule kind | 候选 RS | town/sample | 初始参数与阈值 | status | gap |
|---|---|---|---:|---|---|---|
| `Accident` | `same_direction_obstacle` | `R1, R4` | 7 / 35 | `junction_pre_m=60, junction_post_m=25, veto=no_r2/no_r6` | `auto-ok` | - |
| `AccidentTwoWays` | `twoways_obstacle` | `R1, R2, R4` | 7 / 29 | `two_way_min_pre_m=45, two_way_post_pad_m=20, trigger_close_m=70` | `auto-ok` | - |
| `BlockedIntersection` | `signalized_junction` | `R1, R4` | 4 / 20 | `junction_pre_m=60, junction_post_m=25, blocked_is_event_not_rs` | `auto-ok` | - |
| `ConstructionObstacle` | `same_direction_obstacle` | `R1, R4` | 7 / 35 | `junction_pre_m=60, junction_post_m=25, veto=no_r2/no_r6` | `auto-ok` | - |
| `ConstructionObstacleTwoWays` | `twoways_obstacle` | `R1, R2, R4` | 7 / 29 | `two_way_min_pre_m=45, two_way_post_pad_m=20, trigger_close_m=70` | `auto-ok` | - |
| `ControlLoss` | `default_meta_map` | `R1, R4` | 10 / 50 | `junction_pre_m=50, junction_post_m=25, control_loss_not_rs` | `auto-ok` | - |
| `CrossJunctionDefectTrafficLight` | `defect_junction` | `R1, R5` | 8 / 40 | `junction_pre_m=60, junction_post_m=20, override=r5_over_r4` | `auto-ok` | - |
| `CrossingBicycleFlow` | `default_meta_map` | `R1, R4` | 1 / 5 | `junction_pre_m=50, junction_post_m=25, actor_flow_not_r3` | `auto-ok` | - |
| `DynamicObjectCrossing` | `default_meta_map` | `R1, R4` | 10 / 50 | `junction_pre_m=50, junction_post_m=25, crossing_event_not_rs` | `auto-ok` | - |
| `EnterActorFlow` | `highway_merge` | `R1, R3, R4` | 2 / 10 | `merge_pre_m=30, merge_post_m=40, trigger_close_m=90` | `auto-ok` | - |
| `EnterActorFlowV2` | `highway_merge` | `R1, R3, R4` | 1 / 5 | `merge_pre_m=30, merge_post_m=40, trigger_close_m=90` | `auto-ok` | - |
| `HardBreakRoute` | `default_meta_map` | `R1, R4` | 2 / 10 | `junction_pre_m=50, junction_post_m=25, brake_not_rs` | `auto-ok` | - |
| `HazardAtSideLane` | `default_meta_map` | `R1, R4` | 2 / 10 | `junction_pre_m=50, junction_post_m=25, side_lane_not_twoways` | `auto-ok` | - |
| `HazardAtSideLaneTwoWays` | `twoways_obstacle` | `R1, R2, R4` | 2 / 10 | `two_way_min_pre_m=70, two_way_post_pad_m=20, trigger_close_m=75` | `auto-ok` | - |
| `HighwayCutIn` | `highway_merge` | `R1, R3, R4` | 2 / 10 | `merge_pre_m=40, merge_post_m=40, trigger_close_m=90` | `auto-ok` | - |
| `HighwayExit` | `highway_merge` | `R1, R3, R4` | 2 / 10 | `merge_pre_m=50, merge_post_m=50, trigger_close_m=90` | `auto-ok` | - |
| `InterurbanActorFlow` | `interurban` | `R1, R3, R4, R5` | 2 / 10 | `merge_pre_m=50, merge_post_m=45, junction_pre_m=55, junction_post_m=25` | `auto-ok` | - |
| `InterurbanAdvancedActorFlow` | `interurban_advanced` | `R1, R4, R5` | 2 / 10 | `junction_pre_m=55, junction_post_m=25, r3_requires_topology=true` | `auto-ok` | - |
| `InvadingTurn` | `invading_turn` | `R1, R2, R4` | 2 / 10 | `two_way_min_pre_m=80, two_way_post_pad_m=20, trigger_close_m=75` | `auto-ok` | - |
| `MergerIntoSlowTraffic` | `highway_merge` | `R1, R3, R4` | 2 / 10 | `merge_pre_m=40, merge_post_m=50, trigger_close_m=90, keep_r3_when_slow=true` | `auto-ok` | - |
| `MergerIntoSlowTrafficV2` | `highway_merge` | `R1, R3, R4` | 3 / 12 | `merge_pre_m=40, merge_post_m=50, trigger_close_m=90, keep_r3_when_slow=true` | `auto-ok` | - |
| `NonSignalizedJunctionLeftTurn` | `nonsignalized_junction` | `R1, R5` | 6 / 28 | `junction_pre_m=50, junction_post_m=20` | `auto-gap` | `Town10HD:skipped_no_readable_meta` |
| `NonSignalizedJunctionLeftTurnEnterFlow` | `nonsignalized_junction` | `R1, R5` | 7 / 30 | `junction_pre_m=60, junction_post_m=20, enter_flow_not_r3` | `auto-ok` | - |
| `NonSignalizedJunctionRightTurn` | `nonsignalized_junction` | `R1, R5` | 2 / 10 | `junction_pre_m=45, junction_post_m=20` | `auto-ok` | - |
| `OppositeVehicleRunningRedLight` | `signalized_junction` | `R1, R4` | 10 / 50 | `junction_pre_m=50, junction_post_m=20, violation_not_r5` | `auto-ok` | - |
| `OppositeVehicleTakingPriority` | `nonsignalized_junction` | `R1, R5` | 2 / 10 | `junction_pre_m=50, junction_post_m=20` | `auto-ok` | - |
| `ParkedObstacle` | `same_direction_obstacle` | `R1, R4` | 7 / 35 | `junction_pre_m=60, junction_post_m=25, parked_not_parking_rs` | `auto-ok` | - |
| `ParkedObstacleTwoWays` | `twoways_obstacle` | `R1, R2, R4` | 2 / 10 | `two_way_min_pre_m=50, two_way_post_pad_m=20, trigger_close_m=70, parked_not_r6` | `auto-ok` | - |
| `ParkingCrossingPedestrian` | `parking` | `R1, R4, R6` | 2 / 10 | `parking_pre_m=35, parking_post_m=60, pedestrian_not_rs` | `auto-ok` | - |
| `ParkingCutIn` | `parking` | `R1, R4, R6` | 2 / 10 | `parking_pre_m=30, parking_post_m=50` | `auto-ok` | - |
| `ParkingExit` | `parking_exit` | `R1, R4, R6` | 5 / 25 | `parking_pre_m=20, parking_post_m=60, parking_to_driving_transition` | `auto-ok` | - |
| `PedestrianCrossing` | `pedestrian_crossing` | `R1, R4, R5` | 2 / 10 | `junction_pre_m=40, junction_post_m=40, pedestrian_not_rs` | `auto-ok` | - |
| `PriorityAtJunction` | `nonsignalized_junction` | `R1, R5` | 2 / 10 | `junction_pre_m=50, junction_post_m=20` | `auto-ok` | - |
| `RedLightWithoutLeadVehicle` | `signalized_junction` | `R1, R4` | 10 / 50 | `junction_pre_m=60, junction_post_m=20` | `auto-ok` | - |
| `SignalizedJunctionLeftTurn` | `signalized_junction` | `R1, R4` | 11 / 55 | `junction_pre_m=60, junction_post_m=25` | `auto-ok` | - |
| `SignalizedJunctionLeftTurnEnterFlow` | `signalized_junction` | `R1, R4` | 10 / 49 | `junction_pre_m=60, junction_post_m=25, enter_flow_not_r3` | `auto-ok` | - |
| `SignalizedJunctionRightTurn` | `signalized_junction` | `R1, R4` | 10 / 46 | `junction_pre_m=50, junction_post_m=20` | `auto-ok` | - |
| `StaticCutIn` | `static_cutin` | `R1, R3, R4, R6` | 2 / 10 | `parking_pre_m=35, parking_post_m=55, merge_pre_m=35, merge_post_m=55` | `auto-ok` | - |
| `T_Junction` | `signalized_junction` | `R1, R4` | 6 / 30 | `junction_pre_m=50, junction_post_m=20, review_if_no_tl=true` | `auto-ok` | - |
| `VehicleOpensDoorTwoWays` | `vehicle_opens_door_twoways` | `R1, R2, R4, R6` | 2 / 10 | `two_way_min_pre_m=50, two_way_post_pad_m=20, parking_pre_m=35, parking_post_m=55` | `auto-ok` | - |
| `VehicleTurningRoute` | `vehicle_turning` | `R1, R4, R5` | 10 / 50 | `junction_pre_m=50, junction_post_m=20, multi_trigger=true` | `auto-ok` | - |
| `VehicleTurningRoutePedestrian` | `vehicle_turning` | `R1, R4, R5` | 2 / 10 | `junction_pre_m=50, junction_post_m=40, pedestrian_not_rs` | `auto-ok` | - |
| `noScenarios` | `noscenario` | `R1, R4` | 7 / 35 | `junction_pre_m=50, junction_post_m=25, conservative=true` | `auto-ok` | - |

## 5.9 初始阈值的使用边界与下一步人工校准

本轮参数按 scenario 机制分成 6 类，后续调参必须保留这层分组，而不是把所有场景混成
一个全局阈值：

1. `signalized_junction`: `junction_pre_m=50~60`、`junction_post_m=20~25`。
   灯态有效且 route/junction 同源时 primary=R4；无灯或短时缺灯时不能直接 R5。
2. `nonsignalized_junction`: `junction_pre_m=45~60`、`junction_post_m=20`。
   连续无有效灯态、stop/yield 或 route priority 证据成立时 primary=R5。
3. `twoways_obstacle` / `invading_turn`: `two_way_min_pre_m=45~80`、
   `two_way_post_pad_m=20`、`trigger_close_m=70~75`。
   只有对向车道参与决策、双向单车道拓扑或 trigger/active window 共同成立时 primary=R2。
4. `highway_merge` / `interurban`: `merge_pre_m=30~50`、`merge_post_m=40~50`、
   `trigger_close_m=90`。必须有 ramp/merge/split/highway 或 route-lane topology 支撑 R3。
5. `parking` / `parking_exit` / `static_cutin`: `parking_pre_m=20~35`、
   `parking_post_m=50~60`。R6 必须有 parking/shoulder/curb/side-vehicle 结构证据；
   信号灯路口段仍由 R4/R5 优先。
6. `default_meta_map` / `same_direction_obstacle` / `noscenario`: 默认 R1，
   只有灯态和 junction 同源时临时进入 R4；障碍、行人、刹车、control loss 本身不升级 RS。

每个场景下一步要人工检查：

- `maps/<Town>/<run_id>__route_trigger_ego_trace.png`：ego trace 是否贴 XML route、trigger 是否落在合理位置。
- `rgb/<Town>/<run_id>__sample_contact_sheet.jpg`：每个抽样 id 的 first/middle/last 视觉道路结构是否支持 XODR/XML 判断。
- `meta/<Town>/<run_id>__frame_features.jsonl`：边界帧附近的灯态、junction、active scenario、
  speed 和 `dist_to_*` 是否与初始阈值一致。
- `rules/thresholds.json`：把每个阈值从 `temporary_default_rule_config` 改成带
  `source/supporting_runs/reviewed_artifacts/reason` 的正式阈值。

如果用户后续指出某一帧 RS 错，先回查该 scenario 调研包，不直接改代码：

```text
README -> maps route_trigger_ego_trace -> xml/xodr summary -> meta frame_features ->
rgb contact/boundary frames -> thresholds.json -> runtime rule branch
```

## 5.10 调研结果分析：从 evidence 到逐场景规则

本轮 5-id/town 产物不能只作为“采样完成”记录使用，必须转成帧级规则边界。
下面是把 XML/XODR/meta/RGB 证据映射到代码逻辑时的分层结论。

### A. 同向静态障碍类：事件不等于 RS

适用场景：`Accident`、`ConstructionObstacle`、`ParkedObstacle`。

调研结论：

- XML 的 accident/construction/parked trigger 用来定位障碍事件窗口，不是 R2/R6 的直接边界。
- XODR 若仍为同向 driving lane，且没有对向借道、parking/shoulder/curb 结构证据，primary 保持 R1。
- 有效 traffic light / controller / junction 近邻成立时可进入 R4；障碍距离字段只提高 EVENT 置信度。
- `ParkedObstacle` 名称里有 parked，但本轮规则明确 veto R6；只有 `ParkedObstacleTwoWays`
  这类对向绕行机制才可能在核心窗口进入 R2。

初始阈值：`junction_pre_m=60`、`junction_post_m=25`，并设置 `no_r2/no_r6` 或
`parked_not_parking_rs` veto。高置信需要 XML obstacle window、XODR 同向 lane、
meta active/dist_to_* 至少三源一致；如果 RGB 显示实际占道需要借对向，降级 review 而不是直接改 R2。

### B. TwoWays/对向参与类：R2 必须有拓扑与窗口双证据

适用场景：`AccidentTwoWays`、`ConstructionObstacleTwoWays`、`HazardAtSideLaneTwoWays`、
`ParkedObstacleTwoWays`、`InvadingTurn`、`VehicleOpensDoorTwoWays`。

调研结论：

- `TwoWays` 只说明候选池包含 R2；帧级 R2 必须同时满足 trigger/active window 与 XODR
  对向 driving lane、双向单车道或 lane sign/road topology 证据。
- `InvadingTurn` 是被动对向侵占，不是自车主动绕障；R2 表示对向车道参与决策，EVENT 再区分 U-E5。
- `VehicleOpensDoorTwoWays` 还要保留 R6 secondary：开门车辆靠边/停车结构成立时记录，
  但 primary 在对向绕行核心窗口优先 R2。
- 若只有 `dist_to_*` 变小、RGB 看见障碍但 XODR 没有对向结构，标 medium/low + review。

初始阈值：`two_way_min_pre_m=45~80`、`two_way_post_pad_m=20`、`trigger_close_m=70~75`。
`HazardAtSideLaneTwoWays` 和 `InvadingTurn` 预进入距离更长，原因是侧向危险/对向侵占需要提前让行；
`ParkedObstacleTwoWays` 和开门类保持 50m 左右，避免把远处停车道路全打成 R2。

### C. 高速/合流/匝道类：R3 只由道路结构支持

适用场景：`EnterActorFlow`、`EnterActorFlowV2`、`HighwayCutIn`、`HighwayExit`、
`MergerIntoSlowTraffic`、`MergerIntoSlowTrafficV2`、`InterurbanActorFlow`。

调研结论：

- R3 不是“有车辆流入/切入”的事件标签；必须有 highway/ramp/merge/split 或 XODR lane
  拓扑支撑。
- `EnterActorFlow*` 是自车进入车流，`HighwayCutIn` 是他车切入，二者 EVENT 不同，但 RS
  都依赖合流/高速结构。
- `HighwayExit` 的 R3 退出窗口比 EnterFlow 更长，避免驶出匝道后过早恢复 R1。
- `MergerIntoSlowTraffic*` 即使速度低，只要 XODR/route 仍在合流结构内，保留 R3。
- `InterurbanActorFlow` 允许 R3/R4/R5 并存候选，因为同一 route 可能先有城际合流，再到路口。

初始阈值：`merge_pre_m=30~50`、`merge_post_m=40~50`、`trigger_close_m=90`。
R3 的 high confidence 必须有 XML trigger/route、XODR merge/highway、meta active 或 lane 变化三源一致；
没有 XODR 拓扑时不能只靠 scenario 名输出 R3。

### D. 信号灯路口类：R4 覆盖阻塞/违规事件

适用场景：`BlockedIntersection`、`OppositeVehicleRunningRedLight`、`RedLightWithoutLeadVehicle`、
`SignalizedJunctionLeftTurn`、`SignalizedJunctionLeftTurnEnterFlow`、`SignalizedJunctionRightTurn`、
`T_Junction`。

调研结论：

- R4 由有效灯态、signal/controller、受控 junction 和 route 接近窗口共同决定。
- `BlockedIntersection` 的堵塞、`OppositeVehicleRunningRedLight` 的对方违规，都属于 EVENT；
  只要灯控结构成立，primary 仍是 R4。
- `SignalizedJunctionLeftTurnEnterFlow` 包含 EnterFlow 字样，但本轮 veto R3；该场景是灯控左转进入流，
  不是高速/匝道合流。
- `T_Junction` 如果某些 run 缺有效灯态，需要 `review_if_no_tl=true`，不能自动降到 R5。

初始阈值：`junction_pre_m=50~60`、`junction_post_m=20~25`。
高置信 R4 需要 meta 灯态或 light_hazard、XODR signal/controller、XML junction route 窗口一致；
灯态短暂缺失但 XODR/route 强一致时保持 medium R4 并 review。

### E. 无信号/路权/故障路口类：R5 的边界不能被灯态误导

适用场景：`NonSignalizedJunctionLeftTurn`、`NonSignalizedJunctionLeftTurnEnterFlow`、
`NonSignalizedJunctionRightTurn`、`OppositeVehicleTakingPriority`、`PriorityAtJunction`、
`CrossJunctionDefectTrafficLight`。

调研结论：

- R5 由无有效 traffic light、stop/yield/priority、无信号 junction 或 defect traffic light
  机制支撑。
- `CrossJunctionDefectTrafficLight` 即使 XML/XODR 位置存在 signal，也必须按故障机制 override 到 R5；
  这类错帧优先查 scenario tag/meta defect，而不是查灯控存在性。
- `NonSignalizedJunctionLeftTurnEnterFlow` 不是 R3；enter flow 是路口交互事件，不是合流结构。
- 当前唯一自动缺口是 `NonSignalizedJunctionLeftTurn/Town10HD` 无可读 meta；该 town 已跳过样本调研，
  后续只能等待补 meta 或人工确认，不能用别的 town 阈值冒充完整调研。

初始阈值：`junction_pre_m=45~60`、`junction_post_m=20`。
R5 high confidence 需要 XML route 接近 junction、XODR junction/stop/yield 或无 signal 证据、
meta active/junction 状态一致；如果只靠 scenario 名，必须 low confidence。

### F. Parking/路边结构类：R6 是空间结构，不是 parked 字样

适用场景：`ParkingCrossingPedestrian`、`ParkingCutIn`、`ParkingExit`、`StaticCutIn`。

调研结论：

- R6 必须由 parking/shoulder/curb/side-lane/停车起步空间支撑。
- `ParkingCrossingPedestrian` 的行人横穿是 EVENT，R6 只覆盖停车空间窗口；如果处于灯控路口，R4 优先。
- `ParkingExit` 是从停车区汇入主路，R6 到 R1/R4 的退出边界要看 route/lane topology 和 ego trace。
- `StaticCutIn` 是混合场景：起点可能为 parking，后续可能 merge/highway 或普通道路，因此候选保留
  R1/R3/R4/R6，并要求每个 run 看 map/RGB。

初始阈值：`parking_pre_m=20~35`、`parking_post_m=50~60`；
`StaticCutIn` 同时记录 `merge_pre_m=35`、`merge_post_m=55`。R6 不能由 `dist_to_parked_obstacle`
单独触发，必须有 XODR 或 RGB/map 支持。

### G. 默认 meta/map 类与 noScenarios：保守 R1

适用场景：`ControlLoss`、`CrossingBicycleFlow`、`DynamicObjectCrossing`、`HardBreakRoute`、
`HazardAtSideLane`、`noScenarios`。

调研结论：

- 这些 scenario 的核心变化是车辆行为、动态对象、急刹或侧向风险，不是道路结构变化。
- 默认 primary=R1；只有灯态/junction 强证据成立时短暂进入 R4。
- `CrossingBicycleFlow`、`DynamicObjectCrossing`、`HardBreakRoute` 的对象/刹车字段只进入 EVENT span。
- `HazardAtSideLane` 不因 side lane 字样自动 R2；除非后续 map/RGB 显示对向参与，否则保持 R1/R4。

初始阈值：`junction_pre_m=50`、`junction_post_m=25`，加对应 veto
`control_loss_not_rs`、`actor_flow_not_r3`、`crossing_event_not_rs`、`brake_not_rs`、
`side_lane_not_twoways`、`conservative=true`。

### H. 置信度与错帧归因

所有场景统一采用三档置信度，但归因必须写到具体输入：

- high：scenario prior、XML window、XODR topology、meta signal/junction/active 至少三源一致，
  且 map/RGB 人工检查无冲突。
- medium：两源一致，或强 meta 信号成立但 XODR/RGB 尚未人工确认；边界帧默认不高于 medium。
- low：只有 scenario prior、XODR 弱 hint、meta 缺失、投影误差大或 map/RGB 冲突。

错帧回查时按错误类型分流：

- R4/R5 错：优先查 traffic light state、XODR signal/controller、defect/non-signalized scenario tag。
- R2 错：优先查 XODR 对向/双向 lane、trigger 投影距离、active scenario 与 RGB 是否真的需要借对向。
- R3 错：优先查 merge/split/highway/ramp 拓扑，不查 cut-in 事件本身。
- R6 错：优先查 parking/shoulder/curb/停车起步空间，不查 parked obstacle 字段本身。
- R1 错：检查是否有被 veto 的事件误当结构，或是否 junction/merge/parking 窗口阈值过窄。

## 6. 场景触发窗口（Scenario Window）融合

你现有数据里 `current_active_scenario_type` 已经非常有价值，可直接并入规则层。

推荐策略：

1. 由 `current_active_scenario_type == scenario_name` 形成原始激活掩码。
2. 与 XML trigger 投影、route 进度、CARLA map 拓扑共同形成 ROAD_STRUCTURE 候选窗口。
3. 与距离字段阈值交集，得到 EVENT 可见/可感知窗口。
4. 结构窗口内允许 R2/R3/R6 获得更高先验分；事件窗口内只提升 U-E2/U-E3/U-E4
   等突发事件置信度，不能反过来单独决定道路结构。

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
- 若可见性门控命中，也只能作为 ROAD_STRUCTURE 置信度的辅助证据；例如 R6 必须仍有
  停车 lane / 停车区 / Parking* 场景结构窗口等证据。

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

实际实现更推荐 evidence-aware Viterbi，而不是无脑众数平滑：

- 强证据帧不可被邻域多数票覆盖，例如连续有效灯态、明确 active R3 合流窗口、明确 R2 对向车占道；
- 低置信孤立帧可以合并到左右同类标签；
- R2/R6/R4/R5 的边界帧应保留 `transition_margin=true`，训练时可降权或排除；
- 对 `review_required=true` 的帧，不要让平滑结果升级为高置信。

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
  "primary_road_structure": "R4",
  "secondary_road_structures": ["R6"],
  "physical_road_structure_hint": ["signalized_junction", "parking_context"],
  "confidence_level": "high",
  "review_required": false,
  "transition_margin": false,
  "road_structure_candidates": {
    "R1": 0.12,
    "R4": 0.91,
    "R6": 0.64
  },
  "arbitration_reason": "traffic_light_priority_over_parking_context",
  "confidence": 0.91,
  "evidence": {
    "is_junction": false,
    "traffic_light_state": "Green",
    "current_active_scenario_type": "ParkingCutIn",
    "dist_to_cutin_vehicle": 9.8,
    "route_projection_error_m": 0.42,
    "junction_id_decision": 17,
    "label_source": ["traffic_light_state", "route_junction_match", "parking_context_secondary"],
    "rules_fired": ["r4_tl_confirmed", "r6_parking_context_secondary", "priority_override"]
  }
}
```

注意保留 `evidence.rules_fired`，便于后续人工抽检和阈值调参。训练主监督默认使用
`primary_road_structure`；`secondary_road_structures` 只用于审计、可视化和少量重叠样本分析，
不要把大批帧变成多标签监督。

建议额外输出一个 run 级 `review_spans` 列表，集中收集以下情况：

- XML 缺失、town/xodr 缺失、route projection error 过大；
- 有效灯态存在但 route/junction 同源校验失败；
- R2/R6/R4/R5 分数接近且主标签仲裁不稳定；
- 平滑前后标签变化超过阈值；
- `pos_global` 与 `ego_matrix[:3,3]` 明显不一致；
- scenario 不在候选映射表内，或本地 XML 尚未覆盖该 scenario。

---

## 9. 与你现有 ROAD/EVENT 体系的衔接

你已经有两份关键文档：

- `ROAD_EVENT_CLASSIFICATION_PLAN.md`
- `ROAD_EVENT_CANDIDATE_MAPPING.md`

衔接方式：

1. 本方案先生成每帧 `primary_road_structure`，必要时附带少量 `secondary_road_structures`。
2. 再按映射表第 4/5 节，得到每帧 `EVENT` 候选集合。
3. 最后再决定是否让 Qwen 只在“候选集合内”做细粒度判别。

这样 Qwen 不再需要先猜你在哪种道路结构，错误会明显减少。

---

## 10. 分阶段落地计划（建议）

## Phase A（1~2 天）：不依赖 CARLA 地图，先跑通

输入：`metas + route_xml + candidate_mapping`

- 完成 run->xml 对齐
- 完成帧级 `R1~R6` 候选召回与 primary 仲裁初版（主要依赖 meta 字段）
- `distance_to_next_junction < 50m` 只召回 R4/R5 候选；最终标签仍需灯态、route 穿越路口、
  scenario 先验与时序确认。
- R2/R6 先用 scenario + active/trigger 窗口做弱结构候选，`dist_to_*` 只做事件可见门控。
- 输出每帧 primary/secondary/candidates/evidence。
- 所有缺少 XML、route projection error 大、R2/R6 只有 scenario 证据的帧，默认 `confidence_level=low/medium`，
  不进入强监督集合。

目标：先把 Qwen 的结构判断替换掉 70% 以上。

## Phase B（2~4 天）：引入 CARLA 地图拓扑增强

- 用 `carla.Map.get_waypoint` / junction / lane topology 增强 R2/R3/R4/R5 边界
- 对 XML route 做 1~2m densify，建立 frame -> route_s -> upcoming junction/stopline 的稳定索引
- 增加 ramp/merge 检测与对向车道参与检测
- 增加 parking lane / shoulder / 路边停车结构检测，减少 R6 与 R1/R4 的混淆。
- 对 XML waypoint 很稀疏的 route（例如 ParkingExit 可能只有 2 个点），用 CARLA map
  路网与 meta 位姿补足帧级边界，不让稀疏 polyline 单独决定标签切换。
- 建立受控 junction/stopline 索引，用于校验 `traffic_light_state` 是否属于当前 route 决策路口。

目标：减少 R2/R3 与 R4/R5 混淆。

## Phase C（持续）：阈值自动校准与评估

- 抽样人工标注 200~500 帧作验证集
- 调整阈值与优先级
- 增加失败案例回放清单

---

## 11. 关键实现细节与风险提示

1. `metas/*.pkl` 是 xz 压缩，读取必须 `lzma.open`。
2. 很多距离字段会出现 `inf`，计算阈值前要先过滤。
3. `current_active_scenario_type` 不是全程有效，常出现 `None`，必须与 XML trigger、
   route 进度、地图拓扑和距离字段分层联合。
4. `traffic_light_state` 可能短时抖动，不可单帧硬判 R4/R5。
5. `distance_to_next_junction < 50m` 是候选召回，不是最终 R4/R5 标签阈值。
6. 若拿不到 CARLA 地图，不影响第一版上线；先做 meta+xml 版即可，但 R2/R3/R6
   边界要标低置信并进入抽检队列。
7. `traffic_light_state` 有效不等于当前 route 正受该灯控制；必须结合 route upcoming junction/stopline
   或 hazard/stopline 相关字段做同源校验。
8. `ROAD_STRUCTURE` 是当前决策规则空间，不是纯物理道路分类；物理双向单车道或停车带可以作为 hint，
   但不能独自决定训练主标签。
9. 本地 `data/lead` 覆盖 43 类 scenario；2026-07-03 全量核对为
   `lead_data` 9715 个 run、9294 个唯一 `(Scenario,Town,route_key)`，`data/lead`
   正好 9294 个 XML，缺失 0、冗余 0、命名不规范 0、XML 解析失败 0、内容结构异常 0。
   未命中 XML 只应出现在后续数据版本漂移时，先从 `data_routes` 补齐；
   确实补不了且 `data/lead` 也没有有效 XML 时，再走 meta/map 降级路径，并把
   run 级原因写入 `review_spans`。
10. 40 个 XML 的 `data_routes` 源文件位于不同 scenario 目录（36 个 `noScenarios`、
    4 个 `ConstructionObstacleTwoWays`），但都能按 `(town, route_key)` 找到源 XML；
    这不是缺失项，标注与代码逻辑以 `lead_data` / `data/lead` 的 scenario 目录为准。
11. `ParkedObstacle/Town12_route_Town12_route15.xml` 覆盖有效并与
    `lead_data/ParkedObstacle/Town12_Rep0_Town12_route15_*` 对应，但未在
    `AutoMoT/data/data_routes` 找到直接源文件。它不是 `xml_available=false`，
    后续代码应直接使用 `data/lead` 中的现有 XML。

---

## 12. 伪代码（可直接转脚本）

```text
for run in all_runs:
    scenario = parse_scenario_from_path(run)
    town, route_key = parse_town_route_key(run)
    xml = load_route_xml(scenario, town, route_key)
    route_polyline = xml.waypoints
    trigger_points = xml.scenarios.trigger_point

    frames = load_all_metas_lzma(run)
    primary_labels = []
    secondary_labels = []
    candidate_scores = []
    events = []
    evidence = []

    for t, meta in enumerate(frames):
        ego = extract_ego_xyz_yaw(meta)
        s, d = project_to_route_polyline(ego.xy, route_polyline)

        feat = build_features(meta, ego, s, d)
        # feat: is_junction, traffic_light_state, dist_to_*, active_scenario, ...

        candidates = init_candidate_scores(R1..R6)
        # 50m 只召回 R4/R5 候选；最终确认还要看 route/junction/灯态/时序。
        candidates += recall_r4_r5_candidates(feat, scenario, route_polyline, map_feat)
        candidates += recall_r3_candidates(feat, scenario, map_feat)
        candidates += recall_r2_candidates(feat, scenario, map_feat)
        candidates += recall_r6_candidates(feat, scenario, map_feat)
        candidates += apply_r1_default(feat)

        primary, secondary = arbitrate_primary_road_structure(
            candidates,
            priority=[R4/R5, R3, R2/R6, R1],
            keep_secondary_only_if_meaningful=True,
        )
        primary_labels.append(primary)
        secondary_labels.append(secondary)
        candidate_scores.append(candidates)
        evidence.append(collect_fired_rules())

        # 事件注入要做“可见性门控”，不要把 xml trigger 直接当作 U-E2 起点
        event_gate = build_event_visibility_gate(feat, s, trigger_points)
        events.append(inject_events_with_hysteresis(primary, feat, event_gate))

    primary = temporal_smoothing(primary_labels)
    secondary = temporal_smoothing_secondary(secondary_labels)
    events = temporal_smoothing_events(events)
    dump_framewise_road_and_events(run, primary, secondary, candidate_scores, events, evidence)
```

---

## 13. 结论

你的思路是对的，而且从现有数据字段看已经具备可实施条件。

最务实路径是：

1. 先用 `meta + route xml` 做帧级 ROAD_STRUCTURE 候选与中高置信主标签；
2. R2/R6 做决策窗口触发，不做全程标签；
3. 后续接 CARLA 地图拓扑、route densify、受控 junction/stopline 同源校验来增强边界；
4. 输出 `confidence_level`、`review_required`、`transition_margin`，低置信和过渡帧不直接当强监督；
5. Qwen 只在候选空间内做细分，不再承担主结构判断。

这条线可以显著提升一致性，也更容易解释和调参。
