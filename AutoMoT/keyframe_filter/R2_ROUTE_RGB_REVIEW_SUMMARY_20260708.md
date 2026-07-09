# R2 Route RGB Review Summary 2026-07-08

本轮目标：重新核查 R2 是否真的表示“对向单车道 / 有效可行驶通道被压缩成对向单车道”，避免普通
R1/R-E1 场景被 XODR opposite hint 或场景名误升为 R2。

## Runtime Gate

- 非 TwoWays 场景不再根据场景级 XODR sparse scan 批量开放 R2。
- `LAYOUT_R2_ROUTE_IDS` 当前为空；后续只有逐 route / 逐帧 RGB 确认后，才能把非 TwoWays route
  手工加入白名单。
- 当前允许 primary R2 的来源只剩：
  - `*_TwoWays`：有效可行驶通道为对向单车道；真实灯控 / STOP / 无灯路口仍可覆盖为 R4/R5。
  - `InvadingTurn`：只有对向侵占/双向窄路规则空间成立时才给 R2；对向侵入动作本身由 U-E5 表达。

## Full R2 Candidate Rerun

输出目录：`AutoMoT/keyframe_filter/collection_output/current_r2_reaudit_20260708`

| Scenario | Annotated Routes | Frames | R2 Routes | Primary RS |
|---|---:|---:|---:|---|
| AccidentTwoWays | 457 | 71237 | 457 | R2 80.0%, R4 16.0%, R5 4.0% |
| ConstructionObstacleTwoWays | 454 | 70530 | 454 | R2 81.8%, R4 14.2%, R5 4.0% |
| HazardAtSideLaneTwoWays | 88 | 14923 | 88 | R2 87.2%, R4 6.3%, R5 6.5% |
| ParkedObstacleTwoWays | 96 | 14030 | 96 | R2 81.1%, R4 3.0%, R5 15.9% |
| VehicleOpensDoorTwoWays | 104 | 14157 | 104 | R2 44.8%, R4 49.9%, R5 5.3% |
| InvadingTurn | 98 | 11883 | 59 | R1 72.4%, R2 12.7%, R4 1.0%, R5 13.9% |

普通非 TwoWays 风险场景的当前专项结果：

| Scenario | R2 Routes | Primary RS |
|---|---:|---|
| Accident | 0 | R1 87.0%, R4 13.0% |
| ConstructionObstacle | 0 | R1 85.2%, R4 14.8% |
| ControlLoss | 0 | R1 78.5%, R4 15.0%, R5 6.6% |
| BlockedIntersection | 0 | R1 5.7%, R4 64.6%, R5 29.6% |
| CrossJunctionDefectTrafficLight | 0 | R1 7.4%, R4 92.6% |
| CrossingBicycleFlow | 0 | R1 37.0%, R4 63.0% |

## RGB Evidence

逐帧 contact sheet 输出：

- `AutoMoT/keyframe_filter/collection_output/r2_rgb_route_sheets_20260708/route_sheet_manifest.csv`
- `AutoMoT/keyframe_filter/collection_output/r2_rgb_route_sheets_20260708/scenario_sheet_summary.csv`
- `AutoMoT/keyframe_filter/R2_ROUTE_RGB_REVIEW_INDEX_20260708.csv`

覆盖：

| Scenario | RGB Routes | RGB Frames | Sheet Pages | Abnormal Routes | Annotated R2 Routes | R2 Frames |
|---|---:|---:|---:|---:|---:|---:|
| AccidentTwoWays | 596 | 128415 | 898 | 85 | 457 | 57013 |
| ConstructionObstacleTwoWays | 559 | 119615 | 852 | 94 | 454 | 57698 |
| HazardAtSideLaneTwoWays | 101 | 22440 | 149 | 12 | 88 | 13013 |
| InvadingTurn | 102 | 14583 | 117 | 4 | 59 | 1506 |
| ParkedObstacleTwoWays | 102 | 16573 | 122 | 5 | 96 | 11375 |
| VehicleOpensDoorTwoWays | 117 | 19349 | 154 | 7 | 104 | 6338 |

## Visual Spot Checks

- `ConstructionObstacleTwoWays/Town12_Rep0_1977_0_route0_01_08_09_32_37`：
  夜间无中央隔离窄路，路侧桩/对向交互明显，R2 方向合理；但夜间欠曝，保留 review。
- `ParkedObstacleTwoWays/Town12_Rep0_1212_0_route0_01_10_00_58_25`：
  无隔离双向窄路，路边停车占用空间，R2 合理，不是普通宽路停车。
- `VehicleOpensDoorTwoWays/Town12_Rep0_1544_0_route0_01_09_01_34_14`：
  路窄且两侧停车/开门风险明显，R2 可成立；但该场景 R4 占 49.9%，必须继续核查是否远灯/弱灯控过度覆盖。
- `HazardAtSideLaneTwoWays/Town12_Rep0_467_0_route0_01_10_18_16_11`：
  夜间画面过暗，只能看到对向灯/窄路轮廓；规则输出不能视为视觉完全确认，保留高优先级 review。
- `InvadingTurn/Town12_Rep0_1150_0_route0_01_09_20_35_02`：
  开头为路口/斑马线区域，当前 primary 是 R4/R5/R1，没有把 R2 强压为 primary；U-E5 是否过早仍需逐帧复核。

## Current Conclusion

- R2 的定义应继续收紧为“道路/可行驶空间层面的对向单车道”，不是普通障碍、停车、开门或 R-E1 背景。
- R1 被纠正为 R2 时只修改 ROAD_STRUCTURE，不自动修改 EVENT；R2/R-E1 是正常对向单车道通行背景。
  U-E2/R-E2 必须由 XML trigger、具体障碍距离、TwoWays core/stuck/hazard、door/open、
  强制制动近障碍或真实借道/回正轨迹等独立事件证据触发。
- 普通非 TwoWays 场景当前没有 R2 泄漏；后续若要恢复非 TwoWays 的 R2，必须通过
  `R2_ROUTE_RGB_REVIEW_INDEX_20260708.csv` 逐 route 审核后写入 `LAYOUT_R2_ROUTE_IDS`。
- `VehicleOpensDoorTwoWays`、`HazardAtSideLaneTwoWays`、`InvadingTurn` 是下一轮最需要人工逐帧看的三类：
  前者检查 R4 是否过宽，后两者检查夜间/路口下 R2 或 U-E5 是否过早。
