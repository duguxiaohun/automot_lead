# No-R6 ROAD / EVENT RGB Audit 2026-07-06

本轮目的：检查旧 R6 相关场景在删除独立停车/遮挡 ROAD_STRUCTURE 后，是否已经被正确融入
R1/R2/R4/R5，并同步检查对应 EVENT 是否仍和 RGB 可见动作一致。

## Scope

覆盖旧 R6 高风险场景：

- `ParkingCrossingPedestrian`
- `ParkingExit`
- `ParkingCutIn`
- `StaticCutIn`
- `VehicleOpensDoorTwoWays`
- `ParkedObstacle`
- `ParkedObstacleTwoWays`

全量标注命令：

```bash
python AutoMoT/keyframe_filter/quick_start.py annotate-rs \
  --scenario ParkingCrossingPedestrian,ParkingExit,ParkingCutIn,StaticCutIn,VehicleOpensDoorTwoWays,ParkedObstacle,ParkedObstacleTwoWays \
  --max-frames-per-route 0 \
  --output-dir /tmp/automot_no_r6_all_routes_postfix_20260706
```

结果覆盖 `895` 条可标注 route、`106471` 帧；另有 `16` 条缺 `meta.pkl` 的 route 按数据质量跳过。

## Full-Route Distribution

ROAD_STRUCTURE：

| RS | Frames |
|---|---:|
| R1 | 43444 |
| R2 | 17537 |
| R3 | 5070 |
| R4 | 32463 |
| R5 | 7957 |

EVENT：

| EVENT | Frames |
|---|---:|
| R-E1 | 40220 |
| R-E2 | 12322 |
| R-E3 | 1070 |
| R-E4 | 32055 |
| R-E5 | 7950 |
| U-E2 | 8915 |
| U-E3 | 2272 |
| U-E4 | 1667 |

硬一致性检查：

- runtime 枚举、候选池和本轮输出中均无 R6 类别。
- `primary_road_structure` 越过场景 RS 候选池：0。
- `primary_event` 越过当前 RS allowed events：0。

## Scenario Conclusions

| Scenario | RS result | EVENT result | RGB conclusion |
|---|---|---|---|
| `ParkingCrossingPedestrian` | R1/R4/R5 | R-E1/R-E4/R-E5/U-E4 | 停车环境不再成为 RS；行人横穿由 U-E4 表达，灯控/无灯控制区仍走 R4/R5。 |
| `ParkingExit` | R1/R4 | R-E1/R-E2/R-E4 | 自车从路边/停车侧并入主路是 R1 + R-E2；并入完成回 R-E1，灯控区 R4/R-E4。 |
| `ParkingCutIn` | R1/R4/R5 | R-E1/R-E4/R-E5/U-E3 | 切入车辆是 U-E3，不再生成停车 RS，也不漏出候选外 R-E2。 |
| `StaticCutIn` | R1/R3/R4/R5 | R-E1/R-E2/R-E3/R-E4/R-E5/U-E3 | 高速/merge 桶保留 R3，其余城市/普通段回 R1/R4/R5；静态切入仍用 U-E3。 |
| `VehicleOpensDoorTwoWays` | R2/R4/R5 | R-E1/R-E2/R-E4/R-E5/U-E2 | 双向窄路/开门压缩可行驶空间为 R2，开门/避让核心 U-E2，恢复回正 R-E2。 |
| `ParkedObstacle` | R1/R4/R5 | R-E1/R-E2/R-E4/R-E5/U-E2 | 停放车辆是障碍 EVENT，不是 ROAD_STRUCTURE；绕行核心 U-E2，回目标车道 R-E2。 |
| `ParkedObstacleTwoWays` | R2/R4/R5 | R-E1/R-E2/R-E4/R-E5/U-E2 | 双向单车道/受停放障碍压缩后的有效通道为 R2；STOP/无灯段 R5，障碍交互 U-E2/R-E2。 |

## Manual RGB Checks

全量逐帧标注之后，额外生成了旧 R6 高风险 route 的 RGB contact sheets，重点看
长 `R-E2`、TwoWays 障碍核心、R4/R5 边界和夜间低能见度样本：

```bash
python AutoMoT/keyframe_filter/rs_full_frame_review.py \
  --scenario ParkingCrossingPedestrian,ParkingExit,ParkingCutIn,StaticCutIn,VehicleOpensDoorTwoWays,ParkedObstacle,ParkedObstacleTwoWays \
  --samples-per-town 0 --max-routes-per-town 0 --max-frames-per-route 0 \
  --output-dir /tmp/automot_no_r6_full_rgb_review_20260706 \
  --frames-per-sheet 80 --sheet-cols 4
```

人工 RGB 复核要点：

- `ParkingExit/Town10HD_Rep0_route_001645...`：f0-65 为停车侧/路边并入主路，R1/R-E2 正确；f66 后进入灯控区，R4/R-E4 正确。
- `ParkingExit/Town03_Rep0_route_001731...`：f0-52 为 curbside merge，R1/R-E2 正确；之后稳定跟车回 R1/R-E1。
- `ParkingExit/Town12_Rep0_2179_2...`：f0-50 为停车侧并入，R1/R-E2 正确；之后回 R1/R-E1。
- `ParkingCrossingPedestrian/Town13_Rep0_1085_0...`：前段 R5/R-E5，行人横穿核心切 U-E4，之后回 R1/R-E1；没有独立停车 RS。
- `VehicleOpensDoorTwoWays/Town13_Rep0_route_000011...`：全程非路口主体为 R2；开门/侧向占道核心为 U-E2，恢复阶段 R-E2/R-E1，末尾可见灯控区转 R4/R-E4。
- `ParkedObstacleTwoWays/Town13_Rep0_1351_2...`：起点 STOP/无灯控制区 R5/R-E5，随后夜间双向窄道绕停放障碍为 R2/U-E2，绕过后回正 R-E2/R-E1。

结论：旧 R6 语义已经拆成“当前道路结构 R1/R2/R4/R5 + 停车/开门/遮挡/绕障 EVENT”。
未见需要恢复独立 R6 的 RGB 证据。

## Bug Fixed In This Pass

全量初跑发现 6 帧 EVENT 候选越界，典型为：

- R1 中漏出 `R-E2`；
- R4 中漏出 `R-E2`；
- R4 本应回 `R-E4`。

原因是 `collector.py::_apply_event_candidate_clamp` 会保留早期 route-level rewrite
遗留在 `event_evidence["regular_event"]` 里的 stale regular event。现在 clamp 改为按当前
`primary_road_structure` 重新计算 regular event：

- R1/R2/R3 regular -> `R-E1`
- R4 regular -> `R-E4`
- R5 regular -> `R-E5`

修复后同一全量重跑的 RS / EVENT 候选越界均为 0。

## Remaining Review Notes

- `u2_reaches_route_end_requires_review=40`、`u2_released_far_from_xml_trigger=69` 是保守 review
  标记，不是硬错误；它们用于提示绕障核心和 XML trigger 投影边界仍需抽查。
- 本轮手工 RGB 复核优先覆盖长 span、边界和低能见度高风险 route；全量 106471 帧均已生成逐帧标注，
  可继续按 route 从 `/tmp/automot_no_r6_all_routes_postfix_20260706` 回查。
