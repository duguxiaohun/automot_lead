# RGB R4/R5 全量审计总结

日期：2026-07-04

本轮目标是按 RGB 逐帧筛选所有 scenario / route id，判断每个场景是否存在：

- R4：有信号灯路口 / 灯控 T 路口 / 灯控十字路口。
- R5：无信号灯、STOP、yield、priority、信号失效或无灯 T/十字路口。
- R4/R5 共有、仅 R4、仅 R5、或没有稳定路口证据。

## 运行命令

```bash
python AutoMoT/keyframe_filter/rgb_r4_r5_audit.py \
  --output-dir /tmp/automot_rgb_r4_r5_full_audit \
  --workers 16 \
  --progress-interval 100
```

输出：

- `/tmp/automot_rgb_r4_r5_full_audit/scenario_rgb_r4_r5_summary.csv`
- `/tmp/automot_rgb_r4_r5_full_audit/scenario_rgb_r4_r5_summary.json`
- `/tmp/automot_rgb_r4_r5_full_audit/route_rgb_r4_r5_audit.csv`
- `/tmp/automot_rgb_r4_r5_full_audit/route_rgb_r4_r5_audit.json`
- `/tmp/automot_rgb_r4_r5_full_audit/evidence_sheets/<Scenario>.jpg`

已保留的可 push 关键结果：

- `AutoMoT/keyframe_filter/rgb_r4_r5_audit_results/scenario_rgb_r4_r5_summary.csv`
- `AutoMoT/keyframe_filter/rgb_r4_r5_audit_results/scenario_rgb_r4_r5_summary.json`
- `AutoMoT/keyframe_filter/rgb_r4_r5_audit_results/route_rgb_r4_r5_audit.csv`
- `AutoMoT/keyframe_filter/rgb_r4_r5_audit_results/rule_update_decisions.csv`
- `AutoMoT/keyframe_filter/rgb_r4_r5_audit_results/evidence_sheets/<Scenario>.jpg`
- `AutoMoT/keyframe_filter/rgb_r4_r5_audit_results/manifest.json`

临时目录中的 `route_rgb_r4_r5_audit.json` 约 62MB，主要重复 route CSV 加示例细节，未入库；需要复跑时用本脚本重新生成。

审计发现 43 个 scenario、9715 个 route。按项目约束先剔除异常时长 route 后，逐帧读取并分析 8752 个有效 route、1102886 张 stitched RGB。

## 判定口径

- RGB 是必读证据：每个有效 route 的每一张 `rgb/*.jpg` 都通过 `cv2.imread` 读取。
- R4 需要 RGB 可疑灯控颜色块，同时有 meta `traffic_light_state/light_hazard` 或 bbox `traffic_light` 同源确认。
- R5 需要 RGB 中 junction/stopline/turn/crossing 线索，同时有 meta junction/stop 或 bbox stop/sign/junction 线索；但高速匝道、导流线、merge 线、停车线不能单独当作无灯十字路口。
- XODR/meta/bbox 用来确认或反证 RGB，不替代 RGB。XODR projection error 大时只作弱提示。
- `CrossJunctionDefectTrafficLight` 即使可见信号灯，也按 defect 语义优先 R5，不按正常 R4 回灌。

## 规则回灌

| 场景 | 有效/总 route | RGB 比例 | 回灌结论 |
|---|---:|---|---|
| EnterActorFlow | 80/93 | R4 0.000 / R5 0.562 | 保持 R3/no-R4；R5 检测多为 actor-flow/路口近邻弱证据，不全局加入 |
| EnterActorFlowV2 | 43/43 | R4 0.000 / R5 0.674 | 保持 R3/no-R4 |
| HighwayCutIn | 75/106 | R4 0.120 / R5 0.613 | 恢复 R4 候选，主体仍 R3；R5 弱证据不进入候选 |
| HighwayExit | 88/98 | R4 0.000 / R5 0.614 | 保持 R3/no-R4；R5 多为出口/分流线弱证据 |
| InterurbanActorFlow | 90/91 | R4 0.000 / R5 0.933 | 保持 R1/R3/R5，删除 R4 |
| InterurbanAdvancedActorFlow | 72/78 | R4 0.000 / R5 0.778 | 保持 R1/R5，删除 R4 |
| InvadingTurn | 98/102 | R4 0.010 / R5 0.357 | 保持 R1/R2/R5，删除稳定 R4；单个 R4 route 先按异常/复核处理 |
| MergerIntoSlowTraffic | 88/96 | R4 0.102 / R5 0.648 | 恢复 R4 候选，主体仍 R3；R5 弱证据不进入候选 |
| MergerIntoSlowTrafficV2 | 103/105 | R4 0.000 / R5 0.709 | 保持 R3/no-R4 |
| NonSignalizedJunctionRightTurn | 93/95 | R4 0.129 / R5 0.860 | 改为 R1/R4/R5；大多数 R5，少量灯控右转逐帧给 R4 |
| OppositeVehicleTakingPriority | 97/97 | R4 0.072 / R5 0.814 | 改为 R1/R4/R5；以 R5 为主，少量灯控子集逐帧给 R4 |
| PriorityAtJunction | 99/99 | R4 0.869 / R5 0.778 | 维持 R1/R4/R5 混合 |
| T_Junction | 247/247 | R4 0.874 / R5 0.684 | 改为 R1/R4/R5；T 形路口按灯控/无灯控制源逐帧区分 |

信号灯明确的 `SignalizedJunction*`、`RedLightWithoutLeadVehicle`、`OppositeVehicleRunningRedLight` 仍以 R4 为主；审计里的 R5 命中多来自 approach/stopline/crosswalk 或遮挡时段，不能据此把 signalized 场景全局改成 R4/R5 共有。无灯/defect 场景则相反，必须优先尊重 scenario 语义和 RGB 控制源。

## 后续使用

- 场景级 no-R4 黑名单只保留全量 RGB 未见稳定灯控的场景：`EnterActorFlow`、`EnterActorFlowV2`、`HighwayExit`、`InterurbanActorFlow`、`InterurbanAdvancedActorFlow`、`InvadingTurn`、`MergerIntoSlowTrafficV2`。
- `HighwayCutIn`、`MergerIntoSlowTraffic`、`NonSignalizedJunctionRightTurn`、`OppositeVehicleTakingPriority`、`T_Junction` 不再用场景名压掉 R4。
- R5 不允许从“没有看见红绿灯”直接得出；必须有无灯/STOP/yield/priority/T-junction 证据。
- `collector.py` 已按本口径接入同帧 `bboxes/*.pkl` 的轻量语义摘要：`traffic_light` 可作为 R4 辅助确认，
  `stop_sign/yield/junction/crosswalk` 可作为 R5 辅助确认；但 highway/merge 场景里的弱灯控 hint 若没有同源控制上下文会被降级，不压过默认 R3。
