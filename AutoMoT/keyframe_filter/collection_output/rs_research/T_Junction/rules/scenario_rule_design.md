# T_Junction ROAD_STRUCTURE 调研设计

## 1. Scenario 语义与候选 RS

- rule_kind: `signalized_junction`
- candidate_pool: `R1, R4`
- towns: `Town01, Town02, Town03, Town04, Town05, Town10HD`

## 2. 样本覆盖

- Town01: Town01_Rep0_Town01_Scenario7_0_route0_01_07_23_32_40, Town01_Rep0_Town01_Scenario7_26_route0_01_10_07_50_07, Town01_Rep0_Town01_Scenario7_42_route0_01_08_11_07_40, Town01_Rep0_Town01_Scenario7_57_route0_01_10_22_26_30, Town01_Rep0_Town01_Scenario7_7_route0_01_11_03_19_28
- Town02: Town02_Rep0_Town02_Scenario7_10_route0_01_10_05_04_32, Town02_Rep0_Town02_Scenario7_22_route0_01_09_23_28_26, Town02_Rep0_Town02_Scenario7_2_route0_01_11_02_29_13, Town02_Rep0_Town02_Scenario7_36_route0_01_08_23_27_24, Town02_Rep0_Town02_Scenario7_45_route0_01_09_01_33_06
- Town03: Town03_Rep0_Town03_Scenario7_102_route0_01_10_18_00_29, Town03_Rep0_Town03_Scenario7_2_route0_01_08_13_44_54, Town03_Rep0_Town03_Scenario7_56_route0_01_11_02_39_49, Town03_Rep0_Town03_Scenario7_74_route0_01_10_12_57_37, Town03_Rep0_Town03_Scenario7_94_route0_01_08_03_23_42
- Town04: Town04_Rep0_Town04_Scenario7_103_route0_01_08_21_20_06, Town04_Rep0_Town04_Scenario7_2_route0_01_10_19_40_54, Town04_Rep0_Town04_Scenario7_52_route0_01_10_05_12_32, Town04_Rep0_Town04_Scenario7_74_route0_01_08_05_10_24, Town04_Rep0_Town04_Scenario7_9_route0_01_09_02_59_59
- Town05: Town05_Rep0_Town05_Scenario7_104_route0_01_09_01_19_38, Town05_Rep0_Town05_Scenario7_143_route0_01_09_04_00_40, Town05_Rep0_Town05_Scenario7_191_route0_01_09_00_06_09, Town05_Rep0_Town05_Scenario7_56_route0_01_09_03_04_38, Town05_Rep0_Town05_Scenario7_9_route0_01_10_02_27_34
- Town10HD: Town10HD_Rep0_Town10HD_Scenario7_10_route0_01_09_20_03_24, Town10HD_Rep0_Town10HD_Scenario7_20_route0_01_10_09_52_38, Town10HD_Rep0_Town10HD_Scenario7_29_route0_01_10_00_09_36, Town10HD_Rep0_Town10HD_Scenario7_49_route0_01_09_15_23_25, Town10HD_Rep0_Town10HD_Scenario7_9_route0_01_08_12_18_11

## 3. XML 使用

XML 用于 route 粗投影、trigger 窗口、scenario tag 参数和数据源追溯；不能单独作为帧级 RS 真值。

## 4. XODR 使用

- XML trigger/waypoints 定位 stopline/junction approach
- XODR signal/controller/junction 支持 signalized 证据；灯色仍以 meta 为准

## 5. Meta 使用

运行时优先读取 `pos_global/theta/speed`、灯态、junction、active scenario 和 finite `dist_to_*` 字段。
meta 缺失时只允许低/中置信临时标签，并写入 `diagnostic_attribution`。

## 6. RGB 人工观察结论

本轮自动生成 5 个分散 id 的 contact sheet，作为人工复核入口。正式 complete 前需要人工检查 contact sheet 与边界帧。
必须重点核验 `rgb/*sample_contact_sheet.jpg` 是否与 `maps/*route_trigger_ego_trace.png` 的道路结构判断一致；
若 RGB 与 XML/XODR 冲突，记录到 `failure_modes.md`，对应帧降到 medium/low confidence。

## 7. 自车轨迹与地图对齐

每个抽样 run 已生成 route/trigger/ego trace 图。若 trace 与 XML route 长期偏离，应标记 `projection_untrusted`。
用于调阈值的 run 需要满足：route projection median error <= 3m、p90 error <= 5m、trigger 到 ego trace 最近距离 <= 20m。

## 8. 帧级 RS 分段逻辑

- 有效 Red/Yellow/Green 或 light_hazard -> R4
- scenario trigger 对应受控 junction 窗口 -> R4 medium/high
- 路口外跟车/离开背景 -> R1

## 9. 置信度规则

- high: scenario prior + XML window + XODR topology + meta signal/junction/active 至少三源一致。
- medium: 两源一致，或强 meta 信号成立但 XODR/RGB 支持不足。
- low: 只有 scenario prior 或弱 XODR hint；必须 review。

## 10. Review 规则

出现 XML 缺失、XODR 缺失、meta 缺失、route projection error、候选分数接近、RGB 与地图冲突时必须 review。

## 11. 已知失败模式

- 当前自动审计未发现 town 级输入缺口；仍需人工复核 RGB 边界帧后才能最终确认。

## 12. 当前样本逻辑备注

- 抽样 XML trigger/route 附近有 13 个样本靠近 XODR signal，R4/故障灯窗口有空间证据。
- 抽样 LEAD meta 中有 30 个 route 样本含 traffic_light_state 分布，可用于运行时 R4/R5 复核。
- 抽样 LEAD meta 可读 route 数=30；active_scenario 可读 route 数=30。

## 13. 阈值来源与代码配置

`thresholds.json` 里的每个阈值都必须写明 value/unit/source/supporting_runs/reviewed_artifacts/reason。
如果当前文件只有裸数值或 rule_config 默认值，则只能视为 `threshold_source=temporary_default`，不能作为最终 complete 规则。

## 14. 地图/RGB 对齐验收

- map_rgb_alignment_status: `not_checked`
- 人工验收入口：`maps/*route_trigger_ego_trace.png` 与 `rgb/*sample_contact_sheet.jpg`。
- 若发现 route/trigger/ego trace 不贴合，先修 XML/run 匹配或 projection，不要直接调 RS 阈值。

## 15. 错帧回查路径

按 README -> map trace -> XML/XODR -> meta frame_features -> RGB contact/boundary -> thresholds/code 的顺序回查。
错帧归因必须落到 XML、XODR、meta、RGB、arbitration、threshold 中的一类或多类。

## 16. 完整性状态

- auto_input_complete: `True`
- manual_map_rgb_checked: `False`
- final_complete: `False` until maps/RGB boundary frames and threshold provenance are manually checked.
- 自动产物完成不等于人工最终完成；RGB 边界帧人工复核前，规则仍应保留 review 通道。
