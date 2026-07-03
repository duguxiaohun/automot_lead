# SignalizedJunctionRightTurn ROAD_STRUCTURE 调研设计

## 1. Scenario 语义与候选 RS

- rule_kind: `signalized_junction`
- candidate_pool: `R1, R4`
- towns: `Town01, Town02, Town03, Town04, Town05, Town07, Town10HD, Town12, Town13, Town15`

## 2. 样本覆盖

- Town01: Town01_Rep0_Town01_Scenario9_14_route0_01_08_21_46_06, Town01_Rep0_Town01_Scenario9_3_route0_01_10_17_45_54, Town01_Rep0_Town01_Scenario9_60_route0_01_10_22_11_05, Town01_Rep0_route_002632_route0_01_10_08_39_07, Town01_Rep0_route_002640_route0_01_10_17_56_04
- Town02: Town02_Rep0_Town02_Scenario9_0_route0_01_09_07_10_56, Town02_Rep0_Town02_Scenario9_28_route0_01_09_01_22_35, Town02_Rep0_Town02_Scenario9_42_route0_01_11_03_55_03, Town02_Rep0_route_002652_route0_01_08_04_14_58, Town02_Rep0_route_002657_route0_01_10_10_11_26
- Town03: Town03_Rep0_Town03_Scenario9_0_route0_01_10_14_37_36, Town03_Rep0_Town03_Scenario9_21_route0_01_09_20_13_51, Town03_Rep0_Town03_Scenario9_53_route0_01_10_19_41_08, Town03_Rep0_Town03_Scenario9_93_route0_01_09_09_55_14, Town03_Rep0_route_002600_route0_01_09_09_47_17
- Town04: Town04_Rep0_Town04_Scenario9_101_route0_01_09_09_21_42, Town04_Rep0_Town04_Scenario9_30_route0_01_10_03_01_33, Town04_Rep0_Town04_Scenario9_54_route0_01_08_20_46_04, Town04_Rep0_Town04_Scenario9_77_route0_01_09_05_35_00, Town04_Rep0_route_002614_route0_01_09_22_57_58
- Town05: Town05_Rep0_Town05_Scenario9_101_route0_01_08_17_16_07, Town05_Rep0_Town05_Scenario9_14_route0_01_10_07_34_32, Town05_Rep0_Town05_Scenario9_40_route0_01_08_12_32_01, Town05_Rep0_Town05_Scenario9_74_route0_01_10_03_24_16, Town05_Rep0_route_002560_route0_01_10_11_14_37
- Town07: Town07_Rep0_route_002583_route0_01_09_12_58_34
- Town10HD: Town10HD_Rep0_Town10HD_Scenario9_0_route0_01_08_02_57_12, Town10HD_Rep0_Town10HD_Scenario9_32_route0_01_10_20_02_08, Town10HD_Rep0_Town10HD_Scenario9_39_route0_01_08_10_24_19, Town10HD_Rep0_Town10HD_Scenario9_56_route0_01_10_06_56_22, Town10HD_Rep0_route_002628_route0_01_09_00_07_01
- Town12: Town12_Rep0_1069_0_route0_01_10_20_55_15, Town12_Rep0_2267_0_route0_01_08_21_31_45, Town12_Rep0_3657_0_route0_01_08_22_16_13, Town12_Rep0_521_1_route0_01_10_15_33_48, Town12_Rep0_86_0_route0_01_11_04_05_37
- Town13: Town13_Rep0_1014_0_route0_01_09_05_53_40, Town13_Rep0_1059_1_route0_01_09_01_55_11, Town13_Rep0_1092_2_route0_01_07_23_28_33, Town13_Rep0_1539_1_route0_01_10_20_33_33, Town13_Rep0_2_1_route0_01_10_16_47_41
- Town15: Town15_Rep0_route_002561_route0_01_11_01_39_24, Town15_Rep0_route_002573_route0_01_08_04_47_43, Town15_Rep0_route_002592_route0_01_08_08_41_59, Town15_Rep0_route_002618_route0_01_09_04_13_12, Town15_Rep0_route_002650_route0_01_09_18_51_57

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

- 抽样 XML trigger/route 附近有 19 个样本靠近 XODR signal，R4/故障灯窗口有空间证据。
- 抽样 LEAD meta 中有 46 个 route 样本含 traffic_light_state 分布，可用于运行时 R4/R5 复核。
- 抽样 LEAD meta 可读 route 数=46；active_scenario 可读 route 数=46。

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
