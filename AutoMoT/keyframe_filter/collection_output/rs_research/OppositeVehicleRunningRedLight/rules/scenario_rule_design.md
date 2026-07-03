# OppositeVehicleRunningRedLight ROAD_STRUCTURE 调研设计

## 1. Scenario 语义与候选 RS

- rule_kind: `signalized_junction`
- candidate_pool: `R1, R4`
- towns: `Town01, Town02, Town03, Town04, Town05, Town06, Town07, Town10HD, Town12, Town13`

## 2. 样本覆盖

- Town01: Town01_Rep0_Town01_Scenario8_10_route0_01_09_00_02_37, Town01_Rep0_Town01_Scenario8_24_route0_01_10_05_56_50, Town01_Rep0_Town01_Scenario8_39_route0_01_11_13_30_45, Town01_Rep0_Town01_Scenario8_51_route0_01_11_01_24_18, Town01_Rep0_Town01_Scenario8_71_route0_01_10_03_42_51
- Town02: Town02_Rep0_Town02_Scenario8_15_route0_01_08_12_51_42, Town02_Rep0_Town02_Scenario8_21_route0_01_08_21_02_36, Town02_Rep0_Town02_Scenario8_34_route0_01_09_01_12_38, Town02_Rep0_Town02_Scenario8_43_route0_01_09_13_21_41, Town02_Rep0_Town02_Scenario8_8_route0_01_09_01_17_09
- Town03: Town03_Rep0_Town03_Scenario8_101_route0_01_09_13_15_56, Town03_Rep0_Town03_Scenario8_22_route0_01_10_18_38_41, Town03_Rep0_Town03_Scenario8_49_route0_01_09_12_44_14, Town03_Rep0_Town03_Scenario8_73_route0_01_09_02_16_57, Town03_Rep0_Town03_Scenario8_98_route0_01_10_15_31_06
- Town04: Town04_Rep0_Town04_Scenario8_102_route0_01_09_17_16_49, Town04_Rep0_Town04_Scenario8_1_route0_01_10_00_49_02, Town04_Rep0_Town04_Scenario8_45_route0_01_09_09_23_12, Town04_Rep0_Town04_Scenario8_71_route0_01_08_15_49_04, Town04_Rep0_Town04_Scenario8_99_route0_01_11_02_35_08
- Town05: Town05_Rep0_Town05_Scenario8_0_route0_01_10_17_59_38, Town05_Rep0_Town05_Scenario8_144_route0_01_09_08_58_16, Town05_Rep0_Town05_Scenario8_184_route0_01_10_08_00_27, Town05_Rep0_Town05_Scenario8_49_route0_01_10_02_13_02, Town05_Rep0_Town05_Scenario8_99_route0_01_11_02_35_30
- Town06: Town06_Rep0_Town06_Scenario8_2_route0_01_10_10_15_06, Town06_Rep0_Town06_Scenario8_50_route0_01_08_16_50_55, Town06_Rep0_Town06_Scenario8_70_route0_01_08_19_37_10, Town06_Rep0_Town06_Scenario8_83_route0_01_08_15_28_35, Town06_Rep0_Town06_Scenario8_9_route0_01_10_13_50_48
- Town07: Town07_Rep0_Town07_Scenario8_0_route0_01_11_01_14_03, Town07_Rep0_Town07_Scenario8_19_route0_01_10_20_11_38, Town07_Rep0_Town07_Scenario8_26_route0_01_09_21_32_43, Town07_Rep0_Town07_Scenario8_37_route0_01_09_18_29_15, Town07_Rep0_Town07_Scenario8_8_route0_01_10_21_38_23
- Town10HD: Town10HD_Rep0_Town10HD_Scenario8_15_route0_01_08_19_36_13, Town10HD_Rep0_Town10HD_Scenario8_25_route0_01_10_23_13_22, Town10HD_Rep0_Town10HD_Scenario8_37_route0_01_09_07_11_44, Town10HD_Rep0_Town10HD_Scenario8_47_route0_01_08_10_46_58, Town10HD_Rep0_Town10HD_Scenario8_6_route0_01_09_02_29_34
- Town12: Town12_Rep0_1130_0_route0_01_10_00_15_07, Town12_Rep0_2171_0_route0_01_09_00_51_48, Town12_Rep0_3276_0_route0_01_10_05_30_13, Town12_Rep0_4398_0_route0_01_08_19_03_19, Town12_Rep0_877_0_route0_01_09_08_07_25
- Town13: Town13_Rep0_1047_0_route0_01_10_16_28_42, Town13_Rep0_1087_4_route0_01_09_05_43_45, Town13_Rep0_1245_2_route0_01_09_19_56_36, Town13_Rep0_42_0_route0_01_10_10_31_32, Town13_Rep0_7_5_route0_01_09_06_02_55

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

- 抽样 XML trigger/route 附近有 25 个样本靠近 XODR signal，R4/故障灯窗口有空间证据。
- 抽样 LEAD meta 中有 50 个 route 样本含 traffic_light_state 分布，可用于运行时 R4/R5 复核。
- 抽样 LEAD meta 可读 route 数=50；active_scenario 可读 route 数=50。

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
