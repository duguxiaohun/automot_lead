# VehicleTurningRoute ROAD_STRUCTURE 调研设计

## 1. Scenario 语义与候选 RS

- rule_kind: `vehicle_turning`
- candidate_pool: `R1, R4, R5`
- towns: `Town01, Town02, Town03, Town04, Town05, Town06, Town07, Town10HD, Town12, Town13`

## 2. 样本覆盖

- Town01: Town01_Rep0_Town01_Scenario4_0_route0_01_10_17_00_03, Town01_Rep0_Town01_Scenario4_1_route0_01_08_18_39_18, Town01_Rep0_Town01_Scenario4_2_route0_01_09_20_30_09, Town01_Rep0_Town01_Scenario4_41_route0_01_10_18_41_18, Town01_Rep0_Town01_Scenario4_8_route0_01_10_00_41_31
- Town02: Town02_Rep0_Town02_Scenario4_0_route0_01_10_17_53_34, Town02_Rep0_Town02_Scenario4_17_route0_01_10_20_00_58, Town02_Rep0_Town02_Scenario4_24_route0_01_09_21_17_41, Town02_Rep0_Town02_Scenario4_30_route0_01_10_01_26_16, Town02_Rep0_Town02_Scenario4_9_route0_01_10_23_01_51
- Town03: Town03_Rep0_Town03_Scenario4_0_route0_01_08_12_13_41, Town03_Rep0_Town03_Scenario4_129_route0_01_10_13_51_03, Town03_Rep0_Town03_Scenario4_2_route0_01_10_05_11_36, Town03_Rep0_Town03_Scenario4_5_route0_01_10_08_39_33, Town03_Rep0_Town03_Scenario4_9_route0_01_11_02_47_28
- Town04: Town04_Rep0_Town04_Scenario4_0_route0_01_10_16_18_06, Town04_Rep0_Town04_Scenario4_17_route0_01_09_03_54_47, Town04_Rep0_Town04_Scenario4_46_route0_01_09_08_29_13, Town04_Rep0_Town04_Scenario4_76_route0_01_11_12_42_21, Town04_Rep0_Town04_Scenario4_9_route0_01_09_13_00_06
- Town05: Town05_Rep0_Town05_Scenario4_0_route0_01_09_15_09_27, Town05_Rep0_Town05_Scenario4_129_route0_01_10_00_11_05, Town05_Rep0_Town05_Scenario4_41_route0_01_10_05_33_32, Town05_Rep0_Town05_Scenario4_70_route0_01_10_16_59_28, Town05_Rep0_Town05_Scenario4_9_route0_01_08_03_30_34
- Town06: Town06_Rep0_Town06_Scenario4_0_route0_01_09_15_30_47, Town06_Rep0_Town06_Scenario4_18_route0_01_09_08_36_36, Town06_Rep0_Town06_Scenario4_27_route0_01_10_16_42_30, Town06_Rep0_Town06_Scenario4_42_route0_01_10_14_20_07, Town06_Rep0_Town06_Scenario4_9_route0_01_11_07_56_55
- Town07: Town07_Rep0_Town07_Scenario4_0_route0_01_10_04_15_01, Town07_Rep0_Town07_Scenario4_25_route0_01_09_16_07_48, Town07_Rep0_Town07_Scenario4_47_route0_01_09_04_45_51, Town07_Rep0_Town07_Scenario4_73_route0_01_08_20_31_06, Town07_Rep0_Town07_Scenario4_9_route0_01_10_03_44_37
- Town10HD: Town10HD_Rep0_Town10HD_Scenario4_0_route0_01_08_09_10_47, Town10HD_Rep0_Town10HD_Scenario4_19_route0_01_10_17_47_44, Town10HD_Rep0_Town10HD_Scenario4_2_route0_01_09_13_22_18, Town10HD_Rep0_Town10HD_Scenario4_39_route0_01_11_01_20_17, Town10HD_Rep0_Town10HD_Scenario4_9_route0_01_09_04_53_44
- Town12: Town12_Rep0_1149_0_route0_01_08_04_02_58, Town12_Rep0_2368_0_route0_01_11_05_58_07, Town12_Rep0_319_1_route0_01_10_16_55_09, Town12_Rep0_3986_0_route0_01_10_04_50_41, Town12_Rep0_778_0_route0_01_09_16_43_27
- Town13: Town13_Rep0_1003_0_route0_01_10_19_07_21, Town13_Rep0_1160_0_route0_01_09_18_22_21, Town13_Rep0_1220_0_route0_01_10_17_52_14, Town13_Rep0_1433_0_route0_01_10_06_09_36, Town13_Rep0_80_0_route0_01_08_21_34_43

## 3. XML 使用

XML 用于 route 粗投影、trigger 窗口、scenario tag 参数和数据源追溯；不能单独作为帧级 RS 真值。

## 4. XODR 使用

- XML 多 scenario trigger 全部保留，不能只取第一个
- XODR junction/controller 与 route heading change 共同确认转弯路口

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

- 每个 trigger 建独立转弯窗口；受控路口 -> R4，无灯路权路口 -> R5
- 普通弯道或路口外 -> R1

## 9. 置信度规则

- high: scenario prior + XML window + XODR topology + meta signal/junction/active 至少三源一致。
- medium: 两源一致，或强 meta 信号成立但 XODR/RGB 支持不足。
- low: 只有 scenario prior 或弱 XODR hint；必须 review。

## 10. Review 规则

出现 XML 缺失、XODR 缺失、meta 缺失、route projection error、候选分数接近、RGB 与地图冲突时必须 review。

## 11. 已知失败模式

- 当前自动审计未发现 town 级输入缺口；仍需人工复核 RGB 边界帧后才能最终确认。

## 12. 当前样本逻辑备注

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
