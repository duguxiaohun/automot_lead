# DynamicObjectCrossing ROAD_STRUCTURE 调研设计

## 1. Scenario 语义与候选 RS

- rule_kind: `default_meta_map`
- candidate_pool: `R1, R4`
- towns: `Town01, Town02, Town03, Town04, Town05, Town06, Town07, Town10HD, Town12, Town13`

## 2. 样本覆盖

- Town01: Town01_Rep0_Town01_Scenario3_0_route0_01_09_17_24_20, Town01_Rep0_Town01_Scenario3_1_route0_01_08_15_18_34, Town01_Rep0_Town01_Scenario3_2_route0_01_08_08_41_45, Town01_Rep0_Town01_Scenario3_4_route0_01_10_04_10_36, Town01_Rep0_Town01_Scenario3_5_route0_01_09_13_22_11
- Town02: Town02_Rep0_Town02_Scenario3_0_route0_01_10_05_05_30, Town02_Rep0_Town02_Scenario3_1_route0_01_09_03_56_06, Town02_Rep0_Town02_Scenario3_2_route0_01_10_11_32_53, Town02_Rep0_Town02_Scenario3_4_route0_01_08_10_00_27, Town02_Rep0_Town02_Scenario3_5_route0_01_10_06_42_00
- Town03: Town03_Rep0_Town03_Scenario3_0_route0_01_08_19_50_37, Town03_Rep0_Town03_Scenario3_15_route0_01_09_04_40_12, Town03_Rep0_Town03_Scenario3_1_route0_01_08_12_54_26, Town03_Rep0_Town03_Scenario3_3_route0_01_10_11_45_14, Town03_Rep0_Town03_Scenario3_9_route0_01_08_22_04_39
- Town04: Town04_Rep0_Town04_Scenario3_0_route0_01_10_23_06_01, Town04_Rep0_Town04_Scenario3_22_route0_01_08_13_26_58, Town04_Rep0_Town04_Scenario3_37_route0_01_10_04_55_17, Town04_Rep0_Town04_Scenario3_50_route0_01_08_06_35_21, Town04_Rep0_Town04_Scenario3_9_route0_01_09_13_38_36
- Town05: Town05_Rep0_Town05_Scenario3_0_route0_01_08_11_56_12, Town05_Rep0_Town05_Scenario3_21_route0_01_10_18_39_35, Town05_Rep0_Town05_Scenario3_40_route0_01_08_11_22_35, Town05_Rep0_Town05_Scenario3_51_route0_01_10_09_16_43, Town05_Rep0_Town05_Scenario3_9_route0_01_09_13_45_05
- Town06: Town06_Rep0_Town06_Scenario3_0_route0_01_09_02_00_12, Town06_Rep0_Town06_Scenario3_17_route0_01_07_21_55_18, Town06_Rep0_Town06_Scenario3_23_route0_01_10_22_34_00, Town06_Rep0_Town06_Scenario3_30_route0_01_10_05_08_33, Town06_Rep0_Town06_Scenario3_9_route0_01_10_18_40_07
- Town07: Town07_Rep0_Town07_Scenario3_0_route0_01_11_03_15_40, Town07_Rep0_Town07_Scenario3_11_route0_01_09_11_48_12, Town07_Rep0_Town07_Scenario3_3_route0_01_10_23_12_07, Town07_Rep0_Town07_Scenario3_6_route0_01_11_05_19_17, Town07_Rep0_Town07_Scenario3_9_route0_01_10_20_11_35
- Town10HD: Town10HD_Rep0_Town10HD_Scenario3_0_route0_01_09_05_52_48, Town10HD_Rep0_Town10HD_Scenario3_13_route0_01_09_00_05_43, Town10HD_Rep0_Town10HD_Scenario3_1_route0_01_10_14_08_10, Town10HD_Rep0_Town10HD_Scenario3_4_route0_01_10_03_26_36, Town10HD_Rep0_Town10HD_Scenario3_9_route0_01_08_19_03_37
- Town12: Town12_Rep0_1528_0_route0_01_09_18_36_47, Town12_Rep0_2562_0_route0_01_09_04_07_20, Town12_Rep0_3201_0_route0_01_09_05_18_50, Town12_Rep0_4452_0_route0_01_08_13_21_59, Town12_Rep0_980_0_route0_01_11_01_24_26
- Town13: Town13_Rep0_1126_0_route0_01_09_02_41_33, Town13_Rep0_1317_0_route0_01_10_06_07_20, Town13_Rep0_1510_0_route0_01_08_09_35_55, Town13_Rep0_1603_0_route0_01_09_20_38_32, Town13_Rep0_40_0_route0_01_10_11_54_13

## 3. XML 使用

XML 用于 route 粗投影、trigger 窗口、scenario tag 参数和数据源追溯；不能单独作为帧级 RS 真值。

## 4. XODR 使用

- XML trigger 只给事件上下文与 junction 近邻窗口
- XODR 只用于确认是否存在受控路口

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

- 有效灯态/受控路口窗口 -> R4
- 动态横穿、急刹、失控、侧向危险等行为不改变 RS -> R1

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
