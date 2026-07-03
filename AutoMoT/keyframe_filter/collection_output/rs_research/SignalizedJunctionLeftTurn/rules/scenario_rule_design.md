# SignalizedJunctionLeftTurn ROAD_STRUCTURE 调研设计

## 1. Scenario 语义与候选 RS

- rule_kind: `signalized_junction`
- candidate_pool: `R1, R4`
- towns: `Town01, Town02, Town03, Town04, Town05, Town06, Town07, Town10HD, Town12, Town13, Town15`

## 2. 样本覆盖

- Town01: Town01_Rep0_route_002516_route0_01_08_10_14_49, Town01_Rep0_route_002519_route0_01_11_06_26_54, Town01_Rep0_route_002521_route0_01_10_05_39_31, Town01_Rep0_route_002523_route0_01_08_21_32_34, Town01_Rep0_route_002525_route0_01_10_00_03_01
- Town02: Town02_Rep0_route_002544_route0_01_09_16_53_09, Town02_Rep0_route_002546_route0_01_10_09_46_26, Town02_Rep0_route_002547_route0_01_09_22_58_25, Town02_Rep0_route_002549_route0_01_10_23_32_28, Town02_Rep0_route_002551_route0_01_09_23_04_29
- Town03: Town03_Rep0_route_002435_route0_01_10_17_57_38, Town03_Rep0_route_002440_route0_01_11_04_23_44, Town03_Rep0_route_002445_route0_01_08_23_41_10, Town03_Rep0_route_002452_route0_01_10_15_31_08, Town03_Rep0_route_002456_route0_01_08_15_59_04
- Town04: Town04_Rep0_route_002467_route0_01_10_12_25_34, Town04_Rep0_route_002475_route0_01_09_04_47_10, Town04_Rep0_route_002481_route0_01_11_02_38_08, Town04_Rep0_route_002490_route0_01_10_01_36_33, Town04_Rep0_route_002498_route0_01_10_11_47_06
- Town05: Town05_Rep0_route_002357_route0_01_10_01_20_29, Town05_Rep0_route_002366_route0_01_11_02_32_05, Town05_Rep0_route_002374_route0_01_09_15_34_19, Town05_Rep0_route_002386_route0_01_08_02_47_01, Town05_Rep0_route_002399_route0_01_08_22_17_08
- Town06: Town06_Rep0_Town06_Scenario7_14_route0_01_09_06_56_16, Town06_Rep0_Town06_Scenario7_32_route0_01_09_06_25_09, Town06_Rep0_Town06_Scenario7_47_route0_01_08_23_54_08, Town06_Rep0_Town06_Scenario7_69_route0_01_10_16_51_26, Town06_Rep0_Town06_Scenario7_89_route0_01_08_21_00_08
- Town07: Town07_Rep0_Town07_Scenario7_12_route0_01_11_03_44_03, Town07_Rep0_Town07_Scenario7_28_route0_01_08_18_48_02, Town07_Rep0_Town07_Scenario7_6_route0_01_08_17_08_51, Town07_Rep0_route_002421_route0_01_08_06_35_28, Town07_Rep0_route_002425_route0_01_10_09_14_33
- Town10HD: Town10HD_Rep0_route_002508_route0_01_10_17_04_43, Town10HD_Rep0_route_002509_route0_01_09_20_29_26, Town10HD_Rep0_route_002510_route0_01_08_05_19_35, Town10HD_Rep0_route_002513_route0_01_09_22_53_37, Town10HD_Rep0_route_002514_route0_01_09_19_55_15
- Town12: Town12_Rep0_1002_0_route0_01_10_05_18_33, Town12_Rep0_2066_0_route0_01_08_12_08_23, Town12_Rep0_2681_0_route0_01_10_15_18_16, Town12_Rep0_3670_0_route0_01_09_05_21_46, Town12_Rep0_999_0_route0_01_08_00_31_43
- Town13: Town13_Rep0_1004_0_route0_01_10_14_59_12, Town13_Rep0_1051_0_route0_01_10_07_03_01, Town13_Rep0_1166_0_route0_01_08_17_14_48, Town13_Rep0_1592_1_route0_01_10_05_19_15, Town13_Rep0_9_2_route0_01_09_13_50_57
- Town15: Town15_Rep0_route_002402_route0_01_09_03_57_13, Town15_Rep0_route_002417_route0_01_08_11_30_46, Town15_Rep0_route_002462_route0_01_10_12_10_34, Town15_Rep0_route_002507_route0_01_09_08_59_21, Town15_Rep0_route_002543_route0_01_08_02_21_03

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

- 抽样 XML trigger/route 附近有 22 个样本靠近 XODR signal，R4/故障灯窗口有空间证据。
- 抽样 LEAD meta 中有 55 个 route 样本含 traffic_light_state 分布，可用于运行时 R4/R5 复核。
- 抽样 LEAD meta 可读 route 数=55；active_scenario 可读 route 数=55。

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
