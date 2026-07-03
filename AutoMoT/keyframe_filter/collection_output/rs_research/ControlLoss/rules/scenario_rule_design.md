# ControlLoss ROAD_STRUCTURE 调研设计

## 1. Scenario 语义与候选 RS

- rule_kind: `default_meta_map`
- candidate_pool: `R1, R4`
- towns: `Town01, Town02, Town03, Town04, Town05, Town06, Town07, Town10HD, Town12, Town13`

## 2. 样本覆盖

- Town01: Town01_Rep0_Town01_Scenario1_1_route0_01_09_20_45_28, Town01_Rep0_Town01_Scenario1_2_route0_01_09_21_20_38, Town01_Rep0_Town01_Scenario1_3_route0_01_08_12_08_15, Town01_Rep0_Town01_Scenario1_4_route0_01_09_07_57_40, Town01_Rep0_Town01_Scenario1_5_route0_01_09_04_43_40
- Town02: Town02_Rep0_Town02_Scenario1_0_route0_01_09_03_34_08, Town02_Rep0_Town02_Scenario1_1_route0_01_08_16_11_54, Town02_Rep0_Town02_Scenario1_2_route0_01_08_10_31_25, Town02_Rep0_Town02_Scenario1_4_route0_01_08_14_55_33, Town02_Rep0_Town02_Scenario1_5_route0_01_10_22_20_04
- Town03: Town03_Rep0_Town03_Scenario1_0_route0_01_10_10_45_04, Town03_Rep0_Town03_Scenario1_14_route0_01_08_06_27_13, Town03_Rep0_Town03_Scenario1_19_route0_01_08_08_32_25, Town03_Rep0_Town03_Scenario1_4_route0_01_08_19_06_07, Town03_Rep0_Town03_Scenario1_9_route0_01_10_11_41_53
- Town04: Town04_Rep0_Town04_Scenario1_0_route0_01_10_08_00_02, Town04_Rep0_Town04_Scenario1_23_route0_01_08_00_44_36, Town04_Rep0_Town04_Scenario1_37_route0_01_08_19_01_11, Town04_Rep0_Town04_Scenario1_50_route0_01_08_11_32_24, Town04_Rep0_Town04_Scenario1_9_route0_01_09_02_15_08
- Town05: Town05_Rep0_Town05_Scenario1_0_route0_01_08_07_44_04, Town05_Rep0_Town05_Scenario1_24_route0_01_08_14_43_34, Town05_Rep0_Town05_Scenario1_38_route0_01_10_00_02_02, Town05_Rep0_Town05_Scenario1_50_route0_01_08_07_37_48, Town05_Rep0_Town05_Scenario1_9_route0_01_08_07_39_18
- Town06: Town06_Rep0_Town06_Scenario1_0_route0_01_09_17_10_52, Town06_Rep0_Town06_Scenario1_17_route0_01_09_19_00_48, Town06_Rep0_Town06_Scenario1_23_route0_01_10_09_57_04, Town06_Rep0_Town06_Scenario1_2_route0_01_10_15_41_39, Town06_Rep0_Town06_Scenario1_9_route0_01_09_03_08_08
- Town07: Town07_Rep0_Town07_Scenario1_0_route0_01_10_22_58_17, Town07_Rep0_Town07_Scenario1_1_route0_01_09_10_58_47, Town07_Rep0_Town07_Scenario1_4_route0_01_08_11_50_20, Town07_Rep0_Town07_Scenario1_6_route0_01_09_10_04_41, Town07_Rep0_Town07_Scenario1_9_route0_01_09_23_41_12
- Town10HD: Town10HD_Rep0_Town10HD_Scenario1_0_route0_01_08_21_53_40, Town10HD_Rep0_Town10HD_Scenario1_13_route0_01_10_05_43_19, Town10HD_Rep0_Town10HD_Scenario1_2_route0_01_10_16_34_51, Town10HD_Rep0_Town10HD_Scenario1_5_route0_01_08_05_29_06, Town10HD_Rep0_Town10HD_Scenario1_9_route0_01_09_17_33_53
- Town12: Town12_Rep0_1102_0_route0_01_10_20_51_57, Town12_Rep0_2023_0_route0_01_10_23_15_31, Town12_Rep0_3056_0_route0_01_08_11_24_50, Town12_Rep0_4186_0_route0_01_09_23_10_36, Town12_Rep0_972_0_route0_01_09_17_20_53
- Town13: Town13_Rep0_1048_0_route0_01_08_08_14_09, Town13_Rep0_1341_0_route0_01_08_23_18_06, Town13_Rep0_1449_0_route0_01_10_14_35_55, Town13_Rep0_1561_0_route0_01_11_01_33_22, Town13_Rep0_97_0_route0_01_09_16_58_01

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
