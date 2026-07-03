# CrossJunctionDefectTrafficLight ROAD_STRUCTURE 调研设计

## 1. Scenario 语义与候选 RS

- rule_kind: `defect_junction`
- candidate_pool: `R1, R5`
- towns: `Town03, Town04, Town05, Town07, Town10HD, Town12, Town13, Town15`

## 2. 样本覆盖

- Town03: Town03_Rep0_route_002130_route0_01_11_13_52_37, Town03_Rep0_route_002134_route0_01_11_02_06_33, Town03_Rep0_route_002140_route0_01_09_14_56_53, Town03_Rep0_route_002150_route0_01_11_03_11_04, Town03_Rep0_route_002155_route0_01_08_16_23_06
- Town04: Town04_Rep0_route_002162_route0_01_10_04_20_32, Town04_Rep0_route_002167_route0_01_09_07_11_41, Town04_Rep0_route_002170_route0_01_09_00_23_21, Town04_Rep0_route_002174_route0_01_09_23_28_03, Town04_Rep0_route_002178_route0_01_10_04_32_29
- Town05: Town05_Rep0_route_002052_route0_01_08_08_46_16, Town05_Rep0_route_002061_route0_01_08_19_58_20, Town05_Rep0_route_002070_route0_01_09_22_49_09, Town05_Rep0_route_002079_route0_01_08_14_30_54, Town05_Rep0_route_002090_route0_01_08_17_03_40
- Town07: Town07_Rep0_route_002117_route0_01_10_00_51_30, Town07_Rep0_route_002119_route0_01_10_07_04_06, Town07_Rep0_route_002120_route0_01_08_05_32_26, Town07_Rep0_route_002121_route0_01_11_09_41_22, Town07_Rep0_route_002123_route0_01_10_11_26_45
- Town10HD: Town10HD_Rep0_route_002192_route0_01_09_23_54_55, Town10HD_Rep0_route_002195_route0_01_10_01_29_56, Town10HD_Rep0_route_002196_route0_01_09_17_44_26, Town10HD_Rep0_route_002197_route0_01_08_10_58_44, Town10HD_Rep0_route_002198_route0_01_09_11_20_50
- Town12: Town12_Rep0_route_002103_route0_01_11_01_15_31, Town12_Rep0_route_002106_route0_01_08_14_14_14, Town12_Rep0_route_002110_route0_01_09_03_13_05, Town12_Rep0_route_002112_route0_01_10_17_34_19, Town12_Rep0_route_002115_route0_01_11_15_45_59
- Town13: Town13_Rep0_route_002180_route0_01_08_13_54_39, Town13_Rep0_route_002182_route0_01_08_12_45_46, Town13_Rep0_route_002184_route0_01_09_09_14_06, Town13_Rep0_route_002188_route0_01_11_01_13_24, Town13_Rep0_route_002190_route0_01_10_11_30_23
- Town15: Town15_Rep0_route_002091_route0_01_10_10_09_37, Town15_Rep0_route_002097_route0_01_08_21_11_39, Town15_Rep0_route_002125_route0_01_09_20_36_54, Town15_Rep0_route_002158_route0_01_11_03_04_56, Town15_Rep0_route_002214_route0_01_08_14_26_52

## 3. XML 使用

XML 用于 route 粗投影、trigger 窗口、scenario tag 参数和数据源追溯；不能单独作为帧级 RS 真值。

## 4. XODR 使用

- XML trigger/traffic_direction/source_dist_interval 定义故障路口窗口
- XODR signal/controller 在本场景中是 defect_signal evidence，而非 R4 evidence

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

- trigger 对应 junction 前后窗口 -> R5
- 即使 XODR 有 signal/controller 或 meta 有灯态，也按故障灯语义覆盖 R4
- 找不到 junction 时 R5 medium + review

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
- 抽样 LEAD meta 中有 40 个 route 样本含 traffic_light_state 分布，可用于运行时 R4/R5 复核。
- 抽样 LEAD meta 可读 route 数=40；active_scenario 可读 route 数=40。

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
