# SignalizedJunctionLeftTurnEnterFlow ROAD_STRUCTURE 调研设计

## 1. Scenario 语义与候选 RS

- rule_kind: `signalized_junction`
- candidate_pool: `R1, R4`
- towns: `Town01, Town02, Town03, Town04, Town05, Town07, Town10HD, Town12, Town13, Town15`

## 2. 样本覆盖

- Town01: Town01_Rep0_route_002329_route0_01_08_23_56_14, Town01_Rep0_route_002332_route0_01_09_14_54_11, Town01_Rep0_route_002334_route0_01_08_09_20_23, Town01_Rep0_route_002337_route0_01_10_14_30_52, Town01_Rep0_route_002338_route0_01_09_11_46_42
- Town02: Town02_Rep0_route_002349_route0_01_09_11_00_36, Town02_Rep0_route_002351_route0_01_08_21_46_53, Town02_Rep0_route_002352_route0_01_10_15_19_04, Town02_Rep0_route_002354_route0_01_10_23_42_33, Town02_Rep0_route_002356_route0_01_11_04_16_07
- Town03: Town03_Rep0_route_002267_route0_01_09_05_40_40, Town03_Rep0_route_002273_route0_01_07_21_27_45, Town03_Rep0_route_002277_route0_01_11_15_22_58, Town03_Rep0_route_002281_route0_01_10_16_16_40, Town03_Rep0_route_002288_route0_01_08_20_11_57
- Town04: Town04_Rep0_route_002290_route0_01_10_02_42_46, Town04_Rep0_route_002298_route0_01_08_07_24_22, Town04_Rep0_route_002308_route0_01_08_22_05_36, Town04_Rep0_route_002315_route0_01_08_02_15_16, Town04_Rep0_route_002321_route0_01_08_11_58_53
- Town05: Town05_Rep0_route_002218_route0_01_10_20_47_30, Town05_Rep0_route_002231_route0_01_08_13_28_24, Town05_Rep0_route_002242_route0_01_08_11_22_44, Town05_Rep0_route_002247_route0_01_08_16_33_38, Town05_Rep0_route_002257_route0_01_10_20_24_38
- Town07: Town07_Rep0_route_002259_route0_01_09_01_13_59, Town07_Rep0_route_002260_route0_01_08_00_48_26, Town07_Rep0_route_002263_route0_01_09_13_18_11, Town07_Rep0_route_002265_route0_01_10_01_47_57, Town07_Rep0_route_002266_route0_01_09_11_49_36
- Town10HD: Town10HD_Rep0_route_002322_route0_01_08_16_54_40, Town10HD_Rep0_route_002323_route0_01_10_11_51_19, Town10HD_Rep0_route_002324_route0_01_09_12_53_58, Town10HD_Rep0_route_002327_route0_01_10_19_19_44, Town10HD_Rep0_route_002328_route0_01_10_23_10_27
- Town12: Town12_Rep0_route_002913_route0_01_08_13_14_14, Town12_Rep0_route_002924_route0_01_09_07_27_50, Town12_Rep0_route_002936_route0_01_09_23_02_00, Town12_Rep0_route_002949_route0_01_11_05_26_00, Town12_Rep0_route_002962_route0_01_10_01_07_37
- Town13: Town13_Rep0_route_003263_route0_01_08_02_22_51, Town13_Rep0_route_003275_route0_01_11_01_40_02, Town13_Rep0_route_003288_route0_01_09_10_39_01, Town13_Rep0_route_003300_route0_01_08_08_07_01, Town13_Rep0_route_003312_route0_01_09_14_37_17
- Town15: Town15_Rep0_route_002342_route0_01_10_05_46_36, Town15_Rep0_route_002343_route0_01_11_01_42_58, Town15_Rep0_route_002344_route0_01_10_10_14_06, Town15_Rep0_route_002346_route0_01_09_09_52_14

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
- 抽样 LEAD meta 中有 49 个 route 样本含 traffic_light_state 分布，可用于运行时 R4/R5 复核。
- 抽样 LEAD meta 可读 route 数=49；active_scenario 可读 route 数=49。

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
