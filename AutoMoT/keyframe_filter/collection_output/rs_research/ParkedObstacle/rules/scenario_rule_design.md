# ParkedObstacle ROAD_STRUCTURE 调研设计

## 1. Scenario 语义与候选 RS

- rule_kind: `same_direction_obstacle`
- candidate_pool: `R1, R4`
- towns: `Town03, Town04, Town05, Town06, Town10HD, Town12, Town13`

## 2. 样本覆盖

- Town03: Town03_Rep0_route_001880_route0_01_10_07_48_34, Town03_Rep0_route_001882_route0_01_08_10_59_14, Town03_Rep0_route_001886_route0_01_09_15_53_44, Town03_Rep0_route_001889_route0_01_10_17_16_11, Town03_Rep0_route_001891_route0_01_10_13_37_37
- Town04: Town04_Rep0_route_001895_route0_01_09_03_47_39, Town04_Rep0_route_001901_route0_01_10_05_49_05, Town04_Rep0_route_001904_route0_01_09_10_03_02, Town04_Rep0_route_001908_route0_01_09_01_58_40, Town04_Rep0_route_001912_route0_01_10_20_17_36
- Town05: Town05_Rep0_route_001858_route0_01_08_21_38_44, Town05_Rep0_route_001867_route0_01_09_12_36_33, Town05_Rep0_route_001872_route0_01_08_19_03_34, Town05_Rep0_route_001875_route0_01_09_10_24_12, Town05_Rep0_route_001879_route0_01_11_03_37_53
- Town06: Town06_Rep0_route_001920_route0_01_09_10_49_27, Town06_Rep0_route_001927_route0_01_11_10_46_03, Town06_Rep0_route_001937_route0_01_09_07_06_43, Town06_Rep0_route_001946_route0_01_10_01_09_54, Town06_Rep0_route_001953_route0_01_08_10_14_57
- Town10HD: Town10HD_Rep0_route_001913_route0_01_09_16_27_23, Town10HD_Rep0_route_001915_route0_01_08_16_24_41, Town10HD_Rep0_route_001916_route0_01_08_19_43_10, Town10HD_Rep0_route_001917_route0_01_09_03_28_09, Town10HD_Rep0_route_001919_route0_01_10_09_42_10
- Town12: Town12_Rep0_1006_0_route0_01_10_14_44_57, Town12_Rep0_2474_1_route0_01_09_10_42_18, Town12_Rep0_2967_1_route0_01_10_20_56_17, Town12_Rep0_676_0_route0_01_08_10_05_59, Town12_Rep0_962_1_route0_01_09_14_36_56
- Town13: Town13_Rep0_1244_0_route0_01_10_12_48_15, Town13_Rep0_1263_0_route0_01_09_07_36_11, Town13_Rep0_1395_0_route0_01_09_15_55_13, Town13_Rep0_1395_9_route0_01_11_02_26_39, Town13_Rep0_17_9_route0_01_10_15_52_07

## 3. XML 使用

XML 用于 route 粗投影、trigger 窗口、scenario tag 参数和数据源追溯；不能单独作为帧级 RS 真值。

## 4. XODR 使用

- XML trigger/distance/speed 只记录障碍上下文与路口近邻窗口
- XODR 只用于确认是否进入受控 junction，不用 parking/opposite hint 升级

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

- 有效 traffic_light_state/light_hazard 或同源受控路口窗口 -> R4
- 同向事故/施工/停放障碍只作为事件证据，不改变道路结构 -> R1

## 9. 置信度规则

- high: scenario prior + XML window + XODR topology + meta signal/junction/active 至少三源一致。
- medium: 两源一致，或强 meta 信号成立但 XODR/RGB 支持不足。
- low: 只有 scenario prior 或弱 XODR hint；必须 review。

## 10. Review 规则

出现 XML 缺失、XODR 缺失、meta 缺失、route projection error、候选分数接近、RGB 与地图冲突时必须 review。

## 11. 已知失败模式

- 当前自动审计未发现 town 级输入缺口；仍需人工复核 RGB 边界帧后才能最终确认。

## 12. 当前样本逻辑备注

- 抽样 LEAD meta 可读 route 数=35；active_scenario 可读 route 数=35。

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
