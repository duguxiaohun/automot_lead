# NonSignalizedJunctionLeftTurn ROAD_STRUCTURE 调研设计

## 1. Scenario 语义与候选 RS

- rule_kind: `nonsignalized_junction`
- candidate_pool: `R1, R5`
- towns: `Town03, Town04, Town05, Town07, Town10HD, Town12, Town13`

## 2. 样本覆盖

- Town03: Town03_Rep0_route_000924_route0_01_11_02_27_49, Town03_Rep0_route_000925_route0_01_09_16_26_51, Town03_Rep0_route_000927_route0_01_09_07_01_13
- Town04: Town04_Rep0_route_000930_route0_01_10_01_26_01, Town04_Rep0_route_000931_route0_01_10_04_18_32, Town04_Rep0_route_000932_route0_01_09_23_24_00, Town04_Rep0_route_000933_route0_01_10_12_55_09, Town04_Rep0_route_000935_route0_01_10_17_02_05
- Town05: Town05_Rep0_route_000845_route0_01_10_00_42_03, Town05_Rep0_route_000847_route0_01_11_13_21_56, Town05_Rep0_route_000851_route0_01_08_14_37_53, Town05_Rep0_route_000855_route0_01_09_09_27_10, Town05_Rep0_route_000857_route0_01_09_11_46_11
- Town07: Town07_Rep0_route_000907_route0_01_08_18_53_01, Town07_Rep0_route_000912_route0_01_09_17_07_45, Town07_Rep0_route_000915_route0_01_09_18_08_49, Town07_Rep0_route_000917_route0_01_09_16_36_17, Town07_Rep0_route_000921_route0_01_08_17_19_42
- Town12: Town12_Rep0_1105_0_route0_01_08_04_47_40, Town12_Rep0_1815_0_route0_01_10_10_50_57, Town12_Rep0_3415_0_route0_01_09_15_49_39, Town12_Rep0_route_000879_route0_01_10_00_59_52, Town12_Rep0_route_000905_route0_01_08_04_13_07
- Town13: Town13_Rep0_1140_0_route0_01_08_20_32_50, Town13_Rep0_1201_1_route0_01_09_04_56_18, Town13_Rep0_1429_0_route0_01_10_00_18_34, Town13_Rep0_1633_0_route0_01_08_19_23_46, Town13_Rep0_route_000954_route0_01_10_02_57_17

## 3. XML 使用

XML 用于 route 粗投影、trigger 窗口、scenario tag 参数和数据源追溯；不能单独作为帧级 RS 真值。

## 4. XODR 使用

- XML trigger/flow/source_dist_interval 定义接近与冲突流窗口
- XODR junction 且缺少 signal controller、存在 stop/yield hint 时增强 R5

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

- 无有效正常灯态 + trigger/junction/stop-yield 窗口 -> R5
- 出现连续有效灯态时降级 review，不直接 high R5
- 路口外 -> R1

## 9. 置信度规则

- high: scenario prior + XML window + XODR topology + meta signal/junction/active 至少三源一致。
- medium: 两源一致，或强 meta 信号成立但 XODR/RGB 支持不足。
- low: 只有 scenario prior 或弱 XODR hint；必须 review。

## 10. Review 规则

出现 XML 缺失、XODR 缺失、meta 缺失、route projection error、候选分数接近、RGB 与地图冲突时必须 review。

## 11. 已知失败模式

- Town10HD: incomplete because meta_sample_available

## 12. 当前样本逻辑备注

- 抽样 town 的 XODR 存在 controller；无灯/路权标签需要运行时用 meta 灯态做冲突审计。
- 抽样 LEAD meta 可读 route 数=28；active_scenario 可读 route 数=28。

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

- auto_input_complete: `False`
- manual_map_rgb_checked: `False`
- final_complete: `False` until maps/RGB boundary frames and threshold provenance are manually checked.
- 自动产物完成不等于人工最终完成；RGB 边界帧人工复核前，规则仍应保留 review 通道。
