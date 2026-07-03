# HazardAtSideLaneTwoWays ROAD_STRUCTURE 调研设计

## 1. Scenario 语义与候选 RS

- rule_kind: `twoways_obstacle`
- candidate_pool: `R1, R2, R4`
- towns: `Town12, Town13`

## 2. 样本覆盖

- Town12: Town12_Rep0_1067_0_route0_01_10_00_13_56, Town12_Rep0_1871_0_route0_01_09_17_08_51, Town12_Rep0_2986_0_route0_01_11_00_37_04, Town12_Rep0_4259_0_route0_01_08_20_22_18, Town12_Rep0_886_0_route0_01_09_11_32_45
- Town13: Town13_Rep0_1227_0_route0_01_10_21_55_12, Town13_Rep0_1302_1_route0_01_08_15_46_08, Town13_Rep0_1370_0_route0_01_09_12_10_06, Town13_Rep0_1466_1_route0_01_09_17_21_47, Town13_Rep0_1659_1_route0_01_08_08_35_22

## 3. XML 使用

XML 用于 route 粗投影、trigger 窗口、scenario tag 参数和数据源追溯；不能单独作为帧级 RS 真值。

## 4. XODR 使用

- XML distance/front/behind/frequency 定义绕障核心窗口
- XODR lane_id 符号反转与同向车道数决定是否确实需要对向参与

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

- 有效灯态/受控路口窗口优先 -> R4，R2 仅作为 secondary
- TwoWays trigger/distance/active 窗口 + XODR 对向 driving lane + 同向车道不足 -> R2
- XODR 不可用时 scenario+XML 窗口只给 medium R2；窗口外 -> R1

## 9. 置信度规则

- high: scenario prior + XML window + XODR topology + meta signal/junction/active 至少三源一致。
- medium: 两源一致，或强 meta 信号成立但 XODR/RGB 支持不足。
- low: 只有 scenario prior 或弱 XODR hint；必须 review。

## 10. Review 规则

出现 XML 缺失、XODR 缺失、meta 缺失、route projection error、候选分数接近、RGB 与地图冲突时必须 review。

## 11. 已知失败模式

- 当前自动审计未发现 town 级输入缺口；仍需人工复核 RGB 边界帧后才能最终确认。

## 12. 当前样本逻辑备注

- 抽样 XML 含 distance/offset，TwoWays/InvadingTurn 核心窗口可被 route trigger 约束。
- 抽样 LEAD meta 可读 route 数=10；active_scenario 可读 route 数=10。

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
