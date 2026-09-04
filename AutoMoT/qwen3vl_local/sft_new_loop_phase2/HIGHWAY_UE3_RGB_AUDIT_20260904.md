# Highway UE3 逐帧 RGB 审计与接入记录（2026-09-04）

## 结论

高速/匝道上的他车 cut-in 属于现有 `UE3`，不新增类别。`HIGHWAY_CUTIN` 只是训练、
generation eval 和独立 eval 的审计子型；UE3 recall 当前只统计、不阻断 checkpoint 或流水线。

不能把 `HighwayCutIn` scenario 或 `R3` 直接当成 UE3。现有源标注中该 scenario 的帧
几乎全是 R3，EVENT 却只有 R-E1/R-E2/R-E4，没有 U-E3；其中既包含真实 cut-in，也包含
并排行驶、稳定跟车和 ego 超车视差。仅改 prompt 而不补 RGB 真值，会继续把高速 cut-in
监督成 all-NO。

## 逐帧证据

本次先打开既有 `full_route_rgb_label_review_20260809/HighwayCutIn` 的完整 all-frame
contact sheets，按连续帧而不是 scenario 名检查：

- `Town13_Rep0_1586_9_route0_01_08_19_01_08`：警车从相邻通道持续横移，车头/车身相对
  可见分道线发生连续变化并侵入 ego 当前通道；侵入前和驶离后的帧不属于 UE3。
- `Town13_Rep0_1586_11_route0_01_09_19_15_55`：紫色 SUV 从左侧相邻通道跨线进入 ego
  corridor；完成动作、成为稳定前车后不再续标 UE3。
- `Town13_Rep0_1283_10_route0_01_08_20_13_30`：黑色 SUV 从右侧车道逐帧跨过分道线进入
  ego lane；早期仍在右道、后期已经稳定居中分别作为 NO 边界。
- `Town12_Rep0_2954_1_route0_01_08_00_18_48`：补充查看原始 RGB 序列的保守核心段；黑车
  从右侧进入、跨线并占用 ego 通道。该路线只取高置信核心，边界不向外扩张。
- `Town12_Rep0_1394_0_route0_01_10_10_37_32`、`...2954_4...`、`...953_4...`：
  完整路线里车辆主要是停车、并行、稳定跟车或被 ego 超过，保留为高速 all-NO 对照。

人工 span 和可见证据写入 `highway_ue3_rgb_decisions_v1.jsonl`。其中 YES span 才能覆盖
正式训练标签；NO span 只用于边界/硬负例审计。构建器不会按 scenario、R3 或源 EVENT 名
自动扩张正例。

## 提示词边界

新问题仍只问 UE3，但明确覆盖 highway/ramp：

- YES：他车相对可见分道线持续横移，车头或车身跨线进入 ego 当前车道/近期 corridor；
- NO：普通并排行驶、稳定同道前车、ego 超车造成的图像位移、仅仅距离近或车型大；
- 起点：四帧 history 已支持持续横移，且最新帧已跨线或可见地即将侵入；
- 终点：车辆已经稳定居中且不再有即时 cut-in/避让影响，或已经离开 corridor。

prompt 名升级为 `sft_new_loop_phase2_direct_event_visual_v5_highway_ue3`。旧 v3/v4 adapter
与新 prompt hash 不兼容，必须重训，不能把旧 adapter 直接拿来做新合同的 LoRA 评测。

## 审计与统计

### 全部 UE3 场景兼容性

对 `collection_output/*_result.json` 全量检索后，源 taxonomy 中只有三个 scenario 显式含
`U-E3`；加上本次逐帧 RGB 回灌的 HighwayCutIn，共四类 UE3 来源：

| 来源 | 显式 UE3 帧 | route | Phase2 处理 |
|---|---:|---:|---|
| DynamicObjectCrossing | 251 | 84 | `OTHER_UE3` |
| ParkingCutIn | 330 | 80 | `OTHER_UE3` |
| StaticCutIn | 770 | 54 | `OTHER_UE3` |
| HighwayCutIn RGB YES spans | 73 | 4 | `HIGHWAY_CUTIN` |

原有 1351 个 `U-E3` 帧中，1318 个 primary RS=R1；另有 33 个是 R4 interrupted overlay
（ParkingCutIn 4、StaticCutIn 29）。旧 `_target_class` 会静默丢掉后者。现改为：任何显式
源 `U-E3` 都保留为 UE3，并固定通过 `ROAD_CORRIDOR` 问组监督；没有 `U-E3` 的 R4/R5
仍按路口域处理，不能凭 scenario 名或 RS 生成 UE3。已有 ParkingCutIn 213–215 全帧图可见
蓝车从右侧切入，而 R4 来自交通灯候选接管，验证了 interrupted overlay 不能被 RS gate
覆盖。dataset manifest 额外写入每个 split 的 UE3 raw/sampled scenario counts，重建后可直接
核对四种来源是否都进入采样。

- frame index：保留 `target_event_class=UE3`，增加 `ue3_subtype=HIGHWAY_CUTIN` 与
  `event_label_source=manual_highway_ue3_rgb_decision_v1`。
- dataset manifest：记录决策 span/frame/route、split 覆盖、匹配与缺失 guard，以及采样前后
  UE3 子型数量。
- generation eval / 独立 eval：同时报告 `slice/ue3_*`、`highway_ue3` 与 `other_ue3`；
  高速子型统计不会从总体 UE3 中扣除。
- RGB audit：统一由 `audit_eval_cases.py` 生成；它同时保留 `ue3_fn` 与
  `highway_ue3_fn`，逐例检查“相对分道线变化”和“并行/稳定跟车/ego 超车”边界。

本次只建立数据、prompt 与审计合同，没有用这些帧打开历史 unseen，也没有把高速子型加入
新的 production guard。先训练并观察总体 UE3、HIGHWAY_CUTIN、OTHER_UE3、UE6、INVALID 和
RE 的共同变化，再决定是否需要子型门槛。
