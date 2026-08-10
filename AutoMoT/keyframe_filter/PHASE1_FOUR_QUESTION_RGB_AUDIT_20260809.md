# Phase1 四问 RGB 全量复核摘要（2026-08-09）

本文记录 `phase1_four_question_answer_table.json` 的人工 RGB + 标签复核口径。证据产物仍保留在
`AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/`。其中最终四问标签表、
batch matrix 与人工 JSONL notes 属于轻量标签结果，已列入 push 白名单；RGB contact sheet、
montage、candidate anomalies、route/town/scenario/global summary 等大体量或可再生证据仍默认不入库、不 push。
`collection_output` 下各 Phase1 目录的主入口、legacy/superseded 关系和复用流程见
[`PHASE1_COLLECTION_OUTPUT_INDEX.md`](PHASE1_COLLECTION_OUTPUT_INDEX.md)；后续复核应优先复用已有
`full_route_rgb_label_review_20260809/` 证据，不要重复生成新的全量 RGB 文件夹。

## 1. 审计对象与证据

- 答案表：
  `AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/phase1_four_question_answer_table.json`
- 逐帧 RGB + 原始标签 sheet：
  `AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809/<Scenario>/<Town>/<run_id>/sheets/all_frames_*.jpg`
- 人工复核笔记：
  `AutoMoT/keyframe_filter/collection_output/phase1_four_question_audit/manual_visual_audit_notes.jsonl`
- `noScenarios` 已排除，不参与本轮四问表复核。
- 复核流程：每个非 `noScenarios` 场景按 Town 查 RGB+标签；每个 Town 尽量至少看 3 个 ID，若 evidence set 不足 3 个则看完全部可用 ID 并在 notes 中注明。对缩略总览不清的组合继续打开原始 sheet 放大确认。
- 本轮统一证据生成记录：42 个非 `noScenarios` 场景、197 个 scenario-Town、582 条 route、68,073 帧、2,003 张逐帧 RGB+RS/EVENT 标签 sheet；`skipped=0`，`noScenarios` 未进入输出。每个 Town 默认抽 3 条 route；5 个 Town 的可读源 route 少于 3 条，均已审完全部可用 route：`AccidentTwoWays/Town05`、`ConstructionObstacleTwoWays/Town05`、`MergerIntoSlowTrafficV2/Town06`、`NonSignalizedJunctionLeftTurnEnterFlow/Town10HD`、`SignalizedJunctionRightTurn/Town07`。早期 `manual_visual_audit_notes.jsonl` 的 277 条逐 route 笔记只作为补充证据，不再替代本轮统一覆盖统计。

## 2. 四问统一判定口径

四个答案分别是 `HIGHWAY`、`OBSTACLE`、`VULNERABLE`、`TRAFFIC_LIGHT_ABNORMAL`。它们是当前 `scenario × RS × EVENT` 组合的可见语义答案，不是简单复读 RS/EVENT 代码。

- `HIGHWAY=YES`：必须看到高速/快速路/匝道/出入口/导流 gore/连续隔离或受控通行等道路拓扑证据。普通直道、宽路、空旷路、护栏或车速感单独出现都不能算高速。
- `OBSTACLE=YES`：必须是占用/压缩 ego lane 或 ego path 的物理障碍，例如事故车、施工物、停放/静止车辆、开门突出物、静态切入车辆等。正常运动的其他交通流不算障碍，除非它实际侵入并阻断当前路径。
- `VULNERABLE=YES`：行人、骑行者、摩托/自行车等弱势参与者必须在 ego path、正在进入冲突区或对当前驾驶决策有直接影响。远处人行道背景行人不算。
- `TRAFFIC_LIGHT_ABNORMAL=YES`：必须是信号灯系统本身异常，例如冲突相位、横纵向同时绿/异常全红、闪烁、熄灭/暗灯等。车辆闯红灯、无前车红灯停车、普通红绿灯相位变化都不算灯异常。

这些口径应写进后续 Qwen 提示词：尤其 `HIGHWAY` 要像教初学者一样强调“受控通行拓扑”而非“直/宽/快”；`TRAFFIC_LIGHT_ABNORMAL` 则只需明确“看灯本身是否坏/冲突/异常”，不要把交通参与者违规混进去。

## 3. 本轮对答案表的关键修正

本轮最终 JSON 校验通过：`rows=196`，`group_count=196`，`excluded_scenarios=["noScenarios"]`。
2026-08-09 的复核随后发现旧的回答生成器仍把 `R3` 当作 `HIGHWAY=YES` 的机械规则，且一次人工回写误把
`Accident/R1/{R-E1,R-E2}` 套用了 `EnterActorFlow` 的理由。这两行已按各 Town 实图改回 `HIGHWAY=NO`。
最终表升级为 v2：默认行是审计过的 `scenario × RS × EVENT` 答案，混合拓扑只允许由显式
`visual_subgroup_overrides` 覆盖，不能凭 Town、场景名或 RS 标签自行翻转。

| 组合 | 修正 | RGB 依据 |
|---|---|---|
| `EnterActorFlow / R1 / R-E1` | `HIGHWAY=false -> true` | Town12/Town13 多 ID 显示高速/快速路入口、并入、护栏/隔离、受控通行或导流区拓扑；不能把它当普通直道。 |
| `EnterActorFlow / R3 / R-E1` | `HIGHWAY=false -> true` | 同一批完整 route 的 R3 regular-flow 段仍在受控主线/匝道环境；`R-E1` 仅表示常规跟车，不表示离开高速。 |
| `EnterActorFlowV2 / R1 / R-E1` | `HIGHWAY=false -> true` | Town12 多 ID 同样是高速/快速路 actor-flow/merge 背景，R1 只是分段标签，四问里仍应答 highway。 |
| `InterurbanActorFlow / R3 / R-E1` | `HIGHWAY=true -> false` | Town12 R3 短段实际是城际/郊区普通道路或路口附近，未见封闭高速、匝道、连续分隔主线或出入口控制。不能仅凭 R3 标签、直道或宽路置真。 |
| `Accident / R1 / {R-E1,R-E2}` | `HIGHWAY=true -> false` | 全 Town 审计均为城市、郊区、山路或普通分隔主干道；即使有多车道、护栏或雾雨，也没有受控匝道/出入口/主线拓扑。旧值是错误复制的 actor-flow 理由。 |

额外确认：

- `StaticCutIn / R3` 保持 `HIGHWAY=YES`：Town13 R3 样本有高速/快速路/匝道式隔离与受控道路形态；`U-E3` 是静态/侧向车辆侵入 ego path，`OBSTACLE=YES`。
- `VehicleTurningRoute` 与 `VehicleTurningRoutePedestrian` 的 `U-E4` 保持 `VULNERABLE=YES`：逐 Town 可见行人/骑行者进入或贴近 ego 转弯路径，不是普通转弯车误判。
- `CrossJunctionDefectTrafficLight / U-E7` 保持 `TRAFFIC_LIGHT_ABNORMAL=YES`：这是灯本身异常场景。
- `OppositeVehicleRunningRedLight / U-E6` 保持 `OBSTACLE=YES` 且 `TRAFFIC_LIGHT_ABNORMAL=NO`：对向/横向车辆违规穿越冲突区是路径冲突，不是灯故障。
- `ParkedObstacle × U-E2` 按用户要求统一保持 `OBSTACLE=YES`，即便个别当前帧因时间分散不一定清楚看到占道物。

## 4. 混合拓扑 caveat

`ParkedObstacle` 存在 Town12 highway-like fast-road subgroup：本轮全帧 sheet 中
`Town12_Rep0_1006_0_route0_01_10_14_44_57`、`Town12_Rep0_2967_1_route0_01_10_20_56_17`、
`Town12_Rep0_962_1_route0_01_09_14_36_56` 均显示分隔、多车道、连续护栏/隔离、快速路/受控通行视觉特征，并在 U-E2 span 看到车辆/静止物占用或压缩 ego path。v2 表将它写成显式 `parked_obstacle_town12_limited_access_fast_road` 覆盖：只有 route 已被 RGB 审计写入该 topology 子组、且当前路径仍受受控快速路结构支配时才给 `HIGHWAY=YES`。该覆盖当前只匹配 `ParkedObstacle/Town12/R1` 子组；Town12、本场景名、护栏或宽直道路本身都不触发覆盖；其余 aggregate 默认行保持 `HIGHWAY=NO`。

后续如果把表继续拆细到 `scenario × Town/topology × RS × EVENT`，应单独给该 highway-like 子组 `HIGHWAY=YES`；其他普通停车/路边障碍子组保持 `HIGHWAY=NO`。

## 5. 提示词纠偏要点

后续用于 Qwen 的四问提示词建议保留以下“教学式”规则：

1. 判断高速时先问：是否有匝道、出入口、导流 gore、连续隔离、封闭/受控通行、多车道快速路主线？如果只是直、宽、空、路边有护栏，一律不要答 highway。
2. 判断障碍时先问：是否有实体正在占用或压缩 ego path？普通动态车流、等待让行车辆、路边背景停车不能自动算 obstacle。
3. 判断弱势参与者时先问：行人/骑行者是否进入 ego path 或冲突区？只在 sidewalk/background 出现不算。
4. 判断灯异常时只看灯系统本身：冲突相位、闪烁、熄灭、全红/全绿等才算异常；车辆闯红灯或普通红灯等待不算。
5. 对 `ParkedObstacle × U-E2` 保持组合级一致性：只要该组合语义是停放/静止物占道绕行，就答 `OBSTACLE=YES`，不要因为某个当前帧暂时看不清占道物而改成 NO。
