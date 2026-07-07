# ROAD / EVENT RS Sync Audit 2026-07-06

本次审计目标：在 ROAD_STRUCTURE 候选收紧后，确认 EVENT 是否需要同步收紧或补充，尤其是高速 R3、TwoWays R2、路口 R4/R5 与异常事件 U-E2/U-E3/U-E4/U-E5 的耦合边界。

## 1. 结论

- 已在 `collector.py` 增加最终 EVENT 候选池兜底：每帧最终 EVENT 必须属于 `scenario 精细白名单 ∩ 当前 primary RS 的 EVENT 候选池`，并始终保留当前 RS 的 regular event。
- R3 是高速/匝道/合流道路结构，不等于全程 R-E3。R3 regular fallback 已明确为 R-E1；只有 merge/exit/actor-flow core 或真实目标导向变道才分别进入 R-E3/R-E2。
- `*_TwoWays` 候选 RS 删除 R1；有效可行驶通道为对向单车道时默认 R2，四车道但两侧停车/开门/障碍导致侧向 lane 不可行驶也按 R2。借对向绕障核心由 U-E2 表达，核心后回目标/原车道由 R-E2 表达。
- R4/R5 默认删除 U-E2/U-E3，避免路口等待、红灯排队、路口起步被误标为绕障/切入；但 `AccidentTwoWays` 在 R2 core 与 R4/R5 重叠时保留 R2 overlay，允许 U-E2/R-E2 优先于 R-E4/R-E5。
- `InvadingTurn` 是另一个例外：即使 primary RS 为 R4/R5，只要是对向异常侵占自车道，U-E5 仍可保留。

## 2. 全量验证

命令：

```bash
python AutoMoT/keyframe_filter/quick_start.py annotate-rs \
  --scenario all \
  --output-dir /tmp/automot_event_rs_sync_all_20260706
```

结果：

- 覆盖 43 个 scenario、8614 条 route、1062401 帧。
- EVENT/当前 RS 候选池违规：0。
- EVENT 分布：R-E1 388744，R-E2 51178，R-E3 10631，R-E4 316425，R-E5 137847，U-E1 3180，U-E2 63563，U-E3 2240，U-E4 33663，U-E5 3010，U-E6 21161，U-E7 16880，U-E8 13879。
- RS 分布：R1 306667，R2 146678，R3 76622，R4 361730，R5 170704。

高风险场景抽查：

| Scenario | EVENT 分布摘要 | RS 分布摘要 | 结论 |
|---|---|---|---|
| HighwayExit | R-E1 79.3%, R-E3 17.0%, R-E2 3.7% | R3 100.0% | 高速普通段回 R-E1，出口核心保留 R-E3 |
| HighwayCutIn | R-E1 98.4%, R-E2 0.7%, R-E4 0.9% | R3 99.1%, R4 0.9% | 不再全程 R-E3；少量灯控子集只随 R4 出现 |
| EnterActorFlow | R-E1 79.9%, R-E3 17.7%, R-E2 2.3% | R3 100.0% | actor-flow 核心才 R-E3，进入后正常跟车回 R-E1 |
| EnterActorFlowV2 | R-E1 84.9%, R-E3 13.4%, R-E2 1.7% | R3 100.0% | 同上 |
| MergerIntoSlowTraffic | R-E1 82.7%, R-E3 15.7%, R-E2 1.5% | R3 99.9%, R4 0.1% | 合流核心不吞掉后续跟车 |
| MergerIntoSlowTrafficV2 | R-E1 85.0%, R-E3 13.7%, R-E2 1.2% | R3 100.0% | 同上 |
| AccidentTwoWays | R-E1 37.9%, U-E2 29.0%, R-E2 17.4%, R-E4 12.0%, R-E5 3.6% | R2 80.8%, R4 15.5%, R5 3.7% | R1 已删除；U-E2/R-E2 保留，路口 regular 不再强行吞核心 |
| ConstructionObstacleTwoWays | U-E2 35.4%, R-E1 35.3%, R-E2 10.8%, R-E4 14.2%, R-E5 4.3% | R2 81.4%, R4 14.3%, R5 4.3% | R2 与 U-E2/R-E2 边界正常 |
| ParkedObstacleTwoWays | R-E1 34.6%, U-E2 31.4%, R-E2 13.8%, R-E5 17.1%, R-E4 3.0% | R2 79.8%, R5 17.1%, R4 3.0% | R2 主体稳定，STOP/无灯路口走 R5/R-E5 |
| VehicleOpensDoorTwoWays | R-E4 49.9%, R-E1 25.6%, R-E2 10.3%, U-E2 8.8%, R-E5 5.3% | R4 49.9%, R2 44.8%, R5 5.3% | 存在大量真实灯控段；开门/对向单车道段仍保留 R2/U-E2/R-E2 |
| StaticCutIn | R-E1 73.3%, R-E3 9.0%, U-E3 6.2%, R-E4 7.1%, R-E5 3.5% | R1 46.8%, R3 42.6%, R4 7.1%, R5 3.5% | 混合场景按路线拓扑拆分，不再保留独立停车结构 |
| VehicleTurningRoutePedestrian | R-E4 35.2%, R-E5 28.3%, R-E1 19.9%, U-E4 16.7% | R5 44.1%, R4 36.1%, R1 19.9% | 行人事件 U-E4 未被 R4/R5 regular 完全吞掉 |
| noScenarios | R-E1 64.3%, R-E5 17.7%, R-E4 12.6%, R-E2 5.3% | R1 69.7%, R5 17.7%, R4 12.6% | 普通场景可含真实路口，默认仍以 R1/R-E1 为主 |

## 3. 当前代码 smoke

全量命令启动后又修正了 R3 regular fallback，因此额外跑当前代码小样本：

```bash
python -m py_compile AutoMoT/keyframe_filter/collector.py AutoMoT/keyframe_filter/quick_start.py
python AutoMoT/keyframe_filter/quick_start.py annotate-rs \
  --scenario EnterActorFlow,EnterActorFlowV2,HighwayExit,HighwayCutIn,MergerIntoSlowTraffic,MergerIntoSlowTrafficV2,AccidentTwoWays,ParkedObstacleTwoWays,ParkingCutIn,StaticCutIn,VehicleOpensDoorTwoWays,InvadingTurn \
  --max-routes 3 \
  --max-frames-per-route 120 \
  --output-dir /tmp/automot_event_rs_sync_postpatch_smoke_20260706
```

结果：

- 覆盖 36 条 route、3760 帧。
- EVENT/当前 RS 候选池违规：0。
- EVENT 分布：R-E1 2180，R-E2 178，R-E3 322，R-E4 374，R-E5 146，U-E2 274，U-E3 93，U-E5 193。
- smoke 示例中 `EnterActorFlow` frame 0-3 为 R3 + R-E1，frame 4 进入 trigger core 才切 R-E3，符合“高速普通跟车是 R-E1，合流核心才 R-E3”的口径。

## 4. 后续注意

- 非常高比例的 R-E4/R-E5 不能单独视为 EVENT 错误；如果 primary RS 本身长时间为 R4/R5，EVENT 跟随 regular 是正确耦合。此时要回查的是 RS 路口窗口是否过宽。
- 若未来继续收缩 R4/R5 窗口，必须同步跑 EVENT/当前 RS 候选池违规检查，防止恢复段 U-E2/R-E2 被误吞或 regular event 未同步。
- `collection_output/` 仍是本地证据目录；本文件只记录可追踪结论和验证命令，不引用本地输出产物入库。
