# UE3 RGB 标签对齐与 train route-diverse 修正（2026-09-03）

## 结论

当前不修改 production prompt v3，也不按单一 metadata 阈值批量重标 UE3。
本轮只修正 Phase2 数据构建的 train 抽样：每个类别先按
`(scenario, route_id)` 轮转覆盖不同 route，再从同一 route 取第二帧；val/test
继续使用旧 deterministic frame sampler，保证 frozen case 身份可比。

该结论来自 `checkpoints/ue3_route_diverse_full_rgb_audit/` 的 32 个 UE3 正例四帧
RGB，而不是由 scenario 名推断。复核分类为：

| RGB visual class | cases |
|---|---:|
| VISIBLE_ACTIVE | 12 |
| PRE_EVENT | 5 |
| POST_EVENT | 6 |
| DOMAIN_CONFLICT | 2 |
| 2RGB_UNOBSERVABLE | 2 |
| AMBIGUOUS | 5 |

12 个清晰 active case 只来自 2 条 route；其中 DynamicObjectCrossing/Town05 的
f23-f25 是三个 seed 全对的清晰横向进入对照，低照度 StaticCutIn/Town12 的
f75-f76、f80-f86 才暴露真实模型漏判。DynamicObjectCrossing/Town02 的 f86-f90
四帧看不到可确认的横向目标，却被连续标成 U-E3；StaticCutIn 的 f77-f79、f87-f89
在 newest 时刻已结束，仍处于 U-E3 span。

## RGB 决策与源规则联表

`audit_ue3_label_alignment.py` 将上述逐帧 RGB decisions 与
`collection_output/*_result.json` 的最终 `frame_event_annotation` 联接。当前 32 帧全部由
同一条 `event_dynamic_cutin_or_occupancy` 规则产生，但它同时覆盖六种相互冲突的视觉类别。

代表性源证据：

- DynamicObjectCrossing/Town02 f86-f94 是一个连续 9 帧 U-E3 span；f86-f90 的
  `dist_to_cutin_vehicle` 全为空，但 `vehicle_hazard/hard_decel/defect_conflict_vehicle`
  均为真。RGB 中 f86-f89 没有目标，说明这些通用 hazard 字段不能直接当作视觉 UE3 真值。
- 同一场景 Town05 f23-f25 的规则名和布尔字段几乎相同，但 RGB 明确显示近车横向进入，
  因此不能简单删除该规则或要求 `dist_to_cutin_vehicle` 非空。
- StaticCutIn active 与 post-event 帧的距离区间重叠；例如 active f84-f86 与 post f87-f89
  都可出现近距离/刹车相关字段，单一距离或 brake 阈值无法稳定切开事件尾部。

所以当前证据只支持降低连续 span 的训练权重，不支持自动改 prompt 或 collector 阈值。

## 代码修正

- `sampling.py` 的 route sampler 现在同时支持训练期对象和 dataset dict row。
- `build_dataset.py` 的 train UE/RE 桶改为 route round-robin；val/test 保持旧 sampler。
- `train.py` 每个 epoch 的 UE/RE work sampler 默认同样启用 route round-robin；因此即使
  UE3 恰好是最小原始桶，正式训练的 `FOCUS_BALANCE_COUNT=2048` 也不会先被少数长 span 占满。
- manifest 新增 raw/sample 后的逐 class route 分布，包括 route 数和
  `max_cases_per_route`，后续不再只看帧数均衡。
- `audit_ue3_label_alignment.py` 输出源规则、metric、U-E3 span、RGB 类别和可选旧 index
  route 集中度；明确标记为非正式指标且不改任何数据。

## 无 GPU 审计命令

默认当前目录为 `AutoMoT/`：

```bash
bash qwen3vl_local/sft_new_loop_phase2/run_ue3_label_alignment_audit.sh
```

输出：

- `checkpoints/ue3_route_diverse_full_rgb_audit/label_alignment/summary.json`
- `checkpoints/ue3_route_diverse_full_rgb_audit/label_alignment/summary.md`
- `checkpoints/ue3_route_diverse_full_rgb_audit/label_alignment/cases.jsonl`

若训练机的 index 位于别处：

```bash
INDEX=checkpoints/sft_new_loop_phase2_data/frame_index.jsonl \
  bash qwen3vl_local/sft_new_loop_phase2/run_ue3_label_alignment_audit.sh
```

## 新数据 smoke（不覆盖旧 index，不训练）

```bash
bash qwen3vl_local/sft_new_loop_phase2/run_route_diverse_data_smoke.sh
```

它会自动构建到 `checkpoints/sft_new_loop_phase2_data_route_diverse_smoke/`，然后比较
旧/新 index：val/test case identity multiset 必须完全相同；train 六桶数量必须不变；
非 INVALID 桶的 route 数不能下降、`max_cases_per_route` 不能上升。任一 guard 失败都会
以非零状态退出并提示不要训练。

预期 `train route_diverse=True`，`val/test=False`。在确认 train UE3 route 数增加、
`max_cases_per_route` 下降，并核对 val/test frozen 身份不变以前，不启动新的三 seed 训练，
也不打开 unseen-456。

正式训练入口默认 `TRAIN_ROUTE_DIVERSE=1`；如需复现实验旧口径才显式设为 0。每个 epoch 的
`balance/epoch_*.json` 会记录实际 route 分布，不能只看类别帧数。
