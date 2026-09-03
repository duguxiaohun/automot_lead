# UE3 RGB 标签对齐与 train route-balanced 修正（2026-09-03）

## 结论

当前不修改 production prompt v3，也不按单一 metadata 阈值批量重标 UE3。
本轮只修正 Phase2 的训练期 UE3 曝光：持续按 `(scenario, route_id)` 轮转，且单帧
每个 epoch 最多重复 10 次；val/test 继续使用旧 deterministic sampler，保证 frozen
case 身份可比。该修正只作用于 UE3，不把本次 RGB 结论外推到其它 EVENT 类别。

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
- 复核发现旧 `route_diverse_sample` 在 `FOCUS_BALANCE_COUNT=2048` 大于 UE3 原始
  1083 帧时，会先取完 1083 帧再循环整份结果；这仍会按原始 span 长度重复，不能视为
  真正的训练 route balance。
- `sampling.py` 新增训练专用 `route_balanced_sample`：原始桶耗尽后仍按 route 轮转，
  route 内先遍历不同帧再重复；单帧重复超过 10 次或总容量不足会直接失败。
- `train.py` 只对 UE3 默认启用上述 sampler；其它 UE/RE 保留原 route-diverse 逻辑，
  INVALID 保留联合签名分层逻辑。模型加载前硬校验 UE3 route 覆盖和单帧重复上限。
- manifest 新增 raw/sample 后的逐 class route 分布，包括 route 数和
  `max_cases_per_route`，后续不再只看帧数均衡。
- `audit_ue3_label_alignment.py` 输出源规则、metric、U-E3 span、RGB 类别和可选旧 index
  route 集中度；明确标记为非正式指标且不改任何数据。

## 无 GPU 审计命令

默认当前目录为 `AutoMoT/`：

```bash
bash qwen3vl_local/sft_new_loop_phase2/run_ue3_label_alignment_audit.sh
```

脚本会按 32-case decisions 身份自动定位顶层副本或 frozen experiment 内的真实审计目录。
如需强制指定，可设置 `AUDIT_ROOT=<包含 manifest.jsonl 的目录>`。输出位于解析后的
`<AUDIT_ROOT>/label_alignment/`：

- `summary.json`
- `summary.md`
- `cases.jsonl`

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

该脚本只验证 index 构建合同。因为 UE3 是最小桶，新旧 index 中 UE3 都可能保持
1083 帧 / 177 routes / max 20 cases per route；所以它的 `passed=true` 不能证明正式训练
目标 2048 下的 UE3 重复权重已经改善。

## 训练采样 smoke（不加载模型）

```bash
bash qwen3vl_local/sft_new_loop_phase2/run_ue3_train_route_balance_smoke.sh
```

输出 `checkpoints/ue3_train_route_balance_smoke/{summary.json,summary.md}`，并比较相同
1083 条 UE3 原始帧在目标 2048 下的两种重复方式。必须同时满足：v3/2RGB prompt hash
未变、目标数正确、177 条 raw route 全保留、新 sampler 的最大 route 曝光严格小于旧
整桶循环、任一帧重复不超过 10。任一 guard 失败都禁止训练。

使用本次 `label_alignment_summary.json` 保存的真实 train route 计数做 CPU 投影：raw 为
1083 cases / 177 routes / max 20 cases per route；旧整桶循环到 2048 后最大 route 曝光为
31；候选 sampler 同为 2048 cases / 177 routes，但最大 route 曝光降到 12，最大单帧重复
为 10。该投影只验证采样数学；训练机仍必须对完整 `frame_index.jsonl` 运行上述 smoke。

正式训练入口默认 `TRAIN_ROUTE_DIVERSE=1`、`TRAIN_UE3_ROUTE_BALANCED=1` 和
`MAX_TRAIN_UE3_FRAME_REPEAT=10`。每个 epoch 的 `balance/epoch_*.json` 会记录实际
route 分布与 UE3 frame repeat；如需复现实验旧口径才关闭新开关。

## 单 seed pilot（smoke 通过后）

只跑 seed 20260810 到 step 4000，不直接启动三 seed，也不打开 unseen：

```bash
GPU_IDS=0,1,2,3 \
HISTORY_RGB_MODE=2rgb_endpoints \
INDEX=checkpoints/sft_new_loop_phase2_data/frame_index.jsonl \
OUTPUT_DIR=checkpoints/sft_new_loop_phase2_ue3_route_balance_pilot/seed_20260810 \
SEED=20260810 MAX_STEPS=4000 SAVE_STEPS=4000 \
FOCUS_BALANCE_COUNT=2048 TRAIN_ROUTE_DIVERSE=1 \
TRAIN_UE3_ROUTE_BALANCED=1 MAX_TRAIN_UE3_FRAME_REPEAT=10 \
GENERATION_EVAL_ROUTE_DIVERSE=0 \
bash qwen3vl_local/sft_new_loop_phase2/train.sh ddp
```

`GENERATION_EVAL_ROUTE_DIVERSE=0` 显式复用旧 seed 20260810 的 validation sampler，避免
同时更换训练和选优 case。先比较 step 2000/4000 的 frozen validation。只有 UE3 recall
高于旧 seed 20260810 的
`0.40625`，同时 UE6/INVALID/applicable RE 继续通过 `0.80/0.80/0.50` guard，才允许
复制到另外两个 seed；否则停止这条路线，不改 prompt、不打开 unseen。
