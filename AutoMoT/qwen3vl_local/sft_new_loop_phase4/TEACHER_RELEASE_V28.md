# v28：修复老师释放路径并扩充独立抽检池

日期：2026-10-05。任务合同 v28/data_v28，老师 v3，批准政策仍为 v2。旧产物需要原源码；本版使用新候选、新生产索引、新数据目录，不能改旧 hash 复用。

## 已落实的修改

- `teacher_rules.scoped_anomalies` / `teacher_replay.route_records`：异常只删失决定性参与者或局部冲突走廊受影响的实例；检查跳变两端。无关的可见删除仍记入路线异常账本和完整场景安全检查，但不再清空所有实例。异常离开完整因果历史后，同一 ID 可凭新的可见冲突重新建实例，不跨断点接续旧状态。缺图、无效几何等整帧问题仍会中断全部实例。
- readiness 不再因自车已动而被强制改成 catchup UNKNOWN；条件仍成立就保持 YES。运动仅用于执行回执，不能代替条件。修复 UNKNOWN 帧误复用上一次推进记录、重复提交执行回执的问题。
- 稳定跟车使用五次连续观察的间距、相对速度、速度变化及闭合趋势；开发跟车时距门槛为 0.5 s，同时保留至少 3 m 间距、闭合和横向约束。短时距或单帧速度相似不能单独判定完成。UE1 可以经 recover_follow → complete 完成，连续可见且充分远离的前车也可支持无约束恢复。**这些是待独立校准的状态观测阈值，不是经过认证的驾驶安全距离。**
- UE1/UE4 的局部通行条件检查本车道适用灯/停车控制；路口距离本身不再导致永久 UNKNOWN。控制身份缺失或灯态不匹配仍弃答，红/黄灯和已知停车义务仍否决推进，并保留 hold。未实现停车义务履行台账；不能凭专家起步认定已履行。
- 候选召回范围从 18 m 扩到 30 m；UE1 支持连续减速的近距前车。实际冲突条件仍只看参与者附近的局部走廊。UE4 的远离运动做自车平移补偿，暗图不能靠几何强行变为可见。
- `teacher_pool` 新增只检查“能否建立实例”的前瞻独立路线筛选，按场景/Town 轮转，物理组去重，排除训练和已有模型评估预约。筛选不调用条件规则或看答案。
- `teacher_approval.budget_requirements` 报告批准预算：95% Wilson 下界的当前阈值下，全对时 YES 至少 73 个、NO 至少 35 个，且分别来自至少 20 条物理路线，转移附近至少 30 个判断；有错需要更多。73+35 已超过总量 100 的最低门槛。
- `teacher_review prepare --rgb-modes 2` 支持先审 2 图。本次每类上限 300、每条路线上限 10。4 图类别仍需自己的批准；没有把 2 图批准自动迁移给 4 图。
- `preflight.sampling_repetition` 报告每事件池大小、实际呈现量、独立题量、重复呈现和最大单题次数。事件 1:1、共享帧 cap8、训练和选优逻辑保持。

## 实跑结果与限制

766 项专项测试通过，2/4 图真实 `MODE=check` 通过。每个模式的 902 道监督题和 539 道待审题 RGB 均核验；train/val/test/review 四个 JSONL 与 v27 逐字节相同：训练 443 题/45 条物理路线，验证 273 题，测试 186 题。

| 开发回放范围 | 每模式 YES / NO / UNKNOWN | proceed YES | 完成实例 |
|---|---:|---:|---:|
| 原 17 路线、1615 帧 | 18 / 55 / 125 | 12 | 1 |
| 扩展 23 条既有训练路线、2598 帧 | 60 / 264 / 697 | 41（UE1 6、UE4 35） | 1 |

原 17 路线的 v27 对照是 15 / 97 / 126，proceed YES 11、完成 0。**并非所有路线都已释放成功，也不是新增了 60 道训练监督。**

1030_0：f72–73 proceed YES，f75 recover_follow YES，f76 complete YES，实例真实结束。3_23 恢复行人释放正例；3_10 f127–129 为释放 YES，后续决定性目标异常仍截止。2679_1 已能建立两个行人实例，但目标异常使其截止；3_26 仍无可靠 UE4 实例；4_38 雾夜仍不通过可见性门槛。不能为了产量把这些改成 YES。

本轮只复看 3 条既有训练路线的 8 张原图：1030_0 f68/72/76，3_23 f134/138/142，3_10 f125/129。前车远离及行人向左离开本车道有画面依据；这是规则作者开发复核，不是独立盲审，不计新完整序列、不新增人工标签。逐图 SHA 见验证 JSON。

## 独立池与可执行材料

按固定顺序筛了 384 条留出路线，新增 69 个物理组（val40/test29），UE1、UE4 各有 40 条带实例的路线，11 条两类重叠。原 40 条预约和原 split 保留，合计 109 条。筛选记录 19,260 次帧源读取/投影异常，均未作为 NO 或许可；此次是有实例路线筛选，不是全量有效帧验收。实例存在也不保证每条边的 YES/NO 路线数达标，实际缺额由审核统计给出。

新冻结清单为 `producer_manual_check_plan_v8_20261004.json`，实例筛选证据为 `teacher_independent_pool_v1_20261004.json`（文件日期沿用本轮工作目录初始命名）；新真实注册表 `teacher_registry_v3.json` 仍为空。扩大的是抽检可用池，没有填写参考答案或降低批准标准。

- 开发公开包：`/tmp/phase4_v28_work/development_review/index.html`，240 卡，2 图。
- 独立公开包：`/tmp/phase4_v28_work/independent_review/index.html`，1485 卡，2 图。
- 审核填写各公开目录的 `decisions.json`。私有答案快照为上一级 `development_private.json`、`independent_private.json`，不要给盲审者。
- 独立图像及逐题答案没有被规则作者目视检查或用于调规则。包生成不等于完成审核。

`/tmp` 不是持久保存位置，迁移时使用新目录重建。当前完整候选清单 `/tmp/phase4_v28_work/candidates_final.json` 仍为 9712 路线/1,637,680 帧，split8166/765/781。

从仓库根目录使用依赖环境（或设置好 `PYTHONPATH=AutoMoT` 的同等环境）：

```bash
PYTHONPATH=AutoMoT /tmp/automot-qwen35-env/bin/python -m qwen3vl_local.sft_new_loop_phase4.teacher_review prepare \
  --candidate-pool /path/current_candidates.json \
  --production-index /path/current_production/index.json \
  --purpose independent --rgb-modes 2 --per-class 300 --per-route 10 \
  --private-snapshot /path/private/independent.json \
  --public-dir /path/reviewer/independent --data-root AutoMoT/lead_data
```

逐项审核完成后的 import/assemble/入库命令沿用 [v27 流程](TEACHER_REVIEW_V27.md)，替换成当前源码生成的产物路径。不能直接复用 v27 包。

## 仍未完成

真实规则批准 0、自动监督新增 0，有限 `trainable=true`、`formal_data_ready=false`。UE1 的真实恢复完成路径已接通，不能再说其“结构上永远不能产生 YES”；但其余八类老师、自动视觉 catchup、十事件独立评估和真实规则准确率仍未完成。扩池不能代替审核，测试通过不能代替规则准确率。

没有运行全 9712 路线标签生产、GPU/4B/DDP/CARLA 验收；全量用时仍不能由少数短路线保证。原人工标注、taxonomy/prompts/calibration/observation、控制器、安全许可、采样、训练、选优及 Phase3/Action 未改。
