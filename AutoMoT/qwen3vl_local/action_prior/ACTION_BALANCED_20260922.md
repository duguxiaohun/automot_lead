# action-balanced 容量回流修订（2026-09-22）

本修订替代同日最初“严格动作等量、1440次/轮”的方案。Phase3 与三条 Action 入口同步。

- RE5 是纵向动作域。4帧右变道均并发 RE2/RE5；保留全局右变道 token 和事件事实，取消 RE5×右变道采样归属，仍进入 RE2。
- 每事件内先完整池循环，再对余量在各动作容量内均分；缺额交给支持充足的动作。不删除合法稀少动作，不改成 KEEP/UNCOND。
- Action 事件严格1:…:1:2，新action-balanced默认单帧全局上限2（event-balanced仍为8）；共享帧冲突通过联合分配回流，最少偏离动作目标后再最大化独立帧覆盖。
- Phase3 choice按主要动作、binary/构建按合法域内原签名，均复用同一容量函数；INVALID独立覆盖规则不变。

完整历史训练池843913帧，最新1/4 rank各7轮回放通过：每轮23784次，各特殊事件1982、背景3964，
21251不同帧，最多重复2次。4条域外RE5右变道归属排除，帧仍保留RE2归属与原token。
共享帧导致10次动作目标配额回流，事件总配额和全局重复上限均满足；目标及实际值分开保存。

| 对照 | 每轮呈现 | 不同帧 | 最大重复 | UE3 RESUME呈现 | UE6 KEEP呈现 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原显式上限8 | 95136 | 65563 | 8 | 328 | 163 |
| 新默认上限2 | 23784 | 21251 | 2 | 82 | 64 |

这是降低重复强度的起点，不是效果最优结论。7轮的总更新数随预算缩小，不能只对齐轮数比较两行效果。
2是整轮跨事件共享帧的硬上限；UE3整体只循环两轮，但每次仍是同样的41帧RESUME，不能算作82份独立证据。

为何不合并低频标签：KEEP有真实保持含义，UNCOND是不提供条件，数量少不证明动作错误；
UE3×RESUME虽仅41帧却有21物理路线，且RESUME全训练池15231帧、七类embedding跨事件共享。
若把少于100帧的动作都抹掉，UE6只剩STOP；当前保留正确语义，减少额外重复。
条件屏蔽仅适合另做明确不可靠标注的消融，必须保留原标签、mask原因和特殊事件身份；当前不启用。

`support.cells/events` 记录帧数、物理路线、呈现次数及呈现/可用帧比值。
低于100帧或10条路线只产生 `review_flags`，不删标签、不改配额、不据val/test重新挑训练阈值。
Phase3构建报告共用 `support_diagnostic`，在 `signature_support` 中记录同样信息；其自身预算规则不变。

| 事件/动作 | 可用帧 | 上限8配额 | 新上限2配额 |
| --- | ---: | ---: | ---: |
| UE1/DECELERATE | 279 | 837 | 279 |
| UE1/STOP | 2525 | 6467 | 1495 |
| UE1/RESUME | 140 | 420 | 140 |
| UE1/KEEP | 68 | 204 | 68 |
| UE2/DECELERATE | 5505 | 1400 | 331 |
| UE2/STOP | 33004 | 1400 | 330 |
| UE2/RESUME | 1469 | 1400 | 330 |
| UE2/LANE_CHANGE_LEFT | 7313 | 1400 | 330 |
| UE2/LANE_CHANGE_RIGHT | 3558 | 1400 | 330 |
| UE2/KEEP | 928 | 928 | 331 |
| UE3/DECELERATE | 184 | 1472 | 368 |
| UE3/STOP | 721 | 5768 | 1442 |
| UE3/RESUME | 41 | 328 | 82 |
| UE3/KEEP | 45 | 360 | 90 |
| UE4/DECELERATE | 1772 | 1772 | 349 |
| UE4/STOP | 19216 | 2075 | 348 |
| UE4/RESUME | 239 | 239 | 239 |
| UE4/LANE_CHANGE_LEFT | 1526 | 1526 | 349 |
| UE4/LANE_CHANGE_RIGHT | 865 | 865 | 348 |
| UE4/KEEP | 1451 | 1451 | 349 |
| UE5/DECELERATE | 835 | 1831 | 496 |
| UE5/STOP | 657 | 1475 | 495 |
| UE5/RESUME | 527 | 1215 | 495 |
| UE5/KEEP | 1623 | 3407 | 496 |
| UE6/DECELERATE | 57 | 289 | 114 |
| UE6/STOP | 1408 | 7043 | 1642 |
| UE6/RESUME | 86 | 433 | 162 |
| UE6/KEEP | 32 | 163 | 64 |
| UE7/DECELERATE | 318 | 2036 | 506 |
| UE7/STOP | 410 | 2588 | 598 |
| UE7/RESUME | 182 | 1220 | 364 |
| UE7/KEEP | 326 | 2084 | 514 |
| RE2/DECELERATE | 5034 | 1346 | 331 |
| RE2/STOP | 1686 | 1346 | 330 |
| RE2/RESUME | 1198 | 1198 | 330 |
| RE2/LANE_CHANGE_LEFT | 2032 | 1346 | 330 |
| RE2/LANE_CHANGE_RIGHT | 5842 | 1346 | 330 |
| RE2/KEEP | 6040 | 1346 | 331 |
| RE3/DECELERATE | 870 | 870 | 459 |
| RE3/STOP | 235 | 235 | 235 |
| RE3/RESUME | 865 | 865 | 458 |
| RE3/LANE_CHANGE_LEFT | 201 | 201 | 201 |
| RE3/LANE_CHANGE_RIGHT | 171 | 171 | 171 |
| RE3/KEEP | 5744 | 5586 | 458 |
| RE5/DECELERATE | 9841 | 1982 | 496 |
| RE5/STOP | 25434 | 1982 | 495 |
| RE5/RESUME | 10489 | 1982 | 495 |
| RE5/KEEP | 6082 | 1982 | 496 |
| REGULAR_BACKGROUND/UNCOND | 503569 | 15856 | 3964 |

Phase3历史完整候选池十context、7轮×两种预算（容量内及超整池）数值回放通过；没有新RGB审阅。
上述输入是在变更前按正式合同核验并读取的旧池，只在临时文件中回放新采样器；没有修改旧manifest或绕过生产校验。
新默认 Phase3 索引为 `checkpoints/sft_new_loop_phase3_data_v22`，提示词/动作标定仍沿用v21。
采样源码加入mapping hash，生产必须重建；Action自动准备按新hash生成候选/full map，不覆盖旧run。
旧训练必须使用原源码及原产物恢复；本次未跑真实GPU、未声称性能提升。

新生产产物准备完成后，可使用（AutoMoT/下）：

```bash
python qwen3vl_local/action_prior/audit_data_capacity.py --action-balanced --high-level-action-token \
  --data-root lead_data --data-dir checkpoints/action_prior_data \
  --event-balance-index "<新full_event_mapping.jsonl>" \
  --num-epochs 7 --world-sizes 1 4 --report /tmp/action_capacity_return.json
```

本地数值回放报告 `/tmp/capacity_return_replay.json`、`/tmp/phase3_capacity_replay.json` 不入库。

## 简易实验命令

以下在AutoMoT/下执行；两个消融使用各自同名入口和相同采样参数。

```bash
# 新默认：action-balanced，全局单帧最多2次；默认四图
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --action-balanced --high-level-action-token
# 单当前图
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --action-balanced --high-level-action-token --rgb-frame-count 1
# 显式禁用重复，或把1改成8做原容量方案对照
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --action-balanced --high-level-action-token --event-balance-max-frame-repeats 1
```

同预算比较采样方法时，两组均给 `--event-balanced-epoch-samples 23784 --event-balance-max-frame-repeats 2`，
只切换 `--event-balanced` / `--action-balanced`，保持token、图数、split、seed、world、累积和轮数一致。
23784来自本次标签池；换数据后先预检共同可行预算，不强行沿用。显式容量不足会报错，不偷偷抬高重复上限。
若要研究不同重复上限本身，必须另外控制总更新数与验证候选点，不能把更少训练直接归因为采样改进。
CLI显式值优先于环境值；续训固定保存上限（含旧8），源码合同仍不允许用新源码续旧run。

最新CPU检查908项通过、2项缺只读runner依赖未执行；没有绕过生产指纹，没有真实GPU效果结论。
新回放报告 `/tmp/cap2_replay.json` 仅作本地数值验证，不入库。
