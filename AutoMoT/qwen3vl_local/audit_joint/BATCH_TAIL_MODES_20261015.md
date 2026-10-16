# 连续段尾部拼接、可见GPU与按模式batch

在v61主干复现用户提供的三项问题，逐项修正。真实小池检查仍只读已有train，不采集新数据、不回放老师、不运行GPU。

## 1. 补齐尾部时保留完整段

共享采样策略升为 `event_equal_contiguous_unseen_first_v3`。曝光优先级不变：问题次数最少、锚点次数最少，然后在batch开头优先取长段；尾部先在当前曝光层级内寻找**完整段长度之和恰好等于空位**的组合。组合优先最少段数，再优先较长段。无完整组合时沿用最长段截取，不先塞一个无法补齐的零碎段导致更多断点。每个锚点仍在合格问题中按action欠额与剩余机会选择。

四卡B2、段长6／6／2／2现得到6＋2、6＋2；6／6／1／1／1／1得到6＋1＋1、6＋1＋1。组合受真实段数量限制，不能把一个长度1的段重复使用两次。这个尾部算法不跨曝光层级追求凑整，不合成跨路线输入历史；action是连续性之后的软目标。它是当前batch尾部的确定性组合，不宣称全epoch的全局最优排布。

报告新增每event的 `tail_packing.whole_piece_tail_choices` 和 `truncated_pieces`。后者仍可能非零：当前曝光层级没有可完整填充的组合时需要截段。策略v3拒绝旧v1／v2曝光history，不静默迁移旧checkpoint。

同一真实1457题train小池，seed20260929、四卡B2、1024/event，源码插桩逐次检查“存在相同action组成的完整短段可补空位却截断长段”：

| 模式 | 旧v2可避免截断选择 | 新v3 | 旧总分段数 | 新总分段数 |
|---|---:|---:|---:|---:|
| 两图 | 156 | 0 | 4214 | 4004 |
| 四图 | 83 | 0 | 3497 | 3266 |

这些是选择次数／分段数，不是受影响题数。两模式仍每event1024、覆盖全部1457个问题。16/event的五轮累计覆盖仍151／278／383／479／575，首轮U-E3仍14个proceed/NO＋2个hold/YES。其它event的action分布可因完整段选择改变，不能宣称所有action目标逐项不变或都完全满足。

## 2. 自动选卡遵守可见范围和空闲阈值

Phase3专用入口和Phase4 benchmark复用现有 `sft_new_loop_phase4.devices.select_gpus`，自动选择仅考虑 `CUDA_VISIBLE_DEVICES` 允许的卡（支持完整UUID或数字索引），并要求显存占用 **<4096 MiB、利用率<20%**。在合格卡中排序选择；不足四张、所有卡忙或nvidia-smi不可用时停止，不退回0／1／2／3，也不自动等待或抢占。

Phase4当前学生 `rgb_short/run.sh`、独立evaluate和reload_check也保留原可见范围，去掉选择前的mask清除。显式 `GPU_IDS` 保持用户指定优先，可覆盖原mask和空闲过滤，使用者应确保指定卡适合本次任务；它不是资源预约或持续占用锁。

新增实际shell＋实际selector的模拟硬件测试，覆盖全部99%／70GB忙卡、只可见卡7、空mask、UUID限制、显存边界、高利用率低显存、显式pin。未启动真实GPU。原生teacher／producer及原生devices源码没有改动。

## 3. Phase4首轮分别配置两种模式

`phase4_first_epoch` 新增 `--batch-size-rgb2`、`--batch-size-rgb4`；原 `--batch-size` 仍是两模式共同的后备值，省略专用参数即沿用它。可选1／2／4／8，每次更新固定32，分别计算累积8／4／2／1。每模式的准入和训练获得自己的BATCH_SIZE；旧shell残留的BATCH_SIZE／ACCUMULATION不会覆盖这些CLI值。

例如**只有独立实测已支持两图B2、四图B1时**，使用：

```bash
GPU_IDS=0,1,2,3 python -m qwen3vl_local.audit_joint.phase4_first_epoch \
  --prepared checkpoints/phase4_v57_20261014_232157_762998141 \
  --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
  --batch-size-rgb2 2 --batch-size-rgb4 1 \
  --output "checkpoints/phase4_modes_epoch1_$(date +%Y%m%d_%H%M%S_%N)"
```

该示例只演示配置能力，不代表已有Phase4GPU选型结果。入口继续按两模式顺序各跑一轮／重载／固定test，并自动交接。`workflow.json.mode_configuration`、每阶段training_configuration、日志均记录实际配置；两模式不同的时候顶层batch_size和accumulation置null，避免误报一个共同值。两模式相同则兼容保留原顶层值。交接包保留完整workflow及各自采样／更新记录。

本轮改变共享采样与当前学生运行源码合同，最终同步后从已有有效v57 data4生成新view；旧view和旧学生checkpoint不改SHA伪装兼容。原生producer和题目成员SHA保持，不需要重做约20小时的原生建库。正在使用旧版源码执行的实验保留其原版本，不在运行中覆盖源码。

当前v62来源清单：`release_sources_v62_batch_tail_modes_20261015.json`，验证回执：`batch_tail_modes_verification_20261015.json`。本轮未push、未目标4B GPU实测；下一步仍是同步最终代码后的Phase4实测，再按各模式结果设置首轮参数。

## 4. 最新复审收尾与剩余验收

用户交回的独立只读审计摘要支持关闭上述三项问题：205项专项通过，150个随机池各4轮的配额／曝光／history／跨rank重组通过；真实小池覆盖、五轮累计覆盖及U-E3补偿与此前本地结果一致。这些数字来自用户提供的摘要，未收到该轮原始测试日志，不与此前本地261项相加，也不记作本轮重新执行。

本次核对保留v62运行源码和采样v3合同，修正文档残留的“采样v2”，补齐实测后复用view并分别传batch的完整命令。没有新增确定代码缺陷时，不再改变训练策略或增加一个合同版本。

剩余步骤依次是：发布并同步v62 → 从有效v57 data4生成当前view → Phase4目标4B四卡固定更新量实测 → 分别按两／四图的有效实测结果选batch → 复用同一view完成正式首轮、重载与test。完整命令和交回文件见 [Phase4四卡实测](PHASE4_BENCHMARK_20261015.md)。Phase4显存／吞吐及模型效果仍待服务器结果；连续性是确定性优先策略，action是软补偿，不保证每batch动作等量或全epoch全局最优。
