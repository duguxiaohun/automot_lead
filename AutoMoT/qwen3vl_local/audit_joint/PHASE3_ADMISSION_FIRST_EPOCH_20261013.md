# 正式 Phase3 数据通过：直接推进首轮训练

## 本次交接结论

收到 `phase3_prepare_report.zip` **163,418字节、19成员**及独立 SHA 回执；ZIP CRC、成员集合/每项SHA、471源码与已发布 `3f9daefe1420e199d0c6772536912b987cda1cec` 均核验一致。服务器退出0、`cpu_data_prepared`。**Phase3正式建库和CPU数据验收可以关闭，不再重建、不再readiness。**

- 正式来源为 `collection_results`；索引 **15,096行 = 14,124 train + 516 val + 456 test**，来源3510条run/3509物理组。
- binary、choice均完成原始meta复算，速度/动作不一致均0；索引物理组跨split重叠0，开发曝光隔离通过，holdout事件/组数补齐无 unresolved/deficit。
- 正式索引SHA：`925053e34c47f35dfe4987f14a1be3a8abdc206d75bea31964dc4ca9b8fcee99`。
- manifest在打包时多一个末尾换行；按原生构建器的无末尾换行序列化可重现登记SHA `12b8dd5391e9baaa0441900463f7063b6dea055cc51c66077dd442e3cd49912e`。这是编码差异，不是输入漂移。
- 本机核验的是服务器交回证据，没有收到完整索引及全部原始信号，不冒称本机再次独立重算全库；本轮新增RGB目视0。

### 事件间1∶1的实际清单

| 阶段 | choice 每事件 | 十事件总题数 | 额外 INVALID（仅binary） |
| --- | ---: | ---: | ---: |
| 一轮训练计划 | 1024 | 10240 | 2048 |
| val loss | 16 | 160 | 86 |
| val generation | 32 | 320 | 86 |
| 固定test | 38 | 380 | 76 |

十事件各自训练计划的1024题也都是独立问题；val/test无放回，已复核ID清单唯一性与顺序SHA。val/test各事件至少5个物理组。binary中的INVALID单独报告，不是第十一个事件，也不要求YES/NO等额。四图及短两图共用该标签池和选题规则。

## 发现并修复的训练边界

旧Phase3训练代码在输入超过 `max_length` 或打包失败时使用零损失继续推进步数，计划等量不能保证实际有效训练等量；teacher-forced val也会跳过此类题后继续评分。**确定代码路径缺口，不是本次服务器发生了训练缺题：本次尚未训练。**

本轮修改 `train.py`：

1. `max_steps=0` 的正式完整轮次在任一rank拒绝真实输入时同步停止；不把少训的轮次称为成功。显式步数限制仍属于smoke/截断实验，不宣称完整等量。
2. 每轮汇总全部rank实际参与损失的事件数，与计划逐项核对并生成 `balance/consumed_epoch_0001.json`；DDP填充项不计入实际样本，损失归零。
3. 固定val loss清单若有任一rank漏题，在汇总后拒绝评分；生成评测本身不跳题。

没有改生产标定、阈值、数据划分或提示词。新源码不需要重建已通过的索引。

## 标签质量与评测边界

十事件训练候选均有多个真实动作/KEEP支持，包括 Phase3 ONCOMING_INVASION 的 RESUME；这与 Phase4 U-E5 的“许可放行”是不同命题，不能互相代替。数据来自自动规则与原始信号，不需要逐题人工标注；沿用已有明确异常隔离和冻结开发清单。

部分holdout的事件内动作支持仍窄：`JUNCTION_RULE_CONFLICT` val候选43题均STOP，`VULNERABLE_CROSSING` test候选120题均STOP；部分转向/恢复动作只来自1–2个物理组。等量test不能证明所有动作都会，须看逐动作分数及多数类基线。binary的同RS错事件负例在val/test为0、已有准入明确标为不支持；首轮先用choice，不宣称已验收binary的该能力。这些限制应记录，**不要求再次重建当前正式池才能开始choice基线**。

Phase4真实U-E5 proceed YES仍0，v53正式全池及联合隔离尚未完成。不要复跑相同89条路线来凑“新证据”，不要用出画、实例截止或恢复运动补成YES。Phase3独立实验可先行；联合G0及两阶段整体交付仍不能称为已完成。

## 服务器下一步：一次命令跑完两模式首轮

新增 `server_phase3_epoch.sh/.py`，随本次白名单发布；**沿用 `sync_automot.sh` 同步后，再执行**。入口核验新的475源码清单，并消费这次已经通过的交接包和原始索引SHA，不再建库或回放全源。

在服务器 AutoMoT 目录、原automot环境中：

```bash
PREPARED_ROOT=checkpoints/phase3_prepare_20261012_231356_193369568 \
OUTPUT_DIR="checkpoints/phase3_epoch_$(date +%Y%m%d_%H%M%S_%N)" \
bash qwen3vl_local/audit_joint/server_phase3_epoch.sh
```

默认现有 `lead_data` 和 `checkpoints/Qwen3.5-4B`；位置不同时给实际 `DATA_ROOT` / `MODEL_DIR`。自动选择4张GPU；明确选卡只用前置 `GPU_IDS=0,1,2,3`。CPU预算失败时不会加载模型训练。

固定范围：candidate提示词、choice、`2rgb_short`→`4rgb`顺序，各从基座新训1轮、各10240次有效呈现、rank16/vision-off，保留现有学习率/预热设置。Phase3短两图为 `[-2,0]`/0.5秒，四图保留 `[-3,-2,-1,0]`/0.75秒，不改变Phase4自身时间合同。

- 首先CPU核对既有索引、计划与提示词合同，绑定验证RGB；按真实基座拓扑计算LoRA磁盘预算，不重复环境readiness。采用已收4B参数数目估算两模式增量约 **4.734GiB + 20GiB预留**，实际以本次 `budget.json` 为准，旧磁盘余量不当当前保证。
- 只保存两份最终adapter，不存best/中间checkpoint、不保存优化器、不复制RGB。原生Phase3没有恢复优化器续训能力；此首轮结束后是否正式长训按结果决定，不假称可无损接续。
- 训练结束的新进程加载**未合并LoRA**，重跑同一320题验证清单，冻结parser重解析raw逐题核对；检查保存/重载预测一致，不拿较早checkpoint的分数比较。
- 重载通过后执行固定380题test，读取全部事件；不以test挑checkpoint。test分数和模型质量独立报告，`first_epoch_verified`仅指一轮/重载/固定评测流程通过，不是已具备生产准确率，也不自动续7轮。
- 全程core限制和20GiB余量守护；失败保留现场，诊断仍自动打包，禁止自动清理旧产物。

交回终端最后打印目录下：

```text
phase3_epoch_report.zip
phase3_epoch_package.json
```

ZIP严格≤30,000,000字节：含预算、合同、实际消费数、最终验证/重载逐题输出、固定test指标/逐题记录、adapter SHA及日志尾部。权重/优化器/RGB不上传。任何失败先交同一对文件，不再次重建索引。

## 本机验证记录

原生CPU小模型训练循环验证了完整40题、十事件各4次的实际消费回执，以及输入被拒绝时不产生完成回执；该测试是合成线性模型、替代图像/损失构造，不是Qwen GPU训练。另有多rank缺题/消费求和、val漏题拒绝、重载raw矛盾/缺题/重复/输入变化、固定参数和排除权重的打包测试。

完整回归 **2396项通过，103.77秒**。首次全套的4个失败均是旧测试期待静默跳过验证题，更新预期后全套通过，原失败日志保留。原生CLI在本机缺服务器索引时退出2且未训练，仍生成26,344字节/3成员已核验包，475源清单matched。

完整回归及原生CLI回执见 `phase3_admission_verification_20261013.json`；本机完整产物在 `checkpoints/phase3_prepare_received_20261013/`。没有本机GPU训练、远程操作或新RGB目视。
