# 正式 Phase3 数据通过：直接推进首轮训练

## 2026-10-13 首轮交回续审：两图已训练，修复后补评测和四图

最新收到 `phase3_epoch_report.zip`（246,461字节、17成员）和独立回执：外层SHA、CRC、全部成员SHA、475源码均与 `81a486f28` 发布一致。包完整不等于实验完成：`handoff=incomplete`、退出1，短两图完成2560个四卡步/10240次实际呈现，final adapter已保存；重载、test、四图尚未完成。

**本轮确认并修复两处编排缺陷：**

1. `server_phase3_epoch` 给LoRA评测显式传 `--history-rgb-mode`，违反原生eval“从adapter读取RGB合同”的要求。现在移除该覆盖，同时由原生配置读取输出/提示词合同，调用前额外验证choice、目标RGB模式和candidate提示词一致。不修改学生合同。
2. 状态文件被 `io.write_json` 的独占新建模式重复写入；这不仅遮蔽评测错误，也会在进入第二模式时失败。新增仅针对本次进度文件的原子替换，保留阶段及原始错误；不可变证据仍使用独占写入，旧文件不覆盖。

**可以复用已完成的两图一轮，不要重训。** 新增 `REUSE_EPOCH_ROOT`，仅接受这份已审ZIP SHA `5e28b686dedc62b941161fdbcdc9a22b977f1bb2d59122d9156d0a5a1266172d`。复用前逐项校验原准备记录、训练生产源码、实际消费回执、原始逐题输出、六份adapter资产及验证RGB的路径/解析目标/SHA。任何漂移阻塞，绝不凭目录名跳过训练。

新结果写入新目录；其中两图 `train` 只引用旧目录，旧权重不复制、不修改，验证及test结果另写。处理顺序：旧两图重载320题 → 固定test380题 → 四图从基座新训10240次 → 四图重载320题 → 固定test380题 → 自动诊断包。不是优化器断点续训，不会自动启动七轮。空间检查仍保留原两模式上限，复用一份权重时预算偏保守。

新版代码发布、按既有 `sync_automot.sh` 同步后，在服务器 AutoMoT/ 执行：

```bash
PREPARED_ROOT=checkpoints/phase3_prepare_20261012_231356_193369568 \
REUSE_EPOCH_ROOT=checkpoints/phase3_epoch_20261013_100331_732771255 \
OUTPUT_DIR="checkpoints/phase3_epoch_recovery_$(date +%Y%m%d_%H%M%S_%N)" \
bash qwen3vl_local/audit_joint/server_phase3_epoch.sh
```

**仍只交新输出目录的 `phase3_epoch_report.zip` + `phase3_epoch_package.json`，≤30MB。** 保留旧epoch目录和原交接两文件供核验；不重跑readiness、不重建数据、不覆盖旧实验。新入口使用 `release_sources_phase3_epoch_recovery_20261013.json`，旧清单留作来源证据。

### 本次模型与等量审计

- 原生parser重解析全部320条raw，与记录预测及all_ok逐项一致；实际验证ID集合与已验收固定清单完全相同、无重复；训练回执十事件各1024、验证各32。test各38是固定计划，尚无实际test成绩。
- 短两图规则一致率215/320=67.1875%，格式有效100%；减速4/58=6.90%召回，其中38道减速答KEEP；恢复18/38，左变道3/8，右变道12/14。不能仅凭总体分数批准长训。
- `POST_BYPASS_RETURN`最低12/32=37.5%；`JUNCTION_RULE_CONFLICT`32/32全对，但参考也全为STOP，不代表学会该事件所有动作。
- 每事件使用本批val多数动作的描述性上限合计177/320=55.3125%，低于模型整体，但切入24/32低于本批多数25/32、弱势交通参与者29/32低于多数31/32。这个统计使用验证标签，不是从train冻结的可部署基线，也不拿它选test或改标签。
- 未新增RGB目视、未独立加载服务器权重；不能把自动源信号一致当作所有事件视觉标签认证。本轮无新依据改标定或提示词，先完成四图对照，之后在训练来源上定向查减速/KEEP混淆。
- Phase3正式自动数据已可支持独立基线，不需要逐题人工标注。Phase4 U-E5真实proceed YES仍缺、v53正式池与联合隔离未验收，不能宣称两阶段整体完成。重复旧89路线不能补出新正例。

新增16项专项（本文件专项共25项）通过；完整CPU回归 **2412项通过，106.69秒**。包括原生eval配置解析、状态连续更新、复用SHA/软链接/资产集合漂移拒绝、合成两模式完整编排及失败打包；不冒称已实测服务器GPU重载。实际交回adapter元数据也通过本机原生CLI解析，未加载权重。本次修复随白名单发布，服务器同步后使用顶部续跑命令。

本机证据：`checkpoints/phase3_epoch_received_20261013/`；可发布核验回执：`phase3_epoch_received_verification_20261013.json`。下文是此前建库通过时的记录，原“从头跑两模式”的命令仅适用于没有已完成首轮的新实验；本次优先使用上面的复用命令。

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
