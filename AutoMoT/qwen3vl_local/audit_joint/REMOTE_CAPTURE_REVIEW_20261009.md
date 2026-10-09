# 首份服务器缺失资产交接审计

审计对象：`checkpoints/capture.audit/`，收到的是解压目录。结论：**G0 诊断交接完成，G0 验收未通过，训练与新方案实验尚未完成。** 这是资产缺失、隔离冲突及执行证据缺失共同造成的阻塞，不能只解释为代码故障，也不能只补回权重就放行。

机器可读证据见 [received_capture_verification_20261009.json](received_capture_verification_20261009.json)。本轮仅本机读取交接包、修改审计工具及执行 CPU 回归，没有连接服务器、运行训练或全量生产，没有更改原始报告、数据、split、标签或合同。新增22项针对性测试，相关完整回归 **1257项通过**（103.91秒）；回归不代替服务器真实回放复用和GPU验收。

## 已核实的证据与边界

- 交接清单中的 **1216 个文件**逐文件大小/SHA及全集核验通过。解压后文件总字节数 **42,161,130**，不与 ZIP 的 30 MB 限额冲突；未收到原 ZIP，无法确认其压缩后大小。
- 包内 receipt 声明 **1086 个原捕获文件**；源码清单为 **1209 个逻辑文件**。源码按内容去重，且打包时改变目录布局，三个数量不能互相代替。已核对包含在交接中的报告、源码和 JSON 资产与原回执绑定；没有声称验证未携带的原捕获 blob。
- 从冻结的 44 份曝光/预约来源重新计算，用途矩阵、路线用途及冲突成员与原报告一致。原服务器 RGB 根目录本机不可访问，复算会将两份需要路线目录检查的来源降为未认证；保留这种状态差异，不把声明关系复算当作实时 RGB 身份复验。
- **44 项外部来源引用 + 11 项基座资产 = 55**，与远程摘要数量一致。但包中没有保存那次 `verify-inputs` 的结果，也没有基座权重，本机不能证明服务器原件现在未漂移。后续应保留该命令输出。
- 服务器 HEAD 是 `8ea0fbb6f434be503b92f42541bbf4f2987eb5f0`，工作树包含未跟踪的核心模块。与本轮修改前的 `0279ac735779b582e1f84bc3b3ccd424c57121ac` 比较，**1170 个共有源码文件 SHA 全相同**，另有39个服务器独有文件；不能仅凭旧 HEAD 判断执行了旧代码，也不能称服务器 Git checkout 与 main 一致。后续同步应保留本地文件，核对实际源码；不要 reset/clean 或把旧历史合并回 main。

## 3 对冲突实际涉及 10 个路线组

均是 `phase3:development_exposure` 与 Phase4 以下用途的 `exposed_holdout_overlap`。这是开发曝光证据；缺少训练池，不能直接声称实际训练泄漏或量化其影响。

| Phase4 用途 | 物理路线组 |
|---|---|
| dev_val | `InvadingTurn/Town12_4195_0`、`InvadingTurn/Town13_1337_0` |
| final_test | `ConstructionObstacleTwoWays/Town12_route_001128`、`CrossJunctionDefectTrafficLight/Town03_route_002138`、`CrossJunctionDefectTrafficLight/Town05_route_002070`、`HardBreakRoute/Town12_2325_0`、`HardBreakRoute/Town12_767_0`、`HardBreakRoute/Town13_1196_0` |
| independent_review | `DynamicObjectCrossing/Town03_Town03_Scenario3_21`、`StaticCutIn/Town13_53_0` |

旧 split 和回执保持历史事实。这10组不能直接充当新联合方案的独立评测/盲审样本；旧集合可作为明确标注曝光限制的回归对照。新独立候选还须与补齐后的三任务完整池和全部曝光记录求交，不能仅排除这10组就宣布独立性通过。本轮没有重划分或自动授予训练用途。

## 缺口和实际进度

| 项目 | 当前状态 |
|---|---|
| 诊断报告、交接文件完整性 | 已完成本次核验 |
| Phase3 索引及选定 adapter | 缺失；无权重副本时重训只能产生新基线 |
| Action 原始三 split、有效池 | 缺失；先补数据与导出合同，不需要先训练 Action |
| Phase4 data2/data4 | 两份 manifest 均缺失，配对 invalid；不能据目录文件数认定完整 |
| 全池覆盖与跨任务隔离 | 9项完整池缺失，3对冲突；未通过 |
| 同 checkpoint 完整评测、固定更新数值性能、四 rank 内核探针 | not_run |
| E0/E1/G1/G2 | not_run；目前审计入口没有实现并运行这些实验 |

环境记录显示 FLA、causal-conv1d 未安装，flash-attn 有安装记录。安装状态不证明实际内核绑定，四卡 forward/backward/generate 和性能仍需服务器实测；本轮不据此指导盲目升级依赖。

## 下一步先做只读复用检查

新增 `recovery-check` 调用原生请求合同、候选池、老师 registry 和全部路线/逐帧回执核验，不触发 `prepare`、回放、编译、重命名或删除。存在原生构建锁时持共享锁，活动 builder 会被拒绝；无锁时须先停止写入者。只写到显式指定、原题库之外的新 JSON。

源码表明：原生续建在 `data2/manifest.json` 缺失时，会把残缺目录重命名保留，再重新编译；每份新题库还会复制完整 production。按用户报告的18 GB回放估算，两份题库仅这部分就会额外写约36 GB，现有13 GB残片仍占盘。该数是粗估，不能据77 GB空闲保证足够，训练输出尚未计入。工具以索引实际引用文件大小计算复制下限，`sufficient_for_build` 固定为未知，不把下限伪装成总预算。

更新审计代码后，在服务器 `AutoMoT/` 执行（输出名须尚未存在）：

```bash
bash qwen3vl_local/audit_joint/run_guarded.sh \
  --space-path checkpoints --min-free-gib 10 -- \
  python -m qwen3vl_local.audit_joint recovery-check \
  --data-dir checkpoints/joint_remote_20261008_211924/phase4_data \
  --data-root lead_data \
  --output checkpoints/phase4_recovery_check_20261009.json
```

返回0只表示旧回放通过当前合同复用检查；返回2表示诊断中仍有阻塞。两者都不代表完整题库可训或 G0 通过。检查会读完回放证据，运行时间受18 GB文件的 I/O/JSON解析影响，本机没有服务器吞吐数据，暂不承诺时长。把这一份小 JSON 交回即可，先不再生成全量题库或重复 G0。

随后按检查结果确定复用/补建和空间预算；并行整理 Phase3 索引、Action 原始三 split 的可重建输入与历史 SHA，再审查完整池隔离。必要时登记新的独立预约/实验版本。之后才启动新 Phase3 adapter 训练、完整评测和四卡性能验收；无法恢复旧权重时不声称完成旧 checkpoint 复现。
