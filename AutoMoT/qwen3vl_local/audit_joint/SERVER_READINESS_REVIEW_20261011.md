# v50服务器交接包复核与最短开训收尾

本轮接收 `/home/codon/Downloads/report.zip` 和 `package.json`。结论：发布清单内源码及模型环境预检通过；服务器CPU回归没有执行完成，运行目录混入不兼容的清单外测试。盘点仅发现旧v42正式题库，没有确认v50正式池或Phase3正式索引。新版三处生产接线已完成，不再列为待修生产缺陷。

## 1. 独立复核结果

| 项目 | 本轮核验 |
| --- | --- |
| ZIP | 28,665字节，小于30MB；SHA `f9e9a460cd5d2bb8ec43b224e743bea0aaa85ff3749ea6e99d1ca566da90b138` 与package.json一致 |
| 包内回执 | 7个成员的全集、大小、SHA通过；没有执行包内程序 |
| 发布源 | 440个文件与本机逐文件一致，发布清单本身SHA也一致；不代表服务器没有其它文件 |
| 服务器pytest | 退出2，3个收集错误，尚未进入测试执行；不能写成2256项服务器通过 |
| 服务器preflight | 退出0；Qwen3.5-4B配置、vendor及环境检查通过，Transformers 5.3.0；两份权重分片存在，不等于权重内容SHA或GPU加载通过 |
| 本机本轮回归 | 2256项通过，104.13秒，退出0；与服务器失败分开记录 |
| 资产盘点 | 29条，无截断/扫描错误；目录软链接不递归，元数据未经原生准入认证 |
| 可用空间 | 38,898,876,416字节，约36.23 GiB；保留20 GiB后仅约16.23 GiB增长空间 |

原包、解出的7份证据、逐项verification和本机完整测试日志保留在 `checkpoints/server_readiness_received_20261011/`，原Downloads文件未修改。

## 2. 需要解决的问题

### P1：服务器混入不兼容的清单外测试

三个报错文件均不在本次440文件发布清单：

- `sft_new_loop_phase3/test_capacity_diagnostic.py`：导入 `_MinCostFlow` 失败。
- `sft_new_loop_phase3/test_invalid_capacity.py`：导入 `_MinCostFlow` 失败。
- `sft_new_loop_phase3/test_v24_near_stop.py`：导入 `near_stop_review` 失败。

`sampling.py` 与 `action_review.py` 等清单内文件已与本机核对一致。包里未提供上述额外文件正文，不能指定它们来自哪个提交；能确认的是目录包含本次发布以外的不兼容测试。这解释了“source_status=matched而pytest失败”，不是模型环境preflight失败，也不是标定准确率失败。

处理：使用现有sync流程将本次源码分发到**此前不存在的独立目录**，旧目录、数据和权重保持。不要覆盖旧目录后认为已隔离，不要删除旧实验，也不要给当前生产代码添加兼容函数来迎合未发布测试。只跳过这三项也不能证明没有其它混入文件。

审计入口值得改进：`server_readiness.sh` 当前先递归收集pytest，打包时才检查发布SHA；`server_readiness.source_check`只检查清单成员。后续应在测试前核对源码，并显式列出运行包目录内额外的Python/test/conftest文件及检查范围，把“清单内匹配”和“测试来源完整”分开。此次按审计请求只报告，不修改已发布清单或脚本，避免服务器新旧证据混用。

### P1：尚无可验收的当前正式池

报告中的正式题库位于 `checkpoints/joint_remote_20261008_211924/phase4_data/`，两模式manifest均声明 **v42**，train330902/val273/test186；旧短视图声明train330902/val239/test123。这些是元数据声明，包没有索引行，不能重新计算十事件分布、条件支持或数据独立性。旧RGB2原生偏移为[-4,0]，也不能当新版短两图[-2,0]。

未扫描到v50题库、自动规则留出包或Phase3正式frame_index。目录软链接与其它挂载根未扫描，结论是“本次盘点未发现”，不是资产一定丢失。旧v42题库/adapter保留原合同；不得手改SHA交给v50继续训练。

`candidates.json`约38.7MB，production index约14.4MB，均超过2MB元数据读取上限；因此本包还不能确定正式候选路线范围。原始 `lead_data` 的真实位置和可访问性也未检查。下一步需要真实数据路径与新版正式候选范围，不能把本机47条或14条开发路线当全源。

### 建库前先解决空间预算

本包没有production目录总占用，新版全量输出大小未知。当前仅16.23 GiB的可增长余量，不能保证能同时容纳新回放、一份RGB4编译副本、索引和后续权重。也不能据此断言新版一定放不下。先使用既有storage-inventory盘点或确认其它磁盘位置；只建一份RGB4再生成小视图的方案保持。不得为腾空间自动删除旧data2/data4或实验，也不要先运行到磁盘守护中断再补预算。

## 3. v50生产接线复核：上轮三项已解决

核对生产调用及本机完整回归：

- Phase3实际val与eval先生成不同题全集，再调用 `evaluation_balance.select_work`；INVALID另列。Phase4训练入口显式传十事件全集；`evaluation_policy.prepare/score_cases` 的固定无放回清单用于实际val/test及选优，缺事件提前失败。目标仍是事件之间1∶1，不要求YES/NO等量。
- `--selection-mode rule_consistency --teacher-references PATH` 已接到原生和短视图训练器，只使用规则训练题和隔离的规则留出；评分为空不再当作有效选优。训练不需要人工逐题批准全部样本。这里报告的是规则一致率，不能拿同一老师自证标定正确率。
- `history_admission.quarantine` 已在编译及自动留出导出中先于UNKNOWN过滤执行；已知路口停车历史歧义会自动隔离，未改写YES/NO或合并不同目标。

这些结论是源码和CPU回归证据，不是Qwen四卡实际训练或正式全池验收。当前不需要重写这些模块或继续堆提示词。

## 4. 十事件标定与可开训范围

本包没有新的RGB或原始标定结果，本轮没有新增目视或回放，因此不能从preflight通过推出十事件标定全正确。沿用已绑定的v50开发证据：历史歧义已隔离；额外14路线中已得到U-E7 complete/readiness YES各模式13条、13物理组，尚未合入正式新库。不能继续把“旧47路线池complete零正例”当成全源无法生成。

正式自动池重点核对U-E3/5/6 proceed YES、U-E2 return及U-E7 complete支持，并报告每事件×边×阶段×答案的不同题数、物理组数。其它事件沿用 [开训报告§3](TRAINING_SCOPE_20261011.md#3-每事件自动规则的剩余问题) 的具体检查项，但其数值仅代表旧局部池，不是v50全源统计。零支持先查来源覆盖，不靠修改阈值或重复帧凑数。

雾天、短两图之外的停车历史和稳定速度证据仍是可观察性限制；无新增确定错标证据时不因本次服务器导入错误修改几何/提示词。自动准入、弃权和风险隔离可以代替全量人工逐题处理，但现有证据不支持“所有事件无标定问题”的保证。

## 5. 服务器下一步：只补一次真正有效的CPU回归与资产位置

先通过已有 `sync_automot.sh` 将同一发布版同步到此前不存在的目录。参数沿用服务器已验证方式，不在本文猜测。进入新应用目录、激活automot后，可直接读取旧目录里的模型和资产：

```bash
MODEL_DIR=/home/cruser1/lda/AutoMoT/checkpoints/Qwen3.5-4B \
CHECKPOINTS_ROOT=/home/cruser1/lda/AutoMoT/checkpoints \
  bash qwen3vl_local/audit_joint/server_readiness.sh
```

上述两个绝对路径来自本次包内回执，路径未迁移时可以原样使用。输出仍写新应用目录的全新时间戳目录，不修改旧实验。交回新report.zip与package.json，同时提供可用的新数据输出根以及 `lead_data` 的真实位置/可访问性；没有其它磁盘就如实说明，先做现有盘空间预算。无需重跑旧U-E1/U-E7拟合，不安装新模型依赖，不启动GPU。

CPU回归通过且数据/空间明确后，进入 [NEXT_TESTS B–D](NEXT_TESTS_20261011.md)：按v50合同完成正式候选回放、单份RGB4自动题库和规则留出，先核对十事件及关键支持，再两模式各完整一轮＋保存重载，之后正常后续轮数。B段仍要求已经存在匹配v50的生产资产；本次只有v42资产，不能直接代入。无需等待全部E1–E6、缓存优化、CARLA或全量人工审计。

本轮未修改生产/标定/提示词/训练代码，未连接服务器、训练、清理或push。

## 6. 用户补充后的同盘执行约束

用户确认没有其它输出目录，原始数据仍可通过 `lead_data` 访问。后续按同盘、保留全部旧实验预算，不再等待其它磁盘，也不把降低20 GiB余量作为默认解决办法。

已复核 `dataset.build`：即便只编译RGB4，仍会逐文件复制一份production；短视图本身不再复制production。因此新建峰值至少包含新回放、RGB4中的回放副本、训练索引与自动留出，训练阶段还需权重/优化器。旧库占用只能作为估算参考，不能直接当v50精确大小。

服务器修复源码目录混版、重跑CPU回归时，同时执行下面一次只读盘点，将终端JSON结果交回即可。无需再次回放或推理：

```bash
python -m qwen3vl_local.audit_joint storage-inventory \
  --root /home/cruser1/lda/AutoMoT/checkpoints/joint_remote_20261008_211924/phase4_data
```

它读取文件元数据，分报full_production/data2/data4实际分配占用和当前余量；不读取索引正文、不删除文件。若预算不满足，先评估减少新生产副本的实现，不能直接启动现有复制式全量流水线，也不牺牲十事件覆盖来适配空间。新的源码目录还须保留可用的 `lead_data` 引用；用户确认的是现有访问条件，不等于新目录已自动建立链接。
