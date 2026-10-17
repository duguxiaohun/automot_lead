# cr02与cee独立资产；cr02标定包审计（2026-10-17）

## 结论

本次cr02建库/自动CPU准入已返回成功，最终交接缺完整准入报告；不是又一次标定规则失败。不要重新跑回放、建库、readiness或GPU诊断。立即补交已有automatic_admission.json即可，它14969873字节，小于30MB，不必打ZIP或更新代码。

cr02已收到的日志根目录为/home/cruser1/lda/AutoMoT；本次资产为checkpoints/phase4_labels_20261017_125230_304771113。对应文件：

```text
checkpoints/phase4_labels_20261017_125230_304771113/automatic_admission.json
```

包内记录该文件SHA256为5d76e18cc8586f89ede5a58d5402666bcd9e6a023b567912882cc153c5eacce8；这是服务端诊断记录，尚未收到该文件独立复算。直接下载到本机Downloads，下一轮无需新服务器计算即可核验两模式最终支持、隔离和等量计划。

## 已独立核验与未到达证据

ZIP46888字节、12成员SHA/CRC通过，ZIP SHA73a0fbb709ce5ab829162194adc3a88eb962f7e03dee66f888c90afa61622ddd。558项来源与历史release_sources_v78_worker_diagnostics_20261017.json逐项一致。原生Phase4范围203项来源与当前本机字节一致，未发现这次又使用过时生产规则的证据；学生运行代码后续有升级，不能直接混用旧view/新checkpoint。

- 完整回放：8614路线、1062401帧，complete_causal_production=true；不是所有帧都有可认证标签。
- 自动训练题：298770；独立自动留出候选val27801题/432路线、test26105题/452路线。
- build_result：admitted_existing_data_development，finished_cpu，errors=[]，训练未启动；命令exit0。
- 构建32011.04秒（8小时53分31秒）；日志回放15738.9秒（4小时22分19秒）。其余阶段没有完整细分计时，不能都算作训练或规则计算。
- manifest、生产index、独立留出文件SHA在收到的摘要之间一致；本包只有摘要，没有原始全池文件，不能声称本地重新验证了全量源数据。
- 预算收到时可用30.30GiB，声明需求26.20GiB（含20GiB保留），条件余额4.10GiB。当前余额须正式入口复核，不能把cee的TB级空闲盘套到cr02。

原生data4和view的val/test=0，是独立references4留出设计，不是缺验证/测试。27801/26105是候选数，不是固定等量评测题数。最终automatic_admission.json被漏包，当前无法独立认证cr02十事件最终训练/val/test等量明细。cee已验收的1024/37/18不能直接作为cr02证据；正式实现仍要求十事件等量，YES:NO不要求1:1。

## 本次确定缺陷

server_finish.package的单输入10MB限制拒收14.97MB准入JSON，外层finish_receipt于是显示blocked_or_interrupted，尽管build_status已admitted且exit0。v80修的是epoch交接器，本轮是另一条fullflow交接器同类遗漏。

已修改：automatic_admission单输入64MB、合计未压缩预算128MB、最终实际ZIP仍硬限30000000字节；不删减准入明细。读前后及实际读取字节SHA绑定，输入漂移进入诊断，真实超限拒绝成功包。已有报告支持package --handoff-output写新目录，保留原finish_receipt、旧handoff、数据和权重；既有或悬空目标拒绝覆盖。诊断失败时额外保留execution_status，区分构建执行结果与交接完整性。

合成14.97MB报告在旧函数复现“构建通过却交接阻塞”，新函数完整保留报告且ZIP15577字节。这是压缩性良好的合成回归，不能冒称已经补回服务器缺失原报告。137项相关回归13.97秒通过；最后补充输入漂移诊断后server_finish专项23项5.20秒通过（22项与主组重叠），不是160项独立测试。当前发布清单565项匹配。学生/teacher/采样/提示词/路线缓存适配合同不改。本轮没有GPU、远程操作、新RGB目视或生产改标。

## 标定范围和质量限制

收到的raw train/readiness支持如下，仅作为编译前必要证据，不是最终学生采样统计：UE2 return YES2题/2物理组；UE3 proceed YES281/96组；UE5 proceed YES0；UE6 proceed YES728/99组；UE7 complete YES18/13组；RE5 proceed YES868/187组。UE5仍明确不支持，不以hold/restrict的YES代替，不强行放宽规则硬补。自动规则可以生成数据，不要求逐题人工审核；“开发准入”不等于十事件全部能力或全量RGB均已认证。本包没有新证据要求堆提示词或改阈值。

cr02训练298770与cee298787不同，manifest/生产index SHA也不同。不能仅凭路线总数相同判定为同一库，亦不能在没有逐题差分时认定17题差异是错标。

## 双服务器操作约定

| 机器 | 本轮已知资产/进度 | 当前下一步 |
|---|---|---|
| cr02 | phase4_labels_20261017_125230_304771113；CPU构建成功；未训练 | 补交原有automatic_admission.json，核验后用本机data4/view/references4做两模式首轮 |
| cee | phase4_restart_20261017_162511；RGB2训练/val/reload通过，test故障；未收到新的RGB4结果 | 原有接续流程，仅使用cee已有权重/view/validated_data |

两台源码可分别同步，数据/缓存/权重不随源码同步。AUTOMOT_SERVER=cr02或cee只作为用户声明；新交接以及首轮/续接/多轮workflow记录真实hostname、当前仓库根、逻辑和解析路径。没有基于hostname猜资产或跨服务器默认路径；缺失按当前机器报告，不自动重建或跨节点搬迁。

用户已授权发布v83。按既有sync_automot.sh同步后，在cr02的AutoMoT目录补打完整交接包：

```bash
export AUTOMOT_SERVER=cr02
export P4_DATA=checkpoints/phase4_labels_20261017_125230_304771113
export P4_HANDOFF="$P4_DATA/handoff_repacked_$(date +%Y%m%d_%H%M%S_%N)"
python -m qwen3vl_local.audit_joint.server_finish package \
  --output "$P4_DATA" --exit-code 0 \
  --handoff-output "$P4_HANDOFF"
ls -lh "$P4_HANDOFF"/fullflow*
```

这条命令只读取现有轻量报告并打包，不调用build/prepare/回放/训练。cee使用时必须改为cee本机实际目录和AUTOMOT_SERVER=cee，不能复制cr02资产路径。任何机器上正在训练时不要同步改动正在执行的源码。

交回新P4_HANDOFF目录中的fullflow_report.zip与fullflow_package.json；收到完整准入明细后再确认cr02的十事件最终等量计划。
