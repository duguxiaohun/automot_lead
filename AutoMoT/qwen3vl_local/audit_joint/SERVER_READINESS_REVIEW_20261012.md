# 2026-10-12 服务器交接复核与开训收尾

**用户确认同步语义后的修正：** sync_automot.sh 从正确仓库克隆，再以 `rsync -a --checksum`（或cp）覆盖更新白名单，保留本地其它文件；本包440文件一致已经证明这部分同步成功。6个额外文件的存在不是同步失败，也不能据此认定生产源码混版。此前readiness2将额外文件一概阻断，和实际覆盖同步方式不匹配；本轮修正审计入口，不要求改同步脚本、新建源码目录或删除文件。

readiness3仍逐文件核验发布SHA，但额外文件仅登记；CPU回归仅执行清单中的已发布测试，禁用外部pytest配置/conftest/自动插件扩展，并在审计进程中拒绝受检包导入清单外模块或来自另一目录的模块。显式实际依赖错误仍失败，不忽略失败的已发布测试。校验范围仍不是所有外部依赖或子进程的完整认证。当前正式池、存储预算和GPU新基线仍待收尾。

本轮readiness3实测：15项专项通过；隔离副本保留6个服务器同名额外文件及一个conftest哨兵，实际发布测试入口 **2264通过、1项本地历史collection依赖跳过，101.76秒，退出0**。哨兵均保留，内容为模拟异常而非服务器原文件；清单外.py/.pyc实际导入、另一目录模块、发布SHA改变均有失败反例。新包26,892字节/6成员通过字节核验（本机未执行模型preflight）；验证见 [readiness3证据](overlay_readiness_verification_20261012.json)。生产v50合同SHA与上轮原生回放相同。

## 已独立核验的服务器证据

- ZIP 51,716字节、7成员全集/大小/SHA通过；外层SHA `4db82822c77d7c33dfd915fb1484296c18118a5afd42429a4d57afcd609096f8` 与package.json一致。
- 440个文件的expected/actual与本机发布清单逐项相同；清单SHA `0d02016ebf06d7396d829d17acf9cd33d352e42905f54c173dad89d3fbf97a53`。
- `source_preflight_exit_code=2`、`pytest_started=false`、`pytest_exit_code=2`；不能记为服务器测试失败2260项或服务器通过。
- `preflight_exit_code=0`，Qwen3.5/Transformers 5.3.0，两权重分片存在；未GPU加载或验权重内容。
- 6个额外文件位于Phase3目录：`capacity_diagnostic.py`、`invalid_capacity.py`、`test_capacity_diagnostic.py`、`test_invalid_capacity.py`、`test_training_pool_audit.py`、`test_v24_near_stop.py`。本机Git已发布版本不跟踪这些路径；未收到正文，不能判断它们来自哪个版本或应否永久删除。
- 29条资产盘点，无错误/截断；仍只有v42题库的未认证元数据，未发现v50正式库/规则参考包或Phase3正式索引。目录链接不递归，这不是全服务器资产不存在的证明。
- 可用49,083,883,520字节，即45.713 GiB；保留20 GiB后25.713 GiB。旧规模新回放加复制式RGB4库约39.132 GiB，差约13.419 GiB，尚未计规则参考和权重/优化器。无需再交相同磁盘盘点。

原包和日志完整留在 `checkpoints/server_readiness_received_20261012/`。Downloads原件未修改。

## 此前隔离诊断（保留证据，不再要求服务器采用此部署方式）

440文件清单是版本核对范围，不是独立依赖全集。实际复制仅440文件到隔离目录后，CPU收集有27处错误，缺Action公共工具、mrope_utils、sft_v3、sft_new_loop_phase2、sft_v2.prompts和keyframe_filter等依赖。因此不能按这份清单裁剪源码后当完整发布。

改用Git提交 `93f05dd7a786af45c9c6331df94446ebdbe6892c` 的完整已跟踪快照，source_check通过且没有额外Python。但新目录缺lead_data时，真实回归67失败、2135通过、54跳过、4错误，均应保留为失败证据；必须显式链接现有原始数据，再做完整回归。旧目录数据可访问不意味着新源码目录自动有数据。这是上轮简略同步步骤需要补全的地方，不用复制RGB或删除旧实验。

最终连接数据后：**2259项通过、1项跳过，101.52秒，退出0**。跳过项为test_native_and_cache_paths_exclude_only_evidence_affected_candidate，依赖本机checkpoints下未发布的历史selected_collection；没有跳过失败项来凑通过，也未把前两次失败记为通过。完整收集仍为2260项。机器摘要见 [verification](server_readiness_received_verification_20261012.json)。

## 训练、提示词与十事件标定结论

| 项目 | 当前结论/必须收尾的内容 |
| --- | --- |
| Phase3训练 | 正式launcher默认每事件1024；缺事件失败，INVALID单列。需要当前mapping/variant正式frame_index，不能用未发现索引的状态直接开训。 |
| Phase3评测 | 实际val/eval消费去重后的固定无放回十事件等量清单；binary/choice均覆盖。不同题等额不等于不同物理组等额。 |
| Phase4训练 | 原生/短视图明确十事件全集，默认phase3_balanced预算10240；事件之间各1024，不承诺YES:NO等量或单题不重复。 |
| Phase4评测/选优 | 显式rule_consistency + teacher-references接通固定val/test清单，物理组隔离，缺事件提前拒绝；自然分布评测仅另报。 |
| 全自动标定 | 不必人工审核全部题；用原生弱规则registry、自动弃权/隔离和规则留出。规则自一致分数不认证老师绝对正确率。 |
| 提示词 | 当前v12已区分上下文与证据、运动与许可、源消失与清空、几何完成与稳定进展；无新证据要求继续追加文字或调整阈值。 |
| 存储实现 | dataset.py仍逐文件copyfile整份production到RGB4库。只省data2仍不足；需新产物单份回放引用/压缩方案及完整峰值预算，当前未实现，不先跑到守护中断。 |

标定逐事件的收尾项来自已绑定开发RGB与原生回放，而非本包新的RGB：

| 事件 | 收尾重点 |
| --- | --- |
| U-E1 | 正常跟车起步/恢复正例偏少；弯道参考系已修，当前应核对完整池运动/静止及恢复支持。 |
| U-E2 | 返回边监督在所查池缺失；实际路线存在导航耗尽截止，不能将驶过障碍直接补return YES。 |
| U-E3 | 所查自动池proceed正例缺、题少；固定走廊/身份门防误建后需看全源召回，不靠重复凑监督。 |
| U-E4 | 行人放行已有自动支持，仍需骑行跟随/绕行及完成分支统计。 |
| U-E5 | 所查实例目标出视野/到后方后截止，不等于放行或完成；全源核对proceed支持。 |
| U-E6 | 所查实例优先权否决后出路口截止；自车运动不是许可，需完整池proceed支持。 |
| U-E7 | 已在额外14训练路线中编译出13个complete YES物理组，旧47路线零正例不代表全源；这些仍未进入正式新库，浓雾/短输入可观察性保留限制。 |
| R-E2 | 绕出/完成主边已有支持；导航变道不自动证明此前有绕障事件。 |
| R-E3 | 多段中间/最终完成边界、完成反例仍需完整池分报。 |
| R-E5 | STOP历史、优先权、完成与环岛分支分别核对；已知路口历史歧义已自动隔离。 |

本轮没有新增RGB目视/原生标定回放，未确认新的二元错标，也未修改生产/标签/提示词。既有局部题数不能冒充v50全池统计。

## 最短开训路径与停止扩展审计的边界

1. 沿用既有sync脚本覆盖同步readiness3，再在原AutoMoT目录执行发布测试检查。旧环境和lead_data保持，6个额外文件保留。
2. 同盘保留旧实验，先完成新产物减少production复制的存储实现及预算；然后用当前规则生成Phase3正式索引、Phase4完整候选回放、单份RGB4库/短视图和自动规则留出。现有B段模板缺新版生产资产，不能直接代入旧v42路径。
3. 对正式池执行原生准入与CPU采样：train/val/test十事件齐全，实际呈现及不同评测题事件等量；上述关键边分报YES/NO题数和物理组、UNKNOWN/隔离/截止原因，缺支持仅定向补来源。
4. 通过后新合同两图/四图各一轮、保存重载和完整val，确认正常再继续后续轮数。无需等待全部G2人工逐题审查、E1–E6优化或CARLA；这些未完成项不冒充独立准确率认证。

当前仍不能宣布全流程已打通，但也没有理由为服务器混版继续重写标定和提示词，或重跑已交付的旧U-E1/U-E7拟合。

## 服务器下一步：沿用现有同步与原目录

本地readiness3验证完成并随白名单发布后，用原sync_automot.sh照常同步。无需改脚本或删额外文件；不要在新版尚未同步时重复旧readiness2命令。进入现有AutoMoT目录、激活automot，执行：

```bash
cd /home/cruser1/lda/AutoMoT
conda activate automot
PYTHONPATH=. \
MODEL_DIR=/home/cruser1/lda/AutoMoT/checkpoints/Qwen3.5-4B \
CHECKPOINTS_ROOT=/home/cruser1/lda/AutoMoT/checkpoints \
  bash qwen3vl_local/audit_joint/server_readiness.sh
```

交回打印的report.zip（≤30MB）和package.json，仍不建库/训练。源码SHA不一致、已发布测试失败或实际导入清单外模块时均如实交包；额外文件名单不再单独构成失败。无需重复空间盘点或旧模型拟合。脚本继续输出独立时间戳报告，不覆盖旧报告。
