# Phase4：只读审计后的控制器与合同修复

依据用户提供的[部分只读审计](/tmp/phase4_readonly_audit_20260929/AUDIT_PARTIAL.md)修复 F1–F7。该报告的覆盖状态仍有效：已看38条完整序列、4290帧；278个事件×场景×Town单元仅7个达到三ID，662条计划序列/94378帧未看，18个单元来源不足三ID。**本轮只修代码并做回归，没有新增目视审计，没有修改标注、RGB边界或曝光名单，不构成全场景验收。** 来源进度见[coverage_progress.csv](/tmp/phase4_readonly_audit_20260929/coverage_progress.csv)。既有局部窗审阅数量与完整序列计数不可相加冒充独立新帧。

## 修复及对应验证

| 问题 | 当前行为 | 验证 |
|---|---|---|
| F1 公共纵向边遗漏 | 每个事件/分支/返回模式、三split检查可达COMMON边以及事件边的readiness YES/NO；catchup不能填缺额。新增missing_longitudinal_edges。 | 原事件边全覆盖合成池不再ready；补全COMMON的合成池通过；删stable正例、UE5 re_yield或cyclist_follow的hold均拒绝。合成通过不代表真实数据可训练。 |
| F2 未执行许可回退 | 每次机会记录自身改动的rollback字段；独立确认的里程碑不回退。确认与待执行机会分开。 | return+stable之后未回执，回到PASS/STABLE；快照恢复后相同；多次questions不重复过期。 |
| F3 横向完成与减速 | 几何完成边可与纵向限制同时成立。事件轴DONE后横向KEEP；APPROACH/HOLD/RECOVER继续回答纵向问题。整体退出看finished/instance_complete，不看state字符串。 | RE2/RE3/UE2/骑行绕行分别验证complete+restrict，以及停止→恢复→稳定后才退出；同一事件轴complete+re_yield仍拒绝。 |
| F4 正常等待误停滞 | 完整明确NO记confirmed_no_transition，可继续等待。分别记录uncertain_observation、permission_not_executed；执行/跟踪异常由显式progress_fault报告。复核保留已存在的STOP/DECELERATE约束，不保留未经确认的横向许可。 | 200次正常NO保持ACTIVE/STOP；缺回执、不确定、外部故障各自触发复核，STOP不丢失。 |
| F5 回放虚假零延迟 | 分别输出answer_delay_frames、accepted_delay_frames、execution_delay_frames；未执行的成立区间保留右删失记录。delay_frames现在仅为执行延迟的兼容别名，missed_ready保留答案漏答含义并另列missed_execution。 | 三次YES无回执：回答延迟[0]、接受延迟[0]、执行延迟[]，未确认1条。第三帧有因果执行回执：执行延迟[2]。冲突拒绝不计接受/执行。 |
| F6 交接缺有效性判断 | 有事件/Phase2判断/导航事件时要求Phase1八个布尔判断与Phase2五个布尔判断齐全，INVALID_EVENT_CONTEXT必须明确False。缺失不等于False。 | 缺有效性、缺UE6、缺Phase1可见事件项均拒绝；全量明确判断可建立事件。 |
| F7 观察模式未绑定 | manifest/run/adapter存明确observation_contract；2图[-4,0]、4图[-6,-4,-2,0]，4Hz、因果顺序及最小帧号一起核对。模型encode、实时tick、bundle预测及序列入口均校验。 | 两方向跨模式/错误间隔在调用模型前拒绝；缺adapter观察合同在加载权重前拒绝；真实小型网络2/4图监督及LoRA反向保留。未真实4B跨模式试验。 |

原审计`probes.py`和`extra_probes.py`已原样重跑，修后结果在`/tmp/phase4_fixed_probes.json`及`/tmp/phase4_fixed_extra_probes.json`，未覆盖只读报告与原复现结果。专项138项通过。

## 事件完成与实例退出

`state == 'DONE'`表示事件轴或横向机动已完成。仅在纵向为STABLE/FOLLOW时，`Episode.finished`及输出`instance_complete`才为true。APPROACH/HOLD/RECOVER仍输出有效纵向先验，并由同一实例处理settle/hold/release/stable等问题。外部消费者必须依据`instance_complete`或整体status，而不能看到state=DONE就清空约束。无需新增一个模型状态或把全流程给模型。

普通长等待没有“经过120帧就异常”的规则。不确定观测及连续未执行许可仍各有计数阈值，需要复核时仅保留已有的制动约束；没有把所有未知自动变成STOP。执行器已收到命令但实际没有进展，须由执行器/因果跟踪显式提供progress_fault；本代码不凭模型连续NO或未来轨迹推断车辆卡死。

## 已开始动作的旧状态同步

首选`Phase4Loop.tick(..., execution_receipts=[receipt])`，或`Episode.confirm_execution(receipt)`。回执包括：

```json
{
  "instance_id": "tracked-obstacle-1",
  "decision_frame": 112,
  "observed_frame": 112,
  "started_frame": 110,
  "edges": ["depart"],
  "source": "causal_tracker",
  "evidence_id": "track-observation-112",
  "successor_observed": true
}
```

112帧仍以旧WAIT状态问depart；跟踪器在112帧实际观察到动作已从110开始，就可确认112帧的当前待执行边，而不是再发送一个横移命令。instance、当前decision、当前observed_frame与完整边集合必须匹配；未来观测、其他实例/边、重复回执、已过期decision拒绝。旧回执迟到时重新观测并绑定当前decision，不能恢复过期许可。`source`仅接受executor/causal_tracker，不能以模型YES替代证据。

已有同步执行端仍可在当前决定后立即调用`acknowledge(instance_id, frame)`；此低层接口要求调用者自己确认实际执行。可序列化快照版本为phase4_loop_v4，含观察合同与待确认记录，拒绝旧格式。已用合成执行观测验收丢回执→恢复快照→旧状态补问→跟踪确认；**没有完成CARLA或真实跟踪器集成验收**。

## 流水线与版本

`run_full_pipeline.sh`现在依次构建、严格预检/训练、读取完整val选出的best、独立test生成评估、打包audit.zip。test不参与选优，审计包包括test/metrics.json和test/cases.jsonl。训练/评估沿用GPU选择，单独train.sh默认目录改为data_v4。MODE=check只构建/回放采样，不假装已经训练或评估。

流程接线用进程边界stub测试，审计包包含性及符号链接拒绝用真实文件测试；缺模型/三split时仍拒绝正式流水线。Phase3的原始证据与所有历史实验开关尚未逐项验收，不能据此宣称功能完全对齐。

任务/源码合同升级phase4_state_pair_binary_v4；提示词仍为v3状态对YES/NO，默认人工标注仍是reviewed_state_pairs_v3.json，标注与证据SHA保持不变。必须构建新数据、新run/adapter；旧数据/adapter用原源码，旧快照不自动迁移。Phase3/Action稳定默认未改。

2/4图重新构建均为429道开发二元题、121条待审，均train-only；新校验明确列出270个公共纵向边×分支×split缺额，还包含事件边/分支/独立holdout缺额。全量历史RGB哈希及各7轮×world1/4采样回放通过，产物位于probe_output/audit_fixes_v4/。这只是现有开发题工程验证；没有添加COMMON或re_yield标注、没有补齐独立三split/分支支持、没有正式训练或完整4B/GPU/DDP验收。ready=false。
