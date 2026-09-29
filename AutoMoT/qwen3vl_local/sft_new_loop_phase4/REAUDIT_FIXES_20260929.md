# Phase4：第二轮部分审计修复

根据用户提供的[第二轮只读审计](/tmp/phase4_reaudit_20260929/AUDIT_PARTIAL.md)处理四个可复现问题和UE3恢复提示词覆盖问题。完整审计仍未完成：报告两轮去重累计121条完整序列、10,900帧，278单元中37个达到三个ID，计划还剩579条，18单元来源不足三个ID。这些是报告的累计计数，不是本次修复新增审阅量。

本次实际复看该报告StaticCutIn/Town13三个ID（389、390、391）的72–95帧，共72张已审图。可见警车/红车/紫车斜停后，自车从其旁侧通过；390、391还有对向车辆在等待期间通过。仅用来核对提示词覆盖，不批准新的YES区间、不增加完整序列计数、不把旁侧空隙等同于可立即通行。

## 修复

| 问题 | 修后行为与回归 |
|---|---|
| 横向冲突丢失STOP且无法复核 | 纯汇总和export保留STOP/DECELERATE，并输出conflict_instances与recheck_instances。runtime.aggregate把冲突落实到对应实例的needs_recheck，公开revalidate可处理；仅修正一个实例时，另一待复核实例不会被自动放行。错误目标即使verified=True也仍会重新报冲突。 |
| 冲突涉及未执行许可 | runtime取消这些待执行许可，保留独立确认事实及原有制动约束；纯export不改状态，但在拒绝冲突许可时同样保留rollback中仍有效的制动约束。冲突实例不能通过acknowledge确认过期许可。 |
| re_yield与stable同时YES变KEEP | 对限制边与stable的矛盾在更新前原子拒绝，进入RECHECK，不接受其中任一状态变化、不输出KEEP。单独re_yield仍减速，re_yield+hold仍STOP；几何完成+restrict保持合法。 |
| 返回完成后目标错误 | current_target_corridor根据实际流程区分：需返回的绕行在RETURN及DONE后都使用return_corridor；不返回、单次变道、汇入和跟随分支仍使用各自target_corridor。覆盖STOP/DECELERATE/RESUME、序列化恢复和最终稳定。 |
| DONE文字暗含整体退出 | DONE改为事件阶段完成。几何complete候选明确“进入/返回既定路线走廊已完成，剩余纵向限制继续生效”；系统文字也区分开始机动的许可与已完成几何事实。仍只有当前状态、候选下一状态和YES/NO问题，不增加流程状态。 |
| UE3仅写跟车恢复 | proceed和release共用恢复描述：侵入不再妨碍既定走廊，既可形成可跟随间距，也可有足够空间从侵入车旁通过；继续横移、冲突来车、通道占用与优先权仍须检查。不要求侵入车必须摆正，不硬编码从左通过，不自动创建新的绕行车道或方向先验。 |

临时UNKNOWN现在汇总为UNKNOWN；尚未达到复核条件时不再声称有一个无对应实例的RECHECK。已有STOP/DECELERATE仍保留。aggregate由runtime持久化冲突；action_prior.export保持纯函数，外部消费者可依据其同样的待复核实例列表调用公开复核接口。

## 证据与隔离

只读报告的23个新曝光物理组此前只存在于临时审计文件，未进入项目隔离集合。已逐项核对它们都有visual_notes记录，登记为Phase4 train-only，记录在reaudit_exposure_20260929.json，包含来源文件SHA与本次三张连续面板的SHA。该登记不添加监督题、不把剩余待审路线登记为已审，也不修改Phase3/Action稳定默认。

原reviewed_state_pairs_v3.json及rgb_boundary_review_v3.json内容SHA未变。默认题数仍为429道二元开发题、121条待审；公共纵向边、re_yield、路线分支及独立val/test支持缺额仍在，ready=false。UE3旁侧通过的完整训练支持尚待逐帧条件标定，不因提示词改宽而声称标注已完整。

## 验证和版本

172项专项通过，包含真实小型Qwen3.5的2/4图答案监督与LoRA反向；新增覆盖冲突汇总/export、公开tick→快照→revalidate、未执行许可取消、制动保留、返回目标、完成文字及UE3双恢复路径。原probes.py和runtime_probe.py原样重跑：汇总STOP、recheck_instances=[a,b]且无revalidation_error；re_yield+stable为RECHECK而非KEEP；返回目标为original_lane。

2/4图重新构建及所有历史图像哈希检查通过，各7轮×world1/4采样回放通过。产物和修后复现输出在probe_output/reaudit_fixes_v5/，未覆盖只读报告或原始复现结果。没有真实4B/GPU/CARLA验收，也没有新训练结果或UE3误答率改善证据。

任务/提示词升级phase4_state_pair_binary_v5，快照升级phase4_loop_v5，先验导出source升级phase4_state_transition_v2；train.sh默认data_v5，人工标注仍用v3。需新数据、新run，旧adapter/快照使用原版本，不硬复用。此前文档中的v4、138项及7/278覆盖是历史记录，以本说明和README的当前状态为准。
