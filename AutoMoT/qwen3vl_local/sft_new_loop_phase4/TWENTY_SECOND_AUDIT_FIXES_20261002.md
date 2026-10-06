# v22：第二十二轮生产者复核与修复

审计中的主要缺陷成立。v21的高弃答率主要来自条件/上下文未实现，不能解释成阈值偏保守。此次修复几何与类别错误、接通当帧上下文和新范围参考校准；尚未完成全量可靠条件生产或补齐训练标签，仍不能正式开训。

## 修复及实际证据

1. **同ID位置跳变。** `disappearances`现在将前后帧参与者位置分别通过`ego_matrix`转到世界坐标，用前一帧速度、采样周期和加速度/位置余量判定不合理位移；包含车、自行车、行人、static/static_prop_car。缺位姿或运动证据为未知，禁止安全适配器签发许可；正常自车转弯/平移有反例回归。146_1第18→19帧实测3824位移59.4066m、3823位移59.9126m，两者均触发候选。原先已确认的RGB消失不作为本轮新增两处消失重复计数。阈值是开发期异常筛查参数，不是轨迹物理认证。
2. **参与者类别。** 投影保留base_type/type_id，自行车及摩托车进入UE4，不再作为普通汽车生成UE1/UE3或稳定跟车候选；施工锥桶、拖车及static_prop_car与走廊相交时进入UE2候选，参与异常检测。VehicleTurningRoute第50帧110/148实际归入UE4。
3. **图像质量门槛。** 按三相机内外参将包围体投影到当前RGB，检查像素尺寸、亮度、对比度和饱和比例；缺图像证据、黑暗/过曝/过小/不支持投影返回UNKNOWN。语义像素数和LiDAR点不能单独证明可见。原图复看VT1第50帧后，其三个自行车的可见性建议均为UNKNOWN。亮度和对比度阈值只是开发期筛查；明亮背景可能使空框通过，不能称已经校准出可靠的目标可见性模型。
4. **对向车误候选。** 仅与自车坐标直线带相交仍会误判弯道相邻对向车。UE5自动候选进一步要求同road/lane的车心侵入证据；676_0第57–87帧旧误候选实测为0。该规则可能漏掉车心未越界的压线侵入，不能替代导航提供的弯曲走廊多边形。
5. **条件正例与当帧上下文。** 新增`privileged_context.py`，`--contexts`按路线/实例/帧和完整因果来源SHA绑定实际状态、导航几何及场景证据，逐路线生产真正传入`geometry_context`与`scene_context`。有完整证据时可计算depart/enter/return，以及proceed/hold/restrict/complete的YES；缺通行权、信号故障已建立、边界或可见空间等证据仍未知。测试覆盖七类proceed、出发/进入/返回、约束与完成的正例和缺证据反例。**这证明条件路径可达，不代表已从原始数据自动取得这些上下文。** 导航、通行权、可见空间完整性和真实状态历史的可靠自动生产来源仍缺，当前的几何候选也未通过标签校准。
6. **假设状态不冒充恢复实例。** 自动参与者候选的建议单列`geometric_target`，`target=UNKNOWN`并记录`state_basis=hypothetical_state_pair`。只有当帧来源绑定的状态输入才走正式条件建议入口；所有结果仍`training_target=UNKNOWN`，不直接批准训练。普通稳定跟车不证明曾经经历冲突/恢复。
7. **横向校准可接新参考。** 保留旧范围参考的排除（98个旧readiness参考不能用于新可见范围任务）；新增`--calibrate-references`，接收新范围逐帧RGB真值与独立contexts，支持depart/enter/return，并报告缺上下文弃答、正确/错误和各边分层。参考答案不输入条件计算，错误范围、RGB/当前状态错配、重复参考、留出路线用于开发校准均拒绝。不能将旧标签改个范围名后当新标注。

离线安全适配器复用新的跳变检测，并在自车位姿缺失时拒绝签发许可。它仍要求真实导航包络、参与者清单完整性、静态可行驶空间与通行权证据；本轮未实现这些证据的上游自动来源，也未接CARLA。手工将布尔值填True不属于有效验收。

## 风险与独立抽检

本轮render_index3的九物理组显式train-only，均原默认train，不挪动冻结评估路线。登记五风险记录：146_1原消失的同ID跳变补充、001997未匹配人形/参与者身份、001997末段第二障碍边界、夜雾自行车可见性、T_Junction源RE5边界。

实际新复看原图3张：VT1第50帧，001997第18/27帧。后两张有车旁人形，但对应bboxes无walker/bicycle/motorcycle，不能确认独立行人，也不能用bbox缺失证明不存在参与者；保持身份/渲染待核。其余新增风险来自用户审计报告及定位文件SHA。无新增完整序列目视审计，无新增人工训练标签；继承192/278格满足三ID、约240条剩余，不把文件读取算逐帧目视。

在未曝光的冻结留出路线中，另预留**40条生产者人工抽检路线**（val/test各20，十个预期事件各2条/每split）。只按场景元数据和固定哈希取样，尚未看RGB；场景是抽样线索，不能保证实际事件存在。当前计划`producer_manual_check_plan_v2_20261002.json`绑定修正后源码；v1保留为修正对向误候选前的历史计划，v2保持同40路线且未发生留出图像曝光。数据split保留原归属，后续开发曝光报错；这些路线不能提供模型选优标签，也不能用于开发校准。**这是前瞻抽检预约，不是已经完成的人工真值集或十事件评估覆盖。**

## 验证结果

- **592项专项测试全部通过**，含旧链路、瞬移/坐标补偿/位姿缺失、参与者类别、暗图门槛、弯道对向误候选、真实文件上下文接线、新横向参考校准及独立抽检隔离。
- 重新清点9712路线、1,637,680帧；训练/验证/测试仍8166/765/781。
- 实际扫描12路线1087帧：68个ID缺失、4个同ID跳变候选，共72个；未逐一人工确认。默认没有绑定的实际状态/导航上下文，2462个假设状态候选仍全部UNKNOWN，没有批准自动监督。具体几何建议保留在`geometric_target`，不会填入训练。
- 旧开发参考校准仍为320题中311弃答、8正确、1错误（HardBreakRoute/Town12_3452_0 settle第83帧）。这不是效果改善；新增上下文路径尚缺真实新范围参考的端到端准确率结果。
- 2/4RGB真实`MODE=check`、每模式894监督题和533待审题全部RGB核验通过；train435/val273/test186及review四JSONL与v21逐字节一致。world1/4均340→428→435，第3轮覆盖全部训练题；七轮事件等额/cap8/恢复历史一致性保持。
- 四冻结规则、人工标签、旧holdout、采样/训练/选优代码SHA保持；未GPU、4B、DDP或CARLA验收。

当前训练435题/42路线，训练题覆盖十事件；独立val/test只有UE1/5/6/7。关键depart/enter/return/recover_follow训练题仍0，UE7 proceed readiness YES仍0，RE5仍YES1/NO27。事件等额下小题池重复的过拟合风险成立，未擅自改变采样分布。cap8是每轮同帧限制，不是七轮累计曝光限制。

## 当前可复用路径与运行

最终候选清单：`/tmp/phase4_v22_work/final_candidates.json`；最终数据：`final_data2/final_data4`；逐帧输出：`final_proposals.jsonl`；校准：`final_calibration.json`。同目录早期`candidates.json`和`source_final_candidates.json`不应与最终源码混用。v21的`verified_candidates.json`也不是v21最终清单；v21最终是`source_final_candidates.json`，但v22仍须新建，不能改旧hash复用。

从AutoMoT目录运行：

```bash
python -m qwen3vl_local.sft_new_loop_phase4.candidate_pool --output /tmp/new_v22_candidates.json
python -m qwen3vl_local.sft_new_loop_phase4.privileged_producer \
  --candidate-pool /tmp/new_v22_candidates.json --route-list /tmp/routes.json \
  --contexts /tmp/current_contexts.json --output /tmp/new_v22_proposals.jsonl
python -m qwen3vl_local.sft_new_loop_phase4.privileged_producer \
  --calibrate-references /tmp/new_visible_references.json \
  --contexts /tmp/current_contexts.json --output /tmp/new_v22_calibration.json
```

contexts是JSON列表，严格字段见`privileged_context.context_index`；其中sources_sha256是该实例当前三帧有序`sources`列表的digest，必须绑定相同当前RGB；scene_context和geometry_context必须来自当帧可靠证据。未提供contexts时只生成假设状态/几何候选，不能从原始字段伪造已审核状态。新参考格式见`calibrate_references`，横向scope必须为`visible_maneuver_conditions_v1`，reference与contexts独立存放。

任务v22/data_v22，运行快照仍v11（控制器状态格式未变），须新候选/数据/run。验证记录：`twenty_second_audit_verification_20261002.json`。Phase3/Action默认未改。

正式开训仍需：可靠当帧上下文与可见性/实例生产、关键边新范围标签、六事件独立模型评估、预留人工抽检的实际审核和误差测量。全量数据不能以UNKNOWN补NO，也不能以有接口或合成测试YES冒充真实标签生产已经完成。
