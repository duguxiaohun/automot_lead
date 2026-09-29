# Phase4 第三轮部分审计修复

依据用户提供的 `/tmp/phase4_third_audit_20260929/AUDIT_PARTIAL.md` 修复三处合同问题。原只读报告和标注不改写。三轮报告累计224条完整序列、26,418帧，92/278单元达到三个ID，仍剩476条计划序列，18单元来源不足三个ID；**完整审计尚未完成**。这些计数来自用户审计，不是本次新增审阅量。

## 修复结果

1. `re_yield`只在没有HOLD时隐式建立APPROACH。七类让行事件均覆盖连续调用：PROCEED/RECOVER→hold YES→PROCEED/HOLD→re_yield YES、release NO→YIELD/HOLD，始终STOP。新的proceed许可可以恢复，但仍需实际执行确认；未确认许可失效时恢复原HOLD。既有re_yield+stable矛盾检查保留。
2. `Phase4Loop.tick`先检查观察合同、整批新鲜度、逐实例上下文及故障参数，再在副本上完成预测、推进、回执核对和汇总；全部成功才提交。重复帧、错误答案、预测器异常、后一个实例回执错误等抛异常路径，整批状态/历史/待执行许可保持不变。原有效回执仍可确认，也可重试同一帧。直接Episode.advance、公开revalidate和acknowledge同样避免失败留下中间更新；成功提交保留原Episode对象引用。合法UNKNOWN/INVALID及语义矛盾会按原协议记录不确定/复核，它们不是输入异常回滚。
3. catchup新增逐帧 `transition_blocking_conflict`，明确当前冲突是否阻止本条转移。几何完成事实为真、当前有独立纵向限制且该字段为false时，readiness和catchup均为YES，控制器仍保留HOLD/STOP。实际转移条件为false仍为NO；INVALID/UNKNOWN不被后继确认覆盖。已有 `current_conflict=false` 的旧审核兼容为无阻挡；`current_conflict=true` 却缺少明确作用范围时拒绝构建，要求复审，不自动反标NO或猜测YES。布尔类型和作用范围相互矛盾的记录也拒绝。

原F3复现数据现在因缺作用范围被拒绝；补入明确的独立纵向限制审核后得到YES/YES。默认429题没有写入这个构造案例，没有更改已有监督标签。现有提示词已明确几何完成不解除纵向限制，因此本次提示词源码逐字节保留。

## RGB复核与边界规则

本次实际查看七张连续拼图，共155张既有审计RGB，均包含页内全部连续帧，不增加完整序列或去重累计计数。面板SHA记录于 `third_audit_exposure_20260930.json`。

| 报告ID/路线 | 本次实际查看 | 观察及处理边界 |
|---|---|---|
| 645，CrossJunctionDefectTrafficLight/Town12/route002115 | f24–71，48帧 | 多轮横向来车穿过，路口内挪动后再等；支持独立保留HOLD，不能将最初proceed许可看作全路口持续授权。 |
| 091，HazardAtSideLane/Town12/1092_2 | f72–95，24帧 | 骑手暂离右侧视野后在自车排队时重新出现并靠近；返回间隙需重新判断。已有侧向进入事实与新的返回许可分别管理，不用首次出画面持续扩窗。 |
| 347，NonSignalizedJunctionLeftTurnEnterFlow/Town10HD/route001073 | f48–95，48帧 | 从前一路口左转后接近另一T口，黄色横车通过后右转；不同路口目标必须由导航/因果跟踪给出不同instance_id，旧实例许可不能复用。 |
| 132，MergerIntoSlowTrafficV2/Town06/route000832 | f0–23、48–58，35帧 | f7→8整组车突变消失，f52→53前方红车突变消失；不能作为自然条件解除标定。生命周期原因仍需采集端日志验证。 |

这些观察只支持上述区分，不批准新的精确YES区间。夜间缩略图不能可靠确认侧后完整间隙。绕行途中再次遇车使用当前纵向restrict/hold/release约束，不能把已执行的横向进入事实整体回滚；未执行的机会在下一有效观察重新核对。实例边界由上游导航/跟踪确认，当前代码不宣称能自动从RGB分割所有新路口或追踪所有骑手。

标定stop.kind新增 `instance_boundary` 和 `calibration_anomaly`：人工确认边界后，从截止帧起excluded，不把截止之后写成NO；新实例须另建独立审核带。`new_conflict`仍可截止已不适宜补问的旧问题。新增两种原因不是自动异常检测器，不会凭actor消失自动批准清空，也不自动排除整条路线。采集异常后何时允许重新标定、因果历史是否仍受突变影响，仍需逐帧复核，不能固定延长或缩短半秒。

## 数据隔离和验证

报告中的11个新曝光物理组已逐项对照实际visual_notes记录，加入Phase4 train-only集合。没有把未审selection登记为已审。来源报告、逐序列笔记、覆盖表和复看面板的SHA已绑定新合同；原v3标注、边界证据及前次曝光登记SHA不变，Phase3/Action稳定目录和默认来源未改。

202项Phase4测试通过，包含原有真实小型Qwen3.5的2/4图监督与LoRA反向。新增测试覆盖连续HOLD→re_yield、重复/过期观察、静态输入拒绝、跨实例预测/回执失败、恢复快照、重试与原许可确认、几何完成/独立停车、冲突作用范围、审核截止和曝光隔离。新鲜NO会按已有语义重置未执行异常计数，这是普通等待，不计为执行故障。

2/4图各重新构建429道训练开发题和121条待审，所有train/val/test题逐行与v5相同（包括标签、题目、图像身份、审核事实）；全部429题各自的因果RGB加载和哈希校验通过，各7轮×world1/4采样回放通过。`probe_output/third_audit_fixes_v6/`保存manifest、preflight、修后探针和validation.json。没有修改或覆盖原审计复现文件。

任务合同v6、快照v6、标定代码v4，提示词保持v5，人工标注文件保持v3，先验导出source仍v2。train.sh默认data_v6，须新构建产物和新run；旧adapter/快照必须使用原版本。副本事务增加了随实例历史增长的CPU复制开销，尚无长期实时吞吐验收；预测器应仅在传入副本上读取，不在外部执行控制，外部推理日志等副作用不属于状态回滚范围。

ready=false：公共纵向边、re_yield、分支和独立val/test标定仍不足，完整4B权重缺失；没有正式4B/GPU/DDP/CARLA验收，没有训练效果提升结论。
