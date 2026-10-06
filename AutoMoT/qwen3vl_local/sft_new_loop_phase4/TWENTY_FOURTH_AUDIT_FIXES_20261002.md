# Phase4 v24：选择方案 B，局部释放建议与逐帧审核入库

用户对 v23 的核心诊断成立：把稳定跟车当成释放前提会错判前车驶离；55m整条规划走廊会让下一路口的车辆污染当前行人事件。只生产待审候选又没有审核入库路径，不能扩大训练集。本轮选择**方案 B**，实现有限数量转移审核、显式逐帧条件确认、编译为已有严格标注格式，并实际加入首批14道监督题。**不是规则老师获准自动标注，也仍不能正式训练。**

## 代码与真实回归

`event_scope.py` 把“原阻挡正在解除”与 `recover_follow` 的“已经稳定跟车”分开。前车身份连续、可见、同车道，当前速度朝前且间距持续扩大时，可以提出释放建议，无须先满足稳定跟车的2秒时距/相对速度门槛。3m间距、运动幅度等数值只用于开发建议，不是认证驾驶策略。原稳定跟车判定、运行时状态机和动作先验未放宽。

当前参与者附近的局部规划段用于提出释放建议；同一区域内其它参与者仍否决或使其未知，下一路口远处车辆单列为范围外参与者。局部区域目前使用6–20m的候选范围与参与者前方余量，这是审核用近似；不能据此证明完整通行权、未来不会有新参与者、或可安全进入下一路口。参与者缺失、暗图、瞬移和自车大幅转向不会产生释放YES。已经观察到的静态异常仍拒绝推断。

- HardBreakRoute 1030_0：局部释放建议第67帧NO、68帧未知、69–84帧正向；旧 v23 的持续NO消除。完整条件题仍因独立证据缺失保持UNKNOWN，不能把局部建议当成完整许可。
- DynamicObjectCrossing Town07 3_1：第49帧仍阻挡，第50–52帧释放建议为真；前方下一路口车辆不再污染行人实例。新近处行人仍能将结果否决。
- 路口候选不再由“35m内”单独触发UE6，也不由有灯/无灯分别猜UE7/RE5。UE6需要可见近处横穿冲突，RE5需明确适用的停车标志；UE7只有Off/Unknown信号才作为待核假设，仍不等于已确认故障。这会漏掉部分复杂事件，应通过明确审核请求补充，不能宣称完整自动召回。
- 已驶到身后的旧参与者只有最近因果历史中仍在前方时短暂保留，不再在自车停车期间无限延续。后续新车辆可建立新实例。

同17条训练路线、1615帧，原始待审状态对从6066降到3700；其中局部释放建议76个真、354个假、929个未知。**这些数值不是教师准确率或自动监督数量。** 满足因果历史长度的3440个状态对经转折筛选、逐事件预算（每事件最多16张卡）形成100张审核卡；未选中不标NO。工具流式处理，仅保留单路线跟踪状态及每事件有限卡片，避免按全量帧无限积累待审表。

## 方案 B 的实际路径

新增 `review_queue.py`：

1. `--proposals` 从局部释放变化、实例首尾选审核锚点；或 `--requests` 明确提供路线、当前状态、边、方向及目标走廊。这使RE2/RE3和UE2绕行目标无需被生产者猜出即可进入审核。
2. `--template` 产生默认pending、条件未知、所有确认项为false的表。`--panels` 显示每个锚点截至当帧的7帧因果RGB，标出左/前/右相机边界。这7帧是2/4图输入的审核支持，**不是固定正例扩窗**。
3. 审核者逐项确认事件身份、状态对、局部冲突与优先权，填写每帧SHA/观察、每项条件及YES/NO。明确区分readiness与旧状态catchup；补问须再确认后继状态和本边冲突范围。里程碑须确认当帧已发生，横向边须重新审核可见范围。
4. `--reviews` 编译为 `reviewed_transition_band`；沿用构建/加载的风险、可见范围、真实输入冲突、因果历史及物理split检查。已知风险路线需逐历史帧显式确认各项风险检查，空白/未知/缺帧/改SHA/错风险ID/错误条件不获批准。
5. 开发审核只允许训练split，40条独立抽检预约和原独立val/test不用于规则开发或此次标注。

这里的逐帧审核由本轮Codex直接查看RGB后填写，**不是额外独立人类标注者的准确率认证**。自动建议不进入模型输入，审核材料也不增加模型文字字段。现有YES/NO、状态对及因果2/4RGB合同不变。

## 新监督：449题、45条训练路线

保留 `reviewed_state_pairs_v7.json` 的152条旧记录逐条不变，新增14条单帧审核到 `reviewed_state_pairs_v8.json`，默认完整流水线已切到v8。四种新题均经过真实2/4图构建与加载：

| 事件/边 | 原图确认的锚点 | 新监督 |
| --- | --- | --- |
| UE1 proceed，1030_0 | 70；71–74 | 1 readiness YES；4 catchup YES |
| UE4 proceed，Town07 3_1 | 47–48；50–52 | 2 readiness NO；3 readiness YES |
| UE2 depart，002000 | 33–34 | 2 readiness NO，RIGHT相邻车道近车清晰可见 |
| RE3 enter，000839 | 44–45 | 2 readiness YES，FORWARD沿当前路线进入左出口分支 |

合计10YES/4NO，其中6 readiness YES、4 readiness NO、4 catchup YES；补问不计入readiness支持。1030_0的第69帧规则建议为真，但本轮没有据此直接给69帧正例。第71–74帧已见恢复前行，单列旧状态补问。000839只确认当帧近处左分支的可见进入条件，浓雾远方及后方安全未获认证；运行时仍需规划器/BEV许可。

训练从435/42路线变成**449/45路线**。RE3 enter首次有2YES，UE2 depart首次有2NO；UE2 depart YES、return、recover_follow仍缺，UE7 proceed readiness YES仍0，RE5仍1YES/27NO。val273/test186及533题待审JSONL与v23逐字节一致，旧435道训练题逐行不变。新增题少，不能称关键转移支持已齐备。

## RGB复核中的两点纠正

本轮实看20张连续联系表、84张不同RGB，6条路线局部；其中4张再核原图，不重复计数。未新增完整序列、未增加三ID覆盖格。所有6组早已显式train-only，没有新增开发曝光组。

| 路线 | 本轮实际观看范围 |
| --- | --- |
| HardBreakRoute 1030_0 | 63–84 |
| DynamicObjectCrossing Town07 3_1 | 41–62 |
| DynamicObjectCrossing Town03 3_16 | 51–55 |
| InvadingTurn 1150_0 | 151–155、240–250 |
| ConstructionObstacle 002000 | 27–34 |
| HighwayExit 000839 | 38–48 |

**疑似重复行人更符合相机重叠。** 原图Town07第55帧、Town03第53帧中，两个形象分别在拼接x=384两侧。三相机yaw为-54.5/0/54.5度，各FOV60度，存在重叠；每帧只记录一个walker。该ID的投影中心分别约368/392像素、379/404像素，与两个形象位置吻合。因此不把它登记为“第二个参与者”或“确认重复渲染”采集风险；已登记解释及投影证据，审核图增加相机分界线。不能据单ID清单普遍断言其它疑似人形都不是新行人。

**1150_0不是151之后永久无冲突。** 第151帧原黄车驶过，152–155近处清空；240–250原图出现新的横穿与对向车辆。应终止旧实例、按新参与者处理，不能仅凭候选最小/最大帧号判定125–250为同一连续误报区间。

另：1030_0的8m是bbox中心相对位置，不是扣除车身后的车头间距；第70帧自车已有约0.14m/s。两者都不能替代逐帧RGB条件标定。

## 验证与仍未完成的工作

630专项测试通过；真实2/4图pipeline check通过，各908题监督与533题待审全部解码核验输入身份。默认七轮world1/4事件1:1、cap8与恢复历史一致，第3轮覆盖449题。新旧条件审核表、编译结果、监督增量和保留数据均有独立断言。

任务合同v24、默认data_v24、快照v11；须新候选/数据/run。taxonomy/prompts/calibration/observation、运行时控制器/安全准入、原人工标注/原holdout、采样/训练/选优源码SHA保持。全目录仍9712路线、1637680帧，split8166/765/781。40条抽检v4保留原路线/原split，在它们未被查看前绑定更新源码；仍无独立人工标签。

**formal_data_ready=false。** 剩余关键转移、六类独立评估、40条抽检、全量可靠监督仍未完成。RE2/RE3及绕行目标的自动发现仍缺，本轮通过明确目标的半自动审核补了有限样本，并未声称自动导航和教师认证已完成。当前范围之外的路口身份、局部区域选取及可见性仍会产生误候选；不能直接将76个释放建议批准为YES监督。无完整4B/GPU/DDP/CARLA训练验收。

本轮未主动扩展剩余约240条完整逐帧审计，继承192/278格覆盖；用户报告累计13条完整/约14条局部与本轮局部复看分别记录，不将程序读取当作目视。

证据：`twenty_fourth_rgb_review_20261002.json`、`twenty_fourth_review_queue_20261002.json`、`twenty_fourth_review_decisions_20261002.json`、`twenty_fourth_audit_verification_20261002.json`。本机最终产物位于 `/tmp/phase4_v24_work/verified_candidates.json`、`verified_data2`、`verified_data4`、`verified_proposals.jsonl`、`verified_triage.json`。

## 复现命令（仓库根目录，使用不存在的新输出）

```bash
PYTHONPATH=AutoMoT /tmp/automot-qwen35-env/bin/python -m pytest -q AutoMoT/qwen3vl_local/sft_new_loop_phase4/tests
PYTHONPATH=AutoMoT python -m qwen3vl_local.sft_new_loop_phase4.review_queue \
  --proposals /tmp/phase4_v24_work/verified_proposals.jsonl --per-event 16 \
  --output /tmp/new_review_queue.json --template /tmp/new_review_form.json --panels /tmp/new_review_panels
# 人工填写表后编译；未填的pending不会变成标签。
PYTHONPATH=AutoMoT python -m qwen3vl_local.sft_new_loop_phase4.review_queue \
  --queue /tmp/new_review_queue.json --reviews /tmp/new_review_form.json --output /tmp/new_reviewed_annotations.json
```

对自动实例尚缺的边，`--requests requests.json`替代`--proposals`，每项提供scenario、route_id、frame_id、episode、edge；方向和目标由当帧导航/审核确定，不从未来专家轨迹倒推。新标注与现有v8合并后使用新路径构建；不要覆盖冻结旧标注或旧manifest。
