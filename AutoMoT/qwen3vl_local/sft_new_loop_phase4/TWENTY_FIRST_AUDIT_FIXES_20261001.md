# v21：特权信息候选生产、校准与离线安全适配

本轮判断：报告关于关键转移监督缺失、源事件片段不能直接定义实例、正式训练未就绪的结论成立。全量特权信息生产是合理方向，但“存在 bboxes/metas”不等于已经拥有可靠条件真值。本轮完成候选生产和校准入口、可接离线回放的安全适配器，以及风险/曝光登记；没有把未经校准的算法输出当成人工监督。

## 对报告的补充核对

- 当前训练集435题、42物理路线，训练题覆盖十类事件；只有val/test局限于UE1/5/6/7四类。应区分训练支持和评估覆盖。depart/enter/return/recover_follow仍没有训练题，UE7 proceed readiness YES仍0、RE5仍1个。
- `lead/lead/expert/expert.py` 的 `rear_danger_8/16` 仅覆盖指定场景及特定rear actor，不是全后方自由空间证明；不能将False当安全许可。
- `expert_data.py` 保存时追加 `future_positions/future_yaws` 等信息。生产计算只允许当前字段及过去帧，不读取未来数组、专家控制、target_speed或源场景事件片段。
- `num_points` 是LiDAR点；`visible_pixels`来自相机语义点计数。它们不能证明RGB在雨夜、缩放后足够可辨。`vehicle_cuts_in`只在特定场景按启发式设置，不能当通用UE3检测器。
- `ego_velocity`是世界速度旋转到自车坐标的绝对速度；跟车比较需要相对速度；安全适配器的整段机动包络固定在当前坐标系，参与者扫掠使用同一坐标系的绝对速度，不能再减自车速度。保存的speed是有符号量，微小负值不能一律当坏数据。
- bboxes收集半径并不证明该帧完整、静态可行驶空间安全或拥有通行权；正常跟车也不单独证明刚发生过UE1制动事件。
- v20已经允许外部有效安全凭据；“所有离线回放永远无法机动”应限定为没有提供凭据的路径。本轮新增计算适配器，但仍不等于CARLA整套闭环已验收。

## 代码变化

`privileged_geometry.py`读取本地受信任的xz pickle，白名单投影当前元信息/参与者几何，绑定原始来源与RGB SHA。提供连续历史跟车、显式导航走廊的可见间隙/通过/进入候选、近距离参与者ID缺失检测。消失不变成道路已清空，LiDAR点数不代替相机可见性，身份断开重新起实例候选。

`privileged_producer.py`逐路线流式扫描冻结清单，保留物理split。每帧输出处置、参与者实例候选、条件建议、弃答原因及异常候选。默认只处理train，可显式选择冻结val/test。输出使用独立schema，所有`training_target=UNKNOWN`且`supervision_approved=false`，不能冒充condition_builder的审核条件流。RE2/RE3导航、UE6/RE5通行权、UE7故障已建立等条件缺失时明确保留未知。当前尚不是十事件完整可靠生产者。

`replay_safety.py`实现`ReplaySafetyAdapter`，`replay.py --offline-geometry-safety`与`evaluate.replay(..., safety_adapter=...)`接入。它依据当前和上一帧bboxes对全机动包络检查后侧/前方参与者的保守运动包络，绑定当前RGB、原始文件SHA、实例/边/方向/目标/分段和请求。导航必须提供包络及目标车身范围、半径/时域/速度/加速度/余量、完整参与者清单与静态可行驶空间/通行权证据。半径不足、未知速度、近距离ID消失或证据不全均不签发许可。安全许可不等于动作执行确认。

控制器区分安全明确拒绝与未知：`clear=false`保留正常等待，不能累计成“不确定120帧”故障；缺许可仍不授权并遵守原不确定复核机制。新增回答矛盾/缺安全信息/明确拒绝帧统计，用实际回放测量RECOVER多题冲突，不凭空删除必要约束或优先采用某个YES。

顺带修复条件编译器旧范围横向题构建为待审后加载失败的问题：来源核对遵守相同UNKNOWN隔离，不允许旧记录自行声明已经完成新范围复审。

## 实际验证

- **567项专项测试通过**，包括因果字段排除、后车冲突、消失弃答、包络/方向/时间身份拒绝、真实lzma文件读取接回放、允许进入但不伪造执行、明确拒绝长期等待、编译/加载隔离和冲突统计。未测试真实4B/GPU/DDP/CARLA效果。
- 全目录重新清点9712路线、1,637,680帧；冻结split仍8166/765/781。
- 实扫新增六路线712帧，全部有可读取来源，得到26个ID缺失候选。2954_1的51→52检出3742、52→53检出3738/3734、119→120检出3696，与复看的原图位置一致。其余候选未经原图核实，不报告为确认异常或新增完整RGB审计。
- 开发集校准排除98个旧范围横向readiness参考；320个有效参考中**311弃答、8正确、1错误**。覆盖率2.81%；已回答9题中错误率11.11%，样本太少且是开发集，不是独立准确率认证。唯一错误为HardBreakRoute/Town12_3452_0的settle第83帧，人工YES/候选NO。没有为凑正例调阈值，也没有批准自动标签。
- 2/4RGB真实`MODE=check`通过，各894监督题与533待审题加载RGB核验；train435/val273/test186及review四份JSONL与v20逐字节一致。world1/4七轮事件等额、共享帧cap8及恢复历史一致性再次核对，七轮覆盖全部435题。详见同目录`twenty_first_audit_verification_20261001.json`。

临时完整产物：`/tmp/phase4_v21_work/`。`calibration_final.json`保留逐参考结果，`final_proposals.jsonl`保留逐帧处置，`verified_data2/verified_data4`是本轮重建数据。仓库内验证记录保留摘要、来源身份与错误项。

## 风险与曝光

六物理组显式train-only，均原本就在train，未挪动冻结holdout。新增七条风险记录：2954_1两处已复看原图的消失，以及五条源实例/可见性风险。Scenario3_15第4→5仍为未确认线索；146_1第18→19沿用v20已登记证据，不重复计数。风险进入构建/加载的逐帧审核门槛，不直接生成NO/YES。

本轮实际只复看2954_1五张原图，无新增完整逐帧序列、无新人工标签。继承192/278格满三ID、约240条剩余的统计，不将程序读取712帧冒充目视审计。

## 运行入口与边界

在AutoMoT目录下，使用新候选清单运行候选生产：

```bash
python -m qwen3vl_local.sft_new_loop_phase4.candidate_pool --output /tmp/v21_candidates.json
python -m qwen3vl_local.sft_new_loop_phase4.privileged_producer \
  --candidate-pool /tmp/v21_candidates.json --splits train --output /tmp/v21_proposals.jsonl
python -m qwen3vl_local.sft_new_loop_phase4.privileged_producer \
  --calibrate-annotations qwen3vl_local/sft_new_loop_phase4/reviewed_state_pairs_v7.json \
  --output /tmp/v21_calibration.json
```

全量扫描的程序入口存在，本轮只实际运行六路线，不宣称完成9712条路线的条件生产。可用`--route-list`传入`scenario/route_id`对象列表作有限校准。输出路径必须新建。

回放每条观察可附加`privileged_source={scenario,route_id,source_sha256}`，其中SHA为当前与上一帧`sources`有序列表的digest；`maneuver_safety_requests`字段格式见`replay_safety.assess`。请求必须由实际当帧导航/覆盖证据提供，不能为跑通而硬填True。缺导航输入的旧序列仍不能得到横向许可。真实源文件读入与回放接线通过合成fixture验证；真实数据上的完整导航证据及CARLA仍未验收。

任务合同v21/data_v21、快照v11；须新候选/数据/run，不改旧hash硬续训。四冻结规则、人工标签、冻结holdout、采样/训练/选优源码核SHA保持；Phase3/Action默认保持。

正式训练仍不可批准：关键边正负样本和六事件独立评估未补齐；可靠十事件生产者、可见性/通行权/导航证据及独立人工误差抽检仍需完成。自动标签生成方法相同只能检查模型对该方法的一致性，不能代替独立人工真值的准确率检验。现有题库只允许明确限定范围的探索训练。
