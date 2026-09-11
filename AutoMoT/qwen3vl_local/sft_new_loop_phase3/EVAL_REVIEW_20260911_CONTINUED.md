# 2026-09-11 第二轮续审：首次跨线、归位与速度确认

接 [首轮逐帧报告](EVAL_REVIEW_20260911.md)。新增查看12个错例、12个run的192张RGB；其中与首轮重叠57张，
实际新增135张、6个run。两轮累计 **89例（78错例、11对照）、64个run、1,217张不同原始RGB**。
另有冻结prompt后盲标的12张新val RGB，仍单列，不计入上述错误分析数量。
全评测765题有247个production错例，本次仍是针对性抽样，不能将78/247解释为标签噪声估计。

本轮优先挑选静态障碍、归位、骑行者中的左右互换，再补匝道速度漏报。全部12例均核对模型实际四图SHA256、
原始meta与被问动作；无数值或计算不一致。每例4张输入+12张未来证据全部查看，
#74 f103/f106、#583 f34/f36另看1152×384原图。未来RGB仅供标定复核。

## 1. 本轮新增结论

**没有足够证据新增错标隔离；保持首轮40帧修订。** 多个看似标签左右反了的样本，实际是模型选错阶段或方向：

|案例|逐帧事实|结论|
|---|---|---|
|#91 施工|f77先左借道，f86才右回归；模型RIGHT|首跨LEFT有支持，末状态不是答案|
|#633 事故|f50左绕到事故车左侧；模型RIGHT|左绕出和持续增速均被漏判|
|#695 事故|f72向右进相邻车道，障碍留在左侧；模型LEFT|真实右绕出反例，静态障碍不能默认左绕|
|#702 “归位”context|f68仍在左绕出，history未确认已经离开原车道|context ID不等于已经完成绕障，模型RIGHT不成立|
|#53 真实归位|对向白车在前，f198右回到原侧；模型LEFT|与#702构成阶段反例，不能把所有此context都当左或右|
|#583 导航换道|明确无绕障history，f36向右进lane2|问题域还包括普通导航换道，不能硬套绕障故事|
|#42/#47 骑行者|分别是保持车道、速度只小降不足20%|风险存在不自动意味着LEFT或DECEL|
|#110 与#552|同run相邻anchor；前者NONE，后者INVALID|不能算两条独立路线，也不能由一次答案判断场景识别稳定|

#702也说明速度确认必须用完整精度逐点算：f69有增速、f70增速更大，但前者未达20%，
f71又未达，因此不是f69–70确认；实际f74–75（1.75–2.0s）才满足连续两点。
#53同样到1.75–2.0s才确认增速，期间有孤立近零点。把“开始动了”和“达到当前合同”混为一谈会误判标签。

## 2. 全部旧题的横向混淆与新索引分布

以下统计旧765题中**256个有效、被问横向**的题；不是只统计本轮12个样本，也不是新prompt成绩。

|context|LEFT误答RIGHT|RIGHT误答LEFT|GT RIGHT中的RIGHT答对|
|---|---:|---:|---:|
|STATIC_BLOCKAGE|5|3|18/21|
|POST_BYPASS_RETURN|2|5|6/21|
|VULNERABLE_CROSSING|3|0|21/21|
|RAMP_MERGE_EXIT|0|0|2/2|

共有18题直接把左右答反；归位的右跨还有10题漏为无横向、5题反成左。
骑行者组RIGHT的21/21不代表整题全对，纵向仍可能错；匝道RIGHT只有2题，不能宣称泛化已好。

新v8训练索引的LEFT/RIGHT正例：静态障碍376/375、归位376/375、骑行者383/361、匝道201/249。
这说明**新索引没有明显的全局左右数量塌缩**，不支持仅为了纠错继续加倍采样LEFT或RIGHT。
它也不证明旧训练过程平衡，更不保证“绕出/等待/归位”的阶段或独立路线已经平衡。
后续应按实际入训的阶段、图像身份和模型错误分桶复核；不能仅凭context名定义阶段真值。
[统计JSON](probe_output/rgb_review_20260911/round2/direction_distribution.json)。

## 3. 已落实的审计工具改进

`prepare_error_review.py --only-ids` 只准备新案例，复用既有77例证据，不批量重新生成旧RGB拼图。
新增 `audit_review_transitions.py` 从已有逐帧meta表记录所有车道身份变化：起点、下一采样确认、
是否已经发生在输入中、是否回到anchor的身份，以及road/section是否可比。
它不把lane ID正负直接翻译成ego左右、不跨缺帧连接、不把road过渡当正常跨线、不把末帧未确认写成已确认。
输出明确标记 `visual_crossing_confirmed=false`，视觉判断以人工笔记为准。

已对两轮89例运行；[变化时间表](probe_output/rgb_review_20260911/lane_identity_timelines.json)。
新增4项回归通过：首跨与回归分开、anchor动作属于历史、末帧未确认、道路过渡及缺帧不误判。
首轮107项回归仍保留，本次仅新增独立审计代码，没有更改模型/标签合同。

**短prompt保持冻结，未追加场景例子、未修改动作阈值，新v8索引仍适用。**
本轮证据支持先验证已经精简的“首次跨线/已发生不重算/事实而非建议动作”规则，
不支持在没有新模型结果时反复改prompt。上一轮全量原meta核验14,472行均一致，尚无新Qwen训练或生成成绩。

从AutoMoT目录复现本轮取证：

```bash
python qwen3vl_local/sft_new_loop_phase3/prepare_error_review.py \
  --bundle checkpoints/sft_new_loop_phase3_20260910_203334_4rgb_audit_bundle \
  --data-root lead_data \
  --output qwen3vl_local/sft_new_loop_phase3/probe_output/rgb_review_20260911/round2 \
  --only-ids 91 633 695 53 583 702 110 114 154 74 42 47

python qwen3vl_local/sft_new_loop_phase3/audit_review_transitions.py \
  --evidence qwen3vl_local/sft_new_loop_phase3/probe_output/rgb_review_20260911/evidence.json \
             qwen3vl_local/sft_new_loop_phase3/probe_output/rgb_review_20260911/round2/evidence.json \
  --output qwen3vl_local/sft_new_loop_phase3/probe_output/rgb_review_20260911/lane_identity_timelines.json
```

## 4. 新增逐例笔记

结构化记录：[EVAL_RGB_REVIEW_20260911_CONTINUED.jsonl](EVAL_RGB_REVIEW_20260911_CONTINUED.jsonl)。

### #42 HazardAtSideLane f53 — 车道内保持／晚减速

GT：`DECELERATE`；预测：`LANE_CHANGE_LEFT+DECELERATE`。[连续帧](probe_output/rgb_review_20260911/round2/case_042.jpg)

f50–53雾中沿最右车道接近骑行者，之后f54–65车道身份始终-3，白车在左侧运动并不代表ego左变道。f60(1.75s)速度降至3.999，f61才近停；GT DECEL无横向有支持，预测多出LEFT。保留标签；减速发生在较晚窗口。

### #47 HazardAtSideLane f38 — 小幅下降不达阈值／横向正确

GT：`LANE_CHANGE_LEFT`；预测：`LANE_CHANGE_LEFT+DECELERATE`。[连续帧](probe_output/rgb_review_20260911/round2/case_047.jpg)

f35–38夜间城市道路，骑行者在前方右侧，白车从左侧经过。f38速度9.388，未来2s最低8.561，仅下降约0.827，小于20%约1.878；f48(2.5s)lane2→1，随后保持。LEFT有连续原图和meta支持，额外DECEL属于幅度误报。

### #53 ConstructionObstacleTwoWays f188 — 真实右归位／低速脉冲

GT：`LANE_CHANGE_RIGHT+RESUME`；预测：`LANE_CHANGE_LEFT+RESUME`。[连续帧](probe_output/rgb_review_20260911/round2/case_053.jpg)

f185–188施工旁处于借用侧，前方白色来车正面朝向ego；原车道在右。未来f191/f193各有孤立近停，未形成相邻两点STOP；f195–196持续增速满足RESUME。f198(2.5s)lane1→-1、f199确认，RGB显示向右回到蓝车所在侧。预测LEFT方向反了，保留RIGHT+RESUME。

### #74 HazardAtSideLaneTwoWays f103 — 暗处骑行者／左借道误判右

GT：`RESUME+LANE_CHANGE_LEFT`；预测：`LANE_CHANGE_RIGHT`。[连续帧](probe_output/rgb_review_20260911/round2/case_074.jpg)

f100–103夜间道路很暗，f103原图仍可辨前方偏右骑行者及边线。f106(0.75s)lane1→-1，之后骑行者移到ego右侧且被超越；f106原图补查支持左借道。速度f103为4.847，f105–106为7.073/8.430，满足持续增速。GT LEFT+RESUME有支持；暗处目标增加难度但不能把GT改RIGHT。

### #91 ConstructionObstacleTwoWays f76 — 先左绕出、后右归位

GT：`RESUME+LANE_CHANGE_LEFT`；预测：`LANE_CHANGE_RIGHT`。[连续帧](probe_output/rgb_review_20260911/round2/case_091.jpg)

f73–76雨雾中黄色施工架仍在ego前方，f77(0.25s)lane1→-1、f78确认，原图中施工架从前方移到右侧。f86(2.5s)才回lane1、f87确认。GT首次LEFT，模型RIGHT与后续归位同向；速度7.070→9.449→10.951支持RESUME。保留标签，不用3秒末位置代替首跨。

### #110 EnterActorFlow f16 — 相邻anchor不同拒绝输出／暗处加速

GT：`RESUME`；预测：`NONE`。[连续帧](probe_output/rgb_review_20260911/round2/case_110.jpg)

与上一轮#552同一run相邻anchor，f13–16仍是极暗护栏连接路，lane2全程保持。f16速度8.003，f17–18持续升9.707/11.272，GT RESUME算术成立；模型NONE漏报。#552在f15输出INVALID，不能把本例当新的独立路线，当前匝道前提的可见性疑问仍保留。

### #114 HighwayExit f60 — 匝道箭头不是变道／后段增速

GT：`RESUME`；预测：`NONE`。[连续帧](probe_output/rgb_review_20260911/round2/case_114.jpg)

f57–60最右车道有右向箭头、右侧护栏连续、左侧有多车。f61–72始终lane-4，ego未跨线，模型全NO的横向部分正确。f66–67(1.5/1.75s)速度14.170/13.848相对11.428超过20%阈值，GT RESUME有效；这是较晚增速遗漏，不能因出口箭头增加RIGHT。

### #154 EnterActorFlow f67 — 明显制动延续漏报

GT：`DECELERATE`；预测：`NONE`。[连续帧](probe_output/rgb_review_20260911/round2/case_154.jpg)

f64–67白天雨雾匝道/主路接入，左侧灰车黄车靠近，自车历史速度20.602→16.177持续下降。未来f69(0.5s)降12.366，下降超过20%，随后保持lane2。GT DECEL有速度和前方车流证据；预测NONE漏掉当前制动延续，无需改阈值或加横向动作。

### #583 InterurbanActorFlow f34 — 导航换道向右／误答左

GT：`LANE_CHANGE_RIGHT`；预测：`LANE_CHANGE_LEFT`。[连续帧](probe_output/rgb_review_20260911/round2/case_583.jpg)

f31–34夜雨中绿车从左侧退出画面，前方道路右弯。f34和f36原图复查：ego由内侧移向右侧车道，分道白线移到ego左侧；f36(0.5s)lane1→2、f37确认。速度约11保持，GT仅RIGHT有支持。所给history明确没有静态绕障历史，不能将问题ID解释为必然绕障回归。

### #633 AccidentTwoWays f47 — 事故左绕出／方向和增速漏判

GT：`LANE_CHANGE_LEFT+RESUME`；预测：`LANE_CHANGE_RIGHT`。[连续帧](probe_output/rgb_review_20260911/round2/case_633.jpg)

f44–47夜雨中黑色事故车仍在前方，f48–50逐渐从其左侧通过，受阻车辆留在右侧；f50(0.75s)lane-1→1、f51确认。f49–50持续增速满足RESUME，2秒后才进一步减速。GT LEFT+RESUME有支持，模型RIGHT既误方向又漏增速。

### #695 Accident f69 — 静态障碍右绕出反例

GT：`LANE_CHANGE_RIGHT+RESUME`；预测：`RESUME+LANE_CHANGE_LEFT`。[连续帧](probe_output/rgb_review_20260911/round2/case_695.jpg)

f66–69夜雨多车道事故，前方偏左有事故车辆，右側仍有通道。f72(0.75s)lane-2→-3、f73确认，随后车辆从ego左侧退去，支持向右绕行；f73–74持续增速满足RESUME。模型RESUME正确但LEFT错误。此例反证不能把STATIC_BLOCKAGE固定关联LEFT。

### #702 Accident f67 — 归位问题仍处于左绕出

GT：`LANE_CHANGE_LEFT+RESUME`；预测：`LANE_CHANGE_RIGHT`。[连续帧](probe_output/rgb_review_20260911/round2/case_702.jpg)

f64–67白天多车道，红车与黑车在前右，ego向左侧空车道驶入；f68(0.25s)lane-4→-3、f69确认，之后保持。f69–70的增速未连续达阈值，直到f74–75(1.75/2.0s)的10.651/10.479才连续超过anchor8.605的20%阈值。所给history只确认之前遇到障碍，没确认已绕出；GT LEFT+RESUME合理，不能因POST_BYPASS_RETURN名称将动作认作RIGHT。
