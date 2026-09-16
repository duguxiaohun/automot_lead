# 2026-09-16 Phase3 逐帧 RGB 续审与修复

> 本文保留首轮审计状态。随后用户要求继续完善标定与提示词，当前已升级 prompt v9 / 动作实现 v8；阈值不变，详细变更和198项验证见 [TEMPORAL_REFINEMENT_20260916.md](TEMPORAL_REFINEMENT_20260916.md)。下文“保持提示词/实现不变”仅指该首轮阶段。

## 结论

继续完善当前版本，保留上一版权重和源码作为历史基线；目前没有足够证据宣布应回退或上一版是最优版本。
09/11 与 09/15 使用不同 holdout，binary 67.57%→63.27% 只是跨数据集的表观下降。
本轮修复的是明确的审计/数据覆盖缺口，尚未训练新模型，也没有新模型提升结论。

四组总体、逐动作 P/R/F1、十类情境、同题比较与训练曲线见
[指标对比](AUDIT_COMPARISON_20260915.md)。本轮不更改 `v8_shared_temporal_rules`
prompt 或 `current_wait_first_crossing_v7_rgb_guard` 数值动作规则。

## 1. 实际审计范围（首批 67 窗口，续审后累计 79）

- 来源为 09/15 四图 binary bundle 的 **67 个窗口：57 个错例、10 个正确对照**，覆盖 54 个 run；
  两图/四图、binary/choice 的逐题配对指标已另行核验。这不是四包所有错例的人工逐帧覆盖。
- 先检查该评测池的 176 条路线是否异常，再逐帧查看四张实际输入及随后最多 13 帧离线证据。
  主拼图共 1,138 个帧面板、1,065 张不同 RGB；另补看 #32 连续 109–132、#355 原图与邻帧 meta。
- #122 缺未来 f48，仅查看到 f47并记录缺失；不补造图像。
  +3.25s 仅用于确认 3s 末端跨线，不扩张预测窗口。
- 67 例原图输入 SHA、原 meta 速度与旧动作规则复算一致；规则一致不等于语义标签全部正确。
  原始未来 meta/RGB 只作离线核验，不作为模型输入。
- [逐例手工笔记](rgb_review_notes_20260916.jsonl)保存观察、实际查看帧号、输入指纹和证据来源。
  [本地 RGB 浏览页](probe_output/rgb_review_20260916/index.html)连接现有拼图，明确标为定向审计。
  本地 RGB/HTML/自动报告是可再生产物，不入库；没有这些文件时可复用笔记与原始 RGB。

## 2. 错误归因与保留的边界

| 类型 | 逐帧依据 | 决定 |
| --- | --- | --- |
| NONE 误触发 | #67/#77/#116/#143/#254/#308/#456：合流、障碍或其他前提存在，并不意味着预测窗内必有动作 | 保留 NONE，不以场景名替代未来实际行为 |
| 历史动作被复制 | #147 历史等待后立即起步；#434 历史左绕、未来右回；#254 左跨线已在输入历史完成 | 保持从最新帧之后预测的合同 |
| STOP/DECEL 边界 | #302 单点零速不足持续近停；#374 首个零速在 +1.5s、第二帧超窗 | 保留 DECEL，不为模型错例调阈值 |
| 横纵向联合混淆 | #214/#345/#448：方向部分正确，但加减速相反 | 不把方向命中当完整动作正确 |
| 横向视觉低置信 | #355 近停时 lane -2→-3，meta 横移/转向连续、编号持续；夜暗无法精确确认轮胎跨线 | 保留标签并披露低置信，不强行改 NO |
| 启动渲染不连续 | #5/#188 首帧与后续天气/actor 出现不连续 | 记录输入风险，不据两例扩大路线过滤 |

#166 已用原始 meta 纠正拼图小字误读：f133=5.829659、f134=5.112775m/s，降幅不足 1.2m/s；
后续只有孤立增速，NONE 保留。人工笔记可以修正，不能把初次视觉读数当作不可变真值。

### #32：事件正例缺乏充分证据

`DynamicObjectCrossing/Town07_Rep0_Town07_Scenario3_6_route0_01_11_05_19_17`：
连续 RGB 显示对向车沿黄线左侧通过，输入 121–124 未确认切入；源规则为
`event_dynamic_cutin_or_occupancy`，只有 `vehicle_hazard`，`brake_cutin=false`、
`dist_to_cutin_vehicle=null`，并带 `dynamic_cutin_actor_unverified` 风险标记。

[精确决定](event_rgb_exclusions_20260916.jsonl)仅在核验的 121–124 窗口撤回 U-E3 正监督，
不改成 NO/INVALID、不补 U-E4、不扩展到整条路线。**实际源标注 121–123 是 R-E1，只有 124 是 U-E3。**
真实 source smoke 显示：默认风险过滤前后均 14 个候选；显式 `include_visual_risk` 时 16→15，
仅移除 f124 的 DYNAMIC_CUTIN。因此它是精确防回流约束，不能声称默认池额外删除了四个正例。

## 3. 新负例：冻结 prompt 后盲审

固定 split seed=20260916，排除开发路线后按原物理路线 hash 自然分割，未按模型预测挑选或改 split。
盲审时 `prompts.py` 源码 SHA（区别于渲染后的 prompt 合同 SHA）为 `1b31876e83bb39bed3e76090559ee27e29836eaa020f74ddca6c4516653cd27a`。
8 条候选均实际查看四帧；6 条接受，2 条夜间证据不足拒绝。未查看模型输出。

| split | scenario / route（省略采集时间戳） | anchor | 决定 |
| --- | --- | --- | --- |
| test | HighwayCutIn / Town12_Rep0_2457_1_route0 | 16 | 接受 |
| test | HighwayCutIn / Town12_Rep0_2944_2_route0 | 16 | 接受 |
| val | HighwayCutIn / Town12_Rep0_4013_4_route0 | 16 | 接受 |
| val | HighwayCutIn / Town12_Rep0_953_3_route0 | 16 | 过暗，拒绝 |
| test | EnterActorFlow / Town12_Rep0_2843_0_route0 | 32 | 接受 |
| val | EnterActorFlow / Town12_Rep0_4078_2_route0 | 32 | 过暗，拒绝 |
| test | EnterActorFlow / Town12_Rep0_4078_4_route0 | 32 | 接受 |
| val | EnterActorFlow / Town13_Rep0_1382_12_route0 | 32 | 接受 |

只给 **R3 + STATIC_BLOCKAGE/VULNERABLE_CROSSING 错误前提**负监督，不否定同帧切入/合流。
新增 val 2 个物理路线、4 题；test 4 个物理路线、8 题。达到最低独立支持并不代表所有道路/事件拒绝能力已覆盖。
[完整盲审记录（含拒绝）](blind_rgb_review_20260916.jsonl)、[正式负例决定](same_rs_invalid_review_20260916.jsonl)
均保留原图 SHA、冻结 prompt、自然分割与观察理由；加载时重新核验 SHA。

## 4. 已落实的代码修复

1. binary 索引的 val/test 各至少 2 条同 RS 错事件物理路线，否则在模型/NCCL 初始化前失败。
   同一物理路线的 Rep 或采集时间不同不增加独立支持；实际 generation 采样后再次检查。
   choice 不评该类题，豁免这一门槛。
2. INVALID 采样、训练/eval 指标及重建索引审计共用物理路线身份；不降低 guard、不靠重复抽样制造支持数。
3. 独立 eval 默认 `cases_per_bin=0` 全量，避免本轮 492 个独立题只评 490 个。
4. 修复 `prepare_error_review.py` 把前四格固定当输入的问题：2RGB 现在只有前两格标为输入，
   后两格明确为未来证据；4RGB 保持四张实际输入。
5. 默认构建/训练/eval/RGB 矩阵入口统一为 **v10**，split seed=20260916。
   [本轮暴露的 176 个 test 物理组](development_route_groups_20260916.json)后续只进 train。
6. 精确排除、新负例和开发路线名单进入 mapping 合同；旧索引/adapter 不能混用。
   原 collection 和旧审计包未回写，Phase1/2 权重不会因此自动得到修正。

## 5. 验证、局限与下一轮

已完成：Phase3 **186 项测试通过**；2RGB/4RGB 帧边界回归；RGB 替换拒绝；物理路线去重与早失败；
6 条真实负例经默认 loader 的原图/meta 检查；1 条真实 collection route 的默认/保留风险双模式 smoke；
新负例配合合成其余 source 的 12 seeds 采样覆盖检查。后者不是全量真实分布验证。

**未重建全量 v10 索引、未运行新模型训练/生成或 CARLA。** v10 的真实全量分布、采样预算和模型能力
必须在全源重建后验证；不能把 CPU 测试通过写成准确率提升，也不能把 67 个精选窗口外推为数据集噪声率。

下一轮按[运行说明](SFT_NEW_LOOP_PHASE3_RUN.md)先全源构建、CPU sampling-only，再新训。
四组比较固定索引、seed、预算及完整评测覆盖，单动作子集和完整 binary 分开报告。
保留旧版本历史结果；若比较旧/新版本能力，应在额外冻结且未用于调规则的路线集上做配对实验。
已暴露的本轮 test 不再作为未见 holdout。

本地浏览页复现（仓库根目录）：

```bash
python AutoMoT/qwen3vl_local/sft_new_loop_phase3/render_review_notes.py \
  --evidence AutoMoT/qwen3vl_local/sft_new_loop_phase3/probe_output/rgb_review_20260916/evidence.json \
  --notes AutoMoT/qwen3vl_local/sft_new_loop_phase3/rgb_review_notes_20260916.jsonl \
  --output AutoMoT/qwen3vl_local/sft_new_loop_phase3/probe_output/rgb_review_20260916/index.html
```


## 6. 同日继续：两图/四图差异及 choice 合流错例

补审 **12 个新窗口**：#52/#148/#204/#212/#236/#300/#344/#350/#356/#382/#470/#482。
均先检查异常路线，逐帧查看四输入及 13 个未来证据帧；原输入 SHA、速度和动作复算全部一致，无缺图。
累计 **79 个窗口、60 个 run、1,245 张不同 RGB、1,342 个面板**，不含盲审和额外局部放大图。
按四图 binary 统计其中 63 错、16 对；新增正确例是其它模式的错例，不等于随机正确对照。
[新增 12 条笔记](rgb_review_notes_20260916_continued.jsonl)与
[续审浏览页](probe_output/rgb_review_20260916_continued/index.html)独立保存，不重复生成旧拼图。

- **#482 历史跨线重报**：f82 最新输入已到左侧 lane-2，随后不再跨线，只有持续增速；两图 binary 多报 LEFT。
- **#236 两个时间窗不同**：f94(+2s)增速尚不足 1.2m/s，f95 后才明显增速；f97/98 左跨线仍在 3s 内。
  所以只标 LEFT，binary 的 RESUME 把纵向窗外动作混入；两图 choice 正确，四图 choice 错。
- **#470 孤立增速峰值**：f103 超过门槛但 f104 不够，f109 也无窗内连续确认；f109/110 右跨明确。
  RIGHT 保留，不能为四组误报纵向动作而改阈值。
- **#382 STOP 跨窗**：首次近停在 +1.75s，不满足 1.5s 停车窗；此前增速峰值不足连续确认。
  DECEL 保留，两图 choice 误判 RESUME。
- **#344/#356 联合动作**：分别为当前连续近停后左跨、持续增速同时左借道；四图 binary 对，两图漏横向。
  #356 后续右回归不改变“第一次左跨”的目标。
- **#52/#148 合流共同错误**：#52 实际持续增速，四组都误选 STOP；#148 输入历史增速但未来明显减速，
  binary 漏报、choice 误选 RESUME。额外中间帧和单选输出都未解决这些例子。
- **#204/#300 NONE**：未来速度变化不够各自阈值、无未来越线，保持 NONE；分别是两图/四图胜出。
- **#212/#350**：分别确认未来右跨与持续增速；四图/两图 choice 各赢一个。雨雾/夜暗影响证据，
  单 seed 的不同权重预测不能把错误因果归结为删掉了哪张中间帧。

本批未发现足以新增标签排除或调整数值规则的证据。#204 只确认动作，看到灯具不等于已确认信号故障语义。
保留现有时间窗和 prompt；下一轮重点单列 NONE、历史已完成跨线、持续增速确认、联合动作与合流切片。
续查代码另修复 `audit_rebuilt_index.py` 的路线口径：`unique_routes` 按物理路线计，`unique_runs`
单列采集次数，防报告与训练 guard 不一致；Rep 重复采集回归通过。

流水线一致性补充：两个索引审计 CLI 与 `run_full_pipeline.sh` 显式传递 `action_output_mode`，防止 choice 在审计中误用 binary 同 RS 门槛。新增 CLI 回归通过；shell 语法检查与 `git diff --check` 通过。
