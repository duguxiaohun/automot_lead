# Phase4 v36 稳定选题、配对训练与异常起点

本轮修复了2/4图训练比较中的题池混杂，并将新确认的不连续按发生帧登记。全部产物按最终源码重建；仍只支持有限弱监督实验，不能据此批准完整十事件正式训练。本机默认4B基座缺失，未运行GPU训练或CARLA。

## 实现

- `teacher_data.selection_key`按路线、参与者实例、状态、分支、当前帧、边、阶段和答案稳定排序，不再用包含图数/源码身份的`question_id`决定逐路线限额。输入可见性、风险及人工冲突仍各模式独立校验，因此不承诺原始两题池自动完全相同。
- 新增显式`--paired-with`训练视图。两套数据先分别通过完整加载合同，然后取同ID且语义、答案一致的训练交集；核对2图历史是4图历史子集，共享图路径、文件SHA及RGB像素SHA一致。缺题/标签不一致单列排除，证据损坏直接拒绝。不会从一个模式复制标签到另一个模式。
- 配对子集重新计算准入和覆盖；双方manifest摘要、配对身份摘要写入训练合同，恢复训练时校验。主机预检使用同一配对参数。各自完整val/test保持，评估继续逐题遍历，不作1:1训练抽样。
- 新风险可显式提供`last_unaffected_frame`与`first_affected_frame`，仅在相邻原图证据和确认状态齐全时使用异常起点。旧登记仍沿用原包络/审核约束；新记录仍核对完整因果历史和更早的STOP证据，包含图间隔。

## 重建结果

全8614路线/1062401帧完成逐帧回放，train/val/test仍7268/658/688；物理组隔离和原split保持。新产物`AutoMoT/checkpoints/phase4_v36_full`，合同v36、老师规则v8、独立预约v16、严格注册表v11为空；v35产物及manifest摘要保持。

训练2/4图分别75489/75489题；443道原人工训练题逐行保留，val273/test186/review539逐字节保留。8614条路线的原始老师答案计数及逐帧处置均与v35一致，说明本轮没有通过改老师条件增加正例。严格配对交集75489题，配对外分别0/0题。非配对题仍保留在各自完整题库。

配对训练在world_size=1和4下各跑七轮计划，两种图数每轮题目ID顺序完全一致；每轮400次、十事件各40次，七轮覆盖705/75489题（0.9339%）。这是有限采样预算，不是全题池遍历；未调高重复上限或虚构缺失答案。

## 风险、曝光与实际目视范围

接入只读审计的26个训练物理组，其中6个为新曝光组；全部原属train，未改变划分。新增9条确认不连续风险，8条相邻证据使用精确异常起点，Town02 Scenario3_5的114–116聚合证据保留完整保守窗口。

本代理仅复看Town07 Scenario3_10的97、98两张训练原图：97行人仍可见，98才消失。97帧YES的完整老师证据止于97，两模式及配对视图均保留。新窗口与旧v35相关训练题的实际异常后因果历史重叠为0，没有因为未来异常撤回当前标签，也没有把弱标签升级为人工认证。

26序列/4671帧/45原图来自用户提供的只读审计，本次没有重新完整目视这些序列。继承229/278格三ID筛查、140条/27082帧待审、18格源不足；无新增完整序列认证、人工标签或独立RGB审阅。

## 使用与验收

纯图数消融须给两个运行互指的配对题库，使用相同seed、采样策略、预算、epochs和world_size。以下只检查采样，不启动GPU：

```bash
P4=/home/codon/automot_lead/AutoMoT/qwen3vl_local/sft_new_loop_phase4
DATA=/home/codon/automot_lead/AutoMoT/checkpoints/phase4_v36_full
PYTHON=/tmp/automot-qwen35-env/bin/python RGB_MODE=2 bash "$P4/train.sh" check --paired-with "$DATA/data4"
PYTHON=/tmp/automot-qwen35-env/bin/python RGB_MODE=4 bash "$P4/train.sh" check --paired-with "$DATA/data2"
```

正式比较仍须同一基座、训练配置及配对评估，以上只证明数据/计划配对，不是模型效果结果。

920 passed in 80.35s (0:01:20)。2/4实际pipeline check、全部训练/人工评估/待审RGB身份核验、world1/4七轮计划、真实配对启动和预检通过。预算10000配对预检均退出2；默认host-preflight因缺MODEL_DIR退出2。1037张独立盲审卡全部未填，未看独立RGB。

关键UE2绕行/返回、不返回/本车道通过分支，UE4绕行、RE2进入、UE7推进正例及六事件人工val/test缺口保留。严格老师批准0，formal_data_ready=false；没有新增门槛，也没有把测试通过当作驾驶能力认证。

证据：`verification.json`、`replay_semantic_comparison.json`、`paired_sampling_verification.json`、`paired_preflight2.log`/`4.log`、`branch_support2.json`/`4.json`及`final_source_acceptance.json`。全部源码及依赖保存在`frozen_source`；验收仅对应所记录SHA，后续修改必须重新验收。
