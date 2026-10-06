# v26 因果规则老师：审计范围与运行方法

本轮把 UE1/UE4 的“实例 → Episode 回放 → 原始规则标签 → 按规则类批准 → 数据集”接起来。
旧 `privileged_producer.py` 保留为假设状态候选/人工审核入口，其 UNKNOWN 不再是自动训练的入口。
新增正式自动入口为 `teacher_replay.py` 和 `teacher_data.py`。没有把旧候选直接改成 YES。

## 已实现与边界

- 身份读取 `role_name`、`scenario_obstacles_ids`、`vehicle_affecting_id`、`walker_affecting_id`，再要求当前可见局部冲突。UE1 仅选最近对齐前车；列入场景障碍的车辆不作为普通前车起点。来源名称不是许可证据。
- 每个实例使用真实 `Episode.questions/advance/confirm_execution`。许可未执行会过期重查；当前/历史自身位移只作执行回执。规则轨迹并不等于模型错误下的状态分布，也不是车辆闭环仿真。
- UE1 释放与稳定跟车分开；UE4 按参与者完整框、局部导航走廊和余量判断。默认余量 1m 加 0.25m 模糊带属于待校准老师参数。已停在走廊外的行人不能仅因不再向外走而重新成为冲突。
- 采集端 `lead/lead/expert/expert_base.py:distance_to_next_junction` 的 `+inf` 是“下一车道端点非路口”的哨兵。允许其参与局部通行判断；缺失、NaN、负无穷仍未知。它不证明整条路线没有路口。
- 第0帧、因果历史不足、负尺寸、可见消失/瞬移均有明确弃答处置。已注册风险路线仍需原有逐帧风险审核，自动编译暂排除这些路线并计数。
- 亮度/对比度与投影变化只是老师的弱视觉证据，不是已认证可见性。2/4图各有独立规则类。自动 catchup 暂不输出监督；自身运动能同步状态，但不能代替输入图中的可见后继证据。
- 其他八事件尚未实现自动老师。没有支持实例的帧明确记为弃答，不能宣称无事件，也不补 NO。重查/目标丢失/序列终止不等于事件完成。

## 批准不再是全局常量

每题保存 `rule_target`、`state_rule_target`、`rule_class`、实例/状态、实际因果 RGB、几何来源 SHA、弃答理由和老师源码身份。
`rule_target` 是模型输入范围内的老师候选；`state_rule_target` 可用于当前追踪器同步，但不能据此绕过 RGB 监督门槛。
批准记录绑定完整老师源码/参数、开发参考与独立参考。旧问题不能仅换新批准清单继续使用。

`teacher_registry_v1.json` 的真实批准列表为空。本轮没有独立审核人，也没有签发真实批准。
测试中的200条合成参考只验证放行/拒绝机制，绝非真实准确率。

每个规则类默认要求至少100个判断、20条物理路线、30个转移附近判断；YES 和 NO 精度的 Wilson 95% 下界分别不低于0.95和0.90。
参考 UNKNOWN 计入错误分母，不删除难题；只含一种答案不能通过。100只是总数下限，不保证两个置信下界能达标。
同路线相邻帧有相关性，Wilson 的逐判断区间不是路线级独立安全保证；报告应连同路线数、抽样范围一起解释。
独立评审需要盲审/独立声明，不得与规则作者同人，不得使用开发路线。身份声明本身无法从代码证明真人确实看过图。

40条预留路线保持原路线和 split；v6仅重新冻结当前源码。它们仍未审核，每事件原预留4条也不自动满足每规则20路线要求。
若不足，应在接触标签前扩展冻结抽样协议。不可把已用于规则修改的开发图算入独立抽检。
错误事件身份、错误状态或 RGB 无法判断的参考应填 UNKNOWN，而不是为了通过审批改成 NO。

## 全量生产与覆盖计算

按路线输出独立 JSONL 和索引，可用多个进程并行；完整产物可以断点复用，源码/清单变化拒绝复用。
每条路线的每一帧都必须恰好有一个处置，缺帧、重复、越界、产物哈希不符会拒绝。
`complete_causal_production` 根据实际索引与冻结全目录计算。全帧有处置不等于全帧有标签：全是 UNKNOWN 也不能通过正式训练准入。
本轮真实重放仅17条开发路线/1615帧，未运行9712条全量生产。

## 运行

从 `AutoMoT` 目录执行，使用安装好项目依赖的 Python；所有输出选择新路径。

```bash
python -m qwen3vl_local.sft_new_loop_phase4.candidate_pool \
  --data-root lead_data --output /path/new_candidates.json

# 默认处理全部冻结 train/val/test；--route-list 可做有明确范围的小样本回放。
python -m qwen3vl_local.sft_new_loop_phase4.teacher_replay \
  --candidate-pool /path/new_candidates.json --data-root lead_data \
  --output-dir /path/new_teacher --workers 4

# 此命令只核验已有真实评审，不生成或填写人工参考。
python -m qwen3vl_local.sft_new_loop_phase4.teacher_approval \
  --registry /path/reviewed_registry.json --output /path/new_approval_report.json

python -m qwen3vl_local.sft_new_loop_phase4.dataset \
  --candidate-pool /path/new_candidates.json --production-index /path/new_teacher/index.json \
  --teacher-registry /path/reviewed_registry.json \
  --annotations qwen3vl_local/sft_new_loop_phase4/reviewed_state_pairs_v9.json \
  --data-root lead_data --output-dir /path/new_data2 --rgb-mode 2
```

4图使用新目录并改 `--rgb-mode 4`。默认每路线最多100题，事件/边/答案轮转，转移附近优先；冲突可见输入整组排除，同答案重复去重。
后续训练沿用事件1:1和每帧重复上限8。也可给现有 `run_full_pipeline.sh` 设置 `PRODUCTION_INDEX`、`TEACHER_REGISTRY`、`CANDIDATE_POOL`，使用 `MODE=check` 验证再训练。

自动标签保存 `label_basis=approved_rule_teacher`、`reference_kind=rule_teacher`，不伪称人工 RGB 审核。
编译暂不消费40条老师抽检及旧保留人工评估路线；其他冻结val/test自动标签只报告老师一致率。
`evaluate.metrics` 顶层准确率/选优仅使用人工参考；自动参考单列 `teacher_consistency.agreement`。
若评估全是老师标签，没有独立人工分数，不能把一致率偷偷传给现有准确率选优器。

## 本轮验证与尚未完成

703项专项测试通过。17路线1615帧中112帧有规则问题，1503帧有明确弃答处置；2/4图各238道原始题：15 YES、97 NO、126 UNKNOWN。
其中 proceed readiness YES 为11道（UE1第72帧1道，UE4第52–61帧10道）。这不是新增获批监督。
与现有开发标注精确匹配的17题，每种图数均10一致、5弃答、2人工参考仍UNKNOWN（UE1第72帧、UE4第52帧）。后两题不是已证实错误，也不能算正确；不报告“100%准确”。
2/4图真实pipeline check通过，各902监督+539待审RGB核验；train443/val273/test186，四JSONL与v25逐字节一致。
详见 `twenty_sixth_audit_verification_20261002.json`。
只复看2条既有开发路线的6张原尺寸RGB，没有新增完整序列/独立标注/曝光组。
HardBreakRoute 1030_0 的68/70/72帧：72比68的前车距离变化更明显；70仍弃答。
Town07 Scenario3_1 的48/52/56帧：行人向左离开，但52帧没有清晰车道边界；不能凭这次已知规则输出的复看认证1m余量。

仍缺真实规则批准、自动 catchup 视觉老师、其余八事件老师、独立十事件评估和全量生产。
没有进行GPU训练、完整4B、DDP或CARLA闭环验收。当前人工题库仍可做限定评估范围的探索训练。
