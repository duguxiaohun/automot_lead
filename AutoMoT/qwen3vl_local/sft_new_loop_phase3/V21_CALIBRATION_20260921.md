# Phase3 v21：已确认起步、场景语义与分布审计

同日后续全量容量修订见 [CAPACITY_AUDIT_20260921.md](CAPACITY_AUDIT_20260921.md)：v9动作/prompt不变，补齐未曝光holdout来源，mapping合同更新；不能继续复用此前仅局部回放产物的哈希。

新训练使用 `sft_new_loop_phase3_data_v21`、`v21_confirmed_pullaway` prompt 和
`current_wait_first_crossing_v9_confirmed_pullaway` 标定规则。默认仍为 4rgb + choice，入口 seed 为 20260920。
这是根据 [噪声审计](DATASET_NOISE_AUDIT_20260921.md) 落实的规则修订，不是模型效果报告。

## 1. 起步与等待的可执行合同

此前两个近零速度采样无条件优先返回 STOP，覆盖了已给油并开始起步的帧。
现在只在原 `current_confirmed_wait` 分支之前增加已确认起步例外；其余速度、首次跨线及域完整性规则保留。

令 `v[0:9]` 是当前至 +2s 的完整速度，`delta=max(1.2, 0.2*max(v[0],1.0))`。
`g` 为原规则首次满足 `min(v[g],v[g+1])-v[0] >= delta` 的下标，范围 1…7。
同时满足以下条件，纵向改为 RESUME，原因 `current_confirmed_pullaway`：

1. 当前与下一帧均不高于 0.5m/s，原规则本会给 `current_confirmed_wait`。
2. 当前明确 `brake=False`，且有限、合法的 `throttle>0.1`。
3. `v[2]>v[0]`：最迟 +0.5s 已开始增速；`v[8]>=2m/s`。
4. 原两点增速确认存在；`k=0…g` 全部满足 `v[k+1]>=v[k]`，零容差。
5. `k=g+1…8` 全部满足 `v[k]-v[0]>=delta`，允许确认后调速，但不能放弃显著净增速或再次近停。

九点之外的数据不改变标签。沿用 `_future_speeds` 对 [-0.05,0) 微小负速归零的原处理；
不把展示用三位小数或 0.001m/s 容差用于标定。仅控制释放、后来加速、缓慢蠕行、缺控制或缺窗口均不能触发例外。
控制缺失/非法保留 null；不能用 `bool(None)==False` 或默认油门零伪造证据。

**为什么没有把“全窗单调”的25帧筛查直接当规则？**
HardBreak/Town12_1254 f73 在严格子集中；Construction/Town12_1256 f84 却在起步后从
7.369 回调至 5.820m/s，会被全窗单调排除。此前连续 RGB 已确认后者实际起步并左绕施工障碍。
新合同允许这类起步后仍保留显著净增速的调整；同时保留25帧严格子集作为诊断与回归，
而不是将25/4262当作错标率或完整的起步定义。

`label_actions`、`action_evidence`、`action_review` 和边界审计共用 `longitudinal_from_signals`。
近停速度对仍记为 `current_near_stop_pair`，已确认起步时 `stop_qualifies=False`，避免审计又把它写成有效 STOP。
审计时刻仍是数值判据触发点，不是完整行为阶段真值。录制初期起步另标 `pullaway_with_padded_startup_history`；
未通过例外但已释放控制的近停帧单列 `released_near_stop_without_confirmed_pullaway`。

主要动作优先级仍为 STOP > 首次跨线 > 速度。故 STOP+LEFT 改为 RESUME+LEFT 后，choice 是 LEFT；
binary 同时保留 RESUME:YES 与 LEFT:YES。没有把所有起步组合强制投影为 RESUME。

## 2. 十类场景与模型输入

| 范围 | 最终处理及依据 |
|---|---|
| 所有场景 | 两种题型共用等待/立即起步语义；事件可跨接近、响应和恢复，事件持续不要求重复已完成动作。条件性目的不是实际动机或安全空隙真值。 |
| LEAD_BRAKE | 情境改为响应此前制动/减速的前车，包括后续等待与恢复；不在持续静止帧反复断言“刚刚突然刹车”。 |
| STATIC_BLOCKAGE / POST_BYPASS_RETURN | 保留真实首次跨线、绕出/恢复由历史判断、目标符号不能选左右。v20 两处精确 lane-section 隔离继续生效；撤回的四条候选不新增隔离。 |
| DYNAMIC_CUTIN / VULNERABLE_CROSSING / ONCOMING_INVASION | 保留原明确作用于 ego 路径的 scope 和已确认逐帧修正；不把雾暗或暂不可辨直接改成 invalid，也不从个别路线的提前量推广整类。 |
| JUNCTION_RULE_CONFLICT | 保留给定优先权冲突语义；画面中的横穿车不必就是记录的约束对象。CrossJunctionDefectTrafficLight 不套用另一个 scenario 的 32m/26m 门槛。 |
| SIGNAL_FAILURE | 提示词明确故障为场景前提，任务是条件下预测 ego 动作。单个绿灯不能推翻故障，多色灯头需要进口归属才能证明冲突双绿。mapping 另写场景/RS审核范围及查询 None 语义，不声称逐帧物理激活已验证。 |
| UNSIGNALIZED_PRIORITY / RAMP_MERGE_EXIT | 继续要求当前决策受路口/合流出口影响；不将最近路口30m代理或另一Town的低flip率做成硬过滤。已有出口箭头与分流证据保留。 |
| 导航 | `Final destination` 改为 `Recorded route endpoint`，说明坐标随 ego 移动/转向变化，y符号不能指定下一次跨线。坐标计算不变，不再保证等于原XML末点。 |

仍只输入原 RGB 历史、当前速度、导航和场景文字。油门、制动、未来速度、阈值及审计分支不进入提示词。
十类动作目的与 KEEP 语义保留：KEEP 需题域证据完整，允许阶段内调速，不表示风险已消失；
invalid、缺失和横向歧义不能回填 KEEP。未根据10.0%的有效相邻帧变化率增加标签平滑或更改20%速度阈值。

U-E7 的冻结 tick、物理灯头归属和原始停止线位置仍未闭合。位置代理与灯态查询缺失的强关联，
不构成故障状态等价关系；不因此批量删393帧、改事件或伪造“看不见灯”的标签。
风险过滤维持来源可信度门控，不把它称为 RGB 可见性检验，不按夜间/雾天统一删样本。

## 3. 分布及曝光合同

新增 `primary_action_distribution`：STOP+LEFT/RIGHT 一并投影 STOP；INVALID 独立计数，同时给出含/不含 INVALID 的分母。

- manifest：`sampling.balance.<split>.sampled_primary_action_distribution`，标记 `index_rows_before_epoch_resampling`。
- `train_balance.json`：`train.primary_action_distribution`，明确第一轮采样快照。
- `balance/epoch_*.json`：对应轮次的相同字段；`--sampling-only` 也直接输出第一轮主要动作分布和路线覆盖。
- 路线支持报告补充 Town 列表、Town数与 scenario数，结合既有每路线最大题数，暴露结构性覆盖缺口。

原审计包的 4648/14244=32.6%（含INVALID）、4648/11870=39.2%（有效索引），
与 (3270+262+222)/10240=36.7%（epoch采样）是不同层，不能混报。
这些是旧审计包数字，不是新v21生产分布。`resample_each_epoch=True` 和 `seed+epoch*1000003` 继续生效；快照不是累计训练分布。
签名均衡仍是容量封顶后余额回流到有容量的签名；不靠复制稀有签名实现不可达的动作均衡，不把原U-E6的82.6%说成重复稀有签名造成。

242条已用于本次规则开发的路线写入 `development_route_groups_noise_20260921.json`。
与历史名单合并新增155个物理组，总计1609组 train-only；Rep/重复采集不能绕过物理组隔离。
没有改现存生产 split 或索引文件；新构建使用这项政策。

## 4. 验证及实际影响

独立的v20缓存与原meta回放：242条路线、31,668帧；397帧纵向 STOP→RESUME，其中303帧没有目标 context。
带context的94帧含全窗非递减严格子集25帧，另69帧允许确认后调速；29/94属于f0–f2的录制初期，不能全部称为路中等待后释放。
风险与导航筛查后86帧；再按真实构建器的完整题域、历史、映射等门控，实际候选变更84帧：
64个主要动作为RESUME、18个LEFT、2个RIGHT。横向布尔位及其它原始动作标签无变化。
这些是规则变更数量，不是新增逐帧RGB确认的错标数量。

本轮以先前927张不同RGB/37路线的复审为视觉依据，没有将242条数值回放写成242条完整目视。
一般化的起步规则需要后续新数据/模型验证，不能据此保证解决 STOP 偏向或提高成绩。

- Phase3 全部587项测试通过；含起步、等待、缺控制、速度回落、窗口末端、主要动作、四种prompt组合及CPU分布验证。
- Action 数据准备/共享动作 token 相关63项测试通过；在已有pvi环境禁用CUDA执行，无Qwen训练或GPU效果验证。
- 原始数据局部构建7900候选→384行训练开发索引；146条路线原始证据回读、1408次题型/图数重放一致。
- 开发索引仅有train。生产preflight按预期拒绝缺val/test，未关闭或放宽检查；开发行回放工具不替代生产验收。

持久化脚本：`audit_v21_rule_delta.py` 对比旧缓存与原meta；`audit_development_replay.py` 核对每行原始证据、导航、控制、动作、review及prompt。
25帧精确回归数据在 `pullaway_regressions_20260921.json`；本次结果摘要与变更位置在 `V21_REPLAY_20260921.json`。
大规模缓存与局部产物保留在 `/tmp/p3audit/v21`，临时目录不是长期数据存储承诺。

```bash
# 在AutoMoT目录；原审计缓存路径由调用方提供。
python qwen3vl_local/sft_new_loop_phase3/audit_v21_rule_delta.py \
  --baseline-dirs /tmp/p3audit/cases /tmp/p3audit/cases2 /tmp/p3audit/cases3 \
  --data-root lead_data --output /tmp/p3audit/v21/replayed_delta.json
python qwen3vl_local/sft_new_loop_phase3/audit_development_replay.py \
  --index /tmp/p3audit/v21/final_index/frame_index.jsonl \
  --data-root lead_data --output /tmp/p3audit/v21/development_replay.json
```

## 5. 新训练和恢复

运行原 `run_full_pipeline.sh` 会采用v21目录及新prompt；先全量构建并完成生产preflight，再新开训练。
不要给旧v20索引手改version/hash，也不要用 `SKIP_BUILD=1` 跳过必要重建。
旧run必须使用其原源码及原索引恢复。Action若消费共享Phase3 candidate/full map，也必须重新生成绑定新源码与划分的产物；
比较条件需共用相同数据版本、split、seed和预算。本轮没有改Action训练入口，没有全量生产重建、GPU训练或新模型提升结论。
