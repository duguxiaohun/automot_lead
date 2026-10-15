# 连续 batch 最新复审修正

按用户交回的四项问题，对 v60 主干复核并修正。Phase4 benchmark 的传输差异已在 v60 消除，本轮保留正式 encode_batch 路径并回归验证；不是再次改动正式模型编码器。其它修正如下。

## Phase3 学习率计划

通用 `sft_new_loop_phase3/train.sh` 与直接 Python 入口不再默认固定 warmup2000。未指定 `WARMUP_STEPS`／`--warmup-steps` 时，使用 `WARMUP_RATIO=0.1`／`--warmup-ratio 0.1`，对实际总微步除以累积次数后向上取整的更新次数计算预热，并保证至少留一次非预热更新。总更新为1时预热为0。

| 四卡、1024/event、3轮 | 总更新 | 新通用默认 warmup |
|---|---:|---:|
| B2，累积1 | 3840 | 384 |
| B4，累积1 | 1920 | 192 |
| B8，累积1 | 960 | 96 |

显式 warmup 步数仍优先，但必须非负且小于实际总更新数；无效配置在加载模型前拒绝，避免训练结束仍未完成预热。`train_phase3_batch.sh` 的四卡B2／累积2配置仍明确使用500次预热（总1920更新），不会悄悄切到192。缩短该入口的 epoch／MAX_STEPS 时，应一起调整显式 WARMUP_STEPS。总更新、实际预热、来源、比例记录到运行清单和 adapter 配置。

默认10%是通用入口的预算适配策略，未声称已通过GPU收敛对照；此前短吞吐benchmark并未验收学习率效果。实际scheduler CPU回归验证不同batch／累积下可达到峰值并进入衰减。

## 同一锚点的 action 补偿

构造连续段时保留每个锚点的全部合格问题，不再按ID提前丢掉替代题。合格仍先满足最少问题曝光、最少锚点曝光；段长优先级保留。选择同锚点问题时，考虑 action 当前欠额及剩余可选锚点数，逐个更新临时欠额，防止丢掉稀少action的仅有机会。相同评分仍按稳定ID消歧，输入文件顺序不影响计划；段间评分保留原策略。

采样策略升为 `event_equal_contiguous_unseen_first_v2`。旧v1曝光history明确拒绝，不混用旧history继续新策略。action仍是软补偿，不要求YES:NO等量，也不保证所有event都能达到软目标；不以重复稀少题或缩短连续段强行补齐。

真实既有1457题小池，seed20260929、四卡B2、16/event：两图和四图 U-E3 都从旧 `16 proceed/NO` 改为 `14 proceed/NO + 2 hold/YES`，恰好达到该event目标。同一批U-E3锚点、连续段成员和长度未变；两图中间两段交换顺序，不称整个序列逐位不变。两模式五轮累计不同问题覆盖仍为151／278／383／479／575。另检查1024/event正式预算的CPU计划；空留出小池不替代正式准入或GPU消费。

## 容量与可启动配置分别报告

`batch_probe` 保留原始容量结果 `conservative_measured_candidates`，但严格取所有被测GPU都实际测过且留余量的batch交集，不再用各卡最大值的最小值隐含推断其它batch。

新增 `training_compatible_candidates`（每个候选含对应accumulation）和 `training_exclusions`。默认目标训练world4、1024/event、Phase3每更新16／Phase4每更新32，与逐卡probe的GPU数量分开。候选必须同时满足事件配额和更新量能被 `world×B` 整除；因此容量能到B7不等于正式入口可选B7。Phase4固定入口只认可world4、1024/event。Phase3专用实测profile仍仅B2，其它候选需要通用训练入口。

可用 `--training-world-size`、`--training-per-event`、`--training-update-samples` 标明其它训练合同；这些参数仅用于可启动性筛选，不改变显存worker，也不自动改正式训练预算。所有候选都只是受测输入下的容量／配置交集，不是速度排名，更不等于DDP或全池GPU验收。

## 回归和合同边界

修复 `test_short_diagnostics` 对已移除 train.encode 的patch及旧单ID记录断言，改为实际encode_batch和ids字段；没有通过删除测试回避失败。新增同锚点稀少action、确定性、跨epoch曝光及旧history拒绝、warmup预算与实际scheduler、容量交集和正式入口排除测试。

共享采样源码改变学生合同：在最终同步本轮源码后，从现有有效v57 data4生成新学生view。原生producer、老师、标签、时间间隔不改；旧view和旧学生checkpoint不改SHA伪装兼容。不要重跑约20小时的原生建库。Phase3索引语义合同保持，使用已有正式索引开始新实验。

当前验证回执为 `batch_audit_fixes_verification_20261015.json`，新来源清单 `release_sources_v61_batch_audit_20261015.json`。用户已授权本轮白名单发布；完整Phase4实测命令见 [Phase4四卡实测](PHASE4_BENCHMARK_20261015.md)。无远程或目标4B GPU训练。历史四卡速度结果仅作为已测配置证据，不冒充本轮新采样策略的GPU验收。
