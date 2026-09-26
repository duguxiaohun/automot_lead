# SFT New Loop Phase3 当前运行入口（2026-09-26，v24）

## 2026-09-26 v24 当前默认（覆盖下方历史版本默认）

新训仍默认 4rgb + choice；数据目录改为 `sft_new_loop_phase3_data_v24`，提示词为
`v24_motion_reference`。只澄清最新速度基准和短暂近停，动作 v9 阈值、窗口、主动作优先级及采样策略保留。
近停连续片段诊断只供审计，新增176个已曝光物理组为train-only（累计1955）；历史数据容量回放可补齐holdout，非生产重建验收。
必须重新构建索引/full map并新建run，不能用SKIP_BUILD复用旧哈希或用新源码续训旧run。

760项相关CPU检查通过，460题原始动作回放不变、1840次提示词回放通过；未跑新GPU训练。
训练历史对比及60段/1020张RGB逐帧证据见 [v24复核](V24_RGB_CALIBRATION_20260926.md) 和
[训练审计](AUDIT_COMPARISON_20260926.md)。下方v23及更早记录均为历史说明，当前路径以上方为准。

```bash
# 从 AutoMoT/ 运行；用默认新目录构建并训练，不复用旧SKIP_BUILD设置
bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```


## 2026-09-23 采样接线修复后的新训默认

默认 `--sampling-policy smooth_cap --sampling-repeat-cap 8 --sampling-smooth-power 0.5`。
构建器额外发布完整 `train_sampling_pool.jsonl` 并在manifest绑定SHA256；新训练读取全部train正例，
不再只能在旧均衡索引截取的子集上轮转。验证/测试继续读取 `frame_index.jsonl`，没有训练游标。
十context预算不变，动作温和提权、共享帧全epoch限次；binary自动INVALID按原分层数量联合分配。
`--sampling-policy cycle_even` 可作旧选样对照；`--max-frames` 显式截断后只承诺覆盖保留池。

启动脚本支持 `SAMPLING_POLICY`、`SAMPLING_REPEAT_CAP`、`SAMPLING_SMOOTH_POWER`，CLI优先。
训练manifest、adapter配置保存实际设置和训练池身份；逐轮balance文件记录起止游标、目标/实际动作配额、
重复直方图及容量回流。同成本联合分配按未选轮数和每帧累计曝光公平分配，`pools` 保存子池历史，
跨轮同步推进 `next_pool_history`。Phase3没有optimizer断点恢复入口，不把adapter当作完整恢复checkpoint。

完整管线的 RGB/原始meta审计显式要求带哈希训练池，检查完整train池＋原val/test；
文件/物理帧IO去重，不丢弃不同上下文或冲突证据。两份报告的 `input_coverage.sources` 分来源
记录行数、独立帧和case数，`counts_scope=original_index` 区分原索引预检计数。单独运行两个审计
程序也会自动发现manifest训练池；传 `--include-training-pool` 可强制要求，缺池拒绝继续。
必须重建当前v23索引（不要SKIP_BUILD复用旧hash产物）并新训，旧run使用原源码。
详见 [实现、约束与验证范围](HIERARCHICAL_SAMPLING_PLAN_20260923.md)。


## 2026-09-23 当前默认：有效历史、主要动作额度、物理路线支持

新训练默认 `4rgb + choice`、`sft_new_loop_phase3_data_v23`、`v23_grounded_stage` 提示词。
anchor<4 的初始化历史统一排除；稀少纵横组合并入主要动作采样额度，保留原始 YES/NO 标签。
候选 holdout 默认每事件32帧、5物理组，已曝光1779组仍 train-only；所有 split 优先物理路线轮转。
Action/两个消融共享初始化过滤、主要动作容量回流及独立 Action split 支持补齐。
final 权重新增独立验证记录；旧run须原源码，新索引/full map/新run，不能沿用旧缓存或修改hash。
报告、回放数量和验证边界见 [v23 RGB与支持量审计](V23_RGB_SUPPORT_20260923.md)。

```bash
# 从 AutoMoT/ 运行；自动构建 v23，然后新训
bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```

以下按日期保留历史说明；涉及 v22/v21 的默认值以本节为准。

## 2026-09-22 支持量审计补齐

构建报告的 `signature_support` 同时列出候选帧数、去除Rep/采集后缀后的物理路线数、实际呈现配额，
低于100帧或10条路线标为复查线索；这些阈值不参与采样、标签、过滤或holdout分配。
合法稀少动作保持真值，不能因为出现少就变成KEEP；Phase3没有把低频动作变成UNCOND答案的接口。
Action主线/两消融使用相同诊断，并把新action-balanced默认全局重复上限降到2；
这不是Phase3重复上限：Phase3仍按自身事件预算做完整池循环和余量回流，默认自动预算不因本轮诊断改变。
当前仍为v22目录，采样/构建源码hash已更新，已存在的旧hash产物须重建；旧run使用原源码。


## 2026-09-22 稀少动作容量回流

默认索引改为 **v22**；动作规则、KEEP语义和v21提示词保留。构建器、binary/choice训练及其均衡验证计划
共用容量回流：事件内预算不超过池容量时，稀少动作最多抽其现有帧数；超出时循环完整池，再分配余量。
例如KEEP 4帧、STOP 1000帧，预算100抽4/96；预算1500抽8/1492。取消整桶随机补额，不专门放大稀少格子。
不因稀少改成KEEP/UNCOND；未观察到的动作不合成，INVALID覆盖及物理split隔离不变。
Action/两个消融同步使用此函数，并只把全局动作分入支持它的事件域；RE5不再产生右变道采样格。

新 `sampling_policy` 写入manifest/训练配置，采样源码绑定mapping hash；旧索引必须重建，新条件新训。
Action自动准备按新hash建独立缓存，显式旧full map会拒绝。旧run使用原源码，不用修改manifest绕过校验。
Phase3默认pipeline会构建新v22目录，无需新增开关；不要用 `SKIP_BUILD=1` 复用旧v21。
当前验证为CPU回归及旧候选池数值回放，未全量生产重建、未跑真实GPU。

```bash
# 默认四图choice，构建新v22索引并训练
bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```

以下v21容量数字是历史构建记录；新的生产计数以v22 manifest为准。

## 2026-09-21 后续：全量容量检查与划分补齐

全量检查发现旧哈希划分在开发路线隔离后缺少多个 val/test context。当前 v21 构建器只从未曝光 train 物理组补容量，默认 `--min-holdout-context-frames 32`，同组所有 Rep/context 一起移动，开发路线仍 train-only；实际比例及移动清单见 `split_coverage.json` / manifest。源容量不足仍失败，不靠重复冒充新样本。新源码哈希须重建索引、另开 run。

最终全量构建为 train/val/test=11580/408/384；十个正例 context 各965/34/32行，INVALID=1930/68/64。choice/binary 七轮及默认16/32验证预算已回放通过。训练呈现仍可按既有规则重采样；帧数不等于物理路线支持，四类 holdout 仍只有一个物理组。完整结果、运行边界见 [容量审计](CAPACITY_AUDIT_20260921.md)。

默认 **4rgb + choice**，索引 **sft_new_loop_phase3_data_v22**，split seed **20260920**。
v21 使用 v9 已确认起步规则：近零速时以明确释放的控制和持续起步轨迹区分 RESUME 与等待；
允许起步确认后的调速，保留原速度阈值、有限窗口和主要动作优先级。模型不接收控制或未来数值。
两种题型同步事件持续阶段、故障前提与记录终点措辞；分开报告索引和每轮采样的主要动作分布。
完整规则、影响范围和复现说明见 [V21_CALIBRATION_20260921.md](V21_CALIBRATION_20260921.md)。

模型每次输出一个主要动作，包括明确的 **KEEP**；纵向域三个变化动作加 KEEP，机动域五个变化动作加 KEEP。
`scene_context` 仅描述道路、事件和已提供的历史；动作选项按十种上下文分别给出一至两句因果说明。
UE2 减速/停车同时保留防碰撞、观察邻车与间隙、为可能绕行创造时机的联系；不是已经有安全空隙或必定变道。
v16 删除全部未来数值时间窗、采样数和速度判定公式，模型判断主要的后续动作；历史RGB时间只描述已观察的输入。
v17 将显式 KEEP 同步到判断题的提示词、标注、解析和评估；协议说明见 [V17_BINARY_KEEP_20260920.md](V17_BINARY_KEEP_20260920.md)。
v18 复看十类26片段/506帧面板（补64帧早期历史），精确隔离两段 UE4 共35帧；
动作说明统一条件、动作和作用，选择题明确主要机动，判断题说明纵横动作可先后发生。
新边界审计只分桶报告，不修改速度阈值或自动过滤 KEEP；当前说明见 [V18_RGB_CAUSAL_REFINEMENT_20260920.md](V18_RGB_CAUSAL_REFINEMENT_20260920.md)。
v18 同日复审修复：RE5 RESUME 同时覆盖等待后起步与行进中持续增速；边界审计支持训练验证的嵌套道路字段，
报告匹配/漏配数量并对全漏配告警。该次修复当时沿用 v18；随后升级为下述 v19；当前新训练使用 v22 索引及 v21 提示词。
v19 进一步复看 5 片段/91 帧面板，覆盖 12 条 KEEP 伴随车辆风险与零目标速度请求的题。
十类 KEEP 明确允许阶段内短暂制动、不能解读成风险已消失；所有新候选和索引补充独立 `action_review`，
记录控制响应、约束对象和原动作判据触发时刻，仅供审查，不进入模型输入或改变答案。
边界审计另设 response/eval_response 桶，详见 [V19_RECORDED_RESPONSE_20260920.md](V19_RECORDED_RESPONSE_20260920.md)。
去除未来数值倒计时的设计沿用 [V16_DIRECT_NEXT_ACTION_20260920.md](V16_DIRECT_NEXT_ACTION_20260920.md)。
标注设计及此前26片段/442帧面板复核见 [V15_CAUSAL_KEEP_20260920.md](V15_CAUSAL_KEEP_20260920.md)，其中v15提示词是历史版本。

离线标注中，KEEP 允许小幅调速：机动域还要求未来3秒无新跨线；纵向域只表示速度阶段保持，不断言车道保持。
已完成变道后事件可以继续成立；缺证据、歧义窗、错误前提不能变成 KEEP，当前持续等待仍为 STOP。
v9 保留原轨迹窗口/速度阈值、STOP > 首跨 > 速度的优先级、异常 route 过滤和物理路线隔离；
主要跨线前可以先减速，当前标签不等于逐阶段预测最先执行的反应。
原始动作布尔证据留作审计；binary 增加独立 KEEP 行，合法保持不再用全 NO 隐含表达。
KEEP 与所有变化动作互斥；invalid 时包括 KEEP 在内的所有动作均 NO。默认 choice 只输出一个动作名称。
KEEP 的 support/precision/recall 参与 best 守卫；因果说明属于公开场景知识，不是逐帧驾驶员意图真值。

必须重建 v22 索引并新训，旧 run 用原源码恢复；不要用 `SKIP_BUILD=1` 复用旧索引。
当前 Phase3 沿用 v18 精确隔离规则并完善提示词与审查证据；action_prior 保持旧 NONE/门控协议，其 oracle 继续读取原始布尔动作证据。
不能把 v17 KEEP 文本直接塞进旧 action_prior 外部预测接口；该接口会按不支持的输入拒绝。
历史曝光隔离继续生效；v20 累计1454组，v21 本次242条开发路线再新增155个物理组，总计1609组 train-only，禁止复审后仍当盲测。
v20 完成57片段、964帧面板（874张不重复RGB）的视觉复核；局部原始数据重建/CPU验证与限制见本轮报告，尚未重建全量生产索引或训练模型。

直接重新训练并自动评测（默认自动选四张空闲 GPU）：

```bash
bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
# 相同配置，显式指定卡：
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```

其它组合分别新训，默认自动选四张空闲 GPU：

```bash
# 2RGB + choice
HISTORY_RGB_MODE=2rgb_endpoints bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
GPU_IDS=0,1,2,3 HISTORY_RGB_MODE=2rgb_endpoints bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
# 4RGB + binary
ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
GPU_IDS=0,1,2,3 ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
# 2RGB + binary
HISTORY_RGB_MODE=2rgb_endpoints ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
GPU_IDS=0,1,2,3 HISTORY_RGB_MODE=2rgb_endpoints ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```

流程复用已有审计记录并执行自动索引一致性检查，无需新一轮人工逐帧观看 RGB。
仍按 validation 选择 `best_generation`，不硬编码历史 step 6000。

从 `AutoMoT/` 工作目录，先构建并检查新索引（使用已有 PyTorch 环境）：

```bash
python qwen3vl_local/sft_new_loop_phase3/build_dataset.py
python qwen3vl_local/sft_new_loop_phase3/preflight.py \
  --index checkpoints/sft_new_loop_phase3_data_v22/frame_index.jsonl
python qwen3vl_local/sft_new_loop_phase3/train.py --sampling-only \
  --index checkpoints/sft_new_loop_phase3_data_v22/frame_index.jsonl \
  --focus-balance-count 1024 --eval-balance-count 16 --generation-eval-balance-count 32
```

`preflight` 报告 binary val/test 同 RS 错事件的独立物理路线数；不同 Rep/采集时间不增加支持。
不足两条标记 `insufficient_support`，允许训练，该子组不参与 checkpoint 守卫；不能宣称事件拒绝评估已充分覆盖。
`--sampling-only` 再核验实际生成验证采样；不读取模型权重或初始化 NCCL。默认 choice，检查 binary 时显式加 `--action-output-mode binary`。
自动错误 RS 负例仍是必需训练桶；来源/真实道路/事件缺失、标签错误和物理路线泄漏继续报错。
人工负例保留为补充诊断，无需为了启动训练补盲审路线；独立评测不会把缺少这项证据的模型标成 `production_ready`。

预检后新训；已构建且通过合同检查时可显式复用索引：

```bash
SKIP_BUILD=1 bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
GPU_IDS=0,1,2,3 SKIP_BUILD=1 bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```

不设置 `SKIP_BUILD=1` 时 pipeline 默认重新构建。独立 eval 默认 `CASES_PER_BIN=0` 全量；choice 保留有效单动作、组合投影与 KEEP；仅排除 invalid。
`INDEX`、`DATA_DIR`、`SPLIT_SEED`、`CASES_PER_BIN` 等已有环境变量会覆盖默认值，启动前检查。
历史09/16负例曾增加 val 2/test 4 个物理路线，仅覆盖 R3 的两个错事件；新划分支持需重验，不能沿用旧计数。
历史及本轮曝光路线均为 train-only。下面按日期保留历史说明，旧索引题数不代表 v21。
两图/binary 对照按上方示例显式设置并分别新训；无需常规跑四组矩阵。

## 2026-09-20：自动 INVALID 组合均衡及人工事件负例覆盖

当前实现和重跑方法见 [INVALID_COMBINATIONS_20260920.md](INVALID_COMBINATIONS_20260920.md)。
自动枚举安全的错误RS/事件组合，不再只取第一个错误RS；按来源、真实RS/事件、错误RS分层均衡。
同一来源至少保留一个已有自动负例，避免后续验证增大预算时只能重复人工题。小预算可能继续自动上调；
五个人工题集中于一个来源的回归由旧41提高到51，以保留该来源的自动负例。无需手动调整目标数量。

## 2026-09-20：构建扫描结束后 INVALID target=30 / feasible_target=41

这是索引构建中的配额不足，与训练时的判断格式无关。当前确定性覆盖方案需要 RE5 来源5条，
其余9个来源在最大差1的约束下至少各4条，故该方案可行总量为41。
它不证明数据缺失、RGB错误或 KEEP 错标；也不宣称41是所有可能覆盖方案的全局最小值。
此前 `train._validation_work` 已对验证采样增容，但 `build_dataset._balanced_invalid_rows` 尚未接入，
因此全量扫描后会终止。局部冒烟原 INVALID 预算64，没有触发这个分支。

现在构建器仅捕获 `InvalidQuotaError` 并按可行预算重试，输出例如：
`[new-phase3-build] split=val INVALID quota expanded 30->41; source_seed_counts=...`。
这里的 val 只是示例，实际以新日志的 split 为准。正例配额不变，未发生不足的桶抽样保持原样；
缺来源、坏签名、同RS人工题不足等真实数据错误不被吞掉。
`invalid_ratio` 仍表示请求比例；实际请求和有效数量记录在
`manifest.sampling.balance.<split>.requested_target_invalid/target_invalid` 及 `invalid_balance.quota`。

新增6项纯CPU构建回归，包括真实入口30→41、重复随机种子一致、缺源/坏签名仍失败、
充足配额不重试、上层长度断言及报告使用41。Phase3共459项CPU测试通过，
4项取消选择、3个torch模块未收集；局部1148候选和384行除mapping哈希外一致，无全量/GPU结论。

失败时构建器已经保存 `candidate_frames.jsonl`、`candidate_counts.json` 和人工候选文件，
但构建源码参与 mapping 合同。更新本修复后应重建一次，不手工改缓存哈希绕过一致性检查。

四组对照的完整命令如下（从 AutoMoT 目录运行，默认自动选卡）。
仅第一组构建索引，后续三组复用；`&&` 在任何一组失败时停止：

```bash
HISTORY_RGB_MODE=4rgb ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh &&
SKIP_BUILD=1 HISTORY_RGB_MODE=2rgb_endpoints ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh &&
SKIP_BUILD=1 HISTORY_RGB_MODE=4rgb ACTION_OUTPUT_MODE=choice bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh &&
SKIP_BUILD=1 HISTORY_RGB_MODE=2rgb_endpoints ACTION_OUTPUT_MODE=choice bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```

对应显式指定四张卡的写法：

```bash
GPU_IDS=0,1,2,3 HISTORY_RGB_MODE=4rgb ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh &&
GPU_IDS=0,1,2,3 SKIP_BUILD=1 HISTORY_RGB_MODE=2rgb_endpoints ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh &&
GPU_IDS=0,1,2,3 SKIP_BUILD=1 HISTORY_RGB_MODE=4rgb ACTION_OUTPUT_MODE=choice bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh &&
GPU_IDS=0,1,2,3 SKIP_BUILD=1 HISTORY_RGB_MODE=2rgb_endpoints ACTION_OUTPUT_MODE=choice bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```

不要全局设置 `SKIP_BUILD=1` 跳过首次重建；如设置了 DATA_DIR/INDEX 等环境变量，确保四组指向同一份新索引。
默认题型已是 choice，前两组省略 ACTION_OUTPUT_MODE 会重复训练两次 choice。

## 2026-09-15：启动时报 INVALID quota 不足

`INVALID quota cannot retain reviewed same-RS context STATIC_BLOCKAGE` 可能来自小验证预算，
即使候选齐全也会发生：人工同 RS 负例集中在少数 source，原先随机分配的 source 配额不够保留它们。
现在先规划人工题、真实 RS 和问题域覆盖，再把整数余数分给需要它的 source。
仍要求 source 数量最大差 1、保留已有人工问题域、每个同 RS 人工输入最多使用一次。

若预算仍不足，loss/generation 验证默认分别自动增容，保持十类各 N 条、INVALID 2N 条。
只捕获明确的 `InvalidQuotaError`；缺类别、缺来源、缺 RS/问题域、坏签名仍报错。
规划给出确定的可行预算，不保证全局最小；训练预算和独立 eval 的预算保持显式配置。
choice 无 INVALID，不因该逻辑增容；两帧/四帧使用相同采样规则。

先对已构建的远端索引运行只读 CPU 预检（使用已有 PyTorch 环境）：

```bash
python qwen3vl_local/sft_new_loop_phase3/train.py --sampling-only \
  --index checkpoints/sft_new_loop_phase3_data_v9/frame_index.jsonl \
  --focus-balance-count 1024 --eval-balance-count 16 --generation-eval-balance-count 32
```

用 `--history-rgb-mode 2rgb_endpoints` / `--action-output-mode choice` 可检查其它组合。
自定义训练参数时，预检须传相同 index、split、seed、采样预算和截断参数。
预检共用正式采样路径，不读取 RGB/模型权重、不建立 NCCL 进程组、不创建训练目录；
它验证的是索引合同和采样，不检查图像可读性或 GPU 推理。

随后可照原 pipeline 命令启动新的 run，`AUTO_EVAL_BALANCE_COUNT=1` 默认生效，无需手工猜预算。
严格配对实验可用 `AUTO_EVAL_BALANCE_COUNT=0`，并显式给所有 run 相同的
`EVAL_BALANCE_COUNT` / `GENERATION_EVAL_BALANCE_COUNT`；Python 对应 `--no-auto-eval-balance-count`。
预算不足时应据错误中的 `feasible_target` 上调，不关闭覆盖检查。

启动日志 `[validation-balance]` 和 `validation_sampling`（train_balance、run metadata、adapter config）
保存 requested/effective、是否增容、原因、各类实际呈现数与独立题数；自由生成仍去重计分，
增容后的呈现比例不代表去重后的评分比例。增容会增加验证耗时，按实际 generation 进度评估超时预算。
本次仅 CPU/合成验证，远端原索引及真实 Qwen/DDP 训练尚待验证；标签、prompt 和权重未改变。

> 2026-09-11：收到20260910四图结果：production 518/765，审计见 [AUDIT_SUMMARY_20260911.md](AUDIT_SUMMARY_20260911.md)。77例逐帧复核后，prompt改为v7 compact，默认新索引为v8；精确隔离、文本缩减及验证见 [EVAL_REVIEW_20260911.md](EVAL_REVIEW_20260911.md)。
> 本次v8重建test每类46题，完整配对评测请用 `CASES_PER_BIN=0 bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh`；默认64会先补齐再去重，不能把呈现预算当独立题数。
>
> 历史（由顶部 v13 规则覆盖）：2026-09-11 新增并精修 `ACTION_OUTPUT_MODE=choice`：训练和输出改为**事件所属 high-level 的一个完整动作词组**，不是在每个事件重复问完五个 YES/NO，也不输出 A/B/C。纵向事件严格三选一 `DECELERATE / STOP / RESUME`；机动事件严格五选一。每条 case 会稳定地打乱候选词组顺序，target 始终是实际动作词组。旧标签的全 NO、invalid、或多动作同时 YES 不能凭空折成某个动作，choice 会显式排除并在 manifest/metrics 报告数量。choice 的 prompt/hash/adapter 合同独立，必须重新训练，不能拿旧 binary adapter 直接评测。

> 2026-09-10：当前默认数据为 v7，行为预测 prompt 与旧 adapter 不兼容；修复、重建及训练状态见 [REPAIR_20260910.md](REPAIR_20260910.md)。旧成绩不表示新模型验证。

> 2026-09-07：当前规则已升级为 current_wait_first_crossing_v6，prompt为v5_current_phase，旧索引/adapter不兼容。最新修订、逐帧证据和运行方式见 [REPAIR_20260907.md](REPAIR_20260907.md)；下文v5动作规则及旧成绩属于历史。

# SFT New Loop Phase3 运行说明

2026-09-08 DDP 验证超时缓解：teacher-forced val 后的自由生成只在 rank0 串行运行，
其余 rank 等待 NCCL barrier。默认 balance count=32 对应去重前约384题，可能超过
NCCL 原默认600秒。现在 `DDP_TIMEOUT_SECONDS` / `--ddp-timeout-seconds` 默认3600秒，
新旧 PyTorch 初始化分支均显式传入；该预算影响整个进程组，真实通信故障也可能更晚报错。
它只延长等待，不加速验证，也不能修复读图或 CUDA 卡死。

更新代码后，在原训练命令前加以下变量即可（默认自动选4张空闲卡）：

```bash
DDP_TIMEOUT_SECONDS=3600 GENERATION_EVAL_LOG_EVERY=10 \
  bash qwen3vl_local/sft_new_loop_phase3/train.sh ddp
# 显式 pin 同一配置：
GPU_IDS=0,1,2,3 DDP_TIMEOUT_SECONDS=3600 GENERATION_EVAL_LOG_EVERY=10 \
  bash qwen3vl_local/sft_new_loop_phase3/train.sh ddp
```

`run_full_pipeline.sh` 同样继承这两个环境变量。日志显示 generation 开始时的抽样/去重数、
样本边界的 route/frame、完成数/耗时/ETA 及各 rank barrier 进入/完成；耗时也写入指标。
默认每10题或样本边界间隔30秒打印，首尾必打印；**不是后台心跳**，单题卡住时不会刷新。
需要精确定位时用 `GENERATION_EVAL_LOG_EVERY=1`，最后一个 case 与其 progress 是否成对可定位问题。

远端短验可在原配置前加 `MAX_STEPS=12 EVAL_STEPS=10 GENERATION_EVAL_STEPS=10`，
确认出现 `generation-val-sync ... complete` 后继续到step11/12，再启动长训。
保留原 INDEX、RGB mode、采样与完整验证规模，短验只检查执行链，不判断训练质量。
若进度一直推进但仍超预算，按实测耗时加余量调大超时；若进度停住，应检查对应读图/推理，
不要只无限加大预算。缩小 `GENERATION_EVAL_BALANCE_COUNT` 会改变验证覆盖和选优可比性，
并可能触发 INVALID 覆盖硬校验；增加验证步频间隔不缩短单次等待。

当前入口只保存 adapter，没有 optimizer/scheduler 的断点恢复接口；不要把已有 best adapter
当成完整训练断点。若首次2000步尚未保存即退出，不能直接从step2000精确续训。

`sft_new_loop_phase3` 是 Phase1（RS + 三个可见事实）和 Phase2（EVENT）之后的
**high-level 动作决策**阶段。它把前两阶段已经确定的道路结构与异常事件标志当作
待核对前提写进 prompt；R-E2/R-E3/R-E5 可由显式导航/历史 gate 或
`dispatch.plan_candidate_requests` 提出候选，binary 可在同轮核对 invalid；choice 的前提有效性由上游门控承担，KEEP 不能充当 invalid。

- 输入只有一个 system turn 和一个 user turn；
- user turn = 四帧（或两端点）拼接 RGB history + 场景前提文本 + route 目标点的
  ego 相对坐标；
- 不渲染任何 `R1/R4/U-E2/UE3` 之类的数据集 code，也没有 synthetic assistant 前缀；
- binary 问题组最后回答 `INVALID_ACTION_CONTEXT`；默认 choice 只输出一个主要动作名称或 KEEP。

### 输出模式：`binary` 与 `choice`

显式 `ACTION_OUTPUT_MODE=binary` 保留逐动作二值输出：纵向 context 输出三条速度动作、
`KEEP` 和 `INVALID_ACTION_CONTEXT`（共五行），机动 context 输出五条变化动作、KEEP 和 invalid（共七行）。默认 `choice` 则由已确认的
事件 context 决定候选集合，模型只输出一行完整动作词组，例如 `STOP`。候选词组按 case
seed 稳定打乱；同一 case 可复现，换 case 的显示顺序会变化，答案词组本身不变：

| 事件域 | 动作候选 | 额外候选 | 输出例子 |
| --- | --- | --- | --- |
| 纵向让行（U-E1/U-E3/U-E5/U-E6/U-E7/R-E5） | `DECELERATE`、`STOP`、`RESUME` | `KEEP` | `STOP` |
| 机动（U-E2/U-E4/R-E2/R-E3） | 五个 high-level 动作 | `KEEP` | `LANE_CHANGE_LEFT` |

候选按 `动作名称: 场景内因果释义` 显示，名称和释义一起乱序，展示当前事件所属的三/五个动作加 KEEP：

| 动作名称 | 释义要点 |
| --- | --- |
| `DECELERATE` | 明显减速，但不满足优先的 STOP 条件 |
| `STOP` | 达到或保持持续近停，包括继续停车等待 |
| `RESUME` | 持续增速，不要求此前停过车 |
| `LANE_CHANGE_LEFT` | 最新帧之后第一次跨越车道边界，方向为自车朝向的左侧 |
| `LANE_CHANGE_RIGHT` | 最新帧之后第一次跨越车道边界，方向为自车朝向的右侧 |
| `KEEP` | 继续当前行进阶段，允许小幅调速；机动域无新跨线，纵向域不判断横向 |

时间窗和数值阈值仅由离线标注代码执行，模型输入不再显示；共用段只解释动作阶段。模型只输出冒号前的动作名称，例如
`LANE_CHANGE_LEFT`，不能附带释义。释义进入两种模式的 prompt hash；新训练使用此合同，旧 adapter 不能套用新提示词。

例如 UE1 某次乱序后的候选是（实际运行仍包含共用时间窗规则）：

```text
RESUME: Gain speed as the lead vehicle pulls away or following space opens, using the changing gap rather than the earlier braking event.
STOP: Stop or continue waiting behind the lead vehicle to avoid a rear-end collision while the forward gap remains blocked.
DECELERATE: Slow to avoid closing on the braking lead vehicle and rebuild following space while tracking its speed.
KEEP: Continue at roughly the current speed while tracking the lead vehicle and following gap; earlier braking need not cause another speed change now. This speed-only choice makes no claim about lane changes.
```

若答案是继续停车等待，模型只输出 `STOP`。变道问题忽略输入历史中已经发生的跨线，
并只预测未来窗口的第一次跨线，之后的归位不另选一次。

证据完整、有效题域中的全 false 动作证据进入 `KEEP`，纵横组合按 STOP > 首次跨线 > 纵向动作投影；仅 invalid 前提被剔除。
原始 `answers` / `action_signature` 留在索引供 binary 与证据审计，`primary_action` 记录主要动作，`keep_scope` 记录保持的题域；`primary_action_evidence_status` 记录完整性。
choice 的 case `gt` 为主要动作监督，`action_answers` 为原始证据；不能混用两种 exact 指标。
parser 严格接受恰好一行候选名称（包含 KEEP），不允许解释或复合字符串。
`production_ready` 要求严格格式、整体 exact，以及五种动作与 KEEP 各自的支持、precision/recall。

choice 没有独立 audit 输出格式，因而 `eval.sh` 默认将 `RUN_AUDIT_PROMPT_EVAL=auto` 解析为
`0`，避免和 production 做一遍相同生成；仍会生成 production 错例 RGB 审计包和可视化审计。
如需调试可显式设 `RUN_AUDIT_PROMPT_EVAL=1`，其结果只是重复生成，不能作为独立证据。

## 1. Phase1 / Phase2 答案 → ROAD_STRUCTURE / EVENT

Phase1 完整问法输出 8 行；subset/hierarchical 只包含实际问过的结构行，不能把未问当 NO。完整问法：`HIGHWAY / STATIC_OBSTACLE / VULNERABLE /
TRAFFIC_LIGHT_ABNORMAL` 和 `RS1 / RS2 / RS4 / RS5`。
Phase2 按问题域输出 `UE1/UE3/UE5`（道路走廊）或 `UE6`（局部路口），外加
`INVALID_EVENT_CONTEXT`。两者可恢复的映射如下：

| Phase1 / Phase2 答案 | ROAD_STRUCTURE / EVENT |
| --- | --- |
| `RS1=YES` | R1 常规道路 / 同向可行驶道路 |
| `RS2=YES` | R2 双向单车道 / 借对向车道 |
| `RS4=YES` | R4 信号灯路口 |
| `RS5=YES` | R5 无信号灯 / 路权路口 |
| 完整四个 RS 全 NO（与 `HIGHWAY` 独立） | R3 高速 / 匝道 / 合流 / 驶出 |
| `STATIC_OBSTACLE=YES` | U-E2 静态障碍物占道 |
| `VULNERABLE=YES` | U-E4 决策相关行人/骑车人，含沿路骑行 |
| `TRAFFIC_LIGHT_ABNORMAL=YES` | U-E7 已安装信号系统故障/异常 |
| Phase2 `UE1=YES` | U-E1 前车急刹 / 突然减速 |
| Phase2 `UE3=YES` | U-E3 动态车辆切入 / 动态占道 |
| Phase2 `UE5=YES` | U-E5 对向车辆异常侵占自车道 |
| Phase2 `UE6=YES` | U-E6 路口违规车辆冲突 |
| 全部异常 UE 为 NO | 仅表示没有这七个异常 UE；**不能**识别任何 R-E* |

因此可唯一确定的是 **U-E1..U-E7 七个异常事件标志**，不是单一 primary event：
Phase1/Phase2 的多行可以同时为 YES。需要 single-context 调度时，上游必须保留
事件集合和自身的时序状态，不得在本阶段凭固定优先级虚构“主事件”。
`keyframe_filter` 里的第八类 U-E8（前方道路暂时阻塞）在两个阶段都没有对应问题，
无法唯一确定，本阶段不作为上下文。

`R-E2/R-E3/R-E5` 没有上游直接输出。离线构建以既有 taxonomy 的显式 span 产生
transition-gate 候选，不代表新一轮 RGB 已确认全部帧。`R-E2` 包含绕障恢复与普通
目标变道；在线需要已观察到的恢复状态或当前导航目标车道。`R-E3` 需要可见活动
ramp/merge/exit 过渡，R-E5 需要局部常规路权语境，不能由 all-NO 自动推出。

运行时的最小调度合同如下。它避免把离线 GT 泄漏给模型，同时让每个 Phase3 问题
有真实来源：

```text
Phase1 RS + Phase2 valid UE flags
  ├─ exactly one compatible abnormal UE → its Phase3 context
  ├─ multiple abnormal UE flags          → upstream temporal scheduler retains the set;
  │                                       do not silently select one with a training-only priority
  ├─ observed bypass departure with recovery still pending → POST_BYPASS_RETURN candidate
  └─ R3 + route-planner transition cue (merge/exit command, changed route, or audited gate)
                                          → RAMP_MERGE_EXIT candidate
             ↓
4 RGB frames + natural-language context + ego route-target offset → Phase3 LoRA
             ↓
binary：事件域动作 YES/NO 行 + INVALID_ACTION_CONTEXT；choice：一个动作词组
```

The route-planner cue is only a caller-side candidate gate. It is never rendered as an answer,
dataset code, hidden map value, or future trajectory in the Qwen prompt; the LoRA must still
reject a visibly incompatible candidate through `INVALID_ACTION_CONTEXT`.

当前还可直接调用 `dispatch.plan_candidate_requests`，在 R3/R5 中提出相应常规
候选供 Phase3 核对，不要求另外一个已训练的RE视觉分类器。该调用不把候选写成
已确定事实。`candidate_response` 的 `CANDIDATE_REJECTED` 表示回常规流程；
`NOT_REJECTED_NO_ACTION` 表示未驳回且无所问动作，两者都不自动清除恢复状态。
纯调度接口尚未连接CARLA，invalid=NO也不是额外的RE分类精度指标。

## 2. 十个动作上下文

七个异常语境与 R-E2/R-E3/R-E5 共十桶，具体 RS 兼容表由 `context_taxonomy.py`
统一管理。事件可与 interrupted junction overlay 共存；RS 只限制几何，不自动产生事件。
`VULNERABLE_CROSSING` 是兼容 ID，包含沿道路骑行，并提供五动作以覆盖安全绕行。
`POST_BYPASS_RETURN` 也是兼容 ID，表示 R-E2 目标/恢复车道候选；仅有真实既往绕障
标志才在 `context_detail` 写入那段历史。R-E2 不再按 24 帧截断。

两条变道行 NO 只说明未来三秒暂不跨线，可能仍在借用车道等待，不能清除恢复状态。
在线可用 `transition_state.RecoveryState`，仅在可见/导航确认回到原车道后退出。
最终目的地的 y 符号不能选择当前变道方向；例如目标在左，绕障后仍可能必须向右返回。

`UNSIGNALIZED_PRIORITY` 对应显式 R5/R-E5，描述 STOP/yield 和正常路权让行。
普通无灯路口不自动成为 U-E7；U-E6 仍需对方违规证据。旧 U7 通过 Phase1 的
已审计灯故障答案适配；高速 U3 复用 Phase2 的显式 RGB YES 清单。

生产 prompt 现为 `v4_context_recheck`，数据名仍为 `sft_new_loop_phase3_mapping_v2`，
动作规则为 `ordered_speed_driving_lane_v5`。train/eval 同时检查动作规则版本与 `mapping_contract_hash`（绑定实际RGB映射决定），
旧索引必须从原始 meta 重建；旧 prompt adapter 不能直接评测新合同。
Phase1/2 的 prompt 和输出格式没有修改。最新审计见 `BOUNDARY_AUDIT_20260905.md`。

## 3. 五个变化动作及 KEEP

| 动作 | 含义 | 轨迹判据 |
| --- | --- | --- |
| `DECELERATE` | 明显减速，含为变道/合流等待时机的减速 | 排除即时 1.5s 持续停车后，未来 2s 首先达到减速阈值 `max(1.2 m/s, 20%)` |
| `STOP` | 停车 / 保持静止等待 | 未来 1.5s 内有连续两帧 ≤ 0.5 m/s；排除已停后持续起步 |
| `RESUME` | 持续增加速度，含从静止起步或绕行中提速，不暗示已经完成让行 | 未来 2s 连续两个采样达到加速阈值，且没有应优先处理的减速/持续停车 |
| `LANE_CHANGE_LEFT` | 向左变道 / 借对向车道 / 向左合流 | 未来 3s 同 road 连续两帧确认新 Driving lane；整个观察窗必须 Driving；road 变更/路肩/缺类型不强标 NO；仍需 RGB/车道段复核 |
| `LANE_CHANGE_RIGHT` | 向右变道 / 回原车道 / 向右驶出 | 同上，方向为右 |

`DECELERATE / STOP / RESUME` 互斥；变道与纵向动作独立。有效场景且证据完整时，
所问变化动作均 NO 就显式标为 `KEEP:YES`；不表示逐帧精确恒速，也不表示事件已经结束。
有任何所问动作 YES 时 KEEP 为 NO；错误前提下 INVALID 为 YES、包括 KEEP 在内的动作均 NO。
只问纵向时，未问的横向动作保持未知。STOP 与变道 YES 使用不同时间窗，不要求同时执行。
起步判据允许速度在正常行驶范围轻微回落；一旦达到 2 m/s，即时窗剩余部分持续不低于
2 m/s，才将初始静止与继续停车分开，不能要求速度严格单调。

标定口径来自 2026-09-04 的逐帧 meta 轨迹 + RGB 复核：

- 纵向只看未来真实速度曲线，不看 scenario 名。`STOP` 用更短的即时窗，所以
  “已经停稳但马上起步”是 `RESUME` 而不是继续 `STOP`；
- 横向绝不用航向角或 steer 判定。弯道会让 steer/yaw 长期非零却不换车道，
  `road_id` + `lane_id` 来自 Any waypoint，会投影到 Shoulder；因此只在完整 Driving
  窗口内生成变道候选。不能混用缺少配套 road_id 的 `ego_lane_id`；
- 方向由 OpenDRIVE 的 lane id 排序乘以**自车首次合法进入该 road 时的行驶方向符号**
  决定。用当前 lane id 会把“从对向车道回原车道”错判成左变道；
- ego frame 的左右符号由 `probe_ego_frame_sign.py` 在左转/右转 scenario 上取证：
  `x` 正为正前方，`y` **负为左、正为右**（CARLA 左手系）。

复核工具与产物：

```bash
python qwen3vl_local/sft_new_loop_phase3/probe_ego_frame_sign.py
python qwen3vl_local/sft_new_loop_phase3/probe_trajectory.py --scenario AccidentTwoWays --max-routes 2
python qwen3vl_local/sft_new_loop_phase3/render_action_contact_sheet.py \
  --scenario AccidentTwoWays --event R-E2 --start-frame 100 --frame-step 4 --max-frames 9
```

输出写在 `qwen3vl_local/sft_new_loop_phase3/probe_output/`：
`probe_*.txt` 是逐帧 RS/EVENT + 轨迹信号对照，`sheet_*.jpg` 是逐帧 RGB
叠加派生动作标签的 contact sheet。

## 4. 导航目标坐标

prompt 里的 `[NAVIGATION_GOAL]` 与 `sft_base` / `sft_v3` / `sft_v4` 的
`EGO_TO_GOAL_XY` 同源（`next_target_points[-1]` 转 ego frame），但额外显式写明
坐标语义与一句自然语言翻译，例如：

```text
ROUTE_TARGET_XY: (x=+61.2 m, y=-40.5 m)
x is the signed distance straight ahead of ego, positive in front and negative behind.
y is the signed lateral distance, negative to ego's LEFT and positive to ego's RIGHT.
The route target is about 61.2 m ahead of ego and about 40.5 m to ego's left.
```

这表示最终目的地的方位，不能据此确定下一条目标车道。实测最终目的地在左侧约 35m，
仍可能需要向右回原车道；应结合可见车道边界、已发生的绕障历史和当前导航指令判断。
仅最终目标坐标不足以唯一确定所有普通导航变道，相关样本仍须核查可观察性。

## 5. INVALID 合同

`INVALID_ACTION_CONTEXT=YES` 只表示“给定RS与可见道路明确不符，或同RS下事件前提被可见证据明确反驳”，
并要求所有动作行为 NO。夜间、雾、遮挡、拥堵、或者“当前不需要任何动作”都不是
invalid；不需要动作时应当所有动作行为 NO 且 invalid 也为 NO。

wrong-RS错配由几何硬约束构造，不靠场景名；same-RS错事件只能来自显式RGB人工决定。几何错配例子：

- 真实 R1/R2、不在路口、且距下一个路口 ≥ 25m → 可问局部路口冲突 / 信号失效 / 匝道合流；
- 真实 R3、不在路口 → 可问局部路口冲突 / 信号失效；
- 真实 R4/R5 → 可问匝道合流 / 驶出。

另外将明确错误的 RS 前提与各个 asked context 组合，使十个 context 都有 invalid，
包括本身可与多种 RS 共存的 U-E1/U-E2/U-E3/U-E4/U-E5/R-E2。
这里的错误是给定 RS 与图像矛盾，不是断言这些事件不能发生在真实 RS 上。

invalid 行按 `source=<上下文>|true_rs=<R*>|asked_context=<错误上下文>` 三维签名
均衡，训练、评测与审计都保留这三个维度。构建、训练、评测默认都会硬校验这三维的
完整覆盖；只有 route 受限的 smoke 子集才用
`--no-require-invalid-true-rs-coverage`（构建）或 `--no-require-invalid-coverage`
（训练/评测）临时放开，正式流水线不要关闭。

## 6. 构建数据

```bash
python qwen3vl_local/sft_new_loop_phase3/build_dataset.py
```

默认 `--target-per-context 0`（取该 split 最小的上下文桶）、`--invalid-ratio 0.20`。
比例的分母是有效样本：十桶各 N、invalid 2N，占总量 16.7%。训练默认
`INVALID_FOCUS_MULTIPLIER=2.0`，定额 eval/generation 同样取 2N；全量 eval 保留原索引比例。
每个上下文桶内部再按动作签名（`STOP` / `DECELERATE` / `RESUME` /
`LANE_CHANGE_*` / `KEEP` / 组合）尽量均分，保证五个动作都有足够正类。
train 用 route 轮转选帧，val/test 用确定性抽样。

这段构建口径保留原始多标签证据。choice 读取后剔除 invalid，把其余行投影为主要动作或 KEEP，
在每个 context 内按主要动作平衡。`choice_filter` 报告有效动作（包含 KEEP）及 invalid 剔除量；
组合行参与主要动作任务，指标不再表示纵横两个分量都预测正确。

快速 smoke（每个 scenario 只取 40 条 route）：

```bash
python qwen3vl_local/sft_new_loop_phase3/build_dataset.py \
  --max-routes-per-scenario 40 \
  --no-require-invalid-true-rs-coverage \
  --output-dir checkpoints/sft_new_loop_phase3_data_smoke
```

## 7. 训练

轻量链路检查：

```bash
bash qwen3vl_local/sft_new_loop_phase3/train.sh check
```

默认四卡、显式单卡与显式四卡：

```bash
bash qwen3vl_local/sft_new_loop_phase3/train.sh
GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase3/train.sh single
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_new_loop_phase3/train.sh ddp
```

选择题训练示例（新 run 名会带 `_choice`，避免与 binary 输出混放）：

```bash
# 先走 2 step 执行链检查；不产生成绩结论
ACTION_OUTPUT_MODE=choice CHECK_MAX_STEPS=2 \
  bash qwen3vl_local/sft_new_loop_phase3/train.sh check

# 正式 DDP 训练；choice adapter 必须从头训练
ACTION_OUTPUT_MODE=choice GPU_IDS=0,1,2,3 \
  bash qwen3vl_local/sft_new_loop_phase3/train.sh ddp
```

常用覆盖：

```bash
EVAL_STEPS=2000 \
GENERATION_EVAL_STEPS=2000 \
GENERATION_EVAL_BALANCE_COUNT=32 \
GENERATION_EVAL_MIN_INVALID_EXACT=0.80 \
GENERATION_EVAL_MIN_LANE_CHANGE_RECALL=0.60 \
GENERATION_EVAL_MIN_STOP_RECALL=0.80 \
GENERATION_EVAL_MIN_NO_ACTION_EXACT=0.50 \
FOCUS_BALANCE_COUNT=1024 \
bash qwen3vl_local/sft_new_loop_phase3/train.sh ddp
```

训练默认每 2000 optimizer step 跑 teacher-forced val 和固定均衡的自由生成 val，
每 20000 step 保存 checkpoint。训练会输出：

- `train_balance.json`：上下文、动作签名、真实 RS，以及 invalid 三维子类别采样；
- `balance/epoch_*.json`：每轮的类别、动作签名与 invalid 均衡 guard；
- `train_eval_metrics.jsonl` / `generation_val_cases.jsonl`；
- `tb/`：loss、每个问题准确率、每个动作的 recall/precision、invalid joint 指标；
- `best_val/`、通过全部门槛时的 `best_generation/`、仅诊断用的 `fallback_generation/`、
  `generation_selection_status.json`、`checkpoint-*`、`final/`；
- `sft_new_loop_phase3_adapter_config.json`：prompt hash、history mode、采样与输入合同。

## 8. 独立评测

Base：

```bash
GPU_IDS=0 python qwen3vl_local/sft_new_loop_phase3/eval.py \
  --index checkpoints/sft_new_loop_phase3_data/frame_index.jsonl \
  --data-root lead_data \
  --cases-per-bin 64
```

LoRA：

```bash
GPU_IDS=0 python qwen3vl_local/sft_new_loop_phase3/eval.py \
  --index checkpoints/sft_new_loop_phase3_data/frame_index.jsonl \
  --data-root lead_data \
  --adapter-dir checkpoints/sft_new_loop_phase3_runs/latest \
  --cases-per-bin 64
```

一键 base + LoRA + audit prompt + 不超过 30MB 的错例审计压缩包：

```bash
ADAPTER_DIR=checkpoints/sft_new_loop_phase3_runs/latest \
  bash qwen3vl_local/sft_new_loop_phase3/eval.sh
```

`eval.sh` 从 adapter 配置读取并硬校验 `action_output_mode`，base、LoRA production 与 audit
会使用同一个 mode。choice 的完整测试集评测例子：

```bash
CASES_PER_BIN=0 ADAPTER_DIR=checkpoints/sft_new_loop_phase3_runs/<choice-run> \
  bash qwen3vl_local/sft_new_loop_phase3/eval.sh
```

传 run 根目录时，`eval.py` 与 `eval.sh` 都按
`best_generation/ → final/ → fallback_generation/` 解析，并要求候选目录中存在
`sft_new_loop_phase3_adapter_config.json`，不会因残留空目录误选权重。
adapter 的 `production_prompt_sha256` 与当前 prompt 不一致时会直接报错，
避免拿不同 prompt 合同的 adapter 互相比较。
`eval.sh` 结束时会生成
`checkpoints/sft_new_loop_phase3_eval_review/<ts>/sft_new_loop_phase3_<ts>_<rgb_mode>_audit_bundle.tar.gz`
（或 `OUTPUT_ROOT/<BUNDLE_BASENAME>.tar.gz`），并打印 `[phase3-eval] audit bundle: ...`。
压缩包复制 metrics、summary、manifest、adapter 文本元数据和少量下采样 RGB 错例；
权重、checkpoint、TensorBoard 和大 JSONL 不进入包内。可用
`BUNDLE_MAX_MB=20` 或 `BUNDLE_BASENAME=<name>` 覆盖默认上限与文件名。

## 9. 错例 RGB 审计包

`eval.sh` 最后会自动调用 `audit_eval_cases.py` 生成散目录
`lora_production_audit_samples/`，随后再把关键评测文件和抽样 RGB 打成 `.tar.gz`；
也可以单独对任意 eval 目录跑：

```bash
python qwen3vl_local/sft_new_loop_phase3/audit_eval_cases.py \
  --eval-dir checkpoints/sft_new_loop_phase3_eval_review/<ts>/lora_production \
  --output-dir checkpoints/sft_new_loop_phase3_audit_samples/lora \
  --per-target 12 --overwrite
```

它按动作错误类型分桶抽样并复制模型真实输入的 RGB：
`decelerate_fn/fp`、`stop_fn/fp`、`resume_fn/fp`、`lane_change_left/right_fn/fp`、
`lane_change_side_swap`（左右判反）、`longitudinal_multi_yes`（纵向互斥被破坏）、
`invalid_context_fn/fp`、`invalid_context_not_all_no`、`no_action_fp`
（本应 KEEP 却给了变化动作）、`invalid_answer`（格式失效）。
每个错例目录里带一份 `audit_note.md`，含逐帧复核清单：情境是否可信、自车是否已在
刹车/静止、是否仍在同两条车道线之间、是否被弯道误判成变道、目标点侧向是否与所需
变道一致。

## 10. 一键全流程与输入合同矩阵

```bash
bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
bash qwen3vl_local/sft_new_loop_phase3/run_rgb_mode_matrix.sh

# 从构建到评测都使用 choice；完整 test 不补齐呈现预算
ACTION_OUTPUT_MODE=choice CASES_PER_BIN=0 \
  bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```

## 11. 合同测试

```bash
python -m pytest qwen3vl_local/sft_new_loop_phase3/test_action_contract.py -q
```

覆盖 Phase1/Phase2 完整/部分答案到 RS 和并发 EVENT 的恢复、七类 U-E 的上下文归属、
普通无灯与灯故障的区分、不按固定帧数截断的恢复状态、纵向动作互斥、借道/回原车道方向、
ego frame 左右符号、prompt 不泄漏数据集 code、弯道不能当变道证据、
严格解析器与 audit 证据合同、以及 invalid 的几何硬约束。


## 2026-09-05 审计产物与限制

当前规则、具体 RGB 疑点和覆盖范围见 [MAPPING_AUDIT_20260905.md](MAPPING_AUDIT_20260905.md)。
`rgb_mapping_review_20260905.jsonl` 只记录真正看过的序列；机器遍历 582 条 route
不是人工完成 582 条逐帧动作确认。`mapping_rgb_decisions_v2.jsonl` 是按 route/frame
隔离的明确疑点，默认参与 Phase3 构建，不反写 Phase1/2。

`build_dataset.py --use-review-cache` 复用既有 RGB 审计缓存，用于候选审计；
`--candidate-cache <candidate_frames.jsonl>` 仅复用同版本动作规则的候选再均衡。
正式全量入口仍读 collection_results。无论采用哪种来源，manifest 明确记录范围。
`--include-visual-risk` 仅适用于检查被过滤内容，产物不能宣称已通过视觉质量筛选。
候选保存先于均衡检查，缺少任一 split/event 桶时明确失败，不靠重复其他事件填补。


## 追加：同 RS 错事件监督

主构建器现读取 `same_rs_invalid_review_v1.jsonl`，仅对显式看过的实际history构造
`same_rs_wrong_event`，与原 `wrong_road_structure` 一起进入INVALID桶。相应的
`invalid_reason` 贯穿train/eval/TensorBoard与错例审计；已具备的同RS题覆盖在二次
采样后必须保留，不能只报告候选池存在。source桶仍均衡，数量在联合配额允许时
争取同RS占INVALID的25%，报告实际值，不用伪造负例填缺口。

现有审核集是 `probe_output/mapping_audit_20260905/filtered_v12/`，1152题，七个UE
同RS题在三个split均保留；RE同RS覆盖范围见 `BOUNDARY_AUDIT_20260905.md`。
这一审核子集及复用其样本的独立challenge导出都不能冒充全量数据/独立泛化成绩。
