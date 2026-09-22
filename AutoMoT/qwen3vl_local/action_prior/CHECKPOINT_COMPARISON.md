# 多模型同帧轨迹对比

统一入口是 `action_prior/compare_checkpoints.sh`，支持 `bev_only`、`qwen_simple`、`action_prior` 任意组合及任意数量的训练目录。两个消融目录中的同名脚本转到此入口。只增加对比文件，没有修改训练、标签、已有eval或严格合同校验。

## 最常用操作

默认当前目录为远端 `AutoMoT/`。编辑 `qwen3vl_local/action_prior/compare_checkpoints.sh` 开头的 `CKPT_DIRS` 数组，填入带时间戳的训练结果文件夹，之后运行：

```bash
bash qwen3vl_local/action_prior/compare_checkpoints.sh
# 如需指定一张卡或四张卡：
GPU_IDS=0 bash qwen3vl_local/action_prior/compare_checkpoints.sh
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/compare_checkpoints.sh
```

数组已经填入本次两份审计记录对应的原始训练目录。**不能填 `training_audit_1/2`：它们只有审计信息，没有模型权重。** 可以在数组中继续添加第三、第四个目录。相对路径优先按执行命令时的当前目录解析，也识别 `AutoMoT/` 根和仓库根；绝对路径支持空格。已存在的软链接会固定到真实run目录。

也可以直接传目录，覆盖脚本数组：

```bash
bash qwen3vl_local/action_prior/compare_checkpoints.sh \
  checkpoints/action_expert_ablation/bev_only/run_20260921_235146 \
  checkpoints/action_expert_ablation/bev_only/run_20260921_235150 \
  --names bev_without_token bev_with_token
```

`--names` 可省略；指定时数量必须与目录数一致，放在所有目录后面。第三个目录可以来自qwen_simple或action_prior，无需手动选择运行器。

脚本顶部可直接填写每个类别的数量：

```bash
CASES_PER_CATEGORY=50     # 未单独指定的每类，在 train/test 各最多检查50个候选
EVENT_CASES=("UE1=12" "UE4=20" "RE5=10" "test/UE7=6")
ACTION_CASES=("STOP=12" "LANE_CHANGE_LEFT=20" "KEEP=6" "UNCOND=0")
METHOD_NAMES=("BEV without token" "BEV with token")  # 对应CKPT_DIRS顺序
```

不带split前缀时同时作用于train/test；`train/` 或 `test/` 只覆盖相应集合。`0` 表示该分类目录不选案例；同一帧仍可能因其它event/action类别被选中。只看指定类别时可将 `CASES_PER_CATEGORY=0`，再指定需要的类别。类别拼写错误会报错；命令行同名配置覆盖脚本设置，split专属设置优先于通用设置。

```bash
bash qwen3vl_local/action_prior/compare_checkpoints.sh \
  --cases-per-category 0 \
  --event-cases 'UE2=12,test/UE2=6' \
  --action-cases 'STOP=10,LANE_CHANGE_LEFT=12'
```

默认每类每split准备最多50个候选；同一录制中的同类案例至少间隔8帧（约2秒），优先不同物理路线，不足不重复凑数。可调：

```bash
bash qwen3vl_local/action_prior/compare_checkpoints.sh \
  --cases-per-category 12 --min-frame-gap 8 --sampling-seed auto --seed 2026 --workers 4
```

先检查选择清单、条件与标签而不运行模型：

```bash
bash qwen3vl_local/action_prior/compare_checkpoints.sh --plan-only
```

plan-only仍需要原训练环境的Python依赖、真实checkpoint、BEV/Qwen文件和原数据索引以核对内容身份，但不查询GPU、不构造模型、不运行GPU推理。结果始终写新的时间目录，不覆盖旧测试。

## 分别开关event与action

在统一sh中直接编辑，两项默认true，三个入口共用：

```bash
ENABLE_EVENT=false
ENABLE_ACTION=true
```

上述配置只从action子类选择候选、搜索错误并生成action的train/test目录；不会为event额外选案例或生成event目录。反过来设置true/false则只测event，true/true同时启用。两项都false会在GPU查询、预检及建输出目录之前报错。CLI可用 `--no-event` / `--no-action` 关闭，或 `--event` / `--action` 覆盖sh设置。

关闭风格下的EVENT_CASES/ACTION_CASES配额不参与采样；启用风格仍遵守每子类候选上限、误差阈值与命中目标，train/test各自独立。manifest的enabled_styles记录实际选择。开关只控制分类方式，不删除case的event/action审计标签，也不改变checkpoint保存的action token或文字先验输入条件。普通比较及误差搜索两种模式都适用。

## 每次重新选案例并打乱执行顺序

脚本中直接配置，无需在命令前传环境变量：

```bash
CASES_PER_CATEGORY=50
SAMPLING_SEED=auto  # 每次运行按纳秒时间+系统随机源生成新的采样种子
EVAL_SEED=2026      # 独立的模型配对评估噪声种子
```

`auto` 在父进程启动时只生成一次；各模型共享同一采样计划。采样仍保持原类别配额、路线多样性、最小帧间隔和train/test物理隔离，各split选中的案例另按本次种子打乱，所有模型共享相同的逻辑清单顺序。模型训练seed、原train/test划分和先验噪声条件不变；不重新训练。每个worker先处理自己的train再处理test；多卡分片只取公共顺序中的子序列，实际开始/完成时序由运行耗时决定。

实际seed立即打印并写入本次 `sampling.json`，随后写入 `manifest.json` 和 `REPORT.md`。需要重放时，把sh中的 `SAMPLING_SEED=auto` 改为该次保存的整数，其余数据、采样数量、最小帧间隔及代码相同即可复现；CLI也支持 `--sampling-seed 12345`。`--seed` 现在只对应EVAL_SEED，不再同时控制案例选择。旧版本默认2026选例的集合可用 `--sampling-seed 2026` 重选，但新版的执行顺序会打乱。

新种子不等于保证每次案例都不重合：数据少或配额覆盖全部候选时，集合可能相同；不维护跨运行已查看列表。即使某个case重复出现，其评估噪声仍由case身份和EVAL_SEED确定，便于对照。查看test案例属于诊断曝光，不代表新的独立盲测。

本轮76项相关CPU测试通过，覆盖同一时刻auto种子仍变化、固定种子重放、不同seed改变选例和顺序、各模型计划同序及推理seed独立；未运行真实GPU模型。

后续实现其它对比脚本时，可沿用以下种子职责划分：

| 随机来源 | 本工具策略 | 用途及约束 |
|---|---|---|
| 原训练/数据划分seed | 从checkpoint合同恢复 | 不因浏览新案例而改变split归属或训练条件 |
| SAMPLING_SEED | 默认auto；主进程生成一次并保存 | 控制选例和逻辑顺序，所有checkpoint共用 |
| EVAL_SEED | 默认固定2026 | 与case身份共同决定FM的eps/t/ODE噪声，不混入GPU号、worker号或完成次序 |

实现要点：`(time.time_ns() ^ secrets.randbits(63)) & ((1 << 63)-1)`只在主入口执行一次；先稳定归一候选顺序，再用局部Random做分层抽样/打乱，不能让每个GPU各自按时间抽样。先把完整计划落盘，再按计划切片。更改卡数时仍使用相同采样seed和EVAL_SEED，能够保持案例集合及每case的评估噪声；多卡浮点运算和不同硬件不承诺逐位一致。图像筛选、误差门槛及分组汇总在每批全模型配对完成后执行，不参与候选种子选择；早停的实际检查集还受批次边界影响。

跨运行需要“尽量看新案例”时可用auto；需要同一组可复现对比时固定采样seed；需要研究噪声敏感性时固定采样seed并显式改变EVAL_SEED。种子只是记录条件，不能把反复筛看test当作未曝光的盲测。

## Waypoint误差搜索：每类最多检查50个，找到5个就停

三个shell入口共用以下配置，直接在 `action_prior/compare_checkpoints.sh` 修改，无需前置环境变量：

```bash
CASES_PER_CATEGORY=50       # 每个event/action、每个train/test的候选检查上限
ERROR_ONLY=true             # 默认开启分批误差搜索；false恢复全部候选的普通比较
ERROR_CASES_PER_CATEGORY=5  # 每类最终保留目标
ERROR_ADE_THRESHOLD_M=1.0
ERROR_FDE_THRESHOLD_M=3.0
```

保存后直接运行，选卡仍自动：

```bash
bash qwen3vl_local/action_prior/compare_checkpoints.sh
# 显式pin示例：
GPU_IDS=0 bash qwen3vl_local/action_prior/compare_checkpoints.sh
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/compare_checkpoints.sh
# CLI最后生效，也可以显式指定：
bash qwen3vl_local/action_prior/compare_checkpoints.sh --cases-per-category 50 --error-only --error-cases-per-category 5 --error-ade-threshold-m 1 --error-fde-threshold-m 3
```

只检查 **waypoint**，route的ADE/FDE不参与筛选（轨迹仍可绘图、指标仍可记录）。对任一模型与GT、任意两个模型，逐时刻算二维欧氏距离，其平均值是ADE，末点距离是FDE；**ADE严格大于1米，或者FDE严格大于3米**就算命中，等于不触发。不是比较两个ADE/FDE标量之差。所有轨迹必须有相同时间点数且坐标有限；缺帧、坏轨迹、GT不一致会报错。旧 `ERROR_THRESHOLD_M` / `--error-threshold-m` 已被两个独立阈值替代。

流程如下：

1. 在原训练合同的有效train/test集合内，按同源标签、路线多样性和帧间隔规则，各类别最多准备50个候选；数据不足或帧间隔约束后不足50时用实际数量。`EVENT_CASES` / `ACTION_CASES`仍是逐类**候选预算**覆盖，0关闭该类。不会为了找错例遍历预算之外的完整数据集。
2. 在所有尚未满额类别间轮转取未评估候选，同一case仅评估一次且所有checkpoint同帧同噪声。按GPU数量自适应分片，批量固定为 `max(8, 2×所选卡数)` 个去重case。
3. 每批完成全模型配对后，按派发顺序检查误差，命中就加入所属且尚未满额的event/action类别。某类达到5例就停止为其单独派发；其它类别继续。某类最多检查其候选预算内的50个，检查完不足5就保留实际命中数。
4. 已派发的一批会完成，所以实际检查数可能略超过“第5个命中”的位置；满额类与其它类别重叠的case也可能继续被检查。但每类最终目录**最多5例**，不因多卡返回次序改变入选顺序。

例如某类前8个里有8个命中，只展示前5个，不再单独检查剩余42个；若50个仅2个命中，则检查完50个，保留2个并报告缺额3。train与test分别搜索、分别计数，不混用名额；完全没命中仍输出空图库及原因。

常驻GPU服务会跨批复用当前checkpoint上下文，避免同模型每批重新读权重/合同并加载Qwen/BEV；换模型时释放旧上下文并重新严格检查。checkpoint多于GPU时会有模型切换，不能保证零重复加载。分片始终保留原条件及EMA评估，失败/中断会回收本次所有服务及加载子进程，不跳过错误case凑结果。

输出说明：

- `candidate_plan.json`：完整候选预算及原抽样计划；`_plan/`保存每模型输入行与哈希。
- `search.json`：每个split/类别的candidates、evaluated、matched、retained、target、shortfall和reason；结束原因是target_reached、candidates_exhausted或disabled。运行时每批刷新，终端也报告进度。
- `search_audit.jsonl`：实际评估的每个case的waypoint误差与触发者。批次原始结果在 `_search/batch_*/`；`_models/`按case整理已评估的预测和实际输入。
- `manifest.json`：搜索后实际已评估case、各类最终保留列表及搜索统计；`error_filter.json`包含误差命中标记kept和最终配额选择quota_selected。命中但配额已满的案例不会额外出图。
- `event/`、`action/`和图库只发布达到条件且进入该类配额的案例；每例图中标注触发模型对、ADE/FDE及阈值，`case.json`记录同样依据。
- `summary.json`的means/paired_vs_first以该类**实际已评估候选**为分母，displayed_means只针对最终保留案例。REPORT包含搜索预算/检查数/命中数/保留数/结束原因；提前停止会改变统计分布，不能拿来代表全量test表现或重新选checkpoint。

复现需固定采样seed、评估seed、候选预算、阈值、命中目标、数据和代码。更换GPU数量可能改变批次边界及额外完成的case，候选池与每case噪声不变，但早停后的实际检查/保留集合不保证完全一样。搜索只用于错误分析，train/test归属不改。

本轮132项CPU检查通过，包括ADE/FDE独立触发、任意模型对、route忽略、严格边界、50预算/5配额、跨类别去重、零命中、常驻进程复用/失败与SIGTERM回收、三模型入口上下文缓存以及实际合成图片发布；未运行远端真实GPU模型。

## GPU启动前长时间停在preflight

终端的 `model 2/2` 是“开始检查第二个模型”，不是两个模型已经检查完成。GPU选择只决定后续分配；原权重完整val选优、CPU读取checkpoint、冻结权重/源码/索引哈希、百万帧full map与动作候选验证、有效split扫描和分层选例都发生在真正启动GPU worker之前。少量可视化case仍需从完整类别池选择，降低每类数量不会消除这些固定开销。

对比入口现在分别报告加载权重、合同/映射校验、checkpoint SHA256、有效pool读取、标签标注、分层选帧及模型配对检查。每个耗时阶段每15秒打印耗时、RSS内存和主线程当前函数位置，并更新 `preflight.json` / `preflight.log`。某函数位置持续相同不直接证明死锁；结合CPU时间、RSS及阶段耗时判断。旧版本仅有两行model日志，无法据此区分正常预检、慢I/O或内存压力。

同一次选帧只对选中的case计算文件夹SHA256，不再对全部候选帧重复计算；临时标签池选完即释放，避免与第二模型全量pool同时保留。原训练源码/哈希/标签校验未改，未跳过任何合同。GPU推理队列开始时会出现 `[comparison] start model_01 GPU=...`，之后看独立模型日志；该worker自身仍会先校验合同再加载模型。

对已在远端运行的旧版本，可在另一个终端只读检查：

```bash
pgrep -af 'compare_checkpoints.py'
# 用上面查到的父进程PID替换12345：
ps -p 12345 -o pid,etime,pcpu,pmem,rss,stat
free -h
```

高CPU可能在解析/标注；D状态或低CPU可能在等I/O；内存和swap情况能帮助识别内存压力。这些指标只能辅助定位，不能单凭GPU利用率0断定卡死。正在运行的旧进程不会自动获得新增日志，需要部署新版后另开测试。本机没有访问远端进程，也没有测得真实服务器预检时长。

## GPU模型complete后还没有总对比图

`[comparison] complete model_01 GPU=0` 仅表示该GPU worker结束。所有worker成功后，父进程更新 `status.json` 为 `rendering`，在CPU上核对配对预测/GT、读取场景、绘制每例多模型主图、历史输入和各模型单独图，再发布分类目录及汇总。`scheduler.json` 的complete仅指推理队列；总任务以 `status.json` 的complete和最终 `[comparison] complete: ...` 为准。

旧版此阶段没有终端进度。新版显示总case数、当前case序号，每15秒更新 `render.json` / `render.log`（阶段、耗时、RSS、调用位置）；每例先生成 `comparison.png/.pdf`，打印可查看路径，再生成历史和单模型图。最后写完REPORT和图库后才报告整体完成。绘图失败保留失败阶段并由主入口写入status.json，不能把已有部分图片视为全部完成。

每例 `comparison.png/.pdf` 将GT和所有checkpoint轨迹叠加在同一组三相机视图及俯视坐标中；`model_01.png` 等仅是单模型对GT。绘制中的去重图位于 `_cases/<train或test>/<id>/comparison.png`；整个split完成后才发布到 `event/<split>/<event>/<id>/` 和 `action/<split>/<action>/<id>/`，图库 `index.html` 最后生成。

在AutoMoT目录另开终端，只读查看已有运行（替换实际时间目录）：

```bash
RUN_DIR=test/run_20260922_114725_362629
cat "$RUN_DIR/status.json"
rg --files "$RUN_DIR/_cases" | rg '/comparison\.png$'
# 新版额外可查看；旧运行没有此文件：
tail -n 10 "$RUN_DIR/render.log"
```

间隔一段时间比较图像数量/更新时间，并结合父进程CPU和I/O判断是否仍推进；单独的rendering状态可能在强杀后残留，不能证明进程存活。更新代码不会给已经进入旧绘图函数的进程补日志。本轮58项CPU测试通过，包括真实合成图中三个相机均含GT和两模型的验证；没有远端真实数据绘图耗时或GPU验收。

## GPU自适应并发

脚本顶部 `GPU_COUNT=4` 表示默认最多自动选4张卡。按 `nvidia-smi` 显存占用、利用率从低到高选卡，检测到少于4张会自动减少。旧版把GPU数截断到checkpoint数，导致两模型只用两卡；当前先为每个模型分配一份任务，再将空余卡均匀分配为额外案例分片。一张GPU同时只运行一个worker，checkpoint过多时排队，卡空出后立即领取下一个任务。

| 检测/指定的卡数 | checkpoint数 | 调度（候选案例足够时） |
|---:|---:|---|
| 4 | 2 | 每模型2个分片，各自加载模型，共4个worker并发 |
| 4 | 3 | 模型分片数2/1/1，共4个worker并发 |
| 4 | 4 | 每模型1个worker，共4个并发 |
| 4 | 6 | 每模型1份，先跑4个，剩余2个排队 |
| 2 | 6 | 同时运行2个，其余排队 |
| 1 | 任意 | 各模型顺序运行 |

分片数不超过该模型的去重case总数，不用空任务或重复case占卡。例如两模型各只有1个case，即使指定4卡也只会启动2个worker。分片按原随机顺序轮转切分train/test整体清单，允许某片某split为空；每个模型的全部案例恰好执行一次，原计划/hash、条件合同及checkpoint内容校验继续生效。GPU数不参与采样种子或噪声种子。

普通比较及搜索的每批任务都按上述方式分片。这属于多个模型副本分担独立case，每个worker仍需要在单卡装入其完整模型及冻结Qwen/BEV。不是把一个巨大模型拆开装到多卡，也不修改训练DDP或重训模型。CPU数据加载 `--workers` 是**每个worker**的数量，默认4；四个worker最多有16个加载子进程及四份模型CPU内存。多副本会重复加载与合同校验；小case量可能被启动/I/O开销主导。分片按数量分配，不预测每帧或模型的耗时，也不在尾部搬迁运行中模型，因此不保证全程满卡或线性加速。CPU预检及绘图阶段仍不会占满GPU。

```bash
# 自动最多选两张卡
bash qwen3vl_local/action_prior/compare_checkpoints.sh --gpus 2
# 或通过脚本环境变量设置自动选卡上限
GPU_COUNT=4 bash qwen3vl_local/action_prior/compare_checkpoints.sh
# 显式pin以GPU_IDS卡数为准，覆盖自动选卡上限，并跳过nvidia-smi选址
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/compare_checkpoints.sh
```

以下一次性队列日志描述适用于ERROR_ONLY=false；误差搜索使用常驻服务，日志是 `logs/search_worker_XX.log`，scheduler记录每批任务及服务GPU/PID。

自动选卡是启动时的空闲程度排序，不是跨任务的GPU资源锁；按项目入口规则覆盖已有可见卡mask，指定卡号使用 `GPU_IDS`。终端打印实际workers、parallel、shards/model及unused GPUs。单分片日志保持 `logs/model_XX.log`，多分片日志为 `logs/model_XX_shard_YY.log`；`scheduler.json`记录任务ID、模型ID、GPU、PID、开始/结束与退出码，完成数量指worker任务数。任一worker失败后停止派发并回收本次其它进程组；Ctrl-C/SIGTERM同样清理，保留已有结果，不生成不完整汇总。

各worker拥有独立输出/缓存：多分片原始结果位于 `_models/model_XX/shards/shard_YY/<split>/`，单分片仍用 `_models/model_XX/<split>/`。`manifest.json` 的 `evaluation_shards` 记录输出位置和分片case身份；`execution`记录卡数与分配。所有worker成功后，父进程逐片检查漏帧、重复、错误身份，再核对各模型GT并按case ID配对；**汇总从逐case指标计算，绝不直接平均大小不同分片的metrics均值**。最终event/action图库及每case的全部方法比较形式不变。

本轮99项已有相关CPU检查及18项分片CPU检查通过（共117项），包括两模型/四GPU槽位的真实CPU子进程、2/1/1分配、卡少于模型、极少/空split、hash校验、缺帧拒绝、不同大小分片汇总与单任务一致及实际输入图像读取。没有在远端跑真实GPU模型，不宣称实际吞吐收益。


## 输出

默认 `AutoMoT/test/run_YYYYMMDD_HHMMSS_微秒/`，`test/`与 `checkpoints/` 同级；无论从哪个当前目录启动，默认位置都相同。脚本顶部 `OUTPUT_ROOT` 可修改，也可传 `--output-root`（显式相对路径按启动时当前目录解析）。已有旧结果保持原位，新测试使用新目录。

```text
AutoMoT/
  checkpoints/                原训练目录及权重
  test/
    run_YYYYMMDD_HHMMSS_微秒/  本次完整测试结果
```

每次结果目录结构：

```text
REPORT.md                     模型选择、各类别指标表
index.html                    本地浏览逐例拼图
manifest.json                 权重SHA256、原/实际参数、标签来源、case及覆盖计划
summary.json                  类别均值、相对第一个模型的配对差值/胜出case数
preflight.json / preflight.log CPU预检阶段、耗时、RSS与调用位置
render.json / render.log       CPU绘图case序号、耗时、RSS与调用位置
status.json                   planned_only / searching / evaluating / evaluated / rendering / complete / failed
search.json                   各类候选预算、检查/命中/保留数及停止原因（误差搜索）
scheduler.json                每个worker任务/模型的GPU、PID、日志和状态
sampling.json                 本次采样seed、auto/fixed模式及评估噪声seed
logs/                         各模型/分片运行日志
 event/                       事件风格
   train/UE1/<case_id>/
   test/UE1/<case_id>/
   ... UE2–UE7、RE2/RE3/RE5、REGULAR_BACKGROUND、UNCONFIRMED
 action/                      动作风格
   train/STOP/<case_id>/
   test/STOP/<case_id>/
   ... DECELERATE、RESUME、LANE_CHANGE_LEFT/RIGHT、KEEP、UNCOND
```

每个类别还有 `summary.json`，即使无案例也保留目录并报告shortfall。每个case目录：

- `comparison.png` / `comparison.pdf`：论文式组合图，上方为真实当前左/前/右RGB并投影GT及所有方法的waypoints，下方为录制道路、车道线和目标框组成的俯视场景，分别叠加Route与约2秒waypoints。有已标定的第三人称截图时加入该视角。固定颜色对应方法，GT为黑色虚线；方法名、step、EMA、event/action、实际token/文字动作、ADE写在图上。
- `model_01.png` / `.pdf` 等：相同布局的单模型对GT，方便检查重叠遮挡。
- `input_history.png`：额外的输入审计拼图，完整展示不同模型实际使用的历史RGB及米制轨迹。当前预测不投影到历史帧，避免坐标时刻错配。
- `inputs/model_01/input_rgb_*.png`：实际预处理函数返回的RGB；bev_only只保存真正使用的当前图，其它模型保留自己的1/4图。不同输入不会冒充同一历史。
- `case.json`：ID、scenario/run、帧号、物理路线、event/动作、模型实际token与文字动作、输入状态和loss/ADE/FDE；不塞长坐标数组。
- `trajectories.json`：需要复画时使用的GT及预测坐标。
- `visualization_calibration.json`：该例实际使用的相机和道路图显示标定；`case.json`另记素材来源哈希。

内部 `_models/` 保留共享评估器原始逐例审计，主线的prompt/条件/动作门控等以原运行器已记录字段为准；`_plan/` 保留每模型选帧清单；`_cases/` 保存去重渲染。两种风格重合的case只推理一次，文件优先硬链接，单独复制分类目录也可阅读。

## 场景视角与标定

默认读同一anchor的 `hdmap/帧.png` 和 `bboxes/帧.pkl` 绘制道路、车道线、车辆/行人；缺道路图时回退原始LAZ点云，再无点云则使用米制坐标。缺目标框时只画明确记录为示意的自车框。文件存在但坏数据/标定尺寸不符会报错。场景标注只用于显示，绝不额外输入模型。历史审计图仍显示点云，缺LAZ会记录在JSON中。

所有几何使用LEAD `inv(ego_matrix) @ world` 的 **CARLA ego x向前、y向右**；按照原采集代码反转道路图存盘旋转。RGB默认优先读取同anchor `metas/*.pkl` 的 `sensor_information.camera_calibration`，按采集端rgb_1/2/3顺序对应三个分块；只有缺字段/缺meta时才回退只读 `lead/lead/common/config_base.py` 的LEAD三相机名义标定：三张384×384、水平FOV60°、yaw=-54.5/0/+54.5°，各自位置独立。采用水平地面近似投影（默认z=0），不做遮挡检测；坡道、俯仰、被遮挡的远端轨迹可能不能贴合路面，不把这种显示偏差当轨迹误差。GT和所有方法使用相同投影，数值指标仍由原评估器计算。来源/回退原因写入case JSON，图注也注明实际录制/显式/默认回退标定；REPORT和manifest按去重case汇总来源。已知相机数量不兼容或标定坏数据会报错。轨迹线段在近平面及四条视野边界精确裁剪，画面外端点之间的可见部分仍显示，不跨不可见区间连线。源码链路、旧投影函数的约定差异和数值核对见 [PROJECTION_AUDIT_20260922.md](PROJECTION_AUDIT_20260922.md)。

**道路俯视图不是高空摄影截图。** 这套工具将各方法的开环轨迹叠到同一段真实录制场景，不启动CARLA重新演绎。若原录制没有保存第三人称图，无法从三张行车RGB还原真实的高空视角。已有 `3rd_person/` 图且知道当时相机标定时，可在 [comparison_cameras.example.json](comparison_cameras.example.json) 的副本中填入 `third_person`，例如下列采集配置；必须按实际数据修改，不能仅凭图片尺寸猜外参：

```json
"third_person": {
  "name": "Recorded overhead",
  "pos": [30.0, 0.0, 75.0],
  "rot": [0.0, -90.0, 0.0],
  "width": 786,
  "height": 786,
  "fov": 90.0,
  "image_pattern": "3rd_person/{frame:04d}.jpg"
}
```

`pos`为ego坐标米制位置，`rot`依次为roll/pitch/yaw（度，CARLA约定），`fov`为水平角；这里的例子来自仓库HIGH_BEV配置，不证明某张图使用了该配置。填好完整JSON后，在脚本设置 `CAMERA_CONFIG="/path/to/cameras.json"` 或传 `--camera-config`。每条route内的 `comparison_camera.json` 优先于全局JSON；两种显式JSON都优先于每帧录制标定。混合采集配置默认自动读取每帧meta，也可用route JSON明确覆盖。图像尺寸严格匹配，不静默拉伸标定。未保存第三人称图时仍输出RGB＋道路俯视布局，并记录 `third_person_available=false`。


## 选权重和公平配对

优先使用训练产生的 `best.pt`，核对其中step与完整val最优记录；有冲突立即失败。缺best时只从run直属目录的 `.pt` 中选择有**该权重本步完整val**记录的最佳文件，并标记 `best_available_full_val`；不把latest里的历史best分数当latest本身成绩。没有这种证据则失败。best存在但缺完整val文件时，清单明确 `validation_verified=false`，只能依据训练保存的best身份。选优使用原 `best_selection_metric` 与route/waypoint权重，**不按FM loss、test或可视化结果挑权重**。

评估固定EMA，复用 `training_core.evaluate` 的GT提取、FM诊断损失和纯噪声Euler采样；所有模型使用相同case身份与评估seed，分别恢复各自训练条件。先验噪声仍保持各模型训练配置，清单记录所有配置差异；相同评估噪声不意味着所有训练配置是单因素消融。

三split原索引哈希、实际有效train/test帧集合、物理路线隔离、轨迹形状/缩放/时间与损失口径都检查一致。每个checkpoint的训练源码、冻结权重、依赖及条件合同仍调用正式入口严格校验；旧权重不能通过新工具跳过校验。计划后权重/选帧清单发生变化会拒绝。丢case、重复case、GT不同也会失败，不静默取交集。

`train` 指训练代码 `read_rows` 返回的有效train池，包括按既有规则移入train的开发路线；事件均衡训练并不保证池中每个帧都实际采样过。这里测试的是训练划分，不声称所选每帧都已被优化器看过。`test` 同理使用有效test池。均不使用训练抽样的重复权重来凑案例。

Event分组复用正式评估的完整语义桶，special_filtered仍属于相应event；多个并发事件保留多个分组，汇总不能把各桶相加当独立总量。Action分组复用原始Phase3候选/full map的六动作＋UNCOND，UNCOND不是KEEP。所有模型使用一套审计分类标签；**没有启用token的模型仍不接收token**，其JSON/图标为DISABLED。开启token的模型实际输入还要与审计标签逐帧一致。文字动作保留模型自身Phase1/2门控结果，和不经文字门控的token分开显示。

## 文件搬迁与运行边界

保持原 `checkpoints/` 结构时，工具会尝试把旧绝对路径的 `AutoMoT/`、`checkpoints/`、`lead_data/` 后缀映射到本机；LoRA主线优先恢复run旁的固定副本，不能重新搜索adapter。full map应连同原manifest与candidate文件一起搬迁，不能修改manifest哈希来骗过校验。

目录结构不同可显式指定：

```bash
bash qwen3vl_local/action_prior/compare_checkpoints.sh \
  --data-root lead_data \
  --data-dir checkpoints/action_prior_data \
  --event-balance-index checkpoints/action_prior_prepared/full_原hash/full_event_mapping.jsonl
```

还有 `--model-dir`、`--lead-bev-ckpt`、`--high-level-action-index`、`--prior-labels`、两个阶段的training-index覆盖项。这些是原文件搬迁，仍检查内容。`--event-balance-index` 影响模型原条件恢复，不能给原本没有full map的uniform模型临时加训练条件；若所有模型都不使用full map，使用 `--label-index` 单独提供分类来源，保持模型原split/条件，配对不一致仍拒绝。工具不会重建或重标注数据。

本机累计86项相关回归通过（含真实CPU子进程的并发队列、环境隔离、动态补位、失败/信号清理及输出路径检查），其中53项为对比与可视化无模型回归（含分类配额、投影方向/外参、道路图旋转、合成图片端到端PNG/PDF发布），并用一条已有train-only、通过异常时长过滤的录制路线检查真实RGB/道路/车辆框与GT绘图，缺PyTorch/laspy、真实权重及只读offline runner，尚未验证真实Qwen/BEV GPU推理。真实运行应在原训练环境进行。当前默认每类最多50候选、保留5个waypoint大误差案例用于可视化诊断，不替代全量test；查看test并据此改模型后，应记录开发曝光，不再称这些case为盲测。

本次两包详细审计见 [TRAINING_AUDIT_20260922.md](../action_expert_ablation/bev_only/TRAINING_AUDIT_20260922.md)。
