# Phase3／4 连续片段 batch 实验

> 发布更新：本轮已获用户授权按白名单发布，统一 v60_batch_profiles_20261015（523源），发布前209项相关回归通过。同步最终发布源码后执行本文入口；历史段落中的“未push”描述保留当时状态。

> Phase3 四卡端到端实测已完成，当前优化入口采用 B2／累积2／checkpointing；Phase4 独立对照和固定更新量衔接见 [实测配置与命令](BATCH_TRAINING_20261015.md)。下文旧容量探测步骤保留供追溯，无需再跑 Phase3 探测。

> Phase4 正式 v57 库已交回验收。实际路径和当前补丁后的新视图／首轮命令见 [20261015正式包验收](PHASE4_V57_ACCEPTANCE_20261015.md)。补丁绑定审计SHA，已有旧学生视图需重建，原生题库不动；本轮未push。

最终确认：两阶段当前学生训练统一为两图 `[t-2,t]`（锚点推进2帧），四图 `[t-3,t-2,t-1,t]`（锚点推进1帧）。4卡、每卡2片段时，每次全局前向包含8片段，完整连续段分别需要9个两图采样时刻（原始帧跨度17帧）／11个连续四图时刻。每个片段独立前向，attention不会跨片段；共享源图只缓存本批CPU解码，GPU视觉token仍按每个样本计费，不能用9／11张唯一源图估算显存。

当前Phase4统一入口是 `sft_new_loop_phase4.rgb_short.train`，同时负责两图和四图。旧原生 `sft_new_loop_phase4.train`、生产回放、标定、老师和原生稀疏四图数据合同保留，供旧数据来源核验；新实验不要调用旧训练入口。新四图输入的目标来自原生四图参考，明确记录 transferred_reference，不能称为新视图已获人工可见性认证。

## 先生成新学生视图，再测显存

在服务器 `AutoMoT` 目录和既有训练环境中使用同步后的本轮代码。本轮代码经用户授权白名单发布，服务器同步新版后再使用。不要沿用旧 `server_phase3_epoch.sh` 的固定旧来源/旧采样验收流程，也不要重做Phase3数据准备。

以下变量中，P4_NATIVE必须指向已完成的 **v57** 原生 `data4`，不是旧v54池。加载器仍会拒绝旧生产合同；不改旧SHA绕过。若服务器v57正式CPU池尚未完成，应先完成已经安排的那一次正式池。

```bash
cd /mnt/workspace/cee3sgh/AutoMoT
conda activate automot
export DATA_ROOT=lead_data
export MODEL_DIR=checkpoints/Qwen3.5-4B
export PREPARED_ROOT=checkpoints/phase3_prepare_20261012_231356_193369568
export INDEX="$(python -c 'import json,os; from pathlib import Path; p=Path(os.environ["PREPARED_ROOT"]); print(json.loads((p/"prepare_result.json").read_text())["bindings"]["index"]["logical_path"])')"
export PROMPT_VARIANT=v23_rgb_stage_candidate_20261006
# 将下行改为服务器已完成的 v57 data4 路径：
export P4_NATIVE=/path/to/completed_v57/data4
export VIEW_RUN="checkpoints/sequence_view_$(date +%Y%m%d_%H%M%S_%N)"
python -m qwen3vl_local.sft_new_loop_phase4.rgb_short.build \
  --dataset4 "$P4_NATIVE" --data-root "$DATA_ROOT" --output "$VIEW_RUN"
export DATASET="$VIEW_RUN/view"
```

这一步只从既有原生数据及RGB生成新视图和图片身份sidecar，不重放老师、不复制RGB、不覆盖旧目录。会读取新增时间点的实际图片；缺图或源漂移则拒绝，不伪造跨路线历史。

分别测试两阶段，每个命令都测两图／四图及每卡batch=1至8，逐卡、逐配置独立进程，OOM后该卡该模式停止增大：

```bash
python -m qwen3vl_local.audit_joint.batch_probe \
  --phase 3 --index "$INDEX" --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --prompt-variant "$PROMPT_VARIANT" --gpus 0,1,2,3 \
  --output "checkpoints/phase3_batch_probe_$(date +%Y%m%d_%H%M%S_%N)"

python -m qwen3vl_local.audit_joint.batch_probe \
  --phase 4 --dataset "$DATASET" --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --gpus 0,1,2,3 \
  --output "checkpoints/phase4_batch_probe_$(date +%Y%m%d_%H%M%S_%N)"
```

可以用 `--batches 1 2 4` 先粗测，也可用 `--modes 2` 单独测试。默认每个配置覆盖十事件各2个实际batch，包括前向、反向、梯度裁剪、AdamW状态初始化及后续更新。没有保存模型权重或优化器文件。Phase3默认不开gradient checkpointing，Phase4默认开启，与当前训练默认一致；若显式更改探测的 `--gradient-checkpointing`，训练也必须同步设置。

交回两个探测目录的 `summary.json` 即可；每个trial另有完整JSON及日志。结果包括每卡 allocated/reserved 峰值、事件题目ID、token长度、耗时和配置。推荐值取所有被测卡均能容纳且预留15%显存的最大**已测试**batch，不宣称绝对最大容量，也不把单卡测试说成四卡DDP验收。八片段/四卡的batch size是每卡2，不是梯度累积2。

## 实测后使用真实 batch

两种模式可以使用不同BATCH_SIZE。每个event的预算必须整除 `卡数 × BATCH_SIZE`；默认1024/event支持4卡B=1/2/4/8。B=3时可明确使用1032/event（Phase4总预算10320）；代码拒绝不能整除的预算，不暗中丢尾部或用重复padding冒充等量。

Phase3当前实测配置：

```bash
GPU_IDS=0,1,2,3 INDEX="$INDEX" DATA_ROOT="$DATA_ROOT" MODEL_DIR="$MODEL_DIR" \
PROMPT_VARIANT="$PROMPT_VARIANT" HISTORY_RGB_MODE=2rgb_short ACTION_OUTPUT_MODE=choice \
BATCH_SIZE=2 FOCUS_BALANCE_COUNT=1024 NUM_EPOCHS=3 GRAD_ACCUM=2 GRADIENT_CHECKPOINTING=1 \
EVAL_STEPS=1280 GENERATION_EVAL_STEPS=1280 EVAL_BALANCE_COUNT=32 GENERATION_EVAL_BALANCE_COUNT=32 \
SAVE_STEPS=0 OUTPUT_DIR="checkpoints/phase3_batch_rgb2_$(date +%Y%m%d_%H%M%S_%N)" \
bash qwen3vl_local/audit_joint/train_phase3_batch.sh ddp
```

四图使用 `HISTORY_RGB_MODE=4rgb` 和新的OUTPUT_DIR；默认锚点步长自动为1。B改变时每轮步数为 `10240/(4*B)`，相应调整验证间隔。显存探测若开启checkpointing，训练加 `GRADIENT_CHECKPOINTING=1`。正式训练前可将末尾 `ddp` 换成 `sampling` 做实际CPU采样预检。

Phase3最终test使用新实验的final，完整可用池按最小event容量无放回等量取样：

```bash
python -m qwen3vl_local.sft_new_loop_phase3.eval \
  --index "$INDEX" --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --adapter-dir /path/to/new_phase3_run/final --split test \
  --cases-per-bin 0 --evaluation-distribution equal_events \
  --output-dir /path/to/new_phase3_run/test_equal
```

Phase4继续原自动参考准入，设置当前v57的TEACHER_REFERENCES及FIRST_EPOCH_BUDGET，新视图会重新核对关键支持、隔离及等量评估。首轮示例：

```bash
DATASET="$DATASET" DATA_ROOT="$DATA_ROOT" MODEL_DIR="$MODEL_DIR" \
TEACHER_REFERENCES=/path/to/completed_v57/references4 \
FIRST_EPOCH_BUDGET=/path/to/completed_v57/first_epoch_budget.json \
AUTOMATIC_COVERAGE_POLICY=existing_data_development \
GPU_IDS=0,1,2,3 RGB_MODE=2 BATCH_SIZE=2 \
OUTPUT_DIR="checkpoints/phase4_batch_$(date +%Y%m%d_%H%M%S_%N)" \
bash qwen3vl_local/sft_new_loop_phase4/rgb_short/run_automatic.sh
```

四图改 `RGB_MODE=4`，使用其独立实测batch值和新输出目录。该入口包含重载val和固定test。多epoch训练器会保存实际曝光history，并在epoch边界resume时核对数据、batch、步长、来源和优化器；扩大epoch需相应磁盘预算。Phase3只在同一次多epoch运行中持续记录曝光，原有adapter不含可恢复优化器，不能宣称断点无损续训。

## 采样和评估约束

- 十事件训练配额严格1:1；val/test默认是固定、无放回的十事件等量子集，缺事件拒绝。不会用重复评估凑分母。连续评估块的成员选择不看action/YESNO标签，模式之间保留同一题目成员但可按各自锚点间隔重排；新评估成员与旧实验可能不同，旧成绩不能直接作为相同题集对照。
- 训练优先少见题，再优先少见锚点，再最长连续片段，最后按动作容量目标弥补欠额。新片段未覆盖前不为稀少action单独重复；动作均衡是软目标，报告目标/实际偏差，不能承诺每个batch动作均匀。
- 不够长的序列在同event内接续其它真实序列。拼接的是batch中的独立片段，不是跨路线合成一条历史。
- transition允许相邻片段共享RGB，也允许同一真实帧的不同问题；保留题目身份。train/val/test的物理路线隔离不放宽。
- Phase3、Phase4记录每轮曝光、未见数量、连续段断点和逐batch成员。Phase4交接器能核对每个batch内所有ID、位置和更新步数；仍支持旧batch=1回执的读取。
- Action prior及两类Action消融的共享评估默认十事件无放回等量；共享事件帧通过联合分配仅算一次总体分母。背景不冒充第十一个event，原始语义标签保留；`--evaluation-distribution natural`仅为显式诊断选择。旧checkpoint仍需遵守其源码合同。

本轮未修改生产标定/老师规则，未新采集、未远程操作、未正式GPU训练、未push。CPU/小型真实模型测试不等于目标Qwen3.5-4B四卡实测。

## 本机验收记录

最终相关回归225项通过（9.77秒），含真实小型Qwen两／四图batch的logits、loss与梯度数值检查，以及实际Phase4训练循环两轮的批消费与曝光持久化。前一轮扩展检查另有两项Action消融源码合同测试因本机缺 `leaderboard/team_code/mot_lead_offline_runner.py` 未完成，日志保留；没有宣称全套回归通过。

v58清单515源SHA matched，所声明Python包无额外文件，五个改动shell入口语法通过。既有v57十事件train小池生成最终新视图，两模式各1457题、ID/标签配对、原生数据文件SHA未变；不是重新标定。4卡B=2的首轮CPU计划每event1024；另用16/event/epoch做五轮覆盖检查，累计未重复题数151→278→383→479→575，两模式相同。十事件池大小不同，因此小事件覆盖完后会重复，大事件继续优先未见；不能承诺各event同时完整覆盖。该小池val/test为空，仅用于本地诊断。

完整本机日志和视图：`checkpoints/sequence_batch_20261015/`，当前有效视图为 `real_view_final/view`。轻量回执：`sequence_batch_verification_20261015.json`。目标4B模型权重本机缺失，未提供任何虚构显存数字；实际容量以服务器探测输出为准。

## 空变量报错修正（2026-10-15）

服务器报 `PosixPath('.') has an empty name` 已用 `check_index(Path(""))` 原生复现：空INDEX被Path转换为当前目录，在寻找同目录manifest时失败，尚未模型前向，不能据此判断显存。旧版本先在当前shell设置INDEX、DATA_ROOT、MODEL_DIR、PROMPT_VARIANT，再运行；推荐命令参数使用 `${INDEX:?请先设置INDEX}` 等shell非空守卫。已有正式索引为 `checkpoints/phase3_prepare_20261012_231356_193369568/data/frame_index.jsonl`；路径必须以服务器实际存在为准，不重建或猜选其它索引。

本地补丁在任何worker启动前拒绝空路径、目录型索引、缺manifest/model config/数据根，以及空prompt variant；6项probe专项通过（13项无关测试未重复）。新清单release_sources_v58_probe_inputs_20261015.json保留旧清单；不改训练器/学生视图合同，不要求重建既有v58新视图。该修正已获用户明确授权按白名单发布；服务器同步新版即可获得启动前诊断，旧版本也可直接通过正确设置变量恢复探测。

## 固定样本四卡性能对照（2026-10-15，新增入口）

容量探测结果已收到：Phase3开启checkpointing后GPU0两图B8预留66.31GiB满足15%余量，四图B8预留89.33GiB不满足；两图B2约2.04s/片段、B8约2.29s，四图B2约3.21s、B8约3.36s。但旧probe各B的样本数量/成员不同，不作为严格性能排名。停止盲目扩大B，改用 `audit_joint.batch_benchmark` 固定样本及有效更新批量比较。

新入口同时启动指定的四卡DDP，默认B1/2/4，每event正式计时16次呈现（全局160），每次optimizer更新固定16样本（共10更新），因此四卡累积步数分别4/2/1。两图/四图及各B共享题目成员、每次更新成员，按各自步长在更新组内重排；这份固定对照计划与正式epoch预算不同，不冒充正式训练。每event另预热1次更新（全局160呈现）且不计入吞吐分母，显存峰值包含预热/优化器状态。预热和正式计时都保持十事件1:1，稀少池重复情况明确报告。

默认两个独立进程组重复，顺序B1→2→4再B4→2→1，报告中位数及范围。样本哈希/更新组哈希、每rank完整成员与顺序、更新次数、所有rank完成且满足显存余量均通过，才给最快已测候选；不会只按容量最大选batch。固定训练例/dropout seed/更新批量控制了主要比较条件，不承诺不同batch的浮点/dropout路径逐位相同，也不据短测预测收敛。

在AutoMoT和训练环境中，先同步本轮代码，再运行：

```bash
python -m qwen3vl_local.audit_joint.batch_benchmark \
  --phase 3 \
  --index checkpoints/phase3_prepare_20261012_231356_193369568/data/frame_index.jsonl \
  --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
  --prompt-variant v23_rgb_stage_candidate_20261006 \
  --gradient-checkpointing --gpus 0,1,2,3 \
  --modes 2 4 --batches 1 2 4 \
  --samples-per-event 16 --update-samples 16 --warmup-updates 1 --repeats 2 \
  --timing detailed \
  --output "checkpoints/p3_batch_benchmark_$(date +%Y%m%d_%H%M%S_%N)"
```

分段包括CPU图片读取/校验/解码/processor/组批、整批CPU→GPU传输、forward+loss+训练统计、backward（同步步包括DDP通信）、clip+AdamW。分段显存为阶段结束当前占用及从预热开始的累计峰值，不伪称瞬间数值是该阶段独立峰值。逐阶段CUDA同步会改变调度，故该模式用于定位，不能把它当无插桩吞吐；每个trial的worker.log能看到更新进度。

定位后同一命令改 `--timing end_to_end`、使用新输出目录验证实际吞吐。该模式仅在每microbatch尾同步，与当前训练器一致，阶段字段明确为host_submission_seconds，不冒称GPU完成耗时。总吞吐取最慢rank墙钟，包含读图/组批/计算/更新/记录及末尾等待，不包含模型加载和预热。初始和后续试验的OS缓存/共享负载可能不同，重复范围保留；没有清空系统缓存或设置硬件时钟。

Phase4使用当前 `benchmark_phase4_batch.sh`，固定每次更新32样本，与正式四卡B1／累积8一致；B2相应累积4。工具只读train池，不能代替正式准入。当前benchmark直接调用正式训练的设备端encode_batch，替代早期工具的CPU组批路径；detailed中对应阶段名为encode_batch_device。rule_baseline及run_automatic来源绑定已变，需在最终同步后从现有v57 data4生成新学生视图，不重放老师。完整命令见 [实测配置](BATCH_TRAINING_20261015.md)。

每个输出目录交回 `summary.json` 和 `request.json`。如有失败，另附相应trial的 `worker.log` 和 `rank*.json`；逐步JSONL保留本机供抽查。超时或rank异常会保留已有记录，torchrun终止其它worker；不会保存权重/优化器，也不占用其它GPU。默认单trial上限1800秒、DDP collective上限180秒，可显式调整；超时绝不计作完成。

早期v59工具发布时，实际2进程CPU/Gloo梯度累积与整批更新一致；真实v57小池1457题的六组对照均160呈现／151唯一题、10更新，成员哈希一致。当时尚无GPU实测。随后收到的Phase3四卡结果及当前入口补丁见 [实测配置](BATCH_TRAINING_20261015.md)，不可把早期工具发布状态当作本轮补丁已经push。
