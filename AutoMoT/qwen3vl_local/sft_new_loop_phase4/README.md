# Phase4：条件成立即推进

最新修复见[FOURTH_AUDIT_FIXES_20260930.md](FOURTH_AUDIT_FIXES_20260930.md)：旧格式catchup拒绝构建，须逐帧复审；回放遇真值缺口记录删失，后续YES另起延迟区间。222项测试通过；2/4图各429题、121待审均与v6逐行一致。任务合同v7、标定代码v5；提示词v5、人工标注v3和运行快照v6不变，须新数据、新run。本轮无新增RGB目视审阅。累计完整审计仍为224条/26,418帧，92/278单元达到三个ID，剩476条计划序列、18单元来源不足；未完成全覆盖。

Phase4维护事件实例与当前阶段，LoRA只判断某条转移的条件：`YES / NO`。动作先验由状态机派生，不再训练动作选择问答。Phase1/2建立事件后，把结果交给`runtime.handoff`；正常推进不重新问Phase1/2，事件轴完成且纵向约束已稳定后退出实例；失效、持续不确定、许可长期未执行或外部执行故障才请求复核。可并发维护多个实例。

本目录是独立实验，不替换Phase3稳定版v23_io1、Action默认来源、旧adapter或冻结数据。默认本地Qwen3.5-4B；旧Qwen3-VL adapter不兼容。运行不下载模型。

最新逐帧边界标定见 [RGB_BOUNDARIES_V3_20260929.md](RGB_BOUNDARIES_V3_20260929.md)。本轮复查45窗、701张不同RGB、36条物理路线、10个Town；其中新增125张，累计审阅1,541张不同RGB。覆盖十事件，不代表全数据集逐帧认证。

| 事件/路线分支 | 最小事件流程 |
|---|---|
| UE1、UE3、UE5 | YIELD → PROCEED → DONE |
| UE4横穿、沿路跟随 | YIELD → PROCEED → DONE |
| UE6、UE7、RE5 | YIELD → PROCEED → DONE |
| UE2绕行、UE4沿路骑行绕行 | WAIT → DEPART → PASS → RETURN → DONE |
| 上述绕行且路线不要求返回 | WAIT → DEPART → PASS → DONE |
| UE2本车道通过 | YIELD → PROCEED → DONE |
| RE2变道、RE3汇入/驶出 | WAIT → CROSS → DONE |

PROCEED期间出现新冲突可回到YIELD。纵向用共享小状态因子区分STABLE、APPROACH、FOLLOW、HOLD、RECOVER，对应KEEP、DECELERATE、KEEP、STOP、RESUME；因此“等待进入走廊”不强制停车，已稳定跟随也不持续输出减速。几何完成可与减速/停车同时成立：此时state=DONE但instance_complete=false，横向KEEP，纵向继续处理直到STABLE/FOLLOW。不能仅凭state字符串清空实例。左右方向、是否返回及目标走廊来自当时可用的导航/因果跟踪，不从未来GT路线回推。RE3允许FORWARD；路段弯曲不能产生变道token。

`controller.Episode`每个新观察至多推进事件轴和纵向轴各一次。已开始横向运动时，对“完成了吗”回答NO仍保持该运动先验。YES许可需要执行器回执`acknowledge(instance_id, frame_id)`；未执行的许可下一观察重新判断，不一直锁存。执行端应确认**实际开始执行**，不能以模型YES自动充当回执。返回的是目标走廊约束和先验，不是直接执行的油门/转向命令；到达目标后不能重复横移到下一条车道。

`action_prior.export`输出七类token名称/编号、文字、每个实例的纵横向约束。UNKNOWN/INVALID不冒充UNCOND；并发停车/减速约束保留，冲突横向目标交回复核，并保留STOP/DECELERATE及conflict_instances/recheck_instances；runtime持久化实例复核状态，可调用revalidate逐项解决。临时UNKNOWN仍为UNKNOWN。只有instance_complete=true才释放该实例约束；复核时保留已有STOP/DECELERATE约束。

提示词、训练与推理共用`prompts.py`。视觉输入为原始2/4张因果RGB；文字只展示“当前状态”和“候选下一状态”，把转移原因、方向及动作含义融入这两个描述，最后问是否转移。候选下一状态是流程目标，不是下一帧真实状态。速度及帧号仅用于输入检查，不渲染到提示词；不展示整个状态链或后续分支。场景名、Town、route ID、未来轨迹、控制量、标注事实和审计文字不进入模型。图片有因果帧号和内容哈希检查；初始帧<4不使用，缺历史不复制填充。训练及adapter保存observation_contract；实时/序列与训练图数和历史间隔必须一致。Phase4Loop(rgb_mode=2或4)显式选择，快照也保存该合同。

默认标注使用`reviewed_state_pairs_v3.json`。用户要求按不同事件、Town和路线的实际RGB决定前后范围，上一版直接固定±0.5秒的实现已撤销。

- `reviewed_transition_band`逐帧声明readiness/catchup/uncertain/excluded，并记录可见参考帧和停止采旧问题的理由。参考帧只用于汇总偏移，不能自动产生YES；不同路线的前后范围可以不对称。
- readiness要求当前条件成立；完成类转移必须已经观察到完成，不因“快完成”提前标YES。catchup仍问同一旧状态对，须确认后继已达成并审核冲突是否阻止本条转移，单独统计。current_conflict=true时必须明确transition_blocking_conflict；独立纵向限制不否定已经完成的几何事实，缺少作用范围则要求复审。
- 在下一阶段、新冲突或普通行进占主导时停止采旧问题；instance_boundary和calibration_anomaly分别记录人工确认的实例边界与采集异常截止；截止之后不自动标NO。`review_end`表示只看到窗口结束，还未确定最晚边界。
- `context_valid=false`为内部INVALID，证据不足为内部UNKNOWN，进入`review_queue.jsonl`，不进入YES/NO模型监督。每帧绑定RGB SHA256、当帧观察和审核来源。
- 旧`transition_point_window`构建输入明确拒绝；v2标注仅保留历史。旧逐帧事实格式仅保留readiness；旧slice=catchup和旧catchup_label均拒绝，须人工复审为reviewed_transition_band，不自动迁移时间区间。

新默认含46条转移审核记录和11条保留的既有区间，构建429道二元开发题（323 readiness、106 catchup）及121条待审，全部train-only。**缺独立三split和完整分支标注，不能正式开训；也没有训练好的Phase4 LoRA。** 下列命令从`AutoMoT/`运行。

```bash
# 看完整绕行/路线不返回/骑行绕行demo；无需GPU，答案是合成条件
python -m qwen3vl_local.sft_new_loop_phase4.demo
python -m qwen3vl_local.sft_new_loop_phase4.demo --no-return --direction RIGHT
python -m qwen3vl_local.sft_new_loop_phase4.demo --event U-E4 --branch cyclist_bypass
python -m qwen3vl_local.sft_new_loop_phase4.demo --event R-E3 --direction FORWARD

# 审核后的区间→开发数据；output必须是新目录
python -m qwen3vl_local.sft_new_loop_phase4.dataset \
  --annotations qwen3vl_local/sft_new_loop_phase4/reviewed_state_pairs_v3.json \
  --output-dir checkpoints/sft_new_loop_phase4_data_v7 --rgb-mode 4

# 只检查/回放采样，不伪装成正式训练通过
DATASET=checkpoints/sft_new_loop_phase4_data_v7 bash qwen3vl_local/sft_new_loop_phase4/train.sh check
python -m qwen3vl_local.sft_new_loop_phase4.preflight \
  --dataset checkpoints/sft_new_loop_phase4_data_v7 --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B

# 有独立标注三split及完整本地基座后执行；未满足会拒绝
ANNOTATIONS=/path/to/reviewed_state_pairs.json bash qwen3vl_local/sft_new_loop_phase4/run_full_pipeline.sh
GPU_IDS=0 DATASET=/path/to/ready_dataset bash qwen3vl_local/sft_new_loop_phase4/train.sh single
GPU_IDS=0,1,2,3 DATASET=/path/to/ready_dataset bash qwen3vl_local/sft_new_loop_phase4/train.sh ddp
# 2/4RGB分别构建、分别新run，不混用身份
ANNOTATIONS=/path/to/reviewed_state_pairs.json bash qwen3vl_local/sft_new_loop_phase4/run_rgb_mode_matrix.sh

# 严格原合同的epoch边界恢复；不能恢复epoch中途
DATASET=/path/to/ready_dataset bash qwen3vl_local/sft_new_loop_phase4/train.sh single \
  --resume checkpoints/sft_new_loop_phase4_runs/OLD/epoch_002 --epochs 7

# 指定adapter目录；best.json记录本run最佳完整val，test不选优
bash qwen3vl_local/sft_new_loop_phase4/eval.sh \
  --dataset /path/to/ready_dataset --adapter /path/to/run/epoch_003 \
  --split test --output-dir checkpoints/sft_new_loop_phase4_eval_NEW
python -m qwen3vl_local.sft_new_loop_phase4.replay \
  --sequence /path/to/sequence.json --adapter /path/to/run/epoch_003 \
  --output checkpoints/phase4_replay_NEW.json
python -m qwen3vl_local.sft_new_loop_phase4.audit_bundle \
  --run /path/to/run --output /path/to/phase4_audit.zip
```

默认每轮按事件/转移/答案/readiness、catchup分层、物理路线轮转；同一问答每轮最多一次，同物理帧跨所有问题全局cap8。容量不足报错，不合成数据或静默补重复。epoch旋转队列起点补偿未入选样本。验证/测试不重采样。2/4RGB使用同一标注、同一物理路线划分与seed，输入长度改变需独立合同和run。

训练复用本地Qwen3.5/DeltaNet LoRA加载，监督仅assistant答案与结束符；支持视觉LoRA off/merger/last4/all、单卡/DDP、梯度累积尾部、逐epoch完整自由生成验证、best/latest/final与优化器epoch边界恢复。源码、提示词、数据manifest、采样参数、基座资产均绑定合同。`selection.select_best`先核验兼容性再比较val；不会按“最新目录”捡adapter。

与Phase3的功能对应：构建/预检/单卡-DDP/两四图矩阵/完整生成评估/错误题JSONL/审计包/源码合同已有独立入口；原Phase3六动作choice或多动作binary标签不继承，因为本任务输出必须是转移条件。RGB审阅面板用`rgb_audit.py`；在线调用入口用`Phase4Loop`，未接管现有CARLA/Action生产runner，未修改其默认先验来源。没有宣称所有Phase3旧实验开关逐一兼容。

序列文件包含`initial`（Episode构造字段）和按帧升序的`observations`；每项有`frame_id`、`observation={frame_id,history_frames,speed_mps}`、`images`相对路径、`image_sha256`、可选`truth={edge:答案}`、`context_valid`和来自外部的`execution_committed`。必须按当前**预测状态**重新构题，不逐帧灌入GT阶段。缺边真值计uncovered，不能补NO；已开启区间在未知真值处以truth_unknown删失，后续YES重新起算。整帧缺失和不再询问的边也分别截止；last_ready_frame记录最后一个已证实成立帧。精确延迟仅覆盖逐帧已审核的连续YES区间，不能把旧跨缺口指标直接混比。报告提前YES、答案/控制器接受/执行确认各自的延迟、未执行右删失记录和整体完成；delay_frames仅指执行确认延迟，不能再当作答案延迟。正常明确NO的等待不判停滞；这是固定录制观测上的预测状态回放，不能冒称车辆动作改变后的闭环仿真。

审计包仅打包指定run的指标/错误题/合同，30MB上限，超限明确报错，不上传、不包含模型权重或本机聊天。RGB与生成输出位于忽略目录或checkpoints，不进源码仓库。

本机验证使用独立`/tmp/automot-qwen35-env`（Transformers 5.3.0），真实小型Qwen3.5多模态网络做过条件答案mask和LoRA反向测试；完整4B权重缺失，因此未真实4B/GPU/DDP多卡或CARLA验收。本轮172项专项通过；另完成2/4图真实开发索引构建、各7轮×world_size 1/4采样回放、全部历史图像哈希核验及shell语法检查。当前开发规则仍需要独立路线标注、模型训练与同题序列对照之后才能评价效果。

`coverage_report`同时检查十事件的独立路线/正负支持和每个事件流程边及实际可达公共纵向边的训练转移正负支持（仅readiness），包含UE2不返回/本车道通过、UE4跟随/绕行及不返回分支；catchup的YES不能填补训练转移边的YES缺额，缺额字段为`missing_transition_edges`，其中公共纵向边另列`missing_longitudinal_edges`（现有数据共270项缺额）。这个支持量门槛只是开训完整性检查，不是泛化能力证明。

完整流水线现在串接构建→预检/训练→按val所选best做独立test→审计打包；MODE=check只做开发采样检查。此接线有进程边界测试，未真实GPU验收，仍不能认证Phase3所有实验开关/原始证据审计已完全对齐。

handoff接收明确完成的Phase1八项（HIGHWAY/STATIC_OBSTACLE/VULNERABLE/TRAFFIC_LIGHT_ABNORMAL/RS1/RS2/RS4/RS5）和Phase2五项（UE1/UE3/UE5/UE6/INVALID_EVENT_CONTEXT）布尔判断；建立实例时缺项拒绝，有效性必须明确False。旧状态补问可用executor或causal_tracker的结构化execution_receipt确认动作已经开始，绑定当前实例/决定/边/因果观察；具体协议及快照恢复用例见修复说明，不以模型YES充当执行证据。

第二轮审计报告的23个新曝光物理组已写入Phase4 train-only隔离（reaudit_exposure_20260929.json），不加入盲测，不新增监督标签。UE3旁侧通过仍须有既定通道、足够间隙和优先条件，不由提示词自动指定左绕或产生未经确认的变道先验。

第三轮已审的11个新曝光组已登记train-only（third_audit_exposure_20260930.json）。tick在副本上完成整批输入/预测/回执核对后统一提交；抛异常时旧状态及待执行许可不变，允许重试或确认原有效决策。跨路口的新目标使用独立instance_id，不能沿用旧许可。
