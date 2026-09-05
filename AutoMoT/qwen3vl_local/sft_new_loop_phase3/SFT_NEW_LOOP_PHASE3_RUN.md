# SFT New Loop Phase3 运行说明

`sft_new_loop_phase3` 是 Phase1（RS + 三个可见事实）和 Phase2（EVENT）之后的
**high-level 动作决策**阶段。它把前两阶段已经确定的道路结构与异常事件标志当作
待核对前提写进 prompt；R-E2/R-E3/R-E5 可由显式导航/历史 gate 或
`dispatch.plan_candidate_requests` 提出候选，再用同一轮 invalid + 动作问答核对。

- 输入只有一个 system turn 和一个 user turn；
- user turn = 四帧（或两端点）拼接 RGB history + 场景前提文本 + route 目标点的
  ego 相对坐标；
- 不渲染任何 `R1/R4/U-E2/UE3` 之类的数据集 code，也没有 synthetic assistant 前缀；
- 每个问题组最后都回答 `INVALID_ACTION_CONTEXT`。

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
five action YES/NO lines + INVALID_ACTION_CONTEXT
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

## 3. 五个 high-level 动作

| 动作 | 含义 | 轨迹判据 |
| --- | --- | --- |
| `DECELERATE` | 明显减速，含为变道/合流等待时机的减速 | 排除即时 1.5s 持续停车后，未来 2s 首先达到减速阈值 `max(1.2 m/s, 20%)` |
| `STOP` | 停车 / 保持静止等待 | 未来 1.5s 内有连续两帧 ≤ 0.5 m/s；排除已停后持续起步 |
| `RESUME` | 持续增加速度，含从静止起步或绕行中提速，不暗示已经完成让行 | 未来 2s 连续两个采样达到加速阈值，且没有应优先处理的减速/持续停车 |
| `LANE_CHANGE_LEFT` | 向左变道 / 借对向车道 / 向左合流 | 未来 3s 同 road 连续两帧确认新 Driving lane；整个观察窗必须 Driving；road 变更/路肩/缺类型不强标 NO；仍需 RGB/车道段复核 |
| `LANE_CHANGE_RIGHT` | 向右变道 / 回原车道 / 向右驶出 | 同上，方向为右 |

`DECELERATE / STOP / RESUME` 互斥；变道与纵向动作独立。五行全 NO 表示
“没有达到所问高层动作的判据”，这是合法结果，不是 invalid；不表示逐帧精确恒速。
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
`LANE_CHANGE_*` / `NONE` / 组合）尽量均分，保证五个动作都有足够正类。
train 用 route 轮转选帧，val/test 用确定性抽样。

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

一键 base + LoRA + audit prompt + 错例审计包：

```bash
ADAPTER_DIR=checkpoints/sft_new_loop_phase3_runs/latest \
  bash qwen3vl_local/sft_new_loop_phase3/eval.sh
```

传 run 根目录时，`eval.py` 与 `eval.sh` 都按
`best_generation/ → final/ → fallback_generation/` 解析，并要求候选目录中存在
`sft_new_loop_phase3_adapter_config.json`，不会因残留空目录误选权重。
adapter 的 `production_prompt_sha256` 与当前 prompt 不一致时会直接报错，
避免拿不同 prompt 合同的 adapter 互相比较。

## 9. 错例 RGB 审计包

`eval.sh` 最后会自动调用 `audit_eval_cases.py`；也可以单独对任意 eval 目录跑：

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
（本该全 NO 却给了动作）、`invalid_answer`（格式失效）。
每个错例目录里带一份 `audit_note.md`，含逐帧复核清单：情境是否可信、自车是否已在
刹车/静止、是否仍在同两条车道线之间、是否被弯道误判成变道、目标点侧向是否与所需
变道一致。

## 10. 一键全流程与输入合同矩阵

```bash
bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
bash qwen3vl_local/sft_new_loop_phase3/run_rgb_mode_matrix.sh
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
