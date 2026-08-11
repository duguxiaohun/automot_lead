# Agent Loop Phase 1: 四个无 memory 的视觉事实问题

第一轮不是 RS/EVENT 分类器，也不读上一帧 memory。它只为后续 loop 提供当前帧可复核的
视觉事实：

```text
HIGHWAY: YES|NO
STATIC_OBSTACLE: YES|NO
VULNERABLE: YES|NO
TRAFFIC_LIGHT_ABNORMAL: YES|NO
```

训练、原始 Qwen 测试和 LoRA 复测命令见 `SFT_LOOP_PHASE1_RUN.md`。

`YES` 的语义均为“当前最新帧、结合最多几帧短历史，能看见且会影响或足以影响 ego 当前
驾驶决策”。这避免把某个 scenario 中已离开/尚未出现的对象，错误扩散到全场景。

## 标签与 RS/EVENT 的关系

RS/EVENT 是分层审计键，不是四问的答案。下面的对应关系只能用来安排人工审计优先级：

| 当前标注 | 需要重点检查的问题 | 不能直接推导的原因 |
|---|---|---|
| `R3` | HIGHWAY | R3 的拓扑需在 RGB 中可见；直路、宽路或场景名都不够。 |
| `U-E2` | STATIC_OBSTACLE | 需确认事故/停放车、施工物或开门等固定物真实占住 ego 当前走廊。 |
| `U-E1/U-E3/U-E5/U-E6/U-E8` | 后续 DYNAMIC_OBSTACLE loop | 这些都是动态减速、切入、对向侵入、违规抢行或队列状态，不得混入第一轮静态问题。 |
| `U-E4` | VULNERABLE | 需确认行人/骑行者在当前帧仍可见且与 ego 路径有关。 |
| `U-E7` | TRAFFIC_LIGHT_ABNORMAL | U-E7 可以是一般路权不确定；只有控制 ego 的灯真实矛盾/失效才回答 YES。 |
| `U-E6` | TRAFFIC_LIGHT_ABNORMAL | 对方闯红灯是对方违规，不等于灯异常，因此通常应为 NO。 |

因此不可保存“某场景的永远答案”。训练数据必须是 frame-level；场景 × RS × EVENT
矩阵只用于确保每个组合都已抽到不同 town 的真实 RGB。若同一初始组合的视觉事实不一致，
必须按实际可见道路拓扑、信号控制状态或冲突对象拆为独立的视觉子组；不能用初始 RS/EVENT
或场景名把它们压成同一个 YES/NO。

当前真实视觉复核明确排除 `noScenarios`：该集合不是单一可见驾驶语义，且同一 `R3` 内可同时
出现城市主干道与受限出入道路，不能作为统一四问训练来源。后续 matrix、答案表、训练集都必须
跳过这个场景，除非另行建立 route-level 道路拓扑标注。

## Prompt 合同

`prompts.build_phase1_prompt()` 是生产版本：没有 memory、没有 CoT、没有 scenario 名，也不
允许额外文本。它要求模型查看道路、车道、信号灯、车辆、行人和三视角，并显式给出：

- 高速的正证据（受限出入、高速主线/匝道/合流/分流、隔离、连续多车道、开阔走廊）和反例；
- 静态障碍必须是相对道路固定的实体占住 ego 路径；动态切入、抢行、对向侵入和普通跟车全部留给下一轮；
- 弱势参与者必须处于或可能迅速进入 ego 冲突区，远处安全人行道不算；
- 正常红黄绿灯、另一进口的灯、闯红灯车辆不算灯异常；只有控制 ego 的灯矛盾、坏灭或短历史
  中不可信闪烁才算。

当前 prompt 名称为 `sft_loop_phase1_static_obstacle_prompt`。它将历史混合 `OBSTACLE` 拆成
只问固定占道物的 `STATIC_OBSTACLE`；动态车辆冲突将在下一轮单独提问。
该 prompt 从
`PHASE1_FOUR_QUESTION_RGB_AUDIT_20260809.md`、最终 answer table、全帧 manual notes 和代表性
RGB sheet 中补入以下强约束：

- `HIGHWAY` 用“受限出入拓扑链”判断：主线/匝道/合流/分流/出口/connector、导流 gore、
  加减速车道、连续隔离、多同向车道、无普通街道接入。草地、树、空旷、少楼、直路、宽路、
  雾天、护栏、桥/隧道/封闭通道只能作为弱背景，不能单独推出高速。
- `InterurbanActorFlow/R3/R-E1` 这类乡间/城际普通道路是高速负例；`EnterActorFlow*` 的
  R1/R3 正例来自真实匝道/受控主线拓扑，不是因为 R1/R3 标签本身。
- `STATIC_OBSTACLE` 只看固定实体是否占住或封闭 ego 可行驶走廊。正例是事故/失效/停放车压住车道、
  施工锥桶或隔离物封道、打开车门伸入车道；普通前车、排队等灯、救护车横穿、切入/抢行车、
  对向侵入车和安全路边停车均为 NO，留给下一轮动态问题。
- `VULNERABLE` 专门看行人/骑行者等未保护参与者是否进入或即将进入 ego 冲突区；远处 sidewalk
  背景人、停车自行车、广告图或反光不算。
- `TRAFFIC_LIGHT_ABNORMAL` 只看控制系统本身是否坏/矛盾/冲突放行/异常熄灭或闪烁。
  `OppositeVehicleRunningRedLight/U-E6` 是车辆违规冲突，`STATIC_OBSTACLE=NO`，
  `TRAFFIC_LIGHT_ABNORMAL=NO`；普通红灯等待、普通红黄绿相位、无灯路口和不同进口正常异色都不算。

`audit=True` 的 debug prompt 则先输出四条短 `EVIDENCE_*`。它们只能描述可见证据，不能要求
模型暴露 chain-of-thought；用于把每个错答归为道路拓扑、目标物、弱势参与者或信号灯的观察
错误，再修改生产 prompt。

## 训练 / 测试均衡合同

Phase1 虽然每个样本都一次性输出四个答案，但 train/eval 采样必须带一个不可见的
`focus_question` 字段来做均衡。模型 prompt 不显示这个字段，仍然回答全部四行。

采样桶固定为 8 个：

```text
HIGHWAY:YES
HIGHWAY:NO
STATIC_OBSTACLE:YES
STATIC_OBSTACLE:NO
VULNERABLE:YES
VULNERABLE:NO
TRAFFIC_LIGHT_ABNORMAL:YES
TRAFFIC_LIGHT_ABNORMAL:NO
```

训练集、验证集和测试集都按 route 互斥，并在各自 split 内按两层 exact balance：

1. 四个 `focus_question` 之间 `1:1:1:1`。
2. 每个 `focus_question` 内部 `YES:NO = 1:1`。

也就是说，一个 `HIGHWAY:NO` 样本仍然要求模型同时回答 static-obstacle/vulnerable/traffic-light 三项；
只是这个样本被计入高速问题的负例桶，另外三项答案保持真实分布，不在该桶里强行平衡。
`STATIC_OBSTACLE`、`VULNERABLE`、`TRAFFIC_LIGHT_ABNORMAL` 同理。这样可以避免为了平衡高速而破坏其它
问题的真实共现关系，同时让四个问题都有同等训练/测试压力。

实现采样时优先按 route 分组后再抽帧，防止同一 route 的相邻帧同时进入 train/val/test
或在某个 YES 桶里重复过多。推荐流程：

1. 先从最终 `phase1_four_question_answer_table.json` 解析 `scenario × RS × EVENT` 四问答案，
   排除 `noScenarios` 和异常时长/data-missing route。
2. 按 route 做稳定随机 split；同一个 route 只能出现在 train、validation、test 的一侧。
3. 在各 split 内为每帧生成 4 个候选 focus 视图，分别归入上述 8 桶。
4. 构建索引时，train/test（`val_ratio>0` 时还有 val）任一 split 缺少任一原始桶都会硬失败，
   不留下半成品 `frame_index.jsonl`。每个桶非空后，训练或测试才可在桶内稳定 repeat 到同一目标数。
5. `manifest.json.focus_bin_availability` 记录每个 split 的原始八桶数；训练的
   `train_balance.json` 和 eval 的 `metrics.json.sampling_verification` 同时记录原始数与最终等量数。
   任一最终 work/case 桶不是目标数会立刻抛错。
6. 训练中 periodic validation 使用 validation split 的 8 桶小均衡集，只看 teacher-forced
   loss 和答案 token accuracy，用来判断是否过拟合。
7. 正式 eval 默认也用 test split 的 8 桶均衡集报告；均衡指标只用于四问可比测试，不能替代未来的
   full-distribution 部署评估。

当前 `build_dataset.py` 默认 `test_ratio=0.10`、`val_ratio=0.05`。如果旧
`frame_index.jsonl` 是在 `val_ratio=0.00` 下生成的，必须重构一次数据索引，否则训练中
没有独立 validation 曲线，只能训练后再跑完整 eval。

## RGB-first 审计

先生成矩阵：

```bash
cd AutoMoT
python -m qwen3vl_local.sft_loop_phase1.audit_matrix --samples-per-town 3
```

它会先剔除超过 90 秒的非白名单 route，读取 `collection_output/*_result.json` 的逐帧
RS/EVENT，并在 `keyframe_filter/collection_output/phase1_four_question_audit/` 写入：

- `phase1_four_question_matrix.json`：全部 `scenario × primary RS × primary EVENT` 组合；
- `sheets/*.jpg`：每组每个 town 三条不同 route、每条 route 三个时间分散的实际 stitched RGB。

注意：该矩阵用于组合级审计和交叉 town/id 抽查；对已选 route 的**逐帧**核验必须再运行
原始标签逐帧 RGB 备料脚本。该脚本只把原始 `collection_output/*_result.json` 中已经存在的
RS/EVENT 标签叠到每帧 stitched RGB 上，供人工视觉审核；它不重算标签，也不自动产生四问答案：

```bash
python -m qwen3vl_local.sft_loop_phase1.fullframe_rgb_label_review \
  --scenarios <ScenarioA>,<ScenarioB> \
  --output-dir keyframe_filter/collection_output/phase1_fullframe_rgb_original_labels
```

输出目录结构为
`keyframe_filter/collection_output/phase1_fullframe_rgb_original_labels/<Scenario>/<Town>/<run_id>/sheets/*.jpg`，
每张 sheet 默认包含 12 帧，并在图上标出原始帧号、RS 和 EVENT。`review_manifest.json`
只说明 evidence 已生成，不代表人工已审核。人工审核进度必须单独记录到
`keyframe_filter/collection_output/phase1_four_question_audit/manual_visual_audit_notes.jsonl`。

每个 scenario 的每个 town 必须检查三条独立 route 的全部帧；随后再随机抽取矩阵中未用于
初审的 route 作复查。不能只看 scenario 名、几个代表帧、既有 EVENT 名称，或把 evidence
生成进度当成视觉审核进度。

用户已确认的统一口径只适用于完成 RGB 复核后的视觉子组，而不再适用于未经验证的初始
`scenario × RS × EVENT`。矩阵的 RGB 联系表用于发现需要拆分的 town/route/时间段；最终答案表
必须保存原始组键、视觉子组键、覆盖的 towns/routes、逐帧审阅状态和人工确认的四项答案。当前
最终答案表的默认行已由全帧复核冻结；唯一已发现的混合拓扑通过
`parked_obstacle_town12_limited_access_fast_road` 明示，且必须由 route-level RGB topology 标注触发。
`U-E2` 的已确认静态占道子组即使在少数边界帧被遮挡，也统一是 `STATIC_OBSTACLE=YES`；这不授权把
没有障碍证据的其它初始组也映射为 YES。审计输出是可再生 evidence，按项目规则不入库。

2026-08-08 的首批实图复核已经显示这一门槛确有必要：
`CrossingBicycleFlow × R4 × U-E4` 的抽样帧能直接看到驶入冲突区的自行车；但
`ParkedObstacle × R1 × U-E2` 的时间分散样本里，部分当前帧已看不清明确的占道物。
所以仍需检查组内 RGB 是否有标注错误或需要拆新的 EVENT；但在当前用户确认的组合级合同下，
`U-E2 -> STATIC_OBSTACLE=YES` 是正式训练规则。旧 `OBSTACLE` 混合任务的历史结果只可用于
分析错因，不能与新任务的 F1 横比。

全量 collection 很大时可拆批生成，并合并小型 manifest（不复制 evidence 图）：

```bash
cd AutoMoT
python -m qwen3vl_local.sft_loop_phase1.audit_matrix \
  --merge-inputs keyframe_filter/collection_output/phase1_four_question_audit/batch_a,keyframe_filter/collection_output/phase1_four_question_audit/batch_b \
  --output-dir keyframe_filter/collection_output/phase1_four_question_audit/all_merged
```

## 训练前门槛

1. 每一个场景 × RS × EVENT 组合均有可打开的 RGB 证据；异常时长和数据缺失 route 已剔除。
2. 每个问题的正、负、模糊例都用 `audit=True` 跑 base Qwen，保存 model 的 `EVIDENCE_*` 与四答；
   先按错因更新 prompt，再冻结 prompt 版本。
3. 只有人工确认的 frame-level 四元标签或由它们组成的同质视觉子组，才可作为
   `build_phase1_target()` 的监督；模糊/遮挡帧应单列为 review 或从第一阶段训练集中排除，
   不能为了凑齐场景级表格强行写 YES/NO。
