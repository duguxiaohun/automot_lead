# SFT v3 方案说明

## 0. OPSD 与 v4 prompt 同步规则

SFT v3 现在是 **offline OPSD** 路线：每帧仍由 student 先自由生成，student rollout
决定本帧和下一帧的 memory 走向；但训练梯度不再来自 teacher 文本的 hard CE。
privileged teacher 使用同一份 base 模型、在 `eval() + no_grad + disable_adapter()` 下读取
GT road/scene/status/subgoal 上下文，并在**同一段未裁剪的 student rollout token** 上输出
full-vocabulary logits。v3 loss 对 analysis 与
`ROAD_STRUCTURE/SCENE/STATUS/SUBGOAL` token 位置做 forward-KL 分布匹配。
当前代码口径是 `loss_type=forward_kl_teacher_to_student`，不是 JSD；若后续实验要切到
JSD，必须显式新增 loss type 开关并同步 v3/v4 文档与测试。

SFT v3 和 SFT v4 的 prompt / Memory / 状态机必须严格同步。规范实现只放在
`qwen3vl_local/sft_v4/prompts.py`；`sft_v3/prompts.py` 是 thin re-export + v3 兼容别名。
因此：

- 改 `sft_v4/prompts.py` 的 prompt、label、memory 字段、trigger helper、target span 时，
  必须同时跑 v3/v4 相关 mask 和 memory 测试。
- 不允许在 `sft_v3/prompts.py` 重新实现第二份 prompt 或状态机。
- v3 与 v4 的差异只在训练数据流：v3 是离线 on-policy OPSD；v4 是 replay/collector/learner
  off-policy actor-learner。

> 本文件是 SFT v3 的设计冻结版。当前代码已落地到同目录
> `prompts.py` / `build_dataset.py` / `train.py` / `train.sh` / `eval.py` /
> `probe.py` / `SFT_V3_RUN.md` 以及配套 test 脚本。
>
> v3 **不替代** v2，v2 仍保留作为单帧串行选择题基线。

---

## 1. v3 与 v2 的本质区别

| 维度 | v2 | v3 |
|---|---|---|
| 训练单元 | 单帧 anchor 的串行选择题 | **一个 sub-scenario 的时间序列**（外循环按时间步推进） |
| Memory | 无；只有 `PREVIOUS_STATUS_HINT` 字段 | **学生自维护的纯文本 Memory**，每帧外循环之间链接 |
| Teacher | 无 | **Frozen Qwen3-VL-4B-Instruct teacher**（与 student 共享 base，通过 `disable_adapter` 切换），读 v4 privileged prompt，在 student rollout token 上给 full-vocab logits |
| 分析监督 | 无（已废弃 ANALYSIS 路线） | **OPSD 范式**：student 自由生成的 analysis token 是 on-policy rollout，teacher logits 对同一批 token 做 forward-KL 分布监督 |
| 离散监督 | scene / status / subgoal 值 token CE | `ROAD_STRUCTURE/SCENE/STATUS/SUBGOAL` 值 token 也走 OPSD forward-KL，权重显著大于分析 loss |
| Wrong-scene 增强 | jsonl 里 `--wrong-scene-ratio` 注入错场景 | 错场景**来自学生自身 memory 漂移**，天然产生；phase B 反向用 GT scene 注入 |
| 数据持久化 | jsonl 训练样本 | **不写训练样本**，只写 `episode_index.jsonl` |

整体精神：用 v4 的同一套 prompt/Memory/状态机产生 student on-policy rollout，
再让 privileged base teacher 在这些 token 上提供分布级偏好，把"错记忆 → 正记忆"
这个动作蒸馏进 student。

---

## 1.1 代码地图与中文注释约定

当前子包代码已按“下个维护者先读注释再读实现”的口径补齐中文说明：

| 文件 | 主要职责 | 先读位置 |
|---|---|---|
| `prompts.py` | Memory 文本格式、状态机更新、三步 prompt、输出解析、legacy GT leak hook | module docstring、`Memory`、`update_memory_after_step2/3` |
| `build_dataset.py` | 从 `keyframes_all_scenarios.json` 构建 episode index，只写元数据不写 per-frame 样本 | 文件头、`build_episode`、`load_keyframe_runs` |
| `train.py` | LoRA 训练主入口；每帧 step1/2/3 OPSD 内循环；KV 追加；student rollout token 上的 privileged-teacher forward-KL | 文件头、`KVState`、`_append_token_ids_with_logits`、`_opsd_loss_from_states`、`iter_episode_loss_packs` |
| `eval.py` | 自由生成评估；不做 Phase B GT 注入；可选 teacher BLEU | 文件头、`_generate_next_with_kv`、`main` |
| `probe.py` | case-level dump；可选 teacher privileged prompt/text | 文件头、`main` |
| `check_loss_mask.py` / `test_*.py` | 静态 mask、memory 状态机、KV 复用、v4 legacy hook 同步测试 | 各自 `main` docstring |
| `train.sh` | 训练 launcher、GPU 自动选址、run 子目录、防 core dump | 脚本顶部和 mode 分支注释 |

注释原则：

- 函数 docstring 写“为什么这样做”和“与 v3 思路的关系”，避免只复述函数名。
- 关键代码块注释写状态机边界，例如 Phase B 强制覆盖、scene 错误时跳过 step3、
  student 自由生成用于 memory 推进、teacher 只提供 logits 分布监督的区别。
- 不在注释里复制大段 prompt 或源码；prompt 真实文本仍以 `prompts.py` 为准。

---

## 2. 时间轴与训练窗口

### 2.1 Anchor 定义（每个 sub-scenario 固定 5 个时间点）

```
anchor[0] = sub-scenario 起始（init 时刻）
anchor[1] = subgoal 1 时刻
anchor[2] = subgoal 2 时刻
anchor[3] = subgoal 3 时刻
anchor[4] = sub-scenario 结束
```

每个 sub-scenario **固定 3 个 subgoal**，因此 anchor 列表长度恒为 5。
若数据里出现 ≠ 3 subgoal 的 sub-scenario，`build_dataset.py` 直接过滤掉并打 warning。

### 2.2 δ 公式（per episode）

```python
raw_delta = min(anchor[1] - anchor[0], anchor[2] - anchor[1]) // 2
delta = min(raw_delta, 10)   # 允许 δ=0，只封顶 10
```

每条 episode 算一次，写进 episode index；保证 `[anchor[1] - δ, anchor[1] + δ]`
不会越过 anchor[0] 或越过 anchor[2] 的中点。

**允许 δ=0（C1 修正）**：δ = 0 会让 Phase A 退化为单帧 `[f1, f1]`，但这是
公式本身表达的数据事实，不能静默改成 1 后越过 anchor[1]。代码只在
`raw_delta == 0` 时打 warning，提示用户该 episode 的 Phase A 已退化为单帧；长期可考虑
直接过滤掉，目前只做诊断。

### 2.3 训练窗口与 Phase 划分

```
帧索引:          ... anchor0 -- anchor1 -- anchor2 -- anchor3 -- anchor4 ...
意义:                init        sub1       sub2       sub3        end
窗口:           [    不训练    ][← Phase A →][————— Phase B ——————][ 不训练 ]
                              f1-δ        f1+δ                   f3
```

- **Phase A**：`t ∈ [anchor[1] - δ, anchor[1] + δ]`
  - memory 由学生**自更新**（含场景翻转）。
  - 三步全训：分析 + scene + status + subgoal。
- **Phase B**：`t ∈ (anchor[1] + δ, anchor[3]]`
  - **每帧外循环开始时**对 `memory.scene` 做**弱纠偏**（D2 拍板）：
    - 若 `memory.scene == gt_scene` → 全 no-op，**status / subgoal 跨帧保留**；
    - 若 `memory.scene != gt_scene` → 走与 step2 翻转完全相同的"scene change →
      status = canonical init event、subgoal = EVENT_SEQUENCE[gt_scene][1]"重置路径。
  - 这样 Phase B 的语义是"假设场景认知已经被矫正，让你专心练 status/subgoal
    顺着真实事件链一步步推"，sub2/sub3 附近的 e2/e3 才能拿到密集监督；如果每帧
    都把 status 拉回 init，e2/e3 的监督密度被人为压扁，丢失链式推理（D2 推翻
    了"无条件 reset 三件套"的旧口径）。
  - 然后正常跑三步内循环：仍训 scene / status / subgoal，student 输出的 SCENE
    名字 == GT scene 时即天然"keep"；模型输出永远是场景名，不是 "keep" 字样。
  - 这是反向的 wrong-scene 增强：教模型"已经对的就别瞎改"，与 Phase A 的
    "错了要敢于改"对称。

> Phase B 也是按帧推进、step 3 也照样训，所以 sub2 / sub3 附近的 status /
> subgoal 监督密度很高；anchor[3] 之后到 anchor[4] 不训。

### 2.4 外循环步长

**默认每帧前进一次**（`--outer-stride 1`）。每条 episode 长度 ≈ `(anchor[3] - anchor[1] + δ + 1)` 帧。

代码内保留 `--outer-stride K` 参数（K > 1 时均匀降采样），但默认 1，符合
"step by step 都要学到"的要求。

### 2.5 Episode 构造

- 一条 episode = 一个 run/sub-scenario；当前数据约定为**一个 run 就是一条
  3-middle episode**。`build_dataset.py` 对可用 run 执行严格契约检查：必须是
  `initial + middle[3] + final`，anchors 严格递增，事件序列与
  `prompt_pipeline.get_full_sequence` 完全一致；否则直接报错退出，不静默过滤。
- `build_dataset.py` 只产 **episode index**（不产 per-frame 训练样本）。输入兼容
  `keyframes_all_scenarios.json` 当前的顶层对象 schema（从 `runs` 列表取 run），也兼容
  早期顶层直接是 run list 的临时 dump。每条 row：
  ```json
  {
    "run_id": "...",
    "scenario": "...",
    "anchors": [f0, f1, f2, f3, f4],
    "delta": 7,
    "frame_range": [f1 - delta, f3],
    "gt_scene": "...",
    "gt_event_sequence": ["initial", "...", "...", "...", "final"],
    "ego_to_goal_xy_per_frame": "<lazy: 训练时即时算>",
    "split": "train" | "val"
  }
  ```
- 训练入口的 DataLoader 每次吐一条 episode；DataLoader workers 只做轻量元数据
  和路径解析，**不做图像预读**（图像在 train loop 内按外循环帧号现读）。

---

## 3. Memory 结构

### 3.1 文本格式（每帧重写，塞进 prompt）

```
[MEMORY]
BELIEVED_SCENE: <scene_name>
  Description: <SCENARIO_LABELS[scene]>
  EventSequence: e0 -> e1 -> e2 -> e3 -> e4
BELIEVED_STATUS: <event_name>
  Description: <EVENT_DESCRIPTIONS[event]>
BELIEVED_SUBGOAL: <event_name>
  Description: <EVENT_DESCRIPTIONS[event]>
EGO_TO_GOAL_XY: x=+12.3 m, y=-4.5 m
[/MEMORY]
```

- 描述字段直接复用 [prompt_pipeline.py](../prompt_pipeline.py) 的 `SCENARIO_LABELS` /
  `EVENT_DESCRIPTIONS`，保持与 v2 prompt 风格一致。
- `EventSequence` 一并列出，让模型在 step3 之前就看到该 scene 的完整事件链，
  step3 prompt 不必再重复。
- `EGO_TO_GOAL_XY` 精度 `+.1f` 米。**必须保留**：十字路口左/右转场景靠这条信息
  消歧（"目的地在车体左前 → 大概率左转场景"）。
- Memory **不**携带上一帧的学生分析文本，避免 prompt 滚动膨胀。

### 3.2 初始化（在 t = anchor[1] - δ）

- 初始化统一调用 v4 的 `init_memory(...)`，不在 v3 单独维护采样逻辑；
  `seed = hash((run_id, sub_scenario_id, seed_salt))` 保证可复现。
- 默认 `p_init_correct=0.7`：ROAD_STRUCTURE 以 0.7 概率命中 GT 桶；命中时 scene
  仍有 50% 概率从同桶其它 canonical scene 扰动，未命中时 scene 从错误桶内采样。
  这样 memory 始终保持“road bucket 与 scene 同桶”的自洽状态，也和 v4 collector
  / learner 的 prompt 分布同步。
- `believed_status` ← `initial_event(believed_scene)`。
- `believed_subgoal` ← `first_subgoal(believed_scene)`。
- `ego_to_goal_xy` ← 当前帧 ego 系下的 final destination 坐标。
  与 leadmot final_goal 同源：`meta["next_target_points"][-1]` 转 ego（见
  `PROJECT_CONTEXT.md`）。**meta 缺失或解析失败时 `train.py` 直接抛 RuntimeError**，
  禁止静默 fallback 到 measurements / (0, 0)——污染过的 ego_to_goal 会让"目的地
  在车体左前 → 大概率左转场景"这条核心消歧信号失效。

### 3.3 更新规则（每帧外循环末尾）

按 Phase 分支：

**Phase A**：
1. **step 2 结束后立即更新 scene**：
   - 学生 step2 输出 SCENE ≠ memory.scene → memory.scene 改写为新 scene，
    同时强制 `memory.status ← canonical init event`、
     `memory.subgoal ← EVENT_SEQUENCE[new_scene][1]`。
   - 否则 keep。
2. **step 3 触发条件**（关键，且与 step2 是否改动**无关**）：
   ```
   if memory.scene_after_step2 == gt_scene:
       run step 3
   else:
       skip step 3
   ```
   所以四种组合：
   | 初始 memory.scene | step2 输出 | scene 翻转? | scene 是否正确 | step 3 |
   |---|---|---|---|---|
   | 正确 | 同 GT | 否 | ✓ | **跑** |
   | 正确 | 改成别的 | 是 | ✗ | 跳过 |
   | 错误 | 同 memory（仍错） | 否 | ✗ | 跳过 |
   | 错误 | 改成 GT | 是 | ✓ | **跑** |
3. **step 3 跑了的话**：
   - `memory.status ← student_step3_pred_status`
   - `memory.subgoal ← student_step3_pred_subgoal`
4. 帧末为下一帧预取 `ego_to_goal_xy`，写入 memory。
   （C3 修正：当前帧 prompt 使用进入本帧前已经写入 memory 的坐标；本帧结束后
   再读取下一帧 meta，字面上与“进入下一帧前重算”一致。）

`should_trigger_step2/3(before, after, gt, reset_this_frame)` 这组 stair-step
触发门被抽成 `prompts.py` helper；训练入口和 `test_memory_update.py` 共用同一组
真值函数，避免状态机有第二处隐式实现。

**Phase B**（D2 拍板：弱纠偏 + 跨帧推进）：
1. **帧外循环开始前**调用 `force_memory_to_gt_chain(memory, gt_road_structure=gt_rs, gt_scene=gt_scene)`：
   ```
   if memory.road_structure != gt_rs:
       memory.road_structure = gt_rs
       memory.scene = first scene in gt_rs bucket
       memory.status = canonical init event
       memory.subgoal = first subgoal of memory.scene
   if memory.scene != gt_scene:
       memory.scene = gt_scene
       memory.status = canonical init event
       memory.subgoal = EVENT_SEQUENCE[gt_scene][1]
   ```
   这条规则与 v4 prompt 的三层状态机同款：先稳定 ROAD_STRUCTURE，再稳定 SCENE，
   避免 v3 私自维护第二套状态机。
2. 正常跑 step1/step2/step3（step3 触发条件依然走"step2 之后 memory.scene
   == gt_scene"，由于 step2 开始时 memory 已是 GT，多数情况 step3 会触发）。
3. 帧末更新 status / subgoal 同 Phase A 第 3 步——所以 sub2 / sub3 附近的
   status / subgoal 会**跨帧累积推进**到 e2 / e3，监督密度起得来。
4. Phase B 内 memory.scene 跨帧也保持等于 GT（step2 之后只可能 keep 或 flip，
   flip 会被下一帧的弱纠偏拉回；但只要 student 不在 Phase B 内还在 flip，
   弱纠偏永远 no-op）。

> Phase A 与 Phase B 的 prompt **完全一致**，唯一差别只在"帧开头是否被强制
> 改 memory"。这意味着 student 看不出自己在哪个 phase，自然学到 phase-agnostic
> 的修正/保持能力。

---

## 4. 内循环（每帧 3 步，v4 prompt + OPSD 范式）

### 4.1 共享 KV cache 结构

每帧外循环开始时，student 先按 v4 step1 prompt prefill 图像前缀：

```
<SYSTEM_PROMPT_STEP1> ← 来自 qwen3vl_local/sft_v4/prompts.py
<4 stitched RGB>      ← 历史四帧，从旧到新
                        （图像 token 占绝大多数 prefix 长度）
```

> 注意：v3 不复制 prompt 文本；`sft_v3/prompts.py` 只 re-export v4。
> student 端与 v4 collector 一样：step1 带图 prefill，step2/3 作为后续 user turn
> 追加到 student KV 中，memory 从 step2/3 才进入。teacher 端每个 step 用 v4 对应的
> `SYSTEM_PROMPT_STEP1/2/3` 与 privileged user prompt 独立 prefill，然后在同一段
> student step 输出 token 上计算 full-vocab forward-KL。

三步在前缀基础上依次 append user turn → assistant turn，KV cache 链式延伸：

```
prefix → [step1 user] → [step1 assistant analysis]
       → [step2 user (with memory)] → [step2 assistant analysis + SCENE]
       → [step3 user (with selected scene)] → [step3 assistant analysis + STATUS + SUBGOAL]
```

每个 step 的 user turn 都从上一步的 KV cache 继续，不重 prefill。

> student 的 on-policy rollout 只物理 prefill 一次图像并串行追加三步；teacher
> 由于必须 `disable_adapter()` 且每步 system prompt 不同，不能复用 student KV。
> teacher 只负责给同一段 student step token 的 next-token distribution 打分。

### 4.2 Teacher 模型

- **Teacher = student 共享同一份 base 权重**，通过 PEFT 的
  `with student.disable_adapter():` 关掉 LoRA，再 `model.eval()` +
  `torch.no_grad()` 跑 teacher prefill/logits。teacher 可在 probe 中生成诊断文本，
  但训练梯度不来自 teacher 文本 CE。
- teacher 侧上下文必须三件套一起出现：`disable_adapter()` 防止 LoRA 污染，
  `eval()` 关闭 dropout，`no_grad()` 避免保存 teacher 计算图。`train.py` 的
  `_teacher_eval_context` 是唯一实现入口，新增 step 不要手写散落版 teacher context。
- 不实例化第二份模型，零额外显存。
- 每帧 teacher 一共做 1~3 次 privileged prefill/logit scoring（step3 仅在触发时）。
- 切换 adapter on/off 会让 cuDNN benchmark 有少量抖动，可以忍受；并且这种切换
  与 v2 `train.py` 现有"禁用 adapter 跑 teacher、再启用 adapter 跑 student
  forward"的代码路径同构，可直接复用。

### 4.3 Step 1：ROAD_STRUCTURE 判断（OPSD，无特权答案但有 v4 step1 prompt）

**Teacher prompt**（在共享前缀上 append）：
```
<step1_user>:
  [STEP1]
  4 images are ordered oldest to newest; the last image is now.
  Describe visible surroundings and recent motion. Do not use memory.
```

- Student 先自由生成 step1 文本，解析 `ROAD_STRUCTURE` 并更新 memory。
- Teacher 使用 v4 step1 teacher prompt 与 `disable_adapter()` base Qwen，对这段
  student step1 文本的 analysis token 与 `ROAD_STRUCTURE` 值 token 输出 full-vocab logits。
- **L_A1 / L_RS1**：student logits 与 teacher logits 在同一段 student token 上做 forward-KL。

### 4.4 Step 2：场景判断（OPSD，teacher 吃 GT scene）

**Teacher prompt**（v4 step2 system prompt + privileged user prompt）：
```
<step2_user>:
  [MEMORY] ...                 ← 当前学生 memory（带描述）
  [SCENE_CHOICES] ...          ← 全集（同 v2 sft_v2/prompts.py 的 scenario_choices_block）
  ANSWER_ROAD_STRUCTURE / ANSWER_SCENE 等 privileged 字段只在 teacher prompt 中出现
```

- Student 先自由生成 step2 文本并更新 `memory.scene`。
- Teacher 对这段 student step2 文本给 logits，而不是生成 hard target 文本。
- **GT leak hook**：v4 当前将 `check_gt_leak_scene` 保留为 legacy no-op；
  v3 不另写第二套正则。teacher/student 视角隔离主要依赖 v4 prompt contract 与
  teacher target 清洗规则；`train/gt_leak_skip_rate/step2` 只是兼容旧日志项，正常应为 0。

**Student prompt**：使用 v4 `build_step2_student_prompt(memory)`，只读 student
当前 memory 与当前 road bucket 下的 `SCENE_CHOICES`。
- **L_A2 / L_S2**：analysis token 与 `SCENE` 值 token 的 OPSD forward-KL。

### 4.5 Step 3：状态 / 子目标判断（OPSD，teacher 吃 GT status/subgoal）

**触发条件（重申）**：`memory.scene_after_step2 == gt_scene`。否则整步跳过，
本帧也不更新 status / subgoal。

**Teacher prompt**（v4 step3 system prompt + privileged user prompt）：
```
<step3_user>:
  [MEMORY] ...                                  ← step2 后更新过的学生 memory
  [EVENT_OPTIONS] ...                           ← 该 scene 的事件序列及描述
  ANSWER_STATUS / ANSWER_SUBGOAL 等 privileged 字段只在 teacher prompt 中出现
```

- Student 先自由生成 step3 文本并更新 `memory.status/subgoal`。
- **L_A3 / L_S3_status / L_S3_subgoal**：在同一段 student step3 token 上做
  privileged-teacher logits forward-KL。

---

## 5. 损失与权重

### 5.1 单帧 loss

| 名称 | 含义 | 默认权重 | 备注 |
|---|---|---|---|
| L_A1 | step1 分析 token forward-KL / 监督 token 数 | 0.2 | 已 per-token normalize |
| L_A2 | step2 分析 token forward-KL / 监督 token 数 | 0.2 | 同上 |
| L_A3 | step3 分析 token forward-KL / 监督 token 数 | 0.2 | 同上 |
| L_S2 | step2 SCENE 值 token forward-KL | **1.0** | 主信号 |
| L_S3_status | step3 STATUS 值 token forward-KL | **1.0** | 主信号 |
| L_S3_subgoal | step3 SUBGOAL 值 token forward-KL | **1.0** | 主信号 |

```
L_frame = w_A1 * L_A1
        + w_A2 * L_A2 + w_S2 * L_S2
        + 1{step3_triggered} * (w_A3 * L_A3 + w_S3_status * L_S3_status + w_S3_subgoal * L_S3_subgoal)
```

- **Per-token normalize**：每个分析 loss = `sum_CE / num_supervised_tokens`，
  防止 step1 一长串分析淹没 1-token 的 SCENE。
- 当前 v4 legacy leak hook 不跳 `L_A*`；若未来 v4 重新启用过滤，v3 会通过
  re-export 同步继承，且值 token 仍照常保留。
- 若 step3 未触发，整段 step3 系数为 0。

### 5.2 梯度累积粒度

- 每帧外循环结束做 **一次** backward + optimizer step（grad accumulate = 1
  episode-frame）。
- Episode 内默认**每帧 backward + optimizer step**（`grad_accum=1`），避免显存随
  episode 长度爆炸，并与 OPSD memory 在线更新口径一致。
- 多 rank 下不再强制 `grad_accum=1`（work-stealing + local-SGD 不依赖 rank 间
  per-step 锁步），但默认仍 1；尾部不足 `grad_accum` 的梯度会在 sync/epoch
  边界按实际帧数补尺度后再 step，避免 tail step 被系统性压小。梯度同步从
  per-step 切到每 rank 目标 K 个 episode 后做参数平均，详见 9.1。
- `per_device_batch_size` 在 v3 work-stealing 口径下固定为 1：每个 worker pull
  一条 episode，episode 内逐帧推进 memory。

### 5.3 TensorBoard 记录

最低必须有的标量：

```
train/loss_total
train/loss/{a1, a2, a3, s2, s3_status, s3_subgoal}
train/loss_weight/{a1, a2, a3, s2, s3_status, s3_subgoal}    ← 静态权重，可视化用
train/step3_trigger_rate                                       ← 每帧粒度
train/scene_flip_rate                                          ← step2 改 scene 的比例
train/gt_leak_skip_rate/{step2, step3}                       ← legacy hook 兼容项，当前 v4 no-op 下应为 0
train/phase_a_frame_frac                                       ← 每 step batch 内 Phase A 帧占比
train/lr
train/grad_norm/{language, vision}                             ← 与 v2 同
train/param_norm/lora_{language, vision}                       ← 与 v2 同
train/vision_guard_bad_steps                                   ← 与 v2 同
val/scene_acc_per_step
val/scene_recovery_steps
val/status_acc_given_correct_scene
val/subgoal_acc_given_correct_scene
val/all_acc_per_step
val/analysis_bleu_vs_teacher (eval.py --with-teacher-ref 时输出)
```

---

## 6. 模型 / Teacher / LoRA 设置

- **Base**：`AutoMoT/checkpoints/Qwen3-VL-4B-Instruct`，`local_files_only=True`，
  与 v2 同源。
- **Student**：base + PEFT LoRA，与 v2 完全相同的视觉 LoRA 接口：
  - `--lora-vision-scope` ∈ `{off, merger, last4, all}`，默认 `off`；
  - 视觉 LR 缩放 `--vision-lr-scale=0.1`、上限 `--max-vision-lr-scale=0.25`；
  - 语言 / 视觉分组梯度裁剪 `--language-clip-norm=1.0` / `--vision-clip-norm=0.3`；
  - `STRICT_VISION_SCOPE=1` 命名漂移硬拒绝；
  - `VISION_GUARD_ENABLED=1` 运行时视觉熔断；
  - 熔断时写 `fuse_stop_step_<N>/`、`fuse_reason.txt`，跳过 `final/`。
- **Teacher**：复用 student base，**不创建第二份模型**。每次 teacher generate
  时：
  ```python
  with model.disable_adapter():
      with torch.no_grad():
          model.eval()
          out = model.generate(...)
          model.train()
  ```
  恢复时间常数 < 1ms 量级。
- Adapter 保存与 v2 同：base 只读，只存 adapter delta + `sft_v3_adapter_config.json`
  （记录 LoRA scope、视觉保险参数、训练窗口参数 δ / phase 配置）。

### 6.1 Teacher generate 超参

| Step | max_new_tokens | do_sample | repetition_penalty | 备注 |
|---|---|---|---|---|
| 1 | 80 | False | 1.05 | prompt 约束 ≤60 tokens，generate 留 20 token 余量 |
| 2 | 60 | False | 1.05 | prompt 约束 ≤40 tokens + "SCENE: …"，generate 留余量 |
| 3 | 60 | False | 1.05 | prompt 约束 ≤40 tokens + "STATUS:"/"SUBGOAL:"，generate 留余量 |

prompt 内的 token 上限故意比 `max_new_tokens` 更保守，预留格式行、分词误差和
少量冗余；如发现 99 分位 token 数贴近生成上限，调高 max_new_tokens 10~20，
并同步评估是否需要放宽 prompt 内的保守上限，不要静默放宽。

实现层面，`train.py` 的 `_kv_generate_text` 已经把 `repetition_penalty=1.05`
落实成 HF 风格的 logits 后处理：每步生成前对 `state.decoded_input_ids ∪ 已生成
token` 的并集施加 penalty（正分除以 1.05、负分乘以 1.05），等价 transformers
`RepetitionPenaltyLogitsProcessor` 在 greedy decode 下的行为。max 80 token 的开销
可忽略，主要避免 step1 出现"I see I see ..."之类复读污染 L_A1 目标分布（B1 拍板）。

---

## 7. 数据 / IO 层

- **不写 jsonl 训练样本**。`build_dataset.py` 只产 `episode_index.jsonl`，
  字段见 §2.5。
- 训练时图像懒加载：DataLoader workers 只解析路径，train loop 内按外循环帧号
  现读 4 张 stitched RGB（与 v2 / leadmot 同一套 image_io）。
- `ego_to_goal_xy_per_frame` 在 train/eval/probe loop 内按帧末预取方式即时算；
  来源必须是 meta 内 `next_target_points[-1]`。
- val episode 数量在 `build_dataset.py` 阶段固定切分，不在每 step 重采样。

---

## 8. Eval 框架

### 8.1 E1 离线 eval（对应 v2 `eval.py`）

- **不挂 teacher**，只跑 student；**不做 Phase B 强制覆盖**；memory 全程由
  student 自更新。
- 默认评测区间 = 训练区间 `[anchor[1] - δ, anchor[3]]`；
  `--full-range` 时扩到 `[anchor[0], anchor[4]]`（做 OOD 诊断，结果只看不进
  主指标）。
- 主指标：
  - `scene_acc_per_step`：每帧 step2 输出 SCENE == GT scene 的比例。
  - `scene_recovery_steps`：从随机错初始化到首次预测对 SCENE 的帧数（蒸馏要解决
    的核心问题）。
  - `scene_stick_rate`：scene 已对的下一帧是否保持对。
  - `scene_flip_rate`：step2 改 scene 的比例（高 = 不稳定）。
  - `step3_trigger_rate`。
  - `status_acc_given_correct_scene` / `subgoal_acc_given_correct_scene`（口径与
    v2 `*_valid_scene` 一致）。
  - `all_acc_per_step`（串行：scene + status + subgoal 全对）。
  - `phase_breakdown`：Phase A 区间 vs Phase B 区间分别统计同上指标。
  - `invalid_scene_rate`、`invalid_status_for_pred_scene_rate`：与 v2 同。
- 可选指标：`analysis_bleu_vs_teacher`。`eval.py --with-teacher-ref` 会额外加载
  一份 base Qwen teacher，按 step1/2/3 的分析文本计算轻量 BLEU；默认关闭，
  避免普通 eval 双倍显存/耗时。`analysis_token_ce_vs_teacher` 仍为后续扩展项。

### 8.2 E2 case probe（对应 v2 `probe.py`）

随机抽 N 条 episode，逐帧 dump 到 `case_<idx>/frame_<t>/`：

```
case_0/frame_42/
  rgb_oldest.png … rgb_newest.png       ← 4 张拼接 RGB
  step1_prompt.txt   step1_teacher.txt   step1_student.txt
  step2_prompt.txt   step2_teacher.txt   step2_student.txt
  step3_prompt.txt   step3_teacher.txt   step3_student.txt   (若触发)
  memory_before.json memory_after.json
  flags.json          ← step3_triggered / legacy_gt_leak_hook / scene_flip / phase
case_0/timeline.png   ← scene-flip 与 step3 触发的时间线
case_0/episode_meta.json
```

`probe.py --with-teacher` 会额外加载一份 base Qwen teacher，逐帧写出
`step*_teacher.txt`、`step2_teacher_user.txt`、`step3_teacher_user.txt`，并在
`flags.json` 记录 step1/2/3 的 `analysis_bleu_vs_teacher`；默认不加载 teacher，
避免普通 case dump 双倍占显存。

### 8.3 E3 训练时 in-loop val

单卡时每 `EVAL_STEPS` 抽少量 val episode 跑 OPSD quick loss。多 rank 下
不做 in-loop eval（参数还未平均时各 rank 模型不同，eval 没意义）；若
`WORLD_SIZE>1` 且 `--eval-steps > 0`，`train.py` 直接报错，要求训练后单独运行
`eval.py`。

### 8.4 E4 单元 / 烟雾测试

- `check_loss_mask.py`：实际覆盖 6 路 loss（L_A1 / L_A2 / L_S2 / L_A3 /
  L_S3_status / L_S3_subgoal）的 token mask 正确性：分析段 token 集 / 值 token 集
  互不重叠，per-token normalize 分母对得上 train.py 里 `_append_token_ids` 的
  位置切分逻辑。
- `test_memory_update.py`：纯 Python 模拟外循环，覆盖：
  - `init_memory` 排除 GT scene（D3 拍板）；
  - Phase A `update_memory_after_step2` 的 4 种翻转组合；
  - Phase B 弱纠偏（D2 拍板）：scene == GT 时 noop、status/subgoal 跨帧保留；
    scene != GT 时走 scene-change reset；
  - scene 翻转 → status = canonical init event、subgoal 重置；
  - step3 触发条件正确（直接调用 `should_trigger_step3` helper，与翻转无关、
    只看 `memory.scene_after_step2 == gt_scene`）。
- `test_kv_reuse.py`：构造一条 mini episode，对比"step1/2/3 复用 KV vs 全量
  重 prefill"的 student logits 数值一致（误差 < 1e-5）。
- `test_gt_leak_filter.py`：构造含答案字面的样例，验证 v3 与 v4 一样把
  `check_gt_leak_*` 保持为 legacy no-op，防止 v3 悄悄恢复第二套正则。

---

## 9. 关键工程约束

### 9.1 多卡分布式：work-stealing + local-SGD（已替换原 DDP+Join 方案）

- 历史教训：v3 的"每帧 ~330 次 DDP forward + 1 次 backward"训练循环跟标准
  DDP+Join 不兼容——episode 帧数差异让各 rank collective 序列严重不一致
  （同一 SeqNum 上有 rank 发 33M-elt grad allreduce、有 rank 发 1-elt Join 探测），
  NCCL 直接 watchdog 超时。因此 v3 不再用 DDP wrap。
- **work-stealing 调度**：所有 rank 加载同一份 `train_ds.rows`，每个 epoch 用同
  seed 重排得到 `epoch_order`；rank0 在 init_process_group 自带的 TCPStore 上重置
  `sft_v3_epoch_<n>_counter=0`，各 rank 通过 `store.wait([counter_key])`
  确认本轮 counter 已写入后，再用 `store.add(key, 1)`
  原子递增抢下一个 `idx`。**谁空闲谁抢，全部 episode 都被训，没有截断**。
- **初始化同步**：local-SGD 不包 DDP，因此模型创建后会先把 rank0 的 trainable
  LoRA 参数广播到所有 rank，保证所有 worker 从同一个 adapter 起点出发。
- **独立 forward/backward/optimizer.step**：每个 rank 各自维护 PEFT 模型副本
  （不包 DDP），各自做 forward + backward + clip + step + scheduler.step。
  无 per-step allreduce，per-rank 速度差异不再造成死锁。
- **周期 LoRA 参数平均（local-SGD）**：参数 `--sync-every-episodes K`
  （**默认 4**）。K 表示每个 rank 目标处理的 episode 数；每个 epoch 被切成若干个
  `K * world_size` 全局 episode 的 sync round（`K=0` 时整 epoch 一轮）。每轮
  counter 只允许抢 `[round_start, round_end)`，不会越过同步边界提前训练下一轮。
  轮末先 flush 未满 `grad_accum` 的梯度，再对所有 trainable LoRA 参数做按本轮
  optimizer step 数加权的参数平均；空闲 rank 权重为 0，只接收平均后的 adapter，
  不再用旧参数稀释真正训练过的 rank。同步 `fuse_stopped` / `max_steps` 停止标志，
  并用 allreduce 汇总出的 `all_rank_steps` 把各 rank scheduler 对齐到同一 LR
  曲线位置。Adam 的 m/v 不平均，各 rank 自留——标准 local-SGD 做法。
- **NCCL watchdog 规避**：work-stealing 只让任务分配异步，参数平均仍是周期同步点。
  快 rank 可能比慢 rank 早很多跑完 round；因此进入任何 NCCL allreduce / broadcast
  前，先用 TCPStore `add/wait/set` 做 CPU 侧 rendezvous，所有 rank 到齐后再发 NCCL
  collective，避免快 rank 在 NCCL work 上空等超过 watchdog timeout。此外
  `setup_distributed` 把 `init_process_group(timeout=...)` 显式设到 **2 小时**
  （同时影响默认 TCPStore.wait/get），给 sft_v3 这种 teacher/student 各 ~80 step
  自由生成的高成本内循环留出充足同步窗口；NCCL 默认 10 分钟对一轮 5~20 min
  完全不够。
- **同步诊断 stderr 日志**：进入 `do_sync_round` 每个 rank 都会打一条
  `[sync-enter] key=... rank=R local_steps=... local_eps=...`；每个 rank 抢到
  episode 也会限频打 `[claim]`（每轮第 1 条 + 每 8 条一条）。这两路都走 stderr，
  方便从 `[train]` 高频 stdout 行里筛出来诊断 work-stealing 是否真的负载均衡，
  以及死锁时哪个 rank 落后。
- **同步诊断日志**：rank0 普通训练日志中的 `step` 是 rank0 本地 optimizer step；
  sync 日志中的 `all_rank_steps` 才是所有 rank 的 optimizer step 汇总，并用于
  checkpoint step 与 scheduler 对齐。`round_eps` / `total_eps` 只用于确认
  work-stealing 完整消费 episode 与观察负载，不参与参数平均；参数平均仍只按
  本轮 optimizer step 数加权。TensorBoard 写入
  `train/sync/{round_weight,episodes_this_round,episodes_total,all_rank_steps}`。
- **checkpoint 只保存平均后参数**：多 rank 下 `checkpoint-*` 与 `final/` 都只在
  sync round 结束、LoRA 参数平均完成后由 rank0 保存；checkpoint 名中的 step 使用
  all-rank optimizer step 汇总值。
- **`max_steps/check` 是 debug 口径**：`max_steps>0` 会在 episode 内截断，普通训练
  默认直接拒绝；只有 `--check` 或显式 `--allow-max-steps-truncation` 才允许。多
  rank debug 时会把 sync round 缩到 1 个 episode，并允许本 rank 达到本地步数后
  提前等待 barrier；全局停止仍在 sync 后广播。完整训练默认 `MAX_STEPS=0`，不会截断
  episode。
- **sync round 数量守恒**：每 epoch 每个 rank 做相同数量的 sync round，
  = `ceil(train_total/(K * world_size))`（`K=0` 时为 1 个 epoch-end round），否则 NCCL
  collective 数对不上还是会死锁。轮级 counter 天然保证所有 rank 在同一轮结束
  后一起参数平均，不再需要旧版 catch-up 循环。
- **K 的取舍**：K=1 时每个 rank 通常至多跑 1 条 episode 后同步，最接近同步 SGD
  （慢但严格）；K=16 默认在收敛与吞吐间取平衡；K=0 仅 epoch 末同步（最快但参数
  漂移最大，不推荐除非 epoch 很短）。
- **rank0** 负责 TensorBoard、checkpoint/final adapter 保存和主日志；非 rank0
  只训练。日志里的 `step` 是 rank0 自己的 optimizer.step 计数；sync 日志里的
  `all_rank_steps` 是 allreduce 汇总后的全 rank 实际 optimizer step 数。
- **多 rank 下禁用 in-loop eval**：参数还未平均时各 rank 模型不同，eval 没有
  意义；训练后用 `eval.py` 跑完整指标。

### 9.2 防覆盖目录约定

与 v2 一致：用户给 `OUTPUT_DIR` 后，`train.sh` 在下层套 `run_<RUN_TAG>/`
子目录，`RUN_TAG` 默认时间戳；base 层维护 `latest` symlink；
`NO_RUN_SUBDIR=1` 回退；`HF_HOME` 钉在 base 层。

### 9.3 GPU 选址

与项目通用规则一致：

- 单卡：`nvidia-smi` 自动挑 1 张空闲 GPU，覆盖已有 mask。
- `torchrun --nproc_per_node=N`：自动挑 N 张最空闲。
- `GPU_IDS=0` / `GPU_IDS=0,1,2,3` 是唯一允许的显式 pin 写法。
- 禁止在 RUN.md 里手写 `export CUDA_VISIBLE_DEVICES=...`。

### 9.4 路径约定

- LEAD 数据根：`AutoMoT/lead_data`；keyframes：`AutoMoT/lead_data/keyframes_all_scenarios.json`。
- 保存路径：`AutoMoT/checkpoints/sft_v3_lora/...`。
- 训练命令以 `AutoMoT/` 为 cwd，不写 `AutoMoT/` 前缀。

---

## 10. 已确认的设计决定（讨论闭环）

下面是在 v3 设计讨论里已经定案、不再回滚的决定，留作未来重读时的参照：

1. **训练单元**：一个 sub-scenario = 一条 episode；外循环按 1 帧前进。
2. **anchor 数**：固定 5 个（含 3 个 subgoal），非 3-subgoal 数据过滤。
3. **δ 公式**：`min(Δf_{0→1}, Δf_{1→2}) // 2`，允许 δ=0（C1 修正），只封顶 10；
   原始 δ < 4 时 `build_dataset.py` 打 warning，提示 anchor 间距过窄。
4. **训练区间**：`[f1-δ, f3]`，其中 `[f1-δ, f1+δ]` 为 Phase A，
   `(f1+δ, f3]` 为 Phase B。anchor[3] 之后到 anchor[4] 不训。
5. **Phase B 行为（D2 拍板，覆盖旧口径）**：每帧开头做**弱纠偏**——只在
   `memory.scene != gt_scene` 时把 scene 拉回 GT 并走 scene-change reset；
   `memory.scene == gt_scene` 时全 no-op，让 step3 推进过的 status/subgoal
   跨帧累积。这样 e2/e3 的状态/子目标监督密度才能起来。step2 仍训 SCENE
   （输出场景名，与 memory.scene 同名即"keep"），让模型同时学会"对的别改"。
   旧口径"每帧无条件 reset 三件套到 init"已废弃。
6. **Memory 内容**：BELIEVED_SCENE / STATUS / SUBGOAL（含描述）+
   `EGO_TO_GOAL_XY`（含十字路口左右转消歧）；**不**带上一帧分析文本；
   不带 KV cache 跨帧。`ego_to_goal_xy` 必须来自 `meta["next_target_points"][-1]`
   转 ego；meta 缺失 / 解析失败一律 `raise RuntimeError`，不允许静默 fallback。
   实现按帧末预取下一帧坐标（C3 修正），不再用“帧首重算”描述。
7. **Memory 初始化（与 v4 同步）**：`init_memory` 来自
   `sft_v4/prompts.py`，默认 `p_init_correct=0.7`。先采样 ROAD_STRUCTURE：
   命中 GT 桶时 scene 仍有 50% 同桶扰动；未命中时 scene 从错误桶采样，保证
   memory 内部始终自洽。v3 不再维护"100% 排除 GT scene"的独立旧口径。
8. **Stair-step 触发条件（与 v4 同步）**：`should_trigger_step2/3` 要求上层
   memory 在本帧 step 前后都已经稳定等于 GT，且不是脚本层刚 reset。road 刚被
   纠正的帧不跑 step2，scene 刚被纠正的帧不跑 step3；train.py 与
   `test_memory_update.py` 共享 v4 helper，禁止有第二份隐式实现。
9. **OPSD 蒸馏**：teacher = frozen base + disable_adapter，读 v4 privileged
   prompt，在 student rollout 的同一批 token 上给 full-vocab logits；student
   用 forward-KL 对齐，不再把 teacher 文本当 hard CE target。实现上使用
   student 自由生成时真实进入 KV 的 token ids 打分；文本只用于解析标签和值 span，
   且不允许先 `.strip()`，避免 loss token 序列和真正推进 memory / 后续 KV 的
   轨迹出现边界错位。若 step1 立刻 EOS/空输出，该帧跳过，不用 GT teacher target
   兜底。
10. **GT leak hook 同步**：当前 v4 的 `check_gt_leak_*` 是 legacy no-op，v3 必须
    继承该行为；私有字段泄露由 v4 prompt contract / teacher target 清洗控制，不在
    v3 另写正则。
11. **Teacher 长度**：step1 ≤3 句（max_new=80），step2/3 ≤2 句（max_new=60）。
    `repetition_penalty=1.05`（B1 拍板）已在 `_kv_generate_text` 内按 HF 风格
    施加 logits 后处理，与 `do_sample=False` 配合避免短文本复读。
12. **Loss 权重**：分析 0.2 × 3、离散 1.0 × 3，分析 per-token normalize；
    分项 TB 记录。
13. **LoRA 接口**：与 v2 完全同构（`--lora-vision-scope` + 全套保险），
    默认 `off`。
14. **多卡训练**：已改为 work-stealing + local-SGD；不包 DDP、不静态分片、不截断
    episode 尾部；启动后广播 rank0 LoRA 初始权重，按本轮 optimizer step 数加权
    平均 LoRA 参数，sync 后保存 averaged checkpoint，`grad_accum` 不再被多卡强制改为 1。
15. **数据持久化**：只写 episode index，不写训练样本。
16. **KV cache 共享语义（D1 拍板）**：所谓"prefix 只 prefill 一次"指的是同一个
    模型内部 step1 → step2 → step3 链式复用 KV；teacher / student 各自 prefill
    一次，不互相共享 KV。理由：LoRA on/off 的 K/V 数值不同，强行共用会让
    teacher 被 LoRA 污染，OPSD 退化为自蒸馏。当前实现已经覆盖图像 prefill 的算力
    优化目标。

---

## 11. 与现有子包 / 文档的关系

- **不替换 v2**：v2 子包保留作为单帧串行选择题基线；v3 是连续推理的进化版。
- **与 leadmot 关系**：v3 输出仍是离散场景 / 状态 / 子目标，不直接影响 leadmot
  的 route / waypoint head；但训练后的 student 可作为 leadmot prefix 的 Qwen
  backbone（merge_and_unload 后冻结），需要时再补 leadmot 侧的接入文档。
- **与 eval_carla 关系**：v3 student 训出来如果想做闭环，仍走 eval_carla 子包；
  闭环时 memory 用类似 §3 的方式实时维护（初始 memory 用启动时刻的 keyframe
  反查，或仍用随机初始 + 让模型自纠错）。这部分细节落到 eval_carla 侧未来的
  扩展文档，不在本 PLAN 范围内。
