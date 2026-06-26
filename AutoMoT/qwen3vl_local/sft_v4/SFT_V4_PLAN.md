# SFT v4 方案说明

> 本文件是 SFT v4 的设计冻结版。当前代码已落地到同目录
> `prompts.py` / `build_dataset.py` / `replay.py` / `collect.py` /
> `learn.py` / `launch_offpolicy.sh` / `eval.py` / `probe.py` /
> `SFT_V4_RUN.md` 以及配套 test 脚本。
>
> v4 **不替代** v2，v2 仍保留作为单帧串行选择题基线。

---

## 0. v4 off-policy 总体方案

v4 把 v3 的"每帧 teacher generate + student generate + loss 串行"on-policy 流程拆成
**actor-learner 异步流水线**：collector 进程负责 rollout（teacher / student 自由生成 +
memory 自更新），把整条 episode 固化成 trajectory 文件写入磁盘 replay；learner 进程
从 replay 抽样，只做 teacher-forced loss + backward。**collector / learner 之间只通过
磁盘 replay + LoRA snapshot 交换状态，没有共享 Python 对象、没有共享 process group**。

### 0.1 总体架构

```
┌────────────────────────────────────────────────────────────────┐
│ Collector 组（独立 Python 进程，无 DDP）                       │
│   GPU 2 [coll0]                                                 │
│   GPU 3 [coll1]                                                 │
│                                                                │
│   每个 collector loop:                                         │
│     抢 episode (TCPStore counter, 或本地随机)                  │
│     周期 reload latest_lora                                    │
│     no_grad rollout: teacher generate + student generate +     │
│       memory 自更新（含 Phase A 70% init / Phase B 噪声 / skip 纠偏） │
│     atomic write ready/<traj>.jsonl                            │
└────────────────────────────────────────────────────────────────┘
                            │ 写 (rename)
                            ▼
                ┌────────────────────────────┐
                │  $OUTPUT_DIR/replay/       │
                │    pending/   写入中        │
                │    ready/     FIFO 256-512  │
                │    failed/    validation 错 │
                └────────────────────────────┘
                            │ 读 (random.choice)
                            ▼
┌────────────────────────────────────────────────────────────────┐
│ Learner 组（DDP world_size=2）                                 │
│   GPU 0 [trainer rank0]                                        │
│   GPU 1 [trainer rank1]                                        │
│                                                                │
│   每个 trainer loop:                                           │
│     pick traj                                                  │
│     for frame in traj:                                         │
│       teacher-forced CE (无 generate, 用存好的 teacher 文本)   │
│       per-frame micro-backward                                 │
│     manual LoRA grad allreduce once / traj                     │
│     optimizer.step                                             │
│     rank0 周期写 latest_lora/v_{step}/ + current_version.txt   │
└────────────────────────────────────────────────────────────────┘
                            │ 写
                            ▼
                ┌────────────────────────────┐
                │  $OUTPUT_DIR/latest_lora/  │
                │    v_0/  v_1000/  v_2000/  │
                │    current_version.txt     │
                └────────────────────────────┘
                            ▲ 读 (collector 周期 reload)
```

### 0.2 4 × H20 96GB 部署方案（默认保守）

**显存预算（单进程实测/估算）**

| 部件 | 显存占用 | 来源 |
|---|---|---|
| Qwen3-VL-4B-Instruct bf16 base | ~8 GB | model weights |
| PEFT LoRA r=16 adapter | ~0.2 GB | adapter weights |
| KV cache (4 stitched RGB + step1/2/3 chain) | ~3 GB | 36 layers × 8 head × hidden × seq_len |
| Activations (forward, eval mode) | ~3 GB | bf16 hidden states |
| **小计 (collector, no_grad)** | **~14 GB** | 实际 v3 测过 ~22 GB（含临时 buffer 与碎片） |
| Activations (training, with grad) | ~6 GB | 额外保存反传所需中间激活 |
| Gradients (LoRA only, 33M × 2 byte) | ~0.07 GB | LoRA 是唯一 trainable |
| Adam state (m + v, fp32, LoRA only) | ~0.5 GB | 33M × 2 tensor × 4 byte |
| **小计 (learner, with grad)** | **~22 GB** | 实际 v3 测过 ~32 GB（含 backward buffer + 碎片） |

实测 v3 单进程稳态占用 ~32 GB（learner）/ ~22 GB（collector 估算），都不到 H20 96GB
的 1/3，多进程的瓶颈不在显存，**真正瓶颈是 SM 算力**——下面会展开。

**默认部署表**

| GPU | 角色 | 进程数 | 单进程显存 | 卡内合计 | 卡剩余 | 备注 |
|---|---|---|---|---|---|---|
| 0 | learner DDP rank0 | 1 | ~32 GB | 32 GB | 64 GB | TB / checkpoint / LoRA snapshot 在 rank0 |
| 1 | learner DDP rank1 | 1 | ~32 GB | 32 GB | 64 GB | DDP 副 rank |
| 2 | collector × 1 | 1 | ~22 GB | 22 GB | 74 GB | no_grad inference |
| 3 | collector × 1 | 1 | ~22 GB | 22 GB | 74 GB | no_grad inference |

合计 4 个 model 副本（2 learner + 2 collector），全部跑在 4 张 H20 上。默认先保证
CUDA context、显存碎片和服务器 compute mode 都稳定；确认单卡允许多进程且 replay
长期不足时，再手动把 `COLLECTORS_PER_GPU` 调到 2 或 3。

### 0.3 GPU 资源配置深度分析（这是你让我重点考虑的部分）

#### 0.3.1 为什么 learner 一张卡只跑 1 个进程，不混部 collector

**显存上能塞**：trainer 32 GB + collector 22 GB = 54 GB，H20 96GB 装得下。
**SM 算力上撞死**：Qwen3-VL-4B 单步 forward / decode 占用 ~30-50 SM（H20 有 132 SM）。
collector 的 80-step decode 是连续 80 次 forward；如果跟 trainer backward 撞在一张卡：

- 不开 MPS：两个 process 的 kernel 在 SM 调度器上**时分复用**，trainer backward 被
  collector decode 抢占；trainer 步进时间从 30 秒涨到 100+ 秒
- 开 MPS：kernel 可以并发但仍抢同样 132 SM，每个进程实际拿到 ~50% 算力；trainer
  从 30 秒涨到 ~60 秒

无论开不开 MPS，**混部都把 learner 拖慢到无法接受**。learner DDP allreduce per
backward 还会因为两 rank 步进时间不一致暴露问题（虽然 2h timeout 兜底，但训练吞吐
直接腰斩）。结论：**learner 卡绝不混部 collector**。

#### 0.3.2 为什么默认 collector 一张卡 1 个进程

H20 96GB / 22GB ≈ 4 个理论上限，但实际有效进程数还取决于服务器 compute mode、
CUDA context 限制、PyTorch allocator 碎片和启动期并发加载。部分机器在同一卡同时起
多个 4B 模型进程时会报 `CUDA-capable device(s) is/are busy or unavailable`，因此
生产脚本默认先用 1 个 collector / 卡，跑稳后再调高。

**单卡多 collector 吞吐实测预估**（按 SM 抢占模型估）：

| collector / 卡 | 单进程速度衰减 | 卡内总吞吐倍数 | MPS 关 | MPS 开 |
|---|---|---|---|---|
| 1 | 0% | 1.0× | 1.0× | 1.0× |
| 2 | ~30% | 1.4× | 1.4× | 1.7× |
| 3 | ~45% | 1.65× | 1.65× | 2.2× |
| 4 | ~58% | 1.68× | 1.68× | 2.4× |
| 5 | ~65% | 1.75× | 1.75× | 2.5× |
| 6 | ~70% | 1.8× | 1.8× | 2.5× |

**3 个/卡** 仍可作为手动吞吐优化档：从 2→3 卡内吞吐涨 18%（1.4 → 1.65），
从 3→4 只涨 2%（1.65 → 1.68），但增加 1 个 CUDA context (~500MB)、增加内核调度
overhead、增加 OS 进程切换。继续往上加纯亏。

**MPS 是后续优化项，不是默认**：
- MPS daemon 启动后可以让 3 个/卡的吞吐倍数从 1.65× 提升到 2.2×（**约 33% 增益**）
- 但 MPS 增加了一个 system-level 依赖；某个 collector 内部崩溃可能拖累其他进程
- 推荐：**第一阶段不开 MPS 且 1 个 collector/卡**，跑稳后用 `nvidia-smi dmon` 看 GPU-Util，如果 collector
  卡持续 < 70% util、replay 长期空，再考虑开 MPS

#### 0.3.3 吞吐量平衡核对

经验值（v3 训练日志：rank0 95min 跑 932 frame ≈ 6.1 sec/frame）：

- **v3 完整 on-policy** 一帧：teacher prefill+decode×2 + student prefill+decode×2 +
  3-6 个 loss forward + 1 backward ≈ 6 sec
- **v4 collector** 一帧（省了 student step1 自由生成 + 全部 loss/backward）：
  ≈ 4 sec/frame × 14 frame = **56 sec/traj**
- **v4 learner** 一帧（只重 prefill 一次 + 3-6 个 teacher-forced 小 forward）：
  ≈ 3 sec/frame × 14 + 1 个 traj-end backward 3 sec = **45 sec/traj**

**生产 vs 消费**（带 SM 竞争因子）：

| 资源 | solo speed | × 进程数 | 多进程因子 | 实际 traj/min |
|---|---|---|---|---|
| 2 collectors (1/卡 × 2 卡, 默认) | 1.07 traj/min | 2 | 1.0 | **2.14** |
| 6 collectors (3/卡 × 2 卡, 手动扩容) | 1.07 traj/min | 6 | 0.55 | **3.5** |
| 2 learners (1/卡 × 2 卡) | 1.33 traj/min | 2 | 1.0 (无竞争) | **2.66** |

默认 production 2.14 / consumption 2.66 ≈ **0.80×**，优先保证启动稳定；若
`train/replay/size` 长期偏低，再把 collector 手动调到 2/卡或 3/卡。3/卡时
production 3.5 / consumption 2.66 ≈ **1.32×**，buffer 会缓慢充满。开 MPS 后
production 可继续提升，但 staleness 上升——MPS 是吞吐优化但代价是更老的样本。

#### 0.3.4 监控指标与扩容触发条件

跑起来后看以下指标决定要不要调：

| 现象 | 含义 | 怎么调 |
|---|---|---|
| `train/replay/size` 长期 < 30 | collector 跟不上 | 先把 collector 加到 2/卡，再试 3/卡或 MPS |
| `train/replay/size` 长期 ≥ CAPACITY | learner 跟不上 | 把 collector 减到 1/卡，或加 grad_accum |
| `train/replay/avg_age_minutes` > 90 | staleness 过大 | 减 REPLAY_CAPACITY 到 128 |
| `train/replay/avg_age_minutes` < 10 | 接近 on-policy 没好处 | 加 REPLAY_CAPACITY 或减 collector |
| collector GPU-Util < 60% | SM 没占满 | 加 collector 进程或开 MPS |
| learner GPU-Util < 80% | 数据等待 | replay 不够，加 collector |

#### 0.3.5 显存余量用来做什么

每张 collector 卡还有 30GB 富余，每张 learner 卡还有 64GB 富余。这些**故意不去用**：

- 留出 backward 的临时 buffer 突发（峰值显存可能比稳态高 5-10 GB）
- 留出图像 LRU cache（trainer 反复读同一组图像时省 IO）
- 留出 KV cache fragment 碎片（长 episode 时 PyTorch allocator 碎片化）
- 给 nvidia-smi / monitoring / 数据集元信息一些空间

如果之后想榨干显存（不推荐），可以试探性把 collector 加到 4/卡，但**不要超过 4**，
也不要在 learner 卡上叠任何东西。

### 0.4 同步 / 异步规则

| 关系 | 同步性 | 协调机制 |
|---|---|---|
| collector ↔ collector | 完全异步 | 独立 TCPStore 或文件锁抢 episode；文件 rename 原子写入 |
| learner ↔ learner | 同步（DDP） | NCCL allreduce per backward, world_size=2 lockstep |
| collector ↔ learner | 异步 | 磁盘 replay + LoRA snapshot 版本号 |

**DDP 不报错的关键**：collector 进程**不调用 `dist.init_process_group`**。learner 的
DDP process group 只包含 learner ranks（rank0 在 GPU0、rank1 在 GPU1），与
collector 进程在 OS 层完全独立。同机器跑 collector 和 learner 不会让 DDP 出错——
它们是不同进程，不在同一个 process group 里。

**"一张卡多进程是否安全"**：CUDA 多进程在 H20 上稳定（驱动 570.211.01 支持），但
要避免：
- 显存 + activations 估算后留 ≥10% buffer（避免 OOM kill 后整个 DDP 跟着崩）
- learner 进程的 `--no-grad-checkpoint` 默认开（grad checkpointing 与 KV reuse 冲突，
  v3 已踩过坑）
- 不要在 learner GPU 上再叠 collector——collector 的 80-step decode 会跟 learner 的
  backward 抢算力，learner 步进时间会从 30 秒涨到 100+ 秒

### 0.5 Trajectory 文件 schema（1 文件 = 1 episode）

文件命名：`ready/coll{N}_{utc_ms}_{run_id}.jsonl`，一行一条记录（首行是 episode 头部，
后续每行 1 帧）：

```json
{"schema":"sft_v4_rollout_v2","kind":"header",
 "collector_id":"coll0",
 "policy_version":12000,
 "created_at":1719200000.0,
 "frame_count":14,
 "episode":{"run_id":"...","scenario":"...",
            "anchors":[f0,f1,f2,f3,f4],"delta":7,
            "gt_scene":"...",
            "gt_event_sequence":["initial","e1","e2","e3","final"]}}
{"kind":"frame","frame_idx":123,"phase":"A",
 "image_paths":["oldest.jpg","...","newest.jpg"],
 "memory_before_frame":{"road_structure":"JUNCTION","scene":"...","status":"...","subgoal":"...",
                        "ego_to_goal_xy":[12.3,-1.4]},
 "memory_after_step1":{"road_structure":"JUNCTION","scene":"...","status":"...","subgoal":"...",
                       "ego_to_goal_xy":[12.3,-1.4]},
 "init_was_correct":true,         // Phase A 首帧才有意义；同时要求 road_structure 和 scene 正确
 "noise_injected":false,           // Phase B 帧首是否触发噪声扰动
 "skip_correction_applied":false,  // 上一帧 step1 未过导致跳过 step2/3 后，下一帧帧首是否纠偏
 "skip_correction_scene_noisy":false, // skip 纠偏时 scene 是否用了同桶非 GT 小扰动
 "teacher_step1_text":"...",       // base teacher step1 原始分析
 "teacher_step1_target":"...",     // 规范化后的 ANALYSIS + ROAD_STRUCTURE 训练 target
 "teacher_step2_raw":"...",        // 仅 step2_fired=true 时有；否则 null
 "teacher_step2_target":"...",     // 仅 step2_fired=true 时有；否则 null
 "teacher_step3_raw":"...",        // base teacher 原始输出，仅 step3_fired=true 时有；否则 null
 "teacher_step3_target":"...",     // 规范化后的 ANALYSIS + STATUS/SUBGOAL target；否则 null
 "student_step1_raw":"...",        // 用来复现 step2 前的 student 对话上下文
 "student_step2_raw":"...",        // 仅 step2_fired=true 时有；用于复现 step3 前上下文
 "student_step3_raw":"...",        // 仅用来复现 memory 帧末更新；否则 null
 "step2_fired":true,
 "step3_fired":true,
 "teacher_targets":{"step1":"...","step2":"...","step3":"..."},
 "teacher_raw_outputs":{"step1":"...","step2":"...","step3":"..."},
 "student_outputs":{"step1":"...","step2":"...","step3":"..."},
 "gt":{"road_structure":"JUNCTION","scene":"...","status":"...","subgoal":"..."},
 "flags":{"step2_ran":true,"step3_ran":true,"rs_flip":false,"scene_flip":false,
          "skip_correction_applied":false,"skip_correction_scene_noisy":false}}
{"kind":"frame","frame_idx":124,...}
...
```

**不存图像本身**：14 帧 stitched RGB ~14MB / traj × 256 ≈ 3.5GB，会占满磁盘；
lead_data 本来就常驻盘，trainer 按路径实时读 + LRU cache 即可。

**不存 KV cache**：单帧 KV ~150MB，14 帧 ~2GB / traj × 256 ≈ 500GB，磁盘和 IO 都吃不消；
而且 KV 与 LoRA 版本强绑定，跨版本反序列化等同灾难。trainer 直接重 prefill，单帧 ~2 秒，
按 traj 算 ~30 秒，远不是瓶颈。

**为什么 memory_after_step1 也要存**：step2 的 `SCENE_CHOICES` 必须来自
step1 更新后的 `memory.road_structure`。collector 如果在 step1 后把桶从 A 改成 B，
learner 重放时也必须用 B 桶构造 step2 prompt；否则 teacher target 虽然是 GT scene，
学生看到的候选表却可能来自旧桶，训练信号会错配。

**为什么 student_step2_raw / student_step3_raw 也要存**：trainer 不再现场生成，但需要
"假装 collector 当时生成了这些"才能正确推进 KV 对话上下文和 memory。trainer 直接从
trajectory 拿这些，跳过任何 generate。

### 0.6 Collector loop

```python
# qwen3vl_local/sft_v4/collect.py 伪代码
parse_args()                        # --replay-dir, --lora-dir, --collector-id, ...
seed_per_collector()                # seed = base_seed + hash(collector_id)
load_base_qwen()                    # frozen, no LoRA
peft_model = inject_lora()          # 初始 r=16 LoRA 用 zero-init B
load_lora_from(latest_lora_dir / f"v_{read_pointer()}")
current_lora_version = pointer

last_reload_at = 0
ep_count = 0
while not exists(stop_sentinel):
    # 1. 抢 episode (TCPStore counter; epoch 概念由 collector 自己 wrap 实现)
    idx = atomic_claim_next_episode()       # idx 跨 epoch 自然 wrap
    ep = train_ds.rows[idx % train_total]

    # 2. 周期 reload LoRA
    if ep_count - last_reload_at >= REFRESH_EVERY_EPS \
            or time() - last_reload_time >= REFRESH_EVERY_SEC:
        pointer = read_pointer()
        if pointer > current_lora_version:
            load_lora_from(latest_lora_dir / f"v_{pointer}")
            current_lora_version = pointer
            last_reload_at = ep_count
            last_reload_time = time()

    # 3. roll out
    traj = []
    memory = init_memory_v4(ep, phase_a_correct_prob=0.7)   # 70% Phase A 初始 road_structure = GT
    need_skip_correction = False
    for frame in range(ep.frame_start, ep.frame_end + 1):
        phase = "A" if (f1 - delta) <= frame <= (f1 + delta) else "B"
        noise = False
        skip_correction = False
        if phase == "B":
            memory = force_memory_to_gt_chain(memory)      # 分层弱纠偏
            if random() < PHASE_B_NOISE_PROB:              # 默认 0.15
                memory.scene = random_non_gt_scene_in_bucket(memory)
                noise = True
        if need_skip_correction:
            memory = correct_memory_after_step1_skip(memory, gt_scene)
            skip_correction = True
            need_skip_correction = False

        images = load_images(ep, frame)

        # Step 1：teacher/student 都基于帧首 memory 做 road_structure 判断。
        teacher_step1_raw = teacher_generate_step1(images, memory, gt_road_structure)
        teacher_step1_target = canonicalize_teacher_step1(teacher_step1_raw, gt_road_structure)
        student_step1_raw = student_generate_step1(images, memory)
        memory = update_memory_after_step1(memory, parse_road_structure(student_step1_raw))
        memory_after_step1 = memory.copy()

        # Step 2：只有 layer-1 命中时才跑；prompt 的 SCENE_CHOICES 来自 memory_after_step1。
        step2_fired = should_trigger_step2(memory.road_structure, gt_road_structure)
        if step2_fired:
            teacher_step2_raw = teacher_generate_step2(images, memory, gt_scene)
            teacher_step2_target = canonicalize_teacher_step2(teacher_step2_raw, gt_scene)
            student_step2_raw = student_generate_step2(images, memory)
            memory = update_memory_after_step2(memory, parse_scene(student_step2_raw))
            memory_after_step2 = memory.copy()
        else:
            teacher_step2_raw = teacher_step2_target = student_step2_raw = None
            memory_after_step2 = memory.copy()
        need_skip_correction = not step2_fired

        # Step 3：只有 layer-2 scene 命中时才跑；未触发时不监督 status/subgoal。
        step3_fired = step2_fired and should_trigger_step3(memory.scene, ep.gt_scene)
        if step3_fired:
            teacher_step3_raw = teacher_generate(images, memory, gt_status, gt_subgoal)
            teacher_step3_target = canonicalize_teacher_step3(teacher_step3_raw, gt_status, gt_subgoal)
            student_step3_raw = student_generate(images, memory)
            memory = update_memory_after_step3(memory, parse_status_subgoal(student_step3_raw))
        else:
            teacher_step3_raw = None
            teacher_step3_target = None
            student_step3_raw = None

        traj.append(frame_record(...))

        # 帧末只预取下一帧 goal；三层 memory 已在 step1/2/3 各自结束时更新。
        memory = prefetch_next_frame_goal_xy(memory, ep, frame + 1)

    # 4. atomic write
    tmp = pending_dir / f"{collector_id}_{utc_ms()}_{ep.run_id}.tmp"
    final = ready_dir / f"{collector_id}_{utc_ms()}_{ep.run_id}.jsonl"
    write_jsonl(tmp, header=ep_header, frames=traj)
    os.rename(tmp, final)

    # 5. FIFO 驱逐（race-tolerant: try unlink, EEXIST 或 ENOENT 都吞掉）
    if ep_count % EVICT_EVERY == 0:
        prune_oldest(ready_dir, capacity=REPLAY_CAPACITY)

    ep_count += 1
```

### 0.7 Learner loop（DDP world_size = 2）

```python
# qwen3vl_local/sft_v4/learn.py 伪代码
dist.init_process_group("nccl", timeout=timedelta(hours=2))
load_base_qwen()                    # frozen
peft_model = inject_lora()
broadcast_lora_from_rank0()         # 复用 v3 helper
ddp_model = DDP(peft_model,
                find_unused_parameters=False,
                broadcast_buffers=False)
optimizer = AdamW(lora_params, ...)
scheduler = LambdaLR(...)

# rank0 先把 v_0 dump 出去，让 collector 有起点
if is_rank0:
    save_lora_snapshot(version=0)

while step < max_steps and not exists(stop_sentinel):
    # 1. 等 replay 有数据
    while count_ready_files() == 0:
        time.sleep(2)

    # 2. 各 rank 独立 random.choice 一条 traj（数据多样性 = batch=2 等效）
    traj_path = random.choice(list_ready_files())
    try:
        traj = read_jsonl(traj_path)
    except (FileNotFoundError, JSONDecodeError):  # 被 collector 驱逐 / 写一半被读
        continue

    # 3. 复现 teacher-forced loss；每帧立刻 backward，避免整条 traj 计算图堆到显存里
    optimizer.zero_grad(set_to_none=True)
    frame_count = len(traj["frames"])
    with ddp_model.no_sync():
        for frame in traj["frames"]:
            images = load_images(frame["image_paths"])
            memory_before = frame["memory_before_frame"]
            prompt_state = kv_prefill(system + images + step1_user(memory_before))
            a1, rs1 = teacher_forced_ce_step1(prompt_state, frame["teacher_step1_target"])
            prompt_state = append_assistant(prompt_state, frame["student_step1_raw"])

            if frame["step2_fired"]:
                # 这里必须使用 collector 写下的 memory_after_step1，不能使用 memory_before。
                memory_after_step1 = frame["memory_after_step1"]
                prompt_state2 = append_user(prompt_state, step2_user(memory_after_step1))
                a2, sc = teacher_forced_ce_step2(prompt_state2, frame["teacher_step2_target"])
                prompt_state2 = append_assistant(prompt_state2, frame["student_step2_raw"])
            else:
                a2 = sc = 0

            if frame["step3_fired"]:
                memory_after_step2 = frame["memory_after_step2"]
                prompt_state3 = append_user(prompt_state2, step3_user(memory_after_step2))
                a3, st, sg = teacher_forced_ce_step3(prompt_state3, frame["teacher_step3_target"])
            else:
                a3 = st = sg = 0
            frame_loss = W_A1*a1 + W_RS1*rs1 + W_A2*a2 + W_SC*sc + W_A3*a3 + W_ST*st + W_SG*sg
            (frame_loss / max(frame_count, 1)).backward()
    manual_allreduce_mean_lora_grads()

    # 4. clip/step；optimizer 仍然是一条 trajectory 一步
    clip_grad_norm_(lora_params, MAX_NORM)
    optimizer.step()
    scheduler.step()
    step += 1

    # 5. rank0 周期保存 LoRA snapshot
    if is_rank0 and step % SNAPSHOT_EVERY_STEPS == 0:
        save_lora_snapshot(version=step)
        cleanup_old_versions(keep=3)

    # 6. rank0 周期保存训练 checkpoint (含 optimizer state)
    if is_rank0 and step % CHECKPOINT_EVERY_STEPS == 0:
        save_full_checkpoint(step)
```

frame loop 内使用 DDP `no_sync()`，所以不同 rank 抽到不同长度 trajectory 时不会产生
不同数量的 DDP gradient collective。每个 rank 本地累完一条 trajectory 后，再按 LoRA
参数的固定顺序手动 `all_reduce(SUM) / world_size` 一次。这样既不会把整条 episode
的计算图攒到显存里，也不会出现 collective 数对不上的死锁。

### 0.8 LoRA snapshot 原子切换协议

```
$OUTPUT_DIR/latest_lora/
  v_0/                       <- rank0 启动时立刻 dump
    adapter_config.json
    adapter_model.safetensors
  v_1000/
  v_2000/
  current_version.txt        <- 内容是版本号字符串，例如 "2000"
```

**rank0 写入流程**：

1. `peft_model.save_pretrained(latest_lora_dir / f"v_{step}_writing")`
2. `os.rename(v_{step}_writing, v_{step})` ← 原子
3. 写 `current_version.txt.tmp` 内容 `str(step)` → `os.rename` 成 `current_version.txt` ← 原子
4. 删除最旧的 v_X 目录（保留最近 3 个），但**保护当前 pointer 指向的版本和上一个版本**
   不被删（避免 collector 正在加载时被删走）

**collector 读取流程**：

1. `pointer = int(open(current_version.txt).read().strip())`
2. 若 `pointer > self.current_lora_version` 且 `latest_lora_dir / f"v_{pointer}"` 存在：
3. `model.load_adapter(latest_lora_dir / f"v_{pointer}", adapter_name="default")`
4. `self.current_lora_version = pointer`

**竞态保险**：rank0 不删 ≥ pointer-1 的版本；collector 读 pointer 后立刻加载，最坏只
会撞上"pointer-1 被删但 pointer 还在"，加载新版本仍然成功。

### 0.9 Replay FIFO 设计与驱逐

```
$OUTPUT_DIR/replay/
  pending/      <- 写入中的临时文件 (.tmp 后缀)
  ready/        <- 完整可读 traj，FIFO 容量 CAPACITY
  failed/       <- 写入后 validation 失败的（schema 错、frames 为空等）
```

**容量**：默认 `REPLAY_CAPACITY=256`，约 256 × 3MB ≈ 750MB 磁盘。

**抽样**：trainer 用 `random.choice(os.listdir(ready_dir))`。256 个文件 listdir + choice
开销 ~ 几毫秒，可忽略。

**驱逐**：每个 collector 每写完 N 条（默认 N=8）执行一次驱逐：
```python
files = sorted(ready_dir.glob("*.jsonl"), key=trajectory_header_created_at)
while len(files) > REPLAY_CAPACITY:
    try:
        files[0].unlink()
        files.pop(0)
    except FileNotFoundError:   # 别的 collector 已经删了
        files.pop(0)
```
`trajectory_header_created_at` 读取 jsonl 第一行 header 的 `created_at`，与
`train/replay/avg_age_minutes` 的统计口径完全一致；只有旧文件或坏文件才回退到
文件 mtime。

**staleness 控制**：
- `REPLAY_CAPACITY` 小（64-128）→ 更新鲜，但 collector 喘息时间短，buffer 容易空、
  trainer 等待
- `REPLAY_CAPACITY` 大（512-1024）→ trainer 永远有数据，但样本年龄分布更老，policy
  drift 更明显
- 默认 256 在估算上 1 小时左右换一遍，对应 ~200 trainer step 的 LoRA 漂移，比较温和

### 0.10 Memory 扰动新规则（v4 对 v3 的核心改动）

v3 的"Phase A 100% 非 GT 初始 + Phase B 100% 修回 GT"对 student 来说太硬：Phase A
从第一帧就要面对最难的"看证据推翻 memory"任务，收敛慢。v4 在两端都引入软化：

| 时间点 | v4 规则 | 与 v3 对比 |
|---|---|---|
| **Phase A 初始**（frame=f1-δ） | `random() < P_INIT_CORRECT (默认 0.7)` 时 road_structure 设为 GT 桶，scene 在 GT 桶内联合采样；否则从非 GT 桶联合采样 | v3 D3：100% 非 GT |
| Phase A 帧末 memory 更新 | 同 v3：学生 step2/step3 输出自更新 | 不变 |
| Phase B 帧首默认 | `force_memory_to_gt_chain`：road_structure / scene 分层弱纠偏，scene 变更时重置 status/subgoal | 扩展 v3 弱纠偏 |
| **Phase B 帧首扰动** | `random() < PHASE_B_NOISE_PROB (默认 0.15)` 时只在当前 road_structure 桶内把 scene 改为非 GT | v3 无此规则 |
| **skip 后下一帧帧首纠偏** | 仅上一帧 `step2_fired=False` 时触发一次；road_structure 强制 GT 桶，scene 大概率 GT / 小概率同桶非 GT，status/subgoal 重置为所选 scene 的 init | 新增 D28，防止 step1 连错导致长期没有 step2/3 样本 |
| step2/step3 触发条件 | step1 后 `memory.road_structure == gt_road_structure` 才跑 step2；step2 后 `memory.scene == gt_scene` 才跑 step3 | v4 新增 layer-1 门控 |

设计意图：

- **70% Phase A 初始 layer-1 正确**：让 Phase A 同时包含"对的别改"和"错了要改"两种监督。
  v3 把这两件事分给 Phase A / Phase B 完全分离的设计被实际训练数据打脸——很多 epoch
  跑下来 Phase A 的 status/subgoal CE 一直比 Phase B 高，因为 student 在错初始下
  连 step1 都很难学到位。0.7 默认值让 layer-1 门控更早有密集监督，同时 scene 仍在
  GT 桶内保留 50% 错率用于同桶纠偏。
- **Phase B 15% 噪声**：v3 Phase B 几乎所有帧 scene=GT、step3 必触发。如果模型在
  Phase A 没学会"看证据推翻"，Phase B 这条信号永远训练不到。15% 噪声让 Phase B 偶尔
  暴露这个监督，但 85% 帧仍然 step3 主导，**status/subgoal 监督密度不会丢**。
- **触发链对称**：step1 错桶时跳过 step2/3，只计 L_A1/L_RS1；step1 正确后才在收窄
  的 bucket 内训练 scene；scene 正确后才训练 status/subgoal。这样 teacher target 永远
  出现在学生实际看到的候选表里，不会发生跨桶监督错配。
- **skip 后纠偏不常态化**：只有已经发生一次 step2/3 跳过后，下一帧进入内循环前才调用
  `correct_memory_after_step1_skip`。纠偏后的 `BELIEVED_SCENE` 默认是真实 scene，
  `SKIP_CORRECTION_SCENE_NOISE_PROB=0.15` 时可同桶小扰动；`STATUS/SUBGOAL` 始终重置为
  该 scene 的 init/first subgoal，保证后续 `SCENE_CHOICES` 与 memory 下游内容同源。

### 0.11 启动 / 退出协议

启动顺序（由 `launch_offpolicy.sh` 编排）：

1. learner rank0 先单独跑起来，`build_dataset` 检查通过后立刻 dump `v_0/`
2. learner rank0 写 `current_version.txt = "0"`
3. learner rank1 通过 torchrun join，DDP world_size=2 建立完成
4. learner 等 `ready/` 非空（设 `REPLAY_STARTUP_TIMEOUT_SEC=600` 兜底；timeout
   判定先在 learner ranks 间 allreduce，rank0 写 `STOP`，所有 rank barrier 后 cleanup，
   最后抛 `TimeoutError`，避免启动期 collective 顺序错位）
5. collectors 以独立 Python 进程批量启动（任意顺序），各自读 `current_version.txt`
   加载 v_0，开始 rollout 写 `ready/`
6. trainer 开始消费

退出协议：

1. learner 到 `max_steps` / `--check` 退出条件时，保存 `final/` 并写 `STOP` 哨兵到 `$OUTPUT_DIR/`
2. learner rank0 dump `final/` LoRA
3. 各 collector 每个 episode 结束查 `STOP`，存在则正常退出（不强 kill，让正在写
   `pending/` 的 traj 完成）
4. `launch_offpolicy.sh` 监控所有 collector 进程退出后收尾日志
5. 如果用户或 launcher 预先写入 `$OUTPUT_DIR/STOP`，learner 所有 rank 会先用一个
   轻量 allreduce 同步该停止请求，然后在下一个 step 边界退出；非视觉熔断场景下仍保存
   当前 adapter 到 `final/`，便于手动早停后继续 eval/probe。

### 0.12 当前 v4 子包训练入口状态

v4 off-policy 已经落地为四个新入口：

- `replay.py`：trajectory schema、原子写入、文件锁 counter、FIFO 驱逐与读取。
- `collect.py`：collector 入口，独立进程、无 DDP，负责 rollout 并写 `replay/ready/`。
- `learn.py`：learner DDP 入口，world_size=2，只读 replay 做 teacher-forced loss +
  backward，并周期发布 LoRA snapshot。
- `launch_offpolicy.sh`：一键编排 2 learner + 默认 2 collector，处理 GPU 切分、run 子目录、
  STOP 哨兵和日志。

`sft_v4/train.py` / `train.sh` 保留为 on-policy 兼容调试入口，启动时会打印 warning；
生产训练只走 `launch_offpolicy.sh`。`eval.py` / `probe.py` 仍用于最终 adapter 的自由生成
评估和 case dump。

### 0.13 已实现落地清单

本节记录实现闭环，方便后续 review：

| 序号 | 目标 | 文件 | 验收 |
|---|---|---|---|
| 1 | replay 抽象 | `replay.py` | 已实现 schema 校验、原子写、文件锁 counter、FIFO 驱逐 |
| 2 | collector 单进程 | `collect.py` | 已实现 snapshot 等待/刷新、rollout、Phase B 噪声、trajectory 写盘 |
| 3 | learner 单卡 | `learn.py` | 已实现单进程读取 replay、teacher-forced loss、optimizer/scheduler |
| 4 | learner DDP | `learn.py` | 已实现 learner-only DDP；按帧 micro-backward，frame loop `no_sync()`，每 traj 固定顺序手动平均 LoRA grad |
| 5 | 多 collector 并发 | `replay.py` / `collect.py` | 已实现文件锁 counter，collector 不进 NCCL |
| 6 | snapshot 切换 | `learn.py` / `collect.py` | 已实现 `latest_lora/v_<step>/` 原子发布与 collector reload |
| 7 | launcher 编排 | `launch_offpolicy.sh` | 已实现 2 learner + 默认 2 collector、STOP 哨兵、run 子目录与日志 |

### 0.14 已锁定参数 / 不再待拍板

以下是本轮已经确认的 v4 默认值；后续实现按这些值写 CLI 默认参数和文档示例：

| 项 | 已锁定默认值 | 说明 |
|---|---|---|
| Phase A 初始正确率 | `P_INIT_CORRECT=0.7` | 70% GT road_structure；scene 在对应桶内联合采样；D27 覆盖早期 0.5 设定 |
| Phase B 噪声率 | `PHASE_B_NOISE_PROB=0.15` | 弱纠偏到 GT 后，15% 概率再注入同桶非 GT scene；可调范围保留 `[0.0, 0.3]` |
| skip 后纠偏 scene 扰动率 | `SKIP_CORRECTION_SCENE_NOISE_PROB=0.15` | 仅上一帧 step1 未过并跳过 step2/3 后，下一帧帧首触发一次；scene 大概率 GT，小概率同桶非 GT |
| learner DDP world size | `2` | GPU0/GPU1 各 1 个 learner rank；只 learner 进入 NCCL process group |
| collector 并发 | 默认 `2` | GPU2/GPU3 各 1 个 collector；确认单卡多 CUDA 进程稳定后可手动设 `COLLECTORS_PER_GPU=2/3` |
| learner batch | 每 rank `1` 条 trajectory | effective batch = 2；不做 per-frame batch 拼接，降低 mask/padding 风险 |
| snapshot 频率 | `SNAPSHOT_EVERY_STEPS=1000` | rank0 每 1000 learner step 发布一版 `latest_lora/v_{step}/` |
| LR schedule | `--max-steps` + cosine + `warmup_ratio=0.03` | off-policy 没有 epoch 概念，用总 step 定义 scheduler |
| in-loop eval | 默认关闭 | 训练后用 `eval.py`；需要快检时从 replay subsample 做 quick eval，不能阻塞 DDP 主循环 |
| replay 抽样 | 各 learner rank 独立 `random.choice` | 不用 `DistributedSampler`；off-policy replay 允许重抽，两个 rank 抽到不同 traj 即形成 batch 多样性 |
| teacher analysis 清洗 | 只剥 label 行 + prompt marker | D29 拍板放弃严格 4-heading 校验，scripted target = raw 全文；分 step 监督下 analysis 含 GT/选项名不污染 L_RS1/L_SC/L_ST/L_SG |

资源部署也已锁定：**learner 卡不混部 collector**。虽然 H20 96GB 显存能塞下
1 learner + 1 collector，但 collector 长 decode 会抢 SM，让 learner backward 变慢并放大
DDP rank 间步进抖动；因此 GPU0/GPU1 只训练，GPU2/GPU3 只采集。

---

## 1. v4 与 v2 的本质区别

| 维度 | v2 | v4 |
|---|---|---|
| 训练单元 | 单帧 anchor 的串行选择题 | **一个 sub-scenario 的时间序列**（外循环按时间步推进） |
| Memory | 无；只有 `PREVIOUS_STATUS_HINT` 字段 | **学生自维护的纯文本 Memory**，每帧外循环之间链接 |
| Teacher | 无 | **Frozen Qwen3-VL-4B-Instruct teacher**（与 student 共享 base，通过 `disable_adapter` 切换），喂特权 GT，产出"以学生口吻"的纠错分析 |
| 分析监督 | 无（已废弃 ANALYSIS 路线） | **Hindsight Oracle / OPD 范式**：teacher 即时生成的分析就是 GT answer，student token-CE 对齐 |
| 离散监督 | scene / status / subgoal 值 token CE | 同 v2，**权重显著大于分析 loss** |
| Wrong-scene 增强 | jsonl 里 `--wrong-scene-ratio` 注入错场景 | 错场景**来自学生自身 memory 漂移**，天然产生；phase B 反向用 GT scene 注入 |
| 数据持久化 | jsonl 训练样本 | **不写训练样本**，只写 `episode_index.jsonl` |

整体精神：用 teacher 的自然语言纠错把"错记忆 → 正记忆"这个动作蒸馏进 student，
让 student 既学单帧分类，又学"看到证据就推翻 memory"的连续推理。

---

## 1.1 代码地图与中文注释约定

当前子包代码已按“下个维护者先读注释再读实现”的口径补齐中文说明。新增的
off-policy 文件现在都有三层注释：

1. module docstring：说明这个进程/文件在 actor-learner 里的角色。
2. function docstring：说明函数输入输出、并发边界、为什么这么做。
3. 关键代码块注释：解释原子写、DDP 同步、snapshot 发布、Phase B 噪声、memory 推进等
   容易误读的实现点。

| 文件 | 主要职责 | 先读位置 | 状态 |
|---|---|---|---|
| `prompts.py` | Memory 文本格式、状态机更新（含 `p_init_correct=0.7` 联合初始化）、三步 prompt、输出解析、teacher analysis 清洗 | module docstring、`Memory`、`init_memory`、`update_memory_after_step1/2/3` | 已实现 |
| `build_dataset.py` | 从 `keyframes_all_scenarios.json` 构建 episode index，只写元数据 | 文件头、`build_episode`、`load_keyframe_runs` | 保持不变（off-policy 仍以 episode 为采集单位） |
| `replay.py` | trajectory schema、原子写、FIFO 驱逐、读取、文件锁 counter | 文件头、`ensure_replay_dirs`、`directory_lock`、`claim_episode_index`、`write_trajectory`、`evict_old` | 已补详细中文注释 |
| `collect.py` | collector 入口：抢 episode、rollout、写 replay、加载 LoRA snapshot | 文件头、`_load_adapter_state_if_present`、`_inject_phase_b_noise`、`collect_episode`、`main` | 已补详细中文注释 |
| `learn.py` | learner DDP 入口：从 replay 抽 traj、teacher-forced loss + backward、发布 snapshot | 文件头、`setup_distributed`、`_sync_bool`、`trajectory_loss`、`publish_snapshot`、`main` | 已补详细中文注释 |
| `launch_offpolicy.sh` | 一键启动 collectors + learners 编排脚本 | 脚本顶部、路径/超参/env 块、`pick_idle_gpus`、learner/collector 启动块、STOP 收尾块 | 已补详细中文注释 |
| `train.py` / `train.sh` | on-policy 兼容入口，仍按 work-stealing+local-SGD 跑；只用于 debug / baseline 对照 | `main` 入口 warning | **v4 生产不走这条路径** |
| `eval.py` | 自由生成评估；不做 Phase B GT 注入；可选 teacher BLEU | 文件头、`_generate_next_with_kv`、`main` | 保持不变 |
| `probe.py` | case-level dump；可选 teacher privileged prompt/text | 文件头、`main` | 保持不变 |
| `check_loss_mask.py` / `test_*.py` | 静态 mask、memory 状态机、KV 复用、teacher 输出诊断测试 | 各自 `main` docstring | 保持不变 |

注释原则：

- 函数 docstring 写“为什么这样做”和“与 v4 思路的关系”，避免只复述函数名。
- 关键代码块注释写状态机边界，例如 Phase B 强制覆盖、scene 错误时跳过 step3、
  teacher 自由生成与 student teacher-forced loss 的区别。
- 并发相关注释必须写清楚谁会调用、是否进入 DDP/NCCL、是否需要文件锁、是否可重试。
- 磁盘协议相关注释必须写清楚 pending/ready/current_version.txt 的原子切换顺序。
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
BELIEVED_ROAD_STRUCTURE=<road_structure> (<ROAD_STRUCTURE_LABELS[value]>)
BELIEVED_SCENE=<scene_name> (<SCENARIO_LABELS[scene]>)
BELIEVED_STATUS=<event_name> (<EVENT_DESCRIPTIONS[event]>)
BELIEVED_SUBGOAL=<event_name> (<EVENT_DESCRIPTIONS[event]>)
EGO_TO_GOAL_XY=(+12.3, -4.5) m
[/MEMORY]
```

- 描述字段直接复用 [prompt_pipeline.py](../prompt_pipeline.py) 的 `SCENARIO_LABELS` /
  `EVENT_DESCRIPTIONS`，保持与 v2 prompt 风格一致。
- `EventSequence` 不再放进 memory；step3 的 `[EVENT_OPTIONS]` 单独列出当前 scene
  事件链，避免每个 step 重复长文本。
- `EGO_TO_GOAL_XY` 精度 `+.1f` 米。**必须保留**：十字路口左/右转场景靠这条信息
  消歧（"目的地在车体左前 → 大概率左转场景"）。
- Memory **不**携带上一帧的学生分析文本，避免 prompt 滚动膨胀。

### 3.2 初始化（在 t = anchor[1] - δ）

- **`believed_road_structure` 由概率 `P_INIT_CORRECT` 决定**（默认 0.7，D27 覆盖早期 0.5）：
  - `random() < P_INIT_CORRECT` → `believed_road_structure = gt_road_structure`，
    `believed_scene` 在 GT 桶内按 50% 正确率采样（**初始结构正确**，让模型练
    "对的别改"与同桶 scene 纠偏）
  - 否则从非 GT road_structure 桶均匀随机抽，并从该错桶内抽 scene（**初始结构错误**，
    让模型练"看证据翻 memory"）
  - `seed = hash((run_id, sub_scenario_id, collector_id))` 保证 collector 端可复现；
    trainer 端 traj 里已经存好 `memory_before_frame`，不再二次采样。
- **覆盖 v3 D3 决议**：v3 D3 强制 100% 非 GT 初始，理由是"Phase A 的稀缺信号是
  翻转，混入 50% 已对的样本会浪费 Phase A 监督预算"。实测发现该论点的代价是 student
  在 Phase A 一开始就被怼到最难的任务，收敛慢；0.7 默认让 Phase A 同时承担"翻转"和
  "保持"两种监督，并补偿 layer-1 门控触发率。如果你想退回 v3 行为，把
  `P_INIT_CORRECT=0.0` 即可。
- `believed_status` ← canonical init event token（当前 `prompt_pipeline.get_full_sequence`
  里是 `"initial"`；下文口语里的 init 都指这个状态机首 token）。
- `believed_subgoal` ← `EVENT_SEQUENCE[believed_scene][1]`（init 的下一项）。
- `ego_to_goal_xy` ← 当前帧 ego 系下的 final destination 坐标。
  与 leadmot final_goal 同源：`meta["next_target_points"][-1]` 转 ego（见
  `PROJECT_CONTEXT.md`）。**meta 缺失或解析失败时 collector 直接抛 RuntimeError**，
  禁止静默 fallback 到 measurements / (0, 0)——污染过的 ego_to_goal 会让"目的地
  在车体左前 → 大概率左转场景"这条核心消歧信号失效。

### 3.3 更新规则（每帧外循环末尾）

按 Phase 分支：

**Phase A**：
1. **step 1 结束后立即更新 road_structure**：
   - 学生 step1 输出合法 `ROAD_STRUCTURE` 且不同于 memory → 改写 `memory.road_structure`。
   - 只要 layer-1 翻转，就把 `memory.scene` reset 到新桶第一个 scene，并把
     `memory.status/subgoal` reset 到该 scene 的 canonical init / first subgoal。
   - 若 `memory.road_structure_after_step1 != gt_road_structure`，直接跳过 step2/step3。
2. **step 2 结束后立即更新 scene**（仅 step1 命中 GT road_structure 时执行）：
   - 学生 step2 输出 SCENE ≠ memory.scene → memory.scene 改写为新 scene，
    同时强制 `memory.status ← canonical init event`、
     `memory.subgoal ← EVENT_SEQUENCE[new_scene][1]`。
   - 否则 keep。
3. **step 3 触发条件**（关键，且与 step2 是否改动**无关**）：
   ```
   if step2_fired and memory.scene_after_step2 == gt_scene:
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
4. **step 3 跑了的话**：
   - `memory.status ← student_step3_pred_status`
   - `memory.subgoal ← student_step3_pred_subgoal`
5. 帧末为下一帧预取 `ego_to_goal_xy`，写入 memory。
   （C3 修正：当前帧 prompt 使用进入本帧前已经写入 memory 的坐标；本帧结束后
   再读取下一帧 meta，字面上与“进入下一帧前重算”一致。）

`should_trigger_step2(memory_road_structure_after_step1, gt_road_structure)` 与
`should_trigger_step3(memory_scene_after_step2, gt_scene)` 都被抽成 `prompts.py`
helper，训练、collector、learner 重放、eval/probe 和测试共用同一真值函数，避免状态机
有第二处隐式实现。

**Phase B**（D2 弱纠偏 + 跨帧推进，加上 v4 的概率噪声）：
1. **帧外循环开始前两步串联**：先弱纠偏，再以概率注入噪声：
   ```python
   # 步 1: 分层弱纠偏
   memory = force_memory_to_gt_chain(memory, gt_road_structure, gt_scene)
       # 已等于 GT 的字段 noop；错 road_structure / scene 时连带 reset 下游字段。

   # 步 2: 概率噪声 (v4 新增；默认 PHASE_B_NOISE_PROB = 0.15)
   if random() < PHASE_B_NOISE_PROB:
       memory.scene = random_choice(current_road_structure_bucket \ {gt_scene})
       memory.status = canonical init event
       memory.subgoal = EVENT_SEQUENCE[memory.scene][1]
       noise_injected = True
   else:
       noise_injected = False
   ```
   `noise_injected` 标志写进 trajectory，方便事后审计噪声帧的 loss 占比。
2. 正常跑 step1/step2/step3，step3 触发条件依然是 `should_trigger_step3(...)`：
   - 默认帧（85%）：memory 进入时已 = GT，student 大概率预测 GT → step3 必触发。
   - 噪声帧（15%）：memory.scene = wrong，student 看图像 → 大概率仍预测 GT scene →
     step3 仍触发；少数被噪声带偏的帧 step3 跳过，但 step1 analysis + step2 SCENE
     的"看证据翻 memory"监督被加强。
3. 帧末更新 status / subgoal 同 Phase A 第 3 步——sub2 / sub3 附近的 status / subgoal
   会**跨帧累积推进**到 e2 / e3，监督密度起得来。
4. 噪声帧不打破 Phase B 的整体"假设 scene 已纠正"语义：85% 帧仍 scene=GT，跨帧推进
   status/subgoal 不受影响；15% 噪声帧只是把当前帧的 step1/step2 拉回 Phase A 式监督。

> Phase A 与 Phase B 的 prompt **完全一致**，唯一差别只在"帧开头是否被强制
> 改 memory + 是否注入噪声"。student 看不出自己在哪个 phase，自然学到 phase-agnostic
> 的修正 / 保持能力。trajectory 里 `phase` 和 `noise_injected` 字段只供 trainer 复现
> loss 时定位帧、不进入 prompt。

---

## 4. 内循环（每帧 3 步，Hindsight Oracle / OPD 范式）

### 4.1 KV cache 结构（teacher fresh / student serial）

student 仍是一条三步串行对话：step1 自由生成 ROAD_STRUCTURE，更新 memory；
step2 在更新后的 road bucket 下生成 SCENE；step3 在 scene 命中后生成 STATUS/SUBGOAL。
因此 student 的 step2/step3 继续从上一轮 KV cache 追加 user turn，保持真实 rollout
上下文。

teacher 不再串行复用 step1/step2 KV。teacher 的 step1、step2、step3 都是 fresh
dialog：每一步都重新吃 `<system_v4> + 4 stitched RGB + 本步 privileged context`，
只生成本步分析。这样 step1 的退化文本或风格漂移不会污染
step2/step3 老师分析。teacher 与 student 也不共享 KV，因为 LoRA on/off 的 K/V
数值不同。

### 4.2 Teacher 模型

- **Teacher = student 共享同一份 base 权重**，通过 PEFT 的
  `with student.disable_adapter():` 关掉 LoRA，再 `model.eval()` +
  `torch.no_grad()` 跑 generate。generate 结束后恢复 `model.train()`。
- 不实例化第二份模型，零额外显存。
- 每帧 teacher 一共 generate 三次（step1 / step2 / step3，step3 仅在触发时）。
- 切换 adapter on/off 会让 cuDNN benchmark 有少量抖动，可以忍受；并且这种切换
  与 v2 `train.py` 现有"禁用 adapter 跑 teacher、再启用 adapter 跑 student
  forward"的代码路径同构，可直接复用。

### 4.3 Step 1：Road structure 分析（OPD，road-only privileged context）

**Teacher prompt**（fresh dialog，重新吃 4 张图）：
```
<step1_user>:
  [STEP1_ROAD_CONTEXT]
  BELIEVED_ROAD_STRUCTURE=<memory road>
  EGO_TO_GOAL_XY=(x, y) m
  GROUND_TRUTH_ROAD_STRUCTURE=<gt road>
  [/STEP1_ROAD_CONTEXT]

  [ROAD_STRUCTURE_CHOICES]
  ...
  [/ROAD_STRUCTURE_CHOICES]

  4 images are ordered oldest to newest; the last image is now.
  Output exactly four lines:
  Scene Description: ...
  Critical Object Description: ...
  Reasoning on Intent: ...
  Memory Judgment: ...
  Judge whether the believed road structure should be kept or changed toward the ground truth.
```

- Teacher generate → `analysis_1_teacher`（鼓励按
  `Scene Description:` / `Critical Object Description:` /
  `Reasoning on Intent:` / `Memory Judgment:` 四行 plain-text 顺序写，但
  不再做严格清洗：监督按 step 分开训练，analysis 里出现 GT / 选项名都不污染监督），
  脚本拼接 `"\nROAD_STRUCTURE: <gt_road_structure>"`，
  `max_new_tokens=384`、`do_sample=False`、
  `repetition_penalty=1.05`。
- **Student** teacher-forced：把 `analysis_1_teacher` 作为 assistant target，
  对其每个 token 算 CE → **L_A1**。
- 这一步 teacher 看到 `BELIEVED_ROAD_STRUCTURE` 与 `GROUND_TRUTH_ROAD_STRUCTURE`，
  负责解释当前 road memory 是否应 KEEP/CHANGE。

### 4.4 Step 2：场景判断（OPD，teacher 吃 GT scene）

**Teacher prompt**（fresh dialog，重新吃 4 张图）：
```
<step2_user>:
  [STEP2_SCENE_CONTEXT]
  GROUND_TRUTH_ROAD_STRUCTURE=<gt_road_structure>
  BELIEVED_SCENE=<memory.scene>
  GROUND_TRUTH_SCENE=<gt_scene>
  [/STEP2_SCENE_CONTEXT]
  [SCENE_CHOICES] under GROUND_TRUTH_ROAD_STRUCTURE = <gt_road_structure>
  [STEP2_TEACHER]
  Output exactly the same four-line analysis format as step1.
```

- Teacher generate → `analysis_2_teacher`，脚本拼接 `"\nSCENE: <gt_scene>"`，
  `max_new_tokens=384`。
- **Teacher analysis 清洗**：三步共用极简清洗器，只剥两类危险行——
  `ROAD_STRUCTURE:` / `SCENE:` / `STATUS:` / `SUBGOAL:` 整行（避免脚本追加的 GT
  被 parser 取错）和 `[STEPx]` / `[MEMORY]` 等 prompt marker。其余文本原样进
  target；只有剥完后真为空字符串才退回四行 fallback，然后由脚本追加
  `SCENE: <gt_scene>`。

**Student prompt**：与 teacher 完全相同，**只是去掉 `[GROUND_TRUTH]` 那一块**。
- Teacher-forced target = `analysis_2_teacher + "\nSCENE: <gt_scene>"`。
- **L_A2**：分析文本段 token CE。
- **L_S2**：`SCENE:` 后那一个值 token 的 CE，与 v2 口径一致
  （从 [sft_v2/prompts.py:317-329](../sft_v2/prompts.py#L317-L329) `target_spans` 思路迁移）。

### 4.5 Step 3：状态 / 子目标判断（OPD，teacher 吃 GT status/subgoal）

**触发条件（重申）**：`memory.scene_after_step2 == gt_scene`。否则整步跳过，
本帧也不更新 status / subgoal。

**Teacher prompt**（fresh dialog，重新吃 4 张图）：
```
<step3_user>:
  [STEP3_EVENT_CONTEXT]
  GROUND_TRUTH_ROAD_STRUCTURE=<gt_road_structure>
  GROUND_TRUTH_SCENE=<gt_scene>
  BELIEVED_STATUS=<memory.status>
  BELIEVED_SUBGOAL=<memory.subgoal>
  GROUND_TRUTH_STATUS=<gt_status>
  GROUND_TRUTH_SUBGOAL=<gt_subgoal>
  [/STEP3_EVENT_CONTEXT]
  [EVENT_OPTIONS] ...                           ← GT scene 的事件序列及描述
  [STEP3_TEACHER]
  Output exactly the same four-line analysis format as step1.
```

- Teacher generate → `analysis_3_teacher`，脚本拼接
  `"\nSTATUS: <gt_status>\nSUBGOAL: <gt_subgoal>"`，`max_new_tokens=384`。
- Teacher analysis 清洗同 step 2：只剥 GT 标签行和 prompt marker，其余原样进
  target；仅在剥完后真为空时退回四行 fallback，然后脚本再追加 `STATUS/SUBGOAL`。

**Student prompt**：相同，**去掉 `[GROUND_TRUTH]` 两行**。
- Teacher-forced target = teacher 全文（含 STATUS / SUBGOAL 值 token）。
- **L_A3** + **L_S3_status** + **L_S3_subgoal**。

---

## 5. 损失与权重

### 5.1 单帧 loss

| 名称 | 含义 | 默认权重 | 备注 |
|---|---|---|---|
| L_A1 | step1 分析 token CE / 监督 token 数 | 0.2 | 已 per-token normalize |
| L_A2 | step2 分析 token CE / 监督 token 数 | 0.2 | 同上 |
| L_A3 | step3 分析 token CE / 监督 token 数 | 0.2 | 同上 |
| L_S2 | step2 SCENE 值 token CE | **1.0** | 主信号 |
| L_S3_status | step3 STATUS 值 token CE | **1.0** | 主信号 |
| L_S3_subgoal | step3 SUBGOAL 值 token CE | **1.0** | 主信号 |

```
L_frame = w_A1 * L_A1
        + w_A2 * L_A2 + w_S2 * L_S2
        + 1{step3_triggered} * (w_A3 * L_A3 + w_S3_status * L_S3_status + w_S3_subgoal * L_S3_subgoal)
```

- **Per-token normalize**：每个分析 loss = `sum_CE / num_supervised_tokens`，
  防止 step1 一长串分析淹没 1-token 的 SCENE。
- `L_A*` 始终监督分析段；离散标签段由脚本追加并单独加权。
- 若 step3 未触发，整段 step3 系数为 0。

### 5.2 梯度累积粒度

- 每个 learner rank 每个 optimizer step 随机抽 **一条 trajectory**。
- trajectory 内按帧 micro-backward：第 `i` 帧 loss 乘 `1 / frame_count` 后立即
  `backward()`，释放该帧 KV/activation 图，避免显存随 episode 长度线性堆积。
- optimizer / scheduler 仍然每条 trajectory step 一次，因此 effective batch 仍是
  `world_size` 条 trajectory。
- DDP 下两个 rank 的 trajectory 长度可能不同；frame loop 放在 DDP `no_sync()` 中，
  本地累完后再按固定参数顺序手动 mean-reduce LoRA grad，保证 collective 序列完全一致。
- `per_device_batch_size` 固定为 1：每个 rank pull 一条 episode，episode 内逐帧推进
  memory 并累积到同一个 optimizer step。

### 5.3 TensorBoard 记录

最低必须有的标量：

```
train/loss_total
train/loss/{a1, rs1, a2, a3, s2, s3_status, s3_subgoal}
train/loss/{L_A1, L_RS1, L_A2, L_SC, L_A3, L_ST, L_SG}          ← PLAN 口径别名
train/loss_weight/{a1, rs1, a2, a3, s2, s3_status, s3_subgoal}  ← 静态权重，可视化用
train/step2_trigger_rate                                       ← 每帧粒度，等价 layer-1 命中率
train/step3_trigger_rate                                       ← 每帧粒度
train/fire_rate/{step2, step3}                                 ← PLAN 口径别名
train/accuracy/road_structure                                  ← PLAN 口径，当前等价 step2 fire rate
train/rs_flip_rate                                             ← step1 改 road_structure 的比例
train/scene_flip_rate                                          ← step2 改 scene 的比例
train/phase_a_frame_frac                                       ← 每 step batch 内 Phase A 帧占比
train/lr
train/grad_norm/{language, vision}                             ← 兼容口径；off-policy 实现写 train/grad_norm/language 与 train/grad_norm/vision
train/param_norm/lora_{language, vision}                       ← 兼容口径；off-policy 实现写 train/param_norm/lora_language 与 train/param_norm/lora_vision
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
  - 熔断时写 `fuse_stop_after_step_<N>/`、`fuse_reason.txt`，跳过 `final/`；
    `N` 是最后一个已完成 optimizer step，当前异常 step 的梯度会先 `zero_grad`，不会写入 adapter。
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
- Adapter 保存与 v2 同：base 只读，只存 adapter delta + `sft_v4_adapter_config.json`
  （记录 LoRA scope、视觉保险参数、训练窗口参数 δ / phase 配置）。

### 6.1 Teacher generate 超参

| Step | max_new_tokens | forced_min_tokens | do_sample | repetition_penalty | no_repeat_ngram_size | 备注 |
|---|---:|---:|---|---:|---:|---|
| 1 | 384 | 0 | False | 1.05 | 3 | road-only context + 严格四行 plain-text 分析；标签由脚本拼回 |
| 2 | 384 | 0 | False | 1.05 | 3 | true road/scene context + 严格四行 plain-text 分析；标签由脚本拼回 |
| 3 | 384 | 0 | False | 1.05 | 3 | true road/scene/event context + 严格四行 plain-text 分析；标签由脚本拼回 |

`max_new_tokens` 只作为异常生成的技术上限，不是 prompt 里的词数或句长要求；
四行 heading 内部不再限制词数，也不把清洗后的分析截到第一句。

实现层面，`train.py` 的 `_kv_generate_text` 已经把 `repetition_penalty=1.05`
落实成 HF 风格的 logits 后处理；teacher 调用不再传 `min_new_tokens`，不强制最少输出；
`max_new_tokens` 只作为异常生成护栏。`no_repeat_ngram_size`
只统计本轮生成 token，不把 prefix 算进去，避免把 prompt 里的选择表误当成重复历史。
`eval.py` / `probe.py` 通过 `LocalQwen3VLInstructEngine(repetition_penalty=1.05)`
走同一类 logits 后处理，避免训练、评测自由生成口径漂移。

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
  flags.json          ← step3_triggered / scene_flip / phase
case_0/timeline.png   ← scene-flip 与 step3 触发的时间线
case_0/episode_meta.json
```

`probe.py --with-teacher` 会额外加载一份 base Qwen teacher，逐帧写出
`step*_teacher.txt`、`step2_teacher_user.txt`、`step3_teacher_user.txt`，并在
`flags.json` 记录 step1/2/3 的 `analysis_bleu_vs_teacher`；默认不加载 teacher，
避免普通 case dump 双倍占显存。

### 8.3 E3 训练时 in-loop val

off-policy `learn.py` 不做 in-loop eval：learner 主循环只消费 replay 做
teacher-forced loss + backward，避免在 DDP 主循环里插入额外生成或 val 逻辑。训练后
单独运行 `eval.py` / `probe.py`。历史 on-policy `train.py` 仍保留自己的 quick eval
调试入口，但它不是 v4 生产路径。

### 8.4 E4 单元 / 烟雾测试

- `check_loss_mask.py`：实际覆盖 7 路 loss（L_A1 / L_RS1 / L_A2 / L_S2 / L_A3 /
  L_S3_status / L_S3_subgoal）的 token mask 正确性：分析段 token 集 / 值 token 集
  互不重叠，per-token normalize 分母对得上 train.py 里 `_append_token_ids` 的
  位置切分逻辑。
- `test_memory_update.py`：纯 Python 模拟外循环，覆盖：
  - `init_memory` 的 `P_INIT_CORRECT` 联合概率初始化（默认 0.7，同时覆盖 `0.0` / `1.0` 边界）；
  - Phase A `update_memory_after_step2` 的 4 种翻转组合；
  - Phase B 弱纠偏（D2 / D22 / D27 拍板）：road_structure / scene / status /
    subgoal 分层拉回 GT chain；已等于 GT 的字段保持 noop；
  - scene 翻转 → status = canonical init event、subgoal 重置；
  - step2 / step3 触发条件正确（直接调用 `should_trigger_step2` /
    `should_trigger_step3` helper，与翻转无关，只看更新后的 memory 是否命中 GT）。
- `test_kv_reuse.py`：构造一条 mini episode，对比"step1/2/3 复用 KV vs 全量
  重 prefill"的 student logits 数值一致（误差 < 1e-5）。
- `test_gt_leak_filter.py`：legacy 兼容测试，确认旧 no-op 接口不会关闭分析监督。

---

## 9. 关键工程约束

### 9.1 多卡分布式：off-policy actor-learner（v4 范式，废弃 v3 的 work-stealing+local-SGD）

v4 把 rollout 拆出 trainer 后，trainer 每步只做 teacher-forced loss + backward，标准
DDP 就够 lockstep——不再需要 v3 那套 work-stealing + local-SGD 兜底。

- **learner DDP world_size=2**：两个 trainer 进程通过 `torchrun --nproc_per_node=2`
  起 NCCL 进程组，默认包 `DistributedDataParallel(find_unused_parameters=False,
  broadcast_buffers=False)`。若显式开启视觉 LoRA，learner 会切到
  `find_unused_parameters=True` 并打印 warning，因为 v4 的图像 prefill 默认 no_grad，
  视觉侧 LoRA 可能没有梯度；生产默认仍是 `LORA_VISION_SCOPE=off`。每个 optimizer step
  仍是每 rank 一条 traj，但 backward 按帧 micro-step 执行；frame loop 使用
  `no_sync()`，随后手动按固定参数顺序 mean-reduce LoRA grad，因此不同帧数也不会打乱
  collective 序列。
- **collector 完全不进 DDP**：collector 进程**不调** `dist.init_process_group(NCCL)`。
  collector 之间用**独立 TCPStore 服务或纯文件锁**做 episode 抢任务计数，不复用 learner
  DDP 的默认 store，避免把采集生命周期绑到 learner process group 上。某个 collector
  卡死或挂掉只减少一个生产者，learner 和其他 collector 继续跑。
- **NCCL timeout 2 小时**：learner `init_process_group(timeout=timedelta(hours=2))`，沿用
  v3 修复后的口径。teacher-forced loss step 一般 30-40 秒/traj，DDP allreduce 永远撞不上
  超时。collector 慢不会让 learner 发起 collective——learner 在 replay 空时是 sleep
  重试，没有 NCCL 操作。
- **同卡多进程互操作**：DDP 进程组按 PID 区分，跟同一张卡上跑几个进程无关；只要每个
  进程独立调 `init_process_group`（learner）或完全不调（collector），互不干扰。
  H20 96GB 单卡跑 1 learner + 1 collector 在显存上能站住，但**实际不推荐**混部——
  collector 长 decode 会跟 learner backward 抢 SM，learner 步进会从 ~30 秒涨到 100+ 秒。
- **梯度同步语义**：DDP 自动 mean-reduce 跨 rank 梯度；effective batch = 2（每 rank
  1 条 traj）。两个 rank 抽到不同 traj 反而提高 batch 多样性，不需要 `DistributedSampler`
  做严格不重抽样——off-policy replay 本来就允许重抽。
- **初始权重一致**：rank0 在 build_dataset 通过后立刻把 LoRA 参数广播给 rank1（复用 v3
  `_broadcast_lora_params_from_rank0`），DDP 包装前后都对。
- **采集队列**：collector 可复用 v3 work-stealing 的原子 counter 思路，但 store 必须是
  collector 自己的独立协调通道，或退化成文件锁。counter 跨 epoch 自动 wrap
  （`idx % train_total`）——off-policy 没有 epoch 边界，collector 永远循环抢任务。
- **checkpoint 与 LoRA snapshot 分两套**：
  - `latest_lora/v_{step}/`：给 collector 用的采集策略，只含 adapter，频繁更新（默认
    每 1000 trainer step 一次），保留最近 3 个版本。
  - `checkpoint-{step}/`：恢复训练用，含 LoRA + optimizer + scheduler，稀疏保存（默认
    每 5000 trainer step），保留最近 3 份。
- **`max_steps` / `--check`**：off-policy 没有 epoch 概念，`--max-steps N` 是唯一终止
  判据（trainer 走 N 步退）。`--check` 跑 2 步退出，用于烟雾测试。
- **TB 标量**：learner rank0 写 `train/loss_total`、`train/loss/{a1,a2,s2,a3,s3_status,
  s3_subgoal}`、`train/lr`、`train/grad_norm`、`train/replay/size`、
  `train/replay/avg_age_minutes`、`train/lora_snapshot/version` 等；collector 不写 TB，
  但每条 traj 写一行 `[collect]` log 到 stderr。
- **v3 work-stealing+local-SGD 已彻底废弃**：v4 off-policy replay 解耦了采集与训练，
  learner 只需标准 DDP + frame-count padding，不需要再周期
  `_weighted_average_lora_params_inplace`。
  下面 §9.1.legacy 仅作为历史口径参考，**v4 实现路径不要走那条**。

### 9.1.legacy 历史口径：work-stealing + local-SGD（仅作 v3 兼容入口的参考，不是 v4 的训练路径）

> `sft_v4/train.py` 是 on-policy 兼容入口，仍按这套历史口径跑。**v4 off-policy 生产
> 训练请用 `launch_offpolicy.sh`，不要继续维护 `train.py` 的多卡逻辑。**

- 历史教训：旧 on-policy 的"每帧 ~330 次 DDP forward + 1 次 backward"训练循环跟标准
  DDP+Join 不兼容——episode 帧数差异让各 rank collective 序列严重不一致
  （同一 SeqNum 上有 rank 发 33M-elt grad allreduce、有 rank 发 1-elt Join 探测），
  NCCL 直接 watchdog 超时。因此旧 on-policy 入口才保留 local-SGD 历史口径；off-policy
  `learn.py` 不走这条路径。
- **work-stealing 调度**：所有 rank 加载同一份 `train_ds.rows`，每个 epoch 用同
  seed 重排得到 `epoch_order`；rank0 在 init_process_group 自带的 TCPStore 上重置
  `sft_v4_epoch_<n>_counter=0`，各 rank 通过 `store.wait([counter_key])`
  确认本轮 counter 已写入后，再用 `store.add(key, 1)`
  原子递增抢下一个 `idx`。**谁空闲谁抢，全部 episode 都被训，没有截断**。
- **初始化同步**：local-SGD 不包 DDP，因此模型创建后会先把 rank0 的 trainable
  LoRA 参数广播到所有 rank，保证所有 worker 从同一个 adapter 起点出发。
- **独立 forward/backward/optimizer.step**：每个 rank 各自维护 PEFT 模型副本
  （不包 DDP），各自做 forward + backward + clip + step + scheduler.step。
  无 per-step allreduce，per-rank 速度差异不再造成死锁。
- **周期 LoRA 参数平均（local-SGD）**：参数 `--sync-every-episodes K`
  （默认 16）。K 表示每个 rank 目标处理的 episode 数；每个 epoch 被切成若干个
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
  collective，避免快 rank 在 NCCL work 上空等超过 watchdog timeout。
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
- 保存路径：`AutoMoT/checkpoints/sft_v4_lora/...`。
- 训练命令以 `AutoMoT/` 为 cwd，不写 `AutoMoT/` 前缀。

---

## 10. 已确认的设计决定（讨论闭环）

下面是在 v4 设计讨论里已经定案、不再回滚的决定，留作未来重读时的参照：

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
7. **Memory 初始化（D3v4 / D22 / D27 拍板，覆盖 v3 D3）**：Phase A 初始
   road_structure 按概率 `P_INIT_CORRECT`（默认 0.7）等于 GT 桶；scene 在对应桶内
   联合采样；status=canonical init event；subgoal=该 scene 的第一个事件。理由见
   §3.2：v3 D3 强制 100% 非 GT 让 student 在 Phase A 第一帧就被怼到最难的"翻转"任务，
   收敛慢；0.7 默认让 Phase A 同时承担"翻转"和"保持"监督，并补偿 layer-1 门控触发率。
   `P_INIT_CORRECT=0.0`
   可以退回 v3 行为。**v3 D3 决议作废**，旧的"Phase A 100% 翻转监督、Phase B 100%
   保持监督"分工不再适用。
8. **Step 3 触发条件**：`should_trigger_step3(memory_scene_after_step2, gt_scene)`
   helper（C4 拍板抽出），仅看 step2 后 memory.scene 是否等于 GT，与是否翻转
   过无关。train.py 与 `test_memory_update.py` 共享同一真值函数，禁止有第二份
   隐式实现。
9. **Hindsight Oracle / OPD 蒸馏**：teacher = frozen base + disable_adapter，
   实时 generate 分析 + 离散答案；student token-CE 对齐。
10. **Teacher analysis 清洗**：v4 是分 step 监督学习，analysis 里出现 GT / 选项名 /
    事件名都不影响监督。因此 `build_step{1,2,3}_teacher_target` 只做最小清洗——
    剥掉 `ROAD_STRUCTURE:` / `SCENE:` / `STATUS:` / `SUBGOAL:` 整行（避免脚本追加的
    GT 被 parser 取错）和 `[STEPx]` / `[MEMORY]` 等 prompt marker——其余文字原样
    进 target。prompt 鼓励按 `Scene Description:` / `Critical Object Description:` /
    `Reasoning on Intent:` / `Memory Judgment:` 四行 plain-text 顺序写，但不强制；
    只有剥完后真为空字符串才退回四行 fallback，结构化标签仍由脚本追加。
11. **Teacher 长度**：三步均允许完整分析（max_new=384 仅作异常生成护栏，min_new=0）。
    `repetition_penalty=1.05`（B1 拍板）已在 `_kv_generate_text` 内按 HF 风格
    施加 logits 后处理，与 `do_sample=False` 配合避免重复循环。
12. **Loss 权重**：分析 0.2 × 3、离散 1.0 × 3，分析 per-token normalize；
    分项 TB 记录。
13. **LoRA 接口**：与 v2 完全同构（`--lora-vision-scope` + 全套保险），
    默认 `off`。
14. **多卡训练（D14v4 拍板，覆盖 v3 work-stealing+local-SGD）**：v4 采用 off-policy
    actor-learner——collector 组（默认 2 个进程，分布在 GPU2/GPU3 × 1；稳定后可手动调
    `COLLECTORS_PER_GPU=2/3`）异步 rollout 写
    replay；learner 组（默认 2 个进程，DDP world_size=2，分别在 GPU0/GPU1）从 replay
    随机抽 traj 做 teacher-forced loss + DDP allreduce。optimizer 仍是每 traj 一步；
    backward 按帧 micro-step 执行，并在每条 traj 后手动平均 LoRA grad，不再需要 v3 的
    周期 `_weighted_average_lora_params_inplace`。
    详见 §0 / §9.1。`sft_v4/train.py` 保留为 on-policy 兼容入口，**生产路径用
    `launch_offpolicy.sh`**。
15. **数据持久化（D15v4 拍板）**：v4 必须写训练 trajectory。`build_dataset.py` 仍只产
    episode index；`collect.py` 产 trajectory 文件到 `replay/ready/`，FIFO 容量默认 256。
    Trajectory schema 见 §0.5；图像本身不入库，按路径实时读。trainer 不再现场 generate。
16. **KV cache 共享语义（D1 修订）**：student 仍然在 step1 → step2 → step3 中
    链式复用 KV，因为学生要学习连续 memory 自更新；teacher 不再链式复用 KV。
    teacher 的 step1/step2/step3 都是 fresh dialog：每一步重新吃 4 张图和该步
    privileged context，避免 step1 退化文本污染 step2/step3 分析。teacher 与
    student 也不互相共享 KV，因为 LoRA on/off 的 K/V 数值不同。
17. **Phase B 噪声扰动（D17v4 / D23 扩展）**：Phase B 帧首在
    `force_memory_to_gt_chain` 后，以概率 `PHASE_B_NOISE_PROB`（默认 0.15）将 scene
    改为当前 road_structure 桶内的随机非 GT，status/subgoal 跟着重置。详见 §3.3。
    `noise_injected` 标志写入 trajectory 供事后审计。`PHASE_B_NOISE_PROB=0`
    可退回无扰动行为。
18. **LoRA snapshot 与 collector 同步（D18v4 新增）**：trainer rank0 每
    `SNAPSHOT_EVERY_STEPS`（默认 1000）写一份 `latest_lora/v_{step}/` + 更新
    `current_version.txt`；collector 每 `REFRESH_EVERY_EPS`（默认 4）条 episode 或
    `REFRESH_EVERY_SEC`（默认 60）秒检查 pointer，加载新版本。**保留最近 3 个版本**，
    rank0 不删 ≥ pointer-1 的目录避免 collector 加载竞态。
19. **Replay FIFO 容量（D19v4 新增）**：默认 `REPLAY_CAPACITY=256`，约 750MB 磁盘。
    抽样 `random.choice`，驱逐按 trajectory header `created_at` 取最旧。`REPLAY_CAPACITY=64` 接近 on-policy，
    `REPLAY_CAPACITY=1024` 接近高 staleness off-policy，默认值在 staleness ~1 小时、
    LoRA 漂移 ~200 step 量级，对 r=16 LoRA + LR=3e-5 cosine 来说很温和。
20. **终止协议（D20v4 新增）**：learner 跑到 `--max-steps` 后写 `$OUTPUT_DIR/STOP`
    哨兵，collectors 每条 episode 结束查哨兵存在则退出（不强 kill 让 pending 写完），
    `launch_offpolicy.sh` 监控全部 collector 退出后收尾。
21. **SCENE_CHOICES 分层（D21）**：42 个 scene 按视觉道路结构分成 6 桶
    （JUNCTION / HIGHWAY_MERGE / ROADSIDE_HAZARD / PARKING_AREA / VRU_CROSSING /
    OPEN_ROAD_DYNAMICS）。`ROAD_STRUCTURE_LABELS` 与 `SCENE_TO_ROAD_STRUCTURE`
    放在 `sft_v4/prompts.py`（不动 `prompt_pipeline.py`）。详见 §12.1。
22. **联合 memory 初始化（D22）**：road_structure 与 scene 不解耦——init memory
    必然内部自洽，错 road_structure 时 scene 强制从该错桶选。详见 §12.2。
23. **Phase B 噪声仅扰 scene（D23）**：layer-1 始终保持 = GT（叠加 D27 的 Phase B
    弱纠偏强制拉回）。详见 §12.2 末段。
24. **三步触发链对称（D24）**：`should_trigger_step2 = memory.road_structure(after
    step1) == gt_road_structure`，`should_trigger_step3` 维持原义。step2 的
    SCENE_CHOICES 来自触发后的 `memory.road_structure`；因触发前已要求它等于 GT 桶，
    teacher-forced target 与学生候选表保持一致。详见 §12.4。
25. **L_RS1 权重 = 1.0（D25）**：与 L_SC / L_ST / L_SG 平权，layer-1 是后续触发
    的门。如训练后期 layer-1 已 saturate 而 scene 学不动，再下调到 0.5。
26. **Step 1 输出结构（D26）**：保留视觉描述前缀 + KEEP/CHANGE 论证 +
    一行 `ROAD_STRUCTURE: <name>`；不限制描述字数；loss 三段独立 mask 与
    加权。详见 §12.3 / §12.5。
27. **p_init_correct 默认 0.7（D27）**：联合初始化下补偿 step3 触发率被 layer-1
    拖累的腰斩问题。配套 Phase B 帧首 `force_memory_to_gt_chain` 先拉回 layer-1=GT，
    使 Phase B 阶段 step3 触发率回到与现有 v4 持平。`p_init_correct=0.0`
    保留为退回 v3 行为的对照实验入口。
28. **step1 skip 后下一帧纠偏（D28）**：仅上一帧 `step2_fired=False` 时置位，下一帧
    进入内循环前触发一次。`BELIEVED_ROAD_STRUCTURE` 拉回 GT 桶，`BELIEVED_SCENE`
    默认 GT、按 `SKIP_CORRECTION_SCENE_NOISE_PROB=0.15` 小概率同桶扰动，
    `STATUS/SUBGOAL` 重置为所选 scene 的 init/first subgoal。
29. **放弃严格 4-heading 清洗，老师自由发挥（D29，覆盖 D10 旧"严格四行 fallback"口径）**：
    v4 是分 step 监督学习——L_RS1 / L_SC / L_ST / L_SG 各自只盯一个离散值 token，
    所以 teacher analysis 里出现 GT / chosen label / event name 都不会泄露监督信号。
    `_teacher_structured_analysis_instructions` 不再用 `<...>` 占位词、不再写
    "do not copy these placeholder words" 反向警告，只列 4 行 heading 顺序 + 一句禁
    markdown；`build_step{1,2,3}_teacher_prompt` 不再附 `Keep the analysis about XXX
    only.`；`_clean_teacher_analysis` 简化为只剥两类危险行——`ROAD_STRUCTURE:` /
    `SCENE:` / `STATUS:` / `SUBGOAL:` 整行（防 parser 取错）和
    `[STEPx]` / `[MEMORY]` 等 prompt marker——其余文字（含 markdown / bullets /
    半截选项名 / 单 heading 噪声）原样进 target。`_fallback_teacher_analysis` 保留，
    但只在剥完后真为空字符串时触发。teacher_report 实测显示：本地 Qwen3-VL-4B base
    在该 prompt 下 raw 经常是"单 `Scene Description:` heading + 大段半句子噪声"或纯
    repetition loop；旧严格 4-heading 校验下这些 case 100% 会落 fallback，导致学生学
    不到 base 真实的视觉线索。新口径下 scripted target = raw 全文，监督负担转给
    L_RS1/L_SC/L_ST/L_SG 离散权重，分析段权重 0.2 × 3 维持不变。后续若发现学生被
    噪声带偏，优先调 `DEFAULT_W_ANALYSIS` 而不是回退到严格清洗。

---

## 11. 与现有子包 / 文档的关系

- **不替换 v2**：v2 子包保留作为单帧串行选择题基线；v4 是连续推理的进化版。
- **与 leadmot 关系**：v4 输出仍是离散场景 / 状态 / 子目标，不直接影响 leadmot
  的 route / waypoint head；但训练后的 student 可作为 leadmot prefix 的 Qwen
  backbone（merge_and_unload 后冻结），需要时再补 leadmot 侧的接入文档。
- **与 eval_carla 关系**：v4 student 训出来如果想做闭环，仍走 eval_carla 子包；
  闭环时 memory 用类似 §3 的方式实时维护（初始 memory 用启动时刻的 keyframe
  反查，或仍用随机初始 + 让模型自纠错）。这部分细节落到 eval_carla 侧未来的
  扩展文档，不在本 PLAN 范围内。

---

## 12. 设计（已落地）：SCENE_CHOICES 分层 + Step 1 ROAD_STRUCTURE 选择

> **状态：D21~D29 已拍板并实现。** §12.6 第 5 点的生成端兜底
> （取消强制最少生成 + no_repeat_ngram_size + repetition_penalty 范围）
> 已与分层重构同步落地。

### 12.0 动机

通过 [`inspect_teacher.py`](inspect_teacher.py) 的实测报告（详见
[SFT_V4_RUN.md §7](SFT_V4_RUN.md)），定位到三类问题：

1. **老师 step2/3 输出 degenerate 成 "The left. The right." 循环**——根因是
   完整 42 行 SCENE_CHOICES 把 prompt 推到 ~1500 token，base Qwen3-VL-4B-Instruct
   注意力被稀释；旧版强制最少生成又延长了坏轨迹 → 贪心 argmax 更容易进入复读。
2. **学生 step2 必须在 42 个 scene 里精挑** → 一上来错的概率远大于对的，前期
   `should_trigger_step3` 几乎不命中，status/subgoal 监督密度过低。
3. **小模型一次性吞下 "完整 42 选 1"** 不符合人类决策路径——人类先看"是十字
   路口还是高速合流"，再在那个大类里挑。

### 12.1 分层结构（6 桶定稿）

把 42 个 scene 按**视觉道路结构**分成 6 桶。判定依据：4 张 stitched RGB 里能直接
看到的道路几何 / 交通形态。

| Layer-1 token | 描述（写进 prompt 用） | 包含的 Layer-2 scene |
|---|---|---|
| `JUNCTION` | Intersection / crossroads / T-shape; turn or yield at a meeting of multiple road directions, with or without traffic signal. | BlockedIntersection, CrossJunctionDefectTrafficLight, NonSignalizedJunctionLeftTurn, NonSignalizedJunctionLeftTurnEnterFlow, NonSignalizedJunctionRightTurn, OppositeVehicleRunningRedLight, OppositeVehicleTakingPriority, PriorityAtJunction, RedLightWithoutLeadVehicle, SignalizedJunctionLeftTurn, SignalizedJunctionLeftTurnEnterFlow, SignalizedJunctionRightTurn, T_Junction, VehicleTurningRoute, VehicleTurningRoutePedestrian (15) |
| `HIGHWAY_MERGE` | Multi-lane highway or interurban road; merging into / exiting from a fast lane, or reacting to a cut-in vehicle. | EnterActorFlow, EnterActorFlowV2, HighwayCutIn, HighwayExit, InterurbanActorFlow, InterurbanAdvancedActorFlow, MergerIntoSlowTraffic, MergerIntoSlowTrafficV2, StaticCutIn (9) |
| `ROADSIDE_HAZARD` | Otherwise normal straight / curved road, but a static or quasi-static hazard (accident, construction, parked vehicle, side-lane object) sits adjacent to the lane. | Accident, AccidentTwoWays, ConstructionObstacle, ConstructionObstacleTwoWays, HazardAtSideLane, HazardAtSideLaneTwoWays, ParkedObstacle, ParkedObstacleTwoWays (8) |
| `PARKING_AREA` | Parking lot / parking-area environment; vehicle entering, exiting, or interacting with parked traffic and pedestrians around parked cars. | ParkingCrossingPedestrian, ParkingCutIn, ParkingExit, VehicleOpensDoorTwoWays (4) |
| `VRU_CROSSING` | Pedestrian, cyclist or other dynamic object crossing the lane on a normal road segment. | CrossingBicycleFlow, DynamicObjectCrossing, PedestrianCrossing (3) |
| `OPEN_ROAD_DYNAMICS` | Open road, no special structure; ego stability or lead-vehicle behavior is the dominant cue. | ControlLoss, HardBreakRoute, InvadingTurn (3) |

合计 15 + 9 + 8 + 4 + 3 + 3 = 42 ✓。最大桶 15 项（JUNCTION），prompt 收窄到原来
~36%；最小桶 3 项，prompt 收窄到 ~7%；平均 7 项/桶。

> **D21 拍板：6 桶**。5 桶丢失 ControlLoss / HardBreakRoute / InvadingTurn 与
> 路边异物视觉特征的区分；7 桶在 signalized vs non-signalized 上加复杂度收益不大。

新增两份只读字典，放在 `sft_v4/prompts.py`（不动 `prompt_pipeline.py`，把 v4 专属
分层隔离开）：

```python
ROAD_STRUCTURE_LABELS: Dict[str, str] = {
    "JUNCTION": "Intersection, crossroad, turn, or traffic-light junction",
    "HIGHWAY_MERGE": "High-speed or multi-lane road with merge, exit, or cut-in flow",
    "ROADSIDE_HAZARD": "Normal road with blocked lane, accident, construction, or parked obstacle",
    "PARKING_AREA": "Parking lot, parking exit, door opening, or low-speed parking interaction",
    "VRU_CROSSING": "Pedestrian, cyclist, or small moving actor crossing ego path",
    "OPEN_ROAD_DYNAMICS": "Open road with lead vehicle, braking, control loss, or invading turn",
}

SCENE_TO_ROAD_STRUCTURE: Dict[str, str] = {
    "BlockedIntersection": "JUNCTION",
    ...  # 42 entries, derived from the table above
}
```

`scene_choices_block_for(structure)` 渲染只列该桶 layer-2 的 SCENE_CHOICES。

### 12.2 Memory 字段扩展

`Memory` dataclass 新增 `road_structure: str` 字段；`format_text()` 在
压缩 prompt 后按每层一行渲染：

```
[MEMORY]
BELIEVED_ROAD_STRUCTURE=JUNCTION (Intersection, crossroad, turn, or traffic-light junction)
BELIEVED_SCENE=SignalizedJunctionLeftTurn (Left turn at signalized junction)
BELIEVED_STATUS=initial (Scenario has started; vehicle is approaching the challenge zone.)
BELIEVED_SUBGOAL=junction_approach (Ego vehicle is approaching a junction or intersection.)
EGO_TO_GOAL_XY=(+12.3, -4.5) m
[/MEMORY]
```

`init_memory(p_init_correct=0.7)` 用同一概率联合采样：
- 概率 p：road_structure = GT 桶，scene 在该 GT 桶内按 50% 正确率挑（GT scene 或
  其它同桶 scene 之一）；
- 概率 1-p：road_structure 从其它 5 桶里均匀随机；scene 必须从该错桶内的
  layer-2 candidates 里均匀挑（**memory 内部自洽**——不会出现 "JUNCTION + Accident"
  这种矛盾配置）。

status/subgoal 始终按 scene 的 canonical init + first subgoal 初始化。

> **D22 拍板：联合初始化**。Memory 始终内部自洽，更接近真实推理路径（人不会同时
> 相信"是路口"和"是事故"）。
>
> **D27 拍板：p_init_correct 默认提到 0.7**。配套缓解 step3 触发率被腰斩的问题：
> Phase A 内三层联合命中率从 0.5×0.5=0.25 抬到 0.7×0.5=0.35（layer-1 单独 0.7，
> scene 在 GT 桶内仍是 0.5 错率）；Phase B 弱纠偏后 layer-1 始终 = GT，scene
> 命中率回到当前水平，step3 触发率不再被 layer-1 拖累。`p_init_correct=0.0`
> 仍可退回 v3 行为做对照实验。

`update_memory_after_step1(memory, student_road_structure)`：与
`update_memory_after_step2` 同构。若 layer-1 翻转，连带把 `scene` reset 成
**新桶的第一个 layer-2 候选**（与 scene-change reset 的语义一致），并把
status/subgoal 重置成新 scene 的 init + first subgoal。

`force_memory_to_gt_scene`（Phase B 弱纠偏）改名 / 扩展为
`force_memory_to_gt_chain`，依次拉回 road_structure / scene / status / subgoal，
保留"== GT 时全 noop"语义。

`inject_phase_b_noise` 维持只扰 `scene`，不扰 `road_structure`。噪声候选限定在当前
`memory.road_structure` 桶内的非 GT scene，保证 Phase B 弱纠偏后的 memory 仍内部
自洽；layer-1 由 `force_memory_to_gt_chain` 保持为 GT 桶，scene 错误留给 step2 纠偏。

> **D23 拍板：Phase B 噪声只扰 scene**。layer-1 监督密度由 Phase A 初始
> `p_init_correct=0.7` + Phase B 帧首"force layer-1=GT"机制保证（参见 D27），
> 不靠噪声补。

### 12.3 三步内循环改造

| Step | 旧任务 | v4 定稿任务 |
|---|---|---|
| 1 | 仅描述视觉，禁 memory / 禁标签 | 学生读完整 memory，描述视觉 + 选 ROAD_STRUCTURE；老师开启 fresh dialog，只读 road-only context（`BELIEVED_ROAD_STRUCTURE` / `EGO_TO_GOAL_XY` / `GROUND_TRUTH_ROAD_STRUCTURE`）并输出 4 行结构分析；脚本追加 `ROAD_STRUCTURE: <name>` |
| 2 | 读 memory + 完整 42 行 SCENE_CHOICES → 输出 SCENE | 学生读 memory + 预测 road 桶下 layer-2 候选；老师开启 fresh dialog，读 `GROUND_TRUTH_ROAD_STRUCTURE`、`BELIEVED_SCENE`、`GROUND_TRUTH_SCENE` 并分析 scene keep/change；脚本追加 `SCENE: <gt>` |
| 3 | 不变 | 学生串行读 memory + event options；老师开启 fresh dialog，读 `GROUND_TRUTH_ROAD_STRUCTURE`、`GROUND_TRUTH_SCENE`、believed/GT status-subgoal 并分析 event keep/change；脚本追加 `STATUS/SUBGOAL` |

**Step 1 prompt（学生 / 老师）**：

学生 prompt 结构如下（真实文本以 `prompts.py` 为准）：

```
{memory.format_text()}

[ROAD_STRUCTURE_CHOICES]
- JUNCTION: ...
- HIGHWAY_MERGE: ...
- ROADSIDE_HAZARD: ...
- PARKING_AREA: ...
- VRU_CROSSING: ...
- OPEN_ROAD_DYNAMICS: ...
[/ROAD_STRUCTURE_CHOICES]

[STEP1]
4 images are ordered oldest to newest; the last image is now.
Describe visible road geometry / agents / lighting. Explain whether
BELIEVED_ROAD_STRUCTURE matches what you see. Finally output exactly one
label line by copying one option name verbatim:
ROAD_STRUCTURE: <name>
```

老师 step1 prompt 与学生 prompt 分离：不再喂完整 `[MEMORY]`，只喂
`[STEP1_ROAD_CONTEXT]`，其中包含 `BELIEVED_ROAD_STRUCTURE`、`EGO_TO_GOAL_XY`
和 `GROUND_TRUTH_ROAD_STRUCTURE`，再列出 6 个 road structure 选项。KEEP 时要求
分析哪些道路布局证据支持当前 believed road；CHANGE 时要求先说明 believed road
哪里不符合，再说明真值 road structure 与视觉证据如何吻合。老师只输出严格四行 plain-text 分析：
`Scene Description:`、`Critical Object Description:`、
`Reasoning on Intent:`、`Memory Judgment:`；
标签由 `build_step1_teacher_target` 追加。

**Step 2 prompt（学生 / 老师）**：

学生 prompt 仍然串行接在 step1 学生输出之后，`SCENE_CHOICES` 来自
`memory_after_step1.road_structure`：

```
[SCENE_CHOICES] under BELIEVED_ROAD_STRUCTURE = {memory.road_structure}
- {scene_a}: ...
- {scene_b}: ...
... (该桶 layer-2，平均 7 项)
[/SCENE_CHOICES]
```

老师 step2 不再复用 step1 teacher KV，而是重新吃 4 张图并使用独立 context：

```
[STEP2_SCENE_CONTEXT]
GROUND_TRUTH_ROAD_STRUCTURE=<gt_road_structure> (...)
BELIEVED_SCENE=<memory.scene> (...)
GROUND_TRUTH_SCENE=<gt_scene> (...)
EGO_TO_GOAL_XY=(..., ...) m
[/STEP2_SCENE_CONTEXT]
```

KEEP 时老师分析当前交通场景为何符合 believed scene；CHANGE 时老师先描述当前情况，
再解释 believed scene 哪里不符合、GT scene 哪里符合。输出仍是同一套严格四行 plain-text 结构，
最后一行 `Memory Judgment` 专门判断 scene memory。

**Step 3 prompt（学生 / 老师）**：

学生 prompt 仍然串行接在 step2 学生输出之后，按当前 memory.scene 列出 event options。
老师 step3 重新吃图并使用独立 context：

```
[STEP3_EVENT_CONTEXT]
GROUND_TRUTH_ROAD_STRUCTURE=<gt_road_structure> (...)
GROUND_TRUTH_SCENE=<gt_scene> (...)
BELIEVED_STATUS=<memory.status> (...)
BELIEVED_SUBGOAL=<memory.subgoal> (...)
GROUND_TRUTH_STATUS=<gt_status> (...)
GROUND_TRUTH_SUBGOAL=<gt_subgoal> (...)
EGO_TO_GOAL_XY=(..., ...) m
[/STEP3_EVENT_CONTEXT]
```

KEEP 时老师分析当前 actor flow / ego progress / risk 为什么支持 believed status-subgoal；
CHANGE 时老师解释当前 believed status-subgoal 为什么不对，以及 GT event 为什么符合。
输出仍是同一套严格四行 plain-text 结构，最后一行 `Memory Judgment` 专门判断 status/subgoal memory。

### 12.4 触发链：step1 错 → 跳 step2/3

仿照现有 `should_trigger_step3`：

- `should_trigger_step2(memory_road_structure_after_step1, gt_road_structure) → bool`
  只有 step1 后 `memory.road_structure == gt_road_structure` 才进 step2。
- `should_trigger_step3` 维持原义，但前提是 step2 已经跑过。
- 若 step1 layer-1 翻车 → step2 / step3 全跳过，loss 仍计 step1 的两段
  （ROAD_STRUCTURE 离散 + 分析）。

> **D24 拍板：A**。step2 SCENE_CHOICES 来自触发后的 `memory.road_structure`；由于
> step2 只在 layer-1 命中 GT 桶时触发，teacher-forced target 里的 `SCENE: <gt>`
> 一定在学生看到的候选表里，监督信号始终一致。错 layer-1 直接跳 step2/step3，
> loss 只计 step1 两段（L_A1 + L_RS1）。

### 12.5 损失结构

| Loss 项 | 内容 | 权重 | 备注 |
|---|---|---|---|
| `L_A1` | step1 老师分析 token CE（per-token mean） | 0.2 | 与现有相同 |
| `L_RS1` | step1 `ROAD_STRUCTURE: <name>` value token CE | 1.0 | **新增**，与 L_SC 同权重 |
| `L_A2` | step2 老师分析 token CE | 0.2 | 与现有相同 |
| `L_SC` | step2 `SCENE: <name>` value token CE | 1.0 | 与现有相同 |
| `L_A3` | step3 老师分析 token CE（触发时）| 0.2 | 与现有相同 |
| `L_ST` | step3 `STATUS:` value token CE（触发时）| 1.0 | 与现有相同 |
| `L_SG` | step3 `SUBGOAL:` value token CE（触发时）| 1.0 | 与现有相同 |

总和 = 0.2 × 3 + 1.0 × 4 = 4.6。TB 必须分项记录 `loss/L_RS1` 与
`fire_rate/step2` / `fire_rate/step3` 以便监控 layer-1 命中率与 layer-2 触发率。

> **D25 拍板：1.0**。与 L_SC / L_ST / L_SG 平权。layer-1 是后续触发的门，必须优先
> 学好；如果训练后期 TB 上 `loss/L_RS1` 一直贴地、而 `loss/L_SC` 不下，再考虑
> 降到 0.5 释放权重给 scene 学习。

### 12.6 prompt 简化总原则（针对 degenerate 问题）

无论 D21~D25 怎么定，**所有老师 / 学生 prompt 都按以下原则重写**：

1. **削冗余指令**：删掉重复出现的 "Do not mention 'ground truth' / 'verdict'"——
   塞进 SYSTEM_PROMPT 一次即可，每 step 不再复述。
2. **缩证据清单**：Step1 保留道路结构 / 周围 actor / 信号灯等视觉锚点；
   Step2/Step3 不再重复环境分析，只做候选内 keep/correct 引导。
3. **结构化分析**：三步 teacher 都是 fresh dialog，默认
   `max_new_tokens=384` 仅作异常生成护栏，统一输出严格四行 plain-text 结构：
   整体场景、关键目标、意图关系、memory 判断；不限制每行字数。
4. **拆 user turn**：原本 step2 / step3 各自 1 个 user turn 含 [MEMORY] +
   [CHOICES] + [INSTRUCTIONS]，太长。考虑把 [MEMORY] 留在 turn 头、[CHOICES] +
   [INSTRUCTIONS] 压在 turn 尾，中间留一行空，让模型注意力更易锚定指令。
5. **生成端兜底**（与 prompt 改动**正交**，必须同步动手）：
   - teacher 调用不传 `min_new_tokens`，不强制最少输出；
   - 加 `no_repeat_ngram_size=3`，阻断 "The left. The right." 这种 3-gram 循环；
   - `repetition_penalty` 的 `seen_unique` 只算**本轮生成**的 token，不算 prefix。

§12.6 第 5 点是 [SFT_V4_RUN.md §7.8](SFT_V4_RUN.md) 抽检结论里早就识别出的修法，
现在与分层重构一并写入 PLAN，避免遗漏。

### 12.7 与现有约定的交互

- §3.2 Memory 初始化：默认 `p_init_correct=0.7`，并扩到三层联合采样（D22 / D27）。
- §3.3 Phase B 弱纠偏：扩成"先拉回 layer-1，再 scene，再 event"，对仍 = GT 的
  字段 no-op（与现有 D2 决议一致）。
- §4.5 Step 3 触发：维持现状，但前提是 step2 已经跑（D24）。
- §5 损失权重：扩到 7 项（§12.5）。
- §6.1 Teacher 超参：取消强制最少输出长度、加 no_repeat_ngram_size、调
  repetition_penalty 范围（§12.6.5）。
- §7 数据 / IO：`build_dataset.py` 不需要改 jsonl schema——`gt_road_structure`
  在训练时从 `SCENE_TO_ROAD_STRUCTURE[ep.gt_scene]` 现算即可，不必落盘。
- §10 已确认决策：D21~D29 已并入当前实现，不回滚任何既有决策；与 D8（step3
  触发条件）完全对称扩展。

### 12.8 决策速查表（D21~D29 全部锁定）

| 决策点 | 内容 | 锁定值 |
|---|---|---|
| D21 | 分桶数（5 / 6 / 7） | **6** |
| D22 | road_structure 与 scene 是否联合初始化 | **联合**（自洽 memory）|
| D23 | Phase B 噪声是否扰 road_structure | **不扰**（仅扰 scene）|
| D24 | step2 触发条件 | **A**：layer-1 命中才触发，SCENE_CHOICES 来自触发后的 memory road_structure（此时等于 GT 桶） |
| D25 | L_RS1 权重 | **1.0**（训后视情况下调）|
| D26 | Step 1 是否保留纯视觉描述前缀 | **保留**：视觉环境描述 + verdict 判断 + 标签，不限制描述字数，loss 三段独立计 |
| D27 | step3 触发率腰斩对策 | **p_init_correct 默认 0.7**（联合初始化下三层联合命中 0.7×0.5=0.35，Phase B 弱纠偏后回升）|
| D28 | step1 失败后下一帧是否强制纠偏 | **只在上一帧 step2/3 已跳过时触发一次**；road_structure=GT 桶，scene 大概率 GT / 0.15 同桶扰动，status/subgoal 回 init |
| D29 | teacher analysis 清洗强度 | **只剥 label 行 + prompt marker**；放弃严格 4-heading 校验，scripted target = raw 全文；分 step 监督下 analysis 含 GT/选项名不污染 L_RS1/L_SC/L_ST/L_SG |

### 12.9 实施清单（D21~D29 已落地）

1. `[x] sft_v4/prompts.py`：
   - 新增 `ROAD_STRUCTURE_LABELS`（6 项）+ `SCENE_TO_ROAD_STRUCTURE`（42 项 1:1 映射）；
   - 新增 `road_structure_choices_block()` 与 `scene_choices_block_for(structure)`；
   - `Memory` 加 `road_structure` 字段、`format_text()` 多渲染一行；
   - 新增 `build_step1_user_prompt`（学生）/ `build_step1_teacher_prompt` /
     `build_step1_teacher_target` 三件套；
   - 改 `build_step2_*` 用 `scene_choices_block_for(memory.road_structure)`；
   - 新增 `should_trigger_step2`；扩 `update_memory_after_step1` /
     `force_memory_to_gt_chain`；
   - `init_memory` 改成联合采样，默认 `p_init_correct=0.7`；
   - 新增 `correct_memory_after_step1_skip`，用于上一帧 step1 未过后下一帧帧首纠偏；
   - `_kv_generate_text` 加 `no_repeat_ngram_size=3` 选项，teacher 不传强制最少生成参数，
     `seen_unique` 只算本轮
     生成 token。
2. `[x] sft_v4/train.py` / `sft_v4/collect.py`：
   - 三步循环 → 仍是三步，但 step1 多生成 ROAD_STRUCTURE 标签 + memory 更新；
   - `should_trigger_step2` / `should_trigger_step3` 链式判断；
   - loss 加 `L_A1` / `L_RS1`（step1 分析 + 标签），其余 L_A2/L_SC/L_A3/L_ST/L_SG
     维持，但 `L_A2`/`L_SC` 仅在 step2 触发时计，`L_A3`/`L_ST`/`L_SG` 仅在 step3
     触发时计；
   - TB scalars 增加 `loss/L_A1` / `loss/L_RS1`、`fire_rate/step2` /
     `fire_rate/step3`、`accuracy/road_structure`、`skip_correction_rate` 等监控项。
3. `[x] sft_v4/replay.py`：trajectory schema bump 到 `sft_v4_rollout_v2`，旧
   `sft_v4_rollout_v1` 文件会被 `validate_trajectory` 拒收，并由 learner 移到
   `failed/`。
4. `[x] sft_v4/eval.py` / `sft_v4/probe.py`：三步同步加 ROAD_STRUCTURE 解析与统计；
   eval 报告增加 `road_structure_acc` / `step2_fire_rate` / `step3_fire_rate`。
5. `[x] sft_v4/inspect_teacher.py`：已覆盖 5 种 memory 模式
   （`all_keep` / `rs_change` / `scene_change_same_rs` / `event_change` /
   `scene_change_cross_rs`），覆盖完整状态机分支。
6. `[x] sft_v4/test_memory_update.py` / `test_gt_leak_filter.py` / `check_loss_mask.py`：新增
   road_structure 字段测试 + 联合初始化测试；`test_gt_leak_filter.py` 仅作为 legacy no-op 兼容测试；
   `test_memory_update.py` 额外覆盖
   replay schema v2 的 step2 门控契约（step2-skip 合法、step2-fired 必须带
   `memory_after_step1`）。
7. `[x] SFT_V4_RUN.md` §1 / §3 / §7 同步更新 memory 字段渲染示例、
   `--p-init-correct` 默认值与 inspect_teacher 5 模式文档。
