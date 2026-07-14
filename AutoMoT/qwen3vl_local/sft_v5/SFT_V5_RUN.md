# SFT v5 Runbook

SFT v5 是 RS / EVENT 两问串行的 OPSD 训练入口。数据来自
`keyframe_filter/collection_output/*_result.json`，一条样本是一条 route sequence；
训练时 student 自己维护 `RS + EVENT` memory，privileged teacher 只用来在 student
自由生成 token 上提供 forward-KL logits。

本文默认当前目录是 `AutoMoT/`。

## 1. 构建数据

```bash
python qwen3vl_local/sft_v5/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_v5_data \
  --val-ratio 0.1 \
  --seed 42
```

快速 smoke：

```bash
python qwen3vl_local/sft_v5/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir /tmp/sft_v5_smoke \
  --scenarios Accident \
  --max-routes 1 \
  --max-frames-per-route 3
```

构建阶段会跳过 `noScenarios_result.json`、异常时长 skip、数据缺失 skip、
`status != success`、缺 XML、缺 RGB、缺 meta 或缺逐帧 annotation 的 route。
`review_required=true` 仍保留训练。

Q2 候选优先使用逐帧 `frame_event_annotation.allowed_events`，旧结果缺字段时才
fallback 到 `scenario_event_candidates ∩ EVENT_CANDIDATES_BY_RS[current_rs]`。
所有 `R-E*` 会在 prompt 中折叠为一个 `RE`，原始 regular code 只保存在
`event_code` / `regular_event_codes` 用于审计。

## 2. 静态检查

```bash
python qwen3vl_local/sft_v5/check_loss_mask.py
python qwen3vl_local/sft_v5/test_memory_update.py
python qwen3vl_local/sft_v5/test_dataset_contract.py
python qwen3vl_local/sft_v5/test_streaming_optimizer.py
python qwen3vl_local/sft_v5/test_parallel_kl_microbatch.py
python -m py_compile qwen3vl_local/sft_v5/*.py
```

检查 dataset / padding / prompt / memory，不加载模型：

```bash
python qwen3vl_local/sft_v5/train.py \
  --train-index checkpoints/sft_v5_data/train_sequence_index.jsonl \
  --output-dir /tmp/sft_v5_train_check \
  --check
```

## 3. 训练

### 3.0 OPSD 采样与训练关系

OPSD 不是先离线生成一批 teacher 数据再训练。v5 当前实现是同步 on-policy：

1. 每个 DDP rank 读取自己的 route sequence batch。
2. 对每个有效 frame，当前 student 先自由生成 Q1；Q1 RS 正确时再自由生成 Q2。
3. 这些 student 生成出来的 token 就是本 step 的 on-policy 采样数据，只在内存中临时保存。
4. Teacher 关闭 LoRA，读取 privileged prompt，在同一批 student token 上 forward 得到 logits。
5. Student/teacher logits 只裁剪到需要监督的 span token，做 weighted forward-KL。
6. 当前 rank 每帧算完立即 backward，只累计 LoRA 梯度，不保留跨帧计算图。
7. 默认累计 512 个 global 有效 frame，或最迟 32 个 global timestep，在完整 timestep
   边界手动 all-reduce LoRA 梯度并执行 optimizer step；各 route memory 原样继续。

因此 H20 四卡下，当前 v5 是“四张卡都边采样边训练”，不是“几张卡专门采数据、几张卡专门训练”。
实现上使用 torchrun 多进程和手动梯度 all-reduce，不把模型包进
`DistributedDataParallel` wrapper：Q2 是否触发取决于各 rank 的 Q1 student 输出，
forward 次数会不一致，DDP wrapper 的 forward hook/collective 容易出现 NCCL
watchdog 卡死。
这和 v4 的 off-policy collector/learner 不同。若要改成异步架构，可以参考 v4：
`3` 张 H20 跑 collector 持续写 replay，`1` 张 H20 跑 learner 训练；但那需要新增 v5
collector/replay/learner 代码，当前 `train.py` 没有这个分卡角色。

### Launcher 模式

`train.sh` 支持 `single/ddp/check` 三种模式，并会维护
`OUTPUT_DIR/run_<RUN_TAG>/` 与 `OUTPUT_DIR/latest`。四卡 H20 正式训练直接用：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp
```

默认就是 `BATCH_PROFILE=max_util`：

```text
NPROC=4
BATCH_PROFILE=max_util
PER_DEVICE_BATCH_SIZE=8
QWEN_BATCH_SIZE=8
PARALLEL_KL_MICROBATCH_SIZE=4
GRAD_ACCUM=1
UPDATE_MODE=streaming_frames
TARGET_GLOBAL_FRAMES_PER_STEP=512
MAX_TIMESTEPS_PER_STEP=32
LEARNING_RATE=1e-5
```

只想确认 launcher 会解析成 8 路、不加载模型时：

```bash
DRY_RUN=1 GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp
```

8 路不稳时按顺序回退：

```bash
BATCH_PROFILE=balanced GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp  # 6 路
BATCH_PROFILE=debug GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp     # 4 路
```

只用于复现旧实验的整批 route 更新口径：

```bash
UPDATE_MODE=batch LEARNING_RATE=3e-5 GPU_IDS=0,1,2,3 \
bash qwen3vl_local/sft_v5/train.sh ddp
```

正式训练默认使用 `streaming_frames`；旧 `batch` 模式可能一次累计上万帧，optimizer
step 和正式 loss 曲线都会延迟数小时。

单卡调试和静态检查：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_v5/train.sh single
GPU_IDS=0 bash qwen3vl_local/sft_v5/train.sh check
```

启动后第一条 `[batch-start]` 应显示 `routes=8 / qwen_batch=8`。由于 Qwen
自回归 decode、CPU/IO 和 rank 间阶段差异会让 `nvidia-smi` 瞬时值波动，文档不承诺
每一秒固定 100%，但 `max_util` 是当前同步 on-policy v5 里最接近满载的口径。

`PARALLEL_KL` 默认开启。Q1/Q2 student rollout 仍按 8 路并行，但有 autograd graph、
显存更重的 teacher/student KL 默认按 `PARALLEL_KL_MICROBATCH_SIZE=4` 拆成 `4+4`，
每个微批立即 backward。若某个 4 路 KL forward 因更长上下文 OOM，代码会自动二分
为 `2+2`，不会降低 token 上限或重新采样 student 输出。

Qwen 输出不做 `ABNORMAL:` / `EVENT:` 字段早停；为了防止模型不出 EOS 时无限生成，
launcher 只保留一个很大的安全上限，默认：

```bash
MAX_NEW_TOKENS_Q1=1024
MAX_NEW_TOKENS_Q2=1024
```

正式训练不要再用很小的 token cap；只有排查 OOM 或做极短 smoke 时才临时调低。
因此这里不是“完全无限输出”，而是“无字段早停 + EOS / `<|im_end|>` 自然停止 +
1024 token 安全上限”。

`GPU_IDS` 非空时优先使用显式卡号；否则可用 `DDP_GPU_COUNT=4` 自动选 4 张空闲卡。
通常不要绕过 `train.sh` 手写 `torchrun`；launcher 会统一处理 GPU、端口、日志和
`run_<RUN_TAG>/latest` 防覆盖目录。视觉 LoRA 默认关闭，需要时再显式设置
`LORA_VISION_SCOPE`。

关键训练口径：

- DataLoader collate 只做本 rank local padding；主训练进程再 all-reduce 得到
  当前 batch 的 global `max_T`，padding timestep 不读图、不进 Qwen、不产 loss。
- 多卡训练使用 torchrun + 默认 `LengthBalancedDistributedSampler` + 手动 LoRA 梯度 all-reduce；
  不使用 `DistributedDataParallel(model)` wrapper，避免动态 Q2 分支造成 rank 间
  forward collective 不匹配。
- `SAMPLER_MODE=length_balanced` 会在每个 rank route 数一致的前提下按 route frame
  数均衡分片，减少长 route rank 拖住其它 rank；`SAMPLER_MODE=distributed` 可切回
  PyTorch 原生 route 数均衡分片做对照。
- 训练循环不再把整条 route sequence 的 Qwen 计算图攒到 batch 末尾；每帧
  OPSD loss 立刻 backward，只累计 LoRA 梯度，降低 H20 上长序列 OOM 风险。
- rollout batch 与 KL 训练微批解耦：默认保持 8 路 Q1/Q2 生成吞吐，Q1/Q2 KL 按
  4 路微批构图并立即 backward；forward OOM 时只二分当前 KL 微批。
- 默认 `UPDATE_MODE=streaming_frames`：四卡累计 512 个实际 global frame，或达到
  32 个 global timestep 后立即同步更新，不等待完整 DataLoader route batch。
- 更新只发生在完整 timestep 结束后，绝不会夹在同一帧 Q1/Q2/KL 中间；optimizer
  step 后保留各 route 的离散 RS/EVENT memory，下一帧使用更新后的 student。
- 每个 frame loss 先除以 effective target frame 数再 backward；同步时梯度跨 rank
  求 SUM，并按窗口实际 global frame 数修正，最终严格等价于全局逐帧平均。
- LoRA 梯度按 device/dtype 合并成约 64 MiB 的连续 bucket 后再 all-reduce，避免数百个
  小参数逐个发起 NCCL collective；这只减少同步开销，不改变梯度数值。
- 每个 checkpoint/final 的 `sft_v5_adapter_config.json` 同时记录原始阈值、经过
  `GRAD_ACCUM` 放大后的 effective 阈值、LR 和梯度同步策略，便于复现实验。
- `train.sh` 默认 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，减少
  KV/logits 张量反复申请释放带来的 CUDA allocator 碎片化。
- 流式模式中 `GRAD_ACCUM` 是窗口倍率：例如 2 表示 `1024 frame / 64 timestep`；
  默认保持 1。`UPDATE_MODE=batch` 时才恢复“累计若干 DataLoader batch”的旧语义。
- epoch 末尾不足目标 frame/timestep 的窗口也会 flush，不会静默丢弃尾部梯度。
- v5 默认关闭 gradient checkpointing，因为 Qwen3-VL KV cache 续接必须保持
  `use_cache=True`；`--grad-checkpoint` 仅保留为实验开关。
- Q1 输出 `Scene Description / Critical Object Description / Reasoning on Intent / RS / ABNORMAL`；
  天气、道路、车道线、交通灯和周围运动都压缩写进 `Scene Description`，没有单独天气分类 loss。
- System prompt 明确提醒关注交通灯/标志、周围车辆/行人/障碍物、车道线/道路结构、
  以及影响自车决策的关键因素，但仍保持短句。
- Q1 使用 road-only `MEMORY`：只含自然语言 `BELIEVED_RS` 和
  `EGO_TO_GOAL_XY=(+x, +y) m`，不提前暴露 `BELIEVED_EVENT`。
- Q2 才使用 road + event `MEMORY`；memory 文本只写自然语言描述，不写 A-E 选项字母
  或 `RE/U-E*` 标签代码。`EGO_TO_GOAL_XY` 由 build_dataset 从当前帧 meta
  `next_target_points[-1]` 转 ego frame 写入；缺失该坐标的 route 不进入新数据集。
  如果旧 probe 里仍看到 `EGO_TO_GOAL_XY=UNKNOWN`，说明它来自旧 sequence index；
  当前 `RouteSequenceDataset` 会跳过缺坐标 frame，需要先重跑 build_dataset 再重跑 probe。
- Student 不看 XML weather；XML weather 只进入 teacher prompt，并且 teacher target
  清洗成学生视角。若 XML weather 与 RGB 可见天气或能见度冲突，teacher 以 RGB
  证据为准。
- Q2 是 Q1 assistant 输出后的第二轮 user turn，训练、eval 和 probe 的模型路径都复用
  Q1 KV cache，不重新问一轮 fresh Q2。
- Q1 RS 错误时只结束当前帧：跳过本帧 Q2，下一有效帧恢复 `GT RS + RE`。
- Q2 非法 option 不污染 memory，下一有效帧同样恢复 `GT RS + RE`。
- Q2 候选字母每帧可复现随机，不能假设某个字母固定代表 `RE`。

## 4. TensorBoard

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_v5_runs/latest/tb
```

当前 v5 训练脚本至少写：

- `run/alive` / `run/world_size` / `run/per_device_batch_size` /
  `run/qwen_batch_size` / `run/update_mode_streaming` /
  `run/target_global_frames_per_step` / `run/max_timesteps_per_step`：writer 创建后立即
  写入，用来确认 events 文件不是空壳，并核对流式更新配置。
- `progress/*`：首个 optimizer step 之前的轻量心跳，记录当前 batch/frame/sync
  进度、optimizer window 的 local/global frame、timestep 和 CUDA 显存；横轴是
  rank0 已处理的本地 frame 数。
- `train/global_frames_per_step` / `train/timesteps_per_step`：每次流式更新的实际窗口；
  常规阶段应接近 512 frame，低吞吐尾段可能由 32 timestep 上限提前触发。
- `train/update_reason_code`：`1=target_frames`、`2=max_timesteps`、`3=batch/epoch_flush`。
- `train/learning_rate` / `time/grad_sync_seconds` / `time/optimizer_step_seconds`：当前
  scheduler LR、LoRA 梯度同步和 AdamW 更新耗时。
- `ddp/grad_allreduce_buckets`：每次同步的连续梯度 bucket 数，用于确认没有退回逐参数
  collective。
- `train/loss_frame`：当前 logging window 内所有 rank 聚合后的 frame 平均 loss。
- `train/loss/q1_analysis` / `train/loss/q1_rs` / `train/loss/q1_abnormal`：
  Q1 OPSD KL 的分项 loss，按有效 frame 平均。
- `train/loss/q2_analysis` / `train/loss/q2_event`：Q2 OPSD KL 的分项 loss，按实际进入
  Q2 的 frame 平均；如果窗口内没有 Q2，会用 1 作分母避免 NaN。
- `train/q2_trigger_rate`：Q1 RS 正确后进入 Q2 的比例，也就是第二问实际采样率。
- `train/q1_rs_acc_window` / `train/q1_abnormal_acc_window`：当前窗口的 Q1 解析正确率。
- `train/q2_event_acc_window`：进入 Q2 的帧中，EVENT 解析是否命中动态真值。
- `train/q2_invalid_output` / `train/reset_next`：非法输出和下一帧 reset 次数。
- `train/rollout_tokens_per_frame`：student on-policy 采样出的 Q1+Q2 token 平均长度。
- `train/q1_token_cap_hit_rate` / `train/q2_token_cap_hit_rate`：Q1/Q2 生成是否打满
  `MAX_NEW_TOKENS_Q1/Q2`。如果长期非 0，说明模型经常没有自然输出 EOS / `<|im_end|>`，
  1024 安全上限正在截断输出，parallel KL 也会更吃显存。
- `qwen/q1_batched_frame_rate`：所有已训练 Q1 frame 中，真正进入 size>=2 batched rollout
  的比例；这是判断 `QWEN_BATCH_SIZE>1` 是否真有收益的主指标。
- `qwen/q1_grouped_frame_rate`：所有已训练 Q1 frame 中，进入 grouped 路径的比例；
  `QWEN_BATCH_SIZE=1` 或尾部单 frame chunk 不进入这个分母。
- `qwen/q1_batched_frame_rate_grouped`：只在 grouped 路径内部计算的 batched frame 比例；
  它不能代表全训练 frame 的真实 batch 比例。
- `qwen/q1_batched_groups` / `qwen/q1_singleton_groups`：真实 batched group 和 singleton
  group 数。
- `qwen/q1_length_seconds_per_chunk`：为了记录 padding pressure 而计算 processor input
  length 的平均耗时；Q1 现在允许 mixed-length padded rollout，这个指标主要用于审计
  prompt 长度差异。
- `parallel_kl/frame_rate`：当前 logging window 内有多少 frame 走了 chunk 级 batched
  teacher/student KL。
- `parallel_kl/seconds_per_chunk`：chunk 级 KL scoring + backward 的平均耗时。
- `parallel_kl/microbatches_per_chunk` / `parallel_kl/frames_per_microbatch`：8 路 rollout
  chunk 实际拆成多少个 KL 微批，以及每个微批的平均 frame 数；默认应接近 `2 / 4`。
- `parallel_kl/oom_splits`：KL forward 因上下文过长触发的安全二分次数；偶发非 0 可继续
  训练，若长期非 0，直接把 `PARALLEL_KL_MICROBATCH_SIZE` 从 4 调成 2。
- `parallel_kl/fallbacks`：旧兼容统计。微批开始 backward 后不允许整块 fallback，避免
  已完成微批被重复累计；普通异常会明确中止并打印 traceback 开关。
- `ddp/padding_rate`：global padding 后的 None frame 占位比例。
- `ddp/max_T_global_avg`：logging window 内多进程对齐后的平均 `max_T_global`。

stdout / `log.txt` 还会写 rank0 心跳，避免长时间看不到训练状态：

- `[sampler]`：当前 epoch sampler 的 rank 间 frame 数均衡情况；重点看
  `min_rank_frames/max_rank_frames` 是否差距很大。
- `[batch-start]`：当前 batch 的 route 数、local/global padding 长度、有效 frame 数。
- `[frame-start]` / `[frame-done]`：当前 rank0 正在处理的 route/frame、memory、耗时、
  当前 batch 内 frame 进度、loss、Q1/Q2 rollout token、是否进入 Q2、是否 reset、
  CUDA 显存。
- `[batch-local-done]` / `[batch-global-done]`：本 rank 和已逐 timestep 汇总的 global
  frame 数；流式模式不会等到这里才做 optimizer step。
- `[sync-start]` / `[sync-done]`：optimizer step 前的 LoRA 梯度 all-reduce 是否开始/结束。

v5 默认每 512 global frame 或 32 timestep 更新一次，因此不用再等完整长 route batch
才看到 loss。`LOGGING_STEPS=1` 会在每次流式 optimizer step 后输出聚合 `[train]`
指标。默认 rank0 前 3 个 frame 都打印，之后按 `PROGRESS_FRAMES` 输出；单个长操作
超过 `HEARTBEAT_SECONDS=120` 秒也会补心跳。排查卡顿时建议：

```bash
PROGRESS_FRAMES=1 HEARTBEAT_SECONDS=60 GPU_IDS=0,1,2,3 \
bash qwen3vl_local/sft_v5/train.sh ddp
```

如果最后一条日志停在 `[frame-start]`，说明某个单帧 Qwen OPSD 很慢或卡在图像/生成；
停在 `[batch-local-done]` 则优先查 rank 间是否有某个进程落后；停在 `[sync-start]`
则优先查 LoRA 梯度 all-reduce / NCCL。

如果 TensorBoard 仍提示 `No dashboards are active`，先确认这个 run 是新代码产生的。
新 run 的 events 文件即使还没到第一条 `[train]`，也应该能看到 `run/*` 或
`progress/*` tags；只有旧 run 或 writer 尚未创建时，events 才可能长期只有 88B
header。新默认下第一组 `train/loss/*` 应在首个 512-frame/32-timestep 窗口后出现，
不再等待 `max_T_global` 全部跑完。

### 4.1 真正并行 Qwen 的阶段开关

`QWEN_BATCH_SIZE` 会在同一个 rank、同一个 timestep 内，把多条 route 的 Q1/Q2 student rollout 合成 padded Qwen batch。默认四卡命令已经是 8 路；跑对时重点看：

- `[batch-start] ... routes=8 ... qwen_batch=8`
- `[q1-grouped] ... batched_frames=8`
- `[chunk-train] ... parallel_kl=1 kl_microbatches=[4, 4] kl_oom_splits=0`

训练前只保留一个真实 batched smoke 命令：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/test_batched_qwen_smoke.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --num-cases 2 \
  --candidate-pool 256 \
  --require-batched-group \
  --no-prefer-different-lengths \
  --check-parallel-kl \
  --output-json checkpoints/sft_v5_runs/batched_qwen_smoke_require_batch.json
```

如果频繁看到 `[warn] q1 batch fallback ...`，说明 rollout 回退到了旧逐帧路径；需要时
打开 `Q1_BATCH_TRACEBACK=1` / `Q2_BATCH_TRACEBACK=1`。若看到
`[warn] parallel KL adaptive split ...`，说明 rollout 仍是 8 路，只是 KL scoring 为了
显存安全继续拆小。时间维 `t -> t+1` 不能并行，因为 memory 依赖上一帧学生输出。

不要在训练阶段做“结构字段早停”：不能因为已经生成到 `ABNORMAL:` 或 `EVENT:` 就
提前截断 student rollout。v5 的 OPSD loss 需要完整的学生分析 token 和离散答案
token 共同接受 teacher logits 监督；训练时强行按字段早停会把学生推向短答案/少分析，
破坏 CoT 分布。若要提速，应优先调小数据规模、优化 prompt 简洁度、降低 token 上限
并用 probe 检查是否截断，而不是改变训练 rollout 的停止规则。

当前已记录 `train/loss/{q1_analysis,q1_rs,q1_abnormal,q2_analysis,q2_event}`。
后续如果扩展视觉 LoRA 保险，可继续沿用 v3/v4 的
`grad_norm/{language,vision}` 和 `param_norm/lora_{language,vision}` 命名。

## 5. Eval

默认只跑学生，不加载 teacher，不做 GT 注入；memory 全程由学生 Q1/Q2 输出维护。
Q1 RS 错时跳过本帧 Q2，下一有效帧恢复 `GT RS + RE`，与训练采样口径一致。

自由生成评估：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/eval.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_v5_runs/latest/final \
  --output-json checkpoints/sft_v5_runs/latest/eval_metrics.json
```

只检查 index 可读：

```bash
python qwen3vl_local/sft_v5/eval.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --check
```

输出：

- `eval_metrics.json`

核心指标：

- `rs_acc`
- `abnormal_acc`
- `event_acc_when_rs_correct`
- `ue_acc`
- `re_acc`
- `q2_trigger_rate`
- `q2_candidate_mismatch`
- `q2_invalid_output`
- `rs_wrong_resets`

## 6. Probe / 可视化输入输出

### 6.1 方法速查

| 目的 | 命令 | 是否加载模型 | 主要产物 |
|---|---|---|---|
| 训练前 base Qwen OPSD 能力体检 | `GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py --index checkpoints/sft_v5_data/val_sequence_index.jsonl --model-dir checkpoints/Qwen3-VL-4B-Instruct --output-dir checkpoints/sft_v5_runs/pre_opsd_base_probe --num-cases 8 --with-model --with-teacher-model --with-teacher` | 是，纯默认/base Qwen，不传 `--adapter-dir`，不加载任何 LoRA | RGB 副本、system/user/messages、student output、teacher target/output、memory、flags、timeline |
| 训练前 grouped/parallel Qwen 等价性体检 | `GPU_IDS=0 python qwen3vl_local/sft_v5/test_batched_qwen_smoke.py --index checkpoints/sft_v5_data/val_sequence_index.jsonl --model-dir checkpoints/Qwen3-VL-4B-Instruct --num-cases 2 --output-json checkpoints/sft_v5_runs/batched_qwen_smoke.json`；强制真实 batched rollout 和 parallel KL 时加 `--candidate-pool 256 --require-batched-group --no-prefer-different-lengths --check-parallel-kl` | 是，纯默认/base Qwen，不传 `--adapter-dir` | single-vs-grouped Q1/Q2 文本、token、训练 logits diff、parallel-KL-vs-逐帧-KL 总 loss / case loss / parts diff、actual group sizes、input length / padding pressure |
| 训练后 adapter 学生可视化 | `GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py --index checkpoints/sft_v5_data/val_sequence_index.jsonl --model-dir checkpoints/Qwen3-VL-4B-Instruct --adapter-dir checkpoints/sft_v5_runs/latest/final --output-dir checkpoints/sft_v5_runs/latest/probe_with_adapter --num-cases 8 --with-model --with-teacher` | 是，只加载 student adapter | 上述静态产物 + `q1_student_output.txt` / `q2_student_output.txt` |
| 静态检查教师/学生输入合同 | `python qwen3vl_local/sft_v5/probe.py --index checkpoints/sft_v5_data/val_sequence_index.jsonl --output-dir checkpoints/sft_v5_runs/latest/probe_static --num-cases 24 --with-teacher` | 否 | RGB 副本、student prompt、teacher prompt、teacher target、memory、flags、timeline |
| 检查 teacher 合同 | `python qwen3vl_local/sft_v5/inspect_teacher.py --index checkpoints/sft_v5_data/train_sequence_index.jsonl --output-dir checkpoints/sft_v5_runs/latest/teacher_inspect --num-cases 64` | 否 | `teacher_report.json` / `teacher_report.md` |

同一份速查记录也保存在 `qwen3vl_local/sft_v5/SFT_V5_VISUALIZATION_RECORD.md`，
方便在 v5 子包内直接回看。

### 6.2 Probe 入口

`probe.py` 是 v5 的主要可视化/审计入口。它有三种用途，必须分开理解：

- 训练前 base Qwen OPSD 体检：`--with-model --with-teacher-model`，不传
  `--adapter-dir`，不加载任何 LoRA，用默认 Qwen 分别跑 student prompt 和
  privileged teacher prompt，判断模型基础能力与 prompt 合同是否足够支撑 OPSD。
- 训练前 grouped Qwen 等价性体检：运行 `test_batched_qwen_smoke.py`，不传
  `--adapter-dir`，用默认 Qwen 检查 `QWEN_BATCH_SIZE>1` 的 Q1 grouped rollout 是否
  和单样本路径在 Q1/Q2 文本、token 以及训练 logits 上一致。
- 训练后 adapter 学生可视化：`--with-model --adapter-dir ...`，只看训练出的学生在
  真实状态机下的 Q1/Q2 输出。
- 静态 prompt / target 快检：不加载模型，只 dump RGB、student prompt、teacher prompt、
  脚本化 teacher target、label、memory 和 timeline。

训练前 base Qwen OPSD 能力体检：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/sft_v5_runs/pre_opsd_base_probe \
  --num-cases 8 \
  --with-model \
  --with-teacher-model \
  --with-teacher
```

这个命令不传 `--adapter-dir`，所以 student 和 teacher 都是默认/base Qwen；
区别是 student 只看 RGB + 学生 prompt，teacher 看 RGB + privileged teacher prompt。
这一类训练前体检不要加载任何 adapter，否则就不是在测试普通 Qwen 是否足够支撑
OPSD。
如果同卡同时加载两份 Qwen 显存不够，可以拆成 student-only 的 `--with-model` 和
teacher-only 的 `--with-teacher-model` 两次跑。

训练后 adapter 学生可视化：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_v5_runs/latest/final \
  --output-dir checkpoints/sft_v5_runs/latest/probe_with_adapter \
  --num-cases 8 \
  --with-model \
  --with-teacher
```

静态 prompt / target 快检：

```bash
python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --output-dir checkpoints/sft_v5_runs/latest/probe_static \
  --num-cases 24 \
  --with-teacher
```

`--with-teacher` 是 v3 兼容参数；v5 始终写 teacher privileged prompt 和脚本化
teacher target。只有显式加 `--with-teacher-model` 时，才会额外加载 base Qwen 并生成
`q1_teacher_output.txt` / `q2_teacher_output.txt`。

输出结构仿 v3 probe：

- 顶层 `manifest.json`：列出每条 route dump 的目录、scenario、route_id 和帧数。
- 每条 route 一个 `route_<idx>__<scenario>__<route_id>/` 目录。
- 每条 route 下有 `timeline.json` 和 `timeline.png`，红点表示 RS 错/reset，蓝点表示
  进入 Q2，绿点表示无模型的 teacher-forced 静态 dump。
- 每帧一个 `frame_<frame_id>/` 目录。

每个 `frame_*` 目录会保存：

- `rgb_00.jpg` / `rgb_01.jpg` / `rgb_02.jpg` / `rgb_03.jpg`：真实进入 Qwen 的 4 帧
  stitched RGB history 副本。
- `rgb_paths.json`：RGB 原路径与复制状态。

- `q1_system_prompt.txt` / `q2_system_prompt.txt`：v5 固定 system prompt。
- `q1_student_user_prompt.txt` / `q1_teacher_user_prompt.txt`：Q1 的 student / teacher
  user prompt；`q1_student_prompt.txt` / `q1_teacher_prompt.txt` 仍作为兼容别名保留。
- `q2_student_user_prompt.txt` / `q2_teacher_user_prompt.txt`：Q2 的 student / teacher
  user prompt；`q2_student_prompt.txt` / `q2_teacher_prompt.txt` 仍作为兼容别名保留。
- `q1_student_messages.json` / `q1_teacher_messages.json` / `q2_student_messages.json` /
  `q2_teacher_messages.json`：可序列化的 system + user messages，图片用 `rgb_*.jpg`
  文件名和原路径表示，方便直接检查 role 分界。
- `q1_student_prompt.txt`：Q1 学生真实输入，不含 XML weather / GT。
- `q1_teacher_prompt.txt`：Q1 privileged teacher 输入，含 XML weather、GT RS、GT abnormal、
  原始 `event_code`。
- `q1_teacher_target.txt`：脚本化学生视角 target，用于审计合同和 loss mask。
- `q1_teacher_output.txt`：只有 `--with-teacher-model` 时非空，用于训练前检查 base
  teacher。它应当和 student 一样从 `Scene Description:` 开始输出分析；如果复读
  `[MEMORY]` / `[RS_CHOICES]` / `[REFERENCE]`，说明是旧 demo 或 prompt 合同未生效，
  需要重跑 probe。
- `q2_student_prompt.txt`：Q2 学生真实输入，含逐帧随机 `EVENT_CHOICES`；`RE` 会展开
  当前帧 `regular_event_codes` 的自然语言含义。
- `q2_teacher_prompt.txt`：Q2 privileged teacher 输入，含 answer event option 与
  `event_code` 审计字段。
- `q2_teacher_target.txt`：脚本化 Q2 target。
- `q2_teacher_output.txt`：只有 `--with-teacher-model` 时非空，用于训练前检查 base
  teacher，也应从 `Scene Description:` 开始并最终输出 `EVENT:`。
- `q1_student_output.txt` / `q2_student_output.txt`：目录结构固定；只有 `--with-model` 时内容非空。
- `step1_user.txt` / `step1_student.txt` / `step1_teacher_user.txt` / `step1_teacher.txt`：
  v3 风格别名，对应 Q1。
- `step2_user.txt` / `step2_student.txt` / `step2_teacher_user.txt` / `step2_teacher.txt`：
  v3 风格别名，对应 Q2。
- `memory_before.json` / `memory_after.json`：该帧前后的内部 `RS + EVENT + EGO_TO_GOAL_XY`
  memory。实际 Q1 user prompt 只渲染 road-only memory，Q2 user prompt 才渲染 event。
- `flags.json`：解析出的 student/teacher 输出、是否 RS 正确、是否进入 Q2、
  是否 candidate mismatch、是否 reset 下一帧，以及
  `q2_student_continued_from_q1_kv` / `q2_teacher_continued_from_q1_kv` 等诊断字段。
- `labels.json`：`history_rgb_paths`、`rs_label/rs_option`、`event_label/event_code`、
  `abnormal`、`event_option_map`、`frame_allowed_events_raw`、`regular_event_codes`、
  `event_candidate_codes`、`ego_to_goal_xy` 与 `weather_text_teacher_only`。

### 6.3 Teacher 合同抽检

Teacher 合同抽检：

```bash
python qwen3vl_local/sft_v5/inspect_teacher.py \
  --index checkpoints/sft_v5_data/train_sequence_index.jsonl \
  --output-dir checkpoints/sft_v5_runs/latest/teacher_inspect \
  --num-cases 64
```

`inspect_teacher.py` 会写：

- `teacher_report.json`
- `teacher_report.md`

检查项包括：

- XML weather 只出现在 teacher prompt，不进入 student prompt。
- teacher target 不泄漏 `ANSWER_` / `REFERENCE` / `XML_WEATHER` 等私有标记。
- Q2 option map 非空。
- Q2 student prompt 不泄漏 scenario name。

## 7. 常见失败与定位

### 7.1 NCCL watchdog 卡死

如果日志停在 NCCL collective，随后出现 watchdog timeout，先确认当前代码没有把
`bundle.model` 包进 `DistributedDataParallel(model)`。v5 的 Q2 是否触发取决于每个
rank 的 Q1 student 输出，rank 间 forward 次数天然不一致；DDP wrapper 的 forward hook
可能产生 unmatched collective。当前实现只用 torchrun 启多进程，optimizer step 前
手动 all-reduce LoRA 梯度。

### 7.2 Cache 类型错误

如果出现：

```text
AttributeError: 'tuple' object has no attribute 'get_mask_sizes'
```

说明新版 Transformers 的 Qwen3-VL 收到了 legacy tuple cache。`engine.py::_clone_cache`
必须保持带 `get_mask_sizes` / `get_seq_length` 的 `Cache` 对象类型，不能把新版 cache
退化成 tuple。

### 7.3 CUDA OOM

如果 H20 仍然 OOM，先看 traceback 所在阶段。当前训练侧已经：

- 每帧 OPSD loss 立刻 backward，不把整条 route sequence 的 Qwen 计算图攒到 batch 末尾。
- Teacher/student logits 只裁剪到监督 span 后参与 forward-KL。
- 8 路 rollout 与 4 路 KL 微批解耦，KL forward OOM 会在 backward 前自动二分。
- `train.sh` 默认设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

若日志里的 `parallel_kl/oom_splits` 长期非 0，先只缩小 KL 微批，不改 8 路 rollout、
不改 1024 token 上限：

```bash
PARALLEL_KL_MICROBATCH_SIZE=2 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v5/train.sh ddp
```

只有 OOM 明确发生在 `[q1-grouped]` / `[q2-grouped]` student rollout，而不是
`_opsd_loss_batch_states` KL scoring 时，才需要考虑降低 `QWEN_BATCH_SIZE`；正式训练不应
为了 KL 峰值先缩短 Qwen 输出上限。

### 7.4 `loss does not require grad`

如果旧 run 出现：

```text
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

通常是某个 student rollout 没有输出可监督字段，导致 `_opsd_loss` 或 chunk-level
parallel KL 返回了一个纯 0 loss。当前代码会用第一个可训练 LoRA 参数构造
graph-connected zero，仍保持本帧 / 本 chunk loss=0，但允许 `backward()` 正常执行，
避免某个 rank 因一个坏输出中断全局训练。如果日志出现
`[warn] loss has no grad; replace with graph-zero ...` 或
`[warn] chunk loss has no grad; replace with graph-zero ...`，优先检查对应 frame/chunk
的 `q1_student_output.txt` / `q2_student_output.txt` 或 probe 输出，看模型是否没有写
`RS:`、`ABNORMAL:`、`EVENT:` 等字段。

## 8. 代码注释维护要求

`sft_v5` 代码已经按中文注释口径补充：

- 函数/docstring 说明入口职责。
- 关键逻辑块说明“为什么这样做”，包括逐帧 `allowed_events` 优先、`R-E* -> RE`
  折叠、双标签单标签化、Q1 RS 错误截断、DDP global padding、OPSD teacher/student
  logits 对齐，以及训练前纯 base Qwen 体检不加载 LoRA。
- 测试脚本说明各自防止哪类回归。

后续改 prompt、候选池、memory、训练状态机、probe 输出或 DDP padding 时，需要同步更新
对应代码块注释和本运行文档，避免实现和人工检查口径漂移。
