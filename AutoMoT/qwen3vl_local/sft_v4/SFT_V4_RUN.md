# SFT v4 Runbook

SFT v4 是 sequence-memory OPD 路线：一条 episode 是一个 sub-scenario 时间序列，
学生用上一帧输出维护文本 memory，teacher 用 base Qwen + GT hindsight 生成分析文本，
学生同时学习分析 token 和 `SCENE/STATUS/SUBGOAL` 值 token。

本文默认当前目录是远端 `AutoMoT/`。

当前状态：off-policy actor-learner 代码已经落地。生产训练入口是
`launch_offpolicy.sh`，它会启动 2 个 learner DDP rank + 2 个异步 collector：
默认 GPU0/GPU1 各 1 个 learner，GPU2/GPU3 各 1 个 collector。Phase A 初始正确率
`P_INIT_CORRECT=0.5`，Phase B 噪声率 `PHASE_B_NOISE_PROB=0.15`，learner rank0
每 1000 step 发布一次 LoRA snapshot 给 collectors。

`train.sh` / `train.py` 只保留为 on-policy 兼容调试入口，不是 v4 生产路径。

代码已补中文 module docstring、函数说明和关键逻辑块注释。需要读实现时建议顺序：

1. `prompts.py`：先理解 memory 格式、状态机更新和三步 prompt。
2. `build_dataset.py`：理解 episode index 的数据契约。
3. `replay.py`：理解 trajectory schema、原子写入、FIFO 和文件锁。
4. `collect.py`：理解 rollout、snapshot reload、Phase B 噪声和 trajectory 写盘。
5. `learn.py`：理解 learner-only DDP、teacher-forced loss 和 snapshot 发布。
6. `eval.py` / `probe.py`：理解自由生成评估和 case dump。

源码注释索引：

- `replay.py`：看 `ensure_replay_dirs` 理解目录语义；看 `directory_lock` /
  `claim_episode_index` 理解多 collector 抢任务；看 `write_trajectory` 理解
  `pending -> ready` 原子切换；看 `sample_ready_file` 理解为什么 learner 允许重抽。
- `collect.py`：看 `_load_adapter_state_if_present` / `_maybe_refresh_snapshot` 理解
  LoRA snapshot 加载；看 `_inject_phase_b_noise` 理解 Phase B 噪声；看
  `collect_episode` 里的 teacher/student 分支和 step3 触发注释，理解一条 trajectory
  如何生成。
- `learn.py`：看 `_sync_bool` 理解 replay 空时为什么不会发 NCCL collective；看
  `trajectory_loss` 理解 learner 如何用 student raw output 复现 KV 上下文但不 generate；
  看 `publish_snapshot` / `save_checkpoint` 理解 snapshot 与 checkpoint 的区别。
- `launch_offpolicy.sh`：脚本顶部写了进程布局；路径/env/选卡/STOP 收尾块都有中文注释，
  改部署比例时优先看这里。

当前关键边界：

- `delta = min(anchor[1]-anchor[0], anchor[2]-anchor[1]) // 2`，只封顶 10，允许
  `delta=0`；窄间距 episode 只 warning，不静默改大。
- `EGO_TO_GOAL_XY` 必须来自 `meta["next_target_points"][-1]` 转 ego，缺 meta 或字段
  解析失败会直接报错，不 fallback 到 measurements 或 `(0,0)`。
- memory 的 goal 坐标在帧末为下一帧预取；step3 触发统一走
  `should_trigger_step3(memory_scene_after_step2, gt_scene)`。

---

## 1. 构建 Episode Index

只生成 episode index，不生成 per-frame 训练样本。

```bash
python qwen3vl_local/sft_v4/build_dataset.py \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --output-dir checkpoints/sft_v4_data \
  --val-ratio 0.1 \
  --seed 42
```

快速检查 keyframes schema 和 episode 数：

```bash
python qwen3vl_local/sft_v4/build_dataset.py \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --output-dir checkpoints/sft_v4_data \
  --dry-run
```

当前脚本兼容两种 keyframes 输入：顶层对象里的 `runs` 列表，以及早期顶层直接为 run list
的临时 dump。

---

## 2. Off-Policy 训练

### 生产入口

```bash
bash qwen3vl_local/sft_v4/launch_offpolicy.sh
```

显式 pin 4 卡：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v4/launch_offpolicy.sh
```

launcher 会做这些事：

- 建立 `${OUTPUT_DIR}/run_<RUN_TAG>/`，并维护 `${OUTPUT_DIR}/latest` symlink。
- 启动 `torchrun --nproc_per_node=2 qwen3vl_local/sft_v4/learn.py`，只让 learner 进入
  DDP / NCCL。
- 在 collector GPU 上启动 `COLLECTORS_PER_GPU=1` 个 `collect.py` 进程；collector
  不调用 DDP，只读 LoRA snapshot、写 replay。
- learner rank0 先发布 `latest_lora/v_0/`，collector 等到初始 snapshot 后开始采集。
- learner 到 `MAX_STEPS` 后保存 `final/`、写 `STOP`，collector 完成当前 episode 后退出。

### 常用环境变量

```bash
OUTPUT_DIR=checkpoints/sft_v4_lora \
RUN_TAG=debug_v4 \
MAX_STEPS=10000 \
LR=3e-5 \
SNAPSHOT_EVERY_STEPS=1000 \
SAVE_STEPS=5000 \
REPLAY_CAPACITY=256 \
COLLECTORS_PER_GPU=1 \
GPU_MAX_USED_MB=0 \
ALLOW_BUSY_GPUS=0 \
P_INIT_CORRECT=0.5 \
PHASE_B_NOISE_PROB=0.15 \
VISION_LR_SCALE=0.1 \
MAX_VISION_LR_SCALE=0.25 \
LANGUAGE_CLIP_NORM=1.0 \
VISION_CLIP_NORM=0.3 \
VISION_GUARD_ENABLED=1 \
GPU_IDS=0,1,2,3 \
bash qwen3vl_local/sft_v4/launch_offpolicy.sh
```

`GPU_IDS` 未显式给定时，launcher 只会自动选择 `memory.used <= GPU_MAX_USED_MB`
的 GPU。默认阈值为 0 MiB，避免把已有进程的忙卡分配给 learner 或 collector。
`ALLOW_BUSY_GPUS=1` 只用于你确认要覆盖该检查的调试场景。

`COLLECTORS_PER_GPU` 默认保守设为 1，避免服务器处于 CUDA `Exclusive_Process`
模式时，同一张 GPU 被多个 CUDA 进程同时占用并触发
`CUDA-capable device(s) is/are busy or unavailable`。确认 `nvidia-smi -q -d COMPUTE`
显示允许多进程、且单卡显存/吞吐稳定后，再试 `COLLECTORS_PER_GPU=2` 或 `3`。

视觉 LoRA 默认关闭。需要打开时：

```bash
LORA_VISION_SCOPE=merger GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v4/launch_offpolicy.sh
```

注意：off-policy learner 为了控制显存，图像 prefill 默认在 `no_grad` 下执行；显式开启
视觉 LoRA 时 DDP 会自动使用 `find_unused_parameters=True`，视觉侧梯度可能为 0。生产训练
默认建议保持 `LORA_VISION_SCOPE=off`。如果视觉 LoRA 的梯度/参数范数连续异常，
learner 会写 `fuse_stop_step_<N>/` 和 `fuse_reason.txt`，同时跳过正常 `final/`
保存，避免误用异常 adapter。

### 烟雾检查

轻量烟雾建议把 step 和 collector 数压低：

```bash
MAX_STEPS=2 COLLECTORS_PER_GPU=1 REPLAY_CAPACITY=8 GPU_IDS=0,1,2,3 \
bash qwen3vl_local/sft_v4/launch_offpolicy.sh
```

这仍会加载模型，不适合在没有本地 Qwen 权重的机器上跑；本地只做静态检查时看第 6 节。

### 输出结构

`OUTPUT_DIR` 下会自动套 `run_<RUN_TAG>/` 子目录，并维护 `latest` symlink。
`HF_HOME` 默认固定在 `${OUTPUT_DIR}/.hf_cache`，不会跟着 run 子目录重复下载。

主要产物：

- `offpolicy.log`：launcher、learner、collector 的合并日志。
- `replay/ready/*.jsonl`：collector 写出的 trajectory FIFO。
- `latest_lora/v_<step>/` 与 `latest_lora/current_version.txt`：给 collector 用的策略快照。
- `checkpoint-<step>/`：可恢复训练状态，含 adapter + optimizer + scheduler。
- `final/`：最终 adapter，供 eval/probe 使用。
- `fuse_stop_step_<N>/`：视觉 LoRA guard 触发时的 emergency adapter；此时不会写正常
  `final/`。
- `STOP`：正常停止哨兵，collector 会在 episode 结束后观察并退出。

### 兼容入口

从最近 checkpoint 恢复：

```bash
RESUME_FROM_CHECKPOINT=latest GPU_IDS=0,1,2,3 \
bash qwen3vl_local/sft_v4/launch_offpolicy.sh
```

也可以指定明确目录：

```bash
RESUME_FROM_CHECKPOINT=checkpoints/sft_v4_lora/latest/checkpoint-5000 GPU_IDS=0,1,2,3 \
bash qwen3vl_local/sft_v4/launch_offpolicy.sh
```

恢复后 learner 会加载 adapter/optimizer/scheduler，并从恢复 step 重新发布
`latest_lora/v_<step>/` 给 collectors。

`train.sh` / `train.py` 仍可用于单卡或历史 on-policy debug，但它会回到
work-stealing + local-SGD 口径，不是 v4 off-policy 生产训练路径：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_v4/train.sh check
```

---

## 3. TensorBoard

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_v4_lora/latest/tb
```

重点看：

- `train/loss_total`
- `train/loss/{a1,a2,a3,s2,s3_status,s3_subgoal}`
- `train/step3_trigger_rate`
- `train/scene_flip_rate`
- `train/gt_leak_skip_rate/{step2,step3}`
- `train/phase_a_frame_frac`
- `grad_norm/{language,vision}`
- `param_norm/lora_{language,vision}`

---

## 4. Eval

默认只跑学生，不加载 teacher，不做 Phase B GT 注入，memory 全程由学生自更新。

```bash
GPU_IDS=0 python qwen3vl_local/sft_v4/eval.py \
  --jsonl checkpoints/sft_v4_data/val.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --lora-dir checkpoints/sft_v4_lora/latest/final \
  --save-root checkpoints/sft_v4_lora/latest
```

输出：

- `checkpoints/sft_v4_lora/latest/eval_v4/metrics.json`
- `checkpoints/sft_v4_lora/latest/eval_v4/episodes.json`

核心指标：

- `scene_acc_per_step`
- `scene_recovery_steps`
- `phase_a_scene_recovery_steps`
- `phase_b_scene_recovery_steps`
- `scene_stick_rate`
- `scene_flip_rate`
- `step3_trigger_rate`
- `status_acc_given_correct_scene`
- `subgoal_acc_given_correct_scene`
- `all_acc_per_step`

可选 teacher-ref BLEU：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v4/eval.py \
  --jsonl checkpoints/sft_v4_data/val.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --lora-dir checkpoints/sft_v4_lora/latest/final \
  --save-root checkpoints/sft_v4_lora/latest \
  --with-teacher-ref
```

这会额外加载一份 base Qwen teacher，记录 `analysis_bleu_vs_teacher`。默认关闭，因为显存和耗时约增加一倍。

---

## 5. Probe

随机抽 episode，逐帧 dump 三步 prompt、输出、memory 和 flags。
`probe.py` 与 `eval.py` 一样会默认自动选择空闲 GPU；需要固定卡时使用 `GPU_IDS=0`。

```bash
GPU_IDS=0 python qwen3vl_local/sft_v4/probe.py \
  --jsonl checkpoints/sft_v4_data/val.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --lora-dir checkpoints/sft_v4_lora/latest/final \
  --save-root checkpoints/sft_v4_lora/latest \
  --num-episodes 4 \
  --seed 0
```

带 teacher 文本：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v4/probe.py \
  --jsonl checkpoints/sft_v4_data/val.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --lora-dir checkpoints/sft_v4_lora/latest/final \
  --save-root checkpoints/sft_v4_lora/latest \
  --num-episodes 2 \
  --with-teacher
```

`--with-teacher` 会额外写：

- `step1_teacher.txt`
- `step2_teacher_user.txt`
- `step2_teacher.txt`
- `step3_teacher_user.txt`
- `step3_teacher.txt`
- `flags.json` 里的 `analysis_bleu_vs_teacher`

---

## 6. 轻量测试

```bash
python qwen3vl_local/sft_v4/test_memory_update.py
python qwen3vl_local/sft_v4/test_gt_leak_filter.py
python qwen3vl_local/sft_v4/check_loss_mask.py
```

KV 复用测试会加载本地 Qwen 权重；有模型时再跑：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v4/test_kv_reuse.py \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct
```

