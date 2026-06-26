# SFT v4 Runbook

SFT v4 是 sequence-memory OPD 路线：一条 episode 是一个 sub-scenario 时间序列，
学生用上一帧输出维护文本 memory，teacher 用 base Qwen + GT hindsight 生成分析文本，
学生同时学习分析 token 和 `ROAD_STRUCTURE/SCENE/STATUS/SUBGOAL` 值 token。

本文默认当前目录是远端 `AutoMoT/`。

当前状态：off-policy actor-learner 代码已经落地。生产训练入口是
`launch_offpolicy.sh`，它会启动 2 个 learner DDP rank + 2 个异步 collector：
默认 GPU0/GPU1 各 1 个 learner，GPU2/GPU3 各 1 个 collector。Phase A 初始正确率
`P_INIT_CORRECT=0.7`，Phase B 噪声率 `PHASE_B_NOISE_PROB=0.15`。如果某帧 step1
未过导致 step2/3 跳过，下一帧帧首会触发一次 skip 纠偏：`BELIEVED_SCENE` 大概率回真实
scene，按 `SKIP_CORRECTION_SCENE_NOISE_PROB=0.15` 小概率同桶扰动，`STATUS/SUBGOAL`
回 init。learner rank0 每 1000 step 发布一次 LoRA snapshot 给 collectors。

当前 prompt 口径：学生仍是三步串行对话，`[MEMORY]` 每层一行并在 step1/2/3 后自更新。
老师三步都是 fresh dialog，不复用上一部 teacher KV，并且每步重新吃 4 张 RGB：
Step1 老师只喂 `BELIEVED_ROAD_STRUCTURE / EGO_TO_GOAL_XY / GROUND_TRUTH_ROAD_STRUCTURE`；
Step2 老师喂 `GROUND_TRUTH_ROAD_STRUCTURE / BELIEVED_SCENE / GROUND_TRUTH_SCENE`；
Step3 老师喂 true road、true scene、believed/GT status-subgoal。老师只输出分析，
且三步统一严格输出 4 个非空行：`### Scene Description:`、
`### Critical Object Description:`、`### Reasoning on Intent:`、
`### Memory Judgment:`；结构化标签由脚本追加。teacher 默认生成上限为
`192/192/192`，软最小长度为 `12/12/12`。

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
  `correct_memory_after_step1_skip` / `need_skip_correction` 理解 step1 skip 后的下一帧纠偏；看
  `collect_episode` 里的 teacher/student 分支和 step3 触发注释，理解一条 trajectory
  如何生成。
- `learn.py`：看 `_sync_bool` 理解 replay 空时为什么不会发 NCCL collective；看
  `trajectory_backward` / `trajectory_loss` 理解 learner 如何用 student raw output 复现
  KV 上下文但不 generate，并用逐帧 micro-backward 控制显存；看
  `publish_snapshot` / `save_checkpoint` 理解 snapshot 与 checkpoint 的区别。
- `eval.py`：模块顶部先执行 GPU 自动选址和 HF offline 环境变量设置，再 import
  `torch` / `engine` / `train`；`probe.py` 先导入 `eval.py`，复用同一套 import 前选卡逻辑。
- `launch_offpolicy.sh`：脚本顶部写了进程布局；路径/env/选卡/STOP 收尾块都有中文注释，
  改部署比例时优先看这里。

当前关键边界：

- `delta = min(anchor[1]-anchor[0], anchor[2]-anchor[1]) // 2`，只封顶 10，允许
  `delta=0`；窄间距 episode 只 warning，不静默改大。
- `EGO_TO_GOAL_XY` 必须来自 `meta["next_target_points"][-1]` 转 ego，缺 meta 或字段
  解析失败会直接报错，不 fallback 到 measurements 或 `(0,0)`。
- memory 含 `BELIEVED_ROAD_STRUCTURE / BELIEVED_SCENE / BELIEVED_STATUS /
  BELIEVED_SUBGOAL / EGO_TO_GOAL_XY`；goal 坐标在帧末为下一帧预取。
- 触发链固定为：step1 后 `should_trigger_step2(memory_road_structure_after_step1,
  gt_road_structure)`；只有 layer-1 命中才跑 step2；step2 后再走
  `should_trigger_step3(memory_scene_after_step2, gt_scene)`。
- skip 纠偏不是每帧强制改 memory；只有上一帧 step1 未过并跳过 step2/3 后，下一帧
  进入内循环前才触发一次。纠偏后的 scene 与 status/subgoal 始终来自同一个 bucket。

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
- learner 每个 optimizer step 仍随机消费一条 trajectory；但不会把整条 trajectory 的
  计算图攒到最后，而是逐帧 `backward()` 并释放图。frame loop 放在 DDP `no_sync()`
  下，本地累完一条 trajectory 后再按固定参数顺序手动 mean-reduce LoRA grad，避免不同
  帧数导致 NCCL collective 序列不一致。
- learner 到 `MAX_STEPS` 后保存 `final/`、写 `STOP`，collector 完成当前 episode 后退出。
- 如果启动后一直没有 collector 写出第一条 `replay/ready/*.jsonl`，learner 会在
  `REPLAY_STARTUP_TIMEOUT_SEC` 后写 `STOP` 并报错退出，避免数据路径或 collector
  加载错误时无限等待。这个 timeout 会先在两个 learner rank 间同步，再 barrier/cleanup，
  因此不会留下半退出的 NCCL 进程。

### 常用环境变量

```bash
OUTPUT_DIR=checkpoints/sft_v4_lora \
RUN_TAG=debug_v4 \
MAX_STEPS=10000 \
LR=3e-5 \
SNAPSHOT_EVERY_STEPS=1000 \
SAVE_STEPS=5000 \
REPLAY_STARTUP_TIMEOUT_SEC=600 \
REPLAY_CAPACITY=256 \
COLLECTORS_PER_GPU=1 \
GPU_MAX_USED_MB=0 \
ALLOW_BUSY_GPUS=0 \
P_INIT_CORRECT=0.7 \
PHASE_B_NOISE_PROB=0.15 \
SKIP_CORRECTION_SCENE_NOISE_PROB=0.15 \
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
learner 会写 `fuse_stop_after_step_<N>/` 和 `fuse_reason.txt`，同时跳过正常 `final/`
保存，避免误用异常 adapter。这里的 `N` 表示最后一个已经完成的 optimizer step；
触发熔断的当前坏 step 会先 `zero_grad`，不会写入 emergency adapter。

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
- `replay/ready/*.jsonl`：collector 写出的 `sft_v4_rollout_v2` trajectory FIFO；
  v2 schema 显式保存 `memory_before_frame`、`memory_after_step1`、
  `memory_after_step2`、三步触发标志和 skip 纠偏标志。旧 `sft_v4_rollout_v1` 会被 learner 拒收并
  移到 `replay/failed/`。
- `latest_lora/v_<step>/` 与 `latest_lora/current_version.txt`：给 collector 用的策略快照。
- `checkpoint-<step>/`：可恢复训练状态，含 adapter + optimizer + scheduler。
- `final/`：最终 adapter，供 eval/probe 使用。
- `sft_v4_adapter_config.json`：随 snapshot / checkpoint / final 一起写出，记录
  `route="sft_v4_offpolicy"`、learner/collector 配置、视觉 LoRA guard 参数和完整
  loss 权重；其中 `loss_weights.rs1` 对应 `L_RS1` 的 ROAD_STRUCTURE 值 token CE。
- `fuse_stop_after_step_<N>/`：视觉 LoRA guard 触发时的 emergency adapter；`N` 是最后
  一个已完成 step，此时不会写正常 `final/`。
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
- `train/loss/{a1,rs1,a2,a3,s2,s3_status,s3_subgoal}`
- `train/loss/{L_A1,L_RS1,L_A2,L_SC,L_A3,L_ST,L_SG}`（PLAN 口径别名）
- `train/step2_trigger_rate`
- `train/step3_trigger_rate`
- `train/fire_rate/{step2,step3}`（PLAN 口径别名）
- `train/accuracy/road_structure`（当前等价于 step2 fire rate）
- `train/rs_flip_rate`
- `train/scene_flip_rate`
- `train/gt_leak_skip_rate/{step1,step2,step3}`（新 teacher prompt 下通常为 0；旧诊断指标保留）
- `train/phase_a_frame_frac`
- `train/skip_correction_rate`
- `train/skip_correction_scene_noise_rate`
- `train/grad_norm/language`
- `train/grad_norm/vision`
- `train/param_norm/lora_language`
- `train/param_norm/lora_vision`
- `train/vision_guard_bad_steps`

---

## 4. Eval

默认只跑学生，不加载 teacher，不做 Phase B GT 注入，memory 全程由学生自更新。
`eval.py` 会在 import torch 前自动选空闲 GPU；显式 pin 仍统一使用 `GPU_IDS=0`，
`--device cpu` / `--device cuda:N` 会跳过自动选址并尊重用户指定。

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

- `road_structure_acc`
- `step2_fire_rate`
- `scene_acc_per_step`
- `scene_recovery_steps`
- `phase_a_scene_recovery_steps`
- `phase_b_scene_recovery_steps`
- `scene_stick_rate`
- `road_structure_flip_rate`
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

随机抽 episode，逐帧 dump 三步 prompt、输出、三层 memory 和 flags。
`probe.py` 与 `eval.py` 共用同一套 import 前 GPU 选址逻辑：默认自动选择空闲 GPU；
需要固定卡时使用 `GPU_IDS=0`，CPU 冒烟时可传 `--device cpu`。

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

- `step1_teacher_user.txt`
- `step1_teacher.txt`
- `step2_teacher_user.txt`
- `step2_teacher.txt`
- `step3_teacher_user.txt`
- `step3_teacher.txt`
- `flags.json` 里的 `road_structure_ok` / `step2_trigger` /
  `analysis_bleu_vs_teacher`

---

## 6. 轻量测试

```bash
python qwen3vl_local/sft_v4/test_memory_update.py
python qwen3vl_local/sft_v4/test_gt_leak_filter.py
python qwen3vl_local/sft_v4/check_loss_mask.py
```

`test_memory_update.py` 现在同时覆盖三层 memory 状态机和 replay schema v2 的关键门控：
step2 未触发时允许没有 step2 target；step2 已触发时必须保存 `memory_after_step1`，
否则 learner 无法按 collector 当时的 `ROAD_STRUCTURE` 桶重放收窄后的 `SCENE_CHOICES`。
它也覆盖 skip 后下一帧纠偏：scene 只能回 GT 或同桶小扰动，status/subgoal 必须跟随
所选 scene 回 init。

KV 复用测试会加载本地 Qwen 权重；有模型时再跑：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v4/test_kv_reuse.py \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct
```

---

## 7. Teacher 输出抽检（`inspect_teacher.py`）

用途：在不跑训练的前提下，**离线评估当前 prompt 设计下老师真实生成内容是否合理**。
该脚本是诊断老师 prompt 迭代的主入口，跑完直接看 Markdown 报告即可决定是否需要再
调 prompt / `min_new_tokens` / `max_new_tokens`。

### 7.1 它做了什么

- 从 `--train-jsonl` 中**随机抽 N 条 episode × 每条 M 帧**；
- 对每帧分别构造 **5 种 memory 起点**，覆盖三层 prompt 状态机的关键分支：

  | 模式 | `memory.road_structure` | `memory.scene` | step1 | step2 | step3 |
  |---|---|---|---|---|---|
  | `all_keep` | = GT | = GT | KEEP | KEEP | KEEP |
  | `rs_change` | 非 GT 桶 | 错桶首个 scene | CHANGE | 跳过 | 跳过 |
  | `scene_change_same_rs` | = GT | 同桶非 GT | KEEP | CHANGE | 跳过 |
  | `event_change` | = GT | = GT | KEEP | KEEP | CHANGE |
  | `scene_change_cross_rs` | = GT | 跨桶非 GT | KEEP | CHANGE | 跳过 |

- 老师路径与 `collect.py` 100% 一致：`load_model_with_lora` →
  `model.disable_adapter()` → `_teacher_generate_kv`（含 `min_new_tokens` 反早停）
  → 全程 `torch.no_grad()`，等价于 frozen base Qwen3-VL-4B-Instruct；
- 逐帧记录 system / user (含图像路径) / teacher-assistant 三类 role 的 prompt 与
  生成内容，加上 token 数和 legacy GT-token 诊断。

### 7.2 GPU 选址（与训练入口一致）

> 项目硬性规则：禁止在命令里手写 `export CUDA_VISIBLE_DEVICES=...`。
> 详见 CLAUDE.md / AGENTS.md §5。

`inspect_teacher.py` 与 `eval.py` / `probe.py` 一样，会在 import torch 前完成 GPU 选址：

- **默认**（自动选址）：`python qwen3vl_local/sft_v4/inspect_teacher.py ...` ——
  脚本自动 `nvidia-smi` 找最空闲 1 张 GPU，并覆盖继承下来的 `CUDA_VISIBLE_DEVICES`；
- **显式 pin**：在命令前加 `GPU_IDS=2`（或 `GPU_IDS=0`）——脚本跳过自动选址，
  直接用给定的物理卡号；
- **CPU 调试**：`--device cpu`——绕过 GPU 选址逻辑（auto 才会触发自动挑卡），仅
  用于代码冒烟，正式跑老师必须 GPU。

不进 torchrun、不进 DDP；单进程单卡。

### 7.3 默认命令（自动选址）

```bash
python qwen3vl_local/sft_v4/inspect_teacher.py \
  --train-jsonl checkpoints/sft_v4_data/train.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --out-dir checkpoints/sft_v4_inspect/run_$(date +%Y%m%d_%H%M%S) \
  --num-episodes 3 \
  --frames-per-episode 4
```

### 7.4 指定 GPU 跑（显式 pin 单卡）

```bash
GPU_IDS=2 python qwen3vl_local/sft_v4/inspect_teacher.py \
  --train-jsonl checkpoints/sft_v4_data/train.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --out-dir checkpoints/sft_v4_inspect/run_$(date +%Y%m%d_%H%M%S) \
  --num-episodes 3 \
  --frames-per-episode 4
```

### 7.5 只跑某个 memory 模式（局部调试）

`--modes` 接逗号分隔列表，五档：`all_keep` / `rs_change` /
`scene_change_same_rs` / `event_change` / `scene_change_cross_rs`。
默认全跑；只想验某条分支时用：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v4/inspect_teacher.py \
  --train-jsonl checkpoints/sft_v4_data/train.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --out-dir checkpoints/sft_v4_inspect/quick \
  --num-episodes 2 --frames-per-episode 2 \
  --modes rs_change,scene_change_same_rs
```

### 7.6 常用参数速查

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--train-jsonl` | `checkpoints/sft_v4_data/train.jsonl` | episode 索引 |
| `--model-dir` | `checkpoints/Qwen3-VL-4B-Instruct` | base Qwen 权重目录 |
| `--out-dir` | `checkpoints/sft_v4_inspect/latest` | 报告产物目录 |
| `--num-episodes` | 3 | 随机抽样的 episode 条数 |
| `--frames-per-episode` | 4 | 每条 episode 抽样的帧数（等步长 + 小抖动） |
| `--modes` | `all_keep,rs_change,scene_change_same_rs,event_change,scene_change_cross_rs` | 启用的 memory 模式 |
| `--seed` | 20260624 | 抽样与随机 wrong-scene 选择的随机种子 |
| `--device` | `auto` | `auto` / `cuda:0` / `cpu` |
| `--lora-*` | 与训练默认对齐 | 仅为加载 PEFT bundle，全程 disable_adapter |

### 7.7 产物

输出目录下两份文件：

- `teacher_report.md`：人类可读 Markdown。结构：

  ```
  # SFT v4 Teacher Inspection Report
  # Episode <run_id>
  ## Frame <idx> (phase A/B)
      [4 张 stitched RGB 路径]
  ### Mode <all_keep|rs_change|scene_change_same_rs|event_change|scene_change_cross_rs>
      [当前 memory 字段]
  #### Step 1 — road-structure verdict <KEEP|CHANGE> (tokens: N, leak=False)
      [ROLE = system]      ... system prompt 全文
      [ROLE = user (with 4 stitched RGB images attached)] ... step1 user prompt
      [ROLE = assistant — teacher raw output]             ... 老师真实输出
      [ROLE = scripted target (this is what the student is supervised on)]
  #### Step 2 — scene verdict <KEEP|CHANGE> (tokens: N, leak=False)
      （或 "Step 2 — skipped (memory.road_structure != gt_road_structure)"）
      [ROLE = user (fresh teacher dialog with 4 images)]
      [ROLE = assistant — teacher raw output]
      [ROLE = scripted target (this is what the student is supervised on)]
  #### Step 3 — status/subgoal verdict <KEEP|CHANGE> (tokens: N, leak=False)
      （或 "Step 3 — skipped (step2 did not fire or memory.scene != gt_scene)"）
  ```

  其中 **`scripted target`** 是 `build_step{1,2,3}_teacher_target` 拼出的实际监督
  字符串——直接对比"老师推理"和"脚本兜底标签"可以一眼看出 prompt 设计是否在让老师
  做有价值的推理（而不是只输出垃圾被脚本兜底）。

- `teacher_report.jsonl`：同份内容机器可读，每行一帧 + 一种模式，方便后续聚合
  统计（比如平均 token 数、leak 比率、step3 触发占比）。

### 7.8 评估老师输出的关键检查点

跑完 `teacher_report.md` 后，按顺序检查：

1. **token_count 是否 ≥ `TEACHER_MIN_NEW_TOKENS_STEP{1,2,3}`**
   （prompts.py 默认 12 / 12 / 12）。如果还出现 5~10 token 的老师输出，
   说明 min_new_tokens 闸门没生效或者 EOS id 集合不全。
2. **leak_detected 仅作诊断**。新口径下 step2/step3 prompt 会显式给 GT token，
   所以该字段不再决定训练是否跳过分析 loss；真正要看的是 raw analysis 是否有
   prompt marker、半截选项名或结构化标签泄漏到 scripted target。
3. **teacher raw output 是否严格为 4 个非空行**，且每行分别以
   `### Scene Description:`、`### Critical Object Description:`、
   `### Reasoning on Intent:`、`### Memory Judgment:` 开头；不要出现 bullet、
   JSON、代码块或额外行。若 raw 不达标，`scripted target` 会退回四行 fallback，
   这说明该 case 的分析监督价值不足。
4. **`all_keep` 模式下老师是否真的论证 KEEP**（不要去翻案）；
   **`rs_change` 模式下老师是否先纠正道路结构**；**`scene_change_same_rs` /
   `scene_change_cross_rs` 模式下老师是否描述"实际场景看起来像 X"而不是简单复述
   memory scene**；**`event_change` 模式下老师是否解释"虽然 scene 对，但
   status/subgoal 应该推进"**。
5. **step3 老师是否能围绕 EVENT_OPTIONS 做有效 keep/correct 引导**，不要只输出
   "I observe the current driving phase" 这类空话。

如果第 3-5 条不达标，回头修 `build_step{1,2,3}_teacher_prompt` 里的 `focus_line`；
优先保持短 prompt，只在必要时给 1 个最关键的证据锚点，不要恢复长证据清单。

---
