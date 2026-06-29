# SFT v3 Runbook

SFT v3 当前是 offline OPSD 路线：student 自由生成的文本就是 on-policy rollout；
privileged teacher 不再提供 hard teacher text CE，而是在同一批未裁剪的 student step token
上给出 full-vocabulary logits，当前训练用 forward-KL 分布匹配监督 analysis 与
`ROAD_STRUCTURE/SCENE/STATUS/SUBGOAL` 值 token；如果后续要改成 JSD，需要显式新增
loss type 开关，并同步 v3/v4 prompt/文档验证。

Prompt 同步规则：v3 与 v4 的 prompt、Memory、状态机和 target span 严格同步，唯一实现源是
`qwen3vl_local/sft_v4/prompts.py`；`qwen3vl_local/sft_v3/prompts.py` 只 re-export
v4 并保留少量 v3 兼容别名。改 prompt 时必须同时检查 v3/v4，训练方式差异仅是：
v3 offline on-policy OPSD；v4 off-policy actor-learner replay。
因此任何 heading、label、Memory 字段、trigger helper、target span 或 canonical scene
规则的改动，都要同时静态编译 `sft_v3/*.py` 与 `sft_v4/*.py`，并至少跑对应的
memory / mask / KV smoke test；不要只验证 v4 collector/learner。

一条 episode 仍是一个 sub-scenario 时间序列，学生用上一帧输出维护文本 memory；
但监督信号来自 OPSD 的 privileged-teacher logit distribution，而不是离线物化 teacher 文本。
学生同时学习 analysis token 和 `ROAD_STRUCTURE/SCENE/STATUS/SUBGOAL` 值 token。
teacher 端始终通过 `eval() + no_grad + disable_adapter()` 走 frozen/base Qwen，并分别使用
v4 的 `SYSTEM_PROMPT_STEP1/2/3`；student 端按 v4 collector 的 on-policy 对话顺序自由生成。
OPSD scoring 使用 student 自由生成时真实进入 KV 的 token ids；原始文本只用于解析标签和值
span，且不先 `.strip()`，避免 batch_decode 后重分词造成 loss token 与实际 memory/KV 轨迹漂移。
`build_dataset.py` 与 v4 一样保留 `raw_gt_scene`，训练/eval 标签使用 canonical `gt_scene`。

本文默认当前目录是远端 `AutoMoT/`。

代码已补中文 module docstring、函数说明和关键逻辑块注释。需要读实现时建议顺序：

1. `prompts.py`：先理解 memory 格式、状态机更新和三步 prompt。
2. `build_dataset.py`：理解 episode index 的数据契约。
3. `train.py`：重点看 `KVState`、`_append_token_ids_with_logits`、
   `_opsd_loss_from_states`、`iter_episode_loss_packs`。
4. `eval.py` / `probe.py`：理解自由生成评估和 case dump。

当前关键边界：

- `delta = min(anchor[1]-anchor[0], anchor[2]-anchor[1]) // 2`，只封顶 10，允许
  `delta=0`；窄间距 episode 只 warning，不静默改大。
- `EGO_TO_GOAL_XY` 必须来自 `meta["next_target_points"][-1]` 转 ego，缺 meta 或字段
  解析失败会直接报错，不 fallback 到 measurements 或 `(0,0)`。
- memory 的 goal 坐标在帧末为下一帧预取；step2/step3 触发统一走 v4 的
  `should_trigger_step2/3(before, after, gt, reset_this_frame)` 稳定门。

---

## 1. 构建 Episode Index

只生成 episode index，不生成 per-frame 训练样本。

```bash
python qwen3vl_local/sft_v3/build_dataset.py \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --output-dir checkpoints/sft_v3_data \
  --val-ratio 0.1 \
  --seed 42
```

快速检查 keyframes schema 和 episode 数：

```bash
python qwen3vl_local/sft_v3/build_dataset.py \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --output-dir checkpoints/sft_v3_data \
  --dry-run
```

当前脚本兼容两种 keyframes 输入：顶层对象里的 `runs` 列表，以及早期顶层直接为 run list
的临时 dump。

---

## 2. 训练

### 单卡

```bash
bash qwen3vl_local/sft_v3/train.sh single
```

显式 pin 单卡：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_v3/train.sh single
```

### 多卡 work-stealing local-SGD

```bash
DDP_GPU_COUNT=4 bash qwen3vl_local/sft_v3/train.sh ddp
```

显式 pin 4 卡：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v3/train.sh ddp
```

多卡模式不再包 DDP，也不做静态 rank 分片：所有 rank 从同一个 TCPStore counter
抢 episode，谁空闲谁抢，全部 episode 都会被训练，不截断尾部。`SYNC_EVERY_EPISODES`
表示每个 rank 目标处理多少个 episode 后同步一次；实际每轮最多开放
`SYNC_EVERY_EPISODES * world_size` 条全局 episode。**默认 4**：sft_v3 每帧需要
teacher/student 各 ~80 step 自由生成（≈ 6 sec/帧、14 帧/episode ≈ 85 sec/episode），
K=4 时每个 rank 一轮约 5.6 分钟，给 work-stealing 不均衡留出余量；想要更松的同步
（少同步、参数漂移更大）可显式调大如 16；设为 `1` 最接近同步 SGD；`0` 仅 epoch 末同步。

NCCL watchdog 默认 10 分钟太短，`setup_distributed` 已经把超时放宽到 **2 小时**
（同时影响 TCPStore.wait），所以即便某 rank 在 episode 里卡几分钟也不会触发死锁。

local-SGD 启动后会先广播 rank0 的 LoRA 初始权重；`checkpoint-*` 和 `final/`
都在参数平均后由 rank0 保存。参数平均按本轮各 rank 的 optimizer step 数加权，
空闲 rank 不贡献旧参数，只接收平均后的 adapter。`PER_DEVICE_BS` 固定为 1。
保存的 `sft_v3_adapter_config.json` 会记录 `distributed_train` 口径，便于后续
eval/probe 或审计确认 adapter 来自 work-stealing local-SGD。

注意：这里的“异步”指 episode 分配异步，参数仍会周期同步。快 rank 到达同步点后
会先在 TCPStore 上等慢 rank，所有 rank 到齐后才进入 NCCL allreduce / broadcast；
这样等待慢 episode 时不会占着 NCCL collective 超过 watchdog timeout。

sync 日志里：

- `step` 是 rank0 本地 optimizer step，不代表多卡总训练步数。
- `all_rank_steps` 是所有 rank 的 optimizer step 汇总，用作 checkpoint step、
  scheduler 对齐与 TensorBoard sync 横轴。
- `round_eps` 是本轮所有 rank 实际完成的 episode 数；`total_eps` 是累计已完成
  episode 数，用来确认 work-stealing 是否完整消费数据。
- TensorBoard 同步标量写在 `train/sync/{round_weight,episodes_this_round,episodes_total,all_rank_steps}`。

`MAX_STEPS>0` 会在 episode 内截断，只允许烟雾/调试使用：`check` 模式自动允许；
普通训练若确实要截断，必须显式设置 `ALLOW_MAX_STEPS_TRUNCATION=1`。

如果视觉 LoRA 熔断在任意 rank 触发，sync 后由 rank0 写
`fuse_stop_step_<N>/fuse_reason.txt`，其中包含触发 rank、梯度/参数 norm 与
all-rank step，避免非 rank0 触发时丢诊断文件。

多卡不跑 in-loop eval；如果设置 `EVAL_STEPS>0`，`train.py` 会直接报错，训练后
单独跑 `eval.py`。

### 烟雾检查

```bash
GPU_IDS=0 bash qwen3vl_local/sft_v3/train.sh check
```

`check` 模式只确认数据 / 模型 / LoRA 链路能启动，**不会保存 `final/` adapter**。

### 单卡轻量验证（训练后 eval/probe）

如果要像 v2 一样快速验证“训练出来的模型能否被 eval/probe 加载”，用下面的单卡 tiny train：

```bash
RUN_TAG=quick_verify \
MAX_STEPS=1 \
ALLOW_MAX_STEPS_TRUNCATION=1 \
SAVE_STEPS=999999 \
GPU_IDS=0 \
bash qwen3vl_local/sft_v3/train.sh single
```

这会在 `checkpoints/sft_v3_lora/run_quick_verify/final/` 保存一个极小步数 adapter，
同时把 `latest` 指向该 run。它只用于闭环验收，不代表模型质量。

训练后立即跑小样本 eval：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v3/eval.py \
  --jsonl checkpoints/sft_v3_data/val.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --lora-dir checkpoints/sft_v3_lora/latest/final \
  --save-root checkpoints/sft_v3_lora/latest \
  --max-episodes 4
```

再跑 probe，把逐帧输入、学生输出、teacher 目标和真值全部落盘：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v3/probe.py \
  --jsonl checkpoints/sft_v3_data/val.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --lora-dir checkpoints/sft_v3_lora/latest/final \
  --save-root checkpoints/sft_v3_lora/latest \
  --num-episodes 2 \
  --seed 0 \
  --with-teacher
```

轻量验证产物：

- `checkpoints/sft_v3_lora/latest/eval_v3/metrics.json`：小样本指标。
- `checkpoints/sft_v3_lora/latest/eval_v3/episodes.json`：每条 episode 的逐帧预测、
  memory 轨迹、GT scene/status/subgoal 与触发标志。
- `checkpoints/sft_v3_lora/latest/probe_v3/`：case dump；每帧目录保存
  `rgb_*.jpg`（实际输入图像副本）、`step{1,2,3}_user.txt`（学生真实输入 prompt）、
  `step{1,2,3}_student.txt`（学生输出）、`step{1,2,3}_teacher.txt` /
  `step{2,3}_teacher_user.txt`（`--with-teacher` 时的监督目标和 teacher 私有输入）、
  `memory_before.json`、`memory_after.json`、`flags.json`（含
  `gt_scene/gt_status/gt_subgoal`）、`timeline.json/png` 和顶层 `manifest.json`。

### 常用环境变量

```bash
OUTPUT_DIR=checkpoints/sft_v3_lora \
RUN_TAG=debug_v3 \
NUM_EPOCHS=1 \
LR=3e-5 \
OUTER_STRIDE=1 \
SYNC_EVERY_EPISODES=4 \
W_A1=0.2 W_A2=0.2 W_A3=0.2 \
W_RS1=1.0 W_S2=1.0 W_S3_STATUS=1.0 W_S3_SUBGOAL=1.0 \
GPU_IDS=0 \
bash qwen3vl_local/sft_v3/train.sh single
```

视觉 LoRA 默认关闭。需要打开时：

```bash
LORA_VISION_SCOPE=merger GPU_IDS=0 bash qwen3vl_local/sft_v3/train.sh single
```

`OUTPUT_DIR` 下会自动套 `run_<RUN_TAG>/` 子目录，并维护 `latest` symlink。
`HF_HOME` 默认固定在 `${OUTPUT_DIR}/.hf_cache`，不会跟着 run 子目录重复下载。

---

## 3. TensorBoard

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_v3_lora/latest/tb
```

重点看：

- `train/loss_total`
- `train/loss/{a1,rs1,a2,a3,s2,s3_status,s3_subgoal}`
- `train/step2_trigger_rate`
- `train/step3_trigger_rate`
- `train/road_structure_flip_rate`
- `train/scene_flip_rate`
- `train/gt_leak_skip_rate/{step2,step3}`：legacy hook 兼容项；当前 v4 no-op 下应为 0
- `train/phase_a_frame_frac`
- `grad_norm/{language,vision}`
- `param_norm/lora_{language,vision}`

---

## 4. Eval

默认只跑学生，不加载 teacher，不做 Phase B GT 注入，memory 全程由学生自更新。
LoRA 加载只使用 PEFT 读取 adapter，然后默认 `merge_and_unload` 合并进 base；
后续增量 decode 走 `qwen3vl_local/mrope_utils.py::qwen3vl_incremental_forward`，不会再走
PEFT wrapper 的 `prepare_inputs_for_generation`，因此不会触发旧的 `cache_position`
被裁掉问题。

`sft_v3_adapter_config.json` 为了审计会保存完整 target module 路径；PEFT 的
`adapter_config.json` 可能只保存 `q_proj/down_proj/...` 短名。加载端按 PEFT 后缀匹配
语义校验二者兼容，同时仍检查视觉 LoRA scope 和 adapter 权重 key，避免普通/视觉 LoRA
混用时静默漏挂。

推理路径已用 `torch.inference_mode()` 包住 full prefill、decode 和 step2/3 KV 续写，
不会在 eval/probe 中构建 autograd graph。`--with-teacher` 仍会同卡额外加载一份 base
Qwen teacher，显存接近翻倍；teacher step1/2/3 生成后会立即释放自己的 KV cache，
新的 full generate 也会先清上一轮 `_last_decode_state`。如果只看 student 指标，
先不要加 `--with-teacher`。

```bash
GPU_IDS=0 python qwen3vl_local/sft_v3/eval.py \
  --jsonl checkpoints/sft_v3_data/val.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --lora-dir checkpoints/sft_v3_lora/latest/final \
  --save-root checkpoints/sft_v3_lora/latest
```

输出：

- `checkpoints/sft_v3_lora/latest/eval_v3/metrics.json`
- `checkpoints/sft_v3_lora/latest/eval_v3/episodes.json`

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
GPU_IDS=0 python qwen3vl_local/sft_v3/eval.py \
  --jsonl checkpoints/sft_v3_data/val.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --lora-dir checkpoints/sft_v3_lora/latest/final \
  --save-root checkpoints/sft_v3_lora/latest \
  --with-teacher-ref
```

这会额外加载一份 base Qwen teacher，记录 `analysis_bleu_vs_teacher`。默认关闭，因为显存和耗时约增加一倍。

---

## 5. Probe

随机抽 episode，逐帧 dump 三步 prompt、输出、memory 和 flags。

```bash
GPU_IDS=0 python qwen3vl_local/sft_v3/probe.py \
  --jsonl checkpoints/sft_v3_data/val.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --lora-dir checkpoints/sft_v3_lora/latest/final \
  --save-root checkpoints/sft_v3_lora/latest \
  --num-episodes 4 \
  --seed 0
```

带 teacher 文本：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v3/probe.py \
  --jsonl checkpoints/sft_v3_data/val.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --lora-dir checkpoints/sft_v3_lora/latest/final \
  --save-root checkpoints/sft_v3_lora/latest \
  --num-episodes 2 \
  --with-teacher
```

`--with-teacher` 是重诊断模式，会额外常驻一份 base Qwen 并对 teacher step 独立重喂
4 张 RGB；建议只配合 `--num-episodes 1-2` 使用。OOM 时先去掉 `--with-teacher`，
或换空闲大显存卡后再跑 teacher dump。

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
python qwen3vl_local/sft_v3/test_memory_update.py
python qwen3vl_local/sft_v3/test_gt_leak_filter.py
python qwen3vl_local/sft_v3/check_loss_mask.py
```

`test_gt_leak_filter.py` 现在检查的是 v3/v4 hook 同步：当前 v4 的
`check_gt_leak_*` 是 legacy no-op，v3 不应维护第二套字面正则。

KV 复用测试会加载本地 Qwen 权重；有模型时再跑：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v3/test_kv_reuse.py \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct
```
