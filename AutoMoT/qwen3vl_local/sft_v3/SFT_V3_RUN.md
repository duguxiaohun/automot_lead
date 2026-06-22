# SFT v3 Runbook

SFT v3 是 sequence-memory OPD 路线：一条 episode 是一个 sub-scenario 时间序列，
学生用上一帧输出维护文本 memory，teacher 用 base Qwen + GT hindsight 生成分析文本，
学生同时学习分析 token 和 `SCENE/STATUS/SUBGOAL` 值 token。

本文默认当前目录是远端 `AutoMoT/`。

代码已补中文 module docstring、函数说明和关键逻辑块注释。需要读实现时建议顺序：

1. `prompts.py`：先理解 memory 格式、状态机更新和三步 prompt。
2. `build_dataset.py`：理解 episode index 的数据契约。
3. `train.py`：重点看 `KVState`、`_append_token_ids`、`iter_episode_loss_packs`。
4. `eval.py` / `probe.py`：理解自由生成评估和 case dump。

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

### 多卡 DDP

```bash
DDP_GPU_COUNT=4 bash qwen3vl_local/sft_v3/train.sh ddp
```

显式 pin 4 卡：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_v3/train.sh ddp
```

DDP 下 `grad_accum` 固定要求为 1。DDP 不跑 in-loop eval；如果设置
`EVAL_STEPS>0`，`train.py` 会直接报错，训练后单独跑 `eval.py`。

### 烟雾检查

```bash
GPU_IDS=0 bash qwen3vl_local/sft_v3/train.sh check
```

### 常用环境变量

```bash
OUTPUT_DIR=checkpoints/sft_v3_lora \
RUN_TAG=debug_v3 \
NUM_EPOCHS=1 \
LR=3e-5 \
OUTER_STRIDE=1 \
W_A1=0.2 W_A2=0.2 W_A3=0.2 \
W_S2=1.0 W_S3_STATUS=1.0 W_S3_SUBGOAL=1.0 \
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

KV 复用测试会加载本地 Qwen 权重；有模型时再跑：

```bash
GPU_IDS=0 python qwen3vl_local/sft_v3/test_kv_reuse.py \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct
```
