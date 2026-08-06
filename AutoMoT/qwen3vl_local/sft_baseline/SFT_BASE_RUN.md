# SFT Baseline Run

从远端 `AutoMoT/` 目录运行。数据根默认是 `lead_data`，输出默认写到 `checkpoints/sft_baseline_*`。

## Build Dataset

```bash
python qwen3vl_local/sft_baseline/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_baseline_data
```

Smoke：

```bash
python qwen3vl_local/sft_baseline/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_baseline_data_smoke \
  --max-routes 4 \
  --max-frames-per-route 16
```

## Static Checks

```bash
python qwen3vl_local/sft_baseline/check_loss_mask.py
python qwen3vl_local/sft_baseline/test_memory_curriculum.py
python qwen3vl_local/sft_baseline/test_prompt_snapshots.py
python qwen3vl_local/sft_baseline/test_train_resume.py
```

## Train

单卡：

```bash
GPU_IDS=0 bash qwen3vl_local/sft_baseline/train.sh single
```

4 卡：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_baseline/train.sh ddp
```

常用覆盖：

```bash
OUTPUT_DIR=checkpoints/sft_baseline_runs \
TRAIN_SAMPLING_MODE=transition_segments \
SEGMENT_LENGTH=24 \
SEGMENTS_PER_ROUTE=4 \
NEGATIVE_SEGMENT_RATIO=0.25 \
TRANSITION_LABEL_MODE=binary \
TRANSITION_REPEAT_MODE=add \
HIGHWAY_ROUTE_SAMPLE_TARGET=0.5 \
ROAD_LOSS_BALANCE_MODE=none \
JOINT_BALANCE_REPEAT_MODE=inverse_sqrt \
JOINT_BALANCE_REPEAT_COMBINE=add \
JOINT_BALANCE_DROP_MAJORITY=1 \
JOINT_TARGET_BALANCE_MODE=exact \
JOINT_TARGET_BALANCE_COUNT=0 \
UE_EVENT_LOSS_WEIGHT=2.0 \
RE_EVENT_LOSS_WEIGHT=1.0 \
MEMORY_RS_WRONG_PROB=0.30 \
MEMORY_RS_UNKNOWN_PROB=0.40 \
MEMORY_EVENT_WRONG_PROB=0.35 \
MEMORY_EVENT_UNKNOWN_PROB=0.35 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_baseline/train.sh ddp
```

当前 launcher 默认已经使用小片段 + 四格目标均衡策略；`HIGHWAY_ROUTE_SAMPLE_TARGET`
默认是 `0.5`，`JOINT_TARGET_BALANCE_MODE=exact` 会把最终训练 work list 按
`HIGHWAY/NON_HIGHWAY x UE/RE` 四格尽量拉到等量。若要复现旧整条 route 训练：

```bash
TRAIN_SAMPLING_MODE=full_route \
TRANSITION_LABEL_MODE=fine \
TRANSITION_REPEAT_MODE=max \
ROAD_LOSS_BALANCE_MODE=none \
HIGHWAY_ROUTE_SAMPLE_TARGET=0 \
JOINT_BALANCE_REPEAT_MODE=none \
JOINT_BALANCE_DROP_MAJORITY=0 \
JOINT_TARGET_BALANCE_MODE=none \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_baseline/train.sh ddp
```

纯视觉 no-memory 基线：

```bash
PROMPT_MEMORY_MODE=hidden \
TRAIN_SAMPLING_MODE=transition_segments \
LORA_VISION_SCOPE=merger \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_baseline/train.sh ddp
```

每隔 N 个 optimizer step 跑 5-10 条 validation route 的 closed-loop probe：

```bash
CLOSED_LOOP_PROBE_STEPS=50 \
CLOSED_LOOP_PROBE_ROUTES=8 \
CLOSED_LOOP_PROBE_TRANSITION_CASES=64 \
CLOSED_LOOP_PROBE_WRITE_FRAMES=0 \
CLOSED_LOOP_PROBE_GPU_IDS=0 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_baseline/train.sh ddp
```

probe 会由 rank0 临时保存当前 adapter，然后连续调用 `eval.py --task full/road/event`；
其它 rank 等待 barrier，任一任务失败都会中止训练。输出位于当前 run 的
`closed_loop_probe/step_<STEP>/{eval_full,eval_road,eval_event}/`。其中 full 使用
`CLOSED_LOOP_PROBE_ROUTES` 条自然分布 route，road/event 使用
`CLOSED_LOOP_PROBE_TRANSITION_CASES` 个 label-balanced 变化邻域帧，避免自然抽样漏掉
HIGHWAY 或 UE 起跳，且 ROAD/EVENT 各自保持正负接近 1:1。
probe subprocess 会清掉 torchrun 的
`WORLD_SIZE/RANK/LOCAL_RANK/MASTER_*` 等分布式环境，强制按单进程 eval 跑；
`CLOSED_LOOP_PROBE_GPU_IDS` 可用于把 probe 固定到指定可见 GPU。

ROAD route 采样冒烟：

```bash
OUTPUT_DIR=checkpoints/sft_baseline_smoke \
MAX_STEPS=20 SAVE_STEPS=20 EVAL_STEPS=10 LOGGING_STEPS=1 \
MAX_EVAL_SAMPLES=64 \
HIGHWAY_ROUTE_SAMPLE_TARGET=0.5 \
JOINT_TARGET_BALANCE_MODE=exact \
ROAD_LOSS_BALANCE_MODE=none \
CLOSED_LOOP_PROBE_STEPS=0 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_baseline/train.sh ddp
```

启动日志会打印 `route_sampling`，训练日志重点看 `road_highway_rate` 均值是否在
0.45-0.55，零 HIGHWAY step 是否明显下降；同时确认 `event_ue_rate` 也在约
0.45-0.55，避免 ROAD route 采样把 EVENT 分布带偏。

默认 `LORA_VISION_SCOPE=off`，只训练语言侧 LoRA，不微调视觉塔。需要做视觉 LoRA
消融时再显式加 `LORA_VISION_SCOPE=merger|last4|all`。

默认仍使用防覆盖目录：

```text
checkpoints/sft_baseline_runs/run_<RUN_TAG>/
checkpoints/sft_baseline_runs/latest -> run_<RUN_TAG>
```

## Eval

完整 route 抽样：

```bash
GPU_IDS=0 python qwen3vl_local/sft_baseline/eval.py \
  --adapter-dir checkpoints/sft_baseline_runs/latest/final \
  --task full
```

四格均衡 full eval（用于判断模型本身，不被自然分布里的 NON_HIGHWAY/RE 多数类淹没）：

```bash
GPU_IDS=0 python qwen3vl_local/sft_baseline/eval.py \
  --adapter-dir checkpoints/sft_baseline_runs/latest/final \
  --task full \
  --full-balance-mode joint \
  --full-balance-cases-per-bin 128 \
  --no-write-frames
```

高速/非高速转折：

```bash
GPU_IDS=0 python qwen3vl_local/sft_baseline/eval.py \
  --adapter-dir checkpoints/sft_baseline_runs/latest/final \
  --task road \
  --max-transition-cases 128 \
  --transition-balance-mode label \
  --no-write-frames
```

RE/UE 转折：

```bash
GPU_IDS=0 python qwen3vl_local/sft_baseline/eval.py \
  --adapter-dir checkpoints/sft_baseline_runs/latest/final \
  --task event \
  --max-transition-cases 128 \
  --transition-balance-mode label \
  --no-write-frames
```

黑图消融：

```bash
GPU_IDS=0 python qwen3vl_local/sft_baseline/eval.py \
  --adapter-dir checkpoints/sft_baseline_runs/latest/final \
  --task full \
  --image-ablation black \
  --ablate-goal
```

阈值/PR 诊断（不重训）：`prediction-mode=score` 会对四个
`ROAD x EVENT` 组合做 teacher-forced 值 token 打分，再按 bias 选择二分类输出。
它仍是 closed-loop，下一帧 memory 来自当前 bias 下的预测。

```bash
GPU_IDS=0 python qwen3vl_local/sft_baseline/eval.py \
  --adapter-dir checkpoints/sft_baseline_runs/latest/final \
  --task event \
  --prediction-mode score \
  --event-logit-bias 0.0 \
  --max-transition-cases 128 \
  --no-write-frames \
  --output-dir checkpoints/sft_baseline_runs/latest/eval_results/event_score_bias_0
```

扫 EVENT bias：

```bash
CKPT=checkpoints/sft_baseline_runs/latest/final
for B in -4 -3 -2 -1 0 1 2 3 4; do
  GPU_IDS=0 python qwen3vl_local/sft_baseline/eval.py \
    --adapter-dir "$CKPT" \
    --task event \
    --prediction-mode score \
    --event-logit-bias "$B" \
    --max-transition-cases 128 \
    --no-write-frames \
    --output-dir "checkpoints/sft_baseline_runs/latest/eval_results/event_score_bias_${B}"
done
```

扫 ROAD bias 时把 `--task road` 和 `--road-logit-bias "$B"` 配对使用。

也可以直接跑一键诊断脚本。默认 `TRIAGE_PROFILE=fast`，只跑四格均衡 full、
ROAD label-balanced transition、EVENT label-balanced transition 三组，避免长时间 sweep。
如需旧的完整诊断，再显式设置 `TRIAGE_PROFILE=full`。输出
集中写到同一个 `triage_eval_<timestamp>/` 目录，并生成 `triage_summary.csv` /
`triage_summary.md`：

```bash
CKPT=checkpoints/sft_baseline_runs/run_v3_event_cooldown_probe3/final \
TRIAGE_PROFILE=fast \
BALANCED_CASES_PER_BIN=128 \
TRANSITION_CASES=128 \
GPU_IDS=0 bash qwen3vl_local/sft_baseline/run_triage_eval.sh
```

4 卡评估更快，脚本会按 `GPU_IDS` 数量自动改用 `torchrun`：

```bash
CKPT=checkpoints/sft_baseline_runs/run_v3_event_cooldown_probe3/final \
TRIAGE_PROFILE=fast \
BALANCED_CASES_PER_BIN=128 \
TRANSITION_CASES=128 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_baseline/run_triage_eval.sh
```

脚本结束时会打印对应的 `tar -czf ...` 打包命令。

no-memory adapter 评估时要使用同款 prompt：

```bash
GPU_IDS=0 python qwen3vl_local/sft_baseline/eval.py \
  --adapter-dir checkpoints/sft_baseline_runs/latest/final \
  --task full \
  --prompt-memory-mode hidden
```

eval 会校验 adapter config 里的 `prompt_memory_mode`。如果 hidden/no-memory adapter
忘记加 `--prompt-memory-mode hidden`，会直接报错，避免混用 prompt 得到无意义指标。

默认输出：

```text
checkpoints/sft_baseline_runs/latest/eval_results/<full_route|road_transition|event_transition>/<timestamp>/
  metrics.json
  frames.jsonl
  summary.md
  report.html
  tb/
```

`report.html` 是对齐 `sft_base` 的单文件可视化报告，直接打开即可看 ROAD/EVENT
二分类 confusion matrix 和 change matrix。`tb/` 是简易 TensorBoard eval 日志，包含
核心 scalar、ROAD 二分类 confusion matrix 文本、EVENT 二分类 confusion matrix 文本。
训练日志仍写在 run 目录的 `tb/` 下；统一查看：

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_baseline_runs/latest
```

关键指标看：

- `road_acc`
- `highway_f1`
- `event_acc`
- `ue_f1`
- `joint_acc`
- `road_change_f1`
- `event_change_f1`

训练 TB 额外关注：

- `train/unweighted_loss`：不带类别权重的 teacher-forced loss，便于和历史曲线对照
- `train/tf_road_acc_last_batch` / `train/tf_event_acc_last_batch`
- `train/tf_highway_recall_last_batch` / `train/tf_ue_recall_last_batch`
- `train/selected_frame_rate_last_batch`：片段采样实际选中帧比例
- `train/road_highway_rate_last_batch` / `train/event_ue_rate_last_batch`

第一批训练日志优先看：

- `selected_frame_rate_last_batch`：接近 1.0 说明片段采样覆盖过宽
- `road_highway_rate_last_batch` / `event_ue_rate_last_batch`：检查采样是否接近预期长尾增强
- `event_ue_weight_share_last_batch`：若仍接近 0.7，可继续把 `UE_EVENT_LOSS_WEIGHT` 往 1.5 降
- `tf_ue_recall_last_batch` 对比 closed-loop eval 的 `ue_recall`：差值就是 exposure bias
