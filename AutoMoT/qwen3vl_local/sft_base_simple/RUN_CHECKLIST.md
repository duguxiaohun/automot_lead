# SFT Base Simple 验收与执行清单

> 生成日期：2026-08-06
> 适用范围：`AutoMoT/qwen3vl_local/sft_base_simple/`
> 运行目录：远端 `AutoMoT/`（所有命令都相对该目录）
> 硬件假设：4 张 GPU

---

## 0. 代码评审结论

### 0.1 已修复项

| 编号 | 问题 | 修复方式 | 代码位置 |
|---|---|---|---|
| P0 | early-UE 期间 memory 可能落到 UE，形成 copy-shortcut | guard 从「扰动前 clean 值」移到「扰动后结果 family」，early-UE 内若结果是 UE，以 `early_ue_resample_prob=0.70` 重抽为 RE（无 RE 候选则 UNKNOWN） | `memory_curriculum.py` `RouteMemoryCorruptor.corrupt()` 末尾 |
| P4 | 概率归一化到 1.0 导致 `keep=0`，UE span 内永远看不到合法 UE 延续先验 | early-UE 分支改用 `ue_cap = 0.85`，保留 ≥15% keep 地板 | `memory_curriculum.py` `_scaled_event_probs()` |
| N1 | TB 的 `event_ue_weight_sum` / `event_re_weight_sum` 走旧 `_event_loss_weights`（inverse-sqrt + `single_candidate_re_scale=0.1`），与真实 loss 的扁平 1.0/1.0 不一致 | 统计处改为直接使用扁平权重，与真实训练路径一致 | `train.py` L1314 附近 |

### 0.2 关于 `wrong` 方向的重要说明

`memory_curriculum.wrong_event_for_frame()` 是**相对当前帧 GT 取反**，不是相对传入的 memory 值取反：

```python
def wrong_event_for_frame(rng, current, frame, rs_label):
    del current                      # 完全不看传入的 memory 值
    if bool(getattr(frame, "abnormal", False)):
        return default_regular_event_for_rs(str(rs_label))   # GT=UE -> wrong 给 RE
    ...                                                       # GT=RE -> wrong 给 UE
```

因此在 UE span 内任意一帧（含 `early_ue_age=0`），`wrong` 只会产出 RE。
`test_memory_curriculum.py` 中 `age0_ue == 0` 的断言正是由这个性质保证。

post-perturbation guard 的真实作用是**加固**：堵住"某个 wrong/keep segment 在进入 UE span 之前就已抽定为 UE 值、跨帧携带进 span"这条路径。

### 0.3 数值预期

默认参数下（`event_wrong_prob=0.35`、`event_unknown_prob=0.35`、
`early_ue_wrong_scale=1.75`、`early_ue_unknown_scale=1.35`、`ue_cap=0.85`）：

- 放大后原始值：`wrong = 0.6125`、`unknown = 0.4725`，合计 `1.085 > 0.85`
- 归一化系数：`0.85 / 1.085 ≈ 0.7834`
- **effective**：`wrong ≈ 0.480`、`unknown ≈ 0.370`、`keep ≈ 0.150`
- `early_ue_age = 0`（GT=UE，clean memory=RE）：结果只能是 `RE / UNKNOWN / HIDDEN`，UE 概率为 **0**
- `early_ue_age ≥ 1`（clean memory=UE）：残留 UE memory ≈ `0.85 × 0.15 × (1 − 0.70) ≈ 3.8%`

这满足原始需求："前三帧是 RE、当前帧是 UE，那么 memory 应该是 RE"，
同时保留少量合法 UE 延续先验，避免模型被训成"看到 `PREVIOUS_EVENT=UE` 就必须改口"。

### 0.4 仍开放的次要项（不阻塞训练）

| 编号 | 内容 | 处理方式 |
|---|---|---|
| P1 | eval joint 模式从 frame 0 闭环 rollout，`rollout_frames / frames` 可能很高；`--prediction-mode score` 再 ×4 | 已有 `--max-frames-per-route` 可封顶；先按 §4 量实际比值再定预算 |
| P2 | 稀有桶（尤其 `HIGHWAY:UE`）在 `_balance_work_by_joint_target` 中被 `items[idx % len(items)]` 复制到 8 份，同 step 梯度相关 | 开训后盯 `train/fourbin_highway_ue_last_batch`，必要时调大 `FOURBIN_ROUTES_PER_BATCH` |
| P3 | change matrix 覆盖全部 rollout 帧，但 `frames.jsonl` 只写受评帧，无法离线复算；且存在 route 前缀偏置 | 已有 `change_metric_scope: "closed_loop_rollout_adjacent_frames"` 元数据标注口径，作为 baseline 可接受 |
| — | 训练 memory 是 GT teacher-forced、eval 是 student 闭环，存在 exposure bias | 设计如此；解读掉点时需计入 |

---

## 1. 单测（CPU，秒级）

```bash
python qwen3vl_local/sft_base_simple/check_loss_mask.py
python qwen3vl_local/sft_base_simple/test_memory_curriculum.py
python qwen3vl_local/sft_base_simple/test_prompt_snapshots.py
python qwen3vl_local/sft_base_simple/test_train_resume.py
python qwen3vl_local/sft_base_simple/test_dataset_contract.py
python qwen3vl_local/sft_base_simple/test_eval_metrics.py
python qwen3vl_local/sft_base_simple/test_eval_candidates.py
python qwen3vl_local/sft_base_simple/test_regular_remap.py
```

全部通过才进入下一步。

---

## 2. 建数据集

`DATASET_VERSION` 已改为 `sft_base_simple_highway_reue_fourbin_v1`，旧目录不能复用。

先跑 smoke：

```bash
python qwen3vl_local/sft_base_simple/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_base_simple_data_smoke \
  --max-routes 4 \
  --max-frames-per-route 16
```

通过后建全量（耗时较长，建议 `tmux` / `nohup`）：

```bash
python qwen3vl_local/sft_base_simple/build_dataset.py \
  --collection-dir keyframe_filter/collection_output \
  --data-root lead_data \
  --output-dir checkpoints/sft_base_simple_data
```

---

## 3. check 模式走通链路

```bash
GPU_IDS=0 bash qwen3vl_local/sft_base_simple/train.sh check
```

**这一步是本次改动的主要验收点，重点看启动日志：**

- `[memory]` 打印的 early-UE **effective** 概率，应为
  `wrong ≈ 0.480 / unknown ≈ 0.370 / keep ≈ 0.150`
  （若打印成 `0.6125 / 0.4725`，说明 `ue_cap` 归一化没生效）
- 四桶实际数量 `fourbin_*_last_batch`
- `fourbin_missing_bins` 是否为空

---

## 4. 小规模 eval：量 rollout 成本

用 base（不加 adapter）先量比值，决定正式 eval 预算：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base_simple/eval.py \
  --task full \
  --full-balance-cases-per-bin 4 \
  --max-frames-per-route 200 \
  --no-write-frames \
  --output-dir checkpoints/sft_base_simple_eval_costprobe
```

看 `metrics.json` 里的 `rollout_frames / frames`：

- 比值 ≤ 40：正式 eval 可不加限制
- 比值 > 40：正式 eval 带上 `--max-frames-per-route`

---

## 5. 正式训练（4 卡）

```bash
OUTPUT_DIR=checkpoints/sft_base_simple_runs \
CLOSED_LOOP_PROBE_STEPS=50 \
CLOSED_LOOP_PROBE_FOURBIN_CASES=128 \
CLOSED_LOOP_PROBE_WRITE_FRAMES=0 \
CLOSED_LOOP_PROBE_GPU_IDS=0 \
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_base_simple/train.sh ddp
```

TensorBoard：

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/sft_base_simple_runs/latest
```

### 前 200 step 必盯指标

| 指标 | 期望 | 异常含义 |
|---|---|---|
| `memory/early_ue_event_ue_rate_last_batch` | ≈ 0.04 | 明显偏高 → post-guard 没生效 |
| `memory/early_ue_event_re_rate_last_batch` | 显著高于 ue_rate | 偏低 → keep 地板被压掉 |
| `train/fourbin_highway_ue_last_batch` | 稳定 = 8 | 长期靠复制凑数 → 调大 `FOURBIN_ROUTES_PER_BATCH` |
| `train/fourbin_missing_bins` | 空 | 非空 → 某桶在 16-route 内取不到样本 |
| `road_acc` / `highway_f1` | 稳步上升 | — |
| `event_acc` / `ue_f1` | 稳步上升 | `ue_f1` 长期为 0 → UE 侧监督失效 |

---

## 6. 训练后 eval

四格均衡 full eval（默认测试口径）：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base_simple/eval.py \
  --adapter-dir checkpoints/sft_base_simple_runs/latest/final \
  --task full \
  --full-balance-cases-per-bin 64 \
  --output-dir checkpoints/sft_base_simple_eval_final
```

整 route 闭环（看完整相邻帧 change matrix）：

```bash
GPU_IDS=0 python qwen3vl_local/sft_base_simple/eval.py \
  --adapter-dir checkpoints/sft_base_simple_runs/latest/final \
  --task full \
  --full-balance-mode none \
  --sample-routes 64 \
  --output-dir checkpoints/sft_base_simple_eval_fullroute
```

> 注意：`--initial-memory-noise none` 与 joint eval 组合会被拒绝，
> 这是为了防止把当前帧 GT 写进 prompt 造成泄漏。

---

## 7. 入库

代码目前是 untracked 状态。§3 跑通后按精确路径 add：

```bash
git status
git add AutoMoT/qwen3vl_local/sft_base_simple/
git commit -m "<一句话说明本次改了什么、为什么>"
```

`push` 前先确认。目录内 `.gitignore` 已排除 `__pycache__/`。
