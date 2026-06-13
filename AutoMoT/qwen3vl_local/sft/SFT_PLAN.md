# SFT Plan

SFT 子包负责 Qwen3-VL-4B-Instruct 的 LoRA 微调，让范式 A 文本 runner 更稳定地输出
`ANALYSIS / STATUS / SUBGOAL`。

## 1. 目标

给定 4 帧 RGB clip + memory（上一帧 GT status），让模型输出当前帧正确的
`STATUS`，并由状态机给出对应 `SUBGOAL`。

核心痛点是 anchor 早期帧过早推进：模型在 GT 转换点之前“反向编理由”，把
`STATUS` 提前切到下一阶段。

## 2. 当前文件职责

本子包只保留统一 LoRA SFT 路线，不再维护双轨命名与 ms-swift 训练入口：

- `build_dataset.py`：生成 `dataset_version="pending"` 的 train/val jsonl。
- `train.sh` / `train.py`：用 torch DDP + PEFT 直接注入 LoRA，并在 batch 内现场跑 teacher。
- `build_teacher.py`：可选离线 dump teacher ANALYSIS，仅用于 review / 统计，不参与默认训练。
- `eval.py` / `probe.py`：加载 base 或 LoRA 后做指标评估与 case-level dump。
- `check_loss_mask.py` / `inspect_teacher_outputs.py`：静态检查 loss 权重与 teacher 输出质量。

训练不再保留 teacher cache + manifest 复用机制：每次启动训练，frozen base Qwen
在 train batch 内现场跑 ANALYSIS teacher，不写盘。

## 3. 数据 schema

每条 jsonl 样本：

```json
{
  "scenario": "Accident",
  "run_id": "Town03_Rep0_route_001783_route0_01_11_02_37_46",
  "anchor": 12,
  "prev_anchor": 8,
  "images": ["/data/.../rgb/0009.jpg", "...", "...", "/data/.../rgb/0012.jpg"],
  "messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "<image><image><image><image>\n<user prompt with MEMORY>"},
    {"role": "assistant", "content": "ANALYSIS: __TEACHER_PENDING__\nSTATUS: initial\nSUBGOAL: hazard_detect"}
  ],
  "is_transition_sample": false,
  "dataset_version": "pending",
  "teacher_meta_input": {
    "target_status": "initial",
    "target_subgoal": "hazard_detect",
    "memory_in_status": "initial",
    "transition": "keep"
  }
}
```

约束：

- `messages[0]` 来自 `prompt_pipeline.build_system_prompt()`。
- user `[MEMORY]` 中的 `STATUS` 是 `prev_anchor` 的 GT，防泄漏当前帧标签。
- assistant `STATUS` 是当前 `anchor` 的 GT；`ANALYSIS` 是 `__TEACHER_PENDING__` 占位。
- `images` 长度固定为 4，按 oldest -> newest 排序。
- `teacher_meta_input` 给 `train.py` / `build_teacher.py` 拼 PRIVILEGED prompt 用，
  让 frozen base 在“看到 GT”的前提下输出更高质量 ANALYSIS。

## 4. 数据生成（`build_dataset.py`）

1. 读 `keyframes_all_scenarios.json`，只保留 run status 为 `Completed` / `Perfect` 的样本。
2. 根据 initial / middle / final 帧号构造闭区间状态时间轴。
3. 采样保持类和推进类：
   - 保持类避开转换帧前的 buffer 帧；
   - 推进类保留 GT 转换帧后 `K` 帧窗口内的跨段样本；
   - 默认推进类目标占比 35%。
4. 按 run_id 划分 train / val，避免同一 route 的相邻帧跨集合泄漏。
5. assistant ANALYSIS 段写 `__TEACHER_PENDING__`。

## 5. 训练 loop（`train.py`）

LoRA 注入方式：

```python
from peft import LoraConfig, get_peft_model
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
model = get_peft_model(base_model, lora_cfg)
```

每个 train batch（per_device_batch_size=1，逐样本处理）：

1. **Phase A — teacher**：
   ```python
   peft_model = unwrap_ddp(model)
   base_model = peft_model.get_base_model()
   with peft_model.disable_adapter():
       out = base_model.generate(prompt_with_PRIVILEGED, images, max_new_tokens=256)
   analysis_text, fallback = postprocess_teacher(out)
   ```
   注意：adapter 开关由 PEFT 管，但 generate 调底层 Qwen，避免
   `PeftModel.generate` 在 Qwen3-VL M-RoPE / `prepare_inputs_for_generation`
   路径上的错位问题。
2. **Phase B — student**：
   - 拼完整 assistant 文本：
     `"ANALYSIS: {analysis_text}\nSTATUS: {gt_status}\nSUBGOAL: {gt_subgoal}<|im_end|>\n"`
   - tokenize 后拼接到 prompt token 序列；labels 上 prompt 段 = -100，assistant 段 = self
   - 算 per-token weight：
     - prompt 段 = 0
     - ANALYSIS body = `SFT_ANALYSIS_WEIGHT`（默认 0.3）
     - 起手 `ANALYSIS:` 字面、段切换 `\nSTATUS:` / `\nSUBGOAL:` 字面、STATUS event_name、SUBGOAL event_name、tail / EOS 全部 = 1.0
   - `loss = sum(F.cross_entropy(reduction='none') * weight) / sum(weight)`
3. 反向 + grad clip + AdamW step + cosine LR。

DDP：用 `torch.distributed` + `DistributedDataParallel(find_unused_parameters=True)`，
每 rank 处理 `DistributedSampler` 切到自己的 batch；teacher generate / student forward
都在各 rank 自己显存里跑。

DDP 健壮性补丁（v2 → 当前路线）：

- **同进同退**：`_ddp_all_ranks_valid` 用 `all_reduce(MIN)` 让所有 rank 对"本样本是否
  进入 backward" 达成一致。单条坏图 / assistant 超长不再让多卡训练整体 raise，
  只丢这一个 micro-batch 继续训。
- **no_sync()**：前 `(grad_accum - 1)` 个 micro-step 包在 `bundle.model.no_sync()`
  里，只在最后一个 micro-step / 尾批 `finish_optimizer_step` 触发一次 all-reduce(AVG)，
  把每个 optimizer step 的 all-reduce 次数从 grad_accum 次降到 1 次。
- **`use_reentrant=False`**：`gradient_checkpointing_enable` 强制非 reentrant，
  规避老 transformers + `find_unused_parameters=True` 反传图错位坑（loss=NaN /
  反传卡死）。
- **尾批不放大梯度**：epoch 末尾 `accum_count < grad_accum` 的尾批不再做
  `expected_total / n_total` 放大，rescale ≡ 1.0，避免 cosine 末段尾批 step 比
  正常 step 强 `grad_accum / accum_count` 倍。
- **tokenize 边界 sanity**：进程第一条样本走 `build_student_inputs` 时跑一次
  `tokenize(prompt) ⊕ tokenize(assistant) == tokenize(prompt + assistant)` 验证，
  防止 Qwen BPE 在边界 merge 让训练 ≠ 推理。通过即 silent，无每 batch 开销。

## 6. 关键超参（默认）

| 项 | 默认 | 说明 |
|---|---:|---|
| rank / alpha | 16 / 32 | 当前默认 LoRA 容量 |
| lora_dropout | 0.1 | |
| learning rate | 3e-5 | 与当前 per-token 监督量匹配 |
| max_length | 3584 | teacher ANALYSIS 80-150 token + system + user + 视觉 token |
| num_epochs | 2 | |
| per_device_batch_size | 1 | batch 内 teacher 串行；DDP 用样本并行 |
| grad_accum | 2 | 等效 batch size = 2 * world_size |
| save_steps / eval_steps | 10000 | |
| save_total_limit | 3 | |
| SFT_ANALYSIS_WEIGHT | 0.3 | env override |
| SFT_TEACHER_MAX_NEW_TOKENS | 256 | env override |
| SFT_TEACHER_TEMPERATURE | 0.0 (greedy) | env override |

训练 launcher 默认在 `OUTPUT_DIR/run_<RUN_TAG>/` 写本次 run，base 层维护
`latest` symlink；`HF_HOME` 固定在 base 层。每个训练 run 目录追加 `log.txt`
保存本次终端 stdout/stderr。

显式 pin 卡统一在训练命令前置 `GPU_IDS=0`（单卡）或 `GPU_IDS=0,1,2,3`（4 卡 DDP）；
`GPU_IDS` 非空时跳过 nvidia-smi 自动选址，DDP 卡数从逗号数推断，`DDP_GPU_COUNT` 被忽略。

## 7. Teacher 节奏成本

每个 train step 都要跑一次 frozen base 4B greedy generate（ANALYSIS body 通常 80-150 token），
等效训练时间相比 LoRA-only ≈ 3-4 倍：

- 5k samples × 2 epoch × 2 卡 ≈ 8-10h
- 想跳过现场 teacher 做链路 sanity：直接 `--skip-teacher`（或 train.sh sanity 模式 /
  `SKIP_TEACHER=1`），ANALYSIS 全部替换成固定 fallback。**注意：用这条路径训出
  的 LoRA 不可用于生产**，仅用来排查 student forward / DDP / 优化器是否正常。
- 想要离线物化 teacher 输出做样本审计，可以跑 `build_teacher.py` 得 materialized
  jsonl（不参与训练，仅用于 review）。

## 8. 评估

`eval.py` 输出四个核心指标：

| 指标 | 计算 | 目标 |
|---|---|---|
| `keep_accuracy` | keep 样本 STATUS == GT | 越高越好 |
| `advance_accuracy` | advance 样本 STATUS == GT | 越高越好 |
| `early_advance_rate` | keep 样本 STATUS == next(GT) | 越低越好 |
| `anchor12_sanity` | 典型早推进 fail case 是否回到 initial | 必须通过 |

加载 LoRA 默认 `merge_and_unload`，避免 PeftModel wrapper 在 Qwen3-VL 上
generation 第二步起出乱码（M-RoPE + `prepare_inputs_for_generation` 不兼容）。
`--max-gen-tokens` 默认 256，保证 ANALYSIS 段不挤掉 STATUS / SUBGOAL。
eval 终端输出追加到 `<save-root>/eval/log.txt`，probe 终端输出追加到
`<save-root>/eval_cases/log.txt`。

## 9. 文件清单

| 文件 | 用途 |
|---|---|
| `SFT_PLAN.md` | 本设计文档 |
| `SFT_RUN.md` | 运行手册 |
| `build_dataset.py` | pending jsonl 生成 |
| `build_teacher.py` | 可选离线 teacher 物化（manifest 复用机制已去除） |
| `train.py` | torch DDP + PEFT + 内置加权 loss |
| `train.sh` | bash launcher（GPU 选址 / run 子目录 / log tee） |
| `eval.py` / `probe.py` | 共享评估与 case dump |
| `check_loss_mask.py` | token 级 loss sanity（验 train.py 内置 mask） |
| `inspect_teacher_outputs.py` | teacher 输出抽检（支持 `--live` 现场重跑） |
| `../tb_serve.sh` | 通用 TensorBoard launcher |

## 10. 风险

| 风险 | 处理 |
|---|---|
| PEFT + Qwen3-VL generate 不兼容 | train teacher 调底层 Qwen generate；eval / probe / runner 默认 `merge_and_unload` |
| teacher 套话或太短 | 用 `inspect_teacher_outputs.py --live` 看样本，必要时改 teacher prompt |
| 训练时间 ~3x base LoRA | 用 `--check` 模式快速验链路，全量训练前先在小集 spot-check |
| DDP teacher generate 卡死 | 所有 rank 都进 adapter-disabled teacher 路径；不在 rank0 内做单独 generate |
