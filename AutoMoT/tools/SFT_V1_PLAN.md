# SFT v1 训练计划 — Qwen3-VL-4B-Instruct LoRA

> **目标**：让范式 A runner 在 anchor 早期帧（GT 转换点之前）严格保持当前 STATUS，
> 不再"反向编理由"提前推进。v1 只学 STATUS（顺带 SUBGOAL token），ANALYSIS 完全
> 不算 loss，等 v1 收敛后再做 v2 的 ANALYSIS 蒸馏。

---

## 1. 训练目标（一句话）

给定 4 帧 RGB clip + memory（含**上一帧**的 GT status），让模型输出的
`STATUS:` 字段等于**当前帧**的 GT status。其它一切（ANALYSIS 内容、视觉特征）v1 不动。

## 2. Loss 组成（按 token 拆开）

每条样本的 assistant 输出固定三段：

```
ANALYSIS: Observations recorded.
STATUS: <event_name>
SUBGOAL: <event_name>
```

| Token 段 | labels | 权重 | 备注 |
|---|---|---|---|
| `ANALYSIS: Observations recorded.\n` 全段 | **-100**（mask） | 0 | v1 不学 analysis 内容；占位字符串只为保住三段输出格式 |
| `STATUS: ` 字面 | 算 | 1.0 | 学输出格式 |
| `<event_name>` for STATUS | 算 | 1.0 | **主信号** |
| `\nSUBGOAL: ` 字面 | 算 | 1.0 | 学输出格式 |
| `<event_name>` for SUBGOAL | 算 | 1.0 | 由 STATUS 推导，但仍算 loss 强化关联；token 数少天然权重小 |

实现：在 build 阶段 assistant content 写完整三段；ms-swift 3.12.x 训练侧用
`tools/sft_v1_loss_scale_plugin.py` 注册 `sft_v1_analysis_mask` 策略，把
ANALYSIS 段权重设为 0。

> 备注：ms-swift 3.12.6 的 `--loss_scale` 只接受已注册策略名，不接受任意
> JSON regex。插件内部仍用 `ANALYSIS:.*?(?=\nSTATUS:)` 做文本切分。

## 3. 数据集 schema

每条样本（jsonl 一行）：

```json
{
  "scenario":    "Accident",
  "run_id":      "Town03_Rep0_route_001783_route0_01_11_02_37_46",
  "anchor":      12,
  "prev_anchor": 8,
  "images":      ["/data/.../rgb/0009.jpg", "...", "...", "/data/.../rgb/0012.jpg"],
  "messages": [
    {"role": "system",    "content": "<完整 system prompt>"},
    {"role": "user",      "content": "<image><image><image><image>\n<完整 user prompt 含 [MEMORY]>"},
    {"role": "assistant", "content": "ANALYSIS: Observations recorded.\nSTATUS: initial\nSUBGOAL: hazard_detect"}
  ],
  "is_transition_sample": false
}
```

**关键约束**：
- `messages[0]` (system) 必须等于 `prompt_pipeline.build_system_prompt()` 当前版本
- `messages[1]` (user) 内嵌的 `[MEMORY]` 块 `STATUS:` 字段 = **prev 帧** GT status（防 leak）
- `messages[2]` (assistant) `STATUS:` 字段 = **当前帧** GT status
- `images` 长度 = 4，按 oldest → newest 排序

## 4. 数据生成流程

`AutoMoT/tools/build_sft_dataset_v1.py` 完成（**纯 CPU、不需要 GPU**）：

```
对 keyframes_all_scenarios.json["runs"] 中每条 run：
    若 run["status"] not in {"Completed", "Perfect"} → 跳过
    取 diagnostics.seconds_per_frame 和 total_frames
    构造 status_timeline[f] = lookup_status_at_frame(f) 闭区间映射：
        [0, frame_m1-1]               → "initial"
        [frame_m1, frame_m2-1]        → middle[0]
        [frame_m2, frame_m3-1]        → middle[1]
        [frame_m3, frame_final-1]     → middle[2]
        [frame_final, total_frames-1] → "final"

    采样候选：
        转换帧 ±2 → 全部丢弃（buffer）
        对每个 status 段：取段内所有可用帧
        对每个 GT 转换帧 f_t：构造推进类样本 (prev=f_t-K 落在上一段, curr=f_t 落在新段)

    stratified 取样到目标配比：
        推进类 25% / 保持类 75%（保持类 4 段平均分）
        每场景目标 200 样本

    每个采样帧 f：
        构造 anchor=f, prev_anchor=f-K=f-4
        images = [rgb 路径 for f-3, f-2, f-1, f]（按 oldest→newest）
        memory_in.status   = status_timeline[prev_anchor]
        memory_in.subgoal  = next_event_in_sequence(memory_in.status)
        target_status      = status_timeline[anchor]
        target_subgoal     = next_event_in_sequence(target_status)
        写入 jsonl
```

**train / val 划分**：按 run_id 划，**不按 sample 划**（防 leak）。10% run 留作 val。

## 5. LoRA 配置

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",     # attention 4 件套
        "gate_proj", "up_proj", "down_proj",        # MLP 3 件套
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

冻结：
- ViT 整个 vision tower（`visual.*`）
- vision-text merger / projector（`merger.*` / `visual_token_merger.*`，看模型实际结构）
- embedding 与 lm_head
- 所有非 LoRA 参数

ms-swift 命令行：
```
--train_type lora
--target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
--lora_rank 16 --lora_alpha 32 --lora_dropout 0.05
--freeze_vit true
```

可训练参数预估：~30M（占 4B 模型 ~0.75%）。

## 6. 训练超参

| 项 | DDP (8×H20) | 单卡 |
|---|---|---|
| precision | bf16 | bf16 |
| num_train_epochs | 3 | 3 |
| per_device_train_batch_size | 2 | 4 |
| gradient_accumulation_steps | 2 | 2 |
| **等效 batch_size** | 32 | 8 |
| learning_rate | 1e-4 | 1e-4 |
| warmup_ratio | 0.03 | 0.03 |
| lr_scheduler_type | cosine | cosine |
| weight_decay | 0.01 | 0.01 |
| max_length | 3072 | 3072 |
| gradient_checkpointing | true | true |
| save_steps / eval_steps | 100 | 200 |
| logging_steps | 5 | 10 |

总 step 数（DDP）：8400 train × 0.9 / 32 × 3 ≈ **710 step**，单 H20 ~25 min/epoch，约 **1.5 小时**全部跑完。

## 7. 8×H20 DDP 启动

```bash
# DDP 模式
NPROC_PER_NODE=8 swift sft \
    --model "checkpoints/Qwen3-VL-4B-Instruct" \
    --dataset "checkpoints/sft_v1_data/train.jsonl" \
    --val_dataset "checkpoints/sft_v1_data/val.jsonl" \
    --train_type lora \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
    --freeze_vit true \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --bf16 true \
    --gradient_checkpointing true \
    --max_length 3072 \
    --output_dir "checkpoints/sft_v1_lora" \
    --logging_steps 5 --save_steps 100 --eval_steps 100 \
    --save_only_model true \
    --external_plugins "tools/sft_v1_loss_scale_plugin.py" \
    --loss_scale "sft_v1_analysis_mask"
```

`save_only_model=true` → 只存 LoRA adapter（~120 MB / step），不存 optimizer state。
`freeze_vit=true` → 显式冻结 ViT，配合 LoRA `target_modules` 只命中 LLM decoder。

完整命令在 `AutoMoT/tools/sft_v1_train.sh`。

## 8. 评估

`AutoMoT/tools/eval_sft_v1.py` 在 val.jsonl 上跑推理，输出 4 个指标：

| 指标 | 计算 | v1 目标 |
|---|---|---|
| **保持类 accuracy** | `is_transition_sample=False` 的样本里 STATUS == GT 的比例 | ≥ 95% |
| **推进类 accuracy** | `is_transition_sample=True` 的样本里 STATUS == GT 的比例 | ≥ 60% |
| **提前推进率** | 保持类样本里 STATUS == next(GT) 的比例 | ≤ 5% |
| **anchor=12 sanity** | 跑原 fail case route，看 STATUS 是否回 `initial` | 必须通过 |

eval 时打开 `cache_system_prompt=True`：所有样本 system prompt 相同，prefill 一次复用 KV cache，可省约 50% 推理时间。

## 9. v2（v1 收敛后再启动，现在不细化）

锁两条原则：
1. **ANALYSIS 真值生成**用 v1 LoRA + 当前帧 GT status 喂 prompt 自蒸馏（不是冻结 base）
2. **v2 LoRA 在 v1 LoRA 上继续训**，rank 不变，放开 ANALYSIS token 的 loss

## 10. 文件清单（CLAUDE.md / AGENTS.md 白名单已加）

| 文件 | 用途 | 在哪儿跑 |
|---|---|---|
| `AutoMoT/tools/SFT_V1_PLAN.md` | 本文件 | — |
| `AutoMoT/tools/build_sft_dataset_v1.py` | 解析 keyframes + LEAD route → jsonl | 本地或远程，纯 CPU |
| `AutoMoT/tools/sft_v1_train.sh` | swift sft 启动脚本（DDP + 单卡两个模式） | 远程 |
| `AutoMoT/tools/sft_v1_loss_scale_plugin.py` | ms-swift 3.12.x 自定义 loss_scale：mask ANALYSIS、保留 STATUS/SUBGOAL loss | 远程 |
| `AutoMoT/tools/eval_sft_v1.py` | 离线 STATUS 评估 + anchor=12 sanity | 远程 |
| `AutoMoT/tools/check_loss_mask.py` | 纯 tokenizer 可视化 ANALYSIS / STATUS / SUBGOAL 段 token 级 mask | 本地或远程，纯 CPU |

## 11. 已知风险与回退路径

| 风险 | 触发条件 | 回退 |
|---|---|---|
| ms-swift 自定义 loss_scale 插件未注册 | `KeyError: 'sft_v1_analysis_mask'` 或训练日志里找不到 `external_plugins` | 确认从 `AutoMoT/` 目录运行，且 `tools/sft_v1_loss_scale_plugin.py` 存在；必要时用绝对路径传 `--external_plugins` |
| ms-swift loss_scale 在多模态 template 上不生效 | 训前几个 step `loss` 完全不下降，或 ANALYSIS 段也产生强梯度 | 再退一步写 `AutoMoT/tools/sft_v1_preprocessor.py` 手动 mask labels |
| 训练侧 `<image>` 文本占位与 runner / engine.build_messages 的 structured image content 在 chat template 展开后**不完全一致**（vision token 数差、位置偏） | LoRA 训完后 eval STATUS accuracy 远低于训练 loss 体现的水平；或 anchor12_sanity 完全没改善 | 训完第一个 checkpoint 后，对**同一条** val sample 分别走（a）swift 训练 collator 的 input_ids 与（b）runner engine.generate 的 prefill input_ids，diff token 序列。Mismatch ≤ 5 token 可忽略；> 20 必须排查模板差异 |
| 插件 regex 与 `PLACEHOLDER_ANALYSIS` 不匹配（占位句改了 regex 没改、或反过来） | 训练初始 loss 异常（< 3 表示全段被 mask；> 12 表示 ANALYSIS 也算 loss） | 训前先跑 `python tools/check_loss_mask.py` 看 token 级 mask 是否对；再跑 `bash tools/sft_v1_train.sh check` 看 2 step loss 数值 |
| `Completed/Perfect` filter 后某场景样本不够 200 | 数据生成时 warning | `--samples_per_scenario` 调低，或允许该场景全收 |
| 推进类样本天然稀少（每 route 仅 4 转换帧） | val 集推进类样本 < 30 | 取消"按 run_id 划 val"改"按 scenario 内 8:2 划"，但 leak 风险上升 |
| 训完保持类 accuracy 高但推进类 < 50% | 模型学过头变成"永远不推进" | 调推进类配比到 35%；或加 v1.5 阶段把推进类反复训 |
| 8 卡 DDP NCCL OOM 或卡顿 | H20 NVLink 带宽 / NCCL 配置 | 退到 4 卡或 2 卡 DDP；`per_device_train_batch_size` 降到 1 |
