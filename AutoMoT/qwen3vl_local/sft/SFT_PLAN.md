# SFT Plan

SFT 子包负责 Qwen3-VL-4B-Instruct 的 LoRA 微调，让范式 A 文本 runner 更稳定地输出
`ANALYSIS / STATUS / SUBGOAL`。v1/v2 共用采样、训练 launcher 风格、评估和 probe；
区别集中在 ANALYSIS 段是否参与监督。

## 1. 目标

给定 4 帧 RGB clip + memory（上一帧 GT status），让模型输出当前帧正确的
`STATUS`，并由状态机给出对应 `SUBGOAL`。

核心痛点是 anchor 早期帧过早推进：模型在 GT 转换点之前“反向编理由”，把
`STATUS` 提前切到下一阶段。

## 2. v1 / v2 差异

| 项 | v1 | v2 |
|---|---|---|
| 数据模式 | `build_sft_dataset_v1.py` 默认 | `build_sft_dataset_v1.py --mode v2` |
| `dataset_version` | 不写，视作 v1 | `v2_pending` 先占位，teacher 后改为 `v2` |
| ANALYSIS | 固定 `Observations recorded.` | frozen Qwen + PRIVILEGED prompt 生成真实分析 |
| loss | 只训 STATUS/SUBGOAL 事件名 | ANALYSIS body 0.3，结构字面和事件名 1.0 |
| 训练脚本 | `sft_v1_train.sh` | `sft_v2_train.sh` |
| 评估/probe | `eval_sft_v1.py` / `probe_sft_v1.py` | 同一套，显式传 runtime teacher val |

v1 是稳定 STATUS/SUBGOAL 的保守基线；v2 在 v1 目标上增加视觉分析蒸馏，目标是让
keep 样本的 `early_advance_rate` 进一步下降。

## 3. 数据 schema

每条 jsonl 样本包含：

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
    {"role": "assistant", "content": "ANALYSIS: ...\nSTATUS: initial\nSUBGOAL: hazard_detect"}
  ],
  "is_transition_sample": false
}
```

关键约束：

- `messages[0]` 来自 `prompt_pipeline.build_system_prompt()`。
- user `[MEMORY]` 中的 `STATUS` 是 `prev_anchor` 的 GT，防止泄漏当前帧标签。
- assistant `STATUS` 是当前 `anchor` 的 GT。
- `images` 长度固定为 4，按 oldest -> newest 排序。

## 4. 数据生成

`build_sft_dataset_v1.py` 同时承载 v1/v2：

1. 读 `keyframes_all_scenarios.json`，只保留 run status 为 `Completed` / `Perfect` 的样本。
2. 根据 initial / middle / final 帧号构造闭区间状态时间轴。
3. 采样保持类和推进类：
   - 保持类避开转换帧前的 buffer 帧；
   - 推进类保留 GT 转换帧后 `K` 帧窗口内的跨段样本；
   - 默认推进类目标占比 35%。
4. 按 run_id 划分 train / val，避免同一路线相邻帧跨集合泄漏。
5. v1 写固定 ANALYSIS；v2 写 `__TEACHER_PENDING__`，等待 teacher 物化。

## 5. Loss 设计

v1 assistant：

```text
ANALYSIS: Observations recorded.
STATUS: <event_name>
SUBGOAL: <event_name>
```

v1 的 `sft_v1_analysis_mask` 只保留两个事件名 token 段，其他权重为 0。

v2 assistant：

```text
ANALYSIS: <teacher visual analysis>
STATUS: <event_name>
SUBGOAL: <event_name>
```

v2 的 `sft_v2_analysis_supervised`：

- ANALYSIS body 默认权重 0.3，可用 `SFT_V2_ANALYSIS_WEIGHT` 调整。
- `ANALYSIS:`、`\nSTATUS:`、`\nSUBGOAL:` 字面和 STATUS/SUBGOAL 事件名权重 1.0。
- tail/EOS 参与 loss，避免三段输出被截断后无终止信号。

旧版 v2 “结构字面 mask=0” 是已知陷阱，不要恢复。

## 6. Teacher 物化

v2 pending 不直接训练。`sft_v2_train.sh` 在进入 swift 前检查 runtime teacher cache：

- 完整 cache + manifest 匹配：复用。
- manifest 缺失、行数不匹配、model_dir / seed / 生成参数不匹配：清理后重物化。
- debug 的 `--max-samples N` 不写正式 manifest，避免半截 cache 被误复用。
- `check` 模式默认写独立 `runtime_teacher_check_data/`，不污染正式
  `runtime_teacher_data/`。

teacher prompt 带 PRIVILEGED 信息，仅用于生成 ANALYSIS 真值，不写回 pending 源数据。

## 7. LoRA 与训练

两版都只训 language decoder 的 LoRA：

```text
q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
```

冻结视觉塔、embedding、lm_head 和所有非 LoRA 参数。默认配置：

| 项 | v1 | v2 |
|---|---:|---:|
| rank / alpha | 16 / 32 | 16 / 32 |
| dropout | 0.1 | 0.1 |
| learning rate | 5e-5 | 3e-5 |
| max_length | 3072 | 3584 |
| num_epochs | 2 | 2 |
| save/eval | steps | steps |

训练 launcher 默认在 `OUTPUT_DIR/run_<RUN_TAG>/` 写本次 run，base 层维护
`latest` symlink；`HF_HOME` 和 v2 runtime teacher cache 固定在 base 层，避免每个 run
重复物化或重复缓存。每个训练 run 目录追加 `log.txt` 保存本次终端 stdout/stderr。

## 8. 评估

`eval_sft_v1.py` 输出四个核心指标：

| 指标 | 计算 | 目标 |
|---|---|---|
| `keep_accuracy` | keep 样本 STATUS == GT | 越高越好 |
| `advance_accuracy` | advance 样本 STATUS == GT | 越高越好 |
| `early_advance_rate` | keep 样本 STATUS == next(GT) | 越低越好 |
| `anchor12_sanity` | 典型早推进 fail case 是否回到 initial | 必须通过 |

Qwen3-VL 上 PEFT wrapper generation 可能错位，所以 eval/probe 默认 `merge_and_unload`。
v2 的 `max_gen_tokens` 默认 256，避免只生成 ANALYSIS 就被截断。
eval 终端输出追加到 `<save-root>/eval/log.txt`，probe 终端输出追加到
`<save-root>/eval_cases/log.txt`。

## 9. 文件清单

| 文件 | 用途 |
|---|---|
| `SFT_PLAN.md` | 本设计文档 |
| `SFT_RUN.md` | v1/v2 合并运行手册 |
| `build_sft_dataset_v1.py` | v1/v2 数据生成 |
| `build_sft_dataset_v2_teacher.py` | v2 teacher ANALYSIS 物化 |
| `sft_v1_train.sh` / `sft_v2_train.sh` | v1/v2 训练入口 |
| `sft_v1_loss_scale_plugin.py` / `sft_v2_loss_scale_plugin.py` | ms-swift loss 策略 |
| `check_loss_mask.py` / `check_loss_mask_v2.py` | token 级 loss sanity |
| `eval_sft_v1.py` / `probe_sft_v1.py` | 共享评估与 case dump |
| `inspect_teacher_outputs.py` | v2 teacher 预览 |
| `../tb_serve.sh` | 通用 TensorBoard launcher |

## 10. 风险

| 风险 | 处理 |
|---|---|
| ms-swift loss_scale 插件未注册 | 先跑对应 `check_loss_mask*.py`，确认插件路径从 `AutoMoT/` cwd 可访问 |
| swift chat template 与 runner structured image 不一致 | 对同一 val sample 比较训练 collator input_ids 和 runner prefill input_ids |
| v1 输出循环复读 `STATUS:` | 降 lr / 选更早 checkpoint / 看 early_advance 与 advance 曲线 |
| v2 teacher 套话或太短 | 用 `inspect_teacher_outputs.py --live` 看样本，改 prompt 后刷新 cache |
| runtime teacher cache 旧配置残留 | `RUNTIME_TEACHER_REFRESH=1` 或删除 `runtime_teacher_data/` |
