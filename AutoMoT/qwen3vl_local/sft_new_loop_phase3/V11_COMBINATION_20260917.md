# Phase3 v11：v7 紧凑提示词 + v10 标定合同

## 目标

v11 只吸收历史版本已观察到的提示词长处，不根据新的逐帧 RGB 检查、模型错例或测试集结果改标签。
它恢复 09/11 `v7_compact_observed_forecast` 的短 system/user 表达和事件域单选措辞，同时保留后续已经独立验证的 v10 数据与动作合同。

## 固定组合

| 层 | v11 采用 | 不采用 |
| --- | --- | --- |
| Prompt | v7 紧凑措辞、choice 单动作词组 | v8/v9 追加的冗长边界说明 |
| 动作标定 | `current_wait_first_crossing_v8_bounded_window` | v7 动作规则回退 |
| 标签/映射 | 09/14、09/16 的精确 RGB 修复、排除和同 RS 负例 | 撤销已有修复或扩大排除范围 |
| 划分 | v10 的开发路线隔离、split seed `20260916` | 09/11 的旧 holdout |
| 数据与权重 | `sft_new_loop_phase3_data_v11` + 新 LoRA | v10 索引或任意历史 adapter 静默复用 |

`build_dataset.py` 将 v11 的 name 与四种 production prompt SHA 写入 `manifest.json`；preflight 会比对它们。因此动作标签即使在标准九采样窗口与 v10 一致，也必须通过 v11 重建流程取得可训练索引。

## 为什么这样组合

历史结果不能做跨 holdout 的因果归因，但 v7 是“精简且已有 choice”的首个完整版本，在精简版 binary 中总体 exact 和左右变道 F1 最强；v8 的 choice overall 略高但完整 binary 与 NONE 明显下降；v9 的 choice 及减速/横向 F1 继续下降。完整数值见 [VERSION_RESULT_MAP_20260917.md](VERSION_RESULT_MAP_20260917.md) 与 [AUDIT_COMPARISON_20260917.md](AUDIT_COMPARISON_20260917.md)。

反过来，v10 标定并非用于追逐模型分数：它限制长速度序列的越窗影响，并记录可追查的纵向判定轨迹；标准九采样及既有真实 meta 配对没有大规模标签变化。保留它避免把可靠的离线语义合同与提示词长度实验混为一谈，详见 [TEMPORAL_REFINEMENT_20260916.md](TEMPORAL_REFINEMENT_20260916.md)。

## 运行与评测

从 `AutoMoT/` 目录执行：

```bash
python qwen3vl_local/sft_new_loop_phase3/build_dataset.py
python qwen3vl_local/sft_new_loop_phase3/preflight.py \
  --index checkpoints/sft_new_loop_phase3_data_v11/frame_index.jsonl
python qwen3vl_local/sft_new_loop_phase3/train.py --sampling-only \
  --index checkpoints/sft_new_loop_phase3_data_v11/frame_index.jsonl \
  --focus-balance-count 1024 --eval-balance-count 16 --generation-eval-balance-count 32
```

之后 binary 与 choice 分开新训；独立 eval 使用 `CASES_PER_BIN=0`。v11 的比较应在这次重建后冻结的新 holdout 上进行，完整 binary 与单动作 choice 分开报告。v11 尚无训练或闭环成绩，历史分数不能移植为 v11 结论。
