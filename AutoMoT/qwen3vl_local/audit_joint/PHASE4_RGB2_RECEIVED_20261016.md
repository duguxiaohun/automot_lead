# Phase4 RGB2 首轮交回审计（2026-10-16）

结论：RGB2 首轮训练消费、固定验证及保存重载通过；可以继续同一运行合同的 RGB4 首轮。RGB2 固定 test 尚缺，RGB4 尚无结果，不能宣布两模式全流程验收。无需重新建库、老师回放、RGB2 训练或重载。本轮不改标定阈值、提示词和训练实现，不 push。

## 收件与复核范围

收到 Downloads/phase4_epoch_report.zip（3,740,004 字节）及 phase4_epoch_package.json。ZIP SHA256：`35ad45a1be161771980460b98416a96a4ffe8ffbb7892fe284dd5dfab49a84f8`。27 成员 SHA、CRC、路径及大小检查通过。解包与可重跑审计脚本在 `checkpoints/phase4_rgb2_received_20261016/`，详细计算结果为 `audit_results.json`。

重新核对四 rank 消费顺序/ID/数量与哈希计划，重解析全部 raw 答案，重算逐事件及逐边指标；370 条原验证与重载案例完整相同。权重未随包交回，服务器重载回执通过不等于本机重新加载过权重。本轮没有新增 RGB 目视，也没有重新认证全源标签或跨任务 G0。

## 实际消费与结果

| 项目 | 结果 |
|---|---|
| 训练 | 10,240 次呈现，十事件各 1,024；四 rank 各 2,560 条及 320 次更新记录 |
| 独立训练题 | 10,036 / 298,787（3.36%），最多重复 2 次 |
| 验证 | 370 道，十事件各 37，无重复 ID，固定计划顺序一致 |
| test | 计划十事件各 18；实际结果缺失，不能称实际等量通过 |
| 规则一致率 | 347/370 = 93.78% |
| YES / NO 召回 | 31/50 = 62%；316/320 = 98.75% |
| 全猜 NO 基线 | 86.49%；YES/NO 平均召回为 80.375% |
| 重载 | 370 道预测及 raw 全部一致，无解析失败 |

事件归属继承已绑定采样计划，实际 step 记录的 ID/顺序独立核验；没有在本机重读服务器全题库来重新证明归属。一轮是固定 10,240 次预算，并非遍历全部 298,787 道题。无需强制 YES:NO 等额。

## 不能被总分遮住的问题

1. **R-E5 STOP 放行 YES 为 0/3**。该事件整体 YES 为 6/10，但正确项主要在 hold/restrict；训练实际计划中有 158 次 proceed/YES，不能解释为完全没有正例暴露。仅凭三个失败不能认定停车标签错，应与同合同 RGB4 比较后再定位。
2. **R-E2 YES 为 0/2**；R-E3 YES 为 2/8、U-E6 为 1/3、U-E7 为 1/4。整体仍偏 NO，23 个错误中 19 个为 YES→NO、4 个为 NO→YES。小分母需保留，不能当精确总体能力估计。
3. **事件等量不等于关键动作覆盖齐全。** 验证没有 U-E2 return、U-E7 complete；U-E1/U-E3/U-E5 的 proceed 没有 YES。U-E5 的 37/37 来自已覆盖题型，不能证明放行能力。正式池 U-E5 proceed/YES 仍为 0，开发模式限制不变。
4. 本轮 U-E2 return/YES 实际采样为 0，U-E7 complete/YES 为 4；少见支持未必在第一轮出现。保留跨轮曝光记录，不通过复制题目或把 UNKNOWN 改 YES 来补数。

自动规则可继续用于限定范围开发训练，不要求全量人工标注；规则一致率不是独立真实准确率。此次评测集与历史短跑不同，不将高分解释为严格对照下的性能提升。不依据已看过的错误题重选固定 val/test。

## 版本来源与耗时

启动 source_check 的 523 源文件匹配 v60 清单；run/adapter/reload 中 15 个学生源码身份全部匹配 v61。可见差异是 sequence_batch.py：启动记录 `9301496c…`，训练记录 `f19b027e…`，使用采样 v2，**不是当前本机 v62/v3**。训练准入计划、实际消费、重载内部一致。仅凭这些记录不能确定改动发生时间，也不能把启动清单当作训练时全源码快照。

完成此次对照应保持原运行合同。不要先同步 v62，再修改旧 manifest/SHA 让它通过；如果服务器已更新，下面的合同检查会拒绝，此时应先恢复独立的原版本运行环境，再继续，不能覆盖活动源码或重训已完成 RGB2。

| 已记录阶段 | 用时 |
|---|---|
| 视图构建及核验 | 2 小时 58 分 26 秒 |
| RGB2 训练（最慢 rank） | 1 小时 30 分 55 秒 |
| 训练内验证 370 题 | 2 分 59 秒 |
| 独立重载整段 | 2 小时 49 分 1 秒 |

这些不是重复 20 小时老师回放。独立重载包括数据/来源核验、加载模型和推理，缺少阶段细分，不能将 2 小时 49 分全部归因于 370 题 GPU 生成，更不能据此精确估计 RGB4。未来可补阶段计时，但无需为测耗时重复本次任务。

## 服务器接续（AutoMoT 根目录，保留原运行源码）

先确认没有另一个会话仍在跑本目录的 test/RGB4。以下命令不重跑首轮总编排、不新建 view、不回放老师。每一段成功后再执行下一段；合同检查失败则停止并交回报错。

### 1. 固定路径与原合同检查

```bash
export P4_RUN=checkpoints/phase4_epoch1_20261015_214227_800375064
export P4_NATIVE=checkpoints/phase4_v57_20261014_232157_762998141
python - <<'PY'
import json, os
from pathlib import Path
from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import check_contract
root = Path(os.environ['P4_RUN'])
record = json.loads((root/'run_first_epoch_rgb2/epoch_000/phase4_contract.json').read_text())
check_contract(record['source'])
assert (root/'student/view').is_dir()
print('原训练合同匹配，可继续同版本实验')
PY
```

### 2. 只补 RGB2 固定 test

如果 test_fixed 已存在，先保留并交回，勿删除后重跑。确认不存在、没有同目录活动任务再运行：

```bash
(
set -euo pipefail
test ! -e "$P4_RUN/run_first_epoch_rgb2/test_fixed"
bash qwen3vl_local/audit_joint/run_guarded.sh --space-path "$P4_RUN" --min-free-gib 20 -- \
  python -m qwen3vl_local.sft_new_loop_phase4.rgb_short.evaluate \
  --dataset "$P4_RUN/student/view" --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B \
  --adapter "$P4_RUN/run_first_epoch_rgb2/epoch_000" --rgb-mode 2 \
  --split test --output-dir "$P4_RUN/run_first_epoch_rgb2/test_fixed"
)
```

### 3. RGB4 首轮、重载、固定 test

复用相同 view、原始参数（B1、累积8、有效更新32），不把尚未实测的 v62 参数混入本次比较。默认选卡方式以服务器原版本为准，需保证四张卡空闲；已有其它 GPU 任务时不要争用。

```bash
(
set -euo pipefail
test ! -e "$P4_RUN/run_first_epoch_rgb4"
test ! -e "$P4_RUN/rgb4.log"
DATASET="$P4_RUN/student/view" \
TEACHER_REFERENCES="$P4_NATIVE/references4" \
FIRST_EPOCH_BUDGET="$P4_NATIVE/first_epoch_budget.json" \
OUTPUT_DIR="$P4_RUN" RUN_TAG=first_epoch RGB_MODE=4 \
BATCH_SIZE=1 UPDATE_SAMPLES=32 ACCUMULATION=8 \
AUTOMATIC_COVERAGE_POLICY=existing_data_development \
bash qwen3vl_local/sft_new_loop_phase4/rgb_short/run_automatic.sh \
  > "$P4_RUN/rgb4.log" 2>&1
)
```

在另一终端 `tail -f "$P4_RUN/rgb4.log"` 查看（需同样设置 P4_RUN）；不要把日志暂时不增长认定任务已死。若失败保留目录和日志，不重复覆盖启动。

### 4. 成功或失败都重新收集

```bash
python -m qwen3vl_local.audit_joint.phase4_epoch_handoff \
  --root "$P4_RUN" \
  --output "$P4_RUN/handoff_followup_$(date +%Y%m%d_%H%M%S_%N)"
```

交回新 handoff 目录的 `phase4_epoch_report.zip` 和 `phase4_epoch_package.json`。原 workflow 的 SIGINT/130 留痕可能仍使整体摘要为 interrupted/incomplete，不能手改回执；后续根据补齐的两模式训练、重载、固定 test 证据判定。若合同检查失败，先交回该报错及当前版本信息，不运行依赖步骤。

下一步只需补齐以上执行证据；本轮未发现要求重建正式库或改提示词才能继续 RGB4 的新确定缺陷。Phase3 不重做，Phase4 完整能力和联合 G0 仍不在本轮已通过范围内。
