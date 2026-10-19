# Phase3 dense v5：逐帧语言缓存与严格四帧融合实验

用户已明确允许新增“逐帧独立语言编码＋窗口融合”协议，作为独立实验，不假定等价于原四图模型。新增入口 `--experiment windows_bank`；独立 base、future、stream 和原 windows 的训练目标不因此改成新架构。新密集合同为 `phase3_dense_strict_frame_bank_v5`。

## 新协议如何工作

1. 图像之前的相同系统提示词只计算一次，得到可反向传播的纯文本 KV、卷积和 DeltaNet 状态。
2. 八张源图各自从这个文本状态开始，作为八个互不包含其他帧的语言分支并行编码。每帧使用固定512位置的RoPE槽位，以保持帧顺序；它不是512个新增输入token。
3. 五个读出分别取帧 `[0,1,2,3]`、`[1,2,3,4]`、…、`[4,5,6,7]` 的 **full-attention KV**，加纯文本前缀组成各自的融合缓存。窗口外的帧KV不进入该读出。
4. 查询的卷积和 DeltaNet 递归状态从**纯文本前缀**开始，既不拼接四份递归状态，也不继承之前窗口的状态。图像融合通过查询对选中四帧KV的注意力完成。
5. 五个查询在一个batch中并行，目标A–F只在外部计算loss。缓存的计算图保留，梯度通过选中的图像KV及公共前缀回传；每次forward重新创建，optimizer更新后不保留旧缓存。

因此，本协议实现的是**严格四帧可见性＋共享逐帧图像语言KV＋五个监督位置**。它不是裁剪原生连续八图KV，也不声称与原生四图全因果前向等价。原生四图中的跨图递归和跨图语言编码，在这个协议中改成逐帧独立编码及查询融合，这是明确的架构差异。

共享实现的等价性参照是“同一新协议中，每个窗口重新编码自己的四帧”，即20次帧语言编码；共享版只做8次。测试比较两者的logits、全部参数梯度和分布式LoRA参数更新。这个参照仅用于验证缓存实现，不是重新推理历史最优模型，也不是效果对照结论。

| 模式 | 图像语言编码方式 | 每个读出可见图像 | 语言状态共享 |
|---|---|---|---|
| stream | 连续增长前缀 | 4/5/6/7/8 | 包含图像的增长前缀 |
| windows | 五份独立原生四图上下文 | 严格4张 | 无；仅复用视觉编码 |
| windows_prefix | 五份原生四图后缀 | 严格4张 | 只共享图像前的纯文本状态 |
| windows_bank | 八份独立单帧语言编码＋五个融合查询 | 严格4张 | 共享被选帧的full-attention KV及纯文本状态 |

`windows_prefix` 保留为实现对照：确定性设置下与完整原生窗口logits/梯度等价，**不称为图像语言KV共享**。只有 `windows_bank` 引入新图像融合协议。

## 训练配置与对照边界

两个缓存模式显式使用 **LoRA dropout=0、关闭gradient checkpointing**。前者避免公共前缀共享随机mask造成等价性混杂，后者避免重算时修改同一可变混合缓存。训练时若检测到不兼容的随机dropout或checkpoint重放，会拒绝执行，不静默改行为。它们的single/all使用相同设置；独立base及原future/stream/windows仍保留原设置。

这些变化和新架构都可能影响效果。应在 `windows_bank single` 与 `windows_bank all` 内比较密集监督收益，不能把bank与旧windows的差异全部归因于多loss。仍使用同一已完成独立base作为初始化、相同额外7680更新预算、相同输入计划、逐事件/动作loss总权重、固定评测锚点。quick停在2560更新，complete用同版本接续剩余两轮。

主评测继续取与训练匹配的最后一个查询，四图原生读出单独作为迁移诊断。训练和评测辅助标签仍需同split原始meta复算；未来标签、当前动作或其他窗口的答案不会进入输入。

## 显存准入

本版正式探针取消v4强制二十次不去重视觉编码。普通窗口保留20个呈现位置，但探针按时间槽只编码8个源帧，并复用到相应位置；即使八张图像像素相同，也强制每个源帧编码一次。bank本来就只有8个帧语言输入，其探针同样覆盖8张视觉编码。另保留真实内容去重路径，用于检查其真实分配开销。

形状扫描包括query位置引起的logits行数、语言计算token、帧/查询注意力工作量、视觉patch与注意力上界。bank额外计算公共前缀、八帧缓存和五个融合缓存同时存活时的token槽位，并检查RoPE位置小于8192。`--max-batch-tokens` 对bank约束这个缓存槽位总量，而不只是五窗口文字长度。探针会核对CPU预计几何与真实前向回执，并拒绝缺失或非有限的训练梯度。

缓存模式关闭checkpoint后可能增加激活显存，减少语言编码次数不等于必然更省显存或更快。仍需实际四张H20的训练反向、Adam状态和评测前向探针通过；CPU测试不提供H20余量保证。日志中 `input_tokens` 对缓存模式记录实际执行的补齐语言token，逐步batch记录的 `dense_compute` 保留帧语言编码次数、视觉编码数、缓存槽位等回执。原始逻辑窗口长度仍在形状报告中。

## 服务器步骤

先验收独立base的同口径结果。若已经完成独立base，不为本实验重训，也不重跑历史最优模型。下面变量必须填当前机器真实资产；不指定必须使用哪台机器、不猜最新目录。

在当前服务器的AutoMoT根目录，同步本次白名单源码后设置：

```bash
P3_BANK_INDEX=/本机冻结历史题库/frame_index.jsonl
P3_BANK_BASE=/本机已完成独立base的目录/train/final
P3_BANK_DATA=/本机原始数据根目录
P3_BANK_MODEL=/本机/Qwen3.5-4B
P3_BANK_AUX=()
# 需要补充标签时，填写已按原生规则生成且保持原split的辅助索引。
# P3_BANK_AUX=(--aux-index /本机辅助索引/frame_index.jsonl)
P3_BANK_ARGS=(--experiment windows_bank --index "$P3_BANK_INDEX"
  --init-adapter "$P3_BANK_BASE" --data-root "$P3_BANK_DATA" --model-dir "$P3_BANK_MODEL"
  "${P3_BANK_AUX[@]}" --max-batch-tokens 32768
  --min-future-coverage 0.8 --min-eval-coverage 0.8)
```

先只运行CPU准入，结果目录必须是新的；日志位于当前实验目录：

```bash
(
  set -euo pipefail
  mkdir phase3_bank_v5_prepare
  bash qwen3vl_local/audit_joint/phase3_base_trial.sh "${P3_BANK_ARGS[@]}" \
    --supervision all --prepare-only --output phase3_bank_v5_prepare/result \
    2>&1 | tee phase3_bank_v5_prepare/launch.log
)
```

先反馈这个结果。覆盖不足或缺原始来源时不要用复制标签、改SHA或替换固定评测题绕过。准入通过且新base已验收后，再顺序执行两组quick（不同时占用同一组卡）：

```bash
(
  set -euo pipefail
  for P3_BANK_ARM in single all; do
    mkdir "phase3_bank_v5_${P3_BANK_ARM}"
    GPU_IDS=0,1,2,3 bash qwen3vl_local/audit_joint/phase3_base_trial.sh "${P3_BANK_ARGS[@]}" \
      --supervision "$P3_BANK_ARM" --stage quick \
      --output "phase3_bank_v5_${P3_BANK_ARM}/result" \
      2>&1 | tee "phase3_bank_v5_${P3_BANK_ARM}/launch.log"
  done
)
```

每组交 `result/handoff/phase3_base_report.zip` 和 `phase3_base_package.json`，前置失败交launch.log。完成quick后仍须用同版本、相同模式及参数，指定各自 `result/train/training_states/epoch_0001`，新输出目录接续 `--stage complete`。不能将v4断点或另一组的断点套入v5。

目前没有服务器新模型结果，不能宣布恢复历史70%以上动作准确率、bank提升了性能或通过闭环驾驶验证。

## 本地验证回执

最终相关回归 **171项通过，179.06秒**；发布清单 **250项源码SHA匹配**。包含真实小型Qwen3.5、官方processor夹具、四CPU rank及两次LoRA更新，不是H20或完整服务器题库验收。详见 [轻量验证回执](phase3_dense_v5_verification_20261019.json)。
