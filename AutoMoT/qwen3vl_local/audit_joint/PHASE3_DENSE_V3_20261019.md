# Phase3 dense v3：第二轮审查修复

> 当前提示词与探针合同已更新为 [v4](PHASE3_DENSE_V4_20261019.md)。以下为 v3 历史说明，启动新实验请使用 v4 新输出目录。

本页取代 v2 密集实验启动和验收说明。新合同为 `phase3_dense_verified_targets_v3`。独立 base 的事件/动作采样、原提示词、原规则与三轮 7680 更新不变；v2 checkpoint 必须用 v2 原版完成恢复，不能改 SHA 迁移。已完成的独立 base final 仍可显式作为新密集实验初始化。

## 这次改变了什么

* 旧公开入口转到新入口时，`SKIP_TRAIN=1`、`PIPELINE_CHECK_ONLY=1` 任一成立都会传递 `--prepare-only`，包括直接传新 CLI 参数的情况。旧 `check` 模式也只做 CPU 准入。未知布尔值拒绝，不默认为训练。
* single/all **共用原始锚点计划的逐事件、逐动作 loss 总权重**，不再分别按启用 query 重算动作配额。每个动作的单 query 权重仍因启用次数不同而不同，但整轮总量相等；`planned_query_loss_mass` 和 `anchor_action_quotas` 必须相同。额外位置出现锚点配额不支持的动作时不重新分配权重；记录零权重项，并按每事件/每位置检查有效、正权重监督覆盖。默认至少 80%，single 也执行同一潜在覆盖门禁，避免对照池不同或“有标签却没有额外 loss”。
* 主 `dense_val/test_summary.json` 改为**训练匹配读出**：future 取完整 forecast 输入中的 query 0；stream 取八图后的 query 4；windows 取最后四帧窗口的读出。`dense_*_transfer_summary.json` 单独报告四图 query 0 的迁移表现。native 160 题验证 loss 仍是四图迁移诊断，不是主对照指标。主摘要记录覆盖、分母和匹配成员 SHA，不能把两种分数混比。
* `--aux-index` 现在可补 train/val/test 的目标，但只能属于 base 中**同路线、同物理组、同 split** 的资产。辅助行不加入锚点池；评测锚点的身份、顺序和数量保持冻结。训练选样只读 train 行，val/test 不进入 loss。
* 补充标签的新行必须从本机原始 `metas/*.pkl` 重新运行当前 `RouteTrajectory.signals(frame_id)`、原生动作规则及动作投影，核对完整 action_evidence、answers、速度、导航目标、action_signature，以及确属该帧的四张 RGB 路径。沿用原生 source quarantine。只改 frame_id、保留旧证据/图片的行无法通过。此复算证明动作目标与原始轨迹帧一致，不等于视觉盲审、情境分类全源认证或外部可信签名。
* 固定 val/test 每事件、每位置的标签/完整窗口覆盖默认至少 80%，通过 `--min-eval-coverage` 在 `[0.5,1]` 设置。不足会在训练前阻塞；不替换缺失锚点、不插值、不拿训练路线补 holdout。允许部分缺失时，主指标明确报告已支持分母和固定总数，single/all 比较必须核对同一匹配成员 SHA。设为 1 可以要求全部固定锚点完整。
* 输入形状审计现在覆盖三轮剩余训练计划、固定 160 题 loss 验证、320/380 题四图迁移输入以及完整密集评测输入，包括 quick 尚不打分的 test。分别记录 `unique_training_anchors` 和 `unique_planned_inputs`。GPU 探针按训练/评测与输入模式分别挑最大成本形状：训练做 DDP/AdamW/backward，holdout 仅前向检查分配，不输出预测/指标、不做 CE/backward/optimizer。正式验证逐输入重核 token 长度。

`stream` 仍是共享增长前缀的 4/5/6/7/8 图读出；`windows` 仍是严格四图窗口、独立语言状态、共享冻结视觉计算。**严格滑窗同时共享语言状态仍未实现**，本轮没有把它改名成已完成。

## 原始来源复算与日志

CPU preflight 对辅助新行逐路线复算一次，写入本次输出下的 `dense_aux_provenance.json`，包含原始 meta/RGB 来源 SHA、文件身份/大小/mtime/ctime、路线 meta 成员及行复算摘要。后续 worker 验证该本地回执和所有来源文件身份/元数据、meta 成员是否变化，不重复执行原始轨迹复算；变化立即拒绝。此为本地不变性检查，不是对抗管理员伪造的签名认证。

完整来源清单留本机，避免把大清单或数据打进反馈包；反馈包只包含 `dense_aux_summary.json`（含完整清单 SHA）。不会复制 RGB/meta、重写旧索引或删除现有资产。没有原始 meta 或错误帧的辅助索引，不能通过补填一个 manifest/旧 SHA 绕过复算。

新报告还包括 `dense_evaluation_plan.json`、`dense_input_shapes.json`、两类评测摘要和原有实际消费记录。失败时 `dense_admission_failure.json` 与阶段日志保留原因。所有日志仍位于本次指定输出目录，不转到 `/tmp`。

## 执行顺序

先完成并验收独立 base，再用同一个 final adapter 启动 single/all；不要为此重跑旧最优模型。当前服务器全池覆盖及 H20 显存未在本地验收，因此先运行 CPU 准入。以下变量须指向当前机器明确存在的资产，不猜最新目录、不混用两台机器的数据：

```bash
set -o pipefail
P3_INDEX=checkpoints/phase3_four_data_20261017_223254_120427458/frame_index.jsonl
P3_BASE_FINAL=/本机已完成独立base的目录/train/final
P3_AUX_ARGS=()
# 有原生辅助索引时填写；应含保持原split的补充评测目标，并有本机原始meta。
# P3_AUX_ARGS=(--aux-index /本机辅助索引/frame_index.jsonl)

mkdir -p phase3_future_v3_prepare
bash qwen3vl_local/audit_joint/phase3_base_trial.sh \
  --experiment future --supervision all --prepare-only \
  --index "$P3_INDEX" "${P3_AUX_ARGS[@]}" --init-adapter "$P3_BASE_FINAL" \
  --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
  --forecast-span-seconds 6 --forecast-step-seconds auto \
  --min-future-coverage 0.8 --min-eval-coverage 0.8 \
  --output phase3_future_v3_prepare/result 2>&1 | tee phase3_future_v3_prepare/launch.log
```

`--aux-index` 可省略，但旧池必须本身满足训练和评测覆盖，否则准入失败。不要通过降低到零、修改规则 SHA 或复制标签让它通过。先反馈失败包定位原始支持缺口。

准入通过后，两组分别在新输出运行（同组参数必须一致，不能同时占用同一组卡）：

```bash
mkdir -p phase3_future_v3_single phase3_future_v3_all
GPU_IDS=0,1,2,3 bash qwen3vl_local/audit_joint/phase3_base_trial.sh \
  --experiment future --supervision single --stage quick \
  --index "$P3_INDEX" "${P3_AUX_ARGS[@]}" --init-adapter "$P3_BASE_FINAL" \
  --output phase3_future_v3_single/result 2>&1 | tee phase3_future_v3_single/launch.log

GPU_IDS=0,1,2,3 bash qwen3vl_local/audit_joint/phase3_base_trial.sh \
  --experiment future --supervision all --stage quick \
  --index "$P3_INDEX" "${P3_AUX_ARGS[@]}" --init-adapter "$P3_BASE_FINAL" \
  --output phase3_future_v3_all/result 2>&1 | tee phase3_future_v3_all/launch.log
```

正式命令同样会重新核验自己的来源与完整计划，并执行四卡探针；CPU prepare 不被当作跨输出目录的 GPU 准入。方案二可把同组两条命令的 experiment 同时改为 `stream` 或 `windows`，并换输出名。所有模式 quick 只跑首轮 2560 更新，保留三轮调度；随后用同版本 `--stage complete --resume <各自epoch_0001>`、相同合同参数、新输出续到各自额外 7680 更新。

比较顺序：输入计划 SHA → 逐事件/动作 loss mass → 实际消费 → 固定评测匹配成员 SHA → 主 matched 指标及逐事件/动作结果 → 次要 transfer 指标。不能仅凭某一组提升宣布 dense 收益或历史性能已恢复。

每次交 `result/handoff/phase3_base_report.zip` 与 `phase3_base_package.json`；前置未建输出时交当前实验目录的 `launch.log`。没有执行服务器 H20 训练、真实全池准入或新模型效果验证前，本页不宣称这些已通过。

## 本地验证

当前代码 144 项回归（108.23 秒）及随后补充的 1 项来源/打包边界测试（2.01 秒）通过；旧行为兼容 9 项（2.54 秒）通过。旧发布源码不可变性断言留在其原版 checkout 执行，未重写旧清单。新发布 244 项源码 SHA 全匹配。测试包括真实小 Qwen、四 CPU rank、原生 meta 复算合成路线和官方实现的处理器；这些不是服务器全池或 H20 验收。轻量回执见 `phase3_dense_v3_verification_20261019.json`。
