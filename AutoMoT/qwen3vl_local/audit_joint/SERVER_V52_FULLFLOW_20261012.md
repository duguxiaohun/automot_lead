# v52正式规则池 → 自动首轮训练

> **v53修复后的执行边界：** U-E5离场现已统一使用固定实例走廊，老师升为v18、生产合同v53。下文是v52历史操作，**不要用新版源码直接执行这里的旧pilot、旧回放或旧view命令**；它们将被合同检查拒绝。原产物保留，不修改SHA或覆盖。当前不重复服务器建库/readiness/旧fit；新版定向重放与下一步见[U-E5走廊修复报告](UE5_CORRIDOR_V53_20261012.md)。此前第0步诊断也要求原v52源码，当前不再要求服务器执行。

**最新：8614路线正式建库已完成，保留现有产物，不重复第1步。** 两模式训练1024/事件、val37/事件、test18/事件及首轮预算均通过；唯一首轮门报错是U-E5/proceed/readiness/YES缺失。详见[FULLPOOL_RECEIVED](FULLPOOL_RECEIVED_20261012.md)。当前尚未启动GPU。

新诊断代码发布后继续沿已有`sync_automot.sh`从`duguxiaohun/automot_lead`覆盖同步，新清单462文件（fullflow3）；不使用服务器遗留`.git`判断版本。本轮生产和学生合同均不变，不需重建现有view。此次本地89条训练路线扩展与RGB复核见[U-E5离场诊断](UE5_DEPARTURE_AUDIT_20261012.md)，正例缺口仍未解决。

## 0. 当前只补U-E5的既有回放证据

新工具发布同步后，在AutoMoT根目录、automot环境执行：

```bash
bash qwen3vl_local/audit_joint/run_guarded.sh \
  --space-path checkpoints/v52_auto_baseline_20261012 --min-free-gib 20 -- \
  nice -n 10 python -m qwen3vl_local.audit_joint.event_gap \
  --candidate-pool checkpoints/v52_pilot_20261012_133645_029509259/candidates.json \
  --production-index checkpoints/v52_pilot_20261012_133645_029509259/production/index.json \
  --expected-index-sha256 fedadd305e4f9424f338b6e65b232a90615a510fcfd779664bdbe3167ac8d284 \
  --event U-E5 --data-root lead_data \
  --output checkpoints/v52_ue5_gap_20261012.json
```

读取声明含U-E5的现有回放、核验原生收据并统计；`--data-root`额外读取有限训练样例的七帧RGB/meta/bbox，核对源SHA及几何，不执行整路线重放、编译或训练，不修改旧目录。val/test仅汇总，不抽题目视。交回一个 `checkpoints/v52_ue5_gap_20261012.json`，工具限制30MB以内。退出0仅表示诊断核验通过，不是开训准入通过；原始样例缺失/漂移时保留已核验回放统计，报告`incomplete_raw_diagnostic`、退出2，也交回报告。若输出已存在，换新文件名保留旧报告；不要取消index SHA检查。该报告用于绑定服务器正式证据，本地规则研究不必等待它。

## 1. 已完成的CPU正式池步骤（保留记录，当前不要重跑）

以下pilot目录来自本次回执。输出目录必须尚不存在：

```bash
PILOT_ROOT=checkpoints/v52_pilot_20261012_133645_029509259 \
OUTPUT_DIR=checkpoints/v52_auto_baseline_20261012 \
DATA_ROOT=lead_data MODEL_DIR=checkpoints/Qwen3.5-4B REPLAY_WORKERS=4 \
  bash qwen3vl_local/audit_joint/server_finish.sh
```

自动复用原候选和30路线，完成其余回放；单份gzip回放共享给RGB4库，不再复制两份production。随后导出自动规则val/test、生成RGB2 `[-2,0]` 与RGB4 `[-6,-4,-2,0]`视图，核验正式池/均衡/关键条件支持，计算当前磁盘上的两模式首轮预算。原始库val/test为空不妨碍单独导出自动参考。

本次已测得：增量回放7980.4秒，整个CPU命令18508.81秒（约5小时8分），后续编译及准入占其余时间。

新版入口在完整回放核验后先检查五项关键readiness正反例。原始训练支持为0时写`raw_training_support.json`并提前结束，`compilation_skipped=true`，不再编译注定无法准入的库；支持非0仍须通过后续实际过滤、隔离、等量和预算检查。此门只影响未来新构建，本次已有正式库不要重跑。

运行设置20GiB最低余量和约12GiB本次盘净新增轮询限制；同时监控回放和输出所在的同一文件系统。若空间或流程失败，会停止并保留产物；不要删除旧实验，也不要跳过门继续训练。局部失败后的data4/references/view不能直接当成完整库复用；保留报告后决定恢复方式。

## 2. 看结果并交回两个文件

查看`checkpoints/v52_auto_baseline_20261012/finish_receipt.json`及`automatic_admission.json`（若已生成）。

- `admitted_rule_baseline`且退出0：正式规则数据与首轮门通过，可执行下一步。
- 新打包器的 `admission_blocked`：CPU已完成但准入不合格，不是中断；本次旧包虽写 `blocked_or_interrupted`，内部阶段已确认 `finished_cpu`。
- 退出2/75或其他失败：先交包，不启动GPU。缺事件、关键正反例或容量不足会列出，不能把UNKNOWN补成YES/NO。

交回终端打印的：

```text
checkpoints/v52_auto_baseline_20261012/handoff/fullflow_report.zip
checkpoints/v52_auto_baseline_20261012/handoff/fullflow_package.json
```

ZIP≤30,000,000字节，包含源码核对、阶段结果、容量依据、准入报告和日志尾部；权重/RGB/完整回放不随包。源码同步失败只交`source_check.json`；主机断电等无法完成打包时，保留目录和日志反馈即可。

## 3. 仅在准入通过后，两模式顺序各训练一轮

```bash
export DATASET=checkpoints/v52_auto_baseline_20261012/view
export TEACHER_REFERENCES=checkpoints/v52_auto_baseline_20261012/references4
export FIRST_EPOCH_BUDGET=checkpoints/v52_auto_baseline_20261012/first_epoch_budget.json
export DATA_ROOT=lead_data
export MODEL_DIR=checkpoints/Qwen3.5-4B

RGB_MODE=2 OUTPUT_DIR=checkpoints/v52_rgb2_epoch1 \
  bash qwen3vl_local/sft_new_loop_phase4/rgb_short/run_automatic.sh
RGB_MODE=4 OUTPUT_DIR=checkpoints/v52_rgb4_epoch1 \
  bash qwen3vl_local/sft_new_loop_phase4/rgb_short/run_automatic.sh
```

默认自动寻找4张空闲GPU。显式选卡时，在同一条命令前加`GPU_IDS=0,1,2,3`，例如：

```bash
GPU_IDS=0,1,2,3 RGB_MODE=2 OUTPUT_DIR=checkpoints/v52_rgb2_epoch1 \
  bash qwen3vl_local/sft_new_loop_phase4/rgb_short/run_automatic.sh
GPU_IDS=0,1,2,3 RGB_MODE=4 OUTPUT_DIR=checkpoints/v52_rgb4_epoch1 \
  bash qwen3vl_local/sft_new_loop_phase4/rgb_short/run_automatic.sh
```

上面两组是自动选卡与显式选卡的替代写法，选择一组执行，避免重复训练。入口固定rank16/vision-off、每事件1024次、总10240次、一轮，自动保存、独立进程重载验证及本次epoch_000的固定十事件等量test。每次启动会重查当前余量与数据身份；不续旧adapter，不直接改成7轮。预算为顺序运行两模式、各保留一个checkpoint，若修改训练规模/模型结构必须重新预算。

训练完成后按已有白名单交接流程交两模式的run、sampling/samples、validation/cases、checkpoint回执、reload_check及test_fixed指标/逐题结果；不传权重或optimizer。CPU完成包与训练结果包分开，避免把数据就绪当作GPU训练通过。
