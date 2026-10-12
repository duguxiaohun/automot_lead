# v52正式规则池 → 自动首轮训练

当前服务器30路线试建已通过，不再重跑pilot/readiness/旧fit。**本轮新代码发布后**继续沿已有`sync_automot.sh`从`duguxiaohun/automot_lead`覆盖同步。入口核验458个发布文件；不使用服务器遗留`.git`判断版本。原生生产合同仍为v52，已完成回放可复用。

## 1. 在AutoMoT根目录、automot环境的tmux中完成CPU正式池

以下pilot目录来自本次回执。输出目录必须尚不存在：

```bash
PILOT_ROOT=checkpoints/v52_pilot_20261012_133645_029509259 \
OUTPUT_DIR=checkpoints/v52_auto_baseline_20261012 \
DATA_ROOT=lead_data MODEL_DIR=checkpoints/Qwen3.5-4B REPLAY_WORKERS=4 \
  bash qwen3vl_local/audit_joint/server_finish.sh
```

自动复用原候选和30路线，完成其余回放；单份gzip回放共享给RGB4库，不再复制两份production。随后导出自动规则val/test、生成RGB2 `[-2,0]` 与RGB4 `[-6,-4,-2,0]`视图，核验正式池/均衡/关键条件支持，计算当前磁盘上的两模式首轮预算。原始库val/test为空不妨碍单独导出自动参考。

预计需要数小时：试建回放的简单线性估算为单worker约8小时，4worker理想约2小时，未含I/O争用、较长路线及编译/准入耗时，不作完成时限保证。

运行设置20GiB最低余量和约12GiB本次盘净新增轮询限制；同时监控回放和输出所在的同一文件系统。若空间或流程失败，会停止并保留产物；不要删除旧实验，也不要跳过门继续训练。局部失败后的data4/references/view不能直接当成完整库复用；保留报告后决定恢复方式。

## 2. 看结果并交回两个文件

查看`checkpoints/v52_auto_baseline_20261012/finish_receipt.json`及`automatic_admission.json`（若已生成）。

- `admitted_rule_baseline`且退出0：正式规则数据与首轮门通过，可执行下一步。
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
