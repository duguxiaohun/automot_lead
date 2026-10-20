# Phase3：独立 base 与单图方案一

当前只运行 base 和新的单图方案一，不调用旧 base/A/B suite，不运行方案二，不重新评测旧模型。两组分别从同一个本机 Qwen3.5-4B 初始化，各三轮、四卡 B1/累积1、每事件每轮1024、7680次更新、warmup2000、lr=1e-5。不会在 base 权重上继续训练方案一。

没有增加训练预算。两组总共15360次更新；本机未测H20耗时，不承诺一天内完成。每阶段默认超时48小时、磁盘绝对保留20GiB，首错停止、不自动重试。GPU probe只做资源/执行检查，权重不带入训练，不证明模型精度恢复。

## 运行前与目录

当前目录为本机 `AutoMoT/`，使用本机 automot Python环境。可任选服务器；数据、模型、索引和旧断点必须属于当前机器。源码可先沿已有 `bash ./sync_automot.sh` 同步，不复制或清理数据/权重。

`P3_INDEX` 必须指向原四图题库的本机路径，旁边保留原manifest及候选题库合同。冻结SHA为 `925053e34c47f35dfe4987f14a1be3a8abdc206d75bea31964dc4ca9b8fcee99`；不猜最新目录、不改SHA绕过。默认数据 `lead_data`，模型 `checkpoints/Qwen3.5-4B`；若本机不同，显式改参数。

固定目录（存在即拒绝覆盖，不自动续训）：

- base：`phase3_base_v1/run_base_independent_v1/`
- 方案一：`phase3_single_frame_v1/run_single_frame_nearest_v1/`
- `experiment/`：真实训练、断点、评测；各阶段日志在run目录。
- 方案一派生数据：`data/dataset.json`，仍保留原Phase3规则，原索引/原始数据不改写。
- `handoff/phase3_report.zip` 与 `phase3_package.json`：成功或失败的轻量报告，不含权重/派生数据/RGB。

## 两组依次运行

先替换下面唯一的索引路径。`GPU_IDS=0,1,2,3` 为显式选卡；删除它则自动选四张最空闲卡。两组使用相同卡时顺序运行，不同时启动。

```bash
export P3_INDEX='/本机/原四图题库/frame_index.jsonl'
export PYTHON="$(command -v python)"
export GPU_IDS=0,1,2,3

bash qwen3vl_local/audit_joint/phase3_launch.sh \
  --method base --index "$P3_INDEX" --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B --output phase3_base_v1 \
&& bash qwen3vl_local/audit_joint/phase3_launch.sh \
  --method single_frame --index "$P3_INDEX" --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B --build-data --output phase3_single_frame_v1
```

长任务后台运行可以把上面两条命令放进以下同一个进程。日志名已存在时先改为新的明确名称，避免覆盖旧记录：

```bash
(
  set -C
  nohup bash -c '
    set -euo pipefail
    bash qwen3vl_local/audit_joint/phase3_launch.sh --method base \
      --index "$P3_INDEX" --data-root lead_data --model-dir checkpoints/Qwen3.5-4B --output phase3_base_v1
    bash qwen3vl_local/audit_joint/phase3_launch.sh --method single_frame \
      --index "$P3_INDEX" --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
      --build-data --output phase3_single_frame_v1
  ' > phase3_base_a_v1.launch.log 2>&1 < /dev/null &
)
tail -f phase3_base_a_v1.launch.log
```

只需方案一时直接运行第二条。若已有**当前发布合同匹配**的派生数据，用 `--dataset /明确路径/dataset.json` 替代 `--build-data`。旧数据需先显式 `phase3_single_frame.refresh_runtime` 核验兼容，在新目录更新运行时绑定；不能以重建数据作为续训前置步骤。

## 开训前检查及已修问题

共同入口先检查源码、冻结索引、完整原生题库manifest、模型文件，随后分别完成CPU全输入检查和四卡probe，再训练和评测。新方案一保持nearest_short、不人工增加UNKNOWN、不填未来标签。tokenizer会在完整提示词上检查六动作历史ID与输出候选一致。

base额外扫描三轮实际选题、固定loss/val/test的四图解码及token预算，验证最终test成员在开训前存在。资源probe覆盖训练反向/AdamW、loss验证、val生成和test生成的不同输入形状；保持原val生成64与test生成256 token上限。单图方案一沿原四查询probe，并在prepare前移原生题库/模型检查。训练后仍必须核对真实消费与7680步。

旧base test入口给adapter同时传`--history-rgb-mode`，会被原生评测器拒绝。新入口通过独立`phase3_launch_worker`移除该冲突参数，让评测器读取保存的四图合同；保持原生trainer、原生eval及冻结发布清单字节不变。不要把旧`phase3_base_trial.sh`的完整流程当作本次修复后的运行入口。

所有torchrun命令显式隔开应用参数；`SKIP_TRAIN=1`、`PIPELINE_CHECK_ONLY=1`和`--prepare-only`只做CPU准备，不选GPU或训练。计划目录加`_plan`，不更新`latest`。正常训练成功才更新`latest`；旧数据/失败目录不删除。

## 只准备与接续

在正常命令末尾加`--prepare-only`。方案一准备成功后，正式训练改用明确的计划产物：

```bash
bash qwen3vl_local/audit_joint/phase3_launch.sh \
  --method single_frame --index "$P3_INDEX" --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B \
  --dataset phase3_single_frame_v1/run_single_frame_nearest_v1_plan/data/dataset.json \
  --output phase3_single_frame_training_v1
```

接续只支持完整epoch1/2，同源码/模型/数据合同；创建旧运行之外的新输出基目录。例：

```bash
bash qwen3vl_local/audit_joint/phase3_launch.sh \
  --method single_frame --index "$P3_INDEX" --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B \
  --dataset phase3_single_frame_v1/run_single_frame_nearest_v1/data/dataset.json \
  --resume phase3_single_frame_v1/run_single_frame_nearest_v1/experiment/train/training_states/epoch_0002 \
  --output phase3_single_frame_resume_v1
```

base接续同理使用`--method base --resume .../epoch_0002 --output phase3_base_resume_v1`，不传dataset。源码升级不能直接恢复旧合同断点，需要原发布版本。第三轮中途失败从epoch2重做第三轮，不声称逐步续训。

成功以`launch_workflow.json/status=completed_quality_pending`为准，表示执行完成、精度待审。交回两组handoff目录；失败也交，若入口尚未创建run则交总launch日志。不要因为打包或评测失败重训已经完成的模型。

本地CPU/小模型与解析器验证不等于正式4B/H20全流程验收，也不能保证未来资产变动或硬件故障不会报错。当前没有H20性能结论。
