# Phase3：独立 base 与单图方案一

2026-10-21：已有方案一在第2560步周期验证失败时，按 [周期验证修复与原库复用](PHASE3_PERIODIC_VALIDATION_FIX_20261021.md) 更新运行绑定并使用新输出；不重复建库或 base。下方首次运行命令适用于尚未派生方案一数据的环境。

当前只运行 base 和新的单图方案一，不调用旧 base/A/B suite，不运行方案二，不重新评测旧模型。两组分别从同一个本机 Qwen3.5-4B 初始化，各三轮、四卡 B1/累积1、每事件每轮1024、7680次更新、warmup2000、lr=1e-5。不会在 base 权重上继续训练方案一。

没有增加训练预算。两组总共15360次更新；本机未测H20耗时，不承诺一天内完成。每阶段默认超时48小时、磁盘绝对保留20GiB，首错停止、不自动重试。GPU probe只做资源/执行检查，权重不带入训练，不证明模型精度恢复。

## 运行前与目录

当前目录为本机 `AutoMoT/`，使用本机 automot Python环境。可任选服务器；数据、模型、索引和旧断点必须属于当前机器。源码可先沿已有 `bash ./sync_automot.sh` 同步，不复制或清理数据/权重。

**现在不需要设置 `P3_INDEX`，也不需要填写旧时间戳路径。** 省略 `--index` 时，首次按原生 Phase3 规则从本机 `lead_data` 和 `keyframe_filter/collection_output/*_result.json` 建库，后续校验并复用固定目录 `checkpoints/phase3_common_v1/`。原生规则、阈值、事件/action采样预算不变。默认模型 `checkpoints/Qwen3.5-4B`；本机路径不同时用 `--data-root`、`--collection-dir`、`--model-dir` 指定。

固定通用库包含：

- `native/frame_index.jsonl` 和 `native/manifest.json`：base与方案一共用的原生索引。
- `reference.json`：固定loss160、val320、test380成员；按事件均衡、无放回、选题不看答案。
- `single_frame/dataset.json`：首次运行方案一时派生并缓存，默认nearest_short，未来标签不强填。
- `common_receipt.json`：源代码、索引、manifest、评测名单和本机数据根的绑定。

若新索引恰好与旧SHA相同，保留原评测名单；否则冻结新名单并明确不计算相对历史70%结果的准确率差值。不会篡改旧参考文件或给新库伪造旧SHA。旧固定库显式 `--index` 模式仍保留。

现有完整库只复用，不重新扫描collection；损坏、残缺或换原始数据根会提前拒绝并保留原目录，不自动覆盖。需要新实验数据版本时显式指定新的语义名称 `--common-data-dir checkpoints/phase3_common_v2`，base与方案一必须使用同一个目录。通用库不带构建时间戳，原始route ID中的时间信息保持原样以免破坏数据身份。

固定目录（存在即拒绝覆盖，不自动续训）：

- base：`phase3_base_v1/run_base_independent_v1/`
- 方案一：`phase3_single_frame_v1/run_single_frame_nearest_v1/`
- `experiment/`：真实训练、断点、评测；各阶段日志在run目录。
- 方案一派生数据默认在上述通用库内，独立于训练目录，重跑实验可以复用。
- `handoff/phase3_report.zip` 与 `phase3_package.json`：成功或失败的轻量报告，不含权重/派生数据/RGB。

## 两组依次运行

无需索引占位符。`GPU_IDS=0,1,2,3` 为显式选卡；删除它则自动选四张最空闲卡。两组使用相同卡时顺序运行，不同时启动。

```bash
export PYTHON="$(command -v python)"
export GPU_IDS=0,1,2,3

bash qwen3vl_local/audit_joint/phase3_launch.sh \
  --method base --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B --output phase3_base_v1 \
&& bash qwen3vl_local/audit_joint/phase3_launch.sh \
  --method single_frame --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B --output phase3_single_frame_v1
```

长任务后台运行可以把上面两条命令放进以下同一个进程。日志名已存在时先改为新的明确名称，避免覆盖旧记录：

```bash
(
  set -C
  nohup bash -c '
    set -euo pipefail
    bash qwen3vl_local/audit_joint/phase3_launch.sh --method base \
      --data-root lead_data --model-dir checkpoints/Qwen3.5-4B --output phase3_base_v1
    bash qwen3vl_local/audit_joint/phase3_launch.sh --method single_frame \
      --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
      --output phase3_single_frame_v1
  ' > phase3_base_a_v1.launch.log 2>&1 < /dev/null &
)
tail -f phase3_base_a_v1.launch.log
```

只需方案一时直接运行第二条，也会自动准备通用库和单图派生数据。若要复用其它明确且**当前发布合同匹配**的派生数据，可以使用 `--dataset /明确路径/dataset.json`。旧数据需先显式 `phase3_single_frame.refresh_runtime` 核验兼容，在新目录更新运行时绑定；不能以重建数据作为续训前置步骤。

## 开训前检查及已修问题

共同入口先检查源码、模型文件、创建或核验共用索引与评测名单、完整原生题库manifest，随后分别完成CPU全输入检查和四卡probe，再训练和评测。新方案一保持nearest_short、不人工增加UNKNOWN、不填未来标签。tokenizer会在完整提示词上检查六动作历史ID与输出候选一致。

base额外扫描三轮实际选题、固定loss/val/test的四图解码及token预算，验证最终test成员在开训前存在。资源probe覆盖训练反向/AdamW、loss验证、val生成和test生成的不同输入形状；保持原val生成64与test生成256 token上限。单图方案一沿原四查询probe，并实际执行完整固定loss名单的周期验证（单独统计无监督锚点）；prepare还前置原生题库/模型检查。训练后仍必须核对真实消费与7680步。

旧base test入口给adapter同时传`--history-rgb-mode`，会被原生评测器拒绝。新入口通过独立`phase3_launch_worker`移除该冲突参数，让评测器读取保存的四图合同；保持原生trainer、原生eval及冻结发布清单字节不变。不要把旧`phase3_base_trial.sh`的完整流程当作本次修复后的运行入口。

所有torchrun命令显式隔开应用参数；`SKIP_TRAIN=1`、`PIPELINE_CHECK_ONLY=1`和`--prepare-only`只做CPU准备，不选GPU或训练。计划目录加`_plan`，不更新`latest`。正常训练成功才更新`latest`；旧数据/失败目录不删除。

## 只准备与接续

只建通用数据、不加载模型或选GPU：

```bash
python -m qwen3vl_local.audit_joint.phase3_common_data --with-single-frame
```

首次全量CPU建库可能耗时，日志在 `checkpoints/phase3_common_v1/build.log`；派生日志为 `derive_single_frame.log`。这一步不复制RGB、不启动训练；原始数据不足时不以审计样本冒充完整库。

要进一步做训练前的CPU输入检查，在正常命令末尾加`--prepare-only`。方案一准备成功后，正式训练仍复用通用库：

```bash
bash qwen3vl_local/audit_joint/phase3_launch.sh \
  --method single_frame --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B \
  --output phase3_single_frame_training_v1
```

接续只支持完整epoch1/2，同源码/模型/数据合同；创建旧运行之外的新输出基目录。例：

```bash
bash qwen3vl_local/audit_joint/phase3_launch.sh \
  --method single_frame --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B \
  --dataset checkpoints/phase3_common_v1/single_frame/dataset.json \
  --resume phase3_single_frame_v1/run_single_frame_nearest_v1/experiment/train/training_states/epoch_0002 \
  --output phase3_single_frame_resume_v1
```

base接续同理使用`--method base --resume .../epoch_0002 --output phase3_base_resume_v1`，不传dataset。源码升级不能直接恢复旧合同断点，需要原发布版本。第三轮中途失败从epoch2重做第三轮，不声称逐步续训。

成功以`launch_workflow.json/status=completed_quality_pending`为准，表示执行完成、精度待审。交回两组handoff目录；失败也交，若入口尚未创建run则交总launch日志。不要因为打包或评测失败重训已经完成的模型。

本地CPU/小模型与解析器验证不等于正式4B/H20全流程验收，也不能保证未来资产变动或硬件故障不会报错。当前没有H20性能结论。

## 本机通用库实际生成与验证（2026-10-20）

42个场景、7241条collection路线通过本机路径检查。原生构建完成：train11100、val420、test768行（含INVALID）；实际方案一缓存10240个锚点、2940条路线，原规则与冻结261源SHA不变。base三轮均十事件各1024；方案一三轮均10240呈现、无配额错误，有效监督窗口分别33892/33804/33850。固定val320与test380各event/window的独立支持对数最低为8和15，覆盖门通过；未填未来UNKNOWN。

71项相关测试通过（24.98秒）。实际数据位于本机 `checkpoints/phase3_common_v1/`，不上传GitHub，不代表其它服务器已有这些数据。新索引SHA与旧参考不同，已单独冻结新名单，不宣称恢复了历史70%以上准确率。未运行正式4B/H20；只有CPU采样/覆盖准入，不是GPU性能验收。轻量回执：`phase3_common_data_verification_20261020.json`。
