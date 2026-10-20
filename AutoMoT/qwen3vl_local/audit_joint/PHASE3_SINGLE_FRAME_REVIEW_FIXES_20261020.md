# 单帧 Phase3：保存合同、完整轮次接续与配对评测

当前历史仍为默认 `nearest_short`。本轮不改原生标定、历史填补、未来缺失mask、独立event/action配额、三轮7680更新或每锚点有效窗口loss取平均。

## 保存合同

adapter配置和训练manifest由 `protocol.py` 写入同一合同：`history_rgb_mode=1rgb_current`、`history_rgb_count=actual_input_rgb_frames=1`、四查询、A–F现有词表映射，以及新提示词名称/variant/模板配方SHA。保留的 `[3]` 索引表示从原四路径题库取最后一图，`index_rgb_count=4` 明确只描述题库格式。输入说明不再写RGB历史。

提示词指纹绑定system文字、消息构建函数、前提构建函数及源码、实际history语义文本和填补配置。`protocol.validate(config, artifact)` 核验保存与重载合同，拒绝旧四图字段混入。原生trainer内部参数仍兼容原始题库检查器；推理合同以专用保存配置为准。此adapter须使用 `phase3_single_frame.runtime.predict` 的四查询协议，不能套旧四图单动作生成器。

## 接续

`worker prepare` 新增 `--resume /明确的/旧输出/train/training_states/epoch_0002`（也支持epoch_0001）。输出必须是旧实验之外的新目录；仍拒绝已有输出、不猜最新断点、不覆盖失败现场。完整CPU核验包括题库、数据、源码、模型、world-size、三轮LR计划、轮次/步数、实际采样状态、optimizer/scheduler/RNG及checkpoint文件SHA；开训前和原生恢复时再次检查。新输出需要通过匹配的四卡probe。

接续粒度是完整epoch边界。第三轮中途失败时，从epoch_0002恢复并重做第三轮，不宣称恢复到中途某一步。只支持同合同断点；旧源码/旧数据绑定的checkpoint不会通过改SHA强行迁移。本轮尚无实际H20接续结果。

以下变量由运行机器实际资产确定，不假定服务器或绝对路径：

```bash
python -m qwen3vl_local.audit_joint.phase3_single_frame.worker prepare \
  --index "$P3_INDEX" --data-root "$P3_DATA_ROOT" --model-dir "$P3_MODEL_DIR" \
  --dataset "$P3_DATASET" --output "$P3_RESUME_OUT" \
  --resume "$P3_OLD_OUT/train/training_states/epoch_0002"

torchrun --nproc_per_node=4 --module qwen3vl_local.audit_joint.phase3_single_frame.worker \
  probe --job "$P3_RESUME_OUT/single_frame_job.json"
torchrun --nproc_per_node=4 --module qwen3vl_local.audit_joint.phase3_single_frame.worker \
  train --job "$P3_RESUME_OUT/single_frame_job.json"
python -m qwen3vl_local.audit_joint.phase3_single_frame.worker evaluate \
  --job "$P3_RESUME_OUT/single_frame_job.json" --split val
python -m qwen3vl_local.audit_joint.phase3_single_frame.worker evaluate \
  --job "$P3_RESUME_OUT/single_frame_job.json" --split test
```

## 评测与实际监督量

`paired_repeat_last` 同时统计模型与重复最后已知历史动作在**同一批有效未来窗口**上的命中率、raw质量分、差值及各自独占命中数；无可用历史的锚点单列，不混入这项配对分母。原全体模型指标保留。

正式evaluate重载同一完整checkpoint，固定val/test成员，执行四遍前向：

- full：当前图像和填补后历史。
- no_history：历史四格替换为UNKNOWN，图像与其余文字保留。
- native_history：使用保存的未填补历史。
- blank_image：同尺寸全黑RGB，保留全部历史和文字。

每种保存逐题预测；主metrics里的 `paired_input_ablations` 给出全有效窗口及历史被填补锚点子集的配对结果。四组输入均纳入CPU token/视觉形状扫描，最重输入纳入GPU probe。blank_image是输入扰动；native_history是同权重的输入对照，都不能替代分别训练模型的消融实验。

最多四个loss，不保证每条四个。训练账本每个rank/step写 `supervised_windows` 和 `query_weights`；完成轮次报告实际窗口总量、每锚点1–4窗口直方图、每次optimizer更新平均监督窗口数。固定val的零监督锚点仍记录skipped，不计作正确。修复了原生周期验证从独立选题路径进入时未登记seed映射的问题；该路径也使用四查询loss。

## 不重新标定的数据复用

新增显式 `refresh_runtime`，只允许runtime/objective/worker及新增合同工具改变；原生261项发布依赖、data/history/targets/planning/CLI等绑定必须保持。校验原artifact/回执SHA及所有原始数据后，在新目录保存相同records和父产物SHA，只更新明确的运行时绑定，记录 `labels_recomputed=false`。标签源码变更直接拒绝。

```bash
python -m qwen3vl_local.audit_joint.phase3_single_frame.refresh_runtime \
  --dataset "$P3_OLD_DATASET" --output "$P3_REFRESHED_DATA_DIR"
```

本地当前资产：`checkpoints/phase3_single_frame_v1/review_runtime_train/dataset.json` 和 `review_runtime_val/dataset.json`；它们保留原302/320开发锚点、88/116格历史UNKNOWN，以及946/964个有效未来窗口，不作为服务器全池训练替代品。

最终测试、真实开发数据逐字段复核及源码SHA见 `phase3_single_frame_review_fixes_verification_20261020.json`。CPU真实原生训练循环包含周期val、最终保存、epoch1与epoch2接续，并比较连续/接续的参数、optimizer、scheduler、RNG和采样状态；小Qwen运行四组评测，核对参数及梯度未变。尚未GPU训练或证明性能改善。本轮未push。
