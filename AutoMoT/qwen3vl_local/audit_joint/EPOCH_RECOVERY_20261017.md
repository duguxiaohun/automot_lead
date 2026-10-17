# v73：Phase3 epoch 边界续训与 Phase4 多轮入口

本轮按用户确认只完成续训和多轮入口。每卡B2、两图[t-2,t]/步长2、四图[t-3,t-2,t-1,t]/步长1、事件等量、连续段/action软补偿均不变；不增加GPU benchmark，不修改评估生成或视觉编码路径。

## Phase3：完整 epoch 边界恢复

已选B2入口默认开启 `SAVE_TRAINING_STATE=1`。每个完整epoch验证消费后、梯度累积完成且梯度清空时，保存训练参数、AdamW、学习率scheduler、每rank Python/Torch随机数、实际曝光history以及checkpoint当时的best_val/best_generation/fallback_generation副本。先写临时目录，完整文件SHA回执落盘后原子发布，最后更新 `latest_training_state.json`。坏文件、未完成写入、源码/配置/索引/基座资产/卡数漂移拒绝恢复。

仅支持同一原定总轮数与学习率计划内恢复。例如最初NUM_EPOCHS=3，在第1轮后退出，恢复仍设NUM_EPOCHS=3；不能把已经按1轮学习率训完的旧adapter当成3轮计划的第1轮。恢复使用新输出目录，不覆盖旧run；周期性adapter和旧版仅有权重的checkpoint不是恢复状态。CUDA恢复已接入每rank RNG，但本轮只做CPU小模型验证，不宣称H20逐位确定性。

要求 `MAX_STEPS=0`，且每epoch微步数能被GRAD_ACCUM整除；已选四卡B2、每事件1024、累积2满足。中途被终止时，从最近一个完整epoch恢复；未保存epoch的部分需要重跑。第一轮尚未完成则没有恢复点。通用train.sh默认行为保留，可显式SAVE_TRAINING_STATE=1；sampling-only只报告新计划，不作为resume验收。

以下从AutoMoT目录、已激活训练环境执行。首次正常三轮：

```bash
export INDEX=checkpoints/phase3_prepare_20261012_231356_193369568/data/frame_index.jsonl
export DATA_ROOT=lead_data MODEL_DIR=checkpoints/Qwen3.5-4B GPU_IDS=0,1,2,3
export NUM_EPOCHS=3 HISTORY_RGB_MODE=2rgb_short
unset RESUME STOP_AFTER_EPOCH
export OUTPUT_DIR=checkpoints/p3_recoverable
export RUN_TAG="$(date +%Y%m%d_%H%M%S_%N)"
bash qwen3vl_local/audit_joint/train_phase3_batch.sh ddp
```

四图将HISTORY_RGB_MODE改为4rgb并使用新RUN_TAG。若明确希望第一轮后停下，可在首次启动前设置 `STOP_AFTER_EPOCH=1`，NUM_EPOCHS仍为3，后续恢复前unset。

恢复时把P3_PREVIOUS_RUN设为该模式真实旧run目录，沿用原训练的全部参数：

```bash
(
set -euo pipefail
: "${P3_PREVIOUS_RUN:?设置为旧run的实际目录}"
export P3_PREVIOUS_RUN
RESUME="$(python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['P3_PREVIOUS_RUN']) / 'latest_training_state.json'
print(json.loads(p.read_text())['checkpoint'])
PY
)"
export RESUME
unset STOP_AFTER_EPOCH
export NUM_EPOCHS=3
export OUTPUT_DIR=checkpoints/p3_recovered
export RUN_TAG="$(date +%Y%m%d_%H%M%S_%N)"
bash qwen3vl_local/audit_joint/train_phase3_batch.sh ddp
)
```

## Phase4：两模式多轮／分别恢复

现有 `train_phase4_batch.sh` 新增EPOCHS、RESUME_RGB2、RESUME_RGB4。EPOCHS为绝对目标轮数；例如epoch_000完成后EPOCHS=3，继续第2、3轮。未设置恢复路径的模式从第1轮开始，不猜测latest目录；两模式可以从不同的完整epoch恢复。不设置EPOCHS/恢复路径仍沿原首轮工作流。

多轮入口 `audit_joint.phase4_training` 复用原生训练、逐轮val、最终epoch重载核验和固定test。训练开始前验证两个恢复点的源码合同、模式、四卡B2/累积4配置、checkpoint回执/权重/optimizer/history和历史最佳checkpoint；再用CPU准备核对学生数据、规则参考与采样history。两模式恢复预检均通过后才启动任一GPU训练阶段。沿现有预算扩大到两模式所有目标epoch的checkpoint与日志；运行中仍保持20GiB磁盘保护。失败停止后续阶段，信号清理只处理本次子进程。

Phase4当前学生/producer合同及读取适配器源码未改变：有效v72 view、验证快照和v69路线缓存继续可用，不需要因为v73重建view。仅旧于当前合同的view仍需按已知迁移流程另建新view；旧合同checkpoint不能通过迁移冒充可恢复。多轮入口直接要求有效当前view，不隐式迁移resume的数据。

新训三轮（DATASET须为已经生成成功的当前view，不能填尚未生成的路径）：

```bash
export PREPARED=checkpoints/phase4_v57_20261014_232157_762998141
: "${DATASET:?设置为有效当前学生view目录}"
export DATASET
export P4_PRODUCTION_CACHE=checkpoints/p4_verified_routes_v69
export DATA_ROOT=lead_data MODEL_DIR=checkpoints/Qwen3.5-4B GPU_IDS=0,1,2,3
export BATCH_SIZE=2 ACCUMULATION=4 UPDATE_SAMPLES=32 GRADIENT_CHECKPOINTING=1
export EPOCHS=3
unset RESUME_RGB2 RESUME_RGB4
export OUTPUT_DIR="checkpoints/p4_epochs_$(date +%Y%m%d_%H%M%S_%N)"
bash qwen3vl_local/audit_joint/train_phase4_batch.sh
```

如有同view/源码匹配的 `validated_data`，可显式 `export P4_VALIDATED_DATA=旧输出目录/validated_data` 复用；否则在新输出内每模式准备一次。缓存仍完整执行失效检查。

从已完成的两模式第一轮继续到第3轮，保留上面的数据/模型/规则参数，并设置真实旧run目录：

```bash
: "${P4_PREVIOUS_RGB2:?设置两图旧run目录}"
: "${P4_PREVIOUS_RGB4:?设置四图旧run目录}"
export RESUME_RGB2="$P4_PREVIOUS_RGB2/epoch_000"
export RESUME_RGB4="$P4_PREVIOUS_RGB4/epoch_000"
export EPOCHS=3
export OUTPUT_DIR="checkpoints/p4_resumed_$(date +%Y%m%d_%H%M%S_%N)"
bash qwen3vl_local/audit_joint/train_phase4_batch.sh
```

已完成更多轮则显式指向相应epoch_NNN；保留旧run以及其best引用路径。新结果位于OUTPUT_DIR/run_training_rgb2和run_training_rgb4，最终测试使用epoch_002（目标3轮），不是固定epoch_000。进度/失败入口为OUTPUT_DIR/training_workflow.json及各阶段log；各run保留逐epoch采样、消费、val、latest/best、reload_check和test_fixed。多轮入口不调用只支持首轮的旧ZIP收集器，也不把执行成功写成训练已验收。

## 验证与限制

本地CPU小模型实际训练循环验证：Phase3连续3轮与1轮停下再恢复至3轮的参数、optimizer、scheduler、每rank RNG（本地world1）及history一致，后续batch ID/loss日志一致；Phase4两种RGB各自连续3轮与1+恢复2轮的参数、optimizer、history、样本顺序一致，并继承原最佳checkpoint。另覆盖部分写入拒绝、坏回执、目标轮数冲突、模式错配、未清梯度拒绝、磁盘不足及真实shell参数流。

验证未运行目标4B模型、四卡GPU恢复或服务器完整训练。评估仍主要逐题生成；图像视觉编码未去重；两模式评估可不同题。上述范围按本轮用户确认暂缓。
