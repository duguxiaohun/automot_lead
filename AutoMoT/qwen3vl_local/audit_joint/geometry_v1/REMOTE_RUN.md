# geometry_v1 白名单发布后的远程执行

任选服务器，只使用该机器自己的数据、地图和模型。以下命令不使用 SSH，不迁移资产，不覆盖旧实验。
先完成 CPU 全池建库和三轮采样计划，再启动四卡三轮 7680 次更新及最终 test。
启动预算默认 1 小时，训练单轮预算默认 10 小时；不是总耗时保证。当前没有 CLI resume。

## 1. 同步和填写本机路径

在远程仓库根目录执行；不要用 reset/clean 覆盖该机器已有工作。
先激活已有的项目 Qwen3.5 环境，替换下面四个路径；该 Python 还须能导入 CARLA 的离线地图 API。

```bash
git pull --ff-only origin main
cd AutoMoT
export PYTHON='/填写本机环境/bin/python'
export DATA_ROOT='/填写本机原始数据根目录'
export MAPS_DIR='/填写本机包含TownXX.xodr的目录'
export MODEL_DIR='/填写本机Qwen3.5模型目录'
export COLLECTION_DIR="$PWD/keyframe_filter/collection_output"
export ENTRY="$PWD/qwen3vl_local/audit_joint/geometry_v1/run.sh"
export GEOMETRY_ROOT="$PWD/checkpoints/geometry_v1"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

若 collection 位于别处，请明确修改 `COLLECTION_DIR`。若本机另有未纳入默认开发名单的曝光 manifest，
在 build 命令中追加 `--development-manifest /本机/manifest.json`；该 manifest 必须含 `additional_development.groups`。

## 2. 后台 CPU 建库和计划

`GEOMETRY_ROOT` 必须是新目录；若已存在，先核实原实验状态，不删除或直接重跑。正式构建不使用 `--scoped-smoke-only`。

```bash
test -d "$DATA_ROOT" && test -d "$MAPS_DIR" && test -d "$COLLECTION_DIR" &&
"$PYTHON" -c 'import carla, numpy, torch, transformers, peft; from PIL import Image; import os; from qwen3vl_local.sft_new_loop_phase3.preflight import check_model; check_model(os.environ["MODEL_DIR"])' &&
mkdir -p "$(dirname "$GEOMETRY_ROOT")" && mkdir "$GEOMETRY_ROOT" &&
nohup bash -c '
  set -euo pipefail
  bash "$ENTRY" build --collection "$COLLECTION_DIR" --data-root "$DATA_ROOT" \
    --maps "$MAPS_DIR" --output "$GEOMETRY_ROOT/data"
  bash "$ENTRY" plan --dataset "$GEOMETRY_ROOT/data/dataset.json" \
    --output "$GEOMETRY_ROOT/plan.json" --world-size 4 --per-event 1024 --epochs 3
  printf "%s\n" ready > "$GEOMETRY_ROOT/cpu.done"
' > "$GEOMETRY_ROOT/build_plan.log" 2>&1 < /dev/null &
```

查看 `tail -n 50 "$GEOMETRY_ROOT/build_plan.log"`。只有 `cpu.done` 存在才表示两步都成功。
检查 `data/receipt.json` 的事件／时距／类别数量、物理组支持、拒绝原因，
`data/split_coverage.json` 的最终覆盖和 `plan.json` 的三轮曝光计划。
最低一帧／一组仅表示覆盖，不是泛化认证；缺格不强行补标签。
失败时保留日志和已写回执；搜索预算耗尽是未判定，不能当成确定无解。

## 3. 四卡训练，成功后单卡 test

检查完 CPU 结果，在同一个已设置上述环境变量的终端执行。四个编号改为当前机器实际可用 GPU。
如果换了终端，重新设置第 1 步环境变量，但不要重建目录或重新建库。

```bash
export GPU_IDS=0,1,2,3
export GEOMETRY_EVAL_GPU=0
test -f "$GEOMETRY_ROOT/cpu.done" &&
test ! -e "$GEOMETRY_ROOT/run" && test ! -e "$GEOMETRY_ROOT/test" &&
nohup bash -c '
  set -euo pipefail
  bash "$ENTRY" train --dataset "$GEOMETRY_ROOT/data/dataset.json" \
    --model-dir "$MODEL_DIR" --output "$GEOMETRY_ROOT/run" \
    --per-event 1024 --epochs 3 --max-startup-hours 1 --max-epoch-hours 10
  GPU_IDS="$GEOMETRY_EVAL_GPU" bash "$ENTRY" evaluate \
    --dataset "$GEOMETRY_ROOT/data/dataset.json" --model-dir "$MODEL_DIR" \
    --checkpoint "$GEOMETRY_ROOT/run/epoch_3" --split test --output "$GEOMETRY_ROOT/test"
  printf "%s\n" complete > "$GEOMETRY_ROOT/train_test.done"
' > "$GEOMETRY_ROOT/train_test.log" 2>&1 < /dev/null &
```

查看 `tail -n 50 "$GEOMETRY_ROOT/train_test.log"`。训练失败或预算停止不会继续 test。
`run/validation_*.json` 是各轮固定验证子集指标；最终独立 evaluate 会核验并重载第三轮 checkpoint。
这里只训练 Phase3 九类预测器，不训练连续 expert，也不启动 CARLA 在线仿真。

回传 CPU 的 `receipt.json`、`split_coverage.json`、`plan.json`；训练后回传 `startup.json`、
`epoch_*.json`、`validation_*.json`、`test/metrics.json` 和对应日志。
先不要传原始 `dataset.json`、图像、模型权重或 `state.pt`。
