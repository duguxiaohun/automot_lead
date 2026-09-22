#!/usr/bin/env bash
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 日常只改此数组：填写带时间戳的训练结果目录，可增加任意多个。
# 相对目录以执行命令时的当前目录为准；绝对目录也支持。
CKPT_DIRS=(
  "checkpoints/action_expert_ablation/bev_only/run_20260921_235146"
  "checkpoints/action_expert_ablation/bev_only/run_20260921_235150"
)

# 结果默认在AutoMoT/test/run_<时间>/，与checkpoints同级；不依赖启动目录。
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/../../test}"
# 自动最多选4张最空闲GPU；不足4张或ckpt更少时自动减少并发。
# 一卡一个ckpt，超出卡数的ckpt排队；GPU_IDS=0,1,2,3可显式指定卡号。
GPU_COUNT="${GPU_COUNT:-4}"

# 可视化采样：默认每个类别在 train/test 各取多少例。
CASES_PER_CATEGORY=8
# 按需取消注释并修改；没写的类别使用上面的默认值。0表示不输出该类别。
# train/ 或 test/ 前缀可只覆盖某个集合，例如 "test/UE7=6"。
EVENT_CASES=(
  # "UE1=12" "UE2=12" "UE3=8" "UE4=20" "UE5=8" "UE6=8" "UE7=8"
  # "RE2=12" "RE3=8" "RE5=12" "REGULAR_BACKGROUND=8" "UNCONFIRMED=4"
)
ACTION_CASES=(
  # "DECELERATE=12" "STOP=12" "RESUME=12"
  # "LANE_CHANGE_LEFT=20" "LANE_CHANGE_RIGHT=20" "KEEP=8" "UNCOND=8"
)
# 图中方法名，顺序对应 CKPT_DIRS；为空则自动命名。直接传目录时用 --names。
METHOD_NAMES=()
# 默认读取同帧meta真实相机标定；缺失才用LEAD名义标定。
# 需显式覆盖或添加已保存第三人称图标定时填JSON，正常LEAD三视角留空。
CAMERA_CONFIG=""
OPTIONS=(--cases-per-category "$CASES_PER_CATEGORY" --output-root "$OUTPUT_ROOT" --gpus "$GPU_COUNT")
for ITEM in "${EVENT_CASES[@]}"; do OPTIONS+=(--event-cases "$ITEM"); done
for ITEM in "${ACTION_CASES[@]}"; do OPTIONS+=(--action-cases "$ITEM"); done
if [[ -n "$CAMERA_CONFIG" ]]; then OPTIONS+=(--camera-config "$CAMERA_CONFIG"); fi

# 命令行参数放最后，同名数量设置覆盖脚本默认。
# 传入目录时覆盖上面数组；只传 --help/--plan-only 等选项时沿用数组。
if (( $# > 0 )) && [[ "$1" != --* ]]; then
  exec "${PYTHON:-python}" "$SCRIPT_DIR/compare_checkpoints.py" "${OPTIONS[@]}" "$@"
else
  if (( ${#METHOD_NAMES[@]} )); then OPTIONS+=(--names "${METHOD_NAMES[@]}"); fi
  exec "${PYTHON:-python}" "$SCRIPT_DIR/compare_checkpoints.py" "${CKPT_DIRS[@]}" "${OPTIONS[@]}" "$@"
fi
