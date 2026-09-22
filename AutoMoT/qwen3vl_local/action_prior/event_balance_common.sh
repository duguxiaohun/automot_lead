#!/usr/bin/env bash
# 主线与两个消融共享开关解析；比例与算法只维护在 event_balance.py。
# demo：bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced
ulimit -S -c 0 2>/dev/null || true

action_event_balance_options() {
  ACTION_EVENT_SAMPLING_MODE=uniform
  ACTION_EVENT_BALANCE_INDEX="${EVENT_BALANCE_INDEX:-}"
  ACTION_EVENT_BALANCE_ARGS=()
  if [[ -v HIGH_LEVEL_ACTION_TOKEN ]]; then
    case "$HIGH_LEVEL_ACTION_TOKEN" in
      1) ACTION_EVENT_BALANCE_ARGS+=(--high-level-action-token) ;;
      0) ACTION_EVENT_BALANCE_ARGS+=(--no-high-level-action-token) ;;
      *) echo "HIGH_LEVEL_ACTION_TOKEN must be 0 or 1" >&2; return 2 ;;
    esac
  fi
  if [[ -v RGB_FRAME_COUNT ]]; then
    [[ "$RGB_FRAME_COUNT" == 1 || "$RGB_FRAME_COUNT" == 4 ]] || {
      echo "RGB_FRAME_COUNT must be 1 or 4" >&2; return 2;
    }
    ACTION_EVENT_BALANCE_ARGS+=(--rgb-frame-count "$RGB_FRAME_COUNT")
  fi
  local explicit_mode=0 explicit_index=0
  if [[ -v EVENT_BALANCED ]]; then
    explicit_mode=1
    [[ "$EVENT_BALANCED" != 1 ]] || ACTION_EVENT_SAMPLING_MODE=event_balanced
  fi
  if [[ -v ACTION_BALANCED ]]; then
    [[ "$ACTION_BALANCED" == 0 || "$ACTION_BALANCED" == 1 ]] || { echo "ACTION_BALANCED must be 0 or 1" >&2; return 2; }
    explicit_mode=1
    if [[ "$ACTION_BALANCED" == 1 ]]; then ACTION_EVENT_SAMPLING_MODE=action_balanced; else ACTION_EVENT_SAMPLING_MODE=uniform; fi
  fi
  [[ ! -v EVENT_BALANCE_INDEX ]] || explicit_index=1
  # 环境变量只在显式设置时转发，续训不注入新默认值；后面的 CLI 优先。
  local name option
  for name in EVENT_BALANCED_EPOCH_SAMPLES EVENT_BALANCE_MAX_FRAME_REPEATS BEST_SELECTION_METRIC; do
    case "$name" in
      EVENT_BALANCED_EPOCH_SAMPLES) option=--event-balanced-epoch-samples ;;
      EVENT_BALANCE_MAX_FRAME_REPEATS) option=--event-balance-max-frame-repeats ;;
      BEST_SELECTION_METRIC) option=--best-selection-metric ;;
    esac
    [[ ! -v "$name" ]] || ACTION_EVENT_BALANCE_ARGS+=("$option" "${!name}")
  done
  if [[ -v EVENT_BALANCE_ROUTE_DIVERSE ]]; then
    if [[ "$EVENT_BALANCE_ROUTE_DIVERSE" == 1 ]]; then
      ACTION_EVENT_BALANCE_ARGS+=(--event-balance-route-diverse)
    else
      ACTION_EVENT_BALANCE_ARGS+=(--no-event-balance-route-diverse)
    fi
  fi
  while (( $# )); do
    case "$1" in
      --event-balanced) ACTION_EVENT_SAMPLING_MODE=event_balanced; explicit_mode=1 ;;
      --action-balanced) ACTION_EVENT_SAMPLING_MODE=action_balanced; explicit_mode=1 ;;
      --no-event-balanced|--no-action-balanced) ACTION_EVENT_SAMPLING_MODE=uniform; explicit_mode=1 ;;
      --sampling-mode|--event-balance-index)
        option="$1"
        [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "$option needs a value" >&2; return 2; }
        if [[ "$option" == --sampling-mode ]]; then
          ACTION_EVENT_SAMPLING_MODE="$2"; explicit_mode=1
        else
          ACTION_EVENT_BALANCE_INDEX="$2"; explicit_index=1
        fi
        shift ;;
      --sampling-mode=*) ACTION_EVENT_SAMPLING_MODE="${1#*=}"; explicit_mode=1 ;;
      --event-balance-index=*) ACTION_EVENT_BALANCE_INDEX="${1#*=}"; explicit_index=1 ;;
      *) ACTION_EVENT_BALANCE_ARGS+=("$1") ;;
    esac
    shift
  done
  [[ "$ACTION_EVENT_SAMPLING_MODE" == uniform || "$ACTION_EVENT_SAMPLING_MODE" == event_balanced || "$ACTION_EVENT_SAMPLING_MODE" == action_balanced ]] || {
    echo "invalid sampling mode: $ACTION_EVENT_SAMPLING_MODE" >&2; return 2;
  }
  [[ "$explicit_mode" == 0 ]] || ACTION_EVENT_BALANCE_ARGS+=(--sampling-mode "$ACTION_EVENT_SAMPLING_MODE")
  [[ "$explicit_index" == 0 ]] || ACTION_EVENT_BALANCE_ARGS+=(--event-balance-index "$ACTION_EVENT_BALANCE_INDEX")
  return 0
}

action_prepare_event_balance_index() {
  # 只输出路径，构建/缓存复用/原子发布均由主线 Python 实现负责。
  python "$(dirname -- "${BASH_SOURCE[0]}")/prepare_event_balance.py" \
    --data-root "$1" --action-data-dir "$2"
}

action_shared_index_ready() {
  # manifest 最后发布；三 split 和 manifest 均完整后才能被另一条 pipeline 复用。
  local data_dir="$1"
  [[ -s "$data_dir/manifest.json" && -s "$data_dir/train.jsonl" && -s "$data_dir/val.jsonl" && -s "$data_dir/test.jsonl" ]]
}

action_build_shared_index_if_needed() {
  # 主线和两个消融都持有同一把锁，避免首次并发启动时覆盖同名 .tmp。
  local data_root="$1" data_dir="$2"
  if action_shared_index_ready "$data_dir"; then return 0; fi
  mkdir -p "$data_dir"
  (
    flock -x 9
    if ! action_shared_index_ready "$data_dir"; then
      python "$(dirname -- "${BASH_SOURCE[0]}")/build_dataset.py" \
        --data-root "$data_root" --output-dir "$data_dir"
    fi
  ) 9>"$data_dir/.build.lock"
}
