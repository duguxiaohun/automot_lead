#!/bin/bash
# ============================================================
# LeadMoT closed-loop evaluation on Bench2Drive 220 routes.
#
# 一键用法：
#   bash AutoMoT/qwen3vl_local/eval_carla/run_eval.sh \
#       --leadmot-ckpt /path/to/leadmot/best.pt
#
# 常用：
#   --leadmot-ckpt FILE|DIR    [必填] LeadMoT decoder checkpoint or output dir
#   --sensor-profile 3cam      默认 3cam（唯一支持的 LEAD 训练分布）
#   --step-stride N            每多少 tick 调一次模型，默认 5（4Hz）
#   --rope mrope|mhrope|none   默认 mrope
#   --num-gpus N               默认 1；自动选择 N 张空闲 GPU 并行跑 route
#   --single-test              只跑第一条 route 做烟雾
#
# 三种跑法（互相可叠加）：
#   1. 按场景：--scenario <Name>     可重复；只跑指定 LEAD scenario 子集
#   2. 随机 N：--random N            从筛后池里随机抽 N 条；--seed K 固定种子
#   3. 全量： 什么都不传，跑 220 条
#   附：--route-id <ID>              可重复；指定具体 route_id 跑
#
#   --no-input / --no-debug / --no-demo / --no-grid   关掉对应视频
#   --no-bev-debug             关掉 BEV 顶视 debug 视频
#   --no-aggregate             跑完不自动聚合
#
# CARLA 启动（默认开启）：
#   - 每个 worker 自动在自己 GPU 上启动一个独立 CARLA server（自动扫描空闲三端口块）
#   - worker 结束 / Ctrl+C / 异常退出 时自动 kill 对应 CARLA
#   - 已经手动起 CARLA 时加 `--no-auto-carla` 跳过自动启动（默认 `USE_AUTO_CARLA=1`）
#   - 启动后等 CARLA RPC 端口 listen 最多 `CARLA_BOOT_TIMEOUT=90` 秒
#
# 输出根：${EVAL_OUTPUT_BASE:-${PROJECT_ROOT}/outputs/closed_loop_eval}
#   <ckpt_parent>__<ckpt_stem>__bev{0|1}__ema{0|1}/
#       config.json
#       eval_per_route/eval_<route_id>.json   leaderboard 评测原始 json
#       route<route_id>/
#           input.mp4 debug.mp4 demo.mp4 grid.mp4
#           meta/<step>.json
#           logs/
#       scenarios/<Scenario>/summary.json    （聚合脚本写）
#       summary_all.json
# ============================================================

set -u

# -------------------- 参数解析 --------------------
LEADMOT_CKPT=""
SENSOR_PROFILE="3cam"
STEP_STRIDE="5"
LEADMOT_ROPE="mrope"
SINGLE_TEST="0"
GPU_COUNT="${EVAL_GPU_COUNT:-1}"
# GPU_COUNT 只表示“需要几张卡”；具体卡号始终由下面的 nvidia-smi 空闲选择逻辑决定。
# 这样与项目其它训练/eval 入口一致，不依赖用户手动设置 CUDA_VISIBLE_DEVICES。
SCENARIOS=()
ROUTE_IDS_ARG=()
RANDOM_N=""
RANDOM_SEED="0"
RECORD_INPUT="1"; RECORD_DEBUG="1"; RECORD_DEMO="1"; RECORD_GRID="1"
RECORD_BEV_DEBUG="1"
DO_AGGREGATE="1"
RUN_LABEL_OVERRIDE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --leadmot-ckpt) LEADMOT_CKPT="$2"; shift 2 ;;
        --sensor-profile) SENSOR_PROFILE="$2"; shift 2 ;;
        --step-stride) STEP_STRIDE="$2"; shift 2 ;;
        --rope) LEADMOT_ROPE="$2"; shift 2 ;;
        --num-gpus|--gpus) GPU_COUNT="$2"; shift 2 ;;
        --single-test) SINGLE_TEST="1"; shift ;;
        --scenario) SCENARIOS+=("$2"); shift 2 ;;
        --route-id) ROUTE_IDS_ARG+=("$2"); shift 2 ;;
        --random) RANDOM_N="$2"; shift 2 ;;
        --seed) RANDOM_SEED="$2"; shift 2 ;;
        --no-input)   RECORD_INPUT="0"; shift ;;
        --no-debug)   RECORD_DEBUG="0"; shift ;;
        --no-demo)    RECORD_DEMO="0"; shift ;;
        --no-grid)    RECORD_GRID="0"; shift ;;
        --no-bev-debug) RECORD_BEV_DEBUG="0"; shift ;;
        --no-aggregate) DO_AGGREGATE="0"; shift ;;
        --run-label) RUN_LABEL_OVERRIDE="$2"; shift 2 ;;
        --no-auto-carla) USE_AUTO_CARLA="0"; shift ;;
        --auto-carla) USE_AUTO_CARLA="1"; shift ;;
        -h|--help) sed -n '1,45p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${LEADMOT_CKPT}" ]; then
    echo "ERROR: --leadmot-ckpt is required"
    exit 1
fi
if [ "${SENSOR_PROFILE}" != "3cam" ]; then
    echo "ERROR: LeadMoT CARLA eval only supports --sensor-profile 3cam (LEAD training distribution)"
    exit 1
fi
if ! [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] || [ "${GPU_COUNT}" -lt 1 ]; then
    echo "ERROR: --num-gpus must be a positive integer, got ${GPU_COUNT}"
    exit 1
fi

LEADMOT_CKPT=$(LEADMOT_CKPT="${LEADMOT_CKPT}" python3 - <<'PY'
import os
import pathlib
import sys

root = pathlib.Path(os.environ["LEADMOT_CKPT"]).expanduser().resolve()
if root.is_file():
    print(root)
    sys.exit(0)
if not root.exists():
    print(f"ERROR: --leadmot-ckpt path does not exist: {root}", file=sys.stderr)
    sys.exit(1)
if not root.is_dir():
    print(f"ERROR: --leadmot-ckpt is neither file nor directory: {root}", file=sys.stderr)
    sys.exit(1)

candidates = [
    root / "best.pt",
    root / "latest.pt",
    root / "latest" / "best.pt",
    root / "latest" / "latest.pt",
]
for candidate in candidates:
    if candidate.is_file():
        print(candidate.resolve())
        sys.exit(0)

pools = []
for pattern in ("step-checkpoint-*.pt", "checkpoint-epoch*.pt", "*.pt", "*.safetensors"):
    pools.extend(p for p in root.glob(pattern) if p.is_file())
    latest = root / "latest"
    if latest.is_dir():
        pools.extend(p for p in latest.glob(pattern) if p.is_file())
if pools:
    print(max(pools, key=lambda p: p.stat().st_mtime).resolve())
    sys.exit(0)

print(
    f"ERROR: no LeadMoT checkpoint found under {root}; expected best.pt/latest.pt or checkpoint-*.pt",
    file=sys.stderr,
)
sys.exit(1)
PY
) || exit 1
echo "Resolved LeadMoT checkpoint: ${LEADMOT_CKPT}"

# -------------------- 路径自动探测 --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# AutoMoT/qwen3vl_local/eval_carla/ -> AutoMoT/qwen3vl_local/ -> AutoMoT/
QWEN_LOCAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUTOMOT_ROOT="$(cd "${QWEN_LOCAL_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${AUTOMOT_ROOT}/.." && pwd)"
LEADERBOARD_DIR="${AUTOMOT_ROOT}/leaderboard"

if [ -z "${CARLA_ROOT:-}" ]; then
    if [ -d "$(dirname "${AUTOMOT_ROOT}")/carla" ]; then
        export CARLA_ROOT="$(dirname "${AUTOMOT_ROOT}")/carla"
    elif [ -d "${HOME}/carla" ]; then
        export CARLA_ROOT="${HOME}/carla"
    else
        echo "ERROR: CARLA_ROOT not set"; exit 1
    fi
    echo "Auto-detected CARLA_ROOT=${CARLA_ROOT}"
fi

# -------------------- GPU 自动选址（项目统一规则） --------------------
unset CUDA_VISIBLE_DEVICES
GPU_IDS=()
if command -v nvidia-smi >/dev/null 2>&1; then
    # 按显存占用从低到高排序，取前 GPU_COUNT 张。这里取的是物理 GPU id，
    # 后续 run_evaluation.sh 会把它同时用于 CUDA_VISIBLE_DEVICES 和 CARLA graphicsadapter。
    mapfile -t GPU_IDS < <(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | sort -t ',' -k2 -n \
            | awk -F ',' -v n="${GPU_COUNT}" 'NR<=n {gsub(/ /, "", $1); print $1}'
    )
else
    echo "WARN: nvidia-smi not found; falling back to GPU ids 0..$((GPU_COUNT - 1))"
    for ((i=0; i<GPU_COUNT; i++)); do GPU_IDS+=("${i}"); done
fi
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    GPU_IDS=("0")
fi
if [ "${#GPU_IDS[@]}" -lt "${GPU_COUNT}" ]; then
    echo "WARN: requested ${GPU_COUNT} GPU(s), but only found ${#GPU_IDS[@]}; using ${#GPU_IDS[@]}"
    GPU_COUNT="${#GPU_IDS[@]}"
fi
if [ "${SINGLE_TEST}" = "1" ] && [ "${GPU_COUNT}" -gt 1 ]; then
    # single-test 是烟雾测试，固定只跑第一条 route；开多 worker 反而会浪费 CARLA 实例。
    echo "single-test enabled; using one GPU worker"
    GPU_COUNT="1"
    GPU_IDS=("${GPU_IDS[0]}")
fi
echo "Auto-selected GPU ids: ${GPU_IDS[*]} (workers=${GPU_COUNT})"

PORT_BASE_START="${PORT_BASE_START:-5000}"
PORT_STRIDE="${PORT_STRIDE:-20}"
# 每个 worker 会启动一个 CARLA server。CARLA 除 RPC 端口外还会占用 streaming 等邻近端口，
# 所以用 PORT_STRIDE 给不同 GPU 留出端口槽，降低并行时端口冲突概率。
echo "Port scan: start=${PORT_BASE_START}, stride=${PORT_STRIDE}; each worker gets a free [rpc, streaming, tm] block"

# -------------------- 输出目录 --------------------
EVAL_OUTPUT_BASE="${EVAL_OUTPUT_BASE:-${AUTOMOT_ROOT}/outputs/closed_loop_eval}"
SAVE_PATH="${EVAL_OUTPUT_BASE}"
CKPT_SIGNATURE=$(LEADMOT_CKPT="${LEADMOT_CKPT}" LEADMOT_USE_EMA="${LEADMOT_USE_EMA:-1}" python3 - <<'PY'
import os
import pathlib
import sys

ckpt = pathlib.Path(os.environ["LEADMOT_CKPT"]).resolve()
use_ema = os.environ.get("LEADMOT_USE_EMA", "1") != "0"

raw = {}
try:
    if ckpt.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        raw = load_file(str(ckpt))
    else:
        import torch
        raw = torch.load(str(ckpt), map_location="cpu")
except Exception as exc:
    print(f"[run_eval] ERROR: failed to inspect checkpoint for signature: {exc}", file=sys.stderr)
    sys.exit(1)

def unwrap_ema(state):
    # 兼容 LeadMoT train.py 存的 EMA shadow 格式；signature 只需要知道 raw/EMA 是否用于推理。
    if isinstance(state, dict) and isinstance(state.get("shadow"), dict):
        return state["shadow"]
    return state if isinstance(state, dict) else None

def strip_prefixes(sd):
    # 不同保存方式可能带 module./model./decoder. 前缀；去掉后只检查真实 decoder key。
    if not isinstance(sd, dict):
        return {}
    out = {}
    for key, value in sd.items():
        name = str(key)
        for prefix in ("module.", "model.", "decoder."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        out[name] = value
    return out

cfg = dict(raw.get("decoder_config", {})) if isinstance(raw, dict) else {}
if "use_bev" in cfg:
    use_bev = bool(cfg["use_bev"])
else:
    # 旧 checkpoint 如果没保存 decoder_config，就通过是否存在 bev_projector 参数推断 use_bev。
    state = None
    if isinstance(raw, dict):
        if use_ema:
            state = unwrap_ema(raw.get("ema_state_dict"))
        if state is None:
            for key in ("decoder", "state_dict", "model"):
                if isinstance(raw.get(key), dict):
                    state = raw[key]
                    break
        if state is None:
            state = raw
    state = strip_prefixes(state)
    use_bev = any(str(key).startswith("bev_projector.") for key in state)

print(f"{ckpt.parent.name}__{ckpt.stem}__bev{1 if use_bev else 0}__ema{1 if use_ema else 0}")
PY
)
SIG_DIR="${SAVE_PATH}/${CKPT_SIGNATURE}"
PER_ROUTE_DIR="${SIG_DIR}/eval_per_route"
mkdir -p "${PER_ROUTE_DIR}"

# -------------------- 决定要跑的 route_id 列表 --------------------
# 注意：bash 空数组用 `"${a[@]:-}"` 展开会得到一个空字符串元素（不是真空），
# 这会让 picker 收到 `--scenario ""` / `--route-id ""`，把所有 route 误过滤。
# 必须先用 `${#a[@]}` 长度判断，再用纯 `"${a[@]}"` 展开。
PICKER="${SCRIPT_DIR}/scenario_picker.py"
PICKER_ARGS=()
if [ ${#SCENARIOS[@]} -gt 0 ]; then
    for s in "${SCENARIOS[@]}"; do PICKER_ARGS+=(--scenario "$s"); done
fi
if [ ${#ROUTE_IDS_ARG[@]} -gt 0 ]; then
    for r in "${ROUTE_IDS_ARG[@]}"; do PICKER_ARGS+=(--route-id "$r"); done
fi
if [ -n "${RANDOM_N}" ]; then
    PICKER_ARGS+=(--random "${RANDOM_N}" --seed "${RANDOM_SEED}")
fi

mapfile -t ROUTE_IDS < <(python3 "${PICKER}" "${PICKER_ARGS[@]}")
TOTAL=${#ROUTE_IDS[@]}
if [ "${TOTAL}" -eq 0 ]; then
    echo "No route IDs to evaluate (after filters). Exit."
    exit 0
fi

# -------------------- 计算 RUN_LABEL --------------------
# 按跑法语义自动生成本批次目录名，让 random/scenario/full 的聚合互不污染。
# route 视频和 leaderboard json 仍写在 signature 根（共享、断点续跑）；
# 仅 scenarios/ 和 summary_all.json 落到 runs/<RUN_LABEL>/ 下。
if [ -n "${RUN_LABEL_OVERRIDE}" ]; then
    RUN_LABEL="${RUN_LABEL_OVERRIDE}"
elif [ "${SINGLE_TEST}" = "1" ]; then
    RUN_LABEL="smoke_${ROUTE_IDS[0]}"
else
    parts=()
    if [ ${#SCENARIOS[@]} -gt 0 ]; then
        # 多个 scenario 用 '+' 连接；不做编码以保持可读性
        IFS='+' joined="${SCENARIOS[*]}"; unset IFS
        parts+=("scenario_${joined}")
    fi
    if [ -n "${RANDOM_N}" ]; then
        parts+=("random_N${RANDOM_N}_S${RANDOM_SEED}")
    fi
    if [ ${#ROUTE_IDS_ARG[@]} -gt 0 ]; then
        # 列前 3 个 route_id，避免文件名过长
        head_ids="${ROUTE_IDS_ARG[*]:0:3}"
        head_ids_compact="${head_ids// /+}"
        if [ ${#ROUTE_IDS_ARG[@]} -gt 3 ]; then
            head_ids_compact="${head_ids_compact}_etc${#ROUTE_IDS_ARG[@]}"
        fi
        parts+=("routes_${head_ids_compact}")
    fi
    if [ ${#parts[@]} -eq 0 ]; then
        RUN_LABEL="full"
    else
        # 多个过滤器组合时用双下划线分隔
        IFS='__' joined_parts="${parts[*]}"; unset IFS
        RUN_LABEL="${joined_parts}"
    fi
fi
# 清洗：把不能进文件名的字符替换成 _
RUN_LABEL=$(echo "${RUN_LABEL}" | tr -c 'A-Za-z0-9._+-' '_' | sed 's/__*/_/g; s/^_//; s/_$//')
RUN_DIR="${SIG_DIR}/runs/${RUN_LABEL}"
mkdir -p "${RUN_DIR}"

# 写 run_manifest.json：让后续 aggregate / webapp 能精确知道这次跑了哪些 route_id。
# 用环境变量传递所有数组/字符串，避免 set -u 下空数组展开 unbound variable。
SCENARIOS_STR="${SCENARIOS[*]:-}"
ROUTE_IDS_ARG_STR="${ROUTE_IDS_ARG[*]:-}"
ROUTE_IDS_STR="${ROUTE_IDS[*]:-}"
GPU_IDS_STR="${GPU_IDS[*]:-}"
RUN_LABEL="${RUN_LABEL}" \
RUN_DIR="${RUN_DIR}" \
CKPT_SIGNATURE="${CKPT_SIGNATURE}" \
LEADMOT_CKPT="${LEADMOT_CKPT}" \
LEADMOT_USE_EMA="${LEADMOT_USE_EMA:-1}" \
SINGLE_TEST="${SINGLE_TEST}" \
SCENARIOS_STR="${SCENARIOS_STR}" \
ROUTE_IDS_ARG_STR="${ROUTE_IDS_ARG_STR}" \
RANDOM_N="${RANDOM_N:-}" \
RANDOM_SEED="${RANDOM_SEED}" \
ROUTE_IDS_STR="${ROUTE_IDS_STR}" \
TOTAL="${TOTAL}" \
RECORD_INPUT="${RECORD_INPUT}" \
RECORD_DEBUG="${RECORD_DEBUG}" \
RECORD_BEV_DEBUG="${RECORD_BEV_DEBUG}" \
RECORD_DEMO="${RECORD_DEMO}" \
RECORD_GRID="${RECORD_GRID}" \
SENSOR_PROFILE="${SENSOR_PROFILE}" \
STEP_STRIDE="${STEP_STRIDE}" \
GPU_COUNT="${GPU_COUNT}" \
GPU_IDS_STR="${GPU_IDS_STR}" \
python3 - <<'PY'
import json, os, time, pathlib
def _intlist(s: str) -> list[int]:
    return [int(x) for x in s.split() if x]
def _strlist(s: str) -> list[str]:
    return [x for x in s.split() if x]
manifest = {
    "run_label": os.environ["RUN_LABEL"],
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "signature": os.environ["CKPT_SIGNATURE"],
    "leadmot_ckpt": os.environ["LEADMOT_CKPT"],
    "leadmot_use_ema": int(os.environ.get("LEADMOT_USE_EMA", "1") or "0"),
    "single_test": int(os.environ.get("SINGLE_TEST", "0") or "0"),
    "scenarios_filter": _strlist(os.environ.get("SCENARIOS_STR", "")),
    "route_ids_filter": _strlist(os.environ.get("ROUTE_IDS_ARG_STR", "")),
    "random_n": int(os.environ["RANDOM_N"]) if os.environ.get("RANDOM_N") else 0,
    "random_seed": int(os.environ.get("RANDOM_SEED", "0") or "0"),
    "route_ids": _intlist(os.environ.get("ROUTE_IDS_STR", "")),
    "total_routes": int(os.environ.get("TOTAL", "0") or "0"),
    "record": {
        "input": bool(int(os.environ["RECORD_INPUT"])),
        "debug": bool(int(os.environ["RECORD_DEBUG"])),
        "bev_debug": bool(int(os.environ["RECORD_BEV_DEBUG"])),
        "demo": bool(int(os.environ["RECORD_DEMO"])),
        "grid": bool(int(os.environ["RECORD_GRID"])),
    },
    "sensor_profile": os.environ["SENSOR_PROFILE"],
    "step_stride": int(os.environ["STEP_STRIDE"]),
    "gpu_count": int(os.environ["GPU_COUNT"]),
    "gpu_ids": _intlist(os.environ.get("GPU_IDS_STR", "")),
}
out = pathlib.Path(os.environ["RUN_DIR"]) / "run_manifest.json"
out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[run_eval] wrote manifest: {out}")
PY

# -------------------- TEAM_AGENT --------------------
TEAM_AGENT="${SCRIPT_DIR}/agent.py"
TEAM_CONFIG_PREFIX="${LEADMOT_CKPT}"
ROUTES="${LEADERBOARD_DIR}/data/bench2drive220.xml"
TM_SEED="${TM_SEED:-3407}"

echo "=========================================="
echo "Total routes    : ${TOTAL}"
if [ ${#SCENARIOS[@]} -gt 0 ]; then echo "Filter scenario : ${SCENARIOS[*]}"; fi
if [ ${#ROUTE_IDS_ARG[@]} -gt 0 ]; then echo "Filter route_id : ${ROUTE_IDS_ARG[*]}"; fi
if [ -n "${RANDOM_N}" ]; then echo "Random sample   : ${RANDOM_N} (seed=${RANDOM_SEED})"; fi
echo "Single test     : ${SINGLE_TEST}"
echo "Save path       : ${SAVE_PATH}"
echo "Signature       : ${CKPT_SIGNATURE}"
echo "Run label       : ${RUN_LABEL}"
echo "Run dir         : ${RUN_DIR}"
echo "Leadmot ckpt    : ${LEADMOT_CKPT}"
echo "Sensor profile  : ${SENSOR_PROFILE}"
echo "Step stride     : ${STEP_STRIDE}"
echo "GPU workers     : ${GPU_COUNT} (${GPU_IDS[*]})"
echo "Use EMA weights : ${LEADMOT_USE_EMA:-1}"
echo "Auto CARLA      : ${USE_AUTO_CARLA:-1} (each worker spawns CARLA on its own GPU)"
echo "CARLA_ROOT      : ${CARLA_ROOT:-<unset>}"
echo "Record (i/d/bev/m/g): ${RECORD_INPUT}/${RECORD_DEBUG}/${RECORD_BEV_DEBUG}/${RECORD_DEMO}/${RECORD_GRID}"
echo "=========================================="

WORK_LOG_DIR="${SIG_DIR}/worker_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${WORK_LOG_DIR}"

# ============================================================
# CARLA 启动/停止 helpers
# ============================================================
# 每个 worker 在自己 GPU 上启动一个独立 CARLA server。
# 用 `CUDA_VISIBLE_DEVICES=$gpu_rank` 锁住 CARLA 看到的 GPU 列表，CARLA 自己只感知
# `cuda:0`；run_evaluation.sh 里启动 leaderboard_evaluator.py 时同样设
# `CUDA_VISIBLE_DEVICES=$gpu_rank`，模型和 CARLA 落在同一张物理卡，避免 PCIe 抖动。
# 端口：主进程从 PORT_BASE_START 开始扫描空闲三端口块，分配给每个 worker。
# 启动后 worker 在该端口上跑 evaluation；worker 函数返回时清理对应 CARLA。
# 脚本异常退出（Ctrl+C / kill）由 trap EXIT 兜底清理所有遗留 CARLA 进程。

# 已启动 CARLA 的 port 列表，用于 trap EXIT 时统一回收
LAUNCHED_CARLA_PORTS=()
USE_AUTO_CARLA="${USE_AUTO_CARLA:-1}"      # 设 0 关闭自动启动（手动管理 CARLA 的场景）
CARLA_BOOT_TIMEOUT="${CARLA_BOOT_TIMEOUT:-90}"   # 秒，等 RPC 端口 listen 的超时

is_port_free() {
    # 三种探测方法兜底：lsof > ss > python socket bind。任一可用都行。
    local p="$1"
    if command -v lsof >/dev/null 2>&1; then
        ! lsof -i:"${p}" >/dev/null 2>&1
    elif command -v ss >/dev/null 2>&1; then
        ! ss -ltn "( sport = :${p} )" 2>/dev/null | tail -n +2 | grep -q .
    else
        python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(('127.0.0.1', $p))
    s.close()
except OSError:
    sys.exit(1)
" 2>/dev/null
    fi
}

find_free_port_block() {
    # 从 start 开始按 PORT_STRIDE 步进，找一个 [p, p+1, p+8000] 三端口都空闲的块。
    # CARLA 需要 RPC(p) + Streaming(p+1) + TrafficManager(p+8000) 三个端口都可用。
    # 失败（扫了 2000 还没找到）返回 1。
    local start="$1"
    local p="${start}"
    local max=$((start + 2000))
    while [ "${p}" -lt "${max}" ]; do
        if is_port_free "${p}" && is_port_free "$((p + 1))" && is_port_free "$((p + 8000))"; then
            echo "${p}"
            return 0
        fi
        p=$((p + PORT_STRIDE))
    done
    return 1
}

start_carla_for_worker() {
    # 启动绑定到指定 GPU 的 CARLA，并等 RPC 端口可连接。
    local port="$1"
    local gpu_rank="$2"
    local log_file="$3"

    if [ "${USE_AUTO_CARLA}" != "1" ]; then
        echo "[carla:gpu${gpu_rank}:port${port}] USE_AUTO_CARLA=0, skip start"
        return 0
    fi

    if [ -z "${CARLA_ROOT:-}" ] || [ ! -f "${CARLA_ROOT}/CarlaUE4.sh" ]; then
        echo "[carla:gpu${gpu_rank}:port${port}] ERROR: CARLA_ROOT/CarlaUE4.sh not found (CARLA_ROOT='${CARLA_ROOT:-}')"
        return 1
    fi

    local streaming_port=$((port + 1))
    local tm_port=$((port + 8000))

    # 端口由主进程预先扫描为空闲；这里再做一次防竞态检查，避免误杀其它用户/任务的进程。
    if ! is_port_free "${port}" || ! is_port_free "${streaming_port}" || ! is_port_free "${tm_port}"; then
        echo "[carla:gpu${gpu_rank}:port${port}] ERROR: allocated port block is no longer free"
        return 1
    fi

    echo "[carla:gpu${gpu_rank}:port${port}] launching CARLA (log=${log_file})"
    # `-graphicsadapter=0` 配合 CUDA_VISIBLE_DEVICES 让 CARLA 用唯一可见的 GPU。
    # 关 motion blur 等代价是 0（CARLA 自带 Low 已经够轻）。
    CUDA_VISIBLE_DEVICES="${gpu_rank}" "${CARLA_ROOT}/CarlaUE4.sh" \
        -RenderOffScreen \
        -nosound \
        -carla-rpc-port=${port} \
        -traffic-manager-port=${tm_port} \
        -carla-streaming-port=${streaming_port} \
        -quality-level=Low \
        -resx=800 -resy=600 \
        -graphicsadapter=0 \
        > "${log_file}" 2>&1 &
    local carla_pid=$!
    echo "[carla:gpu${gpu_rank}:port${port}] pid=${carla_pid}, waiting for RPC port..."

    # 轮询 RPC 端口 listen 状态；CARLA 第一次加载 map 可能 30-60 秒。
    local elapsed=0
    while [ ${elapsed} -lt ${CARLA_BOOT_TIMEOUT} ]; do
        if command -v lsof >/dev/null 2>&1; then
            if lsof -i:${port} >/dev/null 2>&1; then
                echo "[carla:gpu${gpu_rank}:port${port}] ready after ${elapsed}s"
                return 0
            fi
        elif command -v ss >/dev/null 2>&1; then
            if ss -ltn "( sport = :${port} )" 2>/dev/null | grep -q LISTEN; then
                echo "[carla:gpu${gpu_rank}:port${port}] ready after ${elapsed}s"
                return 0
            fi
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        # CARLA 进程已经退出（启动失败）→ 提前止损
        if ! kill -0 "${carla_pid}" 2>/dev/null; then
            echo "[carla:gpu${gpu_rank}:port${port}] ERROR: CARLA exited prematurely; see ${log_file}"
            tail -n 30 "${log_file}" 2>/dev/null || true
            return 1
        fi
    done

    echo "[carla:gpu${gpu_rank}:port${port}] ERROR: RPC port not listening within ${CARLA_BOOT_TIMEOUT}s"
    tail -n 30 "${log_file}" 2>/dev/null || true
    return 1
}

stop_carla_for_port() {
    # kill 指定端口的 CARLA 进程（按端口匹配命令行参数定位）。
    local port="$1"
    if [ "${USE_AUTO_CARLA}" != "1" ]; then
        return 0
    fi
    local tm_port=$((port + 8000))
    local streaming_port=$((port + 1))
    pkill -9 -f "CarlaUE4.*-carla-rpc-port=${port}" 2>/dev/null || true
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti:${port} | xargs -r kill -9 2>/dev/null || true
        lsof -ti:${tm_port} | xargs -r kill -9 2>/dev/null || true
        lsof -ti:${streaming_port} | xargs -r kill -9 2>/dev/null || true
    fi
}

cleanup_all_carla() {
    # trap EXIT 兜底：清理 launcher 本次启动的所有 CARLA。
    if [ ${#LAUNCHED_CARLA_PORTS[@]} -eq 0 ]; then
        return 0
    fi
    echo ""
    echo "[carla] cleanup: stopping ${#LAUNCHED_CARLA_PORTS[@]} CARLA process(es)"
    for p in "${LAUNCHED_CARLA_PORTS[@]}"; do
        stop_carla_for_port "${p}"
    done
}
trap cleanup_all_carla EXIT INT TERM

# 主进程串行扫描空闲端口，每个 worker 分配一个独立 3 端口块（RPC / Streaming / TM）。
# 串行分配避免并发竞态：两个 worker 同时探测同一个端口为空闲然后双双启动 CARLA 冲突。
# 分配出来后立即加入 LAUNCHED_CARLA_PORTS，让 trap EXIT 兜底清理（应对 SIGKILL 场景）。
WORKER_PORTS=()
if [ "${USE_AUTO_CARLA}" = "1" ]; then
    next_start="${PORT_BASE_START}"
    for ((widx=0; widx<GPU_COUNT; widx++)); do
        port=$(find_free_port_block "${next_start}") || {
            echo "ERROR: cannot find free 3-port block from ${next_start}" >&2
            exit 1
        }
        WORKER_PORTS+=("${port}")
        LAUNCHED_CARLA_PORTS+=("${port}")
        # 下一个 worker 从 port + PORT_STRIDE 开始扫，避免相邻端口段冲突
        next_start=$((port + PORT_STRIDE))
        echo "[port-alloc] worker ${widx} (gpu=${GPU_IDS[widx]}) -> rpc=${port} streaming=$((port+1)) tm=$((port+8000))"
    done
else
    # USE_AUTO_CARLA=0：沿用旧固定槽位策略，假设用户手动起 CARLA 在 GPU_id 对应槽位
    for ((widx=0; widx<GPU_COUNT; widx++)); do
        WORKER_PORTS+=("$((PORT_BASE_START + GPU_IDS[widx] * PORT_STRIDE))")
    done
fi

run_route_worker() {
    # 单个 GPU worker：拿 worker_idx/gpu_rank 后，只处理 route_idx % GPU_COUNT == worker_idx 的路线。
    # 多卡时各 worker 互不通信，靠 eval_<route_id>.json 是否存在实现断点续跑。
    # 每个 worker 在本卡上自动起一个独立 CARLA server，结束 / trap 时回收。
    local worker_idx="$1"
    local gpu_rank="$2"
    # 端口由主进程串行扫描分配，避免与已有进程冲突；通过 WORKER_PORTS 数组下标查表。
    local base_port="${WORKER_PORTS[$worker_idx]}"
    local base_tm_port=$((base_port + 8000))
    local fail_file="${WORK_LOG_DIR}/failed_worker${worker_idx}.txt"
    local attempted_file="${WORK_LOG_DIR}/attempted_worker${worker_idx}.txt"
    local carla_log="${WORK_LOG_DIR}/carla_gpu${gpu_rank}_port${base_port}.log"
    : > "${fail_file}"
    : > "${attempted_file}"

    echo "[worker ${worker_idx}] gpu=${gpu_rank} port=${base_port} tm_port=${base_tm_port}"

    # 启动该 worker 专属 CARLA。失败时本 worker 直接放弃，让其它 worker 继续跑。
    # 注：subprocess 自身已经被主进程 trap EXIT 兜底；这里 worker 退出时也主动 stop。
    if ! start_carla_for_worker "${base_port}" "${gpu_rank}" "${carla_log}"; then
        echo "[worker ${worker_idx}] FATAL: CARLA failed to start; skipping all routes for this worker"
        return 1
    fi
    # worker subprocess 自己的 trap：正常返回 / kill / Ctrl+C 都清理 CARLA。
    # 主脚本另有 trap EXIT 兜底，这里是双保险（防止主脚本 trap 没触发）。
    trap "stop_carla_for_port ${base_port}" EXIT INT TERM

    for ((route_idx=worker_idx; route_idx<TOTAL; route_idx+=GPU_COUNT)); do
        # round-robin 分片比连续切段更适合断点续跑：如果某张卡提前失败，
        # 下次重跑仍会跳过其它 worker 已完成的 eval_<route_id>.json。
        local route_id="${ROUTE_IDS[$route_idx]}"
        local current=$((route_idx + 1))
        local per_route_json="${PER_ROUTE_DIR}/eval_${route_id}.json"
        if [ -f "${per_route_json}" ]; then
            echo "[worker ${worker_idx}] [${current}/${TOTAL}] skip route ${route_id} (already evaluated)"
            continue
        fi

        echo "[worker ${worker_idx}] [${current}/${TOTAL}] running route ${route_id}"
        echo "${route_id}" >> "${attempted_file}"
        local eval_latest="${PER_ROUTE_DIR}/eval_latest_${route_id}.json"
        LEADMOT_CKPT="${LEADMOT_CKPT}" \
        LEADMOT_USE_EMA="${LEADMOT_USE_EMA:-1}" \
        LEADMOT_ROPE="${LEADMOT_ROPE}" \
        SENSOR_PROFILE="${SENSOR_PROFILE}" \
        STEP_STRIDE="${STEP_STRIDE}" \
        RECORD_INPUT="${RECORD_INPUT}" \
        RECORD_DEBUG="${RECORD_DEBUG}" \
        RECORD_DEMO="${RECORD_DEMO}" \
        RECORD_GRID="${RECORD_GRID}" \
        RECORD_BEV_DEBUG="${RECORD_BEV_DEBUG}" \
        bash "${LEADERBOARD_DIR}/scripts/run_evaluation.sh" \
            "${base_port}" "${base_tm_port}" "True" \
            "${ROUTES}" "${TEAM_AGENT}" "${TEAM_CONFIG_PREFIX}+route${route_id}" \
            "${eval_latest}" "${SAVE_PATH}" "only_traj" "${gpu_rank}" \
            "${route_id}" "${TM_SEED}"
        local rc=$?
        if [ ${rc} -ne 0 ]; then
            echo "[worker ${worker_idx}] WARN: route ${route_id} failed (rc=${rc})"
            echo "${route_id}" >> "${fail_file}"
        fi
        if [ -f "${eval_latest}" ]; then
            cp "${eval_latest}" "${per_route_json}"
            echo "[worker ${worker_idx}] saved: ${per_route_json}"
        else
            echo "[worker ${worker_idx}] WARN: ${eval_latest} not found"
            if [ ${rc} -eq 0 ]; then
                echo "${route_id}" >> "${fail_file}"
            fi
        fi

        if [ "${SINGLE_TEST}" = "1" ]; then
            echo "[worker ${worker_idx}] ===== SINGLE_TEST_BREAK ====="
            break
        fi
    done
}

# 先创建日志文件，再启动 worker / tail，避免 tail 准备阶段截断 worker 的早期输出。
for ((worker_idx=0; worker_idx<GPU_COUNT; worker_idx++)); do
    : > "${WORK_LOG_DIR}/worker${worker_idx}.log"
done

# 实时进度输出：fork 一个 tail -F 跟所有 worker log 行内容
# - 单 worker：tail 仅一个文件，输出干净
# - 多 worker：tail 多文件时会自动加 "==> path <==" 头方便区分。
TAIL_PID=""
if command -v tail >/dev/null 2>&1; then
    if [ "${GPU_COUNT}" -gt 1 ]; then
        tail -n 0 -F "${WORK_LOG_DIR}"/worker*.log 2>/dev/null &
        TAIL_PID="$!"
    else
        # 单 worker 直接全量 follow，更干净
        tail -n 0 -F "${WORK_LOG_DIR}/worker0.log" 2>/dev/null &
        TAIL_PID="$!"
    fi
fi
echo "[run_eval] tail PID=${TAIL_PID:-<none>}, streaming worker logs to stdout..."

PIDS=()
for ((worker_idx=0; worker_idx<GPU_COUNT; worker_idx++)); do
    # 每个 worker 的 stdout/stderr 写独立日志；上面的 tail -F 实时回放到主进程 stdout。
    run_route_worker "${worker_idx}" "${GPU_IDS[$worker_idx]}" > "${WORK_LOG_DIR}/worker${worker_idx}.log" 2>&1 &
    PIDS+=("$!")
done

WORKER_FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        WORKER_FAIL=1
    fi
done

# 所有 worker 跑完后停掉 tail；最后再 cat 一次保证所有 buffer 都 flush
if [ -n "${TAIL_PID}" ]; then
    kill "${TAIL_PID}" 2>/dev/null || true
    wait "${TAIL_PID}" 2>/dev/null || true
fi
echo ""
echo "===== final worker log tails ====="
for log in "${WORK_LOG_DIR}"/worker*.log; do
    [ -f "${log}" ] || continue
    echo ""
    echo "--- $(basename "${log}") (last 30 lines) ---"
    tail -n 30 "${log}"
done

# 聚合所有 worker 的失败 route，最终统一打印，方便下一轮按 --route-id 精确重跑。
mapfile -t FAILED_ROUTES < <(cat "${WORK_LOG_DIR}"/failed_worker*.txt 2>/dev/null | sed '/^$/d' | sort -n | uniq)
ATTEMPTED_COUNT=$(cat "${WORK_LOG_DIR}"/attempted_worker*.txt 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' ')

echo ""
echo "Done. attempted=${ATTEMPTED_COUNT}, failed=${#FAILED_ROUTES[@]}, worker_fail=${WORKER_FAIL}"
if [ "${#FAILED_ROUTES[@]}" -gt 0 ]; then
    echo "Failed routes: ${FAILED_ROUTES[*]}"
fi
if [ "${DO_AGGREGATE}" = "1" ] && [ "${SINGLE_TEST}" != "1" ]; then
    echo "Running aggregation for run_label=${RUN_LABEL}..."
    cd "${AUTOMOT_ROOT}" && python3 -m AutoMoT.qwen3vl_local.eval_carla.aggregate \
        --eval-base "${SAVE_PATH}" \
        --leadmot-ckpt "${LEADMOT_CKPT}" \
        --run-label "${RUN_LABEL}" \
        || echo "WARN: aggregation failed"
fi
echo "=========================================="
