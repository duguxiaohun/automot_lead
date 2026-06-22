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
#   --minimal-videos           等价 --no-debug --no-demo --no-grid（只保留 input + bev_debug）
#   --no-aggregate             跑完不自动聚合
#
# 断点续跑（默认开启）：
#   - 开跑前扫描 eval_per_route/eval_<id>.json：status + score_composed 完整 → 跳过
#   - 半截 / 解析失败 / 无 status / 无 score → 删 eval/eval_latest + route<id>/ 重跑
#   - banner 打印 done / partial_cleaned / to_run 三个计数
#   - 每条 route 完成后用 mkdir-lock 原子自增计数，输出 [done k/Y] 进度
#   --no-resume                关闭断点续跑：清掉 picker 范围旧结果，整张列表强制重跑
#
# CARLA 启动：
#   - launcher 负责给每个 worker 扫描一个空闲端口起点
#   - 真正的 CARLA server 由 leaderboard_evaluator.py 启动并在退出时清理
#   - 这样避免 launcher / evaluator 双重启动 CARLA 抢端口
#
# 输出根：${EVAL_OUTPUT_BASE:-${PROJECT_ROOT}/outputs/closed_loop_eval}
#   <ckpt_parent>__<ckpt_stem>__bev{0|1}__ema{0|1}/
#       config.json
#       eval_per_route/eval_<route_id>.json   leaderboard 评测原始 json
#       route<route_id>/
#           input.mp4 debug.mp4 demo.mp4 grid.mp4
#           meta/<step>.json
#       runs/<RUN_LABEL>/log.txt             本次终端 stdout/stderr
#       runs/<RUN_LABEL>/summary_all.json
#       runs/<RUN_LABEL>/summary_report.md   人类可读实验总结
#       runs/<RUN_LABEL>/scenario_table.csv  论文表格友好的 scenario 汇总
#       runs/<RUN_LABEL>/route_results.csv   route 级明细
# ============================================================

set -u

# 禁用 core dump，避免工具进程异常时生成 core.*。
ulimit -S -c 0 2>/dev/null || true

# -------------------- 早期输出缓存 --------------------
# RUN_DIR 要等 ckpt 反查 + scenario picker + RUN_LABEL 拼好以后才知道，
# 在这之前 echo 出去的状态行（ckpt 解析、CARLA_ROOT、GPU 自动选址、port 起点等）
# 会落不到 log.txt。这里用一个 tempfile 把这些行存起来，等 RUN_DIR 创建后
# 一次性 append 到 log.txt 头部，保证“终端看到什么 log.txt 就有什么”。
# GNU mktemp 要求模板以 X 结尾，不能再加 .log 后缀；这里就用纯随机名做 buffer 文件。
EARLY_LOG=$(mktemp "${TMPDIR:-/tmp}/leadmot_eval_early.XXXXXX") || EARLY_LOG=""
say_early() {
    # 终端实时打印 + 同步追加到 EARLY_LOG；RUN_DIR 已知后再批量 flush 到 log.txt。
    printf "%s\n" "$*"
    if [ -n "${EARLY_LOG}" ] && [ -f "${EARLY_LOG}" ]; then
        printf "%s\n" "$*" >> "${EARLY_LOG}"
    fi
}

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
# 默认开启断点续跑：扫已有 eval_<id>.json 的 status/score 完整性，
# 完整的 skip、半截的删 + 重跑。--no-resume 清掉 picker 范围旧结果后强制全跑。
DO_RESUME="1"

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
        --minimal-videos)
            # 全量跑场景默认推荐：只保留 input + bev_debug，省 ffmpeg / 磁盘
            RECORD_DEBUG="0"; RECORD_DEMO="0"; RECORD_GRID="0"; shift ;;
        --no-resume) DO_RESUME="0"; shift ;;
        --no-aggregate) DO_AGGREGATE="0"; shift ;;
        --run-label) RUN_LABEL_OVERRIDE="$2"; shift 2 ;;
        --no-auto-carla) PRESTART_CARLA="0"; shift ;;
        --auto-carla)
            echo "WARN: --auto-carla is deprecated; CARLA is started by leaderboard_evaluator.py"
            PRESTART_CARLA="0"; shift ;;
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
say_early "Resolved LeadMoT checkpoint: ${LEADMOT_CKPT}"

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
    say_early "Auto-detected CARLA_ROOT=${CARLA_ROOT}"
fi

# -------------------- GPU 选址（项目统一规则） --------------------
# 默认走 nvidia-smi 按显存占用从低到高自动挑 GPU_COUNT 张空闲卡。
# 用户显式 pin 卡的方式（与 SFT / GoalGen / LeadMoT / VAE 训练入口一致）：
#   GPU_IDS=0 bash run_eval.sh ...                 单卡
#   GPU_IDS=0,1,2,3 bash run_eval.sh ...           4 卡指定
#   GPU_IDS=2,5 EVAL_GPU_COUNT=2 bash run_eval.sh  显式 2 个 worker 用卡 2 和 5
# GPU_IDS 非空时跳过 nvidia-smi 自动选址；GPU_COUNT 强制取 GPU_IDS 数量
# （与 --num-gpus / EVAL_GPU_COUNT 不一致会 warn 后以 GPU_IDS 为准）。
GPU_IDS_ENV="${GPU_IDS:-}"
unset CUDA_VISIBLE_DEVICES
GPU_IDS=()
if [ -n "${GPU_IDS_ENV}" ]; then
    # 接受逗号或空格分隔的 GPU id 序列：'0,1,2,3' / '0 1 2 3' 都行
    IFS=', ' read -r -a GPU_IDS <<< "${GPU_IDS_ENV}"
    for gid in "${GPU_IDS[@]}"; do
        if ! [[ "${gid}" =~ ^[0-9]+$ ]]; then
            echo "ERROR: GPU_IDS contains invalid token '${gid}'; expected comma-separated GPU indices like 0,1,2,3"
            exit 1
        fi
    done
    if [ "${#GPU_IDS[@]}" -ne "${GPU_COUNT}" ]; then
        # 不一致时尊重显式给出的 GPU_IDS（用户更可能改 GPU_IDS 来选卡，--num-gpus 留旧值）
        say_early "Note: --num-gpus=${GPU_COUNT} overridden by GPU_IDS=${GPU_IDS_ENV} (count=${#GPU_IDS[@]})"
        GPU_COUNT="${#GPU_IDS[@]}"
    fi
    say_early "GPU pinned by user: GPU_IDS=${GPU_IDS[*]} (workers=${GPU_COUNT})"
elif command -v nvidia-smi >/dev/null 2>&1; then
    # 按显存占用从低到高排序，取前 GPU_COUNT 张。这里取的是物理 GPU id，
    # 后续 run_evaluation.sh 会把它同时用于 CUDA_VISIBLE_DEVICES 和 CARLA graphicsadapter。
    mapfile -t GPU_IDS < <(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | sort -t ',' -k2 -n \
            | awk -F ',' -v n="${GPU_COUNT}" 'NR<=n {gsub(/ /, "", $1); print $1}'
    )
    say_early "Auto-selected GPU ids: ${GPU_IDS[*]} (workers=${GPU_COUNT})"
else
    say_early "WARN: nvidia-smi not found; falling back to GPU ids 0..$((GPU_COUNT - 1))"
    for ((i=0; i<GPU_COUNT; i++)); do GPU_IDS+=("${i}"); done
fi
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    GPU_IDS=("0")
fi
if [ "${#GPU_IDS[@]}" -lt "${GPU_COUNT}" ]; then
    say_early "WARN: requested ${GPU_COUNT} GPU(s), but only found ${#GPU_IDS[@]}; using ${#GPU_IDS[@]}"
    GPU_COUNT="${#GPU_IDS[@]}"
fi
if [ "${SINGLE_TEST}" = "1" ] && [ "${GPU_COUNT}" -gt 1 ]; then
    # single-test 是烟雾测试，固定只跑第一条 route；开多 worker 反而会浪费 CARLA 实例。
    say_early "single-test enabled; using one GPU worker"
    GPU_COUNT="1"
    GPU_IDS=("${GPU_IDS[0]}")
fi

PORT_BASE_START="${PORT_BASE_START:-5000}"
PORT_STRIDE="${PORT_STRIDE:-20}"
# 每个 worker 会启动一个 CARLA server。CARLA 除 RPC 端口外还会占用 streaming 等邻近端口，
# 所以用 PORT_STRIDE 给不同 GPU 留出端口槽，降低并行时端口冲突概率。
say_early "Port scan: start=${PORT_BASE_START}, stride=${PORT_STRIDE}; each worker gets a free [rpc..rpc+3, tm] block"

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

mapfile -t ROUTE_IDS_FULL < <(python3 "${PICKER}" "${PICKER_ARGS[@]}")
PICKER_TOTAL=${#ROUTE_IDS_FULL[@]}
if [ "${PICKER_TOTAL}" -eq 0 ]; then
    # 早退也清理 EARLY_LOG 临时文件，避免 /tmp 残留。
    if [ -n "${EARLY_LOG}" ] && [ -f "${EARLY_LOG}" ]; then
        rm -f "${EARLY_LOG}"
    fi
    echo "No route IDs to evaluate (after filters). Exit."
    exit 0
fi
if [ "${SINGLE_TEST}" = "1" ] && [ "${PICKER_TOTAL}" -gt 1 ]; then
    # 烟雾测试语义固定为 picker 第一条 route；resume 也只围绕这一条判断，不滚到第二条。
    ROUTE_IDS_FULL=("${ROUTE_IDS_FULL[0]}")
    PICKER_TOTAL=1
fi

# -------------------- 断点续跑扫描 --------------------
# 扫 PER_ROUTE_DIR/eval_<id>.json：
#   - status 是 leaderboard 终态(Completed/Perfect/Failed-...) + score_composed 非空 → 完整，跳过
#   - 文件存在但不完整(半截 / 解析失败 / 缺字段) → 删 eval/eval_latest + 整个 route<id>/ 目录，加入待跑
#   - 文件不存在 → 加入待跑；若 route<id>/ 或 eval_latest 残留，也清掉避免混旧视频/旧 checkpoint
# 把 ROUTE_IDS 替换为 to-run 子集；ROUTE_IDS_FULL 保留 picker 全集供 manifest 落 planned。
RESUME_DONE_COUNT=0
RESUME_CLEANED_COUNT=0
RESUME_CLEANED_IDS_STR=""
RESUME_DONE_IDS_STR=""
if [ "${DO_RESUME}" = "1" ]; then
    RESUME_OUT=$(
        PER_ROUTE_DIR="${PER_ROUTE_DIR}" \
        SIG_DIR="${SIG_DIR}" \
        ROUTE_IDS_STR="${ROUTE_IDS_FULL[*]}" \
        python3 - <<'PY'
import json, os, pathlib, re, shutil, sys

per_route_dir = pathlib.Path(os.environ["PER_ROUTE_DIR"])
sig_dir = pathlib.Path(os.environ["SIG_DIR"])
all_ids = [int(x) for x in os.environ.get("ROUTE_IDS_STR", "").split() if x]


def _route_id_from_record(record):
    """leaderboard record 里的 route_id 可能是 int 或 'RouteScenario_1711' 这种字符串。"""
    v = record.get("route_id")
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        m = re.search(r"\d+", v)
        if m:
            return int(m.group(0))
    return None


def is_complete(json_path: pathlib.Path, route_id: int) -> bool:
    """中判据：JSON 可解析 + 找到 route 对应 record + status 是终态 + score_composed 非空。

    leaderboard 半途崩溃留下的半截文件通常缺少 status 或 scores 字段，会被这里判为
    不完整、强制重跑。Status 白名单允许 'Failed - ...'：模型本身就是跑挂的场景
    （撞车 / 卡死）也是 leaderboard 正式给出的结论，不重跑。
    """
    if not json_path.is_file():
        return False
    try:
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    records = data.get("_checkpoint", {}).get("records", [])
    rec = None
    if isinstance(records, list):
        for r in records:
            if isinstance(r, dict) and _route_id_from_record(r) == route_id:
                rec = r
                break
        if rec is None and len(records) == 1 and isinstance(records[0], dict):
            rec = records[0]
    if not isinstance(rec, dict):
        return False
    status = rec.get("status")
    if not (isinstance(status, str) and status.strip()):
        return False
    if not (status in ("Completed", "Perfect") or status.startswith("Failed")):
        return False
    scores = rec.get("scores", {})
    if not isinstance(scores, dict):
        return False
    sc = scores.get("score_composed")
    if not isinstance(sc, (int, float)):
        return False
    return True


done_ids, cleaned_ids, todo_ids = [], [], []
for rid in all_ids:
    jp = per_route_dir / f"eval_{rid}.json"
    latest = per_route_dir / f"eval_latest_{rid}.json"
    route_dir = sig_dir / f"route{rid}"
    if is_complete(jp, rid):
        done_ids.append(rid)
        continue

    # 待跑 route 先清干净：坏 eval、旧 eval_latest、半截视频/meta 都不能混进新结果。
    cleaned = False
    if jp.is_file():
        try:
            jp.unlink()
            cleaned = True
        except Exception as e:
            print(f"[resume] WARN: failed to remove {jp}: {e}", file=sys.stderr)
    if latest.is_file():
        try:
            latest.unlink()
            cleaned = True
        except Exception as e:
            print(f"[resume] WARN: failed to remove {latest}: {e}", file=sys.stderr)
    if route_dir.is_dir():
        try:
            shutil.rmtree(route_dir)
            cleaned = True
        except Exception as e:
            print(f"[resume] WARN: failed to rmtree {route_dir}: {e}", file=sys.stderr)
    if cleaned:
        cleaned_ids.append(rid)
    todo_ids.append(rid)

print(f"DONE={len(done_ids)}")
print(f"CLEANED={len(cleaned_ids)}")
print(f"TO_RUN={len(todo_ids)}")
print("IDS=" + " ".join(str(x) for x in todo_ids))
print("DONE_IDS=" + " ".join(str(x) for x in done_ids))
print("CLEANED_IDS=" + " ".join(str(x) for x in cleaned_ids))
PY
    ) || { echo "ERROR: resume scan failed" >&2; exit 1; }

    RESUME_DONE_COUNT=$(printf "%s\n" "${RESUME_OUT}" | sed -n 's/^DONE=//p')
    RESUME_CLEANED_COUNT=$(printf "%s\n" "${RESUME_OUT}" | sed -n 's/^CLEANED=//p')
    RESUME_TO_RUN_COUNT=$(printf "%s\n" "${RESUME_OUT}" | sed -n 's/^TO_RUN=//p')
    TODO_IDS_STR=$(printf "%s\n" "${RESUME_OUT}" | sed -n 's/^IDS=//p')
    RESUME_DONE_IDS_STR=$(printf "%s\n" "${RESUME_OUT}" | sed -n 's/^DONE_IDS=//p')
    RESUME_CLEANED_IDS_STR=$(printf "%s\n" "${RESUME_OUT}" | sed -n 's/^CLEANED_IDS=//p')

    say_early "Resume scan: picker=${PICKER_TOTAL} | done=${RESUME_DONE_COUNT} | partial_cleaned=${RESUME_CLEANED_COUNT} | to_run=${RESUME_TO_RUN_COUNT}"
    if [ -n "${RESUME_CLEANED_IDS_STR}" ]; then
        say_early "Resume scan: cleaned route ids: ${RESUME_CLEANED_IDS_STR}"
    fi

    if [ -z "${TODO_IDS_STR}" ]; then
        ROUTE_IDS=()
    else
        # 用 read -a 把空格分隔的 id 串拆成数组
        IFS=' ' read -r -a ROUTE_IDS <<< "${TODO_IDS_STR}"
    fi
else
    # 关闭断点续跑：picker 全集旧结果先清掉，再直接当 to-run，保证是真正强制重跑。
    CLEAN_OUT=$(
        PER_ROUTE_DIR="${PER_ROUTE_DIR}" \
        SIG_DIR="${SIG_DIR}" \
        ROUTE_IDS_STR="${ROUTE_IDS_FULL[*]}" \
        python3 - <<'PY'
import os, pathlib, shutil, sys

per_route_dir = pathlib.Path(os.environ["PER_ROUTE_DIR"])
sig_dir = pathlib.Path(os.environ["SIG_DIR"])
all_ids = [int(x) for x in os.environ.get("ROUTE_IDS_STR", "").split() if x]
cleaned_ids = []

for rid in all_ids:
    cleaned = False
    for path in (per_route_dir / f"eval_{rid}.json", per_route_dir / f"eval_latest_{rid}.json"):
        if path.is_file():
            try:
                path.unlink()
                cleaned = True
            except Exception as e:
                print(f"[no-resume] WARN: failed to remove {path}: {e}", file=sys.stderr)
    route_dir = sig_dir / f"route{rid}"
    if route_dir.is_dir():
        try:
            shutil.rmtree(route_dir)
            cleaned = True
        except Exception as e:
            print(f"[no-resume] WARN: failed to rmtree {route_dir}: {e}", file=sys.stderr)
    if cleaned:
        cleaned_ids.append(rid)

print(f"CLEANED={len(cleaned_ids)}")
print("CLEANED_IDS=" + " ".join(str(x) for x in cleaned_ids))
PY
    ) || { echo "ERROR: --no-resume cleanup failed" >&2; exit 1; }
    RESUME_CLEANED_COUNT=$(printf "%s\n" "${CLEAN_OUT}" | sed -n 's/^CLEANED=//p')
    RESUME_CLEANED_IDS_STR=$(printf "%s\n" "${CLEAN_OUT}" | sed -n 's/^CLEANED_IDS=//p')
    ROUTE_IDS=("${ROUTE_IDS_FULL[@]}")
    say_early "Resume scan disabled (--no-resume); cleaned=${RESUME_CLEANED_COUNT}; will (re)run all ${PICKER_TOTAL} routes"
fi
TOTAL=${#ROUTE_IDS[@]}

# -------------------- 计算 RUN_LABEL --------------------
# 按跑法语义自动生成本批次目录名，让 random/scenario/full 的聚合互不污染。
# route 视频和 leaderboard json 仍写在 signature 根（共享、断点续跑）；
# 聚合报告、manifest 和 log.txt 落到 runs/<RUN_LABEL>/ 下。
if [ -n "${RUN_LABEL_OVERRIDE}" ]; then
    RUN_LABEL="${RUN_LABEL_OVERRIDE}"
elif [ "${SINGLE_TEST}" = "1" ]; then
    # resume 可能把 to-run 清空；run label 仍按 picker 第一条命名，避免 set -u 空数组崩溃。
    RUN_LABEL="smoke_${ROUTE_IDS_FULL[0]}"
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
        RUN_LABEL="${parts[0]}"
        for ((i = 1; i < ${#parts[@]}; i++)); do
            RUN_LABEL="${RUN_LABEL}__${parts[$i]}"
        done
    fi
fi
# 清洗：把不能进文件名的字符替换成 _；保留 "__" 作为组合过滤器分隔符。
RUN_LABEL=$(echo "${RUN_LABEL}" | tr -c 'A-Za-z0-9._+-' '_' | sed 's/^_//; s/_$//')
RUN_DIR="${SIG_DIR}/runs/${RUN_LABEL}"
mkdir -p "${RUN_DIR}"
if [ "${QWEN3VL_LOG_TO_FILE:-1}" != "0" ] && [ -z "${QWEN3VL_LOG_ACTIVE:-}" ]; then
    # 先把 RUN_DIR 已知之前的早期诊断（ckpt 路径、CARLA_ROOT、GPU 自动选址、Port 起点等）
    # 直接 append 到 log.txt 头部，再开 tee 接管后续 stdout/stderr；
    # 这样 log.txt 顺序与终端一致，没有“前面几行漏掉”的问题。
    if [ -n "${EARLY_LOG}" ] && [ -f "${EARLY_LOG}" ]; then
        cat "${EARLY_LOG}" >> "${RUN_DIR}/log.txt"
    fi
    export QWEN3VL_LOG_ACTIVE=1
    exec > >(tee -a "${RUN_DIR}/log.txt") 2>&1
    echo "[log] tee stdout/stderr to ${RUN_DIR}/log.txt (early diagnostics already prepended)"
fi
# EARLY_LOG 不再需要；cleanup_all_carla 中也会兜底清理。
if [ -n "${EARLY_LOG}" ] && [ -f "${EARLY_LOG}" ]; then
    rm -f "${EARLY_LOG}"
fi

# 写 run_manifest.json：让后续 aggregate / webapp 能精确知道这次跑了哪些 route_id。
# 用环境变量传递所有数组/字符串，避免 set -u 下空数组展开 unbound variable。
# manifest.route_ids / total_routes 用 picker 全集（planned 语义），to_run_count 单列
# 表示本次实际下发给 worker 跑的数量；这样 aggregate 的 coverage 仍按全量计算。
SCENARIOS_STR="${SCENARIOS[*]:-}"
ROUTE_IDS_ARG_STR="${ROUTE_IDS_ARG[*]:-}"
ROUTE_IDS_STR="${ROUTE_IDS_FULL[*]:-}"
TO_RUN_IDS_STR="${ROUTE_IDS[*]:-}"
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
TO_RUN_IDS_STR="${TO_RUN_IDS_STR}" \
PICKER_TOTAL="${PICKER_TOTAL}" \
TOTAL="${TOTAL}" \
DO_RESUME="${DO_RESUME}" \
RESUME_DONE_COUNT="${RESUME_DONE_COUNT}" \
RESUME_CLEANED_COUNT="${RESUME_CLEANED_COUNT}" \
RESUME_DONE_IDS_STR="${RESUME_DONE_IDS_STR}" \
RESUME_CLEANED_IDS_STR="${RESUME_CLEANED_IDS_STR}" \
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
    # route_ids / total_routes 是 picker 全集（planned）；resume 字段记录拆分情况
    "route_ids": _intlist(os.environ.get("ROUTE_IDS_STR", "")),
    "total_routes": int(os.environ.get("PICKER_TOTAL", "0") or "0"),
    "resume_enabled": int(os.environ.get("DO_RESUME", "1") or "0"),
    "resume_done_count": int(os.environ.get("RESUME_DONE_COUNT", "0") or "0"),
    "resume_done_routes": _intlist(os.environ.get("RESUME_DONE_IDS_STR", "")),
    "resume_cleaned_count": int(os.environ.get("RESUME_CLEANED_COUNT", "0") or "0"),
    "resume_cleaned_routes": _intlist(os.environ.get("RESUME_CLEANED_IDS_STR", "")),
    "to_run_count": int(os.environ.get("TOTAL", "0") or "0"),
    "to_run_routes": _intlist(os.environ.get("TO_RUN_IDS_STR", "")),
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
echo "Picker routes   : ${PICKER_TOTAL}"
if [ "${DO_RESUME}" = "1" ]; then
    echo "Resume          : enabled"
    echo "  already done  : ${RESUME_DONE_COUNT}"
    echo "  partial cleaned: ${RESUME_CLEANED_COUNT}"
    echo "  to run        : ${TOTAL}"
else
    echo "Resume          : disabled (--no-resume)"
    echo "  forced cleaned: ${RESUME_CLEANED_COUNT}"
    echo "  to run        : ${TOTAL}"
fi
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
echo "CARLA launch    : leaderboard_evaluator.py (launcher prestart disabled)"
echo "CARLA_ROOT      : ${CARLA_ROOT:-<unset>}"
echo "Record (i/d/bev/m/g): ${RECORD_INPUT}/${RECORD_DEBUG}/${RECORD_BEV_DEBUG}/${RECORD_DEMO}/${RECORD_GRID}"
echo "=========================================="

# to_run=0 早退：所有 route 都已完成，跳过 CARLA / worker 启动，直接走聚合
if [ "${TOTAL}" -eq 0 ]; then
    echo "[run_eval] All ${PICKER_TOTAL} planned routes already complete; nothing to run."
    if [ "${DO_AGGREGATE}" = "1" ]; then
        echo "[run_eval] Running aggregation for run_label=${RUN_LABEL}..."
        cd "${AUTOMOT_ROOT}" && python3 -m AutoMoT.qwen3vl_local.eval_carla.aggregate \
            --eval-base "${SAVE_PATH}" \
            --leadmot-ckpt "${LEADMOT_CKPT}" \
            --run-label "${RUN_LABEL}" \
            || echo "WARN: aggregation failed"
    fi
    exit 0
fi

# worker stdout/stderr 只放临时目录，用于实时 tail 和统计 attempted/failed。
# 跑完由 trap 删除，避免 closed_loop_eval 里长期堆过程日志。
WORK_LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/leadmot_eval_workers.XXXXXX") || {
    echo "ERROR: failed to create temporary worker log dir" >&2
    exit 1
}
echo "[run_eval] temporary worker logs: ${WORK_LOG_DIR} (will be removed on exit)"

# ============================================================
# 全局完成进度 counter（mkdir 锁，POSIX 原子）
# ============================================================
# 每个 worker 成功落 eval_<id>.json 后调 inc_done_counter，输出
#   [done k/TOTAL] route_id=... status=...
# 给主进程 tail 转发到终端。k 是跨 worker 累计完成数。
DONE_COUNTER_FILE="${WORK_LOG_DIR}/done_counter.txt"
DONE_COUNTER_LOCK="${WORK_LOG_DIR}/.done_counter.lock"
echo 0 > "${DONE_COUNTER_FILE}"

inc_done_counter() {
    local route_id="$1"
    local status="$2"
    # mkdir 作为 POSIX 原子锁：多个 worker 并发自增不会丢更新
    while ! mkdir "${DONE_COUNTER_LOCK}" 2>/dev/null; do sleep 0.05; done
    local cur
    cur=$(cat "${DONE_COUNTER_FILE}" 2>/dev/null || echo 0)
    cur=$((cur + 1))
    echo "${cur}" > "${DONE_COUNTER_FILE}"
    rmdir "${DONE_COUNTER_LOCK}"
    echo "[done ${cur}/${TOTAL}] route_id=${route_id} status=${status}"
}

read_route_status() {
    # 从落盘的 eval_<id>.json 抽 leaderboard status，给进度行显示。
    # 解析失败统一返回 '?'，不阻塞主流程。
    local eval_json="$1"
    local route_id="$2"
    EVAL_JSON="${eval_json}" ROUTE_ID="${route_id}" python3 - <<'PY' 2>/dev/null || echo '?'
import json, os, re, sys
try:
    p = os.environ["EVAL_JSON"]
    rid = int(os.environ["ROUTE_ID"])
    d = json.loads(open(p, encoding="utf-8-sig").read())
    recs = d.get("_checkpoint", {}).get("records", [])
    rec = None
    for r in recs:
        if isinstance(r, dict):
            v = r.get("route_id")
            if isinstance(v, int) and v == rid:
                rec = r; break
            if isinstance(v, str):
                m = re.search(r"\d+", v)
                if m and int(m.group(0)) == rid:
                    rec = r; break
    if rec is None and len(recs) == 1 and isinstance(recs[0], dict):
        rec = recs[0]
    if isinstance(rec, dict):
        s = rec.get("status")
        print(s if isinstance(s, str) and s.strip() else "?")
    else:
        print("?")
except Exception:
    print("?")
PY
}

# ============================================================
# CARLA 端口 helpers
# ============================================================
# 每个 worker 会通过 run_evaluation.sh 进入 leaderboard_evaluator.py，由 evaluator
# 自己启动独立 CARLA server。run_eval.sh 这里只负责预先给 worker 分配端口起点。
# run_evaluation.sh 里启动 leaderboard_evaluator.py 时会设
# `CUDA_VISIBLE_DEVICES=$gpu_rank`，模型和 CARLA 落在同一张物理卡，避免 PCIe 抖动。
# 端口：主进程从 PORT_BASE_START 开始扫描空闲端口块，分配给每个 worker。

# launcher 预启动 CARLA 已禁用；列表保留为空，cleanup 函数只作为兼容兜底。
LAUNCHED_CARLA_PORTS=()
PRESTART_CARLA="0"
CARLA_BOOT_TIMEOUT="${CARLA_BOOT_TIMEOUT:-90}"   # legacy, launcher no longer waits for CARLA

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
    # 从 start 开始按 PORT_STRIDE 步进，找一个 evaluator 不会再改写的空闲块。
    # leaderboard_evaluator.py 会要求 [p, p+1, p+2, p+3] 都可用；TM 默认 p+8000。
    # 失败（扫了 2000 还没找到）返回 1。
    local start="$1"
    local p="${start}"
    local max=$((start + 2000))
    while [ "${p}" -lt "${max}" ]; do
        if is_port_free "${p}" \
            && is_port_free "$((p + 1))" \
            && is_port_free "$((p + 2))" \
            && is_port_free "$((p + 3))" \
            && is_port_free "$((p + 8000))"; then
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

    if [ "${PRESTART_CARLA}" != "1" ]; then
        echo "[carla:gpu${gpu_rank}:port${port}] prestart disabled; evaluator will launch CARLA"
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
    if [ "${PRESTART_CARLA}" != "1" ]; then
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
        :
    else
        echo ""
        echo "[carla] cleanup: stopping ${#LAUNCHED_CARLA_PORTS[@]} CARLA process(es)"
        for p in "${LAUNCHED_CARLA_PORTS[@]}"; do
            stop_carla_for_port "${p}"
        done
    fi
    if [ -n "${WORK_LOG_DIR:-}" ] && [ -d "${WORK_LOG_DIR}" ]; then
        rm -rf "${WORK_LOG_DIR}"
    fi
    # 兜底：如果 RUN_DIR/log.txt flush 之前就崩了，把 EARLY_LOG 也一起清掉。
    if [ -n "${EARLY_LOG:-}" ] && [ -f "${EARLY_LOG}" ]; then
        rm -f "${EARLY_LOG}"
    fi
}
trap cleanup_all_carla EXIT INT TERM

# 主进程串行扫描空闲端口，每个 worker 分配一个独立端口块（RPC..RPC+3 / TM）。
# 串行分配避免并发竞态：两个 worker 同时探测同一个端口为空闲然后双双启动 CARLA 冲突。
WORKER_PORTS=()
next_start="${PORT_BASE_START}"
for ((widx=0; widx<GPU_COUNT; widx++)); do
    port=$(find_free_port_block "${next_start}") || {
        echo "ERROR: cannot find free CARLA port block from ${next_start}" >&2
        exit 1
    }
    WORKER_PORTS+=("${port}")
    if [ "${PRESTART_CARLA}" = "1" ]; then
        LAUNCHED_CARLA_PORTS+=("${port}")
    fi
    # 下一个 worker 从 port + PORT_STRIDE 开始扫，避免相邻端口段冲突
    next_start=$((port + PORT_STRIDE))
    echo "[port-alloc] worker ${widx} (gpu=${GPU_IDS[widx]}) -> rpc=${port} reserve=${port}-$((port+3)) tm=$((port+8000))"
done

run_route_worker() {
    # 单个 GPU worker：拿 worker_idx/gpu_rank 后，只处理 route_idx % GPU_COUNT == worker_idx 的路线。
    # 多卡时各 worker 互不通信，靠 eval_<route_id>.json 是否存在实现断点续跑。
    # 每个 worker 进入 leaderboard_evaluator.py 后在本卡上启动独立 CARLA server。
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

    # 默认不在 launcher 层预启动 CARLA，避免和 leaderboard_evaluator.py 双重启动。
    # evaluator 会在该 worker 进程里用相同 CUDA_VISIBLE_DEVICES 启动 CARLA 并负责退出清理。
    if ! start_carla_for_worker "${base_port}" "${gpu_rank}" "${carla_log}"; then
        echo "[worker ${worker_idx}] FATAL: CARLA failed to start; skipping all routes for this worker"
        return 1
    fi
    if [ "${PRESTART_CARLA}" = "1" ]; then
        trap "stop_carla_for_port ${base_port}" EXIT INT TERM
    fi

    for ((route_idx=worker_idx; route_idx<TOTAL; route_idx+=GPU_COUNT)); do
        # round-robin 分片比连续切段更适合断点续跑：如果某张卡提前失败，
        # 下次重跑仍会跳过其它 worker 已完成的 eval_<route_id>.json。
        local route_id="${ROUTE_IDS[$route_idx]}"
        local current=$((route_idx + 1))
        local per_route_json="${PER_ROUTE_DIR}/eval_${route_id}.json"
        local eval_latest="${PER_ROUTE_DIR}/eval_latest_${route_id}.json"
        if [ "${DO_RESUME}" = "1" ] && [ -f "${per_route_json}" ]; then
            echo "[worker ${worker_idx}] [${current}/${TOTAL}] skip route ${route_id} (already evaluated)"
            # 让全局进度计数仍然推进，避免多 worker race 时 TOTAL 对不上。
            inc_done_counter "${route_id}" "skipped"
            continue
        fi

        echo "[worker ${worker_idx}] [${current}/${TOTAL}] running route ${route_id}"
        echo "${route_id}" >> "${attempted_file}"
        # evaluator 没产出新 checkpoint 时，绝不能把旧 eval_latest 复制回 eval_<id>.json。
        rm -f "${eval_latest}"
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
            # 抽 status 给全局进度行用；解析失败用 '?'
            route_status=$(read_route_status "${per_route_json}" "${route_id}")
            inc_done_counter "${route_id}" "${route_status}"
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
FAILED_ROUTES_STR="${FAILED_ROUTES[*]:-}"

# 把过程状态固化到 run_manifest.json；不再长期保存 worker log。
RUN_DIR="${RUN_DIR}" \
ATTEMPTED_COUNT="${ATTEMPTED_COUNT}" \
FAILED_ROUTES_STR="${FAILED_ROUTES_STR}" \
WORKER_FAIL="${WORKER_FAIL}" \
python3 - <<'PY'
import json, os, pathlib, time

manifest_path = pathlib.Path(os.environ["RUN_DIR"]) / "run_manifest.json"
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception:
    manifest = {}
failed_routes = [int(x) for x in os.environ.get("FAILED_ROUTES_STR", "").split() if x]
manifest.update({
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "attempted_count": int(os.environ.get("ATTEMPTED_COUNT", "0") or "0"),
    "failed_route_count": len(failed_routes),
    "failed_routes": failed_routes,
    "worker_fail": int(os.environ.get("WORKER_FAIL", "0") or "0"),
})
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[run_eval] finalized manifest: {manifest_path}")
PY

echo ""
echo "Done. attempted=${ATTEMPTED_COUNT}, failed=${#FAILED_ROUTES[@]}, worker_fail=${WORKER_FAIL}"
if [ "${#FAILED_ROUTES[@]}" -gt 0 ]; then
    echo "Failed routes: ${FAILED_ROUTES[*]}"
fi
if [ "${DO_AGGREGATE}" = "1" ]; then
    echo "Running aggregation for run_label=${RUN_LABEL}..."
    cd "${AUTOMOT_ROOT}" && python3 -m AutoMoT.qwen3vl_local.eval_carla.aggregate \
        --eval-base "${SAVE_PATH}" \
        --leadmot-ckpt "${LEADMOT_CKPT}" \
        --run-label "${RUN_LABEL}" \
        || echo "WARN: aggregation failed"
fi
echo "=========================================="
