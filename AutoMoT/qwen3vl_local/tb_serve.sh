#!/usr/bin/env bash
# 通用 TensorBoard 启动器 — 一条命令搞定"远程跑 TB + 本地点连接就能看"。
#
# 设计目的：
#   你最常用的"远程训练 → 本地浏览器看 TB"流程被三件事卡住——
#     1. TB 默认绑定 localhost:6006，远端 firewall + 没 --bind_all 时，
#        本地浏览器直接点 stdout 那个 http://hostname:6006/ 必然连不上。
#     2. 端口可能已经被上一次 TB 占了，TB 默认不会自动避让，启动直接报
#        "Address already in use"。
#     3. 你需要在另一个本地终端开 ssh -L 隧道，但常常忘记端口或写错。
#   本脚本把这三步合并：自动挑空闲端口、--bind_all 起 TB、在 stdout 显著
#   打印两条可直接复制的命令（ssh 隧道 + 浏览器 URL）。
#
# 用法（远端，在 AutoMoT/ 目录下）：
#   bash qwen3vl_local/tb_serve.sh checkpoints/sft_v1_lora
#   bash qwen3vl_local/tb_serve.sh checkpoints/goalgen_v1_dit
#
# 想同时看训练 + eval 两条 TB run，把 logdir 指到 OUTPUT_DIR 根目录即可。
# 本脚本会主动发现标准 tb/、eval_tb/ 目录，解析 latest 这类软链接后以
# --logdir_spec 传给 TensorBoard。不要依赖 TensorBoard 2.21 fast loader 对父目录
# 和软链接的递归扫描：该路径在部分版本中会显示 "No dashboards are active"。
#
# 可选环境变量 override：
#   TB_PORT=6007   ← 强制端口；不传时自动从 OS 取空闲端口
#   TB_BIND=0.0.0.0 ← 改 bind 地址；默认 --bind_all（等价 0.0.0.0）
#   TB_EXTRA="--samples_per_plugin images=200"  ← 透传给 tensorboard 的额外参数
#   TB_DISCOVER=0  ← 关闭 event 目录展开，完全交给 TensorBoard 原生 --logdir 扫描
#   TB_DISCOVER_DEPTH=4 ← 默认只枚举此深度内的 tb/、eval_tb/（不逐文件扫描模型/权重）
#   TB_DISCOVER_DEEP=1  ← 罕见的非标准目录布局才启用完整递归扫描；大 checkpoints 会较慢
#   TB_LOAD_FAST=0 ← 传 --load_fast=false；仅用于排查 TensorBoard 自身扫描问题
#   TB_SCALAR_SAMPLES=0 ← scalar 不做 TensorBoard reservoir 抽样（0 = 保留全部点）
#
# 注：本机已有别的方式把远端 6006 通到本地（VSCode Remote 自动端口转发 / 现成隧道 /
# 公网 IP 直连），脚本不再打印 ssh 隧道命令；只打印浏览器直接打开的 URL。
#
# 退出：
#   Ctrl-C 一次即可，trap 会确保 TB 子进程一起退出，不会留僵尸。

set -euo pipefail

# 禁用 core dump，避免工具进程异常时生成 core.*。
ulimit -S -c 0 2>/dev/null || true

LOGDIR="${1:-}"
if [[ -z "${LOGDIR}" ]]; then
    cat >&2 <<EOF
用法: bash qwen3vl_local/tb_serve.sh <logdir>

<logdir> 是 TensorBoard 的 --logdir，常用：
  checkpoints/sft_v1_lora        ← SFT v1：会同时看到 tb/ 和 eval_tb/ 两个 run
  checkpoints/goalgen_v1_dit     ← GoalGen v1：同上
  checkpoints/sft_v1_lora/tb     ← 只看训练曲线（不显示 eval）
  checkpoints/sft_v1_lora/eval_tb ← 只看 eval 指标曲线
EOF
    exit 1
fi

if [[ ! -d "${LOGDIR}" ]]; then
    echo "[tb][warn] logdir 不存在：${LOGDIR}（继续启动，等首次 events 写入后会自动出现 run）" >&2
fi

# ---- 1. 选端口 ----
# 如果用户传了 TB_PORT 且占用，强制覆盖会直接撞 "Address already in use"。
# 这里给用户一次选择：占用就 fallback 到 OS 自动端口，并打印 warn。
pick_free_port() {
    python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}

is_port_free() {
    local port="$1"
    python - "${port}" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

if [[ -n "${TB_PORT:-}" ]]; then
    if is_port_free "${TB_PORT}"; then
        PORT="${TB_PORT}"
    else
        echo "[tb][warn] TB_PORT=${TB_PORT} 已被占用，自动改用空闲端口" >&2
        PORT="$(pick_free_port)"
    fi
else
    PORT="$(pick_free_port)"
fi

# ---- 2. 决定 bind 参数 ----
# 默认用 --bind_all（TensorBoard 老语义：监听所有网络接口）。某些公共集群禁止
# bind 0.0.0.0，可以传 TB_BIND=127.0.0.1 退到 localhost，再用 ssh -L 转发。
BIND_FLAGS=("--bind_all")
if [[ -n "${TB_BIND:-}" ]]; then
    BIND_FLAGS=("--host" "${TB_BIND}")
fi

# ---- 3. 拼额外参数 ----
# TB_EXTRA 允许用户传 `--samples_per_plugin images=200` 之类，扩大图像样例的保留数。
EXTRA_ARGS=()
if [[ -n "${TB_EXTRA:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=(${TB_EXTRA})
fi

# -P 把 latest 之类的父级软链接解析成实际目录。目录不存在时保留一个可读的
# 绝对显示路径。
LOGDIR_ABS="$(cd -P -- "$(dirname -- "${LOGDIR}")" 2>/dev/null && pwd || pwd)/$(basename -- "${LOGDIR}")"

# ---- 4. 找到实际 event 叶目录 ----
# TensorBoard 2.21 默认启用 fast data loading。它在一些环境不会把父目录下的
# latest/ -> run_xxx/ 软链接递归成可用 run；显式 --logdir_spec 指向每个 event
# 所在目录可避开这个版本差异。默认只扫描目录项而不遍历 checkpoint 内所有文件，
# 防止 `checkpoints/` 内模型权重很多时端口提示长期卡住。
TB_INPUT_ARGS=(--logdir "${LOGDIR}")
DISCOVERED_EVENT_DIRS=()
DISCOVERED_EVENT_LABELS=()
if [[ "${TB_DISCOVER:-1}" != "0" && -d "${LOGDIR_ABS}" ]]; then
    declare -A SEEN_EVENT_DIRS=()
    register_event_dir() {
        local event_dir="$1"
        local real_event_dir event_label
        real_event_dir="$(readlink -f -- "${event_dir}" 2>/dev/null || printf '%s' "${event_dir}")"
        # latest/tb 和 run_xxx/tb 可能是同一物理目录；只显示一次。
        if [[ -n "${SEEN_EVENT_DIRS[${real_event_dir}]+x}" ]]; then
            return
        fi
        SEEN_EVENT_DIRS["${real_event_dir}"]=1
        if [[ "${event_dir}" == "${LOGDIR_ABS}" ]]; then
            event_label="$(basename -- "${LOGDIR_ABS}")"
        else
            event_label="${event_dir#"${LOGDIR_ABS%/}/"}"
        fi
        DISCOVERED_EVENT_DIRS+=("${real_event_dir}")
        DISCOVERED_EVENT_LABELS+=("${event_label}")
    }

    case "${TB_DISCOVER_DEEP:-0}" in
        0|false|False|FALSE|"")
            DISCOVER_DEPTH="${TB_DISCOVER_DEPTH:-4}"
            if [[ ! "${DISCOVER_DEPTH}" =~ ^[0-9]+$ ]] || (( DISCOVER_DEPTH < 1 )); then
                echo "[tb][error] TB_DISCOVER_DEPTH 必须是正整数，当前为：${DISCOVER_DEPTH}" >&2
                exit 2
            fi
            echo "[tb] 正在快速发现 tb/eval_tb（最多 ${DISCOVER_DEPTH} 层；不扫描权重文件）..." >&2
            # os.scandir 只检查目录项，且到 tb/eval_tb 立刻停止向下递归；比 find 全量
            # 扫 events 更适合 checkpoints/ 这种同时存放大模型权重的根目录。NUL 输出
            # 保留带空格路径，realpath visited 则防止 latest 链接形成重复/循环扫描。
            while IFS= read -r -d '' candidate_dir; do
                while IFS= read -r -d '' event_dir; do
                    register_event_dir "${event_dir}"
                done < <(
                    find -L "${candidate_dir}" -maxdepth 1 -type f \
                        -name 'events.out.tfevents.*' -printf '%h\0' 2>/dev/null | sort -zu
                )
            done < <(
                python - "${LOGDIR_ABS}" "${DISCOVER_DEPTH}" <<'PY'
import os
import sys

root = os.path.abspath(sys.argv[1])
max_depth = int(sys.argv[2])
leaf_names = {"tb", "eval_tb"}
visited = set()

def emit(path):
    sys.stdout.buffer.write(os.fsencode(path) + b"\0")

def walk(path, remaining, *, is_root=False):
    try:
        resolved = os.path.realpath(path)
    except OSError:
        return
    if resolved in visited:
        return
    visited.add(resolved)
    if is_root:
        # 支持用户直接传某个 tb/ 或任意 event 所在目录。
        emit(path)
    if remaining <= 0:
        return
    try:
        with os.scandir(path) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
    except OSError:
        return
    for child in children:
        try:
            if not child.is_dir(follow_symlinks=True):
                continue
        except OSError:
            continue
        if child.name in leaf_names:
            emit(child.path)
            # event 叶目录不再向下扫描 profiles/ 等内容。
            continue
        walk(child.path, remaining - 1)

walk(root, max_depth, is_root=True)
PY
            )
            ;;
        1|true|True|TRUE)
            echo "[tb][warn] 正在完整递归发现 event；大 checkpoints 可能需要较长时间..." >&2
            while IFS= read -r -d '' event_dir; do
                register_event_dir "${event_dir}"
            done < <(
                find -L "${LOGDIR_ABS}" -type f -name 'events.out.tfevents.*' \
                    -printf '%h\0' 2>/dev/null | sort -zu
            )
            ;;
        *)
            echo "[tb][error] TB_DISCOVER_DEEP 只能是 0 或 1，当前为：${TB_DISCOVER_DEEP}" >&2
            exit 2
            ;;
    esac

    # 单一叶目录仍沿用 --logdir，保留原来的 run 命名；父目录或多叶目录则显式
    # 提供全部实际路径，彻底避开 fast loader 的链接发现差异。
    if (( ${#DISCOVERED_EVENT_DIRS[@]} > 1 )) || {
        (( ${#DISCOVERED_EVENT_DIRS[@]} == 1 )) && [[ "${DISCOVERED_EVENT_DIRS[0]}" != "${LOGDIR_ABS}" ]]
    }; then
        LOGDIR_SPEC=""
        for index in "${!DISCOVERED_EVENT_DIRS[@]}"; do
            # TensorBoard 的 logdir_spec 为 name:path；本项目的 run 路径不含逗号/冒号。
            LOGDIR_SPEC+="${LOGDIR_SPEC:+,}${DISCOVERED_EVENT_LABELS[index]}:${DISCOVERED_EVENT_DIRS[index]}"
        done
        TB_INPUT_ARGS=(--logdir_spec "${LOGDIR_SPEC}")
    fi
fi

# ---- 5. 可选的 TensorBoard loader / scalar 保留策略 ----
LOADER_ARGS=()
case "${TB_LOAD_FAST:-auto}" in
    auto|"") ;;
    0|false|False|FALSE) LOADER_ARGS=(--load_fast=false) ;;
    1|true|True|TRUE) LOADER_ARGS=(--load_fast=true) ;;
    *)
        echo "[tb][error] TB_LOAD_FAST 只能是 auto、0 或 1，当前为：${TB_LOAD_FAST}" >&2
        exit 2
        ;;
esac

SCALAR_ARGS=()
if [[ -n "${TB_SCALAR_SAMPLES:-}" ]]; then
    if [[ ! "${TB_SCALAR_SAMPLES}" =~ ^[0-9]+$ ]]; then
        echo "[tb][error] TB_SCALAR_SAMPLES 必须是非负整数（0 表示保留全部 scalar）" >&2
        exit 2
    fi
    if [[ "${TB_EXTRA:-}" == *"--samples_per_plugin"* ]]; then
        echo "[tb][warn] TB_EXTRA 已包含 --samples_per_plugin，忽略 TB_SCALAR_SAMPLES" >&2
    else
        SCALAR_ARGS=(--samples_per_plugin "scalars=${TB_SCALAR_SAMPLES}")
    fi
fi

# ---- 6. 启动 TB ----
echo "============================================================"
echo "[tb] logdir: ${LOGDIR_ABS}"
echo "[tb] port:   ${PORT}"
echo "[tb] bind:   ${BIND_FLAGS[*]}"
if (( ${#DISCOVERED_EVENT_DIRS[@]} )); then
    echo "[tb] events: ${#DISCOVERED_EVENT_DIRS[@]} 个叶目录（软链接已解析）"
    if [[ "${TB_INPUT_ARGS[0]}" == "--logdir_spec" ]]; then
        echo "[tb] mode:   explicit logdir_spec（父目录/软链接兼容）"
    fi
else
    echo "[tb][warn] 当前未发现 event 文件；训练首次写入后请重启本脚本以展开新的子 run" >&2
fi
if [[ -n "${TB_SCALAR_SAMPLES:-}" ]]; then
    echo "[tb] scalar samples/tag: ${TB_SCALAR_SAMPLES}"
fi
echo "============================================================"

# 后台起 TB，捕获 PID 用于 trap 清理。
tensorboard "${TB_INPUT_ARGS[@]}" --port "${PORT}" "${BIND_FLAGS[@]}" \
    "${LOADER_ARGS[@]}" "${SCALAR_ARGS[@]}" "${EXTRA_ARGS[@]}" &
TB_PID=$!

cleanup() {
    echo ""
    echo "[tb] 收到退出信号，正在关闭 TensorBoard (PID=${TB_PID}) ..."
    # kill -TERM 让 TB 优雅退出；kill -KILL 兜底防止它 hang 住。
    kill -TERM "${TB_PID}" 2>/dev/null || true
    wait "${TB_PID}" 2>/dev/null || true
    echo "[tb] 已退出。"
}
trap cleanup INT TERM EXIT

# 等 TB 启动到能接受连接（默认给 5 秒）。
# 这段不是必须，但能让"连接命令"提示出现得更靠谱：TB 还没 listen 就打印 ssh 命令
# 用户立刻去连会被拒，体验差。
for _ in $(seq 1 50); do
    if ! is_port_free "${PORT}"; then
        # 端口被占 = TB 已经 bind = ready
        break
    fi
    sleep 0.1
done

cat <<EOF

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
[tb] TensorBoard 已启动 → 在本地浏览器直接打开：

      http://localhost:${PORT}

注：
  - 训练 + eval 同时跑时，TB 左侧 run 列表会显示 tb / eval_tb 两个子目录。
  - Ctrl-C 关掉本脚本时 TensorBoard 也会一起退出。
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

EOF

wait "${TB_PID}"
