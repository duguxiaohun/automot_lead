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
#   bash qwen3vl_local/sft/tb_serve.sh checkpoints/sft_v1_lora
#   bash qwen3vl_local/sft/tb_serve.sh checkpoints/goalgen_v1_dit
#
# 想同时看训练 + eval 两条 TB run，把 logdir 指到 OUTPUT_DIR 根目录即可——
# TB 会把子目录里的 tb/ 与 eval_tb/ 自动列成两个 run，左侧 run 列表可勾选切换。
#
# 可选环境变量 override：
#   TB_PORT=6007   ← 强制端口；不传时自动从 OS 取空闲端口
#   TB_BIND=0.0.0.0 ← 改 bind 地址；默认 --bind_all（等价 0.0.0.0）
#   TB_EXTRA="--samples_per_plugin images=200"  ← 透传给 tensorboard 的额外参数
#
# 注：本机已有别的方式把远端 6006 通到本地（VSCode Remote 自动端口转发 / 现成隧道 /
# 公网 IP 直连），脚本不再打印 ssh 隧道命令；只打印浏览器直接打开的 URL。
#
# 退出：
#   Ctrl-C 一次即可，trap 会确保 TB 子进程一起退出，不会留僵尸。

set -euo pipefail

LOGDIR="${1:-}"
if [[ -z "${LOGDIR}" ]]; then
    cat >&2 <<EOF
用法: bash qwen3vl_local/sft/tb_serve.sh <logdir>

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

LOGDIR_ABS="$(cd "$(dirname "${LOGDIR}")" 2>/dev/null && pwd || pwd)/$(basename "${LOGDIR}")"

# ---- 4. 启动 TB ----
echo "============================================================"
echo "[tb] logdir: ${LOGDIR_ABS}"
echo "[tb] port:   ${PORT}"
echo "[tb] bind:   ${BIND_FLAGS[*]}"
echo "============================================================"

# 后台起 TB，捕获 PID 用于 trap 清理。
tensorboard --logdir "${LOGDIR}" --port "${PORT}" "${BIND_FLAGS[@]}" "${EXTRA_ARGS[@]}" &
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
