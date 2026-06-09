#!/usr/bin/env bash
# SFT v2 训练入口 — ms-swift LoRA on Qwen3-VL-4B-Instruct，ANALYSIS 段带蒸馏监督。
#
# 与 sft_v1_train.sh 的核心区别：
#   - --loss_scale 改用 sft_v2_analysis_supervised（v2 plugin，ANALYSIS body 权重 0.3）；
#   - 默认读 sft_v2_data_pending/；首次启动若 runtime cache 不存在，自动调用
#     build_sft_dataset_v2_teacher.py 全量物化 teacher ANALYSIS 到
#     OUTPUT_DIR_BASE/runtime_teacher_data/（如 checkpoints/sft_v2_lora/runtime_teacher_data/），
#     再进 swift sft；之后再启动（任意卡数）会
#     自动检测到完整 cache 直接复用、跳过物化。改 prompt / keyframes 后想强制重跑
#     teacher，显式 RUNTIME_TEACHER_REFRESH=1 或直接 rm -rf runtime_teacher_data/；
#   - LR 5e-5 → 3e-5（v2 监督 token 数 × 5，lr 同步下调避免过冲，详见 SFT_PLAN.md §7）；
#   - MAX_LENGTH 3072 → 3584（v2 ANALYSIS 段更长）。
#
# GPU / MASTER_PORT 自动选址逻辑与 v1 完全相同；GPU 始终自动挑空闲卡并覆盖旧 mask。
#
# 用法（**从 AutoMoT/ 目录运行**）：
#   单卡：       bash qwen3vl_local/sft/sft_v2_train.sh single
#   DDP：        bash qwen3vl_local/sft/sft_v2_train.sh ddp
#   sanity 自检：bash qwen3vl_local/sft/sft_v2_train.sh check
#     （check 模式只跑 2 step、不保存 ckpt，用来确认 loss_scale 是否生效。
#      v2 初始 loss 应在 3-8 区间，比 v1 偏高，因为多了 ANALYSIS 段约 30 个 token
#      参与 loss；判读细节见 SFT_RUN.md §3。）
#
# 数据先用 qwen3vl_local/sft/build_sft_dataset_v1.py --mode v2 生成 pending jsonl。
# 训练脚本首次启动会调用冻结 teacher 一次性物化 ANALYSIS 真值到 runtime 目录，之后
# 任意卡数启动都自动复用同一份 cache，不重复物化。pending 源数据不会被回写。
#
# 常用 override：
#   MODEL_DIR=/path/to/Qwen3-VL-4B-Instruct \
#   TRAIN_JSONL=/path/to/v2_train.jsonl \
#   VAL_JSONL=/path/to/v2_val.jsonl \
#   OUTPUT_DIR=/path/to/sft_v2_lora \
#   DDP_GPU_COUNT=4 \
#   bash qwen3vl_local/sft/sft_v2_train.sh ddp
#
# 调权重（不重启训练前 export 即可生效，详见 qwen3vl_local/sft/sft_v2_loss_scale_plugin.py docstring）：
#   SFT_V2_ANALYSIS_WEIGHT=0.5 bash qwen3vl_local/sft/sft_v2_train.sh ddp     # ANALYSIS 还在漂移时
#   SFT_V2_ANALYSIS_WEIGHT=0.1 bash qwen3vl_local/sft/sft_v2_train.sh ddp     # ANALYSIS 过拟合 teacher 时

set -euo pipefail

MODE="${1:-ddp}"
DDP_GPU_COUNT_WAS_SET=0
if [[ -n "${DDP_GPU_COUNT+x}" ]]; then
    DDP_GPU_COUNT_WAS_SET=1
fi

# ---------------------------------------------------------------------------
# 路径默认值（v2 专用）
# ---------------------------------------------------------------------------
MODEL_DIR="${MODEL_DIR:-checkpoints/Qwen3-VL-4B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-checkpoints/sft_v2_data_pending/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-checkpoints/sft_v2_data_pending/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft_v2_lora}"

# 防覆盖：每次启动自动建 run_<时间戳> 子目录，LoRA adapter / checkpoint-XXX / tb
# 产物全部写进子目录，顶层 OUTPUT_DIR_BASE/ 维护 latest symlink 指向当前 run。
# - RUN_TAG=xxx：用 run_xxx/ 做子目录名；不设用 run_$(date +%Y%m%d_%H%M%S)/；
# - NO_RUN_SUBDIR=1：回退老的"顶层覆盖"行为。
# 关键：runtime_teacher_data/ 与 HF_HOME 都钉在 OUTPUT_DIR_BASE 层（不进 run 子目录），
# 否则每个 run 都会重新物化 teacher（冻结 base Qwen 跑全量生成，极贵）、manifest
# 跨启动复用机制也失效。所以 OUTPUT_DIR_BASE 必须在下方 teacher dir 之前就算好。
OUTPUT_DIR_BASE="${OUTPUT_DIR}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/run_${RUN_TAG}"
fi
# check 模式默认写到独立子目录，避免误清掉正式训练用的 runtime_teacher_data/。
RUNTIME_TEACHER_DIR_WAS_SET=0
if [[ -n "${RUNTIME_TEACHER_DIR+x}" ]]; then
    RUNTIME_TEACHER_DIR_WAS_SET=1
fi
RUNTIME_TEACHER_DIR="${RUNTIME_TEACHER_DIR:-${OUTPUT_DIR_BASE}/runtime_teacher_data}"
if [[ "${MODE}" == "check" && "${RUNTIME_TEACHER_DIR_WAS_SET}" != "1" ]]; then
    RUNTIME_TEACHER_DIR="${OUTPUT_DIR_BASE}/runtime_teacher_check_data"
fi
RUNTIME_TEACHER_SEED="${RUNTIME_TEACHER_SEED:-20260601}"
# single/ddp 默认 0 = 已有完整 runtime cache 时直接复用、跳过物化；显式 1 = 清掉旧
# cache 强制重跑（prompt/keyframes 改过、想丢弃旧 ANALYSIS 时用）。
# check 模式例外：若用户没显式设 REFRESH，默认置 1，确保每次 sanity 看到最新 teacher。
RUNTIME_TEACHER_REFRESH_WAS_SET=0
if [[ -n "${RUNTIME_TEACHER_REFRESH+x}" ]]; then
    RUNTIME_TEACHER_REFRESH_WAS_SET=1
fi
RUNTIME_TEACHER_REFRESH="${RUNTIME_TEACHER_REFRESH:-0}"
if [[ "${MODE}" == "check" && "${RUNTIME_TEACHER_REFRESH_WAS_SET}" != "1" ]]; then
    RUNTIME_TEACHER_REFRESH="1"
fi

# ---------------------------------------------------------------------------
# 超参（v2 调整版，详见 SFT_PLAN.md §7）
# ---------------------------------------------------------------------------
# v2 监督信号从 v1 的 ~6 token 升到 ~30 token（多了 ANALYSIS body），
# 等效"有效梯度量"提高约 5 倍 → lr 同步下调一档（5e-5 → 3e-5）防过冲。
# MAX_LENGTH 从 3072 抬到 3584：teacher ANALYSIS body 约 80-120 token，
# 加上 system + user + 4 张图视觉 token，预留余量避免触发 truncation warning。
# 其它超参（LORA_RANK / dropout / weight_decay / warmup）沿用 v1。
# 2026-06 三轮调优后等效 global batch 从 32 升到 192（6x），cosine schedule 总步数
# 同比缩到 1/6（v2 ddp 约 52k → 8.7k）；NUM_EPOCHS 从 2 升到 4 弥补 step 损失，让
# ANALYSIS 蒸馏有足够更新次数收敛。想精确控总步数显式 NUM_EPOCHS=N。
NUM_EPOCHS="${NUM_EPOCHS:-4}"
# 2026-06 显存优化（两轮）：
#   ① 第一轮：v2 max_seq=3584 + frozen teacher 同驻显存比 v1 紧，PER_DEVICE_BS ×2
#      GRAD_ACC 保持 2，等效 batch 翻 2x，LR 按 sqrt(2)=1.41x 上调 3e-5 → 4.2e-5。
#   ② 第二轮：再翻 per_device 2x，GRAD_ACC 不变，等效 batch 再翻 2x
#      （single 8→16，ddp 8卡 4→8），LR 再 sqrt(2)=1.41x → 5.9e-5。
#   ③ 第三轮（上一版 H20 默认靠近 70-80% 显存）：ddp per_device 8→12，
#      single 16→24，等效 batch 再乘 1.5，LR 按 sqrt(1.5) → 7.2e-5。
#   ④ 第四轮（曾按 H20 80% 附近微调）：ddp per_device 12→15，8 卡等效 batch 192→240。
#   ⑤ 第五轮（修复 full-logits fp32 loss 峰值 OOM）：显式启用 use_logits_to_keep；
#      同时把 micro-batch 拆小、用 GRAD_ACC 补回等效 batch，避免 logits.float() 首步额外吃掉数十 GB。
# check 模式 PER_DEVICE_BS=1 / GRAD_ACC=1 不动以保留快速 sanity。
LR="${LR:-8.0e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-3584}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
USE_LOGITS_TO_KEEP="${USE_LOGITS_TO_KEEP:-true}"

# ---------------------------------------------------------------------------
# Step-based checkpoint 保存（与 v1 同口径）
# ---------------------------------------------------------------------------
# HF Trainer 不允许 epoch + steps 同时 save；load_best_model_at_end 要求
# save_strategy == eval_strategy。所以 v1/v2 都改成纯 step 保存：每 SAVE_STEPS
# 步保存 + 评估，--save_total_limit 控制保留最近 N 个 checkpoint-XXX/。
# 默认 SAVE_STEPS=10000 / SAVE_TOTAL_LIMIT=3 → 等效"保留最近 30k 步"。
# 想换 5k / 20k / 改保留数，export 同名变量即可。
SAVE_STEPS="${SAVE_STEPS:-10000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

# v2 plugin：ANALYSIS body 权重 0.3、STATUS/SUBGOAL event_name 1.0、其它 0。
LOSS_SCALE="sft_v2_analysis_supervised"
LOSS_SCALE_PLUGIN="qwen3vl_local/sft/sft_v2_loss_scale_plugin.py"

# HuggingFace 强制离线（与 v1 同）。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 缓存指向 base 层，避免误读 ~/.cache，也让所有 run 共享 tokenizer/model cache。
export HF_HOME="${HF_HOME:-${OUTPUT_DIR_BASE}/.hf_cache}"
mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"
if [[ "${NO_RUN_SUBDIR:-0}" != "1" ]]; then
    # ln -sfn：force + no-dereference，原子替换旧 symlink；相对目标，base 搬走仍有效。
    ln -sfn "run_${RUN_TAG}" "${OUTPUT_DIR_BASE}/latest"
    echo "[run] OUTPUT_DIR=${OUTPUT_DIR}  (latest -> run_${RUN_TAG})"
fi

# ---------------------------------------------------------------------------
# GPU / MASTER_PORT 选址（与 v1 sft_v1_train.sh 完全相同的函数，复制保持解耦）
# ---------------------------------------------------------------------------
pick_idle_gpus() {
    local want_count="$1"
    local selected

    if command -v nvidia-smi >/dev/null 2>&1; then
        selected="$(
            nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
                | awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); print $2, $3, $1}' \
                | sort -n -k1,1 -k2,2 \
                | head -n "${want_count}" \
                | awk '{print $3}' \
                | paste -sd, -
        )"
        if [[ -n "${selected}" ]]; then
            echo "${selected}"
            return 0
        fi
    fi

    if [[ "${want_count}" -le 1 ]]; then
        echo "0"
    else
        seq -s, 0 "$((want_count - 1))"
    fi
}

count_visible_gpus() {
    local visible="$1"
    if [[ -z "${visible}" ]]; then
        echo "0"
    else
        awk -F',' '{print NF}' <<< "${visible}"
    fi
}

is_port_free() {
    local port="$1"
    python -c 'import socket, sys
port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
' "${port}" >/dev/null 2>&1
}

find_free_master_port() {
    python -c 'import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("", 0))
print(sock.getsockname()[1])
sock.close()
' 2>/dev/null || echo "$((20000 + RANDOM % 20000))"
}

configure_master_port() {
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

    if [[ "${SFT_RESPECT_MASTER_PORT:-0}" == "1" && -n "${MASTER_PORT:-}" ]]; then
        if is_port_free "${MASTER_PORT}"; then
            export MASTER_PORT
            return 0
        fi
        echo "[port][err] MASTER_PORT=${MASTER_PORT} is already in use and SFT_RESPECT_MASTER_PORT=1" >&2
        exit 1
    fi

    if [[ -n "${MASTER_PORT:-}" ]]; then
        if is_port_free "${MASTER_PORT}"; then
            export MASTER_PORT
            return 0
        fi
        echo "[port][warn] MASTER_PORT=${MASTER_PORT} is already in use; selecting a free port"
    fi

    export MASTER_PORT="$(find_free_master_port)"
}

export_torchrun_master_env() {
    export PET_MASTER_ADDR="${MASTER_ADDR}"
    export PET_MASTER_PORT="${MASTER_PORT}"
}

jsonl_dataset_version() {
    local path="$1"
    python -c 'import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            print(json.loads(line).get("dataset_version", "v1"))
            break
' "${path}"
}

# 检测 runtime cache 是否完整可复用。GPU 数无关。
# 强校验：必须有 manifest.json，且 manifest 记录的 pending/runtime 行数与当前实际
# 文件行数严格匹配。manifest 由 build_sft_dataset_v2_teacher.py 在"全集跑完
# (max_samples==0) + train+val 都跑了 + runtime 行数等于 pending 行数"时才写入，
# 所以 32 条 debug cache / 半截 val / skip-train 部分跑都不会通过校验。
runtime_teacher_pair_is_ready() {
    local train_path="${RUNTIME_TEACHER_DIR}/train.jsonl"
    local val_path="${RUNTIME_TEACHER_DIR}/val.jsonl"
    local manifest_path="${RUNTIME_TEACHER_DIR}/manifest.json"

    [[ -s "${train_path}" && -s "${val_path}" && -s "${manifest_path}" ]] || return 1
    [[ "$(jsonl_dataset_version "${train_path}")" == "v2" ]] || return 1
    [[ "$(jsonl_dataset_version "${val_path}")" == "v2" ]] || return 1

    # python 一次性把所有校验做完，避免反复 shell-out。
    # 校验项：
    #   - manifest.max_samples == 0（全集跑）
    #   - manifest.model_dir   == 当前 MODEL_DIR（teacher 模型必须一致）
    #   - manifest.seed        == 当前 RUNTIME_TEACHER_SEED（greedy 时不影响输出但记录用）
    #   - manifest.max_new_tokens == 256（sft_v2_train.sh 始终用 teacher 脚本默认值；
    #     有人手动调用 teacher 脚本改了这俩参数后，下次 sft_v2_train.sh 必须拒绝复用）
    #   - manifest.teacher_temperature == 0.0
    #   - manifest.runtime_train_rows == actual lines in runtime/train.jsonl
    #   - manifest.runtime_val_rows   == actual lines in runtime/val.jsonl
    #   - manifest.pending_train_rows == actual lines in pending/train.jsonl
    #   - manifest.pending_val_rows   == actual lines in pending/val.jsonl
    # 任一失败 → return 1。
    python - "$manifest_path" "$train_path" "$val_path" \
        "$(dirname "${TRAIN_JSONL}")/train.jsonl" \
        "$(dirname "${TRAIN_JSONL}")/val.jsonl" \
        "${MODEL_DIR}" "${RUNTIME_TEACHER_SEED}" <<'PY' || return 1
import json, os, sys, pathlib
mf_path, rt_train, rt_val, pd_train, pd_val = (pathlib.Path(p) for p in sys.argv[1:6])
cur_model_dir = sys.argv[6]
cur_seed = sys.argv[7]

# sft_v2_train.sh 的 teacher 调用始终不传 --max-new-tokens / --teacher-temperature，
# 用的就是 build_sft_dataset_v2_teacher.py 的 argparse 默认值。这里把期望值
# 硬编码与 teacher 脚本默认一致；如果以后 sft_v2_train.sh 想支持调它们，这两
# 行要改成从 env 读 + teacher 脚本 default 也同步。
EXPECTED_MAX_NEW_TOKENS = 256
EXPECTED_TEACHER_TEMPERATURE = 0.0

try:
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
except Exception as e:
    print(f"[reuse-check] manifest parse fail: {e}")
    sys.exit(1)
if int(mf.get("max_samples", -1)) != 0:
    print(f"[reuse-check] reject: manifest.max_samples={mf.get('max_samples')} (need 0)")
    sys.exit(1)
# model_dir 用 realpath 归一化对比，避免相对/绝对路径差异误判。
try:
    mf_model = os.path.realpath(str(mf.get("model_dir", "")))
    cur_model = os.path.realpath(cur_model_dir)
except Exception:
    mf_model, cur_model = str(mf.get("model_dir", "")), cur_model_dir
if mf_model != cur_model:
    print(f"[reuse-check] reject: model_dir manifest={mf.get('model_dir')} (realpath={mf_model}) current={cur_model_dir} (realpath={cur_model})")
    sys.exit(1)
if str(mf.get("seed", "")) != str(cur_seed):
    print(f"[reuse-check] reject: seed manifest={mf.get('seed')} current={cur_seed}")
    sys.exit(1)
# gen 参数校验：schema_version < 2 的旧 manifest 没有这俩字段，按缺失处理 → 拒绝。
if int(mf.get("max_new_tokens", -1)) != EXPECTED_MAX_NEW_TOKENS:
    print(f"[reuse-check] reject: max_new_tokens manifest={mf.get('max_new_tokens')} expected={EXPECTED_MAX_NEW_TOKENS}")
    sys.exit(1)
try:
    mf_temp = float(mf.get("teacher_temperature", -1))
except (TypeError, ValueError):
    mf_temp = -1.0
if mf_temp != EXPECTED_TEACHER_TEMPERATURE:
    print(f"[reuse-check] reject: teacher_temperature manifest={mf.get('teacher_temperature')} expected={EXPECTED_TEACHER_TEMPERATURE}")
    sys.exit(1)
def count(p):
    if not p.exists():
        return -1
    n = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n
rt_train_n, rt_val_n = count(rt_train), count(rt_val)
pd_train_n, pd_val_n = count(pd_train), count(pd_val)
checks = {
    "runtime_train_rows": (mf.get("runtime_train_rows"), rt_train_n),
    "runtime_val_rows":   (mf.get("runtime_val_rows"),   rt_val_n),
    "pending_train_rows": (mf.get("pending_train_rows"), pd_train_n),
    "pending_val_rows":   (mf.get("pending_val_rows"),   pd_val_n),
}
for key, (mf_val, actual) in checks.items():
    if mf_val != actual:
        print(f"[reuse-check] reject: {key} manifest={mf_val} actual={actual}")
        sys.exit(1)
sys.exit(0)
PY
}

materialize_runtime_teacher_if_needed() {
    local train_version
    train_version="$(jsonl_dataset_version "${TRAIN_JSONL}")"
    if [[ "${train_version}" != "v2_pending" ]]; then
        echo "[teacher] TRAIN_JSONL is ${train_version}; use as-is: ${TRAIN_JSONL}"
        return 0
    fi

    local pending_dir
    pending_dir="$(dirname "${TRAIN_JSONL}")"
    local pending_val
    pending_val="${pending_dir}/val.jsonl"
    if [[ "${VAL_JSONL}" != "${pending_val}" ]]; then
        echo "[teacher][warn] pending train/val should live in one dir; override VAL_JSONL=${pending_val}"
        VAL_JSONL="${pending_val}"
    fi

    # 进入物化路径前，先做"复用 / 清理 / 续跑"三种状态判定：
    # 1) cache 完整且校验通过 → reuse 秒进训练
    # 2) manifest 存在但校验失败（model_dir / seed / gen 参数 / 行数等不匹配）
    #    → stale config，rm 全部 (train+val+stats+manifest+.rank*) 重物化
    # 3) manifest 缺失但 final jsonl / .rank* 存在 → orphan cache，rm 全部
    #    防止旧配置分片被 fingerprint 去重误当成当前 teacher 产物
    # 4) 目录为空 → 直接物化
    local manifest_existed=0
    if [[ -f "${RUNTIME_TEACHER_DIR}/manifest.json" ]]; then
        manifest_existed=1
    fi

    # 复用分支：non-check 模式 + 没要求强制刷新 + cache 完整 → 秒进训练。
    # check 模式总是物化 32 条小样本（且默认走 runtime_teacher_check_data/ 独立目录）。
    if [[ "${MODE}" != "check" && "${RUNTIME_TEACHER_REFRESH}" != "1" ]] \
        && runtime_teacher_pair_is_ready; then
        TRAIN_JSONL="${RUNTIME_TEACHER_DIR}/train.jsonl"
        VAL_JSONL="${RUNTIME_TEACHER_DIR}/val.jsonl"
        echo "[teacher] reuse existing runtime teacher cache at ${RUNTIME_TEACHER_DIR}"
        echo "[teacher] cache is GPU-count agnostic; set RUNTIME_TEACHER_REFRESH=1 (or rm -rf the dir) to regenerate"
        echo "[teacher] runtime train=${TRAIN_JSONL}"
        echo "[teacher] runtime val=${VAL_JSONL}"
        return 0
    fi

    mkdir -p "${RUNTIME_TEACHER_DIR}"
    if [[ -z "${RUNTIME_TEACHER_DIR}" || "${RUNTIME_TEACHER_DIR}" == "/" || "${RUNTIME_TEACHER_DIR}" == "." ]]; then
        echo "[teacher][err] unsafe RUNTIME_TEACHER_DIR=${RUNTIME_TEACHER_DIR}" >&2
        exit 2
    fi

    if [[ "${RUNTIME_TEACHER_REFRESH}" == "1" ]]; then
        echo "[teacher] refresh runtime cache because RUNTIME_TEACHER_REFRESH=1"
        rm -f \
            "${RUNTIME_TEACHER_DIR}/train.jsonl" \
            "${RUNTIME_TEACHER_DIR}/val.jsonl" \
            "${RUNTIME_TEACHER_DIR}/stats.json" \
            "${RUNTIME_TEACHER_DIR}/manifest.json" \
            "${RUNTIME_TEACHER_DIR}"/train.jsonl.rank* \
            "${RUNTIME_TEACHER_DIR}"/val.jsonl.rank*
    elif [[ "${manifest_existed}" == "1" && "${MODE}" != "check" ]]; then
        # manifest 存在但 reuse_check 失败 → stale config（model_dir / seed / gen 参数
        # 或 行数不匹配）。这种情况 .rank* 分片也是用旧配置生成的，必须一起清掉，
        # 否则 teacher 脚本会把它们当 done 跳过，最终给 stale 数据补签新 manifest。
        echo "[teacher] manifest.json existed but failed reuse-check → stale config; wiping cache (incl. .rank*) to force fresh materialize"
        rm -f \
            "${RUNTIME_TEACHER_DIR}/train.jsonl" \
            "${RUNTIME_TEACHER_DIR}/val.jsonl" \
            "${RUNTIME_TEACHER_DIR}/stats.json" \
            "${RUNTIME_TEACHER_DIR}/manifest.json" \
            "${RUNTIME_TEACHER_DIR}"/train.jsonl.rank* \
            "${RUNTIME_TEACHER_DIR}"/val.jsonl.rank*
    elif [[ -s "${RUNTIME_TEACHER_DIR}/train.jsonl" ]] \
        || [[ -s "${RUNTIME_TEACHER_DIR}/val.jsonl" ]] \
        || compgen -G "${RUNTIME_TEACHER_DIR}/train.jsonl.rank*" > /dev/null \
        || compgen -G "${RUNTIME_TEACHER_DIR}/val.jsonl.rank*" > /dev/null; then
        # manifest 缺失 + final/rank 残留 = 旧版残留或未完成中间态。
        # rank 分片没有独立 manifest 锁定 model_dir / seed / gen 参数；如果保留，
        # teacher 脚本只按样本 fingerprint 去重，可能把旧配置分片当成当前产物。
        # 因此默认清掉全部不可验证缓存，牺牲一次中断续跑进度，换取 teacher GT 正确性。
        echo "[teacher] manifest.json missing but runtime residue exists → orphan/unverifiable cache; wiping cache (incl. .rank*) to force fresh materialize"
        rm -f \
            "${RUNTIME_TEACHER_DIR}/train.jsonl" \
            "${RUNTIME_TEACHER_DIR}/val.jsonl" \
            "${RUNTIME_TEACHER_DIR}/stats.json" \
            "${RUNTIME_TEACHER_DIR}/manifest.json" \
            "${RUNTIME_TEACHER_DIR}"/train.jsonl.rank* \
            "${RUNTIME_TEACHER_DIR}"/val.jsonl.rank*
    else
        echo "[teacher] no reusable runtime cache; build_sft_dataset_v2_teacher.py will materialize from scratch"
    fi
    echo "[teacher] runtime materialize teacher ANALYSIS (no reusable cache; auto triggered)"
    echo "[teacher] pending_dir=${pending_dir}"
    echo "[teacher] output_dir=${RUNTIME_TEACHER_DIR}"
    echo "[teacher] source pending jsonl is not modified"

    local teacher_args=(
        --pending-dir "${pending_dir}"
        --output-dir "${RUNTIME_TEACHER_DIR}"
        --model-dir "${MODEL_DIR}"
        --seed "${RUNTIME_TEACHER_SEED}"
    )
    if [[ "${MODE}" == "check" ]]; then
        teacher_args+=(--max-samples "${RUNTIME_TEACHER_MAX_SAMPLES:-32}")
    elif [[ -n "${RUNTIME_TEACHER_MAX_SAMPLES:-}" && "${RUNTIME_TEACHER_MAX_SAMPLES}" != "0" ]]; then
        teacher_args+=(--max-samples "${RUNTIME_TEACHER_MAX_SAMPLES}")
        # 警告：sample-limited 跑不会写 manifest，下次 sft_v2_train.sh 启动会拒绝复用、
        # 重新跑 teacher。debug 场景请改用 build_sft_dataset_v2_teacher.py 直接调用 +
        # --output-dir 指向独立的 debug 目录（详见 SFT_RUN.md §4）。
        echo "[teacher][warn] RUNTIME_TEACHER_MAX_SAMPLES=${RUNTIME_TEACHER_MAX_SAMPLES} in ${MODE} mode" >&2
        echo "[teacher][warn]   - this run will NOT write manifest.json" >&2
        echo "[teacher][warn]   - the resulting cache in ${RUNTIME_TEACHER_DIR} is NOT reusable by next launch" >&2
        echo "[teacher][warn]   - consider running build_sft_dataset_v2_teacher.py with --output-dir <debug-dir> instead" >&2
    fi

    if [[ "${MODE}" == "ddp" && "${NPROC_PER_NODE}" -gt 1 ]]; then
        torchrun --nproc_per_node="${NPROC_PER_NODE}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT}" \
            qwen3vl_local/sft/build_sft_dataset_v2_teacher.py \
            "${teacher_args[@]}"
    else
        python qwen3vl_local/sft/build_sft_dataset_v2_teacher.py "${teacher_args[@]}"
    fi

    TRAIN_JSONL="${RUNTIME_TEACHER_DIR}/train.jsonl"
    VAL_JSONL="${RUNTIME_TEACHER_DIR}/val.jsonl"
    echo "[teacher] runtime train=${TRAIN_JSONL}"
    echo "[teacher] runtime val=${VAL_JSONL}"
}

# ---------------------------------------------------------------------------
# 模式分支
# ---------------------------------------------------------------------------
case "${MODE}" in
    single)
        echo "[mode] single-GPU"
        export CUDA_VISIBLE_DEVICES="$(pick_idle_gpus 1)"
        export NPROC_PER_NODE=1
        # 等效 batch = PER_DEVICE_BS * GRAD_ACC = 44；micro-batch 降到 11，
        # 给 max_seq=3584 和 full-logits fp32 loss 峰值留余量。
        PER_DEVICE_BS="${PER_DEVICE_BS:-11}"
        GRAD_ACC="${GRAD_ACC:-4}"
        # step 触发：每 SAVE_STEPS 步保存 + eval；best 仍由 load_best_model_at_end 装回
        # OUTPUT_DIR 顶层 adapter_model.*。epoch 边界不再单独 save，但训练结束时
        # 最后一个 step ckpt ≈ 最后一个 epoch 末快照。
        SAVE_STRATEGY="steps"
        EVAL_STRATEGY="steps"
        REPORT_ARGS=(--report_to tensorboard --logging_dir "${OUTPUT_DIR}/tb")
        EXTRA_LAUNCH=""
        ;;
    check)
        echo "[mode] check (v2 loss_scale sanity, 2 steps only — no checkpoint, no eval)"
        # v2 健康初始 loss 区间：3-8。
        # 比 v1（6-10）偏低，因为 v2 ANALYSIS body 段加 0.3 权重后单 token loss 被稀释；
        # 但比 v1 mask=0 时多了 ~30 个 token 参与 loss，整体仍在可读量级。
        # 判读细节见 SFT_RUN.md §3。
        export CUDA_VISIBLE_DEVICES="$(pick_idle_gpus 1)"
        export NPROC_PER_NODE=1
        PER_DEVICE_BS=1
        GRAD_ACC=1
        SAVE_STRATEGY="no"
        EVAL_STRATEGY="no"
        # swift 在 report_to=tensorboard 时会在训练结束后调用 matplotlib 画 loss 图；
        # check 模式不需要 TB，关掉可避免环境里的 numpy/matplotlib 兼容性问题影响 sanity。
        REPORT_ARGS=(--report_to none)
        EXTRA_LAUNCH="--max_steps 2"
        ;;
    ddp)
        echo "[mode] DDP"
        # 默认等效 batch = 8 GPUs * PER_DEVICE_BS * GRAD_ACC = 240；micro-batch 降到 10，
        # 通过 GRAD_ACC=3 补回吞吐，规避 logits.float() 的首步峰值 OOM。
        # 显存仍不足时优先 PER_DEVICE_BS=5 / GRAD_ACC=6（等效 batch 不变，LR 通常不动）。
        PER_DEVICE_BS="${PER_DEVICE_BS:-10}"
        GRAD_ACC="${GRAD_ACC:-3}"
        # step 触发 + best 跟踪（含义见 single 分支注释）。
        SAVE_STRATEGY="steps"
        EVAL_STRATEGY="steps"
        DDP_GPU_COUNT="${DDP_GPU_COUNT:-8}"
        export CUDA_VISIBLE_DEVICES="$(pick_idle_gpus "${DDP_GPU_COUNT}")"
        ACTUAL_GPU_COUNT="$(count_visible_gpus "${CUDA_VISIBLE_DEVICES}")"
        export NPROC_PER_NODE="${ACTUAL_GPU_COUNT}"
        if [[ "${ACTUAL_GPU_COUNT}" -lt "${DDP_GPU_COUNT}" ]]; then
            echo "[gpu][warn] requested ${DDP_GPU_COUNT} GPUs but only selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        fi
        export NCCL_P2P_LEVEL=NVL
        export NCCL_DEBUG=WARN
        REPORT_ARGS=(--report_to tensorboard --logging_dir "${OUTPUT_DIR}/tb")
        EXTRA_LAUNCH=""
        ;;
    *)
        echo "Unknown mode: ${MODE}. Use 'single' / 'ddp' / 'check'." >&2
        exit 1
        ;;
esac
configure_master_port
export_torchrun_master_env

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[gpu] NPROC_PER_NODE=${NPROC_PER_NODE}"
if [[ "${MODE}" == "ddp" ]]; then
    echo "[gpu] requested DDP_GPU_COUNT=${DDP_GPU_COUNT:-8}"
fi
echo "[port] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} PET_MASTER_PORT=${PET_MASTER_PORT}"
echo "[mem] USE_LOGITS_TO_KEEP=${USE_LOGITS_TO_KEEP} PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

materialize_runtime_teacher_if_needed

if [[ "${MODE}" == "check" ]]; then
    VAL_ARGS=()
else
    VAL_ARGS=(--val_dataset "${VAL_JSONL}")
fi

# ANALYSIS 权重 override 提示，方便事后回溯。
echo "[plugin] SFT_V2_ANALYSIS_WEIGHT=${SFT_V2_ANALYSIS_WEIGHT:-0.3} (default 0.3)"

# ---------------------------------------------------------------------------
# best ckpt 跟踪（与 v1 一致）
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "check" ]]; then
    BEST_ARGS=()
else
    BEST_ARGS=(
        --load_best_model_at_end true
        --metric_for_best_model loss
        --greater_is_better false
    )
fi

# ---------------------------------------------------------------------------
# 启动训练
# ---------------------------------------------------------------------------
swift sft \
    --model "${MODEL_DIR}" \
    --dataset "${TRAIN_JSONL}" \
    "${VAL_ARGS[@]}" \
    --train_type lora \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --freeze_vit true \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --learning_rate "${LR}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --lr_scheduler_type cosine \
    --bf16 true \
    --gradient_checkpointing true \
    --use_logits_to_keep "${USE_LOGITS_TO_KEEP}" \
    --max_length "${MAX_LENGTH}" \
    --output_dir "${OUTPUT_DIR}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps "${SAVE_STEPS}" \
    --eval_strategy "${EVAL_STRATEGY}" \
    --eval_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --save_only_model true \
    "${REPORT_ARGS[@]}" \
    --dataloader_num_workers 4 \
    --external_plugins "${LOSS_SCALE_PLUGIN}" \
    --loss_scale "${LOSS_SCALE}" \
    "${BEST_ARGS[@]}" \
    ${EXTRA_LAUNCH}

echo "[done] v2 LoRA adapter saved under ${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "[hint] 看 TensorBoard："
echo "  bash qwen3vl_local/tb_serve.sh ${OUTPUT_DIR}"
echo ""
echo "[hint] 在 val 集上跑 eval（注意 v2 必须显式 --val-jsonl 指向 v2 数据）："
echo "  python qwen3vl_local/sft/eval_sft_v1.py --lora-dir ${OUTPUT_DIR} \\"
echo "      --val-jsonl ${VAL_JSONL} --save-root ${OUTPUT_DIR}"
echo ""
echo "[hint] 在随机场景上 dump case："
echo "  python qwen3vl_local/sft/probe_sft_v1.py --lora-dir ${OUTPUT_DIR} \\"
echo "      --val-jsonl ${VAL_JSONL} --save-root ${OUTPUT_DIR} --num-per-scenario 4"
echo "============================================================"
