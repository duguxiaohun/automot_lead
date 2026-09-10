#!/usr/bin/env bash
# EVENT_BALANCED 是 train.sh / run_full_pipeline.sh 的轨迹训练开关；本入口不启用均衡训练。
# 构建、开启/关闭以及独立场景先验 demo 见 run.md「UE/特殊 RE 均衡课程」。
ulimit -S -c 0 2>/dev/null || true
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 对比已有指标，默认复制并压缩推荐 LoRA；不申请 GPU、不改源权重。
# --no-export-bundle 仅审计；默认打印 tar.gz 路径、SHA256 和固定组合训练命令。
exec python "$HERE/rank_loras.py" "$@"
