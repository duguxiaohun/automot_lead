#!/usr/bin/env bash
# No-argument entry: prepare once, then run full RGB4 and RGB2 experiments.
set -euo pipefail
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
P4_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$P4_DIR/../.."
[[ $# == 0 ]] || { echo 'Usage: bash qwen3vl_local/sft_new_loop_phase4/run.sh (no parameters needed)' >&2; exit 2; }
P4_PROBE='import torch, torchvision, transformers, peft, accelerate, PIL, numpy, huggingface_hub; assert transformers.__version__ == "5.3.0"'
P4_CANDIDATES=("${PYTHON:-}" "$(command -v python || true)" "$PWD/.venv/bin/python" "$PWD/../.venv/bin/python" /tmp/automot-qwen35-env/bin/python)
for P4_PYTHON in "${P4_CANDIDATES[@]}"; do
  [[ -n "$P4_PYTHON" ]] || continue
  if "$P4_PYTHON" -c "$P4_PROBE" >/dev/null 2>&1; then
    exec "$P4_PYTHON" -m qwen3vl_local.sft_new_loop_phase4.auto_run
  fi
  if [[ -n "${PYTHON:-}" ]]; then
    echo 'The explicitly selected Python lacks the Phase3/Qwen3.5 training dependencies.' >&2; exit 2
  fi
done
echo 'No compatible training environment found. Activate the environment used for Phase3 and rerun this same command.' >&2
exit 2
