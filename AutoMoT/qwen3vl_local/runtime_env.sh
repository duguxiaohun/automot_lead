#!/usr/bin/env bash
# Source with the AutoMoT application root. Does not install or change packages.
QWEN_RUNTIME_APP=$1
if [[ -z ${PYTHON:-} ]]; then
  if [[ -n ${VIRTUAL_ENV:-} && -x "$VIRTUAL_ENV/bin/python" ]]; then PYTHON="$VIRTUAL_ENV/bin/python"
  elif [[ -n ${CONDA_PREFIX:-} && -x "$CONDA_PREFIX/bin/python" ]]; then PYTHON="$CONDA_PREFIX/bin/python"
  elif [[ -x "$QWEN_RUNTIME_APP/.venv/bin/python" ]]; then PYTHON="$QWEN_RUNTIME_APP/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then PYTHON=$(command -v python)
  else PYTHON=$(command -v python3); fi
fi
PYTHON=$(command -v "$PYTHON")
if [[ "$PYTHON" != /* ]]; then PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")"; fi
export PYTHON PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset QWEN_RUNTIME_APP
