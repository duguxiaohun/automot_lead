#!/usr/bin/env bash
# Build/reuse the shared action index, train, then run test eval on best.pt.
set -euo pipefail
source qwen3vl_local/action_expert_ablation/pipeline_common.sh
action_ablation_run_full_pipeline bev_only checkpoints/action_expert_ablation/bev_only "$@"
