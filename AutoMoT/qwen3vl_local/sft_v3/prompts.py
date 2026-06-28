"""SFT v3 prompt compatibility layer.

SFT v3 and SFT v4 intentionally share one prompt/state-machine contract.  Keep
the canonical implementation in ``qwen3vl_local.sft_v4.prompts`` and import it
here, so a prompt edit cannot silently diverge between the offline OPSD trainer
(v3) and the off-policy actor-learner (v4).
"""

from __future__ import annotations

from qwen3vl_local.sft_v4.prompts import *  # noqa: F401,F403
from qwen3vl_local.sft_v4 import prompts as _v4_prompts

DATASET_VERSION = "sft_v3_sequence_opsd"
SYSTEM_PROMPT_V3 = _v4_prompts.SYSTEM_PROMPT_V4

# Historical v3 names kept for launchers/checkpoint metadata.
DEFAULT_W_ANALYSIS = _v4_prompts.DEFAULT_W_ANALYSIS
DEFAULT_W_ROAD_STRUCTURE = _v4_prompts.DEFAULT_W_ROAD_STRUCTURE
DEFAULT_W_SCENE = _v4_prompts.DEFAULT_W_SCENE
DEFAULT_W_STATUS = _v4_prompts.DEFAULT_W_STATUS
DEFAULT_W_SUBGOAL = _v4_prompts.DEFAULT_W_SUBGOAL
