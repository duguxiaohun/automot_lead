"""SFT v3 的 prompt 兼容层。

v3 和 v4 必须共享同一份 prompt / Memory / 状态机 / target span 契约。真实实现只允许
放在 ``qwen3vl_local.sft_v4.prompts``；本文件只做 re-export。这样改 v4 prompt
时，v3 的 offline OPSD trainer 会自动吃到同一套文本协议，避免出现“两个版本提示词
看起来相似但状态机已经分叉”的隐性 bug。
"""

from __future__ import annotations

# noqa 放行通配 re-export：这里的设计目标就是让 v3 暴露与 v4 完全相同的 prompt API。
from qwen3vl_local.sft_v4.prompts import *  # noqa: F401,F403
from qwen3vl_local.sft_v4 import prompts as _v4_prompts

DATASET_VERSION = "sft_v3_sequence_opsd"
# 历史入口仍会读取 SYSTEM_PROMPT_V3。它只是 v4 总 prompt 的别名，新的 step 级训练
# 逻辑应优先调用 get_step_system_prompt("STEP1/2/3")，不要在这里写 v3 专属文本。
SYSTEM_PROMPT_V3 = _v4_prompts.SYSTEM_PROMPT_V4

# 历史 v3 名称保留给 launcher / checkpoint metadata；权重数值仍统一来自 v4 prompt 模块。
DEFAULT_W_ANALYSIS = _v4_prompts.DEFAULT_W_ANALYSIS
DEFAULT_W_ROAD_STRUCTURE = _v4_prompts.DEFAULT_W_ROAD_STRUCTURE
DEFAULT_W_SCENE = _v4_prompts.DEFAULT_W_SCENE
DEFAULT_W_STATUS = _v4_prompts.DEFAULT_W_STATUS
DEFAULT_W_SUBGOAL = _v4_prompts.DEFAULT_W_SUBGOAL
