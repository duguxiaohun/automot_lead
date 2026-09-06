"""按 adapter 的真实名称和哈希恢复已支持的新训练包提示词，禁止改标签冒充新版。"""

from qwen3vl_local.sft_new_loop_phase1 import prompts as phase1
from qwen3vl_local.sft_new_loop_phase2 import prompts as phase2
from qwen3vl_local.action_prior import phase2_v3_prompts


def prompt_module(phase, metadata):
    """只有经过源码恢复、哈希核对的协议可运行；未知历史版本拒绝。"""
    modules = (phase1,) if phase == 1 else (phase2, phase2_v3_prompts)
    for module in modules:
        if metadata.get("prompt_name") == module.PROMPT_NAME:
            hash_fn = module.phase1_prompt_sha256 if phase == 1 else module.event_prompt_sha256
            expected = hash_fn(history_rgb_mode=metadata["history_rgb_mode"])
            if metadata.get("production_prompt_sha256") != expected:
                raise ValueError(f"Phase{phase}: saved prompt hash differs from {module.PROMPT_NAME}")
            return module
    raise ValueError(f'Phase{phase}: unsupported new-package prompt {metadata.get("prompt_name")!r}')
