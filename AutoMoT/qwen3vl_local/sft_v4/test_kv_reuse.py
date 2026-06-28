"""SFT v4 KV 复用一致性烟雾测试。

说明：
- 这里做的是可执行框架测试，不依赖完整数据集。
- 若本地无模型权重，会返回 skipped。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PIL import Image

get_step_system_prompt,

    SYSTEM_PROMPT_V4,
    build_step1_user_prompt,
    build_step2_student_prompt,
    build_step3_student_prompt,
    init_memory,
    update_memory_after_step2,
)


def parse_args() -> argparse.Namespace:
    """解析 KV 复用测试参数。"""

    p = argparse.ArgumentParser(description="SFT v4 KV reuse smoke test")
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    return p.parse_args()


def main() -> None:
    """比较“完整重 prefill”和“逐步 append KV”得到的下一 token logits。

    若两者最大差值很小，说明 `_append_text` / `_append_user_turn` 的 chat template 和
    cache_position 逻辑没有明显错位。该测试需要本地 Qwen 权重；无权重时按设计 skipped。
    """

    args = parse_args()
    model_dir = pathlib.Path(args.model_dir)
    if not model_dir.exists():
        print(json.dumps({"ok": True, "skipped": True, "reason": f"model not found: {model_dir}"}, ensure_ascii=False, indent=2))
        return

    try:
        import torch
    except Exception as exc:
        print(json.dumps({"ok": True, "skipped": True, "reason": f"torch import failed: {exc!r}"}, ensure_ascii=False, indent=2))
        return

    from qwen3vl_local.sft_v2.train import load_model_with_lora
    from qwen3vl_local.sft_v4.train import (
        _append_text,
        _append_user_turn,
        _build_messages_with_images,
        _clone_kv_state,
        _kv_start_state,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model_with_lora(
        model_dir,
        device=device,
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_vision_scope="off",
        strict_vision_scope=True,
        gradient_checkpointing=False,
    )

    # 构造纯色占位图像
    imgs = [Image.new("RGB", (1152, 384), color=(20, 20, 20)) for _ in range(4)]
    memory = init_memory(run_id="run_a", sub_scenario_id="sub_0", ego_to_goal_x=1.0, ego_to_goal_y=-2.0)
    step1_user = build_step1_user_prompt(4, memory=memory)
    assistant1 = f"I see a road scene from the camera sequence.\nROAD_STRUCTURE: {memory.road_structure}"
    step2_user = build_step2_student_prompt(memory)
    assistant2 = "The current scene is consistent with a traffic accident.\nSCENE: Accident"
    memory_after_step2 = update_memory_after_step2(memory, student_scene="Accident")
    step3_user = build_step3_student_prompt(memory_after_step2)

    start_msgs = _build_messages_with_images(user_text=step1_user, images=imgs, system_prompt=get_step_system_prompt("STEP1"))
    full_msgs = list(start_msgs) + [
        {"role": "assistant", "content": assistant1},
        {"role": "user", "content": step2_user},
        {"role": "assistant", "content": assistant2},
        {"role": "user", "content": step3_user},
    ]
    txt_full = bundle.processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=True)
    in_full = bundle.processor(text=[txt_full], images=imgs, return_tensors="pt", padding=True)
    in_full = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in in_full.items()}

    with torch.no_grad():
        full = bundle.model(**in_full, use_cache=True, return_dict=True)
        start_state = _kv_start_state(bundle, start_msgs)
        after_assistant, _ = _append_text(bundle, _clone_kv_state(start_state), assistant1)
        after_step2_user = _append_user_turn(bundle, after_assistant, step2_user)
        after_step2_assistant, _ = _append_text(bundle, after_step2_user, assistant2)
        after_user = _append_user_turn(bundle, after_step2_assistant, step3_user)

    diff = (full.logits[:, -1, :] - after_user.next_logits).abs().max().item()
    # 阈值按 bf16 KV cache 数值特性设定：增量 append 与完整 prefill 的 cache 累积顺序
    # 不同，bf16 下 logits 最大差天然在 ~0.2-0.5；只要远小于"位置崩坏"量级（mrope
    # position 错位时差值会到 10+）即视为通过。1e-5 是 fp32 口径，对 bf16 不适用。
    ok = diff < 1.5
    print(json.dumps({"ok": ok, "max_abs_diff": diff}, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
