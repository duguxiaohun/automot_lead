"""验证本地化后的"自定义 KV 增量解码"与"原生 Qwen"是否一致。

回答两个问题（都需要本地 Qwen3-VL 权重；无权重时 skipped）：

A. **逐 token 漂移测试**：sft_v4/train 的 ``_append_token_ids`` 增量路径，每步算出的
   next-token logits，是否与"从头全量无 cache forward"的金标准在 bf16 噪声内一致。
   增量路径若把 mrope 位置算错（之前的 PEFT bug），logits 会差 10+、argmax 大面积翻车；
   修好后应当 0 翻车、maxabsdiff < 1.5。

B. **engine 自定义 KV vs 原生 generate**：本地 ``LocalQwen3VLInstructEngine`` 的手写
   decode 与 transformers ``model.generate`` 在同一 prompt 贪心下，输出 token 序列应高度
   一致（允许极少 bf16 argmax 抖动）。

运行：
    python qwen3vl_local/sft_v4/test_kv_vs_native.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

_THIS_FILE = pathlib.Path(__file__).resolve()
for _p in (str(_THIS_FILE.parents[2]), str(_THIS_FILE.parents[3])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_DIR = pathlib.Path("checkpoints/Qwen3-VL-4B-Instruct")
RGB_DIR = pathlib.Path(
    "lead_data/InterurbanAdvancedActorFlow/Town12_Rep0_1289_7_route0_01_07_23_36_37/rgb"
)
ANCHOR = 17
GT_SCENE = "InterurbanAdvancedActorFlow"
N_DRIFT_STEPS = 40
MAX_ABS_DIFF_TOL = 1.5  # bf16 cache 噪声量级；位置崩坏时会到 10+
GEN_TOKENS_B = 64


def _skip(reason: str) -> None:
    print(json.dumps({"ok": True, "skipped": True, "reason": reason}, ensure_ascii=False, indent=2))


def main() -> None:
    if not MODEL_DIR.exists():
        _skip(f"model not found: {MODEL_DIR}")
        return
    if not RGB_DIR.exists():
        _skip(f"rgb dir not found: {RGB_DIR}")
        return

    from qwen3vl_local.sft_v2.eval import _maybe_set_idle_gpu_mask

    _maybe_set_idle_gpu_mask()

    import torch

    from qwen3vl_local.sft_v2.train import load_model_with_lora
    from qwen3vl_local.sft_v4.prompts import (
        Memory,
        build_step1_teacher_prompt,
        get_road_structure,
    )
    from qwen3vl_local.sft_v4.train import (
        _append_token_ids,
        _build_messages_with_images,
        _clone_kv_state,
        _collect_images_from_messages,
        _kv_generate_text,
        _load_images,
        _teacher_start_state,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model_with_lora(
        MODEL_DIR, device=device, lora_rank=8, lora_alpha=16, lora_dropout=0.0,
        lora_vision_scope="off", strict_vision_scope=False, gradient_checkpointing=False,
    )
    bundle.device = device
    model = bundle.unwrap()
    model.eval()

    rope_owner = next(m for m in model.modules() if type(m).__name__ == "Qwen3VLModel")

    paths = list(reversed([str(RGB_DIR / f"{max(ANCHOR - i, 0):04d}.jpg") for i in range(4)]))
    images = _load_images(paths)
    gt_rs = get_road_structure(GT_SCENE)
    mem = Memory(road_structure=gt_rs, scene=GT_SCENE, status="initial",
                 subgoal="flow_approach", ego_to_goal_x=78.6, ego_to_goal_y=-86.4)
    messages = _build_messages_with_images(
        user_text=build_step1_teacher_prompt(mem, gt_rs), images=images
    )

    text = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_inputs = bundle.processor(text=[text], images=_collect_images_from_messages(messages),
                                   return_tensors="pt", padding=True)
    full_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in full_inputs.items()}
    vis_kwargs = {k: v for k, v in full_inputs.items() if k not in ("input_ids", "attention_mask")}

    results = {}

    # ---------- A. 逐 token 漂移测试 ----------
    with model.disable_adapter(), torch.no_grad():
        state = _teacher_start_state(bundle, messages)
        cur = state
        gen_ids = []
        mismatches = 0
        max_diff = 0.0
        for _step in range(N_DRIFT_STEPS):
            inc_logits = cur.next_logits
            inc_tok = int(torch.argmax(inc_logits, dim=-1).reshape(-1)[0].item())

            # 金标准：全量无 cache 重前向。这次重前向自身也是一次 prefill，会覆盖
            # rope_owner.rope_deltas，必须存后恢复，避免污染增量路径。
            saved = rope_owner.rope_deltas
            seq = torch.cat(
                [full_inputs["input_ids"],
                 torch.tensor([gen_ids], device=device, dtype=full_inputs["input_ids"].dtype)],
                dim=1,
            ) if gen_ids else full_inputs["input_ids"]
            ref_out = model(input_ids=seq, attention_mask=torch.ones_like(seq),
                            **vis_kwargs, use_cache=False, return_dict=True)
            ref_logits = ref_out.logits[:, -1, :]
            rope_owner.rope_deltas = saved
            ref_tok = int(torch.argmax(ref_logits, dim=-1).reshape(-1)[0].item())

            if inc_tok != ref_tok:
                mismatches += 1
            max_diff = max(max_diff, float((inc_logits - ref_logits).abs().max().item()))

            gen_ids.append(ref_tok)
            cur, _ = _append_token_ids(bundle, cur, torch.tensor([[ref_tok]], device=device))

        results["drift"] = {
            "steps": N_DRIFT_STEPS,
            "argmax_mismatches": mismatches,
            "max_abs_logit_diff": round(max_diff, 4),
            "ok": mismatches == 0 and max_diff < MAX_ABS_DIFF_TOL,
        }

    # ---------- B. engine 自定义 KV vs 原生 generate ----------
    with model.disable_adapter(), torch.no_grad():
        # 自定义 KV（关 ngram / rep_penalty，纯贪心，与原生 greedy 对齐）
        kv_state = _teacher_start_state(bundle, messages)
        kv_text, _ = _kv_generate_text(
            bundle, _clone_kv_state(kv_state), GEN_TOKENS_B,
            repetition_penalty=1.0, no_repeat_ngram_size=0,
        )
        kv_ids = bundle.tokenizer(kv_text, add_special_tokens=False)["input_ids"]

        # 原生 generate（同一 prompt，贪心）
        gen = model.generate(**full_inputs, max_new_tokens=GEN_TOKENS_B, do_sample=False)
        native_ids = gen[0, full_inputs["input_ids"].shape[1]:].tolist()
        native_text = bundle.processor.batch_decode(
            gen[:, full_inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()

    common = 0
    for a, b in zip(kv_ids, native_ids):
        if a == b:
            common += 1
        else:
            break
    denom = max(1, min(len(kv_ids), len(native_ids)))
    results["engine_vs_native"] = {
        "kv_tokens": len(kv_ids),
        "native_tokens": len(native_ids),
        "common_prefix_tokens": common,
        "common_prefix_ratio": round(common / denom, 4),
        "kv_text_head": kv_text[:160],
        "native_text_head": native_text[:160],
        "ok": (common / denom) >= 0.9,
    }

    overall_ok = results["drift"]["ok"] and results["engine_vs_native"]["ok"]
    print(json.dumps({"ok": overall_ok, **results}, ensure_ascii=False, indent=2))
    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
