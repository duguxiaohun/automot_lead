"""SFT v4 离线评测。

评测口径：
1. 只跑学生模型，不使用 teacher。
2. 不做 Phase B GT 强制注入，memory 全程由学生自更新。
3. 默认评测区间与训练一致：[f1-delta, f3]。
4. 输出关键指标：scene_acc、scene_recovery_steps、scene_stick_rate、scene_flip_rate、
   step3_trigger_rate、status/subgoal 条件准确率、all_acc。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PIL import Image
import torch

from qwen3vl_local.sft_v4.train import (
    EpisodeDataset,
    EpisodeRow,
    _analysis_before_labels,
    _build_messages_with_images,
    _build_rgb_paths,
    _gt_status_subgoal,
    _is_phase_a,
    _load_goal_xy,
    _load_images,
    _prefetch_goal_xy_for_next_frame,
)
from qwen3vl_local.sft_v4.prompts import (
    build_step1_user_prompt,
    build_step2_student_prompt,
    build_step2_teacher_prompt,
    build_step3_student_prompt,
    build_step3_teacher_prompt,
    init_memory,
    parse_output,
    should_trigger_step3,
    update_memory_after_step2,
    update_memory_after_step3,
    validate_event,
    validate_scene,
)
from qwen3vl_local.sft_v2.eval import _maybe_set_idle_gpu_mask

from qwen3vl_local.engine import LocalQwen3VLInstructEngine


def _simple_bleu(candidate: str, reference: str, max_n: int = 2) -> float:
    """无第三方依赖的轻量 BLEU，用于可选 teacher 分析对齐指标。

    这里只做诊断，不作为主指标；max_n 默认 2，能粗略反映学生分析是否跟 teacher
    hindsight 分析使用相似证据词，而不会引入 nltk/sacrebleu 依赖。
    """

    import math
    import re
    from collections import Counter

    cand = re.findall(r"\w+", (candidate or "").lower())
    ref = re.findall(r"\w+", (reference or "").lower())
    if not cand or not ref:
        return 0.0
    precisions: List[float] = []
    for n in range(1, max_n + 1):
        if len(cand) < n:
            precisions.append(0.0)
            continue
        cand_counts = Counter(tuple(cand[i : i + n]) for i in range(len(cand) - n + 1))
        ref_counts = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
        overlap = sum(min(count, ref_counts[gram]) for gram, count in cand_counts.items())
        precisions.append((overlap + 1.0) / (sum(cand_counts.values()) + 1.0))
    brevity = 1.0 if len(cand) > len(ref) else math.exp(1.0 - len(ref) / max(len(cand), 1))
    return float(brevity * math.exp(sum(math.log(max(p, 1e-8)) for p in precisions) / max_n))


def _generate(engine: LocalQwen3VLInstructEngine, messages: List[Dict[str, Any]], images: List[Image.Image], max_new_tokens: int) -> str:
    """生成 step1 文本，并在 engine._last_decode_state 中留下 KV cache。

    eval/probe 走 `LocalQwen3VLInstructEngine` 的真实自由生成路径；后续 step2/step3
    会从这个 cache 继续追加 user turn，模拟训练时的三步 OPD 对话。
    """

    system = messages[0]["content"]
    # 将 messages 扁平化到 user prompt（保持与 train 一致的 turn 结构）。
    prompt_lines: List[str] = []
    for m in messages[1:]:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            # 首个 user 带图 + 文本
            text_parts = [x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") == "text"]
            prompt_lines.append("\n".join(text_parts))
        else:
            prompt_lines.append(f"[{role}]\n{content}")
    user_prompt = "\n\n".join([x for x in prompt_lines if x])
    old_max = getattr(engine, "max_gen_tokens", None)
    if old_max is not None:
        engine.max_gen_tokens = int(max_new_tokens)
    try:
        txt, _ = engine.generate(system, user_prompt, images)
    finally:
        if old_max is not None:
            engine.max_gen_tokens = old_max
    return txt.strip()


def _render_user_suffix(engine: LocalQwen3VLInstructEngine, user_prompt: str) -> str:
    """把后续 user turn 渲染为 Qwen chat template 片段。"""

    suffix = engine.processor.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return suffix if suffix.startswith("\n") else "\n" + suffix


def _generate_next_with_kv(
    engine: LocalQwen3VLInstructEngine,
    user_prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    """从上一轮 decode 后的 KV cache 继续追加 user turn 并生成 assistant 文本。

    这个函数是 eval/probe 的 KV 复用核心：不重新喂图，只把 step2/step3 的文本
    user turn 接到已有 cache 后继续 decode。
    """

    state = getattr(engine, "_last_decode_state", None)
    if not state:
        raise RuntimeError("missing previous KV state; call step1 generate first")

    prefix_ids = state.get("cache_input_ids", state["decoded_input_ids"])
    prefix_len = int(prefix_ids.shape[1])
    decoded_ids = state["decoded_input_ids"].to(prefix_ids.device)
    pending_generated_ids = decoded_ids[:, prefix_len:]
    suffix_text = _render_user_suffix(engine, user_prompt)
    if pending_generated_ids.shape[1] == 0:
        suffix_text = "<|im_end|>" + suffix_text
    suffix_ids = engine.processor.tokenizer(
        suffix_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(prefix_ids.device)
    if pending_generated_ids.shape[1] > 0:
        suffix_ids = torch.cat([pending_generated_ids, suffix_ids], dim=1)

    old_attention = state["attention_mask"]
    if old_attention is None:
        old_attention = torch.ones_like(prefix_ids, device=prefix_ids.device)
    attention_mask = torch.cat(
        [old_attention, torch.ones_like(suffix_ids, device=old_attention.device)],
        dim=1,
    )
    decoded_input_ids = torch.cat([prefix_ids, suffix_ids], dim=1)
    cache_position = torch.arange(
        prefix_len,
        prefix_len + suffix_ids.shape[1],
        device=prefix_ids.device,
    )
    model_inputs = engine.model.prepare_inputs_for_generation(
        decoded_input_ids,
        past_key_values=state["past_key_values"],
        attention_mask=attention_mask,
        cache_position=cache_position,
        use_cache=True,
    )
    if state.get("rope_deltas") is not None and "rope_deltas" not in model_inputs:
        model_inputs["rope_deltas"] = state["rope_deltas"]
    outputs = engine.model(**model_inputs, return_dict=True)

    trace = type("_Trace", (), {
        "prefill_cache_summary": {},
        "final_cache_summary": {},
        "decode_steps": [],
        "prefill_cache_file": None,
        "final_cache_file": None,
        "system_prompt_cache_enabled": False,
        "system_prompt_cache_used": False,
        "system_prompt_cache_note": None,
        "system_prompt_cache_summary": {},
    })()
    old_max = getattr(engine, "max_gen_tokens", None)
    if old_max is not None:
        engine.max_gen_tokens = int(max_new_tokens)
    try:
        new_ids = engine.decode(
            {"input_ids": decoded_input_ids, "attention_mask": attention_mask},
            outputs,
            trace,
        )
    finally:
        if old_max is not None:
            engine.max_gen_tokens = old_max
    return engine.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def parse_args() -> argparse.Namespace:
    """解析离线 eval 参数。"""

    p = argparse.ArgumentParser(description="Evaluate SFT v4 sequence LoRA")
    p.add_argument("--jsonl", type=str, default="checkpoints/sft_v4_data/val.jsonl")
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--lora-dir", type=str, default="checkpoints/sft_v4_lora/latest/final")
    p.add_argument("--save-root", type=str, default="checkpoints/sft_v4_lora/latest")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--max-gen-tokens", type=int, default=80)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--full-range", action="store_true",
                   help="诊断模式：评估 [anchor0, anchor4]，默认只评估训练窗口 [anchor1-delta, anchor3]")
    p.add_argument("--with-teacher-ref", action="store_true",
                   help="可选：额外加载 base Qwen teacher，计算 analysis_bleu_vs_teacher")
    return p.parse_args()


def main() -> None:
    """SFT v4 离线自由生成评估入口。

    与训练不同，eval 默认不做 Phase B 的 GT scene 强制覆盖，memory 全程由学生自由
    输出更新；这样才能真实观察 scene recovery、stick、flip 和 step3 trigger。
    """

    _maybe_set_idle_gpu_mask()
    args = parse_args()

    ds = EpisodeDataset(pathlib.Path(args.jsonl))
    episodes = list(ds.rows)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    save_root = pathlib.Path(args.save_root)
    out_dir = save_root / "eval_v4"
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = LocalQwen3VLInstructEngine(
        pathlib.Path(args.model_dir),
        device=args.device,
        max_gen_tokens=args.max_gen_tokens,
        temperature=0.0,
        do_sample=False,
        repetition_penalty=args.repetition_penalty,
    )
    engine.load()
    if args.lora_dir:
        engine.attach_lora_adapter(args.lora_dir, merge=args.merge_lora)

    teacher_engine = None
    if args.with_teacher_ref:
        # teacher-ref 是可选重路径：额外加载一份 base Qwen，只计算分析文本 BLEU。
        teacher_engine = LocalQwen3VLInstructEngine(
            pathlib.Path(args.model_dir),
            device=args.device,
            max_gen_tokens=args.max_gen_tokens,
            temperature=0.0,
            do_sample=False,
            repetition_penalty=args.repetition_penalty,
        )
        teacher_engine.load()

    metrics = defaultdict(float)
    per_episode: List[Dict[str, Any]] = []
    phase_keys = ("phase_a", "phase_b")

    for ep in episodes:
        run_dir = pathlib.Path(ep.run_dir)
        eval_start = int(ep.anchors[0]) if args.full_range else int(ep.frame_start)
        eval_end = int(ep.anchors[4]) if args.full_range else int(ep.frame_end)
        gx, gy = _load_goal_xy(run_dir, eval_start)
        memory = init_memory(
            run_id=ep.run_id,
            sub_scenario_id=f"{ep.run_id}:{ep.anchors[1]}",
            ego_to_goal_x=gx,
            ego_to_goal_y=gy,
            gt_scene=ep.gt_scene,
        )

        frame_total = 0
        scene_correct = 0
        step3_total = 0
        status_correct = 0
        subgoal_correct = 0
        all_correct = 0
        scene_flip = 0
        scene_stick_ok = 0
        scene_stick_total = 0
        invalid_scene = 0
        invalid_status = 0
        invalid_subgoal = 0
        first_recover: Optional[int] = None
        phase_first_frame: Dict[str, Optional[int]] = {pk: None for pk in phase_keys}
        phase_first_recover: Dict[str, Optional[int]] = {pk: None for pk in phase_keys}
        prev_scene_ok: Optional[bool] = None

        for frame in range(eval_start, eval_end + 1):
            phase = "phase_a" if _is_phase_a(ep, frame) else "phase_b"
            if phase_first_frame[phase] is None:
                phase_first_frame[phase] = frame
            image_paths = _build_rgb_paths(run_dir, frame)
            try:
                images = _load_images(image_paths)
            except Exception:
                _prefetch_goal_xy_for_next_frame(memory, run_dir, frame + 1, eval_end)
                continue

            gt_status, gt_subgoal = _gt_status_subgoal(ep, frame)

            step1_user = build_step1_user_prompt(len(images))
            step1_msgs = _build_messages_with_images(user_text=step1_user, images=images)
            teacher_step1_text = ""
            if teacher_engine is not None:
                teacher_step1_text = _generate(teacher_engine, step1_msgs, images, max_new_tokens=80)
            step1_text = _generate(engine, step1_msgs, images, max_new_tokens=80)
            if teacher_engine is not None:
                metrics["analysis_bleu_sum"] += _simple_bleu(
                    _analysis_before_labels(step1_text),
                    _analysis_before_labels(teacher_step1_text),
                )
                metrics["analysis_bleu_count"] += 1

            teacher_step2_text = ""
            if teacher_engine is not None:
                teacher_step2_user = build_step2_teacher_prompt(memory, ep.gt_scene)
                teacher_step2_text = _generate_next_with_kv(teacher_engine, teacher_step2_user, max_new_tokens=60)
            step2_user = build_step2_student_prompt(memory)
            step2_text = _generate_next_with_kv(engine, step2_user, max_new_tokens=60)
            if teacher_engine is not None:
                metrics["analysis_bleu_sum"] += _simple_bleu(
                    _analysis_before_labels(step2_text),
                    _analysis_before_labels(teacher_step2_text),
                )
                metrics["analysis_bleu_count"] += 1
            p2 = parse_output(step2_text)
            if not validate_scene(p2.get("scene")):
                invalid_scene += 1

            old_scene = memory.scene
            memory = update_memory_after_step2(memory, student_scene=p2.get("scene"))
            if memory.scene != old_scene:
                scene_flip += 1

            scene_ok = should_trigger_step3(memory_scene_after_step2=memory.scene, gt_scene=ep.gt_scene)
            if scene_ok:
                scene_correct += 1
                if first_recover is None:
                    first_recover = frame - ep.frame_start
                if phase_first_recover[phase] is None:
                    # Phase 分项 recovery 从该 phase 的第一帧算起，用来观察 Phase A 末尾
                    # 是否已经锁定 GT，以及 Phase B 区间是否能保持。
                    phase_start = phase_first_frame[phase] if phase_first_frame[phase] is not None else frame
                    phase_first_recover[phase] = frame - phase_start

            # stick 统计：上一帧 scene 已正确时，本帧是否保持正确。
            if prev_scene_ok is True:
                scene_stick_total += 1
                if scene_ok:
                    scene_stick_ok += 1

            status_ok = False
            subgoal_ok = False
            if scene_ok:
                # 只有 scene 正确时才进入 step3；这与训练触发条件一致。
                step3_total += 1
                teacher_step3_text = ""
                if teacher_engine is not None:
                    teacher_step3_user = build_step3_teacher_prompt(memory, gt_status, gt_subgoal)
                    teacher_step3_text = _generate_next_with_kv(teacher_engine, teacher_step3_user, max_new_tokens=60)
                step3_user = build_step3_student_prompt(memory)
                step3_text = _generate_next_with_kv(engine, step3_user, max_new_tokens=60)
                if teacher_engine is not None:
                    metrics["analysis_bleu_sum"] += _simple_bleu(
                        _analysis_before_labels(step3_text),
                        _analysis_before_labels(teacher_step3_text),
                    )
                    metrics["analysis_bleu_count"] += 1
                p3 = parse_output(step3_text)

                pred_status = p3.get("status") if validate_event(memory.scene, p3.get("status")) else None
                pred_subgoal = p3.get("subgoal") if validate_event(memory.scene, p3.get("subgoal")) else None
                invalid_status += int(pred_status is None)
                invalid_subgoal += int(pred_subgoal is None)
                memory = update_memory_after_step3(memory, student_status=pred_status, student_subgoal=pred_subgoal)

                status_ok = memory.status == gt_status
                subgoal_ok = memory.subgoal == gt_subgoal
                if status_ok:
                    status_correct += 1
                if subgoal_ok:
                    subgoal_correct += 1

            if scene_ok and status_ok and subgoal_ok:
                all_correct += 1

            frame_total += 1
            prev_scene_ok = scene_ok

            metrics[f"{phase}/frames"] += 1
            metrics[f"{phase}/scene_correct"] += 1 if scene_ok else 0
            metrics[f"{phase}/step3_total"] += 1 if scene_ok else 0
            metrics[f"{phase}/status_correct"] += 1 if status_ok else 0
            metrics[f"{phase}/subgoal_correct"] += 1 if subgoal_ok else 0
            metrics[f"{phase}/all_correct"] += 1 if (scene_ok and status_ok and subgoal_ok) else 0
            _prefetch_goal_xy_for_next_frame(memory, run_dir, frame + 1, eval_end)

        epi = {
            "run_id": ep.run_id,
            "scenario": ep.scenario,
            "frames": frame_total,
            "scene_acc": scene_correct / max(frame_total, 1),
            "scene_recovery_steps": first_recover if first_recover is not None else -1,
            "scene_stick_rate": scene_stick_ok / max(scene_stick_total, 1),
            "scene_flip_rate": scene_flip / max(frame_total, 1),
            "step3_trigger_rate": step3_total / max(frame_total, 1),
            "status_acc_given_correct_scene": status_correct / max(step3_total, 1),
            "subgoal_acc_given_correct_scene": subgoal_correct / max(step3_total, 1),
            "all_acc": all_correct / max(frame_total, 1),
            "invalid_scene_rate": invalid_scene / max(frame_total, 1),
            "invalid_status_for_pred_scene_rate": invalid_status / max(step3_total, 1),
            "invalid_subgoal_for_pred_scene_rate": invalid_subgoal / max(step3_total, 1),
        }
        per_episode.append(epi)

        metrics["episodes"] += 1
        metrics["frames"] += frame_total
        metrics["scene_acc_sum"] += epi["scene_acc"]
        metrics["scene_stick_sum"] += epi["scene_stick_rate"]
        metrics["scene_flip_sum"] += epi["scene_flip_rate"]
        metrics["step3_sum"] += epi["step3_trigger_rate"]
        metrics["status_sum"] += epi["status_acc_given_correct_scene"]
        metrics["subgoal_sum"] += epi["subgoal_acc_given_correct_scene"]
        metrics["all_sum"] += epi["all_acc"]
        metrics["invalid_scene_sum"] += epi["invalid_scene_rate"]
        metrics["invalid_status_sum"] += epi["invalid_status_for_pred_scene_rate"]
        metrics["invalid_subgoal_sum"] += epi["invalid_subgoal_for_pred_scene_rate"]
        if epi["scene_recovery_steps"] >= 0:
            metrics["scene_recover_sum"] += epi["scene_recovery_steps"]
            metrics["scene_recover_count"] += 1
        for pk in phase_keys:
            rec = phase_first_recover[pk]
            if rec is not None:
                metrics[f"{pk}/scene_recover_sum"] += rec
                metrics[f"{pk}/scene_recover_count"] += 1

    n_epi = max(int(metrics["episodes"]), 1)
    summary = {
        "episodes": int(metrics["episodes"]),
        "frames": int(metrics["frames"]),
        "scene_acc_per_step": metrics["scene_acc_sum"] / n_epi,
        "scene_stick_rate": metrics["scene_stick_sum"] / n_epi,
        "scene_flip_rate": metrics["scene_flip_sum"] / n_epi,
        "step3_trigger_rate": metrics["step3_sum"] / n_epi,
        "status_acc_given_correct_scene": metrics["status_sum"] / n_epi,
        "subgoal_acc_given_correct_scene": metrics["subgoal_sum"] / n_epi,
        "all_acc_per_step": metrics["all_sum"] / n_epi,
        "invalid_scene_rate": metrics["invalid_scene_sum"] / n_epi,
        "invalid_status_for_pred_scene_rate": metrics["invalid_status_sum"] / n_epi,
        "invalid_subgoal_for_pred_scene_rate": metrics["invalid_subgoal_sum"] / n_epi,
        "scene_recovery_steps": (
            metrics["scene_recover_sum"] / max(metrics["scene_recover_count"], 1)
            if metrics["scene_recover_count"] > 0
            else -1
        ),
    }

    for pk in phase_keys:
        f = max(metrics[f"{pk}/frames"], 1)
        s3 = max(metrics[f"{pk}/step3_total"], 1)
        summary[f"{pk}_scene_acc_per_step"] = metrics[f"{pk}/scene_correct"] / f
        summary[f"{pk}_step3_trigger_rate"] = metrics[f"{pk}/step3_total"] / f
        summary[f"{pk}_status_acc_given_correct_scene"] = metrics[f"{pk}/status_correct"] / s3
        summary[f"{pk}_subgoal_acc_given_correct_scene"] = metrics[f"{pk}/subgoal_correct"] / s3
        summary[f"{pk}_all_acc_per_step"] = metrics[f"{pk}/all_correct"] / f
        summary[f"{pk}_scene_recovery_steps"] = (
            metrics[f"{pk}/scene_recover_sum"] / max(metrics[f"{pk}/scene_recover_count"], 1)
            if metrics[f"{pk}/scene_recover_count"] > 0
            else -1
        )

    if args.with_teacher_ref:
        summary["analysis_bleu_vs_teacher"] = (
            metrics["analysis_bleu_sum"] / max(metrics["analysis_bleu_count"], 1)
            if metrics["analysis_bleu_count"] > 0
            else -1
        )

    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "episodes.json").write_text(json.dumps(per_episode, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

