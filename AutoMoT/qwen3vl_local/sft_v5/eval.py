"""SFT v5 自由生成评估入口。

评估不使用 teacher，也不做 Phase B 纠偏；它让 student 自己维护 RS/EVENT
memory。RS_SLOW 在稳定时低频运行，错误/recovery 时恢复逐帧；EVENT_FAST 在
每个 RS gate 正确的帧都重新读当前 RGB。RS 错时只跳过本帧 EVENT，后续帧
继续读取学生错误 memory，观察学生是否会自主纠正。独立 reference memory 只按
GT 推演并写入审计结果，绝不回写 student prompt。

常用方式是传 ``--adapter-dir`` 评估训练后的 LoRA；``--output-json`` 保存聚合指标，
``--output-jsonl`` 可选保存逐帧完整输入输出，``--transition-jsonl`` 只保存 RS/UE 变化、
边界误报和漏检。只检查 index 时使用 ``--check``，该模式
不会加载 Qwen。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch

from qwen3vl_local.sft_v3.train import _append_user_turn, _kv_start_state, _student_generate_kv  # noqa: E402
from qwen3vl_local.sft_v5.labels import RS_LABEL_TO_OPTION, option_for_event  # noqa: E402
from qwen3vl_local.sft_v5.metrics import (  # noqa: E402
    StudentMetricsAccumulator,
    build_transition_fields,
    transition_case_from_row,
    transition_case_is_informative,
)
from qwen3vl_local.sft_v5.prompts import (  # noqa: E402
    Memory,
    MemoryCurriculumConfig,
    MemoryCurriculumState,
    build_q1_student_prompt,
    build_q2_student_prompt,
    parse_q1_output,
    parse_q2_output,
    reset_memory_for_frame,
    should_run_rs_slow,
    should_run_event_fast,
    should_trigger_q2,
    update_memory_after_q1,
    update_memory_after_q2,
    update_memory_navigation,
    observe_training_memory,
)
from qwen3vl_local.sft_v5.train import (  # noqa: E402
    RouteSequenceDataset,
    _load_images,
    _messages,
    _rs_target_from_frame,
    _event_target_from_frame,
)


@dataclass
class EvalBundle:
    """评估时传递给 v3 KV helper 的轻量 bundle。"""

    model: Any
    processor: Any
    tokenizer: Any
    device: torch.device

    def unwrap(self) -> Any:
        """提供与训练 bundle 相同接口，供复用的 KV helper 取得底层模型。"""

        return self.model


def load_eval_bundle(model_dir: pathlib.Path, adapter_dir: Optional[pathlib.Path], device: torch.device, *, merge_lora: bool) -> EvalBundle:
    """加载 base Qwen 和可选 LoRA adapter。"""

    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError:
        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass

    kwargs = {"local_files_only": True, "trust_remote_code": True}
    try:
        model = ModelClass.from_pretrained(str(model_dir), dtype=torch.bfloat16, **kwargs)
    except TypeError:
        model = ModelClass.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16, **kwargs)
    if adapter_dir is not None:
        from peft import PeftModel

        # eval/probe 路径默认 merge LoRA，后续 KV decode 直接走普通模型 forward，
        # 避免 PEFT wrapper 在 Qwen3-VL incremental decode 中吞掉 cache_position。
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        if merge_lora and hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    return EvalBundle(model=model, processor=processor, tokenizer=processor.tokenizer, device=device)


def _generate_start(bundle: EvalBundle, images: List[Any], prompt: str, max_new_tokens: int) -> tuple[str, Any]:
    """对 Q1 做 system+image+user prefill，并返回生成后的 KV state。"""

    with torch.inference_mode():
        state = _kv_start_state(bundle, _messages(images, prompt))
        text, after, _ = _student_generate_kv(bundle, state, max_new_tokens)
    return text, after


def _generate_next(bundle: EvalBundle, previous_state: Any, prompt: str, max_new_tokens: int) -> tuple[str, Any]:
    """在上一问 assistant 输出后的 KV cache 上追加新的 user turn。"""

    with torch.inference_mode():
        state = _append_user_turn(bundle, previous_state, prompt)
        text, after, _ = _student_generate_kv(bundle, state, max_new_tokens)
    return text, after


def _generate(bundle: EvalBundle, images: List[Any], prompt: str, max_new_tokens: int) -> str:
    """兼容旧调用：单个 prompt fresh prefill + greedy decode。"""

    text, _ = _generate_start(bundle, images, prompt, max_new_tokens)
    return text


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """执行自由生成大样本评估，并流式累计假阳性/假阴性等指标。"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = RouteSequenceDataset(
        pathlib.Path(args.index),
        max_routes=int(args.max_routes),
        max_frames_per_route=int(args.max_frames_per_route),
    )
    if args.check:
        return {"route_count": len(ds), "check_only": True}
    bundle = load_eval_bundle(
        pathlib.Path(args.model_dir),
        pathlib.Path(args.adapter_dir) if args.adapter_dir else None,
        device,
        merge_lora=bool(args.merge_lora),
    )
    rs_schedule_config = MemoryCurriculumConfig(rs_slow_interval=int(args.rs_slow_interval))
    accumulator = StudentMetricsAccumulator()
    route_summaries: List[Dict[str, Any]] = []
    output_jsonl = pathlib.Path(args.output_jsonl) if args.output_jsonl else None
    transition_jsonl_arg = getattr(args, "transition_jsonl", None)
    transition_jsonl = pathlib.Path(transition_jsonl_arg) if transition_jsonl_arg else None
    jsonl_handle = None
    transition_handle = None
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = output_jsonl.open("w", encoding="utf-8")
    if transition_jsonl is not None:
        transition_jsonl.parent.mkdir(parents=True, exist_ok=True)
        transition_handle = transition_jsonl.open("w", encoding="utf-8")
    processed_frames = 0
    try:
        for route in ds.rows:
            route_accumulator = StudentMetricsAccumulator()
            memory: Optional[Memory] = None
            reference_memory: Optional[Memory] = None
            previous_pred_rs_label: Optional[str] = None
            previous_pred_abnormal: Optional[bool] = None
            rs_schedule_state = MemoryCurriculumState()
            for frame_index, frame in enumerate(route.frames):
                rs_target = _rs_target_from_frame(frame)
                event_target = _event_target_from_frame(frame)
                memory_initialized_from_gt = memory is None
                if memory is None:
                    # route 首帧建立 student/reference 共同起点。之后即使学生答错也不会
                    # 再触发 GT reset；reference 仅用于结果对比。
                    memory = reset_memory_for_frame(rs_target, ego_to_goal_xy=frame.ego_to_goal_xy)
                    reference_memory = reset_memory_for_frame(
                        rs_target,
                        ego_to_goal_xy=frame.ego_to_goal_xy,
                    )
                else:
                    memory = update_memory_navigation(memory, frame.ego_to_goal_xy)
                    assert reference_memory is not None
                    reference_memory = update_memory_navigation(
                        reference_memory,
                        frame.ego_to_goal_xy,
                    )
                assert reference_memory is not None
                memory_before = {
                    "rs_label": memory.rs_label,
                    "event_label": memory.event_label,
                    "ego_to_goal_x": memory.ego_to_goal_x,
                    "ego_to_goal_y": memory.ego_to_goal_y,
                }
                reference_memory_before = {
                    "rs_label": reference_memory.rs_label,
                    "event_label": reference_memory.event_label,
                    "ego_to_goal_x": reference_memory.ego_to_goal_x,
                    "ego_to_goal_y": reference_memory.ego_to_goal_y,
                }
                reference_memory_after_q1_state = update_memory_after_q1(
                    reference_memory,
                    student_rs_label=frame.rs_label,
                )
                images = _load_images(frame.history_rgb_paths)
                run_rs_slow, rs_schedule_reason = should_run_rs_slow(
                    rs_schedule_state,
                    rs_schedule_config,
                    memory=memory,
                    gt_rs_label=frame.rs_label,
                    frame_ordinal=frame_index,
                )
                q1_prompt = build_q1_student_prompt(memory) if run_rs_slow else ""
                q1_text = ""
                q1_after: Optional[Any] = None
                parsed_q1: Dict[str, Optional[str]] = {}
                q1_rs_ok = False
                if run_rs_slow:
                    q1_text, q1_after = _generate_start(
                        bundle,
                        images,
                        q1_prompt,
                        int(args.max_new_tokens_q1),
                    )
                    parsed_q1 = parse_q1_output(q1_text)
                    q1_rs_ok = should_trigger_q2(
                        student_rs_label=parsed_q1.get("rs_label"),
                        target_rs_label=frame.rs_label,
                    )
                    memory = update_memory_after_q1(
                        memory,
                        student_rs_label=parsed_q1.get("rs_label"),
                    )
                # 慢帧必须以“本帧 Q1 解析且答对”为 gate；不能因为旧 RS memory 恰好
                # 正确，就在本帧 Q1 无效/答错后继续问 EVENT。只有没有 Q1 的稳定快帧
                # 才按复用的 RS memory 判 gate。
                rs_gate_ok = should_run_event_fast(
                    rs_slow_ran=run_rs_slow,
                    q1_rs_correct=q1_rs_ok,
                    memory_rs_label=memory.rs_label,
                    target_rs_label=frame.rs_label,
                )
                student_memory_after_q1 = {
                    "rs_label": memory.rs_label,
                    "event_label": memory.event_label,
                    "ego_to_goal_x": memory.ego_to_goal_x,
                    "ego_to_goal_y": memory.ego_to_goal_y,
                }
                reference_memory_after_q1 = {
                    "rs_label": reference_memory_after_q1_state.rs_label,
                    "event_label": reference_memory_after_q1_state.event_label,
                    "ego_to_goal_x": reference_memory_after_q1_state.ego_to_goal_x,
                    "ego_to_goal_y": reference_memory_after_q1_state.ego_to_goal_y,
                }

                q2_triggered = False
                q2_prompt = ""
                q2_text = ""
                parsed_q2: Dict[str, Optional[str]] = {}
                q2_event_correct = False
                q2_candidate_mismatch = False
                q2_invalid = False
                q2_student_memory_input: Optional[Dict[str, Any]] = None
                if not rs_gate_ok:
                    # RS 错误会导致 Q2 候选空间错误，因此本帧不追问 EVENT；端到端
                    # event 指标会把真实 UE 计为漏检，conditional Q2 指标则不纳入分母。
                    q1_after = None
                else:
                    q2_triggered = True
                    q2_student_memory_input = dict(student_memory_after_q1)
                    q2_prompt = build_q2_student_prompt(
                        memory,
                        option_map=frame.event_option_map,
                        regular_event_codes=frame.regular_event_codes,
                    )
                    if run_rs_slow:
                        assert q1_after is not None
                        q2_text, q2_after = _generate_next(
                            bundle,
                            q1_after,
                            q2_prompt,
                            int(args.max_new_tokens_q2),
                        )
                    else:
                        # 稳定 fast frame 没有 Q1 KV，EVENT_FAST 必须 fresh prefill，
                        # 不能伪造或复用上一个慢帧的 ABNORMAL/analysis。
                        q2_text, q2_after = _generate_start(
                            bundle,
                            images,
                            q2_prompt,
                            int(args.max_new_tokens_q2),
                        )
                    # 大样本 eval 不复用 Q2 KV；显式释放，避免普通变量 `_` 把它持有到
                    # 下一帧 full prefill，造成看似随机的显存高水位。
                    del q2_after
                    q1_after = None
                    parsed_q2 = parse_q2_output(q2_text, frame.event_option_map)
                    target = _event_target_from_frame(frame, student_event=parsed_q2.get("event_label"))
                    # target 用 student_event 动态解析，和训练一致：双 UE / 双 RE 时如果
                    # 学生选中可接受标签之一，就按该标签计正确。
                    q2_candidate_mismatch = option_for_event(target.label, frame.event_option_map) is None
                    q2_event_correct = parsed_q2.get("event_label") == target.label
                    memory = update_memory_after_q2(memory, student_event_label=parsed_q2.get("event_label"))
                    q2_invalid = parsed_q2.get("event_label") is None

                pred_event_label = parsed_q2.get("event_label")
                pred_abnormal = None if pred_event_label is None else pred_event_label != "RE"
                rs_schedule_state.frames_seen = frame_index + 1
                observe_training_memory(
                    rs_schedule_state,
                    rs_schedule_config,
                    rs_correct=q1_rs_ok,
                    rs_checked=run_rs_slow,
                    event_checked=q2_triggered,
                    event_correct=q2_event_correct,
                )

                # reference Q2 总是按 GT 推演，用于回答“该问之后 memory 理论上应该是什么”。
                # 它不会影响上面的 student memory，也不会决定下一帧 student prompt。
                reference_memory_after_q2_state = update_memory_after_q2(
                    reference_memory_after_q1_state,
                    student_event_label=event_target.label,
                )
                reference_memory = reference_memory_after_q2_state
                reference_memory_after_q2 = {
                    "rs_label": reference_memory.rs_label,
                    "event_label": reference_memory.event_label,
                    "ego_to_goal_x": reference_memory.ego_to_goal_x,
                    "ego_to_goal_y": reference_memory.ego_to_goal_y,
                }
                student_memory_after_q2 = (
                    {
                        "rs_label": memory.rs_label,
                        "event_label": memory.event_label,
                        "ego_to_goal_x": memory.ego_to_goal_x,
                        "ego_to_goal_y": memory.ego_to_goal_y,
                    }
                    if q2_triggered
                    else None
                )
                student_memory_for_next_frame = {
                    "rs_label": memory.rs_label,
                    "event_label": memory.event_label,
                    "ego_to_goal_x": memory.ego_to_goal_x,
                    "ego_to_goal_y": memory.ego_to_goal_y,
                }
                raw_ue_labels = [
                    code for code in event_target.raw_events if str(code).startswith("U-E")
                ]
                accepted_event_labels = set(raw_ue_labels or ["RE"])
                q1_input_matches_target = memory_before["rs_label"] == frame.rs_label
                q1_after_matches_target = student_memory_after_q1["rs_label"] == frame.rs_label
                q2_input_matches_target = bool(
                    q2_student_memory_input is not None
                    and q2_student_memory_input["event_label"] in accepted_event_labels
                )
                q2_after_matches_target = bool(
                    student_memory_after_q2 is not None
                    and student_memory_after_q2["event_label"] in accepted_event_labels
                )
                would_reset_under_training = bool(not rs_gate_ok or (q2_triggered and q2_invalid))
                rs_memory_known_wrong = bool(
                    memory_before["rs_label"] in RS_LABEL_TO_OPTION
                    and memory_before["rs_label"] != frame.rs_label
                )
                rs_memory_unknown = memory_before["rs_label"] not in RS_LABEL_TO_OPTION
                event_memory_label = (
                    q2_student_memory_input.get("event_label")
                    if q2_student_memory_input is not None
                    else None
                )
                event_memory_known = bool(
                    event_memory_label == "RE"
                    or (isinstance(event_memory_label, str) and event_memory_label.startswith("U-E"))
                )
                event_memory_wrong = bool(
                    q2_triggered
                    and event_memory_known
                    and event_memory_label not in accepted_event_labels
                )
                event_memory_unknown = bool(q2_triggered and not event_memory_known)

                pred_rs_label = memory.rs_label if memory.rs_label in RS_LABEL_TO_OPTION else None
                transition_fields = build_transition_fields(
                    pair_evaluated=frame_index > 0,
                    previous_frame_id=(route.frames[frame_index - 1].frame_id if frame_index > 0 else None),
                    previous_gt_rs_label=(route.frames[frame_index - 1].rs_label if frame_index > 0 else None),
                    gt_rs_label=frame.rs_label,
                    previous_pred_rs_label=previous_pred_rs_label,
                    pred_rs_label=pred_rs_label,
                    previous_gt_abnormal=(
                        bool(route.frames[frame_index - 1].abnormal) if frame_index > 0 else None
                    ),
                    gt_abnormal=bool(frame.abnormal),
                    previous_pred_abnormal=previous_pred_abnormal,
                    pred_abnormal=pred_abnormal,
                )
                frame_log: Dict[str, Any] = {
                    "scenario": route.scenario,
                    "route_id": route.route_id,
                    "frame_id": frame.frame_id,
                    "history_rgb_paths": frame.history_rgb_paths,
                    "gt_rs_label": frame.rs_label,
                    "pred_rs_label": pred_rs_label,
                    "gt_abnormal": bool(frame.abnormal),
                    "pred_abnormal": pred_abnormal,
                    "gt_event_label": event_target.label,
                    "pred_event_label": pred_event_label,
                    "pred_event_is_ue": None if pred_event_label is None else pred_event_label != "RE",
                    "q1_triggered": run_rs_slow,
                    "rs_slow_reason": rs_schedule_reason,
                    "q1_rs_correct": q1_rs_ok,
                    "rs_gate_correct": rs_gate_ok,
                    "event_family_correct": bool(
                        pred_abnormal is not None and pred_abnormal == bool(frame.abnormal)
                    ),
                    # 旧 JSONL schema 兼容别名；实际来自 EVENT_FAST 的 RE/UE family。
                    "q1_abnormal_correct": bool(
                        pred_abnormal is not None and pred_abnormal == bool(frame.abnormal)
                    ),
                    "q2_triggered": q2_triggered,
                    "q2_skipped_rs_wrong": bool(not rs_gate_ok),
                    "q2_event_correct": q2_event_correct,
                    "q2_candidate_mismatch": q2_candidate_mismatch,
                    "q2_invalid_output": q2_invalid,
                    "reset_next": False,
                    "would_reset_under_training": would_reset_under_training,
                    "memory_forced_correction_applied": False,
                    "memory_rs_input_known_wrong": rs_memory_known_wrong,
                    "memory_rs_input_unknown": rs_memory_unknown,
                    "memory_rs_copied_when_wrong": bool(
                        run_rs_slow
                        and rs_memory_known_wrong
                        and pred_rs_label == memory_before["rs_label"]
                    ),
                    "memory_rs_recovered": bool(
                        run_rs_slow and (rs_memory_known_wrong or rs_memory_unknown) and q1_rs_ok
                    ),
                    "memory_event_input_known_wrong": event_memory_wrong,
                    "memory_event_input_unknown": event_memory_unknown,
                    "memory_event_copied_when_wrong": bool(
                        event_memory_wrong and pred_event_label == event_memory_label
                    ),
                    "memory_event_recovered": bool(
                        q2_triggered
                        and (event_memory_wrong or event_memory_unknown)
                        and pred_event_label in accepted_event_labels
                    ),
                    "rs_transition": bool(
                        frame_index > 0 and route.frames[frame_index - 1].rs_label != frame.rs_label
                    ),
                    "abnormal_transition": bool(
                        frame_index > 0
                        and bool(route.frames[frame_index - 1].abnormal) != bool(frame.abnormal)
                    ),
                    **transition_fields,
                    "memory_before": memory_before,
                    "memory_after": student_memory_for_next_frame,
                    "reference_memory_before": reference_memory_before,
                    "reference_memory_after": reference_memory_after_q2,
                    "memory_trace": {
                        "policy": "student_closed_loop",
                        "route_initialized_from_ground_truth": memory_initialized_from_gt,
                        "reference_is_comparison_only": True,
                        "forced_correction_applied": False,
                        "q1": {
                            "triggered": run_rs_slow,
                            "schedule_reason": rs_schedule_reason,
                            "input_student": memory_before,
                            "input_reference": reference_memory_before,
                            "after_student_output": student_memory_after_q1,
                            "after_ground_truth_reference": reference_memory_after_q1,
                            "input_matches_current_frame_target": q1_input_matches_target,
                            "after_matches_current_frame_target": q1_after_matches_target,
                        },
                        "q2": {
                            "triggered": q2_triggered,
                            "input_student": q2_student_memory_input,
                            "input_ground_truth_reference": reference_memory_after_q1,
                            "after_student_output": student_memory_after_q2,
                            "after_ground_truth_reference": reference_memory_after_q2,
                            "input_matches_current_frame_target": (
                                q2_input_matches_target if q2_triggered else None
                            ),
                            "after_matches_current_frame_target": (
                                q2_after_matches_target if q2_triggered else None
                            ),
                        },
                        "autonomous_change": {
                            "q1_rs_corrected_by_student": bool(
                                run_rs_slow
                                and not q1_input_matches_target
                                and q1_after_matches_target
                            ),
                            "q1_rs_corrupted_by_student": bool(
                                run_rs_slow
                                and q1_input_matches_target
                                and not q1_after_matches_target
                            ),
                            "q2_event_corrected_by_student": bool(
                                q2_triggered
                                and not q2_input_matches_target
                                and q2_after_matches_target
                            ),
                            "q2_event_corrupted_by_student": bool(
                                q2_triggered
                                and q2_input_matches_target
                                and not q2_after_matches_target
                            ),
                        },
                    },
                    # JSONL 可选保存完整文本输入输出，方便从大样本指标反查具体 FP/FN。
                    "q1_prompt": q1_prompt,
                    "q1_output": q1_text,
                    "parsed_q1": parsed_q1,
                    "q2_prompt": q2_prompt,
                    "q2_output": q2_text,
                    "parsed_q2": parsed_q2,
                    "event_option_map": frame.event_option_map,
                }
                accumulator.update(frame_log)
                route_accumulator.update(frame_log)
                if jsonl_handle is not None:
                    jsonl_handle.write(json.dumps(frame_log, ensure_ascii=False) + "\n")
                transition_case = transition_case_from_row(frame_log)
                if (
                    transition_handle is not None
                    and transition_case is not None
                    and transition_case_is_informative(transition_case)
                ):
                    transition_handle.write(json.dumps(transition_case, ensure_ascii=False) + "\n")
                # 下一帧的变化判断使用本帧实际模型输出。解析失败时保留
                # None，使后续帧对记为 invalid，不伪装成“未变化”。
                previous_pred_rs_label = pred_rs_label
                previous_pred_abnormal = pred_abnormal
                processed_frames += 1
                progress_every = max(0, int(args.progress_frames))
                if progress_every > 0 and processed_frames % progress_every == 0:
                    if jsonl_handle is not None:
                        jsonl_handle.flush()
                    if transition_handle is not None:
                        transition_handle.flush()
                    print(
                        f"[eval] frames={processed_frames} scenario={route.scenario} "
                        f"route={route.route_id} frame={frame.frame_id}",
                        flush=True,
                    )
            route_summary = route_accumulator.summary()
            route_summaries.append(
                {
                    "scenario": route.scenario,
                    "route_id": route.route_id,
                    "frames": route_summary["frames"],
                    "reset_count": route_summary["reset_count"],
                    "training_reset_recommendation_count": route_summary[
                        "training_reset_recommendation_count"
                    ],
                    "metrics": route_summary["metrics"],
                }
            )
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()
        if transition_handle is not None:
            transition_handle.close()

    summary = accumulator.summary()
    def _mean_defined(values: List[Optional[float]]) -> Optional[float]:
        """route macro 指标忽略没有正类/预测而为 None 的 route。"""

        valid = [float(value) for value in values if value is not None]
        return sum(valid) / len(valid) if valid else None

    route_count = len(route_summaries)
    total_route_frames = sum(int(item["frames"]) for item in route_summaries)
    route_metrics = {
        "route_rs_all_correct_ratio": (
            sum(item["metrics"].get("rs_acc") == 1.0 for item in route_summaries) / route_count
            if route_count > 0
            else None
        ),
        "route_abnormal_f1_macro": _mean_defined(
            [item["metrics"].get("abnormal_f1") for item in route_summaries]
        ),
        "route_ue_f1_macro": _mean_defined(
            [item["metrics"].get("q2_ue_f1") for item in route_summaries]
        ),
        "mean_resets_per_100_frames": (
            100.0 * sum(int(item["reset_count"]) for item in route_summaries) / total_route_frames
            if total_route_frames > 0
            else None
        ),
        "mean_training_reset_recommendations_per_100_frames": (
            100.0
            * sum(
                int(item["training_reset_recommendation_count"])
                for item in route_summaries
            )
            / total_route_frames
            if total_route_frames > 0
            else None
        ),
        "mean_valid_frames_per_route": (
            total_route_frames / route_count if route_count > 0 else None
        ),
    }
    summary.update(route_metrics)
    summary.update(
        {
            "schema_version": "sft_v5_eval_v4",
            "memory_policy": "student_closed_loop_reference_comparison_only",
            "index": str(args.index),
            "model_dir": str(args.model_dir),
            "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
            "output_jsonl": str(output_jsonl) if output_jsonl is not None else None,
            "transition_jsonl": str(transition_jsonl) if transition_jsonl is not None else None,
            "route_metrics": route_metrics,
            "route_summaries": route_summaries,
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    """解析大样本 eval CLI；默认 1024/1024 与正式训练 rollout 对齐。"""

    p = argparse.ArgumentParser(description="Evaluate SFT v5 adapter")
    p.add_argument("--index", type=str, required=True)
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--adapter-dir", type=str, default=None)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--output-jsonl", type=str, default=None, help="可选逐帧完整 prompt/output/解析记录，便于回查 FP/FN")
    p.add_argument(
        "--transition-jsonl",
        type=str,
        default=None,
        help="可选轻量变化帧报告；只写真实/预测 RS 变化、UE 进入/退出及 FP/FN",
    )
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
    # 正式大样本指标必须与训练 rollout 的安全上限对齐，避免长 CoT 被旧 probe 的
    # 256/192 可视化上限截断后误算成非法输出、假阴性或 EVENT 错误。
    p.add_argument("--max-new-tokens-q1", type=int, default=1024)
    p.add_argument("--max-new-tokens-q2", type=int, default=1024)
    p.add_argument("--rs-slow-interval", type=int, default=4, help="稳定 RS 每隔多少有效帧运行一次 RS_SLOW")
    p.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progress-frames", type=int, default=20)
    p.add_argument("--check", action="store_true")
    return p.parse_args()


def main() -> None:
    """运行评估，在终端打印结果并按需写 ``--output-json``。"""

    args = parse_args()
    metrics = evaluate(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_json:
        path = pathlib.Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
