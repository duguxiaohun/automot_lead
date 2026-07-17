"""SFT v5 自由生成评估入口。

评估不使用 teacher，也不做 Phase B 纠偏；它按真实推理方式让 student 自己维护
RS/EVENT memory。Q1 RS 错时跳过本帧 Q2，下一有效帧恢复 GT RS + RE。

常用方式是传 ``--adapter-dir`` 评估训练后的 LoRA；``--output-json`` 保存聚合指标，
``--output-jsonl`` 可选保存逐帧完整输入输出。只检查 index 时使用 ``--check``，该模式
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
from qwen3vl_local.sft_v5.labels import option_for_event  # noqa: E402
from qwen3vl_local.sft_v5.metrics import StudentMetricsAccumulator  # noqa: E402
from qwen3vl_local.sft_v5.prompts import (  # noqa: E402
    Memory,
    build_q1_student_prompt,
    build_q2_student_prompt,
    parse_q1_output,
    parse_q2_output,
    reset_memory_for_frame,
    update_memory_after_q1,
    update_memory_after_q2,
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
    accumulator = StudentMetricsAccumulator()
    route_summaries: List[Dict[str, Any]] = []
    output_jsonl = pathlib.Path(args.output_jsonl) if args.output_jsonl else None
    jsonl_handle = None
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = output_jsonl.open("w", encoding="utf-8")
    processed_frames = 0
    try:
        for route in ds.rows:
            route_accumulator = StudentMetricsAccumulator()
            memory: Optional[Memory] = None
            reset_next = False
            for frame_index, frame in enumerate(route.frames):
                rs_target = _rs_target_from_frame(frame)
                event_target = _event_target_from_frame(frame)
                if memory is None or reset_next:
                    # 评估严格模拟 v5 推理状态机：首帧或上帧失败后，只恢复 GT RS + RE；
                    # 之后的 memory 完全由 student 自己的 Q1/Q2 输出维护。
                    memory = reset_memory_for_frame(rs_target, ego_to_goal_xy=frame.ego_to_goal_xy)
                    reset_next = False
                memory_before = {
                    "rs_label": memory.rs_label,
                    "event_label": memory.event_label,
                    "ego_to_goal_x": memory.ego_to_goal_x,
                    "ego_to_goal_y": memory.ego_to_goal_y,
                }
                images = _load_images(frame.history_rgb_paths)
                q1_prompt = build_q1_student_prompt(memory)
                q1_text, q1_after = _generate_start(
                    bundle,
                    images,
                    q1_prompt,
                    int(args.max_new_tokens_q1),
                )
                parsed_q1 = parse_q1_output(q1_text)
                q1_rs_ok = parsed_q1.get("rs_label") == frame.rs_label
                q1_abnormal = parsed_q1.get("abnormal") == "YES" if parsed_q1.get("abnormal") else None
                q1_abnormal_correct = bool(q1_abnormal == frame.abnormal) if q1_abnormal is not None else False
                memory = update_memory_after_q1(
                    memory,
                    student_rs_label=parsed_q1.get("rs_label"),
                    student_abnormal=q1_abnormal,
                )

                q2_triggered = False
                q2_prompt = ""
                q2_text = ""
                parsed_q2: Dict[str, Optional[str]] = {}
                q2_event_correct = False
                q2_candidate_mismatch = False
                q2_invalid = False
                if not q1_rs_ok:
                    # RS 错误会导致 Q2 候选空间错误，因此本帧不追问 EVENT；端到端
                    # event 指标会把真实 UE 计为漏检，conditional Q2 指标则不纳入分母。
                    reset_next = True
                    q1_after = None
                else:
                    q2_triggered = True
                    q2_prompt = build_q2_student_prompt(
                        memory,
                        option_map=frame.event_option_map,
                        q1_abnormal=bool(q1_abnormal),
                        regular_event_codes=frame.regular_event_codes,
                    )
                    q2_text, q2_after = _generate_next(
                        bundle,
                        q1_after,
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
                    if q2_invalid:
                        reset_next = True

                pred_event_label = parsed_q2.get("event_label")
                frame_log: Dict[str, Any] = {
                    "scenario": route.scenario,
                    "route_id": route.route_id,
                    "frame_id": frame.frame_id,
                    "history_rgb_paths": frame.history_rgb_paths,
                    "gt_rs_label": frame.rs_label,
                    "pred_rs_label": parsed_q1.get("rs_label"),
                    "gt_abnormal": bool(frame.abnormal),
                    "pred_abnormal": q1_abnormal,
                    "gt_event_label": event_target.label,
                    "pred_event_label": pred_event_label,
                    "pred_event_is_ue": None if pred_event_label is None else pred_event_label != "RE",
                    "q1_rs_correct": q1_rs_ok,
                    "q1_abnormal_correct": q1_abnormal_correct,
                    "q2_triggered": q2_triggered,
                    "q2_event_correct": q2_event_correct,
                    "q2_candidate_mismatch": q2_candidate_mismatch,
                    "q2_invalid_output": q2_invalid,
                    "reset_next": reset_next,
                    "rs_transition": bool(
                        frame_index > 0 and route.frames[frame_index - 1].rs_label != frame.rs_label
                    ),
                    "abnormal_transition": bool(
                        frame_index > 0
                        and bool(route.frames[frame_index - 1].abnormal) != bool(frame.abnormal)
                    ),
                    "memory_before": memory_before,
                    "memory_after": {
                        "rs_label": memory.rs_label,
                        "event_label": memory.event_label,
                        "ego_to_goal_x": memory.ego_to_goal_x,
                        "ego_to_goal_y": memory.ego_to_goal_y,
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
                processed_frames += 1
                progress_every = max(0, int(args.progress_frames))
                if progress_every > 0 and processed_frames % progress_every == 0:
                    if jsonl_handle is not None:
                        jsonl_handle.flush()
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
                    "metrics": route_summary["metrics"],
                }
            )
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

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
        "mean_valid_frames_per_route": (
            total_route_frames / route_count if route_count > 0 else None
        ),
    }
    summary.update(route_metrics)
    summary.update(
        {
            "schema_version": "sft_v5_eval_v2",
            "index": str(args.index),
            "model_dir": str(args.model_dir),
            "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
            "output_jsonl": str(output_jsonl) if output_jsonl is not None else None,
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
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
    # 正式大样本指标必须与训练 rollout 的安全上限对齐，避免长 CoT 被旧 probe 的
    # 256/192 可视化上限截断后误算成非法输出、假阴性或 EVENT 错误。
    p.add_argument("--max-new-tokens-q1", type=int, default=1024)
    p.add_argument("--max-new-tokens-q2", type=int, default=1024)
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
