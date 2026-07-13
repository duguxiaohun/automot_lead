"""SFT v5 真实 Qwen batched rollout 对照测试。

这个脚本会加载普通 Qwen / 可选 LoRA，用同一组 frame 分别跑：

1. 单样本 Q1 prefill + greedy rollout；
2. batched Q1 prefill + greedy rollout；
3. 在各自 Q1 KV cache 后继续追加同一个 Q2 user turn；
4. 在同一批 q1_ids 上比较训练真正使用的 student logits。

它的目的不是评估准确率，而是验证 `QWEN_BATCH_SIZE>1` 的阶段 1 优化没有改变
Q1/Q2 续接语义。没有模型目录或 sequence index 时不要运行；服务器上改完 batch
逻辑后应优先跑这个 smoke，再考虑开大训练。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from qwen3vl_local.sft_v3.train import (  # noqa: E402
    _append_token_ids,
    _append_token_ids_with_logits,
    _append_user_turn,
    _clone_kv_state,
    _kv_start_state,
    _student_generate_kv,
)
from qwen3vl_local.sft_v5.eval import load_eval_bundle  # noqa: E402
from qwen3vl_local.sft_v5.prompts import (  # noqa: E402
    Memory,
    build_q1_student_prompt,
    build_q2_student_prompt,
    parse_q1_output,
    update_memory_after_q1,
)
from qwen3vl_local.sft_v5.train import (  # noqa: E402
    FrameRow,
    RouteSequenceDataset,
    _load_images,
    _messages,
    _qwen_message_input_length,
    _reset_memory_for_frame_row,
    _run_q1_rollout_grouped,
)


def _first_token(ids: torch.Tensor) -> Optional[int]:
    """返回生成序列首 token；空输出用 None 表示。"""

    if ids.numel() == 0:
        return None
    return int(ids.reshape(-1)[0].item())


def _ids_list(ids: torch.Tensor) -> List[int]:
    """把 shape=(1,T) 的 ids 转成普通 list，方便 JSON 对照。"""

    if ids.numel() == 0:
        return []
    return [int(x) for x in ids.reshape(-1).detach().cpu().tolist()]


def _run_single_q1(bundle: Any, frame: FrameRow, memory: Memory, max_new_tokens: int) -> Tuple[Any, str, Any, torch.Tensor]:
    """单样本基线：fresh prefill 后生成 Q1。"""

    images = _load_images(frame.history_rgb_paths)
    messages = _messages(images, build_q1_student_prompt(memory))
    with torch.inference_mode():
        state = _kv_start_state(bundle, messages)
        text, after, ids = _student_generate_kv(bundle, state, max_new_tokens)
    return state, text, after, ids


def _run_q2_from_q1(bundle: Any, q1_after: Any, frame: FrameRow, memory: Memory, q1_text: str, max_new_tokens: int) -> Tuple[str, torch.Tensor]:
    """在 Q1 assistant 输出后的 KV cache 上追加 Q2，验证续接状态是否等价。"""

    parsed = parse_q1_output(q1_text)
    abnormal = parsed.get("abnormal") == "YES" if parsed.get("abnormal") else None
    memory_after_q1 = update_memory_after_q1(
        memory,
        student_rs_label=parsed.get("rs_label"),
        student_abnormal=abnormal,
    )
    q2_prompt = build_q2_student_prompt(
        memory_after_q1,
        option_map=frame.event_option_map,
        q1_abnormal=bool(abnormal),
        regular_event_codes=frame.regular_event_codes,
    )
    with torch.inference_mode():
        q2_state = _append_user_turn(bundle, q1_after, q2_prompt)
        q2_text, _q2_after, q2_ids = _student_generate_kv(bundle, q2_state, max_new_tokens)
    return q2_text, q2_ids


def _q1_input_length(bundle: Any, frame: FrameRow, memory: Memory) -> int:
    """计算 Q1 student prompt 的真实 processor input length。"""

    images = _load_images(frame.history_rgb_paths)
    return _qwen_message_input_length(bundle, _messages(images, build_q1_student_prompt(memory)))


def _select_case_indices(
    lengths: List[int],
    *,
    num_cases: int,
    prefer_different_lengths: bool,
    require_batched_group: bool,
) -> List[int]:
    """选择 smoke case；可强制至少选择两个样本以触发 padded batched rollout。"""

    if not lengths:
        return []
    by_length: Dict[int, List[int]] = {}
    for idx, length in enumerate(lengths):
        by_length.setdefault(int(length), []).append(idx)
    if require_batched_group:
        return list(range(min(max(2, num_cases), len(lengths))))
    if not prefer_different_lengths or len(lengths) == 1:
        return list(range(min(num_cases, len(lengths))))
    # 默认模式刻意挑最长/最短输入，制造 padding 压力。当前 Q1/Q2 的 padded
    # rollout 只用于采样 token，KL/Q2 续接会重建单样本精确 KV；这个 smoke
    # 正是用来确认 mixed-length padded rollout 没改变 student 采样语义。
    min_idx = min(range(len(lengths)), key=lambda i: lengths[i])
    max_idx = max(range(len(lengths)), key=lambda i: lengths[i])
    selected: List[int] = []
    for idx in (min_idx, max_idx):
        if idx not in selected:
            selected.append(idx)
    for idx in range(len(lengths)):
        if len(selected) >= num_cases:
            break
        if idx not in selected:
            selected.append(idx)
    return selected[:num_cases]


def _logit_diff_on_ids(bundle: Any, single_state: Any, batch_state: Any, ids: torch.Tensor) -> Dict[str, float]:
    """比较训练 KL 路径实际使用的 logits，而不只比较生成文本。"""

    if ids.numel() == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0}
    with torch.inference_mode():
        _, single_logits, _ = _append_token_ids_with_logits(bundle, _clone_kv_state(single_state), ids)
        _, batch_logits, _ = _append_token_ids_with_logits(bundle, _clone_kv_state(batch_state), ids)
    diff = (single_logits.float() - batch_logits.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _rebuild_q1_state_after(bundle: Any, frame: FrameRow, memory: Memory, ids: torch.Tensor) -> Tuple[Any, Any]:
    """用 batched rollout 采样出的 Q1 token 重建单样本精确 Q1 state/after。"""

    images = _load_images(frame.history_rgb_paths)
    with torch.inference_mode():
        state = _kv_start_state(bundle, _messages(images, build_q1_student_prompt(memory)))
        after, _ = _append_token_ids(bundle, state, ids)
    return state, after


def _collect_cases(index: pathlib.Path, num_cases: int, max_routes: int) -> Tuple[List[FrameRow], List[Memory]]:
    """从 sequence index 顺序取若干帧，保证真实 RGB/meta/prompt 全链路可用。"""

    ds = RouteSequenceDataset(index, max_routes=max_routes, max_frames_per_route=0)
    frames: List[FrameRow] = []
    memories: List[Memory] = []
    for route in ds.rows:
        for frame in route.frames:
            frames.append(frame)
            memories.append(_reset_memory_for_frame_row(frame))
            if len(frames) >= num_cases:
                return frames, memories
    return frames, memories


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    """执行 batch-vs-single 对照并返回 JSON 报告。"""

    if bool(args.require_batched_group) and int(args.num_cases) < 2:
        raise ValueError("--require-batched-group requires --num-cases >= 2.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    candidate_frames, candidate_memories = _collect_cases(
        pathlib.Path(args.index),
        max(int(args.num_cases), int(args.candidate_pool)),
        int(args.max_routes),
    )
    if len(candidate_frames) < 2:
        raise RuntimeError("Need at least two valid frames to test batched Qwen rollout.")

    bundle = load_eval_bundle(
        pathlib.Path(args.model_dir),
        pathlib.Path(args.adapter_dir) if args.adapter_dir else None,
        device,
        merge_lora=bool(args.merge_lora),
    )
    # 先对候选池逐帧计算 Q1 输入长度。当前训练允许 mixed-length padded rollout；
    # length 仍用于报告 padding pressure，帮助判断本次 smoke 是否覆盖了混长 batch。
    candidate_lengths = [
        _q1_input_length(bundle, frame, memory)
        for frame, memory in zip(candidate_frames, candidate_memories)
    ]
    selected_indices = _select_case_indices(
        candidate_lengths,
        num_cases=int(args.num_cases),
        prefer_different_lengths=bool(args.prefer_different_lengths),
        require_batched_group=bool(args.require_batched_group),
    )
    if not selected_indices:
        length_histogram: Dict[int, int] = {}
        for length in candidate_lengths:
            length_histogram[int(length)] = length_histogram.get(int(length), 0) + 1
        raise RuntimeError(
            "Need at least two candidate frames to exercise batched rollout; "
            f"increase --candidate-pool or disable --require-batched-group. length_histogram={length_histogram}"
        )
    frames = [candidate_frames[i] for i in selected_indices]
    memories = [candidate_memories[i] for i in selected_indices]
    input_lengths = [candidate_lengths[i] for i in selected_indices]

    single_q1 = [
        _run_single_q1(bundle, frame, memory, int(args.max_new_tokens_q1))
        for frame, memory in zip(frames, memories)
    ]
    grouped_q1 = _run_q1_rollout_grouped(
        bundle,
        memories,
        frames,
        max_new_tokens_q1=int(args.max_new_tokens_q1),
    )
    batched_q1 = grouped_q1.rollouts
    if bool(args.require_batched_group) and grouped_q1.batched_frames < 2:
        # 防御性检查：要求 grouped 运行结果报告真实 batched_frames>=2，
        # 避免因为后续逻辑变化让测试“假通过”。
        raise RuntimeError(
            "Smoke did not exercise a real batched rollout group even though --require-batched-group was set; "
            f"group_sizes={grouped_q1.group_sizes} input_lengths={input_lengths}"
        )

    cases: List[Dict[str, Any]] = []
    ok = True
    for idx, (frame, memory) in enumerate(zip(frames, memories)):
        single_state, single_text, single_after, single_ids = single_q1[idx]
        batch_state, batch_text, batch_after, batch_ids = batched_q1[idx]
        if batch_state is None or batch_after is None:
            batch_state, batch_after = _rebuild_q1_state_after(bundle, frame, memory, batch_ids)
        # 训练真正优化的是 _append_token_ids_with_logits 上的 KL，不只是 generate 文本。
        # 所以这里要比较同一批 q1_ids 对应的 logits 差异。
        q1_logit_diff = _logit_diff_on_ids(bundle, single_state, batch_state, single_ids)
        single_q2_text, single_q2_ids = _run_q2_from_q1(
            bundle,
            single_after,
            frame,
            memory,
            single_text,
            int(args.max_new_tokens_q2),
        )
        batch_q2_text, batch_q2_ids = _run_q2_from_q1(
            bundle,
            batch_after,
            frame,
            memory,
            batch_text,
            int(args.max_new_tokens_q2),
        )
        q1_ids_equal = _ids_list(single_ids) == _ids_list(batch_ids)
        q2_ids_equal = _ids_list(single_q2_ids) == _ids_list(batch_q2_ids)
        q1_logits_ok = q1_logit_diff["max_abs"] <= float(args.logit_atol)
        # case_ok 同时覆盖四层等价性：
        # 1. 首 token；2. 完整 Q1 token/text；3. 训练 logits；4. Q1 KV 后续接 Q2。
        case_ok = (
            _first_token(single_ids) == _first_token(batch_ids)
            and single_text == batch_text
            and q1_ids_equal
            and q1_logits_ok
            and single_q2_text == batch_q2_text
            and q2_ids_equal
        )
        ok = ok and case_ok
        cases.append(
            {
                "index": idx,
                "frame_id": frame.frame_id,
                "q1_input_length": input_lengths[idx],
                "first_token_single": _first_token(single_ids),
                "first_token_batched": _first_token(batch_ids),
                "q1_ids_equal": q1_ids_equal,
                "q1_text_equal": single_text == batch_text,
                "q1_logits_max_abs": q1_logit_diff["max_abs"],
                "q1_logits_mean_abs": q1_logit_diff["mean_abs"],
                "q1_logits_ok": q1_logits_ok,
                "q2_ids_equal": q2_ids_equal,
                "q2_text_equal": single_q2_text == batch_q2_text,
                "ok": case_ok,
                "single_q1_text": single_text,
                "batched_q1_text": batch_text,
                "single_q2_text": single_q2_text,
                "batched_q2_text": batch_q2_text,
            }
        )

    return {
        "ok": ok,
        "num_cases": len(cases),
        "model_dir": str(args.model_dir),
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "selected_indices": selected_indices,
        "input_lengths": input_lengths,
        "padding_pressure": len(set(input_lengths)) > 1,
        "actual_group_sizes": grouped_q1.group_sizes,
        "actual_batched_group_sizes": grouped_q1.batched_group_sizes,
        "actual_batched_groups": grouped_q1.batched_groups,
        "actual_singleton_groups": grouped_q1.singleton_groups,
        "actual_batched_frames": grouped_q1.batched_frames,
        "length_histogram": grouped_q1.length_histogram,
        "length_seconds": grouped_q1.length_seconds,
        "require_batched_group": bool(args.require_batched_group),
        "logit_atol": float(args.logit_atol),
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare SFT v5 single Qwen rollout with batched Qwen rollout")
    p.add_argument("--index", type=str, required=True)
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--adapter-dir", type=str, default=None)
    p.add_argument("--num-cases", type=int, default=2)
    p.add_argument("--candidate-pool", type=int, default=32)
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--max-new-tokens-q1", type=int, default=256)
    p.add_argument("--max-new-tokens-q2", type=int, default=192)
    p.add_argument("--logit-atol", type=float, default=0.5)
    p.add_argument("--prefer-different-lengths", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-batched-group", action="store_true", help="必须运行至少一个 size>=2 的真实 batched rollout group")
    p.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--no-fail", action="store_true", help="即使发现不一致也只打印 JSON，不返回非 0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_smoke(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_json:
        path = pathlib.Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["ok"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
