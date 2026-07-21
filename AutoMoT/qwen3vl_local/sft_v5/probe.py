"""SFT v5 case-level probe。

默认不加载模型，只按连续短片段检查 label / memory / prompt 合同。传 `--with-model`
后会额外生成 student Q1/Q2 输出；传
`--with-teacher-model` 后会额外用默认/base Qwen 跑 privileged teacher prompt。
训练前 OPSD 能力体检必须不传 `--adapter-dir`，即 teacher/student 都只用普通 Qwen，
不导入任何 LoRA。

默认 ``review`` 按 ``scenarios/<scenario>__<route>/frame_<id>/`` 保存连续帧。每帧只保留
实际输入 RGB、``input.json``、``output.json`` 和 ``memory.json``：输入、学生/老师
输出、解析结构、场景真值与两问 memory 转换都有唯一入口。``compact`` 只写汇总
``results.json``；显式指定 ``--artifact-level full`` 时才额外生成旧式逐项文件。

小样本只保留三种直观模式：``random`` 随机连续片段，``rs_transition`` 查看同一次
RS 变化前后，``ue_transition`` 查看同一次 UE 的进入、持续和退出。不传模型开关时
只生成静态 prompt/target 合同。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_v5.eval import _generate_next, _generate_start, load_eval_bundle  # noqa: E402
from qwen3vl_local.sft_v5.metrics import (  # noqa: E402
    build_transition_fields,
    build_transition_report,
    summarize_student_predictions,
)
from qwen3vl_local.sft_v5.labels import option_for_event  # noqa: E402
from qwen3vl_local.sft_v5.prompts import (  # noqa: E402
    SYSTEM_PROMPT_V5,
    build_q1_student_prompt,
    build_q1_teacher_prompt,
    build_q1_teacher_target,
    build_q2_student_prompt,
    build_q2_teacher_prompt,
    build_q2_teacher_target,
    parse_q1_output,
    parse_q2_output,
    update_memory_after_q1,
    update_memory_after_q2,
    update_memory_navigation,
)
from qwen3vl_local.sft_v5.train import (  # noqa: E402
    FrameRow,
    RouteSequenceDataset,
    SequenceRow,
    _event_target_from_frame,
    _load_images,
    _reset_memory_for_frame_row,
    _rs_target_from_frame,
)


PROBE_SAMPLE_MODE_DESCRIPTIONS = {
    "random": "从 validation 中按固定随机种子抽取连续短片段",
    "rs_transition": "检查同一次 RS 变化的变化前帧、新 RS 首帧和变化后帧",
    "ue_transition": "检查同一 UE 片段的进入前 RE、UE 内部和退出后 RE",
}

PROBE_REASON_DESCRIPTIONS = {
    "random": "随机连续片段中的帧",
    "rs_before_transition": "RS 变化前的旧 RS 帧",
    "rs_transition": "RS 变化后的新 RS 首帧",
    "rs_after_transition": "RS 变化后的新 RS 邻帧",
    "ue_before_entry": "进入 UE 之前的 RE 帧",
    "ue_entry": "从 RE 变为 UE 的首帧",
    "ue_inside": "仍处于 UE 片段内的帧",
    "ue_last_frame": "退出 UE 之前的最后一个 UE 帧",
    "ue_exit": "从 UE 退出后的第一个 RE 帧",
    "ue_after_exit": "退出 UE 后继续保持 RE 的邻帧",
}


def _safe_name(text: str, *, max_len: int = 96) -> str:
    """把 scenario / route id 压成目录安全名称。"""

    # route_id 里通常已经包含 Town/route/time 等有用信息；这里只替换文件系统不友好
    # 的字符，不做哈希，方便人眼从目录名直接定位原始 route。
    keep = []
    for ch in str(text):
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    out = "".join(keep).strip("_")
    return (out[:max_len] or "unknown")


def _memory_json(memory: Any) -> Dict[str, Any]:
    """把 Memory dataclass 转为 v3 probe 风格的可读 JSON。"""

    return {
        "rs_label": getattr(memory, "rs_label", None),
        "rs_option": getattr(memory, "rs_option", None),
        "event_label": getattr(memory, "event_label", None),
        "ego_to_goal_x": getattr(memory, "ego_to_goal_x", None),
        "ego_to_goal_y": getattr(memory, "ego_to_goal_y", None),
    }


def _compare_memory_states(
    student: Optional[Dict[str, Any]],
    reference: Dict[str, Any],
    *,
    accepted_event_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """并排比较 student/reference memory，不把 reference 写回 student。

    Q2 双标签场景使用 ``accepted_event_labels`` 判断 EVENT 是否正确；展示的 reference
    仍保留 canonical 单标签，既能审计默认 GT，也不会把另一个合法 UE 误判为错误。
    """

    if student is None:
        return {
            "student": None,
            "reference": reference,
            "rs_matches": None,
            "event_matches": None,
            "goal_matches": None,
            "discrete_state_matches": None,
        }
    accepted = set(str(item) for item in (accepted_event_labels or ()))
    rs_matches = student.get("rs_label") == reference.get("rs_label")
    event_matches = (
        student.get("event_label") in accepted
        if accepted
        else student.get("event_label") == reference.get("event_label")
    )
    goal_matches = (
        student.get("ego_to_goal_x") == reference.get("ego_to_goal_x")
        and student.get("ego_to_goal_y") == reference.get("ego_to_goal_y")
    )
    return {
        "student": student,
        "reference": reference,
        "rs_matches": rs_matches,
        "event_matches": event_matches,
        "goal_matches": goal_matches,
        "discrete_state_matches": bool(rs_matches and event_matches),
    }


def _messages_json(copied_rgb: List[Dict[str, str]], user_prompt: str) -> List[Dict[str, Any]]:
    """用可序列化形式展示真正送给 Qwen 的 system/user messages。"""

    content: List[Dict[str, str]] = []
    for item in copied_rgb:
        content.append(
            {
                "type": "image",
                "file": item.get("file", ""),
                "source": item.get("source", ""),
            }
        )
    content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": SYSTEM_PROMPT_V5},
        {"role": "user", "content": content},
    ]


def _write_messages(frame_dir: pathlib.Path, name: str, copied_rgb: List[Dict[str, str]], user_prompt: str) -> None:
    """写出 v3 风格的 system/user 分离会话视图。"""

    (frame_dir / f"{name}_system_prompt.txt").write_text(SYSTEM_PROMPT_V5, encoding="utf-8")
    (frame_dir / f"{name}_user_prompt.txt").write_text(user_prompt or "", encoding="utf-8")
    (frame_dir / f"{name}_messages.json").write_text(
        json.dumps(_messages_json(copied_rgb, user_prompt or ""), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _copy_rgb_inputs(frame: FrameRow, frame_dir: pathlib.Path) -> List[Dict[str, str]]:
    """复制模型实际读取的 RGB history 到逐帧 probe 目录。

    文件名前缀固定为 ``input_rgb``，明确表示这些 JPEG 是该帧 Q1
    student/teacher 共用的视觉输入，而不是模型输出或渲染图。
    """

    copied: List[Dict[str, str]] = []
    for idx, src_text in enumerate(frame.history_rgb_paths):
        src = pathlib.Path(src_text)
        dst = frame_dir / f"input_rgb_{idx:02d}.jpg"
        record = {"index": str(idx), "source": str(src), "file": str(dst.name)}
        try:
            if src.exists():
                # 保留原始 JPEG 字节，不重新编码；probe 是证据归档，不应改变视觉输入。
                shutil.copy2(src, dst)
                record["copied"] = "true"
            else:
                record["copied"] = "false"
        except Exception as exc:
            record["copied"] = "false"
            record["error"] = str(exc)
        copied.append(record)
    return copied


def _write_texts(frame_dir: pathlib.Path, files: Dict[str, str]) -> None:
    """写出文本文件。"""

    for name, text in files.items():
        (frame_dir / name).write_text(text or "", encoding="utf-8")


def _write_timeline_png(path: pathlib.Path, frame_logs: List[Dict[str, Any]]) -> None:
    """写轻量时间线图，仿 v3：红色=RS 错/reset，蓝色=进入 Q2，灰色=普通帧。"""

    try:
        from PIL import Image, ImageDraw
    except Exception:
        # timeline.png 是辅助可视化；远端缺 PIL 时不应阻断 prompt/JSON dump。
        return
    width = max(320, 16 * max(len(frame_logs), 1))
    height = 76
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = height // 2
    draw.line((12, y, width - 12, y), fill=(80, 80, 80), width=2)
    n = max(len(frame_logs) - 1, 1)
    for i, log in enumerate(frame_logs):
        x = 12 + int((width - 24) * i / n)
        if log.get("q1_rs_correct") is False:
            color = (210, 40, 40)
            r = 5
        elif log.get("q2_triggered"):
            color = (45, 105, 210)
            r = 4
        elif log.get("teacher_forced"):
            color = (80, 160, 80)
            r = 4
        else:
            color = (150, 150, 150)
            r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    img.save(path)


def _frame_labels(route: SequenceRow, frame: FrameRow) -> Dict[str, Any]:
    """汇总单帧标签与候选池，供 labels/flags/manifest 复用。"""

    return {
        "scenario": route.scenario,
        "route_id": route.route_id,
        "frame_id": frame.frame_id,
        "history_rgb_paths": frame.history_rgb_paths,
        "rs_label": frame.rs_label,
        "rs_option": frame.rs_option,
        "event_label": frame.event_label,
        "event_code": frame.event_code,
        "abnormal": frame.abnormal,
        "event_option_map": frame.event_option_map,
        "frame_allowed_events_raw": frame.raw.get("frame_allowed_events_raw", []),
        "regular_event_codes": frame.raw.get("regular_event_codes", []),
        "event_candidate_codes": frame.raw.get("event_candidate_codes", []),
        "weather_text_teacher_only": frame.weather_text,
        "ego_to_goal_xy": list(frame.ego_to_goal_xy) if frame.ego_to_goal_xy is not None else None,
        "review_required": frame.raw.get("review_required", False),
        "source": frame.raw.get("source", {}),
    }


@dataclass(frozen=True)
class ProbeSelection:
    """一条被选中的小样本 probe 帧及其可审计原因。"""

    route_index: int
    frame_index: int
    scenario: str
    route_id: str
    frame_id: int
    primary_reason: str
    reasons: Tuple[str, ...]


def _probe_candidate_reasons(
    routes: Sequence[SequenceRow],
    *,
    context_radius: int,
) -> Dict[Tuple[int, int], Tuple[str, ...]]:
    """为所有帧标记 UE/RS 边界、邻帧和稳定 RE 对照类别。

    `ue_nearby_re` 特意只标真实 RE 帧，它是检查假阳性的硬负例；UE span 内的帧由
    `ue_positive` 覆盖。`rs_nearby` 则保留变换点前后的视觉上下文。
    """

    radius = max(0, int(context_radius))
    reasons: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    for route_idx, route in enumerate(routes):
        frames = route.frames
        abnormal_indices = [idx for idx, frame in enumerate(frames) if bool(frame.abnormal)]
        for frame_idx, frame in enumerate(frames):
            key = (route_idx, frame_idx)
            if frame.abnormal:
                reasons[key].add("ue_positive")
            abnormal_changed = frame_idx > 0 and bool(frames[frame_idx - 1].abnormal) != bool(frame.abnormal)
            if abnormal_changed or (frame_idx == 0 and frame.abnormal):
                reasons[key].add("ue_boundary")
            rs_changed = frame_idx > 0 and frames[frame_idx - 1].rs_label != frame.rs_label
            if rs_changed:
                reasons[key].add("rs_transition")

        # UE 周围的 RE 是最有价值的假阳性检查样本；围绕所有 UE 帧扩展 radius，长
        # UE span 的内部不会误标为 hard negative。
        for abnormal_idx in abnormal_indices:
            lo = max(0, abnormal_idx - radius)
            hi = min(len(frames), abnormal_idx + radius + 1)
            for frame_idx in range(lo, hi):
                if not frames[frame_idx].abnormal:
                    reasons[(route_idx, frame_idx)].add("ue_nearby_re")

        transition_indices = [
            idx
            for idx in range(1, len(frames))
            if frames[idx - 1].rs_label != frames[idx].rs_label
        ]
        for transition_idx in transition_indices:
            lo = max(0, transition_idx - radius)
            hi = min(len(frames), transition_idx + radius + 1)
            for frame_idx in range(lo, hi):
                if frame_idx != transition_idx:
                    reasons[(route_idx, frame_idx)].add("rs_nearby")

        for frame_idx, frame in enumerate(frames):
            key = (route_idx, frame_idx)
            if not frame.abnormal and not reasons.get(key):
                reasons[key].add("stable_re")
            # 额外保留 RS 类别作为审计 reason，不参与三种公开选帧模式的凑数回退。
            reasons[key].add(f"rs_{frame.rs_label.lower()}")
    return {key: tuple(sorted(value)) for key, value in reasons.items()}


def _route_round_robin_windows(
    windows: Sequence[Tuple[int, int, int]],
) -> List[Tuple[int, int, int]]:
    """按 route 轮询排列时间窗口，避免单条长 route 占满专项样本。

    输入三元组为 ``(route_index, start, end)``。同一 route 内仍按时间顺序，
    route 之间每轮各取一个窗口，让小样本尽量覆盖不同场景。
    """

    by_route: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for window in sorted(windows):
        by_route[int(window[0])].append(window)
    ordered: List[Tuple[int, int, int]] = []
    depth = 0
    while True:
        added = False
        for route_idx in sorted(by_route):
            route_windows = by_route[route_idx]
            if depth < len(route_windows):
                ordered.append(route_windows[depth])
                added = True
        if not added:
            return ordered
        depth += 1


def _rs_transition_windows(routes: Sequence[SequenceRow]) -> List[Tuple[int, int, int]]:
    """列出 RS 变化点，``start=end`` 均指向新 RS 首帧。"""

    windows = []
    for route_idx, route in enumerate(routes):
        for frame_idx in range(1, len(route.frames)):
            if route.frames[frame_idx - 1].rs_label != route.frames[frame_idx].rs_label:
                windows.append((route_idx, frame_idx, frame_idx))
    return _route_round_robin_windows(windows)


def _ue_transition_windows(routes: Sequence[SequenceRow]) -> List[Tuple[int, int, int]]:
    """列出连续 UE span，优先返回同时具有进入前 RE 和退出后 RE 的完整片段。"""

    complete: List[Tuple[int, int, int]] = []
    partial: List[Tuple[int, int, int]] = []
    for route_idx, route in enumerate(routes):
        start: Optional[int] = None
        for frame_idx in range(len(route.frames) + 1):
            abnormal = frame_idx < len(route.frames) and bool(route.frames[frame_idx].abnormal)
            if abnormal and start is None:
                start = frame_idx
            if not abnormal and start is not None:
                end = frame_idx - 1
                window = (route_idx, start, end)
                # 完整 span 可以同时审计 RE->UE 与 UE->RE，因此排在 route
                # 起始就是 UE 或 route 结束仍是 UE 的不完整 span 之前。
                target = complete if start > 0 and end + 1 < len(route.frames) else partial
                target.append(window)
                start = None
    return _route_round_robin_windows(complete) + _route_round_robin_windows(partial)


def build_probe_selection_plan(
    routes: Sequence[SequenceRow],
    *,
    num_cases: int,
    sample_mode: str,
    context_radius: int,
    seed: int,
    sequence_length: int = 24,
) -> List[ProbeSelection]:
    """按三种公开语义构造小样本计划。

    ``random`` 用 seed 随机抽取连续短片段；``rs_transition`` 保留同一 RS 变化点的
    前/当前/后帧；``ue_transition`` 保留同一 UE span 的进入前 RE、UE 内部和
    退出后 RE。UE 模式不会用 ``num_cases`` 截断一个真实 UE span：它保留整段 UE，
    再向前后补 ``context_radius`` 帧，因此长 UE 的实际返回帧数可以超过预算。
    专项模式找不到真实变化时返回空结果，不用普通帧冒充专项样本。
    """

    limit = max(0, int(num_cases))
    if limit == 0:
        return []
    reasons_by_key = _probe_candidate_reasons(routes, context_radius=context_radius)
    mode = str(sample_mode or "random").lower()
    if mode not in {"random", "rs_transition", "ue_transition"}:
        raise ValueError(
            f"unsupported probe sample mode: {sample_mode}; "
            "expected random/rs_transition/ue_transition"
        )
    selected: List[Tuple[Tuple[int, int], str]] = []
    selected_keys: set[Tuple[int, int]] = set()

    def add_frame(
        key: Tuple[int, int],
        reason: str,
        *,
        respect_limit: bool = True,
    ) -> None:
        """去重追加帧；UE 完整 span 可显式绕过普通总帧上限。"""

        if (respect_limit and len(selected) >= limit) or key in selected_keys:
            return
        selected.append((key, reason))
        selected_keys.add(key)

    if mode == "random":
        # random 不是把全数据集帧打散后各抽一帧，而是随机选择 route/start，再连续
        # 推理若干帧。这样 Q1 错误后的 memory 漂移/自主恢复、RS 切换及 UE 进入/退出
        # 都能按真实时间顺序被观察。num_cases 仍是总帧预算，避免自动 checkpoint probe
        # 因语义变化突然扩大推理成本。
        rng = random.Random(int(seed))
        clip_target = max(1, int(sequence_length))
        while len(selected) < limit:
            remaining = limit - len(selected)
            desired = min(clip_target, remaining)
            windows: List[Tuple[int, int, int]] = []

            # 优先寻找完整 desired 长度且与已选帧不重叠的窗口。若所有 route 都太短
            # 或已被占用，再逐级缩短窗口，保证小数据 smoke 仍能尽量填满总预算。
            while desired > 0 and not windows:
                for route_idx, route in enumerate(routes):
                    route_len = len(route.frames)
                    for start in range(0, route_len - desired + 1):
                        keys = [(route_idx, idx) for idx in range(start, start + desired)]
                        if all(key not in selected_keys for key in keys):
                            windows.append((route_idx, start, desired))
                desired -= 1 if not windows else 0
            if not windows:
                break
            route_idx, start, window_len = rng.choice(windows)
            for frame_idx in range(start, start + window_len):
                add_frame((route_idx, frame_idx), "random")
    elif mode == "rs_transition":
        radius = max(1, int(context_radius))
        for route_idx, transition_idx, _end in _rs_transition_windows(routes):
            route_len = len(routes[route_idx].frames)
            roles: Dict[int, str] = {}
            for frame_idx in range(
                max(0, transition_idx - radius),
                min(route_len, transition_idx + radius + 1),
            ):
                roles[frame_idx] = (
                    "rs_before_transition"
                    if frame_idx < transition_idx
                    else "rs_transition"
                    if frame_idx == transition_idx
                    else "rs_after_transition"
                )
            # 配额很小时也必须同时看到变化两侧；多余配额再向外扩邻帧。
            priority = [transition_idx - 1, transition_idx, transition_idx + 1]
            priority.extend(sorted(roles, key=lambda idx: (abs(idx - transition_idx), idx)))
            for frame_idx in priority:
                if frame_idx in roles:
                    add_frame((route_idx, frame_idx), roles[frame_idx])
            if len(selected) >= limit:
                break
    else:  # ue_transition
        radius = max(1, int(context_radius))
        for route_idx, start, end in _ue_transition_windows(routes):
            route_len = len(routes[route_idx].frames)
            roles: Dict[int, str] = {}
            for frame_idx in range(max(0, start - radius), start):
                roles[frame_idx] = "ue_before_entry"
            # UE 是持续状态，不是单个边界点。完整保留 start..end 才能评估进入、
            # 持续和退出是否都稳定；不能按固定 num_cases 截掉 span 中后段。
            for frame_idx in range(start, end + 1):
                roles[frame_idx] = (
                    "ue_entry"
                    if frame_idx == start
                    else "ue_last_frame"
                    if frame_idx == end
                    else "ue_inside"
                )
            for frame_idx in range(end + 1, min(route_len, end + radius + 1)):
                roles[frame_idx] = "ue_exit" if frame_idx == end + 1 else "ue_after_exit"

            # 按时间顺序加入整个窗口。UE 模式把 num_cases 当作“至少希望检查多少帧”
            # 的预算，不允许它截断当前 span；完成一个完整窗口后再判断是否需要下一段。
            for frame_idx in sorted(roles):
                add_frame(
                    (route_idx, frame_idx),
                    roles[frame_idx],
                    respect_limit=False,
                )
            if len(selected) >= limit:
                break

    # 专项模式的选择阶段会优先保住边界核心帧；落盘前恢复 route 内
    # 时间顺序，使 selection_plan 和 timeline 可以直接从前往后阅读。
    # 所有模式都按 route/frame 恢复时间顺序。random 的随机性体现在 route/start
    # 选择，而不是落盘顺序；顺序化后 memory 才能按连续帧自然推进。
    selected.sort(key=lambda item: item[0])

    plan: List[ProbeSelection] = []
    final_selected = selected if mode == "ue_transition" else selected[:limit]
    for (route_idx, frame_idx), primary_reason in final_selected:
        route = routes[route_idx]
        frame = route.frames[frame_idx]
        plan.append(
            ProbeSelection(
                route_index=route_idx,
                frame_index=frame_idx,
                scenario=route.scenario,
                route_id=route.route_id,
                frame_id=frame.frame_id,
                primary_reason=primary_reason,
                reasons=tuple(
                    sorted(set(reasons_by_key.get((route_idx, frame_idx), ())) | {primary_reason})
                ),
            )
        )
    return plan


@contextmanager
def _probe_inference_context(bundle: Any, *, disable_adapter: bool) -> Any:
    """把外部训练 bundle 临时切到稳定推理态，并按需关闭 LoRA。

    CLI 独立 probe 加载的模型本来就是 eval 模式；自动 checkpoint probe 则复用正在
    训练的 rank0 PEFT 模型。后者必须在生成期间关闭 dropout，并在退出后恢复 train
    模式。base/teacher 对照还要进入 ``disable_adapter()``，从同一份 Qwen 权重得到
    纯 base 输出，避免为了对照再加载第二份 4B 模型而触发显存峰值。
    """

    import torch

    model = bundle.model
    raw_model = bundle.unwrap() if hasattr(bundle, "unwrap") else model
    was_training = bool(model.training)
    model.eval()
    adapter_context = (
        raw_model.disable_adapter()
        if disable_adapter and hasattr(raw_model, "disable_adapter")
        else nullcontext()
    )
    try:
        with torch.inference_mode(), adapter_context:
            yield
    finally:
        if was_training:
            model.train()


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    """返回可审计比例；没有有效分母时写 null，而不是伪造 0。"""

    if int(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def summarize_probe(
    frame_logs: List[Dict[str, Any]],
    *,
    student_enabled: bool,
    teacher_enabled: bool,
    student_adapter_dir: Optional[str],
    student_disable_adapter: bool,
) -> Dict[str, Any]:
    """把逐帧 flags 聚合成 base/checkpoint 可直接比较的 summary。"""

    frames = len(frame_logs)
    student_frames = frames if student_enabled else 0
    q2_logs = [item for item in frame_logs if bool(item.get("q2_triggered"))] if student_enabled else []
    teacher_q1_logs = (
        [item for item in frame_logs if item.get("q1_teacher_rs_correct") is not None]
        if teacher_enabled
        else []
    )
    teacher_q2_logs = [
        item
        for item in frame_logs
        if bool(item.get("q2_teacher_triggered")) and item.get("q2_teacher_event_correct") is not None
    ] if teacher_enabled else []
    summary = {
        "frames": frames,
        "student_enabled": bool(student_enabled),
        "teacher_enabled": bool(teacher_enabled),
        "student_adapter_dir": student_adapter_dir,
        "student_adapter_enabled": bool(student_enabled and not student_disable_adapter and student_adapter_dir),
        "student_base_mode": bool(student_enabled and (student_disable_adapter or not student_adapter_dir)),
        "q1_rs_correct": sum(bool(item.get("q1_rs_correct")) for item in frame_logs) if student_enabled else 0,
        "q1_rs_accuracy": _ratio(sum(bool(item.get("q1_rs_correct")) for item in frame_logs), student_frames),
        "q1_abnormal_correct": sum(bool(item.get("q1_abnormal_correct")) for item in frame_logs) if student_enabled else 0,
        "q1_abnormal_accuracy": _ratio(sum(bool(item.get("q1_abnormal_correct")) for item in frame_logs), student_frames),
        "q2_triggered": len(q2_logs),
        "q2_trigger_rate": _ratio(len(q2_logs), student_frames),
        "q2_event_correct": sum(bool(item.get("q2_event_correct")) for item in q2_logs),
        "q2_event_accuracy_when_triggered": _ratio(
            sum(bool(item.get("q2_event_correct")) for item in q2_logs),
            len(q2_logs),
        ),
        "q2_invalid_output": sum(bool(item.get("q2_invalid_output")) for item in q2_logs),
        "teacher_q1_frames": len(teacher_q1_logs),
        "teacher_q1_rs_accuracy": _ratio(
            sum(bool(item.get("q1_teacher_rs_correct")) for item in teacher_q1_logs),
            len(teacher_q1_logs),
        ),
        "teacher_q1_abnormal_accuracy": _ratio(
            sum(bool(item.get("q1_teacher_abnormal_correct")) for item in teacher_q1_logs),
            len(teacher_q1_logs),
        ),
        "teacher_q2_frames": len(teacher_q2_logs),
        "teacher_q2_trigger_rate": _ratio(len(teacher_q2_logs), len(teacher_q1_logs)),
        "teacher_q2_event_accuracy": _ratio(
            sum(bool(item.get("q2_teacher_event_correct")) for item in teacher_q2_logs),
            len(teacher_q2_logs),
        ),
    }
    if student_enabled:
        # 新版严格指标与 eval.py 共用同一实现；旧 q1_rs_accuracy 等 key 继续保留，
        # 现有 checkpoint comparison 不会因为 schema 扩展而失效。
        summary.update(summarize_student_predictions(frame_logs))
    return summary


def build_memory_recovery_report(frame_logs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """统计 RS/EVENT 真值变化后 student memory 的自主恢复延迟。

    每个变化点只在当前连续选帧窗口内向后搜索，并在下一次同类 GT 变化前停止。延迟 0
    表示学生在变化首帧立即改对；``recovered=false`` 表示直到观察窗口结束仍未与
    reference 对齐。该函数只读 probe 日志，不参与任何 memory 更新。
    """

    logs = list(frame_logs)

    def collect(kind: str) -> List[Dict[str, Any]]:
        """收集一种变化的逐 case 恢复结果。"""

        cases: List[Dict[str, Any]] = []
        for start, row in enumerate(logs):
            if bool(row.get("selection_gap_reset")):
                # 窗口首帧没有真实 student 前序，不能测“从旧 memory 自主切换”的延迟。
                continue
            if kind == "rs":
                is_transition = bool(row.get("rs_transition"))
                match_key = "memory_rs_matches_after_q1"
                target = row.get("gt_rs_label")
                boundary_key = "rs_transition"
                transition_name = "rs_change"
            else:
                is_transition = bool(row.get("abnormal_transition"))
                match_key = "memory_event_matches_after_q2"
                target = row.get("gt_event_label")
                boundary_key = "abnormal_transition"
                transition_name = "ue_entry" if bool(row.get("gt_abnormal")) else "ue_exit"
            if not is_transition:
                continue

            recovered_at: Optional[int] = None
            end = start
            for cursor in range(start, len(logs)):
                candidate = logs[cursor]
                if cursor > start and (
                    bool(candidate.get("selection_gap_reset"))
                    or candidate.get("route_id") != row.get("route_id")
                    or bool(candidate.get(boundary_key))
                ):
                    break
                end = cursor
                if candidate.get(match_key) is True:
                    recovered_at = cursor
                    break
            cases.append(
                {
                    "transition": transition_name,
                    "scenario": row.get("scenario"),
                    "route_id": row.get("route_id"),
                    "transition_frame_id": row.get("frame_id"),
                    "target": target,
                    "recovered": recovered_at is not None,
                    "recovery_delay_frames": (
                        recovered_at - start if recovered_at is not None else None
                    ),
                    "recovered_frame_id": (
                        logs[recovered_at].get("frame_id") if recovered_at is not None else None
                    ),
                    "observed_frames": end - start + 1,
                    "last_observed_frame_id": logs[end].get("frame_id"),
                }
            )
        return cases

    rs_cases = collect("rs")
    event_cases = collect("event")
    all_cases = rs_cases + event_cases
    recovered_delays = [
        int(case["recovery_delay_frames"])
        for case in all_cases
        if case.get("recovery_delay_frames") is not None
    ]
    return {
        "meaning": (
            "GT 变化后 student memory 首次自行与 reference 对齐的延迟；reference 只用于比较，"
            "没有回写 student。"
        ),
        "rs_change_cases": rs_cases,
        "event_change_cases": event_cases,
        "summary": {
            "transition_cases": len(all_cases),
            "recovered_cases": sum(bool(case.get("recovered")) for case in all_cases),
            "not_recovered_cases": sum(not bool(case.get("recovered")) for case in all_cases),
            "mean_recovery_delay_frames": (
                sum(recovered_delays) / len(recovered_delays) if recovered_delays else None
            ),
            "max_recovery_delay_frames": max(recovered_delays) if recovered_delays else None,
        },
    }


def dump_probe(
    args: argparse.Namespace,
    *,
    student_bundle: Optional[Any] = None,
    teacher_bundle: Optional[Any] = None,
    student_disable_adapter: bool = False,
    teacher_disable_adapter: bool = False,
) -> Dict[str, Any]:
    """运行连续小样本 probe，并返回逐版本对比所需的汇总指标。

    ``student_bundle`` / ``teacher_bundle`` 仅供训练进程内自动 probe 使用。传入后不会
    重新加载 Qwen；checkpoint student 使用当前 LoRA，base 与 teacher 则在同一模型上
    临时关闭 adapter。默认 ``review`` 按 scenario/frame 分目录，每帧只写 RGB、
    input/output/memory 三个 JSON；``compact`` 只写汇总；``full`` 再展开旧式
    TXT/JSON/timeline 深度审计文件。
    """

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = RouteSequenceDataset(
        pathlib.Path(args.index),
        max_routes=int(args.max_routes),
        max_frames_per_route=int(args.max_frames_per_route),
    )
    sample_mode = str(getattr(args, "sample_mode", "random"))
    context_radius = int(getattr(args, "context_radius", 8))
    sample_seed = int(getattr(args, "seed", 20260711))
    sequence_length = max(1, int(getattr(args, "sequence_length", 24)))
    artifact_level = str(getattr(args, "artifact_level", "review")).lower()
    if artifact_level not in {"compact", "review", "full"}:
        raise ValueError(
            f"unsupported artifact level: {artifact_level}; expected compact/review/full"
        )
    review_artifacts = artifact_level in {"review", "full"}
    full_artifacts = artifact_level == "full"
    selection_plan = build_probe_selection_plan(
        ds.rows,
        num_cases=int(args.num_cases),
        sample_mode=sample_mode,
        context_radius=context_radius,
        seed=sample_seed,
        sequence_length=sequence_length,
    )
    selection_by_key = {
        (item.route_index, item.frame_index): item
        for item in selection_plan
    }
    selection_category_counts = Counter(item.primary_reason for item in selection_plan)
    selection_payload = {
        "sample_mode": sample_mode,
        "sample_mode_description": PROBE_SAMPLE_MODE_DESCRIPTIONS[sample_mode],
        "sequence_length": sequence_length,
        "context_radius": context_radius,
        "seed": sample_seed,
        "requested_cases": int(args.num_cases),
        "selected_cases": len(selection_plan),
        "primary_reason_counts": dict(sorted(selection_category_counts.items())),
        "cases": [
            {
                "route_index": item.route_index,
                "frame_index": item.frame_index,
                "scenario": item.scenario,
                "route_id": item.route_id,
                "frame_id": item.frame_id,
                "primary_reason": item.primary_reason,
                "primary_reason_description": PROBE_REASON_DESCRIPTIONS.get(
                    item.primary_reason, item.primary_reason
                ),
                "reasons": list(item.reasons),
            }
            for item in selection_plan
        ],
    }
    if full_artifacts:
        (out_dir / "selection_plan.json").write_text(
            json.dumps(selection_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    bundle = student_bundle
    if args.with_model and bundle is None:
        import torch

        # `--with-model` 负责生成 student 输出；只有显式传 `--adapter-dir` 时才加载
        # student LoRA。训练前 OPSD 能力体检不要传 adapter，保持纯 base Qwen。
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bundle = load_eval_bundle(
            pathlib.Path(args.model_dir),
            pathlib.Path(args.adapter_dir) if args.adapter_dir else None,
            device,
            merge_lora=bool(args.merge_lora),
        )
    if args.with_teacher_model and teacher_bundle is None:
        import torch

        # 训练前能力体检时，teacher 也用默认/base Qwen，但吃 privileged prompt。
        # 这里不加载 adapter，避免把待训练学生能力混入 teacher 侧判断。
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        teacher_bundle = load_eval_bundle(
            pathlib.Path(args.teacher_model_dir or args.model_dir),
            None,
            device,
            merge_lora=True,
        )

    manifest: List[Dict[str, Any]] = []
    all_frame_logs: List[Dict[str, Any]] = []
    compact_frames: List[Dict[str, Any]] = []
    case_idx = 0
    for route_idx, route in enumerate(ds.rows):
        selected_frame_indices = {
            frame_idx
            for selected_route_idx, frame_idx in selection_by_key
            if selected_route_idx == route_idx
        }
        if not selected_frame_indices:
            continue
        # 人工测试先按 scenario/route 归组，下面直接是连续 frame。route id 合进场景
        # 目录名，既保持用户期望的“场景 -> 帧”，又避免同名 scenario 多条 route 冲突。
        route_dir = (
            out_dir
            / "scenarios"
            / f"{_safe_name(route.scenario)}__{_safe_name(route.route_id)}"
        )
        if review_artifacts:
            route_dir.mkdir(parents=True, exist_ok=True)
        frame_logs: List[Dict[str, Any]] = []
        memory = None
        reference_memory = None
        previous_selected_frame_index: Optional[int] = None
        previous_pred_rs_label: Optional[str] = None
        previous_pred_abnormal: Optional[bool] = None
        for frame_index, frame in enumerate(route.frames):
            selection = selection_by_key.get((route_idx, frame_index))
            if selection is None:
                continue
            selection_gap_reset = bool(
                previous_selected_frame_index is None
                or frame_index != previous_selected_frame_index + 1
            )
            if selection_gap_reset:
                # 定向采样可能从一条 route 取多个不连续窗口。每个窗口首帧分别初始化
                # student/reference；窗口内部 student 只由模型输出推进，reference 只由 GT
                # 推演，二者永不互相覆盖。
                memory = None
                reference_memory = None
                previous_pred_rs_label = None
                previous_pred_abnormal = None
            previous_selected_frame_index = frame_index
            rs_target = _rs_target_from_frame(frame)
            event_target = _event_target_from_frame(frame)
            memory_initialized_from_gt = memory is None
            if memory is None:
                # 连续窗口没有历史输出，只能用当前 GT RS + RE 建立共同起点；该初始化
                # 会写入审计字段，且只发生一次，不会在学生答错后再次触发。
                memory = _reset_memory_for_frame_row(frame)
                reference_memory = _reset_memory_for_frame_row(frame)
            else:
                # 导航坐标是逐帧外部输入，不属于标签纠错；RS/EVENT 保持学生上一帧结果。
                memory = update_memory_navigation(memory, frame.ego_to_goal_xy)
                assert reference_memory is not None
                reference_memory = update_memory_navigation(
                    reference_memory,
                    frame.ego_to_goal_xy,
                )
            assert reference_memory is not None
            memory_at_frame_start = memory
            memory_before = _memory_json(memory)
            reference_memory_before = _memory_json(reference_memory)
            case_dir = route_dir / f"frame_{frame.frame_id:04d}"
            if review_artifacts:
                case_dir.mkdir(parents=True, exist_ok=True)
            if review_artifacts:
                copied_rgb = _copy_rgb_inputs(frame, case_dir)
            else:
                # compact 不创建逐帧目录，只在 results.json 中保留原始图像路径。
                copied_rgb = [
                    {"index": str(idx), "source": str(path), "file": str(path)}
                    for idx, path in enumerate(frame.history_rgb_paths)
                ]
            q1_student = build_q1_student_prompt(memory)
            # teacher prompt 是 privileged 输入：可看 XML weather、GT RS/ABNORMAL、
            # 原始 event_code；target 则被清洗成学生视角，用于人工检查不泄漏私有字段。
            q1_teacher = build_q1_teacher_prompt(
                memory,
                rs_target=rs_target,
                event_target=event_target,
                weather_text=frame.weather_text,
            )
            q1_target = build_q1_teacher_target(
                rs_target=rs_target,
                event_target=event_target,
                weather_text=frame.weather_text,
            )
            q1_output: Optional[str] = None
            q2_output: Optional[str] = None
            q1_teacher_output: Optional[str] = None
            q2_teacher_output: Optional[str] = None
            q1_after: Optional[Any] = None
            q1_teacher_after: Optional[Any] = None
            q2_student = ""
            q2_teacher = ""
            q2_target = ""
            q2_teacher_model_prompt = ""
            q2_teacher_model_target = ""
            q2_student_memory_input: Optional[Dict[str, Any]] = None
            parsed_q1: Dict[str, Optional[str]] = {}
            parsed_q2: Dict[str, Optional[str]] = {}
            parsed_teacher_q1: Dict[str, Optional[str]] = {}
            parsed_teacher_q2: Dict[str, Optional[str]] = {}
            q1_rs_ok = True
            q1_abnormal: Optional[bool] = frame.abnormal
            q1_teacher_rs_correct: Optional[bool] = None
            q1_teacher_abnormal_correct: Optional[bool] = None
            q2_triggered = True
            q2_invalid = False
            q2_event_correct: Optional[bool] = None
            q2_teacher_event_correct: Optional[bool] = None
            q2_teacher_triggered = False
            q2_candidate_mismatch = False
            images: Optional[List[Any]] = None

            def _images_for_generation() -> List[Any]:
                """懒加载该帧 RGB，避免纯静态 probe 读图进入模型路径。"""

                nonlocal images
                if images is None:
                    images = _load_images(frame.history_rgb_paths)
                return images

            # reference 分支只做离线对比，不进入任何 student/teacher prompt。Q1 reference
            # 按 GT RS/ABNORMAL 推演；异常为 YES 时，具体 EVENT 要等 reference Q2 才更新。
            reference_memory_after_q1_state = update_memory_after_q1(
                reference_memory,
                student_rs_label=frame.rs_label,
                student_abnormal=frame.abnormal,
            )
            reference_memory_after_q1 = _memory_json(reference_memory_after_q1_state)
            memory_after_q1 = update_memory_after_q1(
                memory,
                student_rs_label=frame.rs_label,
                student_abnormal=frame.abnormal,
            )
            if teacher_bundle is not None:
                # 训练前体检用：teacher_bundle 永远是纯 base Qwen，不加载 LoRA。
                # 它吃 privileged prompt，用来判断“普通 Qwen 当老师”是否能稳定解析/解释。
                with _probe_inference_context(teacher_bundle, disable_adapter=teacher_disable_adapter):
                    q1_teacher_output, q1_teacher_after = _generate_start(
                        teacher_bundle,
                        _images_for_generation(),
                        q1_teacher,
                        int(args.max_new_tokens_q1),
                    )
                parsed_teacher_q1 = parse_q1_output(q1_teacher_output)
                q1_teacher_rs_correct = parsed_teacher_q1.get("rs_label") == frame.rs_label
                teacher_abnormal_text = parsed_teacher_q1.get("abnormal")
                q1_teacher_abnormal_correct = (
                    (teacher_abnormal_text == "YES") == frame.abnormal if teacher_abnormal_text else False
                )
            if bundle is not None:
                # student bundle 可以是纯 base Qwen（训练前体检）或 base+adapter（训练后可视化）。
                # 是否误传 adapter 会写入 flags.json 的 student_adapter_dir 供人工审计。
                with _probe_inference_context(bundle, disable_adapter=student_disable_adapter):
                    q1_output, q1_after = _generate_start(
                        bundle,
                        _images_for_generation(),
                        q1_student,
                        int(args.max_new_tokens_q1),
                    )
                parsed_q1 = parse_q1_output(q1_output)
                q1_rs_ok = parsed_q1.get("rs_label") == frame.rs_label
                q1_abnormal = parsed_q1.get("abnormal") == "YES" if parsed_q1.get("abnormal") else None
                memory_after_q1 = update_memory_after_q1(memory, student_rs_label=parsed_q1.get("rs_label"), student_abnormal=q1_abnormal)
                if q1_rs_ok:
                    # 只有 Q1 的 RS 正确才进入 Q2；这和训练时的采样/截断规则保持一致。
                    q2_student_memory_input = _memory_json(memory_after_q1)
                    q2_student = build_q2_student_prompt(
                        memory_after_q1,
                        option_map=frame.event_option_map,
                        q1_abnormal=bool(q1_abnormal),
                        regular_event_codes=frame.regular_event_codes,
                    )
                    q2_teacher = build_q2_teacher_prompt(
                        memory_after_q1,
                        option_map=frame.event_option_map,
                        q1_abnormal=bool(q1_abnormal),
                        event_target=event_target,
                        regular_event_codes=frame.regular_event_codes,
                    )
                    q2_target = build_q2_teacher_target(
                        memory_after_q1,
                        option_map=frame.event_option_map,
                        event_target=event_target,
                        regular_event_codes=frame.regular_event_codes,
                    )
                    if q1_after is not None:
                        with _probe_inference_context(bundle, disable_adapter=student_disable_adapter):
                            q2_output, q2_after = _generate_next(
                                bundle,
                                q1_after,
                                q2_student,
                                int(args.max_new_tokens_q2),
                            )
                        # probe 只保存文本；若写成 `q2_output, _ = ...`，普通变量 `_`
                        # 会把完整 Q2 KV 持有到下一帧，和下一次 prefill 叠加显存峰值。
                        del q2_after
                    parsed_q2 = parse_q2_output(q2_output, frame.event_option_map)
                    memory = update_memory_after_q2(memory_after_q1, student_event_label=parsed_q2.get("event_label"))
                    q2_invalid = parsed_q2.get("event_label") is None
                    target_dynamic = _event_target_from_frame(frame, student_event=parsed_q2.get("event_label"))
                    # 双标签 EVENT 的正确性必须按“student 选择是否在可接受集合内”动态计算，
                    # 不能只和 build_dataset 固定 event_label 比较。
                    q2_event_correct = parsed_q2.get("event_label") == target_dynamic.label
                    q2_candidate_mismatch = target_dynamic.label not in set(frame.event_option_map.values())
                else:
                    # RS 错误时本帧停止 Q2，但保留学生 Q1 后的 memory。下一帧继续把该
                    # memory 输入学生，才能观察模型是否会自行纠正，而不是脚本替它纠正。
                    q2_triggered = False
                    memory = memory_after_q1
            else:
                # 静态 dump 模式不跑 student 生成；为了仍然能看到完整 Q2 prompt/target，
                # 这里使用 GT Q1 结果推进一次 memory，相当于 teacher-forced 可视化。
                q2_student_memory_input = _memory_json(memory_after_q1)
                q2_student = build_q2_student_prompt(
                    memory_after_q1,
                    option_map=frame.event_option_map,
                    q1_abnormal=frame.abnormal,
                    regular_event_codes=frame.regular_event_codes,
                )
                q2_teacher = build_q2_teacher_prompt(
                    memory_after_q1,
                    option_map=frame.event_option_map,
                    q1_abnormal=frame.abnormal,
                    event_target=event_target,
                    regular_event_codes=frame.regular_event_codes,
                )
                q2_target = build_q2_teacher_target(
                    memory_after_q1,
                    option_map=frame.event_option_map,
                    event_target=event_target,
                    regular_event_codes=frame.regular_event_codes,
                )
                memory = update_memory_after_q2(memory_after_q1, student_event_label=event_target.label)
                parsed_q1 = {"rs_option": frame.rs_option, "rs_label": frame.rs_label, "abnormal": "YES" if frame.abnormal else "NO"}
                parsed_q2 = {"event_option": None, "event_label": event_target.label}
                q2_candidate_mismatch = event_target.label not in set(frame.event_option_map.values())
                q2_event_correct = not q2_candidate_mismatch

            if teacher_bundle is not None:
                # teacher 能力体检必须自洽：Q2 只依赖 teacher 自己生成的 Q1 KV、RS 和
                # ABNORMAL，不能把 student Q1 构造的 memory/prompt 接到 teacher KV 后面。
                # teacher Q1 的 RS 错误时按正式状态机停止 Q2，该帧不进入 teacher Q2
                # accuracy 分母；trigger rate 会单独暴露这种上层失败。
                if q1_teacher_after is not None and bool(q1_teacher_rs_correct):
                    teacher_abnormal_text = parsed_teacher_q1.get("abnormal")
                    teacher_q1_abnormal = (
                        teacher_abnormal_text == "YES" if teacher_abnormal_text else None
                    )
                    teacher_memory_after_q1 = update_memory_after_q1(
                        memory_at_frame_start,
                        student_rs_label=parsed_teacher_q1.get("rs_label"),
                        student_abnormal=teacher_q1_abnormal,
                    )
                    q2_teacher_model_prompt = build_q2_teacher_prompt(
                        teacher_memory_after_q1,
                        option_map=frame.event_option_map,
                        q1_abnormal=bool(teacher_q1_abnormal),
                        event_target=event_target,
                        regular_event_codes=frame.regular_event_codes,
                    )
                    q2_teacher_model_target = build_q2_teacher_target(
                        teacher_memory_after_q1,
                        option_map=frame.event_option_map,
                        event_target=event_target,
                        regular_event_codes=frame.regular_event_codes,
                    )
                    q2_teacher_triggered = True
                    with _probe_inference_context(teacher_bundle, disable_adapter=teacher_disable_adapter):
                        q2_teacher_output, q2_teacher_after = _generate_next(
                            teacher_bundle,
                            q1_teacher_after,
                            q2_teacher_model_prompt,
                            int(args.max_new_tokens_q2),
                        )
                    del q2_teacher_after
                    parsed_teacher_q2 = parse_q2_output(q2_teacher_output, frame.event_option_map)
                    teacher_dynamic_target = _event_target_from_frame(
                        frame,
                        student_event=parsed_teacher_q2.get("event_label"),
                    )
                    q2_teacher_event_correct = (
                        parsed_teacher_q2.get("event_label") == teacher_dynamic_target.label
                    )

            labels = _frame_labels(route, frame)
            # 双 UE 标签允许学生命中任意一个合法 UE；全 regular 标签则统一折成 RE。
            # 因此人工对照既要保存默认单标签，也要保存根据当前 student 输出重新解析的
            # 动态单标签，否则会出现代码判对、JSON 却像是判错的假冲突。
            student_event_label = parsed_q2.get("event_label") if bundle is not None else None
            resolved_student_event_target = _event_target_from_frame(
                frame,
                student_event=student_event_label,
            )
            raw_ue_labels = [
                code for code in resolved_student_event_target.raw_events if str(code).startswith("U-E")
            ]
            accepted_event_labels = list(dict.fromkeys(raw_ue_labels)) if raw_ue_labels else ["RE"]
            student_memory_after_q1 = _memory_json(memory_after_q1)
            student_memory_after_q2 = _memory_json(memory) if q2_triggered else None
            student_memory_for_next_frame = _memory_json(memory)
            reference_memory_after_q2_state = update_memory_after_q2(
                reference_memory_after_q1_state,
                student_event_label=event_target.label,
            )
            reference_memory_after_q2 = _memory_json(reference_memory_after_q2_state)
            # reference_memory 只沿 GT 轨迹向前推进，用于下一帧继续生成“应该是什么”的
            # 对照；student memory 保持上面模型输出的结果，绝不从这里复制 reference。
            reference_memory = reference_memory_after_q2_state
            memory_trace = {
                "policy": (
                    "student_closed_loop" if bundle is not None else "teacher_forced_static_contract"
                ),
                # before/after 保留旧消费方入口；四个问答节点在 q1/q2 下提供完整细节。
                "before": memory_before,
                "after": student_memory_for_next_frame,
                "window_initialized_from_ground_truth": memory_initialized_from_gt,
                "reference_is_comparison_only": True,
                "forced_correction_applied": False,
                "q1": {
                    "input": _compare_memory_states(
                        memory_before,
                        reference_memory_before,
                    ),
                    "after_student_output": _compare_memory_states(
                        student_memory_after_q1,
                        reference_memory_after_q1,
                    ),
                },
                "q2": {
                    "triggered": q2_triggered if bundle is not None else None,
                    "input": _compare_memory_states(
                        q2_student_memory_input if bundle is not None else None,
                        reference_memory_after_q1,
                    ),
                    "after_student_output": _compare_memory_states(
                        student_memory_after_q2 if bundle is not None else None,
                        reference_memory_after_q2,
                        accepted_event_labels=accepted_event_labels,
                    ),
                },
                "next_frame": {
                    "student": student_memory_for_next_frame,
                    "reference": reference_memory_after_q2,
                    "student_was_not_overwritten_by_reference": True,
                },
            }
            q1_after_matches = memory_trace["q1"]["after_student_output"]["rs_matches"]
            q2_after_event_matches = memory_trace["q2"]["after_student_output"]["event_matches"]
            # “输入是否已对齐当前目标”与“是否匹配 reference 输入历史”不是一回事：
            # RS/UE 刚变化时，正确的 Q1/Q2 输入本来仍可能保存旧状态。自主纠正判断应
            # 以前者为起点，观察学生输出后是否转成当前帧目标。
            q1_input_matches_current_target = memory_before.get("rs_label") == frame.rs_label
            q2_input_matches_current_target = bool(
                q2_student_memory_input is not None
                and q2_student_memory_input.get("event_label") in set(accepted_event_labels)
            )
            memory_trace["q1"]["input_matches_current_frame_target"] = (
                q1_input_matches_current_target if bundle is not None else None
            )
            memory_trace["q2"]["input_matches_current_frame_target"] = (
                q2_input_matches_current_target
                if bundle is not None and q2_triggered
                else None
            )
            memory_trace["autonomous_change"] = {
                "q1_rs_corrected_by_student": bool(
                    bundle is not None
                    and not q1_input_matches_current_target
                    and q1_after_matches is True
                ),
                "q1_rs_corrupted_by_student": bool(
                    bundle is not None
                    and q1_input_matches_current_target
                    and q1_after_matches is False
                ),
                "q2_event_corrected_by_student": bool(
                    bundle is not None
                    and q2_triggered
                    and not q2_input_matches_current_target
                    and q2_after_event_matches is True
                ),
                "q2_event_corrupted_by_student": bool(
                    bundle is not None
                    and q2_triggered
                    and q2_input_matches_current_target
                    and q2_after_event_matches is False
                ),
            }
            would_reset_under_training = bool(
                bundle is not None and (not q1_rs_ok or (q2_triggered and q2_invalid))
            )
            ground_truth_structure = {
                "q1": {
                    "rs_option": frame.rs_option,
                    "rs_label": frame.rs_label,
                    "abnormal": "YES" if frame.abnormal else "NO",
                },
                "q2": {
                    "accepted_event_labels": accepted_event_labels,
                    "default_event_option": option_for_event(event_target.label, frame.event_option_map),
                    "default_event_label": event_target.label,
                    "resolved_for_student_option": option_for_event(
                        resolved_student_event_target.label,
                        frame.event_option_map,
                    ),
                    "resolved_for_student_label": resolved_student_event_target.label,
                    "event_code_audit": resolved_student_event_target.event_code,
                },
            }
            memory_after = student_memory_for_next_frame
            if full_artifacts:
                # 默认 teacher input/output 文件必须一一对应。启用 teacher model 时，Q2
                # output 来自 teacher 自己的 Q1 KV；训练 privileged prompt 单独保存。
                q2_teacher_output_prompt = (
                    q2_teacher_model_prompt if teacher_bundle is not None else q2_teacher
                )
                q2_teacher_output_target = (
                    q2_teacher_model_target if teacher_bundle is not None else q2_target
                )
                files = {
                    "q1_system_prompt.txt": SYSTEM_PROMPT_V5,
                    "q2_system_prompt.txt": SYSTEM_PROMPT_V5,
                    "q1_student_prompt.txt": q1_student,
                    "q1_student_user_prompt.txt": q1_student,
                    "q1_teacher_prompt.txt": q1_teacher,
                    "q1_teacher_user_prompt.txt": q1_teacher,
                    "q1_teacher_target.txt": q1_target,
                    "q2_student_prompt.txt": q2_student,
                    "q2_student_user_prompt.txt": q2_student,
                    "q2_teacher_prompt.txt": q2_teacher_output_prompt,
                    "q2_teacher_user_prompt.txt": q2_teacher_output_prompt,
                    "q2_teacher_target.txt": q2_teacher_output_target,
                    "q2_teacher_training_prompt.txt": q2_teacher,
                    "q2_teacher_training_target.txt": q2_target,
                    "q2_teacher_model_prompt.txt": q2_teacher_model_prompt,
                    "q2_teacher_model_target.txt": q2_teacher_model_target,
                    # v3-style aliases 仅在 full 模式保留，供已有可视化脚本直接复用。
                    "step1_user.txt": q1_student,
                    "step1_teacher_user.txt": q1_teacher,
                    "step1_teacher.txt": q1_target,
                    "step2_user.txt": q2_student,
                    "step2_teacher_user.txt": q2_teacher_output_prompt,
                    "step2_teacher.txt": q2_teacher_output_target,
                    "q1_student_output.txt": q1_output or "",
                    "q2_student_output.txt": q2_output or "",
                    "q1_teacher_output.txt": q1_teacher_output or "",
                    "q2_teacher_output.txt": q2_teacher_output or "",
                    "step1_student.txt": q1_output or "",
                    "step2_student.txt": q2_output or "",
                    "step1_teacher_output.txt": q1_teacher_output or "",
                    "step2_teacher_output.txt": q2_teacher_output or "",
                }
                _write_texts(case_dir, files)
                _write_messages(case_dir, "q1_student", copied_rgb, q1_student)
                _write_messages(case_dir, "q1_teacher", copied_rgb, q1_teacher)
                _write_messages(case_dir, "q2_student", copied_rgb, q2_student)
                _write_messages(case_dir, "q2_teacher", copied_rgb, q2_teacher_output_prompt)
                _write_messages(case_dir, "q2_teacher_training", copied_rgb, q2_teacher)
                _write_messages(case_dir, "q2_teacher_model", copied_rgb, q2_teacher_model_prompt)
                (case_dir / "labels.json").write_text(
                    json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (case_dir / "memory_before.json").write_text(
                    json.dumps(memory_before, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (case_dir / "memory_after.json").write_text(
                    json.dumps(memory_after, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            pred_rs_label = parsed_q1.get("rs_label") if bundle is not None else None
            pred_abnormal = q1_abnormal if bundle is not None else None
            transition_fields = build_transition_fields(
                pair_evaluated=bool(bundle is not None and not selection_gap_reset),
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
            frame_log = {
                **labels,
                # flags.json 聚合三类信息：
                # 1) label/source/candidate 证据；
                # 2) student/teacher 解析结果；
                # 3) student/reference 双轨 memory 与“训练协议是否会建议 reset”的诊断。
                "case_index": case_idx,
                "case_dir": str(case_dir) if full_artifacts else None,
                "copied_rgb": copied_rgb if full_artifacts else None,
                "teacher_forced": bundle is None,
                "teacher_model_enabled": teacher_bundle is not None,
                "teacher_model_dir": str(pathlib.Path(args.teacher_model_dir or args.model_dir)) if teacher_bundle is not None else None,
                "student_adapter_dir": str(pathlib.Path(args.adapter_dir)) if args.adapter_dir else None,
                "student_adapter_enabled": bool(bundle is not None and not student_disable_adapter and args.adapter_dir),
                "student_base_mode": bool(bundle is not None and (student_disable_adapter or not args.adapter_dir)),
                "generation_limits": {
                    "max_new_tokens_q1": int(args.max_new_tokens_q1),
                    "max_new_tokens_q2": int(args.max_new_tokens_q2),
                },
                "selection_primary_reason": selection.primary_reason,
                "selection_primary_reason_description": PROBE_REASON_DESCRIPTIONS.get(
                    selection.primary_reason, selection.primary_reason
                ),
                "selection_reasons": list(selection.reasons),
                "selection_gap_reset": selection_gap_reset,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "reference_memory_before": reference_memory_before,
                "reference_memory_after": reference_memory_after_q2,
                "memory_trace": memory_trace,
                "parsed_q1": parsed_q1,
                "parsed_q2": parsed_q2,
                "parsed_teacher_q1": parsed_teacher_q1,
                "parsed_teacher_q2": parsed_teacher_q2,
                "q1_rs_correct": q1_rs_ok,
                "q1_abnormal_correct": q1_abnormal == frame.abnormal if q1_abnormal is not None else False,
                "q1_teacher_rs_correct": q1_teacher_rs_correct,
                "q1_teacher_abnormal_correct": q1_teacher_abnormal_correct,
                "q2_triggered": q2_triggered,
                "q2_event_correct": q2_event_correct,
                "q2_teacher_event_correct": q2_teacher_event_correct,
                "q2_teacher_triggered": q2_teacher_triggered,
                # 兼容旧 flags 消费方；新代码不再把 GT/student prompt 强接到 teacher Q1 KV。
                "q2_teacher_forced": False,
                "q2_student_continued_from_q1_kv": bool(bundle is not None and q2_triggered and q1_after is not None),
                "q2_teacher_continued_from_q1_kv": bool(q2_teacher_triggered and q1_teacher_after is not None),
                "q2_invalid_output": q2_invalid,
                "q2_candidate_mismatch": q2_candidate_mismatch,
                # 测试永不应用 GT 纠错。保留 would_reset 字段，只用于说明若按训练协议
                # 运行该帧是否会在下一帧 reset，不能据此改写 student memory。
                "rs_wrong_reset": False,
                "reset_next": False,
                "would_reset_under_training": would_reset_under_training,
                "memory_forced_correction_applied": False,
                "student_memory_after_q1_rs_label": student_memory_after_q1.get("rs_label"),
                "student_memory_after_q2_event_label": student_memory_for_next_frame.get(
                    "event_label"
                ),
                "memory_rs_matches_after_q1": q1_after_matches if bundle is not None else None,
                "memory_event_matches_after_q2": (
                    q2_after_event_matches if bundle is not None and q2_triggered else None
                ),
                # probe/eval 共用指标字段。transition 必须基于原始 route 相邻帧，而不是
                # 基于定向采样后相邻的 case，避免跳帧制造假边界。
                "gt_rs_label": frame.rs_label,
                "pred_rs_label": pred_rs_label,
                "gt_abnormal": bool(frame.abnormal),
                "pred_abnormal": pred_abnormal,
                "gt_event_label": event_target.label,
                "pred_event_label": parsed_q2.get("event_label") if bundle is not None else event_target.label,
                "pred_event_is_ue": (
                    None
                    if bundle is not None and parsed_q2.get("event_label") is None
                    else bool((parsed_q2.get("event_label") if bundle is not None else event_target.label) != "RE")
                ),
                "rs_transition": bool(
                    frame_index > 0 and route.frames[frame_index - 1].rs_label != frame.rs_label
                ),
                "abnormal_transition": bool(
                    frame_index > 0
                    and bool(route.frames[frame_index - 1].abnormal) != bool(frame.abnormal)
                ),
                **transition_fields,
            }
            if full_artifacts:
                # flags/case_record 是 full 模式的逐帧深度审计入口。
                (case_dir / "flags.json").write_text(
                    json.dumps(frame_log, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                case_record = {
                    "selection": {
                        "sample_mode": sample_mode,
                        "sample_mode_description": PROBE_SAMPLE_MODE_DESCRIPTIONS[sample_mode],
                        "primary_reason": selection.primary_reason,
                        "primary_reason_description": PROBE_REASON_DESCRIPTIONS.get(
                            selection.primary_reason, selection.primary_reason
                        ),
                        "reasons": list(selection.reasons),
                        "gap_reset": selection_gap_reset,
                    },
                    "labels": labels,
                    "inputs": {
                        "rgb": copied_rgb,
                        "q1_student_messages": _messages_json(copied_rgb, q1_student),
                        "q1_teacher_messages": _messages_json(copied_rgb, q1_teacher),
                        "q2_student_messages": _messages_json(copied_rgb, q2_student),
                        "q2_teacher_training_messages": _messages_json(copied_rgb, q2_teacher),
                        "q2_teacher_model_messages": _messages_json(
                            copied_rgb, q2_teacher_model_prompt
                        ),
                    },
                    "targets": {
                        "structured_ground_truth": ground_truth_structure,
                        "q1_teacher_target": q1_target,
                        "q2_teacher_training_target": q2_target,
                        "q2_teacher_model_target": q2_teacher_model_target,
                    },
                    "outputs": {
                        "q1_student_raw": q1_output or "",
                        "q2_student_raw": q2_output or "",
                        "q1_teacher_raw": q1_teacher_output or "",
                        "q2_teacher_raw": q2_teacher_output or "",
                        "q1_student_parsed": parsed_q1,
                        "q2_student_parsed": parsed_q2,
                        "q1_teacher_parsed": parsed_teacher_q1,
                        "q2_teacher_parsed": parsed_teacher_q2,
                    },
                    "memory": memory_trace,
                    "flags": frame_log,
                }
                (case_dir / "case_record.json").write_text(
                    json.dumps(case_record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            # results.json 不是“只留指标”，而是把训练前 base probe 与训练后 LoRA probe
            # 都需要的证据集中进一个 results.json：真实输入 messages、完整分析输出、
            # privileged teacher 输入/脚本真值、场景标签和 memory 都必须保留。它只省掉
            # 重复的逐文件 TXT/JSON 与 JPEG 副本，避免人工在几十个文件间来回寻找。
            frame_record = {
                "scenario": route.scenario,
                "route_id": route.route_id,
                "frame_index": frame_index,
                "frame_id": frame.frame_id,
                "selection_reason": selection.primary_reason,
                "selection_reason_description": PROBE_REASON_DESCRIPTIONS.get(
                    selection.primary_reason, selection.primary_reason
                ),
                "gap_reset": selection_gap_reset,
                "ground_truth": {
                    # labels 包含 RS/EVENT 真值、原始 event_code、候选池、weather、
                    # goal 坐标和来源，足以回查“场景真值是怎么来的”。
                    **labels,
                    "resolved_event_target": event_target.label,
                    "structured": ground_truth_structure,
                },
                "inputs": {
                    "rgb_history_paths": list(frame.history_rgb_paths),
                    "q1_student_messages": _messages_json(copied_rgb, q1_student),
                    "q1_teacher_messages": _messages_json(copied_rgb, q1_teacher),
                    # Q2 不会重新发送 system/RGB，而是在 Q1 assistant KV 后追加一个
                    # user turn。把 suffix 与续接来源分开记录，避免把可视化误读成
                    # “第二次独立问图”；RS 错误时 student suffix 自然为空。
                    "q2_student_user_turn": {
                        "role": "user",
                        "content": q2_student,
                        "continued_from": "student.q1_output_kv",
                    },
                    "q2_teacher_training_user_turn": {
                        "role": "user",
                        "content": q2_teacher,
                        "continued_from": "student.q1_output_kv",
                    },
                    "q2_teacher_model_user_turn": {
                        "role": "user",
                        "content": q2_teacher_model_prompt,
                        "continued_from": "teacher.q1_output_kv",
                    },
                },
                "teacher_targets": {
                    "structured_ground_truth": ground_truth_structure,
                    "q1": q1_target,
                    # training target 基于 student Q1 rollout 后的 memory，是 OPSD
                    # 实际监督真值；model target 则对应纯 base teacher 自己的 Q1 续接。
                    "q2_training": q2_target,
                    "q2_teacher_model": q2_teacher_model_target,
                },
                "student": {
                    # q1/q2_output 都是完整生成文本，包含 Scene Description、
                    # Critical Object Description、Reasoning on Intent 与最终答案。
                    "q1_output": q1_output or "",
                    "q2_output": q2_output or "",
                    "q1_parsed": parsed_q1 if bundle is not None else {},
                    "q2_parsed": parsed_q2 if bundle is not None else {},
                    "rs_correct": q1_rs_ok if bundle is not None else None,
                    "abnormal_correct": (
                        q1_abnormal == frame.abnormal
                        if bundle is not None and q1_abnormal is not None
                        else None
                    ),
                    "q2_triggered": q2_triggered if bundle is not None else None,
                    "event_correct": q2_event_correct if bundle is not None else None,
                },
                "teacher": {
                    # 只有 --with-teacher-model 时这些字段才非空；该模型始终不加载
                    # student LoRA，用于和训练前纯 base teacher 保持同一比较口径。
                    "q1_output": q1_teacher_output or "",
                    "q2_output": q2_teacher_output or "",
                    "q1_parsed": parsed_teacher_q1,
                    "q2_parsed": parsed_teacher_q2,
                    "rs_correct": q1_teacher_rs_correct,
                    "abnormal_correct": q1_teacher_abnormal_correct,
                    "q2_triggered": q2_teacher_triggered if teacher_bundle is not None else None,
                    "event_correct": q2_teacher_event_correct,
                },
                "memory": memory_trace,
                "transition": transition_fields,
            }

            if review_artifacts:
                # 默认每帧只保留三个 JSON。output 同时放 raw/parsed/GT/correctness，
                # 既满足完整证据审计，也避免同一结论散落到多个重复文件。
                review_files = {
                    "input.json": {
                        "scenario": route.scenario,
                        "route_id": route.route_id,
                        "frame_id": frame.frame_id,
                        "selection": {
                            "reason": frame_record["selection_reason"],
                            "description": frame_record["selection_reason_description"],
                            "gap_reset": frame_record["gap_reset"],
                        },
                        "rgb": copied_rgb,
                        **frame_record["inputs"],
                    },
                    "output.json": {
                        "ground_truth": frame_record["ground_truth"],
                        "teacher_targets": frame_record["teacher_targets"],
                        "student": frame_record["student"],
                        "teacher": frame_record["teacher"],
                        "correctness": {
                            "q1_rs_correct": frame_record["student"]["rs_correct"],
                            "q1_abnormal_correct": frame_record["student"]["abnormal_correct"],
                            "q2_triggered": frame_record["student"]["q2_triggered"],
                            "q2_event_correct": frame_record["student"]["event_correct"],
                            "q2_invalid_output": q2_invalid if bundle is not None else None,
                        },
                        "transition": frame_record["transition"],
                    },
                    "memory.json": frame_record["memory"],
                }
                for filename, payload in review_files.items():
                    (case_dir / filename).write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            compact_frames.append(frame_record)
            frame_logs.append(frame_log)
            all_frame_logs.append(frame_log)
            previous_pred_rs_label = pred_rs_label
            previous_pred_abnormal = pred_abnormal
            case_idx += 1
        if full_artifacts:
            (route_dir / "timeline.json").write_text(
                json.dumps(frame_logs, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _write_timeline_png(route_dir / "timeline.png", frame_logs)
            manifest.append(
                {
                    "route_dir": str(route_dir),
                    "scenario": route.scenario,
                    "route_id": route.route_id,
                    "frames": len(frame_logs),
                    "selection_reason_counts": dict(
                        sorted(Counter(str(item.get("selection_primary_reason")) for item in frame_logs).items())
                    ),
                }
            )
    if full_artifacts:
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    summary = summarize_probe(
        all_frame_logs,
        student_enabled=bundle is not None,
        teacher_enabled=teacher_bundle is not None,
        student_adapter_dir=str(pathlib.Path(args.adapter_dir)) if args.adapter_dir else None,
        student_disable_adapter=bool(student_disable_adapter),
    )
    summary["sampling"] = {
        key: value for key, value in selection_payload.items() if key != "cases"
    }
    summary["artifact_level"] = artifact_level
    summary["generation_limits"] = {
        "max_new_tokens_q1": int(args.max_new_tokens_q1),
        "max_new_tokens_q2": int(args.max_new_tokens_q2),
    }
    transition_report = build_transition_report(all_frame_logs, summary=summary)
    transition_report["student_enabled"] = bundle is not None
    memory_recovery_report = build_memory_recovery_report(all_frame_logs)
    memory_recovery_report["student_enabled"] = bundle is not None
    summary["memory_recovery"] = memory_recovery_report["summary"]
    if full_artifacts:
        (out_dir / "transition_report.json").write_text(
            json.dumps(transition_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["transition_report"] = "transition_report.json"
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "memory_recovery_report.json").write_text(
            json.dumps(memory_recovery_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # review/full 的完整证据已经逐帧落盘，顶层只保留轻量目录索引，避免再次把所有
    # prompt/output/memory 复制进一个巨大 results.json。compact 没有逐帧目录，才把
    # frame_record 直接内嵌，继续满足无文件展开的快速合同检查用途。
    frame_artifacts = [
        {
            "scenario": item["scenario"],
            "route_id": item["route_id"],
            "frame_id": item["frame_id"],
            "directory": str(
                pathlib.Path("scenarios")
                / f"{_safe_name(item['scenario'])}__{_safe_name(item['route_id'])}"
                / f"frame_{int(item['frame_id']):04d}"
            ),
        }
        for item in compact_frames
    ]
    results = {
        "format_version": 2,
        "artifact_level": artifact_level,
        "sampling": {key: value for key, value in selection_payload.items() if key != "cases"},
        "summary": summary,
        "memory_recovery_report": memory_recovery_report,
        "frame_artifacts": frame_artifacts if review_artifacts else [],
        "frames": compact_frames if artifact_level == "compact" else [],
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[write] {results_path} frames={len(compact_frames)} "
        f"artifact_level={artifact_level}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    """解析小样本 probe 参数。

    不传 ``--with-model`` 时只落盘静态合同；传 ``--with-model`` 才运行 student，
    ``--with-teacher-model`` 则额外运行无 LoRA 的 privileged base teacher。
    """

    p = argparse.ArgumentParser(description="Dump SFT v5 probe cases")
    p.add_argument("--index", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="checkpoints/sft_v5_probe")
    p.add_argument(
        "--num-cases",
        type=int,
        default=24,
        help="random/RS 的总帧预算；UE 模式为最小预算且不会截断完整 UE span",
    )
    p.add_argument(
        "--sequence-length",
        type=int,
        default=24,
        help="random 模式每个连续片段的目标帧数；默认观察 24 帧以覆盖延迟纠正",
    )
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
    p.add_argument(
        "--sample-mode",
        choices=("random", "rs_transition", "ue_transition"),
        default="random",
        help="小样本选帧：random 连续片段；RS 取变化前后；UE 取完整持续段及前后邻帧",
    )
    p.add_argument(
        "--context-radius",
        type=int,
        default=8,
        help="RS/UE 专项中边界前后保留的连续帧数；默认观察变化后 8 帧",
    )
    p.add_argument("--seed", type=int, default=20260711, help="random 模式随机种子，用于复现同一批帧")
    p.add_argument(
        "--artifact-level",
        choices=("compact", "review", "full"),
        default="review",
        help="review 每帧保存 RGB + input/output/memory；compact 只写 results；full 再保存 legacy 文件",
    )
    p.add_argument("--with-model", action="store_true")
    p.add_argument("--with-teacher", action="store_true", help="compat flag: v5 always dumps teacher prompt/target")
    p.add_argument("--with-teacher-model", action="store_true", help="load base Qwen without LoRA to generate privileged teacher outputs")
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--teacher-model-dir", type=str, default=None, help="optional base Qwen dir for teacher generation; defaults to --model-dir")
    p.add_argument("--adapter-dir", type=str, default=None)
    p.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    # 手工小样本用于完整审阅，默认与训练 rollout 对齐；训练内自动 probe 会通过
    # checkpoint-probe-max-new-tokens-* 独立显式传入更小的 256/192 旁路上限。
    p.add_argument("--max-new-tokens-q1", type=int, default=1024)
    p.add_argument("--max-new-tokens-q2", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    """CLI 入口：默认生成逐帧精简证据，按需展开 legacy 深度审计产物。"""

    dump_probe(parse_args())


if __name__ == "__main__":
    main()
