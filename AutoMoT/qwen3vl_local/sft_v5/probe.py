"""SFT v5 case-level probe。

默认不加载模型，只按完整 route ID 或变化专项窗口检查 label / memory / prompt 合同。传 `--with-model`
后会额外生成低频 RS_SLOW 和逐帧 EVENT_FAST 输出；传
`--with-teacher-model` 后会额外用默认/base Qwen 跑 privileged teacher prompt。
训练前 OPSD 能力体检必须不传 `--adapter-dir`，即 teacher/student 都只用普通 Qwen，
不导入任何 LoRA。

默认 ``review`` 按 ``scenarios/<scenario>__<route>/frame_<id>/`` 保存连续帧。每帧只保留
实际输入 RGB、``input.json``、``output.json`` 和 ``memory.json``：输入、学生/老师
输出、解析结构、场景真值与两问 memory 转换都有唯一入口。``compact`` 只写汇总
``results.json``；显式指定 ``--artifact-level full`` 时才额外生成旧式逐项文件。

小样本只保留三种直观模式：``random`` 随机完整 route ID，``rs_transition`` 查看同一次
RS 变化前后，``ue_transition`` 查看同一次 UE 的进入、持续和退出。不传模型开关时
只生成静态 prompt/target 合同。

推理模式与大样本 eval 保持一致：每个连续窗口首帧只初始化
一次 memory，后续 student 只由自己的 Q1/Q2 输出推进；reference
只沿 GT 推演并用于比较。慢帧 Q2 续接当帧 Q1 KV，快帧 Q2
使用当前 RGB fresh prefill。这些来源都落在 ``results.json`` 或
逐帧 ``input/output/memory.json`` 中，可用来审计模型是否在沿用
过期 memory，而不是真正重新看图。
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
from qwen3vl_local.sft_v5.labels import RS_LABEL_TO_OPTION, option_for_event  # noqa: E402
from qwen3vl_local.sft_v5.prompts import (  # noqa: E402
    MemoryCurriculumConfig,
    MemoryCurriculumState,
    SYSTEM_PROMPT_V5,
    advance_memory_age,
    build_q1_student_prompt,
    build_q1_teacher_prompt,
    build_q1_teacher_target,
    build_q2_student_prompt,
    build_q2_teacher_prompt,
    build_q2_teacher_target,
    initialize_student_memory,
    observe_inference_rs_schedule,
    parse_q1_output,
    parse_q2_output,
    rs_slow_interval_for_state,
    should_run_rs_slow,
    should_run_event_fast,
    should_trigger_q2,
    update_memory_after_q1,
    update_memory_after_q2,
    update_memory_navigation,
    observe_training_memory,
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
    "random": "从 validation 中按固定随机种子抽取完整 route ID，并测试该 ID 的全部帧",
    "rs_transition": "检查同一次 RS 变化的变化前帧、新 RS 首帧和变化后帧",
    "ue_transition": "检查同一 UE 片段的进入前 RE、UE 内部和退出后 RE",
}

PROBE_REASON_DESCRIPTIONS = {
    "random": "随机完整 route ID 中按时间顺序测试的帧",
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
    """把 scenario / route id 转成目录安全名称。

    只替换非字母数字等字符并限长，不做随机化或哈希；这样
    artifact 目录仍可以被人直接映射回原 scenario/route。
    """

    # route_id 里通常已经包含 Town/route/time 等有用信息；这里只替换文件系统不友好
    # 的字符，不做哈希，方便人眼从目录名直接定位原始 route。
    keep = []
    for ch in str(text):
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    out = "".join(keep).strip("_")
    return (out[:max_len] or "unknown")


def _memory_json(memory: Any) -> Dict[str, Any]:
    """把 Memory 对象投影为稳定、可序列化的审计字段。

    使用 ``getattr`` 是为了同时兼容 CLI eval bundle 与训练内自动
    probe 传入的 memory 实现。返回两个离散 hypothesis、各自连续未变的
    4Hz age 和逐帧更新的 goal 坐标，不会隐式填 GT 或自动纠错。
    """

    return {
        "rs_label": getattr(memory, "rs_label", None),
        "rs_option": getattr(memory, "rs_option", None),
        "event_label": getattr(memory, "event_label", None),
        "ego_to_goal_x": getattr(memory, "ego_to_goal_x", None),
        "ego_to_goal_y": getattr(memory, "ego_to_goal_y", None),
        "rs_age_frames": int(getattr(memory, "rs_age_frames", 0) or 0),
        "event_age_frames": int(getattr(memory, "event_age_frames", 0) or 0),
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

    ``goal_matches`` 单独报告，因为 EGO_TO_GOAL_XY 是每帧外部刷新的
    导航条件，不属于 RS/EVENT 自主恢复。``discrete_state_matches``
    因此只合并 RS 与 EVENT，不让坐标浮点表示影响离散状态指标。
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
    """用可序列化形式还原送给 Qwen 的 system/user messages。

    artifact 中图像记录使用 ``file/source`` 而非 PIL 对象，既可读又能
    回溯原始 JPEG。Q2 真实推理是 KV suffix；这个 helper 在 full
    artifact 里提供展开视图，具体续接来源另由 frame record 标注。
    """

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
    """在 ``full`` 模式写出 system/user 分离会话视图。

    同时写 TXT 与 JSON：TXT 方便人工 diff prompt，JSON 保留角色与图像
    顺序，便于现有 v3/v4 审计脚本直接复用。
    """

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
    复制失败不中断 probe；返回记录会标明 ``copied/error``，让 prompt/
    memory 合同仍可审计，同时不掩盖图像证据缺失。
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
    """批量写出 ``full`` 模式的 prompt/target/output 文本视图。

    ``None``/空输出统一落成空文件，使每帧目录 schema 固定；人工
    审计时可以区分“未触发”与“文件被遗漏”。
    """

    for name, text in files.items():
        (frame_dir / name).write_text(text or "", encoding="utf-8")


def _write_timeline_png(path: pathlib.Path, frame_logs: List[Dict[str, Any]]) -> None:
    """写轻量时间线图，仿 v3 快速定位门控错误。

    红色表示实际 RS_SLOW 答错，蓝色表示进入 EVENT_FAST，绿色
    表示无 student 模型的静态 teacher-forced 合同，灰色为其他帧。
    图只是 full artifact 的导航索引，PIL 缺失时跳过不影响 JSON 真值。
    """

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
    """汇总单帧标签、候选池与数据源证据。

    这份字典是 results/review/full 三种 artifact 的共同 GT schema。
    ``event_code`` 和 raw candidate 只作审计；学生真正作答的是
    ``event_option_map`` 中显式标明 REGULAR/UNUSUAL 的合并选择题。
    weather 也标为 teacher-only，方便检查 privileged 信息是否泄漏。
    """

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
    """一条被选中的 probe 帧及其可审计原因。

    ``primary_reason`` 决定目录/报告中的主分类，``reasons`` 同时保留
    该帧与 UE、RS 边界或稳定 RE 的全部关系，避免多重语义在
    选帧去重时丢失。
    """

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

    ``ue_nearby_re`` 特意只标真实 RE 帧，它是检查假阳性的硬负例；UE span 内的帧由
    `ue_positive` 覆盖。`rs_nearby` 则保留变换点前后的视觉上下文。
    该函数只打标不裁剪，真正的预算、route 轮询和完整 UE span
    保留在 ``build_probe_selection_plan`` 中处理。
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
    """列出 RS 变化点，``start=end`` 均指向新 RS 首帧。

    RS 是离散边界而非持续片段，因此这里只交付 anchor；前后
    ``context_radius`` 由上层选帧计划扩展，以保持统一的预算规则。
    """

    windows = []
    for route_idx, route in enumerate(routes):
        for frame_idx in range(1, len(route.frames)):
            if route.frames[frame_idx - 1].rs_label != route.frames[frame_idx].rs_label:
                windows.append((route_idx, frame_idx, frame_idx))
    return _route_round_robin_windows(windows)


def _ue_transition_windows(routes: Sequence[SequenceRow]) -> List[Tuple[int, int, int]]:
    """列出连续 UE span，优先返回两侧都有 RE 的完整片段。

    一个 UE 持续段是不可分的 probe 单元：只看入口帧无法判断模型
    是否在事件内稳定，也无法测 UE->RE 退出。route 边界导致缺
    一侧 RE 的 partial span 仍保留，但排在可完整审计的 span 之后。
    """

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
    num_routes: int = 1,
) -> List[ProbeSelection]:
    """按三种公开语义构造小样本计划。

    ``random`` 用 seed 随机抽取完整 route ID，并保留该 ID 的全部帧；
    ``rs_transition`` 保留同一 RS 变化点的
    前/当前/后帧；``ue_transition`` 保留同一 UE span 的进入前 RE、UE 内部和
    退出后 RE。UE 模式不会用 ``num_cases`` 截断一个真实 UE span：它保留整段 UE，
    再向前后补 ``context_radius`` 帧，因此长 UE 的实际返回帧数可以超过预算。
    专项模式找不到真实变化时返回空结果，不用普通帧冒充专项样本。

    ``sequence_length`` 仅作为旧命令兼容参数保留，random 模式不再使用它截断 route。
    ``num_routes`` 只控制 random 抽取多少个完整 ID，默认 1。

    返回的计划最终按 ``route_index/frame_index`` 恢复时间顺序。
    这一步不只为可读性：student memory 必须按真实时序推进，
    如果按“最重要 case 优先”的选择顺序直接推理，会制造不存在的
    memory 跳转。
    """

    mode = str(sample_mode or "random").lower()
    if mode not in {"random", "rs_transition", "ue_transition"}:
        raise ValueError(
            f"unsupported probe sample mode: {sample_mode}; "
            "expected random/rs_transition/ue_transition"
        )
    # sequence_length 是兼容入口。显式转换能尽早暴露非法旧参数，但 random 不再使用它。
    _ = max(1, int(sequence_length))
    limit = max(0, int(num_cases))
    if mode != "random" and limit == 0:
        return []
    reasons_by_key = _probe_candidate_reasons(routes, context_radius=context_radius)
    selected: List[Tuple[Tuple[int, int], str]] = []
    selected_keys: set[Tuple[int, int]] = set()

    def add_frame(
        key: Tuple[int, int],
        reason: str,
        *,
        respect_limit: bool = True,
    ) -> None:
        """去重追加帧，并按模式解释总帧预算。

        RS 专项把 ``num_cases`` 当硬上限；UE 专项为了不从中间截断
        span，会使用 ``respect_limit=False`` 先收完当前窗口。random
        按 route 数限制，所以同样不用 frame 上限。
        """

        if (respect_limit and len(selected) >= limit) or key in selected_keys:
            return
        selected.append((key, reason))
        selected_keys.add(key)

    if mode == "random":
        # random 的采样单位是完整 route ID，不是 frame 或短片段。被选 route 的所有帧
        # 都按时间顺序运行，因此可以完整观察 student memory 从首帧到末帧的逐步变化。
        rng = random.Random(int(seed))
        route_indices = [idx for idx, route in enumerate(routes) if route.frames]
        rng.shuffle(route_indices)
        route_limit = min(max(1, int(num_routes)), len(route_indices))
        for route_idx in route_indices[:route_limit]:
            for frame_idx in range(len(routes[route_idx].frames)):
                add_frame((route_idx, frame_idx), "random", respect_limit=False)
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
    # 所有模式都按 route/frame 恢复时间顺序。random 的随机性体现在 route ID 选择，
    # 不是帧顺序；顺序化后 memory 才能从 ID 首帧自然推进到末帧。
    selected.sort(key=lambda item: item[0])

    plan: List[ProbeSelection] = []
    final_selected = selected if mode in {"random", "ue_transition"} else selected[:limit]
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

    上下文同时使用 ``torch.inference_mode()``，保证纯生成阶段不保留
    autograd graph。退出时只恢复原先的 train/eval 模式，不更改
    adapter 权重或 optimizer，因此可安全嵌入训练中 checkpoint probe。
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
    """返回可审计比例；没有有效分母时写 ``null``。

    例如因 RS gate 错误而没有任何 Q2 case 时，EVENT accuracy 是
    “未定义”而不是 0%。保留 ``None`` 可防止下游 route macro 把缺失
    分母错当成模型全错。
    """

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
    """把逐帧 flags 聚合成 base/checkpoint 可直接比较的 summary。

    summary 同时保留三组口径：RS_SLOW 实际触发帧、经 RS gate
    后实际进入的 EVENT_FAST 帧，以及 privileged base teacher 的独立
    生成帧。它们的分母不可互换：否则 RS 复用帧会稀释慢思考
    准确率，或被 RS 跳过的帧会错进 conditional EVENT 分母。
    开启 student 时再调用共用 metrics 补全 precision/recall/F1、FP/FN、
    end-to-end EVENT 和 memory 依赖指标。
    """

    frames = len(frame_logs)
    student_frames = frames if student_enabled else 0
    rs_slow_logs = [item for item in frame_logs if bool(item.get("q1_triggered"))] if student_enabled else []
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
    # q1_rs_correct 只描述真正运行 RS_SLOW 的 Q1 输出；快帧没有 Q1，不能把它当成
    # 错误塞进每帧 RS 准确率。rs_gate_correct 才是“本帧最终使用的 RS 是否正确”，
    # 与 metrics.py 的 rs_acc 口径一致。
    rs_gate_correct_count = (
        sum(bool(item.get("rs_gate_correct")) for item in frame_logs)
        if student_enabled
        else 0
    )
    summary = {
        "frames": frames,
        "student_enabled": bool(student_enabled),
        "teacher_enabled": bool(teacher_enabled),
        "student_adapter_dir": student_adapter_dir,
        "student_adapter_enabled": bool(student_enabled and not student_disable_adapter and student_adapter_dir),
        "student_base_mode": bool(student_enabled and (student_disable_adapter or not student_adapter_dir)),
        # q1_* 是旧 comparison schema 的兼容名字；数值统一为每帧实际 RS gate 口径。
        # 真正的低频 Q1 准确率由 rs_slow_accuracy 单独报告。
        "q1_rs_correct": rs_gate_correct_count,
        "q1_rs_accuracy": _ratio(rs_gate_correct_count, student_frames),
        "rs_gate_accuracy": _ratio(rs_gate_correct_count, student_frames),
        "rs_slow_frames": len(rs_slow_logs),
        "rs_slow_trigger_rate": _ratio(len(rs_slow_logs), student_frames),
        "rs_slow_accuracy": _ratio(sum(bool(item.get("q1_rs_correct")) for item in rs_slow_logs), len(rs_slow_logs)),
        "event_family_correct": sum(bool(item.get("event_family_correct")) for item in q2_logs),
        "event_family_accuracy": _ratio(
            sum(bool(item.get("event_family_correct")) for item in q2_logs),
            len(q2_logs),
        ),
        # 旧 comparison.json 兼容别名；数值现在源自 EVENT 选项的 RE/UE family，
        # 不再表示 Q1 存在独立 ABNORMAL 输出。
        "q1_abnormal_correct": sum(bool(item.get("event_family_correct")) for item in q2_logs),
        "q1_abnormal_accuracy": _ratio(
            sum(bool(item.get("event_family_correct")) for item in q2_logs),
            len(q2_logs),
        ),
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
        "teacher_event_family_accuracy": _ratio(
            sum(bool(item.get("teacher_event_family_correct")) for item in teacher_q2_logs),
            len(teacher_q2_logs),
        ),
        "teacher_q1_abnormal_accuracy": _ratio(
            sum(bool(item.get("teacher_event_family_correct")) for item in teacher_q2_logs),
            len(teacher_q2_logs),
        ),
        "teacher_q2_frames": len(teacher_q2_logs),
        # teacher Q2 同时覆盖慢帧（续接 teacher Q1 KV）与快帧（fresh RGB prefill），
        # 因而必须除以全部 probe frame。若除以 teacher Q1 帧数，快帧较多时比率会大于 1。
        "teacher_q2_trigger_rate": _ratio(len(teacher_q2_logs), frames if teacher_enabled else 0),
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

    RS 恢复在 Q1 后判断，EVENT 恢复在 Q2 后判断。如果定向选帧
    中间有 gap，窗口首帧会被标记 ``selection_gap_reset`` 并排除，
    因为它没有真实的上一帧 student memory，无法表示自主跳转。
    """

    logs = list(frame_logs)

    def collect(kind: str) -> List[Dict[str, Any]]:
        """收集一种变化的逐 case 恢复结果。

        从变化首帧开始向后扫描，遇到 route/选帧窗口断点或下一个
        同类 GT 变化即停止。这样不会把“后来真值又变了”错计为
        对上一个目标的延迟恢复。
        """

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

    函数有三种使用方式：

    1. 不传 bundle，且 CLI 不开 ``--with-model``：只验证选帧、prompt、
       teacher target 和 memory 合同，不读图进 Qwen。
    2. CLI ``--with-model``：加载 base 或 ``--adapter-dir`` student，运行
       closed-loop RS_SLOW/EVENT_FAST。
    3. 训练内传入 bundle：复用 rank0 现有 Qwen，通过
       ``student_disable_adapter/teacher_disable_adapter`` 切换 base/LoRA 对照，
       不另起进程或加载第二份权重。

    返回的 summary 供 checkpoint ``comparison.json`` 聚合；更完整的
    选帧、逐帧 CoT、teacher target、memory 和 transition 证据写入
    ``results.json`` 及所选 artifact 目录。
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
    num_routes = max(1, int(getattr(args, "num_routes", 1)))
    artifact_level = str(getattr(args, "artifact_level", "review")).lower()
    if artifact_level not in {"compact", "review", "full"}:
        raise ValueError(
            f"unsupported artifact level: {artifact_level}; expected compact/review/full"
        )
    review_artifacts = artifact_level in {"review", "full"}
    full_artifacts = artifact_level == "full"
    # 先固定选帧计划，再加载/调用模型。这样 base、checkpoint、
    # final 可用同一 seed 和同一 route 作公平对照，也能在没有 GPU
    # 时先发现选帧/prompt 合同问题。
    selection_plan = build_probe_selection_plan(
        ds.rows,
        num_cases=int(args.num_cases),
        sample_mode=sample_mode,
        context_radius=context_radius,
        seed=sample_seed,
        sequence_length=sequence_length,
        num_routes=num_routes,
    )
    selection_by_key = {
        (item.route_index, item.frame_index): item
        for item in selection_plan
    }
    selection_category_counts = Counter(item.primary_reason for item in selection_plan)
    selected_route_ids = list(dict.fromkeys(item.route_id for item in selection_plan))
    selection_payload = {
        "sample_mode": sample_mode,
        "sample_mode_description": PROBE_SAMPLE_MODE_DESCRIPTIONS[sample_mode],
        "sequence_length": sequence_length,
        "sequence_length_ignored_for_random": sample_mode == "random",
        "requested_routes": num_routes if sample_mode == "random" else None,
        "context_radius": context_radius,
        "seed": sample_seed,
        "requested_cases": int(args.num_cases) if sample_mode != "random" else None,
        "requested_cases_ignored_for_random": sample_mode == "random",
        "selected_cases": len(selection_plan),
        "selected_route_ids": selected_route_ids,
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

    # manifest/full 日志是展开的人工审计入口；compact_frames 是
    # results.json 的统一 schema。all_frame_logs 只保留结构化 flags，供
    # metrics/transition/recovery 聚合，不再复制图像或 KV。
    manifest: List[Dict[str, Any]] = []
    all_frame_logs: List[Dict[str, Any]] = []
    compact_frames: List[Dict[str, Any]] = []
    case_idx = 0
    rs_schedule_config = MemoryCurriculumConfig(
        rs_slow_interval=int(getattr(args, "rs_slow_interval", 4)),
        rs_slow_interval_jitter=int(
            getattr(args, "rs_slow_interval_jitter", 1)
        ),
    )
    # 自动 checkpoint probe 与手工 probe 默认都模拟真实无 GT 启动；oracle/GT 模式只
    # 用于复现旧报告，策略会写进 results，避免和 deployable 指标误混。
    rs_schedule_policy = str(getattr(args, "rs_schedule_policy", "deployable"))
    initial_memory_mode = str(getattr(args, "initial_memory", "unknown"))
    for route_idx, route in enumerate(ds.rows):
        # 未入选的 route 不创建目录、不读图。选中 route 仍按原
        # frame_index 遍历，保证连续窗口的 memory 时序与真实数据一致。
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
        rs_schedule_state = MemoryCurriculumState()
        window_ordinal = 0
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
                rs_schedule_state = MemoryCurriculumState()
                window_ordinal = 0
            previous_selected_frame_index = frame_index
            rs_target = _rs_target_from_frame(frame)
            event_target = _event_target_from_frame(frame)
            memory_initialized_this_frame = memory is None
            memory_initialized_from_gt = bool(
                memory_initialized_this_frame and initial_memory_mode == "ground_truth"
            )
            if memory is None:
                # student 默认以 UNKNOWN/no-prior 开始；reference 才使用 GT。这样首帧
                # RS_SLOW 必须从 RGB 判断，不能靠 prompt 中预填的正确 RS 拿高分。
                memory = initialize_student_memory(
                    rs_target,
                    ego_to_goal_xy=frame.ego_to_goal_xy,
                    mode=initial_memory_mode,
                )
                reference_memory = _reset_memory_for_frame_row(frame)
            else:
                # 导航坐标是逐帧外部输入，不属于标签纠错；RS/EVENT 保持学生上一帧结果。
                memory = advance_memory_age(memory)
                memory = update_memory_navigation(memory, frame.ego_to_goal_xy)
                assert reference_memory is not None
                reference_memory = advance_memory_age(reference_memory)
                reference_memory = update_memory_navigation(
                    reference_memory,
                    frame.ego_to_goal_xy,
                )
            assert reference_memory is not None
            # 默认 deployable 调度不读取 GT：UNKNOWN/非法输出与 RS 标签变化触发确认，
            # 稳定合法标签按周期复核。oracle 模式显式保留旧式 GT mismatch 触发。
            schedule_key = f"{route.scenario}/{route.route_id}"
            scheduled_rs_interval = rs_slow_interval_for_state(
                rs_schedule_state,
                rs_schedule_config,
                schedule_key=schedule_key,
                schedule_seed=int(getattr(args, "seed", 20260711)),
            )
            run_rs_slow, rs_schedule_reason = should_run_rs_slow(
                rs_schedule_state,
                rs_schedule_config,
                memory=memory,
                gt_rs_label=(frame.rs_label if rs_schedule_policy == "oracle" else None),
                frame_ordinal=window_ordinal,
                schedule_key=schedule_key,
                schedule_seed=int(getattr(args, "seed", 20260711)),
            )
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
            q1_student = build_q1_student_prompt(memory) if run_rs_slow else ""
            # RS teacher prompt 是 privileged 输入：可看 XML weather 与 GT RS；EVENT
            # teacher 单独看 GT EVENT。target 都会清洗成学生视角，便于审计泄漏。
            q1_teacher = (
                build_q1_teacher_prompt(
                    memory,
                    rs_target=rs_target,
                    weather_text=frame.weather_text,
                )
                if run_rs_slow
                else ""
            )
            q1_target = (
                build_q1_teacher_target(
                    rs_target=rs_target,
                    weather_text=frame.weather_text,
                )
                if run_rs_slow
                else ""
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
            q1_rs_ok = False
            rs_gate_ok = memory.rs_label == frame.rs_label
            q1_teacher_rs_correct: Optional[bool] = None
            q2_triggered = False
            q2_invalid = False
            q2_event_correct: Optional[bool] = None
            q2_teacher_event_correct: Optional[bool] = None
            q2_teacher_triggered = False
            q2_candidate_mismatch = False
            images: Optional[List[Any]] = None

            def _images_for_generation() -> List[Any]:
                """懒加载并缓存该帧 RGB，仅供当帧生成共享。

                静态 probe 完全不调用此 helper；启用 student/teacher 时，
                同帧 Q1 和 fast Q2 可复用已解码的 PIL 图像，但缓存不跨帧，
                所以 EVENT_FAST 仍明确读的是当前 RGB。
                """

                nonlocal images
                if images is None:
                    images = _load_images(frame.history_rgb_paths)
                return images

            # reference 分支只做离线对比，不进入任何 student/teacher prompt。RS 与
            # EVENT 分别按 GT 推演，不能把 reference 写回 student memory。
            reference_memory_after_q1_state = update_memory_after_q1(
                reference_memory,
                student_rs_label=frame.rs_label,
            )
            reference_memory_after_q1 = _memory_json(reference_memory_after_q1_state)
            memory_after_q1 = memory.copy()
            if teacher_bundle is not None and run_rs_slow:
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
            if bundle is not None:
                # student bundle 可以是纯 base Qwen（训练前体检）或 base+adapter（训练后可视化）。
                # 是否误传 adapter 会写入 flags.json 的 student_adapter_dir 供人工审计。
                if run_rs_slow:
                    with _probe_inference_context(bundle, disable_adapter=student_disable_adapter):
                        q1_output, q1_after = _generate_start(
                            bundle,
                            _images_for_generation(),
                            q1_student,
                            int(args.max_new_tokens_q1),
                        )
                    parsed_q1 = parse_q1_output(q1_output)
                    q1_rs_ok = should_trigger_q2(
                        student_rs_label=parsed_q1.get("rs_label"),
                        target_rs_label=frame.rs_label,
                    )
                    memory_after_q1 = update_memory_after_q1(
                        memory,
                        student_rs_label=parsed_q1.get("rs_label"),
                    )
                # 和 train/eval 保持同一门控：慢帧 Q1 必须在本帧解析且答对；旧 memory
                # 即使碰巧正确，也不能掩盖本帧 Q1 的 invalid/错误。稳定快帧才复用 RS。
                rs_gate_ok = should_run_event_fast(
                    rs_slow_ran=run_rs_slow,
                    q1_rs_correct=q1_rs_ok,
                    memory_rs_label=memory_after_q1.rs_label,
                    target_rs_label=frame.rs_label,
                )
                if rs_gate_ok:
                    # 只有 Q1 的 RS 正确才进入 Q2；这和训练时的采样/截断规则保持一致。
                    q2_student_memory_input = _memory_json(memory_after_q1)
                    q2_student = build_q2_student_prompt(
                        memory_after_q1,
                        option_map=frame.event_option_map,
                        regular_event_codes=frame.regular_event_codes,
                    )
                    q2_teacher = build_q2_teacher_prompt(
                        memory_after_q1,
                        option_map=frame.event_option_map,
                        event_target=event_target,
                        regular_event_codes=frame.regular_event_codes,
                    )
                    q2_target = build_q2_teacher_target(
                        memory_after_q1,
                        option_map=frame.event_option_map,
                        event_target=event_target,
                        regular_event_codes=frame.regular_event_codes,
                    )
                    q2_triggered = True
                    if run_rs_slow and q1_after is not None:
                        # 慢帧已经在 Q1 KV 中编码了当前 RGB、RS prompt 和学生
                        # 自己的 Q1 CoT/答案。Q2 只追加 EVENT user turn，保持真实
                        # 两轮对话；不把 q1_text 重 tokenize，也不用 teacher/GT Q1 替换。
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
                    else:
                        # 稳定快帧没有本帧 Q1 state，因此 Q2 用当前 RGB fresh
                        # prefill。禁止续接上个慢帧 KV，否则会把过期视觉和
                        # analysis 伪装成 EVENT_FAST 的当前帧证据。
                        with _probe_inference_context(bundle, disable_adapter=student_disable_adapter):
                            q2_output, q2_after = _generate_start(
                                bundle,
                                _images_for_generation(),
                                q2_student,
                                int(args.max_new_tokens_q2),
                            )
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
                if run_rs_slow:
                    memory_after_q1 = update_memory_after_q1(
                        memory,
                        student_rs_label=frame.rs_label,
                    )
                    q1_rs_ok = True
                rs_gate_ok = memory_after_q1.rs_label == frame.rs_label
                q2_triggered = rs_gate_ok
                q2_student_memory_input = _memory_json(memory_after_q1) if rs_gate_ok else None
                q2_student = build_q2_student_prompt(
                    memory_after_q1,
                    option_map=frame.event_option_map,
                    regular_event_codes=frame.regular_event_codes,
                )
                q2_teacher = build_q2_teacher_prompt(
                    memory_after_q1,
                    option_map=frame.event_option_map,
                    event_target=event_target,
                    regular_event_codes=frame.regular_event_codes,
                )
                q2_target = build_q2_teacher_target(
                    memory_after_q1,
                    option_map=frame.event_option_map,
                    event_target=event_target,
                    regular_event_codes=frame.regular_event_codes,
                )
                memory = (
                    update_memory_after_q2(memory_after_q1, student_event_label=event_target.label)
                    if rs_gate_ok
                    else memory_after_q1
                )
                parsed_q1 = (
                    {"rs_option": frame.rs_option, "rs_label": frame.rs_label}
                    if run_rs_slow
                    else {}
                )
                parsed_q2 = (
                    {"event_option": None, "event_label": event_target.label}
                    if rs_gate_ok
                    else {}
                )
                q2_candidate_mismatch = event_target.label not in set(frame.event_option_map.values())
                q2_event_correct = not q2_candidate_mismatch

            if teacher_bundle is not None:
                # 慢帧 Q2 续接 teacher 自己的 RS_SLOW KV；稳定 fast 帧没有 Q1，直接用
                # 当前 RGB + memory fresh prefill EVENT_FAST。两条路径都不复用旧 ABNORMAL。
                teacher_rs_gate_ok = bool(
                    (run_rs_slow and q1_teacher_after is not None and q1_teacher_rs_correct)
                    or (not run_rs_slow and memory_at_frame_start.rs_label == frame.rs_label)
                )
                if teacher_rs_gate_ok:
                    # teacher 的 Q2 memory 由 teacher 自己的 Q1 解析结果构造。
                    # 这能分开“base teacher 自身 RS 错”和“给定正确 RS 后 EVENT
                    # 仍错”两种能力问题，也避免 teacher 暗中借用 student/GT KV。
                    teacher_memory_after_q1 = update_memory_after_q1(
                        memory_at_frame_start,
                        student_rs_label=(
                            parsed_teacher_q1.get("rs_label")
                            if run_rs_slow
                            else memory_at_frame_start.rs_label
                        ),
                    )
                    q2_teacher_model_prompt = build_q2_teacher_prompt(
                        teacher_memory_after_q1,
                        option_map=frame.event_option_map,
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
                        if run_rs_slow:
                            q2_teacher_output, q2_teacher_after = _generate_next(
                                teacher_bundle,
                                q1_teacher_after,
                                q2_teacher_model_prompt,
                                int(args.max_new_tokens_q2),
                            )
                        else:
                            q2_teacher_output, q2_teacher_after = _generate_start(
                                teacher_bundle,
                                _images_for_generation(),
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
                "rs_schedule_policy": rs_schedule_policy,
                "student_initial_memory_mode": initial_memory_mode,
                "window_initialized_from_ground_truth": memory_initialized_from_gt,
                "reference_is_comparison_only": True,
                "forced_correction_applied": False,
                "q1": {
                    "triggered": run_rs_slow,
                    "schedule_reason": rs_schedule_reason,
                    "scheduled_interval_frames": int(scheduled_rs_interval),
                    # EVENT 是 EVENT|RS。该布尔量让 review artifact 无需人工对比两份
                    # JSON 就能确认：本帧 RS 变化是否触发了旧 EVENT 失效。
                    "event_context_invalidated_by_rs_change": bool(
                        run_rs_slow
                        and student_memory_after_q1.get("rs_label")
                        != memory_before.get("rs_label")
                    ),
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
                        # 静态 teacher-forced probe 虽然没有 student 输出，但确实构造并
                        # 保存了 Q2 student prompt；因此输入 memory 也必须展示，才能
                        # 审计 RS 变化后 EVENT 是否已变成 UNKNOWN/age=0。只有输出/正确性
                        # 继续用 None 表示“未实际运行模型”。
                        q2_student_memory_input,
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
                    and run_rs_slow
                    and not q1_input_matches_current_target
                    and q1_after_matches is True
                ),
                "q1_rs_corrupted_by_student": bool(
                    bundle is not None
                    and run_rs_slow
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
            pred_rs_label = (
                memory_after_q1.rs_label
                if bundle is not None and memory_after_q1.rs_label in RS_LABEL_TO_OPTION
                else (frame.rs_label if bundle is None else None)
            )
            observed_event_label = (
                parsed_q2.get("event_label")
                if bundle is not None
                else (event_target.label if q2_triggered else None)
            )
            pred_abnormal = (
                None if observed_event_label is None else observed_event_label != "RE"
            )
            would_reset_under_training = bool(
                bundle is not None and (not rs_gate_ok or (q2_triggered and q2_invalid))
            )
            rs_schedule_state.frames_seen = window_ordinal + 1
            if rs_schedule_policy == "oracle":
                rs_schedule_after = observe_training_memory(
                    rs_schedule_state,
                    rs_schedule_config,
                    rs_correct=(q1_rs_ok if bundle is not None else run_rs_slow),
                    rs_checked=run_rs_slow,
                    event_checked=q2_triggered,
                    event_correct=(bool(q2_event_correct) if bundle is not None else q2_triggered),
                    event_context_reset=bool(
                        run_rs_slow
                        and (
                            parsed_q1.get("rs_label")
                            if bundle is not None
                            else frame.rs_label
                        )
                        in RS_LABEL_TO_OPTION
                        and (
                            parsed_q1.get("rs_label")
                            if bundle is not None
                            else frame.rs_label
                        )
                        != memory_before.get("rs_label")
                    ),
                )
            else:
                rs_schedule_after = observe_inference_rs_schedule(
                    rs_schedule_state,
                    rs_checked=run_rs_slow,
                    memory_rs_label_before=memory_before.get("rs_label"),
                    student_rs_label=(
                        parsed_q1.get("rs_label")
                        if bundle is not None
                        else (frame.rs_label if run_rs_slow else None)
                    ),
                )
            memory_trace["rs_schedule_after"] = rs_schedule_after
            window_ordinal += 1
            rs_memory_known_wrong = bool(
                bundle is not None
                and memory_before.get("rs_label") in RS_LABEL_TO_OPTION
                and memory_before.get("rs_label") != frame.rs_label
            )
            rs_memory_unknown = bool(
                bundle is not None and memory_before.get("rs_label") not in RS_LABEL_TO_OPTION
            )
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
                bundle is not None
                and q2_triggered
                and event_memory_known
                and event_memory_label not in set(accepted_event_labels)
            )
            event_memory_unknown = bool(
                bundle is not None and q2_triggered and not event_memory_known
            )
            ground_truth_structure = {
                "q1": {
                    "rs_option": frame.rs_option,
                    "rs_label": frame.rs_label,
                },
                "q2": {
                    "event_family": "UE" if frame.abnormal else "RE",
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

            # pair_evaluated 要求 student 真正运行且本帧与上一选中帧连续。
            # GT 边界仍从原 route frame_index-1 取；定向窗口的跳帧首帧
            # 被记为 invalid/not-evaluated，不伪造 RS/UE 边界 TP/FP。
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
                "q1_triggered": run_rs_slow,
                "rs_slow_reason": rs_schedule_reason,
                "rs_slow_interval_draw": int(scheduled_rs_interval),
                "rs_schedule_policy": rs_schedule_policy,
                "rs_schedule_after": rs_schedule_after,
                "q1_rs_correct": q1_rs_ok,
                "rs_gate_correct": rs_gate_ok,
                "event_family_correct": bool(
                    pred_abnormal is not None and pred_abnormal == bool(frame.abnormal)
                ),
                # 兼容旧 probe schema；实际由 EVENT_FAST 的 RE/UE 选项推导。
                "q1_abnormal_correct": bool(
                    pred_abnormal is not None and pred_abnormal == bool(frame.abnormal)
                ),
                "q1_teacher_rs_correct": q1_teacher_rs_correct,
                "teacher_event_family_correct": bool(
                    parsed_teacher_q2.get("event_label") is not None
                    and (parsed_teacher_q2.get("event_label") != "RE") == bool(frame.abnormal)
                ),
                "q1_teacher_abnormal_correct": bool(
                    parsed_teacher_q2.get("event_label") is not None
                    and (parsed_teacher_q2.get("event_label") != "RE") == bool(frame.abnormal)
                ),
                "q2_triggered": q2_triggered,
                "q2_skipped_rs_wrong": bool(bundle is not None and not rs_gate_ok),
                "q2_event_correct": q2_event_correct,
                "q2_teacher_event_correct": q2_teacher_event_correct,
                "q2_teacher_triggered": q2_teacher_triggered,
                # 兼容旧 flags 消费方；新代码不再把 GT/student prompt 强接到 teacher Q1 KV。
                "q2_teacher_forced": False,
                "q2_student_continued_from_q1_kv": bool(bundle is not None and run_rs_slow and q2_triggered and q1_after is not None),
                "q2_teacher_continued_from_q1_kv": bool(run_rs_slow and q2_teacher_triggered and q1_teacher_after is not None),
                "q2_invalid_output": q2_invalid,
                "q2_candidate_mismatch": q2_candidate_mismatch,
                # 测试永不应用 GT 纠错。would_reset 是旧报告兼容字段，只表示本帧出现
                # Q1 RS 错或 Q2 非法；当前训练课程不会因此在下一帧直接 reset。
                "rs_wrong_reset": False,
                "reset_next": False,
                "would_reset_under_training": would_reset_under_training,
                "memory_forced_correction_applied": False,
                "memory_rs_input_known_wrong": rs_memory_known_wrong,
                "memory_rs_input_unknown": rs_memory_unknown,
                "memory_rs_copied_when_wrong": bool(
                    run_rs_slow
                    and rs_memory_known_wrong
                    and pred_rs_label == memory_before.get("rs_label")
                ),
                "memory_rs_recovered": bool(
                    run_rs_slow and (rs_memory_known_wrong or rs_memory_unknown) and q1_rs_ok
                ),
                "memory_event_input_known_wrong": event_memory_wrong,
                "memory_event_input_unknown": event_memory_unknown,
                "memory_event_copied_when_wrong": bool(
                    event_memory_wrong and parsed_q2.get("event_label") == event_memory_label
                ),
                "memory_event_recovered": bool(
                    bundle is not None
                    and q2_triggered
                    and (event_memory_wrong or event_memory_unknown)
                    and parsed_q2.get("event_label") in set(accepted_event_labels)
                ),
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
                "pred_event_label": observed_event_label,
                "pred_event_is_ue": (
                    None if observed_event_label is None else observed_event_label != "RE"
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
                        "continued_from": (
                            "student.q1_output_kv" if run_rs_slow else "fresh_rgb_prefill"
                        ),
                    },
                    "q2_teacher_training_user_turn": {
                        "role": "user",
                        "content": q2_teacher,
                        "continued_from": (
                            "student.q1_output_kv" if run_rs_slow else "fresh_rgb_prefill"
                        ),
                    },
                    "q2_teacher_model_user_turn": {
                        "role": "user",
                        "content": q2_teacher_model_prompt,
                        "continued_from": (
                            "teacher.q1_output_kv" if run_rs_slow else "fresh_rgb_prefill"
                        ),
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
                        pred_abnormal == frame.abnormal
                        if bundle is not None and pred_abnormal is not None
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
                    "abnormal_correct": (
                        (parsed_teacher_q2.get("event_label") != "RE") == frame.abnormal
                        if parsed_teacher_q2.get("event_label") is not None
                        else None
                    ),
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
                            "q1_triggered": run_rs_slow,
                            "rs_slow_reason": rs_schedule_reason,
                            "q1_rs_correct": frame_record["student"]["rs_correct"],
                            "event_family_correct": frame_record["student"]["abnormal_correct"],
                            # 兼容旧 review 消费方；不代表 Q1 仍输出 ABNORMAL。
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
    # 顶层聚合分三层：summary 供版本快速对比，transition_report
    # 专注 RS/UE 边界，memory_recovery_report 专注变化后多少帧自主对齐。
    # 三者都只读 all_frame_logs，不可再改写 closed-loop memory。
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
    summary["student_initial_memory_mode"] = initial_memory_mode
    summary["event_memory_semantics"] = "event_conditioned_on_rs"
    summary["rs_change_invalidates_event"] = True
    summary["rs_change_resets_event_error_context"] = True
    summary["rs_schedule_policy"] = rs_schedule_policy
    summary["rs_schedule_uses_ground_truth"] = rs_schedule_policy == "oracle"
    summary["rs_slow_interval_center"] = int(rs_schedule_config.rs_slow_interval)
    summary["rs_slow_interval_jitter"] = int(
        rs_schedule_config.rs_slow_interval_jitter
    )
    summary["rs_schedule_seed"] = int(getattr(args, "seed", 20260711))
    # probe 与 eval 保持同一评分边界：RS_SLOW 何时运行默认不看 GT，
    # 但“RS 真错时跳过 EVENT”只能在离线带标签 probe 中用 GT 实现。
    # 显式写入 results，避免将 deployable RS scheduler 误读成整条线上策略。
    summary["event_gate_policy"] = "offline_ground_truth_rs_correctness"
    summary["event_gate_uses_ground_truth"] = True
    summary["fully_deployable_end_to_end"] = False
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
        "format_version": 4,
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

    ``random`` 的预算单位是完整 route，用 ``--num-routes`` 控制；
    ``--num-cases`` 只控制 RS/UE 专项，且 UE 为了保留完整 span
    可超过该最小预算。``review`` 是人工巡检默认，``compact`` 适合
    静态合同快检，``full`` 只在需要拆分 prompt/时间线时使用。
    """

    p = argparse.ArgumentParser(description="Dump SFT v5 probe cases")
    p.add_argument("--index", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="checkpoints/sft_v5_probe")
    p.add_argument(
        "--num-cases",
        type=int,
        default=24,
        help="RS 的总帧预算；UE 为最小预算且不截断 span；random 模式忽略该参数",
    )
    p.add_argument(
        "--num-routes",
        type=int,
        default=1,
        help="random 模式随机抽取的完整 route ID 数；每个 ID 都测试全部帧",
    )
    p.add_argument(
        "--sequence-length",
        type=int,
        default=24,
        help="旧命令兼容参数；random 已改为完整 route ID，不再按该值截断",
    )
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
    p.add_argument(
        "--sample-mode",
        choices=("random", "rs_transition", "ue_transition"),
        default="random",
        help="小样本选帧：random 完整 route ID；RS 取变化前后；UE 取完整持续段及前后邻帧",
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
    p.add_argument("--rs-slow-interval", type=int, default=4)
    p.add_argument(
        "--rs-slow-interval-jitter",
        type=int,
        default=1,
        help="复核间隔在 center±jitter 中按 route/seed 可复现采样；默认 3/4/5",
    )
    p.add_argument(
        "--initial-memory",
        choices=("unknown", "ground_truth"),
        default="unknown",
        help="student 连续窗口首帧 memory；unknown 为默认无先验，ground_truth 仅复现旧 probe",
    )
    p.add_argument(
        "--rs-schedule-policy",
        choices=("deployable", "oracle"),
        default="deployable",
        help="deployable 不看 GT；oracle 用 GT 错误触发逐帧 RS，仅作诊断对照",
    )
    return p.parse_args()


def main() -> None:
    """CLI 入口：解析参数后生成 probe 证据与汇总。

    默认 ``review`` 生成逐帧 RGB + input/output/memory JSON；需要
    旧式拆分文件时显式选 ``full``。函数不负责比较多个
    checkpoint，训练入口会收集每次 ``dump_probe`` 返回的 summary
    并写 ``probes/comparison.json``。
    """

    dump_probe(parse_args())


if __name__ == "__main__":
    main()
