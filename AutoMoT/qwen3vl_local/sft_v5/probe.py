"""SFT v5 case-level probe。

默认不加载模型，只 dump prompt / target / label / memory / RGB，方便先人工检查候选池与
随机选项是否符合预期。传 `--with-model` 后会额外生成 student Q1/Q2 输出；传
`--with-teacher-model` 后会额外用默认/base Qwen 跑 privileged teacher prompt。
训练前 OPSD 能力体检必须不传 `--adapter-dir`，即 teacher/student 都只用普通 Qwen，
不导入任何 LoRA。

产物刻意仿照 sft_v3/probe.py 的组织方式：顶层 manifest、route 级 timeline、
frame 级 RGB/prompt/output/memory/flags。这样人工看 case 时不用在 v3/v5 之间切换
不同心智模型，只需要记住 v5 的 step1=Q1(RS+ABNORMAL)，step2=Q2(EVENT)。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_v5.eval import _generate_next, _generate_start, load_eval_bundle  # noqa: E402
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
    """复制 4 帧 RGB history 到 probe case 目录。"""

    copied: List[Dict[str, str]] = []
    for idx, src_text in enumerate(frame.history_rgb_paths):
        src = pathlib.Path(src_text)
        dst = frame_dir / f"rgb_{idx:02d}.jpg"
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
    (frame_dir / "rgb_paths.json").write_text(json.dumps(copied, ensure_ascii=False, indent=2), encoding="utf-8")
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
        if log.get("rs_wrong_reset"):
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
    return {
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


def dump_probe(
    args: argparse.Namespace,
    *,
    student_bundle: Optional[Any] = None,
    teacher_bundle: Optional[Any] = None,
    student_disable_adapter: bool = False,
    teacher_disable_adapter: bool = False,
) -> Dict[str, Any]:
    """生成 probe case 文件夹，并返回逐版本对比所需的汇总指标。

    ``student_bundle`` / ``teacher_bundle`` 仅供训练进程内自动 probe 使用。传入后不会
    重新加载 Qwen；checkpoint student 使用当前 LoRA，base 与 teacher 则在同一模型上
    临时关闭 adapter。普通 CLI 不传这两个参数时，仍保持原来的独立加载行为。
    """

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = RouteSequenceDataset(
        pathlib.Path(args.index),
        max_routes=int(args.max_routes),
        max_frames_per_route=int(args.max_frames_per_route),
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
    case_idx = 0
    for route_idx, route in enumerate(ds.rows):
        if case_idx >= int(args.num_cases):
            break
        # v3 probe 是 episode/frame 层级；v5 没有 sub-scenario episode 概念，所以用
        # route/frame 层级表达同一件事：一条 route 的 memory 随时间推进。
        route_dir = out_dir / f"route_{route_idx:03d}__{_safe_name(route.scenario)}__{_safe_name(route.route_id)}"
        route_dir.mkdir(parents=True, exist_ok=True)
        frame_logs: List[Dict[str, Any]] = []
        memory = None
        reset_next = False
        for frame in route.frames:
            if case_idx >= int(args.num_cases):
                break
            rs_target = _rs_target_from_frame(frame)
            event_target = _event_target_from_frame(frame)
            if memory is None or reset_next:
                # 训练/eval 口径：首帧或上帧非法/RS 错后，用 GT RS + RE 重置下一帧 memory。
                memory = _reset_memory_for_frame_row(frame)
                reset_next = False
            memory_at_frame_start = memory
            memory_before = _memory_json(memory)
            case_dir = route_dir / f"frame_{frame.frame_id:04d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            copied_rgb = _copy_rgb_inputs(frame, case_dir)
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

            memory_after_q1 = update_memory_after_q1(memory, student_rs_label=frame.rs_label, student_abnormal=frame.abnormal)
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
                    if q2_invalid:
                        reset_next = True
                else:
                    # RS 错误时本帧停止采样，下一帧恢复 GT RS + RE，不让错误 RS 污染事件判断。
                    q2_triggered = False
                    reset_next = True
                    memory = memory_after_q1
            else:
                # 静态 dump 模式不跑 student 生成；为了仍然能看到完整 Q2 prompt/target，
                # 这里使用 GT Q1 结果推进一次 memory，相当于 teacher-forced 可视化。
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

            # 默认 teacher input/output 文件必须一一对应。启用 teacher model 时，Q2
            # output 来自 teacher 自己的 Q1 KV，因此默认 q2_teacher_* 指向自主续接
            # prompt；训练时基于 student rollout 构造的 privileged prompt 另存为
            # q2_teacher_training_*，两者不能混用。
            q2_teacher_output_prompt = q2_teacher_model_prompt if teacher_bundle is not None else q2_teacher
            q2_teacher_output_target = q2_teacher_model_target if teacher_bundle is not None else q2_target
            files = {
                # v5 native names.
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
                # teacher model 自主 Q2 续接使用自己的 Q1 输出；与训练时基于 student
                # rollout 构造的 privileged prompt 分开保存，防止人工审计时误配上下文。
                "q2_teacher_model_prompt.txt": q2_teacher_model_prompt,
                "q2_teacher_model_target.txt": q2_teacher_model_target,
                # v3-style aliases for visual comparison tooling.
                # step1/step2 别名让已有的 v3 case 对比脚本可以直接看 v5 输出。
                "step1_user.txt": q1_student,
                "step1_teacher_user.txt": q1_teacher,
                "step1_teacher.txt": q1_target,
                "step2_user.txt": q2_student,
                "step2_teacher_user.txt": q2_teacher_output_prompt,
                "step2_teacher.txt": q2_teacher_output_target,
            }
            files["q1_student_output.txt"] = q1_output or ""
            files["q2_student_output.txt"] = q2_output or ""
            files["q1_teacher_output.txt"] = q1_teacher_output or ""
            files["q2_teacher_output.txt"] = q2_teacher_output or ""
            files["step1_student.txt"] = q1_output or ""
            files["step2_student.txt"] = q2_output or ""
            files["step1_teacher_output.txt"] = q1_teacher_output or ""
            files["step2_teacher_output.txt"] = q2_teacher_output or ""
            _write_texts(case_dir, files)
            _write_messages(case_dir, "q1_student", copied_rgb, q1_student)
            _write_messages(case_dir, "q1_teacher", copied_rgb, q1_teacher)
            _write_messages(case_dir, "q2_student", copied_rgb, q2_student)
            _write_messages(case_dir, "q2_teacher", copied_rgb, q2_teacher_output_prompt)
            _write_messages(case_dir, "q2_teacher_training", copied_rgb, q2_teacher)
            _write_messages(case_dir, "q2_teacher_model", copied_rgb, q2_teacher_model_prompt)

            labels = _frame_labels(route, frame)
            (case_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
            memory_after = _memory_json(memory)
            (case_dir / "memory_before.json").write_text(json.dumps(memory_before, ensure_ascii=False, indent=2), encoding="utf-8")
            (case_dir / "memory_after.json").write_text(json.dumps(memory_after, ensure_ascii=False, indent=2), encoding="utf-8")

            frame_log = {
                **labels,
                # flags.json 聚合三类信息：
                # 1) label/source/candidate 证据；
                # 2) student/teacher 解析结果；
                # 3) 状态机诊断，例如 Q1 是否截断、Q2 是否非法、下一帧是否 reset。
                "case_index": case_idx,
                "case_dir": str(case_dir),
                "copied_rgb": copied_rgb,
                "teacher_forced": bundle is None,
                "teacher_model_enabled": teacher_bundle is not None,
                "teacher_model_dir": str(pathlib.Path(args.teacher_model_dir or args.model_dir)) if teacher_bundle is not None else None,
                "student_adapter_dir": str(pathlib.Path(args.adapter_dir)) if args.adapter_dir else None,
                "student_adapter_enabled": bool(bundle is not None and not student_disable_adapter and args.adapter_dir),
                "student_base_mode": bool(bundle is not None and (student_disable_adapter or not args.adapter_dir)),
                "memory_before": memory_before,
                "memory_after": memory_after,
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
                "rs_wrong_reset": not q1_rs_ok,
                "reset_next": reset_next,
            }
            # flags.json 是逐帧快速诊断入口；timeline.json/png 只聚合其中几个关键字段。
            (case_dir / "flags.json").write_text(json.dumps(frame_log, ensure_ascii=False, indent=2), encoding="utf-8")
            frame_logs.append(frame_log)
            all_frame_logs.append(frame_log)
            case_idx += 1
        (route_dir / "timeline.json").write_text(json.dumps(frame_logs, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_timeline_png(route_dir / "timeline.png", frame_logs)
        manifest.append(
            {
                "route_dir": str(route_dir),
                "scenario": route.scenario,
                "route_id": route.route_id,
                "frames": len(frame_logs),
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize_probe(
        all_frame_logs,
        student_enabled=bundle is not None,
        teacher_enabled=teacher_bundle is not None,
        student_adapter_dir=str(pathlib.Path(args.adapter_dir)) if args.adapter_dir else None,
        student_disable_adapter=bool(student_disable_adapter),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump SFT v5 probe cases")
    p.add_argument("--index", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="checkpoints/sft_v5_probe")
    p.add_argument("--num-cases", type=int, default=24)
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
    p.add_argument("--with-model", action="store_true")
    p.add_argument("--with-teacher", action="store_true", help="compat flag: v5 always dumps teacher prompt/target")
    p.add_argument("--with-teacher-model", action="store_true", help="load base Qwen without LoRA to generate privileged teacher outputs")
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--teacher-model-dir", type=str, default=None, help="optional base Qwen dir for teacher generation; defaults to --model-dir")
    p.add_argument("--adapter-dir", type=str, default=None)
    p.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-new-tokens-q1", type=int, default=256)
    p.add_argument("--max-new-tokens-q2", type=int, default=192)
    return p.parse_args()


def main() -> None:
    dump_probe(parse_args())


if __name__ == "__main__":
    main()
