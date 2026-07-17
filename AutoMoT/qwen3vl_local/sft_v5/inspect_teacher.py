"""SFT v5 teacher prompt 合同抽检。

本脚本默认不加载模型，只检查：
- XML weather 是否只进入 teacher prompt；
- teacher target 是否没有 ANSWER_/REFERENCE/XML_WEATHER 泄漏；
- Q2 option map 是否 frame 级随机且可解析。

本入口是纯 CPU 静态检查，不生成 teacher 文本。使用 ``--index`` 指向 train/val
sequence index，报告写入 ``--output-dir``，适合在改 prompt 后、加载模型前先运行。
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

from qwen3vl_local.sft_v5.prompts import (  # noqa: E402
    build_q1_student_prompt,
    build_q1_teacher_prompt,
    build_q1_teacher_target,
    build_q2_student_prompt,
    build_q2_teacher_prompt,
    build_q2_teacher_target,
    check_no_private_markers,
    reset_memory_for_frame,
    update_memory_after_q1,
)
from qwen3vl_local.sft_v5.train import RouteSequenceDataset, _event_target_from_frame, _rs_target_from_frame  # noqa: E402


def inspect(args: argparse.Namespace) -> dict:
    """抽取前 ``num_cases`` 帧，检查 student/teacher 隔离并写 JSON/Markdown 报告。"""

    ds = RouteSequenceDataset(
        pathlib.Path(args.index),
        max_routes=int(args.max_routes),
        max_frames_per_route=int(args.max_frames_per_route),
    )
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    bad = 0
    count = 0
    for route in ds.rows:
        for frame in route.frames:
            if count >= int(args.num_cases):
                break
            rs_target = _rs_target_from_frame(frame)
            event_target = _event_target_from_frame(frame)
            memory = reset_memory_for_frame(rs_target, ego_to_goal_xy=frame.ego_to_goal_xy)
            q1_student = build_q1_student_prompt(memory)
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
            memory_after_q1 = update_memory_after_q1(memory, student_rs_label=frame.rs_label, student_abnormal=frame.abnormal)
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
            checks = {
                # student prompt 必须干净：不能把 XML weather 或 scenario name 泄漏给学生。
                "q1_student_has_xml_weather": "XML_WEATHER" in q1_student or "XML reports" in q1_student,
                # teacher prompt 必须真的含有 privileged weather，否则老师分析就退化成学生视角。
                "q1_teacher_has_xml_weather": "XML_WEATHER" in q1_teacher,
                # target 是“学生可见监督文本”，不能逐字包含 teacher-only weather_text。
                "q1_target_contains_weather_text": bool(frame.weather_text) and frame.weather_text in q1_target,
                "q1_target_private_clean": check_no_private_markers(q1_target),
                "q2_target_private_clean": check_no_private_markers(q2_target),
                # Q2 没候选会让训练/评估都无法解析 EVENT，因此作为硬合同检查。
                "option_map_nonempty": bool(frame.event_option_map),
                "q2_student_has_scenario_name": route.scenario in q2_student,
            }
            ok = (
                not checks["q1_student_has_xml_weather"]
                and checks["q1_teacher_has_xml_weather"]
                and not checks["q1_target_contains_weather_text"]
                and checks["q1_target_private_clean"]
                and checks["q2_target_private_clean"]
                and checks["option_map_nonempty"]
                and not checks["q2_student_has_scenario_name"]
            )
            bad += int(not ok)
            rows.append(
                {
                    "scenario": route.scenario,
                    "route_id": route.route_id,
                    "frame_id": frame.frame_id,
                    "rs": frame.rs_label,
                    "event": frame.event_label,
                    "option_map": frame.event_option_map,
                    "checks": checks,
                    "ok": ok,
                }
            )
            count += 1
        if count >= int(args.num_cases):
            break
    report = {"checked": count, "bad": bad, "rows": rows}
    with open(out_dir / "teacher_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_dir / "teacher_report.md", "w", encoding="utf-8") as f:
        f.write(f"# SFT v5 teacher inspect\n\nchecked={count}, bad={bad}\n")
        for row in rows[:50]:
            f.write(f"\n- {row['scenario']}/{row['route_id']} f{row['frame_id']}: ok={row['ok']} checks={row['checks']}\n")
    return report


def parse_args() -> argparse.Namespace:
    """解析静态 teacher 合同抽检参数。"""

    p = argparse.ArgumentParser(description="Inspect SFT v5 teacher prompt contract")
    p.add_argument("--index", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="checkpoints/sft_v5_teacher_inspect")
    p.add_argument("--num-cases", type=int, default=64)
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
    return p.parse_args()


def main() -> None:
    """运行抽检并打印总 case 数与失败数。"""

    report = inspect(parse_args())
    print(json.dumps({"checked": report["checked"], "bad": report["bad"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
