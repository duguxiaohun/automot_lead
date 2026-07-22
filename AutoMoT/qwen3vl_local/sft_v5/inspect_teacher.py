"""SFT v5 teacher prompt 合同抽检。

本脚本默认不加载模型，只检查：
- XML weather 是否只进入 teacher prompt；
- teacher target 是否没有 ANSWER_/REFERENCE/XML_WEATHER 泄漏；
- Q2 option map 是否 frame 级随机且可解析。

本入口是纯 CPU 静态检查，不生成 teacher 文本。使用 ``--index`` 指向 train/val
sequence index，报告写入 ``--output-dir``，适合在改 prompt 后、加载模型前先运行。

这里审计的“teacher”是 OPSD 的 privileged prompt/target 协议，不是另一个
常驻模型进程。Q1 teacher 可读 XML weather 和 GT RS 来组织分析，
但脚本化 target 必须清洗回学生视角；Q2 teacher 可读 GT EVENT，
但学生看到的仍是标清 ``[RE | REGULAR]`` / ``[UE | UNUSUAL]``
的合并选择题，没有独立 NORMAL/ABNORMAL 一问。真实 base-teacher CoT
生成能力需用 ``probe.py --with-teacher-model`` 另行审计。
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
    """抽取前 ``num_cases`` 帧，审计 student/teacher 隔离合同。

    输入是 ``RouteSequenceDataset`` 的 train/val index，可用 ``max_routes``
    和 ``max_frames_per_route`` 先缩小候选集。每帧都用同一初始 memory
    构造 Q1，再用 GT RS 做一次 teacher-forced Q1 memory 转换后构造 Q2；
    这只是静态 prompt/target 合同检查，不模拟 closed-loop 学生输出。

    返回字典中 ``checked`` 是实际检查帧数，``bad`` 是任一硬合同
    失败的帧数，``rows`` 保留逐帧 bool 检查项。同一内容写入
    ``teacher_report.json``，Markdown 只展开前 50 帧便于人工快速浏览。
    """

    ds = RouteSequenceDataset(
        pathlib.Path(args.index),
        max_routes=int(args.max_routes),
        max_frames_per_route=int(args.max_frames_per_route),
    )
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # rows 只保留短小的结构化检查结果，不把全量 prompt/target
    # 复制进报告。需要逐帧查阅完整文本时应使用 probe full artifact。
    rows = []
    bad = 0
    count = 0
    for route in ds.rows:
        for frame in route.frames:
            if count >= int(args.num_cases):
                break
            # target helper 保持与 train/eval/probe 同一标签解析口径。
            # EVENT 仍可能有多个可接受 UE；此处审计候选展示与
            # private-marker 清洗，不评测模型选中哪个。
            rs_target = _rs_target_from_frame(frame)
            event_target = _event_target_from_frame(frame)
            # 每个 case 独立初始化：inspect 要查 prompt 泄漏，不应让前一
            # case 的 memory 对当前字符串造成额外影响。EGO_TO_GOAL_XY 作为
            # 学生合法可见导航条件正常保留。
            memory = reset_memory_for_frame(rs_target, ego_to_goal_xy=frame.ego_to_goal_xy)
            q1_student = build_q1_student_prompt(memory)
            q1_teacher = build_q1_teacher_prompt(
                memory,
                rs_target=rs_target,
                weather_text=frame.weather_text,
            )
            q1_target = build_q1_teacher_target(
                rs_target=rs_target,
                weather_text=frame.weather_text,
            )
            # Q2 的候选空间受 RS gate 约束。静态审计不生成 Q1，所以
            # 显式 teacher-force GT RS 构造合法 Q2 prompt；这不代表推理时
            # 会纠正 student memory。真实 eval/probe 中 RS 错的当帧会跳过 Q2。
            memory_after_q1 = update_memory_after_q1(memory, student_rs_label=frame.rs_label)
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
            # 每项检查都是可独立定位的硬合同。有些 key 命名是
            # “是否发现泄漏”（期望 False），有些是“是否满足要求”
            # （期望 True）；下面 ok 显式列出方向，不用 all(checks.values())
            # 混淆语义。
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
                "event_choices_mark_re_ue": all(
                    f"{letter}. [{'RE | REGULAR' if label == 'RE' else 'UE | UNUSUAL'}]"
                    in q2_student
                    for letter, label in frame.event_option_map.items()
                ),
                "q2_student_has_scenario_name": route.scenario in q2_student,
            }
            ok = (
                not checks["q1_student_has_xml_weather"]
                and checks["q1_teacher_has_xml_weather"]
                and not checks["q1_target_contains_weather_text"]
                and checks["q1_target_private_clean"]
                and checks["q2_target_private_clean"]
                and checks["option_map_nonempty"]
                and checks["event_choices_mark_re_ue"]
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
    # JSON 是机器可读真值；Markdown 只是快速巡检索引，所以限制
    # 50 条避免大 index 时文档膨胀。两份文件都不包含模型生成文本。
    report = {"checked": count, "bad": bad, "rows": rows}
    with open(out_dir / "teacher_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_dir / "teacher_report.md", "w", encoding="utf-8") as f:
        f.write(f"# SFT v5 teacher inspect\n\nchecked={count}, bad={bad}\n")
        for row in rows[:50]:
            f.write(f"\n- {row['scenario']}/{row['route_id']} f{row['frame_id']}: ok={row['ok']} checks={row['checks']}\n")
    return report


def parse_args() -> argparse.Namespace:
    """解析静态 teacher 合同抽检参数。

    ``--index`` 必填；``--num-cases`` 是跨 route 的全局帧上限。
    ``--max-routes``/``--max-frames-per-route`` 先在 dataset 层裁剪，便于
    仅针对小型 smoke index 或特定前缀快检。
    """

    p = argparse.ArgumentParser(description="Inspect SFT v5 teacher prompt contract")
    p.add_argument("--index", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="checkpoints/sft_v5_teacher_inspect")
    p.add_argument("--num-cases", type=int, default=64)
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
    return p.parse_args()


def main() -> None:
    """运行抽检并打印总 case 数与失败数。

    详细失败项已写入 output dir，终端只打印 ``checked/bad``，
    方便 CI 或 shell 人工快速判断是否需要打开报告。
    """

    report = inspect(parse_args())
    print(json.dumps({"checked": report["checked"], "bad": report["bad"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
