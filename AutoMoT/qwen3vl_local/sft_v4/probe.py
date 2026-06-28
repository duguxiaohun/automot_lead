"""SFT v4 case probe。

按 episode/帧导出：
- 三步 prompt 与 student 输出；
- 可选 teacher privileged prompt / teacher 输出；
- memory 前后状态；
- 关键标志（road-structure / scene flip、step2/step3 trigger、teacher BLEU）；
- 4 帧输入图片。

probe 不是评估主指标入口，而是 case-level 调试工具。它保留完整文本，方便人工检查
memory 是否在错误场景下被纠正、scene 正确后 step3 是否合理推进 status/subgoal。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
import sys
from typing import Any, Dict, List

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 先导入 eval：它的模块顶部会在 torch/engine/train 导入前完成 GPU 选址。
from qwen3vl_local.sft_v4.eval import _generate, _generate_next_with_kv, _simple_bleu
from qwen3vl_local.engine import LocalQwen3VLInstructEngine
from qwen3vl_local.sft_v4.train import (
    EpisodeDataset,
    _analysis_before_labels,
    _build_messages_with_images,
    _build_rgb_paths,
    _gt_status_subgoal,
    _load_goal_xy,
    _load_images,
    _prefetch_goal_xy_for_next_frame,
)
from qwen3vl_local.sft_v4.prompts import (
    get_step_system_prompt,
    TEACHER_MAX_NEW_TOKENS_STEP1,
    TEACHER_MAX_NEW_TOKENS_STEP2,
    TEACHER_MAX_NEW_TOKENS_STEP3,
    build_step1_teacher_prompt,
    build_step1_teacher_target,
    build_step1_user_prompt,
    build_step2_student_prompt,
    build_step2_teacher_prompt,
    build_step2_teacher_target,
    build_step3_student_prompt,
    build_step3_teacher_prompt,
    build_step3_teacher_target,
    get_road_structure,
    init_memory,
    parse_output,
    should_trigger_step2,
    should_trigger_step3,
    step1_teacher_verdict,
    step2_teacher_verdict,
    step3_teacher_verdict,
    update_memory_after_step1,
    update_memory_after_step2,
    update_memory_after_step3,
    validate_event,
)


def parse_args() -> argparse.Namespace:
    """解析 case probe 参数。"""

    p = argparse.ArgumentParser(description="Probe SFT v4 cases")
    p.add_argument("--jsonl", type=str, default="checkpoints/sft_v4_data/val.jsonl")
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--lora-dir", type=str, default="checkpoints/sft_v4_lora/latest/final")
    p.add_argument("--save-root", type=str, default="checkpoints/sft_v4_lora/latest")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-episodes", type=int, default=4)
    p.add_argument("--max-gen-tokens", type=int, default=80)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--with-teacher", action="store_true",
                   help="额外加载一份 base Qwen teacher，dump privileged prompts/text 和 analysis BLEU")
    return p.parse_args()


def _write_timeline_png(path: pathlib.Path, frame_logs: List[Dict[str, Any]]) -> None:
    """写一个轻量时间线图，橙色=road-structure flip，红色=scene flip，蓝色=step3 trigger。

    这张图只帮助快速定位“哪几帧发生场景翻转/进入 step3”，不依赖 matplotlib，
    远端环境缺绘图库时也能生成。
    """

    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    width = max(320, 12 * max(len(frame_logs), 1))
    height = 72
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = height // 2
    draw.line((12, y, width - 12, y), fill=(80, 80, 80), width=2)
    n = max(len(frame_logs) - 1, 1)
    for i, log in enumerate(frame_logs):
        x = 12 + int((width - 24) * i / n)
        if log.get("road_structure_flip"):
            color = (235, 140, 35)
            r = 5
        elif log.get("scene_flip"):
            color = (210, 40, 40)
            r = 5
        elif log.get("step3_trigger"):
            color = (45, 105, 210)
            r = 4
        else:
            color = (150, 150, 150)
            r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    img.save(path)


def main() -> None:
    """case dump 主入口。

    默认只加载 student；`--with-teacher` 会额外加载 base Qwen teacher，并把 teacher
    看到的 privileged prompt 也写出，便于确认 teacher 分析是否真的在纠正学生 memory。
    """

    args = parse_args()
    ds = EpisodeDataset(pathlib.Path(args.jsonl))
    rows = list(ds.rows)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: max(1, args.num_episodes)]

    out_dir = pathlib.Path(args.save_root) / "probe_v4"
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
    if args.with_teacher:
        # probe 的 teacher 使用独立 engine，避免和 student LoRA/merge 状态互相影响。
        teacher_engine = LocalQwen3VLInstructEngine(
            pathlib.Path(args.model_dir),
            device=args.device,
            max_gen_tokens=args.max_gen_tokens,
            temperature=0.0,
            do_sample=False,
            repetition_penalty=args.repetition_penalty,
        )
        teacher_engine.load()

    manifest: List[Dict[str, Any]] = []

    for epi_idx, ep in enumerate(rows):
        ep_dir = out_dir / f"episode_{epi_idx:03d}__{ep.scenario}__{ep.run_id}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        run_dir = pathlib.Path(ep.run_dir)
        gx, gy = _load_goal_xy(run_dir, ep.frame_start)
        gt_road_structure = get_road_structure(ep.gt_scene)
        memory = init_memory(
            run_id=ep.run_id,
            sub_scenario_id=f"{ep.run_id}:{ep.anchors[1]}",
            ego_to_goal_x=gx,
            ego_to_goal_y=gy,
            gt_scene=ep.gt_scene,
        )
        episode_meta = {
            "run_id": ep.run_id,
            "scenario": ep.scenario,
            "raw_gt_scene": ep.raw_gt_scene,
            "anchors": [int(x) for x in ep.anchors],
            "delta": int(ep.delta),
            "frame_range": [int(ep.frame_start), int(ep.frame_end)],
            "gt_road_structure": gt_road_structure,
            "gt_scene": ep.gt_scene,
            "gt_event_sequence": list(ep.gt_event_sequence),
            "run_dir": ep.run_dir,
        }
        (ep_dir / "episode_meta.json").write_text(
            json.dumps(episode_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        frame_logs: List[Dict[str, Any]] = []

        for frame in range(ep.frame_start, ep.frame_end + 1):
            frame_dir = ep_dir / f"frame_{frame:04d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            phase = "phase_a" if ep.anchors[1] - ep.delta <= frame <= ep.anchors[1] + ep.delta else "phase_b"

            image_paths = _build_rgb_paths(run_dir, frame)
            try:
                images = _load_images(image_paths)
            except Exception:
                _prefetch_goal_xy_for_next_frame(memory, run_dir, frame + 1, ep.frame_end)
                continue

            for i, src in enumerate(image_paths):
                try:
                    dst = frame_dir / f"rgb_{i:02d}.jpg"
                    shutil.copy2(src, dst)
                except Exception:
                    # 不破坏原文件，失败则忽略复制
                    pass

            gt_status, gt_subgoal = _gt_status_subgoal(ep, frame)

            memory_before = {
                "road_structure": memory.road_structure,
                "scene": memory.scene,
                "status": memory.status,
                "subgoal": memory.subgoal,
                "goal_x": memory.ego_to_goal_x,
                "goal_y": memory.ego_to_goal_y,
            }

            step1_user = build_step1_user_prompt(len(images), memory=memory)
            step1_msgs = _build_messages_with_images(user_text=step1_user, images=images, system_prompt=get_step_system_prompt("STEP1"))
            step1_teacher_text = ""
            step1_teacher_user = ""
            if teacher_engine is not None:
                step1_teacher_user = build_step1_teacher_prompt(memory, gt_road_structure)
                step1_teacher_msgs = _build_messages_with_images(user_text=step1_teacher_user, images=images, system_prompt=get_step_system_prompt("STEP1"))
                raw_t1 = _generate(
                    teacher_engine,
                    step1_teacher_msgs,
                    images,
                    max_new_tokens=TEACHER_MAX_NEW_TOKENS_STEP1,
                )
                analysis_t1 = _analysis_before_labels(raw_t1)
                step1_verdict = step1_teacher_verdict(memory, gt_road_structure)
                step1_teacher_text = build_step1_teacher_target(
                    analysis_t1,
                    gt_road_structure,
                    verdict=step1_verdict,
                )
            step1_text = _generate(engine, step1_msgs, images, max_new_tokens=TEACHER_MAX_NEW_TOKENS_STEP1)

            step2_teacher_text = ""
            step2_teacher_user = ""
            step1_bleu = None
            step2_bleu = None
            step3_bleu = None
            if teacher_engine is not None:
                step1_bleu = _simple_bleu(
                    _analysis_before_labels(step1_text),
                    _analysis_before_labels(step1_teacher_text),
                )

            p1 = parse_output(step1_text)
            old_rs = memory.road_structure
            memory = update_memory_after_step1(memory, student_road_structure=p1.get("road_structure"))
            road_structure_flip = memory.road_structure != old_rs
            road_structure_ok = memory.road_structure == gt_road_structure
            step2_trigger = should_trigger_step2(
                memory_road_structure_before_step1=old_rs,
                memory_road_structure_after_step1=memory.road_structure,
                gt_road_structure=gt_road_structure,
            )

            step2_user = ""
            step2_text = ""
            scene_flip = False
            if step2_trigger:
                if teacher_engine is not None:
                    # teacher step2 是独立专家问答：重新吃图 + road/scene 真值上下文。
                    step2_teacher_user = build_step2_teacher_prompt(memory, gt_road_structure, ep.gt_scene)
                    step2_teacher_msgs = _build_messages_with_images(user_text=step2_teacher_user, images=images, system_prompt=get_step_system_prompt("STEP2"))
                    raw_t2 = _generate(
                        teacher_engine,
                        step2_teacher_msgs,
                        images,
                        max_new_tokens=TEACHER_MAX_NEW_TOKENS_STEP2,
                    )
                    analysis_t2 = _analysis_before_labels(raw_t2)
                    step2_verdict = step2_teacher_verdict(memory, ep.gt_scene)
                    step2_teacher_text = build_step2_teacher_target(
                        analysis_t2,
                        ep.gt_scene,
                        verdict=step2_verdict,
                    )

                step2_user = build_step2_student_prompt(memory)
                step2_text = _generate_next_with_kv(engine, step2_user, max_new_tokens=TEACHER_MAX_NEW_TOKENS_STEP2)
                p2 = parse_output(step2_text)
                if teacher_engine is not None:
                    step2_bleu = _simple_bleu(
                        _analysis_before_labels(step2_text),
                        _analysis_before_labels(step2_teacher_text),
                    )

                old_scene = memory.scene
                memory = update_memory_after_step2(memory, student_scene=p2.get("scene"))
                scene_flip = memory.scene != old_scene
            step3_trigger = (
                step2_trigger
                and should_trigger_step3(
                    memory_scene_before_step2=old_scene,
                    memory_scene_after_step2=memory.scene,
                    gt_scene=ep.gt_scene,
                )
            )

            step3_user = ""
            step3_text = ""
            step3_teacher_text = ""
            step3_teacher_user = ""
            if step3_trigger:
                if teacher_engine is not None:
                    # teacher step3 是独立专家问答：重新吃图 + road/scene/status/subgoal 真值上下文。
                    step3_teacher_user = build_step3_teacher_prompt(
                        memory,
                        gt_road_structure,
                        ep.gt_scene,
                        gt_status,
                        gt_subgoal,
                    )
                    step3_teacher_msgs = _build_messages_with_images(user_text=step3_teacher_user, images=images, system_prompt=get_step_system_prompt("STEP3"))
                    raw_t3 = _generate(
                        teacher_engine,
                        step3_teacher_msgs,
                        images,
                        max_new_tokens=TEACHER_MAX_NEW_TOKENS_STEP3,
                    )
                    analysis_t3 = _analysis_before_labels(raw_t3)
                    step3_verdict = step3_teacher_verdict(memory, ep.gt_scene, gt_status, gt_subgoal)
                    step3_teacher_text = build_step3_teacher_target(
                        analysis_t3,
                        gt_status,
                        gt_subgoal,
                        verdict=step3_verdict,
                    )
                step3_user = build_step3_student_prompt(memory)
                step3_text = _generate_next_with_kv(engine, step3_user, max_new_tokens=TEACHER_MAX_NEW_TOKENS_STEP3)
                if teacher_engine is not None:
                    step3_bleu = _simple_bleu(
                        _analysis_before_labels(step3_text),
                        _analysis_before_labels(step3_teacher_text),
                    )
                p3 = parse_output(step3_text)
                pred_status = p3.get("status") if validate_event(memory.scene, p3.get("status")) else None
                pred_subgoal = p3.get("subgoal") if validate_event(memory.scene, p3.get("subgoal")) else None
                memory = update_memory_after_step3(memory, student_status=pred_status, student_subgoal=pred_subgoal)

            memory_after = {
                "road_structure": memory.road_structure,
                "scene": memory.scene,
                "status": memory.status,
                "subgoal": memory.subgoal,
                "goal_x": memory.ego_to_goal_x,
                "goal_y": memory.ego_to_goal_y,
            }

            (frame_dir / "step1_user.txt").write_text(step1_user, encoding="utf-8")
            (frame_dir / "step1_prompt.txt").write_text(step1_user, encoding="utf-8")
            (frame_dir / "step1_teacher_user.txt").write_text(step1_teacher_user, encoding="utf-8")
            (frame_dir / "step1_student.txt").write_text(step1_text, encoding="utf-8")
            (frame_dir / "step1_teacher.txt").write_text(step1_teacher_text, encoding="utf-8")
            (frame_dir / "step2_user.txt").write_text(step2_user, encoding="utf-8")
            (frame_dir / "step2_prompt.txt").write_text(step2_user, encoding="utf-8")
            (frame_dir / "step2_teacher_user.txt").write_text(step2_teacher_user, encoding="utf-8")
            (frame_dir / "step2_student.txt").write_text(step2_text, encoding="utf-8")
            (frame_dir / "step2_teacher.txt").write_text(step2_teacher_text, encoding="utf-8")
            (frame_dir / "step3_user.txt").write_text(step3_user, encoding="utf-8")
            (frame_dir / "step3_prompt.txt").write_text(step3_user, encoding="utf-8")
            (frame_dir / "step3_teacher_user.txt").write_text(step3_teacher_user, encoding="utf-8")
            (frame_dir / "step3_student.txt").write_text(step3_text, encoding="utf-8")
            (frame_dir / "step3_teacher.txt").write_text(step3_teacher_text, encoding="utf-8")

            frame_log = {
                "frame": frame,
                "phase": phase,
                "gt_road_structure": gt_road_structure,
                "gt_scene": ep.gt_scene,
                "raw_gt_scene": ep.raw_gt_scene,
                "gt_status": gt_status,
                "gt_subgoal": gt_subgoal,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "road_structure_ok": road_structure_ok,
                "road_structure_flip": road_structure_flip,
                "step2_trigger": step2_trigger,
                "scene_flip": scene_flip,
                "step3_trigger": step3_trigger,
                "analysis_bleu_vs_teacher": {
                    "step1": step1_bleu,
                    "step2": step2_bleu,
                    "step3": step3_bleu,
                },
            }
            (frame_dir / "memory_before.json").write_text(
                json.dumps(memory_before, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (frame_dir / "memory_after.json").write_text(
                json.dumps(memory_after, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (frame_dir / "flags.json").write_text(
                json.dumps(frame_log, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            frame_logs.append(frame_log)
            _prefetch_goal_xy_for_next_frame(memory, run_dir, frame + 1, ep.frame_end)

        (ep_dir / "timeline.json").write_text(json.dumps(frame_logs, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_timeline_png(ep_dir / "timeline.png", frame_logs)
        manifest.append(
            {
                "episode_dir": str(ep_dir),
                "scenario": ep.scenario,
                "run_id": ep.run_id,
                "frames": len(frame_logs),
            }
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_dir}")


if __name__ == "__main__":
    main()
