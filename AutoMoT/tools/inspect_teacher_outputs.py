"""可视化检查 v2 teacher 产物。

默认模式 A：只读已经生成好的 v2 jsonl，按场景均匀抽样，落盘 case 目录。
可选模式 B：加 --live，现场重跑 teacher 推理，额外保存 teacher_raw 与后处理信息。

目标：让你快速判断 teacher ANALYSIS 是否符合预期（看图 -> 变化 -> 结论），
并确认 student 看到的三段 GT 与 teacher 文本一致。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[1]
_PROJECT_ROOT = _THIS_FILE.parents[2]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# live 模式下需要确保离线。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


_STUDENT_TAIL_MARKER = "Given the observations above and the memory context"

_TEACHER_SYSTEM_PROMPT = """You are a vision-grounded annotation teacher for an autonomous driving status-tracking task.

Input:
- 4 RGB frames (oldest -> newest), stitched three-camera view.
- MEMORY: the previous anchor (anchor-K) STATUS and EVENT_SEQUENCE.
- PRIVILEGED: the ground-truth current STATUS at the newest frame, and whether this anchor is KEEP (state unchanged) or ADVANCE (state moved forward from MEMORY STATUS).

Task:
Produce a single line of ANALYSIS that a student model (which does NOT see PRIVILEGED) could plausibly infer from images alone. Sentence order MUST be:
1. First sentence: concretely describe what is visible in the LAST frame.
2. Second sentence: describe what CHANGED between the earliest and the latest frame.
3. Third sentence: state whether the observed evidence supports staying at MEMORY STATUS or advancing to the current STATUS, tying it to the visual evidence above.

Constraints:
- Do NOT mention or reference the PRIVILEGED block; write as if from images only.
- Do NOT invent visual content not actually present.
- Be concise, grounded, factual; 2-4 sentences total, all on a single line.
- Do NOT output STATUS or SUBGOAL; only the ANALYSIS body text (no "ANALYSIS:" prefix).

Output EXACTLY one line of text (the ANALYSIS body, no prefix, no trailing newline)."""

_PREFIX_PATTERN = re.compile(r"^\s*ANALYSIS\s*:\s*", re.IGNORECASE)
_STOP_MARKERS = ("\nSTATUS:", "\nSUBGOAL:", "\n\n", "<|im_end|>")


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def strip_image_placeholders(user_content: str) -> str:
    s = user_content.lstrip()
    while s.startswith("<image>"):
        s = s[len("<image>"):]
    return s.lstrip("\n")


def parse_student_assistant(text: str) -> Dict[str, Optional[str]]:
    a = re.search(r"^ANALYSIS:\s*(.*)$", text, flags=re.MULTILINE)
    s = re.search(r"^STATUS:\s*(.+)$", text, flags=re.MULTILINE)
    g = re.search(r"^SUBGOAL:\s*(.+)$", text, flags=re.MULTILINE)
    return {
        "analysis": a.group(1).strip() if a else None,
        "status": s.group(1).strip() if s else None,
        "subgoal": g.group(1).strip() if g else None,
    }


def build_teacher_user_prompt(student_user_no_image: str, meta: Dict[str, Any]) -> str:
    target_status = meta.get("target_status") or "unknown"
    transition = meta.get("transition") or "UNKNOWN"
    prev_status = meta.get("memory_in_status") or "unknown"

    privileged = (
        "\n[PRIVILEGED]\n"
        f"CURRENT_GT_STATUS: {target_status}\n"
        f"TRANSITION: {transition}\n"
        f"PREV_STATUS: {prev_status}\n"
        "[/PRIVILEGED]\n\n"
        "Given the observations, memory, and privileged ground truth, "
        "output the ANALYSIS body that the student should plausibly produce from images alone."
    )

    idx = student_user_no_image.find(_STUDENT_TAIL_MARKER)
    if idx >= 0:
        return student_user_no_image[:idx].rstrip() + privileged
    return student_user_no_image.rstrip() + privileged


def postprocess_teacher(raw_text: str) -> Dict[str, Any]:
    if raw_text is None:
        raw_text = ""

    info: Dict[str, Any] = {
        "raw_chars": len(raw_text),
        "removed_prefix": False,
        "stop_marker": None,
        "newline_collapsed": False,
        "truncated_480": False,
        "fallback": False,
    }

    t = raw_text.strip()
    new_t = _PREFIX_PATTERN.sub("", t)
    if new_t != t:
        info["removed_prefix"] = True
    t = new_t

    cut = len(t)
    chosen: Optional[str] = None
    for stop in _STOP_MARKERS:
        i = t.find(stop)
        if i >= 0 and i < cut:
            cut = i
            chosen = stop
    if chosen is not None:
        info["stop_marker"] = chosen
    t = t[:cut]

    collapsed = re.sub(r"\s+", " ", t).strip()
    if collapsed != t:
        info["newline_collapsed"] = True
    t = collapsed

    if len(t) > 480:
        cut_pos = t.rfind(" ", 0, 480)
        t = t[: cut_pos if cut_pos > 0 else 480].rstrip()
        info["truncated_480"] = True

    if len(t) < 20:
        t = "Observations recorded."
        info["fallback"] = True

    info["cleaned_chars"] = len(t)
    info["cleaned"] = t
    return info


def sample_balanced_by_scenario(
    rows: List[Dict[str, Any]],
    num_per_scenario: int,
    seed: int,
    scenarios: Optional[Sequence[str]],
) -> List[Tuple[int, Dict[str, Any]]]:
    by_scenario: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    allow = set(scenarios) if scenarios else None

    for idx, row in enumerate(rows):
        sc = row.get("scenario", "unknown")
        if allow is not None and sc not in allow:
            continue
        by_scenario[sc].append((idx, row))

    if not by_scenario:
        raise RuntimeError("未找到可采样样本，请检查 --scenarios 或 jsonl")

    rng = random.Random(seed)
    picked: List[Tuple[int, Dict[str, Any]]] = []
    for sc in sorted(by_scenario.keys()):
        bucket = by_scenario[sc]
        rng.shuffle(bucket)
        picked.extend(bucket[:num_per_scenario])
    return picked


def copy_images(image_paths: Sequence[str], out_dir: pathlib.Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for i, src in enumerate(image_paths):
        src_path = pathlib.Path(src)
        if not src_path.exists():
            continue
        ext = src_path.suffix.lower() or ".jpg"
        dst_name = f"{i:02d}{ext}"
        dst = out_dir / dst_name
        shutil.copyfile(src_path, dst)
        copied.append(dst_name)
    return copied


def render_overview(
    row: Dict[str, Any],
    sample_idx: int,
    teacher_analysis: str,
    student_assistant: str,
    teacher_user: str,
    copied_images: Sequence[str],
    meta: Dict[str, Any],
    live_raw: Optional[str],
    live_post: Optional[Dict[str, Any]],
) -> str:
    parsed = parse_student_assistant(student_assistant)
    lines: List[str] = []
    lines.append(f"# Teacher Inspect Case: {row.get('scenario')}/{row.get('run_id')} anchor={row.get('anchor')}")
    lines.append("")
    lines.append(f"- sample_idx: {sample_idx}")
    lines.append(f"- mode: {meta.get('mode')}")
    lines.append(f"- transition: {meta.get('transition')}")
    lines.append(f"- target_status: {meta.get('target_status')}")
    lines.append(f"- teacher_fallback_flag: {meta.get('teacher_fallback_flag')}")
    lines.append(f"- analysis_chars: {len(teacher_analysis)}")
    lines.append("")

    lines.append("## GT 对照")
    lines.append("| 字段 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| STATUS | {parsed.get('status')} |")
    lines.append(f"| SUBGOAL | {parsed.get('subgoal')} |")
    lines.append(f"| ANALYSIS | {teacher_analysis} |")
    lines.append("")

    if copied_images:
        lines.append("## 输入图像（oldest -> newest）")
        lines.append("| 00 | 01 | 02 | 03 |")
        lines.append("|---|---|---|---|")
        imgs = [f"![{name}](input_images/{name})" for name in copied_images]
        while len(imgs) < 4:
            imgs.append("(缺失)")
        lines.append(f"| {imgs[0]} | {imgs[1]} | {imgs[2]} | {imgs[3]} |")
        lines.append("")

    lines.append("## teacher_user（复算）")
    lines.append("```")
    lines.append(teacher_user)
    lines.append("```")
    lines.append("")

    lines.append("## teacher_analysis")
    lines.append("```")
    lines.append(teacher_analysis)
    lines.append("```")
    lines.append("")

    lines.append("## student_assistant（三段 GT）")
    lines.append("```")
    lines.append(student_assistant)
    lines.append("```")
    lines.append("")

    if live_raw is not None:
        lines.append("## live teacher_raw")
        lines.append("```")
        lines.append(live_raw)
        lines.append("```")
        lines.append("")
    if live_post is not None:
        lines.append("## live teacher_postprocess")
        lines.append("```json")
        lines.append(json.dumps(live_post, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    return "\n".join(lines) + "\n"


def run_live_teacher(
    model_dir: pathlib.Path,
    system_prompt: str,
    teacher_user: str,
    image_paths: Sequence[str],
    device: str,
    torch_dtype: str,
) -> Tuple[str, Dict[str, Any]]:
    from PIL import Image  # type: ignore
    from qwen3vl_local.engine import LocalQwen3VLInstructEngine  # type: ignore

    engine = LocalQwen3VLInstructEngine(
        checkpoint_dir=model_dir,
        device=device,
        torch_dtype=torch_dtype,
        max_gen_tokens=160,
        temperature=0.0,
        do_sample=False,
        save_cache=False,
        cache_system_prompt=True,
    )
    engine.load()

    pil_images = [Image.open(p).convert("RGB") for p in image_paths if pathlib.Path(p).exists()]
    raw, _trace = engine.generate(
        system_prompt=system_prompt,
        user_prompt=teacher_user,
        images=pil_images,
        cache_dir=None,
    )
    post = postprocess_teacher(raw)
    return raw, post


def main() -> None:
    parser = argparse.ArgumentParser(description="inspect teacher outputs for sft v2")
    parser.add_argument(
        "--jsonl",
        type=str,
        default=str(_AUTOMOT_ROOT / "checkpoints" / "sft_v2_data" / "train.jsonl"),
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=str(_AUTOMOT_ROOT / "checkpoints" / "sft_v2_teacher_inspect"),
    )
    parser.add_argument("--num-per-scenario", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", type=str, default="")
    parser.add_argument("--case-suffix", type=str, default="")
    parser.add_argument("--live", action="store_true", help="现场重跑 teacher 推理")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B-Instruct"),
        help="仅 --live 时使用",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    jsonl_path = pathlib.Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"[err] jsonl 不存在: {jsonl_path}", file=sys.stderr)
        sys.exit(2)

    rows = read_jsonl(jsonl_path)
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()] or None
    picked = sample_balanced_by_scenario(rows, args.num_per_scenario, args.seed, scenarios)

    mode_name = "B-live" if args.live else "A-jsonl"
    print(f"[inspect] mode={mode_name} total_rows={len(rows)} picked={len(picked)}")

    save_root = pathlib.Path(args.save_root)
    case_root = save_root / "cases"
    case_root.mkdir(parents=True, exist_ok=True)

    # live 模式为了减少重复加载模型，第一次用到时创建 engine。
    live_engine = None
    if args.live:
        from PIL import Image  # type: ignore
        from qwen3vl_local.engine import LocalQwen3VLInstructEngine  # type: ignore

        live_engine = LocalQwen3VLInstructEngine(
            checkpoint_dir=pathlib.Path(args.model_dir),
            device=args.device,
            torch_dtype=args.torch_dtype,
            max_gen_tokens=160,
            temperature=0.0,
            do_sample=False,
            save_cache=False,
            cache_system_prompt=True,
        )
        live_engine.load()

    index_rows: List[Dict[str, Any]] = []
    for sample_idx, row in picked:
        scenario = row.get("scenario", "unknown")
        run_id = row.get("run_id", "norun")
        anchor = int(row.get("anchor", -1))
        case_name = f"{sample_idx:05d}__{scenario}__{run_id}__anchor{anchor}{args.case_suffix}"
        case_dir = case_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        assistant = row["messages"][-1]["content"]
        parsed = parse_student_assistant(assistant)
        teacher_analysis = parsed.get("analysis") or ""

        student_user = strip_image_placeholders(row["messages"][1]["content"])
        teacher_meta_input = row.get("v2_teacher_meta_input", {})
        teacher_user = build_teacher_user_prompt(student_user, teacher_meta_input)

        copied_images = copy_images(row.get("images", []), case_dir / "input_images")

        live_raw: Optional[str] = None
        live_post: Optional[Dict[str, Any]] = None
        if args.live:
            assert live_engine is not None
            from PIL import Image  # type: ignore

            pil_images = [
                Image.open(p).convert("RGB")
                for p in row.get("images", [])
                if pathlib.Path(p).exists()
            ]
            t0 = time.time()
            live_raw, _trace = live_engine.generate(
                system_prompt=_TEACHER_SYSTEM_PROMPT,
                user_prompt=teacher_user,
                images=pil_images,
                cache_dir=None,
            )
            live_post = postprocess_teacher(live_raw)
            live_post["elapsed_sec"] = time.time() - t0

        teacher_fallback_flag = bool(row.get("teacher_meta", {}).get("fallback", False))
        meta = {
            "mode": mode_name,
            "sample_idx": sample_idx,
            "scenario": scenario,
            "run_id": run_id,
            "anchor": anchor,
            "transition": teacher_meta_input.get("transition"),
            "target_status": teacher_meta_input.get("target_status"),
            "memory_in_status": teacher_meta_input.get("memory_in_status"),
            "teacher_fallback_flag": teacher_fallback_flag,
            "analysis_chars": len(teacher_analysis),
            "dataset_version": row.get("dataset_version"),
            "model_dir_live": args.model_dir if args.live else None,
        }

        (case_dir / "teacher_user.txt").write_text(teacher_user, encoding="utf-8")
        (case_dir / "teacher_analysis.txt").write_text(teacher_analysis, encoding="utf-8")
        (case_dir / "student_assistant.txt").write_text(assistant, encoding="utf-8")
        (case_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if live_raw is not None:
            (case_dir / "teacher_raw.txt").write_text(live_raw, encoding="utf-8")
        if live_post is not None:
            (case_dir / "teacher_postprocess.json").write_text(
                json.dumps(live_post, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        overview = render_overview(
            row=row,
            sample_idx=sample_idx,
            teacher_analysis=teacher_analysis,
            student_assistant=assistant,
            teacher_user=teacher_user,
            copied_images=copied_images,
            meta=meta,
            live_raw=live_raw,
            live_post=live_post,
        )
        (case_dir / "overview.md").write_text(overview, encoding="utf-8")

        index_rows.append(
            {
                "sample_idx": sample_idx,
                "scenario": scenario,
                "run_id": run_id,
                "anchor": anchor,
                "transition": meta["transition"],
                "target_status": meta["target_status"],
                "teacher_fallback_flag": teacher_fallback_flag,
                "analysis_chars": len(teacher_analysis),
                "case_dir": str(case_dir),
            }
        )
        print(f"[inspect] done {scenario}/{run_id}/anchor={anchor} -> {case_dir}")

    index_path = save_root / f"index_{'live' if args.live else 'jsonl'}.jsonl"
    with open(index_path, "w", encoding="utf-8") as f:
        for r in index_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[inspect] cases root: {case_root}")
    print(f"[inspect] index: {index_path}")


if __name__ == "__main__":
    main()
