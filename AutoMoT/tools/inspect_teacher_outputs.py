"""可视化检查 v2 teacher 产物。

默认模式 A：只读已经生成好的 v2 jsonl，按场景均匀抽样，落盘 case 目录。
可选模式 B：加 --live，现场重跑 teacher 推理，额外保存 teacher_raw 与后处理信息。
加 --serve --port 0 会自动选空闲端口启动预览网页；live 结果只写 inspect 目录，
不会回写训练 jsonl。

目标：让你快速判断 teacher ANALYSIS 是否符合预期（看图 -> 变化 -> 结论），
并确认 student 看到的三段 GT 与 teacher 文本一致。
"""

from __future__ import annotations

import argparse
import functools
import html
import http.server
import json
import os
import pathlib
import random
import re
import shutil
import socket
import socketserver
import subprocess
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


def _cli_value(name: str) -> Optional[str]:
    prefix = name + "="
    for i, item in enumerate(sys.argv[1:]):
        if item == name and i + 2 <= len(sys.argv[1:]):
            return sys.argv[i + 2]
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def _pick_idle_gpus(n: int = 1) -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[1]), int(parts[2]), parts[0]))
        except ValueError:
            continue
    rows.sort(key=lambda x: (x[0], x[1], int(x[2]) if x[2].isdigit() else 9999))
    return ",".join(row[2] for row in rows[:n])


def _maybe_set_idle_gpu_mask() -> None:
    """live inspect 默认自动挑 1 张空闲 GPU；显式 device / CUDA mask 时保持外部配置。"""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    if os.environ.get("SFT_TEACHER_INSPECT_DISABLE_AUTO_GPU", "0") == "1":
        return
    device_arg = _cli_value("--device")
    if device_arg and device_arg != "auto":
        return
    if "--live" not in sys.argv[1:]:
        return
    selected = _pick_idle_gpus(1)
    if selected:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        print(f"[gpu] auto selected idle CUDA_VISIBLE_DEVICES={selected}; process uses cuda:0/auto")


_maybe_set_idle_gpu_mask()


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
        cleaned = live_post.get("cleaned") if live_post else None
        if cleaned:
            lines.append("## live teacher_analysis_cleaned")
            lines.append("```")
            lines.append(str(cleaned))
            lines.append("```")
            lines.append("")
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


def write_html_index(save_root: pathlib.Path, index_rows: Sequence[Dict[str, Any]]) -> pathlib.Path:
    rows: List[str] = []
    for r in index_rows:
        case_dir = pathlib.Path(str(r["case_dir"]))
        rel = case_dir.relative_to(save_root).as_posix()
        label = (
            f"{r.get('scenario')}/{r.get('run_id')} "
            f"anchor={r.get('anchor')} transition={r.get('transition')}"
        )
        live = r.get("live_analysis_chars")
        live_bits = f" live_chars={live}" if live is not None else ""
        rows.append(
            "<tr>"
            f"<td><a href='{html.escape(rel)}/overview.md'>{html.escape(label)}</a></td>"
            f"<td>{html.escape(str(r.get('target_status')))}</td>"
            f"<td>{html.escape(str(r.get('analysis_chars')))}</td>"
            f"<td>{html.escape(str(r.get('live_fallback')))}</td>"
            f"<td>{html.escape(live_bits)}</td>"
            f"<td><a href='{html.escape(rel)}/teacher_user.txt'>teacher_user</a> "
            f"<a href='{html.escape(rel)}/teacher_raw.txt'>raw</a> "
            f"<a href='{html.escape(rel)}/teacher_analysis_live.txt'>cleaned</a></td>"
            "</tr>"
        )
    body = "\n".join(rows)
    index_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SFT v2 Teacher Inspect</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; line-height: 1.4; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; text-align: left; }}
    a {{ color: #075985; }}
  </style>
</head>
<body>
  <h1>SFT v2 Teacher Inspect</h1>
  <p>Live 模式只把 teacher 输出写到 inspect 目录，不回写训练 jsonl。</p>
  <table>
    <thead>
      <tr><th>case</th><th>target_status</th><th>jsonl_chars</th><th>live_fallback</th><th>live</th><th>files</th></tr>
    </thead>
    <tbody>
{body}
    </tbody>
  </table>
</body>
</html>
"""
    path = save_root / "index.html"
    path.write_text(index_html, encoding="utf-8")
    return path


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def serve_directory(root: pathlib.Path, host: str, port: int) -> None:
    if port <= 0:
        port = find_free_port(host)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    with socketserver.ThreadingTCPServer((host, port), handler) as httpd:
        httpd.daemon_threads = True
        print(f"[serve] root={root}")
        print(f"[serve] open http://{host}:{port}/index.html")
        print("[serve] Ctrl-C 结束服务")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[serve] stopped")


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
        default=str(_AUTOMOT_ROOT / "checkpoints" / "sft_v2_lora" / "runtime_teacher_data" / "train.jsonl"),
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
        "--serve",
        action="store_true",
        help="生成 index.html 后启动静态 HTTP 服务，便于训练前在浏览器检查 teacher 输出",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 = 自动选择空闲端口")
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
    if rows and rows[0].get("dataset_version") == "v2_pending" and not args.live:
        print("[warn] 输入是 v2_pending，占位 ANALYSIS 不适合只读抽检；建议加 --live 现场重跑 teacher。")
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
        live_cleaned = live_post.get("cleaned") if live_post else None
        live_fallback = live_post.get("fallback") if live_post else None
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
            "live_analysis_chars": len(live_cleaned) if live_cleaned else None,
            "live_fallback": live_fallback,
        }

        (case_dir / "teacher_user.txt").write_text(teacher_user, encoding="utf-8")
        (case_dir / "teacher_analysis.txt").write_text(teacher_analysis, encoding="utf-8")
        if live_cleaned:
            (case_dir / "teacher_analysis_live.txt").write_text(str(live_cleaned), encoding="utf-8")
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
                "live_analysis_chars": len(live_cleaned) if live_cleaned else None,
                "live_fallback": live_fallback,
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
    html_path = write_html_index(save_root, index_rows)
    print(f"[inspect] html: {html_path}")
    if args.serve:
        serve_directory(save_root, args.host, args.port)


if __name__ == "__main__":
    main()
