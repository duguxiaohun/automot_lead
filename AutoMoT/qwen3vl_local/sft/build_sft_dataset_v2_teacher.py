"""SFT v2 teacher 推理脚本 — 用冻结 base Qwen3-VL-4B-Instruct 产 ANALYSIS GT。

设计目标见 qwen3vl_local/sft/SFT_PLAN.md §6。本脚本读 build_sft_dataset_v1.py --mode v2
产出的 ``v2_pending`` jsonl，对每条样本拼 teacher prompt（含 PRIVILEGED 块）跑一次
推理，把 ``__TEACHER_PENDING__`` 占位替换成真实 ANALYSIS 文本，写入 ``v2`` jsonl。

**关键 invariant**：teacher prompt 是临时拼装，**不落盘**。最终 v2 jsonl 里
``messages[0/1]`` 与 v1 byte 级完全相同，保证 student 训练 / 推理分布不漂移。

支持的工程特性：
- **中断续跑**：启动时扫已写文件的 ``(scenario, run_id, anchor)`` 指纹，跳过已完成样本。
- **多卡分片**：读 torchrun 提供的 ``RANK`` / ``WORLD_SIZE`` 环境变量，每个 rank 处理
  ``sample_idx % world_size == rank`` 的样本，各自写 ``.rank<R>`` 后缀文件，rank0 合并。
- **system prompt KV cache 复用**：所有样本同一份 teacher system prompt，prefill 一次复用。

典型用法（**从 AutoMoT/ 目录运行**）：

```bash
# 8 卡分片跑全集（约 100 分钟）
torchrun --standalone --nproc_per_node=8 qwen3vl_local/sft/build_sft_dataset_v2_teacher.py \\
    --pending-dir checkpoints/sft_v2_data_pending \\
    --output-dir checkpoints/sft_v2_runtime_debug \\
    --model-dir checkpoints/Qwen3-VL-4B-Instruct \\
    --seed 20260601

# 单卡调试，前 32 条（自动挑 1 张空闲 GPU）
python qwen3vl_local/sft/build_sft_dataset_v2_teacher.py \\
    --pending-dir checkpoints/sft_v2_data_pending \\
    --output-dir checkpoints/sft_v2_runtime_debug \\
    --max-samples 32
```
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Set, Tuple

# 与 build_sft_dataset_v1 相同的 sys.path 注入逻辑：
# 本文件在 AutoMoT/qwen3vl_local/sft/，parents[2]=AutoMoT/，parents[3]=automot_lead 仓库根。
_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


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


# DDP race 兜底：torchrun 多 worker 且外部未预设 CVD 时，只让 rank0 跑 nvidia-smi 挑卡
# → atomic 写共享文件，其它 rank 阻塞读，避免每 worker 各自挑卡导致 set_device 撞同一张卡。
_GPU_PICK_IMPORT_TIME = time.time()
_GPU_PICK_WAIT_TIMEOUT_S = 60.0
_GPU_PICK_STALE_TOLERANCE_S = 30.0
_GPU_PICK_LOCK_PREFIX = "teacher_v2_cvd"


def _share_cvd_via_file_for_ddp(want_count: int) -> str:
    """rank0 挑卡 → 共享文件 → 其它 rank 读；锁文件按 MASTER_ADDR+MASTER_PORT 命名隔离
    不同 run，非 rank0 用 mtime >= 本进程 import 时刻 - 容差 拒绝上一轮残留旧文件。"""
    rank = int(os.environ.get("RANK", "0"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    lock_path = pathlib.Path(tempfile.gettempdir()) / f"{_GPU_PICK_LOCK_PREFIX}_{master_addr}_{master_port}.txt"
    min_mtime = _GPU_PICK_IMPORT_TIME - _GPU_PICK_STALE_TOLERANCE_S
    if rank == 0:
        selected = _pick_idle_gpus(want_count)
        if not selected:
            return ""
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        tmp_path = lock_path.with_suffix(f".tmp_{os.getpid()}")
        tmp_path.write_text(selected, encoding="utf-8")
        os.replace(tmp_path, lock_path)
        return selected
    deadline = time.time() + _GPU_PICK_WAIT_TIMEOUT_S
    while True:
        try:
            mtime = lock_path.stat().st_mtime
        except FileNotFoundError:
            mtime = -1.0
        if mtime >= min_mtime:
            break
        if time.time() > deadline:
            raise RuntimeError(
                f"rank {rank} timed out waiting {_GPU_PICK_WAIT_TIMEOUT_S:.0f}s for "
                f"rank0 to publish CUDA_VISIBLE_DEVICES at {lock_path}"
            )
        time.sleep(0.05)
    return lock_path.read_text(encoding="utf-8").strip()


def _maybe_set_idle_gpu_mask() -> None:
    """teacher 单卡/torchrun 多卡默认自动挑空闲 GPU 并覆盖外层残留的 CUDA_VISIBLE_DEVICES。

    torchrun 多 worker 时由 rank0 挑 N 张经文件 IPC 同步给各 rank（避免每 worker 各自
    nvidia-smi 重挑导致 set_device 撞卡）；单进程挑 1 张。
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        selected = _share_cvd_via_file_for_ddp(world_size)
    else:
        selected = _pick_idle_gpus(1)
    if selected:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        print(
            f"[gpu] auto selected idle CUDA_VISIBLE_DEVICES={selected}; "
            f"world_size={world_size}"
        )


_maybe_set_idle_gpu_mask()

from PIL import Image  # noqa: E402

from qwen3vl_local.engine import LocalQwen3VLInstructEngine  # noqa: E402


# ---------------------------------------------------------------------------
# 常量 / 占位
# ---------------------------------------------------------------------------

# 与 build_sft_dataset_v1.py::PLACEHOLDER_ANALYSIS_V2_PENDING 一致。
_PENDING_PLACEHOLDER = "__TEACHER_PENDING__"

# teacher 输出垮掉时的兜底文本。与 v1 占位一致，保证最坏情况下 student 还能正常训。
_FALLBACK_ANALYSIS = "Observations recorded."


# ---------------------------------------------------------------------------
# Teacher prompt 模板（与 SFT_PLAN.md §6 的 teacher 物化目标一致）
# ---------------------------------------------------------------------------

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

Length target (strict):
- Aim for 40-70 words total across the 3 sentences.
- Going under 25 words tends to skip the visual evidence step and become a bare conclusion - do NOT do that.
- Going over 90 words tends to invent extra details, repeat clauses, or drift off-task - do NOT do that.
- Each sentence should carry one specific visual fact; do not pad with hedging phrases ("it seems that", "we can observe that", "as we can see").

Constraints:
- Do NOT mention or reference the PRIVILEGED block; write as if from images only.
- Do NOT invent visual content not actually present.
- Be concise, grounded, factual; 2-4 sentences total, all on a single line.
- Do NOT output STATUS or SUBGOAL; only the ANALYSIS body text (no "ANALYSIS:" prefix).

Output EXACTLY one line of text (the ANALYSIS body, no prefix, no trailing newline)."""


# student user prompt 末尾固定句的开头，用作 PRIVILEGED 块插入点。
# 见 qwen3vl_local/prompt_pipeline.py::build_user_prompt 末尾：
#   "Given the observations above and the memory context, output your ANALYSIS, STATUS, and SUBGOAL."
_STUDENT_TAIL_MARKER = "Given the observations above and the memory context"


def _strip_image_placeholders(user_content: str) -> str:
    """复用 eval_sft_v1.reconstruct_prompts 的去 <image> 前缀逻辑。

    训练 jsonl 里 user.content 形如 ``<image><image><image><image>\nThe 4 images ...``。
    teacher 走 engine.generate 时图片以 structured message 传入，不需要文本 ``<image>``。
    """
    s = user_content.lstrip()
    while s.startswith("<image>"):
        s = s[len("<image>"):]
    return s.lstrip("\n")


def _build_teacher_user_prompt(student_user_no_image: str, meta: Dict) -> str:
    """在 student user prompt 末尾、`Given ...` 句之前插入 PRIVILEGED 块，
    并把末句指令替换为 teacher 专用（只输出 ANALYSIS body）。
    """

    privileged = (
        "\n[PRIVILEGED]\n"
        f"CURRENT_GT_STATUS: {meta['target_status']}\n"
        f"TRANSITION: {meta['transition']}\n"
        f"PREV_STATUS: {meta['memory_in_status']}\n"
        "[/PRIVILEGED]\n\n"
        "Given the observations, memory, and privileged ground truth, "
        "output the ANALYSIS body that the student should plausibly produce from images alone."
    )

    idx = student_user_no_image.find(_STUDENT_TAIL_MARKER)
    if idx >= 0:
        # 把 student "Given ..." 那段整段砍掉（含其后任何内容），换成 PRIVILEGED 段。
        return student_user_no_image[:idx].rstrip() + privileged
    # fallback：student prompt 没有 marker（理论上不会发生），直接 append。
    return student_user_no_image.rstrip() + privileged


# ---------------------------------------------------------------------------
# teacher 输出后处理
# ---------------------------------------------------------------------------

# teacher 偶尔会复读 "ANALYSIS: " 前缀。case-insensitive 兜底。
_PREFIX_PATTERN = re.compile(r"^\s*ANALYSIS\s*:\s*", re.IGNORECASE)

# teacher 偶尔会自作主张继续写 STATUS / SUBGOAL 行 — 截掉。
_STOP_MARKERS = ("\nSTATUS:", "\nSUBGOAL:", "\n\n", "<|im_end|>")


_MAX_ANALYSIS_CHARS = 420   # 对应 ~70 词 / ~110 token 上限
_MIN_ANALYSIS_CHARS = 80    # 对应 ~12 词 / ~20 token 下限


def _truncate_at_sentence_boundary(t: str, hard_limit: int) -> str:
    """优雅截断：先找最后一个句号/问号/感叹号边界（≤ hard_limit），
    退化为词边界，再退化为硬截。

    设计目的：teacher 偶尔会超长（自我延展），硬截在 word 中间会让训练 GT 末尾
    出现半句话（"The vehicle is approachin"），模型学到这种结尾会让自由生成
    也喜欢在 word 中间停。句号边界优雅截 → 训练 GT 永远是完整句子。
    """
    if len(t) <= hard_limit:
        return t
    window = t[:hard_limit]
    # 句末标点 + 后面跟空格或字符串结尾，倒着找最后一个。
    best = -1
    for punct in (". ", "! ", "? "):
        idx = window.rfind(punct)
        if idx > best:
            best = idx + 1  # 把标点包进保留段，空格不要
    if best > hard_limit // 2:
        return t[:best].rstrip()
    # 句号没找到合适位置，回退到词边界。
    cut_pos = window.rfind(" ")
    return t[: cut_pos if cut_pos > 0 else hard_limit].rstrip()


def _postprocess(text: str) -> Tuple[str, bool]:
    """teacher 输出后处理，返回 ``(cleaned, fallback_used)``。

    具体步骤（与本脚本 teacher 输出约束同口径，2026-06-02 收紧长度边界）：

    1. strip 前后空白 + ``ANALYSIS:`` 前缀。
    2. 截断到第一个 ``STATUS:`` / ``SUBGOAL:`` / 双换行 / im_end 之前。
    3. 把剩余 ``\\n`` 替换为空格 — 强制单行。
    4. 长度上限：超过 ``_MAX_ANALYSIS_CHARS`` (420，约 70 词) → 在句号 / 问号 /
       感叹号边界优雅截，退化为词边界（避免在 word 中间截断）。
    5. 长度下限：< ``_MIN_ANALYSIS_CHARS`` (80，约 12 词) 视为 teacher 输出垮了
       （单纯一句结论、丢了视觉证据），回退到 ``Observations recorded.``
       占位让 student 不至于学到坏样本。

    上下限同时收紧的目的：teacher prompt 里写的 "40-70 words" 是 nudge，
    postprocess 是外部 enforce —— 双保险才稳。下限从 20 字符抬到 80 字符，是因为
    20 字符（~4 词）可以混进 "The car stops." 这种没视觉描述的退化样本；
    80 字符保证至少有一句具体场景描述。
    """

    if not text:
        return _FALLBACK_ANALYSIS, True

    t = _PREFIX_PATTERN.sub("", text.strip())

    # 截断到第一个 stop marker。
    cut = len(t)
    for stop in _STOP_MARKERS:
        i = t.find(stop)
        if i >= 0 and i < cut:
            cut = i
    t = t[:cut]

    # 压扁单行：所有连续空白（包括换行）→ 单空格。
    t = re.sub(r"\s+", " ", t).strip()

    # 句号边界优雅截。
    t = _truncate_at_sentence_boundary(t, _MAX_ANALYSIS_CHARS)

    if len(t) < _MIN_ANALYSIS_CHARS:
        return _FALLBACK_ANALYSIS, True
    return t, False


# ---------------------------------------------------------------------------
# 中断续跑：指纹与已有行
# ---------------------------------------------------------------------------

def _fingerprint(row: Dict) -> Tuple[str, str, int]:
    return (row["scenario"], row["run_id"], int(row["anchor"]))


def _count_jsonl_rows(path: pathlib.Path) -> int:
    """计数 jsonl 非空行数。manifest 完整性校验用。"""

    if not path.exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _scan_done(path: pathlib.Path) -> Set[Tuple[str, str, int]]:
    """读 path 里已写样本的指纹集合。文件不存在 / 损坏行都跳过。"""
    done: Set[Tuple[str, str, int]] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                done.add(_fingerprint(row))
            except (KeyError, TypeError):
                continue
    return done


# ---------------------------------------------------------------------------
# 单 split 处理（train.jsonl 或 val.jsonl）
# ---------------------------------------------------------------------------

def _process_split(
    *,
    engine: LocalQwen3VLInstructEngine,
    pending_path: pathlib.Path,
    output_path: pathlib.Path,
    rank: int,
    world_size: int,
    flush_every: int,
    max_samples: int,
    seed: int,
    log_prefix: str,
) -> Tuple[int, int]:
    """对一个 split 跑 teacher。返回 ``(n_processed, n_fallback)``。

    - ``world_size > 1``：每个 rank 写 ``output_path.with_suffix(".jsonl.rank<R>")``。
    - ``world_size == 1``：直接 append 到 ``output_path``。

    中断续跑：扫 ``write_path`` 与 ``output_path``（若存在）的指纹去重。
    """

    if world_size > 1:
        write_path = output_path.with_suffix(output_path.suffix + f".rank{rank}")
    else:
        write_path = output_path

    # 收集已完成指纹：write_path 自身（本 rank 此前的进度）+ output_path（之前 merge 过的成果）。
    done = _scan_done(write_path)
    if write_path != output_path:
        done.update(_scan_done(output_path))

    # 读 pending；保持 sample_idx 与 v1 原顺序，便于 rank 分片可复现。
    with open(pending_path, "r", encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]

    my_rows = [(i, r) for i, r in enumerate(all_rows) if i % world_size == rank]
    if max_samples > 0:
        my_rows = my_rows[:max_samples]

    todo = [(i, r) for i, r in my_rows if _fingerprint(r) not in done]

    print(
        f"[{log_prefix}] rank={rank}/{world_size} "
        f"assigned={len(my_rows)} already_done={len(my_rows) - len(todo)} todo={len(todo)} "
        f"write={write_path}"
    )

    if not todo:
        return 0, 0

    n_processed = 0
    n_fallback = 0
    t0 = time.time()

    # append 模式打开，配合 flush 策略实现 crash-safe 续跑。
    f_out = open(write_path, "a", encoding="utf-8")
    try:
        for idx_global, row in todo:
            meta_in = row.get("v2_teacher_meta_input")
            if not meta_in:
                print(f"[{log_prefix}] rank={rank} skip idx={idx_global}: missing v2_teacher_meta_input")
                continue

            student_user = _strip_image_placeholders(row["messages"][1]["content"])
            teacher_user = _build_teacher_user_prompt(student_user, meta_in)

            try:
                pil_images = [Image.open(p).convert("RGB") for p in row["images"]]
            except (FileNotFoundError, OSError) as e:
                print(f"[{log_prefix}] rank={rank} image load err idx={idx_global}: {e}")
                cleaned, fb = _FALLBACK_ANALYSIS, True
                raw_text = ""
            else:
                try:
                    raw_text, _trace = engine.generate(
                        system_prompt=_TEACHER_SYSTEM_PROMPT,
                        user_prompt=teacher_user,
                        images=pil_images,
                    )
                except Exception as e:  # noqa: BLE001
                    # 单条失败不应该中断整个数据集生成；落 fallback 继续。
                    print(f"[{log_prefix}] rank={rank} generate err idx={idx_global}: {e}")
                    cleaned, fb = _FALLBACK_ANALYSIS, True
                    raw_text = ""
                else:
                    cleaned, fb = _postprocess(raw_text)

            if fb:
                n_fallback += 1

            # 替换 ANALYSIS 占位。只替换第一个 __TEACHER_PENDING__，
            # 不影响真实文本里万一出现的同串（理论上不会）。
            new_assistant = row["messages"][2]["content"].replace(_PENDING_PLACEHOLDER, cleaned, 1)
            new_row: Dict = dict(row)
            new_row["messages"] = [
                row["messages"][0],
                row["messages"][1],
                {"role": "assistant", "content": new_assistant},
            ]
            new_row["dataset_version"] = "v2"
            new_row["teacher_meta"] = {
                "model_dir": str(engine.checkpoint_dir),
                "seed": seed,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "analysis_chars": len(cleaned),
                "fallback": fb,
            }
            # 保留 v2_teacher_meta_input 不删，方便事后回放 / 重生成。

            f_out.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            n_processed += 1

            if n_processed % flush_every == 0:
                f_out.flush()
                elapsed = time.time() - t0
                rate = n_processed / max(elapsed, 1e-6)
                remain = len(todo) - n_processed
                eta_min = remain / max(rate, 1e-6) / 60.0
                print(
                    f"[{log_prefix}] rank={rank} progress {n_processed}/{len(todo)} "
                    f"rate={rate:.2f}/s eta={eta_min:.1f}min fallback={n_fallback}"
                )
    finally:
        f_out.flush()
        f_out.close()

    return n_processed, n_fallback


# ---------------------------------------------------------------------------
# rank 文件合并（仅 world_size > 1 时需要）
# ---------------------------------------------------------------------------

def _merge_rank_files(output_path: pathlib.Path, world_size: int) -> None:
    """rank0 把 ``.rank<R>`` 分片合并到 ``output_path``，按指纹去重后按 (scenario, run_id, anchor)
    排序，并删除分片文件。

    幂等：output_path 已有内容也会被读入参与合并（中断续跑场景）。
    """

    all_rows: List[Dict] = []
    seen: Set[Tuple[str, str, int]] = set()

    def _ingest(p: pathlib.Path) -> None:
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    fp = _fingerprint(row)
                except (KeyError, TypeError):
                    continue
                if fp in seen:
                    continue
                seen.add(fp)
                all_rows.append(row)

    # 先读已有 output_path（如果存在），再读各 rank 分片。
    _ingest(output_path)
    rank_paths: List[pathlib.Path] = []
    for r in range(world_size):
        rp = output_path.with_suffix(output_path.suffix + f".rank{r}")
        _ingest(rp)
        rank_paths.append(rp)

    all_rows.sort(key=_fingerprint)

    with open(output_path, "w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    for rp in rank_paths:
        if rp.exists():
            rp.unlink()

    print(f"[merge] wrote {len(all_rows)} rows to {output_path}; {len(rank_paths)} rank shards removed")


# ---------------------------------------------------------------------------
# DDP / device 选址
# ---------------------------------------------------------------------------

def _ddp_env() -> Tuple[int, int, int]:
    """读 torchrun 提供的环境变量。无 torchrun 时退化为单进程 (0, 1, 0)。"""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def _pin_local_gpu(local_rank: int, world_size: int) -> None:
    """torchrun 多进程时把每个 rank 钉到对应 LOCAL_RANK 的 GPU。

    自动选卡时每个 rank 会看到同一组 CUDA_VISIBLE_DEVICES；必须 set_device(local_rank)，
    让 LocalQwen3VLInstructEngine 的 device='auto' 落到当前 rank 对应的可见卡。
    """
    if world_size <= 1:
        return
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > local_rank:
            torch.cuda.set_device(local_rank)
        elif torch.cuda.is_available():
            raise RuntimeError(
                f"torchrun local_rank={local_rank} 但当前只看到 "
                f"{torch.cuda.device_count()} 张 GPU；"
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
            )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"failed to pin local_rank={local_rank}: {e}") from e


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SFT v2 teacher: fill ANALYSIS GT into v2_pending jsonl")
    parser.add_argument("--pending-dir", type=str, required=True,
                        help="build_sft_dataset_v1.py --mode v2 的输出目录，含 train.jsonl/val.jsonl")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="v2 数据集落盘目录；teacher 跑完 train.jsonl/val.jsonl 在这里")
    parser.add_argument("--model-dir", type=str,
                        default=str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B-Instruct"),
                        help="冻结 base Qwen 本地目录")
    parser.add_argument("--seed", type=int, default=20260601,
                        help="写入 teacher_meta.seed，供事后回放；temperature=0 时不影响采样")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="teacher 单次生成 token 上限；ANALYSIS 单行约 80-120 token")
    parser.add_argument("--teacher-temperature", type=float, default=0.0,
                        help="0 = greedy。如果 teacher 输出过于套路化可调到 0.3 增加多样性")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 = 不限；>0 时每个 rank 最多处理该数量样本（调试用）")
    parser.add_argument("--flush-every", type=int, default=50,
                        help="每 N 条 flush 一次磁盘并打印进度")
    parser.add_argument("--skip-train", action="store_true", help="跳过 train.jsonl")
    parser.add_argument("--skip-val", action="store_true", help="跳过 val.jsonl")
    args = parser.parse_args()

    rank, world_size, local_rank = _ddp_env()
    _pin_local_gpu(local_rank, world_size)

    pending_dir = pathlib.Path(args.pending_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] rank={rank} world_size={world_size} local_rank={local_rank}")
    print(f"[init] pending_dir={pending_dir} output_dir={output_dir}")

    # 加载冻结 base Qwen。teacher 永远不挂 LoRA。
    # cache_system_prompt=True 让所有样本 prefill 共享同一份 system prompt KV，
    # 实测能省 ~50% 推理时间（与 eval_sft_v1.py 的 anchor12_sanity 同套优化）。
    engine = LocalQwen3VLInstructEngine(
        checkpoint_dir=pathlib.Path(args.model_dir),
        device="auto",
        torch_dtype="bfloat16",
        max_gen_tokens=args.max_new_tokens,
        temperature=args.teacher_temperature,
        do_sample=(args.teacher_temperature > 0.0),
        cache_system_prompt=True,
    )
    engine.load()

    splits = []
    if not args.skip_train:
        splits.append("train.jsonl")
    if not args.skip_val:
        splits.append("val.jsonl")

    total_processed = 0
    total_fallback = 0
    for split in splits:
        pending_path = pending_dir / split
        output_path = output_dir / split
        if not pending_path.exists():
            print(f"[skip] {pending_path} not found")
            continue
        n_proc, n_fb = _process_split(
            engine=engine,
            pending_path=pending_path,
            output_path=output_path,
            rank=rank,
            world_size=world_size,
            flush_every=args.flush_every,
            max_samples=args.max_samples,
            seed=args.seed,
            log_prefix=split,
        )
        total_processed += n_proc
        total_fallback += n_fb

    # 多卡分片：等所有 rank 完成后 rank0 合并 .rank<R> 文件。
    if world_size > 1:
        try:
            import torch.distributed as dist
            if dist.is_available():
                if not dist.is_initialized():
                    # gloo 后端足够，仅用于 barrier，避免对 NCCL 拓扑敏感。
                    dist.init_process_group(backend="gloo")
                dist.barrier()
        except Exception as e:  # noqa: BLE001
            print(f"[barrier] warn: {e}")

        if rank == 0:
            for split in splits:
                output_path = output_dir / split
                _merge_rank_files(output_path, world_size)

    # rank0 写 manifest.json — 复用判定的唯一可信来源。
    # 只在以下条件都满足时写：
    #   1) 当前 rank 是 0；
    #   2) 这次跑是全集（args.max_samples == 0），不是 --max-samples N 的 debug；
    #   3) train + val 两个 pending 文件 + 两个 runtime 文件都存在；
    #   4) runtime 行数与 pending 行数严格相等（每条 pending 都跑出了 runtime 行，
    #      包含 fallback）。
    # 任何一条不满足 → 不写 manifest → sft_v2_train.sh 下次启动会拒绝复用、
    # 重新跑 teacher。这样 32-条 debug cache / 半截 val / skip-train 部分跑都不会被
    # 当成"完整可复用"。
    if rank == 0 and args.max_samples == 0:
        pending_train_path = pending_dir / "train.jsonl"
        pending_val_path = pending_dir / "val.jsonl"
        runtime_train_path = output_dir / "train.jsonl"
        runtime_val_path = output_dir / "val.jsonl"

        all_present = all(
            p.exists() for p in (pending_train_path, pending_val_path,
                                 runtime_train_path, runtime_val_path)
        )
        if not all_present:
            print("[manifest] skip: not all of pending/runtime train+val present "
                  "(skip-train/skip-val or single-split run)")
        else:
            pending_train_rows = _count_jsonl_rows(pending_train_path)
            pending_val_rows = _count_jsonl_rows(pending_val_path)
            runtime_train_rows = _count_jsonl_rows(runtime_train_path)
            runtime_val_rows = _count_jsonl_rows(runtime_val_path)

            if (runtime_train_rows != pending_train_rows
                    or runtime_val_rows != pending_val_rows):
                print(f"[manifest] skip: row count mismatch — "
                      f"pending=({pending_train_rows},{pending_val_rows}) vs "
                      f"runtime=({runtime_train_rows},{runtime_val_rows})")
            else:
                manifest = {
                    # schema_version 升 2：新增 max_new_tokens / teacher_temperature
                    # 字段。sft_v2_train.sh 的 reuse-check 也要校验它们，避免有人
                    # 手动调用 teacher 脚本改了 gen 参数后被 sft_v2_train.sh 误复用。
                    "schema_version": 2,
                    "pending_train_path": str(pending_train_path),
                    "pending_val_path": str(pending_val_path),
                    "pending_train_rows": pending_train_rows,
                    "pending_val_rows": pending_val_rows,
                    "runtime_train_rows": runtime_train_rows,
                    "runtime_val_rows": runtime_val_rows,
                    "model_dir": str(args.model_dir),
                    "seed": int(args.seed),
                    "max_samples": int(args.max_samples),
                    "max_new_tokens": int(args.max_new_tokens),
                    "teacher_temperature": float(args.teacher_temperature),
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                manifest_path = output_dir / "manifest.json"
                manifest_path.write_text(json.dumps(manifest, indent=2,
                                                   ensure_ascii=False))
                print(f"[manifest] wrote {manifest_path} ({pending_train_rows} train, "
                      f"{pending_val_rows} val)")

    print(f"[done] rank={rank} processed={total_processed} fallback={total_fallback}")


if __name__ == "__main__":
    main()
