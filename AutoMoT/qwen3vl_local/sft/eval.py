"""SFT 离线评估：跑 val.jsonl，输出指标 + 小样本完整结果 dump。

复用 AutoMoT/qwen3vl_local/engine.py 的 LocalQwen3VLInstructEngine 做推理；
默认评估 train.sh 写出的 latest/final LoRA；传 ``--lora-dir ''`` 时评 base model。

LoRA 加载方式：默认 ``--merge-lora`` True，即通过 engine attach 后立刻 ``merge_and_unload``
把 LoRA delta 合并进 base 矩阵，避免 PeftModel wrapper 在 Qwen3-VL 上的 M-RoPE +
prepare_inputs_for_generation 不兼容问题（症状：generation 从第二步起输出
"ANALERTA" / "ANAL" 这种 4-10 字符碎片然后 EOS）。非 merge 路径也统一走
engine.attach_lora_adapter(..., merge=False)，不再保留本地第二套 attach 逻辑。

关键开关：

- ``--max-gen-tokens`` 默认 256：teacher 蒸馏的 ANALYSIS body 80-150 token，
  低于 200 会被截断到只剩 ANALYSIS 段，STATUS/SUBGOAL 出不来。
- ``--fallback`` 默认 True：first-pass 缺 STATUS/SUBGOAL 时自动拼 partial
  ``ANALYSIS:<clean>\\nSTATUS:`` 喂回 engine 续解段（详见 PROJECT_CONTEXT.md §18.7）。
  健康 LoRA 上不会触发、无开销；不健康 LoRA 上每个失败样本会多跑 2 次 forward。
  ``--no-fallback`` 关掉看 first-pass 真实表现，与 anchor12 sanity 视角一致。
- ``--teacher-compare`` 默认跟随 full-dump：每条 dump 出来的 case
  额外用 frozen base Qwen（disable_adapter 等价：还没挂 LoRA 时直接调）跑一遍
  teacher prompt（含 PRIVILEGED 块）出对照 ANALYSIS，写到 outputs/expert_analysis.txt
  和 outputs/language_compare.json。
  全集 eval 默认关掉，避免推理时间翻倍。

四个核心指标（与 qwen3vl_local/sft/SFT_PLAN.md §8 一致；含义见 metrics.json["_metric_doc"]）：
  - keep_accuracy:      保持类样本 STATUS == GT 的比例（越大越好）
  - advance_accuracy:   推进类样本 STATUS == GT 的比例（越大越好）
  - early_advance_rate: 保持类样本 STATUS == next(GT) 的比例（越小越好，核心痛点）
  - anchor12_sanity:    anchor=12 fail case 上 STATUS 是否回到 initial（True 即过）

输出布局（与 train.sh 同根，--save-root 必填）：
  <save_root>/eval/metrics.json           聚合指标 + _metric_doc 说明
  <save_root>/eval/predictions.jsonl      每条样本一行（含 raw_text / parsed）
  <save_root>/eval/predictions_diff.jsonl 只保留 pred ≠ gt 的样本（人工查错）
  <save_root>/eval/cases/<scenario>__<run>__<anchor>/   小样本完整 dump（默认开）
      inputs/system_prompt.txt           system prompt 原文
      inputs/user_prompt.txt             user prompt 原文（去 <image> 占位）
      inputs/image_00.jpg ... image_03.jpg  history RGB，**复制**到本地（不 symlink）
      outputs/expert_analysis.txt        base teacher + PRIVILEGED 生成的专家 ANALYSIS
      outputs/language_compare.json      专家语言 / 模型语言 / 物化 GT 语言对比
      outputs/raw_text.txt               模型 raw 输出
      outputs/parsed.json                解析后的 status/subgoal/analysis
      step.json                          单 case 完整元信息
      summary.md                         一页 markdown，顶部突出 SUBGOAL 对比表
  <save_root>/eval_tb/<run_tag>/         可选 TB scalar/text（默认 --no-tb，
                                         因为本项目 TB 入口在步骤二 GoalGen 那侧）

完整 dump 触发条件：
  默认在 --max-samples > 0 时启用（小样本 spot-check 场景），dump 数量 = max-samples；
  也可显式 --full-dump 开 / --no-full-dump 关；--full-dump-limit N 限制 dump 数量。
  当 --max-samples=0（跑全集 val）时，dump 默认关——几百条样本写完整 dump 既慢又占盘。

多卡分片（H）：
  脚本读取 RANK / WORLD_SIZE / LOCAL_RANK 环境变量；torchrun 启动时自动分片，
  每个 rank 处理 sample_idx % world_size == rank 的样本。聚合阶段用
  all_gather_object 把所有 predictions 合到 rank0，再统一写文件 + TB。
  完整 dump 的文件由各 rank 各自落盘（per-case 目录互不冲突）。

典型用法（**从 AutoMoT/ 目录运行**，远程默认 cwd）：

```bash
# 小样本验收 + 完整 dump（推荐：拿到本地做人工复查）
GPU_IDS=0 python qwen3vl_local/sft/eval.py \
  --save-root checkpoints/sft_lora/latest \
  --max-samples 100

# 全集跑指标（不 dump 详情）
GPU_IDS=0 python qwen3vl_local/sft/eval.py \
  --save-root checkpoints/sft_lora/latest

# 多卡分片跑全集
GPU_IDS=0,1,2,3 torchrun --standalone --nproc_per_node=4 qwen3vl_local/sft/eval.py \
  --save-root checkpoints/sft_lora/latest

# 显式评估某个 LoRA checkpoint 时再传 adapter
GPU_IDS=0 python qwen3vl_local/sft/eval.py \
  --lora-dir checkpoints/sft_lora/latest/checkpoint-900 \
  --save-root checkpoints/sft_lora/latest/checkpoint-900 \
  --max-samples 100
```

评估逻辑：
- val.jsonl 里的 assistant message 是 GT，只用于提取 STATUS/SUBGOAL，不会喂给模型。
- user message 中训练用的 `<image>` 占位符会被去掉；真实图片通过 engine 的
  structured image content 传入，和 qwen3vl_instruct_paradigm_a_runner.py 保持一致。
- anchor12_sanity 是额外的单例检查：跑最初触发“过早推进”的 route/anchor，
  看模型是否把 STATUS 保持为 initial。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# HF 离线开关 — 必须在 import transformers / qwen 相关模块之前生效。
import os  # noqa: E402
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _cli_value(name: str) -> Optional[str]:
    """轻量读取启动参数，避免 argparse 之前就 import torch 后无法安全改 CUDA mask。"""
    prefix = name + "="
    for i, item in enumerate(sys.argv[1:]):
        if item == name and i + 2 <= len(sys.argv[1:]):
            return sys.argv[i + 2]
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def _pick_idle_gpus(n: int = 1) -> str:
    """用 nvidia-smi 按显存占用和利用率挑 n 张空闲 GPU；失败时返回空串。"""

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


def _normalize_gpu_ids(value: str) -> str:
    """规范化 GPU id 列表字符串，去掉空白和空项。"""

    ids = [part.strip() for part in str(value).split(",") if part.strip()]
    return ",".join(ids)


def _count_gpu_ids(value: str) -> int:
    """统计规范化后的 GPU id 数量。"""

    normalized = _normalize_gpu_ids(value)
    if not normalized:
        return 0
    return len(normalized.split(","))


# DDP 竞争条件兜底：torchrun 多 worker 且外部未预设 CVD 时，只让 rank0 跑 nvidia-smi 挑卡，
# 再原子写共享文件，其它 rank 阻塞读取，避免每个 worker 各自挑卡导致 set_device 撞同一张卡。
_GPU_PICK_IMPORT_TIME = time.time()
_GPU_PICK_WAIT_TIMEOUT_S = 60.0
_GPU_PICK_STALE_TOLERANCE_S = 30.0
_GPU_PICK_LOCK_PREFIX = "eval_sft_cvd"


def _share_cvd_via_file_for_ddp(want_count: int) -> str:
    """rank0 挑卡后写共享文件，其它 rank 再读取；锁文件按 MASTER_ADDR+MASTER_PORT
    命名以隔离不同 run，非 rank0 用 mtime >= 本进程 import 时刻 - 容差来拒绝上一轮残留旧文件。"""
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
    """默认自动挑空闲 GPU 并覆盖外层残留的 CUDA_VISIBLE_DEVICES；单进程挑 1 张，
    torchrun 多 worker 由 rank0 挑 N 张后经文件 IPC 同步给各 rank（避免每 worker 各自
    nvidia-smi 抖动撞卡）。

    仅 --device 显式传 cpu/cuda[:N] 时尊重用户锁卡，不覆盖。
    """
    device_arg = _cli_value("--device")
    if device_arg and device_arg.strip().lower() not in ("", "auto"):
        print(f"[gpu] using explicit --device {device_arg}; CUDA_VISIBLE_DEVICES is not modified")
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    pinned = _normalize_gpu_ids(os.environ.get("GPU_IDS", ""))
    if pinned:
        picked = _count_gpu_ids(pinned)
        if world_size > 1 and picked < world_size:
            raise RuntimeError(
                f"GPU_IDS={pinned} only provides {picked} GPU(s), "
                f"but torchrun WORLD_SIZE={world_size}; please provide at least {world_size} ids."
            )
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = pinned
        if rank == 0:
            print(
                f"[gpu] using explicit GPU_IDS={pinned}; world_size={world_size}; "
                f"previous CUDA_VISIBLE_DEVICES={previous or '<unset>'}"
            )
        return
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

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

# TensorBoard 是可选依赖：训练机一定有（torch 自带），离线静态分析机可能没装。
# 缺包就静默关闭 TB 写入，不应该让整个 eval 崩。
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: E402
    _TB_AVAILABLE = True
except Exception:
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False

from qwen3vl_local.engine import LocalQwen3VLInstructEngine  # noqa: E402
from qwen3vl_local.prompt_pipeline import (  # noqa: E402
    DrivingMemory,
    build_system_prompt,
    build_user_prompt,
    get_full_sequence,
    parse_vlm_output,
)
from qwen3vl_local.sft.train import (  # noqa: E402
    _TEACHER_SYSTEM_PROMPT,
    build_teacher_user_prompt,
    postprocess_teacher,
)


# ---------------------------------------------------------------------------
# 分布式 helper（H — torchrun 多卡分片）
# ---------------------------------------------------------------------------

def setup_distributed() -> Tuple[int, int, int]:
    """读 torchrun 注入的 RANK / WORLD_SIZE / LOCAL_RANK；单卡跑时三者默认 0/1/0。

    与训练入口同口径：init nccl + set_device 必须在所有 cuda 操作之前完成，
    否则多个进程会抢 cuda:0 然后挂死。
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """退出前销毁 torch.distributed process group。"""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    """rank0 负责聚合记录、写指标和落盘全局文件。"""

    return rank == 0


def all_gather_records(records: List[Dict[str, Any]], world_size: int) -> List[Dict[str, Any]]:
    """跨进程聚合 predictions_records；单卡 / 未初始化时直接原样返回。

    用 all_gather_object 而不是手写 tensor 序列化：
    - records 里有 str / None / int 混合 dict，自己 pad+pickle 反而容易出错；
    - all_gather_object 走 pickle，过程几百条 dict 上限远低于 nccl 默认上限；
    - 量大时（万级样本）才需要换 tensor 路径；目前 SFT val ~800 条，对延迟无感。
    """
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return records
    bucket: List[Optional[List[Dict[str, Any]]]] = [None] * world_size
    dist.all_gather_object(bucket, records)
    merged: List[Dict[str, Any]] = []
    for shard in bucket:
        if shard:
            merged.extend(shard)
    # 按 sample_idx 升序排：分片打散后顺序会乱，统一排序方便人工复查。
    merged.sort(key=lambda r: r.get("sample_idx", 0))
    return merged


def _dump_invocation(output_dir: pathlib.Path, rank: int = 0) -> None:
    """把 sys.argv + 关键 env vars + 元信息写到 ``output_dir/invocations/<ts>_<host>_pid<pid>.txt``。

    只 rank0 写；失败不阻塞 eval（缺 git / IO 错误等都吞掉只打印一行警告）。
    事后想"这版 eval 是哪条命令跑的"直接 cat 就够，不用回翻 shell history。
    """

    if rank != 0:
        return
    try:
        import datetime as _dt
        import platform as _platform
        import shlex as _shlex
        import socket as _socket
        import subprocess as _subprocess

        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        host = _socket.gethostname()
        inv_dir = output_dir / "invocations"
        inv_dir.mkdir(parents=True, exist_ok=True)
        out_path = inv_dir / f"{ts}_{host}_pid{os.getpid()}.txt"

        env_keys = (
            "CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "RANK", "LOCAL_RANK",
            "MASTER_ADDR", "MASTER_PORT", "NCCL_DEBUG", "NCCL_P2P_LEVEL",
            "PYTORCH_CUDA_ALLOC_CONF",
            "GOALGEN_COMPILE_DIT", "GOALGEN_CUDNN_BENCHMARK",
            "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        )
        try:
            git = _subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(pathlib.Path(__file__).resolve().parent),
                capture_output=True, text=True, timeout=5,
            )
            git_commit = git.stdout.strip() if git.returncode == 0 else "<unavailable>"
        except Exception:
            git_commit = "<unavailable>"

        lines = [
            f"# saved at {ts}",
            f"# hostname = {host}",
            f"# pid = {os.getpid()}",
            f"# python = {sys.version.split()[0]}",
            f"# torch = {getattr(torch, '__version__', '<unknown>')}",
            f"# platform = {_platform.platform()}",
            f"# git_commit = {git_commit}",
            "",
            "# ---- selected env vars ----",
            *[f"{k}={os.environ.get(k, '<unset>')}" for k in env_keys],
            "",
            "# ---- sys.argv (one per line) ----",
            *sys.argv,
            "",
            "# ---- shell replay ----",
            " ".join(_shlex.quote(a) for a in sys.argv),
        ]
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[invocation] saved -> {out_path}")
    except Exception as exc:
        print(f"[invocation] 保存失败（不阻塞）：{exc}")


def read_jsonl(path: str) -> List[Dict]:
    """逐行读 jsonl。空行容错。"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def extract_assistant_target(sample: Dict) -> Dict[str, str]:
    """从 messages[-1] 取出 GT 字段。"""
    assistant_text = sample["messages"][-1]["content"]
    parsed = parse_vlm_output(assistant_text)
    return {
        "status": parsed.get("status"),
        "subgoal": parsed.get("subgoal"),
    }


def extract_assistant_target_raw(sample: Dict) -> str:
    """完整 GT 文本（含 ANALYSIS + STATUS + SUBGOAL），供 dump 落 gt.txt 使用。"""
    return sample["messages"][-1]["content"]


def reconstruct_prompts(sample: Dict) -> Dict[str, str]:
    """从 jsonl 还原 system_prompt / user_prompt 字符串与 image 路径。

    engine.generate 接受单独的 system_prompt + user_prompt + images 三件。
    user_content 在 build_dataset.py 里前置了多个 <image>，这里去掉。

    为什么训练和评估这里不同：
    - ms-swift 训练侧用 `<image>` 文本占位符匹配顶层 images 路径；
    - 本项目本地 engine 走 HuggingFace processor 的 structured message，
      图片以 {"type": "image", "image": PIL} 形式传入；
    - 因此 eval 需要还原出“纯 user prompt”，避免 `<image>` 文本被模型当普通文本读。
    """
    system = sample["messages"][0]["content"]
    user_raw = sample["messages"][1]["content"]
    # 去掉前置的 <image>...<image>\n。
    user = user_raw.lstrip()
    while user.startswith("<image>"):
        user = user[len("<image>"):]
    user = user.lstrip("\n")
    return {"system": system, "user": user, "images": sample["images"]}


def _teacher_meta_for_eval(sample: Dict[str, Any], gt_status: Optional[str]) -> Dict[str, str]:
    """还原 teacher prompt 需要的 PRIVILEGED 元信息。

    新数据由 build_dataset.py 写入 teacher_meta_input；这里保留 fallback，方便旧 jsonl
    也能生成专家语言对照。
    """

    meta = sample.get("teacher_meta_input")
    if isinstance(meta, dict):
        return {
            "target_status": str(meta.get("target_status") or gt_status or "unknown"),
            "transition": str(meta.get("transition") or ("advance" if sample.get("is_transition_sample") else "keep")),
            "memory_in_status": str(meta.get("memory_in_status") or "unknown"),
        }
    return {
        "target_status": str(gt_status or "unknown"),
        "transition": "advance" if sample.get("is_transition_sample") else "keep",
        "memory_in_status": "unknown",
    }


def generate_expert_analysis(
    engine: LocalQwen3VLInstructEngine,
    sample: Dict[str, Any],
    images_loader,
) -> Dict[str, Any]:
    """用训练同款 base teacher prompt 生成专家 ANALYSIS，供测试时和模型语言对比。"""

    pieces = reconstruct_prompts(sample)
    gt = extract_assistant_target(sample)
    meta = _teacher_meta_for_eval(sample, gt.get("status"))
    try:
        raw_text, _ = engine.generate(
            system_prompt=_TEACHER_SYSTEM_PROMPT,
            user_prompt=build_teacher_user_prompt(pieces["user"], meta),
            images=images_loader(pieces["images"]),
            cache_dir=None,
        )
        analysis, fallback = postprocess_teacher(raw_text)
        return {
            "analysis": analysis,
            "raw_text": raw_text,
            "fallback": fallback,
            "error": None,
            "teacher_meta_input": meta,
        }
    except Exception as exc:
        return {
            "analysis": None,
            "raw_text": "",
            "fallback": True,
            "error": str(exc),
            "teacher_meta_input": meta,
        }


# ---------------------------------------------------------------------------
# 单样本推理
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# partial-continue 兜底辅助函数（详见 PROJECT_CONTEXT.md §18.7）
# ---------------------------------------------------------------------------
#
# 设计动机：早期结构字面 loss 处理错误时，自由生成可能陷入
# "ANALYSIS×N 循环复读"，never emit STATUS/SUBGOAL。当前健康 ckpt 不应触发
# 兜底；但作为兜底永久保留，推理路径上多一道保险无副作用。

_FALLBACK_MAX_ANALYSIS_CHARS = 400


def _truncate_to_clean_analysis(raw_text: str, max_chars: int = _FALLBACK_MAX_ANALYSIS_CHARS) -> str:
    """raw_text 模型循环复读时可能是 ``ANA body ... ANALYSIS: ANA body 2 ...``。
    截到首次出现的第二段 ``ANALYSIS:`` 或 ``STATUS/SUBGOAL`` 段标记之前，
    再按句号边界截到 max_chars。
    """
    text = raw_text
    first = re.search(r"^\s*ANALYSIS\s*:", text, flags=re.IGNORECASE)
    start = first.end() if first else 0
    body_tail = text[start:]
    cut_points = []
    repeat = re.search(r"(?:\n|\s)ANALYSIS\s*:", body_tail, flags=re.IGNORECASE)
    if repeat is not None:
        cut_points.append(start + repeat.start())
    marker = re.search(r"\n\s*(?:STATUS|SUBGOAL)\s*:", body_tail, flags=re.IGNORECASE)
    if marker is not None:
        cut_points.append(start + marker.start())
    inline_marker = re.search(r"\s(?:STATUS|SUBGOAL)\s*:", body_tail, flags=re.IGNORECASE)
    if inline_marker is not None:
        cut_points.append(start + inline_marker.start())
    if cut_points:
        text = text[: min(cut_points)]
    if len(text) > max_chars:
        window = text[:max_chars]
        best = -1
        for punct in (". ", "! ", "? "):
            i = window.rfind(punct)
            if i > best:
                best = i + 1
        if best > max_chars // 2:
            text = text[:best].rstrip()
        else:
            cut = window.rfind(" ")
            text = text[: cut if cut > 0 else max_chars].rstrip()
    return text.rstrip()


def _normalize_event_name(text: Optional[str], valid_events: Sequence[str]) -> Optional[str]:
    """从短文本开头取第一个合法 event_name。

    模型有时会续解成 ``hazard_detect.``、``STATUS: hazard_detect``，甚至多吐
    一点解释文字。这里不信任自由文本，只接受当前 scenario EVENT_SEQUENCE 内的首个
    word token；常见句末标点会被自然剥离。
    """
    if not text:
        return None
    valid = set(valid_events)
    cleaned = text.strip()
    cleaned = re.sub(r"^(?:STATUS|SUBGOAL)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    m = re.match(r"[A-Za-z0-9_]+", cleaned)
    if not m:
        return None
    token = m.group(0)
    return token if token in valid else None


def _parse_event_name(
    continuation: str,
    valid_events: Sequence[str],
    expected_label: Optional[str] = None,
) -> Optional[str]:
    """从模型续解的文本中取首段 word chars 作为候选 event_name。

    续解返回的文本一般形如 ``" hazard_detect\\nSUBGOAL: ..."`` 或 ``"hazard_detect"``。
    截到第一个换行 / 段切换字面之前，再只接受当前 scenario 的 EVENT_SEQUENCE 内
    token，避免把 ``The`` / ``None`` / 错段 label 这类续解废 token 当成有效预测。
    """
    cleaned = continuation.strip()
    label = re.match(r"^(STATUS|SUBGOAL)\s*:\s*", cleaned, flags=re.IGNORECASE)
    if label is not None:
        got = label.group(1).upper()
        if expected_label and got != expected_label.upper():
            return None
        cleaned = cleaned[label.end():]
    for stop in ("\nSUBGOAL", "\nSTATUS", "\n"):
        idx = cleaned.find(stop)
        if idx >= 0:
            cleaned = cleaned[:idx]
            break
    return _normalize_event_name(cleaned, valid_events)


def _continue_partial(
    engine: LocalQwen3VLInstructEngine,
    pieces: Dict[str, Any],
    pil_images: List[Any],
    partial_assistant: str,
) -> Optional[str]:
    """调 engine.generate_from_partial 续解，返回 raw continuation；异常时返回 None。"""
    try:
        cont, _ = engine.generate_from_partial(
            system_prompt=pieces["system"],
            user_prompt=pieces["user"],
            images=pil_images,
            partial_assistant_text=partial_assistant,
        )
        return cont
    except Exception as e:
        print(f"[fallback][warn] generate_from_partial 抛异常: {e}")
        return None


def predict_full(
    engine: LocalQwen3VLInstructEngine,
    sample: Dict,
    images_loader,
    enable_fallback: bool = True,
) -> Tuple[str, Dict[str, Optional[str]]]:
    """跑一次推理，同时返回 (raw_text, parsed_dict)。

    parsed_dict 至少含 status / subgoal / analysis 三个字段；缺失字段为 None。
    比原来的 predict_status 多返回 raw_text，是为了让 predictions jsonl 能保留
    模型完整输出（用户人工 review case 时定位错在哪个段，比只有 status 直观）。

    自动 partial-continue fallback（``enable_fallback=True``）：first-pass 输出缺 STATUS
    或 SUBGOAL 时（典型场景 = 旧插件 bug 期间训出的 ckpt 陷入 ANALYSIS 循环），
    先复用 first-pass 中已经合法的 STATUS/SUBGOAL，只对缺失段拼 partial 续解。
    重组后的 raw_text 是干净的三段格式。parsed["fallback"] 记录是否触发以及
    哪几段是续解 / 复用出来的，方便人工 review。

    ``enable_fallback=False`` 时跳过续解，直接返回 first-pass 的 raw_text 与解析
    结果（缺段就让 parsed.status/subgoal 维持 None），用于诊断 LoRA 自身的真实
    表现。fallback_info["attempted"] 此时永远为 False，stages=["disabled"]。
    """
    pieces = reconstruct_prompts(sample)
    pil_images = images_loader(pieces["images"])
    raw_text, _ = engine.generate(
        system_prompt=pieces["system"],
        user_prompt=pieces["user"],
        images=pil_images,
        cache_dir=None,
    )
    parsed = parse_vlm_output(raw_text)
    first_pass_raw_text = raw_text

    fallback_info: Dict[str, Any] = {
        "attempted": False,
        "succeeded": False,
        "used": False,  # 兼容旧版消费方的 succeeded 别名。
        "stages": [],
    }
    if not enable_fallback:
        # 显式禁用兜底：把状态留 None，让上层指标看到 first-pass 的真实命中率。
        fallback_info["stages"].append("disabled")
        fallback_info["first_pass_raw_text"] = first_pass_raw_text
        parsed["fallback"] = fallback_info
        return raw_text, parsed

    if parsed.get("status") is None or parsed.get("subgoal") is None:
        fallback_info["attempted"] = True
        fallback_info["first_pass_raw_text"] = first_pass_raw_text
        try:
            valid_events = get_full_sequence(str(sample.get("scenario", "")))
        except Exception:
            valid_events = tuple()

        # 1) 优先使用 parser 已经抽出的 ANALYSIS body。不要直接拿 raw_text 当
        # partial，否则 first-pass 已有 STATUS 但缺 SUBGOAL 时会构造出双 STATUS。
        analysis_body = parsed.get("analysis") or _truncate_to_clean_analysis(raw_text)
        if analysis_body.lstrip().upper().startswith("ANALYSIS:"):
            analysis_line = analysis_body.strip()
        else:
            analysis_line = f"ANALYSIS: {analysis_body.strip()}"

        # 2) first-pass 已有合法 STATUS 时直接复用，只补缺失 SUBGOAL；
        # 否则才续解 STATUS。
        status_event = _normalize_event_name(parsed.get("status"), valid_events)
        partial_status = f"{analysis_line}\nSTATUS:"
        fallback_info["partial_status"] = partial_status
        if status_event:
            fallback_info["stages"].append("status_reused")
        else:
            status_cont = _continue_partial(engine, pieces, pil_images, partial_status)
            fallback_info["status_continuation"] = status_cont
            status_event = _parse_event_name(status_cont, valid_events, expected_label="STATUS") if status_cont else None
            if status_event:
                fallback_info["stages"].append("status")
            else:
                fallback_info["stages"].append("status_failed")

        # 3) first-pass 已有合法 SUBGOAL 时也复用；否则在 STATUS 已知时续解 SUBGOAL。
        subgoal_event: Optional[str] = None
        if status_event:
            subgoal_event = _normalize_event_name(parsed.get("subgoal"), valid_events)
            if subgoal_event:
                fallback_info["stages"].append("subgoal_reused")
            else:
                partial_subgoal = f"{partial_status} {status_event}\nSUBGOAL:"
                fallback_info["partial_subgoal"] = partial_subgoal
                subgoal_cont = _continue_partial(engine, pieces, pil_images, partial_subgoal)
                fallback_info["subgoal_continuation"] = subgoal_cont
                subgoal_event = _parse_event_name(subgoal_cont, valid_events, expected_label="SUBGOAL") if subgoal_cont else None
                if subgoal_event:
                    fallback_info["stages"].append("subgoal")
                else:
                    fallback_info["stages"].append("subgoal_failed")

        # 4) 重组 raw_text 并重新解析。任意一段兜底失败，那段会缺失。
        if status_event:
            rebuilt = f"{analysis_line}\nSTATUS: {status_event}"
            if subgoal_event:
                rebuilt = f"{rebuilt}\nSUBGOAL: {subgoal_event}"
            raw_text = rebuilt
            parsed = parse_vlm_output(raw_text)
            fallback_info["rebuilt_raw_text"] = raw_text
            fallback_info["succeeded"] = parsed.get("status") is not None and parsed.get("subgoal") is not None
            fallback_info["used"] = fallback_info["succeeded"]

    parsed["fallback"] = fallback_info
    return raw_text, parsed


def predict_status(
    engine: LocalQwen3VLInstructEngine,
    sample: Dict,
    images_loader,
    enable_fallback: bool = True,
) -> Optional[str]:
    """对一条样本跑推理，解析出 STATUS。失败返回 None。

    保留旧签名供 anchor12 sanity 等老调用方使用；新代码请用 predict_full
    拿到 raw_text + parsed_dict。

    enable_fallback 透传到 predict_full，保证 --no-fallback 时 anchor12 sanity
    也跟 val 集走同一条 first-pass 路径（否则两边视角不一致 — val 看 first-pass、
    anchor12 看续解后结果，会让用户对 LoRA 健康度产生矛盾印象）。
    """
    _, parsed = predict_full(engine, sample, images_loader, enable_fallback=enable_fallback)
    return parsed.get("status")


# ---------------------------------------------------------------------------
# 完整 dump：把单条样本的输入、输出和摘要全写到一个 case 目录
# ---------------------------------------------------------------------------

def _copy_image(src: str, dst: pathlib.Path) -> bool:
    """把 image 复制到 case 目录（不 symlink；用户要的是"图像存本地"，
    远端跑完拉到本地时 symlink 会断）。源图不存在时返回 False，调用方记日志。
    """
    src_path = pathlib.Path(src)
    if not src_path.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dst)
    return True


def _format_status_subgoal_comparison_md(
    gt_status: Optional[str],
    pred_status: Optional[str],
    gt_subgoal: Optional[str],
    pred_subgoal: Optional[str],
) -> str:
    """渲染最突出的 GT vs Pred 对比表。
    模型每条样本最关心的就是 STATUS / SUBGOAL 两行是不是和真值一致；
    这里加上 ✅/❌ 让人一眼分辨。
    """
    status_match = "✅" if gt_status == pred_status else "❌"
    subgoal_match = "✅" if gt_subgoal == pred_subgoal else "❌"
    return (
        "| field | GT (truth) | Pred (model) | match |\n"
        "|---|---|---|---|\n"
        f"| **STATUS**  | `{gt_status}` | `{pred_status}` | {status_match} |\n"
        f"| **SUBGOAL** | `{gt_subgoal}` | `{pred_subgoal}` | {subgoal_match} |\n"
    )


def _md_cell(value: Any, max_chars: int = 600) -> str:
    """把任意文本压成 Markdown 表格单元，避免 ANALYSIS 里的管道/换行撑坏表格。"""

    text = "" if value is None else str(value)
    text = text.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def _render_case_summary_md(
    sample: Dict[str, Any],
    sample_idx: int,
    system_prompt: str,
    user_prompt: str,
    gt_status: Optional[str],
    gt_subgoal: Optional[str],
    gt_raw: str,
    pred_status: Optional[str],
    pred_subgoal: Optional[str],
    pred_raw: str,
    error_kind: str,
    error_msg: Optional[str],
    saved_images: List[str],
    args: argparse.Namespace,
    expert_info: Optional[Dict[str, Any]] = None,
    fallback_info: Optional[Dict[str, Any]] = None,
) -> str:
    """一页 markdown：顶部 SUBGOAL/STATUS 对比表 → 输入图引用 → 完整 prompt → GT vs Pred 原文。
    刻意把对比表放最上面：人工 review 第一眼就能看到对错。

    fallback_info 不为 None 且 attempted=True 时，会在顶部信息块附加一行 "fallback"，
    把是否续解成功、哪几段是续解出来的写明，便于人工区分"模型自己生成"vs
    "靠 fallback 拼出来"
    的样本。详见 PROJECT_CONTEXT.md §18.7。
    """
    sc = sample.get("scenario", "?")
    rid = sample.get("run_id", "?")
    anc = sample.get("anchor", "?")
    is_trans = sample.get("is_transition_sample", False)
    lines: List[str] = [
        f"# Case: {sc}/{rid} anchor={anc} (transition={is_trans})",
        "",
        f"- val.jsonl sample_idx: **{sample_idx}**",
        f"- error_kind: **{error_kind}**" + (f"（{error_msg}）" if error_msg else ""),
        f"- lora_dir: `{args.lora_dir or '<base>'}`",
        f"- model_dir: `{args.model_dir}`",
    ]
    if fallback_info and fallback_info.get("attempted"):
        stages = ", ".join(fallback_info.get("stages") or []) or "n/a"
        result = "succeeded" if fallback_info.get("succeeded") else "failed"
        lines.append(f"- **fallback attempted**: {result}, stages = `{stages}`（first-pass 缺段，尝试 partial-continue）")
    lines += [
        "",
        "## GT vs Pred",
        _format_status_subgoal_comparison_md(gt_status, pred_status, gt_subgoal, pred_subgoal),
        "",
        "## Input images (history → current，oldest→newest)",
    ]
    src_paths = sample.get("images", [])
    for k, fname in enumerate(saved_images):
        src = src_paths[k] if k < len(src_paths) else ""
        lines.append(f"- ![img{k}](inputs/{fname}) `inputs/{fname}` ← src `{src}`")
    lines.append("")
    lines.append("## System prompt")
    lines.append("```")
    lines.append(system_prompt)
    lines.append("```")
    lines.append("")
    lines.append("## User prompt")
    lines.append("```")
    lines.append(user_prompt)
    lines.append("```")
    lines.append("")
    lines.append("## GT (assistant ground truth)")
    lines.append("```")
    lines.append(gt_raw)
    lines.append("```")
    lines.append("")
    pred_analysis = parse_vlm_output(pred_raw or "").get("analysis")
    if expert_info is not None:
        lines.append("## Expert vs Model Analysis")
        if expert_info.get("error"):
            lines.append(f"- expert generation error: `{expert_info.get('error')}`")
        lines.append("| source | analysis |")
        lines.append("|---|---|")
        expert_analysis = str(expert_info.get("analysis") or "")
        model_analysis = str(pred_analysis or "")
        gt_analysis = str(parse_vlm_output(gt_raw).get("analysis") or "")
        lines.append(f"| expert teacher | {_md_cell(expert_analysis)} |")
        lines.append(f"| model output | {_md_cell(model_analysis)} |")
        if gt_analysis and gt_analysis != "__TEACHER_PENDING__":
            lines.append(f"| materialized GT | {_md_cell(gt_analysis)} |")
        lines.append("")
    lines.append("## Pred (model raw output)")
    lines.append("```")
    lines.append(pred_raw or "<inference error>")
    lines.append("```")
    lines.append("")
    return "\n".join(lines) + "\n"


def dump_case(
    case_dir: pathlib.Path,
    sample: Dict[str, Any],
    sample_idx: int,
    pieces: Dict[str, Any],
    gt_status: Optional[str],
    gt_subgoal: Optional[str],
    gt_raw: str,
    pred_status: Optional[str],
    pred_subgoal: Optional[str],
    pred_raw: str,
    error_kind: str,
    error_msg: Optional[str],
    args: argparse.Namespace,
    expert_info: Optional[Dict[str, Any]] = None,
    fallback_info: Optional[Dict[str, Any]] = None,
) -> None:
    """把一条样本完整 dump 到 <case_dir>/{inputs, outputs, step.json, summary.md}。

    与 qwen3vl_instruct_paradigm_a_runner.dump_record 同口径：inputs / outputs 二分
    + 顶层 summary.md 一页可读；区别是这里没有 KV trace（SFT 推理走 generate，
    KV 内部细节由 probe.py 提供）。
    """
    inputs_dir = case_dir / "inputs"
    outputs_dir = case_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1) 输入：prompt 原文 + 图像复制到本地（用户明确要求"图像也得存本地"）。
    (inputs_dir / "system_prompt.txt").write_text(pieces["system"], encoding="utf-8")
    (inputs_dir / "user_prompt.txt").write_text(pieces["user"], encoding="utf-8")
    saved_image_names: List[str] = []
    for k, src in enumerate(pieces.get("images", [])):
        fname = f"image_{k:02d}.jpg"
        ok = _copy_image(src, inputs_dir / fname)
        if ok:
            saved_image_names.append(fname)
        else:
            print(f"[dump][warn] sample_idx={sample_idx} 源图不存在，跳过：{src}")

    # 2) 输出：原始文本 + 解析结果。
    (outputs_dir / "raw_text.txt").write_text(pred_raw or "<inference error>", encoding="utf-8")
    if fallback_info and fallback_info.get("first_pass_raw_text"):
        (outputs_dir / "first_pass_raw_text.txt").write_text(
            str(fallback_info.get("first_pass_raw_text")),
            encoding="utf-8",
        )
    pred_analysis = parse_vlm_output(pred_raw or "").get("analysis")
    gt_analysis = parse_vlm_output(gt_raw).get("analysis")
    if expert_info is not None:
        (outputs_dir / "expert_analysis.txt").write_text(
            str(expert_info.get("analysis") or "<expert generation error>"),
            encoding="utf-8",
        )
        language_compare = {
            "expert_analysis": expert_info.get("analysis"),
            "model_analysis": pred_analysis,
            "gt_analysis": gt_analysis,
            "gt_analysis_is_pending": gt_analysis == "__TEACHER_PENDING__",
            "expert_error": expert_info.get("error"),
            "expert_fallback": expert_info.get("fallback"),
            "teacher_meta_input": expert_info.get("teacher_meta_input"),
            "note": "expert_analysis is generated by the base teacher prompt with PRIVILEGED; model_analysis is the LoRA/base model's own free output.",
        }
        (outputs_dir / "language_compare.json").write_text(
            json.dumps(language_compare, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    parsed_obj = {
        "pred_status": pred_status,
        "pred_subgoal": pred_subgoal,
        "pred_analysis": pred_analysis,
        "gt_status": gt_status,
        "gt_subgoal": gt_subgoal,
        "gt_analysis": gt_analysis,
        "expert_analysis": expert_info.get("analysis") if expert_info else None,
        "status_match": gt_status == pred_status,
        "subgoal_match": gt_subgoal == pred_subgoal,
        "error_kind": error_kind,
        "error_msg": error_msg,
        # partial-continue 兜底元信息（PROJECT_CONTEXT.md §18.7）。
        "fallback": fallback_info or {"attempted": False, "succeeded": False, "used": False, "stages": []},
    }
    (outputs_dir / "parsed.json").write_text(
        json.dumps(parsed_obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3) step.json：单条 case 的完整元信息（含 val.jsonl 行号，可回溯）。
    step = {
        "sample_idx": sample_idx,
        "scenario": sample.get("scenario"),
        "run_id": sample.get("run_id"),
        "anchor": sample.get("anchor"),
        "is_transition_sample": sample.get("is_transition_sample", False),
        "image_paths_src": sample.get("images", []),
        "image_files_local": saved_image_names,
        "gt": {"status": gt_status, "subgoal": gt_subgoal, "raw": gt_raw},
        "pred": {"status": pred_status, "subgoal": pred_subgoal, "analysis": pred_analysis, "raw": pred_raw},
        "language_compare": {
            "expert_analysis": expert_info.get("analysis") if expert_info else None,
            "gt_analysis": gt_analysis,
            "gt_analysis_is_pending": gt_analysis == "__TEACHER_PENDING__",
            "expert_error": expert_info.get("error") if expert_info else None,
        },
        "error_kind": error_kind,
        "error_msg": error_msg,
        "fallback": fallback_info or {"attempted": False, "succeeded": False, "used": False, "stages": []},
        "lora_dir": args.lora_dir,
        "model_dir": args.model_dir,
    }
    (case_dir / "step.json").write_text(
        json.dumps(step, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 4) summary.md：一页可读，顶部就是 SUBGOAL/STATUS 对比表。
    md = _render_case_summary_md(
        sample, sample_idx, pieces["system"], pieces["user"],
        gt_status, gt_subgoal, gt_raw,
        pred_status, pred_subgoal, pred_raw,
        error_kind, error_msg, saved_image_names, args,
        expert_info=expert_info,
        fallback_info=fallback_info,
    )
    (case_dir / "summary.md").write_text(md, encoding="utf-8")


def next_event_in_seq(scenario: str, status: Optional[str]) -> Optional[str]:
    """返回场景状态机中 status 的下一个事件；不存在时返回 None。"""

    if status is None:
        return None
    seq = get_full_sequence(scenario)
    try:
        idx = seq.index(status)
    except ValueError:
        return None
    return seq[idx + 1] if idx + 1 < len(seq) else None


def build_rgb_paths_from_route(
    route_dir: str,
    anchor: int,
    *,
    frame_step: int = 1,
    frame_count: int = 4,
) -> List[str]:
    """按 runner 规则构造 RGB clip 路径，并兼容 0000/0001 起始命名。

    anchor12 sanity 不来自 val.jsonl，所以需要现场构造图片路径。
    这里复制 build_dataset.py.py 的路径容错逻辑，保证 sanity 单例和验证集样本
    使用同一种 RGB 对齐规则。
    """

    route = pathlib.Path(route_dir)
    rgb_dir = route / "rgb"
    desc = [max(anchor - i * frame_step, 0) for i in range(frame_count)]
    ordered = list(reversed(desc))

    if not rgb_dir.exists():
        return [str(rgb_dir / f"{idx:04d}.jpg") for idx in ordered]

    rgb_files = sorted(rgb_dir.glob("*.jpg"))
    if not rgb_files:
        return [str(rgb_dir / f"{idx:04d}.jpg") for idx in ordered]

    paths: List[str] = []
    for idx in ordered:
        exact = rgb_dir / f"{idx:04d}.jpg"
        if exact.exists():
            paths.append(str(exact))
        elif 0 <= idx < len(rgb_files):
            paths.append(str(rgb_files[idx]))
        else:
            paths.append(str(exact))
    return paths


def build_anchor_sanity_sample(args: argparse.Namespace) -> Dict:
    """构造 anchor=12 fail case 的单样本，用同一套 predict_status 评估。

    这个样本不是训练/验证集的一部分，而是固定回归测试：
    原始 base Qwen 在这个 early anchor 上容易把 Accident 的 STATUS 从 initial
    提前推进到 hazard_detect。统一 SFT 的底线就是这里要回到 initial。
    """

    memory = DrivingMemory.from_scenario(args.anchor12_scenario)
    image_paths = build_rgb_paths_from_route(
        args.anchor12_route_dir,
        args.anchor12_anchor,
    )
    image_description = (
        f"The {len(image_paths)} images above are ordered oldest to newest; "
        "the last image is the current moment."
    )
    user_text = build_user_prompt(memory, image_description=image_description)
    # 构造成和 jsonl 样本一样的形态，后续统一走 reconstruct_prompts()。
    # 这样 sanity 单例和 val 样本不会因为 prompt 复原路径不同而产生额外变量。
    user_content = "".join("<image>" for _ in image_paths) + "\n" + user_text

    return {
        "scenario": args.anchor12_scenario,
        "run_id": pathlib.Path(args.anchor12_route_dir).name,
        "anchor": args.anchor12_anchor,
        "images": image_paths,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": (
                    f"ANALYSIS: Observations recorded.\n"
                    f"STATUS: {args.anchor12_expected_status}\n"
                    f"SUBGOAL: {next_event_in_seq(args.anchor12_scenario, args.anchor12_expected_status)}"
                ),
            },
        ],
        "is_transition_sample": False,
    }


def run_anchor12_sanity(
    engine: LocalQwen3VLInstructEngine,
    args: argparse.Namespace,
    images_loader,
) -> Dict:
    """跑原始 anchor=12 fail case，返回可写入 metrics 的结果。

    这里捕获异常而不是直接 raise：
    - 远程数据路径可能暂时没挂载；
    - 用户可能只想先跑 val 指标；
    - metrics 里记录 error 比整个评估中断更方便排查。
    """

    if args.skip_anchor12_sanity:
        return {"enabled": False, "passed": None}

    sample = build_anchor_sanity_sample(args)
    try:
        # 把 args.fallback 透传过去：用户 --no-fallback 时 anchor12 sanity 同样
        # 走 first-pass 路径，避免 val / anchor12 两边对 LoRA 健康度给出矛盾印象。
        pred = predict_status(engine, sample, images_loader, enable_fallback=args.fallback)
        expected = args.anchor12_expected_status
        return {
            "enabled": True,
            "passed": pred == expected,
            "pred_status": pred,
            "expected_status": expected,
            "scenario": sample["scenario"],
            "run_id": sample["run_id"],
            "anchor": sample["anchor"],
            "images": sample["images"],
            "error": None,
        }
    except Exception as e:
        return {
            "enabled": True,
            "passed": False,
            "pred_status": None,
            "expected_status": args.anchor12_expected_status,
            "scenario": sample["scenario"],
            "run_id": sample["run_id"],
            "anchor": sample["anchor"],
            "images": sample["images"],
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _resolve_output_paths(args: argparse.Namespace) -> Dict[str, pathlib.Path]:
    """所有 eval 产物在 <save_root>/eval/ 与 <save_root>/eval_tb/<run_tag>/ 之下。

    --save-root 是必填（main 里已 argparse required=True 强制）。老 --out-dir /
    --output-json 等已删，路径不再可单文件 override；要分文件夹直接换 --save-root。
    """
    root = pathlib.Path(args.save_root)
    run_tag = (args.run_tag or "").strip() or _default_run_tag(args)
    eval_dir = root / "eval"
    return {
        "eval_dir": eval_dir,
        "metrics_json": eval_dir / "metrics.json",
        "predictions_jsonl": eval_dir / "predictions.jsonl",
        "predictions_diff_jsonl": eval_dir / "predictions_diff.jsonl",
        "cases_dir": eval_dir / "cases",
        "tb_dir": root / "eval_tb" / run_tag,
    }


def _default_run_tag(args: argparse.Namespace) -> str:
    """根据 LoRA 目录名给 TB run 一个易读的 tag。

    base 模型（lora_dir 为空）记为 'base'；
    checkpoint-N 子目录则用 'ckptN' 形式，方便在 TB run 列表里横向对比多个 ckpt。
    """
    if not args.lora_dir:
        return "base"
    name = pathlib.Path(args.lora_dir).name
    if name.startswith("checkpoint-"):
        return name.replace("checkpoint-", "ckpt")
    return name or "lora"


def main():
    """评估主入口：加载 base/LoRA，按 rank 分片推理，聚合指标并 dump case。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--val-jsonl", type=str,
                        default=str(_AUTOMOT_ROOT / "checkpoints" / "sft_data_pending" / "val.jsonl"))
    parser.add_argument("--model-dir", type=str,
                        default=str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B-Instruct"))
    parser.add_argument("--lora-dir", type=str,
                        default=str(_AUTOMOT_ROOT / "checkpoints" / "sft_lora" / "latest" / "final"),
                        help="LoRA adapter 目录。默认指向 train.sh 写出的 latest run 的 final/ 子目录。"
                             "传 --lora-dir '' 评 base 模型；传具体 checkpoint-XXX 子目录评特定 step 快照。")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 表示评估全部 val 样本，>0 时只评估前 N 条做快速验收。")
    parser.add_argument("--device", default="auto",
                        help="默认 auto；未显式设置 CUDA mask 时会自动挑空闲物理 GPU 并映射到 cuda:0。"
                             "显式传 cuda:N 会关闭自动 GPU mask。")
    parser.add_argument("--torch-dtype", default="bfloat16")
    # ---- 统一保存根目录（必填）----
    # metrics / predictions / cases / TB 全部落到 <save_root>/eval/ 与
    # <save_root>/eval_tb/<run_tag>/，与训练 <save_root>/tb/ 同根。
    parser.add_argument("--save-root", type=str, required=True,
                        help="统一保存根目录（必填，通常与 train 的 OUTPUT_DIR 相同）。"
                             "metrics/predictions/cases 落到 <root>/eval/，TB 落到 <root>/eval_tb/<run_tag>/。")
    parser.add_argument("--run-tag", type=str, default="",
                        help="TB run 子目录名，默认根据 --lora-dir 自动派生（base / ckpt300 / lora 等）。")
    parser.add_argument("--tb", action="store_true",
                        help="显式打开 TB 写入；默认 --no-tb（本项目 TB 入口在步骤二 GoalGen 那侧）。")
    parser.add_argument("--no-tb", dest="tb", action="store_false",
                        help="关闭 TB（默认值）。")
    parser.set_defaults(tb=False)
    parser.add_argument("--cache-system-prompt",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="复用 system prompt 的 KV prefix，节省推理时间。"
                             "--no-cache-system-prompt 可关闭。")
    # ---- LoRA 加载方式 + 生成长度上限 ----
    # 历史踩坑：早期 eval 输出是 "ANALERTA" / "ANAL" 这种 4-10 字符乱码——根因
    # 不是 LoRA 训练崩，而是 PeftModel wrapper 在 Qwen3-VL 的 forward 路径
    # （M-RoPE + prepare_inputs_for_generation）上不兼容；merge_and_unload 后立刻
    # 恢复正常英文。所以默认 merge=True；非 merge 也统一走 engine 路径。
    parser.add_argument("--merge-lora",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="是否在加载 LoRA 后调用 merge_and_unload 合并进 base。"
                             "默认 True；PeftModel wrapper 路径在 Qwen3-VL 上实测会让"
                             "生成从第二步起乱码（如 'ANALERTA'）。非 merge 只建议诊断时使用。")
    # 当前 ANALYSIS 是 teacher 蒸馏真值（80-150 token），96 会截断到只剩 ANALYSIS 段，
    # STATUS/SUBGOAL 永远出不来 → parser 解不到 status 全报 None。
    parser.add_argument("--max-gen-tokens", type=int, default=256,
                        help="自回归生成 token 数上限。默认 256。"
                             "ANALYSIS body 长会被截断，必须 ≥ 200。")
    # partial-continue 兜底开关（PROJECT_CONTEXT.md §18.7）：
    # first-pass 缺 STATUS/SUBGOAL 时自动拼 partial 续解。健康 LoRA 上不会触发，无开销；
    # 不健康 LoRA 上每个失败样本会多跑 2 次 forward（STATUS 续解 + SUBGOAL 续解），
    # 极端情况下 eval 时间变成 3 倍。调试时可 --no-fallback 关掉看 first-pass 真实表现。
    parser.add_argument("--fallback",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="是否启用 partial-continue 兜底（缺 STATUS/SUBGOAL 时续解）。"
                             "默认 True；--no-fallback 关掉看 first-pass 真实表现。")
    # ---- 完整 dump 开关（用户最关心的"小样本完整保存"路径）----
    parser.add_argument("--full-dump", dest="full_dump",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="是否每条样本完整 dump（inputs/outputs/summary.md）。"
                             "默认行为：--max-samples > 0 时开，跑全集（max-samples=0）时关。"
                             "可显式 --full-dump / --no-full-dump 覆盖。")
    parser.add_argument("--full-dump-limit", type=int, default=0,
                        help="最多 dump 多少条样本（防止误开后铺满磁盘）。"
                             "0 = 不限（受 --max-samples 限制）。")
    parser.add_argument("--teacher-compare",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="是否为完整 dump 样本额外生成专家 ANALYSIS 并与模型 ANALYSIS 对比。"
                             "默认跟随 full-dump：小样本 eval 开，全集 eval 关。")
    parser.add_argument("--skip-anchor12-sanity", action="store_true",
                        help="跳过原始 anchor=12 fail case 单例检查。")
    parser.add_argument("--anchor12-route-dir", type=str,
                        default="lead_data/Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46")
    parser.add_argument("--anchor12-scenario", type=str, default="Accident")
    parser.add_argument("--anchor12-anchor", type=int, default=12)
    parser.add_argument("--anchor12-expected-status", type=str, default="initial")
    args = parser.parse_args()

    # ---- 分布式初始化（H）----
    # 单卡 = world_size=1，所有 if rank0 分支恒进，无任何行为差异。
    rank, local_rank, world_size = setup_distributed()
    out_paths = _resolve_output_paths(args)
    out_paths["eval_dir"].mkdir(parents=True, exist_ok=True)
    from qwen3vl_local.run_log import install_output_log
    install_output_log(out_paths["eval_dir"], rank=rank)
    _dump_invocation(pathlib.Path(args.save_root), rank=rank)

    if is_rank0(rank):
        print(f"[eval] world_size={world_size} rank={rank} local_rank={local_rank}")

    samples = read_jsonl(args.val_jsonl)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    if is_rank0(rank):
        print(f"[eval] loaded {len(samples)} samples from {args.val_jsonl}")
        # 显式打印数据集版本，方便排查 pending / materialized。
        ds_ver = samples[0].get("dataset_version", "unknown") if samples else "unknown"
        print(f"[eval] dataset_version={ds_ver}")
        if ds_ver == "pending":
            print("[eval][note] dataset_version=pending: ANALYSIS 段是 __TEACHER_PENDING__ 占位。"
                  "STATUS/SUBGOAL 评测不受影响；case dump 里 GT ANALYSIS 会写成占位文本。"
                  "需要真实 teacher 输出做对照，跑 build_teacher.py 离线物化 val。")

    # ---- 完整 dump 模式判定（用户最关心的"小样本完整保存"路径）----
    # 默认行为：传 --max-samples > 0 时开（小样本 spot-check），跑全集时关。
    # 显式 --full-dump / --no-full-dump 覆盖默认。
    if args.full_dump is None:
        full_dump_enabled = args.max_samples > 0
    else:
        full_dump_enabled = bool(args.full_dump)
    # dump 数量上限：先看 --full-dump-limit，再回退到全部样本。
    dump_limit = args.full_dump_limit if args.full_dump_limit > 0 else len(samples)
    teacher_compare_enabled = full_dump_enabled if args.teacher_compare is None else bool(args.teacher_compare)

    # eval 时 jsonl 已经给了绝对路径，直接 PIL 打开就够。
    # 与 runner load_lead_rgb_clip 一样保留 RGB 原图，不做额外 resize/crop；
    # Qwen processor 会自己处理动态分辨率。
    from PIL import Image  # type: ignore

    def images_loader(paths: List[str]):
        """延迟打开 RGB 图片；失败时交给外层记录 bad sample。"""

        # 每次打开后立刻 convert("RGB")，避免 PIL 延迟读取导致文件句柄在生成期间才报错。
        return [Image.open(p).convert("RGB") for p in paths]

    # 启动 engine + 可选挂 LoRA。
    #
    # 注意：engine 构造函数只保存配置，不加载权重。这里显式 engine.load()，
    # 一方面让 PEFT 有 base model 可挂，另一方面让后续 predict_status 不再重复触发加载。
    #
    # 多卡时把 device 直接固定到 cuda:LOCAL_RANK，避免所有 rank 抢 cuda:0：
    # device='auto' 让 engine.load() 自己挑卡时，多个进程的 hf accelerate 路径会
    # 同时落到 cuda:0 然后 OOM / 卡死。
    device = args.device
    if world_size > 1 and torch.cuda.is_available() and args.device == "auto":
        device = f"cuda:{local_rank}"
    engine = LocalQwen3VLInstructEngine(
        checkpoint_dir=pathlib.Path(args.model_dir),
        device=device,
        torch_dtype=args.torch_dtype,
        max_gen_tokens=args.max_gen_tokens,
        temperature=0.0,
        do_sample=False,
        save_cache=False,
        cache_system_prompt=args.cache_system_prompt,
    )
    engine.load()
    # 专家语言对照必须在 LoRA merge 之前跑：默认 eval 会 merge_and_unload，
    # merge 后无法再临时 disable adapter。只为本 rank 会 dump 的样本生成，避免全集评估变慢。
    expert_compare_by_idx: Dict[int, Dict[str, Any]] = {}
    if teacher_compare_enabled:
        if is_rank0(rank):
            print("[teacher-compare] enabled: generating expert ANALYSIS before attaching LoRA")
        local_dump_count = 0
        for j, teacher_sample in enumerate(samples):
            if world_size > 1 and (j % world_size) != rank:
                continue
            if local_dump_count >= dump_limit:
                break
            expert_compare_by_idx[j] = generate_expert_analysis(engine, teacher_sample, images_loader)
            local_dump_count += 1
    if args.lora_dir:
        # 统一走 engine 的 LoRA attach 入口；merge=True 是默认推荐路径，merge=False
        # 仅用于诊断 PEFT wrapper 行为，不再维护 eval.py 本地第二套 attach 逻辑。
        engine.attach_lora_adapter(args.lora_dir, merge=args.merge_lora)

    # 先跑固定 fail case，方便日志最前面就看见“这次 LoRA 是否解决了原问题”。
    # 如果该 route 不存在，可用 --skip-anchor12-sanity 跳过。
    # 多卡时只让 rank0 跑，其它 rank 拿空 dict 占位，聚合时只取 rank0 的。
    if is_rank0(rank):
        anchor12_sanity = run_anchor12_sanity(engine, args, images_loader)
        if anchor12_sanity.get("enabled"):
            print(f"[anchor12] {anchor12_sanity}")
    else:
        anchor12_sanity = {"enabled": False, "passed": None}

    # 计数器。
    # keep/advance 分开统计：只看总体 accuracy 会掩盖“模型永远保持”或“模型总是提前推进”
    # 这两种完全不同的失败模式。
    # 多卡时本地 rank 只累计自己分片的部分，最后通过 all_gather 把 predictions_records
    # 合到 rank0 重新计算总体指标（避免每个 rank 都 all_reduce 一个 dict 麻烦）。
    n_keep = n_keep_correct = n_early_adv = 0
    n_adv = n_adv_correct = 0
    per_scenario: Dict[str, Counter] = defaultdict(Counter)
    # 逐条预测缓存：始终启用，便于 rank 间 all_gather 后由 rank0 重算指标。
    predictions_records: List[Dict[str, Any]] = []

    cases_dir = out_paths["cases_dir"]
    if full_dump_enabled and is_rank0(rank):
        cases_dir.mkdir(parents=True, exist_ok=True)
        print(f"[dump] 完整 dump 启用 → cases_dir={cases_dir}（每条样本一个目录）")
        print(f"[dump] dump 数量上限 = {dump_limit}（每个 rank 各自落盘，互不冲突）")
    dump_count_local = 0  # 本 rank 已经 dump 的样本数

    for i, sample in enumerate(samples):
        # rank 分片：每条样本只在 i % world_size == rank 时由当前 rank 处理。
        # 步长 world_size 比按连续块切分对磁盘缓存更友好（相邻 rank 拿到的样本来自
        # 不同 run，并行读 NFS 时彼此不抢同一段缓存）。
        if world_size > 1 and (i % world_size) != rank:
            continue
        scenario = sample["scenario"]
        gt = extract_assistant_target(sample)
        gt_status = gt["status"]
        gt_subgoal = gt.get("subgoal") if isinstance(gt, dict) else None
        is_trans = sample.get("is_transition_sample", False)

        # 用 predict_full 拿 raw_text + parsed 一起返回；旧 predict_status 改为 wrapper。
        raw_text: Optional[str] = None
        pred: Optional[str] = None
        pred_subgoal: Optional[str] = None
        err: Optional[str] = None
        # fallback_info：predict_full 内部 partial-continue 兜底是否触发，以及触发了哪几段。
        fallback_info: Dict[str, Any] = {
            "attempted": False,
            "succeeded": False,
            "used": False,
            "stages": [],
        }
        try:
            raw_text, parsed = predict_full(
                engine, sample, images_loader,
                enable_fallback=args.fallback,
            )
            pred = parsed.get("status")
            pred_subgoal = parsed.get("subgoal")
            fallback_info = parsed.get("fallback", fallback_info) or fallback_info
        except Exception as e:
            print(f"[err {i}] {e}")
            err = str(e)

        # 对 keep 样本，pred == next(GT) 就是最关心的 early advance。
        # 其它错误（输出 None、跳到更后状态、输出非法状态）不会计入 early_advance，
        # 但会让 keep_accuracy 下降。
        next_gt = next_event_in_seq(scenario, gt_status)

        if not is_trans:
            n_keep += 1
            if pred == gt_status:
                n_keep_correct += 1
                per_scenario[scenario]["keep_correct"] += 1
            elif pred is not None and pred == next_gt:
                n_early_adv += 1
                per_scenario[scenario]["early_advance"] += 1
            per_scenario[scenario]["keep_total"] += 1
        else:
            n_adv += 1
            if pred == gt_status:
                n_adv_correct += 1
                per_scenario[scenario]["adv_correct"] += 1
            per_scenario[scenario]["adv_total"] += 1

        # error_kind 按"为什么 pred 错"分类，方便后续 diff 文件直接做 Counter 统计：
        #   ok                 — pred == gt
        #   early_advance      — pred == next(gt)（keep 样本最关心的错误）
        #   none               — 没有解析到 status（输出格式坏）
        #   inference_error    — generate 阶段抛异常
        #   other              — 其它（跳更后状态 / 非法 token / advance 样本未对齐 / ...）
        if pred is None and err is not None:
            error_kind = "inference_error"
        elif pred is None:
            error_kind = "none"
        elif pred == gt_status:
            error_kind = "ok"
        elif not is_trans and pred == next_gt:
            error_kind = "early_advance"
        else:
            error_kind = "other"
        predictions_records.append({
            "sample_idx": i,
            "scenario": scenario,
            "run_id": sample.get("run_id"),
            "anchor": sample.get("anchor"),
            "is_transition_sample": is_trans,
            "gt_status": gt_status,
            "gt_subgoal": gt_subgoal,
            "pred_status": pred,
            "pred_subgoal": pred_subgoal,
            "pred_analysis": parse_vlm_output(raw_text or "").get("analysis"),
            "expert_analysis": (expert_compare_by_idx.get(i) or {}).get("analysis"),
            "expert_error": (expert_compare_by_idx.get(i) or {}).get("error"),
            "raw_text": raw_text,
            "error_kind": error_kind,
            "error": err,
            # partial-continue 兜底元信息（PROJECT_CONTEXT.md §18.7）：
            #   attempted=True 表示 first-pass 缺 STATUS/SUBGOAL 触发了续解兜底；
            #   succeeded=True 表示续解后拼回完整三段；
            #   stages 列出哪几段是续解出来的（"status"/"subgoal"）或失败原因。
            "fallback": fallback_info,
        })

        # ---- 完整 dump：每条样本一个 case 目录（在 rank 分片内顺序写）----
        # 写到 dump_limit 上限后停 — 防止跑大集合时误开把磁盘灌满。
        if full_dump_enabled and dump_count_local < dump_limit:
            pieces = reconstruct_prompts(sample)
            gt_full_raw = extract_assistant_target_raw(sample)
            case_name = (
                f"{i:05d}__{scenario}__{sample.get('run_id', 'norun')}"
                f"__anchor{sample.get('anchor', 'na')}__{error_kind}"
            )
            try:
                dump_case(
                    case_dir=cases_dir / case_name,
                    sample=sample,
                    sample_idx=i,
                    pieces=pieces,
                    gt_status=gt_status,
                    gt_subgoal=gt_subgoal,
                    gt_raw=gt_full_raw,
                    pred_status=pred,
                    pred_subgoal=pred_subgoal,
                    pred_raw=raw_text or "",
                    error_kind=error_kind,
                    error_msg=err,
                    args=args,
                    expert_info=expert_compare_by_idx.get(i),
                    fallback_info=fallback_info,
                )
                dump_count_local += 1
            except Exception as dump_err:
                # dump 失败不影响主指标；只打印警告。
                print(f"[dump][warn] sample_idx={i} dump 失败：{dump_err}")

        if (i + 1) % 50 == 0 and is_rank0(rank):
            # 多卡时本地 rank 的 n_keep_correct 只是本分片的视角，先打印一个本地估计；
            # 全局精确指标在末尾 all_gather 后由 rank0 重算。
            print(f"[eval][rank{rank}] processed up to sample {i+1}/{len(samples)} (local view)")

    # ---- 跨 rank 聚合（H）----
    # 单卡时 all_gather_records 直接返回原列表，行为完全一致。
    # 多卡时 rank0 拿到全 rank 的 predictions_records 合并并按 sample_idx 排序，
    # 其它 rank 拿到同样的合并结果但不写文件。
    if world_size > 1:
        dist.barrier()
    predictions_records = all_gather_records(predictions_records, world_size)

    # 只有 rank0 计算最终指标 + 写 metrics / predictions / TB。
    # 其它 rank 走 cleanup 退出，避免重复写文件。
    if not is_rank0(rank):
        cleanup_distributed()
        return

    # 重算总体指标：本地累计的 n_keep / n_adv 是单 rank 的视角，多卡下不正确。
    # 用聚合后的 predictions_records 重新统计一次（与之前 per_scenario 字典逻辑同口径）。
    n_keep = n_keep_correct = n_early_adv = 0
    n_adv = n_adv_correct = 0
    per_scenario = defaultdict(Counter)  # type: Dict[str, Counter]
    for row in predictions_records:
        scenario = row.get("scenario") or "<unknown>"
        is_trans = row.get("is_transition_sample", False)
        pred = row.get("pred_status")
        gt_status = row.get("gt_status")
        next_gt = next_event_in_seq(scenario, gt_status)
        if not is_trans:
            n_keep += 1
            if pred == gt_status:
                n_keep_correct += 1
                per_scenario[scenario]["keep_correct"] += 1
            elif pred is not None and pred == next_gt:
                n_early_adv += 1
                per_scenario[scenario]["early_advance"] += 1
            per_scenario[scenario]["keep_total"] += 1
        else:
            n_adv += 1
            if pred == gt_status:
                n_adv_correct += 1
                per_scenario[scenario]["adv_correct"] += 1
            per_scenario[scenario]["adv_total"] += 1

    # metrics 顶部放一个 _metric_doc：人工打开 metrics.json 就能直接看到每个指标含义，
    # 不用再翻文档。用户明确反馈"指标太多看不懂"，文档放在数据旁边最不容易丢。
    metric_doc = {
        "keep_accuracy": "保持类样本 STATUS == GT 的比例（越大越好；模型该 hold 时 hold）",
        "advance_accuracy": "推进类样本 STATUS == GT 的比例（越大越好；模型该 advance 时 advance）",
        "early_advance_rate": "保持类样本 STATUS == next(GT) 的比例（越小越好；模型不该 advance 时 advance — 核心痛点）",
        "anchor12_sanity": "anchor=12 固定 fail case 上 STATUS 是否回到 initial；passed=true 即原始 bug 已修",
        "per_scenario": "按 scenario 拆开的细分计数：{keep_correct, keep_total, early_advance, adv_correct, adv_total}",
        "fallback_count": "partial-continue 兜底尝试的样本数（PROJECT_CONTEXT.md §18.7）；健康 LoRA 应该接近 0，>0 表示模型 first-pass 缺 STATUS/SUBGOAL 段",
        "fallback_rate": "fallback_count / n_total",
        "fallback_success_count": "fallback 尝试后成功拼回完整 STATUS/SUBGOAL 的样本数",
        "fallback_success_rate": "fallback_success_count / max(1, fallback_count)",
    }
    # 统计兜底尝试 / 成功样本数。attempted 比 succeeded 更适合衡量 LoRA
    # first-pass 结构健康度；succeeded 只说明兜底是否救回。
    fallback_count = sum(
        1 for r in predictions_records
        if (r.get("fallback") or {}).get("attempted") is True
    )
    fallback_success_count = sum(
        1 for r in predictions_records
        if (r.get("fallback") or {}).get("succeeded") is True
    )
    n_total = len(predictions_records) if predictions_records else len(samples)
    metrics = {
        "_metric_doc": metric_doc,
        "n_total": n_total,
        "n_keep": n_keep,
        "n_advance": n_adv,
        "keep_accuracy": n_keep_correct / max(1, n_keep),
        "advance_accuracy": n_adv_correct / max(1, n_adv),
        "early_advance_rate": n_early_adv / max(1, n_keep),
        "anchor12_sanity": anchor12_sanity,
        "per_scenario": {k: dict(v) for k, v in per_scenario.items()},
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / max(1, n_total),
        "fallback_success_count": fallback_success_count,
        "fallback_success_rate": fallback_success_count / max(1, fallback_count),
        "config": vars(args),
        "world_size": world_size,
    }
    # ---- 写 metrics.json ----
    metrics_path = out_paths["metrics_json"]
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[done] metrics written to {metrics_path}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_scenario"},
                     ensure_ascii=False, indent=2))

    # 兜底触发率 stdout 高亮：放在 metrics dump 之后单独一行，方便扫日志时一眼看到
    # LoRA 健康度。健康 ckpt 应该 0/N (0.0%)；非零说明 first-pass 缺段，需要回去查
    # 段切换权重或 teacher 分布（PROJECT_CONTEXT.md §18.5/18.7）。
    if args.fallback:
        print(
            f"[eval] partial-continue fallback: "
            f"attempted={fallback_count}/{n_total} ({fallback_count / max(1, n_total):.1%}), "
            f"succeeded={fallback_success_count}/{max(1, fallback_count)} "
            f"({fallback_success_count / max(1, fallback_count):.1%})"
            + ("  ← 0 表示 LoRA 自己出完整三段，健康" if fallback_count == 0 else "")
        )
    else:
        print("[eval] partial-continue fallback: DISABLED (--no-fallback)；"
              "metrics 里的 keep/advance accuracy 反映模型 first-pass 真实表现")

    # ---- 逐条 prediction 落盘（#5.5）----
    # 一行一条 JSON：方便用 `jq .error_kind` / pandas 直接做透视；diff 只挑 error_kind != "ok"，
    # 让人工查错时不被正确样本淹没。
    pred_path = out_paths["predictions_jsonl"]
    if pred_path is not None and predictions_records:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w", encoding="utf-8") as f:
            for row in predictions_records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[done] predictions written to {pred_path} (n={len(predictions_records)})")
    diff_path = out_paths["predictions_diff_jsonl"]
    if diff_path is not None and predictions_records:
        diff_rows = [r for r in predictions_records if r.get("error_kind") != "ok"]
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diff_path, "w", encoding="utf-8") as f:
            for row in diff_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # 顺便打印 error_kind 分布，让 stdout 直接给出错误结构概览。
        kinds = Counter(r.get("error_kind", "?") for r in predictions_records)
        print(f"[done] diff written to {diff_path} (n={len(diff_rows)}); error_kind={dict(kinds)}")

    # ---- TensorBoard 写入（默认关）----
    # 用户明确要求："tb 只需要步骤二（GoalGen）的"。这里默认 --no-tb；用户显式 --tb 才写。
    # 写入时仍然落到 eval_tb/<run_tag>/，与训练 OUTPUT_DIR/tb 同根。
    tb_dir = out_paths["tb_dir"]
    if args.tb and _TB_AVAILABLE:
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        try:
            # 用 ckpt step 作为 global_step（如 checkpoint-300 → step=300），
            # 让"同一 LoRA 不同 step 的多次 eval"在 TB 上形成一条横向曲线；
            # 缺 step 时退到 0，依旧能记录但不连成线。
            step = _infer_ckpt_step(args.lora_dir)
            writer.add_scalar("eval/keep_accuracy", metrics["keep_accuracy"], step)
            writer.add_scalar("eval/advance_accuracy", metrics["advance_accuracy"], step)
            writer.add_scalar("eval/early_advance_rate", metrics["early_advance_rate"], step)
            if anchor12_sanity.get("enabled") and anchor12_sanity.get("passed") is not None:
                writer.add_scalar("eval/anchor12_passed", float(bool(anchor12_sanity["passed"])), step)
            # by_scenario 拆开写：方便看哪个场景把整体指标拉下来了。
            for sc, counts in per_scenario.items():
                keep_total = max(1, counts.get("keep_total", 0))
                adv_total = max(1, counts.get("adv_total", 0))
                writer.add_scalar(f"eval_by_scenario/{sc}/keep_acc",
                                  counts.get("keep_correct", 0) / keep_total, step)
                writer.add_scalar(f"eval_by_scenario/{sc}/early_advance",
                                  counts.get("early_advance", 0) / keep_total, step)
                writer.add_scalar(f"eval_by_scenario/{sc}/adv_acc",
                                  counts.get("adv_correct", 0) / adv_total, step)
            # 文本：前 8 条 pred vs gt 写 markdown，方便在 TB Text 面板里直接对比。
            preview = "\n\n".join(_format_pred_markdown(r) for r in predictions_records[:8])
            writer.add_text("eval/samples_preview", preview, step)
            # 错误类型分布表 — diff 文件之外再 TB 留一份，方便趋势追踪。
            kinds = Counter(r.get("error_kind", "?") for r in predictions_records)
            writer.add_text("eval/error_kind_distribution",
                            "\n".join(f"- {k}: {v}" for k, v in sorted(kinds.items())),
                            step)
            print(f"[tb] eval scalars + text written to {tb_dir}")
        finally:
            writer.close()
    elif not args.tb:
        print("[tb] 默认不写 TB（本项目 TB 入口在步骤二 GoalGen）；需要时加 --tb。")
    elif not _TB_AVAILABLE:
        print("[tb] 警告：SummaryWriter 不可用（torch.utils.tensorboard 导入失败），跳过 TB 写入。")

    if full_dump_enabled and is_rank0(rank):
        # rank0 看不见其它 rank 的本地 dump_count；只汇报本 rank 的实际写入。
        # 用户跑单卡时 rank0 拿到全部 dump，多卡时各 rank 写各自的，目录里数一下即可。
        print(f"[dump] rank0 本地完整 dump 已写 {dump_count_local} 条到 {cases_dir}")

    cleanup_distributed()


def _infer_ckpt_step(lora_dir: str) -> int:
    """从 LoRA 目录名推 step；非 checkpoint-* 形态退到 0。"""
    if not lora_dir:
        return 0
    name = pathlib.Path(lora_dir).name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-", 1)[1])
        except (ValueError, IndexError):
            return 0
    return 0


def _format_pred_markdown(row: Dict[str, Any]) -> str:
    """把单条 prediction 渲染成 TB Text 面板可读的 markdown 片段。"""
    sc = row.get("scenario", "?")
    rid = row.get("run_id", "?")
    anc = row.get("anchor", "?")
    is_t = row.get("is_transition_sample", False)
    kind = row.get("error_kind", "?")
    lines = [
        f"**[{kind}] {sc}/{rid}/anchor={anc} (transition={is_t})**",
        f"- GT  : status={row.get('gt_status')} subgoal={row.get('gt_subgoal')}",
        f"- Pred: status={row.get('pred_status')} subgoal={row.get('pred_subgoal')}",
    ]
    raw = row.get("raw_text")
    if raw:
        # 截前 240 字符；TB Text 面板长 markdown 会被折叠，太长反而不好对比。
        raw_short = raw[:240].replace("\n", " ⏎ ")
        lines.append(f"- raw: `{raw_short}`")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
