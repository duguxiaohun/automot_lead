"""SFT v4 off-policy replay 文件队列。

collector 和 learner 只通过这个模块约定磁盘数据交换协议：

- collector 写 ``pending/*.tmp``，完成校验后原子 rename 到 ``ready/*.jsonl``；
- learner 从 ``ready`` 中独立 ``random.choice`` 抽 trajectory；
- FIFO 驱逐按 ready 文件 mtime 删除最旧项；
- collector 抢 episode 使用 ``mkdir`` 实现的跨进程文件锁，不依赖 DDP / NCCL。
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

SCHEMA = "sft_v4_rollout_v1"


@dataclass
class ReplayStats:
    """ready 队列的轻量统计，供 learner TensorBoard 和日志使用。"""

    ready_count: int
    pending_count: int
    failed_count: int
    avg_age_minutes: float


def ensure_replay_dirs(replay_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
    """创建并返回 replay 的标准子目录。"""

    root = pathlib.Path(replay_dir)
    dirs = {
        "root": root,
        "pending": root / "pending",
        "ready": root / "ready",
        "used": root / "used",
        "failed": root / "failed",
        "state": root / "state",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def ready_files(replay_dir: pathlib.Path) -> List[pathlib.Path]:
    """列出当前可训练 trajectory 文件。"""

    dirs = ensure_replay_dirs(replay_dir)
    return sorted(p for p in dirs["ready"].glob("*.jsonl") if p.is_file())


def replay_stats(replay_dir: pathlib.Path) -> ReplayStats:
    """统计 ready/pending/failed 数量和 ready 文件平均年龄。"""

    dirs = ensure_replay_dirs(replay_dir)
    now = time.time()
    ready = ready_files(replay_dir)
    ages = [max(0.0, now - p.stat().st_mtime) / 60.0 for p in ready]
    pending_count = len(list(dirs["pending"].glob("*")))
    failed_count = len(list(dirs["failed"].glob("*")))
    return ReplayStats(
        ready_count=len(ready),
        pending_count=pending_count,
        failed_count=failed_count,
        avg_age_minutes=sum(ages) / len(ages) if ages else 0.0,
    )


@contextmanager
def directory_lock(lock_dir: pathlib.Path, *, stale_seconds: float = 1800.0) -> Iterator[None]:
    """用原子 ``mkdir`` 实现跨平台文件锁。

    Windows 和 Linux 上目录创建都是原子的；锁持有者异常退出时，超过 ``stale_seconds``
    的旧锁会被清理。这里只保护很短的 counter / FIFO 元数据操作。
    """

    lock_dir = pathlib.Path(lock_dir)
    lock_parent = lock_dir.parent
    lock_parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            try:
                age = time.time() - lock_dir.stat().st_mtime
                if age > stale_seconds:
                    shutil.rmtree(lock_dir, ignore_errors=True)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(0.05)
    try:
        yield
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def claim_episode_index(replay_dir: pathlib.Path, *, total_episodes: int) -> int:
    """返回下一个 collector 应采集的 episode idx，counter 会自动 wrap。"""

    if total_episodes <= 0:
        raise ValueError("total_episodes must be positive")
    dirs = ensure_replay_dirs(replay_dir)
    counter_path = dirs["state"] / "episode_counter.json"
    with directory_lock(dirs["state"] / "episode_counter.lock"):
        if counter_path.exists():
            payload = json.loads(counter_path.read_text(encoding="utf-8"))
            value = int(payload.get("next", 0))
        else:
            value = 0
        idx = value % int(total_episodes)
        tmp = counter_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"next": value + 1}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, counter_path)
    return idx


def make_trajectory_name(collector_id: str, run_id: str) -> str:
    """生成 ready 文件名；run_id 中的路径字符会被替换。"""

    safe_collector = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in collector_id)
    safe_run = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in run_id)
    return f"{safe_collector}_{int(time.time() * 1000)}_{safe_run}.jsonl"


def validate_trajectory(records: List[Dict[str, Any]]) -> None:
    """校验 trajectory 的最小训练契约。"""

    if not records:
        raise ValueError("empty trajectory")
    header = records[0]
    if header.get("schema") != SCHEMA or header.get("kind") != "header":
        raise ValueError(f"invalid trajectory header: {header}")
    frame_count = int(header.get("frame_count", 0))
    frames = [r for r in records[1:] if r.get("kind") == "frame"]
    if frame_count != len(frames):
        raise ValueError(f"frame_count mismatch: header={frame_count}, actual={len(frames)}")
    if not frames:
        raise ValueError("trajectory contains no frame records")
    for i, frame in enumerate(frames):
        if "image_paths" not in frame or not frame["image_paths"]:
            raise ValueError(f"frame {i} missing image_paths")
        if "memory_before" not in frame:
            raise ValueError(f"frame {i} missing memory_before")
        targets = frame.get("teacher_targets") or {}
        if not targets.get("step1") or not targets.get("step2"):
            raise ValueError(f"frame {i} missing teacher step1/step2 targets")
        if bool(frame.get("step3_ran")) and not targets.get("step3"):
            raise ValueError(f"frame {i} step3_ran but missing step3 target")


def write_trajectory(
    replay_dir: pathlib.Path,
    records: List[Dict[str, Any]],
    *,
    collector_id: str,
    run_id: str,
    capacity: int,
) -> pathlib.Path:
    """原子写入一条 trajectory，并按 FIFO 容量驱逐旧 ready 文件。"""

    validate_trajectory(records)
    dirs = ensure_replay_dirs(replay_dir)
    name = make_trajectory_name(collector_id, run_id)
    pending = dirs["pending"] / f"{name}.tmp"
    ready = dirs["ready"] / name
    with open(pending, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(pending, ready)
    evict_old(replay_dir, capacity=capacity)
    return ready


def move_failed(replay_dir: pathlib.Path, pending_path: Optional[pathlib.Path], *, reason: str) -> pathlib.Path:
    """记录 collector 写入失败原因。"""

    dirs = ensure_replay_dirs(replay_dir)
    failed = dirs["failed"] / f"failed_{int(time.time() * 1000)}.txt"
    failed.write_text(str(reason) + "\n", encoding="utf-8")
    if pending_path is not None and pending_path.exists():
        try:
            os.replace(pending_path, dirs["failed"] / pending_path.name)
        except OSError:
            pass
    return failed


def read_trajectory(path: pathlib.Path) -> List[Dict[str, Any]]:
    """读取并校验一条 trajectory。"""

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    validate_trajectory(records)
    return records


def sample_ready_file(replay_dir: pathlib.Path, rng: random.Random) -> Optional[pathlib.Path]:
    """从 ready 队列随机抽一条 trajectory。"""

    files = ready_files(replay_dir)
    if not files:
        return None
    return rng.choice(files)


def evict_old(replay_dir: pathlib.Path, *, capacity: int) -> None:
    """按 mtime FIFO 驱逐超出容量的 ready 文件。"""

    if capacity <= 0:
        return
    dirs = ensure_replay_dirs(replay_dir)
    with directory_lock(dirs["state"] / "evict.lock"):
        files = ready_files(replay_dir)
        overflow = len(files) - int(capacity)
        if overflow <= 0:
            return
        victims = sorted(files, key=lambda p: p.stat().st_mtime)[:overflow]
        for path in victims:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def write_json_atomic(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    """原子写 JSON 文件，用于 snapshot pointer / checkpoint metadata。"""

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def iter_frame_records(records: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """遍历 trajectory 中的 frame 行。"""

    for rec in records:
        if rec.get("kind") == "frame":
            yield rec
