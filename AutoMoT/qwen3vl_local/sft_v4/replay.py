"""SFT v4 off-policy replay 文件队列。

collector 和 learner 只通过这个模块约定磁盘数据交换协议：

- collector 写 ``pending/*.tmp``，完成校验后原子 rename 到 ``ready/*.jsonl``；
- learner 从 ``ready`` 中独立 ``random.choice`` 抽 trajectory；
- FIFO 驱逐按 trajectory header 里的 ``created_at`` 删除最旧项；
- collector 抢 episode 使用 ``mkdir`` 实现的跨进程文件锁，不依赖 DDP / NCCL。

使用方式：

1. collector 启动时调用 ``ensure_replay_dirs`` 初始化目录。
2. 每条 episode 前调用 ``claim_episode_index`` 抢一个全局递增 idx。
3. rollout 完成后把 header + frame records 交给 ``write_trajectory``。
4. learner 通过 ``sample_ready_file`` + ``read_trajectory`` 读取训练样本。

这个模块故意只依赖 Python 标准库，目的是让 replay 队列在 collector / learner 崩溃、
重启、没有 torch.distributed 的情况下仍然可用。
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

# SCHEMA bump 历史：
#  - v1: 初代 trajectory（无 road_structure，单层 scene + status + subgoal）。
#  - v2: 加入 ROAD_STRUCTURE layer-1（PLAN §12）。memory 结构、step1 字段、
#        触发链字段（step2_ran / rs_flip / memory_after_step1 等）都变更。
#  v1 与 v2 不二进制兼容：learner 加载 v1 traj 会被 ``validate_trajectory`` 拒收，
#  collector 重新攒一波 v2 trajectory 即可。SCHEMA_LEGACY 保留作识别旧文件用。
SCHEMA = "sft_v4_rollout_v2"
SCHEMA_LEGACY = ("sft_v4_rollout_v1",)


@dataclass
class ReplayStats:
    """ready 队列的轻量统计，供 learner TensorBoard 和日志使用。

    ``avg_age_minutes`` 是简单 staleness 指标：数值越大，说明 learner 读到的样本越旧，
    也就越偏 off-policy。这个指标不参与训练，只用于调 collector 数量和 FIFO 容量。
    """

    ready_count: int
    pending_count: int
    failed_count: int
    avg_age_minutes: float


def ensure_replay_dirs(replay_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
    """创建并返回 replay 的标准子目录。

    目录语义：

    - ``pending``：collector 正在写的临时文件，learner 永远不读。
    - ``ready``：完整写入并通过 schema 校验的 trajectory，learner 只读这里。
    - ``failed``：坏 trajectory 或 collector 异常原因，方便远端排查。
    - ``state``：跨进程 counter / lock 等小元数据。

    ``used`` 目前预留不用，因为 v4 允许 replay 重抽；保留这个目录是为了之后如果要做
    “消费后归档”不用改目录协议。
    """

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
    """列出当前可训练 trajectory 文件。

    返回时做排序只为日志/调试稳定；真正抽样由 learner 的 ``random.choice`` 完成。
    """

    dirs = ensure_replay_dirs(replay_dir)
    return sorted(p for p in dirs["ready"].glob("*.jsonl") if p.is_file())


def _trajectory_created_at(path: pathlib.Path) -> float:
    """读取 trajectory 写入时刻，供 staleness 统计与 FIFO 驱逐共用。

    ``created_at`` 写在 header 第一行，比文件 mtime 更接近真实采样时间；如果遇到旧文件
    或损坏文件，才回退到 mtime，避免监控/驱逐因为单个坏样本中断。
    """

    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        if first:
            payload = json.loads(first)
            created_at = payload.get("created_at")
            if created_at is not None:
                return float(created_at)
    except Exception:
        pass
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return time.time()


def replay_stats(replay_dir: pathlib.Path) -> ReplayStats:
    """统计 ready/pending/failed 数量和 ready 文件平均年龄。

    这里每次直接扫目录，开销相对 Qwen forward 可以忽略；避免维护额外状态文件，能减少
    collector 崩溃后元数据与真实文件不一致的机会。
    """

    dirs = ensure_replay_dirs(replay_dir)
    now = time.time()
    ready = ready_files(replay_dir)
    ages = [max(0.0, now - _trajectory_created_at(p)) / 60.0 for p in ready]
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
    """返回下一个 collector 应采集的 episode idx，counter 会自动 wrap。

    多个 collector 会同时调用这个函数。文件锁保护 ``episode_counter.json`` 的读-改-写，
    因此每个调用者拿到的 ``value`` 都不同；最终返回 ``value % total_episodes``，
    让采集在没有 epoch 概念的 off-policy 训练里无限循环。
    """

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
        # 先写 tmp 再 os.replace，避免进程在写 JSON 中途崩溃时留下半截 counter。
        tmp = counter_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"next": value + 1}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, counter_path)
    return idx


def make_trajectory_name(collector_id: str, run_id: str) -> str:
    """生成 ready 文件名；run_id 中的路径字符会被替换。

    文件名只用于人工排查，不作为训练语义来源；真实 episode 信息在 header 里。
    """

    safe_collector = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in collector_id)
    safe_run = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in run_id)
    return f"{safe_collector}_{int(time.time() * 1000)}_{safe_run}.jsonl"


def validate_trajectory(records: List[Dict[str, Any]]) -> None:
    """校验 trajectory 的最小训练契约。

    这里不做“语义正确性”校验，例如 scene/status 是否真的属于 GT 序列；那类检查在
    collector 构造 frame record 时完成。replay 层只保证 learner 读到的文件具备足够字段，
    能安全进入 teacher-forced loss。
    """

    if not records:
        raise ValueError("empty trajectory")
    header = records[0]
    schema = header.get("schema")
    if header.get("kind") != "header":
        raise ValueError(f"invalid trajectory header (kind): {header}")
    if schema != SCHEMA:
        # 旧 v1 trajectory 与 v2 字段不兼容（无 road_structure / memory_after_step1 等）。
        # 报错信息显式区分 "legacy 已知格式" vs "完全未知字段"，方便排查。
        if schema in SCHEMA_LEGACY:
            raise ValueError(
                f"trajectory schema {schema!r} is a deprecated legacy version; "
                f"expected {SCHEMA!r}. Re-collect with the current collector."
            )
        raise ValueError(
            f"trajectory schema {schema!r} is unknown; expected {SCHEMA!r}."
        )
    frame_count = int(header.get("frame_count", 0))
    frames = [r for r in records[1:] if r.get("kind") == "frame"]
    if frame_count != len(frames):
        raise ValueError(f"frame_count mismatch: header={frame_count}, actual={len(frames)}")
    if not frames:
        raise ValueError("trajectory contains no frame records")
    for i, frame in enumerate(frames):
        if "image_paths" not in frame or not frame["image_paths"]:
            raise ValueError(f"frame {i} missing image_paths")
        if "memory_before" not in frame and "memory_before_frame" not in frame:
            raise ValueError(f"frame {i} missing memory_before/memory_before_frame")
        mem_before = frame.get("memory_before") or frame.get("memory_before_frame") or {}
        if "road_structure" not in mem_before:
            # v2 schema 的核心变化就是三层 memory。这里宁可硬拒旧文件，也不要
            # 默认填 JUNCTION：默认值会让 learner 构造出“看似能跑、实则错桶”的
            # step2 prompt，问题会被延迟到训练指标里才暴露。
            raise ValueError(f"frame {i} memory_before missing road_structure")
        targets = frame.get("teacher_targets") or {}
        step1_target = targets.get("step1") or frame.get("teacher_step1_target") or frame.get("teacher_step1_text")
        if not step1_target:
            raise ValueError(f"frame {i} missing teacher step1 target")
        step2_flag = bool(frame.get("step2_ran", frame.get("step2_fired", False)))
        step2_target = targets.get("step2") or frame.get("teacher_step2_target") or frame.get("teacher_step2_raw")
        if step2_flag and not step2_target:
            raise ValueError(f"frame {i} step2_ran but missing step2 target")
        if step2_flag and "memory_after_step1" not in frame:
            # step2_fired=True 时必须能复现 collector 当时的候选表；候选表唯一来源
            # 是 step1 更新后的 memory_after_step1。
            raise ValueError(f"frame {i} step2_ran but missing memory_after_step1")
        step3_flag = bool(frame.get("step3_ran", frame.get("step3_fired", False)))
        step3_target = targets.get("step3") or frame.get("teacher_step3_target") or frame.get("teacher_step3_raw")
        if step3_flag and not step3_target:
            raise ValueError(f"frame {i} step3_ran but missing step3 target")


def write_trajectory(
    replay_dir: pathlib.Path,
    records: List[Dict[str, Any]],
    *,
    collector_id: str,
    run_id: str,
    capacity: int,
) -> pathlib.Path:
    """原子写入一条 trajectory，并按 FIFO 容量驱逐旧 ready 文件。

    learner 只扫描 ``ready``，所以写入必须先落到 ``pending``，待完整写完后再
    ``os.replace`` 到 ``ready``。这样 learner 永远不会读到半行 JSON 或 frame_count
    尚未补齐的文件。
    """

    validate_trajectory(records)
    dirs = ensure_replay_dirs(replay_dir)
    name = make_trajectory_name(collector_id, run_id)
    pending = dirs["pending"] / f"{name}.tmp"
    ready = dirs["ready"] / name
    with open(pending, "w", encoding="utf-8") as f:
        for rec in records:
            # 一行一条 JSON，方便 tail / grep，也方便单条损坏时定位到 frame。
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(pending, ready)
    evict_old(replay_dir, capacity=capacity)
    return ready


def move_failed(replay_dir: pathlib.Path, pending_path: Optional[pathlib.Path], *, reason: str) -> pathlib.Path:
    """记录 collector 写入失败原因。

    collector 采集失败时不能让 learner 卡住；这里把错误原因写到 ``failed``，主循环
    稍后继续抢下一条 episode。``pending_path`` 可为空，因为很多失败发生在真正开始
    写临时文件之前。
    """

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
    """读取并校验一条 trajectory。

    learner 读之前不加文件锁：ready 文件写入已经是原子 rename，读写不会并发。若文件
    在 FIFO 驱逐时被删，调用方捕获 ``FileNotFoundError`` 后重抽即可。
    """

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    validate_trajectory(records)
    return records


def sample_ready_file(replay_dir: pathlib.Path, rng: random.Random) -> Optional[pathlib.Path]:
    """从 ready 队列随机抽一条 trajectory。

    两个 learner rank 各自随机抽样，不强制去重。off-policy replay 允许重抽，偶尔同一
    trajectory 被两个 rank 同步训练一次，等价于 batch 中重复样本，成本远低于做分布式
    sampler 协议。
    """

    files = ready_files(replay_dir)
    if not files:
        return None
    return rng.choice(files)


def evict_old(replay_dir: pathlib.Path, *, capacity: int) -> None:
    """按 header ``created_at`` FIFO 驱逐超出容量的 ready 文件。

    驱逐只删 ``ready`` 中最旧的完整文件，不碰 ``pending``。多个 collector 同时写完会
    同时触发驱逐，因此这里也用短文件锁保护，避免两个进程同时删除同一批文件造成噪声。
    这里与 ``replay_stats`` 使用同一个时间口径，保证 TensorBoard 上看到的 staleness
    和实际被驱逐的 FIFO 顺序一致。
    """

    if capacity <= 0:
        return
    dirs = ensure_replay_dirs(replay_dir)
    with directory_lock(dirs["state"] / "evict.lock"):
        files = ready_files(replay_dir)
        overflow = len(files) - int(capacity)
        if overflow <= 0:
            return
        # 与 replay_stats 保持同一年龄口径：优先使用 trajectory header created_at。
        victims = sorted(files, key=_trajectory_created_at)[:overflow]
        for path in victims:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def write_json_atomic(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    """原子写 JSON 文件，用于 snapshot pointer / checkpoint metadata。

    当前 replay 主流程主要用 jsonl；这个 helper 预留给需要整体 JSON 元数据的地方。
    """

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def iter_frame_records(records: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """遍历 trajectory 中的 frame 行。

    header 行只描述 episode，不参与 loss；learner 用这个 helper 避免每处都手写
    ``kind == "frame"`` 判断。
    """

    for rec in records:
        if rec.get("kind") == "frame":
            yield rec
