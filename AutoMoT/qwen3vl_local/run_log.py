"""Small stdout/stderr tee helper for qwen3vl_local entrypoints."""

from __future__ import annotations

import os
import pathlib
import sys
from typing import TextIO


class _Tee:
    """Write one stream to both terminal and a log file."""

    def __init__(self, original: TextIO, log_file: TextIO) -> None:
        self._original = original
        self._log_file = log_file

    def write(self, text: str) -> int:
        written = self._original.write(text)
        self._log_file.write(text)
        return written

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return self._original.isatty()

    def __getattr__(self, name: str):
        return getattr(self._original, name)


def install_output_log(output_dir: str | os.PathLike[str],
                       filename: str = "log.txt",
                       env_flag: str = "QWEN3VL_LOG_ACTIVE",
                       rank: int | None = None) -> pathlib.Path | None:
    """Append terminal stdout/stderr to ``output_dir/filename``.

    Shell launchers set ``QWEN3VL_LOG_ACTIVE=1`` after installing their own tee,
    so direct Python entrypoints get a log file without double-logging wrapped runs.
    Set ``QWEN3VL_LOG_TO_FILE=0`` to disable this fallback.

    多卡（DDP / torchrun）保护：只有 rank 0 进程会安装 Tee，其它 rank 直接返回 None，
    避免多个 rank 同时 append 同一 log.txt 造成乱码。需要保留所有 rank 输出时，
    调用方应自行传 ``filename=f"log_rank{rank}.txt"`` 之类按 rank 分文件。
    """
    if os.environ.get("QWEN3VL_LOG_TO_FILE", "1") == "0":
        return None
    if os.environ.get(env_flag):
        return None
    # 非 rank0 进程默认不写 log.txt，防止 DDP 下多 rank 互相覆盖。
    if rank is not None and rank != 0:
        return None
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / filename
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    os.environ[env_flag] = "1"
    sys.stdout = _Tee(sys.stdout, log_file)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.stderr, log_file)  # type: ignore[assignment]
    prefix = f"[log][rank={rank}]" if rank is not None else "[log]"
    print(f"{prefix} tee stdout/stderr to {log_path}", flush=True)
    return log_path
