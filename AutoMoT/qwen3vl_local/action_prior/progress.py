"""训练/评测的 CPU 进度观察器；不调用 CUDA、随机数或分布式同步。"""

from contextvars import ContextVar
from functools import wraps
import json
import math
import os
from pathlib import Path
import threading
import time


_current = ContextVar("action_progress", default=None)


class Progress:
    """每个 rank 独立报告；心跳只表示 Python 观察线程存活，不代表更新完成。"""

    def __init__(self, interval=30.0):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("ACTION_PROGRESS_SECONDS must be finite and positive")
        self.interval = interval
        self.rank = int(os.environ.get("RANK", "0"))
        self.detail = False
        self.path = None
        self.state = {}
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.last_print = self.stage_started = time.monotonic()
        self.thread = None

    def configure(self, out, label="train"):
        """保存各 rank 最新状态；独立 eval 不覆盖训练状态。"""
        with self.lock:
            self.path = Path(out) / "progress" / f"{label}_rank{self.rank}.json"
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def set(self, stage, announce=False, **values):
        """仅显式完成事件增加进度；阶段改变本身不增加样本/更新计数。"""
        with self.lock:
            self.stage_started = time.monotonic()
            self.state.update(values, stage=stage)
            if announce or self.detail or time.monotonic() - self.last_print >= self.interval:
                self._emit("event")

    def _emit(self, kind):
        """锁内打印并原子替换状态文件；观察 IO 故障不打断模型训练。"""
        now = time.monotonic()
        payload = dict(self.state, rank=self.rank, pid=os.getpid(), kind=kind,
                       timestamp=time.time(), stage_elapsed_s=round(now - self.stage_started, 1))
        print("[progress] " + json.dumps(payload, ensure_ascii=False), flush=True)
        self.last_print = now
        if self.path:
            tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
            try:
                tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                tmp.replace(self.path)
            except OSError as exc:
                print(f"[progress IO warning] {exc}", flush=True)

    def _heartbeat(self):
        """主线程卡在生成/加载/同步时仍报告最后已知阶段，不伪造新进展。"""
        while not self.stop.wait(min(self.interval, 1.0)):
            with self.lock:
                if self.state and time.monotonic() - self.last_print >= self.interval:
                    self._emit("heartbeat_still_running")

    def __enter__(self):
        self.token = _current.set(self)
        self.thread = threading.Thread(target=self._heartbeat, daemon=True, name="action-progress")
        self.thread.start()
        return self

    def __exit__(self, typ, exc, tb):
        self.stop.set()
        self.thread.join(timeout=1)
        self.detail = False
        self.set("failed" if exc else "finished", announce=True,
                 last_stage=self.state.get("stage"),
                 error=f"{typ.__name__}: {exc}" if exc else None)
        _current.reset(self.token)


def current():
    """未启动观察器的推理/单元测试保持原行为。"""
    return _current.get()


def report(stage, announce=False, **values):
    """只接收 CPU 标量/短文本，不能传入 tensor 或完整样本。"""
    progress = current()
    if progress is not None:
        progress.set(stage, announce=announce, **values)


def observed(function):
    """入口生命周期内自动停止心跳，包括异常退出；嵌套调用复用观察器。"""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if current() is not None:
            return function(*args, **kwargs)
        with Progress(float(os.environ.get("ACTION_PROGRESS_SECONDS", "30"))):
            report("startup", announce=True)
            return function(*args, **kwargs)
    return wrapped
