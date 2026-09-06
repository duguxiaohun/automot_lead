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
        self.last_console = float("-inf")
        self.loss_sum = 0.0
        self.loss_count = 0
        self.last_loss_console = None
        self.update_announced = False
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
            if stage == "train/micro_done" and "last_sample_loss" in values:
                self.loss_sum += values["last_sample_loss"]
                self.loss_count += 1
            if announce or self.detail or time.monotonic() - self.last_print >= self.interval:
                self._emit("event")

    def _emit(self, kind):
        """锁内打印并原子替换状态文件；观察 IO 故障不打断模型训练。"""
        now = time.monotonic()
        payload = dict(self.state, rank=self.rank, pid=os.getpid(), kind=kind,
                       timestamp=time.time(), stage_elapsed_s=round(now - self.stage_started, 1))
        stage = payload.get("stage", "startup")
        # 详细阶段留在文件；终端仅 rank0 定时单行，错误仍由每个 rank 报告。
        first_update = stage == "train/update_done" and not self.update_announced
        essential = stage in ("startup", "train/epoch_start", "validation/start",
                              "validation/done", "checkpoint/saved", "finished", "failed")
        if stage == "train/update_done":
            self.update_announced = True
        if stage == "failed" or (self.rank == 0 and (
                essential or first_update or now - self.last_console >= self.interval)):
            print(self._console_line(payload), flush=True)
            self.last_console = now
        self.last_print = now
        if self.path:
            tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
            try:
                tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                tmp.replace(self.path)
                with self.path.with_suffix(".jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except OSError as exc:
                print(f"[progress IO warning] {exc}", flush=True)

    def _console_line(self, payload):
        """时间窗口 loss 是 rank0 已完成样本均值，不冒充全 rank optimizer loss。"""
        stage = payload["stage"]
        labels = {
            "condition/phase1_question": "LoRA-Phase1", "condition/phase2_question": "LoRA-Phase2",
            "condition/base_analysis": "base分析", "condition/base_review": "分析复核",
            "condition/base_final_prefill": "base-prefill", "condition/cache_lookup_or_lock": "缓存读取/等待",
            "train/backward_sync": "反向/同步", "train/backward_accumulate": "反向/累积",
        }
        parts = [time.strftime("%H:%M:%S"), f"[rank{self.rank}]", labels.get(stage, stage)]
        if "epoch" in payload:
            parts += [f"epoch={payload['epoch']}/{payload['epochs']}",
                      f"step={payload.get('optimizer_step', 0)}/{payload['step_limit']}",
                      f"micro={payload.get('rank_completed_micro', 0)}/{payload['rank_epoch_samples']}"]
        if stage.startswith("validation/"):
            parts.append(f"rank_samples={payload.get('rank_evaluated', 0)} total={payload.get('evaluation_samples', '?')}")
            if stage == "validation/done":
                parts.append(f"val_loss(global)={payload['validation_loss']:.4f}")
        elif self.loss_count:
            self.last_loss_console = self.loss_sum / self.loss_count
            parts.append(f"loss(rank{self.rank},{self.loss_count}样本均值)={self.last_loss_console:.4f}")
            self.loss_sum, self.loss_count = 0.0, 0
        elif self.last_loss_console is not None:
            parts.append(f"loss(上次rank{self.rank}均值)={self.last_loss_console:.4f} 本间隔无新样本")
        else:
            parts.append("loss=等待样本完成")
        if "lr" in payload:
            parts.append(f"lr={payload['lr']:.3e}")
        parts.append(f"阶段耗时={payload['stage_elapsed_s']:.1f}s")
        if payload.get("error"):
            parts.append(payload["error"])
        return " | ".join(parts)

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
