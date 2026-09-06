"""慢调用期间心跳必须可见，但不能冒充 optimizer 完成或遗留观察线程。"""

import json
from pathlib import Path
import sys
import threading
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior.progress import Progress, current, report


def test_heartbeat_during_blocked_work_preserves_step(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "2")
    progress = Progress(interval=0.02)
    heartbeat = threading.Event()
    original = progress._emit

    def emit(kind):
        original(kind)
        if kind == "heartbeat_still_running":
            heartbeat.set()

    monkeypatch.setattr(progress, "_emit", emit)
    with progress:
        progress.configure(tmp_path)
        report("condition/base_analysis", announce=True, optimizer_step=0)
        # 模拟主线程等待模型；只有观察线程能发布心跳。
        assert heartbeat.wait(2)
        state = json.loads((tmp_path / "progress/train_rank2.json").read_text())
        assert state["kind"] == "heartbeat_still_running"
        assert state["stage"] == "condition/base_analysis"
        assert state["optimizer_step"] == 0
    assert not progress.thread.is_alive()
    assert current() is None


def test_exception_stops_observer_and_preserves_last_progress(tmp_path):
    progress = Progress()
    with pytest.raises(RuntimeError, match="model failed"):
        with progress:
            progress.configure(tmp_path)
            report("train/backward_sync", optimizer_step=4)
            raise RuntimeError("model failed")
    assert not progress.thread.is_alive()
    assert current() is None
    state = json.loads(progress.path.read_text())
    assert state["optimizer_step"] == 4
    assert state["stage"] == "failed"
