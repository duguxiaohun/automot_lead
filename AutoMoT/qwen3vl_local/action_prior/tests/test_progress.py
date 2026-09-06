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


def test_console_throttles_details_and_labels_local_loss(tmp_path, monkeypatch, capsys):
    import qwen3vl_local.action_prior.progress as module
    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setenv("RANK", "0")
    progress = Progress(interval=30)
    progress.configure(tmp_path)
    progress.set("startup", announce=True)
    capsys.readouterr()
    progress.detail = True
    for loss in (2.0, 4.0):
        progress.set("condition/phase1_question", announce=True, phase1_path="private/long/path")
        progress.set("train/micro_done", last_sample_loss=loss)
    assert capsys.readouterr().out == ""
    now[0] += 31
    progress.set("condition/phase2_question")
    output = capsys.readouterr().out
    assert "loss(rank0,2样本均值)=3.0000" in output
    assert "private/long/path" not in output
    assert len(output.splitlines()) == 1
    # 非 rank0 仍写详细文件，但正常进度不进入终端。
    monkeypatch.setenv("RANK", "1")
    other = Progress()
    other.configure(tmp_path)
    other.set("train/epoch_start", announce=True)
    assert capsys.readouterr().out == ""
    assert json.loads(other.path.read_text())["rank"] == 1
