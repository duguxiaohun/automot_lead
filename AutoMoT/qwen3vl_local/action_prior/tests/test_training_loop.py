"""真实训练循环的 CPU 故障/恢复测试；仅替换昂贵模型与数据 IO。"""

import ast
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior import train, runtime, launch
import qwen3vl_local.leadmot as leadmot


@dataclass
class TinyConfig:
    num_route_queries: int = 10
    num_waypoint_queries: int = 8
    rope_type: str = "mrope"
    dropout: float = 0.1
    use_bev: bool = True
    use_final_goal: bool = True
    use_subgoal: bool = False


class TinyDecoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self):
        x = self.dropout(self.weight.expand(1, 10, 2))
        return {"pred_route": x, "pred_future_waypoints": x[:, :8]}


class Loader(list):
    """模拟可设置独立 generator 的顺序 clip loader。"""

    generator = None


def lightweight_old_helpers():
    """执行原 helper AST，避免原 train 的 CARLA/laspy import；优化器/loss/EMA 用真实代码。"""
    path = Path(train.__file__).parents[1] / "leadmot/train.py"
    names = {
        "_DecoderEMA",
        "_make_scheduler",
        "_planning_loss",
        "_point_loss",
        "_compute_planning_metrics",
        "_optimizer_param_groups",
    }
    tree = ast.parse(path.read_text())
    selected = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    fake = ModuleType("qwen3vl_local.leadmot.train")
    fake.__dict__.update(
        torch=torch, math=math, F=F, contextmanager=contextmanager, Any=object
    )
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        fake.__dict__,
    )
    fake._dtype = lambda name: torch.float32
    fake._init_distributed = lambda: (0, 0, 1)

    def loader(rows, args, **kw):
        return (
            Loader(
                {
                    "sample": r,
                    "clip": {},
                    "gt_route": torch.zeros(10, 2),
                    "gt_waypoints": torch.zeros(8, 2),
                }
                for r in rows
            ),
            None,
        )

    fake._make_loader = loader
    return fake


@pytest.mark.parametrize("failure", ["mid_epoch", "final_validation"])
def test_interruption_resume_matches_uninterrupted(tmp_path, monkeypatch, failure, capsys):
    old = lightweight_old_helpers()
    # 本机没有 TensorBoard；仅替换日志 IO，训练/恢复/EMA 仍执行真实循环。
    tb = ModuleType("torch.utils.tensorboard")
    tb.SummaryWriter = lambda *a, **kw: SimpleNamespace(
        add_scalar=lambda *a, **kw: None,
        close=lambda: None,
        flush=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tb)
    monkeypatch.setitem(sys.modules, "qwen3vl_local.leadmot.train", old)
    monkeypatch.setattr(leadmot, "train", old, raising=False)
    monkeypatch.setattr(leadmot, "LeadMoTPlanningDecoder", TinyDecoder)
    monkeypatch.setattr(leadmot, "LeadMoTPlanningDecoderConfig", TinyConfig)
    monkeypatch.setattr(train, "training_device", lambda local: torch.device("cpu"))
    monkeypatch.setattr(launch, "ensure_gpu", lambda: 1)
    monkeypatch.setenv("ACTION_PRIOR_RUN_READY", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    contract = {
        "schema": "action_prior_base_kv_v1",
        "identity": "test",
        "phase1": {"path": "one"},
        "phase2": {"path": "two"},
    }
    monkeypatch.setattr(train, "build_contract", lambda args: contract)
    # 本测试只替换昂贵模型与其文件 IO；真实复制/删除源/搬迁验证在 test_lora_bundle.py。
    from qwen3vl_local.action_prior import lora_bundle
    monkeypatch.setattr(lora_bundle, "preserve_for_training", lambda contract, out: contract)
    dataset = tmp_path / "data"
    dataset.mkdir()
    rows = {}
    for split in ("train", "val", "test"):
        rows[split] = [
            dict(
                scenario=split,
                run_id="route",
                route_group=f"{split}/route",
                anchor=i,
                split=split,
            )
            for i in range(5 if split == "train" else 1)
        ]
        (dataset / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows[split])
        )
    monkeypatch.setattr(train, "read_rows", lambda args, split: rows[split])
    mode = {"failure": None, "calls": 0}

    class Runtime:
        device = torch.device("cpu")

        def __init__(self):
            self.prior = SimpleNamespace(last_audit=None)

        def forward_sample(self, sample, decoder, config, dtype, clip=None):
            if sample["split"] == "train":
                if mode["failure"] == "mid_epoch" and mode["calls"] == 2:
                    raise RuntimeError("injected crash")
                mode["calls"] += 1
            elif mode["failure"] == "final_validation":
                raise RuntimeError("injected crash")
            self.prior.last_audit = {
                "invalid": {"UE3": "disagreement"},
                "analysis_truncated": False,
            }
            return decoder()

    monkeypatch.setattr(runtime, "make_runtime", lambda *a: Runtime())

    def run(out, resume=""):
        argv = [
            "train",
            "--data-dir",
            str(dataset),
            "--output-dir",
            str(out),
            "--num-epochs",
            "1",
            "--grad-accum-steps",
            "2",
            "--num-workers",
            "0",
            "--val-steps",
            "100",
            "--save-steps",
            "1",
            "--logging-steps",
            "10",
            "--decoder-dtype",
            "float32",
            "--qwen-dtype",
            "float32",
        ]
        if resume:
            argv += ["--resume", str(resume)]
        monkeypatch.setattr(sys, "argv", argv)
        train.main()

    baseline = tmp_path / "baseline"
    run(baseline)
    output = capsys.readouterr().out
    # 总共只有三次更新，仍必须立即输出第一次真实更新；不能等默认十步。
    assert "step=1/3 loss=" in output
    assert "validation/done" in output
    history = (baseline / "progress/train_rank0.jsonl").read_text()
    assert '"stage": "train/micro_done"' in history
    progress_state = json.loads((baseline / "progress/train_rank0.json").read_text())
    assert progress_state["stage"] == "finished"
    assert progress_state["optimizer_step"] == 3
    reference = torch.load(baseline / "latest.pt", weights_only=False)
    mode.update(failure=failure, calls=0)
    interrupted = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="injected crash"):
        run(interrupted)
    failed_state = json.loads((interrupted / "progress/train_rank0.json").read_text())
    assert failed_state["stage"] == "failed"
    assert "injected crash" in failed_state["error"]
    partial = torch.load(interrupted / "latest.pt", weights_only=False)
    assert partial["cursor"].get("validation_pending", False) == (
        failure == "final_validation"
    )
    mode.update(failure=None, calls=0)
    run(interrupted, interrupted / "latest.pt")
    restored = torch.load(interrupted / "latest.pt", weights_only=False)
    assert restored["cursor"] == reference["cursor"] == {"epoch": 1, "micro": 0}
    assert restored["step"] == reference["step"] == 3
    assert (interrupted / "best.pt").is_file()
    audit = json.loads(next((interrupted / "epoch_audit").glob("*.json")).read_text())
    assert audit["samples"] == 5
    assert audit["prior/reason/disagreement"] == 5
    assert torch.equal(restored["decoder"]["weight"], reference["decoder"]["weight"])
    assert torch.equal(
        restored["ema_state_dict"]["shadow"]["weight"],
        reference["ema_state_dict"]["shadow"]["weight"],
    )
