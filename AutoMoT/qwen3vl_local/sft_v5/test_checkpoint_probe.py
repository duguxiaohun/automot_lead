"""SFT v5 自动 checkpoint probe 的轻量回归测试。

不加载 Qwen、不需要 CUDA，主要验证训练内复用模型时 adapter 上下文能正确恢复，以及
base/checkpoint/final 的 summary/comparison 产物具有稳定、可比较的口径。
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from contextlib import contextmanager

_AUTOMOT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOMOT_ROOT))

from qwen3vl_local.sft_v5.probe import _probe_inference_context, summarize_probe  # noqa: E402
from qwen3vl_local.sft_v5.train import _update_probe_comparison  # noqa: E402


class _FakeModel:
    """只实现 probe 上下文需要的最小 PEFT 模型接口。"""

    def __init__(self) -> None:
        """初始状态模拟正在训练且 adapter 已启用。"""

        self.training = True
        self.disable_depth = 0

    def eval(self) -> "_FakeModel":
        """模拟 probe 进入推理态。"""

        self.training = False
        return self

    def train(self) -> "_FakeModel":
        """模拟 probe 结束后恢复训练态。"""

        self.training = True
        return self

    @contextmanager
    def disable_adapter(self):
        """记录嵌套关闭深度，验证 context manager 不会泄漏 LoRA 状态。"""

        self.disable_depth += 1
        try:
            yield
        finally:
            self.disable_depth -= 1


class _FakeBundle:
    """模拟训练 bundle 的 model/unwrap 结构。"""

    def __init__(self) -> None:
        """创建供 probe context 使用的最小模型容器。"""

        self.model = _FakeModel()

    def unwrap(self) -> _FakeModel:
        """返回底层 fake model，与正式训练 bundle 接口一致。"""

        return self.model


def test_probe_context_restores_training_and_adapter() -> None:
    """base probe 退出后必须恢复 train 模式并重新启用 LoRA。"""

    bundle = _FakeBundle()
    with _probe_inference_context(bundle, disable_adapter=True):
        assert not bundle.model.training
        assert bundle.model.disable_depth == 1
    assert bundle.model.training
    assert bundle.model.disable_depth == 0


def test_summary_and_comparison_are_stable() -> None:
    """检查关键准确率分母，以及版本索引的 base/checkpoint/final 顺序。"""

    logs = [
        {
            "q1_triggered": True,
            "q1_rs_correct": True,
            "rs_gate_correct": True,
            "event_family_correct": True,
            "q2_triggered": True,
            "q2_event_correct": True,
            "q2_invalid_output": False,
            "q1_teacher_rs_correct": True,
            "teacher_event_family_correct": True,
            "q2_teacher_triggered": True,
            "q2_teacher_event_correct": True,
        },
        {
            "q1_triggered": True,
            "q1_rs_correct": False,
            "rs_gate_correct": False,
            "event_family_correct": False,
            "q2_triggered": False,
            "q2_event_correct": None,
            "q2_invalid_output": False,
            "q1_teacher_rs_correct": False,
            "teacher_event_family_correct": False,
            "q2_teacher_triggered": False,
            "q2_teacher_event_correct": None,
        },
    ]
    summary = summarize_probe(
        logs,
        student_enabled=True,
        teacher_enabled=True,
        student_adapter_dir="checkpoint-40",
        student_disable_adapter=False,
    )
    assert summary["q1_rs_accuracy"] == 0.5
    assert summary["rs_slow_accuracy"] == 0.5
    assert summary["event_family_accuracy"] == 1.0
    # 旧 key 仅是 EVENT family 的兼容别名，不再是 Q1 输出。
    assert summary["q1_abnormal_accuracy"] == 1.0
    assert summary["q2_trigger_rate"] == 0.5
    assert summary["q2_event_accuracy_when_triggered"] == 1.0
    assert summary["teacher_q1_rs_accuracy"] == 0.5
    assert summary["teacher_q2_trigger_rate"] == 0.5
    assert summary["teacher_q2_event_accuracy"] == 1.0
    static_summary = summarize_probe(
        logs,
        student_enabled=False,
        teacher_enabled=False,
        student_adapter_dir=None,
        student_disable_adapter=False,
    )
    assert static_summary["q1_rs_accuracy"] is None
    assert static_summary["q2_trigger_rate"] is None
    assert static_summary["teacher_q1_rs_accuracy"] is None
    assert static_summary["teacher_q2_trigger_rate"] is None
    assert static_summary["teacher_q2_event_accuracy"] is None

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _update_probe_comparison(root, {"name": "final", "global_step": 81, "status": "ok"})
        _update_probe_comparison(root, {"name": "checkpoint-000040", "global_step": 40, "status": "ok"})
        _update_probe_comparison(root, {"name": "base", "global_step": 0, "status": "ok"})
        # 同名记录必须替换，不能让重跑 probe 产生重复版本。
        _update_probe_comparison(root, {"name": "checkpoint-000040", "global_step": 40, "status": "error"})
        payload = json.loads((root / "probes" / "comparison.json").read_text(encoding="utf-8"))
        assert [item["name"] for item in payload["entries"]] == ["base", "checkpoint-000040", "final"]
        assert payload["entries"][1]["status"] == "error"


def main() -> None:
    """直接运行全部轻量测试。"""

    test_probe_context_restores_training_and_adapter()
    test_summary_and_comparison_are_stable()
    print("[ok] checkpoint probe policy")


if __name__ == "__main__":
    main()
