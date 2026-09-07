"""日志回归：正常域外、未确认与字段原因使用清晰且一致的分母。"""

from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior.train import audit_counts, format_prior_metrics, metrics_from_counts


def test_mixed_window_prior_metrics():
    """全接受、纯域外、混合失败及缺标签按样本计数，原因按字段计数。"""
    audits = [
        ({}, False),
        ({"UE1": "domain_inapplicable", "UE3": "domain_inapplicable"}, False),
        ({"UE6": "domain_inapplicable", "RS1": "disagreement"}, True),
        ({"RS1": "dataset_label_missing", "UE1": "dataset_label_missing"}, False),
    ]
    counts = Counter()
    for invalid, fallback in audits:
        counts.update(audit_counts(dict(
            invalid=invalid, analysis_truncated=False, analysis_fallback=fallback,
        )))
    metrics = metrics_from_counts(counts)
    assert metrics["samples"] == 4
    assert metrics["prior/invalid_samples"] == 0.75
    assert metrics["prior/domain_only_samples"] == 0.25
    assert metrics["prior/unconfirmed_samples"] == 0.5
    assert metrics["prior/invalid_samples"] == (
        metrics["prior/domain_only_samples"] + metrics["prior/unconfirmed_samples"]
    )
    text = format_prior_metrics(metrics)
    assert "invalid_any=0.750 domain_only=0.250 unconfirmed=0.500 fallback=0.250" in text
    assert json.loads(text.split("reason_fields=", 1)[1]) == {
        "domain_inapplicable": 3, "disagreement": 1, "dataset_label_missing": 2,
    }


def test_all_confirmed_window_has_zero_rates():
    """纯正常窗口也显式输出零，不缺键、不把零误当未知。"""
    metrics = metrics_from_counts(audit_counts(dict(invalid={}, analysis_truncated=False)))
    assert format_prior_metrics(metrics) == (
        "invalid_any=0.000 domain_only=0.000 unconfirmed=0.000 fallback=0.000 reason_fields={}"
    )
