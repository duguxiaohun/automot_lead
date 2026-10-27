"""Protect both the same-run and the recorded old-correct answers."""
from qwen3vl_local.audit_joint.phase1_visual_refine.regression import compare as native_compare

def compare(frozen, baseline, candidate, reference):
    paired = native_compare(frozen, baseline, candidate)
    recorded = native_compare(frozen, reference, candidate)
    drift = native_compare(frozen, reference, baseline)
    blocked = any(r['status']=='review_required' for r in (paired, recorded))
    return dict(status='review_required' if blocked else 'no_observed_regression',
                promotion='not_automatic', total=paired['total'], joint_correct=paired['joint_correct'],
                same_run=paired, against_recorded_baseline=recorded, baseline_drift=drift,
                scope='Full membership, original frozen GT; reviewed-label diagnostics cannot waive losses')
