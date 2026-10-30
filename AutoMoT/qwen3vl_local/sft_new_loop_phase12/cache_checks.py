"""Output parity is a gate; floating-point prompt logits remain diagnostic evidence."""
import math
import torch

PROTOCOL = 'exact_generated_tokens_and_stop_v1_20261030'


def logits(actual, expected):
    if actual.shape != expected.shape or actual.ndim != 3 or actual.numel() == 0:
        raise ValueError('Cache/full logit shape mismatch or empty output')
    atol, rtol = (.08, .03) if actual.dtype == torch.bfloat16 else (2e-5, 2e-4)
    count = bad = nonfinite = 0
    maximum = total = 0.
    for start in range(0, actual.shape[1], 16):
        a = actual[:, start:start+16].float(); b = expected[:, start:start+16].float()
        finite = torch.isfinite(a) & torch.isfinite(b)
        delta = (a-b).abs()[finite]
        count += a.numel(); nonfinite += int((~finite).sum())
        bad += int((~torch.isclose(a, b, atol=atol, rtol=rtol)).sum())
        if delta.numel():
            maximum = max(maximum, float(delta.max())); total += float(delta.sum())
    report = dict(elements=count, mismatches=bad, nonfinite=nonfinite,
        max_abs_difference=maximum, mean_abs_difference=total/max(1, count-nonfinite),
        atol=atol, rtol=rtol, all_close=bad == 0 and nonfinite == 0,
        next_token_matches=bool(torch.equal(actual[:, -1].argmax(-1), expected[:, -1].argmax(-1))))
    verify_logits(report)
    return report


def verify_logits(report):
    if not isinstance(report, dict):
        raise ValueError('Missing structured logit diagnostic')
    for name in ('elements', 'mismatches', 'nonfinite'):
        if type(report.get(name)) is not int or report[name] < 0:
            raise ValueError('Invalid logit diagnostic counts')
    if not report['elements'] or report['mismatches'] > report['elements']:
        raise ValueError('Invalid logit diagnostic denominator')
    if report['nonfinite'] or report.get('next_token_matches') is not True:
        raise ValueError('Nonfinite cache logits or changed next-token decision')
    if report.get('all_close') is not (report['mismatches'] == 0):
        raise ValueError('Logit diagnostic incorrectly marked passed')
    if (report.get('atol'), report.get('rtol')) not in ((.08, .03), (2e-5, 2e-4)):
        raise ValueError('Logit diagnostic tolerance changed')
    for name in ('max_abs_difference', 'mean_abs_difference'):
        value = report.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid logit diagnostic magnitude')


def verify_output_evidence(performance, branches, *, require_logits=False):
    from .cache import assert_answers
    if performance.get('cache_acceptance_protocol') != PROTOCOL:
        raise ValueError('Missing/current cache acceptance protocol')
    if performance.get('full_answer_equivalence_checked') is not True:
        raise ValueError('Missing full-answer cache parity')
    assert_answers({'branches': branches}, {'branches': performance['independent_branches']})
    checked = performance.get('full_forward_equivalence_checked')
    if type(checked) is not bool or (require_logits and not checked):
        raise ValueError('Missing prompt-logit diagnostic')
    reports = [b.get('full_comparison') for b in branches]
    if checked:
        for report in reports: verify_logits(report)
        passed = all(r['all_close'] for r in reports)
    else:
        if any(reports): raise ValueError('Unexpected unchecked logit evidence')
        passed = None
    if performance.get('full_forward_equivalence_passed', 'missing') is not passed:
        raise ValueError('Logit diagnostic outcome misreported')
