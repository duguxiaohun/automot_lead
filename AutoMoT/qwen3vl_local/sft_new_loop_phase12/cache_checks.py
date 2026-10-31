"""Versioned acceptance: retained text state vs recomputation; full path is a control."""
import math
import copy
import torch

PROTOCOL = 'exact_generated_tokens_and_stop_v1_20261030'
TEXT_PROTOCOL = 'text_reuse_same_partition_v2_20261031'

def protocol():
    from qwen3vl_local.qwen35 import text_prefix
    return TEXT_PROTOCOL if text_prefix.cache_enabled() else PROTOCOL


def first_difference(cached, independent):
    """Describe the first differing decision, including EOS and token limits."""
    if len(cached['branches'])!=2 or len(independent['branches'])!=2:
        raise ValueError('Missing paired output')
    for arm, (a, b) in enumerate(zip(cached['branches'], independent['branches']), 1):
        for key in ('token_ids', 'stop_token_id', 'reached_token_limit', 'text'):
            if a[key] == b[key]:continue
            x, y = a['token_ids'], b['token_ids']
            index = next((i for i in range(min(len(x),len(y))) if x[i] != y[i]), min(len(x),len(y)))
            def decision(answer, ids):
                if index < len(ids):return dict(kind='token', token_id=ids[index])
                return dict(kind='limit' if answer['reached_token_limit'] else 'stop', token_id=answer['stop_token_id'])
            return dict(arm=arm, field=key, token_index=index, common_token_ids=x[:index],
                        cached=decision(a,x), independent=decision(b,y))
    return None


class CacheParityError(ValueError):
    """Strict failure carrying complete JSON evidence rather than truncated logs."""
    def __init__(self, cached, independent):
        difference=first_difference(cached,independent)
        self.evidence=copy.deepcopy(dict(first_difference=difference,cached=cached,independent=independent))
        super().__init__(f"Shared cache changes generated answer arm={difference['arm']} "
                         f"field={difference['field']} token_index={difference['token_index']}; "
                         f"cached={difference['cached']} independent={difference['independent']}")


def logits(actual, expected, *, diagnostic_only=False):
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
    verify_logits(report, diagnostic_only=diagnostic_only)
    return report


def verify_logits(report, *, diagnostic_only=False):
    if not isinstance(report, dict):
        raise ValueError('Missing structured logit diagnostic')
    for name in ('elements', 'mismatches', 'nonfinite'):
        if type(report.get(name)) is not int or report[name] < 0:
            raise ValueError('Invalid logit diagnostic counts')
    if not report['elements'] or report['mismatches'] > report['elements']:
        raise ValueError('Invalid logit diagnostic denominator')
    if (report['nonfinite'] or type(report.get('next_token_matches')) is not bool
            or (not diagnostic_only and not report['next_token_matches'])):
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
    text = protocol() == TEXT_PROTOCOL
    if performance.get('cache_acceptance_protocol') != protocol():
        raise ValueError('Missing/current cache acceptance protocol')
    if performance.get('full_answer_equivalence_checked') is not True:
        raise ValueError('Missing full-answer cache parity')
    actual={'branches': branches}; full={'branches': performance['independent_branches']}
    if text:
        assert_answers(actual,actual); assert_answers(full,full)
        difference=first_difference(actual,full)
        if (performance.get('full_answer_equivalence_passed') is not (difference is None)
                or performance.get('full_answer_difference','missing') != difference):
            raise ValueError('Full-output diagnostic outcome misreported')
        if performance.get('same_partition_equivalence_checked') is not True:
            raise ValueError('Missing same-partition output acceptance')
        assert_answers(actual,{'branches':performance['recomputed_split_branches']})
    else:
        assert_answers(actual,full)
    checked = performance.get('full_forward_equivalence_checked')
    if type(checked) is not bool or (require_logits and not checked):
        raise ValueError('Missing prompt-logit diagnostic')
    reports = [b.get('full_comparison') for b in branches]
    if checked:
        for report in reports: verify_logits(report, diagnostic_only=text)
        passed = all(r['all_close'] for r in reports)
    else:
        if any(reports): raise ValueError('Unexpected unchecked logit evidence')
        passed = None
    if performance.get('full_forward_equivalence_passed', 'missing') is not passed:
        raise ValueError('Logit diagnostic outcome misreported')

    if text and checked:
        from .cache_structure import verify_reuse
        structural=performance.get('text_reuse_structure',[])
        if len(structural)!=2:raise ValueError('Missing two-branch text state acceptance')
        for report in structural:verify_reuse(report)
