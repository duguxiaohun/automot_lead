"""全量同题回归：重新严格解析，逐条保护原成功题和原正确答案位。只评估已给定集合。"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .paired_eval import load_cases, compare, summary, temporal_slices
from .prompt_candidate import spec_from_case, build_action_prompt as candidate_prompt
from .prompts import parse_action_output, build_action_prompt, SYSTEM_PROMPT, CHOICE_SYSTEM_PROMPT


def validate_prompt_messages(row):
    spec = spec_from_case(row)
    kwargs = dict(spec=spec, history_rgb_mode=row['history_rgb_mode'])
    prompt = row['action_user_prompt']
    if prompt not in (build_action_prompt(**kwargs), candidate_prompt(**kwargs)):
        raise ValueError('unregistered prompt: cannot score arbitrary or future-contaminated text')
    expected = [dict(role='system',content=CHOICE_SYSTEM_PROMPT if spec.action_output_mode=='choice' else SYSTEM_PROMPT),
        dict(role='user',content=[*[dict(type='image',source_path=p) for p in row['history_rgb_paths_used']],
                                 dict(type='text',text=prompt)])]
    if row.get('actual_chat_messages') != expected:
        raise ValueError('actual model messages disagree with declared causal input')


def scored(row):
    spec = spec_from_case(row)
    result = parse_action_output(row['raw_output'], spec=spec)
    parsed = {k: 'INVALID' if v is None else ('YES' if v else 'NO') for k,v in result.items()}
    valid = all(v is not None for v in result.values())
    ok = valid and parsed == row['gt']
    if type(row['all_ok']) is not bool or type(row['strict_format_valid']) is not bool:
        raise ValueError('score flags must be actual booleans')
    if row['all_ok'] != ok or row['strict_format_valid'] != valid or row['parsed'] != parsed:
        raise ValueError('cached score/parsed output disagrees with strict raw-output parsing')
    return ok, parsed


def risk_flags(row, risks):
    flags = []
    f = row['frame_id']
    for risk in risks:
        if [row['scenario'],row['route_id']] != risk['route']:
            continue
        lo, hi = risk['frames']
        phases = {'input': (f-3,f), 'longitudinal': (f,f+8),
                  'lateral': (f,f+13), 'audit': (f-3,f+13)}
        for phase,(a,b) in phases.items():
            if phase == 'lateral' and row['question_domain'] != 'FULL_MANEUVER':
                continue
            if a <= hi and b >= lo:
                flags.append(risk['id']+'/'+phase)
    return flags


def gate(left, right, risks=()):
    # compare 的全量身份检查先执行；随后更严格检查完整离线证据和模型输入。
    report = compare(left, right)
    regressions, bit_regressions, fixes = [], [], []
    protected = []
    groups = defaultdict(Counter)
    format_regressions = []
    for key,a in left.items():
        b = right[key]
        validate_prompt_messages(a)
        validate_prompt_messages(b)
        for field in ('prompt_spec','action_evidence','action_answers','mapping_evidence',
                      'question_domain','action_signature','event'):
            if field not in a or field not in b or a[field] != b[field]:
                raise ValueError(f'paired complete {field} mismatch: {key}')
        hashes = a.get('history_rgb_sha256')
        if not hashes or len(hashes) != len(a['history_rgb_paths_used']) or any(
            not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
            raise ValueError('complete RGB SHA256 fingerprints required')
        oa,pa = scored(a); ob,pb = scored(b)
        item = dict(case_index=a['case_index'], identity=list(key),
                    old=a['raw_output'], new=b['raw_output'], gt=a['gt'])
        if oa: protected.append(a['case_index'])
        if oa and not ob: regressions.append(item)
        if not oa and ob: fixes.append(item)
        lost = [k for k,v in a['gt'].items() if pa[k] == v and pb[k] != v]
        if lost: bit_regressions.append(dict(item, lost_answer_keys=lost))
        if a['strict_format_valid'] and not b['strict_format_valid']:
            format_regressions.append(a['case_index'])
        risk = risk_flags(a, risks)
        flags = ['all', 'context/'+a['context_id'], 'signature/'+a['action_signature'],
                 'risk_overlap' if risk else 'no_registered_overlap']
        flags += ['temporal/'+s for s in temporal_slices(a)]
        flags += ['risk/'+s for s in risk]
        cat = f'left_{int(oa)}_right_{int(ob)}'
        for flag in flags: groups[flag][cat] += 1
    clean = groups['no_registered_overlap']
    strict_improvement = bool(fixes) and clean['left_0_right_1'] > clean['left_1_right_0']
    no_regression = not (regressions or bit_regressions or format_regressions)
    report.update(schema='phase3_regression_gate_v1', regressions=regressions,
        answer_bit_regressions=bit_regressions, fixes=fixes,
        protected_success_case_indices=protected, protected_success_cases=len(protected),
        format_regressions=format_regressions, no_regression_on_observed_cases=no_regression,
        strict_improvement=strict_improvement,
        candidate_gate_passed=no_regression and strict_improvement,
        group_metrics={k:summary(v) for k,v in sorted(groups.items())},
        risks=list(risks), automatic_promotion=False,
        interpretation='Exposed development replay only; no unseen-case guarantee. Gate requires improvement outside registered risk overlaps, zero lost successful cases and zero lost correct answer bits. No automatic promotion or label changes.')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left',required=True,type=Path);p.add_argument('--right',required=True,type=Path)
    p.add_argument('--risks',type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    a,ha=load_cases(args.left);b,hb=load_cases(args.right)
    risks=json.loads(args.risks.read_text())['risks'] if args.risks else []
    result=gate(a,b,risks)
    result['source_sha256']={'left':ha,'right':hb}
    if args.risks: result['risk_sha256']=hashlib.sha256(args.risks.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['paired_cases','protected_success_cases','no_regression_on_observed_cases','strict_improvement','candidate_gate_passed']}))
    return 0 if result['candidate_gate_passed'] else 2

if __name__=='__main__':
    raise SystemExit(main())
