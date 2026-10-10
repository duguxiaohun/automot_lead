"""Apply evidence-bound quarantine and inventory local full-frame replay outputs.

Kept rows are still development candidates: no split certification, manual
precedence, observable-conflict deduplication or independent approval is implied.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from qwen3vl_local.sft_new_loop_phase4.identity import file_sha, write_json
from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS, CONTEXT_TO_EVENT
from . import label_quarantine as quarantine


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def emit(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def verify_replay(path):
    path = Path(path)
    summary = json.loads((path / 'summary.json').read_text())
    if summary.get('status') != 'complete':
        raise ValueError('incomplete replay')
    receipt = json.loads((path / 'receipt.json').read_text())['files']
    required = {'request.json', 'source.json', 'phase3_native_candidates.jsonl',
                'phase3_timeline.jsonl', 'phase4_questions.jsonl', 'joint_timeline.jsonl', 'summary.json'}
    if not required <= set(receipt):
        raise ValueError('replay receipt lacks required files')
    for name, proof in receipt.items():
        p = path / name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink():
            raise ValueError('invalid replay receipt path')
        if p.stat().st_size != proof['bytes'] or file_sha(p) != proof['sha256']:
            raise ValueError('replay artifact drift: ' + name)
    request = json.loads((path / 'request.json').read_text())
    expected = {(r['scenario'], r['route_id'], f) for r in request['routes'] for f in r['rgb_frames']}
    if len(expected) != sum(len(r['rgb_frames']) for r in request['routes']):
        raise ValueError('duplicate requested frames')
    for name in ('phase3_timeline.jsonl', 'joint_timeline.jsonl'):
        seen = set()
        for r in rows(path / name):
            key = r['scenario'], r['route_id'], r['frame_id']
            if key not in expected or key in seen:
                raise ValueError('unexpected/duplicate timeline frame')
            seen.add(key)
        if seen != expected:
            raise ValueError('timeline omits requested RGB frames')
    if summary['total_frames'] != len(expected):
        raise ValueError('replay total disagrees with timeline')
    return request


def presence_status(row):
    # No generated question is not a detector's explicit NO.
    if not row['phase4']:
        return 'phase4_abstention_not_event_absence'
    if row.get('presence_status') == 'phase3_source_unknown':
        return 'phase3_source_unknown'
    if set(row['phase3_events']) == set(row['phase4_events']):
        return 'same_event_sets_not_instance_match'
    return 'event_presence_disagreement_candidate'


def build(replay, output):
    replay, output = Path(replay), Path(output)
    request = verify_replay(replay)
    risks = quarantine.load(request['data_root'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'progress.json', dict(status='in_progress'))
    try:
        events = {e:dict(phase3_candidates=0, phase3_kept=0, phase4_raw=Counter(),
                        phase4_gate=Counter(), phase4_kept=Counter()) for e in EVENTS}
        counts = Counter(); reasons = Counter(); support_rows = []
        with (output / 'phase3_candidates.jsonl').open('x') as p3, \
                (output / 'phase4_gate_candidates.jsonl').open('x') as p4, \
                (output / 'quarantine.jsonl').open('x') as excluded:
            for r in rows(replay / 'phase3_native_candidates.jsonl'):
                e = CONTEXT_TO_EVENT[r['context_id']]; events[e]['phase3_candidates'] += 1
                ids = quarantine.phase3_risks(r, risks)
                if ids:
                    counts['phase3_quarantined'] += 1
                    emit(excluded, dict(phase='phase3', risk_ids=ids, original=r))
                else:
                    events[e]['phase3_kept'] += 1
                    emit(p3, dict(r, diagnostic_only=True, label_approval=False))
            for r in rows(replay / 'phase4_questions.jsonl'):
                q = r['question']; e = q['episode']['event']; mode = str(q['rgb_mode'])
                events[e]['phase4_raw'][mode + '/' + q['rule_target']] += 1
                events[e]['phase4_gate'][mode + '/' + r['admission_target']] += 1
                reasons[r['admission_reason']] += 1
                ids = quarantine.phase4_risks(q, risks)
                if ids:
                    counts['phase4_quarantined'] += 1
                    emit(excluded, dict(phase='phase4', risk_ids=ids, original=r))
                elif r['admission_target'] in ('YES', 'NO'):
                    events[e]['phase4_kept'][mode + '/' + r['admission_target']] += 1
                    emit(p4, dict(r, diagnostic_only=True, label_approval=False))
                    # Support table uses RGB4 only: two views of a question are
                    # not two independent examples. Manual support is separate.
                    if q['rgb_mode'] == 4:
                        support_rows.append(dict(q, target=r['admission_target'],
                            observation=dict(speed_mps=q['ego_speed'])))
        from qwen3vl_local.sft_new_loop_phase4.branch_support import report
        write_json(output / 'phase4_branch_support.json', report({'train': support_rows}))
        presence = Counter(); disposition = Counter(); review = []
        for r in rows(replay / 'joint_timeline.jsonl'):
            category = presence_status(r); presence[category] += 1
            disposition.update(r['reasons'])
            if category == 'event_presence_disagreement_candidate':
                review.append(dict(scenario=r['scenario'], route_id=r['route_id'], frame_id=r['frame_id'],
                    phase3_events=r['phase3_events'], phase4_events=r['phase4_events'],
                    status='requires_participant_and_instance_match', label_error=None))
        with (output / 'event_presence_review.jsonl').open('x') as stream:
            for r in review: emit(stream, r)
        result = dict(status='complete', source_replay=str(replay.resolve()),
            request_sha256=file_sha(replay / 'request.json'), replay_receipt_sha256=file_sha(replay / 'receipt.json'),
            quarantine_ledger_sha256=file_sha(quarantine.LEDGER),
            implementation_sha256={p.name:file_sha(p) for p in (Path(__file__), Path(quarantine.__file__))},
            events=events, counts=counts, gate_reasons=reasons, frame_reasons=disposition, presence=presence,
            approved_for_training=False, strict_teacher_approval=False,
            scope='selected training routes; before final compiler; no full-pool coverage claim; no target rewritten')
        write_json(output / 'summary.json', result)
        write_json(output / 'receipt.json', dict(files={p.name:dict(bytes=p.stat().st_size, sha256=file_sha(p))
            for p in output.iterdir() if p.is_file() and p.name != 'progress.json'}))
        write_json(output / 'progress.json', dict(status='complete'))
        return result
    except BaseException as error:
        write_json(output / 'progress.json', dict(status='failed', error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--replay', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    result = build(args.replay, args.output_dir)
    print(json.dumps(dict(status=result['status'], counts=result['counts']), ensure_ascii=False))


if __name__ == '__main__':
    main()
