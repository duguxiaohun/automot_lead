"""Inventory STOP evidence in an immutable old replay, without replay or training.

Only recorded questions expose control_evidence. Absence here is not proof that
a route has no STOP, and this inventory cannot authorize reuse under new code.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from .io import digest, file_sha, write_json


def inventory(index_path):
    from qwen3vl_local.sft_new_loop_phase4.production_storage import open_route
    index_path=Path(index_path).absolute()
    before=file_sha(index_path);index=json.loads(index_path.read_text())
    if index.get('sha256')!=digest({k:v for k,v in index.items() if k!='sha256'}):
        raise ValueError('production index content SHA mismatch')
    expected_header=dict(kind='header',policy=index['policy'],teacher=index['teacher'],
                         candidate_pool_sha256=index['candidate_pool_sha256'])
    totals=Counter();selected=[];examples=[];errors=[];seen=set()
    for route in index['routes']:
        key=(route['scenario'],route['route_id'])
        try:
            if key in seen:raise ValueError('duplicate route receipt')
            seen.add(key)
            path=index_path.parent/route['artifact'];resolved=path.resolve()
            if not resolved.is_relative_to(index_path.parent.resolve()):
                raise ValueError('artifact escapes production directory')
            if file_sha(path)!=route['sha256']:raise ValueError('route artifact SHA mismatch')
            pending=set();counts=Counter();route_examples=[];last=-1;frames=0
            with open_route(path) as stream:
                if json.loads(next(stream))!=expected_header:raise ValueError('route header differs from index')
                for line in stream:
                    row=json.loads(line);frames+=1
                    if any(row.get(k)!=route[k] for k in ('scenario','route_id','split','physical_group')):
                        raise ValueError('foreign frame in route')
                    if type(row.get('frame_id')) is not int or row['frame_id']<=last:
                        raise ValueError('frame ordering invalid')
                    last=row['frame_id']
                    # Count RGB4 once; RGB2 contains paired duplicates.
                    questions=[q for q in row['questions'] if q['rgb_mode']==4]
                    if not questions:continue
                    duty=questions[0].get('control_evidence',{})
                    if any(q.get('control_evidence',{})!=duty for q in questions):
                        raise ValueError('inconsistent same-frame control evidence')
                    if duty:counts['frames_with_stop_evidence']+=1
                    for ident,record in duty.items():
                        if type(record.get('satisfied')) is not bool:raise ValueError('invalid stop receipt')
                        if record['satisfied']:pending.discard(ident)
                        else:pending.add(ident)
                    vanished=pending-set(duty)
                    for q in questions:
                        if duty:
                            counts['questions_with_stop_evidence']+=1
                        if vanished and q['edge']=='proceed' and q['rule_target']=='YES':
                            counts['proceed_yes_after_pending_receipt_disappeared']+=1
                            if route['split']=='train' and len(route_examples)<8:
                                route_examples.append(dict(frame_id=row['frame_id'],question_id=q['question_id'],
                                    event=q['episode']['event'],missing_stop_ids=sorted(vanished)))
            if frames!=route['frames']:raise ValueError('route frame count differs')
            if path.resolve()!=resolved or file_sha(path)!=route['sha256']:
                raise ValueError('route changed during inventory')
            totals['routes_scanned']+=1;totals['frames_scanned']+=frames
            totals.update(counts)
            if counts:
                selected.append({k:route[k] for k in ('scenario','route_id','split','physical_group','artifact','sha256')}|
                                dict(counts=dict(counts)))
                if len(examples)<20:
                    examples.extend(dict(scenario=route['scenario'],route_id=route['route_id'],**e)
                                    for e in route_examples[:20-len(examples)])
        except (OSError,ValueError,KeyError,TypeError,EOFError,StopIteration,RuntimeError) as exc:
            errors.append(dict(scenario=key[0],route_id=key[1],error=str(exc)))
    if file_sha(index_path)!=before:raise ValueError('index changed during inventory')
    return dict(status='inventoried_not_relabelled' if not errors else 'incomplete',
                index_path=str(index_path),index_sha256=before,teacher=index['teacher']['version'],
                counts=dict(totals),routes_with_recorded_stop_evidence=selected,
                train_examples=examples,errors=errors,reuse_authorized=False,
                scope='Old recorded RGB4 questions only, not all source frames/controls. '
                      'Vanished pending receipts are investigation candidates, not proven wrong labels. '
                      'No new-rule compatibility, label-quality or training approval; val/test examples omitted.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve existing inventory')
    try:report=inventory(a.index)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
        report=dict(status='incomplete',errors=[str(exc)],reuse_authorized=False)
    write_json(a.output,report)
    print(json.dumps({k:report[k] for k in ('status','counts','errors') if k in report}))
    return 0 if report['status']=='inventoried_not_relabelled' else 2


if __name__=='__main__':
    raise SystemExit(main())
