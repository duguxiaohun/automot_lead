"""把真实逐帧观察笔记绑定到原始RGB证据；渲染与已审阅严格分离。"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='qwen3vl_local.sft_new_loop_phase4'

import argparse
from collections import Counter
import json
from pathlib import Path
from .identity import file_sha, write_json


def publish(evidence_paths, notes_path, output):
    evidence = [r for p in evidence_paths for r in json.loads(Path(p).read_text())]
    notes = {f'p4_{i:03d}':(note,tag) for i,note,tag in json.loads(Path(notes_path).read_text())}
    if len({r['review_id'] for r in evidence}) != len(evidence):
        raise ValueError('duplicate review ID')
    if set(notes) != {r['review_id'] for r in evidence}:
        raise ValueError('missing review notes or evidence')
    rows = []
    for r in evidence:
        note,tag = notes[r['review_id']]
        rows.append({k:r[k] for k in ('review_id','scenario','town','route_id','context_id','physical_group','frames')} |
                    dict(observation=note,tag=tag,status='visually_reviewed',
                         review_method='chronological_contact_sheets_every_frame',
                         panel_sha256=[file_sha(p) for p in r['panels']]))
    unique = {f['rgb']:f['rgb_sha256'] for r in rows for f in r['frames']}
    report = dict(windows=len(rows),frame_presentations=sum(len(r['frames']) for r in rows),
                  unique_rgb_paths=len(unique),unique_rgb_hashes=len(set(unique.values())),
                  physical_groups=len({r['physical_group'] for r in rows}),
                  towns=sorted({r['town'] for r in rows}),
                  contexts=dict(Counter(r['context_id'] for r in rows)),
                  tags=dict(Counter(r['tag'] for r in rows)),
                  scope='development review; not all routes; not frame-level permission certification',
                  train_only_groups=sorted({r['physical_group'] for r in rows}),reviews=rows)
    write_json(output,report)
    return {k:v for k,v in report.items() if k not in ('reviews','train_only_groups')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,nargs='+',required=True)
    p.add_argument('--notes',type=Path,default=Path(__file__).with_name('review_notes.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(publish(a.evidence,a.notes,a.output),ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
