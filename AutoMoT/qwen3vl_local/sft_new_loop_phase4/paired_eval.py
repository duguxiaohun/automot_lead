"""Strict paired comparison on identical causal inputs and reference labels."""
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path
from .identity import digest,file_sha,write_json

POLICY='phase4_paired_prediction_v1'
FIELDS=('scenario','route_id','physical_group','split','episode','edge','slice','target','label_basis',
        'observation','images','image_sha256','image_rgb_sha256','model_input_sha256','prompt_sha256')


def case_identity(row):
    body={k:row[k] for k in FIELDS}
    body['reference_kind']=row.get('reference_kind','reviewed_rgb')
    return dict(policy=POLICY,fields=body,sha256=digest(body))


def validate(row):
    identity=row.get('paired_identity',{})
    body=identity.get('fields',{})
    if (identity.get('policy')!=POLICY or set(body)!=set(FIELDS)|{'reference_kind'}
            or identity.get('sha256')!=digest(body)):raise ValueError('missing/invalid paired input identity')
    if any(row.get(k)!=body[k] for k in ('target','edge','slice','reference_kind')) or row.get('event')!=body['episode']['event']:
        raise ValueError('paired output/reference mismatch')
    if row['target'] not in ('YES','NO') or row.get('prediction') not in ('YES','NO','UNKNOWN','MALFORMED'):
        raise ValueError('invalid paired target/prediction')
    n=len(body['observation']['history_frames'])
    if n not in (2,4) or any(len(body[k])!=n for k in ('images','image_sha256','image_rgb_sha256')):
        raise ValueError('invalid paired RGB history')
    for value in [body['model_input_sha256'],body['prompt_sha256'],*body['image_sha256'],*body['image_rgb_sha256']]:
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('invalid paired fingerprint')


def load(path):
    path=Path(path);path=path/'cases.jsonl' if path.is_dir() else path
    rows={}
    for line in path.open():
        r=json.loads(line);validate(r)
        if r['id'] in rows:raise ValueError('duplicate paired case')
        rows[r['id']]=r
    return rows,dict(path=str(path),sha256=file_sha(path))


def summary(counts):
    n=sum(counts.values());left=counts['left_1_right_0']+counts['left_1_right_1'];right=counts['left_0_right_1']+counts['left_1_right_1']
    return dict(count=n,transitions=dict(counts),left_score=left/n if n else None,right_score=right/n if n else None,delta=(right-left)/n if n else None)


def compare(left,right):
    if not left or left.keys()!=right.keys():raise ValueError('paired case sets differ or empty; intersection-only comparison forbidden')
    strata=defaultdict(Counter)
    for key,a in left.items():
        b=right[key];validate(a);validate(b)
        if a['paired_identity']!=b['paired_identity']:raise ValueError('paired input or truth changed')
        category=f"left_{int(a['prediction']==a['target'])}_right_{int(b['prediction']==b['target'])}"
        kind='teacher_consistency' if a['reference_kind']=='rule_teacher' else 'reviewed_accuracy'
        strata[(kind,'all')][category]+=1
        strata[(kind,a['event'])][category]+=1
        strata[(kind,a['event']+'/'+a['edge']+'/'+a['slice'])][category]+=1
    return dict(policy=POLICY,paired_cases=len(left),excluded_cases=0,
        reviewed_accuracy={s:summary(c) for (k,s),c in strata.items() if k=='reviewed_accuracy'},
        teacher_consistency={s:summary(c) for (k,s),c in strata.items() if k=='teacher_consistency'},
        interpretation='identical inputs/references only; UNKNOWN/MALFORMED count as wrong; no claim of equal training budget or complete closed-loop coverage')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    left,ls=load(a.left);right,rs=load(a.right);result=compare(left,right);result['sources']=[ls,rs];write_json(a.output,result);print(json.dumps(result))

if __name__=='__main__':main()
