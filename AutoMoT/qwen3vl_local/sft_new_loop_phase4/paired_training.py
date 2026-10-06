"""Explicit RGB2/RGB4 training intersection; preserve each complete evaluation set."""
import argparse
from collections import Counter
from pathlib import Path
from .identity import digest,write_json

POLICY='paired_causal_training_intersection_v1'


def semantic(row):
    return {k:row[k] for k in ('physical_group','scenario','route_id','episode','edge','slice','target','label_basis')} | {
        'frame_id':row['observation']['frame_id'],'speed_mps':row['observation']['speed_mps']}


def select(left,right):
    """Rows already passed each mode's independent evidence/admission checks."""
    def indexed(rows):
        values={r['id']:r for r in rows}
        if len(values)!=len(rows):raise ValueError('duplicate paired training row ID')
        if any(r['split']!='train' or r['target'] not in ('YES','NO') for r in rows):
            raise ValueError('paired training requires binary training rows')
        return values
    a,b=indexed(left),indexed(right);pairs=[];excluded=Counter()
    for ident in sorted(a.keys()|b.keys()):
        if ident not in a or ident not in b:
            excluded['missing_in_peer']+=1;continue
        x,y=a[ident],b[ident]
        if semantic(x)!=semantic(y):excluded['semantic_or_label_disagreement']+=1;continue
        source_maps=[]
        for row in (x,y):
            fs=row['observation']['history_frames']
            if len(set(fs))!=len(fs):raise ValueError('duplicate paired history frame')
            source_maps.append(dict(zip(fs,zip(row['images'],row['image_sha256'],row['image_rgb_sha256'],strict=True),strict=True)))
        small,big=sorted(source_maps,key=len)
        if len(small)!=2 or len(big)!=4 or not set(small)<=set(big):
            raise ValueError('paired RGB modes require nested causal histories')
        if any(small[f]!=big[f] for f in small):raise ValueError('paired training shared RGB differs')
        pairs.append(ident)
    if not pairs:raise ValueError('empty paired training intersection')
    report=dict(policy=POLICY,pairs=len(pairs),semantic_sha256=digest([[i,semantic(a[i])] for i in pairs]),
                exclusions=dict(excluded),scope='training intersection only; full val/test retained; not proof of label accuracy')
    return [a[i] for i in pairs],[b[i] for i in pairs],report


def load_view(data,manifest,peer_path):
    """Validate both immutable datasets, then select without rewriting either."""
    from .dataset import load_dataset,coverage_report
    from .admission import training_report
    peer,other=load_dataset(peer_path)
    if {manifest['rgb_mode'],other['rgb_mode']}!={2,4}:raise ValueError('paired training needs RGB2 and RGB4')
    for field in ('contract','seed','candidate_pool_sha256','production_index_sha256','teacher_registry_sha256','exposure_sha256'):
        if manifest.get(field)!=other.get(field):raise ValueError('paired dataset binding differs: '+field)
    selected,_,report=select(data['train'],peer['train'])
    view=dict(data,train=selected)
    admission=training_report(view,coverage_report(sum(view.values(),[])))
    if not admission['trainable']:raise ValueError('paired training intersection fails admission: '+str(admission['errors']))
    report.update(datasets={str(m['rgb_mode']):digest(m) for m in (manifest,other)},
                  selected_ids_sha256=digest([r['id'] for r in selected]),
                  original_train_counts={str(manifest['rgb_mode']):len(data['train']),str(other['rgb_mode']):len(peer['train'])},
                  training_admission=admission)
    return view,report


def main():
    from .dataset import load_dataset
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--paired-with',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('use a fresh pairing report')
    data,manifest=load_dataset(a.dataset)
    _,report=load_view(data,manifest,a.paired_with)
    write_json(a.output,report)
    print(f"paired training questions: {report['pairs']}; full validation/test retained")

if __name__=='__main__':main()
