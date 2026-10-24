"""Full CPU admission before GPU allocation; no weakened scarcity thresholds."""
from collections import Counter
import json
from pathlib import Path
from PIL import Image
from .audit import APP,sha,verify_snapshot
from .experiment import code,require_received_source,guard_candidates,guard_records
from .state import atomic
from .paths import image_path
from .training import regions
from ..phase3_single_v3 import dataset,admission,planning


def prepare(native_ready,root):
    require_received_source();verify_snapshot()
    ready=json.loads(Path(native_ready).read_text())
    if ready.get('status')!='ready':raise ValueError('native_single_not_ready: '+str(ready.get('error',ready.get('status'))))
    artifact=dataset.load(ready['dataset'],check_sources=True,expected_sha256=ready['dataset_sha256'])
    check=admission.require(admission.inspect(artifact))
    projection=planning.project(artifact['records'],per_event=1280,epochs=3,world_size=4,seed=20260904)
    if [p['optimizer_steps'] for p in projection['plans']]!=[3200]*3:raise ValueError('budget_mismatch')
    used={k for p in projection['plans'] for k in p['ordered_ids']}
    used.update(r['id'] for r in artifact['records'] if r['split'] in ('val','test'))
    buckets={}
    for r in artifact['records']:guard_candidates(buckets,r)
    guards=guard_records(buckets);used.update(r['id'] for r in guards)
    inputs=[];sizes=Counter();counts=Counter();source_code=code()
    data_root=Path(artifact['data_root']).resolve()
    for r in artifact['records']:
        counts[r['split']]+=1
        if r['id'] not in used:continue
        path=image_path(data_root,r['image'])
        h=sha(path)
        with Image.open(path) as im:
            im.load()
            if min(im.size)<=0:raise ValueError('invalid_RGB_size')
            sizes[str(im.size)]+=1
        if sha(path)!=h:raise ValueError('image_changed_while_decoding')
        inputs.append(dict(id=r['id'],path=r['image'],sha256=h))
        if len(inputs)%2000==0:print(f'[single cpu] RGB {len(inputs)}/{len(used)}',flush=True)
    if len(inputs)!=len(used) or len({r['id'] for r in artifact['records']})!=len(artifact['records']):raise ValueError('member_identity_mismatch')
    if code()!=source_code or sha(ready['dataset'])!=ready['dataset_sha256']:raise ValueError('cpu_inputs_changed')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    atomic(root/'admission.json',check);atomic(root/'plan.json',projection)
    atomic(root/'input_manifest.json',inputs)
    from .delivery import preflight_size
    size=preflight_size(artifact,projection,check)
    report=dict(counts=counts,guard_ids=[r['id'] for r in guards],guard_physical_groups=len({r['physical_group'] for r in guards}),
        decoded_inputs=len(inputs),image_sizes=sizes,package_budget=size,
        native_labels_unchanged=True,split_unchanged=True,sampling_unchanged=True,thresholds_lowered=False,
        source_dataset=ready['dataset'],source_dataset_sha256=ready['dataset_sha256'])
    atomic(root/'pool_report.json',report)
    result=dict(schema='single_refine_train_ready_v1',status='ready',dataset=ready['dataset'],dataset_sha256=ready['dataset_sha256'],
        code=source_code,contract=regions.contract('six'),admission=str(root/'admission.json'),admission_sha256=check['sha256'],
        plan=str(root/'plan.json'),plan_sha256=projection['sha256'],optimizer_steps=[3200]*3,
        inputs_sha256=sha(root/'input_manifest.json'),pool_report_sha256=sha(root/'pool_report.json'),native_ready=str(native_ready))
    atomic(root/'ready_1280.json',result);return result


def load_ready(root,*,check_images=False):
    require_received_source();verify_snapshot()
    root=Path(root);r=json.loads((root/'ready_1280.json').read_text())
    if r.get('schema')!='single_refine_train_ready_v1' or r.get('status')!='ready':raise ValueError('single_refine_not_ready')
    if r['code']!=code() or r['contract']!=regions.contract('six'):raise ValueError('candidate_source_changed_rerun_cpu')
    if sha(r['dataset'])!=r['dataset_sha256']:raise ValueError('dataset_changed')
    for name,key in [('input_manifest.json','inputs_sha256'),('pool_report.json','pool_report_sha256')]:
        if sha(root/name)!=r[key]:raise ValueError('preflight_report_changed')
    from .identity import digest
    for key in ('admission','plan'):
        report=json.loads(Path(r[key]).read_text())
        if report['sha256']!=r[key+'_sha256'] or digest({k:v for k,v in report.items() if k!='sha256'})!=r[key+'_sha256']:raise ValueError('preflight_'+key+'_changed')
    if check_images:
        # Project paths resolve independently on each machine during its CPU stage.
        native=dataset.load(r['dataset'],check_sources=False,expected_sha256=r['dataset_sha256'])
        base=Path(native['data_root'])
        for im in json.loads((root/'input_manifest.json').read_text()):
            if sha(image_path(base,im['path']))!=im['sha256']:raise ValueError('preflight_RGB_changed')
    return r
