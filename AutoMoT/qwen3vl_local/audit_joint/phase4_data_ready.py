"""Resolve local completed Phase4 assets, otherwise run the native CPU stages.

Teacher identity remains global. There is no implicit old-to-new relabel patch.
The original native source check, replay recovery and admission gates remain.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

from . import versioned_data as cache

APP=Path(__file__).resolve().parents[2]


def assets(root):
    root=Path(root)
    state=json.loads((root/'build_result.json').read_text())
    explicit=state.get('existing_assets',{})
    paths={key:Path(explicit[key]['path']) if key in explicit else root/name for key,name in (
        ('native','data4/manifest.json'),('references','references4/references.json'),('view','view/student_view.json'))}
    for key,value in explicit.items():
        if key in paths and cache.bind(paths[key])['sha256']!=value['sha256']:
            raise ValueError('Phase4 admitted asset metadata changed: '+key)
    return state,paths


def validate(root,data):
    from .rule_baseline import ADMITTED,check
    state,paths=assets(root)
    if state.get('status') not in ADMITTED:
        raise ValueError('Phase4 CPU preparation not admitted')
    budget=json.loads((Path(root)/'first_epoch_budget.json').read_text())
    report=check(paths['view'].parent,paths['references'].parent,budget=budget,space_path=Path(root),
                 coverage_policy=state.get('coverage_policy','existing_data_development'),data_root=data,batch_size=2)
    if report['status'] not in ADMITTED:
        raise ValueError('Phase4 native admission no longer passes: '+str(report['errors']))
    return dict(prepared=str(Path(root)),native=str(paths['native'].parent),
                view=str(paths['view'].parent),references=str(paths['references'].parent),
                budget=str(Path(root)/'first_epoch_budget.json'),training_performed=False)


def ensure(data,collection,model,checkpoints,*,existing=None,workers=2):
    from .rule_baseline import ADMITTED
    from .phase4_resume import plan,RELEASE
    from .server_readiness import source_check
    data,collection,model,checkpoints=map(lambda p:Path(p).absolute(),(data,collection,model,checkpoints))
    if workers<1:raise ValueError('workers must be positive')
    if collection.resolve()!=(APP/'keyframe_filter/collection_output').resolve():
        raise ValueError('Native Phase4 preparation uses the project collection; custom collection requires an explicit native recipe')
    # Current release is separate from historical manifests; all child entries
    # select the same release, including missing-view and missing-data branches.
    source=source_check(APP,RELEASE)
    if source['status']!='matched':
        changed=[r['path'] for r in source.get('files',[]) if r['status']!='matched']
        raise ValueError('Published Phase4 sources differ ('+RELEASE.name+'): '+str(changed))
    candidates=[Path(existing)] if existing else [p.parent for p in cache.discover(checkpoints,'build_result.json')]
    compatible={}
    for candidate in candidates:
        path=candidate/'build_result.json'
        if not path.is_file():continue
        state=json.loads(path.read_text())
        if state.get('status') not in ADMITTED:continue
        recorded=state.get('server_context',{}).get('paths',{}).get('data_root',{}).get('resolved')
        if not existing and recorded and recorded!=str(data.resolve()):continue
        # Skip demonstrably other teacher/schema versions, but not corrupted
        # artifacts that claim the same native contract.
        from qwen3vl_local.sft_new_loop_phase4.identity import contract as native_contract
        _,paths=assets(candidate)
        manifest=json.loads(paths['native'].read_text())
        if manifest.get('contract')!=native_contract():continue
        result=validate(candidate,data)
        compatible[candidate]=result
    selected=cache.unique_compatible(compatible,lambda p:{k:cache.bind(Path(compatible[p][k])/name)['sha256']
        for k,name in [('native','manifest.json'),('view','student_view.json'),('references','references.json')]})
    if selected:
        return dict(compatible[selected],action='reuse_admitted',generation_performed=False)
    if not existing:
        from qwen3vl_local.sft_new_loop_phase4 import dataset as native
        from qwen3vl_local.sft_new_loop_phase4.identity import contract as native_contract
        partial=[]
        for path in cache.discover(checkpoints,'manifest.json'):
            if path.parent.name!='data4' or not (path.parent.parent/'references4/references.json').is_file():continue
            manifest=json.loads(path.read_text())
            if manifest.get('contract')!=native_contract():continue
            native.load_dataset(path.parent)
            partial.append(path.parent.parent)
        if len(partial)>1:
            raise ValueError('Multiple compatible partial Phase4 builds; choose --existing: '+str(partial))
        if partial:existing=partial[0]
    # Explicit partial work goes through the native earliest-missing-stage
    # router. It never replays a verified complete historical replay.
    inputs=list(collection.glob('*_result.json'))
    if not inputs and not existing:raise ValueError('No admitted Phase4 data and no local annotations to generate it')
    identity=dict(release=cache.bind(RELEASE),data_root=str(data.resolve(strict=True)),
                  collection=cache.bindings(inputs),model_config=cache.bind(model/'config.json'))
    if existing:
        prepared=Path(existing).absolute()
        identity['prepared']=str(prepared)
        identity['prepared_metadata']=cache.bindings(p for name in (
            'data4/manifest.json','references4/references.json','view/student_view.json','inputs/candidates.json')
            if (p:=prepared/name).is_file())
    key=cache.digest(identity)
    output=checkpoints/'phase4_prepared_versions'/key
    output.parent.mkdir(parents=True,exist_ok=True)
    with cache.lock(checkpoints/'generation_cache/.locks'/('phase4-'+key+'.lock')):
        if output.exists():
            result=validate(output,data)
            return dict(result,action='reuse_admitted',generation_performed=False)
        env=dict(os.environ,DATA_ROOT=str(data),MODEL_DIR=str(model),OUTPUT_DIR=str(output),
                 PYTHON=sys.executable,REPLAY_WORKERS=str(workers))
        if existing:
            selected=plan(prepared,output,data,model)
            command=selected['command'];env.update(selected['environment'])
        else:
            command=['bash',str(APP/'qwen3vl_local/audit_joint/server_phase4_existing.sh')]
        subprocess.run(command,cwd=APP,env=env,check=True)
        return dict(validate(output,data),action='native_prepare_or_resume',generation_performed=True)
