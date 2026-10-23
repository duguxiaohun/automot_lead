"""Content-addressed local study preparation; no timestamp/latest inference."""
from pathlib import Path
import json
from . import dataset, planning, regions, admission
from .. import versioned_data as cache
from ..geometry_v1 import dataset as native

READY_SCHEMA = 'geometry_abc_v2_ready_v1'


def prepare(geometry, maps, root, *, per_event=1280, world_size=4):
    geometry=Path(geometry).resolve(strict=True); maps=Path(maps).resolve(strict=True)
    root=Path(root).resolve()
    parent=native.load(geometry,check_sources=False)
    protected=(geometry.parent,maps,Path(parent['data_root']).resolve())
    towns=tuple(parent['map_sha256'])
    del parent
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError('output_overlaps_input')
    def dependencies():
        return dict(parent=cache.bind(geometry),parent_receipt=cache.bind(geometry.with_name('receipt.json')),
            maps=cache.bindings([maps/(town+'.xodr') for town in towns]),
            sources=dataset.code_identity(),contracts={v:regions.contract(v) for v in regions.VARIANTS})
    with cache.lock(root/'.prepare.lock'):
        ready_path=root/f'ready_{per_event}.json'
        # Invalidate the mutable readiness pointer first; immutable old datasets,
        # plans, admissions and checkpoints remain available for audit.
        state=dict(schema=READY_SCHEMA,status='preparing',per_event=per_event,world_size=world_size)
        cache.write(ready_path,state)
        try:
            result=cache.ensure(root/'generation','geometry_abc_v2','geometry_only_v1',dependencies,
                lambda output:dataset.build(geometry,maps,output),lambda output:dataset.load(output/'dataset.json'),
                protected=protected)
            path=Path(result['directory'])/'dataset.json'
            # cache.ensure has just executed the full native source validator.
            # Rebind the derived bytes without a redundant full raw-file scan.
            artifact=dataset.load(path,check_sources=False)
            check=admission.inspect(artifact)
            admissions=root/'admissions';admissions.mkdir(parents=True,exist_ok=True)
            check_path=admissions/(check['sha256']+'.json')
            immutable_report(check_path,check)
            state.update(dataset=str(path),dataset_sha256=dataset.sources.file_sha(path),
                generation=result,admission=str(check_path),admission_sha256=check['sha256'])
            admission.require(check)
            report=planning.project(artifact['records'],per_event=per_event,epochs=3,world_size=world_size,seed=20260904)
            plans=root/'plans';plans.mkdir(parents=True,exist_ok=True)
            plan_path=plans/(report['sha256']+'.json');immutable_report(plan_path,report)
            state.update(status='ready',plan=str(plan_path),plan_sha256=report['sha256'],
                optimizer_steps=[p['optimizer_steps'] for p in report['plans']])
            cache.write(ready_path,state)
            return state
        except Exception as exc:
            state.update(status='rejected',error=f'{type(exc).__name__}: {exc}')
            cache.write(ready_path,state)
            raise


def immutable_report(path, report):
    if path.exists():
        if json.loads(path.read_text())!=report:raise ValueError('existing_study_report_changed')
    else:cache.write(path,report)


def load_ready(path, *, per_event, world_size):
    """Validate current source/data, fixed validation and full budget before launch."""
    ready=json.loads(Path(path).read_text())
    if (ready.get('schema')!=READY_SCHEMA or ready.get('status')!='ready'
            or ready.get('per_event')!=per_event or ready.get('world_size')!=world_size):
        raise ValueError('study_not_ready_for_requested_budget')
    # The worker's rank 0 performs the full raw-source verification before any
    # model load. Here bind JSON/current code without duplicating that full scan.
    artifact=dataset.load(ready['dataset'],check_sources=False,expected_sha256=ready['dataset_sha256'])
    check=admission.require(admission.inspect(artifact))
    if (check['sha256']!=ready['admission_sha256']
            or json.loads(Path(ready['admission']).read_text())!=check):
        raise ValueError('prepared_admission_changed')
    plan=planning.project(artifact['records'],per_event=per_event,epochs=3,world_size=world_size,seed=20260904)
    if plan['sha256']!=ready['plan_sha256'] or json.loads(Path(ready['plan']).read_text())!=plan:
        raise ValueError('prepared_plan_changed')
    return ready
