"""Versioned, machine-local data preparation. Never starts GPU training.

Built-in Phase1/2, Phase3 base/single-frame, geometry and Phase4 adapters. Other native
generation stages use an explicit dependency recipe with a native validator.
"""
import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

from . import versioned_data as cache

APP=Path(__file__).resolve().parents[2]


@contextmanager
def argv(values):
    old=sys.argv
    try:
        sys.argv=[old[0],*values]
        yield
    finally:sys.argv=old


def phase12(phase, data, collection, checkpoints, *, native_args=(), existing=None):
    """Native global sampling stays global; no unproved per-route resampling."""
    from qwen3vl_local.sft_new_loop_phase1.local_assets import inspect_index
    builder=importlib.import_module(f'qwen3vl_local.sft_new_loop_phase{phase}.build_dataset')
    with argv(['--data-root',str(data),'--collection-dir',str(collection),*native_args]):
        args=builder.parse_args()
    def validate(output):return inspect_index(output/'frame_index.jsonl',phase)
    if existing is not None:
        path=Path(existing).resolve(strict=True)
        result=inspect_index(path,phase)
        return dict(action='reuse_explicit_legacy',index=str(path),native_validation=result,
                    original_producer_source_identity='unknown unless stored by native manifest',
                    generation_performed=False,training_performed=False)
    # Explicit relocation/config flags cannot silently change protected roots.
    data=Path(args.data_root).resolve(strict=True);collection=Path(args.collection_dir).resolve(strict=True)
    # Each new version binds declared options and complete producer code/input
    # files. RGB is only tested for existence by these builders, not decoded.
    def contract():
        source_dirs=[APP/'qwen3vl_local'/n for n in (
            'sft_new_loop_phase1','sft_new_loop_phase2','sft_loop_phase1','sft_loop_phase2_augment')]+[APP/'lead_video_tools']
        code=[p for directory in source_dirs for p in directory.glob('*.py')]
        paths=list(collection.glob('*_result.json'))
        if not paths:raise ValueError('Missing formal collection annotations')
        for key in ('answer_table','coverage_manifest','highway_ue3_rgb_decisions'):
            value=getattr(args,key,None)
            if value and Path(value).is_file():paths.append(Path(value))
        for value in getattr(args,'manual_note_paths','').split(','):
            if value and Path(value).is_file():paths.append(Path(value))
        review=Path(args.review_root)
        if review.exists():paths.extend(review.rglob('*.json'));paths.extend(review.rglob('*.jsonl'))
        members={}
        for scenario in sorted(data.iterdir()):
            if not scenario.is_dir():continue
            for run in sorted(scenario.iterdir()):
                if not run.is_dir():continue
                # _rgb_path supports jpeg/png as well as padded frame filenames.
                rgb=run/'rgb'
                members[str(run.relative_to(data))]=sorted(p.name for p in rgb.iterdir() if p.is_file()) if rgb.is_dir() else None
        options={k:v for k,v in vars(args).items() if k not in ('output_dir','progress_every_routes')}
        return dict(native_dataset_name=builder.DATASET_NAME,options=options,source_files=cache.bindings(code),
                    input_files=cache.bindings(paths),rgb_membership=members,data_root=str(data),
                    adapter_sha256=cache.bind(__file__)['sha256'])
    def build(output):
        args.output_dir=str(output)
        builder.build_dataset(args)
    result=cache.ensure(checkpoints/'generation_cache',f'phase{phase}','native_v1',contract,build,validate,
                        protected=[data,collection])
    result['index']=str(Path(result['directory'])/'frame_index.jsonl')
    return result


def geometry(data, collection, maps, checkpoints, *, existing=None):
    from .geometry_v1 import dataset, sources, splits, codec
    if existing:
        dataset.load(existing)
        return dict(action='reuse_existing',dataset=str(Path(existing).resolve()),training_performed=False,
                    selection='explicit_fixed_artifact',candidate_completeness_verified=False)
    maps=Path(maps).resolve(strict=True)
    from . import label_quarantine
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import _iter_routes_stream
    input_schema='geometry_candidate_inputs_v2'
    def contract():
        development=splits.development_contract()
        paths=sorted(collection.glob('*_result.json'))
        if not paths:raise ValueError('Missing formal collection annotations')
        inputs=[*paths,*map(Path,development['sources']),label_quarantine.LEDGER,*maps.glob('*.xodr')]
        members={}
        # Enumerate BEFORE abnormal/missing-route/anchor filtering. A rejected
        # route or image can become eligible when missing inputs are restored.
        for path in paths:
            scenario=path.stem.removesuffix('_result')
            if scenario=='noScenarios':continue
            for route in _iter_routes_stream(path):
                rid=route.get('route_id')
                if not rid or route.get('status')=='data_missing_skip':continue
                if any(Path(x).name!=x or x in ('.','..') for x in (scenario,rid)):
                    raise ValueError('invalid_logical_route')
                run=data/scenario/rid
                members[f'{scenario}/{rid}']=dict(
                    resolved=str(run.resolve(strict=True)) if run.is_dir() else None,
                    metas=(run/'metas').is_dir(),rgb=(run/'rgb').is_dir())
                inputs.extend((run/'metas').glob('*.pkl'))
                inputs.extend((run/'rgb').glob('*'))
        routes=list(dataset.annotation_routes(collection,data,development=development['groups']))
        for route in routes:
            for risk in label_quarantine.load(data,route=(route['scenario'],route['route_id'])):
                inputs.extend(data/s['path'] for s in risk['sources'])
        return dict(schema=input_schema,code=dataset.code_identity(),contract=codec.contract(),
                    inputs=cache.bindings(p for p in inputs if p.is_file()),route_members=members,
                    routes=routes,data_root=str(data),adapter_sha256=cache.bind(__file__)['sha256'])
    requested=contract()
    skipped=[]
    def skip(path,reason):
        note=dict(path=str(path),reason=reason);skipped.append(note)
        print('[geometry data] skip '+json.dumps(note),flush=True)
    def compatible(path):
        value=json.loads(path.read_text())
        if not isinstance(value,dict) or value.get('contract')!=codec.contract() or value.get('code')!=dataset.code_identity():return None
        if value.get('data_root')!=str(data.resolve()) or value.get('scoped_smoke_only'):return None
        # Check artifact integrity BEFORE deciding it is an outdated input version.
        receipt=json.loads(path.with_name('receipt.json').read_text())
        if cache.bind(path)['sha256']!=receipt['dataset_sha256']:raise ValueError('Existing geometry artifact changed')
        completed=None
        if path.is_relative_to((checkpoints/'generation_cache/geometry').absolute()):
            complete_path=path.parent.parent/'complete.json'
            if not complete_path.is_file():raise ValueError('Incomplete geometry cache preserved: '+str(path.parent))
            completed=json.loads(complete_path.read_text())
            if completed['key']!=cache.digest(completed['identity']):
                raise ValueError('Geometry cache identity changed')
        proof_path=path.with_name('generation_inputs.json')
        bound=receipt.get('generation_inputs_sha256')
        if bound is None:
            if proof_path.exists():raise ValueError('Unbound geometry candidate proof')
            dataset.load(path,check_sources=False)
            skip(path,'legacy_missing_complete_candidate_proof')
            return None
        if cache.bind(proof_path)['sha256']!=bound:
            raise ValueError('Geometry candidate input proof changed')
        proof=json.loads(proof_path.read_text())
        if proof['dataset_sha256']!=receipt['dataset_sha256']:
            raise ValueError('Geometry candidate proof belongs to another dataset')
        if completed is not None:
            if (completed['identity']['dependencies']!=proof['dependencies'] or
                    cache.outputs(path.parent)!=completed['files']):
                raise ValueError('Completed geometry cache changed')
        if proof['dependencies']!=requested:
            skip(path,'candidate_dependencies_changed')
            return None
        dataset.load(path)
        return dict(records=cache.digest(value['records']),maps=value['map_sha256'],proof=value['proof'])
    selected=cache.unique_compatible(cache.discover(checkpoints,'dataset.json',depth=6),compatible)
    if selected:
        if contract()!=requested:raise ValueError('Geometry dependencies changed during discovery')
        return dict(action='reuse_existing',dataset=str(selected),training_performed=False,discovery_skipped=skipped)
    def build(output):
        if contract()!=requested:raise ValueError('Geometry dependencies changed before build')
        paths=[Path(p) for p in requested['inputs'] if Path(p).name.endswith('_result.json')]
        dataset.build(requested['routes'],data,sources.LaneMap(maps),output,input_paths=paths)
        proof_path=output/'generation_inputs.json'
        cache.write(proof_path,dict(dataset_sha256=cache.bind(output/'dataset.json')['sha256'],dependencies=requested))
        receipt=json.loads((output/'receipt.json').read_text())
        receipt['generation_inputs_sha256']=cache.bind(proof_path)['sha256']
        cache.write(output/'receipt.json',receipt)
    result=cache.ensure(checkpoints/'generation_cache','geometry','geometry_v2',contract,
        build,lambda out:dataset.load(out/'dataset.json'),protected=[data,collection,maps])
    result['dataset']=str(Path(result['directory'])/'dataset.json')
    result['discovery_skipped']=skipped
    return result


def recipe(path, checkpoints):
    """Other train/test DATA generation, with explicit full dependencies.

    Does not infer equivalence of teacher versions or migrate old source SHAs.
    The recipe is local executable configuration, not an imported result file.
    """
    path=Path(path).resolve(strict=True);spec=json.loads(path.read_text())
    if spec.get('operation')!='generate_data' or not spec.get('sources') or not spec.get('inputs'):
        raise ValueError('Recipe requires generate_data, full sources and inputs')
    if not spec.get('validate_command') or not spec.get('build_command'):
        raise ValueError('Recipe requires both native build and validation commands')
    cwd=Path(spec.get('cwd',str(APP))).resolve(strict=True)
    paths=lambda key:[(cwd/p).resolve(strict=True) for p in spec[key]]
    def contract():
        return dict(recipe=cache.bind(path),sources=cache.bindings(paths('sources')),
                    inputs=cache.bindings(paths('inputs')),parameters=spec.get('parameters',{}),
                    environment=spec.get('environment',{}))
    def execute(command,out):
        if not isinstance(command,list) or not command or not all(isinstance(x,str) for x in command):
            raise ValueError('Commands must be argv arrays, not shell strings')
        args=[s.replace('{output}',str(out)).replace('{python}',sys.executable) for s in command]
        env=dict(os.environ,**spec.get('environment',{}))
        env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        subprocess.run(args,cwd=cwd,env=env,check=True)
    return cache.ensure(checkpoints/'generation_cache',spec['family'],spec['version'],contract,
        lambda out:execute(spec['build_command'],out),lambda out:execute(spec['validate_command'],out),
        protected=[*paths('inputs'),*paths('sources')])


def geometry_plan(path, checkpoints):
    from .geometry_v1 import dataset, planning
    path=Path(path).resolve(strict=True)
    def contract():
        return dict(dataset=cache.bind(path),code=dataset.code_identity(),
                    world_size=4,per_event=1024,epochs=3,seed=20261021)
    def build(output):
        from .geometry_v1.__main__ import main as native_main
        output.mkdir()
        with argv(['plan','--dataset',str(path),'--output',str(output/'plan.json'),
                   '--world-size','4','--per-event','1024','--epochs','3']):
            native_main()
    def validate(output):
        value=json.loads((output/'plan.json').read_text())
        if value.get('projection_only') is not True or value.get('actual_exposure_advanced') is not False or len(value['plans'])!=3:
            raise ValueError('Incomplete three-epoch geometry plan')
    result=cache.ensure(checkpoints/'generation_cache','geometry_plan','world4_3epochs_v1',contract,build,validate,
                       protected=[path.parent])
    return str(Path(result['directory'])/'plan.json')


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--method',required=True,choices=('phase1','phase2','base','single_frame','geometry','phase4','recipe'))
    p.add_argument('--checkpoints',type=Path,default=APP/'checkpoints')
    p.add_argument('--data-root',type=Path,default=APP/'lead_data')
    p.add_argument('--collection',type=Path,default=APP/'keyframe_filter/collection_output')
    p.add_argument('--model-dir',type=Path,default=APP/'checkpoints/Qwen3.5-4B')
    p.add_argument('--maps',type=Path)
    p.add_argument('--existing',type=Path,help='Local dataset.json/frame_index.jsonl, or Phase4 prepared directory')
    p.add_argument('--recipe',type=Path)
    p.add_argument('--receipt',type=Path,help='Write the selected paths for subsequent native train/eval commands')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('native_args',nargs=argparse.REMAINDER)
    a=p.parse_args();a.checkpoints=a.checkpoints.absolute()
    if a.method in ('base','single_frame'):
        from . import phase3_native_runtime
        if phase3_native_runtime.needed():
            raise SystemExit(phase3_native_runtime.execute('data_prepare',sys.argv[1:],checkpoints=a.checkpoints))
    if a.method=='base' and a.existing:p.error('base discovers common_receipt automatically; do not pass --existing')
    if a.method=='recipe' and a.existing:p.error('recipe dependencies belong in the recipe, not --existing')
    native_args=a.native_args[1:] if a.native_args[:1]==['--'] else a.native_args
    if native_args and a.method not in ('phase1','phase2'):p.error('Native extra arguments supported only for phase1/2')
    if a.method=='recipe':
        if not a.recipe:p.error('--recipe required')
        result=recipe(a.recipe,a.checkpoints)
    elif a.method in ('phase1','phase2'):
        result=phase12(int(a.method[-1]),a.data_root,a.collection,a.checkpoints,native_args=native_args,existing=a.existing)
    elif a.method=='geometry':
        if not a.maps and not a.existing:p.error('--maps required (project map view), no cross-machine path guessing')
        result=geometry(a.data_root.resolve(strict=True),a.collection.resolve(strict=True),a.maps,a.checkpoints,existing=a.existing)
        result['plan']=geometry_plan(result['dataset'],a.checkpoints)
    elif a.method=='phase4':
        from . import phase4_data_ready
        result=phase4_data_ready.ensure(a.data_root,a.collection,a.model_dir,a.checkpoints,
                                       existing=a.existing,workers=a.workers)
    else:
        from . import phase3_common_data as common, phase3_data_ready
        def run(name,command,log):
            from . import phase3_base_trial as trial
            result=trial.run_stage(command,log,dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                                   timeout=172800,reserve_bytes=20*1024**3)
            if result['exit_code']:raise RuntimeError(f'{name} failed: {result}')
        index,reference=common.ensure(a.checkpoints/'phase3_common_v1',a.data_root,a.collection,run,
                                      workers=a.workers,single_frame=a.method=='single_frame' and not a.existing,discover_existing=True)
        result=dict(index=str(index),reference=str(reference),training_performed=False)
        if a.method=='single_frame':
            if a.existing:
                selected=phase3_data_ready.ensure(index,a.data_root,a.checkpoints/'generation_cache',candidates=[a.existing])
                result.update(selected)
            else:result['dataset']=str(common.derived_path(reference.parent))
    if a.receipt:
        a.receipt.parent.mkdir(parents=True,exist_ok=True)
        cache.write(a.receipt,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
