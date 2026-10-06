"""Build both full RGB datasets once; reuse only source-bound, complete pairs."""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from .identity import ROOT, contract, digest, file_sha, write_json

POLICY = 'full_paired_pipeline_v1'


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def build_lock(base):
    # Keep the inode: deleting lock files allows simultaneous builders.
    with (base / '.build.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as ex:
            raise ValueError(f'another pipeline is building {base}; wait for it to finish') from ex
        yield


def request_identity(data_root, annotations, candidate_pool=None, production_index=None, registry=None):
    if bool(production_index) != bool(registry):
        raise ValueError('PRODUCTION_INDEX requires a source-bound teacher registry, and conversely')
    if production_index and not candidate_pool:
        raise ValueError('external PRODUCTION_INDEX requires its CANDIDATE_POOL')
    return dict(policy=POLICY, contract=contract(), data_root=str(Path(data_root).resolve()),
                annotations=dict(path=str(Path(annotations).resolve()), sha256=file_sha(annotations)),
                external={key:dict(path=str(Path(value).resolve()), sha256=file_sha(value))
                          for key,value in (('candidate_pool',candidate_pool),('production_index',production_index),
                                            ('teacher_registry',registry)) if value},
                max_teacher_questions_per_route=0, rgb_modes=[2,4])


def artifact_hashes(base):
    files = {}
    for mode in (2,4):
        path=base/f'data{mode}'
        manifest=read(path/'manifest.json')
        for name in ('manifest.json','review_queue.jsonl','candidate_pool.json',
                     'teacher_registry.json','production/index.json',*manifest['files']):
            files[str((path/name).relative_to(base))]=file_sha(path/name)
    return files


def validate_ready(base, request):
    receipt=read(base/'pipeline_ready.json')
    if receipt['request'] != request:
        raise ValueError('full pipeline source/configuration changed; use a fresh DATA_DIR and SKIP_BUILD=0')
    if receipt['artifacts'] != artifact_hashes(base):
        raise ValueError('full pipeline dataset content changed; refusing reuse')
    if not receipt['pairing']['pairs'] or receipt['pairing']['exclusions']:
        raise ValueError('full pipeline may not discard unpaired training questions')
    from .full_sampling import validate_dataset_scope
    for mode in (2,4):
        manifest=read(base/f'data{mode}'/'manifest.json')
        if manifest['contract'] != request['contract'] or manifest['rgb_mode'] != mode:
            raise ValueError('full pipeline dataset contract/RGB mode mismatch')
        if not manifest.get('production_index_sha256'):
            raise ValueError('full pipeline requires teacher production, not a manual-only dataset')
        validate_dataset_scope(manifest,receipt['pairing'])
    return receipt


def default_registry():
    from . import teacher_rules as rules, teacher_approval as approval
    from .taxonomy import EVENTS, event_edges, COMMON
    from .route_context import RECOVER_FOLLOW
    classes={rules.rule_class(event,edge.key,mode) for event in ('U-E1','U-E4')
             for edge in (*event_edges(EVENTS[event][1]),*COMMON,RECOVER_FOLLOW)
             for mode in (2,4) if edge.key!='restrict'}
    return approval.weak_registry(classes,'full_pipeline_v38_exact_event_equal')


def prepare(base, data_root, annotations, *, workers=16, skip_build=False,
            candidate_pool=None, production_index=None, registry=None):
    base=Path(base).resolve();data_root=Path(data_root).resolve();annotations=Path(annotations).resolve()
    if not data_root.is_dir():raise ValueError(f'DATA_ROOT missing: {data_root}')
    if not 1 <= workers <= 64:raise ValueError('BUILD_WORKERS must be in 1..64')
    request=request_identity(data_root,annotations,candidate_pool,production_index,registry)
    if skip_build:
        if not (base/'pipeline_ready.json').is_file():
            raise ValueError('SKIP_BUILD=1 needs a completed paired full build; run once with SKIP_BUILD=0')
        return validate_ready(base,request)
    base.mkdir(parents=True,exist_ok=True)
    with build_lock(base):
        if (base/'pipeline_ready.json').exists():return validate_ready(base,request)
        request_path=base/'build_request.json'
        if request_path.exists():
            if read(request_path)!=request:
                raise ValueError('build source/configuration changed; choose a fresh DATA_DIR; old artifacts retained')
        else:
            if any(p.name!='.build.lock' for p in base.iterdir()):
                raise ValueError('DATA_DIR is not an owned pipeline build; choose a fresh directory')
            write_json(request_path,request)
        from . import candidate_pool as cp, teacher_replay as replay
        pool_path=Path(candidate_pool) if candidate_pool else base/'candidates.json'
        pool=read(pool_path) if candidate_pool or pool_path.exists() else cp.scan(data_root,pool_path)
        cp.validate(pool)
        registry_path=Path(registry) if registry else base/'weak_registry.json'
        if not registry:write_json(registry_path,default_registry())
        index=Path(production_index) if production_index else base/'full_production'/'index.json'
        if not production_index and not index.exists():
            # Interrupted route files are never accepted as completed evidence.
            for pending in sorted(index.parent.glob('*.pending')):
                pending.rename(pending.with_name(pending.name+f'.interrupted-{time.time_ns()}'))
            print('[Phase4 build] replaying every eligible route in all frozen splits',flush=True)
            replay.produce(pool,data_root,index.parent,workers=workers)
        accounting=replay.accounting(pool,index)
        if not accounting['complete_causal_production']:
            raise ValueError('full pipeline requires complete route/frame accounting')
        for mode in (2,4):
            out=base/f'data{mode}'
            if out.exists():
                # A manifest is written last; partial builds are retained for diagnosis.
                if not (out/'manifest.json').is_file():
                    out.rename(base/f'data{mode}.interrupted-{time.time_ns()}')
                else:continue
            print(f'[Phase4 build] compiling RGB{mode}, no per-route question cap',flush=True)
            subprocess.run([sys.executable,'-m','qwen3vl_local.sft_new_loop_phase4.dataset',
                            '--annotations',str(annotations),'--candidate-pool',str(pool_path),
                            '--production-index',str(index),'--teacher-registry',str(registry_path),
                            '--max-teacher-questions-per-route','0','--data-root',str(data_root),
                            '--output-dir',str(out),'--rgb-mode',str(mode)],check=True)
        from .dataset import load_dataset
        from .paired_training import load_view
        from .full_sampling import validate_dataset_scope
        data,manifest=load_dataset(base/'data2',require_trainable=True)
        _,pairing=load_view(data,manifest,base/'data4')
        for mode in (2,4):validate_dataset_scope(read(base/f'data{mode}'/'manifest.json'),pairing)
        if contract()!=request['contract']:raise ValueError('source changed while building; artifacts not published')
        receipt=dict(request=request,accounting=accounting,pairing=pairing,artifacts=artifact_hashes(base))
        pending=base/'pipeline_ready.pending'
        write_json(pending,receipt);pending.replace(base/'pipeline_ready.json')
        return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data')
    p.add_argument('--annotations',type=Path,default=ROOT/'reviewed_state_pairs_v9.json')
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--skip-build',action='store_true')
    p.add_argument('--candidate-pool',type=Path)
    p.add_argument('--production-index',type=Path)
    p.add_argument('--teacher-registry',type=Path)
    a=p.parse_args()
    try:
        result=prepare(a.data_dir,a.data_root,a.annotations,workers=a.workers,skip_build=a.skip_build,
                       candidate_pool=a.candidate_pool,production_index=a.production_index,registry=a.teacher_registry)
    except (ValueError,OSError,KeyError,subprocess.CalledProcessError) as ex:
        p.exit(2,f'[Phase4 build] {ex}\n')
    print(json.dumps(dict(data_dir=str(a.data_dir),paired_questions=result['pairing']['pairs'],
                         accounting=result['accounting']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
