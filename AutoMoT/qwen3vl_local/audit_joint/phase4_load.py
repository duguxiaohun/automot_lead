"""Bounded read adapter for frozen native datasets; explicit student-view migration.

The producer source files remain byte-identical. The isolated reader reuses one
validated reservation snapshot and validates it again before returning data.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

from .benchmark_runtime import progress, run_owned, trace_waits
from .io import file_sha


@contextmanager
def validated_teacher_sources():
    """Reuse frozen teacher source hashes only during one isolated read.

    Question validation still runs for every call. Recompute source bytes and
    identity at transaction end; no persistent cache or row/proof bypass.
    Patching this module's file_sha also covers previously imported identity
    aliases, without changing any frozen producer source bytes.
    """
    from qwen3vl_local.sft_new_loop_phase4 import teacher_rules as rules
    if threading.current_thread() is not threading.main_thread() or threading.active_count()!=1:
        raise RuntimeError('teacher source adapter requires an isolated single-threaded process')
    original=rules.file_sha
    if getattr(original,'_teacher_snapshot',False):raise RuntimeError('nested teacher source adapter')
    before=deepcopy(rules.identity())
    hashes={rules.ROOT/name:sha for name,sha in before['sources'].items()}
    calls=0
    def cached(path):
        nonlocal calls
        key=Path(path)
        if key not in hashes:return original(path)
        calls+=1
        return hashes[key]
    cached._teacher_snapshot=True
    rules.file_sha=cached
    progress(f'teacher source snapshot START files={len(hashes)}')
    try:
        yield
        rules.file_sha=original
        if rules.identity()!=before:raise ValueError('teacher sources or parameters changed during read transaction')
        progress(f'teacher source snapshot DONE reused_hash_calls={calls}')
    finally:
        rules.file_sha=original


@contextmanager
def validated_question_reuse():
    """Reuse completed checks only for content-identical questions in this load.

    The first visit always calls the frozen validator. Subsequent visits hash
    the entire question again (excluding its self hash), and also check the
    requested teacher identity. Never trust question_id alone or object identity.
    Teacher source/parameter snapshots must surround this context.
    """
    from qwen3vl_local.sft_new_loop_phase4 import teacher_replay as replay, teacher_data, teacher_rules
    from qwen3vl_local.sft_new_loop_phase4.identity import digest
    if not getattr(teacher_rules.file_sha,'_teacher_snapshot',False):
        raise RuntimeError('question reuse requires a verified teacher source transaction')
    original=replay.validate_question; alias=teacher_data.validate_question
    if getattr(original,'_question_reuse',False):raise RuntimeError('nested question reuse')
    if alias is not original:raise RuntimeError('unexpected teacher validator alias')
    teacher_sha=digest(teacher_rules.identity())
    checked={};hits=0;misses=0
    def validate(q,*,expected_teacher_sha=None):
        nonlocal hits,misses
        expected=expected_teacher_sha or teacher_sha
        key=q.get('question_id') if isinstance(q,dict) else None
        if (isinstance(key,str) and checked.get(key)==expected
                and q.get('teacher_sha256')==expected
                and digest({k:v for k,v in q.items() if k!='question_id'})==key):
            hits+=1
            return q
        result=original(q,expected_teacher_sha=expected)
        # Bound auxiliary memory on full production pools; a missed cache entry
        # simply runs the original validator again.
        if len(checked)<300000:checked[q['question_id']]=expected
        misses+=1
        return result
    validate._question_reuse=True
    replay.validate_question=teacher_data.validate_question=validate
    try:
        yield
        progress(f'question validation reuse DONE full_checks={misses} content_verified_reuses={hits}')
    finally:
        replay.validate_question=original;teacher_data.validate_question=alias


@contextmanager
def load_progress():
    """Report completion of native validation stages without replacing checks."""
    from qwen3vl_local.sft_new_loop_phase4 import dataset as native
    from qwen3vl_local.sft_new_loop_phase4 import visible_scope,risk_review,candidate_pool
    targets=[(native,name) for name in ('read_rows','validate_model_inputs','coverage_report','training_report')]
    targets += [(visible_scope,'validate_rows'),(risk_review,'validate_rows'),(candidate_pool,'coverage')]
    originals={(module,name):getattr(module,name) for module,name in targets}
    def wrap(name,fn):
        def run(*args,**kwargs):
            label=name+(f' {args[0]}' if name=='read_rows' else '')
            started=time.monotonic();progress(label+' START')
            result=fn(*args,**kwargs)
            progress(f'{label} DONE seconds={time.monotonic()-started:.2f}'
                     +(f' rows={len(result)}' if name=='read_rows' else ''))
            return result
        return run
    try:
        for (module,name),fn in originals.items():
            label=name if module is native else module.__name__.rsplit('.',1)[-1]+'.'+name
            setattr(module,name,wrap(label,fn))
        yield
    finally:
        for (module,name),fn in originals.items():setattr(module,name,fn)


@contextmanager
def validated_reservations():
    """Single-threaded read transaction, not a persistent or unverified cache.

    Used only by dedicated CPU preparation processes. Restore the native callable
    even on failure; changes to protocol bytes or frozen sources fail the load.
    """
    from qwen3vl_local.sft_new_loop_phase4 import dataset as native
    if threading.current_thread() is not threading.main_thread() or threading.active_count()!=1:
        raise RuntimeError('reservation read adapter requires an isolated single-threaded process')
    original = native.producer_check_reservations
    if getattr(original, '_reservation_snapshot', False):
        raise RuntimeError('nested reservation read adapter')
    path = native.ROOT/native.PRODUCER_CHECK_PLAN
    before = file_sha(path)
    owners = original()
    if file_sha(path) != before:
        raise ValueError('producer protocol changed while opening read transaction')
    calls = 0
    started = time.monotonic()
    def cached():
        nonlocal calls
        calls += 1
        if calls == 1 or calls % 50000 == 0:
            progress(f'protocol snapshot reuse calls={calls} elapsed={time.monotonic()-started:.1f}s')
        return dict(owners)
    cached._reservation_snapshot = True
    native.producer_check_reservations = cached
    try:
        with validated_teacher_sources(), validated_question_reuse(), load_progress():
            yield
        progress('protocol end validation START')
        if file_sha(path) != before or original() != owners or file_sha(path) != before:
            raise ValueError('producer protocol changed during read transaction')
        progress(f'protocol end validation DONE reuse_calls={calls}')
    finally:
        native.producer_check_reservations = original


def inspect_view(view):
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import contract, POLICY
    descriptor = json.loads((view/'student_view.json').read_text())
    current = contract()
    if descriptor.get('schema') != POLICY:
        raise ValueError('unsupported student schema')
    saved = descriptor['source']
    if saved == current:
        return descriptor, current, []
    # Only this independently reviewed runtime-only v61 -> v62 transition.
    release = json.loads(Path(__file__).with_name('release_sources_v61_batch_audit_20261015.json').read_text())['files']
    keys = {'../../sequence_batch.py': 'qwen3vl_local/sequence_batch.py',
            **{name: 'qwen3vl_local/sft_new_loop_phase4/rgb_short/'+name
               for name in ('evaluate.py','reload_check.py','run.sh')}}
    normalized = deepcopy(saved)
    changed = []
    for name, path in keys.items():
        old = normalized.get('sources', {}).get(name)
        new = current['sources'][name]
        if old == new:
            continue
        if old != release[path]:
            raise ValueError('unrecognized historical student source: '+name)
        normalized['sources'][name] = new
        changed.append(name)
    if normalized != current:
        raise ValueError('view differs beyond reviewed runtime-only sources; migration refused')
    return descriptor, current, changed


def migrate_view(view, output, data_root, *, production_cache=None):
    """Create a new descriptor only after full native/member revalidation.

    Reuse the immutable RGB identity sidecar because all RGB transformation and
    labeling sources are identical. No old descriptor/checkpoint is rewritten.
    """
    from qwen3vl_local.sft_new_loop_phase4 import dataset as native
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import data as student
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import observation_contract, contract
    from qwen3vl_local.sft_new_loop_phase4.identity import digest
    from .batch_benchmark import write
    if output.resolve().is_relative_to(view.resolve()):
        raise ValueError('new view must be outside old view')
    descriptor, current, changed = inspect_view(view)
    descriptor_sha = file_sha(view/'student_view.json')
    binding = descriptor['parent']
    parent = Path(binding['logical_path'])
    if str(parent.resolve()) != binding['resolved_path']:
        raise ValueError('parent path changed')
    manifest = json.loads((parent/'manifest.json').read_text())
    if student.parent_binding(parent, manifest) != binding['files']:
        raise ValueError('native parent bytes changed')
    for mode in (2,4):
        if descriptor['observation_contracts'][str(mode)] != observation_contract(mode):
            raise ValueError('observation offsets changed')
    exposed, files = student.exposure_inventory()
    if files != descriptor['exposure_files']:
        raise ValueError('exposure inventory changed')
    identities_path = view/'rgb_identities.json'
    if file_sha(identities_path) != descriptor['rgb_identities_sha256']:
        raise ValueError('RGB identity sidecar changed')
    identities = json.loads(identities_path.read_text())
    progress('native full validation START')
    from .phase4_production_cache import production_routes
    with production_routes(parent, production_cache), validated_reservations():
        data, _ = native.load_dataset(parent)
    progress('native full validation DONE; paired view revalidation START')
    for mode in (2,4):
        selected, exclusions = student.paired_view(data, mode, exposed, identities=identities)
        if (exclusions != descriptor['exclusions'] or
                {s:len(rows) for s,rows in selected.items()} != descriptor['counts'] or
                {s:digest([r['id'] for r in rows]) for s,rows in selected.items()} != descriptor['selected_ids']):
            raise ValueError('student membership changed')
        del selected
        progress(f'paired view RGB={mode} DONE')
    if (file_sha(view/'student_view.json') != descriptor_sha or
            file_sha(identities_path) != descriptor['rgb_identities_sha256'] or
            student.parent_binding(parent, manifest) != binding['files'] or
            student.exposure_inventory()[1] != files or contract()!=current):
        raise ValueError('inputs or source changed during view migration')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(identities_path, output/'rgb_identities.json')
    if file_sha(output/'rgb_identities.json') != descriptor['rgb_identities_sha256']:
        raise ValueError('copied RGB sidecar mismatch')
    new = deepcopy(descriptor)
    new['source'] = current
    new['runtime_migration'] = dict(old_view=str(view.resolve()), old_descriptor_sha256=descriptor_sha,
        changed_sources=changed, full_native_and_paired_membership_revalidated=True,
        new_rgb_content_verification=False, old_checkpoint_resume_allowed=False)
    write(output/'student_view.json', new)
    return dict(status='view_migrated', view=str(output), changed_sources=changed,
                counts=new['counts'], old_descriptor_sha256=descriptor_sha,
                scope='Runtime-only migration; original RGB identity sidecar reused, no new label certification')


def main():
    from .guard import no_core
    from .batch_benchmark import write
    no_core()
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--native',type=Path)
    p.add_argument('--view',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('lead_data'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=120)
    p.add_argument('--production-cache',type=Path,help='Explicit local verified-route cache; never a full-pool admission')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if (a.native is None)==(a.view is None) or a.timeout<1:p.error('choose exactly one of --native / --view and a positive timeout')
    if a.worker:
        trace_waits()
        if a.view:
            result=migrate_view(a.view,a.output/'view',a.data_root,production_cache=a.production_cache)
        else:
            from qwen3vl_local.sft_new_loop_phase4.dataset import load_dataset
            progress('native full validation START')
            from .phase4_production_cache import production_routes
            with production_routes(a.native,a.production_cache), validated_reservations():
                data,manifest=load_dataset(a.native)
            result=dict(status='native_validated',counts={s:len(rows) for s,rows in data.items()})
        write(a.output/'result.json',result)
        print(json.dumps(result),flush=True)
        return
    source = a.view if a.view else a.native
    if a.output.resolve().is_relative_to(source.resolve()):
        p.error('output must be outside the input directory')
    if a.view:
        descriptor, _, _ = inspect_view(a.view)
        if a.output.resolve().is_relative_to(Path(descriptor['parent']['logical_path']).resolve()):
            p.error('output must be outside native parent')
    a.output.mkdir(parents=True,exist_ok=False)
    args=[sys.executable,'-u','-m','qwen3vl_local.audit_joint.phase4_load','--worker',
          '--output',str(a.output),'--data-root',str(a.data_root)]
    args += ['--native',str(a.native)] if a.native else ['--view',str(a.view)]
    if a.production_cache:args += ['--production-cache',str(a.production_cache)]
    progress(f'output={a.output}; wall limit={a.timeout}s; CPU only')
    code=run_owned(args,a.output/'load.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''),a.timeout)
    write(a.output/'status.json',dict(returncode=code,completed=code==0,log=str(a.output/'load.log')))
    raise SystemExit(code)


if __name__=='__main__':main()
