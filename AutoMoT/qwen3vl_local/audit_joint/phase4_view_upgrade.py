"""Explicit known-source view upgrade after sampling/runtime fixes; no teacher replay."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import threading
import time

from . import phase4_load
from .benchmark_runtime import progress, run_owned, trace_waits
from qwen3vl_local.sft_new_loop_phase4.identity import write_json


def inspect_view(view):
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import contract,POLICY
    descriptor=json.loads((Path(view)/'student_view.json').read_text());current=contract()
    if descriptor.get('schema')!=POLICY:raise ValueError('unsupported student schema')
    saved=descriptor['source'];normalized=deepcopy(saved);changes=[]
    versions=['release_sources_v61_batch_audit_20261015.json',
              'release_sources_v70_selected_b2_20261016.json',
              'release_sources_v71_sampling_runtime_20261016.json',
              'release_sources_v72_action_exposure_20261017.json',
              'release_sources_v73_epoch_recovery_20261017.json',
              'release_sources_v74_batch_runtime_20261017.json',
              'release_sources_v78_worker_diagnostics_20261017.json']
    releases=[json.loads(Path(__file__).with_name(v).read_text())['files'] for v in versions]
    # Reviewed runtime/sampling and byte-equivalent view conversion edits only.
    names={'../../sequence_batch.py','../../sft_new_loop_phase3/sequence_evaluation.py',
           'contract.py','data.py','train.py','run.sh','run_automatic.sh','batch_sampling.py',
           'evaluate.py','reload_check.py','runtime_data.py','model.py','../../batch_runtime.py'}
    root=Path('qwen3vl_local/sft_new_loop_phase4/rgb_short')
    for name in names:
        if name in ('runtime_data.py','../../batch_runtime.py') and name not in normalized.get('sources',{}):continue
        if normalized.get('sources',{}).get(name)==current['sources'][name]:continue
        # Resolve lexical ../ segments without relying on files on disk.
        parts=[]
        for part in (root/name).parts:
            if part=='..':parts.pop()
            else:parts.append(part)
        key='/'.join(parts)
        if normalized.get('sources',{}).get(name) not in {r.get(key) for r in releases if r.get(key)}:
            raise ValueError('unrecognized historical student source: '+name)
        normalized['sources'][name]=current['sources'][name];changes.append(name)
    if 'runtime_data.py' not in normalized.get('sources',{}):
        normalized['sources']['runtime_data.py']=current['sources']['runtime_data.py']
        changes.append('runtime_data.py (added)')
    name='rule_baseline.py';key='qwen3vl_local/audit_joint/'+name
    old=normalized.get('automatic_admission_sources',{}).get(name)
    if old!=current['automatic_admission_sources'][name]:
        if old not in {r.get(key) for r in releases if r.get(key)}:
            raise ValueError('unrecognized historical admission source')
        normalized['automatic_admission_sources'][name]=current['automatic_admission_sources'][name]
        changes.append('automatic_admission_sources/'+name)
    name='../../batch_runtime.py'
    if name not in normalized.get('sources',{}):
        key='qwen3vl_local/sft_new_loop_phase4/rgb_short/contract.py'
        if saved.get('sources',{}).get('contract.py') not in {r.get(key) for r in releases if 'qwen3vl_local/batch_runtime.py' not in r}:
            raise ValueError('missing batch runtime binding in nonhistorical view')
        normalized['sources'][name]=current['sources'][name]
        changes.append(name+' (added)')
    name='evaluate.sh'
    if name not in normalized.get('sources',{}):
        key='qwen3vl_local/sft_new_loop_phase4/rgb_short/contract.py'
        if saved.get('sources',{}).get('contract.py') not in {r.get(key) for r in releases}:
            raise ValueError('missing distributed evaluation launcher binding')
        normalized['sources'][name]=current['sources'][name]
        changes.append(name+' (added)')
    if normalized!=current:raise ValueError('view differs beyond reviewed runtime/sampling changes')
    return descriptor,current,changes


def upgrade(view,output,data_root,cache=None):
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import data as student
    if threading.current_thread() is not threading.main_thread() or threading.active_count()!=1:
        raise RuntimeError('view upgrade requires an isolated single-thread process')
    original=phase4_load.inspect_view
    original_pair=student.paired_view
    pending=None
    def paired_once(data,mode,exposed,*,identities=None,data_root=None):
        nonlocal pending
        if pending is None:
            if mode!=2 or identities is None:raise ValueError('unexpected paired migration order')
            started=time.monotonic()
            both,exclusions=student.paired_views(data,exposed,identities=identities)
            progress(f'paired conversion both modes DONE seconds={time.monotonic()-started:.2f}')
            pending=(data,identities,set(exposed),both.pop(4),exclusions)
            return both.pop(2),exclusions
        source,sidecar,groups,four,exclusions=pending
        if mode!=4 or data is not source or identities is not sidecar or groups!=exposed:
            raise ValueError('paired migration input changed')
        pending=None
        return four,exclusions
    phase4_load.inspect_view=inspect_view
    student.paired_view=paired_once
    try:return phase4_load.migrate_view(view,output,data_root,production_cache=cache)
    finally:
        phase4_load.inspect_view=original
        student.paired_view=original_pair


def main():
    from .guard import no_core
    no_core();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--view',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('lead_data'))
    p.add_argument('--production-cache',type=Path)
    p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.timeout<1:p.error('positive timeout required')
    if a.worker:
        trace_waits()
        result=upgrade(a.view,a.output/'view',a.data_root,a.production_cache)
        write_json(a.output/'result.json',result);print(json.dumps(result),flush=True);return
    descriptor,_,_=inspect_view(a.view)
    if any(a.output.resolve().is_relative_to(v.resolve()) for v in
           (a.view,Path(descriptor['parent']['logical_path']))):p.error('output must be outside source view/native')
    a.output.mkdir(parents=True,exist_ok=False)
    args=[sys.executable,'-u','-m','qwen3vl_local.audit_joint.phase4_view_upgrade','--worker',
          '--view',str(a.view),'--output',str(a.output),'--data-root',str(a.data_root)]
    if a.production_cache:args+=['--production-cache',str(a.production_cache)]
    code=run_owned(args,a.output/'load.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''),a.timeout)
    write_json(a.output/'status.json',dict(returncode=code,completed=code==0))
    raise SystemExit(code)


if __name__=='__main__':main()
