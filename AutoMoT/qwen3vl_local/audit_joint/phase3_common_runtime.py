"""Explicit common-reference adapter; frozen historical entry points stay unchanged."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import importlib
import sys
from . import phase3_base_trial as trial, phase3_base_trial_worker as base
from . import phase3_common_data as common

ALLOWED = ('phase3_launch_worker','phase3_base_trial_worker','phase3_single_frame.worker')


@contextmanager
def activate(reference, data):
    reference = Path(reference).resolve(strict=True)
    if reference.name != 'reference.json': raise ValueError('Expected common reference.json')
    common.validate(reference.parent,Path(data).resolve(strict=True))
    expected = trial.file_sha(reference)
    old_path, old_reference, old_select, old_args = trial.REFERENCE, trial.reference, trial.select_reference, base.make_args
    def read_reference():
        if trial.file_sha(reference) != expected: raise ValueError('Common reference changed during execution')
        ref = old_reference()
        if ref.get('reference_kind') == 'new_common_dataset_v1':
            ids = ref['test_plan']['selected_ids']
            if len(ids)!=380 or len(set(ids))!=380: raise ValueError('Malformed common test IDs')
        return ref
    def select(work,ref,**kwargs):
        selected,report = old_select(work,ref,**kwargs)
        if ref.get('reference_kind') == 'new_common_dataset_v1':
            report.update(policy='phase3_fixed_common_members_v1',scope='new shared fixed members; historical scores not comparable')
        return selected,report
    def make_args(job):
        args=old_args(job)
        args.base_trial['common_reference']=dict(reference_sha256=expected,
            adapter_sha256=trial.file_sha(Path(__file__)),
            builder_sha256=trial.file_sha(Path(common.__file__)))
        if read_reference().get('reference_kind') == 'new_common_dataset_v1':
            args.base_trial['validation']='new shared fixed members; no historical score comparison'
        return args
    try:
        trial.REFERENCE=reference; trial.reference=read_reference; trial.select_reference=select; base.make_args=make_args
        read_reference()
        yield
    finally:
        trial.REFERENCE=old_path; trial.reference=old_reference; trial.select_reference=old_select; base.make_args=old_args


def wrap(cmd,reference,data):
    """Insert bootstrap at the application module, leaving torchrun's boundary intact."""
    cmd=list(cmd)
    module=next((x for x in cmd if x in ['qwen3vl_local.audit_joint.'+n for n in ALLOWED]),None)
    if module is None: raise ValueError('Unsupported common-data worker command')
    i=cmd.index(module)
    return cmd[:i]+['qwen3vl_local.audit_joint.phase3_common_runtime','--reference',str(reference),
        '--data-root',str(data),'--module',module.removeprefix('qwen3vl_local.audit_joint.'),'--']+cmd[i+1:]


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--module',choices=ALLOWED,required=True)
    p.add_argument('args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    with activate(a.reference,a.data_root):
        sys.argv=[a.module,*(a.args[1:] if a.args[:1]==['--'] else a.args)]
        # Import the same module instance whose make_args is adapted above.
        # runpy would define a second native worker and lose the resume binding.
        importlib.import_module('qwen3vl_local.audit_joint.'+a.module).main()


if __name__=='__main__': main()
