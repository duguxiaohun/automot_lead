"""Rebuild admissions in a new run; narrowly reuse reviewed processor evidence."""
import json
from pathlib import Path
from qwen3vl_local.audit_joint.abc_delivery import sha,write
from . import contracts


def differences(saved,current):
    return [dict(file=k,before=saved.get(k),after=current.get(k))
            for k in sorted(set(saved)|set(current)) if saved.get(k)!=current.get(k)]


def require_current(gate,*,output=None):
    current=contracts.source_identity();changed=differences(gate.get('code',{}),current)
    if changed:
        report=dict(status='source_version_mismatch',differences=changed,
                    recovery='start a new versioned run with --recover-from OLD_RUN; never overwrite old admission SHA')
        if output:write(output,report)
        raise ValueError('admission_source_changed: '+json.dumps(report,ensure_ascii=False))
    return current


def inspect_old(root):
    """A preparation-only recovery must not silently restart any trained group."""
    root=Path(root).resolve(strict=True)
    from .audit import GROUPS
    occupied=[name for name in GROUPS if (root/name).exists()]
    if occupied:raise ValueError('recovery_has_training_artifacts_keep_original_version: '+','.join(occupied))
    path=root/'admission.json'
    gate=json.loads(path.read_text()) if path.is_file() else {}
    return root,gate


def data_directory(directory):
    """An incompatible data-code contract gets its own build, never a rewritten receipt."""
    from qwen3vl_local.audit_joint.abc_delivery import digest
    directory=Path(directory);path=directory/'manifest.json'
    if not path.is_file():return directory
    manifest=json.loads(path.read_text());current=contracts.source_identity(data_only=True)
    if manifest.get('code')==current:return directory
    version=digest(dict(source=current,parent_manifest=sha(path)))[:16]
    target=directory.with_name(directory.name+'_version_'+version)
    print(f'[ABC recovery] data code changed; prepare separate dataset: {target}; preserve {directory}',flush=True)
    return target


def compatible(saved,current):
    if saved==current:return True
    policy=json.loads(Path(__file__).with_name('processor_compatibility.json').read_text())
    # Only exact reviewed transitions; version strings alone never authorize reuse.
    for item in differences(saved,current):
        rule=policy['changes'].get(item['file'])
        if not rule or item['before'] not in rule['before'] or item['after']!=rule['after']:return False
    return True


def reuse_processor(old_root,new_gate,destination):
    old_root,old_gate=inspect_old(old_root);path=old_root/'processor.json'
    report=dict(status='recompute_required',source=str(old_root),reason=None)
    try:
        if sha(path)!=old_gate.get('processor_sha256'):raise ValueError('old_processor_receipt_hash_mismatch')
        old=json.loads(path.read_text());current=contracts.source_identity()
        for key in ('dataset','splits','manifest_sha256','plan_sha256','resources'):
            if old_gate.get(key)!=new_gate.get(key):raise ValueError('changed_'+key)
        if 'model' in old and old['model']!=new_gate['resources']['qwen']:
            raise ValueError('changed_processor_model')
        if not compatible(old_gate['code'],current) or not compatible(old['code'],current):
            raise ValueError('unreviewed_source_change')
        if (old.get('schema')!='abc_full_pool_processor_v1' or not old.get('full_pool') or not old.get('no_truncation')
            or old.get('dataset')!=new_gate['dataset'] or old.get('splits')!=new_gate['splits']):
            raise ValueError('incomplete_processor_evidence')
        expected=sum(new_gate['record_counts'].values())
        if old.get('checked')!=3*expected or old.get('per_variant')!={v:expected for v in 'abc'}:
            raise ValueError('processor_counts_changed')
        for v in 'abc':
            for split in ('train','val','test'):
                for metric in ('tokens','visual_tokens'):
                    cell=old['extrema'][v][split][metric]
                    if cell['split']!=split or cell['variant']!=v or cell['tokens']<=0 or cell['tokens']>8192 or cell['visual_tokens']<=0:
                        raise ValueError('invalid_processor_extrema')
        proof=dict(source=str(path),sha256=sha(path),admission_sha256=sha(old_root/'admission.json'),
            policy_sha256=sha(Path(__file__).with_name('processor_compatibility.json')),
            changes=differences(old['code'],current),raw_data_and_resources_reverified=True)
        result=dict(old,code=current,reused_processor=proof,model=new_gate['resources']['qwen'])
        write(destination,result);report.update(status='processor_reused',proof=proof)
    except (ValueError,KeyError,FileNotFoundError,TypeError) as exc:
        report['reason']=str(exc)
    write(Path(destination).with_name('recovery.json'),report)
    print('[ABC recovery] '+json.dumps(report,ensure_ascii=False),flush=True)
    return report['status']=='processor_reused'


def failure_package(root,state):
    from qwen3vl_local.audit_joint.abc_delivery import pack
    root=Path(root);required={'pipeline.json':root/'pipeline.json'}
    for name in ('admission.json','processor.json','source_mismatch.json','recovery.json','execution_version.json'):
        if (root/name).is_file():required[name]=root/name
    for p in sorted(root.glob('**/progress/*.json')):
        required[str(p.relative_to(root))]=p
    for p in sorted((root/'logs').glob('*.log')):
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-64000));tail=f.read()
        target=root/'failure_delivery/log_tails'/p.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tail)
        required['log_tails/'+p.name]=target
    return pack(root/'failure_delivery/diagnostic.zip',required,summary=dict(final_results=False,error=state.get('error')))
