"""Explicit runtime-only rebinding of unchanged derived labels; no relabeling.

Only runtime.py/objective.py/worker.py may change among old bound sources.
New contract/recovery utilities may be added. Native release and all dataset,
history, target, planning and CLI sources must still match the old artifact.
"""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import socket
from . import dataset
from qwen3vl_local.audit_joint import phase3_base_trial as trial
from qwen3vl_local.audit_joint.phase3_base_dense import provenance,baseline


def refresh(source,output):
    source=Path(source).resolve(strict=True);output=Path(output).resolve()
    old=trial.read(source);receipt=trial.read(source.with_name('dataset_summary.json'))
    sha=trial.file_sha(source)
    if sha!=receipt['artifact_sha256'] or old['sources']!=receipt['sources'] or old['config']!=receipt['config']:
        raise ValueError('Old dataset/receipt mismatch')
    current=dataset.sources()
    if current['base_release_sha256']!=old['sources']['base_release_sha256']:
        raise ValueError('Native release changed; runtime-only refresh forbidden')
    allowed={'runtime.py','objective.py','worker.py','protocol.py','refresh_runtime.py'}
    before=old['sources']['experiment_sources'];after=current['experiment_sources']
    changed={name for name in before.keys()|after.keys() if before.get(name)!=after.get(name)}
    if not changed<=allowed or not {'dataset.py','history.py','targets.py','planning.py','local_check.py'}<=before.keys():
        raise ValueError('Label/data/sampling source changed; runtime-only refresh forbidden')
    for protected in (source.parent,Path(old['data_root'])):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('Refresh output overlaps protected inputs')
    provenance.verify_sources(Path(old['data_root']),old['raw_proof'])
    if trial.file_sha(source)!=sha:raise ValueError('Old artifact changed during refresh')
    output.mkdir(parents=True,exist_ok=False)
    note=dict(parent_artifact=str(source),parent_sha256=sha,parent_sources=old['sources'],
              changed_runtime_sources=sorted(changed),records_sha256=baseline.digest(old['records']),
              hostname=socket.gethostname(),created_utc=datetime.now(timezone.utc).isoformat(),
              labels_recomputed=False)
    new=dict(old,sources=current,runtime_refresh=note)
    trial.write(output/'dataset.json',new)
    trial.write(output/'dataset_summary.json',dict(receipt,sources=current,runtime_refresh=note,
        artifact_sha256=trial.file_sha(output/'dataset.json')))
    dataset.load(output/'dataset.json')
    return note


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(refresh(a.dataset,a.output))


if __name__=='__main__':main()
