"""Complete, bounded audit ZIPs. Never truncate cases to satisfy the size cap."""
import copy
import json
from pathlib import Path
import random
import zlib
from .audit import sha
from .state import atomic
from ..abc_delivery import pack, LIMIT


def preflight_size(artifact,projection,admission):
    """Conservative sizing rehearsal, not a claim about arbitrary future logs.

    Retain the actual full records and budget incompressible prediction padding.
    Full logs/RGB/weights are outside the required archive contract.
    """
    from .experiment import guard_candidates,guard_records
    buckets={}
    for r in artifact['records']:guard_candidates(buckets,r)
    guards=guard_records(buckets)
    cohorts=[r for r in artifact['records'] if r['split']=='test']
    cohorts += [r for r in artifact['records'] if r['split']=='val']*3+guards*3
    rng=random.Random(20261024);compressor=zlib.compressobj(9);size=0
    for r in cohorts:
        view=dict(r,variant='six');view.pop('representations',None)
        size+=len(compressor.compress(json.dumps(view,separators=(',',':')).encode()))
        size+=len(compressor.compress(rng.randbytes(4096)))
    size+=len(compressor.flush())
    size+=len(zlib.compress(json.dumps(projection).encode(),9))+len(zlib.compress(json.dumps(admission).encode(),9))
    reserve=6_000_000
    if size+reserve>LIMIT:raise ValueError(f'audit_core_budget_insufficient_before_training: {size+reserve}>{LIMIT}')
    return dict(rehearsal_bytes=size,reserve_bytes=reserve,limit_bytes=LIMIT,
        estimated_bytes=size+reserve,case_appearances=len(cohorts),prediction_padding_per_case=4096,
        actual_archive_size_enforced=True,full_logs_RGB_and_weights_excluded=True)


def package(root,run,verification):
    root=Path(root);run=Path(run)
    required={}
    for name in ('ready_1280.json','pool_report.json','input_manifest.json','admission.json','plan.json'):
        required['preflight/'+name]=root/name
    for name in ('training_complete.json','validation_complete.json','plan.json','admission.json','startup.json','progress.json','progress.jsonl','test/cases.json','test/metrics.json'):
        required['run/'+name]=run/name
    for epoch in range(1,4):
        names=[f'epoch_{epoch}.json',f'epoch_{epoch}/receipt.json']
        names += [f'{prefix}_{epoch}_results/{file}' for prefix in ('validation','guard') for file in ('cases.json','metrics.json','receipt.json')]
        for name in names:
            required['run/'+name]=run/name
    proof=root/'completion_verification.json';atomic(proof,verification);required['completion_verification.json']=proof
    source=Path(__file__).parent
    for p in sorted(source.rglob('*')):
        if p.is_file() and p.suffix in ('.py','.json','.jsonl','.txt','.sh','.md') and 'tests' not in p.parts:
            required['source/'+str(p.relative_to(source))]=p
    # The name is content-addressed; repeated delivery never overwrites another run.
    from .identity import digest
    signature=digest({n:sha(p) for n,p in sorted(required.items())})
    output=root/'delivery'/('single_refine_audit_'+signature[:16]+'.zip')
    if output.exists():
        receipt=json.loads(output.with_suffix('.receipt.json').read_text())
        if sha(output)!=receipt['sha256'] or output.stat().st_size>LIMIT:raise ValueError('existing_audit_package_changed')
    else:
        receipt=pack(output,required,summary=dict(method='single_refine',complete_training_and_test=True,
            labels_unchanged=True,prompt_variant='clarified',independent_holdout=False),limit=LIMIT)
    atomic(root/'delivery.json',receipt)
    print(f'AUDIT_PACKAGE={output.resolve()}\nAUDIT_DIRECTORY={output.parent.resolve()}\nAUDIT_BYTES={receipt["bytes"]} (limit {LIMIT})',flush=True)
    return receipt
