"""Recover a complete replay's index without computing or replacing any route.

Only complete native artifacts with the current producer identity are accepted.
Missing/corrupt files stop recovery; this module never calls route_records/produce.
"""
import json
import os
from pathlib import Path
import tempfile

from .io import digest


def recover(pool, production):
    from qwen3vl_local.sft_new_loop_phase4 import candidate_pool, teacher_replay as replay
    from qwen3vl_local.sft_new_loop_phase4.production_storage import production_lock, publication
    candidate_pool.validate(pool)
    production=Path(production)
    if not production.is_dir():
        raise FileNotFoundError('Existing production directory missing on this machine: '+str(production))
    with production_lock(production):
        index=production/'index.json'
        publication(index)  # Never repair a changed/missing published index.
        if index.exists() or index.is_symlink():
            print(json.dumps(dict(stage='validate_existing_index',total_routes=len(pool['routes']),recomputed_routes=0)),flush=True)
            report=replay.accounting(pool,index)
            if not report['complete_causal_production']:
                raise ValueError('Existing index is partial; retained unchanged, no replay started')
            return report | dict(recovery='validated_existing_index', recomputed_routes=0)
        if json.loads((production/'storage.json').read_text()) != dict(compression='gzip'):
            raise ValueError('Recovery requires the original gzip storage contract')
        names=[digest([r['scenario'],r['route_id']])+'.jsonl.gz' for r in pool['routes']]
        missing=[n for n in names if not (production/n).is_file()]
        if missing:
            raise ValueError(f'{len(missing)} complete route artifacts missing; no replay started; first: {missing[:5]}')
        header=dict(policy=replay.POLICY,teacher=replay.rules.identity(),candidate_pool_sha256=pool['sha256'])
        receipts=[]
        for route,name in zip(pool['routes'],names):
            receipts.append(replay.inspect_route(production/name,route,header))
            if len(receipts)%100==0 or len(receipts)==len(names):
                print(json.dumps(dict(stage='validate_existing_replay',validated_routes=len(receipts),
                                      total_routes=len(names),recomputed_routes=0)),flush=True)
        record=dict(**header,routes=receipts)
        record['sha256']=digest(record)
        # Unique temp preserves any interrupted index.json.pending as evidence.
        fd,tmp=tempfile.mkstemp(prefix='.recovered-index-',suffix='.json',dir=production)
        try:
            with os.fdopen(fd,'w') as stream:
                json.dump(record,stream,ensure_ascii=False,indent=2);stream.write('\n')
            # Native accounting verifies every receipt again before publication.
            report=replay.accounting(pool,Path(tmp))
            if not report['complete_causal_production']:
                raise ValueError('Recovered replay is incomplete')
            os.link(tmp,index)  # Exclusive installation: never overwrite an index.
        finally:
            Path(tmp).unlink(missing_ok=True)
        return report | dict(recovery='recovered_complete_index',recomputed_routes=0)
