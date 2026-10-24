"""Restartable per-checkpoint validation and train guards after fixed-budget training."""
import json
from pathlib import Path
from PIL import Image
import torch
from . import dataset,metrics,objective,train
from ..state import atomic
from ..experiment import guard_candidates,guard_records
from ..audit import sha


def verify(folder,records,checkpoint):
    folder=Path(folder);receipt=json.loads((folder/'receipt.json').read_text())
    if (receipt['checkpoint_receipt_sha256']!=sha(checkpoint/'receipt.json')
        or receipt['cases_sha256']!=sha(folder/'cases.json') or receipt['metrics_sha256']!=sha(folder/'metrics.json')):
        raise ValueError('validation_receipt_changed')
    cases=json.loads((folder/'cases.json').read_text())
    if [c['record'] for c in cases]!=records:raise ValueError('validation_membership_mismatch')
    result=metrics.evaluate(cases)
    if result!=json.loads((folder/'metrics.json').read_text()):raise ValueError('validation_metrics_mismatch')
    return result


def run(args):
    from qwen3vl_local.sft_new_loop_phase3 import train as native
    from qwen3vl_local.audit_joint.phase3_suite import model_snapshot
    artifact=dataset.load(args.dataset,variant='six')
    validation=train.training_members(artifact);buckets={}
    for r in artifact['records']:guard_candidates(buckets,r)
    guards=guard_records(buckets);run=Path(args.output)
    identity={k:v['sha256'] for k,v in model_snapshot(args.model_dir).items()}
    bundle=None;summary={}
    for epoch in range(1,4):
        checkpoint=run/f'epoch_{epoch}';receipt=json.loads((checkpoint/'receipt.json').read_text())
        if receipt['binding']['model']!=identity:raise ValueError('base_model_mismatch')
        for name,records in (('validation',validation),('guard',guards)):
            folder=run/f'{name}_{epoch}_results'
            if (folder/'receipt.json').is_file():
                result=verify(folder,records,checkpoint)
            else:
                if folder.exists():
                    import time
                    folder.rename(folder.with_name(folder.name+'_failed_'+str(time.time_ns())))
                folder.mkdir()
                if bundle is None:
                    bundle=native.load_model_with_lora(Path(args.model_dir),device=torch.device('cuda:0'),lora_rank=16,
                        lora_alpha=32,lora_dropout=.05,lora_vision_scope='off',strict_vision_scope=True,gradient_checkpointing=False)
                train.restore(bundle,checkpoint,artifact=artifact,dataset_path=args.dataset)
                cases=[]
                for n,r in enumerate(records,1):
                    with Image.open(Path(artifact['data_root'])/r['image']) as im:
                        prediction=objective.predict(bundle,r,im.convert('RGB'))
                    cases.append(dict(record=r,prediction=prediction))
                    if n%100==0:print(f'[single {name} epoch={epoch}] {n}/{len(records)}',flush=True)
                result=metrics.evaluate(cases)
                atomic(folder/'cases.json',cases);atomic(folder/'metrics.json',result)
                atomic(folder/'receipt.json',dict(checkpoint_receipt_sha256=sha(checkpoint/'receipt.json'),
                    cases_sha256=sha(folder/'cases.json'),metrics_sha256=sha(folder/'metrics.json'),members=len(cases),
                    scope='validation' if name=='validation' else 'exposed_train_guard_not_holdout'))
                verify(folder,records,checkpoint)
            summary[f'{name}_{epoch}']=dict(macro_recall=result['macro_recall'],members=len(records))
    atomic(run/'validation_complete.json',dict(results=summary,selection='final_completed_epoch',checkpoint='epoch_3'))
