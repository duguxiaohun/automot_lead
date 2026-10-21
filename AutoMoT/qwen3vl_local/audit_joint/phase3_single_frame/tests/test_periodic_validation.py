"""Exercise actual native validation collectives, including an empty local shard."""
import copy
from datetime import timedelta
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from qwen3vl_local.audit_joint import phase3_base_trial as trial
from qwen3vl_local.audit_joint.phase3_base_dense import evaluation
from qwen3vl_local.audit_joint.phase3_base_dense.data import Pool
from qwen3vl_local.audit_joint.phase3_single_frame import dataset, objective, runtime, targets, worker
from qwen3vl_local.audit_joint.phase3_single_frame.tests.test_single_frame import fixture
from qwen3vl_local.sft_new_loop_phase3 import train


def _rank(rank, rendezvous, output, job, artifact, reference, admitted):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://'+rendezvous,
        rank=rank, world_size=4, timeout=timedelta(seconds=40))
    try:
        trial.reference=lambda:reference
        train._load_images=lambda paths:[object()]
        # Fast deterministic objective; separate native-loop tests use the real
        # tokenizer/packing/loss/backward and compare resumed optimizer states.
        objective.pack=lambda *a,**kw:{}
        def loss(bundle, packed):
            rec=packed['record'];n=sum(w['valid'] for w in rec['windows'])
            return torch.tensor(rec['test_loss']),dict(supervised=n,correct=n,tokens=100,
                target_entropy=0.,kl_to_training_target=0.)
        objective.loss=loss
        model=torch.nn.Linear(1,1)
        model=torch.nn.parallel.DistributedDataParallel(model)
        bundle=SimpleNamespace(model=model,tokenizer=None,device=torch.device('cpu'))
        before=copy.deepcopy(model.state_dict())
        metrics=worker.probe_periodic_loss(job,artifact,admitted,bundle,
            rank=rank,world=4,device=torch.device('cpu'))
        assert metrics['requested_samples']==10 and metrics['samples']==4
        assert metrics['unlabeled_samples']==6 and metrics['supervised_windows']==14
        assert metrics['skipped']==0 and metrics['loss']==pytest.approx(4.)
        assert model.training
        for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        assert all(p.grad is None for p in model.parameters())
        rows=train._read_rows(Path(job['index']),'val',data_root=Path(job['data_root']))
        work=evaluation.selected(Pool(rows),'val','loss')
        shard=train._split_work_for_rank(work,rank=rank,world_size=4)
        local,audit=runtime.loss_members(shard,artifact['records'])
        if rank==0:assert not local and audit['requested']==3
        cfg=runtime.make_args(job,artifact)
        kwargs=dict(history_rgb_mode=cfg.history_rgb_mode,prompt_variant=cfg.prompt_variant,
            max_length=cfg.max_length,format_loss_weight=cfg.format_loss_weight,
            device=torch.device('cpu'),world_size=4,batch_size=1)
        saved={key:rec['windows'] for key,rec in artifact['records'].items()}
        for rec in artifact['records'].values():rec['windows']=[targets.distribution([None,None])]*4
        with pytest.raises(ValueError,match='no supervised anchors globally'):
            train.evaluate_loss(bundle,shard,**kwargs)
        for key,rec in artifact['records'].items():rec['windows']=saved[key]
        # Genuine input rejection must still fail on EVERY rank, including the
        # rank without any supervised item; do not disable the native guard.
        train._build_inputs=lambda *a,**kw:None
        with pytest.raises(ValueError,match='max_length/packing'):
            train.evaluate_loss(bundle,shard,**kwargs)
        trial.write(Path(output)/f'rank_{rank}.json',dict(metrics=metrics,
            local_eligible=len(local),all_missing_rejected=True,packing_failure_rejected=True))
    finally:torch.distributed.destroy_process_group()


def test_four_rank_periodic_probe_and_missing_label_guards(tmp_path,monkeypatch):
    job,rows,artifact,_=fixture(tmp_path,monkeypatch)
    job['dataset']=str(tmp_path/'single/dataset.json')
    work=evaluation.selected(Pool(rows),'val','loss')
    for i,item in enumerate(work):
        rec=artifact['records'][dataset.identity(item.row)]
        rec['test_loss']=float(i+1)
        if i not in (1,2,3,6):rec['windows']=[targets.distribution([None,None])]*4
        elif i==6:rec['windows'][2:]=[targets.distribution([None,None])]*2
    _,audit=runtime.loss_members(work,artifact['records'])
    admitted=dict(plans=dict(fixed_loss_ids=[dataset.identity(x.row) for x in work],fixed_loss_admission=audit))
    context=mp.spawn(_rank,args=(str(tmp_path/'rendezvous'),str(tmp_path),job,artifact,
        trial.reference(),admitted),nprocs=4,join=False)
    deadline=time.monotonic()+90
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline:pytest.fail('Four-rank periodic validation timed out')
    finally:
        for process in context.processes:
            if process.is_alive():process.terminate()
        for process in context.processes:process.join(timeout=5)
    reports=[trial.read(tmp_path/f'rank_{r}.json') for r in range(4)]
    assert [r['local_eligible'] for r in reports]==[0,1,2,1]
    assert all(r['metrics']==reports[0]['metrics'] for r in reports)
