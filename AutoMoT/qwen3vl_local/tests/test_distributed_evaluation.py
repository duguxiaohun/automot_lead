"""Real four-process Gloo tests: ordered whole batches, tails and failures."""
from pathlib import Path
import json
import os
from datetime import timedelta
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank,folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'init'),rank=rank,world_size=4,
                            timeout=timedelta(seconds=30))
    from qwen3vl_local.batch_runtime import distributed_batches
    reports=[]
    for n in (0,1,3,8,13,16):
        consumed=[]
        def generate(rows):
            consumed.extend(rows)
            assert len(rows)<=2 and rows==list(range(rows[0],rows[0]+len(rows)))
            return [i*i for i in rows]
        values,audit=distributed_batches(list(range(n)),2,generate,identities=list(map(str,range(n))))
        assert values==[i*i for i in range(n)]
        reports.append(dict(n=n,consumed=consumed,audit=audit))
    def failure(rows):
        if rank==2:raise ValueError('intentional worker failure')
        return rows
    with pytest.raises(RuntimeError,match='rank 2.*intentional'):
        distributed_batches(list(range(8)),2,failure,identities=list(map(str,range(8))))
    ids=list(map(str,range(8)))
    if rank==1:ids.reverse()
    with pytest.raises(ValueError,match='differs across ranks'):
        distributed_batches(list(range(8)),2,lambda x:x,identities=ids)
    ids=list(map(str,range(8)))
    if rank==1:ids[-1]=ids[0]
    with pytest.raises(ValueError,match='unique and complete'):
        distributed_batches(list(range(8)),2,lambda x:x,identities=ids)
    # Actual Phase4 evaluation aggregator: all ranks call generate_batch and
    # only their own complete batches; reports retain the global input order.
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import evaluate,data
    from qwen3vl_local.sft_new_loop_phase4.tests.test_short_rgb import source_row
    fixture=Path(folder)/f'fixture{rank}';fixture.mkdir()
    row=data.short_row(source_row(fixture))
    rows=[dict(row,id=f'case{i}') for i in range(13)]
    used=[]
    def predict(bundle,selected,root,length,**kw):
        used.extend(r['id'] for r in selected)
        return [(r['target'],r['target']) for r in selected]
    evaluate.generate_batch=predict
    metrics,cases=evaluate.evaluate_rows(None,rows,fixture,batch_size=2,reuse_rgb=True)
    assert [c['id'] for c in cases]==[r['id'] for r in rows]
    assert metrics['distributed_execution']['per_rank_counts']==[4,4,3,2]
    # Actual Phase3 generation probe: distributed batches, one rank writes cases.
    from types import SimpleNamespace
    from qwen3vl_local.sft_new_loop_phase3 import train
    from qwen3vl_local.sft_new_loop_phase3.test_validation_balance import candidate_rows
    import qwen3vl_local.batch_runtime as runtime
    items=[train._make_item(r,seed=1,action_output_mode='binary') for r in candidate_rows()[:13]]
    train._load_images=lambda paths:[]
    train.build_action_messages=lambda **kw:kw['spec']
    runtime.encode_messages=lambda bundle,messages:messages
    p3used=[]
    def greedy(bundle,specs,**kw):
        p3used.extend(specs)
        return [train.build_action_target(spec) for spec in specs]
    runtime.greedy_batch=greedy
    model=torch.nn.Linear(1,1)
    bundle=SimpleNamespace(model=model,unwrap=lambda:model,tokenizer=None,processor=None,device='cpu')
    result=train.evaluate_generation_probe(bundle,items,history_rgb_mode='4rgb',max_new_tokens=8,
        batch_size=2,reuse_rgb=True,record_path=Path(folder)/'phase3_cases.jsonl')
    assert result['exact_accuracy']==1 and model.training
    (Path(folder)/f'rank{rank}.json').write_text(json.dumps(dict(reports=reports,p4used=used,p3count=len(p3used),p3total=result['samples'])))
    # Run the real Phase4 training loop with four CPU DDP workers and two
    # epochs; every rank must enter validation, not merely its metric helper.
    import sys
    from types import SimpleNamespace
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import train as p4train
    from qwen3vl_local.qwen35 import adapters
    from qwen3vl_local.sft_new_loop_phase4.tests.test_production_selection_v50 import pool,as_cases
    fixture_pool=pool()
    manifest=dict(training_admission={'trainable':True,'evaluation_scope':'test'},
                  coverage={'ready':False},student_view={'selected_ids':{}},observation_contract={})
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.5))
        def forward(self,**kw):return SimpleNamespace(loss=(self.weight-1).square())
    class Processor:
        def save_pretrained(self,path):(path/'processor_config.json').write_text('{}')
    p4train.load_dataset=lambda *a,**kw:(fixture_pool,manifest)
    p4train.plan=lambda *a,**kw:(list(range(16)),dict(next_history={},unique_questions=16))
    p4train.load_bundle=lambda *a,**kw:SimpleNamespace(model=Model(),processor=Processor())
    p4train.encode_batch=lambda b,rows,*a,**kw:{'input_ids':torch.ones(len(rows),3,dtype=torch.long)}
    validations=[]
    def validate_training(bundle,rows,root,max_length,**kw):
        assert isinstance(bundle.model,torch.nn.parallel.DistributedDataParallel)
        cases,audit=distributed_batches(rows,kw['batch_size'],lambda selected:as_cases(selected,.5),
                                       identities=[r['id'] for r in rows])
        validations.append(audit)
        return dict(selection_agreement=.5,distributed_execution=audit),cases
    p4train.evaluate_rows=validate_training
    def save(model,path):
        path.mkdir();torch.save(model.state_dict(),path/'adapter_model.bin')
    adapters.save_adapter=save
    torch.cuda.is_available=lambda:False
    dist.init_process_group=lambda *a,**kw:None  # Already initialized above.
    os.environ.update(RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='4')
    sys.argv=['train','--dataset',folder,'--data-root',folder,'--output-dir',str(Path(folder)/'training'),
              '--epochs','2','--batch-size','2','--accumulation','2','--sampling-policy','legacy_ring']
    p4train.main()
    assert len(validations)==2 and all(v['world_size']==4 and all(v['per_rank_counts']) for v in validations)
    (Path(folder)/f'training_rank{rank}.json').write_text(json.dumps(validations))


def test_four_process_evaluation_exactly_once_and_failure_propagation(tmp_path):
    context=mp.spawn(worker,args=(str(tmp_path),),nprocs=4,join=False)
    import time
    deadline=time.monotonic()+60
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline:raise TimeoutError('four-process CPU check exceeded 60 seconds')
    finally:
        for p in context.processes:
            if p.is_alive():p.kill()
        for p in context.processes:p.join(timeout=5)
    ranks=[json.loads((tmp_path/f'rank{i}.json').read_text()) for i in range(4)]
    for index,n in enumerate((0,1,3,8,13,16)):
        assert sorted(i for r in ranks for i in r['reports'][index]['consumed'])==list(range(n))
    assert sum(r['p3count'] for r in ranks)==ranks[0]['p3total']
    assert len((tmp_path/'phase3_cases.jsonl').read_text().splitlines())==ranks[0]['p3total']
    for rank in range(4):
        assert len(json.loads((tmp_path/f'training_rank{rank}.json').read_text()))==2
    for epoch in (0,1):
        report=json.loads((tmp_path/'training'/f'epoch_{epoch:03d}_validation.json').read_text())
        assert report['distributed_execution']['world_size']==4
        steps=[json.loads(line) for line in (tmp_path/'training'/f'epoch_{epoch:03d}_rank3_steps.jsonl').read_text().splitlines()]
        assert len(steps)==2 and steps[-1]['optimizer_steps']==1


@pytest.mark.parametrize('world',['2','3','8'])
def test_cuda_entry_rejects_partial_world(monkeypatch,world):
    from qwen3vl_local.batch_runtime import launch_four_evaluation
    monkeypatch.setenv('WORLD_SIZE',world)
    with pytest.raises(ValueError,match='four ranks'):launch_four_evaluation('unused','cuda')


def test_cuda_entry_refuses_insufficient_visible_cards(monkeypatch):
    from qwen3vl_local.batch_runtime import launch_four_evaluation
    from qwen3vl_local.sft_new_loop_phase4 import devices
    monkeypatch.delenv('WORLD_SIZE',raising=False);monkeypatch.delenv('LOCAL_RANK',raising=False)
    monkeypatch.setattr(devices,'select_gpus',lambda n:'7')
    with pytest.raises(ValueError,match='four visible'):launch_four_evaluation('unused','cuda')
