"""Actual metrics/pack and native checkpoint IO; heavy training explicitly substituted."""
import copy
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from qwen3vl_local.audit_joint.abc_delivery import sha,write
from qwen3vl_local.audit_joint.phase3_abc_refine import __main__ as cli,bindings,regression,runtime,preflight
from qwen3vl_local.audit_joint.geometry_abc_v2 import metrics,objective,dataset,train
from qwen3vl_local.audit_joint.geometry_abc_v2.tests.test_study import record


def case(r,contract):
    return dict(record=r,prediction=dict(token_ids=[t['token_id'] for t in r['targets']],
        contract=contract,sample_binding=objective.observation_binding(r),format_valid=[True]*4))


def test_pure_improvement_removes_unsupported_false_positive():
    r=record();r['targets']=[dict(valid=True,token_id=4)]*4
    old=case(r,runtime.BASE_CONTRACT('a'));old['prediction']['token_ids']=[4,4,4,5]
    new=copy.deepcopy(old);new['prediction']['token_ids']=[4]*4
    out=regression.compare([old],[new])
    assert out['status']=='observed_cohort_protection_passed'
    assert len(out['new_correct'])==1 and not out['old_correct_losses'] and not out['metric_declines']


def test_net_improvement_never_cancels_old_correct_losses():
    r=record();r['targets']=[dict(valid=True,token_id=4)]*4
    old=case(r,runtime.BASE_CONTRACT('a'));old['prediction']['token_ids']=[4,5,5,5]
    new=copy.deepcopy(old);new['prediction']['token_ids']=[5,4,4,4]
    out=regression.compare([old],[new])
    assert out['status']=='tradeoff_review_required'
    assert len(out['old_correct_losses'])==1 and len(out['new_correct'])==3


def test_supported_class_decline_still_requires_review():
    r=record();r['targets']=[dict(valid=True,token_id=x) for x in [4,4,5,5]]
    old=case(r,runtime.BASE_CONTRACT('a'));new=copy.deepcopy(old);new['prediction']['token_ids']=[4]*4
    out=regression.compare([old],[new])
    assert out['status']=='tradeoff_review_required'
    assert any(d['category']=='5' and d['metric']=='recall' and d['after']==0 for d in out['metric_declines'])


def suite(tmp_path,monkeypatch):
    root=tmp_path/'run';root.mkdir();parent=tmp_path/'baseline.json';parent.write_text('baseline dataset substitute')
    model_dir=tmp_path/'model';model_dir.mkdir();(model_dir/'config.json').write_text('{"tiny":1}')
    (model_dir/'tokenizer.json').write_text('{"tokens":1}')
    model=bindings.model_identity(model_dir)
    for name in ('dataset.json','status.json','receipt.json','admission.json','members.jsonl'):
        write(root/'data'/name,{})
    state=dict(workflow=runtime.WORKFLOW,initialization=runtime.INITIALIZATION,groups={},baseline_test={},baseline_dataset=bindings.file_proof(parent),
               model_binding=dict(path=str(model_dir),model=model))
    processor=dict(code=runtime.identity(),model=model,dataset_sha256=sha(root/'data/dataset.json'),full_pool=True,no_truncation=True)
    write(root/'processor.json',processor)
    records={}
    for v in 'abc':
        val=record(8,variant=v);val['split']='val';test=record(9,variant=v);test['split']='test'
        records[v]=[record(i,variant=v) for i in range(10,14)]+[val,test]
        probe=root/'probes'/v;binding=dict(model=model,contract=runtime.contract(v),code=runtime.identity(),dataset_sha256=sha(root/'data/dataset.json'))
        write(probe/'checkpoint/receipt.json',dict(binding=binding,state_sha256='placeholder'))
        (probe/'checkpoint/state.pt').write_bytes(b'synthetic probe state')
        write(probe/'checkpoint/receipt.json',dict(binding=binding,state_sha256=sha(probe/'checkpoint/state.pt')))
        write(probe/'probe_complete.json',dict(binding=binding,**binding,variant=v,processor_sha256=sha(root/'processor.json'),
             checkpoint_reload_verified=True,checkpoint_sha256=sha(probe/'checkpoint/state.pt'),receipt_sha256=sha(probe/'checkpoint/receipt.json')))
        baseline=[case(test,runtime.BASE_CONTRACT(v))];new=[case(test,runtime.contract(v))]
        bp=tmp_path/'baseline_runs'/f'{v}_1280/test/cases.json';write(bp,baseline);state['baseline_test'][v]=bindings.file_proof(bp)
        write(root/f'eval_{v}/cases.json',new)
        with runtime.activated():write(root/f'eval_{v}/metrics.json',metrics.evaluate(new))
        report=regression.compare(baseline,new);write(root/f'regression_{v}.json',report);state['groups'][v]=report['status']
        write(root/f'train_{v}/training_complete.json',{'binding':{'model':model}})
        write(root/f'train_{v}/plan.json',{})
        (root/f'train_{v}/epoch_3').mkdir();(root/f'train_{v}/epoch_3/state.pt').write_bytes(b'synthetic final state')
    from qwen3vl_local.audit_joint.phase3_abc_refine import probe_inputs
    processor['extrema']={}
    for v,items in records.items():
        for r in items:probe_inputs.update(processor['extrema'],r,v,dict(tokens=100+r['frame'],inputs={'image_grid_thw':torch.tensor([[1,4,4]])}))
    write(root/'processor.json',processor)
    for v in 'abc':
        probe=root/'probes'/v;selected=probe_inputs.selection(records[v],processor,v)
        for rank in range(4):
            write(probe/f'rank{rank}.json',dict(backward=selected['backward'][rank],
                holdout_forward_only=[dict(x,updates=0) for x in selected['holdout_forward_only'][rank::4]]))
        path=probe/'probe_complete.json';value=json.loads(path.read_text())
        value.update(input_selection=selected,processor_sha256=sha(root/'processor.json'),
                     rank_receipts={f'rank{i}.json':sha(probe/f'rank{i}.json') for i in range(4)})
        write(path,value)
    write(root/'pipeline.json',state)
    monkeypatch.setattr(dataset,'load',lambda path,variant,**kw:dict(records=copy.deepcopy(records[variant])))
    monkeypatch.setattr(cli,'verify_training',lambda path,*a:json.loads((path/'training_complete.json').read_text()))
    return root,state,model,records


def test_actual_final_pack_new_contract_and_fresh_evidence(tmp_path,monkeypatch):
    root,state,model,records=suite(tmp_path,monkeypatch)
    result=cli.package(root)
    with zipfile.ZipFile(result['package']) as z:
        assert z.testzip() is None
        for v in 'abc':
            assert f'admission/{v}/paired_validation.json' not in z.namelist()
            compact=json.loads(z.read(f'{v}/cases.json'));assert compact['baseline_rows']==compact['rows']
            assert f'admission/{v}/probe_complete.json' in z.namelist()
            for rank in range(4):assert f'admission/{v}/rank{rank}.json' in z.namelist()
        assert 'admission/processor.json' in z.namelist()
    assert objective.messages is runtime.prompt.ORIGINAL_MESSAGES
    cases=json.loads((root/'eval_a/cases.json').read_text());cases[0]['prediction']['contract']=runtime.BASE_CONTRACT('a')
    write(root/'eval_a/cases.json',cases)
    with pytest.raises(ValueError,match='mixed_evaluation_contract'):cli.package(root)


@pytest.mark.parametrize('change',['holdout_update','holdout_in_backward','holdout_length'])
def test_rehashed_probe_rank_cannot_hide_holdout_update_or_member_change(tmp_path,monkeypatch,change):
    root,state,model,records=suite(tmp_path,monkeypatch)
    p=root/'probes/a/rank0.json';proof=json.loads(p.read_text())
    if change=='holdout_update':proof['holdout_forward_only'][0]['updates']=1
    elif change=='holdout_in_backward':proof['backward']['split']='test'
    else:proof['holdout_forward_only'][0]['tokens']+=1
    write(p,proof)
    receipt_path=p.with_name('probe_complete.json');receipt=json.loads(receipt_path.read_text())
    receipt['rank_receipts'][p.name]=sha(p);write(receipt_path,receipt)
    with pytest.raises(ValueError):bindings.verify_probe(receipt_path,root/'data/dataset.json',model,'a',root/'processor.json')


@pytest.mark.parametrize('changed',['model','tokenizer','probe_variant','probe_weights','baseline_cases','processor'])
def test_cached_admissions_reject_changed_inputs(tmp_path,monkeypatch,changed):
    root,state,model,records=suite(tmp_path,monkeypatch)
    if changed in ('model','tokenizer'):
        name='config.json' if changed=='model' else 'tokenizer.json'
        (Path(state['model_binding']['path'])/name).write_text('changed')
    elif changed=='probe_variant':
        (root/'probes/a/probe_complete.json').write_bytes((root/'probes/b/probe_complete.json').read_bytes())
    elif changed=='probe_weights':(root/'probes/a/checkpoint/state.pt').write_bytes(b'changed weights')
    elif changed=='baseline_cases':Path(state['baseline_test']['a']['path']).write_text('[]')
    elif changed=='processor':
        p=root/'processor.json';r=json.loads(p.read_text());r['model']={'wrong':'model'};write(p,r)
    with pytest.raises(ValueError):cli.admission_evidence(root,state)


def test_changed_model_stops_workflow_before_GPU_or_training(tmp_path,monkeypatch):
    from qwen3vl_local.action_prior import launch
    root,state,model,records=suite(tmp_path,monkeypatch)
    (Path(state['model_binding']['path'])/'tokenizer.json').write_text('changed')
    monkeypatch.setattr(launch,'ensure_gpu',lambda *a:pytest.fail('GPU selection must not run'))
    monkeypatch.setattr(launch,'run_logged',lambda *a:pytest.fail('training must not run'))
    args=SimpleNamespace(operation='all',run_dir=str(root),output=str(root),model_dir=state['model_binding']['path'],
        baseline_runs_root=str(tmp_path/'baseline_runs'),baseline_dataset=state['baseline_dataset']['path'])
    with pytest.raises(ValueError,match='frozen_suite_inputs_changed'):cli.workflow(args)


def test_native_probe_save_restore_checks_all_trainable_weights(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    m=torch.nn.Linear(3,2);bundle=SimpleNamespace(model=m);opt=torch.optim.AdamW(m.parameters());scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    dataset_path=tmp_path/'dataset.json';write(dataset_path,{})
    binding=dict(contract=runtime.contract('a'),code=runtime.identity(),dataset_sha256=sha(dataset_path),model={'tiny':'fixture'})
    artifact=dict(contract=binding['contract'],code=binding['code'])
    m(torch.ones(1,3)).sum().backward();opt.step()
    train.save_checkpoint(tmp_path/'checkpoint',bundle,opt,scheduler,binding=binding,epoch=0,step=1,exposure={'x':1},status='probe_only')
    expected={n:p.detach().clone() for n,p in m.named_parameters()}
    with torch.no_grad():
        for p in m.parameters():p.zero_()
    preflight.reload_probe(bundle,tmp_path/'checkpoint',artifact,dataset_path)
    assert all(torch.equal(p,expected[n]) for n,p in m.named_parameters())
    (tmp_path/'checkpoint/state.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checkpoint_changed'):preflight.reload_probe(bundle,tmp_path/'checkpoint',artifact,dataset_path)


@pytest.mark.parametrize('gate',['processor','probe'])
def test_every_cached_gate_independently_binds_model_content(tmp_path,monkeypatch,gate):
    root,state,model,records=suite(tmp_path,monkeypatch)
    wrong=dict(model);wrong['tokenizer.json']='changed tokenizer hash'
    with pytest.raises(ValueError):
        if gate=='processor':bindings.verify_processor(root/'processor.json',root/'data/dataset.json',wrong)
        else:bindings.verify_probe(root/'probes/a/probe_complete.json',root/'data/dataset.json',wrong,'a',root/'processor.json')
