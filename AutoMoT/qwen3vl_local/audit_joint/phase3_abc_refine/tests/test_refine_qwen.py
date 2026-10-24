from types import SimpleNamespace
import copy
import pytest
import torch
from PIL import Image
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.audit_joint.geometry_abc_v2.tests.test_study import record
from qwen3vl_local.audit_joint.geometry_abc_v2 import objective,regions,dataset,train
from qwen3vl_local.audit_joint.phase3_abc_refine import prompt,runtime


@pytest.mark.parametrize('variant',list('abc'))
def test_candidate_real_processor_backward_and_restore_context(processor,model,tmp_path,variant):
    old=(objective.messages,regions.contract,dataset.code_identity)
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    r=record(variant=variant);image=Image.new('RGB',(32,32));original=objective.pack(bundle,r,image)
    with runtime.activated():
        packed=objective.pack(bundle,r,image)
        assert (torch.equal(original['inputs']['input_ids'],packed['inputs']['input_ids']))==(variant!='a')
        if variant=='a':
            text=prompt.messages(r,None)[1]['content'][-1]['text']
            assert text.index('The radii below')<text.index('[ACTION_QUERY_0]')
            model.eval()
            with torch.no_grad():
                before=objective.readout(bundle,original);after=objective.readout(bundle,packed)
            assert all(not torch.allclose(before[i],after[i]) for i in range(4))
        changed=copy.deepcopy(r);changed['targets']='not_an_observation'
        assert torch.equal(packed['inputs']['input_ids'],objective.pack(bundle,changed,image)['inputs']['input_ids'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        result=train.train_epoch(bundle,[r],optimizer,scheduler,lambda _:packed)
        assert result['steps']==1
        assert model.lm_head.weight.grad is not None
    assert old==(objective.messages,regions.contract,dataset.code_identity)


def test_worker_adapter_restores_after_failure():
    old=(objective.messages,regions.contract,dataset.code_identity)
    with pytest.raises(RuntimeError,match='injected'):
        with runtime.activated():raise RuntimeError('injected')
    assert old==(objective.messages,regions.contract,dataset.code_identity)


def test_streaming_preserves_complete_records_and_native_plan(tmp_path):
    import json
    from qwen3vl_local.audit_joint.phase3_abc_refine import streaming
    from qwen3vl_local.audit_joint.geometry_abc_v2 import planning,admission
    rows=[dataset.combine(record(frame=i,event=e,route=f'Town00_R{i}')) for e in ('LEAD_BRAKE','SIGNAL_FAILURE') for i in range(8,24)]
    obj={'before':{'float':1/3,'unicode':'车'},'records':rows,'after':[1,2,{'records':'nested'}]}
    p=tmp_path/'input.json';p.write_text(json.dumps(obj))
    assert streaming.header(p)=={k:v for k,v in obj.items() if k!='records'}
    assert list(streaming.records(p))==rows
    compact=[streaming.planning_record(r) for r in rows]
    assert planning.project(compact,per_event=4,world_size=4)==planning.project(rows,per_event=4,world_size=4)
    assert admission.inspect({'records':compact})==admission.inspect({'records':rows})


def test_failed_evaluation_retry_preserves_old_evidence(tmp_path):
    from qwen3vl_local.audit_joint.phase3_abc_refine.__main__ import preserve_partial_evaluation
    from pathlib import Path
    out=tmp_path/'eval_a';out.mkdir();(out/'partial.txt').write_text('failed attempt')
    archived=preserve_partial_evaluation(tmp_path,out)
    assert not out.exists() and (Path(archived)/'partial.txt').read_text()=='failed attempt'
    out.mkdir();(out/'metrics.json').write_text('{}')
    with pytest.raises(ValueError,match='completed'):preserve_partial_evaluation(tmp_path,out)
    assert (out/'metrics.json').read_text()=='{}'
