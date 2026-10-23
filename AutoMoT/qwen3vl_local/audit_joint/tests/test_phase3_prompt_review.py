import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from qwen3vl_local.audit_joint import phase3_prompt_review as review
from qwen3vl_local.audit_joint import phase3_matched_image_eval as evaluation
from qwen3vl_local.audit_joint import phase3_matched_image_launch as launch
from qwen3vl_local.audit_joint.phase3_single_frame import objective,targets,history
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_processor import processor


def record(key='a'):
    return dict(id=key,event='LEAD_BRAKE',image='current.png',
                history=['STOP']*4,history_original=['STOP']*4,history_imputed=[False]*4,
                history_changed=[False]*4,history_semantics=history.SEMANTICS['offline_native'],
                premise='Current measured speed: 0 m/s. Allowed choices A B C F.',
                windows=[targets.distribution(['STOP','STOP']) for _ in range(4)])


def case(key='a'):
    return dict(record(key),predictions=['STOP']*4,history_input=['STOP']*4,image_mode='actual')


def test_scope_changes_only_system_and_restores_on_exception():
    r=record();image=object();before=objective.messages(r,image);bound=review.contract(objective)
    original=objective.messages
    with pytest.raises(RuntimeError):
        with review.prompt_scope(objective):
            changed=objective.messages(r,image)
            assert changed[1:]==before[1:]
            assert changed[1]['content'][0]['image'] is image
            assert changed[0]['content']==review.CLARIFICATION+before[0]['content']
            raise RuntimeError('packing failed')
    assert objective.messages is original and review.contract(objective)==bound


def test_native_qwen_prompt_forward_preserves_parameters_records_and_token_ids(tmp_path,processor,model):
    r=record();path=tmp_path/'a.png';Image.new('RGB',(32,32),(30,80,120)).save(path);r['image']=str(path)
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    state=copy.deepcopy(model.state_dict());before=copy.deepcopy(r);contract=review.contract(objective)
    old=objective.pack(bundle,r,Image.open(path))
    with review.prompt_scope(objective):
        packed=objective.pack(bundle,r,Image.open(path))
    assert old['choices']==packed['choices'] and len(packed['positions'])==4
    assert packed['images']==old['images']==1
    assert all(s['same_action_token'] for s in packed['history_token_check']['slots'])
    assert packed['tokens']>old['tokens']
    model.train()
    result=review.predict(bundle,{'a':r},{'a'})
    assert len(result)==1 and len(result[0]['predictions'])==4
    assert model.training and r==before and review.contract(objective)==contract
    for k,v in model.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('tamper',['id','windows','history_input','image_mode'])
def test_prediction_integrity(tamper):
    def fake(bundle,work):
        c=case();c[tamper]={'id':'other','windows':[],'history_input':['KEEP']*4,'image_mode':'blank'}[tamper]
        return [c],{}
    original=objective.messages
    with pytest.raises(ValueError):review.predict(None,{'a':record()},{'a'},predictor=fake)
    assert objective.messages is original


def test_prompt_merge_and_denominators(tmp_path):
    ids=[str(i) for i in range(9)]
    plans=[dict(pairs=[dict(receiver=i,donor=ids[(n+1)%9]) for n,i in enumerate(ids)],
                metadata=[dict(id=i,event='LEAD_BRAKE',group=i) for i in ids])]
    shards=[dict(rank=r,world=4,variant=review.VARIANT,
                 cases=[case(i) for i in sorted(evaluation.receiver_shard(plans,r,4))]) for r in range(4)]
    merged=review.merge(plans,shards)
    (tmp_path/'single_frame_val_cases.json').write_text(json.dumps([case(i) for i in ids]))
    merged['0']['predictions'][0]='KEEP'
    result=review.save(dict(output=str(tmp_path)),plans,tmp_path,merged,review.contract(objective))
    assert result['primary_common']['subsets']['all']['windows']==36
    assert result['primary_common']['subsets']['all']['reference_only']==1
    assert result['predictions_changed']==1
    bad=copy.deepcopy(shards);bad[0]['cases'].append(case('0'))
    with pytest.raises(ValueError):review.merge(plans,bad)
    with pytest.raises(ValueError):review.merge(plans,shards[:-1])


def test_launcher_prompt_flag_preflight_workers_and_merge(tmp_path,monkeypatch):
    # Exercise the actual launcher and scoring, substituting only expensive workers.
    experiment=tmp_path/'original';experiment.mkdir()
    for d in ('data','model','dataset','index'):(tmp_path/d).mkdir()
    job=dict(output=str(experiment),dataset=str(tmp_path/'dataset/dataset.json'),
             data_root=str(tmp_path/'data'),model_dir=str(tmp_path/'model'),index=str(tmp_path/'index/index.jsonl'))
    jobpath=tmp_path/'job.json';jobpath.write_text(json.dumps(job))
    plans=[json.loads((launch.APP/f'qwen3vl_local/audit_joint/phase3_behavior_diagnostic_v1/matched_images_seed_{s}.json').read_text()) for s in (11,29,47)]
    rows=[dict(case(r['id']),event=r['event']) for r in plans[0]['metadata']]
    byid={c['id']:c for c in rows}
    saved=experiment/'single_frame_val_cases.json';saved.write_text(json.dumps(rows))
    common=set.intersection(*({p['receiver'] for p in plan['pairs']} for plan in plans))
    note=dict(job_sha256='job',checkpoint_receipt_sha256='checkpoint',plans=[],source_sha256='source',
        primary_common_receiver_ids=sorted(common),prompt_intervention=review.contract(objective),
        reference_cases_sha256=launch.diag.file_sha(saved),prompt_review_source_sha256=launch.diag.file_sha(review.__file__))
    def snapshot(dest,job):
        target=dest/'qwen3vl_local/audit_joint/phase3_prompt_review.py';target.parent.mkdir(parents=True)
        target.write_bytes(Path(review.__file__).read_bytes());return dest,{}
    calls=[]
    def processes(commands,cwd,envs,logs):
        calls.append(commands)
        for command,env,log in zip(commands,envs,logs):
            assert '--prompt-review' in command
            output=Path(command[command.index('--output')+1]);output.mkdir()
            pre='--preflight-only' in command
            r=dict(note,inference_performed=not pre)
            if not pre:
                rank=int(command[command.index('--rank')+1]);world=int(command[command.index('--world')+1])
                ids=evaluation.receiver_shard(plans,rank,world)
                shard=dict(rank=rank,world=world,variant=review.VARIANT,cases=[byid[i] for i in sorted(ids)])
                (output/'shard.json').write_text(json.dumps(shard));r['unique_prompt_forwards']=len(ids)
                assert env['CUDA_VISIBLE_DEVICES']==str(rank)
            (output/'receipt.json').write_text(json.dumps(r));log.write_text('test worker')
    monkeypatch.setattr(launch,'snapshot',snapshot);monkeypatch.setattr(launch,'processes',processes)
    result=launch.launch(jobpath,tmp_path/'result','0,1,2,3',prompt_review=True)
    assert [len(x) for x in calls]==[1,4]
    assert result['unique_prompt_forwards']==268
    assert result['unique_actual_forwards']==result['unique_swapped_forwards']==0
    metrics=json.loads((tmp_path/'result/metrics.json').read_text())
    assert metrics['primary_common']['anchors']==260 and metrics['predictions_changed']==0


def test_transition_pairs_mixed_missing_history_and_route_denominators():
    a=case('stop');a['windows']=[targets.distribution(['RESUME','RESUME']),
        targets.distribution(['KEEP','KEEP']),targets.distribution(['KEEP','DECELERATE']),
        targets.distribution([None,None])]
    a['predictions']=['STOP']*4
    b=copy.deepcopy(a);b['predictions']=['RESUME','KEEP','DECELERATE','KEEP']
    right=case('right');right['history']=['LANE_CHANGE_RIGHT']*3+['UNKNOWN']
    right['windows']=[targets.distribution(['KEEP','KEEP'])]*4;right['predictions']=['LANE_CHANGE_RIGHT']*4
    newright=copy.deepcopy(right);newright['predictions']=['KEEP']*4
    unknown=case('unknown');unknown['history']=['UNKNOWN']*4
    metadata={c['id']:dict(group='shared_route') for c in (a,right,unknown)}
    result=review.transition_scores([a,right,unknown],[b,newright,copy.deepcopy(unknown)],metadata)
    stop=result['history_to_future']['STOP->RESUME']
    assert stop['windows']==stop['anchors']==stop['physical_groups']==1
    assert stop['alternative_only']==1 and stop['delta']==-1
    mixed=result['history_to_future']['STOP->{DECELERATE,KEEP}']
    assert mixed['windows']==1 and mixed['alternative_hits']==1
    right=result['history_to_future']['LANE_CHANGE_RIGHT->KEEP']
    assert right['windows']==4 and right['anchors']==right['physical_groups']==1
    assert right['history_slots']=={'3':4} and len(right['positions'])==4
    assert result['adjacent_future_targets']['RESUME->KEEP']['windows']==1
    assert result['excluded']==dict(invalid_future=1,no_known_history=4,adjacent_mixed_or_missing=1)
    assert sum(v['windows'] for v in result['history_to_future'].values())==7
    bad=copy.deepcopy(b);bad['windows'][0]=targets.distribution(['KEEP','KEEP'])
    with pytest.raises(ValueError,match='changed labels'):review.transition_scores([a],[bad],metadata)
