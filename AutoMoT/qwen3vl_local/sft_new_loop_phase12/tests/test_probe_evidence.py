"""Training gate must fail with complete evidence before optimizer updates."""
import copy
import json
from types import SimpleNamespace
import pytest
import torch
from qwen3vl_local.sft_new_loop_phase12 import train,cache,cache_checks


def answers():
    answer=dict(token_ids=list(range(400)),generated_tokens=400,stop_token_id=1,reached_token_limit=False,text='native answer')
    return dict(branches=[copy.deepcopy(answer),copy.deepcopy(answer)])


@pytest.mark.parametrize('field',['token_ids','stop_token_id','reached_token_limit','text'])
def test_first_difference_keeps_terminal_evidence(field):
    a=answers();b=copy.deepcopy(a)
    if field=='token_ids':b['branches'][1][field][-1]=999
    elif field=='stop_token_id':b['branches'][1][field]=2
    elif field=='reached_token_limit':
        b['branches'][1][field]=True;b['branches'][1]['stop_token_id']=None
    else:b['branches'][1][field]='changed text'
    with pytest.raises(cache_checks.CacheParityError) as caught:cache.assert_answers(a,b)
    evidence=caught.value.evidence;diff=evidence['first_difference']
    assert diff['arm']==2 and len(evidence['cached']['branches'][1]['token_ids'])==400
    assert diff['token_index']==(399 if field=='token_ids' else 400)
    a['branches'][0]['token_ids'].clear()
    assert len(evidence['cached']['branches'][0]['token_ids'])==400


@pytest.mark.parametrize('layout',['0','1'])
@pytest.mark.parametrize('fails',[False,True])
def test_probe_persists_prior_pass_and_current_complete_failure(tmp_path,monkeypatch,layout,fails):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX',layout)
    b=SimpleNamespace(model=torch.nn.Linear(2,2),device=torch.device('cpu'))
    items=[SimpleNamespace(row=SimpleNamespace(question_domain=d,history_rgb_paths=['x']*4)) for d in ('LOCAL_JUNCTION','ROAD_CORRIDOR')]
    monkeypatch.setattr(train.plan,'evaluation',lambda *a:items)
    monkeypatch.setattr(train.plan,'identity',lambda row:(row.question_domain,))
    spec=SimpleNamespace(output_keys=('A','B'))
    monkeypatch.setattr(train.prompts,'specs',lambda *a:(spec,spec))
    monkeypatch.setattr(train.prompts,'texts',lambda *a:('native one','native two'))
    monkeypatch.setattr(train.prompts,'targets',lambda *a:('NO','NO'))
    closed=[]
    monkeypatch.setattr(train.plan.native,'_load_images',lambda *a:[SimpleNamespace(close=lambda:closed.append(True))]*2)
    monkeypatch.setattr(train.objective,'pack',lambda *a:None)
    calls=[]
    def generate(*a,**kw):
        current=json.loads((tmp_path/'gpu_prefix_probe.json').read_text())
        assert current['status']=='running_before_optimizer_updates'
        assert current['cases'][-1]['status']=='running'
        if calls:
            assert current['cases'][0]['status']=='passed' and len(current['reports'])==1
            if fails:
                left=answers();right=copy.deepcopy(left);right['branches'][0]['token_ids'][-1]=999
                cache.assert_answers(left,right)
        calls.append(True)
        return dict(fixture='native success')
    monkeypatch.setattr(train.cache,'generate',generate)
    def run():train.probe(b,None,None,'2rgb_endpoints',tmp_path,{(i.row.question_domain,):{} for i in items})
    if fails:
        with pytest.raises(cache_checks.CacheParityError):run()
    else:run()
    receipt=json.loads((tmp_path/'gpu_prefix_probe.json').read_text())
    assert receipt['optimizer_updates']==0 and closed==[True]*4
    assert receipt['input_layout']==('text_before_images_v1' if layout=='1' else 'images_before_text_legacy')
    if fails:
        assert receipt['status']=='failed_before_optimizer_updates' and len(receipt['reports'])==1
        evidence=receipt['cases'][1]['failure_evidence']
        assert evidence['first_difference']['token_index']==399
        assert evidence['independent']['branches'][0]['token_ids'][-1]==999
    else:
        assert receipt['status']=='passed_before_optimizer_updates' and len(receipt['reports'])==2
        assert b.model.training
