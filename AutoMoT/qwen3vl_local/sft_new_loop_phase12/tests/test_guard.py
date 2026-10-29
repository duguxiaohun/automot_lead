import copy
import json
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from qwen3vl_local.sft_new_loop_phase12 import cache, compare, validation, prompts, evaluate
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from qwen3vl_local.qwen35.tests.test_backend import model


def comparisons():
    paths=['s/r/rgb/0000.jpg','s/r/rgb/0003.jpg'];hashes={p:'sha' for p in paths}
    old=[dict(case_index=0,key=['s','r',3],rgb=paths,phase1=dict(gt={'VULNERABLE':True},parsed={'VULNERABLE':True}))]
    new=copy.deepcopy(old);new[0]['case_index']=9;new[0]['phase1']['parsed']['VULNERABLE']=False
    return old,new,hashes


def test_renumbered_physical_member_cannot_hide_regression():
    old,new,hashes=comparisons();r=compare.compare(old,new,hashes,hashes,1)
    assert r['counts']['renumbered_matches']==1 and len(r['old_correct_losses'])==1
    assert r['review_required'] and r['status']=='regression_detected'


@pytest.mark.parametrize('problem',['empty','unmatched','partial','ambiguous','question_gap','sha'])
def test_inadequate_comparison_never_certifies_protection(problem):
    old,new,hashes=comparisons();new[0]['phase1']['parsed']['VULNERABLE']=True
    if problem=='empty':new=[]
    if problem=='unmatched':new[0]['key'][0]='different'
    if problem=='partial':old.append({**copy.deepcopy(old[0]),'case_index':8,'key':['s','other',3]})
    if problem=='ambiguous':old.append({**copy.deepcopy(old[0]),'case_index':8})
    if problem=='question_gap':new[0]['phase1']['gt']['RS1']=True;new[0]['phase1']['parsed']['RS1']=True
    r=compare.compare(old,new,hashes,{} if problem=='sha' else hashes,1)
    assert r['review_required'] and r['status']=='insufficient_comparable_evidence'


def test_complete_correct_comparison_can_pass():
    old,new,hashes=comparisons();new[0]['phase1']['parsed']['VULNERABLE']=True
    assert compare.compare(old,new,hashes,hashes,1)['review_required'] is False


@pytest.mark.parametrize('frames',[2,4])
def test_real_nonzero_lora_full_generations_and_alternating_benchmark(model,frames,monkeypatch):
    from peft import LoraConfig,get_peft_model
    adapted=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,lora_dropout=0,
        target_modules=['q_proj','v_proj','o_proj'],task_type='CAUSAL_LM')).eval()
    with torch.no_grad():
        for name,p in adapted.named_parameters():
            if 'lora_B' in name:p.normal_(0,.03)
    prefix=[10]+[v for _ in range(frames) for v in (2,4,4,4,4,3)]
    pixels=torch.randn(frames*16,3*2*16*16);grid=torch.tensor([[1,4,4]]*frames)
    def encode(bundle,images,text):
        ids=torch.tensor([prefix+([11,12,13] if text=='first' else [21,22,23,24,25])])
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=pixels,image_grid_thw=grid)
    monkeypatch.setattr(cache,'encode',encode)
    tokenizer=SimpleNamespace(eos_token_id=1,convert_tokens_to_ids=lambda x:1)
    processor=SimpleNamespace(batch_decode=lambda ids,**kw:[str(ids.flatten().tolist())])
    bundle=SimpleNamespace(model=adapted,device=torch.device('cpu'),tokenizer=tokenizer,processor=processor)
    report=cache.generate(bundle,[],['first','second'],max_new_tokens=8,check_full=True,benchmark_repeats=2)
    assert report['full_answer_equivalence_checked'] and len(report['paired_benchmark']['trials'])==2
    assert report['paired_benchmark']['warmup_pairs_per_method']==1
    assert all(len(b['token_ids'])==b['generated_tokens'] for b in report['branches'])
    assert report['paired_benchmark']['scope'].startswith('model prefill')
    independent={'branches':copy.deepcopy(report['independent_branches'])}
    independent['branches'][0]['stop_token_id']=999
    with pytest.raises(ValueError):cache.assert_answers(report,independent)
    independent={'branches':copy.deepcopy(report['independent_branches'])}
    independent['branches'][0]['token_ids'].append(999)
    with pytest.raises(ValueError):cache.assert_answers(report,independent)


def test_guard_excludes_test_repetitions_and_fusion_training(tmp_path):
    p1=[];p2=[];fusion=[];data_root=tmp_path/'rgb'
    for num,split in [(1,'val'),(2,'val'),(3,'val'),(4,'test')]:
        route=f'Town01_Rep0_{num}_route0_01_01_01_01_01'
        paths=[f's/{route}/rgb/{i:04}.jpg' for i in range(4)]
        for path in paths:
            dest=data_root/path;dest.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(8,8)).save(dest)
        answers={k:bool(num%2) for k in prompts.P1.ANSWER_KEYS}
        p1.append(dict(scenario='s',route_id=route,frame_id=3,split=split,answers=answers,history_rgb_paths=paths))
    p2.append({**p1[1],'route_id':p1[1]['route_id'].replace('Rep0','Rep9'),'split':'test'})
    fusion.append({**p1[2],'split':'train'})
    for path,rows in [('phase1/data/frame_index.jsonl',p1),('phase2/data/frame_index.jsonl',p2),('data/frame_index.jsonl',fusion)]:
        dest=tmp_path/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=validation.prepare(tmp_path,data_root);members=proof.read(tmp_path/'cpu/guard_members.json')
    assert len(members)==1 and members[0]['key'][1]==p1[0]['route_id']
    assert not report['coverage_adequate'] and not report['highway_selection_certified']


def test_guard_promotion_vetoes_old_correct_loss_even_if_aggregate_improves(tmp_path):
    base=dict(case_index=0,key=['s','r',3],rgb=['s/r/rgb/0003.jpg'],scenario='s',
        phase1=dict(gt={'VULNERABLE':True,'STATIC_OBSTACLE':False},parsed={'VULNERABLE':True,'STATIC_OBSTACLE':True}),
        phase2=dict(gt={'UE1':False},parsed={'UE1':False}))
    changed=copy.deepcopy(base);changed['phase1']['parsed'].update(VULNERABLE=False,STATIC_OBSTACLE=False)
    for name,case in [('old',base),('new',changed)]:
        folder=tmp_path/name;folder.mkdir();(folder/'weak_guard').mkdir()
        for sub,phases in [('',(1,2)),('weak_guard',(1,))]:
            (folder/sub/'cases.jsonl').write_text(json.dumps(case)+'\n')
            proof.write(folder/sub/'metrics.json',evaluate.metrics([case],phases))
    r=validation.decision(tmp_path/'new',tmp_path/'old',{'coverage_adequate':True})
    assert not r['eligible'] and len(r['old_correct_losses'])==2
    assert not r['historical_model_protection_verified']


def test_conflicting_hash_aliases_are_rejected():
    old,new,hashes=comparisons()
    hashes['/another/root/s/r/rgb/0000.jpg']='different-sha'
    with pytest.raises(ValueError):compare.compare(old,new,hashes,{},1)


def test_guard_sampling_is_route_diverse_and_reproducible():
    rows=[]
    for group in range(10):
        for frame in range(5):
            rows.append(dict(scenario='s',route_id=f'Town01_Rep0_{group}_route0',frame_id=frame,
                             answers={'VULNERABLE':True}))
    chosen=validation.select(rows,'VULNERABLE',True,8)
    assert chosen==validation.select(list(reversed(rows)),'VULNERABLE',True,8)
    assert len({r['route_id'] for r in chosen})==8


def test_audit_rejects_missing_or_changed_cache_baseline(tmp_path):
    from test_data_delivery import evidence
    from qwen3vl_local.sft_new_loop_phase12 import audit
    evidence(tmp_path)
    path=tmp_path/'evaluation/4rgb/fusion_test/cases.jsonl';case=json.loads(path.read_text())
    case['performance']['independent_branches'][0]['token_ids']=[99]
    path.write_text(json.dumps(case)+'\n')
    with pytest.raises(ValueError):audit.verify_evaluations(tmp_path,'4rgb')


@pytest.mark.parametrize('problem',['missing_validation','false_promotion','missing_timing','wrong_adapter'])
def test_audit_rejects_incomplete_or_bypassed_guard_evidence(tmp_path,problem):
    import shutil
    from test_data_delivery import evidence
    from qwen3vl_local.sft_new_loop_phase12 import audit
    evidence(tmp_path)
    if problem=='missing_validation':shutil.rmtree(tmp_path/'runs/4rgb/val_step_1000')
    elif problem=='false_promotion':
        path=tmp_path/'runs/4rgb/val_step_2000/selection_review.json'
        r=proof.read(path);r['promoted']=True;proof.write(path,r)
    elif problem=='wrong_adapter':
        path=tmp_path/'runs/4rgb/selection.json'
        r=proof.read(path);r['adapter']='best_step_2000';proof.write(path,r)
    else:
        path=tmp_path/'evaluation/4rgb/fusion_test/cases.jsonl'
        r=json.loads(path.read_text());r['performance']['paired_benchmark']['trials'].pop()
        path.write_text(json.dumps(r)+'\n')
    with pytest.raises(ValueError):audit.verify_evaluations(tmp_path,'4rgb')
