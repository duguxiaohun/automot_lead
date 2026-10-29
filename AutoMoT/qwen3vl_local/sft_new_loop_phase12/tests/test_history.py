import copy
import json
from dataclasses import replace
from pathlib import Path
import pytest
from qwen3vl_local.sft_new_loop_phase12 import history,cache,audit,integrity,compare,plan
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from test_data_delivery import evidence


@pytest.mark.parametrize('field',['warmup_pairs_per_method','device','dtype','max_new_tokens','scope','protocol',
    'batch_size','same_encoded_inputs','synchronized','do_sample','repetition_penalty','execution_order','saving_fraction'])
def test_missing_timing_condition_is_rejected(tmp_path,field):
    evidence(tmp_path);path=tmp_path/'evaluation/4rgb/fusion_test/cases.jsonl'
    case=json.loads(path.read_text());del case['performance']['paired_benchmark'][field]
    path.write_text(json.dumps(case)+'\n')
    with pytest.raises((ValueError,KeyError)):audit.verify_evaluations(tmp_path,'4rgb')


@pytest.mark.parametrize('field,value',[('warmup_pairs_per_method',0),('device','cuda:0'),('dtype','torch.bfloat16'),
    ('max_new_tokens',128),('execution_order',[['shared','independent']]*2),('same_encoded_inputs',False)])
def test_changed_timing_condition_is_rejected(tmp_path,field,value):
    evidence(tmp_path);path=tmp_path/'evaluation/4rgb/fusion_test/cases.jsonl'
    case=json.loads(path.read_text());case['performance']['paired_benchmark'][field]=value
    path.write_text(json.dumps(case)+'\n')
    with pytest.raises(ValueError):audit.verify_evaluations(tmp_path,'4rgb')


@pytest.mark.parametrize('problem',['old_adapter','new_data','different_training','resampled_history','leaked_guard'])
def test_history_and_fresh_weight_binding_reject_mismatch(tmp_path,problem):
    evidence(tmp_path)
    if problem in ('old_adapter','new_data','different_training'):
        path=tmp_path/'evaluation/4rgb/weight_binding.json';r=proof.read(path)
        r[{'old_adapter':'adapter_identity','new_data':'ready_sha256','different_training':'training_complete_sha256'}[problem]]='wrong'
        proof.write(path,r)
    elif problem=='resampled_history':
        path=tmp_path/'cpu/reference_members.json';r=proof.read(path);r['4rgb']['1'][0]['case_index']=9;proof.write(path,r)
    else:
        path=tmp_path/'cpu/guard_members.json';r=proof.read(path);r[0]['key']=['s','route',3];proof.write(path,r)
    with pytest.raises(ValueError):audit.verify_evaluations(tmp_path,'4rgb')


def test_historical_route_absent_from_current_index_still_protected(tmp_path):
    from qwen3vl_local.sft_new_loop_phase12.tests.test_guard import test_guard_excludes_test_repetitions_and_fusion_training
    from qwen3vl_local.sft_new_loop_phase12 import validation
    test_guard_excludes_test_repetitions_and_fusion_training(tmp_path)
    # The only remaining parent-val group is now declared historical TEST,
    # even though no parent row labels that group as test.
    with pytest.raises(ValueError,match='No heldout'):
        validation.prepare(tmp_path,tmp_path/'rgb',historical_groups=[('s','Town01_1')])


def test_case_binding_disambiguates_duplicate_physical_members(tmp_path):
    evidence(tmp_path)
    r=proof.read(tmp_path/'cpu/reference_members.json')['4rgb']['1'][0]
    one=dict(case_index=0,historical_case_id='source:0',key=r['key'],rgb=r['rgb'],phase1=dict(gt={'RS1':True},parsed={'RS1':True}))
    two=copy.deepcopy(one);two.update(case_index=1,historical_case_id='source:1')
    new=copy.deepcopy([one,two]);new[1]['phase1']['parsed']['RS1']=False
    hashes={p:'sha' for p in r['rgb']}
    report=compare.compare([one,two],new,hashes,hashes,1)
    assert report['counts']['matched_cases']==2 and len(report['old_correct_losses'])==1


def test_next_epoch_consumes_unseen_frames_before_repeating(tmp_path):
    from qwen3vl_local.sft_new_loop_phase1.test_training_policy import phase1_rows
    from qwen3vl_local.sft_new_loop_phase1 import train as native
    from qwen3vl_local.sft_new_loop_phase1.training_policy import ActionHints,identity
    from qwen3vl_local.sft_new_loop_phase1.exposure_sampling import CoverageSampler,ExposureLedger
    rows=[replace(phase1_rows(16)[0],frame_id=i) for i in range(30)]
    template=[native._make_all_item(rows[0],'RS1',seed=i) for i in range(10)]
    sampler=CoverageSampler(rows,ActionHints());ledger=ExposureLedger();seen=set()
    for epoch in range(3):
        selected,report=sampler.select(template,ledger,seed=epoch)
        report=plan.coverage(sampler,ledger,selected)
        ids={identity(i.row) for i in selected}
        assert len(ids)==10 and not ids&seen
        assert report['new_frames']==10 and report['unseen_after']==20-10*epoch
        # Planning alone must NOT advance the consumed ledger.
        assert set(ledger.frames)==seen
        for item in selected:ledger.record(item,valid=True);ledger.optimizer_applied()
        ledger.finish_epoch(tmp_path,rank=1,step=(epoch+1)*10,epoch=epoch+1,planned=10)
        seen |= ids
    _,report=sampler.select(template,ledger,seed=4)
    report=plan.coverage(sampler,ledger,_)
    assert report['new_frames']==0 and report['pool_exhausted'] and report['unseen_before']==0


def source_fixture(root):
    paths=[f's/route/rgb/{i:04}.jpg' for i in range(4)]
    row=dict(scenario='s',route_id='route',frame_id=3,split='test',history_rgb_paths=paths,answers={'RS1':True})
    item=dict(row=row,spec={'phase2_spec':{'seed_key':'fixed'}},balance_key='RS1')
    case=dict(case_index=0,scenario='s',route_id='route',frame_id=3,history_rgb_paths_used=paths,
              gt={'RS1':'YES'},parsed={'RS1':'YES'},raw_output='RS1: YES')
    source=dict(phase=1,history_rgb_mode='4rgb',arguments={'split':'test'},items=[item],expected_total=1,
                image_sha256={p:'sha' for p in paths})
    root.mkdir();proof.write(root/'evaluation_inputs.json',source)
    (root/'cases.jsonl').write_text(json.dumps(case)+'\n')
    complete=dict(status='verified',phase=1,total_cases=1,inputs_sha256=proof.sha(root/'evaluation_inputs.json'),
                  case_files_sha256={'cases.jsonl':proof.sha(root/'cases.jsonl')})
    proof.write(root/'evaluation_complete.json',complete)
    return root/'evaluation_inputs.json'


@pytest.mark.parametrize('bad',['split','case_sha','label','rgb','incomplete'])
def test_historical_import_rejects_wrong_or_unbound_evidence(tmp_path,bad):
    path=source_fixture(tmp_path/'source');source=proof.read(path)
    assert len(history.import_inputs(path)['records'])==1
    if bad=='split':source['items'][0]['row']['split']='train'
    elif bad=='label':source['items'][0]['row']['answers']['RS1']=False
    elif bad=='rgb':source['items'][0]['row']['history_rgb_paths'][0]='s/other/rgb/0000.jpg'
    elif bad=='incomplete':source['expected_total']=2
    else:(path.parent/'cases.jsonl').write_text('{}\n')
    proof.write(path,source)
    complete=proof.read(path.parent/'evaluation_complete.json');complete['inputs_sha256']=proof.sha(path)
    proof.write(path.parent/'evaluation_complete.json',complete)
    with pytest.raises(ValueError):history.import_inputs(path)


def test_historical_group_promotes_training_rows_without_mutating_parents(tmp_path):
    from test_data_delivery import parents
    from qwen3vl_local.sft_new_loop_phase12 import data
    from qwen3vl_local.sft_new_loop_phase1.training_policy import physical_group,identity
    p1,p2,rgb,rows=parents(tmp_path)
    for r in rows:r['split']='train'
    p1.write_text(''.join(json.dumps(r)+'\n' for r in rows));before=(p1.read_bytes(),p2.read_bytes())
    group=physical_group(identity(rows[0]))
    data.build(p1,p2,rgb,tmp_path/'protected',historical_groups=[group])
    assert {json.loads(l)['split'] for l in (tmp_path/'protected/frame_index.jsonl').open()}=={'test'}
    assert before==(p1.read_bytes(),p2.read_bytes())
