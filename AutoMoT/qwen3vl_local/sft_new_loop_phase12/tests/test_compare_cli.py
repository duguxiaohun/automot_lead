import json
import sys
import pytest
from qwen3vl_local.sft_new_loop_phase12 import compare
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof


def native(tmp_path):
    paths=[f's/route/rgb/{i:04}.jpg' for i in range(4)]
    row=dict(scenario='s',route_id='route',frame_id=3,split='test',history_rgb_paths=paths,answers={'UE1':True})
    case=dict(case_index=0,scenario='s',route_id='route',frame_id=3,history_rgb_paths_used=paths,
              gt={'UE1':'YES'},parsed={'UE1':'YES'},raw_output='UE1: YES')
    inputs=tmp_path/'evaluation_inputs.json'
    proof.write(inputs,dict(phase=2,history_rgb_mode='4rgb',arguments={'split':'test'},
        items=[dict(row=row,spec={'seed_key':'seed'},balance_key='UE1')],expected_total=1,
        image_sha256={p:'sha' for p in paths}))
    cases=tmp_path/'cases.jsonl';cases.write_text(json.dumps(case)+'\n')
    proof.write(tmp_path/'evaluation_complete.json',dict(status='verified',phase=2,total_cases=1,
        inputs_sha256=proof.sha(inputs),case_files_sha256={'cases.jsonl':proof.sha(cases)}))
    new=dict(case_index=99,historical_case_id=f'{proof.sha(inputs)}:0',key=['s','route',3],rgb=paths,
             phase2=dict(gt={'UE1':True},parsed={'UE1':True}))
    return inputs,cases,case,new


@pytest.mark.parametrize('regression',[False,True])
def test_cli_native_old_output_matches_fixed_new_ids_and_detects_loss(tmp_path,monkeypatch,capsys,regression):
    inputs,cases,old,new=native(tmp_path)
    if regression:new['phase2']['parsed']['UE1']=False
    newpath=tmp_path/'new.jsonl';newpath.write_text(json.dumps(new)+'\n')
    hashes=tmp_path/'rgb.json';proof.write(hashes,proof.read(inputs)['image_sha256'])
    monkeypatch.setattr(sys,'argv',['compare','--baseline-cases',str(cases),'--baseline-inputs',str(inputs),
        '--new-cases',str(newpath),'--new-rgb-bindings',str(hashes),'--phase','2'])
    if regression:
        with pytest.raises(SystemExit) as exc:compare.main()
        assert exc.value.code==2
    else:compare.main()
    r=json.loads(capsys.readouterr().out)
    assert r['counts']['matched_cases']==1 and r['counts']['renumbered_matches']==1
    assert len(r['old_correct_losses'])==int(regression)


@pytest.mark.parametrize('bad',['answer','member','id','phase','shard','missing_completion'])
def test_native_id_binding_requires_verified_old_output(tmp_path,bad):
    inputs,cases,old,new=native(tmp_path)
    phase=2
    if bad=='answer':old['parsed']['UE1']='NO'
    elif bad=='member':old['frame_id']=4
    elif bad=='id':old['historical_case_id']='another-source:0'
    elif bad=='phase':phase=1
    elif bad=='shard':cases.write_text('{}\n')
    else:(tmp_path/'evaluation_complete.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        compare.bind_native_baseline([old],inputs,phase)


@pytest.mark.parametrize('changed',['cases','completion'])
def test_native_second_read_cannot_accept_changed_evidence(tmp_path,monkeypatch,changed):
    from qwen3vl_local.sft_new_loop_phase12 import history
    inputs,cases,old,new=native(tmp_path)
    importer=history.import_inputs
    old['parsed']['UE1']='NO'
    def mutate_after_import(path):
        source=importer(path)
        if changed=='cases':cases.write_text(json.dumps(old)+'\n')
        else:
            old['parsed']['UE1']='YES'
            complete=inputs.parent/'evaluation_complete.json'
            complete.write_text(complete.read_text()+' ')
        return source
    monkeypatch.setattr(history,'import_inputs',mutate_after_import)
    with pytest.raises(ValueError,match='changed during comparison'):
        compare.bind_native_baseline([old],inputs,2)
