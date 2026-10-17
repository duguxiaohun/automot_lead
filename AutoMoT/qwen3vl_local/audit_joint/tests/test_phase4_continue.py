import json
from pathlib import Path
import pytest
from qwen3vl_local.audit_joint import phase4_continue as tool
from qwen3vl_local.audit_joint.tests.test_phase4_epoch_handoff import plan,unpack
from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import contract
from qwen3vl_local.sft_new_loop_phase4.identity import write_json


@pytest.fixture
def completed(tmp_path,plan):
    root=tmp_path/'run';run=root/'run_first_epoch_rgb2';(run/'epoch_000').mkdir(parents=True)
    view=root/'view';view.mkdir();prepared=tmp_path/'prepared'
    (root/'validated_data').mkdir()
    state=dict(stage='rgb2',command_exit_code=2,status='failed',dataset=str(view),prepared=str(prepared),stages=[])
    write_json(root/'workflow.json',state)
    write_json(view/'student_view.json',dict(source=contract(),parent={'logical_path':str(prepared/'data4')}))
    ids=[f'v{i}' for i in range(370)]
    record=dict(source=contract(),world_size=4,observation_contract={'rgb_mode':2},training={'evaluation':{'plans':{'val':{'selected_ids':ids}}}})
    for path in (run/'run.json',run/'epoch_000/phase4_contract.json'):write_json(path,record)
    (run/'epoch_000/adapter_model.safetensors').write_bytes(b'fixture')
    samples,sampling,ranks=plan;write_json(run/'epoch_000_sampling.json',sampling)
    data={'epoch_000_samples.jsonl':samples,'epoch_000_cases.jsonl':[dict(id=i) for i in ids]}
    data.update({f'epoch_000_rank{i}_steps.jsonl':rank for i,rank in enumerate(ranks)})
    for name,rows in data.items():(run/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return root


@pytest.mark.parametrize('damage',['none','weight','source','steps','val','existing'])
def test_preflight_requires_real_completed_epoch(completed,damage):
    root=completed;run=root/'run_first_epoch_rgb2'
    if damage=='weight':(run/'epoch_000/adapter_model.safetensors').unlink()
    elif damage=='source':
        p=run/'run.json';r=tool.read(p);r['source']['policy']='wrong';write_json(p,r)
    elif damage=='steps':(run/'epoch_000_rank3_steps.jsonl').write_text('')
    elif damage=='val':(run/'epoch_000_cases.jsonl').write_text('')
    elif damage=='existing':(root/'run_first_epoch_rgb4').mkdir()
    if damage=='none':assert tool.preflight(root)[3]==run
    else:
        with pytest.raises((ValueError,FileExistsError)):tool.preflight(root)


@pytest.mark.parametrize('failure',[None,'rgb2_reload','rgb2_test','rgb4','raise_after_reload'])
def test_continuation_never_retrains_rgb2_and_stops_at_failure(completed,monkeypatch,failure):
    calls=[];original=(completed/'workflow.json').read_bytes()
    monkeypatch.setattr(tool.workflow,'source_check',lambda *a:{'status':'matched'})
    monkeypatch.setattr(tool,'collect',lambda *a:{'status':'collected_not_semantically_verified'})
    def execute(command,log,env):
        stage=log.stem.replace('continuation_','');calls.append((stage,command,env))
        assert env['PYTHON'].endswith('python_dispatch.sh')
        if stage=='rgb2_test' and failure=='raise_after_reload':raise OSError('fixture')
        return 2 if failure==stage else 0
    monkeypatch.setattr(tool.workflow,'execute',execute)
    code=tool.run(completed,Path('lead_data'),Path('model'))
    assert code==(0 if failure is None else 2)
    assert [c[0] for c in calls]==(['rgb2_reload'] if failure=='rgb2_reload' else
        ['rgb2_reload','rgb2_test'] if failure in ('rgb2_test','raise_after_reload') else ['rgb2_reload','rgb2_test','rgb4'])
    assert all(c[2]['RGB_MODE']=='4' for c in calls)
    assert '--run' in calls[0][1] and 'run_automatic.sh' not in ' '.join(calls[0][1])
    assert tool.read(completed/'workflow_before_continuation.json')==json.loads(original)
    assert tool.read(completed/'workflow.json')['command_exit_code']==code


def test_collector_preserves_continuation_failure(tmp_path):
    write_json(tmp_path/'continuation.json',dict(status='failed',command_exit_code=2))
    (tmp_path/'continuation_rgb2_reload.log').write_text('fixture failure')
    tool.collect(tmp_path,tmp_path/'new_handoff')
    members=unpack(tmp_path/'new_handoff')
    assert members['continuation.json']['status']=='failed'
    assert members['continuation_rgb2_reload_log_tail.json']['tail']=='fixture failure'
