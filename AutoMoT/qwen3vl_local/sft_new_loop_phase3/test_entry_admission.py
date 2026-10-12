"""Formal train/eval reject unsafe full indices before output/model/NCCL."""
import json
import sys

import pytest

from . import preflight, candidate_data, train, eval as evaluation
from .prompt_candidate import CANDIDATE_NAME
from .test_rgb_review_20260916 import index_rows, write_index
from .context_taxonomy import CONTEXT_BY_ID


def fixture(tmp_path, defect):
    rows=index_rows(1)
    for row in rows:
        context=CONTEXT_BY_ID[row['context_id']]
        row.update(dataset_name=train.DATASET_NAME,town='Town01',true_rs=context.allowed_rs[0],
            prompt_road_structure=context.allowed_rs[0],question_domain=context.question_domain,
            action_signature='INVALID' if row['invalid_action_context'] else 'KEEP',
            event=context.source_event,goal_ego_xy=[10,0],
            history_rgb_paths=['unused.jpg']*4,latest_rgb_path='unused.jpg')
        if row['invalid_action_context']:
            row['invalid_source']=f"source={row['context_id']}|true_rs={row['true_rs']}|asked_context={row['context_id']}"
    if defect=='overlap':rows.append(dict(rows[0],split='test'))
    if defect=='development':
        group=next(g for g in candidate_data.exposure_contract(candidate_data.EXPOSURE)['groups']
                   if g not in preflight.development_route_groups())
        row=next(r for r in rows if r['split']=='test')
        row['scenario'],row['route_id']=group.split('/',1)
    index=write_index(tmp_path,rows)
    manifest=tmp_path/'manifest.json'
    data=json.loads(manifest.read_text())
    data['additional_development']=candidate_data.exposure_contract(candidate_data.EXPOSURE)
    if defect=='prompt':data['prompt_contract']['production_prompt_sha256']={}
    if defect=='exposure':data['additional_development']['source_sha256']='wrong'
    manifest.write_text(json.dumps(data))
    if defect=='missing':manifest.unlink()
    return index


@pytest.mark.parametrize('entry',['train','eval'])
@pytest.mark.parametrize('variant,defect,error',[
    (v,d,e) for v in ('baseline',CANDIDATE_NAME) for d,e in (
        ('missing','manifest required'),('prompt','prompt hash mismatch'),
        ('development','development route in holdout'),('overlap','physical route leakage'),
        ('exposure','exposure-isolated')) if d!='exposure' or v!='baseline'])
def test_formal_entries_reject_before_reading_or_model(tmp_path,monkeypatch,entry,variant,defect,error):
    index=fixture(tmp_path,defect);out=tmp_path/'output'
    module=train if entry=='train' else evaluation
    argv=[entry+'.py','--index',str(index),'--action-output-mode','choice',
          '--prompt-variant',variant,'--output-dir',str(out),'--device','cpu']
    if entry=='train':argv+=['--sampling-only']
    else:argv+=['--no-timestamp-output']
    monkeypatch.setattr(sys,'argv',argv)
    def forbidden(*a,**kw):pytest.fail('admission must run before sampling/model/NCCL')
    monkeypatch.setattr(module,'_read_rows',forbidden)
    monkeypatch.setattr(module,'setup_distributed',forbidden)
    monkeypatch.setattr(preflight,'check_model',forbidden)
    if entry=='eval':monkeypatch.setattr(module,'load_eval_bundle',forbidden)
    with pytest.raises(ValueError,match=error):
        (train.train if entry=='train' else evaluation.evaluate)(module.parse_args())
    assert not out.exists()


@pytest.mark.parametrize('variant',['baseline',CANDIDATE_NAME])
@pytest.mark.parametrize('mode',['binary','choice'])
def test_valid_full_index_preserves_counts(tmp_path,variant,mode):
    index=fixture(tmp_path,'valid')
    report=preflight.check_index(index,mode,prompt_variant=variant)
    assert report['physical_route_overlap']==0
    assert all(report['counts'][f'{s}/{c}']==1 for s in ('train','val','test')
               for c in preflight.CONTEXT_IDS)


@pytest.mark.parametrize('variant',['baseline',CANDIDATE_NAME])
@pytest.mark.parametrize('entry',['train','eval'])
def test_valid_native_sampling_retains_equal_events(tmp_path,monkeypatch,capsys,variant,entry):
    index=fixture(tmp_path,'valid');out=tmp_path/'output'
    module=train if entry=='train' else evaluation
    argv=[entry+'.py','--index',str(index),'--action-output-mode','choice',
          '--prompt-variant',variant,'--output-dir',str(out),'--device','cpu',
          '--no-require-invalid-coverage']
    if entry=='train':
        argv+=['--sampling-only','--focus-balance-count','4','--eval-balance-count','1',
               '--generation-eval-balance-count','1']
    else:argv+=['--no-timestamp-output']
    monkeypatch.setattr(sys,'argv',argv)
    if entry=='train':
        train.train(train.parse_args())
        assert '[sampling-preflight]' in capsys.readouterr().out
        assert not out.exists()
    else:
        class ModelBoundary(Exception):pass
        def stop(*a,**kw):raise ModelBoundary()
        monkeypatch.setattr(evaluation,'load_eval_bundle',stop)
        with pytest.raises(ModelBoundary):evaluation.evaluate(evaluation.parse_args())
        plan=json.loads((out/'evaluation_plan.json').read_text())['plan']
        assert plan['per_event']==1
        assert len(plan['selected_ids'])==10
