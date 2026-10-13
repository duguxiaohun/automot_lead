"""Limited capability coverage must never excuse invalid or unequal datasets."""
from copy import deepcopy
import json

import pytest

from qwen3vl_local.audit_joint import rule_baseline as rb, server_finish as sf
from qwen3vl_local.audit_joint.tests.test_rule_baseline import budget, diagnostic_pool


def configured(tmp_path, monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import evaluation_policy
    raw=diagnostic_pool(tmp_path,monkeypatch)
    monkeypatch.setattr(evaluation_policy,'pack_binding',lambda _:dict(references='fixture'))
    monkeypatch.setattr(evaluation_policy,'prepare',lambda data,**kw:(data,dict(plans={},exclusions=[])))
    return raw


def check(tmp_path,policy='existing_data_development'):
    return rb.check(tmp_path,tmp_path/'refs',budget=budget(),space_path=tmp_path,coverage_policy=policy)


def test_strict_stays_blocked_and_development_keeps_labels_and_explicit_gap(tmp_path,monkeypatch):
    raw=configured(tmp_path,monkeypatch);before=deepcopy(raw)
    assert check(tmp_path,'strict')['status']=='blocked'
    r=check(tmp_path)
    assert r['status']=='admitted_existing_data_development' and r['errors']==[]
    assert r['unsupported_capabilities']==['U-E5/proceed/YES']
    assert not r['all_critical_capabilities_supported'] and not r['joint_g0_certified']
    assert not r['independent_accuracy_certified'] and raw==before
    for m in r['modes'].values():
        assert m['critical_support']['U-E5/proceed/YES']['rows']==0
        for world in ('1','4'):
            assert len(m['schedule']['worlds'][world][0]['events'])==10
            assert all(e['presentations']==1024 for e in m['schedule']['worlds'][world][0]['events'].values())


@pytest.mark.parametrize('gap',['U-E2/return/YES','U-E5/proceed/NO','U-E7/complete/YES'])
def test_other_capability_gaps_still_block(tmp_path,monkeypatch,gap):
    raw=configured(tmp_path,monkeypatch)
    raw['train'][:]=[r for r in raw['train'] if '/'.join((r['episode']['event'],r['edge'],r['target']))!=gap]
    r=check(tmp_path)
    assert r['status']=='blocked' and any(gap in e for e in r['errors'])


@pytest.mark.parametrize('defect',['event_missing','isolation','references','budget','drift','selection'])
def test_development_preserves_other_admission_failures(tmp_path,monkeypatch,defect):
    from qwen3vl_local.sft_new_loop_phase4 import evaluation_policy,full_sampling
    raw=configured(tmp_path,monkeypatch)
    def reject(*a,**kw):raise ValueError('fixture invalid input')
    if defect=='event_missing':raw['train'][:]=[r for r in raw['train'] if r['episode']['event']!='U-E1']
    if defect=='isolation':monkeypatch.setattr(full_sampling,'validate_dataset_scope',reject)
    if defect=='references':monkeypatch.setattr(evaluation_policy,'pack_binding',reject)
    if defect=='budget':monkeypatch.setattr(rb,'budget_check',lambda *a:dict(sufficient=False))
    if defect=='drift':
        seq=iter([{'sha':'a'},{'sha':'b'}]);monkeypatch.setattr(rb,'dataset_bindings',lambda _:next(seq))
    if defect=='selection':monkeypatch.setattr(evaluation_policy,'prepare',reject)
    assert check(tmp_path)['status']=='blocked'


def test_final_filtered_gap_is_reported_even_when_raw_pool_had_support(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import evaluation_policy
    from qwen3vl_local.sft_new_loop_phase4.tests.test_production_selection_v50 import pool
    raw=configured(tmp_path,monkeypatch)
    raw['train'].append(next(r for r in pool('rule_teacher')['train']
        if r['episode']['event']=='U-E5' and r['target']=='YES'))
    def select(data,**kw):
        data['train']=[r for r in data['train'] if not (r['episode']['event']=='U-E5' and r['target']=='YES')]
        return data,dict(plans={},exclusions=[])
    monkeypatch.setattr(evaluation_policy,'prepare',select)
    r=check(tmp_path)
    assert r['status']=='admitted_existing_data_development'
    assert r['unsupported_capabilities']==['U-E5/proceed/YES']
    assert not r['modes']['2']['source_automatic_support']['missing_critical_support']


def test_unknown_policy_cannot_opt_out(tmp_path,monkeypatch):
    configured(tmp_path,monkeypatch)
    assert check(tmp_path,'ignore_all')['status']=='blocked'


def test_raw_gate_continues_only_for_known_gap(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import candidate_pool,teacher_replay,teacher_data
    from qwen3vl_local.audit_joint import first_epoch_budget
    from qwen3vl_local.audit_joint.io import file_sha
    p=tmp_path/'pilot';(p/'production').mkdir(parents=True);(p/'candidates.json').write_text('{}')
    index=p/'production/index.json';index.write_text('{}')
    out=tmp_path/'out';out.mkdir();calls=[]
    monkeypatch.setattr(candidate_pool,'validate',lambda p:p)
    monkeypatch.setattr(first_epoch_budget,'profile',lambda _:dict(fixture=True))
    monkeypatch.setattr(teacher_replay,'produce',lambda *a,**kw:dict(complete_causal_production=True,index_sha256=file_sha(index)))
    monkeypatch.setattr(sf,'raw_training_support',lambda _:dict(missing_critical_support=['U-E5/proceed/YES']))
    def compile(*a,**kw):
        calls.append('compile');raise ValueError('fixture stops before actual build')
    monkeypatch.setattr(teacher_data,'compile_bundle',compile)
    r=sf.build(p,tmp_path,out,tmp_path/'model',coverage_policy='existing_data_development')
    assert calls==['compile'] and r['status']=='blocked' and r['stage']=='rgb4_rule_dataset'
    assert r['unsupported_capabilities']==['U-E5/proceed/YES'] and not r['training_started']


def test_package_preserves_limited_status_and_original_zero_support(tmp_path):
    result=dict(status='admitted_existing_data_development',stage='finished_cpu',
        coverage_policy='existing_data_development',unsupported_capabilities=['U-E5/proceed/YES'])
    (tmp_path/'build_result.json').write_text(json.dumps(result))
    assert sf.package(tmp_path,0)['status']=='verified'
    r=json.loads((tmp_path/'finish_receipt.json').read_text())
    assert r['status']==result['status'] and r['unsupported_capabilities']==result['unsupported_capabilities']


def test_prepare_missing_model_never_scans_or_generates_data(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import candidate_pool
    monkeypatch.setattr(candidate_pool,'scan',lambda *a:pytest.fail('no scan before model config'))
    monkeypatch.setattr('sys.argv',['finish','prepare','--output',str(tmp_path),
        '--model-dir',str(tmp_path/'missing'),'--coverage-policy','existing_data_development'])
    assert sf.main()==2
    r=json.loads((tmp_path/'build_result.json').read_text())
    assert r['stage']=='candidate_inventory' and not r['training_started']
