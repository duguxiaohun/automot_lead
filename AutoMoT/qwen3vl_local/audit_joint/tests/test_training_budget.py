from copy import deepcopy
import pytest
from qwen3vl_local.audit_joint.training_budget import plan,COMPONENTS


def request():
    return dict(free_bytes=1000,reserve_bytes=100,modes=2,retained_checkpoints_per_mode=7,
                components={k:dict(upper_bound_bytes=10,evidence='test bound') for k in COMPONENTS})


def test_both_models_every_epoch_and_reserve_counted():
    r=plan(request())
    assert r['checkpoint_count']==14
    assert r['required_free_bytes']==440
    assert r['sufficient'] is True


@pytest.mark.parametrize('name',COMPONENTS)
def test_missing_bound_or_evidence_cannot_pass(name):
    for replacement in ({},{'upper_bound_bytes':0}):
        q=request();q['components'][name]=replacement
        r=plan(q)
        assert r['sufficient'] is None and r['required_free_bytes'] is None
        assert r['missing_bounds']==[name]


def test_low_space_fails_and_negative_or_boolean_bounds_rejected():
    q=request();q['free_bytes']=439
    assert plan(q)['sufficient'] is False
    for value in (-1,True,1.5):
        broken=deepcopy(q);broken['components']['temporary_peak']['upper_bound_bytes']=value
        with pytest.raises(ValueError):plan(broken)
