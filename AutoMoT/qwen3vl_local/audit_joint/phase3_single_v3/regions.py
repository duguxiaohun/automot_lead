"""Six native action meanings, current spatial direction, independent eligibility."""
from . import VERSION
from ..phase3_comparison_policy import contract as comparison_contract
VARIANTS = ('six',)
ACTIONS = ('DECELERATE','STOP','RESUME','LANE_CHANGE_LEFT','LANE_CHANGE_RIGHT','KEEP')
def names(variant):
    if variant != 'six': raise ValueError('single_requires_six_representation')
    return ACTIONS
def contract(variant):
    return dict(version=VERSION,variant=variant,names=list(names(variant)),
        six_direction='native confirmed lane side projected onto current lane right; abs(cos)<0.1 invalid',
        six_longitudinal='native per-origin speed and evidence unchanged',
        six_history='offline native intention, later evidence allowed; native gap fill before direction transform',
        missing='native anchor eligible and at least one native future pair; no geometry endpoint intersection',
        candidate_scope='verified native parent current observation; independent native-eligible split v3; no abc target eligibility',
        split_policy='single_native_coverage_preserve_holdouts_v1',
        sampling='six target mass only; event quota; frame cap 1/epoch and 3/run',
        objective='native six full_vocabulary_soft_target_cross_entropy',
        validation_selection='full eligible validation diagnostic; final completed epoch used for test',
        comparison=comparison_contract())
