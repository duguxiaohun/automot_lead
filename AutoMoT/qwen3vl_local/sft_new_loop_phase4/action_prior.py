"""Phase4先验导出。七类词表与Action相同，但不替换稳定Phase3标签入口。"""
from .controller import combine_priors

VOCABULARY=('UNCOND','DECELERATE','STOP','RESUME','LANE_CHANGE_LEFT','LANE_CHANGE_RIGHT','KEEP')


def export(episodes):
    prior=combine_priors(episodes)
    action=prior['action']
    return dict(source='phase4_state_transition_v2',valid=action is not None,
                token_name=action,token_id=VOCABULARY.index(action) if action is not None else None,
                text=prior.get('text'),status=prior['status'],instances=prior['instances'],
                conflict_instances=prior['conflict_instances'],recheck_instances=prior['recheck_instances'])
