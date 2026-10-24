from . import trajectory_action as rule
from .context_taxonomy import CONTEXT_BY_ID, DOMAIN_MANEUVER
from .choice_semantics import choice_annotation

def project(signals,context_id):
    """Apply the native anchor question domain at a new prediction origin."""
    context=CONTEXT_BY_ID[context_id]
    info=dict(anchor_context_id=context_id,question_domain=context.question_domain,
              allowed_actions=[*context.action_keys,'KEEP'])
    if signals is None:return dict(info,action=None,reason='missing_or_unreadable_target_meta')
    evidence=rule.action_evidence(signals)
    labels=rule.label_actions(signals)
    if labels is None:return dict(info,action=None,reason='incomplete_longitudinal_evidence',evidence=evidence)
    # Preserve the original domain admission, including full-maneuver STOP.
    # Longitudinal events never require proof of the absence of a lane crossing.
    if context.question_domain==DOMAIN_MANEUVER and not signals['lateral_observation_complete']:
        reason=signals.get('lateral_window_issue') or 'lateral_rgb_uncertainty'
        return dict(info,action=None,reason='incomplete_lateral_evidence:'+str(reason),evidence=evidence)
    annotation=choice_annotation(labels,context_id,evidence)
    return dict(info,action=annotation['primary_action'],reason='native_anchor_domain_action',
                keep_scope=annotation['keep_scope'],evidence=evidence)


def classify(trajectory,frame,context_id):
    return project(trajectory.signals(frame) if trajectory is not None else None,context_id)

