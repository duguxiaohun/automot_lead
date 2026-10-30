"""Persistent-bundle online Phase1/2/3 inference, with no label input.

Load/validate the matching native adapter first. Images are selected RGB frames,
not paths. Phase3 scene/speed/goal are current observations, never frozen by the
question scheduler. Formal evaluation and training do not call this module.
"""
import torch
from . import text_prefix, question_schedule
from qwen3vl_local.sft_v3.train import _kv_start_state, _student_generate_kv
from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices


@torch.no_grad()
def predict_phase(bundle, phase, images, *, history_rgb_mode='4rgb', domain=None,
                  context_id=None, road_structure=None, context_detail='', goal_xy=None,
                  current_speed_mps=None, action_output_mode='binary', seed='online',
                  schedule=None, max_new_tokens=160):
    if phase not in (1,2,3):raise ValueError('Expected phase 1, 2 or 3')
    if len(images)!=len(history_rgb_indices(history_rgb_mode)):raise ValueError('Wrong RGB count')
    if max_new_tokens<=0:raise ValueError('Invalid generation budget')
    from qwen3vl_local.sft_new_loop_phase1 import prompts as p1
    from qwen3vl_local.sft_new_loop_phase2 import prompts as p2
    from qwen3vl_local.sft_new_loop_phase3 import prompts as p3
    module={1:p1,2:p2,3:p3}[phase]
    if phase==2 and domain not in p2.QUESTION_DOMAINS:raise ValueError('Explicit Phase2 question domain required')
    ticket=None
    if schedule is not False and text_prefix.enabled():
        scheduler=schedule or question_schedule.for_bundle(bundle,f'phase{phase}',seed)
        ticket=scheduler.next();seed=ticket['seed']
    kwargs=dict(variant='all_random_order',answers={},seed_key=seed)
    if phase==2:kwargs['answers']={p2.DOMAIN_ANSWER_KEYS[domain]:True}
    if phase==3:
        kwargs['answers']={key:False for key in (*p3.ACTION_KEYS,p3.INVALID_KEY)}
        kwargs.update(context_id=context_id,road_structure=road_structure,context_detail=context_detail,
            goal_xy=goal_xy,current_speed_mps=current_speed_mps,action_output_mode=action_output_mode)
    spec=module.make_prompt_spec(**kwargs)
    if phase==1:
        messages=[dict(role='system',content=p1.SYSTEM_PROMPT),dict(role='user',content=text_prefix.content(
            images,p1.build_phase1_prompt(spec=spec,history_rgb_mode=history_rgb_mode)))]
    else:
        builder=p2.build_event_messages if phase==2 else p3.build_action_messages
        messages=builder(images=images,spec=spec,history_rgb_mode=history_rgb_mode)
    bundle.model.eval()
    state=_kv_start_state(bundle,messages)
    text,_,tokens=_student_generate_kv(bundle,state,max_new_tokens)
    parser={1:p1.parse_phase1_output,2:p2.parse_event_output,3:p3.parse_action_output}[phase]
    return dict(phase=phase,text=text,answers=parser(text,spec=spec),tokens=int(tokens.numel()),
        question_order=list(spec.output_keys),schedule=ticket,text_prefix=text_prefix.report(bundle))
