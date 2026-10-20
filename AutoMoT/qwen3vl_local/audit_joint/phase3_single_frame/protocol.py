"""Authoritative saved single-image/four-query inference contract."""
import inspect
import json
from pathlib import Path
from . import POLICY, objective, history
from qwen3vl_local.audit_joint import phase3_base_trial as trial
from qwen3vl_local.audit_joint.phase3_base_dense import baseline, data


def contract(artifact):
    recipe = dict(system=objective.INSTRUCTION, message_builder=inspect.getsource(objective.messages),
                  premise_builder=inspect.getsource(data.premise),
                  history_semantics=history.SEMANTICS[artifact['config']['history_policy']],
                  history_texts=sorted({r['history_semantics'] for r in artifact['records'].values()}),
                  history_fill=artifact['config']['history_fill'],
                  letters=dict(zip(data.LETTERS,data.ACTIONS)),
                  template_sources={str(p.relative_to(trial.APP)):trial.file_sha(p) for p in (
                      Path(data.__file__),trial.APP/'qwen3vl_local/sft_new_loop_phase3/prompts.py',
                      trial.APP/'qwen3vl_local/sft_new_loop_phase3/navigation_goal.py')})
    return dict(schema='single_frame_inference_v1',route='phase3_single_frame_four_native_intents',prompt_name=POLICY,prompt_variant=POLICY,
                production_prompt_sha256=baseline.digest(recipe),prompt_recipe=recipe,
                actual_input_rgb_frames=1,history_rgb_mode='1rgb_current',history_rgb_count=1,
                history_rgb_selected_indices=[3],
                history_rgb_contract=dict(version='phase3_single_current_rgb_v1',mode='1rgb_current',
                    selected_indices=[3],index_rgb_count=4,frame_offsets=[0],frame_rate_hz=4,order='current_only'),
                output_query_count=4,action_output_mode='choice',action_keys=list(data.ACTIONS),
                action_token_protocol='full_prompt_history_output_token_identity_v1',
                history_action_encoding='standalone A-F inside newline-delimited HISTORY_ACTION slots; exact output ID required',
                unknown_policy=dict(meaning='missing information',random_history_masking=False,attention_masked=False),
                answer_order=list(data.LETTERS),mapping_contract_hash=baseline.digest(dict(zip(data.LETTERS,data.ACTIONS))),
                input_contract='One current stitched RGB, current speed/navigation/event, four historical action tokens.',
                output_contract='Four parallel A-F readouts; at most four valid soft-target losses; no vocabulary extension.',
                inference_entry='qwen3vl_local.audit_joint.phase3_single_frame.runtime.predict')


def annotate(value,artifact):
    result=dict(value)
    result.update(contract(artifact))
    result['single_frame_contract']=dict(policy=POLICY,config=artifact['config'],sources=artifact['sources'],
                                       inference=contract(artifact))
    return json.loads(json.dumps(result))


def validate(value,artifact):
    expected=contract(artifact)
    if any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('Saved single-frame RGB/prompt/query contract mismatch')
    if value.get('single_frame_contract')!=annotate({},artifact)['single_frame_contract']:
        raise ValueError('Saved single-frame dataset/source contract mismatch')
    return expected
