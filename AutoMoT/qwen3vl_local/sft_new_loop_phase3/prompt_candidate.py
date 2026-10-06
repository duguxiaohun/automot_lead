"""20261006 RGB审计候选；显式调用，不改变 v23 训练/Action 默认合同。"""
from dataclasses import fields
import hashlib
import json
from . import prompts as baseline

CANDIDATE_NAME = 'v23_rgb_stage_candidate_20261006'
SPEED_RULES = (
    "Start from the newest frame and current speed, not the earlier acceleration or braking. "
    "STOP means an immediate sustained near-stop or continued waiting before a later release. "
    "A brief touch of near-zero followed by motion is not sustained waiting. "
    "Low speed alone is not STOP: immediate sustained pull-away is RESUME. "
    "Otherwise judge the first meaningful change from current speed: a clear reduction is DECELERATE even if speed later recovers; "
    "a sustained increase is RESUME even without a previous stop. "
    "Small fluctuations or an isolated speed rise do not establish a new speed stage. "
    "Resuming normal following does not require the lead vehicle to disappear."
)
KEEP_OLD = 'Brief braking can occur within this stage; KEEP does not imply that traffic risks have cleared.'
KEEP_NEW = ('Small speed adjustments can occur within this stage. A distinct initial speed reduction is DECELERATE even if followed by recovery; '
            'KEEP does not imply that traffic risks have cleared.')


def build_action_prompt(*, spec=None, audit=False, history_rgb_mode='4rgb'):
    text = baseline.build_action_prompt(spec=spec, audit=audit, history_rgb_mode=history_rgb_mode)
    if text.count(baseline.SPEED_ACTION_RULES) != 1 or text.count(KEEP_OLD) != 1:
        raise ValueError('candidate requires the unchanged v23 prompt surface')
    return text.replace(baseline.SPEED_ACTION_RULES, SPEED_RULES).replace(KEEP_OLD, KEEP_NEW)


def spec_from_case(row):
    """复用保存的题序，拒绝强制bool转换；答案只供离线parser，不进入生成文本。"""
    saved = row['prompt_spec']
    if type(saved['invalid_context']) is not bool:
        raise ValueError('invalid context must be bool')
    for q in saved['questions']:
        if type(q['answer']) is not bool:
            raise ValueError('question answer must be bool')
    values = {f.name: saved[f.name] for f in fields(baseline.PromptSpec) if f.name != 'questions'}
    values['goal_xy'] = tuple(values['goal_xy']) if values['goal_xy'] is not None else None
    values['questions'] = tuple(baseline.QuestionSpec(**q) for q in saved['questions'])
    spec = baseline.PromptSpec(**values)
    if baseline.prompt_spec_to_json(spec) != saved:
        raise ValueError('saved prompt specification disagrees with v23 contract')
    if {q.output_key: 'YES' if q.answer else 'NO' for q in spec.questions} != row['gt']:
        raise ValueError('saved prompt target disagrees with gt')
    return spec


def prompt_for_case(row, variant):
    spec = spec_from_case(row)
    old = baseline.build_action_prompt(spec=spec, history_rgb_mode=row['history_rgb_mode'])
    if row['action_user_prompt'] != old:
        raise ValueError('only exact v23 production cases may be replayed')
    if variant == 'baseline':
        return old
    if variant != CANDIDATE_NAME:
        raise ValueError('unknown candidate')
    return build_action_prompt(spec=spec, history_rgb_mode=row['history_rgb_mode'])


def fingerprint(variant, mode, output_mode):
    payload = dict(variant=variant, baseline_sha256=baseline.action_prompt_sha256(
        history_rgb_mode=mode, action_output_mode=output_mode))
    if variant == CANDIDATE_NAME:
        payload['candidate_source_sha256'] = hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()
    elif variant != 'baseline':
        raise ValueError('unknown candidate')
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
