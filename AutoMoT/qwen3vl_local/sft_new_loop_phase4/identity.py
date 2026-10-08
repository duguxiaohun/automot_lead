"""独立Phase4合同；不接入或改写Phase3稳定release。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def contract():
    files = ['phase3_sampling.py','data_paths.py','full_pipeline.py','full_sampling.py','train_full.sh','paired_training.py','branch_support.py','paired_eval.py','launch.py','train.sh','run_full_pipeline.sh','route_quality.py','weighted_sampling.py','teacher_evaluation.py','teacher_controls.py','teacher_pool.py','teacher_review.py','teacher_rules.py','teacher_replay.py','teacher_approval.py','teacher_data.py','visual_review.py','event_scope.py','review_queue.py','automatic_context.py','privileged_context.py','privileged_visibility.py','privileged_geometry.py','privileged_producer.py','replay_safety.py',
             'maneuver_safety.py','visible_scope.py','route_context.py','route_prompts.py','route_calibration.py','taxonomy.py','controller.py','prompts.py','calibration.py','dataset.py',
             'sampling.py','model.py','train.py','evaluate.py','runtime.py','identity.py','action_prior.py','observation.py','replay.py','input_identity.py','risk_review.py',
             'admission.py','condition_builder.py','preflight.py','candidate_pool.py','selection.py']
    dependencies = ['../lead_video_tools/abnormal_duration_filter.py','sft_v2/train.py','sft_new_loop_phase3/build_dataset.py',
                    'sft_new_loop_phase3/trajectory_action.py']
    return dict(task='phase4_state_pair_binary_v40',answers=['YES','NO'],
                sources={f:file_sha(ROOT/f) for f in files},
                calibration_assets={f:file_sha(ROOT/f) for f in
                    ('thirty_sixth_audit_exposure_20261006.json','thirty_fifth_audit_exposure_20261006.json','thirty_fourth_audit_exposure_20261006.json','thirty_third_audit_exposure_20261006.json','thirty_second_audit_exposure_20261005.json','thirty_first_audit_exposure_20261005.json','thirtieth_audit_exposure_20261005.json','producer_manual_check_plan_v20_20261008.json','teacher_registry_v15.json','producer_manual_check_plan_v6_20261002.json','teacher_registry_v1.json','reviewed_state_pairs_v9.json','twenty_fifth_rgb_review_20261002.json','twenty_fifth_review_decisions_20261002.json','twenty_fifth_review_queue_20261002.json','producer_manual_check_plan_v5_20261002.json','VISUAL_REVIEW_RUBRIC_V25.md','twenty_fourth_rgb_review_20261002.json','twenty_fourth_review_queue_20261002.json','twenty_fourth_review_decisions_20261002.json','reviewed_state_pairs_v8.json','twenty_third_rgb_review_20261002.json','twenty_third_audit_exposure_20261002.json','producer_manual_check_plan_v4_20261002.json','twenty_second_audit_exposure_20261002.json','twenty_first_audit_exposure_20261001.json','twentieth_audit_exposure_20261001.json','eighteenth_nineteenth_audit_exposure_20261001.json','seventeenth_audit_exposure_20261001.json','sixteenth_audit_exposure_20260930.json','fifteenth_audit_exposure_20260930.json','fourteenth_audit_exposure_20260930.json','twelfth_thirteenth_audit_exposure_20260930.json',
                     'reviewed_state_pairs_v7.json','reviewed_holdout_state_pairs_v3.json','rgb_holdout_review_v3.json',
                     'reviewed_holdout_state_pairs_v1.json','rgb_holdout_review_v1.json',
                     'reviewed_state_pairs_v6.json','reviewed_holdout_state_pairs_v2.json','rgb_holdout_review_v2.json') +
                    ('rgb_boundary_review_v3.json','reviewed_state_pairs_v3.json','rgb_boundary_review_v4.json','reviewed_state_pairs_v4.json','rgb_boundary_review_v5.json','reviewed_state_pairs_v5.json','formal_holdout_plan_20260930.json','reaudit_exposure_20260929.json','third_audit_exposure_20260930.json','eighth_audit_exposure_20260930.json','ninth_audit_exposure_20260930.json','tenth_audit_exposure_20260930.json','eleventh_audit_exposure_20260930.json')},
                dependencies={f:file_sha(ROOT.parent/f) for f in dependencies})


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def contract_differences(recorded,current=None):
    current=contract() if current is None else current
    if not isinstance(recorded,dict):return ['contract: missing or invalid record']
    differences=[]
    for key in sorted(set(recorded)|set(current)):
        old,new=recorded.get(key),current.get(key)
        if old==new:continue
        if key in ('sources','dependencies','calibration_assets') and isinstance(old,dict) and isinstance(new,dict):
            differences.extend(f'{key}/{name}: recorded={old.get(name)}, current={new.get(name)}'
                               for name in sorted(set(old)|set(new)) if old.get(name)!=new.get(name))
        else:differences.append(f'{key}: changed or missing')
    return differences


def check_contract(recorded):
    differences=contract_differences(recorded)
    if differences:
        raise ValueError('Phase4 source/semantic contract mismatch; use original source or rebuild for current source; '
                         +' | '.join(differences))
