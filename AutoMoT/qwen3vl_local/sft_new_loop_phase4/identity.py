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
    files = ['taxonomy.py','controller.py','prompts.py','calibration.py','dataset.py',
             'sampling.py','model.py','train.py','evaluate.py','runtime.py','identity.py','action_prior.py','observation.py','replay.py']
    dependencies = ['sft_v2/train.py','sft_new_loop_phase3/build_dataset.py',
                    'sft_new_loop_phase3/trajectory_action.py']
    return dict(task='phase4_state_pair_binary_v7',answers=['YES','NO'],
                sources={f:file_sha(ROOT/f) for f in files},
                calibration_assets={f:file_sha(ROOT/f) for f in
                    ('rgb_boundary_review_v3.json','reviewed_state_pairs_v3.json','reaudit_exposure_20260929.json','third_audit_exposure_20260930.json')},
                dependencies={f:file_sha(ROOT.parent/f) for f in dependencies})


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def check_contract(recorded):
    if recorded != contract():
        raise ValueError('Phase4 source/semantic contract mismatch; use original source or a new run')
