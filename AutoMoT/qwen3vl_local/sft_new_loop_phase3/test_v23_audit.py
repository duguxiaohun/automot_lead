import json
from types import SimpleNamespace

from .build_dataset import _history


def test_four_frame_history_starts_after_initialization(tmp_path):
    (tmp_path/'rgb').mkdir()
    for i in range(6):
        (tmp_path/'rgb'/f'{i:04d}.jpg').touch()
    assert all(_history(tmp_path,i) is None for i in range(4))
    paths = _history(tmp_path,4)
    assert paths is not None and len(paths)==4
    assert all(p.endswith(f'{i:04d}.jpg') for p,i in zip(paths,range(1,5)))


def test_final_generation_has_own_checkpoint_and_case_file(tmp_path, monkeypatch):
    from . import train
    seen={}
    def evaluate(bundle, work, **kwargs):
        seen.update(kwargs)
        return {'exact_accuracy':.25,'format_valid_rate':1.0}
    monkeypatch.setattr(train,'evaluate_generation_probe',evaluate)
    args=SimpleNamespace(history_rgb_mode='4rgb',generation_eval_max_new_tokens=128,
                         generation_eval_log_every=20,eval_split='val')
    (tmp_path/'generation_val_cases.jsonl').write_text('periodic earlier cases')
    record=train.record_final_generation(object(),['sample'],args,tmp_path,7680)
    assert record['checkpoint']=='final' and record['step']==7680
    assert seen['record_path'].name=='final_generation_val_cases.jsonl'
    assert seen['step']==7680
    assert json.loads((tmp_path/'final_generation.json').read_text())==record
    assert (tmp_path/'generation_val_cases.jsonl').read_text()=='periodic earlier cases'
