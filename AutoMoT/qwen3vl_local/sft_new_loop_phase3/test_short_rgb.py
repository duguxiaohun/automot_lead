import copy
import pytest

from .history_rgb import (history_rgb_contract, history_rgb_indices, select_history_rgb_paths,
                          validate_adapter_history, history_rgb_prompt_description)
from . import prompts
from .test_prompt_candidate import case
from .prompt_candidate import spec_from_case


def test_short_input_uses_same_anchor_and_four_frame_index():
    paths = ['0007.jpg', '0008.jpg', '0009.jpg', '0010.jpg']
    assert select_history_rgb_paths(paths, '2rgb_short') == ['0008.jpg', '0010.jpg']
    assert select_history_rgb_paths(paths, '4rgb') == paths
    assert select_history_rgb_paths(paths, '2rgb_endpoints') == ['0007.jpg', '0010.jpg']
    assert history_rgb_contract('2rgb_short')['frame_offsets'] == [-2, 0]
    assert history_rgb_indices('4rgb') == (0, 1, 2, 3)


@pytest.mark.parametrize('output', ['binary', 'choice'])
def test_short_prompt_has_correct_times_and_unchanged_targets(output):
    row = case(output)
    spec = spec_from_case(row)
    before = prompts.build_action_target(spec)
    text = prompts.build_action_prompt(spec=spec, history_rgb_mode='2rgb_short')
    assert 't-0.50 s and t=0' in text
    assert 't-0.75 s' not in text
    assert prompts.build_action_target(spec) == before
    assert prompts.action_prompt_sha256(history_rgb_mode='2rgb_short', action_output_mode=output) != prompts.action_prompt_sha256(history_rgb_mode='2rgb_endpoints', action_output_mode=output)


@pytest.mark.parametrize('change', ['missing_contract', 'wrong_offsets', 'wrong_indices'])
def test_short_adapter_rejects_missing_or_wrong_binding(change):
    cfg = dict(history_rgb_mode='2rgb_short', history_rgb_contract=history_rgb_contract('2rgb_short'),
               history_rgb_selected_indices=[1, 3])
    validate_adapter_history(cfg)
    cfg = copy.deepcopy(cfg)
    if change == 'missing_contract':
        del cfg['history_rgb_contract']
    elif change == 'wrong_offsets':
        cfg['history_rgb_contract']['frame_offsets'] = [-3, 0]
    else:
        cfg['history_rgb_selected_indices'] = [0, 3]
    with pytest.raises(ValueError, match='RGB'):
        validate_adapter_history(cfg)


def test_legacy_modes_keep_exact_descriptions_and_no_new_required_metadata():
    assert history_rgb_prompt_description('4rgb') == 'four-frame history at t-0.75 s, t-0.50 s, t-0.25 s and t=0'
    assert history_rgb_prompt_description('2rgb_endpoints') == 'two endpoint frames at t-0.75 s and t=0'
    assert validate_adapter_history(dict(history_rgb_mode='2rgb_endpoints'))['frame_offsets'] == [-3, 0]


def test_saved_short_adapter_roundtrip_and_wrong_indices_rejected(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from . import train, eval as evaluation
    from qwen3vl_local.qwen35 import adapters
    monkeypatch.setattr('sys.argv', ['train', '--history-rgb-mode', '2rgb_short', '--model-dir', str(tmp_path / 'base')])
    args = train.parse_args()
    monkeypatch.setattr(adapters, 'save_adapter', lambda *args: None)
    bundle = SimpleNamespace(unwrap=lambda: object(), lora_target_modules=['q_proj'])
    folder = train._save_adapter(bundle, tmp_path, args, step=1)
    cfg = evaluation._validate_action_adapter(folder, Path(args.model_dir))
    assert cfg['history_rgb_contract']['frame_offsets'] == [-2, 0]
    cfg['history_rgb_selected_indices'] = [0, 3]
    (folder / 'sft_new_loop_phase3_adapter_config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='selected indices'):
        evaluation._validate_action_adapter(folder, Path(args.model_dir))


def test_short_pipeline_does_not_borrow_historical_endpoint_results():
    from .pipeline_support import original_cases
    with pytest.raises(ValueError, match='no historical same-input'):
        original_cases('2rgb_short', 'choice', '/unused')


def test_short_completed_results_can_be_packaged(tmp_path):
    import json
    from .pack_results import inspect_eval
    (tmp_path / 'metrics.json').write_text(json.dumps(dict(history_rgb_mode='2rgb_short', action_output_mode='choice', total_cases=1)))
    (tmp_path / 'cases.jsonl').write_text(json.dumps(dict(case_index=0, all_ok=True, history_rgb_mode='2rgb_short', prompt_spec={'action_output_mode': 'choice'})) + '\n')
    _, report = inspect_eval(tmp_path)
    assert report['cases'] == 1
