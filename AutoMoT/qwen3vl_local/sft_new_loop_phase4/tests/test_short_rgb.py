from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.sft_new_loop_phase4 import dataset as native
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT, digest, file_sha
from qwen3vl_local.sft_new_loop_phase4.input_identity import rgb_content_sha, model_input_key
from qwen3vl_local.sft_new_loop_phase4.tests.test_model import example
from qwen3vl_local.sft_new_loop_phase4.rgb_short import data, contract, model as student_model


def source_row(tmp_path, target='YES'):
    row = example(tmp_path, frames=4, target=target)
    hashes = []
    for name in row['images']:
        with Image.open(tmp_path / name) as image:
            hashes.append(rgb_content_sha(image))
    row.update(id='question', scenario='Scene', route_id='route', physical_group='Scene/route', split='train',
               slice='readiness', label_basis='reviewed_transition_band', image_rgb_sha256=hashes,
               evidence_id='original_review', visual_review={'approved': True})
    row['model_input_sha256'] = model_input_key(row)
    return row


def test_short_row_preserves_semantic_target_and_provenance_without_approval(tmp_path):
    row = source_row(tmp_path)
    old = deepcopy(row)
    small = data.short_row(row)
    assert row == old
    assert small['observation']['history_frames'] == [8, 10]
    assert small['images'] == row['images'][2:]
    assert small['image_sha256'] == row['image_sha256'][2:]
    assert small['target'] == row['target'] and small['episode'] == row['episode']
    assert small['student_reference']['source_row_sha256'] == digest(row)
    assert small['student_reference']['actual_input_human_reviewed'] is False
    assert 'visual_review' not in small
    assert small['reference_kind'] == 'transferred_reference'
    assert small['model_input_sha256'] != row['model_input_sha256']


@pytest.mark.parametrize('target', ['YES', 'NO'])
def test_real_short_encoder_masks_only_answer_eos_and_has_two_images(tmp_path, processor, target):
    row = data.short_row(source_row(tmp_path, target))
    bundle = SimpleNamespace(processor=processor, tokenizer=processor.tokenizer, device=torch.device('cpu'),
                             observation_contract=contract.observation_contract(2))
    encoded = student_model.encode(bundle, row, tmp_path, max_length=20000)
    labels = encoded['labels'][0]
    assert processor.tokenizer.decode(labels[labels != -100]) == target + '<|im_end|>'
    assert encoded['image_grid_thw'].shape[0] == 2
    inference = student_model.encode(bundle, row, tmp_path, supervise=False, max_length=20000)
    assert 'labels' not in inference
    assert inference['image_grid_thw'].shape[0] == 2


def test_four_image_encoding_is_tensor_identical_to_native(tmp_path, processor):
    from qwen3vl_local.sft_new_loop_phase4.model import encode
    row = source_row(tmp_path)
    bundle = SimpleNamespace(processor=processor, tokenizer=processor.tokenizer, device=torch.device('cpu'),
                             observation_contract=contract.observation_contract(4))
    old = encode(bundle, row, tmp_path, max_length=20000)
    new = student_model.encode(bundle, row, tmp_path, max_length=20000)
    assert old.keys() == new.keys()
    assert all(torch.equal(old[key], new[key]) for key in old)


def test_short_model_rejects_legacy_two_frame_offsets(tmp_path, processor):
    row = source_row(tmp_path)
    row['observation']['history_frames'] = [6, 10]
    row['images'] = row['images'][1::2]
    bundle = SimpleNamespace(observation_contract=contract.observation_contract(2))
    with pytest.raises(ValueError, match='history mismatch'):
        student_model.encode(bundle, row, tmp_path)


@pytest.mark.parametrize('damage', ['missing', 'changed'])
def test_selected_images_must_exist_and_match_sha(tmp_path, damage):
    small = data.short_row(source_row(tmp_path))
    image = tmp_path / small['images'][0]
    if damage == 'missing':
        image.unlink()
    else:
        Image.new('RGB', (32, 32), 'red').save(image)
    with pytest.raises((ValueError, FileNotFoundError)):
        student_model.load_images(small, tmp_path)


def test_reduced_input_conflicts_and_exposure_excluded_from_both_views(tmp_path):
    one = source_row(tmp_path)
    two = deepcopy(one)
    two.update(id='opposite', target='NO')
    two['image_rgb_sha256'][0] = 'f' * 64  # full histories differ, short histories do not
    three = deepcopy(one)
    three.update(id='exposed_val', split='val', physical_group='exposed/route')
    original = dict(train=[one, two], val=[three], test=[])
    short, exclusions = data.paired_view(original, 2, {'exposed/route'})
    four, other = data.paired_view(original, 4, {'exposed/route'})
    assert short == four == dict(train=[], val=[], test=[])
    assert exclusions == other and len(exclusions) == 3
    assert 'registered_development_exposure' in exclusions[-1]['reasons']


def test_actual_tiny_model_lora_backward_on_short_input(tmp_path, processor, model):
    from peft import LoraConfig, get_peft_model
    from qwen3vl_local.sft_v2.train import _lora_target_modules
    model.config.image_token_id = processor.image_token_id
    model.config.video_token_id = processor.video_token_id
    model.config.vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    model.config.vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    for parameter in model.parameters():
        parameter.requires_grad = False
    targets = _lora_target_modules(model, vision_scope='off', strict_vision_scope=True)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=targets, task_type='CAUSAL_LM'))
    bundle = SimpleNamespace(processor=processor, tokenizer=processor.tokenizer, device=torch.device('cpu'),
                             model=model, observation_contract=contract.observation_contract(2))
    encoded = student_model.encode(bundle, data.short_row(source_row(tmp_path)), tmp_path, max_length=20000)
    loss = model(**encoded, use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


@pytest.fixture
def native_view(tmp_path):
    raw = ROOT.parents[1] / 'lead_data'
    if not raw.is_dir():
        pytest.skip('local external LEAD RGB unavailable')
    annotations = json.loads((ROOT / 'reviewed_state_pairs_v9.json').read_text())[:2]
    parent = tmp_path / 'data4'
    native.build(annotations, raw, parent, rgb_mode=4)
    output = tmp_path / 'view'
    return parent, output


def test_real_native_build_view_roundtrip_read_only_and_no_production_copy(native_view):
    parent, output = native_view
    before = {str(p.relative_to(parent)): file_sha(p) for p in parent.rglob('*') if p.is_file()}
    descriptor = data.create(parent, output)
    two, m2 = data.load_dataset(output, rgb_mode=2)
    four, m4 = data.load_dataset(output, rgb_mode=4)
    assert sum(len(rows) for rows in two.values()) > 0
    for split in two:
        assert [row['id'] for row in two[split]] == [row['id'] for row in four[split]]
        for small, full in zip(two[split], four[split]):
            assert small['images'] == full['images'][2:]
            assert small['target'] == full['target']
    assert m2['observation_contract']['frame_offsets'] == [-2, 0]
    assert m4['observation_contract']['frame_offsets'] == [-6, -4, -2, 0]
    assert list(output.iterdir()) == [output / 'student_view.json']
    assert descriptor['production_copied'] is False
    assert before == {str(p.relative_to(parent)): file_sha(p) for p in parent.rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage', ['parent_bytes', 'redirect', 'index_redirect', 'view_offsets', 'exposure_inventory'])
def test_view_binding_rejects_drift(native_view, damage):
    parent, output = native_view
    data.create(parent, output)
    path = output / 'student_view.json'
    record = json.loads(path.read_text())
    if damage == 'parent_bytes':
        with (parent / 'train.jsonl').open('a') as stream:
            stream.write('\n')
    elif damage == 'redirect':
        original = parent.with_name('old_parent')
        parent.rename(original)
        parent.symlink_to(original, target_is_directory=True)
    elif damage == 'index_redirect':
        original = parent / 'train.jsonl'
        moved = parent / 'train-moved.jsonl'
        original.rename(moved)
        original.symlink_to(moved.name)
    elif damage == 'view_offsets':
        record['observation_contracts']['2']['frame_offsets'] = [-4, 0]
    else:
        record['exposure_files'] = {}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        data.load_dataset(output)


def test_evaluation_reports_agreement_instead_of_new_manual_accuracy(tmp_path, monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import evaluate
    row = data.short_row(source_row(tmp_path))
    monkeypatch.setattr(evaluate, 'generate', lambda *args: ('YES', 'YES'))
    report, cases = evaluate.evaluate_rows(None, [row], tmp_path)
    assert report['selection_agreement'] == 1
    assert report['independent_accuracy_certified'] is False
    assert 'accuracy' not in report
    assert cases[0]['reference_kind'] == 'transferred_reference'


def test_test_answers_cannot_change_training_membership(tmp_path):
    train = source_row(tmp_path)
    held = deepcopy(train)
    held.update(id='test_question', split='test', physical_group='held/route', target='NO')
    original = dict(train=[train], val=[], test=[held])
    selected, _ = data.paired_view(original, 2, set())
    held['target'] = 'YES'
    again, _ = data.paired_view(original, 2, set())
    assert selected['train'] == again['train'] and len(selected['train']) == 1


def test_mode_sampling_uses_identical_ids_order_and_updates(tmp_path):
    from qwen3vl_local.sft_new_loop_phase4.phase3_sampling import plan
    one = source_row(tmp_path)
    other = deepcopy(one)
    other.update(id='second', target='NO')
    other['image_rgb_sha256'][2] = 'b' * 64
    original = dict(train=[one, other], val=[], test=[])
    two, _ = data.paired_view(original, 2, set())
    four, _ = data.paired_view(original, 4, set())
    for epoch in range(2):
        a, _ = plan(two['train'], epoch=epoch, budget=8, world_size=4, expected_events=['U-E4'])
        b, _ = plan(four['train'], epoch=epoch, budget=8, world_size=4, expected_events=['U-E4'])
        assert [two['train'][i]['id'] for i in a] == [four['train'][i]['id'] for i in b]


def test_foreign_adapter_contract_rejected_before_loading_weights(tmp_path):
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'phase4_contract.json').write_text(json.dumps(dict(
        source={'old_native_checkpoint': True}, observation_contract=contract.observation_contract(2))))
    with pytest.raises(ValueError, match='source/producer contract'):
        student_model.load_for_inference(tmp_path / 'missing_model', adapter, 'cpu')


def test_build_uses_existing_replay_only_compiles_four_and_preserves_source(tmp_path, monkeypatch):
    import sys
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import build
    from qwen3vl_local.audit_joint import recovery
    base = tmp_path / 'old'
    base.mkdir()
    (base / '.build.lock').touch()
    (base / 'old_evidence').write_text('preserve')
    before = {p.name: file_sha(p) for p in base.iterdir()}
    report = dict(replay_reusable_under_current_request=True,
                  checks={'build_request_read': {'result': {'external': {}}}},
                  storage={'production': {'referenced_copy_bytes': 10}})
    monkeypatch.setattr(recovery, 'check_recovery', lambda *args, **kwargs: report)
    calls = []
    monkeypatch.setattr(build.subprocess, 'run', lambda argv, **kw: calls.append(argv))
    monkeypatch.setattr(build, 'create', lambda parent, output: dict(counts={}, exclusions=[], observation_contracts={}))
    monkeypatch.setattr('shutil.disk_usage', lambda _: SimpleNamespace(free=100 * 1024 ** 3))
    monkeypatch.setattr(sys, 'argv', ['build', '--phase4-base', str(base), '--output', str(tmp_path / 'new')])
    build.main()
    assert len(calls) == 1
    argv = calls[0]
    assert 'qwen3vl_local.sft_new_loop_phase4.dataset' in argv
    assert argv[argv.index('--rgb-mode') + 1] == '4'
    assert argv[argv.index('--max-teacher-questions-per-route') + 1] == '0'
    assert before == {p.name: file_sha(p) for p in base.iterdir()}


def test_short_trainer_sampling_entry_handles_real_view_without_weights(native_view, monkeypatch, tmp_path):
    import sys
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import train
    parent, output = native_view
    data.create(parent, output)
    # The small native fixture is not a ten-event production pool; legacy_ring
    # exercises the actual CLI/view integration without pretending it is one.
    monkeypatch.setattr(sys, 'argv', ['train', '--dataset', str(output), '--rgb-mode', '2',
        '--output-dir', str(tmp_path / 'unused_weights'), '--sampling-only', '--sampling-policy', 'legacy_ring',
        '--epochs', '1', '--epoch-samples', '4'])
    train.main()
    assert not (tmp_path / 'unused_weights').exists()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_evaluation_cli_pins_gpu_before_loader_and_writes_complete_cases(tmp_path, monkeypatch, device):
    import os
    import sys
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import evaluate
    from qwen3vl_local.sft_new_loop_phase4 import devices
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    manifest = {'observation_contract': contract.observation_contract(2)}
    (adapter / 'phase4_contract.json').write_text(json.dumps(dict(dataset=digest(manifest),
        observation_contract=manifest['observation_contract'])))
    monkeypatch.setattr(evaluate, 'load_dataset', lambda *args, **kwargs: ({'val': [{}]}, manifest))
    calls = []
    def select(n):
        assert n == 1 and 'CUDA_VISIBLE_DEVICES' not in os.environ
        assert os.environ['GPU_IDS'] == '2'
        return '2'
    monkeypatch.setenv('GPU_IDS', '2')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'old_mask')
    monkeypatch.setattr(devices, 'select_gpus', select)
    monkeypatch.setattr(evaluate, 'load_for_inference', lambda model, adapter, selected: calls.append((selected, os.environ['CUDA_VISIBLE_DEVICES'])))
    monkeypatch.setattr(evaluate, 'evaluate_rows', lambda *args: ({'count': 1}, [{'id': 'one'}]))
    output = tmp_path / 'results'
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--dataset', str(tmp_path), '--model-dir', str(tmp_path),
        '--adapter', str(adapter), '--data-root', str(tmp_path), '--output-dir', str(output), '--device', device])
    evaluate.main()
    assert calls == ([('cuda:0', '2')] if device == 'cuda' else [('cpu', 'old_mask')])
    assert json.loads((output / 'metrics.json').read_text())['expected_count'] == 1
    assert len((output / 'cases.jsonl').read_text().splitlines()) == 1


@pytest.mark.parametrize('exit_code', [0, 9])
def test_four_gpu_launcher_passes_selection_and_preserves_failure(tmp_path, exit_code):
    import os
    import subprocess
    import sys
    launcher = ROOT / 'rgb_short' / 'run.sh'
    guard = tmp_path / 'qwen3vl_local' / 'audit_joint' / 'run_guarded.sh'
    guard.parent.mkdir(parents=True)
    guard.write_text('''#!/usr/bin/env bash
set -eu
[[ "$1" == --space-path && "$3" == --min-free-gib && "$4" == 20 && "$5" == -- ]]
shift 5
exec "$@"
''')
    python = tmp_path / 'fake_python'
    python.write_text(f'#!{sys.executable}\n' + '''import json, os, pathlib, sys
if sys.argv[1] == '-c':
    assert 'CUDA_VISIBLE_DEVICES' not in os.environ
    assert os.environ['GPU_IDS'] == '1,3,5,7'
    print('1,3,5,7')
else:
    pathlib.Path(os.environ['CALL_RECORD']).write_text(json.dumps({
        'argv': sys.argv[1:], 'mask': os.environ.get('CUDA_VISIBLE_DEVICES')}))
    raise SystemExit(int(os.environ['TRAIN_EXIT']))
''')
    python.chmod(0o755)
    record = tmp_path / 'call.json'
    output = tmp_path / 'runs'
    env = dict(os.environ, PYTHON=str(python), GPU_IDS='1,3,5,7',
        CUDA_VISIBLE_DEVICES='inherited_mask', DATASET='/data/view', DATA_ROOT='/rgb',
        MODEL_DIR='/model', OUTPUT_DIR=str(output), RGB_MODE='2', RUN_TAG='smoke',
        CALL_RECORD=str(record), TRAIN_EXIT=str(exit_code))
    result = subprocess.run(['bash', str(launcher), '--epochs', '1', '--epoch-samples', '40'],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr
    call = json.loads(record.read_text())
    assert call['mask'] == '1,3,5,7'
    assert call['argv'] == ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
        '-m', 'qwen3vl_local.sft_new_loop_phase4.rgb_short.train', '--dataset', '/data/view',
        '--data-root', '/rgb', '--model-dir', '/model', '--output-dir', str(output / 'run_smoke_rgb2'),
        '--rgb-mode', '2', '--epochs', '1', '--epoch-samples', '40']
    assert (output / 'latest').is_symlink() == (exit_code == 0)
