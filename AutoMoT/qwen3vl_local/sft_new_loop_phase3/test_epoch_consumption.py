from collections import Counter

import pytest
import torch

from . import train
from .test_validation_balance import candidate_rows


def test_full_epoch_rejects_skipped_real_input_and_allows_padding():
    with pytest.raises(ValueError, match='packing rejected'):
        train.require_packed_on_all_ranks(None, padded=False, device='cpu', world_size=1)
    train.require_packed_on_all_ranks({}, padded=False, device='cpu', world_size=1)
    train.require_packed_on_all_ranks(None, padded=True, device='cpu', world_size=1)


def test_missing_input_on_another_rank_stops_local_rank(monkeypatch):
    def remote_missing(tensor, op):
        assert op == train.dist.ReduceOp.MAX
        tensor.fill_(1)
    monkeypatch.setattr(train.dist, 'all_reduce', remote_missing)
    with pytest.raises(ValueError, match='packing rejected'):
        train.require_packed_on_all_ranks({}, padded=False, device='cpu', world_size=4)


def test_actual_counts_are_checked_not_just_planned_counts():
    work = train._balanced_work(candidate_rows(), target_per_bin=4, seed=1, action_output_mode='choice')
    counts = Counter(train._balance_class(i.row) for i in work)
    assert train.verify_epoch_consumption(work, counts, device='cpu', world_size=1)['status'] == 'verified'
    counts[train.CONTEXT_IDS[0]] -= 1
    with pytest.raises(ValueError, match='differs from plan'):
        train.verify_epoch_consumption(work, counts, device='cpu', world_size=1)


def test_all_rank_consumption_reconstructs_full_epoch(monkeypatch):
    work = train._balanced_work(candidate_rows(), target_per_bin=4, seed=1, action_output_mode='choice')
    local = Counter(train._balance_class(i.row) for i in work[::4])
    other = Counter(train._balance_class(i.row) for rank in (1, 2, 3) for i in work[rank::4])
    def sum_remote(tensor, op):
        assert op == train.dist.ReduceOp.SUM
        tensor += torch.tensor([other[k] for k in (*train.CONTEXT_IDS, 'INVALID')])
    monkeypatch.setattr(train.dist, 'all_reduce', sum_remote)
    result = train.verify_epoch_consumption(work, local, device='cpu', world_size=4)
    assert {result['actual'][k] for k in train.CONTEXT_IDS} == {4}


def test_validation_does_not_score_an_incomplete_fixed_list(monkeypatch):
    from types import SimpleNamespace
    model = torch.nn.Linear(1, 1)
    work = train._balanced_work(candidate_rows(), target_per_bin=1, seed=1, action_output_mode='choice')
    monkeypatch.setattr(train, '_load_images', lambda *a: [])
    monkeypatch.setattr(train, '_build_inputs', lambda *a, **kw: None)
    with pytest.raises(ValueError, match='fixed validation cannot skip'):
        train.evaluate_loss(SimpleNamespace(model=model), work, history_rgb_mode='4rgb',
                            max_length=1, format_loss_weight=.25, device=torch.device('cpu'), world_size=1)
    assert model.training


@pytest.mark.parametrize('reject_input', [False, True])
def test_native_cpu_training_loop_records_only_complete_epoch(tmp_path, monkeypatch, reject_input):
    import json
    import sys
    from types import SimpleNamespace
    from .test_entry_admission import fixture
    from . import preflight
    index = fixture(tmp_path, 'valid')
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['train.py', '--index', str(index), '--output-dir', str(output),
        '--device', 'cpu', '--num-epochs', '1', '--focus-balance-count', '4', '--action-output-mode', 'choice',
        '--eval-steps', '0', '--generation-eval-steps', '0', '--save-steps', '0', '--no-save-final',
        '--no-save-best-val', '--no-save-best-generation', '--no-tb', '--no-require-invalid-coverage'])
    model = torch.nn.Linear(1, 1)
    bundle = SimpleNamespace(model=model, unwrap=lambda: model, lora_target_modules=['weight'])
    monkeypatch.setattr(preflight, 'check_model', lambda *a: None)
    monkeypatch.setattr(train, 'load_model_with_lora', lambda *a, **kw: bundle)
    monkeypatch.setattr(train, '_load_images', lambda *a: [])
    monkeypatch.setattr(train, '_build_inputs', lambda *a, **kw: None if reject_input else {})
    monkeypatch.setattr(train, '_loss_one', lambda *a: (model.weight.square().sum() + model.bias.square().sum(), {}))
    path = output / 'balance/consumed_epoch_0001.json'
    if reject_input:
        with pytest.raises(ValueError, match='packing rejected'):
            train.train(train.parse_args())
        assert not path.exists()
    else:
        train.train(train.parse_args())
        result = json.loads(path.read_text())
        assert result['status'] == 'verified'
        assert {result['actual'][k] for k in train.CONTEXT_IDS} == {4}
