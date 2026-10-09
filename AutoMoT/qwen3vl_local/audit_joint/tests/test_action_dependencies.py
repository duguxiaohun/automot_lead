"""Native Action reader integration, without replacing readers or contracts."""
import json
from pathlib import Path

import pytest

from qwen3vl_local.audit_joint.action_pool import export_pool, validate_export
from qwen3vl_local.audit_joint.io import file_sha
from qwen3vl_local.action_prior import event_balance as eb
from qwen3vl_local.action_prior.config import parser, read_rows
from qwen3vl_local.action_prior.phase3_stable.source_mapping import mapping_contract_hash
from qwen3vl_local.action_prior.phase3_stable.build_dataset import FRAME_INDEX_FORMAT
from qwen3vl_local.action_prior.phase3_stable.context_taxonomy import ACTION_KEYS
from qwen3vl_local.action_prior.phase3_stable.trajectory_action import ACTION_RULE_VERSION, action_rule_sha256


def write(path, value):
    path.write_text(json.dumps(value) + '\n')


def native_fixture(root, mode, token):
    for name in ('raw', 'data', 'mapping', 'candidate'):
        (root / name).mkdir()
    contract = mapping_contract_hash()
    for split in ('train', 'val', 'test'):
        row = dict(schema='action_prior_data_v1', scenario='OrdinaryFixture', run_id=split,
                   route_group=f'OrdinaryFixture/{split}', anchor=4, split=split,
                   tp_mode='route_lookahead', rgb_frame_count=4, rgb_frame_step=1)
        write(root / 'data' / f'{split}.jsonl', row)
        rgb = root / 'raw/OrdinaryFixture' / split / 'rgb'
        rgb.mkdir(parents=True)
        for frame in range(5):
            (rgb / f'{frame:04d}.jpg').write_bytes(b'frame-count fixture; reader does not decode images')
    candidate = root / 'candidate/candidate_frames.jsonl'
    write(candidate, dict(scenario='CandidateFixture', route_id='route', frame_id=4,
                         context_id='LEAD_BRAKE', mapping_contract_hash=contract,
                         action_labels={a: a == 'STOP' for a in ACTION_KEYS},
                         action_evidence=dict(rule_version=ACTION_RULE_VERSION, rule_code_sha256=action_rule_sha256(),
                                              longitudinal_decision=dict(eligible=True, action='STOP'),
                                              lateral_observation_complete=True, lane_change_direction='LEFT')))
    write(candidate.with_name('manifest.json'), dict(format=FRAME_INDEX_FORMAT, mapping_contract_hash=contract,
                                                    frame_index='frame_index.jsonl'))
    write(candidate.with_name('candidate_counts.json'), {'train/LEAD_BRAKE': 1})
    write(candidate.with_name('frame_index.jsonl'), {})
    index = root / 'mapping/full_event_mapping.jsonl'
    rows = [dict(schema=eb.FULL_INDEX_SCHEMA, mapping_contract_hash=contract, scenario='OrdinaryFixture',
                 route_id=split, frame_id=4, source_split=split, special_buckets=[], eligible_buckets=[],
                 scene_contexts=[], status=eb.CONFIRMED_REGULAR) for split in ('train', 'val', 'test')]
    index.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write(index.with_name('manifest.json'), dict(schema=eb.FULL_MANIFEST_SCHEMA, index_schema=eb.FULL_INDEX_SCHEMA,
            mapping_policy=eb.EVENT_BALANCE_MAPPING_POLICY, index_file=index.name, index_sha256=file_sha(index),
            mapping_contract_hash=contract, candidate_index='candidate/candidate_frames.jsonl',
            candidate_sha256=file_sha(candidate),
            action_dataset_hashes={s: file_sha(root / 'data' / f'{s}.jsonl') for s in ('train', 'val', 'test')}))
    return ['--data-root', 'raw', '--data-dir', 'data', '--event-balance-index', 'mapping/full_event_mapping.jsonl',
            '--sampling-mode', mode, '--high-level-action-token' if token else '--no-high-level-action-token']


@pytest.mark.parametrize('mode', ['event_balanced', 'action_balanced'])
@pytest.mark.parametrize('token', [False, True])
def test_real_reader_export_modes_cwd_relocation_and_candidate_changes(tmp_path, monkeypatch, mode, token):
    monkeypatch.chdir(tmp_path)
    argv = native_fixture(tmp_path, mode, token)
    args = parser().parse_args(argv)
    needs_candidate = token or mode == 'action_balanced'
    for split in ('train', 'val', 'test'):
        rows = read_rows(args, split)
        assert len(rows) == 1 and rows[0]['split'] == split
        assert ('action_token' in rows[0]) == needs_candidate
        if needs_candidate:
            assert rows[0]['action_token']['name'] == 'UNCOND'
    result = export_pool(argv, tmp_path / 'export')
    assert result['counts'] == dict(train=1, val=1, test=1)
    index = Path(result['index'])
    manifest = json.loads(index.with_name('effective_pool_manifest.json').read_text())
    assets = [tmp_path / 'candidate' / name for name in
              ('candidate_frames.jsonl', 'manifest.json', 'candidate_counts.json', 'frame_index.jsonl')]
    assert all((str(p) in manifest['input_sha256']) == needs_candidate for p in assets)
    assert all((str(p) in manifest['dependency_paths']['other_inputs']) == needs_candidate for p in assets)
    other = tmp_path / 'elsewhere'
    other.mkdir()
    monkeypatch.chdir(other)
    validate_export(manifest, index)
    (tmp_path / 'data').rename(tmp_path / 'moved')
    relocated = {s: str(tmp_path / 'moved' / f'{s}.jsonl') for s in ('train', 'val', 'test')}
    validate_export(manifest, index, relocated)
    for path in assets:
        original = path.read_bytes()
        for mutation in ('delete', 'change'):
            try:
                if mutation == 'delete':
                    path.unlink()
                else:
                    path.write_bytes(original + b'\n')
                if needs_candidate:
                    with pytest.raises((ValueError, FileNotFoundError)):
                        validate_export(manifest, index, relocated)
                else:
                    validate_export(manifest, index, relocated)
            finally:
                path.write_bytes(original)
        validate_export(manifest, index, relocated)
