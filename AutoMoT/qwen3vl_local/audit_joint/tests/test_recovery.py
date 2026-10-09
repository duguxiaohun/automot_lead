import json
import sys

import pytest

from qwen3vl_local.audit_joint.io import file_sha
from qwen3vl_local.audit_joint.recovery import check_recovery
from qwen3vl_local.sft_new_loop_phase4 import full_pipeline, teacher_replay, candidate_pool
from qwen3vl_local.sft_new_loop_phase4.identity import digest


@pytest.fixture
def replay(tmp_path):
    """Native frame accounting, request/registry validation; no GPU or synthetic mocks of these checks."""
    base = tmp_path / 'phase4'
    base.mkdir()
    production = base / 'full_production'
    production.mkdir()
    ann = tmp_path / 'annotations.json'
    ann.write_text('[]')
    (base / 'build_request.json').write_text(json.dumps(full_pipeline.request_identity(tmp_path, ann)))
    (base / 'weak_registry.json').write_text(json.dumps(full_pipeline.default_registry()))
    raw = tmp_path / 'test/Town01_Rep0_1_0_route0_fixture'
    for kind, extension in [('rgb', 'jpg'), ('metas', 'pkl')]:
        (raw / kind).mkdir(parents=True)
        for frame in (4, 5):
            (raw / kind / f'{frame:04d}.{extension}').touch()
    pool = candidate_pool.scan(tmp_path, base / 'candidates.json')
    route = pool['routes'][0]
    header = dict(policy=teacher_replay.POLICY, teacher=teacher_replay.rules.identity(), candidate_pool_sha256=pool['sha256'])
    rows = [dict(kind='header', **header)]
    for frame in route['rgb_frames']:
        rows.append(dict(kind='frame_disposition', **{k: route[k] for k in ('scenario', 'route_id', 'physical_group', 'split')},
                         frame_id=frame, disposition='irrelevant', reasons=['fixture'], questions=[]))
    artifact = production / 'route.jsonl'
    artifact.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    receipt = teacher_replay.inspect_route(artifact, route, header)
    index = dict(**header, routes=[receipt])
    index['sha256'] = digest(index)
    (production / 'index.json').write_text(json.dumps(index))
    (base / 'data2/production').mkdir(parents=True)
    (base / 'data2/production/partial.jsonl').write_text('old partial evidence')
    return base, tmp_path, ann


def run_probe(replay):
    return check_recovery(*replay)


def snapshot(base):
    return {str(p.relative_to(base)): file_sha(p) for p in base.rglob('*') if p.is_file()}


def test_valid_native_replay_read_only_and_space_is_only_lower_bound(replay):
    base, _, _ = replay
    before = snapshot(base)
    report = run_probe(replay)
    assert report['replay_reusable_under_current_request']
    assert report['checks']['replay_accounting']['result']['processed_frames'] == 2
    assert report['checks']['paired_pipeline_receipt']['status'] == 'invalid'
    assert not report['training_ready']
    assert report['datasets']['data2']['native_next_action'] == 'retain_by_rename_then_recompile'
    assert report['storage']['compile_modes'] == [2, 4]
    assert report['storage']['additional_production_copy_bytes_lower_bound'] == sum(
        p.stat().st_size for p in (base / 'full_production').iterdir()) * 2
    assert report['storage']['sufficient_for_build'] is None
    assert snapshot(base) == before
    assert not (base / '.build.lock').exists()


@pytest.mark.parametrize('damage', ['changed_route', 'missing_route', 'empty_route', 'missing_frame', 'incomplete_index'])
def test_replay_drift_or_partial_frames_rejected(replay, damage):
    base, _, _ = replay
    artifact = base / 'full_production/route.jsonl'
    if damage == 'missing_route':
        artifact.unlink()
    elif damage == 'empty_route':
        artifact.write_text('')
    elif damage == 'missing_frame':
        artifact.write_text('\n'.join(artifact.read_text().splitlines()[:-1]) + '\n')
    elif damage == 'incomplete_index':
        path = base / 'full_production/index.json'
        index = json.loads(path.read_text())
        index['routes'] = []
        index['sha256'] = digest({k: v for k, v in index.items() if k != 'sha256'})
        path.write_text(json.dumps(index))
    else:
        artifact.write_text(artifact.read_text().replace('fixture', 'changed'))
    report = run_probe(replay)
    assert report['status'] == 'blocked'
    assert report['checks']['replay_accounting']['status'] == 'invalid'


@pytest.mark.parametrize('content', ['{', '[]', '{"external": []}', '{"external": {"production_index": null}}'])
def test_invalid_request_still_reports_dataset_and_disk_state(replay, content):
    (replay[0] / 'build_request.json').write_text(content)
    report = run_probe(replay)
    assert report['status'] == 'blocked'
    assert report['checks']['build_request_read']['status'] == 'invalid'
    assert report['datasets']['data2']['exists']
    assert report['storage']['sufficient_for_build'] is None


def test_annotation_change_blocks_reuse_without_discarding_replay_evidence(replay):
    replay[2].write_text('["changed"]')
    report = run_probe(replay)
    assert report['checks']['request_compatibility']['status'] == 'invalid'
    assert report['checks']['replay_accounting']['status'] == 'verified'
    assert report['status'] == 'blocked'


def test_native_candidate_validation_is_required(replay):
    path = replay[0] / 'candidates.json'
    pool = json.loads(path.read_text())
    pool['sha256'] = 'changed'
    path.write_text(json.dumps(pool))
    report = check_recovery(*replay)
    assert report['checks']['candidate_pool']['status'] == 'invalid'
    assert report['checks']['replay_accounting']['status'] == 'invalid'
    assert report['status'] == 'blocked'


def test_active_native_builder_rejected(replay):
    with full_pipeline.build_lock(replay[0]):
        with pytest.raises(ValueError, match='builder is active'):
            check_recovery(*replay)


def test_cli_preserves_existing_report_and_refuses_output_in_input_tree(replay, monkeypatch):
    from qwen3vl_local.audit_joint.__main__ import main
    base, data_root, _ = replay
    output = base / 'new_report.json'
    monkeypatch.setattr(sys, 'argv', ['audit', 'recovery-check', '--data-dir', str(base),
                                    '--data-root', str(data_root), '--output', str(output)])
    with pytest.raises(SystemExit) as ex:
        main()
    assert ex.value.code == 2
    assert not output.exists()
