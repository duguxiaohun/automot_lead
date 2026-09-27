"""Compare v23 and its I/O-only maintenance revision using real builders."""
import copy
import errno
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from qwen3vl_local.action_prior import phase3_release as releases, filesystem


def load_builder(directory):
    spec = importlib.util.spec_from_file_location('io_backport_builder', directory / 'build_dataset.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('stale_target', [None, 'candidate_frames.jsonl', 'candidate_counts.json',
    'frame_index.jsonl', 'split_coverage.json', 'same_rs_invalid_candidates.jsonl', 'manifest.json'])
@pytest.mark.parametrize('committed', [False, True])
def test_v23_io_outputs_equal_baseline_even_after_estale(tmp_path, monkeypatch, stale_target, committed):
    from qwen3vl_local.sft_new_loop_phase3.test_build_invalid_quota import candidates
    from qwen3vl_local.action_prior.phase3_stable import same_rs_invalid, trajectory_action
    bases, _ = candidates()
    for base in bases:
        base.update(split='train', mapping_contract_hash='controlled-same-contract')
        base['action_evidence'].update(
            longitudinal_decision=trajectory_action.longitudinal_decision([8, 7, 6, 5, 5, 5, 5, 5, 5]),
            lateral_observation_complete=True, lane_change_direction='')
    monkeypatch.setattr(same_rs_invalid, 'reviewed_invalid_rows', lambda *a: [])
    monkeypatch.setattr(filesystem, 'ESTALE_DELAYS', (0, 0, 0, 0))
    results = []
    for revision in ('v23', 'v23_io1'):
        builder = load_builder(releases.RELEASES / revision)
        output = tmp_path / revision
        monkeypatch.setattr(builder, 'iter_base_frames', lambda *a, **k: iter(copy.deepcopy(bases)))
        monkeypatch.setattr(builder, 'mapping_contract_hash', lambda: 'controlled-same-contract')
        monkeypatch.setattr(builder, 'load_review_coverage', lambda **k: (
            {'synthetic': {'Town01': {'completed_routes': 1}}}, 'synthetic-input-only'))
        monkeypatch.setattr(sys, 'argv', ['builder', '--output-dir', str(output),
                                        '--val-ratio', '0', '--test-ratio', '0'])
        original_replace, faults = Path.replace, []

        def replace(source, target):
            if revision == 'v23_io1' and Path(target).name == stale_target and not faults:
                faults.append(target)
                if committed:
                    original_replace(source, target)
                raise OSError(errno.ESTALE, 'publication fault')
            return original_replace(source, target)

        with monkeypatch.context() as patch:
            patch.setattr(Path, 'replace', replace)
            builder.build_dataset(builder.parse_args())
        artifacts = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
        manifest = json.loads(artifacts.pop('manifest.json'))
        manifest.pop('frame_index')  # Only the output location differs in this paired test.
        results.append((artifacts, manifest))
        if revision == 'v23_io1':
            assert len(faults) == int(stale_target is not None)
    assert results[0] == results[1]


def test_semantic_files_are_unchanged_in_io_revision():
    old_dir, old = releases.verify_release('v23')
    new_dir, new = releases.verify_release('v23_io1')
    changed = {n for n in old['files'] if (old_dir / n).read_bytes() != (new_dir / n).read_bytes()}
    assert changed == {'build_dataset.py', 'parallel_scan.py', 'source_mapping.py'}
    assert old['original_files'] == new['original_files']
    assert new['semantic_release'] == 'v23'
