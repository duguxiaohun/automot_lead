"""Stable release selection, tamper rejection and experiment isolation without GPUs."""
import json
import os
import shutil
import subprocess
import sys

import pytest

from qwen3vl_local.action_prior import phase3_release as releases


def test_historical_v23_matches_restored_phase3():
    from qwen3vl_local.sft_new_loop_phase3 import source_mapping, trajectory_action, prompts
    from qwen3vl_local.action_prior.phase3_stable import source_mapping as stable_mapping
    from qwen3vl_local.action_prior.phase3_stable import trajectory_action as stable_rules
    from qwen3vl_local.action_prior.phase3_stable import prompts as stable_prompts
    assert releases.identity()['release'] == 'v23_io1'
    assert releases.identity()['semantic_release'] == 'v23'
    assert stable_mapping.mapping_contract_hash() == source_mapping.mapping_contract_hash()
    assert stable_rules.action_rule_sha256() == trajectory_action.action_rule_sha256()
    assert stable_prompts.PROMPT_NAME == prompts.PROMPT_NAME
    from qwen3vl_local.action_prior.phase3_stable import build_dataset
    assert len(build_dataset.development_route_groups()) == 1779


@pytest.fixture
def candidate(tmp_path):
    directory, _, selection = releases.active_release()
    destination = tmp_path / 'v25'
    shutil.copytree(directory, destination, ignore=shutil.ignore_patterns('__pycache__'))
    manifest = json.loads((destination / 'release.json').read_text())
    manifest['release'] = 'v25'
    (destination / 'release.json').write_text(json.dumps(manifest))
    registry = tmp_path / 'stable.json'
    registry.write_text(json.dumps(selection))
    report = tmp_path / 'paired_evaluation.json'
    report.write_text(json.dumps({'fixture': 'synthetic promotion test, not model evidence'}))
    review = dict(decision='promote', release='v25', baseline_release=selection['release'],
                  manifest_sha256=releases.sha256(destination / 'release.json'),
                  same_evaluation_cases=True, regressions_reviewed=True,
                  approved_for_action=True, reviewer='test fixture', evaluation_runs=['baseline', 'candidate'],
                  acceptance_metric=dict(name='macro', baseline=0.60, candidate=0.65),
                  report_path=str(report), report_sha256=releases.sha256(report))
    return tmp_path, registry, review


@pytest.mark.parametrize('field,value', [
    ('same_evaluation_cases', False), ('regressions_reviewed', False),
    ('approved_for_action', False), ('evaluation_runs', []),
    ('baseline_release', 'v22'), ('manifest_sha256', 'changed'),
    ('report_sha256', 'changed'), ('reviewer', ''),
    ('acceptance_metric', dict(name='macro', baseline=0.65, candidate=0.65)),
    ('acceptance_metric', dict(name='macro', baseline=0.65, candidate=0.64)),
    ('acceptance_metric', dict(name='macro', baseline=0.60, candidate=float('nan'))),
])
def test_unverified_or_worse_candidate_cannot_promote(candidate, field, value):
    root, registry, review = candidate
    before = registry.read_bytes()
    review[field] = value
    with pytest.raises(ValueError):
        releases.promote('v25', review, releases=root, registry=registry)
    assert registry.read_bytes() == before


def test_only_reviewed_candidate_changes_selection(candidate):
    root, registry, review = candidate
    result = releases.promote('v25', review, releases=root, registry=registry)
    assert result == json.loads(registry.read_text())
    assert result['release'] == 'v25'
    # This process, including its builder children, stays on the original snapshot.
    assert releases.identity()['release'] == 'v23_io1'
    assert releases.identity()['semantic_release'] == 'v23'
    assert json.loads(releases.builder_environment()['AUTOMOT_PHASE3_STABLE_SELECTION'])['release'] == 'v23_io1'


def test_frozen_source_change_is_rejected(candidate):
    root, registry, review = candidate
    with (root / 'v25' / 'choice_semantics.py').open('a') as stream:
        stream.write('\n# unreviewed edit\n')
    with pytest.raises(ValueError, match='stable release changed'):
        releases.promote('v25', review, releases=root, registry=registry)


def test_actions_import_without_any_experimental_phase3_module():
    script = '''
import importlib.abc, sys
class BlockExperiment(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('qwen3vl_local.sft_new_loop_phase3'):
            raise AssertionError('Action imported experimental Phase3: ' + fullname)
sys.meta_path.insert(0, BlockExperiment())
from qwen3vl_local.action_prior import action_input, action_token, action_balance
from qwen3vl_local.action_prior import prepare_event_balance, prepare_action_priors, build_event_balance_index
from qwen3vl_local.action_prior.phase3_stable import build_dataset, source_mapping, prompts
from qwen3vl_local.action_expert_ablation import common
from qwen3vl_local.action_prior.phase3_release import contract_source_paths
from qwen3vl_local.action_prior.phase3_release import identity
assert source_mapping.mapping_contract_hash() == identity()['mapping_contract_hash']
assert identity()['semantic_release'] == 'v23'
assert all('sft_new_loop_phase3/' not in p for p in contract_source_paths())
for variant in ('qwen_simple', 'bev_only'):
    assert all('sft_new_loop_phase3/' not in p for p in common.contract_source_paths(variant))
'''
    subprocess.run([sys.executable, '-c', script], check=True,
                   env=dict(os.environ, PYTHONPATH=str(releases.ROOT)))


def test_real_stable_builder_entrypoint():
    result = subprocess.run([sys.executable, str(releases.source_path('build_dataset.py')), '--help'],
                            env=releases.builder_environment(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--output-dir' in result.stdout


@pytest.fixture
def engineering_candidate(tmp_path):
    for name in ('v23', 'v23_io1'):
        shutil.copytree(releases.RELEASES / name, tmp_path / name,
                        ignore=shutil.ignore_patterns('__pycache__'))
    registry = tmp_path / 'stable.json'
    registry.write_text(json.dumps({'release': 'v23'}))
    review = json.loads(releases.REGISTRY.read_text())['review']
    return tmp_path, registry, review


def test_engineering_fix_does_not_require_claiming_model_improvement(engineering_candidate):
    root, registry, review = engineering_candidate
    assert 'acceptance_metric' not in review and review['decision'] == 'engineering_fix'
    result = releases.promote('v23_io1', review, releases=root, registry=registry)
    assert result['release'] == 'v23_io1'
    assert json.loads(registry.read_text())['review']['fix_kind'] == 'estale_publication'


@pytest.mark.parametrize('field,value', [
    ('semantic_outputs_unchanged', False), ('regression_runs', []),
    ('changed_files', []), ('fix_kind', 'new_prompt'), ('report_sha256', 'bad'),
])
def test_engineering_fix_requires_its_own_evidence(engineering_candidate, field, value):
    root, registry, review = engineering_candidate
    before = registry.read_bytes()
    review[field] = value
    with pytest.raises(ValueError):
        releases.promote('v23_io1', review, releases=root, registry=registry)
    assert registry.read_bytes() == before


@pytest.mark.parametrize('name', ['prompts.py', 'trajectory_action.py', 'sampling.py',
                                  'mapping_rgb_decisions_v2.jsonl'])
def test_engineering_fix_cannot_smuggle_semantic_changes(engineering_candidate, name):
    root, registry, review = engineering_candidate
    directory = root / 'v23_io1'
    path = directory / name
    path.write_bytes(path.read_bytes() + b'\n')
    manifest_path = directory / 'release.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'][name] = releases.sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    review['manifest_sha256'] = releases.sha256(manifest_path)
    review['changed_files'].append(name)
    with pytest.raises(ValueError, match='protected semantic'):
        releases.promote('v23_io1', review, releases=root, registry=registry)
