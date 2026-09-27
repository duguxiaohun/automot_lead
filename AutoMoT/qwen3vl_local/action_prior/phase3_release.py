"""Select an audited Phase3 release; experimental source is never an Action fallback."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from functools import lru_cache

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RELEASES = HERE / 'phase3_releases'
REGISTRY = RELEASES / 'stable.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _release_dir(release, releases=RELEASES):
    if not isinstance(release, str) or not re.fullmatch(r'v[0-9]+(?:_[a-z0-9]+)*', release):
        raise ValueError('invalid Phase3 release id')
    return Path(releases) / release


def verify_release(release, releases=RELEASES):
    directory = _release_dir(release, releases)
    manifest = json.loads((directory / 'release.json').read_text())
    if manifest.get('schema') != 'action_phase3_release_v1' or manifest.get('release') != release:
        raise ValueError('invalid Phase3 release manifest')
    files = manifest.get('files', {})
    if not {'__init__.py', 'build_dataset.py', 'prompts.py', 'sampling.py', 'source_mapping.py',
            'trajectory_action.py', 'choice_semantics.py'} <= files.keys():
        raise ValueError('incomplete Phase3 release')
    for name, expected in files.items():
        if Path(name).name != name or sha256(directory / name) != expected:
            raise ValueError(f'Phase3 stable release changed: {release}/{name}; do not edit frozen releases')
    for name, expected in manifest.get('external_dependencies', {}).items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != expected:
            raise ValueError(f'Phase3 stable dependency changed: {name}; validate a new release')
    return directory, manifest


def _verify_review(review, release, manifest_sha, baseline):
    if (review.get('decision') != 'promote' or review.get('release') != release
            or review.get('manifest_sha256') != manifest_sha
            or review.get('baseline_release') != baseline
            or review.get('same_evaluation_cases') is not True
            or review.get('regressions_reviewed') is not True
            or review.get('approved_for_action') is not True
            or not review.get('reviewer') or not review.get('evaluation_runs')):
        raise ValueError('promotion requires reviewed paired evaluation tied to this release and current baseline')
    metric = review.get('acceptance_metric', {})
    before, after = metric.get('baseline'), metric.get('candidate')
    if (not metric.get('name') or type(before) not in (int, float)
            or type(after) not in (int, float) or not math.isfinite(before)
            or not math.isfinite(after) or after <= before):
        raise ValueError('promotion requires an improved, higher-is-better acceptance metric')
    _verify_report(review)


def _verify_report(review):
    report = Path(review.get('report_path', ''))
    if not report.is_absolute():
        report = ROOT / report
    if not report.is_file() or sha256(report) != review.get('report_sha256'):
        raise ValueError('promotion evaluation report missing or changed')


def _verify_engineering_review(review, release, manifest_sha, baseline, releases=RELEASES):
    """Engineering maintenance is separate from a model-quality promotion.

    This gate supports the reviewed ESTALE backport only. Changes to semantics
    need the paired-evaluation path even if their author calls them a bug fix.
    Allowed builder edits still need code review and output-equivalence tests.
    """
    if (review.get('decision') != 'engineering_fix'
            or review.get('fix_kind') != 'estale_publication'
            or review.get('release') != release or review.get('baseline_release') != baseline
            or review.get('manifest_sha256') != manifest_sha
            or review.get('semantic_outputs_unchanged') is not True
            or review.get('regressions_reviewed') is not True
            or review.get('approved_for_action') is not True
            or not review.get('reviewer') or not review.get('regression_runs')):
        raise ValueError('engineering fix requires reviewed regression and semantic-equivalence evidence')
    _verify_report(review)
    _, parent = verify_release(baseline, releases)
    _, candidate = verify_release(release, releases)
    if (candidate.get('engineering_base_release') != baseline
            or candidate.get('semantic_release') != parent.get('semantic_release', parent['release'])
            or candidate.get('origin_commit') != parent.get('origin_commit')
            or candidate.get('origin_mapping_contract_hash') != parent.get('origin_mapping_contract_hash')
            or candidate.get('original_files') != parent.get('original_files')):
        raise ValueError('engineering fix must retain its validated semantic baseline')
    allowed = {'build_dataset.py', 'parallel_scan.py', 'source_mapping.py'}
    if candidate['files'].keys() != parent['files'].keys():
        raise ValueError('engineering fix cannot add/remove frozen semantic resources')
    changed = {name for name in parent['files'] if parent['files'][name] != candidate['files'][name]}
    if not changed <= allowed:
        raise ValueError('engineering fix changed protected semantic files; use performance promotion')
    if sorted(changed) != sorted(review.get('changed_files', [])):
        raise ValueError('engineering review must enumerate every changed file')
    previous_deps = parent.get('external_dependencies', {})
    next_deps = candidate.get('external_dependencies', {})
    if (any(next_deps.get(name) != digest for name, digest in previous_deps.items())
            or not (set(next_deps) - set(previous_deps)) <= {'qwen3vl_local/action_prior/filesystem.py'}):
        raise ValueError('engineering fix changed unrelated external dependencies')


@lru_cache(maxsize=1)
def active_release():
    """Pin once per process; the next new process observes a promoted release."""
    selection = json.loads(os.environ.get('AUTOMOT_PHASE3_STABLE_SELECTION') or REGISTRY.read_text())
    release = selection.get('release')
    directory, manifest = verify_release(release)
    digest = sha256(directory / 'release.json')
    if selection.get('schema') != 'action_phase3_stable_v1' or selection.get('manifest_sha256') != digest:
        raise ValueError('Phase3 stable registry/manifest mismatch')
    review = selection.get('review', {})
    if review.get('decision') == 'accepted_historical_baseline':
        if (release != 'v23' or manifest.get('origin_commit') !=
                'b433aa605251130e755967c53d89f98966a5525f'
                or review.get('approved_for_action') is not True):
            raise ValueError('invalid historical baseline exception')
    elif review.get('decision') == 'engineering_fix':
        _verify_engineering_review(review, release, digest, review.get('baseline_release'))
    else:
        _verify_review(review, release, digest, review.get('baseline_release'))
    return directory, manifest, selection


def identity():
    _, manifest, selection = active_release()
    return dict(release=manifest['release'], origin_commit=manifest['origin_commit'],
                semantic_release=manifest.get('semantic_release', manifest['release']),
                engineering_revision=manifest.get('engineering_revision'),
                mapping_contract_hash=manifest.get('mapping_contract_hash', manifest['origin_mapping_contract_hash']),
                origin_mapping_contract_hash=manifest['origin_mapping_contract_hash'],
                manifest_sha256=selection['manifest_sha256'],
                selection='reviewed_stable_release_not_latest_experiment')


def source_path(name):
    directory, manifest, _ = active_release()
    if name not in manifest['files']:
        raise ValueError(f'file not in selected Phase3 release: {name}')
    return directory / name


def contract_source_paths():
    directory, manifest, _ = active_release()
    paths = [Path(__file__), HERE/'phase3_stable/__init__.py', directory/'release.json']
    paths.extend(directory/name for name in manifest['files'])
    paths.extend(ROOT/name for name in manifest.get('external_dependencies', {}))
    return sorted({str(p.relative_to(ROOT)) for p in paths})


def builder_environment():
    """Keep builder subprocesses on their parent's verified release across promotion."""
    return dict(os.environ, AUTOMOT_PHASE3_STABLE_SELECTION=json.dumps(active_release()[2]))


def promote(release, review, *, releases=RELEASES, registry=REGISTRY):
    """Only publication step; does not change experiments, caches or existing runs."""
    registry = Path(registry)
    previous_bytes = registry.read_bytes()
    previous = json.loads(previous_bytes)
    directory, _ = verify_release(release, releases)
    digest = sha256(directory/'release.json')
    if review.get('decision') == 'engineering_fix':
        _verify_engineering_review(review, release, digest, previous['release'], releases)
    else:
        _verify_review(review, release, digest, previous['release'])
    if release == previous['release']:
        raise ValueError('release is already stable')
    selection = dict(schema='action_phase3_stable_v1',release=release,
                     manifest_sha256=digest,review=review)
    # Serialize concurrent promotion attempts; never remove another process's lock.
    import fcntl
    with registry.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if registry.read_bytes() != previous_bytes:
            raise ValueError('stable release changed during review; compare against the new baseline')
        temporary = registry.with_suffix('.tmp')
        temporary.write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
        os.replace(temporary,registry)
    return selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['show','promote'])
    parser.add_argument('--release')
    parser.add_argument('--review',type=Path)
    args=parser.parse_args()
    if args.command=='show':
        print(json.dumps(identity(),ensure_ascii=False,indent=2))
    else:
        if not args.release or not args.review:
            parser.error('promote requires --release and --review')
        print(json.dumps(promote(args.release,json.loads(args.review.read_text())),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
