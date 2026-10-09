"""Prepare small paired student views; optionally compile only native RGB4 from existing replay."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from ..identity import ROOT
from .data import create


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset4', type=Path, help='Existing complete native RGB4 dataset')
    source.add_argument('--phase4-base', type=Path, help='Existing source-bound full_production; never starts replay')
    p.add_argument('--data-root', type=Path, default=ROOT.parents[1] / 'lead_data')
    p.add_argument('--annotations', type=Path, default=ROOT / 'reviewed_state_pairs_v9.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('use a new output or reuse an already complete --dataset4; no automatic repeated copies')
    if a.phase4_base:
        from qwen3vl_local.audit_joint.recovery import check_recovery
        base = a.phase4_base.resolve()
        if a.output.resolve().is_relative_to(base):
            raise ValueError('output must be outside the original Phase4 base')
        report = check_recovery(base, a.data_root, a.annotations, min_free_bytes=20 * 1024 ** 3)
        if not report['replay_reusable_under_current_request']:
            raise ValueError('old replay is not reusable; inspect recovery-check diagnostics')
        if report['checks']['build_request_read']['result']['external']:
            raise ValueError('external production configuration requires an explicit complete --dataset4')
        import shutil
        ancestor = a.output.absolute().parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        free = shutil.disk_usage(ancestor).free
        minimum = report['storage']['production']['referenced_copy_bytes']
        if free < minimum + 20 * 1024 ** 3:
            raise ValueError('one production copy plus 20 GiB reserve does not fit; no build started')
        # Do not create a lock in the old base. Native ownership must already exist.
        if not (base / '.build.lock').is_file():
            raise ValueError('old build lock missing; use an already complete --dataset4')
        import fcntl
        with (base / '.build.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            a.output.mkdir(parents=True, exist_ok=False)
            a.dataset4 = a.output / 'data4'
            subprocess.run([sys.executable, '-m', 'qwen3vl_local.sft_new_loop_phase4.dataset',
                '--annotations', str(a.annotations), '--candidate-pool', str(base / 'candidates.json'),
                '--production-index', str(base / 'full_production/index.json'),
                '--teacher-registry', str(base / 'weak_registry.json'),
                '--max-teacher-questions-per-route', '0', '--data-root', str(a.data_root),
                '--output-dir', str(a.dataset4), '--rgb-mode', '4'], check=True)
    else:
        if a.output.resolve().is_relative_to(a.dataset4.resolve()):
            raise ValueError('output must be outside original dataset')
        a.output.mkdir(parents=True, exist_ok=False)
    report = create(a.dataset4, a.output / 'view')
    print(json.dumps(dict(view=str(a.output / 'view'), counts=report['counts'],
                         exclusions=len(report['exclusions']), production_copied_for_view=False,
                         observation_contracts=report['observation_contracts']), ensure_ascii=False))


if __name__ == '__main__':
    main()
