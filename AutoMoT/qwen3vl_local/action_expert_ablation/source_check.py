"""CPU-only availability check for external Action ablation sources; no model imports."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_SOURCES = (
    'leaderboard/team_code/mot_lead_offline_runner.py',
    'Automot/mot/modeling/bev_encoder/bev_encoder_utils.py',
)


def require_sources(root, paths):
    """Fail before GPU selection/data/model loading; never omit absent source hashes."""
    missing = [str(Path(root) / name) for name in dict.fromkeys(paths)
               if not (Path(root) / name).is_file()]
    if missing:
        raise FileNotFoundError('Action ablation required sources missing:\n  '
            + '\n  '.join(missing)
            + '\nRestore the original sources from the matching execution environment. '
              'Do not substitute placeholders or remove contract checks. '
              'Run python -m qwen3vl_local.action_expert_ablation.source_check for a CPU-only report.')


def report(root):
    files = []
    for name in EXTERNAL_SOURCES:
        path = Path(root) / name
        item = dict(path=name, exists=path.is_file())
        if item['exists']:
            try:
                item['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                item['error'] = f'{type(exc).__name__}: {exc}'
        files.append(item)
    ready = all(item['exists'] and 'error' not in item for item in files)
    return dict(status='sources_present' if ready else 'missing_or_unreadable_sources',
                root=str(Path(root).absolute()), files=files,
                scope='external source availability and content hashes only; not import, model, GPU or end-to-end acceptance')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    args = p.parse_args()
    result = report(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status']=='sources_present' else 2


if __name__=='__main__':
    raise SystemExit(main())
