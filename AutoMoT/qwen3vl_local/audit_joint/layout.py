"""Resolve Git and application roots without depending on the caller's cwd."""
from pathlib import Path
import subprocess


def project_layout(project_root=None, *, module_file=None):
    if project_root is None:
        application = Path(module_file or __file__).resolve().parents[2]
        try:
            result = subprocess.run(['git', '-C', str(application), 'rev-parse', '--show-toplevel'],
                                    check=True, capture_output=True, text=True, timeout=30)
            root = Path(result.stdout.strip()).resolve()
        except (OSError, subprocess.SubprocessError):
            # Source-only copies can still produce a blocked diagnostic bundle.
            root = application
    else:
        root = Path(project_root).resolve()
    candidates = [p for p in (root, root / 'AutoMoT') if (p / 'qwen3vl_local').is_dir()]
    if len(candidates) != 1:
        raise ValueError('project root must contain exactly one application layout: '
                         'qwen3vl_local/ or AutoMoT/qwen3vl_local/; '
                         'set --project-root explicitly and prepare a new request: ' + str(root))
    return root, candidates[0]
