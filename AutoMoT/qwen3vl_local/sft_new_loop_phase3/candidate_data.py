"""Explicit candidate-only exposure contract; frozen v23 defaults are unchanged."""
import hashlib
import json
from pathlib import Path

EXPOSURE = Path(__file__).with_name('development_route_groups_rgb_stage_20261006.json')


def exposure_contract(path):
    raw = Path(path).read_bytes()
    groups = json.loads(raw)['groups']
    if not isinstance(groups, list) or not groups or any(not isinstance(g, str) or '/' not in g for g in groups):
        raise ValueError('invalid extra development groups')
    return dict(groups=sorted(set(groups)), source_sha256=hashlib.sha256(raw).hexdigest())


def validate_candidate_index(index):
    manifest = json.loads(Path(index).with_name('manifest.json').read_text())
    expected = exposure_contract(EXPOSURE)
    if manifest.get('additional_development') != expected:
        raise ValueError('candidate training requires the RGB-audit exposure-isolated index; rebuild with --development-groups-extra')
    return expected
