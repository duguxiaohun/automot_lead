"""Shared evidence-bound source exclusions; never converts NO to YES.

Phase3 checks both the visible history and the action-label horizon. Phase4
checks causal input and older instance/STOP evidence. Future defects cannot
invalidate an earlier Phase4 condition. Old datasets keep their original hashes.
"""
import json
from pathlib import Path

from qwen3vl_local.sft_new_loop_phase4.identity import file_sha, digest
from qwen3vl_local.sft_new_loop_phase4.data_paths import data_path

LEDGER = Path(__file__).with_name('label_quarantine_20261010.json')


def load(root, path=LEDGER, *, route=None):
    ledger = json.loads(Path(path).read_text())
    if ledger.get('policy') != 'local_source_quarantine_v1':
        raise ValueError('unsupported source quarantine policy')
    records = []
    seen = set()
    for raw in ledger['records']:
        r = dict(raw)
        if route is not None and (r['scenario'], r['route_id']) != tuple(route):
            continue
        before, after = r['last_unaffected_frame'], r['first_affected_frame']
        if (type(before) is not int or type(after) is not int or before < 0
                or after != before + 1 or not r.get('reason')
                or r.get('kind') != 'confirmed_visible_actor_disappearance'
                or r.get('independent_approval') is not False):
            raise ValueError('invalid confirmed source discontinuity')
        expected = {(f, k) for f in (before, after) for k in ('rgb', 'metas', 'bboxes')}
        if (len(r['sources']) != 6
                or {(s['frame_id'], s['kind']) for s in r['sources']} != expected):
            raise ValueError('quarantine requires before/after RGB and geometry')
        for source in r['sources']:
            suffix = 'jpg' if source['kind'] == 'rgb' else 'pkl'
            expected_path = f"{r['scenario']}/{r['route_id']}/{source['kind']}/{source['frame_id']:04d}.{suffix}"
            if source['path'] != expected_path or file_sha(data_path(root, expected_path)) != source['sha256']:
                raise ValueError('quarantine source path/SHA mismatch')
        key = r['scenario'], r['route_id'], after, r['actor_id']
        if key in seen:
            raise ValueError('duplicate source discontinuity')
        seen.add(key)
        r['risk_id'] = digest(raw)
        records.append(r)
    return records


def affected(records, scenario, route_id, first, last):
    if type(first) is not int or type(last) is not int or first > last:
        raise ValueError('invalid evidence envelope')
    return [r['risk_id'] for r in records
            if (r['scenario'], r['route_id']) == (scenario, route_id)
            and first <= r['first_affected_frame'] <= last]


def phase3_risks(row, records):
    if not records:
        return []
    from qwen3vl_local.sft_new_loop_phase3.trajectory_action import (
        LONGITUDINAL_HORIZON_FRAMES, LATERAL_HORIZON_FRAMES, longitudinal_decision, validate_action_rule)
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID
    frame = row['frame_id']
    # The native candidate stores four images even for a two-image student view.
    # Use their common envelope to keep paired selection, including gaps.
    full = CONTEXT_BY_ID[row['context_id']].question_domain == 'FULL_MANEUVER'
    horizon = LONGITUDINAL_HORIZON_FRAMES
    evidence = row.get('action_evidence', {})
    if evidence:
        validate_action_rule(row)
        # With controls still held, a confirmed current stop cannot become
        # pullaway because of a later disappearance. Preserve these clear STOPs.
        decision = longitudinal_decision(evidence['future_speeds_exact_mps'],
            brake=evidence.get('brake'), throttle=evidence.get('throttle'))
        if (decision['reason'] == 'current_confirmed_wait'
                and decision['anchor_control_released'] is False):
            horizon = 1
    # Native lateral_window_issue and lane_change also read the confirmation
    # sample after the final possible crossing, i.e. t+13 rather than t+12.
    if full:
        horizon = max(horizon, LATERAL_HORIZON_FRAMES + 1)
    return affected(records, row['scenario'], row['route_id'], frame - 3, frame + horizon)


def phase4_risks(row, records):
    """Accept a native teacher question or a compiled row, preserving its target."""
    if row.get('teacher_provenance'):
        return phase4_risks(row['teacher_provenance']['question'], records)
    frames = list(row.get('history_frames') or row.get('observation', {}).get('history_frames', []))
    frames += [s['frame_id'] for key in ('causal_sources', 'instance_sources') for s in row.get(key, [])]
    frames += [s['frame_id'] for duty in row.get('control_evidence', {}).values()
               for observation in duty['observations'] for s in observation['sources']]
    if not frames:
        raise ValueError('missing causal evidence envelope')
    return affected(records, row['scenario'], row['route_id'], min(frames), max(frames))
