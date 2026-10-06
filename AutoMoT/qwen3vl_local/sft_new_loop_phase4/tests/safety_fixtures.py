"""Explicit synthetic planner evidence for controller tests; never RGB labels."""
from qwen3vl_local.sft_new_loop_phase4.maneuver_safety import binding, GUARDED


def clearances(ep, frame, *edges):
    return [dict(**binding(ep, key, frame), source='planner', evidence_id=f'synthetic-test/{frame}/{key}',
                 clear=True, rear_side_coverage_confirmed=True)
            for key in edges if key in GUARDED and
            (ep.return_corridor if key == 'return' else ep.target_corridor) and
            (ep.return_direction if key == 'return' else ep.direction)]


def loop_clearances(loop, frame, *edges):
    return [r for ep in loop.episodes.values() for r in clearances(ep, frame, *edges)]
