"""Navigation-grounded RE5 extension; the frozen v6 evaluation rubric stays intact."""
from dataclasses import replace
from .taxonomy import transitions, edge
from .identity import digest

BOUNDARY = 'route_exit_reached'
RECOVER_FOLLOW = edge('recover_follow', 'longitudinal', 'RECOVER', 'FOLLOW', 'milestone',
    'restricted_progress_established',
    'Has recovery settled into an established moving following gap, without a renewed conflict?')


def validate_route_context(ep):
    context = ep.route_context
    if not isinstance(context, dict):
        raise ValueError('route context must be a mapping')
    if not context:
        return
    if (ep.event != 'R-E5' or ep.branch != 'default'
            or set(context) != {'kind', 'completion_boundary', 'navigation_evidence_id'}
            or context['kind'] != 'roundabout'
            or any(not isinstance(context[k], str) or not context[k].strip()
                   or '\n' in context[k] or '\r' in context[k]
                   for k in context)
            or not isinstance(ep.target_corridor, str) or not ep.target_corridor.strip()
            or '\n' in ep.target_corridor or '\r' in ep.target_corridor):
        raise ValueError('roundabout requires an explicit target exit, completion boundary and navigation evidence')


def route_bound(ep):
    return ep.event == 'R-E5' and bool(ep.route_context or ep.target_corridor or ep.direction)


def route_identity(ep):
    return digest(dict(target=ep.target_corridor, direction=ep.direction,
                       context=ep.route_context)) if route_bound(ep) else None


def episode_edges(ep):
    validate_route_context(ep)
    edges = (*transitions(ep.event, ep.branch, ep.return_required), RECOVER_FOLLOW)
    if not ep.route_context:
        return edges
    # Completing the geometry must not require releasing an independent STOP.
    return tuple(replace(e, criteria=('event_resolved', BOUNDARY))
                 if e.key == 'complete' else e for e in edges)


def episode_edge(ep, key):
    matches = [e for e in episode_edges(ep) if e.key == key]
    if len(matches) != 1:
        raise ValueError('unknown transition')
    return matches[0]


def validate_source_context(record):
    """Known roundabout examples cannot silently fall back to junction semantics.

    This list is an admission guard, not a navigation/label producer. New sources
    still require the upstream event tracker to declare their maneuver context.
    """
    import json
    from .identity import ROOT
    if record['episode']['event'] != 'R-E5':
        return
    ledger = json.loads((ROOT/'eighteenth_nineteenth_audit_exposure_20261001.json').read_text())
    required = {(r['scenario'], r['route_id']) for r in ledger['navigation_context_required_routes']}
    if (record['scenario'], record['route_id']) in required and not record['episode'].get('route_context'):
        raise ValueError('audited roundabout source requires target exit and instance boundary')
