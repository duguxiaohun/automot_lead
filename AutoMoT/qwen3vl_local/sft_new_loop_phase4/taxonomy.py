"""十类事件的最小流程；条件问题与动作派生使用同一转移表。"""
from dataclasses import dataclass

UE3_RELEASE_CONDITION = ('the intrusion no longer prevents progress along the established ego travel corridor: '
    'either a usable following gap exists or sufficient space permits passing alongside the intruding vehicle; '
    'continued lateral intrusion or conflicting approaching traffic still prevents passage')

EVENTS = {
    'U-E1': ('lead vehicle braking response', 'yield', 'The lead vehicle motion and following gap support progress; its continued presence is not a blocker.'),
    'U-E2': ('static blockage', 'bypass', 'Use the available bypass corridor; passing the obstacle and re-entry clearance are separate requirements.'),
    'U-E3': ('vehicle cutting into the ego path', 'yield', UE3_RELEASE_CONDITION),
    'U-E4': ('pedestrian or cyclist conflict', 'yield', 'Check the ego travel corridor and participant motion, including later crossing participants; full street crossing is not required.'),
    'U-E5': ('oncoming vehicle invading the ego corridor', 'yield', 'Check the entire required corridor and following oncoming vehicles, not only the first vehicle passing.'),
    'U-E6': ('junction priority conflict', 'junction', 'Check all relevant crossing paths and priority before entering; being inside the junction does not establish clearance.'),
    'U-E7': ('established traffic signal malfunction', 'junction', 'The malfunction is established upstream; negotiate priority and traffic gaps, without waiting for signal repair.'),
    'R-E2': ('route lane transition', 'lane', 'Use the established target corridor; this may be navigation or bypass recovery, without asserting unseen bypass history.'),
    'R-E3': ('active merge or exit', 'merge', 'Use the target route corridor and its traffic; a curved ramp or road identifier change is not itself a lane change.'),
    'R-E5': ('unsignalized junction priority', 'junction', 'Respect applicable stop and yield obligations and all crossing traffic before entering.'),
}
CONTEXT_TO_EVENT = dict(zip(('LEAD_BRAKE','STATIC_BLOCKAGE','DYNAMIC_CUTIN','VULNERABLE_CROSSING',
    'ONCOMING_INVASION','JUNCTION_RULE_CONFLICT','SIGNAL_FAILURE','POST_BYPASS_RETURN',
    'RAMP_MERGE_EXIT','UNSIGNALIZED_PRIORITY'), EVENTS))
TEMPLATE_STATES = {
    'yield': ('YIELD','PROCEED','DONE'),
    'junction': ('YIELD','PROCEED','DONE'),
    'bypass': ('WAIT','DEPART','PASS','RETURN','DONE'),
    'lane': ('WAIT','CROSS','DONE'),
    'merge': ('WAIT','CROSS','DONE'),
}
LONGITUDINAL = ('STABLE','APPROACH','FOLLOW','HOLD','RECOVER')
STATE_TEXT = {
    'YIELD':'responding to the conflict and awaiting passage', 'PROCEED':'proceeding beyond the conflict',
    'WAIT':'awaiting entry to the established target corridor', 'DEPART':'entering the bypass corridor',
    'PASS':'travelling along the bypass corridor past the obstruction', 'RETURN':'returning to the established route corridor',
    'CROSS':'entering the established route corridor', 'DONE':'the event stage is complete, with any remaining longitudinal constraint still in effect',
    'STABLE':'stable unrestricted progress', 'APPROACH':'reducing progress to establish clearance',
    'FOLLOW':'stable restricted progress', 'HOLD':'coming to or maintaining a stationary wait',
    'RECOVER':'restoring progress after a restriction',
}


@dataclass(frozen=True)
class Edge:
    key: str
    axis: str
    source: str
    target: str
    kind: str
    criteria: tuple[str, ...]
    question: str


def edge(key, axis, source, target, kind, criteria, question):
    return Edge(key,axis,source,target,kind,tuple(criteria.split()),question)


COMMON = (
    edge('restrict','longitudinal','STABLE','APPROACH','opportunity','restriction_present',
         'Has the available corridor or following gap ceased to support the current unrestricted progress?'),
    edge('settle','longitudinal','APPROACH','FOLLOW','milestone','restricted_progress_established',
         'Has a stable moving gap been established while the restriction remains?'),
    edge('hold','longitudinal','*','HOLD','opportunity','stationary_wait_required',
         'Do the available forward space or priority obligations now require a stationary wait?'),
    edge('release','longitudinal','*','RECOVER','opportunity','release_ready priority_satisfied corridor_clear',
         'Do participant motion, available forward space and priority now support restoring progress?'),
    edge('stable','longitudinal','RECOVER','STABLE','milestone','stable_progress_established',
         'Has stable progress been established after the restriction?'),
    edge('renewed_restriction','longitudinal','*','APPROACH','opportunity','restriction_present',
         'Has a renewed restriction made the currently established progress unsuitable?'),
)


def event_edges(template, return_required=True):
    if template in ('yield','junction'):
        return (
            edge('proceed','event','YIELD','PROCEED','opportunity','release_ready corridor_clear priority_satisfied',
                 'Do the conflict geometry, participant motion and priority now permit proceeding?'),
            edge('complete','event','PROCEED','DONE','milestone','event_resolved stable_progress_established',
                 'Has this conflict been passed or settled into ordinary following, with stable progress established?'),
            edge('re_yield','event','PROCEED','YIELD','opportunity','restriction_present',
                 'Has a renewed conflict in this event made continued passage unsuitable?'),
        )
    if template == 'bypass':
        tail = edge('return','event','PASS','RETURN','opportunity',
                    'obstacle_passed target_corridor_known return_gap_clear priority_satisfied',
                    'Has the entire obstruction been passed, and is the established return corridor available now?') if return_required else edge(
                    'complete','event','PASS','DONE','milestone','obstacle_passed route_corridor_reached stable_progress_established',
                    'Has the obstruction been passed with stable progress in the route corridor that requires no return?')
        return (
            edge('depart','event','WAIT','DEPART','opportunity','target_corridor_known entry_gap_clear priority_satisfied',
                 'Is the established bypass corridor available to enter now, considering approaching traffic and clearance?'),
            edge('pass','event','DEPART','PASS','milestone','lateral_entry_complete',
                 'Has ego fully entered the bypass corridor and settled the lateral movement?'), tail,
            *(() if not return_required else (edge('complete','event','RETURN','DONE','milestone',
                'route_corridor_reached lateral_entry_complete',
                'Has ego fully returned to the established route corridor and settled the lateral movement?'),)),
        )
    if template in ('lane','merge'):
        return (
            edge('enter','event','WAIT','CROSS','opportunity','target_corridor_known entry_gap_clear priority_satisfied',
                 'Is the established target corridor available to enter now, considering relevant traffic and priority?'),
            edge('complete','event','CROSS','DONE','milestone','route_corridor_reached lateral_entry_complete',
                 'Has ego become established in the intended route corridor with the entry maneuver completed?'),
        )
    raise ValueError(f'unknown template: {template}')


def template_for(event, branch='default'):
    if event not in EVENTS:
        raise ValueError(f'unsupported event: {event}')
    if branch == 'cyclist_bypass' and event == 'U-E4':
        return 'bypass'
    if (event,branch) in (('U-E4','cyclist_follow'),('U-E2','in_lane_pass')):
        return 'yield'
    if branch != 'default':
        raise ValueError(f'invalid branch {event}/{branch}')
    return EVENTS[event][1]


def transitions(event, branch='default', return_required=True):
    return (*event_edges(template_for(event,branch),return_required), *COMMON)


def get_edge(event, key, branch='default', return_required=True):
    matches=[e for e in transitions(event,branch,return_required) if e.key==key]
    if len(matches)!=1:
        raise ValueError(f'unknown transition: {event}/{key}')
    return matches[0]


def applicable(e, state, longitudinal):
    current=state if e.axis=='event' else longitudinal
    if state=='DONE' and (e.axis=='event' or longitudinal in ('STABLE','FOLLOW')):
        return False
    # 不能在仍需制动/停车/恢复时丢弃本实例的纵向约束。
    if e.key=='complete' and 'stable_progress_established' in e.criteria and longitudinal not in ('STABLE','FOLLOW'):
        return False
    # 进入许可由事件边统一给出，避免在尚不能绕出时单独恢复纵向。
    if e.key=='release' and state in ('WAIT','YIELD'):
        return False
    if e.source!='*':
        return current==e.source
    return current in {'hold':('STABLE','APPROACH','FOLLOW','RECOVER'),
                       'release':('APPROACH','FOLLOW','HOLD'),
                       'renewed_restriction':('FOLLOW','RECOVER')}[e.key]
