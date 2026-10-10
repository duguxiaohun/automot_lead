"""Active route-aware entry point; non-route questions use the versioned common rubric."""
import math
from . import prompts as frozen
from .prompts import MODEL_ANSWERS, parse_answer
from .route_context import episode_edge, route_bound, BOUNDARY
from .taxonomy import STATE_TEXT, EVENTS, applicable

VISIBLE_EDGES = {'depart', 'enter', 'return'}
PROMPT_VERSION = 'phase4_state_pair_binary_v11'
SYSTEM = (frozen.SYSTEM +
    ' Evaluate every stated requirement together; evidence for one does not establish the others.'
    ' Ego movement alone does not establish permission, and a participant disappearing or becoming occluded does not establish clearance.'
    ' A newly appearing restriction requires a new judgment; it does not by itself prove the earlier judgment was wrong.'
    ' Event names and navigation targets identify the question context; they do not establish that a transition condition holds.')


def cyclist_follow(episode):
    return episode.event=='U-E4' and episode.branch=='cyclist_follow'


def prompt_version(episode, edge_key=None):
    if cyclist_follow(episode):return 'phase4_state_pair_cyclist_follow_v11'
    if edge_key in VISIBLE_EDGES | {'recover_follow'}:
        return 'phase4_state_pair_visible_v11'
    return 'phase4_state_pair_route_v11' if route_bound(episode) else PROMPT_VERSION


def state_pair(episode, edge_key):
    if cyclist_follow(episode):
        edge=episode_edge(episode,edge_key)
        if not applicable(edge,episode.state,episode.longitudinal):raise ValueError('question not applicable to controller state')
        causes=dict(frozen.CAUSES)
        causes.update(release_ready='the same-direction cyclist motion and usable moving following gap support restoring progress',
            corridor_clear='the forward following corridor is usable, with no additional conflicting participant',
            event_resolved='the cyclist interaction has settled into ordinary following or the cyclist has visibly left the conflict',
            restricted_progress_established='a stable moving following gap behind the cyclist has been established')
        source=f'{STATE_TEXT[episode.longitudinal]} while responding to a same-direction cyclist; event stage: {STATE_TEXT[episode.state]}'
        destination='; '.join(causes.get(k,k) for k in edge.criteria)
        destination+='; the cyclist may remain ahead; this branch does not authorize overtaking or entering an opposing lane'
        return source,destination
    if edge_key == 'recover_follow':
        edge = episode_edge(episode, edge_key)
        if not applicable(edge, episode.state, episode.longitudinal):
            raise ValueError('question not applicable to controller state')
        return (f'{STATE_TEXT[episode.longitudinal]} in response to {EVENTS[episode.event][0]}',
                'a stable moving following gap has now been established after recovery, '
                'so ordinary following continues without a renewed conflict')
    if edge_key in VISIBLE_EDGES:
        source, destination = frozen.state_pair(episode, edge_key)
        destination = destination.replace('the entry gap permits entry, including approaching traffic',
            'within the observed camera coverage, the entry gap is clear of visible conflicting traffic')
        destination = destination.replace('the return gap permits re-entry',
            'within the observed camera coverage, the return gap is clear of visible conflicting traffic')
        destination += ('; this is a visible-condition judgment only, with no assertion about unseen rear or side traffic; '
                        'execution requires separate current planner clearance of the full maneuver corridor')
        return source, destination
    if not route_bound(episode):
        return frozen.state_pair(episode, edge_key)
    edge = episode_edge(episode, edge_key)
    if not applicable(edge, episode.state, episode.longitudinal):
        raise ValueError('question not applicable to controller state')
    if episode.route_context and edge_key == 'complete':
        source = f'{STATE_TEXT[episode.state]} in response to {EVENTS[episode.event][0]}'
        destination = ''
    else:
        source, destination = frozen.state_pair(episode, edge_key)
    route = episode.target_corridor or 'the navigation-established corridor'
    if episode.direction:
        route += ' (' + episode.direction.lower() + ')'
    source += '; current navigation target: ' + route
    if episode.route_context:
        source += '; roundabout instance ends at: ' + episode.route_context['completion_boundary']
        if edge.key == 'complete':
            causes = dict(frozen.CAUSES)
            causes[BOUNDARY] = ('the ego has visibly reached the specified exit boundary '
                                'and cleared the roundabout circulation conflict')
            destination = ('; '.join(causes[k] for k in edge.criteria) +
                ', so the specified roundabout exit is reached and this geometric stage is complete; '
                'remaining longitudinal restrictions stay in effect')
        elif edge.key == 'proceed':
            destination += ('; permission covers current progress toward that exit; '
                            'renewed conflicts require yielding again')
    destination += '; target: ' + route
    return source, destination


def prompt(episode, edge_key, observation):
    if not cyclist_follow(episode) and not route_bound(episode) and edge_key not in VISIBLE_EDGES | {'recover_follow'}:
        return frozen.prompt(episode, edge_key, observation)
    if set(observation) - frozen.OBSERVATION_FIELDS:
        raise ValueError('non-observable or unknown prompt fields')
    frames, frame = observation['history_frames'], observation['frame_id']
    if (len(frames) not in (2,4) or any(type(f) is not int for f in frames)
            or frames != sorted(set(frames)) or frames[-1] != frame or frames[0] < 4):
        raise ValueError('history must contain 2/4 distinct causal frames ending at current frame')
    speed = observation['speed_mps']
    if not isinstance(speed,(int,float)) or not math.isfinite(speed) or speed < 0:
        raise ValueError('invalid observed speed')
    source, destination = state_pair(episode, edge_key)
    return f'Current state: {source}.\nCandidate next state: {destination}.\nTransition now?'


def messages(episode, edge_key, observation, images, target=None):
    if not cyclist_follow(episode) and not route_bound(episode) and edge_key not in VISIBLE_EDGES | {'recover_follow'}:
        result = frozen.messages(episode, edge_key, observation, images, target)
        result[0]['content'] = SYSTEM
        return result
    if len(images) != len(observation['history_frames']):
        raise ValueError('image/history count mismatch')
    content = [{'type':'image','image':im} for im in images]
    content.append({'type':'text','text':prompt(episode, edge_key, observation)})
    result = [{'role':'system','content':SYSTEM},{'role':'user','content':content}]
    if target is not None:
        if target not in MODEL_ANSWERS:
            raise ValueError('binary transition target must be YES/NO')
        result.append({'role':'assistant','content':target})
    return result
