"""只展示当前状态和候选下一状态；因果条件、动作含义融入状态描述。"""
import math
from .taxonomy import EVENTS, STATE_TEXT, get_edge, applicable, UE3_RELEASE_CONDITION

PROMPT_VERSION = 'phase4_state_pair_binary_v6'
MODEL_ANSWERS = ('YES','NO')
OBSERVATION_FIELDS = {'frame_id','speed_mps','history_frames'}
SYSTEM = ('Use the chronological RGB observations to judge the proposed state transition. '
          'The next state is a candidate, not a known future observation. '
          'Answer only YES or NO. YES means the stated conditions hold now, or the successor has just been achieved while the supplied current state is stale. '
          'Being close to a transition is insufficient. Completed entry, return or passage must already be visible. '
          'Continued or renewed conflict blocks permission to begin entry or proceed. '
          'A completed geometric maneuver does not remove longitudinal restrictions. Otherwise NO.')

CAUSES = {
    'restriction_present':'the current corridor or following gap now requires yielding',
    'restricted_progress_established':'a stable moving gap has been established while the restriction remains',
    'stationary_wait_required':'available space or priority now requires a stationary wait',
    'release_ready':'the participant motion and gap support restoring progress',
    'corridor_clear':'the ego travel corridor is no longer blocked by the conflict',
    'priority_satisfied':'the applicable priority obligations are satisfied',
    'stable_progress_established':'stable progress has been established',
    'event_resolved':'the conflict has been passed or settled into ordinary following',
    'target_corridor_known':'the route target is established',
    'entry_gap_clear':'the entry gap permits entry, including approaching traffic',
    'lateral_entry_complete':'entry into the target corridor has been completed',
    'obstacle_passed':'the whole obstruction, including the last relevant participant, has been passed',
    'return_gap_clear':'the return gap permits re-entry',
    'route_corridor_reached':'the intended route corridor has been reached',
}
RELEASE_CAUSES = {
    'U-E1':'the lead vehicle motion and following gap permit restoring progress',
    'U-E3':UE3_RELEASE_CONDITION,
    'U-E4':'pedestrian or cyclist motion no longer obstructs the ego travel corridor; full street crossing is unnecessary',
    'U-E5':'the relevant oncoming traffic no longer conflicts with the required ego corridor',
}


def state_pair(episode, edge_key):
    e = get_edge(episode.event,edge_key,episode.branch,episode.return_required)
    if not applicable(e,episode.state,episode.longitudinal):
        raise ValueError('question not applicable to controller state')
    current = episode.state if e.axis == 'event' else episode.longitudinal
    source = f'{STATE_TEXT[current]} in response to {EVENTS[episode.event][0]}'
    causes = [RELEASE_CAUSES.get(episode.event,CAUSES[k]) if k=='release_ready' else CAUSES[k] for k in e.criteria]
    destination = STATE_TEXT[e.target]
    if e.key=='complete' and 'stable_progress_established' not in e.criteria:
        maneuver = 'return to' if episode.state=='RETURN' else 'entry into'
        destination = f'{maneuver} the established route corridor is complete; remaining longitudinal restrictions stay in effect'
    if e.key in ('depart','enter','pass','return') or (e.key=='complete' and episode.state in ('CROSS','RETURN','PASS')):
        returning = e.key=='return' or episode.state=='RETURN'
        direction = episode.return_direction if returning else episode.direction
        corridor = episode.return_corridor if returning else episode.target_corridor
        if direction:
            destination += f' ({direction.lower()}: {corridor or "established route corridor"})'
    if episode.route_segments and e.axis == 'event':
        source += '; only the current adjacent route segment is under consideration'
        destination += '; this entry or completion applies only to that segment, without permission for further lateral movement'
    return source, '; '.join(causes) + ', so ' + destination


def prompt(episode, edge_key, observation):
    # Observation metadata is validated for causal image assembly, not rendered as extra text.
    if set(observation) - OBSERVATION_FIELDS:
        raise ValueError('non-observable or unknown prompt fields')
    frames, frame = observation['history_frames'], observation['frame_id']
    if (len(frames) not in (2,4) or any(type(f) is not int for f in frames)
            or frames != sorted(set(frames)) or frames[-1] != frame or frames[0] < 4):
        raise ValueError('history must contain 2/4 distinct causal frames ending at current frame')
    speed = observation['speed_mps']
    if not isinstance(speed,(int,float)) or not math.isfinite(speed) or speed < 0:
        raise ValueError('invalid observed speed')
    source, destination = state_pair(episode,edge_key)
    return f'Current state: {source}.\nCandidate next state: {destination}.\nTransition now?'


def messages(episode, edge_key, observation, images, target=None):
    if len(images) != len(observation['history_frames']):
        raise ValueError('image/history count mismatch')
    content = [{'type':'image','image':im} for im in images]
    content.append({'type':'text','text':prompt(episode,edge_key,observation)})
    result = [{'role':'system','content':SYSTEM},{'role':'user','content':content}]
    if target is not None:
        if target not in MODEL_ANSWERS:
            raise ValueError('binary transition target must be YES/NO')
        result.append({'role':'assistant','content':target})
    return result


def parse_answer(text):
    value = text.strip()
    if value not in MODEL_ANSWERS:
        raise ValueError('malformed binary transition answer')
    return value
