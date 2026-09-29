"""逐帧RGB标定：成立区间与旧状态补问分别取证，时间距离不产生标签。"""
from .taxonomy import get_edge
CALIBRATION_VERSION = 'rgb_reviewed_transition_bands_v5'
SOURCES = {'rgb_review', 'causal_geometry_review'}


def label_condition(episode, edge_key, frame_id, facts, *, context_valid=True):
    if context_valid is False:
        return 'INVALID'
    if context_valid is not True:
        return 'UNKNOWN'
    edge = get_edge(episode.event,edge_key,episode.branch,episode.return_required)
    values = []
    for key in edge.criteria:
        fact = facts.get(key)
        if fact is None:
            values.append(None)
            continue
        if (fact.get('source') not in SOURCES or not fact.get('evidence_id')
                or (fact.get('value') is not None and type(fact['value']) is not bool)):
            raise ValueError('unreviewed or invalid condition evidence')
        if fact.get('observed_until',frame_id+1) > frame_id:
            raise ValueError('future evidence cannot establish present permission')
        values.append(fact['value'] if fact['valid_from'] <= frame_id <= fact['valid_to'] else None)
    if False in values:
        return 'NO'
    return 'YES' if values and all(v is True for v in values) else 'UNKNOWN'


def interval_facts(values, start, end, evidence_id):
    """标注者声明区间内每帧已核验；观察截至该帧，不含未来控制/轨迹。"""
    if start > end or not evidence_id:
        raise ValueError('invalid evidence interval')
    return {key:dict(value=value,source='rgb_review',evidence_id=evidence_id,
                    valid_from=start,valid_to=end,observed_until=start) for key,value in values.items()}


def catchup_label(episode, edge_key, frame_id, *, completed_at, valid_until,
                  subsequent_stage_confirmed, evidence_id, context_valid=True):
    """Deprecated interval-only API cannot establish current transition readiness."""
    raise ValueError('legacy catchup requires reviewed_transition_band with per-frame conditions and conflict scope')


def reviewed_band_labels(episode, edge_key, band, *, context_valid=True):
    """每帧独立审核；reference仅解释前后范围，绝不从点距离推导YES。

    stop是旧问题的采样截止，不表示之后条件变成NO。review_end表示右删失，
    尚未观察到适宜补问的终点，不能把窗末当成该事件的最佳固定宽度。
    """
    edge = get_edge(episode.event, edge_key, episode.branch, episode.return_required)
    start, end = band['review_start'], band['review_end']
    reference, stop = band.get('reference_frame'), band['stop']
    if (any(type(v) is not int or v < 0 for v in (start,end,stop['frame']))
            or start > end or not start < stop['frame'] <= end+1
            or not stop.get('reason') or stop['kind'] not in
            ('next_stage','stable_successor','new_conflict','instance_boundary',
             'calibration_anomaly','review_end','unresolved')):
        raise ValueError('invalid reviewed band boundary')
    if reference is not None and (type(reference) is not int or not start <= reference < stop['frame']):
        raise ValueError('reference outside reviewed question interval')
    reference_kind = band['reference_kind']
    if ((reference is None and reference_kind!='not_observed') or
            (reference is not None and reference_kind not in
             ('successor_observed','condition_onset','milestone_observed')) or
            (reference is not None and edge.kind=='milestone' and reference_kind!='milestone_observed')):
        raise ValueError('reference must distinguish visible execution, readiness and completion')
    if not band.get('reference_observation') or not band.get('start_reason'):
        raise ValueError('missing visual boundary justification')
    if stop['kind']=='review_end' and stop['frame']!=end+1:
        raise ValueError('censored boundary must be the reviewed end')
    if set(band['frames']) != {str(f) for f in range(start,end+1)}:
        raise ValueError('review every frame explicitly, including uncertain/excluded frames')
    labels = {}
    for frame in range(start,end+1):
        item = band['frames'][str(frame)]
        phase = item['phase']
        if (phase not in ('readiness','catchup','uncertain','excluded') or
                not item.get('observation') or type(item.get('observed_until')) is not int
                or item['observed_until'] != frame):
            raise ValueError('missing current causal RGB evidence')
        if (frame >= stop['frame']) != (phase=='excluded'):
            raise ValueError('old question crosses its reviewed stop boundary')
        if phase=='excluded':
            continue
        if phase=='uncertain':
            labels[frame] = ('UNKNOWN','uncertain')
            continue
        if set(item.get('facts',{})) - set(edge.criteria):
            raise ValueError('condition does not belong to this transition')
        facts = interval_facts(item.get('facts',{}),frame,frame,item['observation'])
        condition = label_condition(episode,edge_key,frame,facts,context_valid=context_valid)
        if phase=='catchup':
            if (reference is None or frame < reference
                    or type(item.get('successor_confirmed')) is not bool
                    or type(item.get('current_conflict')) is not bool):
                raise ValueError('catchup requires observed successor and current conflict review')
            # Legacy absence of conflict is unambiguous. A reported conflict needs
            # an explicit scope review; do not guess whether it cancels this edge.
            blocker = item.get('transition_blocking_conflict')
            if blocker is None and item['current_conflict'] is False:
                blocker = False
            if type(blocker) is not bool or (blocker and not item['current_conflict']):
                raise ValueError('catchup requires transition-specific conflict review')
            # Independent longitudinal restrictions do not negate completed geometry.
            if condition in ('NO','INVALID') or blocker:
                target = 'NO' if context_valid is True else condition
            elif context_valid is not True:
                target = condition
            else:
                target = 'YES' if item['successor_confirmed'] else 'UNKNOWN'
        else:
            target = condition
        if target=='YES' and edge.kind=='milestone' and (reference is None or frame < reference):
            raise ValueError('a completion milestone cannot be made true before it is observed')
        labels[frame] = (target,phase)
    return labels


def band_summary(episode, edge_key, band):
    labels = reviewed_band_labels(episode,edge_key,band)
    reference = band.get('reference_frame')
    yes = [f for f,(target,_) in labels.items() if target=='YES']
    return dict(reference_frame=reference,reference_kind=band['reference_kind'],first_yes=min(yes) if yes else None,
                last_yes=max(yes) if yes else None,
                before_frames=reference-min(yes) if reference is not None and yes else None,
                after_frames=max(yes)-reference if reference is not None and yes else None,
                right_censored=band['stop']['kind']=='review_end',
                stop=band['stop'],counts={phase:sum(p==phase and t=='YES' for t,p in labels.values())
                    for phase in ('readiness','catchup')})
