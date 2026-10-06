"""Auditable RGB evidence requirements for scheme B, not an image classifier.

Numeric thresholds are conservative development annotation conventions. They
are not driving limits, and neither metadata nor a checked box proves vision.
Unsupported evidence abstains; it does not make a negative training example.
"""
import math
from .observation import observation_contract

POLICY = 'phase4_mode_specific_visual_review_v1'
VRU_MARGIN_M = 1.0
LEAD_GAP_GROWTH_M = 1.0
MIN_IMAGE_SHIFT_PX = 6.0


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def template(images, frame):
    hashes = {i['frame_id']: i['sha256'] for i in images}
    return dict(policy=POLICY, modes={str(mode): dict(
        frames=[frame+o for o in observation_contract(mode)['frame_offsets']],
        sha256=[hashes[frame+o] for o in observation_contract(mode)['frame_offsets']],
        disposition='uncertain', reason='', current_cues=[], successor=None,
        lead_release=None, vulnerable_clearance=None, signal_identity=None)
        for mode in (2, 4)})


def cue_valid(cue, frames):
    box = cue.get('box', [])
    return (type(cue.get('frame')) is int and cue['frame'] in frames
            and len(box)==4 and all(finite(v) for v in box)
            and 0<=box[0]<box[2]<=1152 and 0<=box[1]<box[3]<=384
            and bool(str(cue.get('observation', '')).strip()))


def motion_valid(motion, frame, kind):
    if not isinstance(motion, dict) or motion.get('kind') != kind:
        return False
    points = motion.get('points', [])
    if len(points)!=2 or [p.get('frame') for p in points]!=[frame-4, frame]:
        return False
    for p in points:
        xy=p.get('xy', [])
        if (len(xy)!=2 or not all(finite(x) for x in xy)
                or not 0<=xy[0]<1152 or not 0<=xy[1]<384):return False
    # Same camera and identified feature; seam crossings are not motion.
    return (int(points[0]['xy'][0]//384)==int(points[1]['xy'][0]//384)
            and math.dist(points[0]['xy'], points[1]['xy'])>=MIN_IMAGE_SHIFT_PX
            and motion.get('same_feature_reviewed') is True
            and motion.get('camera_motion_accounted') is True
            and bool(str(motion.get('description', '')).strip()))


def validate(evidence, *, frame, frames, hashes, event, edge, phase, target):
    """Validate only the actual model input; no intervening/future RGB allowed."""
    if (not isinstance(evidence, dict) or evidence.get('frames')!=frames
            or evidence.get('sha256')!=hashes):
        raise ValueError('visual review must bind exact model input frames and SHA')
    if evidence.get('disposition') not in ('supported','uncertain') or not str(evidence.get('reason','')).strip():
        raise ValueError('explicit mode-specific visual disposition required')
    if evidence['disposition']=='uncertain':return False
    cues=evidence.get('current_cues', [])
    if (not cues or not all(cue_valid(c, frames) for c in cues)
            or not any(c['frame']==frame for c in cues)):
        raise ValueError('localize visible current evidence within the actual RGB input')
    if target=='YES' and phase=='catchup':
        successor=evidence.get('successor')
        if not isinstance(successor,dict):raise ValueError('successor needs actual-input visual evidence')
        kind=successor.get('kind')
        if kind=='ego_progress':
            if not motion_valid(successor,frame,'ego_progress'):
                raise ValueError('ego progress needs identified static feature motion across the shared four-frame baseline')
        elif kind in ('lane_membership','phase_completed'):
            if (successor.get('current_frame')!=frame or successor.get('geometry_visible') is not True
                    or not cue_valid(successor.get('cue',{}),[frame])):
                raise ValueError('successor geometry must be localized in current RGB')
        else:raise ValueError('unsupported successor visual evidence')
    if target=='YES' and phase=='readiness' and event=='U-E1' and edge=='proceed':
        release=evidence.get('lead_release')
        if (not motion_valid(release,frame,'lead_receding') or not finite(release.get('gap_growth_m'))
                or release['gap_growth_m']<LEAD_GAP_GROWTH_M
                or not isinstance(release.get('actor_id'),(int,str))
                or len(release.get('source_sha256',[]))!=2
                or any(not isinstance(s,str) or len(s)!=64 for s in release['source_sha256'])
                or release.get('current_path_clear') is not True):
            raise ValueError('lead release requires visible displacement plus corroborated four-frame gap growth')
    if target=='YES' and event=='U-E4' and edge=='proceed':
        clearance=evidence.get('vulnerable_clearance')
        if not isinstance(clearance,dict):raise ValueError('explicit vulnerable-participant clearance required')
        margin=clearance.get('lower_bound_m')
        separated=clearance.get('fully_on_separated_sidewalk') is True
        if (not (separated or finite(margin) and margin>=VRU_MARGIN_M)
                or clearance.get('moving_away_or_separated') is not True
                or clearance.get('whole_footprint_reviewed') is not True
                or not cue_valid(clearance.get('boundary_cue',{}),[frame])):
            raise ValueError('VRU clearance needs a visible boundary and full-footprint margin, or separated sidewalk')
    if event=='U-E7':
        signal=evidence.get('signal_identity')
        if (not isinstance(signal,dict) or signal.get('applicable_head_matched') is not True
                or signal.get('failure_observable') is not True
                or signal.get('ordinary_signal_wait_excluded') is not True
                or not cue_valid(signal.get('cue',{}),frames)):
            raise ValueError('UE7 requires observable applicable signal failure; source name or red light is insufficient')
    return True


def check_metric_sources(card, proof, root, phase, target):
    """Check the auxiliary metre measurement, never use it as a visual label."""
    if target!='YES' or phase!='readiness' or card['episode']['event']!='U-E1' or card['edge']!='proceed':return
    from .privileged_geometry import load_frame,aabb
    for evidence in proof['modes'].values():
        if evidence['disposition']!='supported':continue
        metric=evidence['lead_release'];snapshots=[load_frame(root,card['scenario'],card['route_id'],f)
                                                  for f in (card['frame_id']-4,card['frame_id'])]
        gaps=[]
        for snapshot in snapshots:
            actor=next((a for a in snapshot['actors'] if a['id']==metric['actor_id']),None)
            ego=next(a for a in snapshot['actors'] if a['class']=='ego_car')
            if actor is None:raise ValueError('metric corroboration actor missing')
            gaps.append(aabb(actor)[0]-ego['extent'][0])
        if (metric['source_sha256']!=[f['causal_sha256'] for f in snapshots]
                or abs(metric['gap_growth_m']-(gaps[1]-gaps[0]))>.01):
            raise ValueError('lead gap measurement does not match causal geometry source')


def validate_all(proof, images, frame, event, edge, phase, target):
    if not isinstance(proof,dict) or proof.get('policy')!=POLICY or set(proof.get('modes',{}))!={'2','4'}:
        raise ValueError('explicit 2/4 RGB visual review required')
    expected=template(images,frame)
    return {mode:validate(proof['modes'][mode],frame=frame,
            frames=expected['modes'][mode]['frames'],hashes=expected['modes'][mode]['sha256'],
            event=event,edge=edge,phase=phase,target=target) for mode in ('2','4')}


def row_supported(row):
    if not str(row.get('evidence_id','')).startswith('semi_review/') and 'visual_review' not in row:return True
    proof=row.get('visual_review')
    if not isinstance(proof,dict) or proof.get('policy')!=POLICY:
        raise ValueError('semi-automatic annotation lacks mode-specific visual review')
    if row['target'] in ('YES','NO') and (row['target']!=proof.get('reviewed_target')
            or row['slice']!=proof.get('reviewed_phase')):
        raise ValueError('visual approval is for a different target or phase')
    return validate(proof['evidence'],frame=row['observation']['frame_id'],
        frames=row['observation']['history_frames'],hashes=row['image_sha256'],
        event=row['episode']['event'],edge=row['edge'],phase=proof['reviewed_phase'],target=proof['reviewed_target'])
