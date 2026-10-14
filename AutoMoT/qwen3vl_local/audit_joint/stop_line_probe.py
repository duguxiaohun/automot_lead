"""Read-only OpenDRIVE STOP-position diagnostics, never label authorization.

Maps are explicit, SHA-bound inputs. Lane direction determines the approach
endpoint; proximity alone cannot establish control ownership or map identity.
Only line/arc plan geometry and polynomial lane widths are supported. This is
a 2-D diagnostic, not a certified painted-line or simulator-trigger reader.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .io import file_sha, write_json
from .stop_proxy import measure


def number(element, key):
    value = float(element.attrib[key])
    if not math.isfinite(value):
        raise ValueError('nonfinite OpenDRIVE '+key)
    return value


def at(elements, s, key='s'):
    elements = list(elements)
    starts = [number(e, key) for e in elements]
    if starts != sorted(set(starts)):
        raise ValueError('duplicate or unordered OpenDRIVE intervals')
    eligible = [e for e in elements if number(e, key) <= s]
    if not eligible:
        raise ValueError('missing OpenDRIVE interval')
    return eligible[-1]


def polynomial(element, ds):
    return sum(number(element, k)*ds**i for i, k in enumerate('abcd'))


def coordinate_budget(xy):
    """Numerical matching budget, not a parking-distance allowance.

    Exported CARLA transforms use float32 world coordinates. Allow eight ULPs
    per planar coordinate for serialization/rigid-transform round trips, with
    a 1cm hard ceiling; unsupported scales fail closed instead of widening.
    """
    if len(xy)!=2 or not all(math.isfinite(v) for v in xy):
        raise ValueError('invalid map coordinates')
    ulps=[2.**(math.floor(math.log2(abs(v)))-23) if v else 2.**-149 for v in xy]
    budget=8*math.hypot(*ulps)+1e-6
    if budget>.01:raise ValueError('coordinate scale exceeds binding precision budget')
    return budget


def road_point(road, s, t=0.):
    """OpenDRIVE reference point plus lateral offset, reflected into CARLA xy."""
    if not all(math.isfinite(v) for v in (s, t)) or not 0 <= s <= number(road, 'length'):
        raise ValueError('road coordinate outside declared length')
    geo = at(road.findall('planView/geometry'), s)
    ds = s-number(geo, 's')
    if ds > number(geo, 'length')+1e-8:
        raise ValueError('gap in road plan geometry')
    h = number(geo, 'hdg'); x = number(geo, 'x'); y = number(geo, 'y')
    if geo.find('line') is not None:
        x += ds*math.cos(h); y += ds*math.sin(h)
    elif geo.find('arc') is not None:
        k = number(geo.find('arc'), 'curvature')
        if k == 0:
            x += ds*math.cos(h); y += ds*math.sin(h)
        else:
            x += (math.sin(h+k*ds)-math.sin(h))/k
            y += (math.cos(h)-math.cos(h+k*ds))/k
            h += k*ds
    else:
        raise ValueError('unsupported plan geometry; no straight-line fallback')
    return [x-t*math.sin(h), -(y+t*math.cos(h))], -h


def lane_center(road, lane_id, s):
    section = at(road.findall('lanes/laneSection'), s)
    side = 'right' if lane_id < 0 else 'left'
    center = 0.
    offsets = road.findall('lanes/laneOffset')
    if offsets:
        offset = at(offsets, s); center = polynomial(offset, s-number(offset, 's'))
    for i in range(1, abs(lane_id)+1):
        lane = section.find(side+'/lane[@id="'+str(-i if lane_id < 0 else i)+'"]')
        if lane is None or (i == abs(lane_id) and lane.get('type') != 'driving'):
            raise ValueError('missing or non-driving source lane')
        width = at(lane.findall('width'), s-number(section, 's'), 'sOffset')
        w = polynomial(width, s-number(section, 's')-number(width, 'sOffset'))
        if not math.isfinite(w) or w <= 0:
            raise ValueError('invalid lane width')
        center += (-1 if lane_id < 0 else 1)*w*(.5 if i == abs(lane_id) else 1.)
    return center


class MapProbe:
    def __init__(self, path):
        self.path = Path(path).absolute(); self.resolved = self.path.resolve()
        payload = self.path.read_bytes(); self.sha = hashlib.sha256(payload).hexdigest()
        root = ET.fromstring(payload)
        if root.tag != 'OpenDRIVE':
            raise ValueError('not an OpenDRIVE map')
        roads = root.findall('road'); junctions = root.findall('junction')
        self.roads = {r.attrib['id']: r for r in roads}
        self.junctions = {j.attrib['id']: j for j in junctions}
        if len(self.roads) != len(roads) or len(self.junctions) != len(junctions):
            raise ValueError('duplicate map identity')

    def verify_unchanged(self):
        if self.path.resolve() != self.resolved or file_sha(self.path) != self.sha:
            raise ValueError('map changed during diagnostic')

    def probe(self, frame, stop_id):
        from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as g
        from qwen3vl_local.sft_new_loop_phase4.teacher_controls import owned_lane
        result = dict(status='diagnostic_only', label_authorization=False,
                      capture_map_identity_verified=False, control_line_binding_verified=False,
                      map_path=str(self.path), map_resolved_path=str(self.resolved), map_sha256=self.sha,
                      proxy=measure(frame, stop_id), candidates=[], errors=[],
                      scope='2-D lane endpoint and same-junction StopLine candidates; '
                            'not painted-line extent, map provenance or control ownership certification')
        try:
            lane = owned_lane(frame)
            if lane is None or lane[1] == 0:
                raise ValueError('missing or contradictory source road/lane')
            road = self.roads[str(lane[0])]
            if road.get('rule', 'RHT') != 'RHT':
                raise ValueError('left-hand traffic direction not implemented')
            # For RHT, negative lanes travel with s, positive lanes against s.
            s = number(road, 'length') if lane[1] < 0 else 0.
            link = road.find('link/'+('successor' if lane[1] < 0 else 'predecessor'))
            if link is None or link.get('elementType') != 'junction':
                raise ValueError('approach endpoint does not link to a junction')
            junction = self.junctions[link.attrib['elementId']]
            connections = []
            for c in junction.findall('connection'):
                if c.get('incomingRoad') == str(lane[0]):
                    for ll in c.findall('laneLink'):
                        if int(ll.get('from')) == lane[1]:
                            connections.append(dict(c.attrib, lane_link=dict(ll.attrib)))
            if not connections:
                raise ValueError('no incoming lane connection at approach junction')
            point, heading = road_point(road, s, lane_center(road, lane[1], s))
            if lane[1] > 0:
                heading += math.pi
            matrix = frame['meta']['ego_matrix']
            if (len(matrix) != 4 or any(len(row) != 4 for row in matrix)
                    or not all(g.finite(v) for row in matrix for v in row)):
                raise ValueError('invalid ego world pose')
            forward = [matrix[0][0], matrix[1][0]]; norm = math.hypot(*forward)
            if norm < .5:
                raise ValueError('degenerate planar ego heading')
            forward = [v/norm for v in forward]; left = [-forward[1], forward[0]]
            ego = next(a for a in frame['actors'] if a['class'] == 'ego_car')
            stop = next(a for a in frame['actors'] if a['class'] == 'stop_sign' and a['id'] == stop_id)
            world_stop = g.world_point(frame, stop)
            if world_stop is None:
                raise ValueError('stop lacks world pose')
            def distances(xy):
                d = [xy[i]-matrix[i][3] for i in range(2)]
                longitudinal = sum(d[i]*forward[i] for i in range(2))
                return dict(ego_forward_m=longitudinal,
                            front_gap_m=longitudinal-g.aabb(ego)[1],
                            lateral_m=sum(d[i]*left[i] for i in range(2)),
                            distance_to_stop_center_m=math.hypot(xy[0]-world_stop[0],xy[1]-world_stop[1]))
            result.update(source_road=lane[0],source_lane=lane[1],junction_id=junction.attrib['id'],
                          incoming_connections=connections,
                          approach_endpoint=dict(s=s,xy=point,heading_rad=heading,**distances(point)))
            failures = []
            connected = {c['connectingRoad'] for c in connections}
            for rd in self.roads.values():
                if rd.get('id') != str(lane[0]) and rd.get('junction') != junction.attrib['id']:
                    continue
                for signal in rd.findall('signals/signal'):
                    if 'stopline' not in signal.get('name','').lower():
                        continue
                    try:
                        xy, h = road_point(rd, number(signal,'s'), number(signal,'t'))
                        h -= number(signal,'hOffset')
                        candidates = dict(road_id=rd.attrib['id'],signal=dict(signal.attrib),xy=xy,
                                          distance_to_approach_endpoint_m=math.dist(xy,point),
                                          line_perpendicular_error_rad=math.asin(min(1.,abs(math.cos(h-heading)))),
                                          directly_connected_road=rd.attrib['id'] in connected,
                                          **distances(xy))
                        result['candidates'].append(candidates)
                    except (ValueError,KeyError,TypeError) as exc:
                        failures.append(dict(road_id=rd.attrib['id'],signal_id=signal.get('id'),error=str(exc)))
            result['candidates'].sort(key=lambda c:(c['distance_to_approach_endpoint_m'],c['road_id'],c['signal']['id']))
            result['unsupported_candidates'] = failures
            result['control_reference'] = self.control_reference(road, lane[1], connections, world_stop)
        except (ValueError,KeyError,TypeError,StopIteration) as exc:
            result['errors'].append(str(exc))
        return result

    def control_reference(self, road, lane_id, connections, world_stop):
        """Trace lane-valid STOP references, not nearest painted-line signals.

        The 3m backshift is a *named upstream implementation model*, not a
        fitted correction or a label threshold. Residuals remain diagnostics;
        capture-time simulator/map provenance is not established by a match.
        """
        refs = []; errors = []
        for connection in connections:
            try:
                rd = self.roads[connection['connectingRoad']]
                approach_link=road.find('link/'+('successor' if lane_id<0 else 'predecessor'))
                if (approach_link is None or approach_link.get('elementType')!='junction'
                        or rd.get('junction')!=approach_link.get('elementId')
                        or rd.get('junction') in (None,'-1')):
                    raise ValueError('connecting road is not in the approach junction')
                target = int(connection['lane_link']['to'])
                entry = connection['contactPoint']
                if (target < 0 and entry != 'start') or (target > 0 and entry != 'end') or target == 0:
                    raise ValueError('connection disagrees with lane travel direction')
                predecessor = rd.find('link/'+('predecessor' if target < 0 else 'successor'))
                expected_contact = 'end' if lane_id < 0 else 'start'
                if (predecessor is None or predecessor.get('elementType') != 'road'
                        or predecessor.get('elementId') != road.attrib['id']
                        or predecessor.get('contactPoint') != expected_contact):
                    raise ValueError('connecting road does not lead back to this approach')
                section = at(rd.findall('lanes/laneSection'),0. if target < 0 else number(rd,'length'))
                target_lane = section.find(('right' if target < 0 else 'left')+'/lane[@id="'+str(target)+'"]')
                if target_lane is None or target_lane.get('type') != 'driving':
                    raise ValueError('signal reference target is not a driving lane')
                upstream=target_lane.find('link/'+('predecessor' if target<0 else 'successor'))
                if upstream is None or upstream.get('id')!=str(lane_id):
                    raise ValueError('target lane predecessor does not match approach lane')
                for ref in rd.findall('signals/signalReference'):
                    validities = [v.attrib for v in ref.findall('validity') if
                                  min(int(v.get('fromLane')),int(v.get('toLane'))) <= target <=
                                  max(int(v.get('fromLane')),int(v.get('toLane')))]
                    if not validities:
                        continue
                    definitions = [(r,sig) for r in self.roads.values() for sig in r.findall('signals/signal')
                                   if sig.get('id') == ref.get('id')]
                    if len(definitions) != 1:
                        raise ValueError('signal definition missing or ambiguous')
                    defining_road, signal = definitions[0]
                    if signal.get('type') != '206':
                        continue
                    if not 0 <= number(ref,'s') <= number(rd,'length'):
                        raise ValueError('signal reference outside road')
                    expected_orientation = '-' if target < 0 else '+'
                    if ref.get('orientation') not in (expected_orientation,'none'):
                        raise ValueError('signal reference orientation disagrees with lane')
                    refs.append(dict(signal_id=ref.get('id'),defining_road=defining_road.attrib['id'],
                                     signal=dict(signal.attrib),connecting_road=rd.attrib['id'],
                                     target_lane=target,reference=dict(ref.attrib),validities=validities))
            except (KeyError,TypeError,ValueError) as exc:
                errors.append(dict(connecting_road=connection['connectingRoad'],error=str(exc)))
        unique = sorted({r['signal_id'] for r in refs})
        result = dict(status='unresolved',references=refs,unique_signal_ids=unique,errors=errors,
                      label_authorization=False,capture_implementation_verified=False,
                      upstream_model='CARLA 0.9.15 StopSignComponent InitializeSign',
                      upstream_source='https://github.com/carla-simulator/carla/blob/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Traffic/StopSignComponent.cpp')
        if errors or len(unique) != 1:
            return result
        # The upstream model relocates an intersection reference to its sole
        # non-junction predecessor, then shifts 1.5+1.5m within that lane section.
        endpoint = number(road,'length') if lane_id < 0 else 0.
        section = at(road.findall('lanes/laneSection'),endpoint)
        section_start = number(section,'s')
        later = [number(s,'s') for s in road.findall('lanes/laneSection') if number(s,'s') > section_start]
        section_end = min(later) if later else number(road,'length')
        if section_end-section_start <= 2e-5 or road.get('junction','-1') != '-1':
            result['errors'].append(dict(error='upstream predecessor assumption not met'))
            return result
        box_s = min(section_end-1e-5,max(section_start+1e-5,endpoint+(-3. if lane_id < 0 else 3.)))
        xy, heading = road_point(road,box_s,lane_center(road,lane_id,box_s))
        if lane_id > 0:
            heading += math.pi
        delta = [world_stop[i]-xy[i] for i in range(2)]
        result.update(status='unique_lane_stop_reference',signal_id=unique[0],
                      upstream_predicted_trigger=dict(s=box_s,xy=xy,heading_rad=heading,
                          observed_xy=world_stop[:2],residual_m=math.hypot(*delta),
                          longitudinal_residual_m=delta[0]*math.cos(heading)+delta[1]*math.sin(heading),
                          lateral_residual_m=-delta[0]*math.sin(heading)+delta[1]*math.cos(heading)))
        return result


class FrozenApproachProbe:
    """Diagnostic anchor selected once; lane changes cannot choose a new STOP.

    This records an unapproved candidate location, not a satisfied obligation.
    Source gaps, missing actors and changed identity remain explicit failures.
    """
    def __init__(self, probe, frame, stop_id):
        from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as g
        probe.verify_unchanged()
        report = probe.probe(frame,stop_id)
        reference = report.get('control_reference',{})
        if report['errors'] or reference.get('status') != 'unique_lane_stop_reference':
            raise ValueError('cannot freeze unresolved approach')
        stop = next(a for a in frame['actors'] if a['class']=='stop_sign' and a['id']==stop_id)
        if stop.get('affects_ego') is not True or frame.get('source_geometry_issues'):
            raise ValueError('cannot establish inactive or invalid source control')
        predicted=reference['upstream_predicted_trigger']
        budget=coordinate_budget(predicted['xy'])
        if not math.isfinite(predicted['residual_m']) or predicted['residual_m']>budget:
            raise ValueError('observed control does not match map reconstruction')
        if list(stop['extent'])!=[1.5,1.5,.5]:
            raise ValueError('unsupported exported STOP shape')
        self.map = probe; self.last = frame['frame_id']; self.stop_id = stop_id
        self.route = (frame.get('scenario'),frame.get('route_id'))
        self.world = g.world_point(frame,stop)
        self.anchor = dict(established_frame=self.last,source_road=report['source_road'],
                           source_lane=report['source_lane'],signal_id=reference['signal_id'],
                           map_sha256=probe.sha,endpoint=deepcopy(report['approach_endpoint']),
                           reference=deepcopy(reference),coordinate_budget_m=budget,
                           sources=deepcopy(frame.get('sources',[])))

    def observe(self, frame):
        from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as g
        from qwen3vl_local.sft_new_loop_phase4.teacher_controls import same_stop
        self.map.verify_unchanged()
        result = dict(label_authorization=False,anchor=deepcopy(self.anchor),errors=[],
                      frame_id=frame['frame_id'],sources=deepcopy(frame.get('sources',[])))
        n=frame['frame_id']; previous=self.last; self.last=n
        if (frame.get('scenario'),frame.get('route_id')) != self.route:
            result['errors'].append('foreign route')
        if n != previous+1:
            result['errors'].append('source discontinuity')
        if frame.get('source_geometry_issues'):
            result['errors'].append('invalid source geometry')
        actor=next((a for a in frame['actors'] if a['class']=='stop_sign' and a['id']==self.stop_id),None)
        if actor is None or not same_stop(frame,actor,self.world):
            result['errors'].append('same physical stop unverified')
        elif (list(actor['extent'])!=[1.5,1.5,.5] or
              math.dist(g.world_point(frame,actor)[:2],self.anchor['reference']['upstream_predicted_trigger']['xy'])>
              self.anchor['coordinate_budget_m']):
            result['errors'].append('control no longer matches frozen map reconstruction')
        if result['errors']:
            return result
        # Fixed world endpoint projected onto fixed approach direction, not the
        # new lane's endpoint. Account for current ego footprint orientation.
        matrix=frame['meta']['ego_matrix']; xy=self.anchor['endpoint']['xy']
        h=self.anchor['endpoint']['heading_rad']; normal=[math.cos(h),math.sin(h)]
        ego=next(a for a in frame['actors'] if a['class']=='ego_car')
        center=g.world_point(frame,ego)
        axes=[[matrix[0][j],matrix[1][j]] for j in range(3)]
        half=sum(ego['extent'][j]*abs(sum(normal[i]*axes[j][i] for i in range(2))) for j in range(3))
        result.update(fixed_approach_front_gap_m=sum((xy[i]-center[i])*normal[i] for i in range(2))-half,
                      same_control_verified=True,current_road=frame['meta'].get('road_id'),
                      current_lane=frame['meta'].get('lane_id'),
                      upstream_trigger_residual_m=self.anchor['reference']['upstream_predicted_trigger']['residual_m'])
        return result


def main():
    from qwen3vl_local.sft_new_loop_phase4.privileged_geometry import load_frame
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('lead_data'))
    p.add_argument('--scenario', required=True); p.add_argument('--route-id', required=True)
    p.add_argument('--frame', type=int, required=True); p.add_argument('--stop-id', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('preserve prior diagnostic')
    probe = MapProbe(a.map); frame = load_frame(a.data_root,a.scenario,a.route_id,a.frame)
    result = probe.probe(frame,a.stop_id); probe.verify_unchanged()
    for source in frame['sources']:
        if file_sha(a.data_root/source['path']) != source['sha256']:
            raise ValueError('source changed during diagnostic')
    write_json(a.output,result)
    print(json.dumps(dict(status=result['status'],errors=result['errors'],label_authorization=False)))
    return 2 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
