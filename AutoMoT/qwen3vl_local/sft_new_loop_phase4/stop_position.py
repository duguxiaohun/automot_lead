"""Matched, fixed incoming-road STOP boundary; never a nearest painted line.

The compact map geometry is a frozen teacher asset. Only a unique source lane
and an observed exported control matching its reconstructed trigger can bind.
Failure revokes position evidence; it never forgives an outstanding duty.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from . import privileged_geometry as g

CATALOG=Path(__file__).with_name('stop_boundary_catalog_v1.json')
BASIS='incoming_lane_end_plane_not_painted_stop_line'


def stamp(path):
    s=path.stat();return str(path.resolve(strict=True)),s.st_size,s.st_mtime_ns,s.st_ctime_ns


class Catalog:
    def __init__(self,path=CATALOG):
        self.path=Path(path);self.stamp=stamp(self.path);payload=self.path.read_bytes()
        self.sha=hashlib.sha256(payload).hexdigest();self.data=json.loads(payload)
        if (self.data.get('policy')!='stop_incoming_boundary_catalog_v1'
                or self.data.get('position_basis')!=BASIS or not self.data.get('records')):
            raise ValueError('invalid STOP boundary catalog')
        self.verify()
    def verify(self):
        if stamp(self.path)!=self.stamp:raise ValueError('STOP boundary catalog changed')
    def match(self,frame,actor):
        from .teacher_controls import owned_lane
        self.verify();lane=owned_lane(frame)
        if lane is None:raise ValueError('STOP approach lane unverified')
        key=f"{frame.get('route_id','').split('_')[0]}/{lane[0]}/{lane[1]}"
        record=self.data['records'].get(key)
        if record is None:raise ValueError('STOP approach absent or unsupported in catalog')
        world=g.world_point(frame,actor)
        self.check_control(record,actor,world)
        return dict(catalog_sha256=self.sha,catalog_key=key,map_sha256=record['map_sha256'],
                    signal_id=record['signal_id'],stop_id=actor['id'],world=world,
                    scenario=frame.get('scenario'),route_id=frame.get('route_id'),
                    established_frame=frame['frame_id'],sources=deepcopy(frame['sources']))
    @staticmethod
    def check_control(record,actor,world):
        if (world is None or list(actor['extent'])!=record['export_extent'] or
                math.dist(world[:2],record['trigger_xy'])>record['coordinate_budget_m']):
            raise ValueError('STOP control does not match fixed map geometry')
    def position(self,frame,actor,binding):
        self.verify()
        if binding['catalog_sha256']!=self.sha:raise ValueError('STOP catalog identity differs')
        record=self.data['records'][binding['catalog_key']]
        if (frame.get('route_id')!=binding['route_id'] or frame.get('scenario')!=binding['scenario']
                or frame.get('route_id','').split('_')[0]!=record['town'] or actor['id']!=binding['stop_id']):
            raise ValueError('STOP binding identity differs')
        self.check_control(record,actor,g.world_point(frame,actor))
        ego=next(a for a in frame['actors'] if a['class']=='ego_car');center=g.world_point(frame,ego)
        if center is None:raise ValueError('STOP ego pose invalid')
        matrix=frame['meta']['ego_matrix'];h=record['heading_rad'];normal=[math.cos(h),math.sin(h)]
        axes=[[matrix[0][j],matrix[1][j]] for j in range(3)]
        half=sum(ego['extent'][j]*abs(sum(normal[i]*axes[j][i] for i in range(2))) for j in range(3))
        return sum((record['endpoint_xy'][i]-center[i])*normal[i] for i in range(2))-half


@lru_cache(maxsize=2)
def _catalog(expected_stamp):
    result=Catalog()
    if result.stamp!=expected_stamp:raise ValueError('STOP catalog changed while reading')
    return result


def catalog():return _catalog(stamp(CATALOG))


def validate_evidence(record,question):
    binding=record.get('position_binding')
    if binding is None:
        if record['satisfied']:raise ValueError('completed STOP lacks fixed boundary binding')
        return
    c=catalog();c.verify();r=c.data['records'].get(binding.get('catalog_key'))
    if (r is None or binding.get('catalog_sha256')!=c.sha or binding.get('map_sha256')!=r['map_sha256']
            or binding.get('signal_id')!=r['signal_id'] or binding.get('stop_id')!=record['stop_id']
            or record.get('position_basis')!=BASIS or record.get('world')!=binding.get('world')
            or question['route_id'].split('_')[0]!=r['town']
            or any(binding.get(k)!=question.get(k) for k in ('scenario','route_id'))
            or record.get('lane')!=[r['road_id'],r['lane_id']]
            or record.get('established_frame')!=binding.get('established_frame')
            or not g.vector(binding.get('world'),3)
            or math.dist(binding['world'][:2],r['trigger_xy'])>r['coordinate_budget_m']):
        raise ValueError('STOP fixed boundary evidence mismatch')
    if record['satisfied']:
        from .teacher_controls import PARAMS
        if record.get('identity_verified') is not True or record.get('position_error'):
            raise ValueError('completed STOP has invalid current position evidence')
        for observation in record['observations']:
            gap=observation.get('front_gap_m');speed=observation.get('speed_mps')
            if (not g.finite(gap) or not 0<=gap<=PARAMS['max_front_gap_m'] or not g.finite(speed)
                    or abs(speed)>PARAMS['stop_speed_mps'] or observation.get('catalog_sha256')!=c.sha):
                raise ValueError('STOP observation fails position or speed evidence')
