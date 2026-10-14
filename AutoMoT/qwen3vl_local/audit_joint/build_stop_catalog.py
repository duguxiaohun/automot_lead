"""Derive compact STOP approach geometry from existing maps, never from labels.

The catalog is a frozen teacher asset. Runtime must match the observed control
and incoming road/lane; the catalog is not a claim of capture-map provenance or
painted-line accuracy. Full maps are unnecessary on training servers.
"""
import argparse
import json
import math
from pathlib import Path
from .stop_line_probe import MapProbe,number,road_point,lane_center,coordinate_budget
from .io import file_sha,write_json

POLICY='stop_incoming_boundary_catalog_v1'


def build(paths):
    records={};maps={};excluded=[]
    for path in sorted(map(Path,paths)):
        probe=MapProbe(path);town=path.stem;maps[town]=dict(name=path.name,sha256=probe.sha)
        candidates={}
        for junction in probe.junctions.values():
            for connection in junction.findall('connection'):
                for lane in connection.findall('laneLink'):
                    key=connection.get('incomingRoad'),int(lane.get('from'))
                    candidates.setdefault(key,[]).append(dict(connection.attrib,lane_link=dict(lane.attrib)))
        for (road_id,lane),connections in sorted(candidates.items()):
            if not lane:continue
            key=f'{town}/{road_id}/{lane}'
            try:
                road=probe.roads[road_id]
                if road.get('rule','RHT')!='RHT' or road.get('junction','-1')!='-1':raise ValueError('unsupported approach')
                reference=probe.control_reference(road,lane,connections,[0.,0.,0.])
                if reference['status']!='unique_lane_stop_reference':
                    # Most junction approaches have no STOP. They are absent,
                    # not assigned a nearby sign or a guessed boundary.
                    continue
                endpoint=number(road,'length') if lane<0 else 0.
                xy,h=road_point(road,endpoint,lane_center(road,lane,endpoint))
                if lane>0:h+=math.pi
                trigger=reference['upstream_predicted_trigger'];budget=coordinate_budget(trigger['xy'])
                records[key]=dict(town=town,road_id=int(road_id),lane_id=lane,
                    signal_id=reference['signal_id'],map_sha256=probe.sha,
                    endpoint_xy=xy,heading_rad=h,trigger_xy=trigger['xy'],coordinate_budget_m=budget,
                    export_extent=[1.5,1.5,.5],
                    connections=[dict(road=r['connecting_road'],lane=r['target_lane']) for r in reference['references']])
            except (ValueError,KeyError,TypeError) as exc:excluded.append(dict(key=key,error=str(exc)))
        probe.verify_unchanged()
    return dict(policy=POLICY,position_basis='incoming_lane_end_plane_not_painted_stop_line',
        scope='automatic weak STOP completion: three low-speed observations with the ego front 0–3m before its matched incoming boundary; not capture-map authentication',
        maps=maps,records=records,excluded=excluded,
        generator_sources={p.name:file_sha(p) for p in (Path(__file__),Path(__file__).with_name('stop_line_probe.py'))})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--map-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve frozen catalog')
    result=build(a.map_dir.glob('*.xodr'))
    if not result['records']:raise ValueError('no supported STOP approaches')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(dict(records=len(result['records']),maps=len(result['maps']),excluded=len(result['excluded']))))

if __name__=='__main__':main()
