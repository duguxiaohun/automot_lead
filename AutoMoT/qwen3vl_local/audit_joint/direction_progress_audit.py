"""Read-only supplemental census for coarse direction x progress research.

Uses local CARLA Map/OpenDRIVE only; never connects to a simulation server.
Requires the existing waypoint census, numpy, and the local carla Python API.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, lzma, pickle, time, warnings
from pathlib import Path
import numpy as np
import carla
warnings.filterwarnings('ignore', category=DeprecationWarning)
MAPS = {}

def scan(job):
    entry, root, maps, dest = job
    town = entry['town']
    if town not in MAPS:
        MAPS[town] = carla.Map(town, (Path(maps)/(town+'.xodr')).read_text())
    mp=MAPS[town]; rows=[]; errors=[]; digest=hashlib.sha256()
    for p in sorted((Path(root)/entry['route']/'metas').glob('*.pkl')):
        raw=p.read_bytes();digest.update(p.name.encode()+b'\0'+hashlib.sha256(raw).digest())
        try:
            m=pickle.loads(lzma.decompress(raw) if raw.startswith(b'\xfd7zXZ') else raw)
            mat=np.asarray(m['ego_matrix'],float); yaw=np.arctan2(mat[1,0],mat[0,0])
            f=np.asarray(m['future_positions'],float)@mat[:3,:3].T
            arcs=np.full(4,np.nan); vel=np.full(4,np.nan); vmean=np.full(4,np.nan); vrange=np.full(4,np.nan)
            speeds=np.asarray(m['future_speeds'],float)
            for k,j in enumerate([10,20,30,40]):
                if len(f)>j and np.isfinite(f[:j+1]).all(): arcs[k]=np.linalg.norm(np.diff(f[:j+1,:2],axis=0),axis=1).sum()
                if len(speeds)>j:
                    vel[k]=speeds[j];vmean[k]=(speeds[:j].sum()+speeds[1:j+1].sum())*.025/(j*.05);vrange[k]=np.ptp(speeds[:j+1])
            w=mp.get_waypoint(carla.Location(*map(float,mat[:3,3])),project_to_road=True,lane_type=carla.LaneType.Driving)
            lane=np.full(9,np.nan)
            if w is not None:
                a=np.deg2rad(w.transform.rotation.yaw);d=mat[:2,3]-np.array([w.transform.location.x,w.transform.location.y])
                lane=[float(m['ego_lane_width']),w.lane_width,(-np.sin(a)*d[0]+np.cos(a)*d[1]),(a-yaw+np.pi)%(2*np.pi)-np.pi,
                      int(w.road_id==m['road_id']),int(w.lane_id==m['ego_lane_id']),int(w.lane_id==m['lane_id']),np.linalg.norm(d),float(m['is_junction'])]
            rows.append((int(p.stem),arcs,vel,vmean,vrange,lane))
        except Exception as e:errors.append({'file':p.name,'error':repr(e)})
    if rows:
        np.savez_compressed(Path(dest)/'routes'/f"{entry['id']:05d}.npz",frames=np.array([x[0] for x in rows]),arc=np.array([x[1] for x in rows],np.float32),velocity=np.array([x[2] for x in rows],np.float32),mean_velocity=np.array([x[3] for x in rows],np.float32),velocity_range=np.array([x[4] for x in rows],np.float32),lane=np.array([x[5] for x in rows],np.float32))
    return {'id':entry['id'],'rows':len(rows),'errors':errors,'meta_digest':digest.hexdigest(),'source_unchanged':digest.hexdigest()==entry['meta_digest']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--maps',required=True);ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=6);a=ap.parse_args()
    b=Path(a.base);out=Path(a.output);(out/'routes').mkdir(parents=True,exist_ok=False)
    entries=[json.loads(x) for x in (b/'route_manifest.jsonl').read_text().splitlines()];inv=json.loads((b/'inventory.json').read_text())
    (out/'inventory.json').write_text(json.dumps({'base':str(b.resolve()),'data_root':inv['data_root'],'maps':str(Path(a.maps).resolve()),'map_sha256':{t:hashlib.sha256((Path(a.maps)/(t+'.xodr')).read_bytes()).hexdigest() for t in sorted({e['town'] for e in entries})},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lane_columns':['recorded_width','map_width','ego_lateral_offset','lane_minus_ego_yaw','road_match','ego_lane_match','lane_match','center_distance','junction'],'horizons':[.5,1,1.5,2]},indent=2))
    start=time.monotonic();count=0
    with (out/'manifest.jsonl').open('x') as fh, concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(scan,[(e,inv['data_root'],a.maps,a.output) for e in entries],chunksize=1),1):
            fh.write(json.dumps(r)+'\n');fh.flush();count+=r['rows']
            if i%500==0 or i==len(entries):print(json.dumps({'routes':i,'rows':count,'seconds':round(time.monotonic()-start,1)}),flush=True)
    allrows=[]
    for e in entries:
        p=out/'routes'/f"{e['id']:05d}.npz"
        if p.exists():
            v=dict(np.load(p));old=np.load(b/'routes'/p.name);assert np.array_equal(v['frames'],old['frames']),e['id'];allrows.append(v)
    for k in allrows[0]:np.save(out/(k+'.npy'),np.concatenate([v[k] for v in allrows]))
    assert np.array_equal(np.load(out/'frames.npy'),np.load(b/'frames.npy'))
    print('Aligned complete',count,flush=True)

if __name__=='__main__': main()
