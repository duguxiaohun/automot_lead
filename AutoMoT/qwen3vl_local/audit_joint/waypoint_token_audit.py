"""Read-only full local trajectory census. No labels, training, or data rewrites.
Run with a numpy-enabled Python. Results are local audit artifacts, not a dataset.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, lzma, os, pickle, re, time, warnings
from collections import Counter
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore', category=DeprecationWarning)

def planar(local, matrix, yaw):
    world = np.asarray(local, dtype=np.float64) @ matrix[:3,:3].T
    c,s=np.cos(yaw),np.sin(yaw)
    return np.stack([world[:,0]*c+world[:,1]*s,-world[:,0]*s+world[:,1]*c],-1)

def scan_route(job):
    rid, root, rel, out=job; run=Path(root)/rel
    files=sorted((run/'metas').glob('*.pkl')); digest=hashlib.sha256(); errors=[]; rows=[]; poses={}; checks=[]
    cnt=Counter(); active_names=[]
    for p in files:
        cnt['files']+=1
        try:
            stat=p.stat(); raw=p.read_bytes(); after=p.stat()
            if (stat.st_size,stat.st_mtime_ns,stat.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino): raise ValueError('changed_during_read')
            digest.update(p.name.encode()+b'\0'+hashlib.sha256(raw).digest());cnt['bytes']+=len(raw)
            m=pickle.loads(lzma.decompress(raw) if raw.startswith(b'\xfd7zXZ') else raw)
            fid=int(p.stem); mat=np.asarray(m['ego_matrix'],float); observed_yaw=float(m['theta']); yaw=float(np.arctan2(mat[1,0],mat[0,0])); speed=float(m['speed'])
            pos=np.asarray(m['pos_global'],float)
            if mat.shape!=(4,4) or pos.shape!=(3,) or not np.isfinite(mat).all() or not np.isfinite([yaw,speed,*pos]).all(): raise ValueError('invalid_pose')
            freq=m.get('dataset_information',{}).get('data_save_freq');cnt['save_freq_'+str(freq)]+=1
            f=np.asarray(m['future_positions'],float)
            if f.ndim!=2 or f.shape[1]!=3: raise ValueError('invalid_future_shape')
            poses[fid]=(mat[:3,3],yaw,speed)
            n=min(len(f),41); xy=planar(f[:n],mat,yaw)
            wp=np.full((8,2),np.nan); valid=np.zeros(8,bool)
            for k,idx in enumerate(range(5,41,5)):
                if idx<n and np.isfinite(xy[:idx+1]).all():wp[k]=xy[idx];valid[k]=True
            hist=np.full((4,2),np.nan); hist_observed=np.full((4,2),np.nan)
            for key,dest in [('privileged_past_positions',hist),('past_positions',hist_observed)]:
                h=np.asarray(m.get(key,[]),float)
                if h.ndim==2 and h.shape[1]>=2:
                    for k,idx in enumerate([10,20,30,40]):
                        if idx<len(h):dest[k]=h[idx,:2]
            ext=np.array([np.nan,np.nan,np.nan,np.nan]); maxstep=np.nan; arclen=np.nan
            if valid[-1]:
                ext=np.array([xy[:,0].min(),xy[:,0].max(),xy[:,1].min(),xy[:,1].max()]); steps=np.linalg.norm(np.diff(xy,axis=0),axis=1);maxstep=steps.max();arclen=steps.sum();cnt['complete_2s']+=1
            else:cnt['incomplete_2s']+=1
            fy=np.asarray(m.get('future_yaws',[]),float); turn=float(fy[40]) if len(fy)>40 else np.nan
            fs=np.asarray(m.get('future_speeds',[]),float); vmax=float(np.max(np.abs(fs[:41]))) if len(fs)>=41 and np.isfinite(fs[:41]).all() else np.nan
            active=str(m.get('current_active_scenario_type') or 'NONE');active_names.append(active)
            # bit 0 discontinuity; bit 1 negative speed; bit 2 nonfinite; bit 3 unexpected save frequency
            flag=int(np.isfinite(maxstep) and maxstep>5.) + 2*int(speed<-.1) + 4*int(not np.isfinite(f[:n]).all()) + 8*int(freq!=5)
            cnt['suspect']+=flag!=0
            rows.append((fid,wp,hist,hist_observed,ext,[speed,turn,maxstep,arclen,vmax,float(m.get('is_junction',False)),float(m.get('is_intersection',False)),flag,float(np.linalg.norm(pos-mat[:3,3])),float(np.max(np.abs(f[0]))) if len(f) else np.nan,float(np.nanmax(np.abs(wp-f[np.arange(5,41,5),:2]))) if len(f)>40 else np.nan,float((observed_yaw-yaw+np.pi)%(2*np.pi)-np.pi),float((float(m.get('privileged_yaw',np.nan))-yaw+np.pi)%(2*np.pi)-np.pi)]))
            checks.append((fid,wp,mat[:3,3],yaw))
        except Exception as e:
            cnt['errors']+=1;errors.append(dict(file=p.name,error=type(e).__name__+': '+str(e)))
    cross=[]
    for row,(fid,wp,origin,yaw) in zip(rows,checks):
        row[5][1]=float((poses[fid+8][1]-yaw+np.pi)%(2*np.pi)-np.pi) if fid+8 in poses else np.nan
        c,s=np.cos(yaw),np.sin(yaw)
        for k,offset in enumerate(range(1,9)):
            if fid+offset in poses and np.isfinite(wp[k]).all():
                d=poses[fid+offset][0]-origin; q=np.array([d[0]*c+d[1]*s,-d[0]*s+d[1]*c]);cross.append(float(np.linalg.norm(q-wp[k])))
    if rows:
        np.savez_compressed(Path(out)/'routes'/f'{rid:05d}.npz',frames=np.array([r[0] for r in rows],np.int32),xy=np.array([r[1] for r in rows],np.float32),history=np.array([r[2] for r in rows],np.float32),history_observed=np.array([r[3] for r in rows],np.float32),extent=np.array([r[4] for r in rows],np.float32),features=np.array([r[5] for r in rows],np.float32),active=np.array(active_names))
    physical=re.sub(r'_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$','',rel);physical=re.sub(r'_route0$','',physical);physical=re.sub(r'_Rep\d+_','_',physical)
    return dict(id=rid,route=rel,physical=physical,scenario=Path(rel).parts[0],town=Path(rel).parts[1].split('_')[0],counts=dict(cnt),meta_digest=digest.hexdigest(),errors=errors,cross_count=len(cross),cross_sum=sum(cross),cross_max=max(cross,default=0),cross_gt_01=sum(x>.1 for x in cross),cross_gt_1=sum(x>1 for x in cross))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=6);a=ap.parse_args()
    root=Path(a.data_root).resolve(strict=True);out=Path(a.output).resolve();(out/'routes').mkdir(parents=True,exist_ok=False)
    runs=sorted(p for s in root.iterdir() if s.is_dir() for p in s.iterdir() if p.is_dir() and (p/'metas').is_dir())
    real=[str(p.resolve()) for p in runs]
    if len(real)!=len(set(real)):raise ValueError('Aliased physical directories; deduplicate explicitly')
    jobs=[(i,str(root),str(p.relative_to(root)),str(out)) for i,p in enumerate(runs)]
    (out/'inventory.json').write_text(json.dumps(dict(data_root=str(root),routes=[j[2] for j in jobs],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),meta_dt=.25,native_dt=.05,source='all local metas, including routes excluded from training',feature_columns=['speed','yaw_2s','max_005s_step','arc_2s','max_speed_2s','is_junction','is_intersection','suspect_flags','position_matrix_error','origin_residual','planar_body_max_difference','observed_matrix_yaw_difference','privileged_matrix_yaw_difference'],coordinate_frame='origin ego_matrix translation; heading atan2(R10,R00); horizontal yaw-only ego frame; native body-frame difference audited'),indent=2))
    t=time.monotonic(); counts=Counter()
    with (out/'route_manifest.jsonl').open('x') as fh, concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(scan_route,jobs,chunksize=1),1):
            fh.write(json.dumps(r)+'\n');fh.flush();counts.update(r['counts'])
            if i%100==0 or i==len(jobs): print(json.dumps(dict(routes=i,total=len(jobs),seconds=round(time.monotonic()-t,1),counts=dict(counts))),flush=True)
    (out/'scan_summary.json').write_text(json.dumps(dict(counts=dict(counts),routes=len(jobs),seconds=time.monotonic()-t),indent=2))
if __name__=='__main__':main()
