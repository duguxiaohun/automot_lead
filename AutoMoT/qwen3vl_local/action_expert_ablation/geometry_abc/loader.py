"""One current RGB/LiDAR clip with the Phase3 observation coordinate contract.

No future images, route-lookahead navigation, or future-waypoint padding. The
original route smoothing/padding, sensor preprocessing and frozen BEV remain.
"""
from pathlib import Path
import io
import lzma
import pickle
import numpy as np
import torch
from PIL import Image
from qwen3vl_local.audit_joint.geometry_v1 import sources
from .data import validate_continuous


class Dataset(torch.utils.data.Dataset):
    def __init__(self,rows,args):
        from .consumption import Sources
        self.rows=rows;self.args=args
        self.sources=Sources(args.data_dir,args.data_root,getattr(args,'abc_dataset_identity',None),
                             getattr(args,'abc_manifest_sha256',None))
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        import laspy
        from qwen3vl_local.action_expert_ablation.common import leadmot_train
        old=leadmot_train();sample=self.rows[index];r=sample['abc_record'];route=Path(sample['route_dir'])
        try:
            raw,meta_sha=self.sources.read(r,route,'metas','pkl')
            meta=pickle.loads(lzma.decompress(raw) if raw.startswith(b'\xfd7zXZ') else raw)
            wp=validate_continuous(meta)
            if not np.allclose(wp,r['waypoint_xy'],rtol=0,atol=0):raise ValueError('runtime_GT_changed')
            matrix,yaw=sources.pose(meta);obs=r['observation'];nav=np.asarray(obs['navigation_xy'],dtype=np.float32)
            from qwen3vl_local.audit_joint.geometry_abc_v2 import regions
            endpoints=sources.future(meta,dict(speed_mps=obs['speed_mps'],reference=obs['reference']),anchor=r['frame'])
            for variant in 'abc':
                native_ids=[regions.relabel(t,variant,r['road_structure'])['token_id'] for t in endpoints]
                if native_ids!=[t['token_id'] for t in r['representations'][variant]['targets']]:
                    raise ValueError('runtime_native_four_endpoint_alignment_changed')
            sample=dict(sample,audit_geometry=dict(rotation_world_xy=matrix[:2,:3].tolist(),
                gt_endpoint_xyz=np.asarray(meta['future_positions'],dtype=np.float64)[[10,20,30,40]].tolist()))
            rgb_bytes,rgb_sha=self.sources.read(r,route,'rgb','jpg')
            with Image.open(io.BytesIO(rgb_bytes)) as image:rgb=np.asarray(image.convert('RGB'))
            lidar_bytes,lidar_sha=self.sources.read(r,route,'lidar','laz')
            cloud=laspy.read(io.BytesIO(lidar_bytes))
            sample['consumed_sources']=dict(metas=meta_sha,rgb=rgb_sha,lidar=lidar_sha)
            xyz=np.stack((cloud.x,cloud.y,cloud.z),axis=1).astype(np.float32)
            if not np.isfinite(xyz).all() or not len(xyz):raise ValueError('runtime_invalid_lidar')
            clip=dict(rgb=rgb[None],lidar_points=[xyz],pos_global=matrix[None,:2,3].astype(np.float32),
                theta=np.asarray([yaw],dtype=np.float32),speed=np.asarray([obs['speed_mps']],dtype=np.float32),
                target_point=nav[0:1],target_point_next=nav[1:2],final_goal=nav[2:3])
            route_gt=np.asarray(meta['route'],dtype=np.float32)
            route_gt=old._smooth_route(route_gt,self.args.route_points) if self.args.smooth_route else old._pad_rows(route_gt,self.args.route_points)
            if route_gt.shape!=(10,2) or not np.isfinite(route_gt).all():raise ValueError('invalid_runtime_route_target')
            return dict(sample=sample,clip=clip,gt_route=torch.from_numpy(route_gt),gt_waypoints=torch.tensor(wp,dtype=torch.float32))
        except (KeyError,ValueError,TypeError,OSError) as exc:
            return dict(sample=sample,_error=f'{type(exc).__name__}: {exc}')


def collate(batch):
    if len(batch)!=1:raise ValueError('ABC_batch_one_required')
    return batch[0]


def make_loader(rows,args,*,rank,world_size,shuffle,epoch_seed):
    if shuffle:raise ValueError('ABC_consumes_frozen_plan_without_reshuffling')
    subset=rows[rank::world_size];workers=max(0,args.num_workers)
    kwargs=dict(batch_size=1,shuffle=False,num_workers=workers,collate_fn=collate,drop_last=False,pin_memory=False)
    if workers:
        kwargs.update(persistent_workers=bool(args.persistent_workers),prefetch_factor=max(1,args.prefetch_factor),
                      multiprocessing_context='spawn')
    return torch.utils.data.DataLoader(Dataset(subset,args),**kwargs),None


class Helpers:
    def __init__(self,old):self.old=old
    def __getattr__(self,name):return getattr(self.old,name)
    _make_loader=staticmethod(make_loader)
