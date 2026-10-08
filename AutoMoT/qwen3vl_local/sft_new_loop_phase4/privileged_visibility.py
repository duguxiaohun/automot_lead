"""Image-level veto for geometry proposals, not a certified visibility detector.

A bright/high-contrast crop can contain background instead of the actor. Passing
this gate is only necessary evidence; production remains unapproved. Dark,
clipped, tiny, or unsupported projections remain UNKNOWN, never clear space.
"""
import math
import numpy as np
from PIL import Image

POLICY='projected_rgb_quality_v1'
# Development veto limits; passing does not certify actor visibility.
MIN_P95=64.0
MIN_CONTRAST=20.0


def image_evidence(path, frame):
    cameras=frame['meta'].get('sensor_information',{}).get('camera_calibration',{})
    with Image.open(path) as image: rgb=np.asarray(image.convert('RGB'),dtype=np.float32)
    results={};offset=0;tiles=[]
    # LEAD concatenates camera indices left -> front -> right.
    for key in sorted(cameras,key=lambda k:int(k)):
        camera=cameras[key];w=camera.get('width');h=camera.get('height');ch=camera.get('cropped_height',h)
        if not all(type(v) is int and v>0 for v in (w,h,ch)) or ch!=h:return {}
        tiles.append((offset,camera));offset+=w
    if offset!=rgb.shape[1] or any(c['height']!=rgb.shape[0] for _,c in tiles):return {}
    for actor in frame['actors']:
        if actor['class']=='ego_car':continue
        samples=[]
        for offset,c in tiles:
            if any(abs(v)>1e-6 for v in c['rot'][:2]):continue
            yaw=math.radians(c['rot'][2]);cy,sy=math.cos(yaw),math.sin(yaw)
            ax,ay,az=actor['position'];ex,ey,ez=actor['extent'];co,si=math.cos(actor['yaw']),math.sin(actor['yaw'])
            points=[]
            # Actor positions are actor origins; extents do not record local box
            # center height. Project a conservative vertical [z-ez,z+2ez] range.
            for x in (-ex,ex):
                for y in (-ey,ey):
                    for z in (-ez,2*ez):
                        dx=ax+co*x-si*y-c['pos'][0];dy=ay+si*x+co*y-c['pos'][1];dz=az+z-c['pos'][2]
                        depth=cy*dx+sy*dy;horizontal=-sy*dx+cy*dy
                        if depth<=.1:continue
                        focal=c['width']/(2*math.tan(math.radians(c['fov']/2)))
                        points.append((c['width']/2+focal*horizontal/depth,c['height']/2-focal*dz/depth))
            if len(points)!=8:continue
            x0=max(0,math.floor(min(x for x,y in points)));x1=min(c['width'],math.ceil(max(x for x,y in points)))
            y0=max(0,math.floor(min(y for x,y in points)));y1=min(c['height'],math.ceil(max(y for x,y in points)))
            if x1-x0<4 or y1-y0<8:continue
            crop=rgb[y0:y1,offset+x0:offset+x1];gray=crop@np.array([.2126,.7152,.0722])
            p10,p90,p95=np.percentile(gray,[10,90,95]);clipped=float((gray>=250).mean())
            samples.append(dict(bounds=[offset+x0,y0,offset+x1,y1],p95=float(p95),contrast=float(p90-p10),
                                saturated_fraction=clipped,usable=bool(p95>=MIN_P95 and p90-p10>=MIN_CONTRAST and clipped<.5)))
        results[str(actor['id'])]=dict(policy=POLICY,crops=samples,quality_pass=any(s['usable'] for s in samples))
    return results


def scene_evidence(path,frame):
    """Weak empty-space quality veto; not a proof of unoccluded road coverage."""
    cameras=frame['meta'].get('sensor_information',{}).get('camera_calibration',{})
    if not cameras:return dict(quality_pass=False,reason='camera_calibration_missing')
    with Image.open(path) as image:rgb=np.asarray(image.convert('RGB'),dtype=np.float32)
    offset=0;checks=[]
    for key in sorted(cameras,key=lambda k:int(k)):
        c=cameras[key];w,h=c.get('width'),c.get('height')
        if type(w) is not int or type(h) is not int or w<=0 or h!=rgb.shape[0]:
            return dict(quality_pass=False,reason='camera_shape_unresolved')
        crop=rgb[h//2:,:][..., :3][:,offset+w//4:offset+3*w//4]
        if not crop.size:return dict(quality_pass=False,reason='camera_crop_empty')
        gray=crop@np.array([.2126,.7152,.0722]);p10,p90,p95=np.percentile(gray,[10,90,95])
        checks.append(bool(p95>=MIN_P95 and p90-p10>=MIN_CONTRAST and (gray>=250).mean()<.5))
        offset+=w
    return dict(quality_pass=offset==rgb.shape[1] and all(checks),camera_checks=checks,
                scope='RGB brightness/contrast veto only; occupancy checked separately')
