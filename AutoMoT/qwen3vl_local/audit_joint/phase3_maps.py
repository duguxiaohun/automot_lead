"""Resolve known map locations with content checks; no broad filesystem search."""
from pathlib import Path
from .geometry_v1.sources import file_sha

TOWNS=tuple([f'Town{i:02d}' for i in range(1,8)]+['Town10HD','Town11','Town12','Town13','Town15'])

def candidates(base,town):
    return [p for p in (base/f'{town}.xodr',base/'OpenDrive'/f'{town}.xodr',
        base/town/'OpenDrive'/f'{town}.xodr') if p.is_file()]

def discover(app,explicit=None):
    app=Path(app).resolve()
    roots=([Path(explicit).resolve(strict=True)] if explicit is not None else [
        app/'CARLA_0915/CarlaUE4/Content/Carla/Maps',
        app/'lead/3rd_party/fail2drive/toolbox/carla_xodr',
        app.parent/'lead/3rd_party/fail2drive/toolbox/carla_xodr'])
    # Choose one complete root; never silently mix maps from different installs.
    incomplete=[]
    for base in roots:
        selected={};missing=[]
        for town in TOWNS:
            paths=candidates(base,town)
            if not paths:missing.append(town);continue
            hashes={file_sha(p) for p in paths}
            if len(hashes)!=1:raise ValueError(f'conflicting_project_maps: {base}: {town}')
            path=paths[0].resolve(strict=True)
            selected[town]=dict(path=str(path),sha256=next(iter(hashes)))
        if not missing:return dict(schema='phase3_map_resolution_v1',root=str(base.resolve()),towns=selected)
        incomplete.append(dict(root=str(base),missing=missing))
    raise ValueError('no_complete_known_map_root: '+str(incomplete))

def resolve(app,view,explicit=None):
    proof=discover(app,explicit);view=Path(view)
    # Check the entire existing view before creating any new link.
    for town,item in proof['towns'].items():
        link=view/(town+'.xodr')
        if link.exists() or link.is_symlink():
            if not link.is_file() or file_sha(link)!=item['sha256']:
                raise ValueError('existing_map_view_changed: '+str(link))
    view.mkdir(parents=True,exist_ok=True)
    for town,item in proof['towns'].items():
        source=Path(item['path']);link=view/(town+'.xodr')
        if file_sha(source)!=item['sha256']:raise ValueError('map_changed_during_resolution')
        if not link.exists():link.symlink_to(source)
        if file_sha(link)!=item['sha256']:raise ValueError('map_changed_during_resolution')
    return view.resolve(strict=True),proof
