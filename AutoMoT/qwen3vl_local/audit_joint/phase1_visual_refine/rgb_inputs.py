"""CPU-only, hash-bound RGB admission; no model imports, downloads or rebuilds."""
import json
from pathlib import Path
from .calibration import sha, resolve_rgb


def read_inputs(inputs):
    path=Path(inputs);digest=sha(path);frozen=json.loads(path.read_text())
    if sha(path)!=digest:raise ValueError('Frozen inputs changed while reading')
    if frozen['phase']!=1 or len(frozen['items'])!=frozen['expected_total'] or not frozen['items']:
        raise ValueError('Full Phase1 frozen membership required')
    if frozen['history_rgb_mode'] not in ('4rgb','2rgb_endpoints'):raise ValueError('Unsupported RGB mode')
    if frozen['arguments'].get('audit_prompt',False):raise ValueError('Use production judgment cases for this experiment')
    return frozen,digest


def verify_images(frozen,root):
    indices=(0,3) if frozen['history_rgb_mode']=='2rgb_endpoints' else (0,1,2,3)
    checked={};prepared=[]
    for index,value in enumerate(frozen['items']):
        row=value['row'];history=row['history_rgb_paths'];paths=[]
        if len(history)!=4:raise ValueError('Four ordered history members required')
        for slot in indices:
            raw=history[slot];parts=Path(raw).parts
            if '..' in parts or tuple(parts[-4:-2])!=(row['scenario'],row['route_id']):
                raise ValueError('Frozen RGB identity differs from its frame')
            local=resolve_rgb(root,str(Path(*parts[-4:])));name=str(local)
            if name not in checked:checked[name]=sha(local)
            if frozen['image_sha256'].get(raw)!=checked[name]:
                raise ValueError(f'Frozen RGB SHA mismatch: logical={raw}; resolved={local}; data_root={root}')
            paths.append(name)
        prepared.append(paths)
        if (index+1)%256==0:print(f'[RGB-CHECK] {index+1}/{len(frozen["items"])} cases; {len(checked)} images',flush=True)
    return prepared,checked


def admit(inputs,explicit_root=None,*,app):
    frozen,digest=read_inputs(inputs);app=Path(app)
    if explicit_root is not None:
        candidates=[Path(explicit_root).expanduser()]
    else:
        recorded=Path(frozen['arguments'].get('data_root') or 'lead_data').expanduser()
        candidates=[recorded if recorded.is_absolute() else app/recorded,app/'lead_data',app/'Automot/lead_data']
    attempted=[];seen=set()
    for root in candidates:
        root=root if root.is_absolute() else app/root
        if str(root.absolute()) in seen:continue
        seen.add(str(root.absolute()))
        print(f'[RGB-ROOT] checking {root}',flush=True)
        try:
            _,checked=verify_images(frozen,root)
        except FileNotFoundError as exc:
            attempted.append(str(exc));continue
        # Wrong contents, malformed membership and permission failures are hard
        # errors, never reasons to silently fall back to another dataset.
        if sha(inputs)!=digest:raise ValueError('Frozen inputs changed during admission')
        return root,dict(status='rgb_checked_cpu_only',data_root=str(root.absolute()),
                         frozen_input_sha256=digest,cases=len(frozen['items']),images=len(checked),
                         absent_candidates=attempted)
    raise FileNotFoundError('No complete frozen RGB dataset found; no inference started.\n'+'\n'.join(attempted))
