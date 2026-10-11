"""Source-bound, staged development RGB review; never automatic G2 approval.

prepare -> lock rgb2 -> reveal rgb4 -> lock rgb4 -> reveal history.
Hidden references stay in the original inventory. This is workflow separation,
not access control or evidence that the reviewer has never seen the case.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from .io import digest, file_sha, write_json
from qwen3vl_local.sft_new_loop_phase4.input_identity import rgb_content_sha

STAGES = ('rgb2', 'rgb4', 'history')


def inventory_rows(inventory):
    inventory = Path(inventory)
    receipt = json.loads((inventory / 'receipt.json').read_text())
    if set(receipt['files']) != {'input_evidence.jsonl', 'summary.json'}:
        raise ValueError('unexpected inventory receipt')
    for name, expected in receipt['files'].items():
        p = inventory / name
        if p.stat().st_size != expected['bytes'] or file_sha(p) != expected['sha256']:
            raise ValueError('inventory changed: ' + name)
    return inventory


def selected_rows(inventory, ids):
    inventory_rows(inventory)
    selected = {}
    with (Path(inventory) / 'input_evidence.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if row['id'] not in ids:
                continue
            key = row['id'], row['rgb_mode']
            if key in selected or row['split'] != 'train':
                raise ValueError('duplicate selection or non-training review')
            selected[key] = row
    if set(selected) != {(i, m) for i in ids for m in (2, 4)}:
        raise ValueError('missing paired input')
    for i in ids:
        a,b = selected[i,2],selected[i,4]
        if any(a[k] != b[k] for k in ('scenario','route_id','frame_id','event','edge','source_row_sha256')):
            raise ValueError('paired semantic identity mismatch')
    return selected


def prepare(inventory, data_root, ids, output, reviewer, prior_exposure):
    output = Path(output)
    if not ids or len(ids)>32 or len(set(ids)) != len(ids) or not reviewer.strip():
        raise ValueError('unique explicit IDs and reviewer required')
    inventory = Path(inventory).absolute()
    if prior_exposure not in ('known','unknown'):raise ValueError('invalid exposure status')
    if any(output.resolve().is_relative_to(Path(p).resolve()) for p in (inventory,data_root)):
        raise ValueError('output must be outside input inventory and raw data')
    if output.exists():raise FileExistsError('use a new review directory')
    rows = selected_rows(inventory, ids)
    output.mkdir(parents=True)
    spec = dict(policy='development_staged_rgb_review_v1', inventory=str(inventory),
        inventory_resolved=str(inventory.resolve()), receipt_sha256=file_sha(inventory/'receipt.json'),
        data_root=str(Path(data_root).absolute()), ids=ids, reviewer=reviewer,
        prior_exposure=prior_exposure, independent_approval=False, g2_status='not_run',
        source_rows={f'{i}/rgb{m}':digest(rows[i,m]) for i in ids for m in (2,4)})
    write_json(output/'request.json', spec)
    return reveal(output, 'rgb2')


def check_request(output):
    output=Path(output);spec=json.loads((output/'request.json').read_text());inv=Path(spec['inventory'])
    if str(inv.resolve()) != spec['inventory_resolved'] or file_sha(inv/'receipt.json') != spec['receipt_sha256']:
        raise ValueError('inventory binding changed')
    rows=selected_rows(inv,spec['ids'])
    if spec['source_rows'] != {f'{i}/rgb{m}':digest(rows[i,m]) for i in spec['ids'] for m in (2,4)}:
        raise ValueError('selected source rows changed')
    return spec,rows


def checked_image(root, source, pixel=None):
    rel=Path(source['path'])
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('invalid source path')
    p=Path(root)/rel
    if file_sha(p) != source['sha256']:raise ValueError('source bytes changed: '+str(rel))
    with Image.open(p) as image:
        rgb=image.convert('RGB')
        if pixel is not None and rgb_content_sha(rgb) != pixel:
            raise ValueError('source pixels changed: '+str(rel))
        return rgb


def verify_stage(output, stage):
    output=Path(output);p=output/stage
    receipt=json.loads((p/'receipt.json').read_text())
    if receipt['request_sha256'] != file_sha(output/'request.json'):
        raise ValueError('request changed')
    for name, sha in receipt['files'].items():
        if Path(name).name != name or file_sha(p/name) != sha:raise ValueError('stage changed')
    if stage != 'rgb2':
        previous=STAGES[STAGES.index(stage)-1]
        verify_stage(output,previous)
        if receipt['previous_lock_sha256'] != file_sha(output/previous/'judgments.json'):
            raise ValueError('previous judgments changed')
    return receipt


def reveal(output, stage):
    if stage not in STAGES:raise ValueError('unknown review stage')
    output=Path(output);spec,rows=check_request(output);p=output/stage
    if p.exists():raise FileExistsError('stage already exists; never overwrite')
    previous=None
    if stage != 'rgb2':
        previous=STAGES[STAGES.index(stage)-1];verify_stage(output,previous)
        # A lock is explicit reviewer data, never created by a reveal request.
        lock=json.loads((output/previous/'judgments.json').read_text())
        if lock['stage_receipt_sha256'] != file_sha(output/previous/'receipt.json'):
            raise ValueError('judgment lock is not bound to stage')
    p.mkdir();public=[]
    for index, ident in enumerate(spec['ids']):
        row=rows[ident,2 if stage=='rgb2' else 4]
        inp=row['input'];sources=[dict(path=path,sha256=sha,frame_id=f,kind='rgb') for path,sha,f in
            zip(inp['images'],inp['file_sha256'],inp['history_frames'],strict=True)]
        pixels=inp['pixel_sha256']
        case=dict(case=index,id=ident,system=inp['system'],prompt=inp['prompt'],
            model_input_sha256=inp['model_input_sha256'],sources=sources,
            input_display='ordered original RGB; sheet resized for inspection; originals bound below')
        if stage=='history':
            evidence=row['evidence'];case['reference']=dict(target=row['reference_target'],label_basis=row['original_label_basis'],facts=row['facts'],evidence=evidence)
            sources=[s for s in evidence['sources'] if s['kind']=='rgb'] if evidence else sources
            for s in (evidence or {}).get('sources',[]):
                rel=Path(s['path'])
                if rel.is_absolute() or '..' in rel.parts or s['frame_id']>row['frame_id']:
                    raise ValueError('invalid/future evidence source')
                if file_sha(Path(spec['data_root'])/rel)!=s['sha256']:raise ValueError('history source changed')
            pixels=[None]*len(sources);case['sources']=sources
            case['scope']='causal evidence envelope only; not a complete route or minimal necessary history'
        images=[checked_image(spec['data_root'],s,pixel) for s,pixel in zip(sources,pixels,strict=True)]
        # No target, prediction, fact or hidden provenance in early-stage sheets.
        sheet=Image.new('RGB',(768,280*len(images)),'white');draw=ImageDraw.Draw(sheet)
        for n,(im,s) in enumerate(zip(images,sources,strict=True)):
            im.thumbnail((768,256));sheet.paste(im,(0,n*280));draw.text((4,n*280+258),f"case {index} / frame {s['frame_id']}",fill='black')
        name=f'case_{index:03d}.jpg';sheet.save(p/name,quality=92);case['sheet']=name;public.append(case)
    write_json(p/'cases.json',public)
    # Catch concurrent inventory/request drift before publishing success.
    check_request(output)
    write_json(p/'receipt.json',dict(request_sha256=file_sha(output/'request.json'),stage=stage,
        previous_lock_sha256=file_sha(output/previous/'judgments.json') if previous else None,
        files={f.name:file_sha(f) for f in p.iterdir() if f.is_file()}))
    return public


def lock(output, stage, judgments):
    if stage not in STAGES:raise ValueError('unknown review stage')
    output=Path(output);spec,_=check_request(output);verify_stage(output,stage)
    for case in json.loads((output/stage/'cases.json').read_text()):
        for source in case['sources']:
            checked_image(spec['data_root'],source)
    if len(judgments)!=len(spec['ids']) or {j.get('id') for j in judgments}!=set(spec['ids']):
        raise ValueError('one judgment for every selected ID required')
    for j in judgments:
        if j.get('answer') not in ('YES','NO','UNKNOWN') or not str(j.get('reason','')).strip():
            raise ValueError('explicit answer and visual reason required')
    value=dict(stage=stage,reviewer=spec['reviewer'],prior_exposure=spec['prior_exposure'],
        stage_receipt_sha256=file_sha(output/stage/'receipt.json'),judgments=judgments,
        independent_approval=False,g2_status='not_run')
    # Exclusive creation preserves first decisions; corrections need a new session.
    with (output/stage/'judgments.json').open('x') as f:
        f.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--inventory',required=True);a.add_argument('--data-root',required=True);a.add_argument('--ids',nargs='+',required=True);a.add_argument('--output',required=True);a.add_argument('--reviewer',required=True);a.add_argument('--prior-exposure',choices=('known','unknown'),default='known')
    for name in ('reveal','lock'):
        a=sub.add_parser(name);a.add_argument('--output',required=True);a.add_argument('--stage',choices=STAGES,required=True)
        if name=='lock':a.add_argument('--judgments',required=True)
    a=vars(p.parse_args());command=a.pop('command')
    if command=='lock':a['judgments']=json.loads(Path(a['judgments']).read_text())
    globals()[command](**a)


if __name__=='__main__':main()
