"""Actual processor extrema split by variant and train/val/test."""
import json
from pathlib import Path
from qwen3vl_local.audit_joint.abc_delivery import sha


def shape(packed,processor):
    grid=packed['image_grid_thw'].detach().cpu().tolist()
    merge=int(processor.image_processor.merge_size)
    patches=sum(int(t)*int(h)*int(w) for t,h,w in grid)
    if len(grid)!=1 or merge<=0 or patches%(merge*merge):raise ValueError('invalid_processor_visual_shape')
    return dict(tokens=int(packed['input_ids'].shape[-1]),image_grid_thw=grid,visual_tokens=patches//(merge*merge))


def update(extrema,record,variant,dimensions):
    cell=extrema.setdefault(variant,{}).setdefault(record['split'],{})
    item=dict(id=record['id'],split=record['split'],variant=variant,**dimensions)
    for metric in ('tokens','visual_tokens'):
        old=cell.get(metric)
        if old is None or (-item[metric],item['id'])<(-old[metric],old['id']):cell[metric]=item


def selection(rows,processor,variant):
    records=[r.get('abc_record',r) for r in rows];lookup={r['id']:r for r in records}
    if len(lookup)!=len(records):raise ValueError('duplicate_probe_members')
    extrema=processor['extrema'][variant];selected=[];heldout=[]
    for split in ('train','val','test'):
        if split not in extrema:raise ValueError('missing_split_processor_extrema')
        for metric in ('tokens','visual_tokens'):
            item=extrema[split][metric];r=lookup.get(item['id'])
            if r is None or r['split']!=split or item['split']!=split or item['variant']!=variant:
                raise ValueError('probe_extremum_split_or_member_changed')
            if item['tokens']<=0 or item['visual_tokens']<=0:raise ValueError('invalid_probe_length')
            target=selected if split=='train' else heldout
            if not any(x['id']==item['id'] for x in target):target.append(item)
    train=sorted((r for r in records if r['split']=='train'),key=lambda r:r['id'])
    for r in train:
        if len(selected)==4:break
        if not any(x['id']==r['id'] for x in selected):selected.append(dict(id=r['id'],split='train',variant=variant))
    if len(selected)!=4:raise ValueError('four_distinct_train_probe_members_required')
    return dict(schema='abc_train_extrema_probe_v2',variant=variant,backward=selected,
                holdout_forward_only=heldout,holdout_updates=False,
                selection='train maximum text and visual tokens, then stable train IDs; holdout extrema inference only')


def verify_processor(path,gate):
    path=Path(path)
    if sha(path)!=gate['processor_sha256']:raise ValueError('admitted_processor_changed')
    result=json.loads(path.read_text())
    if (result['code']!=gate['code'] or result['dataset']!=gate['dataset'] or result['splits']!=gate['splits']
        or not result['full_pool'] or not result['no_truncation']):raise ValueError('processor_contract_changed')
    if 'model' in result and result['model']!=gate['resources']['qwen']:
        raise ValueError('processor_base_model_changed_since_admission')
    return result


def verify_actual_shape(actual,item,branch):
    if branch=='qwen_simple':
        if actual is None:raise ValueError('probe_actual_processor_shape_missing')
        for key in ('tokens','visual_tokens','image_grid_thw'):
            if key in item and actual[key]!=item[key]:raise ValueError('probe_processor_shape_changed')


def verify_probe_inputs(receipt,selection,processor_sha256,rank_directory):
    if receipt.get('input_selection')!=selection or receipt.get('processor_sha256')!=processor_sha256:
        raise ValueError('probe_input_selection_changed')
    files=receipt['rank_receipts']
    if set(files)!={f'rank{i}.json' for i in range(4)}:raise ValueError('four_probe_rank_evidence_required')
    holds=[]
    for rank in range(4):
        path=Path(rank_directory)/f'rank{rank}.json'
        if sha(path)!=files[path.name]:raise ValueError('probe_rank_evidence_changed')
        proof=json.loads(path.read_text());actual=proof['input']
        if actual['id']!=selection['backward'][rank]['id'] or actual['split']!='train':raise ValueError('probe_backward_split_changed')
        verify_actual_shape(actual['processor'],selection['backward'][rank],receipt['binding']['branch'])
        for item in proof['holdout_forward_only']:
            if item['split'] not in ('val','test') or item['updates']!=0:raise ValueError('holdout_probe_updated')
            if not item['decoder_shapes']:raise ValueError('holdout_actual_shapes_missing')
            holds.append({k:v for k,v in item.items() if k not in ('updates','decoder_shapes')})
        if not actual['decoder_shapes'] or not actual['rgb_shape'] or not actual['lidar_points']:
            raise ValueError('probe_actual_shapes_missing')
    if sorted(holds,key=lambda x:x['id'])!=sorted(selection['holdout_forward_only'],key=lambda x:x['id']):
        raise ValueError('holdout_probe_coverage_changed')
