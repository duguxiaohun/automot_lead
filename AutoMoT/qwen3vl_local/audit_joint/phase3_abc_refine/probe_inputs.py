"""Separate processor extrema from the train-only backward probe membership."""
def update(extrema,record,variant,packed):
    grid=packed['inputs']['image_grid_thw'].detach().cpu().tolist()
    item=dict(id=record['id'],split=record['split'],variant=variant,tokens=int(packed['tokens']),image_grid_thw=grid,
              visual_patches=sum(int(t)*int(h)*int(w) for t,h,w in grid))
    cell=extrema.setdefault(variant,{}).setdefault(record['split'],{})
    for metric in ('tokens','visual_patches'):
        old=cell.get(metric)
        if old is None or (-item[metric],item['id'])<(-old[metric],old['id']):cell[metric]=item


def selection(records,processor,variant):
    lookup={r['id']:r for r in records};backward=[];forward=[]
    if len(lookup)!=len(records):raise ValueError('duplicate_probe_members')
    for split in ('train','val','test'):
        for metric in ('tokens','visual_patches'):
            item=processor['extrema'][variant][split][metric];r=lookup.get(item['id'])
            if r is None or r['split']!=split or item['split']!=split or item['variant']!=variant:
                raise ValueError('probe_extremum_split_or_member_changed')
            if item['tokens']<=0 or item['visual_patches']<=0:raise ValueError('invalid_probe_length')
            target=backward if split=='train' else forward
            if not any(x['id']==item['id'] for x in target):target.append(item)
    for r in sorted((r for r in records if r['split']=='train'),key=lambda r:r['id']):
        if len(backward)==4:break
        if not any(x['id']==r['id'] for x in backward):backward.append(dict(id=r['id'],split='train',variant=variant))
    if len(backward)!=4:raise ValueError('four_train_probe_members_required')
    return dict(schema='refine_train_extrema_probe_v2',variant=variant,backward=backward,
                holdout_forward_only=forward,holdout_updates=False)


def verify_shape(packed,item):
    if 'tokens' not in item:return
    grid=packed['inputs']['image_grid_thw'].detach().cpu().tolist()
    if packed['tokens']!=item['tokens'] or grid!=item['image_grid_thw']:
        raise ValueError('probe_actual_processor_shape_changed')


def holdout_forward(bundle,records,items,pack_record,readout):
    import torch
    lookup={r['id']:r for r in records};proofs=[];was_training=bundle.model.training;bundle.model.eval()
    try:
        with torch.no_grad():
            for item in items:
                r=lookup[item['id']]
                if r['split'] not in ('val','test') or r['split']!=item['split']:raise ValueError('holdout_forward_split_changed')
                packed=pack_record(r);verify_shape(packed,item)
                logits=readout(bundle,packed)
                if not torch.isfinite(logits).all():raise ValueError('nonfinite_holdout_forward')
                proofs.append(dict(item,updates=0))
    finally:bundle.model.train(was_training)
    return proofs
