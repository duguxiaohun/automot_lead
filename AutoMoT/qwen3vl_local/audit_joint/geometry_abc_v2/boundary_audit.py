"""Train-only candidate diagnostics. Report alternatives without relabeling data."""
from collections import Counter, defaultdict

def inspect(records):
    counts=Counter();groups=defaultdict(set);near=Counter();total=0
    for r in records:
        if r['split']!='train': continue
        for i,t in enumerate(r['targets']):
            if not t['valid']:continue
            if 'radius_m' not in t or 'lateral_m' not in t:
                raise ValueError('boundary_audit_requires_native_measurements')
            total+=1
            side='LEFT' if t['lateral_m'] < -2 else 'RIGHT' if t['lateral_m'] > 2 else 'CENTER'
            for inner in (3,4,5):
                radial='NEAR' if t['radius_m']<inner else 'FAR' if t['radius_m']>13 else 'MIDDLE'
                key=f'{inner}/horizon_{i}/{side}_{radial}'
                counts[key]+=1;groups[key].add(r['physical_group'])
                if radial=='NEAR':near[str(inner)]+=1
    return dict(schema='train_only_fixed_boundary_candidates_v1',split='train',outer_m=13,
        inner_candidates_m=[3,4,5],valid_queries=total,counts=dict(counts),
        physical_groups={k:len(v) for k,v in groups.items()},
        near_fraction={k:v/total for k,v in near.items()} if total else {},
        thresholds_changed=False,selected_inner_m=3,RGB_review_completed=False,
        decision='Retain 3/13 baseline; larger radii need train boundary RGB/trajectory review before a new label version.')
