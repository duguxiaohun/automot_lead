"""Validation support and sensitivity diagnostics; never choose a checkpoint."""
from collections import Counter, defaultdict

def stability(cases, names, mass):
    counts=Counter(); correct=Counter(); by_group=defaultdict(Counter); hits_group=defaultdict(Counter)
    for case in cases:
        r,p=case['record'],case['prediction']
        for t,choice in zip(r['targets'],p['token_ids']):
            if not t['valid']: continue
            q=mass(t,len(names))
            if 1. not in q: continue
            k=q.index(1.);g=r['physical_group']
            counts[k]+=1;correct[k]+=choice==k;by_group[g][k]+=1;hits_group[g][k]+=choice==k
    supported=sorted(counts)
    group_counts={k:sum(c[k]>0 for c in by_group.values()) for k in range(len(names))}
    stable=[k for k in supported if counts[k]>=20 and group_counts[k]>=5]
    leave=[];undefined=0
    for g,c in by_group.items():
        if any(counts[k]==c[k] for k in supported): undefined+=1;continue
        leave.append(sum((correct[k]-hits_group[g][k])/(counts[k]-c[k]) for k in supported)/len(supported))
    return dict(selection='final_completed_epoch; these diagnostics never select a checkpoint',
        class_support={n:counts[k] for k,n in enumerate(names)},
        class_physical_groups={n:group_counts[k] for k,n in enumerate(names)},
        sparse_classes=[n for k,n in enumerate(names) if counts[k]<20 or group_counts[k]<5],
        support_screen=dict(min_queries=20,min_groups=5,heuristic_not_certification=True),
        supported_subset_macro_recall=sum(correct[k]/counts[k] for k in stable)/len(stable) if stable else None,
        supported_subset_classes=[names[k] for k in stable],
        leave_one_group_out_macro_range=[min(leave),max(leave)] if leave else None,
        leave_one_group_out_undefined=undefined, confidence_interval=False)
