"""Endpoint metrics with missingness, boundary, speed, event and route support."""
from collections import Counter, defaultdict
import math

from . import codec


def evaluate(cases):
    totals=Counter(); hits=Counter(); raw=Counter(); support=Counter(); correct=Counter()
    confusion=[[0]*9 for _ in range(9)]; groups=defaultdict(set); missing=Counter()
    for case in cases:
        record=case['record']; prediction=case['prediction']
        if len(prediction['token_ids'])!=4 or len(record['targets'])!=4:
            raise ValueError('four_endpoint_metrics_required')
        v=record['observation']['speed_mps']
        for i,(t,p) in enumerate(zip(record['targets'],prediction['token_ids'])):
            if not t['valid']:
                missing[str(i)]+=1; continue
            if type(p) is not int or not 0<=p<9: raise ValueError('invalid_prediction_id')
            label=t['token_id']; confusion[label][p]+=1
            cell=f'{codec.HORIZONS[i]}/{codec.NAMES[label]}'
            support[cell]+=1; correct[cell]+=label==p
            keys=['all',f'horizon/{codec.HORIZONS[i]}',f"event/{record['event']}",
                  'speed/'+('low' if v<2 else 'medium' if v<5 else 'high'),
                  'boundary/'+('within_0.1m' if min(t['radial_margin_m'],t['lateral_margin_m'])<=.1 else 'interior')]
            ref=record['observation']['reference']
            if ref.get('is_junction'):keys.append('reference/junction')
            if ref.get('flipped'):keys.append('reference/oriented_against_lane')
            change=record['observation'].get('reference_step_change_rad')
            if change is not None and abs(change)>math.pi/4:keys.append('reference/step_change_gt_45deg')
            for key in keys:
                totals[key]+=1; hits[key]+=p==label
                raw[key]+=bool(prediction['format_valid'][i]) and p==label
                groups[key].add(record['physical_group'])
    return dict(counts=dict(totals), accuracy={k:hits[k]/n for k,n in totals.items()},
        full_vocabulary_accuracy={k:raw[k]/n for k,n in totals.items()},
        per_horizon_class_support=dict(support), per_horizon_class_recall={k:correct[k]/n for k,n in support.items()},
        unsupported_cells=[f'{t}/{name}' for t in codec.HORIZONS for name in codec.NAMES if not support[f'{t}/{name}']],
        confusion=confusion, missing_targets=dict(missing), physical_groups={k:len(v) for k,v in groups.items()},
        adjacent_frames_are_not_independent=True)
