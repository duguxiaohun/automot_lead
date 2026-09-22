"""只按waypoint对应时刻的二维距离筛选：ADE>1m或FDE>3m。"""
from itertools import combinations
import math


def error_selection(predictions, enabled=False, ade_threshold_m=1., fde_threshold_m=3.):
    if any(not math.isfinite(t) or t <= 0 for t in (ade_threshold_m, fde_threshold_m)):
        raise ValueError('error thresholds must be finite and > 0 metres')
    if not predictions:
        raise ValueError('error selection requires predictions')
    def trajectory(data, key):
        try:
            values = data[key]
            if len(values) != 1 or not values[0]:
                raise ValueError
            points = [tuple(float(x) for x in p) for p in values[0]]
            if any(len(p) != 2 or not all(math.isfinite(x) for x in p) for p in points):
                raise ValueError
            return points
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'invalid {key} trajectory') from exc
    gt = trajectory(next(iter(predictions.values())), 'gt_waypoints')
    points = {mid: trajectory(data, 'pred_waypoints') for mid, data in predictions.items()}
    if any(len(p) != len(gt) for p in points.values()):
        raise ValueError('waypoint length mismatch; cannot compare corresponding times')
    if any(trajectory(data, 'gt_waypoints') != gt for data in predictions.values()):
        raise ValueError('waypoint GT mismatch')
    comparisons = {'vs_gt': [(mid, 'GT', p, gt) for mid, p in points.items()],
                   'between_models': [(ma, mb, a, b) for (ma, a), (mb, b) in combinations(points.items(), 2)]}
    maxima, triggers = {}, []
    for relation, pairs in comparisons.items():
        values = []
        for ma, mb, a, b in pairs:
            distances = [math.dist(x, y) for x, y in zip(a, b)]
            values.append((ma, mb, sum(distances)/len(distances), distances[-1]))
        for index, metric, threshold in ((2, 'ade', ade_threshold_m), (3, 'fde', fde_threshold_m)):
            strongest = max(values, key=lambda v: v[index], default=('','',0.,0.))
            maxima[f'waypoint_{relation}_{metric}_m'] = strongest[index]
            if strongest[index] > threshold:
                triggers.append(dict(trajectory='waypoint', metric=metric, pair=list(strongest[:2]),
                                     distance_m=strongest[index], threshold_m=threshold))
    return dict(enabled=bool(enabled), ade_threshold_m=ade_threshold_m, fde_threshold_m=fde_threshold_m,
                kept=not enabled or bool(triggers), max_distances_m=maxima, triggers=triggers)
