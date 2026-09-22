"""按米制轨迹终点距离筛选展示案例，不改变推理、采样或checkpoint选择。"""
from itertools import combinations
import math


def error_selection(predictions, enabled=False, threshold_m=1.0):
    """任一route/waypoint模型-GT或模型-模型终点距离严格超过阈值即保留。"""
    if not math.isfinite(threshold_m) or threshold_m <= 0:
        raise ValueError("error threshold must be finite and > 0 metres")
    if not predictions:
        raise ValueError("error selection requires predictions")
    maxima = {}
    triggers = []
    for kind, predkey, gtkey in (("route", "pred_route", "gt_route"),
                                ("waypoint", "pred_waypoints", "gt_waypoints")):
        def endpoint(data, key):
            try:
                point = data[key][0][-1]
                if len(point) != 2 or not all(math.isfinite(float(x)) for x in point):
                    raise ValueError
                return tuple(float(x) for x in point)
            except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid {key} endpoint") from exc
        gt = endpoint(next(iter(predictions.values())), gtkey)
        points = {mid: endpoint(data, predkey) for mid, data in predictions.items()}
        gt_distances = [(math.dist(point, gt), [mid, "GT"]) for mid, point in points.items()]
        pair_distances = [(math.dist(a, b), [ma, mb]) for (ma, a), (mb, b) in combinations(points.items(), 2)]
        for relation, distances in (("vs_gt", gt_distances), ("between_models", pair_distances)):
            distance, pair = max(distances, key=lambda item: item[0], default=(0., []))
            maxima[f"{kind}_{relation}_m"] = distance
            # 每种轨迹/关系仅记录最强触发者，最多4条，避免多模型JSON膨胀。
            if distance > threshold_m:
                triggers.append(dict(trajectory=kind, pair=pair, distance_m=distance))
    return dict(enabled=bool(enabled), threshold_m=threshold_m, kept=not enabled or bool(triggers),
                max_distances_m=maxima, triggers=triggers)
