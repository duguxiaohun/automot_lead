"""统一对比的选优、配对与分层抽样；不导入模型，也不改训练合同。"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

AUTOMOT_ROOT = Path(__file__).resolve().parents[2]
EVENT_NAMES = tuple([f"UE{i}" for i in range(1, 8)] + ["RE2", "RE3", "RE5", "REGULAR_BACKGROUND", "UNCONFIRMED"])
ACTION_NAMES = ("DECELERATE", "STOP", "RESUME", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT", "KEEP", "UNCOND")


def read_json(path):
    """读取 UTF-8 JSON。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """原子发布可读 JSON，禁止非有限数。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def identity(row):
    """模型间帧身份不依赖服务器绝对路径。"""
    return str(row["scenario"]), str(row["run_id"]), int(row["anchor"])


def case_id(row):
    """保留帧号与身份哈希，避免路线名称和路径字符碰撞。"""
    key = identity(row)
    return f"frame_{key[2]:06d}_{hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]}"


def resolve_path(value, *, run=None):
    """当前目录优先，兼容 AutoMoT 目录和整仓搬迁后的旧路径。"""
    path = Path(value).expanduser()
    choices = [path]
    if not path.is_absolute():
        choices += [AUTOMOT_ROOT / path, AUTOMOT_ROOT.parent / path]
        if run is not None:
            choices.append(Path(run) / path)
    for marker in ("AutoMoT", "checkpoints", "lead_data"):
        if marker in path.parts:
            index = path.parts.index(marker)
            tail = Path(*path.parts[index + (marker == "AutoMoT"):])
            choices.append(AUTOMOT_ROOT / tail)
    for candidate in choices:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"无法定位 {value}；请保留原目录结构或使用路径覆盖参数")


def validation_score(metrics, config):
    """严格沿用训练的 best 指标，不能用 loss 或 test 选优。"""
    prefix = "event_balanced_" if config.get("best_selection_metric") == "event_balanced_ade" else ""
    if prefix and not metrics.get("event_balance_bucket_coverage_complete"):
        raise ValueError("event-balanced validation coverage incomplete")
    score = (float(config["route_loss_weight"]) * float(metrics[prefix + "route_ade_m"])
             + float(config["waypoint_loss_weight"]) * float(metrics[prefix + "waypoint_ade_m"]))
    if not math.isfinite(score):
        raise ValueError("nonfinite validation score")
    return score


def validation_records(run, config):
    """只收完整 val；同一步重复记录必须一致。"""
    records = {}
    for path in sorted((Path(run) / "validation").glob("*.json")):
        record = read_json(path)
        nested_final = (path.name == "final.json" and record.get("split") == "val"
                        and record.get("weight_view") == "ema" and isinstance(record.get("metrics"), dict))
        if not (record.get("validation_full_epoch") or record.get("validation_final")
                or record.get("validation_cycle", 0) or nested_final):
            continue
        metrics = record["metrics"] if nested_final else record
        step = int(record["optimizer_step"])
        score = validation_score(metrics, config)
        if step in records and not math.isclose(records[step]["score"], score, rel_tol=1e-9):
            raise ValueError(f"同一步验证分数冲突: {step}")
        records[step] = dict(step=step, score=score, file=str(path))
    return records


def select_checkpoint(run, load):
    """验证 best.pt 身份；缺 best 时只选有本步完整 val 记录的实际权重。"""
    run = resolve_path(run)
    if not run.is_dir():
        raise ValueError("请传带时间的训练结果目录，而不是权重文件")
    config = read_json(run / "config.json")
    records = validation_records(run, config)
    best = run / "best.pt"
    if best.is_file():
        state = load(best)
        step = int(state["step"])
        score = float(state["best_sampled_trajectory_score"])
        if not math.isfinite(score):
            raise ValueError("best.pt 分数非法")
        if records:
            winner = min(records.values(), key=lambda r: (r["score"], r["step"]))
            if step != winner["step"] or not math.isclose(score, winner["score"], rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"best.pt 与完整 val 最优记录不符: {run}")
        return best.resolve(), state, dict(step=step, score=score, policy="training_best", validation_verified=bool(records))
    candidates = []
    for path in sorted(run.glob("*.pt")):
        state = load(path)
        step = int(state["step"])
        if step in records:
            candidates.append((records[step]["score"], step, str(path)))
        del state
    if not candidates:
        raise ValueError(f"{run}: 无 best.pt，也无带本步完整 val 记录的权重；审计包不能代替训练目录")
    score, step, path = min(candidates)
    return Path(path).resolve(), load(Path(path)), dict(step=step, score=score, policy="best_available_full_val", validation_verified=True)


def event_groups(row):
    """复用评估的完整语义桶；过滤的特殊帧仍保留事件，未知不补普通。"""
    from qwen3vl_local.action_prior.metrics import event_sample_groups
    groups = [g.split("/", 1)[1] for g in event_sample_groups(row) if g.startswith("event_balance/")]
    return groups or ["UNCONFIRMED"]


def parse_category_counts(specifications, names):
    """解析 NAME=N 或 train/NAME=N；后给的同名配置覆盖前值，0表示跳过。"""
    counts = {}
    for spec in specifications:
        for item in spec.split(","):
            key, separator, value = item.strip().partition("=")
            parts = key.split("/")
            if (not separator or parts[-1] not in names or len(parts) > 2
                    or (len(parts) == 2 and parts[0] not in ("train", "test"))
                    or not value.isdecimal()):
                raise ValueError(f"非法采样数量 {item!r}；使用 NAME=非负整数 或 train/NAME=非负整数")
            counts[key] = int(value)
    return counts


def select_cases(rows, *, per_category=8, seed=2026, min_frame_gap=8, category_counts=None, split=None):
    """按物理路线轮转抽帧；只看标签、不看预测，稀缺类别不复制凑数。"""
    pools = {"event": {name: [] for name in EVENT_NAMES}, "action": {name: [] for name in ACTION_NAMES}}
    seen = set()
    for row in rows:
        key = identity(row)
        if key in seen:
            raise ValueError(f"重复帧: {key}")
        seen.add(key)
        for name in event_groups(row):
            pools["event"].setdefault(name, []).append(row)
        pools["action"][row["action_token"]["name"]].append(row)
    # 身份只用于前面的重复检查，释放全量set再进行分桶选帧。
    del seen
    selected, coverage, wanted_keys = {}, {}, set()
    for style, categories in pools.items():
        selected[style], coverage[style] = {}, {}
        for category, candidates in categories.items():
            overrides = (category_counts or {}).get(style, {})
            requested = overrides.get(f"{split}/{category}", overrides.get(category, per_category))
            groups = defaultdict(list)
            for row in candidates:
                groups[row["route_group"]].append(row)
            rng = random.Random(f"{seed}:{style}:{category}")
            order = sorted(groups)
            rng.shuffle(order)
            for group in sorted(groups):
                values = groups[group]
                values.sort(key=identity)
                rng.shuffle(values)
            chosen, taken = [], defaultdict(list)
            while len(chosen) < requested:
                progress = False
                for group in order:
                    while groups[group]:
                        row = groups[group].pop()
                        # 同一物理路线不同 Rep 也算一组，但帧间隔只在同一录制内约束。
                        if any(row["run_id"] == r["run_id"] and abs(int(row["anchor"]) - int(r["anchor"])) < min_frame_gap for r in taken[group]):
                            continue
                        chosen.append(row)
                        taken[group].append(row)
                        progress = True
                        break
                    if len(chosen) == requested:
                        break
                if not progress:
                    break
            selected[style][category] = [case_id(row) for row in chosen]
            wanted_keys.update(identity(row) for row in chosen)
            coverage[style][category] = dict(available_frames=len(candidates), available_physical_routes=len(groups),
                requested=requested, selected=len(chosen), selected_physical_routes=len(taken),
                shortfall=requested - len(chosen))
    # 只对实际选中的少量case计算文件夹ID，不对百万帧重复JSON编码/SHA256。
    return [r for r in rows if identity(r) in wanted_keys], selected, coverage


def require_same_frames(reference, actual, split):
    """实际有效 split 必须一致，拒绝把一个模型的 train 当另一模型的 test。"""
    left, right = {identity(r) for r in reference}, {identity(r) for r in actual}
    if len(left) != len(reference) or len(right) != len(actual) or left != right:
        raise ValueError(f"{split} 有效帧集合不同/重复，无法公平配对: only_first={len(left-right)}, only_current={len(right-left)}")
