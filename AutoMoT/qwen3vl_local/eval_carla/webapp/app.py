"""Flask 浏览器查看 LeadMoT 闭环评测结果。

用法：
    python AutoMoT/qwen3vl_local/eval_carla/webapp/app.py \
        --eval-base outputs/closed_loop_eval --port 5050

页面结构：
- 顶部 tab：Routes / Scenarios
- Routes：左栏按 scenario 分组列 route_id，右栏切 input/debug/demo/grid 视频
  与 leaderboard 指标
- Scenarios：表格汇总每个 scenario 的平均 score_composed / route / penalty
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from urllib.parse import unquote

from flask import Flask, abort, jsonify, render_template, request, send_from_directory


_THIS_FILE = pathlib.Path(__file__).resolve()
_EVAL_CARLA = _THIS_FILE.parent.parent       # .../eval_carla/
sys.path.insert(0, str(_EVAL_CARLA.parent.parent.parent))   # repo root → 可 import AutoMoT.*

# 复用项目的 scenario 反向映射
try:
    from AutoMoT.qwen3vl_local.eval_carla.scenario_picker import (
        build_route_to_scenario,
    )
except ImportError:
    # 直接执行 app.py 时也允许 fallback
    sys.path.insert(0, str(_EVAL_CARLA))
    from scenario_picker import build_route_to_scenario  # type: ignore


app = Flask(
    __name__,
    static_folder=str(_THIS_FILE.parent / "static"),
    template_folder=str(_THIS_FILE.parent / "templates"),
)

# 由 main() 注入
EVAL_BASE: pathlib.Path = pathlib.Path()
ROUTE_TO_SCENARIO: dict[int, list[str]] = {}

_SIG_RE = re.compile(r".+__.+__bev[01](?:__ema[01])?$")
_METRIC_KEYS = ("score_composed", "score_route", "score_penalty")


def _load_json(path: pathlib.Path) -> dict | None:
    """读取 route/summary JSON；web 页面只展示已有结果，读失败返回 None。"""
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None


def _route_id_from_record(record: dict) -> int | None:
    """兼容 leaderboard 里 route_id 的整数/字符串两种写法。"""
    value = record.get("route_id")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        if m:
            return int(m.group(0))
    return None


def _route_record(data: dict, route_id: int) -> dict | None:
    """从 leaderboard JSON 里找当前 route 的 record。"""
    records = data.get("_checkpoint", {}).get("records", [])
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict) and _route_id_from_record(record) == route_id:
                return record
        if len(records) == 1 and isinstance(records[0], dict):
            return records[0]
    return None


def _extract_route_scores(data: dict, route_id: int) -> dict[str, float]:
    """抽取 webapp 左栏和详情页展示的核心分数。"""
    record = _route_record(data, route_id)
    scores = record.get("scores", {}) if isinstance(record, dict) else {}
    if not isinstance(scores, dict):
        scores = {}
    out = {k: float(scores[k]) for k in _METRIC_KEYS if isinstance(scores.get(k), (int, float))}
    if out:
        return out

    global_record = data.get("_checkpoint", {}).get("global_record", {})
    scores_mean = global_record.get("scores_mean", {}) if isinstance(global_record, dict) else {}
    if isinstance(scores_mean, dict):
        return {k: float(scores_mean[k]) for k in _METRIC_KEYS if isinstance(scores_mean.get(k), (int, float))}
    return {}


def _extract_route_infractions(data: dict, route_id: int) -> dict | list:
    """抽取违规信息；新版读 route record，旧版退回 global_record。"""
    record = _route_record(data, route_id)
    infractions = record.get("infractions", {}) if isinstance(record, dict) else {}
    if infractions:
        return infractions
    global_record = data.get("_checkpoint", {}).get("global_record", {})
    return global_record.get("infractions", {}) if isinstance(global_record, dict) else {}


# ---------------------------------------------------------------------------
# 目录扫描
# ---------------------------------------------------------------------------
def list_signatures() -> list[str]:
    """列出所有模型结果目录。

    只接受 ckpt signature 格式，避免把 worker_logs、临时目录或其它文件夹显示到 UI。
    """
    if not EVAL_BASE.is_dir():
        return []
    return [p.name for p in sorted(EVAL_BASE.iterdir())
            if p.is_dir() and _SIG_RE.fullmatch(p.name)]


def list_routes_for_signature(sig: str) -> list[dict]:
    """扫描某个 signature 下的 route 目录。

    返回 [{route_id, scenarios, videos:{input,debug,demo,grid}, metrics:{...}}]。
    这个列表服务 Routes tab 左栏，既能显示视频是否存在，也能显示 score_composed。
    """
    sig_dir = EVAL_BASE / sig
    per_route_dir = sig_dir / "eval_per_route"
    items: list[dict] = []
    if not sig_dir.is_dir():
        return items
    for rdir in sorted(sig_dir.iterdir()):
        if not rdir.is_dir() or not rdir.name.startswith("route"):
            continue
        try:
            rid = int(rdir.name[len("route"):])
        except ValueError:
            continue
        videos = {
            # 四路视频可能按 launcher 开关缺失；前端据此禁用对应 tab。
            "input": (rdir / "input.mp4").is_file(),
            "debug": (rdir / "debug.mp4").is_file(),
            "demo":  (rdir / "demo.mp4").is_file(),
            "grid":  (rdir / "grid.mp4").is_file(),
        }
        metrics = {}
        eval_json = per_route_dir / f"eval_{rid}.json"
        data = _load_json(eval_json)
        if data is not None:
            metrics = _extract_route_scores(data, rid)
        items.append({
            "route_id": rid,
            "scenarios": ROUTE_TO_SCENARIO.get(rid, []),
            "videos": videos,
            "metrics": metrics,
        })
    return items


def load_scenarios_summary(sig: str) -> dict:
    """读取 aggregate.py 生成的 summary_all.json。

    没跑过聚合时返回空结构，让 Scenarios tab 仍能正常渲染。
    """
    p = EVAL_BASE / sig / "summary_all.json"
    if not p.is_file():
        return {"signature": sig, "scenarios": {}, "total_routes": 0}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"signature": sig, "scenarios": {}, "total_routes": 0}


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """主页面：所有数据都由下面的 JSON API 异步加载。"""
    return render_template("index.html")


@app.route("/api/signatures")
def api_signatures():
    """返回可选择的模型 signature 列表。"""
    return jsonify({"signatures": list_signatures()})


@app.route("/api/routes")
def api_routes():
    """按 scenario 分组返回某个 signature 下的 route 摘要。"""
    sig = request.args.get("sig", "")
    if not sig:
        return jsonify({"error": "missing sig"}), 400
    items = list_routes_for_signature(sig)
    # 按 scenario 分组（一个 route 可能不归任何 scenario，归 __unknown__）
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        # 一个 route 可能属于多个 LEAD scenario，前端会在多个组里看到同一 route。
        keys = it["scenarios"] or ["__unknown__"]
        for k in keys:
            groups[k].append(it)
    return jsonify({
        "signature": sig,
        "groups": dict(sorted(groups.items())),
        "total_routes": len(items),
    })


@app.route("/api/route")
def api_route():
    """返回单条 route 的视频可用性、config、leaderboard 分数和 infractions。"""
    sig = request.args.get("sig", "")
    rid = request.args.get("route_id", "")
    if not sig or not rid:
        return jsonify({"error": "missing sig or route_id"}), 400
    try:
        rid_int = int(rid)
    except ValueError:
        return jsonify({"error": "bad route_id"}), 400
    rdir = EVAL_BASE / sig / f"route{rid_int}"
    if not rdir.is_dir():
        return jsonify({"error": "route dir not found"}), 404
    info: dict = {"route_id": rid_int, "videos": {}, "metrics": {}, "infractions": []}
    for v in ("input", "debug", "demo", "grid"):
        info["videos"][v] = (rdir / f"{v}.mp4").is_file()
    # config.json
    cfg = rdir.parent / "config.json"
    if cfg.is_file():
        try:
            with open(cfg, encoding="utf-8") as f:
                info["config"] = json.load(f)
        except Exception:
            info["config"] = None
    # leaderboard json
    per_route = rdir.parent / "eval_per_route" / f"eval_{rid_int}.json"
    data = _load_json(per_route)
    if data is not None:
        info["metrics"] = _extract_route_scores(data, rid_int)
        info["infractions"] = _extract_route_infractions(data, rid_int)
    return jsonify(info)


@app.route("/api/scenarios")
def api_scenarios():
    """返回 scenario 聚合结果，供 Scenarios tab 表格使用。"""
    sig = request.args.get("sig", "")
    if not sig:
        return jsonify({"error": "missing sig"}), 400
    return jsonify(load_scenarios_summary(sig))


@app.route("/video/<path:rel>")
def serve_video(rel: str):
    """安全地从 EVAL_BASE 下取 mp4 文件供 <video> 播放。"""
    rel = unquote(rel)
    full = (EVAL_BASE / rel).resolve()
    try:
        # 防路径穿越：请求的真实路径必须仍在 eval_base 内。
        full.relative_to(EVAL_BASE.resolve())
    except ValueError:
        abort(403)
    if not full.is_file():
        abort(404)
    return send_from_directory(str(full.parent), full.name, mimetype="video/mp4",
                               conditional=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    """启动 Flask webapp，并预先加载 route_id -> scenario 映射。"""
    global EVAL_BASE, ROUTE_TO_SCENARIO

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-base", type=str, required=True)
    ap.add_argument("--benchmark-root", type=str, default=None)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5050)
    args = ap.parse_args()

    EVAL_BASE = pathlib.Path(args.eval_base).resolve()
    if not EVAL_BASE.is_dir():
        print(f"--eval-base does not exist: {EVAL_BASE}", file=sys.stderr)
        sys.exit(1)

    bench = pathlib.Path(args.benchmark_root).resolve() if args.benchmark_root else None
    ROUTE_TO_SCENARIO = build_route_to_scenario(bench)
    print(f"[webapp] eval_base = {EVAL_BASE}")
    print(f"[webapp] scenarios loaded: {len(ROUTE_TO_SCENARIO)} route mappings")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
