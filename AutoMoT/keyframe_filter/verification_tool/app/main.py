from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


def resolve_paths() -> tuple[Path, Path]:
    project_dataset_root = Path(__file__).resolve().parents[2]

    env_root_raw = os.environ.get("DATASET_ROOT")
    dataset_root = Path(env_root_raw).expanduser().resolve() if env_root_raw else project_dataset_root

    env_json_raw = os.environ.get("KEYFRAME_JSON_PATH")
    keyframe_json = (
        Path(env_json_raw).expanduser().resolve()
        if env_json_raw
        else (dataset_root / "keyframes_4scenarios.json").resolve()
    )

    # If env points to a stale path, prefer the local dataset JSON next to this project.
    local_json = (project_dataset_root / "keyframes_4scenarios.json").resolve()
    if not keyframe_json.exists() and local_json.exists():
        keyframe_json = local_json

    # If root is invalid or mismatched, align to JSON parent when it looks like dataset root.
    if keyframe_json.exists():
        json_parent = keyframe_json.parent
        if (
            not dataset_root.exists()
            or not (dataset_root / "Accident").exists()
            or not (dataset_root / "CrossingBicycleFlow").exists()
        ) and (json_parent / "Accident").exists():
            dataset_root = json_parent

        # Honor dataset_root recorded in JSON if present and valid.
        try:
            with keyframe_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            recorded_root = data.get("dataset_root")
            if recorded_root:
                recorded_path = Path(recorded_root).expanduser().resolve()
                if recorded_path.exists() and (recorded_path / "Accident").exists():
                    dataset_root = recorded_path
        except Exception:
            pass

    return dataset_root, keyframe_json


DATASET_ROOT, KEYFRAME_JSON_PATH = resolve_paths()


@dataclass
class EventRecord:
    scenario: str
    run_id: str
    route_id: str
    status: str
    num_infractions: int
    signal_source: str
    rule_confidence: float
    final_success: bool
    diagnostics: Dict[str, Any]
    event: str
    frame: int
    t: float
    confidence: float
    label_text: Optional[str]
    image_path: Path
    image_exists: bool
    raw_run: Dict[str, Any]
    raw_event: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        rel = self.image_path.relative_to(DATASET_ROOT)
        return {
            "scenario": self.scenario,
            "run_id": self.run_id,
            "route_id": self.route_id,
            "status": self.status,
            "num_infractions": self.num_infractions,
            "signal_source": self.signal_source,
            "rule_confidence": self.rule_confidence,
            "final_success": self.final_success,
            "diagnostics": self.diagnostics,
            "event": self.event,
            "frame": self.frame,
            "t": self.t,
            "confidence": self.confidence,
            "label_text": self.label_text,
            "image_exists": self.image_exists,
            "image_url": "/dataset/" + rel.as_posix(),
            "header_text": f"{self.scenario} | {self.event}",
            "raw_run": self.raw_run,
            "raw_event": self.raw_event,
        }


class KeyframeStore:
    def __init__(self, dataset_root: Path, keyframe_json: Path):
        self.dataset_root = dataset_root
        self.keyframe_json = keyframe_json
        self.raw: Dict[str, Any] = {}
        self.runs: List[Dict[str, Any]] = []
        self.events: List[EventRecord] = []
        self._load()

    def _load(self) -> None:
        with self.keyframe_json.open("r", encoding="utf-8") as f:
            self.raw = json.load(f)

        self.runs = list(self.raw.get("runs", []))
        events: List[EventRecord] = []
        for run in self.runs:
            scenario = run.get("scenario", "")
            run_id = run.get("run_id", "")
            route_id = run.get("route_id", "")
            status = run.get("status", "")
            num_infractions = int(run.get("num_infractions", 0))
            signal_source = run.get("signal_source", "")
            rule_confidence = float(run.get("rule_confidence", 0.0))
            diagnostics = dict(run.get("diagnostics", {}))
            final = dict(run.get("final", {}))
            final_success = bool(final.get("final_success", False))

            def add_event(evt: Dict[str, Any]) -> None:
                frame = int(evt.get("frame", 0))
                image_path = self.dataset_root / scenario / run_id / "rgb" / f"{frame:04d}.jpg"
                events.append(
                    EventRecord(
                        scenario=scenario,
                        run_id=run_id,
                        route_id=route_id,
                        status=status,
                        num_infractions=num_infractions,
                        signal_source=signal_source,
                        rule_confidence=rule_confidence,
                        final_success=final_success,
                        diagnostics=diagnostics,
                        event=str(evt.get("event", "")),
                        frame=frame,
                        t=float(evt.get("t", 0.0)),
                        confidence=float(evt.get("confidence", 0.0)),
                        label_text=evt.get("label_text"),
                        image_path=image_path,
                        image_exists=image_path.exists(),
                        raw_run=run,
                        raw_event=evt,
                    )
                )

            add_event(dict(run.get("initial", {})))
            for m in run.get("middle", []):
                add_event(dict(m))
            add_event(final)

        self.events = events

    def scenario_values(self) -> List[str]:
        return sorted({e.scenario for e in self.events})

    def event_values(self) -> List[str]:
        return sorted({e.event for e in self.events})

    def source_values(self) -> List[str]:
        return sorted({e.signal_source for e in self.events})

    def filter_events(
        self,
        scenario: Optional[str],
        event: Optional[str],
        signal_source: Optional[str],
        final_success: Optional[bool],
        run_id_query: Optional[str],
        sort_by: str,
    ) -> List[EventRecord]:
        rows = self.events
        if scenario:
            rows = [r for r in rows if r.scenario == scenario]
        if event:
            rows = [r for r in rows if r.event == event]
        if signal_source:
            rows = [r for r in rows if r.signal_source == signal_source]
        if final_success is not None:
            rows = [r for r in rows if r.final_success == final_success]
        if run_id_query:
            q = run_id_query.lower()
            rows = [r for r in rows if q in r.run_id.lower()]

        if sort_by == "time_asc":
            rows = sorted(rows, key=lambda r: (r.scenario, r.run_id, r.t))
        elif sort_by == "time_desc":
            rows = sorted(rows, key=lambda r: (r.scenario, r.run_id, -r.t))
        elif sort_by == "confidence_desc":
            rows = sorted(rows, key=lambda r: (-r.confidence, r.scenario, r.run_id, r.frame))
        elif sort_by == "confidence_asc":
            rows = sorted(rows, key=lambda r: (r.confidence, r.scenario, r.run_id, r.frame))
        else:
            rows = sorted(rows, key=lambda r: (r.scenario, r.run_id, r.frame))

        return rows


app = FastAPI(title="Keyframe Verification Viewer")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
store = KeyframeStore(DATASET_ROOT, KEYFRAME_JSON_PATH)

app.mount("/dataset", StaticFiles(directory=str(DATASET_ROOT), follow_symlink=True), name="dataset")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "dataset_root": str(DATASET_ROOT),
            "keyframe_json_path": str(KEYFRAME_JSON_PATH),
            "scenario_options": store.scenario_values(),
            "event_options": store.event_values(),
            "signal_source_options": store.source_values(),
            "total_events": len(store.events),
            "total_runs": len(store.runs),
        },
    )


@app.get("/api/events")
def api_events(
    scenario: Optional[str] = Query(default=None),
    event: Optional[str] = Query(default=None),
    signal_source: Optional[str] = Query(default=None),
    final_success: Optional[bool] = Query(default=None),
    run_id_query: Optional[str] = Query(default=None),
    sort_by: str = Query(default="scenario_run"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=60, ge=1, le=240),
) -> JSONResponse:
    rows = store.filter_events(
        scenario=scenario,
        event=event,
        signal_source=signal_source,
        final_success=final_success,
        run_id_query=run_id_query,
        sort_by=sort_by,
    )

    total_count = len(rows)
    total_pages = max(1, math.ceil(total_count / page_size))
    page = min(page, total_pages)
    start = (page - 1) * page_size
    end = start + page_size
    page_rows = rows[start:end]

    return JSONResponse(
        {
            "items": [r.to_dict() for r in page_rows],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages,
            },
            "filters": {
                "scenario": scenario,
                "event": event,
                "signal_source": signal_source,
                "final_success": final_success,
                "run_id_query": run_id_query,
                "sort_by": sort_by,
            },
        }
    )


@app.get("/api/run/{scenario}/{run_id}")
def api_run(scenario: str, run_id: str) -> JSONResponse:
    match = None
    for run in store.runs:
        if run.get("scenario") == scenario and run.get("run_id") == run_id:
            match = run
            break
    if match is None:
        return JSONResponse({"error": "run not found"}, status_code=404)
    return JSONResponse(match)
