"""被删除/隔离的源帧也必须有审计记录，不能只记保留的训练样本。"""
import json
from qwen3vl_local.sft_new_loop_phase3 import audit_annotation_repairs as module


def test_quarantined_frame_is_written_and_abnormal_route_is_not_scanned(tmp_path, monkeypatch):
    collection = tmp_path/"collection"; collection.mkdir()
    data = tmp_path/"data"
    for run in ("good", "abnormal"):
        (data/"HighwayCutIn"/run).mkdir(parents=True)
    ann = dict(frame_id=44, primary_road_structure="R4", primary_event="R-E4",
               evidence=dict(rules_fired=["r4_light_hazard"]))
    (collection/"HighwayCutIn_result.json").write_text(json.dumps(dict(routes=[
        dict(route_id=run, annotations=[ann]) for run in ("good", "abnormal")])))
    monkeypatch.setattr(module, "is_abnormal_lead_route", lambda path, scenario: (path.name=="abnormal", {}))
    output = tmp_path/"audit"
    report = module.audit(collection, data, output)
    rows = [json.loads(line) for line in (output/"annotation_repairs.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["rs_quarantined"]
    assert rows[0]["repair"]["source"]["rs"] == "R4"
    assert not rows[0]["manual_rgb_confirmed"]
    assert report["counts"]["excluded_routes"] == 1
