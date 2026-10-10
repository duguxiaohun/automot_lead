"""Real native Phase3 production and candidate-cache admission, when RGB is present."""
import json,sys
from collections import Counter
from pathlib import Path
import pytest
from qwen3vl_local.audit_joint import label_quarantine as quarantine
from qwen3vl_local.sft_new_loop_phase3 import build_dataset as builder


def test_native_and_cache_paths_exclude_only_evidence_affected_candidate(tmp_path,monkeypatch):
    app=Path(builder.__file__).resolve().parents[2]
    source=app/'checkpoints/ten_event_label_audit_20261010/selected_collection/OppositeVehicleRunningRedLight_result.json'
    root=app/'lead_data'
    if not source.exists() or not root.exists():pytest.skip('local audit RGB/collection unavailable')
    route='Town07_Rep0_Town07_Scenario8_19_route0_01_10_20_11_38'
    selected=[r for r in json.loads(source.read_text())['routes'] if r['route_id']==route]
    assert len(selected)==1
    collection=tmp_path/'collection';collection.mkdir()
    (collection/source.name).write_text(json.dumps(dict(routes=selected)))
    monkeypatch.setattr(sys,'argv',['test']);args=builder.parse_args()
    args.data_root=str(root);args.collection_dir=str(collection);args.workers=0
    args.progress_every_routes=0
    with monkeypatch.context() as patch:
        patch.setattr(quarantine,'load',lambda *a,**k:[])
        before=list(builder.iter_base_frames(args))
    counts=Counter();after=list(builder.iter_base_frames(args,counts))
    ids=lambda rows:{(r['scenario'],r['route_id'],r['frame_id'],r['context_id']) for r in rows}
    assert {key[2] for key in ids(before)-ids(after)}=={43}
    assert counts['source_excluded/confirmed_rgb_discontinuity']==1
    cache=tmp_path/'candidate.jsonl';cache.write_text(''.join(json.dumps(r)+'\n' for r in before))
    args.candidate_cache=str(cache)
    cached=list(builder.iter_base_frames(args))
    assert ids(cached)==ids(after)
