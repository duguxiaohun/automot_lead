import json
from pathlib import Path
import sys

import pytest

from qwen3vl_local.audit_joint import geometry_prepare_profile as profiling,data_prepare,versioned_data as cache
from qwen3vl_local.audit_joint.geometry_v1 import dataset
from qwen3vl_local.audit_joint.tests.test_data_reuse_boundaries import geometry_inputs,snapshot


def test_profile_preserves_native_dataset_identity_and_reuse(geometry_inputs):
    raw,coll,maps,checkpoints,_=geometry_inputs
    original=dataset.annotation_routes;original_bind=cache.bind
    with profiling.Profile().instrument():
        first=data_prepare.geometry(raw,coll,maps,checkpoints)
    before=snapshot(Path(first['dataset']).parent)
    profile=profiling.Profile()
    with profile.instrument():
        second=data_prepare.geometry(raw,coll,maps,checkpoints)
    assert second['dataset']==first['dataset'] and second['action']=='reuse_existing'
    assert before==snapshot(Path(first['dataset']).parent)
    assert dataset.annotation_routes is original and cache.bind is original_bind
    plain=data_prepare.geometry(raw,coll,maps,checkpoints)
    assert plain['dataset']==first['dataset']
    report=profile.report('completed')
    assert report['selections'][0]['action']=='reuse_existing'
    assert report['stages']['candidates.enumerate']['calls']>=2
    assert report['adapter_hashing']['calls']>0 and report['adapter_hashing']['bytes']>0
    assert 'native.build' not in report['stages']
    assert all(0<=s['inclusive_seconds']<=report['total_seconds'] for s in report['stages'].values())


def test_observation_does_not_hide_dependency_failure_and_restores_functions(tmp_path):
    p=profiling.Profile();before=(cache.bind,cache.ensure,dataset.load,dataset.annotation_routes)
    def contract():return dict(input='fixed')
    def fail(out):raise ValueError('original native error')
    with pytest.raises(ValueError,match='original native error'):
        with p.instrument():cache.ensure(tmp_path,'fixture','v1',contract,fail,lambda out:None)
    assert before==(cache.bind,cache.ensure,dataset.load,dataset.annotation_routes)
    report=p.report('failed')
    assert report['stages']['fixture.build']['failures']==1
    assert report['cache_actions']==[]


def test_cli_success_failure_appends_reports_and_preserves_prior_receipts(tmp_path,monkeypatch):
    path=tmp_path/'timing.jsonl';calls=[]
    def main():calls.append(list(sys.argv))
    monkeypatch.setattr(data_prepare,'main',main)
    arguments=['profile','--timing-report',str(path),'--receipt',str(tmp_path/'ready.json')]
    monkeypatch.setattr(sys,'argv',arguments)
    profiling.main()
    before=path.read_bytes()
    assert calls[0][1:]==['--method','geometry','--receipt',str(tmp_path/'ready.json')]
    def failure():raise RuntimeError('native build failed')
    monkeypatch.setattr(data_prepare,'main',failure)
    with pytest.raises(RuntimeError,match='native build failed'):profiling.main()
    assert path.read_bytes().startswith(before) and sys.argv is arguments
    reports=[json.loads(s) for s in path.read_text().splitlines()]
    assert [r['status'] for r in reports]==['completed','failed']
    assert reports[1]['error']=='RuntimeError: native build failed'
    assert reports[0]['run_id']!=reports[1]['run_id']


def test_unrelated_report_file_preserved(tmp_path):
    p=tmp_path/'foreign.jsonl';p.write_text('{"unrelated": true}\n');before=p.read_bytes()
    with pytest.raises(ValueError,match='not this profiler'):
        profiling.append_report(p,profiling.Profile().report('completed'))
    assert p.read_bytes()==before


def test_remote_cpu_entry_collects_timing_alongside_ready_receipt():
    text=Path(data_prepare.__file__).with_name('geometry_v1').joinpath('remote.sh').read_text()
    assert '-m qwen3vl_local.audit_joint.geometry_prepare_profile' in text
    assert '--timing-report "$GEOMETRY_ROOT/prepare_timings.jsonl"' in text
    assert '--receipt "$GEOMETRY_ROOT/data_ready.json"' in text
