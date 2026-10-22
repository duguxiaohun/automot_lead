import copy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time
import pytest
from qwen3vl_local.audit_joint import geometry_study_cpu_audit_v2 as audit
from qwen3vl_local.audit_joint import geometry_study_cpu_continue_v2 as continuation
from qwen3vl_local.audit_joint.geometry_study.tests.test_study import record


@pytest.mark.parametrize('point,expected',[
    (dict(frame=8,imputed=True,imputation={'selected_endpoint_frame':9}),9),
    (dict(frame=8,imputed=True,imputation={'selected_endpoint_frame':9},direction_origin_frame=9),9),
    (dict(frame=8,direction_origin_frame=8),8),(dict(frame=8),8)])
def test_evidence_origin_survives_map_failure(point,expected):
    assert audit.evidence_origin(point)==expected


def test_missing_or_conflicting_donor_rejected():
    with pytest.raises(ValueError,match='missing_direction_evidence_donor'):
        audit.evidence_origin(dict(frame=8,imputed=True))
    with pytest.raises(ValueError,match='donor_mismatch'):
        audit.evidence_origin(dict(frame=8,imputed=True,imputation={'selected_endpoint_frame':9},direction_origin_frame=8))


def test_map_failure_keeps_donor_frame_and_evidence(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import trajectory_action as rule
    from qwen3vl_local.audit_joint import label_quarantine
    r=record(event='STATIC_BLOCKAGE');r['split']='train'
    parent=dict(records=[r],data_root=str(tmp_path));trajectory=SimpleNamespace(metas={9:{}})
    monkeypatch.setattr(rule,'load_route_trajectory',lambda path:trajectory)
    monkeypatch.setattr(label_quarantine,'load',lambda *a,**k:[])
    monkeypatch.setattr(audit.sources,'LaneMap',lambda p:SimpleNamespace(hashes={}))
    monkeypatch.setattr(audit.native,'verify',lambda p:None)
    def failed(*a,**k):raise ValueError('map lookup failed')
    monkeypatch.setattr(audit.sources,'observed',failed)
    def build(r,trajectory,quarantine,lanes,*,native_cache,reference_cache):
        native_cache[8,r['event']]=dict(evidence={'wrong_frame':8},required_end_frame=8)
        native_cache[9,r['event']]=dict(evidence={'confirmed_crossing':9},required_end_frame=10)
        point=dict(frame=8,action='LANE_CHANGE_LEFT',reason='native',imputed=True,
                   imputation={'selected_endpoint_frame':9})
        converted=audit.six.transform(point,trajectory,0.,'Town00',lanes,reference_cache)
        assert converted['action'] is None and 'direction_origin_frame' not in converted
        return dict(anchor_action=None,audit=dict(history_points=[converted],future_points=[]))
    monkeypatch.setattr(audit.six,'build',build)
    result=audit.direction_scan(parent,tmp_path,tmp_path,max_cases=1)
    c=result['cases'][0]
    assert c['native_evidence_origin']==9 and c['native_evidence']=={'confirmed_crossing':9}
    assert c['local_reference'] is None and c['kind']=='other_invalid'
    assert c['point']['action'] is None and not result['independent_maneuver_coverage_certified']


def candidate(group,origin,anchor=None,event='STATIC_BLOCKAGE',town='Town01'):
    anchor=origin if anchor is None else anchor
    return dict(kind='flipped',town=town,event=event,scenario='Synthetic',route_id=group,
        physical_group='Synthetic/'+group,native_evidence_origin=origin,anchor=anchor,
        id=f'{group}/{origin}/{anchor}',role='history_points',case_key=[event,anchor,origin,'history_points','flipped'])


def test_case_bank_diversifies_routes_and_origins_deterministically():
    candidates=[candidate('many_nearby',9,anchor=i) for i in range(100)]
    candidates += [candidate('many_nearby',i) for i in range(10,110)]
    candidates += [candidate(f'route_{i}',20,event=f'E{i}',town=f'Town{i:02d}') for i in range(2,5)]
    bank=audit.CaseBank(4);reverse=audit.CaseBank(4)
    for c in candidates:bank.add(c)
    for c in reversed(candidates):reverse.add(c)
    selected=bank.select()
    assert selected==reverse.select() and len(selected)==4
    assert len({c['physical_group'] for c in selected})==4
    assert len({(c['route_id'],c['native_evidence_origin']) for c in selected})==4
    assert all(len(group)<=4 for b in bank.buckets.values() for group in b.values())


def test_case_bank_does_not_merge_different_routes_in_same_physical_group():
    bank=audit.CaseBank(3)
    a=candidate('route_1',9);b=candidate('route_2',9);b['physical_group']=a['physical_group']
    bank.add(a);bank.add(b)
    assert len(bank.select())==2


def test_v2_preserves_pool_report_contract():
    from qwen3vl_local.audit_joint import geometry_study_cpu_audit as original
    from qwen3vl_local.audit_joint.geometry_study.tests.test_admission import cohort
    a=cohort()
    assert audit.pool_report(a,per_event=1,world_size=1)==original.pool_report(a,per_event=1,world_size=1)


def proc_file(root,pid=10,start='123',state='S'):
    path=root/str(pid)/'stat';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(f'{pid} (name with (parentheses)) '+ ' '.join([state]+['0']*18+[start])+'\n')
    return path


def test_process_disappearance_during_read_is_normal(tmp_path,monkeypatch):
    path=proc_file(tmp_path);original=Path.read_text
    def racing(p,*args,**kwargs):
        if p==path:path.unlink()
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',racing)
    assert continuation.process_stat(10,tmp_path) is None


@pytest.mark.parametrize('current',[None,dict(state='Z',start_time='123'),dict(state='S',start_time='456')])
def test_already_gone_zombie_and_pid_reuse_do_not_wait(current):
    continuation.wait_for_exit(10,'123',observe=lambda pid:current,pause=lambda _:pytest.fail('must not wait'))


def test_proc_permission_errors_are_not_disappearance(tmp_path,monkeypatch):
    def denied(*a,**k):raise PermissionError('denied')
    monkeypatch.setattr(Path,'read_text',denied)
    with pytest.raises(PermissionError):continuation.process_stat(10,tmp_path)


def test_real_short_lived_process_exit():
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.1)'])
    try:
        state=continuation.process_stat(child.pid)
        assert state is not None
        continuation.wait_for_exit(child.pid,state['start_time'],pause=lambda _:time.sleep(.01))
        assert child.wait(timeout=2)==0
    finally:
        if child.poll() is None:child.kill();child.wait()


def job_file(tmp_path):
    source=tmp_path/'bound.py';source.write_text('bound')
    data=tmp_path/'dataset.json';data.write_text('{}')
    ready=tmp_path/'native_ready.json';ready.write_text(json.dumps(dict(dataset=str(data))))
    job=dict(schema='study_cpu_continuation_v2',native_pid=10,native_start_time='123',
        sources={str(source):continuation.file_sha(source)},native_ready=str(ready),app=str(tmp_path),
        maps=str(tmp_path/'maps'),study_root=str(tmp_path/'cache'),output=str(tmp_path/'report'),
        log=str(tmp_path/'child.log'),status=str(tmp_path/'status.json'),
        module='qwen3vl_local.audit_joint.geometry_study_cpu_audit_v2')
    p=tmp_path/'job.json';p.write_text(json.dumps(job));return p,job


def test_exit_race_with_successful_receipt_still_launches_once(tmp_path,monkeypatch):
    path=proc_file(tmp_path);original=Path.read_text
    def racing(p,*a,**k):
        if p==path:path.unlink()
        return original(p,*a,**k)
    jobpath,job=job_file(tmp_path);calls=[]
    monkeypatch.setattr(Path,'read_text',racing)
    def launch(command,**kwargs):
        calls.append(command)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']==''
        out=Path(job['output']);out.mkdir()
        (out/'status.json').write_text(json.dumps(dict(status='completed',formal_conditions_passed=False,ready_status='rejected')))
        return SimpleNamespace(returncode=2)
    assert continuation.run(jobpath,observe=lambda pid:continuation.process_stat(pid,tmp_path),launch=launch)==2
    assert len(calls)==1 and json.loads(Path(job['status']).read_text())['status']=='rejected'
    saved=Path(job['status']).read_text()
    with pytest.raises(FileExistsError):continuation.run(jobpath,observe=lambda pid:None,launch=launch)
    assert len(calls)==1 and Path(job['status']).read_text()==saved


def test_missing_receipt_and_source_mutation_block_launch(tmp_path):
    jobpath,job=job_file(tmp_path)
    Path(job['native_ready']).unlink()
    with pytest.raises(ValueError,match='without_ready'):continuation.run(jobpath,observe=lambda pid:None)
    source=Path(next(iter(job['sources'])));source.write_text('changed')
    with pytest.raises(ValueError,match='source_changed'):continuation.run(jobpath,observe=lambda pid:None)
