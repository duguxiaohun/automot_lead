import copy
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from .test_pipeline import fixture
from .test_core import record
from qwen3vl_local.audit_joint.abc_delivery import write,sha
from qwen3vl_local.action_expert_ablation.geometry_abc import __main__ as cli,group_scope,contracts,worker,audit


@pytest.mark.parametrize('branch',['bev_only','qwen_simple','both'])
def test_only_selected_groups_probe_train_evaluate_and_retain_scope(tmp_path,monkeypatch,branch):
    args,root,calls=fixture(tmp_path,monkeypatch)
    cli.main(args+['--branch',branch]);expected=group_scope.groups(branch)
    assert calls[:len(expected)]==[('probe',g) for g in expected]
    assert calls[len(expected):]==[item for g in expected for item in [('train',g),('evaluate','test')]]
    gate=json.loads((root/'admission.json').read_text());assert gate['selected_groups']==list(expected)
    calls.clear();cli.main(['probe','--run-dir',str(root),'--data-dir',str(tmp_path/'data')])
    assert calls==[] # completed probes of the same admitted scope retained
    before=(root/'admission.json').read_bytes()
    other='qwen_simple' if branch!='qwen_simple' else 'bev_only'
    with pytest.raises(ValueError,match='existing_suite_group_scope_changed'):
        cli.main(args+['--branch',other])
    assert (root/'admission.json').read_bytes()==before


@pytest.mark.parametrize('selection',[[],['a_bev_only'],['a_bev_only']*3,['a_bev_only','b_bev_only','c_qwen_simple'],'bev_only'])
def test_incomplete_or_mixed_scope_rejected(selection):
    with pytest.raises(ValueError,match='invalid_admitted'):group_scope.admitted({'selected_groups':selection})


def test_out_of_scope_worker_refuses_before_runner_or_gpu(tmp_path,monkeypatch):
    from qwen3vl_local.action_expert_ablation import common
    monkeypatch.setattr(common,'leadmot_train',lambda:pytest.fail('must reject before runner'))
    write(tmp_path/'admission.json',dict(code=contracts.source_identity(),selected_groups=list(group_scope.groups('bev_only'))))
    with pytest.raises(ValueError,match='outside_admitted_scope'):
        worker.main(['--abc-variant','a','--branch','qwen_simple','--suite-root',str(tmp_path)])
    assert group_scope.admitted({})==audit.GROUPS


@pytest.mark.parametrize('branch',['bev_only','qwen_simple','both'])
def test_formal_package_contains_exact_selected_groups(tmp_path,monkeypatch,branch):
    # Actual audit loop, metrics recomputation and ZIP writer; expensive model,
    # raw source and probe/checkpoint validation use explicit fixture substitutes.
    from qwen3vl_local.action_expert_ablation.geometry_abc import data,consumption,probe_inputs,diagnostic_coverage,diagnostics
    from qwen3vl_local.audit_joint.geometry_v1.dataset import digest
    rows={s:[record(i,'LEAD_BRAKE',s)] for i,s in enumerate(('train','val','test'))}
    for values in rows.values():
        for r in values:
            r['id']=digest([r['scenario'],r['route_id'],r['frame'],r['event']])
            r['waypoint_xy']=[[0.,0.]]*8;r['observation']['reference']['yaw']=0.
    root=tmp_path/'run';directory=tmp_path/'data';manifest=dict(identity='fixture',splits={},data_root='/unused')
    for name in ('manifest.json','split_report.json','status.json'):write(directory/name,manifest)
    write(directory/'plan.json',{})
    for name in ('processor.json','history_support.json','raw_source_verification.json'):write(root/name,{})
    write(root/'admission.json',dict(manifest_sha256=sha(directory/'manifest.json'),processor_sha256='fixture',selected_groups=list(group_scope.groups(branch))))
    monkeypatch.setattr(data,'load',lambda *a:(rows,manifest))
    monkeypatch.setattr(probe_inputs,'verify_processor',lambda *a:{})
    monkeypatch.setattr(probe_inputs,'selection',lambda *a:{})
    monkeypatch.setattr(probe_inputs,'verify_probe_inputs',lambda *a:None)
    monkeypatch.setattr(consumption,'Sources',lambda *a:SimpleNamespace(proof=lambda r:{}))
    complete=dict(best_sha256='weight',binding={'splits':{}},step=12719)
    seen=[]
    def verified(run,manifest,projection,variant,branch,**kw):
        seen.append(f'{variant}_{branch}');return complete
    monkeypatch.setattr(audit,'verify_training',verified)
    metrics=dict(route_ade_m=0.,route_fde_m=0.,waypoint_ade_m=0.,waypoint_fde_m=0.)
    for group in group_scope.groups(branch):
        v,b=group.split('_',1);run=root/group;r=rows['test'][0]
        for name in ('training_complete.json','condition_contract.json','config.json'):write(run/name,{})
        write(run/'test/metrics.json',dict(checkpoint_sha256='weight',binding=complete['binding'],metrics=dict(metrics,samples=1)))
        write(run/'test/cases/rank0_case0.json',dict(id=r['id'],event=r['event'],physical_group=r['physical_group'],variant=v,branch=b,
            condition_source='oracle',consumed_sources={},region_ids=[x['token_id'] for x in r['representations'][v]['targets']],
            gt_waypoints=[r['waypoint_xy']],pred_waypoints=[r['waypoint_xy']],gt_route=[[[0.,0.]]*10],pred_route=[[[0.,0.]]*10],metrics=metrics,
            geometry_projection=dict(rotation_world_xy=[[1.,0.,0.],[0.,1.,0.]],gt_endpoint_xyz=[[0.,0.,0.]]*4),
            inference=dict(forward_seconds=.1,first_eval_forward=True)))
        selected,coverage=diagnostic_coverage.select(rows['val'],v)
        cases=[dict(id=r['id'],event=r['event'],physical_group=r['physical_group'],
            labels=diagnostic_coverage.labels(r,v,'future'),history_labels=diagnostic_coverage.labels(r,v,'history'),
            slot_responses={'future':[1.]*4,'history':[1.]*4},
            results={key:dict(mean_waypoint_shift_from_oracle_m=1.) for key in ('replace_regions','permute_horizons')}) for r in selected]
        write(run/'test/condition_sensitivity.json',dict(checkpoint_sha256='weight',binding=complete['binding'],coverage=coverage,
            cases=cases,summary=diagnostics.summarize(cases,coverage),same_noise=True))
        for name in ('probe_complete.json','rank0.json','rank1.json','rank2.json','rank3.json'):write(root/'probes'/group/name,{})
    result=audit.build(root,directory)
    assert seen==list(group_scope.groups(branch))
    with zipfile.ZipFile(result['package']) as z:
        manifest=json.loads(z.read('manifest.json'))
        assert manifest['summary']['groups']==list(group_scope.groups(branch))
        assert set(json.loads(z.read('summary.json')))==set(group_scope.groups(branch))
        if branch!='both':assert not any('_'+('qwen_simple' if branch=='bev_only' else 'bev_only')+'/' in n for n in z.namelist())
    assert Path(result['package']).stat().st_size<30_000_000
