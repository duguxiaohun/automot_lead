"""Three-method scheduling, common initialization, shared data and fair scoring."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile
import pytest
import torch

from qwen3vl_local.audit_joint import phase3_suite as suite, phase3_suite_data as data, phase3_suite_compare as compare
from qwen3vl_local.audit_joint import phase3_base_trial as trial, phase3_base_trial_worker as base
from qwen3vl_local.audit_joint.phase3_base_dense import runtime,worker
from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_second_review import raw_fixture
from qwen3vl_local.sft_new_loop_phase3 import build_dataset as builder
from qwen3vl_local.qwen35.tests.test_backend import model


def test_same_qwen_jobs_all_three_epochs_no_adapter(tmp_path,monkeypatch,model):
    j,original,extra=raw_fixture(tmp_path,monkeypatch)
    prepared=tmp_path/'shared';(prepared/'base').mkdir(parents=True);(prepared/'auxiliary').mkdir()
    for name in ('frame_index.jsonl','manifest.json'):
        (prepared/'base'/name).write_bytes((Path(j['index']) if name=='frame_index.jsonl' else Path(j['index']).with_name(name)).read_bytes())
        (prepared/'auxiliary'/name).write_bytes((Path(j['aux_index']) if name=='frame_index.jsonl' else Path(j['aux_index']).with_name(name)).read_bytes())
    trial.write(prepared/'shared_data_receipt.json',{})
    args=SimpleNamespace(data_root=Path(j['data_root']),model_dir=Path(j['model_dir']),
        forecast_span_seconds=6.,forecast_step_seconds='auto')
    jobs=suite.jobs(args,prepared,tmp_path/'out',{'fixture':True})
    assert list(jobs)==list(suite.ARMS)
    for arm,job in jobs.items():
        cfg=base.make_args(job) if arm=='base' else runtime.make_args(job)
        assert cfg.num_epochs==3 and cfg.stop_after_epoch is None and cfg.batch_size==cfg.grad_accum==1
        assert cfg.warmup_steps==2000 and cfg.seed==20260904
        assert cfg.base_trial['suite_binding']==job['suite_binding']
        if arm!='base':
            assert job['initialization']=='qwen' and job['init_adapter'] is None and job['init_adapter_files']=={}
            bundle=SimpleNamespace(model=model)
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            assert runtime.initialize(bundle,job) is bundle
            assert all(torch.equal(before[n],p) for n,p in model.named_parameters())
            monkeypatch.setattr(base,'validate_job',lambda j:None)
            worker.validate(job)
    assert jobs['a_future']['aux_binding']==jobs['b_windows_bank']['aux_binding']


def test_ambiguous_qwen_initialization_rejected(tmp_path,monkeypatch):
    j,_,_=raw_fixture(tmp_path,monkeypatch)
    cfg=SimpleNamespace(index=Path(j['index']),data_root=Path(j['data_root']),model_dir=Path(j['model_dir']),
        resume=None,stage='complete',experiment='future',dense_initialization='qwen',init_adapter=Path(j['init_adapter']))
    with pytest.raises(ValueError,match='cannot also'):trial.build_job(cfg,tmp_path/'out')
    with pytest.raises(ValueError,match='contains adapter'):
        runtime.initialize(None,dict(initialization='qwen',init_adapter='bad'))


def states():return {a:dict(status='not_started') for a in suite.ARMS}


def test_all_admissions_before_training_and_no_extra_epochs(tmp_path,monkeypatch):
    jobs={a:dict(mode=m,output=str(tmp_path/a),stage='complete') for a,m in suite.ARMS.items()}
    calls=[];checked=[]
    monkeypatch.setattr(trial,'verify_training',lambda j:checked.append(j['mode']) or {'optimizer_updates':7680})
    def run(a,o,c):calls.append((a,o));assert '--nproc_per_node=4' in c or o in ('preflight','summarize');return 0
    state=states();assert suite.execute(jobs,run,state)==0
    assert calls[:5]==[(a,'preflight') for a in suite.ARMS]+[('a_future','probe'),('b_windows_bank','probe')]
    assert calls[5:]==[(a,o) for a in suite.ARMS for o in ('train','summarize','test')]
    assert checked==list(suite.ARMS.values())
    assert all(s['training']['optimizer_updates']==7680 for s in state.values())


@pytest.mark.parametrize('failure',[('b_windows_bank','preflight'),('b_windows_bank','probe'),('a_future','train')])
def test_first_failure_stops_without_rebuilding_or_retraining(tmp_path,monkeypatch,failure):
    jobs={a:dict(mode=m,output=str(tmp_path/a)) for a,m in suite.ARMS.items()}
    monkeypatch.setattr(trial,'verify_training',lambda j:{})
    calls=[]
    def run(a,o,c):calls.append((a,o));return 75 if (a,o)==failure else 0
    state=states();assert suite.execute(jobs,run,state)==75
    assert calls[-1]==failure and len(calls)==len(set(calls))
    assert state[failure[0]]['status']=='failed_or_interrupted'
    if failure[1]!='train':assert not any(o=='train' for a,o in calls)


def test_prepare_only_never_probes_or_trains(tmp_path):
    jobs={a:dict(mode=m,output=str(tmp_path/a)) for a,m in suite.ARMS.items()};calls=[]
    assert suite.execute(jobs,lambda a,o,c:calls.append((a,o)) or 0,states(),prepare_only=True)==0
    assert calls==[(a,'preflight') for a in suite.ARMS]


@pytest.mark.parametrize('variable',['SKIP_TRAIN','PIPELINE_CHECK_ONLY'])
def test_shell_honors_existing_check_only_switch(tmp_path,variable):
    script=tmp_path/'python';script.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n');script.chmod(0o755)
    env=dict(os.environ,PYTHON=str(script),**{variable:'1'})
    result=subprocess.run(['bash',str(Path(suite.__file__).with_suffix('.sh')),'--index','fixture'],env=env,text=True,capture_output=True,check=True)
    assert '--prepare-only' in result.stdout.splitlines()


def candidate(row):
    return dict(scenario=row['scenario'],route_id=row['route_id'],split=row['split'],context_id=row['context_id'],
        rs=row['prompt_road_structure'],frame_id=row['frame_id'],town=row['town'],
        action_labels=row['answers'],action_evidence=row['action_evidence'],
        primary_event=row['event'],event_codes=row.get('event_codes',[]),mapping_evidence=row.get('mapping_evidence',{}),
        context_detail=row.get('context_detail',''),goal_x=row['goal_ego_xy'][0],goal_y=row['goal_ego_xy'][1],
        visual_label_risk=False,visual_label_risk_reasons=[],history_rgb_paths=row['history_rgb_paths'],latest_rgb_path=row['latest_rgb_path'])


def test_generate_once_frozen_base_native_new_frames_and_reuse(tmp_path,monkeypatch):
    j,original,extra=raw_fixture(tmp_path,monkeypatch)
    collection=tmp_path/'collection';collection.mkdir()
    for scenario in {r['scenario'] for r in original}:(collection/f'{scenario}_result.json').write_text('{}')
    calls=[]
    def candidates(args,risk):calls.append(1);return iter([candidate(r) for r in extra])
    monkeypatch.setattr(builder,'iter_base_frames',candidates)
    output=tmp_path/'shared'
    cfg=dict(index=j['index'],data_root=j['data_root'],collection_dir=str(collection),output=str(output),
        offset_frames=[0,12,24],release_sha256='fixture')
    # Whitelist deployments may be plain directories without Git metadata.
    monkeypatch.setattr(data.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=128,stdout=''))
    data.build(cfg)
    assert calls==[1]
    assert (output/'base/frame_index.jsonl').read_bytes()==Path(j['index']).read_bytes()
    receipt=data.verify(output,data_root=j['data_root'],offsets=[0,12,24],release='fixture')
    assert receipt['added_rows']==len(extra) and not receipt['RGB_copied']
    assert receipt['git_commit'] is None and receipt['release_sha256']=='fixture'
    all_rows=[json.loads(x) for x in (output/'auxiliary/frame_index.jsonl').read_text().splitlines()]
    assert all_rows[:len(original)]==original
    assert {data.row_key(r) for r in all_rows[len(original):]}=={data.row_key(r) for r in extra}
    # No second build on reuse; bytes are bound independently of output location.
    data.verify(output,data_root=j['data_root'],offsets=[0,12,24],release='fixture')
    assert calls==[1]
    # Exercise the public coordinator with an explicit prepared dataset. No GPU
    # inventory or data builder may be called in prepare-only mode.
    (Path(j['model_dir'])/'model.safetensors').write_bytes(b'fixture')
    monkeypatch.setattr(sys,'argv',['phase3_suite','--index',j['index'],'--data-root',j['data_root'],
        '--model-dir',j['model_dir'],'--prepared-data',str(output),'--prepare-only','--output',str(tmp_path/'suite')])
    monkeypatch.setattr(trial,'gpu_ids',lambda *a:pytest.fail('prepare-only selected GPUs'))
    operations=[]
    def run(cmd,log,env,**kw):
        operations.append(cmd[-1]);Path(log).write_text('CPU fixture admission')
        assert env['CUDA_VISIBLE_DEVICES']==''
        return dict(exit_code=0,seconds=0,log=str(log))
    monkeypatch.setattr(trial,'run_stage',run)
    assert suite.main()==0
    assert operations==['preflight']*3 and calls==[1]
    assert trial.read(tmp_path/'suite/suite_workflow.json')['status']=='prepared_cpu_no_training'
    with pytest.raises(FileExistsError):data.build(cfg)
    (output/'auxiliary/frame_index.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='changed'):data.verify(output,data_root=j['data_root'],offsets=[0,12,24],release='fixture')


def test_new_target_must_match_exact_context_road_and_split(tmp_path,monkeypatch):
    j,original,extra=raw_fixture(tmp_path,monkeypatch)
    c=candidate(extra[0]);variants=[]
    for field,val in [('split','different'),('rs','UNKNOWN'),('context_id','absent'),('frame_id',9999)]:
        variants.append(dict(c,**{field:val}))
    assert list(data.added_rows(original,variants,[0,12,24]))==[]
    assert len(list(data.added_rows(original,[c,c],[0,12,24])))==1


def test_frozen_route_filter_restores_native_functions(monkeypatch):
    original=lambda *a:iter([dict(route_id='a'),dict(route_id='b')])
    monkeypatch.setattr(builder,'_source_routes',original);split=builder._split
    with pytest.raises(RuntimeError):
        with data.frozen_routes(builder,{('s','a'):'val'}):
            assert list(builder._source_routes(None,'s',None))==[dict(route_id='a')]
            assert builder._split('s','a',1,.1,.1)=='val'
            raise RuntimeError()
    assert builder._source_routes is original and builder._split is split


def test_comparison_does_not_hide_missing_windows_or_reward_keep_majority():
    targets={'1':dict(action='KEEP',event='e'),'2':dict(action='STOP',event='e')}
    p={'base':{'1':'KEEP','2':'KEEP'},'a_future':{'1':'KEEP','2':'STOP'},'b_windows_bank':{'1':'KEEP'}}
    out=compare.summarize(targets,p)
    assert out['common_supported_total']==1
    assert out['arms']['b_windows_bank']['coverage']==.5
    assert out['arms']['b_windows_bank']['fixed_anchors_missing_counted_incorrect']['accuracy']==.5
    assert out['arms']['base']['fixed_anchors_missing_counted_incorrect']['by_action']['STOP']['accuracy']==0
    assert out['arms']['a_future']['fixed_anchors_missing_counted_incorrect']['accuracy']==1
    with pytest.raises(ValueError,match='outside'):compare.summarize(targets,dict(p,bad={'3':'KEEP'}))


def test_initial_weight_mutation_stops_following_arm(tmp_path):
    model=tmp_path/'model';model.mkdir();(model/'weights').write_bytes(b'abcd')
    snapshot=suite.model_snapshot(model);suite.check_model_snapshot(model,snapshot)
    (model/'weights').write_bytes(b'efgh')
    with pytest.raises(ValueError,match='changed'):suite.check_model_snapshot(model,snapshot)


def test_three_failure_packages_no_weights(tmp_path):
    for arm in suite.ARMS:
        p=tmp_path/arm/'train/final';p.mkdir(parents=True);(p/'adapter.safetensors').write_bytes(b'weights')
        trial.write(tmp_path/arm/'job.json',{'arm':arm})
    suite.feedback(tmp_path,states(),dict(status='failed_or_interrupted'))
    for arm in suite.ARMS:
        archive=tmp_path/'handoff'/f'{arm}_report.zip'
        receipt=trial.read(tmp_path/'handoff'/f'{arm}_package.json')
        assert trial.file_sha(archive)==receipt['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and not any('safetensors' in n for n in z.namelist())


def test_comparison_reads_native_rank_files_and_dense_query_four(tmp_path,monkeypatch):
    from dataclasses import replace
    from qwen3vl_local.audit_joint.tests.test_phase3_base_trial import fixed_pool
    from qwen3vl_local.audit_joint.phase3_base_dense import evaluation,baseline
    from qwen3vl_local.sft_new_loop_phase3 import train
    from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity
    from qwen3vl_local.sft_new_loop_phase3.prompts import prompt_spec_to_json
    source,_,_=fixed_pool(2)
    pools={s:[replace(r,split=s,route_id=r.route_id+s) for r in source] for s in ('val','test')}
    monkeypatch.setattr(train,'_read_rows',lambda p,s,**kw:pools[s])
    work={s:[train._make_item(r,seed=20260904,action_output_mode='choice') for r in rows] for s,rows in pools.items()}
    monkeypatch.setattr(evaluation,'selected',lambda pool,s:work[s])
    jobs={a:dict(mode=m,output=str(tmp_path/a),index='fixture',data_root='fixture') for a,m in suite.ARMS.items()}
    for split,items in work.items():
        records=[];dense=[]
        for i,x in enumerate(items):
            records.append(dict(step=7680,scenario=x.row.scenario,route_id=x.row.route_id,frame_id=x.row.frame_id,
                context_id=x.row.context_id,prompt_spec=prompt_spec_to_json(x.spec),raw_output='KEEP',all_ok=True))
            dense.append(dict(id=baseline.digest(case_identity(x)),targets=['KEEP']*5,
                predictions=[dict(action='KEEP')]*4+[dict(action='STOP' if i==0 else 'KEEP')]))
        base_dir=tmp_path/'base'/('train' if split=='val' else 'test');base_dir.mkdir(parents=True,exist_ok=True)
        if split=='val':(base_dir/'final_generation_val_cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        else:
            for rank in range(4):(base_dir/f'cases_rank{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records[rank::4]))
        for a in ('a_future','b_windows_bank'):trial.write(tmp_path/a/f'dense_{split}_cases.json',dense)
    result=compare.compare(jobs)
    for split,report in result['results'].items():
        assert report['arms']['base']['fixed_anchors_missing_counted_incorrect']['accuracy']==1
        assert report['arms']['a_future']['fixed_anchors_missing_counted_incorrect']['accuracy']==1
        assert report['arms']['b_windows_bank']['fixed_anchors_missing_counted_incorrect']['correct']==len(source)-1
    broken=trial.read(tmp_path/'a_future/dense_test_cases.json');broken[0]['targets'][0]='STOP'
    trial.write(tmp_path/'a_future/dense_test_cases.json',broken)
    with pytest.raises(ValueError,match='label changed'):compare.compare(jobs)


def test_exception_during_training_is_reported_on_affected_arm(tmp_path):
    jobs={a:dict(mode=m,output=str(tmp_path/a)) for a,m in suite.ARMS.items()}
    state=states()
    def run(a,o,c):
        if o=='train':raise RuntimeError('model changed')
        return 0
    with pytest.raises(RuntimeError,match='model changed'):suite.execute(jobs,run,state)
    assert state['base']['status']=='failed_or_interrupted' and state['a_future']['status']=='gpu_admitted'
