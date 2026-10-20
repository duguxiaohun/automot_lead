"""Fixed data lifecycle and fresh-reference integration without a GPU."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from qwen3vl_local.audit_joint import phase3_common_data as common, phase3_common_runtime as adapter
from qwen3vl_local.audit_joint import phase3_launch as launch, phase3_launch_worker as worker
from qwen3vl_local.audit_joint import phase3_base_trial as trial, phase3_base_trial_worker as base
from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_second_review import raw_fixture


def common_fixture(tmp_path,monkeypatch):
    real_reference=trial.reference
    j,original,_=raw_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(trial,'reference',real_reference)
    root=tmp_path/'common'; native=root/'native';native.mkdir(parents=True)
    index=native/'frame_index.jsonl';rows=[]
    for row in original:
        for offset in range(38 if row['split']!='train' and not row['invalid_action_context'] else 1):
            rows.append(dict(row,frame_id=row['frame_id']+offset*100))
    index.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (native/'manifest.json').write_bytes(Path(j['index']).with_name('manifest.json').read_bytes())
    ref=common.create_reference(index,Path(j['data_root']))
    trial.write(root/'reference.json',ref)
    receipt=dict(schema='phase3_common_dataset_v1',data_root=j['data_root'],release_sha256='fixture',
        files={n:trial.file_sha(root/n) for n in ('native/frame_index.jsonl','native/manifest.json','reference.json')})
    trial.write(root/'common_receipt.json',receipt)
    return j,root,ref


def test_reference_all_native_consumers_and_summary(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import train, preflight
    from qwen3vl_local.sft_new_loop_phase3.prompts import prompt_spec_to_json,choice_target_action,parse_action_output
    from qwen3vl_local.audit_joint.phase3_base_dense import evaluation
    from qwen3vl_local.audit_joint.phase3_base_dense.data import Pool
    j,root,ref=common_fixture(tmp_path,monkeypatch)
    assert ref['historical_scores_comparable'] is False and 'baseline_validation' not in ref
    assert ref==common.create_reference(root/'native/frame_index.jsonl',Path(j['data_root']))
    original_reference=trial.REFERENCE
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    with adapter.activate(root/'reference.json',j['data_root']):
        job=trial.build_job(SimpleNamespace(index=root/'native/frame_index.jsonl',data_root=Path(j['data_root']),
            model_dir=Path(j['model_dir']),resume=None,stage='complete',experiment='base'),tmp_path/'experiment')
        base.validate_job(job);base.preflight(job)
        args=base.make_args(job)
        assert args.base_trial['common_reference']['reference_sha256']==trial.file_sha(root/'reference.json')
        assert args.base_trial['validation'].startswith('new shared')
        rows=[r for split in ('train','val','test') for r in train._read_rows(Path(job['index']),split,data_root=Path(j['data_root']))]
        pool=Pool(rows);work=evaluation.selected(pool,'val')
        assert len(work)==320 and len(evaluation.selected(pool,'val','loss'))==160
        assert len(evaluation.selected(pool,'test'))==380
        records=[]
        for item in work:
            r=item.row;target=choice_target_action(item.spec)
            records.append(dict(step=7680,prompt_spec=prompt_spec_to_json(item.spec),answers=r.answers,
                scenario=r.scenario,route_id=r.route_id,frame_id=r.frame_id,context_id=r.context_id,
                raw_output=target,parsed=parse_action_output(target,spec=item.spec),all_ok=True,format_valid=True))
        folder=Path(job['output'])/'train';folder.mkdir(exist_ok=True)
        (folder/'final_generation_val_cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        worker.summarize(job)
        result=trial.read(Path(job['output'])/'validation_summary.json')
        assert result['correct']==320 and result['historical_scores_comparable'] is False
        assert 'delta_to_historical_three_epoch_pp' not in result
    assert trial.REFERENCE==original_reference


def test_reuse_no_build_and_tamper_no_rebuild(tmp_path,monkeypatch):
    j,root,_=common_fixture(tmp_path,monkeypatch)
    no_build=lambda *a:pytest.fail('Existing common dataset rebuilt')
    index,ref=common.ensure(root,j['data_root'],tmp_path/'absent_collection',no_build)
    assert index==root/'native/frame_index.jsonl'
    index.write_text(index.read_text()+'\n')
    with pytest.raises(ValueError,match='Common dataset changed'):
        common.ensure(root,j['data_root'],tmp_path/'absent_collection',no_build)


def test_new_build_once_and_partial_rejected(tmp_path,monkeypatch):
    j,source,_=common_fixture(tmp_path,monkeypatch)
    root=tmp_path/'new_common';collection=tmp_path/'collection';collection.mkdir()
    monkeypatch.setattr(common,'source_inputs',lambda *a:dict(collection={},routes=30))
    calls=[]
    def run(name,cmd,log):
        calls.append(name); assert '--use-review-cache' not in cmd
        assert '--max-routes' not in cmd and '--require-invalid-true-rs-coverage' in cmd
        dest=Path(cmd[cmd.index('--output-dir')+1]);dest.mkdir()
        for name in ('frame_index.jsonl','manifest.json'):(dest/name).write_bytes((source/'native'/name).read_bytes())
    common.ensure(root,j['data_root'],collection,run)
    common.ensure(root,j['data_root'],collection,run)
    assert calls==['build_common']
    partial=tmp_path/'partial';partial.mkdir()
    with pytest.raises(ValueError,match='Partial common dataset'):
        common.ensure(partial,j['data_root'],collection,run)
    wrong=tmp_path/'otherdata';wrong.mkdir()
    with pytest.raises(ValueError,match='receipt/root mismatch'):
        common.ensure(root,wrong,collection,run)


def test_failed_build_preserved_and_no_automatic_retry(tmp_path,monkeypatch):
    root=tmp_path/'common';data=tmp_path/'raw';data.mkdir()
    monkeypatch.setattr(trial,'verified_sources',lambda:'fixture')
    monkeypatch.setattr(common,'source_inputs',lambda *a:dict(collection={},routes=1))
    def run(*a):raise RuntimeError('failed native build')
    with pytest.raises(RuntimeError,match='failed native build'):
        common.ensure(root,data,tmp_path/'collection',run)
    assert (root/'build_failure.json').is_file()
    with pytest.raises(ValueError,match='Partial common dataset'):
        common.ensure(root,data,tmp_path/'collection',run)


def test_bootstrap_real_torchrun_parser(tmp_path):
    from torch.distributed.run import get_args_parser
    for method in ('base','single_frame'):
        for _,cmd in launch.schedule(method,tmp_path):
            wrapped=adapter.wrap(cmd,tmp_path/'reference.json',tmp_path/'raw')
            if 'torch.distributed.run' in cmd:
                parsed=get_args_parser().parse_args(wrapped[3:])
                assert parsed.training_script=='qwen3vl_local.audit_joint.phase3_common_runtime'
                assert parsed.training_script_args[-2]=='--job'
    with pytest.raises(ValueError,match='Unsupported'):adapter.wrap(['python','bad'],Path('x'),Path('y'))


@pytest.mark.parametrize('method',['base','single_frame'])
def test_no_index_coordinator_uses_common_for_every_stage(tmp_path,monkeypatch,method):
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    j,root,_=common_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    monkeypatch.setattr(trial,'gpu_ids',lambda *a:'0,1,2,3')
    monkeypatch.setattr(trial,'verify_training',lambda job:dict(optimizer_updates=7680))
    monkeypatch.setattr(launch.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    seen=[]
    def ensure(*args,**kwargs):
        assert kwargs['single_frame']==(method=='single_frame')
        return root/'native/frame_index.jsonl',root/'reference.json'
    monkeypatch.setattr(common,'ensure',ensure)
    def run(cmd,log,env,**kwargs):
        seen.append(log.stem);log.write_text('fixture')
        assert 'qwen3vl_local.audit_joint.phase3_common_runtime' in cmd
        if method=='single_frame' and log.stem=='prepare':
            exp=log.parent/'experiment';exp.mkdir()
            trial.write(exp/'single_frame_job.json',dict(j,output=str(exp),mode='base'))
        return dict(exit_code=0,seconds=0,log=str(log))
    monkeypatch.setattr(trial,'run_stage',run)
    out=tmp_path/'runs'
    monkeypatch.setattr(sys,'argv',['launch','--method',method,'--common-data-dir',str(root),
        '--data-root',j['data_root'],'--model-dir',j['model_dir'],'--output',str(out)])
    assert launch.main()==0
    assert seen==(['prepare','probe','train','summarize','test'] if method=='base' else ['prepare','probe','train','val','test'])
    state=trial.read((out/'latest').resolve()/'launch_workflow.json')
    assert state['common_data']['historical_scores_comparable'] is False


def test_bootstrap_uses_adapted_native_module_instance(tmp_path,monkeypatch):
    j,root,_=common_fixture(tmp_path,monkeypatch)
    reached=[]
    def main():
        assert trial.REFERENCE==root/'reference.json'
        assert base.make_args(j).base_trial['common_reference']
        assert sys.argv==['phase3_base_trial_worker','--operation','train','--job','fixture']
        reached.append(True)
    monkeypatch.setattr(base,'main',main)
    monkeypatch.setattr(sys,'argv',['bootstrap','--reference',str(root/'reference.json'),
        '--data-root',j['data_root'],'--module','phase3_base_trial_worker','--',
        '--operation','train','--job','fixture'])
    adapter.main()
    assert reached==[True]


def test_missing_raw_route_rejected_without_scan(monkeypatch,tmp_path):
    from qwen3vl_local.sft_new_loop_phase3 import build_dataset
    coll=tmp_path/'collection';coll.mkdir();(coll/'Example_result.json').write_text('{}')
    data=tmp_path/'data';(data/'Example').mkdir(parents=True)
    monkeypatch.setattr(build_dataset,'_iter_routes_stream',lambda p:iter([dict(route_id='missing')]))
    with pytest.raises(ValueError,match='Raw route missing'):common.source_inputs(coll,data)


def test_common_native_adapter_roundtrip(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import train, eval as evaluate
    from qwen3vl_local.qwen35 import adapters
    j,root,_=common_fixture(tmp_path,monkeypatch)
    with adapter.activate(root/'reference.json',j['data_root']):
        monkeypatch.setattr(adapters,'save_adapter',lambda *a,**kw:None)
        bundle=SimpleNamespace(unwrap=lambda:None,lora_target_modules=['q_proj'])
        train._save_adapter(bundle,Path(j['output'])/'train',base.make_args(j),step=7680)
        monkeypatch.setattr(sys,'argv',['evaluate',*worker.test_arguments(j)])
        args=evaluate.parse_args()
        cfg=evaluate._validate_action_adapter(Path(args.adapter_dir),Path(j['model_dir']))
        assert cfg['global_step']==7680 and cfg['history_rgb_count']==4


@pytest.mark.parametrize('method',['base','single_frame'])
def test_resume_never_generates_missing_common_data(tmp_path,monkeypatch,method):
    monkeypatch.setattr(sys,'argv',['launch','--method',method,'--resume',str(tmp_path/'epoch_0001'),
        '--common-data-dir',str(tmp_path/'missing'),'--output',str(tmp_path/'run')])
    with pytest.raises(ValueError,match='Resume requires'):launch.main()
    assert not (tmp_path/'missing').exists() and not (tmp_path/'run').exists()
