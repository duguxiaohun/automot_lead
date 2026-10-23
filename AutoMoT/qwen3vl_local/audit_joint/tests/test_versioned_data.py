import copy
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pytest
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_processor import processor

from qwen3vl_local.audit_joint import versioned_data as cache, phase3_data_ready as ready
from qwen3vl_local.audit_joint import phase3_prompt_runtime as prompt_runtime


def test_same_version_reuse_changed_inputs_new_version_and_corruption(tmp_path):
    source=tmp_path/'input';source.write_text('first');calls=[]
    contract=lambda:dict(input=cache.bind(source),split=('train','val'))
    def build(out):
        calls.append(str(out));out.mkdir();(out/'labels.json').write_text(source.read_text())
    validate=lambda out:None
    a=cache.ensure(tmp_path/'cache','labels','v1',contract,build,validate,protected=[source])
    b=cache.ensure(tmp_path/'cache','labels','v1',contract,build,validate)
    assert a['key']==b['key'] and b['action']=='reuse' and len(calls)==1
    source.write_text('second')
    c=cache.ensure(tmp_path/'cache','labels','v1',contract,build,validate)
    assert c['key']!=a['key'] and len(calls)==2
    assert (Path(a['directory'])/'labels.json').read_text()=='first'
    (Path(c['directory'])/'labels.json').write_text('broken')
    with pytest.raises(ValueError,match='output changed'):
        cache.ensure(tmp_path/'cache','labels','v1',contract,build,validate)
    assert len(calls)==2


def test_failed_and_concurrent_builds_never_reused(tmp_path):
    calls=[]
    def build(out):calls.append(out);out.mkdir();raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):cache.ensure(tmp_path,'labels','v1',lambda:dict(rule=1),build,lambda p:None)
    with pytest.raises(ValueError,match='Incomplete'):
        cache.ensure(tmp_path,'labels','v1',lambda:dict(rule=1),build,lambda p:None)
    assert len(calls)==1
    with cache.lock(tmp_path/'other.lock'):
        with pytest.raises(ValueError,match='Another dataset'):
            with cache.lock(tmp_path/'other.lock'):pytest.fail('lock acquired twice')


def test_mutating_inputs_validator_and_escaped_paths_rejected(tmp_path):
    x={'value':1}
    def build(out):out.mkdir();(out/'data').write_text('a');x['value']=2
    with pytest.raises(ValueError,match='dependencies changed'):
        cache.ensure(tmp_path/'cache','labels','v1',lambda:dict(x),build,lambda out:None)
    with pytest.raises(ValueError,match='Invalid cache'):
        cache.ensure(tmp_path,'../escape','v1',lambda:x,build,lambda out:None)
    with pytest.raises(ValueError,match='overlaps'):
        cache.ensure(tmp_path,'labels','v1',lambda:x,build,lambda out:None,protected=[tmp_path])
    def links(out):out.mkdir();(out/'escape').symlink_to(tmp_path)
    with pytest.raises(ValueError,match='must not contain links'):
        cache.ensure(tmp_path/'cache2','labels','v1',lambda:dict(rule=1),links,lambda out:None)


def test_discovery_does_not_pick_latest_or_follow_external_links(tmp_path):
    for name in ('old','new'):
        p=tmp_path/name;p.mkdir();(p/'dataset.json').write_text(name)
    outside=tmp_path/'outside';outside.mkdir();(outside/'dataset.json').write_text('external')
    (tmp_path/'old'/'link').symlink_to(outside,target_is_directory=True)
    assert len(cache.discover(tmp_path,'dataset.json'))==3
    with pytest.raises(ValueError,match='Multiple different'):
        cache.unique_compatible([tmp_path/'old/dataset.json',tmp_path/'new/dataset.json'],lambda p:p.read_text())
    assert cache.unique_compatible([tmp_path/'old/dataset.json',tmp_path/'new/dataset.json'],lambda p:'same')==tmp_path/'new/dataset.json'


def test_native_route_cache_matches_full_build_and_rebuilds_only_changed_route(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_second_review import raw_fixture
    from qwen3vl_local.audit_joint.phase3_single_frame import dataset
    from qwen3vl_local.sft_new_loop_phase3 import train
    j,_,_=raw_fixture(tmp_path,monkeypatch)
    index=Path(j['index']);data=Path(j['data_root']);out=tmp_path/'cache'
    rows=[r for s in ('train','val','test') for r in train._read_rows(index,s,data_root=data)]
    expected,_=dataset.build(rows,data,tmp_path/'reference',history_policy='offline_native',index=index)
    a=ready.ensure(index,data,out)
    actual=dataset.load(a['dataset'],index=index)
    assert actual['records']==expected['records'] and actual['raw_proof']==expected['raw_proof']
    assert a['routes_built']>1 and a['routes_reused']==0
    b=ready.ensure(index,data,out)
    assert b['routes_built']==0 and b['routes_reused']==a['routes_built']
    assert b['dataset']==a['dataset']
    frame=next(data.glob('*/*/metas/0020.pkl'));value=pickle.loads(frame.read_bytes());value['speed']=0
    frame.write_bytes(pickle.dumps(value))
    c=ready.ensure(index,data,out,candidates=[Path(a['dataset'])])
    assert c['routes_built']==1 and c['routes_reused']==a['routes_built']-1
    expected2,_=dataset.build(rows,data,tmp_path/'reference2',history_policy='offline_native',index=index)
    assert dataset.load(c['dataset'])['records']==expected2['records']
    assert Path(a['dataset']).exists()  # old source-bound version is preserved


def test_runtime_only_refresh_does_not_generate_labels(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_second_review import raw_fixture
    from qwen3vl_local.audit_joint.phase3_single_frame import dataset
    j,_,_=raw_fixture(tmp_path,monkeypatch)
    a=ready.ensure(j['index'],j['data_root'],tmp_path/'cache')
    old=dataset.load(a['dataset']);changed=copy.deepcopy(old['sources'])
    changed['experiment_sources']['objective.py']='new-runtime-test'
    monkeypatch.setattr(dataset,'sources',lambda:changed)
    monkeypatch.setattr(dataset,'build',lambda *a,**k:pytest.fail('labels recomputed'))
    b=ready.ensure(j['index'],j['data_root'],tmp_path/'cache',candidates=[Path(a['dataset'])])
    new=dataset.load(b['dataset'])
    assert b['labels_recomputed'] is False and new['records']==old['records']
    assert new['runtime_refresh']['changed_runtime_sources']==['objective.py']


def test_prompt_variant_contract_restore_and_old_checkpoint_rejection():
    from qwen3vl_local.audit_joint.phase3_single_frame import objective,protocol,dataset
    from qwen3vl_local.audit_joint.tests.test_phase3_prompt_review import record
    r=record();artifact=dict(config=dataset.config('offline_native',1.5,'nearest_short'),records={'a':r},sources={})
    before=protocol.annotate({},artifact);system=objective.INSTRUCTION
    with prompt_runtime.activate():
        current=protocol.annotate({},artifact)
        assert current['production_prompt_sha256']!=before['production_prompt_sha256']
        assert current['prompt_variant']=='native_clarified_v1'
        assert current['answer_order']==before['answer_order']
        assert current['output_query_count']==4 and current['history_rgb_count']==1
        protocol.validate(current,artifact)
        with pytest.raises(ValueError,match='prompt/query'):protocol.validate(before,artifact)
    assert objective.INSTRUCTION==system and protocol.annotate({},artifact)==before
    with pytest.raises(ValueError,match='prompt/query'):protocol.validate(current,artifact)


def test_prompt_wrapper_preserves_torchrun_and_common_bootstrap(tmp_path):
    from qwen3vl_local.audit_joint import phase3_launch as launch, phase3_common_runtime as common
    from torch.distributed.run import get_args_parser
    for _,cmd in launch.schedule('single_frame',tmp_path):
        cmd=prompt_runtime.wrap(common.wrap(cmd,tmp_path/'reference.json',tmp_path/'raw'))
        if 'torch.distributed.run' in cmd:
            parsed=get_args_parser().parse_args(cmd[3:])
            assert parsed.training_script=='qwen3vl_local.audit_joint.phase3_prompt_runtime'
        assert 'qwen3vl_local.audit_joint.phase3_common_runtime' in cmd


def test_generic_recipe_real_child_process_build_reuse_and_input_invalidation(tmp_path):
    from qwen3vl_local.audit_joint.data_prepare import recipe
    source=tmp_path/'builder.py'
    source.write_text('from pathlib import Path\nimport sys\np=Path(sys.argv[2])\nif sys.argv[1]=="build":\n p.mkdir();(p/"labels").write_text(Path(sys.argv[3]).read_text())\nelse:\n assert (p/"labels").read_text()==Path(sys.argv[3]).read_text()\n')
    raw=tmp_path/'input';raw.write_text('labels1')
    spec=dict(operation='generate_data',family='phase4_test_stage',version='v1',cwd=str(tmp_path),
              sources=['builder.py'],inputs=['input'],
              build_command=['{python}','builder.py','build','{output}','input'],
              validate_command=['{python}','builder.py','verify','{output}','input'])
    path=tmp_path/'recipe.json';path.write_text(json.dumps(spec))
    a=recipe(path,tmp_path/'checkpoints');b=recipe(path,tmp_path/'checkpoints')
    assert a['action']=='build' and b['action']=='reuse' and a['directory']==b['directory']
    raw.write_text('labels2');c=recipe(path,tmp_path/'checkpoints')
    assert c['action']=='build' and c['directory']!=a['directory']


def test_compatible_common_dataset_discovered_without_generation(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.tests.test_phase3_common_data import common_fixture
    from qwen3vl_local.audit_joint import phase3_common_data as common
    j,root,_=common_fixture(tmp_path,monkeypatch)
    index,ref=common.ensure(tmp_path/'default',j['data_root'],tmp_path/'missing_collection',
                           lambda *a:pytest.fail('rebuilt'),discover_existing=True)
    assert index==root/'native/frame_index.jsonl' and not (tmp_path/'default').exists()


@pytest.mark.parametrize('phase',[1,2])
def test_phase12_cached_preparation_and_native_validator(tmp_path,monkeypatch,phase):
    from qwen3vl_local.audit_joint import data_prepare
    from qwen3vl_local.sft_new_loop_phase1.test_training_policy import write_dataset
    import importlib
    module=importlib.import_module(f'qwen3vl_local.sft_new_loop_phase{phase}.build_dataset')
    data=tmp_path/'raw';data.mkdir();collection=tmp_path/'collection';collection.mkdir()
    inputs=collection/'Synthetic_result.json';inputs.write_text('{"routes":[]}')
    calls=[]
    def build(args):calls.append(args.output_dir);write_dataset(Path(args.output_dir),phase)
    monkeypatch.setattr(module,'build_dataset',build)
    a=data_prepare.phase12(phase,data,collection,tmp_path/'checkpoints')
    b=data_prepare.phase12(phase,data,collection,tmp_path/'checkpoints')
    assert a['index']==b['index'] and len(calls)==1 and b['action']=='reuse'
    inputs.write_text('{"routes":[],"revision":2}')
    c=data_prepare.phase12(phase,data,collection,tmp_path/'checkpoints')
    assert c['index']!=a['index'] and len(calls)==2
    assert Path(a['index']).exists()
    # Reusing an explicitly selected old index does not require regenerating
    # (or even retaining) the original collection exports.
    reused=data_prepare.phase12(phase,data,tmp_path/'absent_collection',tmp_path/'checkpoints',existing=Path(a['index']))
    assert reused['generation_performed'] is False and len(calls)==2


def test_changed_common_annotations_new_version_preserves_old(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.tests.test_phase3_common_data import common_fixture
    from qwen3vl_local.audit_joint import phase3_common_data as common,phase3_base_trial as trial
    j,root,_=common_fixture(tmp_path,monkeypatch)
    collection=tmp_path/'collection';collection.mkdir();src=collection/'Synthetic_result.json';src.write_text('old')
    trial.write(root/'input_bindings.json',dict(collection={str(src):trial.file_sha(src)},routes=30))
    receipt=trial.read(root/'common_receipt.json');receipt['input_bindings_sha256']=trial.file_sha(root/'input_bindings.json')
    trial.write(root/'common_receipt.json',receipt)
    before=(root/'native/frame_index.jsonl').read_bytes();src.write_text('new');calls=[]
    monkeypatch.setattr(common,'source_inputs',lambda *a:dict(collection={str(src):trial.file_sha(src)},routes=30))
    def run(name,cmd,log):
        calls.append(name);out=Path(cmd[cmd.index('--output-dir')+1]);out.mkdir()
        for name in ('frame_index.jsonl','manifest.json'):(out/name).write_bytes((root/'native'/name).read_bytes())
    a,_=common.ensure(root,j['data_root'],collection,run,discover_existing=True)
    b,_=common.ensure(root,j['data_root'],collection,run,discover_existing=True)
    assert a==b and a!=root/'native/frame_index.jsonl' and calls==['build_common']
    assert (root/'native/frame_index.jsonl').read_bytes()==before


def test_real_clarified_qwen_backward_saved_contract_roundtrip(tmp_path,model,processor):
    import torch
    from PIL import Image
    from types import SimpleNamespace
    from qwen3vl_local.audit_joint.phase3_single_frame import objective,protocol,dataset
    from qwen3vl_local.audit_joint.tests.test_phase3_prompt_review import record
    r=record();artifact=dict(config=dataset.config('offline_native',1.5,'nearest_short'),records={'a':r},sources={})
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    with prompt_runtime.activate():
        packed=objective.pack(bundle,r,Image.new('RGB',(32,32)))
        value,report=objective.loss(bundle,packed);value.backward()
        assert torch.isfinite(value) and report['supervised']==4
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters())
        config=protocol.annotate({},artifact);path=tmp_path/'adapter_config.json';path.write_text(json.dumps(config))
        protocol.validate(json.loads(path.read_text()),artifact)
    with pytest.raises(ValueError,match='prompt/query'):protocol.validate(config,artifact)


def test_real_synced_native_runtime_reuse_and_frozen_source_check(tmp_path):
    from qwen3vl_local.audit_joint import phase3_native_runtime as native
    import os
    first=native.prepare(tmp_path/'checkpoints')
    second=native.prepare(tmp_path/'checkpoints')
    assert first==second
    result=subprocess.run([sys.executable,'-B','-c',
        'from qwen3vl_local.audit_joint.phase3_base_trial import verified_sources; print(verified_sources())'],
        cwd=first,env=dict(os.environ,PYTHONPATH=str(first),PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    expected=json.loads((native.APP/'qwen3vl_local/audit_joint/phase3_behavior_diagnostic_v1/runtime_sources.json').read_text())
    assert result.stdout.strip()==expected['files']['qwen3vl_local/audit_joint/phase3_base_trial_sources.json']['sha256']


def test_compatibility_bootstrap_preserves_current_machine_paths(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint import phase3_native_runtime as native
    monkeypatch.chdir(tmp_path);runtime=tmp_path/'runtime';runtime.mkdir()
    monkeypatch.setattr(native,'prepare',lambda path:runtime)
    seen=[]
    def call(command,**kwargs):
        seen.append(command)
        assert kwargs['cwd']==runtime and kwargs['env']['AUTOMOT_PHASE3_VERIFIED_RUNTIME']=='1'
        assert command[command.index('--output')+1]==str(tmp_path/'my run')
        assert '--data-root='+str(tmp_path/'data') in command
        assert command[command.index('--model-dir')+1]==str(native.APP/'checkpoints/Qwen3.5-4B')
        return 0
    monkeypatch.setattr(native.subprocess,'call',call)
    assert native.execute('phase3_launch',['--method','single_frame','--output','my run','--data-root=data'])==0
    assert len(seen)==1
