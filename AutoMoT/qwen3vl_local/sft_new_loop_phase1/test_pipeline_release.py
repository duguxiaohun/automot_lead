"""Late-stage wiring, preflight failures and tiny real Qwen3.5 round trips."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from qwen3vl_local.sft_new_loop_phase1 import train as p1
from qwen3vl_local.sft_new_loop_phase2 import train as p2
from qwen3vl_local.sft_new_loop_phase1.test_training_policy import phase1_rows, phase2_rows, write_dataset
from qwen3vl_local.sft_new_loop_phase1.exposure_sampling import CoverageSampler, prepare_epoch_templates
from qwen3vl_local.sft_new_loop_phase1.training_policy import ActionHints
from qwen3vl_local.sft_new_loop_phase1.local_assets import check_launch, inspect_index
from qwen3vl_local.sft_new_loop_phase1.pipeline_preflight import check
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_processor import processor


def test_future_epoch_error_is_before_model_and_no_actual_history():
    rows = phase1_rows(32)
    first = p1._balanced_work(rows, target_per_bin=6, seed=1, variant_weights=p1.TRAIN_VARIANT_WEIGHTS)
    def broken(seed):
        raise ValueError('future seed lacks compatible quota')
    with pytest.raises(ValueError, match='future seed'):
        prepare_epoch_templates(first, seed=1, epoch_count=3, build=broken,
            sampler=CoverageSampler(rows, ActionHints()), strength=.5, max_repeat=10)


@pytest.mark.parametrize('phase', [1,2])
def test_preflight_checks_all_splits_images_and_boolean_labels(tmp_path, monkeypatch, phase):
    from qwen3vl_local.qwen35 import preflight, backend
    monkeypatch.setattr(preflight, 'check', lambda *a: {'ready':True,'errors':[]})
    monkeypatch.setattr(backend.LocalProcessor, 'from_pretrained', lambda *a: None)
    index = write_dataset(tmp_path/'data', phase)
    for i in range(4):
        Image.new('RGB', (32,32)).save(tmp_path/f'{i}.jpg')
    args = SimpleNamespace(phase=phase, model_dir=str(tmp_path), index=str(index), data_root=str(tmp_path),
        history_rgb_mode='2rgb_endpoints', eval_split='test', cases_per_bin=4, seed=20260810,
        highway_ue3_fraction=.125, route_diverse=False, exclude_cases='', expected_excluded=0,expected_total=0)
    report = check(args)
    assert report['image_headers_verified']==2
    # Broken unused middle frame is irrelevant to endpoints mode, but caught for 4rgb.
    (tmp_path/'1.jpg').write_text('corrupt')
    args.history_rgb_mode='4rgb'
    with pytest.raises(ValueError, match='BEFORE training'): check(args)
    args.history_rgb_mode='2rgb_endpoints'
    (tmp_path/'3.jpg').unlink()
    with pytest.raises(ValueError, match='BEFORE training'): check(args)
    records=[json.loads(x) for x in index.read_text().splitlines()]
    records[-1]['answers'][next(iter(records[-1]['answers']))]='False'
    index.write_text(''.join(json.dumps(x)+'\n' for x in records))
    with pytest.raises(ValueError, match='non-boolean'): inspect_index(index,phase)


def test_launch_collision_invalid_gpu_and_duplicate_modes(tmp_path):
    with pytest.raises(ValueError, match='GPU_IDS'): check_launch('2rgb_endpoints','0,00',[])
    with pytest.raises(ValueError, match='HISTORY_RGB_MODES'): check_launch('4rgb 4rgb','0',[])
    with pytest.raises(ValueError, match='share one'): check_launch('4rgb 2rgb_endpoints','0',[str(tmp_path/'run')]*2)
    with pytest.raises(ValueError, match='already exists'): check_launch('4rgb','0',[str(tmp_path)])


@pytest.mark.parametrize('phase', [1,2])
def test_real_qwen35_two_images_loss_save_reload_generate(tmp_path,monkeypatch,model,processor,phase):
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen3vl_local.sft_v2.train import _lora_target_modules
    from qwen3vl_local.qwen35.backend import LocalModel
    from qwen3vl_local.qwen35.adapters import LocalPeftModel
    module=p1 if phase==1 else p2
    for key,token in [('image_token_id','<|image_pad|>'),('video_token_id','<|video_pad|>'),
                      ('vision_start_token_id','<|vision_start|>'),('vision_end_token_id','<|vision_end|>')]:
        setattr(model.config,key,processor.tokenizer.convert_tokens_to_ids(token))
    model.save_pretrained(tmp_path/'base')
    base=LocalModel.from_pretrained(tmp_path/'base')
    targets=_lora_target_modules(base,vision_scope='off',strict_vision_scope=True)
    adapted=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=targets,task_type=TaskType.CAUSAL_LM))
    bundle=SimpleNamespace(model=adapted,processor=processor,tokenizer=processor.tokenizer,
                           device=torch.device('cpu'),unwrap=lambda:adapted,lora_target_modules=targets)
    row=phase1_rows(16)[0] if phase==1 else phase2_rows(8)[0]
    item=module._make_all_item(row,'RS1' if phase==1 else module._target_class(row),seed=7)
    images=[Image.new('RGB',(32,32))]*2
    if phase==1:
        target=module.build_phase1_target(row.answers,spec=item.spec)
        packed=module._build_inputs(bundle,images=images,
            # Byte-level fixture tokenizer cannot compress the long Chinese production prompt.
            prompt='Report visible facts from the two images.',
            target=target,output_keys=item.spec.output_keys,semantic_output_keys=None,
            semantic_output_weights=None,max_length=20000,format_loss_weight=.25)
        loss,_=module._loss_one(bundle,packed)
    else:
        packed=module._build_inputs(bundle,images=images,history_rgb_mode='2rgb_endpoints',
            target=module.build_event_target(item.spec),spec=item.spec,max_length=20000,format_loss_weight=.25)
        loss,_=module._loss_one(bundle,packed,item.spec)
    assert torch.isfinite(loss)
    loss.backward()
    opt=torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad],lr=.001)
    opt.step();opt.zero_grad()
    monkeypatch.setattr(sys,'argv',['train','--model-dir',str(tmp_path/'base'),'--history-rgb-mode','2rgb_endpoints'])
    config=module.parse_args()
    saved=module._save_adapter(bundle,tmp_path/'run',config,step=1)
    restored=LocalPeftModel.from_pretrained(LocalModel.from_pretrained(tmp_path/'base'),saved).eval()
    # Real save contract + two-image generation after reload, without judging accuracy.
    ids=packed['input_ids'].unsqueeze(0)
    vision={k:v for k,v in packed['vision'].items()}
    with torch.no_grad():
        output=restored.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
            **vision,max_new_tokens=2,do_sample=False)
    assert output.shape[1]>ids.shape[1]
    assert json.loads((saved/f'sft_new_loop_phase{phase}_adapter_config.json').read_text())['history_rgb_mode']=='2rgb_endpoints'


@pytest.mark.parametrize("phase", [1,2])
def test_independent_shell_eval_and_real_bundle(tmp_path, phase):
    """Run real launchers and packers; stand in only for expensive model jobs.

    The shim is confined to this temporary fixture. Separate tests above execute
    native Qwen loss/save/reload and the actual dataset/image preflight.
    """
    source=Path(p1.__file__).resolve().parents[1]
    root=tmp_path/'AutoMoT';package=root/'qwen3vl_local';package.mkdir(parents=True)
    for child in source.iterdir():
        if child.name in ('sft_new_loop_phase1','sft_new_loop_phase2'):
            shutil.copytree(child,package/child.name,ignore=shutil.ignore_patterns('__pycache__','probe_output','.pytest_cache'))
        else:
            (package/child.name).symlink_to(child,target_is_directory=child.is_dir())
    for selected_phase in (phase,):
        folder=root/f'checkpoints/sft_new_loop_phase{selected_phase}_data';folder.parent.mkdir(exist_ok=True)
        write_dataset(folder,selected_phase)
    for index in root.glob('checkpoints/sft_new_loop_phase*_data/frame_index.jsonl'):
        for line in index.read_text().splitlines():
            for name in json.loads(line)['history_rgb_paths']:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():Image.new('RGB',(32,32)).save(path)
    base=root/'checkpoints/fixture_base';base.mkdir()
    (base/'model.safetensors').write_bytes(b'fixture base, never loaded')
    (base/'config.json').write_text('{"model_type":"qwen3_5","fixture_only":true}')
    bindir=tmp_path/'bin';bindir.mkdir()
    shim=bindir/'python'
    shim.write_text('#!'+sys.executable+'\n'+r'''
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
def get(key,default=None):return args[args.index(key)+1] if key in args else default
target=args[0] if args else ''
if args[:2]==['-m','qwen3vl_local.sft_new_loop_phase1.pipeline_preflight']:
    from qwen3vl_local.sft_new_loop_phase1.local_assets import inspect_index
    Path(get('--output')).write_text(json.dumps({'fixture_only':True,'dataset':inspect_index(get('--index'),int(get('--phase')))}))
elif target.endswith('/train.py') and 'sft_new_loop_phase' in target:
    phase=1 if 'phase1/' in target else 2
    if os.environ.get('FAIL_FIXTURE_PHASE')==str(phase):sys.exit(17)
    from qwen3vl_local.sft_new_loop_phase1.test_integrity import train_fixture
    train_fixture(phase,args[1:])
elif target.endswith('/eval.py') and 'sft_new_loop_phase' in target:
    phase=1 if 'phase1/' in target else 2
    from qwen3vl_local.sft_new_loop_phase1.test_integrity import eval_fixture
    eval_fixture(phase,args[1:])
elif target.endswith('/audit_eval_cases.py'):
    dest=Path(get('--output-dir'));dest.mkdir(parents=True)
    (dest/'summary.json').write_text('{"fixture_only":true}')
elif target.endswith('/visual_audit.py'):
    dest=Path(get('--output'));dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text('{"fixture_only":true}')
else:
    os.execv(os.environ['REAL_PYTHON'],[os.environ['REAL_PYTHON']]+args)
''');shim.chmod(0o755)
    runner=bindir/'torchrun'
    runner.write_text('#!'+sys.executable+'\nimport os,sys\na=sys.argv[1:];a=a[a.index("--")+1:]\nos.execv(os.environ["FIXTURE_PYTHON"],[os.environ["FIXTURE_PYTHON"]]+a)\n');runner.chmod(0o755)
    env=os.environ.copy();env.update(PATH=str(bindir)+':'+env['PATH'],REAL_PYTHON=sys.executable,
        FIXTURE_PYTHON=str(shim),PYTHONPATH=str(root),GPU_IDS='0,1,2,3',HISTORY_RGB_MODES='2rgb_endpoints',RUN_BUILD='0',
        MODEL_DIR=str(base),DATA_ROOT=str(root),TRAIN_FOCUS_BALANCE_COUNT='6',TRAIN_MAX_STEPS='5',FINAL_CASES_PER_BIN='4')
    other=3-phase
    failed=root/f'checkpoints/sft_new_loop_phase{other}_pipeline/failed_existing/status.json'
    failed.parent.mkdir(parents=True);failed.write_text('{"status":"rejected"}')
    command=['bash',f'qwen3vl_local/sft_new_loop_phase{phase}/run_full_pipeline.sh']
    result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    archives=list(root.glob('checkpoints/sft_new_loop_phase*_pipeline/**/*.tar.gz'))
    assert len(archives)==1
    for archive in archives:
        assert archive.stat().st_size<30*1024*1024
        with tarfile.open(archive) as tf:
            names=tf.getnames()
            assert any(x.endswith('/bundle_manifest.json') for x in names)
            assert any(x.endswith('/exposure_summary.json') for x in names)
            assert not any(x.endswith(('.safetensors','.bin','.pt')) for x in names)
            assert any(x.endswith('/integrity_files.json') for x in names)
            assert any(x.endswith('/dataset_receipt.json') for x in names)
            assert any(x.endswith('/pipeline_preflight.json') for x in names)
    assert (root/f'checkpoints/sft_new_loop_phase{phase}_runs/latest/final').is_dir()
    assert not (root/f'checkpoints/sft_new_loop_phase{other}_runs').exists()
    assert not (root/f'checkpoints/sft_new_loop_phase{other}_data').exists()
    assert failed.read_text()=='{"status":"rejected"}'
    env.update(PIPELINE_TIMESTAMP='failed_fixture',FAIL_FIXTURE_PHASE=str(phase))
    result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==17
    assert not (root/f'checkpoints/sft_new_loop_phase{other}_pipeline/failed_fixture_2rgb_endpoints').exists()
    assert 'failed_fixture' not in str((root/f'checkpoints/sft_new_loop_phase{phase}_runs/latest').resolve())
    assert failed.read_text()=='{"status":"rejected"}'
