"""Training math is real; lifecycle GPU calls in orchestration tests are explicit substitutes."""
import copy
import json
import random
import zipfile
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.audit_joint.phase3_single_v3.tests.test_single import record
from qwen3vl_local.audit_joint.phase3_single_v3 import regions as old_regions
from .. import pipeline,delivery,preflight
from ..state import atomic,lock
from ..audit import sha
from ..training import train,objective,regions,metrics
from ..training.validation import verify


def test_real_candidate_training_checkpoint_prediction_metrics(processor,model,tmp_path):
    b=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    r=record(variant='six');im=Image.new('RGB',(32,32));opt=torch.optim.AdamW(model.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.);progress=[]
    result=train.train_epoch(b,[r],opt,sched,lambda row:objective.pack(b,row,im),progress=progress.append)
    assert result['steps']==1 and result['actual_exposure']=={r['id']:1} and progress[-1]['updates']==1
    assert model.lm_head.weight.grad is not None
    ds=tmp_path/'dataset.json';ds.write_text('{}');artifact=dict(contract=regions.contract('six'),code={'test':'real_small_Qwen'})
    binding=dict(contract=artifact['contract'],code=artifact['code'],dataset_sha256=sha(ds))
    train.save_checkpoint(tmp_path/'ckpt',b,opt,sched,binding=binding,epoch=1,step=1,exposure=result['actual_exposure'],status='epoch_complete')
    before=objective.predict(b,r,im)
    with torch.no_grad():model.lm_head.weight.add_(1)
    train.restore(b,tmp_path/'ckpt',artifact=artifact,dataset_path=ds)
    after=objective.predict(b,r,im);assert before['probabilities']==after['probabilities']
    assert metrics.evaluate([dict(record=r,prediction=after)])['macro_recall'] is not None
    with pytest.raises(ValueError,match='contract'):
        train.restore(b,tmp_path/'ckpt',artifact=dict(artifact,contract=old_regions.contract('six')),dataset_path=ds)


def test_atomic_and_duplicate_pipeline_lock(tmp_path):
    p=tmp_path/'state.json';atomic(p,dict(status='a'));atomic(p,dict(status='b'));assert json.loads(p.read_text())['status']=='b'
    with lock(tmp_path):
        with pytest.raises(RuntimeError,match='another_single'):
            with lock(tmp_path):pass


@pytest.mark.parametrize('count,explicit,expected',[(4,None,['1','2','3','0']),(1,'2',['2']),(1,'3,2,1,0',['3'])])
def test_gpu_selection_is_shared_and_eval_can_use_one(monkeypatch,count,explicit,expected):
    monkeypatch.setattr(pipeline.subprocess,'check_output',lambda *_a,**_kw:'0, 12000\n1, 18000\n2, 17000\n3, 16000\n')
    monkeypatch.delenv('GPU_IDS',raising=False)
    if explicit:monkeypatch.setenv('GPU_IDS',explicit)
    assert pipeline.selected_gpus(count)==expected

@pytest.mark.parametrize('ids',['0,0,1,2','1,2','0,1,2,99'])
def test_invalid_gpu_set_rejected(monkeypatch,ids):
    monkeypatch.setenv('GPU_IDS',ids);monkeypatch.setattr(pipeline.subprocess,'check_output',lambda *_a,**_kw:'0, 12000\n1, 18000\n2, 17000\n3, 16000\n')
    with pytest.raises(ValueError):pipeline.selected_gpus()


def test_pack_core_never_silently_drops_cases(tmp_path):
    required=tmp_path/'cases.json';required.write_text(json.dumps({'payload':random.Random(5).randbytes(20000).hex()}))
    out=tmp_path/'audit.zip'
    with pytest.raises(ValueError,match='core_package_exceeds'):
        delivery.pack(out,{'cases.json':required},limit=1000)
    assert not out.exists()
    receipt=delivery.pack(out,{'cases.json':required},limit=100000)
    with zipfile.ZipFile(out) as z:assert z.read('cases.json')==required.read_bytes() and z.testzip() is None
    assert receipt['bytes']<=100000


def test_size_rehearsal_includes_full_case_members():
    r=record();r['split']='test'
    report=delivery.preflight_size(dict(records=[r]),{'plans':[]},{'status':'synthetic'})
    assert report['case_appearances']==1 and report['prediction_padding_per_case']==4096


@pytest.fixture
def lifecycle(tmp_path,monkeypatch):
    root=tmp_path/'candidate';root.mkdir();native=tmp_path/'native_ready.json';native.write_text('{}')
    ready=dict(dataset='dataset',dataset_sha256='sha');events=[]
    monkeypatch.setattr(preflight,'load_ready',lambda *_a,**_kw:ready)
    monkeypatch.setattr(pipeline,'model_identity',lambda _:dict(model='sha'))
    monkeypatch.setattr(pipeline,'selected_gpus',lambda count=4:[str(i) for i in range(count)])
    monkeypatch.setattr(pipeline.shutil,'disk_usage',lambda _:SimpleNamespace(free=100_000_000_000))
    monkeypatch.setattr(pipeline,'check_probe',lambda *_a:None)
    monkeypatch.setattr(pipeline,'training_complete',lambda *_a:dict(binding=dict(model=dict(model='sha'))))
    def worker(op,root,ready,model,ids,output,**kwargs):
        events.append(op);output.mkdir(parents=True,exist_ok=True)
        if op=='probe':atomic(output/'probe_complete.json',{})
        if op=='train':atomic(output/'training_complete.json',{})
        if op=='validate':atomic(output/'validation_complete.json',{})
        if op=='evaluate':atomic(output/'cases.json',[]);atomic(output/'metrics.json',{})
    monkeypatch.setattr(pipeline,'worker',worker)
    def finish(*_a):events.append('package');return {'package':'synthetic.zip'}
    monkeypatch.setattr(pipeline,'finish',finish)
    return root,native,events,worker


def test_orchestration_orders_stages_and_auto_packages(lifecycle):
    root,native,events,_=lifecycle
    pipeline.run('train',root,native,root.parent/'model')
    assert events==['probe','train','validate','evaluate','package']


def test_validation_failure_can_continue_without_retraining(lifecycle,monkeypatch):
    root,native,events,worker=lifecycle
    def failing(op,*args,**kwargs):
        if op=='validate':raise RuntimeError('synthetic validation failure')
        worker(op,*args,**kwargs)
    monkeypatch.setattr(pipeline,'worker',failing)
    with pytest.raises(RuntimeError,match='validation failure'):pipeline.run('train',root,native,root.parent/'model')
    assert (root/'run/training_complete.json').exists()
    monkeypatch.setattr(pipeline,'worker',worker)
    pipeline.run('train',root,native,root.parent/'model')
    assert events.count('train')==1 and events[-3:]==['validate','evaluate','package']


def test_completed_run_delivery_needs_no_gpu(lifecycle,monkeypatch):
    root,native,events,_=lifecycle
    pipeline.run('train',root,native,root.parent/'model')
    monkeypatch.setattr(pipeline,'model_identity',lambda _:pytest.fail('must not load model identity'))
    monkeypatch.setattr(pipeline,'selected_gpus',lambda *_a:pytest.fail('must not query GPU'))
    pipeline.run('train',root,native,root.parent/'model')
    assert events.count('train')==1 and events[-1]=='package'


def test_failed_cpu_never_leaves_ready(lifecycle,monkeypatch):
    root,native,events,_=lifecycle
    def reject(*_a):raise ValueError('capacity_insufficient')
    monkeypatch.setattr(preflight,'prepare',reject)
    with pytest.raises(ValueError,match='capacity'):pipeline.run('cpu',root,native,root.parent/'model')
    assert json.loads((root/'ready_1280.json').read_text())['status']=='rejected' and not events


def test_incomplete_training_is_preserved(lifecycle):
    root,native,events,_=lifecycle;(root/'run').mkdir();(root/'run/plan.json').write_text('keep')
    with pytest.raises(ValueError,match='incomplete_training'):pipeline.run('train',root,native,root.parent/'model')
    assert (root/'run/plan.json').read_text()=='keep' and events==[]


def test_ready_rejects_original_source_drift_before_gpu_or_file_reads(tmp_path,monkeypatch):
    def drift():raise ValueError('baseline_runtime_changed')
    monkeypatch.setattr(preflight,'require_received_source',drift)
    with pytest.raises(ValueError,match='baseline_runtime_changed'):
        preflight.load_ready(tmp_path/'not_created')


def test_incomplete_probe_rejected_before_model_or_gpu(lifecycle,monkeypatch):
    root,native,events,_=lifecycle;(root/'probe').mkdir()
    monkeypatch.setattr(pipeline,'model_identity',lambda _:pytest.fail('late model check'))
    with pytest.raises(ValueError,match='incomplete_probe'):pipeline.run('train',root,native,root.parent/'model')
    assert not events


def test_validation_receipt_rejects_missing_members_and_tamper(tmp_path):
    r=record();pred=dict(token_ids=[5]*4,format_valid=[True]*4,contract=regions.contract('six'),sample_binding=objective.observation_binding(r))
    folder=tmp_path/'val';folder.mkdir();checkpoint=tmp_path/'ckpt';checkpoint.mkdir();atomic(checkpoint/'receipt.json',dict(state='test'))
    cases=[dict(record=r,prediction=pred)];atomic(folder/'cases.json',cases);atomic(folder/'metrics.json',metrics.evaluate(cases))
    atomic(folder/'receipt.json',dict(checkpoint_receipt_sha256=sha(checkpoint/'receipt.json'),cases_sha256=sha(folder/'cases.json'),metrics_sha256=sha(folder/'metrics.json')))
    verify(folder,[r],checkpoint)
    with pytest.raises(ValueError,match='membership'):verify(folder,[],checkpoint)
    atomic(folder/'metrics.json',{})
    with pytest.raises(ValueError,match='receipt'):verify(folder,[r],checkpoint)

@pytest.mark.parametrize('bad',[False,True])
def test_real_four_gloo_optimizer_reload_and_coordinated_failure(tmp_path,bad):
    import subprocess,sys
    root=tmp_path/'ddp'
    subprocess.run([sys.executable,'-B','-m','qwen3vl_local.audit_joint.phase3_single_refine.tests._distributed',str(root),'bad' if bad else 'ok'],check=True,timeout=120)
    reports=[json.loads((root/f'rank_{i}.json').read_text()) for i in range(4)]
    assert {r['status'] for r in reports}==({'coordinated_input_failure'} if bad else {'passed'})
    if not bad:assert len({r['loss'] for r in reports})==1


def test_native_small_model_validation_test_and_reusable_results(processor,model,tmp_path,monkeypatch):
    """Real Qwen inference and state files; full-pool admission is a declared fixture."""
    from ..training import validation,dataset
    from qwen3vl_local.sft_new_loop_phase3 import train as native,preflight as native_preflight
    from qwen3vl_local.audit_joint import phase3_suite
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    data=tmp_path/'data';data.mkdir();Image.new('RGB',(32,32)).save(data/'rgb.jpg')
    rows=[]
    for split in ('train','val','test'):
        r=record();r.update(id=split,split=split,image='rgb.jpg',physical_group=split);rows.append(r)
    artifact=dict(contract=regions.contract('six'),code={'fixture':'tiny_Qwen'},records=rows,data_root=str(data))
    source=tmp_path/'source';source.mkdir()
    ds=source/'dataset.json';ds.write_text('{}');run=tmp_path/'run';run.mkdir()
    binding=dict(contract=artifact['contract'],code=artifact['code'],dataset_sha256=sha(ds),model={'config':'fixture'})
    opt=torch.optim.AdamW(model.parameters(),lr=.001);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    # Actual optimization was covered above; these step values exercise final-epoch policy only.
    for e in range(1,4):train.save_checkpoint(run/f'epoch_{e}',bundle,opt,sched,binding=binding,epoch=e,step=e*3200,exposure={},status='epoch_complete')
    atomic(run/'training_complete.json',dict(selection='final_completed_epoch',total_steps=9600,selected_checkpoint='epoch_3',binding=binding,
        checkpoint_receipts={f'epoch_{i}':sha(run/f'epoch_{i}/receipt.json') for i in range(1,4)}))
    monkeypatch.setattr(dataset,'load',lambda *_a,**_kw:artifact)
    monkeypatch.setattr(train,'training_members',lambda _: [rows[1]])
    monkeypatch.setattr(train,'verify_admission_binding',lambda *_a:None)
    monkeypatch.setattr(native,'load_model_with_lora',lambda *_a,**_kw:bundle)
    monkeypatch.setattr(native_preflight,'check_model',lambda *_a:None)
    monkeypatch.setattr(phase3_suite,'model_snapshot',lambda *_a:{'config':{'sha256':'fixture'}})
    args=SimpleNamespace(dataset=ds,model_dir=tmp_path/'model',output=run)
    validation.run(args)
    assert (run/'validation_complete.json').exists()
    for e in range(1,4):
        assert verify(run/f'validation_{e}_results',[rows[1]],run/f'epoch_{e}')['macro_recall'] is not None
    evaluation=SimpleNamespace(dataset=ds,variant='six',model_dir=args.model_dir,checkpoint=run/'epoch_3',split='test',ablations=False,output=run/'test')
    train.evaluate(evaluation)
    assert [x['record']['id'] for x in json.loads((run/'test/cases.json').read_text())]==['test']
    monkeypatch.setattr(native,'load_model_with_lora',lambda *_a,**_kw:pytest.fail('completed validation must be reusable'))
    validation.run(args)
    # Exercise the actual delivery member list on the real tiny-model results.
    # Ancillary full-pool receipts below are explicitly synthetic test fixtures.
    delivery_root=tmp_path/'delivery_root';delivery_root.mkdir()
    for name in ('ready_1280.json','pool_report.json','input_manifest.json','admission.json','plan.json'):
        atomic(delivery_root/name,dict(synthetic_fixture=True))
    for name in ('plan.json','admission.json','startup.json'):
        atomic(run/name,dict(synthetic_fixture=True))
    train.write_progress(run,3,dict(update=3200,updates=3200,loss=0.,elapsed_seconds=1.,remaining_seconds=0.))
    for e in range(1,4):atomic(run/f'epoch_{e}.json',dict(synthetic_fixture=True))
    proof=dict(saved_results_verified=True,synthetic_fixture=True)
    receipt=delivery.package(delivery_root,run,proof)
    again=delivery.package(delivery_root,run,proof)
    assert receipt==again and receipt['bytes']<=30_000_000
    with zipfile.ZipFile(receipt['package']) as archive:
        assert len(json.loads(archive.read('run/test/cases.json')))==1
        assert not any(n.endswith('.pt') for n in archive.namelist())
        assert all(f'run/validation_{e}_results/cases.json' in archive.namelist() for e in range(1,4))
