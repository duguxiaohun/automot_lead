import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from qwen3vl_local.audit_joint import phase4_training as tool
from qwen3vl_local.audit_joint.training_budget import COMPONENTS
from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import contract,observation_contract
from qwen3vl_local.sft_new_loop_phase4.rgb_short.train import adapter_identity,best_reference
from qwen3vl_local.audit_joint.io import file_sha


def budget():
    return dict(modes=2,retained_checkpoints_per_mode=1,free_bytes=100*1024**3,reserve_bytes=20*1024**3,
                components={k:dict(upper_bound_bytes=1024,evidence='fixture') for k in COMPONENTS})


def make_checkpoint(tmp_path,mode):
    root=tmp_path/f'old{mode}';p=root/'epoch_000';p.mkdir(parents=True)
    training=dict(batch_size=2,anchor_stride=2 if mode==2 else 1,accumulation=4,epoch_samples=10240,
        seed=20260929,lr=2e-5,max_length=8192,lora_rank=16,lora_alpha=32,vision_scope='off',cap=8,
        require_complete_coverage=False,sampling_policy='contiguous_event_equal',max_question_repeat=None,
        evaluation=dict(mode='rule_consistency'))
    run=dict(source=contract(),world_size=4,observation_contract=observation_contract(mode),training=training)
    (p/'phase4_contract.json').write_text(json.dumps(run));(p/'selection.json').write_text(json.dumps(dict(epoch=0,score=.5)))
    (p/'adapter_config.json').write_text('{}')
    torch.save(dict(optimizer={},next_epoch=1,best=.5,best_checkpoint=best_reference(p,p,.5,0),
                    sampling_history=dict(next_epoch=1)),p/'training_state.pt')
    state=p/'training_state.pt'
    record=dict(epoch=0,adapter=p.name,files=adapter_identity(p),training_state=dict(bytes=state.stat().st_size,sha256=file_sha(state)))
    (root/'epoch_000_checkpoint.json').write_text(json.dumps(record));return p


@pytest.mark.parametrize('mode',[2,4])
def test_resume_inspection_rejects_damage_wrong_mode_and_finished_target(tmp_path,mode):
    p=make_checkpoint(tmp_path,mode)
    record,_=tool.checkpoint(p,mode,3)
    assert record['next_epoch']==1
    with pytest.raises(ValueError,match='target already completed'):tool.checkpoint(p,mode,1)
    with pytest.raises(ValueError,match='mode/world'):tool.checkpoint(p,6-mode,3)
    with (p/'training_state.pt').open('ab') as f:f.write(b'bad')
    with pytest.raises(ValueError,match='bytes changed'):tool.checkpoint(p,mode,3)


def test_extended_budget_accounts_for_all_epochs_and_refuses_low_disk(tmp_path,monkeypatch):
    monkeypatch.setattr(tool.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    request=budget();report=tool.extended_budget(request,tmp_path,3)
    assert report['checkpoint_count']==6 and report['request']['components']['logs_and_reports']['upper_bound_bytes']==3072
    assert request['retained_checkpoints_per_mode']==1
    monkeypatch.setattr(tool.shutil,'disk_usage',lambda p:SimpleNamespace(free=20*1024**3+15000))
    with pytest.raises(ValueError,match='multi-epoch'):tool.extended_budget(request,tmp_path,3)


@pytest.mark.parametrize('failure',[None,'resume_check_rgb4','train_rgb2'])
def test_both_resume_checks_before_gpu_and_absolute_final_test_target(tmp_path,monkeypatch,failure):
    prepared=tmp_path/'prepared';model=tmp_path/'model';data=tmp_path/'data';view=tmp_path/'view';view.mkdir();data.mkdir()
    for name in ('data4/manifest.json','references4/references.json','references4/val.jsonl','references4/test.jsonl'):
        p=prepared/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    (prepared/'first_epoch_budget.json').write_text(json.dumps(budget()));model.mkdir();(model/'config.json').write_text('{}')
    (view/'student_view.json').write_text(json.dumps(dict(source=contract(),parent=dict(logical_path=str(prepared/'data4')))))
    two=make_checkpoint(tmp_path,2);four=make_checkpoint(tmp_path,4)
    monkeypatch.setattr(tool.first,'source_check',lambda *a:dict(status='matched'))
    calls=[]
    def execute(command,log,env):
        calls.append((log.stem,command,env));log.write_text('fixture');return 7 if log.stem==failure else 0
    monkeypatch.setattr(tool.first,'execute',execute)
    out=tmp_path/'output'
    assert tool.run(prepared,view,out,data,model,epochs=3,resume_rgb2=two,resume_rgb4=four)==(7 if failure else 0)
    stages=[x[0] for x in calls]
    assert stages[:3]==['admission','resume_check_rgb2','resume_check_rgb4']
    if failure=='resume_check_rgb4':assert len(calls)==3
    for stage,args,env in calls:
        if stage.startswith('train'):
            assert args[args.index('--epochs')+1]=='3' and args[args.index('--batch-size')+1]=='2'
            assert args[args.index('--accumulation')+1]=='4'
            assert args[args.index('--resume')+1]==str(two if stage.endswith('2') else four)
        if stage.startswith('test'):
            assert args[args.index('--adapter')+1].endswith('epoch_002')
        assert env['P4_VALIDATED_DATA']==str(out/'validated_data')
    report=json.loads((out/'training_workflow.json').read_text())
    assert report['training_accepted'] is False
    assert report['mode_configuration']['2']['first_epoch']==1


@pytest.mark.parametrize('mode',[2,4])
def test_actual_phase4_epoch_resume_matches_three_epoch_run(tmp_path,monkeypatch,mode):
    import sys
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import train,evaluation_policy
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
    from qwen3vl_local.qwen35 import adapters
    from peft.utils import save_and_load
    rows=[dict(id=f'{event}/{i}',scenario='S',route_id=event,physical_group=event,split='train',
        episode=dict(event=event,instance_id=event),observation=dict(frame_id=i),edge='hold',target='YES',
        label_basis='weak_rule_teacher') for event in EVENTS for i in range(6)]
    manifest=dict(training_admission={},coverage={'ready':False},student_view={'selected_ids':{}},observation_contract=observation_contract(mode))
    monkeypatch.setattr(train,'load_dataset',lambda *a,**kw:({'train':rows,'val':[],'test':[]},manifest))
    monkeypatch.setattr(evaluation_policy,'prepare',lambda data,**kw:(data,dict(training_admission={'evaluation_scope':'fixture'},
        selection_metric='fixture',plans={'val':{'selected_ids':[]}})))
    monkeypatch.setattr(evaluation_policy,'score_cases',lambda *a,**kw:dict(selection_agreement=.5))
    monkeypatch.setattr(train,'evaluate_rows',lambda *a,**kw:({},[]))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.));self.drop=torch.nn.Dropout(.25)
        def forward(self,input_ids,**kw):return SimpleNamespace(loss=self.drop(self.weight*input_ids.float()).square().mean())
    class Processor:
        def save_pretrained(self,path):(path/'tokenizer.json').write_text('{}')
    models=[]
    def load(*a,**kw):
        model=Model();models.append(model);return SimpleNamespace(model=model,processor=Processor(),device='cpu')
    monkeypatch.setattr(train,'load_bundle',load)
    monkeypatch.setattr(train,'encode_batch',lambda b,rows,*a,**kw:dict(input_ids=torch.tensor([[int(r['id'].split('/')[-1])+1]*3 for r in rows])))
    def save(model,path):
        path.mkdir(parents=True);(path/'adapter_config.json').write_text('{}');torch.save(model.state_dict(),path/'weights.pt')
    monkeypatch.setattr(adapters,'save_adapter',save);monkeypatch.setattr(adapters,'validate_adapter',lambda *a,**kw:None)
    monkeypatch.setattr(save_and_load,'load_peft_weights',lambda path,**kw:torch.load(Path(path)/'weights.pt',weights_only=True))
    monkeypatch.setattr(save_and_load,'set_peft_model_state_dict',lambda model,state:model.load_state_dict(state))
    def run(name,epochs,resume=None):
        out=tmp_path/name
        args=['train','--dataset','fixture','--rgb-mode',str(mode),'--output-dir',str(out),'--epochs',str(epochs),
              '--epoch-samples','40','--batch-size','2','--accumulation','4']
        if resume:args+=['--resume',str(resume)]
        monkeypatch.setattr(sys,'argv',args);train.main();return out
    full=run('full',3);first=run('first',1);resumed=run('resumed',3,first/'epoch_000')
    from qwen3vl_local.sft_new_loop_phase3.test_training_state import equal
    a=torch.load(full/'epoch_002/training_state.pt',weights_only=True)
    b=torch.load(resumed/'epoch_002/training_state.pt',weights_only=True)
    for k in ('optimizer','sampling_history','next_epoch'):equal(a[k],b[k])
    equal(torch.load(full/'epoch_002/weights.pt',weights_only=True),torch.load(resumed/'epoch_002/weights.pt',weights_only=True))
    for epoch in (1,2):assert (full/f'epoch_{epoch:03d}_samples.jsonl').read_bytes()==(resumed/f'epoch_{epoch:03d}_samples.jsonl').read_bytes()
    assert json.loads((resumed/'best.json').read_text())['inherited'] is True


def test_selected_shell_forwards_total_epochs_and_independent_resume_paths(tmp_path):
    import os,subprocess,sys
    binary=tmp_path/'python';capture=tmp_path/'args.json'
    binary.write_text('#!'+sys.executable+'\nimport sys,json,os\nfrom pathlib import Path\nPath(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n')
    binary.chmod(0o755)
    env={k:v for k,v in os.environ.items() if k not in ('BATCH_SIZE','ACCUMULATION','UPDATE_SAMPLES','GRADIENT_CHECKPOINTING')}
    env.update(PYTHON=str(binary),CAPTURE=str(capture),PREPARED='prepared',DATASET='view',P4_PRODUCTION_CACHE='routes',
               OUTPUT_DIR='new out',EPOCHS='3',RESUME_RGB2='old two/epoch_000',RESUME_RGB4='old four/epoch_001')
    script=tool.first.APP/'qwen3vl_local/audit_joint/train_phase4_batch.sh'
    result=subprocess.run(['bash',str(script)],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    args=json.loads(capture.read_text())
    assert args[:2]==['-m','qwen3vl_local.audit_joint.phase4_training']
    for flag,value in [('--epochs','3'),('--resume-rgb2','old two/epoch_000'),('--resume-rgb4','old four/epoch_001')]:
        assert args[args.index(flag)+1]==value
