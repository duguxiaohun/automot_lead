import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from qwen3vl_local.sft_new_loop_phase4 import dataset, preflight
from qwen3vl_local.sft_new_loop_phase4.admission import training_report
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT


def small_data():
    return {split:[dict(split=split,target=a,slice='readiness',physical_group=split,episode={'event':'U-E1'},edge='proceed')
                   for a in ('YES','NO')] for split in ('train','val','test')}


def test_incomplete_coverage_is_not_training_rejection():
    data=small_data();c=dataset.coverage_report(sum(data.values(),[]))
    assert not c['ready']
    r=training_report(data,c)
    assert r['trainable'] and not r['complete_coverage']
    assert r['missing_transition_cells']>0


@pytest.mark.parametrize('problem', ['empty_train','empty_val','empty_test','one_class','leak','catchup_only'])
def test_real_prerequisites_still_block(problem):
    d=small_data()
    if problem.startswith('empty_'):d[problem[6:]]=[]
    elif problem=='one_class':d['train']=d['train'][:1]
    elif problem=='leak':d['val'][0]['physical_group']='train'
    else:
        for r in d['train']:r['slice']='catchup'
    assert not training_report(d,dataset.coverage_report(sum(d.values(),[])))['trainable']


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    root=ROOT.parents[1]/'lead_data'
    if not root.exists():pytest.skip('external RGB unavailable')
    out=tmp_path_factory.mktemp('p4_admission')/'data'
    dataset.build(json.loads((ROOT/'reviewed_state_pairs_v7.json').read_text()),root,out,rgb_mode=2)
    return out


def test_current_data_can_train_without_full_coverage(built):
    data,m=dataset.load_dataset(built,require_trainable=True)
    assert m['trainable'] and not m['complete_coverage']
    assert m['counts']=={'train':423,'val':273,'test':186}
    with pytest.raises(ValueError,match='support'):
        dataset.load_dataset(built,require_complete_coverage=True)
    r=preflight.inspect(built)
    assert r['data_ready'] and r['ready'] is None and not r['model_checked']
    assert r['scope']=='data_only'


def test_server_model_checked_only_when_requested(built,tmp_path,monkeypatch):
    import qwen3vl_local.qwen35.preflight as base
    calls=[]
    def check(path,**kw):calls.append(path);return dict(ready=path.name=='server-model',errors=[])
    monkeypatch.setattr(base,'check',check)
    monkeypatch.setattr(preflight,'load_images',lambda *args:[])
    report=preflight.inspect(built)
    assert not calls and report['ready'] is None
    assert preflight.inspect(built,tmp_path/'server-model',tmp_path)['ready'] is True
    assert preflight.inspect(built,tmp_path/'missing-model',tmp_path)['ready'] is False


def test_require_ready_needs_actual_host_inputs(built):
    p=subprocess.run([sys.executable,str(ROOT/'preflight.py'),'--dataset',str(built),'--require-ready'],capture_output=True,text=True)
    assert p.returncode==2 and 'training-host check' in p.stderr


def test_real_training_loop_reaches_optimization_with_partial_coverage(built,tmp_path,monkeypatch):
    # Real torch optimizer and checkpoint orchestration; model/metrics are test
    # doubles. This is not a production Qwen/GPU quality or throughput claim.
    import torch
    from qwen3vl_local.sft_new_loop_phase4 import train
    import qwen3vl_local.qwen35.adapters as adapters
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(0.0))
        def forward(self,labels,use_cache):return SimpleNamespace(loss=(self.weight-labels).square()+.1)
    model=Tiny();bundle=SimpleNamespace(model=model,processor=SimpleNamespace(save_pretrained=lambda p:None))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.delenv('WORLD_SIZE',raising=False);monkeypatch.delenv('RANK',raising=False)
    monkeypatch.setattr(train,'load_bundle',lambda *args,**kwargs:bundle)
    monkeypatch.setattr(train,'encode',lambda *args,**kwargs:dict(labels=torch.tensor(1.0)))
    monkeypatch.setattr(train,'evaluate_rows',lambda *args,**kwargs:(dict(groups={'fixture':dict(accuracy=.5)},event_macro_accuracy_observed=.5),[]))
    monkeypatch.setattr(adapters,'save_adapter',lambda model,path:path.mkdir())
    out=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['train','--dataset',str(built),'--output-dir',str(out),'--epochs','1','--epoch-samples','20','--accumulation','2'])
    train.main()
    assert float(model.weight.detach())>0
    assert (out/'epoch_000/training_state.pt').is_file()
    assert (out/'best.json').is_file() and (out/'latest.json').is_file()
    admission=json.loads((out/'data_admission.json').read_text())
    assert admission['training']['trainable'] and not admission['coverage']['ready']
    report=json.loads((out/'final_generation.json').read_text())
    assert report['coverage_complete'] is False and 'subsets' in report['evaluation_scope']


def test_standalone_evaluation_no_longer_requires_exhaustive_coverage(built,tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import evaluate
    from qwen3vl_local.sft_new_loop_phase4.identity import digest
    _,m=dataset.load_dataset(built,require_ready=True)
    adapter=tmp_path/'adapter';adapter.mkdir()
    (adapter/'phase4_contract.json').write_text(json.dumps(dict(dataset=digest(m))))
    monkeypatch.setattr(evaluate,'load_for_inference',lambda *a:None)
    monkeypatch.setattr(evaluate,'evaluate_rows',lambda bundle,rows,root:(dict(count=len(rows)),[]))
    out=tmp_path/'eval'
    monkeypatch.setattr(sys,'argv',['evaluate','--dataset',str(built),'--adapter',str(adapter),'--output-dir',str(out),'--device','cpu'])
    evaluate.main()
    report=json.loads((out/'metrics.json').read_text())
    assert report['count']==186 and report['split']=='test'
    assert report['coverage_complete'] is False
    assert report['coverage']['missing_transition_edges']


def test_resume_best_history_is_local_to_selected_epoch(built,tmp_path,monkeypatch):
    """Actual optimizer/save/resume loop; tiny model and validation scores are doubles."""
    import torch
    import peft.utils.save_and_load as peft_io
    import qwen3vl_local.qwen35.adapters as adapters
    from qwen3vl_local.sft_new_loop_phase4 import train
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(0.))
        def forward(self,labels,use_cache):return SimpleNamespace(loss=(self.weight-labels).square()+.1)
    def load(*args,**kw):
        return SimpleNamespace(model=Tiny(),processor=SimpleNamespace(save_pretrained=lambda p:None))
    def save(model,p):
        p.mkdir();torch.save(model.state_dict(),p/'tiny.pt')
    for name in ('RANK','WORLD_SIZE','LOCAL_RANK'):monkeypatch.delenv(name,raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(train,'load_bundle',load)
    monkeypatch.setattr(train,'encode',lambda *a,**k:dict(labels=torch.tensor(1.)))
    monkeypatch.setattr(adapters,'save_adapter',save)
    monkeypatch.setattr(adapters,'validate_adapter',lambda *a,**k:None)
    monkeypatch.setattr(peft_io,'load_peft_weights',lambda p,**k:torch.load(Path(p)/'tiny.pt',weights_only=True))
    monkeypatch.setattr(peft_io,'set_peft_model_state_dict',lambda m,s:m.load_state_dict(s))
    def run(name,scores,epochs,resume=None):
        scores=iter(scores);out=tmp_path/name
        argv=['train','--dataset',str(built),'--output-dir',str(out),'--epochs',str(epochs),
              '--epoch-samples','20','--accumulation','2']
        if resume:argv+=['--resume',str(resume)]
        monkeypatch.setattr(sys,'argv',argv)
        monkeypatch.setattr(train,'evaluate_rows',lambda *a,**k:(dict(event_macro_accuracy_observed=next(scores)),[]))
        train.main()
        return out,json.loads((out/'best.json').read_text())
    from pathlib import Path
    original,last=run('original',[.5,.7,.9],3)
    assert last['epoch']==2
    baseline_history=torch.load(original/'epoch_001/training_state.pt',weights_only=True)['sampling_history']
    for name,score in [('worse',.4),('tie',.5),('better',.6)]:
        out,best=run(name,[score],2,original/'epoch_000')
        selected=(out/best['adapter']).resolve()
        assert selected==((out/'epoch_001') if score>.5 else (original/'epoch_000'))
        assert best['score']==max(.5,score) and best['epoch']==int(score>.5)
        resumed_history=torch.load(out/'epoch_001/training_state.pt',weights_only=True)['sampling_history']
        assert resumed_history==baseline_history
        assert json.loads((out/'epoch_001_sampling.json').read_text())==json.loads((original/'epoch_001_sampling.json').read_text())
    # Removing the mutable parent summary must not break a later restore.
    (original/'best.json').unlink()
    inherited,record=run('again',[.45],3,tmp_path/'worse/epoch_001')
    assert (inherited/record['adapter']).resolve()==original/'epoch_000'
    # Relocating the whole lineage preserves relative checkpoint references.
    import shutil
    moved=tmp_path/'moved';moved.mkdir()
    for name in ('original','worse'):shutil.copytree(tmp_path/name,moved/name)
    restored,record=run('relocated',[.55],3,moved/'worse/epoch_001')
    assert (restored/record['adapter']).resolve()==restored/'epoch_002'


@pytest.mark.parametrize('fault',['legacy','future','wrong_score','wrong_epoch','nonfinite','changed_assets','missing_assets','contract','selection'])
def test_inconsistent_best_history_rejected(tmp_path,fault):
    from qwen3vl_local.sft_new_loop_phase4.train import best_reference,restore_best,adapter_identity
    from qwen3vl_local.sft_new_loop_phase4.identity import write_json
    selected=tmp_path/'epoch_000';selected.mkdir();resume=tmp_path/'epoch_001';resume.mkdir()
    contract={'fixture':True}
    write_json(selected/'phase4_contract.json',contract)
    write_json(selected/'selection.json',dict(epoch=0,score=.5))
    (selected/'weights').write_bytes(b'original')
    state=dict(next_epoch=2,best=.5,best_checkpoint=best_reference(resume,selected,.5,0))
    if fault=='legacy':del state['best_checkpoint']
    elif fault=='future':state['best_checkpoint']['epoch']=2
    elif fault=='wrong_score':state['best']=.9
    elif fault=='wrong_epoch':state['best_checkpoint']['epoch']=1
    elif fault=='nonfinite':state['best']=state['best_checkpoint']['score']=float('nan')
    elif fault=='changed_assets':(selected/'weights').write_bytes(b'replacement')
    elif fault=='missing_assets':(selected/'weights').unlink()
    else:
        file='phase4_contract.json' if fault=='contract' else 'selection.json'
        write_json(selected/file,{'wrong':True})
        state['best_checkpoint']['files']=adapter_identity(selected)
    with pytest.raises(ValueError):restore_best(resume,state,contract)


def test_zero_permission_positives_reported_without_blocking_partial_training(built):
    _,m=dataset.load_dataset(built)
    for key,no in [('train/U-E2/default/return/return',0),('train/R-E3/default/enter',0),
                   ('train/U-E7/default/proceed',4)]:
        assert m['coverage']['transition_support'][key]==dict(YES=0,NO=no)
        assert key in m['training_admission']['train_missing_readiness_yes']
    assert m['trainable'] and not m['complete_coverage']
