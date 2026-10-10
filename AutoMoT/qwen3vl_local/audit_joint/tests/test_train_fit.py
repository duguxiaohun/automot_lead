from copy import deepcopy
import json
from pathlib import Path

import pytest

from qwen3vl_local.audit_joint import train_fit as tool
from qwen3vl_local.sft_new_loop_phase4.identity import digest, write_json
from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import contract, observation_contract
from qwen3vl_local.sft_new_loop_phase4.rgb_short.diagnostics import sample_binding
from qwen3vl_local.sft_new_loop_phase4.paired_eval import case_identity
from qwen3vl_local.sft_new_loop_phase4.tests.test_short_rgb import source_row


@pytest.fixture
def sample_run(tmp_path):
    root=tmp_path/'run';root.mkdir()
    row=source_row(tmp_path)
    row.update(id='train',split='train',slice='readiness')
    row['episode'].update(event='U-E4')
    row['edge']='complete'
    val=deepcopy(row);val.update(id='val',split='val',target='NO')
    sample=dict(id='train',position=0,rank=0,local_index=0,target=row['target'],original_label_basis=row['label_basis'])
    selected={s:digest([r['id'] for r in rs]) for s,rs in dict(train=[row],val=[val],test=[]).items()}
    view=dict(source=contract(),selected_ids=selected,counts=dict(train=1,val=1,test=0),
              observation_contracts={'4':observation_contract(4)})
    run=dict(source=contract(),dataset='manifest',world_size=1,observation_contract=observation_contract(4),
             training=dict(accumulation=8,max_length=8192,paired_training={'selected_ids':selected}))
    write_json(root/'run.json',run);write_json(root/'student_view.json',view)
    cell='/'.join((*tool.coarse(row)[:2],row['target'],row['slice'],tool.motion(row)))
    write_json(root/'epoch_000_sampling.json',dict(epoch=0,world_size=1,presentations=1,
        ordered_samples=sample_binding([sample],1),strata={cell:1}))
    tool.write_rows(root/'epoch_000_samples.jsonl',[sample])
    tool.write_rows(root/'epoch_000_rank0_steps.jsonl',[dict(id='train',epoch=0,position=0,rank=0,local_index=0,
        loss=.5,optimizer_update=True,optimizer_steps=1)])
    write_json(root/'epoch_000_runtime.json',dict(validation_status='complete',ranks=[dict(rank=0,microbatches=1,optimizer_steps=1)]))
    case=dict(id='val',event='U-E4',edge='complete',slice='readiness',target='NO',prediction='YES',raw='YES',
              reference_kind=val.get('reference_kind','reviewed_rgb'),paired_identity=case_identity(val))
    tool.write_rows(root/'epoch_000_cases.jsonl',[case])
    write_json(root/'epoch_000_validation.json',dict(epoch=0,split='val',dataset='manifest',count=1,expected_count=1,
        **tool.validation_metrics([case])))
    write_json(root/'data_admission.json',dict(training=dict(counts={s:{'rows':n} for s,n in view['counts'].items()},
        train_readiness_catchup_support={'U-E4/complete':{'readiness':{'YES':1,'NO':0}}})))
    return root,dict(train=[row],val=[val],test=[])


def test_support_separates_missing_answer_from_sampled_fit_unknown(sample_run):
    root,data=sample_run
    ev=tool.evidence(root,0)
    report=tool.support_report(ev)
    assert report['val_without_pool_support']==1
    assert report['cells'][0]['status']=='missing_pool_support'
    assert report['cells'][0]['training_fit']=='not_run'
    ev['admission']['train_readiness_catchup_support']['U-E4/complete']['readiness']['NO']=1
    ev['view']['counts']['train']=2
    assert tool.support_report(ev)['cells'][0]['status']=='pool_present_not_sampled'
    ev['sampling']['strata']={'U-E4/complete/NO/readiness/'+tool.motion(data['train'][0]):1}
    assert tool.support_report(ev)['cells'][0]['status']=='sampled_fit_unknown'


@pytest.mark.parametrize('damage',['order','missing_step','wrong_step','wrong_update','val_identity','aggregate','metrics'])
def test_incomplete_or_inconsistent_evidence_rejected(sample_run,damage):
    root,_=sample_run
    if damage=='order':
        p=root/'epoch_000_samples.jsonl';v=tool.rows(p);v[0]['position']=1;p.write_text(json.dumps(v[0])+'\n')
    elif damage in ('missing_step','wrong_step','wrong_update'):
        p=root/'epoch_000_rank0_steps.jsonl';v=tool.rows(p)
        if damage=='missing_step':p.write_text('')
        else:
            v[0]['id' if damage=='wrong_step' else 'optimizer_steps']='bad'
            p.write_text(json.dumps(v[0])+'\n')
    elif damage=='val_identity':
        p=root/'epoch_000_cases.jsonl';v=tool.rows(p);v[0]['target']='YES';p.write_text(json.dumps(v[0])+'\n')
    elif damage=='metrics':
        p=root/'epoch_000_validation.json';v=tool.read(p);v['selection_agreement']=1;write_json(p,v)
    else:
        p=root/'data_admission.json';v=tool.read(p);v['training']['train_readiness_catchup_support']={};write_json(p,v)
    with pytest.raises(ValueError):tool.support_report(tool.evidence(root,0))


def test_train_selection_is_complete_deduplicated_and_checks_actual_rows(sample_run):
    root,data=sample_run;ev=tool.evidence(root,0)
    selected,counts=tool.select_training(data,ev,'U-E4','complete')
    assert [r['id'] for r in selected]==['train'] and counts['train']==1
    assert tool.select_training(data,ev,'U-E7','release')[0]==[]
    data['train'][0]['target']='NO'
    with pytest.raises(ValueError,match='bound train row'):tool.select_training(data,ev,'U-E4','complete')


def test_same_category_different_state_is_not_counted_as_matching(sample_run):
    _,data=sample_run
    data['val'][0]['target']='YES'
    data['val'][0]['episode']['longitudinal']='HOLD'
    data['train'][0]['episode']['longitudinal']='STABLE'
    report=tool.distribution_report(data,{'train':1},'U-E4','complete')
    assert report[0]['pool_rows']==report[0]['sampled_presentations']==0


def test_fit_unique_vs_exposure_weighting(sample_run):
    _,data=sample_run;r=data['train'][0]
    a=dict(paired_identity=case_identity(r),original_label_basis=r['label_basis'],sampled_presentations=3,
           target='YES',prediction='YES')
    b={**a,'target':'NO','prediction':'YES','sampled_presentations':1,
       'paired_identity':case_identity(data['val'][0])}
    result=tool.fit_summary([a,b])
    assert result['unique_questions']['count']==2 and result['exposure_weighted']['count']==4
    assert result['unique_questions']['targets']=={'YES':1,'NO':1}
    assert result['exposure_weighted']['targets']=={'YES':3,'NO':1}


@pytest.mark.parametrize('outcome',['success','failure','partial_failure','prepare_only','no_samples','view_drift','adapter_drift','interrupt','raw_conflict'])
def test_fit_cli_preserves_results_and_failure(sample_run,tmp_path,monkeypatch,outcome):
    import sys
    root,data=sample_run
    # Exercise native selection and case serialization; only expensive model/dataset loading is substituted.
    if outcome=='partial_failure':
        other=deepcopy(data['train'][0]);other['id']='train_second';data['train'].append(other)
        view=tool.read(root/'student_view.json');view['counts']['train']=2
        view['selected_ids']['train']=digest(['train','train_second']);write_json(root/'student_view.json',view)
        run=tool.read(root/'run.json');run['training']['paired_training']['selected_ids']=view['selected_ids'];write_json(root/'run.json',run)
        admission=tool.read(root/'data_admission.json');admission['training']['counts']['train']['rows']=2
        admission['training']['train_readiness_catchup_support']['U-E4/complete']['readiness']['YES']=2;write_json(root/'data_admission.json',admission)
        first=tool.rows(root/'epoch_000_samples.jsonl')[0]
        samples=[first,{**first,'id':'train_second','position':1,'local_index':1}]
        (root/'epoch_000_samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in samples))
        sampling=tool.read(root/'epoch_000_sampling.json');sampling['presentations']=2
        sampling['ordered_samples']=sample_binding(samples,1)
        sampling['strata']={k:2 for k in sampling['strata']};write_json(root/'epoch_000_sampling.json',sampling)
        steps=[dict(epoch=0,loss=.5,optimizer_update=bool(i),optimizer_steps=i,
                    **{k:r[k] for k in ('id','rank','position','local_index')}) for i,r in enumerate(samples)]
        (root/'epoch_000_rank0_steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in steps))
        write_json(root/'epoch_000_runtime.json',dict(validation_status='complete',ranks=[dict(rank=0,microbatches=2,optimizer_steps=1)]))
    manifest={'dataset':'fixture'}
    run=tool.read(root/'run.json');run['dataset']=digest(manifest);write_json(root/'run.json',run)
    v=tool.read(root/'epoch_000_validation.json');v['dataset']=digest(manifest);write_json(root/'epoch_000_validation.json',v)
    adapter=root/'epoch_000';adapter.mkdir();write_json(adapter/'phase4_contract.json',run)
    (adapter/'adapter_model.safetensors').write_bytes(b'fixture')
    write_json(root/'epoch_000_checkpoint.json',dict(epoch=0,adapter='epoch_000',files=tool.adapter_identity(adapter)))
    (tmp_path/'dataset').mkdir()
    write_json(tmp_path/'dataset'/'student_view.json',tool.read(root/'student_view.json'))
    monkeypatch.setattr(tool,'load_dataset',lambda *a,**kw:(data,manifest))
    calls=[]
    monkeypatch.setattr(tool,'load_for_inference',lambda *a:calls.append(a))
    generated=[]
    def fail(*args):
        if outcome=='interrupt':raise KeyboardInterrupt()
        if outcome=='partial_failure' and not generated:
            generated.append(True)
            return 'YES','YES'
        raise RuntimeError('inference failed')
    def generate(*args):
        if outcome=='raw_conflict':return 'YES','NO\n'
        if outcome=='view_drift':
            view=tmp_path/'dataset'/'student_view.json'
            replacement=tmp_path/'replacement.json';replacement.write_bytes(view.read_bytes())
            view.unlink();view.symlink_to(replacement)
        elif outcome=='adapter_drift':
            (adapter/'adapter_model.safetensors').write_bytes(b'changed')
        return 'YES','YES'
    monkeypatch.setattr(tool,'generate',fail if outcome in ('failure','partial_failure','interrupt') else generate)
    output=tmp_path/'fit'
    monkeypatch.setattr(sys,'argv',['fit','fit','--run',str(root),'--dataset',str(tmp_path/'dataset'),
        '--model-dir',str(tmp_path/'model'),'--data-root',str(tmp_path/'rgb'),'--output',str(output),
        '--device','cpu','--event','U-E7' if outcome=='no_samples' else 'U-E4']
        +(['--prepare-only'] if outcome=='prepare_only' else []))
    assert tool.main()==(130 if outcome=='interrupt' else 2 if outcome in ('failure','partial_failure','no_samples','view_drift','adapter_drift','raw_conflict') else 0)
    report=tool.read(output/'fit_report.json')
    if outcome in ('partial_failure','view_drift','adapter_drift'):
        assert report['status']=='failed' and report['completed_count']==1
        assert report['training_fit']=='partial'
        assert len(tool.rows(output/'cases.jsonl'))==1
        assert not (output/'fit_metrics.json').exists()
    elif outcome in ('failure','interrupt','raw_conflict'):
        assert report['stage']=='inference' and report['status']=='failed'
        assert report['completed_count']==0 and report['training_fit']=='not_run'
    elif outcome=='success':
        assert report['completed_count']==1 and report['training_fit']=='measured'
        assert tool.read(output/'fit_metrics.json')['unique_questions']['reference_agreement']==1
        assert tool.verify_fit(output)['status']=='internal_consistency_verified'
        assert tool.verify_fit(output,run=root,dataset=tmp_path/'dataset')['status']=='verified'
    else:
        assert calls==[] and report['training_fit']=='not_run'
    assert (output/'fit_selection.jsonl').is_file()
    if outcome!='success':
        with pytest.raises(ValueError,match='not complete'):tool.verify_fit(output)
    with pytest.raises(FileExistsError):tool.main()


@pytest.mark.parametrize('damage',['none','changed_bytes','missing_case','metric','selection_identity'])
def test_received_fit_recomputes_complete_membership_and_metrics(sample_run,tmp_path,damage):
    _,data=sample_run
    row=data['train'][0]
    output=tmp_path/'received';output.mkdir()
    item=dict(id=row['id'],paired_identity=case_identity(row),sampled_presentations=1,
              original_label_basis=row['label_basis'],cell=list(tool.state_cell(row)))
    case=dict(item,**{k:row[k] for k in ('target','edge','slice')},event=row['episode']['event'],
          reference_kind=row.get('reference_kind','reviewed_rgb'),prediction='YES',raw='YES')
    tool.write_rows(output/'fit_selection.jsonl',[item])
    tool.write_rows(output/'cases.jsonl',[case])
    metrics=tool.fit_summary([case])
    if damage=='metric':metrics['unique_questions']['reference_agreement']=0
    if damage=='missing_case':(output/'cases.jsonl').write_text('')
    if damage=='selection_identity':
        item['sampled_presentations']=2
        (output/'fit_selection.jsonl').write_text(json.dumps(item)+'\n')
    write_json(output/'fit_metrics.json',metrics)
    write_json(output/'fit_report.json',dict(schema='joint_training_fit_v1',command='fit',status='complete',
        training_fit='measured',checkpoint_assets_verified=True,selection_count=1,completed_count=1,
        epoch=0,request=dict(command='fit',epoch=0,event='U-E4',edge='complete',phase='readiness'),
        output_files_sha256={n:tool.file_sha(output/n) for n in tool.FIT_FILES}))
    if damage=='changed_bytes':(output/'cases.jsonl').write_text('')
    if damage=='none':assert tool.verify_fit(output)['unique_questions']==1
    else:
        with pytest.raises(ValueError):tool.verify_fit(output)


def test_support_rejects_sampled_category_absent_from_pool_even_without_val(sample_run):
    root,_=sample_run;ev=tool.evidence(root,0)
    key=next(iter(ev['sampling']['strata']))
    ev['sampling']['strata']={key.replace('complete','release'):1}
    with pytest.raises(ValueError,match='absent from pool'):tool.support_report(ev)


def test_same_sample_id_cannot_change_answer_with_consistent_order_hash(sample_run):
    root,_=sample_run
    first=tool.rows(root/'epoch_000_samples.jsonl')[0]
    samples=[first,{**first,'position':1,'local_index':1,'target':'NO'}]
    (root/'epoch_000_samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in samples))
    sampling=tool.read(root/'epoch_000_sampling.json')
    sampling.update(presentations=2,ordered_samples=sample_binding(samples,1))
    write_json(root/'epoch_000_sampling.json',sampling)
    with pytest.raises(ValueError,match='conflicting target'):tool.evidence(root,0)


def test_fit_files_are_included_in_handoff(tmp_path):
    from qwen3vl_local.audit_joint.handoff import result_files
    names={'fit_report.json','fit_metrics.json','fit_selection.jsonl','support_report.json','cases.jsonl'}
    for name in names|(set(['adapter_model.safetensors'])):(tmp_path/name).write_text('{}')
    included,_=result_files(tmp_path)
    assert set(included)==names


@pytest.fixture
def bound_fit(sample_run,tmp_path,monkeypatch):
    root,data=sample_run
    second=deepcopy(data['train'][0]);second['id']='train_second'
    outside=deepcopy(second);outside.update(id='other_edge',edge='proceed',slice='catchup')
    data['train'].extend([second,outside])
    view=tool.read(root/'student_view.json');view['counts']['train']=3
    view['selected_ids']['train']=digest([r['id'] for r in data['train']])
    write_json(root/'student_view.json',view)
    manifest={'fixture':'complete original pool'}
    run=tool.read(root/'run.json');run['dataset']=digest(manifest)
    run['training']['paired_training']['selected_ids']=view['selected_ids'];write_json(root/'run.json',run)
    val=tool.read(root/'epoch_000_validation.json');val['dataset']=digest(manifest)
    write_json(root/'epoch_000_validation.json',val)
    exposure_rows=[data['train'][i] for i in (0,1,0,2)]
    samples=[dict(id=r['id'],target=r['target'],original_label_basis=r['label_basis'],
                  position=i,rank=0,local_index=i) for i,r in enumerate(exposure_rows)]
    (root/'epoch_000_samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in samples))
    strata=tool.Counter('/'.join((r['episode']['event'],r['edge'],r['target'],r['slice'],tool.motion(r))) for r in exposure_rows)
    write_json(root/'epoch_000_sampling.json',dict(epoch=0,world_size=1,presentations=4,
        ordered_samples=sample_binding(samples,1),strata=dict(strata)))
    steps=[dict(epoch=0,loss=.5,optimizer_update=i==3,optimizer_steps=int(i==3),
                **{k:r[k] for k in ('id','rank','position','local_index')}) for i,r in enumerate(samples)]
    (root/'epoch_000_rank0_steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in steps))
    write_json(root/'epoch_000_runtime.json',dict(validation_status='complete',ranks=[dict(rank=0,microbatches=4,optimizer_steps=1)]))
    write_json(root/'data_admission.json',dict(training=dict(counts={s:{'rows':n} for s,n in view['counts'].items()},
        train_readiness_catchup_support={'U-E4/complete':{'readiness':{'YES':2}},'U-E4/proceed':{'catchup':{'YES':1}}})))
    checkpoint=dict(epoch=0,adapter='epoch_000',files={'adapter_model.safetensors':'test-receipt-only'})
    write_json(root/'epoch_000_checkpoint.json',checkpoint)
    dataset=tmp_path/'original_view';dataset.mkdir();write_json(dataset/'student_view.json',view)
    monkeypatch.setattr(tool,'load_dataset',lambda *a,**kw:(data,manifest))
    ev=tool.evidence(root,0)
    output=tmp_path/'fit';output.mkdir()
    selected,counts=tool.select_training(data,ev,'U-E4','complete')
    selection=[dict(id=r['id'],paired_identity=case_identity(r),sampled_presentations=counts[r['id']],
                    original_label_basis=r['label_basis'],cell=list(tool.state_cell(r))) for r in selected]
    cases=[dict(s,**{k:r[k] for k in ('target','edge','slice')},event=r['episode']['event'],
                reference_kind=r.get('reference_kind','reviewed_rgb'),prediction='YES',raw='YES\n') for r,s in zip(selected,selection)]
    report=dict(schema='joint_training_fit_v1',command='fit',status='complete',training_fit='measured',
        checkpoint_assets_verified=True,selection_count=2,completed_count=2,epoch=0,
        request=dict(command='fit',epoch=0,event='U-E4',edge='complete',phase='readiness'),
        sources_sha256=ev['hashes'],dependency_source=run['source'],checkpoint_files_sha256=checkpoint['files'],
        checkpoint_receipt_sha256=tool.file_sha(root/'epoch_000_checkpoint.json'))
    save_fit_fixture(output,selection,cases,report)
    return output,root,dataset,data


def save_fit_fixture(output,selection,cases,report):
    # Deliberately rehash coherent counterexamples: verification must check independent evidence.
    for name,items in [('fit_selection.jsonl',selection),('cases.jsonl',cases)]:
        (output/name).write_text(''.join(json.dumps(r)+'\n' for r in items))
    write_json(output/'fit_metrics.json',tool.fit_summary(cases))
    report['selection_count']=len(selection);report['completed_count']=len(cases)
    report['output_files_sha256']={n:tool.file_sha(output/n) for n in tool.FIT_FILES}
    write_json(output/'fit_report.json',report)


@pytest.mark.parametrize('damage',['none','wrong_query','missing_hashes','wrong_hash','omitted_sample',
    'extra_id','wrong_exposure','wrong_identity','wrong_epoch','wrong_checkpoint','raw_conflict'])
def test_bound_verification_requires_original_query_and_full_sampled_set(bound_fit,damage):
    output,root,dataset,_=bound_fit
    selection=tool.rows(output/'fit_selection.jsonl');cases=tool.rows(output/'cases.jsonl')
    report=tool.read(output/'fit_report.json')
    if damage=='wrong_query':report['request'].update(event='U-E1',edge='proceed',phase='catchup');report.pop('sources_sha256')
    elif damage=='missing_hashes':report.pop('sources_sha256')
    elif damage=='wrong_hash':report['sources_sha256']['run.json']='0'*64
    elif damage=='omitted_sample':selection.pop();cases.pop()
    elif damage=='extra_id':selection[0]['id']=cases[0]['id']='not_sampled'
    elif damage=='wrong_exposure':selection[0]['sampled_presentations']=cases[0]['sampled_presentations']=1
    elif damage=='wrong_identity':
        for item in (selection[0],cases[0]):
            item['paired_identity']['fields']['images'][0]='different/0004.jpg'
            item['paired_identity']['sha256']=digest(item['paired_identity']['fields'])
    elif damage=='wrong_epoch':report['request']['epoch']=1
    elif damage=='wrong_checkpoint':report['checkpoint_receipt_sha256']='0'*64
    elif damage=='raw_conflict':cases[0]['raw']='NO\n'
    save_fit_fixture(output,selection,cases,report)
    if damage=='none':
        assert tool.verify_fit(output,run=root,dataset=dataset)['selection_completeness']=='verified'
    else:
        with pytest.raises(ValueError):tool.verify_fit(output,run=root,dataset=dataset)
        if damage in ('wrong_query','raw_conflict'):
            with pytest.raises(ValueError):tool.verify_fit(output)


def test_without_source_or_dataset_only_internal_consistency_can_pass(bound_fit,monkeypatch,capsys):
    import sys
    output,root,_,_=bound_fit
    for options in ({},{'run':root}):
        result=tool.verify_fit(output,**options)
        assert result['status']=='internal_consistency_verified'
        assert result['selection_completeness'].startswith('not_verified')
    report=tool.read(output/'fit_report.json');report.pop('sources_sha256');write_json(output/'fit_report.json',report)
    monkeypatch.setattr(sys,'argv',['audit','verify-fit',str(output)])
    assert tool.main()==2
    assert json.loads(capsys.readouterr().out)['status']=='internal_consistency_verified'
    with pytest.raises(FileNotFoundError):tool.verify_fit(output,run=root/'missing')


def test_original_package_binding_allows_cross_cwd_and_relocated_run(bound_fit,tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.tests.test_handoff import HandoffTests
    from qwen3vl_local.audit_joint.handoff import pack
    import zipfile
    output,root,dataset,_=bound_fit
    capture=HandoffTests();capture.setUp()
    try:
        archive=pack(capture.capture,results=[('original',root)])['archive']
        package=tmp_path/'received_original'
        with zipfile.ZipFile(archive) as z:z.extractall(package)
        monkeypatch.chdir(tmp_path)
        result=tool.verify_fit(output,package=package,result='original',dataset=dataset)
        assert result['status']=='verified' and result['package']['status']=='verified'
        with pytest.raises(ValueError):tool.verify_fit(output,package=package,result='../original',dataset=dataset)
    finally:capture.doCleanups()


@pytest.mark.parametrize('raw,prediction,valid',[('NO\n','YES',False),('YES explanation','YES',False),
    ('YES explanation','MALFORMED',True),('  NO\n','NO',True),('UNKNOWN','UNKNOWN',False),
    ('UNKNOWN','MALFORMED',True),('', 'MALFORMED',True),(None,'MALFORMED',False)])
def test_frozen_parser_rechecks_validation_and_fit(bound_fit,raw,prediction,valid):
    output,root,_,_=bound_fit
    path=root/'epoch_000_cases.jsonl';items=tool.rows(path)
    items[0].update(raw=raw,prediction=prediction)
    path.write_text(''.join(json.dumps(r)+'\n' for r in items))
    metrics=tool.read(root/'epoch_000_validation.json');metrics.update(tool.validation_metrics(items))
    write_json(root/'epoch_000_validation.json',metrics)
    selection=tool.rows(output/'fit_selection.jsonl');cases=tool.rows(output/'cases.jsonl')
    cases[0].update(raw=raw,prediction=prediction)
    save_fit_fixture(output,selection,cases,tool.read(output/'fit_report.json'))
    if valid:
        tool.evidence(root,0)
        assert tool.verify_fit(output)['status']=='internal_consistency_verified'
    else:
        with pytest.raises(ValueError,match='frozen parsing|raw prediction'):tool.evidence(root,0)
        with pytest.raises(ValueError,match='frozen parsing|raw prediction'):tool.verify_fit(output)
