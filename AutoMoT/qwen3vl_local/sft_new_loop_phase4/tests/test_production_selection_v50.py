from copy import deepcopy
import json
import sys
import pytest
from qwen3vl_local.sft_new_loop_phase4 import evaluation_policy as ep
from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
from qwen3vl_local.sft_new_loop_phase4.history_admission import quarantine
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import question, seal


def pool(kind='reviewed_rgb'):
    return {split: [dict(id=f'{split}/{e}/{i}', split=split, physical_group=f'{split}/{e}',
        episode=dict(event=e,instance_id=f'{split}/{e}', state='YIELD', longitudinal='HOLD'), edge='proceed', slice='readiness',
        target='YES' if i%2 else 'NO', reference_kind=kind,
        label_basis='weak_rule_teacher' if kind=='rule_teacher' else 'reviewed_transition_band',
        scenario='S',route_id=f'{split}{e}', observation={'frame_id':i})
        for e in EVENTS for i in range(2)] for split in ('train','val','test')}


def as_cases(rows, score=1.):
    correct=round(len(rows)*score)
    return [dict(id=r['id'],event=r['episode']['event'],edge=r['edge'],slice=r['slice'],
        target=r['target'],prediction=r['target'] if i<correct else ('NO' if r['target']=='YES' else 'YES'),
        reference_kind=r.get('reference_kind','reviewed_rgb')) for i,r in enumerate(rows)]


def rule_pack(tmp_path, monkeypatch, data=None):
    from qwen3vl_local.sft_new_loop_phase4 import teacher_evaluation
    path=tmp_path/'references';path.mkdir()
    for name in ('references.json','val.jsonl','test.jsonl'):
        (path/name).write_text('{}')
    data=data or pool('rule_teacher')
    monkeypatch.setattr(teacher_evaluation,'load',lambda path,split,rgb_mode:deepcopy(data[split]))
    return path


def test_reviewed_selection_is_fixed_equal_and_no_replacement():
    data=pool();extra=deepcopy(data['val'][0]);extra['id']='extra';data['val'].append(extra)
    selected,record=ep.prepare(data)
    assert len(selected['val'])==20
    assert record['plans']['val']['per_event']==2
    assert len(set(record['plans']['val']['selected_ids']))==20
    reverse={s:list(reversed(rows)) for s,rows in data.items()}
    assert ep.prepare(reverse)[1]['plans']==record['plans']
    assert len(ep.prepare(data,natural=True)[0]['val'])==21


@pytest.mark.parametrize('split',['train','val','test'])
def test_missing_event_rejected_before_training(split):
    data=pool();data[split]=[r for r in data[split] if r['episode']['event']!='U-E7']
    with pytest.raises(ValueError,match='events'):
        ep.prepare(data)


def test_rule_only_selection_score_is_nonempty_and_named_consistency(tmp_path,monkeypatch):
    data=pool('rule_teacher');data['train'].append({**data['train'][0],'id':'manual','label_basis':'reviewed_transition_band'})
    path=rule_pack(tmp_path,monkeypatch)
    selected,record=ep.prepare(data,mode='rule_consistency',references=path)
    assert 'manual' not in [r['id'] for r in selected['train']]
    cases=[dict(id=r['id'],event=r['episode']['event'],edge=r['edge'],slice=r['slice'],target=r['target'],
                prediction=r['target'],reference_kind=r['reference_kind']) for r in selected['val']]
    score=ep.score_cases(cases,'rule_consistency',record['plans']['val']['selected_ids'])
    assert score['selection_agreement']==1 and score['selection_metric']=='ten_event_rule_consistency'
    assert not score['independent_accuracy_certified']
    with pytest.raises(ValueError,match='provenance'):ep.score_cases(cases,'reviewed')
    with pytest.raises(ValueError,match='IDs'):ep.score_cases(cases[::-1],'rule_consistency',record['plans']['val']['selected_ids'])


def test_leakage_in_full_reference_pool_rejected_before_subset(tmp_path,monkeypatch):
    data=pool('rule_teacher');held=deepcopy(data);held['val'][0]['physical_group']=data['train'][0]['physical_group']
    path=rule_pack(tmp_path,monkeypatch,held)
    with pytest.raises(ValueError,match='leakage'):ep.prepare(data,mode='rule_consistency',references=path)


def test_restore_binds_reference_symlink_even_same_bytes(tmp_path,monkeypatch):
    data=pool('rule_teacher');path=rule_pack(tmp_path,monkeypatch)
    selected,record=ep.prepare(data,mode='rule_consistency',references=path)
    run=dict(training={'evaluation':record},observation_contract={'rgb_mode':4})
    assert ep.restore(data,run)==selected
    old=path/'val.jsonl';other=path/'other.jsonl';other.write_bytes(old.read_bytes());old.unlink();old.symlink_to(other)
    with pytest.raises(ValueError,match='binding changed'):ep.restore(data,run)


@pytest.mark.parametrize('mode,path',[('reviewed',True),('rule_consistency',False)])
def test_explicit_reference_mode_required(mode,path):
    with pytest.raises(ValueError,match='requires'):ep.prepare(pool(),mode=mode,references='x' if path else None)


def ambiguous_pair():
    a=question('UNKNOWN',event='U-E7');a['reasons']=['post_stop_release_requires_visual_catchup']
    a['motion_phase_evidence']={'target':'NO'};a['slice']='catchup';a['episode']['instance_id']='old';seal(a)
    b=question('YES',event='U-E7');b['episode']['instance_id']='new';seal(b)
    return [a,b]


def test_history_unknown_is_quarantined_with_yes_before_filtering():
    qs=ambiguous_pair();original=deepcopy(qs)
    kept,records=quarantine(qs)
    assert kept==[] and len(records)==1 and qs==original
    assert not records[0]['inherited_history'] and not records[0]['labels_rewritten']


@pytest.mark.parametrize('difference',['input','event','edge','target','same_instance','unrelated_unknown','history_absent'])
def test_history_gate_does_not_merge_unrelated_propositions(difference):
    a,b=ambiguous_pair()
    if difference=='input':b['input_sources'][0]['sha256']='0'*64
    if difference=='event':b['episode']['event']='U-E6'
    if difference=='edge':b['edge']='release'
    if difference=='target':b['episode']['target_corridor']='another_destination'
    if difference=='same_instance':b['episode']['instance_id']='old'
    if difference=='unrelated_unknown':a['reasons']=['occluded']
    if difference=='history_absent':a['motion_phase_evidence']={}
    seal(a);seal(b)
    kept,records=quarantine([a,b]);assert len(kept)==2 and not records


@pytest.mark.parametrize('short',[False,True])
def test_actual_training_entry_cpu_plan_uses_rule_selection(tmp_path,monkeypatch,short,capsys):
    if short:
        from qwen3vl_local.sft_new_loop_phase4.rgb_short import train
    else:
        from qwen3vl_local.sft_new_loop_phase4 import train
    data=pool('rule_teacher');path=rule_pack(tmp_path,monkeypatch)
    manifest=dict(rgb_mode=4,training_admission={'weak_supervision_rows':40},coverage={},trainable=False)
    monkeypatch.setattr(train,'load_dataset',lambda *a,**kw:(data,manifest))
    # This entry test uses native four-image references; short conversion is covered separately.
    if short:
        from qwen3vl_local.sft_new_loop_phase4.rgb_short import data as sd
        monkeypatch.setattr(sd,'paired_view',lambda d,mode,exposed,**kw:(d,[]))
        monkeypatch.setattr(sd,'exposure_inventory',lambda:(set(),{}))
    out=tmp_path/'no_run'
    monkeypatch.setattr(sys,'argv',['train','--dataset','unused','--output-dir',str(out),
        '--sampling-only','--sampling-policy','phase3_balanced','--epochs','1',
        '--selection-mode','rule_consistency','--teacher-references',str(path)])
    train.main()
    text=capsys.readouterr().out
    assert '10240' in text and not out.exists()


def test_independent_teacher_audit_routes_cannot_select_model(tmp_path,monkeypatch):
    data=pool('rule_teacher');path=rule_pack(tmp_path,monkeypatch)
    monkeypatch.setattr(ep,'teacher_audit_groups',lambda:{data['val'][0]['physical_group']})
    with pytest.raises(ValueError,match='missing events'):
        ep.prepare(data,mode='rule_consistency',references=path)


@pytest.mark.parametrize('exporter',['train','heldout'])
def test_compiler_and_exporter_quarantine_before_any_binary_processing(tmp_path,monkeypatch,exporter):
    from qwen3vl_local.sft_new_loop_phase4 import teacher_data,teacher_evaluation,candidate_pool,dataset
    qs=ambiguous_pair()
    route=tmp_path/'route.jsonl';route.write_text('{}\n'+json.dumps({'questions':qs})+'\n')
    index=tmp_path/'index.json';index.write_text(json.dumps({'routes':[{'split':'val','artifact':route.name}]}))
    if exporter=='train':
        monkeypatch.setattr(teacher_data,'accounting',lambda *a:{})
        monkeypatch.setattr(teacher_data,'validate_registry',lambda *a:{})
        monkeypatch.setattr(dataset,'holdout_reservations',lambda:{})
        monkeypatch.setattr(teacher_data,'target',lambda *a:pytest.fail('filter ran before history gate'))
        anns,report=teacher_data.compile_bundle({},index,{},rgb_mode=2)
        assert not anns and report['counts']['history_ambiguity_excluded']==2
    else:
        monkeypatch.setattr(teacher_evaluation,'accounting',lambda *a:{})
        monkeypatch.setattr(candidate_pool,'validate',lambda *a:None)
        report=teacher_evaluation.export({'sha256':'fixture'},index,tmp_path/'no_rgb',tmp_path/'out',rgb_mode=2)
        assert report['exclusions']['history_ambiguity_excluded']==2
        assert report['files']['val']['count']==0


def test_rule_reference_short_and_four_views_share_fixed_ids(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4.tests.test_short_rgb import source_row
    from qwen3vl_local.sft_new_loop_phase4.controller import Episode
    from qwen3vl_local.sft_new_loop_phase4.route_prompts import prompt
    from qwen3vl_local.sft_new_loop_phase4.identity import digest
    from qwen3vl_local.sft_new_loop_phase4.input_identity import model_input_key
    base=source_row(tmp_path)
    held=pool('rule_teacher')
    for split in ('val','test'):
        for i,row in enumerate(held[split]):
            original=deepcopy(row)
            row.update(deepcopy(base));row.update(original)
            episode=Episode(original['episode']['event'],original['id'],longitudinal='HOLD')
            row['edge']=episode.questions()[0].key
            row['episode']=episode.to_dict();row['observation']=deepcopy(base['observation'])
            # Synthetic distinct input identities; this test exercises view conversion, not decoding.
            row['image_rgb_sha256']=[digest([split,i,f]) for f in range(4)]
            row['prompt_sha256']=digest(prompt(Episode(**row['episode']),row['edge'],row['observation']))
            row['model_input_sha256']=model_input_key(row)
    path=rule_pack(tmp_path,monkeypatch,held)
    data=pool('rule_teacher')
    two,a=ep.prepare(data,mode='rule_consistency',references=path,short=True,rgb_mode=2)
    four,b=ep.prepare(data,mode='rule_consistency',references=path,short=True,rgb_mode=4)
    assert a['plans']['val']['selected_ids']==b['plans']['val']['selected_ids']
    assert all(r['observation']['history_frames']==[8,10] for r in two['val'])
    assert all(len(r['images'])==4 for r in four['val'])
    assert all(r['student_reference']['original_reference_kind']=='rule_teacher' for r in two['val'])
