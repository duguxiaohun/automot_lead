"""A known blocker must survive a different unresolved permission criterion."""
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_events as events, teacher_rules as rules
from qwen3vl_local.sft_new_loop_phase4.tests.test_ten_event_teacher import frames, seed, episode


@pytest.mark.parametrize('clear,priority,expected', [
    (False, False, True), (False, True, True), (False, None, True),
    (True, False, True), (True, True, False), (True, None, None),
    (None, False, True), (None, True, None), (None, None, None)])
def test_partial_information_preserves_restrictions(clear, priority, expected, monkeypatch):
    monkeypatch.setattr(events, 'clearance', lambda *a, **k: clear)
    monkeypatch.setattr(events, 'control_priority', lambda *a, **k: priority)
    ep=episode('R-E5')
    facts,_=events.facts(frames('R-E5',blocked=True),ep,seed('R-E5'))
    assert facts['restriction_present'] is expected
    assert facts['stationary_wait_required'] is expected
    # The event transition still requires every permissive condition.
    edge=next(e for e in ep.questions() if e.key=='proceed')
    assert rules.answer(edge.criteria,facts)==('YES' if clear is True and priority is True else
                                             'NO' if clear is False or priority is False else 'UNKNOWN')


def test_shared_discontinuity_enters_native_teacher_admission():
    from qwen3vl_local.audit_joint.label_quarantine import LEDGER
    from qwen3vl_local.sft_new_loop_phase4.risk_review import registry,teacher_window_check
    import json
    for raw in json.loads(LEDGER.read_text())['records']:
        risks=[r for r in registry()[raw['scenario'],raw['route_id']] if r['ledger']==LEDGER.name
               and r['record']['first_affected_frame']==raw['first_affected_frame']]
        assert risks
        before,after=raw['last_unaffected_frame'],raw['first_affected_frame']
        q=dict(causal_sources=[dict(frame_id=before-6),dict(frame_id=before)])
        assert teacher_window_check(q,risks)['outside_registered_windows']
        q['causal_sources'].append(dict(frame_id=after+6))
        assert not teacher_window_check(q,risks)['outside_registered_windows']


def test_source_exclusion_cannot_be_removed_or_promoted_to_yes():
    import json
    from copy import deepcopy
    from qwen3vl_local.audit_joint.label_quarantine import LEDGER
    from qwen3vl_local.sft_new_loop_phase4.risk_review import registry, source_quarantine, validate_rows
    raw=json.loads(LEDGER.read_text())['records'][0]
    row=dict(scenario=raw['scenario'],route_id=raw['route_id'],target='UNKNOWN',
             label_basis='reviewed_transition_band',observation={'history_frames':[58,60]})
    known={(raw['scenario'],raw['route_id']):[r for r in registry()[raw['scenario'],raw['route_id']] if r['ledger']==LEDGER.name]}
    row['source_quarantine']=source_quarantine(row,known)
    assert validate_rows([row],known)['row_dispositions']['confirmed_source_quarantine']==1
    for field,value in [('target','YES'),('source_quarantine',None)]:
        bad=deepcopy(row);bad[field]=value
        with pytest.raises(ValueError,match='quarantine'):validate_rows([bad],known)


def test_original_manual_annotations_build_without_whole_band_reapproval(tmp_path):
    import json
    from qwen3vl_local.sft_new_loop_phase4.identity import ROOT
    from qwen3vl_local.sft_new_loop_phase4.dataset import build,load_dataset,read_rows
    root=ROOT.parents[1]/'lead_data'
    if not root.exists():pytest.skip('external RGB unavailable')
    anns=json.loads((ROOT/'reviewed_state_pairs_v9.json').read_text())
    ann=next(a for a in anns if a['evidence_id']=='p4_076/boundary_v3/enter')
    out=tmp_path/'dataset';build([ann],root,out,rgb_mode=4)
    data,_=load_dataset(out)
    rows=sum(data.values(),[])+read_rows(out/'review_queue.jsonl')
    affected=[r for r in rows if 59<=r['observation']['frame_id']<=65]
    assert len(affected)==7 and all(r['target']=='UNKNOWN' and r['source_quarantine'] for r in affected)
    assert all('source_quarantine' not in r for r in rows if r['observation']['frame_id']<59)
