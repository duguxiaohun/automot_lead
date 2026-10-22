"""Pure and mixed targets stay separate; transition detection is not correctness."""
from collections import Counter, defaultdict
from . import regions, objective
from .dataset import mass


def evaluate(cases, *, legacy_geometry=False):
    if not cases: raise ValueError('empty_evaluation')
    variant = 'a' if legacy_geometry else cases[0]['record']['variant']; names = regions.names(variant)
    expected_contract = regions.contract(variant)
    bind = objective.observation_binding
    if legacy_geometry:
        from ..geometry_v1 import codec
        from ..geometry_v1.identity import observation_binding
        expected_contract = codec.contract(); bind = observation_binding
    totals=Counter(); hits=Counter(); formats=Counter(); groups=defaultdict(set)
    support=Counter(); correct=Counter(); confusion=[[0]*len(names) for _ in names]
    transitions=defaultdict(Counter); details=[]; history=Counter(); mixed=Counter(); missing=Counter()
    seen=set(); decoding=Counter()
    for case in cases:
        r,pred=case['record'],case['prediction']
        if r['id'] in seen: raise ValueError('duplicate_evaluation_member')
        seen.add(r['id'])
        if (not legacy_geometry and r['variant'] != variant) or pred['contract'] != expected_contract:
            raise ValueError('mixed_evaluation_contract')
        if pred['sample_binding'] != bind(r): raise ValueError('prediction_sample_mismatch')
        ps=pred['token_ids']
        raw_classes=pred.get('raw_class_ids', ps)  # Historical diagnostics lacked this field.
        if len(ps)!=4 or len(pred['format_valid'])!=4 or any(type(p) is not int or not 0<=p<len(names) for p in ps):
            raise ValueError('invalid_four_query_prediction')
        if len(raw_classes)!=4: raise ValueError('invalid_raw_four_query_prediction')
        if not legacy_geometry and any(p not in objective.allowed_indices(r) for p in ps):
            raise ValueError('prediction_outside_event_domain')
        if 'raw_class_ids' in pred:
            for raw_id in raw_classes:
                decoding['queries']+=1
                decoding['non_letter']+=raw_id==-1
                if not legacy_geometry:
                    decoding['letter_outside_event_domain']+=raw_id>=0 and raw_id not in objective.allowed_indices(r)
        pure=[]
        h=r['observation']['history'][-1]; h=h['token_id'] if h['valid'] else None
        for i,(t,p) in enumerate(zip(r['targets'],ps)):
            q=mass(t,len(names)); label=q.index(1.) if 1. in q else None
            pure.append(label)
            if not t['valid']: missing[str(i)]+=1; continue
            keys=['all',f'horizon/{i}',f"event/{r['event']}",f"event_horizon/{r['event']}/{i}",
                  'speed/'+('moving' if r['observation']['speed_mps']>=.1 else 'stationary'),
                  'road/'+r['road_structure']]
            if label is None:
                mixed['count']+=1; mixed['allowed_hit']+=q[p]>0
                mixed['raw_mass_credit']+=(t.get('raw') or q)[p]
                history['mixed_excluded']+=1
                continue
            confusion[label][p]+=1; support[names[label]]+=1; correct[names[label]]+=p==label
            keys += [f'class/{names[label]}',f'horizon_class/{i}/{names[label]}',
                     f"event_horizon_class/{r['event']}/{i}/{names[label]}"]
            for key in keys:
                totals[key]+=1; hits[key]+=p==label; formats[key]+=pred['format_valid'][i] and raw_classes[i]==label
                groups[key].add(r['physical_group'])
            if h is None: history['unknown_history']+=1
            else:
                group='matches' if label==h else 'conflicts'
                history[group]+=1; history[group+'_correct']+=p==label
                history[group+'_copy']+=p==h
        for i,label in enumerate(pure):
            before=h if i==0 else pure[i-1]
            kind='history_to_future' if i==0 else 'adjacent_future'
            if before is None or label is None:
                transitions[kind]['mixed_or_missing_excluded']+=1; continue
            previous_prediction=h if i==0 else ps[i-1]
            changed=before!=label; detected=previous_prediction!=ps[i]
            transitions[kind]['eligible']+=1
            transitions[kind]['true_change' if changed else 'true_stable']+=1
            transitions[kind]['detected_change' if changed else 'false_change']+=detected
            transitions[kind]['both_endpoints_correct']+=(previous_prediction==before and ps[i]==label)
            details.append(dict(id=r['id'],route=r['scenario']+'/'+r['route_id'],frame=r['frame'],query=i,
                kind=kind,true_change=changed,predicted_change=detected,truth=[before,label],
                prediction=[previous_prediction,ps[i]]))
    transition_report={}
    for k,c in transitions.items():
        transition_report[k]=dict(c, recall=c['detected_change']/c['true_change'] if c['true_change'] else None,
            false_transition_rate=c['false_change']/c['true_stable'] if c['true_stable'] else None)
    recall={k:correct[k]/n for k,n in support.items()}
    return dict(variant=variant,legacy_geometry=legacy_geometry,counts=dict(totals),accuracy={k:hits[k]/n for k,n in totals.items()},
        full_vocabulary_accuracy={k:formats[k]/n for k,n in totals.items()},
        class_support=dict(support),class_recall=recall,unsupported_classes=[k for k in names if not support[k]],
        macro_recall=sum(recall.values())/len(recall) if recall else None,
        physical_groups={k:len(v) for k,v in groups.items()},confusion=confusion,
        mixed_targets=dict(mixed),missing_targets=dict(missing),history=dict(history),
        unrestricted_decoding=dict(decoding),
        transitions=transition_report,transition_cases=details,
        interpretation='classification tasks differ; pure targets only in accuracy/macro/transition metrics; mixed reported separately',
        independent_holdout_claim=False)


def intervention(record, mode):
    import copy
    r=copy.deepcopy(record)
    if mode=='history_unknown':
        for h in r['observation']['history']: h.update(valid=False,token_id=-1,name='UNKNOWN')
    elif mode=='history_permuted':
        original=record['observation']['history']
        # Change only tokens, keep target slot duration/reference metadata intact.
        for h,donor in zip(r['observation']['history'],reversed(original)):
            h.update({k:donor[k] for k in ('valid','token_id','name')})
            if r.get('variant','a')!='six' and h['valid']:
                h.update(reference_frame='current',displacement='past_minus_current',
                    reference_speed_mps=r['observation']['speed_mps'],
                    reference_relative_yaw=r['observation']['reference']['relative_yaw'])
    elif mode not in ('observed','image_black'): raise ValueError('unknown_intervention')
    return r


def paired_summary(baseline, changed):
    if [x['record']['id'] for x in baseline] != [x['record']['id'] for x in changed]:
        raise ValueError('intervention_members_or_order_changed')
    count=changed_count=benefit=harm=0
    for left,right in zip(baseline,changed):
        for t,a,b in zip(left['record']['targets'],left['prediction']['token_ids'],right['prediction']['token_ids']):
            if not t['valid']: continue
            q=mass(t,len(regions.names(left['record'].get('variant','a'))))
            if 1. not in q: continue
            y=q.index(1.); count+=1; changed_count+=a!=b; benefit+=a!=y and b==y; harm+=a==y and b!=y
    changed_histories=sum([(h['valid'],h['token_id']) for h in a['record']['observation']['history']] !=
                         [(h['valid'],h['token_id']) for h in b['record']['observation']['history']]
                         for a,b in zip(baseline,changed))
    return dict(pure_endpoints=count,prediction_changes=changed_count,benefit=benefit,harm=harm,
                history_changed_cases=changed_histories,
                causal_interpretation='input sensitivity on frozen validation; not proof that history causes all errors')
