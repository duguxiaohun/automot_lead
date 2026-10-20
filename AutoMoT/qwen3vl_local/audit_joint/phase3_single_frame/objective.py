"""One present image; four parallel A-F distributions, no future teacher forcing."""
import math
import torch
import torch.nn.functional as F
from qwen3vl_local.audit_joint.phase3_base_dense import objectives as old
from qwen3vl_local.audit_joint.phase3_base_dense.data import ACTIONS, LETTERS

INSTRUCTION = (
    'Use the single image observed at t=0, current speed, navigation and the stated event. '
    'A=DECELERATE; B=STOP; C=RESUME; D=LANE_CHANGE_LEFT; E=LANE_CHANGE_RIGHT; F=KEEP. '
    'Choose exactly one allowed letter for each of four future windows. '
    'Each window compresses the native upcoming-action intentions at its two stated origins; '
    'it does NOT describe the action executed throughout a half-second interval. '
    'Keep the current event action domain at all origins, even when later events change. '
    'History contains native labels at past origins, not a plan that must continue. UNKNOWN means unavailable native labeling, not KEEP. '
    'When history is UNKNOWN, use the image, current premises and other known history; do not interpret it as an action. '
    'Use the current observation to decide whether to continue or change; do not assume an event fixes an action sequence. '
    + old.QUERY_TIME_RULES + old.QUERY_SPEED_RULES + ' ' + old.QUERY_LANE_RULES + ' ' + old.PRIMARY_CHOICE_RULES
)


def messages(record, image, *, history=True):
    content = [dict(type='image', image=image), dict(type='text', text=record['premise'])]
    values = record['history'] if history else ['UNKNOWN'] * 4
    items = []
    for i, action in enumerate(values):
        if action not in ACTIONS and action!='UNKNOWN':raise ValueError('Unsupported historical action')
        token = LETTERS[ACTIONS.index(action)] if action in ACTIONS else 'UNKNOWN'
        # A letter on its own line avoids the whitespace-prefixed BPE token.
        # Exact identity is still asserted after the complete processor template.
        items.append(f'({-2+i*.5:g},{-1.5+i*.5:g}]s:\n[HISTORY_ACTION_{i}]\n{token}\n[/HISTORY_ACTION_{i}]')
    content.append(dict(type='text', text='HISTORY: '+record['history_semantics']+'\n'+'; '.join(items)))
    for i in range(4):
        content.append(dict(type='text', text=(
            f'\nFUTURE ({i*.5:g},{(i+1)*.5:g}]s. Native prediction origins: '
            f'+{i*.5+.25:g}s and +{i*.5+.5:g}s. Output one action.\n[ACTION_QUERY_{i}]\n')))
    return [dict(role='system', content=INSTRUCTION), dict(role='user', content=content)]


def check_history_token_ids(tokenizer,input_ids,actions,choices):
    """Assert exact history/output ID identity in the expanded full prompt."""
    if len(actions)!=4:raise ValueError('Exactly four historical slots required')
    values=input_ids.tolist() if hasattr(input_ids,'tolist') else list(input_ids)
    text=tokenizer.decode(values,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids']!=values:raise ValueError('History token alignment round trip changed IDs')
    slots=[]
    for i,action in enumerate(actions):
        begin=f'[HISTORY_ACTION_{i}]\n';end=f'\n[/HISTORY_ACTION_{i}]'
        if text.count(begin)!=1 or text.count(end)!=1:
            raise ValueError('Missing/ambiguous historical token marker')
        first=text.index(begin)+len(begin);last=text.index(end)
        expected=LETTERS[ACTIONS.index(action)] if action in ACTIONS else 'UNKNOWN'
        if action not in ACTIONS and action!='UNKNOWN':raise ValueError('Unsupported historical action')
        if text[first:last]!=expected:raise ValueError('Historical action text changed inside complete prompt')
        positions=[j for j,(a,b) in enumerate(encoded['offset_mapping']) if a<last and b>first]
        ids=[values[j] for j in positions]
        candidate=choices[ACTIONS.index(action)] if action in ACTIONS else None
        if candidate is not None and (ids!=[candidate] or tuple(encoded['offset_mapping'][positions[0]])!=(first,last)):
            raise ValueError(f'History/output action token ID mismatch in slot {i}: {ids} != {[candidate]}')
        slots.append(dict(slot=i,action=action,positions=positions,input_ids=ids,output_id=candidate,
                          same_action_token=candidate is not None))
    return dict(policy='full_prompt_history_output_token_identity_v1',slots=slots,
                existing_vocabulary_only=True,unknown_semantics='missing information, not a target action')


def pack(bundle, record, image, *, history=True, max_length=8192, max_pixels=442368):
    text = bundle.processor.apply_chat_template(messages(record, image, history=history),
                                                tokenize=False, add_generation_prompt=False)
    inputs = bundle.processor(text=[text], images=[image], return_tensors='pt', padding=False,
        images_kwargs={'size': {'shortest_edge': 1024, 'longest_edge': max_pixels}})
    if inputs['input_ids'].shape[1] > max_length:
        raise ValueError('Single-frame input exceeds token budget; no truncation')
    if len(inputs['image_grid_thw']) != 1 or int(inputs['image_grid_thw'][0].prod()) > max_pixels//256+64:
        raise ValueError('Single-frame image budget violated')
    choices=old.choice_ids(bundle.tokenizer)
    token_check=check_history_token_ids(bundle.tokenizer,inputs['input_ids'][0],
        record['history'] if history else ['UNKNOWN']*4,choices)
    return dict(inputs=inputs, positions=old.locate_queries(bundle.tokenizer, inputs['input_ids'][0], 4),
                choices=choices, history_token_check=token_check, windows=record['windows'],
                tokens=inputs['input_ids'].shape[1], images=1)


def tokenizer_admission(bundle,record,image):
    """Probe all six actions in every historical slot, never add training rows."""
    before=len(bundle.tokenizer);checks={}
    for action in (*ACTIONS,'UNKNOWN'):
        view=dict(record,history=[action]*4)
        packed=pack(bundle,view,image)
        checks[action]=packed['history_token_check']
    if len(bundle.tokenizer)!=before:raise ValueError('Action token admission changed vocabulary')
    return dict(status='passed',choices=dict(zip(LETTERS,old.choice_ids(bundle.tokenizer))),
                vocabulary_size=before,checks=checks,synthetic_probe_only=True,
                history_dropout=False,attention_masked_unknown=False)


def loss_from_logits(logits, windows, choices):
    if len(windows) != 4 or logits.shape[0] != 4:
        raise ValueError('Exactly four query readouts required')
    valid = [i for i, w in enumerate(windows) if w['valid']]
    if not valid:
        raise ValueError('No complete future window; never silently train a zero-label anchor')
    q = torch.tensor([windows[i]['target'] for i in valid], device=logits.device, dtype=torch.float32)
    if q.shape != (len(valid), 6) or not torch.isfinite(q).all() or (q < 0).any() or not torch.allclose(q.sum(-1), torch.ones(len(valid), device=q.device)):
        raise ValueError('Invalid soft target')
    # Full-vocabulary normalization penalizes outputs other than the six letters.
    logp = F.log_softmax(logits[valid].float(), dim=-1)[:, choices]
    ce = -(q * logp).sum(-1)
    entropy = -(q * q.clamp_min(1e-30).log()).sum(-1)
    predictions = logits.argmax(-1).tolist()
    hits = sum(predictions[i] in [choices[j] for j, v in enumerate(windows[i]['raw']) if v] for i in valid)
    return ce.mean(), dict(supervised=len(valid), correct=hits, predictions=predictions,
        cross_entropy=ce.mean().item(), target_entropy=entropy.mean().item(),
        kl_to_training_target=(ce-entropy).mean().item(), loss_mass=1.,
        query_weights=[1/len(valid) if i in valid else 0. for i in range(4)])


def loss(bundle, packed):
    logits = old.readout_logits(bundle.model, packed, bundle.device)
    value, report = loss_from_logits(logits, packed['windows'], packed['choices'])
    return value, dict(report, tokens=packed['tokens'], images=1)


def metrics(cases):
    """Raw (unweighted) targets determine quality; preference is training-only."""
    from collections import Counter
    totals, hit, mass_hit, support, recall, changes = (Counter() for _ in range(6))
    fit=Counter();fit_counts=Counter();raw_hits=Counter()
    unknown_baseline = 0
    missing_anchors=Counter()
    paired = dict(windows=0,model_hits=0,repeat_last_hits=0,model_raw_mass=0.,repeat_last_raw_mass=0.,
                  model_only_hits=0,repeat_last_only_hits=0)
    for case in cases:
        native_missing=case.get('history_original',case['history']).count('UNKNOWN')
        input_missing=case.get('history_input',case['history']).count('UNKNOWN')
        missing_anchors[f'native_unknown_slots_{native_missing}']+=1
        missing_anchors[f'input_unknown_slots_{input_missing}']+=1
        last = next((a for a in reversed(case['history']) if a != 'UNKNOWN'), None)
        unknown_baseline += last is None
        previous = None
        for i, (w, pred) in enumerate(zip(case['windows'], case['predictions'])):
            if not w['valid']:
                previous = None
                continue
            j = ACTIONS.index(pred) if pred in ACTIONS else None
            for key in ('all', f'query_{i}', 'event/'+case['event'],
                        f'native_unknown_slots_{native_missing}',f'input_unknown_slots_{input_missing}',
                        'mixed' if len(w['allowed_answers']) > 1 else 'pure',
                        'change_from_history' if last and last not in w['allowed_answers'] else
                        'history_compatible' if last else 'history_unknown'):
                totals[key] += 1
                hit[key] += pred in w['allowed_answers']
                mass_hit[key] += w['raw'][j] if j is not None else 0
                if 'raw_predictions' in case:raw_hits[key] += case['raw_predictions'][i] in w['allowed_answers']
                if 'probabilities' in case:
                    probabilities=case['probabilities'][i]
                    ce=-sum(q*math.log(max(p,1e-30)) for q,p in zip(w['raw'],probabilities))
                    entropy=-sum(q*math.log(q) for q in w['raw'] if q)
                    fit[key+'/raw_cross_entropy']+=ce
                    fit[key+'/raw_KL']+=ce-entropy
                    fit[key+'/target_entropy']+=entropy
                    fit[key+'/six_plus_other_brier']+=sum((p-q)**2 for p,q in zip(probabilities,w['raw']))+(1-sum(probabilities))**2
                    fit_counts[key]+=1
            for a in w['allowed_answers']:
                support[a] += 1
                recall[a] += pred == a
            if last:
                totals['repeat_last'] += 1
                hit['repeat_last'] += last in w['allowed_answers']
                model_hit=pred in w['allowed_answers'];baseline_hit=last in w['allowed_answers']
                paired['windows']+=1;paired['model_hits']+=model_hit;paired['repeat_last_hits']+=baseline_hit
                paired['model_raw_mass']+=w['raw'][j] if j is not None else 0.
                paired['repeat_last_raw_mass']+=w['raw'][ACTIONS.index(last)]
                paired['model_only_hits']+=model_hit and not baseline_hit
                paired['repeat_last_only_hits']+=baseline_hit and not model_hit
            if previous is not None:
                changes['predicted_switches'] += pred != previous
                changes['adjacent_valid_pairs'] += 1
            previous = pred
    n=paired['windows']
    paired.update(model_set_accuracy=paired['model_hits']/n if n else None,
                  repeat_last_set_accuracy=paired['repeat_last_hits']/n if n else None,
                  set_accuracy_delta=(paired['model_hits']-paired['repeat_last_hits'])/n if n else None,
                  model_raw_mass_score=paired['model_raw_mass']/n if n else None,
                  repeat_last_raw_mass_score=paired['repeat_last_raw_mass']/n if n else None,
                  scope='identical valid windows on anchors with at least one known filled history action')
    return dict(natural_missingness=dict(anchor_counts=dict(missing_anchors),artificial_training_masking=False,
                    interpretation='Observed missingness strata, not random noise or proof of robustness improvement'),
                paired_repeat_last=paired,total=dict(totals), set_accuracy={k:hit[k]/n for k,n in totals.items()},
                raw_mass_score={k:mass_hit[k]/n for k,n in totals.items() if k!='repeat_last'},
                action_support=dict(support), action_recall={a:recall[a]/n for a,n in support.items()},
                prediction_switches=dict(changes), repeat_last_unknown_anchors=unknown_baseline,
                distribution_fit={k:v/fit_counts[k.rsplit('/',1)[0]] for k,v in fit.items()},
                unconstrained_set_accuracy={k:raw_hits[k]/n for k,n in fit_counts.items()},
                output_policy='one event-allowed A-F action per window; full-vocabulary argmax audited separately',
                caveat='Single output loses within-window ordering and may omit short actions; not historical base accuracy')


def paired_comparison(reference,alternative):
    """Compare two inference interventions on identical IDs and target windows."""
    if [r['id'] for r in reference]!=[r['id'] for r in alternative]:
        raise ValueError('Paired evaluation anchor order differs')
    n=left=right=0;lm=rm=0.;changed_n=changed_left=changed_right=0
    for a,b in zip(reference,alternative):
        if a['windows']!=b['windows']:raise ValueError('Paired evaluation targets differ')
        changed=any(a.get('history_changed',[]))
        for w,p,q in zip(a['windows'],a['predictions'],b['predictions']):
            if not w['valid']:continue
            x=p in w['allowed_answers'];y=q in w['allowed_answers']
            n+=1;left+=x;right+=y
            lm+=w['raw'][ACTIONS.index(p)] if p in ACTIONS else 0.
            rm+=w['raw'][ACTIONS.index(q)] if q in ACTIONS else 0.
            if changed:changed_n+=1;changed_left+=x;changed_right+=y
    return dict(windows=n,reference_hits=left,alternative_hits=right,
        reference_set_accuracy=left/n if n else None,alternative_set_accuracy=right/n if n else None,
        set_accuracy_delta=(left-right)/n if n else None,
        raw_mass_score_delta=(lm-rm)/n if n else None,
        imputed_history_anchor_subset=dict(windows=changed_n,reference_hits=changed_left,alternative_hits=changed_right,
            set_accuracy_delta=(changed_left-changed_right)/changed_n if changed_n else None),
        interpretation='Same checkpoint input intervention; not a retraining experiment or causal proof')
