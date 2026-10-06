"""Scheme B: bounded transition triage -> explicit causal RGB review -> annotations.

The queue is not supervision. Reviewed facts use the existing frozen rubric;
no automatic suggestion is copied into labels. Development review cannot use
reserved validation/test routes. Rendered packets mark camera seams and show
only the current anchor and its causal history.
"""
import argparse
import json
from collections import defaultdict,Counter
from pathlib import Path
from .identity import ROOT,digest,file_sha,write_json

POLICY='phase4_transition_review_queue_v2'


def card(root,scenario,route_id,frame,episode,edge, *, expected_split='train'):
    from .dataset import source_route,groups,split_for,holdout_reservations
    from .controller import Episode
    from .route_context import episode_edge
    from .taxonomy import applicable
    from .risk_review import registry,manual_risks
    row=dict(scenario=scenario,route_id=route_id);group=source_route(row)
    actual_split=split_for(group,groups(),reservations=holdout_reservations())
    if expected_split not in ('train','val','test') or actual_split!=expected_split:
        raise ValueError('review cannot cross frozen route split')
    if expected_split!='train':
        from .dataset import producer_check_reservations
        if group in producer_check_reservations():
            raise ValueError('reserved producer accuracy routes cannot supply model evaluation cards')
    ep=Episode(**episode);transition=episode_edge(ep,edge)
    if not applicable(transition,ep.state,ep.longitudinal):raise ValueError('inapplicable review state pair')
    if type(frame) is not int or frame<10:raise ValueError('review requires complete 2/4 RGB history')
    images=[];root=Path(root).resolve()
    for f in range(frame-6,frame+1):
        path=Path(scenario)/route_id/'rgb'/f'{f:04d}.jpg';absolute=(root/path).resolve()
        if not absolute.is_relative_to(root):raise ValueError('review RGB escapes root')
        images.append(dict(frame_id=f,path=str(path),sha256=file_sha(absolute)))
    body=dict(**row,frame_id=frame,episode=episode,edge=edge,criteria=list(transition.criteria),images=images,
              risk_ids=[r['risk_id'] for r in manual_risks(registry().get((scenario,route_id),[]),
                                                        [i['frame_id'] for i in images])])
    if expected_split!='train':body['review_split']=expected_split
    return dict(body,id=digest(body))


def select(items,root, *, per_event=16, split='train'):
    """Stratified, route-diverse transition triage; suggestions stay blinded.

    Store at most per_event route anchors per observed stratum. The budget is
    per event, not per stratum. Report omitted strata rather than claim a
    representative probability sample when the requested budget is too small.
    """
    import re
    if split not in ('train','val','test'):raise ValueError('invalid review split')
    from .dataset import producer_check_reservations
    reserved=producer_check_reservations() if split!='train' else {}
    if type(per_event) is not int or per_event<1:raise ValueError('positive review budget required')
    tracks={};pools=defaultdict(dict);raw=0;anchors=0;last_route=None;offered=Counter()
    def offer(row,p,value,previous):
        nonlocal anchors
        event=p['instance']['event'];town=re.search(r'Town\d+(?:HD)?',row['route_id'])
        town=town.group() if town else 'unknown'
        answer='YES' if value is True else 'NO' if value is False else 'UNKNOWN'
        stratum=(event,row['scenario'],town,p['edge'],answer)
        group=row.get('physical_group') or (row['scenario'],row['route_id'])
        key=(row['scenario'],row['route_id'],row['frame_id'],event,p['edge'],p['instance']['instance_id'])
        rank=0 if previous is not None and value!=previous else 1
        anchors+=1;offered[stratum]+=1;pool=pools[stratum];order=(rank,digest(key))
        if group not in pool or order<pool[group][0]:pool[group]=(order,row,p,group)
        if len(pool)>per_event:del pool[max(pool,key=lambda k:pool[k][0])]
    def flush():
        for row,p,value in tracks.values():offer(row,p,value,value)
        tracks.clear()
    for row in items:
        if row.get('split')!=split or row.get('physical_group') in reserved:continue
        route=(row['scenario'],row['route_id'])
        if last_route is not None and route!=last_route:flush()
        last_route=route
        for p in row.get('proposals',[])+row.get('identity_review_hints',[]):
            if row['frame_id']<10:continue
            raw+=1;suggestion=p['proposal'].get('release_suggestion')
            value=suggestion.get('value') if suggestion else None
            if p['edge']!='proceed':value={'YES':True,'NO':False}.get(p['proposal'].get('geometric_target'))
            key=(p['instance']['instance_id'],p['edge']);previous=tracks.get(key)
            if previous is None or value!=previous[2] or row['frame_id']!=previous[0]['frame_id']+1:
                offer(row,p,value,previous[2] if previous else None)
            tracks[key]=(row,p,value)
    flush();cards=[];counts={};selected=Counter();selected_routes=Counter()
    for event in sorted({s[0] for s in pools}):
        options=[(s,item) for s,pool in pools.items() if s[0]==event for item in pool.values()]
        used=Counter();dimensions=[Counter() for _ in range(4)];chosen=[]
        def score(option):
            s,(order,row,p,group)=option
            marginal=[dimensions[i][s[i+1]] for i in range(4)]
            return (used[group],max(marginal),sum(marginal),selected[s],order)
        while options and len(chosen)<per_event:
            option=min(options,key=score);options.remove(option);stratum,(_,row,p,group)=option
            args=(root,row['scenario'],row['route_id'],row['frame_id'],p['episode'],p['edge'])
            chosen.append(card(*args) if split=='train' else card(*args,expected_split=split))
            selected[stratum]+=1;used[group]+=1
            for i in range(4):dimensions[i][stratum[i+1]]+=1
        counts[event]=len(chosen);selected_routes[event]=len(used);cards+=chosen
    # Reviewers receive no per-card rule answer, ranking, or suggested facts.
    cards.sort(key=lambda c:digest(c))
    strata=[dict(event=s[0],scenario=s[1],town=s[2],edge=s[3],suggestion=s[4],
                 anchor_offers=offered[s],selected=selected[s]) for s in sorted(offered)]
    return dict(policy=POLICY,split=split,purpose='development' if split=='train' else 'independent_model_evaluation_pending_protocol_review',cards=cards,statistics=dict(raw_questions=raw,
        anchor_offers=anchors,selected=len(cards),by_event=counts,physical_routes_by_event=dict(selected_routes),
        strata=strata,unselected_strata=sum(x['selected']==0 for x in strata),
        sampling='deterministic diversity triage, not a probability sample or accuracy estimate',
        queue_storage_bound='one route tracker plus per_event anchors per observed stratum'),supervision_approved=False)



def compile_reviews(queue,review,root):
    from .controller import Episode
    from .route_context import episode_edge
    from .route_calibration import reviewed_band_labels
    from .risk_review import POLICY as RISK_POLICY,CHECKS
    from .visible_scope import POLICY as VISIBLE_POLICY
    from .maneuver_safety import GUARDED
    if queue.get('split','train')!='train' or any(c.get('review_split','train')!='train' for c in queue['cards']):
        raise ValueError('independent evaluation packets require an independent reviewed evaluation protocol; development compiler cannot admit them')
    if (queue.get('policy')!=POLICY or review.get('policy')!=POLICY
            or review.get('queue_sha256')!=digest(queue)
            or not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip()):
        raise ValueError('review must bind queue and reviewer')
    lookup={c['id']:c for c in queue['cards']};seen=set();annotations=[]
    for d in review['decisions']:
        ident=d['card_id']
        if ident in seen or ident not in lookup:raise ValueError('duplicate or unknown review card')
        seen.add(ident);c=lookup[ident]
        if c!=card(root,c['scenario'],c['route_id'],c['frame_id'],c['episode'],c['edge']):
            raise ValueError('stale review RGB/state/risk identity')
        if d.get('status') in ('pending','rejected','uncertain'):continue
        if d.get('status') not in ('accepted','deferred'):raise ValueError('invalid review disposition')
        if any(d.get(k) is not True for k in ('event_identity_reviewed','state_pair_reviewed','local_scope_and_priority_reviewed')):
            raise ValueError('explicit event/state/scope review required')
        if d.get('risk_ids_reviewed')!=c['risk_ids']:raise ValueError('review current risk IDs explicitly')
        observations=d.get('rgb_reviews',{})
        if set(observations)!={str(i['frame_id']) for i in c['images']}:raise ValueError('review every causal RGB')
        for i in c['images']:
            item=observations[str(i['frame_id'])]
            if (item.get('rgb_sha256')!=i['sha256'] or item.get('observed_until')!=i['frame_id']
                    or item.get('usable') is not True or not isinstance(item.get('observation'),str) or not item['observation'].strip()):
                raise ValueError('missing/uncertain/noncausal RGB review')
            if c['risk_ids'] and any(item.get(k) is not True for k in CHECKS):
                raise ValueError('explicit per-frame risk checks required')
        f=c['frame_id'];ep=Episode(**c['episode']);edge=episode_edge(ep,c['edge']);facts=d.get('facts')
        if (not isinstance(facts,dict) or set(facts)!=set(edge.criteria)
                or any(v is not None and type(v) is not bool for v in facts.values())):
            raise ValueError('review each criterion explicitly; unresolved criteria stay unknown')
        target=d.get('target')
        if target not in ('YES','NO','UNKNOWN') or (target=='UNKNOWN')!=(d['status']=='deferred'):
            raise ValueError('binary approval or explicit deferred UNKNOWN required')
        milestone=edge.kind=='milestone' and target=='YES'
        if milestone and d.get('milestone_observed') is not True:raise ValueError('completion needs current milestone review')
        phase=d.get('phase')
        if phase not in ('readiness','catchup','uncertain') or (phase=='uncertain')!=(target=='UNKNOWN'):
            raise ValueError('review readiness versus old-state catchup or deferred uncertainty explicitly')
        from .visual_review import validate_all,check_metric_sources
        validate_all(d.get('visual_review'),c['images'],f,ep.event,c['edge'],phase,target)
        check_metric_sources(c,d['visual_review'],root,phase,target)
        if phase=='catchup' and (d.get('successor_confirmed') is not True or type(d.get('current_conflict')) is not bool
                                 or type(d.get('transition_blocking_conflict')) is not bool):
            raise ValueError('catchup requires successor and transition-conflict review')
        note=observations[str(f)]['observation'];evidence='semi_review/'+digest([digest(queue),d,review['reviewer']])
        band=dict(review_start=f,review_end=f,reference_frame=f if milestone or phase=='catchup' else None,
                  reference_kind='milestone_observed' if milestone else 'successor_observed' if phase=='catchup' else 'not_observed',
                  reference_observation=note,start_reason='Individually reviewed causal anchor; no label window inferred.',
                  stop=dict(frame=f+1,kind='review_end',reason='Only this anchor approved; adjacent anchors require separate review.'),
                  frames={str(f):dict(phase=phase,observation=note,observed_until=f,facts=facts)})
        if phase=='catchup':
            band['frames'][str(f)].update({k:d[k] for k in ('successor_confirmed','current_conflict','transition_blocking_conflict')})
        labels=reviewed_band_labels(ep,c['edge'],band)
        if labels[f][0]!=target:raise ValueError('reviewed facts and target disagree')
        ann=dict(scenario=c['scenario'],route_id=c['route_id'],episode=c['episode'],edge=c['edge'],
                 label_basis='reviewed_transition_band',context_valid=True,reviewer=review['reviewer'],evidence_id=evidence,
                 observation=note,transition_band=band,frame_sha256={str(f):c['images'][-1]['sha256']},
                 visual_review=d['visual_review'])
        if c['risk_ids']:
            ann['risk_review']=dict(policy=RISK_POLICY,reviewer=review['reviewer'],evidence_id=evidence,
                 risk_ids=c['risk_ids'],frames={str(i['frame_id']):dict(status='usable',reason=observations[str(i['frame_id'])]['observation'],
                    observed_until=i['frame_id'],rgb_sha256=i['sha256'],**{k:observations[str(i['frame_id'])][k] for k in CHECKS}) for i in c['images']})
        if c['edge'] in GUARDED:
            if d.get('visible_maneuver_scope_reviewed') is not True:raise ValueError('fresh visible-scope review required')
            ann['visible_scope_review']=dict(policy=VISIBLE_POLICY,reviewer=review['reviewer'],evidence_id=evidence,
                 frames={str(i['frame_id']):dict(rgb_sha256=i['sha256'],observed_until=i['frame_id'],visible_conditions_reviewed=True,
                    observation=observations[str(i['frame_id'])]['observation'],target=target if i['frame_id']==f else None) for i in c['images']})
        annotations.append(ann)
    return annotations


def render(queue,root,output):
    from PIL import Image,ImageDraw
    output=Path(output)
    if output.exists():raise FileExistsError('new review panel directory required')
    output.mkdir(parents=True)
    for c in queue['cards']:
        if c!=card(root,c['scenario'],c['route_id'],c['frame_id'],c['episode'],c['edge'],expected_split=c.get('review_split','train')):
            raise ValueError('stale or invalid render card')
        from .observation import observation_contract
        for mode in (2,4):
            allowed={c['frame_id']+o for o in observation_contract(mode)['frame_offsets']}
            images=[i for i in c['images'] if i['frame_id'] in allowed]
            panel=Image.new('RGB',(1152,404*len(images)),'white');draw=ImageDraw.Draw(panel)
            for n,i in enumerate(images):
                path=Path(root)/i['path']
                if file_sha(path)!=i['sha256']:raise ValueError('stale RGB while rendering')
                with Image.open(path) as im:
                    if im.size!=(1152,384):raise ValueError('camera layout needs explicit calibration')
                    panel.paste(im,(0,n*404+20))
                draw.text((4,n*404),f"mode {mode} | frame {i['frame_id']} | left / front / right | anchor {c['frame_id']}",fill='black')
                for x in (384,768):draw.line((x,n*404+20,x,n*404+403),fill='red',width=2)
            panel.save(output/f"{c['id']}_{mode}rgb.jpg")



def blank_reviews(queue):
    from .visual_review import template
    return dict(policy=POLICY,queue_sha256=digest(queue),reviewer='',decisions=[dict(
        card_id=c['id'],status='pending',phase=None,target=None,event_identity_reviewed=False,state_pair_reviewed=False,
        local_scope_and_priority_reviewed=False,visible_maneuver_scope_reviewed=False,milestone_observed=False,
        visual_review=template(c['images'],c['frame_id']),
        risk_ids_reviewed=[],facts={k:None for k in c['criteria']},rgb_reviews={str(i['frame_id']):dict(
        rgb_sha256=i['sha256'],observed_until=i['frame_id'],usable=False,observation='',
        participant_identity_reviewed=False,transition_scope_reviewed=False,instance_boundary_reviewed=False)
        for i in c['images']}) for c in queue['cards']])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--proposals',type=Path);p.add_argument('--queue',type=Path)
    p.add_argument('--requests',type=Path,help='explicit scenario/route/frame/episode/edge review requests, including RE2/RE3 and bypass targets');p.add_argument('--template',type=Path);p.add_argument('--reviews',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data');p.add_argument('--per-event',type=int,default=16)
    p.add_argument('--panels',type=Path);p.add_argument('--split',choices=('train','val','test'),default='train');a=p.parse_args()
    if a.output.exists():raise FileExistsError('new output required')
    if a.template and a.template.exists():raise FileExistsError('new review template required')
    if a.proposals and a.requests:p.error('choose proposals or explicit requests')
    if a.reviews:
        if not a.queue:p.error('--queue required with --reviews')
        q=json.loads(a.queue.read_text());write_json(a.output,compile_reviews(q,json.loads(a.reviews.read_text()),a.data_root))
    elif a.requests:
        requests=json.loads(a.requests.read_text());cards=[card(a.data_root,r['scenario'],r['route_id'],r['frame_id'],r['episode'],r['edge'],expected_split=a.split) for r in requests]
        if len({c['id'] for c in cards})!=len(cards):raise ValueError('duplicate explicit request')
        q=dict(policy=POLICY,split=a.split,cards=cards,selection='explicit current-state review requests',supervision_approved=False)
        write_json(a.output,q)
        if a.panels:render(q,a.data_root,a.panels)
    else:
        if not a.proposals:p.error('--proposals required for triage')
        from .privileged_producer import producer_identity
        with a.proposals.open() as f:
            header=json.loads(next(f))
            if header.get('producer')!=producer_identity():raise ValueError('stale proposal source; regenerate')
            q=select((json.loads(line) for line in f if line.strip()),a.data_root,per_event=a.per_event,split=a.split)
        q['proposal_sha256']=file_sha(a.proposals);write_json(a.output,q)
        if a.panels:render(q,a.data_root,a.panels)
    if a.template:
        if a.reviews:p.error('template belongs to queue preparation, not compilation')
        write_json(a.template,blank_reviews(q))

if __name__=='__main__':main()
