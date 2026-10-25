"""Recompute full-case continuous errors and publish one six-group audit ZIP."""
from collections import defaultdict,Counter
import json
from pathlib import Path
import numpy as np
from qwen3vl_local.audit_joint.abc_delivery import sha,write,pack,digest
from . import data,contracts

GROUPS=tuple(f'{v}_{b}' for v in 'abc' for b in ('bev_only','qwen_simple'))


def region_support(rows):
    counts=Counter();groups=defaultdict(set);present=defaultdict(set)
    for split,records in rows.items():
        for r in records:
            present[split].add(r['event'])
            for v in 'abc':
                for h,t in enumerate(r['representations'][v]['targets']):
                    key=(split,r['event'],v,h,t['token_id'])
                    counts[key]+=1;groups[key].add(r['physical_group'])
    return {'/'.join(map(str,key)):dict(anchors=counts[key],physical_groups=len(groups.get(key,())),
                sparse_group_support=len(groups.get(key,()))<5)
            for split,events in present.items() for e in sorted(events) for v in 'abc' for h in range(4)
            for c in range(9) for key in [(split,e,v,h,c)]}


def verify_training(run,manifest,projection,variant,branch,*,manifest_sha256):
    """Verify actual seven-epoch consumption, checkpoint content and identity."""
    import torch
    from types import SimpleNamespace
    from .worker import numerical_config
    run=Path(run);complete=json.loads((run/'training_complete.json').read_text())
    from .preflight import plan_digest
    if plan_digest(projection)!=projection['sha256'] or projection['total_updates']!=12719 or projection['total_presentations']!=813792:
        raise ValueError('completed_plan_changed')
    binding=complete['binding'];config=json.loads((run/'config.json').read_text())
    expected=dict(representation=contracts.contract(variant),branch=branch,source=contracts.source_identity(),
                  consumption_sources=dict(manifest_sha256=manifest_sha256,protocol='same_hashed_bytes_decode_v1'),
                  dataset=manifest['identity'],splits=manifest['splits'],plan=projection['sha256'],
                  numerical_config=numerical_config(SimpleNamespace(**config)))
    if any(binding.get(k)!=v for k,v in expected.items()):raise ValueError('completed_training_binding_changed')
    if json.loads((run/'condition_contract.json').read_text())!=binding:raise ValueError('saved_condition_contract_changed')
    if complete['step']!=12719 or len(projection['plans'])!=7:raise ValueError('incomplete_training_budget')
    for epoch,plan in enumerate(projection['plans'],1):
        if plan['samples']!=116256 or plan['optimizer_steps']!=1817 or plan['quotas']!=contracts.QUOTAS:
            raise ValueError('completed_epoch_budget_changed')
        paths=list((run/'epoch_audit').glob(f'epoch_{epoch:03d}_*.json'))
        if len(paths)!=1 or complete['epochs'].get(str(epoch))!=sha(paths[0]):raise ValueError('epoch_consumption_receipt_changed')
        values=json.loads(paths[0].read_text())
        actual={k.removeprefix('exposure/'):v for k,v in values.items() if k.startswith('exposure/')}
        if actual!=dict(Counter(plan['ordered_ids'])):raise ValueError('completed_actual_exposure_changed')
        verified={k.removeprefix('verified_exposure/'):v for k,v in values.items() if k.startswith('verified_exposure/')}
        if verified!=actual:raise ValueError('completed_source_verification_exposure_changed')
    files={p.name:sha(p) for p in (run/'token_diagnostics').glob('*.json')}
    if not files or files!=complete.get('token_diagnostics'):raise ValueError('token_diagnostic_files_changed')
    for rank in range(4):
        proof=json.loads((run/'token_diagnostics'/f'step_00012719_rank{rank}.json').read_text())
        if (proof['rank']!=rank or proof['optimizer_steps']!=12719 or proof['slot_exposure']['samples']!=813792//4
            or np.asarray(proof['gradient_norms']).shape!=(10,) or not np.isfinite(proof['gradient_norms']).all()):
            raise ValueError('completed_token_gradient_evidence_missing')
        for role in ('future','history'):
            if sum(v for k,v in proof['slot_exposure'].items() if k.startswith(role+'/'))!=813792:
                raise ValueError('completed_token_slot_exposure_changed')
    for name in ('latest','best'):
        path=run/(name+'.pt')
        if complete[name+'_sha256']!=sha(path):raise ValueError('completed_checkpoint_changed')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['schema']!='abc_trajectory_checkpoint_v1' or state['condition_contract']!=binding:
            raise ValueError('completed_checkpoint_content_changed')
        if not 0<state['step']<=12719:raise ValueError('checkpoint_step_outside_budget')
        if name=='latest':
            from qwen3vl_local.action_prior.training_core import budget_complete
            if not budget_complete(state['step'],{'actual_step_limit':12719,'samples_per_epoch':116256},state['cursor']):
                raise ValueError('latest_checkpoint_budget_incomplete')
        del state
    return complete


def members_archive(rows,output):
    """Reconstruct IDs from route/frame/event; never store millions of random hashes."""
    from qwen3vl_local.audit_joint.geometry_v1.dataset import digest as native_digest
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    routes=sorted({(r['scenario'],r['route_id'],r['physical_group']) for values in rows.values() for r in values})
    route_ids={r:i for i,r in enumerate(routes)};events=sorted({r['event'] for values in rows.values() for r in values})
    event_ids={e:i for i,e in enumerate(events)};indices=[];labels=[];lookup={}
    for split_index,split in enumerate(('train','val','test')):
        for r in rows[split]:
            if r['id']!=native_digest([r['scenario'],r['route_id'],r['frame'],r['event']]):
                raise ValueError('member_id_not_reconstructible')
            if r['id'] in lookup:raise ValueError('duplicate_member_id')
            lookup[r['id']]=len(indices)
            indices.append((route_ids[r['scenario'],r['route_id'],r['physical_group']],r['frame'],event_ids[r['event']],split_index))
            labels.append([[t['token_id'] for t in r['representations'][v]['targets']] for v in 'abc'])
    np.savez_compressed(output/'members.npz',indices=np.asarray(indices,dtype=np.int32),regions=np.asarray(labels,dtype=np.uint8))
    write(output/'membership.json',dict(routes=routes,events=events,splits=['train','val','test'],
        columns=['route_index','frame','event_index','split_index'],variants=['a','b','c'],members=len(indices),
        id_recipe='sha256(json.dumps([scenario,route_id,frame,event], sort_keys=True, allow_nan=False).encode())',
        numeric_storage='exact integers; future class ids 0..8'))
    return lookup,{'members.npz':output/'members.npz','membership.json':output/'membership.json'}


def error_archive(path,indices,values,*,predicted_regions=None,gt_zero_height_regions=None):
    values=np.asarray(values,dtype=np.float64)
    reduced=values.astype(np.float32)
    # Native trajectory metrics are normally exact float32 values. Use float64
    # when round-trip equality fails; no precision rounding to fit the package.
    stored=reduced if np.array_equal(reduced.astype(np.float64),values) else values
    extra={}
    if predicted_regions is not None:
        extra=dict(predicted_regions=np.asarray(predicted_regions,dtype=np.uint8),
                   gt_zero_height_regions=np.asarray(gt_zero_height_regions,dtype=np.uint8))
    np.savez_compressed(path,member_index=np.asarray(indices,dtype=np.int32),errors=stored,**extra)


def drill(directory,output):
    """Actual cohort cardinality with noncompressible floating-point errors."""
    directory=Path(directory);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows,manifest=data.load(directory)
    lookup,required=members_archive(rows,output)
    write(output/'region_support.json',region_support(rows))
    required['data/region_support.json']=output/'region_support.json'
    required.update({'data/manifest.json':directory/'manifest.json','data/split_report.json':directory/'split_report.json'})
    rng=np.random.default_rng(2026)
    from .diagnostic_coverage import select,labels
    from .diagnostics import summarize
    selections={v:select(rows['val'],v) for v in 'abc'}
    for group in GROUPS:
        path=output/(group+'.npz')
        error_archive(path,[lookup[r['id']] for r in rows['test']],rng.uniform(0,30,(len(rows['test']),4)),
                      predicted_regions=rng.integers(0,9,(len(rows['test']),4)),
                      gt_zero_height_regions=rng.integers(0,9,(len(rows['test']),4)))
        required[group+'/cases.npz']=path
        variant=group.split('_',1)[0];selected,coverage=selections[variant];cases=[]
        for r in selected:
            cases.append(dict(id=r['id'],event=r['event'],physical_group=r['physical_group'],
                labels=labels(r,variant,'future'),history_labels=labels(r,variant,'history'),
                slot_responses={role:rng.uniform(0,1,4).tolist() for role in ('future','history')},
                results={mode:dict(mean_waypoint_shift_from_oracle_m=float(rng.uniform()),
                    route_ade_m=float(rng.uniform()),waypoint_ade_m=float(rng.uniform()))
                    for mode in ('oracle','replace_regions','permute_horizons','no_future')}))
        diagnostic=output/(group+'_condition_sensitivity.json')
        write(diagnostic,dict(synthetic=True,coverage=coverage,cases=cases,summary=summarize(cases,coverage)))
        required[group+'/condition_sensitivity.json']=diagnostic
    # Leave 4 MB for actual receipts, token diagnostics and per-file SHA manifests.
    receipt=pack(output/'delivery'/'pack_drill.zip',required,limit=26_000_000,
                 summary=dict(synthetic_errors=True,test_cases=len(rows['test']),real_members=True))
    write(output/'report.json',dict(receipt,dataset=manifest['identity'],splits=manifest['splits'],
        package_ready=True,synthetic_model_results=True,reserved_bytes=4_000_000))
    return receipt


def trajectory_errors(case):
    errors={}
    for kind,n in (('route',10),('waypoints',8)):
        gt=np.asarray(case['gt_'+kind],dtype=np.float64);pred=np.asarray(case['pred_'+kind],dtype=np.float64)
        if gt.shape!=(1,n,2) or pred.shape!=gt.shape or not np.isfinite(gt).all() or not np.isfinite(pred).all():
            raise ValueError('invalid_full_trajectory')
        errors[kind]=np.linalg.norm(gt-pred,axis=-1)[0].tolist()
    for key,value in dict(route_ade_m=np.mean(errors['route']),route_fde_m=errors['route'][-1],
                          waypoint_ade_m=np.mean(errors['waypoints']),waypoint_fde_m=errors['waypoints'][-1]).items():
        if not np.isclose(case['metrics'][key],value,atol=1e-4,rtol=1e-5):raise ValueError('case_metric_not_reproducible: '+key)
    return errors


def motion_stratum(record):
    wp=np.asarray(record['waypoint_xy']);radius=np.linalg.norm(wp,axis=1)
    if record['observation']['speed_mps']<=.2:
        return 'low_current_speed_future_motion' if radius.max()>.5 else 'low_current_speed_small_displacement'
    return 'moving_current_low_final_segment_speed' if np.linalg.norm(wp[-1]-wp[-2])/.25<=.2 else 'other_moving'


def reclassified_regions(case,record,variant):
    from qwen3vl_local.audit_joint.geometry_v1 import codec
    from qwen3vl_local.audit_joint.geometry_abc_v2 import regions
    projection=case['geometry_projection'];matrix=np.asarray(projection['rotation_world_xy'],dtype=np.float64)
    xyz=np.asarray(projection['gt_endpoint_xyz'],dtype=np.float64)
    if matrix.shape!=(2,3) or xyz.shape!=(4,3) or not np.isfinite(matrix).all() or not np.isfinite(xyz).all():
        raise ValueError('invalid_endpoint_projection')
    indices=[1,3,5,7]
    if not np.allclose(xyz[:,:2],np.asarray(case['gt_waypoints'])[0,indices],rtol=1e-6,atol=1e-6):
        raise ValueError('native_ticks_do_not_align_with_trajectory_times')
    obs=record['observation']
    def classify(points):
        world=np.c_[points,np.zeros(4)]@matrix.T
        return [regions.relabel(codec.classify(p,t,obs['speed_mps'],obs['reference']['yaw']),variant,record['road_structure'])['token_id']
                for p,t in zip(world,(.5,1.,1.5,2.))]
    return classify(np.asarray(case['pred_waypoints'])[0,indices]),classify(xyz[:,:2])


def summarize(rows):
    def mean(values):
        r=np.asarray([x['route_errors'] for x in values]);w=np.asarray([x['waypoint_errors'] for x in values])
        return dict(cases=len(values),route_ADE=float(r.mean()),route_FDE=float(r[:,-1].mean()),
                    waypoint_ADE=float(w.mean()),waypoint_FDE=float(w[:,-1].mean()),waypoint_by_time=w.mean(0).tolist())
    groups=defaultdict(list);events=defaultdict(list)
    for r in rows:groups[r['physical_group']].append(r);events[r['event']].append(r)
    event={e:dict(mean(v),physical_groups=len({r['physical_group'] for r in v})) for e,v in events.items()}
    physical={g:mean(v) for g,v in groups.items()}
    result=dict(all=mean(rows),events=event,physical_groups=len(groups))
    strata=defaultdict(list)
    for row in rows:strata[row['motion_stratum']].append(row)
    result['motion_strata']={key:mean(values) for key,values in strata.items()}
    result['motion_strata_definition']=dict(current_speed_threshold_mps=.2,max_future_displacement_threshold_m=.5,
        final_segment_speed_threshold_mps=.2,labels_from_GT_only=True)
    predicted=np.asarray([r['predicted_regions'] for r in rows]);truth=np.asarray([r['region_ids'] for r in rows])
    projected=np.asarray([r['gt_zero_height_regions'] for r in rows])
    result['region_consistency']=dict(predicted_vs_oracle_by_horizon=(predicted==truth).mean(0).tolist(),
        all_four=float((predicted==truth).all(1).mean()),GT_zero_height_vs_oracle_by_horizon=(projected==truth).mean(0).tolist(),
        projection='current full rotation; predicted ego z=0; true v2 uses recorded xyz; GT projection gap reported separately')
    for name,values in [('all',[r['inference']['forward_seconds'] for r in rows]),
                        ('excluding_first_per_rank',[r['inference']['forward_seconds'] for r in rows if not r['inference']['first_eval_forward']])]:
        result['inference_'+name]=dict(cases=len(values),mean_s=float(np.mean(values)),p50_s=float(np.quantile(values,.5)),
            p95_s=float(np.quantile(values,.95)),scope='synchronized current RGB/LiDAR preprocessing + frozen encoders + FM; excludes file IO') if values else None
    for name,values in (('event_macro',[v for k,v in event.items() if k!='REGULAR_BACKGROUND']),('group_macro',list(physical.values()))):
        result[name]={k:sum(v[k] for v in values)/len(values) for k in ('route_ADE','route_FDE','waypoint_ADE','waypoint_FDE')}
    return result


def build(root,directory):
    root=Path(root);rows,manifest=data.load(directory);expected={r['id']:r for r in rows['test']}
    projection=json.loads((Path(directory)/'plan.json').read_text())
    from .probe_inputs import verify_processor,selection,verify_probe_inputs
    gate=json.loads((root/'admission.json').read_text());processor=verify_processor(root/'processor.json',gate)
    flat=[r for records in rows.values() for r in records]
    probe_selections={v:selection(flat,processor,v) for v in 'abc'};del flat
    from .consumption import Sources
    manifest_sha256=sha(Path(directory)/'manifest.json')
    if json.loads((root/'admission.json').read_text())['manifest_sha256']!=manifest_sha256:
        raise ValueError('admitted_source_manifest_changed')
    sources=Sources(directory,manifest['data_root'],manifest['identity'],manifest_sha256)
    lookup,required=members_archive(rows,root/'audit');summary={};gt_identity={};quality_reviews=[]
    support_path=root/'audit'/'region_support.json';write(support_path,region_support(rows))
    required['data/region_support.json']=support_path
    for name in ('admission.json','processor.json','history_support.json','raw_source_verification.json'):
        required['admission/'+name]=root/name
    for name in ('execution_version.json','recovery.json'):
        if (root/name).is_file():required['admission/'+name]=root/name
    for name in ('manifest.json','split_report.json','status.json'):
        required['data/'+name]=Path(directory)/name
    for group in GROUPS:
        variant,branch=group.split('_',1);run=root/group;compact=[];ids=set();core_errors={}
        complete=verify_training(run,manifest,projection,variant,branch,manifest_sha256=manifest_sha256)
        metrics=json.loads((run/'test'/'metrics.json').read_text())
        if metrics['checkpoint_sha256']!=complete['best_sha256']:raise ValueError('test_not_selected_checkpoint')
        if metrics['binding']!=complete['binding'] or complete['binding']['splits']!=manifest['splits']:
            raise ValueError('test_binding_changed')
        for p in sorted((run/'test'/'cases').glob('rank*_case*.json')):
            c=json.loads(p.read_text());key=c['id']
            if key in ids or key not in expected:raise ValueError('duplicate_or_unknown_test_member')
            ids.add(key);r=expected[key]
            if c.get('consumed_sources')!=sources.proof(r):raise ValueError('test_consumed_source_proof_changed')
            if (c['event']!=r['event'] or c['physical_group']!=r['physical_group'] or c['variant']!=variant
                or c['branch']!=branch or c['condition_source']!='oracle'
                or c['region_ids']!=[t['token_id'] for t in r['representations'][variant]['targets']]):
                raise ValueError('case_oracle_or_identity_changed')
            if not np.allclose(np.asarray(c['gt_waypoints'])[0],r['waypoint_xy'],rtol=1e-6,atol=1e-6):
                raise ValueError('continuous_GT_differs_from_dataset')
            truth=[c['gt_route'],c['gt_waypoints']]
            if key in gt_identity and gt_identity[key]!=truth:raise ValueError('six_group_GT_differs')
            gt_identity[key]=truth;error=trajectory_errors(c)
            predicted_regions,zero_regions=reclassified_regions(c,r,variant)
            core_errors[key]=[c['metrics'][k] for k in ('route_ade_m','route_fde_m','waypoint_ade_m','waypoint_fde_m')]
            compact.append(dict(id=key,event=c['event'],physical_group=c['physical_group'],
                region_ids=c['region_ids'],route_errors=error['route'],waypoint_errors=error['waypoints']))
            compact[-1].update(predicted_regions=predicted_regions,gt_zero_height_regions=zero_regions,
                               motion_stratum=motion_stratum(r),inference=c['inference'])
        if ids!=expected.keys():raise ValueError('incomplete_test_cases_no_sampling_allowed')
        compact.sort(key=lambda r:r['id']);table=summarize(compact)
        if int(metrics['metrics']['samples'])!=len(compact):raise ValueError('test_metric_member_count_changed')
        for key,metric in (('route_ADE','route_ade_m'),('route_FDE','route_fde_m'),('waypoint_ADE','waypoint_ade_m'),('waypoint_FDE','waypoint_fde_m')):
            if not np.isclose(table['all'][key],metrics['metrics'][metric],rtol=1e-5,atol=1e-4):raise ValueError('aggregate_metrics_changed')
        compact_path=root/'audit'/(group+'_cases.npz')
        error_archive(compact_path,[lookup[r['id']] for r in compact],[core_errors[r['id']] for r in compact],
                      predicted_regions=[r['predicted_regions'] for r in compact],
                      gt_zero_height_regions=[r['gt_zero_height_regions'] for r in compact])
        table_path=root/'audit'/(group+'_metrics.json');write(table_path,table)
        required[group+'/cases.npz']=compact_path;required[group+'/metrics.json']=table_path
        for name in ('training_complete.json','condition_contract.json','config.json'):
            required[group+'/'+name]=run/name
        for p in (run/'token_diagnostics').glob('*.json'):required[group+'/tokens/'+p.name]=p
        for p in (run/'performance').glob('*.json'):required[group+'/performance/'+p.name]=p
        required[group+'/condition_sensitivity.json']=run/'test'/'condition_sensitivity.json'
        sensitivity=json.loads((run/'test'/'condition_sensitivity.json').read_text())
        if sensitivity['checkpoint_sha256']!=complete['best_sha256'] or sensitivity['binding']!=complete['binding']:
            raise ValueError('condition_diagnostic_checkpoint_changed')
        validation={r['id']:r for r in rows['val']};condition_ids=set()
        from .diagnostic_coverage import select,labels
        _,expected_coverage=select(rows['val'],variant)
        if sensitivity.get('coverage')!=expected_coverage:raise ValueError('condition_diagnostic_coverage_changed')
        for case in sensitivity['cases']:
            if case['id'] not in validation or case['id'] in condition_ids:raise ValueError('invalid_condition_diagnostic_members')
            condition_ids.add(case['id']);truth=validation[case['id']]
            if case['event']!=truth['event'] or case['labels']!=[t['token_id'] for t in truth['representations'][variant]['targets']]:
                raise ValueError('condition_diagnostic_labels_changed')
            if case['history_labels']!=labels(truth,variant,'history') or case['physical_group']!=truth['physical_group']:
                raise ValueError('condition_diagnostic_history_or_group_changed')
        from .diagnostics import summarize as summarize_sensitivity
        observed=summarize_sensitivity(sensitivity['cases'],expected_coverage)
        if observed!=sensitivity['summary'] or not sensitivity['same_noise']:raise ValueError('condition_diagnostic_changed')
        if observed['status']=='condition_use_review_required':quality_reviews.append(group)
        required[group+'/probe_complete.json']=root/'probes'/group/'probe_complete.json'
        probe_root=root/'probes'/group;probe=json.loads((probe_root/'probe_complete.json').read_text())
        verify_probe_inputs(probe,probe_selections[variant],gate['processor_sha256'],probe_root)
        for i in range(4):required[group+f'/probe_rank{i}.json']=probe_root/f'rank{i}.json'
        inventory_path=run/'test'/'full_prediction_inventory.json'
        write(inventory_path,{p.name:sha(p) for p in sorted((run/'test'/'cases').glob('rank*_case*.json'))})
        summary[group]=dict(cases=len(compact),best_sha256=complete['best_sha256'],metrics=table['all'],
            full_prediction_inventory_sha256=sha(inventory_path),
            columns=['route_ADE','route_FDE','waypoint_ADE','waypoint_FDE'],
            compact_recomputes='all/event/physical-group ADE and FDE; per-time errors require full bound predictions')
    summary_path=root/'audit'/'summary.json';write(summary_path,summary);required['summary.json']=summary_path
    receipt=pack(root/'delivery'/'abc_trajectory_audit.zip',required,summary=dict(groups=list(GROUPS),test_cases=len(expected),
        all_core_cases=True,condition_source='oracle',old_semantic_comparison='historical_only',condition_reviews=quality_reviews))
    return dict(receipt,quality_status='condition_use_review_required' if quality_reviews else 'observed_condition_response')
