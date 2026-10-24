"""Same sample and FM noise; interventions are diagnostics, not trained controls."""
from collections import defaultdict
from qwen3vl_local.audit_joint.abc_delivery import write


def summarize(cases,coverage=None):
    tolerance=1e-6
    replaced=[c['results']['replace_regions']['mean_waypoint_shift_from_oracle_m'] for c in cases]
    reordered=[c['results']['permute_horizons']['mean_waypoint_shift_from_oracle_m']
               for c in cases if c['labels']!=list(reversed(c['labels']))]
    reasons=[]
    if not replaced or max(replaced)<=tolerance:reasons.append('no_observed_response_to_region_replacement')
    if reordered and max(reordered)<=tolerance:reasons.append('no_observed_response_to_nontrivial_horizon_permutation')
    from .diagnostic_coverage import statistics,detailed
    breakdown=detailed(cases,coverage) if coverage is not None else None
    if breakdown and breakdown['review_cells']:reasons.append('partial_or_absent_isolated_slot_response')
    return dict(status='condition_use_review_required' if reasons else 'observed_condition_response',
                reasons=reasons,numerical_identity_tolerance_m=tolerance,replacement_cases=len(replaced),
                nontrivial_permutation_cases=len(reordered),permutation_supported=bool(reordered),
                max_replacement_shift_m=max(replaced,default=0.),max_permutation_shift_m=max(reordered,default=0.),
                replacement_statistics=statistics(replaced),permutation_statistics=statistics(reordered),
                role_horizon_region=breakdown,
                prediction_quality_proven=False)


def sensitivity(runtime,model,config,rows,args,dtype,rank,world,out):
    import torch
    import torch.distributed as dist
    from qwen3vl_local.action_expert_ablation.common import leadmot_train
    from qwen3vl_local.action_prior.flow_matching import make_evaluation_flow
    from .loader import Dataset
    from .diagnostic_coverage import select,labels
    old=leadmot_train();selected,coverage=select(rows,runtime.variant)
    cases=[];was_training=model.training;model.eval()
    try:
        for sample in selected[rank::world]:
            prepared=Dataset([sample],args)[0]
            if prepared.get('_error'):raise ValueError(prepared['_error'])
            route=prepared['gt_route'][None].to(runtime.device);wp=prepared['gt_waypoints'][None].to(runtime.device)
            flow=make_evaluation_flow(route,wp,model.flow_config,sample,args.seed)
            base=None;result={}
            for mode in (None,'replace_regions','permute_horizons','no_future'):
                runtime.intervention=mode
                with torch.no_grad():
                    output=runtime.forward_sample(sample,model,config,dtype,clip=prepared['clip'],
                        flow_sample_noise=flow['flow_sample_noise'],sample_trajectory=True)
                if base is None:base=output['pred_future_waypoints']
                result[mode or 'oracle']=dict(old._compute_planning_metrics(output,route,wp),
                    mean_waypoint_shift_from_oracle_m=float((output['pred_future_waypoints']-base).norm(dim=-1).mean()))
            slots={}
            for role in ('future','history'):
                slots[role]=[]
                for slot in range(4):
                    runtime.intervention=dict(role=role,slot=slot)
                    with torch.no_grad():
                        output=runtime.forward_sample(sample,model,config,dtype,clip=prepared['clip'],
                            flow_sample_noise=flow['flow_sample_noise'],sample_trajectory=True)
                    slots[role].append(float((output['pred_future_waypoints']-base).norm(dim=-1).mean()))
            cases.append(dict(id=sample['abc_record']['id'],event=sample['abc_record']['event'],
                              physical_group=sample['abc_record']['physical_group'],
                              labels=labels(sample['abc_record'],runtime.variant,'future'),
                              history_labels=labels(sample['abc_record'],runtime.variant,'history'),
                              slot_responses=slots,results=result))
    finally:runtime.intervention=None;model.train(was_training)
    gathered=[None]*world
    if world>1:dist.all_gather_object(gathered,cases)
    else:gathered=[cases]
    if rank==0:
        merged=[c for part in gathered for c in part]
        write(out,dict(scope='validation_condition_sensitivity',same_noise=True,
            no_future_is_out_of_distribution=True,trained_unconditional_baseline=False,
            coverage=coverage,summary=summarize(merged,coverage),cases=merged))
