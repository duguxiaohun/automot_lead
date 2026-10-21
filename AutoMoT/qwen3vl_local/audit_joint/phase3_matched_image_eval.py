"""Run a prepared validation image intervention with the completed old model.

Use the original training checkout and explicit job path. Existing result files,
datasets and weights are read-only; output must be new. No training is started.
"""
import argparse
import copy
import json
from pathlib import Path

from . import phase3_behavior_diagnostics as diag


def checked_pairs(plan, records, data_root):
    """Validate all input metadata and RGB before loading a GPU model."""
    from PIL import Image, ImageStat
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group
    metadata=diag.validate_plan(plan);root=Path(data_root).resolve(strict=True)
    if not plan['pairs']:raise ValueError('No matched image support; inference would be empty')
    used={x[k] for x in plan['pairs'] for k in ('receiver','donor')}
    for key in used:
        r=records[key];m=metadata[key]
        if (r['split']!='val' or r['event']!=m['event'] or
            physical_route_group(r['scenario'],r['route_id'])!=m['group']):
            raise ValueError('Plan identity/event/physical group differs from admitted dataset')
        path=(root/m['image']).resolve(strict=True)
        if not path.is_relative_to(root) or Path(r['image']).resolve(strict=True)!=path:
            raise ValueError('Plan RGB does not match admitted record')
        if diag.file_sha(path)!=m['rgb_sha256']:raise ValueError('Intervention source RGB changed')
        with Image.open(path) as im:
            if list(im.size)!=m['image_size']:raise ValueError('Intervention image shape changed')
            brightness=ImageStat.Stat(im.convert('L')).mean[0]
            if abs(brightness-m['brightness'])>1e-8:raise ValueError('Image brightness metadata changed')
    return metadata


def predict_pairs(bundle, records, plan, *, predict=None, reference_cache=None, swapped_cache=None):
    """Change only the image path; retain receiver IDs, premises and targets."""
    if predict is None:
        from .phase3_single_frame.runtime import predict
    reference_cache={} if reference_cache is None else reference_cache
    swapped_cache={} if swapped_cache is None else swapped_cache
    reference=[records[p['receiver']] for p in plan['pairs'] if p['receiver'] not in reference_cache]
    pending=[p for p in plan['pairs'] if (p['receiver'],p['donor']) not in swapped_cache]
    swapped=[dict(records[p['receiver']],image=records[p['donor']]['image']) for p in pending]
    snapshot=copy.deepcopy([records[p['receiver']] for p in plan['pairs']])
    if reference:
        values,_=predict(bundle,reference)
        if [c['id'] for c in values]!=[r['id'] for r in reference]:raise ValueError('Prediction identities changed')
        reference_cache.update({c['id']:c for c in values})
    if swapped:
        values,_=predict(bundle,swapped)
        if [c['id'] for c in values]!=[r['id'] for r in swapped]:raise ValueError('Prediction identities changed')
        for c,p in zip(values,pending):
            swapped_cache[p['receiver'],p['donor']]=dict(c,image_mode='matched_other_route',donor_id=p['donor'])
    if [records[p['receiver']] for p in plan['pairs']]!=snapshot:
        raise ValueError('Inference mutated the original records')
    full=[reference_cache[p['receiver']] for p in plan['pairs']]
    wrong=[swapped_cache[p['receiver'],p['donor']] for p in plan['pairs']]
    return full,wrong


def run(job_path, plan_paths, output, device='cuda:0', preflight_only=False):
    import torch
    from . import phase3_base_trial as trial, phase3_base_trial_worker as base
    from .phase3_single_frame import dataset,runtime,worker
    from .phase3_suite import check_model_snapshot
    from qwen3vl_local.sft_new_loop_phase3 import training_state
    job_path=Path(job_path).resolve(strict=True);output=Path(output).resolve()
    if output.exists():raise ValueError('Use a new output directory; no overwrite')
    job=trial.read(job_path)
    protected=[Path(job['output']).resolve(),Path(job['dataset']).resolve().parent,
               Path(job['data_root']).resolve(),Path(job['model_dir']).resolve()]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Diagnostic output overlaps original experiment/data/model')
    base.validate_job(job)
    check_model_snapshot(Path(job['model_dir']),job['model_snapshot'])
    # Native source/asset verification stays intact, including the old release.
    artifact=dataset.load(job['dataset'],index=job['index'])
    admitted=trial.read(Path(job['output'])/'single_frame_preflight.json')
    expected=runtime.make_args(job,artifact).single_frame_contract
    if admitted['contract']!=expected:raise ValueError('Original inference contract changed')
    plans=[trial.read(p) for p in plan_paths]
    if not plans or len({p['seed'] for p in plans})!=len(plans):raise ValueError('Empty or duplicate intervention seeds')
    common_ids=set.intersection(*({p['receiver'] for p in plan['pairs']} for plan in plans))
    if not common_ids:raise ValueError('No common receivers across intervention seeds')
    fixed=admitted['plans']['val']['ordered_ids']
    for p in plans:
        if [r['id'] for r in p['metadata']]!=fixed:
            raise ValueError('Plan does not use the exact original fixed validation membership/order')
        checked_pairs(p,artifact['records'],job['data_root'])
    checkpoint=Path(job['output'])/'train/training_states/epoch_0003'
    receipt=trial.read(checkpoint/'receipt.json')
    if receipt['files']!=training_state.files(checkpoint):raise ValueError('Checkpoint files changed')
    state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
    if (state['binding']!=receipt['binding'] or state['global_step']!=7680 or state['next_epoch']!=3 or
        state['binding']['configuration']['single_frame_contract']!=expected):
        raise ValueError('Completed checkpoint and original single-frame contract mismatch')
    note=dict(policy=diag.POLICY,mode='preflight_only' if preflight_only else 'inference',
        job_sha256=diag.file_sha(job_path),checkpoint_receipt_sha256=diag.file_sha(checkpoint/'receipt.json'),
        plans=[dict(seed=p['seed'],sha256=p['plan_sha256'],matched=len(p['pairs'])) for p in plans],
        primary_common_receiver_ids=sorted(common_ids),
        inference_performed=False,training_performed=False,source_sha256=diag.file_sha(__file__))
    output.mkdir(parents=True)
    trial.write(output/'receipt.json',note)
    if preflight_only:return note
    b=worker.bundle(job,torch.device(device))
    params={n:p for n,p in b.model.named_parameters() if p.requires_grad}
    if set(params)!=set(state['trainable']):raise ValueError('Checkpoint trainable topology differs')
    with torch.no_grad():
        for n,p in params.items():p.copy_(state['trainable'][n].to(device=p.device,dtype=p.dtype))
    del state
    # Only unique receiver and receiver/donor pairs are inferred across seeds.
    # Recompute actual-image reference under the same runtime as the intervention.
    full_cache={};wrong_cache={};results={}
    for plan in plans:
        full,wrong=predict_pairs(b,artifact['records'],plan,reference_cache=full_cache,swapped_cache=wrong_cache)
        metrics=diag.paired_scores(full,wrong,{r['id']:r for r in plan['metadata']})
        metrics['primary_common_across_seeds']=diag.paired_scores(
            [c for c in full if c['id'] in common_ids],[c for c in wrong if c['id'] in common_ids],
            {r['id']:r for r in plan['metadata']})
        metrics['primary_common_anchors']=len(common_ids)
        # Explicitly report inference reproducibility, never silently replace the
        # previously published full score with a matched-subset denominator.
        saved=trial.read(Path(job['output'])/'single_frame_val_cases.json')
        old={c['id']:c for c in saved}
        for c in full:
            if c['windows']!=old[c['id']]['windows']:raise ValueError('Published target windows changed')
        metrics['reference_prediction_changes_from_received']=sum(c['predictions']!=old[c['id']]['predictions'] for c in full)
        metrics['paired_anchors']=len(full);metrics['excluded_anchors']=len(plan['excluded'])
        trial.write(output/f'seed_{plan["seed"]}_reference_cases.json',full)
        trial.write(output/f'seed_{plan["seed"]}_wrong_image_cases.json',wrong)
        trial.write(output/f'seed_{plan["seed"]}_metrics.json',metrics)
        results[str(plan['seed'])]=metrics
    note.update(inference_performed=True,unique_actual_forwards=len(full_cache),unique_swapped_forwards=len(wrong_cache))
    trial.write(output/'receipt.json',note);trial.write(output/'metrics.json',results)
    return note


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--job',type=Path,required=True);p.add_argument('--plan',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.job,a.plan,a.output,a.device,a.preflight_only)))


if __name__=='__main__':main()
