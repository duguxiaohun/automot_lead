"""Explicit four-RGB choice prompt experiment; never changes training contracts.

Prepare binds the entire received validation cohort and local images. Replay runs
both prompts with one adapter and fresh KV state, then reports paired regressions.
"""
import argparse
from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import zipfile

from .phase3_base_regression_review import compare, indexed
from qwen3vl_local.sft_new_loop_phase3 import prompts, prompt_contract
from qwen3vl_local.sft_new_loop_phase3.prompt_candidate import CANDIDATE_NAME

VARIANT = 'base_horizon_clarity_v1_20261024'
NOTE = (
    'Prediction timing is measured from the newest image (t=0). '
    'Near-stop confirmation uses two consecutive 0.25-second samples at or below 0.5 m/s, '
    'both within t=0 to +1.5 s; a stop confirmed later does not by itself qualify as immediate STOP. '
    'The first meaningful speed change is assessed through +2 s, including a reduction first reached at +2 s. '
    'A reduction need not persist for two samples; a speed gain requires two consecutive samples. '
    'For listed lane-change options, assess the first future lane-boundary crossing through +3 s. '
    'These are forecast definitions, not supplied future observations. '
    'Do not infer release merely because traffic could move later: continued waiting and immediate confirmed pull-away are different.\n\n'
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def source_hashes():
    app = Path(__file__).resolve().parents[2]
    names = ['audit_joint/phase3_base_prompt_experiment.py', 'audit_joint/phase3_base_regression_review.py',
             'sft_new_loop_phase3/prompts.py', 'sft_new_loop_phase3/prompt_candidate.py',
             'sft_new_loop_phase3/prompt_contract.py', 'sft_new_loop_phase3/choice_semantics.py',
             'sft_new_loop_phase3/trajectory_action.py', 'sft_new_loop_phase3/eval.py', 'sft_v3/train.py']
    return {name: sha(app/'qwen3vl_local'/name) for name in names}


def write(path, obj):
    with Path(path).open('x') as f: json.dump(obj, f, ensure_ascii=False, indent=2)


def select_replay_device(device):
    """Choose one physical GPU before importing the inference runtime."""
    if device == 'cpu': return device
    if device not in ('cuda','cuda:0'):
        raise ValueError('single-process replay uses cuda:0; select the physical GPU with GPU_IDS')
    selected=os.environ.get('GPU_IDS','').strip()
    if selected:
        if not selected.isdigit(): raise ValueError('replay GPU_IDS must name one physical GPU')
    else:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True)
        rows=[tuple(map(int,line.split(','))) for line in output.splitlines() if line.strip()]
        if not rows: raise ValueError('no GPU available for replay')
        selected=str(min(rows,key=lambda r:(r[1],r[2],r[0]))[0])
    os.environ['CUDA_VISIBLE_DEVICES']=selected
    return 'cuda:0'


def render(saved):
    if saved['action_output_mode'] != 'choice' or saved['invalid_context'] is not False:
        raise ValueError('requires valid four-RGB choice cohort')
    if any(type(q['answer']) is not bool for q in saved['questions']):
        raise ValueError('non-boolean target')
    kw = {f.name: saved[f.name] for f in fields(prompts.PromptSpec) if f.name != 'questions'}
    kw['goal_xy'] = tuple(kw['goal_xy']) if kw['goal_xy'] is not None else None
    spec = prompts.PromptSpec(**kw, questions=tuple(prompts.QuestionSpec(**q) for q in saved['questions']))
    if prompts.prompt_spec_to_json(spec) != saved: raise ValueError('saved prompt contract mismatch')
    def build(s):
        return prompt_contract.build_action_prompt(spec=s, prompt_variant=CANDIDATE_NAME, history_rgb_mode='4rgb')
    original = build(spec)
    flipped = replace(spec, questions=tuple(replace(q, answer=not q.answer) for q in spec.questions))
    if build(flipped) != original: raise ValueError('answer leaks into prompt')
    return original, NOTE + original


def prepare(archive, reference, data_root, output):
    ref_bytes = reference.read_bytes()
    ref = [r for r in json.loads(ref_bytes) if r['key'][0] == 'val']
    lookup = indexed(ref)
    if not lookup: raise ValueError('empty validation cohort')
    archive_hash = sha(archive)
    with zipfile.ZipFile(archive) as z:
        cfg = json.loads(z.read('experiment/common_reference.json'))
        if cfg['prompt_variant'] != CANDIDATE_NAME: raise ValueError('unexpected trained prompt')
        rows = [json.loads(x) for x in z.read('experiment/train/final_generation_val_cases.jsonl').splitlines()]
    requests = []
    for row in rows:
        key = ['val', row['scenario'], row['route_id'], row['frame_id'], row['context_id']]
        saved = row['prompt_spec']
        old, new = render(saved)
        r = lookup.get(tuple(key))
        if r is None or r['gt'] != saved['choice_target_action'] or r['pred'] != row['raw_output'].strip():
            raise ValueError('reference target/prediction/member mismatch')
        if row.get('action_user_prompt', old) != old: raise ValueError('saved prompt mismatch')
        paths = row['history_rgb_paths_used']
        expected = [[row['scenario'], row['route_id'], 'rgb', f'{i:04d}.jpg'] for i in range(row['frame_id']-3, row['frame_id']+1)]
        if [list(Path(p).parts[-4:]) for p in paths] != expected:
            raise ValueError('four history images/order mismatch')
        local = [data_root.joinpath(*parts).resolve() for parts in expected]
        if any(not p.is_relative_to(data_root.resolve()) for p in local): raise ValueError('image escapes data root')
        hashes = [sha(p) for p in local]
        if row.get('history_rgb_sha256', hashes) != hashes: raise ValueError('received image SHA mismatch')
        requests.append(dict(key=key, images=[str(p.relative_to(data_root.resolve())) for p in local],
            image_sha256=hashes, system=prompts.CHOICE_SYSTEM_PROMPT, original=old, candidate=new,
            allowed_actions=saved['choice_options']))
    if set(indexed(requests)) != set(lookup): raise ValueError('incomplete validation cohort')
    compare(ref, [dict(key=r['key'], raw_output=r['pred'], reference_sha256='check') for r in ref], 'check')
    if sha(archive) != archive_hash or reference.read_bytes() != ref_bytes: raise ValueError('inputs changed during prepare')
    output.mkdir(parents=True, exist_ok=False)
    write(output/'reference.json', ref)
    write(output/'requests.json', requests)
    manifest = dict(schema=VARIANT, cases=len(requests), archive_sha256=archive_hash,
        source_reference_sha256=hashlib.sha256(ref_bytes).hexdigest(),
        reference_sha256=sha(output/'reference.json'), requests_sha256=sha(output/'requests.json'),
        code_sha256=sha(__file__), sources=source_hashes(), original_variant=CANDIDATE_NAME, training_admitted=False,
        remote_val_image_equality='only verified when received per-image hashes exist',
        model_inference=False, prompt_chars=dict(original=[min(len(r['original']) for r in requests),max(len(r['original']) for r in requests)],
                                               candidate=[min(len(r['candidate']) for r in requests),max(len(r['candidate']) for r in requests)]))
    write(output/'manifest.json', manifest)
    return manifest


def load_plan(plan, data_root):
    manifest = json.loads((plan/'manifest.json').read_text())
    if manifest['schema'] != VARIANT or manifest['code_sha256'] != sha(__file__): raise ValueError('experiment source changed; prepare a new plan')
    if manifest['sources'] != source_hashes(): raise ValueError('runtime source changed; prepare a new plan')
    for name in ('reference', 'requests'):
        if sha(plan/(name+'.json')) != manifest[name+'_sha256']: raise ValueError('plan SHA mismatch')
    ref = json.loads((plan/'reference.json').read_text()); req = json.loads((plan/'requests.json').read_text())
    if set(indexed(ref)) != set(indexed(req)) or len(req) != manifest['cases']: raise ValueError('plan membership mismatch')
    for r in req:
        for p,h in zip(r['images'],r['image_sha256'], strict=True):
            path=(data_root/p).resolve()
            if not path.is_relative_to(data_root.resolve()) or sha(path) != h: raise ValueError('plan RGB mismatch')
    return manifest, ref, req


def replay(plan, data_root, model, adapter, output, device):
    manifest, ref, requests = load_plan(plan, data_root)
    device = select_replay_device(device)
    from qwen3vl_local.sft_new_loop_phase3.eval import _validate_action_adapter, load_eval_bundle, _kv_start_state, _student_generate_kv
    import torch
    from PIL import Image
    cfg = _validate_action_adapter(adapter, model, action_output_mode='choice')
    if cfg['history_rgb_mode'] != '4rgb' or cfg['prompt_variant'] != CANDIDATE_NAME: raise ValueError('wrong adapter contract')
    # Bind exact local model/adapter bytes, not only their mutable directory names.
    assets = {str(p):sha(p) for root in (model,adapter) for p in sorted(root.rglob('*')) if p.is_file()}
    output.mkdir(parents=True, exist_ok=False)
    write(output/'started.json', dict(plan=manifest, assets=assets, completed=False,
        device=device, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES') if device!='cpu' else None))
    bundle = load_eval_bundle(model, adapter, torch.device(device), merge_lora=True)
    predictions = {'original': [], 'candidate': []}
    with (output/'generations.jsonl').open('x') as stream:
        for r in requests:
            images=[]
            for p,h in zip(r['images'],r['image_sha256'],strict=True):
                if sha(data_root/p) != h: raise ValueError('RGB changed before inference')
                with Image.open(data_root/p) as im: images.append(im.convert('RGB'))
            for variant in predictions:
                torch.manual_seed(20260904)
                messages=[dict(role='system',content=r['system']),dict(role='user',content=[
                    *[dict(type='image', image=im) for im in images],dict(type='text',text=r[variant])])]
                # Count the actual multimodal input; never truncate away history or rules.
                packed=bundle.processor(text=[bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
                    images=images, return_tensors='pt', padding=True, truncation=False)
                tokens=int(packed['input_ids'].shape[-1])
                config=getattr(bundle.model.config, 'text_config', bundle.model.config)
                limit=getattr(config, 'max_position_embeddings', None)
                if not isinstance(limit, int) or tokens+32>limit: raise ValueError('unknown/exceeded model context budget')
                del packed
                with torch.inference_mode():
                    state=_kv_start_state(bundle,messages)
                    raw,_,_=_student_generate_kv(bundle,state,32)
                del state
                # Domain-invalid tokens remain failures even if they name another valid action.
                scored=raw if raw.strip() in r['allowed_actions'] else 'INVALID'
                item=dict(key=r['key'],raw_output=scored,reference_sha256=manifest['reference_sha256'])
                predictions[variant].append(item)
                stream.write(json.dumps(dict(**item, variant=variant, generated_text=raw, input_tokens=tokens))+'\n');stream.flush()
    load_plan(plan, data_root)
    if any(sha(p)!=h for p,h in assets.items()): raise ValueError('model assets changed during replay')
    stability=compare(ref,predictions['original'],manifest['reference_sha256'])
    fresh = [dict(r,pred=p['raw_output'].strip(),correct=p['raw_output'].strip()==r['gt']) for r,p in zip(ref, [indexed(predictions['original'])[tuple(r['key'])] for r in ref])]
    write(output/'fresh_reference.json',fresh)
    fresh_hash=sha(output/'fresh_reference.json')
    fresh_predictions=[dict(p,reference_sha256=fresh_hash) for p in predictions['candidate']]
    paired=compare(fresh,fresh_predictions,fresh_hash)
    write(output/'original_predictions.json',predictions['original'])
    write(output/'candidate_predictions.json',fresh_predictions)
    write(output/'report.json',dict(completed=True, saved_baseline_replay=stability, paired_prompt_effect=paired,
        reference_kind='fresh original-prompt inference on same weights and images',
        saved_baseline_predictions_changed=sum(r['pred']!=indexed(predictions['original'])[tuple(r['key'])]['raw_output'].strip() for r in ref),
        promotion_authorized=False, training_performed=False, plan=manifest))


def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,resource.getrlimit(resource.RLIMIT_CORE)[1]))
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    a.add_argument('--archive',type=Path,required=True);a.add_argument('--reference',type=Path,required=True)
    b=sub.add_parser('replay')
    b.add_argument('--plan',type=Path,required=True);b.add_argument('--model',type=Path,required=True)
    b.add_argument('--adapter',type=Path,required=True);b.add_argument('--device',default='cuda:0')
    for parser in (a,b):
        parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=vars(p.parse_args()); command=args.pop('command')
    result=prepare(**args) if command=='prepare' else replay(**args)
    if result: print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
