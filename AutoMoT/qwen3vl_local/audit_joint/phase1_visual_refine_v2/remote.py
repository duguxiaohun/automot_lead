"""Both-mode admission, one candidate per invocation, full replay and evidence."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import tarfile
from qwen3vl_local.audit_joint.phase1_visual_refine.remote import APP, MODES, DEFAULT_PIPELINE, discover, choose_gpu
from qwen3vl_local.audit_joint.phase1_visual_refine.rgb_inputs import admit
from qwen3vl_local.audit_joint.phase1_visual_refine.calibration import sha

def package(output, inputs):
    from .experiment import verify_source_identity
    from .regression import compare
    from .prompts import fingerprint
    from qwen3vl_local.audit_joint.phase1_visual_refine.regression import read_cases
    output = Path(output); done = json.loads((output/'completion.json').read_text())
    if done.get('status') != 'completed' or done.get('model_inference') is not True:
        raise ValueError('Cannot package incomplete model replay')
    if set(done['outputs']) != {'baseline.jsonl','candidate.jsonl','reference.jsonl','reference_completion.json'}:
        raise ValueError('Missing full paired/reference outputs')
    for name, digest in done['outputs'].items():
        if sha(output/name) != digest:
            raise ValueError('Output binding mismatch')
    if (sha(inputs) != done['frozen_input_sha256'] or sha(output/'comparison.json') != done['comparison_sha256']
            or sha(output/'reference.jsonl') != done['reference_sha256']
            or sha(output/'reference_completion.json') != done['reference_completion_sha256']):
        raise ValueError('Input/comparison/reference binding mismatch')
    if done['candidate_prompt_sha256'] != fingerprint(candidate=done['candidate'], history_rgb_mode=done['mode']):
        raise ValueError('Candidate fingerprint mismatch')
    verify_source_identity(done['source_sha256'])
    computed = compare(json.loads(Path(inputs).read_text()), *[read_cases(output/n) for n in ('baseline.jsonl','candidate.jsonl','reference.jsonl')])
    if computed != json.loads((output/'comparison.json').read_text()):
        raise ValueError('Raw score recomputation mismatch')
    archive = output.with_name(output.name+'_audit_bundle.tar.gz')
    with tarfile.open(archive, 'x:gz') as tar:
        for name in (*done['outputs'],'completion.json','comparison.json','preparation.json'):
            tar.add(output/name, arcname=name, recursive=False)
        tar.add(inputs, arcname='evaluation_inputs.json', recursive=False)
        for folder in (Path(__file__).parent, Path(__file__).parent.parent/'phase1_visual_refine'):
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix in ('.py','.sh','.md'):
                    tar.add(p, arcname='code/'+folder.name+'/'+p.name, recursive=False)
    if archive.stat().st_size > 30*1024*1024:
        raise ValueError(f'Full bundle exceeds 30 MiB; retained: {archive}')
    return archive

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('all',*MODES), default='all', nargs='?')
    p.add_argument('--candidate', required=True, choices=('rs2_same_direction','rs4_signal_head'))
    p.add_argument('--pipeline-root', default=DEFAULT_PIPELINE)
    p.add_argument('--model-dir', default='checkpoints/Qwen3.5-4B')
    p.add_argument('--data-root')
    p.add_argument('--output-root', default='checkpoints/phase1_visual_refine_v2_remote')
    a = p.parse_args(); jobs = []
    for mode in MODES if a.mode == 'all' else (a.mode,):
        inputs, adapter = discover(a.pipeline_root, mode)
        data_root, admission = admit(inputs, a.data_root, app=APP)
        # The original evaluation's raw outputs protect historical old-correct cases.
        reference = inputs.parent
        jobs.append((mode, inputs, adapter, data_root, reference, admission))
    if not Path(a.model_dir, 'model.safetensors.index.json').is_file():
        raise FileNotFoundError(f'Missing local base assets: {a.model_dir}')
    gpu = choose_gpu(); os.environ['GPU_IDS']=gpu; os.environ['CUDA_VISIBLE_DEVICES']=gpu
    print(f'[gpu] GPU_IDS={gpu}; candidate={a.candidate}; model quality pending', flush=True)
    from .experiment import prepare, replay
    # Select the physical device before importing the model backend. All selected
    # modes still pass CPU prompt/reference admission before any model load.
    for mode, inputs, adapter, data_root, reference, admission in jobs:
        prepare(inputs, data_root, reference, a.candidate)
    root = Path(a.output_root)/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}_{a.candidate}')
    root.mkdir(parents=True, exist_ok=False); results = {}
    for mode, inputs, adapter, data_root, reference, admission in jobs:
        (root/f'{mode}_rgb_admission.json').write_text(json.dumps(admission, indent=2)+'\n')
        report = replay(inputs, data_root, a.model_dir, adapter, root/mode, reference, a.candidate)
        print('[BUNDLE]', package(root/mode, inputs), flush=True)
        results[mode] = report['status']
        if report['status'] == 'review_required':
            print(f'[REVIEW_REQUIRED] {mode}: quality regression; inference/package complete', flush=True)
    review = 'review_required' in results.values()
    (root/'status.json').write_text(json.dumps(dict(status='review_required' if review else 'no_observed_regression',
                                                   modes=results, candidate=a.candidate, training_started=False,
                                                   production_default_changed=False), indent=2)+'\n')
    print('[REVIEW_REQUIRED]' if review else '[DONE]', root, flush=True)
    if review:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
