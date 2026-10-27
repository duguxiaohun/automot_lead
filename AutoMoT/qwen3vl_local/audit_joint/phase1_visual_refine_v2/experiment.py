"""Frozen, full-set paired replay for exactly one narrow candidate."""
import json
from pathlib import Path
from . import prompts, regression
from qwen3vl_local.audit_joint.phase1_visual_refine import experiment as legacy
from qwen3vl_local.audit_joint.phase1_visual_refine.calibration import sha
from qwen3vl_local.audit_joint.phase1_visual_refine.regression import read_cases, expected
from qwen3vl_local.sft_new_loop_phase1 import eval as native_eval, prompts as native, run_integrity

def source_identity():
    result = legacy.source_identity()
    result.update({'v2/'+p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    return result

def verify_source_identity(record):
    legacy.verify_source_identity({k:v for k,v in record.items() if not k.startswith('v2/')})
    for name, digest in record.items():
        if name.startswith('v2/'):
            if Path(name[3:]).name != name[3:] or sha(Path(__file__).parent/name[3:]) != digest:
                raise ValueError(f'V2 source changed: {name}')

def reference_bytes(records):
    return ''.join(json.dumps(records[i], ensure_ascii=False, sort_keys=True)+'\n' for i in sorted(records)).encode()

def read_reference(reference, inputs):
    folder = Path(reference)
    if not folder.is_dir() or not (folder/'evaluation_complete.json').is_file():
        raise ValueError('Reference requires the completed original evaluation directory')
    done = json.loads((folder/'evaluation_complete.json').read_text())
    if done.get('status') != 'verified' or done.get('phase') != 1 or done.get('inputs_sha256') != sha(inputs):
        raise ValueError('Reference completion/input mismatch')
    names = done.get('case_files_sha256', {})
    actual = {p.name for p in folder.glob('cases_rank*.jsonl')}
    if not names or actual != set(names):
        raise ValueError('Reference raw-file membership mismatch')
    for name, digest in names.items():
        if Path(name).name != name or sha(folder/name) != digest:
            raise ValueError('Reference raw output changed since evaluation completion')
    records = read_cases(folder)
    if len(records) != done.get('total_cases'):
        raise ValueError('Reference completion count mismatch')
    return records, sha(folder/'evaluation_complete.json')

def prepare(inputs, data_root, reference, candidate):
    # Validate candidate before loading the dataset; no ground-truth driven routing.
    if candidate not in prompts.RULES:
        raise ValueError('Unknown candidate')
    frozen, prepared, receipt = legacy.prepare(inputs, data_root)
    records, reference_completion_sha = read_reference(reference, inputs)
    regression.native_compare(frozen, records, records)  # identity/spec/GT/full membership
    texts = []
    for item, _ in prepared:
        old = native.build_phase1_prompt(spec=item.spec, history_rgb_mode=receipt['mode'])
        new = prompts.build(candidate=candidate, spec=item.spec, history_rgb_mode=receipt['mode'])
        texts.append(old != new)
    if not any(texts):
        raise ValueError('Candidate definition never rendered; refuse a silent no-op experiment')
    import hashlib
    receipt.update(candidate=candidate, candidate_prompt_sha256=prompts.fingerprint(candidate=candidate, history_rgb_mode=receipt['mode']),
                   source_sha256=source_identity(), reference_sha256=hashlib.sha256(reference_bytes(records)).hexdigest(),
                   reference_completion_sha256=reference_completion_sha,
                   changed_prompt_cases=sum(texts), unchanged_prompt_cases=len(texts)-sum(texts),
                   model_regression_status='not_run', production_default_changed=False)
    return frozen, prepared, records, receipt

def generate_arm(bundle, images, item, mode, candidate, tokens):
    previous = native_eval.SYSTEM_PROMPT
    try:
        native_eval.SYSTEM_PROMPT = native.SYSTEM_PROMPT
        prompt = (native.build_phase1_prompt(spec=item.spec, history_rgb_mode=mode) if candidate is None
                  else prompts.build(candidate=candidate, spec=item.spec, history_rgb_mode=mode))
        raw = native_eval._generate(bundle, images, prompt, tokens)
        return raw, dict(getattr(bundle, '_generation_measurement', {}))
    finally:
        native_eval.SYSTEM_PROMPT = previous

def replay(inputs, data_root, model_dir, adapter_dir, output, reference, candidate, device='cuda:0'):
    import torch
    frozen, prepared, recorded, receipt = prepare(inputs, data_root, reference, candidate)
    output = Path(output)
    if output.exists():
        raise ValueError('Output already exists; no implicit overwrite/resume')
    identity = run_integrity.verify_adapter(Path(adapter_dir))
    if identity != frozen['adapter_identity']:
        raise ValueError('Adapter differs from frozen run')
    cfg = native_eval._validate_phase1_adapter(Path(adapter_dir), Path(model_dir))
    if native_eval._resolve_history_rgb_mode(None, cfg)[0] != receipt['mode']:
        raise ValueError('Adapter RGB mode mismatch')
    output.mkdir(parents=True, exist_ok=False)
    (output/'preparation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    (output/'reference.jsonl').write_bytes(reference_bytes(recorded))
    (output/'reference_completion.json').write_bytes((Path(reference)/'evaluation_complete.json').read_bytes())
    try:
        bundle = native_eval.load_eval_bundle(Path(model_dir), Path(adapter_dir), torch.device(device), merge_lora=True)
        assets = getattr(bundle.model.config, 'local_asset_sha256', None)
        if not assets or assets != frozen['base_asset_sha256']:
            raise ValueError('Base assets differ from frozen run')
        tokens = int(frozen['arguments']['max_new_tokens'])
        if tokens <= 0:
            raise ValueError('Invalid generation budget')
        records = [{}, {}]
        with (output/'baseline.jsonl').open('x') as old, (output/'candidate.jsonl').open('x') as new:
            streams = (old, new)
            for i, (item, paths) in enumerate(prepared):
                images = native_eval._load_images(paths)
                try:
                    for side in ((0, 1) if i % 2 == 0 else (1, 0)):
                        raw, timing = generate_arm(bundle, images, item, receipt['mode'], candidate if side else None, tokens)
                        record = {**expected(item, i), 'raw_output':raw, 'performance':timing}
                        records[side][i] = record
                        streams[side].write(json.dumps(record, ensure_ascii=False)+'\n'); streams[side].flush()
                finally:
                    for image in images:
                        image.close()
                print(f'[paired-v2:{candidate}] {i+1}/{len(prepared)}', flush=True)
        for path, digest in receipt['local_image_sha256'].items():
            if sha(path) != digest:
                raise ValueError('RGB changed during replay')
        if sha(inputs) != receipt['frozen_input_sha256'] or run_integrity.verify_adapter(Path(adapter_dir)) != identity:
            raise ValueError('Frozen input/adapter changed during replay')
        final_reference, final_reference_sha = read_reference(reference, inputs)
        if reference_bytes(final_reference) != reference_bytes(recorded) or final_reference_sha != receipt['reference_completion_sha256']:
            raise ValueError('Recorded reference changed during replay')
        if sha(output/'reference.jsonl') != receipt['reference_sha256']:
            raise ValueError('Saved reference changed')
        verify_source_identity(receipt['source_sha256'])
        report = regression.compare(frozen, *records, recorded)
        (output/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        receipt.update(status='completed', model_inference=True, model_regression_status=report['status'],
                       adapter_identity=identity, base_asset_sha256=assets, environment=run_integrity.environment(),
                       max_new_tokens=tokens, comparison_sha256=sha(output/'comparison.json'),
                       outputs={name:sha(output/name) for name in ('baseline.jsonl','candidate.jsonl','reference.jsonl','reference_completion.json')})
        (output/'completion.json').write_text(json.dumps(receipt, indent=2)+'\n')
        return report
    except BaseException as exc:
        (output/'rejected.json').write_text(json.dumps(dict(status='rejected', error=str(exc)))+'\n')
        raise
