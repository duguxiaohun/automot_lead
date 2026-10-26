"""Explicit, single-process paired prompt experiment on frozen native cases.

The adapter retains its original identity. The candidate prompt is an experiment,
not a migrated adapter. Each arm calls the native fresh-KV generator separately.
"""
import json
from pathlib import Path
from .calibration import sha, resolve_rgb
from . import prompts
from .regression import expected, compare
from qwen3vl_local.sft_new_loop_phase1 import eval as native_eval
from qwen3vl_local.sft_new_loop_phase1 import prompts as native
from qwen3vl_local.sft_new_loop_phase1 import run_integrity
from qwen3vl_local.sft_new_loop_phase1.evaluation_integrity import item_from_json


def source_identity():
    result = run_integrity.sources(1)
    result.update({'candidate/'+p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    return result


def verify_source_identity(record):
    for name, digest in record.items():
        path = (Path(__file__).parent/name.removeprefix('candidate/')
                if name.startswith('candidate/') else run_integrity.ROOT/name)
        if sha(path) != digest: raise ValueError(f'Source changed: {name}')


def prepare(inputs, data_root):
    path = Path(inputs); before = sha(path); frozen = json.loads(path.read_text())
    if frozen['phase'] != 1 or len(frozen['items']) != frozen['expected_total'] or not frozen['items']:
        raise ValueError('Full Phase1 frozen membership required')
    mode = frozen['history_rgb_mode']
    if mode not in ('4rgb', '2rgb_endpoints'): raise ValueError('Unsupported RGB mode')
    if frozen['arguments'].get('audit_prompt', False): raise ValueError('Use production judgment cases for this experiment')
    checked = {}; prepared = []; hashes = frozen['image_sha256']
    for index, value in enumerate(frozen['items']):
        item = item_from_json(1, value); paths = []
        selected = native_eval.select_history_rgb_paths(item.row.history_rgb_paths, mode)
        for raw in selected:
            # The frozen server path is used only as a hash key. Resolve locally.
            parts = Path(raw).parts
            if '..' in parts or tuple(parts[-4:-2]) != (item.row.scenario, item.row.route_id):
                raise ValueError('Frozen RGB identity differs from its frame')
            local = resolve_rgb(data_root, str(Path(*parts[-4:])))
            digest = checked.setdefault(str(local), sha(local)) if str(local) not in checked else checked[str(local)]
            if hashes.get(raw) != digest: raise ValueError(f'Frozen RGB SHA mismatch: {raw}')
            paths.append(str(local))
        prepared.append((item, paths))
    if sha(path) != before: raise ValueError('Frozen inputs changed')
    receipt = dict(status='prepared_cpu_only', frozen_input_sha256=before, mode=mode,
                   cases=len(prepared), images_verified=len(checked), local_image_sha256=checked,
                   candidate_prompt_sha256=prompts.fingerprint(history_rgb_mode=mode),
                   parent_prompt_sha256=native.phase1_prompt_sha256(history_rgb_mode=mode),
                   source_sha256=source_identity(), model_inference=False)
    return frozen, prepared, receipt


def generate_arm(bundle, images, item, mode, candidate, tokens):
    # Only this CLI's own process is patched, never training files or saved metadata.
    original = native_eval.SYSTEM_PROMPT
    try:
        native_eval.SYSTEM_PROMPT = prompts.SYSTEM_PROMPT if candidate else native.SYSTEM_PROMPT
        prompt = (prompts.build if candidate else native.build_phase1_prompt)(spec=item.spec, history_rgb_mode=mode)
        raw = native_eval._generate(bundle, images, prompt, tokens)
        return raw, dict(getattr(bundle, '_generation_measurement', {}))
    finally:
        native_eval.SYSTEM_PROMPT = original


def replay(inputs, data_root, model_dir, adapter_dir, output, device='cuda:0'):
    import torch
    frozen, prepared, receipt = prepare(inputs, data_root)
    model_dir, adapter_dir, output = Path(model_dir), Path(adapter_dir), Path(output)
    if output.exists(): raise ValueError('Output already exists; no implicit overwrite/resume')
    adapter_identity = run_integrity.verify_adapter(adapter_dir)
    if adapter_identity != frozen['adapter_identity']: raise ValueError('Adapter differs from frozen run')
    cfg = native_eval._validate_phase1_adapter(adapter_dir, model_dir)
    if native_eval._resolve_history_rgb_mode(None, cfg)[0] != receipt['mode']: raise ValueError('Adapter RGB mode mismatch')
    output.mkdir(parents=True, exist_ok=False)
    (output/'preparation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    try:
        bundle = native_eval.load_eval_bundle(model_dir, adapter_dir, torch.device(device), merge_lora=True)
        assets = getattr(bundle.model.config, 'local_asset_sha256', None)
        if not assets or assets != frozen['base_asset_sha256']: raise ValueError('Base assets differ from frozen run')
        records = [{}, {}]; tokens = int(frozen['arguments']['max_new_tokens'])
        if tokens <= 0: raise ValueError('Invalid generation budget')
        with (output/'baseline.jsonl').open('x') as old, (output/'candidate.jsonl').open('x') as new:
            streams = (old, new)
            for index, (item, paths) in enumerate(prepared):
                images = native_eval._load_images(paths)
                try:
                    for side in ((0, 1) if index % 2 == 0 else (1, 0)):
                        raw, timing = generate_arm(bundle, images, item, receipt['mode'], bool(side), tokens)
                        record = {**expected(item, index), 'raw_output':raw, 'performance':timing}
                        records[side][index] = record
                        streams[side].write(json.dumps(record, ensure_ascii=False)+'\n'); streams[side].flush()
                finally:
                    for img in images: img.close()
                print(f'[paired-replay] {index+1}/{len(prepared)}', flush=True)
        for path, digest in receipt['local_image_sha256'].items():
            if sha(path) != digest: raise ValueError('RGB changed during replay')
        if sha(inputs) != receipt['frozen_input_sha256']: raise ValueError('Input changed during replay')
        verify_source_identity(receipt['source_sha256'])
        if run_integrity.verify_adapter(adapter_dir) != adapter_identity: raise ValueError('Adapter changed during replay')
        report = compare(frozen, *records)
        (output/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        receipt.update(status='completed', model_inference=True, regression_status=report['status'],
                       adapter_identity=adapter_identity, base_asset_sha256=assets,
                       environment=run_integrity.environment(), max_new_tokens=tokens,
                       comparison_sha256=sha(output/'comparison.json'),
                       outputs={p.name:sha(p) for p in (output/'baseline.jsonl',output/'candidate.jsonl')})
        (output/'completion.json').write_text(json.dumps(receipt, indent=2)+'\n')
        return report
    except BaseException as exc:
        (output/'rejected.json').write_text(json.dumps({'status':'rejected','error':str(exc)})+'\n')
        raise
