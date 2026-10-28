"""Recorded packaging-only supplement for the unchanged Phase2 frozen runtime.

Never edits/chmods old evidence. The Python shim changes only the exact bound
inline packer, excluding code from the text pass before its single tree copy.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

NAMES = ('base_production', 'base_audit_prompt', 'lora_production', 'lora_audit_prompt')
AUDITS = ('audit_base_production', 'audit_lora_production')
LEGACY_CODE = {
    'recover_eval.py': '3ab2ed56736e0d58626a0b4afed850e5797dc5b27a1be1d7520433eee781c082',
    'cpu_adapter_entry.py': 'e1ca5773de70083fac1d31c647fdadab61aa35b3072ef7e56d7c18dba1a03ac1',
    'recovery_profile.json': '502fd6ab241d684fa88fa43de58b33c0d3346d26a14014b1e49a65bc27817bca',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inline_packer(shell):
    function = shell.split('build_bundle() {\n', 1)[1]
    return function.split("python - <<'PY'\n", 1)[1].split('\nPY\n}', 1)[0] + '\n'


def patch_packer(code):
    old = 'return first == "audit_bundle" or first.endswith("_audit_bundle")'
    if code.count(old) != 1:
        raise ValueError('Unsupported packaging copy policy')
    return code.replace(old, 'return first == "code" or first == "audit_bundle" or first.endswith("_audit_bundle")')


def bound_packer(tree, expected):
    script = Path(tree) / 'qwen3vl_local/sft_new_loop_phase2/eval.sh'
    if sha(script) != expected:
        raise ValueError('Frozen eval.sh differs from recovery package profile')
    return inline_packer(script.read_text())


def verify_hashes(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError('Reused evaluation evidence changed: ' + path)


def stage_existing(source, old, out, version, adapter, ready, profile):
    """Strictly admit the reported legacy recovery, then copy only finished evidence."""
    from qwen3vl_local.sft_new_loop_phase1 import delivery, run_integrity as proof
    source, old = Path(source).resolve(), Path(old).resolve()
    parts = source.relative_to(old).parts
    if (len(parts) != 5 or not parts[0].startswith('recovery_') or
            parts[1:] != ('pipeline', '2rgb_endpoints', '2rgb_endpoints', 'eval_review')):
        raise ValueError('Reuse requires this run original two-image recovery evaluation')
    previous = old / parts[0]
    if not (previous / 'failed.json').is_file():
        raise ValueError('Prior recovery has no failure receipt; may still be running')
    for name, expected in LEGACY_CODE.items():
        if sha(previous / 'code' / name) != expected:
            raise ValueError('Unsupported previous recovery supplement: ' + name)
    plan = proof.read(previous / 'recovery_plan.json')
    if (plan['old_run'] != str(old) or plan['version'] != str(version) or
            Path(plan['two_adapter']).resolve() != adapter.resolve() or
            plan['data'] != ready['data'] or plan['source_profile'] != profile):
        raise ValueError('Previous evaluation belongs to different run/data/source/adapter')
    identity = proof.verify_adapter(adapter)
    for name in NAMES:
        delivery.verified_eval(source / name)
        inputs = proof.read(source / name / 'evaluation_inputs.json')
        if (inputs['phase'] != 2 or inputs['history_rgb_mode'] != '2rgb_endpoints' or
                inputs['arguments']['split'] != 'test' or
                not proof.same_data(inputs['data'], ready['data']) or
                inputs['adapter_identity'] != (identity if name.startswith('lora_') else None)):
            raise ValueError('Reused evaluation identity differs: ' + name)
    for name in AUDITS:
        if not (source / name / 'summary.json').is_file():
            raise ValueError('Missing completed error audit: ' + name)
    traces = sorted((previous / 'traces').glob('adapter_read_*.jsonl'))
    records = [json.loads(line) for p in traces for line in p.read_text().splitlines()]
    if not records or any(r['torch_device'] != 'cpu' or Path(r['adapter']).resolve() != adapter.resolve() for r in records):
        raise ValueError('Missing/mismatched original adapter CPU loading traces')
    inputs = proof.read(adapter.parent / 'training_inputs.json')
    proof.verify_sources(inputs['source_sha256'], root=source / 'code')
    metadata = [source / 'dataset_receipt.json', source / 'pipeline_preflight.json']
    provenance = [previous / 'recovery_plan.json', previous / 'failed.json',
                  *(previous / 'code' / name for name in LEGACY_CODE), *traces]
    files = [p for name in (*NAMES, *AUDITS) for p in (source / name).rglob('*') if p.is_file()]
    files += metadata + provenance + [source / 'code' / name for name in inputs['source_sha256']]
    if any(p.is_symlink() for name in (*NAMES, *AUDITS) for p in (source / name).rglob('*')):
        raise ValueError('Symlinks are not allowed in reused evaluation evidence')
    hashes = {str(p): sha(p) for p in files}
    target = out / 'pipeline/2rgb_endpoints/2rgb_endpoints/eval_review'
    target.mkdir(parents=True, exist_ok=False)
    for name in (*NAMES, *AUDITS):
        shutil.copytree(source / name, target / name)
    copied = []
    for p in provenance:
        dest = out / 'reused_two_source' / p.relative_to(previous)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        copied.append(dest)
    verify_hashes(hashes)
    proof.write(out / 'reused_evaluation.json', dict(source=str(source), source_sha256=hashes,
                inference_repeated=False, note='Native full raw-output scoring and delivery checks required'))
    return target, metadata, hashes, copied


def package_environment(env, root, adapter, old, info, metadata):
    return {**env, 'PHASE_NAME': 'sft_new_loop_phase2', 'OUTPUT_ROOT': str(root),
        'ADAPTER_DIR': str(adapter), 'ADAPTER_INPUT': str(adapter),
        'ADAPTER_CONFIG_NAME': 'sft_new_loop_phase2_adapter_config.json',
        'BUNDLE_MAX_MB': '30', 'BUNDLE_BASENAME': f'sft_new_loop_phase2_{old.name}_2rgb_endpoints_audit_bundle',
        'TIMESTAMP': old.name, 'MODEL_DIR': info['model_dir'], 'INDEX': info['index'],
        'SPLIT': 'test', 'HISTORY_RGB_MODE': '2rgb_endpoints',
        'EVAL_SCRIPT': 'qwen3vl_local/sft_new_loop_phase2/eval.py',
        'RUN_AUDIT_PROMPT_EVAL': '1', 'RUN_VISUAL_AUDIT': '0',
        'DATASET_RECEIPT': str(metadata[0]), 'PREFLIGHT_RECEIPT': str(metadata[1])}


def main():
    args = sys.argv[1:]
    if not args or args[0] != '-':
        os.execv(sys.executable, [sys.executable, *args])
    code = sys.stdin.read()
    if 'def build_attempt(' in code:
        original = bound_packer(os.environ['PHASE12_RECOVERY_TREE'], os.environ['PHASE12_RECOVERY_EVAL_SHA'])
        if code != original:
            raise ValueError('Unknown inline packer; refusing unbound runtime modification')
        code = patch_packer(code)
        trace = dict(policy='copy_frozen_code_once',
                     original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                     executed_sha256=hashlib.sha256(code.encode()).hexdigest())
        dest = Path(os.environ['PHASE12_RECOVERY_TRACE']) / f'package_copy_{os.getpid()}.json'
        dest.write_text(json.dumps(trace, indent=2) + '\n')
        print('[package-recovery] copy frozen code once; native verification unchanged', flush=True)
    raise SystemExit(subprocess.run([sys.executable, *args], input=code, text=True).returncode)


if __name__ == '__main__':
    main()
