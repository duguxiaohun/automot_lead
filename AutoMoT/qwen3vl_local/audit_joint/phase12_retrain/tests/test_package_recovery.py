"""Real permission failure, native CPU score/receipt checks and packer execution.

Only the expensive VLM is substituted by the established Linear/output fixtures.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from qwen3vl_local.audit_joint.phase12_retrain import package_recovery as package
from qwen3vl_local.sft_new_loop_phase1 import delivery, run_integrity as proof
from qwen3vl_local.sft_new_loop_phase1.test_integrity import completed_fixture, eval_fixture


def evaluated(root, phase):
    index, model, run = completed_fixture(root / 'assets', phase)
    review = root / 'review'
    for name in package.NAMES:
        args = ['--index', str(index), '--data-root', str(root / 'assets'), '--model-dir', str(model),
                '--output-dir', str(review / name), '--no-timestamp-output', '--cases-per-bin', '4',
                '--no-save-error-rgb']
        args += (['--adapter-dir', str(run / 'final')] if name.startswith('lora_') else
                 ['--history-rgb-mode', '2rgb_endpoints'])
        if name.endswith('audit_prompt'):
            args.append('--audit-prompt')
        eval_fixture(phase, args)
    for name in package.AUDITS:
        (review / name).mkdir()
        (review / name / 'summary.json').write_text('{"fixture_only":true}')
    return index, model, run, review


def env_for(review, phase, index, model, run):
    return dict(PHASE_NAME=f'sft_new_loop_phase{phase}', OUTPUT_ROOT=str(review),
                ADAPTER_DIR=str(run / 'final'), ADAPTER_INPUT=str(run / 'final'),
                ADAPTER_CONFIG_NAME=f'sft_new_loop_phase{phase}_adapter_config.json',
                BUNDLE_MAX_MB='30', BUNDLE_BASENAME='fixture_audit_bundle', TIMESTAMP='fixture',
                MODEL_DIR=str(model), INDEX=str(index), SPLIT='test', HISTORY_RGB_MODE='2rgb_endpoints',
                RUN_AUDIT_PROMPT_EVAL='1', RUN_AUDIT_CASES='1', RUN_VISUAL_AUDIT='0',
                DATASET_RECEIPT='', PREFLIGHT_RECEIPT='')


@pytest.mark.parametrize('phase', [1, 2])
def test_real_readonly_failure_then_native_packer_and_retry(tmp_path, monkeypatch, phase):
    assert os.geteuid() != 0, 'Permission regression must execute as an unprivileged user'
    index, model, run, review = evaluated(tmp_path, phase)
    broken = tmp_path / 'broken'
    shutil.copytree(review, broken)
    real_prepare = delivery.prepare
    def readonly_prepare(*args, **kwargs):
        result = real_prepare(*args, **kwargs)
        for p in (Path(args[0]) / 'code').rglob('*'):
            if p.is_file():
                p.chmod(0o444)
        return result
    monkeypatch.setattr(delivery, 'prepare', readonly_prepare)
    shell = (proof.ROOT / f'qwen3vl_local/sft_new_loop_phase{phase}/eval.sh').read_text()
    code = package.inline_packer(shell)
    old_code = code.replace('return first == "code" or first == "audit_bundle"', 'return first == "audit_bundle"')
    for key, value in env_for(broken, phase, index, model, run).items():
        monkeypatch.setenv(key, value)
    with pytest.raises(shutil.Error, match='Permission denied'):
        exec(compile(old_code, '<legacy-packer>', 'exec'), {'__name__': '__main__'})
    # Same real records, new output; no model inference between failed and fixed packaging.
    for key, value in env_for(review, phase, index, model, run).items():
        monkeypatch.setenv(key, value)
    namespace = {'__name__': '__main__'}
    exec(compile(code, '<fixed-packer>', 'exec'), namespace)
    bundle = review / 'fixture_audit_bundle'
    assert delivery.verify_package(bundle)['status'] == 'verified'
    assert (review / 'fixture_audit_bundle.tar.gz').stat().st_size < 30 * 1024 * 1024
    assert (bundle / 'code/qwen3vl_local/qwen35/requirements.txt').stat().st_mode & 0o777 == 0o444
    # Exercise a second size-reduction attempt with read-only code already in the first bundle.
    namespace['build_attempt'](3, 1, 384, 40)
    assert delivery.verify_package(bundle)['status'] == 'verified'
    for name in package.NAMES:
        for p in (broken / name).glob('*.json*'):
            assert p.read_bytes() == (review / name / p.name).read_bytes()


def test_python_shim_bound_packer_and_passthrough(tmp_path):
    tree = tmp_path / 'tree'; target = tree / 'qwen3vl_local/sft_new_loop_phase2/eval.sh'
    target.parent.mkdir(parents=True)
    code = ('def build_attempt():\n    first="code"\n'
            '    return first == "audit_bundle" or first.endswith("_audit_bundle")\n'
            'assert build_attempt() is True\nprint("PACKED")\n')
    target.write_text("build_bundle() {\npython - <<'PY'\n" + code + 'PY\n}\n')
    trace = tmp_path / 'traces'; trace.mkdir()
    env = {**os.environ, 'PHASE12_RECOVERY_TREE': str(tree),
           'PHASE12_RECOVERY_EVAL_SHA': package.sha(target), 'PHASE12_RECOVERY_TRACE': str(trace)}
    cmd = [sys.executable, package.__file__, '-']
    result = subprocess.run(cmd, input=code, text=True, capture_output=True, env=env, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'PACKED' in result.stdout and len(list(trace.glob('package_copy_*.json'))) == 1
    for bad in [code + '# changed\n', code.replace('def build_attempt():', 'def build_attempt(changed=None):')]:
        result = subprocess.run(cmd, input=bad, text=True, capture_output=True, env=env, timeout=10)
        assert result.returncode != 0 and 'Unknown inline packer' in result.stderr
    result = subprocess.run(cmd + ['argument'], input='import sys; print(sys.argv[1])\n',
                            text=True, capture_output=True, env=env, timeout=10)
    assert result.returncode == 0 and result.stdout.strip() == 'argument'
    result = subprocess.run([sys.executable, package.__file__, '-c', 'print("FORWARD")'],
                            text=True, capture_output=True, env=env, timeout=10)
    assert result.returncode == 0 and result.stdout.strip() == 'FORWARD'
    target.write_text(target.read_text() + '# changed shell\n')
    result = subprocess.run(cmd, input=code, text=True, capture_output=True, env=env, timeout=10)
    assert result.returncode != 0 and 'differs from recovery' in result.stderr


def previous_recovery(tmp_path, monkeypatch):
    index, model, run, review = evaluated(tmp_path, 2)
    old = tmp_path / 'old'; previous = old / 'recovery_fixture'
    source = previous / 'pipeline/2rgb_endpoints/2rgb_endpoints/eval_review'
    source.parent.mkdir(parents=True)
    shutil.copytree(review, source)
    data = proof.data_identity(index)
    receipt = source.parent / 'dataset_receipt.json'; proof.write(receipt, data)
    preflight = source.parent / 'preflight.json'; proof.write(preflight, {'dataset': data})
    delivery.prepare(source, 2, run / 'final', index, dataset_receipt=receipt, preflight=preflight)
    version = tmp_path / 'version'
    profile = {'fixture_profile': 'bound'}
    (previous / 'code').mkdir(); hashes = {}
    for name in package.LEGACY_CODE:
        p = previous / 'code' / name; p.write_text('explicit historical-code fixture: ' + name)
        hashes[name] = package.sha(p)
    monkeypatch.setattr(package, 'LEGACY_CODE', hashes)
    (previous / 'traces').mkdir()
    proof.write(previous / 'failed.json', {'error': 'fixture packaging failure'})
    proof.write(previous / 'recovery_plan.json', dict(old_run=str(old), version=str(version),
                two_adapter=str(run / 'final'), data=data, source_profile=profile))
    (previous / 'traces/adapter_read_1.jsonl').write_text(json.dumps(dict(torch_device='cpu', adapter=str(run / 'final'))) + '\n')
    return source, old, version, run / 'final', dict(data=data, index=str(index)), profile, model


def test_reuse_real_native_receipts_without_inference(tmp_path, monkeypatch):
    source, old, version, adapter, ready, profile, model = previous_recovery(tmp_path, monkeypatch)
    before = {str(p): package.sha(p) for p in old.rglob('*') if p.is_file()}
    out = tmp_path / 'new'
    target, metadata, hashes, provenance = package.stage_existing(source, old, out, version, adapter, ready, profile)
    assert provenance and hashes
    assert not (target / 'code').exists() and not (target / 'verified_training').exists()
    for name in package.NAMES:
        assert delivery.verified_eval(source / name) == delivery.verified_eval(target / name)
    # Run the actual repack command/shim/native packer in a child, with the original
    # double-copy policy in its bound shell. No inference or verifier is replaced.
    tree = tmp_path / 'frozen_packer'
    script = tree / 'qwen3vl_local/sft_new_loop_phase2/eval.sh'
    script.parent.mkdir(parents=True)
    native = (proof.ROOT / 'qwen3vl_local/sft_new_loop_phase2/eval.sh').read_text()
    script.write_text(native.replace('return first == "code" or first == "audit_bundle"', 'return first == "audit_bundle"'))
    body = out / 'repack_two.py'; body.write_text(package.bound_packer(tree, package.sha(script)))
    (out / 'bin').mkdir(); (out / 'traces').mkdir()
    import shlex
    shim = out / 'bin/python'
    shim.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(package.__file__)+' "$@"\n')
    shim.chmod(0o755)
    env = package.package_environment({**os.environ, 'PATH':str(out / 'bin') + os.pathsep + os.environ['PATH'],
        'PYTHONPATH':str(proof.ROOT), 'PHASE12_RECOVERY_TREE':str(tree),
        'PHASE12_RECOVERY_EVAL_SHA':package.sha(script), 'PHASE12_RECOVERY_TRACE':str(out / 'traces')},
        target, adapter, old, {'model_dir':str(model), 'index':ready['index']}, metadata)
    result = subprocess.run(['bash', '-c', 'python - < "$1"', 'repack', str(body)],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    bundle = target / env['BUNDLE_BASENAME']
    assert delivery.verify_package(bundle)['status'] == 'verified'
    assert bundle.with_name(bundle.name + '.tar.gz').is_file()
    assert len(list((out / 'traces').glob('package_copy_*.json'))) == 1
    package.verify_hashes(before)
    # Post-admission tampering must also fail.
    p = source / 'lora_production/metrics.json'; p.write_text('{}')
    with pytest.raises(ValueError, match='evidence changed'):
        package.verify_hashes(hashes)


@pytest.mark.parametrize('bad', ['missing_receipt', 'raw_output', 'wrong_adapter', 'previous_code', 'wrong_plan'])
def test_reuse_rejects_incomplete_or_mismatched_evidence(tmp_path, monkeypatch, bad):
    source, old, version, adapter, ready, profile, _ = previous_recovery(tmp_path, monkeypatch)
    if bad == 'missing_receipt':
        (source / 'lora_audit_prompt/evaluation_complete.json').unlink()
    elif bad == 'raw_output':
        path = source / 'lora_production/cases.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['raw_output'] = 'CHANGED'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    elif bad == 'wrong_adapter':
        adapter = tmp_path / 'wrong_adapter'
    elif bad == 'previous_code':
        (old / 'recovery_fixture/code/cpu_adapter_entry.py').write_text('changed')
    else:
        plan = old / 'recovery_fixture/recovery_plan.json'
        record = proof.read(plan); record['version'] = 'other'; proof.write(plan, record)
    out = tmp_path / 'new'
    with pytest.raises((ValueError, FileNotFoundError)):
        package.stage_existing(source, old, out, version, adapter, ready, profile)
    assert not out.exists()
