#!/usr/bin/env python3
"""Continue an explicit Phase3 epoch checkpoint in a verified private runtime.

Standalone standard-library launcher. Never edits the original run, dataset,
working source or model. GPU workers use the original training_state reader.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import zipfile

POLICY = 'phase3_explicit_continuation_v1'
LEGACY_SHA = '8477a5333b0da3e04ff84366045bd73824a5327e1c824e470c275c1cd254886e'
LEGACY_TEST_SHA = '07eed103948a457671bc36cbd160fc643dd9970214728c83b4ebc9149903c8d7'
SOURCE_SUFFIXES = {'.py', '.sh', '.json', '.yaml', '.yml', '.txt', '.jinja', '.md'}
RULE_JSONL_PACKAGES = ('sft_new_loop_phase2', 'sft_new_loop_phase3')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.pending')
    with temp.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def local(root, name):
    path = Path(name)
    return (path if path.is_absolute() else root / path).resolve(strict=True)


def member(folder, name):
    path = folder / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError('receipt member escapes directory: ' + name)
    return path


def verify_files(folder, expected):
    for name in expected:
        member(folder, name)
    for p in folder.rglob('*'):
        if p.is_symlink():
            raise ValueError('checkpoint must not contain symlink members')
    actual = {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*')
              if p.is_file() and p.name != 'receipt.json'}
    if actual != expected:
        raise ValueError('checkpoint files missing, extra or changed: ' + str(folder))


def gpu_ids(value):
    if not value:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=15)
        devices = [tuple(int(x.strip()) for x in line.split(',')) for line in rows.splitlines() if line.strip()]
        value = ','.join(str(d[0]) for d in sorted(devices, key=lambda d: (d[1], d[2], d[0]))[:4])
    ids = value.split(',')
    if len(ids) != 4 or any(not s.isascii() or not s.isdigit() for s in ids) or len({int(s) for s in ids}) != 4:
        raise ValueError('recovery requires exactly four distinct GPU_IDS')
    return ','.join(str(int(s)) for s in ids)


def sources(runtime):
    root = runtime / 'qwen3vl_local'
    paths = {p for name in ('sft_new_loop_phase3', 'sft_v2', 'qwen35')
             for p in (root/name).rglob('*.py') if not p.name.startswith('test_') and 'tests' not in p.parts}
    paths.update(root/name for name in ('sequence_batch.py', 'batch_runtime.py', 'image_io.py', 'mrope_utils.py', 'cache_utils.py'))
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}


def prepare_runtime(root, runtime, binding, pipeline):
    runtime.mkdir()
    source = root/'qwen3vl_local'
    rule_directories = {source/name for name in RULE_JSONL_PACKAGES}
    # Preserve server-only source files, not just files present in a Git commit.
    for p in source.rglob('*'):
        rule_asset = p.suffix == '.jsonl' and p.parent in rule_directories
        if not p.is_file() or not (p.suffix in SOURCE_SUFFIXES or rule_asset) or '__pycache__' in p.parts:
            continue
        if p.stat().st_size > 16*1024**2:
            raise ValueError('unexpected oversized source file: ' + str(p))
        target = runtime/'qwen3vl_local'/p.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    target = runtime/'qwen3vl_local/sequence_batch.py'
    expected = binding['sources']['sequence_batch.py']
    if sha(target) != expected:
        legacy = Path(__file__).with_name('phase3_legacy_sequence.py')
        if expected != LEGACY_SHA or sha(legacy) != LEGACY_SHA:
            raise ValueError('no exact legacy sequence source for this checkpoint')
        shutil.copyfile(legacy, target)
    if sources(runtime) != binding['sources']:
        raise ValueError('private runtime differs from checkpoint source set/SHA; original source required')
    p3 = runtime/'qwen3vl_local/sft_new_loop_phase3'
    test_name = 'test_epoch_consumption.py'
    test_expected = pipeline['source_sha256'].get(test_name)
    if test_expected and sha(p3/test_name) != test_expected:
        # The original pipeline also hashed tests. Later fixture-only changes
        # must not be ignored or falsely treated as changed training semantics.
        legacy = Path(__file__).with_name('phase3_legacy_epoch_fixture.py')
        if test_expected != LEGACY_TEST_SHA or sha(legacy) != LEGACY_TEST_SHA:
            raise ValueError('no exact historical epoch test for original pipeline')
        shutil.copyfile(legacy, p3/test_name)
    actual = {p.name: sha(p) for p in p3.iterdir() if p.suffix in ('.py', '.sh', '.json')}
    if actual != pipeline['source_sha256']:
        raise ValueError('pipeline/evaluation source differs from original run')
    # Absolute and relative configuration strings stay unchanged. Assets resolve
    # back to their original inode/path, as required by the native base binding.
    for p in root.iterdir():
        if p.name != 'qwen3vl_local':
            (runtime/p.name).symlink_to(p.resolve(), target_is_directory=p.is_dir())
    return {str(p.relative_to(runtime)): sha(p) for p in (runtime/'qwen3vl_local').rglob('*') if p.is_file()}


def completed(root, run, mode):
    run = local(root, run)
    pipeline = read(run/'pipeline_manifest.json')
    package = read(run/'packaging_manifest.json')
    metrics = read(run/'eval/lora_production/metrics.json')
    if (pipeline.get('group') != '2rgb_short_'+mode or
            pipeline.get('status') not in ('completed', 'completed_without_historical_regression') or
            package.get('status') != 'completed' or
            (metrics.get('history_rgb_mode'), metrics.get('action_output_mode')) != ('2rgb_short', mode)):
        raise ValueError('previous two-image group is not complete: ' + str(run))
    # Existing archives are explicit paths, never discovered by newest timestamp.
    if not package.get('packages'):
        raise ValueError('completed group has no package')
    for item in package['packages']:
        archive = local(root, item['archive'])
        if archive.stat().st_size != item['bytes']:
            raise ValueError('old audit archive missing/changed size: ' + str(archive))
    return dict(run=str(run), status='retained_without_rerun',
                pipeline_sha256=sha(run/'pipeline_manifest.json'),
                package_sha256=sha(run/'packaging_manifest.json'), metrics_sha256=sha(run/'eval/lora_production/metrics.json'))


def execute(command, *, cwd, env, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print('[stage] '+log.stem+' '+str(log), flush=True)
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        try:
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f'[running] {log.stem} pid={process.pid} elapsed={time.monotonic()-started:.0f}s log={log}', flush=True)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f'{log.stem} exited {code}; see {log}')


def validate_dataset(job):
    """Run the native full index/semantic contract checks before any GPU worker.

    Check both output modes without building data, changing hashes or loading
    Qwen. This also resolves non-Python dependencies of the frozen mapping code.
    """
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_index
    config = job['binding']['configuration']
    reports = {}
    for mode in ('choice', 'binary'):
        reports[mode] = check_index(config['index'], action_output_mode=mode,
                                   prompt_variant=config.get('prompt_variant', 'baseline'))
    print(json.dumps(dict(status='native_dataset_preflight_passed', modes=reports)), flush=True)
    return reports


def worker(job_path, preflight=False):
    job = read(job_path)
    runtime = Path(job['runtime'])
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    for name, expected in job.get('runtime_rule_assets', {}).items():
        if sha(member(runtime, name)) != expected:
            raise ValueError('runtime rule asset changed: '+name)
    from qwen3vl_local.sft_new_loop_phase3 import training_state as recovery
    from argparse import Namespace
    original = job['binding']
    if recovery.binding(Namespace(**original['configuration']), 4, original['steps_per_epoch']) != original:
        raise ValueError('native source/config/data/base/world binding changed')
    if sha(Path(original['configuration']['index'])) != original['configuration']['training_index_sha256']:
        raise ValueError('index changed')
    if preflight:
        state = recovery.read(Path(job['checkpoint']), original)
        print(json.dumps(dict(status='native_checkpoint_read_passed', next_epoch=state['next_epoch'],
                              global_step=state['global_step'])), flush=True)
        validate_dataset(job)
        return
    from qwen3vl_local.sft_new_loop_phase3 import train
    sys.argv = [sys.argv[0]]
    args = train.parse_args()
    for k, v in original['configuration'].items():
        setattr(args, k, v)
    args.action_output_mode = job['mode']
    args.output_dir = job['output']
    args.resume = job['checkpoint'] if job['mode'] == 'choice' else None
    args.save_training_state = True
    args.stop_after_epoch = None
    args.device = 'cuda'
    args.ddp_timeout_seconds = 3600
    args.no_tb = True
    train.train(args)


def handoff(output, state):
    write(output/'workflow.json', state)
    files = {'workflow.json': (output/'workflow.json').read_bytes()}
    for p in sorted(output.rglob('*')):
        if not p.is_file() or 'runtime' in p.relative_to(output).parts or 'tool_repo' in p.parts:
            continue
        # Metadata only. No raw image, model, tensor, full dataset or environment.
        if p.name in ('metrics.json', 'packaging_manifest.json', 'dataset_reference.json') and p.stat().st_size < 2*1024**2:
            files[str(p.relative_to(output))] = p.read_bytes()
        elif p.suffix == '.log' or p.name in ('stdout.log', 'stderr.log'):
            with p.open('rb') as stream:
                stream.seek(max(0, p.stat().st_size-60000))
                files[str(p.relative_to(output))] = stream.read()
    dest = output/'handoff'; dest.mkdir(exist_ok=True)
    archive = dest/'phase3_continue_report.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    write(dest/'phase3_continue_package.json', dict(status=state['status'], archive_sha256=sha(archive),
        bytes=archive.stat().st_size, files={k: hashlib.sha256(v).hexdigest() for k, v in files.items()}))
    print('[handoff] '+str(dest), flush=True)


def run(args):
    root = args.root.resolve(strict=True)
    output = args.output.absolute()
    checkpoint = local(root, args.checkpoint)
    failed = checkpoint.parent.parent.parent
    receipt = read(checkpoint/'receipt.json')
    config = receipt['binding']['configuration']
    protected = [failed, local(root, args.completed_choice), local(root, args.completed_binary),
                 local(root, config['index']).parent, local(root, config['data_root']),
                 local(root, config['model_dir'])]
    if output.exists() or output.is_symlink() or any(output.resolve().is_relative_to(p) for p in protected):
        raise ValueError('output must be a new directory outside the original pipeline')
    output.mkdir(parents=True)
    state = dict(policy=POLICY, status='preflight', root=str(root), hostname=socket.gethostname(),
                 server_label=os.environ.get('AUTOMOT_SERVER', 'unspecified'),
                 checkpoint=str(checkpoint), output=str(output), stages=[],
                 source_sha256=sha(Path(__file__)), data_rebuilt=False, historical_regression=False)
    fd = os.open(checkpoint, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state['retained'] = [completed(root, args.completed_choice, 'choice'),
                             completed(root, args.completed_binary, 'binary')]
        binding = receipt['binding']; config = binding['configuration']
        if (receipt.get('policy') != 'phase3_epoch_training_state_v1' or binding['world_size'] != 4 or
                config['history_rgb_mode'] != '4rgb' or config['action_output_mode'] != 'choice' or
                config['num_epochs'] != 3 or config['sampling_only'] or config['max_steps']):
            raise ValueError('this continuation requires the original three-epoch four-image choice plan')
        state['active_stage'] = 'checkpoint_files_verify'
        write(output/'workflow.json', state)
        print('[preflight] verifying complete checkpoint file SHA: '+str(checkpoint), flush=True)
        verify_files(checkpoint, receipt['files'])
        pipeline = read(failed/'pipeline_manifest.json')
        if pipeline.get('historical_regression_enabled'):
            raise ValueError('historical-regression runs require their original replay inputs')
        index = local(root, config['index'])
        if (sha(index) != config['training_index_sha256'] or
                config['training_index_sha256'] != pipeline['data_identity']['index_sha256'] or
                sha(index.with_name('manifest.json')) != pipeline['data_identity']['manifest_sha256']):
            raise ValueError('original dataset changed')
        if shutil.disk_usage(output).free < 21*1024**3:
            raise ValueError('less than 21 GiB available; native epoch save also enforces its dynamic 20 GiB reserve')
        runtime = output/'runtime'
        print('[preflight] preserving original source in private runtime: '+str(runtime), flush=True)
        state['runtime_sources'] = prepare_runtime(root, runtime, binding, pipeline)
        ids = gpu_ids(args.gpu_ids)
        state['gpu_ids'] = ids
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=ids, GPU_IDS=ids, PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=str(runtime), PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH', ''),
                   TOKENIZERS_PARALLELISM='false', TORCH_NCCL_TRACE_BUFFER_SIZE='8192',
                   TORCH_NCCL_DUMP_ON_TIMEOUT='1', TORCH_NCCL_DESYNC_DEBUG='1', NCCL_DEBUG='WARN')
        if __package__:
            from .dataset_catalog import register
        else:
            from dataset_catalog import register
        state['dataset'] = register(index, root/'checkpoints/dataset_catalog', 'phase3',
                                    producer_sources=binding['sources'], root=root, origin='existing_original_run')
        write(output/'dataset_reference.json', state['dataset'])
        script = str(Path(__file__).resolve())

        def stage(name, command, extra=None):
            if sources(runtime) != binding['sources']:
                raise ValueError('private runtime changed between stages')
            state['status'] = 'running'; state['active_stage'] = name
            write(output/'workflow.json', state)
            started = time.time()
            execute(command, cwd=runtime, env=dict(env, **(extra or {})), log=output/(name+'.log'))
            state['stages'].append(dict(name=name, exit_code=0, elapsed_seconds=round(time.time()-started, 2)))
            write(output/'workflow.json', state)

        for mode in ('choice', 'binary'):
            group = output/('4rgb_'+mode); group.mkdir()
            group_state = dict(group='4rgb_'+mode, prompt_variant=config.get('prompt_variant', 'baseline'),
                status='data_checked', automatic_promotion=False, audits_enabled=True,
                historical_regression_enabled=False, source_sha256=pipeline['source_sha256'],
                index=str(index), run_root=str(group/'train'),
                data_identity={k:pipeline['data_identity'][k] for k in ('index_sha256', 'manifest_sha256')},
                continuation_source=str(failed), resume_from=str(checkpoint) if mode == 'choice' else None)
            write(group/'pipeline_manifest.json', group_state)
            job = dict(runtime=str(runtime), binding=binding, checkpoint=str(checkpoint), mode=mode,
                       output=str(group/'train'), runtime_rule_assets={
                           k:v for k,v in state['runtime_sources'].items() if k.endswith('.jsonl')})
            job_path = group/'job.json'; write(job_path, job)
            if mode == 'choice':
                stage('checkpoint_cpu_verify', [sys.executable, script, '--worker', str(job_path), '--preflight'])
            stage(mode+'_train', [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node=4', '--max_restarts=0', '--log-dir', str(group/'rank_logs'), '--tee', '3',
                '--', script, '--worker', str(job_path)],
                {'TORCH_NCCL_DEBUG_INFO_TEMP_FILE': str(group/'nccl_trace_rank_')})
            group_state['status'] = 'trained_without_test'
            write(group/'pipeline_manifest.json', group_state)
            # Native eval resolves the restored best/final/fallback adapter itself.
            stage(mode+'_test_audit', ['bash', 'qwen3vl_local/sft_new_loop_phase3/eval.sh'], dict(
                ADAPTER_DIR=str(group/'train'), OUTPUT_ROOT=str(group/'eval'), INDEX=config['index'],
                DATA_ROOT=config['data_root'], MODEL_DIR=config['model_dir'], SPLIT='test',
                CASES_PER_BIN='0', MAX_EVAL_FRAMES='0', MAX_NEW_TOKENS='256', RUN_BASE_EVAL='1',
                RUN_AUDITS='1', RUN_VISUAL_AUDIT='1', RUN_AUDIT_PROMPT_EVAL='auto',
                EXCLUDE_CASES_JSONL='', EXPECTED_EXCLUDED_CASES='0', EXPECTED_TOTAL_CASES='0',
                ROUTE_DIVERSE_SAMPLING='0', REQUIRE_INVALID_COVERAGE='1',
                EVAL_BATCH_SIZE='2', EVAL_GPU_COUNT='4', REUSE_FROZEN_RGB='1', SCAN_VISUAL_RISKS='0'))
            group_state['status'] = 'completed_without_historical_regression'
            group_state['regression_gate_passed'] = None
            write(group/'pipeline_manifest.json', group_state)
            stage(mode+'_package', [sys.executable, '-m', 'qwen3vl_local.sft_new_loop_phase3.pack_results',
                str(group/'eval'), '--output-dir', str(group/'eval'), '--receipt', str(group/'packaging_manifest.json')])
        state['status'] = 'completed_without_historical_regression'
        state['active_stage'] = None
    except BaseException as exc:
        state['status'] = 'failed_or_interrupted'; state['error'] = str(exc)
        traceback.print_exc()
        return 2
    finally:
        os.close(fd)
        handoff(output, state)
    return 0


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, resource.getrlimit(resource.RLIMIT_CORE)[1]))
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--preflight', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--completed-choice', type=Path)
    parser.add_argument('--completed-binary', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu-ids', default=os.environ.get('GPU_IDS', ''))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.preflight)
        return 0
    if any(getattr(args, key) is None for key in ('root', 'checkpoint', 'completed_choice', 'completed_binary', 'output')):
        parser.error('root, checkpoint, completed-choice, completed-binary and output are required')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
