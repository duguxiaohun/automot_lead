"""Pinned CPU continuation with race-safe process observation and one launcher lock."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def process_stat(pid, proc_root=Path('/proc')):
    # Read once: existence checks cannot make /proc reads atomic with exit.
    try:raw=(Path(proc_root)/str(pid)/'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):return None
    # comm can contain spaces and parentheses; split after its final ')'.
    prefix,sep,tail=raw.rpartition(')')
    fields=tail.split()
    if not sep or len(fields)<20 or not fields[19].isdigit():raise ValueError('malformed_process_stat')
    return dict(state=fields[0],start_time=fields[19])


def wait_for_exit(pid, start_time, *, observe=process_stat, pause=time.sleep):
    while True:
        current=observe(pid)
        if current is None or current['start_time']!=start_time or current['state'] in ('Z','X'):return
        pause(10)


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(job):
    for path,expected in job['sources'].items():
        if file_sha(path)!=expected:raise ValueError('pinned_continuation_source_changed: '+path)


def write_status(path, **values):
    path=Path(path)
    values.update(supervisor_pid=os.getpid(),updated_unix=time.time(),GPU=False)
    temporary=path.with_name(path.name+f'.{os.getpid()}.pending')
    with temporary.open('x') as stream:
        json.dump(values,stream,indent=2,ensure_ascii=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def run(job_path, *, observe=process_stat, pause=time.sleep, launch=subprocess.run):
    job_path=Path(job_path);raw=job_path.read_bytes();job=json.loads(raw)
    job_sha256=hashlib.sha256(raw).hexdigest()
    if job.get('schema')!='study_cpu_continuation_v2':raise ValueError('invalid_continuation_job')
    if job['module']!='qwen3vl_local.audit_joint.geometry_study_cpu_audit_v2':raise ValueError('unexpected_CPU_entry')
    status=Path(job['status']);output=Path(job['output']);log=Path(job['log'])
    def save(**values):
        write_status(status,native_pid=job['native_pid'],native_start_time=job['native_start_time'],
                     job_sha256=job_sha256,job=str(job_path),**values)
    with job_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # A repeated invocation must preserve the prior result, not relabel it
        # failed merely because the protected output already exists.
        if output.exists() or log.exists():raise FileExistsError('continuation_output_exists_no_duplicate_launch')
        try:
            verify_sources(job)
            save(stage='waiting_for_native',status='running')
            wait_for_exit(job['native_pid'],job['native_start_time'],observe=observe,pause=pause)
            # Missing process is normal, but a missing receipt is not success.
            ready_path=Path(job['native_ready'])
            if not ready_path.is_file():raise ValueError('native_ended_without_ready_receipt')
            ready=json.loads(ready_path.read_text())
            dataset=Path(ready['dataset']).resolve(strict=True)
            if not dataset.is_file():raise ValueError('native_ready_dataset_not_file')
            if file_sha(job_path)!=job_sha256:raise ValueError('continuation_job_changed')
            verify_sources(job)
            command=[sys.executable,'-u','-m',job['module'],'--geometry',str(dataset),'--maps',job['maps'],
                     '--study-root',job['study_root'],'--output',str(output),'--directions']
            save(stage='study_prepare_and_audit',status='running',command=command,
                 native_ready_sha256=file_sha(ready_path))
            with log.open('x') as stream:
                result=launch(command,cwd=job['app'],env=dict(os.environ,PYTHONPATH=job['app'],
                    CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
                    stdout=stream,stderr=subprocess.STDOUT)
            report=output/'status.json';audit=json.loads(report.read_text()) if report.exists() else {}
            outcome='failed'
            if audit.get('status')=='completed':
                if result.returncode==0 and audit.get('formal_conditions_passed') is True and audit.get('ready_status')=='ready':outcome='admitted'
                elif result.returncode==2:outcome='rejected'
            save(stage='finished',status=outcome,exit_code=result.returncode,audit_status=audit,training_started=False)
            return result.returncode if outcome!='failed' else 1
        except Exception as exc:
            save(stage='continuation',status='failed',error=f'{type(exc).__name__}: {exc}')
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args();return run(args.job)


if __name__=='__main__':raise SystemExit(main())
