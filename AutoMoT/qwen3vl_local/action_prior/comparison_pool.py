"""搜索用常驻GPU子进程：同模型跨批复用，换模型释放旧上下文。"""
from collections import deque
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from qwen3vl_local.action_prior.comparison_cases import read_json, write_json
from qwen3vl_local.action_prior.comparison_scheduler import worker_environment, _stop_groups


def worker_service():
    from qwen3vl_local.action_prior.comparison_runtime import evaluate_worker
    cache = {}
    for line in sys.stdin:
        request = json.loads(line)
        job = read_json(request['job'])
        evaluate_worker(job, cache=cache)
        write_json(request['done'], dict(status='complete'))


class SearchPool:
    """一卡一服务，模型亲和派发；失败或外部中断回收所有本次进程组。"""
    def __init__(self, out, gpu_plan, command=None):
        self.out, self.plan = Path(out), gpu_plan
        self.command = command or [sys.executable, str(Path(__file__).with_name('compare_checkpoints.py')), '--worker-service']
        self.active, self.records, self.previous = {}, [], {}

    def __enter__(self):
        def interrupted(signum, frame):
            raise SystemExit(128+signum)
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                self.previous[signum] = signal.getsignal(signum)
                signal.signal(signum, interrupted)
            for index, gpu in enumerate(self.plan['selected_ids']):
                path = self.out/'logs'/f'search_worker_{index:02d}.log'
                path.parent.mkdir(parents=True, exist_ok=True)
                log = path.open('w', encoding='utf-8')
                try:
                    process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=log,
                        stderr=subprocess.STDOUT, text=True, env=worker_environment(gpu), start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                self.active[index] = dict(process=process, log=log, gpu=gpu, model=None, task=None, log_path=str(path))
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def save(self, status):
        write_json(self.out/'scheduler.json', dict(status=status, **self.plan, jobs=self.records,
            workers=[dict(gpu=w['gpu'], pid=w['process'].pid, model=w['model'], log=w['log_path']) for w in self.active.values()]))

    def run(self, tasks):
        pending = deque(tasks)
        last = 0.
        while pending or any(w['task'] for w in self.active.values()):
            for worker in self.active.values():
                code = worker['process'].poll()
                if code is not None:
                    raise RuntimeError(f"search worker GPU={worker['gpu']} exited {code}; log={worker['log_path']}")
                if worker['task']:
                    if Path(worker['task']['done']).is_file():
                        if read_json(worker['task']['done'])['status'] != 'complete':
                            raise RuntimeError('search worker returned invalid completion')
                        worker['task']['status'] = 'complete'
                        worker['task'] = None
                    else:
                        continue
                if pending:
                    task = next((t for t in pending if t['model'] == worker['model']), pending[0])
                    pending.remove(task)
                    done = str(Path(task['job']).with_suffix('.done.json'))
                    if Path(done).exists():
                        raise ValueError('search completion path already exists')
                    job = read_json(task['job'])
                    job['runtime_output'] = str(self.out/'_runtime'/f"gpu_{worker['gpu']}"/task['model'])
                    write_json(task['job'], job)
                    record = dict(task, gpu=worker['gpu'], pid=worker['process'].pid, log=worker['log_path'], done=done, status='running')
                    self.records.append(record)
                    worker['model'], worker['task'] = task['model'], record
                    worker['process'].stdin.write(json.dumps(dict(job=task['job'], done=done))+'\n')
                    worker['process'].stdin.flush()
                    print(f"[search] dispatch {task['id']} GPU={worker['gpu']} cases={task['samples']}", flush=True)
            if time.monotonic()-last >= 15:
                self.save('searching')
                print(f"[search] running={sum(bool(w['task']) for w in self.active.values())} pending={len(pending)}; {self.out/'search.json'}", flush=True)
                last = time.monotonic()
            time.sleep(.05)
        self.save('batch_complete')

    def __exit__(self, typ, exc, tb):
        for signum in self.previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if typ is not None:
                for record in self.records:
                    if record['status'] == 'running':
                        record['status'] = 'cancelled'
            try:
                self.save('failed' if typ else 'evaluated')
            finally:
                # 即使状态写盘失败，也必须回收GPU服务及其loader进程组。
                _stop_groups(self.active, timeout=5.)
                for worker in self.active.values():
                    try:
                        worker['process'].stdin.close()
                    except BrokenPipeError:
                        pass
        finally:
            for signum, handler in self.previous.items():
                signal.signal(signum, handler)
