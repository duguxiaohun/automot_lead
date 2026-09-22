"""对比预检的阶段心跳：只观测，不改训练代码或内容校验。"""
from contextlib import contextmanager
from pathlib import Path
import resource
import sys
import threading
import time

from qwen3vl_local.action_prior.comparison_cases import write_json


class PreflightProgress:
    """耗时调用期间每15秒发布耗时、RSS和调用位置，避免CPU预检静默。"""
    def __init__(self, out, interval=15.):
        self.out = Path(out)
        self.interval = interval
        self.started = time.monotonic()
        self.history = []

    @contextmanager
    def stage(self, name):
        started = time.monotonic()
        owner = threading.get_ident()
        stopped = threading.Event()
        def report(status):
            now = time.monotonic()
            rss = None
            try:
                for line in Path('/proc/self/status').read_text().splitlines():
                    if line.startswith('VmRSS:'):
                        rss = round(int(line.split()[1]) / 1024., 1)
                        break
            except OSError:
                pass
            frame = sys._current_frames().get(owner)
            stack = []
            while frame is not None:
                if frame.f_code.co_filename != __file__:
                    stack.append(f'{Path(frame.f_code.co_filename).name}:{frame.f_lineno}:{frame.f_code.co_name}')
                frame = frame.f_back
            value = dict(status=status, stage=name, stage_seconds=round(now-started, 1),
                         total_seconds=round(now-self.started, 1), rss_mb=rss,
                         cpu_seconds=round(resource.getrusage(resource.RUSAGE_SELF).ru_utime, 1),
                         location=stack[:4], completed_stages=self.history.copy())
            write_json(self.out / 'preflight.json', value)
            detail = ' / '.join(stack[:2]) if status == 'running' else ''
            message = f'[preflight] {status} {name} | {value["stage_seconds"]:.1f}s | RSS={rss}MB {detail}'
            print(message, flush=True)
            with (self.out / 'preflight.log').open('a', encoding='utf-8') as log:
                log.write(message+'\n')
        def heartbeat():
            while not stopped.wait(self.interval):
                report('running')
        report('start')
        thread = threading.Thread(target=heartbeat, name='comparison-preflight-progress', daemon=True)
        thread.start()
        status = 'done'
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            stopped.set()
            thread.join()
            self.history.append(dict(stage=name, status=status, seconds=round(time.monotonic()-started, 1)))
            report(status)
