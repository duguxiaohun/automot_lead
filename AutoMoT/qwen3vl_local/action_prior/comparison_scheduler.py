"""对比专用GPU任务队列；不改训练launcher或训练源码合同。"""
from collections import deque
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import time

from qwen3vl_local.action_prior.comparison_cases import write_json


def select_gpus(requested=4, model_count=1):
    """自动按空闲程度排序并降并发；GPU_IDS显式pin时以其数量为准。"""
    if requested < 1 or model_count < 1:
        raise ValueError("GPU数量和checkpoint数量必须为正整数")
    explicit = os.environ.get("GPU_IDS", "").strip()
    if explicit:
        tokens = [token.strip() for token in explicit.split(",")]
        if not all(token.isascii() and token.isdecimal() for token in tokens):
            raise ValueError("GPU_IDS必须为不重复的非负整数卡号，例如0,1,2,3")
        ids = [str(int(token)) for token in tokens]
        if len(set(ids)) != len(ids):
            raise ValueError("GPU_IDS含重复卡号")
        source = "GPU_IDS"
        detected = None
    else:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                 "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
        entries = []
        for line in result.stdout.splitlines():
            if line.strip():
                fields = tuple(int(value.strip()) for value in line.split(","))
                if len(fields) != 3:
                    raise ValueError("nvidia-smi返回的GPU状态格式错误")
                entries.append(fields)
        if not entries:
            raise ValueError("未检测到GPU，无法运行checkpoint推理")
        if len({v[0] for v in entries}) != len(entries):
            raise ValueError("nvidia-smi返回重复GPU卡号")
        entries.sort(key=lambda item: (item[1], item[2], item[0]))
        ids = [str(item[0]) for item in entries[:requested]]
        source, detected = "nvidia-smi", len(entries)
    selected = ids[:model_count]
    return dict(policy="one_checkpoint_per_gpu", source=source, requested_gpus=requested,
                detected_gpus=detected, candidate_ids=ids, selected_ids=selected,
                parallel_models=len(selected), checkpoint_count=model_count)


def worker_environment(gpu):
    """子进程只见分配的一张卡；不继承外部DDP rank，也不修改父进程环境。"""
    env = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
                "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), GPU_IDS=str(gpu), ACTION_PRIOR_GPU_READY="1",
               PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    return env


def _stop_groups(active, timeout):
    """终止本次启动的进程组（含数据loader）；宽限到期强制回收。"""
    for task in active.values():
        try:
            os.killpg(task["process"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    for task in active.values():
        process = task["process"]
        try:
            process.wait(timeout=max(0., deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    # 主worker已退出时loader仍可能存活，因此仍向原进程组发KILL。
    for task in active.values():
        try:
            os.killpg(task["process"].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        task["process"].wait()
        task["log"].close()


def run_queue(commands, gpu_plan, out, *, poll_interval=.2, stop_timeout=10.):
    """空出的GPU立即领取下一个checkpoint；任一失败即停止派发并回收其余worker。"""
    out = Path(out)
    gpu_ids = gpu_plan["selected_ids"]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("任务队列需要不重复且非空的GPU列表")
    pending, free, active = deque(range(len(commands))), deque(gpu_ids), {}
    records = [dict(model=f"model_{i+1:02d}", status="pending", gpu=None,
                    log=str(out / "logs" / f"model_{i+1:02d}.log")) for i in range(len(commands))]
    def save(status):
        write_json(out / "scheduler.json", dict(status=status, **gpu_plan, jobs=records))
        write_json(out / "status.json", dict(status=status,
            completed=sum(r["status"] == "complete" for r in records), total=len(records),
            running=[dict(model=r["model"], gpu=r["gpu"]) for r in records if r["status"] == "running"]))
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    previous = {}
    succeeded = False
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        save("evaluating")
        last_heartbeat = time.monotonic()
        while pending or active:
            while pending and free:
                index, gpu = pending.popleft(), free.popleft()
                record = records[index]
                record.update(gpu=gpu, started_at=datetime.now().isoformat())
                path = Path(record["log"])
                path.parent.mkdir(parents=True, exist_ok=True)
                log = path.open("w", encoding="utf-8")
                log.write(f"[scheduler] {record['model']} GPU={gpu}\n")
                log.flush()
                try:
                    process = subprocess.Popen(commands[index], stdout=log, stderr=subprocess.STDOUT,
                                               env=worker_environment(gpu), start_new_session=True)
                except BaseException:
                    record["status"] = "failed"
                    log.close()
                    raise
                active[index] = dict(process=process, gpu=gpu, log=log)
                record.update(status="running", pid=process.pid)
                print(f"[comparison] start {record['model']} GPU={gpu}; log={path}", flush=True)
                save("evaluating")
            finished = [(index, task["process"].poll()) for index, task in active.items()]
            # 先处理失败，不能在另一个已失败的worker之后继续派新任务。
            failure = next(((index, code) for index, code in finished if code not in (None, 0)), None)
            if failure:
                index, code = failure
                records[index].update(status="failed", returncode=code, finished_at=datetime.now().isoformat())
                raise RuntimeError(f"{records[index]['model']} GPU={records[index]['gpu']} 退出码{code}；查看 {records[index]['log']}")
            for index, code in finished:
                if code is None:
                    continue
                task = active.pop(index)
                task["log"].close()
                records[index].update(status="complete", returncode=0, finished_at=datetime.now().isoformat())
                free.append(task["gpu"])
                print(f"[comparison] complete {records[index]['model']} GPU={task['gpu']}", flush=True)
                save("evaluating")
            if time.monotonic() - last_heartbeat >= 30:
                print(f"[comparison] running={len(active)} pending={len(pending)}; progress: {out / 'scheduler.json'}", flush=True)
                last_heartbeat = time.monotonic()
            if active and not (pending and free):
                time.sleep(poll_interval)
        succeeded = True
    finally:
        # 清理期间不让第二个Ctrl-C打断子进程回收。
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if active:
                _stop_groups(active, stop_timeout)
            if not succeeded:
                for record in records:
                    if record["status"] in ("pending", "running"):
                        record["status"] = "cancelled"
                save("failed")
            else:
                save("evaluated")
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    return records
