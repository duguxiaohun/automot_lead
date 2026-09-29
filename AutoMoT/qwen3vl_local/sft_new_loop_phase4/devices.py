"""入口共用的空闲GPU选择；显式GPU_IDS优先，尊重已有可见设备限制。"""
import csv
import os
import subprocess


def select_gpus(maximum=4):
    explicit=os.environ.get('GPU_IDS')
    if explicit:
        ids=explicit.split(',')
        if len(set(ids))!=len(ids) or any(not x.strip() for x in ids):
            raise ValueError('invalid GPU_IDS')
        return ','.join(ids[:maximum])
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    allowed=os.environ.get('CUDA_VISIBLE_DEVICES')
    allowed=set(allowed.split(',')) if allowed is not None else None
    candidates=[]
    for row in csv.reader(text.splitlines()):
        index,uuid,memory,util=[x.strip() for x in row]
        if allowed is not None and index not in allowed and uuid not in allowed:
            continue
        if int(memory)<4096 and int(util)<20:
            candidates.append((int(memory),int(util),int(index),index))
    if not candidates:
        raise RuntimeError('no idle visible GPU; set GPU_IDS explicitly to choose devices')
    return ','.join(r[3] for r in sorted(candidates)[:maximum])
