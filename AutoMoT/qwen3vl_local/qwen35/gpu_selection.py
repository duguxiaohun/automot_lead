"""Training defaults to physical GPUs 0..3; single-card probes remain automatic."""
import os
import subprocess


def select(count=4, requested=None):
    if requested is None:requested=os.environ.get('GPU_IDS','')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    available={}
    for line in raw.strip().splitlines():
        index,memory,util=[x.strip() for x in line.split(',')]
        available[index]=(float(memory),float(util))
    if requested:ids=requested.split(',')
    elif count==4:ids=['0','1','2','3']
    else:ids=[k for k,v in available.items() if v[0]<256 and v[1]<=5][:count]
    if len(ids)!=count or len(set(ids))!=count or any(x not in available for x in ids):
        raise ValueError(f'{count} distinct idle physical GPUs required; training defaults to 0,1,2,3; override with GPU_IDS')
    if any(available[x][0]>=256 or available[x][1]>5 for x in ids):
        raise ValueError(f'Requested GPU is busy ({",".join(ids)}); no automatic substitution or process/compute-mode changes')
    return ids
