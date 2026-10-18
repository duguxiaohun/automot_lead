"""Build free-space policy, independent of dataset size and server name."""
import argparse
import json
import math
import shutil
from pathlib import Path

from .storage import existing_parent, GIB


def plan(output, reserve_gib=20):
    reserve_gib = float(reserve_gib)
    if not math.isfinite(reserve_gib) or reserve_gib < 20:
        raise ValueError('MIN_FREE_GIB must be finite and at least 20')
    location = existing_parent(output)
    free = shutil.disk_usage(location).free
    reserve = math.ceil(reserve_gib * GIB)
    return dict(policy='absolute_free_reserve_v1', filesystem_path=str(location),
                free_bytes=free, reserve_bytes=reserve, reserve_gib=reserve_gib,
                writable_before_reserve_bytes=max(0, free-reserve),
                can_start=free>reserve, sufficient_for_complete_build=None,
                scope='Live guard reserve; not a complete build size estimate. Other writers share this space.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-free-gib',default='20')
    a=p.parse_args()
    r=plan(a.output,a.min_free_gib)
    (a.output/'build_storage.json').write_text(json.dumps(r,indent=2)+'\n')
    if not r['can_start']:
        raise SystemExit('Available disk space is at/below reserve; no build launched. See build_storage.json')
    print(r['reserve_gib'])


if __name__=='__main__':main()
