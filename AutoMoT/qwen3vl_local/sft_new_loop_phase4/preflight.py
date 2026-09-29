"""逐层预检：数据/支持量/因果输入/采样/离线模型；缺项不伪造ready。"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='qwen3vl_local.sft_new_loop_phase4'

import argparse
import json
from pathlib import Path
from .dataset import load_dataset,coverage_report
from .identity import ROOT
from .sampling import plan
from .model import load_images
from .controller import Episode
from .prompts import prompt


def inspect(dataset,model_dir=None,data_root=None):
    data,m=load_dataset(dataset)
    report=dict(data_ready=coverage_report(sum(data.values(),[]))['ready'],coverage=m['coverage'])
    if data['train']:
        _,report['sampling']=plan(data['train'],epoch=0)
    if data_root:
        for rows in data.values():
            for row in rows:
                prompt(Episode(**row['episode']),row['edge'],row['observation'])
                load_images(row,data_root)
        report['rgb_verified']=sum(map(len,data.values()))
    if model_dir:
        from qwen3vl_local.qwen35.preflight import check
        report['model']=check(Path(model_dir),action=False)
    report['ready']=report['data_ready'] and bool(report.get('model',{}).get('ready',False))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--model-dir',type=Path)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--require-ready',action='store_true')
    a=p.parse_args()
    report=inspect(a.dataset,a.model_dir,a.data_root)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if a.require_ready and not report['ready']:
        raise SystemExit(2)

if __name__=='__main__':
    main()
