"""Phase3 index admission handoff only; no training, model or dataset build."""
import argparse
import json
import os
from pathlib import Path

from .io import write_json


def inventory(root):
    root=Path(root).absolute()
    paths=[];errors=[];truncated=False
    if not root.is_dir():
        return dict(status='missing',root=str(root),paths=[],scope='file presence only')
    for directory,dirs,files in os.walk(root,followlinks=False,onerror=lambda e:errors.append(str(e))):
        parent=Path(directory)
        depth=len(parent.relative_to(root).parts)
        if depth>=4:
            truncated=truncated or bool(dirs)
            dirs[:]=[]
        else:
            dirs[:]=sorted(d for d in dirs if d not in ('.git','.cache','__pycache__','production','lead_data'))
        if 'frame_index.jsonl' in files:
            paths.append(str(parent/'frame_index.jsonl'))
        if len(paths)>=100:
            truncated=True;break
    return dict(status='inventoried',root=str(root),paths=paths,errors=errors,truncated=truncated,
                scope='bounded depth-4 path discovery only; no automatic index selection; directory symlinks not followed')


def inspect(index, variant, mode):
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_index
    from qwen3vl_local.sft_new_loop_phase4.production_storage import binding, check_binding
    index=Path(index).absolute();manifest=index.with_name('manifest.json')
    report=dict(index=str(index),prompt_variant=variant,action_output_mode=mode,status='blocked')
    members={}
    try:
        if manifest.stat().st_size>2_000_000:
            raise ValueError('manifest exceeds 2MB handoff limit')
        bindings={k:binding(p) for k,p in [('index',index),('manifest',manifest)]}
        report['bindings']=bindings
        members['manifest.json']=json.loads(manifest.read_text())
        report['preflight']=check_index(index,mode,prompt_variant=variant)
        for value in bindings.values():check_binding(value)
        report['status']='index_admitted'
    except (OSError,ValueError,KeyError,TypeError) as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    report['scope']='formal index contract/isolation check; not raw-label, budget or joint G0 certification'
    return report,members


def main():
    from .server_readiness import source_check
    from .server_pilot import package
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path)
    p.add_argument('--checkpoints',type=Path,default=Path('checkpoints'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prompt-variant',default='v23_rgb_stage_candidate_20261006')
    p.add_argument('--action-output-mode',choices=('binary','choice'),default='choice')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    expected=Path(__file__).with_name('release_sources_v53_20261012.json')
    sources=source_check(Path(__file__).resolve().parents[2],expected)
    members={'source_check.json':sources}
    report=dict(status='source_mismatch',training_started=False)
    if sources['status']=='matched':
        if a.index is None:
            report=dict(status='index_not_selected',inventory=inventory(a.checkpoints),training_started=False)
        else:
            report,extra=inspect(a.index,a.prompt_variant,a.action_output_mode)
            members.update(extra)
    members['phase3_index_report.json']=report
    write_json(a.output/'phase3_index_report.json',report)
    package(a.output,members)
    (a.output/'pilot_report.zip').rename(a.output/'phase3_index_report.zip')
    (a.output/'pilot_package.json').rename(a.output/'phase3_index_package.json')
    print(json.dumps(dict(status=report['status'],output=str(a.output)),ensure_ascii=False))
    return 0 if report['status']=='index_admitted' else 2


if __name__=='__main__':
    raise SystemExit(main())
