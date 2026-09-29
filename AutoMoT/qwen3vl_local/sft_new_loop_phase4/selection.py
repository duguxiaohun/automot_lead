"""按已保存完整验证分数选择兼容adapter，不按目录时间或test成绩选优。"""
import json
import math
from pathlib import Path
from .identity import check_contract
from .observation import check_observation_contract


def select_best(run,base_assets):
    from qwen3vl_local.qwen35.adapters import validate_adapter
    candidates=[]
    errors=[]
    paths=set(Path(run).glob('epoch_*'))
    pointer=Path(run)/'best.json'
    if pointer.exists():
        paths.add((Path(run)/json.loads(pointer.read_text())['adapter']).resolve())
    for path in sorted(paths):
        if not path.is_dir():
            continue
        try:
            recorded=json.loads((path/'phase4_contract.json').read_text())
            check_contract(recorded['source'])
            check_observation_contract(recorded.get('observation_contract'))
            if recorded != json.loads((Path(run)/'run.json').read_text()):
                raise ValueError('candidate belongs to a different run/dataset')
            validate_adapter(path,base_assets=base_assets)
            report=json.loads((path.parent/(path.name+'_validation.json')).read_text())
            if (not report.get('count') or report.get('macro_score') is None
                    or not math.isfinite(report['macro_score']) or report.get('split')!='val'
                    or report.get('count')!=report.get('expected_count')
                    or report.get('dataset')!=recorded['dataset']):
                raise ValueError('missing complete validation')
            candidates.append((report['macro_score'],path.name,path))
        except (ValueError,KeyError,FileNotFoundError) as ex:
            errors.append((str(path),str(ex)))
    if not candidates:
        raise ValueError(f'no compatible evaluated adapter: {errors}')
    return max(candidates)[2]
