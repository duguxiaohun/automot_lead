"""Lightweight launch configuration check; no GPU initialization or downloads."""
import argparse
import json
from pathlib import Path
from .identity import check_contract


def check(dataset,model_dir=None,data_root=None):
    errors=[];dataset=Path(dataset);manifest=dataset/'manifest.json'
    if not manifest.is_file():errors.append(f'DATASET={dataset}: manifest.json missing; select a built data2/data4 directory or run the build pipeline')
    else:
        try:
            m=json.loads(manifest.read_text());check_contract(m['contract'])
            for name in ('train.jsonl','val.jsonl','test.jsonl'):
                if not (dataset/name).is_file():errors.append(f'DATASET={dataset}: {name} missing')
        except (ValueError,KeyError,TypeError) as ex:errors.append(f'DATASET={dataset}: {ex}')
    if model_dir is not None:
        model_dir=Path(model_dir)
        if not (model_dir/'config.json').is_file():errors.append(f'MODEL_DIR={model_dir}: local base model missing; set MODEL_DIR to the complete installed base model')
        elif not any(model_dir.glob('*.safetensors')):errors.append(f'MODEL_DIR={model_dir}: model weight files missing')
    if data_root is not None and not Path(data_root).is_dir():errors.append(f'DATA_ROOT={data_root}: RGB root missing')
    return dict(configuration_valid=not errors,errors=errors,scope='path and dataset contract check only; full model/RGB preflight still required',downloads_performed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--model-dir',type=Path);p.add_argument('--data-root',type=Path);a=p.parse_args()
    result=check(a.dataset,a.model_dir,a.data_root);print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['errors']:raise SystemExit(2)

if __name__=='__main__':main()
