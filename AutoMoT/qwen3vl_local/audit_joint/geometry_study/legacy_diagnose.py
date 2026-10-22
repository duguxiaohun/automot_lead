"""Validation interventions using the unchanged geometry_v1 model/prompt contract."""
import json
from pathlib import Path
import torch
from PIL import Image
from . import metrics
from ..geometry_v1 import dataset, objective, planning, train


def run(args):
    from qwen3vl_local.sft_new_loop_phase3 import train as native
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_model
    from ..phase3_suite import model_snapshot
    if args.split!='val': raise ValueError('legacy_interventions_require_validation')
    artifact=dataset.load(args.dataset)
    checkpoint=Path(args.checkpoint)
    receipt=json.loads((checkpoint/'receipt.json').read_text())
    if (receipt['binding']['contract']!=artifact['contract'] or receipt['binding']['code']!=artifact['code'] or
            receipt['binding']['dataset_sha256']!=dataset.sources.file_sha(args.dataset)):
        raise ValueError('legacy_checkpoint_dataset_contract_mismatch')
    check_model(args.model_dir)
    if receipt['binding']['model']!={k:v['sha256'] for k,v in model_snapshot(Path(args.model_dir)).items()}:
        raise ValueError('legacy_base_model_mismatch')
    output=Path(args.output).resolve()
    for p in (Path(artifact['data_root']),Path(args.dataset).resolve().parent,checkpoint.resolve(),Path(args.model_dir).resolve()):
        if output.is_relative_to(p) or p.is_relative_to(output): raise ValueError('output_overlaps_input')
    output.mkdir(parents=True,exist_ok=False)
    bundle=native.load_model_with_lora(Path(args.model_dir),device=torch.device('cuda:0'),
        lora_rank=16,lora_alpha=32,lora_dropout=.05,lora_vision_scope='off',strict_vision_scope=True,gradient_checkpointing=False)
    train.restore(bundle,checkpoint,artifact=artifact,dataset_path=args.dataset)
    rows=planning.evaluation(artifact['records'],'val')
    baseline=None
    for mode in ('observed','history_unknown','history_permuted','image_black'):
        cases=[]
        for original in rows:
            r=metrics.intervention(original,mode)
            with Image.open(Path(artifact['data_root'])/r['image']) as image: rgb=image.convert('RGB')
            if mode=='image_black': rgb=Image.new('RGB',rgb.size)
            prediction=objective.predict(bundle,r,rgb)
            if mode!='observed': prediction.update(source='diagnostic_predicted',intervention=mode)
            cases.append(dict(record=r,prediction=prediction))
        if baseline is None: baseline=cases
        dataset.write(output/(mode+'.json'),dict(metrics=metrics.evaluate(cases,legacy_geometry=True),
            paired=metrics.paired_summary(baseline,cases),cases=cases,intervention=mode,
            parent_checkpoint_receipt_sha256=dataset.sources.file_sha(checkpoint/'receipt.json'),
            original_prompt_and_weights=True,training_executed=False))
