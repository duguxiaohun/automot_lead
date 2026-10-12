"""CPU/meta parameter accounting for the fixed rank16, vision-off first epoch.

No weights are loaded and no GPU is allocated. Storage bounds include explicit
headroom, not a claim that model training or full-pool labels have been certified.
"""
import json
from pathlib import Path
import shutil

from .io import file_sha
from .training_budget import COMPONENTS, plan

GIB = 1024**3
MIB = 1024**2


def lora_profile(model, targets, rank=16):
    import torch
    modules = dict(model.named_modules())
    shapes = {}
    for name in targets:
        layer = modules[name]
        if not isinstance(layer, torch.nn.Linear):
            raise ValueError('budget only supports native Linear LoRA targets')
        shapes[name] = [layer.in_features, layer.out_features]
    if not shapes or len(shapes) != len(targets):
        raise ValueError('empty or duplicate LoRA targets')
    return dict(rank=rank, vision_scope='off', target_shapes=shapes,
                parameters=sum(rank*sum(shape) for shape in shapes.values()))


def profile(model_dir):
    from qwen3vl_local.qwen35.backend import local_model_dir, require_runtime
    from qwen3vl_local.qwen35.vendor.configuration_qwen3_5 import Qwen3_5Config
    from qwen3vl_local.qwen35.vendor.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    from qwen3vl_local.sft_v2.train import _lora_target_modules
    import torch
    root = local_model_dir(model_dir); require_runtime()
    config = Qwen3_5Config.from_pretrained(str(root), local_files_only=True)
    # Native module topology only; no from_pretrained(model), weights or CUDA.
    with torch.device('meta'):
        model = Qwen3_5ForConditionalGeneration(config)
    targets = _lora_target_modules(model, vision_scope='off', strict_vision_scope=True)
    result = lora_profile(model, targets)
    result.update(config_sha256=file_sha(root/'config.json'), model_dir=str(root),
                  processor_metadata_bytes=sum(p.stat().st_size for p in root.iterdir()
                      if p.is_file() and p.suffix in ('.json','.jinja','.txt','.model')),
                  scope='native meta topology, same target selector; no weight/model readiness rerun')
    return result


def request(profile_record, output, *, stored_evidence):
    params = profile_record['parameters']
    if (type(params) is not int or params < 1 or profile_record.get('rank')!=16
            or profile_record.get('vision_scope')!='off'):
        raise ValueError('invalid adapter parameter count')
    evidence = 'already stored on this filesystem; bound SHA records in build_result.json'
    components = {k:dict(upper_bound_bytes=0, evidence=evidence) for k in COMPONENTS}
    components['phase3_index']['evidence'] = 'Phase4-only first epoch; no Phase3 index is generated here'
    # LoRA FP32 plus 2x tensor headroom; metadata/tokenizer allocated 128MiB.
    metadata=max(128*MIB,2*profile_record.get('processor_metadata_bytes',0)+32*MIB)
    components['adapters_per_checkpoint'] = dict(upper_bound_bytes=8*params+metadata,
        evidence='rank16 vision-off native Linear LoRA topology: <=4 bytes/parameter, 2x headroom +max(128MiB, 2x measured processor metadata +32MiB)')
    components['optimizer_per_checkpoint'] = dict(upper_bound_bytes=16*params+256*MIB,
        evidence='two FP32 Adam moments, 2x tensor headroom +256MiB step/history/serialization metadata')
    components['logs_and_reports'] = dict(upper_bound_bytes=2*GIB,
        evidence='allocated 2GiB combined two-mode one-epoch metrics/cases/run records; runtime guard remains active')
    components['temporary_peak'] = dict(upper_bound_bytes=2*GIB,
        evidence='allocated 2GiB checkpoint/report staging; modes execute sequentially')
    result = dict(free_bytes=shutil.disk_usage(output).free,reserve_bytes=20*GIB,
                  modes=2,retained_checkpoints_per_mode=1,components=components,
                  profile=profile_record,stored_evidence=stored_evidence,
                  limits='fixed automatic entry only: rank16, vision off, one epoch per mode; allocations are conservative engineering bounds, not measured peaks')
    plan(result)
    return result
