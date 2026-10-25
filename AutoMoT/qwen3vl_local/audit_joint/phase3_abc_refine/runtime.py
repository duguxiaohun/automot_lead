"""Scoped worker adapter for the existing trainer, without editing frozen sources.

Only a dedicated CLI process enters this context. All patched references are
restored on exit. New data/checkpoints bind the additional experiment sources.
"""
from contextlib import contextmanager
from pathlib import Path
from ..abc_delivery import sha
from ..geometry_abc_v2 import dataset, objective, regions
from . import VERSION, prompt

WORKFLOW = 'abc_fresh_lora_posttraining_regression_v2'
INITIALIZATION = 'original_qwen35_fresh_lora_no_prior_adapter'

BASE_CONTRACT = regions.contract
BASE_IDENTITY = dataset.code_identity


def contract(variant):
    return dict(BASE_CONTRACT(variant), prompt_experiment=VERSION,
                initialization=INITIALIZATION, workflow=WORKFLOW, label_policy='unchanged_frozen_v2', prompt_change='a_only_explicit_endpoint_radii')


def identity():
    app=Path(__file__).resolve().parents[3]
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parents[1]/'abc_delivery.py']
    return dict(BASE_IDENTITY(), **{str(p.relative_to(app)):sha(p) for p in paths})


@contextmanager
def activated():
    saved=(regions.contract,dataset.code_identity,objective.messages)
    if saved != (BASE_CONTRACT,BASE_IDENTITY,prompt.ORIGINAL_MESSAGES):
        raise RuntimeError('refine_requires_unmodified_dedicated_worker')
    regions.contract=contract;dataset.code_identity=identity;objective.messages=prompt.messages
    try: yield
    finally: regions.contract,dataset.code_identity,objective.messages=saved


@contextmanager
def fresh_training():
    """禁止训练过程恢复旧 adapter；评测/探针只重载本轮生成的权重。"""
    from ..geometry_abc_v2 import train
    saved=train.restore
    def forbidden(*args,**kwargs):
        raise ValueError('prior_adapter_restore_forbidden_during_fresh_training')
    train.restore=forbidden
    try:yield
    finally:train.restore=saved
