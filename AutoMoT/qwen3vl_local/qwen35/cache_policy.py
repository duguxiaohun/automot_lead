"""Portable cache configuration, not a certificate of model quality or speed."""
import json
from pathlib import Path

FIELD='text_cache_policy'
ATTR='_text_cache_policy'
SCHEMA='text_cache_deployment_v1_20261031'


def current():
    from . import text_prefix as tp
    if not tp.enabled():return None
    return dict(schema=SCHEMA,input_layout=tp.identity(),enabled=tp.cache_enabled(),
                limits=tp.cache_limits(),partition='canonical_complete_or_capped_text_v1')


def serialize(policy):
    return json.dumps(policy,sort_keys=True,allow_nan=False)


def validate(policy,owner=None):
    """Compare immutable metadata to both environment and the actual resident bank."""
    actual=current()
    if serialize(policy)!=serialize(actual):
        raise ValueError(f'Text cache deployment policy differs from adapter; expected={policy}, actual={actual}. '
                         'Use the adapter settings; changed settings require new acceptance.')
    if policy is not None:
        bank=getattr(owner,'_text_prefix_cache',None)
        if bank is not None and {k:getattr(bank,k) for k in ('max_entries','max_bytes','max_tokens')}!=policy['limits']:
            raise ValueError('Resident cache limits differ from adapter deployment policy')
    return policy


def from_metadata(metadata):
    from . import text_prefix as tp
    layout=metadata.get('input_layout',metadata.get('prompt',{}).get('input_layout','images_before_text_legacy'))
    if layout!=tp.identity():raise ValueError('Adapter input layout differs from deployment')
    policy=metadata.get(FIELD)
    if layout==tp.CONTRACT and not isinstance(policy,dict):
        raise ValueError('Text-first adapter lacks text_cache_policy; use newly bound/validated weights, not inferred defaults')
    return validate(policy)


def bind(owner,policy):
    validate(policy,owner)
    setattr(owner,ATTR,serialize(policy))
    return getattr(owner,ATTR)


def check(owner):
    """Also cover native eval bundles whose bound policy lives on the loaded model."""
    model=getattr(owner,'model',None);model=getattr(model,'module',model)
    if hasattr(model,'get_base_model'):model=model.get_base_model()
    records=[getattr(x,ATTR) for x in (owner,model) if x is not None and hasattr(x,ATTR)]
    for record in records:validate(json.loads(record),owner)


def start_training(bundle):
    policy=current()
    bind(bundle,policy)
    model=getattr(bundle.model,'module',bundle.model)
    if hasattr(model,'get_base_model'):model=model.get_base_model()
    bind(model,policy)


def saved_fields(bundle):
    check(bundle)
    from . import text_prefix as tp
    if not tp.enabled():return {}
    if not hasattr(bundle,ATTR):raise ValueError('Training cache deployment policy was not bound at startup')
    return {FIELD:json.loads(getattr(bundle,ATTR))}


def adapter_policy(path):
    """Native self-description is sufficient; no access to the original ready path."""
    root=Path(path)
    paths=[root/name for name in ('fusion_config.json',*[f'sft_new_loop_phase{p}_adapter_config.json' for p in (1,2,3)]) if (root/name).is_file()]
    if len(paths)>1:raise ValueError('Ambiguous native adapter cache policy')
    return (True,from_metadata(json.loads(paths[0].read_text()))) if paths else (False,None)
