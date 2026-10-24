"""Fixed clarified text for training and evaluation; no environment-selected prompt."""
from ..objective import choice_ids, allowed_indices, observation_binding, loss_from_logits, readout
from .. import objective as independent
from . import regions

def messages(record,image):
    return independent.messages(record,image,prompt_variant='clarified')

def pack(bundle,record,image,**kwargs):
    return independent.pack(bundle,record,image,prompt_variant='clarified',**kwargs)

def predict(bundle,record,image):
    result=independent.predict(bundle,record,image,prompt_variant='clarified')
    result.update(contract=regions.contract(record['variant']),source='predicted')
    return result
