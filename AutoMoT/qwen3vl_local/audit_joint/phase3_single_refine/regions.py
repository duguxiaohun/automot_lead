from .frozen.vocabulary import ACTIONS
from . import VERSION

def names(variant):
    if variant!='six':raise ValueError('single_requires_six')
    return ACTIONS

def contract(variant):
    return dict(version=VERSION,variant=variant,names=list(names(variant)),diagnostic_only=True,expert_compatible=False)
