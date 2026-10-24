"""Strictly verified original labels with a separate candidate checkpoint contract.

No dataset mutation, filtering, relabeling or split repair occurs in this view.
"""
from ...phase3_single_v3 import dataset as original
from ...phase3_single_v3.dataset import digest, mass, write, sources
from ..experiment import code, require_received_source
from . import regions

def code_identity():
    require_received_source()
    return {**original.code_identity(), **code()}

def load(path, *, variant=None, with_sha256=False, **kwargs):
    require_received_source()
    result=original.load(path,variant=variant,with_sha256=with_sha256,**kwargs)
    artifact,sha=result if with_sha256 else (result,None)
    artifact['native_contract']=artifact['contract']
    artifact['contract']=regions.contract(variant) if variant else {v:regions.contract(v) for v in regions.VARIANTS}
    artifact['code']=code_identity()
    return (artifact,sha) if with_sha256 else artifact
