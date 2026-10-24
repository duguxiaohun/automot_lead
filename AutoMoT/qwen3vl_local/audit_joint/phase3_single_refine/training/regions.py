"""Distinct checkpoint identity for the independently frozen single candidate."""
from ...phase3_single_v3.regions import names, ACTIONS, VARIANTS
from ...phase3_single_v3.regions import contract as original_contract
from ..audit import sha
from .. import prompt
from ..identity import digest
from pathlib import Path

def contract(variant):
    result = dict(original_contract(variant))
    result.update(version='single_frozen_clarified_training_v1',
                  parent_version=result['version'],prompt_variant='clarified',
                  prompt_sha256=digest(prompt.instruction('clarified')),
                  frozen_provenance_sha256=sha(Path(__file__).parents[1]/'provenance.json'))
    return result
