"""Versioned clarified prompt for a NEW single-frame train/evaluate run.

Frozen label builders and old checkpoints are unchanged. Every worker must use
this bootstrap; the saved inference contract distinguishes the new prompt.
"""
import argparse
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys

from .phase3_prompt_review import CLARIFICATION, VARIANT
from . import versioned_data as cache


@contextmanager
def activate():
    from .phase3_single_frame import objective, protocol
    original_system, original_contract = objective.INSTRUCTION, protocol.contract
    def contract(artifact):
        result=original_contract(artifact)
        result.update(prompt_name=VARIANT,prompt_variant=VARIANT,
                      prompt_runtime_sources={p.name:cache.bind(p)['sha256'] for p in
                        (Path(__file__),Path(__file__).with_name('phase3_prompt_review.py'))})
        return result
    try:
        objective.INSTRUCTION=CLARIFICATION+original_system
        protocol.contract=contract
        yield
    finally:
        objective.INSTRUCTION=original_system
        protocol.contract=original_contract


def wrap(command):
    command=list(command)
    modules=('qwen3vl_local.audit_joint.phase3_single_frame.worker',
             'qwen3vl_local.audit_joint.phase3_common_runtime')
    module=next((x for x in command if x in modules),None)
    if module is None:raise ValueError('Prompt variant only supports single-frame workers')
    i=command.index(module)
    return command[:i]+['qwen3vl_local.audit_joint.phase3_prompt_runtime','--module',module,'--']+command[i+1:]


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--module',required=True,choices=(
        'qwen3vl_local.audit_joint.phase3_single_frame.worker','qwen3vl_local.audit_joint.phase3_common_runtime'))
    p.add_argument('args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    with activate():
        sys.argv=[a.module,*(a.args[1:] if a.args[:1]==['--'] else a.args)]
        importlib.import_module(a.module).main()


if __name__=='__main__':main()
