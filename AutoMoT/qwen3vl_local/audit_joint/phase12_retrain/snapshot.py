"""New content-addressed training source; never edits an existing snapshot."""
import ast
import copy
import hashlib
from pathlib import Path
import shutil
import tempfile

from qwen3vl_local import runtime_version as runtime
from qwen3vl_local.audit_joint.phase1_visual_refine_v2.prompts import RULES
from . import CONTRACT, REVIEW_PATH

PROMPT = 'qwen3vl_local/sft_new_loop_phase1/prompts.py'


def training_prompt(source):
    """Change only P1 rendering/name. No shared Phase2 definition mutation."""
    anchor = 'blocks.append(phase2_prompts.RS_DEFINITIONS[item])'
    name = 'sft_new_loop_phase1_phase1_phase2_combined_v5_rgb_audited_rs4_hardware'
    if source.count(anchor) != 1 or source.count(name) != 1:
        raise ValueError('Native Phase1 prompt changed; explicit review required')
    clauses = {value[0]: value[1] for value in RULES.values()}
    text = source.replace(name, CONTRACT + '_rs2_rs4_trained')
    text = text.replace(anchor, 'blocks.append(phase2_prompts.RS_DEFINITIONS[item] + ' + repr(clauses) + '.get(item, ""))')
    # Every other function, including specs/order/targets/parser, must match AST.
    def functions(s):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    old,new=functions(source),functions(text)
    assert set(old)==set(new)
    assert [k for k in old if old[k]!=new[k]]==['_phase2_questions_and_rules']
    return text


def prepare(app, phase):
    if phase not in (1,2):
        raise ValueError('Phase must be 1 or 2')
    app=Path(app).resolve()
    cache=app/'checkpoints/phase12_retrain_runtime'
    cache.mkdir(parents=True,exist_ok=True)
    with runtime.lock(cache/'.build.lock'):
        original=runtime.inventory(app)
        record=copy.deepcopy(original)
        replacements={}
        # Separate latest pointers and output namespaces from the old production
        # runs. Only literal output roots change, never sampler/trainer Python.
        for basename in ('train.sh','run_full_pipeline.sh'):
            name=f'qwen3vl_local/sft_new_loop_phase{phase}/{basename}'
            source=(app/name).read_text()
            old=f'checkpoints/sft_new_loop_phase{phase}_runs'
            if old not in source:raise ValueError('Native output-root anchor missing')
            replacements[name]=source.replace(old,f'checkpoints/sft_new_loop_phase{phase}_retrain_runs').encode()
        if phase==1:
            replacements[PROMPT]=training_prompt((app/PROMPT).read_text()).encode()
            # Materialize only the approved small review leaf beneath shared data.
            rel=Path(REVIEW_PATH)
            for i in range(1,len(rel.parts)):
                prefix=Path(*rel.parts[:i])
                if str(prefix) in record['assets']:
                    del record['assets'][str(prefix)]
                    for child in (app/prefix).iterdir():
                        record['assets'][str(prefix/child.name)]=str(child.resolve())
            record['assets'].pop(REVIEW_PATH,None)
            replacements[REVIEW_PATH]=(app/REVIEW_PATH).read_bytes()
        for name,content in replacements.items():
            record['files'][name]=hashlib.sha256(content).hexdigest()
        record['training_revision']={'contract':CONTRACT,'phase':phase,
            'parent_inventory_sha256':runtime.digest(original),
            'modified_sources':sorted(k for k in replacements if k.startswith('qwen3vl_local/')),
            'parent_prompt_sha256':original['files'].get(PROMPT),
            'initialization':'fresh_base_no_previous_adapter'}
        target=cache/runtime.digest(record)
        if target.exists():
            runtime.verify(target)
            return target
        pending=Path(tempfile.mkdtemp(prefix='.new-',dir=cache))
        try:
            tree=pending/'app';tree.mkdir()
            for name,digest in record['files'].items():
                dest=tree/name;dest.parent.mkdir(parents=True,exist_ok=True)
                if name in replacements:dest.write_bytes(replacements[name])
                else:shutil.copyfile(app/name,dest)
                if runtime.sha(dest)!=digest:raise ValueError('Source changed during capture: '+name)
                dest.chmod(0o555 if dest.suffix=='.sh' else 0o444)
            for name,link in record['assets'].items():
                dest=tree/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(link)
            for name,link in record['siblings'].items():(pending/name).symlink_to(link)
            if runtime.inventory(app)!=original:raise ValueError('Source inventory changed during capture')
            if phase==1 and (app/REVIEW_PATH).read_bytes()!=replacements[REVIEW_PATH]:
                raise ValueError('Review changed during capture')
            runtime.write(pending/'manifest.json',record)
            pending.rename(target);runtime.verify(target)
            return target
        finally:
            if pending.exists():shutil.rmtree(pending)
