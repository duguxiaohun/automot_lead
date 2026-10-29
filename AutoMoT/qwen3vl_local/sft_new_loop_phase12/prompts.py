"""One system/image prefix; task text and answers are independent branches."""
import hashlib
import sys
import types
from pathlib import Path
from qwen3vl_local.audit_joint.phase12_retrain import refinement, snapshot
from qwen3vl_local.sft_new_loop_phase1 import prompts as original1
from qwen3vl_local.sft_new_loop_phase2 import prompts as original2


def _module(name, source):
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, name, 'exec'), module.__dict__)
    return module


_source1=refinement.phase1_prompt(snapshot.training_prompt(Path(original1.__file__).read_text()))
_source1=refinement.replace_once(_source1,
    'return self.phase1_output_keys + tuple(_fused_question_output_key(q) for q in self.phase2_spec.questions)',
    'return tuple(sorted(ANSWER_KEYS, key=lambda key: hashlib.sha256((self.phase2_spec.seed_key + ":fusion_flat8:" + key).encode()).digest()))')
P1 = _module(__name__+'.direct8', _source1)
P2 = _module(__name__+'.event', refinement.phase2_prompt(Path(original2.__file__).read_text()))
SYSTEM = ('You are a visual driving-scene assistant. The input is a stitched left/front/right RGB history, '
          'ordered oldest to newest. Answer the current user task independently from visible evidence at '
          'the newest moment. Older images may confirm motion, visibility and occlusion. Do not infer '
          'missing images or use hidden annotations, scenario names or another task answer. Follow the '
          'current task definitions and output only its requested YES/NO lines.')


def specs(item, phase1_answers):
    first = P1.make_prompt_spec(variant='all_random_order', answers=phase1_answers,
                               seed_key=item.spec.seed_key)
    # Native Phase2 spec is deliberately retained, including INVALID/domain.
    return first, item.spec


def texts(first, second, mode):
    return (P1.build_phase1_prompt(spec=first, history_rgb_mode=mode),
            P2.build_event_prompt(spec=second, history_rgb_mode=mode))


def messages(images, text, target=None):
    result = [dict(role='system', content=SYSTEM), dict(role='user', content=[
        *[dict(type='image', image=im) for im in images], dict(type='text', text=text)])]
    if target is not None:
        result.append(dict(role='assistant', content=target))
    return result


def targets(answers, first, second):
    return P1.build_phase1_target(answers, spec=first), P2.build_event_target(spec=second)


def identity(mode):
    return dict(system_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),
                phase1=P1.phase1_prompt_sha256(history_rgb_mode=mode),
                phase2=P2.event_prompt_sha256(history_rgb_mode=mode),
                phase1_keys=list(P1.ANSWER_KEYS),phase1_order='global_eight_key_shuffle_v1', inference='independent_prefix_forks',
                training='equal_task_loss_full_gradient_no_persistent_cache')
