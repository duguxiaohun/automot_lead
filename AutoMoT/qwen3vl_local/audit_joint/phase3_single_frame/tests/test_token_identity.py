"""Full prompt token identity, including whitespace-merge counterexamples."""
import copy
from types import SimpleNamespace

import pytest
from PIL import Image
from tokenizers.pre_tokenizers import ByteLevel

from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.qwen35.vendor.tokenization_qwen3_5 import Qwen3_5Tokenizer
from qwen3vl_local.audit_joint.phase3_single_frame.tests.test_single_frame import record
from qwen3vl_local.audit_joint.phase3_single_frame import objective
from qwen3vl_local.audit_joint.phase3_base_dense.data import ACTIONS, LETTERS


def merged_bundle(processor, *, newline_merge=False):
    specials=list(processor.tokenizer.all_special_tokens)
    vocab=processor.tokenizer.get_vocab().copy()
    prefix='Ċ' if newline_merge else 'Ġ'
    for letter in LETTERS:vocab[prefix+letter]=len(vocab)
    tokenizer=Qwen3_5Tokenizer(vocab=vocab,merges=[(prefix,x) for x in LETTERS],
        additional_special_tokens=specials,eos_token='<|im_end|>')
    if newline_merge:
        # Deliberately unsafe tokenizer: standalone A-F still pass the old
        # check, but each contextual newline+letter becomes a different ID.
        tokenizer._tokenizer.pre_tokenizer=ByteLevel(add_prefix_space=False,use_regex=False)
    processor.tokenizer=tokenizer
    return SimpleNamespace(processor=processor,tokenizer=tokenizer)


def test_all_six_actions_full_prompt_identity_with_real_space_merges(processor):
    bundle=merged_bundle(processor)
    choices=objective.old.choice_ids(bundle.tokenizer)
    for letter,candidate in zip(LETTERS,choices):
        assert bundle.tokenizer('s: '+letter,add_special_tokens=False)['input_ids'][-1]!=candidate
    rec=record();before=copy.deepcopy(rec);vocab=bundle.tokenizer.get_vocab().copy()
    report=objective.tokenizer_admission(bundle,rec,Image.new('RGB',(32,32)))
    assert report['status']=='passed' and report['history_dropout'] is False
    assert report['attention_masked_unknown'] is False
    assert rec==before and bundle.tokenizer.get_vocab()==vocab
    for action,candidate in zip(ACTIONS,choices):
        slots=report['checks'][action]['slots']
        assert len(slots)==4
        assert all(s['input_ids']==[candidate] and s['same_action_token'] for s in slots)
    unknown=objective.pack(bundle,rec,Image.new('RGB',(32,32)))
    for slot in unknown['history_token_check']['slots']:
        assert slot['action']=='UNKNOWN' and slot['output_id'] is None
        assert slot['positions']
        assert all(unknown['inputs']['attention_mask'][0,p]==1 for p in slot['positions'])


def test_full_prompt_rejects_context_merge_despite_valid_standalone_choices(processor):
    bundle=merged_bundle(processor,newline_merge=True)
    assert len(set(objective.old.choice_ids(bundle.tokenizer)))==6
    rec=dict(record(),history=[ACTIONS[0]]*4)
    with pytest.raises(ValueError,match='History/output action token ID mismatch'):
        objective.pack(bundle,rec,Image.new('RGB',(32,32)))


@pytest.mark.parametrize('tamper',['duplicate','missing','changed'])
def test_history_alignment_rejects_ambiguous_or_altered_prompt(processor,tamper):
    bundle=merged_bundle(processor);rec=dict(record(),history=[ACTIONS[0]]*4)
    packed=objective.pack(bundle,rec,Image.new('RGB',(32,32)))
    text=bundle.tokenizer.decode(packed['inputs']['input_ids'][0],skip_special_tokens=False)
    marker='[HISTORY_ACTION_0]\n'
    if tamper=='duplicate':text+=marker
    elif tamper=='missing':text=text.replace(marker,'[NO_HISTORY_SLOT]\n')
    else:text=text.replace(marker+'A',marker+'B')
    ids=bundle.tokenizer(text,add_special_tokens=False)['input_ids']
    with pytest.raises(ValueError,match='historical token marker|Historical action text changed'):
        objective.check_history_token_ids(bundle.tokenizer,ids,rec['history'],packed['choices'])


def test_natural_missingness_groups_keep_data_and_same_metric_denominators():
    cases=[dict(record(),id='known',history=['STOP']*4,history_original=['UNKNOWN','STOP','STOP','STOP'],
                predictions=['STOP']*4),
           dict(record(),id='missing',predictions=['KEEP']*4)]
    before=copy.deepcopy(cases);report=objective.metrics(cases)
    assert cases==before
    assert report['total']['all']==8
    assert report['total']['input_unknown_slots_0']==report['total']['input_unknown_slots_4']==4
    assert report['set_accuracy']['input_unknown_slots_0']==1
    assert report['set_accuracy']['input_unknown_slots_4']==0
    assert report['natural_missingness']['anchor_counts']=={
        'native_unknown_slots_1':1,'input_unknown_slots_0':1,
        'native_unknown_slots_4':1,'input_unknown_slots_4':1}
    assert report['natural_missingness']['artificial_training_masking'] is False
