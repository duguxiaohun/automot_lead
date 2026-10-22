from types import SimpleNamespace
import pytest
import torch
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID
from .. import objective, metrics, regions, expert
from .test_study import record, prediction


def test_native_event_domains_and_explicit_prompt():
    for event,context in CONTEXT_BY_ID.items():
        r=record(event=event,variant='six')
        assert [regions.ACTIONS[i] for i in objective.allowed_indices(r)]==list(context.action_keys)+['KEEP']
    text=objective.messages(record(variant='six'),'image')[1]['content'][1]['text']
    line=next(s for s in text.splitlines() if s.startswith('Allowed answers'))
    assert 'A=DECELERATE' in line and 'F=KEEP' in line and 'D=' not in line and 'E=' not in line


@pytest.mark.parametrize('variant',regions.VARIANTS)
def test_out_of_domain_top_logit_constrained_with_unrestricted_diagnostics(monkeypatch,variant):
    r=record(variant=variant);choices=list(range(10,10+len(regions.names(variant))))
    logits=torch.zeros(4,32);logits[:,13]=10;logits[:,15]=9  # D highest, F second.
    model=torch.nn.Linear(1,1);bundle=SimpleNamespace(model=model)
    monkeypatch.setattr(objective,'pack',lambda *a:dict(choices=choices))
    monkeypatch.setattr(objective,'readout',lambda *a:logits)
    pred=objective.predict(bundle,r,None)
    assert model.training
    assert pred['unrestricted_token_ids']==[3]*4 and pred['raw_class_ids']==[3]*4
    assert pred['letter_format_valid']==[True]*4
    assert pred['token_ids']==([5]*4 if variant=='six' else [3]*4)
    assert pred['format_valid']==([False]*4 if variant=='six' else [True]*4)
    if variant=='six':
        report=metrics.evaluate([dict(record=r,prediction=pred)])
        assert report['macro_recall']==1 and report['full_vocabulary_accuracy']['all']==0
        assert expert.condition(r,expected_variant='six',prediction=pred)['geometry_token_ids'].tolist()==[[5]*4]
        with pytest.raises(ValueError,match='outside_event_domain'):
            expert.condition(r,expected_variant='six',prediction=prediction(r,[3]*4))
    logits[:,31]=11
    pred=objective.predict(bundle,r,None)
    assert pred['raw_class_ids']==[-1]*4 and pred['format_valid']==[False]*4
