from types import SimpleNamespace
import copy
import torch
from PIL import Image
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.audit_joint import phase3_matched_image_eval as evaluation
from qwen3vl_local.audit_joint import phase3_behavior_diagnostics as diagnostics
from qwen3vl_local.audit_joint.phase3_single_frame import targets,history


def test_native_small_qwen_paired_pixels_no_label_or_parameter_mutation(tmp_path,processor,model):
    records={}
    for i,color in enumerate(((255,0,0),(0,0,255))):
        path=tmp_path/f'{i}.png';Image.new('RGB',(32,32),color).save(path)
        records[str(i)]=dict(id=str(i),event='LEAD_BRAKE',image=str(path),
            history=['STOP']*4,history_original=['STOP']*4,history_imputed=[False]*4,history_changed=[False]*4,
            history_semantics=history.SEMANTICS['offline_native'],
            premise='Current measured speed: 0 m/s. Allowed choices A B C F.',
            windows=[targets.distribution(['STOP','STOP']) for _ in range(4)])
    before=copy.deepcopy(model.state_dict());saved=copy.deepcopy(records)
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    model.train()
    plan=dict(pairs=[dict(receiver='0',donor='1'),dict(receiver='1',donor='0')])
    actual,wrong=evaluation.predict_pairs(bundle,records,plan)
    assert records==saved and model.training
    assert [c['id'] for c in actual]==[c['id'] for c in wrong]==['0','1']
    assert all(len(c['predictions'])==4 for c in actual+wrong)
    for a,b in zip(actual,wrong):
        assert a['windows']==b['windows'] and a['history_input']==b['history_input']
        assert a['probabilities']!=b['probabilities']
    result=diagnostics.paired_scores(actual,wrong,{str(i):dict(group=str(i)) for i in range(2)})
    assert result['subsets']['all']['windows']==8
    for k,value in model.state_dict().items():torch.testing.assert_close(value,before[k],rtol=0,atol=0)
    assert all(p.grad is None for p in model.parameters())
