"""不依赖 torch，复现全量构建末尾的 INVALID 30→51 配额不足。"""
from collections import Counter
import random
from types import SimpleNamespace

import pytest

from qwen3vl_local.sft_new_loop_phase3 import build_dataset as builder
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS, CONTEXT_IDS
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import (
    InvalidQuotaError, balanced_invalid_items,
)


def candidates(per_context=5):
    """十个来源、五种真实道路，五个人工问题集中于 RE5 来源。"""
    bases = []
    for source in CONTEXT_IDS:
        for i in range(per_context):
            bases.append(dict(scenario='synthetic',route_id=f'{source}_{i}',frame_id=i,town='Town01',
                split='val',context_id=source,rs=f'R{i%5+1}',primary_event='',event_codes=[],
                action_labels={k:k=='DECELERATE' for k in ACTION_KEYS},
                action_evidence={'speed_mps':8.},goal_x=10.,goal_y=0.,is_junction=False,
                distance_to_next_junction=100.,visual_label_risk=False,visual_label_risk_reasons=[],
                history_rgb_paths=['unused.jpg']*4,latest_rgb_path='unused.jpg'))
    reviewed=[]
    for i,asked in enumerate(CONTEXT_IDS[:5]):
        base={**bases[0], 'route_id':f'review_{i}','context_id':'UNSIGNALIZED_PRIORITY',
              'rs':'R5','invalid_prompt_rs':'R5'}
        row=builder._make_row(base=base,context_id=asked,invalid=True,
            invalid_source=f'source=UNSIGNALIZED_PRIORITY|true_rs=R5|asked_context={asked}')
        reviewed.append({**row,'invalid_reason':'same_rs_wrong_event'})
    return bases,reviewed


@pytest.mark.parametrize('seed', [0,1,19])
def test_builder_expands_only_invalid_and_retains_coverage(seed):
    """直接调用真实构建入口，自动重试与相同随机种子下显式51得到相同负例。"""
    bases,reviewed=candidates()
    rows,report=builder._balanced_invalid_rows(bases,split='val',target=30,
        rng=random.Random(seed),same_rs_rows=reviewed)
    assert len(rows)==51
    assert report['quota']['requested_target']==30
    assert report['quota']['effective_target']==51 and report['quota']['adjusted']
    counts=report['source_class']['counts']
    assert counts['UNSIGNALIZED_PRIORITY']==6
    assert sorted(counts.values())==[5]*9+[6]
    assert report['same_rs_unique_cases']==5 and report['same_rs_max_case_repeat']==1
    assert all(report['guards'].values())
    explicit,report=builder._balanced_invalid_rows(bases,split='val',target=51,
        rng=random.Random(seed),same_rs_rows=reviewed)
    assert rows==explicit and not report['quota']['adjusted']


def test_builder_does_not_swallow_missing_sources_or_bad_signatures():
    """只扩容类型化配额异常，不把真正的数据缺口当成容量问题。"""
    bases,reviewed=candidates()
    with pytest.raises(ValueError,match='missing_sources') as caught:
        builder._balanced_invalid_rows([b for b in bases if b['context_id']!='LEAD_BRAKE'],
            split='val',target=30,rng=random.Random(0),same_rs_rows=reviewed)
    assert not isinstance(caught.value,InvalidQuotaError)
    with pytest.raises(ValueError,match='signature'):
        builder._balanced_invalid_rows(bases,split='val',target=30,rng=random.Random(0),
            same_rs_rows=[{**reviewed[0],'invalid_source':'broken'}])


def test_sufficient_builder_budget_preserves_original_sampler(monkeypatch):
    """原预算足够时保持现有抽样序列，且只调用一次采样器。"""
    from qwen3vl_local.sft_new_loop_phase3 import invalid_balance
    bases,reviewed=candidates()
    calls=[]
    def checked(pool,**kwargs):
        """核对参数与未修改的公共抽样器输出。"""
        calls.append(kwargs['target'])
        return balanced_invalid_items(pool,**kwargs)
    monkeypatch.setattr(invalid_balance,'balanced_invalid_items',checked)
    _,report=builder._balanced_invalid_rows(bases,split='val',target=64,
        rng=random.Random(7),same_rs_rows=reviewed)
    assert calls==[64] and report['quota']['effective_target']==64
    assert not report['quota']['adjusted']


def test_split_builder_uses_effective_invalid_target_in_manifest(monkeypatch,tmp_path):
    """覆盖上层长度断言和报告；正例十类各15条，负例从30增加到51。"""
    from qwen3vl_local.sft_new_loop_phase3 import same_rs_invalid
    bases,reviewed=candidates(15)
    # 全部视为 train，避免测试依赖外部 RGB 数据或划分。
    bases=[{**b,'split':'train'} for b in bases]
    reviewed=[{**r,'split':'train'} for r in reviewed]
    monkeypatch.setattr(builder,'iter_base_frames',lambda *a,**k:iter(bases))
    monkeypatch.setattr(same_rs_invalid,'reviewed_invalid_rows',lambda *a:reviewed)
    monkeypatch.setattr(builder,'choice_annotation',lambda *a:{})
    args=SimpleNamespace(output_dir=str(tmp_path),val_ratio=0.,test_ratio=0.,target_per_context=0,
                         invalid_ratio=.2,split_seed=20260920,require_invalid_true_rs_coverage=True)
    rows,report=builder._balanced_rows_by_split(args,Counter())
    report=report['balance']['train']
    assert len(rows)==201
    assert report['target_per_context']==15
    assert report['requested_target_invalid']==30 and report['target_invalid']==51
    assert report['sampled_counts']=={**dict.fromkeys(CONTEXT_IDS,15),'INVALID':51}
    assert (tmp_path/'candidate_frames.jsonl').is_file()
