import json
from types import SimpleNamespace
import pytest

from qwen3vl_local.action_prior.split_support import make_split_plan


def test_action_plan_moves_whole_routes_and_preserves_exposed_train(tmp_path):
    rows, records = [], {}
    for i in range(16):
        run = f'route{i}'
        for anchor in range(44):
            row = dict(schema='action_prior_data_v1', scenario='s', run_id=run,
                       route_group=f's/{run}', split='train', anchor=anchor)
            rows.append(row)
            records[('s', run, anchor)] = {'eligible_buckets':['UE1']}
    for split in ('train','val','test'):
        (tmp_path/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows) if split=='train' else '')
    assignments, report = make_split_plan(tmp_path, records, {'s/route0'}, ('UE1',))
    assert assignments['s/route0']=='train'
    assert report['before']['train']['UE1']==16*40  # excludes 0..3
    assert report['physical_groups_after']=={'train':{'UE1':6},'val':{'UE1':5},'test':{'UE1':5}}
    assert set(assignments.values())=={'train','val','test'}
    assert make_split_plan(tmp_path, dict(reversed(list(records.items()))), {'s/route0'}, ('UE1',))[0]==assignments
    (tmp_path/'test.jsonl').write_text(json.dumps({**rows[0],'split':'test'})+'\n')
    with pytest.raises(ValueError,match='physical route leakage'):
        make_split_plan(tmp_path, records, {'s/route0'}, ('UE1',))


@pytest.mark.parametrize('variant', ['prior','qwen_simple','bev_only'])
@pytest.mark.parametrize('rgb_count',[1,4])
def test_all_entries_exclude_initialization_history(tmp_path, monkeypatch, variant, rgb_count):
    from qwen3vl_local.action_prior import config, event_balance, action_token
    from qwen3vl_local.action_expert_ablation import common
    import lead_video_tools.abnormal_duration_filter as abnormal
    (tmp_path/'s'/'r').mkdir(parents=True)
    rows=[dict(schema='action_prior_data_v1',scenario='s',run_id='r',route_group='s/r',
               split='train',anchor=i,tp_mode='route_lookahead',rgb_frame_count=4,rgb_frame_step=1)
          for i in range(6)]
    for split in ('train','val','test'):
        (tmp_path/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows) if split=='train' else '')
    monkeypatch.setattr(abnormal,'is_abnormal_lead_route',lambda *a:(False,{}))
    monkeypatch.setattr(event_balance,'annotate_rows',lambda *a:None)
    monkeypatch.setattr(action_token,'annotate_tokens',lambda *a:None)
    parser = config.parser() if variant=='prior' else common.parser(variant)
    args=parser.parse_args(['--data-root',str(tmp_path),'--data-dir',str(tmp_path),'--rgb-frame-count',str(rgb_count)])
    result=config.read_rows(args,'train') if variant=='prior' else common.read_rows(args,'train')
    assert [r['anchor'] for r in result]==[4,5]
    assert all(r['rgb_frame_count']==rgb_count for r in result)
