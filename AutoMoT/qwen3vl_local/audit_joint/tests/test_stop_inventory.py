import json
import pytest
from qwen3vl_local.audit_joint import stop_inventory as tool
from qwen3vl_local.audit_joint.io import digest,file_sha


def fixture(tmp_path,split='train'):
    route=dict(scenario='scene',route_id='route',split=split,physical_group='scene/route',
               frames=2,artifact='route.jsonl')
    header=dict(policy='old',teacher={'version':'old'},candidate_pool_sha256='pool')
    rows=[]
    for frame in (4,5):
        q=dict(rgb_mode=4,edge='proceed',rule_target='NO' if frame==4 else 'YES',
               episode={'event':'R-E5'},question_id=str(frame),
               control_evidence={'9':{'satisfied':False}} if frame==4 else {})
        rows.append({k:route[k] for k in ('scenario','route_id','split','physical_group')}|
                    dict(frame_id=frame,questions=[q,dict(q,rgb_mode=2)]))
    artifact=tmp_path/route['artifact']
    artifact.write_text('\n'.join(json.dumps(r) for r in [dict(kind='header',**header),*rows])+'\n')
    route['sha256']=file_sha(artifact)
    index=dict(**header,routes=[route]);index['sha256']=digest(index)
    path=tmp_path/'index.json';path.write_text(json.dumps(index))
    return path,artifact


@pytest.mark.parametrize('split',['train','val','test'])
def test_old_inventory_finds_candidates_without_certifying_labels(tmp_path,split):
    path,_=fixture(tmp_path,split)
    report=tool.inventory(path)
    assert report['counts']['proceed_yes_after_pending_receipt_disappeared']==1
    assert bool(report['train_examples']) is (split=='train')
    assert not report['reuse_authorized']


def test_corrupt_route_cannot_contribute_partial_counts(tmp_path):
    path,artifact=fixture(tmp_path);artifact.write_text(artifact.read_text()+'{\n')
    report=tool.inventory(path)
    assert report['status']=='incomplete' and report['errors']
    assert report['counts']=={}


def test_index_sha_change_rejected(tmp_path):
    path,_=fixture(tmp_path)
    index=json.loads(path.read_text());index['routes']=[];path.write_text(json.dumps(index))
    with pytest.raises(ValueError,match='SHA'):tool.inventory(path)
