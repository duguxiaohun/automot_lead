import copy
import pickle
import pytest
from PIL import Image
from .. import dataset,regions,objective,planning,admission
from ...geometry_study.tests.test_study import record as original_record
from ...geometry_v1 import dataset as native
from ...geometry_v1.tests.test_geometry import Lanes,meta

def record(frame=8,event='LEAD_BRAKE',variant='six',route='Town00_Test'):
    r=original_record(frame,event,variant,route)
    r['sampling_mass']={k:v for k,v in r['sampling_mass'].items() if k.startswith('six/')}
    if not r['sampling_mass']:
        r['sampling_mass']={f'six/{i}/{c}':q for i,t in enumerate(r['targets'])
            for c,q in enumerate(dataset.mass(t,6)) if q}
    return r

def prediction(r,ids):
    return dict(token_ids=ids,format_valid=[True]*4,contract=regions.contract('six'),
        sample_binding=objective.observation_binding(r),source='predicted')

def test_no_geometry_future_or_history_required(tmp_path,monkeypatch):
    root=tmp_path/'raw';run=root/'Synthetic/Town00_Test';maps=tmp_path/'maps';maps.mkdir()
    (run/'metas').mkdir(parents=True);(run/'rgb').mkdir()
    for i in range(40):
        m=meta(i);m.update(pos_global=[i*.75,0,0],theta=0,road_id=1,lane_id=-1,lane_type_str='Driving')
        # Saved observed positions still support native future intentions.
        m.pop('future_positions',None)
        (run/'metas'/f'{i:04d}.pkl').write_bytes(pickle.dumps(m))
    Image.new('RGB',(32,32)).save(run/'rgb/0008.jpg')
    route=dict(scenario='Synthetic',route_id='Town00_Test',town='Town00',split='train',
        anchors=[dict(frame=8,event='LEAD_BRAKE',road_structure='R1')])
    parent,_=native.build([route],root,Lanes(),tmp_path/'parent',require_coverage=False)
    assert not any(t['valid'] for t in parent['records'][0]['targets'])
    monkeypatch.setattr(dataset.sources,'LaneMap',lambda _:Lanes())
    data,receipt=dataset.build(tmp_path/'parent/dataset.json',maps,tmp_path/'single')
    assert receipt['anchors']==1 and not receipt['geometry_endpoint_filter_used']
    assert set(data['records'][0]['representations'])=={'six'}
    row=dataset.load(tmp_path/'single/dataset.json',variant='six')['records'][0]
    assert all(t['valid'] for t in row['targets'])
    assert all(k.startswith('six/') for k in row['sampling_mass'])
    assert objective.allowed_indices(row)==[0,1,2,5]
    assert not any('radius_m' in t for t in row['targets'])
    source=run/'metas/0000.pkl';source.write_bytes(source.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='source_content_changed'):
        dataset.load(tmp_path/'single/dataset.json')

def test_single_soft_targets_and_current_direction_unchanged():
    from ...geometry_study import objective as old,six as old_six
    import torch
    r=record();logits=torch.randn(4,20,requires_grad=True)
    assert torch.equal(objective.loss_from_logits(logits,r['targets'],list(range(6))),
                       old.loss_from_logits(logits,r['targets'],list(range(6))))
    assert objective.messages(r,'image')==old.messages(r,'image')
    from .. import six
    assert six.build.__code__.co_code==old_six.build.__code__.co_code

def test_mixed_only_validation_does_not_choose_or_block_final_epoch():
    from .test_admission import cohort
    artifact=cohort(val_per_event=20)
    for r in artifact['records']:
        if r['split']=='val':
            for t in r['representations']['six']['targets']:
                t.update(target=[.5,0,0,0,0,.5],token_id=-1)
    report=admission.require(admission.inspect(artifact))
    assert len(report['fixed_validation_ids'])==200
    assert not report['validation']['six']['macro_recall_defined']
    assert report['validation']['six']['queries']['mixed']==800
