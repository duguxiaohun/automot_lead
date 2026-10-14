"""Map-only catalog construction never learns stopping positions from labels."""
import json
import xml.etree.ElementTree as ET
import pytest
from qwen3vl_local.audit_joint.build_stop_catalog import build
from qwen3vl_local.audit_joint.tests.test_stop_line_probe import referenced_fixture


@pytest.mark.parametrize('lane',[-1,1])
def test_catalog_reproducible_unique_control_geometry(tmp_path,lane):
    path,_=referenced_fixture(tmp_path,lane);a=build([path]);b=build([path])
    assert a==b
    r=a['records'][f'map/10/{lane}']
    assert r['signal_id']=='900' and r['map_sha256']==a['maps']['map']['sha256']
    assert r['endpoint_xy'][0]==(20 if lane<0 else 0)
    assert abs(r['endpoint_xy'][0]-r['trigger_xy'][0])==3


@pytest.mark.parametrize('fault',['junction','lane','ambiguous'])
def test_unsupported_connection_not_published_as_stop_boundary(tmp_path,fault):
    path,_=referenced_fixture(tmp_path);root=ET.fromstring(path.read_text());rd=root.find('road[@id="21"]')
    if fault=='junction':rd.set('junction','-1')
    elif fault=='lane':rd.find('lanes/laneSection/right/lane/link/predecessor').set('id','-2')
    else:
        from copy import deepcopy
        signal=deepcopy(rd.find('signals/signal[@id="900"]'));signal.set('id','901');rd.find('signals').append(signal)
        ref=deepcopy(rd.find('signals/signalReference'));ref.set('id','901');rd.find('signals').append(ref)
    path.write_text(ET.tostring(root,encoding='unicode'))
    assert 'map/10/-1' not in build([path])['records']
