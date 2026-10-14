"""Map probing must never silently become permission to label a STOP YES."""
from copy import deepcopy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from qwen3vl_local.audit_joint.stop_line_probe import FrozenApproachProbe,MapProbe,road_point


def road(ident,x=0,junction='-1'):
    return f'''<road id="{ident}" length="20" junction="{junction}">
      <link><predecessor elementType="junction" elementId="30"/>
            <successor elementType="junction" elementId="20"/></link>
      <planView><geometry s="0" x="{x}" y="0" hdg="0" length="20"><line/></geometry></planView>
      <lanes><laneSection s="0"><left><lane id="1" type="driving">
        <width sOffset="0" a="3" b="0" c="0" d="0"/></lane></left>
        <right><lane id="-1" type="driving"><width sOffset="0" a="3" b="0" c="0" d="0"/>
        </lane></right></laneSection></lanes>
      <signals><signal id="{ident}0" name="StopLine2" s="0" t="-1.5" hOffset="1.5707963267948966"/></signals>
      </road>'''


def fixture(tmp_path,lane=-1):
    path=tmp_path/'map.xodr'
    path.write_text('<OpenDRIVE>'+road('10')+road('21',20,'20')+road('31',0,'30')+'''
      <junction id="20"><connection id="0" incomingRoad="10" connectingRoad="21" contactPoint="start">
        <laneLink from="-1" to="-1"/></connection></junction>
      <junction id="30"><connection id="0" incomingRoad="10" connectingRoad="31" contactPoint="start">
        <laneLink from="1" to="1"/></connection></junction></OpenDRIVE>''')
    direction=1 if lane<0 else -1
    frame=dict(frame_id=8,meta=dict(road_id=10,lane_id=lane,speed=0.,
        ego_matrix=[[direction,0,0,15 if lane<0 else 5],[0,direction,0,1.5 if lane<0 else -1.5],
                    [0,0,1,0],[0,0,0,1]]),actors=[
            dict(id=0,**{'class':'ego_car'},position=[0,0,0],extent=[2,1,1],yaw=0),
            dict(id=1,**{'class':'stop_sign'},position=[3,0,0],extent=[1.5,1.5,.5],yaw=0,affects_ego=True)],sources=[])
    return path,frame


@pytest.mark.parametrize('lane,endpoint,junction',[(-1,20,'20'),(1,0,'30')])
def test_lane_direction_changes_endpoint_not_stationary_permission(tmp_path,lane,endpoint,junction):
    path,frame=fixture(tmp_path,lane);before=deepcopy(frame)
    probe=MapProbe(path);r=probe.probe(frame,1)
    assert r['errors']==[] and r['junction_id']==junction
    assert r['approach_endpoint']['s']==endpoint
    assert r['approach_endpoint']['front_gap_m']==pytest.approx(3)
    assert not r['label_authorization'] and not r['control_line_binding_verified']
    assert not r['capture_map_identity_verified'] and before==frame
    assert r['map_sha256']==probe.sha


def test_same_junction_projection_and_perpendicular_angle(tmp_path):
    path,frame=fixture(tmp_path);r=MapProbe(path).probe(frame,1)
    line=r['candidates'][0]
    assert line['road_id']=='21' and line['directly_connected_road']
    assert line['distance_to_approach_endpoint_m']==pytest.approx(0)
    assert line['line_perpendicular_error_rad']==pytest.approx(0,abs=1e-12)
    assert line['front_gap_m']==pytest.approx(3)


def test_nearest_line_does_not_invent_control_ownership(tmp_path):
    path,frame=fixture(tmp_path);s=path.read_text()
    path.write_text(s.replace('</OpenDRIVE>',road('22',20,'20')+'</OpenDRIVE>'))
    r=MapProbe(path).probe(frame,1)
    assert len([c for c in r['candidates'] if c['distance_to_approach_endpoint_m']==0])==2
    assert not next(c for c in r['candidates'] if c['road_id']=='22')['directly_connected_road']
    assert not r['control_line_binding_verified'] and not r['label_authorization']


@pytest.mark.parametrize('problem',['missing_lane','contradictory_lane','no_connection','non_driving','left_traffic','bad_pose'])
def test_bad_association_fails_closed(tmp_path,problem):
    path,frame=fixture(tmp_path)
    if problem=='missing_lane':frame['meta'].pop('lane_id')
    if problem=='contradictory_lane':frame['actors'][0]['lane_id']=1
    if problem=='no_connection':path.write_text(path.read_text().replace('from="-1"','from="-2"'))
    if problem=='non_driving':path.write_text(path.read_text().replace('type="driving"','type="shoulder"'))
    if problem=='left_traffic':path.write_text(path.read_text().replace('<road id="10"','<road rule="LHT" id="10"'))
    if problem=='bad_pose':frame['meta']['ego_matrix'][0][0]=float('nan')
    r=MapProbe(path).probe(frame,1)
    assert r['errors'] and not r['label_authorization']


def test_arc_projection_and_lateral_offset():
    r=ET.fromstring(road('10').replace('<line/>','<arc curvature="0.1"/>'))
    point,h=road_point(r,math.pi*5,2)
    assert point==pytest.approx([8,-10]) and h==pytest.approx(-math.pi/2)


@pytest.mark.parametrize('replacement',['<spiral curvStart="0" curvEnd="1"/>','<paramPoly3/>'])
def test_unsupported_geometry_is_not_replaced_by_line(replacement):
    r=ET.fromstring(road('10').replace('<line/>',replacement))
    with pytest.raises(ValueError,match='unsupported'):road_point(r,10)


def test_out_of_range_and_geometry_gap_are_rejected():
    r=ET.fromstring(road('10'))
    with pytest.raises(ValueError,match='outside'):road_point(r,21)
    r.find('planView/geometry').set('length','5')
    with pytest.raises(ValueError,match='gap'):road_point(r,10)


@pytest.mark.parametrize('change',['bytes','symlink'])
def test_map_drift_is_rejected(tmp_path,change):
    path,_=fixture(tmp_path)
    logical=tmp_path/'logical.xodr';logical.symlink_to(path)
    probe=MapProbe(logical)
    if change=='bytes':path.write_text(path.read_text()+'\n')
    else:
        second=tmp_path/'second.xodr';second.write_bytes(path.read_bytes())
        logical.unlink();logical.symlink_to(second)
    with pytest.raises(ValueError,match='changed'):probe.verify_unchanged()


def test_duplicate_road_is_rejected(tmp_path):
    path,_=fixture(tmp_path)
    path.write_text(path.read_text().replace('</OpenDRIVE>',road('10')+'</OpenDRIVE>'))
    with pytest.raises(ValueError,match='duplicate'):MapProbe(path)


def referenced_fixture(tmp_path,lane=-1):
    path,frame=fixture(tmp_path,lane)
    frame['actors'][1]['position'][0]=2.
    root=ET.fromstring(path.read_text());target='21' if lane<0 else '31'
    rd=root.find('road[@id="'+target+'"]')
    back=rd.find('link/'+('predecessor' if lane<0 else 'successor'))
    back.attrib=dict(elementType='road',elementId='10',contactPoint='end' if lane<0 else 'start')
    c=root.find('junction[@id="'+('20' if lane<0 else '30')+'"]/connection')
    c.set('contactPoint','start' if lane<0 else 'end')
    section=rd.find('lanes/laneSection')
    target_lane=section.find(('right' if lane<0 else 'left')+'/lane[@id="'+str(lane)+'"]')
    links=ET.SubElement(target_lane,'link')
    ET.SubElement(links,'predecessor' if lane<0 else 'successor',id=str(lane))
    signals=rd.find('signals')
    ET.SubElement(signals,'signal',id='900',type='206',name='Sign_Stop')
    ref=ET.SubElement(signals,'signalReference',id='900',s='0' if lane<0 else '20',orientation='-' if lane<0 else '+')
    ET.SubElement(ref,'validity',fromLane=str(lane),toLane=str(lane))
    path.write_text(ET.tostring(root,encoding='unicode'))
    return path,frame


@pytest.mark.parametrize('lane',[-1,1])
def test_signal_reference_reconstructs_upstream_trigger_without_authorizing_labels(tmp_path,lane):
    path,frame=referenced_fixture(tmp_path,lane);r=MapProbe(path).probe(frame,1)
    binding=r['control_reference']
    assert binding['status']=='unique_lane_stop_reference' and binding['signal_id']=='900'
    assert binding['upstream_predicted_trigger']['residual_m']==pytest.approx(0)
    assert not binding['capture_implementation_verified'] and not binding['label_authorization']
    assert not r['control_line_binding_verified']


@pytest.mark.parametrize('defect',['wrong_lane','wrong_direction','missing_definition','duplicate_definition',
                                  'wrong_predecessor','ambiguous_stop','wrong_type','outside_s'])
def test_reference_defects_cannot_establish_unique_stop(tmp_path,defect):
    path,frame=referenced_fixture(tmp_path)
    root=ET.fromstring(path.read_text());road=root.find('road[@id="21"]');signals=road.find('signals')
    ref=signals.find('signalReference');definition=signals.find('signal[@id="900"]')
    if defect=='wrong_lane':ref.find('validity').attrib=dict(fromLane='1',toLane='1')
    if defect=='wrong_direction':ref.set('orientation','+')
    if defect=='missing_definition':signals.remove(definition)
    if defect=='duplicate_definition':signals.append(deepcopy(definition))
    if defect=='wrong_predecessor':road.find('link/predecessor').set('elementId','99')
    if defect=='ambiguous_stop':
        other=deepcopy(definition);other.set('id','901');signals.append(other)
        other=deepcopy(ref);other.set('id','901');signals.append(other)
    if defect=='wrong_type':definition.set('type','205')
    if defect=='outside_s':ref.set('s','100')
    path.write_text(ET.tostring(root,encoding='unicode'))
    result=MapProbe(path).probe(frame,1)['control_reference']
    assert result['status']=='unresolved' and not result['label_authorization']
    assert 'upstream_predicted_trigger' not in result


def test_trigger_mismatch_remains_explicit_residual_not_false_map_certification(tmp_path):
    path,frame=referenced_fixture(tmp_path);frame['actors'][1]['position'][1]+=4
    result=MapProbe(path).probe(frame,1)['control_reference']
    assert result['upstream_predicted_trigger']['residual_m']==pytest.approx(4)
    assert not result['label_authorization'] and not result['capture_implementation_verified']


def test_fixed_anchor_survives_flag_and_lane_change_without_reassignment(tmp_path):
    path,frame=referenced_fixture(tmp_path);probe=FrozenApproachProbe(MapProbe(path),frame,1)
    changed=deepcopy(frame);changed['frame_id']+=1
    changed['meta'].update(road_id=99,lane_id=2)
    changed['actors'][1]['affects_ego']=False
    r=probe.observe(changed)
    assert r['errors']==[] and r['anchor']['source_road']==10
    assert r['current_road']==99 and r['fixed_approach_front_gap_m']==pytest.approx(3)
    assert not r['label_authorization']


@pytest.mark.parametrize('defect',['missing','identity','gap','route','geometry'])
def test_frozen_anchor_rejects_missing_or_unreliable_control(tmp_path,defect):
    path,frame=referenced_fixture(tmp_path);probe=FrozenApproachProbe(MapProbe(path),frame,1)
    changed=deepcopy(frame);changed['frame_id']+=1
    if defect=='missing':changed['actors'].pop()
    if defect=='identity':changed['actors'][1]['position'][0]+=5
    if defect=='gap':changed['frame_id']+=1
    if defect=='route':changed['route_id']='foreign'
    if defect=='geometry':changed['source_geometry_issues']=['bad geometry']
    r=probe.observe(changed)
    assert r['errors'] and 'fixed_approach_front_gap_m' not in r and not r['label_authorization']


def test_freezing_inactive_stop_is_rejected(tmp_path):
    path,frame=referenced_fixture(tmp_path);frame['actors'][1]['affects_ego']=False
    with pytest.raises(ValueError,match='inactive'):FrozenApproachProbe(MapProbe(path),frame,1)


def test_frozen_observation_detects_map_drift(tmp_path):
    path,frame=referenced_fixture(tmp_path);probe=FrozenApproachProbe(MapProbe(path),frame,1)
    path.write_text(path.read_text()+'\n');frame['frame_id']+=1
    with pytest.raises(ValueError,match='changed'):probe.observe(frame)


@pytest.mark.parametrize('axis',[0,1])
def test_displaced_initial_control_cannot_be_frozen(tmp_path,axis):
    path,frame=referenced_fixture(tmp_path);frame['actors'][1]['position'][axis]+=100
    with pytest.raises(ValueError,match='map reconstruction'):FrozenApproachProbe(MapProbe(path),frame,1)


@pytest.mark.parametrize('junction',['-1','999'])
def test_nonmatching_junction_does_not_apply_predecessor_relocation(tmp_path,junction):
    path,frame=referenced_fixture(tmp_path);root=ET.fromstring(path.read_text())
    root.find('road[@id="21"]').set('junction',junction)
    path.write_text(ET.tostring(root,encoding='unicode'))
    reference=MapProbe(path).probe(frame,1)['control_reference']
    assert reference['status']=='unresolved' and reference['errors']
    with pytest.raises(ValueError,match='unresolved'):FrozenApproachProbe(MapProbe(path),frame,1)


def test_wrong_lane_predecessor_is_not_unique_upstream_model(tmp_path):
    path,frame=referenced_fixture(tmp_path);root=ET.fromstring(path.read_text())
    root.find('road[@id="21"]/lanes/laneSection/right/lane/link/predecessor').set('id','-2')
    path.write_text(ET.tostring(root,encoding='unicode'))
    assert MapProbe(path).probe(frame,1)['control_reference']['status']=='unresolved'


def test_actor_cannot_drift_within_old_one_metre_identity_budget(tmp_path):
    path,frame=referenced_fixture(tmp_path);probe=FrozenApproachProbe(MapProbe(path),frame,1)
    frame['frame_id']+=1;frame['actors'][1]['position'][1]+=.1
    r=probe.observe(frame)
    assert 'control no longer matches frozen map reconstruction' in r['errors']
    assert 'fixed_approach_front_gap_m' not in r


def test_unknown_export_shape_cannot_inherit_fixed_box_model(tmp_path):
    path,frame=referenced_fixture(tmp_path);frame['actors'][1]['extent']=[2,2,1]
    with pytest.raises(ValueError,match='shape'):FrozenApproachProbe(MapProbe(path),frame,1)
