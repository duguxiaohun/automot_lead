from pathlib import Path
import pytest
from qwen3vl_local.audit_joint import phase3_maps as maps

def write_root(root,style='flat'):
    for town in maps.TOWNS:
        path=(root/(town+'.xodr') if style=='flat' else root/town/'OpenDrive'/(town+'.xodr'))
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(town)

def test_local_known_root_fallback_and_idempotent_view(tmp_path):
    app=tmp_path/'AutoMoT';app.mkdir()
    local=tmp_path/'lead/3rd_party/fail2drive/toolbox/carla_xodr';write_root(local)
    view,proof=maps.resolve(app,tmp_path/'view')
    assert len(proof['towns'])==12 and Path(proof['root'])==local
    assert all((view/(t+'.xodr')).is_symlink() for t in maps.TOWNS)
    assert maps.resolve(app,view)==(view,proof)

def test_remote_carla_complete_root_preferred_without_local_assumption(tmp_path):
    app=tmp_path/'app';carla=app/'CARLA_0915/CarlaUE4/Content/Carla/Maps';write_root(carla,'nested')
    fallback=app/'lead/3rd_party/fail2drive/toolbox/carla_xodr';write_root(fallback)
    assert maps.discover(app)['root']==str(carla)
    assert maps.discover(app,fallback)['root']==str(fallback)

def test_explicit_incomplete_root_does_not_fallback(tmp_path):
    root=tmp_path/'incomplete';root.mkdir()
    with pytest.raises(ValueError,match='no_complete_known_map_root'):maps.discover(tmp_path,root)

def test_conflicts_and_changed_existing_view_reject_before_writes(tmp_path):
    root=tmp_path/'maps';write_root(root)
    duplicate=root/'OpenDrive/Town01.xodr';duplicate.parent.mkdir();duplicate.write_text('different')
    with pytest.raises(ValueError,match='conflicting_project_maps'):maps.resolve(tmp_path,tmp_path/'view',root)
    assert not (tmp_path/'view').exists()
    duplicate.write_text('Town01');view=tmp_path/'view';view.mkdir();(view/'Town01.xodr').write_text('old')
    with pytest.raises(ValueError,match='existing_map_view_changed'):maps.resolve(tmp_path,view,root)
    assert len(list(view.iterdir()))==1
