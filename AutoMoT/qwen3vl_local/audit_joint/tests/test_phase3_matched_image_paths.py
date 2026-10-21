import copy
from pathlib import Path

import pytest
from PIL import Image, ImageStat

from qwen3vl_local.audit_joint import phase3_behavior_diagnostics as diag
from qwen3vl_local.audit_joint import phase3_matched_image_eval as evaluation
from qwen3vl_local.audit_joint.phase3_base_dense import provenance
from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group


def example(tmp_path, mount='scenario'):
    root=tmp_path/'lead_data';external=tmp_path/'mounted'
    external.mkdir();scenario='HardBreakRoute'
    if mount=='root':root.symlink_to(external,target_is_directory=True)
    else:root.mkdir()
    if mount=='scenario':
        (external/scenario).mkdir();(root/scenario).symlink_to(external/scenario,target_is_directory=True)
    else:(root/scenario).mkdir()
    records={};rows=[];sources={}
    for i in (1,2):
        route=f'Town13_Rep0_{i}_0_route0_01_08_23_15_29'
        folder=root/scenario/route
        if mount=='route':
            target=external/route;target.mkdir();folder.symlink_to(target,target_is_directory=True)
        else:folder.mkdir()
        (folder/'rgb').mkdir()
        logical=folder/'rgb'/f'{i:04d}.png'
        if mount=='image':
            physical=external/f'{i}.png';logical.symlink_to(physical)
        else:physical=logical
        Image.new('RGB',(12,4),(40+i,40+i,40+i)).save(physical)
        relative=str(logical.relative_to(root))
        with Image.open(logical) as im:brightness=ImageStat.Stat(im.convert('L')).mean[0]
        rows.append(dict(id=str(i),split='val',event='LEAD_BRAKE',
            group=physical_route_group(scenario,route),image=relative,
            rgb_sha256=diag.file_sha(logical),image_size=[12,4],speed=3.,brightness=brightness))
        records[str(i)]=dict(id=str(i),split='val',event='LEAD_BRAKE',scenario=scenario,
            route_id=route,anchor=i,image=str(logical))
        sources[relative]=dict(resolved_path=str(logical.resolve()),stat=provenance.stamp(logical),sha256=diag.file_sha(logical))
    return root,rows,records,dict(sources=sources)


@pytest.mark.parametrize('mount',['scenario','route','image','root','none'])
def test_native_logical_mounts_and_unchanged_records(tmp_path,mount):
    root,rows,records,proof=example(tmp_path,mount)
    plan=diag.matched_plan(rows,11);before=copy.deepcopy(records)
    assert len(plan['pairs'])==2
    if mount in ('scenario','route','image'):
        # Reproduce the exact previous failure condition on correctly bound data.
        assert all(not (root/r['image']).resolve().is_relative_to(root.resolve()) for r in rows)
    result=evaluation.checked_pairs(plan,records,root,proof)
    assert set(result)=={'1','2'} and records==before


def test_identical_bytes_at_different_record_file_rejected_with_paths(tmp_path):
    root,rows,records,proof=example(tmp_path)
    other=tmp_path/'other.png';other.write_bytes(Path(records['1']['image']).read_bytes())
    records['1']['image']=str(other)
    with pytest.raises(ValueError,match='different admitted file') as caught:
        evaluation.checked_pairs(diag.matched_plan(rows,11),records,root,proof)
    assert 'plan_resolved' in str(caught.value) and 'record_resolved' in str(caught.value)


@pytest.mark.parametrize('bad',['/absolute.png','../outside.png','HardBreakRoute/wrong_route/rgb/0001.png'])
def test_invalid_logical_plan_path_rejected(tmp_path,bad):
    root,rows,records,proof=example(tmp_path)
    rows[0]['image']=bad
    with pytest.raises(ValueError,match='logical|relative'):
        evaluation.checked_pairs(diag.matched_plan(rows,11),records,root,proof)


def test_retargeted_image_symlink_cannot_bypass_admitted_proof(tmp_path):
    root,rows,records,proof=example(tmp_path,'image')
    logical=Path(records['1']['image']);other=tmp_path/'same_bytes.png'
    other.write_bytes(logical.read_bytes());logical.unlink();logical.symlink_to(other)
    with pytest.raises(ValueError,match='raw provenance'):
        evaluation.checked_pairs(diag.matched_plan(rows,11),records,root,proof)


def test_modified_pixels_and_missing_provenance_rejected(tmp_path):
    root,rows,records,proof=example(tmp_path)
    del proof['sources'][rows[0]['image']]
    with pytest.raises(ValueError,match='raw provenance'):
        evaluation.checked_pairs(diag.matched_plan(rows,11),records,root,proof)
    Image.new('RGB',(12,4),(90,90,90)).save(records['1']['image'])
    with pytest.raises(ValueError,match='RGB changed'):
        evaluation.checked_pairs(diag.matched_plan(rows,11),records,root)
