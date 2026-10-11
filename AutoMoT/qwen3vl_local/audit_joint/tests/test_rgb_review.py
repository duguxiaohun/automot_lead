import json
from pathlib import Path
import pytest
from PIL import Image
from qwen3vl_local.audit_joint import rgb_review as r
from qwen3vl_local.audit_joint.io import file_sha,write_json
from qwen3vl_local.sft_new_loop_phase4.input_identity import rgb_content_sha


def fixture(tmp):
    root=tmp/'rgb';root.mkdir();sources=[];pixels=[]
    for n in range(4):
        image=Image.new('RGB',(12,8),(n*40,10,20));path=root/f'{n}.png';image.save(path)
        sources.append(dict(path=path.name,sha256=file_sha(path),frame_id=n,kind='rgb'));pixels.append(rgb_content_sha(image))
    inv=tmp/'inventory';inv.mkdir();rows=[]
    for mode,idx in ((2,[1,3]),(4,[0,1,2,3])):
        rows.append(dict(id='a',rgb_mode=mode,split='train',scenario='s',route_id='r',frame_id=3,event='U-E1',edge='complete',source_row_sha256='original',
            reference_target='YES',original_label_basis='weak_rule_teacher',facts={'secret_condition':True},
            input=dict(images=[sources[i]['path'] for i in idx],file_sha256=[sources[i]['sha256'] for i in idx],pixel_sha256=[pixels[i] for i in idx],history_frames=idx,system='system',prompt='question',model_input_sha256=f'input{mode}'),
            evidence=dict(sources=sources)))
    (inv/'input_evidence.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows));write_json(inv/'summary.json',{})
    write_json(inv/'receipt.json',dict(files={n:dict(bytes=(inv/n).stat().st_size,sha256=file_sha(inv/n)) for n in ('input_evidence.jsonl','summary.json')}))
    return inv,root,tmp/'review'


def start(tmp):
    inv,root,out=fixture(tmp);r.prepare(inv,root,['a'],out,'developer','known');return inv,root,out


def answers():return [dict(id='a',answer='UNKNOWN',reason='gap change cannot be established from displayed frames')]


def test_stages_hide_reference_until_both_judgments_locked(tmp_path):
    _,_,out=start(tmp_path)
    for stage,previous in [('rgb2',None),('rgb4','rgb2')]:
        if previous:r.lock(out,previous,answers());r.reveal(out,stage)
        raw=(out/stage/'cases.json').read_text()
        assert 'secret_condition' not in raw and 'reference_target' not in raw and '"reference"' not in raw
    with pytest.raises(FileNotFoundError):r.reveal(out,'history')
    r.lock(out,'rgb4',answers());cases=r.reveal(out,'history')
    assert cases[0]['reference']['facts']=={'secret_condition':True}
    assert len(cases[0]['sources'])==4
    assert json.loads((out/'rgb2/judgments.json').read_text())['g2_status']=='not_run'


def test_no_overwrite_or_unrecorded_reveal(tmp_path):
    inv,root,out=start(tmp_path)
    with pytest.raises(FileExistsError):r.prepare(inv,root,['a'],out,'reviewer','known')
    with pytest.raises(FileNotFoundError):r.reveal(out,'rgb4')
    r.lock(out,'rgb2',answers())
    with pytest.raises(FileExistsError):r.lock(out,'rgb2',answers())
    with pytest.raises(FileExistsError):r.reveal(out,'rgb2')


@pytest.mark.parametrize('mutation',['inventory','request','stage','rgb'])
def test_changed_sources_fail(tmp_path,mutation):
    inv,root,out=start(tmp_path)
    if mutation=='inventory':(inv/'input_evidence.jsonl').write_text('{}\n')
    elif mutation=='request':
        j=json.loads((out/'request.json').read_text());j['reviewer']='other';(out/'request.json').write_text(json.dumps(j))
    elif mutation=='stage':(out/'rgb2/cases.json').write_text('[]')
    else:
        r.lock(out,'rgb2',answers());(root/'0.png').write_bytes(b'changed')
        with pytest.raises(ValueError,match='source bytes'):r.reveal(out,'rgb4')
        assert not (out/'rgb4/receipt.json').exists();return
    with pytest.raises(ValueError):r.lock(out,'rgb2',answers())


@pytest.mark.parametrize('judgments',[[],[dict(id='a',answer='YES',reason='')],[dict(id='a',answer='MAYBE',reason='uncertain')],[dict(id='b',answer='NO',reason='other')]])
def test_incomplete_or_invalid_decisions_rejected(tmp_path,judgments):
    _,_,out=start(tmp_path)
    with pytest.raises(ValueError):r.lock(out,'rgb2',judgments)
    assert not (out/'rgb2/judgments.json').exists()


def test_changed_previous_lock_detected_after_next_stage(tmp_path):
    _,_,out=start(tmp_path);r.lock(out,'rgb2',answers());r.reveal(out,'rgb4')
    (out/'rgb2/judgments.json').write_text('{}')
    with pytest.raises(ValueError):r.lock(out,'rgb4',answers())


def test_path_traversal_and_pixel_drift_rejected(tmp_path):
    _,root,_=fixture(tmp_path)
    with pytest.raises(ValueError):r.checked_image(root,dict(path='../outside.png',sha256='x'))
    with pytest.raises(ValueError,match='pixels'):r.checked_image(root,dict(path='0.png',sha256=file_sha(root/'0.png')),'wrong')


def test_missing_pair_and_nontrain_rejected(tmp_path):
    inv,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='missing paired'):r.selected_rows(inv,['missing'])
    path=inv/'input_evidence.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()];rows[0]['split']='val';path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    receipt=json.loads((inv/'receipt.json').read_text());receipt['files'][path.name]=dict(bytes=path.stat().st_size,sha256=file_sha(path));(inv/'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='non-training'):r.selected_rows(inv,['a'])


def test_source_drift_between_view_and_lock_rejected(tmp_path):
    _,root,out=start(tmp_path);(root/'1.png').write_bytes(b'changed after view')
    with pytest.raises(ValueError,match='source bytes'):r.lock(out,'rgb2',answers())
    assert not (out/'rgb2/judgments.json').exists()


def test_output_cannot_modify_raw_data_or_inventory(tmp_path):
    inv,root,_=fixture(tmp_path)
    for output in (root/'review',inv/'review'):
        with pytest.raises(ValueError,match='outside'):r.prepare(inv,root,['a'],output,'reviewer','known')
        assert not output.exists()
