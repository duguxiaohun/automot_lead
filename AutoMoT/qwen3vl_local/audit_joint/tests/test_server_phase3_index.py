import json

from qwen3vl_local.audit_joint import server_phase3_index as tool
from qwen3vl_local.sft_new_loop_phase3.test_entry_admission import fixture
from qwen3vl_local.sft_new_loop_phase3.prompt_candidate import CANDIDATE_NAME


def test_inventory_does_not_choose_or_read_indexes(tmp_path):
    folder=tmp_path/'candidate';folder.mkdir()
    (folder/'frame_index.jsonl').write_text('not JSON')
    report=tool.inventory(tmp_path)
    assert report['paths']==[str(folder/'frame_index.jsonl')]
    assert report['scope'].startswith('bounded')


def test_admission_keeps_manifest_and_checks_real_index(tmp_path):
    index=fixture(tmp_path,'valid')
    report,members=tool.inspect(index,CANDIDATE_NAME,'choice')
    assert report['status']=='index_admitted'
    assert members['manifest.json']==json.loads(index.with_name('manifest.json').read_text())
    assert set(report['bindings'])=={'index','manifest'}


def test_missing_or_bad_index_leaves_diagnostic(tmp_path):
    index=fixture(tmp_path,'development')
    report,members=tool.inspect(index,CANDIDATE_NAME,'choice')
    assert report['status']=='blocked' and 'development route' in report['error']
    assert 'manifest.json' in members
    index.with_name('manifest.json').unlink()
    report,members=tool.inspect(index,CANDIDATE_NAME,'choice')
    assert report['status']=='blocked' and not members
