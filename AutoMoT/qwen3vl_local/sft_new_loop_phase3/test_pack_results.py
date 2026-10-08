import json
from pathlib import Path
import tarfile

import pytest
from . import pack_results as pack


def fixture(root, rgb='4rgb', action='binary', count=503):
    directory=root/'eval/lora_production';directory.mkdir(parents=True)
    metrics=dict(history_rgb_mode=rgb,action_output_mode=action,total_cases=count)
    (directory/'metrics.json').write_text(json.dumps(metrics))
    rows=[dict(case_index=i,all_ok=i%2==0,history_rgb_mode=rgb,prompt_spec=dict(action_output_mode=action)) for i in range(count)]
    (directory/'cases_rank0.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (directory/'cases_rank1.jsonl').write_text('')
    image=directory/'error_cases/context/case_1/rgb/0.jpg';image.parent.mkdir(parents=True);image.write_bytes(b'original RGB bytes')
    (image.parent.parent/'case.json').write_text(json.dumps(rows[1]))
    (directory/'model.safetensors').write_bytes(b'do not include weights')
    (root/'pipeline_manifest.json').write_text(json.dumps(dict(status='completed_without_historical_regression')))
    return directory.parent


@pytest.mark.parametrize('rgb,action', sorted(pack.GROUPS))
def test_complete_posthoc_archive_without_model_or_rgb_source(tmp_path,rgb,action):
    root=fixture(tmp_path/'run',rgb,action)
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    archive=tmp_path/'export.tar.gz'
    report=pack.package(root,archive)
    assert report['evaluations']['lora_production']['cases']==503
    with tarfile.open(archive) as tar:
        names=tar.getnames()
        assert not any(name.endswith('.safetensors') for name in names)
        assert tar.extractfile('audit_bundle/lora_production/cases_rank0.jsonl').read()==(root/'lora_production/cases_rank0.jsonl').read_bytes()
        assert tar.extractfile('audit_bundle/lora_production/error_cases/context/case_1/rgb/0.jpg').read()==b'original RGB bytes'
        manifest=json.load(tar.extractfile('audit_bundle/bundle_manifest.json'))
        assert manifest['inference_performed'] is False
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError): pack.package(root,archive)


@pytest.mark.parametrize('failure', ['missing_rows','duplicate','mixed_mode'])
def test_rejects_incomplete_or_mixed_results(tmp_path,failure):
    root=fixture(tmp_path/'run')
    path=root/'lora_production/cases_rank0.jsonl'
    lines=path.read_text().splitlines()
    if failure=='missing_rows': lines.pop()
    if failure=='duplicate': lines[-1]=lines[0]
    if failure=='mixed_mode':
        row=json.loads(lines[-1]);row['history_rgb_mode']='2rgb_endpoints';lines[-1]=json.dumps(row)
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError): pack.package(root,tmp_path/'failed.tar.gz')
    assert not (tmp_path/'failed.tar.gz').exists()


def test_latest_four_selects_completed_groups_and_requires_missing_groups(tmp_path):
    with pytest.raises(ValueError,match='Missing completed groups'): pack.discover(tmp_path)
    expected=[]
    for rgb,action in sorted(pack.GROUPS): expected.append(fixture(tmp_path/f'{rgb}_{action}',rgb,action))
    failed=fixture(tmp_path/'failed')
    (failed.parent/'pipeline_manifest.json').write_text(json.dumps(dict(status='failed')))
    assert pack.discover(tmp_path)==expected


def test_source_mutation_during_packaging_does_not_publish(tmp_path,monkeypatch):
    root=fixture(tmp_path/'run');source=root/'lora_production/metrics.json'
    original=pack.tarfile.TarFile.add
    def mutate(self,*args,**kwargs):
        original(self,*args,**kwargs)
        if Path(args[0])==source: source.write_text(source.read_text()+'\n')
    monkeypatch.setattr(pack.tarfile.TarFile,'add',mutate)
    with pytest.raises(ValueError,match='changed while packaging'): pack.package(root,tmp_path/'failed.tar.gz')
    assert not (tmp_path/'failed.tar.gz').exists()



def test_default_cli_packages_all_four_with_exact_receipt(tmp_path,monkeypatch):
    import sys
    expected=[]
    for rgb,action in sorted(pack.GROUPS): expected.append(fixture(tmp_path/'runs'/f'{rgb}_{action}',rgb,action))
    out=tmp_path/'exports'
    monkeypatch.setattr(sys,'argv',['pack_results','--pipeline-root',str(tmp_path/'runs'),'--output-dir',str(out)])
    pack.main()
    assert len(list(out.glob('*.tar.gz')))==4
    report=json.loads(next(out.glob('packaging_*.json')).read_text())
    assert report['status']=='completed' and report['all_four_groups'] is True
    assert {tuple(p['group']) for p in report['packages']}==pack.GROUPS
    assert {p['source_eval'] for p in report['packages']}==set(map(str,expected))
    for p in report['packages']:
        with tarfile.open(p['archive']) as tar:
            manifest=json.load(tar.extractfile('audit_bundle/bundle_manifest.json'))
            assert manifest['group']==p['group']


def test_no_partial_batch_when_a_group_has_incomplete_cases(tmp_path,monkeypatch):
    import sys
    for rgb,action in sorted(pack.GROUPS): root=fixture(tmp_path/'runs'/f'{rgb}_{action}',rgb,action)
    (root/'lora_production/cases_rank0.jsonl').write_text('')
    out=tmp_path/'exports'
    monkeypatch.setattr(sys,'argv',['pack_results','--pipeline-root',str(tmp_path/'runs'),'--output-dir',str(out)])
    with pytest.raises(ValueError,match='incomplete'): pack.main()
    assert not list(out.glob('*.tar.gz'))
