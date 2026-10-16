from collections import Counter
import json
from pathlib import Path
import sys

import pytest

from qwen3vl_local.audit_joint import phase4_quick_benchmark as q
from qwen3vl_local.audit_joint import benchmark_runtime as runtime
from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS


def rows(n=40):
    return [dict(id=f'{e}/{f}',split='train',target='YES',episode={'event':e,'instance_id':e},
        scenario=e,route_id='route',edge='proceed',physical_group=e,
        observation={'frame_id':f},padding='x'*100) for e in EVENTS for f in range(10,10+n)]


def write_rows(path, values):
    path.write_text(''.join(json.dumps(r)+'\n' for r in values))


def test_window_scan_reaches_later_events_with_bounded_io(tmp_path):
    path=tmp_path/'train.jsonl';write_rows(path,rows(100))
    budget=path.stat().st_size//4
    selected,proof=q.scan_train(path,per_event=1,candidates_per_event=2,max_rows=128,max_bytes=budget)
    assert Counter(r['episode']['event'] for r in selected)==dict.fromkeys(EVENTS,2)
    assert proof['bytes_read']<=budget and proof['rows_scanned']<=128
    assert len(proof['windows'])>1 and not proof['full_train_sha_verified']


def test_short_scan_never_requires_a_full_pool(tmp_path):
    path=tmp_path/'train.jsonl';write_rows(path,rows(2))
    with path.open('a') as f:f.write('THIS MUST NOT BE PARSED\n')
    selected,proof=q.scan_train(path,per_event=2,candidates_per_event=2,max_rows=100,max_bytes=100000)
    assert len(selected)==20 and proof['rows_scanned']==20


def test_missing_event_and_holdout_rejected(tmp_path):
    path=tmp_path/'train.jsonl';write_rows(path,rows(2)[:-2])
    with pytest.raises(ValueError,match='no GPU trial started'):
        q.scan_train(path,per_event=1,candidates_per_event=2,max_rows=100,max_bytes=100000)
    r=rows(1);r[0]['split']='val';write_rows(path,r)
    with pytest.raises(ValueError,match='non-training'):
        q.scan_train(path,per_event=1,candidates_per_event=2,max_rows=100,max_bytes=100000)


def fixture(tmp_path,monkeypatch,mode=2):
    from qwen3vl_local.sft_new_loop_phase4 import identity
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import data
    native=tmp_path/'native';native.mkdir()
    write_rows(native/'train.jsonl',rows())
    (native/'manifest.json').write_text(json.dumps({'contract':{},'rgb_mode':4}))
    (tmp_path/'config.json').write_text('{}')
    output=tmp_path/'prepared';output.mkdir()
    argv=['--native',str(native),'--data-root',str(tmp_path),'--model-dir',str(tmp_path),
          '--output',str(output),'--rgb-mode',str(mode)]
    a=q.parser().parse_args(argv)
    monkeypatch.setattr(identity,'check_contract',lambda c:None)
    monkeypatch.setattr(q,'source_binding',lambda:{'source':'fixed'})
    monkeypatch.setattr(data,'image_identities',lambda *a:{})
    monkeypatch.setattr(data,'paired_view',lambda native,*a,**k:(native,[]))
    monkeypatch.setattr(data,'load_dataset',lambda *a,**k:pytest.fail('must not load full dataset'))
    return a,argv


@pytest.mark.parametrize('mode',[2,4])
def test_fixed_selection_across_batches_and_source_bound_reuse(tmp_path,monkeypatch,mode):
    a,argv=fixture(tmp_path,monkeypatch,mode);q.prepare(a)
    path=a.output/'prepared.json';d=q.check_prepared(path,a)
    assert len(d['items'])==80 and d['selection']['unique_questions']==80
    assert d['selection']['event_counts']==dict.fromkeys(EVENTS,8)
    assert not d['diagnostic']['full_pool_admitted']
    a.prepared_samples=path;a.prepared_sha256=q.file_sha(path);a.modes=[mode]
    ids=[]
    for batch in (1,2):
        a.batches=[batch]
        items,records,manifest,updates,selection=runtime.read_prepared(a)
        assert selection['accumulation']==2//batch
        assert selection['anchor_stride']==(2 if mode==2 else 1)
        ids.append([[items[i]['id'] for i in block] for block in updates])
        for block in updates:
            for rank in range(4):
                for micro in q.benchmark.rank_microbatches(block,rank,4,batch):
                    assert len(set(records[i]['id'] for i in micro))==batch
    assert ids[0]==ids[1]
    (a.native/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='binding mismatch'):q.check_prepared(path,a)


def test_changed_records_rejected(tmp_path,monkeypatch):
    a,_=fixture(tmp_path,monkeypatch);q.prepare(a);path=a.output/'prepared.json'
    d=json.loads(path.read_text());d['records'][0]['event']='other';q.benchmark.write(path,d)
    with pytest.raises(ValueError,match='records differ'):q.check_prepared(path,a)


def test_prepare_timeout_stops_before_any_gpu(tmp_path,monkeypatch):
    a,argv=fixture(tmp_path,monkeypatch);calls=[]
    argv[argv.index('--output')+1]=str(tmp_path/'run')
    monkeypatch.setattr(sys,'argv',['quick',*argv])
    monkeypatch.setattr(q,'run_owned',lambda *args:calls.append(args) or 124)
    with pytest.raises(SystemExit) as ex:q.main()
    assert ex.value.code==124 and len(calls)==1
    cmd,log,env,timeout=calls[0]
    assert '--prepare-worker' in cmd and env['CUDA_VISIBLE_DEVICES']=='' and timeout==60
    result=json.loads((tmp_path/'run/summary.json').read_text())
    assert result['status']=='preparation_failed' and not result['full_pool_admitted']


def test_selection_reuse_and_first_trial_failure_stop(tmp_path,monkeypatch):
    a,argv=fixture(tmp_path,monkeypatch);q.prepare(a);calls=[]
    argv[argv.index('--output')+1]=str(tmp_path/'run')
    argv+=['--selection',str(a.output/'prepared.json')]
    monkeypatch.setattr(sys,'argv',['quick',*argv])
    monkeypatch.setattr(q,'run_owned',lambda *args:calls.append(args) or 124)
    with pytest.raises(SystemExit) as ex:q.main()
    assert ex.value.code==124 and len(calls)==1
    cmd,log,env,timeout=calls[0]
    assert '--prepare-worker' not in cmd and '--worker' in cmd and timeout==180
    assert env['CUDA_VISIBLE_DEVICES']=='0,1,2,3'
    report=json.loads((tmp_path/'run/summary.json').read_text())
    assert 'fastest_measured_candidates' not in report and not report['full_pool_admitted']
    assert len(report['trials'])==1


def test_summary_cannot_be_mistaken_for_production_approval():
    result=q.summary([])
    assert not result['full_pool_admitted'] and 'fastest_measured_candidates' not in result
    assert 'production' in result['note']


def test_prepare_only_needs_no_model_weights(tmp_path,monkeypatch):
    a,argv=fixture(tmp_path,monkeypatch)
    (tmp_path/'config.json').unlink();argv+=['--prepare-only','--selection',str(a.output/'prepared.json')]
    q.prepare(a);argv[argv.index('--output')+1]=str(tmp_path/'run')
    monkeypatch.setattr(sys,'argv',['quick',*argv])
    monkeypatch.setattr(q,'run_owned',lambda *a:pytest.fail('reuse must not start processes'))
    q.main()
    assert json.loads((tmp_path/'run/summary.json').read_text())['status']=='prepared_cpu_only'
