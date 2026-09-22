"""候选上限与逐类命中早停；真实发布及CPU常驻进程编排回归。"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior.comparison_cases import case_id, read_json, write_json
from qwen3vl_local.action_prior.comparison_search import CategorySearch, search_cases
from qwen3vl_local.action_prior.comparison_pool import SearchPool
from qwen3vl_local.action_prior.comparison_render import publish, paired_cases
from qwen3vl_local.action_prior.contracts import file_hash


def test_category_stops_at_five_other_category_continues_no_duplicates():
    groups={'event':{'A':list(range(50)), 'B':list(range(50,100))},'action':{'STOP':list(range(100))}}
    search=CategorySearch(groups,5)
    checked=[]
    while batch:=search.next_batch(8):
        for cid in batch:
            checked.append(cid)
            search.accept(cid,cid<50 or cid in (98,99))
    assert len(search.kept['event','A'])==5
    assert search.kept['event','B']==[98,99]
    assert len(search.kept['action','STOP'])==5
    assert len(checked)==len(set(checked))
    assert sum(cid<50 for cid in checked)<50
    assert search.report()['event/B']['reason']=='candidates_exhausted'
    assert search.report()['event/B']['shortfall']==3


def test_empty_and_overlap_and_all_misses():
    search=CategorySearch({'event':{'A':['a','b'], 'empty':[]},'action':{'STOP':['a','b']}},5)
    assert search.next_batch(8)==['a','b']
    search.accept('a',False);search.accept('b',False)
    assert search.next_batch(8)==[]
    assert all(v['retained']==0 for v in search.report().values())
    assert search.report()['event/A']['evaluated']==2
    disabled=CategorySearch({'event':{'off':[]}},5,disabled={('event','off')})
    assert disabled.report()['event/off']['reason']=='disabled'
    assert disabled.report()['event/off']['shortfall']==0


def make_search_fixture(tmp_path, count):
    from qwen3vl_local.action_prior.tests.test_checkpoint_comparison import make_outputs
    source=tmp_path/'source';manifest,oldid=make_outputs(source)
    oldlabel=manifest['splits']['test']['cases'][oldid]
    labels=[dict(oldlabel,anchor=i) for i in range(count)]
    ids=[case_id(label) for label in labels]
    plan=manifest['splits']['test'];plan['cases']=dict(zip(ids,labels))
    plan['effective_frames']=count;plan['selected_unique_frames']=count
    for style,cats in plan['groups'].items():
        for name in cats:
            cats[name]=list(ids)
            plan['coverage'][style][name].update(requested=50,selected=count,shortfall=50-count)
    manifest['error_filter']=dict(enabled=True,ade_threshold_m=1.,fde_threshold_m=3.)
    out=tmp_path/'result';jobs=[]
    for model in manifest['models']:
        data=read_json(source/'_models'/model['id']/'test/cases/rank0_case000000.json')
        rows=[dict(data['sample'],anchor=i) for i in range(count)]
        path=out/'_plan'/f"{model['id']}_test.json";write_json(path,rows)
        jobs.append(dict(output=str(out/'_models'/model['id']),checkpoint='fake',seed=2026,workers=0,
                         rows={'test':str(path)},row_hashes={'test':file_hash(path)}))
    return manifest,jobs,out,source,oldid


@pytest.mark.parametrize('count,hits,expected,checked',[(50,set(range(50)),5,8),(50,{2,49},2,50),(3,set(),0,3)])
def test_search_pipeline_budget_quota_pairing_and_publish(tmp_path,count,hits,expected,checked):
    manifest,jobs,out,source,oldid=make_search_fixture(tmp_path,count)
    calls=[]
    class FakePool:
        def __init__(self,*args): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def run(self,tasks):
            for task in tasks:
                job=read_json(task['job']);mid=task['model']
                template=read_json(source/'_models'/mid/'test/cases/rank0_case000000.json')
                for index,row in enumerate(read_json(job['rows']['test'])):
                    calls.append((mid,row['anchor']))
                    data=copy.deepcopy(template);data['sample']=row
                    if row['anchor'] in hits and mid=='model_02':
                        for point in data['pred_waypoints'][0]: point[0]+=2.
                    root=Path(job['output'])/'test'
                    write_json(root/'cases'/f'rank0_case{index:06d}.json',data)
                    shutil.copytree(source/'_models'/mid/'test/inputs'/oldid,root/'inputs'/case_id(row))
    search_cases(jobs,manifest,dict(selected_ids=['0','1','2','3']),out,5,pool_factory=FakePool)
    assert len(calls)==2*checked and len(calls)==len(set(calls))
    report=read_json(out/'search.json')
    assert report['status']=='complete'
    group=report['splits']['test']['event/UE1']
    assert group['retained']==expected and group['evaluated']==checked
    assert group['reason']==('target_reached' if expected==5 else 'candidates_exhausted')
    assert len(manifest['splits']['test']['groups']['action']['STOP'])==expected
    assert len(paired_cases(out,manifest,'test')['model_01'])==checked
    publish(out,manifest)
    assert len(list((out/'action/test/STOP').glob('*/comparison.png')))==expected
    summary=read_json(out/'summary.json')['action/test/STOP']
    assert summary['visualization']['sampled']==checked
    assert summary['visualization']['retained']==expected
    assert (out/'candidate_plan.json').is_file()


def test_persistent_cpu_services_reused_across_batches(tmp_path):
    worker=tmp_path/'service.py'
    worker.write_text('''import sys,json,os
from pathlib import Path
for line in sys.stdin:
    r=json.loads(line)
    job=json.loads(Path(r['job']).read_text())
    Path(job['output']).write_text(str(os.getpid()))
    tmp=Path(r['done']+'.tmp');tmp.write_text('{"status":"complete"}');tmp.replace(r['done'])
''')
    with SearchPool(tmp_path,dict(selected_ids=['0','1']),command=[sys.executable,str(worker)]) as pool:
        pids=[]
        for batch in range(2):
            tasks=[]
            for model in ('a','b'):
                job=tmp_path/f'{batch}_{model}.json'
                write_json(job,dict(output=str(tmp_path/f'{batch}_{model}.pid')))
                tasks.append(dict(id=f'{batch}_{model}',model=model,job=str(job),samples=1))
            pool.run(tasks)
            pids.append([int((tmp_path/f'{batch}_{m}.pid').read_text()) for m in ('a','b')])
        assert pids[0]==pids[1] and len(set(pids[0]))==2
    for pid in pids[0]:
        with pytest.raises(ProcessLookupError):os.kill(pid,0)


def test_service_failure_reaps_other_process(tmp_path):
    worker=tmp_path/'fail.py'
    worker.write_text('import sys\nsys.stdin.readline()\nraise SystemExit(7)\n')
    with pytest.raises(RuntimeError,match='exited 7'):
        with SearchPool(tmp_path,dict(selected_ids=['0','1']),command=[sys.executable,str(worker)]) as pool:
            pids=[w['process'].pid for w in pool.active.values()]
            job=tmp_path/'job.json';write_json(job,{})
            pool.run([dict(id='fail',model='a',job=str(job),samples=1)])
    for pid in pids:
        with pytest.raises(ProcessLookupError):os.kill(pid,0)


def test_pool_sigterm_cleans_children(tmp_path):
    worker=tmp_path/'interrupt_worker.py'
    worker.write_text('import os,signal,sys,time\nsys.stdin.readline()\nos.kill(os.getppid(),signal.SIGTERM)\ntime.sleep(20)\n')
    job=tmp_path/'job.json';write_json(job,{})
    script=tmp_path/'parent.py'
    script.write_text(f'''from qwen3vl_local.action_prior.comparison_pool import SearchPool
with SearchPool({str(tmp_path)!r},dict(selected_ids=['0','1']),command=[{sys.executable!r},{str(worker)!r}]) as pool:
    pool.run([dict(id='signal',model='a',job={str(job)!r},samples=1)])
''')
    run=subprocess.run([sys.executable,str(script)],env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[3])},
                       capture_output=True,text=True,timeout=10)
    assert run.returncode==143,run.stderr
    for worker in read_json(tmp_path/'scheduler.json')['workers']:
        with pytest.raises(ProcessLookupError):os.kill(worker['pid'],0)
