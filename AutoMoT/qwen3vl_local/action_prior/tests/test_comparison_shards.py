"""案例分片与真实CPU队列回归；不加载模型、不宣称多GPU验收。"""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior.comparison_cases import case_id, read_json, write_json
from qwen3vl_local.action_prior.comparison_shards import allocate_shards, plan_shards
from qwen3vl_local.action_prior.comparison_render import paired_cases, publish
from qwen3vl_local.action_prior.comparison_scheduler import run_queue
from qwen3vl_local.action_prior.contracts import file_hash


@pytest.mark.parametrize('gpus,capacities,expected', [
    (4, [20, 20], [2, 2]), (4, [20]*3, [2, 1, 1]), (4, [20]*6, [1]*6),
    (2, [20]*6, [1]*6), (1, [20, 20], [1, 1]), (8, [1, 1], [1, 1]),
    (4, [1, 10], [1, 3]), (8, [2, 2], [2, 2]), (4, [20], [4])])
def test_allocation(gpus, capacities, expected):
    assert allocate_shards(gpus, capacities) == expected


def setup_plan(out, model_count=2, train=5, test=3):
    models, jobs = [], []
    pools = {split: [dict(scenario='scene', run_id=split, anchor=i, route_group=split)
                     for i in range(n)] for split, n in [('train', train), ('test', test)]}
    for i in range(model_count):
        mid = f'model_{i+1:02d}'
        models.append(dict(id=mid))
        job = dict(seed=2026, checkpoint=f'ckpt_{i}', output=str(out/'_models'/mid), rows={}, row_hashes={})
        for split, rows in pools.items():
            path = out/'_plan'/f'{mid}_{split}.json'
            write_json(path, rows)
            job['rows'][split], job['row_hashes'][split] = str(path), file_hash(path)
        jobs.append(job)
    manifest = dict(models=models, splits={split: dict(cases={case_id(r): r for r in rows}) for split, rows in pools.items()})
    return jobs, manifest


@pytest.mark.parametrize('gpus,models,train,test', [(4,2,5,3),(4,3,5,3),(2,5,5,3),(8,2,1,0),(4,2,1,1),(4,2,0,6)])
def test_plan_exact_once_all_splits_preserves_seed_order_and_hashes(tmp_path,gpus,models,train,test):
    jobs, manifest = setup_plan(tmp_path,models,train,test)
    plan = dict(selected_ids=[str(i) for i in range(gpus)])
    tasks = plan_shards(jobs,manifest,plan,tmp_path)
    assert len(tasks) == max(models, min(gpus, models*(train+test)))
    assert plan['parallel_workers'] == min(len(tasks),gpus)
    for model in manifest['models']:
        seen = {split: [] for split in manifest['splits']}
        for task in [t for t in tasks if t['model'] == model['id']]:
            job = read_json(task['job'])
            assert job['seed'] == 2026 and task['samples'] > 0
            for split,path in job['rows'].items():
                assert file_hash(path) == job['row_hashes'][split]
                ids = [case_id(r) for r in read_json(path)]
                common = list(manifest['splits'][split]['cases'])
                assert ids == sorted(ids,key=common.index)
                seen[split].extend(ids)
        for split,ids in seen.items():
            assert len(ids) == len(set(ids))
            assert set(ids) == set(manifest['splits'][split]['cases'])
    assert len({read_json(t['job'])['output'] for t in tasks}) == len(tasks)


def test_plan_rejects_changed_rows(tmp_path):
    jobs,manifest = setup_plan(tmp_path)
    write_json(jobs[0]['rows']['train'], [])
    with pytest.raises(ValueError,match='清单已被修改'):
        plan_shards(jobs,manifest,dict(selected_ids=['0','1','2','3']),tmp_path)


def test_four_gpu_slots_two_models_real_cpu_workers_and_exact_merge(tmp_path):
    jobs, manifest = setup_plan(tmp_path)
    plan = dict(selected_ids=['0','1','2','3'])
    tasks = plan_shards(jobs,manifest,plan,tmp_path)
    worker = tmp_path/'worker.py'
    worker.write_text('''import json,sys,os,time
from pathlib import Path
job=json.loads(Path(sys.argv[1]).read_text())
time.sleep(.1)
for split,path in job['rows'].items():
    for i,row in enumerate(json.loads(Path(path).read_text())):
        root=Path(job['output'])/split/'cases'; root.mkdir(parents=True,exist_ok=True)
        trajectory=[[[0.,0.],[1.,0.]]]
        data=dict(sample=row,gt_route=trajectory,gt_waypoints=trajectory,gpu=os.environ['CUDA_VISIBLE_DEVICES'])
        (root/f'rank0_case{i:06d}.json').write_text(json.dumps(data))
''')
    commands = [[sys.executable,str(worker),t['job']] for t in tasks]
    records = run_queue(commands,plan,tmp_path,poll_interval=.005)
    assert len({r['gpu'] for r in records}) == 4
    assert {r['model'] for r in records} == {'model_01','model_02'}
    assert len({r['task'] for r in records}) == 4
    for split in manifest['splits']:
        merged = paired_cases(tmp_path,manifest,split)
        for model in manifest['models']:
            assert set(merged[model['id']]) == set(manifest['splits'][split]['cases'])
    # A missing shard result must fail the entire pairing, never silently shrink the sample.
    path = next((tmp_path/manifest['models'][0]['evaluation_shards'][0]['output']/'train/cases').glob('*.json'))
    path.unlink()
    with pytest.raises(ValueError,match='未覆盖'):
        paired_cases(tmp_path,manifest,'train')


def test_sharded_publish_matches_unsharded_summary_and_reads_actual_inputs(tmp_path):
    import shutil
    from qwen3vl_local.action_prior.tests.test_checkpoint_comparison import make_outputs
    manifest,cid = make_outputs(tmp_path)
    ids = [cid]
    for index in (1, 2):
        label = dict(manifest['splits']['test']['cases'][cid], anchor=12+index)
        new_id = case_id(label)
        ids.append(new_id)
        manifest['splits']['test']['cases'][new_id] = label
        for model in manifest['models']:
            source = tmp_path/'_models'/model['id']/'test'
            data = read_json(source/'cases/rank0_case000000.json')
            data['sample']['anchor'] = 12+index
            data['metrics'] = {key: value+index for key,value in data['metrics'].items()}
            write_json(source/'cases'/f'rank0_case{index:06d}.json', data)
            shutil.copytree(source/'inputs'/cid,source/'inputs'/new_id)
    for style,categories in manifest['splits']['test']['groups'].items():
        for category in categories:
            categories[category] = ids
            manifest['splits']['test']['coverage'][style][category].update(selected=3,shortfall=5,available_frames=3)
    publish(tmp_path,copy.deepcopy(manifest))
    reference = read_json(tmp_path/'summary.json')
    assert reference['action/test/STOP']['means']['model_01']['loss'] == pytest.approx(1.1)
    other = tmp_path/'sharded'
    for model in manifest['models']:
        source = tmp_path/'_models'/model['id']/'test'
        model['evaluation_shards'] = []
        for shard_index, indices in enumerate(([0,2],[1])):
            target = other/'_models'/model['id']/'shards'/f'shard_{shard_index+1:02d}'
            for index in indices:
                data = read_json(source/'cases'/f'rank0_case{index:06d}.json')
                write_json(target/'test/cases'/f'rank0_case{index:06d}.json',data)
                shutil.copytree(source/'inputs'/ids[index],target/'test/inputs'/ids[index])
            # Unequal shard means must never be averaged with equal weights.
            write_json(target/'test/metrics.json',dict(samples=len(indices),loss=-999))
            model['evaluation_shards'].append(dict(output=str(target.relative_to(other)),case_ids={'test':[ids[i] for i in indices]}))
    publish(other,manifest)
    assert read_json(other/'summary.json') == reference
    assert (other/'action/test/STOP'/cid/'comparison.png').is_file()
    assert (other/'action/test/STOP'/cid/'inputs/model_02/input_rgb_00.png').is_file()
