"""按可用GPU将既定配对案例分片，不改变采样、模型条件或评估噪声。"""
from pathlib import Path

from qwen3vl_local.action_prior.comparison_cases import read_json, write_json, case_id
from qwen3vl_local.action_prior.contracts import file_hash


def allocate_shards(gpus, capacities):
    """模型先各一份，空余卡均匀分摊，分片数不超过该模型的案例数。"""
    if gpus < 1 or not capacities or any(c < 1 for c in capacities):
        raise ValueError("分片需要正数GPU及每模型至少一个case")
    counts = [1] * len(capacities)
    for _ in range(max(0, min(gpus, sum(capacities)) - len(counts))):
        eligible = [i for i, count in enumerate(counts) if count < capacities[i]]
        index = min(eligible, key=lambda i: (counts[i], i))
        counts[index] += 1
    return counts


def plan_shards(jobs, manifest, gpu_plan, out):
    """每片独立输出/缓存/日志，父进程留存全量配对清单及各片身份清单。"""
    out = Path(out)
    if len(jobs) != len(manifest['models']):
        raise ValueError("模型与job数量不一致")
    pools = []
    for job in jobs:
        pool = {}
        for split, path in job['rows'].items():
            if file_hash(path) != job['row_hashes'][split]:
                raise ValueError("分片前case清单已被修改")
            rows = read_json(path)
            if [case_id(r) for r in rows] != list(manifest['splits'][split]['cases']):
                raise ValueError("分片前模型case顺序/身份与配对计划不一致")
            pool[split] = rows
        if set(pool) != set(manifest['splits']):
            raise ValueError("分片前模型split不完整")
        pools.append(pool)
    counts = allocate_shards(len(gpu_plan['selected_ids']), [sum(map(len, p.values())) for p in pools])
    tasks = []
    for model, job, pool, count in zip(manifest['models'], jobs, pools, counts):
        shards = [{split: [] for split in pool} for _ in range(count)]
        offset = 0
        for split, rows in pool.items():
            for index, row in enumerate(rows):
                shards[(offset + index) % count][split].append(row)
            offset += len(rows)
        model['evaluation_shards'] = []
        for index, shard in enumerate(shards):
            tid = model['id'] if count == 1 else f"{model['id']}_shard_{index+1:02d}"
            output = Path(job['output']) if count == 1 else Path(job['output']) / 'shards' / f'shard_{index+1:02d}'
            worker = dict(job, output=str(output), rows={}, row_hashes={})
            for split, rows in shard.items():
                path = out / '_plan' / f'{tid}_{split}.json'
                write_json(path, rows)
                worker['rows'][split] = str(path)
                worker['row_hashes'][split] = file_hash(path)
            path = out / '_plan' / f'worker_{tid}.json'
            write_json(path, worker)
            descriptor = dict(id=tid, output=str(output.relative_to(out)),
                              case_ids={split: [case_id(r) for r in rows] for split, rows in shard.items()})
            model['evaluation_shards'].append(descriptor)
            tasks.append(dict(id=tid, model=model['id'], shard_index=index, shard_count=count,
                              samples=sum(map(len, shard.values())), job=str(path)))
    # 首批优先覆盖每个模型，然后启动其额外分片。
    tasks.sort(key=lambda task: (task['shard_index'], task['model']))
    gpu_plan.update(tasks=tasks, shards_per_model={m['id']: n for m, n in zip(manifest['models'], counts)},
                    parallel_workers=min(len(tasks), len(gpu_plan['selected_ids'])),
                    unused_gpu_count=max(0, len(gpu_plan['selected_ids'])-len(tasks)))
    print(f"[comparison] workers={len(tasks)} parallel={gpu_plan['parallel_workers']}; "
          f"shards/model={gpu_plan['shards_per_model']}; unused GPUs={gpu_plan['unused_gpu_count']}", flush=True)
    return tasks
