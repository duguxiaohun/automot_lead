"""固定候选预算内自适应搜索：每类命中目标即停止派发，绝不重复评估case。"""
from collections import defaultdict
import copy
from pathlib import Path

from qwen3vl_local.action_prior.comparison_cases import read_json, write_json, case_id
from qwen3vl_local.action_prior.contracts import file_hash
from qwen3vl_local.action_prior.comparison_errors import error_selection
from qwen3vl_local.action_prior.comparison_shards import plan_shards
from qwen3vl_local.action_prior.comparison_render import paired_cases, linked_copy
from qwen3vl_local.action_prior.comparison_pool import SearchPool


class CategorySearch:
    def __init__(self, groups, target, disabled=()):
        if target < 1:
            raise ValueError('search target must be positive')
        self.target = target
        self.groups = {(style, name): list(ids) for style, cats in groups.items() for name, ids in cats.items()}
        self.targets = {key: 0 if key in disabled else target for key in self.groups}
        self.kept = {key: [] for key in self.groups}
        self.seen, self.matches = set(), set()
        self.membership = defaultdict(list)
        self.cursor = {key: 0 for key in self.groups}
        for key, ids in self.groups.items():
            if len(ids) != len(set(ids)):
                raise ValueError('duplicate candidate in category')
            for cid in ids:
                self.membership[cid].append(key)

    def next_batch(self, size):
        batch, planned = [], set(self.seen)
        while len(batch) < size:
            progress = False
            for key, ids in self.groups.items():
                if len(self.kept[key]) >= self.targets[key]:
                    continue
                while self.cursor[key] < len(ids) and ids[self.cursor[key]] in planned:
                    self.cursor[key] += 1
                if self.cursor[key] == len(ids):
                    continue
                cid = ids[self.cursor[key]]
                self.cursor[key] += 1
                planned.add(cid)
                batch.append(cid)
                progress = True
                if len(batch) == size:
                    break
            if not progress:
                break
        return batch

    def accept(self, cid, matched):
        if cid in self.seen or cid not in self.membership:
            raise ValueError('duplicate/unplanned search result')
        self.seen.add(cid)
        if matched:
            self.matches.add(cid)
            for key in self.membership[cid]:
                if len(self.kept[key]) < self.targets[key]:
                    self.kept[key].append(cid)

    def report(self):
        result = {}
        for key, ids in self.groups.items():
            checked = sum(cid in self.seen for cid in ids)
            retained = len(self.kept[key])
            target = self.targets[key]
            result['/'.join(key)] = dict(candidates=len(ids), evaluated=checked,
                matched=sum(cid in self.matches for cid in ids), retained=retained, target=target,
                shortfall=max(0,target-retained), reason='disabled' if target == 0 else 'target_reached' if retained >= target else
                'candidates_exhausted' if checked == len(ids) else 'searching')
        return result


def search_cases(jobs, manifest, gpu_plan, out, target, *, pool_factory=SearchPool):
    out = Path(out)
    original = copy.deepcopy(manifest['splits'])
    write_json(out/'candidate_plan.json', manifest)
    pools = []
    for job in jobs:
        pool = {}
        for split, path in job['rows'].items():
            if file_hash(path) != job['row_hashes'][split]:
                raise ValueError('candidate rows changed after preflight')
            rows = read_json(path)
            if [case_id(r) for r in rows] != list(original[split]['cases']):
                raise ValueError('candidate identities/order differ from manifest')
            pool[split] = {case_id(r): r for r in rows}
        pools.append(pool)
    batch_size = max(8, len(gpu_plan['selected_ids'])*2)
    state = dict(status='searching', target_per_category=target, batch_size=batch_size, splits={})
    manifest['search'] = state
    for model in manifest['models']:
        model.pop('evaluation_shards', None)
    batch_number = 0
    with pool_factory(out, gpu_plan) as workers:
        for split, plan in original.items():
            disabled = {(style, name) for style, cats in plan['coverage'].items() for name, cov in cats.items() if cov['requested'] == 0}
            search = CategorySearch(plan['groups'], target, disabled)
            evaluated = {}
            batch = search.next_batch(batch_size)
            state['splits'][split] = search.report()
            write_json(out/'search.json', state)
            while batch:
                batch_number += 1
                root = out/'_search'/f'batch_{batch_number:05d}'
                bm = copy.deepcopy(manifest)
                bm.pop('search', None)
                bm['splits'] = {split: dict(cases={cid: plan['cases'][cid] for cid in batch})}
                batch_jobs = []
                for model, job, pool in zip(bm['models'], jobs, pools):
                    path = root/'_plan'/f"{model['id']}_{split}.json"
                    write_json(path, [pool[split][cid] for cid in batch])
                    batch_jobs.append(dict(job, output=str(root/'_models'/model['id']), rows={split:str(path)},
                                           row_hashes={split:file_hash(path)}))
                execution = copy.deepcopy(gpu_plan)
                tasks = plan_shards(batch_jobs, bm, execution, root)
                for task in tasks:
                    task['id'] = f"batch_{batch_number:05d}_{task['id']}"
                write_json(root/'manifest.json', bm)
                write_json(out/'status.json', dict(status='searching', split=split, batch=batch_number))
                workers.run(tasks)
                results = paired_cases(root, bm, split)
                with (out/'search_audit.jsonl').open('a', encoding='utf-8') as audit:
                    import json
                    for cid in batch:
                        predictions = {m['id']: results[m['id']][cid] for m in manifest['models']}
                        decision = error_selection(predictions, **manifest['error_filter'])
                        search.accept(cid, decision['kept'])
                        evaluated[cid] = plan['cases'][cid]
                        audit.write(json.dumps(dict(id=cid, split=split, **decision), ensure_ascii=False)+'\n')
                        # Publish a canonical per-model view; raw batch audits remain available.
                        for mid, data in predictions.items():
                            source = Path(data['_comparison_input_dir'])
                            target_dir = out/'_models'/mid/split/'inputs'/cid
                            if source.is_dir():
                                import shutil
                                shutil.copytree(source, target_dir, copy_function=lambda a,b: linked_copy(Path(a),Path(b)))
                            clean = {k:v for k,v in data.items() if k != '_comparison_input_dir'}
                            write_json(out/'_models'/mid/split/'cases'/f'rank0_case_{cid}.json', clean)
                state['splits'][split] = search.report()
                state['batch'] = batch_number
                write_json(out/'search.json', state)
                print(f"[search] {split} checked={len(search.seen)}; "
                      f"categories complete={sum(v['reason'] != 'searching' for v in search.report().values())}/{len(search.groups)}", flush=True)
                batch = search.next_batch(batch_size)
            final = copy.deepcopy(plan)
            final['cases'] = evaluated
            final['selected_unique_frames'] = len(evaluated)
            final['evaluated_groups'] = {}
            for (style, name), ids in search.groups.items():
                final['groups'][style][name] = search.kept[(style,name)]
                final['evaluated_groups'].setdefault(style,{})[name] = [cid for cid in ids if cid in search.seen]
                coverage = final['coverage'][style][name]
                coverage.update(candidate_limit=coverage['requested'], planned_candidates=coverage['selected'],
                    requested=search.targets[(style,name)], selected=len(search.kept[(style,name)]),
                    selected_physical_routes=len({plan['cases'][cid]['route_group'] for cid in search.kept[(style,name)]}),
                    **search.report()['/'.join((style,name))])
            manifest['splits'][split] = final
    state['status'] = 'complete'
    write_json(out/'search.json', state)
    manifest['execution'] = dict(gpu_plan, policy='persistent_workers_batched_search', batch_size=batch_size)
    write_json(out/'manifest.json', manifest)
