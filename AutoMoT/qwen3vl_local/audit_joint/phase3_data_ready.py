"""Native Phase3 single-frame labels: exact reuse, then route-level generation.

The legacy builder/rules stay frozen. A prompt/runtime change can reassemble the
same labels into an explicitly documented current-source artifact, not relabel.
"""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from . import versioned_data as cache

VERSION = 'native_halfsecond_v1'
RUNTIME_ONLY = {'runtime.py', 'objective.py', 'worker.py', 'protocol.py', 'refresh_runtime.py'}


def main():
    import argparse
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--index',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--search-root',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    result=ensure(a.index,a.data_root,a.cache_root,candidates=cache.discover(a.search_root,'dataset.json',depth=3))
    cache.write(a.receipt,dict(result,dataset_sha256=cache.bind(result['dataset'])['sha256']))
    print(json.dumps(result,ensure_ascii=False))


def label_sources(bound):
    return dict(base_release_sha256=bound['base_release_sha256'],
                experiment_sources={k:v for k,v in bound['experiment_sources'].items() if k not in RUNTIME_ONLY})


def ensure(index, data_root, cache_root, *, candidates=(), history_fill='nearest_short', lateral_weight=1.5):
    from . import phase3_base_trial as trial, label_quarantine
    from .phase3_single_frame import dataset
    from .phase3_base_dense import provenance
    from .phase3_base_dense.data import Pool
    from qwen3vl_local.sft_new_loop_phase3 import train
    index = Path(index).resolve(strict=True); data = Path(data_root).resolve(strict=True)
    cfg = json.loads(json.dumps(dataset.config('offline_native', lateral_weight, history_fill)))
    bound = dataset.sources(); index_binding = cache.bind(index)
    valid = {}
    def inspect(path):
        value = trial.read(path)
        if not isinstance(value, dict) or 'raw_proof' not in value or 'sources' not in value:
            return None
        if value.get('data_root') != str(data) or value.get('index_sha256') != index_binding['sha256'] or value.get('config') != cfg:
            return None
        if label_sources(value['sources']) != label_sources(bound):
            return None
        report = trial.read(path.with_name('dataset_summary.json'))
        if (trial.file_sha(path) != report['artifact_sha256'] or value['sources'] != report['sources'] or value['config'] != report['config']):
            raise ValueError('Existing single-frame artifact/receipt changed: ' + str(path))
        try:
            provenance.verify_sources(data, value['raw_proof'])
        except ValueError:
            # Intact old artifact + changed raw sources is an older INPUT
            # version, not corrupted output. New route contracts decide reuse.
            return None
        valid[path] = value
        # Runtime metadata does not make otherwise identical labels ambiguous.
        return dict(records=cache.digest(value['records']), proof=value['raw_proof'])
    selected = cache.unique_compatible(candidates, inspect)
    if selected is not None:
        previous = valid[selected]
        if previous['sources'] == bound:
            dataset.load(selected, index=index)
            return dict(action='reuse_existing', dataset=str(selected), labels_recomputed=False,
                        version=VERSION, training_performed=False)
        def contract():
            return dict(parent=cache.bind(selected), index=cache.bind(index), sources=dataset.sources(), config=cfg)
        def build(output):
            output.mkdir()
            note=dict(parent_artifact=str(selected),parent_sha256=trial.file_sha(selected),
                      parent_sources=previous['sources'],changed_runtime_sources=sorted(
                          k for k in RUNTIME_ONLY if previous['sources']['experiment_sources'].get(k)!=bound['experiment_sources'].get(k)),
                      records_sha256=cache.digest(previous['records']),labels_recomputed=False)
            value=dict(previous,sources=bound,runtime_refresh=note)
            trial.write(output/'dataset.json',value)
            report=trial.read(selected.with_name('dataset_summary.json'))
            trial.write(output/'dataset_summary.json',dict(report,sources=bound,runtime_refresh=note,
                        artifact_sha256=trial.file_sha(output/'dataset.json')))
        result=cache.ensure(cache_root,'phase3_runtime',VERSION,contract,build,
                            lambda p:dataset.load(p/'dataset.json',index=index),protected=[data,index.parent,selected.parent])
        result.update(dataset=str(Path(result['directory'])/'dataset.json'),labels_recomputed=False)
        return result

    rows=[r for split in ('train','val','test') for r in train._read_rows(index,split,data_root=data)]
    if cache.bind(index)!=index_binding:raise ValueError('Index changed while reading')
    pool=Pool(rows)  # Global physical train/val/test separation before sharding.
    groups=defaultdict(list)
    physical=set()
    for row in pool.anchor_rows:
        if Pool.action(row) is not None:groups[row.scenario,row.route_id].append(row)
    if not groups:raise ValueError('No single-frame anchors')
    shards=[]
    for (scenario,route), anchors in sorted(groups.items()):
        if any(Path(n).name!=n or n in ('.','..') for n in (scenario,route)):
            raise ValueError('Invalid route identity')
        run=data/scenario/route
        if run.resolve() in physical:raise ValueError('Aliased physical route')
        physical.add(run.resolve())
        def contract():
            risks=label_quarantine.load(data,route=(scenario,route))
            files=[*sorted((run/'metas').glob('*.pkl')),
                   *(Path(r.history_rgb_paths[-1]) for r in anchors),
                   *(data/s['path'] for risk in risks for s in risk['sources']),label_quarantine.LEDGER]
            return dict(label_sources=label_sources(dataset.sources()),config=cfg,rows=[asdict(r) for r in anchors],
                        raw=cache.bindings(files),raw_stamps={str(p):provenance.stamp(p) for p in files},
                        route=str(run.resolve(strict=True)),
                        builder_sha256=cache.bind(__file__)['sha256'])
        def build(output):
            dataset.build(anchors,data,output,history_policy='offline_native',history_fill=history_fill,
                          lateral_weight=lateral_weight,index=None)
        def validate(output):
            path=output/'dataset.json';value=trial.read(path);report=trial.read(output/'dataset_summary.json')
            if (trial.file_sha(path)!=report['artifact_sha256'] or value['sources']!=report['sources'] or
                label_sources(value['sources'])!=label_sources(dataset.sources()) or value['config']!=cfg or
                value['config']!=report['config'] or value['data_root']!=str(data)):
                raise ValueError('Cached route label contract changed')
            provenance.verify_sources(data,value['raw_proof'])
        result=cache.ensure(cache_root,'phase3_route',VERSION,contract,build,validate,protected=[data,index.parent])
        shards.append(result)
        print(f'[data] {scenario}/{route}: {result["action"]}',flush=True)
    def contract():
        return dict(sources=dataset.sources(),index=cache.bind(index),config=cfg,
                    shards=[cache.bind(Path(s['directory'])/'dataset.json') for s in shards])
    def assemble(output):
        records={};proof=dict(sources={},meta_members={},directory_targets={})
        for shard in shards:
            value=trial.read(Path(shard['directory'])/'dataset.json')
            if records.keys() & value['records'].keys():raise ValueError('Duplicate assembled anchor')
            records.update(value['records'])
            for key,items in proof.items():
                for name,item in value['raw_proof'][key].items():
                    if name in items and items[name]!=item:raise ValueError('Conflicting raw source proof')
                    items[name]=item
        provenance.verify_sources(data,proof)
        output.mkdir()
        value=dict(config=cfg,sources=dataset.sources(),data_root=str(data),index_sha256=index_binding['sha256'],
                   records=records,raw_proof=proof)
        trial.write(output/'dataset.json',value)
        trial.write(output/'dataset_summary.json',dict(status='assembled_not_training_admission',
            config=cfg,sources=value['sources'],artifact_sha256=trial.file_sha(output/'dataset.json'),
            summary=dataset.summary(records),routes=len(shards),anchors=len(records),
            raw_source_files=len(proof['sources']),native_index_rewritten=False,RGB_copied=False,
            route_cache=dict(built=sum(s['action']=='build' for s in shards),reused=sum(s['action']=='reuse' for s in shards))))
    result=cache.ensure(cache_root,'phase3_single_frame',VERSION,contract,assemble,
                        lambda p:dataset.load(p/'dataset.json',index=index),protected=[data,index.parent])
    result.update(dataset=str(Path(result['directory'])/'dataset.json'),
                  routes_built=sum(s['action']=='build' for s in shards),routes_reused=sum(s['action']=='reuse' for s in shards))
    return result


if __name__=='__main__':main()
