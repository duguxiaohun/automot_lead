"""Independent P1 historical route exclusion; original tests are never enlarged."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from qwen3vl_local.sft_new_loop_phase1.training_policy import identity, physical_group
from qwen3vl_local.sft_new_loop_phase12 import history

POLICY = 'independent_phase1_historical_holdout_v1'
FILES = ('historical_contract.json', 'historical_exclusions.jsonl', 'historical_isolation.json')


def groups(contract):
    sources = [s for s in contract['sources'] if s['phase'] == 1]
    if not sources:
        raise ValueError('Missing Phase1 historical tests')
    return {physical_group(tuple(r['key'])) for s in sources for r in s['records']}


def derive(index, contract, output):
    index, contract, output = (Path(p).resolve() for p in (index, contract, output))
    historical = history.load(contract)
    forbidden = groups(historical)
    if output.exists() or output.is_relative_to(index.parent) or contract.is_relative_to(output):
        raise ValueError('Choose a fresh separate historical isolation output')
    before = proof.sha(index)
    manifest_path = index.parent / 'manifest.json'
    manifest_sha = proof.sha(manifest_path)
    manifest = proof.read(manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    counts, removed = Counter(), Counter()
    test_hash = hashlib.sha256()
    with index.open('rb') as source, (output/'frame_index.jsonl').open('xb') as target, \
            (output/'historical_exclusions.jsonl').open('xb') as excluded:
        for raw in source:
            row = json.loads(raw)
            split = row['split']
            if split not in ('train', 'val', 'test'):
                raise ValueError('Invalid source split')
            if split != 'test' and physical_group(identity(row)) in forbidden:
                excluded.write(raw)
                removed[split] += 1
            else:
                target.write(raw)
                counts[split] += 1
                if split == 'test':
                    test_hash.update(raw)
    if any(not counts[s] for s in ('train', 'val', 'test')):
        raise ValueError('Historical isolation emptied a required split')
    if proof.sha(index) != before or proof.sha(manifest_path) != manifest_sha:
        raise ValueError('Source changed during historical isolation')
    shutil.copyfile(contract, output/'historical_contract.json')
    history.load(output/'historical_contract.json')
    receipt = dict(policy=POLICY, phase=1, parent_index_sha256=before,
        parent_manifest_sha256=manifest_sha, historical_contract_sha256=proof.sha(output/'historical_contract.json'),
        historical_groups=len(forbidden), excluded_counts=dict(removed), retained_counts=dict(counts),
        index_sha256=proof.sha(output/'frame_index.jsonl'),
        exclusions_sha256=proof.sha(output/'historical_exclusions.jsonl'),
        test_sha256=test_hash.hexdigest(), test_bytes_unchanged=True,
        retained_rows_byte_identical=True, sampler_changed=False,
        scope='Registered Phase1 historical tests only; removed train/val rows are NOT added to test')
    proof.write(output/'historical_isolation.json', receipt)
    # Statistics from before exclusion must never masquerade as current counts.
    statistics = {k: manifest.pop(k) for k in ('counts', 'sampling', 'route_counts',
        'answer_counts', 'visual_label_risk_counts', 'phase_group_focus_counts',
        'focus_bin_availability', 'phase_group_focus_availability',
        'cross_phase_highway_r3_consistency_counts') if k in manifest}
    manifest.update(frame_index=str(output/'frame_index.jsonl'),
        counts={f'frames/{s}': n for s, n in counts.items()},
        pre_isolation_statistics=statistics, historical_isolation=receipt)
    for name in ('training_revision.json', 'preservation.json'):
        record = dict(historical_isolation=receipt, label_derivation=proof.read(index.parent/name))
        proof.write(output/name, record)
        if name == 'training_revision.json':
            manifest['fresh_training_revision'] = record
    proof.write(output/'manifest.json', manifest)
    verify(output/'frame_index.jsonl')
    return receipt


def verify(index):
    index = Path(index)
    receipt = proof.read(index.parent/'historical_isolation.json')
    forbidden = groups(history.load(index.parent/'historical_contract.json'))
    if (receipt['policy'] != POLICY or receipt['phase'] != 1
            or receipt['historical_contract_sha256'] != proof.sha(index.parent/'historical_contract.json')
            or receipt['exclusions_sha256'] != proof.sha(index.parent/'historical_exclusions.jsonl')
            or receipt['index_sha256'] != proof.sha(index)):
        raise ValueError('Historical isolation evidence changed')
    counts = Counter()
    test_hash = hashlib.sha256()
    for raw in index.open('rb'):
        row = json.loads(raw)
        if row['split'] != 'test' and physical_group(identity(row)) in forbidden:
            raise ValueError('Historical Phase1 route leaked into '+row['split'])
        counts[row['split']] += 1
        if row['split'] == 'test':
            test_hash.update(raw)
    if dict(counts) != receipt['retained_counts'] or test_hash.hexdigest() != receipt['test_sha256']:
        raise ValueError('Historical isolation counts/test binding changed')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True)
    p.add_argument('--history-contract', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(derive(a.parent, a.history_contract, a.output), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
