#!/usr/bin/env python3
"""全流程内部的数据准备入口：自动生成候选与 full map，stdout 只返回最终索引路径。"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.action_prior.contracts import digest, file_hash
from qwen3vl_local.action_prior.event_balance import EventBalanceIndex
from qwen3vl_local.action_prior.build_event_balance_index import _candidate_membership
from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapping_contract_hash

HERE = Path(__file__).resolve().parent


def run_builder(script, arguments):
    """继承离线构建日志到 stderr；禁止混入 stdout 的路径结果。"""
    subprocess.run([sys.executable, str(script), *map(str, arguments)], check=True, stdout=sys.stderr)


def prepare(data_root, action_data_dir, collection_dir, cache_root):
    """按来源内容缓存，持有锁后检查完整性，只原子发布验证成功的新目录。"""
    data_root, action_data_dir = Path(data_root).resolve(), Path(action_data_dir).resolve()
    collection_dir, cache_root = Path(collection_dir).resolve(), Path(cache_root).resolve()
    source_files = sorted(collection_dir.glob('*_result.json'))
    if not source_files:
        raise FileNotFoundError(f'no collection annotations in {collection_dir}')
    action_hashes = {split: file_hash(action_data_dir / f'{split}.jsonl') for split in ('train', 'val', 'test')}
    mapping_hash = mapping_contract_hash()
    candidate_identity = dict(
        preparation_schema='action_prior_auto_prepare_v1', data_root=str(data_root),
        mapping_contract_hash=mapping_hash,
        collection={p.name: file_hash(p) for p in source_files},
    )
    candidate_dir = cache_root / ('phase3_' + digest(candidate_identity))
    candidate_path = candidate_dir / 'candidate_frames.jsonl'
    cache_root.mkdir(parents=True, exist_ok=True)
    # 缓存是共享构建产物，不覆盖人工索引。进程退出自动释放 flock；tmp 失败自动清理。
    with (cache_root / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if candidate_dir.exists():
            _candidate_membership(candidate_path, mapping_hash)
            print(f'[prepare] reuse candidates: {candidate_dir}', file=sys.stderr, flush=True)
        else:
            print('[prepare] build current Phase3 candidates (data only, no Phase3 training)', file=sys.stderr, flush=True)
            with tempfile.TemporaryDirectory(prefix='.candidate-', dir=cache_root) as temporary:
                staging = Path(temporary) / 'index'
                run_builder(HERE.parent / 'sft_new_loop_phase3/build_dataset.py', [
                    '--data-root', data_root, '--collection-dir', collection_dir, '--output-dir', staging,
                ])
                # manifest 中的可见路径指向最终发布目录，避免保存即将清理的临时路径。
                manifest_path = staging / 'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                manifest['frame_index'] = str(candidate_dir / 'frame_index.jsonl')
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
                _candidate_membership(staging / 'candidate_frames.jsonl', mapping_hash)
                staging.rename(candidate_dir)
        full_identity = dict(
            candidate=candidate_identity, candidate_sha256=file_hash(candidate_path),
            action_dataset_hashes=action_hashes,
            builder_sha256=file_hash(HERE / 'build_event_balance_index.py'),
            loader_sha256=file_hash(HERE / 'event_balance.py'),
        )
        full_dir = cache_root / ('full_' + digest(full_identity))
        full_path = full_dir / 'full_event_mapping.jsonl'
        if full_dir.exists():
            EventBalanceIndex(full_path).validate_action_dataset(action_data_dir)
            print(f'[prepare] reuse full map: {full_path}', file=sys.stderr, flush=True)
        else:
            print('[prepare] build full event mapping', file=sys.stderr, flush=True)
            with tempfile.TemporaryDirectory(prefix='.full-', dir=cache_root) as temporary:
                staging = Path(temporary) / 'index'
                run_builder(HERE / 'build_event_balance_index.py', [
                    '--collection-dir', collection_dir, '--candidate-index', candidate_path,
                    '--action-data-dir', action_data_dir, '--output-dir', staging,
                ])
                EventBalanceIndex(staging / 'full_event_mapping.jsonl').validate_action_dataset(action_data_dir)
                staging.rename(full_dir)
        return full_path


def main():
    """由 run_full_pipeline.sh 调用，原始 LEAD 数据和人工标注仍须在训练机上存在。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='lead_data')
    parser.add_argument('--action-data-dir', required=True)
    parser.add_argument('--collection-dir', default='keyframe_filter/collection_output')
    parser.add_argument('--cache-root', default='checkpoints/action_prior_prepared')
    args = parser.parse_args()
    print(prepare(args.data_root, args.action_data_dir, args.collection_dir, args.cache_root))


if __name__ == '__main__':
    main()
