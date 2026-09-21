#!/usr/bin/env python3
"""全流程内部的数据准备入口：自动生成候选与 full map，stdout 只返回最终索引路径。"""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.action_prior.contracts import digest, file_hash
from qwen3vl_local.action_prior.event_balance import EventBalanceIndex
from qwen3vl_local.action_prior.build_event_balance_index import _candidate_membership
from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapping_contract_hash

HERE = Path(__file__).resolve().parent


def run_builder(script, arguments):
    """继承离线构建日志到 stderr；禁止混入 stdout 的路径结果。"""
    subprocess.run([sys.executable, str(script), *map(str, arguments)], check=True, stdout=sys.stderr)


def _reuse_or_quarantine(directory, validate):
    """只修复自动缓存：残缺产物移到旁边保留证据，不删除或覆盖原内容。"""
    if not directory.exists() and not directory.is_symlink():
        return False
    try:
        validate(directory)
    except (ValueError, KeyError, TypeError, AttributeError,
            FileNotFoundError, NotADirectoryError, IsADirectoryError) as exc:
        # JSON 语法正确也可能是 null/list 或缺字段；此处只隔离旧缓存。
        # 新构建产物的 validate 在 _publish 外层执行，程序错误不会被吞掉重试。
        quarantine = directory.with_name(f'.invalid-{directory.name}-{uuid.uuid4().hex}')
        directory.rename(quarantine)
        print(f'[prepare] invalid cache preserved: {quarantine}; reason: {exc}',
              file=sys.stderr, flush=True)
        return False
    print(f'[prepare] reuse cache: {directory}', file=sys.stderr, flush=True)
    return True


def _publish(staging, destination, validate, payloads):
    """校验后原子发布；兼容 EEXIST/ENOTEMPTY，不能把任意 rename 错误当缓存命中。"""
    validate(staging)
    for _ in range(3):
        try:
            staging.rename(destination)
            print(f'[prepare] published: {destination}', file=sys.stderr, flush=True)
            return
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
        # flock 只约束遵守同一锁的进程；构建期间仍可能有外部发布者写入目标。
        # 只有真实内容一致才复用，避免同名目录掩盖来源变化或非确定构建。
        if _reuse_or_quarantine(destination, validate):
            if any(file_hash(staging / name) != file_hash(destination / name) for name in payloads):
                raise ValueError(f'{destination}: publication conflict: valid cache content differs; '
                                 'check concurrent builders and source changes')
            print(f'[prepare] reuse concurrent publication: {destination}', file=sys.stderr, flush=True)
            return
    raise RuntimeError(f'{destination}: repeated publication conflicts; '
                       'check external writers and shared filesystem locking')


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
        preparation_schema='action_prior_auto_prepare_v2_independent_splits', data_root=str(data_root),
        candidate_split_policy='all_train_pool_action_uses_own_physical_splits',
        mapping_contract_hash=mapping_hash,
        collection={p.name: file_hash(p) for p in source_files},
    )
    candidate_dir = cache_root / ('phase3_' + digest(candidate_identity))
    candidate_path = candidate_dir / 'candidate_frames.jsonl'
    cache_root.mkdir(parents=True, exist_ok=True)
    # 缓存是共享构建产物，不覆盖人工索引。进程退出自动释放 flock；tmp 失败自动清理。
    with (cache_root / '.prepare.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'[prepare] waiting for cache lock: {cache_root / ".prepare.lock"}',
                  file=sys.stderr, flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)

        def validate_candidates(directory):
            """临时目录、历史缓存与并发发布结果使用同一候选完整性检查。"""
            _candidate_membership(directory / 'candidate_frames.jsonl', mapping_hash)

        if not _reuse_or_quarantine(candidate_dir, validate_candidates):
            print('[prepare] build current Phase3 candidates (data only, no Phase3 training)', file=sys.stderr, flush=True)
            with tempfile.TemporaryDirectory(prefix='.candidate-', dir=cache_root) as temporary:
                staging = Path(temporary) / 'index'
                run_builder(HERE.parent / 'sft_new_loop_phase3/build_dataset.py', [
                    '--data-root', data_root, '--collection-dir', collection_dir, '--output-dir', staging,
                    '--val-ratio', '0', '--test-ratio', '0',
                ])
                # manifest 中的可见路径指向最终发布目录，避免保存即将清理的临时路径。
                manifest_path = staging / 'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                manifest['frame_index'] = str(candidate_dir / 'frame_index.jsonl')
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
                _publish(staging, candidate_dir, validate_candidates,
                         ('candidate_frames.jsonl', 'candidate_counts.json', 'frame_index.jsonl'))
        full_identity = dict(
            candidate=candidate_identity, candidate_sha256=file_hash(candidate_path),
            action_dataset_hashes=action_hashes,
            builder_sha256=file_hash(HERE / 'build_event_balance_index.py'),
            loader_sha256=file_hash(HERE / 'event_balance.py'),
        )
        full_dir = cache_root / ('full_' + digest(full_identity))
        full_path = full_dir / 'full_event_mapping.jsonl'

        def validate_full(directory):
            """full map 同时绑定当前 action split 和实际候选字节。"""
            index = EventBalanceIndex(directory / 'full_event_mapping.jsonl')
            index.validate_action_dataset(action_data_dir)
            if index.source.candidate_sha256 != full_identity['candidate_sha256']:
                raise ValueError('full map candidate hash mismatch')

        if not _reuse_or_quarantine(full_dir, validate_full):
            print('[prepare] build full event mapping', file=sys.stderr, flush=True)
            with tempfile.TemporaryDirectory(prefix='.full-', dir=cache_root) as temporary:
                staging = Path(temporary) / 'index'
                run_builder(HERE / 'build_event_balance_index.py', [
                    '--collection-dir', collection_dir, '--candidate-index', candidate_path,
                    '--action-data-dir', action_data_dir, '--output-dir', staging,
                ])
                _publish(staging, full_dir, validate_full, ('full_event_mapping.jsonl',))
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
