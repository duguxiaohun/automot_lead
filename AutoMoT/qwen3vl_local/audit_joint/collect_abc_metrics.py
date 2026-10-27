"""CPU-only ABC metrics collection, with content deduplication and solid LZMA ZIP.

Original evidence bytes are stored once in evidence.bin. MANIFEST.json maps each
source to offset/size/SHA. Nested audit ZIP members are retained individually;
ZIP container bytes are identified by SHA but not redundantly embedded.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_LZMA
from datetime import datetime
import argparse
import hashlib
import json
import os
import tempfile
import time

LIMIT = 30_000_000
CHUNK = 1024 * 1024


def signature(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def collect(run, out, limit=LIMIT, variants="abc"):
    variants = tuple(variants)
    if not variants or len(set(variants)) != len(variants) or any(v not in "abc" for v in variants):
        raise ValueError("variants must be distinct members of a/b/c")
    if not 0 < limit <= LIMIT:
        raise ValueError('limit must be between 1 and 30,000,000 bytes')
    run = Path(run).resolve()
    for v in variants:
        if not (run / (v + '_bev_only') / 'config.json').is_file():
            raise ValueError('当前机器缺少 ABC 分组，请修改 ABC_AUDIT_RUN：' + str(run / (v + '_bev_only')))
    out = Path(out).resolve()
    if out == run or out.is_relative_to(run):
        raise ValueError('output must be outside input run')
    out.mkdir(parents=True, exist_ok=False)
    core, tb, archives, missing = [], [], [], []
    for v in variants:
        group = run / (v + '_bev_only')
        core += [p for p in group.glob('*.json') if p.is_file()]
        for name in ('training_audit', 'validation', 'performance', 'progress', 'test', 'eval_test'):
            folder = group / name
            count = 0
            if folder.is_dir():
                for base, dirs, files in os.walk(folder, followlinks=False):
                    dirs[:] = sorted(d for d in dirs if d not in ('cases', 'delivery') and not (Path(base)/d).is_symlink())
                    for filename in sorted(files):
                        p = Path(base) / filename
                        if p.suffix in ('.json', '.jsonl') or filename == 'eval.log':
                            core.append(p)
                            count += 1
            if not count:
                missing.append(v + '_bev_only/' + name + ': no matching loose records; audit archive may supply them')
        fallback = group / 'training_audit.zip'
        if fallback.is_file():
            archives.append(fallback)
        if (group / 'tb').is_dir():
            tb += [p for p in (group / 'tb').rglob('*tfevents*') if p.is_file()]
    core, tb = sorted(set(core)), sorted(set(tb))
    package = out / ((variants[0] + '_metrics_supplement.zip') if len(variants) == 1 else 'abc_metrics_supplement.zip')
    sources, archive_records, unique = {}, {}, {}
    last_heartbeat = [time.monotonic()]

    def heartbeat(message):
        if time.monotonic() - last_heartbeat[0] >= 10:
            print('[CPU] ' + message, flush=True)
            last_heartbeat[0] = time.monotonic()

    with tempfile.TemporaryDirectory(prefix='.snapshot_', dir=out) as temp:
        snapshot = Path(temp) / 'evidence.bin'
        with snapshot.open('w+b') as dst:
            def capture(stream, name):
                if name in sources:
                    raise ValueError('duplicate source identifier: ' + name)
                offset = dst.tell()
                h, size = hashlib.sha256(), 0
                for block in iter(lambda: stream.read(CHUNK), b''):
                    dst.write(block)
                    h.update(block)
                    size += len(block)
                    heartbeat('读取指标快照：' + name)
                digest = h.hexdigest()
                key = (digest, size)
                if key in unique:
                    # Verify bytes before using a deduplication alias.
                    existing = unique[key]
                    for position in range(0, size, CHUNK):
                        length = min(CHUNK, size-position)
                        dst.seek(existing + position)
                        first = dst.read(length)
                        dst.seek(offset + position)
                        if first != dst.read(length):
                            raise ValueError('hash collision or snapshot corruption')
                    dst.seek(offset)
                    dst.truncate()
                else:
                    unique[key] = offset
                sources[name] = dict(offset=unique[key], bytes=size, sha256=digest)

            print('[CPU] 合并完整指标快照；展开审计 ZIP；按内容去重', flush=True)
            for p in core:
                before = signature(p)
                name = str(p.relative_to(run))
                with p.open('rb') as stream:
                    capture(stream, name)
                if signature(p) != before or sources[name]['bytes'] != before[2]:
                    raise RuntimeError('文件正在变化，请运行停止后再收集：' + str(p))
            for p in archives:
                before = signature(p)
                name = str(p.relative_to(run))
                container_sha = sha_file(p)
                with ZipFile(p) as z:
                    infos = [i for i in z.infolist() if not i.is_dir()]
                    if len(infos) != len({i.filename for i in infos}):
                        raise ValueError('duplicate ZIP members: ' + name)
                    captured, omitted = [], []
                    for info in infos:
                        # Evidence archives normally contain JSON/Markdown. Never copy weights.
                        if Path(info.filename).suffix.lower() not in ('.json', '.jsonl', '.md', '.txt', '.log'):
                            omitted.append(dict(member=info.filename, bytes=info.file_size, reason='nonmetric_archive_member'))
                            continue
                        identifier = name + '::' + info.filename
                        with z.open(info) as stream:
                            capture(stream, identifier)
                        if sources[identifier]['bytes'] != info.file_size:
                            raise ValueError('incomplete ZIP member: ' + identifier)
                        captured.append(identifier)
                if signature(p) != before:
                    raise RuntimeError('审计 ZIP 正在变化：' + str(p))
                archive_records[name] = dict(bytes=before[2], sha256=container_sha,
                    storage='members_unpacked; container bytes not embedded', members=captured, omitted=omitted)
            core_size = dst.tell()
            core_sources = dict(sources)
            core_unique = len(unique)
            for p in tb:
                before = signature(p)
                name = str(p.relative_to(run))
                with p.open('rb') as stream:
                    capture(stream, name)
                if signature(p) != before or sources[name]['bytes'] != before[2]:
                    raise RuntimeError('TensorBoard 文件仍在变化：' + str(p))
            full_size = dst.tell()

        summary = dict(schema='abc_metrics_stream_v2', run=str(run), variants=list(variants), collection_only=True,
            weights_included=False, core_source_count=len(core_sources),
            core_original_bytes=sum(r['bytes'] for r in core_sources.values()),
            core_unique_bytes=core_size, core_unique_contents=core_unique,
            tensorboard_files_found=len(tb), missing=missing,
            largest_core_sources=sorted((dict(source=k, **v) for k,v in core_sources.items()), key=lambda r:r['bytes'], reverse=True)[:25])
        (out / 'collection_inventory.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
        print('[CPU] 核心来源 %d，原始 %.2fMB，去重后 %.2fMB；开始跨文件压缩' % (
            len(core_sources), summary['core_original_bytes']/1e6, core_size/1e6), flush=True)

        def pack(include_tb):
            length = full_size if include_tb else core_size
            selected = sources if include_tb else core_sources
            manifest = dict(summary, sources=selected, archives=archive_records,
                tensorboard_included=bool(include_tb and tb),
                omitted_tensorboard=[] if include_tb else [str(p.relative_to(run)) for p in tb],
                storage=dict(member='evidence.bin', bytes=length, format='original source bytes; seek by offset, read bytes, verify sha256'),
                preservation='All selected source bytes unchanged; identical contents share an offset. Nested ZIP member bytes preserved, not original ZIP container bytes.')
            h = hashlib.sha256()
            with ZipFile(package, 'w', ZIP_LZMA, allowZip64=True) as z:
                with snapshot.open('rb') as src, z.open('evidence.bin', 'w', force_zip64=True) as target:
                    remaining = length
                    while remaining:
                        block = src.read(min(CHUNK, remaining))
                        if not block:
                            raise RuntimeError('snapshot truncated')
                        target.write(block)
                        h.update(block)
                        remaining -= len(block)
                        heartbeat('压缩完整快照，剩余 %.1fMB' % (remaining/1e6))
                manifest['storage']['sha256'] = h.hexdigest()
                z.writestr('MANIFEST.json', json.dumps(manifest, ensure_ascii=False, separators=(',', ':')))
                z.writestr('README.txt', 'ABC metrics stream v2. Read MANIFEST.json. For each sources entry, seek evidence.bin to offset, read bytes, verify sha256. Source names containing :: identify original audit ZIP members. No metric rows are sampled or rounded.\n')
            return manifest

        try:
            manifest = pack(bool(tb))
            first_bytes = package.stat().st_size
            if first_bytes > limit and tb:
                print('[CPU] 含 TensorBoard 包 %.2fMB；保留全部核心指标重新压缩' % (first_bytes/1e6), flush=True)
                manifest = pack(False)
            size = package.stat().st_size
            if size > limit:
                raise RuntimeError('完整核心去重压缩后仍为 %.2fMB，超过 %.2fMB；未发布。体积清单：%s' % (size/1e6, limit/1e6, out/'collection_inventory.json'))
        except BaseException:
            package.unlink(missing_ok=True)
            raise
    result = dict(result_directory=str(out), package=str(package), bytes=size,
        sha256=sha_file(package), schema='abc_metrics_stream_v2',
        variants=list(variants), tensorboard_included=manifest['tensorboard_included'], core_source_count=len(core_sources), missing=missing)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def collect_split(run, out, limit=LIMIT, variants="abc"):
    """Independent per-group delivery; a failed group never removes other packages."""
    variants = tuple(variants)
    if not variants or len(set(variants)) != len(variants) or any(v not in "abc" for v in variants):
        raise ValueError("variants must be distinct members of a/b/c")
    if not 0 < limit <= LIMIT:
        raise ValueError('limit must be between 1 and 30,000,000 bytes')
    run, out = Path(run).resolve(), Path(out).resolve()
    if out == run or out.is_relative_to(run):
        raise ValueError('output must be outside input run')
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema='abc_metrics_split_delivery_v1', run=str(run),
        result_directory=str(out), per_package_limit_bytes=limit,
        requested_groups=list(variants), groups={}, errors={}, status='collecting')
    receipt_path = out / 'delivery.json'

    def save():
        pending = receipt_path.with_suffix('.tmp')
        pending.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
        pending.replace(receipt_path)

    save()
    for v in variants:
        print('[CPU] 单独收集 %s；该包上限 %.2fMB' % (v, limit/1e6), flush=True)
        try:
            result = collect(run, out / v, limit=limit, variants=v)
            receipt['groups'][v] = result
        except Exception as exc:
            receipt['errors'][v] = dict(type=type(exc).__name__, message=str(exc),
                diagnostic_directory=str(out / v))
            print('[CPU] %s 未完成：%s；继续其它分组' % (v, exc), flush=True)
        save()
    receipt['status'] = ('completed' if not receipt['errors'] else
                         'partial' if receipt['groups'] else 'failed')
    save()
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description='ABC 指标独立打包：a/b/c 各一包，每包不超过30MB，不使用GPU')
    parser.add_argument('--variant', choices=('all', 'a', 'b', 'c'), default='all',
                        help='默认分别打包三组；可只补收某一组')
    args = parser.parse_args(argv)
    root = Path.cwd()
    if not (root / 'qwen3vl_local').is_dir():
        raise SystemExit('请先进入当前服务器的 AutoMoT 根目录')
    run = Path(os.environ.get('ABC_AUDIT_RUN', 'checkpoints/action_expert_ablation/geometry_abc_bev_only/run_20261026_122636')).expanduser()
    out = root / 'checkpoints/audit_exports' / ('abc_metrics_split_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    result = collect_split(run, out, variants='abc' if args.variant == 'all' else args.variant)
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
