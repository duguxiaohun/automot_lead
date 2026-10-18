"""Delete only explicitly listed obsolete outputs after conservative reference checks.

No latest-directory guessing, backups, training, source changes or wildcard deletes.
Unknown references, scan failures, mounts and live project processes block deletion.
"""
import argparse
from collections import deque
import fcntl
import gzip
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import struct
import time
import zipfile


# User requested removal of old outputs while preserving the current experiments.
# Exact names from the verified cr02 report; never apply by prefix or date alone.
CANDIDATES = '''phase3_epoch_20261013_100331_732771255
phase3_epoch_recovery_20261013_121248_683529276
phase3_index_handoff_20261012_222651_851882451
phase3_labels_20261017_122537_843240561
phase3_prepare_20261012_231356_193369568
phase3_three_epochs_20261013_174517_669756526
phase3_three_epochs_new_20261014_234529_372181097
phase4_existing_20261014_080623_118281694
phase4_full_epoch1_handoff_20261009.audit
phase4_full_epoch1_handoff_20261009.audit.zip
phase4_full_epoch1_rgb2_20261009
phase4_full_epoch1_rgb4_20261009
phase4_labels_20261017_125230_304771113
phase4_recovery_check_20261009.json
phase4_short_observed_20261009
phase4_short_rgb_20261009
phase4_short_smoke
phase4_short_smoke_handoff_20261009.audit.zip
phase4_short_smoke_rgb4
phase4_support_review_20261010
phase4_train_fit_20261010_rgb2
phase4_train_fit_20261010_rgb4
phase4_train_fit_20261010_rgb4_failed_by_tool_error
phase4_train_fit_handoff_20261010.audit.zip
phase4_ue7_fit_20261011_rgb2_complete
phase4_ue7_fit_20261011_rgb2_proceed
phase4_ue7_fit_20261011_rgb4_complete
phase4_ue7_fit_20261011_rgb4_proceed
phase4_ue7_fit_handoff_20261011.audit.zip'''.splitlines()
CANDIDATES += '''p3_batch8_gc_20261015_174830
p3_batch_benchmark_20261015_194054
p3_batch_e2e_20261015_204857
p3_batch_probe_20261015_163611
p3_batch_probe_20261015_165901
p3_batch_probe_gc_20261015_171811
p4_batch_measure_20261015_222211_961641529
p4_batch_measure_20261016_091655_046884216
server_readiness_v50_20261011_212253_659194867
server_readiness_v50_20261012_000640_435016189
server_readiness_v50_20261012_082405_720782054
epoch1_rgb_train_20261009.log
train_fit_20261010.log
train_fit_rgb4_retry_20261010.log
ue7_fit_20261011.log'''.splitlines()
ASSETS = {'.jpg', '.jpeg', '.png', '.webp', '.mp4'}
REQUIRED = [
    'Qwen3.5-4B', 'phase3_four_data_20261017_223254_120427458',
    'phase4_teacher_v24_20261018_121758_779650998',
    *['sft_new_loop_phase3_pipeline/'+name for name in (
        '20261017_223254_126705688_2rgb_short_choice',
        '20261018_034614_519706091_2rgb_short_binary',
        '20261018_095559_213634718_4rgb_choice')],
]
TRACE_OWNERS = ('sft_new_loop_phase3_pipeline', 'label_release_cr02_20261018_114934_308707697')


def stamp(path):
    s = path.lstat()
    # Some filesystems expose coarse directory timestamps: compare membership too.
    members = tuple(sorted(os.listdir(path))) if stat.S_ISDIR(s.st_mode) else None
    return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns, members)


def live_processes(root):
    rows = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in (os.getpid(), os.getppid()):
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            cmd = (p/'cmdline').read_bytes().decode(errors='replace')
            if not any(x in cmd for x in ('python', 'torchrun', 'server_phase', 'train.sh', 'eval.sh', 'sync_automot')):
                continue
            cwd = Path(os.readlink(p/'cwd'))
            if cwd.is_relative_to(root) or str(root) in cmd:
                rows.append({'pid': int(p.name), 'cwd': str(cwd)})
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise ValueError('cannot inspect own process: '+str(p)) from exc
    return rows


def trace_link(path):
    """Record every hop, including intermediates, and terminate exact link cycles.

    Unlike Path.resolve(), a cycle is diagnostic rather than an exception. Missing
    destinations remain references; other I/O errors and excessive depth still fail.
    """
    pending = deque(path.parts[1:])
    current = Path(path.anchor)
    references = [path]
    seen = set()
    while pending:
        part = pending.popleft()
        if part == '.':
            continue
        if part == '..':
            current = current.parent
            continue
        target = current/part
        try:
            is_link = stat.S_ISLNK(target.lstat().st_mode)
        except FileNotFoundError:
            is_link = False
        if not is_link:
            current = target
            continue
        state = (str(target), tuple(pending))
        if state in seen:
            return references, str(target)
        if len(seen) >= 256:
            raise ValueError('symlink traversal limit: '+str(path))
        seen.add(state)
        raw = Path(os.readlink(target))
        destination = raw if raw.is_absolute() else target.parent/raw
        # Keep the unnormalised spelling as well: a/../b still traverses a.
        references.extend((destination, Path(os.path.abspath(destination))))
        current = Path(raw.anchor) if raw.is_absolute() else target.parent
        pending.extendleft(reversed(raw.parts[1:] if raw.is_absolute() else raw.parts))
    references.append(current)
    return references, None


class ReferenceMatcher:
    """Exact substring matching without scanning large payloads once per name.

    Find a shared literal first, then check full names only at possible starts.
    Every pattern contains its group's anchor; overlapping matches are preserved.
    """
    def __init__(self, names):
        dated, other = [], []
        for name in names:
            needle = name.encode()
            (dated if b'202610' in needle else other).append((name, needle))
        self.groups = []
        if dated:
            self.groups.append((b'202610', [(n, b, b.index(b'202610')) for n, b in dated]))
        if other:
            common = os.path.commonprefix([b for _, b in other])
            if len(common) >= 4:
                self.groups.append((common, [(n, b, 0) for n, b in other]))
            else:
                self.groups.extend((b, [(n, b, 0)]) for n, b in other)

    def find(self, data, known=frozenset()):
        found = set()
        for anchor, patterns in self.groups:
            remaining = [(n, b, offset) for n, b, offset in patterns if n not in known]
            if not remaining:
                continue
            position = data.find(anchor)
            while position >= 0:
                for name, needle, offset in remaining:
                    start = position-offset
                    if start >= 0 and data.startswith(needle, start):
                        found.add(name)
                remaining = [row for row in remaining if row[0] not in found]
                if not remaining:
                    break
                position = data.find(anchor, position+1)
        return found


def inventory(cp, candidates, deadline, max_bytes, feedback_files=frozenset(),
              owners=None, reference_trace=None):
    """Scan checkpoint-local references, including compressed text and pickle metadata.

    Bare candidate-name matches are conservative: logs can retain a directory even
    when it is no longer needed. No negative match overrides a scan error.
    """
    matcher = ReferenceMatcher(candidates)
    entries = sorted(cp.iterdir()) if owners is None else [cp/name for name in owners]
    edges = {p.name: set() for p in entries}
    snapshots = {str(cp): stamp(cp)}
    files = {owner: [] for owner in edges}
    sizes = {n: 0 for n in candidates}
    consumed = 0
    last_progress = time.monotonic()
    device = cp.stat().st_dev
    cycles = []

    def scan(stream, owner, source, member=None):
        nonlocal consumed, last_progress
        carry = b''
        seen = set()
        offset = 0
        while True:
            if time.monotonic() > deadline:
                raise ValueError('reference scan time limit reached; nothing deleted')
            chunk = stream.read(1024**2)
            if not chunk:
                break
            consumed += len(chunk)
            if time.monotonic()-last_progress > 20:
                print(f'[scan] {owner}: {consumed / 1024**3:.2f} GiB read', flush=True)
                last_progress = time.monotonic()
            if consumed > max_bytes:
                raise ValueError('reference scan byte budget reached; nothing deleted')
            data = carry+chunk
            found = matcher.find(data, edges[owner] if reference_trace is None else seen)
            edges[owner].update(found)
            if reference_trace is not None:
                for name in sorted(found):
                    if len(reference_trace) >= 2000:
                        raise ValueError('reference trace limit reached; partial evidence only')
                    at = data.find(name.encode())
                    reference_trace.append(dict(owner=owner, candidate=name,
                        file=str(source.relative_to(cp)), archive_member=member,
                        decoded_byte_offset=offset-len(carry)+at,
                        excerpt=data[max(0, at-160):at+len(name)+160].decode(errors='replace')))
                seen.update(found)
            carry = data[-512:]
            offset += len(chunk)

    def visit(path, owner):
        if time.monotonic() > deadline:
            raise ValueError('reference scan time limit reached; nothing deleted')
        if path in feedback_files:
            # Only the three validated files of this invocation are excluded.
            # The parent directory's membership remains part of the snapshot.
            if path.is_symlink() or not path.is_file():
                raise ValueError('feedback file replaced: '+str(path))
            return
        identity = stamp(path)
        snapshots[str(path)] = identity
        if identity[0] != device or (path != cp and os.path.ismount(path)):
            raise ValueError('nested filesystem found; refuse cleanup: '+str(path))
        if path.is_symlink():
            references, cycle = trace_link(path)
            if cycle:
                cycles.append({'path': str(path), 'cycle_at': cycle})
            for link in references:
                matches = {n for n in candidates if n in str(link) or (cp/n).is_relative_to(link)}
                edges[owner].update(matches)
                if reference_trace is not None:
                    for name in sorted(matches):
                        if len(reference_trace) >= 2000:
                            raise ValueError('reference trace limit reached; partial evidence only')
                        reference_trace.append(dict(owner=owner, candidate=name,
                            file=str(path.relative_to(cp)), symlink_target=str(link)))
            return
        if path.is_dir():
            for child in sorted(path.iterdir()):
                visit(child, owner)
            return
        if not path.is_file():
            raise ValueError('special file found; refuse cleanup: '+str(path))
        if owner in sizes:
            sizes[owner] += path.stat().st_blocks*512
        files[owner].append(path)

    def read_references(path, owner):
        identity = snapshots[str(path)]
        if path.suffix.lower() in ASSETS or 'events.out.tfevents' in path.name:
            return
        if path.suffix == '.safetensors':
            # Safetensors may have textual provenance in its header, never in tensor data.
            import io
            with path.open('rb') as stream:
                header_size = struct.unpack('<Q', stream.read(8))[0]
                if header_size > 100_000_000 or header_size > path.stat().st_size-8:
                    raise ValueError('invalid safetensors header: '+str(path))
                header = stream.read(header_size)
                json.loads(header)
                scan(io.BytesIO(header), owner, path)
        elif path.suffix in ('.zip', '.pt', '.pth', '.bin', '.ckpt', '.npz'):
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as archive:
                    torch_archive = any(Path(i.filename).name == 'data.pkl' for i in archive.infolist())
                    for member in archive.infolist():
                        entry = Path(member.filename)
                        name_matches = {n for n in candidates if n in member.filename}
                        edges[owner].update(name_matches)
                        if reference_trace is not None:
                            for name in sorted(name_matches):
                                if len(reference_trace) >= 2000:
                                    raise ValueError('reference trace limit reached; partial evidence only')
                                reference_trace.append(dict(owner=owner, candidate=name,
                                    file=str(path.relative_to(cp)), archive_member=member.filename,
                                    evidence_kind='archive_member_name'))
                        if member.is_dir() or entry.suffix.lower() in ASSETS:
                            continue
                        # PyTorch tensor storage is not a reference. Read all metadata,
                        # including extensionless members, without loading pickle code.
                        if torch_archive and path.suffix in ('.pt', '.pth', '.bin', '.ckpt') and entry.parent.name == 'data' and entry.name.isdigit():
                            continue
                        with archive.open(member) as stream:
                            scan(stream, owner, path, member.filename)
            else:
                # Legacy pickle may contain paths; inspect bytes without deserialization.
                with path.open('rb') as stream:
                    scan(stream, owner, path)
        elif path.suffix == '.gz':
            with gzip.open(path, 'rb') as stream:
                scan(stream, owner, path)
        else:
            with path.open('rb') as stream:
                scan(stream, owner, path)
        if stamp(path) != identity:
            raise ValueError('file changed during scan: '+str(path))

    for entry in entries:
        visit(entry, entry.name)
    # Only retained roots (and their transitive dependencies) need their contents
    # read. Corrupt, unreachable obsolete outputs cannot be dependencies of them.
    pending = list(sorted(set(edges)-set(candidates)))
    read = set()
    while pending:
        owner = pending.pop()
        if owner in read:
            continue
        read.add(owner)
        for path in files[owner]:
            read_references(path, owner)
        pending.extend((edges[owner]-read) & edges.keys())
    return edges, snapshots, sizes, consumed, cycles


def plan(root, *, timeout=600, max_bytes=128*1024**3, feedback_files=frozenset()):
    cp = root/'checkpoints'
    present = {n for n in CANDIDATES if (cp/n).exists()}
    if any((cp/n).is_symlink() for n in CANDIDATES):
        raise ValueError('candidate is a symlink; refuse cleanup')
    active = live_processes(root)
    if active:
        raise ValueError('project processes are still running; do not stop them automatically: '+json.dumps(active))
    edges, snapshots, sizes, count, cycles = inventory(cp, present, time.monotonic()+timeout, max_bytes, feedback_files)
    protected = set(edges)-present
    pending = list(protected)
    while pending:
        owner = pending.pop()
        for target in edges[owner]-protected:
            protected.add(target)
            pending.append(target)
    selected = sorted(present-protected)
    return dict(delete=selected, kept_referenced=sorted(present & protected),
        protected_roots=sorted(set(edges)-present),
        references={k: sorted(v) for k, v in edges.items() if v},
        estimated_bytes={n: sizes[n] for n in selected}, scanned_bytes=count,
        symlink_cycles=cycles), snapshots


def execute(root, report, snapshots):
    if live_processes(root):
        raise ValueError('project process started after scan; nothing deleted')
    for name, expected in snapshots.items():
        if stamp(Path(name)) != expected:
            raise ValueError('inventory changed after scan; nothing deleted: '+name)
    if not shutil.rmtree.avoids_symlink_attacks:
        raise ValueError('platform lacks fd-based safe rmtree')
    deleted = []
    for name in report['delete']:
        if name not in CANDIDATES or '/' in name:
            raise ValueError('not an exact authorized candidate')
        path = root/'checkpoints'/name
        if path.is_symlink() or stamp(path) != snapshots[str(path)]:
            raise ValueError('candidate replaced or changed')
        print('[delete] '+str(path), flush=True)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        deleted.append(name)
        report['deleted'] = deleted.copy()


def feedback_files(root, out, script):
    """Allow a dedicated current-directory handoff without hiding existing assets."""
    if out.exists() or out.is_symlink():
        raise ValueError('Use a new output JSON')
    if not out.resolve().is_relative_to(root):
        return frozenset()
    folder = out.parent
    if (folder.is_symlink() or folder.resolve() != folder
            or folder.parent not in (root, root/'checkpoints')
            or not folder.name.startswith('automot_cleanup.')
            or out.name not in ('cleanup_report.json', 'reference_trace.json')):
        raise ValueError('Use a fresh automot_cleanup.* directory directly under the project or checkpoints')
    source = folder/'checkpoints_cleanup.py'
    log = folder/'launch.log'
    if script.resolve() != source or set(folder.iterdir()) != {source, log}:
        raise ValueError('Feedback directory must contain only this script and launch.log')
    for path in (source, log):
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError('Feedback files must be regular files, not links')
    return frozenset((source, log, out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true', help='Permanently delete unreferenced exact historical candidates')
    mode.add_argument('--trace-retained', action='store_true', help='Read-only: locate retained references in the two known source roots')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if root != Path('/home/cruser1/lda/AutoMoT'):
        parser.error('This explicit cleanup list is authorized only for /home/cruser1/lda/AutoMoT on cr02')
    if socket.gethostname().split('.')[0] != 'szh-gpu-cr02':
        parser.error('This list was reviewed for szh-gpu-cr02 only')
    if (root/'checkpoints').is_symlink():
        parser.error('checkpoints must not be a symlink')
    for name in REQUIRED:
        if not (root/'checkpoints'/name).is_dir():
            parser.error('protected asset missing: '+name)
    out = args.output.absolute()
    try:
        excluded = feedback_files(root, out, Path(__file__))
    except ValueError as exc:
        parser.error(str(exc))
    report = dict(schema='exact_checkpoints_cleanup_v1', root=str(root), hostname=socket.gethostname(),
                  apply=args.apply, deleted=[], backups_created=False, status='checking')
    # Ensure the feedback file is writable before considering any deletion.
    output = out.open('x')
    fd = os.open(root/'checkpoints', os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report['free_before'] = shutil.disk_usage(root/'checkpoints').free
        if args.trace_retained:
            report.update(scope='Read-only reference origins in two roots; not a deletion authorization',
                          reference_sources=[], trace_roots=list(TRACE_OWNERS))
            print('[trace] two known reference roots only; max 180s / 16 GiB', flush=True)
            cp = root/'checkpoints'
            present = {name for name in CANDIDATES if (cp/name).exists()}
            edges, snapshots, _, count, cycles = inventory(cp, present, time.monotonic()+180,
                16*1024**3, excluded, owners=TRACE_OWNERS, reference_trace=report['reference_sources'])
            for name, expected in snapshots.items():
                if stamp(Path(name)) != expected:
                    raise ValueError('inventory changed during reference tracing: '+name)
            report.update(status='references_collected', scanned_bytes=count, symlink_cycles=cycles,
                          references={name: sorted(values) for name, values in edges.items()})
            print(f"[trace] collected {len(report['reference_sources'])} source records; no deletion", flush=True)
        else:
            print('[scan] retained references and old outputs; max 600s / 128 GiB decoded metadata', flush=True)
            selected, snapshots = plan(root, feedback_files=excluded)
            report.update(selected)
            print(json.dumps(selected, ensure_ascii=False, indent=2), flush=True)
            if args.apply:
                execute(root, report, snapshots)
            report['status'] = 'completed' if args.apply else 'preview'
    except Exception as exc:
        report.update(status='blocked_or_partial', error=str(exc))
    finally:
        os.close(fd)
        report['free_after'] = shutil.disk_usage(root/'checkpoints').free
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.close()
        print(f"Deleted {len(report['deleted'])} entries; free space now {report['free_after']/1024**3:.2f} GiB", flush=True)
        print('REPORT '+str(out), flush=True)
    return 0 if report['status'] in ('completed', 'preview', 'references_collected') else 2


if __name__ == '__main__':
    raise SystemExit(main())
