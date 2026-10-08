"""Launch one owned process group with core limits and a free-space reserve."""
import argparse
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys

from .storage import GIB, existing_parent


def core_pattern():
    return Path('/proc/sys/kernel/core_pattern').read_text().strip()


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def stop_group(process):
    # Only the group we created; never pkill Python/torch or unrelated jobs.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # The leader may have exited while a worker ignored SIGTERM.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run(command, space_path, min_free_bytes=2 * GIB, poll_seconds=1):
    if not command or min_free_bytes < 0 or poll_seconds <= 0:
        raise ValueError('command, nonnegative reserve and positive polling interval required')
    location = existing_parent(space_path)
    pattern = core_pattern()
    if pattern.startswith('|'):
        raise ValueError('piped core collector detected: RLIMIT_CORE cannot guarantee no dump. '
                         'No job launched. Inspect collector policy with the server administrator; '
                         'no global kernel setting was changed. core_pattern=' + pattern)
    if shutil.disk_usage(location).free < min_free_bytes:
        raise OSError(f'free space below reserve {min_free_bytes} bytes at {location}; no job launched')
    process = subprocess.Popen(command, start_new_session=True, preexec_fn=no_core)
    print(f'[audit guard] pid={process.pid} core_soft=0 core_hard=0 '
          f'filesystem={location} reserve_bytes={min_free_bytes}', file=sys.stderr, flush=True)
    interrupted = []
    def on_signal(number, frame):
        interrupted.append(number)
    handlers = {number: signal.signal(number, on_signal) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        while True:
            if interrupted:
                stop_group(process)
                return 128 + interrupted[0]
            if shutil.disk_usage(location).free < min_free_bytes:
                print('[audit guard] low disk space; stopping this process group. '
                      'Existing files retained; no successful completion claimed.', file=sys.stderr, flush=True)
                stop_group(process)
                return 75
            try:
                code = process.wait(timeout=poll_seconds)
                return code if code >= 0 else 128 - code
            except subprocess.TimeoutExpired:
                pass
    except BaseException:
        stop_group(process)
        raise
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--space-path', type=Path, required=True)
    parser.add_argument('--min-free-gib', type=float, default=2)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        parser.error('--min-free-gib must be finite and nonnegative')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        return run(command, args.space_path, int(args.min_free_gib * GIB))
    except (ValueError, OSError) as error:
        print('[audit guard] ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
