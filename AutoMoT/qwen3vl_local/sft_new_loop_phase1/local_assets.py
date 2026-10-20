"""Stable local dataset names, explicit bindings and non-overwriting run slots."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import wraps
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket


def claim_training_output(output_dir):
    """Shell may have created train.log; every actual trainer gets one claim."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    unexpected = [p.name for p in path.iterdir() if p.name != "train.log"]
    if unexpected:
        raise ValueError(f"Training output already contains results: {path}: {unexpected[:5]}; choose an explicit new method/experiment directory")
    with (path / "training.claim").open("x") as handle:
        json.dump({"pid": os.getpid(), "hostname": socket.gethostname()}, handle)


def exclusive_build(function):
    """Keep the existing directory lock for the entire build; no lock deletion."""
    @wraps(function)
    def wrapped(args):
        path = Path(args.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f"Dataset build already active: {path}") from exc
            return function(args)
        finally:
            os.close(fd)
    return wrapped


def inspect_index(index, phase):
    index = Path(index).expanduser()
    manifest_path = index.parent / "manifest.json"
    if not index.is_file() or not manifest_path.is_file():
        raise ValueError(f"incomplete dataset: need {index} and {manifest_path}; no implicit rebuild")
    manifest = json.loads(manifest_path.read_text())
    expected = f"sft_new_loop_phase{phase}"
    # Use the phase's own dataset constant, which may carry a semantic suffix.
    if phase == 1:
        from qwen3vl_local.sft_new_loop_phase1 import DATASET_NAME
        from qwen3vl_local.sft_new_loop_phase1.prompts import ANSWER_KEYS
    else:
        from qwen3vl_local.sft_new_loop_phase2 import DATASET_NAME
        from qwen3vl_local.sft_new_loop_phase2.prompts import ANSWER_KEYS
    if manifest.get("dataset_name") != DATASET_NAME:
        raise ValueError(f"wrong dataset method for {expected}: {manifest.get('dataset_name')}")
    if phase == 2 and manifest.get("format") != "sft_new_loop_phase2_frame_index_v2_highway_ue3":
        raise ValueError("Phase2 index predates the existing highway UE3 contract; cannot silently relabel or rebuild it")
    digest = hashlib.sha256()
    counts = Counter()
    with index.open("rb") as handle:
        for number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("dataset_name") != DATASET_NAME or row.get("split") not in {"train", "val", "test"}:
                raise ValueError(f"invalid method/split at {index}:{number}")
            if not isinstance(row.get("answers"), dict) or len(row.get("history_rgb_paths", [])) != 4:
                raise ValueError(f"invalid answers/RGB history at {index}:{number}")
            if any(type(row['answers'].get(key)) is not bool for key in ANSWER_KEYS):
                raise ValueError(f"missing/non-boolean labels at {index}:{number}; no implicit False coercion")
            if not row.get("scenario") or not row.get("route_id") or type(row.get("frame_id")) is not int:
                raise ValueError(f"invalid frame identity at {index}:{number}")
            counts[row["split"]] += 1
    for split in ("train", "val", "test"):
        if not counts[split] or manifest.get("counts", {}).get(f"frames/{split}") != counts[split]:
            raise ValueError(f"missing/mismatched {split} count: index={counts[split]} manifest={manifest.get('counts')}")
    return {"method": expected, "index": str(index.absolute()), "resolved_index": str(index.resolve()),
            "index_sha256": digest.hexdigest(), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "counts": dict(counts), "hostname": socket.gethostname(), "reused": True,
            "note": "Same method name on two machines does not imply equal data. No source/RGB/label regeneration."}


def ensure_data(index, phase, run_build):
    path = Path(index).expanduser()
    if path.exists() or path.is_symlink() or (path.parent / "manifest.json").exists():
        return "reuse", inspect_index(path, phase)
    if run_build == "0":
        raise ValueError(f"RUN_BUILD=0 but dataset missing: {path}; bind an existing local dataset")
    # A nonempty directory may be an interrupted build: never overwrite it.
    if path.parent.exists() and any(path.parent.iterdir()):
        raise ValueError(f"dataset directory is nonempty but incomplete: {path.parent}; choose/bind an explicit intact local dataset")
    return "build", {"method": f"sft_new_loop_phase{phase}", "index": str(path), "reused": False}


def bind_data(source, target, phase):
    source, target = Path(source).expanduser().resolve(), Path(target).expanduser().absolute()
    if source == target or target.exists() or target.is_symlink():
        raise ValueError(f"binding target already exists: {target}; existing assets are never replaced")
    receipt = inspect_index(source / "frame_index.jsonl", phase)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)
    return receipt


def check_launch(modes, gpu_ids, output_dirs):
    modes = modes.split()
    if not modes or len(set(modes)) != len(modes) or set(modes) - {'4rgb','2rgb_endpoints'}:
        raise ValueError(f"Invalid/duplicate HISTORY_RGB_MODES: {modes}")
    ids = gpu_ids.split(',')
    if not ids or any(not x.isdigit() for x in ids) or len(set(map(int,ids))) != len(ids):
        raise ValueError(f"Invalid/duplicate GPU_IDS: {gpu_ids}")
    resolved = [Path(x).expanduser().resolve() for x in output_dirs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Multiple RGB modes share one TRAIN_OUTPUT_DIR; use separate directories")
    for path in output_dirs:
        path = Path(path).expanduser()
        if path.exists() or path.is_symlink():
            raise ValueError(f"Training output already exists BEFORE dataset/model work: {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    ensure = sub.add_parser("ensure")
    ensure.add_argument("--index", required=True)
    ensure.add_argument("--phase", type=int, choices=(1,2), required=True)
    ensure.add_argument("--run-build", choices=("auto", "0", "1"), default="auto")
    ensure.add_argument("--receipt", required=True)
    bind = sub.add_parser("bind")
    bind.add_argument("--source", required=True)
    bind.add_argument("--target", required=True)
    bind.add_argument("--phase", type=int, choices=(1,2), required=True)
    reserve = sub.add_parser("reserve-output")
    reserve.add_argument("--path", required=True)
    launch = sub.add_parser('check-launch')
    launch.add_argument('--modes', required=True)
    launch.add_argument('--gpu-ids', required=True)
    launch.add_argument('--train-output', action='append', default=[])
    args = p.parse_args()
    if args.command == "ensure":
        action, receipt = ensure_data(args.index, args.phase, args.run_build)
        Path(args.receipt).write_text(json.dumps(receipt, indent=2)+"\n")
        print(action)
    elif args.command == "bind":
        print(json.dumps(bind_data(args.source, args.target, args.phase), indent=2))
    elif args.command == 'check-launch':
        check_launch(args.modes, args.gpu_ids, args.train_output)
    else:
        path = Path(args.path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.mkdir()
        except FileExistsError as exc:
            raise ValueError(f"Output already exists: {path}. Preserved unchanged. Use an explicit method/experiment name in OUTPUT_DIR or PIPELINE_ROOT; never auto-resume/restart over it.") from exc


if __name__ == "__main__":
    main()
