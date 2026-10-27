"""Read-only CPU evidence collection after capacity rejection; no model imports."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

from qwen3vl_local.sft_new_loop_phase1.training_policy import ActionHints

LIMIT = 64 * 1024 * 1024


def collect(app, run, output, action_index="auto"):
    app, run, output = map(lambda p: Path(p).resolve(), (app, run, output))
    if not run.is_dir():
        raise ValueError("Missing failed run directory: " + str(run))
    state = json.loads((run / "status.json").read_text())
    if state.get("phase") != 2 or state.get("status") != "failed" or state.get("stage") != "cpu_admission":
        raise ValueError("Only a Phase2 CPU-admission failure may be collected")
    if "Coverage quota exceeds compatible frame capacity" not in (run / "cpu_admission.log").read_text():
        raise ValueError("No capacity rejection in CPU log")
    # Bind the exact index recorded by the failed plan, including explicit parents.
    plan = run / "cpu/2rgb_endpoints_sampling.json"
    command = json.loads(plan.read_text())["command"]
    index = Path(command[command.index("--index") + 1]).resolve()
    selected_action = command[command.index("--action-index") + 1]
    if selected_action != "auto":
        action_index = selected_action
    payloads = {}
    def add(path, name):
        path = Path(path)
        if not path.is_file() or path.stat().st_size > LIMIT:
            raise ValueError("Missing/oversized evidence: " + str(path))
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Evidence changed during collection: " + str(path))
        payloads[name] = raw
    add(index, "data/frame_index.jsonl")
    add(index.parent / "manifest.json", "data/manifest.json")
    for path in sorted(run.rglob("*")):
        if path.is_symlink():
            raise ValueError("Unexpected symlink inside failed run: " + str(path))
        if path.is_file() and path.suffix in (".json", ".log"):
            add(path, "run/" + str(path.relative_to(run)))
    version = Path(state["version"])
    add(version / "manifest.json", "runtime/manifest.json")
    source_names = ["sft_new_loop_phase2/train.py", "sft_new_loop_phase2/invalid_balance.py",
                    "sft_new_loop_phase1/training_policy.py", "sft_new_loop_phase1/exposure_sampling.py"]
    for name in source_names:
        add(version / "app/qwen3vl_local" / name, "runtime/" + name)
    rows = [SimpleNamespace(**json.loads(line)) for line in payloads["data/frame_index.jsonl"].splitlines() if line.strip()]
    hints = ActionHints.load(action_index, rows, app)
    payloads["effective_action_hints.json"] = json.dumps(dict(
        requested=action_index, source=hints.receipt,
        values=[[*key, value] for key, value in sorted(hints.values.items())],
        scope="current read-only reconstruction; original failed process did not persist action receipt",
    ), ensure_ascii=False, indent=2).encode()
    receipt = dict(status="collected_for_local_capacity_audit", phase=2, run=str(run), index=str(index),
        model_loaded=False, rgb_read=False, dataset_changed=False,
        files={name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for name,raw in payloads.items()})
    payloads["collection.json"] = json.dumps(receipt, ensure_ascii=False, indent=2).encode()
    if sum(map(len, payloads.values())) > 128 * 1024 * 1024:
        raise ValueError("Evidence exceeds 128 MiB; no partial archive written")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as tar:
        for name, raw in payloads.items():
            info = tarfile.TarInfo(name); info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    print("[BUNDLE] " + str(output), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(Path(__file__).resolve().parents[3], args.run_dir, args.output)


if __name__ == "__main__":
    main()
