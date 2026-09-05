"""不加载真实大模型，验证权重/提示词/数据合同。"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pytest
from qwen3vl_local.action_prior.contracts import (
    inspect_adapter,
    select_adapter,
    require_contract,
)
from qwen3vl_local.action_prior import priors
from qwen3vl_local.action_prior.build_dataset import route_group, split_for
from qwen3vl_local.action_prior.config import parser, validate_args
from qwen3vl_local.action_prior.train import audit_counts, metrics_from_counts
from qwen3vl_local.sft_new_loop_phase1 import prompts as p1
from qwen3vl_local.sft_new_loop_phase2 import prompts as p2


def fixture_adapter(root, phase=1, slot="best_generation", score=0.8, mode="4rgb"):
    path = root / slot
    path.mkdir(parents=True)
    cfg = dict(
        prompt_name=(p1 if phase == 1 else p2).PROMPT_NAME,
        production_prompt_sha256=(
            p1.phase1_prompt_sha256 if phase == 1 else p2.event_prompt_sha256
        )(history_rgb_mode=mode),
        git={"commit": "abcdef"},
        history_rgb_mode=mode,
        history_rgb_count=4 if mode == "4rgb" else 2,
        history_rgb_selected_indices=[0, 1, 2, 3] if mode == "4rgb" else [0, 3],
        base_model_dir=str(root.parent / "base"),
        global_step=100,
        generation_format_valid_gate=0.9,
    )
    (path / f"sft_new_loop_phase{phase}_adapter_config.json").write_text(
        json.dumps(cfg)
    )
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "bias": "none"})
    )
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    (root / "train_eval_metrics.jsonl").write_text(
        json.dumps(
            dict(
                step=100, type="generation", exact_accuracy=score, format_valid_rate=1.0
            )
        )
        + "\n"
    )
    (root / "best_generation.json").write_text(
        json.dumps(
            dict(step=100, generation_exact_accuracy=score, generation_guards_ok=True)
        )
    )
    return path


@pytest.mark.parametrize("phase", [1, 2])
def test_auto_only_best_and_highest_validated(tmp_path, phase):
    a = fixture_adapter(tmp_path / "a", phase, score=0.7)
    b = fixture_adapter(tmp_path / "b", phase, score=0.9, mode="2rgb_endpoints")
    fixture_adapter(tmp_path / "c", phase, slot="final", score=1.0)
    got = select_adapter(tmp_path, phase, tmp_path / "base")
    assert got["path"] == str(b)
    assert got["metadata"]["history_rgb_selected_indices"] == [0, 3]


def test_no_final_fallback(tmp_path):
    fixture_adapter(tmp_path / "a", slot="final")
    with pytest.raises(ValueError, match="no compatible best_generation"):
        select_adapter(tmp_path, 1, tmp_path / "base")


@pytest.mark.parametrize(
    "key,value",
    [
        ("prompt_name", "old"),
        ("production_prompt_sha256", "bad"),
        ("git", {}),
        ("history_rgb_count", 2),
        ("base_model_dir", "/wrong"),
    ],
)
def test_bad_metadata_rejected(tmp_path, key, value):
    path = fixture_adapter(tmp_path / "a")
    file = path / "sft_new_loop_phase1_adapter_config.json"
    cfg = json.loads(file.read_text())
    cfg[key] = value
    file.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        inspect_adapter(path, 1, tmp_path / "base")


def test_explicit_final_recorded_not_best(tmp_path):
    path = fixture_adapter(tmp_path / "a", slot="final")
    got = select_adapter(tmp_path, 1, tmp_path / "base", str(path))
    assert got["selection"] == "explicit" and not got["is_best_generation"]


def test_incompatible_resume_rejected():
    with pytest.raises(ValueError):
        require_contract({"schema": "leadmot_qwen_backbone_v1"}, {"identity": "a"})


def test_repeat_routes_same_split():
    a = route_group("X", "Town12_Rep0_route_1054_0_route0_07_03_01_02_03")
    b = route_group("X", "Town12_Rep3_route_1054_0_route0_09_01_01_02_03")
    assert a == b and split_for(a, 2026) == split_for(b, 2026)
    assert "Town12_route15" in route_group(
        "X", "Town12_Rep0_Town12_route15_07_03_01_02_03"
    )


@pytest.mark.parametrize(
    "raw", ["B: YES\nA: NO", "A: NO\nB: YES\nextra", "A: NO\nA: NO", "A: MAYBE\nB: YES"]
)
def test_strict_parser(raw):
    assert all(v is None for v in priors.strict_answers(raw, ["A", "B"]).values())


def ask_fixture(disagree=None, bad_domain=False, multi=False):
    def ask(phase, spec, history):
        if phase == 1:
            answers = {k: "NO" for k in spec.output_keys}
            if "RS1" in answers:
                answers["RS1"] = "YES"
            if multi and "RS2" in answers:
                answers["RS2"] = "YES"
            if "GROUP" in answers:
                group = spec.phase2_spec.questions[1].question_id
                answers["GROUP"] = (
                    "YES" if "R1" in p1.GROUP_DEFINITIONS[group][3] else "NO"
                )
            if history and disagree in answers:
                answers[disagree] = "NO" if answers[disagree] == "YES" else "YES"
        else:
            answers = {k: "NO" for k in spec.output_keys}
            if spec.question_domain == p2.JUNCTION_DOMAIN:
                answers[p2.INVALID_KEY] = "YES"
            if history and bad_domain:
                answers[p2.INVALID_KEY] = "YES"
        return "\n".join(f"{k}: {answers[k]}" for k in spec.output_keys), "prompt"

    return ask


def test_all_rs_rechecked_and_both_event_domains_asked():
    x = priors.collect_priors(ask_fixture(), "case")
    assert len(x["calls"]) == 9
    assert x["conditions"]["ROAD_STRUCTURE"] == "R1"
    assert x["conditions"]["UE3"] == "NO"
    assert x["conditions"]["UE6"] is None
    assert x["invalid"]["UE6"] == "domain_inapplicable"
    assert all(c["history"] for c in x["calls"][1:5])


@pytest.mark.parametrize("key", ["HIGHWAY", "RS1", "RS4"])
def test_disagreement_preserves_other_conditions(key):
    x = priors.collect_priors(ask_fixture(disagree=key), "case")
    assert x["conditions"][key] is None
    assert x["conditions"]["UE3"] == "NO"
    if key.startswith("RS"):
        assert x["conditions"]["ROAD_STRUCTURE"] is None


def test_domain_disagreement_blanks_ues():
    x = priors.collect_priors(ask_fixture(bad_domain=True), "case")
    assert x["conditions"]["UE3"] is None
    assert x["invalid"]["UE3"] == "domain_unconfirmed"


def test_multi_yes_not_accepted():
    x = priors.collect_priors(ask_fixture(multi=True), "case")
    assert x["conditions"]["ROAD_STRUCTURE"] is None


def test_invalid_still_in_denominator():
    c = audit_counts(dict(invalid={"UE3": "disagreement"}, analysis_truncated=False))
    c.update(audit_counts(dict(invalid={}, analysis_truncated=False)))
    m = metrics_from_counts(c)
    assert m["samples"] == 2 and m["prior/invalid_samples"] == 0.5


def test_forbid_future_truth_navigation():
    args = parser().parse_args(["--tp-mode", "future_truth"])
    with pytest.raises(ValueError):
        validate_args(args)


def test_builder_synthetic_all_splits_and_abnormal(tmp_path, monkeypatch):
    from qwen3vl_local.action_prior import build_dataset as build

    data, out = tmp_path / "data", tmp_path / "index"
    # 保证三个哈希 split 均有路线；文件只用于枚举，不加载实际 RGB/meta。
    found = {}
    for i in range(100):
        run = f"Town01_Rep0_route_{i:06d}_route0_01_01_00_00_00"
        split = build.split_for(build.route_group("Scenario", run), 2026)
        found.setdefault(split, run)
        if len(found) == 3:
            break
    for run in found.values():
        for folder, ext in (("rgb", ".jpg"), ("metas", ".pkl"), ("lidar", ".laz")):
            d = data / "Scenario" / run / folder
            d.mkdir(parents=True)
            for i in range(12):
                (d / f"{i:04d}{ext}").touch()
    abnormal = data / "Accident" / "Town01_Rep0_route_999_route0_01_01_00_00_00"
    for folder in ("rgb", "metas", "lidar"):
        (abnormal / folder).mkdir(parents=True)
    for i in range(361):
        (abnormal / "rgb" / f"{i:04d}.jpg").touch()
    monkeypatch.setattr(
        sys, "argv", ["build", "--data-root", str(data), "--output-dir", str(out)]
    )
    build.main()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"] == {"train": 4, "val": 4, "test": 4}
    assert manifest["skipped"]["abnormal_duration"] == 1
    for split in ("train", "val", "test"):
        rows = [
            json.loads(s) for s in (out / f"{split}.jsonl").read_text().splitlines()
        ]
        assert [r["anchor"] for r in rows] == [0, 1, 2, 3]
        assert all(not Path(r["route_dir"]).is_absolute() for r in rows)
