"""数据集标定真值先验：条件展开、索引查表、噪声注入、参数校验与合同身份；不加载真实模型。"""

from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pytest
from qwen3vl_local.action_prior import dataset_labels as labels
from qwen3vl_local.action_prior.config import build_contract, parser, validate_args
from qwen3vl_local.action_prior.contracts import decoder_identity, require_contract
from qwen3vl_local.action_prior.provenance import annotate_upstream
from qwen3vl_local.sft_new_loop_phase1 import prompts as p1
from qwen3vl_local.sft_new_loop_phase2 import prompts as p2

NO_FACTS = {key: False for key in p1.PHASE1_ANSWER_KEYS}
ROAD_STRUCTURES = labels.ROAD_STRUCTURES


def write_index(root, rows):
    """写出与 build_prior_labels.py 相同的索引与 manifest。"""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "prior_labels.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            dict(schema=labels.LABEL_SCHEMA, counts=dict(labeled_frames=len(rows)))
        ),
        encoding="utf-8",
    )
    return path


def row(scenario="S", route="R", frame=0, rs="R1", target="RE",
        domain=p2.ROAD_DOMAIN, invalid_domain=p2.JUNCTION_DOMAIN, facts=None):
    return dict(
        scenario=scenario,
        route_id=route,
        frame_id=frame,
        road_structure=rs,
        target_event_class=target,
        question_domain=domain,
        invalid_domain=invalid_domain,
        phase1_answers=dict(NO_FACTS, **(facts or {})),
    )


def conditions(**kwargs):
    item = row(**kwargs)
    return labels.conditions_from_label(
        labels.pack(
            item["road_structure"],
            item["target_event_class"],
            item["question_domain"],
            item["invalid_domain"],
            item["phase1_answers"],
        )
    )


def test_road_structure_and_facts_expand_without_default_no():
    values, invalid, _ = conditions(rs="R4", target="UE6", domain=p2.JUNCTION_DOMAIN,
                                    invalid_domain=p2.ROAD_DOMAIN, facts={"VULNERABLE": True})
    assert values["ROAD_STRUCTURE"] == "R4" and values["RS4"] == "YES"
    assert [values[k] for k in ("RS1", "RS2", "RS5")] == ["NO", "NO", "NO"]
    assert values["VULNERABLE"] == "YES" and values["HIGHWAY"] == "NO"
    assert values["UE6"] == "YES"
    assert values[f"{p2.JUNCTION_DOMAIN}/{p2.INVALID_KEY}"] == "NO"
    assert values[f"{p2.ROAD_DOMAIN}/{p2.INVALID_KEY}"] == "YES"
    assert {invalid[k] for k in p2.event_keys_for_domain(p2.ROAD_DOMAIN)} == {
        "domain_inapplicable"
    }


def test_r3_is_not_a_highway_fact():
    values, _, _ = conditions(rs="R3")
    assert values["RS_HIGHWAY"] is None
    assert all(values[key] == "NO" for key in p1.PHASE2_ANSWER_KEYS)
    assert conditions(rs="R1")[0]["RS_HIGHWAY"] is None


def test_event_overlay_leaves_the_other_domain_unknown():
    values, invalid, _ = conditions(rs="R5", target="UE3", domain=p2.ROAD_DOMAIN,
                                    invalid_domain=None)
    assert values["UE3"] == "YES" and values["UE1"] == "NO"
    assert values["UE6"] is None
    assert invalid[f"{p2.JUNCTION_DOMAIN}/{p2.INVALID_KEY}"] == "dataset_domain_unlabeled"
    assert invalid["UE6"] == "domain_unconfirmed"


def test_pack_rejects_invalid_domain_equal_to_question_domain():
    with pytest.raises(ValueError, match="must differ"):
        labels.pack("R1", "RE", p2.ROAD_DOMAIN, p2.ROAD_DOMAIN, NO_FACTS)


def test_missing_frame_stays_completely_unknown(tmp_path):
    index = labels.PriorLabelIndex(write_index(tmp_path / "labels", [row(frame=7)]))
    hit = index.priors(("S", "R", 7))
    assert hit["conditions"]["ROAD_STRUCTURE"] == "R1"
    assert hit["invalid"] == {"UE6": "domain_inapplicable"}
    assert hit["calls"] == [] and hit["prior_source"] == labels.PRIOR_SOURCE
    miss = index.priors(("S", "R", 8))
    assert set(miss["conditions"]) == set(labels.CONDITION_KEYS)
    assert all(value is None for value in miss["conditions"].values())
    assert set(miss["invalid"].values()) == {"dataset_label_missing"}
    coverage = index.coverage(
        {"val": [dict(scenario="S", run_id="R", anchor=7), dict(scenario="S", run_id="R", anchor=8)]}
    )
    assert coverage["val/labeled"] == 1 and coverage["val/missing"] == 1


def test_row_count_must_match_manifest(tmp_path):
    path = write_index(tmp_path / "labels", [row()])
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row(frame=1)) + "\n")
    with pytest.raises(ValueError, match="row count"):
        labels.PriorLabelIndex(path)


def test_switch_requires_prior_mode_and_a_label_index(tmp_path):
    path = write_index(tmp_path / "labels", [row()])
    validate_args(parser().parse_args(["--dataset-priors", "--prior-labels", str(path)]))
    with pytest.raises(ValueError, match="prior-labels"):
        validate_args(parser().parse_args(["--dataset-priors"]))
    with pytest.raises(ValueError, match="condition-mode prior"):
        validate_args(parser().parse_args(
            ["--dataset-priors", "--prior-labels", str(path), "--condition-mode", "base"]))
    with pytest.raises(ValueError, match="only used together"):
        validate_args(parser().parse_args(["--prior-labels", str(path)]))


def dataset_args(tmp_path):
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    (base / "model.safetensors").write_bytes(b"base fixture")
    bev = tmp_path / "bev.pt"
    bev.write_bytes(b"BEV fixture")
    path = write_index(tmp_path / "labels", [row()])
    return parser().parse_args(
        ["--model-dir", str(base), "--lead-bev-ckpt", str(bev),
         "--dataset-priors", "--prior-labels", str(path)]
    )


def test_contract_pins_labels_and_never_selects_an_adapter(tmp_path):
    contract = build_contract(dataset_args(tmp_path))
    assert contract["phase1"] is None and contract["phase2"] is None
    assert contract["lora_adapters_loaded"] is False
    assert contract["prior_labels"]["schema"] == labels.LABEL_SCHEMA
    assert contract["identity_payload"]["prior_source"] == labels.PRIOR_SOURCE
    require_contract(contract, contract)
    exposure = annotate_upstream(
        [dict(route_group="S/R")], contract["upstream_sources"]
    )
    assert exposure["combined/dataset_label_lookup"] == 1


def test_explicit_adapter_options_are_rejected_in_dataset_mode(tmp_path):
    args = dataset_args(tmp_path)
    args.phase1_adapter = str(tmp_path)
    with pytest.raises(ValueError, match="never load a LoRA"):
        build_contract(args)


def noisy_index(tmp_path, rate, invalid_share=0.25, seed=0, count=4000):
    rows = [
        row(route=f"R{i % 7}", frame=i, rs=ROAD_STRUCTURES[i % 5],
            target=("UE6" if ROAD_STRUCTURES[i % 5] in ("R4", "R5") else "UE1"),
            domain=(p2.JUNCTION_DOMAIN if ROAD_STRUCTURES[i % 5] in ("R4", "R5")
                    else p2.ROAD_DOMAIN),
            invalid_domain=(p2.ROAD_DOMAIN if ROAD_STRUCTURES[i % 5] in ("R4", "R5")
                            else p2.JUNCTION_DOMAIN))
        for i in range(count)
    ]
    path = write_index(tmp_path / f"labels_{rate}_{seed}", rows)
    return rows, labels.PriorLabelIndex(
        path, labels.PriorNoise(rate, invalid_share, seed)
    )


def test_noise_hits_the_requested_rate_and_splits_rs_event(tmp_path):
    rows, index = noisy_index(tmp_path, 0.1)
    notes = [
        index.priors((r["scenario"], r["route_id"], r["frame_id"]))["dataset_label"]["noise"]
        for r in rows
    ]
    hit = [n for n in notes if n]
    assert 0.08 <= len(hit) / len(rows) <= 0.12
    channels = Counter(n["channel"] for n in hit)
    assert 0.35 <= channels["rs"] / len(hit) <= 0.65
    modes = Counter(n["mode"] for n in hit)
    assert 0.15 <= modes["invalid"] / len(hit) <= 0.35


def test_zero_rate_leaves_every_frame_clean(tmp_path):
    rows, index = noisy_index(tmp_path, 0.0)
    assert index.noise is None
    assert all(
        index.priors((r["scenario"], r["route_id"], r["frame_id"]))["dataset_label"]["noise"] is None
        for r in rows
    )


def test_same_frame_and_seed_always_draw_the_same_noise(tmp_path):
    """文本缓存保存的是该帧先验对应的 base 输出，逐 epoch 重抽会让两者不一致。"""
    rows, index = noisy_index(tmp_path, 0.5, seed=7)
    identity = (rows[0]["scenario"], rows[0]["route_id"], rows[0]["frame_id"])
    first = index.priors(identity)
    assert first["conditions"] == index.priors(identity)["conditions"]
    _, other_seed = noisy_index(tmp_path, 0.5, seed=8)
    changed = sum(
        index.priors((r["scenario"], r["route_id"], r["frame_id"]))["conditions"]
        != other_seed.priors((r["scenario"], r["route_id"], r["frame_id"]))["conditions"]
        for r in rows
    )
    assert changed > 0


def test_confusion_noise_stays_a_well_formed_wrong_answer(tmp_path):
    rows, index = noisy_index(tmp_path, 1.0, invalid_share=0.0)
    seen = Counter()
    for r in rows:
        priors = index.priors((r["scenario"], r["route_id"], r["frame_id"]))
        note, values = priors["dataset_label"]["noise"], priors["conditions"]
        seen[note["channel"]] += 1
        assert note["mode"] == "confusion" and note["original"] != note["corrupted"]
        assert not any(v.startswith("noise_") for v in priors["invalid"].values())
        if note["channel"] == "rs":
            rs = values["ROAD_STRUCTURE"]
            assert rs == note["corrupted"] and rs != r["road_structure"]
            # 错误的 RS 仍然是自洽向量：单一 YES；压缩标签没有独立 highway bit。
            assert sum(values[k] == "YES" for k in p1.PHASE2_ANSWER_KEYS) == (rs != "R3")
            assert values["RS_HIGHWAY"] is None
        else:
            domain = r["question_domain"]
            answered = [k for k in p2.event_keys_for_domain(domain) if values[k] == "YES"]
            assert answered == ([note["corrupted"]] if note["corrupted"] != "RE" else [])
            assert values[f"{domain}/{p2.INVALID_KEY}"] == "NO"
    assert seen["rs"] and seen["event"]


def test_invalid_noise_reproduces_real_failure_shapes(tmp_path):
    rows, index = noisy_index(tmp_path, 1.0, invalid_share=1.0)
    seen = Counter()
    for r in rows:
        priors = index.priors((r["scenario"], r["route_id"], r["frame_id"]))
        note, values = priors["dataset_label"]["noise"], priors["conditions"]
        seen[note["channel"]] += 1
        assert note["mode"] == "invalid"
        if note["channel"] == "rs":
            assert all(values[k] is None for k in labels.RS_KEYS)
            assert priors["invalid"]["ROAD_STRUCTURE"] == "noise_rs_unresolved"
        else:
            domain = r["question_domain"]
            assert values[f"{domain}/{p2.INVALID_KEY}"] == "YES"
            assert all(values[k] is None for k in p2.event_keys_for_domain(domain))
            assert set(
                priors["invalid"][k] for k in p2.event_keys_for_domain(domain)
            ) == {"noise_event_domain_invalid"}
    assert seen["rs"] and seen["event"]


def test_noise_changes_contract_identity_and_is_recorded(tmp_path):
    clean = build_contract(dataset_args(tmp_path))
    args = dataset_args(tmp_path)
    args.prior_noise = 0.1
    noisy = build_contract(args)
    assert noisy["identity"] != clean["identity"]
    assert noisy["identity_payload"]["prior_noise"]["rate"] == 0.1
    assert clean["prior_labels"]["noise"] is None
    assert noisy["prior_labels"]["noise"]["model"] == labels.NOISE_MODEL
    # 只换先验来源/噪声时，decoder 侧身份必须仍然一致，才允许显式对照评测。
    assert decoder_identity(noisy) == decoder_identity(clean)
    assert require_contract(clean, noisy, allow_prior_source_change=True) == "prior_source_override"


def test_noise_seed_changes_contract_identity(tmp_path):
    """同样噪声率但不同 seed 会改写不同帧，必须是不同语言条件。"""
    first = dataset_args(tmp_path)
    first.prior_noise = 0.1
    first.seed = 7
    second = dataset_args(tmp_path)
    second.prior_noise = 0.1
    second.seed = 8
    c1 = build_contract(first)
    c2 = build_contract(second)
    assert c1["identity"] != c2["identity"]
    assert c1["identity_payload"]["prior_noise"]["seed"] == 7
    assert c2["identity_payload"]["prior_noise"]["seed"] == 8
    assert decoder_identity(c1) == decoder_identity(c2)
    assert require_contract(c1, c2, allow_prior_source_change=True) == "prior_source_override"


def test_noise_requires_dataset_priors_and_a_valid_rate(tmp_path):
    path = write_index(tmp_path / "labels", [row()])
    validate_args(parser().parse_args(
        ["--dataset-priors", "--prior-labels", str(path), "--prior-noise", "0.1"]))
    with pytest.raises(ValueError, match="perturbs dataset ground-truth priors"):
        validate_args(parser().parse_args(["--prior-noise", "0.1"]))
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        validate_args(parser().parse_args(
            ["--dataset-priors", "--prior-labels", str(path), "--prior-noise", "1.5"]))


def test_runtime_never_loads_a_lora_and_generates_once_without_review(tmp_path, monkeypatch):
    """analysis_review=False 时数据集先验每帧只有 1 次 base 生成 + 1 次最终 prefill。"""
    import json
    from types import SimpleNamespace
    import torch
    from PIL import Image
    from test_tensor_runtime import TinyBase
    import qwen3vl_local.action_prior.runtime as rt
    from qwen3vl_local.action_prior import prompts
    from qwen3vl_local.action_prior.runtime import PriorEngine

    def unreachable(*a, **kw):
        raise AssertionError("dataset priors must not query the phase LoRAs")

    monkeypatch.setattr(rt, "collect_priors", unreachable)
    prefills = []
    engine = SimpleNamespace(
        model=TinyBase(), _last_decode_state=None, _system_prompt_cache=None
    )
    engine.build_messages = lambda system, user, images: [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    engine.processor = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: str(messages)
    )
    engine.prepare_inputs = lambda text, images: {
        "input_ids": torch.arange(len(text)).reshape(1, -1)
    }

    def prefill(inputs):
        prefills.append(inputs)
        return SimpleNamespace(
            past_key_values=torch.ones(1, 4), rope_deltas=torch.tensor([[-3]])
        )

    engine.prefill = prefill
    index = labels.PriorLabelIndex(
        write_index(tmp_path / "labels", [row(rs="R4", target="UE6",
                                              domain=p2.JUNCTION_DOMAIN,
                                              invalid_domain=p2.ROAD_DOMAIN)])
    )
    draft = (
        "At a signalised local junction, a rule-violating crossing conflict is present; "
        "at 4 m/s the forward target and available crossing space guide near-term planning."
    )
    images = [Image.new("RGB", (3, 3)) for _ in range(4)]

    def run(analysis_review):
        engine_prior = PriorEngine(
            engine, {"identity": "dataset"}, labels=index, analysis_review=analysis_review
        )
        assert engine_prior.adapters is None
        generated = []

        def generate(system, prompt, images, **kwargs):
            generated.append(system)
            text = (
                json.dumps({key: True for key in prompts.REVIEW_KEYS})
                if system == prompts.REVIEW_SYSTEM
                else draft
            )
            return text, SimpleNamespace(decode_steps=[SimpleNamespace(is_eos=True)])

        engine_prior.generate_messages = generate
        prefills.clear()
        engine_prior.condition(images, "nav", "case", ("S", "R", 0))
        return engine_prior, generated

    engine_prior, generated = run(False)
    assert generated == [prompts.SYSTEM_PROMPT] and len(prefills) == 1
    audit = engine_prior.last_audit
    assert audit["analysis"] == draft and not audit["analysis_fallback"]
    assert audit["analysis_acceptance"] == "format_only"
    assert audit["analysis_review"] is None and audit["analysis_semantic_guarantee"] is False
    assert audit["calls"] == []
    assert audit["conditions"]["UE6"] == "YES" and audit["privileged_label_conditioning"]

    reviewed, generated = run(True)
    assert generated == [prompts.SYSTEM_PROMPT, prompts.REVIEW_SYSTEM]
    assert reviewed.last_audit["analysis_acceptance"] == "base_model_review"

    with pytest.raises(ValueError, match="never run a LoRA"):
        with engine_prior.mode("phase1"):
            pass
    with pytest.raises(ValueError, match="identity"):
        engine_prior.condition(images, "nav", "case")


def test_review_switch_is_part_of_the_conditioning_identity(tmp_path):
    reviewed = dataset_args(tmp_path)
    reviewed.analysis_review = True
    plain = dataset_args(tmp_path)
    plain.analysis_review = False
    assert build_contract(reviewed)["identity"] != build_contract(plain)["identity"]
    # 复核开关属于条件化协议，不在允许显式切换的先验来源字段里。
    assert decoder_identity(build_contract(reviewed)) != decoder_identity(build_contract(plain))


@pytest.mark.parametrize("analysis_review", [True, False])
def test_truncated_analysis_always_falls_back(tmp_path, monkeypatch, analysis_review):
    """恰好凑成三段格式的截断文本不能当完整分析，与复核开关无关。"""
    import json
    from types import SimpleNamespace
    import torch
    from PIL import Image
    from test_tensor_runtime import TinyBase
    import qwen3vl_local.action_prior.runtime as rt
    from qwen3vl_local.action_prior import prompts
    from qwen3vl_local.action_prior.runtime import PriorEngine

    monkeypatch.setattr(rt, "collect_priors", lambda *a, **k: None)
    engine = SimpleNamespace(
        model=TinyBase(), _last_decode_state=None, _system_prompt_cache=None
    )
    engine.build_messages = lambda system, user, images: [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    engine.processor = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: str(messages)
    )
    engine.prepare_inputs = lambda text, images: {
        "input_ids": torch.arange(len(text)).reshape(1, -1)
    }
    engine.prefill = lambda inputs: SimpleNamespace(
        past_key_values=torch.ones(1, 4), rope_deltas=torch.tensor([[-3]])
    )
    index = labels.PriorLabelIndex(write_index(tmp_path / "labels", [row()]))
    engine_prior = PriorEngine(
        engine, {"identity": "dataset"}, labels=index, analysis_review=analysis_review
    )
    cut = "The ordinary surface-road corridor continues toward the navigation target and the"
    assert prompts.analysis_format_valid(cut)

    def generate(system, prompt, images, **kwargs):
        text = (
            json.dumps({key: True for key in prompts.REVIEW_KEYS})
            if system == prompts.REVIEW_SYSTEM
            else cut
        )
        # 只有分析这一次被截断；复核若发生则正常结束。
        eos = system != prompts.SYSTEM_PROMPT
        return text, SimpleNamespace(decode_steps=[SimpleNamespace(is_eos=eos)])

    engine_prior.generate_messages = generate
    engine_prior.condition(
        [Image.new("RGB", (3, 3)) for _ in range(4)], "nav", "case", ("S", "R", 0)
    )
    audit = engine_prior.last_audit
    assert audit["analysis_truncated"] and audit["analysis_fallback"]
    assert audit["analysis_rejection"] == "generation_truncated"
    assert audit["analysis_acceptance"] == "fallback" and audit["analysis"] != cut
