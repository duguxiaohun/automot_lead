#!/usr/bin/env python3
"""新 Phase2 单轮输入、highway RE 与 invalid 联合约束的轻量回归测试。"""

from __future__ import annotations

import random
import unittest
from collections import Counter
import json
import pathlib
import tempfile
from types import SimpleNamespace

from qwen3vl_local.leadmot.config import (
    build_qwen_backbone_contract,
    require_qwen_backbone_match,
    resolve_qwen_adapter_dir,
)
from qwen3vl_local.sft_new_loop_phase2 import build_dataset as dataset
from qwen3vl_local.sft_new_loop_phase2 import audit_eval_cases as event_audit
from qwen3vl_local.sft_new_loop_phase2 import eval as event_eval
from qwen3vl_local.sft_new_loop_phase2 import sampling
from qwen3vl_local.sft_new_loop_phase2 import train as event_train
from qwen3vl_local.sft_new_loop_phase2 import visual_audit
from qwen3vl_local.sft_new_loop_phase2.highway_ue3_audit import HIGHWAY_UE3_SUBTYPE
from qwen3vl_local.sft_new_loop_phase2 import highway_ue3_audit
from qwen3vl_local.sft_new_loop_phase2.history_rgb import history_rgb_indices
from qwen3vl_local.sft_new_loop_phase2.prompts import (
    DOMAIN_ANSWER_KEYS,
    EVENT_KEYS,
    INVALID_KEY,
    JUNCTION_DOMAIN,
    QUESTION_DOMAINS,
    ROAD_DOMAIN,
    PROMPT_NAME,
    build_event_messages,
    build_event_prompt,
    build_event_target,
    event_prompt_sha256,
    make_prompt_spec,
    parse_event_answer_lines,
    parse_event_output,
)


def _base(rs: str = "R3") -> dict[str, object]:
    """构造不依赖真实磁盘的最小基础帧。"""

    return {
        "scenario": "SmokeScenario",
        "route_id": "SmokeRoute",
        "town": "TownSmoke",
        "split": "train",
        "frame_id": 7,
        "rs": rs,
        "primary_event": "R-E1",
        "event_codes": (),
        "target_event_class": "RE",
        "visual_label_risk": False,
        "visual_label_risk_reasons": [],
        "history_rgb_paths": ["0.jpg", "1.jpg", "2.jpg", "3.jpg"],
        "latest_rgb_path": "3.jpg",
    }


def _answers(target: str, domain: str) -> dict[str, bool]:
    """构造训练/eval balance smoke 使用的完整答案。"""

    out = {key: False for key in EVENT_KEYS}
    out[DOMAIN_ANSWER_KEYS[domain]] = True
    out[INVALID_KEY] = target == "INVALID"
    if target in EVENT_KEYS:
        out[target] = True
    return out


def _balance_rows(module: object) -> list[object]:
    """为 train/eval 两个同构 FrameRow 构造可循环抽样的六类样本。"""

    rows = []
    idx = 0
    for target, domain, rs, invalid_source in (
        ("UE1", ROAD_DOMAIN, "R1", ""),
        ("UE3", ROAD_DOMAIN, "R1", ""),
        ("UE5", ROAD_DOMAIN, "R2", ""),
        ("UE6", JUNCTION_DOMAIN, "R4", ""),
        ("RE", ROAD_DOMAIN, "R3", ""),
        ("RE", JUNCTION_DOMAIN, "R5", ""),
        ("INVALID", JUNCTION_DOMAIN, "R1", "source=UE1|true_rs=R1|question_domain=LOCAL_JUNCTION"),
        ("INVALID", JUNCTION_DOMAIN, "R2", "source=UE3|true_rs=R2|question_domain=LOCAL_JUNCTION"),
        ("INVALID", JUNCTION_DOMAIN, "R2", "source=UE5|true_rs=R2|question_domain=LOCAL_JUNCTION"),
        ("INVALID", ROAD_DOMAIN, "R4", "source=UE6|true_rs=R4|question_domain=ROAD_CORRIDOR"),
        ("INVALID", ROAD_DOMAIN, "R5", "source=UE6|true_rs=R5|question_domain=ROAD_CORRIDOR"),
        ("INVALID", JUNCTION_DOMAIN, "R3", "source=RE|true_rs=R3|question_domain=LOCAL_JUNCTION"),
    ):
        kwargs = dict(
            scenario="SmokeScenario",
            route_id=f"route-{idx}",
            town="TownSmoke",
            frame_id=idx,
            true_rs=rs,
            question_domain=domain,
            event=target,
            split="test",
            history_rgb_paths=["0.jpg", "1.jpg", "2.jpg", "3.jpg"],
            latest_rgb_path="3.jpg",
            answers=_answers(target, domain),
            invalid_source=invalid_source,
        )
        if module is event_eval:
            kwargs["idx"] = idx
        rows.append(module.FrameRow(**kwargs))
        idx += 1
    return rows


class DirectEventContractTest(unittest.TestCase):
    """守住用户要求的无伪 RS、highway valid 和 invalid all-NO 合同。"""





    def test_leadmot_qwen_backbone_contract_binds_real_adapter_bytes(self) -> None:
        """LeadMoT checkpoint 合同必须区分 base、正确 LoRA 和被替换的 LoRA。"""

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = pathlib.Path(raw_tmp)
            model = root / "model"
            adapter = root / "adapter"
            model.mkdir()
            adapter.mkdir()
            (model / "config.json").write_text('{"model_type":"qwen3_vl"}', encoding="utf-8")
            (adapter / "adapter_config.json").write_text(
                json.dumps({"target_modules": ["q_proj", "v_proj"]}), encoding="utf-8"
            )
            (adapter / "adapter_model.bin").write_bytes(b"adapter-v1")
            (adapter / "sft_new_loop_phase2_adapter_config.json").write_text(
                json.dumps(
                    {
                        "schema": "sft_new_loop_phase2_adapter_config",
                        "prompt_name": PROMPT_NAME,
                        "history_rgb_mode": "2rgb_endpoints",
                        "seed": 20260810,
                    }
                ),
                encoding="utf-8",
            )
            base_contract = build_qwen_backbone_contract(model)
            lora_contract = build_qwen_backbone_contract(model, adapter)
            self.assertFalse(base_contract["adapter_enabled"])
            self.assertTrue(lora_contract["adapter_enabled"])
            self.assertEqual(
                resolve_qwen_adapter_dir("auto", lora_contract),
                str(adapter.resolve()),
            )
            require_qwen_backbone_match(lora_contract, lora_contract, "ckpt.pt")
            with self.assertRaises(ValueError):
                require_qwen_backbone_match(lora_contract, base_contract, "ckpt.pt")
            require_qwen_backbone_match(None, base_contract, "legacy.pt")
            with self.assertRaises(ValueError):
                require_qwen_backbone_match(None, lora_contract, "legacy.pt")
            old_hash = lora_contract["adapter_sha256"]
            (adapter / "adapter_model.bin").write_bytes(b"adapter-v2")
            self.assertNotEqual(
                old_hash,
                build_qwen_backbone_contract(model, adapter)["adapter_sha256"],
            )



    def test_messages_have_no_synthetic_rs_turn(self) -> None:
        """模型输入只能是 system + 单个 image/text user turn。"""

        answers = {key: False for key in EVENT_KEYS}
        answers[DOMAIN_ANSWER_KEYS[ROAD_DOMAIN]] = True
        answers[INVALID_KEY] = False
        spec = make_prompt_spec(variant="all_random_order", answers=answers, seed_key="single-turn")
        messages = build_event_messages(images=["a", "b"], spec=spec)
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        rendered = repr(messages)
        for forbidden in ("CTX_R", "RS1:", "RS2:", "RS4:", "RS5:", "synthetic_rs_context"):
            self.assertNotIn(forbidden, rendered)

    def test_rgb_modes_are_all_four_or_first_latest_endpoints(self) -> None:
        """2RGB 固定选原 history 的首帧和最新帧。"""

        self.assertEqual(history_rgb_indices("4rgb"), (0, 1, 2, 3))
        self.assertEqual(history_rgb_indices("2rgb_endpoints"), (0, 3))

    def test_prompt_v5_encodes_observed_rgb_and_highway_cutin_boundaries(self) -> None:
        """逐帧错例归纳出的 newest/UE 互斥及高速 cut-in 边界必须留在生产合同中。"""

        answers = {key: False for key in EVENT_KEYS}
        answers[DOMAIN_ANSWER_KEYS[ROAD_DOMAIN]] = True
        answers[INVALID_KEY] = False
        spec = make_prompt_spec(variant="all_random_order", answers=answers, seed_key="rgb-v5")
        prompt = build_event_prompt(spec=spec)
        self.assertTrue(PROMPT_NAME.endswith("visual_v5_highway_ue3"))
        self.assertIn("The newest frame decides whether an event is active", prompt)
        self.assertIn("same-lane lead vehicle that suddenly slows is UE1", prompt)
        self.assertIn("moving laterally into ego's future corridor is UE3", prompt)
        self.assertIn("both the invasion and its effect have ended", prompt)
        self.assertIn("do not mark the question set invalid merely because visibility is poor", prompt)
        self.assertIn("ego passes it is not lateral motion", prompt)
        self.assertIn("stationary crash or construction actors", prompt)
        self.assertIn("ego passing parked or queued vehicles", prompt)
        self.assertNotIn("parking-side or roadside vehicle advancing", prompt)
        self.assertIn("including on a highway or ramp", prompt)
        self.assertIn("progressively crosses that divider into ego's current lane", prompt)
        self.assertIn("ordinary highway or local adjacent-lane traffic", prompt)

    def test_generation_checkpoint_score_guards_all_critical_slices(self) -> None:
        """非零门槛仍可守住各切片；默认零门槛不会让 UE3 阻断选择。"""

        guarded = event_train.generation_checkpoint_score(
            {
                "slice/ue3_target_recall": 0.625,
                "slice/ue6_target_recall": 0.875,
                "slice/invalid_exact": 0.85,
                "slice/applicable_regular_exact": 0.60,
                "exact_accuracy": 0.80,
                "pattern/all_random_order_pattern_exact": 0.80,
            },
            min_ue3_target_recall=0.625,
            min_ue6_target_recall=0.80,
            min_invalid_exact=0.80,
            min_applicable_regular_exact=0.50,
        )
        high_exact_but_regressed = event_train.generation_checkpoint_score(
            {
                "slice/ue3_target_recall": 0.70,
                "slice/ue6_target_recall": 0.70,
                "slice/invalid_exact": 0.90,
                "slice/applicable_regular_exact": 0.70,
                "exact_accuracy": 0.90,
                "pattern/all_random_order_pattern_exact": 0.90,
            },
            min_ue3_target_recall=0.625,
            min_ue6_target_recall=0.80,
            min_invalid_exact=0.80,
            min_applicable_regular_exact=0.50,
        )
        lower_failed = event_train.generation_checkpoint_score(
            {
                "slice/ue3_target_recall": 0.25,
                "slice/ue6_target_recall": 0.60,
                "slice/invalid_exact": 0.90,
                "slice/applicable_regular_exact": 0.70,
                "exact_accuracy": 0.95,
                "pattern/all_random_order_pattern_exact": 0.95,
            },
            min_ue3_target_recall=0.625,
            min_ue6_target_recall=0.80,
            min_invalid_exact=0.80,
            min_applicable_regular_exact=0.50,
        )
        self.assertGreater(guarded, high_exact_but_regressed)
        self.assertGreater(high_exact_but_regressed, lower_failed)
        report = event_train.generation_checkpoint_guards(
            {
                "slice/ue3_target_recall": 0.70,
                "slice/ue6_target_recall": 0.70,
                "slice/invalid_exact": 0.90,
                "slice/applicable_regular_exact": 0.70,
            },
            min_ue3_target_recall=0.625,
            min_ue6_target_recall=0.80,
            min_invalid_exact=0.80,
            min_applicable_regular_exact=0.50,
        )
        self.assertFalse(report["all_ok"])
        self.assertFalse(report["passed"]["ue6_target_recall"])
        ue3_disabled = event_train.generation_checkpoint_guards(
            {
                "slice/ue3_target_recall": 0.0,
                "slice/ue6_target_recall": 0.80,
                "slice/invalid_exact": 0.80,
                "slice/applicable_regular_exact": 0.50,
            },
            min_ue3_target_recall=0.0,
            min_ue6_target_recall=0.80,
            min_invalid_exact=0.80,
            min_applicable_regular_exact=0.50,
        )
        self.assertTrue(ue3_disabled["all_ok"])
        self.assertTrue(ue3_disabled["passed"]["ue3_target_recall"])

    def test_frozen_holdout_excludes_prior_eval_cases_exactly(self) -> None:
        """旧 dev cases 必须按稳定身份从 test split 精确排除。"""

        rows = _balance_rows(event_eval)
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = pathlib.Path(tmp)
            payloads = []
            for row in rows[:2]:
                payloads.append(
                    {
                        "scenario": row.scenario,
                        "route_id": row.route_id,
                        "frame_id": row.frame_id,
                        "question_domain": row.question_domain,
                        "event": row.event,
                        "invalid_source": row.invalid_source,
                    }
                )
            (case_dir / "cases_rank0.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in payloads), encoding="utf-8"
            )
            filtered, report = event_eval._exclude_prior_cases(
                rows, [case_dir], expected_excluded_cases=2
            )
            self.assertEqual(len(filtered), len(rows) - 2)
            self.assertEqual(report["matched"], 2)
            with self.assertRaises(ValueError):
                event_eval._exclude_prior_cases(rows, [case_dir], expected_excluded_cases=3)



    def test_explicit_event_review_is_visual_risk(self) -> None:
        """EVENT 标注自身要求 RGB 复核时，默认 clean pool 不能继续静默纳入。"""

        annotation = {
            "frame_rs_annotation": {"review_reasons": []},
            "frame_event_annotation": {
                "review_required": True,
                "review_reasons": ["event_boundary_requires_rgb_confirmation"],
            },
            "event_evidence": {
                "review_required": True,
                "review_reasons": ["event_boundary_requires_rgb_confirmation"],
            },
        }
        risk, reasons = visual_audit.frame_visual_risk(annotation)
        self.assertTrue(risk)
        self.assertEqual(reasons, ["event:event_boundary_requires_rgb_confirmation"])

    def test_unreviewed_highway_is_valid_road_regular(self) -> None:
        """未被 RGB 正例覆盖的 R3/highway 仍是 valid all-NO，而不是 invalid。"""

        row = dataset._make_row(
            base=_base("R3"),
            question_domain=dataset._native_question_domain("R3"),
            target_class="RE",
            invalid=False,
        )
        self.assertEqual(row["question_domain"], ROAD_DOMAIN)
        self.assertFalse(row["answers"][INVALID_KEY])
        self.assertFalse(any(row["answers"][key] for key in EVENT_KEYS))

    def test_rgb_reviewed_highway_cutin_remains_ue3(self) -> None:
        """高速 cut-in 是 UE3 审计子型，不新增互斥类别。"""

        base = _base("R3")
        base["ue3_subtype"] = HIGHWAY_UE3_SUBTYPE
        base["event_label_source"] = "manual_highway_ue3_rgb_decision_v1"
        row = dataset._make_row(
            base=base,
            question_domain=ROAD_DOMAIN,
            target_class="UE3",
            invalid=False,
        )
        self.assertEqual(row["target_event_class"], "UE3")
        self.assertEqual(row["ue3_subtype"], HIGHWAY_UE3_SUBTYPE)
        self.assertTrue(row["answers"]["UE3"])
        self.assertFalse(row["answers"][INVALID_KEY])

    def test_explicit_ue3_interrupted_overlay_is_not_lost_by_rs_gate(self) -> None:
        """R4/R5 overlay 中的显式 U-E3 仍是 UE3，并通过道路问组监督。"""

        base = _base("R4")
        self.assertEqual(dataset._target_class("R4", ["R-E4", "U-E3"]), "UE3")
        self.assertEqual(dataset._target_class("R4", ["U-E1", "U-E3"]), "UE3")
        self.assertEqual(dataset._target_question_domain(base, "UE3"), ROAD_DOMAIN)
        base["target_event_class"] = "UE3"
        self.assertFalse(dataset._can_construct_invalid_from(base))
        base["target_event_class"] = "RE"
        self.assertTrue(dataset._can_construct_invalid_from(base))
        self.assertEqual(dataset._target_class("R4", ["R-E4"]), "RE")

    def test_highway_ue3_decisions_are_explicit_and_split_complete(self) -> None:
        """人工清单必须同时覆盖 train/val/test，且只展开显式 YES span。"""

        path = pathlib.Path(__file__).with_name("highway_ue3_rgb_decisions_v1.jsonl")
        positives, report = highway_ue3_audit.load_highway_ue3_decisions(path)
        self.assertEqual(len(positives), report["positive_override_frames"])
        self.assertGreater(report["split_frame_counts"]["train/YES"], 0)
        self.assertGreater(report["split_frame_counts"]["val/YES"], 0)
        self.assertGreater(report["split_frame_counts"]["test/YES"], 0)
        self.assertGreater(report["frame_counts"]["NO"], report["frame_counts"]["YES"])
        self.assertEqual(
            set(highway_ue3_audit.SOURCE_UE3_AUDIT_SNAPSHOT["frames_by_scenario"]),
            {"DynamicObjectCrossing", "ParkingCutIn", "StaticCutIn"},
        )
        self.assertEqual(
            sum(highway_ue3_audit.SOURCE_UE3_AUDIT_SNAPSHOT["frames_by_scenario"].values()),
            1351,
        )

    def test_eval_reserves_highway_ue3_inside_existing_ue3_class(self) -> None:
        """高速子型只占 UE3 内部配额，不产生第七个训练目标类。"""

        rows = _balance_rows(event_eval)
        local_ue3 = next(row for row in rows if event_eval._target_class(row) == "UE3")
        rows.append(
            event_eval.FrameRow(
                idx=999,
                scenario="HighwayCutIn",
                route_id="highway-positive",
                town="Town13",
                frame_id=100,
                true_rs="R3",
                question_domain=ROAD_DOMAIN,
                event="R-E1",
                split="test",
                history_rgb_paths=["0.jpg", "1.jpg", "2.jpg", "3.jpg"],
                latest_rgb_path="3.jpg",
                answers=dict(local_ue3.answers),
                ue3_subtype=HIGHWAY_UE3_SUBTYPE,
            )
        )
        cases = event_eval._balanced_cases(
            rows,
            cases_per_bin=4,
            seed=7,
            highway_ue3_fraction=0.25,
        )
        ue3 = [item for item in cases if event_eval._target_class(item.row) == "UE3"]
        self.assertEqual(len(ue3), 4)
        self.assertEqual(sum(item.row.ue3_subtype == HIGHWAY_UE3_SUBTYPE for item in ue3), 1)
        self.assertTrue(all(item.balance_key.endswith("/class/UE3") for item in ue3))

    def test_cross_domain_invalid_is_all_no(self) -> None:
        """跨问题域样本只打开 invalid，不允许同时打开 UE。"""

        for true_rs, expected_domain in (("R1", JUNCTION_DOMAIN), ("R4", ROAD_DOMAIN)):
            row = dataset._make_row(
                base=_base(true_rs),
                question_domain=dataset._mismatched_question_domain(true_rs),
                target_class="INVALID",
                invalid=True,
            )
            self.assertEqual(row["question_domain"], expected_domain)
            self.assertTrue(row["answers"][INVALID_KEY])
            self.assertFalse(any(row["answers"][key] for key in EVENT_KEYS))

    def test_target_round_trip_for_both_domains(self) -> None:
        """两类问题域的严格 target 都能被 parser 无损恢复。"""

        for domain in QUESTION_DOMAINS:
            answers = {key: False for key in EVENT_KEYS}
            answers[DOMAIN_ANSWER_KEYS[domain]] = True
            answers[INVALID_KEY] = False
            spec = make_prompt_spec(variant="all_random_order", answers=answers, seed_key=domain)
            parsed = parse_event_output(build_event_target(spec), spec=spec)
            self.assertTrue(all(value is not None for value in parsed.values()))

    def test_parser_rejects_reordered_or_extra_output(self) -> None:
        """行乱序和额外解释都不能再被算作 format-valid/exact。"""

        answers = {key: False for key in EVENT_KEYS}
        answers[DOMAIN_ANSWER_KEYS[ROAD_DOMAIN]] = True
        answers["UE3"] = True
        answers[INVALID_KEY] = False
        spec = make_prompt_spec(variant="all_random_order", answers=answers, seed_key="strict-format")
        target_lines = build_event_target(spec).splitlines()
        for malformed in (
            "\n".join(reversed(target_lines)),
            "\n".join(target_lines) + "\nextra text",
            "\n".join(target_lines[:-1]),
            "\n".join([target_lines[0], target_lines[0], *target_lines[2:]]),
        ):
            parsed = parse_event_output(malformed, spec=spec)
            self.assertTrue(all(value is None for value in parsed.values()))

    def test_audit_parser_requires_ordered_evidence_only(self) -> None:
        """audit 模式只额外允许同顺序的短 evidence 行。"""

        answers = {key: False for key in EVENT_KEYS}
        answers[DOMAIN_ANSWER_KEYS[JUNCTION_DOMAIN]] = True
        answers[INVALID_KEY] = False
        spec = make_prompt_spec(variant="all_random_order", answers=answers, seed_key="strict-audit")
        target = build_event_target(spec)
        evidence = "\n".join(f"EVIDENCE_{key}: visible junction cue" for key in spec.output_keys)
        parsed = parse_event_output(f"{target}\n{evidence}", spec=spec, audit=True)
        self.assertTrue(all(value is not None for value in parsed.values()))
        malformed = f"{target}\n{evidence}\nextra"
        self.assertTrue(all(value is None for value in parse_event_output(malformed, spec=spec, audit=True).values()))

        blank_evidence = "\n".join(
            f"EVIDENCE_{key}: {'visible junction cue' if index else ''}"
            for index, key in enumerate(spec.output_keys)
        )
        raw = f"{target}\n{blank_evidence}"
        self.assertTrue(
            all(value is None for value in parse_event_output(raw, spec=spec, audit=True).values())
        )
        expected_answers = {q.output_key: bool(q.answer) for q in spec.questions}
        self.assertEqual(parse_event_answer_lines(raw, spec=spec), expected_answers)
        audit_prompt = build_event_prompt(spec=spec, audit=True)
        self.assertIn("Every EVIDENCE line is mandatory", audit_prompt)
        self.assertIn("never leave text after the colon blank", audit_prompt)

    def test_train_and_eval_balancers_preserve_required_margins(self) -> None:
        """训练和独立评测都守住 UE 1:1:1:1 与 RE highway 25%。"""

        train_work = event_train._balanced_work(
            _balance_rows(event_train),
            target_per_bin=8,
            seed=1,
            regular_multiplier=1.0,
            invalid_multiplier=1.0,
            highway_regular_fraction=0.25,
        )
        eval_work = event_eval._balanced_cases(
            _balance_rows(event_eval),
            cases_per_bin=8,
            seed=1,
            highway_regular_fraction=0.25,
        )
        for work, classify in (
            (train_work, event_train._target_class),
            (eval_work, event_eval._target_class),
        ):
            counts = Counter(classify(item.row) for item in work)
            self.assertEqual([counts[key] for key in EVENT_KEYS], [8, 8, 8, 8])
            self.assertEqual(counts["RE"], 8)
            highway_re = sum(classify(item.row) == "RE" and item.row.true_rs == "R3" for item in work)
            self.assertEqual(highway_re, 2)
            invalid_report = (
                event_train.invalid_subgroup_report(work)
                if classify is event_train._target_class
                else event_eval.invalid_subgroup_report(work)
            )
            self.assertTrue(invalid_report["guards"]["source_class_max_deviation_le_1"])
            self.assertTrue(
                invalid_report["guards"]["joint_signature_within_source_max_deviation_le_1"]
            )

    def test_route_diverse_sampler_rotates_routes_before_consecutive_frames(self) -> None:
        """小 validation 桶必须先覆盖不同 route，不能先抽同一 span 的连续帧。"""

        items = []
        for route_id, count in (("long-span", 9), ("route-b", 2), ("route-c", 1)):
            for frame_id in range(count):
                row = SimpleNamespace(
                    scenario="DynamicObjectCrossing",
                    route_id=route_id,
                    frame_id=frame_id,
                )
                items.append(SimpleNamespace(row=row))
        selected = sampling.route_diverse_sample(
            items,
            target=5,
            rng=random.Random(7),
        )
        first_round = {(item.row.scenario, item.row.route_id) for item in selected[:3]}
        counts = Counter(item.row.route_id for item in selected)
        self.assertEqual(len(first_round), 3)
        self.assertEqual(max(counts.values()), 2)
        self.assertEqual(selected, sampling.route_diverse_sample(items, target=5, rng=random.Random(7)))
        self.assertEqual(sampling.route_diversity_report(selected)["unique_routes"], 3)

    def test_route_diverse_sampler_supports_dataset_dict_rows(self) -> None:
        """dataset 构建阶段的 dict row 也必须先覆盖不同 route。"""

        items = [
            {"scenario": "StaticCutIn", "route_id": "long", "frame_id": frame_id}
            for frame_id in range(8)
        ]
        items.extend(
            [
                {"scenario": "StaticCutIn", "route_id": "short-a", "frame_id": 1},
                {"scenario": "DynamicObjectCrossing", "route_id": "short-b", "frame_id": 1},
            ]
        )
        selected = dataset._sample_bucket(items, target=3, rng=random.Random(9))
        self.assertEqual(len({(row["scenario"], row["route_id"]) for row in selected}), 3)
        report = sampling.route_diversity_report(selected)
        self.assertEqual(report["unique_routes"], 3)
        self.assertEqual(report["max_cases_per_route"], 1)


    def test_dataset_val_test_sampler_keeps_legacy_frame_shuffle(self) -> None:
        """关闭 route-diverse 时必须保持旧 val/test shuffle+truncate 身份。"""

        items = [
            {"scenario": "StaticCutIn", "route_id": f"route-{index // 3}", "frame_id": index}
            for index in range(9)
        ]
        expected = list(items)
        random.Random(17).shuffle(expected)
        selected = dataset._sample_bucket(
            items,
            target=5,
            rng=random.Random(17),
            route_diverse=False,
        )
        self.assertEqual(selected, expected[:5])




    def test_generation_eval_route_diversity_is_opt_in_to_shared_balancer(self) -> None:
        """generation validation 开关启用后，同类先覆盖 route；普通训练采样接口仍兼容。"""

        rows = _balance_rows(event_train)
        ue3_template = next(row for row in rows if event_train._target_class(row) == "UE3")
        rows = [row for row in rows if event_train._target_class(row) != "UE3"]
        for route_id, count in (("ue3-long", 8), ("ue3-b", 2), ("ue3-c", 1)):
            for frame_id in range(count):
                rows.append(
                    event_train.FrameRow(
                        **{
                            **ue3_template.__dict__,
                            "route_id": route_id,
                            "frame_id": frame_id,
                        }
                    )
                )
        work = event_train._balanced_work(
            rows,
            target_per_bin=3,
            seed=11,
            regular_multiplier=1.0,
            invalid_multiplier=1.0,
            highway_regular_fraction=0.0,
            route_diverse=True,
        )
        ue3_routes = {
            item.row.route_id for item in work if event_train._target_class(item.row) == "UE3"
        }
        self.assertEqual(ue3_routes, {"ue3-long", "ue3-b", "ue3-c"})


    def test_train_launcher_enables_route_diversity_by_default(self) -> None:
        """正式 launcher 只启用所有类别共用的 route-diverse。"""

        train_sh = (pathlib.Path(__file__).with_name("train.sh")).read_text(encoding="utf-8")
        self.assertIn('TRAIN_ROUTE_DIVERSE:-1', train_sh)
        self.assertIn('COMMON_ARGS+=(--train-route-diverse)', train_sh)
        self.assertNotIn('TRAIN_UE3_ROUTE_BALANCED', train_sh)
        self.assertNotIn('--train-ue3-route-balanced', train_sh)
        self.assertNotIn('MAX_TRAIN_UE3_FRAME_REPEAT', train_sh)
        self.assertIn('SAVE_FINAL:-1', train_sh)
        self.assertIn('COMMON_ARGS+=(--no-save-final)', train_sh)

    def test_eval_run_dir_prefers_best_then_final_without_generation_block(self) -> None:
        """run 根目录优先验证集选优 best；缺失时才回退 final，不因缺 best 阻断。"""

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp)
            for slot in ("final", "best_generation", "fallback_generation"):
                slot_dir = run_dir / slot
                slot_dir.mkdir()
                (slot_dir / "sft_new_loop_phase2_adapter_config.json").write_text(
                    "{}", encoding="utf-8"
                )
            resolved, source = event_eval._resolve_adapter_dir(run_dir)
            self.assertEqual(resolved, run_dir / "best_generation")
            self.assertEqual(source, "run_dir_best_generation")
            (run_dir / "best_generation" / "sft_new_loop_phase2_adapter_config.json").unlink()
            resolved, source = event_eval._resolve_adapter_dir(run_dir)
            self.assertEqual(resolved, run_dir / "final")
            self.assertEqual(source, "run_dir_final")

    def test_default_pipeline_adapter_prefers_valid_best_but_keeps_final_fallback(self) -> None:
        """selection status 不能再把实际 best 错记为 final。"""

        self.assertEqual(
            event_train.default_pipeline_adapter(
                best_generation_available=True,
                final_available=True,
                fallback_generation_available=True,
            ),
            "best_generation",
        )
        self.assertEqual(
            event_train.default_pipeline_adapter(
                best_generation_available=False,
                final_available=True,
                fallback_generation_available=True,
            ),
            "final",
        )
        self.assertEqual(
            event_train.default_pipeline_adapter(
                best_generation_available=False,
                final_available=False,
                fallback_generation_available=True,
            ),
            "fallback_generation",
        )



    def test_balancers_reject_missing_class_and_zero_uses_smallest_bucket(self) -> None:
        """截断索引缺桶必须失败；train target=0 必须真的按最小桶均衡。"""

        train_rows = _balance_rows(event_train)
        eval_rows = _balance_rows(event_eval)
        train_without_invalid = [row for row in train_rows if event_train._target_class(row) != "INVALID"]
        eval_without_invalid = [row for row in eval_rows if event_eval._target_class(row) != "INVALID"]
        with self.assertRaisesRegex(ValueError, r"missing=\['INVALID'\]"):
            event_train._balanced_work(train_without_invalid, target_per_bin=4, seed=1)
        with self.assertRaisesRegex(ValueError, r"missing=\['INVALID'\]"):
            event_eval._balanced_cases(eval_without_invalid, cases_per_bin=4, seed=1)

        work = event_train._balanced_work(
            [train_rows[0], train_rows[0], *train_rows],
            target_per_bin=0,
            seed=2,
            regular_multiplier=1.0,
            invalid_multiplier=1.0,
            highway_regular_fraction=0.0,
        )
        counts = Counter(event_train._target_class(item.row) for item in work)
        self.assertEqual({key: counts[key] for key in (*EVENT_KEYS, "RE", "INVALID")}, {
            key: 1 for key in (*EVENT_KEYS, "RE", "INVALID")
        })

    def test_effective_target_uses_only_six_required_classes(self) -> None:
        """target=0 的日志基数不能被 highway 等旁路统计中的稀疏桶污染。"""

        rows = _balance_rows(event_train)
        by_class = {
            key: [row for row in rows if event_train._target_class(row) == key]
            for key in (*EVENT_KEYS, "RE", "INVALID")
        }
        expanded = []
        for key in EVENT_KEYS:
            expanded.extend([by_class[key][0]] * 3)
        expanded.extend([by_class["RE"][0], by_class["RE"][1], by_class["RE"][1]])
        expanded.extend(by_class["INVALID"])
        self.assertEqual(event_train._raw_focus_bin_counts(expanded)["regular_kind/highway_r3"], 1)
        self.assertEqual(event_train._effective_class_target(expanded, 0), 3)

    def test_invalid_signature_is_required_and_row_consistent(self) -> None:
        """INVALID 行缺签名或签名与行字段不一致时必须硬失败。"""

        rows = _balance_rows(event_train)
        invalid_row = next(row for row in rows if event_train._target_class(row) == "INVALID")
        invalid_row.invalid_source = ""
        with self.assertRaisesRegex(ValueError, "invalid_source"):
            event_train._balanced_work(rows, target_per_bin=5, seed=3)

        rows = _balance_rows(event_eval)
        invalid_row = next(row for row in rows if event_eval._target_class(row) == "INVALID")
        invalid_row.invalid_source = invalid_row.invalid_source.replace("true_rs=R1", "true_rs=R2")
        with self.assertRaisesRegex(ValueError, "true_rs mismatch"):
            event_eval._balanced_cases(rows, cases_per_bin=5, seed=3)

    def test_train_and_eval_readers_preserve_invalid_source(self) -> None:
        """frame index 的 invalid_source 不能在任一运行入口读取时丢失。"""

        signature = "source=UE6|true_rs=R5|question_domain=ROAD_CORRIDOR"
        payload = {
            "dataset_name": "sft_new_loop_phase2_direct_event",
            "scenario": "SmokeScenario",
            "route_id": "SmokeRoute",
            "town": "TownSmoke",
            "frame_id": 1,
            "true_rs": "R5",
            "question_domain": ROAD_DOMAIN,
            "event": "INVALID",
            "split": "test",
            "history_rgb_paths": ["0.jpg", "1.jpg", "2.jpg", "3.jpg"],
            "latest_rgb_path": "3.jpg",
            "answers": _answers("INVALID", ROAD_DOMAIN),
            "invalid_source": signature,
        }
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = pathlib.Path(raw_tmp) / "frame_index.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            self.assertEqual(event_train._read_rows(path, "test")[0].invalid_source, signature)
            self.assertEqual(event_eval._read_rows(path, "test")[0].invalid_source, signature)

    def test_eval_zero_keeps_all_rows_but_still_validates_invalid(self) -> None:
        """cases_per_bin=0 是全量模式，不得绕过 INVALID 签名和覆盖守卫。"""

        rows = _balance_rows(event_eval)
        work = event_eval._balanced_cases(rows, cases_per_bin=0, seed=4)
        self.assertEqual(len(work), len(rows))
        self.assertEqual(
            sum(event_eval._target_class(item.row) == "INVALID" for item in work),
            sum(event_eval._target_class(row) == "INVALID" for row in rows),
        )
        invalid_row = next(row for row in rows if event_eval._target_class(row) == "INVALID")
        invalid_row.invalid_source = ""
        with self.assertRaisesRegex(ValueError, "invalid_source"):
            event_eval._balanced_cases(rows, cases_per_bin=0, seed=4)

    def test_audit_summary_row_and_note_include_invalid_subgroups(self) -> None:
        """抽样审计无需回到原始 eval case 才能看懂 INVALID 来源。"""

        signature = "source=UE6|true_rs=R5|question_domain=ROAD_CORRIDOR"
        payload = {
            "case_index": 9,
            "scenario": "SmokeScenario",
            "town": "TownSmoke",
            "route_id": "SmokeRoute",
            "frame_id": 3,
            "true_rs": "R5",
            "question_domain": ROAD_DOMAIN,
            "event": "INVALID",
            "invalid_source": signature,
            "invalid_subgroups": {
                "source_class": "UE6",
                "true_rs": "R5",
                "wrong_question_domain": ROAD_DOMAIN,
                "joint_signature": signature,
            },
            "gt": {INVALID_KEY: "YES"},
            "parsed": {INVALID_KEY: "NO"},
            "raw_output": "INVALID_EVENT_CONTEXT: NO",
            "history_rgb_selected_indices": [],
            "history_rgb_paths_used": [],
        }
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = pathlib.Path(raw_tmp)
            row = event_audit._copy_case(
                payload,
                target="invalid_context_fn",
                index=0,
                output_dir=root / "audit",
                data_root=root,
            )
            self.assertEqual(row["invalid_source"], signature)
            self.assertEqual(row["invalid_subgroups"]["source_class"], "UE6")
            note = pathlib.Path(row["case_dir"]) / "audit_note.md"
            note_text = note.read_text(encoding="utf-8")
            self.assertIn(signature, note_text)
            self.assertIn('"source_class": "UE6"', note_text)
            self.assertIn("defining actor visible in newest frame", note_text)
            self.assertIn("error owner: `MODEL / LABEL_OR_BOUNDARY / BOTH / FORMAT`", note_text)

    def test_full_pipeline_delegates_selected_eval_and_bounded_bundle(self) -> None:
        """full pipeline 默认评测 best、缺失回退 final，并透传 30MB bundle 上限。"""

        pipeline_path = pathlib.Path(__file__).with_name("run_full_pipeline.sh")
        pipeline = pipeline_path.read_text(encoding="utf-8")
        eval_sh = pipeline_path.with_name("eval.sh").read_text(encoding="utf-8")
        train_sh = pipeline_path.with_name("train.sh").read_text(encoding="utf-8")
        self.assertIn('RUN_EVAL_SH="${RUN_EVAL_SH:-1}"', pipeline)
        self.assertIn('DDP_GPU_COUNT="${DDP_GPU_COUNT:-${NPROC_PER_NODE:-4}}"', pipeline)
        self.assertIn("HISTORY_RGB_MODES=2rgb_endpoints", pipeline)
        self.assertIn("bash qwen3vl_local/sft_new_loop_phase2/eval.sh", pipeline)
        self.assertIn('BUNDLE_MAX_MB="${BUNDLE_MAX_MB:-30}"', pipeline)
        final_eval_block = pipeline.split('if [[ "${RUN_EVAL_SH}" == "1" ]]', 1)[1]
        self.assertNotIn('HISTORY_RGB_MODE="${HISTORY_RGB_MODE}"', final_eval_block)
        self.assertIn('MODE="${1:-${MODE:-ddp}}"', train_sh)
        self.assertIn('GENERATION_EVAL_BALANCE_COUNT:-32', train_sh)
        self.assertIn('GENERATION_EVAL_MIN_UE3_TARGET_RECALL:-0.0', train_sh)
        self.assertIn('for slot in best_generation final fallback_generation', pipeline)
        self.assertIn('sft_new_loop_phase2_adapter_config.json', pipeline)
        self.assertNotIn("ALLOW_FALLBACK_ADAPTER", pipeline)
        self.assertIn('"${input}/best_generation" "${input}/final"', eval_sh)
        matrix = pipeline_path.with_name("run_rgb_mode_matrix.sh").read_text(encoding="utf-8")
        self.assertIn('for slot in best_generation final fallback_generation', matrix)
        self.assertNotIn('best_val', matrix)
        self.assertNotIn("REQUESTED_HISTORY_RGB_MODE", eval_sh)
        self.assertIn('BASE_HISTORY_RGB_MODE="$(read_adapter_history_rgb_mode', eval_sh)
        self.assertIn("history_rgb_selected_indices", eval_sh)
        self.assertIn('${PHASE_NAME}_${TIMESTAMP}_${BASE_HISTORY_RGB_MODE}_audit_bundle', eval_sh)
        self.assertIn("validated_expected_files", eval_sh)

    def test_adapter_validation_rejects_prompt_or_base_model_mismatch(self) -> None:
        """旧 prompt adapter 和错误 base model 都必须在加载权重前硬失败。"""

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = pathlib.Path(raw_tmp)
            adapter_dir = root / "adapter"
            model_dir = root / "model"
            wrong_model_dir = root / "wrong_model"
            adapter_dir.mkdir()
            model_dir.mkdir()
            wrong_model_dir.mkdir()
            cfg = {
                "route": "sft_new_loop_phase2_direct_event",
                "dataset_name": "sft_new_loop_phase2_direct_event",
                "prompt_name": PROMPT_NAME,
                "history_rgb_mode": "4rgb",
                "production_prompt_sha256": "stale-hash",
                "base_model_dir": str(model_dir),
            }
            cfg_path = adapter_dir / "sft_new_loop_phase2_adapter_config.json"
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "production_prompt_sha256 mismatch"):
                event_eval._validate_event_adapter(adapter_dir, model_dir)

            cfg["production_prompt_sha256"] = event_prompt_sha256(
                audit=False,
                history_rgb_mode="4rgb",
            )
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "base_model_dir mismatch"):
                event_eval._validate_event_adapter(adapter_dir, wrong_model_dir)
            self.assertEqual(event_eval._validate_event_adapter(adapter_dir, model_dir)["prompt_name"], PROMPT_NAME)

    def test_actual_dataset_pairs_must_exist_in_rgb_review_coverage(self) -> None:
        """coverage manifest 不能只自洽，还必须覆盖本次实际扫描到的 pair。"""

        coverage = {"ScenarioA": {"Town01": {"completed_routes": 1}}}
        dataset._assert_actual_review_coverage({("ScenarioA", "Town01")}, coverage)
        with self.assertRaisesRegex(ValueError, "ScenarioB/Town02"):
            dataset._assert_actual_review_coverage(
                {("ScenarioA", "Town01"), ("ScenarioB", "Town02")},
                coverage,
            )


if __name__ == "__main__":
    unittest.main()
