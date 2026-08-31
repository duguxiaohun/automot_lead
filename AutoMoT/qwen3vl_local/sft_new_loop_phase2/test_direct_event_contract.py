#!/usr/bin/env python3
"""新 Phase2 单轮输入、highway RE 与 invalid 联合约束的轻量回归测试。"""

from __future__ import annotations

import unittest
from collections import Counter
import json
import pathlib
import tempfile

from qwen3vl_local.sft_new_loop_phase2 import build_dataset as dataset
from qwen3vl_local.sft_new_loop_phase2 import check_acceptance
from qwen3vl_local.sft_new_loop_phase2 import audit_eval_cases as event_audit
from qwen3vl_local.sft_new_loop_phase2 import eval as event_eval
from qwen3vl_local.sft_new_loop_phase2 import select_seed_checkpoint
from qwen3vl_local.sft_new_loop_phase2 import train as event_train
from qwen3vl_local.sft_new_loop_phase2 import visual_audit
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

    def test_prompt_v3_encodes_observed_rgb_error_boundaries(self) -> None:
        """逐帧错例归纳出的 newest/UE 互斥/低能见度边界必须留在生产合同中。"""

        answers = {key: False for key in EVENT_KEYS}
        answers[DOMAIN_ANSWER_KEYS[ROAD_DOMAIN]] = True
        answers[INVALID_KEY] = False
        spec = make_prompt_spec(variant="all_random_order", answers=answers, seed_key="rgb-v3")
        prompt = build_event_prompt(spec=spec)
        self.assertTrue(PROMPT_NAME.endswith("visual_v3"))
        self.assertIn("The newest frame decides whether an event is active", prompt)
        self.assertIn("same-lane lead vehicle that suddenly slows is UE1", prompt)
        self.assertIn("moving laterally into ego's future corridor is UE3", prompt)
        self.assertIn("both the invasion and its effect have ended", prompt)
        self.assertIn("do not mark the question set invalid merely because visibility is poor", prompt)
        self.assertIn("ego passes it is not lateral motion", prompt)
        self.assertIn("stationary crash or construction actors", prompt)
        self.assertIn("ego passing parked or queued vehicles", prompt)
        self.assertNotIn("parking-side or roadside vehicle advancing", prompt)

    def test_generation_checkpoint_score_guards_all_critical_slices(self) -> None:
        """正式最优点必须同时守住 UE3/UE6/INVALID/RE，不能单类换总分。"""

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

    def test_seed_selection_rejects_fallback_only_runs(self) -> None:
        """多 seed 选择只能读取通过全部 validation 门槛的 best_generation。"""

        base = {
            "prompt_name": PROMPT_NAME,
            "production_prompt_sha256": "hash",
            "history_rgb_mode": "2rgb_endpoints",
            "status": {},
            "fallback_generation": {"generation_exact_accuracy": 0.99},
        }
        fallback_only = [{**base, "run_root": "/tmp/a", "seed": 1, "best_generation": None}]
        with self.assertRaises(ValueError):
            select_seed_checkpoint.select_checkpoint(fallback_only, required_seeds=1)
        eligible = [
            {
                **base,
                "run_root": "/tmp/a",
                "seed": 1,
                "best_generation": {
                    "generation_guards_ok": True,
                    "generation_exact_accuracy": 0.81,
                    "generation": {"pattern/all_random_order_pattern_exact": 0.81},
                },
            },
            {
                **base,
                "run_root": "/tmp/b",
                "seed": 2,
                "best_generation": {
                    "generation_guards_ok": True,
                    "generation_exact_accuracy": 0.83,
                    "generation": {"pattern/all_random_order_pattern_exact": 0.83},
                },
            },
        ]
        selected = select_seed_checkpoint.select_checkpoint(eligible, required_seeds=2)
        self.assertEqual(selected["selected_seed"], 2)

    def test_unseen_acceptance_requires_every_frozen_floor(self) -> None:
        """总分达标但 UE6 退化时，unseen 验收必须失败。"""

        metrics = {
            "total_cases": 456,
            "exact_match_accuracy": 0.82,
            "variant_reports": {"all_random_order": {"format_valid_rate": 1.0}},
            "slice_reports": {"applicable_regular": {"exact_match_accuracy": 0.60}},
            "per_question": {
                "UE3": {"recall": 0.82},
                "UE6": {"recall": 0.75},
                "INVALID_EVENT_CONTEXT": {"recall": 0.82},
            },
        }
        args = type(
            "Args",
            (),
            {
                "metrics": "metrics.json",
                "min_overall_exact": 0.80,
                "min_format_valid_rate": 1.0,
                "min_ue3_recall": 0.80,
                "min_ue6_recall": 0.80,
                "min_invalid_recall": 0.80,
                "min_applicable_regular_exact": 0.50,
            },
        )()
        result = check_acceptance.evaluate_acceptance(metrics, args)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["passed"]["ue6_recall"])

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

    def test_highway_is_valid_road_regular(self) -> None:
        """R3/highway 是 ROAD_CORRIDOR all-NO hard negative，不是 invalid。"""

        row = dataset._make_row(
            base=_base("R3"),
            question_domain=dataset._native_question_domain("R3"),
            target_class="RE",
            invalid=False,
        )
        self.assertEqual(row["question_domain"], ROAD_DOMAIN)
        self.assertFalse(row["answers"][INVALID_KEY])
        self.assertFalse(any(row["answers"][key] for key in EVENT_KEYS))

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

    def test_full_pipeline_delegates_final_eval_and_bounded_bundle(self) -> None:
        """full pipeline 默认必须进入 eval.sh，且透传 30MB bundle 上限。"""

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
        self.assertIn('GENERATION_EVAL_MIN_UE3_TARGET_RECALL:-0.625', train_sh)
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
