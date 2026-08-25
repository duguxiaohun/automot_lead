#!/usr/bin/env python3
"""Regression tests for the RGB-error-driven fused prompt/loss refinements."""

from __future__ import annotations

import pathlib
import unittest

from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices
from qwen3vl_local.sft_new_loop_phase1.prompts import (
    ANSWER_KEYS,
    PHASE1_ANSWER_KEYS,
    build_phase1_prompt,
    build_phase1_target,
    make_prompt_spec,
    parse_phase1_audit_output,
    parse_phase1_output,
    phase2_output_keys,
    prompt_spec_to_json,
)
from qwen3vl_local.sft_new_loop_phase1.train import (
    FrameRow,
    WorkItem,
    _line_value_span,
    _semantic_class_weights,
    _semantic_output_keys,
    _target_token_weights,
)


class _CharTokenizer:
    """Tiny offset-preserving tokenizer used without loading Qwen weights."""

    def __call__(self, text: str, **_: object) -> dict[str, object]:
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


class _Bundle:
    tokenizer = _CharTokenizer()


def _answers() -> dict[str, bool]:
    return {key: key in {"VULNERABLE", "RS5"} for key in ANSWER_KEYS}


def _row() -> FrameRow:
    return FrameRow(
        scenario="unit",
        route_id="route",
        town="Town01",
        frame_id=3,
        rs="R5",
        event="U-E1",
        split="train",
        history_rgb_paths=[],
        latest_rgb_path="",
        answers=_answers(),
    )


class RefinedContractTest(unittest.TestCase):
    def test_rgb_modes_are_all_four_or_first_latest_endpoints(self) -> None:
        """2RGB 固定是首帧+最新帧，不允许退化为任意相邻两帧。"""

        self.assertEqual(history_rgb_indices("4rgb"), (0, 1, 2, 3))
        self.assertEqual(history_rgb_indices("2rgb_endpoints"), (0, 3))

    def test_pipeline_and_eval_own_rgb_mode_contract(self) -> None:
        """训练入口展示 4/2RGB；最终 eval 只能相信 checkpoint 配置。"""

        root = pathlib.Path(__file__).parent
        pipeline = (root / "run_full_pipeline.sh").read_text(encoding="utf-8")
        eval_sh = (root / "eval.sh").read_text(encoding="utf-8")
        self.assertIn('HISTORY_RGB_MODES="${HISTORY_RGB_MODES:-4rgb}"', pipeline)
        self.assertIn('DDP_GPU_COUNT="${DDP_GPU_COUNT:-${NPROC_PER_NODE:-4}}"', pipeline)
        self.assertIn("HISTORY_RGB_MODES=2rgb_endpoints", pipeline)
        self.assertIn('RUN_EVAL_SH="${RUN_EVAL_SH:-1}"', pipeline)
        final_eval_block = pipeline.split('if [[ "${RUN_EVAL_SH}" == "1" ]]', 1)[1]
        self.assertIn("bash qwen3vl_local/sft_new_loop_phase1/eval.sh", final_eval_block)
        self.assertNotIn('HISTORY_RGB_MODE="${HISTORY_RGB_MODE}"', final_eval_block)
        self.assertNotIn("REQUESTED_HISTORY_RGB_MODE", eval_sh)
        self.assertIn('BASE_HISTORY_RGB_MODE="$(read_adapter_history_rgb_mode', eval_sh)
        self.assertIn("history_rgb_selected_indices", eval_sh)
        self.assertIn('${PHASE_NAME}_${TIMESTAMP}_${BASE_HISTORY_RGB_MODE}_audit_bundle', eval_sh)
        self.assertIn("validated_expected_files", eval_sh)

    def test_phase1_order_is_randomized_reproducibly_for_train_and_eval_specs(self) -> None:
        specs = [
            make_prompt_spec(
                variant="all_random_order",
                answers=_answers(),
                seed_key=f"case:{index}",
                focus="RS1",
            )
            for index in range(512)
        ]
        orders = {spec.phase1_output_keys for spec in specs}
        self.assertEqual(len(orders), 24)
        self.assertEqual(
            specs[17].phase1_output_keys,
            make_prompt_spec(
                variant="all_random_order",
                answers=_answers(),
                seed_key="case:17",
                focus="RS1",
            ).phase1_output_keys,
        )
        for spec in specs:
            self.assertEqual(set(spec.phase1_output_keys), set(PHASE1_ANSWER_KEYS))
            self.assertEqual(spec.output_keys[:4], spec.phase1_output_keys)
            self.assertEqual(spec.output_keys[4:], phase2_output_keys(spec))
            self.assertEqual(prompt_spec_to_json(spec)["phase1_output_keys"], list(spec.phase1_output_keys))
            target_keys = tuple(line.split(":", 1)[0] for line in build_phase1_target(_answers(), spec=spec).splitlines())
            self.assertEqual(target_keys, spec.output_keys)

    def test_audit_evidence_yes_no_is_not_parsed_as_unknown_answer(self) -> None:
        spec = make_prompt_spec(
            variant="subset_random",
            answers=_answers(),
            seed_key="parser",
            focus="RS5",
            subset_count=1,
        )
        target = build_phase1_target(_answers(), spec=spec)
        # Exactly ``NO`` reproduces the old ambiguity: the generic answer
        # regex also matched this line before evidence was checked first.
        evidence = "\n".join(f"EVIDENCE_{key}: NO" for key in spec.output_keys)
        parsed = parse_phase1_output(f"{target}\n{evidence}", spec=spec, audit=True)
        self.assertEqual(parsed, dict(line.split(": ", 1) for line in target.splitlines()))

    def test_unknown_audit_evidence_remains_strictly_invalid(self) -> None:
        spec = make_prompt_spec(
            variant="subset_random",
            answers=_answers(),
            seed_key="strict",
            focus="RS5",
            subset_count=1,
        )
        target = build_phase1_target(_answers(), spec=spec)
        parsed = parse_phase1_output(f"{target}\nEVIDENCE_UNKNOWN: NO", spec=spec, audit=True)
        self.assertTrue(all(value is None for value in parsed.values()))

        semantic, diagnostics = parse_phase1_audit_output(
            f"{target}\nEVIDENCE_UNKNOWN: NO", spec=spec
        )
        self.assertEqual(semantic, dict(line.split(": ", 1) for line in target.splitlines()))
        self.assertTrue(diagnostics["answers_valid"])
        self.assertFalse(diagnostics["contract_valid"])

    def test_prompt_contains_only_evidence_backed_boundary_checks(self) -> None:
        prompt = build_phase1_prompt(
            spec=make_prompt_spec(
                variant="all_random_order",
                answers=_answers(),
                seed_key="prompt",
                focus="RS1",
            )
        )
        for phrase in (
            "two independent passes",
            "small or briefly visible vulnerable users",
            "continuous access-controlled multi-lane/barrier corridor is not RS1",
            "RS5 needs a local road opening/conflict",
            "TRAFFIC_LIGHT_ABNORMAL needs readable abnormal signal hardware",
        ):
            self.assertIn(phrase, prompt)

    def test_main_semantic_loss_is_focus_only(self) -> None:
        spec = make_prompt_spec(
            variant="all_random_order",
            answers=_answers(),
            seed_key="loss",
            focus="RS5",
        )
        item = WorkItem(_row(), "VULNERABLE", spec, "VULNERABLE:YES", "unit")
        semantic_keys = _semantic_output_keys(item)
        self.assertEqual(semantic_keys, ("VULNERABLE",))
        target = build_phase1_target(_answers(), spec=spec)
        _, weights, _ = _target_token_weights(
            _Bundle(),
            target,
            output_keys=spec.output_keys,
            semantic_output_keys=semantic_keys,
            semantic_output_weights=None,
            format_loss_weight=0.25,
        )
        for key in spec.output_keys:
            lo, hi = _line_value_span(target, key)
            expected = 1.0 if key == "VULNERABLE" else 0.0
            self.assertTrue(all(weights[i] == expected for i in range(lo, hi)), key)

    def test_hierarchical_derived_lines_keep_semantic_loss(self) -> None:
        spec = make_prompt_spec(
            variant="hierarchical_probe",
            answers=_answers(),
            seed_key="hier",
            focus="RS5",
            group_id="JUNCTION_CONTROL_ZONE",
            detail_key="RS5",
        )
        item = WorkItem(_row(), "RS5", spec, "RS5:YES", "unit")
        self.assertEqual(set(_semantic_output_keys(item)), {"RS5", "RS_HIGHWAY", "GROUP"})
        self.assertTrue(set(PHASE1_ANSWER_KEYS).isdisjoint(_semantic_output_keys(item)))

    def test_focus_class_weights_are_equal_when_focus_bins_are_equal(self) -> None:
        yes_spec = make_prompt_spec(
            variant="all_random_order",
            answers=_answers(),
            seed_key="yes",
            focus="RS5",
        )
        no_answers = {**_answers(), "VULNERABLE": False}
        no_row = _row()
        no_row.answers = no_answers
        no_spec = make_prompt_spec(
            variant="all_random_order",
            answers=no_answers,
            seed_key="no",
            focus="RS5",
        )
        work = [
            WorkItem(_row(), "VULNERABLE", yes_spec, "VULNERABLE:YES", "unit/yes"),
            WorkItem(no_row, "VULNERABLE", no_spec, "VULNERABLE:NO", "unit/no"),
        ]
        weights = _semantic_class_weights(work)
        self.assertEqual(weights, {"VULNERABLE:NO": 1.0, "VULNERABLE:YES": 1.0})


if __name__ == "__main__":
    unittest.main()
