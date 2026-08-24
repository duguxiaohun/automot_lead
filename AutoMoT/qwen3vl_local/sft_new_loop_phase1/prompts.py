"""Fused Phase1 + Phase2 prompt, targets, and strict YES/NO parser.

The public function names intentionally match ``sft_loop_phase1.prompts`` so
the proven Phase1 training/eval code can be reused with a wider answer set.
The prompt text itself reuses the finalized Phase1 visible-fact wording and the
newer Phase2 ROAD_STRUCTURE wording; where those instructions overlap, the
Phase2 road-structure definitions are treated as the authority for RS answers.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

from qwen3vl_local.sft_loop_phase1 import prompts as phase1_prompts
from qwen3vl_local.sft_loop_phase2_augment import prompts as phase2_prompts
from qwen3vl_local.sft_new_loop_phase1.history_rgb import (
    DEFAULT_HISTORY_RGB_MODE,
    HISTORY_RGB_MODE_ALL4,
    history_rgb_prompt_description,
    validate_history_rgb_mode,
)


PROMPT_NAME = "sft_new_loop_phase1_phase1_phase2_combined_v2_rgb_error_refined"
PHASE1_ANSWER_KEYS = (
    "HIGHWAY",
    "STATIC_OBSTACLE",
    "VULNERABLE",
    "TRAFFIC_LIGHT_ABNORMAL",
)
PHASE2_ANSWER_KEYS = ("RS1", "RS2", "RS4", "RS5")
ANSWER_KEYS = PHASE1_ANSWER_KEYS + PHASE2_ANSWER_KEYS
ANSWER_VALUES = ("YES", "NO")
ANSWER_PHASE = {key: "phase1" for key in PHASE1_ANSWER_KEYS} | {key: "phase2" for key in PHASE2_ANSWER_KEYS}
VARIANT_WEIGHTS = dict(phase2_prompts.VARIANT_WEIGHTS)
TRAIN_VARIANT_WEIGHTS = dict(phase2_prompts.TRAIN_VARIANT_WEIGHTS)
VARIANT_ORDER = tuple(phase2_prompts.VARIANT_ORDER)
SUBSET_COUNTS = tuple(phase2_prompts.SUBSET_COUNTS)
GROUP_DEFINITIONS = phase2_prompts.GROUP_DEFINITIONS

SYSTEM_PROMPT = """You are a perception step of an autonomous-driving agent.
The input is a stitched three-camera RGB history, ordered from oldest to newest. Classify only the newest moment. Inspect the complete left/front/right scene and use older frames only to confirm road geometry, motion, visibility, occlusion, or signal changes. Do not use scenario names, dataset labels, maps, hidden metadata, or memory. Answer each listed question independently from visible RGB evidence. Phase1 questions ask visible traffic facts; Phase2 questions ask the driving-rule road structure visible now."""


@dataclass(frozen=True)
class PromptSpec:
    """一次 fused forward 的 Phase2 增强 spec。"""

    phase2_spec: phase2_prompts.PromptSpec
    phase1_answers: Tuple[Tuple[str, bool], ...] = tuple()

    @property
    def variant(self) -> str:
        """返回 Phase2 增强类型。"""

        return self.phase2_spec.variant

    @property
    def output_keys(self) -> Tuple[str, ...]:
        """返回本次 assistant 必须输出的行顺序。"""

        return PHASE1_ANSWER_KEYS + tuple(_fused_question_output_key(q) for q in self.phase2_spec.questions)

    @property
    def phase1_answer_map(self) -> Dict[str, bool]:
        """返回构造 spec 时绑定的 Phase1 四问 GT。"""

        return {key: bool(value) for key, value in self.phase1_answers}


def _extract_decision_rules(text: str) -> str:
    """只抽 Phase1 prompt 的决策规则，避免嵌套 PROMPT_NAME/VISUAL_CHECK_ORDER。"""

    match = re.search(r"\[DECISION_RULES\]\s*(.*?)\s*\[/DECISION_RULES\]", text, flags=re.DOTALL)
    if not match:
        raise ValueError("cannot find Phase1 DECISION_RULES block")
    return match.group(1).strip()


def _fused_output_key(key: str) -> str:
    """避免 Phase2 hierarchical 的 HIGHWAY 输出行撞上 Phase1 HIGHWAY。"""

    return "RS_HIGHWAY" if key == "HIGHWAY" else str(key)


def _fused_question_output_key(q: phase2_prompts.QuestionSpec) -> str:
    """返回融合后的输出键；DETAIL 用真实 RS 键，便于 focus 指标归因。"""

    if q.output_key == "DETAIL" and q.question_id in PHASE2_ANSWER_KEYS:
        return q.question_id
    return _fused_output_key(q.output_key)


def make_prompt_spec(
    *,
    variant: str,
    answers: Mapping[str, bool],
    seed_key: str,
    focus: str = "RS1",
    subset_count: int = 1,
    group_id: str = "JUNCTION_CONTROL_ZONE",
    detail_key: str = "RS4",
) -> PromptSpec:
    """构造融合 prompt spec；Phase2 部分完全复用 phase2_augment 采样语义。"""

    phase2_answers = {key: bool(answers.get(key, False)) for key in PHASE2_ANSWER_KEYS}
    phase2_focus = focus if focus in PHASE2_ANSWER_KEYS else detail_key
    spec = phase2_prompts.make_prompt_spec(
        variant=variant,
        answers=phase2_answers,
        seed_key=seed_key,
        focus=phase2_focus,
        subset_count=subset_count,
        group_id=group_id,
        detail_key=detail_key,
    )
    phase1_answers = tuple((key, bool(answers.get(key, False))) for key in PHASE1_ANSWER_KEYS)
    return PromptSpec(phase2_spec=spec, phase1_answers=phase1_answers)


def prompt_spec_to_json(spec: PromptSpec) -> Dict[str, object]:
    """把融合 spec 写入 case/audit JSON。"""

    payload = phase2_prompts.prompt_spec_to_json(spec.phase2_spec)
    payload["fused_output_keys"] = list(spec.output_keys)
    payload["phase1_output_keys"] = list(PHASE1_ANSWER_KEYS)
    return payload


def _phase2_questions_and_rules(spec: PromptSpec) -> Tuple[str, str]:
    """按当前 Phase2 spec 渲染问题与规则块。"""

    question_lines = "\n".join(
        f"{idx}. {_fused_question_output_key(q)}: {q.question}"
        for idx, q in enumerate(spec.phase2_spec.questions, start=1)
    )
    definition_ids = []
    for q in spec.phase2_spec.questions:
        if q.question_id in phase2_prompts.RS_DEFINITIONS and q.question_id not in definition_ids:
            definition_ids.append(q.question_id)
        if q.metric_key.startswith("GROUP:"):
            group_id = q.metric_key.split(":", 1)[1]
            if group_id not in definition_ids:
                definition_ids.append(group_id)
    blocks = []
    for item in definition_ids:
        if item in phase2_prompts.RS_DEFINITIONS:
            blocks.append(phase2_prompts.RS_DEFINITIONS[item])
        elif item in phase2_prompts.GROUP_DEFINITIONS:
            blocks.append(phase2_prompts.GROUP_DEFINITIONS[item][2])
    return question_lines, "\n\n".join(blocks)


def build_phase1_prompt(
    *,
    spec: Optional[PromptSpec] = None,
    audit: bool = False,
    history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE,
) -> str:
    """Build the fused one-turn prompt.

    ``audit`` asks for short visible evidence after the requested answer lines. It
    is for prompt/debug review only; train and normal eval should keep it false.
    """

    mode = validate_history_rgb_mode(history_rgb_mode)
    history = history_rgb_prompt_description(mode)
    endpoint_notice = "" if mode == HISTORY_RGB_MODE_ALL4 else (
        " This input contains only the first and fourth frames from the original four-frame history; "
        "do not assume intermediate evidence exists."
    )
    if spec is None:
        spec = make_prompt_spec(
            variant="all_random_order",
            answers={key: False for key in ANSWER_KEYS},
            seed_key="default",
            focus="RS1",
        )
    phase1_rules = _extract_decision_rules(
        phase1_prompts.build_phase1_prompt(audit=False, history_rgb_mode=mode)
    )
    phase2_questions, phase2_rules = _phase2_questions_and_rules(spec)
    answer_lines = "\n".join(f"{key}: <YES or NO>" for key in spec.output_keys)
    if audit:
        evidence_lines = "\n".join(f"EVIDENCE_{key}: <one short RGB cue; max 12 words>" for key in spec.output_keys)
        output_lines = f"{answer_lines}\n{evidence_lines}"
        output_tag = "AUDIT_OUTPUT"
        audit_notice = " Write exactly the requested answer lines first, then one short visible evidence line for each requested answer."
    else:
        output_lines = answer_lines
        output_tag = "OUTPUT"
        audit_notice = ""

    return f"""
[PROMPT_NAME]
{PROMPT_NAME}
[/PROMPT_NAME]

[VISUAL_CHECK_ORDER]
Classify the newest frame using the {history}. Use two independent passes. First scan left/front/right for visible road topology and trace the ego vehicle's usable corridor without letting pedestrians, cyclists, event vehicles, or obstacles decide the road-structure answers. Second scan every available frame again for small or briefly visible vulnerable users, fixed lane obstructions, and readable traffic-signal hardware. One clear older-frame witness may confirm a still-relevant target, but darkness, fog, a scenario-like setting, a crosswalk alone, or an unreadable object is not a witness. Before output, recheck three error-prone boundaries: a continuous access-controlled multi-lane/barrier corridor is not RS1; RS5 needs a local road opening/conflict together with stop/yield/priority or gap-acceptance evidence rather than a bend or turning actor alone; and TRAFFIC_LIGHT_ABNORMAL needs readable abnormal signal hardware at the same junction. The answer is for the newest moment.{endpoint_notice}{audit_notice}
[/VISUAL_CHECK_ORDER]

[AUGMENT_VARIANT]
{spec.variant}
[/AUGMENT_VARIANT]

[PHASE1_VISIBLE_FACT_RULES]
{phase1_rules}
[/PHASE1_VISIBLE_FACT_RULES]

[PHASE2_ROAD_STRUCTURE_QUESTIONS]
{phase2_questions}
[/PHASE2_ROAD_STRUCTURE_QUESTIONS]

[PHASE2_ROAD_STRUCTURE_RULES]
{phase2_rules}

ROAD STRUCTURE PRIORITY:
Use the latest Phase2 ROAD_STRUCTURE contract for RS1/RS2/RS4/RS5. First decide whether a higher-priority visible structure governs the newest moment: limited-access highway/ramp topology, then an immediate opposing-lane sharing constraint, then a local signalized junction, then a local no-signal priority/gap-acceptance junction. RS1 is YES only after those local structures are ruled out. Do not let Phase1 HIGHWAY, STATIC_OBSTACLE, VULNERABLE, or TRAFFIC_LIGHT_ABNORMAL answers force any RS answer; decide every RS line from its own visible road-structure definition.

INDEPENDENT ANSWER CHECK:
The output lines are separate YES/NO questions. A YES in one line does not force a YES or NO in another line. If Phase1 HIGHWAY and Phase2 RS geometry appear to overlap, keep the Phase2 definitions authoritative for RS1/RS2/RS4/RS5/RS_HIGHWAY/GROUP/DETAIL, and keep the audited Phase1 definition authoritative for HIGHWAY.
[/PHASE2_ROAD_STRUCTURE_RULES]

[{output_tag}]
Output exactly these lines and nothing else:
{output_lines}
[/{output_tag}]
""".strip()


def phase1_prompt_sha256(*, audit: bool = False, history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE) -> str:
    """Return a fingerprint for the complete fused augment prompt surface."""

    dummy_answers = {key: False for key in ANSWER_KEYS}
    parts = []
    stable_contract = {
        "prompt_name": PROMPT_NAME,
        "system_prompt": SYSTEM_PROMPT,
        "phase1_answer_keys": list(PHASE1_ANSWER_KEYS),
        "phase2_answer_keys": list(PHASE2_ANSWER_KEYS),
        "variant_order": list(VARIANT_ORDER),
        "eval_variant_weights": dict(VARIANT_WEIGHTS),
        "train_variant_weights": dict(TRAIN_VARIANT_WEIGHTS),
        "subset_counts": list(SUBSET_COUNTS),
        "source_phase1_sha": phase1_prompts.phase1_prompt_sha256(audit=False, history_rgb_mode=history_rgb_mode),
        "source_phase2_sha": phase2_prompts.phase2_prompt_sha256(audit=False, history_rgb_mode=history_rgb_mode),
        "output_key_mapping": {
            "phase2_HIGHWAY": "RS_HIGHWAY",
            "phase2_DETAIL": "question_id_when_RS_detail",
        },
    }
    parts.append(json.dumps(stable_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    for order in itertools.permutations(PHASE2_ANSWER_KEYS):
        phase2_questions = tuple(phase2_prompts._rs_question(key, {key: False for key in PHASE2_ANSWER_KEYS}) for key in order)
        spec = PromptSpec(
            phase2_spec=phase2_prompts.PromptSpec("all_random_order", phase2_questions, "fingerprint"),
            phase1_answers=tuple((key, False) for key in PHASE1_ANSWER_KEYS),
        )
        parts.append(build_phase1_prompt(spec=spec, audit=audit, history_rgb_mode=history_rgb_mode))
    for count in SUBSET_COUNTS:
        for subset in itertools.permutations(PHASE2_ANSWER_KEYS, int(count)):
            phase2_questions = tuple(phase2_prompts._rs_question(key, {key: False for key in PHASE2_ANSWER_KEYS}) for key in subset)
            spec = PromptSpec(
                phase2_spec=phase2_prompts.PromptSpec("subset_random", phase2_questions, "fingerprint"),
                phase1_answers=tuple((key, False) for key in PHASE1_ANSWER_KEYS),
            )
            parts.append(build_phase1_prompt(spec=spec, audit=audit, history_rgb_mode=history_rgb_mode))
    for group_id in GROUP_DEFINITIONS:
        for detail_key in PHASE2_ANSWER_KEYS:
            spec = make_prompt_spec(
                variant="hierarchical_probe",
                answers=dummy_answers,
                seed_key="fingerprint",
                group_id=group_id,
                detail_key=detail_key,
            )
            parts.append(build_phase1_prompt(spec=spec, audit=audit, history_rgb_mode=history_rgb_mode))
    payload = {
        "surface_parts": parts,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_ANSWER_RE = re.compile(r"(?im)^\s*([A-Z0-9_]+)\s*:\s*(YES|NO)\s*$")
_ANSWER_LINE_RE = re.compile(r"^\s*([A-Z0-9_]+)\s*:\s*(YES|NO)\s*$", re.IGNORECASE)
_EVIDENCE_LINE_RE = re.compile(r"^\s*EVIDENCE_([A-Z0-9_]+)\s*:\s*\S.*$", re.IGNORECASE)


def parse_phase1_output(text: str, *, spec: Optional[PromptSpec] = None, audit: bool = False) -> Dict[str, Optional[str]]:
    """Parse expected YES/NO lines; duplicate/missing/extra lines are invalid."""

    if audit:
        answers, diagnostics = parse_phase1_audit_output(text, spec=spec)
        if diagnostics["contract_valid"]:
            return answers
        return {key: None for key in answers}

    keys = tuple(spec.output_keys) if spec is not None else ANSWER_KEYS
    found: Dict[str, Optional[str]] = {key: None for key in keys}
    duplicate = set()
    extra = False
    evidence_seen = set()
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Audit evidence lines can end in YES/NO (for example
        # ``EVIDENCE_HIGHWAY: NO readable ramp cue``).  Match the more
        # specific namespace first so it cannot be mistaken for an unknown
        # answer key and invalidate an otherwise correct audit response.
        evidence_match = _EVIDENCE_LINE_RE.match(line)
        if audit and evidence_match:
            key = evidence_match.group(1).upper()
            if key in found and key not in evidence_seen:
                evidence_seen.add(key)
                continue
            extra = True
            continue
        answer_match = _ANSWER_LINE_RE.match(line)
        if answer_match:
            key, value = answer_match.group(1).upper(), answer_match.group(2).upper()
            if key not in found:
                extra = True
                continue
            if found[key] is not None:
                duplicate.add(key)
            found[key] = value
            continue
        extra = True
    for key in duplicate:
        found[key] = None
    if extra:
        for key in found:
            found[key] = None
    for key in duplicate:
        found[key] = None
    return found


def parse_phase1_audit_output(
    text: str,
    *,
    spec: Optional[PromptSpec] = None,
) -> Tuple[Dict[str, Optional[str]], Dict[str, object]]:
    """分别解析 audit 的答案语义与 evidence/整体格式合同。

    Evidence 缺失、重复或额外结束标记会使 ``contract_valid=False``，但不会
    擦除已经合法解析的答案。eval 因而能分别报告问答能力和审计格式质量。
    """

    keys = tuple(spec.output_keys) if spec is not None else ANSWER_KEYS
    found: Dict[str, Optional[str]] = {key: None for key in keys}
    answer_duplicates = set()
    evidence_counts: Dict[str, int] = {key: 0 for key in keys}
    unexpected_lines = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        evidence_match = _EVIDENCE_LINE_RE.match(line)
        if evidence_match:
            key = evidence_match.group(1).upper()
            if key in evidence_counts:
                evidence_counts[key] += 1
            else:
                unexpected_lines.append(line)
            continue
        answer_match = _ANSWER_LINE_RE.match(line)
        if answer_match:
            key, value = answer_match.group(1).upper(), answer_match.group(2).upper()
            if key not in found:
                unexpected_lines.append(line)
                continue
            if found[key] is not None:
                answer_duplicates.add(key)
            found[key] = value
            continue
        unexpected_lines.append(line)
    for key in answer_duplicates:
        found[key] = None
    answers_valid = all(found[key] in ANSWER_VALUES for key in keys)
    evidence_complete = all(evidence_counts[key] == 1 for key in keys)
    diagnostics: Dict[str, object] = {
        "answers_valid": bool(answers_valid),
        "evidence_complete": bool(evidence_complete),
        "contract_valid": bool(answers_valid and evidence_complete and not unexpected_lines),
        "missing_evidence_keys": [key for key in keys if evidence_counts[key] == 0],
        "duplicate_evidence_keys": [key for key in keys if evidence_counts[key] > 1],
        "duplicate_answer_keys": sorted(answer_duplicates),
        "unexpected_lines": unexpected_lines,
    }
    return found, diagnostics


def _phase2_answer_map(spec: PromptSpec) -> Dict[str, bool]:
    """返回当前 Phase2 spec 的输出行答案。"""

    return {_fused_question_output_key(q): bool(q.answer) for q in spec.phase2_spec.questions}


def phase2_output_keys(spec: PromptSpec) -> Tuple[str, ...]:
    """返回当前 fused prompt 中真正属于 Phase2 的输出行。"""

    return tuple(_fused_question_output_key(q) for q in spec.phase2_spec.questions)


def build_phase1_target(answers: Dict[str, bool], *, spec: Optional[PromptSpec] = None) -> str:
    """Render the fused supervised target for this variant."""

    if spec is None:
        spec = make_prompt_spec(
            variant="all_random_order",
            answers=answers,
            seed_key="target_default",
            focus="RS1",
        )
    phase2 = _phase2_answer_map(spec)
    target_values = {key: bool(answers[key]) for key in PHASE1_ANSWER_KEYS}
    target_values.update(phase2)
    missing = [key for key in spec.output_keys if key not in target_values]
    if missing:
        raise ValueError(f"fused target missing answers: {missing}")
    return "\n".join(f"{key}: {'YES' if bool(target_values[key]) else 'NO'}" for key in spec.output_keys)


def spec_metric_items(spec: PromptSpec) -> Iterable[Tuple[str, str, bool]]:
    """Yield (output_key, metric_key, gt_bool) for metrics."""

    phase1_answers = spec.phase1_answer_map
    for key in PHASE1_ANSWER_KEYS:
        yield key, key, bool(phase1_answers.get(key, False))
    for q in spec.phase2_spec.questions:
        metric_key = "RS_HIGHWAY" if q.metric_key == "HIGHWAY" else q.metric_key
        yield _fused_question_output_key(q), metric_key, bool(q.answer)


def focus_phase(key: str) -> str:
    """Return ``phase1`` or ``phase2`` for a focus answer key."""

    if key not in ANSWER_PHASE:
        raise KeyError(key)
    return ANSWER_PHASE[key]
