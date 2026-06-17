"""Prompt helpers for SFT v2 serial scene/status/subgoal supervision.

This version removes ANALYSIS entirely. The model first predicts SCENE from
images and scene choices. A second follow-up prompt then asks for STATUS and
SUBGOAL using the predicted scene's event sequence while reusing the scene-step
context/KV.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from qwen3vl_local.prompt_pipeline import (
    EVENT_DESCRIPTIONS,
    SCENARIO_EVENT_SEQUENCES,
    SCENARIO_LABELS,
    get_full_sequence,
)


DATASET_VERSION = "sft_v2_serial_choice"


SCENE_SYSTEM_PROMPT = """\
You are an autonomous driving serial scene/status classifier.

Input:
- You receive a short RGB clip ordered oldest to newest.
- Each frame is a stitched three-camera view: left, front, and right.
- The newest frame is the current moment.

Rules:
- This is a choice task, not free-form generation.
- Copy scenario names and event names verbatim from the choices provided in
  the current user message.
- Do not invent scenario names or event names.
- Do not output ANALYSIS or explanations.
- If the visual evidence is ambiguous, prefer the previous status hint when it
  is provided and do not advance STATUS without a clear visual transition."""


STATUS_SYSTEM_PROMPT = """\
You are an autonomous driving status and subgoal classifier.

Input:
- You receive a short RGB clip ordered oldest to newest.
- Each frame is a stitched three-camera view: left, front, and right.
- The newest frame is the current moment.
- You also receive exactly one selected SCENE and that scene's EVENT_SEQUENCE.

Task:
1. Use only the selected SCENE's EVENT_SEQUENCE.
2. Choose STATUS by copying exactly one event token from that sequence.
3. Choose SUBGOAL as the immediate next event after STATUS in that same
   EVENT_SEQUENCE. If STATUS is final, SUBGOAL is final.

Output exactly two lines and nothing else:
STATUS: <event_name>
SUBGOAL: <event_name>

Rules:
- This is a choice task, not free-form generation.
- STATUS and SUBGOAL must be copied verbatim from the selected scene's
  EVENT_SEQUENCE.
- Do not invent event names.
- Do not output SCENE, ANALYSIS, or explanations.
- If the visual evidence is ambiguous, prefer the previous status hint and do
  not advance STATUS without a clear visual transition."""


# Backward-compatible alias for older callers.
SYSTEM_PROMPT = SCENE_SYSTEM_PROMPT


def scenario_choices_block() -> str:
    """Return all scenario names with descriptions for the prompt."""

    lines = ["[SCENE_CHOICES]"]
    for name in sorted(SCENARIO_LABELS):
        lines.append(f"- {name}: {SCENARIO_LABELS[name]}")
    lines.append("[/SCENE_CHOICES]")
    return "\n".join(lines)


def event_choices_block() -> str:
    """Return every scenario's allowed sequence and event descriptions."""

    lines = ["[SCENE_EVENT_CHOICES]"]
    for scenario in sorted(SCENARIO_EVENT_SEQUENCES):
        seq = get_full_sequence(scenario)
        lines.append(f"- {scenario}: {' -> '.join(seq)}")
        for event in seq:
            lines.append(f"  * {event}: {EVENT_DESCRIPTIONS.get(event, event)}")
    lines.append("[/SCENE_EVENT_CHOICES]")
    return "\n".join(lines)


def build_scene_user_prompt(
    *,
    image_count: int,
) -> str:
    """Build the first-stage scene-choice user prompt."""

    return (
        f"The {image_count} images above are ordered oldest to newest; "
        "the last image is the current moment.\n\n"
        f"{scenario_choices_block()}\n\n"
        "Choose SCENE from SCENE_CHOICES. Output exactly one line and nothing "
        "else:\nSCENE: <scenario_name>"
    )


def selected_event_block(scene: str) -> str:
    """Return the selected scene's event sequence and descriptions."""

    seq = get_full_sequence(scene)
    lines = [
        "[SELECTED_SCENE]",
        f"SCENE: {scene}",
        f"DESCRIPTION: {SCENARIO_LABELS.get(scene, scene)}",
        "[/SELECTED_SCENE]",
        "",
        "[EVENT_SEQUENCE]",
        " -> ".join(seq),
    ]
    for event in seq:
        lines.append(f"- {event}: {EVENT_DESCRIPTIONS.get(event, event)}")
    lines.append("[/EVENT_SEQUENCE]")
    return "\n".join(lines)


def build_status_user_prompt(
    *,
    image_count: int,
    selected_scene: str,
    previous_status: str,
    previous_subgoal: str,
) -> str:
    """Build the second-stage status/subgoal prompt for one selected scene."""

    return (
        f"The {image_count} images above are ordered oldest to newest; "
        "the last image is the current moment.\n\n"
        f"{selected_event_block(selected_scene)}\n\n"
        "[PREVIOUS_STATUS_HINT]\n"
        f"STATUS: {previous_status}\n"
        f"SUBGOAL: {previous_subgoal}\n"
        "[/PREVIOUS_STATUS_HINT]\n\n"
        "Task:\n"
        "1. Use only the selected scene's EVENT_SEQUENCE above.\n"
        "2. Choose STATUS by copying exactly one event token from that sequence.\n"
        "3. Choose SUBGOAL as the immediate next event after STATUS in that same sequence. "
        "If STATUS is final, SUBGOAL is final.\n\n"
        "Output exactly two lines and nothing else:\n"
        "STATUS: <event_name>\n"
        "SUBGOAL: <event_name>\n\n"
        "Do not output SCENE, ANALYSIS, or explanations."
    )


def build_user_prompt(
    *,
    image_count: int,
    previous_status: str,
    previous_subgoal: str,
) -> str:
    """Build the legacy single-prompt text for older one-stage callers."""

    return (
        f"{build_scene_user_prompt(image_count=image_count)}\n\n"
        f"{event_choices_block()}\n\n"
        "[PREVIOUS_STATUS_HINT]\n"
        f"STATUS: {previous_status}\n"
        f"SUBGOAL: {previous_subgoal}\n"
        "[/PREVIOUS_STATUS_HINT]\n\n"
        "Output SCENE, STATUS, and SUBGOAL now."
    )


def next_event(scenario: str, status: str) -> str:
    """Return the immediate next event for scenario/status; final is self."""

    seq = get_full_sequence(scenario)
    try:
        idx = seq.index(status)
    except ValueError:
        return "final"
    return seq[idx + 1] if idx + 1 < len(seq) else "final"


def format_assistant(scene: str, status: str, subgoal: Optional[str] = None) -> str:
    """Format the supervised assistant target."""

    if subgoal is None:
        subgoal = next_event(scene, status)
    return f"SCENE: {scene}\nSTATUS: {status}\nSUBGOAL: {subgoal}"


def format_scene_assistant(scene: str) -> str:
    """Format the supervised first-stage target."""

    return f"SCENE: {scene}"


def format_status_assistant(status: str, subgoal: str) -> str:
    """Format the supervised second-stage target."""

    return f"STATUS: {status}\nSUBGOAL: {subgoal}"


_SCENE_RE = re.compile(r"^\s*SCENE\s*:\s*(\S+)", re.MULTILINE | re.IGNORECASE)
_STATUS_RE = re.compile(r"^\s*STATUS\s*:\s*(\S+)", re.MULTILINE | re.IGNORECASE)
_SUBGOAL_RE = re.compile(r"^\s*SUBGOAL\s*:\s*(\S+)", re.MULTILINE | re.IGNORECASE)


def parse_output(text: str) -> Dict[str, Optional[str]]:
    """Parse model output into scene/status/subgoal fields."""

    scene_m = _SCENE_RE.search(text or "")
    status_m = _STATUS_RE.search(text or "")
    subgoal_m = _SUBGOAL_RE.search(text or "")
    return {
        "scene": scene_m.group(1).strip() if scene_m else None,
        "status": status_m.group(1).strip() if status_m else None,
        "subgoal": subgoal_m.group(1).strip() if subgoal_m else None,
    }


def validate_choice(scene: Optional[str], status: Optional[str], subgoal: Optional[str]) -> Dict[str, bool]:
    """Check whether parsed choices obey the serial-choice constraints."""

    scene_ok = bool(scene in SCENARIO_EVENT_SEQUENCES)
    if not scene_ok:
        return {
            "scene_valid": False,
            "status_valid_for_scene": False,
            "subgoal_valid_for_scene": False,
            "subgoal_matches_status": False,
        }
    seq = get_full_sequence(str(scene))
    status_ok = bool(status in seq)
    subgoal_ok = bool(subgoal in seq)
    expected = next_event(str(scene), str(status)) if status_ok else None
    return {
        "scene_valid": True,
        "status_valid_for_scene": status_ok,
        "subgoal_valid_for_scene": subgoal_ok,
        "subgoal_matches_status": bool(subgoal_ok and subgoal == expected),
    }


def extract_gt(assistant_text: str) -> Dict[str, str]:
    """Extract supervised labels from an assistant target."""

    parsed = parse_output(assistant_text)
    return {
        "scene": parsed.get("scene") or "",
        "status": parsed.get("status") or "",
        "subgoal": parsed.get("subgoal") or "",
    }


def target_spans(assistant_text: str) -> Dict[str, Tuple[int, int]]:
    """Return char spans for SCENE/STATUS/SUBGOAL values."""

    spans: Dict[str, Tuple[int, int]] = {}
    for key, regex in (("scene", _SCENE_RE), ("status", _STATUS_RE), ("subgoal", _SUBGOAL_RE)):
        m = regex.search(assistant_text)
        if m:
            spans[key] = (m.start(1), m.end(1))
    return spans
