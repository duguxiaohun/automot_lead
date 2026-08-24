"""New Loop Phase2 RGB history input contract.

The dataset index always stores the same four chronological RGB paths. This
module chooses the runtime view subset so train and eval cannot accidentally
use different temporal evidence for the same adapter.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


HISTORY_RGB_MODE_ALL4 = "4rgb"
HISTORY_RGB_MODE_END2 = "2rgb_endpoints"
DEFAULT_HISTORY_RGB_MODE = HISTORY_RGB_MODE_ALL4
HISTORY_RGB_MODES = (HISTORY_RGB_MODE_ALL4, HISTORY_RGB_MODE_END2)


def validate_history_rgb_mode(mode: str) -> str:
    """Validate and normalize the persisted runtime RGB-history mode."""

    normalized = str(mode).strip().lower()
    if normalized not in HISTORY_RGB_MODES:
        raise ValueError(
            f"unsupported history_rgb_mode={mode!r}; expected one of {list(HISTORY_RGB_MODES)}"
        )
    return normalized


def history_rgb_mode_tag(mode: str) -> str:
    """Return the stable filesystem/result tag for one image-history contract."""

    return validate_history_rgb_mode(mode)


def history_rgb_indices(mode: str) -> Tuple[int, ...]:
    """Return positions from the fixed four-frame chronological index."""

    return (0, 1, 2, 3) if validate_history_rgb_mode(mode) == HISTORY_RGB_MODE_ALL4 else (0, 3)


def history_rgb_prompt_description(mode: str) -> str:
    """Describe the visible temporal evidence without exposing unavailable frames."""

    if validate_history_rgb_mode(mode) == HISTORY_RGB_MODE_ALL4:
        return "four-frame history"
    return "two endpoint frames (the first and fourth frames from the four-frame history)"


def select_history_rgb_paths(paths: Sequence[str], mode: str) -> List[str]:
    """Select runtime paths while preserving the original four-frame data index."""

    values = [str(path) for path in paths]
    if len(values) != 4:
        raise ValueError(
            "sft_new_loop_phase2 expects exactly four chronological history_rgb_paths in the dataset index; "
            f"got {len(values)}"
        )
    return [values[index] for index in history_rgb_indices(mode)]
