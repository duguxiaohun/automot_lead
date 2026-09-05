"""New Loop Phase3 RGB history input contract.

数据索引固定保存同样四帧时序 RGB 路径；本模块只决定运行时实际喂给模型的子集，
确保同一个 adapter 的训练与评测不会用到不同的时间证据。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


HISTORY_RGB_MODE_ALL4 = "4rgb"
HISTORY_RGB_MODE_END2 = "2rgb_endpoints"
DEFAULT_HISTORY_RGB_MODE = HISTORY_RGB_MODE_ALL4
HISTORY_RGB_MODES = (HISTORY_RGB_MODE_ALL4, HISTORY_RGB_MODE_END2)


def validate_history_rgb_mode(mode: str) -> str:
    """校验并规范化运行时 RGB history 模式。"""

    normalized = str(mode).strip().lower()
    if normalized not in HISTORY_RGB_MODES:
        raise ValueError(
            f"unsupported history_rgb_mode={mode!r}; expected one of {list(HISTORY_RGB_MODES)}"
        )
    return normalized


def history_rgb_mode_tag(mode: str) -> str:
    """返回稳定的文件系统/结果标签。"""

    return validate_history_rgb_mode(mode)


def history_rgb_indices(mode: str) -> Tuple[int, ...]:
    """从固定四帧索引里选出运行时使用的位置。"""

    return (0, 1, 2, 3) if validate_history_rgb_mode(mode) == HISTORY_RGB_MODE_ALL4 else (0, 3)


def history_rgb_prompt_description(mode: str) -> str:
    """描述可见时间证据，避免 prompt 提到不存在的帧。"""

    if validate_history_rgb_mode(mode) == HISTORY_RGB_MODE_ALL4:
        return "four-frame history"
    return "two endpoint frames (the first and fourth frames from the four-frame history)"


def select_history_rgb_paths(paths: Sequence[str], mode: str) -> List[str]:
    """按模式选择运行时路径，同时保留索引里原始的四帧数据。"""

    values = [str(path) for path in paths]
    if len(values) != 4:
        raise ValueError(
            "sft_new_loop_phase3 expects exactly four chronological history_rgb_paths in the dataset index; "
            f"got {len(values)}"
        )
    return [values[index] for index in history_rgb_indices(mode)]
