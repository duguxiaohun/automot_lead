#!/usr/bin/env python3
"""Phase3 导航目标渲染：把 ``next_target_points[-1]`` 的 ego 相对坐标写成模型可读文本。

坐标口径与 `sft_base`/`sft_v3`/`sft_v4` 的 ``EGO_TO_GOAL_XY`` 完全同源。
2026-09-04 用 `probe_ego_frame_sign.py` 在左转/右转 scenario 上取证确认：
LEAD ego frame 是 CARLA 左手系，``x`` 正为正前方，``y`` **负为左、正为右**。
Phase3 必须把这条约定显式写进 prompt，模型才可能自己判断“目标点在左前方多少米”，
进而在借道绕障后自然产生回目标车道的动作。
"""

from __future__ import annotations

from typing import Optional, Tuple


LATERAL_NEAR_CENTER_M = 1.5


def goal_side(goal_y: float) -> str:
    """返回目标点相对自车的左右侧描述。"""

    if goal_y <= -LATERAL_NEAR_CENTER_M:
        return "left"
    if goal_y >= LATERAL_NEAR_CENTER_M:
        return "right"
    return "straight ahead"


def goal_sentence(goal_x: float, goal_y: float) -> str:
    """把相对坐标翻译成一句自然语言，避免模型只看到裸数字。"""

    ahead = "ahead of" if goal_x >= 0.0 else "behind"
    side = goal_side(goal_y)
    if side == "straight ahead":
        return f"The route target is about {abs(goal_x):.1f} m {ahead} ego and almost straight ahead laterally."
    return (
        f"The route target is about {abs(goal_x):.1f} m {ahead} ego and about "
        f"{abs(goal_y):.1f} m to ego's {side}."
    )


def render_navigation_goal(goal: Optional[Tuple[float, float]]) -> str:
    """渲染 prompt 中的 ``[NAVIGATION_GOAL]`` 块。"""

    if goal is None:
        return (
            "[NAVIGATION_GOAL]\n"
            "ROUTE_TARGET_XY: UNKNOWN\n"
            "No route target offset is available for this moment; decide from the RGB history alone.\n"
            "[/NAVIGATION_GOAL]"
        )
    goal_x, goal_y = float(goal[0]), float(goal[1])
    return (
        "[NAVIGATION_GOAL]\n"
        f"ROUTE_TARGET_XY: (x={goal_x:+.1f} m, y={goal_y:+.1f} m)\n"
        "This offset is expressed in ego coordinates at the newest frame. x is the signed distance "
        "straight ahead of ego, positive in front and negative behind. y is the signed lateral "
        "distance, negative to ego's LEFT and positive to ego's RIGHT.\n"
        f"{goal_sentence(goal_x, goal_y)}\n"
        "Use this only as the navigation target of the route. It is not a label, it does not say "
        "which lane is currently drivable, and it never replaces the visible RGB evidence.\n"
        "[/NAVIGATION_GOAL]"
    )
