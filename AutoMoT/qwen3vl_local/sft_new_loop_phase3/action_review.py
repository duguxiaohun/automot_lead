"""离线动作审查证据：控制响应、记录的约束对象和时间节点，不参与动作标定或提示词。"""
import math
from numbers import Integral, Real

from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID, DOMAIN_MANEUVER
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import primary_choice
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import (
    FRAME_DT_SECONDS, LATERAL_HORIZON_FRAMES, longitudinal_from_signals,
)

ACTION_REVIEW_VERSION = "recorded_response_and_motion_v2_pullaway"
BOOL_FIELDS = ("brake", "vehicle_hazard", "walker_hazard", "light_hazard",
               "stop_sign_hazard", "brake_cutin", "slower_bad_visibility", "slower_clutterness")


def _number(value):
    """缺失、非有限值和布尔量不能伪装成数值零。"""
    return float(value) if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) else None


def _actor_id(value):
    """对象 ID 只接受非负整数；未知不猜对象。"""
    return int(value) if isinstance(value, Integral) and not isinstance(value, bool) and value >= 0 else None


def controller_snapshot(meta):
    """提取当前记录；未知布尔值保留 null，对象关系仅表示名单成员身份。"""
    result = {key: meta.get(key) if type(meta.get(key)) is bool else None for key in BOOL_FIELDS}
    for key in ("target_speed", "target_speed_limit", "speed_reduced_by_obj_distance"):
        value = _number(meta.get(key))
        result[key] = value if value is not None and value >= 0 else None
    actor = _actor_id(meta.get("speed_reduced_by_obj_id"))
    result["speed_reduced_by_obj_id"] = actor
    actor_type = meta.get("speed_reduced_by_obj_type")
    result["speed_reduced_by_obj_type"] = actor_type if isinstance(actor_type, str) and actor_type else None
    # 不在名单中并不证明是背景车辆；背景身份需要 bbox/actor 的独立核验。
    memberships = {}
    for field in ("scenario_obstacles_ids", "scenario_actors_ids", "cut_in_actors_ids"):
        values = meta.get(field)
        memberships[field] = (actor in values if actor is not None and isinstance(values, (list, tuple)) else None)
    result["object_membership"] = memberships
    result["unknown_fields"] = [key for key in (*BOOL_FIELDS, "target_speed") if result[key] is None]
    return result


def _known_limiter(meta):
    """区分明确无对象与字段缺失，避免把缺失误写成约束释放。"""
    if "speed_reduced_by_obj_id" not in meta:
        return False, None
    value = meta["speed_reduced_by_obj_id"]
    actor = _actor_id(value)
    return value is None or actor is not None, actor


def build_action_review(trajectory, frame_id, context_id, labels, signals=None):
    """共用构建和审计；时间节点是原判据触发点，不是新增阶段序列真值。"""
    context = CONTEXT_BY_ID[context_id]
    signals = trajectory.signals(frame_id) if signals is None else signals
    decision = longitudinal_from_signals(signals)
    if not decision["eligible"]:
        raise ValueError("action review requires eligible motion evidence")
    selected = primary_choice(labels, context.action_keys)
    anchor = trajectory.metas[frame_id]
    controller = controller_snapshot(anchor)
    flags = []
    if decision["confirmed_pullaway"]:
        flags.append("confirmed_pullaway_at_near_stop")
    if decision["confirmed_pullaway"] and frame_id < 3:
        flags.append("pullaway_with_padded_startup_history")
    if decision["current_near_stop_pair"] and decision["anchor_control_released"] and not decision["confirmed_pullaway"]:
        flags.append("released_near_stop_without_confirmed_pullaway")
    if selected == "KEEP":
        if controller["brake"] is True:
            flags.append("keep_with_brake_command")
        if controller["brake"] is True and controller["target_speed"] == 0 and controller["vehicle_hazard"] is True:
            flags.append("keep_with_vehicle_stop_request")
    # 只收集原速度判据中实际存在的节点，不用控制命令生成 STOP/DECELERATE。
    milestones = []
    for action, start, confirmed in (
        ("DECELERATE", decision["first_drop_s"], decision["first_drop_s"]),
        ("RESUME", decision["gain_start_s"], decision["gain_confirmed_s"]),
        ("STOP", decision["stop_start_s"] if decision["stop_qualifies"] else None,
         decision["stop_confirmed_s"]),
    ):
        if start is not None:
            milestones.append(dict(action=action, start_s=start, confirmed_s=confirmed,
                                   selected_speed_stage=action == decision["action"]))
    speed_start = {"DECELERATE": decision["first_drop_s"], "RESUME": decision["gain_start_s"],
                   "STOP": decision["stop_start_s"], "NONE": None}[decision["action"]]
    crossing_start = None
    if context.question_domain == DOMAIN_MANEUVER:
        if not signals["lateral_observation_complete"]:
            raise ValueError("maneuver review requires complete lateral evidence")
        if signals["lane_change_direction"]:
            offset = next(i for i in range(1, LATERAL_HORIZON_FRAMES+1)
                          if trajectory.lane_change(frame_id, horizon=i))
            crossing_start = offset * FRAME_DT_SECONDS
            milestones.append(dict(action="LANE_CHANGE_"+signals["lane_change_direction"],
                                   start_s=crossing_start, confirmed_s=crossing_start+FRAME_DT_SECONDS))
            if speed_start is not None and speed_start < crossing_start:
                flags.append("speed_before_crossing")
    # 约束对象记录改变和目标速度解除只记录事实，不推断道路已安全或障碍已消失。
    limiter_change = target_release = None
    limiter_known, limiter = _known_limiter(anchor)
    target_search_known = controller["target_speed"] is not None
    controller_window_complete = True
    for offset in range(1, LATERAL_HORIZON_FRAMES+1):
        meta = trajectory.metas.get(frame_id+offset)
        if meta is None:
            controller_window_complete = False
            break
        known, actor = _known_limiter(meta)
        # 显式 None 是已知“无对象”，同样需要记录约束首次出现；未知仍中断证据链。
        if limiter_known and limiter_change is None:
            if not known:
                limiter_known = False
            elif actor != limiter:
                limiter_change = dict(at_s=offset*FRAME_DT_SECONDS, from_id=limiter, to_id=actor)
        value = _number(meta.get("target_speed"))
        if value is None or value < 0:
            target_search_known = False
        following = trajectory.metas.get(frame_id+offset+1) if offset < LATERAL_HORIZON_FRAMES else None
        if target_search_known and controller["target_speed"] == 0 and target_release is None and following is not None:
            values = [_number(m.get("target_speed")) for m in (meta, following)]
            if all(v is not None and v > 0 for v in values):
                target_release = dict(start_s=offset*FRAME_DT_SECONDS, confirmed_s=(offset+1)*FRAME_DT_SECONDS)
    return dict(version=ACTION_REVIEW_VERSION, source="recorded_meta", source_context_id=context_id,
                review_only=True, purpose_status="conditional_not_verified_intent",
                controller=controller, anchor_controls=decision["anchor_controls"],
                longitudinal_reason=decision["reason"], flags=flags, primary_action=selected,
                motion_milestones=sorted(milestones, key=lambda m:(m["start_s"],m["action"])),
                milestone_semantics="bounded_rule_triggers_not_complete_stage_sequence",
                speed_action_start_s=speed_start, crossing_start_s=crossing_start,
                controller_window_complete=controller_window_complete,
                target_search_known=target_search_known,
                logged_limiter_change=limiter_change, positive_target_after_zero=target_release)
