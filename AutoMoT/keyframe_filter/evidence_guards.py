"""采集证据门控：控制器 hazard/普通 trigger 不是道路几何真值。"""

MERGE_SCENARIOS = frozenset({"EnterActorFlow", "EnterActorFlowV2",
    "MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2", "HighwayExit"})


def has_signal_support(valid_light_state, ego_light_bbox, trusted_near_signal):
    """light_hazard 只能与独立的信号证据共同支持 R4，is_junction 不足以证明有灯。"""
    return bool(valid_light_state or ego_light_bbox or trusted_near_signal)


def merge_trigger_supported(scenario, trusted_ramp_hint):
    """只有合流任务的 trigger 或可信匝道拓扑可以激活合流候选。"""
    return scenario in MERGE_SCENARIOS or bool(trusted_ramp_hint)
