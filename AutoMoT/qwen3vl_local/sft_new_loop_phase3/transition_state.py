"""在线恢复候选接口；调用者提供可见/导航确认，不能用未来真值或 all-NO 开 gate。"""
from dataclasses import dataclass


@dataclass
class RecoveryState:
    """保存已经发生的绕障历史，等待和动作全 NO 都不能清除它。"""
    bypass_observed: bool = False
    lane_departure_confirmed: bool = False
    recovery_pending: bool = False

    def observe(self, *, blockage: bool = False, departed_lane: bool = False,
                restored_lane: bool = False, route_reset: bool = False) -> None:
        if route_reset:
            self.bypass_observed = self.lane_departure_confirmed = self.recovery_pending = False
            return
        if blockage:
            self.bypass_observed = True
        if departed_lane and self.bypass_observed:
            self.lane_departure_confirmed = True
            self.recovery_pending = True
        if restored_lane and self.lane_departure_confirmed:
            self.bypass_observed = self.lane_departure_confirmed = self.recovery_pending = False

    def candidate(self, *, blockage_active: bool = False) -> str | None:
        return "POST_BYPASS_RETURN" if self.recovery_pending and not blockage_active else None
